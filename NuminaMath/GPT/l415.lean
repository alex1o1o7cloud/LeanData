import Mathlib

namespace exist_non_negative_product_l415_41576

theorem exist_non_negative_product (a1 a2 a3 a4 a5 a6 a7 a8 : ℝ) :
  0 ≤ a1 * a3 + a2 * a4 ∨
  0 ≤ a1 * a5 + a2 * a6 ∨
  0 ≤ a1 * a7 + a2 * a8 ∨
  0 ≤ a3 * a5 + a4 * a6 ∨
  0 ≤ a3 * a7 + a4 * a8 ∨
  0 ≤ a5 * a7 + a6 * a8 :=
sorry

end exist_non_negative_product_l415_41576


namespace percentage_increase_correct_l415_41589

variable {R1 E1 P1 R2 E2 P2 R3 E3 P3 : ℝ}

-- Conditions
axiom H1 : P1 = R1 - E1
axiom H2 : R2 = 1.20 * R1
axiom H3 : E2 = 1.10 * E1
axiom H4 : P2 = R2 - E2
axiom H5 : P2 = 1.15 * P1
axiom H6 : R3 = 1.25 * R2
axiom H7 : E3 = 1.20 * E2
axiom H8 : P3 = R3 - E3
axiom H9 : P3 = 1.35 * P2

theorem percentage_increase_correct :
  ((P3 - P1) / P1) * 100 = 55.25 :=
by sorry

end percentage_increase_correct_l415_41589


namespace number_of_special_three_digit_numbers_l415_41560

noncomputable def count_special_three_digit_numbers : ℕ :=
  Nat.choose 9 3

theorem number_of_special_three_digit_numbers : count_special_three_digit_numbers = 84 := by
  sorry

end number_of_special_three_digit_numbers_l415_41560


namespace smallest_product_not_factor_60_l415_41524

theorem smallest_product_not_factor_60 : ∃ (a b : ℕ), a ≠ b ∧ a ∣ 60 ∧ b ∣ 60 ∧ ¬ (a * b) ∣ 60 ∧ a * b = 8 := sorry

end smallest_product_not_factor_60_l415_41524


namespace find_m_l415_41506

open Real

/-- Define Circle C1 and C2 as having the given equations
and verify their internal tangency to find the possible m values -/
theorem find_m (m : ℝ) :
  (∃ (x y : ℝ), (x - m)^2 + (y + 2)^2 = 9) ∧ 
  (∃ (x y : ℝ), (x + 1)^2 + (y - m)^2 = 4) ∧ 
  (by exact (sqrt ((m + 1)^2 + (-2 - m)^2)) = 3 - 2) → 
  m = -2 ∨ m = -1 := 
sorry -- Proof is omitted

end find_m_l415_41506


namespace value_of_stamp_collection_l415_41541

theorem value_of_stamp_collection 
  (n m : ℕ) (v_m : ℝ)
  (hn : n = 18) 
  (hm : m = 6)
  (hv_m : v_m = 15)
  (uniform_value : ∀ (k : ℕ), k ≤ m → v_m / m = v_m / k):
  ∃ v_total : ℝ, v_total = 45 :=
by 
  sorry

end value_of_stamp_collection_l415_41541


namespace smallest_number_of_slices_l415_41565

def cheddar_slices : ℕ := 12
def swiss_slices : ℕ := 28
def gouda_slices : ℕ := 18

theorem smallest_number_of_slices : Nat.lcm (Nat.lcm cheddar_slices swiss_slices) gouda_slices = 252 :=
by 
  sorry

end smallest_number_of_slices_l415_41565


namespace simplify_expression_l415_41552

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b ^ 2 + 2 * b) - 4 * b ^ 2 = 9 * b ^ 3 + 2 * b ^ 2 :=
by
  sorry

end simplify_expression_l415_41552


namespace complement_intersection_l415_41549

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection (hU : U = {2, 3, 6, 8}) (hA : A = {2, 3}) (hB : B = {2, 6, 8}) :
  ((U \ A) ∩ B) = {6, 8} := 
by
  sorry

end complement_intersection_l415_41549


namespace smallest_d_l415_41539

-- Constants and conditions
variables (c d : ℝ)
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions involving c and d
def conditions (c d : ℝ) : Prop :=
  2 < c ∧ c < d ∧ ¬triangle_inequality 2 c d ∧ ¬triangle_inequality (1/d) (1/c) 2

-- Goal statement: the smallest possible value of d
theorem smallest_d (c d : ℝ) (h : conditions c d) : d = 2 + Real.sqrt 2 :=
sorry

end smallest_d_l415_41539


namespace gillian_more_than_three_times_sandi_l415_41562

-- Definitions of the conditions
def sandi_initial : ℕ := 600
def sandi_spent : ℕ := sandi_initial / 2
def gillian_spent : ℕ := 1050
def three_times_sandi_spent : ℕ := 3 * sandi_spent

-- Theorem statement with the proof to be added
theorem gillian_more_than_three_times_sandi :
  gillian_spent - three_times_sandi_spent = 150 := 
sorry

end gillian_more_than_three_times_sandi_l415_41562


namespace algebra_eq_iff_sum_eq_one_l415_41587

-- Definitions from conditions
def expr1 (a b c : ℝ) : ℝ := a + b * c
def expr2 (a b c : ℝ) : ℝ := (a + b) * (a + c)

-- Lean statement for the proof problem
theorem algebra_eq_iff_sum_eq_one (a b c : ℝ) : expr1 a b c = expr2 a b c ↔ a + b + c = 1 :=
by
  sorry

end algebra_eq_iff_sum_eq_one_l415_41587


namespace sum_of_integers_l415_41554

theorem sum_of_integers (a b c d : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) (h4 : d > 1)
    (h_prod : a * b * c * d = 1000000)
    (h_gcd1 : Nat.gcd a b = 1) (h_gcd2 : Nat.gcd a c = 1) (h_gcd3 : Nat.gcd a d = 1)
    (h_gcd4 : Nat.gcd b c = 1) (h_gcd5 : Nat.gcd b d = 1) (h_gcd6 : Nat.gcd c d = 1) : 
    a + b + c + d = 15698 :=
sorry

end sum_of_integers_l415_41554


namespace biscuits_per_dog_l415_41593

-- Define constants for conditions
def total_biscuits : ℕ := 6
def number_of_dogs : ℕ := 2

-- Define the statement to prove
theorem biscuits_per_dog : total_biscuits / number_of_dogs = 3 := by
  -- Calculation here
  sorry

end biscuits_per_dog_l415_41593


namespace linda_winning_probability_l415_41556

noncomputable def probability_linda_wins : ℝ :=
  (1 / 16 : ℝ) / (1 - (1 / 32 : ℝ))

theorem linda_winning_probability :
  probability_linda_wins = 2 / 31 :=
sorry

end linda_winning_probability_l415_41556


namespace largest_N_cannot_pay_exactly_without_change_l415_41555

theorem largest_N_cannot_pay_exactly_without_change
  (N : ℕ) (hN : N ≤ 50) :
  (¬ ∃ a b : ℕ, N = 5 * a + 6 * b) ↔ N = 19 := by
  sorry

end largest_N_cannot_pay_exactly_without_change_l415_41555


namespace inequality_proof_l415_41584

theorem inequality_proof
  (x y z : ℝ)
  (h_x : x ≥ 0)
  (h_y : y ≥ 0)
  (h_z : z > 0)
  (h_xy : x ≥ y)
  (h_yz : y ≥ z) :
  (x + y + z) * (x + y - z) * (x - y + z) / (x * y * z) ≥ 3 := by
  sorry

end inequality_proof_l415_41584


namespace equiangular_hexagon_sides_l415_41546

variable {a b c d e f : ℝ}

-- Definition of the equiangular hexagon condition
def equiangular_hexagon (a b c d e f : ℝ) := true

theorem equiangular_hexagon_sides (h : equiangular_hexagon a b c d e f) :
  a - d = e - b ∧ e - b = c - f :=
by
  sorry

end equiangular_hexagon_sides_l415_41546


namespace dishonest_dealer_uses_correct_weight_l415_41550

noncomputable def dishonest_dealer_weight (profit_percent : ℝ) (true_weight : ℝ) : ℝ :=
  true_weight - (profit_percent / 100 * true_weight)

theorem dishonest_dealer_uses_correct_weight :
  dishonest_dealer_weight 11.607142857142861 1 = 0.8839285714285714 :=
by
  -- We skip the proof here
  sorry

end dishonest_dealer_uses_correct_weight_l415_41550


namespace jean_total_jail_time_l415_41568

def arson_counts := 3
def burglary_counts := 2
def petty_larceny_multiplier := 6
def arson_sentence_per_count := 36
def burglary_sentence_per_count := 18
def petty_larceny_fraction := 1/3

def total_jail_time :=
  arson_counts * arson_sentence_per_count +
  burglary_counts * burglary_sentence_per_count +
  (petty_larceny_multiplier * burglary_counts) * (petty_larceny_fraction * burglary_sentence_per_count)

theorem jean_total_jail_time : total_jail_time = 216 :=
by
  sorry

end jean_total_jail_time_l415_41568


namespace value_of_b_l415_41515

theorem value_of_b (b : ℝ) (h1 : 1/2 * (b / 3) * b = 6) (h2 : b ≥ 0) : b = 6 := sorry

end value_of_b_l415_41515


namespace days_for_30_men_to_build_wall_l415_41563

theorem days_for_30_men_to_build_wall 
  (men1 days1 men2 k : ℕ)
  (h1 : men1 = 18)
  (h2 : days1 = 5)
  (h3 : men2 = 30)
  (h_k : men1 * days1 = k)
  : (men2 * 3 = k) := by 
sorry

end days_for_30_men_to_build_wall_l415_41563


namespace range_of_a_l415_41508

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) :=
sorry

end range_of_a_l415_41508


namespace reciprocal_of_lcm_24_221_l415_41537

theorem reciprocal_of_lcm_24_221 : (1 / Nat.lcm 24 221) = (1 / 5304) :=
by 
  sorry

end reciprocal_of_lcm_24_221_l415_41537


namespace greatest_number_of_matching_pairs_l415_41579

theorem greatest_number_of_matching_pairs 
  (original_pairs : ℕ := 27)
  (lost_shoes : ℕ := 9) 
  (remaining_pairs : ℕ := original_pairs - (lost_shoes / 1))
  : remaining_pairs = 18 := by
  sorry

end greatest_number_of_matching_pairs_l415_41579


namespace average_customers_per_day_l415_41511

-- Define the number of customers each day:
def customers_per_day : List ℕ := [10, 12, 15, 13, 18, 16, 11]

-- Define the total number of days in a week
def days_in_week : ℕ := 7

-- Define the theorem stating the average number of daily customers
theorem average_customers_per_day :
  (customers_per_day.sum : ℚ) / days_in_week = 13.57 :=
by
  sorry

end average_customers_per_day_l415_41511


namespace positive_multiples_of_6_l415_41559

theorem positive_multiples_of_6 (k a b : ℕ) (h₁ : a = (3 + 3 * k))
  (h₂ : b = 24) (h₃ : a^2 - b^2 = 0) : k = 7 :=
sorry

end positive_multiples_of_6_l415_41559


namespace village_population_rate_l415_41536

theorem village_population_rate (R : ℕ) :
  (76000 - 17 * R = 42000 + 17 * 800) → R = 1200 :=
by
  intro h
  -- The actual proof is omitted.
  sorry

end village_population_rate_l415_41536


namespace number_of_participants_l415_41502

theorem number_of_participants (n : ℕ) (h : n * (n - 1) / 2 = 171) : n = 19 :=
by
  sorry

end number_of_participants_l415_41502


namespace Margo_paired_with_Irma_probability_l415_41528

noncomputable def probability_Margo_paired_with_Irma : ℚ :=
  1 / 29

theorem Margo_paired_with_Irma_probability :
  let total_students := 30
  let number_of_pairings := total_students - 1
  probability_Margo_paired_with_Irma = 1 / number_of_pairings := 
by
  sorry

end Margo_paired_with_Irma_probability_l415_41528


namespace problem_statement_l415_41591

def number_of_combinations (n k : ℕ) : ℕ := Nat.choose n k

def successful_outcomes : ℕ :=
  (number_of_combinations 3 1) * (number_of_combinations 5 1) * (number_of_combinations 4 5) +
  (number_of_combinations 3 2) * (number_of_combinations 4 5)

def total_outcomes : ℕ := number_of_combinations 12 7

def probability_at_least_75_cents : ℚ :=
  successful_outcomes / total_outcomes

theorem problem_statement : probability_at_least_75_cents = 3 / 22 := by
  sorry

end problem_statement_l415_41591


namespace red_pieces_count_l415_41586

-- Define the conditions
def total_pieces : ℕ := 3409
def blue_pieces : ℕ := 3264

-- Prove the number of red pieces
theorem red_pieces_count : total_pieces - blue_pieces = 145 :=
by sorry

end red_pieces_count_l415_41586


namespace tangent_perpendicular_intersection_x_4_l415_41529

noncomputable def f (x : ℝ) := (x^2 / 4) - (4 * Real.log x)
noncomputable def f' (x : ℝ) := (1/2 : ℝ) * x - 4 / x

theorem tangent_perpendicular_intersection_x_4 :
  ∀ x : ℝ, (0 < x) → (f' x = 1) → (x = 4) :=
by {
  sorry
}

end tangent_perpendicular_intersection_x_4_l415_41529


namespace repeating_decimal_fraction_l415_41577

noncomputable def repeating_decimal := 4 + 36 / 99

theorem repeating_decimal_fraction : 
  repeating_decimal = 144 / 33 := 
sorry

end repeating_decimal_fraction_l415_41577


namespace tangent_line_through_point_l415_41551

theorem tangent_line_through_point (t : ℝ) :
    (∃ l : ℝ → ℝ, (∃ m : ℝ, (∀ x, l x = 2 * m * x - m^2) ∧ (t = m - 2 * m + 2 * m * m) ∧ m = 1/2) ∧ l t = 0)
    → t = 1/4 :=
by
  sorry

end tangent_line_through_point_l415_41551


namespace square_area_with_tangent_circles_l415_41523

theorem square_area_with_tangent_circles :
  let r := 3 -- radius of each circle in inches
  let d := 2 * r -- diameter of each circle in inches
  let side_length := 2 * d -- side length of the square in inches
  let area := side_length * side_length -- area of the square in square inches
  side_length = 12 ∧ area = 144 :=
by
  let r := 3
  let d := 2 * r
  let side_length := 2 * d
  let area := side_length * side_length
  sorry

end square_area_with_tangent_circles_l415_41523


namespace polynomial_divisible_by_5040_l415_41520

theorem polynomial_divisible_by_5040 (n : ℤ) (hn : n > 3) :
  5040 ∣ (n^7 - 14 * n^5 + 49 * n^3 - 36 * n) :=
sorry

end polynomial_divisible_by_5040_l415_41520


namespace min_area_triangle_l415_41500

theorem min_area_triangle (m n : ℝ) (h1 : (1 : ℝ) / m + (2 : ℝ) / n = 1) (h2 : m > 0) (h3 : n > 0) :
  ∃ A B C : ℝ, 
  ((0 < A) ∧ (0 < B) ∧ ((1 : ℝ) / A + (2 : ℝ) / B = 1) ∧ (A * B = C) ∧ (2 / C = mn)) ∧ (C = 4) :=
by
  sorry

end min_area_triangle_l415_41500


namespace hexagon_largest_angle_l415_41570

theorem hexagon_largest_angle (x : ℝ) 
    (h_sum : (x + 2) + (2*x + 4) + (3*x - 6) + (4*x + 8) + (5*x - 10) + (6*x + 12) = 720) :
    (6*x + 12) = 215 :=
by
  sorry

end hexagon_largest_angle_l415_41570


namespace problem_1_problem_2_l415_41517

open Set Real

noncomputable def A : Set ℝ := {x | x^2 - 3 * x - 18 ≤ 0}

noncomputable def B (m : ℝ) : Set ℝ := {x | m - 8 ≤ x ∧ x ≤ m + 4}

theorem problem_1 : (m = 3) → ((compl A) ∩ (B m) = {x | (-5 ≤ x ∧ x < -3) ∨ (6 < x ∧ x ≤ 7)}) :=
by
  sorry

theorem problem_2 : (A ∩ (B m) = A) → (2 ≤ m ∧ m ≤ 5) :=
by
  sorry

end problem_1_problem_2_l415_41517


namespace area_triangle_QXY_l415_41573

-- Definition of the problem
def length_rectangle (PQ PS : ℝ) : Prop :=
  PQ = 8 ∧ PS = 6

def diagonal_division (PR : ℝ) (X Y : ℝ) : Prop :=
  PR = 10 ∧ X = 2.5 ∧ Y = 2.5

-- The statement we need to prove
theorem area_triangle_QXY
  (PQ PS PR X Y : ℝ)
  (h1 : length_rectangle PQ PS)
  (h2 : diagonal_division PR X Y)
  : ∃ (A : ℝ), A = 6 := by
  sorry

end area_triangle_QXY_l415_41573


namespace gcd_45_75_l415_41522

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l415_41522


namespace parallel_planes_l415_41505

variables {Point Line Plane : Type}
variables (a : Line) (α β : Plane)

-- Conditions
def line_perpendicular_plane (l: Line) (p: Plane) : Prop := sorry
def planes_parallel (p₁ p₂: Plane) : Prop := sorry

-- Problem statement
theorem parallel_planes (h1: line_perpendicular_plane a α) 
                        (h2: line_perpendicular_plane a β) : 
                        planes_parallel α β :=
sorry

end parallel_planes_l415_41505


namespace talia_father_age_l415_41594

theorem talia_father_age 
  (t tf tm ta : ℕ) 
  (h1 : t + 7 = 20)
  (h2 : tm = 3 * t)
  (h3 : tf + 3 = tm)
  (h4 : ta = (tm - t) / 2)
  (h5 : ta + 2 = tf + 5) : 
  tf = 36 :=
by
  sorry

end talia_father_age_l415_41594


namespace large_circle_radius_l415_41514

noncomputable def radius_of_large_circle : ℝ :=
  let r_small := 1
  let side_length := 2 * r_small
  let diagonal_length := Real.sqrt (side_length ^ 2 + side_length ^ 2)
  let radius_large := (diagonal_length / 2) + r_small
  radius_large + r_small

theorem large_circle_radius :
  radius_of_large_circle = Real.sqrt 2 + 2 :=
by
  sorry

end large_circle_radius_l415_41514


namespace real_solutions_eq_31_l415_41590

noncomputable def number_of_real_solutions : ℕ :=
  let zero := 0
  let fifty := 50
  let neg_fifty := -50
  let num_intervals := 8
  let num_solutions_per_interval := 2
  let total_solutions := num_intervals * num_solutions_per_interval * 2 - 1
  total_solutions

theorem real_solutions_eq_31 : number_of_real_solutions = 31 := by
  sorry

end real_solutions_eq_31_l415_41590


namespace circumscribed_circle_radius_l415_41519

-- Definitions of side lengths
def a : ℕ := 5
def b : ℕ := 12

-- Defining the hypotenuse based on the Pythagorean theorem
def hypotenuse (a b : ℕ) : ℕ := Nat.sqrt (a * a + b * b)

-- Radius of the circumscribed circle of a right triangle
def radius (hypotenuse : ℕ) : ℕ := hypotenuse / 2

-- Theorem: The radius of the circumscribed circle of the right triangle is 13 / 2 = 6.5
theorem circumscribed_circle_radius : 
  radius (hypotenuse a b) = 13 / 2 :=
by
  sorry

end circumscribed_circle_radius_l415_41519


namespace inequality_sqrt_three_l415_41547

theorem inequality_sqrt_three (a b : ℤ) (h1 : a > b) (h2 : b > 1)
  (h3 : (a + b) ∣ (a * b + 1))
  (h4 : (a - b) ∣ (a * b - 1)) : a < Real.sqrt 3 * b := by
  sorry

end inequality_sqrt_three_l415_41547


namespace solve_for_x_l415_41588

theorem solve_for_x :
  ∃ x : ℝ, (1 / (x + 5) + 1 / (x + 2) + 1 / (2 * x) = 1 / x) ∧ (1 / (x + 5) + 1 / (x + 2) = 1 / (x + 3)) ∧ x = 2 :=
by
  sorry

end solve_for_x_l415_41588


namespace unit_digit_div_l415_41580

theorem unit_digit_div (n : ℕ) : (33 * 10) % (2 ^ 1984) = n % 10 :=
by
  have h := 2 ^ 1984
  have u_digit_2_1984 := 6 -- Since 1984 % 4 = 0, last digit in the cycle of 2^n for n ≡ 0 [4] is 6
  sorry
  
example : (33 * 10) / (2 ^ 1984) % 10 = 6 :=
by sorry

end unit_digit_div_l415_41580


namespace travel_distance_l415_41564

noncomputable def distance_traveled (AB BC : ℝ) : ℝ :=
  let BD := Real.sqrt (AB^2 + BC^2)
  let arc1 := (2 * Real.pi * BD) / 4
  let arc2 := (2 * Real.pi * AB) / 4
  arc1 + arc2

theorem travel_distance (hAB : AB = 3) (hBC : BC = 4) : 
  distance_traveled AB BC = 4 * Real.pi := by
    sorry

end travel_distance_l415_41564


namespace carlson_total_land_l415_41585

open Real

theorem carlson_total_land 
  (initial_land : ℝ)
  (cost_additional_land1 : ℝ)
  (cost_additional_land2 : ℝ)
  (cost_per_square_meter : ℝ) :
  initial_land = 300 →
  cost_additional_land1 = 8000 →
  cost_additional_land2 = 4000 →
  cost_per_square_meter = 20 →
  (initial_land + (cost_additional_land1 + cost_additional_land2) / cost_per_square_meter) = 900 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  done

end carlson_total_land_l415_41585


namespace smallest_angle_of_trapezoid_l415_41566

theorem smallest_angle_of_trapezoid (a d : ℝ) (h1 : a + 3 * d = 120) (h2 : 4 * a + 6 * d = 360) :
  a = 60 := by
  sorry

end smallest_angle_of_trapezoid_l415_41566


namespace geometric_mean_2_6_l415_41516

theorem geometric_mean_2_6 : ∃ x : ℝ, x^2 = 2 * 6 ∧ (x = 2 * Real.sqrt 3 ∨ x = - (2 * Real.sqrt 3)) :=
by
  sorry

end geometric_mean_2_6_l415_41516


namespace seventh_oblong_number_l415_41595

/-- An oblong number is the number of dots in a rectangular grid where the number of rows is one more than the number of columns. -/
def is_oblong_number (n : ℕ) (x : ℕ) : Prop :=
  x = n * (n + 1)

/-- The 7th oblong number is 56. -/
theorem seventh_oblong_number : ∃ x, is_oblong_number 7 x ∧ x = 56 :=
by 
  use 56
  unfold is_oblong_number
  constructor
  rfl -- This confirms the computation 7 * 8 = 56
  sorry -- Wrapping up the proof, no further steps needed

end seventh_oblong_number_l415_41595


namespace find_second_sum_l415_41582

def sum : ℕ := 2717
def interest_rate_first : ℚ := 3 / 100
def interest_rate_second : ℚ := 5 / 100
def time_first : ℕ := 8
def time_second : ℕ := 3

theorem find_second_sum (x : ℚ) (h : x * interest_rate_first * time_first = (sum - x) * interest_rate_second * time_second) : 
  sum - x = 2449 :=
by
  sorry

end find_second_sum_l415_41582


namespace intersection_point_correct_l415_41599

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Line :=
(p1 : Point3D) (p2 : Point3D)

structure Plane :=
(trace : Line) (point : Point3D)

noncomputable def intersection_point (l : Line) (β : Plane) : Point3D := sorry

theorem intersection_point_correct (l : Line) (β : Plane) (P : Point3D) :
  let res := intersection_point l β
  res = P :=
sorry

end intersection_point_correct_l415_41599


namespace area_of_garden_l415_41545

theorem area_of_garden (L P : ℝ) (H1 : 1500 = 30 * L) (H2 : 1500 = 12 * P) (H3 : P = 2 * L + 2 * (P / 2 - L)) : 
  (L * (P/2 - L)) = 625 :=
by
  sorry

end area_of_garden_l415_41545


namespace quadratic_real_roots_m_l415_41581

theorem quadratic_real_roots_m (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x1 + 4 * x1 + m = 0 ∧ x2 * x2 + 4 * x2 + m = 0) →
  m ≤ 4 :=
by
  sorry

end quadratic_real_roots_m_l415_41581


namespace inequality_holds_l415_41597

theorem inequality_holds (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 := 
sorry

end inequality_holds_l415_41597


namespace exceeding_fraction_l415_41532

def repeatingDecimal : ℚ := 8 / 33
def decimalFraction : ℚ := 6 / 25
def difference : ℚ := repeatingDecimal - decimalFraction

theorem exceeding_fraction :
  difference = 2 / 825 := by
  sorry

end exceeding_fraction_l415_41532


namespace find_stream_speed_l415_41501

variable (boat_speed dist_downstream dist_upstream : ℝ)
variable (stream_speed : ℝ)

noncomputable def speed_of_stream (boat_speed dist_downstream dist_upstream : ℝ) : ℝ :=
  let t_downstream := dist_downstream / (boat_speed + stream_speed)
  let t_upstream := dist_upstream / (boat_speed - stream_speed)
  if t_downstream = t_upstream then stream_speed else 0

theorem find_stream_speed
  (h : speed_of_stream 20 26 14 stream_speed = stream_speed) :
  stream_speed = 6 :=
sorry

end find_stream_speed_l415_41501


namespace evaluate_expression_l415_41583

theorem evaluate_expression : 
  (1 - (2 / 5)) / (1 - (1 / 4)) = (4 / 5) := 
by 
  sorry

end evaluate_expression_l415_41583


namespace distance_between_ports_l415_41504

theorem distance_between_ports (x : ℝ) (speed_ship : ℝ) (speed_water : ℝ) (time_difference : ℝ) 
  (speed_downstream := speed_ship + speed_water) 
  (speed_upstream := speed_ship - speed_water) 
  (time_downstream := x / speed_downstream) 
  (time_upstream := x / speed_upstream) 
  (h : time_downstream + time_difference = time_upstream) 
  (h_ship : speed_ship = 26)
  (h_water : speed_water = 2)
  (h_time : time_difference = 3) : x = 504 :=
by
  -- The proof is omitted 
  sorry

end distance_between_ports_l415_41504


namespace lunch_break_duration_l415_41533

/-- Paula and her two helpers start at 7:00 AM and paint 60% of a house together,
    finishing at 5:00 PM. The next day, only the helpers paint and manage to
    paint 30% of another house, finishing at 3:00 PM. On the third day, Paula
    paints alone and paints the remaining 40% of the house, finishing at 4:00 PM.
    Prove that the length of their lunch break each day is 1 hour (60 minutes). -/
theorem lunch_break_duration :
  ∃ (L : ℝ), 
    (0 < L) ∧ 
    (L < 10) ∧
    (∃ (p h : ℝ), 
       (10 - L) * (p + h) = 0.6 ∧
       (8 - L) * h = 0.3 ∧
       (9 - L) * p = 0.4) ∧  
    L = 1 :=
by
  sorry

end lunch_break_duration_l415_41533


namespace smallest_x_for_multiple_l415_41530

theorem smallest_x_for_multiple (x : ℕ) (h720 : 720 = 2^4 * 3^2 * 5) (h1250 : 1250 = 2 * 5^4) : 
  (∃ x, (x > 0) ∧ (1250 ∣ (720 * x))) → x = 125 :=
by
  sorry

end smallest_x_for_multiple_l415_41530


namespace axis_of_symmetry_l415_41512

theorem axis_of_symmetry {a b c : ℝ} (h1 : (2 : ℝ) * (a * 2 + b) + c = 5) (h2 : (4 : ℝ) * (a * 4 + b) + c = 5) : 
  (2 + 4) / 2 = 3 := 
by 
  sorry

end axis_of_symmetry_l415_41512


namespace at_least_three_points_in_circle_l415_41574

noncomputable def point_in_circle (p : ℝ × ℝ) (c : ℝ × ℝ) (r : ℝ) : Prop :=
(dist p c) ≤ r

theorem at_least_three_points_in_circle (points : Fin 51 → (ℝ × ℝ)) (side_length : ℝ) (circle_radius : ℝ)
  (h_side_length : side_length = 1) (h_circle_radius : circle_radius = 1 / 7) : 
  ∃ (c : ℝ × ℝ), ∃ (p1 p2 p3 : Fin 51), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    point_in_circle (points p1) c circle_radius ∧ 
    point_in_circle (points p2) c circle_radius ∧ 
    point_in_circle (points p3) c circle_radius :=
sorry

end at_least_three_points_in_circle_l415_41574


namespace worker_b_time_l415_41548

theorem worker_b_time (T_B : ℝ) : 
  (1 / 10) + (1 / T_B) = 1 / 6 → T_B = 15 := by
  intro h
  sorry

end worker_b_time_l415_41548


namespace inequality_reciprocal_of_negative_l415_41567

variable {a b : ℝ}

theorem inequality_reciprocal_of_negative (h : a < b) (h_neg_a : a < 0) (h_neg_b : b < 0) : 
  (1 / a) > (1 / b) := by
  sorry

end inequality_reciprocal_of_negative_l415_41567


namespace find_point_B_coordinates_l415_41535

theorem find_point_B_coordinates : 
  ∃ B : ℝ × ℝ, 
    (∀ A C B : ℝ × ℝ, A = (2, 3) ∧ C = (0, 1) ∧ 
    (B.1 - A.1, B.2 - A.2) = (-2) • (C.1 - B.1, C.2 - B.2)) → B = (-2, -1) :=
by 
  sorry

end find_point_B_coordinates_l415_41535


namespace smallest_n_inequality_l415_41543

theorem smallest_n_inequality:
  ∃ n : ℤ, (∀ x y z : ℝ, (x^2 + 2 * y^2 + z^2)^2 ≤ n * (x^4 + 3 * y^4 + z^4)) ∧ n = 4 :=
by
  sorry

end smallest_n_inequality_l415_41543


namespace extreme_value_at_one_symmetric_points_range_l415_41561

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then
  x^2 + 3 * a * x
else
  2 * Real.exp x - x^2 + 2 * a * x

theorem extreme_value_at_one (a : ℝ) :
  (∀ x > 0, f x a = 2 * Real.exp x - x^2 + 2 * a * x) →
  (∀ x < 0, f x a = x^2 + 3 * a * x) →
  (∀ x > 0, deriv (fun x => f x a) x = 2 * Real.exp x - 2 * x + 2 * a) →
  deriv (fun x => f x a) 1 = 0 →
  a = 1 - Real.exp 1 :=
  sorry

theorem symmetric_points_range (a : ℝ) :
  (∃ x0 > 0, (∃ y0 : ℝ, 
  (f x0 a = y0 ∧ f (-x0) a = -y0))) →
  a ≥ 2 * Real.exp 1 :=
  sorry

end extreme_value_at_one_symmetric_points_range_l415_41561


namespace sum_of_star_tip_angles_l415_41571

noncomputable def sum_star_tip_angles : ℝ :=
  let segment_angle := 360 / 8
  let subtended_arc := 3 * segment_angle
  let theta := subtended_arc / 2
  8 * theta

theorem sum_of_star_tip_angles:
  sum_star_tip_angles = 540 := by
  sorry

end sum_of_star_tip_angles_l415_41571


namespace no_such_point_exists_l415_41569

theorem no_such_point_exists 
  (side_length : ℝ)
  (original_area : ℝ)
  (total_area_after_first_rotation : ℝ)
  (total_area_after_second_rotation : ℝ)
  (no_overlapping_exists : Prop) :
  side_length = 12 → 
  original_area = 144 → 
  total_area_after_first_rotation = 211 → 
  total_area_after_second_rotation = 287 →
  no_overlapping_exists := sorry

end no_such_point_exists_l415_41569


namespace quadratic_solution_l415_41598

theorem quadratic_solution (x : ℝ) : -x^2 + 4 * x + 5 < 0 ↔ x > 5 ∨ x < -1 :=
sorry

end quadratic_solution_l415_41598


namespace Ivan_defeats_Koschei_l415_41592

-- Definitions of the springs and conditions based on the problem
section

variable (S: ℕ → Prop)  -- S(n) means the water from spring n
variable (deadly: ℕ → Prop)  -- deadly(n) if water from nth spring is deadly

-- Conditions
axiom accessibility (n: ℕ): (1 ≤ n ∧ n ≤ 9 → ∀ i: ℕ, S i)
axiom koschei_access: S 10
axiom lethality (n: ℕ): (S n → deadly n)
axiom neutralize (i j: ℕ): (1 ≤ i ∧ i < j ∧ j ≤ 9 → ∃ k: ℕ, S k ∧ k > j → ¬deadly i)

-- Statement to prove
theorem Ivan_defeats_Koschei:
  ∃ i: ℕ, (1 ≤ i ∧ i ≤ 9) → (S 10 → ¬deadly i) ∧ (S 0 ∧ (S 10 → deadly 0)) :=
sorry

end

end Ivan_defeats_Koschei_l415_41592


namespace cube_volume_l415_41521

theorem cube_volume (d_AF : Real) (h : d_AF = 6 * Real.sqrt 2) : ∃ (V : Real), V = 216 :=
by {
  sorry
}

end cube_volume_l415_41521


namespace provisions_remaining_days_l415_41518

-- Definitions based on the conditions
def initial_men : ℕ := 1000
def initial_provisions_days : ℕ := 60
def days_elapsed : ℕ := 15
def reinforcement_men : ℕ := 1250

-- Mathematical computation for Lean
def total_provisions : ℕ := initial_men * initial_provisions_days
def provisions_left : ℕ := initial_men * (initial_provisions_days - days_elapsed)
def total_men_after_reinforcement : ℕ := initial_men + reinforcement_men

-- Statement to prove
theorem provisions_remaining_days : provisions_left / total_men_after_reinforcement = 20 :=
by
  -- The proof steps will be filled here, but for now, we use sorry to skip them.
  sorry

end provisions_remaining_days_l415_41518


namespace infinite_perfect_squares_of_form_l415_41544

theorem infinite_perfect_squares_of_form (k : ℕ) (hk : 0 < k) : 
  ∃ (n : ℕ), ∀ m : ℕ, ∃ a : ℕ, (n + m) * 2^k - 7 = a^2 :=
sorry

end infinite_perfect_squares_of_form_l415_41544


namespace smaller_number_is_24_l415_41507

theorem smaller_number_is_24 (x y : ℕ) (h1 : x + y = 44) (h2 : 5 * x = 6 * y) : x = 24 :=
by
  sorry

end smaller_number_is_24_l415_41507


namespace max_distance_sum_l415_41525

theorem max_distance_sum {P : ℝ × ℝ} 
  (C : Set (ℝ × ℝ)) 
  (hC : ∀ (P : ℝ × ℝ), P ∈ C ↔ (P.1 - 3)^2 + (P.2 - 4)^2 = 1)
  (A : ℝ × ℝ := (0, -1))
  (B : ℝ × ℝ := (0, 1)) :
  ∃ P : ℝ × ℝ, 
    P ∈ C ∧ (P = (18 / 5, 24 / 5)) :=
by
  sorry

end max_distance_sum_l415_41525


namespace relationship_among_log_sin_exp_l415_41542

theorem relationship_among_log_sin_exp (x : ℝ) (h₁ : 0 < x) (h₂ : x < 1) (a b c : ℝ) 
(h₃ : a = Real.log 3 / Real.log x) (h₄ : b = Real.sin x)
(h₅ : c = 2 ^ x) : a < b ∧ b < c := 
sorry

end relationship_among_log_sin_exp_l415_41542


namespace find_value_of_x_l415_41596

theorem find_value_of_x :
  (0.47 * 1442 - 0.36 * 1412) + 65 = 234.42 := 
by
  sorry

end find_value_of_x_l415_41596


namespace length_linear_function_alpha_increase_l415_41513

variable (l : ℝ) (l₀ : ℝ) (t : ℝ) (α : ℝ)

theorem length_linear_function 
  (h_formula : l = l₀ * (1 + α * t)) : 
  ∃ (f : ℝ → ℝ), (∀ t, f t = l₀ + l₀ * α * t ∧ (l = f t)) :=
by {
  -- Proof would go here
  sorry
}

theorem alpha_increase 
  (h_formula : l = l₀ * (1 + α * t))
  (h_initial : t = 1) :
  α = (l - l₀) / l₀ :=
by {
  -- Proof would go here
  sorry
}

end length_linear_function_alpha_increase_l415_41513


namespace no_positive_integer_exists_l415_41526

theorem no_positive_integer_exists
  (P1 P2 : ℤ → ℤ)
  (a : ℤ)
  (h_a_neg : a < 0)
  (h_common_root : P1 a = 0 ∧ P2 a = 0) :
  ¬ ∃ b : ℤ, b > 0 ∧ P1 b = 2007 ∧ P2 b = 2008 :=
sorry

end no_positive_integer_exists_l415_41526


namespace maximum_profit_is_achieved_at_14_yuan_l415_41557

-- Define the initial conditions
def cost_per_unit : ℕ := 8
def initial_selling_price : ℕ := 10
def initial_selling_quantity : ℕ := 100

-- Define the sales volume decrease per price increase
def decrease_per_yuan_increase : ℕ := 10

-- Define the profit function
def profit (price_increase : ℕ) : ℕ :=
  let new_selling_price := initial_selling_price + price_increase
  let new_selling_quantity := initial_selling_quantity - (decrease_per_yuan_increase * price_increase)
  (new_selling_price - cost_per_unit) * new_selling_quantity

-- Define the statement to be proved
theorem maximum_profit_is_achieved_at_14_yuan :
  ∃ price_increase : ℕ, price_increase = 4 ∧ profit price_increase = profit 4 := by
  sorry

end maximum_profit_is_achieved_at_14_yuan_l415_41557


namespace distance_between_sets_is_zero_l415_41572

noncomputable def A (x : ℝ) : ℝ := 2 * x - 1
noncomputable def B (x : ℝ) : ℝ := x^2 + 1

theorem distance_between_sets_is_zero : 
  ∃ (a b : ℝ), (∃ x₀ : ℝ, a = A x₀) ∧ (∃ y₀ : ℝ, b = B y₀) ∧ abs (a - b) = 0 := 
sorry

end distance_between_sets_is_zero_l415_41572


namespace value_of_algebraic_expression_l415_41538

noncomputable def quadratic_expression (m : ℝ) : ℝ :=
  3 * m * (2 * m - 3) - 1

theorem value_of_algebraic_expression (m : ℝ) (h : 2 * m^2 - 3 * m - 1 = 0) : quadratic_expression m = 2 :=
by {
  sorry
}

end value_of_algebraic_expression_l415_41538


namespace number_of_white_balls_l415_41553

theorem number_of_white_balls (total : ℕ) (freq_red freq_black : ℚ) (h1 : total = 120) 
                              (h2 : freq_red = 0.15) (h3 : freq_black = 0.45) : 
                              (total - total * freq_red - total * freq_black = 48) :=
by sorry

end number_of_white_balls_l415_41553


namespace shirts_and_pants_neither_plaid_nor_purple_l415_41510

variable (total_shirts total_pants plaid_shirts purple_pants : Nat)

def non_plaid_shirts (total_shirts plaid_shirts : Nat) : Nat := total_shirts - plaid_shirts
def non_purple_pants (total_pants purple_pants : Nat) : Nat := total_pants - purple_pants

theorem shirts_and_pants_neither_plaid_nor_purple :
  total_shirts = 5 → total_pants = 24 → plaid_shirts = 3 → purple_pants = 5 →
  non_plaid_shirts total_shirts plaid_shirts + non_purple_pants total_pants purple_pants = 21 :=
by
  intros
  -- Placeholder for proof to ensure the theorem builds correctly
  sorry

end shirts_and_pants_neither_plaid_nor_purple_l415_41510


namespace arc_lengths_l415_41509

-- Definitions for the given conditions
def circumference : ℝ := 80  -- Circumference of the circle

-- Angles in degrees
def angle_AOM : ℝ := 45
def angle_MOB : ℝ := 90

-- Radius of the circle using the formula C = 2 * π * r
noncomputable def radius : ℝ := circumference / (2 * Real.pi)

-- Calculate the arc lengths using the angles
noncomputable def arc_length_AM : ℝ := (angle_AOM / 360) * circumference
noncomputable def arc_length_MB : ℝ := (angle_MOB / 360) * circumference

-- The theorem stating the required lengths
theorem arc_lengths (h : circumference = 80 ∧ angle_AOM = 45 ∧ angle_MOB = 90) :
  arc_length_AM = 10 ∧ arc_length_MB = 20 :=
by
  sorry

end arc_lengths_l415_41509


namespace part1_part2_part3_l415_41540

-- Part 1
theorem part1 (a b : ℝ) : 
    3 * (a - b) ^ 2 - 6 * (a - b) ^ 2 + 2 * (a - b) ^ 2 = - (a - b) ^ 2 := 
    sorry

-- Part 2
theorem part2 (x y : ℝ) (h : x ^ 2 - 2 * y = 4) : 
    3 * x ^ 2 - 6 * y - 21 = -9 := 
    sorry

-- Part 3
theorem part3 (a b c d : ℝ) (h1 : a - 5 * b = 3) (h2 : 5 * b - 3 * c = -5) (h3 : 3 * c - d = 10) : 
    (a - 3 * c) + (5 * b - d) - (5 * b - 3 * c) = 8 := 
    sorry

end part1_part2_part3_l415_41540


namespace anne_trip_shorter_l415_41527

noncomputable def john_walk_distance : ℝ := 2 + 1

noncomputable def anne_walk_distance : ℝ := Real.sqrt (2^2 + 1^2)

noncomputable def distance_difference : ℝ := john_walk_distance - anne_walk_distance

noncomputable def percentage_reduction : ℝ := (distance_difference / john_walk_distance) * 100

theorem anne_trip_shorter :
  20 ≤ percentage_reduction ∧ percentage_reduction < 30 :=
by
  sorry

end anne_trip_shorter_l415_41527


namespace number_of_students_l415_41503

theorem number_of_students (B S : ℕ) 
  (h1 : S = 9 * B + 1) 
  (h2 : S = 10 * B - 10) : 
  S = 100 := 
by 
  { sorry }

end number_of_students_l415_41503


namespace decrease_hours_worked_l415_41534

theorem decrease_hours_worked (initial_hourly_wage : ℝ) (initial_hours_worked : ℝ) :
  let new_hourly_wage := initial_hourly_wage * 1.25
  let new_hours_worked := (initial_hourly_wage * initial_hours_worked) / new_hourly_wage
  initial_hours_worked > 0 → 
  initial_hourly_wage > 0 → 
  new_hours_worked < initial_hours_worked :=
by
  intros initial_hours_worked_pos initial_hourly_wage_pos
  let new_hourly_wage := initial_hourly_wage * 1.25
  let new_hours_worked := (initial_hourly_wage * initial_hours_worked) / new_hourly_wage
  sorry

end decrease_hours_worked_l415_41534


namespace compute_value_l415_41575

theorem compute_value
  (x y z : ℝ)
  (h1 : (xz / (x + y)) + (yx / (y + z)) + (zy / (z + x)) = -9)
  (h2 : (yz / (x + y)) + (zx / (y + z)) + (xy / (z + x)) = 15) :
  (y / (x + y)) + (z / (y + z)) + (x / (z + x)) = 13.5 :=
by
  sorry

end compute_value_l415_41575


namespace rectangular_garden_shorter_side_length_l415_41578

theorem rectangular_garden_shorter_side_length
  (a b : ℕ)
  (h1 : 2 * a + 2 * b = 46)
  (h2 : a * b = 108) :
  b = 9 :=
by 
  sorry

end rectangular_garden_shorter_side_length_l415_41578


namespace contradiction_even_odd_l415_41558

theorem contradiction_even_odd (a b c : ℕ) (h1 : (a % 2 = 1 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ c % 2 = 1) ∨ (b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1)) :
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) :=
by
  -- proof by contradiction
  sorry

end contradiction_even_odd_l415_41558


namespace eighth_term_l415_41531

noncomputable def S (n : ℕ) (a : ℕ → ℤ) : ℤ := (n * (a 1 + a n)) / 2

variables {a : ℕ → ℤ} {d : ℤ}

-- Conditions
axiom sum_of_first_n_terms : ∀ n : ℕ, S n a = (n * (a 1 + a n)) / 2
axiom second_term : a 2 = 3
axiom sum_of_first_five_terms : S 5 a = 25

-- Question
theorem eighth_term : a 8 = 15 :=
sorry

end eighth_term_l415_41531

import Mathlib

namespace bruce_money_left_l142_142478

-- Definitions for the given values
def initial_amount : ℕ := 71
def shirt_cost : ℕ := 5
def number_of_shirts : ℕ := 5
def pants_cost : ℕ := 26

-- The theorem that Bruce has $20 left
theorem bruce_money_left : initial_amount - (shirt_cost * number_of_shirts + pants_cost) = 20 :=
by
  sorry

end bruce_money_left_l142_142478


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l142_142941

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l142_142941


namespace sum_reciprocals_factors_12_l142_142354

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l142_142354


namespace sum_of_reciprocals_factors_12_l142_142300

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l142_142300


namespace goats_at_farm_l142_142216

theorem goats_at_farm (G C D P : ℕ) 
  (h1: C = 2 * G)
  (h2: D = (G + C) / 2)
  (h3: P = D / 3)
  (h4: G = P + 33) :
  G = 66 :=
by
  sorry

end goats_at_farm_l142_142216


namespace rhombus_triangle_area_l142_142590

theorem rhombus_triangle_area (d1 d2 : ℝ) (h_d1 : d1 = 15) (h_d2 : d2 = 20) :
  ∃ (area : ℝ), area = 75 := 
by
  sorry

end rhombus_triangle_area_l142_142590


namespace amanda_speed_l142_142672

-- Defining the conditions
def distance : ℝ := 6 -- 6 miles
def time : ℝ := 3 -- 3 hours

-- Stating the question with the conditions and the correct answer
theorem amanda_speed : (distance / time) = 2 :=
by 
  -- the proof is skipped as instructed
  sorry

end amanda_speed_l142_142672


namespace girls_exceed_boys_by_402_l142_142665

theorem girls_exceed_boys_by_402 : 
  let girls := 739
  let boys := 337
  girls - boys = 402 :=
by
  sorry

end girls_exceed_boys_by_402_l142_142665


namespace probability_student_major_b_and_below_25_l142_142669

theorem probability_student_major_b_and_below_25
  (p_student_major_b : ℝ) (p_student_major_b_below_25 : ℝ) :
  p_student_major_b = 0.30 →
  p_student_major_b_below_25 = 0.60 →
  (p_student_major_b * p_student_major_b_below_25 = 0.18) :=
by
  intros h1 h2
  simp [h1, h2]
  sorry

end probability_student_major_b_and_below_25_l142_142669


namespace inequality_power_cubed_l142_142139

theorem inequality_power_cubed
  (x y a : ℝ)
  (h_condition : (0 < a ∧ a < 1) ∧ a ^ x < a ^ y) : x^3 > y^3 :=
by {
  sorry
}

end inequality_power_cubed_l142_142139


namespace smallest_three_digit_integer_solution_l142_142233

theorem smallest_three_digit_integer_solution :
  ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    (∃ a b c : ℕ,
      n = 100 * a + 10 * b + c ∧
      1 ≤ a ∧ a ≤ 9 ∧
      0 ≤ b ∧ b ≤ 9 ∧ 
      0 ≤ c ∧ c ≤ 9 ∧
      2 * n = 100 * c + 10 * b + a + 5) ∧ 
    n = 102 := by
{
  sorry
}

end smallest_three_digit_integer_solution_l142_142233


namespace bus_children_l142_142595

theorem bus_children (X : ℕ) (initial_children : ℕ) (got_on : ℕ) (total_children_after : ℕ) 
  (h1 : initial_children = 28) 
  (h2 : got_on = 82) 
  (h3 : total_children_after = 30) 
  (h4 : initial_children + got_on - X = total_children_after) : 
  got_on - X = 2 :=
by 
  -- h1, h2, h3, and h4 are conditions from the problem
  sorry

end bus_children_l142_142595


namespace combined_distance_is_12_l142_142579

-- Define the distances the two ladies walked
def distance_second_lady : ℝ := 4
def distance_first_lady := 2 * distance_second_lady

-- Define the combined total distance
def combined_distance := distance_first_lady + distance_second_lady

-- Statement of the problem as a proof goal in Lean
theorem combined_distance_is_12 : combined_distance = 12 :=
by
  -- Definitions required for the proof
  let second := distance_second_lady
  let first := distance_first_lady
  let total := combined_distance
  
  -- Insert the necessary calculations and proof steps here
  -- Conclude with the desired result
  sorry

end combined_distance_is_12_l142_142579


namespace min_value_sin6_cos6_l142_142840

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l142_142840


namespace arithmetic_sequence_minimum_value_S_l142_142926

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l142_142926


namespace complement_of_intersection_eq_l142_142529

-- Definitions of sets with given conditions
def U : Set ℝ := {x | 0 ≤ x ∧ x < 10}
def A : Set ℝ := {x | 2 < x ∧ x ≤ 4}
def B : Set ℝ := {x | 3 < x ∧ x ≤ 5}

-- Complement of a set with respect to U
def complement_U (S : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ S}

-- Intersect two sets
def intersection (S1 S2 : Set ℝ) : Set ℝ := {x | x ∈ S1 ∧ x ∈ S2}

theorem complement_of_intersection_eq :
  complement_U (intersection A B) = {x | (0 ≤ x ∧ x ≤ 2) ∨ (5 < x ∧ x < 10)} := 
by
  sorry

end complement_of_intersection_eq_l142_142529


namespace sum_reciprocals_factors_12_l142_142359

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l142_142359


namespace larger_integer_is_21_l142_142753

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l142_142753


namespace x_is_48_percent_of_z_l142_142659

variable {x y z : ℝ}

theorem x_is_48_percent_of_z (h1 : x = 1.20 * y) (h2 : y = 0.40 * z) : x = 0.48 * z :=
by
  sorry

end x_is_48_percent_of_z_l142_142659


namespace race_meeting_time_l142_142183

noncomputable def track_length : ℕ := 500
noncomputable def first_meeting_from_marie_start : ℕ := 100
noncomputable def time_until_first_meeting : ℕ := 2
noncomputable def second_meeting_time : ℕ := 12

theorem race_meeting_time
  (h1 : track_length = 500)
  (h2 : first_meeting_from_marie_start = 100)
  (h3 : time_until_first_meeting = 2)
  (h4 : ∀ t v1 v2 : ℕ, t * (v1 + v2) = track_length)
  (h5 : 12 = second_meeting_time) :
  second_meeting_time = 12 := by
  sorry

end race_meeting_time_l142_142183


namespace find_c_l142_142860

-- Define the problem conditions and statement

variables (a b c : ℝ) (A B C : ℝ)
variable (cos_C : ℝ)
variable (sin_A sin_B : ℝ)

-- Given conditions
axiom h1 : a = 2
axiom h2 : cos_C = -1/4
axiom h3 : 3 * sin_A = 2 * sin_B
axiom sine_rule : sin_A / a = sin_B / b

-- Using sine rule to derive relation between a and b
axiom h4 : 3 * a = 2 * b

-- Cosine rule axiom
axiom cosine_rule : c^2 = a^2 + b^2 - 2 * a * b * cos_C

-- Prove c = 4
theorem find_c : c = 4 :=
by
  sorry

end find_c_l142_142860


namespace incorrect_statement_D_l142_142092

theorem incorrect_statement_D (k b x : ℝ) (hk : k < 0) (hb : b > 0) (hx : x > -b / k) :
  k * x + b ≤ 0 :=
by
  sorry

end incorrect_statement_D_l142_142092


namespace arithmetic_sequence_min_value_S_l142_142952

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l142_142952


namespace sum_of_reciprocals_factors_12_l142_142422

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l142_142422


namespace sum_of_reciprocals_factors_12_l142_142398

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l142_142398


namespace first_place_beats_joe_by_two_points_l142_142679

def points (wins draws : ℕ) : ℕ := 3 * wins + draws

theorem first_place_beats_joe_by_two_points
  (joe_wins joe_draws first_place_wins first_place_draws : ℕ)
  (h1 : joe_wins = 1)
  (h2 : joe_draws = 3)
  (h3 : first_place_wins = 2)
  (h4 : first_place_draws = 2) :
  points first_place_wins first_place_draws - points joe_wins joe_draws = 2 := by
  sorry

end first_place_beats_joe_by_two_points_l142_142679


namespace possible_scenario_l142_142520

variable {a b c d : ℝ}

-- Conditions
def abcd_positive : a * b * c * d > 0 := sorry
def a_less_than_c : a < c := sorry
def bcd_negative : b * c * d < 0 := sorry

-- Statement
theorem possible_scenario :
  (a < 0) ∧ (b > 0) ∧ (c < 0) ∧ (d > 0) :=
sorry

end possible_scenario_l142_142520


namespace netSalePrice_correct_l142_142889

-- Definitions for item costs and fees
def purchaseCostA : ℝ := 650
def handlingFeeA : ℝ := 0.02 * purchaseCostA
def totalCostA : ℝ := purchaseCostA + handlingFeeA

def purchaseCostB : ℝ := 350
def restockingFeeB : ℝ := 0.03 * purchaseCostB
def totalCostB : ℝ := purchaseCostB + restockingFeeB

def purchaseCostC : ℝ := 400
def transportationFeeC : ℝ := 0.015 * purchaseCostC
def totalCostC : ℝ := purchaseCostC + transportationFeeC

-- Desired profit percentages
def profitPercentageA : ℝ := 0.40
def profitPercentageB : ℝ := 0.25
def profitPercentageC : ℝ := 0.30

-- Net sale prices for achieving the desired profit percentages
def netSalePriceA : ℝ := totalCostA + (profitPercentageA * totalCostA)
def netSalePriceB : ℝ := totalCostB + (profitPercentageB * totalCostB)
def netSalePriceC : ℝ := totalCostC + (profitPercentageC * totalCostC)

-- Expected values
def expectedNetSalePriceA : ℝ := 928.20
def expectedNetSalePriceB : ℝ := 450.63
def expectedNetSalePriceC : ℝ := 527.80

-- Theorem to prove the net sale prices match the expected values
theorem netSalePrice_correct :
  netSalePriceA = expectedNetSalePriceA ∧
  netSalePriceB = expectedNetSalePriceB ∧
  netSalePriceC = expectedNetSalePriceC :=
by
  unfold netSalePriceA netSalePriceB netSalePriceC totalCostA totalCostB totalCostC
         handlingFeeA restockingFeeB transportationFeeC
  sorry

end netSalePrice_correct_l142_142889


namespace li_li_age_this_year_l142_142763

theorem li_li_age_this_year (A B : ℕ) (h1 : A + B = 30) (h2 : A = B + 6) : B = 12 := by
  sorry

end li_li_age_this_year_l142_142763


namespace weights_divide_three_piles_l142_142629

theorem weights_divide_three_piles (n : ℕ) (h : n > 3) :
  (∃ (k : ℕ), n = 3 * k ∨ n = 3 * k + 2) ↔
  (∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range (n + 1) ∧
   A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
   A.sum id = (n * (n + 1)) / 6 ∧ B.sum id = (n * (n + 1)) / 6 ∧ C.sum id = (n * (n + 1)) / 6) :=
sorry

end weights_divide_three_piles_l142_142629


namespace sum_of_reciprocals_of_factors_of_12_l142_142297

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142297


namespace sum_of_reciprocals_of_factors_of_12_l142_142317

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142317


namespace slope_of_l_l142_142862

def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def parallel_lines (slope : ℝ) : Prop :=
  ∃ m : ℝ, ∀ x y : ℝ, y = slope * x + m

def intersects_ellipse (slope : ℝ) : Prop :=
  parallel_lines slope ∧ ∃ x y : ℝ, ellipse x y ∧ y = slope * x + (y - slope * x)

theorem slope_of_l {l_slope : ℝ} :
  (∃ (m : ℝ) (x y : ℝ), intersects_ellipse (1 / 4) ∧ (y - l_slope * x = m)) →
  (l_slope = -2) :=
sorry

end slope_of_l_l142_142862


namespace fraction_product_simplified_l142_142070

theorem fraction_product_simplified:
  (2 / 9 : ℚ) * (5 / 8 : ℚ) = 5 / 36 :=
by {
  sorry
}

end fraction_product_simplified_l142_142070


namespace max_intersections_l142_142225

-- Define the number of circles and lines
def num_circles : ℕ := 2
def num_lines : ℕ := 3

-- Define the maximum number of intersection points of circles
def max_circle_intersections : ℕ := 2

-- Define the number of intersection points between each line and each circle
def max_line_circle_intersections : ℕ := 2

-- Define the number of intersection points among lines (using the combination formula)
def num_line_intersections : ℕ := (num_lines.choose 2)

-- Define the greatest number of points of intersection
def total_intersections : ℕ :=
  max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections

-- Prove the greatest number of points of intersection is 17
theorem max_intersections : total_intersections = 17 := by
  -- Calculating individual parts for clarity
  have h1: max_circle_intersections = 2 := rfl
  have h2: num_lines * num_circles * max_line_circle_intersections = 12 := by
    calc
      num_lines * num_circles * max_line_circle_intersections
        = 3 * 2 * 2 := by rw [num_lines, num_circles, max_line_circle_intersections]
        ... = 12 := by norm_num
  have h3: num_line_intersections = 3 := by
    calc
      num_line_intersections = (3.choose 2) := rfl
      ... = 3 := by norm_num

  -- Adding the parts to get the total intersections
  calc
    total_intersections
      = max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections := rfl
      ... = 2 + 12 + 3 := by rw [h1, h2, h3]
      ... = 17 := by norm_num

end max_intersections_l142_142225


namespace natalia_apartment_number_unit_digit_l142_142688

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def true_statements (n : ℕ) : Prop :=
  (n % 3 = 0 → true) ∧   -- Statement (1): divisible by 3
  (∃ k : ℕ, k^2 = n → true) ∧  -- Statement (2): square number
  (n % 2 = 1 → true) ∧   -- Statement (3): odd
  (n % 10 = 4 → true)     -- Statement (4): ends in 4

def three_out_of_four_true (n : ℕ) : Prop :=
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 = 1 ∧ n % 10 ≠ 4) ∨
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 ≠ 1 ∧ n % 10 = 4) ∨
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 ≠ n) ∧ n % 2 = 1 ∧ n % 10 = 4) ∨
  (n % 3 ≠ 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 = 1 ∧ n % 10 = 4)

theorem natalia_apartment_number_unit_digit :
  ∀ n : ℕ, two_digit_number n → three_out_of_four_true n → n % 10 = 1 :=
by sorry

end natalia_apartment_number_unit_digit_l142_142688


namespace cyclist_total_heartbeats_l142_142107

theorem cyclist_total_heartbeats
  (heart_rate : ℕ := 120) -- beats per minute
  (race_distance : ℕ := 50) -- miles
  (pace : ℕ := 4) -- minutes per mile
  : (race_distance * pace) * heart_rate = 24000 := by
  sorry

end cyclist_total_heartbeats_l142_142107


namespace sum_of_reciprocals_factors_12_l142_142425

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l142_142425


namespace sum_reciprocals_factors_12_l142_142405

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142405


namespace simplify_expression_l142_142194

theorem simplify_expression : (2^8 + 4^5) * ((1^3 - (-1)^3)^8) = 327680 := by
  sorry

end simplify_expression_l142_142194


namespace simple_fraction_pow_l142_142484

theorem simple_fraction_pow : (66666^4 / 22222^4) = 81 := by
  sorry

end simple_fraction_pow_l142_142484


namespace greatest_q_minus_r_l142_142708

theorem greatest_q_minus_r : 
  ∃ (q r : ℕ), 1001 = 17 * q + r ∧ q - r = 43 :=
by
  sorry

end greatest_q_minus_r_l142_142708


namespace find_k_l142_142566

theorem find_k (k : ℝ) (h : ∃ (k : ℝ), 3 = k * (-1) - 2) : k = -5 :=
by
  rcases h with ⟨k, hk⟩
  sorry

end find_k_l142_142566


namespace largest_three_digit_base7_to_decimal_l142_142980

theorem largest_three_digit_base7_to_decimal :
  (6 * 7^2 + 6 * 7^1 + 6 * 7^0) = 342 :=
by
  sorry

end largest_three_digit_base7_to_decimal_l142_142980


namespace sum_reciprocals_factors_12_l142_142408

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142408


namespace find_2u_plus_3v_l142_142147

theorem find_2u_plus_3v (u v : ℚ) (h1 : 5 * u - 6 * v = 28) (h2 : 3 * u + 5 * v = -13) :
  2 * u + 3 * v = -7767 / 645 := 
sorry

end find_2u_plus_3v_l142_142147


namespace sum_reciprocals_factors_of_12_l142_142286

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l142_142286


namespace sum_reciprocal_factors_12_l142_142255

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l142_142255


namespace larger_integer_is_21_l142_142725

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l142_142725


namespace sum_reciprocal_factors_12_l142_142246

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l142_142246


namespace first_place_beats_joe_by_two_points_l142_142678

def points (wins draws : ℕ) : ℕ := 3 * wins + draws

theorem first_place_beats_joe_by_two_points
  (joe_wins joe_draws first_place_wins first_place_draws : ℕ)
  (h1 : joe_wins = 1)
  (h2 : joe_draws = 3)
  (h3 : first_place_wins = 2)
  (h4 : first_place_draws = 2) :
  points first_place_wins first_place_draws - points joe_wins joe_draws = 2 := by
  sorry

end first_place_beats_joe_by_two_points_l142_142678


namespace sum_of_reciprocals_of_factors_of_12_l142_142432

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142432


namespace find_angle_LBC_l142_142543

variables {A B C H L : Type*} [EuclideanGeometry B H]
variables (BL BH BC AC right_angle : ℝ)
variables {AH : ℝ}

-- Conditions
def conditions :=
  (right_angle > 0) ∧
  (BL = 4) ∧
  (AH = 9 / (2 * Real.sqrt 7)) ∧
  (BH * BH = BL * BL - ((AH * 2 * (Real.sqrt 7) - 1)^2))

-- Proof goal
theorem find_angle_LBC
  (A B C H L : Type*)
  [EuclideanGeometry B H]
  (BL BH BC AC right_angle : ℝ)
  (AH : ℝ)
  (hcond : conditions A B C H L BL BH BC AC right_angle AH) :
  ∃ θ, θ = Real.arccos (23 / (4 * Real.sqrt 37)) :=
sorry

end find_angle_LBC_l142_142543


namespace arithmetic_seq_a9_l142_142536

theorem arithmetic_seq_a9 (a : ℕ → ℤ) (h1 : a 3 - a 2 = -2) (h2 : a 7 = -2) : a 9 = -6 := 
by sorry

end arithmetic_seq_a9_l142_142536


namespace robotics_club_problem_l142_142189

theorem robotics_club_problem 
    (total_students cs_students eng_students both_students : ℕ)
    (h1 : total_students = 120)
    (h2 : cs_students = 75)
    (h3 : eng_students = 50)
    (h4 : both_students = 10) :
    total_students - (cs_students - both_students + eng_students - both_students + both_students) = 5 := by
  sorry

end robotics_club_problem_l142_142189


namespace sum_reciprocals_factors_12_l142_142334

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142334


namespace larger_integer_is_21_l142_142731

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l142_142731


namespace determine_a_for_line_l142_142122

theorem determine_a_for_line (a : ℝ) (h : a ≠ 0)
  (intercept_condition : ∃ (k : ℝ), 
    ∀ x y : ℝ, (a * x - 6 * y - 12 * a = 0) → (x = 12) ∧ (y = 2 * a * x / 6) ∧ (12 = 3 * (-2 * a))) : 
  a = -2 :=
by
  sorry

end determine_a_for_line_l142_142122


namespace sum_reciprocals_of_factors_of_12_l142_142328

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l142_142328


namespace solve_system_of_equations_l142_142512

theorem solve_system_of_equations (x y : ℝ) (h1 : 3 * x - 2 * y = 1) (h2 : x + y = 2) : x^2 - 2 * y^2 = -1 :=
by
  sorry

end solve_system_of_equations_l142_142512


namespace sum_reciprocals_factors_12_l142_142406

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142406


namespace sum_reciprocals_factors_12_l142_142257

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l142_142257


namespace cost_of_candy_l142_142686

theorem cost_of_candy (initial_amount pencil_cost remaining_after_candy : ℕ) 
  (h1 : initial_amount = 43) 
  (h2 : pencil_cost = 20) 
  (h3 : remaining_after_candy = 18) :
  ∃ candy_cost : ℕ, candy_cost = initial_amount - pencil_cost - remaining_after_candy :=
by
  sorry

end cost_of_candy_l142_142686


namespace completing_square_transformation_l142_142450

theorem completing_square_transformation (x : ℝ) :
  x^2 - 2 * x - 5 = 0 -> (x - 1)^2 = 6 :=
by {
  sorry -- Proof to be completed
}

end completing_square_transformation_l142_142450


namespace sum_reciprocals_factors_of_12_l142_142280

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l142_142280


namespace Jorge_goals_total_l142_142683

theorem Jorge_goals_total : 
  let last_season_goals := 156
  let this_season_goals := 187
  last_season_goals + this_season_goals = 343 := 
by
  sorry

end Jorge_goals_total_l142_142683


namespace find_multiple_l142_142049

theorem find_multiple (x y m : ℕ) (h1 : y + x = 50) (h2 : y = m * x - 43) (h3 : y = 31) : m = 4 :=
by
  sorry

end find_multiple_l142_142049


namespace num_pass_students_is_85_l142_142662

theorem num_pass_students_is_85 (T P F : ℕ) (avg_all avg_pass avg_fail : ℕ) (weight_pass weight_fail : ℕ) 
  (h_total_students : T = 150)
  (h_avg_all : avg_all = 40)
  (h_avg_pass : avg_pass = 45)
  (h_avg_fail : avg_fail = 20)
  (h_weight_ratio : weight_pass = 3 ∧ weight_fail = 1)
  (h_total_marks : (weight_pass * avg_pass * P + weight_fail * avg_fail * F) / (weight_pass * P + weight_fail * F) = avg_all)
  (h_students_sum : P + F = T) :
  P = 85 :=
by
  sorry

end num_pass_students_is_85_l142_142662


namespace flowers_per_bouquet_l142_142789

theorem flowers_per_bouquet :
  let red_seeds := 125
  let yellow_seeds := 125
  let orange_seeds := 125
  let purple_seeds := 125
  let red_killed := 45
  let yellow_killed := 61
  let orange_killed := 30
  let purple_killed := 40
  let bouquets := 36
  let red_flowers := red_seeds - red_killed
  let yellow_flowers := yellow_seeds - yellow_killed
  let orange_flowers := orange_seeds - orange_killed
  let purple_flowers := purple_seeds - purple_killed
  let total_flowers := red_flowers + yellow_flowers + orange_flowers + purple_flowers
  let flowers_per_bouquet := total_flowers / bouquets
  flowers_per_bouquet = 9 :=
by
  sorry

end flowers_per_bouquet_l142_142789


namespace last_two_digits_of_sum_l142_142816

noncomputable def last_two_digits_sum_factorials : ℕ :=
  let fac : List ℕ := List.map (fun n => Nat.factorial (n * 3)) [1, 2, 3, 4, 5, 6, 7]
  fac.foldl (fun acc x => (acc + x) % 100) 0

theorem last_two_digits_of_sum : last_two_digits_sum_factorials = 6 :=
by
  sorry

end last_two_digits_of_sum_l142_142816


namespace horse_buying_problem_l142_142061

variable (x y z : ℚ)

theorem horse_buying_problem :
  (x + 1/2 * y + 1/2 * z = 12) →
  (y + 1/3 * x + 1/3 * z = 12) →
  (z + 1/4 * x + 1/4 * y = 12) →
  x = 60/17 ∧ y = 136/17 ∧ z = 156/17 :=
by
  sorry

end horse_buying_problem_l142_142061


namespace sum_reciprocal_factors_12_l142_142253

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l142_142253


namespace train_speed_l142_142601

-- Definitions of the given conditions
def platform_length : ℝ := 250
def train_length : ℝ := 470.06
def time_taken : ℝ := 36

-- Definition of the total distance covered
def total_distance := platform_length + train_length

-- The proof problem: Prove that the calculated speed is approximately 20.0017 m/s
theorem train_speed :
  (total_distance / time_taken) = 20.0017 :=
by
  -- The actual proof goes here, but for now we leave it as sorry
  sorry

end train_speed_l142_142601


namespace maria_earnings_l142_142961

def cost_of_brushes : ℕ := 20
def cost_of_canvas : ℕ := 3 * cost_of_brushes
def cost_per_liter_of_paint : ℕ := 8
def liters_of_paint : ℕ := 5
def cost_of_paint : ℕ := liters_of_paint * cost_per_liter_of_paint
def total_cost : ℕ := cost_of_brushes + cost_of_canvas + cost_of_paint
def selling_price : ℕ := 200

theorem maria_earnings : (selling_price - total_cost) = 80 := by
  sorry

end maria_earnings_l142_142961


namespace sum_reciprocals_factors_12_l142_142276

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l142_142276


namespace smallest_positive_integer_with_divisors_l142_142075

theorem smallest_positive_integer_with_divisors :
  ∃ n : ℕ, (∀ d : ℕ, d ∣ n → nat.odd d) ∧ (finset.filter nat.odd (finset.divisors n)).card = 8 ∧ 
           (∃ m : ℕ, ∀ d : ℕ, d ∣ m → nat.even d ∧ m = n → (finset.filter nat.even (finset.divisors m)).card = 16)
             → n = 756 :=
by
  sorry

end smallest_positive_integer_with_divisors_l142_142075


namespace sum_of_reciprocals_of_factors_of_12_l142_142295

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142295


namespace sum_reciprocals_factors_12_l142_142267

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l142_142267


namespace min_sixth_power_sin_cos_l142_142851

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l142_142851


namespace arithmetic_sequence_and_minimum_sum_l142_142915

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l142_142915


namespace pow_addition_l142_142594

theorem pow_addition : (-2 : ℤ)^2 + (2 : ℤ)^2 = 8 :=
by
  sorry

end pow_addition_l142_142594


namespace percentage_invalid_l142_142537

theorem percentage_invalid (total_votes valid_votes_A : ℕ) (percent_A : ℝ) (total_valid_votes : ℝ) (percent_invalid : ℝ) :
  total_votes = 560000 →
  valid_votes_A = 333200 →
  percent_A = 0.70 →
  (1 - percent_invalid / 100) * total_votes = total_valid_votes →
  percent_A * total_valid_votes = valid_votes_A →
  percent_invalid = 15 :=
by
  intros h_total_votes h_valid_votes_A h_percent_A h_total_valid_votes h_valid_poll_A
  sorry

end percentage_invalid_l142_142537


namespace cost_price_of_pots_l142_142858

variable (C : ℝ)

-- Define the conditions
def selling_price (C : ℝ) := 1.25 * C
def total_revenue (selling_price : ℝ) := 150 * selling_price

-- State the main proof goal
theorem cost_price_of_pots (h : total_revenue (selling_price C) = 450) : C = 2.4 := by
  sorry

end cost_price_of_pots_l142_142858


namespace sum_reciprocals_factors_12_l142_142363

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l142_142363


namespace minimum_value_exists_l142_142486

noncomputable def min_value (a b c : ℝ) : ℝ :=
  a / (3 * b^2) + b / (4 * c^3) + c / (5 * a^4)

theorem minimum_value_exists :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → abc = 1 → min_value a b c ≥ 1 :=
by
  sorry

end minimum_value_exists_l142_142486


namespace suitable_for_census_l142_142453

-- Definitions based on the conditions in a)
def survey_A := "The service life of a batch of batteries"
def survey_B := "The height of all classmates in the class"
def survey_C := "The content of preservatives in a batch of food"
def survey_D := "The favorite mathematician of elementary and middle school students in the city"

-- The main statement to prove
theorem suitable_for_census : survey_B = "The height of all classmates in the class" := by
  -- We assert that the height of all classmates is the suitable survey for a census based on given conditions
  sorry

end suitable_for_census_l142_142453


namespace sum_of_reciprocals_factors_12_l142_142307

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l142_142307


namespace sqrt_defined_iff_ge_neg1_l142_142010

theorem sqrt_defined_iff_ge_neg1 (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x + 1)) ↔ x ≥ -1 := by
  sorry

end sqrt_defined_iff_ge_neg1_l142_142010


namespace arithmetic_sequence_minimum_value_S_n_l142_142936

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l142_142936


namespace minimum_sum_PE_PC_l142_142890

noncomputable def point := (ℝ × ℝ)
noncomputable def length (p1 p2 : point) : ℝ := Real.sqrt (((p1.1 - p2.1)^2) + ((p1.2 - p2.2)^2))

theorem minimum_sum_PE_PC :
  let A : point := (0, 3)
  let B : point := (3, 3)
  let C : point := (3, 0)
  let D : point := (0, 0)
  ∃ P E : point, E.1 = 3 ∧ E.2 = 1 ∧ (∃ t : ℝ, t ≥ 0 ∧ t ≤ 3 ∧ P.1 = 3 - t ∧ P.2 = t) ∧
    (length P E + length P C = Real.sqrt 13) :=
by
  sorry

end minimum_sum_PE_PC_l142_142890


namespace arithmetic_sequence_ratio_l142_142144

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ (d : ℝ), ∀ n, a (n + 1) = a n + d

variable {a b : ℕ → ℝ}
variable {S T : ℕ → ℝ}

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n + 1) * (a 0 + a n) / 2

variable (S_eq_k_mul_n_plus_2 : ∀ n, S n = (n + 2) * (S 0 / (n + 2)))
variable (T_eq_k_mul_n_plus_1 : ∀ n, T n = (n + 1) * (T 0 / (n + 1)))

theorem arithmetic_sequence_ratio (h₁ : arithmetic_sequence a) (h₂ : arithmetic_sequence b)
  (h₃ : ∀ n, S n = sum_first_n_terms a n)
  (h₄ : ∀ n, T n = sum_first_n_terms b n)
  (h₅ : ∀ n, (S n) / (T n) = (n + 2) / (n + 1))
  : a 6 / b 8 = 13 / 16 := 
sorry

end arithmetic_sequence_ratio_l142_142144


namespace one_hundred_fifty_sixth_digit_is_five_l142_142583

def repeated_sequence := [0, 6, 0, 5, 1, 3]
def target_index := 156 - 1
def block_length := repeated_sequence.length

theorem one_hundred_fifty_sixth_digit_is_five :
  repeated_sequence[target_index % block_length] = 5 :=
by
  sorry

end one_hundred_fifty_sixth_digit_is_five_l142_142583


namespace smallest_integer_with_odd_and_even_divisors_l142_142074

theorem smallest_integer_with_odd_and_even_divisors :
  ∃ n : ℕ,
    0 < n ∧
    (∀ d : ℕ, d ∣ n → (odd d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 1}) ∨ (even d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 0})) ∧ 
    (↑8 = (∑ d in (finset.filter (λ d, odd d) (finset.divisors n)), 1)) ∧
    (↑16 = (∑ d in (finset.filter (λ d, even d) (finset.divisors n)), 1)) ∧
    (24 = (∑ d in (finset.divisors n), 1)) ∧
    n = 108 :=
begin
  sorry
end

end smallest_integer_with_odd_and_even_divisors_l142_142074


namespace a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l142_142514

theorem a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l142_142514


namespace find_four_digit_number_l142_142666

noncomputable def reverse_num (n : ℕ) : ℕ := -- assume definition to reverse digits
  sorry

theorem find_four_digit_number :
  ∃ (A : ℕ), 1000 ≤ A ∧ A ≤ 9999 ∧ reverse_num (9 * A) = A ∧ 9 * A = reverse_num A ∧ A = 1089 :=
sorry

end find_four_digit_number_l142_142666


namespace regular_triangular_prism_cosine_l142_142496

-- Define the regular triangular prism and its properties
structure RegularTriangularPrism :=
  (side : ℝ) -- the side length of the base and the lateral edge

-- Define the vertices of the prism
structure Vertices :=
  (A : ℝ × ℝ × ℝ) 
  (B : ℝ × ℝ × ℝ) 
  (C : ℝ × ℝ × ℝ)
  (A1 : ℝ × ℝ × ℝ)
  (B1 : ℝ × ℝ × ℝ)
  (C1 : ℝ × ℝ × ℝ)

-- Define the cosine calculation
def cos_angle (prism : RegularTriangularPrism) (v : Vertices) : ℝ := sorry

-- Prove that the cosine of the angle between diagonals AB1 and BC1 is 1/4
theorem regular_triangular_prism_cosine (prism : RegularTriangularPrism) (v : Vertices)
  : cos_angle prism v = 1 / 4 :=
sorry

end regular_triangular_prism_cosine_l142_142496


namespace arithmetic_sequence_and_minimum_sum_l142_142917

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l142_142917


namespace sum_reciprocals_12_l142_142240

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l142_142240


namespace twin_prime_probability_split_17_l142_142467

theorem twin_prime_probability_split_17 :
  let primes := {2, 3, 5, 7}
  let twin_prime (p q : ℕ) := (Nat.Prime p) ∧ (Nat.Prime q) ∧ (|p - q| = 2)
  let twin_prime_pairs := (primes.toFinset.pairCombinations.filter (fun pq => twin_prime pq.1 pq.2)).toFinset
  let total_combinations := primes.toFinset.pairCombinations.card
  let twin_prime_probability := twin_prime_pairs.card / total_combinations
  twin_prime_probability = 1/3 := 
by
  sorry

end twin_prime_probability_split_17_l142_142467


namespace total_toothpicks_needed_l142_142108

theorem total_toothpicks_needed (length width : ℕ) (hl : length = 50) (hw : width = 40) : 
  (length + 1) * width + (width + 1) * length = 4090 := 
by
  -- proof omitted, replace this line with actual proof
  sorry

end total_toothpicks_needed_l142_142108


namespace product_of_two_numbers_l142_142064

theorem product_of_two_numbers
  (x y : ℝ)
  (h_diff : x - y ≠ 0)
  (h1 : x + y = 5 * (x - y))
  (h2 : x * y = 15 * (x - y)) :
  x * y = 37.5 :=
by
  sorry

end product_of_two_numbers_l142_142064


namespace total_oranges_after_increase_l142_142185

theorem total_oranges_after_increase :
  let Mary := 122
  let Jason := 105
  let Tom := 85
  let Sarah := 134
  let increase_rate := 0.10
  let new_Mary := Mary + Mary * increase_rate
  let new_Jason := Jason + Jason * increase_rate
  let new_Tom := Tom + Tom * increase_rate
  let new_Sarah := Sarah + Sarah * increase_rate
  let total_new_oranges := new_Mary + new_Jason + new_Tom + new_Sarah
  Float.round total_new_oranges = 491 := 
by
  sorry

end total_oranges_after_increase_l142_142185


namespace option_B_is_correct_l142_142159

-- Definitions and Conditions
variable {Line : Type} {Plane : Type}
variable (m n : Line) (α β γ : Plane)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Conditions
axiom m_perp_β : perpendicular m β
axiom m_parallel_α : parallel m α

-- Statement to prove
theorem option_B_is_correct : perpendicular_planes α β :=
by
  sorry

end option_B_is_correct_l142_142159


namespace min_sin6_cos6_l142_142839

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l142_142839


namespace arithmetic_sequence_min_value_S_l142_142905

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l142_142905


namespace larger_integer_is_21_l142_142755

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l142_142755


namespace product_of_sequence_l142_142016

theorem product_of_sequence : 
  (∃ (a : ℕ → ℚ), (a 1 * a 2 * a 3 * a 4 * a 5 = -32) ∧ 
  ((∀ n : ℕ, 3 * a (n + 1) + a n = 0) ∧ a 2 = 6)) :=
sorry

end product_of_sequence_l142_142016


namespace tim_movie_marathon_duration_l142_142992

-- Define the durations of each movie
def first_movie_duration : ℕ := 2

def second_movie_duration : ℕ := 
  first_movie_duration + (first_movie_duration / 2)

def combined_first_two_movies_duration : ℕ :=
  first_movie_duration + second_movie_duration

def last_movie_duration : ℕ := 
  combined_first_two_movies_duration - 1

-- Define the total movie marathon duration
def total_movie_marathon_duration : ℕ := 
  first_movie_duration + second_movie_duration + last_movie_duration

-- Problem statement to be proved
theorem tim_movie_marathon_duration : total_movie_marathon_duration = 9 := by
  sorry

end tim_movie_marathon_duration_l142_142992


namespace part1_part2_l142_142148

/-- Part (1) -/
theorem part1 (a : ℝ) (p : ∀ x : ℝ, x^2 - a*x + 4 > 0) (q : ∀ x y : ℝ, (0 < x ∧ x < y) → x^a < y^a) : 
  0 < a ∧ a < 4 :=
sorry

/-- Part (2) -/
theorem part2 (a : ℝ) (p_iff: ∀ x : ℝ, x^2 - a*x + 4 > 0 ↔ -4 < a ∧ a < 4)
  (q_iff: ∀ x y : ℝ, (0 < x ∧ x < y) ↔ x^a < y^a ∧ a > 0) (hp : ∃ x : ℝ, ¬(x^2 - a*x + 4 > 0))
  (hq : ∀ x y : ℝ, (x^a < y^a) → (0 < x ∧ x < y)) : 
  (a >= 4) ∨ (-4 < a ∧ a <= 0) :=
sorry

end part1_part2_l142_142148


namespace movie_marathon_duration_l142_142995

theorem movie_marathon_duration :
  let first_movie := 2
  let second_movie := first_movie + 0.5 * first_movie
  let combined_time := first_movie + second_movie
  let third_movie := combined_time - 1
  first_movie + second_movie + third_movie = 9 := by
  sorry

end movie_marathon_duration_l142_142995


namespace sum_of_reciprocals_of_factors_of_12_l142_142441

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142441


namespace ratio_m_of_q_l142_142003

theorem ratio_m_of_q
  (m n p q : ℚ)
  (h1 : m / n = 18)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 12) :
  m / q = 3 / 4 := 
sorry

end ratio_m_of_q_l142_142003


namespace min_value_sin6_cos6_l142_142842

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l142_142842


namespace grape_juice_amount_l142_142463

theorem grape_juice_amount 
  (T : ℝ) -- total amount of the drink 
  (orange_juice_percentage watermelon_juice_percentage : ℝ) -- percentages 
  (combined_amount_of_oj_wj : ℝ) -- combined amount of orange and watermelon juice 
  (h1 : orange_juice_percentage = 0.15)
  (h2 : watermelon_juice_percentage = 0.60)
  (h3 : combined_amount_of_oj_wj = 120)
  (h4 : combined_amount_of_oj_wj = (orange_juice_percentage + watermelon_juice_percentage) * T) : 
  (T * (1 - (orange_juice_percentage + watermelon_juice_percentage)) = 40) := 
sorry

end grape_juice_amount_l142_142463


namespace digit_156_of_fraction_47_over_777_is_9_l142_142582

theorem digit_156_of_fraction_47_over_777_is_9 :
  let r := 47 / 777 in
  let decimal_expansion := 0.0 * 10^0 + 6 * 10^(-1) + 0 * 10^(-2) + 4 * 10^(-3) + 5 * 10^(-4) + 9 * 10^(-5) + -- and so on, repeating every 5 digits as "60459"
  (r = 0 + 6 * 10^(-1) + 0 * 10^(-2) + 4 * 10^(-3) + 5 * 10^(-4) + 9 * 10^(-5)) ∧ -- and so on
  let d := 156 in
  decimal_expansion.nth_digit(d) = 9 :=
sorry

end digit_156_of_fraction_47_over_777_is_9_l142_142582


namespace proportion_of_solution_r_l142_142062

def fill_rate (a b c : ℕ) : ℚ := (1 / a : ℚ) + (1 / b : ℚ) + (1 / c : ℚ)

def proportion_solution_r (a b c : ℕ) (time_elapsed : ℚ) : ℚ :=
  let total_filled := fill_rate a b c * time_elapsed
  let r_amount := (1 / c) * time_elapsed
  r_amount / total_filled

theorem proportion_of_solution_r (a b c : ℕ) (time_elapsed : ℚ) (ha : a = 20) (hb : b = 20) (hc : c = 30) (ht : time_elapsed = 3) :
  proportion_solution_r a b c time_elapsed = 1 / 4 :=
by
  sorry

end proportion_of_solution_r_l142_142062


namespace sum_of_reciprocals_of_factors_of_12_l142_142440

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142440


namespace selection_methods_including_both_boys_and_girls_l142_142971

def boys : ℕ := 4
def girls : ℕ := 3
def total_people : ℕ := boys + girls
def select : ℕ := 4

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_methods_including_both_boys_and_girls :
  combination 7 4 - combination boys 4 = 34 :=
by
  sorry

end selection_methods_including_both_boys_and_girls_l142_142971


namespace population_increase_rate_is_20_percent_l142_142982

noncomputable def population_increase_rate 
  (initial_population final_population : ℕ) : ℕ :=
  ((final_population - initial_population) * 100) / initial_population

theorem population_increase_rate_is_20_percent :
  population_increase_rate 2000 2400 = 20 :=
by
  unfold population_increase_rate
  sorry

end population_increase_rate_is_20_percent_l142_142982


namespace cards_given_to_Jeff_l142_142689

theorem cards_given_to_Jeff
  (initial_cards : ℕ)
  (cards_given_to_John : ℕ)
  (remaining_cards : ℕ)
  (cards_left : ℕ)
  (h_initial : initial_cards = 573)
  (h_given_John : cards_given_to_John = 195)
  (h_left_before_Jeff : remaining_cards = initial_cards - cards_given_to_John)
  (h_final : cards_left = 210)
  (h_given_Jeff : remaining_cards - cards_left = 168) :
  (initial_cards - cards_given_to_John - cards_left = 168) :=
by
  sorry

end cards_given_to_Jeff_l142_142689


namespace sum_reciprocals_of_factors_12_l142_142376

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l142_142376


namespace even_numbers_average_19_l142_142612

theorem even_numbers_average_19 (n : ℕ) (h1 : (n / 2) * (2 + 2 * n) / n = 19) : n = 18 :=
by {
  sorry
}

end even_numbers_average_19_l142_142612


namespace smallest_integer_has_8_odd_and_16_even_divisors_l142_142079

/-!
  Prove that the smallest positive integer with exactly 8 positive odd integer divisors
  and exactly 16 positive even integer divisors is 540.
-/
def smallest_integer_with_divisors : ℕ :=
  540

theorem smallest_integer_has_8_odd_and_16_even_divisors 
  (n : ℕ) 
  (h1 : (8 : ℕ) = nat.count (λ d, d % 2 = 1) (nat.divisors n))
  (h2 : (16 : ℕ) = nat.count (λ d, d % 2 = 0) (nat.divisors n)) :
  n = 540 :=
sorry

end smallest_integer_has_8_odd_and_16_even_divisors_l142_142079


namespace sin_negative_300_eq_l142_142443

theorem sin_negative_300_eq : Real.sin (-(300 * Real.pi / 180)) = Real.sqrt 3 / 2 :=
by
  -- Periodic property of sine function: sin(theta) = sin(theta + 360 * n)
  have periodic_property : ∀ θ n : ℤ, Real.sin θ = Real.sin (θ + n * 2 * Real.pi) :=
    by sorry
  -- Known value: sin(60 degrees) = sqrt(3)/2
  have sin_60 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  -- Apply periodic_property to transform sin(-300 degrees) to sin(60 degrees)
  sorry

end sin_negative_300_eq_l142_142443


namespace isosceles_triangle_perimeter_l142_142808

-- Define what it means to be a root of the equation x^2 - 5x + 6 = 0
def is_root (x : ℝ) : Prop := x^2 - 5 * x + 6 = 0

-- Define the perimeter based on given conditions
theorem isosceles_triangle_perimeter (x : ℝ) (base : ℝ) (h_base : base = 4) (h_root : is_root x) :
    2 * x + base = 10 :=
by
  -- Insert proof here
  sorry

end isosceles_triangle_perimeter_l142_142808


namespace sum_reciprocals_of_factors_of_12_l142_142324

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l142_142324


namespace sum_reciprocals_factors_12_l142_142351

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l142_142351


namespace find_n_l142_142581

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 103) (h3 : 100 * n ≡ 85 [MOD 103]) : n = 6 := 
sorry

end find_n_l142_142581


namespace digit_d_for_5678d_is_multiple_of_9_l142_142624

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem digit_d_for_5678d_is_multiple_of_9 : 
  ∃ d : ℕ, d < 10 ∧ is_multiple_of_9 (56780 + d) ∧ d = 1 :=
by
  sorry

end digit_d_for_5678d_is_multiple_of_9_l142_142624


namespace mrs_hilt_money_left_l142_142187

theorem mrs_hilt_money_left (initial_money : ℕ) (cost_of_pencil : ℕ) (money_left : ℕ) (h1 : initial_money = 15) (h2 : cost_of_pencil = 11) : money_left = 4 :=
by
  sorry

end mrs_hilt_money_left_l142_142187


namespace determinant_problem_l142_142140

theorem determinant_problem 
  (x y z w : ℝ) 
  (h : x * w - y * z = 7) : 
  ((x * (8 * z + 4 * w)) - (z * (8 * x + 4 * y))) = 28 :=
by 
  sorry

end determinant_problem_l142_142140


namespace geometric_progression_product_sum_sumrecip_l142_142002

theorem geometric_progression_product_sum_sumrecip (P S S' : ℝ) (n : ℕ)
  (hP : P = a ^ n * r ^ ((n * (n - 1)) / 2))
  (hS : S = a * (1 - r ^ n) / (1 - r))
  (hS' : S' = (r ^ n - 1) / (a * (r - 1))) :
  P = (S / S') ^ (1 / 2 * n) :=
  sorry

end geometric_progression_product_sum_sumrecip_l142_142002


namespace viola_final_jump_l142_142067

variable (n : ℕ) (T : ℝ) (x : ℝ)

theorem viola_final_jump (h1 : T = 3.80 * n)
                        (h2 : (T + 3.99) / (n + 1) = 3.81)
                        (h3 : T + 3.99 + x = 3.82 * (n + 2)) : 
                        x = 4.01 :=
sorry

end viola_final_jump_l142_142067


namespace sum_reciprocals_factors_12_l142_142258

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l142_142258


namespace g_g_3_eq_3606651_l142_142878

def g (x: ℤ) : ℤ := 3 * x^3 + 3 * x^2 - x + 1

theorem g_g_3_eq_3606651 : g (g 3) = 3606651 := 
by {
  sorry
}

end g_g_3_eq_3606651_l142_142878


namespace length_of_second_train_l142_142103

/-- 
The length of the second train can be determined given the length and speed of the first train,
the speed of the second train, and the time they take to cross each other.
-/
theorem length_of_second_train (speed1_kmph : ℝ) (length1_m : ℝ) (speed2_kmph : ℝ) (time_s : ℝ) :
  (speed1_kmph = 120) →
  (length1_m = 230) →
  (speed2_kmph = 80) →
  (time_s = 9) →
  let relative_speed_m_per_s := (speed1_kmph * 1000 / 3600) + (speed2_kmph * 1000 / 3600)
  let total_distance := relative_speed_m_per_s * time_s
  let length2_m := total_distance - length1_m
  length2_m = 269.95 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  let relative_speed_m_per_s := (120 * 1000 / 3600) + (80 * 1000 / 3600)
  let total_distance := relative_speed_m_per_s * 9
  let length2_m := total_distance - 230
  exact sorry

end length_of_second_train_l142_142103


namespace right_triangle_no_k_values_l142_142211

theorem right_triangle_no_k_values (k : ℕ) (h : k > 0) : 
  ¬ (∃ k, k > 0 ∧ ((17 > k ∧ 17^2 = 13^2 + k^2) ∨ (k > 17 ∧ k < 30 ∧ k^2 = 13^2 + 17^2))) :=
sorry

end right_triangle_no_k_values_l142_142211


namespace bobArrivesBefore845Prob_l142_142112

noncomputable def probabilityBobBefore845 (totalTime: ℕ) (cutoffTime: ℕ) : ℚ :=
  let totalArea := (totalTime * totalTime) / 2
  let areaOfInterest := (cutoffTime * cutoffTime) / 2
  (areaOfInterest : ℚ) / totalArea

theorem bobArrivesBefore845Prob (totalTime: ℕ) (cutoffTime: ℕ) (ht: totalTime = 60) (hc: cutoffTime = 45) :
  probabilityBobBefore845 totalTime cutoffTime = 9 / 16 := by
  sorry

end bobArrivesBefore845Prob_l142_142112


namespace minimize_sin_cos_six_l142_142827

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l142_142827


namespace larger_integer_value_l142_142720

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l142_142720


namespace arithmetic_sequence_min_value_S_l142_142950

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l142_142950


namespace sum_reciprocals_factors_12_l142_142348

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l142_142348


namespace larger_integer_is_21_l142_142730

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l142_142730


namespace arithmetic_sequence_min_value_Sn_l142_142948

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l142_142948


namespace factorization_correct_l142_142091

-- Defining the expressions
def expr1 (x : ℝ) : ℝ := 4 * x^2 + 4 * x
def expr2 (x : ℝ) : ℝ := 4 * x * (x + 1)

-- Theorem statement: Prove that expr1 and expr2 are equivalent
theorem factorization_correct (x : ℝ) : expr1 x = expr2 x :=
by 
  sorry

end factorization_correct_l142_142091


namespace correct_analytical_method_l142_142774

-- Definitions of the different reasoning methods
def reasoning_from_cause_to_effect : Prop := ∀ (cause effect : Prop), cause → effect
def reasoning_from_effect_to_cause : Prop := ∀ (cause effect : Prop), effect → cause
def distinguishing_and_mutually_inferring : Prop := ∀ (cause effect : Prop), (cause ↔ effect)
def proving_converse_statement : Prop := ∀ (P Q : Prop), (P → Q) → (Q → P)

-- Definition of the analytical method
def analytical_method : Prop := reasoning_from_effect_to_cause

-- Theorem stating that the analytical method is the method of reasoning from effect to cause
theorem correct_analytical_method : analytical_method = reasoning_from_effect_to_cause := 
by 
  -- Complete this proof with refined arguments
  sorry

end correct_analytical_method_l142_142774


namespace sum_of_reciprocals_of_factors_of_12_l142_142384

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142384


namespace sum_of_reciprocals_factors_12_l142_142390

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l142_142390


namespace minimize_wage_l142_142191

def totalWorkers : ℕ := 150
def wageA : ℕ := 2000
def wageB : ℕ := 3000

theorem minimize_wage : ∃ (a : ℕ), a = 50 ∧ (totalWorkers - a) ≥ 2 * a ∧ 
  (wageA * a + wageB * (totalWorkers - a) = 400000) := sorry

end minimize_wage_l142_142191


namespace larger_integer_is_21_l142_142752

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l142_142752


namespace ending_number_of_sequence_divisible_by_11_l142_142057

theorem ending_number_of_sequence_divisible_by_11 : 
  ∃ (n : ℕ), 19 < n ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 6 → n = 19 + 11 * k) ∧ n = 77 :=
by
  sorry

end ending_number_of_sequence_divisible_by_11_l142_142057


namespace sum_reciprocals_factors_12_l142_142275

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l142_142275


namespace sum_reciprocals_factors_12_l142_142335

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142335


namespace sum_of_reciprocals_factors_12_l142_142393

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l142_142393


namespace triangle_angle_relation_l142_142694

theorem triangle_angle_relation 
  (a b c : ℝ)
  (α β γ : ℝ)
  (h1 : b = (a + c) / Real.sqrt 2)
  (h2 : β = (α + γ) / 2)
  (h3 : c > a)
  : γ = α + 90 :=
sorry

end triangle_angle_relation_l142_142694


namespace percent_calculation_l142_142447

theorem percent_calculation (x : ℝ) : 
  (∃ y : ℝ, y / 100 * x = 0.3 * 0.7 * x) → ∃ y : ℝ, y = 21 :=
by
  sorry

end percent_calculation_l142_142447


namespace ratio_pentagon_area_l142_142817

noncomputable def square_side_length := 1
noncomputable def square_area := (square_side_length : ℝ)^2
noncomputable def total_area := 3 * square_area
noncomputable def area_triangle (base height : ℝ) := 0.5 * base * height
noncomputable def GC := 2 / 3 * square_side_length
noncomputable def HD := 2 / 3 * square_side_length
noncomputable def area_GJC := area_triangle GC square_side_length
noncomputable def area_HDJ := area_triangle HD square_side_length
noncomputable def area_AJKCB := square_area - (area_GJC + area_HDJ)

theorem ratio_pentagon_area :
  (area_AJKCB / total_area) = 1 / 9 := 
sorry

end ratio_pentagon_area_l142_142817


namespace probability_at_least_two_balls_in_a_box_l142_142776

theorem probability_at_least_two_balls_in_a_box :
  ∀ (boxes: ℕ → ℕ) (balls: ℕ → ℕ), (∀ n, P(boxes(n)) = (1/2^n)) → P(∃ n, boxes(n) ≥ 2) = 5/7 :=
by
  sorry

end probability_at_least_two_balls_in_a_box_l142_142776


namespace sum_reciprocals_factors_12_l142_142407

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142407


namespace joan_seashells_l142_142021

variable (initialSeashells seashellsGiven remainingSeashells : ℕ)

theorem joan_seashells : initialSeashells = 79 ∧ seashellsGiven = 63 ∧ remainingSeashells = initialSeashells - seashellsGiven → remainingSeashells = 16 :=
by
  intros
  sorry

end joan_seashells_l142_142021


namespace Yoque_monthly_payment_l142_142454

theorem Yoque_monthly_payment :
  ∃ m : ℝ, m = 15 ∧ ∀ a t : ℝ, a = 150 ∧ t = 11 ∧ (a + 0.10 * a) / t = m :=
by
  sorry

end Yoque_monthly_payment_l142_142454


namespace sale_first_month_l142_142791

-- Declaration of all constant sales amounts in rupees
def sale_second_month : ℕ := 6927
def sale_third_month : ℕ := 6855
def sale_fourth_month : ℕ := 7230
def sale_fifth_month : ℕ := 6562
def sale_sixth_month : ℕ := 6791
def average_required : ℕ := 6800
def months : ℕ := 6

-- Total sales computed from the average sale requirement
def total_sales_needed : ℕ := months * average_required

-- The sum of sales for the second to sixth months
def total_sales_last_five_months := sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month

-- Prove the sales in the first month given the conditions
theorem sale_first_month :
  total_sales_needed - total_sales_last_five_months = 6435 :=
by
  sorry

end sale_first_month_l142_142791


namespace quadratic_coefficients_l142_142574

theorem quadratic_coefficients :
  ∃ a b c : ℤ, a = 4 ∧ b = 0 ∧ c = -3 ∧ 4 * x^2 = 3 := sorry

end quadratic_coefficients_l142_142574


namespace jessy_initial_reading_plan_l142_142020

theorem jessy_initial_reading_plan (x : ℕ) (h : (7 * (3 * x + 2) = 140)) : x = 6 :=
sorry

end jessy_initial_reading_plan_l142_142020


namespace prove_arithmetic_sequence_minimum_value_S_l142_142930

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l142_142930


namespace ratio_sqrt_2_l142_142173

theorem ratio_sqrt_2 {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : a^2 + b^2 = 6 * a * b) :
  (a + b) / (a - b) = Real.sqrt 2 :=
by
  sorry

end ratio_sqrt_2_l142_142173


namespace min_value_a4b3c2_l142_142549

theorem min_value_a4b3c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : 1/a + 1/b + 1/c = 9) : (∀ a b c : ℝ, a^4 * b^3 * c^2 ≥ 1/(9^9)) :=
by
  sorry

end min_value_a4b3c2_l142_142549


namespace digit_five_occurrences_l142_142780

variable (fives_ones fives_tens fives_hundreds : ℕ)

def count_fives := fives_ones + fives_tens + fives_hundreds

theorem digit_five_occurrences :
  ( ∀ (fives_ones fives_tens fives_hundreds : ℕ), 
    fives_ones = 100 ∧ fives_tens = 100 ∧ fives_hundreds = 100 → 
    count_fives fives_ones fives_tens fives_hundreds = 300 ) :=
by
  sorry

end digit_five_occurrences_l142_142780


namespace range_m_if_neg_p_implies_neg_q_range_x_if_m_is_5_and_p_or_q_true_p_and_q_false_l142_142511

-- Question 1
def prop_p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def prop_q (x m : ℝ) : Prop := 1 - m ≤ x + 1 ∧ x + 1 < 1 + m ∧ m > 0
def neg_p (x : ℝ) : Prop := ¬ prop_p x
def neg_q (x m : ℝ) : Prop := ¬ prop_q x m

theorem range_m_if_neg_p_implies_neg_q : 
  (∀ x, neg_p x → neg_q x m) → 0 < m ∧ m ≤ 1 :=
by
  sorry

-- Question 2
theorem range_x_if_m_is_5_and_p_or_q_true_p_and_q_false : 
  (∀ x, (prop_p x ∨ prop_q x 5) ∧ ¬ (prop_p x ∧ prop_q x 5)) → 
  ∀ x, (x = 5 ∨ (-5 ≤ x ∧ x < -1)) :=
by
  sorry

end range_m_if_neg_p_implies_neg_q_range_x_if_m_is_5_and_p_or_q_true_p_and_q_false_l142_142511


namespace jellybean_probability_l142_142458

theorem jellybean_probability :
  let total_ways := Nat.choose 15 4
  let red_ways := Nat.choose 5 2
  let blue_ways := Nat.choose 3 2
  let favorable_ways := red_ways * blue_ways
  let probability := favorable_ways / total_ways
  probability = (2 : ℚ) / 91 := by
  sorry

end jellybean_probability_l142_142458


namespace sum_reciprocals_factors_of_12_l142_142281

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l142_142281


namespace gain_percent_l142_142106

theorem gain_percent (cp sp : ℝ) (h_cp : cp = 900) (h_sp : sp = 1080) :
    ((sp - cp) / cp) * 100 = 20 :=
by
    sorry

end gain_percent_l142_142106


namespace pencils_given_out_l142_142215
-- Define the problem conditions
def students : ℕ := 96
def dozens_per_student : ℕ := 7
def pencils_per_dozen : ℕ := 12

-- Define the expected total pencils
def expected_pencils : ℕ := 8064

-- Define the statement to be proven
theorem pencils_given_out : (students * (dozens_per_student * pencils_per_dozen)) = expected_pencils := 
  by
  sorry

end pencils_given_out_l142_142215


namespace dons_profit_l142_142175

-- Definitions from the conditions
def bundles_jamie_bought := 20
def bundles_jamie_sold := 15
def profit_jamie := 60

def bundles_linda_bought := 34
def bundles_linda_sold := 24
def profit_linda := 69

def bundles_don_bought := 40
def bundles_don_sold := 36

-- Variables representing the unknown prices
variables (b s : ℝ)

-- Conditions written as equalities
axiom eq_jamie : bundles_jamie_sold * s - bundles_jamie_bought * b = profit_jamie
axiom eq_linda : bundles_linda_sold * s - bundles_linda_bought * b = profit_linda

-- Statement to prove Don's profit
theorem dons_profit : bundles_don_sold * s - bundles_don_bought * b = 252 :=
by {
  sorry -- proof goes here
}

end dons_profit_l142_142175


namespace sum_of_reciprocals_factors_12_l142_142430

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l142_142430


namespace max_points_of_intersection_l142_142224

theorem max_points_of_intersection (circles : Fin 2 → Circle) (lines : Fin 3 → Line) :
  number_of_intersections circles lines = 17 :=
sorry

end max_points_of_intersection_l142_142224


namespace sum_of_reciprocals_of_factors_of_12_l142_142380

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142380


namespace cubic_expression_l142_142006

theorem cubic_expression (a b c : ℝ) (h1 : a + b + c = 13) (h2 : ab + ac + bc = 30) : a^3 + b^3 + c^3 - 3 * abc = 1027 :=
sorry

end cubic_expression_l142_142006


namespace sum_reciprocal_factors_of_12_l142_142414

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l142_142414


namespace smallest_digit_N_divisible_by_6_l142_142071

theorem smallest_digit_N_divisible_by_6 : 
  ∃ N : ℕ, N < 10 ∧ 
          (14530 + N) % 6 = 0 ∧
          ∀ M : ℕ, M < N → (14530 + M) % 6 ≠ 0 := sorry

end smallest_digit_N_divisible_by_6_l142_142071


namespace find_k_l142_142018

open Classical

theorem find_k 
    (z x y k : ℝ) 
    (k_pos_int : k > 0 ∧ ∃ n : ℕ, k = n)
    (prop1 : z - y = k * x)
    (prop2 : x - z = k * y)
    (cond : z = (5 / 3) * (x - y)) :
    k = 3 :=
by
  sorry

end find_k_l142_142018


namespace sum_reciprocals_12_l142_142238

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l142_142238


namespace sum_reciprocal_factors_of_12_l142_142413

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l142_142413


namespace sum_reciprocals_12_l142_142241

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l142_142241


namespace non_neg_int_solutions_m_value_integer_values_of_m_l142_142527

-- 1. Non-negative integer solutions of x + 2y = 3
theorem non_neg_int_solutions (x y : ℕ) :
  x + 2 * y = 3 ↔ (x = 3 ∧ y = 0) ∨ (x = 1 ∧ y = 1) :=
sorry

-- 2. If (x, y) = (1, 1) satisfies both x + 2y = 3 and x + y = 2, then m = -4
theorem m_value (m : ℝ) :
  (1 + 2 * 1 = 3) ∧ (1 + 1 = 2) ∧ (1 - 2 * 1 + m * 1 = -5) → m = -4 :=
sorry

-- 3. Given n = 3, integer values of m are -2 or 0
theorem integer_values_of_m (m : ℤ) :
  ∃ x y : ℤ, 3 * x + 4 * y = 5 ∧ x - 2 * y + m * x = -5 → m = -2 ∨ m = 0 :=
sorry

end non_neg_int_solutions_m_value_integer_values_of_m_l142_142527


namespace coeff_x3_in_expansion_l142_142495

theorem coeff_x3_in_expansion : 
  ∃ c : ℕ, (c = 80) ∧ (∃ r : ℕ, r = 1 ∧ (2 * x + 1 / x) ^ 5 = (2 * x) ^ (5 - r) * (1 / x) ^ r)
:= sorry

end coeff_x3_in_expansion_l142_142495


namespace susan_change_sum_susan_possible_sums_l142_142562

theorem susan_change_sum
  (change : ℕ)
  (h_lt_100 : change < 100)
  (h_nickels : ∃ k : ℕ, change = 5 * k + 2)
  (h_quarters : ∃ m : ℕ, change = 25 * m + 5) :
  change = 30 ∨ change = 55 ∨ change = 80 :=
sorry

theorem susan_possible_sums :
  30 + 55 + 80 = 165 :=
by norm_num

end susan_change_sum_susan_possible_sums_l142_142562


namespace parallel_vectors_have_proportional_direction_ratios_l142_142875

theorem parallel_vectors_have_proportional_direction_ratios (m : ℝ) :
  let a := (1, 2)
  let b := (m, 1)
  (a.1 / b.1) = (a.2 / b.2) → m = 1/2 :=
by
  let a := (1, 2)
  let b := (m, 1)
  intro h
  sorry

end parallel_vectors_have_proportional_direction_ratios_l142_142875


namespace mogs_and_mags_to_migs_l142_142034

theorem mogs_and_mags_to_migs:
  (∀ mags migs, 1 * mags = 8 * migs) ∧ 
  (∀ mogs mags, 1 * mogs = 6 * mags) → 
  10 * (6 * 8) + 6 * 8 = 528 := by 
  sorry

end mogs_and_mags_to_migs_l142_142034


namespace sum_reciprocals_factors_of_12_l142_142283

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l142_142283


namespace _l142_142863

noncomputable def tan_alpha_theorem (α : ℝ) (h1 : Real.tan (Real.pi / 4 + α) = 2) : Real.tan α = 1 / 3 :=
by
  sorry

noncomputable def evaluate_expression_theorem (α β : ℝ) 
  (h1 : Real.tan (Real.pi / 4 + α) = 2) 
  (h2 : Real.tan β = 1 / 2) 
  (h3 : Real.tan α = 1 / 3) : 
  (Real.sin (α + β) - 2 * Real.sin α * Real.cos β) / (2 * Real.sin α * Real.sin β + Real.cos (α + β)) = 1 / 7 :=
by
  sorry

end _l142_142863


namespace chord_length_of_tangent_l142_142702

theorem chord_length_of_tangent (R r : ℝ) (h : R^2 - r^2 = 25) : ∃ c : ℝ, c = 10 :=
by
  sorry

end chord_length_of_tangent_l142_142702


namespace sum_of_reciprocals_factors_12_l142_142310

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l142_142310


namespace angle_in_third_quadrant_half_l142_142158

theorem angle_in_third_quadrant_half {
  k : ℤ 
} (h1: (k * 360 + 180) < α) (h2 : α < k * 360 + 270) :
  (k * 180 + 90) < (α / 2) ∧ (α / 2) < (k * 180 + 135) :=
sorry

end angle_in_third_quadrant_half_l142_142158


namespace exists_positive_integer_pow_not_integer_l142_142974

theorem exists_positive_integer_pow_not_integer
  (α β : ℝ)
  (hαβ : α ≠ β)
  (h_non_int : ¬(↑⌊α⌋ = α ∧ ↑⌊β⌋ = β)) :
  ∃ n : ℕ, 0 < n ∧ ¬∃ k : ℤ, α^n - β^n = k :=
by
  sorry

end exists_positive_integer_pow_not_integer_l142_142974


namespace first_place_beat_joe_l142_142677

theorem first_place_beat_joe (joe_won joe_draw first_place_won first_place_draw points_win points_draw : ℕ) 
    (h1 : joe_won = 1) (h2 : joe_draw = 3) (h3 : first_place_won = 2) (h4 : first_place_draw = 2)
    (h5 : points_win = 3) (h6 : points_draw = 1) : 
    (first_place_won * points_win + first_place_draw * points_draw) - (joe_won * points_win + joe_draw * points_draw) = 2 :=
by
   sorry

end first_place_beat_joe_l142_142677


namespace solve_system_of_equations_l142_142973

theorem solve_system_of_equations (x y : ℝ) (h1 : x - y = -5) (h2 : 3 * x + 2 * y = 10) : x = 0 ∧ y = 5 := by
  sorry

end solve_system_of_equations_l142_142973


namespace sin_cos_sixth_min_l142_142854

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l142_142854


namespace smallest_diff_PR_PQ_l142_142218

theorem smallest_diff_PR_PQ (PQ PR QR : ℤ) (h1 : PQ < PR) (h2 : PR ≤ QR) (h3 : PQ + PR + QR = 2021) : 
  ∃ PQ PR QR : ℤ, PQ < PR ∧ PR ≤ QR ∧ PQ + PR + QR = 2021 ∧ PR - PQ = 1 :=
by
  sorry

end smallest_diff_PR_PQ_l142_142218


namespace trucks_transport_l142_142988

variables {x y : ℝ}

theorem trucks_transport (h1 : 2 * x + 3 * y = 15.5)
                         (h2 : 5 * x + 6 * y = 35) :
  3 * x + 2 * y = 17 :=
sorry

end trucks_transport_l142_142988


namespace sum_reciprocals_of_factors_of_12_l142_142327

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l142_142327


namespace lines_intersect_at_point_l142_142602

def ParametricLine1 (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 4 - 3 * t)

def ParametricLine2 (u : ℝ) : ℝ × ℝ :=
  (-2 + 3 * u, 5 - u)

theorem lines_intersect_at_point :
  ∃ t u : ℝ, ParametricLine1 t = ParametricLine2 u ∧ ParametricLine1 t = (-5, 13) :=
by
  sorry

end lines_intersect_at_point_l142_142602


namespace students_multiple_activities_l142_142888

theorem students_multiple_activities (total_students only_debate only_singing only_dance no_activities students_more_than_one : ℕ) 
  (h1 : total_students = 55) 
  (h2 : only_debate = 10) 
  (h3 : only_singing = 18) 
  (h4 : only_dance = 8)
  (h5 : no_activities = 5)
  (h6 : students_more_than_one = total_students - (only_debate + only_singing + only_dance + no_activities)) :
  students_more_than_one = 14 := by
  sorry

end students_multiple_activities_l142_142888


namespace minimum_road_length_l142_142967

/-- Define the grid points A, B, and C with their coordinates. -/
def A : ℤ × ℤ := (0, 0)
def B : ℤ × ℤ := (3, 2)
def C : ℤ × ℤ := (4, 3)

/-- Define the side length of each grid square in meters. -/
def side_length : ℕ := 100

/-- Calculate the Manhattan distance between two points on the grid. -/
def manhattan_distance (p q : ℤ × ℤ) : ℕ :=
  (Int.natAbs (p.1 - q.1) + Int.natAbs (p.2 - q.2)) * side_length

/-- Statement: The minimum total length of the roads (in meters) to connect A, B, and C is 1000 meters. -/
theorem minimum_road_length : manhattan_distance A B + manhattan_distance B C + manhattan_distance C A = 1000 := by
  sorry

end minimum_road_length_l142_142967


namespace product_of_binomials_l142_142129

theorem product_of_binomials (x : ℝ) : 
  (4 * x - 3) * (2 * x + 7) = 8 * x^2 + 22 * x - 21 := by
  sorry

end product_of_binomials_l142_142129


namespace sum_of_reciprocals_of_factors_of_12_l142_142437

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142437


namespace sum_of_reciprocals_factors_12_l142_142309

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l142_142309


namespace algebraic_expression_value_l142_142508

-- Define the equation and its roots.
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 = 0

def is_root (x : ℝ) : Prop := quadratic_eq x

-- The main theorem.
theorem algebraic_expression_value (x1 x2 : ℝ) (h1 : is_root x1) (h2 : is_root x2) :
  (x1 + x2) / (1 + x1 * x2) = 1 :=
sorry

end algebraic_expression_value_l142_142508


namespace right_triangle_exists_l142_142901

theorem right_triangle_exists (a b c d : ℕ) (h1 : ab = cd) (h2 : a + b = c - d) : 
  ∃ (x y z : ℕ), x * y / 2 = ab ∧ x^2 + y^2 = z^2 :=
sorry

end right_triangle_exists_l142_142901


namespace arithmetic_sequence_minimum_value_S_l142_142928

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l142_142928


namespace tim_movie_marathon_duration_l142_142991

-- Define the durations of each movie
def first_movie_duration : ℕ := 2

def second_movie_duration : ℕ := 
  first_movie_duration + (first_movie_duration / 2)

def combined_first_two_movies_duration : ℕ :=
  first_movie_duration + second_movie_duration

def last_movie_duration : ℕ := 
  combined_first_two_movies_duration - 1

-- Define the total movie marathon duration
def total_movie_marathon_duration : ℕ := 
  first_movie_duration + second_movie_duration + last_movie_duration

-- Problem statement to be proved
theorem tim_movie_marathon_duration : total_movie_marathon_duration = 9 := by
  sorry

end tim_movie_marathon_duration_l142_142991


namespace part1_part2_l142_142507

noncomputable def f (x : ℝ) := |x - 3| + |x - 4|

theorem part1 (a : ℝ) (h : ∃ x : ℝ, f x < a) : a > 1 :=
sorry

theorem part2 (x : ℝ) : f x ≥ 7 + 7 * x - x ^ 2 ↔ x ≤ 0 ∨ 7 ≤ x :=
sorry

end part1_part2_l142_142507


namespace min_value_fraction_sum_l142_142166

theorem min_value_fraction_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (1 / a + 4 / b) ≥ 9 :=
begin
  sorry
end

end min_value_fraction_sum_l142_142166


namespace proof_problem_l142_142517

variable {a b : ℝ}
variable (cond : sqrt a > sqrt b)

theorem proof_problem (h1 : a > b) (h2 : 0 ≤ a) (h3 : 0 ≤ b) :
  (a^2 > b^2) ∧
  ((b + 1) / (a + 1) > b / a) ∧
  (b + 1 / (b + 1) ≥ 1) :=
by
  sorry

end proof_problem_l142_142517


namespace tim_movie_marathon_duration_is_9_l142_142998

-- Define the conditions:
def first_movie_duration : ℕ := 2
def second_movie_duration : ℕ := first_movie_duration + (first_movie_duration / 2)
def combined_duration_first_two_movies : ℕ := first_movie_duration + second_movie_duration
def third_movie_duration : ℕ := combined_duration_first_two_movies - 1
def total_marathon_duration : ℕ := first_movie_duration + second_movie_duration + third_movie_duration

-- The theorem to prove the marathon duration is 9 hours
theorem tim_movie_marathon_duration_is_9 :
  total_marathon_duration = 9 :=
by sorry

end tim_movie_marathon_duration_is_9_l142_142998


namespace max_c_in_range_f_l142_142620

theorem max_c_in_range_f (c : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + c = 2) ↔ c ≤ 11 :=
begin
  sorry
end

end max_c_in_range_f_l142_142620


namespace Mrs_Martin_pays_32_l142_142701

def kiddie_scoop_cost : ℕ := 3
def regular_scoop_cost : ℕ := 4
def double_scoop_cost : ℕ := 6

def num_regular_scoops : ℕ := 2
def num_kiddie_scoops : ℕ := 2
def num_double_scoops : ℕ := 3

def total_cost : ℕ := 
  (num_regular_scoops * regular_scoop_cost) + 
  (num_kiddie_scoops * kiddie_scoop_cost) + 
  (num_double_scoops * double_scoop_cost)

theorem Mrs_Martin_pays_32 :
  total_cost = 32 :=
by
  sorry

end Mrs_Martin_pays_32_l142_142701


namespace log_diff_eq_35_l142_142534

theorem log_diff_eq_35 {a b : ℝ} (h₁ : a > b) (h₂ : b > 1)
  (h₃ : (1 / Real.log a / Real.log b) + (1 / (Real.log b / Real.log a)) = Real.sqrt 1229) :
  (1 / (Real.log b / Real.log (a * b))) - (1 / (Real.log a / Real.log (a * b))) = 35 :=
sorry

end log_diff_eq_35_l142_142534


namespace sum_reciprocals_factors_12_l142_142404

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142404


namespace sqrt_30_estimate_l142_142489

theorem sqrt_30_estimate : 5 < Real.sqrt 30 ∧ Real.sqrt 30 < 6 := by
  sorry

end sqrt_30_estimate_l142_142489


namespace sum_reciprocals_factors_12_l142_142274

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l142_142274


namespace sum_reciprocal_factors_of_12_l142_142416

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l142_142416


namespace point_in_fourth_quadrant_l142_142631

theorem point_in_fourth_quadrant (m n : ℝ) (h₁ : m < 0) (h₂ : n > 0) : 
  2 * n - m > 0 ∧ -n + m < 0 := by
  sorry

end point_in_fourth_quadrant_l142_142631


namespace larger_integer_is_21_l142_142732

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l142_142732


namespace samantha_mean_correct_l142_142697

-- Given data: Samantha's assignment scores
def samantha_scores : List ℕ := [84, 89, 92, 88, 95, 91, 93]

-- Definition of the arithmetic mean of a list of scores
def arithmetic_mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / (scores.length : ℚ)

-- Prove that the arithmetic mean of Samantha's scores is 90.29
theorem samantha_mean_correct :
  arithmetic_mean samantha_scores = 90.29 := 
by
  -- The proof steps would be filled in here
  sorry

end samantha_mean_correct_l142_142697


namespace exercise_b_c_values_l142_142640

open Set

universe u

theorem exercise_b_c_values : 
  ∀ (b c : ℝ), let U : Set ℝ := {2, 3, 5}
               let A : Set ℝ := {x | x^2 + b * x + c = 0}
               (U \ A = {2}) → (b = -8 ∧ c = 15) :=
by
  intros b c U A H
  let U : Set ℝ := {2, 3, 5}
  let A : Set ℝ := {x | x^2 + b * x + c = 0}
  have H1 : U \ A = {2} := H
  sorry

end exercise_b_c_values_l142_142640


namespace smallest_int_with_divisors_l142_142083

theorem smallest_int_with_divisors :
  ∃ n : ℕ, 
    (∀ m, n = 2^2 * m → 
      (∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 
      (m = p^3 * q) ∧ 
      (nat.divisors p^7).count 8) ∧ 
    nat.divisors_count n = 24 ∧ 
    (nat.divisors (2^2 * n)).count 8) (n = 2^2 * 3^3 * 5) :=
begin
  sorry
end

end smallest_int_with_divisors_l142_142083


namespace sum_of_reciprocals_factors_12_l142_142302

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l142_142302


namespace simplify_expr_l142_142811

def A (a b : ℝ) := b^2 - a^2 + 5 * a * b
def B (a b : ℝ) := 3 * a * b + 2 * b^2 - a^2

theorem simplify_expr (a b : ℝ) : 2 * (A a b) - (B a b) = -a^2 + 7 * a * b := by
  -- actual proof omitted
  sorry

example : (2 * (A 1 2) - (B 1 2)) = 13 := by
  -- actual proof omitted
  sorry

end simplify_expr_l142_142811


namespace sum_of_reciprocals_factors_12_l142_142427

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l142_142427


namespace probability_of_X_eq_2_l142_142553

-- Define the random variable distribution condition
def random_variable_distribution (a : ℝ) (P : ℝ → ℝ) : Prop :=
  P 1 = 1 / (2 * a) ∧ P 2 = 2 / (2 * a) ∧ P 3 = 3 / (2 * a) ∧
  (1 / (2 * a) + 2 / (2 * a) + 3 / (2 * a) = 1)

-- State the theorem given the conditions and the result
theorem probability_of_X_eq_2 (a : ℝ) (P : ℝ → ℝ) (h : random_variable_distribution a P) : 
  P 2 = 1 / 3 :=
sorry

end probability_of_X_eq_2_l142_142553


namespace xyz_expression_l142_142552

theorem xyz_expression (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
    (h4 : x + y + z = 0) (h5 : xy + xz + yz ≠ 0) :
    (x^3 + y^3 + z^3) / (xyz * (xy + xz + yz)) = -3 / (2 * (x^2 + y^2 + xy)) :=
by sorry

end xyz_expression_l142_142552


namespace correct_calculation_l142_142772

theorem correct_calculation (x : ℝ) (h : (x / 2) + 45 = 85) : (2 * x) - 45 = 115 :=
by {
  -- Note: Proof steps are not needed, 'sorry' is used to skip the proof
  sorry
}

end correct_calculation_l142_142772


namespace sum_reciprocal_factors_of_12_l142_142411

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l142_142411


namespace simplify_expression_l142_142022

theorem simplify_expression (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_condition : a^3 + b^3 = 3 * (a + b)) : 
  (a / b + b / a + 1 / (a * b) = 4 / (a * b) + 1) :=
by
  sorry

end simplify_expression_l142_142022


namespace multiplication_result_l142_142481

theorem multiplication_result :
  10 * 9.99 * 0.999 * 100 = (99.9)^2 := 
by
  sorry

end multiplication_result_l142_142481


namespace sum_of_reciprocals_of_factors_of_12_l142_142378

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142378


namespace area_parallelogram_proof_l142_142217

/-- We are given a rectangle with a length of 10 cm and a width of 8 cm.
    We transform it into a parallelogram with a height of 9 cm.
    We need to prove that the area of the parallelogram is 72 square centimeters. -/
def area_of_parallelogram_from_rectangle (length width height : ℝ) : ℝ :=
  width * height

theorem area_parallelogram_proof
  (length width height : ℝ)
  (h_length : length = 10)
  (h_width : width = 8)
  (h_height : height = 9) :
  area_of_parallelogram_from_rectangle length width height = 72 :=
by
  sorry

end area_parallelogram_proof_l142_142217


namespace digits_base8_2015_l142_142156

theorem digits_base8_2015 : ∃ n : Nat, (8^n ≤ 2015 ∧ 2015 < 8^(n+1)) ∧ n + 1 = 4 := 
by 
  sorry

end digits_base8_2015_l142_142156


namespace mike_ride_distance_l142_142964

/-- 
Mike took a taxi to the airport and paid a starting amount plus $0.25 per mile. 
Annie took a different route to the airport and paid the same starting amount plus $5.00 in bridge toll fees plus $0.25 per mile. 
Each was charged exactly the same amount, and Annie's ride was 26 miles. 
Prove that Mike's ride was 46 miles given his starting amount was $2.50.
-/
theorem mike_ride_distance
  (S C A_miles : ℝ)                  -- S: starting amount, C: cost per mile, A_miles: Annie's ride distance
  (bridge_fee total_cost : ℝ)        -- bridge_fee: Annie's bridge toll fee, total_cost: total cost for both
  (M : ℝ)                            -- M: Mike's ride distance
  (hS : S = 2.5)
  (hC : C = 0.25)
  (hA_miles : A_miles = 26)
  (h_bridge_fee : bridge_fee = 5)
  (h_total_cost_equal : total_cost = S + bridge_fee + (C * A_miles))
  (h_total_cost_mike : total_cost = S + (C * M)) :
  M = 46 :=
by 
  sorry

end mike_ride_distance_l142_142964


namespace proof_a_squared_plus_1_l142_142649

theorem proof_a_squared_plus_1 (a : ℤ) (h1 : 3 < a) (h2 : a < 5) : a^2 + 1 = 17 :=
  by
  sorry

end proof_a_squared_plus_1_l142_142649


namespace larger_integer_is_21_l142_142736

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l142_142736


namespace sum_reciprocals_factors_12_l142_142339

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142339


namespace sum_of_reciprocals_of_factors_of_12_l142_142316

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142316


namespace jorge_total_goals_l142_142681

theorem jorge_total_goals (last_season_goals current_season_goals : ℕ) (h_last : last_season_goals = 156) (h_current : current_season_goals = 187) : 
  last_season_goals + current_season_goals = 343 :=
by
  sorry

end jorge_total_goals_l142_142681


namespace number_of_apples_l142_142600

theorem number_of_apples (C : ℝ) (A : ℕ) (total_cost : ℝ) (price_diff : ℝ) (num_oranges : ℕ)
  (h_price : C = 0.26)
  (h_price_diff : price_diff = 0.28)
  (h_num_oranges : num_oranges = 7)
  (h_total_cost : total_cost = 4.56) :
  A * C + num_oranges * (C + price_diff) = total_cost → A = 3 := 
by
  sorry

end number_of_apples_l142_142600


namespace sum_reciprocals_of_factors_12_l142_142367

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l142_142367


namespace sum_of_reciprocals_of_factors_of_12_l142_142435

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142435


namespace arithmetic_sequence_min_value_S_l142_142909

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l142_142909


namespace leak_emptying_time_l142_142779

-- Definitions based on given conditions
def tank_fill_rate_without_leak : ℚ := 1 / 3
def combined_fill_and_leak_rate : ℚ := 1 / 4

-- Leak emptying time to be proven
theorem leak_emptying_time (R : ℚ := tank_fill_rate_without_leak) (C : ℚ := combined_fill_and_leak_rate) :
  (1 : ℚ) / (R - C) = 12 := by
  sorry

end leak_emptying_time_l142_142779


namespace chess_group_players_count_l142_142987

theorem chess_group_players_count (n : ℕ)
  (h1 : ∀ (x y : ℕ), x ≠ y → ∃ k, k = 2)
  (h2 : n * (n - 1) / 2 = 45) :
  n = 10 := sorry

end chess_group_players_count_l142_142987


namespace fraction_subtraction_l142_142634

theorem fraction_subtraction (a b : ℝ) (h1 : 2 * b = 1 + a * b) (h2 : a ≠ 1) (h3 : b ≠ 1) :
  (a + 1) / (a - 1) - (b + 1) / (b - 1) = 2 :=
by
  sorry

end fraction_subtraction_l142_142634


namespace cloth_woven_on_30th_day_l142_142563

theorem cloth_woven_on_30th_day :
  (∃ d : ℚ, (30 * 5 + ((30 * 29) / 2) * d = 390) ∧ (5 + 29 * d = 21)) :=
by sorry

end cloth_woven_on_30th_day_l142_142563


namespace total_cost_of_ice_cream_l142_142700

theorem total_cost_of_ice_cream :
  let kiddie_scoop_cost := 3 in
  let regular_scoop_cost := 4 in
  let double_scoop_cost := 6 in
  let mr_and_mrs_martin_scoops := 2 in
  let children_scoops := 2 in
  let teenage_children_scoops := 3 in
  mr_and_mrs_martin_scoops * regular_scoop_cost +
  children_scoops * kiddie_scoop_cost +
  teenage_children_scoops * double_scoop_cost = 32 :=
by
  sorry

end total_cost_of_ice_cream_l142_142700


namespace find_smallest_number_l142_142576

theorem find_smallest_number (a b c : ℕ) 
  (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : b = 31)
  (h4 : c = b + 6)
  (h5 : (a + b + c) / 3 = 30) :
  a = 22 := 
sorry

end find_smallest_number_l142_142576


namespace sum_reciprocals_factors_12_l142_142352

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l142_142352


namespace quadrilateral_is_trapezoid_or_parallelogram_l142_142573

noncomputable def quadrilateral_property (s1 s2 s3 s4 : ℝ) : Prop :=
  (s1 + s2) * (s3 + s4) = (s1 + s4) * (s2 + s3)

theorem quadrilateral_is_trapezoid_or_parallelogram
  (s1 s2 s3 s4 : ℝ) (h : quadrilateral_property s1 s2 s3 s4) :
  (s1 = s3) ∨ (s2 = s4) ∨ -- Trapezoid conditions
  ∃ (p : ℝ), (p * s1 = s3 * (s1 + s4)) := -- Add necessary conditions to represent a parallelogram
sorry

end quadrilateral_is_trapezoid_or_parallelogram_l142_142573


namespace legendre_polynomial_expansion_l142_142823

noncomputable def f (α β γ : ℝ) (θ : ℝ) : ℝ := α + β * Real.cos θ + γ * Real.cos θ ^ 2

noncomputable def P0 (x : ℝ) : ℝ := 1
noncomputable def P1 (x : ℝ) : ℝ := x
noncomputable def P2 (x : ℝ) : ℝ := (3 * x ^ 2 - 1) / 2

theorem legendre_polynomial_expansion (α β γ : ℝ) (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
    f α β γ θ = (α + γ / 3) * P0 (Real.cos θ) + β * P1 (Real.cos θ) + (2 * γ / 3) * P2 (Real.cos θ) := by
  sorry

end legendre_polynomial_expansion_l142_142823


namespace find_larger_integer_l142_142741

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l142_142741


namespace Mary_ends_with_31_eggs_l142_142186

theorem Mary_ends_with_31_eggs (a b : ℕ) (h1 : a = 27) (h2 : b = 4) : a + b = 31 := by
  sorry

end Mary_ends_with_31_eggs_l142_142186


namespace total_money_l142_142162

namespace MoneyProof

variables (B J T : ℕ)

-- Given conditions
def condition_beth : Prop := B + 35 = 105
def condition_jan : Prop := J - 10 = B
def condition_tom : Prop := T = 3 * (J - 10)

-- Proof that the total money is $360
theorem total_money (h1 : condition_beth B) (h2 : condition_jan B J) (h3 : condition_tom J T) :
  B + J + T = 360 :=
by
  sorry

end MoneyProof

end total_money_l142_142162


namespace probability_red_in_both_jars_l142_142897

def original_red_buttons : ℕ := 6
def original_blue_buttons : ℕ := 10
def total_original_buttons : ℕ := original_red_buttons + original_blue_buttons
def remaining_buttons : ℕ := (2 * total_original_buttons) / 3
def moved_buttons : ℕ := total_original_buttons - remaining_buttons
def moved_red_buttons : ℕ := 2
def moved_blue_buttons : ℕ := 3

theorem probability_red_in_both_jars :
  moved_red_buttons = moved_blue_buttons →
  remaining_buttons = 11 →
  (∃ m n : ℚ, m / remaining_buttons = 4 / 11 ∧ n / (moved_red_buttons + moved_blue_buttons) = 2 / 5 ∧ (m / remaining_buttons) * (n / (moved_red_buttons + moved_blue_buttons)) = 8 / 55) :=
by sorry

end probability_red_in_both_jars_l142_142897


namespace monotone_decreasing_sequence_monotone_increasing_sequence_l142_142510

theorem monotone_decreasing_sequence (f : ℝ → ℝ) (a : ℕ → ℝ) (c : ℝ) :
  (∀ n : ℕ, a (n + 1) = f (a n)) →
  (a 1 = 0) →
  (∀ x : ℝ, f x = f (1 - x)) →
  (∀ x : ℝ, f x = -x^2 + x + c) →
  (∀ n : ℕ, a (n + 1) < a n) ↔ c < 0 :=
by sorry

theorem monotone_increasing_sequence (f : ℝ → ℝ) (a : ℕ → ℝ) (c : ℝ) :
  (∀ n : ℕ, a (n + 1) = f (a n)) →
  (a 1 = 0) →
  (∀ x : ℝ, f x = f (1 - x)) →
  (∀ x : ℝ, f x = -x^2 + x + c) →
  (∀ n : ℕ, a (n + 1) > a n) ↔ c > 1/4 :=
by sorry

end monotone_decreasing_sequence_monotone_increasing_sequence_l142_142510


namespace larger_integer_is_21_l142_142757

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l142_142757


namespace find_xy_l142_142456

noncomputable def xy_value (x y : ℝ) := x * y

theorem find_xy :
  ∃ x y : ℝ, (x + y = 2) ∧ (x^2 * y^3 + y^2 * x^3 = 32) ∧ xy_value x y = -8 :=
by
  sorry

end find_xy_l142_142456


namespace ratio_of_surface_areas_l142_142165

theorem ratio_of_surface_areas (r1 r2 : ℝ) (h : r1 / r2 = 1 / 2) :
  (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 4 :=
by
  sorry

end ratio_of_surface_areas_l142_142165


namespace sum_reciprocals_factors_12_l142_142259

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l142_142259


namespace sum_of_reciprocals_of_factors_of_12_l142_142442

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142442


namespace sum_reciprocals_factors_12_l142_142337

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142337


namespace min_sin6_cos6_l142_142846

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l142_142846


namespace length_ab_l142_142541

section geometry

variables {A B C D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Define the lengths and perimeters as needed
variables (AB AC BC CD DE CE : ℝ)

-- Isosceles Triangle properties
axiom isosceles_abc : AC = BC
axiom isosceles_cde : CD = DE

-- Conditons given in the problem
axiom perimeter_cde : CE + CD + DE = 22
axiom perimeter_abc : AB + BC + AC = 24
axiom length_ce : CE = 8

-- Goal: To prove the length of AB
theorem length_ab : AB = 10 :=
by 
  sorry

end geometry

end length_ab_l142_142541


namespace min_sixth_power_sin_cos_l142_142850

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l142_142850


namespace highest_student_id_in_sample_l142_142663

theorem highest_student_id_in_sample
    (total_students : ℕ)
    (sample_size : ℕ)
    (included_student_id : ℕ)
    (interval : ℕ)
    (first_id in_sample : ℕ)
    (k : ℕ)
    (highest_id : ℕ)
    (total_students_eq : total_students = 63)
    (sample_size_eq : sample_size = 7)
    (included_student_id_eq : included_student_id = 11)
    (k_def : k = total_students / sample_size)
    (included_student_id_in_second_pos : included_student_id = first_id + k)
    (interval_eq : interval = first_id - k)
    (in_sample_eq : in_sample = interval)
    (highest_id_eq : highest_id = in_sample + k * (sample_size - 1)) :
  highest_id = 56 := sorry

end highest_student_id_in_sample_l142_142663


namespace subset_size_condition_l142_142100

open Set

-- Define the nature of expressions as tuples within a specified range
def Expression (n : ℕ) (k : ℕ) := Fin n → Fin (k + 1)

-- Main theorem statement
theorem subset_size_condition (n k : ℕ) (h_k : 1 < k)
  (P Q : Finset (Expression n k))
  (h : ∀ p ∈ P, ∀ q ∈ Q, ∃ m, p m = q m) :
  P.card ≤ k^(n - 1) ∨ Q.card ≤ k^(n - 1) :=
  by sorry

end subset_size_condition_l142_142100


namespace baking_time_one_batch_l142_142118

theorem baking_time_one_batch (x : ℕ) (time_icing_per_batch : ℕ) (num_batches : ℕ) (total_time : ℕ)
  (h1 : num_batches = 4)
  (h2 : time_icing_per_batch = 30)
  (h3 : total_time = 200)
  (h4 : total_time = num_batches * x + num_batches * time_icing_per_batch) :
  x = 20 :=
by
  rw [h1, h2, h3] at h4
  sorry

end baking_time_one_batch_l142_142118


namespace total_number_of_elementary_events_is_16_l142_142664

def num_events_three_dice : ℕ := 6 * 6 * 6

theorem total_number_of_elementary_events_is_16 :
  num_events_three_dice = 16 := 
sorry

end total_number_of_elementary_events_is_16_l142_142664


namespace arithmetic_sequence_min_value_S_l142_142954

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l142_142954


namespace remainder_x_squared_div_25_l142_142882

theorem remainder_x_squared_div_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 0 [ZMOD 25] :=
sorry

end remainder_x_squared_div_25_l142_142882


namespace cube_inequality_l142_142515

theorem cube_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := 
sorry

end cube_inequality_l142_142515


namespace factor_expression_l142_142616

theorem factor_expression (x : ℝ) :
  (7 * x^6 + 36 * x^4 - 8) - (3 * x^6 - 4 * x^4 + 6) = 2 * (2 * x^6 + 20 * x^4 - 7) :=
  sorry

end factor_expression_l142_142616


namespace sum_of_reciprocals_of_factors_of_12_l142_142292

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142292


namespace arithmetic_sequence_minimum_value_of_Sn_l142_142921

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l142_142921


namespace savings_calculation_l142_142567

theorem savings_calculation (income expenditure : ℝ) (h_ratio : income = 5 / 4 * expenditure) (h_income : income = 19000) :
  income - expenditure = 3800 := 
by
  -- The solution will be filled in here,
  -- showing the calculus automatically.
  sorry

end savings_calculation_l142_142567


namespace arithmetic_sequence_min_value_S_l142_142951

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l142_142951


namespace sum_difference_even_odd_l142_142098

theorem sum_difference_even_odd :
  let x := (100 / 2) * (2 + 200)
  let y := (100 / 2) * (1 + 199)
  x - y = 100 :=
by
  sorry

end sum_difference_even_odd_l142_142098


namespace arithmetic_sequence_min_value_S_l142_142908

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l142_142908


namespace linear_equations_not_always_solvable_l142_142115

theorem linear_equations_not_always_solvable 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : 
  ¬(∀ x y : ℝ, (a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂) ↔ 
                   a₁ * b₂ - a₂ * b₁ ≠ 0) :=
sorry

end linear_equations_not_always_solvable_l142_142115


namespace probability_of_sum_multiple_of_3_l142_142986

noncomputable def card_numbers := {1, 2, 3, 4, 5}

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def pairs_sum_to_multiple_of_3 (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s |>.filter (λ p, is_multiple_of_3 (p.1 + p.2) ∧ p.1 < p.2)

def total_pairs (s : Finset ℕ) : ℕ :=
  (s.card.choose 2)

def favorable_pairs (s : Finset ℕ) : ℕ :=
  (pairs_sum_to_multiple_of_3 s).card

theorem probability_of_sum_multiple_of_3 :
  @Finset.card _ _ card_numbers > 1 → 
  (favorable_pairs card_numbers : ℚ) / (total_pairs card_numbers) = 2 / 5 :=
by
  sorry

end probability_of_sum_multiple_of_3_l142_142986


namespace largest_integer_value_of_x_l142_142584

theorem largest_integer_value_of_x (x : ℤ) (h : 8 - 5 * x > 22) : x ≤ -3 :=
sorry

end largest_integer_value_of_x_l142_142584


namespace Sue_made_22_buttons_l142_142177

-- Definitions of the conditions and the goal
def Mari_buttons : ℕ := 8
def Kendra_buttons := 4 + 5 * Mari_buttons
def Sue_buttons := Kendra_buttons / 2

-- Theorem statement
theorem Sue_made_22_buttons : Sue_buttons = 22 :=
by
  -- Definitions used here
  unfold Mari_buttons Kendra_buttons Sue_buttons
  -- Calculation
  rw [show 4 + 5 * 8 = 44, by norm_num, show 44 / 2 = 22, by norm_num]
  rfl

end Sue_made_22_buttons_l142_142177


namespace quadratic_inequality_l142_142558

noncomputable def ax2_plus_bx_c (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |ax2_plus_bx_c a b c x| ≤ 1 / 2) →
  ∀ x : ℝ, |x| ≥ 1 → |ax2_plus_bx_c a b c x| ≤ x^2 - 1 / 2 :=
by
  sorry

end quadratic_inequality_l142_142558


namespace number_of_children_l142_142782

theorem number_of_children (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 390)) : C = 780 :=
by
  sorry

end number_of_children_l142_142782


namespace sum_reciprocals_factors_12_l142_142266

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l142_142266


namespace arithmetic_sequence_minimum_value_of_Sn_l142_142920

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l142_142920


namespace hydrocarbon_tree_configurations_l142_142645

theorem hydrocarbon_tree_configurations (n : ℕ) 
  (h1 : 3 * n + 2 > 0) -- Total vertices count must be positive
  (h2 : 2 * n + 2 > 0) -- Leaves count must be positive
  (h3 : n > 0) -- Internal nodes count must be positive
  : (n:ℕ) ^ (n-2) = n ^ (n-2) :=
sorry

end hydrocarbon_tree_configurations_l142_142645


namespace sin_810_cos_neg60_l142_142487

theorem sin_810_cos_neg60 :
  Real.sin (810 * Real.pi / 180) + Real.cos (-60 * Real.pi / 180) = 3 / 2 :=
by
  sorry

end sin_810_cos_neg60_l142_142487


namespace larger_integer_is_21_l142_142726

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l142_142726


namespace sum_of_squares_and_cube_unique_l142_142539

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

theorem sum_of_squares_and_cube_unique : 
  ∃! (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_cube c ∧ a + b + c = 100 :=
sorry

end sum_of_squares_and_cube_unique_l142_142539


namespace smallest_side_length_1008_l142_142470

def smallest_side_length_original_square :=
  let n := Nat.lcm 7 8
  let n := Nat.lcm n 9
  let lcm := Nat.lcm n 10
  2 * lcm

theorem smallest_side_length_1008 :
  smallest_side_length_original_square = 1008 := by
  sorry

end smallest_side_length_1008_l142_142470


namespace sufficient_not_necessary_condition_l142_142123

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Iic (-2) → (x^2 + 2 * a * x - 2) ≤ ((x - 1)^2 + 2 * a * (x - 1) - 2)) ↔ a ≤ 2 := by
  sorry

end sufficient_not_necessary_condition_l142_142123


namespace base_conversion_problem_l142_142089

theorem base_conversion_problem :
  ∃ A B : ℕ, 0 ≤ A ∧ A < 8 ∧ 0 ≤ B ∧ B < 6 ∧
           8 * A + B = 6 * B + A ∧
           8 * A + B = 45 :=
by
  sorry

end base_conversion_problem_l142_142089


namespace expected_winnings_of_peculiar_die_l142_142468

noncomputable def peculiar_die_expected_winnings : ℝ := 
  let P_6 := 1 / 4
  let P_even := 2 / 5
  let P_odd := 3 / 5
  let winnings_6 := 4
  let winnings_even := -2
  let winnings_odd := 1
  P_6 * winnings_6 + P_even * winnings_even + P_odd * winnings_odd

theorem expected_winnings_of_peculiar_die :
  peculiar_die_expected_winnings = 0.80 :=
by
  sorry

end expected_winnings_of_peculiar_die_l142_142468


namespace part1_zero_of_f_part2_a_range_l142_142637

-- Define the given function f
def f (x a b : ℝ) : ℝ := (x - a) * |x| + b

-- Define the problem statement for Part 1
theorem part1_zero_of_f :
  ∀ (x : ℝ),
    f x 2 3 = 0 ↔ x = -1 := 
by
  sorry

-- Define the problem statement for Part 2
theorem part2_a_range :
  ∀ (a : ℝ),
    (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → f x a (-2) < 0) ↔ a > -1 :=
by
  sorry

end part1_zero_of_f_part2_a_range_l142_142637


namespace arithmetic_sequence_min_value_S_l142_142906

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l142_142906


namespace sum_reciprocals_12_l142_142237

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l142_142237


namespace suma_work_rate_l142_142588

theorem suma_work_rate (r s : ℝ) (hr : r = 1 / 5) (hrs : r + s = 1 / 4) : 1 / s = 20 := by
  sorry

end suma_work_rate_l142_142588


namespace smallest_positive_integer_with_divisors_l142_142076

theorem smallest_positive_integer_with_divisors :
  ∃ n : ℕ, (∀ d : ℕ, d ∣ n → nat.odd d) ∧ (finset.filter nat.odd (finset.divisors n)).card = 8 ∧ 
           (∃ m : ℕ, ∀ d : ℕ, d ∣ m → nat.even d ∧ m = n → (finset.filter nat.even (finset.divisors m)).card = 16)
             → n = 756 :=
by
  sorry

end smallest_positive_integer_with_divisors_l142_142076


namespace robert_total_interest_l142_142038

theorem robert_total_interest
  (inheritance : ℕ)
  (part1 part2 : ℕ)
  (rate1 rate2 : ℝ)
  (time : ℝ) :
  inheritance = 4000 →
  part2 = 1800 →
  part1 = inheritance - part2 →
  rate1 = 0.05 →
  rate2 = 0.065 →
  time = 1 →
  (part1 * rate1 * time + part2 * rate2 * time) = 227 :=
by
  intros
  sorry

end robert_total_interest_l142_142038


namespace gcd_lcm_sum_l142_142087

theorem gcd_lcm_sum :
  gcd 42 70 + lcm 15 45 = 59 :=
by sorry

end gcd_lcm_sum_l142_142087


namespace bullying_instances_l142_142545

-- Let's denote the total number of suspension days due to bullying and serious incidents.
def total_suspension_days : ℕ := (3 * (10 + 10)) + 14

-- Each instance of bullying results in a 3-day suspension.
def days_per_instance : ℕ := 3

-- The number of instances of bullying given the total suspension days.
def instances_of_bullying := total_suspension_days / days_per_instance

-- We must prove that Kris is responsible for 24 instances of bullying.
theorem bullying_instances : instances_of_bullying = 24 := by
  sorry

end bullying_instances_l142_142545


namespace sum_of_reciprocals_of_factors_of_12_l142_142311

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142311


namespace difference_pencils_l142_142116

theorem difference_pencils (x : ℕ) (h1 : 162 = x * n_g) (h2 : 216 = x * n_f) : n_f - n_g = 3 :=
by
  sorry

end difference_pencils_l142_142116


namespace sum_reciprocals_factors_12_l142_142344

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l142_142344


namespace tea_to_cheese_ratio_l142_142580

-- Definitions based on conditions
def total_cost : ℝ := 21
def tea_cost : ℝ := 10
def butter_to_cheese_ratio : ℝ := 0.8
def bread_to_butter_ratio : ℝ := 0.5

-- Main theorem statement
theorem tea_to_cheese_ratio (B C Br : ℝ) (hBr : Br = B * bread_to_butter_ratio) (hB : B = butter_to_cheese_ratio * C) (hTotal : B + Br + C + tea_cost = total_cost) :
  10 / C = 2 :=
  sorry

end tea_to_cheese_ratio_l142_142580


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l142_142944

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l142_142944


namespace arithmetic_sequence_problem_l142_142904

theorem arithmetic_sequence_problem
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 : a 1 = 1)
  (a3 : a 3 = 5)
  (Sn : ∀ n, S n = n * (2 + (n - 1) * 2) / 2)
  (S_diff : ∀ k, S (k + 2) - S k = 36)
  : ∃ k : ℕ, k = 8 :=
by
  sorry

end arithmetic_sequence_problem_l142_142904


namespace solve_for_m_l142_142131

-- Define the operation ◎ for real numbers a and b
def op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Lean statement for the proof problem
theorem solve_for_m (m : ℝ) (h : op (m + 1) (m - 2) = 16) : m = 3 ∨ m = -2 :=
sorry

end solve_for_m_l142_142131


namespace mike_books_before_yard_sale_l142_142033

-- Problem definitions based on conditions
def books_bought_at_yard_sale : ℕ := 21
def books_now_in_library : ℕ := 56
def books_before_yard_sale := books_now_in_library - books_bought_at_yard_sale

-- Theorem to prove the equivalent proof problem
theorem mike_books_before_yard_sale : books_before_yard_sale = 35 := by
  sorry

end mike_books_before_yard_sale_l142_142033


namespace gcd_lcm_product_l142_142502

theorem gcd_lcm_product (a b : ℕ) (ha : a = 90) (hb : b = 150) : 
  Nat.gcd a b * Nat.lcm a b = 13500 := 
by 
  sorry

end gcd_lcm_product_l142_142502


namespace maria_cupcakes_l142_142506

variable (initial : ℕ) (additional : ℕ) (remaining : ℕ)

theorem maria_cupcakes (h_initial : initial = 19) (h_additional : additional = 10) (h_remaining : remaining = 24) : initial + additional - remaining = 5 := by
  sorry

end maria_cupcakes_l142_142506


namespace bah_rah_yah_equiv_l142_142653

-- We define the initial equivalences given in the problem statement.
theorem bah_rah_yah_equiv (bahs rahs yahs : ℕ) :
  (18 * bahs = 30 * rahs) ∧
  (12 * rahs = 20 * yahs) →
  (1200 * yahs = 432 * bahs) :=
by
  -- Placeholder for the actual proof
  sorry

end bah_rah_yah_equiv_l142_142653


namespace digit_d_makes_multiple_of_9_l142_142627

theorem digit_d_makes_multiple_of_9 :
  ∃ d : ℕ, d < 10 ∧ (26 + d) % 9 = 0 ∧ d = 1 :=
by {
  have h1 : 26 % 9 = 8 := rfl,
  use 1,
  split,
  { linarith },
  split,
  { norm_num },
  { refl }
}

end digit_d_makes_multiple_of_9_l142_142627


namespace quadrilateral_angles_combinations_pentagon_angles_combination_l142_142221

-- Define angle types
inductive AngleType
| acute
| right
| obtuse

open AngleType

-- Define predicates for sum of angles in a quadrilateral and pentagon
def quadrilateral_sum (angles : List AngleType) : Bool :=
  match angles with
  | [right, right, right, right] => true
  | [right, right, acute, obtuse] => true
  | [right, acute, obtuse, obtuse] => true
  | [right, acute, acute, obtuse] => true
  | [acute, obtuse, obtuse, obtuse] => true
  | [acute, acute, obtuse, obtuse] => true
  | [acute, acute, acute, obtuse] => true
  | _ => false

def pentagon_sum (angles : List AngleType) : Prop :=
  -- Broad statement, more complex combinations possible
  ∃ a b c d e : ℕ, (a + b + c + d + e = 540) ∧
    (a < 90 ∨ a = 90 ∨ a > 90) ∧
    (b < 90 ∨ b = 90 ∨ b > 90) ∧
    (c < 90 ∨ c = 90 ∨ c > 90) ∧
    (d < 90 ∨ d = 90 ∨ d > 90) ∧
    (e < 90 ∨ e = 90 ∨ e > 90)

-- Prove the possible combinations for a quadrilateral and a pentagon
theorem quadrilateral_angles_combinations {angles : List AngleType} :
  quadrilateral_sum angles = true :=
sorry

theorem pentagon_angles_combination :
  ∃ angles : List AngleType, pentagon_sum angles :=
sorry

end quadrilateral_angles_combinations_pentagon_angles_combination_l142_142221


namespace remainder_x_squared_mod_25_l142_142879

theorem remainder_x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 4 [ZMOD 25] :=
sorry

end remainder_x_squared_mod_25_l142_142879


namespace parallel_lines_condition_l142_142213

theorem parallel_lines_condition (a : ℝ) : 
  (a = -2) ↔ (∀ x y : ℝ, ax + 2 * y = 0 → y = 1 + x) := 
sorry

end parallel_lines_condition_l142_142213


namespace sum_of_reciprocals_of_factors_of_12_l142_142289

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142289


namespace sum_of_reciprocals_of_factors_of_12_l142_142433

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142433


namespace sum_reciprocals_factors_of_12_l142_142284

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l142_142284


namespace P_sufficient_but_not_necessary_for_Q_l142_142135

-- Definitions based on given conditions
def P (x : ℝ) : Prop := abs (2 * x - 3) < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

-- The theorem to prove that P is sufficient but not necessary for Q
theorem P_sufficient_but_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬P x) :=
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l142_142135


namespace sequence_properties_l142_142984

theorem sequence_properties :
  ∀ {a : ℕ → ℝ} {b : ℕ → ℝ},
  a 1 = 1 ∧ 
  (∀ n, b n > 4 / 3) ∧ 
  (∀ n, (∀ x, x^2 - b n * x + a n = 0 → (x = a (n + 1) ∨ x = 1 + a n))) →
  (a 2 = 1 / 2 ∧ ∃ n, b n > 4 / 3 ∧ n = 5) := by
  sorry

end sequence_properties_l142_142984


namespace fruit_seller_profit_l142_142790

theorem fruit_seller_profit 
  (SP : ℝ) (Loss_Percentage : ℝ) (New_SP : ℝ) (Profit_Percentage : ℝ) 
  (h1: SP = 8) 
  (h2: Loss_Percentage = 20) 
  (h3: New_SP = 10.5) 
  (h4: Profit_Percentage = 5) :
  ((New_SP - (SP / (1 - (Loss_Percentage / 100.0))) / (SP / (1 - (Loss_Percentage / 100.0)))) * 100) = Profit_Percentage := 
sorry

end fruit_seller_profit_l142_142790


namespace sum_reciprocals_of_factors_of_12_l142_142326

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l142_142326


namespace pow_addition_l142_142593

theorem pow_addition : (-2 : ℤ)^2 + (2 : ℤ)^2 = 8 :=
by
  sorry

end pow_addition_l142_142593


namespace arithmetic_sequence_minimum_value_S_n_l142_142935

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l142_142935


namespace proof_U_eq_A_union_complement_B_l142_142554

noncomputable def U : Set Nat := {1, 2, 3, 4, 5, 7}
noncomputable def A : Set Nat := {1, 3, 5, 7}
noncomputable def B : Set Nat := {3, 5}
noncomputable def complement_U_B := U \ B

theorem proof_U_eq_A_union_complement_B : U = A ∪ complement_U_B := by
  sorry

end proof_U_eq_A_union_complement_B_l142_142554


namespace prove_arithmetic_sequence_minimum_value_S_l142_142933

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l142_142933


namespace last_two_digits_of_17_pow_17_l142_142819

theorem last_two_digits_of_17_pow_17 : (17 ^ 17) % 100 = 77 := 
by sorry

end last_two_digits_of_17_pow_17_l142_142819


namespace sum_reciprocals_of_factors_of_12_l142_142325

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l142_142325


namespace max_cards_l142_142675

def card_cost : ℝ := 0.85
def budget : ℝ := 7.50

theorem max_cards (n : ℕ) : card_cost * n ≤ budget → n ≤ 8 :=
by sorry

end max_cards_l142_142675


namespace number_of_pots_of_rosemary_l142_142473

-- Definitions based on the conditions
def total_leaves_basil (pots_basil : ℕ) (leaves_per_basil : ℕ) : ℕ := pots_basil * leaves_per_basil
def total_leaves_rosemary (pots_rosemary : ℕ) (leaves_per_rosemary : ℕ) : ℕ := pots_rosemary * leaves_per_rosemary
def total_leaves_thyme (pots_thyme : ℕ) (leaves_per_thyme : ℕ) : ℕ := pots_thyme * leaves_per_thyme

-- The given problem conditions
def pots_basil : ℕ := 3
def leaves_per_basil : ℕ := 4
def leaves_per_rosemary : ℕ := 18
def pots_thyme : ℕ := 6
def leaves_per_thyme : ℕ := 30
def total_leaves : ℕ := 354

-- Proving the number of pots of rosemary
theorem number_of_pots_of_rosemary : 
  ∃ (pots_rosemary : ℕ), 
  total_leaves_basil pots_basil leaves_per_basil + 
  total_leaves_rosemary pots_rosemary leaves_per_rosemary + 
  total_leaves_thyme pots_thyme leaves_per_thyme = 
  total_leaves ∧ pots_rosemary = 9 :=
by
  sorry  -- proof is omitted

end number_of_pots_of_rosemary_l142_142473


namespace sum_of_reciprocals_of_factors_of_12_l142_142383

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142383


namespace smallest_integer_with_divisors_l142_142081

theorem smallest_integer_with_divisors :
  ∃ (n : ℕ), 
    (∀ d : ℕ, d ∣ n → d % 2 = 1 → (∃! k : ℕ, d = (3 ^ k) * 5 ^ (7 - k))) ∧ 
    (∀ d : ℕ, d ∣ n → d % 2 = 0 → (∃! k : ℕ, d = 2 ^ k * m)) ∧ 
    (n = 1080) :=
sorry

end smallest_integer_with_divisors_l142_142081


namespace find_larger_integer_l142_142745

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l142_142745


namespace length_of_ab_l142_142777

variable (a b c d e : ℝ)
variable (bc cd de ac ae ab : ℝ)

axiom bc_eq_3cd : bc = 3 * cd
axiom de_eq_7 : de = 7
axiom ac_eq_11 : ac = 11
axiom ae_eq_20 : ae = 20
axiom ac_def : ac = ab + bc -- Definition of ac
axiom ae_def : ae = ab + bc + cd + de -- Definition of ae

theorem length_of_ab : ab = 5 := by
  sorry

end length_of_ab_l142_142777


namespace H_double_prime_coordinates_l142_142556

/-- Define the points of the parallelogram EFGH and their reflections. --/
structure Point := (x : ℝ) (y : ℝ)

def E : Point := ⟨3, 4⟩
def F : Point := ⟨5, 7⟩
def G : Point := ⟨7, 4⟩
def H : Point := ⟨5, 1⟩

/-- Reflection of a point across the x-axis changes the y-coordinate sign. --/
def reflect_x (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Reflection of a point across y=x-1 involves translation and reflection across y=x. --/
def reflect_y_x_minus_1 (p : Point) : Point :=
  let translated := Point.mk p.x (p.y + 1)
  let reflected := Point.mk translated.y translated.x
  Point.mk reflected.x (reflected.y - 1)

def H' : Point := reflect_x H
def H'' : Point := reflect_y_x_minus_1 H'

theorem H_double_prime_coordinates : H'' = ⟨0, 4⟩ :=
by
  sorry

end H_double_prime_coordinates_l142_142556


namespace time_bob_cleans_room_l142_142113

variable (timeAlice : ℕ) (fractionBob : ℚ)

-- Definitions based on conditions from the problem
def timeAliceCleaningRoom : ℕ := 40
def fractionOfTimeBob : ℚ := 3 / 8

-- Prove the time it takes Bob to clean his room
theorem time_bob_cleans_room : (timeAliceCleaningRoom * fractionOfTimeBob : ℚ) = 15 := 
by
  sorry

end time_bob_cleans_room_l142_142113


namespace num_two_digit_congruent_to_3_mod_4_l142_142000

open Set

theorem num_two_digit_congruent_to_3_mod_4 : 
  (finset.card (finset.filter (λ n, n % 4 = 3) (finset.Icc 10 99)) = 23) := 
by 
  sorry

end num_two_digit_congruent_to_3_mod_4_l142_142000


namespace max_digit_sum_in_24_hour_format_l142_142599

def digit_sum (n : ℕ) : ℕ := 
  (n / 10) + (n % 10)

theorem max_digit_sum_in_24_hour_format :
  (∃ (h m : ℕ), 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 ∧ digit_sum h + digit_sum m = 19) ∧
  ∀ (h m : ℕ), 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 → digit_sum h + digit_sum m ≤ 19 :=
by
  sorry

end max_digit_sum_in_24_hour_format_l142_142599


namespace ratio_problem_l142_142647

open Classical 

variables {q r s t u : ℚ}

theorem ratio_problem (h1 : q / r = 8) (h2 : s / r = 5) (h3 : s / t = 1 / 4) (h4 : u / t = 3) :
  u / q = 15 / 2 :=
by
  sorry

end ratio_problem_l142_142647


namespace max_product_of_xy_l142_142533

open Real

theorem max_product_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 1) :
  x * y ≤ 1 / 16 := 
sorry

end max_product_of_xy_l142_142533


namespace sum_reciprocals_factors_12_l142_142353

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l142_142353


namespace stone_length_l142_142792

theorem stone_length (hall_length_m : ℕ) (hall_breadth_m : ℕ) (number_of_stones : ℕ) (stone_width_dm : ℕ) 
    (length_in_dm : 10 > 0) :
    hall_length_m = 36 → hall_breadth_m = 15 → number_of_stones = 2700 → stone_width_dm = 5 →
    ∀ L : ℕ, 
    (10 * hall_length_m) * (10 * hall_breadth_m) = number_of_stones * (L * stone_width_dm) → 
    L = 4 :=
by
  intros h1 h2 h3 h4
  simp at *
  sorry

end stone_length_l142_142792


namespace petya_numbers_l142_142969

-- Define the arithmetic sequence property
def arithmetic_seq (a d : ℕ) : ℕ → ℕ
| 0     => a
| (n+1) => a + (n + 1) * d

-- Given conditions
theorem petya_numbers (a d : ℕ) : 
  (arithmetic_seq a d 0 = 6) ∧
  (arithmetic_seq a d 1 = 15) ∧
  (arithmetic_seq a d 2 = 24) ∧
  (arithmetic_seq a d 3 = 33) ∧
  (arithmetic_seq a d 4 = 42) :=
sorry

end petya_numbers_l142_142969


namespace smallest_floor_sum_l142_142877

theorem smallest_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    ⌊(a + b) / c⌋ + ⌊(b + c) / a⌋ + ⌊(c + a) / b⌋ ≥ 4 :=
sorry

end smallest_floor_sum_l142_142877


namespace sum_reciprocal_factors_12_l142_142249

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l142_142249


namespace dress_designs_count_l142_142598

-- Define the number of colors, fabric types, and patterns
def num_colors : Nat := 3
def num_fabric_types : Nat := 4
def num_patterns : Nat := 3

-- Define the total number of dress designs
def total_dress_designs : Nat := num_colors * num_fabric_types * num_patterns

-- Define the theorem to prove the equivalence
theorem dress_designs_count :
  total_dress_designs = 36 :=
by
  -- This is to show the theorem's structure; proof will be added here.
  sorry

end dress_designs_count_l142_142598


namespace probability_one_boy_one_girl_l142_142464

theorem probability_one_boy_one_girl (boys girls : ℕ) (h_boys : boys = 2) (h_girls : girls = 1) :
  (∃ (students : ℕ) (choose : ℕ) (total_pairs : ℕ) (favorable_pairs : ℕ),
    students = boys + girls ∧
    choose = 2 ∧
    total_pairs = (students.choose choose) ∧
    favorable_pairs = 2 ∧
    (favorable_pairs : ℚ) / (total_pairs : ℚ) = 2 / 3) :=
begin
  use [3, 2, (3.choose 2), 2],
  split,
  { exact eq_add_of_add_eq h_boys h_girls, },
  split,
  { refl, },
  split,
  { apply nat.choose},
  split,
  { refl, },
  { norm_num,
    rw [nat.cast_two, nat.cast_choose ℚ 3 2],
    norm_num, },
end

end probability_one_boy_one_girl_l142_142464


namespace sum_reciprocals_of_factors_12_l142_142369

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l142_142369


namespace sum_of_reciprocals_factors_12_l142_142431

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l142_142431


namespace modulus_complex_number_l142_142569

theorem modulus_complex_number (i : ℂ) (h : i = Complex.I) : 
  Complex.abs (1 / (i - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end modulus_complex_number_l142_142569


namespace quadratic_expression_positive_l142_142040

theorem quadratic_expression_positive
  (a b c : ℝ) (x : ℝ)
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  b^2 * x^2 + (b^2 + c^2 - a^2) * x + c^2 > 0 :=
sorry

end quadratic_expression_positive_l142_142040


namespace medical_team_combinations_l142_142056

-- Number of combinations function
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem medical_team_combinations :
  let maleDoctors := 6
  let femaleDoctors := 5
  let requiredMale := 2
  let requiredFemale := 1
  choose maleDoctors requiredMale * choose femaleDoctors requiredFemale = 75 :=
by
  sorry

end medical_team_combinations_l142_142056


namespace pyramid_section_rhombus_l142_142695

structure Pyramid (A B C D : Type) := (point : Type)

def is_parallel (l1 l2 : ℝ) : Prop :=
  ∀ (m n : ℝ), m * l1 = n * l2

def is_parallelogram (K L M N : Type) : Prop :=
  sorry

def is_rhombus (K L M N : Type) : Prop :=
  sorry

noncomputable def side_length_rhombus (a b : ℝ) : ℝ :=
  (a * b) / (a + b)

/-- Prove that the section of pyramid ABCD with a plane parallel to edges AC and BD is a parallelogram,
and under certain conditions, this parallelogram is a rhombus. Find the side of this rhombus given AC = a and BD = b. -/
theorem pyramid_section_rhombus (A B C D K L M N : Type) (a b : ℝ) :
  is_parallel AC BD →
  is_parallelogram K L M N →
  is_rhombus K L M N →
  side_length_rhombus a b = (a * b) / (a + b) :=
by
  sorry

end pyramid_section_rhombus_l142_142695


namespace sum_reciprocal_factors_12_l142_142251

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l142_142251


namespace sum_of_reciprocals_of_factors_of_12_l142_142382

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142382


namespace isosceles_triangle_perimeter_eq_10_l142_142806

theorem isosceles_triangle_perimeter_eq_10 (x : ℝ) 
(base leg : ℝ)
(h_base : base = 4)
(h_leg_root : x^2 - 5 * x + 6 = 0)
(h_iso : leg = x)
(triangle_ineq : leg + leg > base):
  2 * leg + base = 10 := 
begin
  cases (em (x = 2)) with h1 h2,
  { rw h1 at h_leg_root,
    rw [←h_iso, h1] at triangle_ineq,
    simp at triangle_ineq,
    contradiction },
  { rw h_iso,
    have : x = 3,
    { by_contra,
      simp [not_or_distrib, h1, h, sub_eq_zero] at h_leg_root },
    rw this,
    simp,
    linarith }
end

# Testing if the theorem can be evaluated successfully
# theorem_example : isosceles_triangle_perimeter_eq_10 3 4 3 rfl rfl sorry sorry rfl :=
# sorry

end isosceles_triangle_perimeter_eq_10_l142_142806


namespace probability_of_diamond_ace_joker_l142_142461

noncomputable def probability_event (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  event_cards / total_cards

noncomputable def probability_not_event (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  1 - probability_event total_cards event_cards

noncomputable def probability_none_event_two_trials (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  (probability_not_event total_cards event_cards) * (probability_not_event total_cards event_cards)

noncomputable def probability_at_least_one_event_two_trials (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  1 - probability_none_event_two_trials total_cards event_cards

theorem probability_of_diamond_ace_joker 
  (total_cards : ℕ := 54) (event_cards : ℕ := 18) :
  probability_at_least_one_event_two_trials total_cards event_cards = 5 / 9 :=
by
  sorry

end probability_of_diamond_ace_joker_l142_142461


namespace probability_calculation_l142_142199

noncomputable def probability_same_color (pairs_black pairs_brown pairs_gray : ℕ) : ℚ :=
  let total_shoes := 2 * (pairs_black + pairs_brown + pairs_gray)
  let prob_black := (2 * pairs_black : ℚ) / total_shoes * (pairs_black : ℚ) / (total_shoes - 1)
  let prob_brown := (2 * pairs_brown : ℚ) / total_shoes * (pairs_brown : ℚ) / (total_shoes - 1)
  let prob_gray := (2 * pairs_gray : ℚ) / total_shoes * (pairs_gray : ℚ) / (total_shoes - 1)
  prob_black + prob_brown + prob_gray

theorem probability_calculation :
  probability_same_color 7 4 3 = 37 / 189 :=
by
  sorry

end probability_calculation_l142_142199


namespace line_bisects_circle_area_l142_142047

theorem line_bisects_circle_area (b : ℝ) :
  (∀ x y : ℝ, y = 2 * x + b ↔ x^2 + y^2 - 2 * x - 4 * y + 4 = 0) → b = 0 :=
by
  sorry

end line_bisects_circle_area_l142_142047


namespace sum_of_reciprocals_of_factors_of_12_l142_142294

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142294


namespace amount_solution_y_correct_l142_142099

-- Define conditions
def solution_x_alcohol_percentage : ℝ := 0.10
def solution_y_alcohol_percentage : ℝ := 0.30
def volume_solution_x : ℝ := 300.0
def target_alcohol_percentage : ℝ := 0.18

-- Define the main question as a theorem
theorem amount_solution_y_correct (y : ℝ) :
  (30 + 0.3 * y = 0.18 * (300 + y)) → y = 200 :=
by
  sorry

end amount_solution_y_correct_l142_142099


namespace find_5y_45_sevenths_l142_142001

theorem find_5y_45_sevenths (x y : ℝ) 
(h1 : 3 * x + 4 * y = 0) 
(h2 : x = y + 3) : 
5 * y = -45 / 7 :=
by
  sorry

end find_5y_45_sevenths_l142_142001


namespace sum_of_reciprocals_factors_12_l142_142429

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l142_142429


namespace find_a_l142_142528

theorem find_a (a : ℝ) (U A CU: Set ℝ) (hU : U = {2, 3, a^2 - a - 1}) (hA : A = {2, 3}) (hCU : CU = {1}) (hComplement : CU = U \ A) :
  a = -1 ∨ a = 2 :=
by
  sorry

end find_a_l142_142528


namespace tax_percentage_l142_142673

theorem tax_percentage (total_pay take_home_pay: ℕ) (h1 : total_pay = 650) (h2 : take_home_pay = 585) :
  ((total_pay - take_home_pay) * 100 / total_pay) = 10 :=
by
  -- Assumptions
  have hp1 : total_pay = 650 := h1
  have hp2 : take_home_pay = 585 := h2
  -- Calculate tax paid
  let tax_paid := total_pay - take_home_pay
  -- Calculate tax percentage
  let tax_percentage := (tax_paid * 100) / total_pay
  -- Prove the tax percentage is 10%
  sorry

end tax_percentage_l142_142673


namespace find_larger_integer_l142_142742

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l142_142742


namespace kati_age_l142_142900

/-- Define the age of Kati using the given conditions -/
theorem kati_age (kati_age : ℕ) (brother_age kati_birthdays : ℕ) 
  (h1 : kati_age = kati_birthdays) 
  (h2 : kati_age + brother_age = 111) 
  (h3 : kati_birthdays = kati_age) : 
  kati_age = 18 :=
by
  sorry

end kati_age_l142_142900


namespace ephraim_keiko_same_tails_l142_142544

def outcomes : List (List Char) := [
  ['H', 'H'], ['H', 'T'], ['T', 'H'], ['T', 'T']
]

def count_tails (lst : List Char) : Nat :=
  lst.count (· == 'T')

def favorable_cases : Nat :=
  List.filter (λ a, List.any (λ b, count_tails a = count_tails b) outcomes) outcomes |>.length

theorem ephraim_keiko_same_tails : favorable_cases / outcomes.length.toRat ^ 2 = 3 / 8 := by
  sorry

end ephraim_keiko_same_tails_l142_142544


namespace sum_of_reciprocals_of_factors_of_12_l142_142298

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142298


namespace find_larger_integer_l142_142747

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l142_142747


namespace sum_reciprocals_factors_12_l142_142264

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l142_142264


namespace larger_integer_value_l142_142717

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l142_142717


namespace sum_reciprocals_factors_12_l142_142343

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142343


namespace quartic_polynomial_eval_l142_142163

noncomputable def f (x : ℝ) : ℝ := sorry  -- f is a monic quartic polynomial

theorem quartic_polynomial_eval (h_monic: true)
    (h1 : f (-1) = -1)
    (h2 : f 2 = -4)
    (h3 : f (-3) = -9)
    (h4 : f 4 = -16) : f 1 = 23 :=
sorry

end quartic_polynomial_eval_l142_142163


namespace sum_reciprocals_of_factors_of_12_l142_142332

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l142_142332


namespace sum_of_reciprocals_of_factors_of_12_l142_142381

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142381


namespace sum_reciprocals_of_factors_12_l142_142371

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l142_142371


namespace substance_same_number_of_atoms_l142_142093

def molecule (kind : String) (atom_count : ℕ) := (kind, atom_count)

def H3PO4 := molecule "H₃PO₄" 8
def H2O2 := molecule "H₂O₂" 4
def H2SO4 := molecule "H₂SO₄" 7
def NaCl := molecule "NaCl" 2 -- though it consists of ions, let's denote it as 2 for simplicity
def HNO3 := molecule "HNO₃" 5

def mol_atoms (mol : ℝ) (molecule : ℕ) : ℝ := mol * molecule

theorem substance_same_number_of_atoms :
  mol_atoms 0.2 H3PO4.2 = mol_atoms 0.4 H2O2.2 :=
by
  unfold H3PO4 H2O2 mol_atoms
  sorry

end substance_same_number_of_atoms_l142_142093


namespace piesEatenWithForksPercentage_l142_142105

def totalPies : ℕ := 2000
def notEatenWithForks : ℕ := 640
def eatenWithForks : ℕ := totalPies - notEatenWithForks

def percentageEatenWithForks := (eatenWithForks : ℚ) / totalPies * 100

theorem piesEatenWithForksPercentage : percentageEatenWithForks = 68 := by
  sorry

end piesEatenWithForksPercentage_l142_142105


namespace auntie_em_can_park_l142_142604

-- Define the conditions as formal statements in Lean
def parking_lot_spaces : ℕ := 20
def cars_arriving : ℕ := 14
def suv_adjacent_spaces : ℕ := 2

-- Define the total number of ways to park 14 cars in 20 spaces
def total_ways_to_park : ℕ := Nat.choose parking_lot_spaces cars_arriving
-- Define the number of unfavorable configurations where the SUV cannot park
def unfavorable_configs : ℕ := Nat.choose (parking_lot_spaces - suv_adjacent_spaces + 1) (parking_lot_spaces - cars_arriving)

-- Final probability calculation
def probability_park_suv : ℚ := 1 - (unfavorable_configs / total_ways_to_park)

-- Mathematically equivalent statement to be proved
theorem auntie_em_can_park : probability_park_suv = 850 / 922 :=
by sorry

end auntie_em_can_park_l142_142604


namespace min_value_sin_cos_l142_142830

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l142_142830


namespace bc_guilty_l142_142989

-- Definition of guilty status of defendants
variables (A B C : Prop)

-- Conditions
axiom condition1 : A ∨ B ∨ C
axiom condition2 : A → ¬B → ¬C

-- Theorem stating that one of B or C is guilty
theorem bc_guilty : B ∨ C :=
by {
  -- Proof goes here
  sorry
}

end bc_guilty_l142_142989


namespace find_number_l142_142883

theorem find_number (x q : ℕ) (h1 : x = 3 * q) (h2 : q + x + 3 = 63) : x = 45 :=
sorry

end find_number_l142_142883


namespace maria_earnings_l142_142958

-- Define the conditions
def costOfBrushes : ℕ := 20
def costOfCanvas : ℕ := 3 * costOfBrushes
def costPerLiterOfPaint : ℕ := 8
def litersOfPaintNeeded : ℕ := 5
def sellingPriceOfPainting : ℕ := 200

-- Define the total cost calculation
def totalCostOfMaterials : ℕ := costOfBrushes + costOfCanvas + (costPerLiterOfPaint * litersOfPaintNeeded)

-- Define the final earning calculation
def mariaEarning : ℕ := sellingPriceOfPainting - totalCostOfMaterials

-- State the theorem
theorem maria_earnings :
  mariaEarning = 80 := by
  sorry

end maria_earnings_l142_142958


namespace tim_movie_marathon_duration_is_9_l142_142999

-- Define the conditions:
def first_movie_duration : ℕ := 2
def second_movie_duration : ℕ := first_movie_duration + (first_movie_duration / 2)
def combined_duration_first_two_movies : ℕ := first_movie_duration + second_movie_duration
def third_movie_duration : ℕ := combined_duration_first_two_movies - 1
def total_marathon_duration : ℕ := first_movie_duration + second_movie_duration + third_movie_duration

-- The theorem to prove the marathon duration is 9 hours
theorem tim_movie_marathon_duration_is_9 :
  total_marathon_duration = 9 :=
by sorry

end tim_movie_marathon_duration_is_9_l142_142999


namespace trigonometric_expression_in_third_quadrant_l142_142101

theorem trigonometric_expression_in_third_quadrant (α : ℝ) 
  (h1 : Real.sin α < 0) 
  (h2 : Real.cos α < 0) 
  (h3 : Real.tan α > 0) : 
  ¬ (Real.tan α - Real.sin α < 0) :=
sorry

end trigonometric_expression_in_third_quadrant_l142_142101


namespace gcd_two_powers_l142_142633

noncomputable def gcd_expression (m n : ℕ) : ℕ :=
  Int.gcd (2^m + 1) (2^n - 1)

theorem gcd_two_powers (m n : ℕ) (hm : m > 0) (hn : n > 0) (odd_n : n % 2 = 1) : 
  gcd_expression m n = 1 :=
by
  sorry

end gcd_two_powers_l142_142633


namespace arithmetic_sequence_min_value_Sn_l142_142947

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l142_142947


namespace sequence_problem_l142_142910

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l142_142910


namespace sum_reciprocal_factors_12_l142_142250

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l142_142250


namespace isosceles_triangle_perimeter_eq_10_l142_142805

theorem isosceles_triangle_perimeter_eq_10 (x : ℝ) 
(base leg : ℝ)
(h_base : base = 4)
(h_leg_root : x^2 - 5 * x + 6 = 0)
(h_iso : leg = x)
(triangle_ineq : leg + leg > base):
  2 * leg + base = 10 := 
begin
  cases (em (x = 2)) with h1 h2,
  { rw h1 at h_leg_root,
    rw [←h_iso, h1] at triangle_ineq,
    simp at triangle_ineq,
    contradiction },
  { rw h_iso,
    have : x = 3,
    { by_contra,
      simp [not_or_distrib, h1, h, sub_eq_zero] at h_leg_root },
    rw this,
    simp,
    linarith }
end

# Testing if the theorem can be evaluated successfully
# theorem_example : isosceles_triangle_perimeter_eq_10 3 4 3 rfl rfl sorry sorry rfl :=
# sorry

end isosceles_triangle_perimeter_eq_10_l142_142805


namespace sum_reciprocals_of_factors_12_l142_142375

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l142_142375


namespace train_speed_l142_142798

theorem train_speed (length_of_train time_to_cross : ℝ) (h_length : length_of_train = 800) (h_time : time_to_cross = 12) : (length_of_train / time_to_cross) = 66.67 :=
by
  sorry

end train_speed_l142_142798


namespace min_sin6_cos6_l142_142837

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l142_142837


namespace smallest_integer_y_l142_142767

theorem smallest_integer_y (y : ℤ) : (5 : ℝ) / 8 < (y : ℝ) / 17 → y = 11 := by
  sorry

end smallest_integer_y_l142_142767


namespace roberto_valid_outfits_l142_142192

-- Definitions based on the conditions
def total_trousers : ℕ := 6
def total_shirts : ℕ := 8
def total_jackets : ℕ := 4
def restricted_jacket : ℕ := 1
def restricted_shirts : ℕ := 2

-- Theorem statement
theorem roberto_valid_outfits : 
  total_trousers * total_shirts * total_jackets - total_trousers * restricted_shirts * restricted_jacket = 180 := 
by
  sorry

end roberto_valid_outfits_l142_142192


namespace sum_reciprocals_factors_12_l142_142399

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142399


namespace laptop_selection_l142_142134

open Nat

theorem laptop_selection :
  ∃ (n : ℕ), n = (choose 4 2) * (choose 5 1) + (choose 4 1) * (choose 5 2) := 
sorry

end laptop_selection_l142_142134


namespace sum_of_reciprocals_factors_12_l142_142426

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l142_142426


namespace remainder_of_c_plus_d_l142_142490

-- Definitions based on conditions
def c (k : ℕ) : ℕ := 60 * k + 53
def d (m : ℕ) : ℕ := 40 * m + 29

-- Statement of the problem
theorem remainder_of_c_plus_d (k m : ℕ) :
  ((c k + d m) % 20) = 2 :=
by
  unfold c
  unfold d
  sorry

end remainder_of_c_plus_d_l142_142490


namespace find_y_value_l142_142096

theorem find_y_value : (12 : ℕ)^3 * (6 : ℕ)^2 / 432 = 144 := by
  -- assumptions and computations are not displayed in the statement
  sorry

end find_y_value_l142_142096


namespace Area_S_inequality_l142_142546

def S (t : ℝ) (x y : ℝ) : Prop :=
  let T := Real.sin (Real.pi * t)
  |x - T| + |y - T| ≤ T

theorem Area_S_inequality (t : ℝ) :
  let T := Real.sin (Real.pi * t)
  0 ≤ 2 * T^2 := by
  sorry

end Area_S_inequality_l142_142546


namespace nancy_carrots_l142_142687

-- Definitions based on the conditions
def initial_carrots := 12
def carrots_to_cook := 2
def new_carrot_seeds := 5
def growth_factor := 3
def kept_carrots := 10
def poor_quality_ratio := 3

-- Calculate new carrots grown from seeds
def new_carrots := new_carrot_seeds * growth_factor

-- Total carrots after new ones are added
def total_carrots := kept_carrots + new_carrots

-- Calculate poor quality carrots (integer part only)
def poor_quality_carrots := total_carrots / poor_quality_ratio

-- Calculate good quality carrots
def good_quality_carrots := total_carrots - poor_quality_carrots

-- Statement to prove
theorem nancy_carrots : good_quality_carrots = 17 :=
by
  sorry -- proof is not required

end nancy_carrots_l142_142687


namespace sum_of_reciprocals_factors_12_l142_142423

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l142_142423


namespace find_larger_integer_l142_142744

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l142_142744


namespace sum_of_reciprocals_factors_12_l142_142306

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l142_142306


namespace general_formula_sum_and_min_value_l142_142685

variables {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Given conditions
def a1 := (a 1 = -5)
def a_condition := (3 * a 3 + a 5 = 0)

-- Prove the general formula for an arithmetic sequence
theorem general_formula (a1 : a 1 = -5) (a_condition : 3 * a 3 + a 5 = 0) : 
  ∀ n, a n = 2 * n - 7 := 
by
  sorry

-- Using the general formula to find the sum Sn and its minimum value
theorem sum_and_min_value (a1 : a 1 = -5) (a_condition : 3 * a 3 + a 5 = 0)
  (h : ∀ n, a n = 2 * n - 7) : 
  ∀ n, S n = n^2 - 6 * n ∧ ∃ n, S n = -9 :=
by
  sorry

end general_formula_sum_and_min_value_l142_142685


namespace min_sixth_power_sin_cos_l142_142849

theorem min_sixth_power_sin_cos (x : ℝ) : ∃ (c : ℝ), c = (1 / 4) ∧ ∀ x, (sin x)^6 + (cos x)^6 ≥ c :=
by
  sorry

end min_sixth_power_sin_cos_l142_142849


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l142_142836

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l142_142836


namespace line_through_intersection_points_of_circles_l142_142152

theorem line_through_intersection_points_of_circles :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 4*x - 4*y - 1 = 0) ∧ (x^2 + y^2 + 2*x - 13 = 0) →
    (x - 2*y + 6 = 0) :=
by
  intro x y h
  -- Condition of circle 1
  have circle1 : x^2 + y^2 + 4*x - 4*y - 1 = 0 := h.left
  -- Condition of circle 2
  have circle2 : x^2 + y^2 + 2*x - 13 = 0 := h.right
  sorry

end line_through_intersection_points_of_circles_l142_142152


namespace roots_quadratic_l142_142613

theorem roots_quadratic (a b c d : ℝ) :
  (a + b = 3 * c / 2 ∧ a * b = 4 * d ∧ c + d = 3 * a / 2 ∧ c * d = 4 * b)
  ↔ ( (a = 4 ∧ b = 8 ∧ c = 4 ∧ d = 8) ∨
      (a = -2 ∧ b = -22 ∧ c = -8 ∧ d = 11) ∨
      (a = -8 ∧ b = 2 ∧ c = -2 ∧ d = -4) ) :=
by
  sorry

end roots_quadratic_l142_142613


namespace Ricardo_coin_difference_l142_142696

theorem Ricardo_coin_difference (p : ℕ) (h₁ : 1 ≤ p) (h₂ : p ≤ 3029) :
    let max_value := 15150 - 4 * 1
    let min_value := 15150 - 4 * 3029
    max_value - min_value = 12112 := by
  sorry

end Ricardo_coin_difference_l142_142696


namespace sum_of_reciprocals_of_factors_of_12_l142_142377

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142377


namespace sum_reciprocal_factors_12_l142_142252

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l142_142252


namespace infinitely_many_arithmetic_sequences_l142_142037

theorem infinitely_many_arithmetic_sequences (x : ℕ) (hx : 0 < x) :
  ∃ y z : ℕ, y = 5 * x + 2 ∧ z = 7 * x + 3 ∧ x * (x + 1) < y * (y + 1) ∧ y * (y + 1) < z * (z + 1) ∧
  y * (y + 1) - x * (x + 1) = z * (z + 1) - y * (y + 1) :=
by
  sorry

end infinitely_many_arithmetic_sequences_l142_142037


namespace graph_passes_through_point_l142_142979

theorem graph_passes_through_point (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : 
    ∃ y : ℝ, y = a^0 + 1 ∧ y = 2 :=
by
  use 2
  simp
  sorry

end graph_passes_through_point_l142_142979


namespace no_intersection_at_roots_l142_142646

theorem no_intersection_at_roots {f g : ℝ → ℝ} (h : ∀ x, f x = x ∧ g x = x - 3) :
  ¬ (∃ x, (x = 0 ∨ x = 3) ∧ (f x = g x)) :=
by
  intros 
  sorry

end no_intersection_at_roots_l142_142646


namespace sum_reciprocals_12_l142_142243

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l142_142243


namespace incorrect_statement_isosceles_trapezoid_l142_142775

-- Define the properties of an isosceles trapezoid
structure IsoscelesTrapezoid (a b c d : ℝ) :=
  (parallel_bases : a = c ∨ b = d)  -- Bases are parallel
  (equal_diagonals : a = b) -- Diagonals are equal
  (equal_angles : ∀ α β : ℝ, α = β)  -- Angles on the same base are equal
  (axisymmetric : ∀ x : ℝ, x = -x)  -- Is an axisymmetric figure

-- Prove that the statement "The two bases of an isosceles trapezoid are parallel and equal" is incorrect
theorem incorrect_statement_isosceles_trapezoid (a b c d : ℝ) (h : IsoscelesTrapezoid a b c d) :
  ¬ (a = c ∧ b = d) :=
sorry

end incorrect_statement_isosceles_trapezoid_l142_142775


namespace exists_a_satisfying_inequality_l142_142127

theorem exists_a_satisfying_inequality (x : ℝ) : 
  x < -2 ∨ (0 < x ∧ x < 1) ∨ 1 < x → 
  ∃ a ∈ Set.Icc (-1 : ℝ) 2, (2 - a) * x^3 + (1 - 2 * a) * x^2 - 6 * x + 5 + 4 * a - a^2 < 0 := 
by 
  intros h
  sorry

end exists_a_satisfying_inequality_l142_142127


namespace bruce_money_left_l142_142476

theorem bruce_money_left :
  let initial_amount := 71
  let cost_per_shirt := 5
  let number_of_shirts := 5
  let cost_of_pants := 26
  let total_cost := number_of_shirts * cost_per_shirt + cost_of_pants
  let money_left := initial_amount - total_cost
  money_left = 20 :=
by
  sorry

end bruce_money_left_l142_142476


namespace find_k_value_l142_142145

theorem find_k_value (k : ℝ) :
  (∃ x1 x2 : ℝ, (2 * x1^2 + k * x1 - 2 * k + 1 = 0) ∧ 
                (2 * x2^2 + k * x2 - 2 * k + 1 = 0) ∧ 
                (x1 ≠ x2)) ∧
  ((x1^2 + x2^2 = 29/4)) ↔ (k = 3) := 
sorry

end find_k_value_l142_142145


namespace sum_of_reciprocals_factors_12_l142_142392

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l142_142392


namespace jorge_total_goals_l142_142680

theorem jorge_total_goals (last_season_goals current_season_goals : ℕ) (h_last : last_season_goals = 156) (h_current : current_season_goals = 187) : 
  last_season_goals + current_season_goals = 343 :=
by
  sorry

end jorge_total_goals_l142_142680


namespace range_of_a_l142_142526

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x - 1 ≤ 0) → a ≤ -1 :=
sorry

end range_of_a_l142_142526


namespace sum_of_reciprocals_factors_12_l142_142301

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l142_142301


namespace ratio_of_books_sold_l142_142032

theorem ratio_of_books_sold
  (T W R : ℕ)
  (hT : T = 7)
  (hW : W = 3 * T)
  (hTotal : T + W + R = 91) :
  R / W = 3 :=
by
  sorry

end ratio_of_books_sold_l142_142032


namespace sum_reciprocals_12_l142_142235

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l142_142235


namespace product_gcd_lcm_l142_142505

theorem product_gcd_lcm (a b : ℕ) (ha : a = 90) (hb : b = 150) :
  Nat.gcd a b * Nat.lcm a b = 13500 := by
  sorry

end product_gcd_lcm_l142_142505


namespace just_passed_students_l142_142538

theorem just_passed_students (total_students : ℕ) 
  (math_first_division_perc : ℕ) 
  (math_second_division_perc : ℕ)
  (eng_first_division_perc : ℕ)
  (eng_second_division_perc : ℕ)
  (sci_first_division_perc : ℕ)
  (sci_second_division_perc : ℕ) 
  (math_just_passed : ℕ)
  (eng_just_passed : ℕ)
  (sci_just_passed : ℕ) :
  total_students = 500 →
  math_first_division_perc = 35 →
  math_second_division_perc = 48 →
  eng_first_division_perc = 25 →
  eng_second_division_perc = 60 →
  sci_first_division_perc = 40 →
  sci_second_division_perc = 45 →
  math_just_passed = (100 - (math_first_division_perc + math_second_division_perc)) * total_students / 100 →
  eng_just_passed = (100 - (eng_first_division_perc + eng_second_division_perc)) * total_students / 100 →
  sci_just_passed = (100 - (sci_first_division_perc + sci_second_division_perc)) * total_students / 100 →
  math_just_passed = 85 ∧ eng_just_passed = 75 ∧ sci_just_passed = 75 :=
by
  intros ht hf1 hf2 he1 he2 hs1 hs2 hjm hje hjs
  sorry

end just_passed_students_l142_142538


namespace percent_calculation_l142_142448

theorem percent_calculation (x : ℝ) : 
  (∃ y : ℝ, y / 100 * x = 0.3 * 0.7 * x) → ∃ y : ℝ, y = 21 :=
by
  sorry

end percent_calculation_l142_142448


namespace no_integer_solutions_l142_142499

theorem no_integer_solutions :
  ¬ (∃ a b : ℤ, 3 * a^2 = b^2 + 1) :=
by 
  sorry

end no_integer_solutions_l142_142499


namespace max_value_y_on_interval_l142_142568

noncomputable def y (x: ℝ) : ℝ := x^4 - 8 * x^2 + 2

theorem max_value_y_on_interval : 
  ∃ x ∈ Set.Icc (-1 : ℝ) (3 : ℝ), y x = 11 ∧ ∀ z ∈ Set.Icc (-1 : ℝ) (3 : ℝ), y z ≤ 11 := 
sorry

end max_value_y_on_interval_l142_142568


namespace larger_integer_21_l142_142715

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l142_142715


namespace sum_of_reciprocals_of_factors_of_12_l142_142313

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142313


namespace sum_reciprocals_factors_12_l142_142261

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l142_142261


namespace count_sums_of_three_cubes_l142_142872

theorem count_sums_of_three_cubes :
  let possible_sums := {n | ∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ n = a^3 + b^3 + c^3}
  ∃ unique_sums : Finset ℕ, (∀ x ∈ possible_sums, x < 1000) ∧ unique_sums.card = 153 :=
by sorry

end count_sums_of_three_cubes_l142_142872


namespace sum_reciprocals_12_l142_142234

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l142_142234


namespace arithmetic_sequence_min_value_S_l142_142907

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry -- definition from condition of arithmetic sequence

theorem arithmetic_sequence (S a : ℕ → ℤ) (h1 : ∀ n, (2 * S n / n + n = 2 * a n + 1)) :
  ∀ n, a (n + 1) = a n + 1 :=
sorry

theorem min_value_S (S a : ℕ → ℤ) (h2 : a 4 = -9 ∧ a 7 = -6 ∧ a 9 = -4) :
  ∀ n, S n = (n → ℤ) → -78 :=
sorry

end arithmetic_sequence_min_value_S_l142_142907


namespace sum_of_reciprocals_of_factors_of_12_l142_142320

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142320


namespace factorize_quadratic_l142_142493

theorem factorize_quadratic (x : ℝ) : 2 * x^2 + 12 * x + 18 = 2 * (x + 3)^2 :=
by
  sorry

end factorize_quadratic_l142_142493


namespace larger_integer_value_l142_142718

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l142_142718


namespace digit_d_makes_multiple_of_9_l142_142628

theorem digit_d_makes_multiple_of_9 :
  ∃ d : ℕ, d < 10 ∧ (26 + d) % 9 = 0 ∧ d = 1 :=
by {
  have h1 : 26 % 9 = 8 := rfl,
  use 1,
  split,
  { linarith },
  split,
  { norm_num },
  { refl }
}

end digit_d_makes_multiple_of_9_l142_142628


namespace isosceles_triangle_perimeter_l142_142804

theorem isosceles_triangle_perimeter {a : ℝ} (h_base : 4 ≠ 0) (h_roots : a^2 - 5 * a + 6 = 0) :
  a = 3 → (4 + 2 * a = 10) :=
by
  sorry

end isosceles_triangle_perimeter_l142_142804


namespace triangle_product_l142_142894

theorem triangle_product (a b c: ℕ) (p: ℕ)
    (h1: ∃ k1 k2 k3: ℕ, a * k1 * k2 = p ∧ k2 * k3 * b = p ∧ k3 * c * a = p) 
    : (1 ≤ c ∧ c ≤ 336) :=
by
  sorry

end triangle_product_l142_142894


namespace loss_percentage_l142_142800

theorem loss_percentage (CP SP_gain L : ℝ) 
  (h1 : CP = 1500)
  (h2 : SP_gain = CP + 0.05 * CP)
  (h3 : SP_gain = CP - (L/100) * CP + 225) : 
  L = 10 :=
by
  sorry

end loss_percentage_l142_142800


namespace larger_integer_is_21_l142_142735

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l142_142735


namespace _l142_142548

variable {Ω : Type*} [ProbabilitySpace Ω]

def Cramers_theorem (ξ η : Ω → ℝ) (hξη : Indep ξ η) : 
  (IsGaussian (ξ + η) ↔ IsGaussian ξ ∧ IsGaussian η) :=
sorry

end _l142_142548


namespace solution_l142_142531

theorem solution (a b : ℝ) (h1 : a^2 + 2 * a - 2016 = 0) (h2 : b^2 + 2 * b - 2016 = 0) :
  a^2 + 3 * a + b = 2014 := 
sorry

end solution_l142_142531


namespace larger_integer_is_21_l142_142727

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l142_142727


namespace sum_reciprocals_factors_12_l142_142263

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l142_142263


namespace sufficient_but_not_necessary_condition_l142_142035

theorem sufficient_but_not_necessary_condition :
  (∀ (x : ℝ), x = 1 → x^2 - 3 * x + 2 = 0) ∧ ¬(∀ (x : ℝ), x^2 - 3 * x + 2 = 0 → x = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l142_142035


namespace sum_of_reciprocals_of_factors_of_12_l142_142385

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142385


namespace first_divisor_l142_142053

theorem first_divisor (y : ℝ) (x : ℝ) (h1 : 320 / (y * 3) = x) (h2 : x = 53.33) : y = 2 :=
sorry

end first_divisor_l142_142053


namespace min_value_sin_cos_l142_142828

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l142_142828


namespace upper_limit_of_range_l142_142769

theorem upper_limit_of_range (n : ℕ) (h : (10 + 10 * n) / 2 = 255) : 10 * n = 500 :=
by 
  sorry

end upper_limit_of_range_l142_142769


namespace partition_property_l142_142027

open Finset

theorem partition_property (k : ℕ) (hk : 0 < k) :
  ∃ (X Y : Finset ℕ), disjoint X Y ∧
  X ∪ Y = range (2^(k+1)) ∧
  (∀ m ∈ (range (k+1)).erase 0, ∑ x in X, x^m = ∑ y in Y, y^m) :=
sorry

end partition_property_l142_142027


namespace prove_arithmetic_sequence_minimum_value_S_l142_142934

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l142_142934


namespace milk_revenue_l142_142965

theorem milk_revenue :
  let yesterday_morning := 68
  let yesterday_evening := 82
  let this_morning := yesterday_morning - 18
  let total_milk_before_selling := yesterday_morning + yesterday_evening + this_morning
  let milk_left := 24
  let milk_sold := total_milk_before_selling - milk_left
  let cost_per_gallon := 3.50
  let revenue := milk_sold * cost_per_gallon
  revenue = 616 := by {
    sorry
}

end milk_revenue_l142_142965


namespace factorization_eq1_factorization_eq2_l142_142492

-- Definitions for the given conditions
variables (a b x y m : ℝ)

-- The problem statement as Lean definitions and the goal theorems
def expr1 : ℝ := -6 * a * b + 3 * a^2 + 3 * b^2
def factored1 : ℝ := 3 * (a - b)^2

def expr2 : ℝ := y^2 * (2 - m) + x^2 * (m - 2)
def factored2 : ℝ := (m - 2) * (x + y) * (x - y)

-- Theorem statements for equivalence
theorem factorization_eq1 : expr1 a b = factored1 a b :=
by
  sorry

theorem factorization_eq2 : expr2 x y m = factored2 x y m :=
by
  sorry

end factorization_eq1_factorization_eq2_l142_142492


namespace arcade_game_monster_perimeter_l142_142015

theorem arcade_game_monster_perimeter :
  let r := 1 -- radius of the circle in cm
  let theta := 60 -- central angle of the missing sector in degrees
  let circumference := 2 * Real.pi * r -- circumference of the full circle
  let arc_fraction := (360 - theta) / 360 -- fraction of the circle forming the arc
  let arc_length := arc_fraction * circumference -- length of the arc
  let perimeter := arc_length + 2 * r -- total perimeter (arc + two radii)
  perimeter = (5 / 3) * Real.pi + 2 :=
by
  sorry

end arcade_game_monster_perimeter_l142_142015


namespace sum_reciprocals_of_factors_of_12_l142_142331

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l142_142331


namespace movie_marathon_duration_l142_142994

theorem movie_marathon_duration :
  let first_movie := 2
  let second_movie := first_movie + 0.5 * first_movie
  let combined_time := first_movie + second_movie
  let third_movie := combined_time - 1
  first_movie + second_movie + third_movie = 9 := by
  sorry

end movie_marathon_duration_l142_142994


namespace min_value_sin6_cos6_l142_142833

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l142_142833


namespace xiaohua_amount_paid_l142_142606

def cost_per_bag : ℝ := 18
def discount_rate : ℝ := 0.1
def price_difference : ℝ := 36

theorem xiaohua_amount_paid (x : ℝ) 
  (h₁ : 18 * (x+1) * (1 - 0.1) = 18 * x - 36) :
  18 * (x + 1) * (1 - 0.1) = 486 := 
sorry

end xiaohua_amount_paid_l142_142606


namespace find_larger_integer_l142_142748

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l142_142748


namespace find_four_digit_number_l142_142617

theorem find_four_digit_number :
  ∃ (N : ℕ), 1000 ≤ N ∧ N < 10000 ∧ 
    (N % 131 = 112) ∧ 
    (N % 132 = 98) ∧ 
    N = 1946 :=
by
  sorry

end find_four_digit_number_l142_142617


namespace mosquito_drops_per_feed_l142_142603

-- Defining the constants and conditions.
def drops_per_liter : ℕ := 5000
def liters_to_die : ℕ := 3
def mosquitoes_to_kill : ℕ := 750

-- The assertion we want to prove.
theorem mosquito_drops_per_feed :
  (drops_per_liter * liters_to_die) / mosquitoes_to_kill = 20 :=
by
  sorry

end mosquito_drops_per_feed_l142_142603


namespace sum_reciprocals_factors_12_l142_142338

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142338


namespace sum_of_reciprocals_factors_12_l142_142308

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l142_142308


namespace sum_reciprocals_factors_12_l142_142341

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142341


namespace total_participating_students_l142_142794

-- Define the given conditions
def field_events_participants : ℕ := 15
def track_events_participants : ℕ := 13
def both_events_participants : ℕ := 5

-- Define the total number of students calculation
def total_students_participating : ℕ :=
  (field_events_participants - both_events_participants) + 
  (track_events_participants - both_events_participants) + 
  both_events_participants

-- State the theorem that needs to be proved
theorem total_participating_students : total_students_participating = 23 := by
  sorry

end total_participating_students_l142_142794


namespace cost_effectiveness_l142_142690

-- Define general parameters and conditions given in the problem
def a : ℕ := 70 -- We use 70 since it must be greater than 50

-- Define the scenarios
def cost_scenario1 (a: ℕ) : ℕ := 4500 + 27 * a
def cost_scenario2 (a: ℕ) : ℕ := 4400 + 30 * a

-- The theorem to be proven
theorem cost_effectiveness (h : a > 50) : cost_scenario1 a < cost_scenario2 a :=
  by
  -- First, let's replace a with 70 (this step is unnecessary in the proof since a = 70 is fixed)
  let a := 70
  -- Now, prove the inequality
  sorry

end cost_effectiveness_l142_142690


namespace direct_proportion_solution_l142_142138

theorem direct_proportion_solution (m : ℝ) (h1 : m + 3 ≠ 0) (h2 : m^2 - 8 = 1) : m = 3 :=
sorry

end direct_proportion_solution_l142_142138


namespace prob_A_and_B_l142_142983

open MeasureTheory

-- Define the probability space
variable (Ω : Type*) [MeasurableSpace Ω] (P : MeasureTheory.Measure Ω)

-- Define events A and B
variables (A B : Set Ω)

-- Given conditions
axiom prob_B : P B = 0.4
axiom prob_A_or_B : P (A ∪ B) = 0.6
axiom prob_A : P A = 0.45

-- The goal is to prove P (A ∩ B) = 0.25
theorem prob_A_and_B : P (A ∩ B) = 0.25 :=
by
  have h1 : P (A ∪ B) = P A + P B - P (A ∩ B) := sorry -- Inclusion-exclusion principle
  have h2 : 0.6 = 0.45 + 0.4 - P (A ∩ B) := sorry -- Substitute known values
  have h3 : P (A ∩ B) = 0.25 := sorry -- Solve the equation
  exact h3

end prob_A_and_B_l142_142983


namespace smallest_integer_value_l142_142585

theorem smallest_integer_value (x : ℤ) (h : 7 - 3 * x < 22) : x ≥ -4 := 
sorry

end smallest_integer_value_l142_142585


namespace find_initial_children_l142_142560

-- Definition of conditions
def initial_children_on_bus (X : ℕ) := 
  let final_children := (X + 40) - 60 
  final_children = 2

-- Theorem statement
theorem find_initial_children : 
  ∃ X : ℕ, initial_children_on_bus X ∧ X = 22 :=
by
  sorry

end find_initial_children_l142_142560


namespace partition_sum_condition_l142_142036

theorem partition_sum_condition (X : Finset ℕ) (hX : X = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ∀ (A B : Finset ℕ), A ∪ B = X → A ∩ B = ∅ →
  ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a + b = c := 
by
  -- sorry is here to acknowledge that no proof is required per instructions.
  sorry

end partition_sum_condition_l142_142036


namespace melanie_marbles_l142_142963

noncomputable def melanie_blue_marbles : ℕ :=
  let sandy_dozen_marbles := 56
  let dozen := 12
  let sandy_marbles := sandy_dozen_marbles * dozen
  let ratio := 8
  sandy_marbles / ratio

theorem melanie_marbles (h1 : ∀ sandy_dozen_marbles dozen ratio, 56 = sandy_dozen_marbles ∧ sandy_dozen_marbles * dozen = 672 ∧ ratio = 8) : melanie_blue_marbles = 84 := by
  sorry

end melanie_marbles_l142_142963


namespace larger_integer_is_21_l142_142728

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l142_142728


namespace darry_small_ladder_climbs_l142_142119

-- Define the constants based on the conditions
def full_ladder_steps := 11
def full_ladder_climbs := 10
def small_ladder_steps := 6
def total_steps := 152

-- Darry's total steps climbed via full ladder
def full_ladder_total_steps := full_ladder_steps * full_ladder_climbs

-- Define x as the number of times Darry climbed the smaller ladder
variable (x : ℕ)

-- Prove that x = 7 given the conditions
theorem darry_small_ladder_climbs (h : full_ladder_total_steps + small_ladder_steps * x = total_steps) : x = 7 :=
by 
  sorry

end darry_small_ladder_climbs_l142_142119


namespace sufficient_condition_for_parallel_lines_l142_142180

-- Define the condition for lines to be parallel
def lines_parallel (a b c d e f : ℝ) : Prop :=
(∃ k : ℝ, a = k * c ∧ b = k * d)

-- Define the specific lines given in the problem
def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + y - 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := x + a * y + 5

theorem sufficient_condition_for_parallel_lines (a : ℝ) :
  (lines_parallel (a) (1) (-1) (1) (-1) (1 + 5)) ↔ (a = -1) :=
sorry

end sufficient_condition_for_parallel_lines_l142_142180


namespace path_area_and_cost_correct_l142_142793

def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.8
def area_of_path : ℝ := 759.36
def cost_per_sqm : ℝ := 2
def total_cost : ℝ := 1518.72

theorem path_area_and_cost_correct :
    let length_with_path := length_field + 2 * path_width
    let width_with_path := width_field + 2 * path_width
    let area_with_path := length_with_path * width_with_path
    let area_field := length_field * width_field
    let calculated_area_of_path := area_with_path - area_field
    let calculated_total_cost := calculated_area_of_path * cost_per_sqm
    calculated_area_of_path = area_of_path ∧ calculated_total_cost = total_cost :=
by
    sorry

end path_area_and_cost_correct_l142_142793


namespace positive_divisors_840_multiple_of_4_l142_142530

theorem positive_divisors_840_multiple_of_4 :
  let n := 840
  let prime_factors := (2^3 * 3^1 * 5^1 * 7^1)
  (∀ k : ℕ, k ∣ n → k % 4 = 0 → ∀ a b c d : ℕ, 2 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 →
  k = 2^a * 3^b * 5^c * 7^d) → 
  (∃ count, count = 16) :=
by {
  sorry
}

end positive_divisors_840_multiple_of_4_l142_142530


namespace sum_of_reciprocals_of_factors_of_12_l142_142386

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142386


namespace larger_integer_is_21_l142_142729

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l142_142729


namespace roots_of_polynomial_l142_142125

def p (x : ℝ) : ℝ := x^3 + x^2 - 4*x - 4

theorem roots_of_polynomial :
  (p (-1) = 0) ∧ (p 2 = 0) ∧ (p (-2) = 0) ∧ 
  ∀ x, p x = 0 → (x = -1 ∨ x = 2 ∨ x = -2) :=
by
  sorry

end roots_of_polynomial_l142_142125


namespace eagles_win_at_least_three_matches_l142_142200

-- Define the conditions
def n : ℕ := 5
def p : ℝ := 0.5

-- Binomial coefficient function
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probability function for the binomial distribution
noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial n k) * p^k * (1 - p)^(n - k)

-- Theorem stating the main result
theorem eagles_win_at_least_three_matches :
  (binomial_prob n 3 p + binomial_prob n 4 p + binomial_prob n 5 p) = 1 / 2 :=
by
  sorry

end eagles_win_at_least_three_matches_l142_142200


namespace gcd_lcm_product_l142_142503

theorem gcd_lcm_product (a b : ℕ) (ha : a = 90) (hb : b = 150) : 
  Nat.gcd a b * Nat.lcm a b = 13500 := 
by 
  sorry

end gcd_lcm_product_l142_142503


namespace each_person_pays_50_97_l142_142985

noncomputable def total_bill (original_bill : ℝ) (tip_percentage : ℝ) : ℝ :=
  original_bill + original_bill * tip_percentage

noncomputable def amount_per_person (total_bill : ℝ) (num_people : ℕ) : ℝ :=
  total_bill / num_people

theorem each_person_pays_50_97 :
  let original_bill := 139.00
  let number_of_people := 3
  let tip_percentage := 0.10
  let expected_amount := 50.97
  abs (amount_per_person (total_bill original_bill tip_percentage) number_of_people - expected_amount) < 0.01
:= sorry

end each_person_pays_50_97_l142_142985


namespace hamburger_cost_l142_142809

def annie's_starting_money : ℕ := 120
def num_hamburgers_bought : ℕ := 8
def price_milkshake : ℕ := 3
def num_milkshakes_bought : ℕ := 6
def leftover_money : ℕ := 70

theorem hamburger_cost :
  ∃ (H : ℕ), 8 * H + 6 * price_milkshake = annie's_starting_money - leftover_money ∧ H = 4 :=
by
  use 4
  sorry

end hamburger_cost_l142_142809


namespace coefficient_a2_l142_142865

theorem coefficient_a2 :
  ∀ (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ),
  (x^10 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + 
  a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + 
  a_7 * (x + 1)^7 + a_8 * (x + 1)^8 + a_9 * (x + 1)^9 + 
  a_10 * (x + 1)^10) →
  a_2 = 45 :=
by
  sorry

end coefficient_a2_l142_142865


namespace sum_of_reciprocals_of_factors_of_12_l142_142291

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142291


namespace platform_length_l142_142786

theorem platform_length (train_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) 
  (h_train_length : train_length = 300) (h_time_pole : time_pole = 12) (h_time_platform : time_platform = 39) : 
  ∃ L : ℕ, L = 675 :=
by
  sorry

end platform_length_l142_142786


namespace sum_reciprocal_factors_of_12_l142_142410

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l142_142410


namespace inequality_D_no_solution_l142_142471

theorem inequality_D_no_solution :
  ¬ ∃ x : ℝ, 2 - 3 * x + 2 * x^2 ≤ 0 := 
sorry

end inequality_D_no_solution_l142_142471


namespace parabola_ratio_l142_142869

noncomputable def ratio_AF_BF (p : ℝ) (h_pos : p > 0) : ℝ :=
  let y1 := (Real.sqrt (2 * p * (3 / 2 * p)))
  let y2 := (Real.sqrt (2 * p * (1 / 6 * p)))
  let dist1 := Real.sqrt ((3 / 2 * p - (p / 2))^2 + y1^2)
  let dist2 := Real.sqrt ((1 / 6 * p - p / 2)^2 + y2^2)
  dist1 / dist2

theorem parabola_ratio (p : ℝ) (h_pos : p > 0) : ratio_AF_BF p h_pos = 3 :=
  sorry

end parabola_ratio_l142_142869


namespace simplify_and_evaluate_l142_142196

def my_expression (x : ℝ) := (x + 2) * (x - 2) + 3 * (1 - x)

theorem simplify_and_evaluate : 
  my_expression (Real.sqrt 2) = 1 - 3 * Real.sqrt 2 := by
    sorry

end simplify_and_evaluate_l142_142196


namespace sourav_distance_l142_142088

def D (t : ℕ) : ℕ := 20 * t

theorem sourav_distance :
  ∀ (t : ℕ), 20 * t = 25 * (t - 1) → 20 * t = 100 :=
by
  intros t h
  sorry

end sourav_distance_l142_142088


namespace max_possible_value_l142_142542

-- Define the expressions and the conditions
def expr1 := 10 * 10
def expr2 := 10 / 10
def expr3 := expr1 + 10
def expr4 := expr3 - expr2

-- Define our main statement that asserts the maximum value is 109
theorem max_possible_value: expr4 = 109 := by
  sorry

end max_possible_value_l142_142542


namespace larger_integer_value_l142_142716

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l142_142716


namespace perpendicular_chords_square_sum_l142_142012

theorem perpendicular_chords_square_sum (d : ℝ) (r : ℝ) (x y : ℝ) 
  (h1 : r = d / 2)
  (h2 : x = r)
  (h3 : y = r) 
  : (x^2 + y^2) + (x^2 + y^2) = d^2 :=
by
  sorry

end perpendicular_chords_square_sum_l142_142012


namespace sum_of_reciprocals_factors_12_l142_142389

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l142_142389


namespace sum_reciprocals_factors_12_l142_142358

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l142_142358


namespace geometric_sequence_common_ratio_l142_142524

-- Define the geometric sequence with properties
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n < a (n + 1)

-- Main theorem
theorem geometric_sequence_common_ratio {a : ℕ → ℝ} {q : ℝ} (h_seq : increasing_geometric_sequence a q) (h_a1 : a 0 > 0) (h_eqn : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l142_142524


namespace find_m_range_l142_142136

-- Define the mathematical objects and conditions
def condition_p (m : ℝ) : Prop :=
  (|1 - m| / Real.sqrt 2) > 1

def condition_q (m : ℝ) : Prop :=
  m < 4

-- Define the proof problem
theorem find_m_range (p q : Prop) (m : ℝ) 
  (hp : ¬ p) (hq : q) (hpq : p ∨ q)
  (hP_imp : p → condition_p m)
  (hQ_imp : q → condition_q m) : 
  1 - Real.sqrt 2 ≤ m ∧ m ≤ 1 + Real.sqrt 2 := 
sorry

end find_m_range_l142_142136


namespace cubic_roots_identity_l142_142026

theorem cubic_roots_identity (p q r : ℝ) 
  (h1 : p + q + r = 0) 
  (h2 : p * q + q * r + r * p = -3) 
  (h3 : p * q * r = -2) : 
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = 0 := 
by
  sorry

end cubic_roots_identity_l142_142026


namespace first_place_beat_joe_l142_142676

theorem first_place_beat_joe (joe_won joe_draw first_place_won first_place_draw points_win points_draw : ℕ) 
    (h1 : joe_won = 1) (h2 : joe_draw = 3) (h3 : first_place_won = 2) (h4 : first_place_draw = 2)
    (h5 : points_win = 3) (h6 : points_draw = 1) : 
    (first_place_won * points_win + first_place_draw * points_draw) - (joe_won * points_win + joe_draw * points_draw) = 2 :=
by
   sorry

end first_place_beat_joe_l142_142676


namespace square_diagonal_y_coordinate_l142_142540

theorem square_diagonal_y_coordinate 
(point_vertex : ℝ × ℝ) 
(x_int : ℝ) 
(area_square : ℝ) 
(y_int : ℝ) :
(point_vertex = (-6, -4)) →
(x_int = 3) →
(area_square = 324) →
(y_int = 5) → 
y_int = 5 := 
by
  intros h1 h2 h3 h4
  exact h4

end square_diagonal_y_coordinate_l142_142540


namespace sum_reciprocals_of_factors_of_12_l142_142322

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l142_142322


namespace sufficient_condition_range_a_l142_142137

theorem sufficient_condition_range_a (a : ℝ) :
  (∀ x, (2 * a ≤ x ∧ x ≤ a^2 + 1) → (x^2 - 3 * (a + 1) * x + 6 * a + 2 ≤ 0)) ↔
  (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) := by
  sorry

end sufficient_condition_range_a_l142_142137


namespace total_books_l142_142684

def keith_books : ℕ := 20
def jason_books : ℕ := 21

theorem total_books : keith_books + jason_books = 41 :=
by
  sorry

end total_books_l142_142684


namespace tank_capacity_l142_142090

theorem tank_capacity (C : ℝ) (h1 : 0.40 * C = 0.90 * C - 36) : C = 72 := 
sorry

end tank_capacity_l142_142090


namespace sin_cos_sixth_min_l142_142852

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l142_142852


namespace common_difference_arithmetic_sequence_l142_142014

theorem common_difference_arithmetic_sequence
  (a : ℕ) (d : ℚ) (n : ℕ) (a_n : ℕ) (S_n : ℕ)
  (h1 : a = 2)
  (h2 : a_n = 20)
  (h3 : S_n = 132)
  (h4 : a_n = a + (n - 1) * d)
  (h5 : S_n = n * (a + a_n) / 2) :
  d = 18 / 11 := sorry

end common_difference_arithmetic_sequence_l142_142014


namespace knives_more_than_forks_l142_142214

variable (F K S T : ℕ)
variable (x : ℕ)

-- Initial conditions
def initial_conditions : Prop :=
  (F = 6) ∧ 
  (K = F + x) ∧ 
  (S = 2 * K) ∧
  (T = F / 2)

-- Total cutlery added
def total_cutlery_added : Prop :=
  (F + 2) + (K + 2) + (S + 2) + (T + 2) = 62

-- Prove that x = 9
theorem knives_more_than_forks :
  initial_conditions F K S T x →
  total_cutlery_added F K S T →
  x = 9 := 
by
  sorry

end knives_more_than_forks_l142_142214


namespace sum_reciprocal_factors_of_12_l142_142419

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l142_142419


namespace f_prime_neg_one_l142_142867

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f x = f (-x)
axiom h2 : ∀ x : ℝ, f (x + 1) - f (1 - x) = 2 * x

theorem f_prime_neg_one : f' (-1) = -1 := by
  -- The proof is omitted
  sorry

end f_prime_neg_one_l142_142867


namespace rectangle_area_l142_142575

theorem rectangle_area (w l : ℕ) (h1 : l = 15) (h2 : (2 * l + 2 * w) / w = 5) : l * w = 150 :=
by
  -- We provide the conditions in the theorem's signature:
  -- l is the length which is 15 cm, given by h1
  -- The ratio of the perimeter to the width is 5:1, given by h2
  sorry

end rectangle_area_l142_142575


namespace evaluate_expression_l142_142005

theorem evaluate_expression (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 
  (x^5 + 3*y^2 + 7) / (x + 4) = 298 / 7 := by
  sorry

end evaluate_expression_l142_142005


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l142_142835

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l142_142835


namespace sum_reciprocals_factors_12_l142_142362

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l142_142362


namespace sum_of_reciprocals_of_factors_of_12_l142_142321

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142321


namespace sum_of_reciprocals_factors_12_l142_142421

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l142_142421


namespace zookeeper_fish_excess_l142_142054

theorem zookeeper_fish_excess :
  let emperor_ratio := 3
  let adelie_ratio := 5
  let total_penguins := 48
  let total_ratio := emperor_ratio + adelie_ratio
  let emperor_penguins := (emperor_ratio / total_ratio) * total_penguins
  let adelie_penguins := (adelie_ratio / total_ratio) * total_penguins
  let emperor_fish_needed := emperor_penguins * 1.5
  let adelie_fish_needed := adelie_penguins * 2
  let total_fish_needed := emperor_fish_needed + adelie_fish_needed
  let fish_zookeeper_has := total_penguins * 2.5
  (fish_zookeeper_has - total_fish_needed = 33) :=
  
by {
  sorry
}

end zookeeper_fish_excess_l142_142054


namespace sum_of_reciprocals_of_factors_of_12_l142_142438

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142438


namespace expression_equals_eight_l142_142577

theorem expression_equals_eight
  (a b c : ℝ)
  (h1 : a + b = 2 * c)
  (h2 : b + c = 2 * a)
  (h3 : a + c = 2 * b) :
  (a + b) * (b + c) * (a + c) / (a * b * c) = 8 := by
  sorry

end expression_equals_eight_l142_142577


namespace rest_area_location_l142_142205

theorem rest_area_location :
  ∃ (rest_area : ℝ), rest_area = 35 + (95 - 35) / 2 :=
by
  -- Here we set the variables for the conditions
  let fifth_exit := 35
  let seventh_exit := 95
  let rest_area := 35 + (95 - 35) / 2
  use rest_area
  sorry

end rest_area_location_l142_142205


namespace right_triangle_area_l142_142146

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := 1 / 2 * a * b

theorem right_triangle_area {a b : ℝ} 
  (h1 : a + b = 4) 
  (h2 : a^2 + b^2 = 14) : 
  area_of_right_triangle a b = 1 / 2 :=
by 
  sorry

end right_triangle_area_l142_142146


namespace equal_amounts_hot_and_cold_water_l142_142462

theorem equal_amounts_hot_and_cold_water (time_to_fill_cold : ℕ) (time_to_fill_hot : ℕ) (t_c : ℤ) : 
  time_to_fill_cold = 19 → 
  time_to_fill_hot = 23 → 
  t_c = 2 :=
by
  intros h_c h_h
  sorry

end equal_amounts_hot_and_cold_water_l142_142462


namespace sum_reciprocals_factors_12_l142_142277

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l142_142277


namespace minimum_value_of_expression_l142_142621

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) :
  3 * x + 5 + 2 / x^5 ≥ 10 + 3 * (2 / 5) ^ (1 / 5) := by
sorry

end minimum_value_of_expression_l142_142621


namespace geometric_sequence_r_value_l142_142657

theorem geometric_sequence_r_value (S : ℕ → ℚ) (r : ℚ) (n : ℕ) (h : n ≥ 2) (h1 : ∀ n, S n = 3^n + r) :
    r = -1 :=
sorry

end geometric_sequence_r_value_l142_142657


namespace maximum_distance_l142_142469

noncomputable def point_distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

def square_side_length := 2

def distance_condition (u v w : ℝ) : Prop := 
  u^2 + v^2 = 2 * w^2

theorem maximum_distance 
  (x y : ℝ) 
  (h1 : point_distance x y 0 0 = u) 
  (h2 : point_distance x y 2 0 = v) 
  (h3 : point_distance x y 2 2 = w)
  (h4 : distance_condition u v w) :
  ∃ (d : ℝ), d = point_distance x y 0 2 ∧ d = 2 * Real.sqrt 5 := sorry

end maximum_distance_l142_142469


namespace hyperbola_asymptotes_l142_142525

theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, (x^2)/4 - y^2 = 1) →
  (∀ x : ℝ, y = x / 2 ∨ y = -x / 2) :=
by
  intro h1
  sorry

end hyperbola_asymptotes_l142_142525


namespace sum_reciprocals_of_factors_12_l142_142372

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l142_142372


namespace total_students_in_circle_l142_142198

theorem total_students_in_circle (N : ℕ) (h1 : ∃ (students : Finset ℕ), students.card = N)
  (h2 : ∃ (a b : ℕ), a = 6 ∧ b = 16 ∧ b - a = N / 2): N = 18 :=
by
  sorry

end total_students_in_circle_l142_142198


namespace boy_actual_height_is_236_l142_142202

def actual_height (n : ℕ) (incorrect_avg correct_avg wrong_height : ℕ) : ℕ :=
  let incorrect_total := n * incorrect_avg
  let correct_total := n * correct_avg
  let diff := incorrect_total - correct_total
  wrong_height + diff

theorem boy_actual_height_is_236 :
  ∀ (n incorrect_avg correct_avg wrong_height actual_height : ℕ),
  n = 35 → 
  incorrect_avg = 183 → 
  correct_avg = 181 → 
  wrong_height = 166 → 
  actual_height = wrong_height + (n * incorrect_avg - n * correct_avg) →
  actual_height = 236 :=
by
  intros n incorrect_avg correct_avg wrong_height actual_height hn hic hg hw ha
  rw [hn, hic, hg, hw] at ha
  -- At this point, we would normally proceed to prove the statement.
  -- However, as per the requirements, we just include "sorry" to skip the proof.
  sorry

end boy_actual_height_is_236_l142_142202


namespace sum_of_reciprocals_factors_12_l142_142395

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l142_142395


namespace remainder_x_squared_mod_25_l142_142880

theorem remainder_x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 4 [ZMOD 25] :=
sorry

end remainder_x_squared_mod_25_l142_142880


namespace min_value_of_expression_l142_142029

theorem min_value_of_expression (α β : ℝ) (h : α + β = π / 2) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = 65 := 
sorry

end min_value_of_expression_l142_142029


namespace polynomial_real_roots_l142_142855

theorem polynomial_real_roots :
  ∀ x : ℝ, (x^4 - 3 * x^3 + 3 * x^2 - x - 6 = 0) ↔ (x = 3 ∨ x = 2 ∨ x = -1) := 
by
  sorry

end polynomial_real_roots_l142_142855


namespace sum_of_reciprocals_factors_12_l142_142388

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l142_142388


namespace evaluate_expression_l142_142007

theorem evaluate_expression (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (5 - x) + (5 - x) ^ 2 = 49 := 
sorry

end evaluate_expression_l142_142007


namespace Deepak_age_l142_142609

theorem Deepak_age : ∃ (A D : ℕ), (A / D = 4 / 3) ∧ (A + 6 = 26) ∧ (D = 15) :=
by
  sorry

end Deepak_age_l142_142609


namespace smallest_integer_solution_l142_142072

theorem smallest_integer_solution : ∀ x : ℤ, (x < 2 * x - 7) → (8 = x) :=
by
  sorry

end smallest_integer_solution_l142_142072


namespace find_larger_integer_l142_142737

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l142_142737


namespace calc_value_l142_142141

noncomputable def f : ℝ → ℝ := sorry 

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom non_const_zero : ∃ x : ℝ, f x ≠ 0
axiom functional_eq : ∀ x : ℝ, x * f (x + 1) = (x + 1) * f x

theorem calc_value : f (f (5 / 2)) = 0 :=
sorry

end calc_value_l142_142141


namespace sum_reciprocals_of_factors_12_l142_142373

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l142_142373


namespace sum_reciprocals_factors_12_l142_142270

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l142_142270


namespace total_points_scored_l142_142607

def num_members : ℕ := 12
def num_absent : ℕ := 4
def points_per_member : ℕ := 8

theorem total_points_scored : 
  (num_members - num_absent) * points_per_member = 64 := by
  sorry

end total_points_scored_l142_142607


namespace sum_of_reciprocals_of_factors_of_12_l142_142315

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142315


namespace no_integer_solutions_l142_142500

theorem no_integer_solutions :
  ¬ (∃ a b : ℤ, 3 * a^2 = b^2 + 1) :=
by 
  sorry

end no_integer_solutions_l142_142500


namespace ratio_equivalence_to_minutes_l142_142102

-- Define conditions and equivalence
theorem ratio_equivalence_to_minutes :
  ∀ (x : ℝ), (8 / 4 = 8 / x) → x = 4 / 60 :=
by
  intro x
  sorry

end ratio_equivalence_to_minutes_l142_142102


namespace inequality_proof_l142_142142

theorem inequality_proof (x y z : ℝ) (hx : -1 < x) (hy : -1 < y) (hz : -1 < z) :
    (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
sorry

end inequality_proof_l142_142142


namespace time_to_run_round_square_field_l142_142778

theorem time_to_run_round_square_field
  (side : ℝ) (speed_km_hr : ℝ)
  (h_side : side = 45)
  (h_speed_km_hr : speed_km_hr = 9) : 
  (4 * side / (speed_km_hr * 1000 / 3600)) = 72 := 
by 
  sorry

end time_to_run_round_square_field_l142_142778


namespace sum_reciprocals_factors_12_l142_142365

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l142_142365


namespace find_larger_integer_l142_142743

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l142_142743


namespace possible_values_of_m_l142_142008

theorem possible_values_of_m (m : ℝ) (A B : Set ℝ) (hA : A = {-1, 1}) (hB : B = {x | m * x = 1}) (hUnion : A ∪ B = A) : m = 0 ∨ m = 1 ∨ m = -1 :=
sorry

end possible_values_of_m_l142_142008


namespace sum_reciprocals_factors_12_l142_142265

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l142_142265


namespace num_points_P_on_ellipse_l142_142207

noncomputable def ellipse : Set (ℝ × ℝ) := {p | (p.1)^2 / 16 + (p.2)^2 / 9 = 1}
noncomputable def line : Set (ℝ × ℝ) := {p | p.1 / 4 + p.2 / 3 = 1}
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  (1 / 2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem num_points_P_on_ellipse (A B : ℝ × ℝ) 
  (hA_on_line : A ∈ line) (hA_on_ellipse : A ∈ ellipse) 
  (hB_on_line : B ∈ line) (hB_on_ellipse : B ∈ ellipse)
  : ∃ P1 P2 : ℝ × ℝ, P1 ∈ ellipse ∧ P2 ∈ ellipse ∧ 
    area_triangle A B P1 = 3 ∧ area_triangle A B P2 = 3 ∧ 
    P1 ≠ P2 ∧ 
    (∀ P : ℝ × ℝ, P ∈ ellipse ∧ area_triangle A B P = 3 → P = P1 ∨ P = P2) := 
sorry

end num_points_P_on_ellipse_l142_142207


namespace product_of_fractions_is_eight_l142_142810

theorem product_of_fractions_is_eight :
  (8 / 4) * (14 / 7) * (20 / 10) * (25 / 50) * (9 / 18) * (12 / 6) * (21 / 42) * (16 / 8) = 8 :=
by
  sorry

end product_of_fractions_is_eight_l142_142810


namespace side_length_correct_l142_142660

noncomputable def find_side_length (b : ℝ) (angleB : ℝ) (sinA : ℝ) : ℝ :=
  let sinB := Real.sin angleB
  let a := b * sinA / sinB
  a

theorem side_length_correct (b : ℝ) (angleB : ℝ) (sinA : ℝ) (a : ℝ) 
  (hb : b = 4)
  (hangleB : angleB = Real.pi / 6)
  (hsinA : sinA = 1 / 3)
  (ha : a = 8 / 3) : 
  find_side_length b angleB sinA = a :=
by
  sorry

end side_length_correct_l142_142660


namespace sum_reciprocals_factors_12_l142_142364

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l142_142364


namespace unique_pairs_of_socks_l142_142783

-- Defining the problem conditions
def pairs_socks : Nat := 3

-- The main proof statement
theorem unique_pairs_of_socks : ∃ (n : Nat), n = 3 ∧ 
  (∀ (p q : Fin 6), (p / 2 ≠ q / 2) → p ≠ q) →
  (n = (pairs_socks * (pairs_socks - 1)) / 2) :=
by
  sorry

end unique_pairs_of_socks_l142_142783


namespace sum_reciprocals_of_factors_12_l142_142370

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l142_142370


namespace sum_of_reciprocals_of_factors_of_12_l142_142290

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142290


namespace isosceles_triangle_perimeter_l142_142807

-- Define what it means to be a root of the equation x^2 - 5x + 6 = 0
def is_root (x : ℝ) : Prop := x^2 - 5 * x + 6 = 0

-- Define the perimeter based on given conditions
theorem isosceles_triangle_perimeter (x : ℝ) (base : ℝ) (h_base : base = 4) (h_root : is_root x) :
    2 * x + base = 10 :=
by
  -- Insert proof here
  sorry

end isosceles_triangle_perimeter_l142_142807


namespace negation_of_proposition_l142_142981

-- Definitions and conditions from the problem
def original_proposition (x : ℝ) : Prop := x^3 - x^2 + 1 > 0

-- The proof problem: Prove the negation
theorem negation_of_proposition : (¬ ∀ x : ℝ, original_proposition x) ↔ ∃ x : ℝ, ¬original_proposition x := 
by
  -- here we insert our proof later
  sorry

end negation_of_proposition_l142_142981


namespace candy_partition_l142_142114

theorem candy_partition :
  let candies := 10
  let boxes := 3
  ∃ ways : ℕ, ways = Nat.choose (candies + boxes - 1) (boxes - 1) ∧ ways = 66 :=
by
  let candies := 10
  let boxes := 3
  let ways := Nat.choose (candies + boxes - 1) (boxes - 1)
  have h : ways = 66 := sorry
  exact ⟨ways, ⟨rfl, h⟩⟩

end candy_partition_l142_142114


namespace sin_cos_sixth_min_l142_142853

theorem sin_cos_sixth_min (x : ℝ) : 
  sin x ^ 2 + cos x ^ 2 = 1 → sin x ^ 6 + cos x ^ 6 ≥ 3 / 4 := 
by 
  intro h
  -- proof goes here
  sorry

end sin_cos_sixth_min_l142_142853


namespace sum_reciprocals_factors_12_l142_142409

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142409


namespace trig_identity_solution_l142_142094

noncomputable def solve_trig_identity (x : ℝ) : Prop :=
  (∃ k : ℤ, x = (Real.pi / 8 * (4 * k + 1))) ∧
  (Real.sin (2 * x))^4 + (Real.cos (2 * x))^4 = Real.sin (2 * x) * Real.cos (2 * x)

theorem trig_identity_solution (x : ℝ) :
  solve_trig_identity x :=
sorry

end trig_identity_solution_l142_142094


namespace single_discount_eq_l142_142465

/--
A jacket is originally priced at $50. It is on sale for 25% off. After applying the sale discount, 
John uses a coupon that gives an additional 10% off of the discounted price. If there is a 5% sales 
tax on the final price, what single percent discount (before taxes) is equivalent to these series 
of discounts followed by the tax? --/
theorem single_discount_eq :
  let P0 := 50
  let discount1 := 0.25
  let discount2 := 0.10
  let tax := 0.05
  let discounted_price := P0 * (1 - discount1) * (1 - discount2)
  let after_tax_price := discounted_price * (1 + tax)
  let single_discount := (P0 - discounted_price) / P0
  single_discount * 100 = 32.5 :=
by
  sorry

end single_discount_eq_l142_142465


namespace complement_of_A_in_U_l142_142151

open Set

def U : Set ℕ := {x | x < 8}
def A : Set ℕ := {x | (x - 1) * (x - 3) * (x - 4) * (x - 7) = 0}

theorem complement_of_A_in_U : (U \ A) = {0, 2, 5, 6} := by
  sorry

end complement_of_A_in_U_l142_142151


namespace bruce_money_left_l142_142475

theorem bruce_money_left :
  let initial_amount := 71
  let cost_per_shirt := 5
  let number_of_shirts := 5
  let cost_of_pants := 26
  let total_cost := number_of_shirts * cost_per_shirt + cost_of_pants
  let money_left := initial_amount - total_cost
  money_left = 20 :=
by
  sorry

end bruce_money_left_l142_142475


namespace sum_of_numbers_l142_142052

theorem sum_of_numbers (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 10) (h3 : x^2 - y^2 = 50) : x + y = 5 :=
by
  sorry

end sum_of_numbers_l142_142052


namespace prob_less_than_9_l142_142572

def prob_10 : ℝ := 0.24
def prob_9 : ℝ := 0.28
def prob_8 : ℝ := 0.19

theorem prob_less_than_9 : prob_10 + prob_9 + prob_8 < 1 → 1 - prob_10 - prob_9 = 0.48 := 
by {
  sorry
}

end prob_less_than_9_l142_142572


namespace proof_of_intersection_l142_142555

open Set

theorem proof_of_intersection :
  let U := ℝ
  let M := compl { x : ℝ | x^2 > 4 }
  let N := { x : ℝ | 1 < x ∧ x ≤ 3 }
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } := by
sorry

end proof_of_intersection_l142_142555


namespace min_trig_expression_l142_142895

theorem min_trig_expression (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = Real.pi) : 
  ∃ (x : ℝ), (x = 16 - 8 * Real.sqrt 2) ∧ (∀ A B C, 0 < A → 0 < B → 0 < C → A + B + C = Real.pi → 
    (1 / (Real.sin A)^2 + 1 / (Real.sin B)^2 + 4 / (1 + Real.sin C)) ≥ x) := 
sorry

end min_trig_expression_l142_142895


namespace pyramid_base_side_length_correct_l142_142976

def sideLengthBase (s : ℕ) : Prop :=
  let area : ℕ := 100
  let slant_height : ℕ := 20
  let lateral_face_area := (1/2:ℚ) * s * slant_height
  lateral_face_area.toNat = area → s = 10

theorem pyramid_base_side_length_correct (s : ℕ) (h: s * 10 = 100) : sideLengthBase s :=
  by
    intros
    simp [sideLengthBase]
    assume lateral_face_area h
    exact h
    sorry

end pyramid_base_side_length_correct_l142_142976


namespace Einstein_sold_25_cans_of_soda_l142_142124

def sell_snacks_proof : Prop :=
  let pizza_price := 12
  let fries_price := 0.30
  let soda_price := 2
  let goal := 500
  let pizza_boxes := 15
  let fries_packs := 40
  let still_needed := 258
  let earned_from_pizza := pizza_boxes * pizza_price
  let earned_from_fries := fries_packs * fries_price
  let total_earned := earned_from_pizza + earned_from_fries
  let total_have := goal - still_needed
  let earned_from_soda := total_have - total_earned
  let cans_of_soda_sold := earned_from_soda / soda_price
  cans_of_soda_sold = 25

theorem Einstein_sold_25_cans_of_soda : sell_snacks_proof := by
  sorry

end Einstein_sold_25_cans_of_soda_l142_142124


namespace solve_system_of_equations_l142_142972

theorem solve_system_of_equations (x y : ℝ) (h1 : x - y = -5) (h2 : 3 * x + 2 * y = 10) : x = 0 ∧ y = 5 := by
  sorry

end solve_system_of_equations_l142_142972


namespace percent_of_x_eq_21_percent_l142_142445

theorem percent_of_x_eq_21_percent (x : Real) : (0.21 * x = 0.30 * 0.70 * x) := by
  sorry

end percent_of_x_eq_21_percent_l142_142445


namespace probability_odd_sum_l142_142060

-- Definitions given based on conditions
def primes_not_exceeding_19 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- The statement to prove
theorem probability_odd_sum :
  (primes_not_exceeding_19.card = 8) →
  let total_combinations := primes_not_exceeding_19.card.choose 2 in
  let odd_sum_combinations := (primes_not_exceeding_19.filter nat.prime.pred).card in
  ↑odd_sum_combinations / ↑total_combinations = (1 : ℚ) / 4 :=
by sorry

end probability_odd_sum_l142_142060


namespace emily_saves_more_using_promotion_a_l142_142605

-- Definitions based on conditions
def price_per_pair : ℕ := 50
def promotion_a_cost : ℕ := price_per_pair + price_per_pair / 2
def promotion_b_cost : ℕ := price_per_pair + (price_per_pair - 20)

-- Statement to prove the savings
theorem emily_saves_more_using_promotion_a :
  promotion_b_cost - promotion_a_cost = 5 := by
  sorry

end emily_saves_more_using_promotion_a_l142_142605


namespace min_value_sin_cos_l142_142829

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l142_142829


namespace min_sin6_cos6_l142_142838

theorem min_sin6_cos6 (x : ℝ) :
  sin x ^ 2 + cos x ^ 2 = 1 →  ∃ y : ℝ, y = sin x ^ 6 + cos x ^ 6 ∧ y = 1 / 4 :=
by
  sorry

end min_sin6_cos6_l142_142838


namespace sum_of_reciprocals_of_factors_of_12_l142_142387

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142387


namespace part1_part2_l142_142547

def U : Set ℝ := {x : ℝ | True}

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part 1: Prove the range of m when 4 ∈ B(m) is [5/2, 3]
theorem part1 (m : ℝ) : (4 ∈ B m) → (5/2 ≤ m ∧ m ≤ 3) := by
  sorry

-- Part 2: Prove the range of m when x ∈ A is a necessary but not sufficient condition for x ∈ B(m) 
theorem part2 (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ∧ ¬(∀ x, x ∈ A → x ∈ B m) → (m ≤ 3) := by
  sorry

end part1_part2_l142_142547


namespace train_speed_l142_142796

def train_length : ℝ := 800
def crossing_time : ℝ := 12
def expected_speed : ℝ := 66.67 

theorem train_speed (h_len : train_length = 800) (h_time : crossing_time = 12) : 
  train_length / crossing_time = expected_speed := 
by {
  sorry
}

end train_speed_l142_142796


namespace arithmetic_sequence_minimum_value_of_Sn_l142_142922

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l142_142922


namespace arithmetic_sequence_min_value_Sn_l142_142946

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l142_142946


namespace arithmetic_sequence_minimum_value_S_l142_142925

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l142_142925


namespace min_sixth_power_sin_cos_l142_142845

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l142_142845


namespace train_speed_l142_142799

theorem train_speed (length_of_train time_to_cross : ℝ) (h_length : length_of_train = 800) (h_time : time_to_cross = 12) : (length_of_train / time_to_cross) = 66.67 :=
by
  sorry

end train_speed_l142_142799


namespace g_three_fifths_l142_142705

-- Given conditions
variable (g : ℝ → ℝ)
variable (h₀ : g 0 = 0)
variable (h₁ : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
variable (h₂ : ∀ ⦃x : ℝ⦄, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
variable (h₃ : ∀ ⦃x : ℝ⦄, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3)

-- Proof statement
theorem g_three_fifths : g (3 / 5) = 2 / 3 := by
  sorry

end g_three_fifths_l142_142705


namespace sum_reciprocals_factors_12_l142_142349

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l142_142349


namespace sum_reciprocal_factors_of_12_l142_142412

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l142_142412


namespace tangent_line_y_intercept_l142_142788

/-
  Prove that the y-intercept of the line tangent to both circles at points in the first quadrant is 9,
  given the following conditions:
  1. Circle 1 has radius 3 and center (3, 0).
  2. Circle 2 has radius 1 and center (7, 0).
-/
theorem tangent_line_y_intercept
  (circle1_center : ℝ × ℝ)
  (circle1_radius : ℝ)
  (circle2_center : ℝ × ℝ)
  (circle2_radius : ℝ)
  (line_tangent : ℝ → ℝ)
  (circle1_tangent_point circle2_tangent_point : ℝ × ℝ) :
  circle1_center = (3, 0) →
  circle1_radius = 3 →
  circle2_center = (7, 0) →
  circle2_radius = 1 →
  -- Prove the y-intercept of the tangent line is 9
  line_tangent 0 = 9 :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end tangent_line_y_intercept_l142_142788


namespace fraction_product_simplified_l142_142069

theorem fraction_product_simplified:
  (2 / 9 : ℚ) * (5 / 8 : ℚ) = 5 / 36 :=
by {
  sorry
}

end fraction_product_simplified_l142_142069


namespace Roger_needs_to_delete_20_apps_l142_142622

def max_apps := 50
def recommended_apps := 35
def current_apps := 2 * recommended_apps
def apps_to_delete := current_apps - max_apps

theorem Roger_needs_to_delete_20_apps : apps_to_delete = 20 := by
  sorry

end Roger_needs_to_delete_20_apps_l142_142622


namespace arithmetic_sequence_min_value_Sn_l142_142949

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l142_142949


namespace units_digit_17_pow_2107_l142_142614

theorem units_digit_17_pow_2107 : (17 ^ 2107) % 10 = 3 := by
  -- Definitions derived from conditions:
  -- 1. Powers of 17 have the same units digit as the corresponding powers of 7.
  -- 2. Units digits of powers of 7 cycle: 7, 9, 3, 1.
  -- 3. 2107 modulo 4 gives remainder 3.
  sorry

end units_digit_17_pow_2107_l142_142614


namespace number_is_more_than_sum_l142_142586

theorem number_is_more_than_sum : 20.2 + 33.8 - 5.1 = 48.9 :=
by
  sorry

end number_is_more_than_sum_l142_142586


namespace max_points_of_intersection_l142_142228

-- Definitions from the conditions
def circles := 2
def lines := 3

-- Define the problem of the greatest intersection number
theorem max_points_of_intersection (c : ℕ) (l : ℕ) (h_c : c = circles) (h_l : l = lines) : 
  (2 + (l * 2 * c) + (l * (l - 1) / 2)) = 17 :=
by
  rw [h_c, h_l]
  -- We have 2 points from circle intersections
  -- 12 points from lines intersections with circles
  -- 3 points from lines intersections with lines
  -- Hence, 2 + 12 + 3 = 17
  exact Eq.refl 17

end max_points_of_intersection_l142_142228


namespace find_x_if_perpendicular_l142_142031

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (2 * x - 1, 3)
def vec_n : ℝ × ℝ := (1, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_x_if_perpendicular (x : ℝ) : 
  dot_product (vec_m x) vec_n = 0 ↔ x = 2 :=
by
  sorry

end find_x_if_perpendicular_l142_142031


namespace range_of_a_l142_142149

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, 
    1 ≤ x ∧ x ≤ 2 ∧ 
    2 ≤ y ∧ y ≤ 3 → 
    x * y ≤ a * x^2 + 2 * y^2) ↔ 
  a ≥ -1 :=
by
  sorry

end range_of_a_l142_142149


namespace pow_addition_l142_142591

theorem pow_addition : (-2)^2 + 2^2 = 8 :=
by
  sorry

end pow_addition_l142_142591


namespace corey_gave_more_books_l142_142691

def books_given_by_mike : ℕ := 10
def total_books_received_by_lily : ℕ := 35
def books_given_by_corey : ℕ := total_books_received_by_lily - books_given_by_mike
def difference_in_books (a b : ℕ) : ℕ := a - b

theorem corey_gave_more_books :
  difference_in_books books_given_by_corey books_given_by_mike = 15 := by
sorry

end corey_gave_more_books_l142_142691


namespace sum_of_reciprocals_of_factors_of_12_l142_142379

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142379


namespace conditional_two_exits_one_effective_l142_142485

def conditional_structure (decide : Bool) : Prop :=
  if decide then True else False

theorem conditional_two_exits_one_effective (decide : Bool) :
  conditional_structure decide ↔ True :=
by
  sorry

end conditional_two_exits_one_effective_l142_142485


namespace sum_reciprocals_factors_of_12_l142_142285

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l142_142285


namespace sum_of_distinct_product_GH_l142_142570

def divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_single_digit (d : ℕ) : Prop :=
  d < 10

theorem sum_of_distinct_product_GH : 
  ∀ (G H : ℕ), 
    is_single_digit G ∧ is_single_digit H ∧ 
    divisible_by_45 (8620000307 + 10000000 * G + H) → 
    (if H = 5 then GH = 6 else if H = 0 then GH = 0 else GH = 0) := 
  sorry

-- Note: This is a simplified representation; tailored more complex conditions and steps may be encapsulated in separate definitions and theorems as needed.

end sum_of_distinct_product_GH_l142_142570


namespace cycling_route_length_l142_142204

-- Conditions (segment lengths)
def segment1 : ℝ := 4
def segment2 : ℝ := 7
def segment3 : ℝ := 2
def segment4 : ℝ := 6
def segment5 : ℝ := 7

-- Specify the total length calculation
noncomputable def total_length : ℝ :=
  2 * (segment1 + segment2 + segment3) + 2 * (segment4 + segment5)

-- The theorem we want to prove
theorem cycling_route_length :
  total_length = 52 :=
by
  sorry

end cycling_route_length_l142_142204


namespace arithmetic_sequence_minimum_value_S_l142_142929

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l142_142929


namespace a_n_formula_b_n_geometric_sequence_l142_142564

noncomputable def a_n (n : ℕ) : ℝ := 3 * n - 1

def S_n (n : ℕ) : ℝ := sorry -- Sum of the first n terms of b_n

def b_n (n : ℕ) : ℝ := 2 - 2 * S_n n

theorem a_n_formula (n : ℕ) : a_n n = 3 * n - 1 :=
by { sorry }

theorem b_n_geometric_sequence : ∀ n ≥ 2, b_n n / b_n (n - 1) = 1 / 3 :=
by { sorry }

end a_n_formula_b_n_geometric_sequence_l142_142564


namespace number_of_people_in_team_l142_142761

def total_distance : ℕ := 150
def distance_per_member : ℕ := 30

theorem number_of_people_in_team :
  (total_distance / distance_per_member) = 5 := by
  sorry

end number_of_people_in_team_l142_142761


namespace hours_per_day_l142_142785

-- Define the parameters
def A1 := 57
def D1 := 12
def H2 := 6
def A2 := 30
def D2 := 19

-- Define the target Equation
theorem hours_per_day :
  A1 * D1 * H = A2 * D2 * H2 → H = 5 :=
by
  sorry

end hours_per_day_l142_142785


namespace sum_reciprocals_factors_of_12_l142_142282

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l142_142282


namespace sum_reciprocals_12_l142_142242

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l142_142242


namespace one_boy_one_girl_prob_one_boy_one_girl_given_one_boy_one_boy_one_girl_given_boy_born_on_monday_l142_142596

namespace ProofProblems

open ProbabilityTheory

-- Question 1
theorem one_boy_one_girl_prob {eqlikely : Bool} (h_eqlikely : eqlikely = true) :
  (prob_one_boy_one_girl_eq_2_children : ℚ) = 1 / 2 := by
  sorry

-- Question 2
theorem one_boy_one_girl_given_one_boy {eqlikely : Bool} (h_eqlikely : eqlikely = true) :
  (prob_one_boy_one_girl_given_one_boy : ℚ) = 2 / 3 := by
  sorry

-- Question 3
theorem one_boy_one_girl_given_boy_born_on_monday {eqlikely : Bool} (h_eqlikely : eqlikely = true) :
  (prob_one_boy_one_girl_given_boy_born_on_monday : ℚ) = 14 / 27 := by
  sorry

end ProofProblems

end one_boy_one_girl_prob_one_boy_one_girl_given_one_boy_one_boy_one_girl_given_boy_born_on_monday_l142_142596


namespace derivative_y_l142_142824

noncomputable def y (a α x : ℝ) :=
  (Real.exp (a * x)) * (3 * Real.sin (3 * x) - α * Real.cos (3 * x)) / (a ^ 2 + 9)

theorem derivative_y (a α x : ℝ) :
  (deriv (y a α) x) =
    (Real.exp (a * x)) * ((3 * a + 3 * α) * Real.sin (3 * x) + (9 - a * α) * Real.cos (3 * x)) / (a ^ 2 + 9) := 
sorry

end derivative_y_l142_142824


namespace true_statements_count_is_two_l142_142638

def original_proposition (a : ℝ) : Prop :=
  a < 0 → ∃ x : ℝ, x^2 + x + a = 0

def contrapositive (a : ℝ) : Prop :=
  ¬ (∃ x : ℝ, x^2 + x + a = 0) → a ≥ 0

def converse (a : ℝ) : Prop :=
  (∃ x : ℝ, x^2 + x + a = 0) → a < 0

def negation (a : ℝ) : Prop :=
  a < 0 → ¬ ∃ x : ℝ, x^2 + x + a = 0

-- Prove that there are exactly 2 true statements among the four propositions: 
-- original_proposition, contrapositive, converse, and negation.

theorem true_statements_count_is_two : 
  ∀ (a : ℝ), original_proposition a ∧ contrapositive a ∧ ¬(converse a) ∧ ¬(negation a) → 
  (original_proposition a ∧ contrapositive a ∧ ¬(converse a) ∧ ¬(negation a)) ↔ (2 = 2) := 
by
  sorry

end true_statements_count_is_two_l142_142638


namespace sum_reciprocals_factors_12_l142_142273

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l142_142273


namespace minimum_value_expression_l142_142955

noncomputable def minimum_expression (a b c : ℝ) : ℝ :=
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2

theorem minimum_value_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minimum_expression a b c ≥ 126 :=
by
  sorry

end minimum_value_expression_l142_142955


namespace simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l142_142813

variable (a b : ℤ)
def A : ℤ := b^2 - a^2 + 5 * a * b
def B : ℤ := 3 * a * b + 2 * b^2 - a^2

theorem simplify_2A_minus_B : 2 * A a b - B a b = -a^2 + 7 * a * b := by
  sorry

theorem evaluate_2A_minus_B_at_1_2 : 2 * A 1 2 - B 1 2 = 13 := by
  sorry

end simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l142_142813


namespace systematic_sampling_sequence_l142_142121

theorem systematic_sampling_sequence :
  ∃ (s : Set ℕ), s = {3, 13, 23, 33, 43} ∧
  (∀ n, n ∈ s → n ≤ 50 ∧ ∃ k, k < 5 ∧ n = 3 + k * 10) :=
by
  sorry

end systematic_sampling_sequence_l142_142121


namespace sum_of_reciprocals_of_factors_of_12_l142_142296

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142296


namespace pyramid_base_side_length_l142_142977

theorem pyramid_base_side_length (area : ℕ) (slant_height : ℕ) (s : ℕ) 
  (h1 : area = 100) 
  (h2 : slant_height = 20) 
  (h3 : area = (1 / 2) * s * slant_height) :
  s = 10 := 
by 
  sorry

end pyramid_base_side_length_l142_142977


namespace sum_octal_eq_1021_l142_142130

def octal_to_decimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let r1 := n / 10
  let d1 := r1 % 10
  let r2 := r1 / 10
  let d2 := r2 % 10
  (d2 * 64) + (d1 * 8) + d0

def decimal_to_octal (n : ℕ) : ℕ :=
  let d0 := n % 8
  let r1 := n / 8
  let d1 := r1 % 8
  let r2 := r1 / 8
  let d2 := r2 % 8
  d2 * 100 + d1 * 10 + d0

theorem sum_octal_eq_1021 :
  decimal_to_octal (octal_to_decimal 642 + octal_to_decimal 157) = 1021 := by
  sorry

end sum_octal_eq_1021_l142_142130


namespace area_of_A_inter_B_l142_142871

-- Define sets A and B as given
def setA : Set (ℝ × ℝ) := {p | (p.2 - p.1) * (p.2 - 1 / p.1) ≥ 0}
def setB : Set (ℝ × ℝ) := {p | (p.1 - 1) ^ 2 + (p.2 - 1) ^ 2 ≤ 1}

-- Define the theorem to prove the area of A ∩ B
theorem area_of_A_inter_B : 
  (measure_theory.measure_space.volume (Set (ℝ × ℝ))).measurable_measure (setA ∩ setB) = π / 2 :=
sorry

end area_of_A_inter_B_l142_142871


namespace shortest_distance_ln_x_to_line_is_sqrt2_l142_142758

noncomputable def shortest_distance_ln_x_to_line : ℝ :=
  let line := λ x, x + 1 in
  let curve := λ x, log x in
  let tangent_slope_at_x := λ x, 1/x in
  let tangent_point := (1, log 1) in
  let distance := λ (p : ℝ × ℝ) (l : ℝ → ℝ), abs (l p.1 - p.2) / real.sqrt (1^2 + (-1)^2) in
  distance tangent_point line

theorem shortest_distance_ln_x_to_line_is_sqrt2 :
  shortest_distance_ln_x_to_line = real.sqrt 2 :=
sorry

end shortest_distance_ln_x_to_line_is_sqrt2_l142_142758


namespace sum_reciprocal_factors_of_12_l142_142420

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l142_142420


namespace find_larger_integer_l142_142740

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l142_142740


namespace solve_quadratic_solve_cubic_l142_142856

theorem solve_quadratic (x : ℝ) (h : 2 * x^2 - 32 = 0) : x = 4 ∨ x = -4 := 
by sorry

theorem solve_cubic (x : ℝ) (h : (x + 4)^3 + 64 = 0) : x = -8 := 
by sorry

end solve_quadratic_solve_cubic_l142_142856


namespace point_not_in_region_l142_142048

-- Define the inequality
def inequality (x y : ℝ) : Prop := 3 * x + 2 * y < 6

-- Points definition
def point := ℝ × ℝ

-- Points to be checked
def p1 : point := (0, 0)
def p2 : point := (1, 1)
def p3 : point := (0, 2)
def p4 : point := (2, 0)

-- Conditions stating that certain points satisfy the inequality
axiom h1 : inequality p1.1 p1.2
axiom h2 : inequality p2.1 p2.2
axiom h3 : inequality p3.1 p3.2

-- Goal: Prove that point (2,0) does not satisfy the inequality
theorem point_not_in_region : ¬ inequality p4.1 p4.2 :=
sorry -- Proof omitted

end point_not_in_region_l142_142048


namespace missing_digit_divisibility_by_13_l142_142762

theorem missing_digit_divisibility_by_13 (B : ℕ) (H : 0 ≤ B ∧ B ≤ 9) : 
  (13 ∣ (200 + 10 * B + 5)) ↔ B = 12 :=
by sorry

end missing_digit_divisibility_by_13_l142_142762


namespace average_speed_round_trip_l142_142597

theorem average_speed_round_trip (d : ℝ) (h_d_pos : d > 0) : 
  let t1 := d / 80
  let t2 := d / 120
  let d_total := 2 * d
  let t_total := t1 + t2
  let v_avg := d_total / t_total
  v_avg = 96 :=
by
  sorry

end average_speed_round_trip_l142_142597


namespace sum_of_reciprocals_factors_12_l142_142397

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l142_142397


namespace sqrt_nested_l142_142160

theorem sqrt_nested (x : ℝ) (hx : 0 ≤ x) : Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15 / 16) := by
  sorry

end sqrt_nested_l142_142160


namespace find_larger_integer_l142_142749

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l142_142749


namespace temperature_on_friday_l142_142589

theorem temperature_on_friday 
  (M T W Th F : ℤ) 
  (h1 : (M + T + W + Th) / 4 = 48) 
  (h2 : (T + W + Th + F) / 4 = 46) 
  (h3 : M = 43) : 
  F = 35 := 
by
  sorry

end temperature_on_friday_l142_142589


namespace find_ellipse_equation_l142_142045

noncomputable def ellipse_equation : Prop :=
  ∃ (a b : ℝ), a^2 - b^2 = 5 ∧ (4*(-3)^2)/(9*a^2) + (9*2^2)/(4*b^2) = 36 ∧ 
    (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1)
 
theorem find_ellipse_equation : (∃ c : ℝ, 0 < c ∧ c = sqrt 5) → 
        (∃ (a b : ℝ), a^2 - b^2 = c^2 ∧ 
            ((4*(-3)^2)/(9*a^2) + (9*2^2)/(4*b^2) = 36) ∧ 
           (\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1)) := 
begin
  sorry
end

end find_ellipse_equation_l142_142045


namespace inequality_solution_set_l142_142041

theorem inequality_solution_set (x : ℝ) :
  (3 * (x + 2) - x > 4) ∧ ((1 + 2 * x) / 3 ≥ x - 1) ↔ (-1 < x ∧ x ≤ 4) :=
by
  sorry

end inequality_solution_set_l142_142041


namespace arithmetic_sequence_minimum_value_S_l142_142927

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l142_142927


namespace solve_for_t_l142_142157

theorem solve_for_t (s t : ℤ) (h1 : 11 * s + 7 * t = 160) (h2 : s = 2 * t + 4) : t = 4 :=
by
  sorry

end solve_for_t_l142_142157


namespace min_jumps_to_visit_all_points_and_return_l142_142055

theorem min_jumps_to_visit_all_points_and_return (n : ℕ) (h : n = 2016) : 
  ∀ jumps : ℕ, (∀ p : Fin n, ∃ k : ℕ, p = (2 * k) % n ∨ p = (3 * k) % n) → 
  jumps = 2017 :=
by 
  intros jumps h
  sorry

end min_jumps_to_visit_all_points_and_return_l142_142055


namespace cub_eqn_root_sum_l142_142004

noncomputable def cos_x := Real.cos (Real.pi / 5)

theorem cub_eqn_root_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
(h3 : a * cos_x ^ 3 - b * cos_x - 1 = 0) : a + b = 12 :=
sorry

end cub_eqn_root_sum_l142_142004


namespace half_angle_third_quadrant_l142_142874

theorem half_angle_third_quadrant (α : ℝ) (k : ℤ) (h1 : k * 360 + 180 < α) (h2 : α < k * 360 + 270) : 
  (∃ n : ℤ, n * 360 + 90 < (α / 2) ∧ (α / 2) < n * 360 + 135) ∨ 
  (∃ n : ℤ, n * 360 + 270 < (α / 2) ∧ (α / 2) < n * 360 + 315) := 
sorry

end half_angle_third_quadrant_l142_142874


namespace polynomial_factorization_l142_142615

-- Define the given polynomial expression
def given_poly (x : ℤ) : ℤ :=
  3 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 5 * x^2

-- Define the supposed factored form
def factored_poly (x : ℤ) : ℤ :=
  x * (3 * x^3 + 117 * x^2 + 1430 * x + 14895)

-- The theorem stating the equality of the two expressions
theorem polynomial_factorization (x : ℤ) : given_poly x = factored_poly x :=
  sorry

end polynomial_factorization_l142_142615


namespace sum_of_reciprocals_of_factors_of_12_l142_142318

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142318


namespace sum_of_reciprocals_of_factors_of_12_l142_142293

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142293


namespace sum_reciprocals_factors_12_l142_142333

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142333


namespace response_rate_percentage_l142_142787

theorem response_rate_percentage (number_of_responses_needed number_of_questionnaires_mailed : ℕ) 
  (h1 : number_of_responses_needed = 300) 
  (h2 : number_of_questionnaires_mailed = 500) : 
  (number_of_responses_needed / number_of_questionnaires_mailed : ℚ) * 100 = 60 :=
by 
  sorry

end response_rate_percentage_l142_142787


namespace chess_grandmaster_time_l142_142174

theorem chess_grandmaster_time :
  let time_to_learn_rules : ℕ := 2
  let factor_to_get_proficient : ℕ := 49
  let factor_to_become_master : ℕ := 100
  let time_to_get_proficient := factor_to_get_proficient * time_to_learn_rules
  let combined_time := time_to_learn_rules + time_to_get_proficient
  let time_to_become_master := factor_to_become_master * combined_time
  let total_time := time_to_learn_rules + time_to_get_proficient + time_to_become_master
  total_time = 10100 :=
by
  sorry

end chess_grandmaster_time_l142_142174


namespace least_number_subtracted_l142_142444

theorem least_number_subtracted (a b c : ℕ) (h1 : a = 7) (h2 : b = 9) (h3 : c = 11) :
  ∃ x, 0 ≤ x ∧ x < 1398 ∧ (1398 - x) % a = 5 ∧ (1398 - x) % b = 5 ∧ (1398 - x) % c = 5 ∧ x = 22 :=
by {
  sorry
}

end least_number_subtracted_l142_142444


namespace segment_outside_spheres_l142_142063

noncomputable def fraction_outside_spheres (α : ℝ) : ℝ :=
  (1 - (Real.cos (α / 2))^2) / (1 + (Real.cos (α / 2))^2)

theorem segment_outside_spheres (R α : ℝ) (hR : R > 0) (hα : 0 < α ∧ α < 2 * Real.pi) :
  fraction_outside_spheres α = (1 - Real.cos (α / 2)^2) / (1 + (Real.cos (α / 2))^2) :=
  by sorry

end segment_outside_spheres_l142_142063


namespace trimino_tilings_greater_l142_142509

noncomputable def trimino_tilings (n : ℕ) : ℕ := sorry
noncomputable def domino_tilings (n : ℕ) : ℕ := sorry

theorem trimino_tilings_greater (n : ℕ) (h : n > 1) : trimino_tilings (3 * n) > domino_tilings (2 * n) :=
sorry

end trimino_tilings_greater_l142_142509


namespace a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l142_142513

theorem a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l142_142513


namespace min_sixth_power_sin_cos_l142_142844

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l142_142844


namespace solve_for_x_l142_142698

theorem solve_for_x (x : ℕ) (hx : 1000^4 = 10^x) : x = 12 := 
by
  sorry

end solve_for_x_l142_142698


namespace solution_l142_142023

theorem solution 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : (a^2 / (b - c)^2) + (b^2 / (c - a)^2) + (c^2 / (a - b)^2) = 0) :
  (a^3 / (b - c)^3) + (b^3 / (c - a)^3) + (c^3 / (a - b)^3) = 0 := 
sorry 

end solution_l142_142023


namespace no_valid_n_l142_142857

theorem no_valid_n (n : ℕ) (h₁ : 100 ≤ n / 4) (h₂ : n / 4 ≤ 999) (h₃ : 100 ≤ 4 * n) (h₄ : 4 * n ≤ 999) : false := by
  sorry

end no_valid_n_l142_142857


namespace sum_reciprocals_factors_12_l142_142260

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l142_142260


namespace sandy_has_four_times_more_marbles_l142_142019

-- Definitions based on conditions
def jessica_red_marbles : ℕ := 3 * 12
def sandy_red_marbles : ℕ := 144

-- The theorem to prove
theorem sandy_has_four_times_more_marbles : sandy_red_marbles = 4 * jessica_red_marbles :=
by
  sorry

end sandy_has_four_times_more_marbles_l142_142019


namespace problem1_problem2_l142_142639

def A (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 7
def S (x : ℝ) (k : ℝ) : Prop := k + 1 ≤ x ∧ x ≤ 2 * k - 1

theorem problem1 (k : ℝ) : (∀ x, S x k → A x) → k ≤ 4 :=
by
  sorry

theorem problem2 (k : ℝ) : (∀ x, ¬(A x ∧ S x k)) → k < 2 ∨ k > 6 :=
by
  sorry

end problem1_problem2_l142_142639


namespace num_congruent_2_mod_11_l142_142643

theorem num_congruent_2_mod_11 : 
  ∃ (n : ℕ), n = 28 ∧ ∀ k : ℤ, 1 ≤ 11 * k + 2 ∧ 11 * k + 2 ≤ 300 ↔ 0 ≤ k ∧ k ≤ 27 :=
sorry

end num_congruent_2_mod_11_l142_142643


namespace average_of_angles_l142_142668

theorem average_of_angles (p q r s t : ℝ) (h : p + q + r + s + t = 180) : 
  (p + q + r + s + t) / 5 = 36 :=
by
  sorry

end average_of_angles_l142_142668


namespace find_larger_integer_l142_142738

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l142_142738


namespace difference_q_r_share_l142_142472

theorem difference_q_r_share (p q r : ℕ) (x : ℕ) (h_ratio : p = 3 * x) (h_ratio_q : q = 7 * x) (h_ratio_r : r = 12 * x) (h_diff_pq : q - p = 4400) : q - r = 5500 :=
by
  sorry

end difference_q_r_share_l142_142472


namespace length_of_largest_square_l142_142970

-- Define the conditions of the problem
def side_length_of_shaded_square : ℕ := 10
def side_length_of_largest_square : ℕ := 24

-- The statement to prove
theorem length_of_largest_square (x : ℕ) (h1 : x = side_length_of_shaded_square) : 
  4 * x = side_length_of_largest_square :=
  by
  -- Insert the proof here
  sorry

end length_of_largest_square_l142_142970


namespace arithmetic_sequence_and_minimum_sum_l142_142918

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l142_142918


namespace train_speed_l142_142797

def train_length : ℝ := 800
def crossing_time : ℝ := 12
def expected_speed : ℝ := 66.67 

theorem train_speed (h_len : train_length = 800) (h_time : crossing_time = 12) : 
  train_length / crossing_time = expected_speed := 
by {
  sorry
}

end train_speed_l142_142797


namespace solve_for_a_l142_142009

open Complex

noncomputable def question (a : ℝ) : Prop :=
  ∃ z : ℂ, z = (a + I) / (1 - I) ∧ z.im ≠ 0 ∧ z.re = 0

theorem solve_for_a (a : ℝ) (h : question a) : a = 1 :=
sorry

end solve_for_a_l142_142009


namespace divides_a_square_minus_a_and_a_cube_minus_a_l142_142182

theorem divides_a_square_minus_a_and_a_cube_minus_a (a : ℤ) : 
  (2 ∣ a^2 - a) ∧ (3 ∣ a^3 - a) :=
by
  sorry

end divides_a_square_minus_a_and_a_cube_minus_a_l142_142182


namespace sequence_problem_l142_142912

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l142_142912


namespace base_eight_to_base_ten_l142_142222

theorem base_eight_to_base_ten : (4 * 8^1 + 5 * 8^0 = 37) := by
  sorry

end base_eight_to_base_ten_l142_142222


namespace minimum_ab_bc_ca_l142_142650

theorem minimum_ab_bc_ca {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = a^3) (h5 : a * b * c = a^3) : 
  ab + bc + ca ≥ 9 :=
sorry

end minimum_ab_bc_ca_l142_142650


namespace sum_reciprocals_factors_12_l142_142256

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l142_142256


namespace sum_reciprocal_factors_of_12_l142_142415

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l142_142415


namespace impossible_to_achieve_12_percent_return_l142_142109

-- Define the stock parameters and their individual returns
def stock_A_price : ℝ := 52
def stock_A_dividend_rate : ℝ := 0.09
def stock_A_transaction_fee_rate : ℝ := 0.02

def stock_B_price : ℝ := 80
def stock_B_dividend_rate : ℝ := 0.07
def stock_B_transaction_fee_rate : ℝ := 0.015

def stock_C_price : ℝ := 40
def stock_C_dividend_rate : ℝ := 0.10
def stock_C_transaction_fee_rate : ℝ := 0.01

def tax_rate : ℝ := 0.10
def desired_return : ℝ := 0.12

theorem impossible_to_achieve_12_percent_return :
  false :=
sorry

end impossible_to_achieve_12_percent_return_l142_142109


namespace range_of_k_l142_142025

theorem range_of_k (f : ℝ → ℝ) (a : ℝ) (k : ℝ) 
  (h₀ : ∀ x > 0, f x = 2 - 1 / (a - x)^2) 
  (h₁ : ∀ x > 0, k^2 * x + f (1 / 4 * x + 1) > 0) : 
  k ≠ 0 :=
by
  -- proof goes here
  sorry

end range_of_k_l142_142025


namespace digit_d_for_5678d_is_multiple_of_9_l142_142623

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem digit_d_for_5678d_is_multiple_of_9 : 
  ∃ d : ℕ, d < 10 ∧ is_multiple_of_9 (56780 + d) ∧ d = 1 :=
by
  sorry

end digit_d_for_5678d_is_multiple_of_9_l142_142623


namespace dot_product_eq_neg29_l142_142153

-- Given definitions and conditions
variables (a b : ℝ × ℝ)

-- Theorem to prove the dot product condition.
theorem dot_product_eq_neg29 (h1 : a + b = (2, -4)) (h2 : 3 • a - b = (-10, 16)) :
  a.1 * b.1 + a.2 * b.2 = -29 :=
sorry

end dot_product_eq_neg29_l142_142153


namespace sum_reciprocals_factors_12_l142_142350

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l142_142350


namespace sum_reciprocals_factors_12_l142_142347

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l142_142347


namespace incorrect_transformation_is_not_valid_l142_142765

-- Define the system of linear equations
def eq1 (x y : ℝ) := 2 * x + y = 5
def eq2 (x y : ℝ) := 3 * x + 4 * y = 7

-- The definition of the correct transformation for x from equation eq2
def correct_transformation (x y : ℝ) := x = (7 - 4 * y) / 3

-- The definition of the incorrect transformation for x from equation eq2
def incorrect_transformation (x y : ℝ) := x = (7 + 4 * y) / 3

theorem incorrect_transformation_is_not_valid (x y : ℝ) 
  (h1 : eq1 x y) 
  (h2 : eq2 x y) :
  ¬ incorrect_transformation x y := 
by
  sorry

end incorrect_transformation_is_not_valid_l142_142765


namespace cos_double_angle_l142_142859

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 1 / 2) : Real.cos (2 * α) = 3 / 5 :=
by
  sorry

end cos_double_angle_l142_142859


namespace congruence_solution_count_l142_142143

theorem congruence_solution_count :
  ∀ y : ℕ, y < 150 → (y ≡ 20 + 110 [MOD 46]) → y = 38 ∨ y = 84 ∨ y = 130 :=
by
  intro y
  intro hy
  intro hcong
  sorry

end congruence_solution_count_l142_142143


namespace cheryl_material_left_l142_142046

-- Conditions
def initial_material_type1 (m1 : ℚ) : Prop := m1 = 2/9
def initial_material_type2 (m2 : ℚ) : Prop := m2 = 1/8
def used_material (u : ℚ) : Prop := u = 0.125

-- Define the total material bought
def total_material (m1 m2 : ℚ) : ℚ := m1 + m2

-- Define the material left
def material_left (t u : ℚ) : ℚ := t - u

-- The target theorem
theorem cheryl_material_left (m1 m2 u : ℚ) 
  (h1 : initial_material_type1 m1)
  (h2 : initial_material_type2 m2)
  (h3 : used_material u) : 
  material_left (total_material m1 m2) u = 2/9 :=
by
  sorry

end cheryl_material_left_l142_142046


namespace sum_reciprocals_factors_12_l142_142262

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l142_142262


namespace quadratic_intersection_l142_142565

def quadratic (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

theorem quadratic_intersection:
  ∃ a b c : ℝ, 
  quadratic a b c (-3) = 16 ∧ 
  quadratic a b c 0 = -5 ∧ 
  quadratic a b c 3 = -8 ∧ 
  quadratic a b c (-1) = 0 :=
sorry

end quadratic_intersection_l142_142565


namespace sum_of_reciprocals_of_factors_of_12_l142_142439

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142439


namespace no_solution_in_natural_numbers_l142_142190

theorem no_solution_in_natural_numbers (x y z : ℕ) : ¬((2 * x) ^ (2 * x) - 1 = y ^ (z + 1)) := 
  sorry

end no_solution_in_natural_numbers_l142_142190


namespace smallest_integer_with_divisors_properties_l142_142085

def number_of_odd_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 1)

def number_of_even_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 0)

theorem smallest_integer_with_divisors_properties :
  ∃ n : ℕ, number_of_odd_divisors n = 8 ∧ number_of_even_divisors n = 16 ∧ n = 4000 :=
by
  sorry

end smallest_integer_with_divisors_properties_l142_142085


namespace trapezoid_area_condition_l142_142667

theorem trapezoid_area_condition
  (a x y z : ℝ)
  (h_sq  : ∀ (ABCD : ℝ), ABCD = a * a)
  (h_trap: ∀ (EBCF : ℝ), EBCF = x * a)
  (h_rec : ∀ (JKHG : ℝ), JKHG = y * z)
  (h_sum : y + z = a)
  (h_area : x * a = a * a - 2 * y * z) :
  x = a / 2 :=
by
  sorry

end trapezoid_area_condition_l142_142667


namespace find_radius_l142_142975

theorem find_radius
  (sector_area : ℝ)
  (arc_length : ℝ)
  (sector_area_eq : sector_area = 11.25)
  (arc_length_eq : arc_length = 4.5) :
  ∃ r : ℝ, 11.25 = (1/2 : ℝ) * r * arc_length ∧ r = 5 := 
by
  sorry

end find_radius_l142_142975


namespace repair_time_calculation_l142_142161

-- Assume amount of work is represented as units
def work_10_people_45_minutes := 10 * 45
def work_20_people_20_minutes := 20 * 20

-- Assuming the flood destroys 2 units per minute as calculated in the solution
def flood_rate := 2

-- Calculate total initial units of the dike
def dike_initial_units :=
  work_10_people_45_minutes - flood_rate * 45

-- Given 14 people are repairing the dam
def repair_rate_14_people := 14 - flood_rate

-- Statement to prove that 14 people need 30 minutes to repair the dam
theorem repair_time_calculation :
  dike_initial_units / repair_rate_14_people = 30 :=
by
  sorry

end repair_time_calculation_l142_142161


namespace rowing_distance_correct_l142_142455

variable (D : ℝ) -- distance to the place
variable (speed_in_still_water : ℝ := 10) -- rowing speed in still water
variable (current_speed : ℝ := 2) -- speed of the current
variable (total_time : ℝ := 30) -- total time for round trip
variable (effective_speed_with_current : ℝ := speed_in_still_water + current_speed) -- effective speed with current
variable (effective_speed_against_current : ℝ := speed_in_still_water - current_speed) -- effective speed against current

theorem rowing_distance_correct : 
  D / effective_speed_with_current + D / effective_speed_against_current = total_time → 
  D = 144 := 
by
  intros h
  sorry

end rowing_distance_correct_l142_142455


namespace sum_reciprocals_of_factors_12_l142_142366

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l142_142366


namespace total_original_grain_l142_142059

-- Define initial conditions
variables (initial_warehouse1 : ℕ) (initial_warehouse2 : ℕ)
-- Define the amount of grain transported away from the first warehouse
def transported_away := 2500
-- Define the amount of grain in the second warehouse
def warehouse2_initial := 50200

-- Prove the total original amount of grain in the two warehouses
theorem total_original_grain 
  (h1 : transported_away = 2500)
  (h2 : warehouse2_initial = 50200)
  (h3 : initial_warehouse1 - transported_away = warehouse2_initial) : 
  initial_warehouse1 + warehouse2_initial = 102900 :=
sorry

end total_original_grain_l142_142059


namespace number_of_toys_gained_l142_142466

theorem number_of_toys_gained
  (num_toys : ℕ) (selling_price : ℕ) (cost_price_one_toy : ℕ)
  (total_cp := num_toys * cost_price_one_toy)
  (profit := selling_price - total_cp)
  (num_toys_equiv_to_profit := profit / cost_price_one_toy) :
  num_toys = 18 → selling_price = 23100 → cost_price_one_toy = 1100 → num_toys_equiv_to_profit = 3 :=
by
  intros h1 h2 h3
  -- Proof to be completed
  sorry

end number_of_toys_gained_l142_142466


namespace simplify_expr_l142_142812

def A (a b : ℝ) := b^2 - a^2 + 5 * a * b
def B (a b : ℝ) := 3 * a * b + 2 * b^2 - a^2

theorem simplify_expr (a b : ℝ) : 2 * (A a b) - (B a b) = -a^2 + 7 * a * b := by
  -- actual proof omitted
  sorry

example : (2 * (A 1 2) - (B 1 2)) = 13 := by
  -- actual proof omitted
  sorry

end simplify_expr_l142_142812


namespace triangle_problem_l142_142011

noncomputable theory

section
variables (a b c : ℝ) (A B C : ℝ)
  
theorem triangle_problem
  (h₁ : b = 3)
  (h₂ : b * sin A = sqrt 3 * a * cos B)
  (h₃ : sin C = 2 * sin A)
  (h₄ : c = 2 * a)
  (h₅ : B = Real.pi / 3) :
  a = sqrt 3 ∧ c = 2 * sqrt 3 ∧ (1/2 * a * c * sin B = 3 * sqrt 3 / 2) :=
by
  sorry
end

end triangle_problem_l142_142011


namespace sum_of_reciprocals_factors_12_l142_142303

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l142_142303


namespace sum_of_reciprocals_factors_12_l142_142391

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l142_142391


namespace prove_arithmetic_sequence_minimum_value_S_l142_142932

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l142_142932


namespace larger_integer_21_l142_142710

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l142_142710


namespace sum_of_integers_l142_142068

theorem sum_of_integers {n : ℤ} (h : n + 2 = 9) : n + (n + 1) + (n + 2) = 24 := by
  sorry

end sum_of_integers_l142_142068


namespace quadrilateral_midpoint_distance_squared_l142_142535

noncomputable section

open Classical

variables {A B C D X Y : ℝ}

def is_convex_quadrilateral (A B C D : ℝ): Prop :=
  -- Definition of the convex quadrilateral
  -- to be expanded according to the specific geometry if needed
  true

def midpoint (a b : ℝ) : ℝ :=
  (a + b) / 2

def squared_distance (x y : ℝ) : ℝ :=
  (x - y) ^ 2

theorem quadrilateral_midpoint_distance_squared
  (h_conv: is_convex_quadrilateral A B C D)
  (h_AB: A = B ∧ B = 15)
  (h_CD: C = D ∧ D = 20)
  (h_angle: A = 60)
  (h_midpoints: X = midpoint B C ∧ Y = midpoint D A) :
  squared_distance X Y = (5375 / 4) + 25 * Real.sqrt 15 :=
sorry

end quadrilateral_midpoint_distance_squared_l142_142535


namespace larger_integer_21_l142_142712

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l142_142712


namespace larger_integer_value_l142_142721

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l142_142721


namespace larger_integer_is_21_l142_142724

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l142_142724


namespace maximum_z_l142_142120

theorem maximum_z (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) : z ≤ 13 / 3 :=
sorry

end maximum_z_l142_142120


namespace perimeter_of_square_land_is_36_diagonal_of_square_land_is_27_33_l142_142201

def square_land (A P D : ℝ) :=
  (5 * A = 10 * P + 45) ∧
  (3 * D = 2 * P + 10)

theorem perimeter_of_square_land_is_36 (A P D : ℝ) (h1 : 5 * A = 10 * P + 45) (h2 : 3 * D = 2 * P + 10) :
  P = 36 :=
sorry

theorem diagonal_of_square_land_is_27_33 (A P D : ℝ) (h1 : P = 36) (h2 : 3 * D = 2 * P + 10) :
  D = 82 / 3 :=
sorry

end perimeter_of_square_land_is_36_diagonal_of_square_land_is_27_33_l142_142201


namespace larger_integer_is_21_l142_142751

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l142_142751


namespace compare_logs_l142_142181

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log 5 / Real.log (1 / 2)

theorem compare_logs : c < a ∧ a < b := by
  have h0 : a = Real.log 2 / Real.log 3 := rfl
  have h1 : b = Real.log 3 / Real.log 2 := rfl
  have h2 : c = Real.log 5 / Real.log (1 / 2) := rfl
  sorry

end compare_logs_l142_142181


namespace arithmetic_sequence_min_value_S_l142_142953

-- Let S_n be the sum of the first n terms of the sequence {a_n}
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

-- Given condition: For all n, (2 * S n) / n + n = 2 * a n + 1
axiom given_condition (n : ℕ) : (2 * S n) / n + n = 2 * a n + 1

-- Question 1: Prove that {a_n} is an arithmetic sequence.
theorem arithmetic_sequence (n : ℕ) : a (n + 1) = a n + 1 :=
sorry

-- Question 2: Given a_4, a_7, and a_9 form a geometric sequence, find the minimum value of S_n.
-- Additional condition for part 2:
axiom geometric_sequence : (a 7) ^ 2 = (a 4) * (a 9)

-- Goal: Find the minimum value of S_n
noncomputable def minimum_S : ℝ :=
-78

-- Prove that the minimum value of S_n is -78
theorem min_value_S (n : ℕ) (h_geometric : geometric_sequence) : S n = -78 :=
sorry

end arithmetic_sequence_min_value_S_l142_142953


namespace greatest_distance_centers_of_circles_in_rectangle_l142_142219

/--
Two circles are drawn in a 20-inch by 16-inch rectangle,
each circle with a diameter of 8 inches.
Prove that the greatest possible distance between 
the centers of the two circles without extending beyond the 
rectangular region is 4 * sqrt 13 inches.
-/
theorem greatest_distance_centers_of_circles_in_rectangle :
  let diameter := 8
  let width := 20
  let height := 16
  let radius := diameter / 2
  let reduced_width := width - 2 * radius
  let reduced_height := height - 2 * radius
  let distance := Real.sqrt ((reduced_width ^ 2) + (reduced_height ^ 2))
  distance = 4 * Real.sqrt 13 := by
    sorry

end greatest_distance_centers_of_circles_in_rectangle_l142_142219


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l142_142943

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l142_142943


namespace find_divisor_l142_142692

theorem find_divisor (d : ℕ) (h : 127 = d * 5 + 2) : d = 25 :=
by 
  -- Given conditions
  -- 127 = d * 5 + 2
  -- We need to prove d = 25
  sorry

end find_divisor_l142_142692


namespace common_ratio_is_two_l142_142522

-- Define the geometric sequence
def geom_seq (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ := a_1 * q^n

-- Define the conditions
variables (a_1 q : ℝ)
variables (h_inc : 1 < q) (h_pos : 0 < a_1)
variables (h_seq : ∀ n : ℕ, 2 * (geom_seq a_1 q n + geom_seq a_1 q (n+2)) = 5 * geom_seq a_1 q (n+1))

-- Statement to prove
theorem common_ratio_is_two : q = 2 :=
by
  sorry

end common_ratio_is_two_l142_142522


namespace ethan_pages_left_l142_142821

-- Definitions based on the conditions
def total_pages := 360
def pages_read_morning := 40
def pages_read_night := 10
def pages_read_saturday := pages_read_morning + pages_read_night
def pages_read_sunday := 2 * pages_read_saturday
def total_pages_read := pages_read_saturday + pages_read_sunday

-- Lean 4 statement for the proof problem
theorem ethan_pages_left : total_pages - total_pages_read = 210 := by
  sorry

end ethan_pages_left_l142_142821


namespace distance_a_beats_b_l142_142104

noncomputable def time_a : ℕ := 90 -- A's time in seconds 
noncomputable def time_b : ℕ := 180 -- B's time in seconds 
noncomputable def distance : ℝ := 4.5 -- distance in km

theorem distance_a_beats_b : distance = (distance / time_a) * (time_b - time_a) :=
by
  -- sorry placeholder for proof
  sorry

end distance_a_beats_b_l142_142104


namespace sequence_problem_l142_142914

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l142_142914


namespace compute_m_div_18_l142_142501

noncomputable def ten_pow (n : ℕ) : ℕ := Nat.pow 10 n

def valid_digits (m : ℕ) : Prop :=
  ∀ d ∈ m.digits 10, d = 0 ∨ d = 8

def is_multiple_of_18 (m : ℕ) : Prop :=
  m % 18 = 0

theorem compute_m_div_18 :
  ∃ m, valid_digits m ∧ is_multiple_of_18 m ∧ m / 18 = 493827160 :=
by
  sorry

end compute_m_div_18_l142_142501


namespace probability_not_green_l142_142058

theorem probability_not_green :
  let red_balls := 6
  let yellow_balls := 3
  let black_balls := 4
  let green_balls := 5
  let total_balls := red_balls + yellow_balls + black_balls + green_balls
  let not_green_balls := red_balls + yellow_balls + black_balls
  total_balls = 18 ∧ not_green_balls = 13 → (not_green_balls : ℚ) / total_balls = 13 / 18 := 
by
  intros
  sorry

end probability_not_green_l142_142058


namespace sum_reciprocal_factors_12_l142_142248

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l142_142248


namespace mass_percentage_Al_in_Al2CO33_l142_142128
-- Importing the required libraries

-- Define the necessary constants for molar masses
def molar_mass_Al : ℝ := 26.98
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00
def molar_mass_Al2CO33 : ℝ := 2 * molar_mass_Al + 3 * molar_mass_C + 9 * molar_mass_O
def mass_Al_in_Al2CO33 : ℝ := 2 * molar_mass_Al

-- Define the main theorem to prove the mass percentage of Al in Al2(CO3)3
theorem mass_percentage_Al_in_Al2CO33 :
  (mass_Al_in_Al2CO33 / molar_mass_Al2CO33) * 100 = 23.05 :=
by
  simp [molar_mass_Al, molar_mass_C, molar_mass_O, molar_mass_Al2CO33, mass_Al_in_Al2CO33]
  -- Calculation result based on given molar masses
  sorry

end mass_percentage_Al_in_Al2CO33_l142_142128


namespace number_that_multiplies_b_l142_142655

variable (a b x : ℝ)

theorem number_that_multiplies_b (h1 : 7 * a = x * b) (h2 : a * b ≠ 0) (h3 : (a / 8) / (b / 7) = 1) : x = 8 := 
sorry

end number_that_multiplies_b_l142_142655


namespace seventh_term_of_arithmetic_sequence_l142_142050

variable (a d : ℕ)

theorem seventh_term_of_arithmetic_sequence (h1 : 5 * a + 10 * d = 15) (h2 : a + 3 * d = 4) : a + 6 * d = 7 := 
by
  sorry

end seventh_term_of_arithmetic_sequence_l142_142050


namespace optimal_discount_sequence_saves_more_l142_142706

theorem optimal_discount_sequence_saves_more :
  (let initial_price := 30
   let flat_discount := 5
   let percent_discount := 0.25
   let first_seq_price := ((initial_price - flat_discount) * (1 - percent_discount))
   let second_seq_price := ((initial_price * (1 - percent_discount)) - flat_discount)
   first_seq_price - second_seq_price = 1.25) :=
by
  sorry

end optimal_discount_sequence_saves_more_l142_142706


namespace min_percentage_of_people_owning_95_percent_money_l142_142017

theorem min_percentage_of_people_owning_95_percent_money 
  (total_people: ℕ) (total_money: ℕ) 
  (P: ℕ) (M: ℕ) 
  (H1: P = total_people * 10 / 100) 
  (H2: M = total_money * 90 / 100)
  (H3: ∀ (people_owning_90_percent: ℕ), people_owning_90_percent = P → people_owning_90_percent * some_money = M) :
      P = total_people * 55 / 100 := 
sorry

end min_percentage_of_people_owning_95_percent_money_l142_142017


namespace correct_option_C_l142_142208

def number_of_stamps : String := "the number of the stamps"
def number_of_people : String := "a number of people"

def is_singular (subject : String) : Prop := subject = number_of_stamps
def is_plural (subject : String) : Prop := subject = number_of_people

def correct_sentence (verb1 verb2 : String) : Prop :=
  verb1 = "is" ∧ verb2 = "want"

theorem correct_option_C : correct_sentence "is" "want" :=
by
  show correct_sentence "is" "want"
  -- Proof is omitted
  sorry

end correct_option_C_l142_142208


namespace min_value_sin6_cos6_l142_142832

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l142_142832


namespace larger_integer_is_21_l142_142733

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l142_142733


namespace parabola_points_l142_142203

noncomputable def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_points :
  ∃ (a c m n : ℝ),
  a = 2 ∧ c = -2 ∧
  parabola a 1 c 2 = m ∧
  parabola a 1 c n = -2 ∧
  m = 8 ∧
  n = -1 / 2 :=
by
  use 2, -2, 8, -1/2
  simp [parabola]
  sorry

end parabola_points_l142_142203


namespace arithmetic_sequence_minimum_value_of_Sn_l142_142923

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l142_142923


namespace arithmetic_sequence_min_value_Sn_l142_142945

-- Define the sequence a_n and the sum S_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- The given condition
axiom condition : ∀ n : ℕ, n > 0 → (2 * S n / n) + n = 2 * a n + 1

-- Arithmetic sequence proof
theorem arithmetic_sequence : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 1 :=
by sorry

-- Minimum value of S_n when a_4, a_7, a_9 are geometric
theorem min_value_Sn (G : ℝ) (h : a 4 * a 9 = a 7 ^ 2) : ∃ n : ℕ, S n = -78 :=
by sorry

end arithmetic_sequence_min_value_Sn_l142_142945


namespace sum_reciprocals_factors_12_l142_142346

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l142_142346


namespace min_sin6_cos6_l142_142848

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l142_142848


namespace sum_of_reciprocals_factors_12_l142_142396

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l142_142396


namespace completing_square_transformation_l142_142451

theorem completing_square_transformation (x : ℝ) :
  x^2 - 2 * x - 5 = 0 -> (x - 1)^2 = 6 :=
by {
  sorry -- Proof to be completed
}

end completing_square_transformation_l142_142451


namespace larger_integer_21_l142_142714

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l142_142714


namespace sum_reciprocal_factors_12_l142_142254

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l142_142254


namespace length_of_BD_l142_142474

noncomputable def points_on_circle (A B C D E : Type) (BD AE BC CD : ℝ) (y z : ℝ) : Prop :=
  BC = 4 ∧ CD = 4 ∧ AE = 6 ∧ (0 < y) ∧ (0 < z) ∧ (AE * 2 = y * z) ∧ (8 > y + z)

theorem length_of_BD (A B C D E : Type) (BD AE BC CD : ℝ) (y z : ℝ)
  (h : points_on_circle A B C D E BD AE BC CD y z) : 
  BD = 7 :=
by
  sorry

end length_of_BD_l142_142474


namespace sum_of_reciprocals_of_factors_of_12_l142_142319

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142319


namespace sum_reciprocal_factors_12_l142_142245

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l142_142245


namespace minimize_sin_cos_six_l142_142825

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l142_142825


namespace product_gcd_lcm_l142_142504

theorem product_gcd_lcm (a b : ℕ) (ha : a = 90) (hb : b = 150) :
  Nat.gcd a b * Nat.lcm a b = 13500 := by
  sorry

end product_gcd_lcm_l142_142504


namespace find_k_l142_142873

theorem find_k 
  (k : ℤ) 
  (h : 2^2000 - 2^1999 - 2^1998 + 2^1997 = k * 2^1997) : 
  k = 3 :=
sorry

end find_k_l142_142873


namespace division_of_2301_base4_by_21_base4_l142_142488

noncomputable def divide_in_base4 : ℕ := sorry

theorem division_of_2301_base4_by_21_base4 :
  let q := 112
  let r := 0
  let q_base10 := 22
  divide_in_base4 = (q, r) ∧ q = 112 ∧ r = 0 ∧ q_base10 = 22 :=
sorry

end division_of_2301_base4_by_21_base4_l142_142488


namespace valid_raise_percentage_l142_142898

-- Define the conditions
def raise_between (x : ℝ) : Prop :=
  0.05 ≤ x ∧ x ≤ 0.10

def salary_increase_by_fraction (x : ℝ) : Prop :=
  x = 0.06

-- Define the main theorem 
theorem valid_raise_percentage (x : ℝ) (hx_between : raise_between x) (hx_fraction : salary_increase_by_fraction x) :
  x = 0.06 :=
sorry

end valid_raise_percentage_l142_142898


namespace maximum_sets_l142_142169

-- define the initial conditions
def dinner_forks : Nat := 6
def knives : Nat := dinner_forks + 9
def soup_spoons : Nat := 2 * knives
def teaspoons : Nat := dinner_forks / 2
def dessert_forks : Nat := teaspoons / 3
def butter_knives : Nat := 2 * dessert_forks

def max_capacity_g : Nat := 20000

def weight_dinner_fork : Nat := 80
def weight_knife : Nat := 100
def weight_soup_spoon : Nat := 85
def weight_teaspoon : Nat := 50
def weight_dessert_fork : Nat := 70
def weight_butter_knife : Nat := 65

-- Calculate the total weight of the existing cutlery
def total_weight_existing : Nat := 
  (dinner_forks * weight_dinner_fork) + 
  (knives * weight_knife) + 
  (soup_spoons * weight_soup_spoon) + 
  (teaspoons * weight_teaspoon) + 
  (dessert_forks * weight_dessert_fork) + 
  (butter_knives * weight_butter_knife)

-- Calculate the weight of one 2-piece cutlery set (1 knife + 1 dinner fork)
def weight_set : Nat := weight_knife + weight_dinner_fork

-- The remaining capacity in the drawer
def remaining_capacity_g : Nat := max_capacity_g - total_weight_existing

-- The maximum number of 2-piece cutlery sets that can be added
def max_2_piece_sets : Nat := remaining_capacity_g / weight_set

-- Theorem: maximum number of 2-piece cutlery sets that can be added is 84
theorem maximum_sets : max_2_piece_sets = 84 :=
by
  sorry

end maximum_sets_l142_142169


namespace count_four_digit_integers_divisible_by_15_l142_142644

theorem count_four_digit_integers_divisible_by_15 : 
  { n : Nat // 1000 ≤ n ∧ n < 10000 ∧ n % 15 = 0 }.card = 600 :=
by
  sorry

end count_four_digit_integers_divisible_by_15_l142_142644


namespace sum_reciprocals_factors_of_12_l142_142279

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l142_142279


namespace minimum_value_inequality_maximum_value_inequality_l142_142028

noncomputable def minimum_value (x1 x2 x3 : ℝ) : ℝ :=
  (x1 + 3 * x2 + 5 * x3) * (x1 + x2 / 3 + x3 / 5)

theorem minimum_value_inequality (x1 x2 x3 : ℝ) (h : 0 ≤ x1) (h : 0 ≤ x2) (h : 0 ≤ x3) (sum_eq : x1 + x2 + x3 = 1) :
  1 ≤ minimum_value x1 x2 x3 :=
sorry

theorem maximum_value_inequality (x1 x2 x3 : ℝ) (h : 0 ≤ x1) (h : 0 ≤ x2) (h : 0 ≤ x3) (sum_eq : x1 + x2 + x3 = 1) :
  minimum_value x1 x2 x3 ≤ 9/5 :=
sorry

end minimum_value_inequality_maximum_value_inequality_l142_142028


namespace max_points_of_intersection_l142_142227

-- Definitions from the conditions
def circles := 2
def lines := 3

-- Define the problem of the greatest intersection number
theorem max_points_of_intersection (c : ℕ) (l : ℕ) (h_c : c = circles) (h_l : l = lines) : 
  (2 + (l * 2 * c) + (l * (l - 1) / 2)) = 17 :=
by
  rw [h_c, h_l]
  -- We have 2 points from circle intersections
  -- 12 points from lines intersections with circles
  -- 3 points from lines intersections with lines
  -- Hence, 2 + 12 + 3 = 17
  exact Eq.refl 17

end max_points_of_intersection_l142_142227


namespace chalkboard_area_l142_142184

theorem chalkboard_area (width : ℝ) (h_w : width = 3) (h_l : 2 * width = length) : width * length = 18 := 
by 
  sorry

end chalkboard_area_l142_142184


namespace trapezoid_perimeter_is_183_l142_142671

-- Declare the lengths of the sides of the trapezoid
def EG : ℕ := 35
def FH : ℕ := 40
def GH : ℕ := 36

-- Declare the relation between the bases EF and GH
def EF : ℕ := 2 * GH

-- The statement of the problem
theorem trapezoid_perimeter_is_183 : EF = 72 ∧ (EG + GH + FH + EF) = 183 := by
  sorry

end trapezoid_perimeter_is_183_l142_142671


namespace Sue_made_22_buttons_l142_142178

def Mari_buttons : Nat := 8
def Kendra_buttons : Nat := 5 * Mari_buttons + 4
def Sue_buttons : Nat := Kendra_buttons / 2

theorem Sue_made_22_buttons : Sue_buttons = 22 :=
by
  -- proof to be added
  sorry

end Sue_made_22_buttons_l142_142178


namespace sum_reciprocal_factors_of_12_l142_142418

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l142_142418


namespace sum_reciprocals_12_l142_142244

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l142_142244


namespace work_days_A_l142_142460

theorem work_days_A (x : ℝ) (h1 : ∀ y : ℝ, y = 20) (h2 : ∀ z : ℝ, z = 5) 
  (h3 : ∀ w : ℝ, w = 0.41666666666666663) :
  x = 15 :=
  sorry

end work_days_A_l142_142460


namespace sequence_problem_l142_142913

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l142_142913


namespace sum_reciprocals_factors_12_l142_142272

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l142_142272


namespace problem1_l142_142117

variable (x : ℝ)

theorem problem1 : 5 * x^2 * x^4 + x^8 / (-x)^2 = 6 * x^6 :=
  sorry

end problem1_l142_142117


namespace value_of_x_l142_142770

theorem value_of_x (x : ℝ) : (9 - x) ^ 2 = x ^ 2 → x = 4.5 :=
by
  sorry

end value_of_x_l142_142770


namespace monotonic_m_range_l142_142532

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x - 12

-- Prove the range of m where f(x) is monotonic on [m, m+4]
theorem monotonic_m_range {m : ℝ} :
  (∀ x y : ℝ, m ≤ x ∧ x ≤ m + 4 ∧ m ≤ y ∧ y ≤ m + 4 → (x ≤ y → f x ≤ f y ∨ f x ≥ f y))
  ↔ (m ≤ -5 ∨ m ≥ 2) :=
sorry

end monotonic_m_range_l142_142532


namespace simplify_expression_l142_142784

theorem simplify_expression :
  6^6 + 6^6 + 6^6 + 6^6 + 6^6 + 6^6 = 6^7 :=
by sorry

end simplify_expression_l142_142784


namespace find_sum_a100_b100_l142_142866

-- Definitions of arithmetic sequences and their properties
structure arithmetic_sequence (an : ℕ → ℝ) :=
  (a1 : ℝ)
  (d : ℝ)
  (def_seq : ∀ n, an n = a1 + (n - 1) * d)

-- Given conditions
variables (a_n b_n : ℕ → ℝ)
variables (ha : arithmetic_sequence a_n)
variables (hb : arithmetic_sequence b_n)

-- Specified conditions
axiom cond1 : a_n 5 + b_n 5 = 3
axiom cond2 : a_n 9 + b_n 9 = 19

-- The goal to be proved
theorem find_sum_a100_b100 : a_n 100 + b_n 100 = 383 :=
sorry

end find_sum_a100_b100_l142_142866


namespace number_of_prize_orders_l142_142610

/-- At the end of a professional bowling tournament, the top 6 bowlers have a playoff.
    - #6 and #5 play a game. The loser receives the 6th prize and the winner plays #4.
    - The loser of the second game receives the 5th prize and the winner plays #3.
    - The loser of the third game receives the 4th prize and the winner plays #2.
    - The loser of the fourth game receives the 3rd prize and the winner plays #1.
    - The winner of the final game gets 1st prize and the loser gets 2nd prize.

    We want to determine the number of possible orders in which the bowlers can receive the prizes.
-/
theorem number_of_prize_orders : 2^5 = 32 := by
  sorry

end number_of_prize_orders_l142_142610


namespace gcd_840_1764_l142_142206

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l142_142206


namespace smallest_int_with_divisors_l142_142084

theorem smallest_int_with_divisors :
  ∃ n : ℕ, 
    (∀ m, n = 2^2 * m → 
      (∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 
      (m = p^3 * q) ∧ 
      (nat.divisors p^7).count 8) ∧ 
    nat.divisors_count n = 24 ∧ 
    (nat.divisors (2^2 * n)).count 8) (n = 2^2 * 3^3 * 5) :=
begin
  sorry
end

end smallest_int_with_divisors_l142_142084


namespace larger_integer_21_l142_142709

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l142_142709


namespace no_integer_solutions_l142_142498

theorem no_integer_solutions (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
by {
  sorry
}

end no_integer_solutions_l142_142498


namespace min_sixth_power_sin_cos_l142_142843

theorem min_sixth_power_sin_cos : ∀ x : ℝ, sin x ^ 6 + cos x ^ 6 ≥ 1 := 
by
  sorry

end min_sixth_power_sin_cos_l142_142843


namespace smallest_n_with_divisors_l142_142077

-- Definitions of the divisors
def d_total (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)
def d_even (a b c : ℕ) : ℕ := a * (b + 1) * (c + 1)
def d_odd (b c : ℕ) : ℕ := (b + 1) * (c + 1)

-- Math problem and proving smallest n
theorem smallest_n_with_divisors (a b c : ℕ) (n : ℕ) (h_1 : d_odd b c = 8) (h_2 : d_even a b c = 16) : n = 60 :=
  sorry

end smallest_n_with_divisors_l142_142077


namespace commutative_star_not_distributive_star_special_case_star_no_identity_star_not_associative_star_l142_142766

def binary_star (x y : ℝ) : ℝ := (x - 1) * (y - 1) - 1

-- Statement (A): Commutativity
theorem commutative_star (x y : ℝ) : binary_star x y = binary_star y x := sorry

-- Statement (B): Distributivity (proving it's not distributive)
theorem not_distributive_star (x y z : ℝ) : ¬(binary_star x (y + z) = binary_star x y + binary_star x z) := sorry

-- Statement (C): Special case
theorem special_case_star (x : ℝ) : binary_star (x + 1) (x - 1) = binary_star x x - 1 := sorry

-- Statement (D): Identity element
theorem no_identity_star (x e : ℝ) : ¬(binary_star x e = x ∧ binary_star e x = x) := sorry

-- Statement (E): Associativity (proving it's not associative)
theorem not_associative_star (x y z : ℝ) : ¬(binary_star x (binary_star y z) = binary_star (binary_star x y) z) := sorry

end commutative_star_not_distributive_star_special_case_star_no_identity_star_not_associative_star_l142_142766


namespace sum_of_reciprocals_of_factors_of_12_l142_142436

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142436


namespace arithmetic_sequence_minimum_value_S_n_l142_142938

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l142_142938


namespace max_intersections_two_circles_three_lines_l142_142229

theorem max_intersections_two_circles_three_lines :
  ∀ (C1 C2 : ℝ × ℝ × ℝ) (L1 L2 L3 : ℝ × ℝ × ℝ), 
  C1 ≠ C2 → L1 ≠ L2 → L2 ≠ L3 → L1 ≠ L3 →
  ∃ (P : ℕ), P = 17 :=
by 
  sorry

end max_intersections_two_circles_three_lines_l142_142229


namespace days_for_A_to_complete_work_l142_142459

theorem days_for_A_to_complete_work (
  B_work_days : ℕ,
  together_work_days : ℕ,
  work_left_fraction : ℝ
) : (B_work_days = 20) → (together_work_days = 5) → (work_left_fraction = 0.41666666666666663) → 
    ∃ x, 5 * (1 / x + 1 / 20) = 1 - 0.41666666666666663 ∧ x = 15 := 
by 
  intros hB ht hw 
  use 15
  sorry

end days_for_A_to_complete_work_l142_142459


namespace sum_reciprocals_factors_12_l142_142403

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142403


namespace problem_statement_l142_142802

theorem problem_statement (a b c : ℝ) (h : a * c^2 > b * c^2) (hc : c ≠ 0) : 
  a > b :=
by 
  sorry

end problem_statement_l142_142802


namespace min_sin6_cos6_l142_142847

theorem min_sin6_cos6 (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 :=
by
  sorry

end min_sin6_cos6_l142_142847


namespace total_emails_received_l142_142674

theorem total_emails_received (emails_morning emails_afternoon : ℕ) 
  (h1 : emails_morning = 3) 
  (h2 : emails_afternoon = 5) : 
  emails_morning + emails_afternoon = 8 := 
by 
  sorry

end total_emails_received_l142_142674


namespace volume_of_one_wedge_l142_142795

theorem volume_of_one_wedge 
  (circumference : ℝ)
  (h : circumference = 15 * Real.pi) 
  (radius : ℝ) 
  (volume : ℝ) 
  (wedge_volume : ℝ) 
  (h_radius : radius = 7.5)
  (h_volume : volume = (4 / 3) * Real.pi * radius^3)
  (h_wedge_volume : wedge_volume = volume / 5)
  : wedge_volume = 112.5 * Real.pi :=
by
  sorry

end volume_of_one_wedge_l142_142795


namespace computation_result_l142_142815

-- Define the vectors and scalar multiplications
def v1 : ℤ × ℤ := (3, -9)
def v2 : ℤ × ℤ := (2, -7)
def v3 : ℤ × ℤ := (-1, 4)

noncomputable def result : ℤ × ℤ := 
  let scalar_mult (m : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (m * v.1, m * v.2)
  scalar_mult 5 v1 - scalar_mult 3 v2 + scalar_mult 2 v3

-- The main theorem
theorem computation_result : result = (7, -16) :=
  by 
    -- Skip the proof as required
    sorry

end computation_result_l142_142815


namespace sum_reciprocals_12_l142_142239

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l142_142239


namespace dot_product_eq_negative_29_l142_142154

def vector := ℝ × ℝ

variables (a b : vector)

theorem dot_product_eq_negative_29 
  (h1 : a + b = (2, -4))
  (h2 : 3 * a - b = (-10, 16)) :
  a.1 * b.1 + a.2 * b.2 = -29 :=
sorry

end dot_product_eq_negative_29_l142_142154


namespace larger_integer_is_21_l142_142754

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l142_142754


namespace sum_reciprocals_factors_12_l142_142357

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l142_142357


namespace maria_earnings_l142_142959

-- Define the conditions
def costOfBrushes : ℕ := 20
def costOfCanvas : ℕ := 3 * costOfBrushes
def costPerLiterOfPaint : ℕ := 8
def litersOfPaintNeeded : ℕ := 5
def sellingPriceOfPainting : ℕ := 200

-- Define the total cost calculation
def totalCostOfMaterials : ℕ := costOfBrushes + costOfCanvas + (costPerLiterOfPaint * litersOfPaintNeeded)

-- Define the final earning calculation
def mariaEarning : ℕ := sellingPriceOfPainting - totalCostOfMaterials

-- State the theorem
theorem maria_earnings :
  mariaEarning = 80 := by
  sorry

end maria_earnings_l142_142959


namespace train_arrival_time_l142_142760

-- Define the time type
structure Time where
  hour : Nat
  minute : Nat

namespace Time

-- Define the addition of minutes to a time.
def add_minutes (t : Time) (m : Nat) : Time :=
  let new_minutes := t.minute + m
  if new_minutes < 60 then 
    { hour := t.hour, minute := new_minutes }
  else 
    { hour := t.hour + new_minutes / 60, minute := new_minutes % 60 }

-- Define the departure time
def departure_time : Time := { hour := 9, minute := 45 }

-- Define the travel time in minutes
def travel_time : Nat := 15

-- Define the expected arrival time
def expected_arrival_time : Time := { hour := 10, minute := 0 }

-- The theorem we need to prove
theorem train_arrival_time:
  add_minutes departure_time travel_time = expected_arrival_time := by
  sorry

end train_arrival_time_l142_142760


namespace find_amount_l142_142654

theorem find_amount (x : ℝ) (h1 : 0.25 * x = 0.15 * 1500 - 30) (h2 : x = 780) : 30 = 30 :=
by
  sorry

end find_amount_l142_142654


namespace TrainTravelDays_l142_142111

-- Definition of the problem conditions
def train_start (days: ℕ) : ℕ := 
  if days = 0 then 0 -- no trains to meet on the first day
  else days -- otherwise, meet 'days' number of trains

/-- 
  Prove that if a train comes across 4 trains on its way from Amritsar to Bombay and starts at 9 am, 
  then it takes 5 days for the train to reach its destination.
-/
theorem TrainTravelDays (meet_train_count : ℕ) : meet_train_count = 4 → train_start (meet_train_count) + 1 = 5 :=
by
  intro h
  rw [h]
  sorry

end TrainTravelDays_l142_142111


namespace positional_relationship_l142_142651

-- Definitions of skew_lines and parallel_lines
def skew_lines (a b : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, ¬ (a x y ∨ b x y) 

def parallel_lines (a c : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∀ x y, c x y = a (k * x) (k * y)

-- Theorem statement
theorem positional_relationship (a b c : ℝ → ℝ → Prop) 
  (h1 : skew_lines a b) 
  (h2 : parallel_lines a c) : 
  skew_lines c b ∨ (∃ x y, c x y ∧ b x y) :=
sorry

end positional_relationship_l142_142651


namespace curve_symmetric_about_y_eq_x_l142_142978

def curve_eq (x y : ℝ) : Prop := x * y * (x + y) = 1

theorem curve_symmetric_about_y_eq_x :
  ∀ (x y : ℝ), curve_eq x y ↔ curve_eq y x :=
by sorry

end curve_symmetric_about_y_eq_x_l142_142978


namespace sum_of_remainders_l142_142818

theorem sum_of_remainders (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 5) 
  (h3 : c % 30 = 20) : 
  (a + b + c) % 30 = 10 := 
by sorry

end sum_of_remainders_l142_142818


namespace smallest_number_l142_142188

theorem smallest_number (x y z : ℕ) (h1 : y = 4 * x) (h2 : z = 2 * y) 
(h3 : (x + y + z) / 3 = 78) : x = 18 := 
by 
    sorry

end smallest_number_l142_142188


namespace sum_reciprocals_factors_12_l142_142356

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l142_142356


namespace sum_reciprocals_factors_12_l142_142361

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l142_142361


namespace cube_inequality_l142_142516

theorem cube_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := 
sorry

end cube_inequality_l142_142516


namespace radius_of_shorter_cone_l142_142066

theorem radius_of_shorter_cone {h : ℝ} (h_ne_zero : h ≠ 0) :
  ∀ r : ℝ, ∀ V_taller V_shorter : ℝ,
   (V_taller = (1/3) * π * (5 ^ 2) * (4 * h)) →
   (V_shorter = (1/3) * π * (r ^ 2) * h) →
   V_taller = V_shorter →
   r = 10 :=
by
  intros
  sorry

end radius_of_shorter_cone_l142_142066


namespace pow_addition_l142_142592

theorem pow_addition : (-2)^2 + 2^2 = 8 :=
by
  sorry

end pow_addition_l142_142592


namespace necessary_and_sufficient_condition_l142_142648

theorem necessary_and_sufficient_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : (a + b > a * b) ↔ (a = 1 ∨ b = 1) := 
sorry

end necessary_and_sufficient_condition_l142_142648


namespace arithmetic_sequence_and_minimum_sum_l142_142919

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l142_142919


namespace range_of_a_l142_142519

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x = 3 ∧ 3 * x - (a * x + 1) / 2 < 4 * x / 3) → a > 3 :=
by
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  sorry

end range_of_a_l142_142519


namespace neil_initial_games_l142_142642

theorem neil_initial_games (N : ℕ) 
  (H₀ : ℕ) (H₀_eq : H₀ = 58)
  (H₁ : ℕ) (H₁_eq : H₁ = H₀ - 6)
  (H₁_condition : H₁ = 4 * (N + 6)) : N = 7 :=
by {
  -- Substituting the given values and simplifying to show the final equation
  sorry
}

end neil_initial_games_l142_142642


namespace probability_heart_seven_jack_l142_142764

def prob_heart_seven_jack (total : ℕ) (hearts : ℕ) (sevens : ℕ) (jacks : ℕ) : ℚ :=
  (hearts / total) * (sevens / (total - 1)) * (jacks / (total - 2))

def prob_cases : ℚ :=
  (12 / 52) * (3 / 51) * (4 / 50) + 
  (11 / 52) * (1 / 51) * (3 / 50) + 
  (1 / 52) * (3 / 51) * (4 / 50) + 
  (1 / 52) * (1 / 51) * (3 / 50)

theorem probability_heart_seven_jack : prob_cases = 8 / 5525 := 
  by linarith

end probability_heart_seven_jack_l142_142764


namespace find_other_percentage_l142_142896

noncomputable def percentage_other_investment
  (total_investment : ℝ)
  (investment_10_percent : ℝ)
  (total_interest : ℝ)
  (interest_rate_10_percent : ℝ)
  (other_investment_interest : ℝ) : ℝ :=
  let interest_10_percent := investment_10_percent * interest_rate_10_percent
  let interest_other_investment := total_interest - interest_10_percent
  let amount_other_percentage := total_investment - investment_10_percent
  interest_other_investment / amount_other_percentage

theorem find_other_percentage :
  ∀ (total_investment : ℝ)
    (investment_10_percent : ℝ)
    (total_interest : ℝ)
    (interest_rate_10_percent : ℝ),
    total_investment = 31000 ∧
    investment_10_percent = 12000 ∧
    total_interest = 1390 ∧
    interest_rate_10_percent = 0.1 →
    percentage_other_investment total_investment investment_10_percent total_interest interest_rate_10_percent 190 = 0.01 :=
by
  intros total_investment investment_10_percent total_interest interest_rate_10_percent h
  sorry

end find_other_percentage_l142_142896


namespace minimum_value_expression_l142_142635

theorem minimum_value_expression (F M N : ℝ × ℝ) (x y : ℝ) (a : ℝ) (k : ℝ) :
  (y ^ 2 = 16 * x ∧ F = (4, 0) ∧ l = (k * (x - 4), y) ∧ (M = (x₁, y₁) ∧ N = (x₂, y₂)) ∧
  0 ≤ x₁ ∧ y₁ ^ 2 = 16 * x₁ ∧ 0 ≤ x₂ ∧ y₂ ^ 2 = 16 * x₂) →
  (abs (dist F N) / 9 - 4 / abs (dist F M) ≥ 1 / 3) :=
sorry -- proof will be provided

end minimum_value_expression_l142_142635


namespace max_value_f_l142_142707

noncomputable def f (x : ℝ) : ℝ := Real.sin (2*x) - 2 * Real.sqrt 3 * (Real.sin x)^2

theorem max_value_f : ∃ x : ℝ, f x = 2 - Real.sqrt 3 :=
  sorry

end max_value_f_l142_142707


namespace no_integer_solutions_l142_142497

theorem no_integer_solutions (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
by {
  sorry
}

end no_integer_solutions_l142_142497


namespace sum_reciprocals_factors_12_l142_142400

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142400


namespace ivan_max_13_bars_a_ivan_max_13_bars_b_l142_142801

variable (n : ℕ) (ivan_max_bags : ℕ)

-- Condition 1: initial count of bars in the chest
def initial_bars := 13

-- Condition 2: function to check if transfers are possible
def can_transfer (bars_in_chest : ℕ) (bars_in_bag : ℕ) (last_transfer : ℕ) : Prop :=
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ t₁ ≠ last_transfer ∧ t₂ ≠ last_transfer ∧
           t₁ + bars_in_bag ≤ initial_bars ∧ bars_in_chest - t₁ + t₂ = bars_in_chest

-- Proof Problem (a): Given initially 13 bars, prove Ivan can secure 13 bars
theorem ivan_max_13_bars_a 
  (initial_bars : ℕ := 13) 
  (target_bars : ℕ := 13)
  (can_transfer : ∀ (bars_in_chest bars_in_bag last_transfer : ℕ), can_transfer bars_in_chest bars_in_bag last_transfer) 
  (h_initial_bars : initial_bars = 13) :
  ivan_max_bags = target_bars :=
by
  sorry

-- Proof Problem (b): Given initially 14 bars, prove Ivan can secure 13 bars
theorem ivan_max_13_bars_b 
  (initial_bars : ℕ := 14)
  (target_bars : ℕ := 13)
  (can_transfer : ∀ (bars_in_chest bars_in_bag last_transfer : ℕ), can_transfer bars_in_chest bars_in_bag last_transfer) 
  (h_initial_bars : initial_bars = 14) :
  ivan_max_bags = target_bars :=
by
  sorry

end ivan_max_13_bars_a_ivan_max_13_bars_b_l142_142801


namespace sum_of_x_and_y_l142_142658

-- Define integers x and y
variables (x y : ℤ)

-- Define conditions
def condition1 : Prop := x - y = 200
def condition2 : Prop := y = 250

-- Define the main statement
theorem sum_of_x_and_y (h1 : condition1 x y) (h2 : condition2 y) : x + y = 700 := 
by
  sorry

end sum_of_x_and_y_l142_142658


namespace simplify_expression1_simplify_expression2_l142_142197

-- Problem 1
theorem simplify_expression1 (x y : ℤ) :
  (-3) * x + 2 * y - 5 * x - 7 * y = -8 * x - 5 * y :=
by sorry

-- Problem 2
theorem simplify_expression2 (a b : ℤ) :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = 3 * a^2 * b - a * b^2 :=
by sorry

end simplify_expression1_simplify_expression2_l142_142197


namespace Nicky_time_before_catchup_l142_142966

-- Define the given speeds and head start time as constants
def v_C : ℕ := 5 -- Cristina's speed in meters per second
def v_N : ℕ := 3 -- Nicky's speed in meters per second
def t_H : ℕ := 12 -- Head start in seconds

-- Define the running time until catch up
def time_Nicky_run : ℕ := t_H + (36 / (v_C - v_N))

-- Prove that the time Nicky has run before Cristina catches up to him is 30 seconds
theorem Nicky_time_before_catchup : time_Nicky_run = 30 :=
by
  -- Add the steps for the proof
  sorry

end Nicky_time_before_catchup_l142_142966


namespace sum_reciprocals_factors_12_l142_142336

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142336


namespace polygonal_line_exists_l142_142902

theorem polygonal_line_exists (A : Type) (n q : ℕ) (lengths : Fin q → ℝ)
  (yellow_segments : Fin q → (A × A))
  (h_lengths : ∀ i j : Fin q, i < j → lengths i < lengths j)
  (h_yellow_segments_unique : ∀ i j : Fin q, i ≠ j → yellow_segments i ≠ yellow_segments j) :
  ∃ (m : ℕ), m ≥ 2 * q / n :=
sorry

end polygonal_line_exists_l142_142902


namespace smallest_possible_r_l142_142551

theorem smallest_possible_r (p q r : ℤ) (hpq: p < q) (hqr: q < r) 
  (hgeo: q^2 = p * r) (harith: 2 * q = p + r) : r = 4 :=
sorry

end smallest_possible_r_l142_142551


namespace find_B_and_b_range_l142_142887

variables {a b c : ℝ} {A B C : ℝ}

def triangle_ABC (A B C a b c : ℝ) : Prop :=
  A + B + C = π ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
  a = 2 * sin (A/2) * cos (B/2 + C/2) ∧
  b = 2 * sin (B/2) * cos (A/2 + C/2) ∧
  c = 2 * sin (C/2) * cos (A/2 + B/2)

theorem find_B_and_b_range
  (h1 : triangle_ABC A B C a b c)
  (h2 : cos C + (cos A - real.sqrt 3 * sin A) * cos B = 0)
  (h3 : a + c = 1) :
  B = π / 3 ∧ (1 / 2) ≤ b ∧ b < 1 := 
sorry

end find_B_and_b_range_l142_142887


namespace sector_area_l142_142886

theorem sector_area (theta r : ℝ) (h1 : theta = 2 * Real.pi / 3) (h2 : r = 2) :
  (1 / 2 * r ^ 2 * theta) = 4 * Real.pi / 3 := by
  sorry

end sector_area_l142_142886


namespace max_intersections_two_circles_three_lines_l142_142230

theorem max_intersections_two_circles_three_lines :
  ∀ (C1 C2 : ℝ × ℝ × ℝ) (L1 L2 L3 : ℝ × ℝ × ℝ), 
  C1 ≠ C2 → L1 ≠ L2 → L2 ≠ L3 → L1 ≠ L3 →
  ∃ (P : ℕ), P = 17 :=
by 
  sorry

end max_intersections_two_circles_three_lines_l142_142230


namespace movie_marathon_duration_l142_142996

theorem movie_marathon_duration :
  let first_movie := 2
  let second_movie := first_movie + 0.5 * first_movie
  let combined_time := first_movie + second_movie
  let third_movie := combined_time - 1
  first_movie + second_movie + third_movie = 9 := by
  sorry

end movie_marathon_duration_l142_142996


namespace domain_of_sqrt_cos_function_l142_142044

theorem domain_of_sqrt_cos_function:
  (∀ k : ℤ, ∀ x : ℝ, 2 * Real.cos x + 1 ≥ 0 ↔ x ∈ Set.Icc (2 * k * Real.pi - 2 * Real.pi / 3) (2 * k * Real.pi + 2 * Real.pi / 3)) :=
by
  sorry

end domain_of_sqrt_cos_function_l142_142044


namespace arithmetic_progression_cubic_eq_l142_142193

theorem arithmetic_progression_cubic_eq (x y z u : ℤ) (d : ℤ) :
  (x, y, z, u) = (3 * d, 4 * d, 5 * d, 6 * d) →
  x^3 + y^3 + z^3 = u^3 →
  ∃ d : ℤ, x = 3 * d ∧ y = 4 * d ∧ z = 5 * d ∧ u = 6 * d :=
by sorry

end arithmetic_progression_cubic_eq_l142_142193


namespace larger_integer_is_21_l142_142734

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l142_142734


namespace plane_figures_l142_142636

def polyline_two_segments : Prop := -- Definition for a polyline composed of two line segments
  sorry

def polyline_three_segments : Prop := -- Definition for a polyline composed of three line segments
  sorry

def closed_three_segments : Prop := -- Definition for a closed figure composed of three line segments
  sorry

def quadrilateral_equal_opposite_sides : Prop := -- Definition for a quadrilateral with equal opposite sides
  sorry

def trapezoid : Prop := -- Definition for a trapezoid
  sorry

def is_plane_figure (fig : Prop) : Prop :=
  sorry  -- Axiom or definition that determines whether a figure is a plane figure.

-- Translating the proof problem
theorem plane_figures :
  is_plane_figure polyline_two_segments ∧
  ¬ is_plane_figure polyline_three_segments ∧
  is_plane_figure closed_three_segments ∧
  ¬ is_plane_figure quadrilateral_equal_opposite_sides ∧
  is_plane_figure trapezoid :=
by
  sorry

end plane_figures_l142_142636


namespace problem_statement_l142_142571

noncomputable theory

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def domain_of_f_div_2 : set ℝ := { x | 2 ≤ x ∧ x ≤ 4 }

theorem problem_statement
  (f1 : ℝ → ℝ)
  (f2 : ℝ → ℝ)
  (f3 : ℝ → ℝ)
  (f4 : ℝ → ℝ)
  (h1 : f1 = λ x, x^2 - (1 / x^2))
  (h2 : ∀ x, f2 (x / 2) ∈ domain_of_f_div_2 → x ∈ Icc 10 100)
  (h3 : ∀ x, -x^2 + 4*x + 5 > 0 → f3 x = log (-x^2 + 4*x + 5))
  (h4 : (∀ x, f4 x = sqrt (2^(mx^2 + 4*mx + 3) - 1)) → (0 ≤ m ∧ m ≤ 3 / 4)) :
  4 = 4 :=
sorry

end problem_statement_l142_142571


namespace polygon_sides_l142_142167

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1440) : n = 10 :=
by sorry

end polygon_sides_l142_142167


namespace larger_integer_21_l142_142713

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l142_142713


namespace complement_union_correct_l142_142957

open Set

variable (U A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3, 4})
variable (hA : A = {0, 1, 2})
variable (hB : B = {2, 3})

theorem complement_union_correct :
  (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end complement_union_correct_l142_142957


namespace beta_minus_alpha_l142_142641

open Real

noncomputable def vector_a (α : ℝ) := (cos α, sin α)
noncomputable def vector_b (β : ℝ) := (cos β, sin β)

theorem beta_minus_alpha (α β : ℝ)
  (h₁ : 0 < α)
  (h₂ : α < β)
  (h₃ : β < π)
  (h₄ : |2 * vector_a α + vector_b β| = |vector_a α - 2 * vector_b β|) :
  β - α = π / 2 :=
sorry

end beta_minus_alpha_l142_142641


namespace proof_problem_l142_142155

variable {R : Type*} [LinearOrderedField R]

theorem proof_problem 
  (a1 a2 a3 b1 b2 b3 : R)
  (h1 : a1 < a2) (h2 : a2 < a3) (h3 : b1 < b2) (h4 : b2 < b3)
  (h_sum : a1 + a2 + a3 = b1 + b2 + b3)
  (h_pair_sum : a1 * a2 + a1 * a3 + a2 * a3 = b1 * b2 + b1 * b3 + b2 * b3)
  (h_a1_lt_b1 : a1 < b1) :
  (b2 < a2) ∧ (a3 < b3) ∧ (a1 * a2 * a3 < b1 * b2 * b3) ∧ ((1 - a1) * (1 - a2) * (1 - a3) > (1 - b1) * (1 - b2) * (1 - b3)) :=
by {
  sorry
}

end proof_problem_l142_142155


namespace sum_of_reciprocals_of_factors_of_12_l142_142299

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142299


namespace min_value_sin6_cos6_l142_142841

theorem min_value_sin6_cos6 (x : ℝ) : 
  let s := sin x
      c := cos x in
  s^2 + c^2 = 1 → 
  ∃ y, y = s^6 + c^6 ∧ y = 1/4 :=
by
  sorry

end min_value_sin6_cos6_l142_142841


namespace angle_SVU_l142_142892

theorem angle_SVU (TU SV SU : ℝ) (angle_STU_T : ℝ) (angle_STU_S : ℝ) :
  TU = SV → angle_STU_T = 75 → angle_STU_S = 30 →
  TU = SU → SU = SV → S_V_U = 65 :=
by
  intros H1 H2 H3 H4 H5
  -- skip proof
  sorry

end angle_SVU_l142_142892


namespace time_until_meeting_l142_142065

theorem time_until_meeting (v1 v2 : ℝ) (t2 t1 : ℝ) 
    (h1 : v1 = 6) 
    (h2 : v2 = 4) 
    (h3 : t2 = 10)
    (h4 : v2 * t1 = v1 * (t1 - t2)) : t1 = 30 := 
sorry

end time_until_meeting_l142_142065


namespace geometric_sequence_a7_l142_142891

theorem geometric_sequence_a7
  (a : ℕ → ℤ)
  (is_geom_seq : ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r)
  (h1 : a 1 = -16)
  (h4 : a 4 = 8) :
  a 7 = -4 := 
sorry

end geometric_sequence_a7_l142_142891


namespace cheryl_initial_skitttles_l142_142483

-- Given conditions
def cheryl_ends_with (ends_with : ℕ) : Prop := ends_with = 97
def kathryn_gives (gives : ℕ) : Prop := gives = 89

-- To prove: cheryl_starts_with + kathryn_gives = cheryl_ends_with
theorem cheryl_initial_skitttles (cheryl_starts_with : ℕ) :
  (∃ ends_with gives, cheryl_ends_with ends_with ∧ kathryn_gives gives ∧ 
  cheryl_starts_with + gives = ends_with) →
  cheryl_starts_with = 8 :=
by
  sorry

end cheryl_initial_skitttles_l142_142483


namespace larger_integer_value_l142_142722

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l142_142722


namespace allocation_methods_count_l142_142661

theorem allocation_methods_count :
  (∃ g : ℕ → ℕ, g 1 + g 2 + g 3 = 5 ∧ g 1 > 0 ∧ g 2 > 0 ∧ g 3 > 0 ∧ 
    ∃ n k : ℕ, n = 4 ∧ k = 2 ∧ C(n, k) = 6) :=
begin
  sorry
end

end allocation_methods_count_l142_142661


namespace smallest_integer_with_divisors_properties_l142_142086

def number_of_odd_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 1)

def number_of_even_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 0)

theorem smallest_integer_with_divisors_properties :
  ∃ n : ℕ, number_of_odd_divisors n = 8 ∧ number_of_even_divisors n = 16 ∧ n = 4000 :=
by
  sorry

end smallest_integer_with_divisors_properties_l142_142086


namespace find_larger_integer_l142_142750

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l142_142750


namespace ratio_of_areas_l142_142220

theorem ratio_of_areas (OR : ℝ) (h : OR > 0) :
  let OY := (1 / 3) * OR
  let area_OY := π * OY^2
  let area_OR := π * OR^2
  (area_OY / area_OR) = (1 / 9) :=
by
  -- Definitions
  let OY := (1 / 3) * OR
  let area_OY := π * OY^2
  let area_OR := π * OR^2
  sorry

end ratio_of_areas_l142_142220


namespace completing_the_square_correct_l142_142771

theorem completing_the_square_correct :
  (∃ x : ℝ, x^2 - 6 * x + 5 = 0) →
  (∃ x : ℝ, (x - 3)^2 = 4) :=
by
  sorry

end completing_the_square_correct_l142_142771


namespace simplify_polynomial_l142_142195

theorem simplify_polynomial (q : ℤ) :
  (4*q^4 - 2*q^3 + 3*q^2 - 7*q + 9) + (5*q^3 - 8*q^2 + 6*q - 1) =
  4*q^4 + 3*q^3 - 5*q^2 - q + 8 :=
sorry

end simplify_polynomial_l142_142195


namespace tangent_line_eq_l142_142619

theorem tangent_line_eq
    (f : ℝ → ℝ) (f_def : ∀ x, f x = x ^ 2)
    (tangent_point : ℝ × ℝ) (tangent_point_def : tangent_point = (1, 1))
    (f' : ℝ → ℝ) (f'_def : ∀ x, f' x = 2 * x)
    (slope_at_1 : f' 1 = 2) :
    ∃ (a b : ℝ), a = 2 ∧ b = -1 ∧ ∀ x y, y = a * x + b ↔ (2 * x - y - 1 = 0) :=
sorry

end tangent_line_eq_l142_142619


namespace ratio_ac_l142_142097

-- Definitions based on conditions
variables (a b c : ℕ)
variables (x y : ℕ)

-- Conditions
def ratio_ab := (a : ℚ) / (b : ℚ) = 2 / 3
def ratio_bc := (b : ℚ) / (c : ℚ) = 1 / 5

-- Theorem to prove the desired ratio
theorem ratio_ac (h1 : ratio_ab a b) (h2 : ratio_bc b c) : (a : ℚ) / (c : ℚ) = 2 / 15 :=
by
  sorry

end ratio_ac_l142_142097


namespace sum_of_reciprocals_of_factors_of_12_l142_142314

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142314


namespace sum_of_reciprocals_factors_12_l142_142424

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l142_142424


namespace cone_base_radius_l142_142990

-- Definitions based on conditions
def sphere_radius : ℝ := 1
def cone_height : ℝ := 2

-- Problem statement
theorem cone_base_radius {r : ℝ} 
  (h1 : ∀ x y z : ℝ, (x = sphere_radius ∧ y = sphere_radius ∧ z = sphere_radius) → 
                     (x + y + z = 3 * sphere_radius)) 
  (h2 : ∃ (O O1 O2 O3 : ℝ), (O = 0) ∧ (O1 = 1) ∧ (O2 = 1) ∧ (O3 = 1)) 
  (h3 : ∀ x y z : ℝ, (x + y + z = 3 * sphere_radius) → 
                     (y = z) → (x = z) → y * z + x * z + x * y = 3 * sphere_radius ^ 2)
  (h4 : ∀ h : ℝ, h = cone_height) :
  r = (Real.sqrt 3 / 6) :=
sorry

end cone_base_radius_l142_142990


namespace sum_reciprocals_factors_of_12_l142_142288

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l142_142288


namespace problem1_problem2_l142_142168

noncomputable def triangle_boscos_condition (a b c A B : ℝ) : Prop :=
  b * Real.cos A = (2 * c + a) * Real.cos (Real.pi - B)

noncomputable def triangle_area (a b c : ℝ) (S : ℝ) : Prop :=
  S = (1 / 2) * a * c * Real.sin (2 * Real.pi / 3)

noncomputable def triangle_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
  P = b + a + c

theorem problem1 (a b c A : ℝ) (h : triangle_boscos_condition a b c A (2 * Real.pi / 3)) : 
  ∃ B : ℝ, B = 2 * Real.pi / 3 :=
by
  sorry

theorem problem2 (a c : ℝ) (b : ℝ := 4) (area : ℝ := Real.sqrt 3) (P : ℝ) (h : triangle_area a b c area) (h_perim : triangle_perimeter a b c P) :
  ∃ x : ℝ, x = 4 + 2 * Real.sqrt 5 :=
by
  sorry

end problem1_problem2_l142_142168


namespace length_inequality_l142_142132

noncomputable def l_a (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def l_b (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def l_c (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def perimeter (A B C : ℝ) : ℝ :=
  A + B + C

theorem length_inequality (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0) :
  (l_a A B C * l_b A B C * l_c A B C) / (perimeter A B C)^3 ≤ 1 / 64 :=
by
  sorry

end length_inequality_l142_142132


namespace intersection_A_complement_B_l142_142030

def A := { x : ℝ | x ≥ -1 }
def B := { x : ℝ | x > 2 }
def complement_B := { x : ℝ | x ≤ 2 }

theorem intersection_A_complement_B :
  A ∩ complement_B = { x : ℝ | -1 ≤ x ∧ x ≤ 2 } :=
sorry

end intersection_A_complement_B_l142_142030


namespace martha_makes_40_cookies_martha_needs_7_5_cups_l142_142962

theorem martha_makes_40_cookies :
  (24 / 3) * 5 = 40 :=
by
  sorry

theorem martha_needs_7_5_cups :
  60 / (24 / 3) = 7.5 :=
by
  sorry

end martha_makes_40_cookies_martha_needs_7_5_cups_l142_142962


namespace sum_of_reciprocals_factors_12_l142_142305

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l142_142305


namespace sugar_bought_l142_142209

noncomputable def P : ℝ := 0.50
noncomputable def S : ℝ := 2.0

theorem sugar_bought : 
  (1.50 * S + 5 * P = 5.50) ∧ 
  (3 * 1.50 + P = 5) ∧
  ((1.50 : ℝ) ≠ 0) → (S = 2) :=
by
  sorry

end sugar_bought_l142_142209


namespace geometric_sequence_common_ratio_l142_142523

-- Define the geometric sequence with properties
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n < a (n + 1)

-- Main theorem
theorem geometric_sequence_common_ratio {a : ℕ → ℝ} {q : ℝ} (h_seq : increasing_geometric_sequence a q) (h_a1 : a 0 > 0) (h_eqn : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l142_142523


namespace max_intersections_l142_142226

-- Define the number of circles and lines
def num_circles : ℕ := 2
def num_lines : ℕ := 3

-- Define the maximum number of intersection points of circles
def max_circle_intersections : ℕ := 2

-- Define the number of intersection points between each line and each circle
def max_line_circle_intersections : ℕ := 2

-- Define the number of intersection points among lines (using the combination formula)
def num_line_intersections : ℕ := (num_lines.choose 2)

-- Define the greatest number of points of intersection
def total_intersections : ℕ :=
  max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections

-- Prove the greatest number of points of intersection is 17
theorem max_intersections : total_intersections = 17 := by
  -- Calculating individual parts for clarity
  have h1: max_circle_intersections = 2 := rfl
  have h2: num_lines * num_circles * max_line_circle_intersections = 12 := by
    calc
      num_lines * num_circles * max_line_circle_intersections
        = 3 * 2 * 2 := by rw [num_lines, num_circles, max_line_circle_intersections]
        ... = 12 := by norm_num
  have h3: num_line_intersections = 3 := by
    calc
      num_line_intersections = (3.choose 2) := rfl
      ... = 3 := by norm_num

  -- Adding the parts to get the total intersections
  calc
    total_intersections
      = max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections := rfl
      ... = 2 + 12 + 3 := by rw [h1, h2, h3]
      ... = 17 := by norm_num

end max_intersections_l142_142226


namespace find_c_l142_142868

noncomputable def cubic_function (x : ℝ) (c : ℝ) : ℝ :=
  x^3 - 3 * x + c

theorem find_c (c : ℝ) :
  (∃ x₁ x₂ : ℝ, cubic_function x₁ c = 0 ∧ cubic_function x₂ c = 0 ∧ x₁ ≠ x₂) →
  (c = -2 ∨ c = 2) :=
by
  sorry

end find_c_l142_142868


namespace sum_of_coefficients_l142_142491

noncomputable def expand_and_sum_coefficients (d : ℝ) : ℝ :=
  let poly := -2 * (4 - d) * (d + 3 * (4 - d))
  let expanded := -4 * d^2 + 40 * d - 96
  let sum_coefficients := (-4) + 40 + (-96)
  sum_coefficients

theorem sum_of_coefficients (d : ℝ) : expand_and_sum_coefficients d = -60 := by
  sorry

end sum_of_coefficients_l142_142491


namespace sum_reciprocals_factors_12_l142_142360

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l142_142360


namespace percent_of_x_eq_21_percent_l142_142446

theorem percent_of_x_eq_21_percent (x : Real) : (0.21 * x = 0.30 * 0.70 * x) := by
  sorry

end percent_of_x_eq_21_percent_l142_142446


namespace product_of_two_numbers_l142_142051

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 26) (h2 : x - y = 8) : x * y = 153 :=
by
  -- proof goes here
  sorry

end product_of_two_numbers_l142_142051


namespace tim_movie_marathon_duration_l142_142993

-- Define the durations of each movie
def first_movie_duration : ℕ := 2

def second_movie_duration : ℕ := 
  first_movie_duration + (first_movie_duration / 2)

def combined_first_two_movies_duration : ℕ :=
  first_movie_duration + second_movie_duration

def last_movie_duration : ℕ := 
  combined_first_two_movies_duration - 1

-- Define the total movie marathon duration
def total_movie_marathon_duration : ℕ := 
  first_movie_duration + second_movie_duration + last_movie_duration

-- Problem statement to be proved
theorem tim_movie_marathon_duration : total_movie_marathon_duration = 9 := by
  sorry

end tim_movie_marathon_duration_l142_142993


namespace solution_set_correct_l142_142212

theorem solution_set_correct (a b : ℝ) :
  (∀ x : ℝ, - 1 / 2 < x ∧ x < 1 / 3 → ax^2 + bx + 2 > 0) →
  (a - b = -10) :=
by
  sorry

end solution_set_correct_l142_142212


namespace sum_reciprocals_factors_12_l142_142355

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l142_142355


namespace ratio_x_y_l142_142232

theorem ratio_x_y (x y : ℤ) (h : (8 * x - 5 * y) * 3 = (11 * x - 3 * y) * 2) :
  x / y = 9 / 2 := by
  sorry

end ratio_x_y_l142_142232


namespace circles_tangent_l142_142703

theorem circles_tangent (m : ℝ) :
  (∀ (x y : ℝ), (x - m)^2 + (y + 2)^2 = 9 → 
                (x + 1)^2 + (y - m)^2 = 4 →
                ∃ m, m = -1 ∨ m = -2) := 
sorry

end circles_tangent_l142_142703


namespace sequence_problem_l142_142911

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def form_geometric_sequence (a : ℕ → ℤ) (n m k : ℕ) : Prop :=
  a m ^ 2 = a n * a k

def min_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) : ℤ :=
  ((S 12) < (S 13) → -78) ∧ ((S 12) ≥ (S 13) → -78)

axiom sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a i

theorem sequence_problem
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (h1 : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1)
    (h2 : form_geometric_sequence a 3 6 8)
    (h3 : sum_first_n_terms a S) :
    (is_arithmetic_sequence a) ∧ (min_S_n a S = -78) :=
begin
  sorry
end

end sequence_problem_l142_142911


namespace stormi_additional_money_needed_l142_142561

noncomputable def earnings_from_jobs : ℝ :=
  let washing_cars := 5 * 8.50
  let walking_dogs := 4 * 6.75
  let mowing_lawns := 3 * 12.25
  let gardening := 2 * 7.40
  washing_cars + walking_dogs + mowing_lawns + gardening

noncomputable def discounted_prices : ℝ :=
  let bicycle := 150.25 * (1 - 0.15)
  let helmet := 35.75 - 5.00
  let lock := 24.50
  bicycle + helmet + lock

noncomputable def total_cost_after_tax : ℝ :=
  let cost_before_tax := discounted_prices
  cost_before_tax * 1.05

noncomputable def amount_needed : ℝ :=
  total_cost_after_tax - earnings_from_jobs

theorem stormi_additional_money_needed : amount_needed = 71.06 := by
  sorry

end stormi_additional_money_needed_l142_142561


namespace sum_reciprocals_factors_12_l142_142269

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l142_142269


namespace cost_price_one_meter_l142_142095

theorem cost_price_one_meter (selling_price : ℤ) (total_meters : ℤ) (profit_per_meter : ℤ) 
  (h1 : selling_price = 6788) (h2 : total_meters = 78) (h3 : profit_per_meter = 29) : 
  (selling_price - (profit_per_meter * total_meters)) / total_meters = 58 := 
by 
  sorry

end cost_price_one_meter_l142_142095


namespace find_number_l142_142781

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 126) : x = 5600 := 
by
  -- Proof goes here
  sorry

end find_number_l142_142781


namespace sum_reciprocals_12_l142_142236

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l142_142236


namespace Jorge_goals_total_l142_142682

theorem Jorge_goals_total : 
  let last_season_goals := 156
  let this_season_goals := 187
  last_season_goals + this_season_goals = 343 := 
by
  sorry

end Jorge_goals_total_l142_142682


namespace symm_diff_complement_symm_diff_union_subset_symm_diff_inter_subset_l142_142039

-- Definitions of sequences of events and symmetric difference
variable (A : ℕ → Set α) (B : ℕ → Set α)

-- Definition of symmetric difference
def symm_diff (S T : Set α) : Set α := (S \ T) ∪ (T \ S)

-- Theorems to be proven
theorem symm_diff_complement (A1 B1 : Set α) :
  symm_diff A1 B1 = symm_diff (Set.compl A1) (Set.compl B1) := sorry

theorem symm_diff_union_subset :
  symm_diff (⋃ n, A n) (⋃ n, B n) ⊆ ⋃ n, symm_diff (A n) (B n) := sorry

theorem symm_diff_inter_subset :
  symm_diff (⋂ n, A n) (⋂ n, B n) ⊆ ⋃ n, symm_diff (A n) (B n) := sorry

end symm_diff_complement_symm_diff_union_subset_symm_diff_inter_subset_l142_142039


namespace arithmetic_sequence_minimum_value_S_n_l142_142937

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l142_142937


namespace sum_reciprocal_factors_12_l142_142247

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l142_142247


namespace minimum_value_expression_l142_142956

-- Define the conditions for positive real numbers
variables (a b c : ℝ)
variable (h_a : 0 < a)
variable (h_b : 0 < b)
variable (h_c : 0 < c)

-- State the theorem to prove the minimum value of the expression
theorem minimum_value_expression (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : 
  (a / b) + (b / c) + (c / a) ≥ 3 := 
sorry

end minimum_value_expression_l142_142956


namespace sum_reciprocals_factors_12_l142_142402

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142402


namespace rhombus_area_correct_l142_142457

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

theorem rhombus_area_correct :
  rhombus_area 30 12 = 180 :=
by
  sorry

end rhombus_area_correct_l142_142457


namespace Bruce_remaining_amount_l142_142480

/--
Given:
1. initial_amount: the initial amount of money that Bruce's aunt gave him, which is 71 dollars.
2. shirt_cost: the cost of one shirt, which is 5 dollars.
3. num_shirts: the number of shirts Bruce bought, which is 5.
4. pants_cost: the cost of one pair of pants, which is 26 dollars.
Show:
Bruce's remaining amount of money after buying the shirts and the pants is 20 dollars.
-/
theorem Bruce_remaining_amount
  (initial_amount : ℕ)
  (shirt_cost : ℕ)
  (num_shirts : ℕ)
  (pants_cost : ℕ)
  (total_amount_spent : ℕ)
  (remaining_amount : ℕ) :
  initial_amount = 71 →
  shirt_cost = 5 →
  num_shirts = 5 →
  pants_cost = 26 →
  total_amount_spent = shirt_cost * num_shirts + pants_cost →
  remaining_amount = initial_amount - total_amount_spent →
  remaining_amount = 20 :=
by
  intro h_initial h_shirt_cost h_num_shirts h_pants_cost h_total_spent h_remaining
  rw [h_initial, h_shirt_cost, h_num_shirts, h_pants_cost, h_total_spent, h_remaining]
  rfl

end Bruce_remaining_amount_l142_142480


namespace sum_of_reciprocals_of_factors_of_12_l142_142434

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142434


namespace sum_reciprocals_factors_12_l142_142345

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l142_142345


namespace D_180_equals_43_l142_142179

-- Define D(n) as the number of ways to express the positive integer n
-- as a product of integers strictly greater than 1, where the order of factors matters.
def D (n : Nat) : Nat := sorry  -- The actual implementation is not provided, as per instructions.

theorem D_180_equals_43 : D 180 = 43 :=
by
  sorry  -- The proof is omitted as the task specifies.

end D_180_equals_43_l142_142179


namespace sum_reciprocals_factors_of_12_l142_142278

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l142_142278


namespace sum_reciprocals_of_factors_of_12_l142_142330

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l142_142330


namespace max_ab_l142_142884

theorem max_ab (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ∃ (M : ℝ), M = 1 / 8 ∧ ∀ (a b : ℝ), (a + 2 * b = 1) → 0 < a → 0 < b → ab ≤ M :=
sorry

end max_ab_l142_142884


namespace smallest_positive_integer_x_l142_142768

theorem smallest_positive_integer_x (x : ℕ) (h : 725 * x ≡ 1165 * x [MOD 35]) : x = 7 :=
sorry

end smallest_positive_integer_x_l142_142768


namespace smallest_integer_with_odd_and_even_divisors_l142_142073

theorem smallest_integer_with_odd_and_even_divisors :
  ∃ n : ℕ,
    0 < n ∧
    (∀ d : ℕ, d ∣ n → (odd d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 1}) ∨ (even d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 0})) ∧ 
    (↑8 = (∑ d in (finset.filter (λ d, odd d) (finset.divisors n)), 1)) ∧
    (↑16 = (∑ d in (finset.filter (λ d, even d) (finset.divisors n)), 1)) ∧
    (24 = (∑ d in (finset.divisors n), 1)) ∧
    n = 108 :=
begin
  sorry
end

end smallest_integer_with_odd_and_even_divisors_l142_142073


namespace min_value_sin6_cos6_exists_min_value_sin6_cos6_l142_142834

theorem min_value_sin6_cos6 (x : ℝ) : (sin x)^2 + (cos x)^2 = 1 → (sin x)^6 + (cos x)^6 ≥ 1/2 :=
sorry

theorem exists_min_value_sin6_cos6 : ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ (sin x)^6 + (cos x)^6 = 1/2 :=
sorry

end min_value_sin6_cos6_exists_min_value_sin6_cos6_l142_142834


namespace imaginary_part_of_z_l142_142518

-- Step 1: Define the imaginary unit.
def i : ℂ := Complex.I  -- ℂ represents complex numbers in Lean and Complex.I is the imaginary unit.

-- Step 2: Define the complex number z.
noncomputable def z : ℂ := (4 - 3 * i) / i

-- Step 3: State the theorem.
theorem imaginary_part_of_z : Complex.im z = -4 :=
by 
  sorry

end imaginary_part_of_z_l142_142518


namespace sequence_terms_l142_142150

/-- Given the sequence {a_n} with the sum of the first n terms S_n = n^2 - 3, 
    prove that a_1 = -2 and a_n = 2n - 1 for n ≥ 2. --/
theorem sequence_terms (a : ℕ → ℤ) (S : ℕ → ℤ)
  (hS : ∀ n : ℕ, S n = n^2 - 3)
  (h1 : ∀ n : ℕ, a n = S n - S (n - 1)) :
  a 1 = -2 ∧ (∀ n : ℕ, n ≥ 2 → a n = 2 * n - 1) :=
by {
  sorry
}

end sequence_terms_l142_142150


namespace find_min_k_l142_142903

theorem find_min_k (k : ℕ) 
  (h1 : k > 0) 
  (h2 : ∀ (A : Finset ℕ), A ⊆ (Finset.range 26).erase 0 → A.card = k → ∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (2 / 3 : ℝ) ≤ x / y ∧ x / y ≤ (3 / 2 : ℝ)) : 
  k = 7 :=
by {
  sorry
}

end find_min_k_l142_142903


namespace minimize_sin_cos_six_l142_142826

theorem minimize_sin_cos_six (x : ℝ) : sin x ^ 6 + cos x ^ 6 ≥ 1 / 4 := 
  sorry

end minimize_sin_cos_six_l142_142826


namespace sum_of_reciprocals_of_factors_of_12_l142_142312

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l142_142312


namespace arithmetic_sequence_and_minimum_sum_l142_142916

theorem arithmetic_sequence_and_minimum_sum 
    (S : ℕ → ℝ) 
    (a : ℕ → ℝ) 
    (h1 : ∀ n, (2 * S n / n) + n = 2 * a n + 1) 
    (geo_cond : (a 4) * (a 9) = (a 7)^2)
    : IsArithmeticSeq a ∧  S 12 = -78 ∨ S 13 = -78 := 
sorry

-- Define the helper structure for recognizing an arithmetic sequence
structure IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  (d : ℝ)
  (h2 : ∀ n, a (n+1) = a n + d)

end arithmetic_sequence_and_minimum_sum_l142_142916


namespace card_probability_ratio_l142_142133

theorem card_probability_ratio :
  let total_cards := 40
  let numbers := 10
  let cards_per_number := 4
  let choose (n k : ℕ) := Nat.choose n k
  let p := 10 / choose total_cards 4
  let q := 1440 / choose total_cards 4
  (q / p) = 144 :=
by
  sorry

end card_probability_ratio_l142_142133


namespace larger_integer_is_21_l142_142723

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l142_142723


namespace sum_reciprocals_factors_of_12_l142_142287

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l142_142287


namespace range_of_b_over_a_l142_142630

noncomputable def f (a b x : ℝ) : ℝ := (x - a)^3 * (x - b)
noncomputable def g_k (a b k x : ℝ) : ℝ := (f a b x - f a b k) / (x - k)

theorem range_of_b_over_a (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : b < 1)
    (hk_inc : ∀ k : ℤ, ∀ x : ℝ, k < x → g_k a b k x ≥ g_k a b k (k + 1)) :
  1 < b / a ∧ b / a ≤ 3 :=
by
  sorry


end range_of_b_over_a_l142_142630


namespace max_points_of_intersection_l142_142223

theorem max_points_of_intersection (circles : Fin 2 → Circle) (lines : Fin 3 → Line) :
  number_of_intersections circles lines = 17 :=
sorry

end max_points_of_intersection_l142_142223


namespace intersection_correct_l142_142632

open Set

noncomputable def A := {x : ℕ | x^2 - x - 2 ≤ 0}
noncomputable def B := {x : ℝ | -1 ≤ x ∧ x < 2}
noncomputable def A_cap_B := A ∩ {x : ℕ | (x : ℝ) ∈ B}

theorem intersection_correct : A_cap_B = {0, 1} :=
sorry

end intersection_correct_l142_142632


namespace expand_expression_l142_142822

theorem expand_expression (x y : ℝ) : (3 * x + 15) * (4 * y + 12) = 12 * x * y + 36 * x + 60 * y + 180 := 
  sorry

end expand_expression_l142_142822


namespace smallest_integer_with_divisors_l142_142082

theorem smallest_integer_with_divisors :
  ∃ (n : ℕ), 
    (∀ d : ℕ, d ∣ n → d % 2 = 1 → (∃! k : ℕ, d = (3 ^ k) * 5 ^ (7 - k))) ∧ 
    (∀ d : ℕ, d ∣ n → d % 2 = 0 → (∃! k : ℕ, d = 2 ^ k * m)) ∧ 
    (n = 1080) :=
sorry

end smallest_integer_with_divisors_l142_142082


namespace arithmetic_sequence_minimum_value_of_Sn_l142_142924

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Given the initial condition
axiom given_condition : ∀ n : ℕ, (2 * S n) / n + n = 2 * a n + 1

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + 1 := 
sorry

-- Part 2: Find the minimum value of S_n
axiom geometric_sequence_condition : (a 7)^2 = a 4 * a 9

theorem minimum_value_of_Sn : S 12 = -78 ∨ S 13 = -78 :=
sorry

end arithmetic_sequence_minimum_value_of_Sn_l142_142924


namespace problem_1_problem_2_l142_142861

noncomputable def a : ℝ := Real.sqrt 7 + 2
noncomputable def b : ℝ := Real.sqrt 7 - 2

theorem problem_1 : a^2 * b + b^2 * a = 6 * Real.sqrt 7 := by
  sorry

theorem problem_2 : a^2 + a * b + b^2 = 25 := by
  sorry

end problem_1_problem_2_l142_142861


namespace smallest_n_with_divisors_l142_142078

-- Definitions of the divisors
def d_total (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)
def d_even (a b c : ℕ) : ℕ := a * (b + 1) * (c + 1)
def d_odd (b c : ℕ) : ℕ := (b + 1) * (c + 1)

-- Math problem and proving smallest n
theorem smallest_n_with_divisors (a b c : ℕ) (n : ℕ) (h_1 : d_odd b c = 8) (h_2 : d_even a b c = 16) : n = 60 :=
  sorry

end smallest_n_with_divisors_l142_142078


namespace sum_reciprocals_of_factors_of_12_l142_142323

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l142_142323


namespace Bruce_remaining_amount_l142_142479

/--
Given:
1. initial_amount: the initial amount of money that Bruce's aunt gave him, which is 71 dollars.
2. shirt_cost: the cost of one shirt, which is 5 dollars.
3. num_shirts: the number of shirts Bruce bought, which is 5.
4. pants_cost: the cost of one pair of pants, which is 26 dollars.
Show:
Bruce's remaining amount of money after buying the shirts and the pants is 20 dollars.
-/
theorem Bruce_remaining_amount
  (initial_amount : ℕ)
  (shirt_cost : ℕ)
  (num_shirts : ℕ)
  (pants_cost : ℕ)
  (total_amount_spent : ℕ)
  (remaining_amount : ℕ) :
  initial_amount = 71 →
  shirt_cost = 5 →
  num_shirts = 5 →
  pants_cost = 26 →
  total_amount_spent = shirt_cost * num_shirts + pants_cost →
  remaining_amount = initial_amount - total_amount_spent →
  remaining_amount = 20 :=
by
  intro h_initial h_shirt_cost h_num_shirts h_pants_cost h_total_spent h_remaining
  rw [h_initial, h_shirt_cost, h_num_shirts, h_pants_cost, h_total_spent, h_remaining]
  rfl

end Bruce_remaining_amount_l142_142479


namespace find_triples_l142_142494

-- Define the conditions in Lean 4
def is_prime (p : ℕ) : Prop := Nat.Prime p

def is_positive_integer (n : ℕ) : Prop := n > 0

-- Define the math proof problem
theorem find_triples (m n p : ℕ) (hp : is_prime p) (hm : is_positive_integer m) (hn : is_positive_integer n) : 
  p^n + 3600 = m^2 ↔ (m = 61 ∧ n = 2 ∧ p = 11) ∨ (m = 65 ∧ n = 4 ∧ p = 5) ∨ (m = 68 ∧ n = 10 ∧ p = 2) :=
by
  sorry

end find_triples_l142_142494


namespace least_positive_integer_l142_142231

theorem least_positive_integer :
  ∃ (a : ℕ), (a ≡ 1 [MOD 3]) ∧ (a ≡ 2 [MOD 4]) ∧ (∀ b, (b ≡ 1 [MOD 3]) → (b ≡ 2 [MOD 4]) → b ≥ a → b = a) :=
sorry

end least_positive_integer_l142_142231


namespace maria_earnings_l142_142960

def cost_of_brushes : ℕ := 20
def cost_of_canvas : ℕ := 3 * cost_of_brushes
def cost_per_liter_of_paint : ℕ := 8
def liters_of_paint : ℕ := 5
def cost_of_paint : ℕ := liters_of_paint * cost_per_liter_of_paint
def total_cost : ℕ := cost_of_brushes + cost_of_canvas + cost_of_paint
def selling_price : ℕ := 200

theorem maria_earnings : (selling_price - total_cost) = 80 := by
  sorry

end maria_earnings_l142_142960


namespace sum_reciprocals_factors_12_l142_142268

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l142_142268


namespace find_larger_integer_l142_142746

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l142_142746


namespace common_ratio_is_two_l142_142521

-- Define the geometric sequence
def geom_seq (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ := a_1 * q^n

-- Define the conditions
variables (a_1 q : ℝ)
variables (h_inc : 1 < q) (h_pos : 0 < a_1)
variables (h_seq : ∀ n : ℕ, 2 * (geom_seq a_1 q n + geom_seq a_1 q (n+2)) = 5 * geom_seq a_1 q (n+1))

-- Statement to prove
theorem common_ratio_is_two : q = 2 :=
by
  sorry

end common_ratio_is_two_l142_142521


namespace digit_makes_5678d_multiple_of_9_l142_142625

def is_multiple_of_9 (n : Nat) : Prop :=
  n % 9 = 0

theorem digit_makes_5678d_multiple_of_9 (d : Nat) (h : d ≥ 0 ∧ d < 10) :
  is_multiple_of_9 (5 * 10000 + 6 * 1000 + 7 * 100 + 8 * 10 + d) ↔ d = 1 := 
by
  sorry

end digit_makes_5678d_multiple_of_9_l142_142625


namespace sum_reciprocals_factors_12_l142_142401

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142401


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l142_142942

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l142_142942


namespace larger_integer_is_21_l142_142756

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l142_142756


namespace problem_statement_l142_142024

variables {c c' d d' : ℝ}

theorem problem_statement (hc : c ≠ 0) (hc' : c' ≠ 0)
  (h : (-d) / (2 * c) = 2 * ((-d') / (3 * c'))) :
  (d / (2 * c)) = 2 * (d' / (3 * c')) :=
by
  sorry

end problem_statement_l142_142024


namespace find_n_l142_142864

theorem find_n (n : ℤ) (h : (1 : ℤ)^2 + 3 * 1 + n = 0) : n = -4 :=
sorry

end find_n_l142_142864


namespace fraction_unchanged_l142_142656

theorem fraction_unchanged (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (2 * x) / (2 * (x + y)) = x / (x + y) :=
by
  sorry

end fraction_unchanged_l142_142656


namespace rotated_ellipse_sum_is_four_l142_142704

noncomputable def rotated_ellipse_center (h' k' : ℝ) : Prop :=
h' = 3 ∧ k' = -5

noncomputable def rotated_ellipse_axes (a' b' : ℝ) : Prop :=
a' = 4 ∧ b' = 2

noncomputable def rotated_ellipse_sum (h' k' a' b' : ℝ) : ℝ :=
h' + k' + a' + b'

theorem rotated_ellipse_sum_is_four (h' k' a' b' : ℝ) 
  (hc : rotated_ellipse_center h' k') (ha : rotated_ellipse_axes a' b') :
  rotated_ellipse_sum h' k' a' b' = 4 :=
by
  -- The proof would be provided here.
  -- Since we're asked not to provide the proof but just to ensure the statement is correct, we use sorry.
  sorry

end rotated_ellipse_sum_is_four_l142_142704


namespace sum_reciprocal_factors_of_12_l142_142417

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l142_142417


namespace isosceles_triangle_perimeter_l142_142803

theorem isosceles_triangle_perimeter {a : ℝ} (h_base : 4 ≠ 0) (h_roots : a^2 - 5 * a + 6 = 0) :
  a = 3 → (4 + 2 * a = 10) :=
by
  sorry

end isosceles_triangle_perimeter_l142_142803


namespace cone_central_angle_l142_142042

theorem cone_central_angle (l : ℝ) (α : ℝ) (h : (30 : ℝ) * π / 180 > 0) :
  α = π := 
sorry

end cone_central_angle_l142_142042


namespace find_a_b_c_l142_142618

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_a_b_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hp1 : is_prime (a + b * c))
  (hp2 : is_prime (b + a * c))
  (hp3 : is_prime (c + a * b))
  (hdiv1 : (a + b * c) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)))
  (hdiv2 : (b + a * c) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)))
  (hdiv3 : (c + a * b) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1))) :
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end find_a_b_c_l142_142618


namespace find_larger_integer_l142_142739

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l142_142739


namespace sum_of_reciprocals_factors_12_l142_142428

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l142_142428


namespace simplify_fraction_l142_142820

theorem simplify_fraction : (8 / (5 * 42) = 4 / 105) :=
by
    sorry

end simplify_fraction_l142_142820


namespace sum_reciprocals_factors_12_l142_142342

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142342


namespace max_k_l142_142670

def seq (a : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = k * (a n) ^ 2 + 1

def bounded (a : ℕ → ℝ) (c : ℝ) : Prop :=
∀ n : ℕ, a n < c

theorem max_k (k : ℝ) (c : ℝ) (a : ℕ → ℝ) :
  a 1 = 1 →
  seq a k →
  bounded a c →
  0 < k ∧ k ≤ 1 / 4 :=
by
  sorry

end max_k_l142_142670


namespace sum_reciprocals_of_factors_of_12_l142_142329

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l142_142329


namespace larger_integer_value_l142_142719

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l142_142719


namespace sum_reciprocals_factors_12_l142_142340

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l142_142340


namespace isosceles_triangle_length_l142_142171

theorem isosceles_triangle_length (BC : ℕ) (area : ℕ) (h : ℕ)
  (isosceles : AB = AC)
  (BC_val : BC = 16)
  (area_val : area = 120)
  (height_val : h = (2 * area) / BC)
  (AB_square : ∀ BD AD : ℕ, BD = BC / 2 → AD = h → AB^2 = AD^2 + BD^2)
  : AB = 17 :=
by
  sorry

end isosceles_triangle_length_l142_142171


namespace total_boys_l142_142043

theorem total_boys (T F : ℕ) 
  (avg_all : 37 * T = 39 * 110 + 15 * F) 
  (total_eq : T = 110 + F) : 
  T = 120 := 
sorry

end total_boys_l142_142043


namespace digit_makes_5678d_multiple_of_9_l142_142626

def is_multiple_of_9 (n : Nat) : Prop :=
  n % 9 = 0

theorem digit_makes_5678d_multiple_of_9 (d : Nat) (h : d ≥ 0 ∧ d < 10) :
  is_multiple_of_9 (5 * 10000 + 6 * 1000 + 7 * 100 + 8 * 10 + d) ↔ d = 1 := 
by
  sorry

end digit_makes_5678d_multiple_of_9_l142_142626


namespace candy_bar_calories_l142_142482

theorem candy_bar_calories
  (miles_walked : ℕ)
  (calories_per_mile : ℕ)
  (net_calorie_deficit : ℕ)
  (total_calories_burned : ℕ)
  (candy_bar_calories : ℕ)
  (h1 : miles_walked = 3)
  (h2 : calories_per_mile = 150)
  (h3 : net_calorie_deficit = 250)
  (h4 : total_calories_burned = miles_walked * calories_per_mile)
  (h5 : candy_bar_calories = total_calories_burned - net_calorie_deficit) :
  candy_bar_calories = 200 := 
by
  sorry

end candy_bar_calories_l142_142482


namespace airplane_distance_difference_l142_142164

theorem airplane_distance_difference (a : ℕ) : 
  let against_wind_distance := (a - 20) * 3
  let with_wind_distance := (a + 20) * 4
  with_wind_distance - against_wind_distance = a + 140 :=
by
  sorry

end airplane_distance_difference_l142_142164


namespace sum_of_reciprocals_factors_12_l142_142304

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l142_142304


namespace simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l142_142814

variable (a b : ℤ)
def A : ℤ := b^2 - a^2 + 5 * a * b
def B : ℤ := 3 * a * b + 2 * b^2 - a^2

theorem simplify_2A_minus_B : 2 * A a b - B a b = -a^2 + 7 * a * b := by
  sorry

theorem evaluate_2A_minus_B_at_1_2 : 2 * A 1 2 - B 1 2 = 13 := by
  sorry

end simplify_2A_minus_B_evaluate_2A_minus_B_at_1_2_l142_142814


namespace incorrect_calculation_d_l142_142773

theorem incorrect_calculation_d : (1 / 3) / (-1) ≠ 3 * (-1) := 
by {
  -- we'll leave the body of the proof as sorry.
  sorry
}

end incorrect_calculation_d_l142_142773


namespace length_of_rooms_l142_142899

-- Definitions based on conditions
def width : ℕ := 18
def num_rooms : ℕ := 20
def total_area : ℕ := 6840

-- Theorem stating the length of the rooms
theorem length_of_rooms : (total_area / num_rooms) / width = 19 := by
  sorry

end length_of_rooms_l142_142899


namespace root_polynomial_satisfies_expression_l142_142550

noncomputable def roots_of_polynomial (x : ℕ) : Prop :=
  x^3 - 15 * x^2 + 25 * x - 10 = 0

theorem root_polynomial_satisfies_expression (p q r : ℕ) 
    (h1 : roots_of_polynomial p)
    (h2 : roots_of_polynomial q)
    (h3 : roots_of_polynomial r)
    (h_sum : p + q + r = 15)
    (h_prod : p*q + q*r + r*p = 25) :
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 :=
by sorry

end root_polynomial_satisfies_expression_l142_142550


namespace maria_total_distance_in_miles_l142_142557

theorem maria_total_distance_in_miles :
  ∀ (steps_per_mile : ℕ) (full_cycles : ℕ) (remaining_steps : ℕ),
    steps_per_mile = 1500 →
    full_cycles = 50 →
    remaining_steps = 25000 →
    (100000 * full_cycles + remaining_steps) / steps_per_mile = 3350 := by
  intros
  sorry

end maria_total_distance_in_miles_l142_142557


namespace factorization_example_l142_142587

theorem factorization_example :
  (4 : ℤ) * x^2 - 1 = (2 * x + 1) * (2 * x - 1) := 
by
  sorry

end factorization_example_l142_142587


namespace square_area_fraction_shaded_l142_142693

theorem square_area_fraction_shaded (s : ℝ) :
  let R := (s / 2, s)
  let S := (s, s / 2)
  -- Area of triangle RSV
  let area_RSV := (1 / 2) * (s / 2) * (s * Real.sqrt 2 / 4)
  -- Non-shaded area
  let non_shaded_area := area_RSV
  -- Total area of the square
  let total_area := s^2
  -- Shaded area
  let shaded_area := total_area - non_shaded_area
  -- Fraction shaded
  (shaded_area / total_area) = 1 - Real.sqrt 2 / 16 :=
by
  sorry

end square_area_fraction_shaded_l142_142693


namespace solve_first_equation_solve_second_equation_l142_142559

theorem solve_first_equation (x : ℤ) : 4 * x + 3 = 5 * x - 1 → x = 4 :=
by
  intros h
  sorry

theorem solve_second_equation (x : ℤ) : 4 * (x - 1) = 1 - x → x = 1 :=
by
  intros h
  sorry

end solve_first_equation_solve_second_equation_l142_142559


namespace larger_integer_21_l142_142711

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l142_142711


namespace find_side_length_of_cut_out_square_l142_142110

noncomputable def cardboard_box (x : ℝ) : Prop :=
  let length_initial := 80
  let width_initial := 60
  let area_base := 1500
  let length_final := length_initial - 2 * x
  let width_final := width_initial - 2 * x
  length_final * width_final = area_base

theorem find_side_length_of_cut_out_square : ∃ x : ℝ, cardboard_box x ∧ 0 ≤ x ∧ (80 - 2 * x) > 0 ∧ (60 - 2 * x) > 0 ∧ x = 15 :=
by
  sorry

end find_side_length_of_cut_out_square_l142_142110


namespace naomi_saw_wheels_l142_142611

theorem naomi_saw_wheels :
  let regular_bikes := 7
  let children's_bikes := 11
  let wheels_per_regular_bike := 2
  let wheels_per_children_bike := 4
  let total_wheels := regular_bikes * wheels_per_regular_bike + children's_bikes * wheels_per_children_bike
  total_wheels = 58 := by
  sorry

end naomi_saw_wheels_l142_142611


namespace find_total_photos_l142_142578

noncomputable def total_photos (T : ℕ) (Paul Tim Tom : ℕ) : Prop :=
  Tim = T - 100 ∧ Paul = Tim + 10 ∧ Tom = 38 ∧ Tom + Tim + Paul = T

theorem find_total_photos : ∃ T, total_photos T (T - 90) (T - 100) 38 :=
sorry

end find_total_photos_l142_142578


namespace prove_arithmetic_sequence_minimum_value_S_l142_142931

-- Given sequence and sum conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

theorem prove_arithmetic_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, (2 * S a n / n) + n = 2 * a n + 1) : 
  is_arithmetic_seq a :=
sorry

theorem minimum_value_S (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a)
  (h_geo : (a 7) ^ 2 = a 4 * a 9) : ∃ n : ℕ, n ≥ 12 ∧ S a n = -78 :=
sorry

end prove_arithmetic_sequence_minimum_value_S_l142_142931


namespace arithmetic_sequence_minimum_value_S_n_l142_142939

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l142_142939


namespace option_C_correct_l142_142876

variable {a b c d : ℝ}

theorem option_C_correct (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by sorry

end option_C_correct_l142_142876


namespace circle_cartesian_line_circle_intersect_l142_142870

noncomputable def L_parametric (t : ℝ) : ℝ × ℝ :=
  (t, 1 + 2 * t)

noncomputable def C_polar (θ : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

def L_cartesian (x y : ℝ) : Prop :=
  y = 2 * x + 1

def C_cartesian (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem circle_cartesian :
  ∀ x y : ℝ, C_polar x = y ↔ C_cartesian x y :=
sorry

theorem line_circle_intersect (x y : ℝ) :
  L_cartesian x y → C_cartesian x y → True :=
sorry

end circle_cartesian_line_circle_intersect_l142_142870


namespace compare_a_b_l142_142652

theorem compare_a_b (a b : ℝ) (h1 : a = 2 * Real.sqrt 7) (h2 : b = 3 * Real.sqrt 5) : a < b :=
by {
  sorry -- We'll leave the proof as a placeholder.
}

end compare_a_b_l142_142652


namespace flight_duration_l142_142176

theorem flight_duration (takeoff landing : Nat)
  (h m : Nat) (h_pos : 0 < m) (m_lt_60 : m < 60)
  (time_takeoff : takeoff = 9 * 60 + 27)
  (time_landing : landing = 11 * 60 + 56)
  (flight_duration : (landing - takeoff) = h * 60 + m) :
  h + m = 31 :=
sorry

end flight_duration_l142_142176


namespace surface_area_hemisphere_radius_1_l142_142759

noncomputable def surface_area_hemisphere (r : ℝ) : ℝ :=
  2 * Real.pi * r^2 + Real.pi * r^2

theorem surface_area_hemisphere_radius_1 :
  surface_area_hemisphere 1 = 3 * Real.pi :=
by
  sorry

end surface_area_hemisphere_radius_1_l142_142759


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l142_142940

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l142_142940


namespace smallest_integer_has_8_odd_and_16_even_divisors_l142_142080

/-!
  Prove that the smallest positive integer with exactly 8 positive odd integer divisors
  and exactly 16 positive even integer divisors is 540.
-/
def smallest_integer_with_divisors : ℕ :=
  540

theorem smallest_integer_has_8_odd_and_16_even_divisors 
  (n : ℕ) 
  (h1 : (8 : ℕ) = nat.count (λ d, d % 2 = 1) (nat.divisors n))
  (h2 : (16 : ℕ) = nat.count (λ d, d % 2 = 0) (nat.divisors n)) :
  n = 540 :=
sorry

end smallest_integer_has_8_odd_and_16_even_divisors_l142_142080


namespace find_may_monday_l142_142885

noncomputable def weekday (day_of_month : ℕ) (first_day_weekday : ℕ) : ℕ :=
(day_of_month + first_day_weekday - 1) % 7

theorem find_may_monday (r n : ℕ) (condition1 : weekday r 5 = 5) (condition2 : weekday n 5 = 1) (condition3 : 15 < n ∧ n < 25) : 
  n = 20 :=
by
  -- Proof omitted.
  sorry

end find_may_monday_l142_142885


namespace proof_stage_constancy_l142_142893

-- Definitions of stages
def Stage1 := "Fertilization and seed germination"
def Stage2 := "Flowering and pollination"
def Stage3 := "Meiosis and fertilization"
def Stage4 := "Formation of sperm and egg cells"

-- Question: Which stages maintain chromosome constancy and promote genetic recombination in plant life?
def Q := "Which stages maintain chromosome constancy and promote genetic recombination in plant life?"

-- Correct answer
def Answer := Stage3

-- Conditions
def s1 := Stage1
def s2 := Stage2
def s3 := Stage3
def s4 := Stage4

-- Theorem statement
theorem proof_stage_constancy : Q = Answer := by
  sorry

end proof_stage_constancy_l142_142893


namespace paco_initial_salty_cookies_l142_142968

variable (S : ℕ)
variable (sweet_cookies : ℕ := 40)
variable (salty_cookies_eaten1 : ℕ := 28)
variable (sweet_cookies_eaten : ℕ := 15)
variable (extra_salty_cookies_eaten : ℕ := 13)

theorem paco_initial_salty_cookies 
  (h1 : salty_cookies_eaten1 = 28)
  (h2 : sweet_cookies_eaten = 15)
  (h3 : extra_salty_cookies_eaten = 13)
  (h4 : sweet_cookies = 40)
  : (S = (salty_cookies_eaten1 + (extra_salty_cookies_eaten + sweet_cookies_eaten))) :=
by
  -- starting with the equation S = number of salty cookies Paco
  -- initially had, which should be equal to the total salty 
  -- cookies he ate.
  sorry

end paco_initial_salty_cookies_l142_142968


namespace solve_equation_l142_142126

noncomputable def fourthRoot (x : ℝ) := Real.sqrt (Real.sqrt x)

theorem solve_equation (x : ℝ) (hx : x ≥ 0) :
  fourthRoot x = 18 / (9 - fourthRoot x) ↔ x = 81 ∨ x = 1296 :=
by
  sorry

end solve_equation_l142_142126


namespace tim_movie_marathon_duration_is_9_l142_142997

-- Define the conditions:
def first_movie_duration : ℕ := 2
def second_movie_duration : ℕ := first_movie_duration + (first_movie_duration / 2)
def combined_duration_first_two_movies : ℕ := first_movie_duration + second_movie_duration
def third_movie_duration : ℕ := combined_duration_first_two_movies - 1
def total_marathon_duration : ℕ := first_movie_duration + second_movie_duration + third_movie_duration

-- The theorem to prove the marathon duration is 9 hours
theorem tim_movie_marathon_duration_is_9 :
  total_marathon_duration = 9 :=
by sorry

end tim_movie_marathon_duration_is_9_l142_142997


namespace range_of_x_sqrt_4_2x_l142_142210

theorem range_of_x_sqrt_4_2x (x : ℝ) : (4 - 2 * x ≥ 0) ↔ (x ≤ 2) :=
by
  sorry

end range_of_x_sqrt_4_2x_l142_142210


namespace bus_routes_arrangement_l142_142172

-- Define the lines and intersection points (stops).
def routes := Fin 10
def stops (r1 r2 : routes) : Prop := r1 ≠ r2 -- Representing intersection

-- First condition: Any subset of 9 routes will cover all stops.
def covers_all_stops (routes_subset : Finset routes) : Prop :=
  routes_subset.card = 9 → ∀ r1 r2 : routes, r1 ≠ r2 → stops r1 r2

-- Second condition: Any subset of 8 routes will miss at least one stop.
def misses_at_least_one_stop (routes_subset : Finset routes) : Prop :=
  routes_subset.card = 8 → ∃ r1 r2 : routes, r1 ≠ r2 ∧ ¬stops r1 r2

-- The theorem to prove that this arrangement is possible.
theorem bus_routes_arrangement : 
  (∃ stops_scheme : routes → routes → Prop, 
    (∀ subset_9 : Finset routes, covers_all_stops subset_9) ∧ 
    (∀ subset_8 : Finset routes, misses_at_least_one_stop subset_8)) :=
by
  sorry

end bus_routes_arrangement_l142_142172


namespace sum_reciprocals_of_factors_12_l142_142368

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l142_142368


namespace min_value_sin6_cos6_l142_142831

theorem min_value_sin6_cos6 : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 → (sin x ^ 6 + cos x ^ 6) ≥ (1 / 4) :=
by {
  sorry
}

end min_value_sin6_cos6_l142_142831


namespace problem_c_l142_142452

theorem problem_c (x y : ℝ) (h : x - 3 = y - 3): x - y = 0 :=
by
  sorry

end problem_c_l142_142452


namespace rectangle_ratio_of_semicircles_l142_142608

theorem rectangle_ratio_of_semicircles (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h : a * b = π * b^2) : a / b = π := by
  sorry

end rectangle_ratio_of_semicircles_l142_142608


namespace singing_only_pupils_l142_142013

theorem singing_only_pupils (total_pupils debate_only both : ℕ) (h1 : total_pupils = 55) (h2 : debate_only = 10) (h3 : both = 17) :
  total_pupils - debate_only = 45 :=
by
  -- skipping proof
  sorry

end singing_only_pupils_l142_142013


namespace sum_reciprocals_factors_12_l142_142271

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l142_142271


namespace biased_coin_problem_l142_142449

theorem biased_coin_problem 
  (h : ℝ)
  (H : 7.choose 2 * h^2 * (1-h)^5 = 7.choose 3 * h^3 * (1-h)^4)
  (h_ne_zero : h ≠ 0)
  (h_ne_one : h ≠ 1)
  : ∃ p q : ℕ, nat.gcd p q = 1 ∧ (35 * h^4 * (1-h)^3).num = p ∧ (35 * h^4 * (1-h)^3).denom = q ∧ p + q = 187 := 
by
  sorry

end biased_coin_problem_l142_142449


namespace remainder_x_squared_div_25_l142_142881

theorem remainder_x_squared_div_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 0 [ZMOD 25] :=
sorry

end remainder_x_squared_div_25_l142_142881


namespace bruce_money_left_l142_142477

-- Definitions for the given values
def initial_amount : ℕ := 71
def shirt_cost : ℕ := 5
def number_of_shirts : ℕ := 5
def pants_cost : ℕ := 26

-- The theorem that Bruce has $20 left
theorem bruce_money_left : initial_amount - (shirt_cost * number_of_shirts + pants_cost) = 20 :=
by
  sorry

end bruce_money_left_l142_142477


namespace constant_term_expansion_l142_142170

theorem constant_term_expansion : 
  (∃ r : ℕ, (Binomial (8:ℕ) r) * ((1 / 2)^(8 - r)) * ((-1)^r) * ((2^(r - 8)) * 1) = 28) :=
sorry

end constant_term_expansion_l142_142170


namespace sum_of_reciprocals_factors_12_l142_142394

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l142_142394


namespace total_number_of_shirts_l142_142699

variable (total_cost : ℕ) (num_15_dollar_shirts : ℕ) (cost_15_dollar_shirts : ℕ) 
          (cost_remaining_shirts : ℕ) (num_remaining_shirts : ℕ) 

theorem total_number_of_shirts :
  total_cost = 85 →
  num_15_dollar_shirts = 3 →
  cost_15_dollar_shirts = 15 →
  cost_remaining_shirts = 20 →
  (num_remaining_shirts * cost_remaining_shirts) + (num_15_dollar_shirts * cost_15_dollar_shirts) = total_cost →
  num_15_dollar_shirts + num_remaining_shirts = 5 :=
by
  intros
  sorry

end total_number_of_shirts_l142_142699


namespace sum_reciprocals_of_factors_12_l142_142374

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l142_142374

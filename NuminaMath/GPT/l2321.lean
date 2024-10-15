import Mathlib

namespace NUMINAMATH_GPT_total_gold_coins_l2321_232169

theorem total_gold_coins (n c : ℕ) 
  (h1 : n = 11 * (c - 3))
  (h2 : n = 7 * c + 5) : 
  n = 75 := 
by 
  sorry

end NUMINAMATH_GPT_total_gold_coins_l2321_232169


namespace NUMINAMATH_GPT_point_K_is_intersection_of_diagonals_l2321_232170

variable {K A B C D : Type}

/-- A quadrilateral is circumscribed if there exists a circle within which all four vertices lie. -/
noncomputable def is_circumscribed (A B C D : Type) : Prop :=
sorry

/-- Distances from point K to the sides of the quadrilateral ABCD are proportional to the lengths of those sides. -/
noncomputable def proportional_distances (K A B C D : Type) : Prop :=
sorry

/-- A point is the intersection point of the diagonals AC and BD of quadrilateral ABCD. -/
noncomputable def intersection_point_of_diagonals (K A C B D : Type) : Prop :=
sorry

theorem point_K_is_intersection_of_diagonals 
  (K A B C D : Type) 
  (circumQ : is_circumscribed A B C D) 
  (propDist : proportional_distances K A B C D) 
  : intersection_point_of_diagonals K A C B D :=
sorry

end NUMINAMATH_GPT_point_K_is_intersection_of_diagonals_l2321_232170


namespace NUMINAMATH_GPT_area_of_trapezium_l2321_232152

variables (x : ℝ) (h : x > 0)

def shorter_base := 2 * x
def altitude := 2 * x
def longer_base := 6 * x

theorem area_of_trapezium (hx : x > 0) :
  (1 / 2) * (shorter_base x + longer_base x) * altitude x = 8 * x^2 := 
sorry

end NUMINAMATH_GPT_area_of_trapezium_l2321_232152


namespace NUMINAMATH_GPT_base7_divisible_by_5_l2321_232189

theorem base7_divisible_by_5 :
  ∃ (d : ℕ), (0 ≤ d ∧ d < 7) ∧ (344 * d + 56) % 5 = 0 ↔ d = 1 :=
by
  sorry

end NUMINAMATH_GPT_base7_divisible_by_5_l2321_232189


namespace NUMINAMATH_GPT_sequence_value_2009_l2321_232111

theorem sequence_value_2009 
  (a : ℕ → ℝ)
  (h_recur : ∀ n ≥ 2, a n = a (n - 1) * a (n + 1))
  (h_a1 : a 1 = 1 + Real.sqrt 3)
  (h_a1776 : a 1776 = 4 + Real.sqrt 3) :
  a 2009 = (3 / 2) + (3 * Real.sqrt 3 / 2) := 
sorry

end NUMINAMATH_GPT_sequence_value_2009_l2321_232111


namespace NUMINAMATH_GPT_digit_multiplication_sum_l2321_232175

-- Define the main problem statement in Lean 4
theorem digit_multiplication_sum (A B E F : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) 
                                            (h2 : 0 ≤ B ∧ B ≤ 9) 
                                            (h3 : 0 ≤ E ∧ E ≤ 9)
                                            (h4 : 0 ≤ F ∧ F ≤ 9)
                                            (h5 : A ≠ B) 
                                            (h6 : A ≠ E) 
                                            (h7 : A ≠ F)
                                            (h8 : B ≠ E)
                                            (h9 : B ≠ F)
                                            (h10 : E ≠ F)
                                            (h11 : (100 * A + 10 * B + E) * F = 1001 * E + 100 * A)
                                            : A + B = 5 :=
sorry

end NUMINAMATH_GPT_digit_multiplication_sum_l2321_232175


namespace NUMINAMATH_GPT_intersection_M_N_l2321_232161

def M (x : ℝ) : Prop := (x - 3) / (x + 1) > 0
def N (x : ℝ) : Prop := 3 * x + 2 > 0

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 3 < x} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2321_232161


namespace NUMINAMATH_GPT_solution_set_of_system_of_inequalities_l2321_232136

theorem solution_set_of_system_of_inequalities :
  {x : ℝ | |x| - 1 < 0 ∧ x^2 - 3 * x < 0} = {x : ℝ | 0 < x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_solution_set_of_system_of_inequalities_l2321_232136


namespace NUMINAMATH_GPT_solution_set_of_f_inequality_l2321_232151

variable {f : ℝ → ℝ}
variable (h1 : f 1 = 1)
variable (h2 : ∀ x, f' x < 1/2)

theorem solution_set_of_f_inequality :
  {x : ℝ | f (x^2) < x^2 / 2 + 1 / 2} = {x : ℝ | x < -1 ∨ 1 < x} :=
sorry

end NUMINAMATH_GPT_solution_set_of_f_inequality_l2321_232151


namespace NUMINAMATH_GPT_abc_inequality_l2321_232188

theorem abc_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ ab + ac + bc :=
by
  sorry

end NUMINAMATH_GPT_abc_inequality_l2321_232188


namespace NUMINAMATH_GPT_calculate_expression_l2321_232174

open Complex

def B : Complex := 5 - 2 * I
def N : Complex := -3 + 2 * I
def T : Complex := 2 * I
def Q : ℂ := 3

theorem calculate_expression : B - N + T - 2 * Q = 2 - 2 * I := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2321_232174


namespace NUMINAMATH_GPT_circle1_standard_form_circle2_standard_form_l2321_232128

-- Define the first circle equation and its corresponding answer in standard form
theorem circle1_standard_form :
  ∀ x y : ℝ, (x^2 + y^2 + 2*x + 4*y - 4 = 0) ↔ ((x + 1)^2 + (y + 2)^2 = 9) :=
by
  intro x y
  sorry

-- Define the second circle equation and its corresponding answer in standard form
theorem circle2_standard_form :
  ∀ x y : ℝ, (3*x^2 + 3*y^2 + 6*x + 3*y - 15 = 0) ↔ ((x + 1)^2 + (y + 1/2)^2 = 25/4) :=
by
  intro x y
  sorry

end NUMINAMATH_GPT_circle1_standard_form_circle2_standard_form_l2321_232128


namespace NUMINAMATH_GPT_product_of_fractions_l2321_232106

-- Define the fractions as ratios.
def fraction1 : ℚ := 2 / 5
def fraction2 : ℚ := 7 / 10

-- State the theorem that proves the product of the fractions is equal to the simplified result.
theorem product_of_fractions : fraction1 * fraction2 = 7 / 25 :=
by
  -- Skip the proof.
  sorry

end NUMINAMATH_GPT_product_of_fractions_l2321_232106


namespace NUMINAMATH_GPT_car_travel_distance_l2321_232125

theorem car_travel_distance :
  ∃ S : ℝ, 
    (S > 0) ∧ 
    (∃ v1 v2 t1 t2 t3 t4 : ℝ, 
      (S / 2 = v1 * t1) ∧ (26.25 = v2 * t2) ∧ 
      (S / 2 = v2 * t3) ∧ (31.2 = v1 * t4) ∧ 
      (∃ k : ℝ, k = (S - 31.2) / (v1 + v2) ∧ k > 0 ∧ 
        (S = 58))) := sorry

end NUMINAMATH_GPT_car_travel_distance_l2321_232125


namespace NUMINAMATH_GPT_student_A_more_stable_l2321_232186

-- Given conditions
def average_score (n : ℕ) (score : ℕ) := score = 110
def variance_A := 3.6
def variance_B := 4.4

-- Prove that student A has more stable scores than student B
theorem student_A_more_stable : variance_A < variance_B :=
by
  -- Skipping the actual proof
  sorry

end NUMINAMATH_GPT_student_A_more_stable_l2321_232186


namespace NUMINAMATH_GPT_b_share_l2321_232165

theorem b_share (a b c : ℕ) (h1 : a + b + c = 120) (h2 : a = b + 20) (h3 : a = c - 20) : b = 20 :=
by
  sorry

end NUMINAMATH_GPT_b_share_l2321_232165


namespace NUMINAMATH_GPT_triangle_inequality_l2321_232104

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b) / (a + b + c) > 1 / 2 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l2321_232104


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2321_232193

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ((0 < x ∧ x < 5) → (|x - 2| < 3)) ∧ ¬ ((|x - 2| < 3) → (0 < x ∧ x < 5)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2321_232193


namespace NUMINAMATH_GPT_acute_angles_in_triangle_l2321_232150

theorem acute_angles_in_triangle (α β γ : ℝ) (A_ext B_ext C_ext : ℝ) 
  (h_sum : α + β + γ = 180) 
  (h_ext1 : A_ext = 180 - β) 
  (h_ext2 : B_ext = 180 - γ) 
  (h_ext3 : C_ext = 180 - α) 
  (h_ext_acute1 : A_ext < 90 → β > 90) 
  (h_ext_acute2 : B_ext < 90 → γ > 90) 
  (h_ext_acute3 : C_ext < 90 → α > 90) : 
  ((α < 90 ∧ β < 90) ∨ (α < 90 ∧ γ < 90) ∨ (β < 90 ∧ γ < 90)) ∧ 
  ((A_ext < 90 → ¬ (B_ext < 90 ∨ C_ext < 90)) ∧ 
   (B_ext < 90 → ¬ (A_ext < 90 ∨ C_ext < 90)) ∧ 
   (C_ext < 90 → ¬ (A_ext < 90 ∨ B_ext < 90))) :=
sorry

end NUMINAMATH_GPT_acute_angles_in_triangle_l2321_232150


namespace NUMINAMATH_GPT_equivalent_discount_l2321_232101

theorem equivalent_discount (original_price : ℝ) (d1 d2 single_discount : ℝ) :
  original_price = 50 →
  d1 = 0.15 →
  d2 = 0.10 →
  single_discount = 0.235 →
  original_price * (1 - d1) * (1 - d2) = original_price * (1 - single_discount) :=
by
  intros
  sorry

end NUMINAMATH_GPT_equivalent_discount_l2321_232101


namespace NUMINAMATH_GPT_hunter_3_proposal_l2321_232131

theorem hunter_3_proposal {hunter1_coins hunter2_coins hunter3_coins : ℕ} :
  hunter3_coins = 99 ∧ hunter1_coins = 1 ∧ (hunter1_coins + hunter3_coins + hunter2_coins = 100) :=
  sorry

end NUMINAMATH_GPT_hunter_3_proposal_l2321_232131


namespace NUMINAMATH_GPT_bullet_trains_crossing_time_l2321_232103

theorem bullet_trains_crossing_time
  (length : ℝ)
  (time1 time2 : ℝ)
  (speed1 speed2 : ℝ)
  (relative_speed : ℝ)
  (total_distance : ℝ)
  (cross_time : ℝ)
  (h_length : length = 120)
  (h_time1 : time1 = 10)
  (h_time2 : time2 = 20)
  (h_speed1 : speed1 = length / time1)
  (h_speed2 : speed2 = length / time2)
  (h_relative_speed : relative_speed = speed1 + speed2)
  (h_total_distance : total_distance = length + length)
  (h_cross_time : cross_time = total_distance / relative_speed) :
  cross_time = 240 / 18 := 
by
  sorry

end NUMINAMATH_GPT_bullet_trains_crossing_time_l2321_232103


namespace NUMINAMATH_GPT_algae_coverage_double_l2321_232167

theorem algae_coverage_double (algae_cov : ℕ → ℝ) (h1 : ∀ n : ℕ, algae_cov (n + 2) = 2 * algae_cov n)
  (h2 : algae_cov 24 = 1) : algae_cov 18 = 0.125 :=
by
  sorry

end NUMINAMATH_GPT_algae_coverage_double_l2321_232167


namespace NUMINAMATH_GPT_range_of_a_l2321_232144

theorem range_of_a (x a : ℝ) (p : Prop) (q : Prop) (H₁ : p ↔ (x < -3 ∨ x > 1))
  (H₂ : q ↔ (x > a))
  (H₃ : ¬p → ¬q) (H₄ : ¬q → ¬p → false) : a ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2321_232144


namespace NUMINAMATH_GPT_enrollment_increase_1991_to_1992_l2321_232119

theorem enrollment_increase_1991_to_1992 (E E_1992 E_1993 : ℝ)
    (h1 : E_1993 = 1.26 * E)
    (h2 : E_1993 = 1.05 * E_1992) :
    ((E_1992 - E) / E) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_enrollment_increase_1991_to_1992_l2321_232119


namespace NUMINAMATH_GPT_calc_expression_l2321_232163

theorem calc_expression : 112 * 5^4 * 3^2 = 630000 := by
  sorry

end NUMINAMATH_GPT_calc_expression_l2321_232163


namespace NUMINAMATH_GPT_mow_lawn_time_l2321_232126

noncomputable def time_to_mow (lawn_length lawn_width: ℝ) 
(swat_width overlap width_conversion: ℝ) (speed: ℝ) : ℝ :=
(lawn_length * lawn_width) / (((swat_width - overlap) / width_conversion) * lawn_length * speed)

theorem mow_lawn_time : 
  time_to_mow 120 180 30 6 12 6000 = 1.8 := 
by
  -- Given:
  -- Lawn dimensions: 120 feet by 180 feet
  -- Mower swath: 30 inches with 6 inches overlap
  -- Walking speed: 6000 feet per hour
  -- Conversion factor: 12 inches = 1 foot
  sorry

end NUMINAMATH_GPT_mow_lawn_time_l2321_232126


namespace NUMINAMATH_GPT_add_fifteen_sub_fifteen_l2321_232124

theorem add_fifteen (n : ℕ) (m : ℕ) : n + m = 195 :=
by {
  sorry  -- placeholder for the actual proof
}

theorem sub_fifteen (n : ℕ) (m : ℕ) : n - m = 165 :=
by {
  sorry  -- placeholder for the actual proof
}

-- Let's instantiate these theorems with the specific values from the problem:
noncomputable def verify_addition : 180 + 15 = 195 :=
by exact add_fifteen 180 15

noncomputable def verify_subtraction : 180 - 15 = 165 :=
by exact sub_fifteen 180 15

end NUMINAMATH_GPT_add_fifteen_sub_fifteen_l2321_232124


namespace NUMINAMATH_GPT_problem_solution_l2321_232162

theorem problem_solution (x : ℝ) (h : (18 / 100) * 42 = (27 / 100) * x) : x = 28 :=
sorry

end NUMINAMATH_GPT_problem_solution_l2321_232162


namespace NUMINAMATH_GPT_Balint_claim_impossible_l2321_232155

-- Declare the lengths of the ladders and the vertical projection distance
def AC : ℝ := 3
def BD : ℝ := 2
def E_proj : ℝ := 1

-- State the problem conditions and what we need to prove
theorem Balint_claim_impossible (h1 : AC = 3) (h2 : BD = 2) (h3 : E_proj = 1) :
  False :=
  sorry

end NUMINAMATH_GPT_Balint_claim_impossible_l2321_232155


namespace NUMINAMATH_GPT_car_travel_distance_l2321_232153

noncomputable def car_distance_in_30_minutes : ℝ := 
  let train_speed : ℝ := 96
  let car_speed : ℝ := (5 / 8) * train_speed
  let travel_time : ℝ := 0.5  -- 30 minutes is 0.5 hours
  car_speed * travel_time

theorem car_travel_distance : car_distance_in_30_minutes = 30 := by
  sorry

end NUMINAMATH_GPT_car_travel_distance_l2321_232153


namespace NUMINAMATH_GPT_range_of_p_l2321_232133

def p (x : ℝ) : ℝ := (x^3 + 3)^2

theorem range_of_p :
  (∀ y, ∃ x ∈ Set.Ici (-1 : ℝ), p x = y) ↔ y ∈ Set.Ici (4 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_range_of_p_l2321_232133


namespace NUMINAMATH_GPT_jamie_hours_each_time_l2321_232105

theorem jamie_hours_each_time (hours_per_week := 2) (weeks := 6) (rate := 10) (total_earned := 360) : 
  ∃ (h : ℕ), h = 3 ∧ (hours_per_week * weeks * rate * h = total_earned) := 
by
  sorry

end NUMINAMATH_GPT_jamie_hours_each_time_l2321_232105


namespace NUMINAMATH_GPT_determine_base_l2321_232129

theorem determine_base (b : ℕ) (h : (3 * b + 1)^2 = b^3 + 2 * b + 1) : b = 10 :=
by
  sorry

end NUMINAMATH_GPT_determine_base_l2321_232129


namespace NUMINAMATH_GPT_top_z_teams_l2321_232181

theorem top_z_teams (n : ℕ) (h : (n * (n - 1)) / 2 = 45) : n = 10 := 
sorry

end NUMINAMATH_GPT_top_z_teams_l2321_232181


namespace NUMINAMATH_GPT_sequence_term_10_l2321_232164

theorem sequence_term_10 : ∃ n : ℕ, (1 / (n * (n + 2)) = 1 / 120) ∧ n = 10 := by
  sorry

end NUMINAMATH_GPT_sequence_term_10_l2321_232164


namespace NUMINAMATH_GPT_final_value_of_A_l2321_232147

theorem final_value_of_A (A : ℤ) (h₁ : A = 15) (h₂ : A = -A + 5) : A = -10 := 
by 
  sorry

end NUMINAMATH_GPT_final_value_of_A_l2321_232147


namespace NUMINAMATH_GPT_value_of_a_l2321_232120

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → |a * x + 1| ≤ 3) ↔ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l2321_232120


namespace NUMINAMATH_GPT_triangle_inequality_shortest_side_l2321_232157

theorem triangle_inequality_shortest_side (a b c : ℝ) (h_triangle: a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) 
  (h_inequality : a^2 + b^2 > 5 * c^2) : c ≤ a ∧ c ≤ b :=
sorry

end NUMINAMATH_GPT_triangle_inequality_shortest_side_l2321_232157


namespace NUMINAMATH_GPT_weight_of_brand_b_l2321_232195

theorem weight_of_brand_b (w_a w_b : ℕ) (vol_a vol_b : ℕ) (total_volume total_weight : ℕ) 
  (h1 : w_a = 950) 
  (h2 : vol_a = 3) 
  (h3 : vol_b = 2) 
  (h4 : total_volume = 4) 
  (h5 : total_weight = 3640) 
  (h6 : vol_a + vol_b = total_volume) 
  (h7 : vol_a * w_a + vol_b * w_b = total_weight) : 
  w_b = 395 := 
by {
  sorry
}

end NUMINAMATH_GPT_weight_of_brand_b_l2321_232195


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2321_232179

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 3 + a 9 + a 15 + a 21 = 8) :
  a 1 + a 23 = 4 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2321_232179


namespace NUMINAMATH_GPT_find_four_digit_numbers_l2321_232109

def isFourDigitNumber (n : ℕ) : Prop := (1000 ≤ n) ∧ (n < 10000)

noncomputable def solveABCD (AB CD : ℕ) : ℕ := 100 * AB + CD

theorem find_four_digit_numbers :
  ∀ (AB CD : ℕ),
    isFourDigitNumber (solveABCD AB CD) →
    solveABCD AB CD = AB * CD + AB ^ 2 →
      solveABCD AB CD = 1296 ∨ solveABCD AB CD = 3468 :=
by
  intros AB CD h1 h2
  sorry

end NUMINAMATH_GPT_find_four_digit_numbers_l2321_232109


namespace NUMINAMATH_GPT_total_cost_l2321_232171

def num_of_rings : ℕ := 2

def cost_per_ring : ℕ := 12

theorem total_cost : num_of_rings * cost_per_ring = 24 :=
by sorry

end NUMINAMATH_GPT_total_cost_l2321_232171


namespace NUMINAMATH_GPT_intersection_and_complement_find_m_l2321_232134

-- Define the sets A, B, C
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def C (m : ℝ) : Set ℝ := {x | m+1 ≤ x ∧ x ≤ 3*m}

-- State the first proof problem: intersection A ∩ B and complement of B
theorem intersection_and_complement (x : ℝ) : 
  (x ∈ (A ∩ B) ↔ (2 ≤ x ∧ x ≤ 3)) ∧ 
  (x ∈ (compl B) ↔ (x < 1 ∨ x > 4)) :=
by 
  sorry

-- State the second proof problem: find m satisfying A ∪ C(m) = A
theorem find_m (m : ℝ) (x : ℝ) : 
  (∀ x, (x ∈ A ∪ C m) ↔ (x ∈ A)) ↔ (m = 1) :=
by 
  sorry

end NUMINAMATH_GPT_intersection_and_complement_find_m_l2321_232134


namespace NUMINAMATH_GPT_number_of_pen_refills_l2321_232132

-- Conditions
variable (k : ℕ) (x : ℕ) (hk : k > 0) (hx : (4 + k) * x = 6)

-- Question and conclusion as a theorem statement
theorem number_of_pen_refills (hk : k > 0) (hx : (4 + k) * x = 6) : 2 * x = 2 :=
sorry

end NUMINAMATH_GPT_number_of_pen_refills_l2321_232132


namespace NUMINAMATH_GPT_sequence_from_625_to_629_l2321_232108

def arrows_repeating_pattern (n : ℕ) : ℕ := n % 5

theorem sequence_from_625_to_629 :
  arrows_repeating_pattern 625 = 0 ∧ arrows_repeating_pattern 629 = 4 →
  ∃ (seq : ℕ → ℕ), 
    (seq 0 = arrows_repeating_pattern 625) ∧
    (seq 1 = arrows_repeating_pattern (625 + 1)) ∧
    (seq 2 = arrows_repeating_pattern (625 + 2)) ∧
    (seq 3 = arrows_repeating_pattern (625 + 3)) ∧
    (seq 4 = arrows_repeating_pattern 629) := 
sorry

end NUMINAMATH_GPT_sequence_from_625_to_629_l2321_232108


namespace NUMINAMATH_GPT_max_regions_with_6_chords_l2321_232138

-- Definition stating the number of regions created by k chords
def regions_by_chords (k : ℕ) : ℕ :=
  1 + (k * (k + 1)) / 2

-- Lean statement for the proof problem
theorem max_regions_with_6_chords : regions_by_chords 6 = 22 :=
  by sorry

end NUMINAMATH_GPT_max_regions_with_6_chords_l2321_232138


namespace NUMINAMATH_GPT_diagonal_length_l2321_232173

noncomputable def rectangle_diagonal (p : ℝ) (r : ℝ) (d : ℝ) : Prop :=
  ∃ k : ℝ, p = 2 * ((5 * k) + (2 * k)) ∧ r = 5 / 2 ∧ 
           d = Real.sqrt (((5 * k)^2 + (2 * k)^2)) 

theorem diagonal_length 
  (p : ℝ) (r : ℝ) (d : ℝ)
  (h₁ : p = 72) 
  (h₂ : r = 5 / 2)
  : rectangle_diagonal p r d ↔ d = 194 / 7 := 
sorry

end NUMINAMATH_GPT_diagonal_length_l2321_232173


namespace NUMINAMATH_GPT_tank_capacity_l2321_232177

theorem tank_capacity (T : ℚ) (h1 : 0 ≤ T)
  (h2 : 9 + (3 / 4) * T = (9 / 10) * T) : T = 60 :=
sorry

end NUMINAMATH_GPT_tank_capacity_l2321_232177


namespace NUMINAMATH_GPT_greatest_b_max_b_value_l2321_232100

theorem greatest_b (b y : ℤ) (h : b > 0) (hy : y^2 + b*y = -21) : b ≤ 22 :=
sorry

theorem max_b_value : ∃ b : ℤ, (∀ y : ℤ, y^2 + b*y = -21 → b > 0) ∧ (b = 22) :=
sorry

end NUMINAMATH_GPT_greatest_b_max_b_value_l2321_232100


namespace NUMINAMATH_GPT_sequence_subsequence_l2321_232192

theorem sequence_subsequence :
  ∃ (a : Fin 101 → ℕ), 
  (∀ i, a i = i + 1) ∧ 
  ∃ (b : Fin 11 → ℕ), 
  (b 0 < b 1 ∧ b 1 < b 2 ∧ b 2 < b 3 ∧ b 3 < b 4 ∧ b 4 < b 5 ∧ 
  b 5 < b 6 ∧ b 6 < b 7 ∧ b 7 < b 8 ∧ b 8 < b 9 ∧ b 9 < b 10) ∨ 
  (b 0 > b 1 ∧ b 1 > b 2 ∧ b 2 > b 3 ∧ b 3 > b 4 ∧ b 4 > b 5 ∧ 
  b 5 > b 6 ∧ b 6 > b 7 ∧ b 7 > b 8 ∧ b 8 > b 9 ∧ b 9 > b 10) :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_subsequence_l2321_232192


namespace NUMINAMATH_GPT_tanya_total_sticks_l2321_232178

theorem tanya_total_sticks (n : ℕ) (h : n = 11) : 3 * (n * (n + 1) / 2) = 198 :=
by
  have H : n = 11 := h
  sorry

end NUMINAMATH_GPT_tanya_total_sticks_l2321_232178


namespace NUMINAMATH_GPT_minimum_cost_l2321_232160

noncomputable def total_cost (x : ℝ) : ℝ :=
  (1800 / (x + 5)) + 0.5 * x

theorem minimum_cost : 
  (∃ x : ℝ, x = 55 ∧ total_cost x = 57.5) :=
  sorry

end NUMINAMATH_GPT_minimum_cost_l2321_232160


namespace NUMINAMATH_GPT_anes_age_l2321_232122

theorem anes_age (w w_d : ℤ) (n : ℤ) 
  (h1 : 1436 ≤ w ∧ w < 1445)
  (h2 : 1606 ≤ w_d ∧ w_d < 1615)
  (h3 : w_d = w + n * 40) : 
  n = 4 :=
sorry

end NUMINAMATH_GPT_anes_age_l2321_232122


namespace NUMINAMATH_GPT_intersection_point_x_value_l2321_232116

theorem intersection_point_x_value :
  ∃ x y : ℚ, (y = 3 * x - 22) ∧ (3 * x + y = 100) ∧ (x = 20 + 1 / 3) := by
  sorry

end NUMINAMATH_GPT_intersection_point_x_value_l2321_232116


namespace NUMINAMATH_GPT_sandy_fingernails_length_l2321_232118

/-- 
Sandy, who just turned 12 this month, has a goal for tying the world record for longest fingernails, 
which is 26 inches. Her fingernails grow at a rate of one-tenth of an inch per month. 
She will be 32 when she achieves the world record. 
Prove that her fingernails are currently 2 inches long.
-/
theorem sandy_fingernails_length 
  (current_age : ℕ) (world_record_length : ℝ) (growth_rate : ℝ) (years_to_achieve : ℕ) : 
  current_age = 12 → 
  world_record_length = 26 → 
  growth_rate = 0.1 → 
  years_to_achieve = 20 →
  (world_record_length - growth_rate * 12 * years_to_achieve) = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sandy_fingernails_length_l2321_232118


namespace NUMINAMATH_GPT_train_length_l2321_232112

/-- Proof problem: 
  Given the speed of a train is 52 km/hr and it crosses a 280-meter long platform in 18 seconds,
  prove that the length of the train is 259.92 meters.
-/
theorem train_length (speed_kmh : ℕ) (platform_length : ℕ) (time_sec : ℕ) (speed_mps : ℝ) 
  (distance_covered : ℝ) (train_length : ℝ) :
  speed_kmh = 52 → platform_length = 280 → time_sec = 18 → 
  speed_mps = (speed_kmh * 1000) / 3600 → distance_covered = speed_mps * time_sec →
  train_length = distance_covered - platform_length →
  train_length = 259.92 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_train_length_l2321_232112


namespace NUMINAMATH_GPT_consignment_shop_total_items_l2321_232158

variable (x y z t n : ℕ)

noncomputable def totalItems (n : ℕ) := n + n + n + 3 * n

theorem consignment_shop_total_items :
  ∃ (x y z t n : ℕ), 
    3 * n * y + n * x + n * z + n * t = 240 ∧
    t = 10 * n ∧
    z + x = y + t + 4 ∧
    x + y + 24 = t + z ∧
    y ≤ 6 ∧
    totalItems n = 18 :=
by
  sorry

end NUMINAMATH_GPT_consignment_shop_total_items_l2321_232158


namespace NUMINAMATH_GPT_last_four_digits_of_5_pow_2011_l2321_232135

theorem last_four_digits_of_5_pow_2011 : (5^2011 % 10000) = 8125 := by
  sorry

end NUMINAMATH_GPT_last_four_digits_of_5_pow_2011_l2321_232135


namespace NUMINAMATH_GPT_find_third_side_of_triangle_l2321_232180

noncomputable def area_triangle_given_sides_angle {a b c : ℝ} (A : ℝ) : Prop :=
  A = 1/2 * a * b * Real.sin c

noncomputable def cosine_law_third_side {a b c : ℝ} (cosα : ℝ) : Prop :=
  c^2 = a^2 + b^2 - 2 * a * b * cosα

theorem find_third_side_of_triangle (a b : ℝ) (Area : ℝ) (h_a : a = 2 * Real.sqrt 2) (h_b : b = 3) (h_Area : Area = 3) :
  ∃ c : ℝ, (c = Real.sqrt 5 ∨ c = Real.sqrt 29) :=
by
  sorry

end NUMINAMATH_GPT_find_third_side_of_triangle_l2321_232180


namespace NUMINAMATH_GPT_division_addition_example_l2321_232123

theorem division_addition_example : 12 / (1 / 6) + 3 = 75 := by
  sorry

end NUMINAMATH_GPT_division_addition_example_l2321_232123


namespace NUMINAMATH_GPT_cost_of_mozzarella_cheese_l2321_232114

-- Define the problem conditions as Lean definitions
def blendCostPerKg : ℝ := 696.05
def romanoCostPerKg : ℝ := 887.75
def weightMozzarella : ℝ := 19
def weightRomano : ℝ := 18.999999999999986  -- Practically the same as 19 in context
def totalWeight : ℝ := weightMozzarella + weightRomano

-- Define the expected result for the cost per kilogram of mozzarella cheese
def expectedMozzarellaCostPerKg : ℝ := 504.40

-- Theorem statement to verify the cost of mozzarella cheese
theorem cost_of_mozzarella_cheese :
  weightMozzarella * (expectedMozzarellaCostPerKg : ℝ) + weightRomano * romanoCostPerKg = totalWeight * blendCostPerKg := by
  sorry

end NUMINAMATH_GPT_cost_of_mozzarella_cheese_l2321_232114


namespace NUMINAMATH_GPT_candy_store_problem_l2321_232110

variable (S : ℝ)
variable (not_caught_percentage : ℝ) (sample_percentage : ℝ)
variable (caught_percentage : ℝ := 1 - not_caught_percentage)

theorem candy_store_problem
  (h1 : not_caught_percentage = 0.15)
  (h2 : sample_percentage = 25.88235294117647) :
  caught_percentage * sample_percentage = 22 := by
  sorry

end NUMINAMATH_GPT_candy_store_problem_l2321_232110


namespace NUMINAMATH_GPT_isosceles_triangle_median_length_l2321_232198

noncomputable def median_length (b h : ℝ) : ℝ :=
  let a := Real.sqrt ((b / 2) ^ 2 + h ^ 2)
  let m_a := Real.sqrt ((2 * a ^ 2 + 2 * b ^ 2 - a ^ 2) / 4)
  m_a

theorem isosceles_triangle_median_length :
  median_length 16 10 = Real.sqrt 146 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_median_length_l2321_232198


namespace NUMINAMATH_GPT_rectangle_perimeter_l2321_232194

theorem rectangle_perimeter (l d : ℝ) (h_l : l = 8) (h_d : d = 17) :
  ∃ w : ℝ, (d^2 = l^2 + w^2) ∧ (2*l + 2*w = 46) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l2321_232194


namespace NUMINAMATH_GPT_max_sum_of_abc_l2321_232117

theorem max_sum_of_abc (A B C : ℕ) (h₁ : A ≠ B) (h₂ : B ≠ C) (h₃ : A ≠ C) (h₄ : A * B * C = 2310) : 
  A + B + C ≤ 52 :=
sorry

end NUMINAMATH_GPT_max_sum_of_abc_l2321_232117


namespace NUMINAMATH_GPT_find_a_minus_b_plus_c_l2321_232146

def a_n (n : ℕ) : ℕ := 4 * n - 3

def S_n (a b c n : ℕ) : ℕ := 2 * a * n ^ 2 + b * n + c

theorem find_a_minus_b_plus_c
  (a b c : ℕ)
  (h : ∀ n : ℕ, n > 0 → S_n a b c n = 2 * n ^ 2 - n)
  : a - b + c = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_minus_b_plus_c_l2321_232146


namespace NUMINAMATH_GPT_number_of_men_in_first_group_l2321_232143

-- Define the conditions
def condition1 (M : ℕ) : Prop := M * 80 = 20 * 40

-- State the main theorem to be proved
theorem number_of_men_in_first_group (M : ℕ) (h : condition1 M) : M = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_men_in_first_group_l2321_232143


namespace NUMINAMATH_GPT_radius_of_large_circle_l2321_232176

/-- Five circles are described with the given properties. -/
def small_circle_radius : ℝ := 2

/-- The angle between any centers of the small circles is 72 degrees due to equal spacing. -/
def angle_between_centers : ℝ := 72

/-- The final theorem states that the radius of the larger circle is as follows. -/
theorem radius_of_large_circle (number_of_circles : ℕ)
        (radius_small : ℝ)
        (angle : ℝ)
        (internally_tangent : ∀ (i : ℕ), i < number_of_circles → Prop)
        (externally_tangent : ∀ (i j : ℕ), i ≠ j → i < number_of_circles → j < number_of_circles → Prop) :
  number_of_circles = 5 →
  radius_small = small_circle_radius →
  angle = angle_between_centers →
  (∃ R : ℝ, R = 4 * Real.sqrt 5 - 2) 
:= by
  -- mathematical proof goes here
  sorry

end NUMINAMATH_GPT_radius_of_large_circle_l2321_232176


namespace NUMINAMATH_GPT_probability_of_non_defective_pens_l2321_232184

-- Define the number of total pens, defective pens, and pens to be selected
def total_pens : ℕ := 15
def defective_pens : ℕ := 5
def selected_pens : ℕ := 3

-- Define the number of non-defective pens
def non_defective_pens : ℕ := total_pens - defective_pens

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the total ways to choose 3 pens from 15 pens
def total_ways : ℕ := combination total_pens selected_pens

-- Define the ways to choose 3 non-defective pens from the non-defective pens
def non_defective_ways : ℕ := combination non_defective_pens selected_pens

-- Define the probability
def probability : ℚ := non_defective_ways / total_ways

-- Statement we need to prove
theorem probability_of_non_defective_pens : probability = 120 / 455 := by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_probability_of_non_defective_pens_l2321_232184


namespace NUMINAMATH_GPT_examination_duration_in_hours_l2321_232107

theorem examination_duration_in_hours 
  (total_questions : ℕ)
  (type_A_questions : ℕ)
  (time_for_A_problems : ℝ) 
  (time_ratio_A_to_B : ℝ)
  (total_time_for_A : ℝ) 
  (total_time : ℝ) :
  total_questions = 200 → 
  type_A_questions = 15 → 
  time_ratio_A_to_B = 2 → 
  total_time_for_A = 25.116279069767444 →
  total_time = (total_time_for_A + 185 * (25.116279069767444 / 15 / 2)) → 
  total_time / 60 = 3 :=
by sorry

end NUMINAMATH_GPT_examination_duration_in_hours_l2321_232107


namespace NUMINAMATH_GPT_calculate_lassis_from_nine_mangoes_l2321_232172

variable (mangoes_lassis_ratio : ℕ → ℕ → Prop)
variable (cost_per_mango : ℕ)

def num_lassis (mangoes : ℕ) : ℕ :=
  5 * mangoes
  
theorem calculate_lassis_from_nine_mangoes
  (h1 : mangoes_lassis_ratio 15 3)
  (h2 : cost_per_mango = 2) :
  num_lassis 9 = 45 :=
by
  sorry

end NUMINAMATH_GPT_calculate_lassis_from_nine_mangoes_l2321_232172


namespace NUMINAMATH_GPT_commute_times_abs_difference_l2321_232191

theorem commute_times_abs_difference (x y : ℝ)
  (h_avg : (x + y + 10 + 11 + 9) / 5 = 10)
  (h_var : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) 
  : |x - y| = 4 :=
sorry

end NUMINAMATH_GPT_commute_times_abs_difference_l2321_232191


namespace NUMINAMATH_GPT_max_books_per_student_l2321_232154

theorem max_books_per_student
  (total_students : ℕ)
  (students_0_books : ℕ)
  (students_1_book : ℕ)
  (students_2_books : ℕ)
  (students_at_least_3_books : ℕ)
  (avg_books_per_student : ℕ)
  (max_books_limit : ℕ)
  (total_books_available : ℕ) :
  total_students = 20 →
  students_0_books = 2 →
  students_1_book = 10 →
  students_2_books = 5 →
  students_at_least_3_books = total_students - students_0_books - students_1_book - students_2_books →
  avg_books_per_student = 2 →
  max_books_limit = 5 →
  total_books_available = 60 →
  avg_books_per_student * total_students = 40 →
  total_books_available = 60 →
  max_books_limit = 5 :=
by sorry

end NUMINAMATH_GPT_max_books_per_student_l2321_232154


namespace NUMINAMATH_GPT_possible_triangle_perimeters_l2321_232121

theorem possible_triangle_perimeters :
  {p | ∃ (a b c : ℝ), ((a = 3 ∨ a = 6) ∧ (b = 3 ∨ b = 6) ∧ (c = 3 ∨ c = 6)) ∧
                        (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
                        p = a + b + c} = {9, 15, 18} :=
by
  sorry

end NUMINAMATH_GPT_possible_triangle_perimeters_l2321_232121


namespace NUMINAMATH_GPT_conditional_probability_l2321_232139

-- Given probabilities:
def p_a : ℚ := 5/23
def p_b : ℚ := 7/23
def p_c : ℚ := 1/23
def p_a_and_b : ℚ := 2/23
def p_a_and_c : ℚ := 1/23
def p_b_and_c : ℚ := 1/23
def p_a_and_b_and_c : ℚ := 1/23

-- Theorem statement to prove:
theorem conditional_probability : p_a_and_b_and_c / p_a_and_c = 1 :=
by
  sorry

end NUMINAMATH_GPT_conditional_probability_l2321_232139


namespace NUMINAMATH_GPT_find_2nd_month_sales_l2321_232185

def sales_of_1st_month : ℝ := 2500
def sales_of_3rd_month : ℝ := 9855
def sales_of_4th_month : ℝ := 7230
def sales_of_5th_month : ℝ := 7000
def sales_of_6th_month : ℝ := 11915
def average_sales : ℝ := 7500
def months : ℕ := 6
def total_required_sales : ℝ := average_sales * months
def total_known_sales : ℝ := sales_of_1st_month + sales_of_3rd_month + sales_of_4th_month + sales_of_5th_month + sales_of_6th_month

theorem find_2nd_month_sales : 
  ∃ (sales_of_2nd_month : ℝ), total_required_sales = sales_of_1st_month + sales_of_2nd_month + sales_of_3rd_month + sales_of_4th_month + sales_of_5th_month + sales_of_6th_month ∧ sales_of_2nd_month = 10500 := by
  sorry

end NUMINAMATH_GPT_find_2nd_month_sales_l2321_232185


namespace NUMINAMATH_GPT_prove_solutions_l2321_232148

noncomputable def solution1 (x : ℝ) : Prop :=
  3 * x^2 + 6 = abs (-25 + x)

theorem prove_solutions :
  solution1 ( (-1 + Real.sqrt 229) / 6 ) ∧ solution1 ( (-1 - Real.sqrt 229) / 6 ) :=
by
  sorry

end NUMINAMATH_GPT_prove_solutions_l2321_232148


namespace NUMINAMATH_GPT_stephan_cannot_afford_laptop_l2321_232141

noncomputable def initial_laptop_price : ℝ := sorry

theorem stephan_cannot_afford_laptop (P₀ : ℝ) (h_rate : 0 < 0.06) (h₁ : initial_laptop_price = P₀) : 
  56358 < P₀ * (1.06)^2 :=
by 
  sorry

end NUMINAMATH_GPT_stephan_cannot_afford_laptop_l2321_232141


namespace NUMINAMATH_GPT_existence_not_implied_by_validity_l2321_232102

-- Let us formalize the theorem and then show that its validity does not imply the existence of such a function.

-- Definitions for condition (A) and the theorem statement
axiom condition_A (f : ℝ → ℝ) : Prop
axiom theorem_239 : ∀ f, condition_A f → ∃ T, ∀ x, f (x + T) = f x

-- Translation of the problem statement into Lean
theorem existence_not_implied_by_validity :
  (∀ f, condition_A f → ∃ T, ∀ x, f (x + T) = f x) → 
  ¬ (∃ f, condition_A f) :=
sorry

end NUMINAMATH_GPT_existence_not_implied_by_validity_l2321_232102


namespace NUMINAMATH_GPT_total_opponent_score_l2321_232199

-- Definitions based on the conditions
def team_scores : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

def lost_by_one_point (scores : List ℕ) : Bool :=
  scores = [3, 4, 5]

def scored_twice_as_many (scores : List ℕ) : Bool :=
  scores = [2, 3]

def scored_three_times_as_many (scores : List ℕ) : Bool :=
  scores = [2, 3, 3]

-- Proof problem:
theorem total_opponent_score :
  ∀ (lost_scores twice_scores thrice_scores : List ℕ),
    lost_by_one_point lost_scores →
    scored_twice_as_many twice_scores →
    scored_three_times_as_many thrice_scores →
    (lost_scores.sum + twice_scores.sum + thrice_scores.sum) = 25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_opponent_score_l2321_232199


namespace NUMINAMATH_GPT_simplify_expression_l2321_232197

variable (x : ℝ)

def expr := (5*x^10 + 8*x^8 + 3*x^6) + (2*x^12 + 3*x^10 + x^8 + 4*x^6 + 2*x^2 + 7)

theorem simplify_expression : expr x = 2*x^12 + 8*x^10 + 9*x^8 + 7*x^6 + 2*x^2 + 7 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2321_232197


namespace NUMINAMATH_GPT_perimeter_percent_increase_l2321_232127

noncomputable def side_increase (s₁ s₂_ratio s₃_ratio s₄_ratio s₅_ratio : ℝ) : ℝ :=
  let s₂ := s₂_ratio * s₁
  let s₃ := s₃_ratio * s₂
  let s₄ := s₄_ratio * s₃
  let s₅ := s₅_ratio * s₄
  s₅

theorem perimeter_percent_increase (s₁ : ℝ) (s₂_ratio s₃_ratio s₄_ratio s₅_ratio : ℝ) (P₁ := 3 * s₁)
    (P₅ := 3 * side_increase s₁ s₂_ratio s₃_ratio s₄_ratio s₅_ratio) :
    s₁ = 4 → s₂_ratio = 1.5 → s₃_ratio = 1.3 → s₄_ratio = 1.5 → s₅_ratio = 1.3 →
    P₅ = 45.63 →
    ((P₅ - P₁) / P₁) * 100 = 280.3 :=
by
  intros
  -- proof goes here
  sorry

end NUMINAMATH_GPT_perimeter_percent_increase_l2321_232127


namespace NUMINAMATH_GPT_car_speed_624km_in_2_2_5_hours_l2321_232145

theorem car_speed_624km_in_2_2_5_hours : 
  ∀ (distance time_in_hours : ℝ), distance = 624 → time_in_hours = 2 + (2/5) → distance / time_in_hours = 260 :=
by
  intros distance time_in_hours h_dist h_time
  sorry

end NUMINAMATH_GPT_car_speed_624km_in_2_2_5_hours_l2321_232145


namespace NUMINAMATH_GPT_total_earnings_l2321_232190

noncomputable def daily_wage_a (C : ℝ) := (3 * C) / 5
noncomputable def daily_wage_b (C : ℝ) := (4 * C) / 5
noncomputable def daily_wage_c (C : ℝ) := C

noncomputable def earnings_a (C : ℝ) := daily_wage_a C * 6
noncomputable def earnings_b (C : ℝ) := daily_wage_b C * 9
noncomputable def earnings_c (C : ℝ) := daily_wage_c C * 4

theorem total_earnings (C : ℝ) (h : C = 115) : 
  earnings_a C + earnings_b C + earnings_c C = 1702 :=
by
  sorry

end NUMINAMATH_GPT_total_earnings_l2321_232190


namespace NUMINAMATH_GPT_pool_buckets_l2321_232168

theorem pool_buckets (buckets_george_per_round buckets_harry_per_round rounds : ℕ) 
  (h_george : buckets_george_per_round = 2) 
  (h_harry : buckets_harry_per_round = 3) 
  (h_rounds : rounds = 22) : 
  buckets_george_per_round + buckets_harry_per_round * rounds = 110 := 
by 
  sorry

end NUMINAMATH_GPT_pool_buckets_l2321_232168


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l2321_232156

variable {a b c x y z : ℝ}

-- Define the conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : x = a + 1 / b - 1
axiom h5 : y = b + 1 / c - 1
axiom h6 : z = c + 1 / a - 1
axiom h7 : x > 0
axiom h8 : y > 0
axiom h9 : z > 0

-- The statement we need to prove
theorem cyclic_sum_inequality : (x * y) / (Real.sqrt (x * y) + 2) + (y * z) / (Real.sqrt (y * z) + 2) + (z * x) / (Real.sqrt (z * x) + 2) ≥ 1 :=
sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l2321_232156


namespace NUMINAMATH_GPT_total_fencing_cost_l2321_232187

def side1 : ℕ := 34
def side2 : ℕ := 28
def side3 : ℕ := 45
def side4 : ℕ := 50
def side5 : ℕ := 55

def cost1_per_meter : ℕ := 2
def cost2_per_meter : ℕ := 2
def cost3_per_meter : ℕ := 3
def cost4_per_meter : ℕ := 3
def cost5_per_meter : ℕ := 4

def total_cost : ℕ :=
  side1 * cost1_per_meter +
  side2 * cost2_per_meter +
  side3 * cost3_per_meter +
  side4 * cost4_per_meter +
  side5 * cost5_per_meter

theorem total_fencing_cost : total_cost = 629 := by
  sorry

end NUMINAMATH_GPT_total_fencing_cost_l2321_232187


namespace NUMINAMATH_GPT_find_h_for_expression_l2321_232196

theorem find_h_for_expression (a k : ℝ) (h : ℝ) :
  (∃ a k : ℝ, ∀ x : ℝ, x^2 - 6*x + 1 = a*(x - h)^3 + k) ↔ h = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_h_for_expression_l2321_232196


namespace NUMINAMATH_GPT_outermost_diameter_l2321_232137

def radius_of_fountain := 6 -- derived from the information that 12/2 = 6
def width_of_garden := 9
def width_of_inner_walking_path := 3
def width_of_outer_walking_path := 7

theorem outermost_diameter :
  2 * (radius_of_fountain + width_of_garden + width_of_inner_walking_path + width_of_outer_walking_path) = 50 :=
by
  sorry

end NUMINAMATH_GPT_outermost_diameter_l2321_232137


namespace NUMINAMATH_GPT_ellipse_focus_distance_l2321_232183

theorem ellipse_focus_distance :
  ∀ {x y : ℝ},
    (x^2) / 25 + (y^2) / 16 = 1 →
    (dist (x, y) (3, 0) = 8) →
    dist (x, y) (-3, 0) = 2 :=
by
  intro x y h₁ h₂
  sorry

end NUMINAMATH_GPT_ellipse_focus_distance_l2321_232183


namespace NUMINAMATH_GPT_cos_double_angle_l2321_232166

theorem cos_double_angle (α : ℝ) (h : Real.sin (Real.pi + α) = 1 / 3) : Real.cos (2 * α) = 7 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_cos_double_angle_l2321_232166


namespace NUMINAMATH_GPT_shaded_area_l2321_232142

-- Let A be the length of the side of the smaller square
def A : ℝ := 4

-- Let B be the length of the side of the larger square
def B : ℝ := 12

-- The problem is to prove that the area of the shaded region is 10 square inches
theorem shaded_area (A B : ℝ) (hA : A = 4) (hB : B = 12) :
  (A * A) - (1/2 * (B / (B + A)) * A * B) = 10 := by
  sorry

end NUMINAMATH_GPT_shaded_area_l2321_232142


namespace NUMINAMATH_GPT_ratio_of_increase_to_current_l2321_232149

-- Define the constants for the problem
def current_deductible : ℝ := 3000
def increase_deductible : ℝ := 2000

-- State the theorem that needs to be proven
theorem ratio_of_increase_to_current : 
  (increase_deductible / current_deductible) = (2 / 3) :=
by sorry

end NUMINAMATH_GPT_ratio_of_increase_to_current_l2321_232149


namespace NUMINAMATH_GPT_least_five_digit_perfect_square_cube_l2321_232140

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end NUMINAMATH_GPT_least_five_digit_perfect_square_cube_l2321_232140


namespace NUMINAMATH_GPT_price_per_maple_tree_l2321_232115

theorem price_per_maple_tree 
  (cabin_price : ℕ) (initial_cash : ℕ) (remaining_cash : ℕ)
  (num_cypress : ℕ) (price_cypress : ℕ)
  (num_pine : ℕ) (price_pine : ℕ)
  (num_maple : ℕ) 
  (total_raised_from_trees : ℕ) :
  cabin_price = 129000 ∧ 
  initial_cash = 150 ∧ 
  remaining_cash = 350 ∧ 
  num_cypress = 20 ∧ 
  price_cypress = 100 ∧ 
  num_pine = 600 ∧ 
  price_pine = 200 ∧ 
  num_maple = 24 ∧ 
  total_raised_from_trees = 129350 - initial_cash → 
  (price_maple : ℕ) = 300 :=
by 
  sorry

end NUMINAMATH_GPT_price_per_maple_tree_l2321_232115


namespace NUMINAMATH_GPT_alex_chairs_l2321_232159

theorem alex_chairs (x y z : ℕ) (h : x + y + z = 74) : z = 74 - x - y :=
by
  sorry

end NUMINAMATH_GPT_alex_chairs_l2321_232159


namespace NUMINAMATH_GPT_mean_of_remaining_number_is_2120_l2321_232182

theorem mean_of_remaining_number_is_2120 (a1 a2 a3 a4 a5 a6 : ℕ) 
    (h1 : a1 = 1451) (h2 : a2 = 1723) (h3 : a3 = 1987) (h4 : a4 = 2056) 
    (h5 : a5 = 2191) (h6 : a6 = 2212) 
    (mean_five : (a1 + a2 + a3 + a4 + a5) = 9500):
-- Prove that the mean of the remaining number a6 is 2120
  (a6 = 2120) :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_mean_of_remaining_number_is_2120_l2321_232182


namespace NUMINAMATH_GPT_range_of_f_l2321_232113

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^x else -Real.log x / Real.log 2

theorem range_of_f : Set.Iic 2 = Set.range f :=
  by sorry

end NUMINAMATH_GPT_range_of_f_l2321_232113


namespace NUMINAMATH_GPT_discount_percentage_l2321_232130

theorem discount_percentage (original_price new_price : ℕ) (h₁ : original_price = 120) (h₂ : new_price = 96) : 
  ((original_price - new_price) * 100 / original_price) = 20 := 
by
  -- sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_discount_percentage_l2321_232130

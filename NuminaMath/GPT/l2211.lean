import Mathlib

namespace solution1_solution2_l2211_221114

noncomputable def problem1 : Prop :=
  ∃ (a b : ℤ), 
  (∃ (n : ℤ), 3*a - 14 = n ∧ a - 2 = n) ∧ 
  (b - 15 = -27) ∧ 
  a = 4 ∧ 
  b = -12 ∧ 
  (4*a + b = 4)

noncomputable def problem2 : Prop :=
  ∀ (a b : ℤ), 
  (a = 4) ∧ 
  (b = -12) → 
  (4*a + b = 4) → 
  (∃ n, n^2 = 4 ∧ (n = 2 ∨ n = -2))

theorem solution1 : problem1 := by { sorry }
theorem solution2 : problem2 := by { sorry }

end solution1_solution2_l2211_221114


namespace probability_of_drawing_ball_1_is_2_over_5_l2211_221191

noncomputable def probability_of_drawing_ball_1 : ℚ :=
  let total_balls := [1, 2, 3, 4, 5]
  let draw_pairs := [ (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5) ]
  let favorable_pairs := [ (1, 2), (1, 3), (1, 4), (1, 5) ]
  (favorable_pairs.length : ℚ) / (draw_pairs.length : ℚ)

theorem probability_of_drawing_ball_1_is_2_over_5 :
  probability_of_drawing_ball_1 = 2 / 5 :=
by sorry

end probability_of_drawing_ball_1_is_2_over_5_l2211_221191


namespace false_log_exists_x_l2211_221109

theorem false_log_exists_x {x : ℝ} : ¬ ∃ x : ℝ, Real.log x = 0 :=
by sorry

end false_log_exists_x_l2211_221109


namespace abs_sum_eq_two_l2211_221198

theorem abs_sum_eq_two (a b c : ℤ) (h : (a - b) ^ 10 + (a - c) ^ 10 = 1) : 
  abs (a - b) + abs (b - c) + abs (c - a) = 2 := 
sorry

end abs_sum_eq_two_l2211_221198


namespace questionnaire_visitors_l2211_221105

noncomputable def total_visitors :=
  let V := 600
  let E := (3 / 4) * V
  V

theorem questionnaire_visitors:
  ∃ (V : ℕ), V = 600 ∧
  (∀ (E : ℕ), E = (3 / 4) * V ∧ E + 150 = V) :=
by
    use 600
    sorry

end questionnaire_visitors_l2211_221105


namespace pentagon_PT_value_l2211_221150

-- Given conditions
def length_QR := 3
def length_RS := 3
def length_ST := 3
def angle_T := 90
def angle_P := 120
def angle_Q := 120
def angle_R := 120

-- The target statement to prove
theorem pentagon_PT_value (a b : ℝ) (h : PT = a + 3 * Real.sqrt b) : a + b = 6 :=
sorry

end pentagon_PT_value_l2211_221150


namespace wife_weekly_savings_correct_l2211_221178

-- Define constants
def monthly_savings_husband := 225
def num_months := 4
def weeks_per_month := 4
def num_weeks := num_months * weeks_per_month
def stocks_per_share := 50
def num_shares := 25
def invested_amount := num_shares * stocks_per_share
def total_savings := 2 * invested_amount

-- Weekly savings amount to prove
def weekly_savings_wife := 100

-- Total savings calculation condition
theorem wife_weekly_savings_correct :
  (monthly_savings_husband * num_months + weekly_savings_wife * num_weeks) = total_savings :=
by
  sorry

end wife_weekly_savings_correct_l2211_221178


namespace possible_value_of_sum_l2211_221137

theorem possible_value_of_sum (p q r : ℝ) (h₀ : q = p * (4 - p)) (h₁ : r = q * (4 - q)) (h₂ : p = r * (4 - r)) 
  (h₃ : p ≠ q ∧ p ≠ r ∧ q ≠ r) : p + q + r = 6 :=
sorry

end possible_value_of_sum_l2211_221137


namespace mother_kept_one_third_l2211_221161

-- Define the problem conditions
def total_sweets : ℕ := 27
def eldest_sweets : ℕ := 8
def youngest_sweets : ℕ := eldest_sweets / 2
def second_sweets : ℕ := 6
def total_children_sweets : ℕ := eldest_sweets + youngest_sweets + second_sweets
def sweets_mother_kept : ℕ := total_sweets - total_children_sweets
def fraction_mother_kept : ℚ := sweets_mother_kept / total_sweets

-- Prove the fraction of sweets the mother kept
theorem mother_kept_one_third : fraction_mother_kept = 1 / 3 := 
  by
    sorry

end mother_kept_one_third_l2211_221161


namespace find_second_number_l2211_221134

theorem find_second_number
  (a : ℝ) (b : ℝ)
  (h : a = 1280)
  (h_percent : 0.25 * a = 0.20 * b + 190) :
  b = 650 :=
sorry

end find_second_number_l2211_221134


namespace original_number_l2211_221180

theorem original_number (x : ℕ) : 
  (∃ y : ℕ, y = x + 28 ∧ (y % 5 = 0) ∧ (y % 6 = 0) ∧ (y % 4 = 0) ∧ (y % 3 = 0)) → x = 32 :=
by
  sorry

end original_number_l2211_221180


namespace car_owners_without_motorcycles_l2211_221155

theorem car_owners_without_motorcycles 
    (total_adults : ℕ) 
    (car_owners : ℕ) 
    (motorcycle_owners : ℕ) 
    (total_with_vehicles : total_adults = 500) 
    (total_car_owners : car_owners = 480) 
    (total_motorcycle_owners : motorcycle_owners = 120) : 
    car_owners - (car_owners + motorcycle_owners - total_adults) = 380 := 
by
    sorry

end car_owners_without_motorcycles_l2211_221155


namespace train_length_l2211_221103

theorem train_length 
  (speed_jogger_kmph : ℕ)
  (initial_distance_m : ℕ)
  (speed_train_kmph : ℕ)
  (pass_time_s : ℕ)
  (h_speed_jogger : speed_jogger_kmph = 9)
  (h_initial_distance : initial_distance_m = 230)
  (h_speed_train : speed_train_kmph = 45)
  (h_pass_time : pass_time_s = 35) : 
  ∃ length_train_m : ℕ, length_train_m = 580 := sorry

end train_length_l2211_221103


namespace alpha_in_third_quadrant_l2211_221159

theorem alpha_in_third_quadrant (α : ℝ)
 (h₁ : Real.tan (α - 3 * Real.pi) > 0)
 (h₂ : Real.sin (-α + Real.pi) < 0) :
 (0 < α % (2 * Real.pi) ∧ α % (2 * Real.pi) < Real.pi) := 
sorry

end alpha_in_third_quadrant_l2211_221159


namespace expansion_l2211_221149

variable (x : ℝ)

noncomputable def expr : ℝ := (3 / 4) * (8 / (x^2) + 5 * x - 6)

theorem expansion :
  expr x = (6 / (x^2)) + (15 * x / 4) - 4.5 :=
by
  sorry

end expansion_l2211_221149


namespace quadratic_equation_real_roots_k_value_l2211_221116

theorem quadratic_equation_real_roots_k_value :
  (∀ k : ℕ, (∃ x : ℝ, k * x^2 - 3 * x + 2 = 0) <-> k = 1) :=
by
  sorry
  
end quadratic_equation_real_roots_k_value_l2211_221116


namespace valid_q_range_l2211_221171

noncomputable def polynomial_has_nonneg_root (q : ℝ) : Prop :=
  ∃ x : ℝ, x ≥ 0 ∧ (x^4 + q*x^3 + x^2 + q*x + 4 = 0)

theorem valid_q_range (q : ℝ) : polynomial_has_nonneg_root q → q ≤ -2 * Real.sqrt 2 := 
sorry

end valid_q_range_l2211_221171


namespace shekar_average_is_81_9_l2211_221160

def shekar_average_marks (marks : List ℕ) : ℚ :=
  (marks.sum : ℚ) / marks.length

theorem shekar_average_is_81_9 :
  shekar_average_marks [92, 78, 85, 67, 89, 74, 81, 95, 70, 88] = 81.9 :=
by
  sorry

end shekar_average_is_81_9_l2211_221160


namespace third_candle_remaining_fraction_l2211_221139

theorem third_candle_remaining_fraction (t : ℝ) 
  (h1 : 0 < t)
  (second_candle_fraction_remaining : ℝ := 2/5)
  (third_candle_fraction_remaining : ℝ := 3/7)
  (second_candle_burned_fraction : ℝ := 3/5)
  (third_candle_burned_fraction : ℝ := 4/7)
  (second_candle_burn_rate : ℝ := 3 / (5 * t))
  (third_candle_burn_rate : ℝ := 4 / (7 * t))
  (remaining_burn_time_second : ℝ := (2 * t) / 3)
  (third_candle_burned_in_remaining_time : ℝ := (2 * t * 4) / (3 * 7 * t))
  (common_denominator_third : ℝ := 21)
  (converted_third_candle_fraction_remaining : ℝ := 9 / 21)
  (third_candle_fraction_subtracted : ℝ := 8 / 21) :
  (converted_third_candle_fraction_remaining - third_candle_fraction_subtracted) = 1 / 21 := by
  sorry

end third_candle_remaining_fraction_l2211_221139


namespace complement_union_l2211_221152

open Set

namespace ProofFormalization

/-- Declaration of the universal set U, and sets A and B -/
def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

def complement {α : Type*} (s t : Set α) : Set α := t \ s

/-- Theorem statement that proves the complement of A ∪ B with respect to U is {5} -/
theorem complement_union :
  complement (A ∪ B) U = {5} :=
by
  sorry

end ProofFormalization

end complement_union_l2211_221152


namespace proof_problem_l2211_221122

-- Define the equation of the parabola
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 1

-- Define the circle C with center (h, k) and radius r
def circle_eq (h k r : ℝ) (x y : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Define condition of line that intersects the circle C at points A and B
def line_eq (a : ℝ) (x y : ℝ) : Prop := x - y + a = 0

-- Condition: OA ⊥ OB
def perpendicular_cond (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem stating the proof problem
theorem proof_problem :
  (∃ (h k r : ℝ),
    circle_eq h k r 3 1 ∧
    circle_eq h k r 5 0 ∧
    circle_eq h k r 1 0 ∧
    h = 3 ∧ k = 1 ∧ r = 3) ∧
    (∃ (a : ℝ),
      (∀ (x1 y1 x2 y2 : ℝ),
        line_eq a x1 y1 ∧
        circle_eq 3 1 3 x1 y1 ∧
        line_eq a x2 y2 ∧
        circle_eq 3 1 3 x2 y2 → 
        perpendicular_cond x1 y1 x2 y2) →
      a = -1) :=
by
  sorry

end proof_problem_l2211_221122


namespace least_number_remainder_5_l2211_221182

theorem least_number_remainder_5 (n : ℕ) : 
  n % 12 = 5 ∧ n % 15 = 5 ∧ n % 20 = 5 ∧ n % 54 = 5 → n = 545 := 
  by
  sorry

end least_number_remainder_5_l2211_221182


namespace frac_addition_l2211_221169

theorem frac_addition :
  (3 / 5) + (2 / 15) = 11 / 15 :=
sorry

end frac_addition_l2211_221169


namespace boat_travel_time_downstream_l2211_221125

-- Define the given conditions and statement to prove
theorem boat_travel_time_downstream (B : ℝ) (C : ℝ) (Us : ℝ) (Ds : ℝ) :
  (C = B / 4) ∧ (Us = B - C) ∧ (Ds = B + C) ∧ (Us = 3) ∧ (15 / Us = 5) ∧ (15 / Ds = 3) :=
by
  -- Provide the proof here; currently using sorry to skip the proof
  sorry

end boat_travel_time_downstream_l2211_221125


namespace family_travel_time_l2211_221111

theorem family_travel_time (D : ℕ) (v1 v2 : ℕ) (d1 d2 : ℕ) (t1 t2 : ℕ) :
  D = 560 → 
  v1 = 35 → 
  v2 = 40 → 
  d1 = D / 2 →
  d2 = D / 2 →
  t1 = d1 / v1 →
  t2 = d2 / v2 → 
  t1 + t2 = 15 :=
by
  sorry

end family_travel_time_l2211_221111


namespace no_such_continuous_function_exists_l2211_221147

theorem no_such_continuous_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (Continuous f) ∧ ∀ x : ℝ, ((∃ q : ℚ, f x = q) ↔ ∀ q' : ℚ, f (x + 1) ≠ q') :=
sorry

end no_such_continuous_function_exists_l2211_221147


namespace yuna_grandfather_age_l2211_221101

def age_yuna : ℕ := 8
def age_father : ℕ := age_yuna + 20
def age_grandfather : ℕ := age_father + 25

theorem yuna_grandfather_age : age_grandfather = 53 := by
  sorry

end yuna_grandfather_age_l2211_221101


namespace cost_of_paving_l2211_221162

noncomputable def length : Float := 5.5
noncomputable def width : Float := 3.75
noncomputable def cost_per_sq_meter : Float := 600

theorem cost_of_paving :
  (length * width * cost_per_sq_meter) = 12375 := by
  sorry

end cost_of_paving_l2211_221162


namespace remainder_of_x_mod_11_l2211_221107

theorem remainder_of_x_mod_11 {x : ℤ} (h : x % 66 = 14) : x % 11 = 3 :=
sorry

end remainder_of_x_mod_11_l2211_221107


namespace student_average_always_less_l2211_221181

theorem student_average_always_less (w x y z: ℝ) (hwx: w < x) (hxy: x < y) (hyz: y < z) :
  let A' := (w + x + y + z) / 4
  let B' := (2 * w + 2 * x + y + z) / 6
  B' < A' :=
by
  intro A' B'
  sorry

end student_average_always_less_l2211_221181


namespace non_zero_real_x_solution_l2211_221142

noncomputable section

variables {x : ℝ} (hx : x ≠ 0)

theorem non_zero_real_x_solution 
  (h : (3 * x)^5 = (9 * x)^4) : 
  x = 27 := by
  sorry

end non_zero_real_x_solution_l2211_221142


namespace total_cost_proof_l2211_221136

-- Define the cost of items
def cost_of_1kg_of_mango (M : ℚ) : Prop := sorry
def cost_of_1kg_of_rice (R : ℚ) : Prop := sorry
def cost_of_1kg_of_flour (F : ℚ) : Prop := F = 23

-- Condition 1: cost of some kg of mangos is equal to the cost of 24 kg of rice
def condition1 (M R : ℚ) (x : ℚ) : Prop := M * x = R * 24

-- Condition 2: cost of 6 kg of flour equals to the cost of 2 kg of rice
def condition2 (R : ℚ) : Prop := 23 * 6 = R * 2

-- Final proof problem
theorem total_cost_proof (M R F : ℚ) (x : ℚ) 
  (h1: condition1 M R x) 
  (h2: condition2 R) 
  (h3: cost_of_1kg_of_flour F) :
  4 * (69 * 24 / x) + 3 * R + 5 * 23 = 1978 :=
sorry

end total_cost_proof_l2211_221136


namespace mean_of_remaining_three_numbers_l2211_221174

variable {a b c : ℝ}

theorem mean_of_remaining_three_numbers (h1 : (a + b + c + 103) / 4 = 90) : (a + b + c) / 3 = 85.7 :=
by
  -- Sorry placeholder for the proof
  sorry

end mean_of_remaining_three_numbers_l2211_221174


namespace equation_circle_iff_a_equals_neg_one_l2211_221140

theorem equation_circle_iff_a_equals_neg_one :
  (∀ x y : ℝ, ∃ k : ℝ, a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = k * (x^2 + y^2)) ↔ 
  a = -1 :=
by sorry

end equation_circle_iff_a_equals_neg_one_l2211_221140


namespace arithmetic_sequence_twenty_fourth_term_l2211_221196

-- Given definitions (conditions)
def third_term (a d : ℚ) : ℚ := a + 2 * d
def tenth_term (a d : ℚ) : ℚ := a + 9 * d
def twenty_fourth_term (a d : ℚ) : ℚ := a + 23 * d

-- The main theorem to be proved
theorem arithmetic_sequence_twenty_fourth_term 
  (a d : ℚ) 
  (h1 : third_term a d = 7) 
  (h2 : tenth_term a d = 27) :
  twenty_fourth_term a d = 67 := by
  sorry

end arithmetic_sequence_twenty_fourth_term_l2211_221196


namespace continuous_stripe_probability_l2211_221175

noncomputable def probability_continuous_stripe_encircle_cube : ℚ :=
  let total_combinations : ℕ := 2^6
  let favor_combinations : ℕ := 3 * 4 -- 3 pairs of parallel faces, with 4 valid combinations each
  favor_combinations / total_combinations

theorem continuous_stripe_probability :
  probability_continuous_stripe_encircle_cube = 3 / 16 :=
by
  sorry

end continuous_stripe_probability_l2211_221175


namespace find_girls_l2211_221187

theorem find_girls (n : ℕ) (h : 1 - (1 / Nat.choose (3 + n) 3) = 34 / 35) : n = 4 :=
  sorry

end find_girls_l2211_221187


namespace regular_price_of_each_shirt_l2211_221129

theorem regular_price_of_each_shirt (P : ℝ) :
    let total_shirts := 20
    let sale_price_per_shirt := 0.8 * P
    let tax_rate := 0.10
    let total_paid := 264
    let total_price := total_shirts * sale_price_per_shirt * (1 + tax_rate)
    total_price = total_paid → P = 15 :=
by
  intros
  sorry

end regular_price_of_each_shirt_l2211_221129


namespace circle_diameter_tangents_l2211_221123

open Real

theorem circle_diameter_tangents {x y : ℝ} (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) :
  ∃ d : ℝ, d = sqrt (x * y) :=
by
  sorry

end circle_diameter_tangents_l2211_221123


namespace transformed_line_equation_l2211_221118

theorem transformed_line_equation {A B C x₀ y₀ : ℝ} 
    (h₀ : ¬(A = 0 ∧ B = 0)) 
    (h₁ : A * x₀ + B * y₀ + C = 0) : 
    ∀ {x y : ℝ}, A * x + B * y + C = 0 ↔ A * (x - x₀) + B * (y - y₀) = 0 :=
by
    sorry

end transformed_line_equation_l2211_221118


namespace sara_spent_on_hotdog_l2211_221189

-- Define variables for the costs
def costSalad : ℝ := 5.1
def totalLunchBill : ℝ := 10.46

-- Define the cost of the hotdog
def costHotdog : ℝ := totalLunchBill - costSalad

-- The theorem we need to prove
theorem sara_spent_on_hotdog : costHotdog = 5.36 := by
  -- Proof would go here (if required)
  sorry

end sara_spent_on_hotdog_l2211_221189


namespace triangle_inequality_l2211_221197

theorem triangle_inequality (a b c : ℝ) (h1 : a + b + c = 2) :
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
sorry

end triangle_inequality_l2211_221197


namespace problem_l2211_221163

theorem problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
by
  sorry

end problem_l2211_221163


namespace sqrt_difference_of_cubes_is_integer_l2211_221104

theorem sqrt_difference_of_cubes_is_integer (a b : ℕ) (h1 : a = 105) (h2 : b = 104) :
  (Int.sqrt (a^3 - b^3) = 181) :=
by
  sorry

end sqrt_difference_of_cubes_is_integer_l2211_221104


namespace max_perimeter_of_triangle_l2211_221124

theorem max_perimeter_of_triangle (x : ℕ) 
  (h1 : 3 < x) 
  (h2 : x < 15) 
  (h3 : 7 + 8 > x) 
  (h4 : 7 + x > 8) 
  (h5 : 8 + x > 7) :
  x = 14 ∧ 7 + 8 + x = 29 := 
by {
  sorry
}

end max_perimeter_of_triangle_l2211_221124


namespace problem_1_l2211_221145

theorem problem_1 :
  (-7/4) - (19/3) - 9/4 + 10/3 = -7 := by
  sorry

end problem_1_l2211_221145


namespace negation_of_universal_statement_l2211_221193

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x^2 ≠ x) ↔ ∃ x : ℝ, x^2 = x :=
by sorry

end negation_of_universal_statement_l2211_221193


namespace abcd_zero_l2211_221131

theorem abcd_zero (a b c d : ℝ) (h1 : a + b + c + d = 0) (h2 : ab + ac + bc + bd + ad + cd = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end abcd_zero_l2211_221131


namespace common_area_of_rectangle_and_circle_l2211_221126

theorem common_area_of_rectangle_and_circle :
  let l := 10
  let w := 2 * Real.sqrt 5
  let r := 3
  ∃ (common_area : ℝ), common_area = 9 * Real.pi :=
by
  let l := 10
  let w := 2 * Real.sqrt 5
  let r := 3
  have common_area := 9 * Real.pi
  use common_area
  sorry

end common_area_of_rectangle_and_circle_l2211_221126


namespace trigonometric_equation_solution_l2211_221127

theorem trigonometric_equation_solution (x : ℝ) (k : ℤ) :
  5.14 * (Real.sin (3 * x)) + Real.sin (5 * x) = 2 * (Real.cos (2 * x)) ^ 2 - 2 * (Real.sin (3 * x)) ^ 2 →
  (∃ k : ℤ, x = (π / 2) * (2 * k + 1)) ∨ (∃ k : ℤ, x = (π / 18) * (4 * k + 1)) :=
  by
  intro h
  sorry

end trigonometric_equation_solution_l2211_221127


namespace leila_toys_l2211_221176

theorem leila_toys:
  ∀ (x : ℕ),
  (∀ l m : ℕ, l = 2 * x ∧ m = 3 * 19 ∧ m = l + 7 → x = 25) :=
by
  sorry

end leila_toys_l2211_221176


namespace problem_l2211_221143

def otimes (x y : ℝ) : ℝ := x^3 + 5 * x * y - y

theorem problem (a : ℝ) : 
  otimes a (otimes a a) = 5 * a^4 + 24 * a^3 - 10 * a^2 + a :=
by
  sorry

end problem_l2211_221143


namespace max_sum_prod_48_l2211_221190

theorem max_sum_prod_48 (spadesuit heartsuit : Nat) (h: spadesuit * heartsuit = 48) : spadesuit + heartsuit ≤ 49 :=
sorry

end max_sum_prod_48_l2211_221190


namespace water_level_balance_l2211_221133

noncomputable def exponential_decay (a n t : ℝ) : ℝ := a * Real.exp (n * t)

theorem water_level_balance
  (a : ℝ)
  (n : ℝ)
  (m : ℝ)
  (h5 : exponential_decay a n 5 = a / 2)
  (h8 : exponential_decay a n m = a / 8) :
  m = 10 := by
  sorry

end water_level_balance_l2211_221133


namespace abs_sub_lt_five_solution_set_l2211_221170

theorem abs_sub_lt_five_solution_set (x : ℝ) : |x - 3| < 5 ↔ -2 < x ∧ x < 8 :=
by sorry

end abs_sub_lt_five_solution_set_l2211_221170


namespace carl_city_mileage_l2211_221164

noncomputable def city_mileage (miles_city mpg_highway cost_per_gallon total_cost miles_highway : ℝ) : ℝ :=
  let total_gallons := total_cost / cost_per_gallon
  let gallons_highway := miles_highway / mpg_highway
  let gallons_city := total_gallons - gallons_highway
  miles_city / gallons_city

theorem carl_city_mileage :
  city_mileage 60 40 3 42 200 = 20 / 3 := by
  sorry

end carl_city_mileage_l2211_221164


namespace luke_initial_stickers_l2211_221184

theorem luke_initial_stickers (x : ℕ) (h : x + 12 + 20 - 5 - 8 = 39) : x = 20 := 
by 
  sorry

end luke_initial_stickers_l2211_221184


namespace average_salary_all_workers_l2211_221130

-- Definitions based on the conditions
def num_technicians : ℕ := 7
def num_other_workers : ℕ := 7
def avg_salary_technicians : ℕ := 12000
def avg_salary_other_workers : ℕ := 8000
def total_workers : ℕ := 14

-- Total salary calculations based on the conditions
def total_salary_technicians : ℕ := num_technicians * avg_salary_technicians
def total_salary_other_workers : ℕ := num_other_workers * avg_salary_other_workers
def total_salary_all_workers : ℕ := total_salary_technicians + total_salary_other_workers

-- The statement to be proved
theorem average_salary_all_workers : total_salary_all_workers / total_workers = 10000 :=
by
  -- proof will be added here
  sorry

end average_salary_all_workers_l2211_221130


namespace combine_quadratic_radicals_l2211_221167

theorem combine_quadratic_radicals (x : ℝ) (h : 3 * x + 5 = 2 * x + 7) : x = 2 :=
by
  sorry

end combine_quadratic_radicals_l2211_221167


namespace range_of_m_l2211_221151

theorem range_of_m (m : ℝ) :
  (∀ x : ℕ, (x = 1 ∨ x = 2 ∨ x = 3) → (3 * x - m ≤ 0)) ↔ 9 ≤ m ∧ m < 12 :=
by
  sorry

end range_of_m_l2211_221151


namespace total_spider_legs_l2211_221192

theorem total_spider_legs (num_legs_single_spider group_spider_count: ℕ) 
      (h1: num_legs_single_spider = 8) 
      (h2: group_spider_count = (num_legs_single_spider / 2) + 10) :
      group_spider_count * num_legs_single_spider = 112 := 
by
  sorry

end total_spider_legs_l2211_221192


namespace circle_area_l2211_221146

open Real

def given_equation (r θ : ℝ) : Prop := r = 3 * cos θ - 4 * sin θ

theorem circle_area (r θ : ℝ) (h : given_equation r θ) : 
  ∃ (c : ℝ × ℝ) (R : ℝ), c = (3 / 2, -2) ∧ R = 5 / 2 ∧ π * R^2 = 25 / 4 * π :=
sorry

end circle_area_l2211_221146


namespace bobs_total_profit_l2211_221153

theorem bobs_total_profit :
  let cost_parent_dog := 250
  let num_parent_dogs := 2
  let num_puppies := 6
  let cost_food_vaccinations := 500
  let cost_advertising := 150
  let selling_price_parent_dog := 200
  let selling_price_puppy := 350
  let total_cost_parent_dogs := num_parent_dogs * cost_parent_dog
  let total_cost_puppies := cost_food_vaccinations + cost_advertising
  let total_revenue_puppies := num_puppies * selling_price_puppy
  let total_revenue_parent_dogs := num_parent_dogs * selling_price_parent_dog
  let total_revenue := total_revenue_puppies + total_revenue_parent_dogs
  let total_cost := total_cost_parent_dogs + total_cost_puppies
  let total_profit := total_revenue - total_cost
  total_profit = 1350 :=
by
  sorry

end bobs_total_profit_l2211_221153


namespace find_b_l2211_221128

theorem find_b (b : ℚ) (h : b * (-3) - (b - 1) * 5 = b - 3) : b = 8 / 9 :=
by
  sorry

end find_b_l2211_221128


namespace remainder_of_98_mult_102_div_12_l2211_221102

theorem remainder_of_98_mult_102_div_12 : (98 * 102) % 12 = 0 := by
    sorry

end remainder_of_98_mult_102_div_12_l2211_221102


namespace isosceles_triangle_perimeter_l2211_221141

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 2 ∧ b = 5 ∨ a = 5 ∧ b = 2):
  ∃ c : ℕ, (c = a ∨ c = b) ∧ 2 * c + (if c = a then b else a) = 12 :=
by
  sorry

end isosceles_triangle_perimeter_l2211_221141


namespace problem_1_problem_2_l2211_221120

-- Define the sets A, B, C
def SetA (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def SetB : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def SetC : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

-- Problem 1
theorem problem_1 (a : ℝ) : SetA a = SetB → a = 5 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (SetA a ∩ SetB).Nonempty ∧ (SetA a ∩ SetC = ∅) → a = -2 := by
  sorry

end problem_1_problem_2_l2211_221120


namespace relay_race_total_time_is_correct_l2211_221194

-- Define the time taken by each runner
def time_Ainslee : ℕ := 72
def time_Bridget : ℕ := (10 * time_Ainslee) / 9
def time_Cecilia : ℕ := (3 * time_Bridget) / 4
def time_Dana : ℕ := (5 * time_Cecilia) / 6

-- Define the total time and convert to minutes and seconds
def total_time_seconds : ℕ := time_Ainslee + time_Bridget + time_Cecilia + time_Dana
def total_time_minutes := total_time_seconds / 60
def total_time_remainder := total_time_seconds % 60

theorem relay_race_total_time_is_correct :
  total_time_minutes = 4 ∧ total_time_remainder = 22 :=
by
  -- All intermediate values can be calculated using the definitions
  -- provided above correctly.
  sorry

end relay_race_total_time_is_correct_l2211_221194


namespace factorization_solution_1_factorization_solution_2_factorization_solution_3_l2211_221199

noncomputable def factorization_problem_1 (m : ℝ) : Prop :=
  -3 * m^3 + 12 * m = -3 * m * (m + 2) * (m - 2)

noncomputable def factorization_problem_2 (x y : ℝ) : Prop :=
  2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2

noncomputable def factorization_problem_3 (a : ℝ) : Prop :=
  a^4 + 3 * a^2 - 4 = (a^2 + 4) * (a + 1) * (a - 1)

-- Lean statements for the proofs
theorem factorization_solution_1 (m : ℝ) : factorization_problem_1 m :=
  by sorry

theorem factorization_solution_2 (x y : ℝ) : factorization_problem_2 x y :=
  by sorry

theorem factorization_solution_3 (a : ℝ) : factorization_problem_3 a :=
  by sorry

end factorization_solution_1_factorization_solution_2_factorization_solution_3_l2211_221199


namespace tan_add_l2211_221158

open Real

-- Define positive acute angles
def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2

-- Theorem: Tangent addition formula
theorem tan_add (α β : ℝ) (hα : acute_angle α) (hβ : acute_angle β) :
  tan (α + β) = (tan α + tan β) / (1 - tan α * tan β) :=
  sorry

end tan_add_l2211_221158


namespace Mia_and_dad_time_to_organize_toys_l2211_221135

theorem Mia_and_dad_time_to_organize_toys :
  let total_toys := 60
  let dad_add_rate := 6
  let mia_remove_rate := 4
  let net_gain_per_cycle := dad_add_rate - mia_remove_rate
  let seconds_per_cycle := 30
  let total_needed_cycles := (total_toys - 2) / net_gain_per_cycle -- 58 toys by the end of repeated cycles, 2 is to ensure dad's last placement
  let last_cycle_time := seconds_per_cycle
  let total_time_seconds := total_needed_cycles * seconds_per_cycle + last_cycle_time
  let total_time_minutes := total_time_seconds / 60
  total_time_minutes = 15 :=
by
  sorry

end Mia_and_dad_time_to_organize_toys_l2211_221135


namespace henry_books_l2211_221195

theorem henry_books (initial_books packed_boxes each_box room_books coffee_books kitchen_books taken_books : ℕ)
  (h1 : initial_books = 99)
  (h2 : packed_boxes = 3)
  (h3 : each_box = 15)
  (h4 : room_books = 21)
  (h5 : coffee_books = 4)
  (h6 : kitchen_books = 18)
  (h7 : taken_books = 12) :
  initial_books - (packed_boxes * each_box + room_books + coffee_books + kitchen_books) + taken_books = 23 :=
by
  sorry

end henry_books_l2211_221195


namespace problem1_l2211_221115

theorem problem1 (f : ℝ → ℝ) (x : ℝ) : 
  (f (x + 1/x) = x^2 + 1/x^2) -> f x = x^2 - 2 := 
sorry

end problem1_l2211_221115


namespace new_sign_cost_l2211_221148

theorem new_sign_cost 
  (p_s : ℕ) (p_c : ℕ) (n : ℕ) (h_ps : p_s = 30) (h_pc : p_c = 26) (h_n : n = 10) : 
  (p_s - p_c) * n / 2 = 20 := 
by 
  sorry

end new_sign_cost_l2211_221148


namespace gcd_paving_courtyard_l2211_221121

theorem gcd_paving_courtyard :
  Nat.gcd 378 595 = 7 :=
by
  sorry

end gcd_paving_courtyard_l2211_221121


namespace gift_exchange_equation_l2211_221110

theorem gift_exchange_equation
  (x : ℕ)
  (total_gifts : ℕ)
  (H : total_gifts = 56)
  (H1 : 2 * total_gifts = x * (x - 1)) :
  x * (x - 1) = 56 :=
by
  sorry

end gift_exchange_equation_l2211_221110


namespace oxen_count_b_l2211_221117

theorem oxen_count_b 
  (a_oxen : ℕ) (a_months : ℕ)
  (b_months : ℕ) (x : ℕ)
  (c_oxen : ℕ) (c_months : ℕ)
  (total_rent : ℝ) (c_rent : ℝ)
  (h1 : a_oxen * a_months = 70)
  (h2 : c_oxen * c_months = 45)
  (h3 : c_rent / total_rent = 27 / 105)
  (h4 : total_rent = 105) :
  x = 12 :=
by 
  sorry

end oxen_count_b_l2211_221117


namespace solve_x_l2211_221166

theorem solve_x (x : ℝ) (h : 9 - 4 / x = 7 + 8 / x) : x = 6 := 
by 
  sorry

end solve_x_l2211_221166


namespace total_chips_eaten_l2211_221165

theorem total_chips_eaten (dinner_chips after_dinner_chips : ℕ) (h1 : dinner_chips = 1) (h2 : after_dinner_chips = 2 * dinner_chips) : dinner_chips + after_dinner_chips = 3 := by
  sorry

end total_chips_eaten_l2211_221165


namespace correct_speed_l2211_221183

noncomputable def distance (t : ℝ) := 50 * (t + 5 / 60)
noncomputable def distance2 (t : ℝ) := 70 * (t - 5 / 60)

theorem correct_speed : 
  ∃ r : ℝ, 
    (∀ t : ℝ, distance t = distance2 t → r = 55) := 
by
  sorry

end correct_speed_l2211_221183


namespace purple_marble_probability_l2211_221132

theorem purple_marble_probability (blue green : ℝ) (p : ℝ) 
  (h_blue : blue = 0.25)
  (h_green : green = 0.4)
  (h_sum : blue + green + p = 1) : p = 0.35 :=
by
  sorry

end purple_marble_probability_l2211_221132


namespace pool_half_capacity_at_6_hours_l2211_221138

noncomputable def double_volume_every_hour (t : ℕ) : ℕ := 2 ^ t

theorem pool_half_capacity_at_6_hours (V : ℕ) (h : ∀ t : ℕ, V = double_volume_every_hour 8) : double_volume_every_hour 6 = V / 2 := by
  sorry

end pool_half_capacity_at_6_hours_l2211_221138


namespace percentage_spent_on_household_items_l2211_221186

def Raja_income : ℝ := 37500
def clothes_percentage : ℝ := 0.20
def medicines_percentage : ℝ := 0.05
def savings_amount : ℝ := 15000

theorem percentage_spent_on_household_items : 
  (Raja_income - (clothes_percentage * Raja_income + medicines_percentage * Raja_income + savings_amount)) / Raja_income * 100 = 35 :=
  sorry

end percentage_spent_on_household_items_l2211_221186


namespace evaluate_expression_l2211_221177

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) :
  2 * a^2 - 3 * b^2 + 4 * a * b = -43 :=
by
  sorry

end evaluate_expression_l2211_221177


namespace problem_solution_l2211_221172

theorem problem_solution (u v : ℤ) (h₁ : 0 < v) (h₂ : v < u) (h₃ : u^2 + 3 * u * v = 451) : u + v = 21 :=
sorry

end problem_solution_l2211_221172


namespace find_x_given_y_and_ratio_l2211_221173

variable (x y k : ℝ)

theorem find_x_given_y_and_ratio :
  (∀ x y, (5 * x - 6) / (2 * y + 20) = k) →
  (5 * 3 - 6) / (2 * 5 + 20) = k →
  y = 15 →
  x = 21 / 5 :=
by 
  intro h1 h2 hy
  -- proof steps would go here
  sorry

end find_x_given_y_and_ratio_l2211_221173


namespace regular_octagon_diagonal_l2211_221156

variable {a b c : ℝ}

-- Define a function to check for a regular octagon where a, b, c are respective side, shortest diagonal, and longest diagonal
def is_regular_octagon (a b c : ℝ) : Prop :=
  -- Here, we assume the standard geometric properties of a regular octagon.
  -- In a real formalization, we might model the octagon directly.

  -- longest diagonal c of a regular octagon (spans 4 sides)
  c = 2 * a

theorem regular_octagon_diagonal (a b c : ℝ) (h : is_regular_octagon a b c) : c = 2 * a :=
by
  exact h

end regular_octagon_diagonal_l2211_221156


namespace rhombus_diagonal_l2211_221179

theorem rhombus_diagonal (d2 : ℝ) (area : ℝ) (d1 : ℝ) : d2 = 15 → area = 127.5 → d1 = 17 :=
by
  intros h1 h2
  sorry

end rhombus_diagonal_l2211_221179


namespace find_original_strength_l2211_221108

variable (original_strength : ℕ)
variable (total_students : ℕ := original_strength + 12)
variable (original_avg_age : ℕ := 40)
variable (new_students : ℕ := 12)
variable (new_students_avg_age : ℕ := 32)
variable (new_avg_age_reduction : ℕ := 4)
variable (new_avg_age : ℕ := original_avg_age - new_avg_age_reduction)

theorem find_original_strength (h : (original_avg_age * original_strength + new_students * new_students_avg_age) / total_students = new_avg_age) :
  original_strength = 12 := 
sorry

end find_original_strength_l2211_221108


namespace least_value_of_d_l2211_221119

theorem least_value_of_d (c d : ℕ) (hc_pos : 0 < c) (hd_pos : 0 < d)
  (hc_factors : (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a ≠ b ∧ c = a * b) ∨ (∃ p : ℕ, p > 1 ∧ c = p^3))
  (hd_factors : ∃ factors : ℕ, factors = c ∧ ∃ divisors : Finset ℕ, divisors.card = factors ∧ ∀ k ∈ divisors, d % k = 0)
  (div_cd : d % c = 0) : d = 18 :=
sorry

end least_value_of_d_l2211_221119


namespace initial_toys_count_l2211_221112

theorem initial_toys_count (T : ℕ) (h : 10 * T + 300 = 580) : T = 28 :=
by
  sorry

end initial_toys_count_l2211_221112


namespace john_frank_age_ratio_l2211_221168

theorem john_frank_age_ratio
  (F J : ℕ)
  (h1 : F + 4 = 16)
  (h2 : J - F = 15)
  (h3 : ∃ k : ℕ, J + 3 = k * (F + 3)) :
  (J + 3) / (F + 3) = 2 :=
by
  sorry

end john_frank_age_ratio_l2211_221168


namespace net_emails_received_l2211_221188

-- Define the conditions
def emails_received_morning : ℕ := 3
def emails_sent_morning : ℕ := 2
def emails_received_afternoon : ℕ := 5
def emails_sent_afternoon : ℕ := 1

-- Define the problem statement
theorem net_emails_received :
  emails_received_morning - emails_sent_morning + emails_received_afternoon - emails_sent_afternoon = 5 := by
  sorry

end net_emails_received_l2211_221188


namespace prairie_total_area_l2211_221113

theorem prairie_total_area :
  let dust_covered := 64535
  let untouched := 522
  (dust_covered + untouched) = 65057 :=
by {
  let dust_covered := 64535
  let untouched := 522
  trivial
}

end prairie_total_area_l2211_221113


namespace simple_annual_interest_rate_l2211_221154

noncomputable def monthly_interest_payment : ℝ := 216
noncomputable def principal_amount : ℝ := 28800
noncomputable def number_of_months_in_a_year : ℕ := 12

theorem simple_annual_interest_rate :
  ((monthly_interest_payment * number_of_months_in_a_year) / principal_amount) * 100 = 9 := by
sorry

end simple_annual_interest_rate_l2211_221154


namespace rectangle_area_l2211_221106

variables (y : ℝ) (length : ℝ) (width : ℝ)

-- Definitions based on conditions
def is_diagonal_y (length width y : ℝ) : Prop :=
  y^2 = length^2 + width^2

def is_length_three_times_width (length width : ℝ) : Prop :=
  length = 3 * width

-- Statement to prove
theorem rectangle_area (y : ℝ) (length width : ℝ)
  (h1 : is_diagonal_y length width y)
  (h2 : is_length_three_times_width length width) :
  length * width = 3 * (y^2 / 10) :=
sorry

end rectangle_area_l2211_221106


namespace derivative_of_constant_function_l2211_221157

-- Define the constant function
def f (x : ℝ) : ℝ := 0

-- State the theorem
theorem derivative_of_constant_function : deriv f 0 = 0 := by
  -- Proof will go here, but we use sorry to skip it
  sorry

end derivative_of_constant_function_l2211_221157


namespace min_likes_both_l2211_221100

-- Definitions corresponding to the conditions
def total_people : ℕ := 200
def likes_beethoven : ℕ := 160
def likes_chopin : ℕ := 150

-- Problem statement to prove
theorem min_likes_both : ∃ x : ℕ, x = 110 ∧ x = likes_beethoven - (total_people - likes_chopin) := by
  sorry

end min_likes_both_l2211_221100


namespace original_cost_price_l2211_221185

theorem original_cost_price (P : ℝ) 
  (h1 : P - 0.07 * P = 0.93 * P)
  (h2 : 0.93 * P + 0.02 * 0.93 * P = 0.9486 * P)
  (h3 : 0.9486 * P * 1.05 = 0.99603 * P)
  (h4 : 0.93 * P * 0.95 = 0.8835 * P)
  (h5 : 0.8835 * P + 0.02 * 0.8835 * P = 0.90117 * P)
  (h6 : 0.99603 * P - 5 = (0.90117 * P) * 1.10)
: P = 5 / 0.004743 :=
by
  sorry

end original_cost_price_l2211_221185


namespace marie_daily_rent_l2211_221144

noncomputable def daily_revenue (bread_loaves : ℕ) (bread_price : ℝ) (cakes : ℕ) (cake_price : ℝ) : ℝ :=
  bread_loaves * bread_price + cakes * cake_price

noncomputable def total_profit (daily_revenue : ℝ) (days : ℕ) (cash_register_cost : ℝ) : ℝ :=
  cash_register_cost

noncomputable def daily_profit (total_profit : ℝ) (days : ℕ) : ℝ :=
  total_profit / days

noncomputable def daily_profit_after_electricity (daily_profit : ℝ) (electricity_cost : ℝ) : ℝ :=
  daily_profit - electricity_cost

noncomputable def daily_rent (daily_revenue : ℝ) (daily_profit_after_electricity : ℝ) : ℝ :=
  daily_revenue - daily_profit_after_electricity

theorem marie_daily_rent
  (bread_loaves : ℕ) (bread_price : ℝ) (cakes : ℕ) (cake_price : ℝ)
  (days : ℕ) (cash_register_cost : ℝ) (electricity_cost : ℝ) :
  bread_loaves = 40 → bread_price = 2 → cakes = 6 → cake_price = 12 →
  days = 8 → cash_register_cost = 1040 → electricity_cost = 2 →
  daily_rent (daily_revenue bread_loaves bread_price cakes cake_price)
             (daily_profit_after_electricity (daily_profit (total_profit (daily_revenue bread_loaves bread_price cakes cake_price) days cash_register_cost) days) electricity_cost) = 24 :=
by
  intros h0 h1 h2 h3 h4 h5 h6
  sorry

end marie_daily_rent_l2211_221144

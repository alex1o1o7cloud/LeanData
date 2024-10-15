import Mathlib

namespace NUMINAMATH_GPT_range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3_l1451_145157

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.exp x + Real.exp (-x)) + x^2

theorem range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3 :
  {x : ℝ | f (2 * x) > f (x + 3)} = {x : ℝ | x < -1 ∨ x > 3} :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3_l1451_145157


namespace NUMINAMATH_GPT_number_property_l1451_145150

theorem number_property : ∀ n : ℕ, (∀ q : ℕ, q > 0 → n % q^2 < q^(q^2) / 2) ↔ n = 1 ∨ n = 4 :=
by sorry

end NUMINAMATH_GPT_number_property_l1451_145150


namespace NUMINAMATH_GPT_allowance_is_14_l1451_145188

def initial := 11
def spent := 3
def final := 22

def allowance := final - (initial - spent)

theorem allowance_is_14 : allowance = 14 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_allowance_is_14_l1451_145188


namespace NUMINAMATH_GPT_original_price_is_1611_11_l1451_145195

theorem original_price_is_1611_11 (profit: ℝ) (rate: ℝ) (original_price: ℝ) (selling_price: ℝ) 
(h1: profit = 725) (h2: rate = 0.45) (h3: profit = rate * original_price) : 
original_price = 725 / 0.45 := 
sorry

end NUMINAMATH_GPT_original_price_is_1611_11_l1451_145195


namespace NUMINAMATH_GPT_janet_saves_time_l1451_145189

theorem janet_saves_time (looking_time_per_day : ℕ := 8) (complaining_time_per_day : ℕ := 3) (days_per_week : ℕ := 7) :
  (looking_time_per_day + complaining_time_per_day) * days_per_week = 77 := 
sorry

end NUMINAMATH_GPT_janet_saves_time_l1451_145189


namespace NUMINAMATH_GPT_a_seq_correct_b_seq_max_m_l1451_145136

noncomputable def a_seq (n : ℕ) : ℕ :=
if n = 0 then 3 else (n + 1)^2 + 2

-- Verification that the sequence follows the provided conditions.
theorem a_seq_correct (n : ℕ) : 
  (a_seq 0 = 3) ∧
  (a_seq 1 = 6) ∧
  (a_seq 2 = 11) ∧
  (∀ m : ℕ, m ≥ 1 → a_seq (m + 1) - a_seq m = 2 * m + 1) := sorry

noncomputable def b_seq (n : ℕ) : ℝ := 
(a_seq n : ℝ) / (3 ^ (Real.sqrt (a_seq n - 2)))

theorem b_seq_max_m (m : ℝ) : 
  (∀ n : ℕ, b_seq n ≤ m) ↔ (1 ≤ m) := sorry

end NUMINAMATH_GPT_a_seq_correct_b_seq_max_m_l1451_145136


namespace NUMINAMATH_GPT_find_f2_l1451_145159

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f2 (h : ∀ x : ℝ, f x + 3 * f (8 - x) = x) : f 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l1451_145159


namespace NUMINAMATH_GPT_sufficient_condition_for_reciprocal_inequality_l1451_145173

theorem sufficient_condition_for_reciprocal_inequality (a b : ℝ) (h : b < a ∧ a < 0) : (1 / a) < (1 / b) :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_reciprocal_inequality_l1451_145173


namespace NUMINAMATH_GPT_betty_total_cost_l1451_145155

theorem betty_total_cost :
    (6 * 2.5) + (4 * 1.25) + (8 * 3) = 44 :=
by
    sorry

end NUMINAMATH_GPT_betty_total_cost_l1451_145155


namespace NUMINAMATH_GPT_point_in_second_quadrant_l1451_145109

theorem point_in_second_quadrant (a : ℝ) (h1 : 2 * a + 1 < 0) (h2 : 1 - a > 0) : a < -1 / 2 := 
sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l1451_145109


namespace NUMINAMATH_GPT_my_op_evaluation_l1451_145124

def my_op (x y : Int) : Int := x * y - 3 * x + y

theorem my_op_evaluation : my_op 5 3 - my_op 3 5 = -8 := by 
  sorry

end NUMINAMATH_GPT_my_op_evaluation_l1451_145124


namespace NUMINAMATH_GPT_max_sum_x_y_l1451_145144

theorem max_sum_x_y 
  (x y : ℝ)
  (h1 : x^2 + y^2 = 100)
  (h2 : x * y = 40) :
  x + y = 6 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_max_sum_x_y_l1451_145144


namespace NUMINAMATH_GPT_hallie_net_earnings_correct_l1451_145166

noncomputable def hallieNetEarnings : ℚ :=
  let monday_hours := 7
  let monday_rate := 10
  let monday_tips := 18
  let tuesday_hours := 5
  let tuesday_rate := 12
  let tuesday_tips := 12
  let wednesday_hours := 7
  let wednesday_rate := 10
  let wednesday_tips := 20
  let thursday_hours := 8
  let thursday_rate := 11
  let thursday_tips := 25
  let thursday_discount := 0.10
  let friday_hours := 6
  let friday_rate := 9
  let friday_tips := 15
  let income_tax := 0.05

  let monday_earnings := monday_hours * monday_rate
  let tuesday_earnings := tuesday_hours * tuesday_rate
  let wednesday_earnings := wednesday_hours * wednesday_rate
  let thursday_earnings := thursday_hours * thursday_rate
  let thursday_earnings_after_discount := thursday_earnings * (1 - thursday_discount)
  let friday_earnings := friday_hours * friday_rate

  let total_hourly_earnings := monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings + friday_earnings
  let total_tips := monday_tips + tuesday_tips + wednesday_tips + thursday_tips + friday_tips

  let total_tax := total_hourly_earnings * income_tax
  
  let net_earnings := (total_hourly_earnings - total_tax) - (thursday_earnings - thursday_earnings_after_discount) + total_tips
  net_earnings

theorem hallie_net_earnings_correct : hallieNetEarnings = 406.10 := by
  sorry

end NUMINAMATH_GPT_hallie_net_earnings_correct_l1451_145166


namespace NUMINAMATH_GPT_find_t_l1451_145115

def vector := (ℝ × ℝ)

def a : vector := (-3, 4)
def b : vector := (-1, 5)
def c : vector := (2, 3)

def parallel (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem find_t (t : ℝ) : 
  parallel (a.1 - c.1, a.2 - c.2) ((2 * t) + b.1, (3 * t) + b.2) ↔ t = -24 / 17 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l1451_145115


namespace NUMINAMATH_GPT_total_amount_divided_l1451_145138

theorem total_amount_divided (A B C : ℝ) (h1 : A = 2/3 * (B + C)) (h2 : B = 2/3 * (A + C)) (h3 : A = 80) : 
  A + B + C = 200 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_divided_l1451_145138


namespace NUMINAMATH_GPT_evaluate_expression_l1451_145107

theorem evaluate_expression : 3^(2 + 3 + 4) - (3^2 * 3^3 + 3^4) = 19359 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1451_145107


namespace NUMINAMATH_GPT_geometric_sequence_general_term_geometric_sequence_sum_n_l1451_145171

theorem geometric_sequence_general_term (a : ℕ → ℝ) (r : ℝ) (n : ℕ) 
  (h1 : a 3 = 12) (h2 : a 8 = 3 / 8) (h3 : ∀ (n : ℕ), a n = 48 * (1 / 2) ^ (n - 1)) : 
  a n = 48 * (1 / 2) ^ (n - 1) := 
by {
  sorry
}

theorem geometric_sequence_sum_n (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 3 = 12) (h2 : a 8 = 3 / 8) 
  (h3 : ∀ (n : ℕ), a n = 48 * (1 / 2) ^ (n - 1)) 
  (h4 : ∀ (n : ℕ), S n = 48 * (1 - (1 / 2) ^ n) / (1 - 1 / 2))
  (h5 : S 5 = 93) : 
  5 = 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_general_term_geometric_sequence_sum_n_l1451_145171


namespace NUMINAMATH_GPT_simplify_expression_correct_l1451_145141

noncomputable def simplify_expression : ℝ :=
  2 - 2 / (2 + Real.sqrt 5) - 2 / (2 - Real.sqrt 5)

theorem simplify_expression_correct : simplify_expression = 10 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_correct_l1451_145141


namespace NUMINAMATH_GPT_multiple_choice_options_l1451_145137

-- Define the problem conditions
def num_true_false_combinations : ℕ := 14
def num_possible_keys (n : ℕ) : ℕ := num_true_false_combinations * n^2
def total_keys : ℕ := 224

-- The theorem problem
theorem multiple_choice_options : ∃ n : ℕ, num_possible_keys n = total_keys ∧ n = 4 := by
  -- We don't need to provide the proof, so we use sorry. 
  sorry

end NUMINAMATH_GPT_multiple_choice_options_l1451_145137


namespace NUMINAMATH_GPT_spherical_to_rectangular_conversion_l1451_145179

noncomputable def convert_spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  convert_spherical_to_rectangular 8 (5 * Real.pi / 4) (Real.pi / 4) = (-4, -4, 4 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_conversion_l1451_145179


namespace NUMINAMATH_GPT_dice_probability_l1451_145140

def first_die_prob : ℚ := 3 / 8
def second_die_prob : ℚ := 3 / 4
def combined_prob : ℚ := first_die_prob * second_die_prob

theorem dice_probability :
  combined_prob = 9 / 32 :=
by
  -- Here we write the proof steps.
  sorry

end NUMINAMATH_GPT_dice_probability_l1451_145140


namespace NUMINAMATH_GPT_votes_cast_is_330_l1451_145180

variable (T A F : ℝ)

theorem votes_cast_is_330
  (h1 : A = 0.40 * T)
  (h2 : F = A + 66)
  (h3 : T = F + A) :
  T = 330 :=
by
  sorry

end NUMINAMATH_GPT_votes_cast_is_330_l1451_145180


namespace NUMINAMATH_GPT_serving_guests_possible_iff_even_l1451_145125

theorem serving_guests_possible_iff_even (n : ℕ) : 
  (∀ seats : Finset ℕ, ∀ p : ℕ → ℕ, (∀ i : ℕ, i < n → p i ∈ seats) → 
    (∀ i j : ℕ, i < j → p i ≠ p j) → (n % 2 = 0)) = (n % 2 = 0) :=
by sorry

end NUMINAMATH_GPT_serving_guests_possible_iff_even_l1451_145125


namespace NUMINAMATH_GPT_train_length_proper_l1451_145134

noncomputable def train_length (speed distance_time pass_time : ℝ) : ℝ :=
  speed * pass_time

axiom speed_of_train : ∀ (distance_time : ℝ), 
  (10 * 1000 / (15 * 60)) = 11.11

theorem train_length_proper :
  train_length 11.11 900 10 = 111.1 := by
  sorry

end NUMINAMATH_GPT_train_length_proper_l1451_145134


namespace NUMINAMATH_GPT_angle_bisector_relation_l1451_145163

theorem angle_bisector_relation (a b : ℝ) (h : a = -b ∨ a = -b) : a = -b :=
sorry

end NUMINAMATH_GPT_angle_bisector_relation_l1451_145163


namespace NUMINAMATH_GPT_magnitude_a_minus_2b_l1451_145168

noncomputable def magnitude_of_vector_difference : ℝ :=
  let a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
  let b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2)

theorem magnitude_a_minus_2b :
  let a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
  let b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_a_minus_2b_l1451_145168


namespace NUMINAMATH_GPT_isosceles_base_lines_l1451_145117
open Real

theorem isosceles_base_lines {x y : ℝ} (h1 : 7 * x - y - 9 = 0) (h2 : x + y - 7 = 0) (hx : x = 3) (hy : y = -8) :
  (x - 3 * y - 27 = 0) ∨ (3 * x + y - 1 = 0) :=
sorry

end NUMINAMATH_GPT_isosceles_base_lines_l1451_145117


namespace NUMINAMATH_GPT_parabola_focus_distance_l1451_145184

noncomputable def parabolic_distance (x y : ℝ) : ℝ :=
  x + x / 2

theorem parabola_focus_distance : 
  (∃ y : ℝ, (1 : ℝ) = (1 / 2) * y^2) → 
  parabolic_distance 1 y = 3 / 2 :=
by 
  intros hy
  obtain ⟨y, hy⟩ := hy
  unfold parabolic_distance
  have hx : 1 = (1 / 2) * y^2 := hy
  sorry

end NUMINAMATH_GPT_parabola_focus_distance_l1451_145184


namespace NUMINAMATH_GPT_initial_percentage_of_alcohol_l1451_145101

theorem initial_percentage_of_alcohol :
  ∃ P : ℝ, (P / 100 * 11) = (33 / 100 * 14) :=
by
  use 42
  sorry

end NUMINAMATH_GPT_initial_percentage_of_alcohol_l1451_145101


namespace NUMINAMATH_GPT_can_form_triangle_l1451_145114

theorem can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

example : can_form_triangle 8 6 3 := by
  sorry

end NUMINAMATH_GPT_can_form_triangle_l1451_145114


namespace NUMINAMATH_GPT_Sn_minimum_value_l1451_145118

theorem Sn_minimum_value {a : ℕ → ℤ} (n : ℕ) (S : ℕ → ℤ)
  (h1 : a 1 = -11)
  (h2 : a 4 + a 6 = -6)
  (S_def : ∀ n, S n = n * (-12 + n)) :
  ∃ n, S n = S 6 :=
sorry

end NUMINAMATH_GPT_Sn_minimum_value_l1451_145118


namespace NUMINAMATH_GPT_number_of_children_in_group_l1451_145193

-- Definitions based on the conditions
def num_adults : ℕ := 55
def meal_for_adults : ℕ := 70
def meal_for_children : ℕ := 90
def remaining_children_after_adults : ℕ := 81
def num_adults_eaten : ℕ := 7
def ratio_adult_to_child : ℚ := (70 : ℚ) / 90

-- Statement of the problem to prove number of children in the group
theorem number_of_children_in_group : 
  ∃ C : ℕ, 
    (meal_for_adults - num_adults_eaten) * (ratio_adult_to_child) = (remaining_children_after_adults) ∧
    C = remaining_children_after_adults := 
sorry

end NUMINAMATH_GPT_number_of_children_in_group_l1451_145193


namespace NUMINAMATH_GPT_slope_transformation_l1451_145139

theorem slope_transformation :
  ∀ (b : ℝ), ∃ k : ℝ, 
  (∀ x : ℝ, k * x + b = k * (x + 4) + b + 1) → k = -1/4 :=
by
  intros b
  use -1/4
  intros h
  sorry

end NUMINAMATH_GPT_slope_transformation_l1451_145139


namespace NUMINAMATH_GPT_ken_summit_time_l1451_145126

variables (t : ℕ) (s : ℕ)

/--
Sari and Ken climb up a mountain. 
Ken climbs at a constant pace of 500 meters per hour,
and reaches the summit after \( t \) hours starting from 10:00.
Sari starts climbing 2 hours before Ken at 08:00 and is 50 meters behind Ken when he reaches the summit.
Sari is already 700 meters ahead of Ken when he starts climbing.
Prove that Ken reaches the summit at 15:00.
-/
theorem ken_summit_time (h1 : 500 * t = s * (t + 2) + 50)
  (h2 : s * 2 = 700) : t + 10 = 15 :=

sorry

end NUMINAMATH_GPT_ken_summit_time_l1451_145126


namespace NUMINAMATH_GPT_find_result_l1451_145121

theorem find_result : ∀ (x : ℝ), x = 1 / 3 → 5 - 7 * x = 8 / 3 := by
  intros x hx
  sorry

end NUMINAMATH_GPT_find_result_l1451_145121


namespace NUMINAMATH_GPT_even_of_square_even_l1451_145183

theorem even_of_square_even (a : Int) (h1 : ∃ n : Int, a = 2 * n) (h2 : Even (a ^ 2)) : Even a := 
sorry

end NUMINAMATH_GPT_even_of_square_even_l1451_145183


namespace NUMINAMATH_GPT_inequality_chain_l1451_145128

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem inequality_chain (a b : ℝ) (h1 : b > a) (h2 : a > 3) :
  f b < f ((a + b) / 2) ∧ f ((a + b) / 2) < f (Real.sqrt (a * b)) ∧ f (Real.sqrt (a * b)) < f a :=
by
  sorry

end NUMINAMATH_GPT_inequality_chain_l1451_145128


namespace NUMINAMATH_GPT_rowing_speed_in_still_water_l1451_145104

theorem rowing_speed_in_still_water (d t1 t2 : ℝ) 
  (h1 : d = 750) (h2 : t1 = 675) (h3 : t2 = 450) : 
  (d / t1 + (d / t2 - d / t1) / 2) = 1.389 := 
by
  sorry

end NUMINAMATH_GPT_rowing_speed_in_still_water_l1451_145104


namespace NUMINAMATH_GPT_time_of_free_fall_l1451_145113

theorem time_of_free_fall (h : ℝ) (t : ℝ) (height_fall_eq : h = 4.9 * t^2) (initial_height : h = 490) : t = 10 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_time_of_free_fall_l1451_145113


namespace NUMINAMATH_GPT_find_f_2017_l1451_145105

def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom even_f_shifted : ∀ x : ℝ, f (1 - x) = f (x + 1)
axiom f_neg_one : f (-1) = 2

theorem find_f_2017 : f 2017 = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2017_l1451_145105


namespace NUMINAMATH_GPT_arithmetic_sequence_a7_l1451_145160

/--
In an arithmetic sequence {a_n}, it is known that a_1 = 2 and a_3 + a_5 = 10.
Then, we need to prove that a_7 = 8.
-/
theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 2) 
  (h2 : a 3 + a 5 = 10) 
  (h3 : ∀ n, a n = 2 + (n - 1) * d) : 
  a 7 = 8 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a7_l1451_145160


namespace NUMINAMATH_GPT_product_a3_a10_a17_l1451_145116

-- Let's define the problem setup
variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a r : α) (n : ℕ) : α := a * r ^ (n - 1)

theorem product_a3_a10_a17 
  (a r : α)
  (h1 : geometric_sequence a r 2 + geometric_sequence a r 18 = -15) 
  (h2 : geometric_sequence a r 2 * geometric_sequence a r 18 = 16) 
  (ha2pos : geometric_sequence a r 18 ≠ 0) 
  (h3 : r < 0) :
  geometric_sequence a r 3 * geometric_sequence a r 10 * geometric_sequence a r 17 = -64 :=
sorry

end NUMINAMATH_GPT_product_a3_a10_a17_l1451_145116


namespace NUMINAMATH_GPT_probability_black_ball_BoxB_higher_l1451_145167

def boxA_red_balls : ℕ := 40
def boxA_black_balls : ℕ := 10
def boxB_red_balls : ℕ := 60
def boxB_black_balls : ℕ := 40
def boxB_white_balls : ℕ := 50

theorem probability_black_ball_BoxB_higher :
  (boxA_black_balls : ℚ) / (boxA_red_balls + boxA_black_balls) <
  (boxB_black_balls : ℚ) / (boxB_red_balls + boxB_black_balls + boxB_white_balls) :=
by
  sorry

end NUMINAMATH_GPT_probability_black_ball_BoxB_higher_l1451_145167


namespace NUMINAMATH_GPT_intersection_complement_l1451_145148

def U : Set ℤ := Set.univ
def M : Set ℤ := {1, 2}
def P : Set ℤ := {-2, -1, 0, 1, 2}
def CUM : Set ℤ := {x : ℤ | x ∉ M}

theorem intersection_complement :
  P ∩ CUM = {-2, -1, 0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1451_145148


namespace NUMINAMATH_GPT_tan_half_alpha_third_quadrant_sine_cos_expression_l1451_145169

-- Problem (1): Proof for tan(α/2) = -5 given the conditions
theorem tan_half_alpha_third_quadrant (α : ℝ) (h1 : α ∈ Set.Ioo π (3 * π / 2))
  (h2 : Real.sin α = -5/13) :
  Real.tan (α / 2) = -5 := by
  sorry

-- Problem (2): Proof for sin²(π - α) + 2sin(3π/2 + α)cos(π/2 + α) = 8/5 given the condition
theorem sine_cos_expression (α : ℝ) (h : Real.tan α = 2) :
  Real.sin (π - α) ^ 2 + 2 * Real.sin (3 * π / 2 + α) * Real.cos (π / 2 + α) = 8 / 5 := by
  sorry

end NUMINAMATH_GPT_tan_half_alpha_third_quadrant_sine_cos_expression_l1451_145169


namespace NUMINAMATH_GPT_pictures_left_after_deletion_l1451_145100

variable (zoo museum deleted : ℕ)

def total_pictures_taken (zoo museum : ℕ) : ℕ := zoo + museum

def pictures_remaining (total deleted : ℕ) : ℕ := total - deleted

theorem pictures_left_after_deletion (h1 : zoo = 50) (h2 : museum = 8) (h3 : deleted = 38) :
  pictures_remaining (total_pictures_taken zoo museum) deleted = 20 :=
by
  sorry

end NUMINAMATH_GPT_pictures_left_after_deletion_l1451_145100


namespace NUMINAMATH_GPT_moli_bought_7_clips_l1451_145133

theorem moli_bought_7_clips (R C S x : ℝ) 
  (h1 : 3*R + x*C + S = 120) 
  (h2 : 4*R + 10*C + S = 164) 
  (h3 : R + C + S = 32) : 
  x = 7 := 
by
  sorry

end NUMINAMATH_GPT_moli_bought_7_clips_l1451_145133


namespace NUMINAMATH_GPT_div_d_a_value_l1451_145151

variable {a b c d : ℚ}

theorem div_d_a_value (h1 : a / b = 3) (h2 : b / c = 5 / 3) (h3 : c / d = 2) : d / a = 1 / 10 := by
  sorry

end NUMINAMATH_GPT_div_d_a_value_l1451_145151


namespace NUMINAMATH_GPT_trig_identity_tangent_l1451_145106

variable {θ : ℝ}

theorem trig_identity_tangent (h : Real.tan θ = 2) : 
  (Real.sin θ * (Real.cos θ * Real.cos θ - Real.sin θ * Real.sin θ)) / (Real.cos θ - Real.sin θ) = 6 / 5 := 
sorry

end NUMINAMATH_GPT_trig_identity_tangent_l1451_145106


namespace NUMINAMATH_GPT_ratio_of_black_to_white_areas_l1451_145103

theorem ratio_of_black_to_white_areas :
  let π := Real.pi
  let radii := [2, 4, 6, 8]
  let areas := [π * (radii[0])^2, π * (radii[1])^2, π * (radii[2])^2, π * (radii[3])^2]
  let black_areas := [areas[0], areas[2] - areas[1]]
  let white_areas := [areas[1] - areas[0], areas[3] - areas[2]]
  let total_black_area := black_areas.sum
  let total_white_area := white_areas.sum
  let ratio := total_black_area / total_white_area
  ratio = 3 / 5 := sorry

end NUMINAMATH_GPT_ratio_of_black_to_white_areas_l1451_145103


namespace NUMINAMATH_GPT_shaded_region_area_l1451_145164

theorem shaded_region_area (RS : ℝ) (n_shaded : ℕ)
  (h1 : RS = 10) (h2 : n_shaded = 20) :
  (20 * (RS / (2 * Real.sqrt 2))^2) = 250 :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l1451_145164


namespace NUMINAMATH_GPT_sum_composite_l1451_145177

theorem sum_composite (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 34 * a = 43 * b) : ∃ d : ℕ, d > 1 ∧ d < a + b ∧ d ∣ (a + b) :=
by
  sorry

end NUMINAMATH_GPT_sum_composite_l1451_145177


namespace NUMINAMATH_GPT_order_of_x_y_z_l1451_145198

theorem order_of_x_y_z (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  let x : ℝ := (a + b) * (c + d)
  let y : ℝ := (a + c) * (b + d)
  let z : ℝ := (a + d) * (b + c)
  x < y ∧ y < z :=
by
  let x : ℝ := (a + b) * (c + d)
  let y : ℝ := (a + c) * (b + d)
  let z : ℝ := (a + d) * (b + c)
  sorry

end NUMINAMATH_GPT_order_of_x_y_z_l1451_145198


namespace NUMINAMATH_GPT_fraction_subtraction_l1451_145149

theorem fraction_subtraction : 
  (3 + 6 + 9) = 18 ∧ (2 + 5 + 8) = 15 ∧ (2 + 5 + 8) = 15 ∧ (3 + 6 + 9) = 18 →
  (18 / 15 - 15 / 18) = 11 / 30 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l1451_145149


namespace NUMINAMATH_GPT_max_min_z_l1451_145182

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop :=
  x^2 + 4*y^2 = 4*x

-- Define the function z
def z (x y : ℝ) : ℝ :=
  x^2 - y^2

-- Define the required points
def P1 (x y : ℝ) :=
  x = 4 ∧ y = 0

def P2 (x y : ℝ) :=
  x = 2/5 ∧ (y = 3/5 ∨ y = -3/5)

-- Theorem stating the required conditions
theorem max_min_z (x y : ℝ) (h : on_ellipse x y) :
  (P1 x y → z x y = 16) ∧ (P2 x y → z x y = -1/5) :=
by
  sorry

end NUMINAMATH_GPT_max_min_z_l1451_145182


namespace NUMINAMATH_GPT_perpendicular_lines_sum_l1451_145130

theorem perpendicular_lines_sum (a b : ℝ) :
  (∃ (x y : ℝ), 2 * x - 5 * y + b = 0 ∧ a * x + 4 * y - 2 = 0 ∧ x = 1 ∧ y = -2) ∧
  (-a / 4) * (2 / 5) = -1 →
  a + b = -2 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_sum_l1451_145130


namespace NUMINAMATH_GPT_algebraic_identity_specific_case_l1451_145142

theorem algebraic_identity (a b : ℝ) : (a - b)^2 = a^2 + b^2 - 2 * a * b :=
by sorry

theorem specific_case : 2021^2 - 2021 * 4034 + 2017^2 = 16 :=
by sorry

end NUMINAMATH_GPT_algebraic_identity_specific_case_l1451_145142


namespace NUMINAMATH_GPT_russel_carousel_rides_l1451_145127

variable (tickets_used : Nat) (tickets_shooting : Nat) (tickets_carousel : Nat)
variable (total_tickets : Nat)
variable (times_shooting : Nat)

theorem russel_carousel_rides :
    times_shooting = 2 →
    tickets_shooting = 5 →
    tickets_carousel = 3 →
    total_tickets = 19 →
    tickets_used = total_tickets - (times_shooting * tickets_shooting) →
    tickets_used / tickets_carousel = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_russel_carousel_rides_l1451_145127


namespace NUMINAMATH_GPT_justin_run_time_l1451_145196

theorem justin_run_time : 
  let flat_ground_rate := 2 / 2 -- Justin runs 2 blocks in 2 minutes on flat ground
  let uphill_rate := 2 / 3 -- Justin runs 2 blocks in 3 minutes uphill
  let total_blocks := 10 -- Justin is 10 blocks from home
  let uphill_blocks := 6 -- 6 of those blocks are uphill
  let flat_ground_blocks := total_blocks - uphill_blocks -- Remainder are flat ground
  let flat_ground_time := flat_ground_blocks * flat_ground_rate
  let uphill_time := uphill_blocks * uphill_rate
  let total_time := flat_ground_time + uphill_time
  total_time = 13 := 
by 
  sorry

end NUMINAMATH_GPT_justin_run_time_l1451_145196


namespace NUMINAMATH_GPT_parabola_fixed_point_thm_l1451_145161

-- Define the parabola condition
def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

-- Define the focus condition
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Define the slope product condition
def slope_product (A B : ℝ × ℝ) : Prop :=
  (A.1 ≠ 0 ∧ B.1 ≠ 0) → ((A.2 / A.1) * (B.2 / B.1) = -1 / 3)

-- Define the fixed point condition
def fixed_point (A B : ℝ × ℝ) : Prop :=
  ∃ x : ℝ, A ≠ B ∧ (x = 12) ∧ ((A.2 - B.2) / (A.1 - B.1)) * 12 = A.2

-- Problem statement in Lean
theorem parabola_fixed_point_thm (A B : ℝ × ℝ) (p : ℝ) :
  (∃ O : ℝ × ℝ, O = (0, 0)) →
  (∃ C : ℝ → ℝ → ℝ → Prop, C = parabola) →
  (∃ F : ℝ × ℝ, focus F) →
  parabola A.2 A.1 p →
  parabola B.2 B.1 p →
  slope_product A B →
  fixed_point A B :=
by 
-- Sorry is used to skip the proof
sorry

end NUMINAMATH_GPT_parabola_fixed_point_thm_l1451_145161


namespace NUMINAMATH_GPT_dyed_pink_correct_l1451_145181

def silk_dyed_green := 61921
def total_yards_dyed := 111421
def yards_dyed_pink := total_yards_dyed - silk_dyed_green

theorem dyed_pink_correct : yards_dyed_pink = 49500 := by 
  sorry

end NUMINAMATH_GPT_dyed_pink_correct_l1451_145181


namespace NUMINAMATH_GPT_pet_store_cages_l1451_145131

def initial_puppies : ℕ := 18
def puppies_sold : ℕ := 3
def puppies_per_cage : ℕ := 5

def remaining_puppies : ℕ := initial_puppies - puppies_sold
def number_of_cages : ℕ := remaining_puppies / puppies_per_cage

theorem pet_store_cages : number_of_cages = 3 :=
by sorry

end NUMINAMATH_GPT_pet_store_cages_l1451_145131


namespace NUMINAMATH_GPT_sum_of_digits_of_smallest_N_l1451_145119

-- Defining the conditions
def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k
def P (N : ℕ) : ℚ := ((2/3 : ℚ) * N * (1/3 : ℚ) * N) / ((N + 2) * (N + 3))
def S (n : ℕ) : ℕ := (n % 10) + ((n / 10) % 10) + (n / 100)

-- The statement of the problem
theorem sum_of_digits_of_smallest_N :
  ∃ N : ℕ, is_multiple_of_6 N ∧ P N < (4/5 : ℚ) ∧ S N = 6 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_of_smallest_N_l1451_145119


namespace NUMINAMATH_GPT_geometric_seq_sum_S40_l1451_145170

noncomputable def geometric_seq_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q ≠ 1 then a1 * (1 - q^n) / (1 - q) else a1 * n

theorem geometric_seq_sum_S40 :
  ∃ (a1 q : ℝ), (0 < q ∧ q ≠ 1) ∧ 
                geometric_seq_sum a1 q 10 = 10 ∧
                geometric_seq_sum a1 q 30 = 70 ∧
                geometric_seq_sum a1 q 40 = 150 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_sum_S40_l1451_145170


namespace NUMINAMATH_GPT_sequence_form_l1451_145185

-- Defining the sequence a_n as a function f
def seq (f : ℕ → ℕ) : Prop :=
  ∃ c : ℝ, (0 < c) ∧ ∀ m n : ℕ, Nat.gcd (f m + n) (f n + m) > (c * (m + n))

-- Proving that if there exists such a sequence, then it is of the form n + c
theorem sequence_form (f : ℕ → ℕ) (h : seq f) :
  ∃ c : ℤ, ∀ n : ℕ, f n = n + c :=
sorry

end NUMINAMATH_GPT_sequence_form_l1451_145185


namespace NUMINAMATH_GPT_add_to_make_divisible_l1451_145176

theorem add_to_make_divisible :
  ∃ n, n = 34 ∧ ∃ k : ℕ, 758492136547 + n = 51 * k := by
  sorry

end NUMINAMATH_GPT_add_to_make_divisible_l1451_145176


namespace NUMINAMATH_GPT_constant_c_for_local_maximum_l1451_145172

theorem constant_c_for_local_maximum (c : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x * (x - c) ^ 2) (h2 : ∃ δ > 0, ∀ x, |x - 2| < δ → f x ≤ f 2) : c = 6 :=
sorry

end NUMINAMATH_GPT_constant_c_for_local_maximum_l1451_145172


namespace NUMINAMATH_GPT_max_servings_l1451_145129

def servings_prepared (peppers brynza tomatoes cucumbers : ℕ) : ℕ :=
  min (peppers)
      (min (brynza / 75)
           (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings :
  servings_prepared 60 4200 116 117 = 56 :=
by sorry

end NUMINAMATH_GPT_max_servings_l1451_145129


namespace NUMINAMATH_GPT_find_m_value_l1451_145162

theorem find_m_value (m x y : ℝ) (hx : x = 2) (hy : y = -1) (h_eq : m * x - y = 3) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l1451_145162


namespace NUMINAMATH_GPT_tilings_of_3_by_5_rectangle_l1451_145158

def num_tilings_of_3_by_5_rectangle : ℕ := 96

theorem tilings_of_3_by_5_rectangle (h : ℕ := 96) :
  (∃ (tiles : List (ℕ × ℕ)),
    tiles = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)] ∧
    -- Whether we are counting tiles in the context of a 3x5 rectangle
    -- with all distinct rotations and reflections allowed.
    True
  ) → num_tilings_of_3_by_5_rectangle = h :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_tilings_of_3_by_5_rectangle_l1451_145158


namespace NUMINAMATH_GPT_avg_of_14_23_y_is_21_l1451_145143

theorem avg_of_14_23_y_is_21 (y : ℝ) (h : (14 + 23 + y) / 3 = 21) : y = 26 :=
by
  sorry

end NUMINAMATH_GPT_avg_of_14_23_y_is_21_l1451_145143


namespace NUMINAMATH_GPT_sum_of_numbers_in_third_column_is_96_l1451_145154

theorem sum_of_numbers_in_third_column_is_96 :
  ∃ (a : ℕ), (136 = a + 16 * a) ∧ (272 = 2 * a + 32 * a) ∧ (12 * a = 96) :=
by
  let a := 8
  have h1 : 136 = a + 16 * a := by sorry  -- Proof here that 136 = 8 + 16 * 8
  have h2 : 272 = 2 * a + 32 * a := by sorry  -- Proof here that 272 = 2 * 8 + 32 * 8
  have h3 : 12 * a = 96 := by sorry  -- Proof here that 12 * 8 = 96
  existsi a
  exact ⟨h1, h2, h3⟩

end NUMINAMATH_GPT_sum_of_numbers_in_third_column_is_96_l1451_145154


namespace NUMINAMATH_GPT_remainder_expr_div_by_5_l1451_145123

theorem remainder_expr_div_by_5 (n : ℤ) : 
  (7 - 2 * n + (n + 5)) % 5 = (-n + 2) % 5 := 
sorry

end NUMINAMATH_GPT_remainder_expr_div_by_5_l1451_145123


namespace NUMINAMATH_GPT_fraction_geq_81_l1451_145186

theorem fraction_geq_81 {p q r s : ℝ} (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) :
  ((p^2 + p + 1) * (q^2 + q + 1) * (r^2 + r + 1) * (s^2 + s + 1)) / (p * q * r * s) ≥ 81 :=
by
  sorry

end NUMINAMATH_GPT_fraction_geq_81_l1451_145186


namespace NUMINAMATH_GPT_certain_number_is_45_l1451_145122

-- Define the variables and condition
def x : ℝ := 45
axiom h : x * 7 = 0.35 * 900

-- The statement we need to prove
theorem certain_number_is_45 : x = 45 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_45_l1451_145122


namespace NUMINAMATH_GPT_find_big_bonsai_cost_l1451_145102

-- Given definitions based on conditions
def small_bonsai_cost : ℕ := 30
def num_small_bonsai_sold : ℕ := 3
def num_big_bonsai_sold : ℕ := 5
def total_earnings : ℕ := 190

-- Define the function to calculate total earnings from bonsai sales
def calculate_total_earnings (big_bonsai_cost: ℕ) : ℕ :=
  (num_small_bonsai_sold * small_bonsai_cost) + (num_big_bonsai_sold * big_bonsai_cost)

-- The theorem state
theorem find_big_bonsai_cost (B : ℕ) : calculate_total_earnings B = total_earnings → B = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_big_bonsai_cost_l1451_145102


namespace NUMINAMATH_GPT_distance_C_to_C_l1451_145165

noncomputable def C : ℝ × ℝ := (-3, 2)
noncomputable def C' : ℝ × ℝ := (3, -2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_C_to_C' : distance C C' = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_GPT_distance_C_to_C_l1451_145165


namespace NUMINAMATH_GPT_total_cost_correct_l1451_145192

noncomputable def cost_4_canvases : ℕ := 40
noncomputable def cost_paints : ℕ := cost_4_canvases / 2
noncomputable def cost_easel : ℕ := 15
noncomputable def cost_paintbrushes : ℕ := 15
noncomputable def total_cost : ℕ := cost_4_canvases + cost_paints + cost_easel + cost_paintbrushes

theorem total_cost_correct : total_cost = 90 :=
by
  unfold total_cost
  unfold cost_4_canvases
  unfold cost_paints
  unfold cost_easel
  unfold cost_paintbrushes
  simp
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1451_145192


namespace NUMINAMATH_GPT_part_one_part_one_equality_part_two_l1451_145178

-- Given constants and their properties
variables (a b c d : ℝ)

-- Statement for the first problem
theorem part_one : a^6 + b^6 + c^6 + d^6 - 6 * a * b * c * d ≥ -2 :=
sorry

-- Statement for the equality condition in the first problem
theorem part_one_equality (h : |a| = 1 ∧ |b| = 1 ∧ |c| = 1 ∧ |d| = 1) : 
  a^6 + b^6 + c^6 + d^6 - 6 * a * b * c * d = -2 :=
sorry

-- Statement for the second problem (existence of Mk for k >= 4 and odd)
theorem part_two (k : ℕ) (hk1 : 4 ≤ k) (hk2 : k % 2 = 1) : ∃ Mk : ℝ, ∀ a b c d : ℝ, a^k + b^k + c^k + d^k - k * a * b * c * d ≥ Mk :=
sorry

end NUMINAMATH_GPT_part_one_part_one_equality_part_two_l1451_145178


namespace NUMINAMATH_GPT_simplify_and_calculate_expression_l1451_145132

theorem simplify_and_calculate_expression (a b : ℤ) (ha : a = -1) (hb : b = -2) :
  (2 * a + b) * (b - 2 * a) - (a - 3 * b) ^ 2 = -25 :=
by 
  -- We can use 'by' to start the proof and 'sorry' to skip it
  sorry

end NUMINAMATH_GPT_simplify_and_calculate_expression_l1451_145132


namespace NUMINAMATH_GPT_find_beta_l1451_145135

open Real

theorem find_beta 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h1 : sin α = sqrt 5 / 5)
  (h2 : sin (α - β) = - sqrt 10 / 10):
  β = π / 4 :=
sorry

end NUMINAMATH_GPT_find_beta_l1451_145135


namespace NUMINAMATH_GPT_delta_value_l1451_145199

theorem delta_value (Delta : ℤ) (h : 5 * (-3) = Delta - 3) : Delta = -12 := 
by 
  sorry

end NUMINAMATH_GPT_delta_value_l1451_145199


namespace NUMINAMATH_GPT_equalize_costs_l1451_145146

theorem equalize_costs (A B : ℝ) (h_lt : A < B) :
  (B - A) / 2 = (A + B) / 2 - A :=
by sorry

end NUMINAMATH_GPT_equalize_costs_l1451_145146


namespace NUMINAMATH_GPT_find_function_l1451_145111

/-- A function f satisfies the equation f(x) + (x + 1/2) * f(1 - x) = 1. -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x + (x + 1 / 2) * f (1 - x) = 1

/-- We want to prove two things:
 1) f(0) = 2 and f(1) = -2
 2) f(x) =  2 / (1 - 2x) for x ≠ 1/2
 -/
theorem find_function (f : ℝ → ℝ) (h : satisfies_equation f) :
  (f 0 = 2 ∧ f 1 = -2) ∧ (∀ x : ℝ, x ≠ 1 / 2 → f x = 2 / (1 - 2 * x)) ∧ (f (1 / 2) = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_function_l1451_145111


namespace NUMINAMATH_GPT_percentage_sum_of_v_and_w_l1451_145147

variable {x y z v w : ℝ} 

theorem percentage_sum_of_v_and_w (h1 : 0.45 * z = 0.39 * y) (h2 : y = 0.75 * x) 
                                  (h3 : v = 0.80 * z) (h4 : w = 0.60 * y) :
                                  v + w = 0.97 * x :=
by 
  sorry

end NUMINAMATH_GPT_percentage_sum_of_v_and_w_l1451_145147


namespace NUMINAMATH_GPT_total_birds_on_fence_l1451_145190

-- Definitions based on conditions.
def initial_birds : ℕ := 12
def additional_birds : ℕ := 8

-- Theorem corresponding to the problem statement.
theorem total_birds_on_fence : initial_birds + additional_birds = 20 := by 
  sorry

end NUMINAMATH_GPT_total_birds_on_fence_l1451_145190


namespace NUMINAMATH_GPT_min_value_of_expression_l1451_145112

noncomputable def min_expression_value (a b c d : ℝ) : ℝ :=
  (a ^ 8) / ((a ^ 2 + b) * (a ^ 2 + c) * (a ^ 2 + d)) +
  (b ^ 8) / ((b ^ 2 + c) * (b ^ 2 + d) * (b ^ 2 + a)) +
  (c ^ 8) / ((c ^ 2 + d) * (c ^ 2 + a) * (c ^ 2 + b)) +
  (d ^ 8) / ((d ^ 2 + a) * (d ^ 2 + b) * (d ^ 2 + c))

theorem min_value_of_expression (a b c d : ℝ) (h : a + b + c + d = 4) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  min_expression_value a b c d = 1 / 2 :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l1451_145112


namespace NUMINAMATH_GPT_greatest_b_value_l1451_145174

theorem greatest_b_value (b : ℝ) : 
  (-b^3 + b^2 + 7 * b - 10 ≥ 0) ↔ b ≤ 4 + Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_greatest_b_value_l1451_145174


namespace NUMINAMATH_GPT_train_length_l1451_145194

-- Definitions of the conditions as Lean terms/functions
def V (L : ℕ) := (L + 170) / 15
def U (L : ℕ) := (L + 250) / 20

-- The theorem to prove that the length of the train is 70 meters.
theorem train_length : ∃ L : ℕ, (V L = U L) → L = 70 := by
  sorry

end NUMINAMATH_GPT_train_length_l1451_145194


namespace NUMINAMATH_GPT_complex_z_modulus_l1451_145110

noncomputable def i : ℂ := Complex.I

theorem complex_z_modulus (z : ℂ) (h : (1 + i) * z = 2 * i) : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_complex_z_modulus_l1451_145110


namespace NUMINAMATH_GPT_quadratic_one_solution_set_l1451_145197

theorem quadratic_one_solution_set (a : ℝ) :
  (∃ x : ℝ, ax^2 + x + 1 = 0 ∧ (∀ y : ℝ, ax^2 + x + 1 = 0 → y = x)) ↔ (a = 0 ∨ a = 1 / 4) :=
by sorry

end NUMINAMATH_GPT_quadratic_one_solution_set_l1451_145197


namespace NUMINAMATH_GPT_number_of_girls_in_class_l1451_145156

variable (B S G : ℕ)

theorem number_of_girls_in_class
  (h1 : (3 / 4 : ℚ) * B = 18)
  (h2 : B = (2 / 3 : ℚ) * S) :
  G = S - B → G = 12 := by
  intro hg
  sorry

end NUMINAMATH_GPT_number_of_girls_in_class_l1451_145156


namespace NUMINAMATH_GPT_triangle_area_l1451_145152

noncomputable def area_triangle_ACD (t p : ℝ) : ℝ :=
  1 / 2 * p * (t - 2)

theorem triangle_area (t p : ℝ) (ht : 0 < t ∧ t < 12) (hp : 0 < p ∧ p < 12) :
  area_triangle_ACD t p = 1 / 2 * p * (t - 2) :=
sorry

end NUMINAMATH_GPT_triangle_area_l1451_145152


namespace NUMINAMATH_GPT_sum_of_squares_l1451_145108

theorem sum_of_squares (a b c : ℝ) (h_arith : a + b + c = 30) (h_geom : a * b * c = 216) 
(h_harm : 1/a + 1/b + 1/c = 3/4) : a^2 + b^2 + c^2 = 576 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1451_145108


namespace NUMINAMATH_GPT_days_to_clear_land_l1451_145191

-- Definitions of all the conditions
def length_of_land := 200
def width_of_land := 900
def area_cleared_by_one_rabbit_per_day_square_yards := 10
def number_of_rabbits := 100
def conversion_square_yards_to_square_feet := 9
def total_area_of_land := length_of_land * width_of_land
def area_cleared_by_one_rabbit_per_day_square_feet := area_cleared_by_one_rabbit_per_day_square_yards * conversion_square_yards_to_square_feet
def area_cleared_by_all_rabbits_per_day := number_of_rabbits * area_cleared_by_one_rabbit_per_day_square_feet

-- Theorem to prove the number of days required to clear the land
theorem days_to_clear_land :
  total_area_of_land / area_cleared_by_all_rabbits_per_day = 20 := by
  sorry

end NUMINAMATH_GPT_days_to_clear_land_l1451_145191


namespace NUMINAMATH_GPT_probability_volleyball_is_one_third_l1451_145120

-- Define the total number of test items
def total_test_items : ℕ := 3

-- Define the number of favorable outcomes for hitting the wall with a volleyball
def favorable_outcomes_volleyball : ℕ := 1

-- Define the probability calculation
def probability_hitting_wall_with_volleyball : ℚ :=
  favorable_outcomes_volleyball / total_test_items

-- Prove the probability is 1/3
theorem probability_volleyball_is_one_third :
  probability_hitting_wall_with_volleyball = 1 / 3 := 
sorry

end NUMINAMATH_GPT_probability_volleyball_is_one_third_l1451_145120


namespace NUMINAMATH_GPT_average_age_before_new_students_l1451_145175

theorem average_age_before_new_students
  (A : ℝ) (N : ℕ)
  (h1 : N = 15)
  (h2 : 15 * 32 + N * A = (N + 15) * (A - 4)) :
  A = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_age_before_new_students_l1451_145175


namespace NUMINAMATH_GPT_ellipse_range_of_k_l1451_145145

theorem ellipse_range_of_k (k : ℝ) :
  (1 - k > 0) ∧ (1 + k > 0) ∧ (1 - k ≠ 1 + k) ↔ (-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_range_of_k_l1451_145145


namespace NUMINAMATH_GPT_proj_v_onto_w_l1451_145187

open Real

noncomputable def v : ℝ × ℝ := (8, -4)
noncomputable def w : ℝ × ℝ := (2, 3)

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let coeff := dot_product v w / dot_product w w
  (coeff * w.1, coeff * w.2)

theorem proj_v_onto_w :
  projection v w = (8 / 13, 12 / 13) :=
by
  sorry

end NUMINAMATH_GPT_proj_v_onto_w_l1451_145187


namespace NUMINAMATH_GPT_pedro_furniture_area_l1451_145153

theorem pedro_furniture_area :
  let width : ℝ := 2
  let length : ℝ := 2.5
  let door_arc_area := (1 / 4) * Real.pi * (0.5 ^ 2)
  let window_arc_area := 2 * (1 / 2) * Real.pi * (0.5 ^ 2)
  let room_area := width * length
  room_area - door_arc_area - window_arc_area = (80 - 9 * Real.pi) / 16 := 
by
  sorry

end NUMINAMATH_GPT_pedro_furniture_area_l1451_145153

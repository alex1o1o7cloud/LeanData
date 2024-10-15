import Mathlib

namespace NUMINAMATH_GPT_teacher_zhang_friends_l282_28258

-- Define the conditions
def num_students : ℕ := 50
def both_friends : ℕ := 30
def neither_friend : ℕ := 1
def diff_in_friends : ℕ := 7

-- Prove that Teacher Zhang has 43 friends on social media
theorem teacher_zhang_friends : ∃ x : ℕ, 
  x + (x - diff_in_friends) - both_friends + neither_friend = num_students ∧ x = 43 := 
by
  sorry

end NUMINAMATH_GPT_teacher_zhang_friends_l282_28258


namespace NUMINAMATH_GPT_magnitude_of_z_l282_28224

open Complex

theorem magnitude_of_z :
  ∃ z : ℂ, (1 + 2 * Complex.I) * z = -1 + 3 * Complex.I ∧ Complex.abs z = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_of_z_l282_28224


namespace NUMINAMATH_GPT_total_marbles_proof_l282_28248

def dan_violet_marbles : Nat := 64
def mary_red_marbles : Nat := 14
def john_blue_marbles (x : Nat) : Nat := x

def total_marble (x : Nat) : Nat := dan_violet_marbles + mary_red_marbles + john_blue_marbles x

theorem total_marbles_proof (x : Nat) : total_marble x = 78 + x := by
  sorry

end NUMINAMATH_GPT_total_marbles_proof_l282_28248


namespace NUMINAMATH_GPT_modulus_of_2_plus_i_over_1_plus_2i_l282_28266

open Complex

noncomputable def modulus_of_complex_fraction : ℂ := 
  let z : ℂ := (2 + I) / (1 + 2 * I)
  abs z

theorem modulus_of_2_plus_i_over_1_plus_2i :
  modulus_of_complex_fraction = 1 := by
  sorry

end NUMINAMATH_GPT_modulus_of_2_plus_i_over_1_plus_2i_l282_28266


namespace NUMINAMATH_GPT_triploid_fruit_fly_chromosome_periodicity_l282_28250

-- Define the conditions
def normal_chromosome_count (organism: Type) : ℕ := 8
def triploid_fruit_fly (organism: Type) : Prop := true
def XXY_sex_chromosome_composition (organism: Type) : Prop := true
def periodic_change (counts: List ℕ) : Prop := counts = [9, 18, 9]

-- State the theorem
theorem triploid_fruit_fly_chromosome_periodicity (organism: Type)
  (h1: triploid_fruit_fly organism) 
  (h2: XXY_sex_chromosome_composition organism)
  (h3: normal_chromosome_count organism = 8) : 
  periodic_change [9, 18, 9] :=
sorry

end NUMINAMATH_GPT_triploid_fruit_fly_chromosome_periodicity_l282_28250


namespace NUMINAMATH_GPT_polynomial_coefficient_sum_l282_28233

theorem polynomial_coefficient_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (2 * x - 3) ^ 5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_coefficient_sum_l282_28233


namespace NUMINAMATH_GPT_distance_light_travels_500_years_l282_28292

-- Define the given conditions
def distance_in_one_year_miles : ℝ := 5.87e12
def years_traveling : ℝ := 500
def miles_to_kilometers : ℝ := 1.60934

-- Define the expected distance in kilometers after 500 years
def expected_distance_in_kilometers : ℝ  := 4.723e15

-- State the theorem: the distance light travels in 500 years in kilometers
theorem distance_light_travels_500_years :
  (distance_in_one_year_miles * years_traveling * miles_to_kilometers) 
    = expected_distance_in_kilometers := 
by
  sorry

end NUMINAMATH_GPT_distance_light_travels_500_years_l282_28292


namespace NUMINAMATH_GPT_alice_is_10_years_older_l282_28256

-- Problem definitions
variables (A B : ℕ)

-- Conditions of the problem
def condition1 := A + 5 = 19
def condition2 := A + 6 = 2 * (B + 6)

-- Question to prove
theorem alice_is_10_years_older (h1 : condition1 A) (h2 : condition2 A B) : A - B = 10 := 
by
  sorry

end NUMINAMATH_GPT_alice_is_10_years_older_l282_28256


namespace NUMINAMATH_GPT_largest_perimeter_l282_28208

-- Define the problem's conditions
def side1 := 7
def side2 := 9
def integer_side (x : ℕ) : Prop := (x > 2) ∧ (x < 16)

-- Define the perimeter calculation
def perimeter (a b c : ℕ) := a + b + c

-- The theorem statement which we want to prove
theorem largest_perimeter : ∃ x : ℕ, integer_side x ∧ perimeter side1 side2 x = 31 :=
by
  sorry

end NUMINAMATH_GPT_largest_perimeter_l282_28208


namespace NUMINAMATH_GPT_marie_eggs_total_l282_28217

variable (x : ℕ) -- Number of eggs in each box

-- Conditions as definitions
def egg_weight := 10 -- weight of each egg in ounces
def total_boxes := 4 -- total number of boxes
def remaining_boxes := 3 -- boxes left after one is discarded
def remaining_weight := 90 -- total weight of remaining eggs in ounces

-- Proof statement
theorem marie_eggs_total : remaining_boxes * egg_weight * x = remaining_weight → total_boxes * x = 12 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_marie_eggs_total_l282_28217


namespace NUMINAMATH_GPT_quadratic_intersection_y_axis_l282_28297

theorem quadratic_intersection_y_axis :
  (∃ y, y = 3 * (0: ℝ)^2 - 4 * (0: ℝ) + 5 ∧ (0, y) = (0, 5)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_intersection_y_axis_l282_28297


namespace NUMINAMATH_GPT_bacteria_growth_time_l282_28210

theorem bacteria_growth_time : 
  (∀ n : ℕ, 2 ^ n = 4096 → (n * 15) / 60 = 3) :=
by
  sorry

end NUMINAMATH_GPT_bacteria_growth_time_l282_28210


namespace NUMINAMATH_GPT_distance_to_second_picture_edge_l282_28212

/-- Given a wall of width 25 feet, with a first picture 5 feet wide centered on the wall,
and a second picture 3 feet wide centered in the remaining space, the distance 
from the nearest edge of the second picture to the end of the wall is 13.5 feet. -/
theorem distance_to_second_picture_edge :
  let wall_width := 25
  let first_picture_width := 5
  let second_picture_width := 3
  let side_space := (wall_width - first_picture_width) / 2
  let remaining_space := side_space
  let second_picture_side_space := (remaining_space - second_picture_width) / 2
  10 + 3.5 = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_second_picture_edge_l282_28212


namespace NUMINAMATH_GPT_max_value_of_f_l282_28226

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_of_f : ∀ x : ℝ, x > 0 → f x ≤ (Real.log (Real.exp 1)) / (Real.exp 1) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l282_28226


namespace NUMINAMATH_GPT_face_value_of_share_l282_28298

theorem face_value_of_share (FV : ℝ) (dividend_percent : ℝ) (interest_percent : ℝ) (market_value : ℝ) :
  dividend_percent = 0.09 → 
  interest_percent = 0.12 →
  market_value = 33 →
  (0.09 * FV = 0.12 * 33) → FV = 44 :=
by
  intros
  sorry

end NUMINAMATH_GPT_face_value_of_share_l282_28298


namespace NUMINAMATH_GPT_xiaotian_sep_usage_plan_cost_effectiveness_l282_28213

noncomputable def problem₁ (units : List Int) : Real :=
  units.sum / 1024 + 5 * 6

theorem xiaotian_sep_usage (units : List Int) (h : units = [200, -100, 100, -100, 212, 200]) :
  problem₁ units = 30.5 :=
sorry

def plan_cost_a (x : Int) : Real := 5 * x + 4

def plan_cost_b (x : Int) : Real :=
  if h : 20 < x ∧ x <= 23 then 5 * x - 1
  else 3 * x + 45

theorem plan_cost_effectiveness (x : Int) (h : x > 23) :
  plan_cost_a x > plan_cost_b x :=
sorry

end NUMINAMATH_GPT_xiaotian_sep_usage_plan_cost_effectiveness_l282_28213


namespace NUMINAMATH_GPT_arithmetic_seq_8th_term_l282_28275

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_seq_8th_term_l282_28275


namespace NUMINAMATH_GPT_area_enclosed_by_curves_l282_28203

theorem area_enclosed_by_curves (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, (x + a * y)^2 = 16 * a^2) ∧ (∀ x y : ℝ, (a * x - y)^2 = 4 * a^2) →
  ∃ A : ℝ, A = 32 * a^2 / (1 + a^2) :=
by
  sorry

end NUMINAMATH_GPT_area_enclosed_by_curves_l282_28203


namespace NUMINAMATH_GPT_no_five_consecutive_terms_divisible_by_2005_l282_28230

noncomputable def a (n : ℕ) : ℤ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_terms_divisible_by_2005 : ¬ ∃ n : ℕ, (a n % 2005 = 0) ∧ (a (n+1) % 2005 = 0) ∧ (a (n+2) % 2005 = 0) ∧ (a (n+3) % 2005 = 0) ∧ (a (n+4) % 2005 = 0) := sorry

end NUMINAMATH_GPT_no_five_consecutive_terms_divisible_by_2005_l282_28230


namespace NUMINAMATH_GPT_initial_number_l282_28263

theorem initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_l282_28263


namespace NUMINAMATH_GPT_buffet_dishes_l282_28290

-- To facilitate the whole proof context, but skipping proof parts with 'sorry'

-- Oliver will eat if there is no mango in the dishes

variables (D : ℕ) -- Total number of dishes

-- Conditions:
variables (h1 : 3 <= D) -- there are at least 3 dishes with mango salsa
variables (h2 : 1 ≤ D / 6) -- one-sixth of dishes have fresh mango
variables (h3 : 1 ≤ D) -- there's at least one dish with mango jelly
variables (h4 : D / 6 ≥ 2) -- Oliver can pick out the mangoes from 2 of dishes with fresh mango
variables (h5 : D - (3 + (D / 6 - 2) + 1) = 28) -- there are 28 dishes Oliver can eat

theorem buffet_dishes : D = 36 :=
by
  sorry -- Skip the actual proof

end NUMINAMATH_GPT_buffet_dishes_l282_28290


namespace NUMINAMATH_GPT_exchange_rate_l282_28267

theorem exchange_rate (a b : ℕ) (h : 5000 = 60 * a) : b = 75 * a → b = 6250 := by
  sorry

end NUMINAMATH_GPT_exchange_rate_l282_28267


namespace NUMINAMATH_GPT_continuous_linear_function_l282_28247

theorem continuous_linear_function {f : ℝ → ℝ} (h_cont : Continuous f) 
  (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_a_half : a < 1/2) (h_b_half : b < 1/2) 
  (h_eq : ∀ x : ℝ, f (f x) = a * f x + b * x) : 
  ∃ k : ℝ, (∀ x : ℝ, f x = k * x) ∧ (k * k - a * k - b = 0) := 
sorry

end NUMINAMATH_GPT_continuous_linear_function_l282_28247


namespace NUMINAMATH_GPT_range_of_x_l282_28215

theorem range_of_x (x : ℝ) (h : ∃ y : ℝ, y = (x - 3) ∧ y > 0) : x > 3 :=
sorry

end NUMINAMATH_GPT_range_of_x_l282_28215


namespace NUMINAMATH_GPT_derivative_at_0_l282_28216

noncomputable def f (x : ℝ) := Real.exp x / (x + 2)

theorem derivative_at_0 : deriv f 0 = 1 / 4 := sorry

end NUMINAMATH_GPT_derivative_at_0_l282_28216


namespace NUMINAMATH_GPT_sheela_monthly_income_l282_28201

variable (deposits : ℝ) (percentage : ℝ) (monthly_income : ℝ)

-- Conditions
axiom deposit_condition : deposits = 3400
axiom percentage_condition : percentage = 0.15
axiom income_condition : deposits = percentage * monthly_income

-- Proof goal
theorem sheela_monthly_income :
  monthly_income = 3400 / 0.15 :=
sorry

end NUMINAMATH_GPT_sheela_monthly_income_l282_28201


namespace NUMINAMATH_GPT_sum_first_n_terms_arithmetic_sequence_l282_28223

theorem sum_first_n_terms_arithmetic_sequence 
  (S : ℕ → ℕ) (m : ℕ) (h1 : S m = 2) (h2 : S (2 * m) = 10) :
  S (3 * m) = 24 :=
sorry

end NUMINAMATH_GPT_sum_first_n_terms_arithmetic_sequence_l282_28223


namespace NUMINAMATH_GPT_orange_harvest_exists_l282_28282

theorem orange_harvest_exists :
  ∃ (A B C D : ℕ), A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ A + B + C + D = 56 :=
by
  use 10
  use 15
  use 16
  use 15
  repeat {split};
  sorry

end NUMINAMATH_GPT_orange_harvest_exists_l282_28282


namespace NUMINAMATH_GPT_Sean_Julie_ratio_l282_28220

-- Define the sum of the first n natural numbers
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of even numbers up to 2n
def sum_even (n : ℕ) : ℕ := 2 * sum_n n

theorem Sean_Julie_ratio : 
  (sum_even 250) / (sum_n 250) = 2 := 
by
  sorry

end NUMINAMATH_GPT_Sean_Julie_ratio_l282_28220


namespace NUMINAMATH_GPT_pow_mod_sub_l282_28273

theorem pow_mod_sub (a b : ℕ) (n : ℕ) (h1 : a ≡ 5 [MOD 6]) (h2 : b ≡ 4 [MOD 6]) : (a^n - b^n) % 6 = 1 :=
by
  let a := 47
  let b := 22
  let n := 1987
  sorry

end NUMINAMATH_GPT_pow_mod_sub_l282_28273


namespace NUMINAMATH_GPT_total_fish_count_l282_28270

theorem total_fish_count (kyle_caught_same_as_tasha : ∀ kyle tasha : ℕ, kyle = tasha) 
  (carla_caught : ℕ) (kyle_caught : ℕ) (tasha_caught : ℕ)
  (h0 : carla_caught = 8) (h1 : kyle_caught = 14) (h2 : tasha_caught = kyle_caught) : 
  8 + 14 + 14 = 36 :=
by sorry

end NUMINAMATH_GPT_total_fish_count_l282_28270


namespace NUMINAMATH_GPT_number_of_divisions_l282_28234

-- Definitions
def hour_in_seconds : ℕ := 3600

def is_division (n m : ℕ) : Prop :=
  n * m = hour_in_seconds ∧ n > 0 ∧ m > 0

-- Proof problem statement
theorem number_of_divisions : ∃ (count : ℕ), count = 44 ∧ 
  (∀ (n m : ℕ), is_division n m → ∃ (d : ℕ), d = count) :=
sorry

end NUMINAMATH_GPT_number_of_divisions_l282_28234


namespace NUMINAMATH_GPT_range_of_a_l282_28239

open Real

-- Definitions based on given conditions
def p (a : ℝ) : Prop := a > 2
def q (a : ℝ) : Prop := ∀ (x : ℝ), x > 0 → -3^x ≤ a

-- The main proposition combining the conditions
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l282_28239


namespace NUMINAMATH_GPT_find_k_l282_28276

theorem find_k (k : ℝ) :
  (∃ x : ℝ, 8 * x - k = 2 * (x + 1) ∧ 2 * (2 * x - 3) = 1 - 3 * x) → k = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l282_28276


namespace NUMINAMATH_GPT_negation_proposition_equivalence_l282_28283

theorem negation_proposition_equivalence :
  (¬ ∃ x₀ : ℝ, (2 / x₀ + Real.log x₀ ≤ 0)) ↔ (∀ x : ℝ, 2 / x + Real.log x > 0) := 
sorry

end NUMINAMATH_GPT_negation_proposition_equivalence_l282_28283


namespace NUMINAMATH_GPT_ornamental_rings_remaining_l282_28296

-- Definitions based on conditions
variable (initial_stock : ℕ) (final_stock : ℕ)

-- Condition 1
def condition1 := initial_stock + 200 = 3 * initial_stock

-- Condition 2
def condition2 := final_stock = (200 + initial_stock) * 1 / 4 - (200 + initial_stock) / 4 + 300 - 150

-- Theorem statement to prove the final stock is 225
theorem ornamental_rings_remaining
  (h1 : condition1 initial_stock)
  (h2 : condition2 initial_stock final_stock) :
  final_stock = 225 :=
sorry

end NUMINAMATH_GPT_ornamental_rings_remaining_l282_28296


namespace NUMINAMATH_GPT_investment_worth_l282_28244

theorem investment_worth {x : ℝ} (x_pos : 0 < x) :
  ∀ (initial_investment final_value : ℝ) (years : ℕ),
  (initial_investment * 3^years = final_value) → 
  initial_investment = 1500 → final_value = 13500 → 
  8 = x → years = 2 →
  years * (112 / x) = 28 := 
by
  sorry

end NUMINAMATH_GPT_investment_worth_l282_28244


namespace NUMINAMATH_GPT_pyramid_height_l282_28200

noncomputable def height_of_pyramid (h : ℝ) : Prop :=
  let cube_edge_length := 6
  let pyramid_base_edge_length := 12
  let V_cube := cube_edge_length ^ 3
  let V_pyramid := (1 / 3) * (pyramid_base_edge_length ^ 2) * h
  V_cube = V_pyramid → h = 4.5

theorem pyramid_height : height_of_pyramid 4.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_pyramid_height_l282_28200


namespace NUMINAMATH_GPT_number_of_girls_l282_28221

theorem number_of_girls (total_students : ℕ) (sample_size : ℕ) (girls_sampled_minus : ℕ) (girls_sampled_ratio : ℚ) :
  total_students = 1600 →
  sample_size = 200 →
  girls_sampled_minus = 20 →
  girls_sampled_ratio = 90 / 200 →
  (∃ x, x / (total_students : ℚ) = girls_sampled_ratio ∧ x = 720) :=
by intros _ _ _ _; sorry

end NUMINAMATH_GPT_number_of_girls_l282_28221


namespace NUMINAMATH_GPT_find_x_l282_28236

theorem find_x (x : ℝ) (h : (2 * x + 8 + 5 * x + 3 + 3 * x + 9) / 3 = 3 * x + 2) : x = -14 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l282_28236


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l282_28227

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ)
  (h_geom : ∃ q, ∀ n, a (n+1) = a n * q)
  (h1 : a 1 = 1 / 8)
  (h4 : a 4 = -1) :
  ∃ q, q = -2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l282_28227


namespace NUMINAMATH_GPT_average_weight_of_whole_class_l282_28288

theorem average_weight_of_whole_class :
  let students_A := 50
  let students_B := 50
  let avg_weight_A := 60
  let avg_weight_B := 80
  let total_students := students_A + students_B
  let total_weight_A := students_A * avg_weight_A
  let total_weight_B := students_B * avg_weight_B
  let total_weight := total_weight_A + total_weight_B
  let avg_weight := total_weight / total_students
  avg_weight = 70 := 
by 
  sorry

end NUMINAMATH_GPT_average_weight_of_whole_class_l282_28288


namespace NUMINAMATH_GPT_number_of_self_inverse_subsets_is_15_l282_28272

-- Define the set M
def M : Set ℚ := ({-1, 0, 1/2, 1/3, 1, 2, 3, 4} : Set ℚ)

-- Definition of self-inverse set
def is_self_inverse (A : Set ℚ) : Prop := ∀ x ∈ A, 1/x ∈ A

-- Theorem stating the number of non-empty self-inverse subsets of M
theorem number_of_self_inverse_subsets_is_15 :
  (∃ S : Finset (Set ℚ), S.card = 15 ∧ ∀ A ∈ S, A ⊆ M ∧ is_self_inverse A) :=
sorry

end NUMINAMATH_GPT_number_of_self_inverse_subsets_is_15_l282_28272


namespace NUMINAMATH_GPT_unique_perpendicular_line_through_point_l282_28279

variables (a b : ℝ → ℝ) (P : ℝ)

def are_skew_lines (a b : ℝ → ℝ) : Prop :=
  ¬∃ (t₁ t₂ : ℝ), a t₁ = b t₂

def is_point_not_on_lines (P : ℝ) (a b : ℝ → ℝ) : Prop :=
  ∀ (t : ℝ), P ≠ a t ∧ P ≠ b t

theorem unique_perpendicular_line_through_point (ha : are_skew_lines a b) (hp : is_point_not_on_lines P a b) :
  ∃! (L : ℝ → ℝ), (∀ (t : ℝ), L t ≠ P) ∧ (∀ (L' : ℝ → ℝ), (∀ (t : ℝ), L' t ≠ P) → L' = L) := sorry

end NUMINAMATH_GPT_unique_perpendicular_line_through_point_l282_28279


namespace NUMINAMATH_GPT_rectangle_perimeter_gt_16_l282_28228

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_gt_16_l282_28228


namespace NUMINAMATH_GPT_remainder_of_8_pow_2023_l282_28249

theorem remainder_of_8_pow_2023 :
  8^2023 % 100 = 12 :=
sorry

end NUMINAMATH_GPT_remainder_of_8_pow_2023_l282_28249


namespace NUMINAMATH_GPT_gold_stickers_for_second_student_l282_28214

theorem gold_stickers_for_second_student :
  (exists f : ℕ → ℕ,
      f 1 = 29 ∧
      f 3 = 41 ∧
      f 4 = 47 ∧
      f 5 = 53 ∧
      f 6 = 59 ∧
      (∀ n, f (n + 1) - f n = 6 ∨ f (n + 2) - f n = 12)) →
  (∃ f : ℕ → ℕ, f 2 = 35) :=
by
  sorry

end NUMINAMATH_GPT_gold_stickers_for_second_student_l282_28214


namespace NUMINAMATH_GPT_find_x_plus_inv_x_l282_28268

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + (1/x)^3 = 110) : x + (1/x) = 5 :=
sorry

end NUMINAMATH_GPT_find_x_plus_inv_x_l282_28268


namespace NUMINAMATH_GPT_equality_of_fractions_l282_28274

theorem equality_of_fractions
  (a b c x y z : ℝ)
  (h1 : a = b * z + c * y)
  (h2 : b = c * x + a * z)
  (h3 : c = a * y + b * x)
  (hx : x ≠ 1 ∧ x ≠ -1)
  (hy : y ≠ 1 ∧ y ≠ -1)
  (hz : z ≠ 1 ∧ z ≠ -1) :
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  (a^2) / (1 - x^2) = (b^2) / (1 - y^2) ∧ (b^2) / (1 - y^2) = (c^2) / (1 - z^2) :=
by
  sorry

end NUMINAMATH_GPT_equality_of_fractions_l282_28274


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l282_28237

theorem sufficient_but_not_necessary (x y : ℝ) (h : x ≥ 1 ∧ y ≥ 1) : x ^ 2 + y ^ 2 ≥ 2 ∧ ∃ (x y : ℝ), x ^ 2 + y ^ 2 ≥ 2 ∧ (¬ (x ≥ 1 ∧ y ≥ 1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l282_28237


namespace NUMINAMATH_GPT_find_pure_imaginary_solutions_l282_28294

noncomputable def poly_eq_zero (x : ℂ) : Prop :=
  x^4 - 6 * x^3 + 13 * x^2 - 42 * x - 72 = 0

noncomputable def is_imaginary (x : ℂ) : Prop :=
  x.im ≠ 0 ∧ x.re = 0

theorem find_pure_imaginary_solutions :
  ∀ x : ℂ, poly_eq_zero x ∧ is_imaginary x ↔ (x = Complex.I * Real.sqrt 7 ∨ x = -Complex.I * Real.sqrt 7) :=
by sorry

end NUMINAMATH_GPT_find_pure_imaginary_solutions_l282_28294


namespace NUMINAMATH_GPT_man_speed_in_still_water_l282_28241

theorem man_speed_in_still_water (upstream_speed downstream_speed : ℝ) (h1 : upstream_speed = 25) (h2 : downstream_speed = 45) :
  (upstream_speed + downstream_speed) / 2 = 35 :=
by
  sorry

end NUMINAMATH_GPT_man_speed_in_still_water_l282_28241


namespace NUMINAMATH_GPT_square_AP_square_equals_2000_l282_28287

noncomputable def square_side : ℝ := 100
noncomputable def midpoint_AB : ℝ := square_side / 2
noncomputable def distance_MP : ℝ := 50
noncomputable def distance_PC : ℝ := square_side

/-- Given a square ABCD with side length 100, midpoint M of AB, MP = 50, and PC = 100, prove AP^2 = 2000 -/
theorem square_AP_square_equals_2000 :
  ∃ (P : ℝ × ℝ), (dist (P.1, P.2) (midpoint_AB, 0) = distance_MP) ∧ (dist (P.1, P.2) (square_side, square_side) = distance_PC) ∧ ((P.1) ^ 2 + (P.2) ^ 2 = 2000) := 
sorry


end NUMINAMATH_GPT_square_AP_square_equals_2000_l282_28287


namespace NUMINAMATH_GPT_xy_product_l282_28260

theorem xy_product (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 44) : x * y = -24 := 
by {
  sorry
}

end NUMINAMATH_GPT_xy_product_l282_28260


namespace NUMINAMATH_GPT_sum_first_five_terms_geometric_sequence_l282_28281

noncomputable def sum_first_five_geometric (a0 : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a0 * (1 - r^n) / (1 - r)

theorem sum_first_five_terms_geometric_sequence : 
  sum_first_five_geometric (1/3) (1/3) 5 = 121 / 243 := 
by 
  sorry

end NUMINAMATH_GPT_sum_first_five_terms_geometric_sequence_l282_28281


namespace NUMINAMATH_GPT_range_of_m_decreasing_l282_28219

theorem range_of_m_decreasing (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (m - 3) * x₁ + 5 > (m - 3) * x₂ + 5) ↔ m < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_decreasing_l282_28219


namespace NUMINAMATH_GPT_probability_meeting_part_a_l282_28269

theorem probability_meeting_part_a :
  ∃ p : ℝ, p = (11 : ℝ) / 36 :=
sorry

end NUMINAMATH_GPT_probability_meeting_part_a_l282_28269


namespace NUMINAMATH_GPT_increasing_interval_f_l282_28242

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval_f :
  ∀ x, (2 < x) → (∃ ε > 0, ∀ δ > 0, δ < ε → f (x + δ) ≥ f x) :=
by
  sorry

end NUMINAMATH_GPT_increasing_interval_f_l282_28242


namespace NUMINAMATH_GPT_gcd_m_n_is_one_l282_28299

def m : ℕ := 122^2 + 234^2 + 344^2

def n : ℕ := 123^2 + 235^2 + 343^2

theorem gcd_m_n_is_one : Nat.gcd m n = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_m_n_is_one_l282_28299


namespace NUMINAMATH_GPT_prove_square_ratio_l282_28251
noncomputable section

-- Definitions from given conditions
variables (a b : ℝ) (d : ℝ := Real.sqrt (a^2 + b^2))

-- Condition from the problem
def ratio_condition : Prop := a / b = (a + 2 * b) / d

-- The theorem we need to prove
theorem prove_square_ratio (h : ratio_condition a b d) : 
  ∃ k : ℝ, k = a / b ∧ k^4 - 3*k^2 - 4*k - 4 = 0 := 
by
  sorry

end NUMINAMATH_GPT_prove_square_ratio_l282_28251


namespace NUMINAMATH_GPT_probability_of_blue_or_orange_jelly_bean_is_5_over_13_l282_28218

def total_jelly_beans : ℕ := 7 + 9 + 8 + 10 + 5

def blue_or_orange_jelly_beans : ℕ := 10 + 5

def probability_blue_or_orange : ℚ := blue_or_orange_jelly_beans / total_jelly_beans

theorem probability_of_blue_or_orange_jelly_bean_is_5_over_13 :
  probability_blue_or_orange = 5 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_blue_or_orange_jelly_bean_is_5_over_13_l282_28218


namespace NUMINAMATH_GPT_angle_sum_triangle_l282_28284

theorem angle_sum_triangle (A B C : ℝ) 
  (hA : A = 20)
  (hC : C = 90) :
  B = 70 := 
by
  -- In a triangle the sum of angles is 180 degrees
  have h_sum : A + B + C = 180 := sorry
  -- Substitute the given angles A and C
  rw [hA, hC] at h_sum
  -- Simplify the equation to find B
  have hB : 20 + B + 90 = 180 := sorry
  linarith

end NUMINAMATH_GPT_angle_sum_triangle_l282_28284


namespace NUMINAMATH_GPT_birdhouse_flown_distance_l282_28243

-- Definition of the given conditions.
def car_distance : ℕ := 200
def lawn_chair_distance : ℕ := 2 * car_distance
def birdhouse_distance : ℕ := 3 * lawn_chair_distance

-- Statement of the proof problem.
theorem birdhouse_flown_distance : birdhouse_distance = 1200 := by
  sorry

end NUMINAMATH_GPT_birdhouse_flown_distance_l282_28243


namespace NUMINAMATH_GPT_sector_area_l282_28271

theorem sector_area (r θ : ℝ) (hr : r = 2) (hθ : θ = (45 : ℝ) * (Real.pi / 180)) : 
  (1 / 2) * r^2 * θ = Real.pi / 2 := 
by
  sorry

end NUMINAMATH_GPT_sector_area_l282_28271


namespace NUMINAMATH_GPT_range_m_for_p_range_m_for_q_range_m_for_not_p_or_q_l282_28253

-- Define the propositions p and q
def p (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + 2 + m = 0

def q (m : ℝ) : Prop :=
  1 - 2 * m < 0 ∧ m + 2 > 0 ∨ 1 - 2 * m > 0 ∧ m + 2 < 0 -- Hyperbola condition

-- Prove the ranges of m
theorem range_m_for_p {m : ℝ} (hp : p m) : m ≤ -2 ∨ m ≥ 1 :=
sorry

theorem range_m_for_q {m : ℝ} (hq : q m) : m < -2 ∨ m > (1 / 2) :=
sorry

theorem range_m_for_not_p_or_q {m : ℝ} (h_not_p : ¬ (p m)) (h_not_q : ¬ (q m)) : -2 < m ∧ m ≤ (1 / 2) :=
sorry

end NUMINAMATH_GPT_range_m_for_p_range_m_for_q_range_m_for_not_p_or_q_l282_28253


namespace NUMINAMATH_GPT_right_triangle_angles_l282_28254

theorem right_triangle_angles (α β : ℝ) (h : α + β = 90) 
  (h_ratio : (180 - α) / (90 + α) = 9 / 11) : 
  (α = 58.5 ∧ β = 31.5) :=
by sorry

end NUMINAMATH_GPT_right_triangle_angles_l282_28254


namespace NUMINAMATH_GPT_temperature_on_friday_l282_28222

def temperatures (M T W Th F : ℝ) : Prop :=
  (M + T + W + Th) / 4 = 48 ∧
  (T + W + Th + F) / 4 = 40 ∧
  M = 42

theorem temperature_on_friday (M T W Th F : ℝ) (h : temperatures M T W Th F) : 
  F = 10 :=
  by
    -- problem statement
    sorry

end NUMINAMATH_GPT_temperature_on_friday_l282_28222


namespace NUMINAMATH_GPT_multiple_of_savings_l282_28225

theorem multiple_of_savings (P : ℝ) (h : P > 0) :
  let monthly_savings := (1 / 4) * P
  let monthly_non_savings := (3 / 4) * P
  let total_yearly_savings := 12 * monthly_savings
  ∃ M : ℝ, total_yearly_savings = M * monthly_non_savings ∧ M = 4 := 
by
  sorry

end NUMINAMATH_GPT_multiple_of_savings_l282_28225


namespace NUMINAMATH_GPT_mostWaterIntake_l282_28245

noncomputable def dailyWaterIntakeDongguk : ℝ := 5 * 0.2 -- Total water intake in liters per day for Dongguk
noncomputable def dailyWaterIntakeYoonji : ℝ := 6 * 0.3 -- Total water intake in liters per day for Yoonji
noncomputable def dailyWaterIntakeHeejin : ℝ := 4 * 500 / 1000 -- Total water intake in liters per day for Heejin (converted from milliliters)

theorem mostWaterIntake :
  dailyWaterIntakeHeejin = max dailyWaterIntakeDongguk (max dailyWaterIntakeYoonji dailyWaterIntakeHeejin) :=
by
  sorry

end NUMINAMATH_GPT_mostWaterIntake_l282_28245


namespace NUMINAMATH_GPT_increasing_sequence_a_range_l282_28291

theorem increasing_sequence_a_range (f : ℕ → ℝ) (a : ℝ)
  (h1 : ∀ n, f n = if n ≤ 7 then (3 - a) * n - 3 else a ^ (n - 6))
  (h2 : ∀ n : ℕ, f n < f (n + 1)) :
  2 < a ∧ a < 3 :=
sorry

end NUMINAMATH_GPT_increasing_sequence_a_range_l282_28291


namespace NUMINAMATH_GPT_linda_original_amount_l282_28285

-- Define the original amount of money Lucy and Linda have
variables (L : ℕ) (lucy_initial : ℕ := 20)

-- Condition: If Lucy gives Linda $5, they have the same amount of money.
def condition := (lucy_initial - 5) = (L + 5)

-- Theorem: The original amount of money that Linda had
theorem linda_original_amount (h : condition L) : L = 10 := 
sorry

end NUMINAMATH_GPT_linda_original_amount_l282_28285


namespace NUMINAMATH_GPT_principal_sum_l282_28235

theorem principal_sum (R P : ℝ) (h : (P * (R + 3) * 3) / 100 = (P * R * 3) / 100 + 81) : P = 900 :=
by
  sorry

end NUMINAMATH_GPT_principal_sum_l282_28235


namespace NUMINAMATH_GPT_area_to_paint_l282_28289

def height_of_wall : ℝ := 10
def length_of_wall : ℝ := 15
def window_height : ℝ := 3
def window_length : ℝ := 3
def door_height : ℝ := 1
def door_length : ℝ := 7

theorem area_to_paint : 
  let total_wall_area := height_of_wall * length_of_wall
  let window_area := window_height * window_length
  let door_area := door_height * door_length
  let area_to_paint := total_wall_area - window_area - door_area
  area_to_paint = 134 := 
by 
  sorry

end NUMINAMATH_GPT_area_to_paint_l282_28289


namespace NUMINAMATH_GPT_existence_of_xyz_l282_28261

theorem existence_of_xyz (n : ℕ) (hn_pos : 0 < n)
    (a b c : ℕ) (ha : 0 < a ∧ a ≤ 3 * n^2 + 4 * n) 
                (hb : 0 < b ∧ b ≤ 3 * n^2 + 4 * n) 
                (hc : 0 < c ∧ c ≤ 3 * n^2 + 4 * n) : 
  ∃ (x y z : ℤ), (|x| ≤ 2 * n) ∧ (|y| ≤ 2 * n) ∧ (|z| ≤ 2 * n) ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ a * x + b * y + c * z = 0 := by
  sorry

end NUMINAMATH_GPT_existence_of_xyz_l282_28261


namespace NUMINAMATH_GPT_sum_of_x_values_proof_l282_28211

noncomputable def sum_of_x_values : ℝ := 
  (-(-4)) / 1 -- Sum of roots of x^2 - 4x - 7 = 0

theorem sum_of_x_values_proof (x : ℝ) (h : 7 = (x^3 - 2 * x^2 - 8 * x) / (x + 2)) : sum_of_x_values = 4 :=
sorry

end NUMINAMATH_GPT_sum_of_x_values_proof_l282_28211


namespace NUMINAMATH_GPT_product_of_geometric_progressions_is_geometric_general_function_form_geometric_l282_28229

variables {α β γ : Type*} [CommSemiring α] [CommSemiring β] [CommSemiring γ]

-- Define the terms of geometric progressions
def term (a r : α) (k : ℕ) : α := a * r ^ (k - 1)

-- Define a general function with respective powers
def general_term (a r : α) (k p : ℕ) : α := a ^ p * (r ^ p) ^ (k - 1)

theorem product_of_geometric_progressions_is_geometric
  {a b c : α} {r1 r2 r3 : α} (k : ℕ) :
  term a r1 k * term b r2 k * term c r3 k = 
  (a * b * c) * (r1 * r2 * r3) ^ (k - 1) := 
sorry

theorem general_function_form_geometric
  {a b c : α} {r1 r2 r3 : α} {p q r : ℕ} (k : ℕ) :
  general_term a r1 k p * general_term b r2 k q * general_term c r3 k r = 
  (a^p * b^q * c^r) * (r1^p * r2^q * r3^r) ^ (k - 1) := 
sorry

end NUMINAMATH_GPT_product_of_geometric_progressions_is_geometric_general_function_form_geometric_l282_28229


namespace NUMINAMATH_GPT_line_does_not_pass_through_second_quadrant_l282_28232
-- Import the Mathlib library

-- Define the properties of the line
def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the condition for a point to be in the second quadrant:
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the proof statement
theorem line_does_not_pass_through_second_quadrant:
  ∀ x y : ℝ, line_eq x y → ¬ in_second_quadrant x y :=
by
  sorry

end NUMINAMATH_GPT_line_does_not_pass_through_second_quadrant_l282_28232


namespace NUMINAMATH_GPT_man_speed_is_approximately_54_009_l282_28280

noncomputable def speed_in_kmh (d : ℝ) (t : ℝ) : ℝ := 
  -- Convert distance to kilometers and time to hours
  let distance_km := d / 1000
  let time_hours := t / 3600
  distance_km / time_hours

theorem man_speed_is_approximately_54_009 :
  abs (speed_in_kmh 375.03 25 - 54.009) < 0.001 := 
by
  sorry

end NUMINAMATH_GPT_man_speed_is_approximately_54_009_l282_28280


namespace NUMINAMATH_GPT_charles_initial_bananas_l282_28246

theorem charles_initial_bananas (W C : ℕ) (h1 : W = 48) (h2 : C = C - 35 + W - 13) : C = 35 := by
  -- W = 48
  -- Charles loses 35 bananas
  -- Willie will have 13 bananas
  sorry

end NUMINAMATH_GPT_charles_initial_bananas_l282_28246


namespace NUMINAMATH_GPT_sin_360_eq_0_l282_28202

theorem sin_360_eq_0 : Real.sin (360 * Real.pi / 180) = 0 := by
  sorry

end NUMINAMATH_GPT_sin_360_eq_0_l282_28202


namespace NUMINAMATH_GPT_train_carriages_l282_28264

theorem train_carriages (num_trains : ℕ) (total_wheels : ℕ) (rows_per_carriage : ℕ) 
  (wheels_per_row : ℕ) (carriages_per_train : ℕ) :
  num_trains = 4 →
  total_wheels = 240 →
  rows_per_carriage = 3 →
  wheels_per_row = 5 →
  carriages_per_train = 
    (total_wheels / (rows_per_carriage * wheels_per_row)) / num_trains →
  carriages_per_train = 4 :=
by
  sorry

end NUMINAMATH_GPT_train_carriages_l282_28264


namespace NUMINAMATH_GPT_min_value_geometric_sequence_l282_28262

theorem min_value_geometric_sequence (a_2 a_3 : ℝ) (r : ℝ) 
(h_a2 : a_2 = 2 * r) (h_a3 : a_3 = 2 * r^2) : 
  (6 * a_2 + 7 * a_3) = -18 / 7 :=
by
  sorry

end NUMINAMATH_GPT_min_value_geometric_sequence_l282_28262


namespace NUMINAMATH_GPT_abs_eq_sets_l282_28207

theorem abs_eq_sets (x : ℝ) : 
  (|x - 25| + |x - 15| = |2 * x - 40|) → (x ≤ 15 ∨ x ≥ 25) :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_sets_l282_28207


namespace NUMINAMATH_GPT_biology_to_general_ratio_l282_28255

variable (g b m : ℚ)

theorem biology_to_general_ratio (h1 : g = 30) 
                                (h2 : m = (3/5) * (g + b)) 
                                (h3 : g + b + m = 144) : 
                                b / g = 2 / 1 := 
by 
  sorry

end NUMINAMATH_GPT_biology_to_general_ratio_l282_28255


namespace NUMINAMATH_GPT_inequality_N_value_l282_28257

theorem inequality_N_value (a c : ℝ) (ha : 0 < a) (hc : 0 < c) (b : ℝ) (hb : b = 2 * a) : 
  (a^2 + b^2) / c^2 > 5 / 9 := 
by sorry

end NUMINAMATH_GPT_inequality_N_value_l282_28257


namespace NUMINAMATH_GPT_sum_base5_eq_l282_28278

theorem sum_base5_eq :
  (432 + 43 + 4 : ℕ) = 1034 :=
by sorry

end NUMINAMATH_GPT_sum_base5_eq_l282_28278


namespace NUMINAMATH_GPT_find_g4_l282_28293

variables (g : ℝ → ℝ)

-- Given conditions
axiom condition1 : ∀ x : ℝ, g x + 3 * g (2 - x) = 2 * x^2 + x - 1
axiom condition2 : g 4 + 3 * g (-2) = 35
axiom condition3 : g (-2) + 3 * g 4 = 5

theorem find_g4 : g 4 = -5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_g4_l282_28293


namespace NUMINAMATH_GPT_geese_in_marsh_l282_28252

theorem geese_in_marsh (D : ℝ) (hD : D = 37.0) (G : ℝ) (hG : G = D + 21) : G = 58.0 := 
by 
  sorry

end NUMINAMATH_GPT_geese_in_marsh_l282_28252


namespace NUMINAMATH_GPT_height_of_box_l282_28277

-- Definitions of given conditions
def length_box : ℕ := 9
def width_box : ℕ := 12
def num_cubes : ℕ := 108
def volume_cube : ℕ := 3
def volume_box : ℕ := num_cubes * volume_cube  -- Volume calculated from number of cubes and volume of each cube

-- The statement to prove
theorem height_of_box : 
  ∃ h : ℕ, volume_box = length_box * width_box * h ∧ h = 3 := by
  sorry

end NUMINAMATH_GPT_height_of_box_l282_28277


namespace NUMINAMATH_GPT_exp4_is_odd_l282_28209

-- Define the domain for n to be integers and the expressions used in the conditions
variable (n : ℤ)

-- Define the expressions
def exp1 := (n + 1) ^ 2
def exp2 := (n + 1) ^ 2 - (n - 1)
def exp3 := (n + 1) ^ 3
def exp4 := (n + 1) ^ 3 - n ^ 3

-- Prove that exp4 is always odd
theorem exp4_is_odd : ∀ n : ℤ, exp4 n % 2 = 1 := by {
  -- Lean code does not require a proof here, we'll put sorry to skip the proof
  sorry
}

end NUMINAMATH_GPT_exp4_is_odd_l282_28209


namespace NUMINAMATH_GPT_line_through_two_quadrants_l282_28295

theorem line_through_two_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
sorry

end NUMINAMATH_GPT_line_through_two_quadrants_l282_28295


namespace NUMINAMATH_GPT_william_library_visits_l282_28240

variable (W : ℕ) (J : ℕ)
variable (h1 : J = 4 * W)
variable (h2 : 4 * J = 32)

theorem william_library_visits : W = 2 :=
by
  sorry

end NUMINAMATH_GPT_william_library_visits_l282_28240


namespace NUMINAMATH_GPT_part1_part2_part3_l282_28231

variable {x y z : ℝ}

-- Given condition
variables (hx : x > 0) (hy : y > 0) (hz : z > 0)

theorem part1 : 
  (x / y + y / z + z / x) / 3 ≥ 1 := sorry

theorem part2 :
  x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ (x / y + y / z + z / x)^2 / 3 := sorry

theorem part3 :
  x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ x / y + y / z + z / x := sorry

end NUMINAMATH_GPT_part1_part2_part3_l282_28231


namespace NUMINAMATH_GPT_inequality_and_equality_condition_l282_28238

theorem inequality_and_equality_condition (x : ℝ) (hx : x > 0) : 
  x + 1 / x ≥ 2 ∧ (x + 1 / x = 2 ↔ x = 1) :=
  sorry

end NUMINAMATH_GPT_inequality_and_equality_condition_l282_28238


namespace NUMINAMATH_GPT_berries_difference_l282_28286

theorem berries_difference (total_berries : ℕ) (dima_rate : ℕ) (sergey_rate : ℕ)
  (sergey_berries_picked : ℕ) (dima_berries_picked : ℕ)
  (dima_basket : ℕ) (sergey_basket : ℕ) :
  total_berries = 900 →
  sergey_rate = 2 * dima_rate →
  sergey_berries_picked = 2 * (total_berries / 3) →
  dima_berries_picked = total_berries / 3 →
  sergey_basket = sergey_berries_picked / 2 →
  dima_basket = (2 * dima_berries_picked) / 3 →
  sergey_basket > dima_basket ∧ sergey_basket - dima_basket = 100 :=
by
  intro h_total h_rate h_sergey_picked h_dima_picked h_sergey_basket h_dima_basket
  sorry

end NUMINAMATH_GPT_berries_difference_l282_28286


namespace NUMINAMATH_GPT_fraction_to_decimal_l282_28259

theorem fraction_to_decimal :
  ∀ x : ℚ, x = 52 / 180 → x = 0.1444 := 
sorry

end NUMINAMATH_GPT_fraction_to_decimal_l282_28259


namespace NUMINAMATH_GPT_total_stars_l282_28205

-- Define the daily stars earned by Shelby
def shelby_monday : Nat := 4
def shelby_tuesday : Nat := 6
def shelby_wednesday : Nat := 3
def shelby_thursday : Nat := 5
def shelby_friday : Nat := 2
def shelby_saturday : Nat := 3
def shelby_sunday : Nat := 7

-- Define the daily stars earned by Alex
def alex_monday : Nat := 5
def alex_tuesday : Nat := 3
def alex_wednesday : Nat := 6
def alex_thursday : Nat := 4
def alex_friday : Nat := 7
def alex_saturday : Nat := 2
def alex_sunday : Nat := 5

-- Define the total stars earned by Shelby in a week
def total_shelby_stars : Nat := shelby_monday + shelby_tuesday + shelby_wednesday + shelby_thursday + shelby_friday + shelby_saturday + shelby_sunday

-- Define the total stars earned by Alex in a week
def total_alex_stars : Nat := alex_monday + alex_tuesday + alex_wednesday + alex_thursday + alex_friday + alex_saturday + alex_sunday

-- The proof problem statement
theorem total_stars (total_shelby_stars total_alex_stars : Nat) : total_shelby_stars + total_alex_stars = 62 := by
  sorry

end NUMINAMATH_GPT_total_stars_l282_28205


namespace NUMINAMATH_GPT_annual_rent_per_sqft_l282_28204

theorem annual_rent_per_sqft
  (length width monthly_rent : ℕ)
  (H_length : length = 10)
  (H_width : width = 8)
  (H_monthly_rent : monthly_rent = 2400) :
  (12 * monthly_rent) / (length * width) = 360 := by
  sorry

end NUMINAMATH_GPT_annual_rent_per_sqft_l282_28204


namespace NUMINAMATH_GPT_inf_pos_integers_n_sum_two_squares_l282_28206

theorem inf_pos_integers_n_sum_two_squares:
  ∃ (s : ℕ → ℕ), (∀ (k : ℕ), ∃ (a₁ b₁ a₂ b₂ : ℕ),
   a₁ > 0 ∧ b₁ > 0 ∧ a₂ > 0 ∧ b₂ > 0 ∧ s k = n ∧
   n = a₁^2 + b₁^2 ∧ n = a₂^2 + b₂^2 ∧ 
  (a₁ ≠ a₂ ∨ b₁ ≠ b₂)) := sorry

end NUMINAMATH_GPT_inf_pos_integers_n_sum_two_squares_l282_28206


namespace NUMINAMATH_GPT_Lakota_spent_l282_28265

-- Define the conditions
def U : ℝ := 9.99
def Mackenzies_cost (N : ℝ) : ℝ := 3 * N + 8 * U
def cost_of_Lakotas_disks (N : ℝ) : ℝ := 6 * N + 2 * U

-- State the theorem
theorem Lakota_spent (N : ℝ) (h : Mackenzies_cost N = 133.89) : cost_of_Lakotas_disks N = 127.92 :=
by
  sorry

end NUMINAMATH_GPT_Lakota_spent_l282_28265

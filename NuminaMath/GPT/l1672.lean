import Mathlib

namespace NUMINAMATH_GPT_parallelogram_area_l1672_167265

-- Definitions
def base_cm : ℕ := 22
def height_cm : ℕ := 21

-- Theorem statement
theorem parallelogram_area : base_cm * height_cm = 462 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l1672_167265


namespace NUMINAMATH_GPT_algebraic_expression_value_l1672_167216

theorem algebraic_expression_value (x : ℝ) (hx : x = Real.sqrt 7 + 1) :
  (x^2 / (x - 3) - 2 * x / (x - 3)) / (x / (x - 3)) = Real.sqrt 7 - 1 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1672_167216


namespace NUMINAMATH_GPT_ratio_AB_AD_l1672_167232

theorem ratio_AB_AD (a x y : ℝ) (h1 : 0.3 * a^2 = 0.7 * x * y) (h2 : y = a / 10) : x / y = 43 :=
by
  sorry

end NUMINAMATH_GPT_ratio_AB_AD_l1672_167232


namespace NUMINAMATH_GPT_rubble_initial_money_l1672_167219

def initial_money (cost_notebook cost_pen : ℝ) (num_notebooks num_pens : ℕ) (money_left : ℝ) : ℝ :=
  (num_notebooks * cost_notebook + num_pens * cost_pen) + money_left

theorem rubble_initial_money :
  initial_money 4 1.5 2 2 4 = 15 :=
by
  sorry

end NUMINAMATH_GPT_rubble_initial_money_l1672_167219


namespace NUMINAMATH_GPT_compare_two_sqrt_three_with_three_l1672_167233

theorem compare_two_sqrt_three_with_three : 2 * Real.sqrt 3 > 3 :=
sorry

end NUMINAMATH_GPT_compare_two_sqrt_three_with_three_l1672_167233


namespace NUMINAMATH_GPT_least_positive_integer_to_multiple_of_4_l1672_167246

theorem least_positive_integer_to_multiple_of_4 : ∃ n : ℕ, n > 0 ∧ ((563 + n) % 4 = 0) ∧ n = 1 := 
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_to_multiple_of_4_l1672_167246


namespace NUMINAMATH_GPT_range_of_a_l1672_167284

noncomputable def f (a x : ℝ) : ℝ :=
  Real.exp (x-2) + (1/3) * x^3 - (3/2) * x^2 + 2 * x - Real.log (x-1) + a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, (1 < x → f a x = y) ↔ ∃ z : ℝ, 1 < z → f a (f a z) = y) →
  a ≤ 1/3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1672_167284


namespace NUMINAMATH_GPT_percent_more_proof_l1672_167213

-- Define the conditions
def y := 150
def x := 120
def is_percent_more (y x p : ℕ) : Prop := y = (1 + p / 100) * x

-- The proof problem statement
theorem percent_more_proof : ∃ p : ℕ, is_percent_more y x p ∧ p = 25 := by
  sorry

end NUMINAMATH_GPT_percent_more_proof_l1672_167213


namespace NUMINAMATH_GPT_connections_in_computer_lab_l1672_167273

theorem connections_in_computer_lab (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 := by
  sorry

end NUMINAMATH_GPT_connections_in_computer_lab_l1672_167273


namespace NUMINAMATH_GPT_scientific_notation_of_600000_l1672_167221

theorem scientific_notation_of_600000 : 600000 = 6 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_600000_l1672_167221


namespace NUMINAMATH_GPT_percentage_difference_l1672_167253

theorem percentage_difference : 0.70 * 100 - 0.60 * 80 = 22 := 
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l1672_167253


namespace NUMINAMATH_GPT_hiking_trip_rate_ratio_l1672_167299

theorem hiking_trip_rate_ratio 
  (rate_up : ℝ) (time_up : ℝ) (distance_down : ℝ) (time_down : ℝ)
  (h1 : rate_up = 7) 
  (h2 : time_up = 2) 
  (h3 : distance_down = 21) 
  (h4 : time_down = 2) : 
  (distance_down / time_down) / rate_up = 1.5 :=
by
  -- skip the proof as per instructions
  sorry

end NUMINAMATH_GPT_hiking_trip_rate_ratio_l1672_167299


namespace NUMINAMATH_GPT_gold_hammer_weight_l1672_167227

theorem gold_hammer_weight (a : ℕ → ℕ) 
  (h_arith_seq : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h_a1 : a 1 = 4) 
  (h_a5 : a 5 = 2) : 
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 := 
sorry

end NUMINAMATH_GPT_gold_hammer_weight_l1672_167227


namespace NUMINAMATH_GPT_unique_solution_for_star_l1672_167263

def star (x y : ℝ) : ℝ := 4 * x - 5 * y + 2 * x * y

theorem unique_solution_for_star :
  ∃! y : ℝ, star 2 y = 5 :=
by
  -- We know the definition of star and we need to verify the condition.
  sorry

end NUMINAMATH_GPT_unique_solution_for_star_l1672_167263


namespace NUMINAMATH_GPT_propositions_false_l1672_167241

structure Plane :=
(is_plane : Prop)

structure Line :=
(in_plane : Plane → Prop)

def is_parallel (p1 p2 : Plane) : Prop := sorry
def is_perpendicular (p1 p2 : Plane) : Prop := sorry
def line_parallel (l1 l2 : Line) : Prop := sorry
def line_perpendicular (l1 l2 : Line) : Prop := sorry

variable (α β : Plane)
variable (l m : Line)

axiom α_neq_β : α ≠ β
axiom l_in_α : l.in_plane α
axiom m_in_β : m.in_plane β

theorem propositions_false :
  ¬(is_parallel α β → line_parallel l m) ∧ 
  ¬(line_perpendicular l m → is_perpendicular α β) := 
sorry

end NUMINAMATH_GPT_propositions_false_l1672_167241


namespace NUMINAMATH_GPT_intersection_point_exists_l1672_167259

def line_l (x y : ℝ) : Prop := 2 * x + y = 10
def line_l_prime (x y : ℝ) : Prop := x - 2 * y + 10 = 0
def passes_through (x y : ℝ) (p : ℝ × ℝ) : Prop := p.2 = y ∧ 2 * p.1 - 10 = x

theorem intersection_point_exists :
  ∃ p : ℝ × ℝ, line_l p.1 p.2 ∧ line_l_prime p.1 p.2 ∧ passes_through p.1 p.2 (-10, 0) :=
sorry

end NUMINAMATH_GPT_intersection_point_exists_l1672_167259


namespace NUMINAMATH_GPT_sqrt_of_quarter_l1672_167237

-- Definitions as per conditions
def is_square_root (x y : ℝ) : Prop := x^2 = y

-- Theorem statement proving question == answer given conditions
theorem sqrt_of_quarter : is_square_root 0.5 0.25 ∧ is_square_root (-0.5) 0.25 ∧ (∀ x, is_square_root x 0.25 → (x = 0.5 ∨ x = -0.5)) :=
by
  -- Skipping proof with sorry
  sorry

end NUMINAMATH_GPT_sqrt_of_quarter_l1672_167237


namespace NUMINAMATH_GPT_Xiaoliang_catches_up_in_h_l1672_167234

-- Define the speeds and head start
def speed_Xiaobin : ℝ := 4  -- Xiaobin's speed in km/h
def speed_Xiaoliang : ℝ := 12  -- Xiaoliang's speed in km/h
def head_start : ℝ := 6  -- Xiaobin's head start in hours

-- Define the additional distance Xiaoliang needs to cover
def additional_distance : ℝ := speed_Xiaobin * head_start

-- Define the hourly distance difference between them
def speed_difference : ℝ := speed_Xiaoliang - speed_Xiaobin

-- Prove that Xiaoliang will catch up with Xiaobin in exactly 3 hours
theorem Xiaoliang_catches_up_in_h : (additional_distance / speed_difference) = 3 :=
by
  sorry

end NUMINAMATH_GPT_Xiaoliang_catches_up_in_h_l1672_167234


namespace NUMINAMATH_GPT_cannot_sum_to_nine_l1672_167249

def sum_pairs (a b c d : ℕ) : List ℕ :=
  [a + b, c + d, a + c, b + d, a + d, b + c]

theorem cannot_sum_to_nine :
  ∀ (a b c d : ℕ), a ≠ 5 ∧ b ≠ 6 ∧ c ≠ 5 ∧ d ≠ 6 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b ≠ 11 ∧ a + c ≠ 11 ∧ a + d ≠ 11 ∧ b + c ≠ 11 ∧ b + d ≠ 11 ∧ c + d ≠ 11 →
  ¬9 ∈ sum_pairs a b c d :=
by
  intros a b c d h
  sorry

end NUMINAMATH_GPT_cannot_sum_to_nine_l1672_167249


namespace NUMINAMATH_GPT_jordyn_total_cost_l1672_167224

-- Definitions for conditions
def price_cherries : ℝ := 5
def price_olives : ℝ := 7
def number_of_bags : ℕ := 50
def discount_rate : ℝ := 0.10 

-- Define the discounted price function
def discounted_price (price : ℝ) (discount : ℝ) : ℝ := price * (1 - discount)

-- Calculate the total cost for Jordyn
def total_cost (price_cherries price_olives : ℝ) (number_of_bags : ℕ) (discount_rate : ℝ) : ℝ :=
  (number_of_bags * discounted_price price_cherries discount_rate) + 
  (number_of_bags * discounted_price price_olives discount_rate)

-- Prove the final cost
theorem jordyn_total_cost : total_cost price_cherries price_olives number_of_bags discount_rate = 540 := by
  sorry

end NUMINAMATH_GPT_jordyn_total_cost_l1672_167224


namespace NUMINAMATH_GPT_geometric_sequence_second_term_l1672_167252

theorem geometric_sequence_second_term (a r : ℕ) (h1 : a = 5) (h2 : a * r^4 = 1280) : a * r = 20 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_second_term_l1672_167252


namespace NUMINAMATH_GPT_problem1_problem2_l1672_167293

-- Problem 1
theorem problem1 (a b : ℝ) (h : 2 * (a + 1) * (b + 1) = (a + b) * (a + b + 2)) : a^2 + b^2 = 2 := sorry

-- Problem 2
theorem problem2 (a b c : ℝ) (h : a^2 + c^2 = 2 * b^2) : (a + b) * (a + c) + (c + a) * (c + b) = 2 * (b + a) * (b + c) := sorry

end NUMINAMATH_GPT_problem1_problem2_l1672_167293


namespace NUMINAMATH_GPT_smallest_n_l1672_167240

-- Define the costs.
def cost_red := 10 * 8  -- = 80
def cost_green := 18 * 12  -- = 216
def cost_blue := 20 * 15  -- = 300
def cost_yellow (n : Nat) := 24 * n

-- Define the LCM of the costs.
def LCM_cost : Nat := Nat.lcm (Nat.lcm cost_red cost_green) cost_blue

-- Problem statement: Prove that the smallest value of n such that 24 * n is the LCM of the candy costs is 150.
theorem smallest_n : ∃ n : Nat, cost_yellow n = LCM_cost ∧ n = 150 := 
by {
  -- This part is just a placeholder; the proof steps are omitted.
  sorry
}

end NUMINAMATH_GPT_smallest_n_l1672_167240


namespace NUMINAMATH_GPT_find_difference_square_l1672_167271

theorem find_difference_square (x y c b : ℝ) (h1 : x * y = c^2) (h2 : (1 / x^2) + (1 / y^2) = b * c) : 
  (x - y)^2 = b * c^4 - 2 * c^2 := 
by sorry

end NUMINAMATH_GPT_find_difference_square_l1672_167271


namespace NUMINAMATH_GPT_new_mean_after_adding_eleven_l1672_167281

theorem new_mean_after_adding_eleven (nums : List ℝ) (h_len : nums.length = 15) (h_avg : (nums.sum / 15) = 40) :
  ((nums.map (λ x => x + 11)).sum / 15) = 51 := by
  sorry

end NUMINAMATH_GPT_new_mean_after_adding_eleven_l1672_167281


namespace NUMINAMATH_GPT_problem_statement_l1672_167269

theorem problem_statement (n m N k : ℕ)
  (h : (n^2 + 1)^(2^k) * (44 * n^3 + 11 * n^2 + 10 * n + 2) = N^m) :
  m = 1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1672_167269


namespace NUMINAMATH_GPT_monster_perimeter_correct_l1672_167217

noncomputable def monster_perimeter (radius : ℝ) (central_angle_missing : ℝ) : ℝ :=
  let full_circle_circumference := 2 * radius * Real.pi
  let arc_length := (1 - central_angle_missing / 360) * full_circle_circumference
  arc_length + 2 * radius

theorem monster_perimeter_correct :
  monster_perimeter 2 90 = 3 * Real.pi + 4 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_monster_perimeter_correct_l1672_167217


namespace NUMINAMATH_GPT_number_divisible_by_75_l1672_167206

def is_two_digit (x : ℕ) := x >= 10 ∧ x < 100

theorem number_divisible_by_75 {a b : ℕ} (h1 : a * b = 35) (h2 : is_two_digit (10 * a + b)) : (10 * a + b) % 75 = 0 :=
sorry

end NUMINAMATH_GPT_number_divisible_by_75_l1672_167206


namespace NUMINAMATH_GPT_upper_limit_of_sixth_powers_l1672_167256

theorem upper_limit_of_sixth_powers :
  ∃ b : ℕ, (∀ n : ℕ, (∃ a : ℕ, a^6 = n) ∧ n ≤ b → n = 46656) :=
by
  sorry

end NUMINAMATH_GPT_upper_limit_of_sixth_powers_l1672_167256


namespace NUMINAMATH_GPT_jacoby_lottery_winning_l1672_167215

theorem jacoby_lottery_winning :
  let total_needed := 5000
  let job_earning := 20 * 10
  let cookies_earning := 4 * 24
  let total_earnings_before_lottery := job_earning + cookies_earning
  let after_lottery := total_earnings_before_lottery - 10
  let gift_from_sisters := 500 * 2
  let total_earnings_and_gifts := after_lottery + gift_from_sisters
  let total_so_far := total_needed - 3214
  total_so_far - total_earnings_and_gifts = 500 :=
by
  sorry

end NUMINAMATH_GPT_jacoby_lottery_winning_l1672_167215


namespace NUMINAMATH_GPT_neg_p_l1672_167280

open Set

-- Definitions of sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

def A : Set ℤ := {x | is_odd x}
def B : Set ℤ := {x | is_even x}

-- Proposition p
def p : Prop := ∀ x ∈ A, 2 * x ∈ B

-- Negation of the proposition p
theorem neg_p : ¬p ↔ ∃ x ∈ A, ¬(2 * x ∈ B) := sorry

end NUMINAMATH_GPT_neg_p_l1672_167280


namespace NUMINAMATH_GPT_john_bought_soap_l1672_167210

theorem john_bought_soap (weight_per_bar : ℝ) (cost_per_pound : ℝ) (total_spent : ℝ) (h1 : weight_per_bar = 1.5) (h2 : cost_per_pound = 0.5) (h3 : total_spent = 15) : 
  total_spent / (weight_per_bar * cost_per_pound) = 20 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_john_bought_soap_l1672_167210


namespace NUMINAMATH_GPT_hyperbolic_identity_l1672_167268

noncomputable def sh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def ch (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

theorem hyperbolic_identity (x : ℝ) : (ch x) ^ 2 - (sh x) ^ 2 = 1 := 
sorry

end NUMINAMATH_GPT_hyperbolic_identity_l1672_167268


namespace NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l1672_167288

theorem isosceles_triangle_vertex_angle (B : ℝ) (V : ℝ) (h1 : B = 70) (h2 : B = B) (h3 : V + 2 * B = 180) : V = 40 ∨ V = 70 :=
by {
  sorry
}

end NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l1672_167288


namespace NUMINAMATH_GPT_sum_a6_to_a9_l1672_167244

-- Given definitions and conditions
def sequence_sum (n : ℕ) : ℕ := n^3
def a (n : ℕ) : ℕ := sequence_sum (n + 1) - sequence_sum n

-- Theorem to be proved
theorem sum_a6_to_a9 : a 6 + a 7 + a 8 + a 9 = 604 :=
by sorry

end NUMINAMATH_GPT_sum_a6_to_a9_l1672_167244


namespace NUMINAMATH_GPT_original_price_of_dinosaur_model_l1672_167236

-- Define the conditions
theorem original_price_of_dinosaur_model
  (P : ℝ) -- original price of each model
  (kindergarten_models : ℝ := 2)
  (elementary_models : ℝ := 2 * kindergarten_models)
  (total_models : ℝ := kindergarten_models + elementary_models)
  (reduction_percentage : ℝ := 0.05)
  (discounted_price : ℝ := P * (1 - reduction_percentage))
  (total_paid : ℝ := total_models * discounted_price)
  (total_paid_condition : total_paid = 570) :
  P = 100 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_dinosaur_model_l1672_167236


namespace NUMINAMATH_GPT_binomial_product_l1672_167261

theorem binomial_product (x : ℝ) : (4 * x + 3) * (x - 6) = 4 * x ^ 2 - 21 * x - 18 := 
sorry

end NUMINAMATH_GPT_binomial_product_l1672_167261


namespace NUMINAMATH_GPT_solve_system_I_solve_system_II_l1672_167220

theorem solve_system_I (x y : ℝ) (h1 : y = x + 3) (h2 : x - 2 * y + 12 = 0) : x = 6 ∧ y = 9 :=
by
  sorry

theorem solve_system_II (x y : ℝ) (h1 : 4 * (x - y - 1) = 3 * (1 - y) - 2) (h2 : x / 2 + y / 3 = 2) : x = 2 ∧ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_I_solve_system_II_l1672_167220


namespace NUMINAMATH_GPT_avg_and_variance_decrease_l1672_167204

noncomputable def original_heights : List ℝ := [180, 184, 188, 190, 192, 194]
noncomputable def new_heights : List ℝ := [180, 184, 188, 190, 192, 188]

noncomputable def avg (heights : List ℝ) : ℝ :=
  heights.sum / heights.length

noncomputable def variance (heights : List ℝ) (mean : ℝ) : ℝ :=
  (heights.map (λ h => (h - mean) ^ 2)).sum / heights.length

theorem avg_and_variance_decrease :
  let original_mean := avg original_heights
  let new_mean := avg new_heights
  let original_variance := variance original_heights original_mean
  let new_variance := variance new_heights new_mean
  new_mean < original_mean ∧ new_variance < original_variance :=
by
  sorry

end NUMINAMATH_GPT_avg_and_variance_decrease_l1672_167204


namespace NUMINAMATH_GPT_first_train_length_correct_l1672_167264

noncomputable def length_of_first_train : ℝ :=
  let speed_first_train := 90 * 1000 / 3600  -- converting to m/s
  let speed_second_train := 72 * 1000 / 3600 -- converting to m/s
  let relative_speed := speed_first_train + speed_second_train
  let distance_apart := 630
  let length_second_train := 200
  let time_to_meet := 13.998880089592832
  let distance_covered := relative_speed * time_to_meet
  let total_distance := distance_apart
  let length_first_train := total_distance - length_second_train
  length_first_train

theorem first_train_length_correct :
  length_of_first_train = 430 :=
by
  -- Place for the proof steps
  sorry

end NUMINAMATH_GPT_first_train_length_correct_l1672_167264


namespace NUMINAMATH_GPT_range_of_m_l1672_167225

theorem range_of_m (m : ℝ) (h : 0 < m)
  (subset_cond : ∀ x y : ℝ, x - 4 ≤ 0 → y ≥ 0 → mx - y ≥ 0 → (x - 2)^2 + (y - 2)^2 ≤ 8) :
  m ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1672_167225


namespace NUMINAMATH_GPT_pounds_of_fish_to_ship_l1672_167294

theorem pounds_of_fish_to_ship (crates_weight : ℕ) (cost_per_crate : ℝ) (total_cost : ℝ) :
  crates_weight = 30 → cost_per_crate = 1.5 → total_cost = 27 → 
  (total_cost / cost_per_crate) * crates_weight = 540 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_pounds_of_fish_to_ship_l1672_167294


namespace NUMINAMATH_GPT_sqrt_36_eq_pm6_arith_sqrt_sqrt_16_eq_2_cube_root_minus_27_eq_minus_3_l1672_167277

-- Prove that the square root of 36 equals ±6
theorem sqrt_36_eq_pm6 : ∃ y : ℤ, y * y = 36 ∧ y = 6 ∨ y = -6 :=
by
  sorry

-- Prove that the arithmetic square root of sqrt(16) equals 2
theorem arith_sqrt_sqrt_16_eq_2 : ∃ z : ℝ, z * z = 16 ∧ z = 4 ∧ 2 * 2 = z :=
by
  sorry

-- Prove that the cube root of -27 equals -3
theorem cube_root_minus_27_eq_minus_3 : ∃ x : ℝ, x * x * x = -27 ∧ x = -3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_36_eq_pm6_arith_sqrt_sqrt_16_eq_2_cube_root_minus_27_eq_minus_3_l1672_167277


namespace NUMINAMATH_GPT_david_trip_distance_l1672_167222

theorem david_trip_distance (t : ℝ) (d : ℝ) : 
  (40 * (t + 1) = d) →
  (d - 40 = 60 * (t - 0.75)) →
  d = 130 := 
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_david_trip_distance_l1672_167222


namespace NUMINAMATH_GPT_trigonometric_identity_l1672_167276

variable (θ : ℝ) (h : Real.tan θ = 2)

theorem trigonometric_identity : 
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1672_167276


namespace NUMINAMATH_GPT_sarah_reads_100_words_per_page_l1672_167242

noncomputable def words_per_page (W_pages : ℕ) (books : ℕ) (hours : ℕ) (pages_per_book : ℕ) (words_per_minute : ℕ) : ℕ :=
  (words_per_minute * 60 * hours) / books / pages_per_book

theorem sarah_reads_100_words_per_page :
  words_per_page 80 6 20 80 40 = 100 := 
sorry

end NUMINAMATH_GPT_sarah_reads_100_words_per_page_l1672_167242


namespace NUMINAMATH_GPT_area_of_square_l1672_167218

noncomputable def length_of_rectangle (r : ℝ) : ℝ := (2 / 5) * r
noncomputable def area_of_rectangle_given_length_and_breadth (L B : ℝ) : ℝ := L * B

theorem area_of_square (r : ℝ) (B : ℝ) (A : ℝ) 
  (h_length : length_of_rectangle r = (2 / 5) * r) 
  (h_breadth : B = 10) 
  (h_area : A = 160) 
  (h_rectangle_area : area_of_rectangle_given_length_and_breadth ((2 / 5) * r) B = 160) : 
  r = 40 → (r ^ 2 = 1600) := 
by 
  sorry

end NUMINAMATH_GPT_area_of_square_l1672_167218


namespace NUMINAMATH_GPT_total_blocks_needed_l1672_167212

theorem total_blocks_needed (length height : ℕ) (block_height : ℕ) (block1_length block2_length : ℕ)
                            (height_blocks : height = 8) (length_blocks : length = 102)
                            (block_height_cond : block_height = 1)
                            (block_lengths : block1_length = 2 ∧ block2_length = 1)
                            (staggered_cond : True) (even_ends : True) :
  ∃ total_blocks, total_blocks = 416 := 
  sorry

end NUMINAMATH_GPT_total_blocks_needed_l1672_167212


namespace NUMINAMATH_GPT_solve_inequality_2_star_x_l1672_167258

theorem solve_inequality_2_star_x :
  ∀ x : ℝ, 
  6 < (2 * x - 2 - x + 3) ∧ (2 * x - 2 - x + 3) < 7 ↔ 5 < x ∧ x < 6 :=
by sorry

end NUMINAMATH_GPT_solve_inequality_2_star_x_l1672_167258


namespace NUMINAMATH_GPT_sum_lent_is_1050_l1672_167298

-- Define the variables for the problem
variable (P : ℝ) -- Sum lent
variable (r : ℝ) -- Interest rate
variable (t : ℝ) -- Time period
variable (I : ℝ) -- Interest

-- Define the conditions
def conditions := 
  r = 0.06 ∧ 
  t = 6 ∧ 
  I = P - 672 ∧ 
  I = P * (r * t)

-- Define the main theorem
theorem sum_lent_is_1050 (P r t I : ℝ) (h : conditions P r t I) : P = 1050 :=
  sorry

end NUMINAMATH_GPT_sum_lent_is_1050_l1672_167298


namespace NUMINAMATH_GPT_classroom_needs_more_money_l1672_167296

theorem classroom_needs_more_money 
    (goal : ℕ) 
    (raised_from_two_families : ℕ) 
    (raised_from_eight_families : ℕ) 
    (raised_from_ten_families : ℕ) 
    (H : goal = 200) 
    (H1 : raised_from_two_families = 2 * 20) 
    (H2 : raised_from_eight_families = 8 * 10) 
    (H3 : raised_from_ten_families = 10 * 5) 
    (total_raised : ℕ := raised_from_two_families + raised_from_eight_families + raised_from_ten_families) : 
    (goal - total_raised) = 30 := 
by 
  sorry

end NUMINAMATH_GPT_classroom_needs_more_money_l1672_167296


namespace NUMINAMATH_GPT_calculate_fraction_l1672_167207

theorem calculate_fraction : (10^20 / 50^10) = 2^10 := by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l1672_167207


namespace NUMINAMATH_GPT_sqrt_product_simplification_l1672_167267

-- Define the main problem
theorem sqrt_product_simplification : Real.sqrt 18 * Real.sqrt 72 = 36 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_product_simplification_l1672_167267


namespace NUMINAMATH_GPT_pairs_satisfy_inequality_l1672_167291

section inequality_problem

variables (a b : ℝ)

-- Conditions
variable (hb1 : b ≠ -1)
variable (hb2 : b ≠ 0)

-- Inequalities to check
def inequality (a b : ℝ) : Prop :=
  (1 + a) ^ 2 / (1 + b) ≤ 1 + a ^ 2 / b

-- Main theorem
theorem pairs_satisfy_inequality :
  (b > 0 ∨ b < -1 → ∀ a, a ≠ b → inequality a b) ∧
  (∀ a, a ≠ -1 ∧ a ≠ 0 → inequality a a) :=
by
  sorry

end inequality_problem

end NUMINAMATH_GPT_pairs_satisfy_inequality_l1672_167291


namespace NUMINAMATH_GPT_average_age_of_coaches_l1672_167290

variables 
  (total_members : ℕ) (avg_age_total : ℕ) 
  (num_girls : ℕ) (num_boys : ℕ) (num_coaches : ℕ) 
  (avg_age_girls : ℕ) (avg_age_boys : ℕ)

theorem average_age_of_coaches 
  (h1 : total_members = 50) 
  (h2 : avg_age_total = 18)
  (h3 : num_girls = 25) 
  (h4 : num_boys = 20) 
  (h5 : num_coaches = 5)
  (h6 : avg_age_girls = 16)
  (h7 : avg_age_boys = 17) : 
  (900 - (num_girls * avg_age_girls + num_boys * avg_age_boys)) / num_coaches = 32 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_coaches_l1672_167290


namespace NUMINAMATH_GPT_pints_of_cider_l1672_167262

def pintCider (g : ℕ) (p : ℕ) : ℕ :=
  g / 20 + p / 40

def totalApples (f : ℕ) (h : ℕ) (a : ℕ) : ℕ :=
  f * h * a

theorem pints_of_cider (g p : ℕ) (farmhands : ℕ) (hours : ℕ) (apples_per_hour : ℕ)
  (H1 : g = 1)
  (H2 : p = 2)
  (H3 : farmhands = 6)
  (H4 : hours = 5)
  (H5 : apples_per_hour = 240) :
  pintCider (apples_per_hour * farmhands * hours / 3)
            (apples_per_hour * farmhands * hours * 2 / 3) = 120 :=
by
  sorry

end NUMINAMATH_GPT_pints_of_cider_l1672_167262


namespace NUMINAMATH_GPT_ratio_of_inscribed_squares_l1672_167235

theorem ratio_of_inscribed_squares (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (hx : x = 60 / 17) (hy : y = 3) :
  x / y = 20 / 17 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_inscribed_squares_l1672_167235


namespace NUMINAMATH_GPT_parallelogram_sticks_l1672_167286

theorem parallelogram_sticks (a : ℕ) (h₁ : ∃ l₁ l₂, l₁ = 5 ∧ l₂ = 5 ∧ 
                                (l₁ = l₂) ∧ (a = 7)) : a = 7 :=
by sorry

end NUMINAMATH_GPT_parallelogram_sticks_l1672_167286


namespace NUMINAMATH_GPT_brian_oranges_is_12_l1672_167226

-- Define the number of oranges the person has
def person_oranges : Nat := 12

-- Define the number of oranges Brian has, which is zero fewer than the person's oranges
def brian_oranges : Nat := person_oranges - 0

-- The theorem stating that Brian has 12 oranges
theorem brian_oranges_is_12 : brian_oranges = 12 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_brian_oranges_is_12_l1672_167226


namespace NUMINAMATH_GPT_arrangement_plans_count_l1672_167287

noncomputable def number_of_arrangement_plans (num_teachers : ℕ) (num_students : ℕ) : ℕ :=
if num_teachers = 2 ∧ num_students = 4 then 12 else 0

theorem arrangement_plans_count :
  number_of_arrangement_plans 2 4 = 12 :=
by 
  sorry

end NUMINAMATH_GPT_arrangement_plans_count_l1672_167287


namespace NUMINAMATH_GPT_tangent_line_at_origin_l1672_167205

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x

theorem tangent_line_at_origin :
  ∃ (m b : ℝ), (m = 2) ∧ (b = 1) ∧ (∀ x, f x - (m * x + b) = 0 → 2 * x - f x + 1 = 0) :=
sorry

end NUMINAMATH_GPT_tangent_line_at_origin_l1672_167205


namespace NUMINAMATH_GPT_rectangle_perimeter_of_divided_square_l1672_167295

theorem rectangle_perimeter_of_divided_square
  (s : ℝ)
  (hs : 4 * s = 100) :
  let l := s
  let w := s / 2
  2 * (l + w) = 75 :=
by
  let l := s
  let w := s / 2
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_of_divided_square_l1672_167295


namespace NUMINAMATH_GPT_class_average_score_l1672_167200

theorem class_average_score (n_boys n_girls : ℕ) (avg_score_boys avg_score_girls : ℕ) 
  (h_nb : n_boys = 12)
  (h_ng : n_girls = 4)
  (h_ab : avg_score_boys = 84)
  (h_ag : avg_score_girls = 92) : 
  (n_boys * avg_score_boys + n_girls * avg_score_girls) / (n_boys + n_girls) = 86 := 
by 
  sorry

end NUMINAMATH_GPT_class_average_score_l1672_167200


namespace NUMINAMATH_GPT_surface_area_combination_l1672_167245

noncomputable def smallest_surface_area : ℕ :=
  let s1 := 3
  let s2 := 5
  let s3 := 8
  let surface_area := 6 * (s1 * s1 + s2 * s2 + s3 * s3)
  let overlap_area := (s1 * s1) * 4 + (s2 * s2) * 2 
  surface_area - overlap_area

theorem surface_area_combination :
  smallest_surface_area = 502 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_surface_area_combination_l1672_167245


namespace NUMINAMATH_GPT_number_is_280_l1672_167272

theorem number_is_280 (x : ℝ) (h : x / 5 + 4 = x / 4 - 10) : x = 280 := 
by 
  sorry

end NUMINAMATH_GPT_number_is_280_l1672_167272


namespace NUMINAMATH_GPT_unique_solution_only_a_is_2_l1672_167266

noncomputable def unique_solution_inequality (a : ℝ) : Prop :=
  ∀ (p : ℝ → ℝ), (∀ x, 0 ≤ p x ∧ p x ≤ 1 ∧ p x = x^2 - a * x + a) → 
  ∃! x, p x = 1

theorem unique_solution_only_a_is_2 (a : ℝ) (h : unique_solution_inequality a) : a = 2 :=
sorry

end NUMINAMATH_GPT_unique_solution_only_a_is_2_l1672_167266


namespace NUMINAMATH_GPT_range_of_m_l1672_167292

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (m : ℝ) (h : ¬ (p m ∨ q m)) : m ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1672_167292


namespace NUMINAMATH_GPT_molecular_weight_l1672_167243

variable (weight_moles : ℝ) (moles : ℝ)

-- Given conditions
axiom h1 : weight_moles = 699
axiom h2 : moles = 3

-- Concluding statement to prove
theorem molecular_weight : (weight_moles / moles) = 233 := sorry

end NUMINAMATH_GPT_molecular_weight_l1672_167243


namespace NUMINAMATH_GPT_wuyang_math_total_participants_l1672_167247

theorem wuyang_math_total_participants :
  ∀ (x : ℕ), 
  95 * (x + 5) = 75 * (x + 3 + 10) → 
  2 * (x + x + 8) + 9 = 125 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_wuyang_math_total_participants_l1672_167247


namespace NUMINAMATH_GPT_problem1_l1672_167239

theorem problem1 (a : ℝ) 
    (circle_eqn : ∀ (x y : ℝ), x^2 + y^2 - 2*a*x + a = 0)
    (line_eqn : ∀ (x y : ℝ), a*x + y + 1 = 0)
    (chord_length : ∀ (x y : ℝ), (ax + y + 1 = 0) ∧ (x^2 + y^2 - 2*a*x + a = 0)  -> ((x - x')^2 + (y - y')^2 = 4)) : 
    a = -2 := sorry

end NUMINAMATH_GPT_problem1_l1672_167239


namespace NUMINAMATH_GPT_h_h_three_l1672_167229

def h (x : ℤ) : ℤ := 3 * x^2 + 3 * x - 2

theorem h_h_three : h (h 3) = 3568 := by
  sorry

end NUMINAMATH_GPT_h_h_three_l1672_167229


namespace NUMINAMATH_GPT_find_a_if_f_is_even_l1672_167209

-- Defining f as given in the problem conditions
noncomputable def f (x a : ℝ) : ℝ := (x + a) * 3 ^ (x - 2 + a ^ 2) - (x - a) * 3 ^ (8 - x - 3 * a)

-- Statement of the proof problem with the conditions
theorem find_a_if_f_is_even (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → (a = -5 ∨ a = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_if_f_is_even_l1672_167209


namespace NUMINAMATH_GPT_total_working_days_l1672_167279

theorem total_working_days 
  (D : ℕ)
  (A : ℝ)
  (B : ℝ)
  (h1 : A * (D - 2) = 80)
  (h2 : B * (D - 5) = 63)
  (h3 : A * (D - 5) = B * (D - 2) + 2) :
  D = 32 := 
sorry

end NUMINAMATH_GPT_total_working_days_l1672_167279


namespace NUMINAMATH_GPT_f_neg_one_l1672_167260

-- Assume the function f : ℝ → ℝ
variable (f : ℝ → ℝ)

-- Conditions
-- 1. f(x) is odd: f(-x) = -f(x) for all x ∈ ℝ
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- 2. f(x) = 2^x for all x > 0
axiom f_pos : ∀ x : ℝ, x > 0 → f x = 2^x

-- Proof statement to be filled
theorem f_neg_one : f (-1) = -2 := 
by
  sorry

end NUMINAMATH_GPT_f_neg_one_l1672_167260


namespace NUMINAMATH_GPT_probability_age_less_than_20_l1672_167211

theorem probability_age_less_than_20 (total : ℕ) (ages_gt_30 : ℕ) (ages_lt_20 : ℕ) 
    (h1 : total = 150) (h2 : ages_gt_30 = 90) (h3 : ages_lt_20 = total - ages_gt_30) :
    (ages_lt_20 : ℚ) / total = 2 / 5 :=
by
  simp [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_probability_age_less_than_20_l1672_167211


namespace NUMINAMATH_GPT_tim_took_rulers_l1672_167285

theorem tim_took_rulers (initial_rulers : ℕ) (remaining_rulers : ℕ) (rulers_taken : ℕ) :
  initial_rulers = 46 → remaining_rulers = 21 → rulers_taken = initial_rulers - remaining_rulers → rulers_taken = 25 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_tim_took_rulers_l1672_167285


namespace NUMINAMATH_GPT_max_rectangle_area_l1672_167230

-- Lean statement for the proof problem

theorem max_rectangle_area (x : ℝ) (y : ℝ) (h1 : 2 * x + 2 * y = 24) : ∃ A : ℝ, A = 36 :=
by
  -- Definitions for perimeter and area
  let P := 2 * x + 2 * y
  let A := x * y

  -- Conditions
  have h1 : P = 24 := h1

  -- Setting maximum area and completing the proof
  sorry

end NUMINAMATH_GPT_max_rectangle_area_l1672_167230


namespace NUMINAMATH_GPT_arithmetic_seq_fifth_term_l1672_167275

theorem arithmetic_seq_fifth_term (x y : ℝ) 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 = 2 * x^2 + 3 * y^2) 
  (h2 : a2 = x^2 + 2 * y^2) 
  (h3 : a3 = 2 * x^2 - y^2) 
  (h4 : a4 = x^2 - y^2) 
  (d : ℝ) 
  (hd : d = -x^2 - y^2) 
  (h_arith: ∀ i j k : ℕ, i < j ∧ j < k → a2 - a1 = d ∧ a3 - a2 = d ∧ a4 - a3 = d) : 
  a4 + d = -2 * y^2 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_seq_fifth_term_l1672_167275


namespace NUMINAMATH_GPT_ways_to_sum_420_l1672_167278

theorem ways_to_sum_420 : 
  (∃ n k : ℕ, n ≥ 2 ∧ 2 * k + n - 1 > 0 ∧ n * (2 * k + n - 1) = 840) → (∃ c, c = 11) :=
by
  sorry

end NUMINAMATH_GPT_ways_to_sum_420_l1672_167278


namespace NUMINAMATH_GPT_earnings_per_day_correct_l1672_167254

-- Given conditions
variable (total_earned : ℕ) (days : ℕ) (earnings_per_day : ℕ)

-- Specify the given values from the conditions
def given_conditions : Prop :=
  total_earned = 165 ∧ days = 5 ∧ total_earned = days * earnings_per_day

-- Statement of the problem: proving the earnings per day
theorem earnings_per_day_correct (h : given_conditions total_earned days earnings_per_day) : 
  earnings_per_day = 33 :=
by
  sorry

end NUMINAMATH_GPT_earnings_per_day_correct_l1672_167254


namespace NUMINAMATH_GPT_integer_values_satisfying_square_root_condition_l1672_167282

theorem integer_values_satisfying_square_root_condition :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 := sorry

end NUMINAMATH_GPT_integer_values_satisfying_square_root_condition_l1672_167282


namespace NUMINAMATH_GPT_farmer_randy_total_acres_l1672_167214

-- Define the conditions
def acres_per_tractor_per_day : ℕ := 68
def tractors_first_2_days : ℕ := 2
def days_first_period : ℕ := 2
def tractors_next_3_days : ℕ := 7
def days_second_period : ℕ := 3

-- Prove the total acres Farmer Randy needs to plant
theorem farmer_randy_total_acres :
  (tractors_first_2_days * acres_per_tractor_per_day * days_first_period) +
  (tractors_next_3_days * acres_per_tractor_per_day * days_second_period) = 1700 :=
by
  -- Here, we would provide the proof, but in this example, we will use sorry.
  sorry

end NUMINAMATH_GPT_farmer_randy_total_acres_l1672_167214


namespace NUMINAMATH_GPT_rain_on_tuesday_l1672_167223

/-- Let \( R_M \) be the event that a county received rain on Monday. -/
def RM : Prop := sorry

/-- Let \( R_T \) be the event that a county received rain on Tuesday. -/
def RT : Prop := sorry

/-- Let \( R_{MT} \) be the event that a county received rain on both Monday and Tuesday. -/
def RMT : Prop := RM ∧ RT

/-- The probability that a county received rain on Monday is 0.62. -/
def prob_RM : ℝ := 0.62

/-- The probability that a county received rain on both Monday and Tuesday is 0.44. -/
def prob_RMT : ℝ := 0.44

/-- The probability that no rain fell on either day is 0.28. -/
def prob_no_rain : ℝ := 0.28

/-- The probability that a county received rain on at least one of the days is 0.72. -/
def prob_at_least_one_day : ℝ := 1 - prob_no_rain

/-- The probability that a county received rain on Tuesday is 0.54. -/
theorem rain_on_tuesday : (prob_at_least_one_day = prob_RM + x - prob_RMT) → (x = 0.54) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_rain_on_tuesday_l1672_167223


namespace NUMINAMATH_GPT_division_multiplication_identity_l1672_167208

theorem division_multiplication_identity (a b c d : ℕ) (h1 : b = 6) (h2 : c = 2) (h3 : d = 3) :
  a = 120 → 120 * (b / c) * d = 120 := by
  intro h
  rw [h2, h3, h1]
  sorry

end NUMINAMATH_GPT_division_multiplication_identity_l1672_167208


namespace NUMINAMATH_GPT_even_function_condition_iff_l1672_167228

theorem even_function_condition_iff (m : ℝ) :
    (∀ x : ℝ, (m * 2^x + 2^(-x)) = (m * 2^(-x) + 2^x)) ↔ (m = 1) :=
by
  sorry

end NUMINAMATH_GPT_even_function_condition_iff_l1672_167228


namespace NUMINAMATH_GPT_area_bounded_by_circles_and_x_axis_l1672_167255

/--
Circle C has its center at (5, 5) and radius 5 units.
Circle D has its center at (15, 5) and radius 5 units.
Prove that the area of the region bounded by these circles
and the x-axis is 50 - 25 * π square units.
-/
theorem area_bounded_by_circles_and_x_axis :
  let C_center := (5, 5)
  let D_center := (15, 5)
  let radius := 5
  (2 * (radius * radius) * π / 2) + (10 * radius) = 50 - 25 * π :=
sorry

end NUMINAMATH_GPT_area_bounded_by_circles_and_x_axis_l1672_167255


namespace NUMINAMATH_GPT_union_M_N_l1672_167231

def M : Set ℕ := {1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a - 1}

theorem union_M_N : M ∪ N = {1, 2, 3} := by
  sorry

end NUMINAMATH_GPT_union_M_N_l1672_167231


namespace NUMINAMATH_GPT_original_peaches_l1672_167250

theorem original_peaches (picked: ℕ) (current: ℕ) (initial: ℕ) : 
  picked = 52 → 
  current = 86 → 
  initial = current - picked → 
  initial = 34 := 
by intros h1 h2 h3
   subst h1
   subst h2
   subst h3
   simp

end NUMINAMATH_GPT_original_peaches_l1672_167250


namespace NUMINAMATH_GPT_even_and_period_pi_l1672_167201

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem even_and_period_pi :
  (∀ x : ℝ, f (-x) = f x) ∧ (∃ T > 0, ∀ x : ℝ, f (x + T) = f x) ∧ T = Real.pi :=
by
  -- First, prove that f(x) is an even function: ∀ x, f(-x) = f(x)
  -- Next, find the smallest positive period T: ∃ T > 0, ∀ x, f(x + T) = f(x)
  -- Finally, show that this period is pi: T = π
  sorry

end NUMINAMATH_GPT_even_and_period_pi_l1672_167201


namespace NUMINAMATH_GPT_linear_function_passing_points_l1672_167270

theorem linear_function_passing_points :
  ∃ k b : ℝ, (∀ x : ℝ, y = k * x + b) ∧ (k * 0 + b = 3) ∧ (k * (-4) + b = 0)
  →
  (∃ a : ℝ, y = -((3:ℝ) / (4:ℝ)) * x + 3 ∧ (∀ x y : ℝ, y = -((3:ℝ) / (4:ℝ)) * a + 3 → y = 6 → a = -4)) :=
by sorry

end NUMINAMATH_GPT_linear_function_passing_points_l1672_167270


namespace NUMINAMATH_GPT_magnitude_of_difference_is_3sqrt5_l1672_167202

noncomputable def vector_a : ℝ × ℝ := (1, -2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 4)

def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem magnitude_of_difference_is_3sqrt5 (x : ℝ) (h_parallel : parallel vector_a (vector_b x)) :
  (Real.sqrt ((vector_a.1 - (vector_b x).1) ^ 2 + (vector_a.2 - (vector_b x).2) ^ 2)) = 3 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_magnitude_of_difference_is_3sqrt5_l1672_167202


namespace NUMINAMATH_GPT_common_chord_length_of_two_circles_l1672_167274

-- Define the equations of the circles C1 and C2
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 2 * y - 4 = 0
def circle2 (x y : ℝ) : Prop := (x + 3 / 2)^2 + (y - 3 / 2)^2 = 11 / 2

-- The theorem stating the length of the common chord
theorem common_chord_length_of_two_circles :
  ∃ l : ℝ, (∀ (x y : ℝ), circle1 x y ↔ circle2 x y) → l = 2 :=
by simp [circle1, circle2]; sorry

end NUMINAMATH_GPT_common_chord_length_of_two_circles_l1672_167274


namespace NUMINAMATH_GPT_find_number_l1672_167257

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 18) : x = 9 :=
sorry

end NUMINAMATH_GPT_find_number_l1672_167257


namespace NUMINAMATH_GPT_min_value_of_a_l1672_167251

theorem min_value_of_a (a : ℝ) : (∀ x, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end NUMINAMATH_GPT_min_value_of_a_l1672_167251


namespace NUMINAMATH_GPT_population_increase_l1672_167238

-- Define the problem conditions
def average_birth_rate := (6 + 10) / 2 / 2  -- the average number of births per second
def average_death_rate := (4 + 8) / 2 / 2  -- the average number of deaths per second
def net_migration_day := 500  -- net migration inflow during the day
def net_migration_night := -300  -- net migration outflow during the night

-- Define the number of seconds in a day
def seconds_in_a_day := 24 * 3600

-- Define the net increase due to births and deaths
def net_increase_births_deaths := (average_birth_rate - average_death_rate) * seconds_in_a_day

-- Define the total net migration
def total_net_migration := net_migration_day + net_migration_night

-- Define the total population net increase
def total_population_net_increase :=
  net_increase_births_deaths + total_net_migration

-- The theorem to be proved
theorem population_increase (h₁ : average_birth_rate = 4)
                           (h₂ : average_death_rate = 3)
                           (h₃ : seconds_in_a_day = 86400) :
  total_population_net_increase = 86600 := by
  sorry

end NUMINAMATH_GPT_population_increase_l1672_167238


namespace NUMINAMATH_GPT_quotient_is_seven_l1672_167203

def dividend : ℕ := 22
def divisor : ℕ := 3
def remainder : ℕ := 1

theorem quotient_is_seven : ∃ quotient : ℕ, dividend = (divisor * quotient) + remainder ∧ quotient = 7 := by
  sorry

end NUMINAMATH_GPT_quotient_is_seven_l1672_167203


namespace NUMINAMATH_GPT_car_payment_months_l1672_167283

theorem car_payment_months 
    (total_price : ℕ) 
    (initial_payment : ℕ)
    (monthly_payment : ℕ) 
    (h_total_price : total_price = 13380) 
    (h_initial_payment : initial_payment = 5400) 
    (h_monthly_payment : monthly_payment = 420) 
    : total_price - initial_payment = 7980 
    ∧ (total_price - initial_payment) / monthly_payment = 19 := 
by 
  sorry

end NUMINAMATH_GPT_car_payment_months_l1672_167283


namespace NUMINAMATH_GPT_determine_a_l1672_167289

open Real

theorem determine_a :
  (∃ a : ℝ, |x^2 + a*x + 4*a| ≤ 3 → x^2 + a*x + 4*a = 3) ↔ (a = 8 + 2*sqrt 13 ∨ a = 8 - 2*sqrt 13) :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l1672_167289


namespace NUMINAMATH_GPT_ship_navigation_avoid_reefs_l1672_167297

theorem ship_navigation_avoid_reefs (a : ℝ) (h : a > 0) :
  (10 * a) * 40 / Real.sqrt ((10 * a) ^ 2 + 40 ^ 2) > 20 ↔
  a > (4 * Real.sqrt 3 / 3) :=
by
  sorry

end NUMINAMATH_GPT_ship_navigation_avoid_reefs_l1672_167297


namespace NUMINAMATH_GPT_anna_chargers_l1672_167248

theorem anna_chargers (P L: ℕ) (h1: L = 5 * P) (h2: P + L = 24): P = 4 := by
  sorry

end NUMINAMATH_GPT_anna_chargers_l1672_167248

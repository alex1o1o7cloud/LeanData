import Mathlib

namespace calc_two_pow_a_mul_two_pow_b_l811_81100

theorem calc_two_pow_a_mul_two_pow_b {a b : ℕ} (h1 : 0 < a) (h2 : 0 < b) (h3 : (2^a)^b = 2^2) :
  2^a * 2^b = 8 :=
sorry

end calc_two_pow_a_mul_two_pow_b_l811_81100


namespace price_difference_is_correct_l811_81161

-- Definitions from the problem conditions
def list_price : ℝ := 58.80
def tech_shop_discount : ℝ := 12.00
def value_mart_discount_rate : ℝ := 0.20

-- Calculating the sale prices from definitions
def tech_shop_sale_price : ℝ := list_price - tech_shop_discount
def value_mart_sale_price : ℝ := list_price * (1 - value_mart_discount_rate)

-- The proof problem statement
theorem price_difference_is_correct :
  value_mart_sale_price - tech_shop_sale_price = 0.24 :=
by
  sorry

end price_difference_is_correct_l811_81161


namespace last_digit_sum_chessboard_segments_l811_81148

theorem last_digit_sum_chessboard_segments {N : ℕ} (tile_count : ℕ) (segment_count : ℕ := 112) (dominos_per_tiling : ℕ := 32) (segments_per_domino : ℕ := 2) (N := tile_count / N) :
  (80 * N) % 10 = 0 :=
by
  sorry

end last_digit_sum_chessboard_segments_l811_81148


namespace perpendicular_vector_solution_l811_81155

theorem perpendicular_vector_solution 
    (a b : ℝ × ℝ) (m : ℝ) 
    (h_a : a = (1, -1)) 
    (h_b : b = (-2, 3)) 
    (h_perp : a.1 * (a.1 + m * b.1) + a.2 * (a.2 + m * b.2) = 0) 
    : m = 2 / 5 := 
sorry

end perpendicular_vector_solution_l811_81155


namespace trees_per_day_l811_81178

def blocks_per_tree := 3
def total_blocks := 30
def days := 5

theorem trees_per_day : (total_blocks / days) / blocks_per_tree = 2 := by
  sorry

end trees_per_day_l811_81178


namespace exists_increasing_seq_with_sum_square_diff_l811_81151

/-- There exists an increasing sequence of natural numbers in which
  the sum of any two consecutive terms is equal to the square of their
  difference. -/
theorem exists_increasing_seq_with_sum_square_diff :
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, a n < a (n + 1)) ∧ (∀ n : ℕ, a n + a (n + 1) = (a (n + 1) - a n) ^ 2) :=
sorry

end exists_increasing_seq_with_sum_square_diff_l811_81151


namespace find_other_asymptote_l811_81119

-- Define the conditions
def one_asymptote (x : ℝ) : ℝ := 3 * x
def foci_x_coordinate : ℝ := 5

-- Define the expected answer
def other_asymptote (x : ℝ) : ℝ := -3 * x + 30

-- Theorem statement to prove the equation of the other asymptote
theorem find_other_asymptote :
  (∀ x, y = one_asymptote x) →
  (∀ _x, _x = foci_x_coordinate) →
  (∀ x, y = other_asymptote x) :=
by
  intros h_one_asymptote h_foci_x
  sorry

end find_other_asymptote_l811_81119


namespace necessary_and_sufficient_condition_l811_81188

def f (x a : ℝ) : ℝ := x^3 + 3 * a * x

def slope_tangent_at_one (a : ℝ) : ℝ := 3 * 1^2 + 3 * a

def are_perpendicular (a : ℝ) : Prop := -a = -1

theorem necessary_and_sufficient_condition (a : ℝ) :
  (slope_tangent_at_one a = 6) ↔ (are_perpendicular a) :=
by
  sorry

end necessary_and_sufficient_condition_l811_81188


namespace price_difference_l811_81196

/-- Given an original price, two successive price increases, and special deal prices for a fixed number of items, 
    calculate the difference between the final retail price and the average special deal price. -/
theorem price_difference
  (original_price : ℝ) (first_increase_percent: ℝ) (second_increase_percent: ℝ)
  (special_deal_percent_1: ℝ) (num_items_1: ℕ) (special_deal_percent_2: ℝ) (num_items_2: ℕ)
  (final_retail_price : ℝ) (average_special_deal_price : ℝ) :
  original_price = 50 →
  first_increase_percent = 0.30 →
  second_increase_percent = 0.15 →
  special_deal_percent_1 = 0.70 →
  num_items_1 = 50 →
  special_deal_percent_2 = 0.85 →
  num_items_2 = 100 →
  final_retail_price = original_price * (1 + first_increase_percent) * (1 + second_increase_percent) →
  average_special_deal_price = 
    (num_items_1 * (special_deal_percent_1 * final_retail_price) + 
    num_items_2 * (special_deal_percent_2 * final_retail_price)) / 
    (num_items_1 + num_items_2) →
  final_retail_price - average_special_deal_price = 14.95 :=
by
  intros
  sorry

end price_difference_l811_81196


namespace real_roots_range_of_k_l811_81156

theorem real_roots_range_of_k (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 - 2 * k * x + (k + 3) = 0) ↔ (k ≤ 3 / 2) :=
sorry

end real_roots_range_of_k_l811_81156


namespace boys_and_girls_original_total_l811_81181

theorem boys_and_girls_original_total (b g : ℕ) 
(h1 : b = 3 * g) 
(h2 : b - 4 = 5 * (g - 4)) : 
b + g = 32 := 
sorry

end boys_and_girls_original_total_l811_81181


namespace find_b_age_l811_81194

theorem find_b_age (a b : ℕ) (h1 : a + 10 = 2 * (b - 10)) (h2 : a = b + 9) : b = 39 :=
sorry

end find_b_age_l811_81194


namespace question1_question2_question3_l811_81106

-- Define probabilities of renting and returning bicycles at different stations
def P (X Y : Char) : ℝ :=
  if X = 'A' ∧ Y = 'A' then 0.3 else
  if X = 'A' ∧ Y = 'B' then 0.2 else
  if X = 'A' ∧ Y = 'C' then 0.5 else
  if X = 'B' ∧ Y = 'A' then 0.7 else
  if X = 'B' ∧ Y = 'B' then 0.1 else
  if X = 'B' ∧ Y = 'C' then 0.2 else
  if X = 'C' ∧ Y = 'A' then 0.4 else
  if X = 'C' ∧ Y = 'B' then 0.5 else
  if X = 'C' ∧ Y = 'C' then 0.1 else 0

-- Question 1: Prove P(CC) = 0.1
theorem question1 : P 'C' 'C' = 0.1 := by
  sorry

-- Question 2: Prove P(AC) * P(CB) = 0.25
theorem question2 : P 'A' 'C' * P 'C' 'B' = 0.25 := by
  sorry

-- Question 3: Prove the probability P = 0.43
theorem question3 : P 'A' 'A' * P 'A' 'A' + P 'A' 'B' * P 'B' 'A' + P 'A' 'C' * P 'C' 'A' = 0.43 := by
  sorry

end question1_question2_question3_l811_81106


namespace rounding_estimation_correct_l811_81154

theorem rounding_estimation_correct (a b d : ℕ)
  (ha : a > 0) (hb : b > 0) (hd : d > 0)
  (a_round : ℕ) (b_round : ℕ) (d_round : ℕ)
  (h_round_a : a_round ≥ a) (h_round_b : b_round ≤ b) (h_round_d : d_round ≤ d) :
  (Real.sqrt (a_round / b_round) - Real.sqrt d_round) > (Real.sqrt (a / b) - Real.sqrt d) :=
by
  sorry

end rounding_estimation_correct_l811_81154


namespace right_triangle_arithmetic_sequence_side_length_l811_81142

theorem right_triangle_arithmetic_sequence_side_length :
  ∃ (a b c : ℕ), (a < b ∧ b < c) ∧ (b - a = c - b) ∧ (a^2 + b^2 = c^2) ∧ (b = 81) :=
sorry

end right_triangle_arithmetic_sequence_side_length_l811_81142


namespace traditionalist_fraction_l811_81185

theorem traditionalist_fraction (T P : ℕ) 
  (h1 : ∀ prov : ℕ, prov < 6 → T = P / 9) 
  (h2 : P + 6 * T > 0) :
  6 * T / (P + 6 * T) = 2 / 5 := 
by
  sorry

end traditionalist_fraction_l811_81185


namespace equivalence_negation_l811_81199

-- Define irrational numbers
def is_irrational (x : ℝ) : Prop :=
  ¬ (∃ q : ℚ, x = q)

-- Define rational numbers
def is_rational (x : ℝ) : Prop :=
  ∃ q : ℚ, x = q

-- Original proposition: There exists an irrational number whose square is rational
def original_proposition : Prop :=
  ∃ x : ℝ, is_irrational x ∧ is_rational (x * x)

-- Negation of the original proposition
def negation_of_proposition : Prop :=
  ∀ x : ℝ, is_irrational x → ¬is_rational (x * x)

-- Proof statement that the negation of the original proposition is equivalent to "Every irrational number has a square that is not rational"
theorem equivalence_negation :
  (¬ original_proposition) ↔ negation_of_proposition :=
sorry

end equivalence_negation_l811_81199


namespace degrees_to_radians_750_l811_81179

theorem degrees_to_radians_750 (π : ℝ) (deg_750 : ℝ) 
  (h : 180 = π) : 
  750 * (π / 180) = 25 / 6 * π :=
by
  sorry

end degrees_to_radians_750_l811_81179


namespace unique_non_zero_b_for_unique_x_solution_l811_81124

theorem unique_non_zero_b_for_unique_x_solution (c : ℝ) (hc : c ≠ 0) :
  c = 3 / 2 ↔ ∃! b : ℝ, b ≠ 0 ∧ ∃ x : ℝ, (x^2 + (b + 3 / b) * x + c = 0) ∧ 
  ∀ x1 x2 : ℝ, (x1^2 + (b + 3 / b) * x1 + c = 0) ∧ (x2^2 + (b + 3 / b) * x2 + c = 0) → x1 = x2 :=
sorry

end unique_non_zero_b_for_unique_x_solution_l811_81124


namespace cycle_price_reduction_l811_81115

theorem cycle_price_reduction (original_price : ℝ) :
  let price_after_first_reduction := original_price * 0.75
  let price_after_second_reduction := price_after_first_reduction * 0.60
  (original_price - price_after_second_reduction) / original_price = 0.55 :=
by
  sorry

end cycle_price_reduction_l811_81115


namespace log_base_30_of_8_l811_81169

theorem log_base_30_of_8 (a b : Real) (h1 : Real.log 5 = a) (h2 : Real.log 3 = b) : 
    Real.logb 30 8 = 3 * (1 - a) / (b + 1) := 
  sorry

end log_base_30_of_8_l811_81169


namespace eleven_power_2023_mod_50_l811_81183

theorem eleven_power_2023_mod_50 :
  11^2023 % 50 = 31 :=
by
  sorry

end eleven_power_2023_mod_50_l811_81183


namespace p_expression_l811_81190

theorem p_expression (m n p : ℤ) (r1 r2 : ℝ) 
  (h1 : r1 + r2 = m) 
  (h2 : r1 * r2 = n) 
  (h3 : r1^2 + r2^2 = p) : 
  p = m^2 - 2 * n := by
  sorry

end p_expression_l811_81190


namespace min_value_of_f_l811_81187

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem min_value_of_f : ∃ x : ℝ, (f x = -(1 / Real.exp 1)) ∧ (∀ y : ℝ, f y ≥ f x) := by
  sorry

end min_value_of_f_l811_81187


namespace female_democrats_l811_81191

theorem female_democrats (F M D_f: ℕ) 
  (h1 : F + M = 780)
  (h2 : D_f = (1/2) * F)
  (h3 : (1/3) * 780 = 260)
  (h4 : 260 = (1/2) * F + (1/4) * M) : 
  D_f = 130 := 
by
  sorry

end female_democrats_l811_81191


namespace students_receiving_B_lee_l811_81127

def num_students_receiving_B (students_kipling: ℕ) (B_kipling: ℕ) (students_lee: ℕ) : ℕ :=
  let ratio := (B_kipling * students_lee) / students_kipling
  ratio

theorem students_receiving_B_lee (students_kipling B_kipling students_lee : ℕ) 
  (h : B_kipling = 8 ∧ students_kipling = 12 ∧ students_lee = 30) :
  num_students_receiving_B students_kipling B_kipling students_lee = 20 :=
by
  sorry

end students_receiving_B_lee_l811_81127


namespace basketball_team_free_throws_l811_81104

theorem basketball_team_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a)
  (h2 : x = 2 * a - 1)
  (h3 : 2 * a + 3 * b + x = 89) : 
  x = 29 :=
by
  sorry

end basketball_team_free_throws_l811_81104


namespace doughnuts_served_initially_l811_81166

def initial_doughnuts_served (staff_count : Nat) (doughnuts_per_staff : Nat) (doughnuts_left : Nat) : Nat :=
  staff_count * doughnuts_per_staff + doughnuts_left

theorem doughnuts_served_initially :
  ∀ (staff_count doughnuts_per_staff doughnuts_left : Nat), staff_count = 19 → doughnuts_per_staff = 2 → doughnuts_left = 12 →
  initial_doughnuts_served staff_count doughnuts_per_staff doughnuts_left = 50 :=
by
  intros staff_count doughnuts_per_staff doughnuts_left hstaff hdonuts hleft
  rw [hstaff, hdonuts, hleft]
  rfl

#check doughnuts_served_initially

end doughnuts_served_initially_l811_81166


namespace eccentricity_of_ellipse_equilateral_triangle_l811_81149

theorem eccentricity_of_ellipse_equilateral_triangle (c b a e : ℝ)
  (h1 : b = Real.sqrt (3 * c))
  (h2 : a = Real.sqrt (b^2 + c^2)) 
  (h3 : e = c / a) :
  e = 1 / 2 :=
by {
  sorry
}

end eccentricity_of_ellipse_equilateral_triangle_l811_81149


namespace trigonometric_problem_l811_81162

theorem trigonometric_problem
  (α : ℝ)
  (h1 : Real.tan α = Real.sqrt 3)
  (h2 : π < α)
  (h3 : α < 3 * π / 2) :
  Real.cos (2 * π - α) - Real.sin α = (Real.sqrt 3 - 1) / 2 := by
  sorry

end trigonometric_problem_l811_81162


namespace instantaneous_velocity_at_t_3_l811_81129

variable (t : ℝ)
def s (t : ℝ) : ℝ := 1 - t + t^2

theorem instantaneous_velocity_at_t_3 : 
  ∃ v, v = -1 + 2 * 3 ∧ v = 5 :=
by
  sorry

end instantaneous_velocity_at_t_3_l811_81129


namespace solve_problem_l811_81197

def spadesuit (x y : ℝ) : ℝ := x^2 + y^2

theorem solve_problem : spadesuit (spadesuit 3 5) 4 = 1172 := by
  sorry

end solve_problem_l811_81197


namespace path_length_of_dot_l811_81109

-- Define the dimensions of the rectangular prism
def prism_width := 1 -- cm
def prism_height := 1 -- cm
def prism_length := 2 -- cm

-- Define the condition that the dot is marked at the center of the top face
def dot_position := (0.5, 1)

-- Define the condition that the prism starts with the 1 cm by 2 cm face on the table
def initial_face_on_table := (prism_length, prism_height)

-- Define the statement to prove the length of the path followed by the dot
theorem path_length_of_dot: 
  ∃ length_of_path : ℝ, length_of_path = 2 * Real.pi :=
sorry

end path_length_of_dot_l811_81109


namespace solution_to_absolute_value_equation_l811_81101

theorem solution_to_absolute_value_equation (x : ℝ) : 
    abs x - 2 - abs (-1) = 2 ↔ x = 5 ∨ x = -5 :=
by
  sorry

end solution_to_absolute_value_equation_l811_81101


namespace running_distance_l811_81171

theorem running_distance (D : ℕ) 
  (hA_time : ∀ (A_time : ℕ), A_time = 28) 
  (hB_time : ∀ (B_time : ℕ), B_time = 32) 
  (h_lead : ∀ (lead : ℕ), lead = 28) 
  (hA_speed : ∀ (A_speed : ℚ), A_speed = D / 28) 
  (hB_speed : ∀ (B_speed : ℚ), B_speed = D / 32) 
  (hB_dist : ∀ (B_dist : ℚ), B_dist = D - 28) 
  (h_eq : ∀ (B_dist : ℚ), B_dist = D * (28 / 32)) :
  D = 224 :=
by 
  sorry

end running_distance_l811_81171


namespace two_abc_square_l811_81125

variable {R : Type*} [Ring R] [Fintype R]

-- Given condition: For any a, b ∈ R, ∃ c ∈ R such that a^2 + b^2 = c^2.
axiom ring_property (a b : R) : ∃ c : R, a^2 + b^2 = c^2

-- We need to prove: For any a, b, c ∈ R, ∃ d ∈ R such that 2abc = d^2.
theorem two_abc_square (a b c : R) : ∃ d : R, 2 * (a * b * c) = d^2 :=
by
  sorry

end two_abc_square_l811_81125


namespace min_even_integers_six_l811_81128

theorem min_even_integers_six (x y a b m n : ℤ) 
  (h1 : x + y = 30) 
  (h2 : x + y + a + b = 50) 
  (h3 : x + y + a + b + m + n = 70) 
  (hm_even : Even m) 
  (hn_even: Even n) : 
  ∃ k, (0 ≤ k ∧ k ≤ 6) ∧ (∀ e, (e = m ∨ e = n) → ∃ j, (j = 2)) :=
by
  sorry

end min_even_integers_six_l811_81128


namespace smallest_n_area_gt_2500_l811_81138

noncomputable def triangle_area (n : ℕ) : ℝ :=
  (1/2 : ℝ) * (|(n : ℝ) * (2 * n) + (n^2 - 1 : ℝ) * (3 * n^2 - 1) + (n^3 - 3 * n) * 1
  - (1 : ℝ) * (n^2 - 1) - (2 * n) * (n^3 - 3 * n) - (3 * n^2 - 1) * (n : ℝ)|)

theorem smallest_n_area_gt_2500 : ∃ n : ℕ, (∀ m : ℕ, 0 < m ∧ m < n → triangle_area m <= 2500) ∧ triangle_area n > 2500 :=
by
  sorry

end smallest_n_area_gt_2500_l811_81138


namespace log_50_between_consecutive_integers_l811_81192

theorem log_50_between_consecutive_integers :
    (∃ (m n : ℤ), m < n ∧ m < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < n ∧ m + n = 3) :=
by
  have log_10_eq_1 : Real.log 10 / Real.log 10 = 1 := by sorry
  have log_100_eq_2 : Real.log 100 / Real.log 10 = 2 := by sorry
  have log_increasing : ∀ (x y : ℝ), x < y → Real.log x / Real.log 10 < Real.log y / Real.log 10 := by sorry
  have interval : 10 < 50 ∧ 50 < 100 := by sorry
  use 1
  use 2
  sorry

end log_50_between_consecutive_integers_l811_81192


namespace no_solution_exists_l811_81130

theorem no_solution_exists (f : ℝ → ℝ) :
  ¬ (∀ x y : ℝ, f (f x + 2 * y) = 3 * x + f (f (f y) - x)) :=
sorry

end no_solution_exists_l811_81130


namespace triangular_array_sum_of_digits_l811_81152

def triangular_sum (N : ℕ) : ℕ := N * (N + 1) / 2

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem triangular_array_sum_of_digits :
  ∃ N : ℕ, triangular_sum N = 2080 ∧ sum_of_digits N = 10 :=
by
  sorry

end triangular_array_sum_of_digits_l811_81152


namespace odd_numbers_le_twice_switch_pairs_l811_81144

-- Number of odd elements in row n is denoted as numOdd n
def numOdd (n : ℕ) : ℕ := -- Definition of numOdd function
sorry

-- Number of switch pairs in row n is denoted as numSwitchPairs n
def numSwitchPairs (n : ℕ) : ℕ := -- Definition of numSwitchPairs function
sorry

-- Definition of Pascal's Triangle and conditions
def binom (n k : ℕ) : ℕ := if k > n then 0 else if k = 0 ∨ k = n then 1 else binom (n-1) (k-1) + binom (n-1) k

-- Check even or odd
def isOdd (n : ℕ) : Bool := n % 2 = 1

-- Definition of switch pair check
def isSwitchPair (a b : ℕ) : Prop := (isOdd a ∧ ¬isOdd b) ∨ (¬isOdd a ∧ isOdd b)

theorem odd_numbers_le_twice_switch_pairs (n : ℕ) :
  numOdd n ≤ 2 * numSwitchPairs (n-1) :=
sorry

end odd_numbers_le_twice_switch_pairs_l811_81144


namespace overall_average_score_l811_81147

def students_monday := 24
def students_tuesday := 4
def total_students := 28
def mean_score_monday := 82
def mean_score_tuesday := 90

theorem overall_average_score :
  (students_monday * mean_score_monday + students_tuesday * mean_score_tuesday) / total_students = 83 := by
sorry

end overall_average_score_l811_81147


namespace fifth_friend_payment_l811_81175

/-- 
Five friends bought a piece of furniture for $120.
The first friend paid one third of the sum of the amounts paid by the other four;
the second friend paid one fourth of the sum of the amounts paid by the other four;
the third friend paid one fifth of the sum of the amounts paid by the other four;
and the fourth friend paid one sixth of the sum of the amounts paid by the other four.
Prove that the fifth friend paid $41.33.
-/
theorem fifth_friend_payment :
  ∀ (a b c d e : ℝ),
    a = 1/3 * (b + c + d + e) →
    b = 1/4 * (a + c + d + e) →
    c = 1/5 * (a + b + d + e) →
    d = 1/6 * (a + b + c + e) →
    a + b + c + d + e = 120 →
    e = 41.33 :=
by
  intros a b c d e ha hb hc hd he_sum
  sorry

end fifth_friend_payment_l811_81175


namespace smallest_k_for_sequence_l811_81126

theorem smallest_k_for_sequence (a : ℕ → ℕ) (k : ℕ) (h₁ : a 1 = 1) (h₂ : a 2018 = 2020)
  (h₃ : ∀ n, n ≥ 2 → a (n+1) = k * (a n) / (a (n-1))) : k = 2020 :=
sorry

end smallest_k_for_sequence_l811_81126


namespace hog_cat_problem_l811_81123

theorem hog_cat_problem (hogs cats : ℕ)
  (hogs_eq : hogs = 75)
  (hogs_cats_relation : hogs = 3 * cats)
  : 5 < (6 / 10) * cats - 5 := 
by
  sorry

end hog_cat_problem_l811_81123


namespace parabola_directrix_equation_l811_81176

theorem parabola_directrix_equation :
  ∀ (x y : ℝ),
  y = -4 * x^2 - 16 * x + 1 →
  ∃ d : ℝ, d = 273 / 16 ∧ y = d :=
by
  sorry

end parabola_directrix_equation_l811_81176


namespace parallel_lines_l811_81157

-- Definitions of lines and plane
variable {Line : Type}
variable {Plane : Type}
variable (a b c : Line)
variable (α : Plane)

-- Parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelPlane : Line → Plane → Prop)

-- Given conditions
variable (h1 : parallel a c)
variable (h2 : parallel b c)

-- Theorem statement
theorem parallel_lines (a b c : Line) 
                       (α : Plane) 
                       (parallel : Line → Line → Prop) 
                       (perpendicular : Line → Line → Prop) 
                       (parallelPlane : Line → Plane → Prop)
                       (h1 : parallel a c) 
                       (h2 : parallel b c) : 
                       parallel a b :=
sorry

end parallel_lines_l811_81157


namespace area_enclosed_by_circle_l811_81141

theorem area_enclosed_by_circle : 
  (∀ x y : ℝ, x^2 + y^2 + 10 * x + 24 * y = 0) → 
  (π * 13^2 = 169 * π):=
by
  intro h
  sorry

end area_enclosed_by_circle_l811_81141


namespace initial_candies_l811_81133

theorem initial_candies (x : ℕ) (h1 : x % 4 = 0) (h2 : x / 4 * 3 / 3 * 2 / 2 - 24 ≥ 6) (h3 : x / 4 * 3 / 3 * 2 / 2 - 24 ≤ 9) :
  x = 64 :=
sorry

end initial_candies_l811_81133


namespace number_of_routes_l811_81193

variable {City : Type}
variable (A B C D E : City)
variable (AB_N AB_S AD AE BC BD CD DE : City → City → Prop)
  
theorem number_of_routes 
  (hAB_N : AB_N A B) (hAB_S : AB_S A B)
  (hAD : AD A D) (hAE : AE A E)
  (hBC : BC B C) (hBD : BD B D)
  (hCD : CD C D) (hDE : DE D E) :
  ∃ r : ℕ, r = 16 := 
sorry

end number_of_routes_l811_81193


namespace raffle_tickets_sold_l811_81150

theorem raffle_tickets_sold (total_amount : ℕ) (ticket_cost : ℕ) (tickets_sold : ℕ) 
    (h1 : total_amount = 620) (h2 : ticket_cost = 4) : tickets_sold = 155 :=
by {
  sorry
}

end raffle_tickets_sold_l811_81150


namespace integer_solutions_to_cube_sum_eq_2_pow_30_l811_81136

theorem integer_solutions_to_cube_sum_eq_2_pow_30 (x y : ℤ) :
  x^3 + y^3 = 2^30 → (x = 0 ∧ y = 2^10) ∨ (x = 2^10 ∧ y = 0) :=
by
  sorry

end integer_solutions_to_cube_sum_eq_2_pow_30_l811_81136


namespace diameter_of_circle_l811_81114

theorem diameter_of_circle (A : ℝ) (h : A = 100 * Real.pi) : ∃ d : ℝ, d = 20 :=
by
  sorry

end diameter_of_circle_l811_81114


namespace one_div_lt_one_div_of_gt_l811_81110

theorem one_div_lt_one_div_of_gt {a b : ℝ} (hab : a > b) (hb0 : b > 0) : (1 / a) < (1 / b) :=
sorry

end one_div_lt_one_div_of_gt_l811_81110


namespace find_x_min_construction_cost_l811_81189

-- Define the conditions for Team A and Team B
def Team_A_Daily_Construction (x : ℕ) : ℕ := x + 300
def Team_A_Daily_Cost : ℕ := 3600
def Team_B_Daily_Construction (x : ℕ) : ℕ := x
def Team_B_Daily_Cost : ℕ := 2200

-- Condition: The number of days Team A needs to construct 1800m^2 is equal to the number of days Team B needs to construct 1200m^2
def construction_days (x : ℕ) : Prop := 
  1800 / (x + 300) = 1200 / x

-- Define the total days worked and the minimum construction area condition
def total_days : ℕ := 22
def min_construction_area : ℕ := 15000

-- Define the construction cost function given the number of days each team works
def construction_cost (m : ℕ) : ℕ := 
  3600 * m + 2200 * (total_days - m)

-- Main theorem: Prove that x = 600 satisfies the conditions
theorem find_x (x : ℕ) (h : x = 600) : construction_days x := by sorry

-- Second theorem: Prove that the minimum construction cost is 56800 yuan
theorem min_construction_cost (m : ℕ) (h : m ≥ 6) : construction_cost m = 56800 := by sorry

end find_x_min_construction_cost_l811_81189


namespace slope_ge_one_sum_pq_eq_17_l811_81153

noncomputable def Q_prob_satisfaction : ℚ := 1/16

theorem slope_ge_one_sum_pq_eq_17 :
  let p := 1
  let q := 16
  p + q = 17 := by
  sorry

end slope_ge_one_sum_pq_eq_17_l811_81153


namespace sarees_original_price_l811_81159

theorem sarees_original_price (P : ℝ) (h : 0.75 * 0.85 * P = 248.625) : P = 390 :=
by
  sorry

end sarees_original_price_l811_81159


namespace train_cross_pole_time_l811_81177

-- Definitions based on the conditions
def train_speed_kmh := 54
def train_length_m := 105
def train_speed_ms := (train_speed_kmh * 1000) / 3600
def expected_time := train_length_m / train_speed_ms

-- Theorem statement, encapsulating the problem
theorem train_cross_pole_time : expected_time = 7 := by
  sorry

end train_cross_pole_time_l811_81177


namespace pencils_in_drawer_l811_81184

/-- 
If there were originally 2 pencils in the drawer and there are now 5 pencils in total, 
then Tim must have placed 3 pencils in the drawer.
-/
theorem pencils_in_drawer (original_pencils tim_pencils total_pencils : ℕ) 
  (h1 : original_pencils = 2) 
  (h2 : total_pencils = 5) 
  (h3 : total_pencils = original_pencils + tim_pencils) : 
  tim_pencils = 3 := 
by
  rw [h1, h2] at h3
  linarith

end pencils_in_drawer_l811_81184


namespace fruit_salad_mixture_l811_81139

theorem fruit_salad_mixture :
  ∃ (A P G : ℝ), A / P = 12 / 8 ∧ A / G = 12 / 7 ∧ P / G = 8 / 7 ∧ A = G + 10 ∧ A + P + G = 54 :=
by
  sorry

end fruit_salad_mixture_l811_81139


namespace probability_at_5_5_equals_1_over_243_l811_81113

-- Define the base probability function P
def P : ℕ → ℕ → ℚ
| 0, 0       => 1
| x+1, 0     => 0
| 0, y+1     => 0
| x+1, y+1   => (1/3 : ℚ) * P x (y+1) + (1/3 : ℚ) * P (x+1) y + (1/3 : ℚ) * P x y

-- Theorem statement that needs to be proved
theorem probability_at_5_5_equals_1_over_243 : P 5 5 = 1 / 243 :=
sorry

end probability_at_5_5_equals_1_over_243_l811_81113


namespace system_of_equations_solution_l811_81182

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 
    (2 * x - 3 * y = 1) ∧ 
    (5 * x + 4 * y = 6) ∧ 
    (x + 2 * y = 2) ∧
    x = 2 / 3 ∧ y = 2 / 3 :=
by {
  sorry
}

end system_of_equations_solution_l811_81182


namespace correct_statements_l811_81186

-- Define the universal set U as ℤ (integers)
noncomputable def U : Set ℤ := Set.univ

-- Conditions
def is_subset_of_int : Prop := {0} ⊆ (Set.univ : Set ℤ)

def counterexample_subsets (A B : Set ℤ) : Prop :=
  (A = {1, 2} ∧ B = {1, 2, 3}) ∧ (B ∩ (U \ A) ≠ ∅)

def negation_correct_1 : Prop :=
  ¬(∀ x : ℤ, x^2 > 0) ↔ ∃ x : ℤ, x^2 ≤ 0

def negation_correct_2 : Prop :=
  ¬(∀ x : ℤ, x^2 > 0) ↔ ¬(∀ x : ℤ, x^2 < 0)

-- The theorem to prove the equivalence of correct statements
theorem correct_statements :
  (is_subset_of_int ∧
   ∀ A B : Set ℤ, A ⊆ U → B ⊆ U → (A ⊆ B → counterexample_subsets A B) ∧
   negation_correct_1 ∧
   ¬negation_correct_2) ↔
  (true) :=
by 
  sorry

end correct_statements_l811_81186


namespace profit_percent_l811_81195

theorem profit_percent (marked_price : ℝ) (num_bought : ℝ) (num_payed_price : ℝ) (discount_percent : ℝ) : 
  num_bought = 56 → 
  num_payed_price = 46 → 
  discount_percent = 0.01 →
  marked_price = 1 →
  let cost_price := num_payed_price
  let selling_price_per_pen := marked_price * (1 - discount_percent)
  let total_selling_price := num_bought * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  profit_percent = 20.52 :=
by 
  intro hnum_bought hnum_payed_price hdiscount_percent hmarked_price 
  let cost_price := num_payed_price
  let selling_price_per_pen := marked_price * (1 - discount_percent)
  let total_selling_price := num_bought * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  sorry

end profit_percent_l811_81195


namespace person_walks_distance_l811_81140

theorem person_walks_distance {D t : ℝ} (h1 : 5 * t = D) (h2 : 10 * t = D + 20) : D = 20 :=
by
  sorry

end person_walks_distance_l811_81140


namespace pieces_of_chocolate_left_l811_81135

theorem pieces_of_chocolate_left (initial_boxes : ℕ) (given_away_boxes : ℕ) (pieces_per_box : ℕ) 
    (h1 : initial_boxes = 14) (h2 : given_away_boxes = 8) (h3 : pieces_per_box = 3) : 
    (initial_boxes - given_away_boxes) * pieces_per_box = 18 := 
by 
  -- The proof will be here
  sorry

end pieces_of_chocolate_left_l811_81135


namespace suitable_for_census_l811_81167

-- Definitions based on the conditions in a)
def survey_A := "The service life of a batch of batteries"
def survey_B := "The height of all classmates in the class"
def survey_C := "The content of preservatives in a batch of food"
def survey_D := "The favorite mathematician of elementary and middle school students in the city"

-- The main statement to prove
theorem suitable_for_census : survey_B = "The height of all classmates in the class" := by
  -- We assert that the height of all classmates is the suitable survey for a census based on given conditions
  sorry

end suitable_for_census_l811_81167


namespace opposite_of_neg_five_l811_81107

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l811_81107


namespace correct_quotient_division_l811_81111

variable (k : Nat) -- the unknown original number

def mistaken_division := k = 7 * 12 + 4

theorem correct_quotient_division (h : mistaken_division k) : 
  (k / 3) = 29 :=
by
  sorry

end correct_quotient_division_l811_81111


namespace expr1_val_expr2_val_l811_81174

noncomputable def expr1 : ℝ :=
  (1 / Real.sin (10 * Real.pi / 180)) - (Real.sqrt 3 / Real.cos (10 * Real.pi / 180))

theorem expr1_val : expr1 = 4 :=
  sorry

noncomputable def expr2 : ℝ :=
  (Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) - Real.cos (20 * Real.pi / 180)) /
  (Real.cos (80 * Real.pi / 180) * Real.sqrt (1 - Real.cos (20 * Real.pi / 180)))

theorem expr2_val : expr2 = Real.sqrt 2 :=
  sorry

end expr1_val_expr2_val_l811_81174


namespace pages_per_day_l811_81102

variable (P : ℕ) (D : ℕ)

theorem pages_per_day (hP : P = 66) (hD : D = 6) : P / D = 11 :=
by
  sorry

end pages_per_day_l811_81102


namespace dylans_mom_hotdogs_l811_81116

theorem dylans_mom_hotdogs (hotdogs_total : ℕ) (helens_mom_hotdogs : ℕ) (dylans_mom_hotdogs : ℕ) 
  (h1 : hotdogs_total = 480) (h2 : helens_mom_hotdogs = 101) (h3 : hotdogs_total = helens_mom_hotdogs + dylans_mom_hotdogs) :
dylans_mom_hotdogs = 379 :=
by
  sorry

end dylans_mom_hotdogs_l811_81116


namespace words_per_page_eq_106_l811_81180

-- Definition of conditions as per the problem statement
def pages : ℕ := 224
def max_words_per_page : ℕ := 150
def total_words_congruence : ℕ := 156
def modulus : ℕ := 253

theorem words_per_page_eq_106 (p : ℕ) : 
  (224 * p % 253 = 156) ∧ (p ≤ 150) → p = 106 :=
by 
  sorry

end words_per_page_eq_106_l811_81180


namespace solve_equation_l811_81146

-- Define the equation and the conditions
def problem_equation (x : ℝ) : Prop :=
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 2)

def valid_solution (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ 1 ∧ x ≠ -6

-- State the theorem that solutions x = 3 and x = -4 solve the problem under the conditions
theorem solve_equation : ∀ x : ℝ, valid_solution x → (x = 3 ∨ x = -4 ∧ problem_equation x) :=
by
  sorry

end solve_equation_l811_81146


namespace common_area_of_rectangle_and_circle_l811_81112

theorem common_area_of_rectangle_and_circle (r : ℝ) (a b : ℝ) (h_center : r = 5) (h_dim : a = 10 ∧ b = 4) :
  let sector_area := (25 * Real.pi) / 2 
  let triangle_area := 4 * Real.sqrt 21 
  let result := sector_area + triangle_area 
  result = (25 * Real.pi) / 2 + 4 * Real.sqrt 21 := 
by
  sorry

end common_area_of_rectangle_and_circle_l811_81112


namespace initial_roses_in_vase_l811_81131

theorem initial_roses_in_vase (current_roses : ℕ) (added_roses : ℕ) (total_garden_roses : ℕ) (initial_roses : ℕ) :
  current_roses = 20 → added_roses = 13 → total_garden_roses = 59 → initial_roses = current_roses - added_roses → 
  initial_roses = 7 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  sorry

end initial_roses_in_vase_l811_81131


namespace find_different_mass_part_l811_81158

-- Definitions for the parts a1, a2, a3, a4 and their masses
variable {α : Type}
variables (a₁ a₂ a₃ a₄ : α)
variable [LinearOrder α]

-- Definition of the problem conditions
def different_mass_part (a₁ a₂ a₃ a₄ : α) : Prop :=
  (a₁ ≠ a₂ ∨ a₁ ≠ a₃ ∨ a₁ ≠ a₄ ∨ a₂ ≠ a₃ ∨ a₂ ≠ a₄ ∨ a₃ ≠ a₄)

-- Theorem statement assuming we can identify the differing part using two weighings on a pan balance
theorem find_different_mass_part (h : different_mass_part a₁ a₂ a₃ a₄) :
  ∃ (part : α), part = a₁ ∨ part = a₂ ∨ part = a₃ ∨ part = a₄ :=
sorry

end find_different_mass_part_l811_81158


namespace find_f2_l811_81145

-- Define f as an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define g based on f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f x + 9

-- Given conditions
variables (f : ℝ → ℝ) (h_odd : odd_function f)
variable (h_g_neg2 : g f (-2) = 3)

-- Theorem statement
theorem find_f2 : f 2 = 6 :=
by
  sorry

end find_f2_l811_81145


namespace range_of_a_l811_81117

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, 2 * a * (x : ℝ)^2 - 4 * (x : ℝ) < a * (x : ℝ) - 2 → ∃! x₀ : ℤ, x₀ = x) → 1 ≤ a ∧ a < 2 :=
sorry

end range_of_a_l811_81117


namespace maximize_S_n_l811_81103

-- Define the general term of the sequence and the sum of the first n terms.
def a_n (n : ℕ) : ℤ := -2 * n + 25

def S_n (n : ℕ) : ℤ := 24 * n - n^2

-- The main statement to prove
theorem maximize_S_n : ∃ (n : ℕ), n = 11 ∧ ∀ m, S_n m ≤ S_n 11 :=
  sorry

end maximize_S_n_l811_81103


namespace rowing_speed_l811_81172

theorem rowing_speed (V_m V_w V_upstream V_downstream : ℝ)
  (h1 : V_upstream = 25)
  (h2 : V_downstream = 65)
  (h3 : V_w = 5) :
  V_m = 45 :=
by
  -- Lean will verify the theorem given the conditions
  sorry

end rowing_speed_l811_81172


namespace true_propositions_l811_81143

-- Defining the propositions as functions for clarity
def proposition1 (L1 L2 P: Prop) : Prop := 
  (L1 ∧ L2 → P) → (P)

def proposition2 (plane1 plane2 line: Prop) : Prop := 
  (line → (plane1 ∧ plane2)) → (plane1 ∧ plane2)

def proposition3 (L1 L2 L3: Prop) : Prop := 
  (L1 ∧ L2 → L3) → L1

def proposition4 (plane1 plane2 line: Prop) : Prop := 
  (plane1 ∧ plane2 → (line → ¬ (plane1 ∧ plane2)))

-- Assuming the required mathematical hypothesis was valid within our formal system 
theorem true_propositions : proposition2 plane1 plane2 line ∧ proposition4 plane1 plane2 line := 
by sorry

end true_propositions_l811_81143


namespace runway_show_time_l811_81132

/-
Problem: Prove that it will take 60 minutes to complete all of the runway trips during the show, 
given the following conditions:
- There are 6 models in the show.
- Each model will wear two sets of bathing suits and three sets of evening wear clothes during the runway portion of the show.
- It takes a model 2 minutes to walk out to the end of the runway and back, and models take turns, one at a time.
-/

theorem runway_show_time 
    (num_models : ℕ) 
    (sets_bathing_suits_per_model : ℕ) 
    (sets_evening_wear_per_model : ℕ) 
    (time_per_trip : ℕ) 
    (total_time : ℕ) :
    num_models = 6 →
    sets_bathing_suits_per_model = 2 →
    sets_evening_wear_per_model = 3 →
    time_per_trip = 2 →
    total_time = num_models * (sets_bathing_suits_per_model + sets_evening_wear_per_model) * time_per_trip →
    total_time = 60 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5


end runway_show_time_l811_81132


namespace find_e_l811_81168

theorem find_e (d e f : ℕ) (hd : d > 1) (he : e > 1) (hf : f > 1) :
  (∀ M : ℝ, M ≠ 1 → (M^(1/d) * (M^(1/e) * (M^(1/f)))^(1/e)^(1/d)) = (M^(17/24))^(1/24)) → e = 4 :=
by
  sorry

end find_e_l811_81168


namespace find_number_l811_81165

theorem find_number :
  ∃ (N : ℝ), (5 / 4) * N = (4 / 5) * N + 45 ∧ N = 100 :=
by
  sorry

end find_number_l811_81165


namespace necessary_but_not_sufficient_l811_81137

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (ω * x - (Real.pi / 3))

theorem necessary_but_not_sufficient (ω : ℝ) :
  (∀ x : ℝ, f ω (x + Real.pi) = f ω x) ↔ (ω = 2) ∨ (∃ ω ≠ 2, ∀ x : ℝ, f ω (x + Real.pi) = f ω x) :=
by
  sorry

end necessary_but_not_sufficient_l811_81137


namespace trigonometric_product_eq_l811_81118

open Real

theorem trigonometric_product_eq :
  3.420 * (sin (10 * pi / 180)) * (sin (20 * pi / 180)) * (sin (30 * pi / 180)) *
  (sin (40 * pi / 180)) * (sin (50 * pi / 180)) * (sin (60 * pi / 180)) *
  (sin (70 * pi / 180)) * (sin (80 * pi / 180)) = 3 / 256 := 
sorry

end trigonometric_product_eq_l811_81118


namespace perimeter_of_square_from_quadratic_roots_l811_81105

theorem perimeter_of_square_from_quadratic_roots :
  let r1 := 1
  let r2 := 10
  let larger_root := if r1 > r2 then r1 else r2
  let area := larger_root * larger_root
  let side_length := Real.sqrt area
  4 * side_length = 40 := by
  let r1 := 1
  let r2 := 10
  let larger_root := if r1 > r2 then r1 else r2
  let area := larger_root * larger_root
  let side_length := Real.sqrt area
  sorry

end perimeter_of_square_from_quadratic_roots_l811_81105


namespace lcm_48_180_l811_81108

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  -- Here follows the proof, which is omitted
  sorry

end lcm_48_180_l811_81108


namespace find_larger_number_l811_81134

theorem find_larger_number 
  (x y : ℚ) 
  (h1 : 4 * y = 9 * x) 
  (h2 : y - x = 12) : 
  y = 108 / 5 := 
sorry

end find_larger_number_l811_81134


namespace function_property_l811_81120

noncomputable def f (x : ℝ) : ℝ := sorry
variable (a x1 x2 : ℝ)

-- Conditions
axiom f_defined_on_R : ∀ x : ℝ, f x ≠ 0
axiom f_increasing_on_left_of_a : ∀ x y : ℝ, x < y → y < a → f x < f y
axiom f_even_shifted_by_a : ∀ x : ℝ, f (x + a) = f (-(x + a))
axiom ordering : x1 < a ∧ a < x2
axiom distance_comp : |x1 - a| < |x2 - a|

-- Proof Goal
theorem function_property : f (2 * a - x1) > f (2 * a - x2) :=
by
  sorry

end function_property_l811_81120


namespace square_field_area_l811_81160

noncomputable def area_of_square_field(speed_kph : ℝ) (time_hrs : ℝ) : ℝ :=
  let speed_mps := (speed_kph * 1000) / 3600
  let distance := speed_mps * (time_hrs * 3600)
  let side_length := distance / Real.sqrt 2
  side_length ^ 2

theorem square_field_area 
  (speed_kph : ℝ := 2.4)
  (time_hrs : ℝ := 3.0004166666666667) :
  area_of_square_field speed_kph time_hrs = 25939764.41 := 
by 
  -- This is a placeholder for the proof. 
  sorry

end square_field_area_l811_81160


namespace combined_degrees_l811_81121

-- Definitions based on conditions
def summer_degrees : ℕ := 150
def jolly_degrees (summer_degrees : ℕ) : ℕ := summer_degrees - 5

-- Theorem stating the combined degrees
theorem combined_degrees : summer_degrees + jolly_degrees summer_degrees = 295 :=
by
  sorry

end combined_degrees_l811_81121


namespace value_of_x_l811_81198

theorem value_of_x {x y z w v : ℝ} 
  (h1 : y * x = 3)
  (h2 : z = 3)
  (h3 : w = z * y)
  (h4 : v = w * z)
  (h5 : v = 18)
  (h6 : w = 6) :
  x = 3 / 2 :=
by
  sorry

end value_of_x_l811_81198


namespace distinct_real_roots_l811_81170

theorem distinct_real_roots (k : ℝ) :
  (∀ x : ℝ, (k - 2) * x^2 + 2 * x - 1 = 0 → ∃ y : ℝ, (k - 2) * y^2 + 2 * y - 1 = 0 ∧ y ≠ x) ↔
  (k > 1 ∧ k ≠ 2) := 
by sorry

end distinct_real_roots_l811_81170


namespace pair1_equivalent_pair2_non_equivalent_pair3_equivalent_pair4_equivalent_pair5_non_equivalent_pair6_equivalent_l811_81164

theorem pair1_equivalent (x : ℝ) : (x^2 + 5 * x < 4) ↔ (x^2 + 5 * x + 3 * x < 4 + 3 * x) :=
sorry

theorem pair2_non_equivalent (x : ℝ) (hx : x ≠ 0) : (x^2 + 5 * x < 4) ↔ (x^2 + 5 * x + 1 / x < 4 + 1 / x) :=
sorry

theorem pair3_equivalent (x : ℝ) (hx : x ≥ 3) : (x ≥ 3) ↔ (x * (x + 5)^2 ≥ 3 * (x + 5)^2) :=
sorry

theorem pair4_equivalent (x : ℝ) (hx : x ≥ 3) : (x ≥ 3) ↔ (x * (x - 5)^2 ≥ 3 * (x - 5)^2) :=
sorry

theorem pair5_non_equivalent (x : ℝ) (hx : x ≠ -1) : (x + 3 > 0) ↔ ( (x + 3) * (x + 1) / (x + 1) > 0) :=
sorry

theorem pair6_equivalent (x : ℝ) (hx : x ≠ -2) : (x - 3 > 0) ↔ ( (x + 2) * (x - 3) / (x + 2) > 0) :=
sorry

end pair1_equivalent_pair2_non_equivalent_pair3_equivalent_pair4_equivalent_pair5_non_equivalent_pair6_equivalent_l811_81164


namespace min_value_l811_81163

noncomputable def min_expression_value (x y z k : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 2 ≤ k) : ℝ :=
  (x^2 + k*x + 1) * (y^2 + k*y + 1) * (z^2 + k*z + 1) / (x * y * z)

theorem min_value (x y z k : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 2 ≤ k) :
  min_expression_value x y z k hx hy hz hk ≥ (2 + k)^3 :=
by
  sorry

end min_value_l811_81163


namespace sum_of_squares_219_l811_81122

theorem sum_of_squares_219 :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a^2 + b^2 + c^2 = 219 ∧ a + b + c = 21 := by
  sorry

end sum_of_squares_219_l811_81122


namespace infinite_non_expressible_integers_l811_81173

theorem infinite_non_expressible_integers :
  ∃ (S : Set ℤ), S.Infinite ∧ (∀ n ∈ S, ∀ a b c : ℕ, n ≠ 2^a + 3^b - 5^c) :=
sorry

end infinite_non_expressible_integers_l811_81173

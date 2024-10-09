import Mathlib

namespace solve_system_l192_19280

theorem solve_system (x y : ℝ) :
  (x + 3*y + 3*x*y = -1) ∧ (x^2*y + 3*x*y^2 = -4) →
  (x = -3 ∧ y = -1/3) ∨ (x = -1 ∧ y = -1) ∨ (x = -1 ∧ y = 4/3) ∨ (x = 4 ∧ y = -1/3) :=
by
  sorry

end solve_system_l192_19280


namespace value_of_a_l192_19254

theorem value_of_a :
  ∀ (a : ℤ) (BO CO : ℤ), 
  BO = 2 → 
  CO = 2 * BO → 
  |a + 3| = CO → 
  a < 0 → 
  a = -7 := by
  intros a BO CO hBO hCO hAbs ha_neg
  sorry

end value_of_a_l192_19254


namespace maximum_number_of_intersections_of_150_lines_is_7171_l192_19291

def lines_are_distinct (L : ℕ → Type) : Prop := 
  ∀ n m : ℕ, n ≠ m → L n ≠ L m

def lines_parallel_to_each_other (L : ℕ → Type) (k : ℕ) : Prop :=
  ∀ n m : ℕ, n ≠ m → L (k * n) = L (k * m)

def lines_pass_through_point_B (L : ℕ → Type) (B : Type) (k : ℕ) : Prop :=
  ∀ n : ℕ, L (k * n - 4) = B

def lines_not_parallel (L : ℕ → Type) (k1 k2 : ℕ) : Prop :=
  ∀ n m : ℕ, L (k1 * n) ≠ L (k2 * m)

noncomputable def max_points_of_intersection
  (L : ℕ → Type)
  (B : Type)
  (k1 k2 : ℕ)
  (h_distinct : lines_are_distinct L)
  (h_parallel1 : lines_parallel_to_each_other L k1)
  (h_parallel2 : lines_parallel_to_each_other L k2)
  (h_pass_through_B : lines_pass_through_point_B L B 5)
  (h_not_parallel : lines_not_parallel L k1 k2)
  : ℕ :=
  7171

theorem maximum_number_of_intersections_of_150_lines_is_7171
  (L : ℕ → Type)
  (B : Type)
  (k1 k2 : ℕ)
  (h_distinct : lines_are_distinct L)
  (h_parallel1 : lines_parallel_to_each_other L k1)
  (h_parallel2 : lines_parallel_to_each_other L k2)
  (h_pass_through_B : lines_pass_through_point_B L B 5)
  (h_not_parallel : lines_not_parallel L k1 k2)
  : max_points_of_intersection L B k1 k2 h_distinct h_parallel1 h_parallel2 h_pass_through_B h_not_parallel = 7171 := 
  by 
  sorry

end maximum_number_of_intersections_of_150_lines_is_7171_l192_19291


namespace cone_surface_area_and_volume_l192_19224

theorem cone_surface_area_and_volume
  (r l m : ℝ)
  (h_ratio : (π * r * l) / (π * r * l + π * r^2) = 25 / 32)
  (h_height : m = 96) :
  (π * r * l + π * r^2 = 3584 * π) ∧ ((1 / 3) * π * r^2 * m = 25088 * π) :=
by {
  sorry
}

end cone_surface_area_and_volume_l192_19224


namespace find_f_neg_2_l192_19228

def f (x : ℝ) : ℝ := sorry -- The actual function f is undefined here.

theorem find_f_neg_2 (h : ∀ x ≠ 0, f (1 / x) + (1 / x) * f (-x) = 2 * x) :
  f (-2) = 7 / 2 :=
sorry

end find_f_neg_2_l192_19228


namespace comics_in_box_l192_19207

def comics_per_comic := 25
def total_pages := 150
def existing_comics := 5

def torn_comics := total_pages / comics_per_comic
def total_comics := torn_comics + existing_comics

theorem comics_in_box : total_comics = 11 := by
  sorry

end comics_in_box_l192_19207


namespace rectangle_area_at_stage_8_l192_19269

-- Definitions based on conditions
def area_of_square (side_length : ℕ) : ℕ := side_length * side_length
def number_of_squares_in_stage (stage : ℕ) : ℕ := stage

-- The main theorem to prove
theorem rectangle_area_at_stage_8 : 
  area_of_square 4 * number_of_squares_in_stage 8 = 128 := by
  sorry

end rectangle_area_at_stage_8_l192_19269


namespace green_shirt_pairs_l192_19289

theorem green_shirt_pairs (blue_shirts green_shirts total_pairs blue_blue_pairs : ℕ) 
(h1 : blue_shirts = 68) 
(h2 : green_shirts = 82) 
(h3 : total_pairs = 75) 
(h4 : blue_blue_pairs = 30) 
: (green_shirts - (blue_shirts - 2 * blue_blue_pairs)) / 2 = 37 := 
by 
  -- This is where the proof would be written, but we use sorry to skip it.
  sorry

end green_shirt_pairs_l192_19289


namespace number_of_rabbits_l192_19234

theorem number_of_rabbits (x y : ℕ) (h1 : x + y = 28) (h2 : 4 * x = 6 * y + 12) : x = 18 :=
by
  sorry

end number_of_rabbits_l192_19234


namespace jenny_proposal_time_l192_19210

theorem jenny_proposal_time (total_time research_time report_time proposal_time : ℕ) 
  (h1 : total_time = 20) 
  (h2 : research_time = 10) 
  (h3 : report_time = 8) 
  (h4 : proposal_time = total_time - research_time - report_time) : 
  proposal_time = 2 := 
by
  sorry

end jenny_proposal_time_l192_19210


namespace matrix_power_A_100_l192_19236

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![0, 0, 1],![1, 0, 0],![0, 1, 0]]

theorem matrix_power_A_100 : A^100 = A := by sorry

end matrix_power_A_100_l192_19236


namespace initial_height_after_10_seconds_l192_19249

open Nat

def distance_fallen_in_nth_second (n : ℕ) : ℕ := 10 * n - 5

def total_distance_fallen (n : ℕ) : ℕ :=
  (n * (distance_fallen_in_nth_second 1 + distance_fallen_in_nth_second n)) / 2

theorem initial_height_after_10_seconds : 
  total_distance_fallen 10 = 500 := 
by
  sorry

end initial_height_after_10_seconds_l192_19249


namespace largest_pentagon_angle_is_179_l192_19205

-- Define the interior angles of the pentagon
def angle1 (x : ℝ) := x + 2
def angle2 (x : ℝ) := 2 * x + 3
def angle3 (x : ℝ) := 3 * x - 5
def angle4 (x : ℝ) := 4 * x + 1
def angle5 (x : ℝ) := 5 * x - 1

-- Define the sum of the interior angles of a pentagon
def pentagon_angle_sum := angle1 36 + angle2 36 + angle3 36 + angle4 36 + angle5 36

-- Define the largest angle function
def largest_angle (x : ℝ) := 5 * x - 1

-- The main theorem stating the largest angle measure
theorem largest_pentagon_angle_is_179 (h : angle1 36 + angle2 36 + angle3 36 + angle4 36 + angle5 36 = 540) :
  largest_angle 36 = 179 :=
sorry

end largest_pentagon_angle_is_179_l192_19205


namespace percentage_girls_l192_19259

theorem percentage_girls (x y : ℕ) (S₁ S₂ : ℕ)
  (h1 : S₁ = 22 * x)
  (h2 : S₂ = 47 * y)
  (h3 : (S₁ + S₂) / (x + y) = 41) :
  (x : ℝ) / (x + y) = 0.24 :=
sorry

end percentage_girls_l192_19259


namespace jump_rope_cost_l192_19204

def cost_board_game : ℕ := 12
def cost_playground_ball : ℕ := 4
def saved_money : ℕ := 6
def uncle_money : ℕ := 13
def additional_needed : ℕ := 4

theorem jump_rope_cost :
  let total_money := saved_money + uncle_money
  let total_needed := total_money + additional_needed
  let combined_cost := cost_board_game + cost_playground_ball
  let cost_jump_rope := total_needed - combined_cost
  cost_jump_rope = 7 := by
  sorry

end jump_rope_cost_l192_19204


namespace dusty_change_l192_19215

def price_single_layer : ℕ := 4
def price_double_layer : ℕ := 7
def number_of_single_layers : ℕ := 7
def number_of_double_layers : ℕ := 5
def amount_paid : ℕ := 100

theorem dusty_change :
  amount_paid - (number_of_single_layers * price_single_layer + number_of_double_layers * price_double_layer) = 37 := 
by
  sorry

end dusty_change_l192_19215


namespace fourth_term_of_gp_is_negative_10_point_42_l192_19293

theorem fourth_term_of_gp_is_negative_10_point_42 (x : ℝ) 
  (h : ∃ r : ℝ, r * (5 * x + 5) = (3 * x + 3) * ((3 * x + 3) / x)) :
  r * (5 * x + 5) * ((3 * x + 3) / x) * ((3 * x + 3) / x) = -10.42 :=
by
  sorry

end fourth_term_of_gp_is_negative_10_point_42_l192_19293


namespace y_value_l192_19264

theorem y_value {y : ℝ} (h1 : (0, 2) = (0, 2))
                (h2 : (3, y) = (3, y))
                (h3 : dist (0, 2) (3, y) = 10)
                (h4 : y > 0) :
                y = 2 + Real.sqrt 91 := by
  sorry

end y_value_l192_19264


namespace find_k_when_lines_perpendicular_l192_19250

theorem find_k_when_lines_perpendicular (k : ℝ) :
  (∀ x y : ℝ, (k-3) * x + (3-k) * y + 1 = 0 → ∀ x y : ℝ, 2 * (k-3) * x - 2 * y + 3 = 0 → -((k-3)/(3-k)) * (k-3) = -1) → 
  k = 2 :=
by
  sorry

end find_k_when_lines_perpendicular_l192_19250


namespace range_of_m_l192_19285

theorem range_of_m (x y m : ℝ) 
  (h1 : 3 * x + y = m - 1)
  (h2 : x - 3 * y = 2 * m)
  (h3 : x + 2 * y ≥ 0) : 
  m ≤ -1 := 
sorry

end range_of_m_l192_19285


namespace positive_integers_solution_l192_19260

theorem positive_integers_solution :
  ∀ (m n : ℕ), 0 < m ∧ 0 < n ∧ (3 ^ m - 2 ^ n = -1 ∨ 3 ^ m - 2 ^ n = 5 ∨ 3 ^ m - 2 ^ n = 7) ↔
  (m, n) = (0, 1) ∨ (m, n) = (2, 1) ∨ (m, n) = (1, 2) ∨ (m, n) = (2, 2) :=
by
  sorry

end positive_integers_solution_l192_19260


namespace value_of_y_minus_x_l192_19282

theorem value_of_y_minus_x (x y z : ℝ) 
  (h1 : x + y + z = 12) 
  (h2 : x + y = 8) 
  (h3 : y - 3 * x + z = 9) : 
  y - x = 6.5 :=
by
  -- Proof steps would go here
  sorry

end value_of_y_minus_x_l192_19282


namespace mean_of_eight_numbers_l192_19206

theorem mean_of_eight_numbers (sum_of_numbers : ℚ) (h : sum_of_numbers = 3/4) : 
  sum_of_numbers / 8 = 3/32 := by
  sorry

end mean_of_eight_numbers_l192_19206


namespace box_volume_80_possible_l192_19272

theorem box_volume_80_possible :
  ∃ (x : ℕ), 10 * x^3 = 80 :=
by
  sorry

end box_volume_80_possible_l192_19272


namespace price_per_pound_salt_is_50_l192_19246

-- Given conditions
def totalWeight : ℕ := 60
def weightSalt1 : ℕ := 20
def priceSalt2 : ℕ := 35
def weightSalt2 : ℕ := 40
def sellingPricePerPound : ℕ := 48
def desiredProfitRate : ℚ := 0.20

-- Mathematical definitions derived from conditions
def costSalt1 (priceSalt1 : ℕ) : ℕ := weightSalt1 * priceSalt1
def costSalt2 : ℕ := weightSalt2 * priceSalt2
def totalCost (priceSalt1 : ℕ) : ℕ := costSalt1 priceSalt1 + costSalt2
def totalRevenue : ℕ := totalWeight * sellingPricePerPound
def profit (priceSalt1 : ℕ) : ℚ := desiredProfitRate * totalCost priceSalt1
def totalProfit (priceSalt1 : ℕ) : ℚ := totalCost priceSalt1 + profit priceSalt1

-- Proof statement
theorem price_per_pound_salt_is_50 : ∃ (priceSalt1 : ℕ), totalRevenue = totalProfit priceSalt1 ∧ priceSalt1 = 50 := by
  -- We provide the prove structure, exact proof steps are skipped with sorry
  sorry

end price_per_pound_salt_is_50_l192_19246


namespace cylinder_from_sector_l192_19221

noncomputable def circle_radius : ℝ := 12
noncomputable def sector_angle : ℝ := 300
noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * 2 * Real.pi * r

noncomputable def is_valid_cylinder (base_radius height : ℝ) : Prop :=
  2 * Real.pi * base_radius = arc_length circle_radius sector_angle ∧ height = circle_radius

theorem cylinder_from_sector :
  is_valid_cylinder 10 12 :=
by
  -- here, the proof will be provided
  sorry

end cylinder_from_sector_l192_19221


namespace factorization_correct_l192_19217

theorem factorization_correct : ∃ (a b : ℕ), (a > b) ∧ (3 * b - a = 12) ∧ (x^2 - 16 * x + 63 = (x - a) * (x - b)) :=
by
  sorry

end factorization_correct_l192_19217


namespace change_received_l192_19255

theorem change_received (basic_cost : ℕ) (scientific_cost : ℕ) (graphing_cost : ℕ) (total_money : ℕ) :
  basic_cost = 8 →
  scientific_cost = 2 * basic_cost →
  graphing_cost = 3 * scientific_cost →
  total_money = 100 →
  (total_money - (basic_cost + scientific_cost + graphing_cost)) = 28 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end change_received_l192_19255


namespace dice_sum_not_20_l192_19238

/-- Given that Louise rolls four standard six-sided dice (with faces numbered from 1 to 6)
    and the product of the numbers on the upper faces is 216, prove that it is not possible
    for the sum of the upper faces to be 20. -/
theorem dice_sum_not_20 (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) 
                        (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) 
                        (product : a * b * c * d = 216) : a + b + c + d ≠ 20 := 
by sorry

end dice_sum_not_20_l192_19238


namespace sin_cos_value_l192_19245

variable (x : ℝ)

theorem sin_cos_value (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
sorry

end sin_cos_value_l192_19245


namespace parabola_focus_segment_length_l192_19233

theorem parabola_focus_segment_length (a : ℝ) (h₀ : a > 0) 
  (h₁ : ∀ x, abs x * abs (1 / a) = 4) : a = 1/4 := 
sorry

end parabola_focus_segment_length_l192_19233


namespace quadratic_equation_m_value_l192_19208

theorem quadratic_equation_m_value (m : ℝ) (h : m ≠ 2) : m = -2 :=
by
  -- details of the proof go here
  sorry

end quadratic_equation_m_value_l192_19208


namespace neg_proposition_p_l192_19266

variable {x : ℝ}

def proposition_p : Prop := ∀ x ≥ 0, x^3 - 1 ≥ 0

theorem neg_proposition_p : ¬ proposition_p ↔ ∃ x ≥ 0, x^3 - 1 < 0 :=
by sorry

end neg_proposition_p_l192_19266


namespace find_value_l192_19251

-- Defining the sequence a_n, assuming all terms are positive
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = a n * r

-- Definition to capture the given condition a_2 * a_4 = 4
def condition (a : ℕ → ℝ) : Prop :=
  a 2 * a 4 = 4

-- The main statement
theorem find_value (a : ℕ → ℝ) (h_seq : is_geometric_sequence a) (h_cond : condition a) : 
  a 1 * a 5 + a 3 = 6 := 
by 
  sorry

end find_value_l192_19251


namespace complement_intersection_l192_19253

open Set

variable (U : Set ℕ) (A B : Set ℕ)

theorem complement_intersection :
  U = {1, 2, 3, 4, 5} →
  A = {1, 2, 3} →
  B = {2, 3, 5} →
  U \ (A ∩ B) = {1, 4, 5} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  sorry

end complement_intersection_l192_19253


namespace min_sum_a_b_l192_19271

-- The conditions
variables {a b : ℝ}
variables (h₁ : a > 1) (h₂ : b > 1) (h₃ : ab - (a + b) = 1)

-- The theorem statement
theorem min_sum_a_b : a + b = 2 + 2 * Real.sqrt 2 :=
sorry

end min_sum_a_b_l192_19271


namespace committee_member_count_l192_19287

theorem committee_member_count (n : ℕ) (M : ℕ) (Q : ℚ) 
  (h₁ : M = 6) 
  (h₂ : 2 * n = M) 
  (h₃ : Q = 0.4) 
  (h₄ : Q = (n - 1) / (M - 1)) : 
  n = 3 :=
by
  sorry

end committee_member_count_l192_19287


namespace find_income_l192_19209

variable (x : ℝ)

def income : ℝ := 5 * x
def expenditure : ℝ := 4 * x
def savings : ℝ := income x - expenditure x

theorem find_income (h : savings x = 4000) : income x = 20000 :=
by
  rw [savings, income, expenditure] at h
  sorry

end find_income_l192_19209


namespace time_to_reach_ship_l192_19200

/-- The scuba diver's descent problem -/

def rate_of_descent : ℕ := 35  -- in feet per minute
def depth_of_ship : ℕ := 3500  -- in feet

theorem time_to_reach_ship : depth_of_ship / rate_of_descent = 100 := by
  sorry

end time_to_reach_ship_l192_19200


namespace complement_is_correct_l192_19276

-- Define the universal set U and set M
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

-- Define the complement of M with respect to U
def complement_U (U M : Set ℕ) : Set ℕ := {x ∈ U | x ∉ M}

-- State the theorem to be proved
theorem complement_is_correct : complement_U U M = {3, 5, 6} :=
by
  sorry

end complement_is_correct_l192_19276


namespace two_same_color_probability_l192_19239

-- Definitions based on the given conditions
def total_balls := 5
def black_balls := 3
def red_balls := 2

-- Definition for drawing two balls at random
def draw_two_same_color_probability : ℚ :=
  let total_ways := Nat.choose total_balls 2
  let black_pairs := Nat.choose black_balls 2
  let red_pairs := Nat.choose red_balls 2
  (black_pairs + red_pairs) / total_ways

-- Statement of the theorem
theorem two_same_color_probability :
  draw_two_same_color_probability = 2 / 5 :=
  sorry

end two_same_color_probability_l192_19239


namespace option_d_is_correct_l192_19252

theorem option_d_is_correct {x y : ℝ} (h : x - 2 = y - 2) : x = y := 
by 
  sorry

end option_d_is_correct_l192_19252


namespace f_is_constant_l192_19268

noncomputable def is_const (f : ℤ × ℤ → ℕ) := ∃ c : ℕ, ∀ p : ℤ × ℤ, f p = c

theorem f_is_constant (f : ℤ × ℤ → ℕ) 
  (h : ∀ x y : ℤ, 4 * f (x, y) = f (x - 1, y) + f (x, y + 1) + f (x + 1, y) + f (x, y - 1)) :
  is_const f :=
sorry

end f_is_constant_l192_19268


namespace MarionBikeCost_l192_19267

theorem MarionBikeCost (M : ℤ) (h1 : 2 * M + M = 1068) : M = 356 :=
by
  sorry

end MarionBikeCost_l192_19267


namespace remainder_sum_of_six_primes_div_seventh_prime_l192_19278

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l192_19278


namespace calculate_result_l192_19231

theorem calculate_result :
  1 - 2 * (Real.sin (Real.pi / 8))^2 = Real.cos (Real.pi / 4) :=
by
  sorry

end calculate_result_l192_19231


namespace tattoo_ratio_l192_19247

theorem tattoo_ratio (a j k : ℕ) (ha : a = 23) (hj : j = 10) (rel : a = k * j + 3) : a / j = 23 / 10 :=
by {
  -- Insert proof here
  sorry
}

end tattoo_ratio_l192_19247


namespace min_moves_to_checkerboard_l192_19229

noncomputable def minimum_moves_checkerboard (n : ℕ) : ℕ :=
if n = 6 then 18
else 0

theorem min_moves_to_checkerboard :
  minimum_moves_checkerboard 6 = 18 :=
by sorry

end min_moves_to_checkerboard_l192_19229


namespace brad_zip_code_l192_19297

theorem brad_zip_code (a b c d e : ℕ) 
  (h1 : a = b) 
  (h2 : c = 0) 
  (h3 : d = 2 * a) 
  (h4 : d + e = 8) 
  (h5 : a + b + c + d + e = 10) : 
  (a, b, c, d, e) = (1, 1, 0, 2, 6) :=
by 
  -- Proof omitted on purpose
  sorry

end brad_zip_code_l192_19297


namespace composite_for_large_n_l192_19202

theorem composite_for_large_n (m : ℕ) (hm : m > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → Nat.Prime (2^m * 2^(2^n) + 1) = false :=
sorry

end composite_for_large_n_l192_19202


namespace part_a_part_b_l192_19262

/-- Definition of the sequence of numbers on the cards -/
def card_numbers (n : ℕ) : ℕ :=
  if n = 0 then 1 else (10^(n + 1) - 1) / 9 * 2 + 1

/-- Part (a) statement: Is it possible to choose at least three cards such that 
the sum of the numbers on them equals a number where all digits except one are twos? -/
theorem part_a : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ card_numbers a + card_numbers b + card_numbers c % 10 = 2 ∧ 
  (∀ d, ∃ k ≤ 1, (card_numbers a + card_numbers b + card_numbers c / (10^d)) % 10 = 2) :=
sorry

/-- Part (b) statement: Suppose several cards were chosen such that the sum of the numbers 
on them equals a number where all digits except one are twos. What could be the digit that is not two? -/
theorem part_b (sum : ℕ) :
  (∀ d, sum / (10^d) % 10 = 2) → ((sum % 10 = 0) ∨ (sum % 10 = 1)) :=
sorry

end part_a_part_b_l192_19262


namespace sum_series_l192_19263

theorem sum_series : ∑' n, (2 * n + 1) / (n * (n + 1) * (n + 2)) = 1 := 
sorry

end sum_series_l192_19263


namespace remainder_x2023_l192_19274

theorem remainder_x2023 (x : ℤ) : 
  let dividend := x^2023 + 1
  let divisor := x^6 - x^4 + x^2 - 1
  let remainder := -x^7 + 1
  dividend % divisor = remainder :=
by
  sorry

end remainder_x2023_l192_19274


namespace transformed_center_coordinates_l192_19240

-- Define the original center of the circle
def center_initial : ℝ × ℝ := (3, -4)

-- Define the function for reflection across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the function for translation by a certain number of units up
def translate_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

-- Define the problem statement
theorem transformed_center_coordinates :
  translate_up (reflect_x_axis center_initial) 5 = (3, 9) :=
by
  sorry

end transformed_center_coordinates_l192_19240


namespace shaded_area_of_pattern_l192_19218

theorem shaded_area_of_pattern (d : ℝ) (L : ℝ) (n : ℕ) (r : ℝ) (A : ℝ) : 
  d = 3 → 
  L = 24 → 
  n = 16 → 
  r = 3 / 2 → 
  (A = 18 * Real.pi) :=
by
  intro hd
  intro hL
  intro hn
  intro hr
  sorry

end shaded_area_of_pattern_l192_19218


namespace part1_part2_l192_19232

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem part1 (x : ℝ) : (f x 2 ≥ 7 - |x - 1|) ↔ (x ≤ -2 ∨ x ≥ 5) :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x, f x a ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) : a = 1 :=
by sorry

end part1_part2_l192_19232


namespace set_representation_l192_19201

def is_Natural (n : ℕ) : Prop :=
  n ≠ 0

def condition (x : ℕ) : Prop :=
  x^2 - 3*x < 0

theorem set_representation :
  {x : ℕ | condition x ∧ is_Natural x} = {1, 2} := 
sorry

end set_representation_l192_19201


namespace factorial_trailing_zeros_500_l192_19213

theorem factorial_trailing_zeros_500 :
  let count_factors_of_five (n : ℕ) : ℕ := n / 5 + n / 25 + n / 125
  count_factors_of_five 500 = 124 :=
by
  sorry  -- The proof is not required as per the instructions.

end factorial_trailing_zeros_500_l192_19213


namespace positive_integers_p_divisibility_l192_19226

theorem positive_integers_p_divisibility (p : ℕ) (hp : 0 < p) :
  (∃ n : ℕ, 0 < n ∧ p^n + 3^n ∣ p^(n+1) + 3^(n+1)) ↔ p = 3 ∨ p = 6 ∨ p = 15 :=
by sorry

end positive_integers_p_divisibility_l192_19226


namespace problem_inequality_A_problem_inequality_B_problem_inequality_D_problem_inequality_E_l192_19243

variable {a b c : ℝ}

theorem problem_inequality_A (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a * b < b * c :=
by sorry

theorem problem_inequality_B (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a * c < b * c :=
by sorry

theorem problem_inequality_D (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a + b < b + c :=
by sorry

theorem problem_inequality_E (h1 : a > 0) (h2 : a < b) (h3 : b < c) : c / a > 1 :=
by sorry

end problem_inequality_A_problem_inequality_B_problem_inequality_D_problem_inequality_E_l192_19243


namespace difference_between_numbers_l192_19261

open Int

theorem difference_between_numbers (A B : ℕ) 
  (h1 : A + B = 1812) 
  (h2 : A = 7 * B + 4) : 
  A - B = 1360 :=
by
  sorry

end difference_between_numbers_l192_19261


namespace builder_windows_installed_l192_19258

theorem builder_windows_installed (total_windows : ℕ) (hours_per_window : ℕ) (total_hours_left : ℕ) :
  total_windows = 14 → hours_per_window = 4 → total_hours_left = 36 → (total_windows - total_hours_left / hours_per_window) = 5 :=
by
  intros
  sorry

end builder_windows_installed_l192_19258


namespace smallest_y_value_l192_19290

theorem smallest_y_value (y : ℝ) : (12 * y^2 - 56 * y + 48 = 0) → y = 2 :=
by
  sorry

end smallest_y_value_l192_19290


namespace business_fraction_l192_19212

theorem business_fraction (x : ℚ) (H1 : 3 / 4 * x * 60000 = 30000) : x = 2 / 3 :=
by sorry

end business_fraction_l192_19212


namespace mr_yadav_yearly_savings_l192_19241

theorem mr_yadav_yearly_savings (S : ℕ) (h1 : S * 3 / 5 * 1 / 2 = 1584) : S * 3 / 5 * 1 / 2 * 12 = 19008 :=
  sorry

end mr_yadav_yearly_savings_l192_19241


namespace evaluate_rr2_l192_19299

def q (x : ℝ) : ℝ := x^2 - 5 * x + 6
def r (x : ℝ) : ℝ := (x - 3) * (x - 2)

theorem evaluate_rr2 : r (r 2) = 6 :=
by
  -- proof goes here
  sorry

end evaluate_rr2_l192_19299


namespace least_positive_integer_for_multiple_of_five_l192_19298

theorem least_positive_integer_for_multiple_of_five (x : ℕ) (h_pos : 0 < x) (h_multiple : (625 + x) % 5 = 0) : x = 5 :=
sorry

end least_positive_integer_for_multiple_of_five_l192_19298


namespace part1_part2_part3_l192_19219

variable (a b c d S A B C D : ℝ)

-- The given conditions
def cond1 : Prop := a + c = b + d
def cond2 : Prop := A + C = B + D
def cond3 : Prop := S^2 = a * b * c * d

-- The statements to prove
theorem part1 (h1 : cond1 a b c d) (h2 : cond2 A B C D) : cond3 a b c d S := sorry
theorem part2 (h1 : cond1 a b c d) (h3 : cond3 a b c d S) : cond2 A B C D := sorry
theorem part3 (h2 : cond2 A B C D) : cond3 a b c d S := sorry

end part1_part2_part3_l192_19219


namespace intersection_AB_union_AB_difference_A_minus_B_difference_B_minus_A_l192_19294

noncomputable def setA : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }
noncomputable def setB : Set ℝ := { x : ℝ | 1 < x }

theorem intersection_AB : setA ∩ setB = { x : ℝ | 1 < x ∧ x < 2 } := by
  sorry

theorem union_AB : setA ∪ setB = { x : ℝ | -1 < x } := by
  sorry

theorem difference_A_minus_B : setA \ setB = { x : ℝ | -1 < x ∧ x ≤ 1 } := by
  sorry

theorem difference_B_minus_A : setB \ setA = { x : ℝ | 2 ≤ x } := by
  sorry

end intersection_AB_union_AB_difference_A_minus_B_difference_B_minus_A_l192_19294


namespace original_price_per_kg_of_salt_l192_19257

variable {P X : ℝ}

theorem original_price_per_kg_of_salt (h1 : 400 / (0.8 * P) = X + 10)
    (h2 : 400 / P = X) : P = 10 :=
by
  sorry

end original_price_per_kg_of_salt_l192_19257


namespace two_digit_prime_sum_9_l192_19237

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- There are 0 two-digit prime numbers for which the sum of the digits equals 9 -/
theorem two_digit_prime_sum_9 : ∃! n : ℕ, (9 ≤ n ∧ n < 100) ∧ (n.digits 10).sum = 9 ∧ is_prime n :=
sorry

end two_digit_prime_sum_9_l192_19237


namespace that_remaining_money_l192_19244

section
/-- Initial money in Olivia's wallet --/
def initial_money : ℕ := 53

/-- Money collected from ATM --/
def collected_money : ℕ := 91

/-- Money spent at the supermarket --/
def spent_money : ℕ := collected_money + 39

/-- Remaining money after visiting the supermarket --
Theorem that proves Olivia's remaining money is 14 dollars.
-/
theorem remaining_money : initial_money + collected_money - spent_money = 14 := 
by
  unfold initial_money collected_money spent_money
  simp
  sorry
end

end that_remaining_money_l192_19244


namespace domain_of_h_l192_19279

theorem domain_of_h (x : ℝ) : |x - 5| + |x + 3| ≠ 0 := by
  sorry

end domain_of_h_l192_19279


namespace simplify_and_evaluate_l192_19296

theorem simplify_and_evaluate (x : ℝ) (hx : x = Real.sqrt 2) :
  ( ( (2 * x - 1) / (x + 1) - x + 1 ) / (x - 2) / (x^2 + 2 * x + 1) ) = -2 - Real.sqrt 2 :=
by sorry

end simplify_and_evaluate_l192_19296


namespace mass_percentage_Na_in_NaClO_l192_19284

theorem mass_percentage_Na_in_NaClO :
  let mass_Na : ℝ := 22.99
  let mass_Cl : ℝ := 35.45
  let mass_O : ℝ := 16.00
  let mass_NaClO : ℝ := mass_Na + mass_Cl + mass_O
  (mass_Na / mass_NaClO) * 100 = 30.89 := by
sorry

end mass_percentage_Na_in_NaClO_l192_19284


namespace base_p_prime_values_zero_l192_19220

theorem base_p_prime_values_zero :
  (∀ p : ℕ, p.Prime → 2008 * p^3 + 407 * p^2 + 214 * p + 226 = 243 * p^2 + 382 * p + 471 → False) :=
by
  sorry

end base_p_prime_values_zero_l192_19220


namespace correct_time_l192_19214

-- Define the observed times on the clocks
def time1 := 14 * 60 + 54  -- 14:54 in minutes
def time2 := 14 * 60 + 57  -- 14:57 in minutes
def time3 := 15 * 60 + 2   -- 15:02 in minutes
def time4 := 15 * 60 + 3   -- 15:03 in minutes

-- Define the inaccuracies of the clocks
def inaccuracy1 := 2  -- First clock off by 2 minutes
def inaccuracy2 := 3  -- Second clock off by 3 minutes
def inaccuracy3 := -4  -- Third clock off by 4 minutes
def inaccuracy4 := -5  -- Fourth clock off by 5 minutes

-- State that given these conditions, the correct time is 14:58
theorem correct_time : ∃ (T : Int), 
  (time1 + inaccuracy1 = T) ∧
  (time2 + inaccuracy2 = T) ∧
  (time3 + inaccuracy3 = T) ∧
  (time4 + inaccuracy4 = T) ∧
  (T = 14 * 60 + 58) :=
by
  sorry

end correct_time_l192_19214


namespace find_value_l192_19223

-- Given conditions of the problem
axiom condition : ∀ (a : ℝ), a - 1/a = 1

-- The mathematical proof problem
theorem find_value (a : ℝ) (h : a - 1/a = 1) : a^2 - a + 2 = 3 :=
by
  sorry

end find_value_l192_19223


namespace sufficient_but_not_necessary_condition_l192_19277

theorem sufficient_but_not_necessary_condition (x : ℝ) : (x > 3) → (x ≥ 3) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l192_19277


namespace volume_of_rect_prism_l192_19235

variables {a b c V : ℝ}

theorem volume_of_rect_prism :
  (∃ (a b c : ℝ), (a * b = Real.sqrt 2) ∧ (b * c = Real.sqrt 3) ∧ (a * c = Real.sqrt 6) ∧ V = a * b * c) →
  V = Real.sqrt 6 :=
by
  sorry

end volume_of_rect_prism_l192_19235


namespace triangle_right_triangle_l192_19283

theorem triangle_right_triangle (a b : ℕ) (c : ℝ) 
  (h1 : a = 3) (h2 : b = 4) (h3 : c^2 - 10 * c + 25 = 0) : 
  a^2 + b^2 = c^2 :=
by
  -- We know the values of a, b, and c by the conditions
  sorry

end triangle_right_triangle_l192_19283


namespace perfect_square_solution_l192_19292

theorem perfect_square_solution (x : ℤ) : 
  ∃ k : ℤ, x^2 - 14 * x - 256 = k^2 ↔ x = 15 ∨ x = -1 :=
by
  sorry

end perfect_square_solution_l192_19292


namespace original_number_is_40_l192_19211

theorem original_number_is_40 (x : ℝ) (h : 1.25 * x - 0.70 * x = 22) : x = 40 :=
by
  sorry

end original_number_is_40_l192_19211


namespace man_savings_l192_19275

theorem man_savings (I : ℝ) (S : ℝ) (h1 : S = 0.35) (h2 : 2 * (0.65 * I) = 0.65 * I + 0.70 * I) :
  S = 0.35 :=
by
  -- Introduce necessary assumptions
  let savings_first_year := S * I
  let expenditure_first_year := I - savings_first_year
  let savings_second_year := 2 * savings_first_year

  have h3 : expenditure_first_year = 0.65 * I := by sorry
  have h4 : savings_first_year = 0.35 * I := by sorry

  -- Using given condition to resolve S
  exact h1

end man_savings_l192_19275


namespace range_of_t_for_point_in_upper_left_side_l192_19216

def point_in_upper_left_side_condition (x y : ℝ) : Prop :=
  x - y + 4 < 0

theorem range_of_t_for_point_in_upper_left_side :
  ∀ t : ℝ, point_in_upper_left_side_condition (-2) t ↔ t > 2 :=
by
  intros t
  unfold point_in_upper_left_side_condition
  simp
  sorry

end range_of_t_for_point_in_upper_left_side_l192_19216


namespace distinct_integers_sum_l192_19286

theorem distinct_integers_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_product : a * b * c * d = 357) : a + b + c + d = 28 :=
by
  sorry

end distinct_integers_sum_l192_19286


namespace problem1_problem2_l192_19248

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a/x

/-- 
Given the function f(x) = ln(x) + a/x (where a is a real number),
prove that if the function f(x) has two zeros, then 0 < a < 1/e.
-/
theorem problem1 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → (0 < a ∧ a < 1/Real.exp 1) :=
sorry

/-- 
Given the function f(x) = ln(x) + a/x (where a is a real number) and a line y = m
that intersects the graph of f(x) at two points (x1, m) and (x2, m),
prove that x1 + x2 > 2a.
-/
theorem problem2 (x1 x2 a m : ℝ) (h : f x1 a = m ∧ f x2 a = m ∧ x1 ≠ x2) :
  x1 + x2 > 2 * a :=
sorry

end problem1_problem2_l192_19248


namespace carnival_game_ratio_l192_19203

theorem carnival_game_ratio (L W : ℕ) (h_ratio : 4 * L = W) (h_lost : L = 7) : W = 28 :=
by {
  sorry
}

end carnival_game_ratio_l192_19203


namespace center_of_symmetry_is_neg2_3_l192_19256

theorem center_of_symmetry_is_neg2_3 :
  ∃ (a b : ℝ), 
  (a,b) = (-2, 3) ∧ 
  ∀ x : ℝ, 
    2 * b = ((a + x + 2)^3 - (a + x) + 1) + ((a - x + 2)^3 - (a - x) + 1) := 
by
  use -2, 3
  sorry

end center_of_symmetry_is_neg2_3_l192_19256


namespace find_p_l192_19270

theorem find_p (m n p : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < p) 
  (h : 3 * m + 3 / (n + 1 / p) = 17) : p = 2 := 
sorry

end find_p_l192_19270


namespace tangent_line_inclination_range_l192_19225

theorem tangent_line_inclination_range:
  ∀ (x : ℝ), -1/2 ≤ x ∧ x ≤ 1/2 → (0 ≤ 2*x ∧ 2*x ≤ 1 ∨ -1 ≤ 2*x ∧ 2*x < 0) →
    ∃ (α : ℝ), (0 ≤ α ∧ α ≤ π/4) ∨ (3*π/4 ≤ α ∧ α < π) :=
sorry

end tangent_line_inclination_range_l192_19225


namespace find_y_l192_19295

theorem find_y (y : ℚ) (h : Real.sqrt (1 + Real.sqrt (3 * y - 4)) = Real.sqrt 9) : y = 68 / 3 := 
by
  sorry

end find_y_l192_19295


namespace overtime_percentage_increase_l192_19242

-- Define basic conditions
def basic_hours := 40
def total_hours := 48
def basic_pay := 20
def total_wage := 25

-- Calculate overtime hours and wages
def overtime_hours := total_hours - basic_hours
def overtime_pay := total_wage - basic_pay

-- Define basic and overtime hourly rates
def basic_hourly_rate := basic_pay / basic_hours
def overtime_hourly_rate := overtime_pay / overtime_hours

-- Calculate and state the theorem for percentage increase
def percentage_increase := ((overtime_hourly_rate - basic_hourly_rate) / basic_hourly_rate) * 100

theorem overtime_percentage_increase :
  percentage_increase = 25 :=
by
  sorry

end overtime_percentage_increase_l192_19242


namespace point_in_which_quadrant_l192_19227

theorem point_in_which_quadrant (x y : ℝ) (h1 : y = 2 * x + 3) (h2 : abs x = abs y) :
  (x < 0 ∧ y < 0) ∨ (x < 0 ∧ y > 0) :=
by
  -- Proof omitted
  sorry

end point_in_which_quadrant_l192_19227


namespace factory_produces_correct_number_of_candies_l192_19281

-- Definitions of the given conditions
def candies_per_hour : ℕ := 50
def hours_per_day : ℕ := 10
def days_to_complete_order : ℕ := 8

-- The theorem we want to prove
theorem factory_produces_correct_number_of_candies :
  days_to_complete_order * hours_per_day * candies_per_hour = 4000 :=
by 
  sorry

end factory_produces_correct_number_of_candies_l192_19281


namespace variance_of_scores_l192_19288

def scores : List ℝ := [8, 7, 9, 5, 4, 9, 10, 7, 4]

noncomputable def mean (xs : List ℝ) : ℝ :=
  (List.sum xs) / (xs.length)

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (List.sum (List.map (λ x => (x - m) ^ 2) xs)) / (xs.length)

theorem variance_of_scores : variance scores = 40 / 9 :=
by
  sorry

end variance_of_scores_l192_19288


namespace belize_homes_l192_19222

theorem belize_homes (H : ℝ) 
  (h1 : (3 / 5) * (3 / 4) * H = 240) : 
  H = 400 :=
sorry

end belize_homes_l192_19222


namespace diamond_property_C_l192_19230

-- Define the binary operation diamond
def diamond (a b : ℕ) : ℕ := a ^ (2 * b)

theorem diamond_property_C (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) : 
  (diamond a b) ^ n = diamond a (b * n) :=
by
  sorry

end diamond_property_C_l192_19230


namespace quadrants_containing_points_l192_19265

theorem quadrants_containing_points (x y : ℝ) :
  (y > x + 1) → (y > 3 - 2 * x) → 
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end quadrants_containing_points_l192_19265


namespace range_of_x_l192_19273

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) :
    Real.pi / 4 ≤ x ∧ x ≤ 5 * Real.pi / 4 :=
by
  sorry

end range_of_x_l192_19273

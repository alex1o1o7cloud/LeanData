import Mathlib

namespace solve_arctan_eq_pi_over_3_l591_59124

open Real

theorem solve_arctan_eq_pi_over_3 (x : ℝ) :
  arctan (1 / x) + arctan (1 / x^2) = π / 3 ↔ 
  x = (1 + sqrt (13 + 4 * sqrt 3)) / (2 * sqrt 3) ∨
  x = (1 - sqrt (13 + 4 * sqrt 3)) / (2 * sqrt 3) :=
by
  sorry

end solve_arctan_eq_pi_over_3_l591_59124


namespace trains_crossing_time_l591_59146

/-- Define the length of the first train in meters -/
def length_train1 : ℚ := 200

/-- Define the length of the second train in meters -/
def length_train2 : ℚ := 150

/-- Define the speed of the first train in kilometers per hour -/
def speed_train1_kmph : ℚ := 40

/-- Define the speed of the second train in kilometers per hour -/
def speed_train2_kmph : ℚ := 46

/-- Define conversion factor from kilometers per hour to meters per second -/
def kmph_to_mps : ℚ := 1000 / 3600

/-- Calculate the relative speed in meters per second assuming both trains are moving in the same direction -/
def relative_speed_mps : ℚ := (speed_train2_kmph - speed_train1_kmph) * kmph_to_mps

/-- Calculate the combined length of both trains in meters -/
def combined_length : ℚ := length_train1 + length_train2

/-- Prove the time in seconds for the two trains to cross each other when moving in the same direction is 210 seconds -/
theorem trains_crossing_time :
  (combined_length / relative_speed_mps) = 210 := by
  sorry

end trains_crossing_time_l591_59146


namespace two_circles_tangent_internally_l591_59190

-- Define radii and distance between centers
def R : ℝ := 7
def r : ℝ := 4
def distance_centers : ℝ := 3

-- Statement of the problem
theorem two_circles_tangent_internally :
  distance_centers = R - r → 
  -- Positional relationship: tangent internally
  (distance_centers = abs (R - r)) :=
sorry

end two_circles_tangent_internally_l591_59190


namespace oliver_workout_hours_l591_59151

variable (x : ℕ)

theorem oliver_workout_hours :
  (x + (x - 2) + 2 * x + 2 * (x - 2) = 18) → x = 4 :=
by
  sorry

end oliver_workout_hours_l591_59151


namespace a5_a6_value_l591_59109

def S (n : ℕ) : ℕ := n^3

theorem a5_a6_value : S 6 - S 4 = 152 :=
by
  sorry

end a5_a6_value_l591_59109


namespace infinite_series_converges_l591_59171

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l591_59171


namespace correct_option_is_B_l591_59144

noncomputable def smallest_absolute_value := 0

theorem correct_option_is_B :
  (∀ x : ℝ, |x| ≥ 0) ∧ |(0 : ℝ)| = 0 :=
by
  sorry

end correct_option_is_B_l591_59144


namespace eve_distance_ran_more_l591_59105

variable (ran walked : ℝ)

def eve_distance_difference (ran walked : ℝ) : ℝ :=
  ran - walked

theorem eve_distance_ran_more :
  eve_distance_difference 0.7 0.6 = 0.1 :=
by
  sorry

end eve_distance_ran_more_l591_59105


namespace inscribed_circle_circumference_l591_59117

theorem inscribed_circle_circumference (side_length : ℝ) (h : side_length = 10) : 
  ∃ C : ℝ, C = 2 * Real.pi * (side_length / 2) ∧ C = 10 * Real.pi := 
by 
  sorry

end inscribed_circle_circumference_l591_59117


namespace system1_solution_system2_solution_l591_59104

-- System 1
theorem system1_solution (x y : ℝ) 
  (h1 : y = 2 * x - 3)
  (h2 : 3 * x + 2 * y = 8) : 
  x = 2 ∧ y = 1 := 
by
  sorry

-- System 2
theorem system2_solution (x y : ℝ) 
  (h1 : x + 2 * y = 3)
  (h2 : 2 * x - 4 * y = -10) : 
  x = -1 ∧ y = 2 := 
by
  sorry

end system1_solution_system2_solution_l591_59104


namespace cos_double_beta_alpha_plus_double_beta_l591_59139

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h1 : Real.sin α = Real.sqrt 2 / 10)
variable (h2 : Real.sin β = Real.sqrt 10 / 10)

theorem cos_double_beta :
  Real.cos (2 * β) = 4 / 5 := by 
  sorry

theorem alpha_plus_double_beta :
  α + 2 * β = π / 4 := by 
  sorry

end cos_double_beta_alpha_plus_double_beta_l591_59139


namespace candies_on_second_day_l591_59166

noncomputable def total_candies := 45
noncomputable def days := 5
noncomputable def difference := 3

def arithmetic_sum (n : ℕ) (a₁ d : ℕ) :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

theorem candies_on_second_day (a : ℕ) (h : arithmetic_sum days a difference = total_candies) :
  a + difference = 6 := by
  sorry

end candies_on_second_day_l591_59166


namespace pies_left_l591_59108

theorem pies_left (pies_per_batch : ℕ) (batches : ℕ) (dropped : ℕ) (total_pies : ℕ) (pies_left : ℕ)
  (h1 : pies_per_batch = 5)
  (h2 : batches = 7)
  (h3 : dropped = 8)
  (h4 : total_pies = pies_per_batch * batches)
  (h5 : pies_left = total_pies - dropped) :
  pies_left = 27 := by
  sorry

end pies_left_l591_59108


namespace total_spider_legs_l591_59142

variable (numSpiders : ℕ)
variable (legsPerSpider : ℕ)
axiom h1 : numSpiders = 5
axiom h2 : legsPerSpider = 8

theorem total_spider_legs : numSpiders * legsPerSpider = 40 :=
by
  -- necessary for build without proof.
  sorry

end total_spider_legs_l591_59142


namespace triangle_angle_area_l591_59150

theorem triangle_angle_area
  (A B C : ℝ) (a b c : ℝ)
  (h1 : c * Real.cos B = (2 * a - b) * Real.cos C)
  (h2 : C = Real.pi / 3)
  (h3 : c = 2)
  (h4 : a + b + c = 2 * Real.sqrt 3 + 2) :
  ∃ (area : ℝ), area = (2 * Real.sqrt 3) / 3 :=
by 
  -- Proof is omitted
  sorry

end triangle_angle_area_l591_59150


namespace matching_polygons_pairs_l591_59193

noncomputable def are_matching_pairs (n m : ℕ) : Prop :=
  2 * ((n - 2) * 180 / n) = 3 * (360 / m)

theorem matching_polygons_pairs (n m : ℕ) :
  are_matching_pairs n m → (n, m) = (3, 9) ∨ (n, m) = (4, 6) ∨ (n, m) = (5, 5) ∨ (n, m) = (8, 4) :=
sorry

end matching_polygons_pairs_l591_59193


namespace inequality_AM_GM_HM_l591_59112

theorem inequality_AM_GM_HM (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (hab : a ≠ b) : 
  (a + b) / 2 > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > 2 * (a * b) / (a + b) :=
by
  sorry

end inequality_AM_GM_HM_l591_59112


namespace thirteen_members_divisible_by_13_l591_59168

theorem thirteen_members_divisible_by_13 (B : ℕ) (hB : B < 10) : 
  (∃ B, (2000 + B * 100 + 34) % 13 = 0) ↔ B = 6 :=
by
  sorry

end thirteen_members_divisible_by_13_l591_59168


namespace minimum_value_k_eq_2_l591_59157

noncomputable def quadratic_function_min (a m k : ℝ) (h : 0 < a) : ℝ :=
  a * (-(k / 2)) * (-(k / 2) - k)

theorem minimum_value_k_eq_2 (a m : ℝ) (h : 0 < a) :
  quadratic_function_min a m 2 h = -a := 
by
  unfold quadratic_function_min
  sorry

end minimum_value_k_eq_2_l591_59157


namespace sequence_count_l591_59107

def num_sequences (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem sequence_count :
  let x := 490
  let y := 510
  let a : (n : ℕ) → ℕ := fun n => if n = 0 then 0 else if n = 1000 then 2020 else sorry
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 1000 → (a (k + 1) - a k = 1 ∨ a (k + 1) - a k = 3)) →
  (∃ binomial_coeff, binomial_coeff = num_sequences 1000 490) :=
by sorry

end sequence_count_l591_59107


namespace trees_in_one_row_l591_59148

theorem trees_in_one_row (total_revenue : ℕ) (price_per_apple : ℕ) (apples_per_tree : ℕ) (trees_per_row : ℕ)
  (revenue_condition : total_revenue = 30)
  (price_condition : price_per_apple = 1 / 2)
  (apples_condition : apples_per_tree = 5)
  (trees_condition : trees_per_row = 4) :
  trees_per_row = 4 := by
  sorry

end trees_in_one_row_l591_59148


namespace segment_length_aa_prime_l591_59125

/-- Given points A, B, and C, and their reflections, show that the length of AA' is 8 -/
theorem segment_length_aa_prime
  (A : ℝ × ℝ) (A_reflected : ℝ × ℝ)
  (x₁ y₁ y₁_neg : ℝ) :
  A = (x₁, y₁) →
  A_reflected = (x₁, y₁_neg) →
  y₁_neg = -y₁ →
  y₁ = 4 →
  x₁ = 2 →
  |y₁ - y₁_neg| = 8 :=
sorry

end segment_length_aa_prime_l591_59125


namespace blue_pens_count_l591_59186

-- Definitions based on the conditions
def total_pens (B R : ℕ) : Prop := B + R = 82
def more_blue_pens (B R : ℕ) : Prop := B = R + 6

-- The theorem to prove
theorem blue_pens_count (B R : ℕ) (h1 : total_pens B R) (h2 : more_blue_pens B R) : B = 44 :=
by {
  -- This is where the proof steps would normally go.
  sorry
}

end blue_pens_count_l591_59186


namespace find_larger_number_l591_59170

theorem find_larger_number (a b : ℕ) (h1 : a + b = 96) (h2 : a = b + 12) : a = 54 :=
sorry

end find_larger_number_l591_59170


namespace arithmetic_sequence_1000th_term_l591_59110

theorem arithmetic_sequence_1000th_term (a_1 : ℤ) (d : ℤ) (n : ℤ) (h1 : a_1 = 1) (h2 : d = 3) (h3 : n = 1000) : 
  a_1 + (n - 1) * d = 2998 := 
by
  sorry

end arithmetic_sequence_1000th_term_l591_59110


namespace tall_students_proof_l591_59129

variables (T : ℕ) (Short Average Tall : ℕ)

-- Given in the problem:
def total_students := T = 400
def short_students := Short = 2 * T / 5
def average_height_students := Average = 150

-- Prove:
theorem tall_students_proof (hT : total_students T) (hShort : short_students T Short) (hAverage : average_height_students Average) :
  Tall = T - (Short + Average) :=
by
  sorry

end tall_students_proof_l591_59129


namespace largest_of_three_l591_59106

theorem largest_of_three (a b c : ℝ) (h₁ : a = 43.23) (h₂ : b = 2/5) (h₃ : c = 21.23) :
  max (max a b) c = a :=
by
  sorry

end largest_of_three_l591_59106


namespace total_pencils_l591_59134

   variables (n p t : ℕ)

   -- Condition 1: number of students
   def students := 12

   -- Condition 2: pencils per student
   def pencils_per_student := 3

   -- Theorem statement: Given the conditions, the total number of pencils given by the teacher is 36
   theorem total_pencils : t = students * pencils_per_student :=
   by
   sorry
   
end total_pencils_l591_59134


namespace least_integer_remainder_condition_l591_59119

def is_least_integer_with_remainder_condition (n : ℕ) : Prop :=
  n > 1 ∧ (∀ k ∈ [3, 4, 5, 6, 7, 10, 11], n % k = 1)

theorem least_integer_remainder_condition : ∃ (n : ℕ), is_least_integer_with_remainder_condition n ∧ n = 4621 :=
by
  -- The proof will go here.
  sorry

end least_integer_remainder_condition_l591_59119


namespace find_extrema_l591_59132

noncomputable def function_extrema (x : ℝ) : ℝ :=
  (2 / 3) * Real.cos (3 * x - Real.pi / 6)

theorem find_extrema :
  (function_extrema (Real.pi / 18) = 2 / 3 ∧
   function_extrema (7 * Real.pi / 18) = -(2 / 3)) ∧
  (0 < Real.pi / 18 ∧ Real.pi / 18 < Real.pi / 2) ∧
  (0 < 7 * Real.pi / 18 ∧ 7 * Real.pi / 18 < Real.pi / 2) :=
by
  sorry

end find_extrema_l591_59132


namespace total_sum_of_money_is_71_l591_59189

noncomputable def totalCoins : ℕ := 334
noncomputable def coins20Paise : ℕ := 250
noncomputable def coins25Paise : ℕ := totalCoins - coins20Paise
noncomputable def value20Paise : ℕ := coins20Paise * 20
noncomputable def value25Paise : ℕ := coins25Paise * 25
noncomputable def totalValuePaise : ℕ := value20Paise + value25Paise
noncomputable def totalValueRupees : ℚ := totalValuePaise / 100

theorem total_sum_of_money_is_71 :
  totalValueRupees = 71 := by
  sorry

end total_sum_of_money_is_71_l591_59189


namespace identity_n1_n2_product_l591_59188

theorem identity_n1_n2_product :
  (∃ (N1 N2 : ℤ),
    (∀ x : ℚ, (35 * x - 29) / (x^2 - 3 * x + 2) = N1 / (x - 1) + N2 / (x - 2)) ∧
    N1 * N2 = -246) :=
sorry

end identity_n1_n2_product_l591_59188


namespace smallest_x_with_18_factors_and_factors_18_24_l591_59194

theorem smallest_x_with_18_factors_and_factors_18_24 :
  ∃ (x : ℕ), (∃ (a b : ℕ), x = 2^a * 3^b ∧ 18 ∣ x ∧ 24 ∣ x ∧ (a + 1) * (b + 1) = 18) ∧
    (∀ y, (∃ (c d : ℕ), y = 2^c * 3^d ∧ 18 ∣ y ∧ 24 ∣ y ∧ (c + 1) * (d + 1) = 18) → x ≤ y) :=
by
  sorry

end smallest_x_with_18_factors_and_factors_18_24_l591_59194


namespace man_speed_against_current_l591_59183

theorem man_speed_against_current:
  ∀ (V_current : ℝ) (V_still : ℝ) (current_speed : ℝ),
    V_current = V_still + current_speed →
    V_current = 16 →
    current_speed = 3.2 →
    V_still - current_speed = 9.6 :=
by
  intros V_current V_still current_speed h1 h2 h3
  sorry

end man_speed_against_current_l591_59183


namespace EllenBreadMakingTime_l591_59182

-- Definitions based on the given problem
def RisingTimeTypeA : ℕ → ℝ := λ n => n * 4
def BakingTimeTypeA : ℕ → ℝ := λ n => n * 2.5
def RisingTimeTypeB : ℕ → ℝ := λ n => n * 3.5
def BakingTimeTypeB : ℕ → ℝ := λ n => n * 3

def TotalTime (nA nB : ℕ) : ℝ :=
  (RisingTimeTypeA nA + BakingTimeTypeA nA) +
  (RisingTimeTypeB nB + BakingTimeTypeB nB)

theorem EllenBreadMakingTime :
  TotalTime 3 2 = 32.5 := by
  sorry

end EllenBreadMakingTime_l591_59182


namespace whale_ninth_hour_consumption_l591_59174

-- Define the arithmetic sequence conditions
def first_hour_consumption : ℕ := 10
def common_difference : ℕ := 5

-- Define the total consumption over 12 hours
def total_consumption := 12 * (first_hour_consumption + (first_hour_consumption + 11 * common_difference)) / 2

-- Prove the ninth hour (which is the 8th term) consumption
theorem whale_ninth_hour_consumption :
  total_consumption = 450 →
  first_hour_consumption + 8 * common_difference = 50 := 
by
  intros h
  sorry
  

end whale_ninth_hour_consumption_l591_59174


namespace trig_expr_correct_l591_59198

noncomputable def trig_expr : ℝ := Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) - 
                                   Real.cos (160 * Real.pi / 180) * Real.sin (170 * Real.pi / 180)

theorem trig_expr_correct : trig_expr = 1 / 2 := 
  sorry

end trig_expr_correct_l591_59198


namespace new_number_formed_l591_59128

variable (a b : ℕ)

theorem new_number_formed (ha : a < 10) (hb : b < 10) : 
  ((10 * a + b) * 10 + 2) = 100 * a + 10 * b + 2 := 
by
  sorry

end new_number_formed_l591_59128


namespace eccentricity_range_l591_59184

def hyperbola (a b x y : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def right_branch_hyperbola_P (a b c x y : ℝ) : Prop := hyperbola a b x y ∧ (c = a) ∧ (2 * c = a)

theorem eccentricity_range {a b c : ℝ} (h: hyperbola a b c c) (h1 : 2 * a = 2 * c) (h2 : c = a) :
  1 < (c / a) ∧ (c / a) ≤ (Real.sqrt 10 / 2 : ℝ) := by
  sorry

end eccentricity_range_l591_59184


namespace smallest_prime_with_digit_sum_25_l591_59163

-- Definitions used in Lean statement:
-- 1. Prime predicate based on primality check.
-- 2. Digit sum function.

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Lean 4 statement to prove that the smallest prime whose digits sum to 25 is 1699.

theorem smallest_prime_with_digit_sum_25 : ∃ n : ℕ, is_prime n ∧ digit_sum n = 25 ∧ n = 1699 :=
by
  sorry

end smallest_prime_with_digit_sum_25_l591_59163


namespace remaining_episodes_l591_59152

theorem remaining_episodes (total_seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (H1 : total_seasons = 12) (H2 : episodes_per_season = 20) (H3 : fraction_watched = 1/3) : 
  (total_seasons * episodes_per_season) - (fraction_watched * (total_seasons * episodes_per_season)) = 160 :=
by
  sorry

end remaining_episodes_l591_59152


namespace number_of_band_students_l591_59127

noncomputable def total_students := 320
noncomputable def sports_students := 200
noncomputable def both_activities_students := 60
noncomputable def either_activity_students := 225

theorem number_of_band_students : 
  ∃ B : ℕ, either_activity_students = B + sports_students - both_activities_students ∧ B = 85 :=
by
  sorry

end number_of_band_students_l591_59127


namespace gallery_pieces_total_l591_59177

noncomputable def TotalArtGalleryPieces (A : ℕ) : Prop :=
  let D := (1 : ℚ) / 3 * A
  let N := A - D
  let notDisplayedSculptures := (2 : ℚ) / 3 * N
  let totalSculpturesNotDisplayed := 800
  (4 : ℚ) / 9 * A = 800

theorem gallery_pieces_total (A : ℕ) (h : (TotalArtGalleryPieces A)) : A = 1800 :=
by sorry

end gallery_pieces_total_l591_59177


namespace calculate_product_l591_59185

theorem calculate_product : 6^6 * 3^6 = 34012224 := by
  sorry

end calculate_product_l591_59185


namespace grid_square_count_l591_59187

theorem grid_square_count :
  let width := 6
  let height := 6
  let num_1x1 := (width - 1) * (height - 1)
  let num_2x2 := (width - 2) * (height - 2)
  let num_3x3 := (width - 3) * (height - 3)
  let num_4x4 := (width - 4) * (height - 4)
  num_1x1 + num_2x2 + num_3x3 + num_4x4 = 54 :=
by
  sorry

end grid_square_count_l591_59187


namespace inequality_proof_l591_59136

variable {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + c^2 = 1) : 
  a + b + Real.sqrt 2 * c ≤ 2 := 
by 
  sorry

end inequality_proof_l591_59136


namespace ravish_maximum_marks_l591_59120

theorem ravish_maximum_marks (M : ℝ) (h_pass : 0.40 * M = 80) : M = 200 :=
sorry

end ravish_maximum_marks_l591_59120


namespace sum_of_numerical_coefficients_binomial_l591_59137

theorem sum_of_numerical_coefficients_binomial (a b : ℕ) (n : ℕ) (h : n = 8) :
  let sum_num_coeff := (a + b)^n
  sum_num_coeff = 256 :=
by 
  sorry

end sum_of_numerical_coefficients_binomial_l591_59137


namespace non_adjacent_placements_l591_59101

theorem non_adjacent_placements (n : ℕ) : 
  let total_ways := n^2 * (n^2 - 1)
  let adjacent_ways := 2 * n^2 - 2 * n
  (total_ways - adjacent_ways) = n^4 - 3 * n^2 + 2 * n :=
by
  -- Proof is sorted out
  sorry

end non_adjacent_placements_l591_59101


namespace smallest_integer_to_perfect_cube_l591_59147

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem smallest_integer_to_perfect_cube :
  ∃ n : ℕ, 
    n > 0 ∧ 
    is_perfect_cube (45216 * n) ∧ 
    (∀ m : ℕ, m > 0 ∧ is_perfect_cube (45216 * m) → n ≤ m) ∧ 
    n = 7 := sorry

end smallest_integer_to_perfect_cube_l591_59147


namespace calculate_speed_l591_59138

-- Define the distance and time conditions
def distance : ℝ := 390
def time : ℝ := 4

-- Define the expected answer for speed
def expected_speed : ℝ := 97.5

-- Prove that speed equals expected_speed given the conditions
theorem calculate_speed : (distance / time) = expected_speed :=
by
  -- skipped proof steps
  sorry

end calculate_speed_l591_59138


namespace bubble_gum_cost_l591_59180

theorem bubble_gum_cost (n_pieces : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) 
  (h1 : n_pieces = 136) (h2 : total_cost = 2448) : cost_per_piece = 18 :=
by
  sorry

end bubble_gum_cost_l591_59180


namespace ratio_wheelbarrow_to_earnings_l591_59113

theorem ratio_wheelbarrow_to_earnings :
  let duck_price := 10
  let chicken_price := 8
  let chickens_sold := 5
  let ducks_sold := 2
  let resale_earn := 60
  let total_earnings := chickens_sold * chicken_price + ducks_sold * duck_price
  let wheelbarrow_cost := resale_earn / 2
  (wheelbarrow_cost / total_earnings = 1 / 2) :=
by
  sorry

end ratio_wheelbarrow_to_earnings_l591_59113


namespace harkamal_mangoes_l591_59103

theorem harkamal_mangoes (m : ℕ) (h1: 8 * 70 = 560) (h2 : m * 50 + 560 = 1010) : m = 9 :=
by
  sorry

end harkamal_mangoes_l591_59103


namespace evaluate_expression_l591_59167

theorem evaluate_expression :
  (3^1003 + 7^1004)^2 - (3^1003 - 7^1004)^2 = 5.292 * 10^1003 :=
by sorry

end evaluate_expression_l591_59167


namespace service_cost_is_correct_l591_59154

def service_cost_per_vehicle(cost_per_liter: ℝ)
                            (num_minivans: ℕ) 
                            (num_trucks: ℕ)
                            (total_cost: ℝ) 
                            (minivan_tank_liters: ℝ)
                            (truck_size_increase_pct: ℝ) 
                            (total_fuel: ℝ) 
                            (total_fuel_cost: ℝ) 
                            (total_service_cost: ℝ)
                            (num_vehicles: ℕ) 
                            (service_cost_per_vehicle: ℝ) : Prop :=
  cost_per_liter = 0.70 ∧
  num_minivans = 4 ∧
  num_trucks = 2 ∧
  total_cost = 395.4 ∧
  minivan_tank_liters = 65 ∧
  truck_size_increase_pct = 1.2 ∧
  total_fuel = (4 * minivan_tank_liters) + (2 * (minivan_tank_liters * (1 + truck_size_increase_pct))) ∧
  total_fuel_cost = total_fuel * cost_per_liter ∧
  total_service_cost = total_cost - total_fuel_cost ∧
  num_vehicles = num_minivans + num_trucks ∧
  service_cost_per_vehicle = total_service_cost / num_vehicles

-- Now, we state the theorem we want to prove.
theorem service_cost_is_correct :
  service_cost_per_vehicle 0.70 4 2 395.4 65 1.2 546 382.2 13.2 6 2.2 :=
by {
    sorry
}

end service_cost_is_correct_l591_59154


namespace smallest_b_for_factorization_l591_59116

theorem smallest_b_for_factorization :
  ∃ (b : ℕ), (∀ r s : ℕ, (r * s = 3258) → (b = r + s)) ∧ (∀ c : ℕ, (∀ r' s' : ℕ, (r' * s' = 3258) → (c = r' + s')) → b ≤ c) :=
sorry

end smallest_b_for_factorization_l591_59116


namespace correct_options_l591_59196

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function : Prop := ∀ x : ℝ, f x = f (-x)
def function_definition : Prop := ∀ x : ℝ, (0 < x) → f x = x^2 + x

-- Statements to be proved
def option_A : Prop := f (-1) = 2
def option_B_incorrect : Prop := ¬ (∀ x : ℝ, (f x ≥ f 0) ↔ x ≥ 0) -- Reformulated as not a correct statement
def option_C : Prop := ∀ x : ℝ, x < 0 → f x = x^2 - x
def option_D : Prop := ∀ x : ℝ, (0 < x ∧ x < 2) ↔ f (x - 1) < 2

-- Prove that the correct statements are A, C, and D
theorem correct_options (h_even : is_even_function f) (h_def : function_definition f) :
  option_A f ∧ option_C f ∧ option_D f := by
  sorry

end correct_options_l591_59196


namespace business_value_l591_59160

theorem business_value (h₁ : (2/3 : ℝ) * (3/4 : ℝ) * V = 30000) : V = 60000 :=
by
  -- conditions and definitions go here
  sorry

end business_value_l591_59160


namespace product_third_side_approximation_l591_59143

def triangle_third_side (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

noncomputable def product_of_third_side_lengths : ℝ :=
  Real.sqrt 41 * 3

theorem product_third_side_approximation (a b : ℝ) (h₁ : a = 4) (h₂ : b = 5) :
  ∃ (c₁ c₂ : ℝ), triangle_third_side a b c₁ ∧ triangle_third_side a b c₂ ∧
  abs ((c₁ * c₂) - 19.2) < 0.1 :=
sorry

end product_third_side_approximation_l591_59143


namespace Mike_changed_2_sets_of_tires_l591_59155

theorem Mike_changed_2_sets_of_tires
  (wash_time_per_car : ℕ := 10)
  (oil_change_time_per_car : ℕ := 15)
  (tire_change_time_per_set : ℕ := 30)
  (num_washed_cars : ℕ := 9)
  (num_oil_changes : ℕ := 6)
  (total_work_time_minutes : ℕ := 4 * 60) :
  ((total_work_time_minutes - (num_washed_cars * wash_time_per_car + num_oil_changes * oil_change_time_per_car)) / tire_change_time_per_set) = 2 :=
by
  sorry

end Mike_changed_2_sets_of_tires_l591_59155


namespace average_of_integers_l591_59145

theorem average_of_integers (A B C D : ℤ) (h1 : A < B) (h2 : B < C) (h3 : C < D) (h4 : D = 90) (h5 : 5 ≤ A) (h6 : A ≠ B ∧ B ≠ C ∧ C ≠ D) :
  (A + B + C + D) / 4 = 27 :=
by
  sorry

end average_of_integers_l591_59145


namespace cuboid_length_l591_59158

theorem cuboid_length (b h : ℝ) (A : ℝ) (l : ℝ) : b = 6 → h = 5 → A = 120 → 2 * (l * b + b * h + h * l) = A → l = 30 / 11 :=
by
  intros hb hh hA hSurfaceArea
  rw [hb, hh] at hSurfaceArea
  sorry

end cuboid_length_l591_59158


namespace solve_for_a_l591_59192

theorem solve_for_a (x y a : ℤ) (h1 : x = 1) (h2 : y = 2) (h3 : x - a * y = 3) : a = -1 :=
by
  -- Proof is skipped
  sorry

end solve_for_a_l591_59192


namespace pie_eating_contest_l591_59175

def pies_eaten (Adam Bill Sierra Taylor: ℕ) : ℕ :=
  Adam + Bill + Sierra + Taylor

theorem pie_eating_contest (Bill : ℕ) 
  (Adam_eq_Bill_plus_3 : ∀ B: ℕ, Adam = B + 3)
  (Sierra_eq_2times_Bill : ∀ B: ℕ, Sierra = 2 * B)
  (Sierra_eq_12 : Sierra = 12)
  (Taylor_eq_avg : ∀ A B S: ℕ, Taylor = (A + B + S) / 3)
  : pies_eaten Adam Bill Sierra Taylor = 36 := sorry

end pie_eating_contest_l591_59175


namespace largest_corner_sum_l591_59149

-- Definitions based on the given problem
def faces_labeled : List ℕ := [2, 3, 4, 5, 6, 7]
def opposite_faces : List (ℕ × ℕ) := [(2, 7), (3, 6), (4, 5)]

-- Condition that face 2 cannot be adjacent to face 4
def non_adjacent_faces : List (ℕ × ℕ) := [(2, 4)]

-- Function to check adjacency constraints
def adjacent_allowed (f1 f2 : ℕ) : Bool := 
  ¬ (f1, f2) ∈ non_adjacent_faces ∧ ¬ (f2, f1) ∈ non_adjacent_faces

-- Determine the largest sum of three numbers whose faces meet at a corner
theorem largest_corner_sum : ∃ (a b c : ℕ), a ∈ faces_labeled ∧ b ∈ faces_labeled ∧ c ∈ faces_labeled ∧ 
  (adjacent_allowed a b) ∧ (adjacent_allowed b c) ∧ (adjacent_allowed c a) ∧ 
  a + b + c = 18 := 
sorry

end largest_corner_sum_l591_59149


namespace third_candidate_more_votes_than_john_l591_59181

-- Define the given conditions
def total_votes : ℕ := 1150
def john_votes : ℕ := 150
def remaining_votes : ℕ := total_votes - john_votes
def james_votes : ℕ := (7 * remaining_votes) / 10
def john_and_james_votes : ℕ := john_votes + james_votes
def third_candidate_votes : ℕ := total_votes - john_and_james_votes

-- Stating the problem to prove
theorem third_candidate_more_votes_than_john : third_candidate_votes - john_votes = 150 := 
by
  sorry

end third_candidate_more_votes_than_john_l591_59181


namespace men_in_second_group_l591_59164

theorem men_in_second_group (W : ℝ)
  (h1 : W = 18 * 20)
  (h2 : W = M * 30) :
  M = 12 :=
by
  sorry

end men_in_second_group_l591_59164


namespace inequality_one_inequality_two_l591_59123

variable {a b r s : ℝ}

theorem inequality_one (h_a : 0 < a) (h_b : 0 < b) :
  a^2 * b ≤ 4 * ((a + b) / 3)^3 :=
sorry

theorem inequality_two (h_a : 0 < a) (h_b : 0 < b) (h_r : 0 < r) (h_s : 0 < s) 
  (h_eq : 1 / r + 1 / s = 1) : 
  (a^r / r) + (b^s / s) ≥ a * b :=
sorry

end inequality_one_inequality_two_l591_59123


namespace vehicle_value_last_year_l591_59140

variable (v_this_year v_last_year : ℝ)

theorem vehicle_value_last_year:
  v_this_year = 16000 ∧ v_this_year = 0.8 * v_last_year → v_last_year = 20000 :=
by
  -- Proof steps can be added here, but replaced with sorry as per instructions.
  sorry

end vehicle_value_last_year_l591_59140


namespace value_of_y_l591_59114

variable {x y : ℝ}

theorem value_of_y (h1 : x > 2) (h2 : y > 2) (h3 : 1/x + 1/y = 3/4) (h4 : x * y = 8) : y = 4 :=
sorry

end value_of_y_l591_59114


namespace company_a_percentage_l591_59173

theorem company_a_percentage (total_profits: ℝ) (p_b: ℝ) (profit_b: ℝ) (profit_a: ℝ) :
  p_b = 0.40 →
  profit_b = 60000 →
  profit_a = 90000 →
  total_profits = profit_b / p_b →
  (profit_a / total_profits) * 100 = 60 :=
by
  intros h_pb h_profit_b h_profit_a h_total_profits
  sorry

end company_a_percentage_l591_59173


namespace graph_of_equation_is_two_lines_l591_59118

-- define the condition
def equation_condition (x y : ℝ) : Prop :=
  (x - y) ^ 2 = x ^ 2 + y ^ 2

-- state the theorem
theorem graph_of_equation_is_two_lines :
  ∀ x y : ℝ, equation_condition x y → (x = 0) ∨ (y = 0) :=
by
  intros x y h
  -- proof here
  sorry

end graph_of_equation_is_two_lines_l591_59118


namespace sum_of_next_five_even_integers_l591_59162

theorem sum_of_next_five_even_integers (a : ℕ) (x : ℕ) 
  (h : a = x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) : 
  (x + 10) + (x + 12) + (x + 14) + (x + 16) + (x + 18) = a + 50 := by
  sorry

end sum_of_next_five_even_integers_l591_59162


namespace profit_sharing_l591_59159

theorem profit_sharing
  (A_investment B_investment C_investment total_profit : ℕ)
  (A_share : ℕ)
  (ratio_A ratio_B ratio_C : ℕ)
  (hA : A_investment = 6300)
  (hB : B_investment = 4200)
  (hC : C_investment = 10500)
  (hShare : A_share = 3810)
  (hRatio : ratio_A = 3 ∧ ratio_B = 2 ∧ ratio_C = 5)
  (hTotRatio : ratio_A + ratio_B + ratio_C = 10)
  (hShareCalc : A_share = (3/10) * total_profit) :
  total_profit = 12700 :=
sorry

end profit_sharing_l591_59159


namespace percentage_given_to_close_friends_l591_59178

-- Definitions
def total_boxes : ℕ := 20
def pens_per_box : ℕ := 5
def total_pens : ℕ := total_boxes * pens_per_box
def pens_left_after_classmates : ℕ := 45

-- Proposition
theorem percentage_given_to_close_friends (P : ℝ) :
  total_boxes = 20 → pens_per_box = 5 → pens_left_after_classmates = 45 →
  (3 / 4) * (100 - P) = (pens_left_after_classmates : ℝ) →
  P = 40 :=
by
  intros h_total_boxes h_pens_per_box h_pens_left_after h_eq
  sorry

end percentage_given_to_close_friends_l591_59178


namespace correct_divisor_l591_59133

theorem correct_divisor (X D : ℕ) (h1 : X / 72 = 24) (h2 : X / D = 48) : D = 36 :=
sorry

end correct_divisor_l591_59133


namespace intersection_points_count_l591_59176

def line1 (x y : ℝ) : Prop := 4 * y - 3 * x = 2
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y = 9
def line3 (x y : ℝ) : Prop := x - y = 1

theorem intersection_points_count :
  ∃ p1 p2 p3 : ℝ × ℝ,
  (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
  (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
  (line1 p3.1 p3.2 ∧ line3 p3.1 p3.2) ∧
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3) :=
  sorry

end intersection_points_count_l591_59176


namespace weight_in_kilograms_l591_59197

-- Definitions based on conditions
def weight_of_one_bag : ℕ := 250
def number_of_bags : ℕ := 8

-- Converting grams to kilograms (1000 grams = 1 kilogram)
def grams_to_kilograms (grams : ℕ) : ℕ := grams / 1000

-- Total weight in grams
def total_weight_in_grams : ℕ := weight_of_one_bag * number_of_bags

-- Proof that the total weight in kilograms is 2
theorem weight_in_kilograms : grams_to_kilograms total_weight_in_grams = 2 :=
by
  sorry

end weight_in_kilograms_l591_59197


namespace students_with_one_talent_l591_59161

-- Define the given conditions
def total_students := 120
def cannot_sing := 30
def cannot_dance := 50
def both_skills := 10

-- Define the problem statement
theorem students_with_one_talent :
  (total_students - cannot_sing - both_skills) + (total_students - cannot_dance - both_skills) = 130 :=
by
  sorry

end students_with_one_talent_l591_59161


namespace households_used_both_brands_l591_59102

theorem households_used_both_brands (X : ℕ) : 
  (80 + 60 + X + 3 * X = 260) → X = 30 :=
by
  sorry

end households_used_both_brands_l591_59102


namespace number_of_blue_spotted_fish_l591_59179

theorem number_of_blue_spotted_fish : 
  ∀ (fish_total : ℕ) (one_third_blue : ℕ) (half_spotted : ℕ),
    fish_total = 30 →
    one_third_blue = fish_total / 3 →
    half_spotted = one_third_blue / 2 →
    half_spotted = 5 := 
by
  intros fish_total one_third_blue half_spotted ht htb hhs
  sorry

end number_of_blue_spotted_fish_l591_59179


namespace rabbit_jump_lengths_order_l591_59131

theorem rabbit_jump_lengths_order :
  ∃ (R : ℕ) (G : ℕ) (P : ℕ) (F : ℕ),
    R = 2730 ∧
    R = P + 1100 ∧
    P = F + 150 ∧
    F = G - 200 ∧
    R > G ∧ G > P ∧ P > F :=
  by
  -- calculations
  sorry

end rabbit_jump_lengths_order_l591_59131


namespace simplify_and_evaluate_l591_59191

noncomputable section

def x := Real.sqrt 3 + 1

theorem simplify_and_evaluate :
  (x / (x^2 - 1) / (1 - (1 / (x + 1)))) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l591_59191


namespace enlarged_poster_height_l591_59126

def original_poster_width : ℝ := 3
def original_poster_height : ℝ := 2
def new_poster_width : ℝ := 12

theorem enlarged_poster_height :
  new_poster_width / original_poster_width * original_poster_height = 8 := 
by
  sorry

end enlarged_poster_height_l591_59126


namespace complementary_angles_difference_l591_59199

def complementary_angles (θ1 θ2 : ℝ) : Prop :=
  θ1 + θ2 = 90

theorem complementary_angles_difference:
  ∀ (θ1 θ2 : ℝ), 
  (θ1 / θ2 = 4 / 5) → 
  complementary_angles θ1 θ2 → 
  abs (θ2 - θ1) = 10 :=
by
  sorry

end complementary_angles_difference_l591_59199


namespace problem1_problem2_l591_59169

theorem problem1 (x y : ℝ) (h1 : x - y = 4) (h2 : x > 3) (h3 : y < 1) : 
  2 < x + y ∧ x + y < 6 :=
sorry

theorem problem2 (x y m : ℝ) (h1 : y > 1) (h2 : x < -1) (h3 : x - y = m) : 
  m + 2 < x + y ∧ x + y < -m - 2 :=
sorry

end problem1_problem2_l591_59169


namespace ratio_of_b_l591_59111

theorem ratio_of_b (a b k a1 a2 b1 b2 : ℝ) (h_nonzero_a2 : a2 ≠ 0) (h_nonzero_b12: b1 ≠ 0 ∧ b2 ≠ 0) :
  (a * b = k) →
  (a1 * b1 = a2 * b2) →
  (a1 / a2 = 3 / 5) →
  (b1 / b2 = 5 / 3) := 
sorry

end ratio_of_b_l591_59111


namespace second_concert_attendance_correct_l591_59135

def first_concert_attendance : ℕ := 65899
def additional_people : ℕ := 119
def second_concert_attendance : ℕ := 66018

theorem second_concert_attendance_correct :
  first_concert_attendance + additional_people = second_concert_attendance :=
by sorry

end second_concert_attendance_correct_l591_59135


namespace pythagorean_inequality_l591_59115

variables (a b c : ℝ) (n : ℕ)

theorem pythagorean_inequality (h₀ : a > b) (h₁ : b > c) (h₂ : a^2 = b^2 + c^2) (h₃ : n > 2) : a^n > b^n + c^n :=
sorry

end pythagorean_inequality_l591_59115


namespace inequality_proof_l591_59121

theorem inequality_proof (a : ℝ) : 
  2 * a^4 + 2 * a^2 - 1 ≥ (3 / 2) * (a^2 + a - 1) :=
by
  sorry

end inequality_proof_l591_59121


namespace expression_value_l591_59153

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l591_59153


namespace monotonic_implies_m_l591_59122

noncomputable def cubic_function (x m : ℝ) : ℝ := x^3 + x^2 + m * x + 1

theorem monotonic_implies_m (m : ℝ) :
  (∀ x : ℝ, (3 * x^2 + 2 * x + m) ≥ 0) → m ≥ 1 / 3 :=
  sorry

end monotonic_implies_m_l591_59122


namespace arithmetic_sequence_problem_l591_59156

theorem arithmetic_sequence_problem (a : Nat → Int) (d a1 : Int)
  (h1 : ∀ n, a n = a1 + (n - 1) * d) 
  (h2 : a 1 + 3 * a 8 = 1560) :
  2 * a 9 - a 10 = 507 :=
sorry

end arithmetic_sequence_problem_l591_59156


namespace planted_fraction_l591_59100

theorem planted_fraction (length width radius : ℝ) (h_field : length * width = 24)
  (h_circle : π * radius^2 = π) : (24 - π) / 24 = (24 - π) / 24 :=
by
  -- all proofs are skipped
  sorry

end planted_fraction_l591_59100


namespace page_shoes_l591_59130

/-- Page's initial collection of shoes -/
def initial_collection : ℕ := 80

/-- Page donates 30% of her collection -/
def donation (n : ℕ) : ℕ := n * 30 / 100

/-- Page buys additional shoes -/
def additional_shoes : ℕ := 6

/-- Page's final collection after donation and purchase -/
def final_collection (n : ℕ) : ℕ := (n - donation n) + additional_shoes

/-- Proof that the final collection of shoes is 62 given the initial collection of 80 pairs -/
theorem page_shoes : (final_collection initial_collection) = 62 := 
by sorry

end page_shoes_l591_59130


namespace BC_length_47_l591_59165

theorem BC_length_47 (A B C D : ℝ) (h₁ : A ≠ B) (h₂ : B ≠ C) (h₃ : B ≠ D)
  (h₄ : dist A C = 20) (h₅ : dist A D = 45) (h₆ : dist B D = 13)
  (h₇ : C = 0) (h₈ : D = 0) (h₉ : B = A + 43) :
  dist B C = 47 :=
sorry

end BC_length_47_l591_59165


namespace train_speed_l591_59195

theorem train_speed (v : ℝ) : (∃ t : ℝ, 2 * v + t * v = 285 ∧ t = 285 / 38) → v = 30 :=
by
  sorry

end train_speed_l591_59195


namespace arccos_cos_eq_x_div_3_solutions_l591_59141

theorem arccos_cos_eq_x_div_3_solutions (x : ℝ) :
  (Real.arccos (Real.cos x) = x / 3) ∧ (-3 * Real.pi / 2 ≤ x ∧ x ≤ 3 * Real.pi / 2) 
  ↔ x = -3 * Real.pi / 2 ∨ x = 0 ∨ x = 3 * Real.pi / 2 :=
by
  sorry

end arccos_cos_eq_x_div_3_solutions_l591_59141


namespace power_equality_l591_59172

theorem power_equality (x : ℝ) (n : ℕ) (h : x^(2 * n) = 3) : x^(4 * n) = 9 := 
by 
  sorry

end power_equality_l591_59172

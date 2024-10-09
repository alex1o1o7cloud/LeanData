import Mathlib

namespace problem_statement_l1482_148265

theorem problem_statement : 15 * 30 + 45 * 15 + 15 * 15 = 1350 :=
by
  sorry

end problem_statement_l1482_148265


namespace complement_A_union_B_l1482_148268

-- Define the universal set U, and the sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

-- Lean statement to prove the complement of A ∪ B with respect to U
theorem complement_A_union_B : U \ (A ∪ B) = {7, 8} :=
by
sorry

end complement_A_union_B_l1482_148268


namespace measure_of_angle_A_l1482_148226

theorem measure_of_angle_A (A B : ℝ) (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := 
by 
  sorry

end measure_of_angle_A_l1482_148226


namespace max_value_of_g_l1482_148267

noncomputable def g (x : ℝ) : ℝ :=
  Real.sqrt (x * (80 - x)) + Real.sqrt (x * (10 - x))

theorem max_value_of_g :
  ∃ y_0 N, (∀ x, 0 ≤ x ∧ x ≤ 10 → g x ≤ N) ∧ g y_0 = N ∧ y_0 = 33.75 ∧ N = 22.5 := 
by
  -- Proof goes here.
  sorry

end max_value_of_g_l1482_148267


namespace final_replacement_weight_l1482_148259

theorem final_replacement_weight (W : ℝ) (a b c d e : ℝ) 
  (h1 : a = W / 10)
  (h2 : b = (W - 70 + e) / 10)
  (h3 : b - a = 4)
  (h4 : c = (W - 70 + e - 110 + d) / 10)
  (h5 : c - b = -2)
  (h6 : d = (W - 70 + e - 110 + d + 140 - 90) / 10)
  (h7 : d - c = 5)
  : e = 110 ∧ d = 90 ∧ 140 = e + 50 := sorry

end final_replacement_weight_l1482_148259


namespace area_of_region_l1482_148201

theorem area_of_region :
  ∫ y in (0:ℝ)..(1:ℝ), y ^ (2 / 3) = 3 / 5 :=
by
  sorry

end area_of_region_l1482_148201


namespace determine_a_l1482_148286

-- Define the function f as given in the conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 6

-- Formulate the proof statement
theorem determine_a (a : ℝ) (h : f a (-1) = 8) : a = -2 :=
by {
  sorry
}

end determine_a_l1482_148286


namespace area_of_rectangle_l1482_148272

theorem area_of_rectangle (length width : ℝ) (h_length : length = 47.3) (h_width : width = 24) :
  length * width = 1135.2 :=
by
  sorry -- Skip the proof

end area_of_rectangle_l1482_148272


namespace road_length_l1482_148217

theorem road_length (L : ℝ) (h1 : 300 = 200 + 100)
  (h2 : 50 * 100 = 2.5 / (L / 300))
  (h3 : 75 + 50 = 125)
  (h4 : (125 / 50) * (2.5 / 100) * 200 = L - 2.5) : L = 15 := 
by
  sorry

end road_length_l1482_148217


namespace compare_real_numbers_l1482_148289

theorem compare_real_numbers (a b c d : ℝ) (h1 : a = -1) (h2 : b = 0) (h3 : c = 1) (h4 : d = 2) :
  d > a ∧ d > b ∧ d > c :=
by
  sorry

end compare_real_numbers_l1482_148289


namespace xyz_cubed_over_xyz_eq_21_l1482_148291

open Complex

theorem xyz_cubed_over_xyz_eq_21 {x y z : ℂ} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 18)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 21 :=
sorry

end xyz_cubed_over_xyz_eq_21_l1482_148291


namespace smallest_n_for_identity_matrix_l1482_148204

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![ 1 / 2, -Real.sqrt 3 / 2 ],
    ![ Real.sqrt 3 / 2, 1 / 2]
  ]

theorem smallest_n_for_identity_matrix : ∃ (n : ℕ), n > 0 ∧ 
  ∃ (k : ℕ), rotation_matrix ^ n = 1 ∧ n = 3 :=
by
  sorry

end smallest_n_for_identity_matrix_l1482_148204


namespace compute_a_l1482_148210

theorem compute_a 
  (a b : ℚ) 
  (h : ∃ (x : ℝ), x^3 + (a : ℝ) * x^2 + (b : ℝ) * x - 37 = 0 ∧ x = 2 - 3 * Real.sqrt 3) : 
  a = -55 / 23 :=
by 
  sorry

end compute_a_l1482_148210


namespace mul_mixed_number_eq_l1482_148299

theorem mul_mixed_number_eq :
  99 + 24 / 25 * -5 = -499 - 4 / 5 :=
by
  sorry

end mul_mixed_number_eq_l1482_148299


namespace roots_of_cubic_eq_l1482_148241

theorem roots_of_cubic_eq (r s t a b c d : ℂ) (h1 : a ≠ 0) (h2 : r ≠ 0) (h3 : s ≠ 0) 
  (h4 : t ≠ 0) (hrst : ∀ x : ℂ, a * x ^ 3 + b * x ^ 2 + c * x + d = 0 → (x = r ∨ x = s ∨ x = t) ∧ (x = r <-> r + s + t - x = -b / a)) 
  (Vieta1 : r + s + t = -b / a) (Vieta2 : r * s + r * t + s * t = c / a) (Vieta3 : r * s * t = -d / a) :
  (1 / r ^ 3 + 1 / s ^ 3 + 1 / t ^ 3 = c ^ 3 / d ^ 3) := 
by sorry

end roots_of_cubic_eq_l1482_148241


namespace developer_lots_l1482_148270

theorem developer_lots (acres : ℕ) (cost_per_acre : ℕ) (lot_price : ℕ) 
  (h1 : acres = 4) 
  (h2 : cost_per_acre = 1863) 
  (h3 : lot_price = 828) : 
  ((acres * cost_per_acre) / lot_price) = 9 := 
  by
    sorry

end developer_lots_l1482_148270


namespace bike_ride_ratio_l1482_148229

theorem bike_ride_ratio (J : ℕ) (B : ℕ) (M : ℕ) (hB : B = 17) (hM : M = J + 10) (hTotal : B + J + M = 95) :
  J / B = 2 :=
by
  sorry

end bike_ride_ratio_l1482_148229


namespace min_sum_a_b_l1482_148208

theorem min_sum_a_b (a b : ℕ) (h1 : a ≠ b) (h2 : 0 < a ∧ 0 < b) (h3 : (1/a + 1/b) = 1/12) : a + b = 54 :=
sorry

end min_sum_a_b_l1482_148208


namespace cone_radius_l1482_148247

theorem cone_radius (r l : ℝ)
  (h1 : 6 * Real.pi = Real.pi * r^2 + Real.pi * r * l)
  (h2 : 2 * Real.pi * r = Real.pi * l) :
  r = Real.sqrt 2 :=
by
  sorry

end cone_radius_l1482_148247


namespace blocks_calculation_l1482_148246

theorem blocks_calculation
  (total_amount : ℕ)
  (gift_cost : ℕ)
  (workers_per_block : ℕ)
  (H1  : total_amount = 4000)
  (H2  : gift_cost = 4)
  (H3  : workers_per_block = 100)
  : total_amount / gift_cost / workers_per_block = 10 :=
by
  sorry

end blocks_calculation_l1482_148246


namespace arithmetic_sequence_count_l1482_148261

-- Definitions based on the conditions and question
def sequence_count : ℕ := 314 -- The number of common differences for 315-term sequences
def set_size : ℕ := 2014     -- The maximum number in the set {1, 2, 3, ..., 2014}
def min_seq_length : ℕ := 315 -- The length of the arithmetic sequence

-- Lean 4 statement to verify the number of ways to form the required sequence
theorem arithmetic_sequence_count :
  ∃ (ways : ℕ), ways = 5490 ∧
  (∀ (d : ℕ), 1 ≤ d ∧ d ≤ 6 →
  (set_size - (sequence_count * d - 1)) > 0 → 
  ways = (
    if d = 1 then set_size - sequence_count + 1 else
    if d = 2 then set_size - (sequence_count * 2 - 1) + 1 else
    if d = 3 then set_size - (sequence_count * 3 - 1) + 1 else
    if d = 4 then set_size - (sequence_count * 4 - 1) + 1 else
    if d = 5 then set_size - (sequence_count * 5 - 1) + 1 else
    set_size - (sequence_count * 6 - 1) + 1) - 2
  ) :=
sorry

end arithmetic_sequence_count_l1482_148261


namespace prove_b_div_c_equals_one_l1482_148249

theorem prove_b_div_c_equals_one
  (a b c d : ℕ)
  (h_a : a > 0 ∧ a < 4)
  (h_b : b > 0 ∧ b < 4)
  (h_c : c > 0 ∧ c < 4)
  (h_d : d > 0 ∧ d < 4)
  (h_eq : 4^a + 3^b + 2^c + 1^d = 78) :
  b / c = 1 :=
by
  sorry

end prove_b_div_c_equals_one_l1482_148249


namespace adjacent_angles_l1482_148282

variable (θ : ℝ)

theorem adjacent_angles (h : θ + 3 * θ = 180) : θ = 45 ∧ 3 * θ = 135 :=
by 
  -- This is the place where the proof would go
  -- Here we only declare the statement, not the proof
  sorry

end adjacent_angles_l1482_148282


namespace find_x_l1482_148253

theorem find_x :
  ∃ x : ℕ, (5 * 12) / (x / 3) + 80 = 81 ∧ x = 180 :=
by
  sorry

end find_x_l1482_148253


namespace intersection_A_B_l1482_148283

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | ∃ (n : ℤ), (x : ℝ) = n }

theorem intersection_A_B : A ∩ B = {0, 1} := 
by
  sorry

end intersection_A_B_l1482_148283


namespace doug_age_l1482_148220

theorem doug_age (Qaddama Jack Doug : ℕ) 
  (h1 : Qaddama = Jack + 6)
  (h2 : Jack = Doug - 3)
  (h3 : Qaddama = 19) : 
  Doug = 16 := 
by 
  sorry

end doug_age_l1482_148220


namespace sandy_paid_cost_shop2_l1482_148266

-- Define the conditions
def books_shop1 : ℕ := 65
def cost_shop1 : ℕ := 1380
def books_shop2 : ℕ := 55
def avg_price_per_book : ℕ := 19

-- Calculation of the total amount Sandy paid for the books from the second shop
def cost_shop2 (total_books: ℕ) (avg_price: ℕ) (cost1: ℕ) : ℕ :=
  (total_books * avg_price) - cost1

-- Define the theorem we want to prove
theorem sandy_paid_cost_shop2 : cost_shop2 (books_shop1 + books_shop2) avg_price_per_book cost_shop1 = 900 :=
sorry

end sandy_paid_cost_shop2_l1482_148266


namespace probability_green_given_not_red_l1482_148218

theorem probability_green_given_not_red :
  let total_balls := 20
  let red_balls := 5
  let yellow_balls := 5
  let green_balls := 10
  let non_red_balls := total_balls - red_balls

  let probability_green_given_not_red := (green_balls : ℚ) / (non_red_balls : ℚ)

  probability_green_given_not_red = 2 / 3 :=
by
  sorry

end probability_green_given_not_red_l1482_148218


namespace matrix_pow_expression_l1482_148298

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 4, 2; 0, 2, 3; 0, 0, 1]
def I : Matrix (Fin 3) (Fin 3) ℝ := 1

theorem matrix_pow_expression :
  A^5 - 3 • A^4 = !![0, 4, 2; 0, -1, 3; 0, 0, -2] := by
  sorry

end matrix_pow_expression_l1482_148298


namespace sequence_a2017_l1482_148213

theorem sequence_a2017 (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 3)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 2017 = 2 :=
sorry

end sequence_a2017_l1482_148213


namespace mod_exp_value_l1482_148216

theorem mod_exp_value (m : ℕ) (h1: 0 ≤ m) (h2: m < 9) (h3: 14^4 ≡ m [MOD 9]) : m = 5 :=
by
  sorry

end mod_exp_value_l1482_148216


namespace arrange_letters_of_unique_word_l1482_148276

-- Define the problem parameters
def unique_word := ["M₁", "I₁", "S₁", "S₂", "I₂", "P₁", "P₂", "I₃"]
def word_length := unique_word.length
def arrangement_count := Nat.factorial word_length

-- Theorem statement corresponding to the problem
theorem arrange_letters_of_unique_word :
  arrangement_count = 40320 :=
by
  sorry

end arrange_letters_of_unique_word_l1482_148276


namespace problem_1_problem_2_l1482_148200

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + Real.log x)
def g (x : ℝ) : ℝ := x^2

theorem problem_1 (a : ℝ) (ha : a ≠ 0) : 
  (∀ (x : ℝ), f a x = a * (x + Real.log x)) →
  deriv (f a) 1 = deriv g 1 → a = 1 := 
by 
  sorry

theorem problem_2 (a : ℝ) (ha : 0 < a) (hb : a < 1) (x1 x2 : ℝ) 
  (hx1 : 1 ≤ x1) (hx2 : x2 ≤ 2) (hx12 : x1 ≠ x2) : 
  |f a x1 - f a x2| < |g x1 - g x2| := 
by 
  sorry

end problem_1_problem_2_l1482_148200


namespace multiplicative_inverse_mod_l1482_148219

-- We define our variables
def a := 154
def m := 257
def inv_a := 20

-- Our main theorem stating that inv_a is indeed the multiplicative inverse of a modulo m
theorem multiplicative_inverse_mod : (a * inv_a) % m = 1 := by
  sorry

end multiplicative_inverse_mod_l1482_148219


namespace problem_solution_l1482_148224

theorem problem_solution (a b c : ℤ)
  (h1 : ∀ x : ℤ, |x| ≠ |a|)
  (h2 : ∀ x : ℤ, x^2 ≠ b^2)
  (h3 : ∀ x : ℤ, x * c ≤ 1):
  a + b + c = 0 :=
by sorry

end problem_solution_l1482_148224


namespace find_range_of_a_l1482_148212

noncomputable def f (x : ℝ) : ℝ := (1 / Real.exp x) - (Real.exp x) + 2 * x - (1 / 3) * x ^ 3

theorem find_range_of_a (a : ℝ) (h : f (3 * a ^ 2) + f (2 * a - 1) ≥ 0) : a ∈ Set.Icc (-1 : ℝ) (1 / 3) :=
sorry

end find_range_of_a_l1482_148212


namespace exists_unequal_m_n_l1482_148269

theorem exists_unequal_m_n (a b c : ℕ → ℕ) :
  ∃ (m n : ℕ), m ≠ n ∧ a m ≥ a n ∧ b m ≥ b n ∧ c m ≥ c n :=
sorry

end exists_unequal_m_n_l1482_148269


namespace marvin_next_birthday_monday_l1482_148233

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

def day_of_week_after_leap_years (start_day : ℕ) (leap_years : ℕ) : ℕ :=
  (start_day + 2 * leap_years) % 7

def next_birthday_on_monday (year : ℕ) (start_day : ℕ) : ℕ :=
  let next_day := day_of_week_after_leap_years start_day ((year - 2012)/4)
  year + 4 * ((7 - next_day + 1) / 2)

theorem marvin_next_birthday_monday : next_birthday_on_monday 2012 3 = 2016 :=
by sorry

end marvin_next_birthday_monday_l1482_148233


namespace largest_three_digit_geometric_sequence_l1482_148236

-- Definitions based on conditions
def is_three_digit_integer (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def digits_distinct (n : ℕ) : Prop := 
  let d₁ := n / 100
  let d₂ := (n / 10) % 10
  let d₃ := n % 10
  d₁ ≠ d₂ ∧ d₂ ≠ d₃ ∧ d₁ ≠ d₃
def geometric_sequence (n : ℕ) : Prop :=
  let d₁ := n / 100
  let d₂ := (n / 10) % 10
  let d₃ := n % 10
  d₁ != 0 ∧ d₂ != 0  ∧ d₃ != 0 ∧ 
  (∃ r: ℚ, d₂ = d₁ * r ∧ d₃ = d₂ * r)

-- Theorem statement
theorem largest_three_digit_geometric_sequence : 
  ∃ n : ℕ, is_three_digit_integer n ∧ digits_distinct n ∧ geometric_sequence n ∧ n = 964 :=
sorry

end largest_three_digit_geometric_sequence_l1482_148236


namespace typing_time_l1482_148280

-- Definitions based on the problem conditions
def initial_typing_speed : ℕ := 212
def speed_decrease : ℕ := 40
def words_in_document : ℕ := 3440

-- Definition for Barbara's new typing speed
def new_typing_speed : ℕ := initial_typing_speed - speed_decrease

-- Lean proof statement: Proving the time to finish typing is 20 minutes
theorem typing_time :
  (words_in_document / new_typing_speed) = 20 :=
by sorry

end typing_time_l1482_148280


namespace traffic_lights_states_l1482_148203

theorem traffic_lights_states (n k : ℕ) : 
  (k ≤ n) → 
  (∃ (ways : ℕ), ways = 3^k * 2^(n - k)) :=
by
  sorry

end traffic_lights_states_l1482_148203


namespace factorize_expression_l1482_148228

theorem factorize_expression (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := 
by 
  sorry

end factorize_expression_l1482_148228


namespace angle_ratio_in_triangle_l1482_148278

theorem angle_ratio_in_triangle
  (triangle : Type)
  (A B C P Q M : triangle)
  (angle : triangle → triangle → triangle → ℝ)
  (ABC_half : angle A B Q = angle Q B C)
  (BP_BQ_bisect_ABC : angle A B P = angle P B Q)
  (BM_bisects_PBQ : angle M B Q = angle M B P)
  : angle M B Q / angle A B Q = 1 / 4 :=
by 
  sorry

end angle_ratio_in_triangle_l1482_148278


namespace max_elevation_l1482_148227

def elevation (t : ℝ) : ℝ := 144 * t - 18 * t^2

theorem max_elevation : ∃ t : ℝ, elevation t = 288 :=
by
  use 4
  sorry

end max_elevation_l1482_148227


namespace tetrahedron_side_length_l1482_148285

theorem tetrahedron_side_length (s : ℝ) (area : ℝ) (d : ℝ) :
  area = 16 → s^2 = area → d = s * Real.sqrt 2 → 4 * Real.sqrt 2 = d :=
by
  intros _ h1 h2
  sorry

end tetrahedron_side_length_l1482_148285


namespace number_of_people_in_group_l1482_148215

theorem number_of_people_in_group :
  ∀ (N : ℕ), (75 - 35) = 5 * N → N = 8 :=
by
  intros N h
  sorry

end number_of_people_in_group_l1482_148215


namespace opposite_of_neg_2023_l1482_148273

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l1482_148273


namespace max_sum_of_arithmetic_sequence_l1482_148256

theorem max_sum_of_arithmetic_sequence 
  (d : ℤ) (a₁ a₃ a₅ a₁₅ : ℤ) (S : ℕ → ℤ)
  (h₁ : d ≠ 0)
  (h₂ : a₃ = a₁ + 2 * d)
  (h₃ : a₅ = a₃ + 2 * d)
  (h₄ : a₁₅ = a₅ + 10 * d)
  (h_geom : a₃ * a₃ = a₅ * a₁₅)
  (h_a₁ : a₁ = 3)
  (h_S : ∀ n, S n = n * a₁ + (n * (n - 1) / 2) * d) :
  ∃ n, S n = 4 :=
by
  sorry

end max_sum_of_arithmetic_sequence_l1482_148256


namespace milk_students_l1482_148202

theorem milk_students (T : ℕ) (h1 : (1 / 4) * T = 80) : (3 / 4) * T = 240 := by
  sorry

end milk_students_l1482_148202


namespace f_difference_l1482_148222

noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom local_f : ∀ x : ℝ, -2 < x ∧ x < 0 → f x = 2^x

-- State the problem
theorem f_difference :
  f 2012 - f 2011 = -1 / 2 := sorry

end f_difference_l1482_148222


namespace value_multiplied_by_15_l1482_148292

theorem value_multiplied_by_15 (x : ℝ) (h : 3.6 * x = 10.08) : x * 15 = 42 :=
sorry

end value_multiplied_by_15_l1482_148292


namespace problem_1_problem_2_l1482_148231

noncomputable section

variables {A B C : ℝ} {a b c : ℝ}

-- Condition definitions
def triangle_conditions (A B : ℝ) (a b c : ℝ) : Prop :=
  b = 3 ∧ c = 1 ∧ A = 2 * B

-- Problem 1: Prove that a = 2 * sqrt(3)
theorem problem_1 {A B C a : ℝ} (h : triangle_conditions A B a b c) : a = 2 * Real.sqrt 3 := sorry

-- Problem 2: Prove the value of cos(2A + π/6)
theorem problem_2 {A B C a : ℝ} (h : triangle_conditions A B a b c) : 
  Real.cos (2 * A + Real.pi / 6) = (4 * Real.sqrt 2 - 7 * Real.sqrt 3) / 18 := sorry

end problem_1_problem_2_l1482_148231


namespace base8_to_base10_12345_l1482_148245

theorem base8_to_base10_12345 : (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 5349 := by
  sorry

end base8_to_base10_12345_l1482_148245


namespace car_B_speed_is_50_l1482_148252

def car_speeds (v_A v_B : ℕ) (d_init d_ahead t : ℝ) : Prop :=
  v_A * t = v_B * t + d_init + d_ahead

theorem car_B_speed_is_50 :
  car_speeds 58 50 10 8 2.25 :=
by
  sorry

end car_B_speed_is_50_l1482_148252


namespace area_of_rectangular_region_l1482_148288

-- Mathematical Conditions
variables (a b c d : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

-- Lean 4 Statement of the proof problem
theorem area_of_rectangular_region :
  (a + b) * (2 * d + c) = 2 * a * d + a * c + 2 * b * d + b * c :=
by sorry

end area_of_rectangular_region_l1482_148288


namespace Bill_Sunday_miles_l1482_148296

-- Definitions based on problem conditions
def Bill_Saturday (B : ℕ) : ℕ := B
def Bill_Sunday (B : ℕ) : ℕ := B + 4
def Julia_Sunday (B : ℕ) : ℕ := 2 * (B + 4)
def Alex_Total (B : ℕ) : ℕ := B + 2

-- Total miles equation based on conditions
def total_miles (B : ℕ) : ℕ := Bill_Saturday B + Bill_Sunday B + Julia_Sunday B + Alex_Total B

-- Proof statement
theorem Bill_Sunday_miles (B : ℕ) (h : total_miles B = 54) : Bill_Sunday B = 14 :=
by {
  -- calculations and proof would go here if not omitted
  sorry
}

end Bill_Sunday_miles_l1482_148296


namespace pyramid_volume_l1482_148290

theorem pyramid_volume (VW WX VZ : ℝ) (h1 : VW = 10) (h2 : WX = 5) (h3 : VZ = 8)
  (h_perp1 : ∀ (V W Z : ℝ), V ≠ W → V ≠ Z → Z ≠ W → W = 0 ∧ Z = 0)
  (h_perp2 : ∀ (V W X : ℝ), V ≠ W → V ≠ X → X ≠ W → W = 0 ∧ X = 0) :
  let area_base := VW * WX
  let height := VZ
  let volume := 1 / 3 * area_base * height
  volume = 400 / 3 := by
  sorry

end pyramid_volume_l1482_148290


namespace find_b_l1482_148237

theorem find_b (x y z a b : ℝ) (h1 : x + y = 2) (h2 : xy - z^2 = a) (h3 : b = x + y + z) : b = 2 :=
by
  sorry

end find_b_l1482_148237


namespace find_k_l1482_148251

theorem find_k (k : ℕ) (hk : 0 < k) (h : (k + 4) / (k^2 - 1) = 9 / 35) : k = 14 :=
by
  sorry

end find_k_l1482_148251


namespace sum_of_two_squares_iff_double_sum_of_two_squares_l1482_148242

theorem sum_of_two_squares_iff_double_sum_of_two_squares (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2 * x = u^2 + v^2) :=
by
  sorry

end sum_of_two_squares_iff_double_sum_of_two_squares_l1482_148242


namespace number_of_chlorine_atoms_l1482_148295

def molecular_weight_of_aluminum : ℝ := 26.98
def molecular_weight_of_chlorine : ℝ := 35.45
def molecular_weight_of_compound : ℝ := 132.0

theorem number_of_chlorine_atoms :
  ∃ n : ℕ, molecular_weight_of_compound = molecular_weight_of_aluminum + n * molecular_weight_of_chlorine ∧ n = 3 :=
by
  sorry

end number_of_chlorine_atoms_l1482_148295


namespace jeans_discount_rates_l1482_148281

theorem jeans_discount_rates
    (M F P : ℝ) 
    (regular_price_moose jeans_regular_price_fox jeans_regular_price_pony : ℝ)
    (moose_count fox_count pony_count : ℕ)
    (total_discount : ℝ) :
    regular_price_moose = 20 →
    regular_price_fox = 15 →
    regular_price_pony = 18 →
    moose_count = 2 →
    fox_count = 3 →
    pony_count = 2 →
    total_discount = 12.48 →
    (M + F + P = 0.32) →
    (F + P = 0.20) →
    (moose_count * M * regular_price_moose + fox_count * F * regular_price_fox + pony_count * P * regular_price_pony = total_discount) →
    M = 0.12 ∧ F = 0.0533 ∧ P = 0.1467 :=
by
  intros
  sorry -- The proof is not required

end jeans_discount_rates_l1482_148281


namespace problem_inequality_solution_problem_prove_inequality_l1482_148244

-- Function definition for f(x)
def f (x : ℝ) := |2 * x - 3| + |2 * x + 3|

-- Problem 1: Prove the solution set for the inequality f(x) ≤ 8
theorem problem_inequality_solution (x : ℝ) : f x ≤ 8 ↔ -2 ≤ x ∧ x ≤ 2 :=
sorry

-- Problem 2: Prove a + 2b + 3c ≥ 9 given conditions
theorem problem_prove_inequality (a b c : ℝ) (M : ℝ) (h1 : M = 6)
  (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) (h5 : 1 / a + 1 / (2 * b) + 1 / (3 * c) = M / 6) :
  a + 2 * b + 3 * c ≥ 9 :=
sorry

end problem_inequality_solution_problem_prove_inequality_l1482_148244


namespace george_total_blocks_l1482_148205

-- Definitions (conditions).
def large_boxes : ℕ := 5
def small_boxes_per_large_box : ℕ := 8
def blocks_per_small_box : ℕ := 9
def individual_blocks : ℕ := 6

-- Mathematical proof problem statement.
theorem george_total_blocks :
  (large_boxes * small_boxes_per_large_box * blocks_per_small_box + individual_blocks) = 366 :=
by
  -- Placeholder for proof.
  sorry

end george_total_blocks_l1482_148205


namespace jenny_reading_time_l1482_148240

theorem jenny_reading_time 
  (days : ℕ)
  (words_first_book : ℕ)
  (words_second_book : ℕ)
  (words_third_book : ℕ)
  (reading_speed : ℕ) : 
  days = 10 →
  words_first_book = 200 →
  words_second_book = 400 →
  words_third_book = 300 →
  reading_speed = 100 →
  (words_first_book + words_second_book + words_third_book) / reading_speed / days * 60 = 54 :=
by
  intros hdays hwords1 hwords2 hwords3 hspeed
  rw [hdays, hwords1, hwords2, hwords3, hspeed]
  norm_num
  sorry

end jenny_reading_time_l1482_148240


namespace simplify_sqrt_450_l1482_148232

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l1482_148232


namespace tan_C_in_triangle_l1482_148230

theorem tan_C_in_triangle (A B C : ℝ) (hA : Real.tan A = 1 / 2) (hB : Real.cos B = 3 * Real.sqrt 10 / 10) :
  Real.tan C = -1 :=
sorry

end tan_C_in_triangle_l1482_148230


namespace gold_coins_equality_l1482_148260

theorem gold_coins_equality (pouches : List ℕ) 
  (h_pouches_length : pouches.length = 9)
  (h_pouches_sum : pouches.sum = 60)
  : (∃ s_2 : List (List ℕ), s_2.length = 2 ∧ ∀ l ∈ s_2, l.sum = 30) ∧
    (∃ s_3 : List (List ℕ), s_3.length = 3 ∧ ∀ l ∈ s_3, l.sum = 20) ∧
    (∃ s_4 : List (List ℕ), s_4.length = 4 ∧ ∀ l ∈ s_4, l.sum = 15) ∧
    (∃ s_5 : List (List ℕ), s_5.length = 5 ∧ ∀ l ∈ s_5, l.sum = 12) :=
sorry

end gold_coins_equality_l1482_148260


namespace sin_double_angle_value_l1482_148209

open Real

theorem sin_double_angle_value (x : ℝ) 
  (h1 : sin (x + π/3) * cos (x - π/6) + sin (x - π/6) * cos (x + π/3) = 5 / 13)
  (h2 : -π/3 ≤ x ∧ x ≤ π/6) :
  sin (2 * x) = (5 * sqrt 3 - 12) / 26 :=
by
  sorry

end sin_double_angle_value_l1482_148209


namespace max_cells_intersected_10_radius_circle_l1482_148221

noncomputable def max_cells_intersected_by_circle (radius : ℝ) (cell_size : ℝ) : ℕ :=
  if radius = 10 ∧ cell_size = 1 then 80 else 0

theorem max_cells_intersected_10_radius_circle :
  max_cells_intersected_by_circle 10 1 = 80 :=
sorry

end max_cells_intersected_10_radius_circle_l1482_148221


namespace value_of_a_cube_l1482_148235

-- We define the conditions given in the problem.
def A (a : ℤ) : Set ℤ := {5, a^2 + 2 * a + 4}
def a_satisfies (a : ℤ) : Prop := 7 ∈ A a

-- We state the theorem.
theorem value_of_a_cube (a : ℤ) (h1 : a_satisfies a) : a^3 = 1 ∨ a^3 = -27 := by
  sorry

end value_of_a_cube_l1482_148235


namespace single_reduction_equivalent_l1482_148284

/-- If a price is first reduced by 25%, and the new price is further reduced by 30%, 
the single percentage reduction equivalent to these two reductions together is 47.5%. -/
theorem single_reduction_equivalent :
  ∀ P : ℝ, (1 - 0.25) * (1 - 0.30) * P = P * (1 - 0.475) :=
by
  intros
  sorry

end single_reduction_equivalent_l1482_148284


namespace parabola_directrix_eq_l1482_148225

theorem parabola_directrix_eq (p : ℝ) (h : y^2 = 2 * x ∧ p = 1) : x = -p / 2 := by
  sorry

end parabola_directrix_eq_l1482_148225


namespace positive_difference_l1482_148243

/-- Pauline deposits 10,000 dollars into an account with 4% compound interest annually. -/
def Pauline_initial_deposit : ℝ := 10000
def Pauline_interest_rate : ℝ := 0.04
def Pauline_years : ℕ := 12

/-- Quinn deposits 10,000 dollars into an account with 6% simple interest annually. -/
def Quinn_initial_deposit : ℝ := 10000
def Quinn_interest_rate : ℝ := 0.06
def Quinn_years : ℕ := 12

/-- Pauline's balance after 12 years -/
def Pauline_balance : ℝ := Pauline_initial_deposit * (1 + Pauline_interest_rate) ^ Pauline_years

/-- Quinn's balance after 12 years -/
def Quinn_balance : ℝ := Quinn_initial_deposit * (1 + Quinn_interest_rate * Quinn_years)

/-- The positive difference between Pauline's and Quinn's balances after 12 years is $1189 -/
theorem positive_difference :
  |Quinn_balance - Pauline_balance| = 1189 := 
sorry

end positive_difference_l1482_148243


namespace xy_leq_half_x_squared_plus_y_squared_l1482_148238

theorem xy_leq_half_x_squared_plus_y_squared (x y : ℝ) : x * y ≤ (x^2 + y^2) / 2 := 
by 
  sorry

end xy_leq_half_x_squared_plus_y_squared_l1482_148238


namespace regular_tetrahedron_surface_area_l1482_148207

theorem regular_tetrahedron_surface_area {h : ℝ} (h_pos : h > 0) :
  ∃ (S : ℝ), S = (3 * h^2 * Real.sqrt 3) / 2 :=
sorry

end regular_tetrahedron_surface_area_l1482_148207


namespace lions_after_one_year_l1482_148277

def initial_lions : ℕ := 100
def birth_rate : ℕ := 5
def death_rate : ℕ := 1
def months_in_year : ℕ := 12

theorem lions_after_one_year : 
  initial_lions + (birth_rate * months_in_year) - (death_rate * months_in_year) = 148 :=
by
  sorry

end lions_after_one_year_l1482_148277


namespace triangle_groups_count_l1482_148264

theorem triangle_groups_count (total_points collinear_groups groups_of_three total_combinations : ℕ)
    (h1 : total_points = 12)
    (h2 : collinear_groups = 16)
    (h3 : groups_of_three = (total_points.choose 3))
    (h4 : total_combinations = groups_of_three - collinear_groups) :
    total_combinations = 204 :=
by
  -- This is where the proof would go
  sorry

end triangle_groups_count_l1482_148264


namespace betty_sugar_l1482_148211

theorem betty_sugar (f s : ℝ) (hf1 : f ≥ 8 + (3 / 4) * s) (hf2 : f ≤ 3 * s) : s ≥ 4 := 
sorry

end betty_sugar_l1482_148211


namespace cannot_determine_right_triangle_l1482_148294

-- Definitions of conditions
variables {a b c : ℕ}
variables {angle_A angle_B angle_C : ℕ}

-- Context for the proof
def is_right_angled_triangle_via_sides (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def triangle_angle_sum_theorem (angle_A angle_B angle_C : ℕ) : Prop :=
  angle_A + angle_B + angle_C = 180

-- Statements for conditions as used in the problem
def condition_A (a2 b2 c2 : ℕ) : Prop :=
  a2 = 1 ∧ b2 = 2 ∧ c2 = 3

def condition_B (a b c : ℕ) : Prop :=
  a = 3 ∧ b = 4 ∧ c = 5

def condition_C (angle_A angle_B angle_C : ℕ) : Prop :=
  angle_A + angle_B = angle_C

def condition_D (angle_A angle_B angle_C : ℕ) : Prop :=
  angle_A = 45 ∧ angle_B = 60 ∧ angle_C = 75

-- Proof statement
theorem cannot_determine_right_triangle (a b c angle_A angle_B angle_C : ℕ) :
  condition_D angle_A angle_B angle_C →
  ¬(is_right_angled_triangle_via_sides a b c) :=
sorry

end cannot_determine_right_triangle_l1482_148294


namespace michael_clean_times_in_one_year_l1482_148287

-- Definitions from the conditions
def baths_per_week : ℕ := 2
def showers_per_week : ℕ := 1
def weeks_per_year : ℕ := 52

-- Theorem statement for the proof problem
theorem michael_clean_times_in_one_year :
  (baths_per_week + showers_per_week) * weeks_per_year = 156 :=
by
  sorry

end michael_clean_times_in_one_year_l1482_148287


namespace not_p_and_p_l1482_148254

theorem not_p_and_p (p : Prop) : ¬ (p ∧ ¬ p) :=
by 
  sorry

end not_p_and_p_l1482_148254


namespace smallest_solution_l1482_148250

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l1482_148250


namespace no_a_satisfy_quadratic_equation_l1482_148223

theorem no_a_satisfy_quadratic_equation :
  ∀ (a : ℕ), (a > 0) ∧ (a ≤ 100) ∧
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁ * x₂ = 2 * a^2 ∧ x₁ + x₂ = -(3*a + 1)) → false := by
  sorry

end no_a_satisfy_quadratic_equation_l1482_148223


namespace tenth_term_in_sequence_l1482_148263

def seq (n : ℕ) : ℚ :=
  (-1) ^ (n + 1) * ((2 * n - 1) / (n ^ 2 + 1))

theorem tenth_term_in_sequence :
  seq 10 = -19 / 101 :=
by
  -- Proof omitted
  sorry

end tenth_term_in_sequence_l1482_148263


namespace xy_conditions_l1482_148293

theorem xy_conditions (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : x * y = 1) : x^2 + 4 * y^2 = 60 :=
by
  sorry

end xy_conditions_l1482_148293


namespace simplify_fraction_l1482_148239

/-- Given the numbers 180 and 270, prove that 180 / 270 is equal to 2 / 3 -/
theorem simplify_fraction : (180 / 270 : ℚ) = 2 / 3 := 
sorry

end simplify_fraction_l1482_148239


namespace balance_relationship_l1482_148248

theorem balance_relationship (x : ℕ) (hx : 0 ≤ x ∧ x ≤ 5) : 
  ∃ y : ℝ, y = 200 - 36 * x := 
sorry

end balance_relationship_l1482_148248


namespace two_x_plus_two_y_value_l1482_148297

theorem two_x_plus_two_y_value (x y : ℝ) (h1 : x^2 - y^2 = 8) (h2 : x - y = 6) : 2 * x + 2 * y = 8 / 3 := 
by sorry

end two_x_plus_two_y_value_l1482_148297


namespace max_area_of_right_angled_isosceles_triangle_l1482_148206

theorem max_area_of_right_angled_isosceles_triangle (a b : ℝ) (h₁ : a = 12) (h₂ : b = 15) :
  ∃ A : ℝ, A = 72 ∧ 
  (∀ (x : ℝ), x ≤ min a b → (1 / 2) * x^2 ≤ A) :=
by
  use 72
  sorry

end max_area_of_right_angled_isosceles_triangle_l1482_148206


namespace number_division_reduction_l1482_148262

theorem number_division_reduction (x : ℕ) (h : x / 3 = x - 48) : x = 72 := 
sorry

end number_division_reduction_l1482_148262


namespace rational_sqrts_l1482_148271

def is_rational (n : ℝ) : Prop := ∃ (q : ℚ), n = q

theorem rational_sqrts 
  (x y z : ℝ) 
  (hxr : is_rational x) 
  (hyr : is_rational y) 
  (hzr : is_rational z)
  (hw : is_rational (Real.sqrt x + Real.sqrt y + Real.sqrt z)) :
  is_rational (Real.sqrt x) ∧ is_rational (Real.sqrt y) ∧ is_rational (Real.sqrt z) :=
sorry

end rational_sqrts_l1482_148271


namespace mr_lee_broke_even_l1482_148258

theorem mr_lee_broke_even (sp1 sp2 : ℝ) (p1_loss2 : ℝ) (c1 c2 : ℝ) (h1 : sp1 = 1.50) (h2 : sp2 = 1.50) 
    (h3 : c1 = sp1 / 1.25) (h4 : c2 = sp2 / 0.8333) (h5 : p1_loss2 = (sp1 - c1) + (sp2 - c2)) : 
  p1_loss2 = 0 :=
by 
  sorry

end mr_lee_broke_even_l1482_148258


namespace fraction_calculation_correct_l1482_148275

noncomputable def calculate_fraction : ℚ :=
  let numerator := (1 / 2) - (1 / 3)
  let denominator := (3 / 4) + (1 / 8)
  numerator / denominator

theorem fraction_calculation_correct : calculate_fraction = 4 / 21 := 
  by
    sorry

end fraction_calculation_correct_l1482_148275


namespace percentage_increase_of_cars_l1482_148279

theorem percentage_increase_of_cars :
  ∀ (initial final : ℕ), initial = 24 → final = 48 → ((final - initial) * 100 / initial) = 100 :=
by
  intros
  sorry

end percentage_increase_of_cars_l1482_148279


namespace max_newsstands_six_corridors_l1482_148274

def number_of_intersections (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem max_newsstands_six_corridors : number_of_intersections 6 = 15 := 
by sorry

end max_newsstands_six_corridors_l1482_148274


namespace expansion_dissimilar_terms_count_l1482_148257

def number_of_dissimilar_terms (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

theorem expansion_dissimilar_terms_count :
  number_of_dissimilar_terms 7 4 = 120 := by
  sorry

end expansion_dissimilar_terms_count_l1482_148257


namespace inequality_l1482_148234

theorem inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 3) : 
  (1 / (8 * a^2 - 18 * a + 11)) + (1 / (8 * b^2 - 18 * b + 11)) + (1 / (8 * c^2 - 18 * c + 11)) ≤ 3 := 
sorry

end inequality_l1482_148234


namespace partI_l1482_148255

noncomputable def f (x : ℝ) : ℝ := abs (1 - 1/x)

theorem partI (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) (h4 : f a = f b) :
  a * b > 1 :=
  sorry

end partI_l1482_148255


namespace quadratic_polynomial_value_l1482_148214

noncomputable def f (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

theorem quadratic_polynomial_value (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h1 : f 1 (-(a + b + c)) (a*b + b*c + a*c) a = b * c)
  (h2 : f 1 (-(a + b + c)) (a*b + b*c + a*c) b = c * a)
  (h3 : f 1 (-(a + b + c)) (a*b + b*c + a*c) c = a * b) :
  f 1 (-(a + b + c)) (a*b + b*c + a*c) (a + b + c) = a * b + b * c + a * c := sorry

end quadratic_polynomial_value_l1482_148214

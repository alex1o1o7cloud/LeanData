import Mathlib

namespace NUMINAMATH_GPT_compute_c_minus_d_squared_eq_0_l1586_158652

-- Defining conditions
def multiples_of_n_under_m (n m : ℕ) : ℕ :=
  (m - 1) / n

-- Defining the specific values
def c : ℕ := multiples_of_n_under_m 9 60
def d : ℕ := multiples_of_n_under_m 9 60  -- Since every multiple of 9 is a multiple of 3

theorem compute_c_minus_d_squared_eq_0 : (c - d) ^ 2 = 0 := by
  sorry

end NUMINAMATH_GPT_compute_c_minus_d_squared_eq_0_l1586_158652


namespace NUMINAMATH_GPT_simplify_expression_l1586_158644

variable (p : ℤ)

-- Defining the given expression
def initial_expression : ℤ := ((5 * p + 1) - 2 * p * 4) * 3 + (4 - 1 / 3) * (6 * p - 9)

-- Statement asserting the simplification
theorem simplify_expression : initial_expression p = 13 * p - 30 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l1586_158644


namespace NUMINAMATH_GPT_ratio_diagonals_of_squares_l1586_158670

variable (d₁ d₂ : ℝ)

theorem ratio_diagonals_of_squares (h : ∃ k : ℝ, d₂ = k * d₁) (h₁ : 1 < k ∧ k < 9) : 
  (∃ k : ℝ, 4 * (d₂ / Real.sqrt 2) = k * 4 * (d₁ / Real.sqrt 2)) → k = 5 := by
  sorry

end NUMINAMATH_GPT_ratio_diagonals_of_squares_l1586_158670


namespace NUMINAMATH_GPT_valid_x_for_expression_l1586_158686

theorem valid_x_for_expression :
  (∃ x : ℝ, x = 8 ∧ (10 - x ≥ 0) ∧ (x - 4 ≠ 0)) ↔ (∃ x : ℝ, x = 8) :=
by
  sorry

end NUMINAMATH_GPT_valid_x_for_expression_l1586_158686


namespace NUMINAMATH_GPT_johns_remaining_money_l1586_158630

theorem johns_remaining_money (H1 : ∃ (n : ℕ), n = 5376) (H2 : 5376 = 5 * 8^3 + 3 * 8^2 + 7 * 8^1 + 6) :
  (2814 - 1350 = 1464) :=
by {
  sorry
}

end NUMINAMATH_GPT_johns_remaining_money_l1586_158630


namespace NUMINAMATH_GPT_gecko_insects_eaten_l1586_158624

theorem gecko_insects_eaten
    (G : ℕ)  -- Number of insects each gecko eats
    (H1 : 5 * G + 3 * (2 * G) = 66) :  -- Total insects eaten condition
    G = 6 :=  -- Expected number of insects each gecko eats
by
  sorry

end NUMINAMATH_GPT_gecko_insects_eaten_l1586_158624


namespace NUMINAMATH_GPT_chocolate_truffles_sold_l1586_158609

def fudge_sold_pounds : ℕ := 20
def price_per_pound_fudge : ℝ := 2.50
def price_per_truffle : ℝ := 1.50
def pretzels_sold_dozen : ℕ := 3
def price_per_pretzel : ℝ := 2.00
def total_revenue : ℝ := 212.00

theorem chocolate_truffles_sold (dozens_of_truffles_sold : ℕ) :
  let fudge_revenue := (fudge_sold_pounds : ℝ) * price_per_pound_fudge
  let pretzels_revenue := (pretzels_sold_dozen : ℝ) * 12 * price_per_pretzel
  let truffles_revenue := total_revenue - fudge_revenue - pretzels_revenue
  let num_truffles_sold := truffles_revenue / price_per_truffle
  let dozens_of_truffles_sold := num_truffles_sold / 12
  dozens_of_truffles_sold = 5 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_truffles_sold_l1586_158609


namespace NUMINAMATH_GPT_line_param_func_l1586_158651

theorem line_param_func (t : ℝ) : 
    ∃ f : ℝ → ℝ, (∀ t, (20 * t - 14) = 2 * (f t) - 30) ∧ (f t = 10 * t + 8) := by
  sorry

end NUMINAMATH_GPT_line_param_func_l1586_158651


namespace NUMINAMATH_GPT_how_long_to_grow_more_l1586_158643

def current_length : ℕ := 14
def length_to_donate : ℕ := 23
def desired_length_after_donation : ℕ := 12

theorem how_long_to_grow_more : 
  (desired_length_after_donation + length_to_donate - current_length) = 21 := 
by
  -- Leave the proof part for later
  sorry

end NUMINAMATH_GPT_how_long_to_grow_more_l1586_158643


namespace NUMINAMATH_GPT_length_segment_MN_l1586_158641

open Real

noncomputable def line (x : ℝ) : ℝ := x + 2

def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem length_segment_MN :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
    on_circle x₁ y₁ →
    on_circle x₂ y₂ →
    (line x₁ = y₁ ∧ line x₂ = y₂) →
    dist (x₁, y₁) (x₂, y₂) = 2 * sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_length_segment_MN_l1586_158641


namespace NUMINAMATH_GPT_monomial_2023rd_l1586_158637

theorem monomial_2023rd : ∀ (x : ℝ), (2 * 2023 + 1) / 2023 * x ^ 2023 = (4047 / 2023) * x ^ 2023 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_monomial_2023rd_l1586_158637


namespace NUMINAMATH_GPT_arithmetic_sequence_terms_l1586_158612

theorem arithmetic_sequence_terms
  (a : ℕ → ℝ)
  (n : ℕ)
  (h1 : a 1 + a 2 + a 3 = 20)
  (h2 : a (n-2) + a (n-1) + a n = 130)
  (h3 : (n * (a 1 + a n)) / 2 = 200) :
  n = 8 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_terms_l1586_158612


namespace NUMINAMATH_GPT_trigonometric_identity_l1586_158642

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 2) : 
  (Real.cos x + Real.sin x) / (3 * Real.cos x - Real.sin x) = 3 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1586_158642


namespace NUMINAMATH_GPT_time_difference_alice_bob_l1586_158695

theorem time_difference_alice_bob
  (alice_speed : ℕ) (bob_speed : ℕ) (distance : ℕ)
  (h_alice_speed : alice_speed = 7)
  (h_bob_speed : bob_speed = 9)
  (h_distance : distance = 12) :
  (bob_speed * distance - alice_speed * distance) = 24 :=
by
  sorry

end NUMINAMATH_GPT_time_difference_alice_bob_l1586_158695


namespace NUMINAMATH_GPT_distance_between_starting_points_l1586_158688

theorem distance_between_starting_points :
  let speed1 := 70
  let speed2 := 80
  let start_time := 10 -- in hours (10 am)
  let meet_time := 14 -- in hours (2 pm)
  let travel_time := meet_time - start_time
  let distance1 := speed1 * travel_time
  let distance2 := speed2 * travel_time
  distance1 + distance2 = 600 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_starting_points_l1586_158688


namespace NUMINAMATH_GPT_exercise_l1586_158645

-- Define the given expression.
def f (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 5

-- Define the general form expression.
def g (x h k : ℝ) (a : ℝ) := a * (x - h)^2 + k

-- Prove that a + h + k = 6 when expressing f(x) in the form a(x-h)^2 + k.
theorem exercise : ∃ a h k : ℝ, (∀ x : ℝ, f x = g x h k a) ∧ (a + h + k = 6) :=
by
  sorry

end NUMINAMATH_GPT_exercise_l1586_158645


namespace NUMINAMATH_GPT_tysons_speed_in_ocean_l1586_158625

theorem tysons_speed_in_ocean
  (speed_lake : ℕ) (half_races_lake : ℕ) (total_races : ℕ) (race_distance : ℕ) (total_time : ℕ)
  (speed_lake_val : speed_lake = 3)
  (half_races_lake_val : half_races_lake = 5)
  (total_races_val : total_races = 10)
  (race_distance_val : race_distance = 3)
  (total_time_val : total_time = 11) :
  ∃ (speed_ocean : ℚ), speed_ocean = 2.5 := 
by
  sorry

end NUMINAMATH_GPT_tysons_speed_in_ocean_l1586_158625


namespace NUMINAMATH_GPT_point_A_symmetric_to_B_about_l_l1586_158655

variables {A B : ℝ × ℝ} {l : ℝ → ℝ → Prop}

-- define point B
def point_B := (1, 2)

-- define the line equation x + y + 3 = 0 as a property
def line_l (x y : ℝ) := x + y + 3 = 0

-- define that A is symmetric to B about line l
def symmetric_about (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) :=
  (∀ x y : ℝ, l x y → ((A.1 + B.1) / 2 + (A.2 + B.2) / 2 = -(x + y)))
  ∧ ((A.2 - B.2) / (A.1 - B.1) * -1 = -1)

theorem point_A_symmetric_to_B_about_l :
  A = (-5, -4) →
  symmetric_about A B line_l →
  A = (-5, -4) := by
  intros _ sym
  sorry

end NUMINAMATH_GPT_point_A_symmetric_to_B_about_l_l1586_158655


namespace NUMINAMATH_GPT_range_of_f_4_l1586_158611

theorem range_of_f_4 {a b c d : ℝ} 
  (h1 : 1 ≤ a*(-1)^3 + b*(-1)^2 + c*(-1) + d ∧ a*(-1)^3 + b*(-1)^2 + c*(-1) + d ≤ 2) 
  (h2 : 1 ≤ a*1^3 + b*1^2 + c*1 + d ∧ a*1^3 + b*1^2 + c*1 + d ≤ 3) 
  (h3 : 2 ≤ a*2^3 + b*2^2 + c*2 + d ∧ a*2^3 + b*2^2 + c*2 + d ≤ 4) 
  (h4 : -1 ≤ a*3^3 + b*3^2 + c*3 + d ∧ a*3^3 + b*3^2 + c*3 + d ≤ 1) :
  -21.75 ≤ a*4^3 + b*4^2 + c*4 + d ∧ a*4^3 + b*4^2 + c*4 + d ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_f_4_l1586_158611


namespace NUMINAMATH_GPT_symmetric_about_line_periodic_function_l1586_158613

section
variable {α : Type*} [LinearOrderedField α]

-- First proof problem
theorem symmetric_about_line (f : α → α) (a : α) (h : ∀ x, f (a + x) = f (a - x)) : 
  ∀ x, f (2 * a - x) = f x :=
sorry

-- Second proof problem
theorem periodic_function (f : α → α) (a b : α) (ha : a ≠ b)
  (hsymm_a : ∀ x, f (2 * a - x) = f x)
  (hsymm_b : ∀ x, f (2 * b - x) = f x) : 
  ∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x :=
sorry
end

end NUMINAMATH_GPT_symmetric_about_line_periodic_function_l1586_158613


namespace NUMINAMATH_GPT_find_clubs_l1586_158632

theorem find_clubs (S D H C : ℕ) (h1 : S + D + H + C = 13)
  (h2 : S + C = 7) 
  (h3 : D + H = 6) 
  (h4 : D = 2 * S) 
  (h5 : H = 2 * D) 
  : C = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_clubs_l1586_158632


namespace NUMINAMATH_GPT_escher_prints_probability_l1586_158699

theorem escher_prints_probability :
  let total_pieces := 12
  let escher_pieces := 4
  let total_permutations := Nat.factorial total_pieces
  let block_permutations := 9 * Nat.factorial (total_pieces - escher_pieces)
  let probability := block_permutations / (total_pieces * Nat.factorial (total_pieces - 1))
  probability = 1 / 1320 :=
by
  let total_pieces := 12
  let escher_pieces := 4
  let total_permutations := Nat.factorial total_pieces
  let block_permutations := 9 * Nat.factorial (total_pieces - escher_pieces)
  let probability := block_permutations / (total_pieces * Nat.factorial (total_pieces - 1))
  sorry

end NUMINAMATH_GPT_escher_prints_probability_l1586_158699


namespace NUMINAMATH_GPT_evaluate_g_l1586_158671

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_g : 3 * g 2 + 2 * g (-2) = 98 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_g_l1586_158671


namespace NUMINAMATH_GPT_smallest_whole_number_l1586_158639

theorem smallest_whole_number :
  ∃ a : ℕ, a % 3 = 2 ∧ a % 5 = 3 ∧ a % 7 = 3 ∧ ∀ b : ℕ, (b % 3 = 2 ∧ b % 5 = 3 ∧ b % 7 = 3 → a ≤ b) :=
sorry

end NUMINAMATH_GPT_smallest_whole_number_l1586_158639


namespace NUMINAMATH_GPT_tokens_per_pitch_l1586_158696

theorem tokens_per_pitch 
  (tokens_macy : ℕ) (tokens_piper : ℕ)
  (hits_macy : ℕ) (hits_piper : ℕ)
  (misses_total : ℕ) (p : ℕ)
  (h1 : tokens_macy = 11)
  (h2 : tokens_piper = 17)
  (h3 : hits_macy = 50)
  (h4 : hits_piper = 55)
  (h5 : misses_total = 315)
  (h6 : 28 * p = hits_macy + hits_piper + misses_total) :
  p = 15 := 
by 
  sorry

end NUMINAMATH_GPT_tokens_per_pitch_l1586_158696


namespace NUMINAMATH_GPT_savings_percentage_l1586_158662

variable (I : ℝ) -- First year's income
variable (S : ℝ) -- Amount saved in the first year

-- Conditions
axiom condition1 (h1 : S = 0.05 * I) : Prop
axiom condition2 (h2 : S + 0.05 * I = 2 * S) : Prop
axiom condition3 (h3 : (I - S) + 1.10 * (I - S) = 2 * (I - S)) : Prop

-- Theorem that proves the man saved 5% of his income in the first year
theorem savings_percentage : S = 0.05 * I :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_savings_percentage_l1586_158662


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1586_158650

theorem sufficient_not_necessary (x : ℝ) : (x > 3) → (abs (x - 3) > 0) ∧ (¬(abs (x - 3) > 0) → (¬(x > 3))) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1586_158650


namespace NUMINAMATH_GPT_cistern_fill_time_l1586_158664

variable (A_rate : ℚ) (B_rate : ℚ) (C_rate : ℚ)
variable (total_rate : ℚ := A_rate + C_rate - B_rate)

theorem cistern_fill_time (hA : A_rate = 1/7) (hB : B_rate = 1/9) (hC : C_rate = 1/12) :
  (1/total_rate) = 252/29 :=
by
  rw [hA, hB, hC]
  sorry

end NUMINAMATH_GPT_cistern_fill_time_l1586_158664


namespace NUMINAMATH_GPT_julian_comic_book_l1586_158610

theorem julian_comic_book : 
  ∀ (total_frames frames_per_page : ℕ),
    total_frames = 143 →
    frames_per_page = 11 →
    total_frames / frames_per_page = 13 ∧ total_frames % frames_per_page = 0 :=
by
  intros total_frames frames_per_page
  intros h_total_frames h_frames_per_page
  sorry

end NUMINAMATH_GPT_julian_comic_book_l1586_158610


namespace NUMINAMATH_GPT_projection_of_a_onto_b_l1586_158675

open Real

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-2, 4)

theorem projection_of_a_onto_b :
  let dot_product := vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2
  let magnitude_b_squared := vector_b.1 ^ 2 + vector_b.2 ^ 2
  let scalar_projection := dot_product / magnitude_b_squared
  let proj_vector := (scalar_projection * vector_b.1, scalar_projection * vector_b.2)
  proj_vector = (-4/5, 8/5) :=
by
  sorry

end NUMINAMATH_GPT_projection_of_a_onto_b_l1586_158675


namespace NUMINAMATH_GPT_charlie_cortland_apples_l1586_158683

/-- Given that Charlie picked 0.17 bags of Golden Delicious apples, 0.17 bags of Macintosh apples, 
   and a total of 0.67 bags of fruit, prove that the number of bags of Cortland apples picked by Charlie is 0.33. -/
theorem charlie_cortland_apples :
  let golden_delicious := 0.17
  let macintosh := 0.17
  let total_fruit := 0.67
  total_fruit - (golden_delicious + macintosh) = 0.33 :=
by
  sorry

end NUMINAMATH_GPT_charlie_cortland_apples_l1586_158683


namespace NUMINAMATH_GPT_division_dividend_l1586_158604

/-- In a division sum, the quotient is 40, the divisor is 72, and the remainder is 64. We need to prove that the dividend is 2944. -/
theorem division_dividend : 
  let Q := 40
  let D := 72
  let R := 64
  (D * Q + R = 2944) :=
by
  sorry

end NUMINAMATH_GPT_division_dividend_l1586_158604


namespace NUMINAMATH_GPT_vertices_of_regular_hexagonal_pyramid_l1586_158646

-- Define a structure for a regular hexagonal pyramid
structure RegularHexagonalPyramid where
  baseVertices : Nat
  apexVertices : Nat

-- Define a specific regular hexagonal pyramid with given conditions
def regularHexagonalPyramid : RegularHexagonalPyramid :=
  { baseVertices := 6, apexVertices := 1 }

-- The theorem stating the number of vertices of the pyramid
theorem vertices_of_regular_hexagonal_pyramid : regularHexagonalPyramid.baseVertices + regularHexagonalPyramid.apexVertices = 7 := 
  by
  sorry

end NUMINAMATH_GPT_vertices_of_regular_hexagonal_pyramid_l1586_158646


namespace NUMINAMATH_GPT_no_prime_divisible_by_56_l1586_158668

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def divisible_by_56 (n : ℕ) : Prop := 56 ∣ n

theorem no_prime_divisible_by_56 : ¬ ∃ p, is_prime p ∧ divisible_by_56 p := 
  sorry

end NUMINAMATH_GPT_no_prime_divisible_by_56_l1586_158668


namespace NUMINAMATH_GPT_result_l1586_158623

noncomputable def Kolya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  P_A = (1: ℝ) / 2

def Valya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  let P_B := (r * p) / (1 - s * q)
  let P_AorB := P_A + P_B
  P_AorB > (1: ℝ) / 2

theorem result (p r : ℝ) (h1 : Kolya_right p r) (h2 : ¬Valya_right p r) :
  Kolya_right p r ∧ ¬Valya_right p r :=
  ⟨h1, h2⟩

end NUMINAMATH_GPT_result_l1586_158623


namespace NUMINAMATH_GPT_max_digit_sum_l1586_158629

-- Define the condition for the hours and minutes digits
def is_valid_hour (h : ℕ) := 0 ≤ h ∧ h < 24
def is_valid_minute (m : ℕ) := 0 ≤ m ∧ m < 60

-- Define the function to calculate the sum of the digits of a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Main statement: Prove that the maximum sum of the digits in the display is 24
theorem max_digit_sum : ∃ h m: ℕ, is_valid_hour h ∧ is_valid_minute m ∧ 
  sum_of_digits h + sum_of_digits m = 24 :=
sorry

end NUMINAMATH_GPT_max_digit_sum_l1586_158629


namespace NUMINAMATH_GPT_power_expression_simplify_l1586_158618

theorem power_expression_simplify :
  (1 / (-5^2)^3) * (-5)^8 * Real.sqrt 5 = 5^(5/2) :=
by
  sorry

end NUMINAMATH_GPT_power_expression_simplify_l1586_158618


namespace NUMINAMATH_GPT_one_thirds_of_nine_halfs_l1586_158626

theorem one_thirds_of_nine_halfs : (9 / 2) / (1 / 3) = 27 / 2 := 
by sorry

end NUMINAMATH_GPT_one_thirds_of_nine_halfs_l1586_158626


namespace NUMINAMATH_GPT_LemonadeCalories_l1586_158692

noncomputable def total_calories (lemon_juice sugar water honey : ℕ) (cal_per_100g_lemon_juice cal_per_100g_sugar cal_per_100g_honey : ℕ) : ℝ :=
  (lemon_juice / 100) * cal_per_100g_lemon_juice +
  (sugar / 100) * cal_per_100g_sugar +
  (honey / 100) * cal_per_100g_honey

noncomputable def calories_in_250g (total_calories : ℝ) (total_weight : ℕ) : ℝ :=
  (total_calories / total_weight) * 250

theorem LemonadeCalories :
  let lemon_juice := 150
  let sugar := 200
  let water := 300
  let honey := 50
  let cal_per_100g_lemon_juice := 25
  let cal_per_100g_sugar := 386
  let cal_per_100g_honey := 64
  let total_weight := lemon_juice + sugar + water + honey
  let total_cal := total_calories lemon_juice sugar water honey cal_per_100g_lemon_juice cal_per_100g_sugar cal_per_100g_honey
  calories_in_250g total_cal total_weight = 301 :=
by
  sorry

end NUMINAMATH_GPT_LemonadeCalories_l1586_158692


namespace NUMINAMATH_GPT_joy_tape_deficit_l1586_158649

noncomputable def tape_needed_field (width length : ℕ) : ℕ :=
2 * (length + width)

noncomputable def tape_needed_trees (num_trees circumference : ℕ) : ℕ :=
num_trees * circumference

def tape_total_needed (tape_field tape_trees : ℕ) : ℕ :=
tape_field + tape_trees

theorem joy_tape_deficit (tape_has : ℕ) (tape_field tape_trees: ℕ) : ℤ :=
tape_has - (tape_field + tape_trees)

example : joy_tape_deficit 180 (tape_needed_field 35 80) (tape_needed_trees 3 5) = -65 := by
sorry

end NUMINAMATH_GPT_joy_tape_deficit_l1586_158649


namespace NUMINAMATH_GPT_value_of_x_minus_y_l1586_158619

theorem value_of_x_minus_y (x y : ℝ) 
  (h1 : |x| = 2) 
  (h2 : y^2 = 9) 
  (h3 : x + y < 0) : 
  x - y = 1 ∨ x - y = 5 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_x_minus_y_l1586_158619


namespace NUMINAMATH_GPT_pencils_left_l1586_158673

def total_pencils (boxes : ℕ) (pencils_per_box : ℕ) : ℕ :=
  boxes * pencils_per_box

def remaining_pencils (initial_pencils : ℕ) (pencils_given : ℕ) : ℕ :=
  initial_pencils - pencils_given

theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given : ℕ)
  (h_boxes : boxes = 2) (h_pencils_per_box : pencils_per_box = 14) (h_pencils_given : pencils_given = 6) :
  remaining_pencils (total_pencils boxes pencils_per_box) pencils_given = 22 :=
by
  rw [h_boxes, h_pencils_per_box, h_pencils_given]
  norm_num
  sorry

end NUMINAMATH_GPT_pencils_left_l1586_158673


namespace NUMINAMATH_GPT_equation_of_perpendicular_line_l1586_158659

theorem equation_of_perpendicular_line (x y : ℝ) (l1 : 2*x - 3*y + 4 = 0) (pt : x = -2 ∧ y = -3) :
  3*(-2) + 2*(-3) + 12 = 0 := by
  sorry

end NUMINAMATH_GPT_equation_of_perpendicular_line_l1586_158659


namespace NUMINAMATH_GPT_solve_apples_problem_l1586_158682

def apples_problem (marin_apples donald_apples total_apples : ℕ) : Prop :=
  marin_apples = 9 ∧ total_apples = 11 → donald_apples = 2

theorem solve_apples_problem : apples_problem 9 2 11 := by
  sorry

end NUMINAMATH_GPT_solve_apples_problem_l1586_158682


namespace NUMINAMATH_GPT_complete_the_square_b_26_l1586_158661

theorem complete_the_square_b_26 :
  ∃ (a b : ℝ), (∀ x : ℝ, x^2 + 10 * x - 1 = 0 ↔ (x + a)^2 = b) ∧ b = 26 :=
sorry

end NUMINAMATH_GPT_complete_the_square_b_26_l1586_158661


namespace NUMINAMATH_GPT_max_sin_a_l1586_158603

theorem max_sin_a (a b c : ℝ) (h1 : Real.cos a = Real.tan b) 
                                  (h2 : Real.cos b = Real.tan c) 
                                  (h3 : Real.cos c = Real.tan a) : 
  Real.sin a ≤ Real.sqrt ((3 - Real.sqrt 5) / 2) := 
by
  sorry

end NUMINAMATH_GPT_max_sin_a_l1586_158603


namespace NUMINAMATH_GPT_discount_percentage_l1586_158679

theorem discount_percentage (number_of_tshirts : ℕ) (cost_per_tshirt amount_paid : ℝ)
  (h1 : number_of_tshirts = 6)
  (h2 : cost_per_tshirt = 20)
  (h3 : amount_paid = 60) : 
  ((number_of_tshirts * cost_per_tshirt - amount_paid) / (number_of_tshirts * cost_per_tshirt) * 100) = 50 := by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_discount_percentage_l1586_158679


namespace NUMINAMATH_GPT_measure_angle_E_l1586_158681

-- Definitions based on conditions
variables {p q : Type} {A B E : ℝ}

noncomputable def measure_A (A B : ℝ) : ℝ := A
noncomputable def measure_B (A B : ℝ) : ℝ := 9 * A
noncomputable def parallel_lines (p q : Type) : Prop := true

-- Condition: measure of angle A is 1/9 of the measure of angle B
axiom angle_condition : A = (1 / 9) * B

-- Condition: p is parallel to q
axiom parallel_condition : parallel_lines p q

-- Prove that the measure of angle E is 18 degrees
theorem measure_angle_E (y : ℝ) (h1 : A = y) (h2 : B = 9 * y) : E = 18 :=
by
  sorry

end NUMINAMATH_GPT_measure_angle_E_l1586_158681


namespace NUMINAMATH_GPT_double_square_area_l1586_158602

theorem double_square_area (a k : ℝ) (h : (k * a) ^ 2 = 2 * a ^ 2) : k = Real.sqrt 2 := 
by 
  -- Our goal is to prove that k = sqrt(2)
  sorry

end NUMINAMATH_GPT_double_square_area_l1586_158602


namespace NUMINAMATH_GPT_sin_negative_300_eq_l1586_158628

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

end NUMINAMATH_GPT_sin_negative_300_eq_l1586_158628


namespace NUMINAMATH_GPT_triangle_probability_is_correct_l1586_158601

-- Define the total number of figures
def total_figures : ℕ := 8

-- Define the number of triangles among the figures
def number_of_triangles : ℕ := 3

-- Define the probability function for choosing a triangle
def probability_of_triangle : ℚ := number_of_triangles / total_figures

-- The theorem to be proved
theorem triangle_probability_is_correct :
  probability_of_triangle = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_triangle_probability_is_correct_l1586_158601


namespace NUMINAMATH_GPT_probability_within_sphere_correct_l1586_158690

noncomputable def probability_within_sphere : ℝ :=
  let cube_volume := (2 : ℝ) * (2 : ℝ) * (2 : ℝ)
  let sphere_volume := (4 * Real.pi / 3) * (0.5) ^ 3
  sphere_volume / cube_volume

theorem probability_within_sphere_correct (x y z : ℝ) 
  (hx1 : -1 ≤ x) (hx2 : x ≤ 1) 
  (hy1 : -1 ≤ y) (hy2 : y ≤ 1) 
  (hz1 : -1 ≤ z) (hz2 : z ≤ 1) 
  (hx_sq : x^2 ≤ 0.5) 
  (hxyz : x^2 + y^2 + z^2 ≤ 0.25) : 
  probability_within_sphere = Real.pi / 48 :=
by
  sorry

end NUMINAMATH_GPT_probability_within_sphere_correct_l1586_158690


namespace NUMINAMATH_GPT_ellipse_foci_coordinates_l1586_158600

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ), (y^2 / 3 + x^2 / 2 = 1) → (x, y) = (0, -1) ∨ (x, y) = (0, 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_coordinates_l1586_158600


namespace NUMINAMATH_GPT_sequence_formula_l1586_158680

theorem sequence_formula (S : ℕ → ℕ) (a : ℕ → ℕ) (h₁ : ∀ n, S n = n^2 + 1) :
  (a 1 = 2) ∧ (∀ n ≥ 2, a n = 2 * n - 1) := 
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l1586_158680


namespace NUMINAMATH_GPT_total_red_marbles_l1586_158684

-- Definitions derived from the problem conditions
def Jessica_red_marbles : ℕ := 3 * 12
def Sandy_red_marbles : ℕ := 4 * Jessica_red_marbles
def Alex_red_marbles : ℕ := Jessica_red_marbles + 2 * 12

-- Statement we need to prove that total number of marbles is 240
theorem total_red_marbles : 
  Jessica_red_marbles + Sandy_red_marbles + Alex_red_marbles = 240 := by
  -- We provide the proof later
  sorry

end NUMINAMATH_GPT_total_red_marbles_l1586_158684


namespace NUMINAMATH_GPT_salary_proof_l1586_158694

-- Defining the monthly salaries of the officials
def D_Dupon : ℕ := 6000
def D_Duran : ℕ := 8000
def D_Marten : ℕ := 5000

-- Defining the statements made by each official
def Dupon_statement1 : Prop := D_Dupon = 6000
def Dupon_statement2 : Prop := D_Duran = D_Dupon + 2000
def Dupon_statement3 : Prop := D_Marten = D_Dupon - 1000

def Duran_statement1 : Prop := D_Duran > D_Marten
def Duran_statement2 : Prop := D_Duran - D_Marten = 3000
def Duran_statement3 : Prop := D_Marten = 9000

def Marten_statement1 : Prop := D_Marten < D_Dupon
def Marten_statement2 : Prop := D_Dupon = 7000
def Marten_statement3 : Prop := D_Duran = D_Dupon + 3000

-- Defining the constraints about the number of truth and lies
def Told_the_truth_twice_and_lied_once : Prop :=
  (Dupon_statement1 ∧ Dupon_statement2 ∧ ¬Dupon_statement3) ∨
  (Dupon_statement1 ∧ ¬Dupon_statement2 ∧ Dupon_statement3) ∨
  (¬Dupon_statement1 ∧ Dupon_statement2 ∧ Dupon_statement3) ∨
  (Duran_statement1 ∧ Duran_statement2 ∧ ¬Duran_statement3) ∨
  (Duran_statement1 ∧ ¬Duran_statement2 ∧ Duran_statement3) ∨
  (¬Duran_statement1 ∧ Duran_statement2 ∧ Duran_statement3) ∨
  (Marten_statement1 ∧ Marten_statement2 ∧ ¬Marten_statement3) ∨
  (Marten_statement1 ∧ ¬Marten_statement2 ∧ Marten_statement3) ∨
  (¬Marten_statement1 ∧ Marten_statement2 ∧ Marten_statement3)

-- The final proof goal
theorem salary_proof : Told_the_truth_twice_and_lied_once →
  D_Dupon = 6000 ∧ D_Duran = 8000 ∧ D_Marten = 5000 := by 
  sorry

end NUMINAMATH_GPT_salary_proof_l1586_158694


namespace NUMINAMATH_GPT_germination_relative_frequency_l1586_158658

theorem germination_relative_frequency {n m : ℕ} (h₁ : n = 1000) (h₂ : m = 1000 - 90) : 
  (m : ℝ) / (n : ℝ) = 0.91 := by
  sorry

end NUMINAMATH_GPT_germination_relative_frequency_l1586_158658


namespace NUMINAMATH_GPT_monotonically_increasing_range_of_a_l1586_158693

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4 * x - 5)

theorem monotonically_increasing_range_of_a :
  ∀ (a : ℝ), (∀ x : ℝ, x > a → f x > f a) ↔ a ≥ 5 :=
by
  intro a
  unfold f
  sorry

end NUMINAMATH_GPT_monotonically_increasing_range_of_a_l1586_158693


namespace NUMINAMATH_GPT_smallest_number_of_lawyers_l1586_158620

/-- Given that:
- n is the number of delegates, where 220 < n < 254
- m is the number of economists, so the number of lawyers is n - m
- Each participant played with each other participant exactly once.
- A match winner got one point, the loser got none, and in case of a draw, both participants received half a point each.
- By the end of the tournament, each participant gained half of all their points from matches against economists.

Prove that the smallest number of lawyers participating in the tournament is 105. -/
theorem smallest_number_of_lawyers (n m : ℕ) (h1 : 220 < n) (h2 : n < 254)
  (h3 : m * (m - 1) + (n - m) * (n - m - 1) = n * (n - 1))
  (h4 : m * (m - 1) = 2 * (n * (n - 1)) / 4) :
  n - m = 105 :=
sorry

end NUMINAMATH_GPT_smallest_number_of_lawyers_l1586_158620


namespace NUMINAMATH_GPT_double_acute_angle_is_less_than_180_degrees_l1586_158663

theorem double_acute_angle_is_less_than_180_degrees (alpha : ℝ) (h : 0 < alpha ∧ alpha < 90) : 2 * alpha < 180 :=
sorry

end NUMINAMATH_GPT_double_acute_angle_is_less_than_180_degrees_l1586_158663


namespace NUMINAMATH_GPT_sqrt_12_bounds_l1586_158674

theorem sqrt_12_bounds : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_12_bounds_l1586_158674


namespace NUMINAMATH_GPT_intercept_condition_slope_condition_l1586_158631

theorem intercept_condition (m : ℚ) (h : m ≠ -1) : 
  (m^2 - 2 * m - 3) * -3 + (2 * m^2 + m - 1) * 0 + (-2 * m + 6) = 0 → 
  m = -5 / 3 := 
  sorry

theorem slope_condition (m : ℚ) (h : m ≠ -1) : 
  (m^2 + 2 * m - 4) = 0 → 
  m = 4 / 3 := 
  sorry

end NUMINAMATH_GPT_intercept_condition_slope_condition_l1586_158631


namespace NUMINAMATH_GPT_reflection_twice_is_identity_l1586_158689

-- Define the reflection matrix R over the vector (1, 2)
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  -- Note: The specific definition of the reflection matrix over (1, 2) is skipped as we only need the final proof statement.
  sorry

-- Assign the reflection matrix R to variable R
def R := reflection_matrix

-- Prove that R^2 = I
theorem reflection_twice_is_identity : R * R = 1 := by
  sorry

end NUMINAMATH_GPT_reflection_twice_is_identity_l1586_158689


namespace NUMINAMATH_GPT_chairs_built_in_10_days_l1586_158647

-- Define the conditions as variables
def hours_per_day : ℕ := 8
def days_worked : ℕ := 10
def hours_per_chair : ℕ := 5

-- State the problem as a conjecture or theorem
theorem chairs_built_in_10_days : (hours_per_day * days_worked) / hours_per_chair = 16 := by
    sorry

end NUMINAMATH_GPT_chairs_built_in_10_days_l1586_158647


namespace NUMINAMATH_GPT_greatest_perimeter_l1586_158615

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end NUMINAMATH_GPT_greatest_perimeter_l1586_158615


namespace NUMINAMATH_GPT_solution_set_abs_inequality_l1586_158633

theorem solution_set_abs_inequality : {x : ℝ | |1 - 2 * x| < 3} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_solution_set_abs_inequality_l1586_158633


namespace NUMINAMATH_GPT_total_books_l1586_158634

/-- Define Tim’s and Sam’s number of books. -/
def TimBooks : ℕ := 44
def SamBooks : ℕ := 52

/-- Prove that together they have 96 books. -/
theorem total_books : TimBooks + SamBooks = 96 := by
  sorry

end NUMINAMATH_GPT_total_books_l1586_158634


namespace NUMINAMATH_GPT_collin_savings_l1586_158640

theorem collin_savings :
  let cans_home := 12
  let cans_grandparents := 3 * cans_home
  let cans_neighbor := 46
  let cans_dad := 250
  let total_cans := cans_home + cans_grandparents + cans_neighbor + cans_dad
  let money_per_can := 0.25
  let total_money := total_cans * money_per_can
  let savings := total_money / 2
  savings = 43 := 
  by 
  sorry

end NUMINAMATH_GPT_collin_savings_l1586_158640


namespace NUMINAMATH_GPT_necessary_sufficient_condition_l1586_158660

theorem necessary_sufficient_condition (A B C : ℝ)
    (h : ∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) :
    |A - B + C| ≤ 2 * Real.sqrt (A * C) := 
by sorry

end NUMINAMATH_GPT_necessary_sufficient_condition_l1586_158660


namespace NUMINAMATH_GPT_grade_students_difference_condition_l1586_158653

variables (G1 G2 G5 : ℕ)

theorem grade_students_difference_condition (h : G1 + G2 = G2 + G5 + 30) : G1 - G5 = 30 :=
sorry

end NUMINAMATH_GPT_grade_students_difference_condition_l1586_158653


namespace NUMINAMATH_GPT_min_sum_x1_x2_x3_x4_l1586_158614

variables (x1 x2 x3 x4 : ℝ)

theorem min_sum_x1_x2_x3_x4 : 
  (x1 + x2 ≥ 12) → 
  (x1 + x3 ≥ 13) → 
  (x1 + x4 ≥ 14) → 
  (x3 + x4 ≥ 22) → 
  (x2 + x3 ≥ 23) → 
  (x2 + x4 ≥ 24) → 
  (x1 + x2 + x3 + x4 = 37) := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_min_sum_x1_x2_x3_x4_l1586_158614


namespace NUMINAMATH_GPT_tangent_line_is_x_minus_y_eq_zero_l1586_158607

theorem tangent_line_is_x_minus_y_eq_zero : 
  ∀ (f : ℝ → ℝ) (x y : ℝ), 
  f x = x^3 - 2 * x → 
  (x, y) = (1, 1) → 
  (∃ (m : ℝ), m = 3 * (1:ℝ)^2 - 2 ∧ (y - 1) = m * (x - 1)) → 
  x - y = 0 :=
by
  intros f x y h_func h_point h_tangent
  sorry

end NUMINAMATH_GPT_tangent_line_is_x_minus_y_eq_zero_l1586_158607


namespace NUMINAMATH_GPT_evaluate_f_at_2_l1586_158697

def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem evaluate_f_at_2 :
  f 2 = -2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_2_l1586_158697


namespace NUMINAMATH_GPT_ribbon_per_box_l1586_158666

theorem ribbon_per_box (total_ribbon : ℚ) (total_boxes : ℕ) (same_ribbon_each_box : ℚ) 
  (h1 : total_ribbon = 5/12) (h2 : total_boxes = 5) : 
  same_ribbon_each_box = 1/12 :=
sorry

end NUMINAMATH_GPT_ribbon_per_box_l1586_158666


namespace NUMINAMATH_GPT_james_car_purchase_l1586_158672

/-- 
James sold his $20,000 car for 80% of its value, 
then bought a $30,000 sticker price car, 
and he was out of pocket $11,000. 
James bought the new car for 90% of its value. 
-/
theorem james_car_purchase (V_1 P_1 V_2 O P : ℝ)
  (hV1 : V_1 = 20000)
  (hP1 : P_1 = 80)
  (hV2 : V_2 = 30000)
  (hO : O = 11000)
  (hSaleOld : (P_1 / 100) * V_1 = 16000)
  (hDiff : 16000 + O = 27000)
  (hPurchase : (P / 100) * V_2 = 27000) :
  P = 90 := 
sorry

end NUMINAMATH_GPT_james_car_purchase_l1586_158672


namespace NUMINAMATH_GPT_square_area_4900_l1586_158638

/-- If one side of a square is increased by 3.5 times and the other side is decreased by 30 cm, resulting in a rectangle that has twice the area of the square, then the area of the square is 4900 square centimeters. -/
theorem square_area_4900 (x : ℝ) (h1 : 3.5 * x * (x - 30) = 2 * x^2) : x^2 = 4900 :=
sorry

end NUMINAMATH_GPT_square_area_4900_l1586_158638


namespace NUMINAMATH_GPT_min_sum_product_l1586_158654

theorem min_sum_product (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 1/m + 9/n = 1) :
  m * n = 48 :=
sorry

end NUMINAMATH_GPT_min_sum_product_l1586_158654


namespace NUMINAMATH_GPT_number_mod_conditions_l1586_158616

theorem number_mod_conditions :
  ∃ N, (N % 10 = 9) ∧ (N % 9 = 8) ∧ (N % 8 = 7) ∧ (N % 7 = 6) ∧
       (N % 6 = 5) ∧ (N % 5 = 4) ∧ (N % 4 = 3) ∧ (N % 3 = 2) ∧ (N % 2 = 1) ∧
       N = 2519 :=
by
  sorry

end NUMINAMATH_GPT_number_mod_conditions_l1586_158616


namespace NUMINAMATH_GPT_value_of_w_over_y_l1586_158656

theorem value_of_w_over_y (w x y : ℝ) (h1 : w / x = 1 / 3) (h2 : (x + y) / y = 3.25) : w / y = 0.75 :=
sorry

end NUMINAMATH_GPT_value_of_w_over_y_l1586_158656


namespace NUMINAMATH_GPT_combined_area_l1586_158605

noncomputable def diagonal : ℝ := 12 * Real.sqrt 2

noncomputable def side_of_square (d : ℝ) : ℝ := d / Real.sqrt 2

noncomputable def area_of_square (s : ℝ) : ℝ := s ^ 2

noncomputable def radius_of_circle (d : ℝ) : ℝ := d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r ^ 2

theorem combined_area (d : ℝ) (h : d = diagonal) :
  let s := side_of_square d
  let area_sq := area_of_square s
  let r := radius_of_circle d
  let area_circ := area_of_circle r
  area_sq + area_circ = 144 + 72 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_combined_area_l1586_158605


namespace NUMINAMATH_GPT_kevin_speed_first_half_l1586_158669

-- Let's define the conditions as variables and constants
variable (total_distance : ℕ) (distance_20mph : ℕ) (distance_8mph : ℕ)
variable (time_20mph : ℝ) (time_8mph : ℝ) (distance_first_half : ℕ)
variable (speed_first_half : ℝ)

-- Conditions from the problem
def conditions (total_distance : ℕ) (distance_20mph : ℕ) (distance_8mph : ℕ) : Prop :=
  total_distance = 17 ∧ 
  distance_20mph = 20 * 1 / 2 ∧
  distance_8mph = 8 * 1 / 4

-- Proof objective based on conditions and correct answer
theorem kevin_speed_first_half (
  h : conditions total_distance distance_20mph distance_8mph
) : speed_first_half = 10 := by
  sorry

end NUMINAMATH_GPT_kevin_speed_first_half_l1586_158669


namespace NUMINAMATH_GPT_brad_read_more_books_l1586_158622

-- Definitions based on the given conditions
def books_william_read_last_month : ℕ := 6
def books_brad_read_last_month : ℕ := 3 * books_william_read_last_month
def books_brad_read_this_month : ℕ := 8
def books_william_read_this_month : ℕ := 2 * books_brad_read_this_month

-- Totals
def total_books_brad_read : ℕ := books_brad_read_last_month + books_brad_read_this_month
def total_books_william_read : ℕ := books_william_read_last_month + books_william_read_this_month

-- The statement to prove
theorem brad_read_more_books : total_books_brad_read = total_books_william_read + 4 := by
  sorry

end NUMINAMATH_GPT_brad_read_more_books_l1586_158622


namespace NUMINAMATH_GPT_gcd_f_l1586_158677

def f (x: ℤ) : ℤ := x^2 - x + 2023

theorem gcd_f (x y : ℤ) (hx : x = 105) (hy : y = 106) : Int.gcd (f x) (f y) = 7 := by
  sorry

end NUMINAMATH_GPT_gcd_f_l1586_158677


namespace NUMINAMATH_GPT_proposition_not_true_at_9_l1586_158685

variable {P : ℕ → Prop}

theorem proposition_not_true_at_9 (h : ∀ k : ℕ, k ≥ 1 → P k → P (k + 1)) (h10 : ¬P 10) : ¬P 9 :=
by
  sorry

end NUMINAMATH_GPT_proposition_not_true_at_9_l1586_158685


namespace NUMINAMATH_GPT_largest_possible_b_l1586_158648

theorem largest_possible_b (b : ℚ) (h : (3 * b + 7) * (b - 2) = 9 * b) : b ≤ 2 :=
sorry

end NUMINAMATH_GPT_largest_possible_b_l1586_158648


namespace NUMINAMATH_GPT_students_number_l1586_158627

theorem students_number (C P S : ℕ) : C = 315 ∧ 121 + C = P * S -> S = 4 := by
  sorry

end NUMINAMATH_GPT_students_number_l1586_158627


namespace NUMINAMATH_GPT_simple_interest_calculation_l1586_158636

theorem simple_interest_calculation (P R T : ℝ) (H₁ : P = 8925) (H₂ : R = 9) (H₃ : T = 5) : 
  P * R * T / 100 = 4016.25 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_calculation_l1586_158636


namespace NUMINAMATH_GPT_probability_of_four_card_success_l1586_158698

example (cards : Fin 4) (pins : Fin 4) {attempts : ℕ}
  (h1 : ∀ (c : Fin 4) (p : Fin 4), attempts ≤ 3)
  (h2 : ∀ (c : Fin 4), ∃ (p : Fin 4), p ≠ c ∧ attempts ≤ 3) :
  ∃ (three_cards : Fin 3), attempts ≤ 3 :=
sorry

noncomputable def probability_success :
  ℚ := 23 / 24

theorem probability_of_four_card_success :
  probability_success = 23 / 24 :=
sorry

end NUMINAMATH_GPT_probability_of_four_card_success_l1586_158698


namespace NUMINAMATH_GPT_monthly_rent_l1586_158606

-- Definition
def total_amount_saved := 2225
def extra_amount_needed := 775
def deposit := 500

-- Total amount required
def total_amount_required := total_amount_saved + extra_amount_needed
def total_rent_plus_deposit (R : ℝ) := 2 * R + deposit

-- The statement to prove
theorem monthly_rent (R : ℝ) : total_rent_plus_deposit R = total_amount_required → R = 1250 :=
by
  intros h
  exact sorry -- Proof is omitted.

end NUMINAMATH_GPT_monthly_rent_l1586_158606


namespace NUMINAMATH_GPT_range_of_m_minimum_value_l1586_158608

theorem range_of_m (m n : ℝ) (h : 2 * m - n = 3) (ineq : |m| + |n + 3| ≥ 9) : 
  m ≤ -3 ∨ m ≥ 3 := 
sorry

theorem minimum_value (m n : ℝ) (h : 2 * m - n = 3) : 
  ∃ c, c = 3 ∧ c = |(5 / 3) * m - (1 / 3) * n| + |(1 / 3) * m - (2 / 3) * n| := 
sorry

end NUMINAMATH_GPT_range_of_m_minimum_value_l1586_158608


namespace NUMINAMATH_GPT_arithmetic_sequence_m_value_l1586_158657

theorem arithmetic_sequence_m_value (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -2) (h2 : S m = 0) (h3 : S (m + 1) = 3) 
  (h_seq : ∀ n : ℕ, S n = (n + 1) / 2 * (2 * a₁ + n * d)) :
  m = 5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_m_value_l1586_158657


namespace NUMINAMATH_GPT_complex_square_sum_eq_zero_l1586_158665

theorem complex_square_sum_eq_zero (i : ℂ) (h : i^2 = -1) : (1 + i)^2 + (1 - i)^2 = 0 :=
sorry

end NUMINAMATH_GPT_complex_square_sum_eq_zero_l1586_158665


namespace NUMINAMATH_GPT_sandra_tickets_relation_l1586_158678

def volleyball_game : Prop :=
  ∃ (tickets_total tickets_left tickets_jude tickets_andrea tickets_sandra : ℕ),
    tickets_total = 100 ∧
    tickets_left = 40 ∧
    tickets_jude = 16 ∧
    tickets_andrea = 2 * tickets_jude ∧
    tickets_total - tickets_left = tickets_jude + tickets_andrea + tickets_sandra ∧
    tickets_sandra = tickets_jude - 4

theorem sandra_tickets_relation : volleyball_game :=
  sorry

end NUMINAMATH_GPT_sandra_tickets_relation_l1586_158678


namespace NUMINAMATH_GPT_n_four_minus_n_squared_l1586_158635

theorem n_four_minus_n_squared (n : ℤ) : 6 ∣ (n^4 - n^2) :=
by 
  sorry

end NUMINAMATH_GPT_n_four_minus_n_squared_l1586_158635


namespace NUMINAMATH_GPT_regular_polygons_cover_plane_l1586_158667

theorem regular_polygons_cover_plane (n : ℕ) (h_n_ge_3 : 3 ≤ n)
    (h_angle_eq : ∀ n, (180 * (1 - (2 / n)) : ℝ) = (internal_angle : ℝ))
    (h_summation_eq : ∃ k : ℕ, k * internal_angle = 360) :
    n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end NUMINAMATH_GPT_regular_polygons_cover_plane_l1586_158667


namespace NUMINAMATH_GPT_evaluate_expression_l1586_158617

def expression (x y : ℤ) : ℤ :=
  y * (y - 2 * x) ^ 2

theorem evaluate_expression : 
  expression 4 2 = 72 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1586_158617


namespace NUMINAMATH_GPT_tangents_form_rectangle_l1586_158676

-- Define the first ellipse
def ellipse1 (a b x y : ℝ) : Prop := x^2 / a^4 + y^2 / b^4 = 1

-- Define the second ellipse
def ellipse2 (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define conjugate diameters through lines
def conjugate_diameters (a b m : ℝ) : Prop := True -- (You might want to further define what conjugate diameters imply here)

-- Prove the main statement
theorem tangents_form_rectangle
  (a b m : ℝ)
  (x1 y1 x2 y2 k1 k2 : ℝ)
  (h1 : ellipse1 a b x1 y1)
  (h2 : ellipse1 a b x2 y2)
  (h3 : ellipse2 a b x1 y1)
  (h4 : ellipse2 a b x2 y2)
  (conj1 : conjugate_diameters a b m)
  (tangent_slope1 : k1 = -b^2 / a^2 * (1 / m))
  (conj2 : conjugate_diameters a b (-b^4/a^4 * 1/m))
  (tangent_slope2 : k2 = -b^4 / a^4 * (1 / (-b^4/a^4 * (1/m))))
: k1 * k2 = -1 :=
sorry

end NUMINAMATH_GPT_tangents_form_rectangle_l1586_158676


namespace NUMINAMATH_GPT_bobby_initial_candy_l1586_158621

theorem bobby_initial_candy (initial_candy : ℕ) (remaining_candy : ℕ) (extra_candy : ℕ) (total_eaten : ℕ)
  (h_candy_initial : initial_candy = 36)
  (h_candy_remaining : remaining_candy = 4)
  (h_candy_extra : extra_candy = 15)
  (h_candy_total_eaten : total_eaten = initial_candy - remaining_candy) :
  total_eaten - extra_candy = 17 :=
by
  sorry

end NUMINAMATH_GPT_bobby_initial_candy_l1586_158621


namespace NUMINAMATH_GPT_sum_of_geometric_numbers_l1586_158691

def is_geometric (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ∃ r : ℕ, r > 0 ∧ 
  (d2 = d1 * r) ∧ 
  (d3 = d2 * r) ∧ 
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

theorem sum_of_geometric_numbers : 
  (∃ smallest largest : ℕ,
    (smallest = 124) ∧ 
    (largest = 972) ∧ 
    is_geometric (smallest) ∧ 
    is_geometric (largest)
  ) →
  124 + 972 = 1096 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_geometric_numbers_l1586_158691


namespace NUMINAMATH_GPT_average_primes_4_to_15_l1586_158687

theorem average_primes_4_to_15 :
  (5 + 7 + 11 + 13) / 4 = 9 :=
by sorry

end NUMINAMATH_GPT_average_primes_4_to_15_l1586_158687

import Mathlib

namespace sequence_is_periodic_l891_89153

open Nat

def is_periodic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ p > 0, ∀ i, a (i + p) = a i

theorem sequence_is_periodic (a : ℕ → ℕ)
  (h1 : ∀ n, a n < 1988)
  (h2 : ∀ m n, a m + a n ∣ a (m + n)) : is_periodic_sequence a :=
by
  sorry

end sequence_is_periodic_l891_89153


namespace part1_part2_l891_89189

def A (x : ℝ) : Prop := x^2 + 2*x - 3 < 0
def B (x : ℝ) (a : ℝ) : Prop := abs (x + a) < 1

theorem part1 (a : ℝ) (h : a = 3) : (∃ x : ℝ, (A x ∨ B x a)) ↔ (∃ x : ℝ, -4 < x ∧ x < 1) :=
by {
  sorry
}

theorem part2 : (∀ x : ℝ, B x a → A x) ∧ (¬ ∀ x : ℝ, A x → B x a) ↔ 0 ≤ a ∧ a ≤ 2 :=
by {
  sorry
}

end part1_part2_l891_89189


namespace remaining_distance_l891_89198

theorem remaining_distance (total_depth distance_traveled remaining_distance : ℕ) (h_total_depth : total_depth = 1218) 
  (h_distance_traveled : distance_traveled = 849) : remaining_distance = total_depth - distance_traveled := 
by
  sorry

end remaining_distance_l891_89198


namespace value_of_fraction_l891_89175

theorem value_of_fraction (y : ℝ) (h : 4 - 9 / y + 9 / (y^2) = 0) : 3 / y = 2 :=
sorry

end value_of_fraction_l891_89175


namespace parts_of_alloys_l891_89181

def ratio_of_metals_in_alloy (a1 a2 a3 b1 b2 : ℚ) (x y : ℚ) : Prop :=
  let first_metal := (1 / a3) * x + (a1 / b2) * y
  let second_metal := (2 / a3) * x + (b1 / b2) * y
  (first_metal / second_metal) = (17 / 27)

theorem parts_of_alloys
  (x y : ℚ)
  (a1 a2 a3 b1 b2 : ℚ)
  (h1 : a1 = 1)
  (h2 : a2 = 2)
  (h3 : a3 = 3)
  (h4 : b1 = 2)
  (h5 : b2 = 5)
  (h6 : ratio_of_metals_in_alloy a1 a2 a3 b1 b2 x y) :
  x = 9 ∧ y = 35 :=
sorry

end parts_of_alloys_l891_89181


namespace member_pays_48_percent_of_SRP_l891_89141

theorem member_pays_48_percent_of_SRP
  (P : ℝ)
  (h₀ : P > 0)
  (basic_discount : ℝ := 0.40)
  (additional_discount : ℝ := 0.20) :
  ((1 - additional_discount) * (1 - basic_discount) * P) / P * 100 = 48 := by
  sorry

end member_pays_48_percent_of_SRP_l891_89141


namespace read_books_correct_l891_89123

namespace CrazySillySchool

-- Definitions from conditions
def total_books : Nat := 20
def unread_books : Nat := 5
def read_books : Nat := total_books - unread_books

-- Theorem statement
theorem read_books_correct : read_books = 15 :=
by
  -- Mathematical statement that follows from conditions and correct answer
  sorry

end CrazySillySchool

end read_books_correct_l891_89123


namespace area_of_inscribed_square_l891_89158

theorem area_of_inscribed_square (D : ℝ) (h : D = 10) : 
  ∃ A : ℝ, A = 50 :=
by
  sorry

end area_of_inscribed_square_l891_89158


namespace visual_range_increase_percent_l891_89143

theorem visual_range_increase_percent :
  let original_visual_range := 100
  let new_visual_range := 150
  ((new_visual_range - original_visual_range) / original_visual_range) * 100 = 50 :=
by
  sorry

end visual_range_increase_percent_l891_89143


namespace deposit_correct_l891_89168

-- Define the conditions
def monthly_income : ℝ := 10000
def deposit_percentage : ℝ := 0.25

-- Define the deposit calculation based on the conditions
def deposit_amount (income : ℝ) (percentage : ℝ) : ℝ :=
  percentage * income

-- Theorem: Prove that the deposit amount is Rs. 2500
theorem deposit_correct :
    deposit_amount monthly_income deposit_percentage = 2500 :=
  sorry

end deposit_correct_l891_89168


namespace find_a_and_b_l891_89161

theorem find_a_and_b (a b : ℤ) (h1 : 3 * (b + a^2) = 99) (h2 : 3 * a * b^2 = 162) : a = 6 ∧ b = -3 :=
sorry

end find_a_and_b_l891_89161


namespace solve_for_x_l891_89155

theorem solve_for_x :
  (48 = 5 * x + 3) → x = 9 :=
by
  sorry

end solve_for_x_l891_89155


namespace range_of_x_l891_89100

-- Defining the conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y ≤ f x

-- Given conditions in Lean
axiom f : ℝ → ℝ
axiom h_odd : odd_function f
axiom h_decreasing_pos : ∀ x y, 0 < x ∧ x < y → f y ≤ f x
axiom h_f4 : f 4 = 0

-- To prove the range of x for which f(x-3) ≤ 0
theorem range_of_x :
    {x : ℝ | f (x - 3) ≤ 0} = {x : ℝ | -1 ≤ x ∧ x < 3} ∪ {x : ℝ | 7 ≤ x} :=
by
  sorry

end range_of_x_l891_89100


namespace sufficient_not_necessary_l891_89199

theorem sufficient_not_necessary (a : ℝ) :
  a > 1 → (a^2 > 1) ∧ (∀ a : ℝ, a^2 > 1 → a = -1 ∨ a > 1 → false) :=
by {
  sorry
}

end sufficient_not_necessary_l891_89199


namespace infection_never_covers_grid_l891_89109

theorem infection_never_covers_grid (n : ℕ) (H : n > 0) :
  exists (non_infected_cell : ℕ × ℕ), (non_infected_cell.1 < n ∧ non_infected_cell.2 < n) :=
by
  sorry

end infection_never_covers_grid_l891_89109


namespace maria_total_cost_l891_89140

-- Define the conditions as variables in the Lean environment
def daily_rental_rate : ℝ := 35
def mileage_rate : ℝ := 0.25
def rental_days : ℕ := 3
def miles_driven : ℕ := 500

-- Now, state the theorem that Maria’s total payment should be $230
theorem maria_total_cost : (daily_rental_rate * rental_days) + (mileage_rate * miles_driven) = 230 := 
by
  -- no proof required, just state as sorry
  sorry

end maria_total_cost_l891_89140


namespace bunny_burrows_l891_89192

theorem bunny_burrows (x : ℕ) (h1 : 20 * x * 600 = 36000) : x = 3 :=
by
  -- Skipping proof using sorry
  sorry

end bunny_burrows_l891_89192


namespace right_handed_total_l891_89111

theorem right_handed_total (total_players throwers : Nat) (h1 : total_players = 70) (h2 : throwers = 37) :
  let non_throwers := total_players - throwers
  let left_handed := non_throwers / 3
  let right_handed_non_throwers := non_throwers - left_handed
  let right_handed := right_handed_non_throwers + throwers
  right_handed = 59 :=
by
  sorry

end right_handed_total_l891_89111


namespace max_value_of_b_minus_a_l891_89163

theorem max_value_of_b_minus_a (a b : ℝ) (h1 : a < 0) (h2 : ∀ x : ℝ, a < x ∧ x < b → (3 * x^2 + a) * (2 * x + b) ≥ 0) : b - a ≤ 1 / 3 :=
by
  sorry

end max_value_of_b_minus_a_l891_89163


namespace slope_of_tangent_at_1_l891_89145

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem slope_of_tangent_at_1 : (deriv f 1) = 1 / 2 :=
  by
  sorry

end slope_of_tangent_at_1_l891_89145


namespace price_of_other_stamp_l891_89180

-- Define the conditions
def total_stamps : ℕ := 75
def total_value_cents : ℕ := 480
def known_stamp_price : ℕ := 8
def known_stamp_count : ℕ := 40
def unknown_stamp_count : ℕ := total_stamps - known_stamp_count

-- The problem to solve
theorem price_of_other_stamp (x : ℕ) :
  (known_stamp_count * known_stamp_price) + (unknown_stamp_count * x) = total_value_cents → x = 5 :=
by
  sorry

end price_of_other_stamp_l891_89180


namespace problem_solution_l891_89133

noncomputable def inequality_holds (a b : ℝ) (n : ℕ) : Prop :=
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n - 1)

theorem problem_solution (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / a + 1 / b = 1)) (h4 : 0 < n):
  inequality_holds a b n :=
by
  sorry

end problem_solution_l891_89133


namespace fleas_cannot_reach_final_positions_l891_89144

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def initial_A : Point2D := ⟨0, 0⟩
def initial_B : Point2D := ⟨1, 0⟩
def initial_C : Point2D := ⟨0, 1⟩

def area (A B C : Point2D) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def final_A : Point2D := ⟨1, 0⟩
def final_B : Point2D := ⟨-1, 0⟩
def final_C : Point2D := ⟨0, 1⟩

theorem fleas_cannot_reach_final_positions : 
    ¬ (∃ (flea_move_sequence : List (Point2D → Point2D)), 
    area initial_A initial_B initial_C = area final_A final_B final_C) :=
by 
  sorry

end fleas_cannot_reach_final_positions_l891_89144


namespace bills_are_fake_bart_can_give_exact_amount_l891_89108

-- Problem (a)
theorem bills_are_fake : 
  (∀ x, x = 17 ∨ x = 19 → false) :=
sorry

-- Problem (b)
theorem bart_can_give_exact_amount (n : ℕ) :
  (∀ m, m = 323  → (n ≥ m → ∃ a b : ℕ, n = 17 * a + 19 * b)) :=
sorry

end bills_are_fake_bart_can_give_exact_amount_l891_89108


namespace proof_statement_l891_89146

noncomputable def problem_statement (a b : ℤ) : ℤ :=
  (a^3 + b^3) / (a^2 - a * b + b^2)

theorem proof_statement : problem_statement 5 4 = 9 := by
  sorry

end proof_statement_l891_89146


namespace volume_of_tetrahedron_OABC_l891_89148

-- Definitions of side lengths and their squared values
def side_length_A_B := 7
def side_length_B_C := 8
def side_length_C_A := 9

-- Squared values of coordinates
def a_sq := 33
def b_sq := 16
def c_sq := 48

-- Main statement to prove the volume
theorem volume_of_tetrahedron_OABC :
  (1/6) * (Real.sqrt a_sq) * (Real.sqrt b_sq) * (Real.sqrt c_sq) = 2 * Real.sqrt 176 :=
by
  -- Proof steps would go here
  sorry

end volume_of_tetrahedron_OABC_l891_89148


namespace johns_number_l891_89193

theorem johns_number (n : ℕ) (h1 : ∃ k₁ : ℤ, n = 125 * k₁) (h2 : ∃ k₂ : ℤ, n = 180 * k₂) (h3 : 1000 < n) (h4 : n < 3000) : n = 1800 :=
sorry

end johns_number_l891_89193


namespace find_distance_MF_l891_89149

-- Define the parabola and point conditions
def parabola (x y : ℝ) := y^2 = 8 * x

-- Define the focus of the parabola
def F : ℝ × ℝ := (2, 0)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the distance squared between two points
def dist_squared (A B : ℝ × ℝ) : ℝ :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2

-- Prove the required statement
theorem find_distance_MF (x y : ℝ) (hM : parabola x y) (h_dist: dist_squared (x, y) O = 3 * (x + 2)) :
  dist_squared (x, y) F = 9 := by
  sorry

end find_distance_MF_l891_89149


namespace odd_solution_exists_l891_89130

theorem odd_solution_exists (k m n : ℕ) (h : m * n = k^2 + k + 3) : 
∃ (x y : ℤ), (x^2 + 11 * y^2 = 4 * m ∨ x^2 + 11 * y^2 = 4 * n) ∧ (x % 2 ≠ 0 ∧ y % 2 ≠ 0) :=
sorry

end odd_solution_exists_l891_89130


namespace find_replaced_weight_l891_89142

-- Define the conditions and the hypothesis
def replaced_weight (W : ℝ) : Prop :=
  let avg_increase := 2.5
  let num_persons := 8
  let new_weight := 85
  (new_weight - W) = num_persons * avg_increase

-- Define the statement we aim to prove
theorem find_replaced_weight : replaced_weight 65 :=
by
  -- proof goes here
  sorry

end find_replaced_weight_l891_89142


namespace minimum_value_of_expression_l891_89151

theorem minimum_value_of_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
    (x^4 / (y - 1)) + (y^4 / (x - 1)) ≥ 12 := 
sorry

end minimum_value_of_expression_l891_89151


namespace mary_initial_blue_crayons_l891_89157

/-- **Mathematically equivalent proof problem**:
  Given that Mary has 5 green crayons and gives away 3 green crayons and 1 blue crayon,
  and she has 9 crayons left, prove that she initially had 8 blue crayons. 
  -/
theorem mary_initial_blue_crayons (initial_green_crayons : ℕ) (green_given_away : ℕ) (blue_given_away : ℕ)
  (crayons_left : ℕ) (initial_crayons : ℕ) :
  initial_green_crayons = 5 →
  green_given_away = 3 →
  blue_given_away = 1 →
  crayons_left = 9 →
  initial_crayons = crayons_left + (green_given_away + blue_given_away) →
  initial_crayons - initial_green_crayons = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end mary_initial_blue_crayons_l891_89157


namespace find_probability_l891_89154

noncomputable def probability_distribution (X : ℕ → ℝ) := ∀ k, X k = 1 / (2^k)

theorem find_probability (X : ℕ → ℝ) (h : probability_distribution X) :
  X 3 + X 4 = 3 / 16 :=
by
  sorry

end find_probability_l891_89154


namespace marie_ends_with_755_l891_89102

def erasers_end (initial lost packs erasers_per_pack : ℕ) : ℕ :=
  initial - lost + packs * erasers_per_pack

theorem marie_ends_with_755 :
  erasers_end 950 420 3 75 = 755 :=
by
  sorry

end marie_ends_with_755_l891_89102


namespace original_dining_bill_l891_89177

theorem original_dining_bill (B : ℝ) (h1 : B * 1.15 / 5 = 48.53) : B = 211 := 
sorry

end original_dining_bill_l891_89177


namespace two_circles_with_tangents_l891_89183

theorem two_circles_with_tangents
  (a b : ℝ)                -- radii of the circles
  (length_PQ length_AB : ℝ) -- lengths of the tangents PQ and AB
  (h1 : length_PQ = 14)     -- condition: length of PQ is 14
  (h2 : length_AB = 16)     -- condition: length of AB is 16
  (h3 : length_AB^2 + (a - b)^2 = length_PQ^2 + (a + b)^2) -- from the Pythagorean theorem
  : a * b = 15 := 
sorry

end two_circles_with_tangents_l891_89183


namespace total_pages_l891_89182

theorem total_pages (x : ℕ) (h : 9 + 180 + 3 * (x - 99) = 1392) : x = 500 :=
by
  sorry

end total_pages_l891_89182


namespace lengths_AC_CB_ratio_GJ_JH_coords_F_on_DE_values_p_q_KL_l891_89159

-- Problem 1 - Lengths of AC and CB are 15 and 5 respectively.
theorem lengths_AC_CB (x1 y1 x2 y2 x3 y3 : ℝ) :
  (x1, y1) = (1,2) ∧ (x2, y2) = (17,14) ∧ (x3, y3) = (13,11) →
  ∃ (AC CB : ℝ), AC = 15 ∧ CB = 5 :=
by
  sorry

-- Problem 2 - Ratio of GJ and JH is 3:2.
theorem ratio_GJ_JH (x1 y1 x2 y2 x3 y3 : ℝ) :
  (x1, y1) = (11,2) ∧ (x2, y2) = (1,7) ∧ (x3, y3) = (5,5) →
  ∃ (GJ JH : ℝ), GJ / JH = 3 / 2 :=
by
  sorry

-- Problem 3 - Coordinates of point F on DE with ratio 1:2 is (3,7).
theorem coords_F_on_DE (x1 y1 x2 y2 : ℝ) :
  (x1, y1) = (1,6) ∧ (x2, y2) = (7,9) →
  ∃ (x y : ℝ), (x, y) = (3,7) :=
by
  sorry

-- Problem 4 - Values of p and q for point M on KL with ratio 3:4 are p = 15 and q = 2.
theorem values_p_q_KL (x1 y1 x2 y2 x3 y3 : ℝ) :
  (x1, y1) = (1, q) ∧ (x2, y2) = (p, 9) ∧ (x3, y3) = (7,5) →
  ∃ (p q : ℝ), p = 15 ∧ q = 2 :=
by
  sorry

end lengths_AC_CB_ratio_GJ_JH_coords_F_on_DE_values_p_q_KL_l891_89159


namespace sum_consecutive_even_integers_l891_89119

theorem sum_consecutive_even_integers (n : ℕ) (h : 2 * n + 4 = 156) : 
  n + (n + 2) + (n + 4) = 234 := 
by
  sorry

end sum_consecutive_even_integers_l891_89119


namespace sum_of_first_15_terms_of_arithmetic_sequence_l891_89173

theorem sum_of_first_15_terms_of_arithmetic_sequence 
  (a d : ℕ) 
  (h1 : (5 * (2 * a + 4 * d)) / 2 = 10) 
  (h2 : (10 * (2 * a + 9 * d)) / 2 = 50) :
  (15 * (2 * a + 14 * d)) / 2 = 120 :=
sorry

end sum_of_first_15_terms_of_arithmetic_sequence_l891_89173


namespace second_lady_distance_l891_89186

theorem second_lady_distance (x : ℕ) 
  (h1 : ∃ y, y = 2 * x) 
  (h2 : x + 2 * x = 12) : x = 4 := 
by 
  sorry

end second_lady_distance_l891_89186


namespace inequality_proof_l891_89184

variable (a b c : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c > 0)
variable (h4 : a + b + c = 1)

theorem inequality_proof : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := 
by 
  sorry

end inequality_proof_l891_89184


namespace jessica_remaining_time_after_penalties_l891_89137

-- Definitions for the given conditions
def questions_answered : ℕ := 16
def total_questions : ℕ := 80
def time_used_minutes : ℕ := 12
def exam_duration_minutes : ℕ := 60
def penalty_per_incorrect_answer_minutes : ℕ := 2

-- Define the rate of answering questions
def answering_rate : ℚ := questions_answered / time_used_minutes

-- Define the total time needed to answer all questions
def total_time_needed : ℚ := total_questions / answering_rate

-- Define the remaining time after penalties
def remaining_time_after_penalties (x : ℕ) : ℤ :=
  max 0 (0 - penalty_per_incorrect_answer_minutes * x)

-- The theorem to prove
theorem jessica_remaining_time_after_penalties (x : ℕ) : 
  remaining_time_after_penalties x = max 0 (0 - penalty_per_incorrect_answer_minutes * x) := 
by
  sorry

end jessica_remaining_time_after_penalties_l891_89137


namespace problem_l891_89117

theorem problem (a : ℕ → ℝ) (h0 : a 1 = 0) (h9 : a 9 = 0)
  (h2_8 : ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a i > 0) (h_nonneg : ∀ n, 1 ≤ n ∧ n ≤ 9 → a n ≥ 0) : 
  (∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i-1) + a (i+1) < 2 * a i) ∧ (∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i-1) + a (i+1) < 1.9 * a i) := 
sorry

end problem_l891_89117


namespace symmetric_circle_eq_l891_89195

theorem symmetric_circle_eq {x y : ℝ} :
  (∃ x y : ℝ, (x+2)^2 + (y-1)^2 = 5) →
  (x - 1)^2 + (y + 2)^2 = 5 :=
sorry

end symmetric_circle_eq_l891_89195


namespace geom_seq_common_ratio_l891_89170

theorem geom_seq_common_ratio (a1 : ℤ) (S3 : ℚ) (q : ℚ) (hq : -2 * (1 + q + q^2) = - (7 / 2)) : 
  q = 1 / 2 ∨ q = -3 / 2 :=
sorry

end geom_seq_common_ratio_l891_89170


namespace building_height_l891_89191

theorem building_height (h : ℕ) 
  (flagpole_height flagpole_shadow building_shadow : ℕ)
  (h_flagpole : flagpole_height = 18)
  (s_flagpole : flagpole_shadow = 45)
  (s_building : building_shadow = 70) 
  (condition : flagpole_height / flagpole_shadow = h / building_shadow) :
  h = 28 := by
  sorry

end building_height_l891_89191


namespace never_consecutive_again_l891_89121

theorem never_consecutive_again (n : ℕ) (seq : ℕ → ℕ) :
  (∀ k, seq k = seq 0 + k) → 
  ∀ seq' : ℕ → ℕ,
    (∀ i j, i < j → seq' (2*i) = seq i + seq (j) ∧ seq' (2*i+1) = seq i - seq (j)) →
    ¬ (∀ k, seq' k = seq' 0 + k) :=
by
  sorry

end never_consecutive_again_l891_89121


namespace solve_for_c_l891_89103

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (c * x) / (2 * x + 3)

theorem solve_for_c {c : ℝ} (hc : ∀ x ≠ (-3/2), f c (f c x) = x) : c = -3 :=
by
  intros
  -- The proof steps will go here
  sorry

end solve_for_c_l891_89103


namespace min_sum_of_squares_of_y_coords_l891_89122

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line_through_point (m : ℝ) (x y : ℝ) : Prop := x = m * y + 4

theorem min_sum_of_squares_of_y_coords :
  ∃ (m : ℝ), ∀ (x1 y1 x2 y2 : ℝ),
  (line_through_point m x1 y1) →
  (parabola x1 y1) →
  (line_through_point m x2 y2) →
  (parabola x2 y2) →
  x1 ≠ x2 → 
  ((y1 + y2)^2 - 2 * y1 * y2) = 32 :=
sorry

end min_sum_of_squares_of_y_coords_l891_89122


namespace eq_of_divides_l891_89179

theorem eq_of_divides (a b : ℕ) (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b :=
sorry

end eq_of_divides_l891_89179


namespace no_such_decreasing_h_exists_l891_89185

-- Define the interval [0, ∞)
def nonneg_reals := {x : ℝ // 0 ≤ x}

-- Define a decreasing function h on [0, ∞)
def is_decreasing (h : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → h x ≥ h y

-- Define the function f based on h
def f (h : ℝ → ℝ) (x : ℝ) : ℝ := (x^2 - x + 1) * h x

-- Define the increasing property for f on [0, ∞)
def is_increasing_on_nonneg_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x ≤ f y

theorem no_such_decreasing_h_exists :
  ¬ ∃ h : ℝ → ℝ, is_decreasing h ∧ is_increasing_on_nonneg_reals (f h) :=
by sorry

end no_such_decreasing_h_exists_l891_89185


namespace biology_exam_students_l891_89160

theorem biology_exam_students :
  let students := 200
  let score_A := (1 / 4) * students
  let remaining_students := students - score_A
  let score_B := (1 / 5) * remaining_students
  let score_C := (1 / 3) * remaining_students
  let score_D := (5 / 12) * remaining_students
  let score_F := students - (score_A + score_B + score_C + score_D)
  let re_assessed_C := (3 / 5) * score_C
  let final_score_B := score_B + re_assessed_C
  let final_score_C := score_C - re_assessed_C
  score_A = 50 ∧ 
  final_score_B = 60 ∧ 
  final_score_C = 20 ∧ 
  score_D = 62 ∧ 
  score_F = 8 :=
by {
  sorry
}

end biology_exam_students_l891_89160


namespace man_l891_89164

-- Constants and conditions
def V_down : ℝ := 18  -- downstream speed in km/hr
def V_c : ℝ := 3.4    -- speed of the current in km/hr

-- Main statement to prove
theorem man's_speed_against_the_current : (V_down - V_c - V_c) = 11.2 := by
  sorry

end man_l891_89164


namespace total_balls_estimation_l891_89126

theorem total_balls_estimation 
  (num_red_balls : ℕ)
  (total_trials : ℕ)
  (red_ball_draws : ℕ)
  (red_ball_ratio : ℚ)
  (total_balls_estimate : ℕ)
  (h1 : num_red_balls = 5)
  (h2 : total_trials = 80)
  (h3 : red_ball_draws = 20)
  (h4 : red_ball_ratio = 1 / 4)
  (h5 : red_ball_ratio = red_ball_draws / total_trials)
  (h6 : red_ball_ratio = num_red_balls / total_balls_estimate)
  : total_balls_estimate = 20 := 
sorry

end total_balls_estimation_l891_89126


namespace discount_rate_on_pony_jeans_l891_89138

theorem discount_rate_on_pony_jeans 
  (F P : ℝ) 
  (H1 : F + P = 22) 
  (H2 : 45 * F + 36 * P = 882) : 
  P = 12 :=
by
  sorry

end discount_rate_on_pony_jeans_l891_89138


namespace time_away_is_43point64_minutes_l891_89134

theorem time_away_is_43point64_minutes :
  ∃ (n1 n2 : ℝ), 
    (195 + n1 / 2 - 6 * n1 = 120 ∨ 195 + n1 / 2 - 6 * n1 = -120) ∧
    (195 + n2 / 2 - 6 * n2 = 120 ∨ 195 + n2 / 2 - 6 * n2 = -120) ∧
    n1 ≠ n2 ∧
    n1 < 60 ∧
    n2 < 60 ∧
    |n2 - n1| = 43.64 :=
sorry

end time_away_is_43point64_minutes_l891_89134


namespace determine_solution_set_inequality_l891_89171

-- Definitions based on given conditions
def quadratic_inequality_solution (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c > 0
def new_quadratic_inequality_solution (c b a : ℝ) (x : ℝ) := c * x^2 + b * x + a < 0

-- The proof statement
theorem determine_solution_set_inequality (a b c : ℝ):
  (∀ x : ℝ, -1/3 < x ∧ x < 2 → quadratic_inequality_solution a b c x) →
  (∀ x : ℝ, -3 < x ∧ x < 1/2 ↔ new_quadratic_inequality_solution c b a x) := sorry

end determine_solution_set_inequality_l891_89171


namespace bridge_extension_length_l891_89110

theorem bridge_extension_length (river_width bridge_length : ℕ) (h_river : river_width = 487) (h_bridge : bridge_length = 295) : river_width - bridge_length = 192 :=
by
  sorry

end bridge_extension_length_l891_89110


namespace workers_contribution_l891_89104

theorem workers_contribution (N C : ℕ) 
(h1 : N * C = 300000) 
(h2 : N * (C + 50) = 360000) : 
N = 1200 :=
sorry

end workers_contribution_l891_89104


namespace knight_probability_sum_l891_89118

def num_knights := 30
def chosen_knights := 4

-- Calculate valid placements where no knights are adjacent
def valid_placements : ℕ := 26 * 24 * 22 * 20
-- Calculate total unrestricted placements
def total_placements : ℕ := 26 * 27 * 28 * 29
-- Calculate probability
def P : ℚ := 1 - (valid_placements : ℚ) / total_placements

-- Simplify the fraction P to its lowest terms: 553/1079
def simplified_num := 553
def simplified_denom := 1079

-- Sum of the numerator and denominator of simplified P
def sum_numer_denom := simplified_num + simplified_denom

theorem knight_probability_sum :
  sum_numer_denom = 1632 :=
by
  -- Proof is omitted
  sorry

end knight_probability_sum_l891_89118


namespace nadia_flower_shop_l891_89113

theorem nadia_flower_shop :
  let roses := 20
  let lilies := (3 / 4) * roses
  let cost_per_rose := 5
  let cost_per_lily := 2 * cost_per_rose
  let total_cost := roses * cost_per_rose + lilies * cost_per_lily
  total_cost = 250 := by
    sorry

end nadia_flower_shop_l891_89113


namespace total_distance_covered_l891_89167

theorem total_distance_covered :
  let speed1 := 40 -- miles per hour
  let speed2 := 50 -- miles per hour
  let speed3 := 30 -- miles per hour
  let time1 := 1.5 -- hours
  let time2 := 1 -- hour
  let time3 := 2.25 -- hours
  let distance1 := speed1 * time1 -- distance covered in the first part of the trip
  let distance2 := speed2 * time2 -- distance covered in the second part of the trip
  let distance3 := speed3 * time3 -- distance covered in the third part of the trip
  distance1 + distance2 + distance3 = 177.5 := 
by
  sorry

end total_distance_covered_l891_89167


namespace pure_imaginary_b_eq_two_l891_89194

theorem pure_imaginary_b_eq_two (b : ℝ) : (∃ (im_part : ℝ), (1 + b * Complex.I) / (2 - Complex.I) = im_part * Complex.I) ↔ b = 2 :=
by
  sorry

end pure_imaginary_b_eq_two_l891_89194


namespace binomial_inequality_l891_89197

theorem binomial_inequality (n : ℤ) (x : ℝ) (hn : n ≥ 2) (hx : |x| < 1) : 
  2^n > (1 - x)^n + (1 + x)^n := 
sorry

end binomial_inequality_l891_89197


namespace line_eq_l891_89136

variables {x x1 x2 y y1 y2 : ℝ}

theorem line_eq (h : x2 ≠ x1 ∧ y2 ≠ y1) : 
  (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1) :=
sorry

end line_eq_l891_89136


namespace kevin_exchanges_l891_89101

variables (x y : ℕ)

def R (x y : ℕ) := 100 - 3 * x + 2 * y
def B (x y : ℕ) := 100 + 2 * x - 4 * y

theorem kevin_exchanges :
  (∃ x y, R x y >= 3 ∧ B x y >= 4 ∧ x + y = 132) :=
sorry

end kevin_exchanges_l891_89101


namespace range_of_a_l891_89156

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a - 1) * x + 1
noncomputable def f' (a x : ℝ) : ℝ := x^2 - a * x + a - 1

theorem range_of_a (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → f' a x ≤ 0) ∧ (∀ x, 6 < x → f' a x ≥ 0) ↔ 5 ≤ a ∧ a ≤ 7 :=
by
  sorry

end range_of_a_l891_89156


namespace sqrt_interval_l891_89105

theorem sqrt_interval :
  let expr := (Real.sqrt 18) / 3 - (Real.sqrt 2) * (Real.sqrt (1 / 2))
  0 < expr ∧ expr < 1 :=
by
  let expr := (Real.sqrt 18) / 3 - (Real.sqrt 2) * (Real.sqrt (1 / 2))
  sorry

end sqrt_interval_l891_89105


namespace students_in_both_clubs_l891_89172

theorem students_in_both_clubs :
  ∀ (total_students drama_club science_club either_club both_club : ℕ),
  total_students = 300 →
  drama_club = 100 →
  science_club = 140 →
  either_club = 220 →
  (drama_club + science_club - both_club = either_club) →
  both_club = 20 :=
by
  intros total_students drama_club science_club either_club both_club
  intros h1 h2 h3 h4 h5
  sorry

end students_in_both_clubs_l891_89172


namespace martin_waste_time_l891_89139

theorem martin_waste_time : 
  let waiting_traffic := 2
  let trying_off_freeway := 4 * waiting_traffic
  let detours := 3 * 30 / 60
  let meal := 45 / 60
  let delays := (20 + 40) / 60
  waiting_traffic + trying_off_freeway + detours + meal + delays = 13.25 := 
by
  sorry

end martin_waste_time_l891_89139


namespace abby_bridget_adjacent_probability_l891_89116

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def probability_adjacent : ℚ :=
  let total_seats := 9
  let ab_adj_same_row_pairs := 9
  let ab_adj_diagonal_pairs := 4
  let favorable_outcomes := (ab_adj_same_row_pairs + ab_adj_diagonal_pairs) * 2 * factorial 7
  let total_outcomes := factorial total_seats
  favorable_outcomes / total_outcomes

theorem abby_bridget_adjacent_probability :
  probability_adjacent = 13 / 36 :=
by
  sorry

end abby_bridget_adjacent_probability_l891_89116


namespace batsman_average_after_11th_inning_l891_89120

theorem batsman_average_after_11th_inning
  (x : ℝ)  -- the average score of the batsman before the 11th inning
  (h1 : 10 * x + 85 = 11 * (x + 5))  -- given condition from the problem
  : x + 5 = 35 :=   -- goal statement proving the new average
by
  -- We need to prove that new average after the 11th inning is 35
  sorry

end batsman_average_after_11th_inning_l891_89120


namespace tank_A_is_60_percent_of_tank_B_capacity_l891_89150

-- Conditions
def height_A : ℝ := 10
def circumference_A : ℝ := 6
def height_B : ℝ := 6
def circumference_B : ℝ := 10

-- Statement
theorem tank_A_is_60_percent_of_tank_B_capacity (V_A V_B : ℝ) (radius_A radius_B : ℝ)
  (hA : radius_A = circumference_A / (2 * Real.pi))
  (hB : radius_B = circumference_B / (2 * Real.pi))
  (vol_A : V_A = Real.pi * radius_A^2 * height_A)
  (vol_B : V_B = Real.pi * radius_B^2 * height_B) :
  (V_A / V_B) * 100 = 60 :=
by
  sorry

end tank_A_is_60_percent_of_tank_B_capacity_l891_89150


namespace correct_card_assignment_l891_89135

theorem correct_card_assignment :
  ∃ (cards : Fin 4 → Fin 4), 
    (¬ (cards 1 = 3 ∨ cards 2 = 3) ∧
     ¬ (cards 0 = 2 ∨ cards 2 = 2) ∧
     ¬ (cards 0 = 1) ∧
     ¬ (cards 0 = 3)) →
    (cards 0 = 4 ∧ cards 1 = 2 ∧ cards 2 = 1 ∧ cards 3 = 3) := 
by {
  sorry
}

end correct_card_assignment_l891_89135


namespace train_seat_count_l891_89128

theorem train_seat_count (t : ℝ)
  (h1 : ∃ (t : ℝ), t = 36 + 0.2 * t + 0.5 * t) :
  t = 120 :=
by
  sorry

end train_seat_count_l891_89128


namespace greatest_integer_difference_l891_89147

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8) : 
  ∃ d, d = y - x ∧ d = 2 := 
by
  sorry

end greatest_integer_difference_l891_89147


namespace ali_initial_money_l891_89131

theorem ali_initial_money (X : ℝ) (h1 : X / 2 - (1 / 3) * (X / 2) = 160) : X = 480 :=
by sorry

end ali_initial_money_l891_89131


namespace least_multiple_of_11_not_lucky_l891_89174

-- Define what it means for a number to be a lucky integer
def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

-- Define what it means for a number to be a multiple of 11
def is_multiple_of_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- State the problem: the least positive multiple of 11 that is not a lucky integer is 132
theorem least_multiple_of_11_not_lucky :
  ∃ n : ℕ, is_multiple_of_11 n ∧ ¬ is_lucky n ∧ n = 132 :=
sorry

end least_multiple_of_11_not_lucky_l891_89174


namespace complement_of_M_l891_89125

open Set

-- Define the universal set
def U : Set ℝ := univ

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

-- The theorem stating the complement of M in U
theorem complement_of_M : (U \ M) = {y | y < -1} :=
by
  sorry

end complement_of_M_l891_89125


namespace tan_value_l891_89187

variable (a : ℕ → ℝ) (b : ℕ → ℝ)
variable (a_geom : ∀ m n : ℕ, a m / a n = a (m - n))
variable (b_arith : ∃ c d : ℝ, ∀ n : ℕ, b n = c + n * d)
variable (ha : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
variable (hb : b 1 + b 6 + b 11 = 7 * Real.pi)

theorem tan_value : Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 := by
  sorry

end tan_value_l891_89187


namespace alpha_and_2beta_l891_89124

theorem alpha_and_2beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
  (h_tan_alpha : Real.tan α = 1 / 8) (h_sin_beta : Real.sin β = 1 / 3) :
  α + 2 * β = Real.arctan (15 / 56) := by
  sorry

end alpha_and_2beta_l891_89124


namespace minimum_omega_l891_89166

theorem minimum_omega (ω : ℝ) (k : ℤ) (hω : ω > 0) 
  (h_symmetry : ∃ k : ℤ, ω * (π / 12) + π / 6 = k * π + π / 2) : ω = 4 :=
sorry

end minimum_omega_l891_89166


namespace missing_number_unique_l891_89107

theorem missing_number_unique (x : ℤ) 
  (h : |9 - x * (3 - 12)| - |5 - 11| = 75) : 
  x = 8 :=
sorry

end missing_number_unique_l891_89107


namespace student_tickets_sold_l891_89178

theorem student_tickets_sold (S NS : ℕ) (h1 : 9 * S + 11 * NS = 20960) (h2 : S + NS = 2000) : S = 520 :=
by
  sorry

end student_tickets_sold_l891_89178


namespace tangent_line_l891_89152

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 1 / x

theorem tangent_line (x y : ℝ) (h_inter : y = f x ∧ y = g x) :
  (x - 2 * y + 1 = 0) :=
by
  sorry

end tangent_line_l891_89152


namespace solve_for_y_l891_89127

theorem solve_for_y (y : ℝ) (h : 5 * (y ^ (1/3)) - 3 * (y / (y ^ (2/3))) = 10 + (y ^ (1/3))) :
  y = 1000 :=
by {
  sorry
}

end solve_for_y_l891_89127


namespace exponent_division_l891_89165

theorem exponent_division : (23 ^ 11) / (23 ^ 8) = 12167 := 
by {
  sorry
}

end exponent_division_l891_89165


namespace Dylan_needs_two_trays_l891_89106

noncomputable def ice_cubes_glass : ℕ := 8
noncomputable def ice_cubes_pitcher : ℕ := 2 * ice_cubes_glass
noncomputable def tray_capacity : ℕ := 12
noncomputable def total_ice_cubes_used : ℕ := ice_cubes_glass + ice_cubes_pitcher
noncomputable def number_of_trays : ℕ := total_ice_cubes_used / tray_capacity

theorem Dylan_needs_two_trays : number_of_trays = 2 := by
  sorry

end Dylan_needs_two_trays_l891_89106


namespace abs_diff_squares_l891_89115

theorem abs_diff_squares (a b : ℤ) (h_a : a = 105) (h_b : b = 95):
  |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l891_89115


namespace sum_geometric_sequence_l891_89190

theorem sum_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (a1 : ℝ)
  (h1 : a 5 = -2) (h2 : a 8 = 16)
  (hq : q^3 = a 8 / a 5) (ha1 : a 1 = a1)
  (hS : S n = a1 * (1 - q^n) / (1 - q))
  : S 6 = 21 / 8 :=
sorry

end sum_geometric_sequence_l891_89190


namespace proof_problem_l891_89176

theorem proof_problem (x : ℝ) (a : ℝ) :
  (0 < x) → 
  (x + 1 / x ≥ 2) →
  (x + 4 / x^2 ≥ 3) →
  (x + 27 / x^3 ≥ 4) →
  a = 4^4 → 
  x + a / x^4 ≥ 5 :=
  sorry

end proof_problem_l891_89176


namespace least_beans_l891_89132

-- Define the conditions 
variables (r b : ℕ)

-- State the theorem 
theorem least_beans (h1 : r ≥ 2 * b + 8) (h2 : r ≤ 3 * b) : b ≥ 8 :=
by
  sorry

end least_beans_l891_89132


namespace probability_of_passing_l891_89188

theorem probability_of_passing (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end probability_of_passing_l891_89188


namespace abc_minus_def_l891_89114

def f (x y z : ℕ) : ℕ := 5^x * 2^y * 3^z

theorem abc_minus_def {a b c d e f : ℕ} (ha : a = d) (hb : b = e) (hc : c = f + 1) : 
  (100 * a + 10 * b + c) - (100 * d + 10 * e + f) = 1 :=
by
  -- Proof omitted
  sorry

end abc_minus_def_l891_89114


namespace div_by_7_iff_sum_div_by_7_l891_89112

theorem div_by_7_iff_sum_div_by_7 (a b : ℕ) : 
  (101 * a + 10 * b) % 7 = 0 ↔ (a + b) % 7 = 0 := 
by
  sorry

end div_by_7_iff_sum_div_by_7_l891_89112


namespace angle_A_is_correct_l891_89129

-- Define the given conditions and the main theorem.
theorem angle_A_is_correct (A : ℝ) (m n : ℝ × ℝ) 
  (h_m : m = (Real.sin (A / 2), Real.cos (A / 2)))
  (h_n : n = (Real.cos (A / 2), -Real.cos (A / 2)))
  (h_eq : 2 * ((Prod.fst m * Prod.fst n) + (Prod.snd m * Prod.snd n)) + (Real.sqrt ((Prod.fst m)^2 + (Prod.snd m)^2)) = Real.sqrt 2 / 2) 
  : A = 5 * Real.pi / 12 := by
  sorry

end angle_A_is_correct_l891_89129


namespace find_number_l891_89196

theorem find_number (x : ℝ) (h : 140 = 3.5 * x) : x = 40 :=
by
  sorry

end find_number_l891_89196


namespace shaded_region_area_l891_89169

def area_of_square (side : ℕ) : ℕ := side * side

def area_of_triangle (base height : ℕ) : ℕ := (base * height) / 2

def combined_area_of_triangles (base height : ℕ) : ℕ := 2 * area_of_triangle base height

def shaded_area (square_side : ℕ) (triangle_base triangle_height : ℕ) : ℕ :=
  area_of_square square_side - combined_area_of_triangles triangle_base triangle_height

theorem shaded_region_area (h₁ : area_of_square 40 = 1600)
                          (h₂ : area_of_triangle 30 30 = 450)
                          (h₃ : combined_area_of_triangles 30 30 = 900) :
  shaded_area 40 30 30 = 700 :=
by
  sorry

end shaded_region_area_l891_89169


namespace new_selling_price_l891_89162

theorem new_selling_price (C : ℝ) (h1 : 1.10 * C = 88) :
  1.15 * C = 92 :=
sorry

end new_selling_price_l891_89162

import Mathlib

namespace minimum_phi_l2261_226101

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 4))

theorem minimum_phi (φ : ℝ) (hφ : φ > 0) :
  (∃ k : ℤ, φ = (3/8) * Real.pi - (k * Real.pi / 2)) → φ = (3/8) * Real.pi :=
by
  sorry

end minimum_phi_l2261_226101


namespace percentage_difference_l2261_226143

theorem percentage_difference (x y : ℝ) (h : x = 3 * y) : ((x - y) / x) * 100 = 66.67 :=
by
  sorry

end percentage_difference_l2261_226143


namespace correct_statement_C_l2261_226196

def V_m_rho_relation (V m ρ : ℝ) : Prop :=
  V = m / ρ

theorem correct_statement_C (V m ρ : ℝ) (h : ρ ≠ 0) : 
  ((∃ k : ℝ, k = ρ ∧ ∀ V' m' : ℝ, V' = m' / k → V' ≠ V) ∧ 
  (∃ v_var v_var', v_var = V ∧ v_var' = m ∧ V = m / ρ) →
  (∃ ρ_const : ℝ, ρ_const = ρ)) :=
by
  sorry

end correct_statement_C_l2261_226196


namespace proof_problem_l2261_226129

-- Definitions of arithmetic and geometric sequences
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b1 r : ℝ), ∀ n, b n = b1 * r^n

-- Lean statement of the problem
theorem proof_problem 
  (a b : ℕ → ℝ)
  (h_a_arithmetic : is_arithmetic_sequence a)
  (h_b_geometric : is_geometric_sequence b)
  (h_condition : a 1 - (a 7)^2 + a 13 = 0)
  (h_b7_a7 : b 7 = a 7) :
  b 3 * b 11 = 4 :=
sorry

end proof_problem_l2261_226129


namespace total_copies_l2261_226176

-- Conditions: Defining the rates of two copy machines and the time duration
def rate1 : ℕ := 35 -- rate in copies per minute for the first machine
def rate2 : ℕ := 65 -- rate in copies per minute for the second machine
def time : ℕ := 30 -- time in minutes

-- The theorem stating that the total number of copies made by both machines in 30 minutes is 3000
theorem total_copies : rate1 * time + rate2 * time = 3000 := by
  sorry

end total_copies_l2261_226176


namespace clock_hand_speed_ratio_l2261_226149

theorem clock_hand_speed_ratio :
  (360 / 720 : ℝ) / (360 / 60 : ℝ) = (2 / 24 : ℝ) := by
    sorry

end clock_hand_speed_ratio_l2261_226149


namespace part1_part2_l2261_226133

-- Definition for f(x)
def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

-- The first proof problem: Solve the inequality f(x) > 0
theorem part1 {x : ℝ} : f x > 0 ↔ x > 1 ∨ x < -5 :=
sorry

-- The second proof problem: Finding the range of m
theorem part2 {m : ℝ} : (∀ x, f x + 3 * |x - 4| ≥ m) → m ≤ 9 :=
sorry

end part1_part2_l2261_226133


namespace quadratic_equation_in_one_variable_l2261_226162

-- Definitions for each condition
def equation_A (x : ℝ) : Prop := x^2 = -1
def equation_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def equation_C (x : ℝ) : Prop := 2 * (x + 1)^2 = (Real.sqrt 2 * x - 1)^2
def equation_D (x : ℝ) : Prop := x + 1 / x = 1

-- Main theorem statement
theorem quadratic_equation_in_one_variable (x : ℝ) :
  equation_A x ∧ ¬(∃ a b c, equation_B a b c x ∧ a ≠ 0) ∧ ¬equation_C x ∧ ¬equation_D x :=
  sorry

end quadratic_equation_in_one_variable_l2261_226162


namespace original_number_is_24_l2261_226127

theorem original_number_is_24 (N : ℕ) 
  (h1 : (N + 1) % 25 = 0)
  (h2 : 1 = 1) : N = 24 := 
sorry

end original_number_is_24_l2261_226127


namespace single_bill_value_l2261_226112

theorem single_bill_value 
  (total_amount : ℕ) 
  (num_5_dollar_bills : ℕ) 
  (amount_5_dollar_bills : ℕ) 
  (single_bill : ℕ) : 
  total_amount = 45 → 
  num_5_dollar_bills = 7 → 
  amount_5_dollar_bills = 5 → 
  total_amount = num_5_dollar_bills * amount_5_dollar_bills + single_bill → 
  single_bill = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end single_bill_value_l2261_226112


namespace value_of_k_l2261_226140

theorem value_of_k (x y : ℝ) (t : ℝ) (k : ℝ) : 
  (x + t * y + 8 = 0) ∧ (5 * x - t * y + 4 = 0) ∧ (3 * x - k * y + 1 = 0) → k = 5 :=
by
  sorry

end value_of_k_l2261_226140


namespace count_perfect_squares_l2261_226151

theorem count_perfect_squares (N : Nat) :
  ∃ k : Nat, k = 1666 ∧ ∀ m, (∃ n, m = n * n ∧ m < 10^8 ∧ 36 ∣ m) ↔ (m = 36 * k ^ 2 ∧ k < 10^4) :=
sorry

end count_perfect_squares_l2261_226151


namespace intersection_of_M_and_N_l2261_226195

def M : Set ℝ := { x : ℝ | x^2 - x > 0 }
def N : Set ℝ := { x : ℝ | x ≥ 1 }

theorem intersection_of_M_and_N : M ∩ N = { x : ℝ | x > 1 } :=
by
  sorry

end intersection_of_M_and_N_l2261_226195


namespace smallest_constant_inequality_l2261_226155

open Real

theorem smallest_constant_inequality (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
    sqrt (x / (y + z + w)) + sqrt (y / (x + z + w)) + sqrt (z / (x + y + w)) + sqrt (w / (x + y + z)) ≤ 2 := by
  sorry

end smallest_constant_inequality_l2261_226155


namespace alpha_when_beta_neg4_l2261_226110

theorem alpha_when_beta_neg4 :
  (∀ (α β : ℝ), (β ≠ 0) → α = 5 → β = 2 → α * β^2 = α * 4) →
   ∃ (α : ℝ), α = 5 → ∃ β, β = -4 → α = 5 / 4 :=
  by
    intros h
    use 5 / 4
    sorry

end alpha_when_beta_neg4_l2261_226110


namespace second_player_can_ensure_symmetry_l2261_226174

def is_symmetric (seq : List ℕ) : Prop :=
  seq.reverse = seq

def swap_digits (seq : List ℕ) (i j : ℕ) : List ℕ :=
  if h : i < seq.length ∧ j < seq.length then
    seq.mapIdx (λ k x => if k = i then seq.get ⟨j, h.2⟩ 
                        else if k = j then seq.get ⟨i, h.1⟩ 
                        else x)
  else seq

theorem second_player_can_ensure_symmetry (seq : List ℕ) (h : seq.length = 1999) :
  (∃ swappable_seq : List ℕ, is_symmetric swappable_seq) :=
by
  sorry

end second_player_can_ensure_symmetry_l2261_226174


namespace sum_of_arithmetic_sequence_l2261_226153

-- Define the conditions
def is_arithmetic_sequence (first_term last_term : ℕ) (terms : ℕ) : Prop :=
  ∃ (a l : ℕ) (n : ℕ), a = first_term ∧ l = last_term ∧ n = terms ∧ n > 1

-- State the theorem
theorem sum_of_arithmetic_sequence (a l n : ℕ) (h_arith: is_arithmetic_sequence 5 41 10):
  n = 10 ∧ a = 5 ∧ l = 41 → (n * (a + l) / 2) = 230 :=
by
  intros h
  sorry

end sum_of_arithmetic_sequence_l2261_226153


namespace problem_statement_eq_l2261_226150

variable (x y : ℝ)

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem problem_statement_eq :
  dollar ((x + y) ^ 2) ((y + x) ^ 2) = 0 := by
  sorry

end problem_statement_eq_l2261_226150


namespace brandon_skittles_final_l2261_226159
-- Conditions
def brandon_initial_skittles := 96
def brandon_lost_skittles := 9

-- Theorem stating the question and answer
theorem brandon_skittles_final : brandon_initial_skittles - brandon_lost_skittles = 87 := 
by
  -- Proof steps go here
  sorry

end brandon_skittles_final_l2261_226159


namespace derivative_of_f_l2261_226182

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem derivative_of_f (x : ℝ) (hx : x ≠ 0) : deriv f x = ((x * Real.exp x - Real.exp x) / (x * x)) :=
by
  sorry

end derivative_of_f_l2261_226182


namespace jackson_final_grade_l2261_226139

def jackson_hours_playing_video_games : ℕ := 9

def ratio_study_to_play : ℚ := 1 / 3

def time_spent_studying (hours_playing : ℕ) (ratio : ℚ) : ℚ := hours_playing * ratio

def points_per_hour_studying : ℕ := 15

def jackson_grade (time_studied : ℚ) (points_per_hour : ℕ) : ℚ := time_studied * points_per_hour

theorem jackson_final_grade :
  jackson_grade
    (time_spent_studying jackson_hours_playing_video_games ratio_study_to_play)
    points_per_hour_studying = 45 :=
by
  sorry

end jackson_final_grade_l2261_226139


namespace ratio_a7_b7_l2261_226199

variable (a b : ℕ → ℝ)
variable (S T : ℕ → ℝ)

-- Given conditions
axiom sum_S : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * a 2) -- Formula for sum of arithmetic series
axiom sum_T : ∀ n, T n = (n / 2) * (2 * b 1 + (n - 1) * b 2) -- Formula for sum of arithmetic series
axiom ratio_ST : ∀ n, S n / T n = (2 * n + 1) / (n + 3)

-- Prove the ratio of seventh terms
theorem ratio_a7_b7 : a 7 / b 7 = 27 / 16 :=
by
  sorry

end ratio_a7_b7_l2261_226199


namespace count_ways_to_choose_and_discard_l2261_226108

theorem count_ways_to_choose_and_discard :
  let suits := 4 
  let cards_per_suit := 13
  let ways_to_choose_4_different_suits := Nat.choose 4 4
  let ways_to_choose_4_cards := cards_per_suit ^ 4
  let ways_to_discard_1_card := 4
  1 * ways_to_choose_4_cards * ways_to_discard_1_card = 114244 :=
by
  sorry

end count_ways_to_choose_and_discard_l2261_226108


namespace negation_exists_l2261_226105

theorem negation_exists (h : ∀ x : ℝ, 0 < x → Real.sin x < x) : ∃ x : ℝ, 0 < x ∧ Real.sin x ≥ x :=
by
  sorry

end negation_exists_l2261_226105


namespace reciprocal_of_neg_five_l2261_226181

theorem reciprocal_of_neg_five : (1 / (-5 : ℝ)) = -1 / 5 := 
by
  sorry

end reciprocal_of_neg_five_l2261_226181


namespace sale_price_of_sarees_l2261_226191

theorem sale_price_of_sarees 
  (P : ℝ) 
  (d1 d2 d3 d4 tax_rate : ℝ) 
  (P_initial : P = 510) 
  (d1_val : d1 = 0.12) 
  (d2_val : d2 = 0.15) 
  (d3_val : d3 = 0.20) 
  (d4_val : d4 = 0.10) 
  (tax_val : tax_rate = 0.10) :
  let discount_step (price discount : ℝ) := price * (1 - discount)
  let tax_step (price tax_rate : ℝ) := price * (1 + tax_rate)
  let P1 := discount_step P d1
  let P2 := discount_step P1 d2
  let P3 := discount_step P2 d3
  let P4 := discount_step P3 d4
  let final_price := tax_step P4 tax_rate
  abs (final_price - 302.13) < 0.01 := 
sorry

end sale_price_of_sarees_l2261_226191


namespace plants_same_height_after_54_years_l2261_226166

noncomputable def h1 (t : ℝ) : ℝ := 44 + (3 / 2) * t
noncomputable def h2 (t : ℝ) : ℝ := 80 + (5 / 6) * t

theorem plants_same_height_after_54_years :
  ∃ t : ℝ, h1 t = h2 t :=
by
  use 54
  sorry

end plants_same_height_after_54_years_l2261_226166


namespace number_of_red_balls_l2261_226132

theorem number_of_red_balls (total_balls : ℕ) (prob_red : ℚ) (h : total_balls = 20 ∧ prob_red = 0.25) : ∃ x : ℕ, x = 5 :=
by
  sorry

end number_of_red_balls_l2261_226132


namespace greatest_possible_large_chips_l2261_226120

theorem greatest_possible_large_chips (s l : ℕ) (even_prime : ℕ) (h1 : s + l = 100) (h2 : s = l + even_prime) (h3 : even_prime = 2) : l = 49 :=
by
  sorry

end greatest_possible_large_chips_l2261_226120


namespace total_handshakes_l2261_226164

theorem total_handshakes (total_people : ℕ) (first_meeting_people : ℕ) (second_meeting_new_people : ℕ) (common_people : ℕ)
  (total_people_is : total_people = 12)
  (first_meeting_people_is : first_meeting_people = 7)
  (second_meeting_new_people_is : second_meeting_new_people = 5)
  (common_people_is : common_people = 2)
  (first_meeting_handshakes : ℕ := (first_meeting_people * (first_meeting_people - 1)) / 2)
  (second_meeting_handshakes: ℕ := (first_meeting_people * (first_meeting_people - 1)) / 2 - (common_people * (common_people - 1)) / 2):
  first_meeting_handshakes + second_meeting_handshakes = 41 := 
sorry

end total_handshakes_l2261_226164


namespace Jan_height_is_42_l2261_226116

-- Given conditions
def Cary_height : ℕ := 72
def Bill_height : ℕ := Cary_height / 2
def Jan_height : ℕ := Bill_height + 6

-- Statement to prove
theorem Jan_height_is_42 : Jan_height = 42 := by
  sorry

end Jan_height_is_42_l2261_226116


namespace spider_moves_away_from_bee_l2261_226183

noncomputable def bee : ℝ × ℝ := (14, 5)
noncomputable def spider_line (x : ℝ) : ℝ := -3 * x + 25
noncomputable def perpendicular_line (x : ℝ) : ℝ := (1 / 3) * x + 14 / 3

theorem spider_moves_away_from_bee : ∃ (c d : ℝ), 
  (d = spider_line c) ∧ (d = perpendicular_line c) ∧ c + d = 13.37 := 
sorry

end spider_moves_away_from_bee_l2261_226183


namespace distance_between_stations_l2261_226145

/-- Two trains start at the same time from two stations and proceed towards each other.
    Train 1 travels at 20 km/hr.
    Train 2 travels at 25 km/hr.
    When they meet, Train 2 has traveled 55 km more than Train 1.
    Prove that the distance between the two stations is 495 km. -/
theorem distance_between_stations : ∃ x t : ℕ, 20 * t = x ∧ 25 * t = x + 55 ∧ 2 * x + 55 = 495 :=
by {
  sorry
}

end distance_between_stations_l2261_226145


namespace Harriet_age_now_l2261_226177

variable (P H: ℕ)

theorem Harriet_age_now (P : ℕ) (H : ℕ) (h1 : P + 4 = 2 * (H + 4)) (h2 : P = 60 / 2) : H = 13 := by
  sorry

end Harriet_age_now_l2261_226177


namespace fully_simplify_expression_l2261_226130

theorem fully_simplify_expression :
  (3 + 4 + 5 + 6) / 2 + (3 * 6 + 9) / 3 = 18 :=
by
  sorry

end fully_simplify_expression_l2261_226130


namespace cyclic_cosine_inequality_l2261_226167

theorem cyclic_cosine_inequality
  (α β γ : ℝ)
  (hα : 0 ≤ α ∧ α ≤ π / 2)
  (hβ : 0 ≤ β ∧ β ≤ π / 2)
  (hγ : 0 ≤ γ ∧ γ ≤ π / 2)
  (cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  2 ≤ (1 + Real.cos α ^ 2) ^ 2 * (Real.sin α) ^ 4
       + (1 + Real.cos β ^ 2) ^ 2 * (Real.sin β) ^ 4
       + (1 + Real.cos γ ^ 2) ^ 2 * (Real.sin γ) ^ 4 ∧
    (1 + Real.cos α ^ 2) ^ 2 * (Real.sin α) ^ 4
       + (1 + Real.cos β ^ 2) ^ 2 * (Real.sin β) ^ 4
       + (1 + Real.cos γ ^ 2) ^ 2 * (Real.sin γ) ^ 4
      ≤ (1 + Real.cos α ^ 2) * (1 + Real.cos β ^ 2) * (1 + Real.cos γ ^ 2) :=
by 
  sorry

end cyclic_cosine_inequality_l2261_226167


namespace arithmetic_sequence_sum_thirty_l2261_226104

-- Definitions according to the conditions
def arithmetic_seq_sums (S : ℕ → ℤ) : Prop :=
  ∃ a d : ℤ, ∀ n : ℕ, S n = a + n * d

-- Main statement we need to prove
theorem arithmetic_sequence_sum_thirty (S : ℕ → ℤ)
  (h1 : S 10 = 10)
  (h2 : S 20 = 30)
  (h3 : arithmetic_seq_sums S) : 
  S 30 = 50 := 
sorry

end arithmetic_sequence_sum_thirty_l2261_226104


namespace exists_k_l2261_226124

-- Define P as a non-constant homogeneous polynomial with real coefficients
def homogeneous_polynomial (n : ℕ) (P : ℝ → ℝ → ℝ) :=
  ∀ (a b : ℝ), P (a * a) (b * b) = (a * a) ^ n * (b * b) ^ n

-- Define the main problem
theorem exists_k (P : ℝ → ℝ → ℝ) (hP : ∃ n : ℕ, homogeneous_polynomial n P)
  (h : ∀ t : ℝ, P (Real.sin t) (Real.cos t) = 1) :
  ∃ k : ℕ, ∀ x y : ℝ, P x y = (x^2 + y^2) ^ k :=
sorry

end exists_k_l2261_226124


namespace spot_area_l2261_226154

/-- Proving the area of the accessible region outside the doghouse -/
theorem spot_area
  (pentagon_side : ℝ)
  (rope_length : ℝ)
  (accessible_area : ℝ) 
  (h1 : pentagon_side = 1) 
  (h2 : rope_length = 3)
  (h3 : accessible_area = (37 * π) / 5) :
  accessible_area = (π * (rope_length^2) * (288 / 360)) + 2 * (π * (pentagon_side^2) * (36 / 360)) := 
  sorry

end spot_area_l2261_226154


namespace Rahul_batting_average_l2261_226118

theorem Rahul_batting_average 
  (A : ℕ) (current_matches : ℕ := 12) (new_matches : ℕ := 13) (scored_today : ℕ := 78) (new_average : ℕ := 54)
  (h1 : (A * current_matches + scored_today) = new_average * new_matches) : A = 52 := 
by
  sorry

end Rahul_batting_average_l2261_226118


namespace volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron_l2261_226180

theorem volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron (R r : ℝ) (h : r = R / 3) : 
  (4/3 * π * r^3) / (4/3 * π * R^3) = 1 / 27 :=
by
  sorry

end volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron_l2261_226180


namespace range_of_a_in_triangle_l2261_226178

open Real

noncomputable def law_of_sines_triangle (A B C : ℝ) (a b c : ℝ) :=
  sin A / a = sin B / b ∧ sin B / b = sin C / c

theorem range_of_a_in_triangle (b : ℝ) (B : ℝ) (a : ℝ) (h1 : b = 2) (h2 : B = pi / 4) (h3 : true) :
  2 < a ∧ a < 2 * sqrt 2 :=
by
  sorry

end range_of_a_in_triangle_l2261_226178


namespace mass_of_circle_is_one_l2261_226187

variable (x y z : ℝ)

theorem mass_of_circle_is_one (h1 : 3 * y = 2 * x)
                              (h2 : 2 * y = x + 1)
                              (h3 : 5 * z = x + y)
                              (h4 : true) : z = 1 :=
sorry

end mass_of_circle_is_one_l2261_226187


namespace shelves_needed_l2261_226197

variable (total_books : Nat) (books_taken : Nat) (books_per_shelf : Nat)

theorem shelves_needed (h1 : total_books = 14) 
                       (h2 : books_taken = 2) 
                       (h3 : books_per_shelf = 3) : 
    (total_books - books_taken) / books_per_shelf = 4 := by
  sorry

end shelves_needed_l2261_226197


namespace smallest_constant_l2261_226168

theorem smallest_constant (D : ℝ) :
  (∀ (x y : ℝ), x^2 + 2*y^2 + 5 ≥ D*(2*x + 3*y) + 4) → D ≤ Real.sqrt (8 / 17) :=
by
  intros
  sorry

end smallest_constant_l2261_226168


namespace solve_for_m_l2261_226144

theorem solve_for_m : 
  ∀ m : ℝ, (3 * (-2) + 5 = -2 - m) → m = -1 :=
by
  intros m h
  sorry

end solve_for_m_l2261_226144


namespace min_sum_a_b_l2261_226156

theorem min_sum_a_b {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b)
  (h₂ : 1/a + 9/b = 1) : a + b ≥ 16 := 
sorry

end min_sum_a_b_l2261_226156


namespace geometric_sequence_example_l2261_226185

theorem geometric_sequence_example
  (a : ℕ → ℝ)
  (h1 : ∀ n, 0 < a n)
  (h2 : ∃ r, ∀ n, a (n + 1) = r * a n)
  (h3 : Real.log (a 2) / Real.log 2 + Real.log (a 8) / Real.log 2 = 1) :
  a 3 * a 7 = 2 :=
sorry

end geometric_sequence_example_l2261_226185


namespace solve_for_x_l2261_226126

theorem solve_for_x (x y : ℝ) (h1 : 3 * x + y = 75) (h2 : 2 * (3 * x + y) - y = 138) : x = 21 :=
  sorry

end solve_for_x_l2261_226126


namespace simplify_expression_l2261_226135

theorem simplify_expression (x : ℝ) : 8 * x + 15 - 3 * x + 27 = 5 * x + 42 := 
by
  sorry

end simplify_expression_l2261_226135


namespace points_four_units_away_l2261_226190

theorem points_four_units_away (x : ℚ) (h : |x| = 4) : x = -4 ∨ x = 4 := 
by 
  sorry

end points_four_units_away_l2261_226190


namespace scientific_notation_560000_l2261_226136

theorem scientific_notation_560000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 560000 = a * 10 ^ n ∧ a = 5.6 ∧ n = 5 :=
by 
  sorry

end scientific_notation_560000_l2261_226136


namespace students_not_enrolled_in_biology_class_l2261_226161

theorem students_not_enrolled_in_biology_class (total_students : ℕ) (percent_biology : ℕ) 
  (h1 : total_students = 880) (h2 : percent_biology = 35) : 
  total_students - (percent_biology * total_students / 100) = 572 := by
  sorry

end students_not_enrolled_in_biology_class_l2261_226161


namespace area_tripled_radius_increase_l2261_226194

theorem area_tripled_radius_increase (m r : ℝ) (h : (r + m)^2 = 3 * r^2) :
  r = m * (1 + Real.sqrt 3) / 2 :=
sorry

end area_tripled_radius_increase_l2261_226194


namespace solve_for_t_l2261_226109

theorem solve_for_t : ∃ t : ℝ, 3 * 3^t + Real.sqrt (9 * 9^t) = 18 ∧ t = 1 :=
by
  sorry

end solve_for_t_l2261_226109


namespace find_sum_of_squares_of_roots_l2261_226128

theorem find_sum_of_squares_of_roots:
  ∀ (a b c d : ℝ), (a^2 * b^2 * c^2 * d^2 - 15 * a * b * c * d + 56 = 0) → 
  a^2 + b^2 + c^2 + d^2 = 30 := by
  intros a b c d h
  sorry

end find_sum_of_squares_of_roots_l2261_226128


namespace statement_correctness_l2261_226175

def correct_statements := [4, 8]
def incorrect_statements := [1, 2, 3, 5, 6, 7]

theorem statement_correctness :
  correct_statements = [4, 8] ∧ incorrect_statements = [1, 2, 3, 5, 6, 7] :=
  by sorry

end statement_correctness_l2261_226175


namespace multiply_exponents_l2261_226119

theorem multiply_exponents (a : ℝ) : 2 * a^3 * 3 * a^2 = 6 * a^5 := by
  sorry

end multiply_exponents_l2261_226119


namespace percentage_of_hexagon_area_is_closest_to_17_l2261_226152

noncomputable def tiling_area_hexagon_percentage : Real :=
  let total_area := 2 * 3
  let square_area := 1 * 1 
  let squares_count := 5 -- Adjusted count from 8 to fit total area properly
  let square_total_area := squares_count * square_area
  let hexagon_area := total_area - square_total_area
  let percentage := (hexagon_area / total_area) * 100
  percentage

theorem percentage_of_hexagon_area_is_closest_to_17 :
  abs (tiling_area_hexagon_percentage - 17) < 1 :=
sorry

end percentage_of_hexagon_area_is_closest_to_17_l2261_226152


namespace freshmen_count_l2261_226165

theorem freshmen_count (n : ℕ) (h1 : n < 600) (h2 : n % 17 = 16) (h3 : n % 19 = 18) : n = 322 := 
by 
  sorry

end freshmen_count_l2261_226165


namespace base_six_digits_unique_l2261_226146

theorem base_six_digits_unique (b : ℕ) (h : (b-1)^2*(b-2) = 100) : b = 6 :=
by
  sorry

end base_six_digits_unique_l2261_226146


namespace solution_set_inequality_l2261_226163

theorem solution_set_inequality (x : ℝ) : 4 * x < 3 * x + 2 → x < 2 :=
by
  intro h
  -- Add actual proof here, but for now; we use sorry
  sorry

end solution_set_inequality_l2261_226163


namespace walking_times_relationship_l2261_226169

theorem walking_times_relationship (x : ℝ) (h : x > 0) :
  (15 / x) - (15 / (x + 1)) = 1 / 2 :=
sorry

end walking_times_relationship_l2261_226169


namespace divisibility_by_3_divisibility_by_4_l2261_226170

-- Proof that 5n^2 + 10n + 8 is divisible by 3 if and only if n ≡ 2 (mod 3)
theorem divisibility_by_3 (n : ℤ) : (5 * n^2 + 10 * n + 8) % 3 = 0 ↔ n % 3 = 2 := 
    sorry

-- Proof that 5n^2 + 10n + 8 is divisible by 4 if and only if n ≡ 0 (mod 2)
theorem divisibility_by_4 (n : ℤ) : (5 * n^2 + 10 * n + 8) % 4 = 0 ↔ n % 2 = 0 :=
    sorry

end divisibility_by_3_divisibility_by_4_l2261_226170


namespace max_students_l2261_226138

-- Defining the problem's conditions
def cost_bus_rental : ℕ := 100
def max_capacity_students : ℕ := 25
def cost_per_student : ℕ := 10
def teacher_admission_cost : ℕ := 0
def total_budget : ℕ := 350

-- The Lean proof problem
theorem max_students (bus_cost : ℕ) (student_capacity : ℕ) (student_cost : ℕ) (teacher_cost : ℕ) (budget : ℕ) :
  bus_cost = cost_bus_rental → 
  student_capacity = max_capacity_students →
  student_cost = cost_per_student →
  teacher_cost = teacher_admission_cost →
  budget = total_budget →
  (student_capacity ≤ (budget - bus_cost) / student_cost) → 
  ∃ n : ℕ, n = student_capacity ∧ n ≤ (budget - bus_cost) / student_cost :=
by
  intros
  sorry

end max_students_l2261_226138


namespace new_person_weight_l2261_226186

noncomputable def weight_increase (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase

theorem new_person_weight 
  (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) 
  (weight_eqn : weight_increase n avg_increase = new_weight - old_weight) : 
  new_weight = 87.5 :=
by
  have n := 9
  have avg_increase := 2.5
  have old_weight := 65
  have weight_increase := 9 * 2.5
  have weight_eqn := weight_increase = 87.5 - 65
  sorry

end new_person_weight_l2261_226186


namespace additional_cost_per_kg_l2261_226157

theorem additional_cost_per_kg (l a : ℝ) 
  (h1 : 30 * l + 3 * a = 333) 
  (h2 : 30 * l + 6 * a = 366) 
  (h3 : 15 * l = 150) 
  : a = 11 := 
by
  sorry

end additional_cost_per_kg_l2261_226157


namespace michael_peach_pies_l2261_226134

/--
Michael ran a bakeshop and had to fill an order for some peach pies, 4 apple pies and 3 blueberry pies.
Each pie recipe called for 3 pounds of fruit each. At the market, produce was on sale for $1.00 per pound for both blueberries and apples.
The peaches each cost $2.00 per pound. Michael spent $51 at the market buying the fruit for his pie order.
Prove that Michael had to make 5 peach pies.
-/
theorem michael_peach_pies :
  let apple_pies := 4
  let blueberry_pies := 3
  let peach_pie_cost_per_pound := 2
  let apple_blueberry_cost_per_pound := 1
  let pounds_per_pie := 3
  let total_spent := 51
  (total_spent - ((apple_pies + blueberry_pies) * pounds_per_pie * apple_blueberry_cost_per_pound)) 
  / (pounds_per_pie * peach_pie_cost_per_pound) = 5 :=
by
  let apple_pies := 4
  let blueberry_pies := 3
  let peach_pie_cost_per_pound := 2
  let apple_blueberry_cost_per_pound := 1
  let pounds_per_pie := 3
  let total_spent := 51
  have H1 : (total_spent - ((apple_pies + blueberry_pies) * pounds_per_pie * apple_blueberry_cost_per_pound)) 
             / (pounds_per_pie * peach_pie_cost_per_pound) = 5 := sorry
  exact H1

end michael_peach_pies_l2261_226134


namespace kids_played_on_monday_l2261_226193

theorem kids_played_on_monday (m t a : Nat) (h1 : t = 7) (h2 : a = 19) (h3 : a = m + t) : m = 12 := 
by 
  sorry

end kids_played_on_monday_l2261_226193


namespace update_year_l2261_226184

def a (n : ℕ) : ℕ :=
  if n ≤ 7 then 2 * n + 2 else 16 * (5 / 4) ^ (n - 7)

noncomputable def S (n : ℕ) : ℕ :=
  if n ≤ 7 then n^2 + 3 * n else 80 * ((5 / 4) ^ (n - 7)) - 10

noncomputable def avg_maintenance_cost (n : ℕ) : ℚ :=
  (S n : ℚ) / n

theorem update_year (n : ℕ) (h : avg_maintenance_cost n > 12) : n = 9 :=
  by
  sorry

end update_year_l2261_226184


namespace range_of_a_l2261_226137

def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2) : a ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
by
  sorry

end range_of_a_l2261_226137


namespace arithmetic_mean_of_first_40_consecutive_integers_l2261_226113

-- Define the arithmetic sequence with the given conditions
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the given arithmetic sequence
def arithmetic_sum (a₁ d n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Define the arithmetic mean of the first n terms of the given arithmetic sequence
def arithmetic_mean (a₁ d n : ℕ) : ℚ :=
  (arithmetic_sum a₁ d n : ℚ) / n

-- The arithmetic sequence starts at 5, has a common difference of 1, and has 40 terms
theorem arithmetic_mean_of_first_40_consecutive_integers :
  arithmetic_mean 5 1 40 = 24.5 :=
by
  sorry

end arithmetic_mean_of_first_40_consecutive_integers_l2261_226113


namespace root_exists_in_interval_l2261_226122

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

theorem root_exists_in_interval :
  ∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0 := 
sorry

end root_exists_in_interval_l2261_226122


namespace product_base9_conversion_l2261_226114

noncomputable def base_9_to_base_10 (n : ℕ) : ℕ :=
match n with
| 237 => 2 * 9^2 + 3 * 9^1 + 7
| 17 => 9 + 7
| _ => 0

noncomputable def base_10_to_base_9 (n : ℕ) : ℕ :=
match n with
-- Step of conversion from example: 3136 => 4*9^3 + 2*9^2 + 6*9^1 + 4*9^0
| 3136 => 4 * 1000 + 2 * 100 + 6 * 10 + 4 -- representing 4264 in base 9
| _ => 0

theorem product_base9_conversion :
  base_10_to_base_9 ((base_9_to_base_10 237) * (base_9_to_base_10 17)) = 4264 := by
  sorry

end product_base9_conversion_l2261_226114


namespace gcf_75_135_l2261_226121

theorem gcf_75_135 : Nat.gcd 75 135 = 15 :=
  by sorry

end gcf_75_135_l2261_226121


namespace average_of_11_results_l2261_226107

theorem average_of_11_results (a b c : ℝ) (avg_first_6 avg_last_6 sixth_result avg_all_11 : ℝ)
  (h1 : avg_first_6 = 58)
  (h2 : avg_last_6 = 63)
  (h3 : sixth_result = 66) :
  avg_all_11 = 60 :=
by
  sorry

end average_of_11_results_l2261_226107


namespace second_person_time_l2261_226192

theorem second_person_time (x : ℝ) (h1 : ∀ t : ℝ, t = 3) 
(h2 : (1/3 + 1/x) = 5/12) : x = 12 := 
by sorry

end second_person_time_l2261_226192


namespace sum_of_digits_l2261_226115

theorem sum_of_digits (a b : ℕ) (h1 : 4 * 100 + a * 10 + 3 + 984 = 1 * 1000 + 3 * 100 + b * 10 + 7)
  (h2 : (1 + b) - (3 + 7) % 11 = 0) : a + b = 10 := 
by
  sorry

end sum_of_digits_l2261_226115


namespace focaccia_cost_l2261_226117

theorem focaccia_cost :
  let almond_croissant := 4.50
  let salami_cheese_croissant := 4.50
  let plain_croissant := 3.00
  let latte := 2.50
  let total_spent := 21.00
  let known_costs := almond_croissant + salami_cheese_croissant + plain_croissant + 2 * latte
  let focaccia_cost := total_spent - known_costs
  focaccia_cost = 4.00 := 
by
  sorry

end focaccia_cost_l2261_226117


namespace wilson_total_cost_l2261_226123

noncomputable def total_cost_wilson_pays : ℝ :=
let hamburger_price : ℝ := 5
let cola_price : ℝ := 2
let fries_price : ℝ := 3
let sundae_price : ℝ := 4
let nugget_price : ℝ := 1.5
let salad_price : ℝ := 6.25
let hamburger_count : ℕ := 2
let cola_count : ℕ := 3
let nugget_count : ℕ := 4

let total_before_discounts := (hamburger_count * hamburger_price) +
                              (cola_count * cola_price) +
                              fries_price +
                              sundae_price +
                              (nugget_count * nugget_price) +
                              salad_price

let free_nugget_discount := 1 * nugget_price
let total_after_promotion := total_before_discounts - free_nugget_discount
let coupon_discount := 4
let total_after_coupon := total_after_promotion - coupon_discount
let loyalty_discount := 0.10 * total_after_coupon
let total_after_loyalty := total_after_coupon - loyalty_discount

total_after_loyalty

theorem wilson_total_cost : total_cost_wilson_pays = 26.77 := 
by
  sorry

end wilson_total_cost_l2261_226123


namespace laundry_loads_l2261_226189

-- Conditions
def wash_time_per_load : ℕ := 45 -- in minutes
def dry_time_per_load : ℕ := 60 -- in minutes
def total_time : ℕ := 14 -- in hours

theorem laundry_loads (L : ℕ) 
  (h1 : total_time = 14)
  (h2 : total_time * 60 = L * (wash_time_per_load + dry_time_per_load)) :
  L = 8 :=
by
  sorry

end laundry_loads_l2261_226189


namespace max_abs_sum_on_ellipse_l2261_226103

theorem max_abs_sum_on_ellipse :
  ∀ (x y : ℝ), 4 * x^2 + y^2 = 4 -> |x| + |y| ≤ (3 * Real.sqrt 2) / Real.sqrt 5 :=
by
  intro x y h
  sorry

end max_abs_sum_on_ellipse_l2261_226103


namespace count_integers_between_3250_and_3500_with_increasing_digits_l2261_226179

theorem count_integers_between_3250_and_3500_with_increasing_digits :
  ∃ n : ℕ, n = 20 ∧
    (∀ x : ℕ, 3250 ≤ x ∧ x ≤ 3500 →
      ∀ (d1 d2 d3 d4 : ℕ),
        d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧
        (x = d1 * 1000 + d2 * 100 + d3 * 10 + d4) →
        (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4)) :=
  sorry

end count_integers_between_3250_and_3500_with_increasing_digits_l2261_226179


namespace solve_nat_eqn_l2261_226173

theorem solve_nat_eqn (n k l m : ℕ) (hl : l > 1) 
  (h_eq : (1 + n^k)^l = 1 + n^m) : (n, k, l, m) = (2, 1, 2, 3) := 
sorry

end solve_nat_eqn_l2261_226173


namespace original_number_l2261_226198

theorem original_number (y : ℚ) (h : 1 - (1 / y) = 5 / 4) : y = -4 :=
sorry

end original_number_l2261_226198


namespace squares_expression_l2261_226141

theorem squares_expression (a : ℕ) : 
  a^2 + 5*a + 7 = (a+3) * (a+2)^2 + (a+2) * 1^2 := 
by
  sorry

end squares_expression_l2261_226141


namespace max_product_of_sum_300_l2261_226100

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l2261_226100


namespace find_fraction_l2261_226158

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h1 : (a + b + c) / (a + b - c) = 7)
variable (h2 : (a + b + c) / (a + c - b) = 1.75)

theorem find_fraction : (a + b + c) / (b + c - a) = 3.5 := 
by {
  sorry
}

end find_fraction_l2261_226158


namespace common_chord_length_l2261_226147

theorem common_chord_length (x y : ℝ) : 
    (x^2 + y^2 = 4) → 
    (x^2 + y^2 - 4*x + 4*y - 12 = 0) → 
    ∃ l : ℝ, l = 2 * Real.sqrt 2 :=
by
  intros h1 h2
  sorry

end common_chord_length_l2261_226147


namespace hudson_daily_burger_spending_l2261_226148

-- Definitions based on conditions
def total_spent := 465
def days_in_december := 31

-- Definition of the question
def amount_spent_per_day := total_spent / days_in_december

-- The theorem to prove
theorem hudson_daily_burger_spending : amount_spent_per_day = 15 := by
  sorry

end hudson_daily_burger_spending_l2261_226148


namespace no_perfect_square_E_l2261_226102

noncomputable def E (x : ℝ) : ℤ :=
  round x

theorem no_perfect_square_E (n : ℕ) (h : n > 0) : ¬ (∃ k : ℕ, E (n + Real.sqrt n) = k * k) :=
  sorry

end no_perfect_square_E_l2261_226102


namespace scientific_notation_correct_l2261_226172

noncomputable def scientific_notation (x : ℝ) : ℝ × ℤ :=
  let a := x * 10^9
  (a, -9)

theorem scientific_notation_correct :
  scientific_notation 0.000000007 = (7, -9) :=
by
  sorry

end scientific_notation_correct_l2261_226172


namespace number_of_routes_from_A_to_B_l2261_226160

-- Define the grid dimensions
def grid_rows : ℕ := 3
def grid_columns : ℕ := 2

-- Define the total number of steps needed to travel from A to B
def total_steps : ℕ := grid_rows + grid_columns

-- Define the number of right moves (R) and down moves (D)
def right_moves : ℕ := grid_rows
def down_moves : ℕ := grid_columns

-- Calculate the number of different routes using combination formula
def number_of_routes : ℕ := Nat.choose total_steps right_moves

-- The main statement to be proven
theorem number_of_routes_from_A_to_B : number_of_routes = 10 :=
by sorry

end number_of_routes_from_A_to_B_l2261_226160


namespace not_sum_of_squares_of_form_4m_plus_3_l2261_226171

theorem not_sum_of_squares_of_form_4m_plus_3 (n m : ℤ) (h : n = 4 * m + 3) : 
  ¬ ∃ a b : ℤ, n = a^2 + b^2 :=
by
  sorry

end not_sum_of_squares_of_form_4m_plus_3_l2261_226171


namespace hare_wins_by_10_meters_l2261_226188

def speed_tortoise := 3 -- meters per minute
def speed_hare_sprint := 12 -- meters per minute
def speed_hare_walk := 1 -- meters per minute
def time_total := 50 -- minutes
def time_hare_sprint := 10 -- minutes
def time_hare_walk := time_total - time_hare_sprint -- minutes

def distance_tortoise := speed_tortoise * time_total -- meters
def distance_hare := (speed_hare_sprint * time_hare_sprint) + (speed_hare_walk * time_hare_walk) -- meters

theorem hare_wins_by_10_meters : (distance_hare - distance_tortoise) = 10 := by
  -- Proof would go here
  sorry

end hare_wins_by_10_meters_l2261_226188


namespace cos_value_l2261_226131

theorem cos_value (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) : Real.cos (π / 3 - α) = 1 / 3 :=
  sorry

end cos_value_l2261_226131


namespace other_discount_l2261_226125

theorem other_discount (list_price : ℝ) (final_price : ℝ) (first_discount : ℝ) (other_discount : ℝ) :
  list_price = 70 → final_price = 61.74 → first_discount = 10 → (list_price * (1 - first_discount / 100) * (1 - other_discount / 100) = final_price) → other_discount = 2 := 
by
  intros h1 h2 h3 h4
  sorry

end other_discount_l2261_226125


namespace a5_value_l2261_226111

-- Definitions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n, a (n + 1) = q * a n

def positive_terms (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

def product_condition (a : ℕ → ℝ) : Prop :=
  ∀ n, a n * a (n + 1) = 2^(2 * n + 1)

-- Theorem statement
theorem a5_value (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_pos : positive_terms a) (h_prod : product_condition a) : a 5 = 32 :=
sorry

end a5_value_l2261_226111


namespace train_speed_is_correct_l2261_226106

-- Definitions of the problem
def length_of_train : ℕ := 360
def time_to_pass_bridge : ℕ := 25
def length_of_bridge : ℕ := 140
def conversion_factor : ℝ := 3.6

-- Distance covered by the train plus the length of the bridge
def total_distance : ℕ := length_of_train + length_of_bridge

-- Speed calculation in m/s
def speed_in_m_per_s := total_distance / time_to_pass_bridge

-- Conversion to km/h
def speed_in_km_per_h := speed_in_m_per_s * conversion_factor

-- The proof goal: the speed of the train is 72 km/h
theorem train_speed_is_correct : speed_in_km_per_h = 72 := by
  sorry

end train_speed_is_correct_l2261_226106


namespace correct_equation_l2261_226142

variable (x : ℤ)
variable (cost_of_chickens : ℤ)

-- Condition 1: If each person contributes 9 coins, there will be an excess of 11 coins.
def condition1 : Prop := 9 * x - cost_of_chickens = 11

-- Condition 2: If each person contributes 6 coins, there will be a shortage of 16 coins.
def condition2 : Prop := 6 * x - cost_of_chickens = -16

-- The goal is to prove the correct equation given the conditions.
theorem correct_equation (h1 : condition1 (x) (cost_of_chickens)) (h2 : condition2 (x) (cost_of_chickens)) :
  9 * x - 11 = 6 * x + 16 :=
sorry

end correct_equation_l2261_226142

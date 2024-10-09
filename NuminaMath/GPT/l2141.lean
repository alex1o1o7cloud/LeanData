import Mathlib

namespace evaluate_expression_l2141_214126

theorem evaluate_expression:
  let a := 3
  let b := 2
  (a^b)^a - (b^a)^b = 665 :=
by
  sorry

end evaluate_expression_l2141_214126


namespace stuffed_animal_cost_l2141_214196

theorem stuffed_animal_cost
  (M S A C : ℝ)
  (h1 : M = 3 * S)
  (h2 : M = (1/2) * A)
  (h3 : C = (1/2) * A)
  (h4 : C = 2 * S)
  (h5 : M = 6) :
  A = 8 :=
by
  sorry

end stuffed_animal_cost_l2141_214196


namespace inequality_proof_l2141_214114

theorem inequality_proof (a b c d : ℝ) : 
  (a + b + c + d) * (a * b * (c + d) + (a + b) * c * d) - a * b * c * d ≤ 
  (1 / 2) * (a * (b + d) + b * (c + d) + c * (d + a))^2 :=
by
  sorry

end inequality_proof_l2141_214114


namespace line_does_not_pass_second_quadrant_l2141_214151

theorem line_does_not_pass_second_quadrant 
  (A B C x y : ℝ) 
  (h1 : A * C < 0) 
  (h2 : B * C > 0) 
  (h3 : A * x + B * y + C = 0) :
  ¬ (x < 0 ∧ y > 0) := 
sorry

end line_does_not_pass_second_quadrant_l2141_214151


namespace scientific_notation_of_sesame_mass_l2141_214138

theorem scientific_notation_of_sesame_mass :
  0.00000201 = 2.01 * 10^(-6) :=
sorry

end scientific_notation_of_sesame_mass_l2141_214138


namespace minimum_chocolates_l2141_214125

theorem minimum_chocolates (x : ℤ) (h1 : x ≥ 150) (h2 : x % 15 = 7) : x = 157 :=
sorry

end minimum_chocolates_l2141_214125


namespace positive_difference_l2141_214185

theorem positive_difference (a b : ℕ) (h1 : a = (6^2 + 6^2) / 6) (h2 : b = (6^2 * 6^2) / 6) : a < b ∧ b - a = 204 :=
by
  sorry

end positive_difference_l2141_214185


namespace remainder_of_power_mod_l2141_214143

theorem remainder_of_power_mod 
  (n : ℕ)
  (h₁ : 7 ≡ 1 [MOD 6]) : 7^51 ≡ 1 [MOD 6] := 
sorry

end remainder_of_power_mod_l2141_214143


namespace polynomial_division_l2141_214172

noncomputable def poly1 : Polynomial ℤ := Polynomial.X ^ 13 - Polynomial.X + 100
noncomputable def poly2 : Polynomial ℤ := Polynomial.X ^ 2 + Polynomial.X + 2

theorem polynomial_division : ∃ q : Polynomial ℤ, poly1 = poly2 * q :=
by 
  sorry

end polynomial_division_l2141_214172


namespace jason_earnings_l2141_214178

theorem jason_earnings :
  let fred_initial := 49
  let jason_initial := 3
  let emily_initial := 25
  let fred_increase := 1.5 
  let jason_increase := 0.625 
  let emily_increase := 0.40 
  let fred_new := fred_initial * fred_increase
  let jason_new := jason_initial * (1 + jason_increase)
  let emily_new := emily_initial * (1 + emily_increase)
  fred_new = fred_initial * fred_increase ->
  jason_new = jason_initial * (1 + jason_increase) ->
  emily_new = emily_initial * (1 + emily_increase) ->
  jason_new - jason_initial == 1.875 :=
by
  intros
  sorry

end jason_earnings_l2141_214178


namespace find_a_l2141_214130

theorem find_a (a : ℝ) : 
  (∃ x y : ℝ, x - 2 * a * y - 3 = 0 ∧ x^2 + y^2 - 2 * x + 2 * y - 3 = 0) → a = 1 :=
by
  sorry

end find_a_l2141_214130


namespace P_not_77_for_all_integers_l2141_214197

def P (x y : ℤ) : ℤ := x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_not_77_for_all_integers (x y : ℤ) : P x y ≠ 77 :=
sorry

end P_not_77_for_all_integers_l2141_214197


namespace rectangle_perimeter_126_l2141_214186

/-- Define the sides of the rectangle in terms of a common multiplier -/
def sides (x : ℝ) : ℝ × ℝ := (4 * x, 3 * x)

/-- Define the area of the rectangle given the common multiplier -/
def area (x : ℝ) : ℝ := (4 * x) * (3 * x)

example : ∃ (x : ℝ), area x = 972 :=
by
  sorry

/-- Calculate the perimeter of the rectangle given the common multiplier -/
def perimeter (x : ℝ) : ℝ := 2 * ((4 * x) + (3 * x))

/-- The final proof statement, stating that the perimeter of the rectangle is 126 meters,
    given the ratio of its sides and its area. -/
theorem rectangle_perimeter_126 (x : ℝ) (h: area x = 972) : perimeter x = 126 :=
by
  sorry

end rectangle_perimeter_126_l2141_214186


namespace mean_of_data_is_5_l2141_214161

theorem mean_of_data_is_5 (h : s^2 = (1 / 4) * ((3.2 - x)^2 + (5.7 - x)^2 + (4.3 - x)^2 + (6.8 - x)^2))
  : x = 5 := 
sorry

end mean_of_data_is_5_l2141_214161


namespace least_number_of_square_tiles_l2141_214159

-- Definitions based on conditions
def room_length_cm : ℕ := 672
def room_width_cm : ℕ := 432

-- Correct Answer is 126 tiles

-- Lean Statement for the proof problem
theorem least_number_of_square_tiles : 
  ∃ tile_size tiles_needed, 
    (tile_size = Int.gcd room_length_cm room_width_cm) ∧
    (tiles_needed = (room_length_cm / tile_size) * (room_width_cm / tile_size)) ∧
    tiles_needed = 126 := 
by
  sorry

end least_number_of_square_tiles_l2141_214159


namespace count_young_diagrams_4_count_young_diagrams_5_count_young_diagrams_6_count_young_diagrams_7_l2141_214160

-- Define the weight 's'.
variable (s : ℕ)

-- Define the function that counts the number of Young diagrams for a given weight.
def countYoungDiagrams (s : ℕ) : ℕ :=
  -- Placeholder for actual implementation of counting Young diagrams.
  sorry

-- Prove that the count of Young diagrams for s = 4 is 5
theorem count_young_diagrams_4 : countYoungDiagrams 4 = 5 :=
by sorry

-- Prove that the count of Young diagrams for s = 5 is 7
theorem count_young_diagrams_5 : countYoungDiagrams 5 = 7 :=
by sorry

-- Prove that the count of Young diagrams for s = 6 is 11
theorem count_young_diagrams_6 : countYoungDiagrams 6 = 11 :=
by sorry

-- Prove that the count of Young diagrams for s = 7 is 15
theorem count_young_diagrams_7 : countYoungDiagrams 7 = 15 :=
by sorry

end count_young_diagrams_4_count_young_diagrams_5_count_young_diagrams_6_count_young_diagrams_7_l2141_214160


namespace complement_union_l2141_214199

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {4, 5}
def B : Set ℕ := {3, 4}

theorem complement_union : (U \ (A ∪ B)) = {1, 2, 6} :=
by simp only [U, A, B, Set.mem_union, Set.mem_compl, Set.mem_diff];
   sorry

end complement_union_l2141_214199


namespace parametric_to_standard_line_parametric_to_standard_ellipse_l2141_214140

theorem parametric_to_standard_line (t : ℝ) (x y : ℝ) 
  (h₁ : x = 1 - 3 * t)
  (h₂ : y = 4 * t) :
  4 * x + 3 * y - 4 = 0 := by
sorry

theorem parametric_to_standard_ellipse (θ x y : ℝ) 
  (h₁ : x = 5 * Real.cos θ)
  (h₂ : y = 4 * Real.sin θ) :
  (x^2 / 25) + (y^2 / 16) = 1 := by
sorry

end parametric_to_standard_line_parametric_to_standard_ellipse_l2141_214140


namespace tangent_line_at_zero_range_of_a_l2141_214128

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * Real.sin x - 1

theorem tangent_line_at_zero (h : ∀ x, f 1 x = Real.exp x - Real.sin x - 1) :
  ∀ x, Real.exp x - Real.sin x - 1 = f 1 x :=
by
  sorry

theorem range_of_a (h : ∀ x, f a x ≥ 0) : a ∈ Set.Iic 1 :=
by
  sorry

end tangent_line_at_zero_range_of_a_l2141_214128


namespace product_of_two_numbers_l2141_214142

theorem product_of_two_numbers (x y : ℕ) (h₁ : x + y = 40) (h₂ : x - y = 16) : x * y = 336 :=
sorry

end product_of_two_numbers_l2141_214142


namespace train_length_l2141_214134

-- Definitions of speeds and times
def speed_person_A := 5 / 3.6 -- in meters per second
def speed_person_B := 15 / 3.6 -- in meters per second
def time_to_overtake_A := 36 -- in seconds
def time_to_overtake_B := 45 -- in seconds

-- The length of the train
theorem train_length :
  ∃ x : ℝ, x = 500 :=
by
  sorry

end train_length_l2141_214134


namespace no_two_primes_sum_to_53_l2141_214122

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_two_primes_sum_to_53 :
  ¬ ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_two_primes_sum_to_53_l2141_214122


namespace min_value_l2141_214173

noncomputable def conditions (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧ 
  (27^x + y^4 - 3^x - 1 = 0)

theorem min_value (x y : ℝ) (h : conditions x y) : ∃ x y, (x^3 + y^3 = -1) :=
sorry

end min_value_l2141_214173


namespace length_AB_given_conditions_l2141_214171

variable {A B P Q : Type} [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField P] [LinearOrderedField Q]

def length_of_AB (x y : A) : A := x + y

theorem length_AB_given_conditions (x y u v : A) (hx : y = 4 * x) (hv : 5 * u = 2 * v) (hu : u = x + 3) (hv' : v = y - 3) (hPQ : PQ = 3) : length_of_AB x y = 35 :=
by
  sorry

end length_AB_given_conditions_l2141_214171


namespace chemistry_marks_more_than_physics_l2141_214164

theorem chemistry_marks_more_than_physics (M P C x : ℕ) 
  (h1 : M + P = 32) 
  (h2 : (M + C) / 2 = 26) 
  (h3 : C = P + x) : 
  x = 20 := 
by
  sorry

end chemistry_marks_more_than_physics_l2141_214164


namespace height_of_spherical_caps_l2141_214156

theorem height_of_spherical_caps
  (r q : ℝ)
  (m₁ m₂ m₃ m₄ : ℝ)
  (h1 : m₂ = m₁ * q)
  (h2 : m₃ = m₁ * q^2)
  (h3 : m₄ = m₁ * q^3)
  (h4 : m₁ + m₂ + m₃ + m₄ = 2 * r) :
  m₁ = 2 * r * (q - 1) / (q^4 - 1) := 
sorry

end height_of_spherical_caps_l2141_214156


namespace zander_stickers_l2141_214146

theorem zander_stickers (S : ℕ) (h1 : 44 = (11 / 25) * S) : S = 100 :=
by
  sorry

end zander_stickers_l2141_214146


namespace more_knights_than_liars_l2141_214180

theorem more_knights_than_liars 
  (k l : Nat)
  (h1 : (k + l) % 2 = 1)
  (h2 : ∀ i : Nat, i < k → ∃ j : Nat, j < l)
  (h3 : ∀ j : Nat, j < l → ∃ i : Nat, i < k) :
  k > l := 
sorry

end more_knights_than_liars_l2141_214180


namespace find_Sum_4n_l2141_214177

variable {a : ℕ → ℕ} -- Define a sequence a_n

-- Define our conditions about the sums Sn and S3n
axiom Sum_n : ℕ → ℕ 
axiom Sum_3n : ℕ → ℕ 
axiom Sum_4n : ℕ → ℕ 

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def arithmetic_sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a n + a 0)) / 2

axiom h1 : is_arithmetic_sequence a
axiom h2 : Sum_n 1 = 2
axiom h3 : Sum_3n 3 = 12

theorem find_Sum_4n : Sum_4n 4 = 20 :=
sorry

end find_Sum_4n_l2141_214177


namespace triangle_side_range_l2141_214115

theorem triangle_side_range (AB AC x : ℝ) (hAB : AB = 16) (hAC : AC = 7) (hBC : BC = x) :
  9 < x ∧ x < 23 :=
by
  sorry

end triangle_side_range_l2141_214115


namespace polynomial_division_l2141_214145

variable (a p x : ℝ)

theorem polynomial_division :
  (p^8 * x^4 - 81 * a^12) / (p^6 * x^3 - 3 * a^3 * p^4 * x^2 + 9 * a^6 * p^2 * x - 27 * a^9) = p^2 * x + 3 * a^3 :=
by sorry

end polynomial_division_l2141_214145


namespace tangent_lines_inequality_l2141_214166

theorem tangent_lines_inequality (k k1 k2 b b1 b2 : ℝ)
  (h1 : k = - (b * b) / 4)
  (h2 : k1 = - (b1 * b1) / 4)
  (h3 : k2 = - (b2 * b2) / 4)
  (h4 : b = b1 + b2) :
  k ≥ 2 * (k1 + k2) := sorry

end tangent_lines_inequality_l2141_214166


namespace f_neg_one_l2141_214194

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 1/x else - (x^2 + 1/(-x))

theorem f_neg_one : f (-1) = -2 :=
by
  -- This is where the proof would go, but it is left as a sorry
  sorry

end f_neg_one_l2141_214194


namespace lcm_24_150_is_600_l2141_214103

noncomputable def lcm_24_150 : ℕ :=
  let a := 24
  let b := 150
  have h₁ : a = 2^3 * 3 := by sorry
  have h₂ : b = 2 * 3 * 5^2 := by sorry
  Nat.lcm a b

theorem lcm_24_150_is_600 : lcm_24_150 = 600 := by
  -- Use provided primes conditions to derive the result
  sorry

end lcm_24_150_is_600_l2141_214103


namespace middle_number_is_45_l2141_214113

open Real

noncomputable def middle_number (l : List ℝ) (h_len : l.length = 13) 
  (h1 : (l.sum / 13) = 9) 
  (h2 : (l.take 6).sum = 30) 
  (h3 : (l.drop 7).sum = 42): ℝ := 
  l.nthLe 6 sorry  -- middle element (index 6 in 0-based index)

theorem middle_number_is_45 (l : List ℝ) (h_len : l.length = 13) 
  (h1 : (l.sum / 13) = 9) 
  (h2 : (l.take 6).sum = 30) 
  (h3 : (l.drop 7).sum = 42) : 
  middle_number l h_len h1 h2 h3 = 45 := 
sorry

end middle_number_is_45_l2141_214113


namespace find_certain_number_l2141_214124

theorem find_certain_number (x : ℤ) (h : x + 34 - 53 = 28) : x = 47 :=
by {
  sorry
}

end find_certain_number_l2141_214124


namespace range_of_a_l2141_214192

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 + a * x + a > 0) : 0 < a ∧ a < 4 :=
sorry

end range_of_a_l2141_214192


namespace compute_xy_l2141_214112

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = -9 := 
by 
  sorry

end compute_xy_l2141_214112


namespace minimum_n_value_l2141_214150

theorem minimum_n_value : ∃ n : ℕ, n > 0 ∧ ∀ r : ℕ, (2 * n = 5 * r) → n = 5 :=
by
  sorry

end minimum_n_value_l2141_214150


namespace incorrect_quotient_l2141_214102

theorem incorrect_quotient
    (correct_quotient : ℕ)
    (correct_divisor : ℕ)
    (incorrect_divisor : ℕ)
    (h1 : correct_quotient = 28)
    (h2 : correct_divisor = 21)
    (h3 : incorrect_divisor = 12) :
  correct_divisor * correct_quotient / incorrect_divisor = 49 :=
by
  sorry

end incorrect_quotient_l2141_214102


namespace rons_baseball_team_l2141_214144

/-- Ron's baseball team scored 270 points in the year. 
    5 players averaged 50 points each, 
    and the remaining players averaged 5 points each.
    Prove that the number of players on the team is 9. -/
theorem rons_baseball_team : (∃ n m : ℕ, 5 * 50 + m * 5 = 270 ∧ n = 5 + m ∧ 5 = 50 ∧ m = 4) :=
sorry

end rons_baseball_team_l2141_214144


namespace vandermonde_identity_combinatorial_identity_l2141_214123

open Nat

-- Problem 1: Vandermonde Identity
theorem vandermonde_identity (m n k : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < k) (h4 : m + n ≥ k) :
  (Finset.range (k + 1)).sum (λ i => Nat.choose m i * Nat.choose n (k - i)) = Nat.choose (m + n) k :=
sorry

-- Problem 2:
theorem combinatorial_identity (p q n : ℕ) (h1 : 0 < p) (h2 : 0 < q) (h3 : 0 < n) :
  (Finset.range (p + 1)).sum (λ k => Nat.choose p k * Nat.choose q k * Nat.choose (n + k) (p + q)) =
  Nat.choose n p * Nat.choose n q :=
sorry

end vandermonde_identity_combinatorial_identity_l2141_214123


namespace min_digs_is_three_l2141_214175

/-- Represents an 8x8 board --/
structure Board :=
(dim : ℕ := 8)

/-- Each cell either contains the treasure or a plaque indicating minimum steps --/
structure Cell :=
(content : CellContent)

/-- Possible content of a cell --/
inductive CellContent
| Treasure
| Plaque (steps : ℕ)

/-- Function that returns the minimum number of cells to dig to find the treasure --/
def min_digs_to_find_treasure (board : Board) : ℕ := 3

/-- The main theorem stating the minimum number of cells needed to find the treasure on an 8x8 board --/
theorem min_digs_is_three : 
  ∀ board : Board, min_digs_to_find_treasure board = 3 := 
by 
  intro board
  sorry

end min_digs_is_three_l2141_214175


namespace neg_p_l2141_214191

-- Let's define the original proposition p
def p : Prop := ∃ x : ℝ, x ≥ 2 ∧ x^2 - 2 * x - 2 > 0

-- Now, we state the problem in Lean as requiring the proof of the negation of p
theorem neg_p : ¬p ↔ ∀ x : ℝ, x ≥ 2 → x^2 - 2 * x - 2 ≤ 0 :=
by
  sorry

end neg_p_l2141_214191


namespace sliding_window_sash_translation_l2141_214176

def is_translation (movement : Type) : Prop := sorry

def ping_pong_ball_movement : Type := sorry
def sliding_window_sash_movement : Type := sorry
def kite_flight_movement : Type := sorry
def basketball_movement : Type := sorry

axiom ping_pong_not_translation : ¬ is_translation ping_pong_ball_movement
axiom kite_not_translation : ¬ is_translation kite_flight_movement
axiom basketball_not_translation : ¬ is_translation basketball_movement
axiom window_sash_is_translation : is_translation sliding_window_sash_movement

theorem sliding_window_sash_translation :
  is_translation sliding_window_sash_movement :=
by 
  exact window_sash_is_translation

end sliding_window_sash_translation_l2141_214176


namespace seconds_in_3_hours_25_minutes_l2141_214108

theorem seconds_in_3_hours_25_minutes:
  let hours := 3
  let minutesInAnHour := 60
  let additionalMinutes := 25
  let secondsInAMinute := 60
  (hours * minutesInAnHour + additionalMinutes) * secondsInAMinute = 12300 := 
by
  sorry

end seconds_in_3_hours_25_minutes_l2141_214108


namespace evaluate_fraction_l2141_214181

theorem evaluate_fraction : (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = (8 / 21) :=
by
  sorry

end evaluate_fraction_l2141_214181


namespace find_n_l2141_214179

theorem find_n (n : ℕ) (h : n > 0) :
  (n * (n - 1) * (n - 2)) / (6 * n^3) = 1 / 16 ↔ n = 4 :=
by sorry

end find_n_l2141_214179


namespace max_value_4x_plus_y_l2141_214152

theorem max_value_4x_plus_y (x y : ℝ) (h : 16 * x^2 + y^2 + 4 * x * y = 3) :
  ∃ (M : ℝ), M = 2 ∧ ∀ (u : ℝ), (∃ (x y : ℝ), 16 * x^2 + y^2 + 4 * x * y = 3 ∧ u = 4 * x + y) → u ≤ M :=
by
  use 2
  sorry

end max_value_4x_plus_y_l2141_214152


namespace triangle_ABC_angles_l2141_214111

theorem triangle_ABC_angles :
  ∃ (θ φ ω : ℝ), θ = 36 ∧ φ = 72 ∧ ω = 72 ∧
  (ω + φ + θ = 180) ∧
  (2 * ω + θ = 180) ∧
  (φ = 2 * θ) :=
by
  sorry

end triangle_ABC_angles_l2141_214111


namespace NumFriendsNextToCaraOnRight_l2141_214167

open Nat

def total_people : ℕ := 8
def freds_next_to_Cara : ℕ := 7

theorem NumFriendsNextToCaraOnRight (h : total_people = 8) : freds_next_to_Cara = 7 :=
by
  sorry

end NumFriendsNextToCaraOnRight_l2141_214167


namespace cars_through_toll_booth_l2141_214119

noncomputable def total_cars_in_week (n_mon n_tue n_wed n_thu n_fri n_sat n_sun : ℕ) : ℕ :=
  n_mon + n_tue + n_wed + n_thu + n_fri + n_sat + n_sun 

theorem cars_through_toll_booth : 
  let n_mon : ℕ := 50
  let n_tue : ℕ := 50
  let n_wed : ℕ := 2 * n_mon
  let n_thu : ℕ := 2 * n_mon
  let n_fri : ℕ := 50
  let n_sat : ℕ := 50
  let n_sun : ℕ := 50
  total_cars_in_week n_mon n_tue n_wed n_thu n_fri n_sat n_sun = 450 := 
by 
  sorry

end cars_through_toll_booth_l2141_214119


namespace multiplication_with_negative_l2141_214105

theorem multiplication_with_negative (a b : Int) (h1 : a = 3) (h2 : b = -2) : a * b = -6 :=
by
  sorry

end multiplication_with_negative_l2141_214105


namespace five_points_plane_distance_gt3_five_points_space_not_necessarily_gt3_l2141_214148

/-
Problem (a): Given five points on a plane, where the distance between any two points is greater than 2. 
             Prove that there exists a distance between some two of them that is greater than 3.
-/
theorem five_points_plane_distance_gt3 (P : Fin 5 → ℝ × ℝ) 
    (h : ∀ i j : Fin 5, i ≠ j → dist (P i) (P j) > 2) : 
    ∃ i j : Fin 5, i ≠ j ∧ dist (P i) (P j) > 3 :=
sorry

/-
Problem (b): Given five points in space, where the distance between any two points is greater than 2. 
             Prove that it is not necessarily true that there exists a distance between some two of them that is greater than 3.
-/
theorem five_points_space_not_necessarily_gt3 (P : Fin 5 → ℝ × ℝ × ℝ) 
    (h : ∀ i j : Fin 5, i ≠ j → dist (P i) (P j) > 2) : 
    ¬ ∃ i j : Fin 5, i ≠ j ∧ dist (P i) (P j) > 3 :=
sorry

end five_points_plane_distance_gt3_five_points_space_not_necessarily_gt3_l2141_214148


namespace percentage_students_school_A_l2141_214193

theorem percentage_students_school_A
  (A B : ℝ)
  (h1 : A + B = 100)
  (h2 : 0.30 * A + 0.40 * B = 34) :
  A = 60 :=
sorry

end percentage_students_school_A_l2141_214193


namespace oranges_to_friend_is_two_l2141_214121

-- Definitions based on the conditions.

def initial_oranges : ℕ := 12

def oranges_to_brother (n : ℕ) : ℕ := n / 3

def remainder_after_brother (n : ℕ) : ℕ := n - oranges_to_brother n

def oranges_to_friend (n : ℕ) : ℕ := remainder_after_brother n / 4

-- Theorem stating the problem to be proven.
theorem oranges_to_friend_is_two : oranges_to_friend initial_oranges = 2 :=
sorry

end oranges_to_friend_is_two_l2141_214121


namespace minimum_percentage_increase_in_mean_replacing_with_primes_l2141_214165

def mean (S : List ℤ) : ℚ :=
  (S.sum : ℚ) / S.length

noncomputable def percentage_increase (original new : ℚ) : ℚ :=
  ((new - original) / original) * 100

theorem minimum_percentage_increase_in_mean_replacing_with_primes :
  let F := [-4, -1, 0, 6, 9] 
  let G := [2, 3, 0, 6, 9] 
  percentage_increase (mean F) (mean G) = 100 :=
by {
  let F := [-4, -1, 0, 6, 9] 
  let G := [2, 3, 0, 6, 9] 
  sorry 
}

end minimum_percentage_increase_in_mean_replacing_with_primes_l2141_214165


namespace find_ck_l2141_214188

def arithmetic_seq (d : ℕ) (n : ℕ) : ℕ := 1 + (n - 1) * d
def geometric_seq (r : ℕ) (n : ℕ) : ℕ := r^(n - 1)
def c_seq (a_seq : ℕ → ℕ) (b_seq : ℕ → ℕ) (n : ℕ) := a_seq n + b_seq n

theorem find_ck (d r k : ℕ) (a_seq := arithmetic_seq d) (b_seq := geometric_seq r) :
  c_seq a_seq b_seq (k - 1) = 200 →
  c_seq a_seq b_seq (k + 1) = 400 →
  c_seq a_seq b_seq k = 322 :=
by
  sorry

end find_ck_l2141_214188


namespace example_problem_l2141_214195

-- Definitions and conditions derived from the original problem statement
def smallest_integer_with_two_divisors (m : ℕ) : Prop := m = 2
def second_largest_integer_with_three_divisors_less_than_100 (n : ℕ) : Prop := n = 25

theorem example_problem (m n : ℕ) 
    (h1 : smallest_integer_with_two_divisors m) 
    (h2 : second_largest_integer_with_three_divisors_less_than_100 n) : 
    m + n = 27 :=
by sorry

end example_problem_l2141_214195


namespace sufficient_but_not_necessary_condition_l2141_214168

variables (A B C : Prop)

theorem sufficient_but_not_necessary_condition (h1 : B → A) (h2 : C → B) (h3 : ¬(B → C)) : (C → A) ∧ ¬(A → C) :=
by
  sorry

end sufficient_but_not_necessary_condition_l2141_214168


namespace magnitude_a_eq_3sqrt2_l2141_214190

open Real

def a (x: ℝ) : ℝ × ℝ := (3, x)
def b : ℝ × ℝ := (-1, 1)
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem magnitude_a_eq_3sqrt2 (x : ℝ) (h : perpendicular (a x) b) :
  ‖a 3‖ = 3 * sqrt 2 := by
  sorry

end magnitude_a_eq_3sqrt2_l2141_214190


namespace find_m_n_l2141_214116

theorem find_m_n (x : ℝ) (m n : ℝ) 
  (h : (2 * x - 5) * (x + m) = 2 * x^2 - 3 * x + n) :
  m = 1 ∧ n = -5 :=
by
  have h_expand : (2 * x - 5) * (x + m) = 2 * x^2 + (2 * m - 5) * x - 5 * m := by
    ring
  rw [h_expand] at h
  have coeff_eq1 : 2 * m - 5 = -3 := by sorry
  have coeff_eq2 : -5 * m = n := by sorry
  have m_sol : m = 1 := by
    linarith [coeff_eq1]
  have n_sol : n = -5 := by
    rw [m_sol] at coeff_eq2
    linarith
  exact ⟨m_sol, n_sol⟩

end find_m_n_l2141_214116


namespace luke_money_at_end_of_june_l2141_214169

noncomputable def initial_money : ℝ := 48
noncomputable def february_money : ℝ := initial_money - 0.30 * initial_money
noncomputable def march_money : ℝ := february_money - 11 + 21 + 50 * 1.20

noncomputable def april_savings : ℝ := 0.10 * march_money
noncomputable def april_money : ℝ := (march_money - april_savings) - 10 * 1.18 + 0.05 * (march_money - april_savings)

noncomputable def may_savings : ℝ := 0.15 * april_money
noncomputable def may_money : ℝ := (april_money - may_savings) + 100 * 1.22 - 0.25 * ((april_money - may_savings) + 100 * 1.22)

noncomputable def june_savings : ℝ := 0.10 * may_money
noncomputable def june_money : ℝ := (may_money - june_savings) - 0.08 * (may_money - june_savings)
noncomputable def final_money : ℝ := june_money + 0.06 * (may_money - june_savings)

theorem luke_money_at_end_of_june : final_money = 128.15 := sorry

end luke_money_at_end_of_june_l2141_214169


namespace angle_ABC_30_degrees_l2141_214170

theorem angle_ABC_30_degrees 
    (angle_CBD : ℝ)
    (angle_ABD : ℝ)
    (angle_ABC : ℝ)
    (h1 : angle_CBD = 90)
    (h2 : angle_ABC + angle_ABD + angle_CBD = 180)
    (h3 : angle_ABD = 60) :
    angle_ABC = 30 :=
by
  sorry

end angle_ABC_30_degrees_l2141_214170


namespace rainfall_second_week_l2141_214120

theorem rainfall_second_week (r1 r2 : ℝ) (h1 : r1 + r2 = 35) (h2 : r2 = 1.5 * r1) : r2 = 21 := 
  sorry

end rainfall_second_week_l2141_214120


namespace assignment_increase_l2141_214101

-- Define what an assignment statement is
def assignment_statement (lhs rhs : ℕ) : ℕ := rhs

-- Define the conditions and the problem
theorem assignment_increase (n : ℕ) : assignment_statement n (n + 1) = n + 1 :=
by
  -- Here we would prove that the assignment statement increases n by 1
  sorry

end assignment_increase_l2141_214101


namespace solve_for_c_l2141_214118

noncomputable def proof_problem (a b c : ℝ) : Prop :=
  (a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) →
  (6 * 15 * c = 1.5) →
  c = 7

theorem solve_for_c : proof_problem 6 15 7 :=
by sorry

end solve_for_c_l2141_214118


namespace range_of_a_l2141_214104

theorem range_of_a (a : ℝ) :
  (-1 < x ∧ x < 0 → (x^2 - a * x + 2 * a) > 0) ∧
  (0 < x → (x^2 - a * x + 2 * a) < 0) ↔ -1 / 3 < a ∧ a < 0 :=
sorry

end range_of_a_l2141_214104


namespace greatest_value_a4_b4_l2141_214147

theorem greatest_value_a4_b4
    (a b : Nat → ℝ)
    (h_arith_seq : ∀ n, a (n + 1) = a n + a 1)
    (h_geom_seq : ∀ n, b (n + 1) = b n * b 1)
    (h_a1b1 : a 1 * b 1 = 20)
    (h_a2b2 : a 2 * b 2 = 19)
    (h_a3b3 : a 3 * b 3 = 14) :
    ∃ m : ℝ, a 4 * b 4 = 8 ∧ ∀ x, a 4 * b 4 ≤ x -> x = 8 := by
  sorry

end greatest_value_a4_b4_l2141_214147


namespace fraction_days_passed_l2141_214109

-- Conditions
def total_days : ℕ := 30
def pills_per_day : ℕ := 2
def total_pills : ℕ := total_days * pills_per_day -- 60 pills
def pills_left : ℕ := 12
def pills_taken : ℕ := total_pills - pills_left -- 48 pills
def days_taken : ℕ := pills_taken / pills_per_day -- 24 days

-- Question and answer
theorem fraction_days_passed :
  (days_taken : ℚ) / (total_days : ℚ) = 4 / 5 := 
by
  sorry

end fraction_days_passed_l2141_214109


namespace max_value_of_abs_z_plus_4_l2141_214182

open Complex
noncomputable def max_abs_z_plus_4 {z : ℂ} (h : abs (z + 3 * I) = 5) : ℝ :=
sorry

theorem max_value_of_abs_z_plus_4 (z : ℂ) (h : abs (z + 3 * I) = 5) : abs (z + 4) ≤ 10 :=
sorry

end max_value_of_abs_z_plus_4_l2141_214182


namespace determine_B_l2141_214162

-- Declare the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {0, 1}

-- The conditions given in the problem
axiom h1 : A ∩ B = {1}
axiom h2 : A ∪ B = {0, 1, 2}

-- The theorem we want to prove
theorem determine_B : B = {0, 1} :=
by
  sorry

end determine_B_l2141_214162


namespace find_hidden_data_points_l2141_214174

-- Given conditions and data
def student_A_score := 81
def student_B_score := 76
def student_D_score := 80
def student_E_score := 83
def number_of_students := 5
def average_score := 80

-- The total score from the average and number of students
def total_score := average_score * number_of_students

theorem find_hidden_data_points (student_C_score mode_score : ℕ) :
  (student_A_score + student_B_score + student_C_score + student_D_score + student_E_score = total_score) ∧
  (mode_score = 80) :=
by
  sorry

end find_hidden_data_points_l2141_214174


namespace arithmetic_sequence_sum_l2141_214141

theorem arithmetic_sequence_sum (a b d : ℕ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ)
  (h1 : a₁ + a₂ + a₃ = 39)
  (h2 : a₄ + a₅ + a₆ = 27)
  (h3 : a₄ = a₁ + 3 * d)
  (h4 : a₅ = a₂ + 3 * d)
  (h5 : a₆ = a₃ + 3 * d)
  (h6 : a₇ = a₄ + 3 * d)
  (h7 : a₈ = a₅ + 3 * d)
  (h8 : a₉ = a₆ + 3 * d) :
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 81 :=
sorry

end arithmetic_sequence_sum_l2141_214141


namespace evaluate_fraction_l2141_214132

theorem evaluate_fraction :
  1 + (2 / (3 + (6 / (7 + (8 / 9))))) = 409 / 267 :=
by
  sorry

end evaluate_fraction_l2141_214132


namespace assembly_shortest_time_l2141_214198

-- Define the times taken for each assembly path
def time_ACD : ℕ := 3 + 4
def time_EDF : ℕ := 4 + 2

-- State the theorem for the shortest time required to assemble the product
theorem assembly_shortest_time : max time_ACD time_EDF + 4 = 13 :=
by {
  -- Introduction of the given conditions and simplified value calculation
  sorry
}

end assembly_shortest_time_l2141_214198


namespace division_equals_fraction_l2141_214183

theorem division_equals_fraction:
  180 / (8 + 9 * 3 - 4) = 180 / 31 := 
by
  sorry

end division_equals_fraction_l2141_214183


namespace investment_total_amount_l2141_214187

noncomputable def compoundedInvestment (principal : ℝ) (rate : ℝ) (tax : ℝ) (years : ℕ) : ℝ :=
let yearlyNetInterest := principal * rate * (1 - tax)
let rec calculate (year : ℕ) (accumulated : ℝ) : ℝ :=
  if year = 0 then accumulated else
    let newPrincipal := accumulated + yearlyNetInterest
    calculate (year - 1) newPrincipal
calculate years principal

theorem investment_total_amount :
  let finalAmount := compoundedInvestment 15000 0.05 0.10 4
  round finalAmount = 17607 :=
by
  sorry

end investment_total_amount_l2141_214187


namespace total_weight_of_13_gold_bars_l2141_214163

theorem total_weight_of_13_gold_bars
    (C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 : ℝ)
    (w12 w13 w23 w45 w67 w89 w1011 w1213 : ℝ)
    (h1 : w12 = C1 + C2)
    (h2 : w13 = C1 + C3)
    (h3 : w23 = C2 + C3)
    (h4 : w45 = C4 + C5)
    (h5 : w67 = C6 + C7)
    (h6 : w89 = C8 + C9)
    (h7 : w1011 = C10 + C11)
    (h8 : w1213 = C12 + C13) :
    C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10 + C11 + C12 + C13 = 
    (C1 + C2 + C3) + (C4 + C5) + (C6 + C7) + (C8 + C9) + (C10 + C11) + (C12 + C13) := 
  by
  sorry

end total_weight_of_13_gold_bars_l2141_214163


namespace cannot_equal_120_l2141_214106

def positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

theorem cannot_equal_120 (a b : ℕ) (ha : positive_even a) (hb : positive_even b) :
  let A := a * b
  let P' := 2 * (a + b) + 6
  A + P' ≠ 120 :=
sorry

end cannot_equal_120_l2141_214106


namespace part1_part2_l2141_214137

noncomputable def f (x a : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + a)

theorem part1 (x : ℝ) : (∃ a, a = 1) → f x 1 > 1 ↔ -2 < x ∧ x < -(2/3) := by
  sorry

theorem part2 (a : ℝ) : (∀ x, 2 ≤ x → x ≤ 3 → f x a > 0) ↔ (-5/2) < a ∧ a < -2 := by
  sorry

end part1_part2_l2141_214137


namespace remainder_when_dividing_386_l2141_214107

theorem remainder_when_dividing_386 :
  (386 % 35 = 1) ∧ (386 % 11 = 1) :=
by
  sorry

end remainder_when_dividing_386_l2141_214107


namespace S_3n_plus_1_l2141_214136

noncomputable def S : ℕ → ℝ := sorry  -- S_n is the sum of the first n terms of the sequence {a_n}
noncomputable def a : ℕ → ℝ := sorry  -- Sequence {a_n}

-- Given conditions
axiom S3 : S 3 = 1
axiom S4 : S 4 = 11
axiom a_recurrence (n : ℕ) : a (n + 3) = 2 * a n

-- Define S_{3n+1} in terms of n
theorem S_3n_plus_1 (n : ℕ) : S (3 * n + 1) = 3 * 2^(n+1) - 1 :=
sorry

end S_3n_plus_1_l2141_214136


namespace problem1_problem2_l2141_214154

-- Definitions of the sets
def U : Set ℕ := { x | 1 ≤ x ∧ x ≤ 7 }
def A : Set ℕ := { x | 2 ≤ x ∧ x ≤ 5 }
def B : Set ℕ := { x | 3 ≤ x ∧ x ≤ 7 }

-- Problems to prove (statements only, no proofs provided)
theorem problem1 : A ∪ B = {x | 2 ≤ x ∧ x ≤ 7} :=
by
  sorry

theorem problem2 : U \ A ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)} :=
by
  sorry

end problem1_problem2_l2141_214154


namespace unique_solution_condition_l2141_214189

theorem unique_solution_condition (p q : ℝ) : 
  (∃! x : ℝ, 4 * x - 7 + p = q * x + 2) ↔ q ≠ 4 :=
by
  sorry

end unique_solution_condition_l2141_214189


namespace find_z_l2141_214133

theorem find_z (x y k : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (h : 1/x + 1/y = k) :
  ∃ z : ℝ, 1/z = k ∧ z = xy/(x + y) :=
by {
  sorry
}

end find_z_l2141_214133


namespace find_n_l2141_214129

-- Define the hyperbola and its properties
def hyperbola (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ 2 = (m / (m / 2)) ∧ ∃ f : ℝ × ℝ, f = (m, 0)

-- Define the parabola and its properties
def parabola_focus (m : ℝ) : Prop :=
  (m, 0) = (m, 0)

-- The statement we want to prove
theorem find_n (m : ℝ) (n : ℝ) (H_hyperbola : hyperbola m n) (H_parabola : parabola_focus m) : n = 12 :=
sorry

end find_n_l2141_214129


namespace stuffed_animals_mom_gift_l2141_214149

theorem stuffed_animals_mom_gift (x : ℕ) :
  (10 + x) + 3 * (10 + x) = 48 → x = 2 :=
by {
  sorry
}

end stuffed_animals_mom_gift_l2141_214149


namespace find_f_a_l2141_214153

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 4 * Real.logb 2 (-x) else abs (x^2 + a * x)

theorem find_f_a (a : ℝ) (h : a ≠ 0) (h1 : f a (f a (-Real.sqrt 2)) = 4) : f a a = 8 :=
sorry

end find_f_a_l2141_214153


namespace cyclist_first_part_distance_l2141_214131

theorem cyclist_first_part_distance
  (T₁ T₂ T₃ : ℝ)
  (D : ℝ)
  (h1 : D = 9 * T₁)
  (h2 : T₂ = 12 / 10)
  (h3 : T₃ = (D + 12) / 7.5)
  (h4 : T₁ + T₂ + T₃ = 7.2) : D = 18 := by
  sorry

end cyclist_first_part_distance_l2141_214131


namespace am_gm_inequality_l2141_214127

theorem am_gm_inequality {a1 a2 a3 : ℝ} (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) :
  (a1 * a2 / a3) + (a2 * a3 / a1) + (a3 * a1 / a2) ≥ a1 + a2 + a3 := 
by 
  sorry

end am_gm_inequality_l2141_214127


namespace range_of_a_l2141_214117

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x + 1 > 2 * x - 2) → (x < a)) → (a ≥ 3) :=
by
  sorry

end range_of_a_l2141_214117


namespace sunflower_count_l2141_214155

theorem sunflower_count (r l d : ℕ) (t : ℕ) (h1 : r + l + d = 40) (h2 : t = 160) : 
  t - (r + l + d) = 120 := by
  sorry

end sunflower_count_l2141_214155


namespace stratified_sampling_l2141_214157

theorem stratified_sampling (lathe_A lathe_B total_samples : ℕ) (hA : lathe_A = 56) (hB : lathe_B = 42) (hTotal : total_samples = 14) :
  ∃ (sample_A sample_B : ℕ), sample_A = 8 ∧ sample_B = 6 :=
by
  sorry

end stratified_sampling_l2141_214157


namespace find_number_l2141_214158

theorem find_number (x : ℤ) (h : 5 * x + 4 = 19) : x = 3 := sorry

end find_number_l2141_214158


namespace speed_of_A_l2141_214184

theorem speed_of_A (V_B : ℝ) (h_VB : V_B = 4.555555555555555)
  (h_B_overtakes: ∀ (t_A t_B : ℝ), t_A = t_B + 0.5 → t_B = 1.8) 
  : ∃ V_A : ℝ, V_A = 3.57 :=
by
  sorry

end speed_of_A_l2141_214184


namespace intersection_complements_l2141_214135

open Set

variable (U : Set (ℝ × ℝ))
variable (M : Set (ℝ × ℝ))
variable (N : Set (ℝ × ℝ))

noncomputable def complementU (A : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := U \ A

theorem intersection_complements :
  let U := {p : ℝ × ℝ | True}
  let M := {p : ℝ × ℝ | (∃ (x y : ℝ), p = (x, y) ∧ y + 2 = x - 2 ∧ x ≠ 2)}
  let N := {p : ℝ × ℝ | (∃ (x y : ℝ), p = (x, y) ∧ y ≠ x - 4)}
  ((complementU U M) ∩ (complementU U N)) = {(2, -2)} :=
by
  let U := {(x, y) : ℝ × ℝ | True}
  let M := {(x, y) : ℝ × ℝ | (y + 2) = (x - 2) ∧ x ≠ 2}
  let N := {(x, y) : ℝ × ℝ | y ≠ (x - 4)}
  have complement_M := U \ M
  have complement_N := U \ N
  sorry

end intersection_complements_l2141_214135


namespace saree_blue_stripes_l2141_214110

theorem saree_blue_stripes :
  ∀ (brown_stripes gold_stripes blue_stripes : ℕ),
    brown_stripes = 4 →
    gold_stripes = 3 * brown_stripes →
    blue_stripes = 5 * gold_stripes →
    blue_stripes = 60 :=
by
  intros brown_stripes gold_stripes blue_stripes h_brown h_gold h_blue
  sorry

end saree_blue_stripes_l2141_214110


namespace matrix_pow_minus_l2141_214100

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 4], ![0, 2]]

theorem matrix_pow_minus : B ^ 20 - 3 * (B ^ 19) = ![![0, 4 * (2 ^ 19)], ![0, -(2 ^ 19)]] :=
by
  sorry

end matrix_pow_minus_l2141_214100


namespace proof_of_min_value_l2141_214139

def constraints_on_powers (a b c d : ℝ) : Prop :=
  a^4 + b^4 + c^4 + d^4 = 16

noncomputable def minimum_third_power_sum (a b c d : ℝ) : ℝ :=
  a^3 + b^3 + c^3 + d^3

theorem proof_of_min_value : 
  ∃ a b c d : ℝ, constraints_on_powers a b c d → ∃ min_val : ℝ, min_val = minimum_third_power_sum a b c d :=
sorry -- Further method to rigorously find the minimum value.

end proof_of_min_value_l2141_214139

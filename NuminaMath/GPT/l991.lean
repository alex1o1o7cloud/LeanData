import Mathlib

namespace janine_test_score_l991_99120

theorem janine_test_score :
  let num_mc := 10
  let p_mc := 0.80
  let num_sa := 30
  let p_sa := 0.70
  let total_questions := 40
  let correct_mc := p_mc * num_mc
  let correct_sa := p_sa * num_sa
  let total_correct := correct_mc + correct_sa
  (total_correct / total_questions) * 100 = 72.5 := 
by
  sorry

end janine_test_score_l991_99120


namespace convex_pentagon_probability_l991_99170

-- Defining the number of chords and the probability calculation as per the problem's conditions
def number_of_chords (n : ℕ) : ℕ := (n * (n - 1)) / 2
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem conditions
def eight_points_on_circle : ℕ := 8
def chords_chosen : ℕ := 5

-- Total number of chords from eight points
def total_chords : ℕ := number_of_chords eight_points_on_circle

-- The probability calculation
def probability_convex_pentagon :=
  binom 8 5 / binom total_chords chords_chosen

-- Statement to be proven
theorem convex_pentagon_probability :
  probability_convex_pentagon = 1 / 1755 := sorry

end convex_pentagon_probability_l991_99170


namespace sufficient_but_not_necessary_l991_99147

theorem sufficient_but_not_necessary (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : (a^2 + b^2 < 1) → (ab + 1 > a + b) ∧ ¬(ab + 1 > a + b ↔ a^2 + b^2 < 1) := 
sorry

end sufficient_but_not_necessary_l991_99147


namespace monotonically_increasing_sequence_l991_99155

theorem monotonically_increasing_sequence (k : ℝ) : (∀ n : ℕ+, n^2 + k * n < (n + 1)^2 + k * (n + 1)) ↔ k > -3 := by
  sorry

end monotonically_increasing_sequence_l991_99155


namespace expression_value_l991_99184

theorem expression_value :
  let x := (3 + 1 : ℚ)⁻¹ * 2
  let y := x⁻¹ * 2
  let z := y⁻¹ * 2
  z = (1 / 2 : ℚ) :=
by
  sorry

end expression_value_l991_99184


namespace min_pq_sq_min_value_l991_99162

noncomputable def min_pq_sq (α : ℝ) : ℝ :=
  let p := α - 2
  let q := -(α + 1)
  (p + q)^2 - 2 * (p * q)

theorem min_pq_sq_min_value : 
  (∃ (α : ℝ), ∀ p q : ℝ, 
    p^2 + q^2 = (p + q)^2 - 2 * p * q ∧ 
    (p + q = α - 2 ∧ p * q = -(α + 1))) → 
  (min_pq_sq 1) = 5 :=
by
  sorry

end min_pq_sq_min_value_l991_99162


namespace problem_solution_l991_99128

open Set

-- Define the universal set U
def U : Set ℝ := univ

-- Define the set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the set N using the given condition
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- Define the complement of N in U
def complement_N : Set ℝ := U \ N

-- Define the intersection of M and the complement of N
def result_set : Set ℝ := M ∩ complement_N

-- Prove the desired result
theorem problem_solution : result_set = {x | -2 ≤ x ∧ x < 0} :=
sorry

end problem_solution_l991_99128


namespace intersection_is_one_l991_99145

def M : Set ℝ := {x | x - 1 = 0}
def N : Set ℝ := {x | x^2 - 3 * x + 2 = 0}

theorem intersection_is_one : M ∩ N = {1} :=
by
  sorry

end intersection_is_one_l991_99145


namespace arithmetic_sequence_length_l991_99131

theorem arithmetic_sequence_length 
  (a₁ : ℕ) (d : ℤ) (x : ℤ) (n : ℕ) 
  (h_start : a₁ = 20)
  (h_diff : d = -2)
  (h_eq : x = 10)
  (h_term : x = a₁ + (n - 1) * d) :
  n = 6 :=
by
  sorry

end arithmetic_sequence_length_l991_99131


namespace part1_part2_l991_99108

variable (a : ℝ)
variable (x y : ℝ)
variable (P Q : ℝ × ℝ)

-- Part (1)
theorem part1 (hP : P = (2 * a - 2, a + 5)) (h_y : y = 0) : P = (-12, 0) :=
sorry

-- Part (2)
theorem part2 (hP : P = (2 * a - 2, a + 5)) (hQ : Q = (4, 5)) 
    (h_parallel : 2 * a - 2 = 4) : P = (4, 8) ∧ quadrant = "first" :=
sorry

end part1_part2_l991_99108


namespace percentage_increase_in_length_l991_99100

theorem percentage_increase_in_length (L B : ℝ) (hB : 0 < B) (hL : 0 < L) :
  (1 + x / 100) * 1.22 = 1.3542 -> x = 11.016393 :=
by
  sorry

end percentage_increase_in_length_l991_99100


namespace value_b15_l991_99141

def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d
def geometric_sequence (b : ℕ → ℤ) := ∃ q : ℤ, ∀ n : ℕ, b (n+1) = q * b n

theorem value_b15 
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (b : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : ∀ n : ℕ, S n = (n * (a 0 + a (n-1)) / 2))
  (h3 : S 9 = -18)
  (h4 : S 13 = -52)
  (h5 : geometric_sequence b)
  (h6 : b 5 = a 5)
  (h7 : b 7 = a 7) : 
  b 15 = -64 :=
sorry

end value_b15_l991_99141


namespace bottles_needed_l991_99165

theorem bottles_needed (runners : ℕ) (bottles_needed_per_runner : ℕ) (bottles_available : ℕ)
  (h_runners : runners = 14)
  (h_bottles_needed_per_runner : bottles_needed_per_runner = 5)
  (h_bottles_available : bottles_available = 68) :
  runners * bottles_needed_per_runner - bottles_available = 2 :=
by
  sorry

end bottles_needed_l991_99165


namespace sin_15_mul_sin_75_l991_99156

theorem sin_15_mul_sin_75 : Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 4 :=
by
  sorry

end sin_15_mul_sin_75_l991_99156


namespace find_number_l991_99194

variable (x : ℝ)

theorem find_number (h : 2 * x - 6 = (1/4) * x + 8) : x = 8 :=
sorry

end find_number_l991_99194


namespace comparison_l991_99183

open Real

noncomputable def a := 5 * log (2 ^ exp 1)
noncomputable def b := 2 * log (5 ^ exp 1)
noncomputable def c := 10

theorem comparison : c > a ∧ a > b :=
by
  have a_def : a = 5 * log (2 ^ exp 1) := rfl
  have b_def : b = 2 * log (5 ^ exp 1) := rfl
  have c_def : c = 10 := rfl
  sorry -- Proof goes here

end comparison_l991_99183


namespace find_initial_men_l991_99127

def men_employed (M : ℕ) : Prop :=
  let total_hours := 50 * 8
  let completed_hours := 25 * 8
  let remaining_hours := total_hours - completed_hours
  let new_hours := 25 * 10
  let completed_work := 1 / 3
  let remaining_work := 2 / 3
  let total_work := 2 -- Total work in terms of "work units", assuming 2 km = 2 work units
  let first_eq := M * 25 * 8 = total_work * completed_work
  let second_eq := (M + 60) * 25 * 10 = total_work * remaining_work
  (M = 300 → first_eq ∧ second_eq)

theorem find_initial_men : ∃ M : ℕ, men_employed M := sorry

end find_initial_men_l991_99127


namespace find_x_l991_99191

def delta (x : ℝ) : ℝ := 4 * x + 9
def phi (x : ℝ) : ℝ := 9 * x + 6

theorem find_x (x : ℝ) (h : delta (phi x) = 10) : x = -23 / 36 := 
by 
  sorry

end find_x_l991_99191


namespace purely_imaginary_subtraction_l991_99187

-- Definition of the complex number z.
def z : ℂ := Complex.mk 2 (-1)

-- Statement to prove
theorem purely_imaginary_subtraction (h: z = Complex.mk 2 (-1)) : ∃ (b : ℝ), z - 2 = Complex.im b :=
by {
    sorry
}

end purely_imaginary_subtraction_l991_99187


namespace geom_sequence_ratio_and_fifth_term_l991_99154

theorem geom_sequence_ratio_and_fifth_term 
  (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ = 10) 
  (h₂ : a₂ = -15) 
  (h₃ : a₃ = 22.5) 
  (h₄ : a₄ = -33.75) : 
  ∃ r a₅, r = -1.5 ∧ a₅ = 50.625 ∧ (a₂ = r * a₁) ∧ (a₃ = r * a₂) ∧ (a₄ = r * a₃) ∧ (a₅ = r * a₄) := 
by
  sorry

end geom_sequence_ratio_and_fifth_term_l991_99154


namespace polynomial_never_33_l991_99186

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 :=
by
  sorry

end polynomial_never_33_l991_99186


namespace find_b_l991_99188

noncomputable def g (b x : ℝ) : ℝ := b * x^2 - Real.cos (Real.pi * x)

theorem find_b (b : ℝ) (hb : 0 < b) (h : g b (g b 1) = -Real.cos Real.pi) : b = 1 :=
by
  sorry

end find_b_l991_99188


namespace total_bike_clamps_given_away_l991_99101

-- Definitions for conditions
def bike_clamps_per_bike := 2
def bikes_sold_morning := 19
def bikes_sold_afternoon := 27

-- Theorem statement to be proven
theorem total_bike_clamps_given_away :
  bike_clamps_per_bike * bikes_sold_morning +
  bike_clamps_per_bike * bikes_sold_afternoon = 92 :=
by
  sorry -- Proof is to be filled in later

end total_bike_clamps_given_away_l991_99101


namespace value_of_x_l991_99189

theorem value_of_x (a x y : ℝ) 
  (h1 : a^(x - y) = 343) 
  (h2 : a^(x + y) = 16807) : x = 4 :=
by
  sorry

end value_of_x_l991_99189


namespace book_total_pages_eq_90_l991_99130

theorem book_total_pages_eq_90 {P : ℕ} (h1 : (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 30) : P = 90 :=
sorry

end book_total_pages_eq_90_l991_99130


namespace juniors_in_program_l991_99110

theorem juniors_in_program (J S x y : ℕ) (h1 : J + S = 40) 
                           (h2 : x = y) 
                           (h3 : J / 5 = x) 
                           (h4 : S / 10 = y) : J = 12 :=
by
  sorry

end juniors_in_program_l991_99110


namespace smallest_n_with_divisors_2020_l991_99124

theorem smallest_n_with_divisors_2020 :
  ∃ n : ℕ, (∃ α1 α2 α3 : ℕ, 
  n = 2^α1 * 3^α2 * 5^α3 ∧
  (α1 + 1) * (α2 + 1) * (α3 + 1) = 2020) ∧
  n = 2^100 * 3^4 * 5 * 7 := by
  sorry

end smallest_n_with_divisors_2020_l991_99124


namespace base6_addition_correct_l991_99179

-- Define a function to convert a base 6 digit to its base 10 equivalent
def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 5 => 5
  | d => 0 -- for illegal digits, fallback to 0

-- Define a function to convert a number in base 6 to base 10
def convert_base6_to_base10 (n : Nat) : Nat :=
  let units := base6_to_base10 (n % 10)
  let tens := base6_to_base10 ((n / 10) % 10)
  let hundreds := base6_to_base10 ((n / 100) % 10)
  units + 6 * tens + 6 * 6 * hundreds

-- Define a function to convert a base 10 number to a base 6 number
def base10_to_base6 (n : Nat) : Nat :=
  (n % 6) + 10 * ((n / 6) % 6) + 100 * ((n / (6 * 6)) % 6)

theorem base6_addition_correct : base10_to_base6 (convert_base6_to_base10 35 + convert_base6_to_base10 25) = 104 := by
  sorry

end base6_addition_correct_l991_99179


namespace brownies_count_l991_99105

theorem brownies_count (pan_length : ℕ) (pan_width : ℕ) (piece_side : ℕ) 
  (h1 : pan_length = 24) (h2 : pan_width = 15) (h3 : piece_side = 3) : 
  (pan_length * pan_width) / (piece_side * piece_side) = 40 :=
by {
  sorry
}

end brownies_count_l991_99105


namespace common_rational_root_is_negative_non_integer_l991_99181

theorem common_rational_root_is_negative_non_integer 
    (a b c d e f g : ℤ)
    (p : ℚ)
    (h1 : 90 * p^4 + a * p^3 + b * p^2 + c * p + 15 = 0)
    (h2 : 15 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 90 = 0)
    (h3 : ¬ (∃ k : ℤ, p = k))
    (h4 : p < 0) : 
  p = -1 / 3 := 
sorry

end common_rational_root_is_negative_non_integer_l991_99181


namespace determinant_scaled_l991_99102

theorem determinant_scaled
  (x y z w : ℝ)
  (h : x * w - y * z = 10) :
  (3 * x) * (3 * w) - (3 * y) * (3 * z) = 90 :=
by sorry

end determinant_scaled_l991_99102


namespace total_genuine_purses_and_handbags_l991_99106

def TirzahPurses : ℕ := 26
def TirzahHandbags : ℕ := 24
def FakePurses : ℕ := TirzahPurses / 2
def FakeHandbags : ℕ := TirzahHandbags / 4
def GenuinePurses : ℕ := TirzahPurses - FakePurses
def GenuineHandbags : ℕ := TirzahHandbags - FakeHandbags

theorem total_genuine_purses_and_handbags : GenuinePurses + GenuineHandbags = 31 := by
  sorry

end total_genuine_purses_and_handbags_l991_99106


namespace ternary_to_decimal_l991_99196

theorem ternary_to_decimal (n : ℕ) (h : n = 121) : 
  (1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 16 :=
by sorry

end ternary_to_decimal_l991_99196


namespace find_value_l991_99175

theorem find_value :
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800 :=
by
  sorry

end find_value_l991_99175


namespace mph_to_fps_l991_99177

theorem mph_to_fps (C G : ℝ) (x : ℝ) (hC : C = 60 * x) (hG : G = 40 * x) (h1 : 7 * C - 7 * G = 210) :
  x = 1.5 :=
by {
  -- Math proof here, but we insert sorry for now
  sorry
}

end mph_to_fps_l991_99177


namespace relationship_p_q_l991_99135

noncomputable def p (α β : ℝ) : ℝ := Real.cos α * Real.cos β
noncomputable def q (α β : ℝ) : ℝ := Real.cos ((α + β) / 2) ^ 2

theorem relationship_p_q (α β : ℝ) : p α β ≤ q α β :=
by
  sorry

end relationship_p_q_l991_99135


namespace polygon_sides_l991_99116

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  sorry

end polygon_sides_l991_99116


namespace prove_range_of_a_l991_99125

noncomputable def f (x a : ℝ) : ℝ := (x + a - 1) * Real.exp x

def problem_condition1 (x a : ℝ) : Prop := 
  f x a ≥ (x^2 / 2 + a * x)

def problem_condition2 (x : ℝ) : Prop := 
  x ∈ Set.Ici 0 -- equivalent to [0, +∞)

theorem prove_range_of_a (a : ℝ) :
  (∀ x : ℝ, problem_condition2 x → problem_condition1 x a) → a ∈ Set.Ici 1 :=
sorry

end prove_range_of_a_l991_99125


namespace net_effect_on_sale_l991_99119

theorem net_effect_on_sale (P Q : ℝ) :
  let new_price := 0.65 * P
  let new_quantity := 1.8 * Q
  let original_revenue := P * Q
  let new_revenue := new_price * new_quantity
  new_revenue - original_revenue = 0.17 * original_revenue :=
by
  sorry

end net_effect_on_sale_l991_99119


namespace bat_wings_area_l991_99115

structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨0, 0⟩
def Q : Point := ⟨5, 0⟩
def R : Point := ⟨5, 2⟩
def S : Point := ⟨0, 2⟩
def A : Point := ⟨5, 1⟩
def T : Point := ⟨3, 2⟩

def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))

theorem bat_wings_area :
  area_triangle P A T = 5.5 :=
sorry

end bat_wings_area_l991_99115


namespace true_statement_count_l991_99113

def reciprocal (n : ℕ) : ℚ := 1 / n

def statement_i := (reciprocal 4 + reciprocal 8 = reciprocal 12)
def statement_ii := (reciprocal 9 - reciprocal 3 = reciprocal 6)
def statement_iii := (reciprocal 3 * reciprocal 9 = reciprocal 27)
def statement_iv := (reciprocal 15 / reciprocal 3 = reciprocal 5)

theorem true_statement_count :
  (¬statement_i ∧ ¬statement_ii ∧ statement_iii ∧ statement_iv) ↔ (2 = 2) :=
by sorry

end true_statement_count_l991_99113


namespace games_played_so_far_l991_99185

-- Definitions based on conditions
def total_matches := 20
def points_for_victory := 3
def points_for_draw := 1
def points_for_defeat := 0
def points_scored_so_far := 14
def points_needed := 40
def required_wins := 6

-- The proof problem
theorem games_played_so_far : 
  ∃ W D L : ℕ, 3 * W + D + 0 * L = points_scored_so_far ∧ 
  ∃ W' D' L' : ℕ, 3 * W' + D' + 0 * L' + 3 * required_wins = points_needed ∧ 
  (total_matches - required_wins = 14) :=
by 
  sorry

end games_played_so_far_l991_99185


namespace circle_area_ratio_l991_99139

theorem circle_area_ratio (O X P : ℝ) (rOx rOp : ℝ) (h1 : rOx = rOp / 3) :
  (π * rOx^2) / (π * rOp^2) = 1 / 9 :=
by 
  -- Import required theorems and add assumptions as necessary
  -- Continue the proof based on Lean syntax
  sorry

end circle_area_ratio_l991_99139


namespace solve_system_of_equations_l991_99178

theorem solve_system_of_equations
  {a b c d x y z : ℝ}
  (h1 : x + y + z = 1)
  (h2 : a * x + b * y + c * z = d)
  (h3 : a^2 * x + b^2 * y + c^2 * z = d^2)
  (hne1 : a ≠ b)
  (hne2 : a ≠ c)
  (hne3 : b ≠ c) :
  x = (d - b) * (d - c) / ((a - b) * (a - c)) ∧
  y = (d - a) * (d - c) / ((b - a) * (b - c)) ∧
  z = (d - a) * (d - b) / ((c - a) * (c - b)) :=
sorry

end solve_system_of_equations_l991_99178


namespace min_blue_edges_l991_99142

def tetrahedron_min_blue_edges : ℕ := sorry

theorem min_blue_edges (edges_colored : ℕ → Bool) (face_has_blue_edge : ℕ → Bool) 
    (H1 : ∀ face, face_has_blue_edge face)
    (H2 : ∀ edge, face_has_blue_edge edge = True → edges_colored edge = True) : 
    tetrahedron_min_blue_edges = 2 := 
sorry

end min_blue_edges_l991_99142


namespace total_flowers_eaten_l991_99138

theorem total_flowers_eaten (bugs : ℕ) (flowers_per_bug : ℕ) (h_bugs : bugs = 3) (h_flowers_per_bug : flowers_per_bug = 2) :
  (bugs * flowers_per_bug) = 6 :=
by
  sorry

end total_flowers_eaten_l991_99138


namespace zero_if_sum_of_squares_eq_zero_l991_99144

theorem zero_if_sum_of_squares_eq_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end zero_if_sum_of_squares_eq_zero_l991_99144


namespace g_ten_l991_99158

-- Define the function g and its properties
def g : ℝ → ℝ := sorry

axiom g_property1 : ∀ x y : ℝ, g (x * y) = 2 * g x * g y
axiom g_property2 : g 0 = 2

-- Prove that g 10 = 1 / 2
theorem g_ten : g 10 = 1 / 2 :=
by
  sorry

end g_ten_l991_99158


namespace find_x_values_l991_99111

theorem find_x_values (x : ℝ) (h : x ≠ 5) : x + 36 / (x - 5) = -12 ↔ x = -8 ∨ x = 3 :=
by sorry

end find_x_values_l991_99111


namespace older_brother_is_14_l991_99133

theorem older_brother_is_14 {Y O : ℕ} (h1 : Y + O = 26) (h2 : O = Y + 2) : O = 14 :=
by
  sorry

end older_brother_is_14_l991_99133


namespace ratio_bananas_dates_l991_99104

theorem ratio_bananas_dates (s c b d a : ℕ)
  (h1 : s = 780)
  (h2 : c = 60)
  (h3 : b = 3 * c)
  (h4 : a = 2 * d)
  (h5 : s = a + b + c + d) :
  b / d = 1 :=
by sorry

end ratio_bananas_dates_l991_99104


namespace calculate_expression_l991_99148

def f (x : ℝ) := 2 * x^2 - 3 * x + 1
def g (x : ℝ) := x + 2

theorem calculate_expression : f (1 + g 3) = 55 := 
by
  sorry

end calculate_expression_l991_99148


namespace scatter_plot_role_regression_analysis_l991_99164

theorem scatter_plot_role_regression_analysis :
  ∀ (role : String), 
  (role = "Finding the number of individuals" ∨ 
   role = "Comparing the size relationship of individual data" ∨ 
   role = "Exploring individual classification" ∨ 
   role = "Roughly judging whether variables are linearly related")
  → role = "Roughly judging whether variables are linearly related" :=
by
  intros role h
  sorry

end scatter_plot_role_regression_analysis_l991_99164


namespace prove_f_neg_a_l991_99107

noncomputable def f (x : ℝ) : ℝ := x + 1/x - 1

theorem prove_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = -4 :=
by
  sorry

end prove_f_neg_a_l991_99107


namespace albert_complete_laps_l991_99198

theorem albert_complete_laps (D L : ℝ) (I : ℕ) (hD : D = 256.5) (hL : L = 9.7) (hI : I = 6) :
  ⌊(D - I * L) / L⌋ = 20 :=
by
  sorry

end albert_complete_laps_l991_99198


namespace positive_difference_proof_l991_99132

noncomputable def solve_system : Prop :=
  ∃ (x y : ℝ), 
  (x + y = 40) ∧ 
  (3 * y - 4 * x = 10) ∧ 
  abs (y - x) = 8.58

theorem positive_difference_proof : solve_system := 
  sorry

end positive_difference_proof_l991_99132


namespace total_games_won_l991_99169

-- Define the number of games won by the Chicago Bulls
def bulls_games : ℕ := 70

-- Define the number of games won by the Miami Heat
def heat_games : ℕ := bulls_games + 5

-- Define the total number of games won by both the Bulls and the Heat
def total_games : ℕ := bulls_games + heat_games

-- The theorem stating that the total number of games won by both teams is 145
theorem total_games_won : total_games = 145 := by
  -- Proof is omitted
  sorry

end total_games_won_l991_99169


namespace value_range_of_2_sin_x_minus_1_l991_99160

theorem value_range_of_2_sin_x_minus_1 :
  (∀ x : ℝ, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1) →
  (∀ y : ℝ, y = 2 * Real.sin y - 1 → -3 ≤ y ∧ y ≤ 1) :=
by
  sorry

end value_range_of_2_sin_x_minus_1_l991_99160


namespace max_score_exam_l991_99129

theorem max_score_exam (Gibi_percent Jigi_percent Mike_percent Lizzy_percent : ℝ)
  (avg_score total_score M : ℝ) :
  Gibi_percent = 0.59 →
  Jigi_percent = 0.55 →
  Mike_percent = 0.99 →
  Lizzy_percent = 0.67 →
  avg_score = 490 →
  total_score = avg_score * 4 →
  total_score = (Gibi_percent + Jigi_percent + Mike_percent + Lizzy_percent) * M →
  M = 700 :=
by
  intros hGibi hJigi hMike hLizzy hAvg hTotalScore hEq
  sorry

end max_score_exam_l991_99129


namespace problem_solution_l991_99134

theorem problem_solution (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -5) :
  (1 / a) + (1 / b) = -3 / 5 :=
by
  sorry

end problem_solution_l991_99134


namespace javier_savings_l991_99150

theorem javier_savings (regular_price : ℕ) (discount1 : ℕ) (discount2 : ℕ) : 
  (regular_price = 50) 
  ∧ (discount1 = 40)
  ∧ (discount2 = 50) 
  → (30 = (100 * (regular_price * 3 - (regular_price + (regular_price * (100 - discount1) / 100) + regular_price / 2)) / (regular_price * 3))) :=
by
  intros h
  sorry

end javier_savings_l991_99150


namespace geometric_sequence_eighth_term_l991_99123

noncomputable def a_8 : ℕ :=
  let a₁ := 8
  let r := 2
  a₁ * r^(8-1)

theorem geometric_sequence_eighth_term : a_8 = 1024 := by
  sorry

end geometric_sequence_eighth_term_l991_99123


namespace increasing_f_for_x_ge_1_f_gt_1_for_x_gt_0_l991_99137

noncomputable def f (x : ℝ) := x ^ 2 * Real.exp x - Real.log x

theorem increasing_f_for_x_ge_1 : ∀ (x : ℝ), x ≥ 1 → ∀ y > x, f y > f x :=
by
  sorry

theorem f_gt_1_for_x_gt_0 : ∀ (x : ℝ), x > 0 → f x > 1 :=
by
  sorry

end increasing_f_for_x_ge_1_f_gt_1_for_x_gt_0_l991_99137


namespace correct_decision_box_l991_99168

theorem correct_decision_box (a b c : ℝ) (x : ℝ) : 
  x = a ∨ x = b → (x = b → b > a) →
  (c > x) ↔ (max (max a b) c = c) :=
by sorry

end correct_decision_box_l991_99168


namespace survey_households_selected_l991_99149

theorem survey_households_selected 
    (total_households : ℕ) 
    (middle_income_families : ℕ) 
    (low_income_families : ℕ) 
    (high_income_selected : ℕ)
    (total_high_income_families : ℕ)
    (total_selected_households : ℕ) 
    (H1 : total_households = 480)
    (H2 : middle_income_families = 200)
    (H3 : low_income_families = 160)
    (H4 : high_income_selected = 6)
    (H5 : total_high_income_families = total_households - (middle_income_families + low_income_families))
    (H6 : total_selected_households * total_high_income_families = high_income_selected * total_households) :
    total_selected_households = 24 :=
by
  -- The actual proof will go here:
  sorry

end survey_households_selected_l991_99149


namespace sarah_shampoo_and_conditioner_usage_l991_99136

-- Condition Definitions
def shampoo_daily_oz := 1
def conditioner_daily_oz := shampoo_daily_oz / 2
def total_daily_usage := shampoo_daily_oz + conditioner_daily_oz

def days_in_two_weeks := 14

-- Assertion: Total volume used in two weeks.
theorem sarah_shampoo_and_conditioner_usage :
  (days_in_two_weeks * total_daily_usage) = 21 := by
  sorry

end sarah_shampoo_and_conditioner_usage_l991_99136


namespace y_pow_one_div_x_neq_x_pow_y_l991_99157

theorem y_pow_one_div_x_neq_x_pow_y (t : ℝ) (ht : t > 1) : 
  let x := t ^ (2 / (t - 1))
  let y := t ^ (3 / (t - 1))
  (y ^ (1 / x) ≠ x ^ y) :=
by
  let x := t ^ (2 / (t - 1))
  let y := t ^ (3 / (t - 1))
  sorry

end y_pow_one_div_x_neq_x_pow_y_l991_99157


namespace sign_of_f_based_on_C_l991_99163

def is_triangle (a b c : ℝ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem sign_of_f_based_on_C (a b c : ℝ) (R r : ℝ) (A B C : ℝ)
  (h1 : a = 2 * R * Real.sin A) 
  (h2 : b = 2 * R * Real.sin B) 
  (h3 : c = 2 * R * Real.sin C)
  (h4 : r = 4 * R * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2))
  (h5 : A + B + C = Real.pi)
  (h_triangle : is_triangle a b c)
  : (a + b - 2 * R - 2 * r > 0 ↔ C < Real.pi / 2) ∧
    (a + b - 2 * R - 2 * r = 0 ↔ C = Real.pi / 2) ∧
    (a + b - 2 * R - 2 * r < 0 ↔ C > Real.pi / 2) :=
sorry

end sign_of_f_based_on_C_l991_99163


namespace calculate_expression_value_l991_99152

theorem calculate_expression_value (x y : ℚ) (hx : x = 4 / 7) (hy : y = 5 / 8) :
  (7 * x + 5 * y) / (70 * x * y) = 57 / 400 := by
  sorry

end calculate_expression_value_l991_99152


namespace max_value_of_x_times_one_minus_2x_l991_99112

theorem max_value_of_x_times_one_minus_2x : 
  ∀ x : ℝ, 0 < x ∧ x < 1 / 2 → x * (1 - 2 * x) ≤ 1 / 8 :=
by
  intro x 
  intro hx
  sorry

end max_value_of_x_times_one_minus_2x_l991_99112


namespace sum_of_primes_less_than_10_is_17_l991_99153

-- Definition of prime numbers less than 10
def primes_less_than_10 : List ℕ := [2, 3, 5, 7]

-- Sum of the prime numbers less than 10
def sum_primes_less_than_10 : ℕ := List.sum primes_less_than_10

theorem sum_of_primes_less_than_10_is_17 : sum_primes_less_than_10 = 17 := 
by
  sorry

end sum_of_primes_less_than_10_is_17_l991_99153


namespace books_already_read_l991_99199

def total_books : ℕ := 20
def unread_books : ℕ := 5

theorem books_already_read : (total_books - unread_books = 15) :=
by
 -- Proof goes here
 sorry

end books_already_read_l991_99199


namespace factor_is_two_l991_99195

theorem factor_is_two (n f : ℤ) (h1 : n = 121) (h2 : n * f - 140 = 102) : f = 2 :=
by
  sorry

end factor_is_two_l991_99195


namespace pairs_xy_solution_sum_l991_99197

theorem pairs_xy_solution_sum :
  ∃ (x y : ℝ) (a b c d : ℕ), 
    x + y = 5 ∧ 2 * x * y = 5 ∧ 
    (x = (5 + Real.sqrt 15) / 2 ∨ x = (5 - Real.sqrt 15) / 2) ∧ 
    a = 5 ∧ b = 1 ∧ c = 15 ∧ d = 2 ∧ a + b + c + d = 23 :=
by
  sorry

end pairs_xy_solution_sum_l991_99197


namespace MinTransportCost_l991_99159

noncomputable def TruckTransportOptimization :=
  ∃ (x y : ℕ), x + y = 6 ∧ 45 * x + 30 * y ≥ 240 ∧ 400 * x + 300 * y ≤ 2300 ∧ (∃ (min_cost : ℕ), min_cost = 2200 ∧ x = 4 ∧ y = 2)
  
theorem MinTransportCost : TruckTransportOptimization :=
sorry

end MinTransportCost_l991_99159


namespace haley_laundry_loads_l991_99193

theorem haley_laundry_loads (shirts sweaters pants socks : ℕ) 
    (machine_capacity total_pieces : ℕ)
    (sum_of_clothing : 6 + 28 + 10 + 9 = total_pieces)
    (machine_capacity_eq : machine_capacity = 5) :
  ⌈(total_pieces:ℚ) / machine_capacity⌉ = 11 :=
by
  sorry

end haley_laundry_loads_l991_99193


namespace x_squared_eq_y_squared_iff_x_eq_y_or_neg_y_x_squared_eq_y_squared_necessary_but_not_sufficient_l991_99117

theorem x_squared_eq_y_squared_iff_x_eq_y_or_neg_y (x y : ℝ) : 
  (x^2 = y^2) ↔ (x = y ∨ x = -y) := by
  sorry

theorem x_squared_eq_y_squared_necessary_but_not_sufficient (x y : ℝ) :
  (x^2 = y^2 → x = y) ↔ false := by
  sorry

end x_squared_eq_y_squared_iff_x_eq_y_or_neg_y_x_squared_eq_y_squared_necessary_but_not_sufficient_l991_99117


namespace abs_difference_extrema_l991_99103

theorem abs_difference_extrema (x : ℝ) (h : 2 ≤ x ∧ x < 3) :
  max (|x-2| + |x-3| - |x-1|) = 0 ∧ min (|x-2| + |x-3| - |x-1|) = -1 :=
by
  sorry

end abs_difference_extrema_l991_99103


namespace percent_calculation_l991_99118

theorem percent_calculation (x : ℝ) : 
  (∃ y : ℝ, y / 100 * x = 0.3 * 0.7 * x) → ∃ y : ℝ, y = 21 :=
by
  sorry

end percent_calculation_l991_99118


namespace hans_room_count_l991_99151

theorem hans_room_count :
  let total_floors := 10
  let rooms_per_floor := 10
  let unavailable_floors := 1
  let available_floors := total_floors - unavailable_floors
  available_floors * rooms_per_floor = 90 := by
  let total_floors := 10
  let rooms_per_floor := 10
  let unavailable_floors := 1
  let available_floors := total_floors - unavailable_floors
  show available_floors * rooms_per_floor = 90
  sorry

end hans_room_count_l991_99151


namespace prove_a2_l991_99190

def arithmetic_seq (a d : ℕ → ℝ) : Prop :=
  ∀ n m, a n + d (n - m) = a m

theorem prove_a2 (a : ℕ → ℝ) (d : ℕ → ℝ) :
  (∀ n, a n = a 0 + (n - 1) * 2) → 
  (a 1 + 4) / a 1 = (a 1 + 6) / (a 1 + 4) →
  (d 1 = 2) →
  a 2 = -6 :=
by
  intros h_seq h_geo h_common_diff
  sorry

end prove_a2_l991_99190


namespace vance_family_stamp_cost_difference_l991_99126

theorem vance_family_stamp_cost_difference :
    let cost_rooster := 2 * 1.50
    let cost_daffodil := 5 * 0.75
    cost_daffodil - cost_rooster = 0.75 :=
by
    let cost_rooster := 2 * 1.50
    let cost_daffodil := 5 * 0.75
    show cost_daffodil - cost_rooster = 0.75
    sorry

end vance_family_stamp_cost_difference_l991_99126


namespace nat_power_of_p_iff_only_prime_factor_l991_99176

theorem nat_power_of_p_iff_only_prime_factor (p n : ℕ) (hp : Nat.Prime p) :
  (∃ k : ℕ, n = p^k) ↔ (∀ q : ℕ, Nat.Prime q → q ∣ n → q = p) := 
sorry

end nat_power_of_p_iff_only_prime_factor_l991_99176


namespace reduced_travel_time_l991_99172

-- Definition of conditions as given in part a)
def initial_speed := 48 -- km/h
def initial_time := 50/60 -- hours (50 minutes)
def required_speed := 60 -- km/h
def reduced_time := 40/60 -- hours (40 minutes)

-- Problem statement
theorem reduced_travel_time :
  ∃ t2, (initial_speed * initial_time = required_speed * t2) ∧ (t2 = reduced_time) :=
by
  sorry

end reduced_travel_time_l991_99172


namespace probability_of_green_ball_l991_99167

def total_balls : ℕ := 3 + 3 + 6
def green_balls : ℕ := 3

theorem probability_of_green_ball : (green_balls : ℚ) / total_balls = 1 / 4 :=
by
  sorry

end probability_of_green_ball_l991_99167


namespace probability_same_color_given_first_red_l991_99143

-- Definitions of events
def event_A (draw1 : ℕ) : Prop := draw1 = 1 -- Event A: the first ball drawn is red (drawing 1 means the first ball is red)

def event_B (draw1 draw2 : ℕ) : Prop := -- Event B: the two balls drawn are of the same color
  (draw1 = 1 ∧ draw2 = 1) ∨ (draw1 = 2 ∧ draw2 = 2)

-- Given probabilities
def P_A : ℚ := 2 / 5
def P_AB : ℚ := (2 / 5) * (1 / 4)

-- The conditional probability P(B|A)
def P_B_given_A : ℚ := P_AB / P_A

theorem probability_same_color_given_first_red : P_B_given_A = 1 / 4 := 
by 
  unfold P_B_given_A P_A P_AB
  sorry

end probability_same_color_given_first_red_l991_99143


namespace find_y_l991_99182

theorem find_y (x y : ℝ) (h1 : x = 100) (h2 : x^3 * y - 3 * x^2 * y + 3 * x * y = 3000000) : 
  y = 3000000 / (100^3 - 3 * 100^2 + 3 * 100 * 1) :=
by sorry

end find_y_l991_99182


namespace correct_average_marks_l991_99174

theorem correct_average_marks :
  let num_students := 40
  let reported_avg := 65
  let incorrect_marks := [100, 85, 15]
  let correct_marks := [20, 50, 55]
  let incorrect_total_sum := num_students * reported_avg
  let wrong_sum := List.sum incorrect_marks
  let correct_sum := List.sum correct_marks
  let correct_total_sum := incorrect_total_sum - wrong_sum + correct_sum
  let correct_avg := (correct_total_sum : ℚ) / num_students
  correct_avg = 63.125 :=
by
  let num_students := 40
  let reported_avg := 65
  let incorrect_marks := [100, 85, 15]
  let correct_marks := [20, 50, 55]
  let incorrect_total_sum := num_students * reported_avg
  let wrong_sum := List.sum incorrect_marks
  let correct_sum := List.sum correct_marks
  let correct_total_sum := incorrect_total_sum - wrong_sum + correct_sum
  let correct_avg := (correct_total_sum : ℚ) / num_students
  sorry

end correct_average_marks_l991_99174


namespace find_x_for_g_inv_l991_99140

def g (x : ℝ) : ℝ := 5 * x ^ 3 - 4 * x + 1

theorem find_x_for_g_inv (x : ℝ) (h : g 3 = x) : g⁻¹ 3 = 3 :=
by
  sorry

end find_x_for_g_inv_l991_99140


namespace width_of_jesses_room_l991_99180

theorem width_of_jesses_room (length : ℝ) (tile_area : ℝ) (num_tiles : ℕ) (total_area : ℝ) (width : ℝ) :
  length = 2 → tile_area = 4 → num_tiles = 6 → total_area = (num_tiles * tile_area : ℝ) → (length * width) = total_area → width = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end width_of_jesses_room_l991_99180


namespace ratio_areas_l991_99114

theorem ratio_areas (H : ℝ) (L : ℝ) (r : ℝ) (A_rectangle : ℝ) (A_circle : ℝ) :
  H = 45 ∧ (L / H = 4 / 3) ∧ r = H / 2 ∧ A_rectangle = L * H ∧ A_circle = π * r^2 →
  (A_rectangle / A_circle = 17 / π) :=
by
  sorry

end ratio_areas_l991_99114


namespace wholesale_prices_l991_99166

-- Definitions for the problem conditions
variable (p1 p2 d k : ℝ)
variable (h_d : d > 0)
variable (h_k : k > 1)
variable (prices : Finset ℝ)
variable (h_prices : prices = {64, 64, 70, 72})

-- The theorem statement to prove
theorem wholesale_prices :
  ∃ p1 p2, (p1 + d ∈ prices ∧ k * p1 ∈ prices) ∧ 
           (p2 + d ∈ prices ∧ k * p2 ∈ prices) ∧ 
           p1 ≠ p2
:= sorry

end wholesale_prices_l991_99166


namespace problem_statement_l991_99192

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x
noncomputable def F (x : ℝ) : ℝ := f x - g x
noncomputable def m (x x₀ : ℝ) : ℝ := if x ≤ x₀ then f x else g x

-- Statement of the theorem
theorem problem_statement (x₀ x₁ x₂ n : ℝ) (hx₀ : x₀ ∈ Set.Ioo 1 2)
  (hF_root : F x₀ = 0)
  (hm_roots : m x₁ x₀ = n ∧ m x₂ x₀ = n ∧ 1 < x₁ ∧ x₁ < x₀ ∧ x₀ < x₂) :
  x₁ + x₂ > 2 * x₀ :=
sorry

end problem_statement_l991_99192


namespace hire_charges_paid_by_b_l991_99173

theorem hire_charges_paid_by_b (total_cost : ℕ) (hours_a : ℕ) (hours_b : ℕ) (hours_c : ℕ) 
  (total_hours : ℕ) (cost_per_hour : ℕ) : 
  total_cost = 520 → hours_a = 7 → hours_b = 8 → hours_c = 11 → total_hours = hours_a + hours_b + hours_c 
  → cost_per_hour = total_cost / total_hours → 
  (hours_b * cost_per_hour) = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end hire_charges_paid_by_b_l991_99173


namespace algebraic_expression_value_l991_99109

variables (x y : ℝ)

theorem algebraic_expression_value :
  x^2 - 4 * x - 1 = 0 →
  (2 * x - 3)^2 - (x + y) * (x - y) - y^2 = 12 :=
by
  intro h
  sorry

end algebraic_expression_value_l991_99109


namespace ROI_difference_l991_99171

-- Definitions based on the conditions
def Emma_investment : ℝ := 300
def Briana_investment : ℝ := 500
def Emma_yield : ℝ := 0.15
def Briana_yield : ℝ := 0.10
def years : ℕ := 2

-- The goal is to prove that the difference between their 2-year ROI is $10
theorem ROI_difference :
  let Emma_ROI := Emma_investment * Emma_yield * years
  let Briana_ROI := Briana_investment * Briana_yield * years
  (Briana_ROI - Emma_ROI) = 10 :=
by
  sorry

end ROI_difference_l991_99171


namespace even_function_a_value_l991_99146

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (x^2 + (a^2 - 1) * x + (a - 1)) = ((-x)^2 + (a^2 - 1) * (-x) + (a - 1))) → (a = 1 ∨ a = -1) :=
by
  sorry

end even_function_a_value_l991_99146


namespace part1_part2_l991_99161

-- Define the coordinates of point P as functions of n
def pointP (n : ℝ) : ℝ × ℝ := (n + 3, 2 - 3 * n)

-- Condition 1: Point P is in the fourth quadrant
def inFourthQuadrant (n : ℝ) : Prop :=
  let point := pointP n
  point.1 > 0 ∧ point.2 < 0

-- Condition 2: Distance from P to the x-axis is 1 greater than the distance to the y-axis
def distancesCondition (n : ℝ) : Prop :=
  abs (2 - 3 * n) + 1 = abs (n + 3)

-- Definition of point Q
def pointQ (n : ℝ) : ℝ × ℝ := (n, -4)

-- Condition 3: PQ is parallel to the x-axis
def pqParallelX (n : ℝ) : Prop :=
  (pointP n).2 = (pointQ n).2

-- Theorems to prove the coordinates of point P and the length of PQ
theorem part1 (n : ℝ) (h1 : inFourthQuadrant n) (h2 : distancesCondition n) : pointP n = (6, -7) :=
sorry

theorem part2 (n : ℝ) (h1 : pqParallelX n) : abs ((pointP n).1 - (pointQ n).1) = 3 :=
sorry

end part1_part2_l991_99161


namespace smallest_b_for_factors_l991_99121

theorem smallest_b_for_factors (b : ℕ) (h : ∃ r s : ℤ, (x : ℤ) → (x + r) * (x + s) = x^2 + ↑b * x + 2016 ∧ r * s = 2016) :
  b = 90 :=
by
  sorry

end smallest_b_for_factors_l991_99121


namespace Jake_needs_to_lose_12_pounds_l991_99122

theorem Jake_needs_to_lose_12_pounds (J S : ℕ) (h1 : J + S = 156) (h2 : J = 108) : J - 2 * S = 12 := by
  sorry

end Jake_needs_to_lose_12_pounds_l991_99122

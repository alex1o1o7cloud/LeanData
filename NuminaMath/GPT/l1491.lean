import Mathlib

namespace NataliesSisterInitialDiaries_l1491_149186

theorem NataliesSisterInitialDiaries (D : ℕ)
  (h1 : 2 * D - (1 / 4) * 2 * D = 18) : D = 12 :=
by sorry

end NataliesSisterInitialDiaries_l1491_149186


namespace largest_triangle_angle_l1491_149196

theorem largest_triangle_angle (h_ratio : ∃ (a b c : ℕ), a / b = 3 / 4 ∧ b / c = 4 / 9) 
  (h_external_angle : ∃ (θ1 θ2 θ3 θ4 : ℝ), θ1 = 3 * x ∧ θ2 = 4 * x ∧ θ3 = 9 * x ∧ θ4 = 3 * x ∧ θ1 + θ2 + θ3 = 180) :
  ∃ (θ3 : ℝ), θ3 = 101.25 := by
  sorry

end largest_triangle_angle_l1491_149196


namespace find_t_l1491_149110

variable (t : ℚ)

def point_on_line (p1 p2 p3 : ℚ × ℚ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_t (t : ℚ) : point_on_line (3, 0) (0, 7) (t, 8) → t = -3 / 7 := by
  sorry

end find_t_l1491_149110


namespace total_registration_methods_l1491_149192

theorem total_registration_methods (n : ℕ) (h : n = 5) : (2 ^ n) = 32 :=
by
  sorry

end total_registration_methods_l1491_149192


namespace parabola_line_non_intersect_l1491_149112

theorem parabola_line_non_intersect (r s : ℝ) (Q : ℝ × ℝ) (P : ℝ → ℝ)
  (hP : ∀ x, P x = x^2)
  (hQ : Q = (10, 6))
  (h_cond : ∀ m : ℝ, ¬∃ x : ℝ, (Q.snd - 6 = m * (Q.fst - 10)) ∧ (P x = x^2) ↔ r < m ∧ m < s) :
  r + s = 40 :=
sorry

end parabola_line_non_intersect_l1491_149112


namespace arithmetic_sequence_a6_l1491_149160

theorem arithmetic_sequence_a6 (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) : a 6 = 16 :=
sorry

end arithmetic_sequence_a6_l1491_149160


namespace positive_number_representation_l1491_149172

theorem positive_number_representation (a : ℝ) : 
  (a > 0) ↔ (a ≠ 0 ∧ a > 0 ∧ ¬(a < 0)) :=
by 
  sorry

end positive_number_representation_l1491_149172


namespace cost_of_parakeet_l1491_149115

theorem cost_of_parakeet
  (P Py K : ℕ) -- defining the costs of parakeet, puppy, and kitten
  (h1 : Py = 3 * P) -- puppy is three times the cost of parakeet
  (h2 : P = K / 2) -- parakeet is half the cost of kitten
  (h3 : 2 * Py + 2 * K + 3 * P = 130) -- total cost equation
  : P = 10 := 
sorry

end cost_of_parakeet_l1491_149115


namespace output_in_scientific_notation_l1491_149191

def output_kilowatt_hours : ℝ := 448000
def scientific_notation (n : ℝ) : Prop := n = 4.48 * 10^5

theorem output_in_scientific_notation : scientific_notation output_kilowatt_hours :=
by
  -- Proof steps are not required
  sorry

end output_in_scientific_notation_l1491_149191


namespace condition_an_necessary_but_not_sufficient_l1491_149108

-- Definitions for the sequence and properties
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, n ≥ 1 → a (n + 1) = r * (a n)

def condition_an (a : ℕ → ℝ) : Prop :=
  ∀ n, n ≥ 2 → a n = 2 * a (n - 1)

-- The theorem statement
theorem condition_an_necessary_but_not_sufficient (a : ℕ → ℝ) :
  (∀ n, n ≥ 1 → a (n + 1) = 2 * (a n)) → (condition_an a) ∧ ¬(is_geometric_sequence a 2) :=
by
  sorry

end condition_an_necessary_but_not_sufficient_l1491_149108


namespace smallest_a_for_5880_to_be_cube_l1491_149178

theorem smallest_a_for_5880_to_be_cube : ∃ (a : ℕ), a > 0 ∧ (∃ (k : ℕ), 5880 * a = k ^ 3) ∧
  (∀ (b : ℕ), b > 0 ∧ (∃ (k : ℕ), 5880 * b = k ^ 3) → a ≤ b) ∧ a = 1575 :=
sorry

end smallest_a_for_5880_to_be_cube_l1491_149178


namespace sulfuric_acid_reaction_l1491_149161

theorem sulfuric_acid_reaction (SO₃ H₂O H₂SO₄ : ℕ) 
  (reaction : SO₃ + H₂O = H₂SO₄)
  (H₂O_eq : H₂O = 2)
  (H₂SO₄_eq : H₂SO₄ = 2) :
  SO₃ = 2 :=
by
  sorry

end sulfuric_acid_reaction_l1491_149161


namespace problem_statement_l1491_149176

theorem problem_statement : 2456 + 144 / 12 * 5 - 256 = 2260 := 
by
  -- statements and proof steps would go here
  sorry

end problem_statement_l1491_149176


namespace largest_angle_of_consecutive_integers_hexagon_l1491_149163

theorem largest_angle_of_consecutive_integers_hexagon (a b c d e f : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : e < f) 
  (h6 : a + b + c + d + e + f = 720) : 
  ∃ x, f = x + 2 ∧ (x + 2 = 122.5) :=
  sorry

end largest_angle_of_consecutive_integers_hexagon_l1491_149163


namespace original_total_movies_is_293_l1491_149171

noncomputable def original_movies (dvd_to_bluray_ratio : ℕ × ℕ) (additional_blurays : ℕ) (new_ratio : ℕ × ℕ) : ℕ :=
  let original_dvds := dvd_to_bluray_ratio.1
  let original_blurays := dvd_to_bluray_ratio.2
  let added_blurays := additional_blurays
  let new_dvds := new_ratio.1
  let new_blurays := new_ratio.2
  let x := (new_dvds * original_blurays - new_blurays * original_dvds) / (new_blurays * original_dvds - added_blurays * original_blurays)
  let total_movies := (original_dvds * x + original_blurays * x)
  let blurays_after_purchase := original_blurays * x + added_blurays

  if (new_dvds * (original_blurays * x + added_blurays) = new_blurays * (original_dvds * x))
  then 
    (original_dvds * x + original_blurays * x)
  else
    0 -- This case should theoretically never happen if the input ratios are consistent.

theorem original_total_movies_is_293 : original_movies (7, 2) 5 (13, 4) = 293 :=
by sorry

end original_total_movies_is_293_l1491_149171


namespace triangle_side_lengths_expression_neg_l1491_149170

theorem triangle_side_lengths_expression_neg {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2 * a^2 * b^2 - 2 * b^2 * c^2 - 2 * c^2 * a^2 < 0 := 
by 
  sorry

end triangle_side_lengths_expression_neg_l1491_149170


namespace molecular_weight_N2O5_l1491_149184

theorem molecular_weight_N2O5 :
  let atomic_weight_N := 14.01
  let atomic_weight_O := 16.00
  let molecular_weight_N2O5 := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  molecular_weight_N2O5 = 108.02 := 
by
  sorry

end molecular_weight_N2O5_l1491_149184


namespace sum_of_integral_c_l1491_149180

theorem sum_of_integral_c :
  let discriminant (a b c : ℤ) := b * b - 4 * a * c
  ∃ (valid_c : List ℤ),
    (∀ c ∈ valid_c, c ≤ 30 ∧ ∃ k : ℤ, discriminant 1 (-9) (c) = k * k ∧ k > 0) ∧
    valid_c.sum = 32 := 
by
  sorry

end sum_of_integral_c_l1491_149180


namespace translation_result_l1491_149145

-- Define the initial point A
def A : (ℤ × ℤ) := (-2, 3)

-- Define the translation function
def translate (p : (ℤ × ℤ)) (delta_x delta_y : ℤ) : (ℤ × ℤ) :=
  (p.1 + delta_x, p.2 - delta_y)

-- The theorem stating the resulting point after translation
theorem translation_result :
  translate A 3 1 = (1, 2) :=
by
  -- Skipping proof with sorry
  sorry

end translation_result_l1491_149145


namespace find_train_speed_l1491_149143

-- Define the given conditions
def train_length : ℕ := 2500  -- length of the train in meters
def time_to_cross_pole : ℕ := 100  -- time to cross the pole in seconds

-- Define the expected speed
def expected_speed : ℕ := 25  -- expected speed in meters per second

-- The theorem we need to prove
theorem find_train_speed : 
  (train_length / time_to_cross_pole) = expected_speed := 
by 
  sorry

end find_train_speed_l1491_149143


namespace value_of_Y_is_669_l1491_149147

theorem value_of_Y_is_669 :
  let A := 3009 / 3
  let B := A / 3
  let Y := A - B
  Y = 669 :=
by
  sorry

end value_of_Y_is_669_l1491_149147


namespace divisible_by_8640_l1491_149148

theorem divisible_by_8640 (x : ℤ) : 8640 ∣ (x^9 - 6 * x^7 + 9 * x^5 - 4 * x^3) :=
  sorry

end divisible_by_8640_l1491_149148


namespace no_consecutive_squares_l1491_149118

open Nat

-- Define a function to get the n-th prime number
def prime (n : ℕ) : ℕ := sorry -- Use an actual function or sequence that generates prime numbers, this is a placeholder.

-- Define the sequence S_n, the sum of the first n prime numbers
def S : ℕ → ℕ
| 0       => 0
| (n + 1) => S n + prime (n + 1)

-- Define a predicate to check if a number is a perfect square
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- The theorem that no two consecutive terms S_{n-1} and S_n can both be perfect squares
theorem no_consecutive_squares (n : ℕ) : ¬ (is_square (S n) ∧ is_square (S (n + 1))) :=
by
  sorry

end no_consecutive_squares_l1491_149118


namespace card_probability_l1491_149127

theorem card_probability :
  let total_cards := 52
  let hearts := 13
  let clubs := 13
  let spades := 13
  let prob_heart_first := hearts / total_cards
  let remaining_after_heart := total_cards - 1
  let prob_club_second := clubs / remaining_after_heart
  let remaining_after_heart_and_club := remaining_after_heart - 1
  let prob_spade_third := spades / remaining_after_heart_and_club
  (prob_heart_first * prob_club_second * prob_spade_third) = (2197 / 132600) :=
  sorry

end card_probability_l1491_149127


namespace neg_or_false_implies_or_true_l1491_149120

theorem neg_or_false_implies_or_true (p q : Prop) (h : ¬(p ∨ q) = False) : p ∨ q :=
by {
  sorry
}

end neg_or_false_implies_or_true_l1491_149120


namespace number_of_nickels_l1491_149134

-- Define the conditions
variable (m : ℕ) -- Total number of coins initially
variable (v : ℕ) -- Total value of coins initially in cents
variable (n : ℕ) -- Number of nickels

-- State the conditions in terms of mathematical equations
-- Condition 1: Average value is 25 cents
axiom avg_value_initial : v = 25 * m

-- Condition 2: Adding one half-dollar (50 cents) results in average of 26 cents
axiom avg_value_after_half_dollar : v + 50 = 26 * (m + 1)

-- Define the relationship between the number of each type of coin and the total value
-- We sum the individual products of the count of each type and their respective values
axiom total_value_definition : v = 5 * n  -- since the problem already validates with total_value == 25m

-- Question to prove
theorem number_of_nickels : n = 30 :=
by
  -- Since we are not providing proof, we will use sorry to indicate the proof is omitted
  sorry

end number_of_nickels_l1491_149134


namespace total_balloons_l1491_149141

theorem total_balloons (sam_balloons_initial mary_balloons fred_balloons : ℕ) (h1 : sam_balloons_initial = 6)
    (h2 : mary_balloons = 7) (h3 : fred_balloons = 5) : sam_balloons_initial - fred_balloons + mary_balloons = 8 :=
by
  sorry

end total_balloons_l1491_149141


namespace find_f6_l1491_149135

-- Define the function f
variable {f : ℝ → ℝ}
-- The function satisfies f(x + y) = f(x) + f(y) for all real numbers x and y
axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
-- f(4) = 6
axiom f_of_4 : f 4 = 6

theorem find_f6 : f 6 = 9 :=
by
    sorry

end find_f6_l1491_149135


namespace solve_for_x_l1491_149137

variable {x : ℝ}

theorem solve_for_x (h : (4 * x ^ 2 - 3 * x + 2) / (x + 2) = 4 * x - 3) : x = 1 :=
sorry

end solve_for_x_l1491_149137


namespace arrange_in_order_l1491_149128

noncomputable def a := (Real.sqrt 2 / 2) * (Real.sin (17 * Real.pi / 180) + Real.cos (17 * Real.pi / 180))
noncomputable def b := 2 * (Real.cos (13 * Real.pi / 180))^2 - 1
noncomputable def c := Real.sqrt 3 / 2

theorem arrange_in_order : c < a ∧ a < b := 
by
  sorry

end arrange_in_order_l1491_149128


namespace gcd_840_1764_gcd_561_255_l1491_149162

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := 
by
  sorry

theorem gcd_561_255 : Nat.gcd 561 255 = 51 :=
by
  sorry

end gcd_840_1764_gcd_561_255_l1491_149162


namespace problem1_problem2_l1491_149150

-- Problem 1
theorem problem1 (x : ℝ) (h1 : 2 * x > 1 - x) (h2 : x + 2 < 4 * x - 1) : x > 1 := 
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ)
  (h1 : (2 / 3) * x + 5 > 1 - x)
  (h2 : x - 1 ≤ (3 / 4) * x - 1 / 8) :
  -12 / 5 < x ∧ x ≤ 7 / 2 := 
by
  sorry

end problem1_problem2_l1491_149150


namespace Sally_seashells_l1491_149154

/- Definitions -/
def Tom_seashells : Nat := 7
def Jessica_seashells : Nat := 5
def total_seashells : Nat := 21

/- Theorem statement -/
theorem Sally_seashells : total_seashells - (Tom_seashells + Jessica_seashells) = 9 := by
  -- Definitions of seashells found by Tom, Jessica and the total should be used here
  -- Proving the theorem
  sorry

end Sally_seashells_l1491_149154


namespace sum_of_squares_of_roots_eq_zero_l1491_149188

theorem sum_of_squares_of_roots_eq_zero :
  let f : Polynomial ℝ := Polynomial.C 50 + Polynomial.monomial 3 (-2) + Polynomial.monomial 7 5 + Polynomial.monomial 10 1
  ∀ (r : ℝ), r ∈ Multiset.toFinset f.roots → r ^ 2 = 0 :=
by
  sorry

end sum_of_squares_of_roots_eq_zero_l1491_149188


namespace pow_100_mod_18_l1491_149174

theorem pow_100_mod_18 : (5 ^ 100) % 18 = 13 := by
  -- Define the conditions
  have h1 : (5 ^ 1) % 18 = 5 := by norm_num
  have h2 : (5 ^ 2) % 18 = 7 := by norm_num
  have h3 : (5 ^ 3) % 18 = 17 := by norm_num
  have h4 : (5 ^ 4) % 18 = 13 := by norm_num
  have h5 : (5 ^ 5) % 18 = 11 := by norm_num
  have h6 : (5 ^ 6) % 18 = 1 := by norm_num
  
  -- The required theorem is based on the conditions mentioned
  sorry

end pow_100_mod_18_l1491_149174


namespace minimum_value_expr_min_value_reachable_l1491_149132

noncomputable def expr (x y : ℝ) : ℝ :=
  4 * x^2 + 9 * y^2 + 16 / x^2 + 6 * y / x

theorem minimum_value_expr (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  expr x y ≥ (2 * Real.sqrt 564) / 3 :=
sorry

theorem min_value_reachable :
  ∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ expr x y = (2 * Real.sqrt 564) / 3 :=
sorry

end minimum_value_expr_min_value_reachable_l1491_149132


namespace max_quotient_l1491_149159

theorem max_quotient (a b : ℝ) (h1 : 300 ≤ a) (h2 : a ≤ 500) (h3 : 900 ≤ b) (h4 : b ≤ 1800) :
  ∃ (q : ℝ), q = 5 / 9 ∧ (∀ (x y : ℝ), (300 ≤ x ∧ x ≤ 500) ∧ (900 ≤ y ∧ y ≤ 1800) → (x / y ≤ q)) :=
by
  use 5 / 9
  sorry

end max_quotient_l1491_149159


namespace probability_same_color_shoes_l1491_149166

theorem probability_same_color_shoes (pairs : ℕ) (total_shoes : ℕ)
  (each_pair_diff_color : pairs * 2 = total_shoes)
  (select_2_without_replacement : total_shoes = 10 ∧ pairs = 5) :
  let successful_outcomes := pairs
  let total_outcomes := (total_shoes * (total_shoes - 1)) / 2
  successful_outcomes / total_outcomes = 1 / 9 :=
by
  sorry

end probability_same_color_shoes_l1491_149166


namespace tangent_values_l1491_149169

theorem tangent_values (A : ℝ) (h : A < π) (cos_A : Real.cos A = 3 / 5) :
  Real.tan A = 4 / 3 ∧ Real.tan (A + π / 4) = -7 := 
by
  sorry

end tangent_values_l1491_149169


namespace trapezoid_area_l1491_149130

theorem trapezoid_area (AD BC : ℝ) (AD_eq : AD = 18) (BC_eq : BC = 2) (CD : ℝ) (h : CD = 10): 
  ∃ (CH : ℝ), CH = 6 ∧ (1 / 2) * (AD + BC) * CH = 60 :=
by
  sorry

end trapezoid_area_l1491_149130


namespace distinct_equilateral_triangles_in_polygon_l1491_149100

noncomputable def num_distinct_equilateral_triangles (P : Finset (Fin 10)) : Nat :=
  90

theorem distinct_equilateral_triangles_in_polygon (P : Finset (Fin 10)) :
  P.card = 10 →
  num_distinct_equilateral_triangles P = 90 :=
by
  intros
  sorry

end distinct_equilateral_triangles_in_polygon_l1491_149100


namespace smallest_possible_value_l1491_149175

theorem smallest_possible_value (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * ((1 / (a + b)) + (1 / (a + c)) + (1 / (b + d)) + (1 / (c + d))) ≥ 8 := 
sorry

end smallest_possible_value_l1491_149175


namespace total_complaints_l1491_149113

-- Conditions as Lean definitions
def normal_complaints : ℕ := 120
def short_staffed_20 (c : ℕ) := c + c / 3
def short_staffed_40 (c : ℕ) := c + 2 * c / 3
def self_checkout_partial (c : ℕ) := c + c / 10
def self_checkout_complete (c : ℕ) := c + c / 5
def day1_complaints : ℕ := normal_complaints + normal_complaints / 3 + normal_complaints / 5
def day2_complaints : ℕ := normal_complaints + 2 * normal_complaints / 3 + normal_complaints / 10
def day3_complaints : ℕ := normal_complaints + 2 * normal_complaints / 3 + normal_complaints / 5

-- Prove the total complaints
theorem total_complaints : day1_complaints + day2_complaints + day3_complaints = 620 :=
by
  sorry

end total_complaints_l1491_149113


namespace circle_radius_range_l1491_149114

theorem circle_radius_range (r : ℝ) : 
  (∃ P₁ P₂ : ℝ × ℝ, (P₁.2 = 1 ∨ P₁.2 = -1) ∧ (P₂.2 = 1 ∨ P₂.2 = -1) ∧ 
  (P₁.1 - 3) ^ 2 + (P₁.2 + 5) ^ 2 = r^2 ∧ (P₂.1 - 3) ^ 2 + (P₂.2 + 5) ^ 2 = r^2) → (4 < r ∧ r < 6) :=
by
  sorry

end circle_radius_range_l1491_149114


namespace simplify_expression_l1491_149156

theorem simplify_expression (x : ℝ) :
  ((3 * x^2 + 2 * x - 1) + 2 * x^2) * 4 + (5 - 2 / 2) * (3 * x^2 + 6 * x - 8) = 32 * x^2 + 32 * x - 36 :=
sorry

end simplify_expression_l1491_149156


namespace matchsticks_for_3_by_1996_grid_l1491_149187

def total_matchsticks_needed (rows cols : ℕ) : ℕ :=
  (cols * (rows + 1)) + (rows * (cols + 1))

theorem matchsticks_for_3_by_1996_grid : total_matchsticks_needed 3 1996 = 13975 := by
  sorry

end matchsticks_for_3_by_1996_grid_l1491_149187


namespace value_of_a7_l1491_149181

-- Define an arithmetic sequence
structure ArithmeticSeq (a : Nat → ℤ) :=
  (d : ℤ)
  (a_eq : ∀ n, a (n+1) = a n + d)

-- Lean statement of the equivalent proof problem
theorem value_of_a7 (a : ℕ → ℤ) (H : ArithmeticSeq a) :
  (2 * a 4 - a 7 ^ 2 + 2 * a 10 = 0) → a 7 = 4 * H.d :=
by
  sorry

end value_of_a7_l1491_149181


namespace tangent_lines_through_point_l1491_149183

theorem tangent_lines_through_point (x y : ℝ) (hp : (x, y) = (3, 1))
 : ∃ (a b c : ℝ), (y - 1 = (4 / 3) * (x - 3) ∨ x = 3) :=
by
  sorry

end tangent_lines_through_point_l1491_149183


namespace maximum_m_value_l1491_149104

theorem maximum_m_value (a : ℕ → ℤ) (m : ℕ) :
  (∀ n, a (n + 1) - a n = 3) →
  a 3 = -2 →
  (∀ k : ℕ, k ≥ 4 → (3 * k - 8) * (3 * k - 5) / (3 * k - 11) ≥ 3 * m - 11) →
  m ≤ 9 :=
by
  sorry

end maximum_m_value_l1491_149104


namespace lcm_of_18_and_24_l1491_149111

noncomputable def lcm_18_24 : ℕ :=
  Nat.lcm 18 24

theorem lcm_of_18_and_24 : lcm_18_24 = 72 :=
by
  sorry

end lcm_of_18_and_24_l1491_149111


namespace inequality_solution_l1491_149144

theorem inequality_solution (a : ℝ) : (∀ x : ℝ, (a + 1) * x > a + 1 → x < 1) ↔ a < -1 := by
  sorry

end inequality_solution_l1491_149144


namespace solve_for_x_l1491_149107

theorem solve_for_x (x : ℝ) (h₁ : (x + 2) ≠ 0) (h₂ : (|x| - 2) / (x + 2) = 0) : x = 2 := by
  sorry

end solve_for_x_l1491_149107


namespace gcd_1978_2017_l1491_149106

theorem gcd_1978_2017 : Int.gcd 1978 2017 = 1 :=
sorry

end gcd_1978_2017_l1491_149106


namespace diane_coffee_purchase_l1491_149182

theorem diane_coffee_purchase (c d : ℕ) (h1 : c + d = 7) (h2 : 90 * c + 60 * d % 100 = 0) : c = 6 :=
by
  sorry

end diane_coffee_purchase_l1491_149182


namespace gcd_12a_18b_l1491_149152

theorem gcd_12a_18b (a b : ℕ) (h : Nat.gcd a b = 12) : Nat.gcd (12 * a) (18 * b) = 72 :=
sorry

end gcd_12a_18b_l1491_149152


namespace money_left_after_shopping_l1491_149190

def initial_amount : ℕ := 26
def cost_jumper : ℕ := 9
def cost_tshirt : ℕ := 4
def cost_heels : ℕ := 5

theorem money_left_after_shopping : initial_amount - (cost_jumper + cost_tshirt + cost_heels) = 8 :=
by
  sorry

end money_left_after_shopping_l1491_149190


namespace num_broadcasting_methods_l1491_149125

theorem num_broadcasting_methods : 
  let n := 6
  let commercials := 4
  let public_services := 2
  (public_services * commercials!) = 48 :=
by
  let n := 6
  let commercials := 4
  let public_services := 2
  have total_methods : (public_services * commercials!) = 48 := sorry
  exact total_methods

end num_broadcasting_methods_l1491_149125


namespace num_rows_of_gold_bars_l1491_149101

-- Definitions from the problem conditions
def num_bars_per_row : ℕ := 20
def total_worth : ℕ := 1600000

-- Statement to prove
theorem num_rows_of_gold_bars :
  (total_worth / (total_worth / num_bars_per_row)) = 1 := 
by sorry

end num_rows_of_gold_bars_l1491_149101


namespace find_expression_l1491_149165

variable (a b E : ℝ)

-- Conditions
def condition1 := a / b = 4 / 3
def condition2 := E / (3 * a - 2 * b) = 3

-- Conclusion we want to prove
theorem find_expression : condition1 a b → condition2 a b E → E = 6 * b :=
by
  intro h1 h2
  sorry

end find_expression_l1491_149165


namespace tate_education_ratio_l1491_149142

theorem tate_education_ratio
  (n : ℕ)
  (m : ℕ)
  (h1 : n > 1)
  (h2 : (n - 1) + m * (n - 1) = 12)
  (h3 : n = 4) :
  (m * (n - 1)) / (n - 1) = 3 := 
by 
  sorry

end tate_education_ratio_l1491_149142


namespace initial_bleach_percentage_l1491_149140

-- Define variables and constants
def total_volume : ℝ := 100
def drained_volume : ℝ := 3.0612244898
def desired_percentage : ℝ := 0.05

-- Define the initial percentage (unknown)
variable (P : ℝ)

-- Define the statement to be proved
theorem initial_bleach_percentage :
  ( (total_volume - drained_volume) * P + drained_volume * 1 = total_volume * desired_percentage )
  → P = 0.02 :=
  by
    intro h
    -- skipping the proof as per instructions
    sorry

end initial_bleach_percentage_l1491_149140


namespace find_larger_number_l1491_149167

theorem find_larger_number (L S : ℤ) (h1 : L - S = 1365) (h2 : L = 6 * S + 5) : L = 1637 :=
by
  sorry

end find_larger_number_l1491_149167


namespace point_Q_in_first_quadrant_l1491_149131

theorem point_Q_in_first_quadrant (a b : ℝ) (h : a < 0 ∧ b < 0) : (0 < -a) ∧ (0 < -b) :=
by
  have ha : -a > 0 := by linarith
  have hb : -b > 0 := by linarith
  exact ⟨ha, hb⟩

end point_Q_in_first_quadrant_l1491_149131


namespace customers_in_other_countries_l1491_149151

-- Given 
def total_customers : ℕ := 7422
def customers_in_us : ℕ := 723

-- To Prove
theorem customers_in_other_countries : (total_customers - customers_in_us) = 6699 := 
by
  sorry

end customers_in_other_countries_l1491_149151


namespace quadrilateral_sides_equality_l1491_149173

theorem quadrilateral_sides_equality 
  (a b c d : ℕ) 
  (h1 : (b + c + d) % a = 0) 
  (h2 : (a + c + d) % b = 0) 
  (h3 : (a + b + d) % c = 0) 
  (h4 : (a + b + c) % d = 0) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end quadrilateral_sides_equality_l1491_149173


namespace find_positive_Y_for_nine_triangle_l1491_149164

def triangle_relation (X Y : ℝ) : ℝ := X^2 + 3 * Y^2

theorem find_positive_Y_for_nine_triangle (Y : ℝ) : (9^2 + 3 * Y^2 = 360) → Y = Real.sqrt 93 := 
by
  sorry

end find_positive_Y_for_nine_triangle_l1491_149164


namespace total_money_needed_l1491_149122

-- Declare John's initial amount
def john_has : ℝ := 0.75

-- Declare the additional amount John needs
def john_needs_more : ℝ := 1.75

-- The theorem statement that John needs a total of $2.50
theorem total_money_needed : john_has + john_needs_more = 2.5 :=
  by
  sorry

end total_money_needed_l1491_149122


namespace distance_between_home_and_school_l1491_149109

variable (D T : ℝ)

def boy_travel_5kmhr : Prop :=
  5 * (T + 7 / 60) = D

def boy_travel_10kmhr : Prop :=
  10 * (T - 8 / 60) = D

theorem distance_between_home_and_school :
  (boy_travel_5kmhr D T) ∧ (boy_travel_10kmhr D T) → D = 2.5 :=
by
  intro h
  sorry

end distance_between_home_and_school_l1491_149109


namespace find_f_of_16_l1491_149157

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem find_f_of_16 : (∃ a : ℝ, f 2 a = Real.sqrt 2) → f 16 (1/2) = 4 :=
by
  intro h
  sorry

end find_f_of_16_l1491_149157


namespace joan_mortgage_payback_months_l1491_149194

-- Define the conditions and statement

def first_payment : ℕ := 100
def total_amount : ℕ := 2952400

theorem joan_mortgage_payback_months :
  ∃ n : ℕ, 100 * (3^n - 1) / (3 - 1) = 2952400 ∧ n = 10 :=
by
  sorry

end joan_mortgage_payback_months_l1491_149194


namespace isosceles_trapezoid_rotation_produces_frustum_l1491_149195

-- Definitions based purely on conditions
structure IsoscelesTrapezoid :=
(a b c d : ℝ) -- sides
(ha : a = c) -- isosceles property
(hb : b ≠ d) -- non-parallel sides

def rotateAroundSymmetryAxis (shape : IsoscelesTrapezoid) : Type :=
-- We need to define what the rotation of the trapezoid produces
sorry

theorem isosceles_trapezoid_rotation_produces_frustum (shape : IsoscelesTrapezoid) :
  rotateAroundSymmetryAxis shape = Frustum :=
sorry

end isosceles_trapezoid_rotation_produces_frustum_l1491_149195


namespace no_integer_solution_for_equation_l1491_149136

theorem no_integer_solution_for_equation :
    ¬ ∃ (x y z : ℤ), x^2 + y^2 + z^2 = x * y * z - 1 :=
by
  sorry

end no_integer_solution_for_equation_l1491_149136


namespace inequality_sum_geq_three_l1491_149129

theorem inequality_sum_geq_three
  (a b c : ℝ)
  (h : a * b * c = 1) :
  (1 + a + a * b) / (1 + b + a * b) + 
  (1 + b + b * c) / (1 + c + b * c) +
  (1 + c + a * c) / (1 + a + a * c) ≥ 3 := 
sorry

end inequality_sum_geq_three_l1491_149129


namespace sqrt_infinite_nest_eq_two_l1491_149116

theorem sqrt_infinite_nest_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 := 
sorry

end sqrt_infinite_nest_eq_two_l1491_149116


namespace find_pairs_nat_numbers_l1491_149121

theorem find_pairs_nat_numbers (a b : ℕ) :
  (a^3 * b - 1) % (a + 1) = 0 ∧ (a * b^3 + 1) % (b - 1) = 0 ↔ 
  (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
by
  sorry

end find_pairs_nat_numbers_l1491_149121


namespace bug_return_probability_twelfth_move_l1491_149117

-- Conditions
def P : ℕ → ℚ
| 0       => 1
| (n + 1) => (1 : ℚ) / 3 * (1 - P n)

theorem bug_return_probability_twelfth_move :
  P 12 = 14762 / 59049 := by
sorry

end bug_return_probability_twelfth_move_l1491_149117


namespace max_odd_integers_l1491_149119

theorem max_odd_integers (chosen : Fin 5 → ℕ) (hpos : ∀ i, chosen i > 0) (heven : ∃ i, chosen i % 2 = 0) : 
  ∃ odd_count, odd_count = 4 ∧ (∀ i, i < 4 → chosen i % 2 = 1) := 
by 
  sorry

end max_odd_integers_l1491_149119


namespace fifth_term_binomial_expansion_l1491_149138

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem fifth_term_binomial_expansion (b x : ℝ) :
  let term := (binomial 7 4) * ((b / x)^(7 - 4)) * ((-x^2 * b)^4)
  term = -35 * b^7 * x^5 := 
by
  sorry

end fifth_term_binomial_expansion_l1491_149138


namespace product_of_powers_l1491_149179

theorem product_of_powers (x y : ℕ) (h1 : x = 2) (h2 : y = 3) :
  ((x ^ 1 + y ^ 1) * (x ^ 2 + y ^ 2) * (x ^ 4 + y ^ 4) * 
   (x ^ 8 + y ^ 8) * (x ^ 16 + y ^ 16) * (x ^ 32 + y ^ 32) * 
   (x ^ 64 + y ^ 64)) = y ^ 128 - x ^ 128 :=
by
  rw [h1, h2]
  -- We would proceed with the proof here, but it's not needed per instructions.
  sorry

end product_of_powers_l1491_149179


namespace melanie_balloons_l1491_149133

theorem melanie_balloons (joan_balloons melanie_balloons total_balloons : ℕ)
  (h_joan : joan_balloons = 40)
  (h_total : total_balloons = 81) :
  melanie_balloons = total_balloons - joan_balloons :=
by
  sorry

end melanie_balloons_l1491_149133


namespace sum_geometric_series_is_correct_l1491_149185

def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_geometric_series_is_correct
  (a r : ℚ) (n : ℕ)
  (h_a : a = 1/4)
  (h_r : r = 1/4)
  (h_n : n = 5) :
  geometric_series_sum a r n = 341 / 1024 :=
by
  rw [h_a, h_r, h_n]
  -- Now we can skip the proof.
  sorry

end sum_geometric_series_is_correct_l1491_149185


namespace at_least_one_truth_and_not_knight_l1491_149102

def isKnight (n : Nat) : Prop := n = 1   -- Identifier for knights
def isKnave (n : Nat) : Prop := n = 0    -- Identifier for knaves
def isRegular (n : Nat) : Prop := n = 2  -- Identifier for regular persons

def A := 2     -- Initially define A's type as regular (this can be adjusted)
def B := 2     -- Initially define B's type as regular (this can be adjusted)

def statementA : Prop := isKnight B
def statementB : Prop := ¬ isKnight A

theorem at_least_one_truth_and_not_knight :
  statementA ∧ ¬ isKnight A ∨ statementB ∧ ¬ isKnight B :=
sorry

end at_least_one_truth_and_not_knight_l1491_149102


namespace smallest_value_of_x_l1491_149155

theorem smallest_value_of_x :
  ∃ x : Real, (∀ z, (z = (5 * x - 20) / (4 * x - 5)) → (z * z + z = 20)) → x = 0 :=
by
  sorry

end smallest_value_of_x_l1491_149155


namespace correct_total_weight_6_moles_Al2_CO3_3_l1491_149198

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

def num_atoms_Al : ℕ := 2
def num_atoms_C : ℕ := 3
def num_atoms_O : ℕ := 9

def molecular_weight_Al2_CO3_3 : ℝ :=
  (num_atoms_Al * atomic_weight_Al) +
  (num_atoms_C * atomic_weight_C) +
  (num_atoms_O * atomic_weight_O)

def num_moles : ℝ := 6

def total_weight_6_moles_Al2_CO3_3 : ℝ := num_moles * molecular_weight_Al2_CO3_3

theorem correct_total_weight_6_moles_Al2_CO3_3 :
  total_weight_6_moles_Al2_CO3_3 = 1403.94 :=
by
  unfold total_weight_6_moles_Al2_CO3_3
  unfold num_moles
  unfold molecular_weight_Al2_CO3_3
  unfold num_atoms_Al num_atoms_C num_atoms_O atomic_weight_Al atomic_weight_C atomic_weight_O
  sorry

end correct_total_weight_6_moles_Al2_CO3_3_l1491_149198


namespace female_salmon_returned_l1491_149158

theorem female_salmon_returned :
  let total_salmon : ℕ := 971639
  let male_salmon : ℕ := 712261
  total_salmon - male_salmon = 259378 :=
by
  let total_salmon := 971639
  let male_salmon := 712261
  calc
    971639 - 712261 = 259378 := by norm_num

end female_salmon_returned_l1491_149158


namespace stratified_sampling_correct_l1491_149105

-- Definitions for the conditions
def total_employees : ℕ := 750
def young_employees : ℕ := 350
def middle_aged_employees : ℕ := 250
def elderly_employees : ℕ := 150
def sample_size : ℕ := 15
def sampling_proportion : ℚ := sample_size / total_employees

-- Statement to prove
theorem stratified_sampling_correct :
  (young_employees * sampling_proportion = 7) ∧
  (middle_aged_employees * sampling_proportion = 5) ∧
  (elderly_employees * sampling_proportion = 3) :=
by
  sorry

end stratified_sampling_correct_l1491_149105


namespace distinct_banners_count_l1491_149199

def colors : Finset String := 
  {"red", "white", "blue", "green", "yellow"}

def valid_banners (strip1 strip2 strip3 : String) : Prop :=
  strip1 ∈ colors ∧ strip2 ∈ colors ∧ strip3 ∈ colors ∧
  strip1 ≠ strip2 ∧ strip2 ≠ strip3 ∧ strip3 ≠ strip1

theorem distinct_banners_count : 
  ∃ (banners : Finset (String × String × String)), 
    (∀ s1 s2 s3, (s1, s2, s3) ∈ banners ↔ valid_banners s1 s2 s3) ∧
    banners.card = 60 :=
by
  sorry

end distinct_banners_count_l1491_149199


namespace washing_machines_removed_correct_l1491_149177

-- Define the conditions
def crates : ℕ := 10
def boxes_per_crate : ℕ := 6
def washing_machines_per_box : ℕ := 4
def washing_machines_removed_per_box : ℕ := 1

-- Define the initial and final states
def initial_washing_machines_in_crate : ℕ := boxes_per_crate * washing_machines_per_box
def initial_washing_machines_in_container : ℕ := crates * initial_washing_machines_in_crate

def final_washing_machines_in_box : ℕ := washing_machines_per_box - washing_machines_removed_per_box
def final_washing_machines_in_crate : ℕ := boxes_per_crate * final_washing_machines_in_box
def final_washing_machines_in_container : ℕ := crates * final_washing_machines_in_crate

-- Number of washing machines removed
def washing_machines_removed : ℕ := initial_washing_machines_in_container - final_washing_machines_in_container

-- Theorem statement in Lean 4
theorem washing_machines_removed_correct : washing_machines_removed = 60 := by
  sorry

end washing_machines_removed_correct_l1491_149177


namespace number_of_dress_designs_l1491_149153

theorem number_of_dress_designs :
  let colors := 5
  let patterns := 4
  let sleeve_designs := 3
  colors * patterns * sleeve_designs = 60 := by
  sorry

end number_of_dress_designs_l1491_149153


namespace andrew_age_l1491_149168

variables (a g : ℕ)

theorem andrew_age : 
  (g = 16 * a) ∧ (g - a = 60) → a = 4 := by
  sorry

end andrew_age_l1491_149168


namespace t_shirts_to_buy_l1491_149193

variable (P T : ℕ)

def condition1 : Prop := 3 * P + 6 * T = 750
def condition2 : Prop := P + 12 * T = 750

theorem t_shirts_to_buy (h1 : condition1 P T) (h2 : condition2 P T) :
  400 / T = 8 :=
by
  sorry

end t_shirts_to_buy_l1491_149193


namespace twelve_pow_six_mod_eight_l1491_149124

theorem twelve_pow_six_mod_eight : ∃ m : ℕ, 0 ≤ m ∧ m < 8 ∧ 12^6 % 8 = m ∧ m = 0 := by
  sorry

end twelve_pow_six_mod_eight_l1491_149124


namespace move_point_inside_with_25_reflections_cannot_move_point_inside_with_24_reflections_l1491_149139

-- Define the initial conditions
def pointA := (50 : ℝ)
def radius := (1 : ℝ)
def origin := (0 : ℝ)

-- Statement for part (a)
theorem move_point_inside_with_25_reflections :
  ∃ (n : ℕ) (r : ℝ), n = 25 ∧ r = radius + 50 ∧ pointA ≤ r :=
by
  sorry

-- Statement for part (b)
theorem cannot_move_point_inside_with_24_reflections :
  ∀ (n : ℕ) (r : ℝ), n = 24 → r = radius + 48 → pointA > r :=
by
  sorry

end move_point_inside_with_25_reflections_cannot_move_point_inside_with_24_reflections_l1491_149139


namespace max_prob_games_4_choose_best_of_five_l1491_149197

-- Definitions of probabilities for Team A and Team B in different game scenarios
def prob_win_deciding_game : ℝ := 0.5
def prob_A_non_deciding : ℝ := 0.6
def prob_B_non_deciding : ℝ := 0.4

-- Definitions of probabilities for different number of games in the series
def prob_xi_3 : ℝ := (prob_A_non_deciding)^3 + (prob_B_non_deciding)^3
def prob_xi_4 : ℝ := 3 * (prob_A_non_deciding^2 * prob_B_non_deciding * prob_A_non_deciding + prob_B_non_deciding^2 * prob_A_non_deciding * prob_B_non_deciding)
def prob_xi_5 : ℝ := 6 * (prob_A_non_deciding^2 * prob_B_non_deciding^2) * (2 * prob_win_deciding_game)

-- The statement that a series of 4 games has the highest probability
theorem max_prob_games_4 : prob_xi_4 > prob_xi_5 ∧ prob_xi_4 > prob_xi_3 :=
by {
  sorry
}

-- Definitions of winning probabilities in the series for Team A
def prob_A_win_best_of_3 : ℝ := (prob_A_non_deciding)^2 + 2 * (prob_A_non_deciding * prob_B_non_deciding * prob_win_deciding_game)
def prob_A_win_best_of_5 : ℝ := (prob_A_non_deciding)^3 + 3 * (prob_A_non_deciding^2 * prob_B_non_deciding) + 6 * (prob_A_non_deciding^2 * prob_B_non_deciding^2 * prob_win_deciding_game)

-- The statement that Team A has a higher chance of winning in a best-of-five series
theorem choose_best_of_five : prob_A_win_best_of_5 > prob_A_win_best_of_3 :=
by {
  sorry
}

end max_prob_games_4_choose_best_of_five_l1491_149197


namespace find_X_l1491_149146

variable {α : Type} -- considering sets of some type α
variables (A B X : Set α)

theorem find_X (h1 : A ∩ X = B ∩ X ∧ B ∩ X = A ∩ B)
               (h2 : A ∪ B ∪ X = A ∪ B) : X = A ∩ B :=
by {
    sorry
}

end find_X_l1491_149146


namespace find_parameter_a_exactly_two_solutions_l1491_149103

noncomputable def system_has_two_solutions (a : ℝ) : Prop :=
∃ (x y : ℝ), |y - 3 - x| + |y - 3 + x| = 6 ∧ (|x| - 4)^2 + (|y| - 3)^2 = a

theorem find_parameter_a_exactly_two_solutions :
  {a : ℝ | system_has_two_solutions a} = {1, 25} :=
by
  sorry

end find_parameter_a_exactly_two_solutions_l1491_149103


namespace necessary_but_not_sufficient_l1491_149189

noncomputable def is_increasing_on_R (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem necessary_but_not_sufficient (f : ℝ → ℝ) :
  (f 1 < f 2) → (¬∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∨ (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :=
by
  sorry

end necessary_but_not_sufficient_l1491_149189


namespace magic_ink_combinations_l1491_149126

def herbs : ℕ := 4
def essences : ℕ := 6
def incompatible_herbs : ℕ := 3

theorem magic_ink_combinations :
  herbs * essences - incompatible_herbs = 21 := 
  by
  sorry

end magic_ink_combinations_l1491_149126


namespace johns_train_speed_l1491_149149

noncomputable def average_speed_of_train (D : ℝ) (V_t : ℝ) : ℝ := D / (0.8 * D / V_t + 0.2 * D / 20)

theorem johns_train_speed (D : ℝ) (V_t : ℝ) (h1 : average_speed_of_train D V_t = 50) : V_t = 64 :=
by
  sorry

end johns_train_speed_l1491_149149


namespace real_numbers_inequality_l1491_149123

theorem real_numbers_inequality (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1 / 3) * (a + b + c)^2 :=
by
  sorry

end real_numbers_inequality_l1491_149123

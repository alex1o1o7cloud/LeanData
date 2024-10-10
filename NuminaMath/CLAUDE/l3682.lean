import Mathlib

namespace board_length_l3682_368236

/-- Given a board cut into two pieces, prove that its total length is 20.0 feet. -/
theorem board_length : 
  ∀ (shorter longer : ℝ),
  shorter = 8.0 →
  2 * shorter = longer + 4 →
  shorter + longer = 20.0 := by
sorry

end board_length_l3682_368236


namespace largest_n_binomial_equality_l3682_368229

theorem largest_n_binomial_equality : 
  (∃ n : ℕ, (n : ℤ) = 7 ∧ 
    Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n ∧
    ∀ m : ℕ, m > n → Nat.choose 10 3 + Nat.choose 10 4 ≠ Nat.choose 11 m) :=
by sorry

end largest_n_binomial_equality_l3682_368229


namespace function_inequality_l3682_368270

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_deriv : ∀ x : ℝ, x ≠ 1 → (x - 1) * deriv f x > 0) :
  f 0 + f 2 > 2 * f 1 := by
  sorry

end function_inequality_l3682_368270


namespace folded_line_length_squared_l3682_368207

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  is_positive : side_length > 0

-- Define the folding operation
def fold (t : EquilateralTriangle) (fold_point : ℝ) :=
  0 < fold_point ∧ fold_point < t.side_length

-- Theorem statement
theorem folded_line_length_squared 
  (t : EquilateralTriangle) 
  (h_side : t.side_length = 10) 
  (h_fold : fold t 3) : 
  ∃ (l : ℝ), l^2 = 37/4 ∧ l > 0 := by
  sorry

end folded_line_length_squared_l3682_368207


namespace candy_store_lollipops_l3682_368201

/-- The number of milliliters of food coloring used for each lollipop -/
def lollipop_coloring : ℕ := 5

/-- The number of milliliters of food coloring used for each hard candy -/
def hard_candy_coloring : ℕ := 20

/-- The number of hard candies made -/
def hard_candies_made : ℕ := 5

/-- The total amount of food coloring used in milliliters -/
def total_coloring_used : ℕ := 600

/-- The number of lollipops made -/
def lollipops_made : ℕ := 100

theorem candy_store_lollipops :
  lollipops_made * lollipop_coloring + hard_candies_made * hard_candy_coloring = total_coloring_used :=
by sorry

end candy_store_lollipops_l3682_368201


namespace triangle_area_in_circle_l3682_368257

theorem triangle_area_in_circle (R : ℝ) (α β : ℝ) (h_R : R = 2) 
  (h_α : α = π / 3) (h_β : β = π / 4) :
  let γ : ℝ := π - α - β
  let a : ℝ := 2 * R * Real.sin α
  let b : ℝ := 2 * R * Real.sin β
  let c : ℝ := 2 * R * Real.sin γ
  let S : ℝ := (Real.sqrt 3 + 3 : ℝ)
  S = (1 / 2) * a * b * Real.sin γ := by sorry

end triangle_area_in_circle_l3682_368257


namespace arccos_cos_three_l3682_368227

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := by sorry

end arccos_cos_three_l3682_368227


namespace triangle_side_length_l3682_368222

-- Define the triangle ABC
def triangle_ABC (a : ℕ) : Prop :=
  ∃ (A B C : ℝ × ℝ),
    let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
    let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
    let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
    AB = 1 ∧ BC = 2007 ∧ AC = a

-- Theorem statement
theorem triangle_side_length :
  ∀ a : ℕ, triangle_ABC a → a = 2007 :=
by
  sorry


end triangle_side_length_l3682_368222


namespace inequality_proof_l3682_368259

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x + y + z)^2 / 3 ≥ x * Real.sqrt (y * z) + y * Real.sqrt (z * x) + z * Real.sqrt (x * y) := by
  sorry

end inequality_proof_l3682_368259


namespace number_divided_by_ten_l3682_368234

theorem number_divided_by_ten : (120 : ℚ) / 10 = 12 := by
  sorry

end number_divided_by_ten_l3682_368234


namespace M_on_inscribed_square_l3682_368268

/-- Right triangle with squares and inscribed square -/
structure RightTriangleWithSquares where
  -- Right triangle ABC
  a : ℝ
  b : ℝ
  c : ℝ
  -- Pythagorean theorem
  pythagoras : a^2 + b^2 = c^2
  -- Positivity of sides
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  -- Inscribed square side length
  x : ℝ
  x_def : x = (a * b) / (a + b)
  -- Point M
  M : ℝ × ℝ

/-- The theorem stating that M lies on the perimeter of the inscribed square -/
theorem M_on_inscribed_square (t : RightTriangleWithSquares) :
  t.M.1 = t.x ∧ 0 ≤ t.M.2 ∧ t.M.2 ≤ t.x := by
  sorry

end M_on_inscribed_square_l3682_368268


namespace white_pairs_coincide_l3682_368205

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  white_white : ℕ

def figure_problem (counts : TriangleCounts) (pairs : CoincidingPairs) : Prop :=
  counts.red = 4 ∧
  counts.blue = 6 ∧
  counts.white = 10 ∧
  pairs.red_red = 3 ∧
  pairs.blue_blue = 4 ∧
  pairs.red_white = 3 ∧
  pairs.white_white = 7

theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) :
  figure_problem counts pairs → pairs.white_white = 7 := by
  sorry

end white_pairs_coincide_l3682_368205


namespace men_french_percentage_l3682_368204

/-- Represents the percentage of employees who are men -/
def percent_men : ℝ := 0.35

/-- Represents the percentage of employees who speak French -/
def percent_french : ℝ := 0.40

/-- Represents the percentage of women who do not speak French -/
def percent_women_not_french : ℝ := 0.7077

/-- Represents the percentage of men who speak French -/
def percent_men_french : ℝ := 0.60

theorem men_french_percentage :
  percent_men * percent_men_french + (1 - percent_men) * (1 - percent_women_not_french) = percent_french :=
sorry


end men_french_percentage_l3682_368204


namespace spade_calculation_l3682_368273

-- Define the ⋄ operation
def spade (x y : ℝ) : ℝ := (x + y)^2 * (x - y)

-- Theorem statement
theorem spade_calculation : spade 2 (spade 3 6) = 14229845 := by sorry

end spade_calculation_l3682_368273


namespace sum_of_m_and_n_is_zero_l3682_368280

theorem sum_of_m_and_n_is_zero (m n p : ℝ) 
  (h1 : m * n + p^2 + 4 = 0) 
  (h2 : m - n = 4) : 
  m + n = 0 := by
  sorry

end sum_of_m_and_n_is_zero_l3682_368280


namespace largest_square_area_l3682_368272

theorem largest_square_area (original_side : ℝ) (cut_side : ℝ) (largest_area : ℝ) : 
  original_side = 5 →
  cut_side = 1 →
  largest_area = (5 * Real.sqrt 2 / 2) ^ 2 →
  largest_area = 12.5 :=
by sorry

end largest_square_area_l3682_368272


namespace existence_of_special_integer_l3682_368274

theorem existence_of_special_integer (P : Finset Nat) (h_prime : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : Nat, x > 0 ∧ (∀ p : Nat, Nat.Prime p →
    (p ∈ P ↔ ∃ a b : Nat, a > 0 ∧ b > 0 ∧ x = a ^ p + b ^ p)) := by
  sorry

end existence_of_special_integer_l3682_368274


namespace smallest_number_l3682_368277

theorem smallest_number (a b c d : ℤ) (ha : a = 0) (hb : b = -3) (hc : c = 1) (hd : d = -1) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by sorry

end smallest_number_l3682_368277


namespace greatest_x_given_lcm_l3682_368283

def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem greatest_x_given_lcm (x : ℕ) :
  lcm x 15 21 = 105 → x ≤ 105 ∧ ∃ y : ℕ, lcm y 15 21 = 105 ∧ y = 105 :=
sorry

end greatest_x_given_lcm_l3682_368283


namespace gum_pieces_per_package_l3682_368206

theorem gum_pieces_per_package (total_packages : ℕ) (total_pieces : ℕ) : 
  total_packages = 9 → total_pieces = 135 → total_pieces / total_packages = 15 := by
  sorry

end gum_pieces_per_package_l3682_368206


namespace stratified_sampling_seniors_l3682_368208

theorem stratified_sampling_seniors (total_population : ℕ) (senior_population : ℕ) (sample_size : ℕ) : 
  total_population = 2100 → 
  senior_population = 680 → 
  sample_size = 105 → 
  (sample_size * senior_population) / total_population = 34 := by
sorry

end stratified_sampling_seniors_l3682_368208


namespace min_chord_length_proof_l3682_368261

/-- The circle equation: x^2 + y^2 - 6x = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

/-- The point through which the chord passes -/
def point : ℝ × ℝ := (1, 2)

/-- The minimum length of the chord -/
def min_chord_length : ℝ := 2

theorem min_chord_length_proof :
  ∀ (x y : ℝ), circle_equation x y →
  ∀ (chord_length : ℝ),
  (∃ (x1 y1 x2 y2 : ℝ), 
    circle_equation x1 y1 ∧ 
    circle_equation x2 y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = chord_length^2 ∧
    (x1 + x2) / 2 = point.1 ∧ 
    (y1 + y2) / 2 = point.2) →
  chord_length ≥ min_chord_length :=
by sorry

#check min_chord_length_proof

end min_chord_length_proof_l3682_368261


namespace gnomes_and_ponies_count_l3682_368288

/-- Represents the number of gnomes -/
def num_gnomes : ℕ := 12

/-- Represents the number of ponies -/
def num_ponies : ℕ := 3

/-- The total number of heads in the caravan -/
def total_heads : ℕ := 15

/-- The total number of legs in the caravan -/
def total_legs : ℕ := 36

/-- Each gnome has this many legs -/
def gnome_legs : ℕ := 2

/-- Each pony has this many legs -/
def pony_legs : ℕ := 4

theorem gnomes_and_ponies_count :
  (num_gnomes + num_ponies = total_heads) ∧
  (num_gnomes * gnome_legs + num_ponies * pony_legs = total_legs) :=
by sorry

end gnomes_and_ponies_count_l3682_368288


namespace shooting_probabilities_l3682_368260

/-- Represents the probabilities of hitting each ring in a shooting game. -/
structure RingProbabilities where
  ten : Real
  nine : Real
  eight : Real
  seven : Real
  sub_seven : Real

/-- The probabilities sum to 1. -/
axiom prob_sum_to_one (p : RingProbabilities) : 
  p.ten + p.nine + p.eight + p.seven + p.sub_seven = 1

/-- The given probabilities for each ring. -/
def given_probs : RingProbabilities := {
  ten := 0.24,
  nine := 0.28,
  eight := 0.19,
  seven := 0.16,
  sub_seven := 0.13
}

/-- The probability of hitting 10 or 9 rings. -/
def prob_ten_or_nine (p : RingProbabilities) : Real :=
  p.ten + p.nine

/-- The probability of hitting at least 7 ring. -/
def prob_at_least_seven (p : RingProbabilities) : Real :=
  p.ten + p.nine + p.eight + p.seven

/-- The probability of not hitting 8 ring. -/
def prob_not_eight (p : RingProbabilities) : Real :=
  1 - p.eight

theorem shooting_probabilities (p : RingProbabilities) 
  (h : p = given_probs) : 
  prob_ten_or_nine p = 0.52 ∧ 
  prob_at_least_seven p = 0.87 ∧ 
  prob_not_eight p = 0.81 := by
  sorry

end shooting_probabilities_l3682_368260


namespace geometric_sequence_general_term_l3682_368275

/-- A geometric sequence with specific terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_general_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a3 : a 3 = 3)
  (h_a10 : a 10 = 384) :
  ∃ q : ℝ, ∀ n : ℕ, a n = 3 * q^(n - 3) ∧ q = 2 :=
sorry

end geometric_sequence_general_term_l3682_368275


namespace either_odd_or_even_l3682_368215

theorem either_odd_or_even (n : ℤ) : (Odd (2*n - 1)) ∨ (Even (2*n + 1)) := by
  sorry

end either_odd_or_even_l3682_368215


namespace percentage_passed_both_subjects_l3682_368203

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 25)
  (h2 : failed_english = 48)
  (h3 : failed_both = 27) :
  100 - (failed_hindi + failed_english - failed_both) = 54 := by
sorry

end percentage_passed_both_subjects_l3682_368203


namespace camel_moves_divisible_by_three_l3682_368225

/-- Represents the color of a square --/
inductive SquareColor
| Black
| White

/-- Represents a camel's movement --/
def CamelMove := ℕ → SquareColor

/-- A camel's movement pattern that alternates between black and white squares --/
def alternatingPattern : CamelMove :=
  fun n => match n % 3 with
    | 0 => SquareColor.Black
    | 1 => SquareColor.White
    | _ => SquareColor.Black

/-- Theorem: If a camel makes n moves in an alternating pattern and returns to its starting position, then n is divisible by 3 --/
theorem camel_moves_divisible_by_three (n : ℕ) 
  (h1 : alternatingPattern n = alternatingPattern 0) : 
  3 ∣ n :=
by sorry

end camel_moves_divisible_by_three_l3682_368225


namespace ratio_of_sides_l3682_368238

/-- A rectangle with a point inside dividing it into four triangles -/
structure DividedRectangle where
  -- The lengths of the sides of the rectangle
  AB : ℝ
  BC : ℝ
  -- The areas of the four triangles
  area_APD : ℝ
  area_BPA : ℝ
  area_CPB : ℝ
  area_DPC : ℝ
  -- Conditions
  positive_AB : 0 < AB
  positive_BC : 0 < BC
  diagonal_condition : AB^2 + BC^2 = (2*AB)^2
  area_condition : area_APD = 1 ∧ area_BPA = 2 ∧ area_CPB = 3 ∧ area_DPC = 4

/-- The theorem stating the ratio of sides in the divided rectangle -/
theorem ratio_of_sides (r : DividedRectangle) : r.AB / r.BC = 10 / 3 := by
  sorry

end ratio_of_sides_l3682_368238


namespace rectangle_dimensions_l3682_368212

theorem rectangle_dimensions : ∃ (l w : ℝ), 
  (l = 9 ∧ w = 8) ∧
  (l - 3 = w - 2) ∧
  ((l - 3)^2 = (1/2) * l * w) := by
  sorry

end rectangle_dimensions_l3682_368212


namespace interval_of_decrease_l3682_368292

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 30*x - 33

-- Theorem stating the interval of decrease
theorem interval_of_decrease :
  ∀ x ∈ (Set.Ioo (-1 : ℝ) 11), (f' x < 0) ∧
  ∀ y ∉ (Set.Ioo (-1 : ℝ) 11), (f' y ≥ 0) :=
sorry

end interval_of_decrease_l3682_368292


namespace sqrt_real_range_l3682_368251

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = 6 - 2 * x) ↔ x ≤ 3 := by
  sorry

end sqrt_real_range_l3682_368251


namespace number_wall_solution_l3682_368252

/-- Represents a number wall with 4 elements in the bottom row -/
structure NumberWall :=
  (bottom_row : Fin 4 → ℕ)
  (second_row_right : ℕ)
  (top : ℕ)

/-- Checks if a number wall is valid according to the summing rules -/
def is_valid_wall (w : NumberWall) : Prop :=
  w.second_row_right = w.bottom_row 2 + w.bottom_row 3 ∧
  w.top = (w.bottom_row 0 + w.bottom_row 1 + w.bottom_row 2) + w.second_row_right

theorem number_wall_solution (w : NumberWall) 
  (h1 : w.bottom_row 1 = 3)
  (h2 : w.bottom_row 2 = 6)
  (h3 : w.bottom_row 3 = 5)
  (h4 : w.second_row_right = 20)
  (h5 : w.top = 57)
  (h6 : is_valid_wall w) :
  w.bottom_row 0 = 25 := by
  sorry

end number_wall_solution_l3682_368252


namespace power_of_two_equation_l3682_368279

theorem power_of_two_equation : ∃ x : ℕ, 
  8 * (32 ^ 10) = 2 ^ x ∧ x = 53 := by
  sorry

end power_of_two_equation_l3682_368279


namespace expected_value_is_three_halves_l3682_368256

/-- The number of white balls in the bag -/
def white_balls : ℕ := 1

/-- The number of red balls in the bag -/
def red_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 2

/-- X represents the number of red balls drawn -/
def X : Finset ℕ := {1, 2}

/-- The probability mass function of X -/
def prob_X (x : ℕ) : ℚ :=
  if x = 1 then 1/2
  else if x = 2 then 1/2
  else 0

/-- The expected value of X -/
def expected_value_X : ℚ := (1 : ℚ) * (prob_X 1) + (2 : ℚ) * (prob_X 2)

theorem expected_value_is_three_halves :
  expected_value_X = 3/2 := by sorry

end expected_value_is_three_halves_l3682_368256


namespace ball_count_theorem_l3682_368230

/-- Represents the number of balls of each color in a box -/
structure BallCount where
  white : ℕ
  red : ℕ
  blue : ℕ

/-- Checks if the given ball counts satisfy the ratio 4:3:2 for white:red:blue -/
def satisfiesRatio (bc : BallCount) : Prop :=
  4 * bc.red = 3 * bc.white ∧ 4 * bc.blue = 2 * bc.white

theorem ball_count_theorem (bc : BallCount) 
    (h_ratio : satisfiesRatio bc) 
    (h_white : bc.white = 12) : 
    bc.red = 9 ∧ bc.blue = 6 := by
  sorry

end ball_count_theorem_l3682_368230


namespace system_solution_l3682_368269

theorem system_solution : ∃ (x y z : ℤ),
  (5732 * x + 2134 * y + 2134 * z = 7866) ∧
  (2134 * x + 5732 * y + 2134 * z = 670) ∧
  (2134 * x + 2134 * y + 5732 * z = 11464) ∧
  x = 1 ∧ y = -1 ∧ z = 2 := by
  sorry

end system_solution_l3682_368269


namespace xyz_congruence_l3682_368258

theorem xyz_congruence (x y z : Int) : 
  x < 7 → y < 7 → z < 7 →
  (x + 3*y + 2*z) % 7 = 2 →
  (3*x + 2*y + z) % 7 = 5 →
  (2*x + y + 3*z) % 7 = 3 →
  (x * y * z) % 7 = 3 := by
  sorry

end xyz_congruence_l3682_368258


namespace xyz_value_l3682_368213

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14)
  (h3 : (x + y + z)^2 = 25) :
  x * y * z = 31 / 3 := by
sorry

end xyz_value_l3682_368213


namespace meaningful_expression_l3682_368291

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2023)) ↔ x ≠ 2023 := by
sorry

end meaningful_expression_l3682_368291


namespace stone_minimum_speed_l3682_368295

/-- The minimum speed for a stone to pass through both corners of a building without touching the roof -/
theorem stone_minimum_speed (g H l : ℝ) (α : ℝ) (h_g : g > 0) (h_H : H > 0) (h_l : l > 0) (h_α : 0 < α ∧ α < π / 2) :
  ∃ v₀ : ℝ, v₀ > 0 ∧
    v₀ = Real.sqrt (g * (2 * H + l * (1 - Real.sin α) / Real.cos α)) ∧
    ∀ v : ℝ, v > v₀ →
      ∃ (x y : ℝ → ℝ), (∀ t, x t = v * Real.cos α * t ∧ y t = -g * t^2 / 2 + v * Real.sin α * t) ∧
        (∃ t₁ t₂, t₁ ≠ t₂ ∧ x t₁ = 0 ∧ y t₁ = H ∧ x t₂ = l ∧ y t₂ = H - l * Real.tan α) ∧
        (∀ t, 0 ≤ x t ∧ x t ≤ l → y t ≥ H - (x t) * Real.tan α) :=
by sorry

end stone_minimum_speed_l3682_368295


namespace batsman_new_average_l3682_368214

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat

/-- Calculates the new average after an additional inning -/
def newAverage (bp : BatsmanPerformance) : Nat :=
  (bp.totalRuns + 74) / (bp.innings + 1)

/-- Theorem: The batsman's new average is 26 runs -/
theorem batsman_new_average (bp : BatsmanPerformance) 
  (h1 : bp.innings = 16)
  (h2 : newAverage bp = bp.totalRuns / bp.innings + 3)
  : newAverage bp = 26 := by
  sorry

#check batsman_new_average

end batsman_new_average_l3682_368214


namespace ceiling_sum_sqrt_l3682_368262

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end ceiling_sum_sqrt_l3682_368262


namespace determine_c_l3682_368235

/-- A function f(x) = x^2 + ax + b with domain [0, +∞) -/
def f (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The solution set of f(x) < c is (m, m+6) -/
def solution_set (a b c m : ℝ) : Prop :=
  ∀ x, x ∈ Set.Ioo m (m+6) ↔ f a b x < c

theorem determine_c (a b c m : ℝ) :
  (∀ x, x ≥ 0 → f a b x = x^2 + a*x + b) →
  solution_set a b c m →
  c = 9 := by sorry

end determine_c_l3682_368235


namespace flowers_for_maria_l3682_368271

def days_until_birthday : ℕ := 22
def savings_per_day : ℚ := 2
def cost_per_flower : ℚ := 4

theorem flowers_for_maria :
  ⌊(days_until_birthday * savings_per_day) / cost_per_flower⌋ = 11 := by
  sorry

end flowers_for_maria_l3682_368271


namespace total_is_527_given_shares_inconsistent_l3682_368249

/-- Represents the shares of money for three individuals --/
structure Shares :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Calculates the total amount from given shares --/
def total_amount (s : Shares) : ℕ := s.a + s.b + s.c

/-- The given shares --/
def given_shares : Shares := ⟨372, 93, 62⟩

/-- Theorem stating that the total amount is 527 --/
theorem total_is_527 : total_amount given_shares = 527 := by
  sorry

/-- Property that should hold for the shares based on the problem statement --/
def shares_property (s : Shares) : Prop :=
  s.a = (2 * s.b) / 3 ∧ s.b = s.c / 4

/-- Theorem stating that the given shares do not satisfy the problem's conditions --/
theorem given_shares_inconsistent : ¬ shares_property given_shares := by
  sorry

end total_is_527_given_shares_inconsistent_l3682_368249


namespace quadratic_inequality_solution_set_l3682_368265

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 7 * x - 10
  let S : Set ℝ := {x | f x ≥ 0}
  S = {x | x ≥ 10/3 ∨ x ≤ -1} := by
  sorry

end quadratic_inequality_solution_set_l3682_368265


namespace wire_cutting_is_random_event_l3682_368264

/-- An event that can occur but is not certain to occur -/
structure PossibleEvent where
  can_occur : Bool
  not_certain : Bool

/-- A random event is a possible event that exhibits regularity in repeated trials -/
structure RandomEvent extends PossibleEvent where
  exhibits_regularity : Bool

/-- The event of cutting a wire into three pieces to form a triangle -/
def wire_cutting_event (a : ℝ) : PossibleEvent :=
  { can_occur := true,
    not_certain := true }

/-- Theorem: The wire cutting event is a random event -/
theorem wire_cutting_is_random_event (a : ℝ) :
  ∃ (e : RandomEvent), (e.toPossibleEvent = wire_cutting_event a) :=
sorry

end wire_cutting_is_random_event_l3682_368264


namespace quadrilateral_diagonal_angle_l3682_368224

/-- A quadrilateral with specific side lengths and angles has diagonals that intersect at a 60° angle. -/
theorem quadrilateral_diagonal_angle (a b c : ℝ) (angle_ab angle_bc : ℝ) :
  a = 4 * Real.sqrt 3 →
  b = 9 →
  c = Real.sqrt 3 →
  angle_ab = π / 6 →  -- 30° in radians
  angle_bc = π / 2 →  -- 90° in radians
  ∃ (angle_diagonals : ℝ), angle_diagonals = π / 3 :=  -- 60° in radians
by sorry

end quadrilateral_diagonal_angle_l3682_368224


namespace number_of_correct_propositions_is_zero_l3682_368286

/-- Definition of a Frustum -/
structure Frustum where
  -- A frustum has two parallel faces (base and top)
  has_parallel_faces : Bool
  -- The extensions of lateral edges intersect at a point
  lateral_edges_intersect : Bool
  -- The extensions of waists of lateral faces intersect at a point
  waists_intersect : Bool

/-- Definition of a proposition about frustums -/
structure FrustumProposition where
  statement : String
  is_correct : Bool

/-- The three given propositions -/
def propositions : List FrustumProposition := [
  { statement := "Cutting a pyramid with a plane, the part between the base of the pyramid and the section is a frustum",
    is_correct := false },
  { statement := "A polyhedron with two parallel and similar bases, and all other faces being trapezoids, is a frustum",
    is_correct := false },
  { statement := "A hexahedron with two parallel faces and the other four faces being isosceles trapezoids is a frustum",
    is_correct := false }
]

/-- Theorem: The number of correct propositions is zero -/
theorem number_of_correct_propositions_is_zero :
  (propositions.filter (λ p => p.is_correct)).length = 0 := by
  sorry

end number_of_correct_propositions_is_zero_l3682_368286


namespace power_of_two_sum_l3682_368240

theorem power_of_two_sum : 2^3 + 2^3 + 2^3 + 2^3 = 2^5 := by
  sorry

end power_of_two_sum_l3682_368240


namespace greatest_power_of_two_factor_l3682_368266

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 2^502 * k = 15^504 - 6^502) ∧ 
  (∀ m : ℕ, m > 502 → ¬(∃ k : ℕ, 2^m * k = 15^504 - 6^502)) := by
  sorry

end greatest_power_of_two_factor_l3682_368266


namespace new_men_average_age_greater_than_22_l3682_368299

theorem new_men_average_age_greater_than_22 
  (A : ℝ) -- Age of the third man who is not replaced
  (B C : ℝ) -- Ages of the two new men
  (h1 : (A + B + C) / 3 > (A + 21 + 23) / 3) -- Average age increases after replacement
  : (B + C) / 2 > 22 := by
sorry

end new_men_average_age_greater_than_22_l3682_368299


namespace sufficient_not_necessary_l3682_368210

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a + b > 2 ∧ a * b > 1) ∧
  (∃ a b : ℝ, a + b > 2 ∧ a * b > 1 ∧ ¬(a > 1 ∧ b > 1)) :=
by sorry

end sufficient_not_necessary_l3682_368210


namespace complement_of_union_l3682_368263

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_of_union : (U \ (A ∪ B)) = {-2, 3} := by
  sorry

end complement_of_union_l3682_368263


namespace albert_laps_run_l3682_368294

/-- The number of times Albert has already run around the track -/
def laps_run : ℕ := 6

/-- The length of the track in meters -/
def track_length : ℕ := 9

/-- The total distance Albert needs to run in meters -/
def total_distance : ℕ := 99

/-- The number of additional laps Albert will run -/
def additional_laps : ℕ := 5

theorem albert_laps_run :
  laps_run * track_length + additional_laps * track_length = total_distance :=
sorry

end albert_laps_run_l3682_368294


namespace y_intercept_of_given_line_l3682_368211

/-- A line in the form y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis -/
def y_intercept (l : Line) : ℝ := l.b

/-- Given line with equation y = 3x + 2 -/
def given_line : Line := { m := 3, b := 2 }

/-- Theorem: The y-intercept of the given line is 2 -/
theorem y_intercept_of_given_line : 
  y_intercept given_line = 2 := by sorry

end y_intercept_of_given_line_l3682_368211


namespace area_of_circles_with_inscribed_rhombus_l3682_368298

/-- A rhombus inscribed in the intersection of two equal circles -/
structure InscribedRhombus where
  /-- The length of one diagonal of the rhombus -/
  diagonal1 : ℝ
  /-- The length of the other diagonal of the rhombus -/
  diagonal2 : ℝ
  /-- The radius of each circle -/
  radius : ℝ
  /-- The diagonal1 is positive -/
  diagonal1_pos : 0 < diagonal1
  /-- The diagonal2 is positive -/
  diagonal2_pos : 0 < diagonal2
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The rhombus is inscribed in the intersection of the circles -/
  inscribed : (diagonal1 / 2) ^ 2 + (diagonal2 / 2) ^ 2 = radius ^ 2

/-- The theorem stating the relationship between the diagonals and the area of the circles -/
theorem area_of_circles_with_inscribed_rhombus 
  (r : InscribedRhombus) 
  (h1 : r.diagonal1 = 6) 
  (h2 : r.diagonal2 = 12) : 
  π * r.radius ^ 2 = (225 / 4) * π := by
sorry

end area_of_circles_with_inscribed_rhombus_l3682_368298


namespace roof_dimensions_l3682_368209

theorem roof_dimensions (width : ℝ) (length : ℝ) :
  length = 4 * width →
  width * length = 576 →
  length - width = 36 := by
sorry

end roof_dimensions_l3682_368209


namespace income_difference_is_negative_150_l3682_368239

/-- Calculates the difference in income between Janet's first month as a freelancer and her current job -/
def income_difference : ℤ :=
  let current_job_weekly_hours : ℕ := 40
  let current_job_hourly_rate : ℕ := 30
  let freelance_weeks : List ℕ := [30, 35, 40, 50]
  let freelance_rates : List ℕ := [45, 40, 35, 38]
  let extra_fica_tax_weekly : ℕ := 25
  let healthcare_premium_monthly : ℕ := 400
  let increased_rent_monthly : ℕ := 750
  let business_expenses_monthly : ℕ := 150
  let weeks_per_month : ℕ := 4

  let current_job_monthly_income := current_job_weekly_hours * current_job_hourly_rate * weeks_per_month
  
  let freelance_monthly_income := (List.zip freelance_weeks freelance_rates).map (fun (h, r) => h * r) |>.sum
  
  let extra_expenses_monthly := extra_fica_tax_weekly * weeks_per_month + 
                                healthcare_premium_monthly + 
                                increased_rent_monthly + 
                                business_expenses_monthly
  
  let freelance_net_income := freelance_monthly_income - extra_expenses_monthly
  
  freelance_net_income - current_job_monthly_income

theorem income_difference_is_negative_150 : income_difference = -150 := by
  sorry

end income_difference_is_negative_150_l3682_368239


namespace first_number_in_sequence_l3682_368232

def sequence_sum (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 4 → n ≤ 10 → a n = a (n-1) + a (n-2) + a (n-3)

theorem first_number_in_sequence 
  (a : ℕ → ℤ) 
  (h_sum : sequence_sum a) 
  (h_8 : a 8 = 29) 
  (h_9 : a 9 = 56) 
  (h_10 : a 10 = 108) : 
  a 1 = 32 := by sorry

end first_number_in_sequence_l3682_368232


namespace root_sum_reciprocals_l3682_368296

theorem root_sum_reciprocals (p q r s : ℂ) : 
  p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0 →
  q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0 →
  r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0 →
  s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0 →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3 := by
sorry

end root_sum_reciprocals_l3682_368296


namespace problem_solution_l3682_368202

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - |x - 2|

-- State the theorem
theorem problem_solution (m : ℝ) (a b c x y z : ℝ) 
  (h1 : ∀ x, f m (x + 1) ≥ 0 ↔ 0 ≤ x ∧ x ≤ 1)
  (h2 : x^2 + y^2 + z^2 = a^2 + b^2 + c^2)
  (h3 : x^2 + y^2 + z^2 = m) :
  m = 1 ∧ a*x + b*y + c*z ≤ 1 := by
sorry

end problem_solution_l3682_368202


namespace madeline_sleep_hours_madeline_sleeps_eight_hours_l3682_368220

/-- Calculates the number of hours Madeline spends sleeping per day given her weekly schedule. -/
theorem madeline_sleep_hours (class_hours_per_week : ℕ) 
                              (homework_hours_per_day : ℕ) 
                              (work_hours_per_week : ℕ) 
                              (leftover_hours_per_week : ℕ) : ℕ :=
  let total_hours_per_week : ℕ := 24 * 7
  let remaining_hours : ℕ := total_hours_per_week - class_hours_per_week - 
                             (homework_hours_per_day * 7) - work_hours_per_week - 
                             leftover_hours_per_week
  remaining_hours / 7

/-- Proves that Madeline spends 8 hours per day sleeping given her schedule. -/
theorem madeline_sleeps_eight_hours : 
  madeline_sleep_hours 18 4 20 46 = 8 := by
  sorry

end madeline_sleep_hours_madeline_sleeps_eight_hours_l3682_368220


namespace linear_system_solution_l3682_368221

theorem linear_system_solution (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (eq1 : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ = 0)
  (eq2 : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ = 10)
  (eq3 : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ = 100) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ = 210 := by
sorry

end linear_system_solution_l3682_368221


namespace unique_solution_for_pure_imaginary_l3682_368244

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number constructed from m -/
def complex_number (m : ℝ) : ℂ :=
  ⟨m^2 - 5*m + 6, m^2 - 3*m⟩

theorem unique_solution_for_pure_imaginary :
  ∃! m : ℝ, is_pure_imaginary (complex_number m) ∧ m = 2 :=
sorry

end unique_solution_for_pure_imaginary_l3682_368244


namespace intersection_line_equation_l3682_368218

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def circle1 : Circle := { center := (-5, -3), radius := 15 }
def circle2 : Circle := { center := (4, 15), radius := 9 }

/-- The line passing through the intersection points of two circles -/
def intersectionLine (c1 c2 : Circle) : Line := sorry

theorem intersection_line_equation :
  let l := intersectionLine circle1 circle2
  l.a = 1 ∧ l.b = 1 ∧ l.c = -27/4 := by sorry

end intersection_line_equation_l3682_368218


namespace square_of_binomial_constant_l3682_368233

theorem square_of_binomial_constant (b : ℝ) : 
  (∃ (a c : ℝ), ∀ x, 16 * x^2 + 40 * x + b = (a * x + c)^2) → b = 25 := by
  sorry

end square_of_binomial_constant_l3682_368233


namespace golden_rabbit_cards_l3682_368216

def total_cards : ℕ := 10000
def digits_without_6_8 : ℕ := 8

theorem golden_rabbit_cards :
  (total_cards - digits_without_6_8^4 : ℕ) = 5904 := by sorry

end golden_rabbit_cards_l3682_368216


namespace polynomial_integer_solutions_l3682_368237

theorem polynomial_integer_solutions :
  ∀ n : ℤ, n^5 - 2*n^4 - 7*n^2 - 7*n + 3 = 0 ↔ n = -1 ∨ n = 3 := by
  sorry

end polynomial_integer_solutions_l3682_368237


namespace basketball_lineup_count_l3682_368242

theorem basketball_lineup_count :
  let total_players : ℕ := 12
  let point_guards : ℕ := 1
  let other_players : ℕ := 5
  Nat.choose total_players point_guards * Nat.choose (total_players - point_guards) other_players = 5544 := by
  sorry

end basketball_lineup_count_l3682_368242


namespace john_sleep_for_target_score_l3682_368255

/-- Represents the relationship between sleep hours and exam score -/
structure ExamPerformance where
  sleep : ℝ
  score : ℝ

/-- The inverse relationship between sleep and score -/
def inverseRelation (e1 e2 : ExamPerformance) : Prop :=
  e1.sleep * e1.score = e2.sleep * e2.score

theorem john_sleep_for_target_score 
  (e1 : ExamPerformance) 
  (e2 : ExamPerformance) 
  (h1 : e1.sleep = 6) 
  (h2 : e1.score = 80) 
  (h3 : inverseRelation e1 e2) 
  (h4 : (e1.score + e2.score) / 2 = 85) : 
  e2.sleep = 16 / 3 := by
  sorry

end john_sleep_for_target_score_l3682_368255


namespace xyz_value_l3682_368276

theorem xyz_value (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 195)
  (h2 : y * (z + x) = 204)
  (h3 : z * (x + y) = 213) : 
  x * y * z = 1029 := by
sorry

end xyz_value_l3682_368276


namespace linear_combination_existence_l3682_368228

theorem linear_combination_existence (n : ℕ+) (a b c : ℕ+) 
  (ha : a ≤ 3*n^2 + 4*n) (hb : b ≤ 3*n^2 + 4*n) (hc : c ≤ 3*n^2 + 4*n) :
  ∃ (x y z : ℤ), 
    (abs x ≤ 2*n ∧ abs y ≤ 2*n ∧ abs z ≤ 2*n) ∧ 
    (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    (a*x + b*y + c*z = 0) :=
by sorry

end linear_combination_existence_l3682_368228


namespace sum_greatest_odd_divisors_formula_l3682_368226

/-- The greatest odd divisor of a positive integer -/
def greatest_odd_divisor (k : ℕ+) : ℕ+ :=
  sorry

/-- The sum of greatest odd divisors from 1 to 2^n -/
def sum_greatest_odd_divisors (n : ℕ+) : ℕ+ :=
  sorry

theorem sum_greatest_odd_divisors_formula (n : ℕ+) :
  (sum_greatest_odd_divisors n : ℚ) = (4^(n : ℕ) + 5) / 3 :=
sorry

end sum_greatest_odd_divisors_formula_l3682_368226


namespace larger_integer_value_l3682_368217

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (b : ℚ) / (a : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  (b : ℕ) = 21 := by
  sorry

end larger_integer_value_l3682_368217


namespace power_product_six_three_l3682_368247

theorem power_product_six_three : (6 : ℕ)^5 * (3 : ℕ)^5 = 1889568 := by
  sorry

end power_product_six_three_l3682_368247


namespace cheryl_skittles_l3682_368231

theorem cheryl_skittles (initial : ℕ) (given : ℕ) (final : ℕ) : 
  given = 89 → final = 97 → initial + given = final → initial = 8 := by
sorry

end cheryl_skittles_l3682_368231


namespace radhika_final_game_count_l3682_368293

def initial_games_ratio : Rat := 2/3
def christmas_games : Nat := 12
def birthday_games : Nat := 8
def family_gathering_games : Nat := 5
def additional_purchased_games : Nat := 6

theorem radhika_final_game_count :
  let total_gifted_games := christmas_games + birthday_games + family_gathering_games
  let initial_games := (initial_games_ratio * total_gifted_games).floor
  let games_after_gifts := initial_games + total_gifted_games
  let final_game_count := games_after_gifts + additional_purchased_games
  final_game_count = 47 := by
  sorry

end radhika_final_game_count_l3682_368293


namespace family_member_bites_l3682_368250

-- Define the number of mosquito bites Cyrus got on arms and legs
def cyrus_arms_legs_bites : ℕ := 14

-- Define the number of mosquito bites Cyrus got on his body
def cyrus_body_bites : ℕ := 10

-- Define the number of other family members
def family_members : ℕ := 6

-- Define Cyrus' total bites
def cyrus_total_bites : ℕ := cyrus_arms_legs_bites + cyrus_body_bites

-- Define the family's total bites
def family_total_bites : ℕ := cyrus_total_bites / 2

-- Theorem to prove
theorem family_member_bites :
  family_total_bites / family_members = 2 :=
by sorry

end family_member_bites_l3682_368250


namespace go_square_side_count_l3682_368246

/-- Represents a square arrangement of Go stones -/
structure GoSquare where
  side_length : ℕ
  perimeter_stones : ℕ

/-- The number of stones on one side of a GoSquare -/
def stones_on_side (square : GoSquare) : ℕ := square.side_length

/-- The number of stones on the perimeter of a GoSquare -/
def perimeter_count (square : GoSquare) : ℕ := square.perimeter_stones

theorem go_square_side_count (square : GoSquare) 
  (h : perimeter_count square = 84) : 
  stones_on_side square = 22 := by
  sorry

end go_square_side_count_l3682_368246


namespace sum_greater_than_four_l3682_368289

theorem sum_greater_than_four (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (h : 1/a + 1/b = 1) : a + b > 4 := by
  sorry

end sum_greater_than_four_l3682_368289


namespace first_nonzero_digit_of_one_over_157_l3682_368285

theorem first_nonzero_digit_of_one_over_157 : ∃ (n : ℕ), 
  (1000 : ℚ) / 157 > 6 ∧ (1000 : ℚ) / 157 < 7 ∧ 
  (1000 * (1 : ℚ) / 157 - 6) * 10 ≥ 3 ∧ (1000 * (1 : ℚ) / 157 - 6) * 10 < 4 := by
  sorry

end first_nonzero_digit_of_one_over_157_l3682_368285


namespace die_roll_probability_l3682_368245

/-- The probability of getting a specific number on a standard six-sided die -/
def prob_match : ℚ := 1 / 6

/-- The probability of not getting a specific number on a standard six-sided die -/
def prob_no_match : ℚ := 5 / 6

/-- The number of rolls -/
def n : ℕ := 12

/-- The number of ways to choose the position of the first pair of consecutive matches -/
def ways_to_choose_first_pair : ℕ := n - 2

theorem die_roll_probability :
  (ways_to_choose_first_pair : ℚ) * prob_no_match^(n - 3) * prob_match^2 = 19531250 / 362797056 := by
  sorry

end die_roll_probability_l3682_368245


namespace mod_equation_l3682_368200

theorem mod_equation (m : ℕ) (h1 : m < 37) (h2 : (4 * m) % 37 = 1) :
  (3^m)^2 % 37 - 3 % 37 = 19 := by
  sorry

end mod_equation_l3682_368200


namespace rectangle_area_l3682_368278

theorem rectangle_area (width height : ℝ) (h1 : width / height = 0.875) (h2 : height = 24) :
  width * height = 504 := by
  sorry

end rectangle_area_l3682_368278


namespace red_balls_count_l3682_368284

theorem red_balls_count (black_balls white_balls : ℕ) (red_prob : ℝ) : 
  black_balls = 8 → white_balls = 4 → red_prob = 0.4 → 
  (black_balls + white_balls : ℝ) / (1 - red_prob) = black_balls + white_balls + 8 := by
  sorry

#check red_balls_count

end red_balls_count_l3682_368284


namespace wage_increase_percentage_l3682_368281

theorem wage_increase_percentage (original_wage new_wage : ℝ) 
  (h1 : original_wage = 34)
  (h2 : new_wage = 51) :
  (new_wage - original_wage) / original_wage * 100 = 50 := by
  sorry

end wage_increase_percentage_l3682_368281


namespace inverse_of_B_cubed_l3682_368282

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = !![3, -2; 1, 0] →
  (B^3)⁻¹ = !![15, -14; 7, -6] := by
sorry

end inverse_of_B_cubed_l3682_368282


namespace car_rental_savings_l3682_368243

def trip_distance : ℝ := 150
def first_option_cost : ℝ := 50
def second_option_cost : ℝ := 90
def gasoline_efficiency : ℝ := 15
def gasoline_cost_per_liter : ℝ := 0.9

theorem car_rental_savings : 
  let total_distance := 2 * trip_distance
  let gasoline_needed := total_distance / gasoline_efficiency
  let gasoline_cost := gasoline_needed * gasoline_cost_per_liter
  let first_option_total := first_option_cost + gasoline_cost
  second_option_cost - first_option_total = 22 := by sorry

end car_rental_savings_l3682_368243


namespace chocolate_bar_theorem_l3682_368219

theorem chocolate_bar_theorem (n m : ℕ) (h : n * m = 25) :
  (∃ (b w : ℕ), b + w = n * m ∧ b = w + 1 ∧ b = (25 * w) / 3) →
  n + m = 10 := by
sorry

end chocolate_bar_theorem_l3682_368219


namespace insects_in_lab_l3682_368253

/-- The number of insects in a laboratory given the total number of insect legs and legs per insect. -/
def num_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Theorem: There are 8 insects in the laboratory. -/
theorem insects_in_lab : num_insects 48 6 = 8 := by
  sorry

end insects_in_lab_l3682_368253


namespace x_value_proof_l3682_368287

theorem x_value_proof :
  let equation := (2021 / 2022 - 2022 / 2021) + x = 0
  ∃ x, equation ∧ x = 2022 / 2021 - 2021 / 2022 :=
by
  sorry

end x_value_proof_l3682_368287


namespace circle_area_increase_l3682_368297

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := r * 2.5
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 8 := by
sorry

end circle_area_increase_l3682_368297


namespace original_average_rent_l3682_368290

theorem original_average_rent
  (num_friends : ℕ)
  (original_rent : ℝ)
  (increased_rent : ℝ)
  (new_average : ℝ)
  (h1 : num_friends = 4)
  (h2 : original_rent = 1250)
  (h3 : increased_rent = 1250 * 1.16)
  (h4 : new_average = 850)
  : (num_friends * new_average - increased_rent + original_rent) / num_friends = 800 := by
  sorry

end original_average_rent_l3682_368290


namespace initial_men_count_prove_initial_men_count_l3682_368254

/-- The number of men initially working on a project -/
def initial_men : ℕ := 12

/-- The number of hours worked per day by the initial group -/
def initial_hours_per_day : ℕ := 8

/-- The number of days worked by the initial group -/
def initial_days : ℕ := 10

/-- The number of men in the second group -/
def second_group_men : ℕ := 5

/-- The number of hours worked per day by the second group -/
def second_hours_per_day : ℕ := 16

/-- The number of days worked by the second group -/
def second_days : ℕ := 12

theorem initial_men_count : 
  initial_men * initial_hours_per_day * initial_days = 
  second_group_men * second_hours_per_day * second_days :=
by sorry

/-- The main theorem proving the number of men initially working -/
theorem prove_initial_men_count : initial_men = 12 :=
by sorry

end initial_men_count_prove_initial_men_count_l3682_368254


namespace contest_result_l3682_368267

/-- The number of baskets made by Alex, Sandra, Hector, and Jordan -/
def total_baskets (alex sandra hector jordan : ℕ) : ℕ :=
  alex + sandra + hector + jordan

/-- Theorem stating the total number of baskets under given conditions -/
theorem contest_result : ∃ (alex sandra hector jordan : ℕ),
  alex = 8 ∧
  sandra = 3 * alex ∧
  hector = 2 * sandra ∧
  jordan = (alex + sandra + hector) / 5 ∧
  total_baskets alex sandra hector jordan = 96 := by
  sorry

end contest_result_l3682_368267


namespace no_rational_sqrt_sin_cos_l3682_368223

theorem no_rational_sqrt_sin_cos : 
  ¬ ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ 
    ∃ (a b c d : ℕ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∧
    (Real.sqrt (Real.sin θ) = a / b) ∧ 
    (Real.sqrt (Real.cos θ) = c / d) :=
by sorry

end no_rational_sqrt_sin_cos_l3682_368223


namespace least_x_for_integer_fraction_l3682_368241

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem least_x_for_integer_fraction :
  ∀ x : ℝ, (is_integer (24 / (x - 4)) ∧ x < -20) → False :=
by sorry

end least_x_for_integer_fraction_l3682_368241


namespace simson_line_l3682_368248

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the properties and relations
variable (incircle : Point → Point → Point → Point → Prop)
variable (on_circle : Point → Point → Point → Point → Prop)
variable (perpendicular : Point → Point → Line → Prop)
variable (on_line : Point → Line → Prop)
variable (collinear : Point → Point → Point → Prop)

-- Define the theorem
theorem simson_line 
  (A B C P U V W : Point) 
  (circle : Line) 
  (BC CA AB : Line) :
  incircle A B C P →
  on_circle A B C P →
  perpendicular P U BC →
  perpendicular P V CA →
  perpendicular P W AB →
  on_line U BC →
  on_line V CA →
  on_line W AB →
  collinear U V W :=
sorry

end simson_line_l3682_368248

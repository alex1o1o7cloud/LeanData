import Mathlib

namespace NUMINAMATH_CALUDE_m_greater_than_n_l1061_106160

/-- A line with slope -1 and y-intercept b -/
def line (b : ℝ) := fun (x : ℝ) ↦ -x + b

/-- Point A lies on the line -/
def point_A_on_line (m b : ℝ) : Prop := line b (-5) = m

/-- Point B lies on the line -/
def point_B_on_line (n b : ℝ) : Prop := line b 4 = n

theorem m_greater_than_n (m n b : ℝ) :
  point_A_on_line m b → point_B_on_line n b → m > n := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l1061_106160


namespace NUMINAMATH_CALUDE_dobarulho_solutions_l1061_106166

def is_dobarulho (A B C D : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 8 ∧
  1 ≤ B ∧ B ≤ 9 ∧
  1 ≤ C ∧ C ≤ 9 ∧
  D > 1 ∧
  D ∣ (100*A + 10*B + C) ∧
  D ∣ (100*B + 10*C + A) ∧
  D ∣ (100*C + 10*A + B) ∧
  D ∣ (100*(A+1) + 10*C + B) ∧
  D ∣ (100*C + 10*B + (A+1)) ∧
  D ∣ (100*B + 10*(A+1) + C)

theorem dobarulho_solutions :
  ∀ A B C D : ℕ, is_dobarulho A B C D ↔ 
    ((A = 3 ∧ B = 7 ∧ C = 0 ∧ D = 37) ∨
     (A = 4 ∧ B = 8 ∧ C = 1 ∧ D = 37) ∨
     (A = 5 ∧ B = 9 ∧ C = 2 ∧ D = 37)) :=
by sorry

end NUMINAMATH_CALUDE_dobarulho_solutions_l1061_106166


namespace NUMINAMATH_CALUDE_football_team_handedness_l1061_106158

theorem football_team_handedness (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ)
  (h1 : total_players = 70)
  (h2 : throwers = 46)
  (h3 : right_handed = 62)
  (h4 : throwers ≤ right_handed) :
  (total_players - right_handed : ℚ) / (total_players - throwers) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_football_team_handedness_l1061_106158


namespace NUMINAMATH_CALUDE_openai_robot_weight_problem_l1061_106128

/-- The OpenAI robotics competition weight problem -/
theorem openai_robot_weight_problem :
  let standard_weight : ℕ := 100
  let max_weight : ℕ := 210
  let min_additional_weight : ℕ := max_weight - standard_weight - (max_weight - standard_weight) / 2
  (2 * (standard_weight + min_additional_weight) ≤ max_weight) ∧
  (∀ w : ℕ, w < min_additional_weight → 2 * (standard_weight + w) > max_weight) →
  min_additional_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_openai_robot_weight_problem_l1061_106128


namespace NUMINAMATH_CALUDE_triangle_cover_theorem_l1061_106139

/-- A triangle with sides 2, 3, and 4 -/
structure Triangle :=
  (side1 : ℝ := 2)
  (side2 : ℝ := 3)
  (side3 : ℝ := 4)

/-- A circle with unit radius -/
structure UnitCircle :=
  (radius : ℝ := 1)

/-- The minimum number of unit circles required to cover the triangle -/
def minCoverCircles (t : Triangle) : ℕ :=
  3

/-- Theorem stating that the minimum number of unit circles required to cover the triangle is 3 -/
theorem triangle_cover_theorem (t : Triangle) :
  minCoverCircles t = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_cover_theorem_l1061_106139


namespace NUMINAMATH_CALUDE_range_of_x_plus_y_l1061_106190

theorem range_of_x_plus_y (x y : ℝ) (h : x - Real.sqrt (x + 1) = Real.sqrt (y + 1) - y) :
  ∃ (s : ℝ), s ∈ Set.Icc (1 - Real.sqrt 5) (1 + Real.sqrt 5) ∧ x + y = s :=
sorry

end NUMINAMATH_CALUDE_range_of_x_plus_y_l1061_106190


namespace NUMINAMATH_CALUDE_son_work_time_l1061_106180

-- Define the work rates
def man_rate : ℚ := 1 / 5
def combined_rate : ℚ := 1 / 4

-- Define the son's work rate
def son_rate : ℚ := combined_rate - man_rate

-- Theorem statement
theorem son_work_time :
  son_rate = 1 / 20 ∧ (1 / son_rate : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_son_work_time_l1061_106180


namespace NUMINAMATH_CALUDE_survey_respondents_l1061_106100

/-- Prove that the number of customers who responded to a survey is 50, given the following conditions:
  1. The average income of all customers is $45,000
  2. There are 10 wealthiest customers
  3. The average income of the 10 wealthiest customers is $55,000
  4. The average income of the remaining customers is $42,500
-/
theorem survey_respondents (N : ℕ) : 
  (10 * 55000 + (N - 10) * 42500 = N * 45000) → N = 50 := by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l1061_106100


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1061_106135

open Real

theorem min_value_trig_expression (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (sin x + 3 * (1 / sin x))^2 + (cos x + 3 * (1 / cos x))^2 ≥ 52 ∧
  ∃ y, 0 < y ∧ y < π / 2 ∧ (sin y + 3 * (1 / sin y))^2 + (cos y + 3 * (1 / cos y))^2 = 52 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1061_106135


namespace NUMINAMATH_CALUDE_decimal_division_l1061_106134

theorem decimal_division (x y : ℚ) (hx : x = 0.54) (hy : y = 0.006) : x / y = 90 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l1061_106134


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l1061_106143

theorem largest_prime_factors_difference (n : Nat) (h : n = 261943) :
  ∃ (p q : Nat), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p > q ∧ 
    (∀ r : Nat, Nat.Prime r ∧ r ∣ n → r ≤ p) ∧
    (∀ r : Nat, Nat.Prime r ∧ r ∣ n ∧ r ≠ p → r ≤ q) ∧
    p - q = 110 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l1061_106143


namespace NUMINAMATH_CALUDE_equation_solutions_l1061_106164

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁^2 - 4*x₁ - 8 = 0 ∧ x₂^2 - 4*x₂ - 8 = 0) ∧
    x₁ = 2*Real.sqrt 3 + 2 ∧ x₂ = -2*Real.sqrt 3 + 2) ∧
  (∃ y₁ y₂ : ℝ, (3*y₁ - 6 = y₁*(y₁ - 2) ∧ 3*y₂ - 6 = y₂*(y₂ - 2)) ∧
    y₁ = 2 ∧ y₂ = 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1061_106164


namespace NUMINAMATH_CALUDE_binary_10111_equals_23_l1061_106176

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_10111_equals_23 :
  binary_to_decimal [true, true, true, false, true] = 23 := by
  sorry

end NUMINAMATH_CALUDE_binary_10111_equals_23_l1061_106176


namespace NUMINAMATH_CALUDE_dollar_composition_60_l1061_106116

-- Define the $ operation
def dollar (x : ℝ) : ℝ := 0.4 * x + 2

-- State the theorem
theorem dollar_composition_60 : dollar (dollar (dollar 60)) = 6.96 := by
  sorry

end NUMINAMATH_CALUDE_dollar_composition_60_l1061_106116


namespace NUMINAMATH_CALUDE_circle_C_area_l1061_106163

-- Define the circle C
def circle_C (x y r : ℝ) : Prop := (x + 2)^2 + y^2 = r^2

-- Define the parabola D
def parabola_D (x y : ℝ) : Prop := y^2 = 20 * x

-- Define the axis of symmetry
def axis_of_symmetry : ℝ := -5

-- Define the distance between the center of C and the axis of symmetry
def distance_to_axis : ℝ := 3

-- Define the length of AB
def length_AB : ℝ := 8

-- Theorem to prove
theorem circle_C_area :
  ∃ (r : ℝ), 
    (∀ x y, circle_C x y r → parabola_D x y → x = axis_of_symmetry) ∧
    length_AB = 8 ∧
    distance_to_axis = 3 →
    π * r^2 = 25 * π :=
sorry

end NUMINAMATH_CALUDE_circle_C_area_l1061_106163


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1061_106106

/-- An arithmetic sequence with sum S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ

/-- Theorem: For an arithmetic sequence with S_3 = 9 and S_6 = 27, S_9 = 54 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h1 : a.S 3 = 9) 
  (h2 : a.S 6 = 27) : 
  a.S 9 = 54 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1061_106106


namespace NUMINAMATH_CALUDE_steves_return_speed_l1061_106131

/-- Proves that given a round trip of 35 km each way, where the return speed is twice the outbound speed, 
    and the total travel time is 6 hours, the return speed is 17.5 km/h. -/
theorem steves_return_speed (distance : ℝ) (total_time : ℝ) : 
  distance = 35 →
  total_time = 6 →
  ∃ (outbound_speed : ℝ),
    outbound_speed > 0 ∧
    distance / outbound_speed + distance / (2 * outbound_speed) = total_time ∧
    2 * outbound_speed = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_steves_return_speed_l1061_106131


namespace NUMINAMATH_CALUDE_large_monkey_doll_cost_l1061_106182

/-- The cost of a large monkey doll satisfies the given conditions -/
theorem large_monkey_doll_cost : ∃ (L : ℚ), 
  (L > 0) ∧ 
  (320 / (L - 4) = 320 / L + 40) ∧ 
  L = 8 := by
  sorry

end NUMINAMATH_CALUDE_large_monkey_doll_cost_l1061_106182


namespace NUMINAMATH_CALUDE_dark_tile_fraction_is_one_fourth_l1061_106199

/-- Represents a 4x4 tile pattern -/
structure TilePattern :=
  (darkTilesInRow : Fin 4 → Nat)
  (h_valid : ∀ i, darkTilesInRow i ≤ 4)

/-- The specific tile pattern described in the problem -/
def problemPattern : TilePattern :=
  { darkTilesInRow := λ i => if i.val < 2 then 2 else 0,
    h_valid := by sorry }

/-- The fraction of dark tiles in a given pattern -/
def darkTileFraction (pattern : TilePattern) : Rat :=
  (pattern.darkTilesInRow 0 + pattern.darkTilesInRow 1 + 
   pattern.darkTilesInRow 2 + pattern.darkTilesInRow 3) / 16

theorem dark_tile_fraction_is_one_fourth :
  darkTileFraction problemPattern = 1/4 := by sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_is_one_fourth_l1061_106199


namespace NUMINAMATH_CALUDE_three_fourths_of_forty_l1061_106136

theorem three_fourths_of_forty : (3 / 4 : ℚ) * 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_of_forty_l1061_106136


namespace NUMINAMATH_CALUDE_cornelia_european_countries_l1061_106172

/-- Represents the number of countries visited in different regions --/
structure CountriesVisited where
  total : Nat
  southAmerica : Nat
  asia : Nat

/-- Calculates the number of European countries visited --/
def europeanCountries (c : CountriesVisited) : Nat :=
  c.total - c.southAmerica - 2 * c.asia

/-- Theorem stating that Cornelia visited 20 European countries --/
theorem cornelia_european_countries :
  ∃ c : CountriesVisited, c.total = 42 ∧ c.southAmerica = 10 ∧ c.asia = 6 ∧ europeanCountries c = 20 := by
  sorry

end NUMINAMATH_CALUDE_cornelia_european_countries_l1061_106172


namespace NUMINAMATH_CALUDE_locus_and_line_theorem_l1061_106175

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (3, 0)

-- Define the condition for P being outside C
def outside_circle (x y : ℝ) : Prop := x^2 + (y - 2)^2 > 1

-- Define the tangent condition (implicitly used in the problem)
def is_tangent (x y : ℝ) : Prop := 
  ∃ (t : ℝ), circle_C (x + t) (y + t) ∧ ¬(∃ (s : ℝ), s ≠ t ∧ circle_C (x + s) (y + s))

-- Define the PQ = √2 * PA condition
def pq_pa_relation (x y : ℝ) : Prop :=
  (x^2 + (y - 2)^2 - 1) = 2 * ((x - 3)^2 + y^2)

-- Define the locus of P
def locus_P (x y : ℝ) : Prop := x^2 + y^2 - 12*x + 4*y + 15 = 0

-- Define the condition for line l
def line_l (x y : ℝ) : Prop := x = 3 ∨ 5*x - 12*y - 15 = 0

-- Main theorem
theorem locus_and_line_theorem :
  ∀ (x y : ℝ),
    outside_circle x y →
    is_tangent x y →
    pq_pa_relation x y →
    (locus_P x y ∧
     (∃ (m n : ℝ × ℝ),
       locus_P m.1 m.2 ∧
       locus_P n.1 n.2 ∧
       line_l m.1 m.2 ∧
       line_l n.1 n.2 ∧
       (m.1 - n.1)^2 + (m.2 - n.2)^2 = 64)) :=
by sorry

end NUMINAMATH_CALUDE_locus_and_line_theorem_l1061_106175


namespace NUMINAMATH_CALUDE_distinct_paths_eq_120_l1061_106156

/-- The number of distinct paths in a grid from point C to point D,
    where every step must either move up or to the right,
    and one has to move 7 steps to the right and 3 steps up. -/
def distinct_paths : ℕ := Nat.choose 10 3

/-- Theorem stating that the number of distinct paths is equal to 120. -/
theorem distinct_paths_eq_120 : distinct_paths = 120 := by sorry

end NUMINAMATH_CALUDE_distinct_paths_eq_120_l1061_106156


namespace NUMINAMATH_CALUDE_larger_part_is_30_l1061_106132

theorem larger_part_is_30 (x y : ℕ) 
  (sum_eq_52 : x + y = 52) 
  (weighted_sum_eq_780 : 10 * x + 22 * y = 780) : 
  max x y = 30 := by
sorry

end NUMINAMATH_CALUDE_larger_part_is_30_l1061_106132


namespace NUMINAMATH_CALUDE_cylinder_volume_equalization_l1061_106178

/-- The increase in radius and height that equalizes volumes of two cylinders --/
theorem cylinder_volume_equalization (r h : ℝ) (increase : ℝ) : 
  r = 5 ∧ h = 10 ∧ increase > 0 → 
  π * (r + increase)^2 * h = π * r^2 * (h + increase) → 
  increase = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_equalization_l1061_106178


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l1061_106122

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 12500

/-- The scientific notation representation of the number -/
def scientificForm : ScientificNotation := {
  coefficient := 1.25,
  exponent := 4,
  coeff_range := by sorry
}

/-- Theorem stating that the scientific notation form is equal to the original number -/
theorem scientific_notation_correct : 
  (scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent) = number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l1061_106122


namespace NUMINAMATH_CALUDE_parabola_distance_l1061_106188

theorem parabola_distance : ∀ (x_p x_q : ℝ),
  (x_p^2 - 2*x_p - 8 = 8) →
  (x_q^2 - 2*x_q - 8 = -4) →
  (∀ x, x^2 - 2*x - 8 = -4 → |x - x_p| ≥ |x_q - x_p|) →
  |x_q - x_p| = |Real.sqrt 17 - Real.sqrt 5| :=
by sorry

end NUMINAMATH_CALUDE_parabola_distance_l1061_106188


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1061_106189

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ z w : ℝ, z > 0 → w > 0 → 2*z + 8*w - z*w = 0 → x + y ≤ z + w ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 8*y₀ - x₀*y₀ = 0 ∧ x₀ + y₀ = 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1061_106189


namespace NUMINAMATH_CALUDE_exists_square_with_digit_sum_2002_l1061_106127

/-- Sum of digits of a natural number in base 10 -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a square number whose sum of digits in base 10 is 2002 -/
theorem exists_square_with_digit_sum_2002 : 
  ∃ n : ℕ, sum_of_digits (n^2) = 2002 := by sorry

end NUMINAMATH_CALUDE_exists_square_with_digit_sum_2002_l1061_106127


namespace NUMINAMATH_CALUDE_pythagorean_diagonal_l1061_106149

theorem pythagorean_diagonal (m : ℕ) (h : m ≥ 3) :
  let width := 2 * m
  let diagonal := m^2 - 1
  let height := diagonal - 2
  width^2 + height^2 = diagonal^2 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_diagonal_l1061_106149


namespace NUMINAMATH_CALUDE_intersection_chord_length_l1061_106187

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_chord_length :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l1061_106187


namespace NUMINAMATH_CALUDE_triangle_side_length_l1061_106155

/-- Given a triangle ABC with side lengths a and b, and angle C (in radians),
    proves that the length of side c is equal to √2 -/
theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  a = 2 → b = Real.sqrt 3 - 1 → C = π / 6 → c = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1061_106155


namespace NUMINAMATH_CALUDE_equation_solutions_l1061_106165

theorem equation_solutions :
  (∃ x : ℝ, (x + 1)^3 = 64 ∧ x = 3) ∧
  (∃ x : ℝ, (2*x + 1)^2 = 81 ∧ (x = 4 ∨ x = -5)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1061_106165


namespace NUMINAMATH_CALUDE_additional_men_count_l1061_106103

theorem additional_men_count (initial_men : ℕ) (initial_days : ℕ) (final_days : ℕ) :
  initial_men = 600 →
  initial_days = 20 →
  final_days = 15 →
  ∃ (additional_men : ℕ), 
    initial_men * initial_days = (initial_men + additional_men) * final_days ∧
    additional_men = 200 := by
  sorry

end NUMINAMATH_CALUDE_additional_men_count_l1061_106103


namespace NUMINAMATH_CALUDE_expression_evaluation_l1061_106154

theorem expression_evaluation (z p q : ℝ) (hz : z ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0) :
  ((z^(2/p) + z^(2/q))^2 - 4*z^(2/p + 2/q)) / ((z^(1/p) - z^(1/q))^2 + 4*z^(1/p + 1/q)) = (|z^(1/p) - z^(1/q)|)^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1061_106154


namespace NUMINAMATH_CALUDE_distinct_elements_l1061_106194

theorem distinct_elements (x : ℕ) : 
  (5 ≠ x ∧ 5 ≠ x^2 - 4*x ∧ x ≠ x^2 - 4*x) ↔ (x ≠ 5 ∧ x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_distinct_elements_l1061_106194


namespace NUMINAMATH_CALUDE_stevens_cards_l1061_106157

def number_of_groups : ℕ := 5
def cards_per_group : ℕ := 6

theorem stevens_cards : number_of_groups * cards_per_group = 30 := by
  sorry

end NUMINAMATH_CALUDE_stevens_cards_l1061_106157


namespace NUMINAMATH_CALUDE_sum_of_powers_positive_l1061_106173

theorem sum_of_powers_positive (a b c : ℝ) (h1 : a * b * c > 0) (h2 : a + b + c > 0) :
  ∀ n : ℕ, a ^ n + b ^ n + c ^ n > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_positive_l1061_106173


namespace NUMINAMATH_CALUDE_square_of_cube_of_third_smallest_prime_l1061_106162

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem square_of_cube_of_third_smallest_prime :
  (nthSmallestPrime 3) ^ 3 ^ 2 = 15625 := by sorry

end NUMINAMATH_CALUDE_square_of_cube_of_third_smallest_prime_l1061_106162


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1061_106170

/-- Given vectors a and b, prove that if k*a + b is parallel to a - 3*b, then k = -1/3 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : ∃ (t : ℝ), t ≠ 0 ∧ (k • a + b) = t • (a - 3 • b)) :
  k = -1/3 := by
  sorry

#check parallel_vectors_k_value

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1061_106170


namespace NUMINAMATH_CALUDE_limit_of_ratio_l1061_106140

def arithmetic_sequence (n : ℕ) : ℝ := 2 * n - 1

def sum_of_terms (n : ℕ) : ℝ := n^2

theorem limit_of_ratio :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sum_of_terms n / (arithmetic_sequence n)^2 - 1/4| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_ratio_l1061_106140


namespace NUMINAMATH_CALUDE_candle_height_shadow_relation_l1061_106102

/-- Given two positions of a gnomon and the shadows cast, we can relate the height of the object to the shadow lengths and distance between positions. -/
theorem candle_height_shadow_relation 
  (h : ℝ) -- height of the candle
  (d : ℝ) -- distance between gnomon positions
  (a : ℝ) -- length of shadow in first position
  (b : ℝ) -- length of shadow in second position
  (x : ℝ) -- length from base of candle at first position to end of shadow in second position plus d
  (h_pos : h > 0)
  (d_pos : d > 0)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (x_def : x = d + b) -- definition of x
  : x = h * (1 + d / (a + b)) := by
  sorry


end NUMINAMATH_CALUDE_candle_height_shadow_relation_l1061_106102


namespace NUMINAMATH_CALUDE_keith_total_spent_l1061_106148

def speakers_cost : ℚ := 136.01
def cd_player_cost : ℚ := 139.38
def tires_cost : ℚ := 112.46

theorem keith_total_spent : 
  speakers_cost + cd_player_cost + tires_cost = 387.85 := by
  sorry

end NUMINAMATH_CALUDE_keith_total_spent_l1061_106148


namespace NUMINAMATH_CALUDE_max_value_of_f_l1061_106130

theorem max_value_of_f (x y z u v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0) (hv : v > 0) :
  (x * y + y * z + z * u + u * v) / (2 * x^2 + y^2 + 2 * z^2 + u^2 + 2 * v^2) ≤ Real.sqrt 6 / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1061_106130


namespace NUMINAMATH_CALUDE_shirts_washed_l1061_106123

theorem shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (unwashed : ℕ) :
  short_sleeve = 9 →
  long_sleeve = 21 →
  unwashed = 1 →
  short_sleeve + long_sleeve - unwashed = 29 := by
sorry

end NUMINAMATH_CALUDE_shirts_washed_l1061_106123


namespace NUMINAMATH_CALUDE_hexagon_coverage_percentage_l1061_106193

structure Tile :=
  (grid_size : Nat)
  (square_count : Nat)
  (hexagon_count : Nat)

def Region :=
  {t : Tile // t.grid_size = 4 ∧ t.square_count = 8 ∧ t.hexagon_count = 8}

theorem hexagon_coverage_percentage (r : Region) : 
  (r.val.hexagon_count : ℚ) / (r.val.grid_size^2 : ℚ) * 100 = 50 :=
sorry

end NUMINAMATH_CALUDE_hexagon_coverage_percentage_l1061_106193


namespace NUMINAMATH_CALUDE_indira_cricket_minutes_l1061_106119

/-- Sean's daily cricket playing time in minutes -/
def sean_daily_minutes : ℕ := 50

/-- Number of days Sean played cricket -/
def sean_days : ℕ := 14

/-- Total minutes Sean and Indira played cricket together -/
def total_minutes : ℕ := 1512

/-- Calculates the total minutes Sean played cricket -/
def sean_total_minutes : ℕ := sean_daily_minutes * sean_days

/-- Theorem: Indira played cricket for 812 minutes -/
theorem indira_cricket_minutes :
  total_minutes - sean_total_minutes = 812 := by
  sorry

end NUMINAMATH_CALUDE_indira_cricket_minutes_l1061_106119


namespace NUMINAMATH_CALUDE_perpendicular_iff_a_eq_pm_one_l1061_106152

def line1 (a x y : ℝ) : Prop := a * x + y + 2 = 0
def line2 (a x y : ℝ) : Prop := a * x - y + 4 = 0

def perpendicular (a : ℝ) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), line1 a x1 y1 → line2 a x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) *
    ((x2 - x1) * (a * (x2 - x1)) + (y2 - y1) * (-(y2 - y1))) = 0

theorem perpendicular_iff_a_eq_pm_one :
  ∀ a : ℝ, perpendicular a ↔ (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_iff_a_eq_pm_one_l1061_106152


namespace NUMINAMATH_CALUDE_second_investment_value_l1061_106161

theorem second_investment_value (x : ℝ) : 
  (0.07 * 500 + 0.15 * x = 0.13 * (500 + x)) → x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_second_investment_value_l1061_106161


namespace NUMINAMATH_CALUDE_incorrect_y_value_l1061_106174

/-- Represents a quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Represents a sequence of 7 x values with equal differences -/
structure XSequence where
  x : Fin 7 → ℝ
  increasing : ∀ i j, i < j → x i < x j
  equal_diff : ∀ i : Fin 6, x (i + 1) - x i = x 1 - x 0

/-- The given y values -/
def y_values : Fin 7 → ℝ := ![51, 107, 185, 285, 407, 549, 717]

/-- The theorem to prove -/
theorem incorrect_y_value (f : QuadraticFunction) (xs : XSequence) :
  (∀ i : Fin 7, i.val ≠ 5 → y_values i = f.a * (xs.x i)^2 + f.b * (xs.x i) + f.c) →
  y_values 5 ≠ f.a * (xs.x 5)^2 + f.b * (xs.x 5) + f.c ∧
  571 = f.a * (xs.x 5)^2 + f.b * (xs.x 5) + f.c := by
  sorry

end NUMINAMATH_CALUDE_incorrect_y_value_l1061_106174


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1061_106146

open Function Real

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x - f y) = 1 - x - y) →
  (∀ y : ℝ, f y = 1/2 - y) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1061_106146


namespace NUMINAMATH_CALUDE_expression_evaluation_l1061_106126

theorem expression_evaluation :
  let x : ℤ := -1
  (x + 2) * (x - 2) - (x - 1)^2 = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1061_106126


namespace NUMINAMATH_CALUDE_jeffrey_farm_chickens_l1061_106112

/-- Calculates the total number of chickens on Jeffrey's farm -/
def total_chickens (num_hens : ℕ) (hen_to_rooster_ratio : ℕ) (chicks_per_hen : ℕ) : ℕ :=
  let num_roosters := num_hens / hen_to_rooster_ratio
  let num_chicks := num_hens * chicks_per_hen
  num_hens + num_roosters + num_chicks

/-- Proves that the total number of chickens on Jeffrey's farm is 76 -/
theorem jeffrey_farm_chickens :
  total_chickens 12 3 5 = 76 := by
  sorry

end NUMINAMATH_CALUDE_jeffrey_farm_chickens_l1061_106112


namespace NUMINAMATH_CALUDE_log_equation_holds_l1061_106108

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_holds_l1061_106108


namespace NUMINAMATH_CALUDE_simplify_expression_l1061_106124

theorem simplify_expression :
  Real.sqrt 5 * 5^(1/2) + 18 / 3 * 4 - 8^(3/2) + 10 - 3^2 = 30 - 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1061_106124


namespace NUMINAMATH_CALUDE_extreme_value_in_interval_l1061_106168

/-- The function f(x) = x ln x + (1/2)x² - 3x has an extreme value in the interval (3/2, 2) -/
theorem extreme_value_in_interval :
  ∃ x : ℝ, (3/2 < x ∧ x < 2) ∧
    ∀ y : ℝ, (3/2 < y ∧ y < 2) →
      (x * Real.log x + (1/2) * x^2 - 3*x ≤ y * Real.log y + (1/2) * y^2 - 3*y ∨
       x * Real.log x + (1/2) * x^2 - 3*x ≥ y * Real.log y + (1/2) * y^2 - 3*y) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_in_interval_l1061_106168


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l1061_106167

theorem parametric_to_ordinary_equation :
  ∀ (x y t : ℝ), x = t + 1 ∧ y = 3 - t^2 → y = -x^2 + 2*x + 2 := by
sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l1061_106167


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l1061_106195

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l1061_106195


namespace NUMINAMATH_CALUDE_integral_equals_three_l1061_106115

-- Define the integrand
def f (x : ℝ) : ℝ := 2 - |1 - x|

-- State the theorem
theorem integral_equals_three : ∫ x in (0)..(2), f x = 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_three_l1061_106115


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_and_intersection_l1061_106179

theorem quadratic_equation_roots_and_intersection :
  ∀ a : ℚ,
  (∃ x : ℚ, x^2 + a*x + a - 2 = 0) →
  (1^2 + a*1 + a - 2 = 0) →
  (a = 1/2) ∧
  (∃ x : ℚ, x ≠ 1 ∧ x^2 + a*x + a - 2 = 0) ∧
  (∃ x y : ℚ, x ≠ y ∧ x^2 + a*x + a - 2 = 0 ∧ y^2 + a*y + a - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_and_intersection_l1061_106179


namespace NUMINAMATH_CALUDE_f_always_negative_iff_m_in_range_l1061_106144

/-- The function f(x) defined as mx^2 - mx - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

/-- Theorem stating that f(x) < 0 for all real x if and only if m is in the interval (-4, 0] -/
theorem f_always_negative_iff_m_in_range :
  (∀ x : ℝ, f m x < 0) ↔ m ∈ Set.Ioc (-4) 0 := by sorry

end NUMINAMATH_CALUDE_f_always_negative_iff_m_in_range_l1061_106144


namespace NUMINAMATH_CALUDE_remainder_1493827_div_4_l1061_106184

theorem remainder_1493827_div_4 : 1493827 % 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1493827_div_4_l1061_106184


namespace NUMINAMATH_CALUDE_tan_difference_l1061_106183

theorem tan_difference (α β : Real) (h1 : Real.tan α = 2) (h2 : Real.tan β = -3) :
  Real.tan (α - β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_l1061_106183


namespace NUMINAMATH_CALUDE_modulus_of_specific_complex_l1061_106107

open Complex

theorem modulus_of_specific_complex : ‖(1 - I) / I‖ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_specific_complex_l1061_106107


namespace NUMINAMATH_CALUDE_work_time_difference_l1061_106121

def monday_minutes : ℕ := 450
def tuesday_minutes : ℕ := monday_minutes / 2
def wednesday_minutes : ℕ := 300

theorem work_time_difference :
  wednesday_minutes - tuesday_minutes = 75 := by
  sorry

end NUMINAMATH_CALUDE_work_time_difference_l1061_106121


namespace NUMINAMATH_CALUDE_difference_of_differences_l1061_106125

def arithmetic_sequence (start : ℕ) (diff : ℕ) (length : ℕ) : List ℕ :=
  List.range length |>.map (fun n => start + n * diff)

def common_terms (seq1 seq2 : List ℕ) : List ℕ :=
  seq1.filter (fun x => seq2.contains x)

theorem difference_of_differences
  (start end_val : ℕ)
  (common_count : ℕ)
  (ratio_a ratio_b : ℕ)
  (ha : ratio_a > 0)
  (hb : ratio_b > 0)
  (hcommon : common_count > 0)
  (hend : end_val > start) :
  ∃ (len_a len_b : ℕ),
    let diff_a := ratio_a * (end_val - start) / ((len_a - 1) * ratio_a)
    let diff_b := ratio_b * (end_val - start) / ((len_b - 1) * ratio_b)
    let seq_a := arithmetic_sequence start diff_a len_a
    let seq_b := arithmetic_sequence start diff_b len_b
    (seq_a.length > 0) ∧
    (seq_b.length > 0) ∧
    (seq_a.getLast? = some end_val) ∧
    (seq_b.getLast? = some end_val) ∧
    (common_terms seq_a seq_b).length = common_count ∧
    diff_a - diff_b = 12 :=
by
  sorry

#check difference_of_differences

end NUMINAMATH_CALUDE_difference_of_differences_l1061_106125


namespace NUMINAMATH_CALUDE_volume_Q3_is_7_l1061_106159

/-- Represents the volume of the i-th polyhedron in the sequence -/
def volume (i : ℕ) : ℚ :=
  match i with
  | 0 => 1
  | n + 1 => volume n + 6 * (1 / 3) * (1 / 4^n)

/-- The theorem stating that the volume of Q₃ is 7 -/
theorem volume_Q3_is_7 : volume 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_volume_Q3_is_7_l1061_106159


namespace NUMINAMATH_CALUDE_solution_for_m_eq_one_solution_satisfies_equation_l1061_106192

-- Define the system of equations
def system (x y m : ℝ) : Prop :=
  2 * x + y = 4 - m ∧ x - 2 * y = 3 * m

-- Statement 1: When m = 1, the solution is x = 9/5 and y = -3/5
theorem solution_for_m_eq_one :
  system (9/5) (-3/5) 1 := by sorry

-- Statement 2: For any m, the solution satisfies 3x - y = 4 + 2m
theorem solution_satisfies_equation (m : ℝ) (x y : ℝ) :
  system x y m → 3 * x - y = 4 + 2 * m := by sorry

end NUMINAMATH_CALUDE_solution_for_m_eq_one_solution_satisfies_equation_l1061_106192


namespace NUMINAMATH_CALUDE_range_g_eq_range_f_l1061_106133

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }

-- Define the range of f
def range_f : Set ℝ := { y | ∃ x ∈ domain_f, f x = y }

-- State that the range of f is [a, b]
axiom range_f_is_ab : ∃ a b : ℝ, range_f = { y | a ≤ y ∧ y ≤ b }

-- Define the function g(x) = f(x + 4)
def g (x : ℝ) : ℝ := f (x + 4)

-- Define the range of g
def range_g : Set ℝ := { y | ∃ x : ℝ, g x = y }

-- Theorem: The range of g is equal to the range of f
theorem range_g_eq_range_f : range_g = range_f := by sorry

end NUMINAMATH_CALUDE_range_g_eq_range_f_l1061_106133


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_value_l1061_106120

/-- A hyperbola with equation -y²/a² + x²/b² = 1 and eccentricity e -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ

/-- A parabola with equation y² = 16x -/
structure Parabola where

/-- The right focus of a hyperbola coincides with the focus of a parabola y² = 16x -/
def right_focus_coincides (h : Hyperbola) (p : Parabola) : Prop :=
  h.e * h.b = 4

theorem hyperbola_eccentricity_value (h : Hyperbola) (p : Parabola) 
  (h_eq : -h.a^2 + h.b^2 = h.a^2 * h.b^2) 
  (coincide : right_focus_coincides h p) : 
  h.e = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_value_l1061_106120


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1061_106110

def complex_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0

theorem point_in_second_quadrant :
  let z : ℂ := 2 * Complex.I / (2 - Complex.I)
  second_quadrant (complex_point z) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1061_106110


namespace NUMINAMATH_CALUDE_trig_simplification_l1061_106109

theorem trig_simplification (x : ℝ) : 
  (1 + Real.sin (3 * x) - Real.cos (3 * x)) / (1 + Real.sin (3 * x) + Real.cos (3 * x)) = 
  (1 + 3 * (Real.sin x + Real.cos x) - 4 * (Real.sin x ^ 3 + Real.cos x ^ 3)) / 
  (1 + 3 * (Real.sin x - Real.cos x) - 4 * (Real.sin x ^ 3 - Real.cos x ^ 3)) := by
sorry

end NUMINAMATH_CALUDE_trig_simplification_l1061_106109


namespace NUMINAMATH_CALUDE_certain_number_value_l1061_106141

/-- Given two sets of numbers with known means, prove the value of an unknown number in the first set. -/
theorem certain_number_value (x y : ℝ) : 
  (28 + x + y + 78 + 104) / 5 = 62 →
  (48 + 62 + 98 + 124 + x) / 5 = 78 →
  y = 42 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l1061_106141


namespace NUMINAMATH_CALUDE_double_markup_percentage_l1061_106169

theorem double_markup_percentage (original_price : ℝ) (markup_percentage : ℝ) : 
  markup_percentage = 40 →
  let first_markup := original_price * (1 + markup_percentage / 100)
  let second_markup := first_markup * (1 + markup_percentage / 100)
  (second_markup - original_price) / original_price * 100 = 96 := by
sorry

end NUMINAMATH_CALUDE_double_markup_percentage_l1061_106169


namespace NUMINAMATH_CALUDE_product_positive_l1061_106186

theorem product_positive (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x^2 - x > y^2) (h2 : y^2 - y > x^2) : x * y > 0 := by
  sorry

end NUMINAMATH_CALUDE_product_positive_l1061_106186


namespace NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_16385_l1061_106185

/-- The number we're analyzing -/
def n : ℕ := 16385

/-- Function to get the greatest prime divisor of a natural number -/
def greatest_prime_divisor (m : ℕ) : ℕ :=
  sorry

/-- Function to sum the digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the sum of digits of the greatest prime divisor of 16385 is 19 -/
theorem sum_digits_greatest_prime_divisor_16385 :
  sum_of_digits (greatest_prime_divisor n) = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_16385_l1061_106185


namespace NUMINAMATH_CALUDE_distance_rowed_is_90km_l1061_106129

/-- Calculates the distance rowed downstream given the rowing speed in still water,
    the stream speed, and the time spent rowing. -/
def distance_rowed_downstream (rowing_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (rowing_speed + stream_speed) * time

/-- Theorem stating that given the specific conditions of the problem,
    the distance rowed downstream is 90 km. -/
theorem distance_rowed_is_90km
  (rowing_speed : ℝ)
  (stream_speed : ℝ)
  (time : ℝ)
  (h1 : rowing_speed = 10)
  (h2 : stream_speed = 8)
  (h3 : time = 5) :
  distance_rowed_downstream rowing_speed stream_speed time = 90 := by
  sorry

#check distance_rowed_is_90km

end NUMINAMATH_CALUDE_distance_rowed_is_90km_l1061_106129


namespace NUMINAMATH_CALUDE_min_trucks_for_given_problem_l1061_106113

/-- Represents the problem of transporting crates with trucks -/
structure CrateTransportProblem where
  totalWeight : ℝ
  maxCrateWeight : ℝ
  truckCapacity : ℝ

/-- Calculates the minimum number of trucks required -/
def minTrucksRequired (problem : CrateTransportProblem) : ℕ :=
  sorry

/-- Theorem stating the minimum number of trucks required for the given problem -/
theorem min_trucks_for_given_problem :
  let problem : CrateTransportProblem := {
    totalWeight := 10,
    maxCrateWeight := 1,
    truckCapacity := 3
  }
  minTrucksRequired problem = 5 := by sorry

end NUMINAMATH_CALUDE_min_trucks_for_given_problem_l1061_106113


namespace NUMINAMATH_CALUDE_shirt_sale_discount_l1061_106114

/-- Proves that applying a 20% discount to a price that is 80% of the original
    results in a final price that is 64% of the original. -/
theorem shirt_sale_discount (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let first_sale_price := 0.8 * original_price
  let final_price := 0.8 * first_sale_price
  final_price / original_price = 0.64 := by sorry

end NUMINAMATH_CALUDE_shirt_sale_discount_l1061_106114


namespace NUMINAMATH_CALUDE_largest_unorderable_correct_l1061_106147

/-- The largest number of dumplings that cannot be ordered -/
def largest_unorderable : ℕ := 43

/-- The set of possible portion sizes for dumplings -/
def portion_sizes : Finset ℕ := {6, 9, 20}

/-- Predicate to check if a number can be expressed as a combination of portion sizes -/
def is_orderable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 6 * a + 9 * b + 20 * c

theorem largest_unorderable_correct :
  (∀ n > largest_unorderable, is_orderable n) ∧
  ¬(is_orderable largest_unorderable) ∧
  (∀ m < largest_unorderable, ∃ n > m, ¬(is_orderable n)) :=
sorry

end NUMINAMATH_CALUDE_largest_unorderable_correct_l1061_106147


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l1061_106196

/-- Given a line y = (1/2)x and a line of symmetry x = 1, 
    the symmetric line has the equation x + 2y - 2 = 0 -/
theorem symmetric_line_equation : 
  ∀ (x y : ℝ), 
  (y = (1/2) * x) →  -- Original line
  (∃ (x' y' : ℝ), 
    (x' = 1) ∧  -- Line of symmetry
    (y' = y) ∧ 
    (x - 1 = 1 - x')) →  -- Symmetry condition
  (x + 2*y - 2 = 0)  -- Equation of symmetric line
:= by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l1061_106196


namespace NUMINAMATH_CALUDE_equation_solution_l1061_106191

theorem equation_solution : 
  {x : ℝ | x - 12 ≥ 0 ∧ 
    (8 / (Real.sqrt (x - 12) - 10) + 
     2 / (Real.sqrt (x - 12) - 5) + 
     10 / (Real.sqrt (x - 12) + 5) + 
     16 / (Real.sqrt (x - 12) + 10) = 0)} = 
  {208/9, 62} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1061_106191


namespace NUMINAMATH_CALUDE_number_of_blue_marbles_l1061_106138

/-- Given the number of yellow, red, and blue marbles, prove that the number of blue marbles is 108 -/
theorem number_of_blue_marbles
  (yellow red blue : ℕ)
  (h1 : blue = 3 * red)
  (h2 : red = yellow + 15)
  (h3 : yellow + red + blue = 165) :
  blue = 108 := by
  sorry

end NUMINAMATH_CALUDE_number_of_blue_marbles_l1061_106138


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_zero_l1061_106105

/-- The function f(x) = (x+a)e^x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.exp x

/-- The derivative of f(x) --/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 
  (x + a + 1) * Real.exp x

theorem tangent_perpendicular_implies_a_zero (a : ℝ) : 
  (f_derivative a 0 = 1) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_zero_l1061_106105


namespace NUMINAMATH_CALUDE_sunset_time_calculation_l1061_106142

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Represents duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

/-- Adds a duration to a time -/
def addDuration (t : Time) (d : Duration) : Time :=
  sorry

theorem sunset_time_calculation (sunrise : Time) (daylight : Duration) : 
  sunrise.hours = 7 → 
  sunrise.minutes = 3 → 
  daylight.hours = 12 → 
  daylight.minutes = 36 → 
  (addDuration sunrise daylight).hours = 19 ∧ 
  (addDuration sunrise daylight).minutes = 39 := by
  sorry

#check sunset_time_calculation

end NUMINAMATH_CALUDE_sunset_time_calculation_l1061_106142


namespace NUMINAMATH_CALUDE_exactly_three_heads_probability_l1061_106151

/-- The probability of getting exactly k successes in n trials of a binomial experiment -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  ↑(Nat.choose n k) * p^k * (1 - p)^(n - k)

/-- The number of coin flips -/
def num_flips : ℕ := 8

/-- The number of heads we're interested in -/
def num_heads : ℕ := 3

/-- The probability of getting heads on a single flip -/
def prob_heads : ℚ := 1/3

theorem exactly_three_heads_probability :
  binomial_probability num_flips num_heads prob_heads = 1792/6561 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_heads_probability_l1061_106151


namespace NUMINAMATH_CALUDE_peanuts_remaining_l1061_106145

def initial_peanuts : ℕ := 148
def bonita_eaten : ℕ := 29

theorem peanuts_remaining : 
  initial_peanuts - (initial_peanuts / 4) - bonita_eaten = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_peanuts_remaining_l1061_106145


namespace NUMINAMATH_CALUDE_digit_equation_solution_l1061_106171

theorem digit_equation_solution :
  ∃! (A B C D : ℕ),
    (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) ∧
    ((A + B * Real.sqrt 3) ^ 2 = (10 * C + D) + (10 * B + C) * Real.sqrt 3) ∧
    ((A + C * Real.sqrt 3) ^ 2 = (10 * D + C) + (10 * C + D) * Real.sqrt 3) ∧
    A = 6 ∧ B = 2 ∧ C = 4 ∧ D = 8 :=
by sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l1061_106171


namespace NUMINAMATH_CALUDE_agate_precious_stones_l1061_106181

theorem agate_precious_stones (agate olivine diamond : ℕ) : 
  olivine = agate + 5 →
  diamond = olivine + 11 →
  agate + olivine + diamond = 111 →
  agate = 30 := by
sorry

end NUMINAMATH_CALUDE_agate_precious_stones_l1061_106181


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1061_106104

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1061_106104


namespace NUMINAMATH_CALUDE_mirka_has_more_pears_l1061_106111

-- Define the number of pears in the bowl at the start
def initial_pears : ℕ := 14

-- Define Ivan's strategy
def ivan_takes : ℕ := 2

-- Define Mirka's strategy
def mirka_takes (remaining : ℕ) : ℕ := remaining / 2

-- Define the sequence of pear-taking
def pear_sequence (pears : ℕ) : ℕ × ℕ :=
  let after_ivan1 := pears - ivan_takes
  let after_mirka1 := after_ivan1 - mirka_takes after_ivan1
  let after_ivan2 := after_mirka1 - ivan_takes
  let after_mirka2 := after_ivan2 - mirka_takes after_ivan2
  let after_ivan3 := after_mirka2 - ivan_takes
  (3 * ivan_takes, mirka_takes after_ivan1 + mirka_takes after_ivan2)

theorem mirka_has_more_pears :
  let (ivan_total, mirka_total) := pear_sequence initial_pears
  mirka_total = ivan_total + 2 := by sorry

end NUMINAMATH_CALUDE_mirka_has_more_pears_l1061_106111


namespace NUMINAMATH_CALUDE_bc_length_l1061_106118

/-- A right triangle with specific properties -/
structure RightTriangle where
  -- The lengths of the sides
  ab : ℝ
  ac : ℝ
  bc : ℝ
  -- The median from A to BC
  median : ℝ
  -- Conditions
  ab_eq : ab = 3
  ac_eq : ac = 4
  median_eq_bc : median = bc
  pythagorean : bc^2 = ab^2 + ac^2

/-- The length of BC in the specific right triangle is 5 -/
theorem bc_length (t : RightTriangle) : t.bc = 5 := by
  sorry

end NUMINAMATH_CALUDE_bc_length_l1061_106118


namespace NUMINAMATH_CALUDE_dice_sum_pigeonhole_l1061_106117

/-- A type representing a fair six-sided die -/
def Die := Fin 6

/-- The sum of four dice rolls -/
def DiceSum := Nat

/-- The minimum number of throws required to guarantee a repeated sum -/
def MinThrows : Nat := 22

/-- The number of possible distinct sums when rolling four dice -/
def DistinctSums : Nat := 21

theorem dice_sum_pigeonhole :
  MinThrows = DistinctSums + 1 ∧
  ∀ n : Nat, n < MinThrows → 
    ∃ f : Fin n → DiceSum,
      ∀ i j : Fin n, i ≠ j → f i ≠ f j :=
by sorry

end NUMINAMATH_CALUDE_dice_sum_pigeonhole_l1061_106117


namespace NUMINAMATH_CALUDE_reception_hall_tables_l1061_106137

/-- The cost of a linen tablecloth -/
def tablecloth_cost : ℕ := 25

/-- The cost of a single place setting -/
def place_setting_cost : ℕ := 10

/-- The number of place settings per table -/
def place_settings_per_table : ℕ := 4

/-- The cost of a single rose -/
def rose_cost : ℕ := 5

/-- The number of roses per centerpiece -/
def roses_per_centerpiece : ℕ := 10

/-- The cost of a single lily -/
def lily_cost : ℕ := 4

/-- The number of lilies per centerpiece -/
def lilies_per_centerpiece : ℕ := 15

/-- The total decoration budget -/
def total_budget : ℕ := 3500

/-- The cost of decorations for a single table -/
def cost_per_table : ℕ :=
  tablecloth_cost +
  place_setting_cost * place_settings_per_table +
  rose_cost * roses_per_centerpiece +
  lily_cost * lilies_per_centerpiece

/-- The number of tables at the reception hall -/
def number_of_tables : ℕ := total_budget / cost_per_table

theorem reception_hall_tables :
  number_of_tables = 20 :=
sorry

end NUMINAMATH_CALUDE_reception_hall_tables_l1061_106137


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l1061_106153

theorem triangle_max_perimeter :
  ∀ x : ℕ,
  x > 0 →
  x < 17 →
  x + 2*x > 17 →
  x + 17 > 2*x →
  2*x + 17 > x →
  (∀ y : ℕ, y > 0 → y < 17 → y + 2*y > 17 → y + 17 > 2*y → 2*y + 17 > y →
    x + 2*x + 17 ≥ y + 2*y + 17) →
  x + 2*x + 17 = 65 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l1061_106153


namespace NUMINAMATH_CALUDE_stream_speed_l1061_106177

/-- The speed of a stream given a swimmer's still water speed and relative upstream/downstream times -/
theorem stream_speed (still_speed : ℝ) (upstream_time_factor : ℝ) : 
  still_speed = 4.5 ∧ upstream_time_factor = 2 → 
  ∃ stream_speed : ℝ, stream_speed = 1.5 ∧
    upstream_time_factor * (still_speed + stream_speed) = still_speed - stream_speed :=
by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1061_106177


namespace NUMINAMATH_CALUDE_undefined_fraction_l1061_106198

theorem undefined_fraction (b : ℝ) : 
  ¬ (∃ x : ℝ, x = (b + 3) / (b^2 - 9)) ↔ b = -3 ∨ b = 3 := by
sorry

end NUMINAMATH_CALUDE_undefined_fraction_l1061_106198


namespace NUMINAMATH_CALUDE_production_cost_at_most_80_l1061_106197

/-- Represents the monthly production and financial data for an electronic component manufacturer -/
structure ComponentManufacturer where
  shippingCost : ℝ
  fixedCosts : ℝ
  monthlyProduction : ℕ
  lowestSellingPrice : ℝ

/-- Theorem stating that the production cost per component is at most $80 -/
theorem production_cost_at_most_80 (m : ComponentManufacturer) 
  (h1 : m.shippingCost = 5)
  (h2 : m.fixedCosts = 16500)
  (h3 : m.monthlyProduction = 150)
  (h4 : m.lowestSellingPrice = 195) :
  ∃ (productionCost : ℝ), productionCost ≤ 80 ∧ 
    (m.monthlyProduction : ℝ) * productionCost + 
    (m.monthlyProduction : ℝ) * m.shippingCost + 
    m.fixedCosts ≤ (m.monthlyProduction : ℝ) * m.lowestSellingPrice :=
by sorry

end NUMINAMATH_CALUDE_production_cost_at_most_80_l1061_106197


namespace NUMINAMATH_CALUDE_cube_painting_probability_l1061_106150

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The number of color choices for each face -/
def num_colors : ℕ := 2

/-- The total number of ways to paint a single cube -/
def total_paint_ways : ℕ := num_colors ^ num_faces

/-- The total number of ways to paint two cubes -/
def total_two_cube_ways : ℕ := total_paint_ways ^ 2

/-- The number of ways two cubes can be painted to look identical after rotation -/
def identical_after_rotation : ℕ := 363

/-- The probability that two independently painted cubes are identical after rotation -/
def prob_identical_after_rotation : ℚ := identical_after_rotation / total_two_cube_ways

theorem cube_painting_probability :
  prob_identical_after_rotation = 363 / 4096 := by sorry

end NUMINAMATH_CALUDE_cube_painting_probability_l1061_106150


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l1061_106101

/-- Given a hyperbola and a parabola with specific properties, prove that the parameter p of the parabola equals 4. -/
theorem parabola_hyperbola_intersection (a b p k : ℝ) : 
  a > 0 → b > 0 → p > 0 →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  a = 2*Real.sqrt 2 →  -- Real axis length
  b = p/2 →  -- Imaginary axis endpoint coincides with parabola focus
  (∀ x y, x^2 = 2*p*y) →  -- Parabola equation
  (∃ x y, y = k*x - 1 ∧ x^2 = 2*p*y) →  -- Line is tangent to parabola
  k = p/(4*Real.sqrt 2) →  -- Line is parallel to hyperbola asymptote
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l1061_106101

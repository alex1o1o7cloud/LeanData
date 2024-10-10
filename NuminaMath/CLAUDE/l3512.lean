import Mathlib

namespace smallest_M_inequality_l3512_351255

theorem smallest_M_inequality (a b c : ℝ) : ∃ (M : ℝ), 
  (∀ (x y z : ℝ), |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ M*(x^2 + y^2 + z^2)^2) ∧ 
  (M = (9 * Real.sqrt 2) / 64) ∧
  (∀ (N : ℝ), (∀ (x y z : ℝ), |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ N*(x^2 + y^2 + z^2)^2) → M ≤ N) :=
by sorry

end smallest_M_inequality_l3512_351255


namespace ribbons_given_in_afternoon_l3512_351248

/-- Given the initial number of ribbons, the number given away in the morning,
    and the number left at the end, prove that the number of ribbons given away
    in the afternoon is 16. -/
theorem ribbons_given_in_afternoon
  (initial : ℕ)
  (morning : ℕ)
  (left : ℕ)
  (h1 : initial = 38)
  (h2 : morning = 14)
  (h3 : left = 8) :
  initial - morning - left = 16 := by
  sorry

end ribbons_given_in_afternoon_l3512_351248


namespace prob_not_adjacent_l3512_351218

/-- The number of desks in the classroom -/
def num_desks : ℕ := 9

/-- The number of students choosing desks -/
def num_students : ℕ := 2

/-- The number of ways two adjacent desks can be chosen -/
def adjacent_choices : ℕ := num_desks - 1

/-- The probability that two students do not sit next to each other when randomly choosing from a row of desks -/
theorem prob_not_adjacent (n : ℕ) (k : ℕ) (h : n ≥ 2 ∧ k = 2) : 
  (1 : ℚ) - (adjacent_choices : ℚ) / (n.choose k) = 7/9 :=
sorry

end prob_not_adjacent_l3512_351218


namespace shared_triangle_angle_measure_l3512_351254

-- Define the angle measures
def angle1 : Real := 58
def angle2 : Real := 35
def angle3 : Real := 42

-- Define the theorem
theorem shared_triangle_angle_measure :
  ∃ (angle4 angle5 angle6 : Real),
    -- The sum of angles in the first triangle is 180°
    angle1 + angle2 + angle5 = 180 ∧
    -- The sum of angles in the second triangle is 180°
    angle3 + angle5 + angle6 = 180 ∧
    -- The sum of angles in the third triangle (with the unknown angle) is 180°
    angle4 + angle5 + angle6 = 180 ∧
    -- The measure of the unknown angle (angle4) is 135°
    angle4 = 135 := by
  sorry

end shared_triangle_angle_measure_l3512_351254


namespace triangle_side_length_l3512_351284

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Define the triangle
  A + B + C = Real.pi →  -- Sum of angles in a triangle is π radians
  A = Real.pi / 6 →      -- 30° in radians
  C = 7 * Real.pi / 12 → -- 105° in radians
  b = 8 →                -- Given side length
  -- Law of Sines
  b / Real.sin B = a / Real.sin A →
  -- Conclusion
  a = 4 * Real.sqrt 2 := by
sorry

end triangle_side_length_l3512_351284


namespace simplify_expression_l3512_351272

theorem simplify_expression : (3 + 3 + 5) / 2 - 1 / 2 = 5 := by
  sorry

end simplify_expression_l3512_351272


namespace sugar_price_increase_l3512_351295

theorem sugar_price_increase (original_price : ℝ) (consumption_reduction : ℝ) (new_price : ℝ) : 
  original_price = 6 →
  consumption_reduction = 19.999999999999996 →
  (1 - consumption_reduction / 100) * new_price = original_price →
  new_price = 7.5 := by
sorry

end sugar_price_increase_l3512_351295


namespace quadratic_symmetry_inequality_l3512_351282

/-- A quadratic function with axis of symmetry at x = 1 satisfies c > 2b -/
theorem quadratic_symmetry_inequality (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (2 - x)^2 + b * (2 - x) + c) →
  c > 2 * b := by
  sorry

end quadratic_symmetry_inequality_l3512_351282


namespace rectangle_area_l3512_351276

/-- A rectangle with diagonal d and length three times its width has area 3d²/10 -/
theorem rectangle_area (d : ℝ) (w : ℝ) (h : w > 0) : 
  w^2 + (3*w)^2 = d^2 → w * (3*w) = (3 * d^2) / 10 := by
sorry

end rectangle_area_l3512_351276


namespace isosceles_right_triangle_prism_side_length_l3512_351285

theorem isosceles_right_triangle_prism_side_length 
  (XY XZ : ℝ) (height volume : ℝ) : 
  XY = XZ →  -- Base triangle is isosceles
  height = 6 →  -- Height of the prism
  volume = 27 →  -- Volume of the prism
  volume = (1/2 * XY * XY) * height →  -- Volume formula for triangular prism
  XY = 3 ∧ XZ = 3  -- Conclusion: side lengths are 3
  :=
by sorry

end isosceles_right_triangle_prism_side_length_l3512_351285


namespace matching_shoes_probability_l3512_351253

theorem matching_shoes_probability (n : ℕ) (h : n = 100) :
  let total_shoes := 2 * n
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := n
  (matching_pairs : ℚ) / total_combinations = 1 / 199 := by
  sorry

end matching_shoes_probability_l3512_351253


namespace rope_segment_relation_l3512_351275

theorem rope_segment_relation (x : ℝ) : x > 0 ∧ x ≤ 2 →
  (x^2 = 2*(2 - x) ↔ x^2 = (2 - x) * 2) := by
  sorry

end rope_segment_relation_l3512_351275


namespace absolute_value_and_exponents_calculation_l3512_351261

theorem absolute_value_and_exponents_calculation : 
  |(-5 : ℝ)| + (1/3)⁻¹ - (π - 2)^0 = 7 := by sorry

end absolute_value_and_exponents_calculation_l3512_351261


namespace tournament_participants_perfect_square_l3512_351252

-- Define the tournament structure
structure ChessTournament where
  masters : ℕ
  grandmasters : ℕ

-- Define the property that each participant scored half their points against masters
def halfPointsAgainstMasters (t : ChessTournament) : Prop :=
  let totalParticipants := t.masters + t.grandmasters
  (t.masters * (t.masters - 1) + t.grandmasters * (t.grandmasters - 1)) / 2 = t.masters * t.grandmasters

-- Theorem statement
theorem tournament_participants_perfect_square (t : ChessTournament) 
  (h : halfPointsAgainstMasters t) : 
  ∃ n : ℕ, (t.masters + t.grandmasters) = n^2 :=
sorry

end tournament_participants_perfect_square_l3512_351252


namespace quadratic_roots_product_l3512_351263

theorem quadratic_roots_product (c d : ℝ) : 
  (3 * c ^ 2 + 9 * c - 21 = 0) → 
  (3 * d ^ 2 + 9 * d - 21 = 0) → 
  (3 * c - 4) * (6 * d - 8) = -22 := by
sorry

end quadratic_roots_product_l3512_351263


namespace max_value_of_function_l3512_351238

theorem max_value_of_function (x : ℝ) : 
  (3 * Real.sin x + 2 * Real.sqrt (2 + 2 * Real.cos (2 * x))) ≤ 5 := by
  sorry

end max_value_of_function_l3512_351238


namespace cubic_expansion_equals_cube_problem_solution_l3512_351256

theorem cubic_expansion_equals_cube (n : ℕ) : n^3 + 3*(n^2) + 3*n + 1 = (n + 1)^3 := by sorry

theorem problem_solution : 98^3 + 3*(98^2) + 3*98 + 1 = 970299 := by sorry

end cubic_expansion_equals_cube_problem_solution_l3512_351256


namespace gcd_lcm_sum_8_12_l3512_351225

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l3512_351225


namespace arithmetic_sequence_property_l3512_351224

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 3 + a 9 = 20) :
  4 * a 5 - a 7 = 20 := by
sorry

end arithmetic_sequence_property_l3512_351224


namespace proportional_difference_theorem_l3512_351236

theorem proportional_difference_theorem (x y z k₁ k₂ : ℝ) 
  (h1 : y - z = k₁ * x)
  (h2 : z - x = k₂ * y)
  (h3 : k₁ ≠ k₂)
  (h4 : z = 3 * (x - y))
  (h5 : x ≠ 0)
  (h6 : y ≠ 0) :
  (k₁ + 3) * (k₂ + 3) = 8 := by
sorry

end proportional_difference_theorem_l3512_351236


namespace smallest_value_for_y_between_zero_and_one_l3512_351220

theorem smallest_value_for_y_between_zero_and_one
  (y : ℝ) (h : 0 < y ∧ y < 1) :
  y^3 < y ∧ y^3 < 3*y ∧ y^3 < y^(1/3) ∧ y^3 < 1/y :=
by sorry

end smallest_value_for_y_between_zero_and_one_l3512_351220


namespace whisky_replacement_fraction_l3512_351226

/-- Proves the fraction of whisky replaced given initial and final alcohol percentages -/
theorem whisky_replacement_fraction (initial_percent : ℝ) (replacement_percent : ℝ) (final_percent : ℝ) :
  initial_percent = 0.40 →
  replacement_percent = 0.19 →
  final_percent = 0.24 →
  ∃ (fraction : ℝ), fraction = 0.16 / 0.21 ∧
    initial_percent * (1 - fraction) + replacement_percent * fraction = final_percent :=
by sorry

end whisky_replacement_fraction_l3512_351226


namespace octal_addition_521_146_l3512_351231

/-- Represents an octal number as a list of digits (0-7) in reverse order --/
def OctalNumber := List Nat

/-- Converts an octal number to its decimal representation --/
def octal_to_decimal (n : OctalNumber) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

/-- Adds two octal numbers and returns the result in octal --/
def add_octal (a b : OctalNumber) : OctalNumber :=
  sorry

theorem octal_addition_521_146 :
  let a : OctalNumber := [1, 2, 5]  -- 521₈ in reverse order
  let b : OctalNumber := [6, 4, 1]  -- 146₈ in reverse order
  let result : OctalNumber := [7, 6, 6]  -- 667₈ in reverse order
  add_octal a b = result :=
by sorry

end octal_addition_521_146_l3512_351231


namespace simplify_expression_l3512_351208

theorem simplify_expression (a : ℝ) : 3*a - 5*a + a = -a := by
  sorry

end simplify_expression_l3512_351208


namespace end_at_multiple_of_4_probability_l3512_351289

/-- Represents the possible moves on the spinner -/
inductive SpinnerMove
| Left2 : SpinnerMove
| Right2 : SpinnerMove
| Right1 : SpinnerMove

/-- The probability of a specific move on the spinner -/
def spinnerProbability (move : SpinnerMove) : ℚ :=
  match move with
  | SpinnerMove.Left2 => 1/4
  | SpinnerMove.Right2 => 1/2
  | SpinnerMove.Right1 => 1/4

/-- The set of cards Jeff can pick from -/
def cardSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 12}

/-- Whether a number is a multiple of 4 -/
def isMultipleOf4 (n : ℕ) : Prop := ∃ k, n = 4 * k

/-- The probability of ending at a multiple of 4 -/
def probEndAtMultipleOf4 : ℚ := 1/32

theorem end_at_multiple_of_4_probability : 
  probEndAtMultipleOf4 = 1/32 :=
sorry

end end_at_multiple_of_4_probability_l3512_351289


namespace family_money_sum_l3512_351207

/-- Given Madeline has $48, her brother has half as much as her, and their sister has twice as much as Madeline, the total amount of money all three of them have together is $168. -/
theorem family_money_sum (madeline_money : ℕ) (brother_money : ℕ) (sister_money : ℕ) 
  (h1 : madeline_money = 48)
  (h2 : brother_money = madeline_money / 2)
  (h3 : sister_money = madeline_money * 2) : 
  madeline_money + brother_money + sister_money = 168 := by
  sorry

end family_money_sum_l3512_351207


namespace card_combination_problem_l3512_351247

theorem card_combination_problem : Nat.choose 60 8 = 7580800000 := by
  sorry

end card_combination_problem_l3512_351247


namespace min_degree_g_l3512_351298

/-- Given polynomials f, g, and h satisfying the equation 4f + 5g = h, 
    with deg(f) = 10 and deg(h) = 12, the minimum possible degree of g is 12 -/
theorem min_degree_g (f g h : Polynomial ℝ) 
  (eq : 4 • f + 5 • g = h) 
  (deg_f : Polynomial.degree f = 10)
  (deg_h : Polynomial.degree h = 12) :
  Polynomial.degree g ≥ 12 := by
  sorry

end min_degree_g_l3512_351298


namespace book_pages_from_digits_l3512_351245

theorem book_pages_from_digits (total_digits : ℕ) : total_digits = 792 → ∃ (pages : ℕ), pages = 300 ∧ 
  (pages ≤ 9 → total_digits = pages) ∧
  (9 < pages ∧ pages ≤ 99 → total_digits = 9 + 2 * (pages - 9)) ∧
  (99 < pages → total_digits = 189 + 3 * (pages - 99)) :=
by
  sorry

end book_pages_from_digits_l3512_351245


namespace set_size_comparison_l3512_351242

/-- The size of set A for a given n -/
def size_A (n : ℕ) : ℕ := n^3 + n^5 + n^7 + n^9

/-- The size of set B for a given m -/
def size_B (m : ℕ) : ℕ := m^2 + m^4 + m^6 + m^8

/-- Theorem stating the condition for |B| ≥ |A| when n = 6 -/
theorem set_size_comparison (m : ℕ) :
  size_B m ≥ size_A 6 ↔ m ≥ 8 := by
  sorry

end set_size_comparison_l3512_351242


namespace cube_minus_reciprocal_cube_l3512_351239

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 := by
  sorry

end cube_minus_reciprocal_cube_l3512_351239


namespace speed_conversion_l3512_351243

/-- Converts meters per second to kilometers per hour -/
def mps_to_kmph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

/-- Theorem: A speed of 70.0056 meters per second is equivalent to 252.02016 kilometers per hour -/
theorem speed_conversion : mps_to_kmph 70.0056 = 252.02016 := by
  sorry

end speed_conversion_l3512_351243


namespace unique_triangle_side_l3512_351219

/-- A function that checks if a triangle with sides a, b, and c can exist -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that 3 is the only positive integer value of x for which
    a triangle with sides 5, x + 1, and x^3 can exist -/
theorem unique_triangle_side : ∀ x : ℕ+, 
  (is_valid_triangle 5 (x + 1) (x^3) ↔ x = 3) := by sorry

end unique_triangle_side_l3512_351219


namespace arithmetic_proof_l3512_351273

theorem arithmetic_proof : 4 * (9 - 6)^2 / 2 - 7 = 11 := by
  sorry

end arithmetic_proof_l3512_351273


namespace absolute_value_equation_l3512_351269

theorem absolute_value_equation (x : ℝ) : |4*x - 3| + 2 = 2 → x = 3/4 := by
  sorry

end absolute_value_equation_l3512_351269


namespace odd_digits_base4_345_l3512_351283

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 345₁₀ is 3 -/
theorem odd_digits_base4_345 : countOddDigits (toBase4 345) = 3 := by
  sorry

end odd_digits_base4_345_l3512_351283


namespace isosceles_triangle_perimeter_l3512_351227

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ (a = 4 ∧ b = 2 ∧ c = 4) →
  a + b > c → b + c > a → c + a > b →
  a + b + c = 10 :=
by
  sorry


end isosceles_triangle_perimeter_l3512_351227


namespace power_fraction_simplification_l3512_351234

theorem power_fraction_simplification :
  (3^2014 + 3^2012) / (3^2014 - 3^2012) = 5/4 := by
  sorry

end power_fraction_simplification_l3512_351234


namespace probability_diamond_spade_standard_deck_l3512_351204

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (diamonds : ℕ)
  (spades : ℕ)

/-- The probability of drawing a diamond first and then a spade from a standard deck -/
def probability_diamond_then_spade (d : Deck) : ℚ :=
  (d.diamonds : ℚ) / d.total_cards * d.spades / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a diamond first and then a spade from a standard deck -/
theorem probability_diamond_spade_standard_deck :
  ∃ d : Deck, d.total_cards = 52 ∧ d.diamonds = 13 ∧ d.spades = 13 ∧
  probability_diamond_then_spade d = 13 / 204 := by
  sorry

#check probability_diamond_spade_standard_deck

end probability_diamond_spade_standard_deck_l3512_351204


namespace equality_and_inequality_of_expressions_l3512_351217

variable (a : ℝ)

def f (n : ℕ) (x : ℝ) : ℝ := x ^ n

theorem equality_and_inequality_of_expressions (h : a ≠ 1) :
  (∀ n : ℕ, f n a = a ^ n) →
  ((f 11 (f 13 a)) ^ 14 = f 2002 a) ∧
  (f 11 (f 13 (f 14 a)) = f 2002 a) ∧
  ((f 11 a * f 13 a) ^ 14 ≠ f 2002 a) ∧
  (f 11 a * f 13 a * f 14 a ≠ f 2002 a) := by
  sorry

end equality_and_inequality_of_expressions_l3512_351217


namespace handshake_count_l3512_351229

theorem handshake_count (n : ℕ) (h : n = 8) : 
  (2 * n) * ((2 * n) - 2) / 2 = 112 := by
  sorry

end handshake_count_l3512_351229


namespace cube_volume_from_surface_area_l3512_351200

/-- Given a cube with surface area 54 square centimeters, its volume is 27 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ (side_length : ℝ),
  (6 * side_length^2 = 54) →
  side_length^3 = 27 := by
sorry

end cube_volume_from_surface_area_l3512_351200


namespace quadratic_inequality_problem_l3512_351274

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 + 5 * x - 2

-- Define the solution set of f(x) > 0
def solution_set (a : ℝ) := {x : ℝ | f a x > 0}

-- Define the given solution set
def given_set := {x : ℝ | 1/2 < x ∧ x < 2}

theorem quadratic_inequality_problem (a : ℝ) 
  (h : solution_set a = given_set) :
  (a = -2) ∧ 
  ({x : ℝ | a * x^2 - 5 * x + a^2 - 1 > 0} = {x : ℝ | -3 < x ∧ x < 1/2}) :=
sorry

end quadratic_inequality_problem_l3512_351274


namespace participation_schemes_l3512_351235

/-- The number of people to choose from -/
def total_people : ℕ := 5

/-- The number of people to be selected -/
def selected_people : ℕ := 3

/-- The number of projects -/
def num_projects : ℕ := 3

/-- The number of special people (A and B) -/
def special_people : ℕ := 2

/-- Calculates the number of permutations of r items from n -/
def permutations (n r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

theorem participation_schemes : 
  permutations total_people selected_people - 
  permutations (total_people - special_people) selected_people = 54 := by
sorry

end participation_schemes_l3512_351235


namespace mean_median_difference_l3512_351202

-- Define the frequency distribution of sick days
def sick_days_freq : List (Nat × Nat) := [(0, 4), (1, 2), (2, 5), (3, 2), (4, 1), (5, 1)]

-- Total number of students
def total_students : Nat := 15

-- Function to calculate the median
def median (freq : List (Nat × Nat)) (total : Nat) : Nat :=
  sorry

-- Function to calculate the mean
def mean (freq : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

-- Theorem statement
theorem mean_median_difference :
  mean sick_days_freq total_students = (median sick_days_freq total_students : Rat) - 1/5 :=
sorry

end mean_median_difference_l3512_351202


namespace jones_wardrobe_count_l3512_351249

/-- Represents the clothing items of Mr. Jones -/
structure Wardrobe where
  pants : ℕ
  shirts : ℕ
  ties : ℕ
  socks : ℕ

/-- Calculates the total number of clothing items -/
def total_clothes (w : Wardrobe) : ℕ :=
  w.pants + w.shirts + w.ties + w.socks

/-- Theorem stating the total number of clothes Mr. Jones owns -/
theorem jones_wardrobe_count :
  ∃ (w : Wardrobe),
    w.pants = 40 ∧
    w.shirts = 6 * w.pants ∧
    w.ties = (3 * w.shirts) / 2 ∧
    w.socks = w.ties ∧
    total_clothes w = 1000 := by
  sorry

#check jones_wardrobe_count

end jones_wardrobe_count_l3512_351249


namespace max_sum_under_constraints_l3512_351211

theorem max_sum_under_constraints (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) (h2 : 3 * x + 5 * y ≤ 11) : 
  x + y ≤ 31 / 11 := by
  sorry

end max_sum_under_constraints_l3512_351211


namespace manipulation_function_l3512_351221

theorem manipulation_function (f : ℤ → ℤ) (h : 3 * (f 19 + 5) = 129) :
  ∀ x : ℤ, f x = 2 * x := by
  sorry

end manipulation_function_l3512_351221


namespace units_digit_of_k_squared_plus_two_to_k_l3512_351233

def k : ℕ := 2012^2 + 2^2012

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := k) : (k^2 + 2^k) % 10 = 7 := by
  sorry

end units_digit_of_k_squared_plus_two_to_k_l3512_351233


namespace frequency_distribution_forms_l3512_351212

/-- Represents a frequency distribution table -/
structure FrequencyDistributionTable

/-- Represents a frequency distribution histogram -/
structure FrequencyDistributionHistogram

/-- Represents a set of data -/
structure DataSet

/-- A frequency distribution form for a set of data -/
class FrequencyDistributionForm (α : Type) where
  represents : α → DataSet → Prop

/-- Accuracy property for frequency distribution forms -/
class Accurate (α : Type) where
  is_accurate : α → Prop

/-- Intuitiveness property for frequency distribution forms -/
class Intuitive (α : Type) where
  is_intuitive : α → Prop

instance : FrequencyDistributionForm FrequencyDistributionTable where
  represents := sorry

instance : FrequencyDistributionForm FrequencyDistributionHistogram where
  represents := sorry

instance : Accurate FrequencyDistributionTable where
  is_accurate := sorry

instance : Intuitive FrequencyDistributionHistogram where
  is_intuitive := sorry

/-- Theorem stating that frequency distribution tables and histograms are two forms of frequency distribution for a set of data, with tables being accurate and histograms being intuitive -/
theorem frequency_distribution_forms :
  (∃ (t : FrequencyDistributionTable) (h : FrequencyDistributionHistogram) (d : DataSet),
    FrequencyDistributionForm.represents t d ∧
    FrequencyDistributionForm.represents h d) ∧
  (∀ (t : FrequencyDistributionTable), Accurate.is_accurate t) ∧
  (∀ (h : FrequencyDistributionHistogram), Intuitive.is_intuitive h) :=
sorry

end frequency_distribution_forms_l3512_351212


namespace rectangular_field_area_decrease_l3512_351206

theorem rectangular_field_area_decrease :
  ∀ (L W : ℝ),
  L > 0 → W > 0 →
  let original_area := L * W
  let new_length := L * (1 - 0.4)
  let new_width := W * (1 - 0.4)
  let new_area := new_length * new_width
  (original_area - new_area) / original_area = 0.64 := by
  sorry

end rectangular_field_area_decrease_l3512_351206


namespace final_green_probability_l3512_351296

/-- Represents the total number of amoeba in the dish -/
def total_amoeba : ℕ := 10

/-- Represents the initial number of green amoeba -/
def initial_green : ℕ := 7

/-- Represents the initial number of blue amoeba -/
def initial_blue : ℕ := 3

/-- Theorem stating the probability of the final amoeba being green -/
theorem final_green_probability :
  (initial_green : ℚ) / total_amoeba = 7 / 10 :=
sorry

end final_green_probability_l3512_351296


namespace valid_closed_broken_line_segments_l3512_351271

/-- A closed broken line where each segment intersects exactly once and no three segments share a common point. -/
structure ClosedBrokenLine where
  segments : ℕ
  is_closed : Bool
  each_segment_intersects_once : Bool
  no_three_segments_share_point : Bool

/-- Predicate to check if a ClosedBrokenLine is valid -/
def is_valid_closed_broken_line (line : ClosedBrokenLine) : Prop :=
  line.is_closed ∧ line.each_segment_intersects_once ∧ line.no_three_segments_share_point

/-- Theorem stating that a valid ClosedBrokenLine can have 1996 segments but not 1997 -/
theorem valid_closed_broken_line_segments :
  (∃ (line : ClosedBrokenLine), line.segments = 1996 ∧ is_valid_closed_broken_line line) ∧
  (¬ ∃ (line : ClosedBrokenLine), line.segments = 1997 ∧ is_valid_closed_broken_line line) := by
  sorry

end valid_closed_broken_line_segments_l3512_351271


namespace fibonacci_divisibility_l3512_351297

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem fibonacci_divisibility (k m n s : ℕ) (h : m > 0) (h1 : n > 0) :
  m ∣ fibonacci k → m^n ∣ fibonacci (k * m^(n-1) * s) := by
  sorry

end fibonacci_divisibility_l3512_351297


namespace boat_downstream_speed_l3512_351259

/-- Given a boat's speed in still water and its upstream speed, calculate its downstream speed. -/
theorem boat_downstream_speed
  (still_water_speed : ℝ)
  (upstream_speed : ℝ)
  (h1 : still_water_speed = 7)
  (h2 : upstream_speed = 4) :
  still_water_speed + (still_water_speed - upstream_speed) = 10 :=
by sorry

end boat_downstream_speed_l3512_351259


namespace train_crossing_time_l3512_351251

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 300 →
  train_speed_kmh = 120 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 9 := by
  sorry

#check train_crossing_time

end train_crossing_time_l3512_351251


namespace smallest_multiple_of_45_and_75_not_20_l3512_351230

theorem smallest_multiple_of_45_and_75_not_20 : 
  ∃ (n : ℕ), n > 0 ∧ 45 ∣ n ∧ 75 ∣ n ∧ ¬(20 ∣ n) ∧ 
  ∀ (m : ℕ), m > 0 → 45 ∣ m → 75 ∣ m → ¬(20 ∣ m) → n ≤ m :=
by
  -- The proof would go here
  sorry

end smallest_multiple_of_45_and_75_not_20_l3512_351230


namespace angus_has_55_tokens_l3512_351246

/-- The number of tokens Angus has -/
def angus_tokens (elsa_tokens : ℕ) (token_value : ℕ) (value_difference : ℕ) : ℕ :=
  elsa_tokens - (value_difference / token_value)

/-- Theorem stating that Angus has 55 tokens -/
theorem angus_has_55_tokens (elsa_tokens : ℕ) (token_value : ℕ) (value_difference : ℕ)
  (h1 : elsa_tokens = 60)
  (h2 : token_value = 4)
  (h3 : value_difference = 20) :
  angus_tokens elsa_tokens token_value value_difference = 55 := by
  sorry

end angus_has_55_tokens_l3512_351246


namespace jimmy_payment_jimmy_paid_fifty_l3512_351270

/-- The amount Jimmy paid with, given his purchases and change received. -/
theorem jimmy_payment (pen_price notebook_price folder_price : ℕ)
  (pen_count notebook_count folder_count : ℕ)
  (change : ℕ) : ℕ :=
  let total_cost := pen_price * pen_count + notebook_price * notebook_count + folder_price * folder_count
  total_cost + change

/-- Proof that Jimmy paid $50 given his purchases and change received. -/
theorem jimmy_paid_fifty :
  jimmy_payment 1 3 5 3 4 2 25 = 50 := by
  sorry

end jimmy_payment_jimmy_paid_fifty_l3512_351270


namespace max_b_value_l3512_351244

theorem max_b_value (x b : ℤ) : 
  x^2 + b*x = -21 → 
  b > 0 → 
  (∃ (max_b : ℤ), max_b = 22 ∧ ∀ (b' : ℤ), b' > 0 → (∃ (x' : ℤ), x'^2 + b'*x' = -21) → b' ≤ max_b) :=
by sorry

end max_b_value_l3512_351244


namespace flea_market_spending_l3512_351250

/-- Given that Jayda spent $400 and Aitana spent 2/5 times more than Jayda,
    prove that the total amount they spent together is $960. -/
theorem flea_market_spending (jayda_spent : ℝ) (aitana_ratio : ℝ) : 
  jayda_spent = 400 → 
  aitana_ratio = 2/5 → 
  jayda_spent + (jayda_spent + aitana_ratio * jayda_spent) = 960 := by
sorry

end flea_market_spending_l3512_351250


namespace volleyball_team_starters_l3512_351241

theorem volleyball_team_starters (n m k : ℕ) (h1 : n = 14) (h2 : m = 6) (h3 : k = 3) :
  Nat.choose (n - k) (m - k) = 165 := by
  sorry

end volleyball_team_starters_l3512_351241


namespace exists_three_adjacent_sum_exceeds_17_l3512_351215

-- Define a type for jersey numbers
def JerseyNumber := Fin 10

-- Define a type for the circular arrangement of players
def CircularArrangement := Fin 10 → JerseyNumber

-- Define a function to check if three consecutive numbers sum to more than 17
def SumExceeds17 (arrangement : CircularArrangement) (i : Fin 10) : Prop :=
  (arrangement i).val + (arrangement (i + 1)).val + (arrangement (i + 2)).val > 17

-- Theorem statement
theorem exists_three_adjacent_sum_exceeds_17 (arrangement : CircularArrangement) :
  (∀ i j : Fin 10, i ≠ j → arrangement i ≠ arrangement j) →
  ∃ i : Fin 10, SumExceeds17 arrangement i := by
  sorry

end exists_three_adjacent_sum_exceeds_17_l3512_351215


namespace young_bonnet_ratio_l3512_351280

/-- Mrs. Young's bonnet making problem -/
theorem young_bonnet_ratio :
  let monday_bonnets : ℕ := 10
  let thursday_bonnets : ℕ := monday_bonnets + 5
  let friday_bonnets : ℕ := thursday_bonnets - 5
  let total_orphanages : ℕ := 5
  let bonnets_per_orphanage : ℕ := 11
  let total_bonnets : ℕ := total_orphanages * bonnets_per_orphanage
  let tues_wed_bonnets : ℕ := total_bonnets - (monday_bonnets + thursday_bonnets + friday_bonnets)
  tues_wed_bonnets / monday_bonnets = 2 := by
  sorry

end young_bonnet_ratio_l3512_351280


namespace dog_training_weeks_l3512_351201

/-- The number of weeks of training for a seeing-eye dog -/
def training_weeks : ℕ := 12

/-- The adoption fee for an untrained dog in dollars -/
def adoption_fee : ℕ := 150

/-- The cost of training per week in dollars -/
def training_cost_per_week : ℕ := 250

/-- The total certification cost in dollars -/
def certification_cost : ℕ := 3000

/-- The percentage of certification cost covered by insurance -/
def insurance_coverage : ℕ := 90

/-- The total out-of-pocket cost in dollars -/
def total_out_of_pocket : ℕ := 3450

theorem dog_training_weeks :
  adoption_fee +
  training_cost_per_week * training_weeks +
  certification_cost * (100 - insurance_coverage) / 100 =
  total_out_of_pocket :=
by sorry

end dog_training_weeks_l3512_351201


namespace complex_on_imaginary_axis_l3512_351286

theorem complex_on_imaginary_axis (a : ℝ) :
  let z : ℂ := Complex.mk (a^2 - 2*a) (a^2 - a - 2)
  (Complex.re z = 0) → (a = 0 ∨ a = 2) :=
by sorry

end complex_on_imaginary_axis_l3512_351286


namespace sum_of_perpendiculars_equals_5_sqrt_3_l3512_351228

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (side_length : ℝ)

-- Define a point inside the triangle
structure PointInTriangle :=
  (triangle : EquilateralTriangle)
  (inside : Bool)

-- Define the sum of perpendiculars
def sum_of_perpendiculars (p : PointInTriangle) : ℝ := sorry

-- Theorem statement
theorem sum_of_perpendiculars_equals_5_sqrt_3 
  (p : PointInTriangle) 
  (h : p.triangle.side_length = 10) :
  sum_of_perpendiculars p = 5 * Real.sqrt 3 := by sorry

end sum_of_perpendiculars_equals_5_sqrt_3_l3512_351228


namespace circle_area_l3512_351209

theorem circle_area (r : ℝ) (h : r = 11) : π * r^2 = π * 11^2 := by
  sorry

end circle_area_l3512_351209


namespace range_of_a_l3512_351213

-- Define a decreasing function on (-1, 1)
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : IsDecreasingOn f)
  (h2 : f (1 - a) < f (2 * a - 1)) :
  0 < a ∧ a < 2/3 := by
  sorry

end range_of_a_l3512_351213


namespace even_function_implies_a_zero_l3512_351210

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

theorem even_function_implies_a_zero (a : ℝ) :
  IsEven (f a) → a = 0 := by
  sorry

end even_function_implies_a_zero_l3512_351210


namespace fourth_competitor_jump_distance_l3512_351223

/-- Long jump competition with four competitors -/
structure LongJumpCompetition where
  first_jump : ℕ
  second_jump : ℕ
  third_jump : ℕ
  fourth_jump : ℕ

/-- The long jump competition satisfying the given conditions -/
def competition : LongJumpCompetition where
  first_jump := 22
  second_jump := 23
  third_jump := 21
  fourth_jump := 24

/-- Theorem stating the conditions and the result to be proved -/
theorem fourth_competitor_jump_distance :
  let c := competition
  c.first_jump = 22 ∧
  c.second_jump = c.first_jump + 1 ∧
  c.third_jump = c.second_jump - 2 ∧
  c.fourth_jump = c.third_jump + 3 →
  c.fourth_jump = 24 := by
  sorry


end fourth_competitor_jump_distance_l3512_351223


namespace mango_community_ratio_l3512_351299

/-- Represents the mango harvest and sales problem. -/
structure MangoHarvest where
  total_kg : ℕ  -- Total kilograms of mangoes harvested
  sold_market_kg : ℕ  -- Kilograms of mangoes sold to the market
  mangoes_per_kg : ℕ  -- Number of mangoes per kilogram
  mangoes_left : ℕ  -- Number of mangoes left after sales

/-- The ratio of mangoes sold to the community to total mangoes harvested is 1/3. -/
theorem mango_community_ratio (h : MangoHarvest) 
  (h_total : h.total_kg = 60)
  (h_market : h.sold_market_kg = 20)
  (h_per_kg : h.mangoes_per_kg = 8)
  (h_left : h.mangoes_left = 160) :
  (h.total_kg * h.mangoes_per_kg - h.sold_market_kg * h.mangoes_per_kg - h.mangoes_left) / 
  (h.total_kg * h.mangoes_per_kg) = 1 / 3 :=
sorry

end mango_community_ratio_l3512_351299


namespace circle_division_theorem_l3512_351293

/-- A type representing a straight cut through a circle -/
structure Cut where
  -- We don't need to define the internal structure of a cut for this statement

/-- A type representing a circle -/
structure Circle where
  -- We don't need to define the internal structure of a circle for this statement

/-- A function that counts the number of regions created by cuts in a circle -/
def count_regions (circle : Circle) (cuts : List Cut) : ℕ := sorry

/-- Theorem stating that a circle can be divided into 4, 5, 6, and 7 parts using three straight cuts -/
theorem circle_division_theorem (circle : Circle) :
  ∃ (cuts₁ cuts₂ cuts₃ cuts₄ : List Cut),
    (cuts₁.length = 3 ∧ count_regions circle cuts₁ = 4) ∧
    (cuts₂.length = 3 ∧ count_regions circle cuts₂ = 5) ∧
    (cuts₃.length = 3 ∧ count_regions circle cuts₃ = 6) ∧
    (cuts₄.length = 3 ∧ count_regions circle cuts₄ = 7) :=
  sorry

end circle_division_theorem_l3512_351293


namespace tom_teaching_years_l3512_351240

theorem tom_teaching_years (tom devin : ℕ) 
  (h1 : tom + devin = 70)
  (h2 : devin = tom / 2 - 5) :
  tom = 50 := by
  sorry

end tom_teaching_years_l3512_351240


namespace min_tiles_for_square_l3512_351260

theorem min_tiles_for_square (tile_width : ℕ) (tile_height : ℕ) : 
  tile_width = 12 →
  tile_height = 15 →
  ∃ (square_side : ℕ) (num_tiles : ℕ),
    square_side % tile_width = 0 ∧
    square_side % tile_height = 0 ∧
    num_tiles = (square_side * square_side) / (tile_width * tile_height) ∧
    num_tiles = 20 ∧
    ∀ (smaller_side : ℕ) (smaller_num_tiles : ℕ),
      smaller_side < square_side →
      smaller_side % tile_width = 0 →
      smaller_side % tile_height = 0 →
      smaller_num_tiles = (smaller_side * smaller_side) / (tile_width * tile_height) →
      smaller_num_tiles < num_tiles :=
by sorry

end min_tiles_for_square_l3512_351260


namespace janous_inequality_l3512_351277

theorem janous_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ ∃ (k : ℝ), k > 0 ∧ x = 2*k ∧ y = k ∧ z = k) :=
by sorry

end janous_inequality_l3512_351277


namespace complex_modulus_problem_l3512_351278

open Complex

theorem complex_modulus_problem (z : ℂ) (h : (1 - I) * z = 2 * I) : abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l3512_351278


namespace gcd_lcm_45_150_l3512_351237

theorem gcd_lcm_45_150 : 
  (Nat.gcd 45 150 = 15) ∧ (Nat.lcm 45 150 = 450) := by
  sorry

end gcd_lcm_45_150_l3512_351237


namespace min_panels_for_intensity_reduction_l3512_351257

/-- Represents the reduction factor of light intensity when passing through a glass panel -/
def reduction_factor : ℝ := 0.9

/-- Calculates the light intensity after passing through a number of panels -/
def intensity_after_panels (a : ℝ) (x : ℕ) : ℝ := a * reduction_factor ^ x

/-- Theorem stating the minimum number of panels required to reduce light intensity to less than 1/11 of original -/
theorem min_panels_for_intensity_reduction (a : ℝ) (h : a > 0) :
  ∃ x : ℕ, (∀ y : ℕ, y < x → intensity_after_panels a y ≥ a / 11) ∧
           intensity_after_panels a x < a / 11 :=
by sorry

end min_panels_for_intensity_reduction_l3512_351257


namespace ordered_pairs_sum_30_l3512_351291

theorem ordered_pairs_sum_30 :
  (Finset.filter (fun p : ℕ × ℕ => p.1 + p.2 = 30 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 31) (Finset.range 31))).card = 29 := by
  sorry

end ordered_pairs_sum_30_l3512_351291


namespace coin_problem_l3512_351281

theorem coin_problem (total_coins : ℕ) (total_value : ℚ) (nickel_value dime_value : ℚ) :
  total_coins = 28 →
  total_value = 260/100 →
  nickel_value = 5/100 →
  dime_value = 10/100 →
  ∃ (nickels dimes : ℕ),
    nickels + dimes = total_coins ∧
    nickels * nickel_value + dimes * dime_value = total_value ∧
    nickels = 4 :=
by sorry

end coin_problem_l3512_351281


namespace orlies_age_l3512_351205

/-- Proves Orlie's age given the conditions about Ruffy and Orlie's ages -/
theorem orlies_age (ruffy_age orlie_age : ℕ) : 
  ruffy_age = 9 →
  ruffy_age = (3 / 4) * orlie_age →
  ruffy_age - 4 = (1 / 2) * (orlie_age - 4) + 1 →
  orlie_age = 12 := by
  sorry

#check orlies_age

end orlies_age_l3512_351205


namespace rationalization_sqrt_five_l3512_351216

/-- Rationalization of (2+√5)/(2-√5) -/
theorem rationalization_sqrt_five : ∃ (A B C : ℤ), 
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C ∧ 
  A = -9 ∧ B = -4 ∧ C = 5 := by
  sorry

end rationalization_sqrt_five_l3512_351216


namespace haleys_trees_l3512_351258

theorem haleys_trees (dead : ℕ) (survived : ℕ) : 
  dead = 6 → 
  survived = dead + 1 → 
  dead + survived = 13 :=
by
  sorry

end haleys_trees_l3512_351258


namespace triangle_area_l3512_351265

theorem triangle_area (a b c : ℝ) (h1 : c^2 = a^2 + b^2 - 2*a*b + 6) (h2 : 0 < a ∧ 0 < b ∧ 0 < c) : 
  (1/2) * a * b * Real.sin (π/3) = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_area_l3512_351265


namespace system_solution_l3512_351290

theorem system_solution (a₁ a₂ a₃ a₄ : ℝ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄) :
  ∃! (x₁ x₂ x₃ x₄ : ℝ),
    (|a₁ - a₂| * x₂ + |a₁ - a₃| * x₃ + |a₁ - a₄| * x₄ = 1) ∧
    (|a₂ - a₁| * x₁ + |a₂ - a₃| * x₃ + |a₂ - a₄| * x₄ = 1) ∧
    (|a₃ - a₁| * x₁ + |a₃ - a₂| * x₂ + |a₃ - a₄| * x₄ = 1) ∧
    (|a₄ - a₁| * x₁ + |a₄ - a₂| * x₂ + |a₄ - a₃| * x₃ = 1) ∧
    (x₁ = x₂) ∧ (x₂ = x₃) ∧ (x₃ = x₄) ∧
    (x₁ = 1 / (3 * a₁ - (a₂ + a₃ + a₄))) := by
  sorry

end system_solution_l3512_351290


namespace triangle_inequality_check_l3512_351268

def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_check :
  (canFormTriangle 2 3 4) ∧
  ¬(canFormTriangle 3 4 7) ∧
  ¬(canFormTriangle 4 6 2) ∧
  ¬(canFormTriangle 7 10 2) :=
by
  sorry

end triangle_inequality_check_l3512_351268


namespace two_numbers_difference_l3512_351287

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 40)
  (weighted_diff_eq : 2 * y - 3 * x = 10) :
  |y - x| = 12 := by
sorry

end two_numbers_difference_l3512_351287


namespace cow_count_theorem_l3512_351232

/-- The number of cows on a dairy farm -/
def number_of_cows : ℕ := 20

/-- The number of bags of husk eaten by some cows in 20 days -/
def total_bags_eaten : ℕ := 20

/-- The number of bags of husk eaten by one cow in 20 days -/
def bags_per_cow : ℕ := 1

/-- Theorem stating that the number of cows is equal to the total bags eaten divided by the bags eaten per cow -/
theorem cow_count_theorem : number_of_cows = total_bags_eaten / bags_per_cow := by
  sorry

end cow_count_theorem_l3512_351232


namespace g_75_solutions_l3512_351262

-- Define g₀
def g₀ (x : ℝ) : ℝ := x + |x - 150| - |x + 150|

-- Define gₙ recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

-- Theorem statement
theorem g_75_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, g 75 x = 0) ∧
                    (∀ x ∉ S, g 75 x ≠ 0) ∧
                    Finset.card S = 4 := by
  sorry

end g_75_solutions_l3512_351262


namespace max_true_statements_l3512_351203

theorem max_true_statements (x : ℝ) : 
  let statements := [
    (0 < x^2 ∧ x^2 < 1),
    (x^2 > 1),
    (-1 < x ∧ x < 0),
    (0 < x ∧ x < 1),
    (0 < x - x^3 ∧ x - x^3 < 1)
  ]
  ¬∃ (s : Finset (Fin 5)), s.card > 3 ∧ (∀ i ∈ s, statements[i]) :=
by sorry

end max_true_statements_l3512_351203


namespace train_crossing_time_l3512_351214

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 750 ∧ 
  train_speed_kmh = 180 →
  crossing_time = 15 := by sorry

end train_crossing_time_l3512_351214


namespace ellipse_hyperbola_properties_l3512_351264

/-- An ellipse and hyperbola with shared properties -/
structure EllipseHyperbola where
  /-- The distance between the foci -/
  focal_distance : ℝ
  /-- The difference between the major axis of the ellipse and the real axis of the hyperbola -/
  axis_difference : ℝ
  /-- The ratio of eccentricities (ellipse:hyperbola) -/
  eccentricity_ratio : ℝ × ℝ

/-- The equations of the ellipse and hyperbola -/
def curve_equations (eh : EllipseHyperbola) : (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop) :=
  (λ x y ↦ x^2/49 + y^2/36 = 1, λ x y ↦ x^2/9 - y^2/4 = 1)

/-- The area of the triangle formed by the foci and an intersection point -/
def triangle_area (eh : EllipseHyperbola) : ℝ := 12

/-- Theorem stating the properties of the ellipse and hyperbola -/
theorem ellipse_hyperbola_properties (eh : EllipseHyperbola)
    (h1 : eh.focal_distance = 2 * Real.sqrt 13)
    (h2 : eh.axis_difference = 4)
    (h3 : eh.eccentricity_ratio = (3, 7)) :
    curve_equations eh = (λ x y ↦ x^2/49 + y^2/36 = 1, λ x y ↦ x^2/9 - y^2/4 = 1) ∧
    triangle_area eh = 12 := by
  sorry

end ellipse_hyperbola_properties_l3512_351264


namespace product_not_negative_l3512_351292

theorem product_not_negative (x y : ℝ) (n : ℕ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x^(2*n) - y^(2*n) > x) (h2 : y^(2*n) - x^(2*n) > y) :
  x * y > 0 :=
sorry

end product_not_negative_l3512_351292


namespace alley_width_l3512_351267

/-- The width of an alley given a ladder's two configurations -/
theorem alley_width (L : ℝ) (k h : ℝ) : ∃ w : ℝ,
  (k = L / 2) →
  (h = L * Real.sqrt 3 / 2) →
  (w^2 + (L/2)^2 = L^2) →
  (w^2 + (L * Real.sqrt 3 / 2)^2 = L^2) →
  w = L * Real.sqrt 3 / 2 := by
  sorry

end alley_width_l3512_351267


namespace driving_time_to_school_l3512_351279

theorem driving_time_to_school 
  (total_hours : ℕ) 
  (school_days : ℕ) 
  (drives_both_ways : Bool) : 
  total_hours = 50 → 
  school_days = 75 → 
  drives_both_ways = true → 
  (total_hours * 60) / (school_days * 2) = 20 := by
sorry

end driving_time_to_school_l3512_351279


namespace math_contest_schools_count_l3512_351222

/-- Represents a participant in the math contest -/
structure Participant where
  score : ℕ
  rank : ℕ

/-- Represents a school team in the math contest -/
structure School where
  team : Fin 4 → Participant

/-- The math contest -/
structure MathContest where
  schools : List School
  andrea : Participant
  beth : Participant
  carla : Participant

/-- The conditions of the math contest -/
def ContestConditions (contest : MathContest) : Prop :=
  ∀ s₁ s₂ : School, ∀ p₁ p₂ : Fin 4, 
    (s₁ ≠ s₂ ∨ p₁ ≠ p₂) → (s₁.team p₁).score ≠ (s₂.team p₂).score
  ∧ contest.andrea.rank < contest.beth.rank
  ∧ contest.beth.rank = 46
  ∧ contest.carla.rank = 79
  ∧ contest.andrea.rank = (contest.schools.length * 4 + 1) / 2
  ∧ ∀ s : School, ∀ p : Fin 4, contest.andrea.score ≥ (s.team p).score

theorem math_contest_schools_count 
  (contest : MathContest) 
  (h : ContestConditions contest) : 
  contest.schools.length = 19 := by
  sorry


end math_contest_schools_count_l3512_351222


namespace smallest_angle_in_ratio_triangle_l3512_351294

theorem smallest_angle_in_ratio_triangle (α β γ : Real) : 
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- Angles are positive
  β = 2 * α ∧ γ = 3 * α →  -- Angle ratio is 1 : 2 : 3
  α + β + γ = π →         -- Sum of angles in a triangle
  α = π / 6 := by
    sorry

end smallest_angle_in_ratio_triangle_l3512_351294


namespace parallel_distinct_iff_a_eq_3_l3512_351266

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ × ℝ → Prop
  line2 : ℝ × ℝ → Prop
  line1_def : ∀ x y, line1 (x, y) ↔ a * x + 2 * y + 3 * a = 0
  line2_def : ∀ x y, line2 (x, y) ↔ 3 * x + (a - 1) * y = a - 7

/-- The lines are parallel -/
def parallel (tl : TwoLines) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, tl.line1 (x, y) ↔ tl.line2 (k * x, k * y)

/-- The lines are distinct -/
def distinct (tl : TwoLines) : Prop :=
  ∃ p, tl.line1 p ∧ ¬tl.line2 p

/-- The main theorem -/
theorem parallel_distinct_iff_a_eq_3 (tl : TwoLines) :
  parallel tl ∧ distinct tl ↔ tl.a = 3 :=
sorry

end parallel_distinct_iff_a_eq_3_l3512_351266


namespace triangle_rigidity_connected_beams_rigidity_l3512_351288

-- Define a structure for a triangle with three sides
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

-- Define a property for a triangle to be rigid
def is_rigid (t : Triangle) : Prop :=
  ∀ (t' : Triangle), t.side1 = t'.side1 ∧ t.side2 = t'.side2 ∧ t.side3 = t'.side3 →
    t = t'

-- Theorem stating that a triangle with fixed side lengths is rigid
theorem triangle_rigidity (t : Triangle) :
  is_rigid t :=
sorry

-- Define a beam as a line segment with fixed length
def Beam := ℝ

-- Define a structure for the connected beams
structure ConnectedBeams :=
  (beam1 : Beam)
  (beam2 : Beam)
  (beam3 : Beam)

-- Function to convert connected beams to a triangle
def beams_to_triangle (b : ConnectedBeams) : Triangle :=
  { side1 := b.beam1,
    side2 := b.beam2,
    side3 := b.beam3 }

-- Theorem stating that connected beams with fixed lengths form a rigid structure
theorem connected_beams_rigidity (b : ConnectedBeams) :
  is_rigid (beams_to_triangle b) :=
sorry

end triangle_rigidity_connected_beams_rigidity_l3512_351288

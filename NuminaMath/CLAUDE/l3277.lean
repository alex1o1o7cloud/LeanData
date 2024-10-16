import Mathlib

namespace NUMINAMATH_CALUDE_max_d_value_l3277_327733

def is_prime (n : ℕ) : Prop := sorry

def max_list (l : List ℕ) : ℕ := sorry

def min_list (l : List ℕ) : ℕ := sorry

theorem max_d_value (a b c : ℕ) : 
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  is_prime (a + b - c) ∧ is_prime (a + c - b) ∧ is_prime (b + c - a) ∧ is_prime (a + b + c) ∧
  (a + b = 800 ∨ a + c = 800 ∨ b + c = 800) ∧
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  a ≠ (a + b - c) ∧ a ≠ (a + c - b) ∧ a ≠ (b + c - a) ∧ a ≠ (a + b + c) ∧
  b ≠ (a + b - c) ∧ b ≠ (a + c - b) ∧ b ≠ (b + c - a) ∧ b ≠ (a + b + c) ∧
  c ≠ (a + b - c) ∧ c ≠ (a + c - b) ∧ c ≠ (b + c - a) ∧ c ≠ (a + b + c) ∧
  (a + b - c) ≠ (a + c - b) ∧ (a + b - c) ≠ (b + c - a) ∧ (a + b - c) ≠ (a + b + c) ∧
  (a + c - b) ≠ (b + c - a) ∧ (a + c - b) ≠ (a + b + c) ∧
  (b + c - a) ≠ (a + b + c) →
  max_list [a, b, c, a + b - c, a + c - b, b + c - a, a + b + c] - 
  min_list [a, b, c, a + b - c, a + c - b, b + c - a, a + b + c] ≤ 
  max_list [3, 797, c, 800 - c, 3 + c, 797 + c, 800 + c] - 
  min_list [3, 797, c, 800 - c, 3 + c, 797 + c, 800 + c] := by
sorry

end NUMINAMATH_CALUDE_max_d_value_l3277_327733


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3277_327711

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, y^2 = x^3 + 2*x^2 + 2*x + 1 ↔ (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = -1) ∨ (x = 0 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3277_327711


namespace NUMINAMATH_CALUDE_expected_points_100_games_prob_specific_envelope_l3277_327749

/- Define the game parameters -/
def num_envelopes : ℕ := 13
def win_points : ℕ := 6

/- Define the probability of winning a single question -/
def win_prob : ℚ := 1/2

/- Define the expected number of envelopes played in a single game -/
noncomputable def expected_envelopes_per_game : ℝ := 12

/- Theorem for the expected points over 100 games -/
theorem expected_points_100_games :
  ∃ (expected_points : ℕ), expected_points = 465 := by sorry

/- Theorem for the probability of choosing a specific envelope -/
theorem prob_specific_envelope :
  ∃ (prob : ℚ), prob = 12/13 := by sorry

end NUMINAMATH_CALUDE_expected_points_100_games_prob_specific_envelope_l3277_327749


namespace NUMINAMATH_CALUDE_joyce_apple_count_l3277_327789

/-- The number of apples Joyce ends up with after receiving apples from Larry -/
def final_apple_count (initial : ℝ) (received : ℝ) : ℝ :=
  initial + received

/-- Theorem stating that Joyce ends up with 127.0 apples -/
theorem joyce_apple_count : final_apple_count 75.0 52.0 = 127.0 := by
  sorry

end NUMINAMATH_CALUDE_joyce_apple_count_l3277_327789


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l3277_327754

theorem cos_2alpha_value (α : Real) (h : Real.sin (α + 3 * Real.pi / 2) = Real.sqrt 3 / 3) :
  Real.cos (2 * α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l3277_327754


namespace NUMINAMATH_CALUDE_age_difference_ratio_l3277_327729

/-- Represents the current ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.roy = ages.julia + 8 ∧
  ages.roy + 2 = 2 * (ages.julia + 2) ∧
  (ages.roy + 2) * (ages.kelly + 2) = 192

/-- The theorem to be proved -/
theorem age_difference_ratio (ages : Ages) :
  satisfiesConditions ages →
  (ages.roy - ages.julia) / (ages.roy - ages.kelly) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_ratio_l3277_327729


namespace NUMINAMATH_CALUDE_symmetry_and_rotation_sum_l3277_327764

/-- The number of sides in our regular polygon -/
def n : ℕ := 17

/-- The number of lines of symmetry in a regular n-gon -/
def L (n : ℕ) : ℕ := n

/-- The smallest positive angle (in degrees) for which a regular n-gon has rotational symmetry -/
def R (n : ℕ) : ℚ := 360 / n

/-- Theorem: For a regular 17-gon, the sum of its number of lines of symmetry 
    and its smallest positive angle of rotational symmetry (in degrees) 
    is equal to 17 + 360/17 -/
theorem symmetry_and_rotation_sum : 
  (L n : ℚ) + R n = 17 + 360 / 17 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_and_rotation_sum_l3277_327764


namespace NUMINAMATH_CALUDE_square_perimeter_quadrupled_l3277_327724

theorem square_perimeter_quadrupled (s : ℝ) (x : ℝ) :
  x = 4 * s →
  4 * x = 4 * (4 * s) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_quadrupled_l3277_327724


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3277_327701

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_3rd : a 3 = 10)
  (h_6th : a 6 = 20) :
  a 12 = 40 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3277_327701


namespace NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l3277_327782

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 30th term of the specified arithmetic sequence is 264. -/
theorem arithmetic_sequence_30th_term :
  ∀ a : ℕ → ℝ, is_arithmetic_sequence a →
  a 1 = 3 → a 2 = 12 → a 3 = 21 →
  a 30 = 264 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l3277_327782


namespace NUMINAMATH_CALUDE_corrected_mean_specific_corrected_mean_l3277_327713

/-- Given a set of observations with an incorrect entry, calculate the corrected mean -/
theorem corrected_mean (n : ℕ) (original_mean : ℝ) (incorrect_value correct_value : ℝ) :
  n > 0 →
  let total_sum := n * original_mean
  let corrected_sum := total_sum - incorrect_value + correct_value
  corrected_sum / n = (n * original_mean - incorrect_value + correct_value) / n :=
by sorry

/-- The specific problem instance -/
theorem specific_corrected_mean :
  let n : ℕ := 40
  let original_mean : ℝ := 100
  let incorrect_value : ℝ := 75
  let correct_value : ℝ := 50
  (n * original_mean - incorrect_value + correct_value) / n = 99.375 :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_specific_corrected_mean_l3277_327713


namespace NUMINAMATH_CALUDE_star_operation_associative_l3277_327781

-- Define the curve y = x^3
def cubic_curve (x : ℝ) : ℝ := x^3

-- Define a point on the curve
structure CurvePoint where
  x : ℝ
  y : ℝ
  on_curve : y = cubic_curve x

-- Define the * operation
def star_operation (A B : CurvePoint) : CurvePoint :=
  sorry

-- Theorem statement
theorem star_operation_associative :
  ∀ (A B C : CurvePoint),
    star_operation (star_operation A B) C = star_operation A (star_operation B C) := by
  sorry

end NUMINAMATH_CALUDE_star_operation_associative_l3277_327781


namespace NUMINAMATH_CALUDE_dividend_calculation_l3277_327737

theorem dividend_calculation (remainder : ℕ) (divisor : ℕ) (quotient : ℕ) :
  remainder = 6 →
  divisor = 5 * quotient →
  divisor = 3 * remainder + 2 →
  divisor * quotient + remainder = 86 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3277_327737


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l3277_327721

/-- The width of a rectangular prism with given dimensions and diagonal length -/
theorem rectangular_prism_width (l h d : ℝ) (hl : l = 5) (hh : h = 15) (hd : d = 17) :
  ∃ w : ℝ, w ^ 2 = 39 ∧ d ^ 2 = l ^ 2 + w ^ 2 + h ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l3277_327721


namespace NUMINAMATH_CALUDE_square_roots_calculation_l3277_327763

theorem square_roots_calculation : (Real.sqrt 3 + Real.sqrt 2)^2 * (5 - 2 * Real.sqrt 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_calculation_l3277_327763


namespace NUMINAMATH_CALUDE_inscribable_polygons_l3277_327727

/-- The number of evenly spaced holes on the circumference of the circle -/
def num_holes : ℕ := 24

/-- A function that determines if a regular polygon with 'n' sides can be inscribed in the circle -/
def can_inscribe (n : ℕ) : Prop :=
  n ≥ 3 ∧ num_holes % n = 0

/-- The set of numbers of sides for regular polygons that can be inscribed in the circle -/
def valid_polygons : Set ℕ := {n | can_inscribe n}

/-- Theorem stating that the only valid numbers of sides for inscribable regular polygons are 3, 4, 6, 8, 12, and 24 -/
theorem inscribable_polygons :
  valid_polygons = {3, 4, 6, 8, 12, 24} :=
sorry

end NUMINAMATH_CALUDE_inscribable_polygons_l3277_327727


namespace NUMINAMATH_CALUDE_angle_range_in_special_triangle_l3277_327758

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a + b = 2c, then 0 < C ≤ π/3 -/
theorem angle_range_in_special_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a + b = 2 * c →
  0 < C ∧ C ≤ π / 3 := by
  sorry


end NUMINAMATH_CALUDE_angle_range_in_special_triangle_l3277_327758


namespace NUMINAMATH_CALUDE_max_parts_5x5_grid_l3277_327706

/-- Represents a partition of a grid into parts with different areas -/
def GridPartition (n : ℕ) := List ℕ

/-- The sum of areas in a partition should equal the total grid area -/
def validPartition (g : ℕ) (p : GridPartition g) : Prop :=
  p.sum = g * g ∧ p.Nodup

/-- The maximum number of parts in a valid partition of a 5x5 grid -/
theorem max_parts_5x5_grid :
  (∃ (p : GridPartition 5), validPartition 5 p ∧ p.length = 6) ∧
  (∀ (p : GridPartition 5), validPartition 5 p → p.length ≤ 6) := by
  sorry

#check max_parts_5x5_grid

end NUMINAMATH_CALUDE_max_parts_5x5_grid_l3277_327706


namespace NUMINAMATH_CALUDE_smallest_product_increase_l3277_327723

theorem smallest_product_increase (p q r s : ℝ) 
  (h_pos : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s) 
  (h_order : p < q ∧ q < r ∧ r < s) : 
  min (min (min ((p+1)*q*r*s) (p*(q+1)*r*s)) (p*q*(r+1)*s)) (p*q*r*(s+1)) = p*q*r*(s+1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_increase_l3277_327723


namespace NUMINAMATH_CALUDE_son_work_time_l3277_327791

/-- Given a man and his son working on a job, this theorem proves how long it takes the son to complete the job alone. -/
theorem son_work_time (man_time son_time combined_time : ℚ)
  (hman : man_time = 5)
  (hcombined : combined_time = 4)
  (hwork : (1 / man_time) + (1 / son_time) = 1 / combined_time) :
  son_time = 20 := by
  sorry

#check son_work_time

end NUMINAMATH_CALUDE_son_work_time_l3277_327791


namespace NUMINAMATH_CALUDE_max_k_is_19_l3277_327710

/-- Represents a two-digit number -/
def TwoDigitNumber (a b : Nat) : Prop :=
  a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9

/-- Represents a three-digit number formed by inserting a digit between two others -/
def ThreeDigitNumber (a c b : Nat) : Prop :=
  a ≠ 0 ∧ a ≤ 9 ∧ c ≤ 9 ∧ b ≤ 9

/-- The value of a two-digit number -/
def twoDigitValue (a b : Nat) : Nat :=
  10 * a + b

/-- The value of a three-digit number -/
def threeDigitValue (a c b : Nat) : Nat :=
  100 * a + 10 * c + b

/-- The theorem stating that the maximum value of k is 19 -/
theorem max_k_is_19 :
  ∀ a b c k : Nat,
  TwoDigitNumber a b →
  ThreeDigitNumber a c b →
  threeDigitValue a c b = k * twoDigitValue a b →
  k ≤ 19 :=
sorry

end NUMINAMATH_CALUDE_max_k_is_19_l3277_327710


namespace NUMINAMATH_CALUDE_calculation_proof_l3277_327720

theorem calculation_proof : (3.14 - 1) ^ 0 * (-1/4) ^ (-2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3277_327720


namespace NUMINAMATH_CALUDE_connie_initial_marbles_l3277_327731

/-- The number of marbles Connie initially had -/
def initial_marbles : ℕ := 241

/-- The number of marbles Connie bought -/
def bought_marbles : ℕ := 45

/-- The number of marbles Connie gave to Juan -/
def given_to_juan : ℕ := 73

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 70

theorem connie_initial_marbles :
  (initial_marbles + bought_marbles) / 2 - given_to_juan = marbles_left :=
by sorry

end NUMINAMATH_CALUDE_connie_initial_marbles_l3277_327731


namespace NUMINAMATH_CALUDE_x_value_l3277_327744

theorem x_value (x y : ℚ) (h1 : x / y = 15 / 5) (h2 : y = 10) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3277_327744


namespace NUMINAMATH_CALUDE_correct_tree_planting_equation_l3277_327738

/-- Represents the equation for tree planting by students -/
def tree_planting_equation (x : ℕ) : Prop :=
  let total_students : ℕ := 20
  let total_seedlings : ℕ := 52
  let male_seedlings : ℕ := 3
  let female_seedlings : ℕ := 2
  x ≤ total_students ∧ 
  (male_seedlings * x + female_seedlings * (total_students - x) = total_seedlings)

/-- Theorem stating the correct equation for the tree planting problem -/
theorem correct_tree_planting_equation :
  ∃ x : ℕ, tree_planting_equation x ∧ 
  (3 * x + 2 * (20 - x) = 52) :=
sorry

end NUMINAMATH_CALUDE_correct_tree_planting_equation_l3277_327738


namespace NUMINAMATH_CALUDE_defective_units_shipped_percentage_l3277_327797

/-- The percentage of units with Type A defects in the first stage -/
def type_a_defect_rate : ℝ := 0.07

/-- The percentage of units with Type B defects in the second stage -/
def type_b_defect_rate : ℝ := 0.08

/-- The percentage of Type A defects that are reworked and repaired -/
def type_a_repair_rate : ℝ := 0.4

/-- The percentage of Type B defects that are reworked and repaired -/
def type_b_repair_rate : ℝ := 0.3

/-- The percentage of remaining Type A defects that are shipped for sale -/
def type_a_ship_rate : ℝ := 0.03

/-- The percentage of remaining Type B defects that are shipped for sale -/
def type_b_ship_rate : ℝ := 0.06

/-- The theorem stating the percentage of units produced that are defective (Type A or B) and shipped for sale -/
theorem defective_units_shipped_percentage :
  (type_a_defect_rate * (1 - type_a_repair_rate) * type_a_ship_rate +
   type_b_defect_rate * (1 - type_b_repair_rate) * type_b_ship_rate) * 100 =
  0.462 := by sorry

end NUMINAMATH_CALUDE_defective_units_shipped_percentage_l3277_327797


namespace NUMINAMATH_CALUDE_opposite_of_five_l3277_327722

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_five : opposite 5 = -5 := by
  -- The proof goes here
  sorry

-- Lemma to show that the opposite satisfies the required property
lemma opposite_property (a : ℝ) : a + opposite a = 0 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_five_l3277_327722


namespace NUMINAMATH_CALUDE_painted_cube_problem_l3277_327728

theorem painted_cube_problem (n : ℕ) : 
  n > 0 → 
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → 
  n = 2 ∧ n^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_painted_cube_problem_l3277_327728


namespace NUMINAMATH_CALUDE_election_win_percentage_l3277_327777

theorem election_win_percentage 
  (total_votes : ℕ) 
  (geoff_percentage : ℚ) 
  (additional_votes_needed : ℕ) 
  (h1 : total_votes = 6000)
  (h2 : geoff_percentage = 1/200)  -- 0.5% as a rational number
  (h3 : additional_votes_needed = 3000) :
  (((geoff_percentage * total_votes + additional_votes_needed) / total_votes) : ℚ) = 101/200 := by
sorry

end NUMINAMATH_CALUDE_election_win_percentage_l3277_327777


namespace NUMINAMATH_CALUDE_saree_price_calculation_l3277_327709

theorem saree_price_calculation (final_price : ℝ) 
  (h1 : final_price = 224) 
  (discount1 : ℝ) (h2 : discount1 = 0.3)
  (discount2 : ℝ) (h3 : discount2 = 0.2) : 
  ∃ (original_price : ℝ), 
    original_price = 400 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l3277_327709


namespace NUMINAMATH_CALUDE_potato_bag_weight_l3277_327714

/-- The weight of each bag of potatoes -/
def bag_weight (total_potatoes damaged_potatoes : ℕ) (price_per_bag total_revenue : ℚ) : ℚ :=
  (total_potatoes - damaged_potatoes) * price_per_bag / total_revenue

/-- Theorem stating the weight of each bag of potatoes -/
theorem potato_bag_weight :
  bag_weight 6500 150 72 9144 = 50 := by
  sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l3277_327714


namespace NUMINAMATH_CALUDE_unique_prime_sum_of_squares_and_divisibility_l3277_327740

theorem unique_prime_sum_of_squares_and_divisibility :
  ∃! (p : ℕ), Prime p ∧
    (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧
      p = m^2 + n^2 ∧
      p ∣ (m^3 + n^3 + 8*m*n)) ∧
    p = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_sum_of_squares_and_divisibility_l3277_327740


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_b_l3277_327772

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_consecutive_b : 
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (b n) (b (n + 1)) ≤ k ∧ 
  ∃ (m : ℕ), Nat.gcd (b m) (b (m + 1)) = k :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_b_l3277_327772


namespace NUMINAMATH_CALUDE_intersection_tangent_line_constant_l3277_327761

/-- Given two curves f(x) = √x and g(x) = a ln x that intersect and have the same tangent line
    at the point of intersection, prove that a = e/2 -/
theorem intersection_tangent_line_constant (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.sqrt x₀ = a * Real.log x₀ ∧ 
    (1 / (2 * Real.sqrt x₀) : ℝ) = a / x₀) →
  a = Real.exp 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_tangent_line_constant_l3277_327761


namespace NUMINAMATH_CALUDE_factorization_ax2_minus_4ay2_l3277_327708

theorem factorization_ax2_minus_4ay2 (a x y : ℝ) :
  a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ax2_minus_4ay2_l3277_327708


namespace NUMINAMATH_CALUDE_fraction_puzzle_l3277_327746

theorem fraction_puzzle (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : 
  min x y = 1/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_puzzle_l3277_327746


namespace NUMINAMATH_CALUDE_wholesale_price_is_90_l3277_327735

def retail_price : ℝ := 120

def discount_rate : ℝ := 0.1

def profit_rate : ℝ := 0.2

def selling_price (retail : ℝ) (discount : ℝ) : ℝ :=
  retail * (1 - discount)

def profit (wholesale : ℝ) (rate : ℝ) : ℝ :=
  wholesale * rate

theorem wholesale_price_is_90 :
  ∃ (wholesale : ℝ),
    selling_price retail_price discount_rate = wholesale + profit wholesale profit_rate ∧
    wholesale = 90 := by sorry

end NUMINAMATH_CALUDE_wholesale_price_is_90_l3277_327735


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l3277_327750

/-- Given a tetrahedron with volume V, two faces with areas S₁ and S₂, 
    their common edge of length a, and the dihedral angle φ between these faces, 
    prove that V = (2/3) * (S₁ * S₂ * sin(φ)) / a. -/
theorem tetrahedron_volume (V S₁ S₂ a φ : ℝ) 
  (h₁ : V > 0) 
  (h₂ : S₁ > 0) 
  (h₃ : S₂ > 0) 
  (h₄ : a > 0) 
  (h₅ : 0 < φ ∧ φ < π) : 
  V = (2/3) * (S₁ * S₂ * Real.sin φ) / a := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l3277_327750


namespace NUMINAMATH_CALUDE_unique_prime_solution_l3277_327792

theorem unique_prime_solution :
  ∀ p q : ℕ,
    Prime p → Prime q →
    p^3 - q^5 = (p + q)^2 →
    p = 7 ∧ q = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l3277_327792


namespace NUMINAMATH_CALUDE_initial_investment_calculation_l3277_327787

def initial_rate : ℚ := 5 / 100
def additional_rate : ℚ := 8 / 100
def total_rate : ℚ := 6 / 100
def additional_investment : ℚ := 4000

theorem initial_investment_calculation (x : ℚ) :
  initial_rate * x + additional_rate * additional_investment = 
  total_rate * (x + additional_investment) →
  x = 8000 := by
sorry

end NUMINAMATH_CALUDE_initial_investment_calculation_l3277_327787


namespace NUMINAMATH_CALUDE_book_loss_percentage_l3277_327785

/-- If the cost price of 8 books equals the selling price of 16 books, then the loss percentage is 50% -/
theorem book_loss_percentage (C S : ℝ) (h : 8 * C = 16 * S) : (C - S) / C * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_loss_percentage_l3277_327785


namespace NUMINAMATH_CALUDE_expression_evaluation_l3277_327771

theorem expression_evaluation : 5 * 402 + 4 * 402 + 3 * 402 + 401 = 5225 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3277_327771


namespace NUMINAMATH_CALUDE_ladder_angle_approx_l3277_327753

/-- Given a right triangle with hypotenuse 19 meters and adjacent side 9.493063650744542 meters,
    the angle between the hypotenuse and the adjacent side is approximately 60 degrees. -/
theorem ladder_angle_approx (hypotenuse : ℝ) (adjacent : ℝ) (angle : ℝ) 
    (h1 : hypotenuse = 19)
    (h2 : adjacent = 9.493063650744542)
    (h3 : angle = Real.arccos (adjacent / hypotenuse)) :
    ∃ ε > 0, |angle - 60 * π / 180| < ε :=
  sorry

end NUMINAMATH_CALUDE_ladder_angle_approx_l3277_327753


namespace NUMINAMATH_CALUDE_train_travel_time_l3277_327773

/-- Calculates the actual travel time in hours given the total travel time and break time in minutes. -/
def actualTravelTimeInHours (totalTravelTime breakTime : ℕ) : ℚ :=
  (totalTravelTime - breakTime) / 60

/-- Theorem stating that given a total travel time of 270 minutes with a 30-minute break,
    the actual travel time is 4 hours. -/
theorem train_travel_time :
  actualTravelTimeInHours 270 30 = 4 := by sorry

end NUMINAMATH_CALUDE_train_travel_time_l3277_327773


namespace NUMINAMATH_CALUDE_second_order_size_l3277_327766

/-- Proves that given the specified production rates and average output, the second order contains 60 cogs. -/
theorem second_order_size
  (initial_rate : ℝ)
  (initial_order : ℝ)
  (second_rate : ℝ)
  (average_output : ℝ)
  (h1 : initial_rate = 36)
  (h2 : initial_order = 60)
  (h3 : second_rate = 60)
  (h4 : average_output = 45) :
  ∃ (second_order : ℝ),
    (initial_order + second_order) / ((initial_order / initial_rate) + (second_order / second_rate)) = average_output ∧
    second_order = 60 :=
by sorry

end NUMINAMATH_CALUDE_second_order_size_l3277_327766


namespace NUMINAMATH_CALUDE_jack_stairs_problem_l3277_327790

/-- Represents the number of steps in each flight of stairs -/
def steps_per_flight : ℕ := 12

/-- Represents the height of each step in inches -/
def step_height : ℕ := 8

/-- Represents the number of flights Jack went down -/
def flights_down : ℕ := 6

/-- Represents how much further down Jack ended up, in feet -/
def final_position : ℕ := 24

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the height of one flight of stairs in inches -/
def flight_height : ℕ := steps_per_flight * step_height

/-- Represents the number of flights Jack went up initially -/
def flights_up : ℕ := 9

theorem jack_stairs_problem :
  flights_up * flight_height = 
  flights_down * flight_height + feet_to_inches final_position :=
by sorry

end NUMINAMATH_CALUDE_jack_stairs_problem_l3277_327790


namespace NUMINAMATH_CALUDE_inequality_implies_lower_bound_l3277_327742

theorem inequality_implies_lower_bound (m : ℝ) :
  (m > 0) →
  (∀ x : ℝ, x ∈ Set.Ioc 0 1 → |m * x^3 - Real.log x| ≥ 1) →
  m ≥ (1/3) * Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_lower_bound_l3277_327742


namespace NUMINAMATH_CALUDE_baker_cake_difference_l3277_327703

/-- Given Baker's cake inventory and transactions, prove the difference between sold and bought cakes. -/
theorem baker_cake_difference (initial_cakes bought_cakes sold_cakes : ℚ) 
  (h1 : initial_cakes = 8.5)
  (h2 : bought_cakes = 139.25)
  (h3 : sold_cakes = 145.75) :
  sold_cakes - bought_cakes = 6.5 := by
  sorry

#eval (145.75 : ℚ) - (139.25 : ℚ)

end NUMINAMATH_CALUDE_baker_cake_difference_l3277_327703


namespace NUMINAMATH_CALUDE_modified_triangular_array_100th_row_sum_l3277_327774

/-- Sum of numbers in the nth row of the modified triangular array -/
def row_sum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * row_sum (n - 1) + 2

theorem modified_triangular_array_100th_row_sum :
  row_sum 100 = 2^100 - 2 :=
sorry

end NUMINAMATH_CALUDE_modified_triangular_array_100th_row_sum_l3277_327774


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l3277_327784

theorem cow_chicken_problem (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 2 * (cows + chickens) + 16) → cows = 8 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l3277_327784


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l3277_327715

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem
theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l3277_327715


namespace NUMINAMATH_CALUDE_impossible_continuous_coverage_l3277_327751

/-- Represents a runner on the track -/
structure Runner where
  speed : ℕ
  startPosition : ℝ

/-- Represents the circular track with runners -/
structure Track where
  length : ℝ
  spectatorStandLength : ℝ
  runners : List Runner

/-- Checks if a runner is passing the spectator stands at a given time -/
def isPassingStands (runner : Runner) (track : Track) (time : ℝ) : Prop :=
  let position := (runner.startPosition + runner.speed * time) % track.length
  0 ≤ position ∧ position < track.spectatorStandLength

/-- Main theorem statement -/
theorem impossible_continuous_coverage (track : Track) : 
  track.length = 2000 ∧ 
  track.spectatorStandLength = 100 ∧ 
  track.runners.length = 20 ∧
  (∀ i, i ∈ Finset.range 20 → 
    ∃ r ∈ track.runners, r.speed = i + 10) →
  ¬ (∀ t : ℝ, ∃ r ∈ track.runners, isPassingStands r track t) :=
by sorry

end NUMINAMATH_CALUDE_impossible_continuous_coverage_l3277_327751


namespace NUMINAMATH_CALUDE_min_largest_median_l3277_327726

/-- Represents a 5 × 18 rectangle filled with numbers from 1 to 90 -/
def Rectangle := Fin 5 → Fin 18 → Fin 90

/-- The median of a column in the rectangle -/
def columnMedian (rect : Rectangle) (col : Fin 18) : Fin 90 :=
  sorry

/-- The largest median among all columns -/
def largestMedian (rect : Rectangle) : Fin 90 :=
  sorry

/-- Theorem stating the minimum possible value for the largest median -/
theorem min_largest_median :
  ∃ (rect : Rectangle), largestMedian rect = 54 ∧
  ∀ (rect' : Rectangle), largestMedian rect' ≥ 54 :=
sorry

end NUMINAMATH_CALUDE_min_largest_median_l3277_327726


namespace NUMINAMATH_CALUDE_inverse_of_A_squared_l3277_327700

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = ![![-4, 1], ![0, 2]] →
  (A^2)⁻¹ = ![![16, -2], ![0, 4]] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_A_squared_l3277_327700


namespace NUMINAMATH_CALUDE_cost_of_six_lollipops_l3277_327795

/-- The cost of 6 giant lollipops with discounts and promotions -/
theorem cost_of_six_lollipops (regular_price : ℝ) (discount_rate : ℝ) : 
  regular_price = 2.4 / 2 →
  discount_rate = 0.1 →
  6 * regular_price * (1 - discount_rate) = 6.48 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_six_lollipops_l3277_327795


namespace NUMINAMATH_CALUDE_sum_of_roots_tangent_equation_l3277_327743

theorem sum_of_roots_tangent_equation : 
  ∃ (x₁ x₂ : ℝ), 
    0 < x₁ ∧ x₁ < π ∧
    0 < x₂ ∧ x₂ < π ∧
    (Real.tan x₁)^2 - 5 * Real.tan x₁ + 6 = 0 ∧
    (Real.tan x₂)^2 - 5 * Real.tan x₂ + 6 = 0 ∧
    x₁ + x₂ = Real.arctan 3 + Real.arctan 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_tangent_equation_l3277_327743


namespace NUMINAMATH_CALUDE_h_sqrt_two_equals_zero_min_a_plus_b_h_not_arbitrary_quadratic_l3277_327796

-- Define the functions f, g, l, and h
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x
def g (b : ℝ) (x : ℝ) : ℝ := x + b
def l (x : ℝ) : ℝ := 2*x^2 + 3*x - 1

-- Define the property of h being generated by f and g
def is_generated_by_f_and_g (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ (m n : ℝ), ∀ x, h x = m * (f a x) + n * (g b x)

-- Define h as a quadratic function
def h (a b m n : ℝ) (x : ℝ) : ℝ := m * (f a x) + n * (g b x)

-- Theorem 1
theorem h_sqrt_two_equals_zero (a b m n : ℝ) :
  a = 1 → b = 2 → (∀ x, h a b m n x = h a b m n (-x)) → 
  h a b m n (Real.sqrt 2) = 0 := by sorry

-- Theorem 2
theorem min_a_plus_b (a b m n : ℝ) :
  b > 0 → is_generated_by_f_and_g (h a b m n) a b →
  (∃ m' n', ∀ x, h a b m n x = m' * g b x + n' * l x) →
  a + b ≥ 3/2 + Real.sqrt 2 := by sorry

-- Theorem 3
theorem h_not_arbitrary_quadratic (a b : ℝ) :
  ¬ ∀ (p q r : ℝ), ∃ (m n : ℝ), ∀ x, h a b m n x = p*x^2 + q*x + r := by sorry

end NUMINAMATH_CALUDE_h_sqrt_two_equals_zero_min_a_plus_b_h_not_arbitrary_quadratic_l3277_327796


namespace NUMINAMATH_CALUDE_range_of_w_l3277_327725

theorem range_of_w (x y : ℝ) (h : 2*x^2 + 4*x*y + 2*y^2 + x^2*y^2 = 9) :
  let w := 2*Real.sqrt 2*(x + y) + x*y
  ∃ (a b : ℝ), a = -3*Real.sqrt 5 ∧ b = Real.sqrt 5 ∧ 
    (∀ w', w' = w → a ≤ w' ∧ w' ≤ b) ∧
    (∃ w₁ w₂, w₁ = w ∧ w₂ = w ∧ w₁ = a ∧ w₂ = b) :=
by sorry


end NUMINAMATH_CALUDE_range_of_w_l3277_327725


namespace NUMINAMATH_CALUDE_one_sixths_in_eleven_thirds_l3277_327759

theorem one_sixths_in_eleven_thirds : (11 / 3) / (1 / 6) = 22 := by
  sorry

end NUMINAMATH_CALUDE_one_sixths_in_eleven_thirds_l3277_327759


namespace NUMINAMATH_CALUDE_base_number_proof_l3277_327730

theorem base_number_proof (w : ℕ) (x : ℝ) (h1 : w = 12) (h2 : 2^(2*w) = x^(w-4)) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l3277_327730


namespace NUMINAMATH_CALUDE_train_travel_time_l3277_327745

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  hlt24 : hours < 24
  mlt60 : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  totalMinutes2 - totalMinutes1

/-- The train travel time theorem -/
theorem train_travel_time :
  let departureTime : Time := ⟨7, 5, by norm_num, by norm_num⟩
  let arrivalTime : Time := ⟨7, 59, by norm_num, by norm_num⟩
  timeDifference departureTime arrivalTime = 54 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_time_l3277_327745


namespace NUMINAMATH_CALUDE_square_eq_product_sum_seven_l3277_327755

theorem square_eq_product_sum_seven (a b : ℕ) : 
  a * a = b * (b + 7) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
sorry

end NUMINAMATH_CALUDE_square_eq_product_sum_seven_l3277_327755


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3277_327747

def S : Set ℝ := {1, 2, 3, 5, 10}

theorem max_value_of_expression (x y : ℝ) (hx : x ∈ S) (hy : y ∈ S) :
  (x / y + y / x) ≤ 10.1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3277_327747


namespace NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_abs_a_positive_l3277_327798

theorem a_positive_sufficient_not_necessary_for_abs_a_positive :
  (∃ a : ℝ, a > 0 → abs a > 0) ∧ 
  (∃ a : ℝ, abs a > 0 ∧ ¬(a > 0)) :=
by sorry

end NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_abs_a_positive_l3277_327798


namespace NUMINAMATH_CALUDE_hugo_roll_four_given_win_l3277_327734

-- Define the number of players
def num_players : ℕ := 5

-- Define the number of sides on the die
def die_sides : ℕ := 6

-- Define Hugo's winning probability
def hugo_win_prob : ℚ := 1 / num_players

-- Define the probability of rolling a 4
def roll_four_prob : ℚ := 1 / die_sides

-- Define the probability of Hugo winning given his first roll was 4
def hugo_win_given_four : ℚ := 145 / 1296

-- Theorem statement
theorem hugo_roll_four_given_win (
  num_players : ℕ) (die_sides : ℕ) (hugo_win_prob : ℚ) (roll_four_prob : ℚ) (hugo_win_given_four : ℚ) :
  num_players = 5 →
  die_sides = 6 →
  hugo_win_prob = 1 / num_players →
  roll_four_prob = 1 / die_sides →
  hugo_win_given_four = 145 / 1296 →
  (roll_four_prob * hugo_win_given_four) / hugo_win_prob = 145 / 1552 :=
by sorry

end NUMINAMATH_CALUDE_hugo_roll_four_given_win_l3277_327734


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1023_l3277_327799

theorem largest_prime_factor_of_1023 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1023 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1023 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1023_l3277_327799


namespace NUMINAMATH_CALUDE_m_in_open_interval_l3277_327765

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem m_in_open_interval
  (f : ℝ → ℝ)
  (h_decreasing : monotonically_decreasing f)
  (h_inequality : f (m^2) > f m)
  : m ∈ Set.Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_m_in_open_interval_l3277_327765


namespace NUMINAMATH_CALUDE_find_y_value_l3277_327756

theorem find_y_value (x y : ℝ) (h1 : x * y = 4) (h2 : x / y = 81) (h3 : x > 0) (h4 : y > 0) :
  y = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_find_y_value_l3277_327756


namespace NUMINAMATH_CALUDE_max_value_ratio_l3277_327736

theorem max_value_ratio (x y k : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (h : x = k * y) :
  ∃ (M : ℝ), M = 1 ∧ ∀ x y k, x ≠ 0 → y ≠ 0 → k ≠ 0 → x = k * y →
    |x + y| / (|x| + |y|) ≤ M ∧ ∃ x y k, x ≠ 0 ∧ y ≠ 0 ∧ k ≠ 0 ∧ x = k * y ∧ |x + y| / (|x| + |y|) = M :=
sorry

end NUMINAMATH_CALUDE_max_value_ratio_l3277_327736


namespace NUMINAMATH_CALUDE_linear_system_fraction_sum_l3277_327786

theorem linear_system_fraction_sum (a b c x y z : ℝ) 
  (eq1 : 11 * x + b * y + c * z = 0)
  (eq2 : a * x + 24 * y + c * z = 0)
  (eq3 : a * x + b * y + 41 * z = 0)
  (ha : a ≠ 11)
  (hx : x ≠ 0) :
  a / (a - 11) + b / (b - 24) + c / (c - 41) = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_fraction_sum_l3277_327786


namespace NUMINAMATH_CALUDE_unread_books_l3277_327704

theorem unread_books (total : ℕ) (read : ℕ) (h1 : total = 21) (h2 : read = 13) :
  total - read = 8 := by
  sorry

end NUMINAMATH_CALUDE_unread_books_l3277_327704


namespace NUMINAMATH_CALUDE_rap_song_requests_l3277_327718

/-- Represents the number of song requests for different genres in a night --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of rap song requests given the conditions --/
theorem rap_song_requests (r : SongRequests) : r.rap = 2 :=
  by
  have h1 : r.total = 30 := by sorry
  have h2 : r.electropop = r.total / 2 := by sorry
  have h3 : r.dance = r.electropop / 3 := by sorry
  have h4 : r.rock = 5 := by sorry
  have h5 : r.oldies = r.rock - 3 := by sorry
  have h6 : r.dj_choice = r.oldies / 2 := by sorry
  have h7 : r.total = r.electropop + r.dance + r.rock + r.oldies + r.dj_choice + r.rap := by sorry
  sorry

#check rap_song_requests

end NUMINAMATH_CALUDE_rap_song_requests_l3277_327718


namespace NUMINAMATH_CALUDE_ellipse_chord_y_diff_l3277_327757

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Theorem: For the given ellipse, if a chord AB passes through the left focus and the
    inscribed circle of triangle ABF₂ has circumference π, then |y₁ - y₂| = 5/4 -/
theorem ellipse_chord_y_diff (e : Ellipse) (A B : Point) (F₁ F₂ : Point) : 
  e.a = 5 → 
  e.b = 4 → 
  F₁.x = -3 → 
  F₁.y = 0 → 
  F₂.x = 3 → 
  F₂.y = 0 → 
  (A.x - F₁.x) * (B.y - F₁.y) = (B.x - F₁.x) * (A.y - F₁.y) →  -- AB passes through F₁
  2 * π * (A.x * (B.y - F₂.y) + B.x * (F₂.y - A.y) + F₂.x * (A.y - B.y)) / 
    (A.x * (B.y - F₂.y) + B.x * (F₂.y - A.y) + F₂.x * (A.y - B.y) + 
     (A.x - F₂.x) * (B.y - F₂.y) - (B.x - F₂.x) * (A.y - F₂.y)) = π →  -- Inscribed circle circumference
  |A.y - B.y| = 5/4 := by
sorry


end NUMINAMATH_CALUDE_ellipse_chord_y_diff_l3277_327757


namespace NUMINAMATH_CALUDE_blocks_given_by_theresa_l3277_327717

theorem blocks_given_by_theresa (initial_blocks final_blocks : ℕ) 
  (h1 : initial_blocks = 4)
  (h2 : final_blocks = 83) :
  final_blocks - initial_blocks = 79 := by
  sorry

end NUMINAMATH_CALUDE_blocks_given_by_theresa_l3277_327717


namespace NUMINAMATH_CALUDE_parallelogram_height_l3277_327769

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area base height : ℝ) : 
  area = base * height → area = 320 → base = 20 → height = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3277_327769


namespace NUMINAMATH_CALUDE_geometry_theorem_l3277_327778

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_theorem 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (parallel m α ∧ perpendicular n α → perpendicular_lines m n) ∧
  (perpendicular m α ∧ parallel m β → perpendicular_planes α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l3277_327778


namespace NUMINAMATH_CALUDE_price_reduction_equation_l3277_327793

/-- Represents the price reduction scenario for a medicine -/
structure PriceReduction where
  original_price : ℝ
  final_price : ℝ
  reduction_percentage : ℝ

/-- Theorem stating the relationship between original price, final price, and reduction percentage -/
theorem price_reduction_equation (pr : PriceReduction) 
  (h1 : pr.original_price = 25)
  (h2 : pr.final_price = 16)
  : pr.original_price * (1 - pr.reduction_percentage)^2 = pr.final_price := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l3277_327793


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3277_327760

/-- Given two natural numbers m and n, returns true if their product ends with the digit d -/
def product_ends_with (m n d : ℕ) : Prop :=
  (m * n) % 10 = d

/-- Given a natural number x, returns true if its units digit is d -/
def units_digit (x d : ℕ) : Prop :=
  x % 10 = d

theorem units_digit_of_n (m n : ℕ) :
  product_ends_with m n 4 →
  units_digit m 8 →
  units_digit n 3 := by
  sorry

#check units_digit_of_n

end NUMINAMATH_CALUDE_units_digit_of_n_l3277_327760


namespace NUMINAMATH_CALUDE_sin_b_in_arithmetic_sequence_triangle_l3277_327776

/-- In a triangle ABC where the interior angles form an arithmetic sequence, sin B = √3/2 -/
theorem sin_b_in_arithmetic_sequence_triangle (A B C : Real) : 
  A + B + C = Real.pi →  -- Sum of angles in radians
  A + C = 2 * B →        -- Arithmetic sequence property
  Real.sin B = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_b_in_arithmetic_sequence_triangle_l3277_327776


namespace NUMINAMATH_CALUDE_faster_speed_calculation_l3277_327788

theorem faster_speed_calculation (total_distance : ℝ) (initial_speed : ℝ) (initial_time : ℝ) 
  (second_time : ℝ) (remaining_distance : ℝ) 
  (h1 : total_distance = 600)
  (h2 : initial_speed = 50)
  (h3 : initial_time = 3)
  (h4 : second_time = 4)
  (h5 : remaining_distance = 130) : 
  ∃ faster_speed : ℝ, faster_speed = 80 ∧ 
  total_distance = initial_speed * initial_time + faster_speed * second_time + remaining_distance :=
by
  sorry


end NUMINAMATH_CALUDE_faster_speed_calculation_l3277_327788


namespace NUMINAMATH_CALUDE_ages_solution_l3277_327741

/-- Represents the current ages of Justin, Angelina, and Larry --/
structure Ages where
  justin : ℝ
  angelina : ℝ
  larry : ℝ

/-- The conditions given in the problem --/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.angelina = ages.justin + 4 ∧
  ages.angelina + 5 = 40 ∧
  ages.larry = ages.justin * 1.5

/-- The theorem to be proved --/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧ ages.justin = 31 ∧ ages.larry = 46.5 :=
sorry

end NUMINAMATH_CALUDE_ages_solution_l3277_327741


namespace NUMINAMATH_CALUDE_sum_inequality_l3277_327770

theorem sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_sum : 1 / (a^3 + 1) + 1 / (b^3 + 1) + 1 / (c^3 + 1) + 1 / (d^3 + 1) = 2) :
  (1 - a) / (a^2 - a + 1) + (1 - b) / (b^2 - b + 1) + 
  (1 - c) / (c^2 - c + 1) + (1 - d) / (d^2 - d + 1) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3277_327770


namespace NUMINAMATH_CALUDE_collinear_points_implies_b_value_l3277_327762

/-- Three points (x₁, y₁), (x₂, y₂), and (x₃, y₃) are collinear if and only if
    (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁) -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- If the points (5, -3), (2b + 4, 5), and (-3b + 6, -1) are collinear, then b = 5/14 -/
theorem collinear_points_implies_b_value :
  ∀ b : ℝ, collinear 5 (-3) (2*b + 4) 5 (-3*b + 6) (-1) → b = 5/14 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_implies_b_value_l3277_327762


namespace NUMINAMATH_CALUDE_computer_table_price_l3277_327779

/-- The selling price of an item given its cost price and markup percentage -/
def selling_price (cost : ℕ) (markup_percent : ℕ) : ℕ :=
  cost + cost * markup_percent / 100

/-- Theorem stating that for a computer table with cost price 3000 and 20% markup, 
    the selling price is 3600 -/
theorem computer_table_price : selling_price 3000 20 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_price_l3277_327779


namespace NUMINAMATH_CALUDE_fraction_simplification_l3277_327768

theorem fraction_simplification (b x : ℝ) (h : b^2 + x^2 ≠ 0) :
  (Real.sqrt (b^2 + x^2) - (x^2 - b^2) / Real.sqrt (b^2 + x^2)) / (2 * (b^2 + x^2)^2) =
  b^2 / (b^2 + x^2)^(5/2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3277_327768


namespace NUMINAMATH_CALUDE_gcd_38_23_is_1_l3277_327783

/-- The method of continued subtraction for calculating GCD -/
def continuedSubtractionGCD (a b : ℕ) : ℕ :=
  if a = 0 then b
  else if b = 0 then a
  else if a ≥ b then continuedSubtractionGCD (a - b) b
  else continuedSubtractionGCD a (b - a)

/-- Theorem: The GCD of 38 and 23 is 1 using the method of continued subtraction -/
theorem gcd_38_23_is_1 : continuedSubtractionGCD 38 23 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_38_23_is_1_l3277_327783


namespace NUMINAMATH_CALUDE_mrs_anderson_pet_food_l3277_327719

/-- Calculates the total ounces of pet food bought by Mrs. Anderson -/
def total_pet_food_ounces (cat_food_bags : ℕ) (cat_food_weight : ℕ) 
  (dog_food_bags : ℕ) (dog_food_extra_weight : ℕ) (ounces_per_pound : ℕ) : ℕ :=
  let total_cat_food := cat_food_bags * cat_food_weight
  let total_dog_food := dog_food_bags * (cat_food_weight + dog_food_extra_weight)
  let total_pounds := total_cat_food + total_dog_food
  total_pounds * ounces_per_pound

/-- Theorem stating that Mrs. Anderson bought 256 ounces of pet food -/
theorem mrs_anderson_pet_food : 
  total_pet_food_ounces 2 3 2 2 16 = 256 := by
  sorry

end NUMINAMATH_CALUDE_mrs_anderson_pet_food_l3277_327719


namespace NUMINAMATH_CALUDE_largest_power_2024_divides_factorial_l3277_327748

def largest_power_dividing_factorial (n : ℕ) : ℕ :=
  min (sum_floor_div n 11) (sum_floor_div n 23)
where
  sum_floor_div (n p : ℕ) : ℕ :=
    (n / p) + (n / (p * p))

theorem largest_power_2024_divides_factorial :
  largest_power_dividing_factorial 2024 = 91 := by
sorry

#eval largest_power_dividing_factorial 2024

end NUMINAMATH_CALUDE_largest_power_2024_divides_factorial_l3277_327748


namespace NUMINAMATH_CALUDE_expression_value_l3277_327716

theorem expression_value : (2018 - 18 + 20) / 2 = 1010 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3277_327716


namespace NUMINAMATH_CALUDE_ticket_sales_total_l3277_327732

/-- Calculates the total amount collected from ticket sales given the ticket prices, total tickets sold, and number of children attending. -/
def total_amount_collected (child_price adult_price total_tickets children_count : ℕ) : ℕ :=
  let adult_count := total_tickets - children_count
  child_price * children_count + adult_price * adult_count

/-- Theorem stating that the total amount collected from ticket sales is $1875 given the specified conditions. -/
theorem ticket_sales_total :
  total_amount_collected 6 9 225 50 = 1875 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_total_l3277_327732


namespace NUMINAMATH_CALUDE_alien_species_count_l3277_327705

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def alienSpeciesBase7 : List Nat := [5, 1, 2]

/-- The theorem stating that the base 7 number 215₇ is equal to 110 in base 10 --/
theorem alien_species_count : base7ToBase10 alienSpeciesBase7 = 110 := by
  sorry

end NUMINAMATH_CALUDE_alien_species_count_l3277_327705


namespace NUMINAMATH_CALUDE_jennifers_spending_l3277_327794

theorem jennifers_spending (total : ℚ) (sandwich_frac : ℚ) (museum_frac : ℚ) (leftover : ℚ)
  (h1 : total = 150)
  (h2 : sandwich_frac = 1/5)
  (h3 : museum_frac = 1/6)
  (h4 : leftover = 20) :
  let spent_on_sandwich := total * sandwich_frac
  let spent_on_museum := total * museum_frac
  let total_spent := total - leftover
  let spent_on_book := total_spent - spent_on_sandwich - spent_on_museum
  spent_on_book / total = 1/2 := by
sorry

end NUMINAMATH_CALUDE_jennifers_spending_l3277_327794


namespace NUMINAMATH_CALUDE_incorrect_statement_l3277_327712

def U : Finset Nat := {1, 2, 3, 4}
def M : Finset Nat := {1, 2}
def N : Finset Nat := {2, 4}

theorem incorrect_statement : M ∩ (U \ N) ≠ {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_incorrect_statement_l3277_327712


namespace NUMINAMATH_CALUDE_subtraction_difference_l3277_327752

theorem subtraction_difference (original : ℝ) (percentage : ℝ) (flat_amount : ℝ) : 
  original = 200 → percentage = 25 → flat_amount = 25 →
  (original - flat_amount) - (original - percentage / 100 * original) = 25 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_difference_l3277_327752


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3277_327780

theorem fraction_meaningful (m : ℝ) : 
  (∃ (x : ℝ), x = 3 / (m - 4)) ↔ m ≠ 4 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3277_327780


namespace NUMINAMATH_CALUDE_xyz_sum_l3277_327707

theorem xyz_sum (x y z : ℕ+) (h : (x + y * Complex.I)^2 - 46 * Complex.I = z) :
  x + y + z = 552 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_l3277_327707


namespace NUMINAMATH_CALUDE_sector_central_angle_l3277_327739

/-- Given a circular sector with arc length 2 and area 4, prove that its central angle is 1/2 radians. -/
theorem sector_central_angle (arc_length : ℝ) (area : ℝ) (h1 : arc_length = 2) (h2 : area = 4) :
  let r := 2 * area / arc_length
  (arc_length / r) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3277_327739


namespace NUMINAMATH_CALUDE_wendy_second_level_treasures_l3277_327767

def points_per_treasure : ℕ := 5
def treasures_first_level : ℕ := 4
def total_score : ℕ := 35

theorem wendy_second_level_treasures :
  (total_score - points_per_treasure * treasures_first_level) / points_per_treasure = 3 := by
  sorry

end NUMINAMATH_CALUDE_wendy_second_level_treasures_l3277_327767


namespace NUMINAMATH_CALUDE_even_quadratic_implies_m_zero_l3277_327702

/-- A quadratic function f(x) = (m-1)x^2 - 2mx + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + 3

/-- Definition of an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_quadratic_implies_m_zero (m : ℝ) :
  is_even (f m) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_implies_m_zero_l3277_327702


namespace NUMINAMATH_CALUDE_first_month_sale_is_7435_l3277_327775

/-- Calculates the sale in the first month given the sales for months 2-6 and the average sale --/
def first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale2 + sale3 + sale4 + sale5 + sale6)

/-- Proves that the first month's sale is 7435 given the problem conditions --/
theorem first_month_sale_is_7435 :
  first_month_sale 7920 7855 8230 7560 6000 7500 = 7435 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_is_7435_l3277_327775

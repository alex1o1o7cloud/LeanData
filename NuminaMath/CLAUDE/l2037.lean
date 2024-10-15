import Mathlib

namespace NUMINAMATH_CALUDE_valid_three_digit_count_correct_l2037_203796

/-- The count of valid three-digit numbers -/
def valid_three_digit_count : ℕ := 819

/-- The total count of three-digit numbers -/
def total_three_digit_count : ℕ := 900

/-- The count of invalid three-digit numbers where the hundreds and units digits
    are the same but the tens digit is different -/
def invalid_three_digit_count : ℕ := 81

/-- Theorem stating that the count of valid three-digit numbers is correct -/
theorem valid_three_digit_count_correct :
  valid_three_digit_count = total_three_digit_count - invalid_three_digit_count :=
by sorry

end NUMINAMATH_CALUDE_valid_three_digit_count_correct_l2037_203796


namespace NUMINAMATH_CALUDE_red_candy_count_l2037_203704

theorem red_candy_count (total : ℕ) (blue : ℕ) (h1 : total = 3409) (h2 : blue = 3264) :
  total - blue = 145 := by
  sorry

end NUMINAMATH_CALUDE_red_candy_count_l2037_203704


namespace NUMINAMATH_CALUDE_seventh_observation_l2037_203746

theorem seventh_observation (n : ℕ) (initial_avg : ℚ) (new_avg : ℚ) : 
  n = 6 → 
  initial_avg = 12 → 
  new_avg = 11 → 
  (n * initial_avg + (n + 1) * new_avg - n * initial_avg) / (n + 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_seventh_observation_l2037_203746


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l2037_203723

theorem smallest_next_divisor_after_221 (m : ℕ) (h1 : m ≥ 1000 ∧ m ≤ 9999) 
  (h2 : Even m) (h3 : m % 221 = 0) :
  ∃ (d : ℕ), d > 221 ∧ m % d = 0 ∧ d ≥ 238 ∧ 
  ∀ (d' : ℕ), d' > 221 ∧ m % d' = 0 → d' ≥ 238 :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l2037_203723


namespace NUMINAMATH_CALUDE_difference_of_sums_l2037_203738

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_multiple_of_three (n : ℕ) : Prop := ∃ k, n = 3 * k

def smallest_two_digit_multiple_of_three : ℕ := 12

def largest_two_digit_multiple_of_three : ℕ := 99

def smallest_two_digit_non_multiple_of_three : ℕ := 10

def largest_two_digit_non_multiple_of_three : ℕ := 98

theorem difference_of_sums : 
  (largest_two_digit_multiple_of_three + smallest_two_digit_multiple_of_three) -
  (largest_two_digit_non_multiple_of_three + smallest_two_digit_non_multiple_of_three) = 3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_sums_l2037_203738


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2037_203791

theorem sqrt_equation_solution (x : ℝ) :
  (x > 2) → (Real.sqrt (7 * x) / Real.sqrt (4 * (x - 2)) = 3) → (x = 72 / 29) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2037_203791


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l2037_203706

/-- If the equation x²/(4-m) - y²/(2+m) = 1 represents a hyperbola, 
    then the range of m is (-2, 4) -/
theorem hyperbola_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (4 - m) - y^2 / (2 + m) = 1) → 
  -2 < m ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l2037_203706


namespace NUMINAMATH_CALUDE_knights_selection_l2037_203799

/-- The number of ways to select k non-adjacent elements from n elements in a circular arrangement -/
def circularNonAdjacentSelection (n k : ℕ) : ℕ :=
  Nat.choose (n - k + 1) k - Nat.choose (n - k - 1) (k - 2)

/-- The problem statement -/
theorem knights_selection :
  circularNonAdjacentSelection 50 15 = 463991880 := by
  sorry

end NUMINAMATH_CALUDE_knights_selection_l2037_203799


namespace NUMINAMATH_CALUDE_drinks_calculation_l2037_203777

/-- Given a number of pitchers and the number of glasses each pitcher can fill,
    calculate the total number of glasses that can be filled. -/
def total_glasses (num_pitchers : ℕ) (glasses_per_pitcher : ℕ) : ℕ :=
  num_pitchers * glasses_per_pitcher

/-- Theorem: With 9 pitchers and 6 glasses per pitcher, the total number of glasses is 54. -/
theorem drinks_calculation :
  total_glasses 9 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_drinks_calculation_l2037_203777


namespace NUMINAMATH_CALUDE_origami_distribution_l2037_203794

theorem origami_distribution (total_papers : ℝ) (num_cousins : ℝ) (papers_per_cousin : ℝ) : 
  total_papers = 48.0 →
  num_cousins = 6.0 →
  total_papers = num_cousins * papers_per_cousin →
  papers_per_cousin = 8.0 := by
  sorry

end NUMINAMATH_CALUDE_origami_distribution_l2037_203794


namespace NUMINAMATH_CALUDE_inequality_proof_l2037_203737

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  Real.sqrt (b^2 - a*c) > Real.sqrt 3 * a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2037_203737


namespace NUMINAMATH_CALUDE_amount_paid_is_fifty_l2037_203776

/-- Represents the purchase and change scenario --/
structure Purchase where
  book_cost : ℕ
  pen_cost : ℕ
  ruler_cost : ℕ
  change_received : ℕ

/-- Calculates the total cost of items --/
def total_cost (p : Purchase) : ℕ :=
  p.book_cost + p.pen_cost + p.ruler_cost

/-- Calculates the amount paid --/
def amount_paid (p : Purchase) : ℕ :=
  total_cost p + p.change_received

/-- Theorem stating that the amount paid is $50 --/
theorem amount_paid_is_fifty (p : Purchase) 
  (h1 : p.book_cost = 25)
  (h2 : p.pen_cost = 4)
  (h3 : p.ruler_cost = 1)
  (h4 : p.change_received = 20) :
  amount_paid p = 50 := by
  sorry

end NUMINAMATH_CALUDE_amount_paid_is_fifty_l2037_203776


namespace NUMINAMATH_CALUDE_reciprocal_expression_l2037_203720

theorem reciprocal_expression (m n : ℝ) (h : m * n = 1) : m * n^2 - (n - 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_expression_l2037_203720


namespace NUMINAMATH_CALUDE_exist_decreasing_gcd_sequence_l2037_203705

theorem exist_decreasing_gcd_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j : Fin 100, i < j → a i < a j) ∧ 
    (∀ i : Fin 99, Nat.gcd (a i) (a (i + 1)) > Nat.gcd (a (i + 1)) (a (i + 2))) :=
by sorry

end NUMINAMATH_CALUDE_exist_decreasing_gcd_sequence_l2037_203705


namespace NUMINAMATH_CALUDE_equality_of_fractions_l2037_203744

theorem equality_of_fractions (M N : ℚ) : 
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + N = 183 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l2037_203744


namespace NUMINAMATH_CALUDE_arrangement_theorems_l2037_203733

/-- The number of boys in the group -/
def num_boys : ℕ := 3

/-- The number of girls in the group -/
def num_girls : ℕ := 4

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of arrangements with boys together -/
def arrangements_boys_together : ℕ := 720

/-- The number of arrangements with alternating genders -/
def arrangements_alternating : ℕ := 144

/-- The number of arrangements with person A left of person B -/
def arrangements_A_left_of_B : ℕ := 2520

theorem arrangement_theorems :
  (arrangements_boys_together = 720) ∧
  (arrangements_alternating = 144) ∧
  (arrangements_A_left_of_B = 2520) := by sorry

end NUMINAMATH_CALUDE_arrangement_theorems_l2037_203733


namespace NUMINAMATH_CALUDE_inequality_theorem_l2037_203740

/-- A function f: ℝ → ℝ satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, Differentiable ℝ f ∧ (x - 1) * (deriv (deriv f) x) < 0

/-- Theorem stating the inequality for functions satisfying the condition -/
theorem inequality_theorem (f : ℝ → ℝ) (h : SatisfiesCondition f) :
  f 0 + f 2 < 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2037_203740


namespace NUMINAMATH_CALUDE_property_sale_gain_l2037_203736

/-- Represents the sale of two properties with given selling prices and percentage changes --/
def PropertySale (house_price store_price : ℝ) (house_loss store_gain : ℝ) : Prop :=
  ∃ (house_cost store_cost : ℝ),
    house_price = house_cost * (1 - house_loss) ∧
    store_price = store_cost * (1 + store_gain) ∧
    house_price + store_price - (house_cost + store_cost) = 1000

/-- Theorem stating that the given property sale results in a $1000 gain --/
theorem property_sale_gain :
  PropertySale 15000 18000 0.25 0.50 := by
  sorry

#check property_sale_gain

end NUMINAMATH_CALUDE_property_sale_gain_l2037_203736


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2037_203702

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2037_203702


namespace NUMINAMATH_CALUDE_increasing_sequence_bound_l2037_203729

theorem increasing_sequence_bound (a : ℝ) :
  (∀ n : ℕ+, (n.val - a)^2 < ((n + 1).val - a)^2) →
  a < 3/2 := by
sorry

end NUMINAMATH_CALUDE_increasing_sequence_bound_l2037_203729


namespace NUMINAMATH_CALUDE_repeating_37_equals_fraction_l2037_203714

/-- The repeating decimal 0.373737... -/
def repeating_37 : ℚ := 37 / 99

theorem repeating_37_equals_fraction : 
  repeating_37 = 37 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_37_equals_fraction_l2037_203714


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2037_203767

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l2037_203767


namespace NUMINAMATH_CALUDE_no_extra_savings_when_combined_l2037_203707

def book_price : ℕ := 120
def alice_books : ℕ := 10
def bob_books : ℕ := 15

def calculate_cost (num_books : ℕ) : ℕ :=
  let free_books := (num_books / 5) * 2
  let paid_books := num_books - free_books
  paid_books * book_price

def calculate_savings (num_books : ℕ) : ℕ :=
  num_books * book_price - calculate_cost num_books

theorem no_extra_savings_when_combined :
  calculate_savings alice_books + calculate_savings bob_books =
  calculate_savings (alice_books + bob_books) :=
by sorry

end NUMINAMATH_CALUDE_no_extra_savings_when_combined_l2037_203707


namespace NUMINAMATH_CALUDE_merchant_transaction_loss_l2037_203785

theorem merchant_transaction_loss : 
  ∀ (cost_profit cost_loss : ℝ),
  cost_profit * 1.15 = 1955 →
  cost_loss * 0.85 = 1955 →
  (1955 + 1955) - (cost_profit + cost_loss) = -90 :=
by
  sorry

end NUMINAMATH_CALUDE_merchant_transaction_loss_l2037_203785


namespace NUMINAMATH_CALUDE_bubble_theorem_l2037_203710

/-- Given a hemisphere with radius 4∛2 cm and volume double that of an initial spherical bubble,
    prove the radius of the original bubble and the volume of a new sphere with doubled radius. -/
theorem bubble_theorem (r : ℝ) (h1 : r = 4 * Real.rpow 2 (1/3)) :
  let R := Real.rpow 4 (1/3)
  let V_new := (64/3) * Real.pi * Real.rpow 4 (1/3)
  (2/3) * Real.pi * r^3 = 2 * ((4/3) * Real.pi * R^3) ∧ 
  (4/3) * Real.pi * (2*R)^3 = V_new := by
  sorry

end NUMINAMATH_CALUDE_bubble_theorem_l2037_203710


namespace NUMINAMATH_CALUDE_no_primes_in_range_l2037_203792

theorem no_primes_in_range (n : ℕ) (h : n > 2) :
  ∀ k ∈ Set.Icc (n! + 3) (n! + 2*n), ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l2037_203792


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2037_203761

/-- Represents the outcome of a single coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents a sequence of 8 coin flips -/
def CoinSequence := Vector CoinFlip 8

/-- Checks if a given sequence has exactly one pair of consecutive heads and one pair of consecutive tails -/
def hasExactlyOnePairEach (seq : CoinSequence) : Bool :=
  sorry

/-- The total number of possible 8-flip sequences -/
def totalSequences : Nat := 256

/-- The number of favorable sequences (with exactly one pair each of heads and tails) -/
def favorableSequences : Nat := 18

/-- The probability of getting exactly one pair each of heads and tails in 8 flips -/
def probability : Rat := favorableSequences / totalSequences

theorem coin_flip_probability :
  probability = 9 / 128 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2037_203761


namespace NUMINAMATH_CALUDE_pencils_added_l2037_203798

theorem pencils_added (initial_pencils final_pencils : ℕ) (h1 : initial_pencils = 41) (h2 : final_pencils = 71) :
  final_pencils - initial_pencils = 30 := by
  sorry

end NUMINAMATH_CALUDE_pencils_added_l2037_203798


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l2037_203708

theorem rectangular_prism_width (l h d w : ℝ) : 
  l = 5 → h = 7 → d = 15 → d^2 = l^2 + w^2 + h^2 → w^2 = 151 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l2037_203708


namespace NUMINAMATH_CALUDE_parabola_c_value_l2037_203757

/-- A parabola with equation x = ay^2 + by + c, vertex at (-3, -1), and passing through (-1, 1) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := -3
  vertex_y : ℝ := -1
  point_x : ℝ := -1
  point_y : ℝ := 1
  eq_vertex : -3 = a * (-1)^2 + b * (-1) + c
  eq_point : -1 = a * 1^2 + b * 1 + c

/-- The value of c for the given parabola is -2.5 -/
theorem parabola_c_value (p : Parabola) : p.c = -2.5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2037_203757


namespace NUMINAMATH_CALUDE_round_recurring_decimal_to_thousandth_l2037_203773

/-- The repeating decimal 36.3636... -/
def recurring_decimal : ℚ := 36 + 36 / 99

/-- Rounding a number to the nearest thousandth -/
def round_to_thousandth (x : ℚ) : ℚ := 
  (⌊x * 1000 + 0.5⌋) / 1000

/-- Proof that rounding 36.3636... to the nearest thousandth equals 36.363 -/
theorem round_recurring_decimal_to_thousandth : 
  round_to_thousandth recurring_decimal = 36363 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_round_recurring_decimal_to_thousandth_l2037_203773


namespace NUMINAMATH_CALUDE_equality_condition_l2037_203717

theorem equality_condition (x : ℝ) (hx : x > 0) :
  x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) = 15 ↔ x = 3 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l2037_203717


namespace NUMINAMATH_CALUDE_determinant_max_value_l2037_203701

open Real

theorem determinant_max_value : 
  let det := fun θ : ℝ => 
    Matrix.det !![1, 1, 1; 1, 1 + cos θ, 1; 1 + sin θ, 1, 1]
  ∃ (max_val : ℝ), max_val = 1/2 ∧ ∀ θ, det θ ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_determinant_max_value_l2037_203701


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2037_203778

theorem sufficient_not_necessary : 
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ |x| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2037_203778


namespace NUMINAMATH_CALUDE_three_digit_number_difference_l2037_203743

theorem three_digit_number_difference (X Y : ℕ) : 
  X > Y → 
  X + Y = 999 → 
  X ≥ 100 → 
  X ≤ 999 → 
  Y ≥ 100 → 
  Y ≤ 999 → 
  1000 * X + Y = 6 * (1000 * Y + X) → 
  X - Y = 715 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_difference_l2037_203743


namespace NUMINAMATH_CALUDE_square_root_problem_l2037_203756

-- Define the variables
variable (a b : ℝ)

-- State the theorem
theorem square_root_problem (h1 : a = 9) (h2 : b = 4/9) :
  (∃ (x : ℝ), x^2 = a ∧ (x = 3 ∨ x = -3)) ∧
  (Real.sqrt (a * b) = 2) →
  (a = 9 ∧ b = 4/9) ∧
  (∃ (y : ℝ), y^2 = a + 2*b ∧ (y = Real.sqrt 89 / 3 ∨ y = -Real.sqrt 89 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_square_root_problem_l2037_203756


namespace NUMINAMATH_CALUDE_perpendicular_line_plane_iff_perpendicular_two_lines_l2037_203711

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane using a point and a normal vector
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Predicate for a line being perpendicular to a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate for a line being perpendicular to another line -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Two lines are distinct -/
def distinct_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- The main theorem to be proven false -/
theorem perpendicular_line_plane_iff_perpendicular_two_lines (l : Line3D) (p : Plane3D) :
  perpendicular_line_plane l p ↔ 
  ∃ (l1 l2 : Line3D), line_in_plane l1 p ∧ line_in_plane l2 p ∧ 
                      distinct_lines l1 l2 ∧ 
                      perpendicular_lines l l1 ∧ perpendicular_lines l l2 :=
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_plane_iff_perpendicular_two_lines_l2037_203711


namespace NUMINAMATH_CALUDE_unique_perfect_square_polynomial_l2037_203784

theorem unique_perfect_square_polynomial : 
  ∃! x : ℤ, ∃ y : ℤ, x^4 + 8*x^3 + 18*x^2 + 8*x + 36 = y^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_perfect_square_polynomial_l2037_203784


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2037_203751

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x > 1 → x^2 > 1) ↔ (∃ x : ℝ, x > 1 ∧ x^2 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2037_203751


namespace NUMINAMATH_CALUDE_video_voting_result_l2037_203787

/-- Represents the voting system for a video --/
structure VideoVoting where
  totalVotes : ℕ
  likePercentage : ℚ
  finalScore : ℤ

/-- Theorem stating the conditions and the result to be proved --/
theorem video_voting_result (v : VideoVoting) 
  (h1 : v.likePercentage = 3/4)
  (h2 : v.finalScore = 140) :
  v.totalVotes = 280 := by
  sorry

end NUMINAMATH_CALUDE_video_voting_result_l2037_203787


namespace NUMINAMATH_CALUDE_max_ash_win_probability_l2037_203765

/-- Represents the types of monsters -/
inductive MonsterType
  | Fire
  | Grass
  | Water

/-- A lineup of monsters -/
def Lineup := List MonsterType

/-- The number of monsters in each lineup -/
def lineupSize : Nat := 15

/-- Calculates the probability of Ash winning given his lineup strategy -/
noncomputable def ashWinProbability (ashStrategy : Lineup) : ℝ :=
  sorry

/-- Theorem stating the maximum probability of Ash winning -/
theorem max_ash_win_probability :
  ∃ (optimalStrategy : Lineup),
    ashWinProbability optimalStrategy = 1 - (2/3)^lineupSize ∧
    ∀ (strategy : Lineup),
      ashWinProbability strategy ≤ ashWinProbability optimalStrategy :=
  sorry

end NUMINAMATH_CALUDE_max_ash_win_probability_l2037_203765


namespace NUMINAMATH_CALUDE_absolute_sum_vs_square_sum_l2037_203735

theorem absolute_sum_vs_square_sum :
  (∀ x y : ℝ, (abs x + abs y ≤ 1) → (x^2 + y^2 ≤ 1)) ∧
  (∃ x y : ℝ, (x^2 + y^2 ≤ 1) ∧ (abs x + abs y > 1)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_sum_vs_square_sum_l2037_203735


namespace NUMINAMATH_CALUDE_product_equation_solution_l2037_203758

theorem product_equation_solution :
  ∀ B : ℕ,
  B < 10 →
  (10 * B + 4) * (10 * 8 + B) = 7008 →
  B = 7 := by
sorry

end NUMINAMATH_CALUDE_product_equation_solution_l2037_203758


namespace NUMINAMATH_CALUDE_common_root_of_three_equations_l2037_203741

/-- Given nonzero real numbers a, b, c, and the fact that any two of the equations
    ax^11 + bx^4 + c = 0, bx^11 + cx^4 + a = 0, cx^11 + ax^4 + b = 0 have a common root,
    prove that all three equations have a common root. -/
theorem common_root_of_three_equations (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_common_12 : ∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ b * x^11 + c * x^4 + a = 0)
  (h_common_23 : ∃ x : ℝ, b * x^11 + c * x^4 + a = 0 ∧ c * x^11 + a * x^4 + b = 0)
  (h_common_13 : ∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ c * x^11 + a * x^4 + b = 0) :
  ∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ b * x^11 + c * x^4 + a = 0 ∧ c * x^11 + a * x^4 + b = 0 :=
sorry

end NUMINAMATH_CALUDE_common_root_of_three_equations_l2037_203741


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2037_203781

theorem multiplication_puzzle :
  ∀ (A B E F : ℕ),
    A < 10 → B < 10 → E < 10 → F < 10 →
    A ≠ B → A ≠ E → A ≠ F → B ≠ E → B ≠ F → E ≠ F →
    (100 * A + 10 * B + E) * F = 1000 * E + 100 * A + 10 * E + A →
    A + B = 5 := by
sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2037_203781


namespace NUMINAMATH_CALUDE_circumscribed_trapezoid_leg_length_l2037_203730

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The acute angle at the base of the trapezoid -/
  base_angle : ℝ
  /-- The length of the trapezoid's leg -/
  leg_length : ℝ

/-- Theorem stating the relationship between the area, base angle, and leg length of a circumscribed trapezoid -/
theorem circumscribed_trapezoid_leg_length 
  (t : CircumscribedTrapezoid) 
  (h1 : t.area = 32 * Real.sqrt 3)
  (h2 : t.base_angle = π / 3) :
  t.leg_length = 8 := by
  sorry

#check circumscribed_trapezoid_leg_length

end NUMINAMATH_CALUDE_circumscribed_trapezoid_leg_length_l2037_203730


namespace NUMINAMATH_CALUDE_product_less_than_2400_l2037_203731

theorem product_less_than_2400 : 817 * 3 < 2400 := by
  sorry

end NUMINAMATH_CALUDE_product_less_than_2400_l2037_203731


namespace NUMINAMATH_CALUDE_veranda_area_l2037_203774

/-- The area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) 
  (h1 : room_length = 19)
  (h2 : room_width = 12)
  (h3 : veranda_width = 2) : 
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 140 :=
by sorry

end NUMINAMATH_CALUDE_veranda_area_l2037_203774


namespace NUMINAMATH_CALUDE_original_decimal_proof_l2037_203725

theorem original_decimal_proof (x : ℝ) : x * 12 = 84.6 ↔ x = 7.05 := by
  sorry

end NUMINAMATH_CALUDE_original_decimal_proof_l2037_203725


namespace NUMINAMATH_CALUDE_gum_pieces_per_package_l2037_203795

theorem gum_pieces_per_package (total_packages : ℕ) (total_pieces : ℕ) 
  (h1 : total_packages = 9) 
  (h2 : total_pieces = 135) : 
  total_pieces / total_packages = 15 := by
  sorry

end NUMINAMATH_CALUDE_gum_pieces_per_package_l2037_203795


namespace NUMINAMATH_CALUDE_limit_of_function_at_one_l2037_203771

theorem limit_of_function_at_one :
  let f : ℝ → ℝ := λ x ↦ 2 * x - 3 - 1 / x
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |f x - (-2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_function_at_one_l2037_203771


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l2037_203726

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

-- Define the focal length
def focal_length (c : ℝ) : Prop := c = 2

-- Define that C passes through P(2, 5/3)
def passes_through_P (C : ℝ → ℝ → Prop) : Prop :=
  C 2 (5/3)

-- Define line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

-- Define that l passes through M(0, 1)
def passes_through_M (l : ℝ → ℝ → Prop) : Prop :=
  l 0 1

-- Define the condition for A and B
def vector_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 - 0, A.2 - 1) = (-2/3 * (B.1 - 0), -2/3 * (B.2 - 1))

-- Main theorem
theorem ellipse_and_line_theorem :
  ∀ C : ℝ → ℝ → Prop,
  (∀ x y, C x y ↔ x^2 / 9 + y^2 / 5 = 1) →
  focal_length 2 →
  passes_through_P C →
  ∃ k : ℝ, k = 1/3 ∨ k = -1/3 ∧
    ∀ x y, line_l k x y →
    passes_through_M (line_l k) ∧
    ∃ A B : ℝ × ℝ,
      C A.1 A.2 ∧ C B.1 B.2 ∧
      line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧
      vector_condition A B :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l2037_203726


namespace NUMINAMATH_CALUDE_incorrect_arrangements_count_l2037_203728

/-- The number of unique arrangements of the letters "e", "o", "h", "l", "l" -/
def total_arrangements : ℕ := 60

/-- The number of correct arrangements (spelling "hello") -/
def correct_arrangements : ℕ := 1

/-- Theorem stating the number of incorrect arrangements -/
theorem incorrect_arrangements_count :
  total_arrangements - correct_arrangements = 59 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_arrangements_count_l2037_203728


namespace NUMINAMATH_CALUDE_literary_club_probability_l2037_203713

theorem literary_club_probability : 
  let num_clubs : ℕ := 2
  let num_students : ℕ := 3
  let total_outcomes : ℕ := num_clubs ^ num_students
  let same_club_outcomes : ℕ := num_clubs
  let diff_club_probability : ℚ := 1 - (same_club_outcomes : ℚ) / total_outcomes
  diff_club_probability = 3/4 := by sorry

end NUMINAMATH_CALUDE_literary_club_probability_l2037_203713


namespace NUMINAMATH_CALUDE_custom_op_solution_l2037_203745

/-- Custom operation for integers -/
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem: Given the custom operation and x9 = 160, x must equal 21 -/
theorem custom_op_solution : ∃ x : ℤ, customOp x 9 = 160 ∧ x = 21 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_solution_l2037_203745


namespace NUMINAMATH_CALUDE_triangle_perimeter_sum_specific_triangle_perimeter_sum_l2037_203783

/-- The sum of perimeters of an infinite series of equilateral triangles -/
theorem triangle_perimeter_sum (initial_perimeter : ℝ) :
  initial_perimeter > 0 →
  (∑' n, initial_perimeter * (1/2)^n) = 2 * initial_perimeter :=
by sorry

/-- The specific case where the initial triangle has a perimeter of 90 cm -/
theorem specific_triangle_perimeter_sum :
  (∑' n, 90 * (1/2)^n) = 180 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_sum_specific_triangle_perimeter_sum_l2037_203783


namespace NUMINAMATH_CALUDE_lava_lamp_probability_l2037_203755

/-- The probability of a specific arrangement of lava lamps -/
theorem lava_lamp_probability :
  let total_lamps : ℕ := 6
  let red_lamps : ℕ := 3
  let blue_lamps : ℕ := 3
  let lamps_on : ℕ := 3
  let color_arrangements := Nat.choose total_lamps red_lamps
  let on_arrangements := Nat.choose total_lamps lamps_on
  let remaining_lamps : ℕ := 4
  let remaining_red : ℕ := 2
  let remaining_color_arrangements := Nat.choose remaining_lamps remaining_red
  let remaining_on_arrangements := Nat.choose remaining_lamps remaining_red
  (remaining_color_arrangements * remaining_on_arrangements : ℚ) / (color_arrangements * on_arrangements) = 9 / 100 :=
by sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_l2037_203755


namespace NUMINAMATH_CALUDE_abs_sum_diff_less_than_two_l2037_203775

theorem abs_sum_diff_less_than_two (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) :
  |a + b| + |a - b| < 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_diff_less_than_two_l2037_203775


namespace NUMINAMATH_CALUDE_zoes_flower_purchase_l2037_203772

theorem zoes_flower_purchase (flower_price : ℕ) (roses_bought : ℕ) (total_spent : ℕ) : 
  flower_price = 3 →
  roses_bought = 8 →
  total_spent = 30 →
  (total_spent - roses_bought * flower_price) / flower_price = 2 := by
sorry

end NUMINAMATH_CALUDE_zoes_flower_purchase_l2037_203772


namespace NUMINAMATH_CALUDE_right_triangle_angle_calculation_l2037_203779

theorem right_triangle_angle_calculation (x : ℝ) : 
  (3 * x > 3 * x - 40) →  -- Smallest angle condition
  (3 * x + (3 * x - 40) + 90 = 180) →  -- Sum of angles in a triangle
  x = 65 / 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_angle_calculation_l2037_203779


namespace NUMINAMATH_CALUDE_opposite_of_three_minus_one_l2037_203764

theorem opposite_of_three_minus_one :
  -(3 - 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_minus_one_l2037_203764


namespace NUMINAMATH_CALUDE_inequality_property_l2037_203797

theorem inequality_property (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l2037_203797


namespace NUMINAMATH_CALUDE_right_triangle_area_l2037_203786

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) : 
  (1/2) * a * b = 30 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2037_203786


namespace NUMINAMATH_CALUDE_chef_cooked_ten_wings_l2037_203752

/-- The number of additional chicken wings cooked by the chef for a group of friends -/
def additional_wings (num_friends : ℕ) (pre_cooked_wings : ℕ) (wings_per_person : ℕ) : ℕ :=
  num_friends * wings_per_person - pre_cooked_wings

/-- Theorem: Given 3 friends, 8 pre-cooked wings, and 6 wings per person, 
    the number of additional wings cooked is 10 -/
theorem chef_cooked_ten_wings : additional_wings 3 8 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_chef_cooked_ten_wings_l2037_203752


namespace NUMINAMATH_CALUDE_square_side_length_average_l2037_203793

theorem square_side_length_average (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) (h₄ : a₄ = 225) : 
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃ + Real.sqrt a₄) / 4 = 10 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_average_l2037_203793


namespace NUMINAMATH_CALUDE_mary_regular_hours_l2037_203769

/-- Represents Mary's work schedule and earnings --/
structure WorkSchedule where
  regularHours : ℕ
  overtimeHours : ℕ
  regularRate : ℕ
  overtimeRate : ℕ
  totalEarnings : ℕ

/-- Calculates the total earnings based on the work schedule --/
def calculateEarnings (schedule : WorkSchedule) : ℕ :=
  schedule.regularHours * schedule.regularRate + schedule.overtimeHours * schedule.overtimeRate

/-- The main theorem stating Mary's work hours at regular rate --/
theorem mary_regular_hours :
  ∃ (schedule : WorkSchedule),
    schedule.regularHours = 40 ∧
    schedule.regularRate = 8 ∧
    schedule.overtimeRate = 10 ∧
    schedule.regularHours + schedule.overtimeHours ≤ 40 ∧
    calculateEarnings schedule = 360 :=
by
  sorry

#check mary_regular_hours

end NUMINAMATH_CALUDE_mary_regular_hours_l2037_203769


namespace NUMINAMATH_CALUDE_average_age_of_eight_students_l2037_203732

theorem average_age_of_eight_students 
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_group1 : Nat)
  (num_group2 : Nat)
  (average_age_group2 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : num_group1 = 8)
  (h4 : num_group2 = 6)
  (h5 : average_age_group2 = 16)
  (h6 : age_last_student = 17)
  (h7 : total_students = num_group1 + num_group2 + 1) :
  (total_students : ℝ) * average_age_all - 
  (num_group2 : ℝ) * average_age_group2 - 
  age_last_student = (num_group1 : ℝ) * 14 := by
    sorry

#check average_age_of_eight_students

end NUMINAMATH_CALUDE_average_age_of_eight_students_l2037_203732


namespace NUMINAMATH_CALUDE_function_properties_l2037_203763

/-- A function f(x) = x^2 + bx + c where b and c are real numbers -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The theorem statement -/
theorem function_properties (b c : ℝ) 
  (h : ∀ x : ℝ, 2*x + b ≤ f b c x) : 
  (∀ x : ℝ, x ≥ 0 → f b c x ≤ (x + c)^2) ∧ 
  (∃ m : ℝ, m = 3/2 ∧ ∀ b' c' : ℝ, (∀ x : ℝ, 2*x + b' ≤ f b' c' x) → 
    f b' c' c' - f b' c' b' ≤ m*(c'^2 - b'^2) ∧
    ∀ m' : ℝ, (∀ b' c' : ℝ, (∀ x : ℝ, 2*x + b' ≤ f b' c' x) → 
      f b' c' c' - f b' c' b' ≤ m'*(c'^2 - b'^2)) → m ≤ m') := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2037_203763


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l2037_203789

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_and_extrema :
  ∃ (tangent_line : ℝ → ℝ) (max_value min_value : ℝ),
    (∀ x, tangent_line x = 1) ∧
    (f 0 = max_value) ∧
    (f (Real.pi / 2) = min_value) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ max_value) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ min_value) ∧
    (max_value = 1) ∧
    (min_value = -Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l2037_203789


namespace NUMINAMATH_CALUDE_magnitude_of_one_plus_two_i_to_eighth_l2037_203750

theorem magnitude_of_one_plus_two_i_to_eighth : Complex.abs ((1 + 2*Complex.I)^8) = 625 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_one_plus_two_i_to_eighth_l2037_203750


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l2037_203715

-- Define the polynomial Q(x) = x³ - 5x² + 6x - 2
def Q (x : ℝ) : ℝ := x^3 - 5*x^2 + 6*x - 2

-- Theorem statement
theorem cubic_polynomial_root :
  -- Q is a monic cubic polynomial with integer coefficients
  (∀ x, Q x = x^3 - 5*x^2 + 6*x - 2) ∧
  -- The leading coefficient is 1 (monic)
  (∃ a b c, ∀ x, Q x = x^3 + a*x^2 + b*x + c) ∧
  -- All coefficients are integers
  (∃ a b c : ℤ, ∀ x, Q x = x^3 + a*x^2 + b*x + c) ∧
  -- √2 + 2 is a root of Q
  Q (Real.sqrt 2 + 2) = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l2037_203715


namespace NUMINAMATH_CALUDE_vacation_cost_distribution_l2037_203749

/-- Represents the vacation cost distribution problem -/
theorem vacation_cost_distribution 
  (anna_paid ben_paid carol_paid dan_paid : ℚ)
  (a b c : ℚ)
  (h1 : anna_paid = 130)
  (h2 : ben_paid = 150)
  (h3 : carol_paid = 110)
  (h4 : dan_paid = 190)
  (h5 : (anna_paid + ben_paid + carol_paid + dan_paid) / 4 = 145)
  (h6 : a = 5)
  (h7 : b = 5)
  (h8 : c = 35)
  : a - b + c = 35 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_distribution_l2037_203749


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2037_203770

/-- 
Given a positive integer n, we define two properties:
1. 5n is a perfect square
2. 7n is a perfect cube

This theorem states that 1225 is the smallest positive integer satisfying both properties.
-/
theorem smallest_n_square_and_cube : ∀ n : ℕ+, 
  (∃ k : ℕ+, 5 * n = k^2) ∧ 
  (∃ m : ℕ+, 7 * n = m^3) → 
  n ≥ 1225 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2037_203770


namespace NUMINAMATH_CALUDE_mode_of_student_ages_l2037_203753

def student_ages : List ℕ := [13, 14, 15, 14, 14, 15]

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_student_ages :
  mode student_ages = 14 := by sorry

end NUMINAMATH_CALUDE_mode_of_student_ages_l2037_203753


namespace NUMINAMATH_CALUDE_probability_identical_after_rotation_l2037_203734

/-- Represents the colors available for painting the cube faces -/
inductive Color
  | Black
  | White
  | Red

/-- Represents a cube with painted faces -/
structure Cube :=
  (faces : Fin 6 → Color)

/-- Checks if a cube satisfies the adjacent face color constraint -/
def validCube (c : Cube) : Prop := sorry

/-- Counts the number of valid cube colorings -/
def validColoringsCount : Nat := sorry

/-- Counts the number of ways cubes can be identical after rotation -/
def identicalAfterRotationCount : Nat := sorry

/-- Theorem stating the probability of three cubes being identical after rotation -/
theorem probability_identical_after_rotation :
  (identicalAfterRotationCount : ℚ) / (validColoringsCount ^ 3 : ℚ) = 1 / 45 := by sorry

end NUMINAMATH_CALUDE_probability_identical_after_rotation_l2037_203734


namespace NUMINAMATH_CALUDE_equation_solution_l2037_203716

theorem equation_solution : ∃ x : ℚ, (1 / 3 + 1 / x = 2 / 3) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2037_203716


namespace NUMINAMATH_CALUDE_lemons_for_ten_gallons_l2037_203703

/-- The number of lemons required to make a certain amount of lemonade -/
structure LemonadeRecipe where
  lemons : ℕ
  gallons : ℕ

/-- Calculates the number of lemons needed for a given number of gallons,
    based on a known recipe. The result is rounded up to the nearest integer. -/
def calculate_lemons (recipe : LemonadeRecipe) (target_gallons : ℕ) : ℕ :=
  ((recipe.lemons : ℚ) * target_gallons / recipe.gallons).ceil.toNat

/-- The known recipe for lemonade -/
def known_recipe : LemonadeRecipe := ⟨48, 64⟩

/-- The target amount of lemonade to make -/
def target_gallons : ℕ := 10

/-- Theorem stating that 8 lemons are needed to make 10 gallons of lemonade -/
theorem lemons_for_ten_gallons :
  calculate_lemons known_recipe target_gallons = 8 := by sorry

end NUMINAMATH_CALUDE_lemons_for_ten_gallons_l2037_203703


namespace NUMINAMATH_CALUDE_water_per_day_per_man_l2037_203768

/-- Calculates the amount of water needed per day per man on a sea voyage --/
theorem water_per_day_per_man 
  (total_men : ℕ) 
  (miles_per_day : ℕ) 
  (total_miles : ℕ) 
  (total_water : ℕ) : 
  total_men = 25 → 
  miles_per_day = 200 → 
  total_miles = 4000 → 
  total_water = 250 → 
  (total_water : ℚ) / ((total_miles : ℚ) / (miles_per_day : ℚ)) / (total_men : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_water_per_day_per_man_l2037_203768


namespace NUMINAMATH_CALUDE_order_relation_l2037_203748

theorem order_relation (a b c : ℝ) : 
  a = 1 / 2023 ∧ 
  b = Real.tan (Real.exp (1 / 2023) / 2023) ∧ 
  c = Real.sin (Real.exp (1 / 2024) / 2024) →
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_order_relation_l2037_203748


namespace NUMINAMATH_CALUDE_projected_attendance_increase_l2037_203709

theorem projected_attendance_increase (A : ℝ) (h1 : A > 0) : 
  let actual_attendance := 0.8 * A
  let projected_attendance := (1 + P / 100) * A
  0.8 * A = 0.64 * ((1 + P / 100) * A) →
  P = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_projected_attendance_increase_l2037_203709


namespace NUMINAMATH_CALUDE_jamie_dives_for_pearls_l2037_203739

/-- Given that 25% of oysters have pearls, Jamie can collect 16 oysters per dive,
    and Jamie needs to collect 56 pearls, prove that Jamie needs to make 14 dives. -/
theorem jamie_dives_for_pearls (pearl_probability : ℚ) (oysters_per_dive : ℕ) (total_pearls : ℕ) :
  pearl_probability = 1/4 →
  oysters_per_dive = 16 →
  total_pearls = 56 →
  (total_pearls : ℚ) / (pearl_probability * oysters_per_dive) = 14 := by
  sorry

end NUMINAMATH_CALUDE_jamie_dives_for_pearls_l2037_203739


namespace NUMINAMATH_CALUDE_florist_bouquets_l2037_203721

/-- Calculates the number of bouquets that can be made given the initial number of seeds,
    the number of flowers killed by fungus, and the number of flowers per bouquet. -/
def calculateBouquets (seedsPerColor : ℕ) (redKilled yellowKilled orangeKilled purpleKilled : ℕ) (flowersPerBouquet : ℕ) : ℕ :=
  let redLeft := seedsPerColor - redKilled
  let yellowLeft := seedsPerColor - yellowKilled
  let orangeLeft := seedsPerColor - orangeKilled
  let purpleLeft := seedsPerColor - purpleKilled
  let totalFlowersLeft := redLeft + yellowLeft + orangeLeft + purpleLeft
  totalFlowersLeft / flowersPerBouquet

/-- Theorem stating that given the specific conditions of the problem,
    the florist can make 36 bouquets. -/
theorem florist_bouquets :
  calculateBouquets 125 45 61 30 40 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_florist_bouquets_l2037_203721


namespace NUMINAMATH_CALUDE_machine_completion_time_l2037_203724

/-- Given two machines where one takes T hours and the other takes 8 hours to complete an order,
    if they complete the order together in 4.235294117647059 hours, then T = 9. -/
theorem machine_completion_time (T : ℝ) : 
  (1 / T + 1 / 8 = 1 / 4.235294117647059) → T = 9 := by
sorry

end NUMINAMATH_CALUDE_machine_completion_time_l2037_203724


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l2037_203782

theorem intersection_sum_zero (x₁ x₂ : ℝ) (h₁ : x₁^2 + 6^2 = 144) (h₂ : x₂^2 + 6^2 = 144) :
  x₁ + x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l2037_203782


namespace NUMINAMATH_CALUDE_cd_cost_l2037_203790

theorem cd_cost (num_films : ℕ) (num_books : ℕ) (num_cds : ℕ) 
  (film_cost : ℕ) (book_cost : ℕ) (total_spent : ℕ) :
  num_films = 9 →
  num_books = 4 →
  num_cds = 6 →
  film_cost = 5 →
  book_cost = 4 →
  total_spent = 79 →
  (total_spent - (num_films * film_cost + num_books * book_cost)) / num_cds = 3 := by
  sorry

#eval (79 - (9 * 5 + 4 * 4)) / 6

end NUMINAMATH_CALUDE_cd_cost_l2037_203790


namespace NUMINAMATH_CALUDE_payment_calculation_l2037_203780

/-- Represents a store's pricing and promotion options for suits and ties. -/
structure StorePricing where
  suit_price : ℕ
  tie_price : ℕ
  option1 : ℕ → ℕ  -- Function representing the cost for Option 1
  option2 : ℕ → ℕ  -- Function representing the cost for Option 2

/-- Calculates the payment for a customer buying suits and ties under different options. -/
def calculate_payment (pricing : StorePricing) (suits : ℕ) (ties : ℕ) : ℕ × ℕ :=
  (pricing.option1 ties, pricing.option2 ties)

/-- Theorem stating the correct calculation of payments for the given problem. -/
theorem payment_calculation (x : ℕ) (h : x > 20) :
  let pricing := StorePricing.mk 1000 200
    (fun ties => 20000 + 200 * (ties - 20))
    (fun ties => (20 * 1000 + ties * 200) * 9 / 10)
  (calculate_payment pricing 20 x).1 = 200 * x + 16000 ∧
  (calculate_payment pricing 20 x).2 = 180 * x + 18000 := by
  sorry

end NUMINAMATH_CALUDE_payment_calculation_l2037_203780


namespace NUMINAMATH_CALUDE_contradiction_assumption_l2037_203760

theorem contradiction_assumption (x y z : ℝ) :
  (¬ (x > 0 ∨ y > 0 ∨ z > 0)) ↔ (x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l2037_203760


namespace NUMINAMATH_CALUDE_investment_growth_l2037_203762

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.08

/-- The time period in years -/
def time_period : ℕ := 28

/-- The initial investment amount in dollars -/
def initial_investment : ℝ := 3500

/-- The final value after the investment period in dollars -/
def final_value : ℝ := 31500

/-- Compound interest formula: A = P(1 + r)^t 
    Where A is the final amount, P is the principal (initial investment),
    r is the annual interest rate, and t is the time in years -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem investment_growth :
  compound_interest initial_investment interest_rate time_period = final_value := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l2037_203762


namespace NUMINAMATH_CALUDE_max_subtract_add_result_l2037_203722

def S : Set Int := {-20, -10, 0, 5, 15, 25}

theorem max_subtract_add_result (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) :
  (a - b + c) ≤ 70 ∧ ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x - y + z = 70 := by
  sorry

end NUMINAMATH_CALUDE_max_subtract_add_result_l2037_203722


namespace NUMINAMATH_CALUDE_cube_root_of_one_sixty_fourth_l2037_203766

theorem cube_root_of_one_sixty_fourth (x : ℝ) : x^3 = 1/64 → x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_one_sixty_fourth_l2037_203766


namespace NUMINAMATH_CALUDE_angle_sum_at_point_l2037_203727

theorem angle_sum_at_point (x : ℝ) : 
  (120 : ℝ) + x + x + 2*x = 360 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_at_point_l2037_203727


namespace NUMINAMATH_CALUDE_circle_M_properties_l2037_203742

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the center of the circle
def center_M : ℝ × ℝ := (1, -2)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, -1)

-- Define point A
def point_A : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem circle_M_properties :
  (center_M.2 = -2 * center_M.1) ∧ 
  tangent_line point_P.1 point_P.2 ∧
  (∀ x y, tangent_line x y → ¬ circle_M x y) ∧
  circle_M point_P.1 point_P.2 →
  (∀ x y, circle_M x y → 
    Real.sqrt ((x - point_A.1)^2 + (y - point_A.2)^2) ≥ Real.sqrt 2) ∧
  (∃ x y, circle_M x y ∧ 
    Real.sqrt ((x - point_A.1)^2 + (y - point_A.2)^2) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_M_properties_l2037_203742


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2037_203759

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmph = 72 →
  crossing_time = 13.998880089592832 →
  ∃ (bridge_length : ℝ), 
    (169.97 < bridge_length) ∧ 
    (bridge_length < 169.99) ∧
    (bridge_length = train_speed_kmph * (1000 / 3600) * crossing_time - train_length) :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2037_203759


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2037_203754

/-- A rhombus with given properties -/
structure Rhombus where
  /-- Length of the shorter diagonal -/
  shorter_diagonal : ℝ
  /-- Length of the longer diagonal -/
  longer_diagonal : ℝ
  /-- Perimeter of the rhombus -/
  perimeter : ℝ
  /-- The shorter diagonal is 30 cm -/
  shorter_diagonal_length : shorter_diagonal = 30
  /-- The perimeter is 156 cm -/
  perimeter_length : perimeter = 156
  /-- The longer diagonal is longer than the shorter diagonal -/
  diagonal_order : longer_diagonal ≥ shorter_diagonal

/-- Theorem: In a rhombus with one diagonal of 30 cm and a perimeter of 156 cm, 
    the length of the other diagonal is 72 cm -/
theorem rhombus_longer_diagonal (r : Rhombus) : r.longer_diagonal = 72 := by
  sorry

#check rhombus_longer_diagonal

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2037_203754


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2037_203700

theorem no_integer_solutions : ¬ ∃ (a b c : ℤ), a^2 + b^2 = 8*c + 6 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2037_203700


namespace NUMINAMATH_CALUDE_systematic_sampling_most_suitable_for_C_l2037_203747

/-- Characteristics of systematic sampling -/
structure SystematicSampling where
  large_population : Bool
  regular_interval : Bool
  balanced_group : Bool

/-- Sampling scenario -/
structure SamplingScenario where
  population_size : Nat
  sample_size : Nat
  is_homogeneous : Bool

/-- Check if a scenario is suitable for systematic sampling -/
def is_suitable_for_systematic_sampling (scenario : SamplingScenario) : Bool :=
  scenario.population_size > scenario.sample_size ∧ 
  scenario.population_size ≥ 1000 ∧ 
  scenario.sample_size ≥ 100 ∧
  scenario.is_homogeneous

/-- The four sampling scenarios -/
def scenario_A : SamplingScenario := ⟨2000, 200, false⟩
def scenario_B : SamplingScenario := ⟨2000, 5, true⟩
def scenario_C : SamplingScenario := ⟨2000, 200, true⟩
def scenario_D : SamplingScenario := ⟨20, 5, true⟩

theorem systematic_sampling_most_suitable_for_C :
  is_suitable_for_systematic_sampling scenario_C ∧
  ¬is_suitable_for_systematic_sampling scenario_A ∧
  ¬is_suitable_for_systematic_sampling scenario_B ∧
  ¬is_suitable_for_systematic_sampling scenario_D :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_most_suitable_for_C_l2037_203747


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_circle_angles_l2037_203718

theorem right_triangle_inscribed_circle_angles (k : ℝ) (k_pos : k > 0) :
  ∃ (α β : ℝ),
    α + β = π / 2 ∧
    (α = π / 4 - Real.arcsin (Real.sqrt 2 * (k - 1) / (2 * (k + 1))) ∨
     α = π / 4 + Real.arcsin (Real.sqrt 2 * (k - 1) / (2 * (k + 1)))) ∧
    (β = π / 4 - Real.arcsin (Real.sqrt 2 * (k - 1) / (2 * (k + 1))) ∨
     β = π / 4 + Real.arcsin (Real.sqrt 2 * (k - 1) / (2 * (k + 1)))) :=
by sorry


end NUMINAMATH_CALUDE_right_triangle_inscribed_circle_angles_l2037_203718


namespace NUMINAMATH_CALUDE_classroom_ratio_l2037_203788

theorem classroom_ratio :
  ∀ (x y : ℕ),
    x + y = 15 →
    30 * x + 25 * y = 400 →
    x / 15 = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_ratio_l2037_203788


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2037_203719

-- Problem 1
theorem problem_1 : Real.sqrt 9 + |3 - Real.pi| - Real.sqrt ((-3)^2) = Real.pi - 3 := by
  sorry

-- Problem 2
theorem problem_2 : ∃ x : ℝ, 3 * (x - 1)^3 = 81 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2037_203719


namespace NUMINAMATH_CALUDE_triangle_angles_sum_l2037_203712

theorem triangle_angles_sum (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ 8 * x + 13 * y = 130 → x + y = 1289 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_sum_l2037_203712

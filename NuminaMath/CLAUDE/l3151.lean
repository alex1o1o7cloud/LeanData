import Mathlib

namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l3151_315198

def rahul_future_age : ℕ := 32
def years_to_future : ℕ := 4
def deepak_age : ℕ := 21

theorem rahul_deepak_age_ratio :
  let rahul_age := rahul_future_age - years_to_future
  (rahul_age : ℚ) / deepak_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l3151_315198


namespace NUMINAMATH_CALUDE_clown_balloons_l3151_315190

/-- The number of additional balloons blown up by the clown -/
def additional_balloons (initial : ℕ) (total : ℕ) : ℕ :=
  total - initial

theorem clown_balloons : additional_balloons 47 60 = 13 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l3151_315190


namespace NUMINAMATH_CALUDE_arithmetic_sequence_partial_sum_l3151_315139

-- Define the arithmetic sequence and its partial sums
def arithmetic_sequence (n : ℕ) : ℝ := sorry
def S (n : ℕ) : ℝ := sorry

-- State the theorem
theorem arithmetic_sequence_partial_sum :
  S 3 = 6 ∧ S 9 = 27 → S 6 = 15 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_partial_sum_l3151_315139


namespace NUMINAMATH_CALUDE_two_number_problem_l3151_315126

theorem two_number_problem (x y : ℝ) (h1 : x + y = 30) (h2 : 3 * y - 4 * x = 9) :
  |y - x| = 129 / 21 := by
sorry

end NUMINAMATH_CALUDE_two_number_problem_l3151_315126


namespace NUMINAMATH_CALUDE_nathaniel_win_probability_l3151_315110

/-- A fair six-sided die -/
def FairDie : Type := Fin 6

/-- The game state -/
structure GameState :=
  (sum : ℕ)
  (currentPlayer : Bool)  -- true for Nathaniel, false for Obediah

/-- Check if a number is a multiple of 7 -/
def isMultipleOf7 (n : ℕ) : Bool :=
  n % 7 = 0

/-- The probability of Nathaniel winning the game -/
noncomputable def nathanielWinProbability : ℝ :=
  5 / 11

/-- Theorem: The probability of Nathaniel winning the game is 5/11 -/
theorem nathaniel_win_probability :
  nathanielWinProbability = 5 / 11 := by
  sorry

#check nathaniel_win_probability

end NUMINAMATH_CALUDE_nathaniel_win_probability_l3151_315110


namespace NUMINAMATH_CALUDE_fifteenth_in_base_8_l3151_315195

/-- Converts a decimal number to its representation in base 8 -/
def to_base_8 (n : ℕ) : ℕ := sorry

/-- The fifteenth number in base 10 -/
def fifteenth : ℕ := 15

/-- The representation of the fifteenth number in base 8 -/
def fifteenth_base_8 : ℕ := 17

theorem fifteenth_in_base_8 :
  to_base_8 fifteenth = fifteenth_base_8 := by sorry

end NUMINAMATH_CALUDE_fifteenth_in_base_8_l3151_315195


namespace NUMINAMATH_CALUDE_dice_game_probability_l3151_315177

def score (roll1 roll2 : Nat) : Nat := max roll1 roll2

def is_favorable (roll1 roll2 : Nat) : Bool :=
  score roll1 roll2 ≤ 3

def total_outcomes : Nat := 36

def favorable_outcomes : Nat := 9

theorem dice_game_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_dice_game_probability_l3151_315177


namespace NUMINAMATH_CALUDE_valid_circular_arrangement_exists_l3151_315166

/-- A type representing a circular arrangement of 9 numbers -/
def CircularArrangement := Fin 9 → Fin 9

/-- Check if two numbers in the arrangement are adjacent -/
def are_adjacent (arr : CircularArrangement) (i j : Fin 9) : Prop :=
  (j = i + 1) ∨ (i = 8 ∧ j = 0)

/-- Check if a number is valid in the arrangement (1 to 9) -/
def is_valid_number (n : Fin 9) : Prop := n.val + 1 ∈ Finset.range 10

/-- Check if the sum of two numbers is not divisible by 3, 5, or 7 -/
def sum_not_divisible (a b : Fin 9) : Prop :=
  ¬(((a.val + 1) + (b.val + 1)) % 3 = 0) ∧
  ¬(((a.val + 1) + (b.val + 1)) % 5 = 0) ∧
  ¬(((a.val + 1) + (b.val + 1)) % 7 = 0)

/-- The main theorem stating that a valid circular arrangement exists -/
theorem valid_circular_arrangement_exists : ∃ (arr : CircularArrangement),
  (∀ i : Fin 9, is_valid_number (arr i)) ∧
  (∀ i j : Fin 9, are_adjacent arr i j → sum_not_divisible (arr i) (arr j)) ∧
  Function.Injective arr :=
sorry

end NUMINAMATH_CALUDE_valid_circular_arrangement_exists_l3151_315166


namespace NUMINAMATH_CALUDE_inequality_proof_l3151_315121

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3151_315121


namespace NUMINAMATH_CALUDE_intersection_point_l3151_315154

/-- A parabola defined by x = -3y^2 - 4y + 7 -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 7

/-- The line x = m -/
def line (m : ℝ) : ℝ := m

/-- The condition for a single intersection point -/
def single_intersection (m : ℝ) : Prop :=
  ∃! y : ℝ, parabola y = line m

theorem intersection_point (m : ℝ) : single_intersection m ↔ m = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3151_315154


namespace NUMINAMATH_CALUDE_grade_assignments_l3151_315163

/-- The number of possible grades to assign -/
def num_grades : ℕ := 3

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- Theorem: The number of ways to assign grades to students -/
theorem grade_assignments :
  num_grades ^ num_students = 531441 := by
sorry

end NUMINAMATH_CALUDE_grade_assignments_l3151_315163


namespace NUMINAMATH_CALUDE_tan_675_degrees_l3151_315114

theorem tan_675_degrees (m : ℤ) : 
  -180 < m ∧ m < 180 ∧ Real.tan (↑m * π / 180) = Real.tan (675 * π / 180) → m = 135 :=
by sorry

end NUMINAMATH_CALUDE_tan_675_degrees_l3151_315114


namespace NUMINAMATH_CALUDE_expression_evaluation_l3151_315179

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/2
  2 * x^2 + (-x^2 - 2*x*y + 2*y^2) - 3*(x^2 - x*y + 2*y^2) = -10 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3151_315179


namespace NUMINAMATH_CALUDE_unique_factorial_product_l3151_315137

theorem unique_factorial_product (n : ℕ) : (n + 1) * n.factorial = 5040 ↔ n = 6 := by sorry

end NUMINAMATH_CALUDE_unique_factorial_product_l3151_315137


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3151_315141

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 25 (2 * x) = Nat.choose 25 (x + 4)) ↔ (x = 4 ∨ x = 7) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3151_315141


namespace NUMINAMATH_CALUDE_all_students_same_room_probability_l3151_315180

/-- The number of rooms available for assignment. -/
def num_rooms : ℕ := 4

/-- The number of students being assigned to rooms. -/
def num_students : ℕ := 3

/-- The probability of a student being assigned to any specific room. -/
def prob_per_room : ℚ := 1 / num_rooms

/-- The total number of possible assignment outcomes. -/
def total_outcomes : ℕ := num_rooms ^ num_students

/-- The number of favorable outcomes (all students in the same room). -/
def favorable_outcomes : ℕ := num_rooms

/-- The probability that all students are assigned to the same room. -/
theorem all_students_same_room_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_all_students_same_room_probability_l3151_315180


namespace NUMINAMATH_CALUDE_probability_at_least_two_in_same_class_probability_equals_seven_twentyfifths_l3151_315152

/-- The probability that at least 2 out of 3 friends are in the same class, given 10 classes -/
theorem probability_at_least_two_in_same_class : ℝ :=
  let total_classes := 10
  let total_friends := 3
  let prob_all_different := (total_classes * (total_classes - 1) * (total_classes - 2)) / (total_classes ^ total_friends)
  1 - prob_all_different

/-- The probability is equal to 7/25 -/
theorem probability_equals_seven_twentyfifths : probability_at_least_two_in_same_class = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_in_same_class_probability_equals_seven_twentyfifths_l3151_315152


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3151_315132

theorem polynomial_multiplication (t : ℝ) : 
  (3*t^3 + 2*t^2 - 4*t + 3) * (-2*t^2 + 3*t - 4) = 
  -6*t^5 + 5*t^4 + 2*t^3 - 26*t^2 + 25*t - 12 := by
sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3151_315132


namespace NUMINAMATH_CALUDE_power_function_increasing_m_eq_3_l3151_315175

/-- A function f(x) = cx^p where c and p are constants and x > 0 -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ c p : ℝ, ∀ x > 0, f x = c * x^p

/-- A function f is increasing on (0, +∞) if for all x, y > 0, x < y implies f(x) < f(y) -/
def isIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

/-- The main theorem stating that m = 3 is the only value satisfying the conditions -/
theorem power_function_increasing_m_eq_3 :
  ∃! m : ℝ, 
    isPowerFunction (fun x => (m^2 - m - 5) * x^(m-1)) ∧ 
    isIncreasing (fun x => (m^2 - m - 5) * x^(m-1)) :=
sorry

end NUMINAMATH_CALUDE_power_function_increasing_m_eq_3_l3151_315175


namespace NUMINAMATH_CALUDE_square_root_fraction_equality_l3151_315197

theorem square_root_fraction_equality : 
  Real.sqrt (8^2 + 15^2) / Real.sqrt (25 + 16) = (17 * Real.sqrt 41) / 41 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_equality_l3151_315197


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_quadratic_roots_l3151_315118

theorem perpendicular_lines_from_quadratic_roots (b : ℝ) :
  ∀ k₁ k₂ : ℝ, (k₁^2 + b*k₁ - 1 = 0) → (k₂^2 + b*k₂ - 1 = 0) → k₁ * k₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_quadratic_roots_l3151_315118


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3151_315193

theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (b * c / (a^2 + b^2).sqrt = b) →
  (2 * a ≤ b) →
  let e := c / a
  e > Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3151_315193


namespace NUMINAMATH_CALUDE_square_with_equilateral_triangle_l3151_315128

/-- Given a square ABCD with side length (R-1) cm and an equilateral triangle AEF 
    where E and F are points on BC and CD respectively, if the area of triangle AEF 
    is (S-3) cm², then S = 2√3. -/
theorem square_with_equilateral_triangle (R S : ℝ) : 
  let square_side := R - 1
  let triangle_area := S - 3
  square_side > 0 →
  triangle_area > 0 →
  S = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_square_with_equilateral_triangle_l3151_315128


namespace NUMINAMATH_CALUDE_five_digit_division_l3151_315129

/-- A five-digit number with the first digit not zero -/
def FiveDigitNumber (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A four-digit number formed by removing the middle digit of a five-digit number -/
def FourDigitNumber (m n : ℕ) : Prop :=
  FiveDigitNumber n ∧ 
  ∃ (x y z u v : ℕ), 
    n = x * 10000 + y * 1000 + z * 100 + u * 10 + v ∧
    m = x * 1000 + y * 100 + u * 10 + v ∧
    0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9 ∧ 0 ≤ v ∧ v ≤ 9 ∧
    x ≠ 0

theorem five_digit_division (n m : ℕ) : 
  FiveDigitNumber n → FourDigitNumber m n → (∃ k : ℕ, n = k * m) ↔ 
  ∃ (x y : ℕ), n = (10 * x + y) * 1000 ∧ 10 ≤ 10 * x + y ∧ 10 * x + y ≤ 99 :=
sorry

end NUMINAMATH_CALUDE_five_digit_division_l3151_315129


namespace NUMINAMATH_CALUDE_remainder_problem_l3151_315130

theorem remainder_problem (N : ℤ) (h : N % 133 = 16) : N % 50 = 49 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3151_315130


namespace NUMINAMATH_CALUDE_wall_length_proof_l3151_315157

/-- Proves that the length of a wall is 800 cm given the dimensions of bricks and wall, and the number of bricks needed. -/
theorem wall_length_proof (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
                          (wall_width : ℝ) (wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 50 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_width = 600 →
  wall_height = 22.5 →
  num_bricks = 3200 →
  ∃ (wall_length : ℝ), wall_length = 800 := by
  sorry

#check wall_length_proof

end NUMINAMATH_CALUDE_wall_length_proof_l3151_315157


namespace NUMINAMATH_CALUDE_problem_solution_l3151_315169

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * 2^(x+1) + (k-3) * 2^(-x)

theorem problem_solution (k : ℝ) (t : ℝ) :
  (∀ x, f k (-x) = -(f k x)) →
  (∀ x ∈ Set.Icc 1 3, f k (x^2 - x) + f k (t*x + 4) > 0) →
  (k = 1 ∧
   (∀ x₁ x₂, x₁ < x₂ → f k x₁ < f k x₂) ∧
   t > -3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3151_315169


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l3151_315162

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 20)
  (h2 : c + a = 22)
  (h3 : a + b = 24) :
  Real.sqrt (a * b * c * (a + b + c)) = 206.1 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l3151_315162


namespace NUMINAMATH_CALUDE_sum_of_squares_l3151_315189

theorem sum_of_squares (x y z a b : ℝ) 
  (sum_eq : x + y + z = a) 
  (sum_prod_eq : x*y + y*z + x*z = b) : 
  x^2 + y^2 + z^2 = a^2 - 2*b := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3151_315189


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3151_315150

theorem division_remainder_problem : ∃ (A : ℕ), 17 = 5 * 3 + A ∧ A < 5 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3151_315150


namespace NUMINAMATH_CALUDE_parabolas_no_intersection_l3151_315135

/-- The parabolas y = 3x^2 - 6x + 6 and y = -2x^2 + x + 3 do not intersect in the real plane. -/
theorem parabolas_no_intersection : 
  ∀ x y : ℝ, (y = 3*x^2 - 6*x + 6) → (y = -2*x^2 + x + 3) → False :=
by
  sorry

end NUMINAMATH_CALUDE_parabolas_no_intersection_l3151_315135


namespace NUMINAMATH_CALUDE_cyclist_distance_l3151_315167

/-- Proves that a cyclist traveling at 18 km/hr for 2 minutes and 30 seconds covers a distance of 750 meters. -/
theorem cyclist_distance (speed : ℝ) (time_min : ℝ) (time_sec : ℝ) (distance : ℝ) :
  speed = 18 →
  time_min = 2 →
  time_sec = 30 →
  distance = speed * (time_min / 60 + time_sec / 3600) * 1000 →
  distance = 750 := by
sorry


end NUMINAMATH_CALUDE_cyclist_distance_l3151_315167


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l3151_315158

def U : Set ℕ := {x | 0 ≤ x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem intersection_complement_equals : S ∩ (U \ T) = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l3151_315158


namespace NUMINAMATH_CALUDE_retirement_total_is_70_l3151_315188

/-- The retirement eligibility rule for a company -/
structure RetirementRule where
  hire_year : ℕ
  hire_age : ℕ
  eligible_year : ℕ

/-- Calculate the required total of age and years of employment for retirement -/
def retirement_total (rule : RetirementRule) : ℕ :=
  let years_of_employment := rule.eligible_year - rule.hire_year
  let age_at_eligibility := rule.hire_age + years_of_employment
  age_at_eligibility + years_of_employment

/-- Theorem stating the required total for retirement -/
theorem retirement_total_is_70 (rule : RetirementRule) 
  (h1 : rule.hire_year = 1990)
  (h2 : rule.hire_age = 32)
  (h3 : rule.eligible_year = 2009) :
  retirement_total rule = 70 := by
  sorry

#eval retirement_total ⟨1990, 32, 2009⟩

end NUMINAMATH_CALUDE_retirement_total_is_70_l3151_315188


namespace NUMINAMATH_CALUDE_contrapositive_example_l3151_315153

theorem contrapositive_example :
  (∀ x : ℝ, x = 2 → x^2 - 3*x + 2 = 0) ↔
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l3151_315153


namespace NUMINAMATH_CALUDE_spoiled_fish_fraction_l3151_315142

theorem spoiled_fish_fraction (initial_stock sold_fish new_stock final_stock : ℕ) : 
  initial_stock = 200 →
  sold_fish = 50 →
  new_stock = 200 →
  final_stock = 300 →
  (final_stock - new_stock) / (initial_stock - sold_fish) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_spoiled_fish_fraction_l3151_315142


namespace NUMINAMATH_CALUDE_salt_solution_dilution_l3151_315187

/-- Given a salt solution, prove that adding a specific amount of water yields the target concentration -/
theorem salt_solution_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 40 →
  initial_concentration = 0.25 →
  target_concentration = 0.15 →
  water_added = 400 / 15 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = target_concentration := by
  sorry

#eval (400 : ℚ) / 15

end NUMINAMATH_CALUDE_salt_solution_dilution_l3151_315187


namespace NUMINAMATH_CALUDE_m_plus_p_equals_11_l3151_315134

/-- Sum of odd numbers from 1 to n -/
def sumOddNumbers (n : ℕ) : ℕ :=
  (n + 1) * n / 2

/-- Decomposition of p^3 for positive integers p ≥ 2 -/
def decompositionP3 (p : ℕ) : ℕ :=
  2 * p * p - 1

theorem m_plus_p_equals_11 (m p : ℕ) 
  (h1 : m ^ 2 = sumOddNumbers 6)
  (h2 : decompositionP3 p = 21) : 
  m + p = 11 := by
  sorry

#eval sumOddNumbers 6  -- Should output 36
#eval decompositionP3 5  -- Should output 21

end NUMINAMATH_CALUDE_m_plus_p_equals_11_l3151_315134


namespace NUMINAMATH_CALUDE_locus_of_touching_parabolas_l3151_315161

/-- Given a parabola y = x^2 with directrix y = -1/4, this theorem describes
    the locus of points P(u, v) for which there exists a line v parallel to
    the directrix and at a distance s from it, such that the parabola with
    directrix v and focus P touches the given parabola. -/
theorem locus_of_touching_parabolas (s : ℝ) (u v : ℝ) :
  (2 * s ≠ 1 → (v = (1 / (1 - 2 * s)) * u^2 + s / 2 ∨
                v = (1 / (1 + 2 * s)) * u^2 - s / 2)) ∧
  (2 * s = 1 → v = u^2 / 2 - 1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_touching_parabolas_l3151_315161


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3151_315159

/-- A quadratic expression ax^2 + bx + c is a perfect square trinomial if there exists a real number k such that ax^2 + bx + c = (kx + r)^2 for some real r. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ k r : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (k * x + r)^2

/-- If 4x^2 + mx + 9 is a perfect square trinomial, then m = 12 or m = -12. -/
theorem perfect_square_trinomial_condition (m : ℝ) :
  IsPerfectSquareTrinomial 4 m 9 → m = 12 ∨ m = -12 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3151_315159


namespace NUMINAMATH_CALUDE_complex_equation_l3151_315112

theorem complex_equation (z : ℂ) (h : Complex.abs z = 1 + 3*I - z) :
  ((1 + I)^2 * (3 + 4*I)^2) / (2 * z) = 3 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_l3151_315112


namespace NUMINAMATH_CALUDE_acme_vowel_soup_sequences_l3151_315174

/-- The number of distinct elements in the set -/
def num_elements : ℕ := 5

/-- The length of the sequences to be formed -/
def sequence_length : ℕ := 7

/-- The maximum number of times each element can be used -/
def max_repetitions : ℕ := 4

/-- The number of possible sequences -/
def num_sequences : ℕ := num_elements ^ sequence_length

theorem acme_vowel_soup_sequences :
  num_sequences = 78125 :=
sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_sequences_l3151_315174


namespace NUMINAMATH_CALUDE_sin_alpha_cos_beta_value_l3151_315186

/-- Given two points on the unit circle representing the terminal sides of angles α and β,
    prove that sin(α) * cos(β) equals a specific value. -/
theorem sin_alpha_cos_beta_value (α β : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = 12/13 ∧ y = 5/13 ∧ 
   Real.cos α = x ∧ Real.sin α = y) →
  (∃ (u v : Real), u^2 + v^2 = 1 ∧ u = -3/5 ∧ v = 4/5 ∧ 
   Real.cos β = u ∧ Real.sin β = v) →
  Real.sin α * Real.cos β = -15/65 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_cos_beta_value_l3151_315186


namespace NUMINAMATH_CALUDE_no_real_solutions_l3151_315191

theorem no_real_solutions : ¬∃ (x y : ℝ), x^2 + 3*y^2 - 4*x - 6*y + 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3151_315191


namespace NUMINAMATH_CALUDE_bens_baseball_card_boxes_l3151_315133

theorem bens_baseball_card_boxes (basketball_boxes : ℕ) (basketball_cards_per_box : ℕ)
  (baseball_cards_per_box : ℕ) (cards_given_away : ℕ) (cards_left : ℕ) :
  basketball_boxes = 4 →
  basketball_cards_per_box = 10 →
  baseball_cards_per_box = 8 →
  cards_given_away = 58 →
  cards_left = 22 →
  (basketball_boxes * basketball_cards_per_box +
    baseball_cards_per_box * ((cards_given_away + cards_left - basketball_boxes * basketball_cards_per_box) / baseball_cards_per_box)) =
  cards_given_away + cards_left →
  (cards_given_away + cards_left - basketball_boxes * basketball_cards_per_box) / baseball_cards_per_box = 5 :=
by sorry

end NUMINAMATH_CALUDE_bens_baseball_card_boxes_l3151_315133


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l3151_315199

/-- Calculates the principal amount of a loan given the interest rate, time, and total interest. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  interest / (rate * time)

/-- Theorem stating that for a loan with 12% annual simple interest rate,
    where the interest after 3 years is $3600, the principal amount is $10,000. -/
theorem loan_principal_calculation :
  let rate : ℚ := 12 / 100
  let time : ℕ := 3
  let interest : ℚ := 3600
  calculate_principal rate time interest = 10000 := by
  sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l3151_315199


namespace NUMINAMATH_CALUDE_average_of_three_angles_l3151_315143

/-- Given that the average of α and β is 105°, prove that the average of α, β, and γ is 80°. -/
theorem average_of_three_angles (α β γ : ℝ) :
  (α + β) / 2 = 105 → (α + β + γ) / 3 = 80 :=
by sorry

end NUMINAMATH_CALUDE_average_of_three_angles_l3151_315143


namespace NUMINAMATH_CALUDE_tan_theta_value_l3151_315183

theorem tan_theta_value (θ : Real) 
  (h1 : 2 * Real.sin θ + Real.cos θ = Real.sqrt 2 / 3)
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.tan θ = -(90 + 5 * Real.sqrt 86) / 168 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l3151_315183


namespace NUMINAMATH_CALUDE_circle_tangent_line_segment_l3151_315170

/-- Given two circles in a plane with radii r₁ and r₂, centered at O₁ and O₂ respectively,
    touching a line at points M₁ and M₂, and lying on the same side of the line,
    if the ratio of M₁M₂ to O₁O₂ is k, then M₁M₂ can be calculated. -/
theorem circle_tangent_line_segment (r₁ r₂ : ℝ) (k : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 7) (h₃ : k = 2 * Real.sqrt 5 / 5) :
  let M₁M₂ := r₁ - r₂
  M₁M₂ * (Real.sqrt (1 - k^2) / k) = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_line_segment_l3151_315170


namespace NUMINAMATH_CALUDE_mcgillicuddy_kindergarten_count_l3151_315109

/-- Calculates the total number of students present in two kindergarten sessions -/
def total_students (morning_registered : ℕ) (morning_absent : ℕ) 
                   (afternoon_registered : ℕ) (afternoon_absent : ℕ) : ℕ :=
  (morning_registered - morning_absent) + (afternoon_registered - afternoon_absent)

/-- Theorem stating that the total number of students is 42 given the specified conditions -/
theorem mcgillicuddy_kindergarten_count : 
  total_students 25 3 24 4 = 42 := by
  sorry

#eval total_students 25 3 24 4

end NUMINAMATH_CALUDE_mcgillicuddy_kindergarten_count_l3151_315109


namespace NUMINAMATH_CALUDE_distance_from_T_to_S_l3151_315123

theorem distance_from_T_to_S (P Q : ℝ) : 
  let S := P + (3/4) * (Q - P)
  let T := P + (1/3) * (Q - P)
  S - T = 25 := by
sorry

end NUMINAMATH_CALUDE_distance_from_T_to_S_l3151_315123


namespace NUMINAMATH_CALUDE_quadratic_comparison_l3151_315106

/-- A quadratic function f(x) = x^2 - 2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m

theorem quadratic_comparison (m : ℝ) (y₁ y₂ : ℝ) 
  (h1 : f m (-1) = y₁)
  (h2 : f m 2 = y₂) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_comparison_l3151_315106


namespace NUMINAMATH_CALUDE_work_duration_l3151_315102

/-- Given that x does a work in 20 days and x and y together do the same work in 40/3 days,
    prove that y does the work in 40 days. -/
theorem work_duration (x y : ℝ) (h1 : x = 20) (h2 : 1 / x + 1 / y = 3 / 40) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_work_duration_l3151_315102


namespace NUMINAMATH_CALUDE_exists_perpendicular_line_l3151_315146

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define a relation for a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define a relation for two lines being perpendicular
variable (perpendicular : Line → Line → Prop)

-- Theorem statement
theorem exists_perpendicular_line (l : Line) (α : Plane) :
  ∃ (m : Line), in_plane m α ∧ perpendicular m l := by
  sorry

end NUMINAMATH_CALUDE_exists_perpendicular_line_l3151_315146


namespace NUMINAMATH_CALUDE_custom_chess_pieces_l3151_315108

theorem custom_chess_pieces (num_players : Nat) (std_pieces_per_player : Nat)
  (missing_queens : Nat) (missing_knights : Nat) (missing_pawns : Nat)
  (h1 : num_players = 3)
  (h2 : std_pieces_per_player = 16)
  (h3 : missing_queens = 2)
  (h4 : missing_knights = 5)
  (h5 : missing_pawns = 8) :
  let total_missing := missing_queens + missing_knights + missing_pawns
  let total_original := num_players * std_pieces_per_player
  let pieces_per_player := (total_original - total_missing) / num_players
  (pieces_per_player = 11) ∧ (total_original - total_missing = 33) := by
  sorry

end NUMINAMATH_CALUDE_custom_chess_pieces_l3151_315108


namespace NUMINAMATH_CALUDE_max_books_borrowed_l3151_315178

/-- Given a college with the following book borrowing statistics:
    - 200 total students
    - 10 students borrowed 0 books
    - 30 students borrowed 1 book each
    - 40 students borrowed 2 books each
    - 50 students borrowed 3 books each
    - 25 students borrowed 5 books each
    - The average number of books per student is 3

    Prove that the maximum number of books any single student could have borrowed is 215. -/
theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (three_books : ℕ) (five_books : ℕ) (avg_books : ℚ) :
  total_students = 200 →
  zero_books = 10 →
  one_book = 30 →
  two_books = 40 →
  three_books = 50 →
  five_books = 25 →
  avg_books = 3 →
  (zero_books + one_book + two_books + three_books + five_books : ℚ) / total_students = avg_books →
  ∃ (max_books : ℕ), max_books = 215 ∧ 
    ∀ (student_books : ℕ), student_books ≤ max_books :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l3151_315178


namespace NUMINAMATH_CALUDE_center_of_gravity_semicircle_semidisk_l3151_315196

/-- The center of gravity of a homogeneous semicircle and semi-disk -/
theorem center_of_gravity_semicircle_semidisk (r : ℝ) (hr : r > 0) :
  ∃ (y z : ℝ),
    y = (2 * r) / Real.pi ∧
    z = (4 * r) / (3 * Real.pi) ∧
    y > 0 ∧ z > 0 :=
by sorry

end NUMINAMATH_CALUDE_center_of_gravity_semicircle_semidisk_l3151_315196


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3151_315107

theorem polynomial_simplification (x : ℝ) : 
  (3*x - 4)*(x + 6) - (x + 3)*(3*x + 2) = 3*x - 30 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3151_315107


namespace NUMINAMATH_CALUDE_problem_solution_l3151_315185

theorem problem_solution : ∃ x : ℤ, x - (28 - (37 - (15 - 16))) = 55 ∧ x = 65 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3151_315185


namespace NUMINAMATH_CALUDE_sum_expression_l3151_315148

-- Define the variables
variable (x y z : ℝ)

-- State the theorem
theorem sum_expression (h1 : y = 3 * x + 1) (h2 : z = y - x) : 
  x + y + z = 6 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_expression_l3151_315148


namespace NUMINAMATH_CALUDE_negation_of_exists_greater_than_one_l3151_315120

theorem negation_of_exists_greater_than_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_greater_than_one_l3151_315120


namespace NUMINAMATH_CALUDE_born_day_300_years_before_l3151_315122

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the day of the week 300 years before a given Monday -/
def dayOfWeek300YearsBefore (endDay : DayOfWeek) : DayOfWeek :=
  match endDay with
  | DayOfWeek.Monday => DayOfWeek.Wednesday
  | _ => DayOfWeek.Monday  -- This case should never occur in our problem

/-- Theorem stating that 300 years before a Monday is a Wednesday -/
theorem born_day_300_years_before (endDay : DayOfWeek) 
  (h : endDay = DayOfWeek.Monday) : 
  dayOfWeek300YearsBefore endDay = DayOfWeek.Wednesday :=
by sorry

#check born_day_300_years_before

end NUMINAMATH_CALUDE_born_day_300_years_before_l3151_315122


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l3151_315147

theorem complex_magnitude_equality (t : ℝ) (h1 : t > 0) :
  Complex.abs (-3 + t * Complex.I) = 3 * Real.sqrt 5 → t = 6 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l3151_315147


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_calculation_l3151_315125

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  let total_distance := train_speed * crossing_time / 3600 * 1000
  total_distance - train_length

/-- Proof of the bridge length calculation -/
theorem bridge_length_calculation : 
  bridge_length 110 45 30 = 265 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_calculation_l3151_315125


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3151_315111

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) :
  A = 25 * Real.pi → d = (2 * (A / Real.pi).sqrt) → d = 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3151_315111


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3151_315104

theorem sum_of_two_numbers (smaller larger : ℕ) : 
  smaller = 31 → larger = 3 * smaller → smaller + larger = 124 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3151_315104


namespace NUMINAMATH_CALUDE_puppies_per_cage_l3151_315103

/-- Given a pet store scenario with puppies and cages, calculate puppies per cage -/
theorem puppies_per_cage 
  (total_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : total_puppies = 45)
  (h2 : sold_puppies = 39)
  (h3 : num_cages = 3)
  (h4 : sold_puppies < total_puppies) :
  (total_puppies - sold_puppies) / num_cages = 2 := by
sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l3151_315103


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l3151_315182

/-- Proves that mixing two varieties of rice in the ratio 1:2.4, where one costs 4.5 per kg 
    and the other costs 8.75 per kg, results in a mixture costing 7.50 per kg. -/
theorem rice_mixture_cost 
  (cost1 : ℝ) (cost2 : ℝ) (mixture_cost : ℝ) 
  (ratio1 : ℝ) (ratio2 : ℝ) :
  cost1 = 4.5 →
  cost2 = 8.75 →
  mixture_cost = 7.50 →
  ratio1 = 1 →
  ratio2 = 2.4 →
  (cost1 * ratio1 + cost2 * ratio2) / (ratio1 + ratio2) = mixture_cost :=
by sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_l3151_315182


namespace NUMINAMATH_CALUDE_vector_statements_false_l3151_315116

open RealInnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def is_unit_vector (v : V) : Prop := ‖v‖ = 1

def parallel (v w : V) : Prop := ∃ (c : ℝ), v = c • w

theorem vector_statements_false (a₀ : V) (h : is_unit_vector a₀) :
  (∃ (a : V), a ≠ ‖a‖ • a₀) ∧
  (∃ (a : V), parallel a a₀ ∧ a ≠ ‖a‖ • a₀) ∧
  (∃ (a : V), parallel a a₀ ∧ is_unit_vector a ∧ a ≠ a₀) := by
  sorry

end NUMINAMATH_CALUDE_vector_statements_false_l3151_315116


namespace NUMINAMATH_CALUDE_happy_valley_kennel_arrangement_l3151_315115

def num_chickens : Nat := 5
def num_dogs : Nat := 2
def num_cats : Nat := 5
def num_rabbits : Nat := 3
def total_animals : Nat := num_chickens + num_dogs + num_cats + num_rabbits

def animal_types : Nat := 4

theorem happy_valley_kennel_arrangement :
  (animal_types.factorial * num_chickens.factorial * num_dogs.factorial * 
   num_cats.factorial * num_rabbits.factorial) = 4147200 := by
  sorry

end NUMINAMATH_CALUDE_happy_valley_kennel_arrangement_l3151_315115


namespace NUMINAMATH_CALUDE_union_complement_equality_l3151_315165

open Set

def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {1, 4}
def N : Finset ℕ := {2, 5}

theorem union_complement_equality : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equality_l3151_315165


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3151_315149

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the theorem
theorem tangent_line_sum (h : ∀ y, y = f 5 ↔ y = -5 + 8) : f 5 + deriv f 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3151_315149


namespace NUMINAMATH_CALUDE_complex_subtraction_l3151_315173

theorem complex_subtraction (z : ℂ) : (5 - 3*I - z = -1 + 4*I) → z = 6 - 7*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3151_315173


namespace NUMINAMATH_CALUDE_star_value_proof_l3151_315171

def star (a b : ℤ) : ℤ := a^2 + 2*a*b + b^2

theorem star_value_proof (a b : ℤ) (h : 4 ∣ (a + b)) : 
  a = 3 → b = 5 → star a b = 64 := by
  sorry

end NUMINAMATH_CALUDE_star_value_proof_l3151_315171


namespace NUMINAMATH_CALUDE_problem_solution_l3151_315138

theorem problem_solution (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = 1) :
  x + x^4 / y^3 + y^4 / x^3 + y = 228498 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3151_315138


namespace NUMINAMATH_CALUDE_probability_of_meeting_l3151_315176

def knockout_tournament (n : ℕ) := n > 1

def num_matches (n : ℕ) (h : knockout_tournament n) : ℕ := n - 1

def num_pairs (n : ℕ) : ℕ := n.choose 2

theorem probability_of_meeting (n : ℕ) (h : knockout_tournament n) :
  (num_matches n h : ℚ) / (num_pairs n : ℚ) = 31 / 496 :=
sorry

end NUMINAMATH_CALUDE_probability_of_meeting_l3151_315176


namespace NUMINAMATH_CALUDE_equation_solution_l3151_315144

theorem equation_solution :
  ∃ (x : ℝ), x ≠ -1 ∧ x ≠ 1 ∧ (x - 1) / (x + 1) - 3 / (x^2 - 1) = 1 ∧ x = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3151_315144


namespace NUMINAMATH_CALUDE_class_average_theorem_l3151_315140

theorem class_average_theorem (total_students : ℕ) (boys_percentage : ℚ) (girls_percentage : ℚ)
  (boys_score : ℚ) (girls_score : ℚ) :
  boys_percentage = 2/5 →
  girls_percentage = 3/5 →
  boys_score = 4/5 →
  girls_score = 9/10 →
  (boys_percentage * boys_score + girls_percentage * girls_score : ℚ) = 43/50 :=
by sorry

end NUMINAMATH_CALUDE_class_average_theorem_l3151_315140


namespace NUMINAMATH_CALUDE_min_value_expression_l3151_315151

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2)) + (y^2 / (x - 2)) ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3151_315151


namespace NUMINAMATH_CALUDE_max_intersection_points_four_spheres_l3151_315172

/-- A sphere in three-dimensional space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- A line in three-dimensional space -/
structure Line where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- The number of intersection points between a line and a sphere -/
def intersectionPoints (l : Line) (s : Sphere) : ℕ := sorry

/-- The theorem stating the maximum number of intersection points -/
theorem max_intersection_points_four_spheres (s₁ s₂ s₃ s₄ : Sphere) :
  ∃ (l : Line), (intersectionPoints l s₁) + (intersectionPoints l s₂) +
                (intersectionPoints l s₃) + (intersectionPoints l s₄) ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_four_spheres_l3151_315172


namespace NUMINAMATH_CALUDE_constant_is_monomial_l3151_315155

/-- A monomial is a constant or a product of variables with non-negative integer exponents. --/
def IsMonomial (x : ℝ) : Prop :=
  x ≠ 0 ∨ ∃ (n : ℕ), x = 1 ∨ x = -1

/-- Theorem: The constant -2010 is a monomial. --/
theorem constant_is_monomial : IsMonomial (-2010) := by
  sorry

end NUMINAMATH_CALUDE_constant_is_monomial_l3151_315155


namespace NUMINAMATH_CALUDE_angle_symmetry_l3151_315194

/-- Two angles are symmetric about the y-axis if their sum is congruent to 180° modulo 360° -/
def symmetric_about_y_axis (α β : Real) : Prop :=
  ∃ k : ℤ, α + β = k * 360 + 180

theorem angle_symmetry (α β : Real) :
  symmetric_about_y_axis α β →
  ∃ k : ℤ, α + β = k * 360 + 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_symmetry_l3151_315194


namespace NUMINAMATH_CALUDE_range_of_a_l3151_315131

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) → -8 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3151_315131


namespace NUMINAMATH_CALUDE_sqrt_four_minus_2023_power_zero_equals_one_l3151_315113

theorem sqrt_four_minus_2023_power_zero_equals_one :
  Real.sqrt 4 - (2023 : ℝ) ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_minus_2023_power_zero_equals_one_l3151_315113


namespace NUMINAMATH_CALUDE_cube_edge_length_is_sqrt_3_l3151_315192

/-- The edge length of a cube inscribed in a sphere with volume 9π/2 --/
def cube_edge_length (s : Real) (c : Real) : Prop :=
  -- All vertices of the cube are on the surface of the sphere
  -- The volume of the sphere is 9π/2
  (4 / 3 * Real.pi * s^3 = 9 * Real.pi / 2) ∧
  -- The space diagonal of the cube is the diameter of the sphere
  (Real.sqrt 3 * c = 2 * s) →
  c = Real.sqrt 3

/-- Theorem stating that the edge length of the cube is √3 --/
theorem cube_edge_length_is_sqrt_3 :
  ∃ (s : Real) (c : Real), cube_edge_length s c :=
by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_is_sqrt_3_l3151_315192


namespace NUMINAMATH_CALUDE_shipping_percentage_above_50_l3151_315124

def flat_rate_shipping : Real := 5.00
def min_purchase_for_percentage : Real := 50.00

def shirt_price : Real := 12.00
def shirt_quantity : Nat := 3
def socks_price : Real := 5.00
def shorts_price : Real := 15.00
def shorts_quantity : Nat := 2
def swim_trunks_price : Real := 14.00

def total_purchase : Real := shirt_price * shirt_quantity + socks_price + shorts_price * shorts_quantity + swim_trunks_price
def total_bill : Real := 102.00

theorem shipping_percentage_above_50 :
  total_purchase > min_purchase_for_percentage →
  (total_bill - total_purchase) / total_purchase * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_shipping_percentage_above_50_l3151_315124


namespace NUMINAMATH_CALUDE_alloy_ratio_l3151_315156

/-- Proves that the ratio of tin to copper in alloy B is 3:5 given the specified conditions -/
theorem alloy_ratio : 
  ∀ (lead_A tin_A tin_B copper_B : ℝ),
  -- Alloy A has 170 kg total
  lead_A + tin_A = 170 →
  -- Alloy A has lead and tin in ratio 1:3
  lead_A * 3 = tin_A →
  -- Alloy B has 250 kg total
  tin_B + copper_B = 250 →
  -- Total tin in new alloy is 221.25 kg
  tin_A + tin_B = 221.25 →
  -- Ratio of tin to copper in alloy B is 3:5
  tin_B * 5 = copper_B * 3 := by
sorry


end NUMINAMATH_CALUDE_alloy_ratio_l3151_315156


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l3151_315181

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : 0 < std_dev

/-- The value that is exactly n standard deviations less than the mean -/
def value_n_std_dev_below (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 14.0 and standard deviation 1.5,
    the value that is exactly 2 standard deviations less than the mean is 11.0 -/
theorem two_std_dev_below_mean :
  ∃ (d : NormalDistribution),
    d.mean = 14.0 ∧
    d.std_dev = 1.5 ∧
    value_n_std_dev_below d 2 = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l3151_315181


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_range_a_range_l3151_315127

/-- A function f(x) = x^3 - ax that is monotonically decreasing on (-1/2, 0) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

/-- The property of f being monotonically decreasing on (-1/2, 0) -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, -1/2 < x ∧ x < y ∧ y < 0 → f a x > f a y

/-- The theorem stating that if f is monotonically decreasing on (-1/2, 0), then a ≥ 3/4 -/
theorem monotone_decreasing_implies_a_range (a : ℝ) :
  is_monotone_decreasing a → a ≥ 3/4 := by sorry

/-- The main theorem proving the range of a -/
theorem a_range : 
  {a : ℝ | is_monotone_decreasing a} = {a : ℝ | a ≥ 3/4} := by sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_range_a_range_l3151_315127


namespace NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l3151_315164

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n ∧ a n > 0

/-- An arithmetic sequence -/
def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n, b (n + 1) = b n + d

theorem geometric_arithmetic_inequality 
  (a b : ℕ → ℝ) 
  (ha : is_positive_geometric_sequence a) 
  (hb : is_arithmetic_sequence b) 
  (h_eq : a 6 = b 7) : 
  a 3 + a 9 ≥ b 4 + b 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l3151_315164


namespace NUMINAMATH_CALUDE_storage_tub_cost_l3151_315119

/-- The cost of storage tubs problem -/
theorem storage_tub_cost (total_cost : ℕ) (num_large : ℕ) (num_small : ℕ) (small_cost : ℕ) :
  total_cost = 48 →
  num_large = 3 →
  num_small = 6 →
  small_cost = 5 →
  ∃ (large_cost : ℕ), num_large * large_cost + num_small * small_cost = total_cost ∧ large_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_storage_tub_cost_l3151_315119


namespace NUMINAMATH_CALUDE_sqrt_three_minus_one_over_two_gt_one_third_l3151_315136

theorem sqrt_three_minus_one_over_two_gt_one_third : (Real.sqrt 3 - 1) / 2 > 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_one_over_two_gt_one_third_l3151_315136


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l3151_315100

theorem complex_imaginary_part (z : ℂ) : 
  z = -2 + I → Complex.im (z + z⁻¹) = 4/5 := by sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l3151_315100


namespace NUMINAMATH_CALUDE_investment_rate_proof_l3151_315117

/-- Proves that given an investment scenario, the unknown rate is 1% -/
theorem investment_rate_proof (total_investment : ℝ) (amount_at_10_percent : ℝ) (total_interest : ℝ)
  (h1 : total_investment = 31000)
  (h2 : amount_at_10_percent = 12000)
  (h3 : total_interest = 1390)
  : (total_interest - 0.1 * amount_at_10_percent) / (total_investment - amount_at_10_percent) = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l3151_315117


namespace NUMINAMATH_CALUDE_right_angle_on_circle_l3151_315101

/-- The circle C with equation (x - √3)² + (y - 1)² = 1 -/
def circle_C (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + (y - 1)^2 = 1

/-- The point A with coordinates (-t, 0) -/
def point_A (t : ℝ) : ℝ × ℝ := (-t, 0)

/-- The point B with coordinates (t, 0) -/
def point_B (t : ℝ) : ℝ × ℝ := (t, 0)

/-- Predicate to check if a point P forms a right angle with A and B -/
def forms_right_angle (P : ℝ × ℝ) (t : ℝ) : Prop :=
  let A := point_A t
  let B := point_B t
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

theorem right_angle_on_circle (t : ℝ) :
  t > 0 →
  (∃ P : ℝ × ℝ, circle_C P.1 P.2 ∧ forms_right_angle P t) →
  t ∈ Set.Icc 1 3 :=
sorry

end NUMINAMATH_CALUDE_right_angle_on_circle_l3151_315101


namespace NUMINAMATH_CALUDE_sin_shift_l3151_315184

theorem sin_shift (x : ℝ) : Real.sin (2 * x + π / 3) = Real.sin (2 * (x + π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l3151_315184


namespace NUMINAMATH_CALUDE_even_count_pascal_triangle_l3151_315145

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Check if a natural number is even -/
def isEven (n : ℕ) : Bool := sorry

/-- Count even binomial coefficients in a single row of Pascal's Triangle -/
def countEvenInRow (row : ℕ) : ℕ := sorry

/-- Count even binomial coefficients in the first n rows of Pascal's Triangle -/
def countEvenInTriangle (n : ℕ) : ℕ := sorry

/-- The number of even integers in the top 15 rows of Pascal's Triangle is 84 -/
theorem even_count_pascal_triangle : countEvenInTriangle 15 = 84 := by sorry

end NUMINAMATH_CALUDE_even_count_pascal_triangle_l3151_315145


namespace NUMINAMATH_CALUDE_negation_equivalence_l3151_315160

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 3*x + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3151_315160


namespace NUMINAMATH_CALUDE_circuit_current_l3151_315168

/-- Given a voltage V and impedance Z as complex numbers,
    prove that the current I = V / Z equals the expected value. -/
theorem circuit_current (V Z : ℂ) (hV : V = 2 + 3*I) (hZ : Z = 4 - 2*I) :
  V / Z = (1 / 10 : ℂ) + (4 / 5 : ℂ) * I :=
by sorry

end NUMINAMATH_CALUDE_circuit_current_l3151_315168


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3151_315105

theorem contrapositive_equivalence (x : ℝ) :
  (¬(-1 < x ∧ x < 0) ∨ x^2 < 1) ↔ (x^2 ≥ 1 → (x ≥ 0 ∨ x ≤ -1)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3151_315105

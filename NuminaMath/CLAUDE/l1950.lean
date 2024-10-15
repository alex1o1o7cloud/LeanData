import Mathlib

namespace NUMINAMATH_CALUDE_inequality_implication_l1950_195072

theorem inequality_implication (m n : ℝ) (h : m > n) : -3*m < -3*n := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1950_195072


namespace NUMINAMATH_CALUDE_original_jellybeans_proof_l1950_195030

/-- The original number of jellybeans in Jenny's jar -/
def original_jellybeans : ℕ := 50

/-- The fraction of jellybeans remaining after each day -/
def daily_remaining_fraction : ℚ := 4/5

/-- The number of days that have passed -/
def days_passed : ℕ := 2

/-- The number of jellybeans remaining after two days -/
def remaining_jellybeans : ℕ := 32

/-- Theorem stating that the original number of jellybeans is correct -/
theorem original_jellybeans_proof :
  (daily_remaining_fraction ^ days_passed) * original_jellybeans = remaining_jellybeans := by
  sorry

end NUMINAMATH_CALUDE_original_jellybeans_proof_l1950_195030


namespace NUMINAMATH_CALUDE_reaction_enthalpy_change_l1950_195001

/-- Represents the enthalpy change for a chemical reaction --/
def enthalpy_change (bonds_broken bonds_formed : ℝ) : ℝ :=
  bonds_broken - bonds_formed

/-- Bond dissociation energy for CH3-CH2 (C-C) bond --/
def e_cc : ℝ := 347

/-- Bond dissociation energy for CH3-O (C-O) bond --/
def e_co : ℝ := 358

/-- Bond dissociation energy for CH2-OH (O-H) bond --/
def e_oh_alcohol : ℝ := 463

/-- Bond dissociation energy for C=O (COOH) bond --/
def e_co_double : ℝ := 745

/-- Bond dissociation energy for O-H (COOH) bond --/
def e_oh_acid : ℝ := 467

/-- Bond dissociation energy for O=O (O2) bond --/
def e_oo : ℝ := 498

/-- Bond dissociation energy for O-H (H2O) bond --/
def e_oh_water : ℝ := 467

/-- Total energy of bonds broken in reactants --/
def bonds_broken : ℝ := e_cc + e_co + e_oh_alcohol + 1.5 * e_oo

/-- Total energy of bonds formed in products --/
def bonds_formed : ℝ := e_co_double + e_oh_acid + e_oh_water

/-- Theorem stating the enthalpy change for the given reaction --/
theorem reaction_enthalpy_change :
  enthalpy_change bonds_broken bonds_formed = 236 := by
  sorry

end NUMINAMATH_CALUDE_reaction_enthalpy_change_l1950_195001


namespace NUMINAMATH_CALUDE_smallest_n_for_g_equals_seven_g_of_eight_equals_seven_l1950_195053

def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 1 else n/2 + 3

theorem smallest_n_for_g_equals_seven :
  ∀ n : ℕ, n > 0 → g n = 7 → n ≥ 8 :=
by sorry

theorem g_of_eight_equals_seven : g 8 = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_g_equals_seven_g_of_eight_equals_seven_l1950_195053


namespace NUMINAMATH_CALUDE_matrix_identity_l1950_195057

variable {n : Type*} [DecidableEq n] [Fintype n]

theorem matrix_identity (A : Matrix n n ℝ) (h_inv : IsUnit A) 
  (h_eq : (A - 3 • (1 : Matrix n n ℝ)) * (A - 5 • (1 : Matrix n n ℝ)) = 0) :
  A + 8 • A⁻¹ = (7 • A + 64 • (1 : Matrix n n ℝ)) / 15 := by
  sorry

end NUMINAMATH_CALUDE_matrix_identity_l1950_195057


namespace NUMINAMATH_CALUDE_golden_state_total_points_l1950_195032

/-- The Golden State Team's total points calculation -/
theorem golden_state_total_points :
  let draymond_points : ℕ := 12
  let curry_points : ℕ := 2 * draymond_points
  let kelly_points : ℕ := 9
  let durant_points : ℕ := 2 * kelly_points
  let klay_points : ℕ := draymond_points / 2
  draymond_points + curry_points + kelly_points + durant_points + klay_points = 69 := by
  sorry

end NUMINAMATH_CALUDE_golden_state_total_points_l1950_195032


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1950_195086

theorem diophantine_equation_solutions :
  ∀ a b : ℕ, a^2 = b * (b + 7) → (a = 12 ∧ b = 9) ∨ (a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1950_195086


namespace NUMINAMATH_CALUDE_winning_pair_probability_l1950_195013

-- Define the deck
def deck_size : ℕ := 9
def num_colors : ℕ := 3
def num_letters : ℕ := 3

-- Define a winning pair
def is_winning_pair (card1 card2 : ℕ × ℕ) : Prop :=
  (card1.1 = card2.1) ∨ (card1.2 = card2.2)

-- Define the probability of drawing a winning pair
def prob_winning_pair : ℚ :=
  (num_colors * (num_letters.choose 2) + num_letters * (num_colors.choose 2)) / deck_size.choose 2

-- Theorem statement
theorem winning_pair_probability : prob_winning_pair = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_winning_pair_probability_l1950_195013


namespace NUMINAMATH_CALUDE_expression_evaluation_l1950_195096

theorem expression_evaluation :
  ∃ m : ℤ, (3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2 = m * 10^1003 ∧ m = 1372 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1950_195096


namespace NUMINAMATH_CALUDE_digit_200_of_17_over_70_is_2_l1950_195091

/-- The 200th digit after the decimal point in the decimal representation of 17/70 -/
def digit_200_of_17_over_70 : ℕ := 2

/-- Theorem stating that the 200th digit after the decimal point in 17/70 is 2 -/
theorem digit_200_of_17_over_70_is_2 :
  digit_200_of_17_over_70 = 2 := by sorry

end NUMINAMATH_CALUDE_digit_200_of_17_over_70_is_2_l1950_195091


namespace NUMINAMATH_CALUDE_sin_36_degrees_l1950_195037

theorem sin_36_degrees : 
  Real.sin (36 * π / 180) = (1 / 4) * Real.sqrt (10 - 2 * Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_sin_36_degrees_l1950_195037


namespace NUMINAMATH_CALUDE_marias_stamp_collection_l1950_195092

/-- The problem of calculating Maria's stamp collection increase -/
theorem marias_stamp_collection 
  (current_stamps : ℕ) 
  (increase_percentage : ℚ) 
  (h1 : current_stamps = 40)
  (h2 : increase_percentage = 20 / 100) : 
  current_stamps + (increase_percentage * current_stamps).floor = 48 := by
  sorry

end NUMINAMATH_CALUDE_marias_stamp_collection_l1950_195092


namespace NUMINAMATH_CALUDE_exam_class_size_l1950_195061

/-- Represents a class of students with their exam marks. -/
structure ExamClass where
  totalStudents : ℕ
  averageMark : ℚ
  excludedStudents : ℕ
  excludedAverage : ℚ
  remainingAverage : ℚ

/-- Theorem stating the number of students in the class given the conditions. -/
theorem exam_class_size (c : ExamClass)
  (h1 : c.averageMark = 80)
  (h2 : c.excludedStudents = 5)
  (h3 : c.excludedAverage = 50)
  (h4 : c.remainingAverage = 90)
  (h5 : c.totalStudents * c.averageMark = 
        (c.totalStudents - c.excludedStudents) * c.remainingAverage + 
        c.excludedStudents * c.excludedAverage) :
  c.totalStudents = 20 := by
  sorry


end NUMINAMATH_CALUDE_exam_class_size_l1950_195061


namespace NUMINAMATH_CALUDE_min_value_product_quotient_equality_condition_l1950_195082

theorem min_value_product_quotient (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2*a + 1) * (b^2 + 2*b + 1) * (c^2 + 2*c + 1) / (a * b * c) ≥ 64 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2*a + 1) * (b^2 + 2*b + 1) * (c^2 + 2*c + 1) / (a * b * c) = 64 ↔ a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_equality_condition_l1950_195082


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l1950_195068

-- Define the constants and variables
variable (a b c k : ℝ)
variable (y₁ y₂ y₃ : ℝ)

-- State the theorem
theorem inverse_proportion_order (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) 
  (h4 : k > 0)
  (h5 : y₁ = k / (a - b))
  (h6 : y₂ = k / (a - c))
  (h7 : y₃ = k / (c - a)) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l1950_195068


namespace NUMINAMATH_CALUDE_part_one_part_two_l1950_195015

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides opposite to angles A, B, C respectively

-- Define the conditions
variable (h1 : 0 < A) -- A is acute
variable (h2 : A < π / 2) -- A is acute
variable (h3 : 3 * b = 5 * a * Real.sin B) -- Given condition

-- Part 1
theorem part_one : 
  Real.sin (2 * A) + Real.cos ((B + C) / 2) ^ 2 = 53 / 50 := by sorry

-- Part 2
theorem part_two (h4 : a = Real.sqrt 2) (h5 : 1 / 2 * b * c * Real.sin A = 3 / 2) :
  b = Real.sqrt 5 ∧ c = Real.sqrt 5 := by sorry

end

end NUMINAMATH_CALUDE_part_one_part_two_l1950_195015


namespace NUMINAMATH_CALUDE_longest_side_of_special_rectangle_l1950_195081

/-- Given a rectangle with perimeter 240 feet and area equal to eight times its perimeter,
    the length of its longest side is 80 feet. -/
theorem longest_side_of_special_rectangle : 
  ∀ l w : ℝ,
  l > 0 → w > 0 →
  2 * l + 2 * w = 240 →
  l * w = 8 * (2 * l + 2 * w) →
  max l w = 80 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_special_rectangle_l1950_195081


namespace NUMINAMATH_CALUDE_expression_is_integer_l1950_195083

theorem expression_is_integer (m : ℕ+) : ∃ k : ℤ, (m^4 / 24 : ℚ) + (m^3 / 4 : ℚ) + (11 * m^2 / 24 : ℚ) + (m / 4 : ℚ) = k := by
  sorry

end NUMINAMATH_CALUDE_expression_is_integer_l1950_195083


namespace NUMINAMATH_CALUDE_equal_parts_complex_l1950_195049

/-- A complex number is an "equal parts complex number" if its real and imaginary parts are equal -/
def is_equal_parts (z : ℂ) : Prop := z.re = z.im

/-- Given that Z = (1+ai)i is an "equal parts complex number", prove that a = -1 -/
theorem equal_parts_complex (a : ℝ) :
  is_equal_parts ((1 + a * Complex.I) * Complex.I) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_equal_parts_complex_l1950_195049


namespace NUMINAMATH_CALUDE_person_height_from_shadow_ratio_l1950_195044

/-- Proves that given a tree's height and shadow length, and a person's shadow length,
    we can determine the person's height assuming a constant ratio of height to shadow length. -/
theorem person_height_from_shadow_ratio (tree_height tree_shadow person_shadow : ℝ) 
  (h1 : tree_height = 50) 
  (h2 : tree_shadow = 25)
  (h3 : person_shadow = 20) :
  (tree_height / tree_shadow) * person_shadow = 40 := by
  sorry

end NUMINAMATH_CALUDE_person_height_from_shadow_ratio_l1950_195044


namespace NUMINAMATH_CALUDE_logarithmic_function_properties_l1950_195063

-- Define the logarithmic function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem statement
theorem logarithmic_function_properties :
  -- f(x) passes through (8,3)
  f 8 = 3 →
  -- f(x) is a logarithmic function (this is implied by its definition)
  -- Prove the following:
  (-- 1. f(x) = log₂(x) (this is true by definition of f)
   -- 2. The domain of f(x) is (0, +∞)
   (∀ x : ℝ, x > 0 ↔ f x ≠ 0) ∧
   -- 3. For f(1-x) > f(1+x), x ∈ (-1, 0)
   (∀ x : ℝ, f (1 - x) > f (1 + x) ↔ -1 < x ∧ x < 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_logarithmic_function_properties_l1950_195063


namespace NUMINAMATH_CALUDE_ellipse_h_plus_k_l1950_195077

/-- An ellipse with foci at (1, 2) and (4, 2), passing through (-1, 5) -/
structure Ellipse where
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ
  /-- A point on the ellipse -/
  point : ℝ × ℝ
  /-- The center of the ellipse -/
  center : ℝ × ℝ
  /-- The semi-major axis length -/
  a : ℝ
  /-- The semi-minor axis length -/
  b : ℝ
  /-- Constraint: focus1 is at (1, 2) -/
  focus1_def : focus1 = (1, 2)
  /-- Constraint: focus2 is at (4, 2) -/
  focus2_def : focus2 = (4, 2)
  /-- Constraint: point is at (-1, 5) -/
  point_def : point = (-1, 5)
  /-- Constraint: center is the midpoint of foci -/
  center_def : center = ((focus1.1 + focus2.1) / 2, (focus1.2 + focus2.2) / 2)
  /-- Constraint: a is positive -/
  a_pos : a > 0
  /-- Constraint: b is positive -/
  b_pos : b > 0
  /-- Constraint: sum of distances from point to foci equals 2a -/
  sum_distances : Real.sqrt ((point.1 - focus1.1)^2 + (point.2 - focus1.2)^2) +
                  Real.sqrt ((point.1 - focus2.1)^2 + (point.2 - focus2.2)^2) = 2 * a

/-- Theorem: The sum of h and k in the standard form equation of the ellipse is 4.5 -/
theorem ellipse_h_plus_k (e : Ellipse) : e.center.1 + e.center.2 = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_h_plus_k_l1950_195077


namespace NUMINAMATH_CALUDE_f_analytical_expression_l1950_195084

def f : Set ℝ := {x : ℝ | x ≠ -1}

theorem f_analytical_expression :
  ∀ x : ℝ, x ∈ f ↔ x ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_f_analytical_expression_l1950_195084


namespace NUMINAMATH_CALUDE_shekars_english_score_l1950_195014

/-- Given Shekar's scores in four subjects and his average score, prove his English score --/
theorem shekars_english_score
  (math_score science_score social_score biology_score : ℕ)
  (average_score : ℚ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : social_score = 82)
  (h4 : biology_score = 85)
  (h5 : average_score = 71)
  (h6 : (math_score + science_score + social_score + biology_score + english_score : ℚ) / 5 = average_score) :
  english_score = 47 :=
by sorry

end NUMINAMATH_CALUDE_shekars_english_score_l1950_195014


namespace NUMINAMATH_CALUDE_candy_bars_purchased_l1950_195043

theorem candy_bars_purchased (total_cost : ℕ) (price_per_bar : ℕ) (h1 : total_cost = 6) (h2 : price_per_bar = 3) :
  total_cost / price_per_bar = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bars_purchased_l1950_195043


namespace NUMINAMATH_CALUDE_solve_system_l1950_195041

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  q = -25 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l1950_195041


namespace NUMINAMATH_CALUDE_ratio_of_55_to_11_l1950_195090

theorem ratio_of_55_to_11 : 
  let certain_number : ℚ := 55
  let ratio := certain_number / 11
  ratio = 5 / 1 := by sorry

end NUMINAMATH_CALUDE_ratio_of_55_to_11_l1950_195090


namespace NUMINAMATH_CALUDE_characterize_equal_prime_factors_l1950_195035

/-- The set of prime factors of a positive integer n -/
def primeDivisors (n : ℕ) : Set ℕ := sorry

theorem characterize_equal_prime_factors :
  ∀ (a m n : ℕ),
    a > 1 →
    m < n →
    (primeDivisors (a^m - 1) = primeDivisors (a^n - 1)) ↔
    (∃ l : ℕ, l ≥ 2 ∧ a = 2^l - 1 ∧ m = 1 ∧ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_characterize_equal_prime_factors_l1950_195035


namespace NUMINAMATH_CALUDE_third_grade_sample_size_l1950_195099

theorem third_grade_sample_size
  (total_students : ℕ)
  (first_grade_students : ℕ)
  (sample_size : ℕ)
  (second_grade_sample_ratio : ℚ)
  (h1 : total_students = 2800)
  (h2 : first_grade_students = 910)
  (h3 : sample_size = 40)
  (h4 : second_grade_sample_ratio = 3 / 10)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_third_grade_sample_size_l1950_195099


namespace NUMINAMATH_CALUDE_cloth_sales_worth_l1950_195010

-- Define the commission rate as a percentage
def commission_rate : ℚ := 2.5

-- Define the commission earned on a particular day
def commission_earned : ℚ := 15

-- Define the function to calculate the total sales
def total_sales (rate : ℚ) (commission : ℚ) : ℚ :=
  commission / (rate / 100)

-- Theorem statement
theorem cloth_sales_worth :
  total_sales commission_rate commission_earned = 600 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sales_worth_l1950_195010


namespace NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l1950_195075

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l1950_195075


namespace NUMINAMATH_CALUDE_smallest_prime_twelve_less_prime_square_l1950_195022

theorem smallest_prime_twelve_less_prime_square : ∃ (p n : ℕ), 
  p = 13 ∧ 
  Nat.Prime p ∧ 
  Nat.Prime n ∧ 
  p = n^2 - 12 ∧
  ∀ (q m : ℕ), Nat.Prime q ∧ Nat.Prime m ∧ q = m^2 - 12 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_twelve_less_prime_square_l1950_195022


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1950_195000

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * f (x + y)) = f (y * f x) + x^2) →
  (∀ x : ℝ, f x = x ∨ f x = -x) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1950_195000


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1950_195052

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1 > 0) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1950_195052


namespace NUMINAMATH_CALUDE_segment_length_l1950_195078

/-- Given a line segment AB with points P and Q, prove that AB has length 35 -/
theorem segment_length (A B P Q : ℝ × ℝ) : 
  (P.1 - A.1) / (B.1 - A.1) = 1 / 5 →  -- P divides AB in ratio 1:4
  (Q.1 - A.1) / (B.1 - A.1) = 2 / 7 →  -- Q divides AB in ratio 2:5
  abs (Q.1 - P.1) = 3 →                -- Distance between P and Q is 3
  abs (B.1 - A.1) = 35 := by            -- Length of AB is 35
sorry

end NUMINAMATH_CALUDE_segment_length_l1950_195078


namespace NUMINAMATH_CALUDE_zoo_animal_difference_l1950_195045

def zoo_problem (parrots snakes monkeys elephants zebras : ℕ) : Prop :=
  (parrots = 8) ∧
  (snakes = 3 * parrots) ∧
  (monkeys = 2 * snakes) ∧
  (elephants = (parrots + snakes) / 2) ∧
  (zebras = elephants - 3)

theorem zoo_animal_difference :
  ∀ parrots snakes monkeys elephants zebras : ℕ,
  zoo_problem parrots snakes monkeys elephants zebras →
  monkeys - zebras = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_difference_l1950_195045


namespace NUMINAMATH_CALUDE_max_sum_of_other_roots_l1950_195076

/-- Given a polynomial x^3 - kx^2 + 20x - 15 with 3 roots, one of which is 3,
    the sum of the other two roots is at most 5. -/
theorem max_sum_of_other_roots (k : ℝ) :
  let p : ℝ → ℝ := λ x => x^3 - k*x^2 + 20*x - 15
  ∃ (r₁ r₂ : ℝ), (p 3 = 0 ∧ p r₁ = 0 ∧ p r₂ = 0) → r₁ + r₂ ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_other_roots_l1950_195076


namespace NUMINAMATH_CALUDE_smallest_shift_for_scaled_function_l1950_195019

-- Define a periodic function with period 30
def isPeriodic30 (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 30) = g x

-- Define the property we're looking for
def hasProperty (g : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, g ((x - b) / 3) = g (x / 3)

theorem smallest_shift_for_scaled_function (g : ℝ → ℝ) (h : isPeriodic30 g) :
  ∃ b : ℝ, b > 0 ∧ hasProperty g b ∧ ∀ b' : ℝ, b' > 0 → hasProperty g b' → b ≤ b' :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_for_scaled_function_l1950_195019


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_for_surveys_l1950_195017

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents a survey with its characteristics -/
structure Survey where
  totalPopulation : ℕ
  sampleSize : ℕ
  hasDistinctGroups : Bool
  hasSmallDifferences : Bool

/-- Determines the optimal sampling method for a given survey -/
def optimalSamplingMethod (s : Survey) : SamplingMethod :=
  if s.hasDistinctGroups then SamplingMethod.Stratified
  else if s.hasSmallDifferences && s.sampleSize < 10 then SamplingMethod.SimpleRandom
  else SamplingMethod.Systematic

/-- The main theorem stating the optimal sampling methods for the two surveys -/
theorem optimal_sampling_methods_for_surveys :
  let survey1 : Survey := {
    totalPopulation := 500,
    sampleSize := 100,
    hasDistinctGroups := true,
    hasSmallDifferences := false
  }
  let survey2 : Survey := {
    totalPopulation := 15,
    sampleSize := 3,
    hasDistinctGroups := false,
    hasSmallDifferences := true
  }
  (optimalSamplingMethod survey1 = SamplingMethod.Stratified) ∧
  (optimalSamplingMethod survey2 = SamplingMethod.SimpleRandom) :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_sampling_methods_for_surveys_l1950_195017


namespace NUMINAMATH_CALUDE_area_of_region_R_l1950_195070

/-- A square with side length 3 -/
structure Square :=
  (side_length : ℝ)
  (is_three : side_length = 3)

/-- The region R in the square -/
def region_R (s : Square) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1^2 + p.2^2 ≤ (3*Real.sqrt 2/2)^2}

/-- The area of a region -/
noncomputable def area (r : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem to be proved -/
theorem area_of_region_R (s : Square) : area (region_R s) = 9 * Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_R_l1950_195070


namespace NUMINAMATH_CALUDE_dog_grouping_theorem_l1950_195006

/-- The number of ways to divide 12 dogs into specified groups -/
def dog_grouping_ways : ℕ :=
  let total_dogs : ℕ := 12
  let group1_size : ℕ := 4  -- Fluffy's group
  let group2_size : ℕ := 5  -- Nipper's group
  let group3_size : ℕ := 3
  let remaining_dogs : ℕ := total_dogs - 2  -- Excluding Fluffy and Nipper
  let ways_to_fill_group1 : ℕ := Nat.choose remaining_dogs (group1_size - 1)
  let ways_to_fill_group2 : ℕ := Nat.choose (remaining_dogs - (group1_size - 1)) (group2_size - 1)
  ways_to_fill_group1 * ways_to_fill_group2

/-- Theorem stating the number of ways to divide the dogs into groups -/
theorem dog_grouping_theorem : dog_grouping_ways = 4200 := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_theorem_l1950_195006


namespace NUMINAMATH_CALUDE_rectangular_solid_on_sphere_l1950_195042

theorem rectangular_solid_on_sphere (x : ℝ) : 
  let surface_area : ℝ := 18 * Real.pi
  let radius : ℝ := Real.sqrt (surface_area / (4 * Real.pi))
  3^2 + 2^2 + x^2 = 4 * radius^2 → x = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_on_sphere_l1950_195042


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1950_195067

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1950_195067


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l1950_195095

theorem prime_pairs_divisibility (p q : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ 
  (p ∣ 5^q + 1) ∧ (q ∣ 5^p + 1) ↔ 
  ((p = 2 ∧ q = 13) ∨ (p = 13 ∧ q = 2) ∨ (p = 3 ∧ q = 7) ∨ (p = 7 ∧ q = 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l1950_195095


namespace NUMINAMATH_CALUDE_vector_relations_l1950_195036

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Define vector a -/
def a : Vector2D := ⟨1, 1⟩

/-- Define vector b with parameter m -/
def b (m : ℝ) : Vector2D := ⟨2, m⟩

/-- Two vectors are parallel if their components are proportional -/
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : Vector2D) : Prop :=
  v.x * w.x + v.y * w.y = 0

/-- Main theorem -/
theorem vector_relations :
  (∀ m : ℝ, parallel a (b m) → m = 2) ∧
  (∀ m : ℝ, perpendicular a (b m) → m = -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_relations_l1950_195036


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l1950_195021

theorem largest_n_for_equation : ∃ (x y z : ℕ+), 
  (100 : ℤ) = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 12 ∧ 
  ∀ (n : ℕ+), n > 10 → ¬∃ (a b c : ℕ+), 
    (n^2 : ℤ) = a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a + 5*a + 5*b + 5*c - 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l1950_195021


namespace NUMINAMATH_CALUDE_solve_for_k_l1950_195007

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 6

def g (k x : ℝ) : ℝ := x^2 - k * x - 8

theorem solve_for_k : ∃ k : ℝ, f 5 - g k 5 = 20 ∧ k = -10.8 := by sorry

end NUMINAMATH_CALUDE_solve_for_k_l1950_195007


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1950_195033

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^3 * (b + c) + b^3 * (c + a) + c^3 * (a + b)) / ((a + b + c)^4 - 79 * (a * b * c)^(4/3))
  A ≤ 3 ∧ ∃ (x : ℝ), x > 0 ∧ 
    let A' := (3 * x^3 * (2 * x)) / ((3 * x)^4 - 79 * x^4)
    A' = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1950_195033


namespace NUMINAMATH_CALUDE_complete_square_equivalence_l1950_195058

theorem complete_square_equivalence :
  let f₁ : ℝ → ℝ := λ x ↦ x^2 - 2*x + 3
  let f₂ : ℝ → ℝ := λ x ↦ 3*x^2 + 6*x - 1
  let f₃ : ℝ → ℝ := λ x ↦ -2*x^2 + 3*x - 2
  let g₁ : ℝ → ℝ := λ x ↦ (x - 1)^2 + 2
  let g₂ : ℝ → ℝ := λ x ↦ 3*(x + 1)^2 - 4
  let g₃ : ℝ → ℝ := λ x ↦ -2*(x - 3/4)^2 - 7/8
  (∀ x : ℝ, f₁ x = g₁ x) ∧
  (∀ x : ℝ, f₂ x = g₂ x) ∧
  (∀ x : ℝ, f₃ x = g₃ x) := by
  sorry

end NUMINAMATH_CALUDE_complete_square_equivalence_l1950_195058


namespace NUMINAMATH_CALUDE_no_rabbits_perished_l1950_195074

/-- Represents the farm with animals before and after the disease outbreak -/
structure Farm where
  initial_count : ℕ  -- Initial count of each animal type
  surviving_cows : ℕ
  surviving_pigs : ℕ
  surviving_horses : ℕ
  surviving_rabbits : ℕ

/-- The conditions of the farm after the disease outbreak -/
def farm_conditions (f : Farm) : Prop :=
  -- Initially equal number of each animal
  f.initial_count > 0 ∧
  -- One out of every five cows died
  f.surviving_cows = (4 * f.initial_count) / 5 ∧
  -- Number of horses that died equals number of pigs that survived
  f.surviving_horses = f.initial_count - f.surviving_pigs ∧
  -- Proportion of rabbits among survivors is 5/14
  14 * f.surviving_rabbits = 5 * (f.surviving_cows + f.surviving_pigs + f.surviving_horses + f.surviving_rabbits)

/-- The theorem to prove -/
theorem no_rabbits_perished (f : Farm) (h : farm_conditions f) : 
  f.surviving_rabbits = f.initial_count := by
  sorry

end NUMINAMATH_CALUDE_no_rabbits_perished_l1950_195074


namespace NUMINAMATH_CALUDE_cook_selection_l1950_195031

theorem cook_selection (total : ℕ) (vegetarians : ℕ) (cooks : ℕ) :
  total = 10 → vegetarians = 3 → cooks = 2 →
  (Nat.choose vegetarians 1) * (Nat.choose (total - 1) 1) = 27 :=
by sorry

end NUMINAMATH_CALUDE_cook_selection_l1950_195031


namespace NUMINAMATH_CALUDE_allan_bought_three_balloons_l1950_195003

/-- The number of balloons Allan bought at the park -/
def balloons_bought_at_park (initial_balloons final_balloons : ℕ) : ℕ :=
  final_balloons - initial_balloons

/-- Theorem stating that Allan bought 3 balloons at the park -/
theorem allan_bought_three_balloons :
  balloons_bought_at_park 5 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_allan_bought_three_balloons_l1950_195003


namespace NUMINAMATH_CALUDE_milk_for_12_cookies_l1950_195024

/-- The number of cookies that can be baked with 5 liters of milk -/
def cookies_per_5_liters : ℕ := 30

/-- The number of cups in a liter -/
def cups_per_liter : ℕ := 4

/-- The number of cookies we want to bake -/
def target_cookies : ℕ := 12

/-- The function that calculates the number of cups of milk needed for a given number of cookies -/
def milk_needed (cookies : ℕ) : ℚ :=
  (cookies * cups_per_liter * 5 : ℚ) / cookies_per_5_liters

theorem milk_for_12_cookies :
  milk_needed target_cookies = 8 := by sorry

end NUMINAMATH_CALUDE_milk_for_12_cookies_l1950_195024


namespace NUMINAMATH_CALUDE_petya_max_spend_l1950_195098

/-- Represents the cost of a book in rubles -/
def BookCost := ℕ

/-- Represents Petya's purchasing behavior -/
structure PetyaPurchase where
  initialMoney : ℕ  -- Initial amount of money Petya had
  expensiveBookThreshold : ℕ  -- Threshold for expensive books (100 rubles)
  spentHalf : Bool  -- Whether Petya spent exactly half of his money

/-- Theorem stating that Petya couldn't have spent 5000 rubles or more on books -/
theorem petya_max_spend (purchase : PetyaPurchase) : 
  purchase.spentHalf → purchase.expensiveBookThreshold = 100 →
  ∃ (maxSpend : ℕ), maxSpend < 5000 ∧ 
  ∀ (actualSpend : ℕ), actualSpend ≤ maxSpend :=
sorry

end NUMINAMATH_CALUDE_petya_max_spend_l1950_195098


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1950_195064

theorem hyperbola_equation (ellipse : Real → Real → Prop)
  (hyperbola : Real → Real → Prop)
  (h1 : ∀ x y, ellipse x y ↔ x^2/27 + y^2/36 = 1)
  (h2 : ∃ x, hyperbola x 4 ∧ ellipse x 4)
  (h3 : ∀ x y, hyperbola x y → (x = 0 → y^2 = 9) ∧ (y = 0 → x^2 = 9)) :
  ∀ x y, hyperbola x y ↔ y^2/4 - x^2/5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1950_195064


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1950_195062

-- Define the points
def A : ℝ × ℝ := (-6, -1)
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (-3, -2)

-- Define the parallelogram property
def is_parallelogram (A B C M : ℝ × ℝ) : Prop :=
  (B.1 - A.1, B.2 - A.2) = (M.1 - C.1, M.2 - C.2)

-- Theorem statement
theorem parallelogram_fourth_vertex :
  ∃ M : ℝ × ℝ, is_parallelogram A B C M ∧ M = (4, 1) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1950_195062


namespace NUMINAMATH_CALUDE_largest_angle_in_special_right_triangle_l1950_195097

theorem largest_angle_in_special_right_triangle :
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  ∃ (x : ℝ), a = 3*x ∧ b = 2*x →
  max a (max b c) = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_right_triangle_l1950_195097


namespace NUMINAMATH_CALUDE_product_inequality_l1950_195023

theorem product_inequality (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) (h₄ : a₄ > 1) : 
  8 * (a₁ * a₂ * a₃ * a₄ + 1) ≥ (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1950_195023


namespace NUMINAMATH_CALUDE_tallest_player_height_l1950_195069

theorem tallest_player_height (shortest_height : ℝ) (height_difference : ℝ) 
  (h1 : shortest_height = 68.25)
  (h2 : height_difference = 9.5) :
  shortest_height + height_difference = 77.75 := by
  sorry

end NUMINAMATH_CALUDE_tallest_player_height_l1950_195069


namespace NUMINAMATH_CALUDE_sum_of_pairs_l1950_195027

theorem sum_of_pairs : (5 + 2) + (8 + 6) + (4 + 7) + (3 + 2) = 37 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_pairs_l1950_195027


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_l1950_195094

theorem quadratic_roots_reciprocal (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + a
  ∀ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 → x₁ = 1 / x₂ ∧ x₂ = 1 / x₁ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_l1950_195094


namespace NUMINAMATH_CALUDE_average_monthly_production_l1950_195034

/-- Calculates the average monthly salt production for a year given the initial production and monthly increase. -/
theorem average_monthly_production
  (initial_production : ℕ)
  (monthly_increase : ℕ)
  (months : ℕ)
  (h1 : initial_production = 1000)
  (h2 : monthly_increase = 100)
  (h3 : months = 12) :
  (initial_production + (initial_production + monthly_increase * (months - 1)) * (months - 1) / 2) / months = 9800 / 12 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_production_l1950_195034


namespace NUMINAMATH_CALUDE_negation_of_existence_square_leq_one_negation_l1950_195055

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, x < 1 ∧ p x) ↔ (∀ x, x < 1 → ¬ p x) :=
by sorry

theorem square_leq_one_negation :
  (¬ ∃ x : ℝ, x < 1 ∧ x^2 ≤ 1) ↔ (∀ x : ℝ, x < 1 → x^2 > 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_leq_one_negation_l1950_195055


namespace NUMINAMATH_CALUDE_newspaper_buying_percentage_l1950_195087

def newspapers_bought : ℕ := 500
def selling_price : ℚ := 2
def percentage_sold : ℚ := 80 / 100
def profit : ℚ := 550

theorem newspaper_buying_percentage : 
  ∀ (buying_price : ℚ),
    (newspapers_bought : ℚ) * percentage_sold * selling_price - 
    (newspapers_bought : ℚ) * buying_price = profit →
    (selling_price - buying_price) / selling_price = 75 / 100 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_buying_percentage_l1950_195087


namespace NUMINAMATH_CALUDE_cats_puppies_weight_difference_l1950_195020

/-- The number of puppies Hartley has -/
def num_puppies : ℕ := 4

/-- The weight of each puppy in kilograms -/
def puppy_weight : ℚ := 7.5

/-- The number of cats at the rescue center -/
def num_cats : ℕ := 14

/-- The weight of each cat in kilograms -/
def cat_weight : ℚ := 2.5

/-- The total weight of the puppies in kilograms -/
def total_puppy_weight : ℚ := num_puppies * puppy_weight

/-- The total weight of the cats in kilograms -/
def total_cat_weight : ℚ := num_cats * cat_weight

theorem cats_puppies_weight_difference :
  total_cat_weight - total_puppy_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_cats_puppies_weight_difference_l1950_195020


namespace NUMINAMATH_CALUDE_square_sum_equals_z_squared_l1950_195011

theorem square_sum_equals_z_squared (x y z b a : ℝ) 
  (h1 : x * y + x^2 = b)
  (h2 : 1 / x^2 - 1 / y^2 = a)
  (h3 : z = x + y) :
  (x + y)^2 = z^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_z_squared_l1950_195011


namespace NUMINAMATH_CALUDE_sodaCans_theorem_l1950_195008

/-- The number of cans of soda that can be bought for a given amount of euros -/
def sodaCans (S Q E : ℚ) : ℚ :=
  10 * E * S / Q

/-- Theorem stating that the number of cans of soda that can be bought for E euros
    is equal to 10ES/Q, given that S cans can be purchased for Q dimes and
    1 euro is equivalent to 10 dimes -/
theorem sodaCans_theorem (S Q E : ℚ) (hS : S > 0) (hQ : Q > 0) (hE : E ≥ 0) :
  sodaCans S Q E = 10 * E * S / Q :=
by sorry

end NUMINAMATH_CALUDE_sodaCans_theorem_l1950_195008


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a15_l1950_195085

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_a15 (a : ℕ → ℤ) :
  is_arithmetic_sequence a → a 7 = 8 → a 8 = 7 → a 15 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a15_l1950_195085


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l1950_195051

-- Define a triangle
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to A
  b : ℝ  -- side opposite to B
  c : ℝ  -- side opposite to C
  angle_sum : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  
-- State the theorem
theorem triangle_angle_inequality (t : Triangle) :
  Real.sin t.A * Real.sin t.B > Real.sin t.C ^ 2 → t.C < π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l1950_195051


namespace NUMINAMATH_CALUDE_jose_bottle_caps_l1950_195029

/-- The number of bottle caps Jose ends up with after receiving more -/
def total_bottle_caps (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Jose ends up with 9 bottle caps -/
theorem jose_bottle_caps : total_bottle_caps 7 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jose_bottle_caps_l1950_195029


namespace NUMINAMATH_CALUDE_organization_size_after_five_years_l1950_195005

def organization_growth (initial_members : ℕ) (initial_leaders : ℕ) (years : ℕ) : ℕ :=
  let rec growth (year : ℕ) (members : ℕ) : ℕ :=
    if year = 0 then
      members
    else
      growth (year - 1) (4 * members - 18)
  growth years initial_members

theorem organization_size_after_five_years :
  organization_growth 12 6 5 = 6150 := by
  sorry

end NUMINAMATH_CALUDE_organization_size_after_five_years_l1950_195005


namespace NUMINAMATH_CALUDE_final_comfortable_butterflies_l1950_195004

/-- Represents a point in the 2D lattice -/
structure LatticePoint where
  x : ℕ
  y : ℕ

/-- Represents the state of the lattice at any given time -/
def LatticeState := LatticePoint → Bool

/-- The neighborhood of a lattice point -/
def neighborhood (n : ℕ) (c : LatticePoint) : Set LatticePoint :=
  sorry

/-- Checks if a butterfly at a given point is lonely -/
def isLonely (n : ℕ) (state : LatticeState) (p : LatticePoint) : Bool :=
  sorry

/-- Simulates the process of lonely butterflies flying away -/
def simulateProcess (n : ℕ) (initialState : LatticeState) : LatticeState :=
  sorry

/-- Counts the number of comfortable butterflies in the final state -/
def countComfortableButterflies (n : ℕ) (finalState : LatticeState) : ℕ :=
  sorry

/-- The main theorem stating that the number of comfortable butterflies in the final state is n -/
theorem final_comfortable_butterflies (n : ℕ) (h : n > 0) :
  countComfortableButterflies n (simulateProcess n (λ _ => true)) = n :=
sorry

end NUMINAMATH_CALUDE_final_comfortable_butterflies_l1950_195004


namespace NUMINAMATH_CALUDE_negative_cube_squared_l1950_195056

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l1950_195056


namespace NUMINAMATH_CALUDE_lakota_used_cd_count_l1950_195012

/-- The price of a new CD in dollars -/
def new_cd_price : ℝ := 17.99

/-- The price of a used CD in dollars -/
def used_cd_price : ℝ := 9.99

/-- The number of new CDs Lakota bought -/
def lakota_new_cds : ℕ := 6

/-- The total amount Lakota spent in dollars -/
def lakota_total : ℝ := 127.92

/-- The number of new CDs Mackenzie bought -/
def mackenzie_new_cds : ℕ := 3

/-- The number of used CDs Mackenzie bought -/
def mackenzie_used_cds : ℕ := 8

/-- The total amount Mackenzie spent in dollars -/
def mackenzie_total : ℝ := 133.89

/-- The number of used CDs Lakota bought -/
def lakota_used_cds : ℕ := 2

theorem lakota_used_cd_count : 
  lakota_new_cds * new_cd_price + lakota_used_cds * used_cd_price = lakota_total ∧
  mackenzie_new_cds * new_cd_price + mackenzie_used_cds * used_cd_price = mackenzie_total :=
by sorry

end NUMINAMATH_CALUDE_lakota_used_cd_count_l1950_195012


namespace NUMINAMATH_CALUDE_cone_volume_l1950_195018

/-- Given a cone with slant height 5 and lateral surface area 20π, prove its volume is 16π -/
theorem cone_volume (s : ℝ) (l : ℝ) (v : ℝ) : 
  s = 5 → l = 20 * Real.pi → v = (16 : ℝ) * Real.pi → 
  (s^2 * Real.pi / l = s / 4) ∧ 
  (v = (1/3) * (l/s)^2 * (s^2 - (l/(Real.pi * s))^2)) := by
  sorry

#check cone_volume

end NUMINAMATH_CALUDE_cone_volume_l1950_195018


namespace NUMINAMATH_CALUDE_plate_distance_to_bottom_l1950_195038

/-- Given a square table with a round plate, if the distances from the plate to the top, left, and right edges
    of the table are 10, 63, and 20 units respectively, then the distance from the plate to the bottom edge
    of the table is 73 units. -/
theorem plate_distance_to_bottom (d : ℝ) :
  let top_distance : ℝ := 10
  let left_distance : ℝ := 63
  let right_distance : ℝ := 20
  let bottom_distance : ℝ := left_distance + right_distance - top_distance
  bottom_distance = 73 := by
  sorry


end NUMINAMATH_CALUDE_plate_distance_to_bottom_l1950_195038


namespace NUMINAMATH_CALUDE_probability_different_colors_is_three_fifths_l1950_195050

def num_red_balls : ℕ := 3
def num_white_balls : ℕ := 2
def total_balls : ℕ := num_red_balls + num_white_balls

def probability_different_colors : ℚ :=
  (num_red_balls * num_white_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2 : ℚ)

theorem probability_different_colors_is_three_fifths :
  probability_different_colors = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_is_three_fifths_l1950_195050


namespace NUMINAMATH_CALUDE_grocer_bananas_purchase_l1950_195016

/-- The number of pounds of bananas purchased by a grocer -/
def bananas_purchased (buy_price : ℚ) (sell_price : ℚ) (total_profit : ℚ) : ℚ :=
  total_profit / (sell_price / 4 - buy_price / 3)

/-- Theorem stating that the grocer purchased 72 pounds of bananas -/
theorem grocer_bananas_purchase :
  bananas_purchased (1/2) (1/1) 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_grocer_bananas_purchase_l1950_195016


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l1950_195088

/-- Given two concentric circles where a chord is tangent to the smaller circle,
    this theorem calculates the area of the region between the two circles. -/
theorem area_between_concentric_circles
  (outer_radius inner_radius chord_length : ℝ)
  (h_positive : 0 < inner_radius ∧ inner_radius < outer_radius)
  (h_tangent : inner_radius ^ 2 + (chord_length / 2) ^ 2 = outer_radius ^ 2)
  (h_chord : chord_length = 100) :
  (π * (outer_radius ^ 2 - inner_radius ^ 2) : ℝ) = 2000 * π :=
sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l1950_195088


namespace NUMINAMATH_CALUDE_base_length_is_double_half_length_l1950_195065

/-- An isosceles triangle with a line bisector from the vertex angle -/
structure IsoscelesTriangleWithBisector :=
  (base_half_length : ℝ)

/-- The theorem stating that the total base length is twice the length of each half -/
theorem base_length_is_double_half_length (triangle : IsoscelesTriangleWithBisector) 
  (h : triangle.base_half_length = 4) : 
  2 * triangle.base_half_length = 8 := by
  sorry

#check base_length_is_double_half_length

end NUMINAMATH_CALUDE_base_length_is_double_half_length_l1950_195065


namespace NUMINAMATH_CALUDE_fern_fronds_l1950_195009

theorem fern_fronds (total_ferns : ℕ) (total_leaves : ℕ) (leaves_per_frond : ℕ) 
  (h1 : total_ferns = 6)
  (h2 : total_leaves = 1260)
  (h3 : leaves_per_frond = 30) :
  (total_leaves / leaves_per_frond) / total_ferns = 7 := by
sorry

end NUMINAMATH_CALUDE_fern_fronds_l1950_195009


namespace NUMINAMATH_CALUDE_sqrt_x_minus_7_real_implies_x_geq_7_l1950_195073

theorem sqrt_x_minus_7_real_implies_x_geq_7 (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 7) → x ≥ 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_7_real_implies_x_geq_7_l1950_195073


namespace NUMINAMATH_CALUDE_square_sum_equals_b_times_ab_plus_two_l1950_195093

theorem square_sum_equals_b_times_ab_plus_two
  (x y a b : ℝ) 
  (h1 : x * y = b) 
  (h2 : 1 / (x^2) + 1 / (y^2) = a) : 
  (x + y)^2 = b * (a * b + 2) := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_b_times_ab_plus_two_l1950_195093


namespace NUMINAMATH_CALUDE_total_amount_is_265_l1950_195002

/-- Represents the distribution of money among six individuals -/
structure MoneyDistribution where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  t : ℝ
  u : ℝ

/-- The theorem stating the total amount given the conditions -/
theorem total_amount_is_265 (dist : MoneyDistribution) : 
  (dist.p = 3 * (dist.s / 1.95)) →
  (dist.q = 2.70 * (dist.s / 1.95)) →
  (dist.r = 2.30 * (dist.s / 1.95)) →
  (dist.s = 39) →
  (dist.t = 1.80 * (dist.s / 1.95)) →
  (dist.u = 1.50 * (dist.s / 1.95)) →
  (dist.p + dist.q + dist.r + dist.s + dist.t + dist.u = 265) := by
  sorry


end NUMINAMATH_CALUDE_total_amount_is_265_l1950_195002


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l1950_195054

theorem bobby_candy_problem (initial_candy : ℕ) (chocolate : ℕ) (candy_chocolate_diff : ℕ) :
  initial_candy = 38 →
  chocolate = 16 →
  candy_chocolate_diff = 58 →
  (initial_candy + chocolate + candy_chocolate_diff) - initial_candy = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l1950_195054


namespace NUMINAMATH_CALUDE_polygon_sides_from_exterior_angle_l1950_195048

theorem polygon_sides_from_exterior_angle (exterior_angle : ℝ) (n : ℕ) :
  exterior_angle = 36 →
  (360 : ℝ) / exterior_angle = n →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_exterior_angle_l1950_195048


namespace NUMINAMATH_CALUDE_star_op_power_equality_l1950_195059

def star_op (a b : ℕ+) : ℕ+ := a ^ (b.val ^ 2)

theorem star_op_power_equality (a b n : ℕ+) :
  (star_op a b) ^ n.val = star_op a (n * b) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_star_op_power_equality_l1950_195059


namespace NUMINAMATH_CALUDE_percentage_problem_l1950_195066

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : x * (x / 100) = 9) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1950_195066


namespace NUMINAMATH_CALUDE_expansion_terms_count_l1950_195039

def factor1 : ℕ := 3
def factor2 : ℕ := 4
def factor3 : ℕ := 5

theorem expansion_terms_count : factor1 * factor2 * factor3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l1950_195039


namespace NUMINAMATH_CALUDE_donny_remaining_money_l1950_195071

def initial_amount : ℕ := 78
def kite_cost : ℕ := 8
def frisbee_cost : ℕ := 9

theorem donny_remaining_money :
  initial_amount - (kite_cost + frisbee_cost) = 61 := by
  sorry

end NUMINAMATH_CALUDE_donny_remaining_money_l1950_195071


namespace NUMINAMATH_CALUDE_mary_overtime_rate_increase_l1950_195046

/-- Represents Mary's work schedule and pay structure -/
structure WorkSchedule where
  max_hours : ℕ
  regular_hours : ℕ
  regular_rate : ℚ
  total_earnings : ℚ

/-- Calculates the percentage increase in overtime rate compared to regular rate -/
def overtime_rate_increase (w : WorkSchedule) : ℚ :=
  let regular_earnings := w.regular_hours * w.regular_rate
  let overtime_earnings := w.total_earnings - regular_earnings
  let overtime_hours := w.max_hours - w.regular_hours
  let overtime_rate := overtime_earnings / overtime_hours
  ((overtime_rate - w.regular_rate) / w.regular_rate) * 100

/-- Mary's work schedule -/
def mary_schedule : WorkSchedule :=
  { max_hours := 40
  , regular_hours := 20
  , regular_rate := 8
  , total_earnings := 360 }

/-- Theorem stating that Mary's overtime rate increase is 25% -/
theorem mary_overtime_rate_increase :
  overtime_rate_increase mary_schedule = 25 := by
  sorry

end NUMINAMATH_CALUDE_mary_overtime_rate_increase_l1950_195046


namespace NUMINAMATH_CALUDE_julia_tuesday_playmates_l1950_195079

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 6

/-- The difference in number of kids between Monday and Tuesday -/
def difference : ℕ := 1

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := monday_kids - difference

theorem julia_tuesday_playmates : tuesday_kids = 5 := by
  sorry

end NUMINAMATH_CALUDE_julia_tuesday_playmates_l1950_195079


namespace NUMINAMATH_CALUDE_trig_expression_equals_three_halves_l1950_195028

theorem trig_expression_equals_three_halves (θ : Real) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_three_halves_l1950_195028


namespace NUMINAMATH_CALUDE_problem_statement_l1950_195089

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a / (1 + a) + b / (1 + b) = 1) : 
  a / (1 + b^2) - b / (1 + a^2) = a - b := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1950_195089


namespace NUMINAMATH_CALUDE_election_winner_votes_l1950_195025

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) : 
  winner_percentage = 62 / 100 →
  vote_difference = 324 →
  (winner_percentage * total_votes).num = (winner_percentage * total_votes).den * 837 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_votes_l1950_195025


namespace NUMINAMATH_CALUDE_oil_percentage_in_mixtureA_l1950_195026

/-- Represents the composition of a mixture --/
structure Mixture where
  oil : ℝ
  materialB : ℝ

/-- The original mixture A --/
def mixtureA : Mixture := sorry

/-- The weight of the original mixture A in kilograms --/
def originalWeight : ℝ := 8

/-- The weight of oil added to mixture A in kilograms --/
def addedOil : ℝ := 2

/-- The weight of mixture A added to the new mixture in kilograms --/
def addedMixtureA : ℝ := 6

/-- The percentage of material B in the final mixture --/
def finalMaterialBPercentage : ℝ := 70

/-- Theorem stating that the percentage of oil in the original mixture A is 20% --/
theorem oil_percentage_in_mixtureA : mixtureA.oil / (mixtureA.oil + mixtureA.materialB) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_oil_percentage_in_mixtureA_l1950_195026


namespace NUMINAMATH_CALUDE_min_snakes_owned_l1950_195080

/-- Represents the number of people owning a specific combination of pets -/
structure PetOwnership where
  total : ℕ
  onlyDogs : ℕ
  onlyCats : ℕ
  catsAndDogs : ℕ
  allThree : ℕ

/-- The given pet ownership data -/
def givenData : PetOwnership :=
  { total := 59
  , onlyDogs := 15
  , onlyCats := 10
  , catsAndDogs := 5
  , allThree := 3 }

/-- The minimum number of snakes owned -/
def minSnakes : ℕ := givenData.allThree

theorem min_snakes_owned (data : PetOwnership) : 
  data.allThree ≤ minSnakes := by sorry

end NUMINAMATH_CALUDE_min_snakes_owned_l1950_195080


namespace NUMINAMATH_CALUDE_largest_number_l1950_195047

/-- Converts a number from base b to decimal (base 10) --/
def to_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Theorem: 11 in base 3 is greater than 3 in base 10, 11 in base 2, and 3 in base 8 --/
theorem largest_number :
  (to_decimal 11 3 > to_decimal 3 10) ∧
  (to_decimal 11 3 > to_decimal 11 2) ∧
  (to_decimal 11 3 > to_decimal 3 8) :=
sorry

end NUMINAMATH_CALUDE_largest_number_l1950_195047


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1950_195040

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 1 + a 2 + a 3 = 12) 
  (h_prod : a 1 * a 2 * a 3 = 48) :
  ∀ n : ℕ, a n = 2 * n := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1950_195040


namespace NUMINAMATH_CALUDE_fill_time_calculation_l1950_195060

/-- Represents the time to fill a leaky tank -/
def fill_time_with_leak : ℝ := 8

/-- Represents the time for the tank to empty due to the leak -/
def empty_time : ℝ := 56

/-- Represents the time to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 7

/-- Theorem stating that given the fill time with leak and empty time,
    the fill time without leak is 7 hours -/
theorem fill_time_calculation :
  (fill_time_with_leak * empty_time) / (empty_time - fill_time_with_leak) = fill_time_without_leak :=
sorry

end NUMINAMATH_CALUDE_fill_time_calculation_l1950_195060

import Mathlib

namespace NUMINAMATH_CALUDE_cos_75_deg_l2028_202865

/-- Prove that cos 75° = (√6 - √2) / 4 using the angle sum identity for cosine with angles 60° and 15° -/
theorem cos_75_deg : 
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_deg_l2028_202865


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l2028_202876

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n.val 2 = n.val * (n.val - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l2028_202876


namespace NUMINAMATH_CALUDE_unique_base_solution_l2028_202877

/-- Converts a number from base b to decimal --/
def to_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Converts a number from decimal to base b --/
def from_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if a number is valid in base b --/
def is_valid_in_base (n : ℕ) (b : ℕ) : Prop := sorry

theorem unique_base_solution :
  ∃! b : ℕ, b > 6 ∧ 
    is_valid_in_base 243 b ∧
    is_valid_in_base 156 b ∧
    is_valid_in_base 411 b ∧
    to_decimal 243 b + to_decimal 156 b = to_decimal 411 b ∧
    b = 10 := by sorry

end NUMINAMATH_CALUDE_unique_base_solution_l2028_202877


namespace NUMINAMATH_CALUDE_book_words_per_page_l2028_202898

theorem book_words_per_page 
  (total_pages : ℕ) 
  (max_words_per_page : ℕ) 
  (total_words_mod : ℕ) 
  (modulus : ℕ) 
  (h1 : total_pages = 180) 
  (h2 : max_words_per_page = 150) 
  (h3 : total_words_mod = 203) 
  (h4 : modulus = 229) 
  (h5 : ∃ (words_per_page : ℕ), 
    words_per_page ≤ max_words_per_page ∧ 
    (total_pages * words_per_page) % modulus = total_words_mod) :
  ∃ (words_per_page : ℕ), words_per_page = 94 ∧ 
    words_per_page ≤ max_words_per_page ∧ 
    (total_pages * words_per_page) % modulus = total_words_mod :=
sorry

end NUMINAMATH_CALUDE_book_words_per_page_l2028_202898


namespace NUMINAMATH_CALUDE_area_original_triangle_l2028_202858

/-- Given a triangle ABC and its oblique dimetric projection A''B''C'',
    where A''B''C'' is an equilateral triangle with side length a,
    prove that the area of ABC is (√6 * a^2) / 2. -/
theorem area_original_triangle (a : ℝ) (h : a > 0) :
  let s_projection := (Real.sqrt 3 * a^2) / 4
  let ratio := Real.sqrt 2 / 4
  s_projection / ratio = (Real.sqrt 6 * a^2) / 2 := by
sorry

end NUMINAMATH_CALUDE_area_original_triangle_l2028_202858


namespace NUMINAMATH_CALUDE_inscribed_square_arc_length_l2028_202841

/-- Given a square inscribed in a circle with side length 4,
    the arc length intercepted by any side of the square is √2π. -/
theorem inscribed_square_arc_length (s : Real) (r : Real) (arc_length : Real) :
  s = 4 →                        -- Side length of the square is 4
  r = 2 * Real.sqrt 2 →          -- Radius of the circle
  arc_length = Real.sqrt 2 * π → -- Arc length intercepted by any side
  True :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_arc_length_l2028_202841


namespace NUMINAMATH_CALUDE_johns_allowance_l2028_202821

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℚ) : 
  (3/5 : ℚ) * A + (1/3 : ℚ) * (2/5 : ℚ) * A + (4/10 : ℚ) = A → A = (3/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l2028_202821


namespace NUMINAMATH_CALUDE_total_nails_formula_nails_for_40_per_side_l2028_202833

/-- The number of nails used to fix a square metal plate -/
def total_nails (nails_per_side : ℕ) : ℕ :=
  4 * nails_per_side - 4

/-- Theorem: The total number of nails used is equal to 4 times the number of nails on one side, minus 4 -/
theorem total_nails_formula (nails_per_side : ℕ) :
  total_nails nails_per_side = 4 * nails_per_side - 4 := by
  sorry

/-- Corollary: For a square with 40 nails on each side, the total number of nails used is 156 -/
theorem nails_for_40_per_side :
  total_nails 40 = 156 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_formula_nails_for_40_per_side_l2028_202833


namespace NUMINAMATH_CALUDE_fencing_cost_calculation_l2028_202816

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length breadth fencing_cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * fencing_cost_per_meter

/-- Proves that the total cost of fencing the given rectangular plot is 5300 currency units -/
theorem fencing_cost_calculation :
  let length : ℝ := 75
  let breadth : ℝ := 25
  let fencing_cost_per_meter : ℝ := 26.50
  (length = breadth + 50) →
  total_fencing_cost length breadth fencing_cost_per_meter = 5300 := by
  sorry

#eval total_fencing_cost 75 25 26.50

end NUMINAMATH_CALUDE_fencing_cost_calculation_l2028_202816


namespace NUMINAMATH_CALUDE_power_difference_l2028_202892

theorem power_difference (a : ℝ) (m n : ℤ) (h1 : a^m = 2) (h2 : a^n = 16) :
  a^(m-n) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l2028_202892


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2028_202824

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^2 + 3 * x - 7) * (4 * x^3) = 20 * x^5 + 12 * x^4 - 28 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2028_202824


namespace NUMINAMATH_CALUDE_angle_ratio_is_one_fourth_l2028_202894

-- Define the points
variable (A B C P Q M : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- State the conditions
variable (h1 : angle A B P = angle P B Q)  -- BP bisects ∠ABQ
variable (h2 : angle A B C > angle A B Q)  -- BQ is within ∠ABC
variable (h3 : angle P B M = angle M B Q)  -- BM bisects ∠PBQ

-- State the theorem
theorem angle_ratio_is_one_fourth :
  (angle M B Q) / (angle A B Q) = 1/4 := by sorry

end NUMINAMATH_CALUDE_angle_ratio_is_one_fourth_l2028_202894


namespace NUMINAMATH_CALUDE_correct_expression_l2028_202836

theorem correct_expression (x y : ℚ) (h : x / y = 5 / 6) : 
  (x + 3 * y) / x = 23 / 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_expression_l2028_202836


namespace NUMINAMATH_CALUDE_cubic_inequality_l2028_202805

theorem cubic_inequality (a b : ℝ) 
  (h1 : a^3 - b^3 = 2) 
  (h2 : a^5 - b^5 ≥ 4) : 
  a > b ∧ a^2 + b^2 ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2028_202805


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2028_202806

-- Define the quadratic function
def f (x : ℝ) := x^2 + x - 2

-- Define the solution set
def solution_set := {x : ℝ | x ≤ -2 ∨ x ≥ 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ x : ℝ, f x ≥ 0 ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2028_202806


namespace NUMINAMATH_CALUDE_smallest_positive_angle_theorem_l2028_202885

theorem smallest_positive_angle_theorem (θ : Real) : 
  (θ > 0) → 
  (10 * Real.sin θ * (Real.cos θ)^3 - 10 * (Real.sin θ)^3 * Real.cos θ = Real.sqrt 2) →
  (∀ φ, φ > 0 → 10 * Real.sin φ * (Real.cos φ)^3 - 10 * (Real.sin φ)^3 * Real.cos φ = Real.sqrt 2 → θ ≤ φ) →
  θ = (1/4) * Real.arcsin ((2 * Real.sqrt 2) / 5) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_theorem_l2028_202885


namespace NUMINAMATH_CALUDE_max_value_constraint_l2028_202855

theorem max_value_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 * a + 6 * b < 110) :
  a * b * (110 - 5 * a - 6 * b) ≤ 1331000 / 810 := by
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2028_202855


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2028_202893

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = -x + 2}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2028_202893


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2028_202826

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  stratum_size : ℕ
  stratum_sample : ℕ
  total_sample : ℕ

/-- Checks if the sampling is proportionally correct -/
def is_proportional_sampling (s : StratifiedSample) : Prop :=
  s.stratum_sample * s.total_population = s.total_sample * s.stratum_size

theorem stratified_sample_size 
  (s : StratifiedSample) 
  (h1 : s.total_population = 4320)
  (h2 : s.stratum_size = 1800)
  (h3 : s.stratum_sample = 45)
  (h4 : is_proportional_sampling s) :
  s.total_sample = 108 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2028_202826


namespace NUMINAMATH_CALUDE_rubiks_cube_probabilities_l2028_202813

/-- The probability of person A solving the cube within 30 seconds -/
def prob_A : ℝ := 0.8

/-- The probability of person B solving the cube within 30 seconds -/
def prob_B : ℝ := 0.6

/-- The probability of person A succeeding on their third attempt -/
def prob_A_third_attempt : ℝ := (1 - prob_A) * (1 - prob_A) * prob_A

/-- The probability that at least one of them succeeds on their first attempt -/
def prob_at_least_one_first_attempt : ℝ := 1 - (1 - prob_A) * (1 - prob_B)

theorem rubiks_cube_probabilities :
  prob_A_third_attempt = 0.032 ∧ prob_at_least_one_first_attempt = 0.92 := by
  sorry

end NUMINAMATH_CALUDE_rubiks_cube_probabilities_l2028_202813


namespace NUMINAMATH_CALUDE_uncle_dave_nieces_l2028_202846

theorem uncle_dave_nieces (total_sandwiches : ℕ) (sandwiches_per_niece : ℕ) (h1 : total_sandwiches = 143) (h2 : sandwiches_per_niece = 13) :
  total_sandwiches / sandwiches_per_niece = 11 := by
  sorry

end NUMINAMATH_CALUDE_uncle_dave_nieces_l2028_202846


namespace NUMINAMATH_CALUDE_sameColorPairsTheorem_l2028_202845

/-- The number of ways to choose a pair of socks of the same color from a drawer -/
def sameColorPairs (white green brown blue : ℕ) : ℕ :=
  Nat.choose white 2 + Nat.choose green 2 + Nat.choose brown 2 + Nat.choose blue 2

/-- Theorem: Given a drawer with 16 distinguishable socks (6 white, 4 green, 4 brown, and 2 blue),
    the number of ways to choose a pair of socks of the same color is 28. -/
theorem sameColorPairsTheorem :
  sameColorPairs 6 4 4 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_sameColorPairsTheorem_l2028_202845


namespace NUMINAMATH_CALUDE_min_points_for_obtuse_triangle_l2028_202844

/-- A color representing red, yellow, or blue -/
inductive Color
  | Red
  | Yellow
  | Blue

/-- A point on the circumference of a circle -/
structure CirclePoint where
  angle : Real
  color : Color

/-- A function that colors every point on the circle's circumference -/
def colorCircle : Real → Color := sorry

/-- Predicate to check if all three colors are present on the circle -/
def allColorsPresent (colorCircle : Real → Color) : Prop := sorry

/-- Predicate to check if three points form an obtuse triangle -/
def isObtuseTriangle (p1 p2 p3 : CirclePoint) : Prop := sorry

/-- The minimum number of points that guarantees an obtuse triangle of the same color -/
def minPointsForObtuseTriangle : Nat := sorry

/-- Theorem stating the minimum number of points required -/
theorem min_points_for_obtuse_triangle :
  ∀ (colorCircle : Real → Color),
    allColorsPresent colorCircle →
    (∀ (points : Finset CirclePoint),
      points.card ≥ minPointsForObtuseTriangle →
      ∃ (p1 p2 p3 : CirclePoint),
        p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
        p1.color = p2.color ∧ p2.color = p3.color ∧
        isObtuseTriangle p1 p2 p3) ∧
    minPointsForObtuseTriangle = 13 :=
by sorry

end NUMINAMATH_CALUDE_min_points_for_obtuse_triangle_l2028_202844


namespace NUMINAMATH_CALUDE_molecular_weight_CaI2_value_l2028_202888

/-- The atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of Iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of Calcium atoms in CaI2 -/
def num_Ca : ℕ := 1

/-- The number of Iodine atoms in CaI2 -/
def num_I : ℕ := 2

/-- The molecular weight of CaI2 in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca * num_Ca + atomic_weight_I * num_I

theorem molecular_weight_CaI2_value : molecular_weight_CaI2 = 293.88 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_CaI2_value_l2028_202888


namespace NUMINAMATH_CALUDE_largest_of_three_l2028_202848

theorem largest_of_three (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_prod_eq : p*q + p*r + q*r = -6)
  (prod_eq : p*q*r = -18) :
  max p (max q r) = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_largest_of_three_l2028_202848


namespace NUMINAMATH_CALUDE_batsman_score_difference_l2028_202872

theorem batsman_score_difference (total_innings : ℕ) (total_average : ℚ) (reduced_average : ℚ) (highest_score : ℕ) :
  total_innings = 46 →
  total_average = 61 →
  reduced_average = 58 →
  highest_score = 202 →
  ∃ (lowest_score : ℕ),
    (total_average * total_innings : ℚ) = 
      (reduced_average * (total_innings - 2) + (highest_score + lowest_score) : ℚ) ∧
    highest_score - lowest_score = 150 :=
by sorry

end NUMINAMATH_CALUDE_batsman_score_difference_l2028_202872


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l2028_202861

theorem largest_prime_factors_difference (n : Nat) (h : n = 184437) :
  ∃ (p q : Nat), Prime p ∧ Prime q ∧ p > q ∧
  (∀ r : Nat, Prime r → r ∣ n → r ≤ p) ∧
  (p ∣ n) ∧ (q ∣ n) ∧ (p - q = 8776) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l2028_202861


namespace NUMINAMATH_CALUDE_cos_sin_fifteen_degree_identity_l2028_202829

theorem cos_sin_fifteen_degree_identity :
  Real.cos (15 * π / 180) ^ 4 - Real.sin (15 * π / 180) ^ 4 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_fifteen_degree_identity_l2028_202829


namespace NUMINAMATH_CALUDE_gavins_dreams_l2028_202801

/-- The number of dreams Gavin has every day this year -/
def dreams_per_day : ℕ := sorry

/-- The number of days in a year -/
def days_in_year : ℕ := 365

/-- The total number of dreams in two years -/
def total_dreams : ℕ := 4380

theorem gavins_dreams : 
  dreams_per_day * days_in_year + 2 * (dreams_per_day * days_in_year) = total_dreams ∧ 
  dreams_per_day = 4 := by sorry

end NUMINAMATH_CALUDE_gavins_dreams_l2028_202801


namespace NUMINAMATH_CALUDE_debt_doubling_time_l2028_202817

theorem debt_doubling_time (interest_rate : ℝ) (doubling_factor : ℝ) : 
  interest_rate = 0.06 → doubling_factor = 2 → 
  (∀ t : ℕ, t < 12 → (1 + interest_rate)^t ≤ doubling_factor) ∧ 
  (1 + interest_rate)^12 > doubling_factor := by
  sorry

end NUMINAMATH_CALUDE_debt_doubling_time_l2028_202817


namespace NUMINAMATH_CALUDE_skew_symmetric_times_symmetric_is_zero_l2028_202851

/-- Given a skew-symmetric matrix A and a symmetric matrix B, prove that their product is the zero matrix -/
theorem skew_symmetric_times_symmetric_is_zero (a b c : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![0, c, -b; -c, 0, a; b, -a, 0]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![a^2, a*b, a*c; a*b, b^2, b*c; a*c, b*c, c^2]
  A * B = 0 := by sorry

end NUMINAMATH_CALUDE_skew_symmetric_times_symmetric_is_zero_l2028_202851


namespace NUMINAMATH_CALUDE_cord_lengths_l2028_202839

theorem cord_lengths (total_length : ℝ) (a b c : ℝ) : 
  total_length = 60 → -- Total length is 60 decimeters
  a + b + c = total_length * 10 → -- Sum of parts equals total length in cm
  b = a + 1 → -- Second part is 1 cm more than first
  c = b + 1 → -- Third part is 1 cm more than second
  (a, b, c) = (199, 200, 201) := by sorry

end NUMINAMATH_CALUDE_cord_lengths_l2028_202839


namespace NUMINAMATH_CALUDE_sequence_problem_l2028_202812

/-- Given a geometric sequence {a_n} and an arithmetic sequence {b_n} satisfying certain conditions,
    this theorem proves the general formulas for both sequences and the minimum n for which
    the sum of their first n terms exceeds 100. -/
theorem sequence_problem (a b : ℕ → ℝ) (n : ℕ) : 
  (∀ k, a (k + 1) = a k * (a 2 / a 1)) →  -- geometric sequence condition
  (∀ k, b (k + 1) - b k = b 2 - b 1) →   -- arithmetic sequence condition
  a 1 = 1 →
  b 1 = 1 →
  a 1 ≠ a 2 →
  a 1 + b 3 = 2 * a 2 →  -- a₁, a₂, b₃ form an arithmetic sequence
  b 1 * b 4 = (a 2)^2 →  -- b₁, a₂, b₄ form a geometric sequence
  (∀ k, a k = 2^(k-1)) ∧ 
  (∀ k, b k = k) ∧
  (n = 7 ∧ (2^n - 1 + n * (n + 1) / 2 > 100) ∧ 
   ∀ m < n, (2^m - 1 + m * (m + 1) / 2 ≤ 100)) :=
by sorry


end NUMINAMATH_CALUDE_sequence_problem_l2028_202812


namespace NUMINAMATH_CALUDE_equation_proof_l2028_202880

theorem equation_proof (x y : ℝ) (h : x - 2*y = -2) : 3 + 2*x - 4*y = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2028_202880


namespace NUMINAMATH_CALUDE_happy_children_count_l2028_202863

theorem happy_children_count (total : ℕ) (sad : ℕ) (neither : ℕ) (boys : ℕ) (girls : ℕ) 
  (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ) :
  total = 60 →
  sad = 10 →
  neither = 20 →
  boys = 17 →
  girls = 43 →
  happy_boys = 6 →
  sad_girls = 4 →
  neither_boys = 5 →
  ∃ (happy : ℕ), happy = 30 ∧ happy + sad + neither = total :=
by sorry

end NUMINAMATH_CALUDE_happy_children_count_l2028_202863


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l2028_202849

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials_15 :
  lastTwoDigits (sumFactorials 15) = 13 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l2028_202849


namespace NUMINAMATH_CALUDE_bin_game_expectation_l2028_202864

theorem bin_game_expectation (k : ℕ+) : 
  let total_balls : ℕ := 8 + k
  let green_prob : ℚ := 8 / total_balls
  let purple_prob : ℚ := k / total_balls
  let expected_value : ℚ := green_prob * 3 + purple_prob * (-1)
  expected_value = 60 / 100 → k = 12 := by
sorry

end NUMINAMATH_CALUDE_bin_game_expectation_l2028_202864


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2028_202803

/-- The area of a triangle with sides 13, 14, and 15 is 84 -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    let a : ℝ := 13
    let b : ℝ := 14
    let c : ℝ := 15
    let s : ℝ := (a + b + c) / 2
    area = Real.sqrt (s * (s - a) * (s - b) * (s - c)) ∧ area = 84

/-- Proof of the triangle area theorem -/
theorem triangle_area_proof : ∃ area : ℝ, triangle_area area := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2028_202803


namespace NUMINAMATH_CALUDE_simplify_sum_of_square_roots_l2028_202881

theorem simplify_sum_of_square_roots : 
  Real.sqrt (10 + 6 * Real.sqrt 3) + Real.sqrt (10 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sum_of_square_roots_l2028_202881


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l2028_202889

def f (m : ℝ) (x : ℝ) : ℝ := x^4 + (m-1)*x + 1

theorem even_function_implies_m_equals_one (m : ℝ) :
  (∀ x, f m x = f m (-x)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l2028_202889


namespace NUMINAMATH_CALUDE_min_value_product_l2028_202895

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x + 3*y) * (y + 3*z) * (x*z + 2) ≥ 96 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (x₀ + 3*y₀) * (y₀ + 3*z₀) * (x₀*z₀ + 2) = 96 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l2028_202895


namespace NUMINAMATH_CALUDE_no_torn_cards_l2028_202884

/-- The number of baseball cards Mary initially had -/
def initial_cards : ℕ := 18

/-- The number of baseball cards Fred gave to Mary -/
def fred_cards : ℕ := 26

/-- The number of baseball cards Mary bought -/
def bought_cards : ℕ := 40

/-- The total number of baseball cards Mary has now -/
def total_cards : ℕ := 84

/-- The number of torn baseball cards in Mary's initial collection -/
def torn_cards : ℕ := initial_cards - (total_cards - fred_cards - bought_cards)

theorem no_torn_cards : torn_cards = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_torn_cards_l2028_202884


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2028_202811

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - (k + 1) * x + 2 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - (k + 1) * y + 2 = 0 → y = x) ↔ 
  (k = 2 * Real.sqrt 6 - 1 ∨ k = -2 * Real.sqrt 6 - 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2028_202811


namespace NUMINAMATH_CALUDE_cards_distribution_l2028_202887

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) 
  (h2 : num_people = 8) : 
  (num_people - (total_cards % num_people)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l2028_202887


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l2028_202891

theorem roots_sum_of_squares (m n : ℝ) : 
  (m^2 - 5*m + 3 = 0) → (n^2 - 5*n + 3 = 0) → (m^2 + n^2 = 19) := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l2028_202891


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l2028_202808

theorem largest_multiple_of_15_under_500 : ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l2028_202808


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2028_202804

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (1/2 * (3*x^2 - 1) = (x^2 - 50*x - 10)*(x^2 + 25*x + 5)) ∧ x = 25 + Real.sqrt 159 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2028_202804


namespace NUMINAMATH_CALUDE_function_value_order_l2028_202850

noncomputable def f (x : ℝ) := Real.log (abs (x - 2)) + x^2 - 4*x

theorem function_value_order :
  let a := f (Real.log 9 / Real.log 2)
  let b := f (Real.log 18 / Real.log 4)
  let c := f 1
  a > c ∧ c > b :=
by sorry

end NUMINAMATH_CALUDE_function_value_order_l2028_202850


namespace NUMINAMATH_CALUDE_A_initial_investment_l2028_202882

/-- Represents the initial investment of A in rupees -/
def A_investment : ℝ := sorry

/-- Represents B's contribution to the capital in rupees -/
def B_investment : ℝ := 16200

/-- Represents the number of months A's investment was active -/
def A_months : ℝ := 12

/-- Represents the number of months B's investment was active -/
def B_months : ℝ := 5

/-- Represents the ratio of A's profit share -/
def A_profit_ratio : ℝ := 2

/-- Represents the ratio of B's profit share -/
def B_profit_ratio : ℝ := 3

theorem A_initial_investment : 
  (A_investment * A_months) / (B_investment * B_months) = A_profit_ratio / B_profit_ratio →
  A_investment = 4500 := by
sorry

end NUMINAMATH_CALUDE_A_initial_investment_l2028_202882


namespace NUMINAMATH_CALUDE_trivia_game_points_per_question_l2028_202843

theorem trivia_game_points_per_question 
  (first_half_correct : ℕ) 
  (second_half_correct : ℕ) 
  (final_score : ℕ) 
  (h1 : first_half_correct = 5)
  (h2 : second_half_correct = 5)
  (h3 : final_score = 50) :
  final_score / (first_half_correct + second_half_correct) = 5 := by
sorry

end NUMINAMATH_CALUDE_trivia_game_points_per_question_l2028_202843


namespace NUMINAMATH_CALUDE_average_monthly_balance_l2028_202890

def monthly_balances : List ℝ := [100, 200, 250, 250, 150, 100]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℝ) = 175 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l2028_202890


namespace NUMINAMATH_CALUDE_four_integers_with_many_divisors_l2028_202823

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem four_integers_with_many_divisors :
  ∃ (a b c d : ℕ),
    a > 0 ∧ a ≤ 70000 ∧ count_divisors a > 100 ∧
    b > 0 ∧ b ≤ 70000 ∧ count_divisors b > 100 ∧
    c > 0 ∧ c ≤ 70000 ∧ count_divisors c > 100 ∧
    d > 0 ∧ d ≤ 70000 ∧ count_divisors d > 100 :=
by
  use 69300, 50400, 60480, 55440
  sorry

end NUMINAMATH_CALUDE_four_integers_with_many_divisors_l2028_202823


namespace NUMINAMATH_CALUDE_smaller_tetrahedron_volume_ratio_l2028_202897

-- Define a regular tetrahedron
structure RegularTetrahedron where
  edge_length : ℝ
  is_positive : edge_length > 0

-- Define the division of edges
def divide_edges (t : RegularTetrahedron) : ℕ := 3

-- Define the smaller tetrahedron
structure SmallerTetrahedron (t : RegularTetrahedron) where
  division_points : divide_edges t = 3

-- Define the volume ratio
def volume_ratio (t : RegularTetrahedron) (s : SmallerTetrahedron t) : ℚ := 1 / 27

-- Theorem statement
theorem smaller_tetrahedron_volume_ratio 
  (t : RegularTetrahedron) 
  (s : SmallerTetrahedron t) : 
  volume_ratio t s = 1 / 27 := by
  sorry


end NUMINAMATH_CALUDE_smaller_tetrahedron_volume_ratio_l2028_202897


namespace NUMINAMATH_CALUDE_tree_purchase_equations_l2028_202815

/-- Represents the cost of an A-type tree -/
def cost_A : ℕ := 100

/-- Represents the cost of a B-type tree -/
def cost_B : ℕ := 80

/-- Represents the total amount spent -/
def total_spent : ℕ := 8000

/-- Represents the difference in number between A-type and B-type trees -/
def tree_difference : ℕ := 8

theorem tree_purchase_equations (x y : ℕ) :
  (x - y = tree_difference ∧ cost_A * x + cost_B * y = total_spent) ↔
  (x - y = 8 ∧ 100 * x + 80 * y = 8000) :=
sorry

end NUMINAMATH_CALUDE_tree_purchase_equations_l2028_202815


namespace NUMINAMATH_CALUDE_car_selling_price_l2028_202870

/-- Calculates the selling price of a car given its purchase price, repair cost, and profit percentage. -/
theorem car_selling_price (purchase_price repair_cost : ℕ) (profit_percent : ℚ) :
  purchase_price = 42000 →
  repair_cost = 13000 →
  profit_percent = 17272727272727273 / 100000000000000000 →
  (purchase_price + repair_cost) * (1 + profit_percent) = 64500 := by
  sorry

end NUMINAMATH_CALUDE_car_selling_price_l2028_202870


namespace NUMINAMATH_CALUDE_opposite_numbers_with_equation_l2028_202838

theorem opposite_numbers_with_equation (x y : ℝ) : 
  x + y = 0 → (x + 2)^2 - (y + 2)^2 = 4 → x = 1/2 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_with_equation_l2028_202838


namespace NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l2028_202883

/-- The area of a circular sector with central angle 120° and radius √3 is π. -/
theorem sector_area_120_deg_sqrt3_radius (π : ℝ) : 
  let angle : ℝ := 120 * π / 180  -- Convert 120° to radians
  let radius : ℝ := Real.sqrt 3
  let sector_area := (1 / 2) * angle * radius^2
  sector_area = π := by sorry

end NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l2028_202883


namespace NUMINAMATH_CALUDE_angle4_value_l2028_202800

-- Define the angles
variable (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)

-- Define the given conditions
axiom sum_angles_1_2 : angle1 + angle2 = 180
axiom equal_angles_3_4 : angle3 = angle4
axiom new_angle1 : angle1 = 85
axiom new_angle5 : angle5 = 45
axiom triangle_sum : angle1 + angle5 + angle6 = 180

-- Define the theorem to prove
theorem angle4_value : angle4 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_angle4_value_l2028_202800


namespace NUMINAMATH_CALUDE_apollonian_circle_l2028_202814

/-- The Apollonian circle theorem -/
theorem apollonian_circle (r : ℝ) (h_r_pos : r > 0) : 
  (∃! P : ℝ × ℝ, (P.1 - 2)^2 + P.2^2 = r^2 ∧ 
    ((P.1 - 3)^2 + P.2^2) = 4 * (P.1^2 + P.2^2)) → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_apollonian_circle_l2028_202814


namespace NUMINAMATH_CALUDE_f_is_square_iff_n_eq_one_l2028_202852

/-- The number of non-empty subsets of {1, ..., n} with gcd 1 -/
def f (n : ℕ+) : ℕ := sorry

/-- f(n) is a perfect square iff n = 1 -/
theorem f_is_square_iff_n_eq_one (n : ℕ+) : 
  ∃ m : ℕ, f n = m ^ 2 ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_f_is_square_iff_n_eq_one_l2028_202852


namespace NUMINAMATH_CALUDE_price_reduction_proof_l2028_202896

/-- The original selling price -/
def original_price : ℝ := 40

/-- The cost price -/
def cost_price : ℝ := 30

/-- The initial daily sales volume -/
def initial_sales : ℝ := 48

/-- The price after two reductions -/
def reduced_price : ℝ := 32.4

/-- The additional sales per yuan of price reduction -/
def sales_increase_rate : ℝ := 8

/-- The target daily profit -/
def target_profit : ℝ := 504

/-- The percentage reduction in price -/
def reduction_percentage : ℝ := 0.1

/-- The price reduction amount -/
def price_reduction : ℝ := 3

theorem price_reduction_proof :
  (∃ x : ℝ, (1 - x)^2 * original_price = reduced_price ∧ x = reduction_percentage) ∧
  (∃ m : ℝ, (original_price - m - cost_price) * (initial_sales + sales_increase_rate * m) = target_profit ∧ m = price_reduction) := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_proof_l2028_202896


namespace NUMINAMATH_CALUDE_min_time_to_return_l2028_202867

/-- Given a circular track and a person's walking pattern, calculate the minimum time to return to the starting point. -/
theorem min_time_to_return (track_length : ℝ) (speed : ℝ) (t1 t2 t3 : ℝ) : 
  track_length = 400 →
  speed = 6000 / 60 →
  t1 = 1 →
  t2 = 3 →
  t3 = 5 →
  (min_time : ℝ) * speed = track_length - ((t1 - t2 + t3) * speed) →
  min_time = 1 := by
  sorry

#check min_time_to_return

end NUMINAMATH_CALUDE_min_time_to_return_l2028_202867


namespace NUMINAMATH_CALUDE_lawn_mowing_price_l2028_202899

def sneaker_cost : ℕ := 92
def lawns_to_mow : ℕ := 3
def figures_to_sell : ℕ := 2
def figure_price : ℕ := 9
def job_hours : ℕ := 10
def hourly_rate : ℕ := 5

theorem lawn_mowing_price : 
  (sneaker_cost - (figures_to_sell * figure_price + job_hours * hourly_rate)) / lawns_to_mow = 8 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_price_l2028_202899


namespace NUMINAMATH_CALUDE_geometric_and_arithmetic_sequences_l2028_202847

-- Define the geometric sequence a_n
def a (n : ℕ) : ℝ := 3 * 2^(n - 1)

-- Define the arithmetic sequence b_n
def b (n : ℕ) : ℝ := 6 * n - 6

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℝ := 3 * n^2 - 3 * n

theorem geometric_and_arithmetic_sequences :
  (a 1 = 3) ∧ 
  (a 4 = 24) ∧ 
  (b 2 = a 2) ∧ 
  (b 9 = a 5) ∧ 
  (∀ n : ℕ, a n = 3 * 2^(n - 1)) ∧ 
  (∀ n : ℕ, S n = 3 * n^2 - 3 * n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_and_arithmetic_sequences_l2028_202847


namespace NUMINAMATH_CALUDE_jose_profit_share_l2028_202869

/-- Calculates the share of profit for an investor given the total profit and investments --/
def calculate_profit_share (total_profit : ℚ) (investment1 : ℚ) (months1 : ℕ) (investment2 : ℚ) (months2 : ℕ) : ℚ :=
  let total_investment := investment1 * months1 + investment2 * months2
  let share_ratio := (investment2 * months2) / total_investment
  share_ratio * total_profit

/-- Proves that Jose's share of the profit is 3500 given the problem conditions --/
theorem jose_profit_share :
  let tom_investment : ℚ := 3000
  let jose_investment : ℚ := 4500
  let tom_months : ℕ := 12
  let jose_months : ℕ := 10
  let total_profit : ℚ := 6300
  calculate_profit_share total_profit tom_investment tom_months jose_investment jose_months = 3500 := by
  sorry


end NUMINAMATH_CALUDE_jose_profit_share_l2028_202869


namespace NUMINAMATH_CALUDE_dave_breaks_two_strings_per_night_l2028_202819

def shows_per_week : ℕ := 6
def total_weeks : ℕ := 12
def total_strings : ℕ := 144

theorem dave_breaks_two_strings_per_night :
  (total_strings : ℚ) / (shows_per_week * total_weeks) = 2 := by
  sorry

end NUMINAMATH_CALUDE_dave_breaks_two_strings_per_night_l2028_202819


namespace NUMINAMATH_CALUDE_candy_bar_weight_reduction_l2028_202818

/-- Represents the change in weight and price of a candy bar -/
structure CandyBar where
  original_weight : ℝ
  new_weight : ℝ
  price : ℝ
  price_per_ounce_increase : ℝ

/-- The theorem stating the relationship between weight reduction and price per ounce increase -/
theorem candy_bar_weight_reduction (c : CandyBar) 
  (h1 : c.price_per_ounce_increase = 2/3)
  (h2 : c.price > 0)
  (h3 : c.original_weight > 0)
  (h4 : c.new_weight > 0)
  (h5 : c.new_weight < c.original_weight) :
  (c.original_weight - c.new_weight) / c.original_weight = 0.4 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_weight_reduction_l2028_202818


namespace NUMINAMATH_CALUDE_charlyn_visible_area_l2028_202856

/-- The area of the region visible to Charlyn during her walk around a square -/
def visible_area (square_side : ℝ) (visibility_range : ℝ) : ℝ :=
  let inner_square_side := square_side - 2 * visibility_range
  let inner_area := inner_square_side ^ 2
  let outer_rectangles_area := 4 * (square_side * visibility_range)
  let corner_squares_area := 4 * (visibility_range ^ 2)
  (square_side ^ 2 - inner_area) + outer_rectangles_area + corner_squares_area

/-- Theorem stating that the visible area for Charlyn's walk is 160 km² -/
theorem charlyn_visible_area :
  visible_area 10 2 = 160 := by
  sorry

#eval visible_area 10 2

end NUMINAMATH_CALUDE_charlyn_visible_area_l2028_202856


namespace NUMINAMATH_CALUDE_circle_equation_l2028_202857

/-- A circle with center on the x-axis, radius √2, passing through (-2, 1) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  radius_is_sqrt_2 : radius = Real.sqrt 2
  passes_through_point : passes_through = (-2, 1)

/-- The equation of the circle is either (x+1)² + y² = 2 or (x+3)² + y² = 2 -/
theorem circle_equation (c : Circle) :
  (∀ x y : ℝ, (x + 1)^2 + y^2 = 2 ∨ (x + 3)^2 + y^2 = 2 ↔
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2028_202857


namespace NUMINAMATH_CALUDE_complement_of_A_relative_to_U_l2028_202860

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4, 5}

theorem complement_of_A_relative_to_U :
  (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_relative_to_U_l2028_202860


namespace NUMINAMATH_CALUDE_star_difference_l2028_202840

def star (x y : ℤ) : ℤ := x * y - 3 * x

theorem star_difference : (star 6 2) - (star 2 6) = -12 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_l2028_202840


namespace NUMINAMATH_CALUDE_family_photos_l2028_202842

theorem family_photos (total : ℕ) (friends : ℕ) (family : ℕ) 
  (h1 : total = 86) 
  (h2 : friends = 63) 
  (h3 : total = friends + family) : family = 23 := by
  sorry

end NUMINAMATH_CALUDE_family_photos_l2028_202842


namespace NUMINAMATH_CALUDE_cost_of_type_b_books_l2028_202879

/-- Given a total of 100 books, with 'a' books of type A purchased,
    and type B books costing $6 each, prove that the cost of type B books
    is 6(100 - a) dollars. -/
theorem cost_of_type_b_books (a : ℕ) : ℕ :=
  let total_books : ℕ := 100
  let price_b : ℕ := 6
  let num_b : ℕ := total_books - a
  price_b * num_b

#check cost_of_type_b_books

end NUMINAMATH_CALUDE_cost_of_type_b_books_l2028_202879


namespace NUMINAMATH_CALUDE_probability_three_unused_rockets_expected_targets_hit_l2028_202809

/-- Represents a rocket artillery system -/
structure RocketSystem where
  totalRockets : ℕ
  maxShotsPerTarget : ℕ
  hitProbability : ℝ

/-- Calculates the probability of having exactly 3 unused rockets after firing at 5 targets -/
def probabilityThreeUnusedRockets (system : RocketSystem) : ℝ :=
  10 * system.hitProbability^3 * (1 - system.hitProbability)^2

/-- Calculates the expected number of targets hit when firing at 9 targets -/
def expectedTargetsHit (system : RocketSystem) : ℝ :=
  10 * system.hitProbability - system.hitProbability^10

/-- Theorem stating the probability of having exactly 3 unused rockets after firing at 5 targets -/
theorem probability_three_unused_rockets 
  (system : RocketSystem) 
  (h1 : system.totalRockets = 10) 
  (h2 : system.maxShotsPerTarget = 2) :
  probabilityThreeUnusedRockets system = 10 * system.hitProbability^3 * (1 - system.hitProbability)^2 := by
  sorry

/-- Theorem stating the expected number of targets hit when firing at 9 targets -/
theorem expected_targets_hit 
  (system : RocketSystem) 
  (h1 : system.totalRockets = 10) 
  (h2 : system.maxShotsPerTarget = 2) :
  expectedTargetsHit system = 10 * system.hitProbability - system.hitProbability^10 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_unused_rockets_expected_targets_hit_l2028_202809


namespace NUMINAMATH_CALUDE_negative_sqrt_ten_less_than_negative_three_l2028_202868

theorem negative_sqrt_ten_less_than_negative_three :
  -Real.sqrt 10 < -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_ten_less_than_negative_three_l2028_202868


namespace NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l2028_202820

theorem right_triangle_area_and_perimeter :
  ∀ (a b c : ℝ),
  a = 5 ∧ c = 13 ∧ a^2 + b^2 = c^2 →
  (1/2 * a * b = 30) ∧ (a + b + c = 30) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l2028_202820


namespace NUMINAMATH_CALUDE_interior_angles_sum_l2028_202810

theorem interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 3240) → (180 * ((n + 4) - 2) = 3960) := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l2028_202810


namespace NUMINAMATH_CALUDE_power_division_sum_product_l2028_202832

theorem power_division_sum_product : (-6)^6 / 6^4 + 4^3 - 7^2 * 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_sum_product_l2028_202832


namespace NUMINAMATH_CALUDE_minimal_area_circle_circle_center_on_line_l2028_202871

-- Define the points A and B
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (-2, -5)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define a circle passing through two points
def circle_through_points (center : ℝ × ℝ) (r : ℝ) : Prop :=
  (center.1 - A.1)^2 + (center.2 - A.2)^2 = r^2 ∧
  (center.1 - B.1)^2 + (center.2 - B.2)^2 = r^2

-- Theorem for minimal area circle
theorem minimal_area_circle :
  ∀ (center : ℝ × ℝ) (r : ℝ),
  circle_through_points center r →
  (∀ (center' : ℝ × ℝ) (r' : ℝ), circle_through_points center' r' → r ≤ r') →
  center = (0, -4) ∧ r^2 = 5 :=
sorry

-- Theorem for circle with center on the line
theorem circle_center_on_line :
  ∀ (center : ℝ × ℝ) (r : ℝ),
  circle_through_points center r →
  line_eq center.1 center.2 →
  center = (-1, -2) ∧ r^2 = 10 :=
sorry

end NUMINAMATH_CALUDE_minimal_area_circle_circle_center_on_line_l2028_202871


namespace NUMINAMATH_CALUDE_sticker_distribution_l2028_202874

/-- The number of ways to distribute n identical objects into k distinct containers,
    with each container receiving at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 10 identical stickers onto 5 sheets of paper,
    with each sheet receiving at least one sticker -/
theorem sticker_distribution : distribute 10 5 = 7 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2028_202874


namespace NUMINAMATH_CALUDE_symmetric_curve_l2028_202837

/-- The equation of a curve symmetric to y^2 = 4x with respect to the line x = 2 -/
theorem symmetric_curve (x y : ℝ) : 
  (∀ x₀ y₀ : ℝ, y₀^2 = 4*x₀ → (4 - x₀)^2 = 4*(2 - (4 - x₀))) → 
  y^2 = 16 - 4*x :=
sorry

end NUMINAMATH_CALUDE_symmetric_curve_l2028_202837


namespace NUMINAMATH_CALUDE_second_player_eats_53_seeds_l2028_202827

/-- The number of seeds eaten by the first player -/
def first_player_seeds : ℕ := 78

/-- The number of seeds eaten by the second player -/
def second_player_seeds : ℕ := 53

/-- The number of seeds eaten by the third player -/
def third_player_seeds : ℕ := second_player_seeds + 30

/-- The total number of seeds eaten by all players -/
def total_seeds : ℕ := 214

/-- Theorem stating that the given conditions result in the second player eating 53 seeds -/
theorem second_player_eats_53_seeds :
  first_player_seeds + second_player_seeds + third_player_seeds = total_seeds :=
by sorry

end NUMINAMATH_CALUDE_second_player_eats_53_seeds_l2028_202827


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l2028_202886

-- Define the universal set U
def U : Set Nat := {1, 3, 5, 6, 8}

-- Define set A
def A : Set Nat := {1, 6}

-- Define set B
def B : Set Nat := {5, 6, 8}

-- Theorem to prove
theorem complement_intersection_equals_set :
  (U \ A) ∩ B = {5, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l2028_202886


namespace NUMINAMATH_CALUDE_probability_closer_to_center_l2028_202834

theorem probability_closer_to_center (r : ℝ) (h : r > 0) :
  let outer_circle_area := π * r^2
  let inner_circle_area := π * r
  let probability := inner_circle_area / outer_circle_area
  probability = 1/4 := by
sorry

end NUMINAMATH_CALUDE_probability_closer_to_center_l2028_202834


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l2028_202878

/-- A polynomial is a perfect square if it can be expressed as (ax + b)^2 for some real numbers a and b -/
def is_perfect_square (p : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, p x = (a * x + b)^2

/-- The given polynomial -/
def polynomial (m : ℝ) (x : ℝ) : ℝ := m - 10*x + x^2

theorem perfect_square_polynomial (m : ℝ) :
  is_perfect_square (polynomial m) → m = 25 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l2028_202878


namespace NUMINAMATH_CALUDE_legs_walking_on_ground_l2028_202853

theorem legs_walking_on_ground (num_horses : ℕ) (num_men : ℕ) : 
  num_horses = 12 →
  num_men = num_horses →
  num_horses * 4 + (num_men / 2) * 2 = 60 :=
by sorry

end NUMINAMATH_CALUDE_legs_walking_on_ground_l2028_202853


namespace NUMINAMATH_CALUDE_single_weighing_correctness_check_l2028_202854

/-- Represents a weight with its mass and marking -/
structure Weight where
  mass : ℝ
  marking : ℝ

/-- Represents the position of a weight on the scale -/
structure Position where
  weight : Weight
  distance : ℝ

/-- Calculates the moment of a weight at a given position -/
def moment (p : Position) : ℝ := p.weight.mass * p.distance

/-- Theorem: It's always possible to check if all markings are correct in a single weighing -/
theorem single_weighing_correctness_check 
  (weights : Finset Weight) 
  (hweights : weights.Nonempty) 
  (hmasses : ∀ w ∈ weights, ∃ w' ∈ weights, w.mass = w'.marking) 
  (hmarkings : ∀ w ∈ weights, ∃ w' ∈ weights, w.marking = w'.mass) :
  ∃ (left right : Finset Position),
    (∀ p ∈ left, p.weight ∈ weights) ∧
    (∀ p ∈ right, p.weight ∈ weights) ∧
    (left.sum moment = right.sum moment ↔ 
      ∀ w ∈ weights, w.mass = w.marking) :=
sorry

end NUMINAMATH_CALUDE_single_weighing_correctness_check_l2028_202854


namespace NUMINAMATH_CALUDE_geometry_textbook_weight_l2028_202859

theorem geometry_textbook_weight 
  (chemistry_weight : Real) 
  (weight_difference : Real) 
  (h1 : chemistry_weight = 7.12)
  (h2 : weight_difference = 6.5)
  (h3 : chemistry_weight = geometry_weight + weight_difference) :
  geometry_weight = 0.62 :=
by
  sorry

end NUMINAMATH_CALUDE_geometry_textbook_weight_l2028_202859


namespace NUMINAMATH_CALUDE_largest_four_digit_perfect_cube_l2028_202822

theorem largest_four_digit_perfect_cube : 
  ∀ n : ℕ, n ≤ 9999 → n ≥ 1000 → (∃ m : ℕ, n = m^3) → n ≤ 9261 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_perfect_cube_l2028_202822


namespace NUMINAMATH_CALUDE_middle_number_in_ratio_l2028_202873

theorem middle_number_in_ratio (a b c : ℝ) : 
  a / b = 2 / 3 → 
  b / c = 3 / 4 → 
  a^2 + c^2 = 180 → 
  b = 9 := by
sorry

end NUMINAMATH_CALUDE_middle_number_in_ratio_l2028_202873


namespace NUMINAMATH_CALUDE_square_root_problem_l2028_202835

theorem square_root_problem (x : ℝ) :
  (Real.sqrt 1.21) / (Real.sqrt x) + (Real.sqrt 1.44) / (Real.sqrt 0.49) = 3.0892857142857144 →
  x = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l2028_202835


namespace NUMINAMATH_CALUDE_gcd_370_1332_l2028_202831

theorem gcd_370_1332 : Nat.gcd 370 1332 = 74 := by
  sorry

end NUMINAMATH_CALUDE_gcd_370_1332_l2028_202831


namespace NUMINAMATH_CALUDE_last_remaining_card_l2028_202866

/-- The largest power of 2 less than or equal to n -/
def largestPowerOf2 (n : ℕ) : ℕ :=
  (Nat.log2 n).succ

/-- The process of eliminating cards -/
def cardElimination (n : ℕ) : ℕ :=
  let L := largestPowerOf2 n
  2 * (n - 2^L) + 1

theorem last_remaining_card (n : ℕ) (h : n > 0) :
  ∃ (k : ℕ), k ≤ n ∧ cardElimination n = k :=
sorry

end NUMINAMATH_CALUDE_last_remaining_card_l2028_202866


namespace NUMINAMATH_CALUDE_total_bills_and_coins_l2028_202875

/-- Represents the payment details for a grocery bill -/
structure GroceryPayment where
  totalBill : ℕ
  billValue : ℕ
  coinValue : ℕ
  numBills : ℕ
  numCoins : ℕ

/-- Theorem stating the total number of bills and coins used in the payment -/
theorem total_bills_and_coins (payment : GroceryPayment) 
  (h1 : payment.totalBill = 285)
  (h2 : payment.billValue = 20)
  (h3 : payment.coinValue = 5)
  (h4 : payment.numBills = 11)
  (h5 : payment.numCoins = 11)
  : payment.numBills + payment.numCoins = 22 := by
  sorry


end NUMINAMATH_CALUDE_total_bills_and_coins_l2028_202875


namespace NUMINAMATH_CALUDE_binomial_10_3_l2028_202825

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l2028_202825


namespace NUMINAMATH_CALUDE_polynomial_inequality_l2028_202862

theorem polynomial_inequality (x : ℝ) (h : x^2 - 5*x + 6 > 0) :
  x^3 - 5*x^2 + 6*x + 1 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l2028_202862


namespace NUMINAMATH_CALUDE_intersection_implies_m_values_l2028_202807

theorem intersection_implies_m_values (m : ℝ) : 
  let M : Set ℝ := {4, 5, -3*m}
  let N : Set ℝ := {-9, 3}
  (M ∩ N).Nonempty → m = 3 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_values_l2028_202807


namespace NUMINAMATH_CALUDE_trig_identity_l2028_202828

theorem trig_identity (α : ℝ) (h : Real.sin (π / 4 + α) = 1 / 2) :
  Real.sin (5 * π / 4 + α) / Real.cos (9 * π / 4 + α) * Real.cos (7 * π / 4 - α) = - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2028_202828


namespace NUMINAMATH_CALUDE_zero_score_probability_l2028_202802

def num_balls : ℕ := 6
def num_red : ℕ := 1
def num_yellow : ℕ := 2
def num_blue : ℕ := 3
def num_draws : ℕ := 3

def score_red : ℤ := 1
def score_yellow : ℤ := 0
def score_blue : ℤ := -1

def prob_zero_score : ℚ := 11 / 54

theorem zero_score_probability :
  (num_balls = num_red + num_yellow + num_blue) →
  (prob_zero_score = (num_yellow^num_draws + num_red * num_yellow * num_blue * 6) / num_balls^num_draws) :=
by sorry

end NUMINAMATH_CALUDE_zero_score_probability_l2028_202802


namespace NUMINAMATH_CALUDE_base_edge_length_is_six_l2028_202830

/-- A square pyramid with a hemisphere resting on its base -/
structure PyramidWithHemisphere where
  /-- The height of the pyramid -/
  height : ℝ
  /-- The radius of the hemisphere -/
  radius : ℝ
  /-- The hemisphere is tangent to the other four faces of the pyramid -/
  is_tangent : Bool

/-- The edge length of the base of the pyramid -/
def base_edge_length (p : PyramidWithHemisphere) : ℝ :=
  sorry

/-- Theorem stating the edge length of the base of the pyramid is 6 -/
theorem base_edge_length_is_six (p : PyramidWithHemisphere) 
  (h1 : p.height = 12)
  (h2 : p.radius = 4)
  (h3 : p.is_tangent = true) : 
  base_edge_length p = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_edge_length_is_six_l2028_202830

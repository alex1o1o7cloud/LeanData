import Mathlib

namespace sum_and_product_to_a_plus_b_l2271_227103

theorem sum_and_product_to_a_plus_b (a b : ℝ) 
  (sum_eq : (a + Real.sqrt b) + (a - Real.sqrt b) = -8)
  (product_eq : (a + Real.sqrt b) * (a - Real.sqrt b) = 4) :
  a + b = 8 := by
  sorry

end sum_and_product_to_a_plus_b_l2271_227103


namespace union_when_m_neg_one_subset_iff_m_range_disjoint_iff_m_range_l2271_227159

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Theorem 1
theorem union_when_m_neg_one :
  A ∪ B (-1) = {x | -2 < x ∧ x < 3} := by sorry

-- Theorem 2
theorem subset_iff_m_range :
  ∀ m, A ⊆ B m ↔ m ≤ -2 := by sorry

-- Theorem 3
theorem disjoint_iff_m_range :
  ∀ m, A ∩ B m = ∅ ↔ 0 ≤ m := by sorry

end union_when_m_neg_one_subset_iff_m_range_disjoint_iff_m_range_l2271_227159


namespace simplify_fraction_expression_l2271_227136

theorem simplify_fraction_expression (a b : ℝ) (h : a + b ≠ 0) (h' : a + 2*b ≠ 0) (h'' : a^2 - b^2 ≠ 0) :
  (((a - b) / (a + 2*b)) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2))) - 2 = -a / (a + b) := by
  sorry

end simplify_fraction_expression_l2271_227136


namespace inverse_proposition_false_l2271_227164

theorem inverse_proposition_false : ¬ (∀ a b : ℝ, a^2 = b^2 → a = b) := by sorry

end inverse_proposition_false_l2271_227164


namespace sale_price_for_50_percent_profit_l2271_227196

/-- Represents the cost and pricing of an article -/
structure Article where
  cost : ℝ
  profit_price : ℝ
  loss_price : ℝ

/-- The conditions of the problem -/
def problem_conditions (a : Article) : Prop :=
  a.profit_price - a.cost = a.cost - a.loss_price ∧
  a.profit_price = 892 ∧
  1005 = 1.5 * a.cost

/-- The theorem to be proved -/
theorem sale_price_for_50_percent_profit (a : Article) 
  (h : problem_conditions a) : 
  1.5 * a.cost = 1005 := by
  sorry

#check sale_price_for_50_percent_profit

end sale_price_for_50_percent_profit_l2271_227196


namespace six_awards_four_students_l2271_227160

/-- The number of ways to distribute awards to students. -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- The theorem stating the correct number of ways to distribute 6 awards to 4 students. -/
theorem six_awards_four_students :
  distribute_awards 6 4 = 1260 :=
sorry

end six_awards_four_students_l2271_227160


namespace P_intersect_Q_eq_P_l2271_227117

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}

theorem P_intersect_Q_eq_P : P ∩ Q = P := by
  sorry

end P_intersect_Q_eq_P_l2271_227117


namespace expression_simplification_l2271_227135

theorem expression_simplification (x : ℝ) (h : x = 1) :
  (x^2 - 4*x + 4) / (2*x) / ((x^2 - 2*x) / x^2) + 1 = 1/2 := by
  sorry

end expression_simplification_l2271_227135


namespace total_height_increase_two_centuries_l2271_227148

/-- Represents the increase in height (in meters) per decade for a specific species of plants -/
def height_increase_per_decade : ℕ := 90

/-- Represents the number of decades in 2 centuries -/
def decades_in_two_centuries : ℕ := 20

/-- Theorem stating that the total increase in height over 2 centuries is 1800 meters -/
theorem total_height_increase_two_centuries : 
  height_increase_per_decade * decades_in_two_centuries = 1800 := by
  sorry

end total_height_increase_two_centuries_l2271_227148


namespace repeating_decimal_to_fraction_l2271_227127

theorem repeating_decimal_to_fraction :
  ∀ (x : ℚ), (∃ (n : ℕ), x = 0.56 + 0.0056 * (1 - (1/100)^n) / (1 - 1/100)) →
  x = 56/99 := by
  sorry

end repeating_decimal_to_fraction_l2271_227127


namespace grandmas_farm_l2271_227191

theorem grandmas_farm (chickens ducks : ℕ) : 
  chickens = 4 * ducks ∧ chickens = ducks + 600 → chickens = 800 ∧ ducks = 200 := by
  sorry

end grandmas_farm_l2271_227191


namespace maggots_eaten_first_correct_l2271_227166

/-- The number of maggots eaten by the beetle in the first feeding -/
def maggots_eaten_first : ℕ := 17

/-- The total number of maggots served -/
def total_maggots : ℕ := 20

/-- The number of maggots eaten in the second feeding -/
def maggots_eaten_second : ℕ := 3

/-- Theorem stating that the number of maggots eaten in the first feeding is correct -/
theorem maggots_eaten_first_correct : 
  maggots_eaten_first = total_maggots - maggots_eaten_second := by
  sorry

end maggots_eaten_first_correct_l2271_227166


namespace valid_pairs_count_l2271_227105

/-- A function that checks if a positive integer has a zero digit. -/
def has_zero_digit (n : ℕ+) : Prop := sorry

/-- The count of ordered pairs (a,b) of positive integers where a + b = 500 and neither a nor b has a zero digit. -/
def count_valid_pairs : ℕ := sorry

/-- Theorem stating that the count of valid pairs is 329. -/
theorem valid_pairs_count : count_valid_pairs = 329 := by sorry

end valid_pairs_count_l2271_227105


namespace monic_quartic_polynomial_value_l2271_227104

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_polynomial_value (f : ℝ → ℝ) :
  MonicQuarticPolynomial f →
  f (-2) = -4 →
  f 1 = -1 →
  f 3 = -9 →
  f (-4) = -16 →
  f 2 = -28 := by
    sorry

end monic_quartic_polynomial_value_l2271_227104


namespace abs_x_plus_y_equals_three_l2271_227124

theorem abs_x_plus_y_equals_three (x y : ℝ) 
  (eq1 : |x| + 2*y = 2) 
  (eq2 : 2*|x| + y = 7) : 
  |x| + y = 3 := by
  sorry

end abs_x_plus_y_equals_three_l2271_227124


namespace product_from_lcm_and_gcd_l2271_227184

theorem product_from_lcm_and_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 90) 
  (h2 : Nat.gcd a b = 10) : 
  a * b = 900 := by
  sorry

end product_from_lcm_and_gcd_l2271_227184


namespace binomial_fraction_integer_l2271_227141

theorem binomial_fraction_integer (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  (∃ m : ℕ, n + 2 = m * (k + 2)) ↔ 
  ∃ z : ℤ, z = (2*n - 3*k - 2) * (n.choose k) / (k + 2) :=
sorry

end binomial_fraction_integer_l2271_227141


namespace value_k_std_dev_below_mean_two_std_dev_below_mean_l2271_227154

/-- For a normal distribution with mean μ and standard deviation σ,
    the value that is exactly k standard deviations less than the mean is μ - k * σ -/
theorem value_k_std_dev_below_mean (μ σ k : ℝ) :
  μ - k * σ = μ - k * σ := by sorry

/-- For a normal distribution with mean 12 and standard deviation 1.2,
    the value that is exactly 2 standard deviations less than the mean is 9.6 -/
theorem two_std_dev_below_mean :
  let μ : ℝ := 12  -- mean
  let σ : ℝ := 1.2 -- standard deviation
  let k : ℝ := 2   -- number of standard deviations below mean
  μ - k * σ = 9.6 := by sorry

end value_k_std_dev_below_mean_two_std_dev_below_mean_l2271_227154


namespace divisor_implies_value_l2271_227181

theorem divisor_implies_value (k : ℕ) : 
  21^k ∣ 435961 → 7^k - k^7 = 1 := by
  sorry

end divisor_implies_value_l2271_227181


namespace janice_age_l2271_227123

def current_year : ℕ := 2021
def mark_birth_year : ℕ := 1976

def graham_age_difference : ℕ := 3

theorem janice_age :
  let mark_age : ℕ := current_year - mark_birth_year
  let graham_age : ℕ := mark_age - graham_age_difference
  let janice_age : ℕ := graham_age / 2
  janice_age = 21 := by sorry

end janice_age_l2271_227123


namespace novel_reading_time_l2271_227151

theorem novel_reading_time (total_pages : ℕ) (rate_alice rate_bob rate_chandra : ℚ) :
  total_pages = 760 ∧ 
  rate_alice = 1 / 20 ∧ 
  rate_bob = 1 / 45 ∧ 
  rate_chandra = 1 / 30 →
  ∃ t : ℚ, t = 7200 ∧ 
    t * rate_alice + t * rate_bob + t * rate_chandra = total_pages :=
by sorry

end novel_reading_time_l2271_227151


namespace club_leadership_combinations_l2271_227100

/-- Represents the total number of students in the club -/
def total_students : ℕ := 30

/-- Represents the number of boys in the club -/
def num_boys : ℕ := 18

/-- Represents the number of girls in the club -/
def num_girls : ℕ := 12

/-- Represents the number of boy seniors (equal to boy juniors) -/
def num_boy_seniors : ℕ := num_boys / 2

/-- Represents the number of girl seniors (equal to girl juniors) -/
def num_girl_seniors : ℕ := num_girls / 2

/-- Represents the number of genders (boys and girls) -/
def num_genders : ℕ := 2

/-- Represents the number of class years (senior and junior) -/
def num_class_years : ℕ := 2

theorem club_leadership_combinations : 
  (num_genders * num_class_years * num_boy_seniors * num_boy_seniors) + 
  (num_genders * num_class_years * num_girl_seniors * num_girl_seniors) = 324 := by
  sorry

end club_leadership_combinations_l2271_227100


namespace crocodile_coloring_exists_l2271_227161

/-- A coloring function for an infinite checkerboard -/
def ColoringFunction := ℤ → ℤ → Fin 2

/-- The "crocodile" move on a checkerboard -/
def crocodileMove (m n : ℤ) (x y : ℤ) : Set (ℤ × ℤ) :=
  {(x + m, y + n), (x + m, y - n), (x - m, y + n), (x - m, y - n),
   (x + n, y + m), (x + n, y - m), (x - n, y + m), (x - n, y - m)}

/-- Theorem: For any m and n, there exists a coloring function such that
    any two squares connected by a crocodile move have different colors -/
theorem crocodile_coloring_exists (m n : ℤ) :
  ∃ (f : ColoringFunction),
    ∀ (x y : ℤ), ∀ (x' y' : ℤ), (x', y') ∈ crocodileMove m n x y →
      f x y ≠ f x' y' := by
  sorry

end crocodile_coloring_exists_l2271_227161


namespace right_triangle_base_length_l2271_227111

/-- A right-angled triangle with one angle of 30° and base length of 6 units has a base length of 6 units. -/
theorem right_triangle_base_length (a b c : ℝ) (θ : ℝ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  θ = π/6 →  -- 30° angle in radians
  a = 6 →  -- base length
  a = 6 :=
by sorry

end right_triangle_base_length_l2271_227111


namespace michael_has_eight_robots_l2271_227116

/-- The number of animal robots Michael has -/
def michaels_robots : ℕ := sorry

/-- The number of animal robots Tom has -/
def toms_robots : ℕ := 16

/-- Tom has twice as many animal robots as Michael -/
axiom twice_as_many : toms_robots = 2 * michaels_robots

theorem michael_has_eight_robots : michaels_robots = 8 := by sorry

end michael_has_eight_robots_l2271_227116


namespace triangle_side_equation_l2271_227158

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the altitude equations
def altitude1 (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0
def altitude2 (x y : ℝ) : Prop := x + y = 0

-- Define the theorem
theorem triangle_side_equation (ABC : Triangle) 
  (h1 : ABC.A = (1, 2))
  (h2 : altitude1 (ABC.B.1) (ABC.B.2) ∨ altitude1 (ABC.C.1) (ABC.C.2))
  (h3 : altitude2 (ABC.B.1) (ABC.B.2) ∨ altitude2 (ABC.C.1) (ABC.C.2)) :
  ∃ (a b c : ℝ), a * ABC.B.1 + b * ABC.B.2 + c = 0 ∧
                 a * ABC.C.1 + b * ABC.C.2 + c = 0 ∧
                 (a, b, c) = (2, 3, 7) := by
  sorry

end triangle_side_equation_l2271_227158


namespace playground_max_area_l2271_227146

theorem playground_max_area :
  ∀ (width height : ℕ),
    width + height = 75 →
    width * height ≤ 1406 :=
by
  sorry

end playground_max_area_l2271_227146


namespace exactly_one_valid_sequence_of_length_15_l2271_227172

/-- Represents a sequence of A's and B's -/
inductive ABSequence
  | empty : ABSequence
  | cons_a : ABSequence → ABSequence
  | cons_b : ABSequence → ABSequence

/-- Returns true if the given sequence satisfies the run length conditions -/
def valid_sequence (s : ABSequence) : Bool :=
  sorry

/-- Returns the length of the sequence -/
def sequence_length (s : ABSequence) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem exactly_one_valid_sequence_of_length_15 :
  ∃! (s : ABSequence), valid_sequence s ∧ sequence_length s = 15 :=
  sorry

end exactly_one_valid_sequence_of_length_15_l2271_227172


namespace prove_num_sodas_l2271_227192

def sandwich_cost : ℚ := 149/100
def soda_cost : ℚ := 87/100
def total_cost : ℚ := 646/100
def num_sandwiches : ℕ := 2

def num_sodas : ℕ := 4

theorem prove_num_sodas : 
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = total_cost := by
  sorry

#eval num_sodas

end prove_num_sodas_l2271_227192


namespace curve_T_and_fixed_point_l2271_227156

-- Define the points A, B, C, and O
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (0, -1)
def O : ℝ × ℝ := (0, 0)

-- Define the condition for point M
def condition_M (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  (x + 1) * (x - 1) + y * y = y * (y + 1)

-- Define the curve T
def curve_T (x y : ℝ) : Prop := y = x^2 - 1

-- Define the tangent line at point P
def tangent_line (P : ℝ × ℝ) (x y : ℝ) : Prop :=
  let (x₀, y₀) := P
  y - y₀ = 2 * x₀ * (x - x₀)

-- Define the line y = -5/4
def line_y_eq_neg_5_4 (x y : ℝ) : Prop := y = -5/4

-- Define the circle with diameter PQ
def circle_PQ (P Q H : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xq, yq) := Q
  let (xh, yh) := H
  (xh - xp) * (xh - xq) + (yh - yp) * (yh - yq) = 0

-- State the theorem
theorem curve_T_and_fixed_point :
  -- Part 1: The trajectory of point M is curve T
  (∀ M : ℝ × ℝ, condition_M M ↔ curve_T M.1 M.2) ∧
  -- Part 2: The circle with diameter PQ passes through a fixed point
  (∀ P : ℝ × ℝ, P.1 ≠ 0 → curve_T P.1 P.2 →
    ∃ Q : ℝ × ℝ,
      tangent_line P Q.1 Q.2 ∧
      line_y_eq_neg_5_4 Q.1 Q.2 ∧
      circle_PQ P Q (0, -3/4)) := by
  sorry

end curve_T_and_fixed_point_l2271_227156


namespace cos_2alpha_problem_l2271_227173

theorem cos_2alpha_problem (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →
  Real.cos (2 * α) = -(Real.sqrt 5 / 3) := by
sorry

end cos_2alpha_problem_l2271_227173


namespace hyperbola_asymptotes_l2271_227118

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = (3/4) * x ∨ y = -(3/4) * x

/-- Theorem: The asymptotes of the given hyperbola -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end hyperbola_asymptotes_l2271_227118


namespace marcella_shoes_l2271_227178

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_remaining_pairs (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - shoes_lost

theorem marcella_shoes :
  max_remaining_pairs 20 9 = 11 := by
  sorry

end marcella_shoes_l2271_227178


namespace circles_externally_tangent_l2271_227152

/-- Circle O₁ with equation x² + y² = 1 -/
def circle_O₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle O₂ with equation x² + y² - 6x + 8y + 9 = 0 -/
def circle_O₂ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 + 8*p.2 + 9 = 0}

/-- The center of circle O₁ -/
def center_O₁ : ℝ × ℝ := (0, 0)

/-- The radius of circle O₁ -/
def radius_O₁ : ℝ := 1

/-- The center of circle O₂ -/
def center_O₂ : ℝ × ℝ := (3, -4)

/-- The radius of circle O₂ -/
def radius_O₂ : ℝ := 4

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c₁ c₂ : Set (ℝ × ℝ)) (center₁ center₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  ((center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2).sqrt = r₁ + r₂

theorem circles_externally_tangent :
  externally_tangent circle_O₁ circle_O₂ center_O₁ center_O₂ radius_O₁ radius_O₂ := by
  sorry

end circles_externally_tangent_l2271_227152


namespace multiply_65_55_l2271_227198

theorem multiply_65_55 : 65 * 55 = 3575 := by sorry

end multiply_65_55_l2271_227198


namespace one_pencil_one_pen_cost_l2271_227137

def pencil_cost : ℝ → ℝ → Prop := λ p q ↦ 3 * p + 2 * q = 3.75
def pen_cost : ℝ → ℝ → Prop := λ p q ↦ 2 * p + 3 * q = 4.05

theorem one_pencil_one_pen_cost (p q : ℝ) 
  (h1 : pencil_cost p q) (h2 : pen_cost p q) : 
  p + q = 1.56 := by
  sorry

end one_pencil_one_pen_cost_l2271_227137


namespace same_digit_sum_in_arithmetic_progression_l2271_227169

-- Define an arithmetic progression of natural numbers
def arithmeticProgression (a d : ℕ) : ℕ → ℕ := λ n => a + n * d

-- Define the sum of digits function
def sumOfDigits : ℕ → ℕ := sorry

theorem same_digit_sum_in_arithmetic_progression (a d : ℕ) :
  ∃ (k l : ℕ), k ≠ l ∧ sumOfDigits (arithmeticProgression a d k) = sumOfDigits (arithmeticProgression a d l) := by
  sorry

end same_digit_sum_in_arithmetic_progression_l2271_227169


namespace triangle_problem_l2271_227187

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions and the statements to prove --/
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * t.b * cos t.C + t.c = 2 * t.a) 
  (h2 : cos t.A = 1 / 7) : 
  t.B = π / 3 ∧ t.c / t.a = 5 / 8 := by
  sorry

end triangle_problem_l2271_227187


namespace parallelepipeds_in_4x4x4_cube_l2271_227125

/-- The number of distinct rectangular parallelepipeds in a cube of size n --/
def count_parallelepipeds (n : ℕ) : ℕ :=
  (n + 1).choose 2 ^ 3

/-- Theorem stating that in a 4 × 4 × 4 cube, there are 1000 distinct rectangular parallelepipeds --/
theorem parallelepipeds_in_4x4x4_cube :
  count_parallelepipeds 4 = 1000 := by sorry

end parallelepipeds_in_4x4x4_cube_l2271_227125


namespace expression_bounds_bounds_are_tight_l2271_227108

theorem expression_bounds (a b c d e : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) : 
  5 / Real.sqrt 2 ≤ Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
    Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-e)^2) + Real.sqrt (e^2 + (1-a)^2) ∧
  Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + Real.sqrt (c^2 + (1-d)^2) + 
    Real.sqrt (d^2 + (1-e)^2) + Real.sqrt (e^2 + (1-a)^2) ≤ 5 :=
by sorry

theorem bounds_are_tight : 
  ∃ (a b c d e : ℝ), (0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1) ∧ (0 ≤ c ∧ c ≤ 1) ∧ 
    (0 ≤ d ∧ d ≤ 1) ∧ (0 ≤ e ∧ e ≤ 1) ∧
    Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + Real.sqrt (c^2 + (1-d)^2) + 
    Real.sqrt (d^2 + (1-e)^2) + Real.sqrt (e^2 + (1-a)^2) = 5 / Real.sqrt 2 ∧
  ∃ (a' b' c' d' e' : ℝ), (0 ≤ a' ∧ a' ≤ 1) ∧ (0 ≤ b' ∧ b' ≤ 1) ∧ (0 ≤ c' ∧ c' ≤ 1) ∧ 
    (0 ≤ d' ∧ d' ≤ 1) ∧ (0 ≤ e' ∧ e' ≤ 1) ∧
    Real.sqrt (a'^2 + (1-b')^2) + Real.sqrt (b'^2 + (1-c')^2) + Real.sqrt (c'^2 + (1-d')^2) + 
    Real.sqrt (d'^2 + (1-e')^2) + Real.sqrt (e'^2 + (1-a')^2) = 5 :=
by sorry

end expression_bounds_bounds_are_tight_l2271_227108


namespace smallest_prime_after_seven_nonprimes_l2271_227126

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns true if all numbers in a list are non-prime, false otherwise -/
def allNonPrime (list : List ℕ) : Prop := sorry

theorem smallest_prime_after_seven_nonprimes :
  ∃ (n : ℕ), 
    (isPrime (nthPrime n)) ∧ 
    (allNonPrime [nthPrime (n-1) + 1, nthPrime (n-1) + 2, nthPrime (n-1) + 3, 
                  nthPrime (n-1) + 4, nthPrime (n-1) + 5, nthPrime (n-1) + 6, 
                  nthPrime (n-1) + 7]) ∧
    (nthPrime n = 67) ∧
    (∀ (m : ℕ), m < n → 
      ¬(isPrime (nthPrime m) ∧ 
        allNonPrime [nthPrime (m-1) + 1, nthPrime (m-1) + 2, nthPrime (m-1) + 3, 
                     nthPrime (m-1) + 4, nthPrime (m-1) + 5, nthPrime (m-1) + 6, 
                     nthPrime (m-1) + 7])) :=
sorry

end smallest_prime_after_seven_nonprimes_l2271_227126


namespace max_value_5x_3y_l2271_227133

theorem max_value_5x_3y (x y : ℝ) (h : x^2 + y^2 = 10*x + 8*y + 10) :
  ∃ (M : ℝ), M = 105 ∧ 5*x + 3*y ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 10*x₀ + 8*y₀ + 10 ∧ 5*x₀ + 3*y₀ = M :=
sorry

end max_value_5x_3y_l2271_227133


namespace largest_k_for_g_range_l2271_227129

/-- The function g(x) defined as x^2 - 7x + k -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 - 7*x + k

/-- Theorem stating that the largest value of k such that 4 is in the range of g(x) is 65/4 -/
theorem largest_k_for_g_range : 
  (∃ (k : ℝ), ∀ (k' : ℝ), (∃ (x : ℝ), g k' x = 4) → k' ≤ k) ∧ 
  (∃ (x : ℝ), g (65/4) x = 4) := by
  sorry

end largest_k_for_g_range_l2271_227129


namespace complement_of_A_in_U_l2271_227185

def U : Set Nat := {1,2,3,4,5,6,7}
def A : Set Nat := {2,4,5}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {1,3,6,7} := by sorry

end complement_of_A_in_U_l2271_227185


namespace right_triangle_max_ratio_squared_l2271_227182

theorem right_triangle_max_ratio_squared (a b x y : ℝ) : 
  0 < a → 0 < b → a ≥ b → 
  x^2 + b^2 = a^2 + y^2 → -- right triangle condition
  x + y = a → 
  x ≥ 0 → y ≥ 0 → 
  (a / b)^2 ≤ 1 := by
sorry

end right_triangle_max_ratio_squared_l2271_227182


namespace quadratic_roots_range_l2271_227121

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ 
    2 * x₁^2 + (m + 1) * x₁ + m = 0 ∧
    2 * x₂^2 + (m + 1) * x₂ + m = 0) →
  m < 0 :=
by sorry

end quadratic_roots_range_l2271_227121


namespace sum_of_roots_l2271_227109

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  c^2 + a*c + b = 0 →
  d^2 + a*d + b = 0 →
  a^2 + c*a + d = 0 →
  b^2 + c*b + d = 0 →
  a + b + c + d = -2 := by
sorry

end sum_of_roots_l2271_227109


namespace commission_calculation_l2271_227162

/-- Calculates the commission earned from selling a coupe and an SUV --/
theorem commission_calculation (coupe_price : ℝ) (suv_price_multiplier : ℝ) (commission_rate : ℝ) :
  coupe_price = 30000 →
  suv_price_multiplier = 2 →
  commission_rate = 0.02 →
  coupe_price * suv_price_multiplier * commission_rate + coupe_price * commission_rate = 1800 := by
  sorry

end commission_calculation_l2271_227162


namespace complex_square_roots_l2271_227179

theorem complex_square_roots : ∃ (z₁ z₂ : ℂ),
  z₁^2 = -100 - 49*I ∧ 
  z₂^2 = -100 - 49*I ∧ 
  z₁ = (7*Real.sqrt 2)/2 - (7*Real.sqrt 2)/2*I ∧
  z₂ = -(7*Real.sqrt 2)/2 + (7*Real.sqrt 2)/2*I ∧
  ∀ (z : ℂ), z^2 = -100 - 49*I → (z = z₁ ∨ z = z₂) := by
sorry

end complex_square_roots_l2271_227179


namespace min_value_theorem_l2271_227114

theorem min_value_theorem (a b c : ℝ) 
  (h : ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3) :
  ∃ m : ℝ, m = -4 ∧ ∀ k : ℝ, (∃ a' b' c' : ℝ, 
    (∀ x y : ℝ, x + 2*y - 3 ≤ a'*x + b'*y + c' ∧ a'*x + b'*y + c' ≤ x + 2*y + 3) ∧
    k = a' + 2*b' - 3*c') → m ≤ k :=
by sorry

end min_value_theorem_l2271_227114


namespace reflected_hyperbola_l2271_227189

/-- Given a hyperbola with equation xy = 1 reflected over the line y = 2x,
    the resulting hyperbola has the equation 12y² + 7xy - 12x² = 25 -/
theorem reflected_hyperbola (x y : ℝ) :
  (∃ x₀ y₀, x₀ * y₀ = 1 ∧ 
   ∃ x₁ y₁, y₁ = 2 * x₁ ∧
   ∃ x₂ y₂, (x₂ - x₀) = (y₁ - y₀) ∧ (y₂ - y₀) = -(x₁ - x₀) ∧
   x = x₂ ∧ y = y₂) →
  12 * y^2 + 7 * x * y - 12 * x^2 = 25 :=
by sorry


end reflected_hyperbola_l2271_227189


namespace johns_piggy_bank_l2271_227130

theorem johns_piggy_bank (quarters dimes nickels : ℕ) : 
  quarters = 22 →
  dimes = quarters + 3 →
  nickels = quarters - 6 →
  quarters + dimes + nickels = 63 := by
sorry

end johns_piggy_bank_l2271_227130


namespace wages_problem_l2271_227165

/-- Given a sum of money that can pay x's wages for 36 days and y's wages for 45 days,
    prove that it can pay both x and y's wages together for 20 days. -/
theorem wages_problem (S : ℝ) (x y : ℝ → ℝ) :
  (∃ (Wx Wy : ℝ), Wx > 0 ∧ Wy > 0 ∧ S = 36 * Wx ∧ S = 45 * Wy) →
  ∃ D : ℝ, D = 20 ∧ S = D * (x 1 + y 1) :=
by sorry

end wages_problem_l2271_227165


namespace basketball_score_proof_l2271_227150

theorem basketball_score_proof (two_points three_points free_throws : ℕ) : 
  (3 * three_points = 2 * two_points) →
  (free_throws = 2 * three_points) →
  (2 * two_points + 3 * three_points + free_throws = 72) →
  free_throws = 18 := by
sorry

end basketball_score_proof_l2271_227150


namespace g_at_six_l2271_227128

def g (x : ℝ) : ℝ := 2*x^4 - 19*x^3 + 30*x^2 - 12*x - 72

theorem g_at_six : g 6 = 288 := by
  sorry

end g_at_six_l2271_227128


namespace burger_composition_l2271_227186

theorem burger_composition (total_weight filler_weight : ℝ) 
  (h1 : total_weight = 150)
  (h2 : filler_weight = 45) :
  (total_weight - filler_weight) / total_weight * 100 = 70 := by
  sorry

end burger_composition_l2271_227186


namespace power_of_64_l2271_227131

theorem power_of_64 : (64 : ℝ) ^ (3/2) = 512 := by sorry

end power_of_64_l2271_227131


namespace six_balls_two_boxes_l2271_227176

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes,
    with each box containing at least one ball -/
def distribute_balls (n : ℕ) : ℕ :=
  if n < 2 then 0 else 2^(n-1) - 2

/-- The problem statement -/
theorem six_balls_two_boxes : distribute_balls 6 = 30 := by
  sorry

end six_balls_two_boxes_l2271_227176


namespace x_range_l2271_227115

theorem x_range (x : ℝ) : (x^2 - 4 < 0 ∨ |x| = 2) → x ∈ Set.Icc (-2) 2 := by
  sorry

end x_range_l2271_227115


namespace other_root_of_quadratic_l2271_227138

theorem other_root_of_quadratic (m : ℝ) : 
  (1 : ℝ)^2 + m * 1 - 5 = 0 → 
  (-5 : ℝ)^2 + m * (-5) - 5 = 0 :=
by sorry

end other_root_of_quadratic_l2271_227138


namespace complementary_angle_of_25_41_l2271_227167

-- Define a type for angles in degrees and minutes
structure Angle where
  degrees : ℕ
  minutes : ℕ

-- Define addition for Angle
def Angle.add (a b : Angle) : Angle :=
  let totalMinutes := a.minutes + b.minutes
  let extraDegrees := totalMinutes / 60
  { degrees := a.degrees + b.degrees + extraDegrees
  , minutes := totalMinutes % 60 }

-- Define subtraction for Angle
def Angle.sub (a b : Angle) : Angle :=
  let totalMinutes := (a.degrees * 60 + a.minutes) - (b.degrees * 60 + b.minutes)
  { degrees := totalMinutes / 60
  , minutes := totalMinutes % 60 }

-- Define the given angle
def givenAngle : Angle := { degrees := 25, minutes := 41 }

-- Define 90 degrees
def rightAngle : Angle := { degrees := 90, minutes := 0 }

-- Theorem statement
theorem complementary_angle_of_25_41 :
  Angle.sub rightAngle givenAngle = { degrees := 64, minutes := 19 } := by
  sorry


end complementary_angle_of_25_41_l2271_227167


namespace unique_number_in_range_l2271_227107

theorem unique_number_in_range : ∃! x : ℝ, 3 < x ∧ x < 8 ∧ 6 < x ∧ x < 10 ∧ x = 7 := by
  sorry

end unique_number_in_range_l2271_227107


namespace ladder_wood_length_l2271_227180

/-- Calculates the total length of wood needed for ladder rungs -/
theorem ladder_wood_length 
  (rung_length : ℚ)      -- Length of each rung in inches
  (rung_spacing : ℚ)     -- Space between rungs in inches
  (climb_height : ℚ)     -- Height to climb in feet
  (h1 : rung_length = 18)
  (h2 : rung_spacing = 6)
  (h3 : climb_height = 50) :
  (climb_height * 12 / rung_spacing) * (rung_length / 12) = 150 :=
by sorry

end ladder_wood_length_l2271_227180


namespace washing_machine_capacity_l2271_227140

theorem washing_machine_capacity 
  (shirts : ℕ) 
  (sweaters : ℕ) 
  (loads : ℕ) 
  (h1 : shirts = 19) 
  (h2 : sweaters = 8) 
  (h3 : loads = 3) : 
  (shirts + sweaters) / loads = 9 := by
sorry

end washing_machine_capacity_l2271_227140


namespace gcd_3060_561_l2271_227101

theorem gcd_3060_561 : Nat.gcd 3060 561 = 51 := by
  sorry

end gcd_3060_561_l2271_227101


namespace expression_value_l2271_227120

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |m| = 1) : 
  (a + b) * c * d - 2014 * m = -2014 ∨ (a + b) * c * d - 2014 * m = 2014 := by
  sorry

end expression_value_l2271_227120


namespace heather_aprons_tomorrow_l2271_227177

/-- The number of aprons Heather should sew tomorrow -/
def aprons_tomorrow (total : ℕ) (initial : ℕ) (today_multiplier : ℕ) : ℕ :=
  (total - (initial + today_multiplier * initial)) / 2

/-- Theorem: Given the conditions, Heather should sew 49 aprons tomorrow -/
theorem heather_aprons_tomorrow :
  aprons_tomorrow 150 13 3 = 49 := by
  sorry

end heather_aprons_tomorrow_l2271_227177


namespace incorrect_calculation_l2271_227188

theorem incorrect_calculation (x y : ℝ) : 
  (-2 * x^2 * y^2)^3 / (-x * y)^3 ≠ -2 * x^3 * y^3 := by
  sorry

end incorrect_calculation_l2271_227188


namespace rectangle_dimension_l2271_227174

theorem rectangle_dimension (x : ℝ) : 
  (3*x - 5 > 0) ∧ (x + 7 > 0) ∧ ((3*x - 5) * (x + 7) = 15*x - 14) → x = 3 :=
by sorry

end rectangle_dimension_l2271_227174


namespace distance_from_origin_to_point_l2271_227175

/-- The distance from the origin to the point (12, -5) in a rectangular coordinate system is 13 units. -/
theorem distance_from_origin_to_point : Real.sqrt (12^2 + (-5)^2) = 13 := by
  sorry

end distance_from_origin_to_point_l2271_227175


namespace sum_of_integers_l2271_227142

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val^2 + y.val^2 = 130) 
  (h2 : x.val * y.val = 45) : 
  x.val + y.val = 2 * Real.sqrt 55 := by
  sorry

end sum_of_integers_l2271_227142


namespace starting_lineup_combinations_l2271_227171

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem starting_lineup_combinations : choose 13 4 = 715 := by sorry

end starting_lineup_combinations_l2271_227171


namespace students_taking_neither_proof_l2271_227199

def students_taking_neither (total students_music students_art students_dance
                             students_music_art students_art_dance students_music_dance
                             students_all_three : ℕ) : ℕ :=
  total - (students_music + students_art + students_dance
           - students_music_art - students_art_dance - students_music_dance
           + students_all_three)

theorem students_taking_neither_proof :
  students_taking_neither 2500 200 150 100 75 50 40 25 = 2190 := by
  sorry

end students_taking_neither_proof_l2271_227199


namespace square_perimeter_ratio_l2271_227147

theorem square_perimeter_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ k : ℝ, b = k * a * Real.sqrt 2) → (4 * b) / (4 * a) = 5 → 
  b / (a * Real.sqrt 2) = 5 * Real.sqrt 2 / 2 := by
sorry

end square_perimeter_ratio_l2271_227147


namespace product_equals_eight_l2271_227163

theorem product_equals_eight : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end product_equals_eight_l2271_227163


namespace quiz_score_of_dropped_student_l2271_227112

theorem quiz_score_of_dropped_student 
  (initial_students : ℕ)
  (initial_average : ℚ)
  (curve_adjustment : ℕ)
  (remaining_students : ℕ)
  (final_average : ℚ)
  (h1 : initial_students = 16)
  (h2 : initial_average = 61.5)
  (h3 : curve_adjustment = 5)
  (h4 : remaining_students = 15)
  (h5 : final_average = 64) :
  ∃ (dropped_score : ℕ), 
    (initial_students : ℚ) * initial_average - dropped_score + 
    (remaining_students : ℚ) * curve_adjustment = 
    (remaining_students : ℚ) * final_average ∧ 
    dropped_score = 99 := by
  sorry

end quiz_score_of_dropped_student_l2271_227112


namespace six_digit_multiple_of_nine_l2271_227106

theorem six_digit_multiple_of_nine (n : ℕ) (h1 : n ≥ 734601 ∧ n ≤ 734691) 
  (h2 : n % 9 = 0) : 
  ∃ d : ℕ, (d = 6 ∨ d = 9) ∧ n = 734601 + d * 100 :=
sorry

end six_digit_multiple_of_nine_l2271_227106


namespace volleyball_team_selection_l2271_227145

theorem volleyball_team_selection (n : ℕ) (k : ℕ) (t : ℕ) :
  n = 15 →
  k = 6 →
  t = 3 →
  (Nat.choose n k) - (Nat.choose (n - t) k) = 4081 :=
by
  sorry

end volleyball_team_selection_l2271_227145


namespace abs_of_negative_2023_l2271_227183

theorem abs_of_negative_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end abs_of_negative_2023_l2271_227183


namespace arrangement_count_l2271_227149

theorem arrangement_count (n_boys n_girls : ℕ) (h_boys : n_boys = 5) (h_girls : n_girls = 6) :
  (Nat.factorial n_girls) * (Nat.choose (n_girls + 1) n_boys) * (Nat.factorial n_boys) =
  (Nat.factorial n_girls) * 2520 :=
sorry

end arrangement_count_l2271_227149


namespace quadratic_perfect_square_condition_l2271_227153

theorem quadratic_perfect_square_condition (a b : ℤ) :
  (∃ S : Set ℤ, (Set.Infinite S) ∧ (∀ x ∈ S, ∃ y : ℤ, x^2 + a*x + b = y^2)) ↔ a^2 = 4*b :=
sorry

end quadratic_perfect_square_condition_l2271_227153


namespace no_solution_implies_a_leq_3_l2271_227194

theorem no_solution_implies_a_leq_3 :
  (∀ x : ℝ, ¬(x > 3 ∧ x < a)) → a ≤ 3 := by
sorry

end no_solution_implies_a_leq_3_l2271_227194


namespace relay_race_time_l2271_227122

/-- Represents the time taken by each runner in the relay race -/
structure RelayTimes where
  mary : ℕ
  susan : ℕ
  jen : ℕ
  tiffany : ℕ

/-- Calculates the total time of the relay race -/
def total_time (times : RelayTimes) : ℕ :=
  times.mary + times.susan + times.jen + times.tiffany

/-- Theorem stating the total time of the relay race -/
theorem relay_race_time : ∃ (times : RelayTimes),
  times.mary = 2 * times.susan ∧
  times.susan = times.jen + 10 ∧
  times.jen = 30 ∧
  times.tiffany = times.mary - 7 ∧
  total_time times = 223 := by
  sorry

end relay_race_time_l2271_227122


namespace smallest_winning_m_l2271_227134

/-- Represents the state of a square on the board -/
inductive Color
| White
| Green

/-- Represents the game board -/
def Board := Array Color

/-- Represents a player in the game -/
inductive Player
| Ana
| Banana

/-- Ana's strategy function type -/
def AnaStrategy := Board → Fin 2024 → Bool

/-- Banana's strategy function type -/
def BananaStrategy := Board → Nat → Nat

/-- Simulates a single game with given strategies and m -/
def playGame (m : Nat) (anaStrat : AnaStrategy) (bananaStrat : BananaStrategy) : Bool :=
  sorry

/-- Checks if Ana has a winning strategy for a given m -/
def anaHasWinningStrategy (m : Nat) : Bool :=
  sorry

/-- The main theorem stating the smallest m for which Ana can guarantee winning -/
theorem smallest_winning_m :
  (∀ m : Nat, m < 88 → ¬ anaHasWinningStrategy m) ∧
  anaHasWinningStrategy 88 :=
sorry

end smallest_winning_m_l2271_227134


namespace arithmetic_sequence_sum_product_l2271_227168

theorem arithmetic_sequence_sum_product (a b c : ℝ) : 
  (b - a = c - b) →  -- arithmetic sequence condition
  (a + b + c = 12) →  -- sum condition
  (a * b * c = 48) →  -- product condition
  ((a = 2 ∧ b = 4 ∧ c = 6) ∨ (a = 6 ∧ b = 4 ∧ c = 2)) := by
  sorry

end arithmetic_sequence_sum_product_l2271_227168


namespace gas_pressure_change_l2271_227113

/-- Given inverse proportionality of pressure and volume at constant temperature,
    prove that the pressure in a 6-liter container is 4 kPa, given initial conditions. -/
theorem gas_pressure_change (p₁ p₂ v₁ v₂ : ℝ) : 
  p₁ > 0 → v₁ > 0 → p₂ > 0 → v₂ > 0 →  -- Ensuring positive values
  (p₁ * v₁ = p₂ * v₂) →  -- Inverse proportionality
  p₁ = 8 → v₁ = 3 → v₂ = 6 →  -- Initial conditions and new volume
  p₂ = 4 := by sorry

end gas_pressure_change_l2271_227113


namespace first_day_over_500_l2271_227102

def marbles (k : ℕ) : ℕ := 5 * 3^k

theorem first_day_over_500 : (∃ k : ℕ, marbles k > 500) ∧ 
  (∀ j : ℕ, j < 5 → marbles j ≤ 500) ∧ 
  marbles 5 > 500 := by
  sorry

end first_day_over_500_l2271_227102


namespace derivative_f_at_zero_l2271_227155

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.log (Real.cos x)) / x else 0

theorem derivative_f_at_zero :
  deriv f 0 = -1/2 := by
  sorry

end derivative_f_at_zero_l2271_227155


namespace grunters_lineup_count_l2271_227132

/-- Represents the number of players in each position --/
structure TeamComposition where
  guards : ℕ
  forwards : ℕ
  centers : ℕ

/-- Represents the starting lineup requirements --/
structure LineupRequirements where
  total_starters : ℕ
  guards : ℕ
  forwards : ℕ
  centers : ℕ

/-- Calculates the number of possible lineups --/
def calculate_lineups (team : TeamComposition) (req : LineupRequirements) : ℕ :=
  (team.guards.choose req.guards) * (team.forwards.choose req.forwards) * (team.centers.choose req.centers)

theorem grunters_lineup_count :
  let team : TeamComposition := ⟨5, 6, 3⟩  -- 4+1 guards, 5+1 forwards, 3 centers
  let req : LineupRequirements := ⟨5, 2, 2, 1⟩  -- 5 total, 2 guards, 2 forwards, 1 center
  calculate_lineups team req = 60 := by
  sorry

#check grunters_lineup_count

end grunters_lineup_count_l2271_227132


namespace fraction_of_25_problem_l2271_227119

theorem fraction_of_25_problem : ∃ x : ℚ, 
  x * 25 = 80 / 100 * 40 - 12 ∧ 
  x = 4 / 5 := by
sorry

end fraction_of_25_problem_l2271_227119


namespace solution_set_l2271_227195

def equation (x : ℝ) : Prop :=
  1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8

theorem solution_set : {x : ℝ | equation x} = {7, -2} := by sorry

end solution_set_l2271_227195


namespace dans_potatoes_l2271_227170

/-- The number of potatoes Dan has after rabbits eat some -/
def remaining_potatoes (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

theorem dans_potatoes : remaining_potatoes 7 4 = 3 := by
  sorry

end dans_potatoes_l2271_227170


namespace basketball_tryouts_l2271_227143

theorem basketball_tryouts (girls : ℕ) (boys : ℕ) (called_back : ℕ) (not_selected : ℕ) : 
  boys = 14 →
  called_back = 2 →
  not_selected = 21 →
  girls + boys = called_back + not_selected →
  girls = 9 := by
sorry

end basketball_tryouts_l2271_227143


namespace solve_ice_cream_problem_l2271_227190

def ice_cream_problem (aaron_savings : ℚ) (carson_savings : ℚ) (dinner_bill_ratio : ℚ) 
  (ice_cream_cost_per_scoop : ℚ) (change_per_person : ℚ) : Prop :=
  let total_savings := aaron_savings + carson_savings
  let dinner_cost := dinner_bill_ratio * total_savings
  let remaining_money := total_savings - dinner_cost
  let ice_cream_total_cost := remaining_money - 2 * change_per_person
  let total_scoops := ice_cream_total_cost / ice_cream_cost_per_scoop
  (total_scoops / 2 : ℚ) = 6

theorem solve_ice_cream_problem :
  ice_cream_problem 40 40 (3/4) (3/2) 1 :=
by
  sorry

#check solve_ice_cream_problem

end solve_ice_cream_problem_l2271_227190


namespace polynomial_division_quotient_l2271_227144

theorem polynomial_division_quotient : 
  let dividend := fun x : ℚ => 10 * x^3 - 5 * x^2 + 8 * x - 9
  let divisor := fun x : ℚ => 3 * x - 4
  let quotient := fun x : ℚ => (10/3) * x^2 - (55/9) * x - 172/27
  ∀ x : ℚ, dividend x = divisor x * quotient x + (-971/27) := by
  sorry

end polynomial_division_quotient_l2271_227144


namespace solution_characterization_l2271_227139

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The plane z = x -/
def midplane (p : Point3D) : Prop := p.z = p.x

/-- The sphere with center A and radius r -/
def sphere (A : Point3D) (r : ℝ) (p : Point3D) : Prop :=
  (p.x - A.x)^2 + (p.y - A.y)^2 + (p.z - A.z)^2 = r^2

/-- The set of points satisfying both conditions -/
def solution_set (A : Point3D) (r : ℝ) : Set Point3D :=
  {p : Point3D | sphere A r p ∧ midplane p}

theorem solution_characterization (A : Point3D) (r : ℝ) :
  ∀ p : Point3D, p ∈ solution_set A r ↔ 
    (p.x - A.x)^2 + (p.y - A.y)^2 + (p.x - A.z)^2 = r^2 :=
  sorry

end solution_characterization_l2271_227139


namespace amusement_park_trip_distance_l2271_227193

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- The total distance covered by Amanda and her friends -/
theorem amusement_park_trip_distance : 
  let d1 := distance 40 1.5
  let d2 := distance 50 1
  let d3 := distance 30 2.25
  d1 + d2 + d3 = 177.5 := by sorry

end amusement_park_trip_distance_l2271_227193


namespace min_age_difference_proof_l2271_227110

/-- The number of days in a leap year -/
def daysInLeapYear : ℕ := 366

/-- The number of days in a common year -/
def daysInCommonYear : ℕ := 365

/-- The year Adil was born -/
def adilBirthYear : ℕ := 2015

/-- The year Bav was born -/
def bavBirthYear : ℕ := 2018

/-- The minimum age difference in days between Adil and Bav -/
def minAgeDifference : ℕ := daysInLeapYear + daysInCommonYear + 1

theorem min_age_difference_proof :
  minAgeDifference = 732 :=
sorry

end min_age_difference_proof_l2271_227110


namespace wallet_cost_proof_l2271_227157

/-- The cost of a pair of sneakers -/
def sneaker_cost : ℕ := 100

/-- The cost of a backpack -/
def backpack_cost : ℕ := 100

/-- The cost of a pair of jeans -/
def jeans_cost : ℕ := 50

/-- The total amount spent by Leonard and Michael -/
def total_spent : ℕ := 450

/-- The cost of the wallet -/
def wallet_cost : ℕ := 50

theorem wallet_cost_proof :
  wallet_cost + 2 * sneaker_cost + backpack_cost + 2 * jeans_cost = total_spent :=
by sorry

end wallet_cost_proof_l2271_227157


namespace pedal_to_original_triangle_l2271_227197

/-- Given the sides of a pedal triangle, calculate the sides of the original triangle --/
theorem pedal_to_original_triangle 
  (a₁ b₁ c₁ : ℝ) 
  (h_pos : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁) :
  ∃ (a b c : ℝ),
    let s₁ := (a₁ + b₁ + c₁) / 2
    a = a₁ * Real.sqrt (b₁ * c₁ / ((s₁ - b₁) * (s₁ - c₁))) ∧
    b = b₁ * Real.sqrt (a₁ * c₁ / ((s₁ - a₁) * (s₁ - c₁))) ∧
    c = c₁ * Real.sqrt (a₁ * b₁ / ((s₁ - a₁) * (s₁ - b₁))) :=
by
  sorry


end pedal_to_original_triangle_l2271_227197

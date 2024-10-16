import Mathlib

namespace NUMINAMATH_CALUDE_fifteen_factorial_base_nine_zeros_l2771_277145

/-- The number of trailing zeros in n! when written in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ :=
  sorry

theorem fifteen_factorial_base_nine_zeros :
  trailingZeros 15 9 = 3 :=
sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_nine_zeros_l2771_277145


namespace NUMINAMATH_CALUDE_thabo_owns_160_books_l2771_277130

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcoverNonfiction : ℕ
  paperbackNonfiction : ℕ
  paperbackFiction : ℕ

/-- Thabo's book collection satisfying the given conditions -/
def thabosBooks : BookCollection where
  hardcoverNonfiction := 25
  paperbackNonfiction := 25 + 20
  paperbackFiction := 2 * (25 + 20)

/-- The total number of books in a collection -/
def totalBooks (books : BookCollection) : ℕ :=
  books.hardcoverNonfiction + books.paperbackNonfiction + books.paperbackFiction

/-- Theorem stating that Thabo owns 160 books in total -/
theorem thabo_owns_160_books : totalBooks thabosBooks = 160 := by
  sorry


end NUMINAMATH_CALUDE_thabo_owns_160_books_l2771_277130


namespace NUMINAMATH_CALUDE_theresa_required_hours_l2771_277166

/-- The average number of hours Theresa needs to work per week over 4 weeks -/
def required_average : ℝ := 12

/-- The total number of weeks -/
def total_weeks : ℕ := 4

/-- The minimum total hours Theresa needs to work -/
def minimum_total_hours : ℝ := 50

/-- The hours Theresa worked in the first week -/
def first_week_hours : ℝ := 15

/-- The hours Theresa worked in the second week -/
def second_week_hours : ℝ := 8

/-- The number of remaining weeks -/
def remaining_weeks : ℕ := 2

theorem theresa_required_hours :
  let total_worked := first_week_hours + second_week_hours
  let remaining_hours := minimum_total_hours - total_worked
  (remaining_hours / remaining_weeks : ℝ) = 13.5 ∧
  remaining_hours ≥ required_average * remaining_weeks := by
  sorry

end NUMINAMATH_CALUDE_theresa_required_hours_l2771_277166


namespace NUMINAMATH_CALUDE_negation_of_exists_lt_is_forall_ge_l2771_277175

theorem negation_of_exists_lt_is_forall_ge (p : Prop) : 
  (¬ (∃ x : ℝ, x^2 + 2*x < 0)) ↔ (∀ x : ℝ, x^2 + 2*x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_lt_is_forall_ge_l2771_277175


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l2771_277155

theorem right_triangle_leg_length 
  (a b c : ℝ) 
  (right_angle : a^2 + b^2 = c^2) 
  (hypotenuse : c = 25) 
  (known_leg : a = 24) : 
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l2771_277155


namespace NUMINAMATH_CALUDE_min_value_of_f_l2771_277161

/-- The function f(x) = 5x^2 + 10x + 20 -/
def f (x : ℝ) : ℝ := 5 * x^2 + 10 * x + 20

/-- The minimum value of f(x) is 15 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = 15 ∧ ∀ x, f x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2771_277161


namespace NUMINAMATH_CALUDE_derivative_cos_cubed_l2771_277100

theorem derivative_cos_cubed (x : ℝ) :
  let y : ℝ → ℝ := λ x => (Real.cos (2 * x + 3)) ^ 3
  deriv y x = -6 * (Real.cos (2 * x + 3))^2 * Real.sin (2 * x + 3) :=
by sorry

end NUMINAMATH_CALUDE_derivative_cos_cubed_l2771_277100


namespace NUMINAMATH_CALUDE_tetrahedron_vector_sum_same_sign_l2771_277111

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- A point is inside a tetrahedron if it can be expressed as a convex combination of the vertices -/
def IsInsideTetrahedron (O A B C D : V) : Prop :=
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d = 1 ∧
    O = a • A + b • B + c • C + d • D

/-- All real numbers have the same sign if they are all positive or all negative -/
def AllSameSign (α β γ δ : ℝ) : Prop :=
  (α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0) ∨ (α < 0 ∧ β < 0 ∧ γ < 0 ∧ δ < 0)

theorem tetrahedron_vector_sum_same_sign
  (O A B C D : V) (α β γ δ : ℝ)
  (h_inside : IsInsideTetrahedron O A B C D)
  (h_sum : α • (A - O) + β • (B - O) + γ • (C - O) + δ • (D - O) = 0) :
  AllSameSign α β γ δ :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_vector_sum_same_sign_l2771_277111


namespace NUMINAMATH_CALUDE_study_group_probability_l2771_277116

/-- Represents the gender distribution in the study group -/
def gender_distribution : Fin 2 → ℝ
  | 0 => 0.55  -- women
  | 1 => 0.45  -- men

/-- Represents the age distribution for each gender -/
def age_distribution : Fin 2 → Fin 3 → ℝ
  | 0, 0 => 0.20  -- women below 35
  | 0, 1 => 0.35  -- women 35-50
  | 0, 2 => 0.45  -- women above 50
  | 1, 0 => 0.30  -- men below 35
  | 1, 1 => 0.40  -- men 35-50
  | 1, 2 => 0.30  -- men above 50

/-- Represents the profession distribution for each gender and age group -/
def profession_distribution : Fin 2 → Fin 3 → Fin 3 → ℝ
  | 0, 0, 0 => 0.35  -- women below 35, lawyers
  | 0, 0, 1 => 0.45  -- women below 35, doctors
  | 0, 0, 2 => 0.20  -- women below 35, engineers
  | 0, 1, 0 => 0.25  -- women 35-50, lawyers
  | 0, 1, 1 => 0.50  -- women 35-50, doctors
  | 0, 1, 2 => 0.25  -- women 35-50, engineers
  | 0, 2, 0 => 0.20  -- women above 50, lawyers
  | 0, 2, 1 => 0.30  -- women above 50, doctors
  | 0, 2, 2 => 0.50  -- women above 50, engineers
  | 1, 0, 0 => 0.40  -- men below 35, lawyers
  | 1, 0, 1 => 0.30  -- men below 35, doctors
  | 1, 0, 2 => 0.30  -- men below 35, engineers
  | 1, 1, 0 => 0.45  -- men 35-50, lawyers
  | 1, 1, 1 => 0.25  -- men 35-50, doctors
  | 1, 1, 2 => 0.30  -- men 35-50, engineers
  | 1, 2, 0 => 0.30  -- men above 50, lawyers
  | 1, 2, 1 => 0.40  -- men above 50, doctors
  | 1, 2, 2 => 0.30  -- men above 50, engineers

theorem study_group_probability : 
  gender_distribution 0 * age_distribution 0 0 * profession_distribution 0 0 0 +
  gender_distribution 1 * age_distribution 1 2 * profession_distribution 1 2 2 +
  gender_distribution 0 * age_distribution 0 1 * profession_distribution 0 1 1 +
  gender_distribution 1 * age_distribution 1 1 * profession_distribution 1 1 1 = 0.22025 := by
  sorry

end NUMINAMATH_CALUDE_study_group_probability_l2771_277116


namespace NUMINAMATH_CALUDE_total_players_l2771_277150

theorem total_players (kabaddi : ℕ) (kho_kho_only : ℕ) (both : ℕ) 
  (h1 : kabaddi = 10) 
  (h2 : kho_kho_only = 25) 
  (h3 : both = 5) : 
  kabaddi + kho_kho_only - both = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_players_l2771_277150


namespace NUMINAMATH_CALUDE_polynomial_coefficient_G_l2771_277108

-- Define the polynomial p(z)
def p (z E F G H I : ℤ) : ℤ := z^7 - 13*z^6 + E*z^5 + F*z^4 + G*z^3 + H*z^2 + I*z + 36

-- Define the property that all roots are positive integers
def all_roots_positive_integers (p : ℤ → ℤ) : Prop :=
  ∀ z : ℤ, p z = 0 → z > 0

-- Theorem statement
theorem polynomial_coefficient_G (E F G H I : ℤ) :
  all_roots_positive_integers (p · E F G H I) →
  G = -82 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_coefficient_G_l2771_277108


namespace NUMINAMATH_CALUDE_jerry_age_l2771_277129

/-- Given that Mickey's age is 18 and Mickey's age is 2 years less than 400% of Jerry's age,
    prove that Jerry's age is 5. -/
theorem jerry_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 18)
  (h2 : mickey_age = 4 * jerry_age - 2) : 
  jerry_age = 5 := by
sorry

end NUMINAMATH_CALUDE_jerry_age_l2771_277129


namespace NUMINAMATH_CALUDE_perimeter_ABCDEFG_l2771_277165

-- Define the points
variable (A B C D E F G : ℝ × ℝ)

-- Define the conditions
def is_equilateral (X Y Z : ℝ × ℝ) : Prop := 
  dist X Y = dist Y Z ∧ dist Y Z = dist Z X

def is_midpoint (M X Y : ℝ × ℝ) : Prop :=
  M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- State the theorem
theorem perimeter_ABCDEFG (h1 : is_equilateral A B C)
                          (h2 : is_equilateral A D E)
                          (h3 : is_equilateral E F G)
                          (h4 : is_midpoint D A C)
                          (h5 : is_midpoint G A E)
                          (h6 : is_midpoint F E G)
                          (h7 : dist A B = 6) :
  dist A B + dist B C + dist C D + dist D E + dist E F + dist F G + dist G A = 25.5 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ABCDEFG_l2771_277165


namespace NUMINAMATH_CALUDE_product_104_96_l2771_277103

theorem product_104_96 : 104 * 96 = 9984 := by
  sorry

end NUMINAMATH_CALUDE_product_104_96_l2771_277103


namespace NUMINAMATH_CALUDE_divisibility_of_sum_l2771_277162

theorem divisibility_of_sum (a b c d x : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (x^2 - (a+b)*x + a*b) * (x^2 - (c+d)*x + c*d) = 9 →
  ∃ k : ℤ, a + b + c + d = 4 * k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_l2771_277162


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_real_l2771_277124

theorem sqrt_x_plus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_real_l2771_277124


namespace NUMINAMATH_CALUDE_divisors_4k_plus_1_ge_4k_minus_1_l2771_277110

/-- The number of divisors of n of the form 4k+1 -/
def divisors_4k_plus_1 (n : ℕ+) : ℕ := sorry

/-- The number of divisors of n of the form 4k-1 -/
def divisors_4k_minus_1 (n : ℕ+) : ℕ := sorry

/-- The difference between the number of divisors of the form 4k+1 and 4k-1 -/
def D (n : ℕ+) : ℤ := (divisors_4k_plus_1 n : ℤ) - (divisors_4k_minus_1 n : ℤ)

theorem divisors_4k_plus_1_ge_4k_minus_1 (n : ℕ+) : D n ≥ 0 := by sorry

end NUMINAMATH_CALUDE_divisors_4k_plus_1_ge_4k_minus_1_l2771_277110


namespace NUMINAMATH_CALUDE_sufficient_condition_for_equation_l2771_277172

theorem sufficient_condition_for_equation (a : ℝ) (f g h : ℝ → ℝ) 
  (ha : a > 1)
  (h_sum_nonneg : ∀ x, f x + g x + h x ≥ 0)
  (h_common_root : ∃ x₀, f x₀ = 0 ∧ g x₀ = 0 ∧ h x₀ = 0) :
  ∃ x, a^(f x) + a^(g x) + a^(h x) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_equation_l2771_277172


namespace NUMINAMATH_CALUDE_correct_elderly_sample_l2771_277119

/-- Represents the composition of employees in a company and its sample --/
structure EmployeeSample where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ
  sampledYoung : ℕ
  sampledElderly : ℕ

/-- Checks if the employee sample is valid according to the given conditions --/
def isValidSample (s : EmployeeSample) : Prop :=
  s.total = 430 ∧
  s.young = 160 ∧
  s.middleAged = 2 * s.elderly ∧
  s.total = s.young + s.middleAged + s.elderly ∧
  s.sampledYoung = 32

/-- Theorem stating that for a valid sample, the number of sampled elderly should be 18 --/
theorem correct_elderly_sample (s : EmployeeSample) 
  (h : isValidSample s) : s.sampledElderly = 18 := by
  sorry


end NUMINAMATH_CALUDE_correct_elderly_sample_l2771_277119


namespace NUMINAMATH_CALUDE_rectangle_area_function_l2771_277163

/-- For a rectangle with area 10 and adjacent sides x and y, prove that y = 10/x --/
theorem rectangle_area_function (x y : ℝ) (h : x * y = 10) : y = 10 / x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_function_l2771_277163


namespace NUMINAMATH_CALUDE_f_properties_l2771_277171

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 3|

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f x < 7 ↔ -2 < x ∧ x < 5) ∧
  (∀ x : ℝ, f x - |2*x - 7| < x^2 - 2*x + Real.sqrt 26) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l2771_277171


namespace NUMINAMATH_CALUDE_tim_attend_probability_l2771_277104

-- Define the probability of rain
def prob_rain : ℝ := 0.6

-- Define the probability of sun (complementary to rain)
def prob_sun : ℝ := 1 - prob_rain

-- Define the probability Tim attends if it rains
def prob_attend_rain : ℝ := 0.25

-- Define the probability Tim attends if it's sunny
def prob_attend_sun : ℝ := 0.7

-- Theorem statement
theorem tim_attend_probability :
  prob_rain * prob_attend_rain + prob_sun * prob_attend_sun = 0.43 := by
sorry

end NUMINAMATH_CALUDE_tim_attend_probability_l2771_277104


namespace NUMINAMATH_CALUDE_root_value_theorem_l2771_277121

theorem root_value_theorem (a : ℝ) : 2 * a^2 = a + 4 → 4 * a^2 - 2 * a = 8 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l2771_277121


namespace NUMINAMATH_CALUDE_tan_zero_degrees_l2771_277168

theorem tan_zero_degrees : Real.tan 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_zero_degrees_l2771_277168


namespace NUMINAMATH_CALUDE_factorial_sum_not_end_1990_l2771_277140

theorem factorial_sum_not_end_1990 (m n : ℕ) : (m.factorial + n.factorial) % 10000 ≠ 1990 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_not_end_1990_l2771_277140


namespace NUMINAMATH_CALUDE_dvd_packs_calculation_l2771_277191

theorem dvd_packs_calculation (total_money : ℕ) (pack_cost : ℕ) (h1 : total_money = 104) (h2 : pack_cost = 26) :
  total_money / pack_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_dvd_packs_calculation_l2771_277191


namespace NUMINAMATH_CALUDE_fib_150_mod_9_l2771_277138

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Fibonacci sequence modulo 9 -/
def fibMod9 (n : ℕ) : Fin 9 :=
  (fib n).mod 9

/-- The period of Fibonacci sequence modulo 9 -/
def fibMod9Period : ℕ := 24

theorem fib_150_mod_9 :
  fibMod9 150 = 8 := by sorry

end NUMINAMATH_CALUDE_fib_150_mod_9_l2771_277138


namespace NUMINAMATH_CALUDE_max_sin_a_given_condition_l2771_277164

open Real

theorem max_sin_a_given_condition (a b : ℝ) :
  cos (a + b) + sin (a - b) = cos a + cos b →
  ∃ (max_sin_a : ℝ), (∀ x, sin x ≤ max_sin_a) ∧ (max_sin_a = 1) :=
sorry

end NUMINAMATH_CALUDE_max_sin_a_given_condition_l2771_277164


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l2771_277156

theorem simplify_algebraic_expression (a : ℝ) : 
  3*a + 6*a + 9*a + 6 + 12*a + 15 + 18*a = 48*a + 21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l2771_277156


namespace NUMINAMATH_CALUDE_select_volunteers_l2771_277131

theorem select_volunteers (boys girls volunteers : ℕ) 
  (h1 : boys = 6)
  (h2 : girls = 2)
  (h3 : volunteers = 3) :
  (Nat.choose (boys + girls) volunteers) - (Nat.choose boys volunteers) = 36 := by
  sorry

end NUMINAMATH_CALUDE_select_volunteers_l2771_277131


namespace NUMINAMATH_CALUDE_shooting_probabilities_l2771_277195

/-- Let A and B be two individuals conducting 3 shooting trials each.
    The probability of A hitting the target in each trial is 1/2.
    The probability of B hitting the target in each trial is 2/3. -/
theorem shooting_probabilities 
  (probability_A : ℝ) 
  (probability_B : ℝ) 
  (h_prob_A : probability_A = 1/2) 
  (h_prob_B : probability_B = 2/3) :
  /- The probability that A hits the target exactly 2 times -/
  (3 : ℝ) * probability_A^2 * (1 - probability_A) = 3/8 ∧ 
  /- The probability that B hits the target at least 2 times -/
  (3 : ℝ) * probability_B^2 * (1 - probability_B) + probability_B^3 = 20/27 ∧ 
  /- The probability that B hits the target exactly 2 more times than A -/
  (3 : ℝ) * probability_B^2 * (1 - probability_B) * (1 - probability_A)^3 + 
  probability_B^3 * (3 : ℝ) * probability_A * (1 - probability_A)^2 = 1/6 :=
by sorry


end NUMINAMATH_CALUDE_shooting_probabilities_l2771_277195


namespace NUMINAMATH_CALUDE_gcd_of_136_and_1275_l2771_277177

theorem gcd_of_136_and_1275 : Nat.gcd 136 1275 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_136_and_1275_l2771_277177


namespace NUMINAMATH_CALUDE_locus_is_circle_l2771_277183

/-- An isosceles triangle with side length s and base b -/
structure IsoscelesTriangle where
  s : ℝ
  b : ℝ
  s_pos : 0 < s
  b_pos : 0 < b
  triangle_ineq : b < 2 * s

/-- The locus of points P such that the sum of distances from P to the vertices equals a -/
def Locus (t : IsoscelesTriangle) (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧
    Real.sqrt (x^2 + y^2) +
    Real.sqrt ((x - t.b)^2 + y^2) +
    Real.sqrt ((x - t.b/2)^2 + (y - Real.sqrt (t.s^2 - (t.b/2)^2))^2) = a}

/-- The theorem stating that the locus is a circle if and only if a > 2s + b -/
theorem locus_is_circle (t : IsoscelesTriangle) (a : ℝ) :
  (∃ (c : ℝ × ℝ) (r : ℝ), Locus t a = {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}) ↔
  a > 2 * t.s + t.b := by
  sorry

end NUMINAMATH_CALUDE_locus_is_circle_l2771_277183


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2771_277122

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∀ x, x^2 + 16 = 12*x → x = 6 + 2*Real.sqrt 5 ∨ x = 6 - 2*Real.sqrt 5) →
  (let x₁ := 6 + 2*Real.sqrt 5
   let x₂ := 6 - 2*Real.sqrt 5
   x₁ + x₂ = 12) := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2771_277122


namespace NUMINAMATH_CALUDE_green_ball_probability_l2771_277142

/-- Represents a container of balls -/
structure Container where
  red : Nat
  green : Nat

/-- The probability of selecting a specific container -/
def containerProb : ℚ := 1 / 3

/-- The probability of selecting a green ball from a given container -/
def greenBallProb (c : Container) : ℚ := c.green / (c.red + c.green)

/-- The containers A, B, and C -/
def containerA : Container := ⟨5, 5⟩
def containerB : Container := ⟨3, 3⟩
def containerC : Container := ⟨3, 3⟩

/-- The theorem stating the probability of selecting a green ball -/
theorem green_ball_probability :
  containerProb * greenBallProb containerA +
  containerProb * greenBallProb containerB +
  containerProb * greenBallProb containerC = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l2771_277142


namespace NUMINAMATH_CALUDE_six_partitions_into_three_or_fewer_l2771_277113

/-- The number of ways to partition n indistinguishable objects into k or fewer non-empty parts -/
def partitions_into_k_or_fewer (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to partition 6 indistinguishable objects into 3 or fewer non-empty parts -/
theorem six_partitions_into_three_or_fewer : partitions_into_k_or_fewer 6 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_partitions_into_three_or_fewer_l2771_277113


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_max_product_l2771_277115

/-- A cyclic quadrilateral with sides a, b, c, d inscribed in a circle of radius R -/
structure CyclicQuadrilateral where
  R : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  inscribed : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ R > 0

/-- The product of sums of opposite sides pairs -/
def sideProduct (q : CyclicQuadrilateral) : ℝ :=
  (q.a * q.b + q.c * q.d) * (q.a * q.c + q.b * q.d) * (q.a * q.d + q.b * q.c)

/-- Predicate to check if a cyclic quadrilateral is a square -/
def isSquare (q : CyclicQuadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

theorem cyclic_quadrilateral_max_product (q : CyclicQuadrilateral) :
  ∀ q' : CyclicQuadrilateral, q'.R = q.R → sideProduct q ≤ sideProduct q' ↔ isSquare q' :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_max_product_l2771_277115


namespace NUMINAMATH_CALUDE_unique_solution_for_digit_equation_l2771_277198

theorem unique_solution_for_digit_equation :
  ∃! (A B D E : ℕ),
    (A < 10 ∧ B < 10 ∧ D < 10 ∧ E < 10) ∧  -- Base 10 digits
    (A ≠ B ∧ A ≠ D ∧ A ≠ E ∧ B ≠ D ∧ B ≠ E ∧ D ≠ E) ∧  -- Different digits
    (A^(10*A + A) + 10*A + A = 
      B * 10^15 + B * 10^14 + 9 * 10^13 +
      D * 10^12 + E * 10^11 + D * 10^10 +
      B * 10^9 + E * 10^8 + E * 10^7 +
      B * 10^6 + B * 10^5 + B * 10^4 +
      B * 10^3 + B * 10^2 + E * 10^1 + E * 10^0) ∧
    (A = 3 ∧ B = 5 ∧ D = 0 ∧ E = 6) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_digit_equation_l2771_277198


namespace NUMINAMATH_CALUDE_equation_has_solution_equation_has_unique_solution_l2771_277134

-- Define the equation
def equation (a x : ℝ) : Prop :=
  (Real.log x / Real.log a) / (Real.log 2 / Real.log a) +
  (Real.log (2*a - x) / Real.log x) / (Real.log 2 / Real.log x) =
  1 / (Real.log 2 / Real.log (a^2 - 1))

-- Theorem for the first question
theorem equation_has_solution (a : ℝ) :
  (∃ x, equation a x) ↔ (a > 1 ∧ a ≠ Real.sqrt 2) :=
sorry

-- Theorem for the second question
theorem equation_has_unique_solution (a : ℝ) :
  (∃! x, equation a x) ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_equation_has_solution_equation_has_unique_solution_l2771_277134


namespace NUMINAMATH_CALUDE_bead_arrangement_probability_l2771_277144

def num_red : ℕ := 4
def num_white : ℕ := 2
def num_blue : ℕ := 2
def total_beads : ℕ := num_red + num_white + num_blue

def total_arrangements : ℕ := Nat.factorial total_beads / (Nat.factorial num_red * Nat.factorial num_white * Nat.factorial num_blue)

def valid_arrangements : ℕ := 27  -- This is an approximation based on the problem's solution

theorem bead_arrangement_probability :
  (valid_arrangements : ℚ) / total_arrangements = 9 / 140 :=
sorry

end NUMINAMATH_CALUDE_bead_arrangement_probability_l2771_277144


namespace NUMINAMATH_CALUDE_car_gasoline_theorem_l2771_277146

/-- Represents the relationship between remaining gasoline and distance traveled for a car --/
def gasoline_function (x : ℝ) : ℝ := 50 - 0.1 * x

/-- Represents the valid range for the distance traveled --/
def valid_distance (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 500

theorem car_gasoline_theorem :
  ∀ x : ℝ,
  valid_distance x →
  (∀ y : ℝ, y = gasoline_function x → y = 50 - 0.1 * x) ∧
  (x = 200 → gasoline_function x = 30) :=
by sorry

end NUMINAMATH_CALUDE_car_gasoline_theorem_l2771_277146


namespace NUMINAMATH_CALUDE_value_calculation_l2771_277192

theorem value_calculation (number : ℝ) (value : ℝ) : 
  number = 8 → 
  value = 0.75 * number + 2 → 
  value = 8 := by
sorry

end NUMINAMATH_CALUDE_value_calculation_l2771_277192


namespace NUMINAMATH_CALUDE_angle_bisector_ratio_not_unique_l2771_277159

/-- Represents a triangle --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ

/-- Represents the ratio of an angle bisector to its corresponding side --/
def angle_bisector_ratio (t : Triangle) : ℝ := 
  sorry -- Definition of angle bisector ratio

/-- Two triangles are similar if their corresponding angles are equal --/
def similar (t1 t2 : Triangle) : Prop :=
  t1.angle_A = t2.angle_A ∧ t1.angle_B = t2.angle_B ∧ t1.angle_C = t2.angle_C

theorem angle_bisector_ratio_not_unique :
  ∃ (t1 t2 : Triangle) (r : ℝ), 
    angle_bisector_ratio t1 = r ∧ 
    angle_bisector_ratio t2 = r ∧ 
    ¬(similar t1 t2) :=
  sorry


end NUMINAMATH_CALUDE_angle_bisector_ratio_not_unique_l2771_277159


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l2771_277199

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (10*x + 2) * (10*x + 6)^2 * (5*x + 1) = 24 * k) ∧
  (∀ (m : ℤ), m > 24 → ∃ (y : ℤ), Odd y ∧ ¬(∃ (l : ℤ), (10*y + 2) * (10*y + 6)^2 * (5*y + 1) = m * l)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l2771_277199


namespace NUMINAMATH_CALUDE_quadratic_equation_equal_roots_l2771_277181

theorem quadratic_equation_equal_roots (a : ℝ) :
  (∃ x : ℝ, (3 * a - 1) * x^2 - a * x + 1/4 = 0 ∧
   ∀ y : ℝ, (3 * a - 1) * y^2 - a * y + 1/4 = 0 → y = x) →
  a^2 - 2 * a + 2021 + 1/a = 2023 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_equal_roots_l2771_277181


namespace NUMINAMATH_CALUDE_friends_journey_time_l2771_277169

/-- Represents the journey of three friends with a bicycle --/
theorem friends_journey_time :
  -- Define the walking speed of the friends
  ∀ (walking_speed : ℝ),
  -- Define the bicycle speed
  ∀ (bicycle_speed : ℝ),
  -- Conditions
  (walking_speed > 0) →
  (bicycle_speed > 0) →
  -- Second friend walks 6 km in the first hour
  (walking_speed * 1 = 6) →
  -- Third friend rides 12 km in 2/3 hour
  (bicycle_speed * (2/3) = 12) →
  -- Total journey time
  ∃ (total_time : ℝ),
  total_time = 2 + 2/3 :=
by sorry

end NUMINAMATH_CALUDE_friends_journey_time_l2771_277169


namespace NUMINAMATH_CALUDE_six_thirty_six_am_metric_l2771_277118

/-- Represents a time in the metric system -/
structure MetricTime where
  hours : Nat
  minutes : Nat

/-- Converts normal time (in minutes since midnight) to metric time -/
def normalToMetric (normalMinutes : Nat) : MetricTime :=
  let totalMetricMinutes := normalMinutes * 25 / 36
  { hours := totalMetricMinutes / 100
  , minutes := totalMetricMinutes % 100 }

theorem six_thirty_six_am_metric :
  normalToMetric (6 * 60 + 36) = { hours := 2, minutes := 75 } := by
  sorry

#eval 100 * (normalToMetric (6 * 60 + 36)).hours +
      10 * ((normalToMetric (6 * 60 + 36)).minutes / 10) +
      (normalToMetric (6 * 60 + 36)).minutes % 10

end NUMINAMATH_CALUDE_six_thirty_six_am_metric_l2771_277118


namespace NUMINAMATH_CALUDE_circular_garden_radius_l2771_277141

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1 / 5) * π * r^2 → r = 10 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l2771_277141


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l2771_277139

theorem largest_stamps_per_page : Nat.gcd 840 1008 = 168 := by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l2771_277139


namespace NUMINAMATH_CALUDE_num_possible_heights_l2771_277135

/-- The dimensions of each block -/
def block_dimensions : Finset ℕ := {2, 3, 6}

/-- The number of blocks in the tower -/
def num_blocks : ℕ := 4

/-- A function to calculate all possible heights of the tower -/
def possible_heights : Finset ℕ := sorry

/-- The theorem stating that the number of possible heights is 14 -/
theorem num_possible_heights : Finset.card possible_heights = 14 := by sorry

end NUMINAMATH_CALUDE_num_possible_heights_l2771_277135


namespace NUMINAMATH_CALUDE_salt_mixture_concentration_l2771_277189

/-- Given two salt solutions and their volumes, calculate the salt concentration of the mixture -/
theorem salt_mixture_concentration 
  (vol1 : ℝ) (conc1 : ℝ) (vol2 : ℝ) (conc2 : ℝ) 
  (h1 : vol1 = 600) 
  (h2 : conc1 = 0.03) 
  (h3 : vol2 = 400) 
  (h4 : conc2 = 0.12) 
  (h5 : vol1 + vol2 = 1000) :
  (vol1 * conc1 + vol2 * conc2) / (vol1 + vol2) = 0.066 := by
sorry

end NUMINAMATH_CALUDE_salt_mixture_concentration_l2771_277189


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l2771_277123

theorem min_value_of_fraction (x : ℝ) (h : x > 9) :
  x^2 / (x - 9) ≥ 36 ∧ ∃ y > 9, y^2 / (y - 9) = 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l2771_277123


namespace NUMINAMATH_CALUDE_group_size_l2771_277184

theorem group_size (B F BF : ℕ) (h1 : B = 13) (h2 : F = 15) (h3 : BF = 18) : 
  B + F - BF + 3 = 13 := by
sorry

end NUMINAMATH_CALUDE_group_size_l2771_277184


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l2771_277160

theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 + 30*x + a = (3*x + b)^2) → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l2771_277160


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2771_277109

theorem complex_fraction_simplification :
  (2 - Complex.I) / (1 + 2 * Complex.I) = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2771_277109


namespace NUMINAMATH_CALUDE_circle_equation_l2771_277132

theorem circle_equation (x y : ℝ) :
  (x^2 + 8*x + y^2 + 4*y - 36 = 0) ↔
  ((x + 4)^2 + (y + 2)^2 = 4^2) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l2771_277132


namespace NUMINAMATH_CALUDE_equation_solution_l2771_277157

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  x^2 + 36 / x^2 = 13 ↔ x = 2 ∨ x = -2 ∨ x = 3 ∨ x = -3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2771_277157


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l2771_277154

/-- Proves that mixing two varieties of rice in a given ratio results in the specified cost per kg -/
theorem rice_mixture_cost 
  (cost1 : ℝ) 
  (cost2 : ℝ) 
  (ratio : ℝ) 
  (mixture_cost : ℝ) 
  (h1 : cost1 = 7) 
  (h2 : cost2 = 8.75) 
  (h3 : ratio = 2.5) 
  (h4 : mixture_cost = 7.5) : 
  (ratio * cost1 + cost2) / (ratio + 1) = mixture_cost := by
  sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_l2771_277154


namespace NUMINAMATH_CALUDE_number_problem_l2771_277125

theorem number_problem (N : ℝ) : (0.6 * N = 0.5 * 720) → N = 600 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2771_277125


namespace NUMINAMATH_CALUDE_abc_inequality_l2771_277106

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b * c * (a + b + c) = a * b + b * c + c * a) :
  5 * (a + b + c) ≥ 7 + 8 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2771_277106


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l2771_277137

theorem sum_of_squares_and_products (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 52)
  (h5 : x*y + y*z + z*x = 24) : 
  x + y + z = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l2771_277137


namespace NUMINAMATH_CALUDE_cookie_boxes_problem_l2771_277158

theorem cookie_boxes_problem (n : ℕ) : 
  (n ≥ 1) →
  (n - 7 ≥ 1) →
  (n - 2 ≥ 1) →
  ((n - 7) + (n - 2) < n) →
  (n = 8) := by
  sorry

end NUMINAMATH_CALUDE_cookie_boxes_problem_l2771_277158


namespace NUMINAMATH_CALUDE_reflect_x_minus3_minus5_l2771_277120

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The theorem stating that reflecting P(-3,-5) across the x-axis results in (-3,5) -/
theorem reflect_x_minus3_minus5 :
  let P : Point := { x := -3, y := -5 }
  reflect_x P = { x := -3, y := 5 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_x_minus3_minus5_l2771_277120


namespace NUMINAMATH_CALUDE_fraction_inequality_l2771_277112

theorem fraction_inequality (c x y : ℝ) (h1 : c > x) (h2 : x > y) (h3 : y > 0) :
  x / (c - x) > y / (c - y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2771_277112


namespace NUMINAMATH_CALUDE_remainder_sum_l2771_277136

theorem remainder_sum (c d : ℤ) 
  (hc : c % 60 = 47)
  (hd : d % 45 = 28) :
  (c + d) % 30 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2771_277136


namespace NUMINAMATH_CALUDE_trapezoid_area_l2771_277197

/-- The area of a trapezoid with height 2a, one base 5a, and the other base 4a, is 9a² -/
theorem trapezoid_area (a : ℝ) : 
  let height : ℝ := 2 * a
  let base1 : ℝ := 5 * a
  let base2 : ℝ := 4 * a
  let area : ℝ := (height * (base1 + base2)) / 2
  area = 9 * a^2 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2771_277197


namespace NUMINAMATH_CALUDE_divisor_sum_l2771_277117

theorem divisor_sum (k m : ℕ) 
  (h1 : 30^k ∣ 929260) 
  (h2 : 20^m ∣ 929260) : 
  (3^k - k^3) + (2^m - m^3) = 2 := by
sorry

end NUMINAMATH_CALUDE_divisor_sum_l2771_277117


namespace NUMINAMATH_CALUDE_expression_simplification_l2771_277180

theorem expression_simplification (a : ℝ) (ha : a = 2018) :
  (a^2 - 3*a) / (a^2 + a) / ((a - 3) / (a^2 - 1)) * ((a + 1) / (a - 1)) = a := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2771_277180


namespace NUMINAMATH_CALUDE_o2_moles_combined_l2771_277182

-- Define the molecules and their molar ratios in the reaction
structure Reaction :=
  (C2H6_ratio : ℚ)
  (O2_ratio : ℚ)
  (C2H4O_ratio : ℚ)
  (H2O_ratio : ℚ)

-- Define the balanced reaction
def balanced_reaction : Reaction :=
  { C2H6_ratio := 1
  , O2_ratio := 1/2
  , C2H4O_ratio := 1
  , H2O_ratio := 1 }

-- Theorem statement
theorem o2_moles_combined 
  (r : Reaction) 
  (h1 : r.C2H6_ratio = 1) 
  (h2 : r.C2H4O_ratio = 1) 
  (h3 : r = balanced_reaction) : 
  r.O2_ratio = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_o2_moles_combined_l2771_277182


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2771_277105

def A : Set ℝ := {x | 3 * x + 2 > 0}
def B : Set ℝ := {x | (x + 1) * (x - 3) > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | x > 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2771_277105


namespace NUMINAMATH_CALUDE_jaden_toy_cars_l2771_277173

theorem jaden_toy_cars (initial_cars birthday_cars sister_cars friend_cars final_cars : ℕ) :
  initial_cars = 14 →
  birthday_cars = 12 →
  sister_cars = 8 →
  friend_cars = 3 →
  final_cars = 43 →
  ∃ (bought_cars : ℕ), 
    initial_cars + birthday_cars + bought_cars - sister_cars - friend_cars = final_cars ∧
    bought_cars = 28 :=
by sorry

end NUMINAMATH_CALUDE_jaden_toy_cars_l2771_277173


namespace NUMINAMATH_CALUDE_solve_equation_l2771_277128

theorem solve_equation (x y : ℝ) : y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2771_277128


namespace NUMINAMATH_CALUDE_project_scores_mode_l2771_277149

def project_scores : List ℕ := [7, 10, 9, 8, 7, 9, 9, 8]

def mode (l : List ℕ) : ℕ := 
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem project_scores_mode :
  mode project_scores = 9 := by sorry

end NUMINAMATH_CALUDE_project_scores_mode_l2771_277149


namespace NUMINAMATH_CALUDE_only_negative_four_squared_is_correct_l2771_277152

theorem only_negative_four_squared_is_correct : 
  (2^4 ≠ 8) ∧ 
  (-4^2 = -16) ∧ 
  (-8 - 8 ≠ 0) ∧ 
  ((-3)^2 ≠ 6) := by
  sorry

end NUMINAMATH_CALUDE_only_negative_four_squared_is_correct_l2771_277152


namespace NUMINAMATH_CALUDE_quadrilateral_prism_volume_l2771_277143

/-- A quadrilateral prism with specific properties -/
structure QuadrilateralPrism where
  -- The base is a rhombus with apex angle 60°
  base_is_rhombus : Bool
  base_apex_angle : ℝ
  -- The angle between each face and the base is 60°
  face_base_angle : ℝ
  -- There exists a point inside with distance 1 to base and each face
  interior_point_exists : Bool
  -- Volume of the prism
  volume : ℝ

/-- The volume of a quadrilateral prism with specific properties is 8√3 -/
theorem quadrilateral_prism_volume 
  (P : QuadrilateralPrism) 
  (h1 : P.base_is_rhombus = true)
  (h2 : P.base_apex_angle = 60)
  (h3 : P.face_base_angle = 60)
  (h4 : P.interior_point_exists = true) :
  P.volume = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_prism_volume_l2771_277143


namespace NUMINAMATH_CALUDE_double_inequality_l2771_277176

theorem double_inequality (a b : ℝ) (h : a > b) : 2 * a > 2 * b := by
  sorry

end NUMINAMATH_CALUDE_double_inequality_l2771_277176


namespace NUMINAMATH_CALUDE_calculate_expression_l2771_277186

theorem calculate_expression : 150 * (150 - 5) - (150 * 150 - 7) = -743 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2771_277186


namespace NUMINAMATH_CALUDE_toys_ratio_l2771_277107

theorem toys_ratio (s : ℚ) : 
  (s * 20 = (142 - 20 - s * 20) - 2) →
  (s * 20 + (142 - 20 - s * 20) + 20 = 142) →
  (s = 3) :=
by sorry

end NUMINAMATH_CALUDE_toys_ratio_l2771_277107


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_isosceles_triangle_l2771_277126

/-- The radius of the inscribed circle in an isosceles triangle --/
theorem inscribed_circle_radius_isosceles_triangle 
  (A B C : EuclideanSpace ℝ (Fin 2)) 
  (h_isosceles : dist A B = dist A C) 
  (h_AB : dist A B = 7)
  (h_BC : dist B C = 6) :
  let s := (dist A B + dist A C + dist B C) / 2
  let area := Real.sqrt (s * (s - dist A B) * (s - dist A C) * (s - dist B C))
  area / s = (3 * Real.sqrt 10) / 5 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_isosceles_triangle_l2771_277126


namespace NUMINAMATH_CALUDE_hyperbola_symmetric_points_parabola_midpoint_l2771_277190

/-- Given a hyperbola, two symmetric points on it, and their midpoint on a parabola, prove the possible values of m -/
theorem hyperbola_symmetric_points_parabola_midpoint (m : ℝ) : 
  (∃ (M N : ℝ × ℝ),
    -- M and N are on the hyperbola
    (M.1^2 - M.2^2/3 = 1) ∧ (N.1^2 - N.2^2/3 = 1) ∧
    -- M and N are symmetric about y = x + m
    (M.2 + N.2 = M.1 + N.1 + 2*m) ∧
    -- The midpoint of MN is on the parabola y^2 = 18x
    (((M.2 + N.2)/2)^2 = 18 * ((M.1 + N.1)/2))) →
  (m = 0 ∨ m = -8) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_symmetric_points_parabola_midpoint_l2771_277190


namespace NUMINAMATH_CALUDE_orangeade_price_theorem_l2771_277167

/-- Represents the price of orangeade per glass -/
@[ext] structure OrangeadePrice where
  price : ℚ

/-- Represents the composition of orangeade -/
@[ext] structure OrangeadeComposition where
  orange_juice : ℚ
  water : ℚ

/-- Calculates the total volume of orangeade -/
def total_volume (c : OrangeadeComposition) : ℚ :=
  c.orange_juice + c.water

/-- Calculates the revenue from selling orangeade -/
def revenue (price : OrangeadePrice) (volume : ℚ) : ℚ :=
  price.price * volume

theorem orangeade_price_theorem 
  (day1_comp : OrangeadeComposition)
  (day2_comp : OrangeadeComposition)
  (day2_price : OrangeadePrice)
  (h1 : day1_comp.orange_juice = day1_comp.water)
  (h2 : day2_comp.orange_juice = day1_comp.orange_juice)
  (h3 : day2_comp.water = 2 * day2_comp.orange_juice)
  (h4 : day2_price.price = 32/100)
  (h5 : ∃ (day1_price : OrangeadePrice), 
        revenue day1_price (total_volume day1_comp) = 
        revenue day2_price (total_volume day2_comp)) :
  ∃ (day1_price : OrangeadePrice), day1_price.price = 48/100 := by
sorry


end NUMINAMATH_CALUDE_orangeade_price_theorem_l2771_277167


namespace NUMINAMATH_CALUDE_cos_alpha_plus_5pi_over_4_l2771_277193

theorem cos_alpha_plus_5pi_over_4 (α : ℝ) (h : Real.sin (α - π/4) = 1/3) :
  Real.cos (α + 5*π/4) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_5pi_over_4_l2771_277193


namespace NUMINAMATH_CALUDE_max_distance_from_origin_dog_max_distance_l2771_277147

/-- The maximum distance a point on a circle can be from the origin,
    given the circle's center coordinates and radius. -/
theorem max_distance_from_origin (x y r : ℝ) : 
  let center_distance := Real.sqrt (x^2 + y^2)
  let max_distance := center_distance + r
  ∀ p : ℝ × ℝ, (p.1 - x)^2 + (p.2 - y)^2 = r^2 → 
    p.1^2 + p.2^2 ≤ max_distance^2 :=
by
  sorry

/-- The specific case for the dog problem -/
theorem dog_max_distance : 
  let x : ℝ := 6
  let y : ℝ := 8
  let r : ℝ := 15
  let center_distance := Real.sqrt (x^2 + y^2)
  let max_distance := center_distance + r
  max_distance = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_max_distance_from_origin_dog_max_distance_l2771_277147


namespace NUMINAMATH_CALUDE_equation_solution_l2771_277188

theorem equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 3) 
  (h : 3 / a + 6 / b = 2 / 3) : a = 9 * b / (2 * b - 18) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2771_277188


namespace NUMINAMATH_CALUDE_min_students_same_choice_l2771_277127

theorem min_students_same_choice (n : ℕ) (m : ℕ) (h1 : n = 45) (h2 : m = 6) :
  ∃ k : ℕ, k ≥ 16 ∧ k * m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_min_students_same_choice_l2771_277127


namespace NUMINAMATH_CALUDE_reader_group_size_l2771_277187

theorem reader_group_size (S L B : ℕ) (hS : S = 180) (hL : L = 88) (hB : B = 18) :
  S + L - B = 250 := by
  sorry

end NUMINAMATH_CALUDE_reader_group_size_l2771_277187


namespace NUMINAMATH_CALUDE_quadratic_roots_conditions_l2771_277153

/-- The quadratic equation (2m+1)x^2 + 4mx + 2m-3 = 0 has:
    1. Two distinct real roots iff m ∈ (-3/4, -1/2) ∪ (-1/2, ∞)
    2. Two equal real roots iff m = -3/4
    3. No real roots iff m ∈ (-∞, -3/4) -/
theorem quadratic_roots_conditions (m : ℝ) :
  let a := 2*m + 1
  let b := 4*m
  let c := 2*m - 3
  let discriminant := b^2 - 4*a*c
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0) ↔ 
    (m > -3/4 ∧ m ≠ -1/2) ∧
  (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ discriminant = 0) ↔ 
    (m = -3/4) ∧
  (∀ x : ℝ, a*x^2 + b*x + c ≠ 0) ↔ 
    (m < -3/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_conditions_l2771_277153


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2771_277185

theorem inequality_equivalence :
  ∀ x : ℝ, |(7 - 2*x) / 4| < 3 ↔ -2.5 < x ∧ x < 9.5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2771_277185


namespace NUMINAMATH_CALUDE_prob_red_then_king_diamonds_standard_deck_l2771_277179

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (red_cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- Probability of drawing a red card first and then the King of Diamonds second -/
def prob_red_then_king_diamonds (d : Deck) : Rat :=
  if d.total_cards = 52 ∧ d.red_cards = 26 ∧ d.ranks = 13 ∧ d.suits = 4 then
    1 / 102
  else
    0

/-- Theorem stating the probability of drawing a red card first and then the King of Diamonds second -/
theorem prob_red_then_king_diamonds_standard_deck :
  ∃ (d : Deck), prob_red_then_king_diamonds d = 1 / 102 :=
sorry

end NUMINAMATH_CALUDE_prob_red_then_king_diamonds_standard_deck_l2771_277179


namespace NUMINAMATH_CALUDE_max_sin_cos_sum_l2771_277178

theorem max_sin_cos_sum (A : Real) : 2 * Real.sin (A / 2) + Real.cos (A / 2) ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_sin_cos_sum_l2771_277178


namespace NUMINAMATH_CALUDE_log_stack_sum_l2771_277101

theorem log_stack_sum : ∀ (a₁ aₙ n : ℕ),
  a₁ = 12 →
  aₙ = 3 →
  n = 10 →
  (n : ℝ) / 2 * (a₁ + aₙ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l2771_277101


namespace NUMINAMATH_CALUDE_dilan_initial_marbles_l2771_277151

/-- The number of people involved in the marble redistribution --/
def num_people : ℕ := 4

/-- The number of marbles each person has after redistribution --/
def marbles_after : ℕ := 15

/-- Martha's initial number of marbles --/
def martha_initial : ℕ := 20

/-- Phillip's initial number of marbles --/
def phillip_initial : ℕ := 19

/-- Veronica's initial number of marbles --/
def veronica_initial : ℕ := 7

/-- The theorem stating Dilan's initial number of marbles --/
theorem dilan_initial_marbles :
  (num_people * marbles_after) - (martha_initial + phillip_initial + veronica_initial) = 14 :=
by sorry

end NUMINAMATH_CALUDE_dilan_initial_marbles_l2771_277151


namespace NUMINAMATH_CALUDE_positive_y_intercept_l2771_277170

/-- A line that intersects the y-axis in the positive half-plane -/
structure PositiveYInterceptLine where
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line equation is y = 2x + b -/
  equation : ∀ (x y : ℝ), y = 2 * x + b
  /-- The line intersects the y-axis in the positive half-plane -/
  positive_intercept : ∃ (y : ℝ), y > 0 ∧ y = b

/-- The y-intercept of a line that intersects the y-axis in the positive half-plane is positive -/
theorem positive_y_intercept (l : PositiveYInterceptLine) : l.b > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_y_intercept_l2771_277170


namespace NUMINAMATH_CALUDE_last_locker_opened_l2771_277196

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents the set of lockers -/
def Lockers := Fin 2048 → LockerState

/-- Defines the opening pattern for the lockers -/
def openingPattern (n : Nat) (lockers : Lockers) : Lockers :=
  sorry

/-- Defines the process of opening lockers until all are open -/
def openAllLockers (lockers : Lockers) : Nat :=
  sorry

/-- Theorem stating that the last locker to be opened is 1999 -/
theorem last_locker_opened (initialLockers : Lockers) 
    (h : ∀ i, initialLockers i = LockerState.Closed) : 
    openAllLockers initialLockers = 1999 :=
  sorry

end NUMINAMATH_CALUDE_last_locker_opened_l2771_277196


namespace NUMINAMATH_CALUDE_language_courses_enrollment_l2771_277114

theorem language_courses_enrollment (total : ℕ) (french : ℕ) (german : ℕ) (spanish : ℕ)
  (french_german : ℕ) (german_spanish : ℕ) (spanish_french : ℕ) (all_three : ℕ) :
  total = 180 →
  french = 60 →
  german = 50 →
  spanish = 35 →
  french_german = 20 →
  german_spanish = 15 →
  spanish_french = 10 →
  all_three = 5 →
  total - (french + german + spanish - french_german - german_spanish - spanish_french + all_three) = 80 := by
sorry

end NUMINAMATH_CALUDE_language_courses_enrollment_l2771_277114


namespace NUMINAMATH_CALUDE_largest_average_is_17_multiples_l2771_277174

def upper_bound : ℕ := 100810

def average_of_multiples (n : ℕ) : ℚ :=
  let last_multiple := upper_bound - (upper_bound % n)
  (n + last_multiple) / 2

theorem largest_average_is_17_multiples :
  average_of_multiples 17 > average_of_multiples 11 ∧
  average_of_multiples 17 > average_of_multiples 13 ∧
  average_of_multiples 17 > average_of_multiples 19 :=
by sorry

end NUMINAMATH_CALUDE_largest_average_is_17_multiples_l2771_277174


namespace NUMINAMATH_CALUDE_jen_bird_count_l2771_277194

/-- The number of ducks Jen has -/
def num_ducks : ℕ := 150

/-- The number of chickens Jen has -/
def num_chickens : ℕ := (num_ducks - 10) / 4

/-- The total number of birds Jen has -/
def total_birds : ℕ := num_ducks + num_chickens

theorem jen_bird_count : total_birds = 185 := by
  sorry

end NUMINAMATH_CALUDE_jen_bird_count_l2771_277194


namespace NUMINAMATH_CALUDE_volume_ratio_l2771_277133

-- Define the vertices of the larger pyramid
def large_pyramid_vertices : List (Fin 4 → ℚ) := [
  (λ i => if i = 0 then 1 else 0),
  (λ i => if i = 1 then 1 else 0),
  (λ i => if i = 2 then 1 else 0),
  (λ i => if i = 3 then 1 else 0),
  (λ _ => 0)
]

-- Define the center of the base of the larger pyramid
def base_center : Fin 4 → ℚ := λ _ => 1/4

-- Define the vertices of the smaller pyramid
def small_pyramid_vertices : List (Fin 4 → ℚ) := 
  base_center :: (List.range 4).map (λ i => λ j => if i = j then 1/2 else 0)

-- Define a function to calculate the volume of a pyramid
def pyramid_volume (vertices : List (Fin 4 → ℚ)) : ℚ := sorry

-- Theorem stating the volume ratio
theorem volume_ratio : 
  (pyramid_volume small_pyramid_vertices) / (pyramid_volume large_pyramid_vertices) = 3/64 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_l2771_277133


namespace NUMINAMATH_CALUDE_length_of_A_l2771_277148

def A : ℝ × ℝ := (0, 9)
def B : ℝ × ℝ := (0, 12)
def C : ℝ × ℝ := (2, 8)

def on_line_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

def intersect_at (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))

theorem length_of_A'B' :
  ∀ A' B' : ℝ × ℝ,
  on_line_y_eq_x A' →
  on_line_y_eq_x B' →
  intersect_at A A' C →
  intersect_at B B' C →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_length_of_A_l2771_277148


namespace NUMINAMATH_CALUDE_playground_area_l2771_277102

/-- The area of a rectangular playground with perimeter 90 meters and length three times the width -/
theorem playground_area : 
  ∀ (length width : ℝ),
  length > 0 → width > 0 →
  2 * (length + width) = 90 →
  length = 3 * width →
  length * width = 379.6875 := by
sorry

end NUMINAMATH_CALUDE_playground_area_l2771_277102

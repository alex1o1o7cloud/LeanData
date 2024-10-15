import Mathlib

namespace NUMINAMATH_CALUDE_only_tiger_leopard_valid_l2396_239669

-- Define the animals
inductive Animal : Type
| Lion : Animal
| Tiger : Animal
| Leopard : Animal
| Elephant : Animal

-- Define a pair of animals
def AnimalPair := (Animal × Animal)

-- Define the conditions
def validPair (pair : AnimalPair) : Prop :=
  -- Two different animals are sent
  pair.1 ≠ pair.2 ∧
  -- If lion is sent, tiger must be sent
  (pair.1 = Animal.Lion ∨ pair.2 = Animal.Lion) → 
    (pair.1 = Animal.Tiger ∨ pair.2 = Animal.Tiger) ∧
  -- If leopard is not sent, tiger cannot be sent
  (pair.1 ≠ Animal.Leopard ∧ pair.2 ≠ Animal.Leopard) → 
    (pair.1 ≠ Animal.Tiger ∧ pair.2 ≠ Animal.Tiger) ∧
  -- If leopard is sent, elephant cannot be sent
  (pair.1 = Animal.Leopard ∨ pair.2 = Animal.Leopard) → 
    (pair.1 ≠ Animal.Elephant ∧ pair.2 ≠ Animal.Elephant)

-- Theorem: The only valid pair is Tiger and Leopard
theorem only_tiger_leopard_valid :
  ∀ (pair : AnimalPair), validPair pair ↔ 
    ((pair.1 = Animal.Tiger ∧ pair.2 = Animal.Leopard) ∨
     (pair.1 = Animal.Leopard ∧ pair.2 = Animal.Tiger)) :=
by sorry

end NUMINAMATH_CALUDE_only_tiger_leopard_valid_l2396_239669


namespace NUMINAMATH_CALUDE_max_sum_of_remaining_pairs_l2396_239600

/-- Given a set of four distinct real numbers, this function returns the list of their six pairwise sums. -/
def pairwiseSums (a b c d : ℝ) : List ℝ :=
  [a + b, a + c, a + d, b + c, b + d, c + d]

/-- This theorem states that given four distinct real numbers whose pairwise sums include 210, 360, 330, and 300,
    the maximum possible sum of the remaining two pairwise sums is 870. -/
theorem max_sum_of_remaining_pairs (a b c d : ℝ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∃ (l : List ℝ), l = pairwiseSums a b c d ∧ 
    (210 ∈ l) ∧ (360 ∈ l) ∧ (330 ∈ l) ∧ (300 ∈ l)) →
  (∃ (x y : ℝ), x ∈ pairwiseSums a b c d ∧ 
                y ∈ pairwiseSums a b c d ∧ 
                x ≠ 210 ∧ x ≠ 360 ∧ x ≠ 330 ∧ x ≠ 300 ∧
                y ≠ 210 ∧ y ≠ 360 ∧ y ≠ 330 ∧ y ≠ 300 ∧
                x + y ≤ 870) :=
by sorry


end NUMINAMATH_CALUDE_max_sum_of_remaining_pairs_l2396_239600


namespace NUMINAMATH_CALUDE_root_condition_implies_m_range_l2396_239604

theorem root_condition_implies_m_range :
  ∀ (m : ℝ) (x₁ x₂ : ℝ),
    (m + 3) * x₁^2 - 4 * m * x₁ + 2 * m - 1 = 0 →
    (m + 3) * x₂^2 - 4 * m * x₂ + 2 * m - 1 = 0 →
    x₁ * x₂ < 0 →
    (x₁ < 0 ∧ x₂ > 0 → |x₁| > x₂) →
    (x₂ < 0 ∧ x₁ > 0 → |x₂| > x₁) →
    -3 < m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_root_condition_implies_m_range_l2396_239604


namespace NUMINAMATH_CALUDE_mixed_oil_rate_l2396_239650

/-- The rate of mixed oil per litre given two different oils -/
theorem mixed_oil_rate (volume1 : ℝ) (price1 : ℝ) (volume2 : ℝ) (price2 : ℝ) :
  volume1 = 10 →
  price1 = 50 →
  volume2 = 5 →
  price2 = 66 →
  (volume1 * price1 + volume2 * price2) / (volume1 + volume2) = 55.33 := by
  sorry

end NUMINAMATH_CALUDE_mixed_oil_rate_l2396_239650


namespace NUMINAMATH_CALUDE_marley_samantha_apple_ratio_l2396_239619

/-- Proves that the ratio of Marley's apples to Samantha's apples is 3:1 -/
theorem marley_samantha_apple_ratio :
  let louis_oranges : ℕ := 5
  let louis_apples : ℕ := 3
  let samantha_oranges : ℕ := 8
  let samantha_apples : ℕ := 7
  let marley_oranges : ℕ := 2 * louis_oranges
  let marley_total_fruits : ℕ := 31
  let marley_apples : ℕ := marley_total_fruits - marley_oranges
  (marley_apples : ℚ) / samantha_apples = 3 / 1 := by
  sorry


end NUMINAMATH_CALUDE_marley_samantha_apple_ratio_l2396_239619


namespace NUMINAMATH_CALUDE_triangle_area_is_four_thirds_l2396_239639

-- Define the line m: 3x - y + 2 = 0
def line_m (x y : ℝ) : Prop := 3 * x - y + 2 = 0

-- Define the symmetric line l with respect to the x-axis
def line_l (x y : ℝ) : Prop := 3 * x + y + 2 = 0

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Theorem statement
theorem triangle_area_is_four_thirds :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    line_m x₁ y₁ ∧ y_axis x₁ ∧
    line_m x₂ y₂ ∧ x₂ = -2/3 ∧ y₂ = 0 ∧
    line_l x₃ y₃ ∧ y_axis x₃ ∧
    (1/2 * abs (x₂ * (y₁ - y₃))) = 4/3 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_is_four_thirds_l2396_239639


namespace NUMINAMATH_CALUDE_mn_product_is_66_l2396_239693

/-- A parabola shifted from y = x^2 --/
structure ShiftedParabola where
  m : ℝ
  n : ℝ
  h_shift : ℝ := 3  -- left shift
  v_shift : ℝ := 2  -- upward shift

/-- The product of m and n for a parabola y = x^2 + mx + n
    obtained by shifting y = x^2 up by 2 units and left by 3 units --/
def mn_product (p : ShiftedParabola) : ℝ := p.m * p.n

/-- Theorem: The product mn equals 66 for the specified shifted parabola --/
theorem mn_product_is_66 (p : ShiftedParabola) : mn_product p = 66 := by
  sorry

end NUMINAMATH_CALUDE_mn_product_is_66_l2396_239693


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l2396_239640

theorem quadratic_roots_transformation (α β : ℝ) (p : ℝ) : 
  (3 * α^2 + 5 * α + 2 = 0) →
  (3 * β^2 + 5 * β + 2 = 0) →
  ((α^2 + 2) + (β^2 + 2) = -(p)) →
  (p = -49/9) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l2396_239640


namespace NUMINAMATH_CALUDE_range_of_m_l2396_239620

/-- The range of m given specific conditions on the roots of a quadratic equation -/
theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∧ 
  ¬(1 < m ∧ m < 3) ∧
  ¬¬(∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  m ≥ 3 ∨ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2396_239620


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2396_239613

-- Define the universal set U
def U : Set ℝ := {x | x < 5}

-- Define the set A
def A : Set ℝ := {x | x - 2 ≤ 0}

-- State the theorem
theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {x | 2 < x ∧ x < 5} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2396_239613


namespace NUMINAMATH_CALUDE_min_sum_with_log_condition_l2396_239603

theorem min_sum_with_log_condition (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_log : Real.log a + Real.log b = Real.log (a + b)) :
  ∀ x y : ℝ, x > 0 → y > 0 → Real.log x + Real.log y = Real.log (x + y) → a + b ≤ x + y ∧ a + b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_log_condition_l2396_239603


namespace NUMINAMATH_CALUDE_remainder_when_divided_by_nine_l2396_239638

/-- The smallest positive integer satisfying the given conditions -/
def smallest_n : ℕ := sorry

/-- The first condition: n mod 8 = 6 -/
axiom cond1 : smallest_n % 8 = 6

/-- The second condition: n mod 7 = 5 -/
axiom cond2 : smallest_n % 7 = 5

/-- The smallest_n is indeed the smallest positive integer satisfying both conditions -/
axiom smallest : ∀ m : ℕ, m > 0 → m % 8 = 6 → m % 7 = 5 → m ≥ smallest_n

/-- Theorem: The smallest n satisfying both conditions leaves a remainder of 1 when divided by 9 -/
theorem remainder_when_divided_by_nine : smallest_n % 9 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_when_divided_by_nine_l2396_239638


namespace NUMINAMATH_CALUDE_problem_equivalent_l2396_239663

theorem problem_equivalent : (16^1011) / 8 = 2^4033 := by
  sorry

end NUMINAMATH_CALUDE_problem_equivalent_l2396_239663


namespace NUMINAMATH_CALUDE_smallest_b_is_correct_l2396_239644

/-- N(b) is the number of natural numbers a for which x^2 + ax + b = 0 has integer roots -/
def N (b : ℕ) : ℕ := sorry

/-- The smallest value of b for which N(b) = 20 -/
def smallest_b : ℕ := 240

theorem smallest_b_is_correct :
  (N smallest_b = 20) ∧ (∀ b : ℕ, b < smallest_b → N b ≠ 20) := by sorry

end NUMINAMATH_CALUDE_smallest_b_is_correct_l2396_239644


namespace NUMINAMATH_CALUDE_books_second_shop_correct_l2396_239668

/-- The number of books bought from the second shop -/
def books_second_shop : ℕ := 20

/-- The number of books bought from the first shop -/
def books_first_shop : ℕ := 27

/-- The cost of books from the first shop in rupees -/
def cost_first_shop : ℕ := 581

/-- The cost of books from the second shop in rupees -/
def cost_second_shop : ℕ := 594

/-- The average price per book in rupees -/
def average_price : ℕ := 25

theorem books_second_shop_correct : 
  books_second_shop = 20 ∧
  books_first_shop = 27 ∧
  cost_first_shop = 581 ∧
  cost_second_shop = 594 ∧
  average_price = 25 →
  (cost_first_shop + cost_second_shop : ℚ) / (books_first_shop + books_second_shop) = average_price := by
  sorry

end NUMINAMATH_CALUDE_books_second_shop_correct_l2396_239668


namespace NUMINAMATH_CALUDE_combined_8th_grade_percentage_is_21_11_percent_l2396_239633

-- Define the schools and their properties
def parkwood_students : ℕ := 150
def maplewood_students : ℕ := 120
def parkwood_8th_grade_percentage : ℚ := 18 / 100
def maplewood_8th_grade_percentage : ℚ := 25 / 100

-- Define the combined percentage of 8th grade students
def combined_8th_grade_percentage : ℚ := 
  (parkwood_8th_grade_percentage * parkwood_students + maplewood_8th_grade_percentage * maplewood_students) / 
  (parkwood_students + maplewood_students)

-- Theorem statement
theorem combined_8th_grade_percentage_is_21_11_percent : 
  combined_8th_grade_percentage = 2111 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_combined_8th_grade_percentage_is_21_11_percent_l2396_239633


namespace NUMINAMATH_CALUDE_functions_equal_if_surjective_injective_and_greater_or_equal_l2396_239651

theorem functions_equal_if_surjective_injective_and_greater_or_equal
  (f g : ℕ → ℕ)
  (h_surj : Function.Surjective f)
  (h_inj : Function.Injective g)
  (h_ge : ∀ n : ℕ, f n ≥ g n) :
  ∀ n : ℕ, f n = g n := by
  sorry

end NUMINAMATH_CALUDE_functions_equal_if_surjective_injective_and_greater_or_equal_l2396_239651


namespace NUMINAMATH_CALUDE_a_range_when_p_and_q_false_l2396_239683

/-- Proposition p: y = a^x is monotonically decreasing on ℝ -/
def p (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → a^x > a^y

/-- Proposition q: y = log(ax^2 - x + a) has range ℝ -/
def q (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, a * x^2 - x + a > 0 ∧ Real.log (a * x^2 - x + a) = y

/-- If "p and q" is false, then a is in (0, 1/2] ∪ (1, ∞) -/
theorem a_range_when_p_and_q_false (a : ℝ) : ¬(p a ∧ q a) → (0 < a ∧ a ≤ 1/2) ∨ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_when_p_and_q_false_l2396_239683


namespace NUMINAMATH_CALUDE_evaluate_expression_l2396_239632

theorem evaluate_expression : (-3)^4 / 3^2 - 2^5 + 7^2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2396_239632


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2396_239606

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2396_239606


namespace NUMINAMATH_CALUDE_macaroon_weight_l2396_239660

theorem macaroon_weight
  (total_macaroons : ℕ)
  (num_bags : ℕ)
  (remaining_weight : ℚ)
  (h1 : total_macaroons = 12)
  (h2 : num_bags = 4)
  (h3 : remaining_weight = 45)
  (h4 : total_macaroons % num_bags = 0)  -- Ensures equal distribution
  : ∃ (weight_per_macaroon : ℚ),
    weight_per_macaroon * (total_macaroons - total_macaroons / num_bags) = remaining_weight ∧
    weight_per_macaroon = 5 := by
  sorry

end NUMINAMATH_CALUDE_macaroon_weight_l2396_239660


namespace NUMINAMATH_CALUDE_isogonal_conjugate_is_conic_l2396_239662

/-- Trilinear coordinates -/
structure TrilinearCoord where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A triangle -/
structure Triangle where
  A : TrilinearCoord
  B : TrilinearCoord
  C : TrilinearCoord

/-- A line in trilinear coordinates -/
structure Line where
  p : ℝ
  q : ℝ
  r : ℝ

/-- Isogonal conjugation transformation -/
def isogonalConjugate (l : Line) : TrilinearCoord → Prop :=
  fun point => l.p * point.y * point.z + l.q * point.x * point.z + l.r * point.x * point.y = 0

/-- Definition of a conic section -/
def isConicSection (f : TrilinearCoord → Prop) : Prop := sorry

/-- The theorem to be proved -/
theorem isogonal_conjugate_is_conic (t : Triangle) (l : Line) 
  (h1 : l.p ≠ 0) (h2 : l.q ≠ 0) (h3 : l.r ≠ 0)
  (h4 : l.p * t.A.x + l.q * t.A.y + l.r * t.A.z ≠ 0)
  (h5 : l.p * t.B.x + l.q * t.B.y + l.r * t.B.z ≠ 0)
  (h6 : l.p * t.C.x + l.q * t.C.y + l.r * t.C.z ≠ 0) :
  isConicSection (isogonalConjugate l) ∧ 
  isogonalConjugate l t.A ∧ 
  isogonalConjugate l t.B ∧ 
  isogonalConjugate l t.C :=
sorry

end NUMINAMATH_CALUDE_isogonal_conjugate_is_conic_l2396_239662


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l2396_239610

/-- Proves that the average mark of excluded students is 40 given the conditions of the problem -/
theorem excluded_students_average_mark
  (total_students : ℕ)
  (total_average : ℚ)
  (remaining_average : ℚ)
  (excluded_count : ℕ)
  (h_total_students : total_students = 33)
  (h_total_average : total_average = 90)
  (h_remaining_average : remaining_average = 95)
  (h_excluded_count : excluded_count = 3) :
  let remaining_count := total_students - excluded_count
  let total_marks := total_students * total_average
  let remaining_marks := remaining_count * remaining_average
  let excluded_marks := total_marks - remaining_marks
  excluded_marks / excluded_count = 40 := by
  sorry

end NUMINAMATH_CALUDE_excluded_students_average_mark_l2396_239610


namespace NUMINAMATH_CALUDE_parabola_properties_l2396_239659

-- Define the parabola
def parabola (m n x : ℝ) : ℝ := m * x^2 - 2 * m^2 * x + n

-- Define the conditions and theorem
theorem parabola_properties
  (m n x₁ x₂ y₁ y₂ : ℝ)
  (h_m : m ≠ 0)
  (h_parabola₁ : parabola m n x₁ = y₁)
  (h_parabola₂ : parabola m n x₂ = y₂) :
  (x₁ = 1 ∧ x₂ = 3 ∧ y₁ = y₂ → 2 = (x₁ + x₂) / 2) ∧
  (x₁ + x₂ > 4 ∧ x₁ < x₂ ∧ y₁ < y₂ → 0 < m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2396_239659


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l2396_239607

theorem binomial_expansion_problem (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (Real.sqrt 5 * x - 1)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -64 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l2396_239607


namespace NUMINAMATH_CALUDE_shekar_social_studies_score_l2396_239608

theorem shekar_social_studies_score 
  (math_score : ℕ) 
  (science_score : ℕ) 
  (english_score : ℕ) 
  (biology_score : ℕ) 
  (average_score : ℕ) 
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 67)
  (h4 : biology_score = 95)
  (h5 : average_score = 77)
  (h6 : (math_score + science_score + english_score + biology_score + social_studies_score) / 5 = average_score) :
  social_studies_score = 82 :=
by
  sorry

#check shekar_social_studies_score

end NUMINAMATH_CALUDE_shekar_social_studies_score_l2396_239608


namespace NUMINAMATH_CALUDE_remaining_sweets_theorem_l2396_239646

/-- The number of remaining sweets after Aaron's actions -/
def remaining_sweets (C S P R L : ℕ) : ℕ :=
  let eaten_C := (2 * C) / 5
  let eaten_S := S / 4
  let eaten_P := (3 * P) / 5
  let given_C := (C - P / 4) / 3
  let discarded_R := (3 * R) / 2
  let eaten_L := (eaten_S * 6) / 5
  (C - eaten_C - given_C) + (S - eaten_S) + (P - eaten_P) + (if R > discarded_R then R - discarded_R else 0) + (L - eaten_L)

theorem remaining_sweets_theorem :
  remaining_sweets 30 100 60 25 150 = 232 := by
  sorry

end NUMINAMATH_CALUDE_remaining_sweets_theorem_l2396_239646


namespace NUMINAMATH_CALUDE_max_value_implies_ratio_l2396_239628

/-- Given a function f(x) = 3sin(x) + 4cos(x) that reaches its maximum value at x = θ,
    prove that (sin(2θ) + cos²(θ) + 1) / cos(2θ) = 15/7 -/
theorem max_value_implies_ratio (θ : ℝ) 
  (h : ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 3 * Real.sin θ + 4 * Real.cos θ) :
  (Real.sin (2 * θ) + Real.cos θ ^ 2 + 1) / Real.cos (2 * θ) = 15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_ratio_l2396_239628


namespace NUMINAMATH_CALUDE_product_of_roots_l2396_239699

theorem product_of_roots (x : ℝ) : 
  (x^3 - 15*x^2 + 75*x - 50 = 0) → 
  ∃ p q r : ℝ, (x - p)*(x - q)*(x - r) = x^3 - 15*x^2 + 75*x - 50 ∧ p*q*r = 50 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2396_239699


namespace NUMINAMATH_CALUDE_frustum_volume_l2396_239601

/-- Represents a frustum of a cone -/
structure Frustum where
  upper_base_area : ℝ
  lower_base_area : ℝ
  lateral_area : ℝ

/-- Calculate the volume of a frustum -/
def volume (f : Frustum) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem frustum_volume (f : Frustum) 
  (h1 : f.upper_base_area = π)
  (h2 : f.lower_base_area = 4 * π)
  (h3 : f.lateral_area = 6 * π) : 
  volume f = 4 * π := by
  sorry

end NUMINAMATH_CALUDE_frustum_volume_l2396_239601


namespace NUMINAMATH_CALUDE_max_cakes_l2396_239679

/-- Represents the configuration of cuts on a rectangular cake -/
structure CakeCut where
  rows : Nat
  columns : Nat

/-- Calculates the total number of cake pieces after cutting -/
def totalPieces (cut : CakeCut) : Nat :=
  (cut.rows + 1) * (cut.columns + 1)

/-- Calculates the number of interior pieces -/
def interiorPieces (cut : CakeCut) : Nat :=
  (cut.rows - 1) * (cut.columns - 1)

/-- Calculates the number of perimeter pieces -/
def perimeterPieces (cut : CakeCut) : Nat :=
  2 * (cut.rows + cut.columns)

/-- Checks if the cutting configuration satisfies the given condition -/
def isValidCut (cut : CakeCut) : Prop :=
  interiorPieces cut = perimeterPieces cut + 1

/-- The main theorem stating the maximum number of cakes -/
theorem max_cakes : ∃ (cut : CakeCut), isValidCut cut ∧ 
  totalPieces cut = 65 ∧ 
  (∀ (other : CakeCut), isValidCut other → totalPieces other ≤ 65) :=
sorry

end NUMINAMATH_CALUDE_max_cakes_l2396_239679


namespace NUMINAMATH_CALUDE_intersection_A_B_union_B_complement_A_C_subset_complement_B_l2396_239634

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}

-- State the theorems
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 5} := by sorry

theorem union_B_complement_A : B ∪ Aᶜ = {x : ℝ | x ≤ 5 ∨ x ≥ 9} := by sorry

theorem C_subset_complement_B (a : ℝ) :
  C a ⊆ Bᶜ ↔ a < -4 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_B_complement_A_C_subset_complement_B_l2396_239634


namespace NUMINAMATH_CALUDE_solve_for_n_l2396_239685

theorem solve_for_n (P s k m n : ℝ) (h : P = s / (1 + k + m) ^ n) :
  n = Real.log (s / P) / Real.log (1 + k + m) :=
by sorry

end NUMINAMATH_CALUDE_solve_for_n_l2396_239685


namespace NUMINAMATH_CALUDE_max_men_with_all_items_and_married_l2396_239671

theorem max_men_with_all_items_and_married 
  (total_men : ℕ) 
  (married_men : ℕ) 
  (men_with_tv : ℕ) 
  (men_with_radio : ℕ) 
  (men_with_ac : ℕ) 
  (h_total : total_men = 100)
  (h_married : married_men = 85)
  (h_tv : men_with_tv = 75)
  (h_radio : men_with_radio = 85)
  (h_ac : men_with_ac = 70)
  : ∃ (max_all_items_married : ℕ), 
    max_all_items_married ≤ 70 ∧ 
    max_all_items_married ≤ married_men ∧
    max_all_items_married ≤ men_with_tv ∧
    max_all_items_married ≤ men_with_radio ∧
    max_all_items_married ≤ men_with_ac :=
by sorry

end NUMINAMATH_CALUDE_max_men_with_all_items_and_married_l2396_239671


namespace NUMINAMATH_CALUDE_profit_is_three_l2396_239687

/-- Calculates the profit from selling apples and oranges -/
def calculate_profit (apple_buy_price : ℚ) (apple_sell_price : ℚ) 
                     (orange_buy_price : ℚ) (orange_sell_price : ℚ)
                     (apples_sold : ℕ) (oranges_sold : ℕ) : ℚ :=
  let apple_profit := (apple_sell_price - apple_buy_price) * apples_sold
  let orange_profit := (orange_sell_price - orange_buy_price) * oranges_sold
  apple_profit + orange_profit

/-- Proves that the profit from selling 5 apples and 5 oranges is $3 -/
theorem profit_is_three :
  let apple_buy_price : ℚ := 3 / 2  -- $3 for 2 apples
  let apple_sell_price : ℚ := 2     -- $10 for 5 apples, so $2 each
  let orange_buy_price : ℚ := 9 / 10  -- $2.70 for 3 oranges
  let orange_sell_price : ℚ := 1
  let apples_sold : ℕ := 5
  let oranges_sold : ℕ := 5
  calculate_profit apple_buy_price apple_sell_price orange_buy_price orange_sell_price apples_sold oranges_sold = 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_is_three_l2396_239687


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2396_239635

theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (first_term : a 1 = 1) 
  (fifth_term : a 5 = 4) : 
  a 3 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2396_239635


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_five_satisfies_inequality_five_is_largest_integer_l2396_239686

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (x - 1 : ℚ) / 4 - 3 / 7 < 2 / 3 → x ≤ 5 :=
by sorry

theorem five_satisfies_inequality :
  (5 - 1 : ℚ) / 4 - 3 / 7 < 2 / 3 :=
by sorry

theorem five_is_largest_integer :
  ∃ x : ℤ, x = 5 ∧
    ((x - 1 : ℚ) / 4 - 3 / 7 < 2 / 3) ∧
    (∀ y : ℤ, y > x → (y - 1 : ℚ) / 4 - 3 / 7 ≥ 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_five_satisfies_inequality_five_is_largest_integer_l2396_239686


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_9_l2396_239649

def is_divisible_by_9 (n : Nat) : Prop := n % 9 = 0

theorem missing_digit_divisible_by_9 :
  let n : Nat := 65304
  is_divisible_by_9 n ∧ 
  ∃ d : Nat, d < 10 ∧ n = 65000 + 300 + d * 10 + 4 :=
by sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_9_l2396_239649


namespace NUMINAMATH_CALUDE_sprint_stats_change_l2396_239697

theorem sprint_stats_change (n : Nat) (avg_10 : ℝ) (var_10 : ℝ) (time_11 : ℝ) :
  n = 10 →
  avg_10 = 8.2 →
  var_10 = 2.2 →
  time_11 = 8.2 →
  let avg_11 := (n * avg_10 + time_11) / (n + 1)
  let var_11 := (n * var_10 + (time_11 - avg_10)^2) / (n + 1)
  avg_11 = avg_10 ∧ var_11 < var_10 := by
  sorry

#check sprint_stats_change

end NUMINAMATH_CALUDE_sprint_stats_change_l2396_239697


namespace NUMINAMATH_CALUDE_door_unlock_problem_l2396_239652

-- Define the number of buttons and the number of buttons to press
def total_buttons : ℕ := 10
def buttons_to_press : ℕ := 3

-- Define the time for each attempt
def time_per_attempt : ℕ := 2

-- Calculate the total number of combinations
def total_combinations : ℕ := Nat.choose total_buttons buttons_to_press

-- Define the maximum time needed (in seconds)
def max_time : ℕ := total_combinations * time_per_attempt

-- Define the average time needed (in seconds)
def avg_time : ℚ := (1 + total_combinations : ℚ) / 2 * time_per_attempt

-- Define the maximum number of attempts in 60 seconds
def max_attempts_in_minute : ℕ := 60 / time_per_attempt

theorem door_unlock_problem :
  (max_time = 240) ∧
  (avg_time = 121) ∧
  (max_attempts_in_minute = 30) ∧
  ((max_attempts_in_minute - 1 : ℚ) / total_combinations = 29 / 120) := by
  sorry

end NUMINAMATH_CALUDE_door_unlock_problem_l2396_239652


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_can_be_prime_l2396_239622

theorem sum_of_four_consecutive_integers_can_be_prime : 
  ∃ n : ℤ, Prime (n + (n + 1) + (n + 2) + (n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_can_be_prime_l2396_239622


namespace NUMINAMATH_CALUDE_initial_average_marks_l2396_239664

theorem initial_average_marks
  (n : ℕ)  -- number of students
  (correct_avg : ℚ)  -- correct average after fixing the error
  (wrong_mark : ℚ)  -- wrongly noted mark
  (right_mark : ℚ)  -- correct mark
  (h1 : n = 30)  -- there are 30 students
  (h2 : correct_avg = 98)  -- correct average is 98
  (h3 : wrong_mark = 70)  -- wrongly noted mark is 70
  (h4 : right_mark = 10)  -- correct mark is 10
  : (n * correct_avg + (right_mark - wrong_mark)) / n = 100 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_marks_l2396_239664


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2396_239654

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 2x - x^2 -/
def f (x : ℝ) : ℝ := 2*x - x^2

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l2396_239654


namespace NUMINAMATH_CALUDE_bicycle_spoke_ratio_l2396_239656

theorem bicycle_spoke_ratio : 
  ∀ (front_spokes back_spokes : ℕ),
    front_spokes = 20 →
    front_spokes + back_spokes = 60 →
    (back_spokes : ℚ) / front_spokes = 2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_spoke_ratio_l2396_239656


namespace NUMINAMATH_CALUDE_five_thursdays_in_august_l2396_239667

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific date in July or August -/
structure Date where
  month : Nat
  day : Nat

/-- Function to get the day of the week for a given date -/
def dayOfWeek (d : Date) : DayOfWeek := sorry

/-- Function to check if a date is a Monday -/
def isMonday (d : Date) : Prop :=
  dayOfWeek d = DayOfWeek.Monday

/-- Function to check if a date is a Thursday -/
def isThursday (d : Date) : Prop :=
  dayOfWeek d = DayOfWeek.Thursday

/-- Theorem stating that if July has five Mondays, then August has five Thursdays -/
theorem five_thursdays_in_august
  (h1 : ∃ d1 d2 d3 d4 d5 : Date,
    d1.month = 7 ∧ d2.month = 7 ∧ d3.month = 7 ∧ d4.month = 7 ∧ d5.month = 7 ∧
    d1.day < d2.day ∧ d2.day < d3.day ∧ d3.day < d4.day ∧ d4.day < d5.day ∧
    isMonday d1 ∧ isMonday d2 ∧ isMonday d3 ∧ isMonday d4 ∧ isMonday d5)
  (h2 : ∀ d : Date, d.month = 7 → d.day ≤ 31)
  (h3 : ∀ d : Date, d.month = 8 → d.day ≤ 31) :
  ∃ d1 d2 d3 d4 d5 : Date,
    d1.month = 8 ∧ d2.month = 8 ∧ d3.month = 8 ∧ d4.month = 8 ∧ d5.month = 8 ∧
    d1.day < d2.day ∧ d2.day < d3.day ∧ d3.day < d4.day ∧ d4.day < d5.day ∧
    isThursday d1 ∧ isThursday d2 ∧ isThursday d3 ∧ isThursday d4 ∧ isThursday d5 :=
sorry

end NUMINAMATH_CALUDE_five_thursdays_in_august_l2396_239667


namespace NUMINAMATH_CALUDE_five_integers_with_remainder_one_l2396_239698

theorem five_integers_with_remainder_one : 
  ∃! (S : Finset ℕ), 
    S.card = 5 ∧ 
    (∀ n ∈ S, n ≤ 50) ∧ 
    (∀ n ∈ S, n % 11 = 1) :=
by sorry

end NUMINAMATH_CALUDE_five_integers_with_remainder_one_l2396_239698


namespace NUMINAMATH_CALUDE_intersection_points_polar_l2396_239657

/-- The intersection points of ρ = 2sin θ and ρ cos θ = -√3/2 in polar coordinates -/
theorem intersection_points_polar (θ : Real) (h : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  ∃ (ρ₁ ρ₂ θ₁ θ₂ : Real),
    (ρ₁ = 2 * Real.sin θ₁ ∧ ρ₁ * Real.cos θ₁ = -Real.sqrt 3 / 2) ∧
    (ρ₂ = 2 * Real.sin θ₂ ∧ ρ₂ * Real.cos θ₂ = -Real.sqrt 3 / 2) ∧
    ((ρ₁ = 1 ∧ θ₁ = 5 * Real.pi / 6) ∨ (ρ₁ = Real.sqrt 3 ∧ θ₁ = 2 * Real.pi / 3)) ∧
    ((ρ₂ = 1 ∧ θ₂ = 5 * Real.pi / 6) ∨ (ρ₂ = Real.sqrt 3 ∧ θ₂ = 2 * Real.pi / 3)) ∧
    ρ₁ ≠ ρ₂ :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_polar_l2396_239657


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2396_239624

theorem diophantine_equation_solutions :
  (∀ k : ℤ, (101 * (4 + 13 * k) - 13 * (31 + 101 * k) = 1)) ∧
  (∀ k : ℤ, (79 * (-6 + 19 * k) - 19 * (-25 + 79 * k) = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2396_239624


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l2396_239692

/-- Given a person walking at 4 km/hr, if increasing their speed to 5 km/hr
    would result in walking 6 km more in the same time, then the actual
    distance traveled is 24 km. -/
theorem actual_distance_traveled (actual_speed actual_distance : ℝ) 
    (h1 : actual_speed = 4)
    (h2 : actual_distance / actual_speed = (actual_distance + 6) / 5) :
  actual_distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l2396_239692


namespace NUMINAMATH_CALUDE_total_shuttlecocks_distributed_l2396_239641

-- Define the number of students
def num_students : ℕ := 24

-- Define the number of shuttlecocks per student
def shuttlecocks_per_student : ℕ := 19

-- Theorem to prove
theorem total_shuttlecocks_distributed :
  num_students * shuttlecocks_per_student = 456 := by
  sorry

end NUMINAMATH_CALUDE_total_shuttlecocks_distributed_l2396_239641


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l2396_239648

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^2 + 3

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < 4 → f x < f y := by sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l2396_239648


namespace NUMINAMATH_CALUDE_worksheets_graded_before_additional_l2396_239609

/-- The number of worksheets initially given to the teacher to grade. -/
def initial_worksheets : ℕ := 6

/-- The number of additional worksheets turned in later. -/
def additional_worksheets : ℕ := 18

/-- The total number of worksheets to grade after the additional ones were turned in. -/
def total_worksheets : ℕ := 20

/-- The number of worksheets graded before the additional ones were turned in. -/
def graded_worksheets : ℕ := 4

theorem worksheets_graded_before_additional :
  initial_worksheets - graded_worksheets + additional_worksheets = total_worksheets :=
sorry

end NUMINAMATH_CALUDE_worksheets_graded_before_additional_l2396_239609


namespace NUMINAMATH_CALUDE_undefined_values_expression_undefined_l2396_239674

theorem undefined_values (x : ℝ) : 
  (2 * x^2 - 8 * x - 42 = 0) ↔ (x = 7 ∨ x = -3) :=
by sorry

theorem expression_undefined (x : ℝ) :
  ¬ (∃ y : ℝ, y = (3 * x^2 - 1) / (2 * x^2 - 8 * x - 42)) ↔ (x = 7 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_undefined_values_expression_undefined_l2396_239674


namespace NUMINAMATH_CALUDE_store_a_prices_store_b_original_price_l2396_239647

/-- Represents a store selling notebooks -/
structure Store where
  hardcover_price : ℕ
  softcover_price : ℕ
  hardcover_more_expensive : hardcover_price = softcover_price + 3

/-- Theorem for Store A's notebook prices -/
theorem store_a_prices (a : Store) 
  (h1 : 240 / a.hardcover_price = 195 / a.softcover_price) :
  a.hardcover_price = 16 := by
  sorry

/-- Represents Store B's discount policy -/
def discount_policy (price : ℕ) (quantity : ℕ) : ℕ :=
  if quantity ≥ 30 then price - 3 else price

/-- Theorem for Store B's original hardcover notebook price -/
theorem store_b_original_price (b : Store) (m : ℕ)
  (h1 : m < 30)
  (h2 : m + 5 ≥ 30)
  (h3 : m * b.hardcover_price = (m + 5) * (b.hardcover_price - 3)) :
  b.hardcover_price = 18 := by
  sorry

end NUMINAMATH_CALUDE_store_a_prices_store_b_original_price_l2396_239647


namespace NUMINAMATH_CALUDE_intersection_probability_odd_polygon_l2396_239665

/-- 
Given a convex polygon with 2n + 1 vertices, this theorem states that 
the probability of two independently chosen diagonals intersecting
is n(2n - 1) / (3(2n^2 - n - 2)).
-/
theorem intersection_probability_odd_polygon (n : ℕ) : 
  let vertices := 2*n + 1
  let diagonals := vertices * (vertices - 3) / 2
  let intersecting_pairs := (vertices.choose 4)
  let total_pairs := diagonals.choose 2
  (intersecting_pairs : ℚ) / total_pairs = n * (2*n - 1) / (3 * (2*n^2 - n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_probability_odd_polygon_l2396_239665


namespace NUMINAMATH_CALUDE_same_functions_l2396_239626

theorem same_functions (x : ℝ) (h : x ≠ 1) : (x - 1) ^ 0 = 1 / ((x - 1) ^ 0) := by
  sorry

end NUMINAMATH_CALUDE_same_functions_l2396_239626


namespace NUMINAMATH_CALUDE_quadratic_intersection_and_equivalence_l2396_239611

-- Define the quadratic function p
def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0

-- Define the function q
def q (x m : ℝ) : Prop := (x - m) * (x - 3*m) < 0

theorem quadratic_intersection_and_equivalence :
  (∃ (a b : ℝ), a = 4 ∧ b = 5 ∧ 
    (∀ x : ℝ, (p x ∧ q x 4) ↔ (a < x ∧ x < b))) ∧
  (∃ (c d : ℝ), c = 5/3 ∧ d = 2 ∧ 
    (∀ m : ℝ, m > 0 → 
      ((∀ x : ℝ, ¬(q x m) ↔ ¬(p x)) ↔ (c ≤ m ∧ m ≤ d)))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_and_equivalence_l2396_239611


namespace NUMINAMATH_CALUDE_average_marks_of_failed_boys_l2396_239666

theorem average_marks_of_failed_boys
  (total_boys : ℕ)
  (overall_average : ℚ)
  (passed_average : ℚ)
  (passed_boys : ℕ)
  (h1 : total_boys = 120)
  (h2 : overall_average = 38)
  (h3 : passed_average = 39)
  (h4 : passed_boys = 115) :
  (total_boys * overall_average - passed_boys * passed_average) / (total_boys - passed_boys) = 15 := by
sorry

end NUMINAMATH_CALUDE_average_marks_of_failed_boys_l2396_239666


namespace NUMINAMATH_CALUDE_largest_sum_proof_l2396_239655

theorem largest_sum_proof : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/9, 1/3 + 1/8]
  ∀ x ∈ sums, x ≤ 1/3 + 1/4 ∧ 1/3 + 1/4 = 7/12 := by
sorry

end NUMINAMATH_CALUDE_largest_sum_proof_l2396_239655


namespace NUMINAMATH_CALUDE_double_markup_percentage_l2396_239696

theorem double_markup_percentage (initial_price : ℝ) (markup_percentage : ℝ) : 
  markup_percentage = 40 →
  let first_markup := initial_price * (1 + markup_percentage / 100)
  let second_markup := first_markup * (1 + markup_percentage / 100)
  (second_markup - initial_price) / initial_price * 100 = 96 := by
  sorry

end NUMINAMATH_CALUDE_double_markup_percentage_l2396_239696


namespace NUMINAMATH_CALUDE_prime_ones_and_seven_l2396_239621

/-- Represents a number with n-1 digits 1 and one digit 7 -/
def A (n : ℕ) (k : ℕ) : ℕ := (10^n + 54 * 10^k - 1) / 9

/-- Checks if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- All numbers with n-1 digits 1 and one digit 7 are prime -/
def all_prime (n : ℕ) : Prop := ∀ k : ℕ, k < n → is_prime (A n k)

theorem prime_ones_and_seven :
  ∀ n : ℕ, (all_prime n ↔ n = 1 ∨ n = 2) :=
sorry

end NUMINAMATH_CALUDE_prime_ones_and_seven_l2396_239621


namespace NUMINAMATH_CALUDE_vacation_cost_l2396_239631

theorem vacation_cost (C : ℝ) : 
  (C / 6 - C / 8 = 120) → C = 2880 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_l2396_239631


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2396_239643

theorem quadratic_transformation (a b c : ℝ) :
  (∃ (m q : ℝ), ∀ x, ax^2 + bx + c = 5*(x - 3)^2 + 15) →
  (∃ (m p q : ℝ), ∀ x, 4*ax^2 + 4*bx + 4*c = m*(x - p)^2 + q ∧ p = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2396_239643


namespace NUMINAMATH_CALUDE_dessert_and_coffee_percentage_l2396_239670

theorem dessert_and_coffee_percentage :
  let dessert_percentage : ℝ := 100 - 25.00000000000001
  let dessert_and_coffee_ratio : ℝ := 1 - 0.2
  dessert_and_coffee_ratio * dessert_percentage = 59.999999999999992 :=
by sorry

end NUMINAMATH_CALUDE_dessert_and_coffee_percentage_l2396_239670


namespace NUMINAMATH_CALUDE_min_value_x_sqrt_9_minus_x_squared_l2396_239623

theorem min_value_x_sqrt_9_minus_x_squared :
  ∃ (x : ℝ), -3 < x ∧ x < 0 ∧
  x * Real.sqrt (9 - x^2) = -9/2 ∧
  ∀ (y : ℝ), -3 < y ∧ y < 0 →
  y * Real.sqrt (9 - y^2) ≥ -9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_sqrt_9_minus_x_squared_l2396_239623


namespace NUMINAMATH_CALUDE_unique_prime_solution_l2396_239618

theorem unique_prime_solution : ∃! (p : ℕ), 
  Prime p ∧ 
  ∃ (x y : ℕ), 
    x > 0 ∧ 
    y > 0 ∧ 
    p + 49 = 2 * x^2 ∧ 
    p^2 + 49 = 2 * y^2 ∧ 
    p = 23 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l2396_239618


namespace NUMINAMATH_CALUDE_least_x_value_l2396_239680

theorem least_x_value (x y : ℤ) (h : x * y + 6 * x + 8 * y = -4) :
  ∀ z : ℤ, z ≥ -52 ∨ ¬∃ w : ℤ, z * w + 6 * z + 8 * w = -4 :=
sorry

end NUMINAMATH_CALUDE_least_x_value_l2396_239680


namespace NUMINAMATH_CALUDE_root_difference_l2396_239625

theorem root_difference (p q : ℝ) (h : p ≠ q) :
  let r := (p + q + Real.sqrt ((p - q)^2)) / 2
  let s := (p + q - Real.sqrt ((p - q)^2)) / 2
  r - s = |p - q| := by
sorry

end NUMINAMATH_CALUDE_root_difference_l2396_239625


namespace NUMINAMATH_CALUDE_corrected_mean_l2396_239695

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ original_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 48 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) = n * (36.5 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l2396_239695


namespace NUMINAMATH_CALUDE_inequality_proof_l2396_239675

def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2 * |a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2396_239675


namespace NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l2396_239661

/-- The equation of a line with slope 2 and y-intercept 4 is y = 2x + 4 -/
theorem line_equation_from_slope_and_intercept :
  ∀ (x y : ℝ), (∃ (m b : ℝ), m = 2 ∧ b = 4 ∧ y = m * x + b) → y = 2 * x + 4 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l2396_239661


namespace NUMINAMATH_CALUDE_inequality_proof_l2396_239684

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ∧ 
  ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2396_239684


namespace NUMINAMATH_CALUDE_circle_m_equation_and_common_chord_length_l2396_239627

/-- Circle M passes through points (0,-2) and (4,0), and its center lies on the line x-y=0 -/
def CircleM : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 - 2*p.2 - 8 = 0}

/-- Circle N with equation (x-3)^2 + y^2 = 25 -/
def CircleN : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 25}

/-- The common chord between CircleM and CircleN -/
def CommonChord : Set (ℝ × ℝ) :=
  CircleM ∩ CircleN

theorem circle_m_equation_and_common_chord_length :
  (∀ p : ℝ × ℝ, p ∈ CircleM ↔ p.1^2 + p.2^2 - 2*p.1 - 2*p.2 - 8 = 0) ∧
  (∃ a b : ℝ × ℝ, a ∈ CommonChord ∧ b ∈ CommonChord ∧ 
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_m_equation_and_common_chord_length_l2396_239627


namespace NUMINAMATH_CALUDE_iced_coffee_consumption_ratio_l2396_239682

/-- Proves that the ratio of daily servings consumed to servings per container is 1:2 -/
theorem iced_coffee_consumption_ratio 
  (servings_per_bottle : ℕ) 
  (cost_per_bottle : ℚ) 
  (total_cost : ℚ) 
  (duration_weeks : ℕ) 
  (h1 : servings_per_bottle = 6)
  (h2 : cost_per_bottle = 3)
  (h3 : total_cost = 21)
  (h4 : duration_weeks = 2) :
  (total_cost / cost_per_bottle * servings_per_bottle) / (duration_weeks * 7) / servings_per_bottle = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_iced_coffee_consumption_ratio_l2396_239682


namespace NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_y2_l2396_239688

theorem min_value_xy_over_x2_plus_y2 (x y : ℝ) 
  (hx : 1/4 ≤ x ∧ x ≤ 3/5) (hy : 1/5 ≤ y ∧ y ≤ 2/3) :
  x * y / (x^2 + y^2) ≥ 24/73 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_y2_l2396_239688


namespace NUMINAMATH_CALUDE_largest_valid_number_l2396_239694

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧  -- Four-digit number
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧  -- All digits are different
  (∀ i j, i < j → (n / 10^i) % 10 ≤ (n / 10^j) % 10)  -- No two digits can be swapped to form a smaller number

theorem largest_valid_number : 
  is_valid_number 7089 ∧ ∀ m, is_valid_number m → m ≤ 7089 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l2396_239694


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_cube_l2396_239612

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_factor_for_perfect_cube (x : ℕ) (hx : x = 3 * 40 * 75) :
  ∃ y : ℕ, y > 0 ∧ is_perfect_cube (x * y) ∧ ∀ z : ℕ, z > 0 → is_perfect_cube (x * z) → y ≤ z :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_cube_l2396_239612


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_m_range_l2396_239636

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | m / 2 ≤ (p.1 - 2)^2 + p.2^2 ∧ (p.1 - 2)^2 + p.2^2 ≤ m^2}

def B (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * m ≤ p.1 + p.2 ∧ p.1 + p.2 ≤ 2 * m + 1}

-- State the theorem
theorem intersection_nonempty_implies_m_range (m : ℝ) :
  (A m ∩ B m).Nonempty → 1/2 ≤ m ∧ m ≤ 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_m_range_l2396_239636


namespace NUMINAMATH_CALUDE_continued_fraction_evaluation_l2396_239691

theorem continued_fraction_evaluation :
  let x : ℚ := 1 + (3 / (4 + (5 / (6 + (7/8)))))
  x = 85/52 := by
sorry

end NUMINAMATH_CALUDE_continued_fraction_evaluation_l2396_239691


namespace NUMINAMATH_CALUDE_unique_number_divisible_by_792_l2396_239605

theorem unique_number_divisible_by_792 :
  ∀ (x y z : ℕ), x < 10 → y < 10 → z < 10 →
  (13 * 100000 + x * 10000 + y * 1000 + 45 * 10 + z) % 792 = 0 →
  (13 * 100000 + x * 10000 + y * 1000 + 45 * 10 + z) = 1380456 := by
sorry

end NUMINAMATH_CALUDE_unique_number_divisible_by_792_l2396_239605


namespace NUMINAMATH_CALUDE_max_substitutions_is_fifty_l2396_239602

/-- A type representing a fifth-degree polynomial -/
def FifthDegreePolynomial := ℕ → ℕ

/-- Given a list of ten fifth-degree polynomials, returns the maximum number of consecutive
    natural numbers that can be substituted to produce an arithmetic progression -/
def max_consecutive_substitutions (polynomials : List FifthDegreePolynomial) : ℕ :=
  sorry

/-- The main theorem stating that the maximum number of consecutive substitutions is 50 -/
theorem max_substitutions_is_fifty :
  ∀ (polynomials : List FifthDegreePolynomial),
    polynomials.length = 10 →
    max_consecutive_substitutions polynomials = 50 :=
  sorry

end NUMINAMATH_CALUDE_max_substitutions_is_fifty_l2396_239602


namespace NUMINAMATH_CALUDE_total_serving_time_l2396_239614

def total_patients : ℕ := 12
def standard_serving_time : ℕ := 5
def special_needs_ratio : ℚ := 1/3
def special_needs_time_increase : ℚ := 1/5

theorem total_serving_time :
  let special_patients := total_patients * special_needs_ratio
  let standard_patients := total_patients - special_patients
  let special_serving_time := standard_serving_time * (1 + special_needs_time_increase)
  let total_time := standard_patients * standard_serving_time + special_patients * special_serving_time
  total_time = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_serving_time_l2396_239614


namespace NUMINAMATH_CALUDE_owen_burger_spending_l2396_239673

/-- The number of days in June -/
def days_in_june : ℕ := 30

/-- The number of burgers Owen buys per day -/
def burgers_per_day : ℕ := 2

/-- The cost of each burger in dollars -/
def cost_per_burger : ℕ := 12

/-- Theorem: Owen's total spending on burgers in June is $720 -/
theorem owen_burger_spending :
  days_in_june * burgers_per_day * cost_per_burger = 720 := by
  sorry


end NUMINAMATH_CALUDE_owen_burger_spending_l2396_239673


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l2396_239629

theorem inverse_proportion_y_relationship :
  ∀ (y₁ y₂ y₃ : ℝ),
  (y₁ = -3 / (-3)) →
  (y₂ = -3 / (-1)) →
  (y₃ = -3 / (1/3)) →
  (y₃ < y₁) ∧ (y₁ < y₂) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l2396_239629


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l2396_239689

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l2396_239689


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2396_239637

theorem quadratic_factorization (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2396_239637


namespace NUMINAMATH_CALUDE_motorcyclist_hiker_meeting_time_l2396_239690

/-- Calculates the waiting time for a motorcyclist and hiker to meet given their speeds and initial separation time. -/
theorem motorcyclist_hiker_meeting_time 
  (hiker_speed : ℝ) 
  (motorcyclist_speed : ℝ) 
  (separation_time : ℝ) 
  (hᵢ : hiker_speed = 6) 
  (mᵢ : motorcyclist_speed = 30) 
  (tᵢ : separation_time = 12 / 60) : 
  (motorcyclist_speed * separation_time) / hiker_speed = 1 := by
  sorry

#eval (60 : ℕ)  -- Expected result in minutes

end NUMINAMATH_CALUDE_motorcyclist_hiker_meeting_time_l2396_239690


namespace NUMINAMATH_CALUDE_leopard_arrangement_l2396_239630

theorem leopard_arrangement (n : ℕ) (h : n = 8) :
  (2 : ℕ) * Nat.factorial (n - 2) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_leopard_arrangement_l2396_239630


namespace NUMINAMATH_CALUDE_quadratic_has_real_root_l2396_239617

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_real_root_l2396_239617


namespace NUMINAMATH_CALUDE_olympiads_spellings_l2396_239615

def word_length : Nat := 9

-- Function to calculate the number of valid spellings
def valid_spellings (n : Nat) : Nat :=
  if n = 0 then 1
  else if n = word_length then 2^(n-1)
  else 2 * valid_spellings (n-1)

theorem olympiads_spellings :
  valid_spellings word_length = 256 :=
by sorry

end NUMINAMATH_CALUDE_olympiads_spellings_l2396_239615


namespace NUMINAMATH_CALUDE_circle_M_properties_l2396_239681

-- Define the circle M
def circle_M : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ (x + 3)^2 + (y - 4)^2 = 25}

-- Define the points
def point_A : ℝ × ℝ := (-3, -1)
def point_B : ℝ × ℝ := (-6, 8)
def point_C : ℝ × ℝ := (1, 1)
def point_P : ℝ × ℝ := (2, 3)

-- Define the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := x = 2
def tangent_line_2 (x y : ℝ) : Prop := 12 * x - 5 * y - 9 = 0

theorem circle_M_properties :
  (point_A ∈ circle_M) ∧
  (point_B ∈ circle_M) ∧
  (point_C ∈ circle_M) ∧
  (∀ (x y : ℝ), tangent_line_1 x y → (x, y) ∈ circle_M → (x, y) = point_P) ∧
  (∀ (x y : ℝ), tangent_line_2 x y → (x, y) ∈ circle_M → (x, y) = point_P) :=
sorry

end NUMINAMATH_CALUDE_circle_M_properties_l2396_239681


namespace NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l2396_239642

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l2396_239642


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_50_l2396_239616

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sum of factorials from 1 to n -/
def sumFactorials (n : ℕ) : ℕ := 
  (List.range n).map (λ i => factorial (i + 1)) |> List.sum

/-- The units digit of the sum of factorials from 1! to 50! is 3 -/
theorem units_digit_sum_factorials_50 : 
  unitsDigit (sumFactorials 50) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_50_l2396_239616


namespace NUMINAMATH_CALUDE_smallest_integers_difference_difference_is_27720_l2396_239672

theorem smallest_integers_difference : ℕ → Prop :=
  fun d =>
    ∃ n₁ n₂ : ℕ,
      n₁ > 1 ∧ n₂ > 1 ∧
      n₂ > n₁ ∧
      (∀ k : ℕ, 2 ≤ k → k ≤ 11 → n₁ % k = 1) ∧
      (∀ k : ℕ, 2 ≤ k → k ≤ 11 → n₂ % k = 1) ∧
      (∀ m : ℕ, m > 1 → (∀ k : ℕ, 2 ≤ k → k ≤ 11 → m % k = 1) → m ≥ n₁) ∧
      d = n₂ - n₁

theorem difference_is_27720 : smallest_integers_difference 27720 := by sorry

end NUMINAMATH_CALUDE_smallest_integers_difference_difference_is_27720_l2396_239672


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2396_239645

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 16 players, where each player plays twice with every other player, 
    the total number of games played is 480. -/
theorem chess_tournament_games : tournament_games 16 * 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2396_239645


namespace NUMINAMATH_CALUDE_data_set_properties_l2396_239678

def data_set : List ℕ := [67, 57, 37, 40, 46, 62, 31, 47, 31, 30]

def mode (l : List ℕ) : ℕ := sorry

def range (l : List ℕ) : ℕ := sorry

def quantile (l : List ℕ) (p : ℚ) : ℚ := sorry

theorem data_set_properties :
  (mode data_set = 31) ∧
  (range data_set = 37) ∧
  (quantile data_set (1/10) = 30.5) := by
  sorry

end NUMINAMATH_CALUDE_data_set_properties_l2396_239678


namespace NUMINAMATH_CALUDE_infinitely_many_coprime_linear_combination_l2396_239677

theorem infinitely_many_coprime_linear_combination (a b n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hn : n > 0) (hab : Nat.gcd a b = 1) :
  Set.Infinite {k : ℕ | Nat.gcd (a * k + b) n = 1} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_coprime_linear_combination_l2396_239677


namespace NUMINAMATH_CALUDE_sum_a_plus_c_equals_four_l2396_239676

/-- Represents a three-digit number in the form abc -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem sum_a_plus_c_equals_four :
  ∀ (a c : Nat),
  let num1 := ThreeDigitNumber.mk 2 a 3 (by sorry)
  let num2 := ThreeDigitNumber.mk 6 c 9 (by sorry)
  (num1.toNat + 427 = num2.toNat) →
  (num2.toNat % 3 = 0) →
  a + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_plus_c_equals_four_l2396_239676


namespace NUMINAMATH_CALUDE_inequality_proof_l2396_239658

theorem inequality_proof (x y m : ℝ) (h1 : x > y) (h2 : m > 0) : x - y > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2396_239658


namespace NUMINAMATH_CALUDE_cherry_difference_l2396_239653

theorem cherry_difference (initial_cherries left_cherries : ℕ) 
  (h1 : initial_cherries = 16)
  (h2 : left_cherries = 6) :
  initial_cherries - left_cherries = 10 := by
  sorry

end NUMINAMATH_CALUDE_cherry_difference_l2396_239653

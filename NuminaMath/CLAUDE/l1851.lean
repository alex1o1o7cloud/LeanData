import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_l1851_185100

-- Define the quadratic expression
def quadratic (c : ℝ) (x : ℝ) : ℝ := -x^2 + c*x + 8

-- State the theorem
theorem quadratic_roots (c : ℝ) : 
  (∀ x, quadratic c x > 0 ↔ x < -2 ∨ x > 4) → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1851_185100


namespace NUMINAMATH_CALUDE_min_value_function_l1851_185106

theorem min_value_function (p : ℝ) (h_p : p > 0) :
  (∃ (m : ℝ), m = 4 ∧ 
    ∀ (x : ℝ), x > 1 → x + p / (x - 1) ≥ m ∧ 
    ∃ (x₀ : ℝ), x₀ > 1 ∧ x₀ + p / (x₀ - 1) = m) →
  p = 9/4 := by sorry

end NUMINAMATH_CALUDE_min_value_function_l1851_185106


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l1851_185170

theorem min_value_sqrt_sum_squares (m n : ℝ) (h : ∃ α : ℝ, m * Real.sin α + n * Real.cos α = 5) :
  (∀ x y : ℝ, (∃ β : ℝ, x * Real.sin β + y * Real.cos β = 5) → Real.sqrt (x^2 + y^2) ≥ 5) ∧
  (∃ p q : ℝ, (∃ γ : ℝ, p * Real.sin γ + q * Real.cos γ = 5) ∧ Real.sqrt (p^2 + q^2) = 5) :=
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l1851_185170


namespace NUMINAMATH_CALUDE_central_figure_diameter_bound_a_central_figure_diameter_bound_l1851_185120

/-- A convex figure with diameter 1 --/
class ConvexFigure (F : Type*) :=
  (diameter : ℝ)
  (is_one : diameter = 1)

/-- A planar convex figure --/
class PlanarConvexFigure (F : Type*) extends ConvexFigure F

/-- A spatial convex figure --/
class SpatialConvexFigure (F : Type*) extends ConvexFigure F

/-- The diameter of the central figure L(F) --/
def central_figure_diameter (F : Type*) [ConvexFigure F] : ℝ := sorry

/-- The diameter of the a-central figure II(a, F) --/
def a_central_figure_diameter (F : Type*) [ConvexFigure F] (a : ℝ) : ℝ := sorry

theorem central_figure_diameter_bound 
  (F : Type*) [ConvexFigure F] : 
  (PlanarConvexFigure F → central_figure_diameter F ≤ 1/2) ∧ 
  (SpatialConvexFigure F → central_figure_diameter F ≤ Real.sqrt 2 / 2) := by sorry

theorem a_central_figure_diameter_bound 
  (F : Type*) [ConvexFigure F] (a : ℝ) : 
  (PlanarConvexFigure F → a_central_figure_diameter F a ≤ 1 - a^2/2) ∧ 
  (SpatialConvexFigure F → a_central_figure_diameter F a ≤ Real.sqrt (1 - a^2/2)) := by sorry

end NUMINAMATH_CALUDE_central_figure_diameter_bound_a_central_figure_diameter_bound_l1851_185120


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1851_185144

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n → n ≤ 986 ∧ 17 ∣ 986 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1851_185144


namespace NUMINAMATH_CALUDE_ceiling_x_squared_values_l1851_185184

theorem ceiling_x_squared_values (x : ℝ) (h : ⌈x⌉ = 9) :
  ∃ (S : Finset ℕ), (∀ n ∈ S, ∃ y : ℝ, ⌈y⌉ = 9 ∧ ⌈y^2⌉ = n) ∧ S.card = 17 :=
sorry

end NUMINAMATH_CALUDE_ceiling_x_squared_values_l1851_185184


namespace NUMINAMATH_CALUDE_min_value_theorem_l1851_185119

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  (x + 8) / Real.sqrt (x - 1) ≥ 6 ∧
  ((x + 8) / Real.sqrt (x - 1) = 6 ↔ x = 10) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1851_185119


namespace NUMINAMATH_CALUDE_max_additional_plates_l1851_185187

/-- Represents the sets of letters for license plates -/
structure LetterSets :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Calculates the number of possible license plates -/
def numPlates (s : LetterSets) : ℕ :=
  s.first * s.second * s.third

/-- The initial letter sets -/
def initialSets : LetterSets :=
  ⟨5, 3, 4⟩

/-- The number of new letters to be added -/
def newLetters : ℕ :=
  4

/-- Constraint: at least one letter must be added to second and third sets -/
def validDistribution (d : LetterSets) : Prop :=
  d.second > initialSets.second ∧ d.third > initialSets.third ∧
  d.first + d.second + d.third = initialSets.first + initialSets.second + initialSets.third + newLetters

theorem max_additional_plates :
  ∃ (d : LetterSets), validDistribution d ∧
    ∀ (d' : LetterSets), validDistribution d' →
      numPlates d - numPlates initialSets ≥ numPlates d' - numPlates initialSets ∧
      numPlates d - numPlates initialSets = 90 :=
by sorry

end NUMINAMATH_CALUDE_max_additional_plates_l1851_185187


namespace NUMINAMATH_CALUDE_f_value_at_log_half_24_l1851_185183

def f (x : ℝ) : ℝ := sorry

theorem f_value_at_log_half_24 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_periodic : ∀ x, f (x + 2) = f x)
  (h_def : ∀ x, 0 ≤ x ∧ x < 1 → f x = 2^x - 1) :
  f (Real.log 24 / Real.log (1/2)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_f_value_at_log_half_24_l1851_185183


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1851_185188

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = 5 - x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1851_185188


namespace NUMINAMATH_CALUDE_sum_of_divisors_154_l1851_185195

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_154 : sum_of_divisors 154 = 288 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_154_l1851_185195


namespace NUMINAMATH_CALUDE_fraction_decomposition_l1851_185109

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 10) (h2 : x ≠ -2) :
  (7 * x + 3) / (x^2 - 8*x - 20) = (73/12) / (x - 10) + (11/12) / (x + 2) := by
  sorry

#check fraction_decomposition

end NUMINAMATH_CALUDE_fraction_decomposition_l1851_185109


namespace NUMINAMATH_CALUDE_circ_equation_solutions_l1851_185163

/-- Custom operation ∘ -/
def circ (a b : ℤ) : ℤ := a + b - a * b

/-- Theorem statement -/
theorem circ_equation_solutions :
  ∀ x y z : ℤ, circ (circ x y) z + circ (circ y z) x + circ (circ z x) y = 0 ↔
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨
     (x = 0 ∧ y = 2 ∧ z = 2) ∨
     (x = 2 ∧ y = 0 ∧ z = 2) ∨
     (x = 2 ∧ y = 2 ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_circ_equation_solutions_l1851_185163


namespace NUMINAMATH_CALUDE_adjacent_product_sum_nonpositive_l1851_185160

theorem adjacent_product_sum_nonpositive (a b c d : ℝ) (h : a + b + c + d = 0) :
  a * b + b * c + c * d + d * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_product_sum_nonpositive_l1851_185160


namespace NUMINAMATH_CALUDE_cricketer_average_score_l1851_185156

/-- Proves that the average score for the first 6 matches is 41 runs -/
theorem cricketer_average_score (total_matches : ℕ) (overall_average : ℚ) 
  (first_part_matches : ℕ) (second_part_matches : ℕ) (second_part_average : ℚ) :
  total_matches = 10 →
  overall_average = 389/10 →
  first_part_matches = 6 →
  second_part_matches = 4 →
  second_part_average = 143/4 →
  (overall_average * total_matches - second_part_average * second_part_matches) / first_part_matches = 41 :=
by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l1851_185156


namespace NUMINAMATH_CALUDE_diamond_circle_area_l1851_185129

/-- A diamond is a quadrilateral with four equal sides -/
structure Diamond where
  side_length : ℝ
  angle_alpha : ℝ
  angle_beta : ℝ

/-- The inscribed circle of a diamond -/
structure InscribedCircle (d : Diamond) where
  center : Point

/-- The circle passing through vertices A, O, and C -/
structure CircumscribedCircle (d : Diamond) (ic : InscribedCircle d) where
  area : ℝ

/-- Main theorem: The area of the circle passing through A, O, and C in the specified diamond -/
theorem diamond_circle_area 
  (d : Diamond) 
  (ic : InscribedCircle d) 
  (cc : CircumscribedCircle d ic) 
  (h1 : d.side_length = 8) 
  (h2 : d.angle_alpha = Real.pi / 3)  -- 60 degrees in radians
  (h3 : d.angle_beta = 2 * Real.pi / 3)  -- 120 degrees in radians
  : cc.area = 48 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_diamond_circle_area_l1851_185129


namespace NUMINAMATH_CALUDE_lions_after_one_year_l1851_185139

/-- Calculates the number of lions after a given number of months -/
def lions_after_months (initial_population : ℕ) (birth_rate : ℕ) (death_rate : ℕ) (months : ℕ) : ℕ :=
  initial_population + birth_rate * months - death_rate * months

/-- Theorem stating that given the initial conditions, there will be 148 lions after 12 months -/
theorem lions_after_one_year :
  lions_after_months 100 5 1 12 = 148 := by
  sorry

end NUMINAMATH_CALUDE_lions_after_one_year_l1851_185139


namespace NUMINAMATH_CALUDE_amusement_park_average_cost_l1851_185151

theorem amusement_park_average_cost
  (num_people : ℕ)
  (transportation_fee : ℚ)
  (admission_fee : ℚ)
  (h_num_people : num_people = 5)
  (h_transportation_fee : transportation_fee = 9.5)
  (h_admission_fee : admission_fee = 32.5) :
  (transportation_fee + admission_fee) / num_people = 8.2 :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_average_cost_l1851_185151


namespace NUMINAMATH_CALUDE_smallest_n_proof_n_is_minimal_l1851_185111

/-- Represents an answer sheet with 5 questions, each having 4 possible answers -/
def AnswerSheet := Fin 5 → Fin 4

/-- The total number of answer sheets -/
def totalSheets : ℕ := 2000

/-- Checks if two answer sheets have at most 3 matching answers -/
def atMostThreeMatches (sheet1 sheet2 : AnswerSheet) : Prop :=
  (Finset.filter (λ i => sheet1 i = sheet2 i) (Finset.univ : Finset (Fin 5))).card ≤ 3

/-- The smallest number n that satisfies the condition -/
def smallestN : ℕ := 25

theorem smallest_n_proof :
  ∀ (sheets : Finset AnswerSheet),
    sheets.card = totalSheets →
    ∀ (subset : Finset AnswerSheet),
      subset ⊆ sheets →
      subset.card = smallestN →
      ∃ (sheet1 sheet2 sheet3 sheet4 : AnswerSheet),
        sheet1 ∈ subset ∧ sheet2 ∈ subset ∧ sheet3 ∈ subset ∧ sheet4 ∈ subset ∧
        sheet1 ≠ sheet2 ∧ sheet1 ≠ sheet3 ∧ sheet1 ≠ sheet4 ∧
        sheet2 ≠ sheet3 ∧ sheet2 ≠ sheet4 ∧ sheet3 ≠ sheet4 ∧
        atMostThreeMatches sheet1 sheet2 ∧
        atMostThreeMatches sheet1 sheet3 ∧
        atMostThreeMatches sheet1 sheet4 ∧
        atMostThreeMatches sheet2 sheet3 ∧
        atMostThreeMatches sheet2 sheet4 ∧
        atMostThreeMatches sheet3 sheet4 :=
by
  sorry

theorem n_is_minimal :
  ∀ n : ℕ,
    n < smallestN →
    ∃ (sheets : Finset AnswerSheet),
      sheets.card = totalSheets ∧
      ∃ (subset : Finset AnswerSheet),
        subset ⊆ sheets ∧
        subset.card = n ∧
        ∀ (sheet1 sheet2 sheet3 sheet4 : AnswerSheet),
          sheet1 ∈ subset → sheet2 ∈ subset → sheet3 ∈ subset → sheet4 ∈ subset →
          sheet1 ≠ sheet2 → sheet1 ≠ sheet3 → sheet1 ≠ sheet4 →
          sheet2 ≠ sheet3 → sheet2 ≠ sheet4 → sheet3 ≠ sheet4 →
          ¬(atMostThreeMatches sheet1 sheet2 ∧
            atMostThreeMatches sheet1 sheet3 ∧
            atMostThreeMatches sheet1 sheet4 ∧
            atMostThreeMatches sheet2 sheet3 ∧
            atMostThreeMatches sheet2 sheet4 ∧
            atMostThreeMatches sheet3 sheet4) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_proof_n_is_minimal_l1851_185111


namespace NUMINAMATH_CALUDE_count_six_digit_integers_l1851_185177

/-- The number of different positive, six-digit integers that can be formed
    using the digits 2, 2, 2, 5, 5, and 9 -/
def sixDigitIntegers : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem count_six_digit_integers : sixDigitIntegers = 60 := by
  sorry

end NUMINAMATH_CALUDE_count_six_digit_integers_l1851_185177


namespace NUMINAMATH_CALUDE_find_f_2_l1851_185142

theorem find_f_2 (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 5 * x^2 - 4 * x + 1) : 
  f 2 = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_find_f_2_l1851_185142


namespace NUMINAMATH_CALUDE_race_speed_ratio_l1851_185127

theorem race_speed_ratio (v_a v_b : ℝ) (h : v_a > 0 ∧ v_b > 0) :
  (1 / v_a = (1 - 13/30) / v_b) → v_a / v_b = 30 / 17 := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l1851_185127


namespace NUMINAMATH_CALUDE_inequality_proof_l1851_185136

theorem inequality_proof (x : ℝ) (h : x > 0) : x + (2016^2016) / (x^2016) ≥ 2017 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1851_185136


namespace NUMINAMATH_CALUDE_percentage_equivalence_l1851_185178

theorem percentage_equivalence : 
  ∃ P : ℝ, (35 / 100 * 400 : ℝ) = P / 100 * 700 ∧ P = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equivalence_l1851_185178


namespace NUMINAMATH_CALUDE_runner_distance_l1851_185164

/-- Represents the runner's problem --/
def RunnerProblem (speed time distance : ℝ) : Prop :=
  -- Normal condition
  speed * time = distance ∧
  -- Increased speed condition
  (speed + 1) * (2/3 * time) = distance ∧
  -- Decreased speed condition
  (speed - 1) * (time + 3) = distance

/-- Theorem stating the solution to the runner's problem --/
theorem runner_distance : ∃ (speed time : ℝ), RunnerProblem speed time 6 := by
  sorry


end NUMINAMATH_CALUDE_runner_distance_l1851_185164


namespace NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l1851_185148

theorem tan_seventeen_pi_fourths : Real.tan (17 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l1851_185148


namespace NUMINAMATH_CALUDE_x_one_value_l1851_185190

theorem x_one_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h_eq : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + (x₃ - x₄)^2 + x₄^2 = 1/3) :
  x₁ = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_x_one_value_l1851_185190


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l1851_185194

theorem cubic_equation_integer_solutions :
  ∀ x y : ℤ, x^3 + y^3 - 3*x^2 + 6*y^2 + 3*x + 12*y + 6 = 0 ↔ (x = 1 ∧ y = -1) ∨ (x = 2 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l1851_185194


namespace NUMINAMATH_CALUDE_logarithm_inequality_l1851_185169

theorem logarithm_inequality (t : ℝ) (x y z : ℝ) 
  (ht : t > 1)
  (hx : x = Real.log t / Real.log 2)
  (hy : y = Real.log t / Real.log 3)
  (hz : z = Real.log t / Real.log 5) :
  3 * y < 2 * x ∧ 2 * x < 5 * z := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l1851_185169


namespace NUMINAMATH_CALUDE_relay_race_distance_l1851_185167

/-- Represents a runner in the relay race -/
structure Runner where
  name : String
  speed : ℝ
  time : ℝ

/-- Calculates the distance covered by a runner -/
def distance (runner : Runner) : ℝ := runner.speed * runner.time

theorem relay_race_distance (sadie ariana sarah : Runner)
  (h1 : sadie.speed = 3 ∧ sadie.time = 2)
  (h2 : ariana.speed = 6 ∧ ariana.time = 0.5)
  (h3 : sarah.speed = 4)
  (h4 : sadie.time + ariana.time + sarah.time = 4.5) :
  distance sadie + distance ariana + distance sarah = 17 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_distance_l1851_185167


namespace NUMINAMATH_CALUDE_list_average_problem_l1851_185182

theorem list_average_problem (n : ℕ) (original_avg : ℚ) (new_avg : ℚ) (added_num : ℤ) : 
  original_avg = 7 →
  new_avg = 6 →
  added_num = -11 →
  (n : ℚ) * original_avg + added_num = (n + 1 : ℚ) * new_avg →
  n = 17 := by sorry

end NUMINAMATH_CALUDE_list_average_problem_l1851_185182


namespace NUMINAMATH_CALUDE_power_of_64_l1851_185186

theorem power_of_64 : (64 : ℝ) ^ (5/6) = 32 := by sorry

end NUMINAMATH_CALUDE_power_of_64_l1851_185186


namespace NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_a_squared_positive_l1851_185114

theorem a_positive_sufficient_not_necessary_for_a_squared_positive :
  (∃ a : ℝ, a > 0 → a^2 > 0) ∧ 
  (∃ a : ℝ, a^2 > 0 ∧ ¬(a > 0)) := by
  sorry

end NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_a_squared_positive_l1851_185114


namespace NUMINAMATH_CALUDE_constant_difference_l1851_185130

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivative of f and g
variable (f' g' : ℝ → ℝ)

-- Assume f' and g' are the derivatives of f and g respectively
variable (hf : ∀ x, HasDerivAt f (f' x) x)
variable (hg : ∀ x, HasDerivAt g (g' x) x)

-- State the theorem
theorem constant_difference (h : ∀ x, f' x = g' x) :
  ∃ C, ∀ x, f x - g x = C :=
sorry

end NUMINAMATH_CALUDE_constant_difference_l1851_185130


namespace NUMINAMATH_CALUDE_two_girls_probability_l1851_185107

def total_students : ℕ := 10
def num_boys : ℕ := 4
def num_selected : ℕ := 3

def probability_two_girls : ℚ :=
  (Nat.choose (total_students - num_boys) 2 * Nat.choose num_boys 1) /
  Nat.choose total_students num_selected

theorem two_girls_probability :
  probability_two_girls = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_two_girls_probability_l1851_185107


namespace NUMINAMATH_CALUDE_parabola_c_value_l1851_185158

/-- A parabola that passes through the origin -/
def parabola_through_origin (c : ℝ) : ℝ → ℝ := λ x ↦ x^2 - 2*x + c - 4

/-- Theorem: For a parabola y = x^2 - 2x + c - 4 passing through the origin, c = 4 -/
theorem parabola_c_value :
  ∃ c : ℝ, (parabola_through_origin c 0 = 0) ∧ (c = 4) := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1851_185158


namespace NUMINAMATH_CALUDE_special_triangle_tan_b_l1851_185153

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi
  positive : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the properties of the specific triangle
def SpecialTriangle (t : Triangle) : Prop :=
  -- tan A, tan B, tan C are integers
  ∃ (a b c : ℤ), (Real.tan t.A = a) ∧ (Real.tan t.B = b) ∧ (Real.tan t.C = c) ∧
  -- A > B > C
  (t.A > t.B) ∧ (t.B > t.C) ∧
  -- tan A, tan B, tan C are positive
  (0 < a) ∧ (0 < b) ∧ (0 < c)

-- Theorem statement
theorem special_triangle_tan_b (t : Triangle) (h : SpecialTriangle t) : 
  Real.tan t.B = 2 := by sorry

end NUMINAMATH_CALUDE_special_triangle_tan_b_l1851_185153


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l1851_185166

/-- The number of wrapping paper varieties -/
def wrapping_paper_varieties : ℕ := 12

/-- The number of ribbon colors -/
def ribbon_colors : ℕ := 3

/-- The number of gift card types -/
def gift_card_types : ℕ := 6

/-- The number of ribbon colors available for small gifts -/
def small_gift_ribbon_colors : ℕ := 2

/-- Calculates the number of wrapping combinations for small gifts -/
def small_gift_combinations : ℕ :=
  wrapping_paper_varieties * small_gift_ribbon_colors * gift_card_types

/-- Calculates the number of wrapping combinations for large gifts -/
def large_gift_combinations : ℕ :=
  wrapping_paper_varieties * ribbon_colors * gift_card_types

theorem gift_wrapping_combinations :
  small_gift_combinations = 144 ∧ large_gift_combinations = 216 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l1851_185166


namespace NUMINAMATH_CALUDE_longer_piece_length_l1851_185147

def board_length : ℝ := 69

theorem longer_piece_length :
  ∀ (short_piece long_piece : ℝ),
  short_piece + long_piece = board_length →
  long_piece = 2 * short_piece →
  long_piece = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_longer_piece_length_l1851_185147


namespace NUMINAMATH_CALUDE_two_face_cube_probability_l1851_185140

/-- The number of small cubes a large painted cube is sawed into -/
def total_cubes : ℕ := 1000

/-- The number of edges in a cube -/
def cube_edges : ℕ := 12

/-- The number of small cubes along each edge of the large cube -/
def edge_cubes : ℕ := 10

/-- The number of small cubes with two painted faces -/
def two_face_cubes : ℕ := cube_edges * edge_cubes

/-- The probability of randomly picking a small cube with two painted faces -/
def two_face_probability : ℚ := two_face_cubes / total_cubes

theorem two_face_cube_probability :
  two_face_probability = 12 / 125 := by sorry

end NUMINAMATH_CALUDE_two_face_cube_probability_l1851_185140


namespace NUMINAMATH_CALUDE_sum_of_consecutive_multiples_of_three_l1851_185174

theorem sum_of_consecutive_multiples_of_three (a b c : ℕ) : 
  (a % 3 = 0) → 
  (b % 3 = 0) → 
  (c % 3 = 0) → 
  (b = a + 3) → 
  (c = b + 3) → 
  (c = 42) → 
  (a + b + c = 117) := by
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_multiples_of_three_l1851_185174


namespace NUMINAMATH_CALUDE_octagon_diagonals_quadratic_always_positive_l1851_185123

-- Define the number of sides in an octagon
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

-- Theorem stating that an octagon has 20 diagonals
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by sorry

-- Theorem stating that the quadratic expression is always positive
theorem quadratic_always_positive : ∀ x : ℝ, quadratic_expr x > 0 := by sorry

end NUMINAMATH_CALUDE_octagon_diagonals_quadratic_always_positive_l1851_185123


namespace NUMINAMATH_CALUDE_fraction_problem_l1851_185103

theorem fraction_problem (x : ℚ) : 
  (80 / 100 * 45 : ℚ) - (x * 25) = 16 ↔ x = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1851_185103


namespace NUMINAMATH_CALUDE_min_blocks_for_wall_l1851_185149

/-- Represents a block in the wall --/
structure Block where
  height : ℕ
  length : ℕ

/-- Represents a row in the wall --/
structure Row where
  blocks : List Block
  length : ℕ

/-- Represents the entire wall --/
structure Wall where
  rows : List Row
  height : ℕ
  length : ℕ

/-- Checks if the vertical joins are properly staggered --/
def isProperlyStaggered (wall : Wall) : Prop := sorry

/-- Checks if the wall is even on both ends --/
def isEvenOnEnds (wall : Wall) : Prop := sorry

/-- Counts the total number of blocks in the wall --/
def countBlocks (wall : Wall) : ℕ := sorry

/-- Theorem stating the minimum number of blocks required --/
theorem min_blocks_for_wall :
  ∀ (wall : Wall),
    wall.length = 120 ∧ 
    wall.height = 10 ∧ 
    (∀ b : Block, b ∈ (wall.rows.bind Row.blocks) → b.height = 1 ∧ (b.length = 2 ∨ b.length = 3)) ∧
    isProperlyStaggered wall ∧
    isEvenOnEnds wall →
    countBlocks wall ≥ 466 := by sorry

end NUMINAMATH_CALUDE_min_blocks_for_wall_l1851_185149


namespace NUMINAMATH_CALUDE_excess_cans_l1851_185112

def initial_collection : ℕ := 30 + 43 + 55
def daily_collection_rate : ℕ := 8 + 11 + 15
def days : ℕ := 14
def goal : ℕ := 400

theorem excess_cans :
  initial_collection + daily_collection_rate * days - goal = 204 :=
by sorry

end NUMINAMATH_CALUDE_excess_cans_l1851_185112


namespace NUMINAMATH_CALUDE_log_equation_solution_l1851_185192

theorem log_equation_solution (x : ℝ) :
  (x + 5 > 0) → (x - 3 > 0) → (x^2 - x - 15 > 0) →
  (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - x - 15) + 1) →
  (x = 2/3 + Real.sqrt 556 / 6 ∨ x = 2/3 - Real.sqrt 556 / 6) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1851_185192


namespace NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l1851_185150

/-- Given a 25% profit, the ratio of cost price to selling price is 4 : 5 -/
theorem cost_to_selling_price_ratio (cost_price selling_price : ℝ) 
  (h : selling_price = cost_price * (1 + 0.25)) : 
  cost_price / selling_price = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l1851_185150


namespace NUMINAMATH_CALUDE_min_correct_answers_for_score_l1851_185128

/-- AMC 10 scoring system and Mark's strategy -/
structure AMC10 where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Int
  incorrect_points : Int
  unanswered_points : Int

/-- Calculate the score based on the number of correct answers -/
def calculate_score (amc : AMC10) (correct_answers : Nat) : Int :=
  let incorrect_answers := amc.attempted_problems - correct_answers
  let unanswered := amc.total_problems - amc.attempted_problems
  correct_answers * amc.correct_points + 
  incorrect_answers * amc.incorrect_points + 
  unanswered * amc.unanswered_points

/-- Theorem stating the minimum number of correct answers needed -/
theorem min_correct_answers_for_score (amc : AMC10) 
  (h1 : amc.total_problems = 25)
  (h2 : amc.attempted_problems = 20)
  (h3 : amc.correct_points = 8)
  (h4 : amc.incorrect_points = -2)
  (h5 : amc.unanswered_points = 2)
  (target_score : Int)
  (h6 : target_score = 120) :
  (∃ n : Nat, n ≥ 15 ∧ calculate_score amc n ≥ target_score ∧ 
    ∀ m : Nat, m < 15 → calculate_score amc m < target_score) :=
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_for_score_l1851_185128


namespace NUMINAMATH_CALUDE_factorization_equality_l1851_185101

theorem factorization_equality (x : ℝ) : 3 * x^2 * (x - 5) + 5 * (x - 5) = (3 * x^2 + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1851_185101


namespace NUMINAMATH_CALUDE_white_l_shapes_count_l1851_185172

/-- Represents a square in the grid -/
inductive Square
| White
| NonWhite

/-- Represents the grid as a 2D array of squares -/
def Grid := Array (Array Square)

/-- Represents an L-shape as three connected squares -/
structure LShape where
  pos1 : Nat × Nat
  pos2 : Nat × Nat
  pos3 : Nat × Nat

/-- Checks if an L-shape is valid (connected and L-shaped) -/
def isValidLShape (l : LShape) : Bool := sorry

/-- Checks if an L-shape is entirely white in the given grid -/
def isWhiteLShape (grid : Grid) (l : LShape) : Bool := sorry

/-- Counts the number of white L-shapes in the grid -/
def countWhiteLShapes (grid : Grid) : Nat := sorry

theorem white_l_shapes_count (grid : Grid) : 
  countWhiteLShapes grid = 24 := by sorry

end NUMINAMATH_CALUDE_white_l_shapes_count_l1851_185172


namespace NUMINAMATH_CALUDE_range_of_f_l1851_185198

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1851_185198


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1851_185122

theorem solve_linear_equation (x : ℚ) : -3*x - 8 = 8*x + 3 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1851_185122


namespace NUMINAMATH_CALUDE_x_squared_plus_one_is_quadratic_l1851_185162

/-- Definition of a quadratic equation in one variable x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² + 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem stating that x² + 1 = 0 is a quadratic equation in one variable x -/
theorem x_squared_plus_one_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_one_is_quadratic_l1851_185162


namespace NUMINAMATH_CALUDE_N_squared_eq_N_minus_26I_l1851_185191

def N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 8; -4, -2]

theorem N_squared_eq_N_minus_26I : 
  N ^ 2 = N - 26 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by sorry

end NUMINAMATH_CALUDE_N_squared_eq_N_minus_26I_l1851_185191


namespace NUMINAMATH_CALUDE_thirty_switches_connections_l1851_185113

/-- Given a network of switches where each switch is directly connected to
    exactly 4 other switches, calculate the number of unique connections. -/
def uniqueConnections (n : ℕ) : ℕ :=
  (n * 4) / 2

theorem thirty_switches_connections :
  uniqueConnections 30 = 60 := by
  sorry

end NUMINAMATH_CALUDE_thirty_switches_connections_l1851_185113


namespace NUMINAMATH_CALUDE_cost_price_theorem_l1851_185193

/-- The cost price per bowl given the conditions of the problem -/
def cost_price_per_bowl (total_bowls : ℕ) (sold_bowls : ℕ) (selling_price : ℚ) (percentage_gain : ℚ) : ℚ :=
  1400 / 103

/-- Theorem stating that the cost price per bowl is 1400/103 given the problem conditions -/
theorem cost_price_theorem (total_bowls : ℕ) (sold_bowls : ℕ) (selling_price : ℚ) (percentage_gain : ℚ) 
  (h1 : total_bowls = 110)
  (h2 : sold_bowls = 100)
  (h3 : selling_price = 14)
  (h4 : percentage_gain = 27.27272727272727 / 100) :
  cost_price_per_bowl total_bowls sold_bowls selling_price percentage_gain = 1400 / 103 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_theorem_l1851_185193


namespace NUMINAMATH_CALUDE_convex_quadrilateral_exists_l1851_185152

-- Define a point in a 2D plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a set of 5 points
def FivePoints := Fin 5 → Point

-- Define the property that no three points are collinear
def NoThreeCollinear (points : FivePoints) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    (points i).x * ((points j).y - (points k).y) +
    (points j).x * ((points k).y - (points i).y) +
    (points k).x * ((points i).y - (points j).y) ≠ 0

-- Define a convex quadrilateral
def ConvexQuadrilateral (a b c d : Point) : Prop :=
  -- This is a simplified definition. In practice, we would need to define
  -- convexity more rigorously.
  true

-- The main theorem
theorem convex_quadrilateral_exists (points : FivePoints) 
  (h : NoThreeCollinear points) : 
  ∃ (i j k l : Fin 5), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧
    ConvexQuadrilateral (points i) (points j) (points k) (points l) :=
by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_exists_l1851_185152


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1851_185143

/-- Given vectors a and b in ℝ², and c = a + k * b, prove that if a is perpendicular to c, then k = -10/3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) (h1 : a = (3, 1)) (h2 : b = (1, 0)) :
  let c := a + k • b
  (a.1 * c.1 + a.2 * c.2 = 0) → k = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1851_185143


namespace NUMINAMATH_CALUDE_jack_buttons_theorem_l1851_185185

/-- The number of buttons Jack needs for all shirts -/
def total_buttons (shirts_per_kid : ℕ) (num_kids : ℕ) (buttons_per_shirt : ℕ) : ℕ :=
  shirts_per_kid * num_kids * buttons_per_shirt

/-- Theorem stating that Jack needs 63 buttons for all shirts -/
theorem jack_buttons_theorem :
  total_buttons 3 3 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_jack_buttons_theorem_l1851_185185


namespace NUMINAMATH_CALUDE_jamie_used_ten_dimes_l1851_185134

/-- Represents the coin distribution in Jamie's payment --/
structure CoinDistribution where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- The coin distribution satisfies the problem constraints --/
def is_valid_distribution (d : CoinDistribution) : Prop :=
  d.pennies + d.nickels + d.dimes = 50 ∧
  d.pennies + 5 * d.nickels + 10 * d.dimes = 240

/-- The unique valid coin distribution has 10 dimes --/
theorem jamie_used_ten_dimes :
  ∃! d : CoinDistribution, is_valid_distribution d ∧ d.dimes = 10 := by
  sorry


end NUMINAMATH_CALUDE_jamie_used_ten_dimes_l1851_185134


namespace NUMINAMATH_CALUDE_shorts_cost_l1851_185171

theorem shorts_cost (num_players : ℕ) (jersey_cost sock_cost total_cost : ℚ) :
  num_players = 16 ∧ 
  jersey_cost = 25 ∧ 
  sock_cost = 6.80 ∧ 
  total_cost = 752 →
  ∃ shorts_cost : ℚ, 
    shorts_cost = 15.20 ∧ 
    num_players * (jersey_cost + shorts_cost + sock_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_shorts_cost_l1851_185171


namespace NUMINAMATH_CALUDE_system_solution_unique_l1851_185159

theorem system_solution_unique : 
  ∃! (x y : ℝ), (2 * x + y = 4) ∧ (x - y = -1) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1851_185159


namespace NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l1851_185104

theorem gcf_seven_eight_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l1851_185104


namespace NUMINAMATH_CALUDE_rectangle_length_l1851_185173

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area_eq : length * width = 300
  perimeter_eq : 2 * (length + width) = 70

/-- The length of the rectangle is 20 meters -/
theorem rectangle_length (r : Rectangle) : r.length = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l1851_185173


namespace NUMINAMATH_CALUDE_distance_between_points_l1851_185146

def point1 : ℝ × ℝ := (-2, 5)
def point2 : ℝ × ℝ := (4, -1)

theorem distance_between_points :
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1851_185146


namespace NUMINAMATH_CALUDE_no_solution_fibonacci_equation_l1851_185157

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem no_solution_fibonacci_equation :
  ¬∃ n : ℕ, n * (fib n) * (fib (n - 1)) = (fib (n + 2) - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_fibonacci_equation_l1851_185157


namespace NUMINAMATH_CALUDE_digit_sum_properties_l1851_185199

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Permutation of digits relation -/
def is_digit_permutation (m k : ℕ) : Prop := sorry

theorem digit_sum_properties (M K : ℕ) (h : is_digit_permutation M K) : 
  (sum_of_digits (2 * M) = sum_of_digits (2 * K)) ∧ 
  (M % 2 = 0 → K % 2 = 0 → sum_of_digits (M / 2) = sum_of_digits (K / 2)) ∧
  (sum_of_digits (5 * M) = sum_of_digits (5 * K)) := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_properties_l1851_185199


namespace NUMINAMATH_CALUDE_triangle_side_length_l1851_185133

theorem triangle_side_length (a b c : ℝ) : 
  a = 3 ∧ b = 7 → 
  (c = 7 ∧ 
   c + a > b ∧ 
   c + b > a ∧ 
   a + b > c) ∧ 
  (c ≠ 3 ∨ c ≠ 10 ∨ c ≠ 12 ∨ 
   ¬(c + a > b ∧ c + b > a ∧ a + b > c)) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1851_185133


namespace NUMINAMATH_CALUDE_constant_term_proof_l1851_185179

theorem constant_term_proof (a k n : ℤ) (x : ℝ) :
  (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n →
  a - n + k = 7 →
  n = -6 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_proof_l1851_185179


namespace NUMINAMATH_CALUDE_min_cost_container_l1851_185131

/-- Represents the cost function for a rectangular container -/
def cost_function (a b : ℝ) : ℝ := 20 * (a * b) + 10 * 2 * (a + b)

/-- Theorem stating the minimum cost for the container -/
theorem min_cost_container :
  ∀ a b : ℝ,
  a > 0 → b > 0 →
  a * b = 4 →
  cost_function a b ≥ 160 :=
by sorry

end NUMINAMATH_CALUDE_min_cost_container_l1851_185131


namespace NUMINAMATH_CALUDE_equation_solutions_l1851_185118

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 = 4 ↔ x = 1 ∨ x = -3) ∧
  (∀ x : ℝ, 3*x^3 + 4 = -20 ↔ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1851_185118


namespace NUMINAMATH_CALUDE_frustum_smaller_base_area_l1851_185189

/-- A frustum with given properties -/
structure Frustum where
  r : ℝ  -- radius of the smaller base
  h : ℝ  -- slant height
  S : ℝ  -- lateral area

/-- The theorem stating the properties of the frustum and its smaller base area -/
theorem frustum_smaller_base_area (f : Frustum) 
  (h1 : f.h = 3)
  (h2 : f.S = 84 * Real.pi)
  (h3 : 2 * Real.pi * (3 * f.r) = 3 * (2 * Real.pi * f.r)) :
  f.r^2 * Real.pi = 49 * Real.pi := by
  sorry

#check frustum_smaller_base_area

end NUMINAMATH_CALUDE_frustum_smaller_base_area_l1851_185189


namespace NUMINAMATH_CALUDE_correct_calculation_l1851_185154

/-- Represents the cost and quantity relationships of items A and B -/
structure ItemRelationship where
  cost_difference : ℝ  -- Cost difference between A and B
  quantity_ratio : ℝ   -- Ratio of quantities purchasable for 480 yuan
  total_items : ℕ      -- Total number of items to be purchased
  max_cost : ℝ         -- Maximum total cost allowed

/-- Calculates the costs of items A and B and the minimum number of B items to purchase -/
def calculate_costs_and_min_b (r : ItemRelationship) : 
  (ℝ × ℝ × ℕ) :=
  -- The actual calculation would go here
  sorry

/-- Theorem stating the correctness of the calculation -/
theorem correct_calculation (r : ItemRelationship) 
  (h1 : r.cost_difference = 4)
  (h2 : r.quantity_ratio = 3/4)
  (h3 : r.total_items = 200)
  (h4 : r.max_cost = 3000) :
  calculate_costs_and_min_b r = (16, 12, 50) :=
sorry

end NUMINAMATH_CALUDE_correct_calculation_l1851_185154


namespace NUMINAMATH_CALUDE_inequality_proof_l1851_185117

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 3) :
  (x - 1) * (y - 1) * (z - 1) ≤ 1/4 * (x*y*z - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1851_185117


namespace NUMINAMATH_CALUDE_value_of_x_l1851_185132

theorem value_of_x : ∃ x : ℕ, x = 320 * 2 * 3 ∧ x = 1920 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1851_185132


namespace NUMINAMATH_CALUDE_greatest_common_divisor_problem_l1851_185145

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k, p.Prime ∧ k > 0 ∧ n = p ^ k

theorem greatest_common_divisor_problem (m : ℕ) 
  (h1 : (Nat.divisors (Nat.gcd 72 m)).card = 3)
  (h2 : is_prime_power m) :
  Nat.gcd 72 m = 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_problem_l1851_185145


namespace NUMINAMATH_CALUDE_exists_right_triangle_with_different_colors_l1851_185110

-- Define the color type
inductive Color
| Red
| Green
| Blue

-- Define a point in the coordinate plane
structure Point where
  x : ℤ
  y : ℤ

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the condition that all three colors are used
axiom all_colors_used : 
  ∃ p1 p2 p3 : Point, coloring p1 ≠ coloring p2 ∧ coloring p2 ≠ coloring p3 ∧ coloring p3 ≠ coloring p1

-- Define a right triangle
def is_right_triangle (p1 p2 p3 : Point) : Prop := sorry

-- Theorem statement
theorem exists_right_triangle_with_different_colors :
  ∃ p1 p2 p3 : Point, 
    is_right_triangle p1 p2 p3 ∧ 
    coloring p1 ≠ coloring p2 ∧ 
    coloring p2 ≠ coloring p3 ∧ 
    coloring p3 ≠ coloring p1 := by sorry

end NUMINAMATH_CALUDE_exists_right_triangle_with_different_colors_l1851_185110


namespace NUMINAMATH_CALUDE_crop_yield_growth_l1851_185124

theorem crop_yield_growth (x : ℝ) : 
  (∀ y : ℝ, y > 0 → (y * (1 + x))^2 = y * (1 + x)^2) →
  300 * (1 + x)^2 = 363 ↔ 
    (∃ (growth_rate : ℝ), 
      growth_rate > 0 ∧
      growth_rate < 1 ∧
      x = growth_rate ∧
      300 * (1 + growth_rate)^2 = 363) :=
by sorry

end NUMINAMATH_CALUDE_crop_yield_growth_l1851_185124


namespace NUMINAMATH_CALUDE_jacket_price_restoration_l1851_185105

theorem jacket_price_restoration : 
  ∀ (original_price : ℝ), original_price > 0 →
  let price_after_first_reduction := original_price * (1 - 0.25)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.30)
  let restoration_factor := original_price / price_after_second_reduction
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |restoration_factor - 1 - 0.9048| < ε :=
by sorry

end NUMINAMATH_CALUDE_jacket_price_restoration_l1851_185105


namespace NUMINAMATH_CALUDE_count_to_200_words_l1851_185115

/-- Represents the number of words used to express a given number in English. --/
def wordsForNumber (n : ℕ) : ℕ :=
  if n ≤ 20 ∨ n = 30 ∨ n = 40 ∨ n = 50 ∨ n = 60 ∨ n = 70 ∨ n = 80 ∨ n = 90 ∨ n = 100 ∨ n = 200 then 1
  else if n ≤ 99 then 2
  else if n ≤ 199 then 3
  else 3

/-- The total number of words used to count from 1 to 200 in English. --/
def totalWordsUpTo200 : ℕ := (Finset.range 200).sum wordsForNumber + wordsForNumber 200

/-- Theorem stating that the total number of words used to count from 1 to 200 in English is 443. --/
theorem count_to_200_words : totalWordsUpTo200 = 443 := by
  sorry

end NUMINAMATH_CALUDE_count_to_200_words_l1851_185115


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1851_185197

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = x^2 + 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1)}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1851_185197


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1851_185165

theorem completing_square_equivalence (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1851_185165


namespace NUMINAMATH_CALUDE_expression_evaluation_l1851_185176

theorem expression_evaluation (x y : ℝ) (hx : x = 3) (hy : y = 2) : 
  3 * x^2 - 4 * y + 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1851_185176


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l1851_185168

theorem abs_inequality_solution_set (x : ℝ) : 
  2 * |x - 1| - 1 < 0 ↔ 1/2 < x ∧ x < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l1851_185168


namespace NUMINAMATH_CALUDE_union_equality_condition_equivalence_condition_l1851_185121

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 3 - 2*a}
def B : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}

-- Statement for part (1)
theorem union_equality_condition (a : ℝ) :
  A a ∪ B = B ↔ a ∈ Set.Ici (-1/2) :=
sorry

-- Statement for part (2)
theorem equivalence_condition (a : ℝ) :
  (∀ x, x ∈ B ↔ x ∈ A a) ↔ a ∈ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_union_equality_condition_equivalence_condition_l1851_185121


namespace NUMINAMATH_CALUDE_correct_calculation_result_l1851_185196

theorem correct_calculation_result (x : ℤ) (h : x + 63 = 8) : x * 36 = -1980 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_result_l1851_185196


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1851_185180

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (∀ θ : ℝ, θ = 160 ∧ θ * n = 180 * (n - 2)) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1851_185180


namespace NUMINAMATH_CALUDE_total_egg_rolls_l1851_185137

-- Define the number of egg rolls rolled by each person
def omar_rolls : ℕ := 219
def karen_rolls : ℕ := 229
def lily_rolls : ℕ := 275

-- Theorem to prove
theorem total_egg_rolls : omar_rolls + karen_rolls + lily_rolls = 723 := by
  sorry

end NUMINAMATH_CALUDE_total_egg_rolls_l1851_185137


namespace NUMINAMATH_CALUDE_unique_a_value_l1851_185125

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem unique_a_value : ∃! a : ℝ, (9 ∈ (A a ∩ B a)) ∧ ({9} = A a ∩ B a) := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l1851_185125


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1851_185175

theorem sum_of_coefficients (a b c d e : ℚ) :
  (∀ x, 1000 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 92 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1851_185175


namespace NUMINAMATH_CALUDE_ellipse_specific_constants_l1851_185138

/-- Definition of an ellipse passing through a point -/
def ellipse_passes_through (f1 f2 p : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2)
  let d2 := Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2)
  let c := Real.sqrt ((f2.1 - f1.1)^2 + (f2.2 - f1.2)^2) / 2
  d1 + d2 = 2 * Real.sqrt (c^2 + (d1 + d2)^2 / 4)

/-- The standard form equation of an ellipse -/
def ellipse_equation (x y h k a b : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- Theorem: Ellipse with given foci and point has specific equation constants -/
theorem ellipse_specific_constants :
  let f1 : ℝ × ℝ := (8, 1)
  let f2 : ℝ × ℝ := (8, 9)
  let p : ℝ × ℝ := (17, 5)
  ellipse_passes_through f1 f2 p →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), ellipse_equation x y 8 5 9 (Real.sqrt 97) ↔
      ellipse_equation x y 8 5 a b) :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_specific_constants_l1851_185138


namespace NUMINAMATH_CALUDE_division_with_remainder_l1851_185181

theorem division_with_remainder (A : ℕ) : 
  (A / 9 = 6) ∧ (A % 9 = 5) → A = 59 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l1851_185181


namespace NUMINAMATH_CALUDE_robin_candy_problem_l1851_185126

theorem robin_candy_problem (initial_candy : ℕ) (sister_candy : ℕ) (final_candy : ℕ) 
  (h1 : initial_candy = 23)
  (h2 : sister_candy = 21)
  (h3 : final_candy = 37) :
  initial_candy - (final_candy - sister_candy) = 7 := by
  sorry

end NUMINAMATH_CALUDE_robin_candy_problem_l1851_185126


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1851_185161

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (1, -2)
  parallel a b → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1851_185161


namespace NUMINAMATH_CALUDE_five_numbers_product_invariant_l1851_185155

theorem five_numbers_product_invariant :
  ∃ (a b c d e : ℝ),
    a * b * c * d * e ≠ 0 ∧
    (a - 1) * (b - 1) * (c - 1) * (d - 1) * (e - 1) = a * b * c * d * e :=
by sorry

end NUMINAMATH_CALUDE_five_numbers_product_invariant_l1851_185155


namespace NUMINAMATH_CALUDE_lines_intersect_at_same_point_l1851_185135

/-- Three lines intersect at the same point -/
theorem lines_intersect_at_same_point (m : ℝ) :
  ∃ (x y : ℝ), 
    (y = 3 * x + 5) ∧ 
    (y = -4 * x + m) ∧ 
    (y = 2 * x + (m + 30) / 7) := by
  sorry

end NUMINAMATH_CALUDE_lines_intersect_at_same_point_l1851_185135


namespace NUMINAMATH_CALUDE_max_measurements_exact_measurements_l1851_185108

/-- The number of ways to measure a weight P using weights up to 2^n -/
def K (n : ℕ) (P : ℕ) : ℕ := sorry

/-- The maximum number of ways any weight can be measured using weights up to 2^n -/
def K_max (n : ℕ) : ℕ := sorry

/-- The set of available weights -/
def weights : Set ℕ := {w : ℕ | ∃ k : ℕ, w = 2^k ∧ k ≤ 9}

theorem max_measurements (P : ℕ) : K 9 P ≤ 89 := sorry

theorem exact_measurements : K 9 171 = 89 := sorry

#check max_measurements
#check exact_measurements

end NUMINAMATH_CALUDE_max_measurements_exact_measurements_l1851_185108


namespace NUMINAMATH_CALUDE_black_area_after_three_changes_l1851_185141

def black_area_fraction (n : ℕ) : ℚ :=
  (1 / 2) ^ n

theorem black_area_after_three_changes :
  black_area_fraction 3 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_black_area_after_three_changes_l1851_185141


namespace NUMINAMATH_CALUDE_task_force_count_l1851_185102

theorem task_force_count (total : ℕ) (executives : ℕ) (task_force_size : ℕ) (min_executives : ℕ) :
  total = 12 →
  executives = 5 →
  task_force_size = 5 →
  min_executives = 2 →
  (Nat.choose total task_force_size -
   (Nat.choose (total - executives) task_force_size +
    Nat.choose executives 1 * Nat.choose (total - executives) (task_force_size - 1))) = 596 := by
  sorry

end NUMINAMATH_CALUDE_task_force_count_l1851_185102


namespace NUMINAMATH_CALUDE_equation_equivalence_l1851_185116

theorem equation_equivalence (a b x y : ℝ) :
  a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1) ↔ (a*x - 1)*(a^2*y - 1) = a^5*b^5 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1851_185116

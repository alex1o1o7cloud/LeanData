import Mathlib

namespace NUMINAMATH_CALUDE_square_sum_primes_l2611_261149

theorem square_sum_primes (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r)
  (h1 : ∃ a : ℕ, pq + 1 = a^2)
  (h2 : ∃ b : ℕ, pr + 1 = b^2)
  (h3 : ∃ c : ℕ, qr - p = c^2) :
  ∃ d : ℕ, p + 2*q*r + 2 = d^2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_primes_l2611_261149


namespace NUMINAMATH_CALUDE_intersection_M_N_l2611_261155

def M : Set ℤ := {1, 2, 3, 4, 5, 6}

def N : Set ℤ := {x | -2 < x ∧ x < 5}

theorem intersection_M_N : M ∩ N = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2611_261155


namespace NUMINAMATH_CALUDE_booth_active_days_l2611_261180

/-- Represents the carnival snack booth scenario -/
def carnival_booth (days : ℕ) : Prop :=
  let popcorn_revenue := 50
  let cotton_candy_revenue := 3 * popcorn_revenue
  let daily_revenue := popcorn_revenue + cotton_candy_revenue
  let daily_rent := 30
  let ingredient_cost := 75
  let total_revenue := days * daily_revenue
  let total_rent := days * daily_rent
  let profit := total_revenue - total_rent - ingredient_cost
  profit = 895

/-- Theorem stating that the booth was active for 5 days -/
theorem booth_active_days : ∃ (d : ℕ), carnival_booth d ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_booth_active_days_l2611_261180


namespace NUMINAMATH_CALUDE_circle_area_equality_l2611_261197

theorem circle_area_equality (r₁ r₂ r : ℝ) (h₁ : r₁ = 24) (h₂ : r₂ = 35) :
  (π * r₂^2 - π * r₁^2 = π * r^2) → r = Real.sqrt 649 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equality_l2611_261197


namespace NUMINAMATH_CALUDE_ladder_movement_l2611_261101

theorem ladder_movement (ladder_length : ℝ) (initial_distance : ℝ) (slide_down : ℝ) : 
  ladder_length = 25 →
  initial_distance = 7 →
  slide_down = 4 →
  ∃ (final_distance : ℝ),
    final_distance > initial_distance ∧
    final_distance ^ 2 + (ladder_length - slide_down) ^ 2 = ladder_length ^ 2 ∧
    final_distance - initial_distance = 8 :=
by sorry

end NUMINAMATH_CALUDE_ladder_movement_l2611_261101


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l2611_261147

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + 5

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 12 * x^2 + 2 * a * x + b

theorem tangent_line_and_extrema 
  (a b : ℝ) 
  (h1 : f' a b 1 = -12)  -- Slope of tangent line at x=1 is -12
  (h2 : f a b 1 = -12)   -- Point (1, -12) lies on the graph of f(x)
  : 
  (a = -3 ∧ b = -18) ∧   -- Part 1: Coefficients a and b
  (∀ x ∈ Set.Icc (-3) 1, f (-3) (-18) x ≤ 16) ∧  -- Part 2: Maximum value
  (∀ x ∈ Set.Icc (-3) 1, f (-3) (-18) x ≥ -76)   -- Part 2: Minimum value
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l2611_261147


namespace NUMINAMATH_CALUDE_smallest_m_theorem_l2611_261153

/-- The smallest positive value of m for which the equation 12x^2 - mx - 360 = 0 has integral solutions -/
def smallest_m : ℕ := 12

/-- The equation 12x^2 - mx - 360 = 0 has integral solutions -/
def has_integral_solutions (m : ℤ) : Prop :=
  ∃ x : ℤ, 12 * x^2 - m * x - 360 = 0

/-- The theorem stating that the smallest positive m for which the equation has integral solutions is 12 -/
theorem smallest_m_theorem : 
  (∀ m : ℕ, m < smallest_m → ¬(has_integral_solutions m)) ∧ 
  (has_integral_solutions smallest_m) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_theorem_l2611_261153


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2611_261175

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∀ x y : ℝ, (m + 2) * x + m * y + 1 = 0 ∧ 
               (m - 1) * x + (m - 4) * y + 2 = 0 → 
               ((m + 2) * (m - 1) + m * (m - 4) = 0)) → 
  m = 2 ∨ m = -1/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2611_261175


namespace NUMINAMATH_CALUDE_right_triangle_c_squared_l2611_261191

theorem right_triangle_c_squared (a b c : ℝ) : 
  a = 9 → b = 12 → (c^2 = a^2 + b^2 ∨ b^2 = a^2 + c^2) → c^2 = 225 ∨ c^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_c_squared_l2611_261191


namespace NUMINAMATH_CALUDE_integral_equals_six_implies_b_equals_e_to_four_l2611_261164

theorem integral_equals_six_implies_b_equals_e_to_four (b : ℝ) :
  (∫ (x : ℝ) in e..b, 2 / x) = 6 → b = Real.exp 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_six_implies_b_equals_e_to_four_l2611_261164


namespace NUMINAMATH_CALUDE_volume_of_square_cross_section_cylinder_l2611_261163

/-- A cylinder with height 40 cm and a square cross-section when cut along the diameter of the base -/
structure SquareCrossSectionCylinder where
  height : ℝ
  height_eq : height = 40
  square_cross_section : Bool

/-- The volume of the cylinder in cubic decimeters -/
def cylinder_volume (c : SquareCrossSectionCylinder) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specified cylinder is 502.4 cubic decimeters -/
theorem volume_of_square_cross_section_cylinder :
  ∀ (c : SquareCrossSectionCylinder), cylinder_volume c = 502.4 :=
by sorry

end NUMINAMATH_CALUDE_volume_of_square_cross_section_cylinder_l2611_261163


namespace NUMINAMATH_CALUDE_lcm_problem_l2611_261184

-- Define the polynomials
def f (x : ℤ) : ℤ := 300 * x^4 + 425 * x^3 + 138 * x^2 - 17 * x - 6
def g (x : ℤ) : ℤ := 225 * x^4 - 109 * x^3 + 4

-- Define the LCM function for integers
def lcm_int (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

-- Define the LCM function for polynomials
noncomputable def lcm_poly (f g : ℤ → ℤ) : ℤ → ℤ := sorry

theorem lcm_problem :
  (lcm_int 4199 4641 5083 = 98141269893) ∧
  (lcm_poly f g = λ x => (225 * x^4 - 109 * x^3 + 4) * (4 * x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2611_261184


namespace NUMINAMATH_CALUDE_womens_average_age_l2611_261136

theorem womens_average_age (n : ℕ) (initial_avg : ℝ) :
  n = 8 ∧
  initial_avg > 0 ∧
  (n * initial_avg + 60) / n = initial_avg + 2 →
  60 / 2 = 30 := by
sorry

end NUMINAMATH_CALUDE_womens_average_age_l2611_261136


namespace NUMINAMATH_CALUDE_dress_discount_problem_l2611_261192

theorem dress_discount_problem (P D : ℝ) : 
  P * (1 - D) * 1.25 = 71.4 →
  P - 71.4 = 5.25 →
  D = 0.255 := by
  sorry

end NUMINAMATH_CALUDE_dress_discount_problem_l2611_261192


namespace NUMINAMATH_CALUDE_product_equals_half_l2611_261168

/-- Given that a * b * c * d = (√((a + 2) * (b + 3))) / (c + 1) * sin(d) for any a, b, c, and d,
    prove that 6 * 15 * 11 * 30 = 0.5 -/
theorem product_equals_half :
  (∀ a b c d : ℝ, a * b * c * d = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1) * Real.sin d) →
  6 * 15 * 11 * 30 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_half_l2611_261168


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2611_261137

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 2 * x^2 + 6 * x - 18) = 
  x^3 + 2 * x^2 + 3 * x + 13 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2611_261137


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l2611_261156

def product : ℕ := 95 * 97 * 99 * 101

theorem distinct_prime_factors_count :
  (Nat.factors product).toFinset.card = 6 :=
sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l2611_261156


namespace NUMINAMATH_CALUDE_division_power_eq_inv_pow_l2611_261108

/-- Division power of a rational number -/
def division_power (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then a
  else a / (division_power a (n-1))

/-- Theorem: Division power equals inverse raised to power (n-2) -/
theorem division_power_eq_inv_pow (a : ℚ) (n : ℕ) (h1 : a ≠ 0) (h2 : n ≥ 2) :
  division_power a n = (a⁻¹) ^ (n - 2) :=
by sorry

end NUMINAMATH_CALUDE_division_power_eq_inv_pow_l2611_261108


namespace NUMINAMATH_CALUDE_sugar_recipe_reduction_l2611_261172

theorem sugar_recipe_reduction : 
  let original_recipe : ℚ := 5 + 3/4
  let reduced_recipe : ℚ := (1/3) * original_recipe
  reduced_recipe = 1 + 11/12 := by
sorry

end NUMINAMATH_CALUDE_sugar_recipe_reduction_l2611_261172


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l2611_261169

theorem min_value_and_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 2*b + 3*c = 8) →
  (∃ (m : ℝ), m = 1/a + 2/b + 3/c ∧ m ≥ 4.5 ∧ ∀ (x : ℝ), x = 1/a + 2/b + 3/c → x ≥ m) ∧
  (∃ (x : ℝ), (x = a + 1/b ∨ x = b + 1/c ∨ x = c + 1/a) ∧ x ≥ 2) :=
by sorry


end NUMINAMATH_CALUDE_min_value_and_inequality_l2611_261169


namespace NUMINAMATH_CALUDE_smallest_degree_is_five_l2611_261186

/-- The smallest degree of a polynomial p(x) such that (3x^5 - 5x^3 + 4x - 2) / p(x) has a horizontal asymptote -/
def smallest_degree_with_horizontal_asymptote : ℕ := by
  sorry

/-- The numerator of the rational function -/
def numerator (x : ℝ) : ℝ := 3*x^5 - 5*x^3 + 4*x - 2

/-- The rational function has a horizontal asymptote -/
def has_horizontal_asymptote (p : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ), ∀ ε > 0, ∃ M, ∀ x, |x| > M → |numerator x / p x - L| < ε

theorem smallest_degree_is_five :
  smallest_degree_with_horizontal_asymptote = 5 ∧
  ∃ (p : ℝ → ℝ), (∀ x, ∃ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ), p x = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) ∧
    has_horizontal_asymptote p ∧
    ∀ (q : ℝ → ℝ), (∀ x, ∃ (b₀ b₁ b₂ b₃ b₄ : ℝ), q x = b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
      ¬(has_horizontal_asymptote q) := by
  sorry

end NUMINAMATH_CALUDE_smallest_degree_is_five_l2611_261186


namespace NUMINAMATH_CALUDE_excellent_credit_prob_expectation_X_l2611_261110

/-- Credit score distribution --/
def credit_distribution : Finset (ℕ × ℕ) := {(150, 25), (120, 60), (100, 65), (80, 35), (0, 15)}

/-- Total population --/
def total_population : ℕ := 200

/-- Voucher allocation function --/
def voucher (score : ℕ) : ℕ :=
  if score > 150 then 100
  else if score > 100 then 50
  else 0

/-- Probability of selecting 2 people with excellent credit --/
theorem excellent_credit_prob : 
  (Nat.choose 25 2 : ℚ) / (Nat.choose total_population 2) = 3 / 199 := by sorry

/-- Distribution of total vouchers X for 2 randomly selected people --/
def voucher_distribution : Finset (ℕ × ℚ) := {(0, 1/16), (50, 5/16), (100, 29/64), (150, 5/32), (200, 1/64)}

/-- Expectation of X --/
theorem expectation_X : 
  (voucher_distribution.sum (λ (x, p) => x * p)) = 175 / 2 := by sorry

end NUMINAMATH_CALUDE_excellent_credit_prob_expectation_X_l2611_261110


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l2611_261194

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | Real.log (x - 1) ≤ 0}

-- State the theorem
theorem intersection_equals_open_interval :
  A ∩ B = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l2611_261194


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_equals_area_l2611_261151

theorem right_triangle_hypotenuse_equals_area 
  (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : m ≠ n) : 
  let x : ℝ := (m^2 + n^2) / (m * n * (m^2 - n^2))
  let leg1 : ℝ := (m^2 - n^2) * x
  let leg2 : ℝ := 2 * m * n * x
  let hypotenuse : ℝ := (m^2 + n^2) * x
  let area : ℝ := (1/2) * leg1 * leg2
  hypotenuse = area :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_equals_area_l2611_261151


namespace NUMINAMATH_CALUDE_sin_shift_l2611_261139

open Real

theorem sin_shift (x : ℝ) :
  sin (3 * (x - π / 12)) = sin (3 * x - π / 4) := by sorry

end NUMINAMATH_CALUDE_sin_shift_l2611_261139


namespace NUMINAMATH_CALUDE_max_n_for_specific_sequence_l2611_261167

/-- Represents an arithmetic sequence with first term a₁, nth term aₙ, and common difference d. -/
structure ArithmeticSequence where
  a₁ : ℤ
  aₙ : ℤ
  d : ℕ+
  n : ℕ
  h_arithmetic : aₙ = a₁ + (n - 1) * d

/-- The maximum value of n for a specific arithmetic sequence. -/
def maxN (seq : ArithmeticSequence) : ℕ :=
  seq.n

/-- Theorem stating the maximum value of n for the given arithmetic sequence. -/
theorem max_n_for_specific_sequence :
  ∀ seq : ArithmeticSequence,
    seq.a₁ = -6 →
    seq.aₙ = 0 →
    seq.n ≥ 3 →
    maxN seq ≤ 7 ∧ ∃ seq' : ArithmeticSequence, seq'.a₁ = -6 ∧ seq'.aₙ = 0 ∧ seq'.n ≥ 3 ∧ maxN seq' = 7 :=
sorry

end NUMINAMATH_CALUDE_max_n_for_specific_sequence_l2611_261167


namespace NUMINAMATH_CALUDE_system_solutions_l2611_261162

theorem system_solutions :
  ∀ x y : ℝ,
  (y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y) ↔
  ((x = 0 ∧ y = 0) ∨ 
   (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2) ∨ 
   (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2611_261162


namespace NUMINAMATH_CALUDE_ellipse_properties_l2611_261122

-- Define the ellipse C
def Ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the condition for points A and B
def SlopeCondition (xA yA xB yB : ℝ) : Prop :=
  xA ≠ 0 ∧ xB ≠ 0 ∧ (yA / xA) * (yB / xB) = -3/2

theorem ellipse_properties :
  -- Given conditions
  let D := (0, 2)
  let F := (c, 0)
  let E := (4*c/3, -2/3)
  -- The ellipse passes through D and E
  Ellipse D.1 D.2 ∧ Ellipse E.1 E.2 →
  -- |DF| = 3|EF|
  (D.1 - F.1)^2 + (D.2 - F.2)^2 = 9 * ((E.1 - F.1)^2 + (E.2 - F.2)^2) →
  -- Theorem statements
  (∀ x y, Ellipse x y ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∀ xA yA xB yB,
    Ellipse xA yA → Ellipse xB yB → SlopeCondition xA yA xB yB →
    -1 ≤ (xA * xB + yA * yB) ∧ (xA * xB + yA * yB) ≤ 1 ∧
    (xA * xB + yA * yB) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2611_261122


namespace NUMINAMATH_CALUDE_bobsQuestionsRatio_l2611_261127

/-- Represents the number of questions created in each hour -/
structure HourlyQuestions where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the conditions of the problem -/
def bobsQuestions : HourlyQuestions → Prop
  | ⟨first, second, third⟩ =>
    first = 13 ∧
    third = 2 * second ∧
    first + second + third = 91

/-- The theorem to be proved -/
theorem bobsQuestionsRatio (q : HourlyQuestions) :
  bobsQuestions q → q.second / q.first = 2 := by
  sorry

end NUMINAMATH_CALUDE_bobsQuestionsRatio_l2611_261127


namespace NUMINAMATH_CALUDE_bus_children_count_l2611_261142

theorem bus_children_count (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 26 → additional = 38 → total = initial + additional → total = 64 := by
sorry

end NUMINAMATH_CALUDE_bus_children_count_l2611_261142


namespace NUMINAMATH_CALUDE_triangle_cosine_relation_l2611_261132

/-- Given a triangle ABC with angles A, B, C and sides a, b, c opposite to these angles respectively.
    If 2sin²A + 2sin²B = 2sin²(A+B) + 3sinAsinB, then cos C = 3/4. -/
theorem triangle_cosine_relation (A B C a b c : Real) : 
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  2 * (Real.sin A)^2 + 2 * (Real.sin B)^2 = 2 * (Real.sin (A + B))^2 + 3 * Real.sin A * Real.sin B →
  Real.cos C = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_relation_l2611_261132


namespace NUMINAMATH_CALUDE_pencil_distribution_l2611_261182

theorem pencil_distribution (total_pencils : ℕ) (num_students : ℕ) 
  (h1 : total_pencils = 125) (h2 : num_students = 25) :
  total_pencils / num_students = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2611_261182


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l2611_261107

theorem negation_of_existence_proposition :
  (¬∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l2611_261107


namespace NUMINAMATH_CALUDE_student_marks_average_l2611_261148

/-- Given a student's marks in mathematics, physics, and chemistry, 
    where the total marks in mathematics and physics is 50, 
    and the chemistry score is 20 marks more than physics, 
    prove that the average marks in mathematics and chemistry is 35. -/
theorem student_marks_average (m p c : ℕ) : 
  m + p = 50 → c = p + 20 → (m + c) / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_average_l2611_261148


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2611_261124

theorem inequality_solution_set : 
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2611_261124


namespace NUMINAMATH_CALUDE_degree_of_polynomial_l2611_261183

-- Define the polynomial
def p (a b : ℝ) : ℝ := 3 * a^2 - a * b^2 + 2 * a^2 - 3^4

-- Theorem statement
theorem degree_of_polynomial :
  ∃ (n : ℕ), n = 3 ∧ 
  (∀ (m : ℕ), (∃ (a b : ℝ), p a b ≠ 0 ∧ 
    (∀ (c d : ℝ), a^m * b^(n-m) = c^m * d^(n-m) → p a b = p c d)) →
  (∀ (k : ℕ), k > n → 
    (∀ (a b : ℝ), ∃ (c d : ℝ), a^k * b^(n-k) = c^k * d^(n-k) ∧ p a b = p c d))) :=
sorry

end NUMINAMATH_CALUDE_degree_of_polynomial_l2611_261183


namespace NUMINAMATH_CALUDE_solve_for_m_l2611_261103

theorem solve_for_m : ∃ m : ℚ, (10 * (1/2 : ℚ) + m = 2) ∧ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l2611_261103


namespace NUMINAMATH_CALUDE_speed_difference_calc_l2611_261114

/-- Calculates the speed difference between return and outbound trips --/
theorem speed_difference_calc (outbound_time outbound_speed return_time : ℝ) 
  (h1 : outbound_time = 6)
  (h2 : outbound_speed = 60)
  (h3 : return_time = 5)
  (h4 : outbound_time * outbound_speed = return_time * (outbound_speed + speed_diff)) :
  speed_diff = 12 := by
  sorry

#check speed_difference_calc

end NUMINAMATH_CALUDE_speed_difference_calc_l2611_261114


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2611_261174

theorem polynomial_factorization (x a : ℝ) : 
  x^3 - 3*x^2 + (a+2)*x - 2*a = (x^2 - x + a)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2611_261174


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2611_261135

-- Define the universal set I
def I : Set (ℝ × ℝ) := Set.univ

-- Define set A
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 - 3 = p.1 - 2 ∧ p.1 ≠ 2}

-- Define set B
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 1}

-- State the theorem
theorem complement_A_intersect_B : (I \ A) ∩ B = {(2, 3)} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2611_261135


namespace NUMINAMATH_CALUDE_x_value_l2611_261102

theorem x_value : ∃ x : ℝ, x = 80 * (1 + 0.12) ∧ x = 89.6 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2611_261102


namespace NUMINAMATH_CALUDE_loan_repayment_amount_l2611_261177

/-- The amount to be paid back after applying compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Theorem stating that the amount to be paid back is $168.54 -/
theorem loan_repayment_amount : 
  let principal : ℝ := 150
  let rate : ℝ := 0.06
  let time : ℕ := 2
  abs (compound_interest principal rate time - 168.54) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_loan_repayment_amount_l2611_261177


namespace NUMINAMATH_CALUDE_no_valid_labeling_exists_l2611_261159

/-- Represents a labeling of a 45-gon with digits 0-9 -/
def Labeling := Fin 45 → Fin 10

/-- Checks if a labeling is valid according to the problem conditions -/
def is_valid_labeling (l : Labeling) : Prop :=
  ∀ i j : Fin 10, i ≠ j →
    ∃! k : Fin 45, (l k = i ∧ l (k + 1) = j) ∨ (l k = j ∧ l (k + 1) = i)

/-- The main theorem stating that no valid labeling exists -/
theorem no_valid_labeling_exists : ¬∃ l : Labeling, is_valid_labeling l := by
  sorry

end NUMINAMATH_CALUDE_no_valid_labeling_exists_l2611_261159


namespace NUMINAMATH_CALUDE_x_squared_value_l2611_261134

theorem x_squared_value (x : ℝ) (hx : x > 0) (h : Real.sin (Real.arctan x) = 1 / x) : 
  x^2 = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_x_squared_value_l2611_261134


namespace NUMINAMATH_CALUDE_complete_graph_4_vertices_6_edges_l2611_261160

/-- The number of edges in a complete graph with n vertices -/
def complete_graph_edges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A complete graph with 4 vertices has 6 edges -/
theorem complete_graph_4_vertices_6_edges : 
  complete_graph_edges 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_complete_graph_4_vertices_6_edges_l2611_261160


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l2611_261185

theorem min_value_of_fraction (x y : ℝ) 
  (hx : -3 ≤ x ∧ x ≤ 1) 
  (hy : -1 ≤ y ∧ y ≤ 3) 
  (hx_nonzero : x ≠ 0) : 
  (x + y) / x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l2611_261185


namespace NUMINAMATH_CALUDE_expected_heads_equals_55_l2611_261198

/-- The number of coins -/
def num_coins : ℕ := 80

/-- The probability of a coin landing heads on a single flip -/
def p_heads : ℚ := 1/2

/-- The probability of a coin being eligible for a second flip -/
def p_second_flip : ℚ := 1/2

/-- The probability of a coin being eligible for a third flip -/
def p_third_flip : ℚ := 1/2

/-- The expected number of heads after all flips -/
def expected_heads : ℚ := num_coins * (p_heads + p_heads * (1 - p_heads) * p_second_flip + p_heads * (1 - p_heads) * p_second_flip * (1 - p_heads) * p_third_flip)

theorem expected_heads_equals_55 : expected_heads = 55 := by
  sorry

end NUMINAMATH_CALUDE_expected_heads_equals_55_l2611_261198


namespace NUMINAMATH_CALUDE_margin_calculation_l2611_261119

-- Define the sheet dimensions and side margin
def sheet_width : ℝ := 20
def sheet_length : ℝ := 30
def side_margin : ℝ := 2

-- Define the percentage of the page used for typing
def typing_percentage : ℝ := 0.64

-- Define the function to calculate the typing area
def typing_area (top_bottom_margin : ℝ) : ℝ :=
  (sheet_width - 2 * side_margin) * (sheet_length - 2 * top_bottom_margin)

-- Define the theorem
theorem margin_calculation :
  ∃ (top_bottom_margin : ℝ),
    typing_area top_bottom_margin = typing_percentage * sheet_width * sheet_length ∧
    top_bottom_margin = 3 := by
  sorry

end NUMINAMATH_CALUDE_margin_calculation_l2611_261119


namespace NUMINAMATH_CALUDE_cost_of_nuts_l2611_261128

/-- Given Alyssa's purchases and refund, calculate the cost of the pack of nuts -/
theorem cost_of_nuts (grapes_cost refund_cherries total_spent : ℚ) 
  (h1 : grapes_cost = 12.08)
  (h2 : refund_cherries = 9.85)
  (h3 : total_spent = 26.35) :
  grapes_cost - refund_cherries + (total_spent - (grapes_cost - refund_cherries)) = 24.12 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_nuts_l2611_261128


namespace NUMINAMATH_CALUDE_min_value_theorem_l2611_261131

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2)^2) + (y^2 / (x - 2)^2) ≥ 10 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 2 ∧ y₀ > 2 ∧ (x₀^2 / (y₀ - 2)^2) + (y₀^2 / (x₀ - 2)^2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2611_261131


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l2611_261181

theorem absolute_value_equation_product (x : ℝ) :
  (∀ x, |x - 5| - 4 = 3 → x = 12 ∨ x = -2) ∧
  (∃ x₁ x₂, |x₁ - 5| - 4 = 3 ∧ |x₂ - 5| - 4 = 3 ∧ x₁ * x₂ = -24) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l2611_261181


namespace NUMINAMATH_CALUDE_booster_club_tickets_l2611_261112

/-- Represents the ticket information for the Booster Club trip --/
structure TicketInfo where
  num_nine_dollar : Nat
  total_cost : Nat
  cost_seven : Nat
  cost_nine : Nat

/-- Calculates the total number of tickets bought given the ticket information --/
def total_tickets (info : TicketInfo) : Nat :=
  info.num_nine_dollar + (info.total_cost - info.num_nine_dollar * info.cost_nine) / info.cost_seven

/-- Theorem stating that given the specific ticket information, the total number of tickets is 29 --/
theorem booster_club_tickets :
  let info : TicketInfo := {
    num_nine_dollar := 11,
    total_cost := 225,
    cost_seven := 7,
    cost_nine := 9
  }
  total_tickets info = 29 := by sorry

end NUMINAMATH_CALUDE_booster_club_tickets_l2611_261112


namespace NUMINAMATH_CALUDE_intersection_M_N_l2611_261143

-- Define the sets M and N
def M : Set ℝ := Set.univ
def N : Set ℝ := {x | 2 * x - x^2 > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2611_261143


namespace NUMINAMATH_CALUDE_gcd_and_sum_divisibility_l2611_261113

theorem gcd_and_sum_divisibility : 
  (Nat.gcd 42558 29791 = 3) ∧ 
  ¬(72349 % 3 = 0) := by sorry

end NUMINAMATH_CALUDE_gcd_and_sum_divisibility_l2611_261113


namespace NUMINAMATH_CALUDE_f_2005_of_2006_eq_145_l2611_261130

/-- Sum of squares of digits of a positive integer -/
def f (n : ℕ+) : ℕ := sorry

/-- Recursive application of f, k times -/
def f_k (k : ℕ) (n : ℕ+) : ℕ :=
  match k with
  | 0 => n.val
  | k + 1 => f (⟨f_k k n, sorry⟩)

/-- The main theorem to prove -/
theorem f_2005_of_2006_eq_145 : f_k 2005 ⟨2006, sorry⟩ = 145 := by sorry

end NUMINAMATH_CALUDE_f_2005_of_2006_eq_145_l2611_261130


namespace NUMINAMATH_CALUDE_lower_variance_more_stable_student_B_more_stable_l2611_261129

/-- Represents a student's throwing performance -/
structure StudentPerformance where
  name : String
  variance : ℝ

/-- Defines the concept of stability in performance -/
def moreStable (a b : StudentPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two students' performances, the one with lower variance is more stable -/
theorem lower_variance_more_stable (a b : StudentPerformance) :
    moreStable a b ↔ a.variance < b.variance :=
  by sorry

/-- The specific problem instance -/
def studentA : StudentPerformance :=
  { name := "A", variance := 0.2 }

def studentB : StudentPerformance :=
  { name := "B", variance := 0.09 }

/-- Theorem: Student B has more stable performance than Student A -/
theorem student_B_more_stable : moreStable studentB studentA :=
  by sorry

end NUMINAMATH_CALUDE_lower_variance_more_stable_student_B_more_stable_l2611_261129


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2611_261126

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 6 = 5) ∧ 
  (n % 8 = 7) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 6 = 5 ∧ m % 8 = 7 → m ≥ n) ∧
  n = 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2611_261126


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l2611_261118

theorem cube_edge_ratio (a b : ℝ) (h : a ^ 3 / b ^ 3 = 8 / 1) : a / b = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l2611_261118


namespace NUMINAMATH_CALUDE_proportional_segments_l2611_261173

theorem proportional_segments (a : ℝ) :
  (a > 0) → ((a / 2) = (6 / (a + 1))) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_proportional_segments_l2611_261173


namespace NUMINAMATH_CALUDE_allison_wins_prob_l2611_261125

/-- Represents a 6-sided cube with specified face values -/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- Allison's cube configuration -/
def allison_cube : Cube :=
  { faces := λ _ => 7 }

/-- Brian's cube configuration -/
def brian_cube : Cube :=
  { faces := λ i => i.val + 1 }

/-- Noah's cube configuration -/
def noah_cube : Cube :=
  { faces := λ i => if i.val < 2 then 3 else 5 }

/-- The probability of rolling a specific value or less on a given cube -/
def prob_roll_le (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i ≤ n) (Finset.univ : Finset (Fin 6))).card / 6

/-- The main theorem stating the probability of Allison's roll being greater than both Brian's and Noah's -/
theorem allison_wins_prob : 
  prob_roll_le brian_cube 6 * prob_roll_le noah_cube 6 = 1 := by
  sorry


end NUMINAMATH_CALUDE_allison_wins_prob_l2611_261125


namespace NUMINAMATH_CALUDE_no_fixed_point_implies_no_double_fixed_point_no_intersection_implies_no_double_intersection_l2611_261144

-- Part (a)
theorem no_fixed_point_implies_no_double_fixed_point
  (f : ℝ → ℝ) (hf : Continuous f) (h : ∀ x, f x ≠ x) :
  ∀ x, f (f x) ≠ x :=
sorry

-- Part (b)
theorem no_intersection_implies_no_double_intersection
  (f g : ℝ → ℝ) (hf : Continuous f) (hg : Continuous g)
  (h_comm : ∀ x, f (g x) = g (f x)) (h_neq : ∀ x, f x ≠ g x) :
  ∀ x, f (f x) ≠ g (g x) :=
sorry

end NUMINAMATH_CALUDE_no_fixed_point_implies_no_double_fixed_point_no_intersection_implies_no_double_intersection_l2611_261144


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_squared_inequality_l2611_261161

theorem not_sufficient_not_necessary_squared_inequality (a b : ℝ) :
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧ (∃ u v : ℝ, u^2 > v^2 ∧ u ≤ v) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_squared_inequality_l2611_261161


namespace NUMINAMATH_CALUDE_max_value_problem_l2611_261116

theorem max_value_problem (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 3) 
  (h5 : x ≥ y) (h6 : y ≥ z) : 
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 2916/729 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l2611_261116


namespace NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_l2611_261189

-- First expression
theorem simplify_expression1 (a b : ℝ) : (a - 2*b) - (2*b - 5*a) = 6*a - 4*b := by sorry

-- Second expression
theorem simplify_expression2 (m n : ℝ) : -m^2*n + (4*m*n^2 - 3*m*n) - 2*(m*n^2 - 3*m^2*n) = 5*m^2*n + 2*m*n^2 - 3*m*n := by sorry

end NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_l2611_261189


namespace NUMINAMATH_CALUDE_valentine_biscuits_l2611_261100

theorem valentine_biscuits (total_biscuits : ℕ) (num_dogs : ℕ) (biscuits_per_dog : ℕ) :
  total_biscuits = 6 →
  num_dogs = 2 →
  total_biscuits = num_dogs * biscuits_per_dog →
  biscuits_per_dog = 3 := by
  sorry

end NUMINAMATH_CALUDE_valentine_biscuits_l2611_261100


namespace NUMINAMATH_CALUDE_tile_arrangements_l2611_261193

def brown_tiles : ℕ := 2
def purple_tiles : ℕ := 1
def green_tiles : ℕ := 3
def yellow_tiles : ℕ := 4

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem tile_arrangements :
  (Nat.factorial total_tiles) / 
  (Nat.factorial yellow_tiles * Nat.factorial green_tiles * 
   Nat.factorial brown_tiles * Nat.factorial purple_tiles) = 12600 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l2611_261193


namespace NUMINAMATH_CALUDE_pentagon_to_squares_ratio_is_one_eighth_l2611_261152

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a square with given side length and bottom-left corner -/
structure Square :=
  (bottomLeft : Point)
  (sideLength : ℝ)

/-- Configuration of three squares as described in the problem -/
structure SquareConfiguration :=
  (square1 : Square)
  (square2 : Square)
  (square3 : Square)

/-- The ratio of the area of pentagon PAWSR to the total area of three squares -/
def pentagonToSquaresRatio (config : SquareConfiguration) : ℝ :=
  sorry

/-- Theorem stating that the ratio is 1/8 for the given configuration -/
theorem pentagon_to_squares_ratio_is_one_eighth
  (config : SquareConfiguration)
  (h1 : config.square1.sideLength = 1)
  (h2 : config.square2.sideLength = 1)
  (h3 : config.square3.sideLength = 1)
  (h4 : config.square1.bottomLeft.x = config.square2.bottomLeft.x)
  (h5 : config.square1.bottomLeft.y + 1 = config.square2.bottomLeft.y)
  (h6 : config.square2.bottomLeft.x + 1 = config.square3.bottomLeft.x)
  (h7 : config.square2.bottomLeft.y = config.square3.bottomLeft.y) :
  pentagonToSquaresRatio config = 1/8 :=
sorry

end NUMINAMATH_CALUDE_pentagon_to_squares_ratio_is_one_eighth_l2611_261152


namespace NUMINAMATH_CALUDE_ten_percent_of_x_l2611_261138

theorem ten_percent_of_x (x c : ℝ) : 
  3 - (1/4)*2 - (1/3)*3 - (1/7)*x = c → 
  (10/100) * x = 0.7 * (1.5 - c) := by
sorry

end NUMINAMATH_CALUDE_ten_percent_of_x_l2611_261138


namespace NUMINAMATH_CALUDE_complement_S_union_T_eq_less_equal_one_l2611_261150

-- Define the sets S and T
def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

-- State the theorem
theorem complement_S_union_T_eq_less_equal_one :
  (Set.univ \ S) ∪ T = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_S_union_T_eq_less_equal_one_l2611_261150


namespace NUMINAMATH_CALUDE_line_through_circle_center_perpendicular_to_given_line_l2611_261117

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define the given line equation
def given_line (x y : ℝ) : Prop := x + y = 0

-- Define the equation of the line we want to prove
def target_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem line_through_circle_center_perpendicular_to_given_line :
  ∃ (cx cy : ℝ),
    (∀ x y, circle_equation x y ↔ (x - cx)^2 + (y - cy)^2 = (-cx)^2 + (-cy)^2) ∧
    target_line cx cy ∧
    (∀ x y, target_line x y → given_line x y → (x - cx) * (x - cx) + (y - cy) * (y - cy) = 0) :=
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_perpendicular_to_given_line_l2611_261117


namespace NUMINAMATH_CALUDE_original_tomatoes_cost_l2611_261157

def original_order : ℝ := 25
def new_tomatoes : ℝ := 2.20
def old_lettuce : ℝ := 1.00
def new_lettuce : ℝ := 1.75
def old_celery : ℝ := 1.96
def new_celery : ℝ := 2.00
def delivery_tip : ℝ := 8.00
def new_total : ℝ := 35

theorem original_tomatoes_cost (x : ℝ) : 
  x = 3.41 ↔ 
  x + old_lettuce + old_celery + delivery_tip = new_total ∧
  new_tomatoes + new_lettuce + new_celery + delivery_tip = new_total :=
by sorry

end NUMINAMATH_CALUDE_original_tomatoes_cost_l2611_261157


namespace NUMINAMATH_CALUDE_g_one_equals_three_l2611_261146

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem g_one_equals_three (f g : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_even : is_even_function g) 
  (h1 : f (-1) + g 1 = 2) 
  (h2 : f 1 + g (-1) = 4) : 
  g 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_one_equals_three_l2611_261146


namespace NUMINAMATH_CALUDE_find_principal_l2611_261140

/-- Given a sum of money P (principal) and an interest rate R,
    calculate the amount after T years with simple interest. -/
def simpleInterest (P R T : ℚ) : ℚ :=
  P + (P * R * T) / 100

theorem find_principal (R : ℚ) :
  ∃ P : ℚ,
    simpleInterest P R 1 = 1717 ∧
    simpleInterest P R 2 = 1734 ∧
    P = 1700 := by
  sorry

end NUMINAMATH_CALUDE_find_principal_l2611_261140


namespace NUMINAMATH_CALUDE_part_one_part_two_l2611_261106

-- Part 1
theorem part_one (x y : ℝ) (hx : x = Real.sqrt 2 - 1) (hy : y = Real.sqrt 2 + 1) :
  y / x + x / y = 6 := by sorry

-- Part 2
theorem part_two :
  (Real.sqrt 3 + Real.sqrt 2 - 2) * (Real.sqrt 3 - Real.sqrt 2 + 2) = 4 * Real.sqrt 2 - 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2611_261106


namespace NUMINAMATH_CALUDE_quadrilateral_with_parallel_sides_and_congruent_diagonals_is_rectangle_l2611_261188

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of a quadrilateral
def has_parallel_sides (q : Quadrilateral) : Prop := 
  sorry

def has_congruent_diagonals (q : Quadrilateral) : Prop := 
  sorry

def is_rectangle (q : Quadrilateral) : Prop := 
  sorry

-- Theorem statement
theorem quadrilateral_with_parallel_sides_and_congruent_diagonals_is_rectangle 
  (q : Quadrilateral) : 
  has_parallel_sides q → has_congruent_diagonals q → is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_with_parallel_sides_and_congruent_diagonals_is_rectangle_l2611_261188


namespace NUMINAMATH_CALUDE_right_triangle_least_side_l2611_261104

theorem right_triangle_least_side (a b c : ℝ) : 
  a = 8 → b = 15 → c^2 = a^2 + b^2 → min a (min b c) = 8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_least_side_l2611_261104


namespace NUMINAMATH_CALUDE_man_age_twice_son_age_l2611_261123

/-- Proves that the number of years until a man's age is twice his son's age is 2,
    given that the man is currently 24 years older than his son and the son is currently 22 years old. -/
theorem man_age_twice_son_age (son_age : ℕ) (man_age : ℕ) (years : ℕ) : 
  son_age = 22 →
  man_age = son_age + 24 →
  man_age + years = 2 * (son_age + years) →
  years = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_age_twice_son_age_l2611_261123


namespace NUMINAMATH_CALUDE_f_is_odd_and_increasing_l2611_261190

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop := ∀ a b, a < b → f a < f b

-- Theorem statement
theorem f_is_odd_and_increasing : is_odd f ∧ is_increasing f := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_and_increasing_l2611_261190


namespace NUMINAMATH_CALUDE_unemployment_forecast_l2611_261176

theorem unemployment_forecast (x : ℝ) (h1 : x > 0) : 
  let current_unemployed := 0.056 * x
  let next_year_active := 1.04 * x
  let next_year_unemployed := 0.91 * current_unemployed
  (next_year_unemployed / next_year_active) * 100 = 4.9 := by
sorry


end NUMINAMATH_CALUDE_unemployment_forecast_l2611_261176


namespace NUMINAMATH_CALUDE_exam_score_deviation_l2611_261179

/-- Given an exam with mean score 74 and standard deviation σ,
    prove that 58 is 2σ below the mean when 98 is 3σ above the mean. -/
theorem exam_score_deviation (σ : ℝ) : 
  (74 + 3 * σ = 98) → (74 - 2 * σ = 58) := by
  sorry

end NUMINAMATH_CALUDE_exam_score_deviation_l2611_261179


namespace NUMINAMATH_CALUDE_systematic_sampling_third_event_l2611_261187

/-- Given a total of 960 students, selecting every 30th student starting from
    student number 30, the number of selected students in the interval [701, 960] is 9. -/
theorem systematic_sampling_third_event (total_students : Nat) (selection_interval : Nat) 
    (first_selected : Nat) (event_start : Nat) (event_end : Nat) : Nat :=
  have h1 : total_students = 960 := by sorry
  have h2 : selection_interval = 30 := by sorry
  have h3 : first_selected = 30 := by sorry
  have h4 : event_start = 701 := by sorry
  have h5 : event_end = 960 := by sorry
  9

#check systematic_sampling_third_event

end NUMINAMATH_CALUDE_systematic_sampling_third_event_l2611_261187


namespace NUMINAMATH_CALUDE_subcubes_two_plus_painted_faces_count_l2611_261196

/-- Represents a cube with side length n --/
structure Cube (n : ℕ) where
  side_length : ℕ
  painted_faces : ℕ
  h_side : side_length = n
  h_painted : painted_faces = 6

/-- Represents a subcube of a larger cube --/
structure Subcube (n : ℕ) where
  painted_faces : ℕ
  h_painted : painted_faces ≤ 3

/-- The number of subcubes with at least two painted faces in a painted cube --/
def subcubes_with_two_plus_painted_faces (c : Cube 4) : ℕ := sorry

/-- Theorem stating that the number of 1x1x1 subcubes with at least two painted faces
    in a 4x4x4 fully painted cube is 32 --/
theorem subcubes_two_plus_painted_faces_count (c : Cube 4) :
  subcubes_with_two_plus_painted_faces c = 32 := by sorry

end NUMINAMATH_CALUDE_subcubes_two_plus_painted_faces_count_l2611_261196


namespace NUMINAMATH_CALUDE_beta_max_success_ratio_l2611_261199

theorem beta_max_success_ratio 
  (alpha_day1_score alpha_day1_total : ℕ)
  (alpha_day2_score alpha_day2_total : ℕ)
  (beta_day1_score beta_day1_total : ℕ)
  (beta_day2_score beta_day2_total : ℕ) :
  alpha_day1_score = 210 →
  alpha_day1_total = 400 →
  alpha_day2_score = 210 →
  alpha_day2_total = 300 →
  beta_day1_total + beta_day2_total = 700 →
  beta_day1_total < 400 →
  beta_day2_total < 400 →
  beta_day1_score > 0 →
  beta_day2_score > 0 →
  (beta_day1_score : ℚ) / beta_day1_total < (alpha_day1_score : ℚ) / alpha_day1_total →
  (beta_day2_score : ℚ) / beta_day2_total < (alpha_day2_score : ℚ) / alpha_day2_total →
  (alpha_day1_score + alpha_day2_score : ℚ) / (alpha_day1_total + alpha_day2_total) = 3/5 →
  (beta_day1_score + beta_day2_score : ℚ) / 700 ≤ 139/700 :=
by sorry

end NUMINAMATH_CALUDE_beta_max_success_ratio_l2611_261199


namespace NUMINAMATH_CALUDE_range_of_a_l2611_261109

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 8*x - 20 < 0 → x^2 - 2*x + 1 - a^2 ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x + 1 - a^2 ≤ 0 ∧ x^2 - 8*x - 20 ≥ 0) ∧
  (a > 0) →
  a ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2611_261109


namespace NUMINAMATH_CALUDE_line_slope_problem_l2611_261105

theorem line_slope_problem (a : ℝ) : 
  (3 * a - 7) / (a - 2) = 2 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_problem_l2611_261105


namespace NUMINAMATH_CALUDE_dogs_equal_initial_l2611_261120

/-- Calculates the remaining number of dogs in an animal rescue center after a series of events. -/
def remaining_dogs (initial : ℕ) (moved_in : ℕ) (first_adoption : ℕ) (second_adoption : ℕ) : ℕ :=
  initial + moved_in - first_adoption - second_adoption

/-- Theorem stating that the number of remaining dogs equals the initial number under specific conditions. -/
theorem dogs_equal_initial 
  (initial : ℕ) (moved_in : ℕ) (first_adoption : ℕ) (second_adoption : ℕ) 
  (h1 : initial = 200) 
  (h2 : moved_in = 100) 
  (h3 : first_adoption = 40) 
  (h4 : second_adoption = 60) : 
  remaining_dogs initial moved_in first_adoption second_adoption = initial :=
by sorry

end NUMINAMATH_CALUDE_dogs_equal_initial_l2611_261120


namespace NUMINAMATH_CALUDE_triangle_problem_l2611_261133

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  -- Law of cosines
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →
  -- Given conditions
  a = 2 * Real.sqrt 6 →
  b = 3 →
  Real.sin (B + C)^2 + Real.sqrt 2 * Real.sin (2 * A) = 0 →
  -- Conclusion
  c = 3 ∧ Real.cos B = Real.sqrt 6 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2611_261133


namespace NUMINAMATH_CALUDE_x_minus_q_equals_three_l2611_261195

theorem x_minus_q_equals_three (x q : ℝ) (h1 : |x - 3| = q) (h2 : x > 3) :
  x - q = 3 := by sorry

end NUMINAMATH_CALUDE_x_minus_q_equals_three_l2611_261195


namespace NUMINAMATH_CALUDE_product_of_four_numbers_l2611_261166

theorem product_of_four_numbers (A B C D : ℝ) : 
  A > 0 → B > 0 → C > 0 → D > 0 →
  A + B + C + D = 40 →
  A + 3 = B - 3 ∧ A + 3 = C * 3 ∧ A + 3 = D / 3 →
  A * B * C * D = 2666.25 := by
sorry

end NUMINAMATH_CALUDE_product_of_four_numbers_l2611_261166


namespace NUMINAMATH_CALUDE_sophie_journey_l2611_261165

/-- Proves that given the specified journey conditions, the walking distance is 5.1 km -/
theorem sophie_journey (d : ℝ) 
  (h1 : (2/3 * d) / 20 + (1/3 * d) / 4 = 1.8) : 
  1/3 * d = 5.1 := by
  sorry

end NUMINAMATH_CALUDE_sophie_journey_l2611_261165


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2611_261158

def f (x : ℝ) := 3 * x^3 - 9 * x + 5

theorem f_max_min_on_interval :
  ∃ (max min : ℝ) (x_max x_min : ℝ),
    x_max ∈ Set.Icc (-3 : ℝ) 3 ∧
    x_min ∈ Set.Icc (-3 : ℝ) 3 ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ f x_min) ∧
    f x_max = max ∧
    f x_min = min ∧
    max = 59 ∧
    min = -49 ∧
    x_max = 3 ∧
    x_min = -3 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2611_261158


namespace NUMINAMATH_CALUDE_female_rabbits_count_l2611_261170

theorem female_rabbits_count (white_rabbits black_rabbits male_rabbits : ℕ) 
  (h1 : white_rabbits = 11)
  (h2 : black_rabbits = 13)
  (h3 : male_rabbits = 15) : 
  white_rabbits + black_rabbits - male_rabbits = 9 := by
  sorry

end NUMINAMATH_CALUDE_female_rabbits_count_l2611_261170


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_four_l2611_261178

theorem reciprocal_of_negative_four :
  (1 : ℚ) / (-4 : ℚ) = -1/4 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_four_l2611_261178


namespace NUMINAMATH_CALUDE_min_slope_tangent_l2611_261154

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - 1 / (a * x)

theorem min_slope_tangent (a : ℝ) (h : a > 0) :
  let k := (deriv (f a)) 1
  ∀ b > 0, k ≤ (deriv (f b)) 1 ↔ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_slope_tangent_l2611_261154


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2611_261145

theorem arithmetic_expression_equality : 
  1 / 2 + ((2 / 3 * 3 / 8) + 4) - 8 / 16 = 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2611_261145


namespace NUMINAMATH_CALUDE_valentina_share_ratio_l2611_261141

/-- The length of the burger in inches -/
def burger_length : ℚ := 12

/-- The length of each person's share in inches -/
def share_length : ℚ := 6

/-- The ratio of Valentina's share to the whole burger -/
def valentina_ratio : ℚ × ℚ := (share_length, burger_length)

theorem valentina_share_ratio :
  valentina_ratio = (1, 2) := by sorry

end NUMINAMATH_CALUDE_valentina_share_ratio_l2611_261141


namespace NUMINAMATH_CALUDE_uncommon_roots_product_l2611_261121

def P (x : ℝ) : ℝ := x^4 + 2*x^3 - 8*x^2 - 6*x + 15
def Q (x : ℝ) : ℝ := x^3 + 4*x^2 - x - 10

theorem uncommon_roots_product : 
  ∃ (r₁ r₂ : ℝ), 
    P r₁ = 0 ∧ 
    Q r₂ = 0 ∧ 
    r₁ ≠ r₂ ∧
    (∀ x : ℝ, (P x = 0 ∧ Q x ≠ 0) ∨ (Q x = 0 ∧ P x ≠ 0) → x = r₁ ∨ x = r₂) ∧
    r₁ * r₂ = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_uncommon_roots_product_l2611_261121


namespace NUMINAMATH_CALUDE_balloon_problem_l2611_261115

theorem balloon_problem (total_people : ℕ) (total_balloons : ℕ) 
  (x₁ x₂ x₃ x₄ : ℕ) 
  (h1 : total_people = 101)
  (h2 : total_balloons = 212)
  (h3 : x₁ + x₂ + x₃ + x₄ = total_people)
  (h4 : x₁ + 2*x₂ + 3*x₃ + 4*x₄ = total_balloons)
  (h5 : x₄ = x₂ + 13) :
  x₁ = 52 := by
  sorry

end NUMINAMATH_CALUDE_balloon_problem_l2611_261115


namespace NUMINAMATH_CALUDE_surface_area_ratio_of_cubes_l2611_261111

theorem surface_area_ratio_of_cubes (a b : ℝ) (h : a > 0) (k : b > 0) (ratio : a = 4 * b) :
  (6 * a^2) / (6 * b^2) = 16 := by sorry

end NUMINAMATH_CALUDE_surface_area_ratio_of_cubes_l2611_261111


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2611_261171

/-- Given a larger square of side y and a smaller central square of side x,
    surrounded by six congruent rectangles, where the side of each rectangle
    touching the square's sides equals x, and there are two rectangles along
    one side of y, this theorem proves that the perimeter of one of the six
    congruent rectangles is x + y. -/
theorem rectangle_perimeter (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) :
  let rectangle_width := x
  let rectangle_height := (y - x) / 2
  rectangle_width + rectangle_height + rectangle_width + rectangle_height = x + y :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2611_261171

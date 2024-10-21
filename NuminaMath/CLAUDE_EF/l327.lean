import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_equation_solutions_l327_32799

def is_solution (m n : ℕ) (p : ℕ) : Prop :=
  Nat.choose m 2 - 1 = p^n ∧ Nat.Prime p

theorem binomial_coefficient_equation_solutions :
  ∀ m n p : ℕ,
    is_solution m n p ↔ 
      ((m = 3 ∧ n = 2 ∧ p = 2) ∨
       (m = 4 ∧ n = 2 ∧ p = 5) ∨
       (m = 5 ∧ n = 2 ∧ p = 3) ∨
       (m = 8 ∧ n = 3 ∧ p = 3)) :=
by sorry

#check binomial_coefficient_equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_equation_solutions_l327_32799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_finite_l327_32740

def last_digit (n : ℕ) : ℕ := n % 10

def next_term (n : ℕ) : ℕ :=
  if last_digit n ≤ 5 then n / 10 else 9 * n

def sequence_term (a₀ : ℕ) : ℕ → ℕ
  | 0 => a₀
  | n + 1 => next_term (sequence_term a₀ n)

theorem sequence_is_finite (a₀ : ℕ) : 
  ∃ N : ℕ, ∀ n > N, sequence_term a₀ n = 0 := by
  sorry

#check sequence_is_finite

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_finite_l327_32740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_factorial_sum_l327_32763

theorem largest_prime_factor_of_factorial_sum : 
  ∀ p : ℕ, Nat.Prime p → p ∣ (Nat.factorial 7 + Nat.factorial 8) → p ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_factorial_sum_l327_32763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_fourth_term_l327_32720

/-- Geometric sequence with first term 1 and common ratio q -/
noncomputable def geometric_sequence (q : ℝ) : ℕ → ℝ := fun n => q^(n-1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then n else (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_fourth_term 
  (q : ℝ) (h : geometric_sum q 7 - 4 * geometric_sum q 6 + 3 * geometric_sum q 5 = 0) :
  geometric_sum q 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_fourth_term_l327_32720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalence_of_statements_l327_32738

variable (a b c A B C : ℝ)

def statement1 (a b c A B C : ℝ) : Prop :=
  ∃ k, (a / Real.sin A = k ∧ b / Real.sin B = k ∧ c / Real.sin C = k) ∧
  A + B + C = Real.pi

def statement2 (a b c A B C : ℝ) : Prop :=
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A ∧
  b^2 = c^2 + a^2 - 2*c*a*Real.cos B ∧
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C

def statement3 (a b c A B C : ℝ) : Prop :=
  a = b * Real.cos C + c * Real.cos B ∧
  b = c * Real.cos A + a * Real.cos C ∧
  c = a * Real.cos B + b * Real.cos A

theorem equivalence_of_statements
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hA : Real.sin A > 0) (hB : Real.sin B > 0) (hC : Real.sin C > 0) :
  (statement1 a b c A B C ↔ statement2 a b c A B C) ∧
  (statement2 a b c A B C ↔ statement3 a b c A B C) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalence_of_statements_l327_32738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_169_l327_32770

def f : ℕ → ℤ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 3
  | n + 3 => f (n + 2) + f (n + 1) - (n + 3)

theorem f_10_equals_169 : f 10 = 169 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_169_l327_32770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_brought_100_cans_l327_32771

/-- The number of cans brought in by Rachel -/
def rachel_cans : ℕ := sorry

/-- The number of cans brought in by Jaydon -/
def jaydon_cans : ℕ := 5 + 2 * rachel_cans

/-- The number of cans brought in by Mark -/
def mark_cans : ℕ := 4 * jaydon_cans

/-- The total number of cans -/
def total_cans : ℕ := 135

theorem mark_brought_100_cans : 
  rachel_cans + jaydon_cans + mark_cans = total_cans → mark_cans = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_brought_100_cans_l327_32771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_matching_leftmost_digits_l327_32762

theorem no_matching_leftmost_digits : ¬∃ (n : ℕ), 
  (∃ (k : ℕ), 5 * 10^k ≤ (2 : ℝ)^n ∧ (2 : ℝ)^n < 6 * 10^k) ∧ 
  (∃ (m : ℕ), 2 * 10^m ≤ (5 : ℝ)^n ∧ (5 : ℝ)^n < 3 * 10^m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_matching_leftmost_digits_l327_32762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l327_32723

open Real

noncomputable def f (x : ℝ) : ℝ :=
  arctan (sqrt ((log (1/0.5)) * ((sin x) / (sin x + 7))))

theorem f_range :
  Set.range f = Set.Ioc 0 (π / 6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l327_32723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hao_hao_age_l327_32734

def is_leap_year (year : ℕ) : Bool :=
  year % 4 = 0 && (year % 100 ≠ 0 || year % 400 = 0)

def count_leap_years (start_year end_year : ℕ) : ℕ :=
  (List.range (end_year - start_year + 1)).filter (fun y => is_leap_year (y + start_year)) |>.length

theorem hao_hao_age (birth_year : ℕ) (h1 : birth_year % 9 = 0)
  (h2 : count_leap_years birth_year 2015 = 2) : 2016 - birth_year = 9 := by
  sorry

#check hao_hao_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hao_hao_age_l327_32734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_sum_l327_32764

theorem trig_ratio_sum (x y : ℝ) 
  (h1 : Real.sin x / Real.sin y = 4)
  (h2 : Real.cos x / Real.cos y = 1/3) :
  Real.sin (2*x) / Real.sin (2*y) + Real.cos (2*x) / Real.cos (2*y) = 113/23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_sum_l327_32764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_correct_translation_l327_32748

/-- Represents a parabola of the form y = a(x - h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The original parabola y = 2(x+1)^2 + 2 --/
def original : Parabola := { a := 2, h := -1, k := 2 }

/-- The transformed parabola y = 2x^2 --/
def transformed : Parabola := { a := 2, h := 0, k := 0 }

/-- Represents a translation in 2D space --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a parabola --/
def applyTranslation (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a, h := p.h - t.dx, k := p.k - t.dy }

theorem parabola_translation (p : Parabola) (t : Translation) :
  applyTranslation p t = { a := p.a, h := p.h - t.dx, k := p.k - t.dy } :=
by rfl

theorem correct_translation :
  ∃ (t : Translation), applyTranslation original t = transformed ∧ t.dx = 1 ∧ t.dy = -2 :=
by
  use { dx := 1, dy := -2 }
  simp [applyTranslation, original, transformed]
  sorry -- The proof details are omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_correct_translation_l327_32748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l327_32786

-- Define the triangle ABC
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

-- Define the inscribed circle
structure InscribedCircle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Helper functions (not implemented, just signatures)
noncomputable def angle (p1 p2 p3 : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

noncomputable def distance (p1 p2 : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

def touches_midline (circle : InscribedCircle) (t : Triangle) : Prop := sorry

-- Define the theorem
theorem triangle_side_length 
  (t : Triangle) 
  (circle : InscribedCircle) 
  (h1 : circle.radius = 1)
  (h2 : Real.cos (angle t.A t.B t.C) = 0.8)
  (h3 : touches_midline circle t) : 
  distance t.A t.C = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l327_32786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inequality_l327_32746

/-- A right-angled triangle with altitude and points on its sides -/
structure RightTriangleWithAltitude where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  K : ℝ × ℝ
  L : ℝ × ℝ
  right_angle_at_A : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0
  D_on_altitude : (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0
  K_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ K = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  L_on_AC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ L = (A.1 + s * (C.1 - A.1), A.2 + s * (C.2 - A.2))
  KL_through_incenters : True  -- Placeholder, as this condition is complex

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))

/-- The main theorem -/
theorem area_inequality (t : RightTriangleWithAltitude) :
  triangle_area t.A t.B t.C ≥ 2 * triangle_area t.A t.K t.L := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inequality_l327_32746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l327_32753

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + Real.log x

-- State the theorem
theorem tangent_line_at_one :
  (∀ x > 0, Differentiable ℝ (fun x => f x)) →
  (∀ x, f (Real.exp x) = x + Real.exp x) →
  ∃ m b, ∀ x, m * x + b = 2 * x - 1 ∧ 
          m * 1 + b = f 1 ∧
          m = deriv f 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l327_32753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_constant_term_value_l327_32780

theorem constant_term_expansion (x : ℝ) : 
  (x^4 + x^2 + 6) * (x^5 + x^3 + x + 20) = x^9 + x^7 + x^5 + 20*x^4 + x^5 + x^3 + x + 20 + 6*x^5 + 6*x^3 + 6*x + 120 :=
by sorry

theorem constant_term_value : 
  (fun x : ℝ ↦ (x^4 + x^2 + 6) * (x^5 + x^3 + x + 20)) 0 = 120 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_constant_term_value_l327_32780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_parallel_vector_l327_32727

-- Define the function types
noncomputable def f (x : ℝ) : ℝ := 2^x
noncomputable def g (x : ℝ) : ℝ := 2^(x-2) + 3

-- Define the translation vector
def a : ℝ × ℝ := (2, 3)

-- State the theorem
theorem graph_translation_parallel_vector :
  (∀ x : ℝ, g x = f (x - 2) + 3) →
  (∃ k : ℝ, k ≠ 0 ∧ ∃ b : ℝ × ℝ, b = (-2, -3) ∧ a = k • b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_parallel_vector_l327_32727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_and_arithmetic_l327_32768

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List ℕ) (base : ℕ) : ℕ :=
  digits.enum.foldr (fun p acc => acc + p.2 * base^p.1) 0

-- Define the numbers in their original bases
def num1 : List ℕ := [2, 4, 5]  -- 245₈
def num2 : List ℕ := [1, 5]     -- 15₄
def num3 : List ℕ := [2, 3, 2]  -- 232₅
def num4 : List ℕ := [3, 2]     -- 32₆

-- Convert numbers to base 10
def num1_base10 : ℕ := to_base_10 num1 8
def num2_base10 : ℕ := to_base_10 num2 4
def num3_base10 : ℕ := to_base_10 num3 5
def num4_base10 : ℕ := to_base_10 num4 6

-- Perform the calculation
noncomputable def result : ℚ := (num1_base10 : ℚ) / num2_base10 - (num3_base10 : ℚ) / num4_base10

-- Round to the nearest integer
noncomputable def rounded_result : ℤ := round result

theorem base_conversion_and_arithmetic :
  rounded_result = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_and_arithmetic_l327_32768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_product_l327_32710

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def l (k m x y : ℝ) : Prop := y = k*x + m

-- Define the point Q
def Q (k m : ℝ) : ℝ × ℝ := (-4, k*(-4) + m)

-- Define the left focus F
def F : ℝ × ℝ := (-1, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Statement of the theorem
theorem ellipse_constant_product 
  (k m : ℝ) 
  (A B P : ℝ × ℝ) 
  (hA : E A.1 A.2 ∧ l k m A.1 A.2) 
  (hB : E B.1 B.2 ∧ l k m B.1 B.2) 
  (hP : E P.1 P.2) 
  (hOP : P = (A.1 + B.1, A.2 + B.2)) :
  (P.1 - O.1) * (F.1 - (Q k m).1) + (P.2 - O.2) * (F.2 - (Q k m).2) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_product_l327_32710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pokemon_card_value_l327_32721

def jenny_cards : ℕ := 6
def orlando_cards : ℕ := jenny_cards + 2
def richard_cards : ℕ := 3 * orlando_cards

def jenny_rare : ℕ := Int.toNat ((jenny_cards : ℚ) * (1/2) |>.floor)
def orlando_rare : ℕ := Int.toNat ((orlando_cards : ℚ) * (2/5) |>.floor)
def richard_rare : ℕ := Int.toNat ((richard_cards : ℚ) * (1/4) |>.floor)

def rare_value : ℕ := 10
def non_rare_value : ℕ := 3

def total_value : ℕ :=
  (jenny_rare + orlando_rare + richard_rare) * rare_value +
  (jenny_cards - jenny_rare + orlando_cards - orlando_rare + richard_cards - richard_rare) * non_rare_value

theorem total_pokemon_card_value : total_value = 198 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pokemon_card_value_l327_32721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_elevenths_rounded_l327_32794

/-- Rounds a real number to the specified number of decimal places -/
noncomputable def round_to_decimal_places (x : ℝ) (n : ℕ) : ℝ :=
  (⌊x * 10^n + 0.5⌋) / 10^n

/-- The fraction 8/11 rounded to 3 decimal places is equal to 0.727 -/
theorem eight_elevenths_rounded : round_to_decimal_places (8/11) 3 = 0.727 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_elevenths_rounded_l327_32794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_extension_l327_32733

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 4 then x^2 - 2*x
  else if -4 ≤ x ∧ x < 0 then -x^2 - 2*x
  else 0  -- define for other x values to make it total

-- State the theorem
theorem f_odd_extension :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ∈ Set.Icc 0 4, f x = x^2 - 2*x) →  -- f(x) = x^2 - 2x for 0 ≤ x ≤ 4
  (∀ x ∈ Set.Icc (-4) 0, f x = -x^2 - 2*x) :=  -- f(x) = -x^2 - 2x for -4 ≤ x ≤ 0
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_extension_l327_32733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_most_appropriate_l327_32709

/-- Represents the gender of a student -/
inductive Gender
| Male
| Female

/-- Represents whether a student has myopia -/
inductive MyopiaStatus
| Myopic
| NonMyopic

/-- Represents the sample data for the myopia study -/
structure MyopiaStudyData where
  totalMale : Nat
  myopicMale : Nat
  totalFemale : Nat
  myopicFemale : Nat

/-- Represents different statistical methods -/
inductive StatisticalMethod
| ExpectationAndVariance
| PermutationAndCombination
| IndependenceTest
| Probability

/-- Function to determine the most appropriate method for analyzing myopia-gender relationship -/
def mostAppropriateMethod (data : MyopiaStudyData) : StatisticalMethod :=
  StatisticalMethod.IndependenceTest

/-- States that the Independence Test is the most appropriate method for analyzing the relationship between myopia and gender -/
def independenceTestIsAppropriate (data : MyopiaStudyData) : Prop :=
  mostAppropriateMethod data = StatisticalMethod.IndependenceTest

/-- Theorem stating that the Independence Test is the most appropriate method for the given myopia study data -/
theorem independence_test_most_appropriate (data : MyopiaStudyData) 
  (h1 : data.totalMale = 150)
  (h2 : data.myopicMale = 80)
  (h3 : data.totalFemale = 140)
  (h4 : data.myopicFemale = 70) :
  independenceTestIsAppropriate data := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_most_appropriate_l327_32709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_l327_32705

theorem cosine_sum_product : ∃ (a b c d : ℕ+),
  ∀ x : ℝ, (Real.cos (2 * x) + Real.cos (4 * x) + Real.cos (10 * x) + Real.cos (14 * x) = 
   (a : ℝ) * Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x)) ∧
  (a : ℝ) + b + c + d = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_l327_32705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_length_is_three_l327_32701

/-- The length of a rectangular tank satisfying given conditions -/
noncomputable def tank_length (w h cost total_cost : ℝ) : ℝ :=
  let surface_area := λ l => 2 * l * w + 2 * l * h + 2 * w * h
  (total_cost / cost - 2 * w * h) / (2 * w + 2 * h)

/-- Theorem stating that the tank length is 3 feet given the conditions -/
theorem tank_length_is_three :
  tank_length 7 2 20 1640 = 3 := by
  -- Unfold the definition of tank_length
  unfold tank_length
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_length_is_three_l327_32701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_to_others_ratio_approx_l327_32751

/-- Represents the population ratio in a town -/
structure PopulationRatio where
  men : ℝ
  women : ℝ
  children : ℝ
  women_to_men_ratio : women = 0.9 * men
  children_to_adults_ratio : children = 0.6 * (men + women)

/-- The ratio of men to women and children combined as a percentage -/
noncomputable def men_to_others_ratio (p : PopulationRatio) : ℝ :=
  (p.men / (p.women + p.children)) * 100

/-- Theorem stating the ratio of men to women and children combined is approximately 49.02% -/
theorem men_to_others_ratio_approx (p : PopulationRatio) :
  abs (men_to_others_ratio p - 49.02) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_to_others_ratio_approx_l327_32751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_x_minus_y_range_l327_32788

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := ((3/5) * t, 1 + (4/5) * t)

-- Define the curve C in polar coordinates
noncomputable def curve_C_polar (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the curve C in Cartesian coordinates
def curve_C_cartesian (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Theorem 1: Cartesian equation of curve C
theorem curve_C_equation : 
  ∀ x y : ℝ, (∃ θ : ℝ, x = curve_C_polar θ * Real.cos θ ∧ y = curve_C_polar θ * Real.sin θ) 
  ↔ curve_C_cartesian x y :=
sorry

-- Theorem 2: Range of x-y for internal common point
theorem x_minus_y_range :
  ∀ x y t : ℝ, 
    line_l t = (x, y) ∧ x^2 + (y - 1)^2 < 1 
    → -6/5 < x - y ∧ x - y < -4/5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_x_minus_y_range_l327_32788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_AGHB_is_480_l327_32783

-- Define the rectangle
noncomputable def rectangle_length : ℝ := 40
noncomputable def rectangle_width : ℝ := 24

-- Define the midpoints
def G_is_midpoint (AD : ℝ) : Prop := AD / 2 = rectangle_width / 2
def H_is_midpoint (AB : ℝ) : Prop := AB / 2 = rectangle_length / 2

-- Define the area of quadrilateral AGHB
noncomputable def area_AGHB : ℝ := rectangle_length * rectangle_width / 2

-- Theorem statement
theorem area_of_AGHB_is_480 :
  G_is_midpoint rectangle_width →
  H_is_midpoint rectangle_length →
  area_AGHB = 480 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_AGHB_is_480_l327_32783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_times_l327_32789

-- Define the task completion time when working together
noncomputable def combined_time : ℝ := 24

-- Define the relationship between Alfred's and Bill's work rates
noncomputable def alfred_rate_ratio : ℝ := 2/3

-- Define Alfred's individual completion time
noncomputable def alfred_time : ℝ := 60

-- Define Bill's individual completion time
noncomputable def bill_time : ℝ := 40

-- Theorem statement
theorem task_completion_times :
  (1 / alfred_time + 1 / bill_time = 1 / combined_time) ∧
  (1 / alfred_time = alfred_rate_ratio * (1 / bill_time)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_times_l327_32789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_needed_is_1368_l327_32719

-- Define the room dimensions in feet
noncomputable def room_length : ℝ := 15
noncomputable def room_width : ℝ := 16

-- Define the tile dimensions in feet
noncomputable def tile_length : ℝ := 8 / 12
noncomputable def tile_width : ℝ := 3 / 12

-- Define the carpet dimensions in feet
noncomputable def carpet_length : ℝ := 3
noncomputable def carpet_width : ℝ := 4

-- Calculate the total room area
noncomputable def room_area : ℝ := room_length * room_width

-- Calculate the carpet area
noncomputable def carpet_area : ℝ := carpet_length * carpet_width

-- Calculate the tileable area
noncomputable def tileable_area : ℝ := room_area - carpet_area

-- Calculate the area of one tile
noncomputable def tile_area : ℝ := tile_length * tile_width

-- Define the number of tiles needed
noncomputable def tiles_needed : ℕ := Int.toNat ⌈tileable_area / tile_area⌉

-- Theorem statement
theorem tiles_needed_is_1368 : tiles_needed = 1368 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_needed_is_1368_l327_32719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l327_32707

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- Slope of line l1: ax + 8y - 3 = 0 -/
noncomputable def slope_l1 (a : ℝ) : ℝ := -a / 8

/-- Slope of line l2: 2x + ay - a = 0 -/
noncomputable def slope_l2 (a : ℝ) : ℝ := -2 / a

theorem parallel_condition (a : ℝ) :
  (a = 4 → are_parallel (slope_l1 a) (slope_l2 a)) ∧
  (∃ b : ℝ, b ≠ 4 ∧ are_parallel (slope_l1 b) (slope_l2 b)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l327_32707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_distinct_l327_32757

-- Define the inverse proportion function
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

-- State the theorem
theorem quadratic_roots_distinct (k : ℝ) :
  (∀ x₁ x₂, x₁ < 0 ∧ x₂ < 0 ∧ x₁ < x₂ → inverse_proportion k x₁ > inverse_proportion k x₂) →
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + 1 - k = 0 ∧ x₂^2 - 2*x₂ + 1 - k = 0 :=
by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_distinct_l327_32757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_value_l327_32704

-- Define the function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^(m^2 + 3*m + 1)

-- State the theorem
theorem inverse_proportion_m_value :
  ∀ m : ℝ, (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f m x * y = f m 1) → m = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_value_l327_32704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abs_diff_eq_n_squared_l327_32718

/-- Two sequences that satisfy the given conditions -/
structure SpecialSequences (n : ℕ) where
  a : Fin n → ℕ
  b : Fin n → ℕ
  a_decreasing : ∀ i j : Fin n, i < j → a i > a j
  b_increasing : ∀ i j : Fin n, i < j → b i < b j
  contains_all : (Finset.range (2 * n)).card = (Finset.image a (Finset.univ : Finset (Fin n)) ∪ Finset.image b (Finset.univ : Finset (Fin n))).card

/-- The sum of absolute differences equals n² -/
theorem sum_abs_diff_eq_n_squared (n : ℕ) (s : SpecialSequences n) :
  (Finset.univ : Finset (Fin n)).sum (λ i => (Int.natAbs (s.a i - s.b i))) = n ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abs_diff_eq_n_squared_l327_32718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_l327_32798

-- Define the equation
noncomputable def equation (t a : ℝ) : ℝ := 
  (|Real.cos t - 0.5| + |Real.sin t| - a) / (Real.sqrt 3 * Real.sin t - Real.cos t)

-- Define the interval
def interval : Set ℝ := Set.Icc 0 (Real.pi / 2)

-- Define the theorem
theorem equation_solution_exists (a : ℝ) : 
  (∃ t ∈ interval, equation t a = 0) ↔ (0.5 ≤ a ∧ a ≤ 1.5) := by
  sorry

#check equation_solution_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_l327_32798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_cube_edges_l327_32792

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ
  deriving Repr

/-- Represents the solid resulting from removing smaller cubes from the corners of a larger cube -/
structure RemainingCube where
  originalCube : Cube
  removedCubeSize : ℕ
  deriving Repr

/-- Calculates the number of edges in the remaining solid after removing smaller cubes from the corners of a larger cube -/
def countEdges (rc : RemainingCube) : ℕ :=
  sorry

/-- Theorem stating that removing unit cubes from the corners of a cube with side length 3 results in a solid with 84 edges -/
theorem remaining_cube_edges :
  ∀ (rc : RemainingCube),
    rc.originalCube.sideLength = 3 ∧
    rc.removedCubeSize = 1 →
    countEdges rc = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_cube_edges_l327_32792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_comparison_and_radius_counterexample_l327_32773

/-- A convex centrally symmetric polyhedron -/
class ConvexCentrallySymmetricPolyhedron (P : Type*) where
  -- Add necessary structure

/-- A section of a polyhedron -/
class Section (P : Type*) [ConvexCentrallySymmetricPolyhedron P] where
  area : ℝ
  smallest_circle_radius : ℝ

/-- Two sections are parallel -/
def parallel {P : Type*} [ConvexCentrallySymmetricPolyhedron P] 
  (S₁ S₂ : Section P) : Prop := sorry

/-- A section passes through the center of symmetry -/
def passes_through_center {P : Type*} [ConvexCentrallySymmetricPolyhedron P] 
  (S : Section P) : Prop := sorry

theorem area_comparison_and_radius_counterexample 
  {P : Type*} [ConvexCentrallySymmetricPolyhedron P]
  (S₁ S₂ : Section P) :
  parallel S₁ S₂ → passes_through_center S₁ → 
  (S₁.area ≥ S₂.area ∧ 
   ¬ (∀ (P : Type*) [ConvexCentrallySymmetricPolyhedron P]
      (S₁ S₂ : Section P), 
      parallel S₁ S₂ → passes_through_center S₁ → 
      S₁.smallest_circle_radius ≥ S₂.smallest_circle_radius)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_comparison_and_radius_counterexample_l327_32773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_Q_l327_32702

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the point P
def P : ℝ × ℝ := (1, 3)

-- Define the moving line passing through P
def movingLine (k : ℝ) (x : ℝ) : ℝ := 3 + k * (x - 1)

-- Define the intersection points A and B
def intersectionPoints (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x, p.1 = x ∧ p.2 = parabola x ∧ p.2 = movingLine k x}

-- Define the point Q as the intersection of tangents at A and B
noncomputable def Q (k : ℝ) : ℝ × ℝ := (k/2, k - 3)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem min_distance_to_Q :
  ∀ k : ℝ, ∃ min_dist : ℝ,
    (∀ p : ℝ × ℝ, circleEq p.1 p.2 → distance p (Q k) ≥ min_dist) ∧
    (∃ p : ℝ × ℝ, circleEq p.1 p.2 ∧ distance p (Q k) = min_dist) ∧
    min_dist = Real.sqrt 5 - 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_Q_l327_32702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_range_l327_32756

/-- The function for which we're finding the minimum -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 1) / Real.log a

/-- The condition for the existence of a minimum value -/
def has_minimum (a : ℝ) : Prop := ∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m

/-- The main theorem stating the range of a -/
theorem minimum_value_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  has_minimum a → 1 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_range_l327_32756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_rectangle_circle_triangle_l327_32766

/-- The area of the region common to a rectangle, circle, and inscribed right triangle -/
theorem common_area_rectangle_circle_triangle :
  ∀ (rectangle_length rectangle_width circle_radius : ℝ),
    rectangle_length = 10 →
    rectangle_width = 3 →
    circle_radius = 2.5 →
    (3.125 * Real.pi + 1 : ℝ) = (3.125 * Real.pi + 1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_rectangle_circle_triangle_l327_32766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l327_32790

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4)

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (x / 2 - Real.pi / 12)

def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem g_decreasing_interval :
  is_decreasing g (7 * Real.pi / 6) (19 * Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l327_32790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_a_positive_a_positive_implies_two_zeros_l327_32724

/-- A function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - x - 2 * a

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - 1

/-- Theorem stating that if f has two zeros, then a > 0 -/
theorem two_zeros_implies_a_positive (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) →
  a > 0 := by
  sorry

/-- Theorem stating that if a > 0, then f has two zeros -/
theorem a_positive_implies_two_zeros (a : ℝ) :
  a > 0 →
  ∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_a_positive_a_positive_implies_two_zeros_l327_32724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_button_array_theorem_l327_32761

/-- Represents a button in the array -/
inductive ButtonState
| OFF
| ON

/-- Represents the array of buttons -/
def ButtonArray := Fin 40 → Fin 50 → ButtonState

/-- Initial state of the array (all OFF) -/
def initialState : ButtonArray :=
  fun _ _ => ButtonState.OFF

/-- Final state of the array (all ON) -/
def finalState : ButtonArray :=
  fun _ _ => ButtonState.ON

/-- Represents a touch operation on the array -/
def touch (arr : ButtonArray) (row : Fin 40) (col : Fin 50) : ButtonArray :=
  fun r c => if r = row ∨ c = col then
    match arr r c with
    | ButtonState.OFF => ButtonState.ON
    | ButtonState.ON => ButtonState.OFF
  else arr r c

/-- Theorem stating that it's possible to reach the final state and the minimum number of touches -/
theorem button_array_theorem :
  (∃ (touches : List (Fin 40 × Fin 50)),
    touches.foldl (fun arr (r, c) => touch arr r c) initialState = finalState) ∧
  (∀ (touches : List (Fin 40 × Fin 50)),
    touches.foldl (fun arr (r, c) => touch arr r c) initialState = finalState →
    touches.length ≥ 2000) ∧
  (∃ (touches : List (Fin 40 × Fin 50)),
    touches.foldl (fun arr (r, c) => touch arr r c) initialState = finalState ∧
    touches.length = 2000) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_button_array_theorem_l327_32761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_242_div_9_l327_32726

-- Define the triangle's side lengths
def a : ℝ := 5
def b : ℝ := 7
def c : ℝ := 10

-- Define the perimeter of the triangle
def triangle_perimeter : ℝ := a + b + c

-- Define the width of the rectangle
noncomputable def w : ℝ := triangle_perimeter / 6

-- Define the length of the rectangle
noncomputable def l : ℝ := 2 * w

-- Theorem statement
theorem rectangle_area_is_242_div_9 :
  w * l = 242 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_242_div_9_l327_32726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_completion_time_l327_32793

/-- Represents the wall construction scenario -/
structure WallConstruction where
  initialWorkers : ℕ
  additionalWorkers : ℕ
  daysWorked : ℕ
  workCompleted : ℚ
  totalDays : ℕ

/-- Calculates the daily work rate per person -/
def dailyWorkRate (w : WallConstruction) : ℚ :=
  w.workCompleted / (w.initialWorkers * w.daysWorked)

/-- Theorem stating the initial plan was to complete the wall in 33 days -/
theorem wall_completion_time (w : WallConstruction) 
  (h1 : w.initialWorkers = 70)
  (h2 : w.additionalWorkers = 105)
  (h3 : w.daysWorked = 25)
  (h4 : w.workCompleted = 2/5)
  (h5 : ((w.initialWorkers + w.additionalWorkers : ℚ) * 
        (dailyWorkRate w) * (w.totalDays - w.daysWorked : ℚ) = 3/5)) :
  w.totalDays = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_completion_time_l327_32793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_positive_solution_set_l327_32711

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the property of being an odd function
def IsOdd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

-- Define the solution set for a function being positive
def PositiveSolutionSet (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, h x > 0 ↔ a < x ∧ x < b

-- State the theorem
theorem product_positive_solution_set
  (hf_odd : IsOdd f)
  (hg_odd : IsOdd g)
  (hf_pos : PositiveSolutionSet f 4 10)
  (hg_pos : PositiveSolutionSet g 2 5) :
  ∀ x, f x * g x > 0 ↔ (4 < x ∧ x < 5) ∨ (-5 < x ∧ x < -4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_positive_solution_set_l327_32711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_8_l327_32729

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Add this case for n = 0
  | 1 => 2
  | (n + 1) => sequence_a n + n

theorem a_4_equals_8 : sequence_a 4 = 8 := by
  -- Expand the definition of sequence_a
  unfold sequence_a
  -- Reduce the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_8_l327_32729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_integral_part_a_limit_integral_part_b_l327_32735

/-- The limit of n∫₀¹((1-x)/(1+x))ⁿdx as n approaches infinity is 1/2 -/
theorem limit_integral_part_a : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |n * ∫ (x : ℝ) in Set.Icc 0 1, ((1 - x) / (1 + x))^n - (1/2)| < ε :=
sorry

/-- For k ≥ 1, the limit of n^(k+1)∫₀¹((1-x)/(1+x))ⁿxᵏdx as n approaches infinity is k!/(2^(k+1)) -/
theorem limit_integral_part_b (k : ℕ) (hk : k ≥ 1) : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |n^(k+1) * ∫ (x : ℝ) in Set.Icc 0 1, ((1 - x) / (1 + x))^n * x^k - (k.factorial / 2^(k+1))| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_integral_part_a_limit_integral_part_b_l327_32735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cube_difference_divisible_by_91_l327_32782

theorem perfect_cube_difference_divisible_by_91 (S : Finset ℕ) 
  (h_card : S.card = 16) 
  (h_cubes : ∀ n, n ∈ S → ∃ m : ℕ, n = m^3) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ 91 ∣ (a - b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cube_difference_divisible_by_91_l327_32782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l327_32784

def number_list : List Nat := [34, 37, 39, 41, 43]

def is_prime (n : Nat) : Bool :=
  n > 1 && (Nat.factors n).length == 1

def prime_numbers : List Nat := number_list.filter is_prime

theorem arithmetic_mean_of_primes :
  (prime_numbers.sum : Rat) / prime_numbers.length = 121 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l327_32784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_minus_cot_zero_l327_32765

theorem tan_minus_cot_zero (θ : Real) (h : Real.sin θ * Real.cos θ = 1/2) : 
  Real.tan θ - (Real.cos θ / Real.sin θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_minus_cot_zero_l327_32765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_area_circumference_ratio_l327_32730

-- Define the circle properties
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

-- State the theorem
theorem circle_radius_from_area_circumference_ratio 
  (M N : ℝ) (h_positive_M : M > 0) (h_positive_N : N > 0) 
  (h_ratio : M / N = 20) :
  ∃ r : ℝ, r > 0 ∧ circle_area r = M ∧ circle_circumference r = N ∧ r = 40 := by
  sorry

#check circle_radius_from_area_circumference_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_area_circumference_ratio_l327_32730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l327_32752

/-- The repeating decimal 0.363636... as a real number -/
noncomputable def repeating_decimal : ℚ := 4 / 11

/-- The theorem stating that the repeating decimal 0.363636... is equal to 4/11 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 4 / 11 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l327_32752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_specific_area_l327_32717

def right_triangle_area (x y z : ℝ) : Prop :=
  ∃ (angle_y angle_z : ℝ),
    -- XYZ is a right triangle
    angle_y + angle_z = Real.pi / 2 ∧
    -- Angle Y = Angle Z
    angle_y = angle_z ∧
    -- XY = 8√2
    x = 8 * Real.sqrt 2 ∧
    -- Area of triangle XYZ is 64
    (1 / 2) * x * y = 64

theorem right_triangle_specific_area :
  ∃ (x y z : ℝ), right_triangle_area x y z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_specific_area_l327_32717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_and_uniqueness_l327_32795

/-- A polynomial of degree n satisfying P(x - 1/x) = x^n - 1/x^n -/
def SatisfyingPolynomial (P : Polynomial ℝ) (n : ℕ) : Prop :=
  (P.degree = n) ∧ 
  ∀ x : ℝ, x ≠ 0 → P.eval (x - 1/x) = x^n - 1/x^n

theorem polynomial_existence_and_uniqueness :
  ∀ n : ℕ, 
    Odd n → 
    ∃! P : Polynomial ℝ, SatisfyingPolynomial P n ∧
    ∀ m : ℕ, Even m → ¬∃ Q : Polynomial ℝ, SatisfyingPolynomial Q m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_and_uniqueness_l327_32795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_eight_l327_32745

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | (n+2) => fibonacci (n+1) + fibonacci n

theorem fifth_term_is_eight : fibonacci 4 = 8 := by
  rfl

#eval fibonacci 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_eight_l327_32745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_90_l327_32703

theorem sum_of_factors_90 : (Finset.filter (λ x => 90 % x = 0) (Finset.range 91)).sum id = 234 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_90_l327_32703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_sqrt8_and_3pi_l327_32706

theorem whole_numbers_between_sqrt8_and_3pi :
  (Finset.range (Int.toNat (⌊3 * Real.pi⌋ - ⌈Real.sqrt 8⌉ + 1))).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_sqrt8_and_3pi_l327_32706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l327_32767

theorem log_inequality_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : Real.log (2/3) / Real.log a < 1) : a ∈ Set.union (Set.Ioo 0 (2/3)) (Set.Ioi 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l327_32767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_bounded_neg_reals_a_range_when_f_bounded_l327_32785

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + a * (1/2)^x + (1/4)^x

-- Part 1: f(x) is not bounded on (-∞, 0) when a = 1
theorem f_not_bounded_neg_reals : 
  ¬ ∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), x < 0 → |f 1 x| ≤ M :=
by sorry

-- Part 2: Range of a when f(x) is bounded by 3 on [0, +∞)
theorem a_range_when_f_bounded (a : ℝ) :
  (∀ (x : ℝ), x ≥ 0 → |f a x| ≤ 3) → -5 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_bounded_neg_reals_a_range_when_f_bounded_l327_32785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_decreasing_l327_32791

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add a case for 0
  | 1 => 1
  | n + 1 => sequence_a n / (3 * sequence_a n + 1)

theorem sequence_a_decreasing : ∀ n : ℕ, n ≥ 1 → sequence_a (n + 1) < sequence_a n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_decreasing_l327_32791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_pens_two_holders_l327_32754

theorem seven_pens_two_holders : 
  (Finset.sum {2, 3, 4, 5} (fun k => Nat.choose 7 k)) = 112 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_pens_two_holders_l327_32754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l327_32797

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given triangle ABC with specific conditions -/
noncomputable def triangleABC : Triangle where
  a := 1
  b := 2
  c := Real.sqrt 2
  A := Real.arcsin (Real.sqrt 14 / 8)
  B := Real.pi - Real.arcsin (Real.sqrt 14 / 8) - Real.arccos (3/4)
  C := Real.arccos (3/4)

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Theorem stating the perimeter of triangle ABC and the value of sin A -/
theorem triangle_properties (t : Triangle) (h : t = triangleABC) : 
  perimeter t = 3 + Real.sqrt 2 ∧ Real.sin t.A = Real.sqrt 14 / 8 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l327_32797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_translation_on_sine_graphs_l327_32716

theorem point_translation_on_sine_graphs (t s : ℝ) :
  s > 0 →
  t = Real.sin (2 * (π / 4) - π / 3) →
  Real.sin (2 * (π / 4 - s)) = t →
  t = 1 / 2 ∧ ∃ (k : ℤ), s = π / 6 + k * π ∧ ∀ (m : ℤ), s ≤ π / 6 + m * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_translation_on_sine_graphs_l327_32716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_area_original_area_when_projection_equilateral_l327_32796

-- Define the side length of the triangle
variable (a : ℝ)

-- Define the area ratio between projection and original
noncomputable def area_ratio : ℝ := Real.sqrt 2 / 4

-- Define the area of an equilateral triangle
noncomputable def equilateral_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2

-- Theorem 1: Area of the projection
theorem projection_area (h : a > 0) :
  equilateral_area a * area_ratio = (Real.sqrt 6 / 16) * a^2 := by sorry

-- Theorem 2: Area of the original triangle when projection is equilateral
theorem original_area_when_projection_equilateral (h : a > 0) :
  equilateral_area a / area_ratio = (Real.sqrt 6 / 2) * a^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_area_original_area_when_projection_equilateral_l327_32796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_A_l327_32712

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4}

-- Define the complement of A with respect to U
def complement_A : Finset Nat := {2}

-- Define set A based on its complement
def A : Finset Nat := U \ complement_A

-- Theorem to prove
theorem number_of_subsets_A : (Finset.powerset A).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_A_l327_32712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_g_composed_l327_32736

-- Define the function g as noncomputable due to its dependency on Real.sqrt
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x - 2)

-- State the theorem using a different approach to avoid 'isDefined'
theorem smallest_x_in_domain_of_g_composed :
  ∃ x : ℝ, x = 18 ∧ 
    (∀ y : ℝ, (y - 2 ≥ 0 ∧ (Real.sqrt (y - 2) - 2) ≥ 0) → x ≤ y) ∧
    (x - 2 ≥ 0 ∧ (Real.sqrt (x - 2) - 2) ≥ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_g_composed_l327_32736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_logarithms_l327_32728

theorem sum_of_logarithms (a b : ℝ) (h1 : (10 : ℝ)^a = 5) (h2 : (10 : ℝ)^b = 2) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_logarithms_l327_32728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorial_divisibility_l327_32749

def factorial (n : ℕ) : ℕ := Nat.factorial n

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_factorial_divisibility : 
  ∀ x : ℕ, x < 25 → ¬(is_divisible (factorial 10 - 2 * (factorial x)^2) (10^5)) ∧ 
  is_divisible (factorial 10 - 2 * (factorial 25)^2) (10^5) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorial_divisibility_l327_32749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_separating_equidistant_plane_l327_32779

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Function to check if a plane separates two sets of points
def separates (p : Plane3D) (set1 set2 : List Point3D) : Prop :=
  ∀ pt1 pt2, pt1 ∈ set1 → pt2 ∈ set2 →
    (p.a * pt1.x + p.b * pt1.y + p.c * pt1.z + p.d) *
    (p.a * pt2.x + p.b * pt2.y + p.c * pt2.z + p.d) ≤ 0

-- Function to check if all points are equidistant from a plane
def allEquidistant (p : Plane3D) (points : List Point3D) : Prop :=
  ∀ pt1 pt2, pt1 ∈ points → pt2 ∈ points →
    |p.a * pt1.x + p.b * pt1.y + p.c * pt1.z + p.d| =
    |p.a * pt2.x + p.b * pt2.y + p.c * pt2.z + p.d|

-- Theorem statement
theorem exists_separating_equidistant_plane
  (A B C D E F : Point3D) :
  ∃ (p : Plane3D),
    separates p [A, B, C] [D, E, F] ∧
    allEquidistant p [A, B, C, D, E, F] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_separating_equidistant_plane_l327_32779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l327_32743

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * Real.cos x - 1)

theorem domain_of_f :
  {x : ℝ | ∃ k : ℤ, -π/3 + 2*π*↑k ≤ x ∧ x ≤ π/3 + 2*π*↑k} =
  {x : ℝ | f x ∈ Set.range Real.sqrt} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l327_32743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_dissection_into_trapezoids_l327_32713

/-- A dissection of a square into trapezoids -/
structure SquareDissection where
  /-- The number of trapezoids in the dissection -/
  num_trapezoids : ℕ
  /-- Predicate that checks if a given trapezoid is isosceles -/
  is_isosceles : (trapezoid : Set (EuclideanSpace ℝ (Fin 2))) → Prop

/-- Theorem stating that a square can be divided into 12 isosceles trapezoids -/
theorem square_dissection_into_trapezoids :
  ∃ (d : SquareDissection), d.num_trapezoids = 12 ∧
  ∀ (t : Set (EuclideanSpace ℝ (Fin 2))), d.is_isosceles t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_dissection_into_trapezoids_l327_32713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l327_32722

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The gradient of a line through the origin and a point -/
noncomputable def lineGradient (p : ℝ × ℝ) : ℝ :=
  p.2 / p.1

theorem ellipse_area_theorem (e : Ellipse) 
    (h_ecc : eccentricity e = Real.sqrt 3 / 2)
    (h_vertex : ∃ (v : PointOnEllipse e), v.y = -1)
    (p q : PointOnEllipse e)
    (h_gradients : lineGradient (p.x, p.y) * lineGradient (q.x, q.y) = -1/4) :
    ∃ (area : ℝ), area = 1 ∧ 
    area = Real.sqrt ((p.x * q.y - p.y * q.x)^2 / 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l327_32722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_equations_represent_line_l327_32775

/-- Parametric equations representing a line -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The line y = 2x -/
def line_y_eq_2x (x : ℝ) : ℝ := 2 * x

/-- Parametric equations for the line y = 2x -/
noncomputable def parametric_line_y_eq_2x : ParametricLine where
  x := λ θ => Real.tan θ
  y := λ θ => 2 * Real.tan θ

/-- Theorem stating that the parametric equations represent the line y = 2x -/
theorem parametric_equations_represent_line :
  ∀ θ, line_y_eq_2x (parametric_line_y_eq_2x.x θ) = parametric_line_y_eq_2x.y θ :=
by
  intro θ
  simp [line_y_eq_2x, parametric_line_y_eq_2x]
  -- The proof is completed by simplification
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_equations_represent_line_l327_32775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_part_a_remainder_theorem_part_b_l327_32787

def f (x : ℝ) : ℝ := x^243 + x^81 + x^27 + x^9 + x^3 + 1

theorem remainder_theorem_part_a : 
  ∃ q : ℝ → ℝ, f = λ x ↦ (x - 1) * q x + 6 := by sorry

theorem remainder_theorem_part_b : 
  ∃ q : ℝ → ℝ, f = λ x ↦ (x^2 - 1) * q x + (5 * x + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_part_a_remainder_theorem_part_b_l327_32787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_team_permutations_l327_32777

theorem relay_team_permutations : 
  ∀ (n : ℕ), n = 6 → (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_team_permutations_l327_32777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l327_32769

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, x^2 < 5*x - 6 → |x + 1| ≤ 4) ∧
  (∃ x : ℝ, |x + 1| ≤ 4 ∧ ¬(x^2 < 5*x - 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l327_32769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l327_32741

/-- Given two points (x₁, y₁) and (x₂, y₂) satisfying the conditions,
    prove that the minimum squared distance between them is 8/5 -/
theorem min_distance_squared (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : (Real.exp x₁ + 2 * x₁) / (3 * y₁) = 1/3)
  (h2 : (x₂ - 1) / y₂ = 1/3) :
  (∀ a b c d, (Real.exp a + 2 * a) / (3 * c) = 1/3 → (b - 1) / d = 1/3 → 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 ≤ (a - b)^2 + (c - d)^2) →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l327_32741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_divisible_into_8_trapezoids_l327_32708

/-- A non-rectangular trapezoid -/
structure NonRectangularTrapezoid where
  -- Add necessary fields here
  nonRectangular : Bool

/-- A square -/
structure Square where
  sideLength : ℝ
  positive : sideLength > 0

/-- A division of a square into shapes -/
structure SquareDivision (n : ℕ) where
  square : Square
  shapes : Fin n → NonRectangularTrapezoid
  -- Add necessary fields to ensure the shapes form a valid division

/-- Theorem: A square can be divided into 8 non-rectangular trapezoids -/
theorem square_divisible_into_8_trapezoids :
  ∃ (s : Square), ∃ (d : SquareDivision 8), true := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_divisible_into_8_trapezoids_l327_32708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_geometric_sequence_l327_32778

theorem cosine_geometric_sequence (a : ℝ) : 
  0 < a ∧ a < 2 * Real.pi →
  (∃ k : ℝ, Real.cos (2 * a) = k * Real.cos a ∧ Real.cos (3 * a) = k * Real.cos (2 * a)) ↔
  (a = Real.pi / 4 ∨ a = 3 * Real.pi / 4 ∨ a = 5 * Real.pi / 4 ∨ a = 7 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_geometric_sequence_l327_32778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_password_last_digits_l327_32774

def fibonacci_sequence : List Nat := [1, 1, 2, 3, 5, 8, 13]

def is_fibonacci (seq : List Nat) : Prop :=
  seq.length ≥ 2 ∧ 
  ∀ i, 2 ≤ i → i < seq.length → seq[i]! = seq[i-1]! + seq[i-2]!

theorem password_last_digits 
  (given_seq : List Nat)
  (h1 : given_seq = fibonacci_sequence)
  (h2 : is_fibonacci given_seq) :
  (given_seq ++ [21]).length = 10 ∧ is_fibonacci (given_seq ++ [21]) := by
  sorry

#eval fibonacci_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_password_last_digits_l327_32774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_l327_32759

-- Define the number of small circles
def n : ℕ := 7

-- Define the radius of small circles
def r : ℝ := 4

-- Define the side length of the heptagon formed by the centers of small circles
def s : ℝ := 2 * r

-- Define the circumradius of the heptagon
noncomputable def R_heptagon : ℝ := s / (2 * Real.sin (Real.pi / n))

-- Define the radius of the large circle
noncomputable def R : ℝ := R_heptagon + r

-- Theorem: The diameter of the large circle
theorem large_circle_diameter :
  2 * R = 2 * (s / (2 * Real.sin (Real.pi / n)) + r) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_l327_32759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scatter_plot_definition_l327_32742

/-- A scatter plot is a graph that represents a set of data for two variables with a correlation. -/
def scatter_plot : String := "A graph that represents a set of data for two variables with a correlation"

theorem scatter_plot_definition : 
  scatter_plot = "A graph that represents a set of data for two variables with a correlation" := 
by rfl

#check scatter_plot_definition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scatter_plot_definition_l327_32742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_length_l327_32750

/-- The side length of a regular hexagon -/
def HexagonSideLength (hexagon : Set (ℝ × ℝ)) : ℝ :=
sorry

/-- Predicate for a set of points forming a regular polygon with n sides -/
def IsRegularPolygon (s : Set (ℝ × ℝ)) (n : ℕ) : Prop :=
sorry

/-- Given a triangle with sides a, b, and c, and three lines parallel to the sides of the triangle
    that cut off three smaller triangles leaving a regular hexagon, the side length of the hexagon
    is abc / (ab + bc + ac) -/
theorem hexagon_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let x := (a * b * c) / (a * b + b * c + a * c)
  ∃ (hexagon : Set (ℝ × ℝ)), IsRegularPolygon hexagon 6 ∧ HexagonSideLength hexagon = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_length_l327_32750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_distance_before_karen_wins_l327_32731

/-- The distance Tom drives before Karen wins the bet -/
noncomputable def tom_distance (karen_speed tom_speed : ℝ) (karen_delay : ℝ) (winning_margin : ℝ) : ℝ :=
  let tom_speed_per_minute := tom_speed / 60
  let karen_speed_per_minute := karen_speed / 60
  let head_start := tom_speed_per_minute * karen_delay
  let catch_up_distance := head_start + winning_margin
  let catch_up_time := catch_up_distance / (karen_speed_per_minute - tom_speed_per_minute)
  tom_speed_per_minute * catch_up_time

/-- Theorem stating that Tom drives 5.25 miles before Karen wins the bet -/
theorem tom_distance_before_karen_wins :
  tom_distance 60 45 4 4 = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_distance_before_karen_wins_l327_32731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_third_l327_32781

theorem tan_alpha_plus_pi_third (α : Real) 
  (h1 : Real.sin (2 * α) = Real.cos α) 
  (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : 
  Real.tan (α + Real.pi / 3) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_third_l327_32781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_log_function_l327_32725

theorem monotonic_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc (3/2 : ℝ) 2, StrictMono (fun x => Real.log (6*a*x^2 - 2*x + 3) / Real.log a)) ↔
  (a ∈ Set.Ioo (1/24 : ℝ) (1/12) ∪ Set.Ioi 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_log_function_l327_32725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gross_profit_percentage_sleeping_bag_gross_profit_percentage_l327_32737

/-- Calculate the gross profit percentage given the selling price and wholesale cost -/
theorem gross_profit_percentage (selling_price wholesale_cost : ℝ) :
  let gross_profit := selling_price - wholesale_cost
  (gross_profit / wholesale_cost) * 100 = (selling_price - wholesale_cost) / wholesale_cost * 100 := by
  sorry

/-- The gross profit percentage for the sleeping bag sale -/
theorem sleeping_bag_gross_profit_percentage :
  let selling_price : ℝ := 28
  let wholesale_cost : ℝ := 23.93
  let gross_profit_percentage := (selling_price - wholesale_cost) / wholesale_cost * 100
  ∃ (ε : ℝ), ε > 0 ∧ |gross_profit_percentage - 17.004| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gross_profit_percentage_sleeping_bag_gross_profit_percentage_l327_32737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l327_32732

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Helper function to calculate triangle area -/
noncomputable def triangle_area (t : Triangle) : ℝ := 
  (1 / 2) * t.a * t.c * Real.sin t.B

/-- Main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h_order : t.a < t.b ∧ t.b < t.c)
  (h_sin_A : Real.sin t.A = (Real.sqrt 3 * t.a) / (2 * t.b)) :
  t.B = π / 3 ∧ 
  (t.a = 2 ∧ t.b = Real.sqrt 7 → t.c = 3) ∧
  (t.a = 2 ∧ t.b = Real.sqrt 7 → triangle_area t = 3 * Real.sqrt 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l327_32732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l327_32739

theorem divisor_problem :
  ∃ D : ℕ,
    D > 0 ∧
    242 % D = 15 ∧
    698 % D = 27 ∧
    415 % D = 12 ∧
    (242 + 698 + 415) % D = 5 ∧
    D = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l327_32739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_and_q_sufficient_not_necessary_l327_32776

theorem p_and_q_sufficient_not_necessary (p q : Prop) :
  (∃ (h : p ∧ q), ¬¬p) ∧
  ¬(∀ (h : ¬¬p), p ∧ q) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_and_q_sufficient_not_necessary_l327_32776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_in_third_quadrant_fraction_value_with_tan_l327_32714

noncomputable def is_in_third_quadrant (α : Real) : Prop :=
  Real.pi < α ∧ α < 3 * Real.pi / 2

theorem sin_value_in_third_quadrant (α : Real) 
  (h1 : Real.cos α = -4/5) (h2 : is_in_third_quadrant α) : 
  Real.sin α = -3/5 := by sorry

theorem fraction_value_with_tan (α : Real) (h : Real.tan α = -3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 7/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_in_third_quadrant_fraction_value_with_tan_l327_32714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l327_32758

noncomputable def f (x : ℝ) : ℝ := (Real.sin x * Real.sqrt (1 - abs x)) / (abs (x + 2) - 2)

def domain (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1 ∧ x ≠ 0

theorem f_is_odd : ∀ x, domain x → f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l327_32758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_intersection_not_always_parallel_l327_32744

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define the planes and lines
variable (α β : Plane)
variable (m n : Line)

-- State the theorem
theorem parallel_intersection_not_always_parallel :
  ¬(∀ (α β : Plane) (m n : Line),
    (parallel_line_plane m α) ∧ (intersect α β n) → (parallel_line m n)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_intersection_not_always_parallel_l327_32744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l327_32747

/-- An arithmetic sequence with a non-zero first term -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_nonzero : a 1 ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) 
  (geometric_sums : (sum_arithmetic seq 2) ^ 2 = (sum_arithmetic seq 1) * (sum_arithmetic seq 4)) :
  seq.a 2 / seq.a 1 = 1 ∨ seq.a 2 / seq.a 1 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l327_32747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l327_32760

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ Real.sqrt (2*x - x^2)

-- State the theorem
theorem f_monotone_increasing :
  ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x < y → f x < f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l327_32760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_phone_chargers_l327_32700

/-- The number of phone chargers Anna has -/
def phone_chargers : ℕ := sorry

/-- The number of laptop chargers Anna has -/
def laptop_chargers : ℕ := sorry

/-- The total number of chargers Anna has -/
def total_chargers : ℕ := sorry

/-- Laptop chargers are five times the number of phone chargers -/
axiom laptop_chargers_def : laptop_chargers = 5 * phone_chargers

/-- The total number of chargers is 24 -/
axiom total_chargers_def : total_chargers = 24

/-- The total number of chargers is the sum of phone and laptop chargers -/
axiom total_chargers_sum : total_chargers = phone_chargers + laptop_chargers

/-- Theorem: Anna has 4 phone chargers -/
theorem anna_phone_chargers : phone_chargers = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_phone_chargers_l327_32700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_value_l327_32755

-- Define the sequence G
def G : ℕ → ℚ
  | 0 => 3  -- Add a case for 0 to cover all natural numbers
  | 1 => 3
  | (n + 2) => (3 * G (n + 1) + 2) / 3

-- Theorem statement
theorem G_51_value : G 51 = 109 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_value_l327_32755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l327_32715

/-- Given a hyperbola with equation x²/a² - y² = 1, where one of its asymptotes
    is perpendicular to the line y = -x + 1, prove that its focal length is 2√2. -/
theorem hyperbola_focal_length (a : ℝ) :
  (∃ (x y : ℝ), x^2/a^2 - y^2 = 1) →  -- Hyperbola equation
  (∃ (m : ℝ), m * (-1) = -1 ∧ (m = 1/a ∨ m = -1/a)) →  -- Asymptote perpendicular to y = -x + 1
  2 * Real.sqrt (a^2 + 1) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l327_32715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_point_l327_32772

theorem sin_plus_cos_for_point (α : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ r * Real.cos α = 3 ∧ r * Real.sin α = -4) → 
  Real.sin α + Real.cos α = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_point_l327_32772

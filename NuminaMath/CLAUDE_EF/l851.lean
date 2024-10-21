import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_product_l851_85100

theorem solution_product : 
  ∃ a b : ℝ, (|a - 1| = 3 * (|a - 1| - 2) ∧ |b - 1| = 3 * (|b - 1| - 2) ∧ a ≠ b) ∧ a * b = -8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_product_l851_85100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_half_minus_alpha_l851_85132

-- Define the angle α
noncomputable def α : ℝ := Real.arctan (3 / -4)

-- Define the theorem
theorem cos_three_pi_half_minus_alpha :
  Real.cos (3 * π / 2 - α) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_half_minus_alpha_l851_85132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_succ_l851_85123

open BigOperators

def f (n : ℕ) : ℚ := ∑ i in Finset.range (2*n+1), 1 / ((n : ℚ) + (i : ℚ) + 1)

theorem f_succ (k : ℕ) :
  f (k + 1) = f k + 1 / (3*(k : ℚ) + 2) + 1 / (3*(k : ℚ) + 3) + 1 / (3*(k : ℚ) + 4) - 1 / ((k : ℚ) + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_succ_l851_85123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_difference_properties_l851_85170

-- Theorem statement
theorem symmetric_difference_properties {α : Type} (A B C : Set α) :
  ((A \ B ∪ B \ A) \ C ∪ C \ (A \ B ∪ B \ A) = A \ (B \ C ∪ C \ B) ∪ (B \ C ∪ C \ B) \ A) ∧
  (((A \ B ∪ B \ A) \ (B \ C ∪ C \ B) ∪ (B \ C ∪ C \ B) \ (A \ B ∪ B \ A)) = A \ C ∪ C \ A) ∧
  (A \ B ∪ B \ A = C ↔ A = B \ C ∪ C \ B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_difference_properties_l851_85170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_theorem_l851_85198

/-- Given a triangle with area A, side length a, and median length m to that side,
    prove that the sine of the angle θ between the side and the median is (5/6) -/
theorem triangle_sine_theorem (A a m : ℝ) (h1 : A = 24) (h2 : a = 8) (h3 : m = 7.2) :
  Real.sin (Real.arcsin ((2 * A) / (a * m))) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_theorem_l851_85198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_1125_l851_85161

noncomputable section

def square_side_length : ℝ := 40

def triangle1_base : ℝ := 10
def triangle1_height : ℝ := 10

def triangle2_base : ℝ := 15
def triangle2_height : ℝ := 30

def triangle3_base : ℝ := 20
def triangle3_height : ℝ := 20

def square_area : ℝ := square_side_length * square_side_length

def triangle1_area : ℝ := (1/2) * triangle1_base * triangle1_height
def triangle2_area : ℝ := (1/2) * triangle2_base * triangle2_height
def triangle3_area : ℝ := (1/2) * triangle3_base * triangle3_height

def total_unshaded_area : ℝ := triangle1_area + triangle2_area + triangle3_area

theorem shaded_area_is_1125 : 
  square_area - total_unshaded_area = 1125 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_1125_l851_85161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_AEDC_l851_85190

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the point P as the intersection of medians
noncomputable def P (t : Triangle) : ℝ × ℝ := sorry

-- Define the midpoints D and E
noncomputable def D (t : Triangle) : ℝ × ℝ := ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2)
noncomputable def E (t : Triangle) : ℝ × ℝ := ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2)

-- Define the distances
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area of a quadrilateral with perpendicular diagonals
noncomputable def quadrilateralArea (d1 d2 : ℝ) : ℝ := d1 * d2 / 2

-- Theorem statement
theorem area_of_AEDC (t : Triangle) :
  distance (P t) (E t) = 1.5 →
  distance (P t) (D t) = 2 →
  distance (D t) (E t) = 2.5 →
  quadrilateralArea (distance (t.A) (D t)) (distance (t.C) (E t)) = 13.5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_AEDC_l851_85190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_final_price_l851_85179

/-- Calculate the final price of a saree after discounts, tax, and handling fee -/
theorem saree_final_price (original_price : ℝ) (discount1 discount2 discount3 sales_tax handling_fee : ℝ) :
  original_price = 1200 ∧
  discount1 = 0.18 ∧
  discount2 = 0.12 ∧
  discount3 = 0.05 ∧
  sales_tax = 0.03 ∧
  handling_fee = 0.02 →
  abs ((original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)) * (1 + sales_tax + handling_fee) - 863.76) < 0.01 := by
  sorry

#check saree_final_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_final_price_l851_85179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_is_one_intersection_equals_domain_iff_l851_85154

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (3 - abs (x - 1))

-- Define the domain A of f
def A : Set ℝ := {x | 3 - abs (x - 1) > 0}

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 5) * x + 5 * a < 0}

-- Theorem 1: A ∩ B = (1, 4) when a = 1
theorem intersection_when_a_is_one : A ∩ B 1 = Set.Ioo 1 4 := by sorry

-- Theorem 2: A ∩ B = A if and only if a ≤ -2
theorem intersection_equals_domain_iff (a : ℝ) : A ∩ B a = A ↔ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_is_one_intersection_equals_domain_iff_l851_85154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_change_difference_is_40_percent_l851_85139

/-- Represents the percentages of student responses --/
structure ResponsePercentages where
  yes : ℝ
  no : ℝ
  unsure : ℝ
  sum_to_100 : yes + no + unsure = 100

/-- The given initial response percentages --/
def initial : ResponsePercentages := {
  yes := 40,
  no := 40,
  unsure := 20,
  sum_to_100 := by norm_num
}

/-- The given final response percentages --/
def final : ResponsePercentages := {
  yes := 60,
  no := 20,
  unsure := 20,
  sum_to_100 := by norm_num
}

/-- The minimum percentage of students who changed their response --/
noncomputable def min_change : ℝ := max (final.yes - initial.yes) (initial.no - final.no)

/-- The maximum percentage of students who changed their response --/
noncomputable def max_change : ℝ := min (initial.no + initial.unsure) (final.yes + final.no)

/-- The theorem stating the difference between max and min change is 40% --/
theorem change_difference_is_40_percent : max_change - min_change = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_change_difference_is_40_percent_l851_85139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_range_l851_85157

open Real

-- Define the line l
noncomputable def line (k : ℝ) (x : ℝ) : ℝ := k * x - 1

-- Define the curve C
noncomputable def curve (x : ℝ) : ℝ := x - 1 + 1 / (exp x)

-- Theorem statement
theorem no_common_points_range (k : ℝ) :
  (∀ x : ℝ, line k x ≠ curve x) → k ∈ Set.Ioo (1 - exp 1) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_range_l851_85157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l851_85181

-- Define the line
def line (x y : ℝ) : Prop := x + y = 1

-- Define the circle
def circle' (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Theorem statement
theorem intersection_points_count :
  ∃ (p1 p2 : ℝ × ℝ),
    p1 ≠ p2 ∧
    line p1.1 p1.2 ∧ circle' p1.1 p1.2 ∧
    line p2.1 p2.2 ∧ circle' p2.1 p2.2 ∧
    ∀ (p : ℝ × ℝ), line p.1 p.2 ∧ circle' p.1 p.2 → p = p1 ∨ p = p2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l851_85181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_theorem_l851_85197

noncomputable def triangle_abc_problem (A B C : ℝ) (a b c : ℝ) : Prop :=
  let S := (1/2) * a * b * Real.sin C
  (c * Real.sin A = Real.sqrt 3 * a * Real.cos C) ∧
  (c = Real.sqrt 7) ∧
  (Real.sin C + Real.sin (B - A) = 3 * Real.sin (2 * A)) ∧
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

theorem triangle_abc_theorem (A B C : ℝ) (a b c : ℝ) 
  (h : triangle_abc_problem A B C a b c) :
  C = Real.pi/3 ∧ 
  (let S := (1/2) * a * b * Real.sin C
   S = (7 * Real.sqrt 3) / 6 ∨ S = (3 * Real.sqrt 3) / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_theorem_l851_85197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_equality_l851_85142

/-- Given a triangle with sides a, b, and c, where the angle opposite to side a is 60 degrees,
    prove that a² = (a³ + b³ + c³) / (a + b + c) --/
theorem triangle_side_equality (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
    (h_angle : Real.cos (π / 3) = (b^2 + c^2 - a^2) / (2*b*c)) :
  a^2 = (a^3 + b^3 + c^3) / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_equality_l851_85142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_from_asymptotes_and_focus_l851_85110

/-- A hyperbola with given asymptotes and focus -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  k : ℝ
  /-- The x-coordinate of the focus -/
  a : ℝ

/-- The equation of a hyperbola given its asymptotes and focus -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y ↦ x^2 - y^2 / h.k^2 = 1

/-- Theorem: For a hyperbola with asymptotes y = ±3x and focus (√10, 0), 
    its equation is x² - y²/9 = 1 -/
theorem hyperbola_equation_from_asymptotes_and_focus :
  let h : Hyperbola := ⟨3, Real.sqrt 10⟩
  ∀ x y, hyperbola_equation h x y ↔ x^2 - y^2/9 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_from_asymptotes_and_focus_l851_85110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_length_l851_85116

theorem ladder_length 
  (w : ℝ) (k : ℝ) 
  (h_w : w = 10) 
  (h_k : k = 5) : 
  ∃ (a : ℝ), 
    a = 10 ∧ 
    Real.sin (30 * π / 180) = k / a ∧ 
    Real.cos (30 * π / 180) * a = w ∧
    Real.sin (60 * π / 180) * a = w := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_length_l851_85116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_minus_alpha_l851_85117

theorem tan_pi_fourth_minus_alpha (α : Real) 
  (h1 : Real.cos α = -3/5) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.tan (π/4 - α) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_minus_alpha_l851_85117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitudes_l851_85137

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitudes (a b : V) 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖b‖ = 1) 
  (h3 : ‖a - 3 • b‖ = Real.sqrt 13) : 
  ‖a - b‖ = Real.sqrt 3 ∧ ‖a + 3 • b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitudes_l851_85137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l851_85194

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x - (1/3) * x^3

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  (f (2*x + 3) + f 1 < 0) ↔ (x > -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l851_85194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l851_85184

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a
noncomputable def g (x : ℝ) : ℝ := Real.log x - 2*x

-- Define the interval [1/2, 2]
def I : Set ℝ := Set.Icc (1/2) 2

-- State the theorem
theorem function_inequality (a : ℝ) :
  (∃ x₁ ∈ I, ∀ x₂ ∈ I, f a x₁ ≤ g x₂) →
  a ≤ Real.log 2 - 21/4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l851_85184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l851_85124

/-- The complex number z -/
noncomputable def z : ℂ := Complex.I / (1 - Complex.I)

/-- The real part of z -/
noncomputable def real_part : ℝ := z.re

/-- The imaginary part of z -/
noncomputable def imag_part : ℝ := z.im

/-- Theorem: The point corresponding to z is in the second quadrant -/
theorem z_in_second_quadrant : real_part < 0 ∧ imag_part > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l851_85124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_cheese_amount_l851_85172

/-- Amount of cheese the fox got from both crows -/
noncomputable def fox_total_cheese (x : ℝ) : ℝ :=
  (100 - x) + (200 - x/2)

/-- Theorem stating the total amount of cheese the fox got -/
theorem fox_cheese_amount : ∃ x : ℝ, 
  x > 0 ∧ 
  x < 100 ∧
  200 - x/2 = 3*(100 - x) ∧
  fox_total_cheese x = 240 :=
by
  -- We'll use 40 as the value of x
  use 40
  constructor
  · -- Prove x > 0
    norm_num
  constructor
  · -- Prove x < 100
    norm_num
  constructor
  · -- Prove 200 - x/2 = 3*(100 - x)
    ring
  · -- Prove fox_total_cheese x = 240
    unfold fox_total_cheese
    ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_cheese_amount_l851_85172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportion_of_boys_l851_85144

/-- A family in the population -/
structure Family where
  boys : ℕ
  girl : Unit

/-- The probability of having a boy or a girl -/
noncomputable def birth_probability : ℝ := 1/2

/-- The expected number of boys in a family -/
noncomputable def expected_boys : ℝ := 2

/-- The expected total number of children in a family -/
noncomputable def expected_total : ℝ := 3

/-- The theorem stating the proportion of boys in the population -/
theorem proportion_of_boys :
  expected_boys / expected_total = 2/3 := by
  sorry

#eval "Proportion of boys theorem defined"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportion_of_boys_l851_85144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l851_85136

def my_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ a 2 = 2 ∧
  ∀ m n : ℕ, m > 0 → n > 0 → a (2*m - 1) + a (2*n - 1) = 2 * a (m + n - 1) + 2 * (m - n)^2

theorem sequence_properties (a : ℕ → ℕ) (h : my_sequence a) :
  a 3 = 6 ∧ a 4 = 12 ∧ a 5 = 20 ∧
  (∀ n : ℕ, n > 0 → a n = n * (n - 1)) ∧
  ¬ (∃ p q r : ℕ, p > 0 ∧ q > 0 ∧ r > 0 ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (q - p = r - q) ∧ (a q - a p = a r - a q)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l851_85136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_30_value_l851_85113

/-- The sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The property that S_n is the sum of the first n terms of an arithmetic sequence -/
axiom S_is_arithmetic_sum : ∀ n : ℕ, ∃ a d : ℝ, S n = n * (2 * a + (n - 1) * d) / 2

/-- Given conditions -/
axiom S_10 : S 10 = 31
axiom S_20 : S 20 = 122

/-- Theorem to prove -/
theorem S_30_value : S 30 = 273 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_30_value_l851_85113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_product_l851_85138

def number_of_digits : Nat := 65

def eights : Nat := (10^number_of_digits - 1) / 9 * 8

def sevens : Nat := (10^number_of_digits - 1) / 9 * 7

def product : Nat := eights * sevens

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_product : 
  ∃ (X : Nat), X ∈ ({1015, 1300, 1500, 1675, 1980} : Finset Nat) ∧ sum_of_digits product = X :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_product_l851_85138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_alpha_l851_85127

theorem cosine_of_alpha (α β : ℝ) (h1 : 0 < α ∧ α < Real.pi/2) (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.cos (α + β) = -3/5) (h4 : Real.sin β = 12/13) : Real.cos α = 33/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_alpha_l851_85127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_for_all_n_l851_85186

theorem composite_for_all_n : ∃ k : ℕ+, ∀ n : ℕ, ∃ m : ℕ, m > 1 ∧ m ∣ (k.val * 2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_for_all_n_l851_85186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_classification_l851_85148

noncomputable def A1 : Set ℝ := {1, 4, 9}
def B1 : Set ℝ := {-3, -2, -1, 1, 2, 3}
noncomputable def f1 : ℝ → ℝ := λ x => Real.sqrt x

def A2 : Set ℝ := Set.univ
def B2 : Set ℝ := Set.univ
noncomputable def f2 : ℝ → ℝ := λ x => 1 / x

def A3 : Set ℝ := Set.univ
def B3 : Set ℝ := Set.univ
def f3 : ℝ → ℝ := λ x => x^2 - 2

theorem function_classification :
  (∀ x, x ∈ A1 → f1 x ∈ B1 ∧ ∀ y, y ∈ A1 → f1 x = f1 y → x = y) ∧
  ¬(∀ x, x ∈ A2 → f2 x ∈ B2 ∧ ∀ y, y ∈ A2 → f2 x = f2 y → x = y) ∧
  (∀ x, x ∈ A3 → f3 x ∈ B3 ∧ ∀ y, y ∈ A3 → f3 x = f3 y → x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_classification_l851_85148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equidistant_point_l851_85115

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the moving points
structure MovingPoint where
  position : ℝ → ℝ × ℝ
  speed : ℝ

-- Define the problem setup
structure CircleIntersectionProblem where
  circle1 : Circle
  circle2 : Circle
  intersection_point : ℝ × ℝ
  point1 : MovingPoint
  point2 : MovingPoint

-- The theorem statement
theorem exists_equidistant_point
  (problem : CircleIntersectionProblem)
  (h1 : problem.circle1.center ≠ problem.circle2.center)
  (h2 : problem.intersection_point ∈ Set.range problem.point1.position)
  (h3 : problem.intersection_point ∈ Set.range problem.point2.position)
  (h4 : ∀ t, ‖problem.point1.position t - problem.circle1.center‖ = problem.circle1.radius)
  (h5 : ∀ t, ‖problem.point2.position t - problem.circle2.center‖ = problem.circle2.radius)
  (h6 : ∃ T > 0, problem.point1.position 0 = problem.point1.position T ∧ 
                 problem.point2.position 0 = problem.point2.position T) :
  ∃ B : ℝ × ℝ, ∀ t, ‖problem.point1.position t - B‖ = ‖problem.point2.position t - B‖ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equidistant_point_l851_85115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_curve_l851_85131

theorem point_on_curve : ∃ θ : ℝ, 
  Real.sin (2 * θ) = -3/4 ∧ Real.cos θ + Real.sin θ = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_curve_l851_85131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_fraction_l851_85146

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * a * x^2 + 2 * b * x + c

-- Define the theorem
theorem range_of_fraction (a b c : ℝ) (x₁ x₂ : ℝ) :
  (0 < x₁ ∧ x₁ < 1) →
  (1 < x₂ ∧ x₂ < 2) →
  (∀ x, x ≠ x₁ → f a b c x ≤ f a b c x₁) →
  (∀ x, x ≠ x₂ → f a b c x ≥ f a b c x₂) →
  (1/4 < (b - 2) / (a - 1) ∧ (b - 2) / (a - 1) < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_fraction_l851_85146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l851_85183

/-- The function f(x) = 5 - (x-1)^2/3 -/
noncomputable def f (x : ℝ) : ℝ := 5 - (x - 1)^2 / 3

/-- The intersection point of y = f(x) and y = f(x-4) -/
noncomputable def intersection_point : ℝ × ℝ := (5, 1/3)

theorem intersection_point_sum :
  f (intersection_point.fst) = f (intersection_point.fst - 4) ∧
  intersection_point.fst + intersection_point.snd = 16/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l851_85183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_formula_l851_85189

/-- The area of a rectangle with length (3x - 1) and width (2x + 1/2), where x > 0 -/
noncomputable def rectangleArea (x : ℝ) : ℝ :=
  (3 * x - 1) * (2 * x + 1/2)

/-- Theorem stating that the area of the rectangle is equal to 6x^2 - (1/2)x - 1/2 -/
theorem rectangle_area_formula (x : ℝ) (h : x > 0) :
  rectangleArea x = 6 * x^2 - (1/2) * x - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_formula_l851_85189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circles_area_ratio_l851_85125

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : ∀ i j : Fin 8, 
    dist (vertices i) (vertices ((i + 1) % 8)) = 
    dist (vertices j) (vertices ((j + 1) % 8))

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a circle is tangent to a line segment -/
def is_tangent_to_segment (c : Circle) (a b : ℝ × ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ Set.Icc a b ∧ dist p c.center = c.radius

/-- The theorem to be proved -/
theorem octagon_circles_area_ratio 
  (octagon : RegularOctagon)
  (circle1 circle2 : Circle) :
  is_tangent_to_segment circle1 (octagon.vertices 0) (octagon.vertices 1) →
  is_tangent_to_segment circle2 (octagon.vertices 4) (octagon.vertices 5) →
  is_tangent_to_segment circle1 (octagon.vertices 1) (octagon.vertices 2) →
  is_tangent_to_segment circle1 (octagon.vertices 7) (octagon.vertices 0) →
  is_tangent_to_segment circle2 (octagon.vertices 1) (octagon.vertices 2) →
  is_tangent_to_segment circle2 (octagon.vertices 7) (octagon.vertices 0) →
  (π * circle2.radius ^ 2) / (π * circle1.radius ^ 2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circles_area_ratio_l851_85125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_construction_possible_l851_85133

/-- A face diagonal of a cube -/
structure FaceDiagonal where
  line : Set (Fin 3 → ℝ)

/-- A point in 3D space -/
def Point3D := Fin 3 → ℝ

/-- A cube in 3D space -/
structure Cube where
  vertices : Fin 8 → Point3D

/-- Predicate to check if a line is a face diagonal of a cube -/
def isFaceDiagonal (c : Cube) (d : FaceDiagonal) : Prop := sorry

/-- Predicate to check if a point is an endpoint of a diagonal perpendicular to a given line in the parallel face -/
def isPerpendicularEndpoint (c : Cube) (d : FaceDiagonal) (p : Point3D) : Prop := sorry

/-- Theorem stating that a cube can be constructed given a face diagonal and a point -/
theorem cube_construction_possible (e : FaceDiagonal) (A : Point3D) 
  (h : A ∉ e.line) : 
  ∃ (c : Cube), isFaceDiagonal c e ∧ isPerpendicularEndpoint c e A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_construction_possible_l851_85133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_completion_l851_85196

/-- Given 4x^2 + 1, adding one of these monomials results in a perfect square trinomial -/
theorem perfect_square_completion (x : ℝ) : 
  ∃ (m : ℝ → ℝ), (m ∈ ({(λ x => 4*x^4), (λ x => 4*x), (λ x => -4*x), (λ _ => -1), (λ x => -4*x^2)} : Set (ℝ → ℝ))) ∧
  ∃ (a b : ℝ), (4*x^2 + 1 + m x = (a*x + b)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_completion_l851_85196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_valve_rate_difference_l851_85134

-- Define the pool capacity
def pool_capacity : ℚ := 12000

-- Define the time to fill the pool with both valves
def time_both_valves : ℚ := 48

-- Define the time to fill the pool with the first valve alone
def time_first_valve : ℚ := 120

-- Define the rate of the first valve
noncomputable def rate_first_valve : ℚ := pool_capacity / time_first_valve

-- Define the combined rate of both valves
noncomputable def rate_both_valves : ℚ := pool_capacity / time_both_valves

-- Define the rate of the second valve
noncomputable def rate_second_valve : ℚ := rate_both_valves - rate_first_valve

-- Theorem to prove
theorem second_valve_rate_difference : rate_second_valve - rate_first_valve = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_valve_rate_difference_l851_85134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_width_is_14_l851_85121

/-- The width of a rectangular courtyard -/
noncomputable def courtyard_width (length : ℝ) (num_bricks : ℕ) (brick_length : ℝ) (brick_width : ℝ) : ℝ :=
  (num_bricks : ℝ) * brick_length * brick_width / length

/-- Theorem stating the width of the courtyard is 14 meters -/
theorem courtyard_width_is_14 :
  courtyard_width 24 8960 0.25 0.15 = 14 := by
  -- Unfold the definition of courtyard_width
  unfold courtyard_width
  -- Simplify the expression
  simp [Nat.cast_mul, mul_assoc]
  -- Check that the resulting expression equals 14
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_width_is_14_l851_85121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_cos_sum_l851_85171

/-- The smallest positive angle θ, in degrees, such that 
    cos θ = sin 45° + cos 60° - sin 30° - cos 15° is 75° -/
theorem smallest_angle_cos_sum (θ : ℝ) : 
  (0 < θ) → 
  (θ ≤ 360) →
  (Real.cos (θ * π / 180) = Real.sin (45 * π / 180) + Real.cos (60 * π / 180) - 
   Real.sin (30 * π / 180) - Real.cos (15 * π / 180)) →
  θ = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_cos_sum_l851_85171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_G_range_l851_85103

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := m * x + 3
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

-- Define the function G
def G (m : ℝ) (x : ℝ) : ℝ := f m x - g m x - 1

-- State the theorem
theorem decreasing_G_range (m : ℝ) :
  (∀ x y, x ∈ Set.Icc (-1) 0 → y ∈ Set.Icc (-1) 0 → x < y → |G m y| < |G m x|) ↔ (m ≤ 0 ∨ m ≥ 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_G_range_l851_85103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_times_self_equals_75_l851_85182

theorem ceiling_times_self_equals_75 :
  ∃ x : ℝ, (⌈x⌉ : ℝ) * x = 75 ∧ x = 75 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_times_self_equals_75_l851_85182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_x_squared_plus_power_of_two_plus_one_equals_y_cubed_l851_85175

theorem no_solutions_x_squared_plus_power_of_two_plus_one_equals_y_cubed :
  ∀ (k : ℕ), ¬∃ (x y : ℕ), x ≠ 0 ∧ y ≠ 0 ∧ x^2 + 2^(2*k) + 1 = y^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_x_squared_plus_power_of_two_plus_one_equals_y_cubed_l851_85175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_900_by_20_percent_l851_85147

/-- Calculates the value after a percentage increase -/
noncomputable def valueAfterIncrease (originalValue : ℝ) (percentageIncrease : ℝ) : ℝ :=
  originalValue * (1 + percentageIncrease / 100)

/-- Theorem: Increasing 900 by 20% results in 1080 -/
theorem increase_900_by_20_percent : valueAfterIncrease 900 20 = 1080 := by
  -- Unfold the definition of valueAfterIncrease
  unfold valueAfterIncrease
  -- Simplify the arithmetic expression
  simp [mul_add, mul_div_right_comm]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_900_by_20_percent_l851_85147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_numbers_sum_zero_existence_proposition_is_particular_product_nonzero_implies_factors_nonzero_not_necessary_and_sufficient_identify_incorrect_statement_l851_85188

-- Statement A
theorem opposite_numbers_sum_zero : 
  ∀ (x y : ℝ), (x = -y) → (x + y = 0) := by sorry

-- Statement B
theorem existence_proposition_is_particular : 
  (∃ x : ℕ, x^2 + 2*x = 0) → True := by sorry

-- Statement C
theorem product_nonzero_implies_factors_nonzero : 
  ∀ (x y : ℝ), (x * y ≠ 0) → (x ≠ 0 ∧ y ≠ 0) := by sorry

-- Statement D (incorrect)
theorem not_necessary_and_sufficient : 
  ∃ (x y : ℝ), (x + y > 2) ∧ ¬(x > 1 ∧ y > 1) := by sorry

-- Main theorem
theorem identify_incorrect_statement : 
  (∀ (x y : ℝ), (x = -y) → (x + y = 0)) ∧ 
  ((∃ x : ℕ, x^2 + 2*x = 0) → True) ∧
  (∀ (x y : ℝ), (x * y ≠ 0) → (x ≠ 0 ∧ y ≠ 0)) ∧
  (∃ (x y : ℝ), (x + y > 2) ∧ ¬(x > 1 ∧ y > 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_numbers_sum_zero_existence_proposition_is_particular_product_nonzero_implies_factors_nonzero_not_necessary_and_sufficient_identify_incorrect_statement_l851_85188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_negative_l851_85192

/-- Given a differentiable function f : ℝ → ℝ and a point x₀ : ℝ,
    if the equation of the tangent line to the curve y = f(x) at the point (x₀, f(x₀))
    is x ln 3 + y - √3 = 0, then f'(x₀) < 0 -/
theorem tangent_line_slope_negative (f : ℝ → ℝ) (x₀ : ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y, x * Real.log 3 + y - Real.sqrt 3 = 0 ↔ y = f x₀ + (deriv f x₀) * (x - x₀)) :
  deriv f x₀ < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_negative_l851_85192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_estimation_l851_85149

/-- Calculates the approximate amount of wheat grains in a batch of mixed grains -/
noncomputable def approximate_wheat_amount (total_amount : ℝ) (sample_size : ℕ) (wheat_in_sample : ℕ) : ℝ :=
  total_amount * (wheat_in_sample / sample_size)

/-- The problem of estimating wheat amount in a mixed grain batch -/
theorem wheat_estimation :
  let total_amount : ℝ := 1534
  let sample_size : ℕ := 254
  let wheat_in_sample : ℕ := 28
  let estimated_wheat : ℝ := approximate_wheat_amount total_amount sample_size wheat_in_sample
  ∃ (ε : ℝ), ε > 0 ∧ |estimated_wheat - 169| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_estimation_l851_85149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_value_l851_85143

/-- Given a triangle ABC where cos A = 1/4 and b = 2c, prove that sin C = √15/8 -/
theorem sin_C_value (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  Real.cos A = 1/4 →       -- given condition
  b = 2*c →                -- given condition
  Real.sin C = Real.sqrt 15 / 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_value_l851_85143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_beta_l851_85102

theorem sin_alpha_minus_beta (α β : Real) 
  (h1 : Real.sin α = 2 * Real.sqrt 3 / 3)
  (h2 : Real.cos (α + β) = -1/3)
  (h3 : 0 < α ∧ α < Real.pi/2)
  (h4 : 0 < β ∧ β < Real.pi/2) :
  Real.sin (α - β) = 10 * Real.sqrt 2 / 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_beta_l851_85102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_g_is_minimum_of_h_l851_85155

/-- A quadratic function passing through (0,4), symmetric about x=3/2, with minimum 7/4 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 4

/-- The function h(x) derived from f(x) -/
def h (t x : ℝ) : ℝ := f x - (2*t - 3)*x

/-- The minimum value of h(x) on [0,1] for a given t -/
noncomputable def g (t : ℝ) : ℝ :=
  if t < 0 then 4
  else if t ≤ 1 then -t^2 + 4
  else 5 - 2*t

theorem f_properties :
  (f 0 = 4) ∧
  (∀ x, f (3 - x) = f x) ∧
  (∃ x, ∀ y, f x ≤ f y) ∧
  (∃ x, f x = 7/4) := by sorry

theorem g_is_minimum_of_h (t : ℝ) :
  ∀ x ∈ Set.Icc 0 1, g t ≤ h t x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_g_is_minimum_of_h_l851_85155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l851_85114

theorem parity_of_expression (a b c : ℕ) (ha : Even a) (hb : Odd b) :
  Odd (2^a + (b+1)^2 + c) ↔ Even c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l851_85114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l851_85153

/-- Calculates the compound interest amount -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time

/-- Calculates the simple interest amount -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem principal_calculation (principal : ℝ) :
  (compound_interest principal 20 2 - principal) - (simple_interest principal 20 2) = 216 →
  principal = 5400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l851_85153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_inequality_l851_85122

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem increasing_function_inequality 
  (h1 : ∀ x y, x < y → f x < f y) -- f is increasing
  (h2 : ∀ x, x ≥ 0 → Set.Nonempty (Set.range f)) -- f is defined on [0, +∞)
  : {x | f (2*x - 1) < f (1/3)} = Set.Ioc (1/2) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_inequality_l851_85122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inradius_relation_l851_85199

-- Define the necessary types and structures
structure Plane :=
  (carrier : Type*)

structure Point (P : Plane) :=
  (coords : P.carrier)

-- Define the necessary functions and predicates
def RightAngle (P : Plane) (A B C : Point P) : Prop := sorry
def IsCircumcenter (P : Plane) (O A B C : Point P) : Prop := sorry
def IsTangentCircle (P : Plane) (O A B C : Point P) (arc : Set (Point P)) (seg1 seg2 : Set (Point P)) : Prop := sorry
def InRadius (P : Plane) (A B C : Point P) : ℝ := sorry
def CircleRadius (P : Plane) (O : Point P) : ℝ := sorry

theorem right_triangle_inradius_relation {P : Plane} (A B C : Point P) (O O₁ O₂ : Point P) 
  (h_right : RightAngle P A C B)
  (h_circum : IsCircumcenter P O A B C)
  (h_O₁ : IsTangentCircle P O₁ A B C (sorry : Set (Point P)) (sorry : Set (Point P)) (sorry : Set (Point P)))
  (h_O₂ : IsTangentCircle P O₂ A B C (sorry : Set (Point P)) (sorry : Set (Point P)) (sorry : Set (Point P)))
  (r : ℝ) (r₁ : ℝ) (r₂ : ℝ)
  (h_inradius : r = InRadius P A B C)
  (h_radius₁ : r₁ = CircleRadius P O₁)
  (h_radius₂ : r₂ = CircleRadius P O₂) :
  r = (r₁ + r₂) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inradius_relation_l851_85199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l851_85173

noncomputable def line_slope (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (y₂ - y₁) / (x₂ - x₁)

theorem line_through_points (m : ℝ) :
  line_slope 1 m (-2) (Real.sqrt 3) = -(Real.sqrt 3) → m = -(2 * Real.sqrt 3) := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l851_85173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_difference_l851_85108

theorem sine_of_angle_difference (α : ℝ) :
  0 < α → α < π / 2 →
  Real.cos (α + π / 6) = 3 / 5 →
  Real.sin (α - π / 6) = (4 - 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_difference_l851_85108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_division_is_standard_non_standard_multiplication_is_not_standard_non_standard_with_unit_is_not_standard_non_standard_mixed_number_is_not_standard_l851_85176

/-- Standard algebraic notation for division of two variables -/
noncomputable def standard_division (a b : ℚ) : ℚ := b / a

/-- Non-standard multiplication notation -/
def non_standard_multiplication (a : ℚ) : ℚ := a * 7

/-- Non-standard notation with unit -/
def non_standard_with_unit (m : ℚ) : String := s!"{2 * m - 1}元"

/-- Non-standard notation with mixed number -/
def non_standard_mixed_number (x : ℚ) : ℚ := (7 / 2) * x

/-- Theorem stating that standard_division conforms to standard algebraic notation -/
theorem standard_division_is_standard : 
  ∀ (a b : ℚ), a ≠ 0 → standard_division a b = b / a := by
  sorry

/-- Theorem stating that non_standard_multiplication does not conform to standard algebraic notation -/
theorem non_standard_multiplication_is_not_standard : 
  ∃ (a : ℚ), non_standard_multiplication a ≠ 7 * a := by
  sorry

/-- Theorem stating that non_standard_with_unit does not conform to standard algebraic notation -/
theorem non_standard_with_unit_is_not_standard : 
  ∀ (m : ℚ), non_standard_with_unit m ≠ s!"{2 * m - 1}" := by
  sorry

/-- Theorem stating that non_standard_mixed_number does not conform to standard algebraic notation -/
theorem non_standard_mixed_number_is_not_standard : 
  ∀ (x : ℚ), non_standard_mixed_number x = (7 / 2) * x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_division_is_standard_non_standard_multiplication_is_not_standard_non_standard_with_unit_is_not_standard_non_standard_mixed_number_is_not_standard_l851_85176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_l851_85195

/-- The curve y = x^3 - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

theorem tangent_parallel_to_x_axis :
  {x : ℝ | f' x = 0} = {-1, 1} ∧
  {(x, f x) | x ∈ ({-1, 1} : Set ℝ)} = {(-1, 2), (1, -2)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_l851_85195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_computation_l851_85180

-- Define lg as log base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_computation :
  lg 5 * (Real.log 20 / Real.log (Real.sqrt 10)) + (lg (2 ^ Real.sqrt 2))^2 + Real.exp (Real.log π) = 2 + π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_computation_l851_85180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f₁_derivative_f₂_derivative_f₃_l851_85156

open Real

-- Define the functions
def f₁ (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 4

noncomputable def f₂ (x : ℝ) : ℝ := x * log x

noncomputable def f₃ (x : ℝ) : ℝ := cos x / x

-- State the theorems
theorem derivative_f₁ (x : ℝ) : 
  deriv f₁ x = 6 * x^2 - 6 * x := by sorry

theorem derivative_f₂ (x : ℝ) (h : x > 0) : 
  deriv f₂ x = log x + 1 := by sorry

theorem derivative_f₃ (x : ℝ) (h : x ≠ 0) : 
  deriv f₃ x = (-x * sin x - cos x) / x^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f₁_derivative_f₂_derivative_f₃_l851_85156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_identification_l851_85191

/-- Represents a balance scale that can be either accurate or faulty -/
inductive Balance
| Accurate
| Faulty

/-- Represents the result of a weighing -/
inductive WeighingResult
| Equal
| LeftHeavier
| RightHeavier

/-- Represents a coin that can be either genuine or counterfeit -/
inductive Coin
| Genuine
| Counterfeit

/-- 
Theorem: Given 3^(2k) coins with one counterfeit coin and three balances (two accurate, one faulty),
it is possible to identify the counterfeit coin in at most 3k + 1 weighings.
-/
theorem counterfeit_coin_identification (k : ℕ) :
  ∃ (strategy : ℕ → ℕ → ℕ → WeighingResult → ℕ),
    ∀ (coins : Fin (3^(2*k)) → Coin) 
      (balances : Fin 3 → Balance),
    (∃! i, coins i = Coin.Counterfeit) →
    (∃! i, balances i = Balance.Faulty) →
    ∃ (i : Fin (3^(2*k))),
      coins i = Coin.Counterfeit ∧
      (∀ j, j ≠ i → coins j = Coin.Genuine) ∧
      ∃ (weighings : Fin (3*k + 1) → ℕ × ℕ × ℕ),
        ∀ n : Fin (3*k + 1),
          let (a, b, c) := weighings n
          strategy a b c (WeighingResult.Equal) = 0
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_identification_l851_85191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ray_l851_85120

-- Define the points M and N
def M : ℝ × ℝ := (3, 0)
def N : ℝ × ℝ := (1, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points P satisfying the condition
def trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |distance p M - distance p N| = 2}

-- Theorem stating that the trajectory is a ray
theorem trajectory_is_ray : 
  ∃ (origin : ℝ × ℝ) (direction : ℝ × ℝ), 
    trajectory = {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = origin + t • direction} := by
  sorry

#check trajectory_is_ray

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ray_l851_85120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvector_problem_l851_85185

theorem eigenvector_problem (A : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ) (k : ℝ) :
  A = !![3, 4; 6, 3] →
  v ≠ 0 →
  A.mulVec v = k • v ↔ k = 3 + 2 * Real.sqrt 6 ∨ k = 3 - 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvector_problem_l851_85185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_ratio_l851_85159

-- Define the points
variable (A B C D E F G : EuclideanSpace ℝ (Fin 2))

-- Define the distances
def AB : ℝ := 15
def BD : ℝ := 18
def AF : ℝ := 15
def DF : ℝ := 12
def BE : ℝ := 24
def CF : ℝ := 17

-- Define the ratio we want to prove
def ratio_BG_FG : ℚ × ℚ := (27, 17)

-- Theorem statement
theorem prove_ratio :
  ∃ (BG FG : ℝ), 
    BG / FG = ratio_BG_FG.1 / ratio_BG_FG.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_ratio_l851_85159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hollow_sphere_weight_proportion_l851_85106

/-- The surface area of a sphere with radius r -/
noncomputable def sphereSurfaceArea (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- The weight of a hollow sphere given its radius and weight coefficient -/
noncomputable def sphereWeight (r : ℝ) (k : ℝ) : ℝ := k * sphereSurfaceArea r

theorem hollow_sphere_weight_proportion (r₁ r₂ w₂ : ℝ) (h₁ : r₁ = 0.15) (h₂ : r₂ = 0.3) (h₃ : w₂ = 32) :
  ∃ k : ℝ, sphereWeight r₂ k = w₂ ∧ sphereWeight r₁ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hollow_sphere_weight_proportion_l851_85106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l851_85160

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 4*x + 4) / Real.log 0.5

-- State the theorem
theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Iio (-2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l851_85160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sin_2x_plus_pi_over_3_l851_85126

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem symmetry_of_sin_2x_plus_pi_over_3 :
  ∀ x : ℝ, f (Real.pi / 3 + (Real.pi / 3 - x)) = f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sin_2x_plus_pi_over_3_l851_85126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identifiable_value_is_one_l851_85105

-- Define the angle y
noncomputable def y : ℝ := sorry

-- Define the condition that y is strictly between 0 and 90 degrees
axiom y_range : 0 < y ∧ y < Real.pi / 2

-- Define the values of trigonometric functions
noncomputable def sin_y : ℝ := Real.sin y
noncomputable def cos_y : ℝ := Real.cos y
noncomputable def tan_y : ℝ := Real.tan y

-- Define the condition that two people can identify their functions while one cannot
axiom identifiable : 
  (sin_y = cos_y ∧ sin_y ≠ tan_y) ∨ 
  (sin_y = tan_y ∧ sin_y ≠ cos_y) ∨ 
  (cos_y = tan_y ∧ cos_y ≠ sin_y)

-- Theorem: The identifiable value must be 1
theorem identifiable_value_is_one : 
  (sin_y = 1 ∧ cos_y ≠ 1 ∧ tan_y ≠ 1) ∨ 
  (cos_y = 1 ∧ sin_y ≠ 1 ∧ tan_y ≠ 1) ∨ 
  (tan_y = 1 ∧ sin_y ≠ 1 ∧ cos_y ≠ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identifiable_value_is_one_l851_85105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l851_85104

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  asymptote_slope : ℝ
  vertex_asymptote_distance : ℝ
  asymptote_eq : asymptote_slope = Real.sqrt 3 / 3
  distance_eq : vertex_asymptote_distance = Real.sqrt 3

/-- The main theorem about the hyperbola's equation -/
theorem hyperbola_equation (h : Hyperbola) : h.a^2 = 12 ∧ h.b^2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l851_85104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_proof_l851_85174

def S (m : ℕ) : Set ℕ := {x | 3 ≤ x ∧ x ≤ m}

def is_partition (A B : Set ℕ) (m : ℕ) : Prop :=
  A ∩ B = ∅ ∧ A ∪ B = S m

def has_product_triple (X : Set ℕ) : Prop :=
  ∃ a b c, a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ a * b = c

def smallest_m_with_product_triple : ℕ := 243

theorem smallest_m_proof (m : ℕ) :
  (m ≥ 3 ∧
   (∀ A B : Set ℕ, is_partition A B m → (has_product_triple A ∨ has_product_triple B))) ↔
  m ≥ smallest_m_with_product_triple :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_proof_l851_85174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mumu_language_identity_l851_85150

/-- The function f(m,u) represents the number of valid words in the Mumu language
    with m occurrences of 'M' and u occurrences of 'U'. A valid word has each 'M'
    followed by a 'U'. -/
def f : ℕ → ℕ → ℕ := sorry

/-- The main theorem stating the identity for f(m,u) -/
theorem mumu_language_identity (m u : ℕ) (h1 : u ≥ 2) (h2 : m ≥ 3) (h3 : m ≤ 2 * u) :
  f m u - f (2 * u - m + 1) u = f m (u - 1) - f (2 * u - m + 1) (u - 1) := by
  sorry

#check mumu_language_identity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mumu_language_identity_l851_85150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l851_85187

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The circle with equation x^2 + y^2 - 2x - 2y - 2 = 0 -/
def circle_equation (p : Point) : Prop := p.x^2 + p.y^2 - 2*p.x - 2*p.y - 2 = 0

/-- The line passing through (2,3) -/
def line_through_2_3 (k : ℝ) (p : Point) : Prop := p.y - 3 = k * (p.x - 2)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem stating the equation of the line -/
theorem line_equation (A B : Point) (k : ℝ) :
  circle_equation A ∧ circle_equation B ∧
  line_through_2_3 k A ∧ line_through_2_3 k B ∧
  distance A B = 2 * Real.sqrt 3 →
  (k = 0 ∧ A.x = 2 ∧ B.x = 2) ∨
  (k = 3/4 ∧ 3*A.x - 4*A.y + 6 = 0 ∧ 3*B.x - 4*B.y + 6 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l851_85187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_father_overtakes_son_l851_85158

/-- Represents the race scenario between a father and son -/
structure RaceScenario where
  track_length : ℚ
  son_start : ℚ
  father_start : ℚ
  son_step : ℚ
  father_step : ℚ

/-- Defines the specific race scenario from the problem -/
def race : RaceScenario :=
  { track_length := 100
  , son_start := 50
  , father_start := 0
  , son_step := 1  -- Arbitrary unit for step length
  , father_step := 7/4  -- Derived from the condition that 7 son steps = 4 father steps
  }

/-- Theorem stating that the father overtakes the son before the finish line -/
theorem father_overtakes_son (r : RaceScenario) : 
  r.track_length = 100 ∧ 
  r.son_start = 50 ∧ 
  r.father_start = 0 ∧
  r.father_step = 7/4 * r.son_step ∧
  (∃ t : ℚ, t > 0 ∧ r.father_start + (6*t) * r.father_step > r.son_start + (5*t) * r.son_step ∧
             r.father_start + (6*t) * r.father_step < r.track_length) :=
by
  sorry

#check father_overtakes_son race

end NUMINAMATH_CALUDE_ERRORFEEDBACK_father_overtakes_son_l851_85158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OPE_PE_passes_through_F_l851_85101

/-- Parabola C: y² = 4x -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- Point on the parabola -/
def P (x y : ℝ) : Prop := ((x, y) : ℝ × ℝ) ∈ C ∧ (x ≠ 0 ∨ y ≠ 0)

/-- Point where line parallel to PS touches the parabola -/
def E (x y : ℝ) : Prop := ((x, y) : ℝ × ℝ) ∈ C ∧ ∃ k : ℝ, ∀ t : ℝ, ((t, -k*t + k*x + y) : ℝ × ℝ) ∉ C

/-- Area of triangle OPE -/
noncomputable def area_OPE (x y : ℝ) : ℝ := sorry

/-- The minimum area of triangle OPE is 2 -/
theorem min_area_OPE : 
  ∀ x y : ℝ, P x y → E x y → area_OPE x y ≥ 2 ∧ ∃ x₀ y₀ : ℝ, P x₀ y₀ ∧ E x₀ y₀ ∧ area_OPE x₀ y₀ = 2 :=
sorry

/-- Line PE passes through the fixed point F(1,0) -/
theorem PE_passes_through_F :
  ∀ x y : ℝ, P x y → E x y → ∃ t : ℝ, ((1 - t) • ((x, y) : ℝ × ℝ) + t • F : ℝ × ℝ) ∈ C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OPE_PE_passes_through_F_l851_85101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l851_85128

theorem inequality_equivalence (x y : ℝ) :
  y^2 - x^2 < x ↔ (x ≥ 0 ∨ x ≤ -1) ∧ -Real.sqrt (x^2 + x) < y ∧ y < Real.sqrt (x^2 + x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l851_85128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tips_fraction_of_income_l851_85145

-- Define the waitress's income components
variable (salary : ℚ)
variable (tips : ℚ)
variable (total_income : ℚ)

-- Define the relationship between tips and salary
axiom tips_ratio : tips = (11 / 4) * salary

-- Define total income
axiom income_composition : total_income = salary + tips

-- Theorem to prove
theorem tips_fraction_of_income : 
  tips / total_income = 11 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tips_fraction_of_income_l851_85145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_price_for_profit_l851_85166

/-- The cost function in million rubles per year for producing x thousand units -/
noncomputable def cost (x : ℝ) : ℝ := 0.5 * x^2 - 2 * x - 10

/-- The annual profit function in million rubles -/
noncomputable def annual_profit (p x : ℝ) : ℝ := p * x - cost x

/-- The maximum annual profit function -/
noncomputable def max_annual_profit (p : ℝ) : ℝ := (p + 2)^2 / 2 + 10

/-- The theorem stating the minimum value of p for the required profit -/
theorem min_price_for_profit :
  ∀ p : ℝ, p ≥ 0 →
  (∀ x : ℝ, annual_profit p x ≤ max_annual_profit p) →
  (3 * max_annual_profit p ≥ 126) →
  p ≥ 6 := by
  sorry

#check min_price_for_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_price_for_profit_l851_85166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l851_85165

-- Define the ellipse equation
def ellipse_equation (x y m : ℝ) : Prop := 2 * x^2 - m * y^2 = 1

-- Define the focus coordinates
noncomputable def focus : ℝ × ℝ := (0, -Real.sqrt 2)

-- Theorem statement
theorem ellipse_m_value :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), ellipse_equation x y m) ∧ 
    (focus.2)^2 = -(1/2) - (1/m) →
    m = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l851_85165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_min_value_h_max_value_common_tangent_condition_l851_85107

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 1/2 * x^2 - 3/2 * x + m
noncomputable def h (m : ℝ) (x : ℝ) : ℝ := g m x - f x

-- Theorem for the minimum value of h(x) on [1, 3]
theorem h_min_value (m : ℝ) : 
  ∃ (x : ℝ), x ∈ Set.Icc 1 3 ∧ h m x = m - Real.log 2 - 1 ∧ 
  ∀ (y : ℝ), y ∈ Set.Icc 1 3 → h m y ≥ h m x :=
sorry

-- Theorem for the maximum value of h(x) on [1, 3]
theorem h_max_value (m : ℝ) :
  ∃ (x : ℝ), x ∈ Set.Icc 1 3 ∧ h m x = m - 1 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 1 3 → h m y ≤ h m x :=
sorry

-- Theorem for the range of m for common tangent
theorem common_tangent_condition :
  ∀ (m : ℝ), (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
    (deriv f x = deriv (g m) x) ∧ (deriv f y = deriv (g m) y) ∧ 
    (f x - g m x) / (y - x) = deriv f x) → m ≥ 1 + Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_min_value_h_max_value_common_tangent_condition_l851_85107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_decrease_l851_85178

theorem circle_area_decrease (r : ℝ) (h : r > 0) : 
  (π * r^2 - π * (r/2)^2) / (π * r^2) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_decrease_l851_85178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radians_to_degrees_degrees_to_radians_l851_85130

-- Define the conversion factor
noncomputable def deg_per_rad : ℝ := 180 / Real.pi

-- Theorem 1: Convert -5/3π radians to degrees
theorem radians_to_degrees :
  -5/3 * Real.pi * deg_per_rad = -300 := by
  sorry

-- Theorem 2: Convert -135° to radians
theorem degrees_to_radians :
  -135 / deg_per_rad = -3/4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radians_to_degrees_degrees_to_radians_l851_85130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_P_and_Q_l851_85167

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - x - 6 ≥ 0}
def Q : Set ℝ := {x | Real.rpow 2 x ≥ 1}

-- Define the complement of P in ℝ
def C_R_P : Set ℝ := {x | x ∉ P}

-- Theorem statement
theorem intersection_complement_P_and_Q :
  (C_R_P ∩ Q) = {x : ℝ | 0 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_P_and_Q_l851_85167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ln_6_l851_85109

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x - Real.exp (-x) else 0  -- We use 0 as a placeholder for x ≥ 0

-- State the theorem
theorem f_ln_6 (f : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, x < 0 → f x = x - Real.exp (-x)) →  -- definition for x < 0
  f (Real.log 6) = Real.log 6 + 6 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ln_6_l851_85109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_24_l851_85112

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  stillWaterSpeed : ℚ
  downstreamDistance : ℚ
  downstreamTime : ℚ
  upstreamTime : ℚ

/-- Calculates the upstream distance given a swimming scenario -/
def upstreamDistance (s : SwimmingScenario) : ℚ :=
  let streamSpeed := s.downstreamDistance / s.downstreamTime - s.stillWaterSpeed
  (s.stillWaterSpeed - streamSpeed) * s.upstreamTime

/-- Theorem stating that the upstream distance is 24 km given the specific conditions -/
theorem upstream_distance_is_24 :
  let scenario : SwimmingScenario := {
    stillWaterSpeed := 7,
    downstreamDistance := 32,
    downstreamTime := 4,
    upstreamTime := 4
  }
  upstreamDistance scenario = 24 := by
  -- Proof goes here
  sorry

#eval upstreamDistance {
  stillWaterSpeed := 7,
  downstreamDistance := 32,
  downstreamTime := 4,
  upstreamTime := 4
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_24_l851_85112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l851_85141

/-- Calculates the length of a train given the speeds of two trains, the time they take to cross each other, and the length of the other train. -/
noncomputable def calculateTrainLength (speed1 speed2 : ℝ) (crossTime : ℝ) (otherTrainLength : ℝ) : ℝ :=
  let relativeSpeed := (speed1 + speed2) * (1000 / 3600)  -- Convert km/hr to m/s
  let totalDistance := relativeSpeed * crossTime
  totalDistance - otherTrainLength

/-- The length of the first train is approximately 140 meters. -/
theorem train_length_theorem (speed1 speed2 crossTime otherTrainLength : ℝ) 
    (h1 : speed1 = 60)
    (h2 : speed2 = 40)
    (h3 : crossTime = 11.159107271418288)
    (h4 : otherTrainLength = 170) :
  ∃ ε > 0, |calculateTrainLength speed1 speed2 crossTime otherTrainLength - 140| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l851_85141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_voters_after_T_max_voters_after_T_from_max_rating_l851_85118

/-- Represents the state of movie ratings at a given moment -/
structure RatingState where
  n : ℕ  -- number of votes
  x : ℕ  -- current rating (assumed to be an integer)

/-- Calculates the next rating state given the current state and a new vote -/
def nextState (s : RatingState) (_ : ℕ) : RatingState :=
  { n := s.n + 1, x := s.x - 1 }

/-- Checks if a given vote is valid (non-negative and at most 10) -/
def isValidVote (y : ℕ) : Prop := y ≤ 10

/-- Checks if a sequence of votes is valid and decreases the rating by 1 each time -/
def isValidSequence (init : RatingState) (votes : List ℕ) : Prop :=
  votes.foldl (λ acc vote => acc ∧ isValidVote vote ∧ 
    (init.n * init.x + vote) / (init.n + 1) = init.x - 1) True

/-- The main theorem stating the maximum number of voters after moment T -/
theorem max_voters_after_T (init : RatingState) : 
  (∃ votes : List ℕ, votes.length = 5 ∧ isValidSequence init votes) ∧ 
  (∀ votes : List ℕ, votes.length > 5 → ¬isValidSequence init votes) := by
  sorry

/-- An example initial state with maximum possible rating -/
def maxInitialState : RatingState := { n := 1, x := 10 }

/-- Corollary: The maximum number of voters after T is 5 when starting from the highest possible rating -/
theorem max_voters_after_T_from_max_rating : 
  (∃ votes : List ℕ, votes.length = 5 ∧ isValidSequence maxInitialState votes) ∧ 
  (∀ votes : List ℕ, votes.length > 5 → ¬isValidSequence maxInitialState votes) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_voters_after_T_max_voters_after_T_from_max_rating_l851_85118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_number_l851_85135

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ 
  ({a, b, c} : Finset ℕ) = {2, 6, 9}

theorem largest_three_digit_number : 
  (∀ n : ℕ, is_valid_number n → n ≤ 962) ∧ is_valid_number 962 := by
  sorry

#check largest_three_digit_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_number_l851_85135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_black_pairs_60_28_l851_85119

/-- A custom deck of cards -/
structure Deck where
  total : ℕ
  black : ℕ
  red : ℕ
  h1 : black + red = total

/-- The expected number of pairs of adjacent black cards in a circular arrangement -/
def expectedBlackPairs (d : Deck) : ℚ :=
  (d.black : ℚ) * (d.black - 1) / (d.total - 1)

/-- Theorem: For a deck of 60 cards with 28 black cards, the expected number of adjacent black pairs is 756/59 -/
theorem expected_black_pairs_60_28 :
  let d : Deck := ⟨60, 28, 32, by norm_num⟩
  expectedBlackPairs d = 756 / 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_black_pairs_60_28_l851_85119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_F_l851_85129

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : a 1 = 1
  h4 : (a 3) ^ 2 = a 1 * a 13

/-- Sum of the first n terms of the arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- The expression to be minimized -/
noncomputable def F (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (2 * S seq n + 16) / (seq.a n + 3)

/-- The main theorem stating the minimum value of F -/
theorem min_value_of_F (seq : ArithmeticSequence) :
  ∀ n : ℕ, F seq n ≥ 4 ∧ ∃ n₀ : ℕ, F seq n₀ = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_F_l851_85129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l851_85163

/-- Represents the average speed of a train given two segments of its journey -/
noncomputable def averageSpeed (x : ℝ) : ℝ :=
  let distance1 := x
  let speed1 := (40 : ℝ)
  let distance2 := 4 * x
  let speed2 := (20 : ℝ)
  let totalDistance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let totalTime := time1 + time2
  totalDistance / totalTime

/-- Theorem stating that the average speed of the train is 200/9 kmph -/
theorem train_average_speed (x : ℝ) (h : x > 0) : averageSpeed x = 200 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l851_85163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_16_4_l851_85164

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem floor_expression_equals_16_4 :
  (floor 6.5) * (floor (2 / 3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.4) - 6 = 16.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_16_4_l851_85164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l851_85177

def M : Set ℤ := {-1, 1}
def N : Set ℤ := {x | (1/2 : ℝ) < (2 : ℝ)^(x+1) ∧ (2 : ℝ)^(x+1) < 4}

theorem intersection_M_N : M ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l851_85177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_miles_per_tankful_l851_85169

/-- Represents the fuel efficiency and tank capacity of a car -/
structure CarFuelData where
  highway_miles_per_tankful : ℚ
  city_miles_per_gallon : ℚ
  highway_city_mpg_difference : ℚ

/-- Calculates the miles per tankful of gasoline in the city -/
noncomputable def city_miles_per_tankful (data : CarFuelData) : ℚ :=
  let highway_miles_per_gallon := data.city_miles_per_gallon + data.highway_city_mpg_difference
  let tank_capacity := data.highway_miles_per_tankful / highway_miles_per_gallon
  data.city_miles_per_gallon * tank_capacity

/-- Theorem stating that given the conditions, the car traveled 336 miles per tankful in the city -/
theorem car_city_miles_per_tankful (data : CarFuelData)
    (h1 : data.highway_miles_per_tankful = 462)
    (h2 : data.highway_city_mpg_difference = 12)
    (h3 : data.city_miles_per_gallon = 32) :
    city_miles_per_tankful data = 336 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_miles_per_tankful_l851_85169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_eccentricity_l851_85151

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  eccentricity : ℝ

/-- Represents a parabola -/
structure Parabola where
  vertex : Point
  focus : Point

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Eccentricity of specific ellipse and parabola configuration -/
theorem ellipse_parabola_eccentricity 
  (e : Ellipse) 
  (p : Parabola) 
  (P : Point) : 
  e.focus1 = Point.mk (-1) 0 →
  e.focus2 = Point.mk 1 0 →
  p.vertex = e.focus1 →
  p.focus = e.focus2 →
  (distance P e.focus1) / (distance P e.focus2) = e.eccentricity →
  e.eccentricity = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_eccentricity_l851_85151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheila_picnic_probability_l851_85162

theorem sheila_picnic_probability :
  let p_rain := 0.3
  let p_sunny := 1 - p_rain
  let p_attend_if_rain := 0.25
  let p_attend_if_sunny := 0.7
  let p_attend_special := 0.15
  p_rain * p_attend_if_rain + p_sunny * p_attend_if_sunny + p_attend_special -
  (p_rain * p_attend_if_rain * p_attend_special) -
  (p_sunny * p_attend_if_sunny * p_attend_special) = 0.63025 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheila_picnic_probability_l851_85162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l851_85152

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := c * x / (2 * x + 3)

theorem function_composition_identity (c : ℝ) :
  (∀ x : ℝ, x ≠ -3/2 → f c (f c x) = x) → c = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l851_85152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_one_functions_l851_85193

/-- A function f has an "average" of C on its domain D if for all x in D,
    there exists a unique y in D such that (f(x) + f(y))/2 = C -/
def has_average (f : ℝ → ℝ) (D : Set ℝ) (C : ℝ) : Prop :=
  ∀ x, x ∈ D → ∃! y, y ∈ D ∧ (f x + f y) / 2 = C

noncomputable def f₁ : ℝ → ℝ := fun x ↦ x^3
noncomputable def f₂ : ℝ → ℝ := fun x ↦ (1/2)^x
noncomputable def f₃ : ℝ → ℝ := fun x ↦ Real.log x
noncomputable def f₄ : ℝ → ℝ := fun x ↦ 2 * Real.sin x

theorem average_of_one_functions :
  (has_average f₁ Set.univ 1) ∧
  (¬ has_average f₂ Set.univ 1) ∧
  (has_average f₃ (Set.Ioi 0) 1) ∧
  (¬ has_average f₄ Set.univ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_one_functions_l851_85193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_properties_l851_85140

open Matrix

theorem matrix_N_properties :
  ∃ (N : Matrix (Fin 3) (Fin 3) ℝ),
    (∀ (u : Fin 3 → ℝ), N.mulVec u = (7 : ℝ) • u) ∧
    (∀ (t : ℝ), N.mulVec (λ i => if i = 2 then t else 0) = (-3 : ℝ) • (λ i => if i = 2 then t else 0)) ∧
    N = ![![7, 0, 0], ![0, 7, 0], ![0, 0, -3]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_properties_l851_85140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_zeros_implies_m_range_l851_85111

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then Real.log x + x else 2 * x^2 - m * x + m / 2

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x - m

def has_three_zeros (h : ℝ → ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ h x = 0 ∧ h y = 0 ∧ h z = 0

theorem f_g_zeros_implies_m_range :
  ∀ m : ℝ, has_three_zeros (g m) → 1 < m ∧ m ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_zeros_implies_m_range_l851_85111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_problem_l851_85168

/-- Represents the distance traveled by a car in the first hour -/
def initial_distance : ℝ := 0

/-- Calculates the total distance traveled by the car over a given number of hours -/
noncomputable def total_distance (x : ℝ) (hours : ℕ) : ℝ :=
  (hours : ℝ) / 2 * (2 * x + (hours - 1) * 2)

theorem car_distance_problem :
  ∃ x : ℝ, total_distance x 12 = 672 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_problem_l851_85168

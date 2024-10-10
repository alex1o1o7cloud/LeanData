import Mathlib

namespace sum_of_pentagram_angles_l2631_263150

/-- A self-intersecting five-pointed star (pentagram) -/
structure Pentagram where
  vertices : Fin 5 → Point2
  is_self_intersecting : Bool

/-- The sum of angles at the vertices of a pentagram -/
def sum_of_vertex_angles (p : Pentagram) : ℝ := sorry

/-- Theorem: The sum of angles at the vertices of a self-intersecting pentagram is 180° -/
theorem sum_of_pentagram_angles (p : Pentagram) (h : p.is_self_intersecting = true) :
  sum_of_vertex_angles p = 180 := by sorry

end sum_of_pentagram_angles_l2631_263150


namespace marys_tuesday_payment_l2631_263172

theorem marys_tuesday_payment 
  (credit_limit : ℕ) 
  (thursday_payment : ℕ) 
  (remaining_payment : ℕ) : 
  credit_limit - (thursday_payment + remaining_payment) = 15 :=
by
  sorry

#check marys_tuesday_payment 100 23 62

end marys_tuesday_payment_l2631_263172


namespace alternating_square_sum_equals_5304_l2631_263119

def alternatingSquareSum (n : ℕ) : ℤ :=
  let seq := List.range n |> List.reverse |> List.map (λ i => (101 - i : ℤ)^2)
  seq.enum.foldl (λ acc (i, x) => acc + (if i % 4 < 2 then x else -x)) 0

theorem alternating_square_sum_equals_5304 :
  alternatingSquareSum 100 = 5304 := by
  sorry

end alternating_square_sum_equals_5304_l2631_263119


namespace arithmetic_sequence_special_case_l2631_263197

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating that if a_2 and a_10 of an arithmetic sequence are roots of x^2 + 12x - 8 = 0, then a_6 = -6 -/
theorem arithmetic_sequence_special_case (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2)^2 + 12*(a 2) - 8 = 0 →
  (a 10)^2 + 12*(a 10) - 8 = 0 →
  a 6 = -6 := by
  sorry

end arithmetic_sequence_special_case_l2631_263197


namespace f_is_generalized_distance_l2631_263191

-- Define the binary function f
def f (x y : ℝ) : ℝ := x^2 + y^2

-- State the theorem
theorem f_is_generalized_distance :
  (∀ x y : ℝ, f x y ≥ 0 ∧ (f x y = 0 ↔ x = 0 ∧ y = 0)) ∧ 
  (∀ x y : ℝ, f x y = f y x) ∧
  (∀ x y z : ℝ, f x y ≤ f x z + f z y) :=
sorry

end f_is_generalized_distance_l2631_263191


namespace mean_homeruns_is_12_08_l2631_263124

def total_hitters : ℕ := 12

def april_homeruns : List (ℕ × ℕ) := [(5, 4), (6, 4), (8, 2), (10, 1)]
def may_homeruns : List (ℕ × ℕ) := [(5, 2), (6, 2), (8, 3), (10, 2), (11, 1)]

def total_homeruns : ℕ := 
  (april_homeruns.map (λ p => p.1 * p.2)).sum + 
  (may_homeruns.map (λ p => p.1 * p.2)).sum

theorem mean_homeruns_is_12_08 : 
  (total_homeruns : ℚ) / total_hitters = 12.08 := by sorry

end mean_homeruns_is_12_08_l2631_263124


namespace simplify_expression_l2631_263152

theorem simplify_expression (x : ℝ) :
  3 * x^3 + 4 * x^2 + 5 * x + 10 - (-6 + 3 * x^3 - 2 * x^2 + x) = 6 * x^2 + 4 * x + 16 := by
  sorry

end simplify_expression_l2631_263152


namespace g_five_equals_one_l2631_263144

theorem g_five_equals_one (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, g (x + y) = g x * g y) 
  (h2 : ∀ x : ℝ, g x ≠ 0) : 
  g 5 = 1 := by
sorry

end g_five_equals_one_l2631_263144


namespace smallest_degree_is_five_l2631_263131

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

end smallest_degree_is_five_l2631_263131


namespace tiffany_cans_problem_l2631_263129

theorem tiffany_cans_problem (monday_bags : ℕ) (next_day_bags : ℕ) : 
  (monday_bags = next_day_bags + 1) → (next_day_bags = 7) → (monday_bags = 8) :=
by sorry

end tiffany_cans_problem_l2631_263129


namespace find_m_value_l2631_263158

def A : Set ℝ := {-1, 1, 3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem find_m_value : ∀ m : ℝ, B m ⊆ A → m = 1 ∨ m = -1 := by
  sorry

end find_m_value_l2631_263158


namespace area_relationship_l2631_263104

/-- Two congruent isosceles right-angled triangles with inscribed squares -/
structure TriangleWithSquare where
  /-- The side length of the triangle -/
  side : ℝ
  /-- The side length of the inscribed square -/
  square_side : ℝ
  /-- The inscribed square's side is less than the triangle's side -/
  h_square_fits : square_side < side

/-- The theorem stating the relationship between the areas of squares P and R -/
theorem area_relationship (t : TriangleWithSquare) (h_area_p : t.square_side ^ 2 = 45) :
  ∃ (r : ℝ), r ^ 2 = 40 ∧ ∃ (t' : TriangleWithSquare), t'.square_side ^ 2 = r ^ 2 :=
sorry

end area_relationship_l2631_263104


namespace fish_weight_l2631_263120

/-- Represents the weight of a fish with its components -/
structure Fish where
  head : ℝ
  body : ℝ
  tail : ℝ

/-- The fish satisfies the given conditions -/
def validFish (f : Fish) : Prop :=
  f.head = f.tail + f.body / 2 ∧
  f.body = f.head + f.tail ∧
  f.tail = 1

/-- The total weight of the fish -/
def totalWeight (f : Fish) : ℝ :=
  f.head + f.body + f.tail

/-- Theorem stating that a valid fish weighs 8 kg -/
theorem fish_weight (f : Fish) (h : validFish f) : totalWeight f = 8 := by
  sorry

#check fish_weight

end fish_weight_l2631_263120


namespace inversion_of_line_l2631_263188

/-- A circle in a plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- A line in a plane -/
structure Line :=
  (point : ℝ × ℝ)
  (direction : ℝ × ℝ)

/-- The result of inverting a line with respect to a circle -/
inductive InversionResult
  | SameLine : InversionResult
  | Circle : (ℝ × ℝ) → ℝ → InversionResult

/-- Inversion of a line with respect to a circle -/
def invert (l : Line) (c : Circle) : InversionResult :=
  sorry

/-- Theorem: The image of a line under inversion is either the line itself or a circle passing through the center of inversion -/
theorem inversion_of_line (l : Line) (c : Circle) :
  (invert l c = InversionResult.SameLine ∧ l.point = c.center) ∨
  (∃ center radius, invert l c = InversionResult.Circle center radius ∧ center = c.center) :=
sorry

end inversion_of_line_l2631_263188


namespace f_is_increasing_l2631_263166

def f (x : ℝ) := 2 * x + 1

theorem f_is_increasing : Monotone f := by sorry

end f_is_increasing_l2631_263166


namespace fruit_salad_cherries_l2631_263186

theorem fruit_salad_cherries (b r g c : ℕ) : 
  b + r + g + c = 580 →
  r = 2 * b →
  g = 3 * c →
  c = 3 * r →
  c = 129 := by
sorry

end fruit_salad_cherries_l2631_263186


namespace inverse_proportion_problem_l2631_263107

def inversely_proportional (x y : ℝ) : Prop := ∃ k : ℝ, x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : inversely_proportional x₁ y₁)
  (h2 : inversely_proportional x₂ y₂)
  (h3 : x₁ = 5)
  (h4 : y₁ = 15)
  (h5 : y₂ = 30) :
  x₂ = 5/2 := by
sorry

end inverse_proportion_problem_l2631_263107


namespace lcm_problem_l2631_263138

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

end lcm_problem_l2631_263138


namespace quadratic_real_roots_l2631_263192

/-- The quadratic equation kx^2 - 6x + 9 = 0 has real roots if and only if k ≤ 1 and k ≠ 0 -/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end quadratic_real_roots_l2631_263192


namespace volume_of_square_cross_section_cylinder_l2631_263185

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

end volume_of_square_cross_section_cylinder_l2631_263185


namespace existence_of_large_solutions_l2631_263171

theorem existence_of_large_solutions :
  ∃ (x y z u v : ℕ), 
    x > 2000 ∧ y > 2000 ∧ z > 2000 ∧ u > 2000 ∧ v > 2000 ∧
    x^2 + y^2 + z^2 + u^2 + v^2 = x*y*z*u*v - 65 := by
  sorry

end existence_of_large_solutions_l2631_263171


namespace infinite_solutions_implies_c_value_l2631_263135

/-- If infinitely many values of y satisfy the equation 3(5 + 2cy) = 15y + 15 + y^2, then c = 2.5 -/
theorem infinite_solutions_implies_c_value (c : ℝ) : 
  (∀ y : ℝ, 3 * (5 + 2 * c * y) = 15 * y + 15 + y^2) → c = 2.5 := by
  sorry

end infinite_solutions_implies_c_value_l2631_263135


namespace empty_set_implies_a_range_l2631_263199

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + a * x + 1 = 0}

-- State the theorem
theorem empty_set_implies_a_range (a : ℝ) : 
  A a = ∅ → 0 ≤ a ∧ a < 4 := by
  sorry

end empty_set_implies_a_range_l2631_263199


namespace parabola_point_comparison_l2631_263103

theorem parabola_point_comparison :
  ∀ (y₁ y₂ : ℝ),
  y₁ = (-5)^2 + 2*(-5) + 3 →
  y₂ = 2^2 + 2*2 + 3 →
  y₁ > y₂ := by
sorry

end parabola_point_comparison_l2631_263103


namespace cosine_domain_range_minimum_l2631_263112

open Real

theorem cosine_domain_range_minimum (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x = cos x) →
  (∀ x ∈ Set.Icc a b, -1/2 ≤ f x ∧ f x ≤ 1) →
  (∃ x ∈ Set.Icc a b, f x = -1/2) →
  (∃ x ∈ Set.Icc a b, f x = 1) →
  b - a ≥ 2*π/3 :=
by sorry

end cosine_domain_range_minimum_l2631_263112


namespace selection_ways_l2631_263134

def group_size : ℕ := 8
def roles_to_fill : ℕ := 3

theorem selection_ways : (group_size.factorial) / ((group_size - roles_to_fill).factorial) = 336 := by
  sorry

end selection_ways_l2631_263134


namespace rectangle_width_equals_circle_area_l2631_263169

theorem rectangle_width_equals_circle_area (r : ℝ) (l w : ℝ) : 
  r = Real.sqrt 12 → 
  l = 3 * Real.sqrt 2 → 
  π * r^2 = l * w → 
  w = 2 * Real.sqrt 2 * π := by
sorry

end rectangle_width_equals_circle_area_l2631_263169


namespace function_maximum_implies_a_range_l2631_263162

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ a then -(x + 1) * Real.exp x else -2 * x - 1

theorem function_maximum_implies_a_range (a : ℝ) :
  (∃ (M : ℝ), ∀ (x : ℝ), f a x ≤ M) →
  a ≥ -(1/2) - 1/(2 * Real.exp 2) :=
by sorry

end function_maximum_implies_a_range_l2631_263162


namespace max_x_value_l2631_263193

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 7) (prod_sum_eq : x*y + x*z + y*z = 10) :
  x ≤ 3 ∧ ∃ (y' z' : ℝ), x = 3 ∧ y' + z' = 4 ∧ 3*y' + 3*z' + y'*z' = 10 :=
by sorry

end max_x_value_l2631_263193


namespace max_profit_at_150_l2631_263183

/-- Represents the total revenue function for the workshop --/
noncomputable def H (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 200 then 400 * x - x^2 else 40000

/-- Represents the total cost function for the workshop --/
def total_cost (x : ℝ) : ℝ := 7500 + 100 * x

/-- Represents the profit function for the workshop --/
noncomputable def profit (x : ℝ) : ℝ := H x - total_cost x

/-- Theorem stating the maximum profit and corresponding production volume --/
theorem max_profit_at_150 :
  (∃ (x : ℝ), ∀ (y : ℝ), profit y ≤ profit x) ∧
  (∀ (x : ℝ), profit x ≤ 15000) ∧
  profit 150 = 15000 :=
sorry

end max_profit_at_150_l2631_263183


namespace investment_rate_proof_l2631_263136

theorem investment_rate_proof (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (target_income : ℝ) (available_rates : List ℝ) :
  total_investment = 12000 →
  first_investment = 5000 →
  second_investment = 4000 →
  first_rate = 0.03 →
  second_rate = 0.045 →
  target_income = 580 →
  available_rates = [0.05, 0.055, 0.06, 0.065, 0.07] →
  ∃ (optimal_rate : ℝ), 
    optimal_rate ∈ available_rates ∧
    optimal_rate = 0.07 ∧
    ∀ (rate : ℝ), rate ∈ available_rates →
      |((target_income - (first_investment * first_rate + second_investment * second_rate)) / 
        (total_investment - first_investment - second_investment)) - optimal_rate| ≤
      |((target_income - (first_investment * first_rate + second_investment * second_rate)) / 
        (total_investment - first_investment - second_investment)) - rate| :=
by sorry

end investment_rate_proof_l2631_263136


namespace partial_fraction_sum_zero_l2631_263187

theorem partial_fraction_sum_zero : 
  ∃ (A B C D E F : ℝ), 
    (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
      1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) = 
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
    A + B + C + D + E + F = 0 := by
  sorry

end partial_fraction_sum_zero_l2631_263187


namespace fourth_root_equation_solutions_l2631_263173

theorem fourth_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (x^(1/4) : ℝ) - 15 / (8 - (x^(1/4) : ℝ))
  ∀ x : ℝ, f x = 0 ↔ x = 81 ∨ x = 625 :=
by sorry

end fourth_root_equation_solutions_l2631_263173


namespace dress_discount_problem_l2631_263159

theorem dress_discount_problem (P D : ℝ) : 
  P * (1 - D) * 1.25 = 71.4 →
  P - 71.4 = 5.25 →
  D = 0.255 := by
  sorry

end dress_discount_problem_l2631_263159


namespace count_numbers_with_property_l2631_263145

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The reverse of a two-digit number. -/
def reverse (n : ℕ) : ℕ := 
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

/-- The property that a number, when added to its reverse, sums to 144. -/
def hasProperty (n : ℕ) : Prop := n + reverse n = 144

/-- The main theorem stating that there are exactly 6 two-digit numbers satisfying the property. -/
theorem count_numbers_with_property : 
  ∃! (s : Finset ℕ), (∀ n ∈ s, TwoDigitNumber n ∧ hasProperty n) ∧ Finset.card s = 6 :=
sorry

end count_numbers_with_property_l2631_263145


namespace line_property_l2631_263146

/-- Given two points on a line, prove that m - 2b equals 21 --/
theorem line_property (x₁ y₁ x₂ y₂ m b : ℝ) 
  (h₁ : y₁ = m * x₁ + b) 
  (h₂ : y₂ = m * x₂ + b) 
  (h₃ : x₁ = 2) 
  (h₄ : y₁ = -3) 
  (h₅ : x₂ = 6) 
  (h₆ : y₂ = 9) : 
  m - 2 * b = 21 := by
  sorry

#check line_property

end line_property_l2631_263146


namespace conference_handshakes_l2631_263190

theorem conference_handshakes (n : ℕ) (m : ℕ) : 
  n = 15 →  -- number of married couples
  m = 3 →   -- number of men who don't shake hands with each other
  (2 * n * (2 * n - 1) - 2 * n) / 2 - (m * (m - 1)) / 2 = 417 :=
by sorry

end conference_handshakes_l2631_263190


namespace highest_probability_A_l2631_263130

-- Define the sample space
variable (Ω : Type)
-- Define the probability measure
variable (P : Set Ω → ℝ)

-- Define events A, B, and C
variable (A B C : Set Ω)

-- State the theorem
theorem highest_probability_A (hCB : C ⊆ B) (hBA : B ⊆ A) :
  P A ≥ P B ∧ P A ≥ P C := by
  sorry

end highest_probability_A_l2631_263130


namespace determinant_of_specific_matrix_l2631_263143

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 5, -3; 0, 3, -1; 7, -4, 2]
  Matrix.det A = 32 := by
  sorry

end determinant_of_specific_matrix_l2631_263143


namespace arithmetic_sequence_m_value_l2631_263100

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- The main theorem -/
theorem arithmetic_sequence_m_value
  (seq : ArithmeticSequence)
  (h1 : seq.S (m - 1) = -2)
  (h2 : seq.S m = 0)
  (h3 : seq.S (m + 1) = 3)
  : m = 5 := by
  sorry


end arithmetic_sequence_m_value_l2631_263100


namespace four_number_sequence_l2631_263157

theorem four_number_sequence (a b c d : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) → -- Geometric sequence condition
  a + b + c = 19 →
  (∃ q : ℝ, c = b + q ∧ d = c + q) → -- Arithmetic sequence condition
  b + c + d = 12 →
  ((a = 25 ∧ b = -10 ∧ c = 4 ∧ d = 18) ∨ (a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2)) :=
by sorry

end four_number_sequence_l2631_263157


namespace max_value_of_function_l2631_263121

theorem max_value_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, a^(2*x) + 2*a^x - 9 ≤ 6) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, a^(2*x) + 2*a^x - 9 = 6) →
  a = 3 ∨ a = 1/3 := by
sorry

end max_value_of_function_l2631_263121


namespace quadratic_roots_imply_k_value_l2631_263181

theorem quadratic_roots_imply_k_value (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 8 * x + k = 0 ↔ x = -2 + Real.sqrt 6 ∨ x = -2 - Real.sqrt 6) →
  k = -4 :=
by sorry

end quadratic_roots_imply_k_value_l2631_263181


namespace quadratic_function_properties_quadratic_function_root_range_l2631_263151

def f (q : ℝ) (x : ℝ) : ℝ := x^2 - 16*x + q + 3

theorem quadratic_function_properties (q : ℝ) :
  (∃ (min : ℝ), ∀ (x : ℝ), f q x ≥ min ∧ (∃ (x_min : ℝ), f q x_min = min) ∧ min = -60) →
  q = 1 :=
sorry

theorem quadratic_function_root_range (q : ℝ) :
  (∃ (x : ℝ), x ∈ Set.Icc (-1) 1 ∧ f q x = 0) →
  q ∈ Set.Icc (-20) 12 :=
sorry

end quadratic_function_properties_quadratic_function_root_range_l2631_263151


namespace system_solutions_l2631_263184

theorem system_solutions :
  ∀ x y : ℝ,
  (y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y) ↔
  ((x = 0 ∧ y = 0) ∨ 
   (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2) ∨ 
   (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2)) :=
by sorry

end system_solutions_l2631_263184


namespace gum_sharing_proof_l2631_263164

/-- The number of people sharing gum equally -/
def num_people (john_gum cole_gum aubrey_gum pieces_per_person : ℕ) : ℕ :=
  (john_gum + cole_gum + aubrey_gum) / pieces_per_person

/-- Proof that 3 people are sharing the gum -/
theorem gum_sharing_proof :
  num_people 54 45 0 33 = 3 := by
  sorry

end gum_sharing_proof_l2631_263164


namespace parametric_to_cartesian_ellipse_parametric_to_cartesian_line_l2631_263115

-- Equation 1
theorem parametric_to_cartesian_ellipse (x y φ : ℝ) :
  x = 5 * Real.cos φ ∧ y = 4 * Real.sin φ ↔ x^2 / 25 + y^2 / 16 = 1 :=
sorry

-- Equation 2
theorem parametric_to_cartesian_line (x y t : ℝ) :
  x = 1 - 3 * t^2 ∧ y = 4 * t^2 ↔ 4 * x + 3 * y - 4 = 0 ∧ x ≤ 1 :=
sorry

end parametric_to_cartesian_ellipse_parametric_to_cartesian_line_l2631_263115


namespace basketball_free_throws_l2631_263116

theorem basketball_free_throws (total_players : Nat) (captains : Nat) 
  (h1 : total_players = 15)
  (h2 : captains = 2)
  (h3 : captains ≤ total_players) :
  (total_players - 1) * captains = 28 := by
  sorry

end basketball_free_throws_l2631_263116


namespace friend_brought_30_chocolates_l2631_263128

/-- The number of chocolates Nida's friend brought -/
def friend_chocolates (
  initial_chocolates : ℕ)  -- Nida's initial number of chocolates
  (loose_chocolates : ℕ)   -- Number of chocolates not in a box
  (filled_boxes : ℕ)       -- Number of filled boxes initially
  (extra_boxes_needed : ℕ) -- Number of extra boxes needed after friend brings chocolates
  : ℕ :=
  30

/-- Theorem stating that the number of chocolates Nida's friend brought is 30 -/
theorem friend_brought_30_chocolates :
  friend_chocolates 50 5 3 2 = 30 := by
  sorry

end friend_brought_30_chocolates_l2631_263128


namespace training_hours_calculation_l2631_263153

/-- Given a person trains for a specific number of hours per day and a total number of days,
    calculate the total hours spent training. -/
def total_training_hours (hours_per_day : ℕ) (total_days : ℕ) : ℕ :=
  hours_per_day * total_days

/-- Theorem: A person training for 5 hours every day for 42 days spends 210 hours in total. -/
theorem training_hours_calculation :
  let hours_per_day : ℕ := 5
  let initial_days : ℕ := 30
  let additional_days : ℕ := 12
  let total_days : ℕ := initial_days + additional_days
  total_training_hours hours_per_day total_days = 210 := by
  sorry

#check training_hours_calculation

end training_hours_calculation_l2631_263153


namespace arithmetic_sequence_common_difference_l2631_263165

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum1 : a 3 + a 6 = 11)
  (h_sum2 : a 5 + a 8 = 39) :
  ∃ d : ℝ, d = 7 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l2631_263165


namespace bus_journey_distance_l2631_263178

theorem bus_journey_distance :
  ∀ (D s : ℝ),
    -- Original expected travel time
    (D / s = 2 + 1 + (3 * (D - 2 * s)) / (2 * s)) →
    -- Actual travel time (6 hours late)
    (2 + 1 + (3 * (D - 2 * s)) / (2 * s) = D / s + 6) →
    -- Travel time if delay occurred 120 miles further (4 hours late)
    ((2 * s + 120) / s + 1 + (3 * (D - 2 * s - 120)) / (2 * s) = D / s + 4) →
    -- Bus continues at 2/3 of original speed after delay
    (D - 2 * s) / ((2/3) * s) = (3 * (D - 2 * s)) / (2 * s) →
    D = 720 :=
by sorry

end bus_journey_distance_l2631_263178


namespace arctan_sum_three_seven_l2631_263147

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end arctan_sum_three_seven_l2631_263147


namespace couples_in_club_is_three_l2631_263179

/-- Represents a book club with couples and single members -/
structure BookClub where
  weeksPerYear : ℕ
  ronPicksPerYear : ℕ
  singleMembers : ℕ

/-- Calculates the number of couples in the book club -/
def couplesInClub (club : BookClub) : ℕ :=
  (club.weeksPerYear - (2 * club.ronPicksPerYear + club.singleMembers * club.ronPicksPerYear)) / (2 * club.ronPicksPerYear)

/-- Theorem stating that the number of couples in the specified book club is 3 -/
theorem couples_in_club_is_three (club : BookClub) 
  (h1 : club.weeksPerYear = 52)
  (h2 : club.ronPicksPerYear = 4)
  (h3 : club.singleMembers = 5) : 
  couplesInClub club = 3 := by
  sorry

end couples_in_club_is_three_l2631_263179


namespace round_trip_no_car_percentage_l2631_263111

theorem round_trip_no_car_percentage
  (total_round_trip : ℝ)
  (round_trip_with_car : ℝ)
  (h1 : round_trip_with_car = 25)
  (h2 : total_round_trip = 62.5) :
  total_round_trip - round_trip_with_car = 37.5 := by
sorry

end round_trip_no_car_percentage_l2631_263111


namespace min_value_expression_min_value_attained_l2631_263160

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((2 * x^2 + y^2) * (4 * x^2 + y^2)).sqrt) / (x * y) ≥ 3 :=
sorry

theorem min_value_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    (((2 * x₀^2 + y₀^2) * (4 * x₀^2 + y₀^2)).sqrt) / (x₀ * y₀) = 3 :=
sorry

end min_value_expression_min_value_attained_l2631_263160


namespace finite_triples_satisfying_equation_l2631_263140

theorem finite_triples_satisfying_equation : 
  ∃ (S : Set (ℕ × ℕ × ℕ)), Finite S ∧ 
  ∀ (a b c : ℕ), (a * b * c = 2009 * (a + b + c) ∧ a > 0 ∧ b > 0 ∧ c > 0) ↔ (a, b, c) ∈ S :=
sorry

end finite_triples_satisfying_equation_l2631_263140


namespace max_value_theorem_l2631_263176

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  x + Real.sqrt (x * y) + (x * y * z) ^ (1/4) ≤ 7/6 := by
sorry

end max_value_theorem_l2631_263176


namespace arctan_equality_l2631_263113

theorem arctan_equality : 4 * Real.arctan (1/5) - Real.arctan (1/239) = π/4 := by
  sorry

end arctan_equality_l2631_263113


namespace factor_implies_c_value_l2631_263177

theorem factor_implies_c_value (c : ℚ) :
  (∀ x : ℚ, (x + 7) ∣ (c * x^3 + 23 * x^2 - 3 * c * x + 45)) →
  c = 586 / 161 := by
sorry

end factor_implies_c_value_l2631_263177


namespace geometric_sequence_implies_b_eq_4_b_eq_4_not_sufficient_geometric_sequence_sufficient_not_necessary_l2631_263132

/-- A geometric sequence with first term 1, fifth term 16, and middle terms a, b, c -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ a = q ∧ b = q^2 ∧ c = q^3 ∧ 16 = q^4

/-- The statement that b = 4 is a necessary condition for the geometric sequence -/
theorem geometric_sequence_implies_b_eq_4 (a b c : ℝ) :
  is_geometric_sequence a b c → b = 4 :=
sorry

/-- The statement that b = 4 is not a sufficient condition for the geometric sequence -/
theorem b_eq_4_not_sufficient (a b c : ℝ) :
  b = 4 → ¬(∀ a c : ℝ, is_geometric_sequence a b c) :=
sorry

/-- The main theorem stating that the geometric sequence condition is sufficient but not necessary for b = 4 -/
theorem geometric_sequence_sufficient_not_necessary :
  (∃ a b c : ℝ, is_geometric_sequence a b c ∧ b = 4) ∧
  (∃ b : ℝ, b = 4 ∧ ¬(∀ a c : ℝ, is_geometric_sequence a b c)) :=
sorry

end geometric_sequence_implies_b_eq_4_b_eq_4_not_sufficient_geometric_sequence_sufficient_not_necessary_l2631_263132


namespace g_properties_l2631_263108

noncomputable def g (x : ℝ) : ℝ := (4 * Real.sin x ^ 4 + 7 * Real.cos x ^ 2) / (4 * Real.cos x ^ 4 + Real.sin x ^ 2)

theorem g_properties :
  (∀ k : ℤ, g (Real.pi / 3 + k * Real.pi) = 4 ∧ g (-Real.pi / 3 + k * Real.pi) = 4 ∧ g (Real.pi / 2 + k * Real.pi) = 4) ∧
  (∀ x : ℝ, g x ≥ 7 / 4) ∧
  (∀ x : ℝ, g x ≤ 63 / 15) ∧
  (∃ x : ℝ, g x = 7 / 4) ∧
  (∃ x : ℝ, g x = 63 / 15) := by
  sorry

end g_properties_l2631_263108


namespace probability_all_heads_or_tails_proof_l2631_263149

/-- The probability of getting all heads or all tails when flipping six fair coins -/
def probability_all_heads_or_tails : ℚ := 1 / 32

/-- The number of fair coins being flipped -/
def num_coins : ℕ := 6

/-- A fair coin has two possible outcomes -/
def outcomes_per_coin : ℕ := 2

/-- The total number of possible outcomes when flipping the coins -/
def total_outcomes : ℕ := outcomes_per_coin ^ num_coins

/-- The number of favorable outcomes (all heads or all tails) -/
def favorable_outcomes : ℕ := 2

theorem probability_all_heads_or_tails_proof :
  probability_all_heads_or_tails = favorable_outcomes / total_outcomes :=
sorry

end probability_all_heads_or_tails_proof_l2631_263149


namespace faye_coloring_books_l2631_263195

/-- Calculates the number of coloring books Faye bought -/
def coloring_books_bought (initial : ℕ) (given_away : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial - given_away)

theorem faye_coloring_books :
  coloring_books_bought 34 3 79 = 48 := by
  sorry

end faye_coloring_books_l2631_263195


namespace intersection_implies_m_value_subset_complement_iff_m_range_l2631_263125

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 3 ≤ x ∧ x ≤ m}

-- Theorem 1
theorem intersection_implies_m_value :
  ∀ m : ℝ, (A ∩ B m = {x : ℝ | 2 ≤ x ∧ x ≤ 4}) → m = 5 := by
  sorry

-- Theorem 2
theorem subset_complement_iff_m_range :
  ∀ m : ℝ, A ⊆ (Set.univ \ B m) ↔ m < -2 ∨ m > 7 := by
  sorry

end intersection_implies_m_value_subset_complement_iff_m_range_l2631_263125


namespace sin_double_angle_from_infinite_sum_l2631_263170

theorem sin_double_angle_from_infinite_sum (θ : ℝ) 
  (h : ∑' n, (Real.sin θ)^(2*n) = 4) : 
  Real.sin (2 * θ) = Real.sqrt 3 / 2 := by
sorry

end sin_double_angle_from_infinite_sum_l2631_263170


namespace distinct_prime_factors_count_l2631_263168

def product : ℕ := 95 * 97 * 99 * 101

theorem distinct_prime_factors_count :
  (Nat.factors product).toFinset.card = 6 :=
sorry

end distinct_prime_factors_count_l2631_263168


namespace second_box_difference_l2631_263196

/-- Represents the amount of cereal in ounces for each box. -/
structure CerealBoxes where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Defines the properties of the cereal boxes based on the problem conditions. -/
def validCerealBoxes (boxes : CerealBoxes) : Prop :=
  boxes.first = 14 ∧
  boxes.second = boxes.first / 2 ∧
  boxes.second < boxes.third ∧
  boxes.first + boxes.second + boxes.third = 33

/-- Theorem stating that the difference between the third and second box is 5 ounces. -/
theorem second_box_difference (boxes : CerealBoxes) 
  (h : validCerealBoxes boxes) : boxes.third - boxes.second = 5 := by
  sorry

end second_box_difference_l2631_263196


namespace simplify_and_rationalize_l2631_263163

theorem simplify_and_rationalize :
  (Real.sqrt 2 / Real.sqrt 5) * (Real.sqrt 8 / Real.sqrt 9) * (Real.sqrt 3 / Real.sqrt 7) = 
  (4 * Real.sqrt 105) / 105 := by
  sorry

end simplify_and_rationalize_l2631_263163


namespace shopping_tax_theorem_l2631_263123

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def total_tax_percentage (clothing_percent : ℝ) (food_percent : ℝ) (other_percent : ℝ)
                         (clothing_tax_rate : ℝ) (food_tax_rate : ℝ) (other_tax_rate : ℝ) : ℝ :=
  (clothing_percent * clothing_tax_rate + food_percent * food_tax_rate + other_percent * other_tax_rate) * 100

/-- Theorem stating that the total tax percentage is 4.8% given the specified conditions -/
theorem shopping_tax_theorem :
  total_tax_percentage 0.6 0.1 0.3 0.04 0 0.08 = 4.8 := by
  sorry

#eval total_tax_percentage 0.6 0.1 0.3 0.04 0 0.08

end shopping_tax_theorem_l2631_263123


namespace blue_ball_count_l2631_263117

/-- Given a bag of glass balls with yellow and blue colors -/
structure GlassBallBag where
  total : ℕ
  yellowProb : ℝ

/-- Theorem: In a bag of 80 glass balls where the probability of picking a yellow ball is 0.25,
    the number of blue balls is 60 -/
theorem blue_ball_count (bag : GlassBallBag)
    (h_total : bag.total = 80)
    (h_yellow_prob : bag.yellowProb = 0.25) :
    (bag.total : ℝ) * (1 - bag.yellowProb) = 60 := by
  sorry


end blue_ball_count_l2631_263117


namespace inequality_proof_1_inequality_proof_2_l2631_263137

theorem inequality_proof_1 (x : ℝ) : 
  abs (x + 2) + abs (x - 2) > 6 ↔ x < -3 ∨ x > 3 := by sorry

theorem inequality_proof_2 (x : ℝ) : 
  abs (2*x - 1) - abs (x - 3) > 5 ↔ x < -7 ∨ x > 3 := by sorry

end inequality_proof_1_inequality_proof_2_l2631_263137


namespace root_difference_ratio_l2631_263142

/-- Given an equation x^4 - 7x - 3 = 0 with exactly two real roots a and b where a > b,
    the expression (a - b) / (a^4 - b^4) equals 1/7 -/
theorem root_difference_ratio (a b : ℝ) : 
  a > b → 
  a^4 - 7*a - 3 = 0 → 
  b^4 - 7*b - 3 = 0 → 
  (a - b) / (a^4 - b^4) = 1/7 := by
sorry

end root_difference_ratio_l2631_263142


namespace translation_theorem_l2631_263182

/-- Represents a point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  { x := p.x + dx, y := p.y }

/-- Translates a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  { x := p.x, y := p.y + dy }

theorem translation_theorem :
  let M : Point := { x := -4, y := 3 }
  let M1 := translateHorizontal M (-3)
  let M2 := translateVertical M1 2
  M2 = { x := -7, y := 5 } := by
  sorry

end translation_theorem_l2631_263182


namespace intersection_M_N_l2631_263167

def M : Set ℤ := {1, 2, 3, 4, 5, 6}

def N : Set ℤ := {x | -2 < x ∧ x < 5}

theorem intersection_M_N : M ∩ N = {1, 2, 3, 4} := by
  sorry

end intersection_M_N_l2631_263167


namespace min_value_of_fraction_l2631_263139

theorem min_value_of_fraction (x y : ℝ) 
  (hx : -3 ≤ x ∧ x ≤ 1) 
  (hy : -1 ≤ y ∧ y ≤ 3) 
  (hx_nonzero : x ≠ 0) : 
  (x + y) / x ≥ -2 := by
  sorry

end min_value_of_fraction_l2631_263139


namespace correct_factorization_l2631_263114

theorem correct_factorization (m : ℤ) : m^3 + m = m * (m^2 + 1) := by
  sorry

end correct_factorization_l2631_263114


namespace no_valid_y_exists_l2631_263198

theorem no_valid_y_exists : ¬∃ (y : ℝ), y^3 + y - 2 = 0 ∧ abs y < 1 := by
  sorry

end no_valid_y_exists_l2631_263198


namespace christopher_sugar_substitute_cost_l2631_263118

/-- Represents the cost calculation for Christopher's sugar substitute usage --/
theorem christopher_sugar_substitute_cost :
  let packets_per_coffee : ℕ := 1
  let coffees_per_day : ℕ := 2
  let packets_per_box : ℕ := 30
  let cost_per_box : ℚ := 4
  let days : ℕ := 90

  let daily_usage : ℕ := packets_per_coffee * coffees_per_day
  let total_packets : ℕ := daily_usage * days
  let boxes_needed : ℕ := (total_packets + packets_per_box - 1) / packets_per_box
  let total_cost : ℚ := cost_per_box * boxes_needed

  total_cost = 24 :=
by
  sorry


end christopher_sugar_substitute_cost_l2631_263118


namespace coffee_maker_price_l2631_263110

theorem coffee_maker_price (sale_price : ℝ) (discount : ℝ) (original_price : ℝ) : 
  sale_price = 70 → discount = 20 → original_price = sale_price + discount → original_price = 90 := by
  sorry

end coffee_maker_price_l2631_263110


namespace gcd_360_1260_l2631_263109

theorem gcd_360_1260 : Nat.gcd 360 1260 = 180 := by
  sorry

end gcd_360_1260_l2631_263109


namespace fibonacci_like_sequence_l2631_263148

theorem fibonacci_like_sequence (a : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n)
  (h2 : a 11 = 157) :
  a 1 = 3 := by
sorry

end fibonacci_like_sequence_l2631_263148


namespace least_subtraction_for_divisibility_l2631_263101

theorem least_subtraction_for_divisibility :
  ∃! r : ℕ, r < 47 ∧ (3674958423 - r) % 47 = 0 ∧ ∀ s : ℕ, s < r → (3674958423 - s) % 47 ≠ 0 :=
by sorry

end least_subtraction_for_divisibility_l2631_263101


namespace hyperbola_eccentricity_range_l2631_263102

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let E := {(x, y) : ℝ × ℝ | x^2/a^2 - y^2/b^2 = 1}
  let A := (a, 0)
  let C := {(x, y) : ℝ × ℝ | y^2 = 8*a*x}
  let F := (2*a, 0)
  let asymptote := {(x, y) : ℝ × ℝ | y = b/a * x ∨ y = -b/a * x}
  ∃ P ∈ asymptote, (A.1 - P.1) * (F.1 - P.1) + (A.2 - P.2) * (F.2 - P.2) = 0 →
  let e := Real.sqrt (1 + b^2/a^2)
  1 < e ∧ e ≤ 3 * Real.sqrt 2 / 4 :=
sorry

end hyperbola_eccentricity_range_l2631_263102


namespace rosa_bonheur_birthday_l2631_263194

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the number of leap years between two years -/
def leapYearCount (startYear endYear : Nat) : Nat :=
  let totalYears := endYear - startYear
  let potentialLeapYears := totalYears / 4
  potentialLeapYears - 1 -- Excluding 1900

/-- Calculates the day of the week given a starting day and number of days passed -/
def calculateDay (startDay : DayOfWeek) (daysPassed : Nat) : DayOfWeek :=
  match (daysPassed % 7) with
  | 0 => startDay
  | 1 => DayOfWeek.Sunday
  | 2 => DayOfWeek.Monday
  | 3 => DayOfWeek.Tuesday
  | 4 => DayOfWeek.Wednesday
  | 5 => DayOfWeek.Thursday
  | 6 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem rosa_bonheur_birthday 
  (anniversaryDay : DayOfWeek)
  (h : anniversaryDay = DayOfWeek.Wednesday) :
  calculateDay anniversaryDay 261 = DayOfWeek.Sunday := by
  sorry

#check rosa_bonheur_birthday

end rosa_bonheur_birthday_l2631_263194


namespace bryans_bookshelves_l2631_263154

theorem bryans_bookshelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 34) (h2 : books_per_shelf = 17) :
  total_books / books_per_shelf = 2 :=
by sorry

end bryans_bookshelves_l2631_263154


namespace tan_alpha_plus_pi_fourth_l2631_263106

theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h : 2 * Real.sin (2 * α) = 1 + Real.cos (2 * α)) : 
  Real.tan (α + π / 4) = -1 ∨ Real.tan (α + π / 4) = 3 := by
sorry

end tan_alpha_plus_pi_fourth_l2631_263106


namespace monomial_simplification_l2631_263141

theorem monomial_simplification (a : ℕ) (M : ℕ) (h1 : a = 100) (h2 : M = a) :
  (M : ℚ) / (a + 1 : ℚ) - 1 / ((a^2 : ℚ) + a) = 99 / 100 := by
  sorry

end monomial_simplification_l2631_263141


namespace lyra_remaining_budget_l2631_263156

/-- Calculates the remaining budget after Lyra's purchases --/
theorem lyra_remaining_budget (budget : ℝ) (chicken_price : ℝ) (beef_price : ℝ) (beef_weight : ℝ)
  (soup_price : ℝ) (soup_cans : ℕ) (milk_price : ℝ) (milk_discount : ℝ) :
  budget = 80 →
  chicken_price = 12 →
  beef_price = 3 →
  beef_weight = 4.5 →
  soup_price = 2 →
  soup_cans = 3 →
  milk_price = 4 →
  milk_discount = 0.1 →
  budget - (chicken_price + beef_price * beef_weight + 
    (soup_price * ↑soup_cans / 2) + milk_price * (1 - milk_discount)) = 47.9 := by
  sorry

#eval (80 : ℚ) - (12 + 3 * (9/2) + (2 * 3 / 2) + 4 * (1 - 1/10))

end lyra_remaining_budget_l2631_263156


namespace intersection_A_B_zero_range_of_m_l2631_263133

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 < 0
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

-- Theorem 1: Intersection of A and B when m = 0
theorem intersection_A_B_zero : A ∩ B 0 = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem 2: Range of m when q is necessary but not sufficient for p
theorem range_of_m (h : ∀ x, p x → q x m) : 
  m ≤ -2 ∨ m ≥ 4 := by sorry

end intersection_A_B_zero_range_of_m_l2631_263133


namespace polynomial_identity_l2631_263175

theorem polynomial_identity (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (x - 1)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a + a₂ + a₄ + a₆)^2 - (a₁ + a₃ + a₅ + a₇)^2 = 0 := by
sorry

end polynomial_identity_l2631_263175


namespace smallest_n_value_l2631_263122

/-- Represents a rectangular block made of 1-cm cubes -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in the block -/
def Block.totalCubes (b : Block) : ℕ := b.length * b.width * b.height

/-- Calculates the number of invisible cubes when three faces are visible -/
def Block.invisibleCubes (b : Block) : ℕ := (b.length - 1) * (b.width - 1) * (b.height - 1)

/-- Theorem stating the smallest possible value of N -/
theorem smallest_n_value (b : Block) (h : b.invisibleCubes = 300) :
  ∃ (min_b : Block), min_b.invisibleCubes = 300 ∧
    min_b.totalCubes ≤ b.totalCubes ∧
    min_b.totalCubes = 468 := by
  sorry

end smallest_n_value_l2631_263122


namespace expected_sales_after_price_change_l2631_263189

/-- Represents the relationship between price and sales of blenders -/
structure BlenderSales where
  price : ℝ
  units : ℝ

/-- The constant of proportionality for the inverse relationship -/
def k : ℝ := 15 * 500

/-- The inverse proportionality relationship between price and sales -/
def inverse_proportional (bs : BlenderSales) : Prop :=
  bs.price * bs.units = k

/-- The new price after discount -/
def new_price : ℝ := 1000 * (1 - 0.1)

/-- Theorem stating the expected sales under the new pricing scheme -/
theorem expected_sales_after_price_change 
  (initial : BlenderSales) 
  (h_initial : initial.price = 500 ∧ initial.units = 15) 
  (h_inverse : inverse_proportional initial) :
  ∃ (new : BlenderSales), 
    new.price = new_price ∧ 
    inverse_proportional new ∧ 
    (8 ≤ new.units ∧ new.units < 9) := by
  sorry

end expected_sales_after_price_change_l2631_263189


namespace triangle_inscription_exists_l2631_263105

-- Define the triangle type
structure Triangle :=
  (A B C : Point)

-- Define congruence between triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define inscribed triangle
def inscribed (inner outer : Triangle) : Prop := sorry

-- Theorem statement
theorem triangle_inscription_exists (ABC : Triangle) :
  ∃ (PQR : Triangle), ∃ (XYZ : Triangle),
    congruent XYZ ABC ∧ inscribed XYZ PQR := by sorry

end triangle_inscription_exists_l2631_263105


namespace dvd_pack_cost_l2631_263180

theorem dvd_pack_cost (total_cost : ℕ) (num_packs : ℕ) (cost_per_pack : ℕ) :
  total_cost = 2673 →
  num_packs = 33 →
  cost_per_pack = total_cost / num_packs →
  cost_per_pack = 81 := by
  sorry

end dvd_pack_cost_l2631_263180


namespace complement_A_intersect_B_l2631_263126

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {y | y > 0}

-- Define set B
def B : Set ℝ := {-2, -1, 1, 2}

-- Theorem statement
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {-2, -1} := by sorry

end complement_A_intersect_B_l2631_263126


namespace cube_root_of_sum_l2631_263155

theorem cube_root_of_sum (x y : ℝ) : 
  (Real.sqrt (x - 1) + (y + 2)^2 = 0) → 
  (x + y)^(1/3 : ℝ) = -1 := by
sorry

end cube_root_of_sum_l2631_263155


namespace summer_camp_selection_probability_l2631_263174

theorem summer_camp_selection_probability :
  let total_students : ℕ := 9
  let male_students : ℕ := 5
  let female_students : ℕ := 4
  let selected_students : ℕ := 5
  let min_per_gender : ℕ := 2

  let total_combinations := Nat.choose total_students selected_students
  let valid_combinations := Nat.choose male_students min_per_gender * Nat.choose female_students (selected_students - min_per_gender) +
                            Nat.choose male_students (selected_students - min_per_gender) * Nat.choose female_students min_per_gender

  (valid_combinations : ℚ) / total_combinations = 50 / 63 :=
by sorry

end summer_camp_selection_probability_l2631_263174


namespace simplify_expression_l2631_263127

theorem simplify_expression :
  ∃ (a b c : ℕ+),
    ((Real.sqrt 3 - 1) ^ (2 - Real.sqrt 2)) / ((Real.sqrt 3 + 1) ^ (2 + Real.sqrt 2)) = a - b * Real.sqrt c ∧
    ¬ ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ k > 1 ∧ p ^ k ∣ c.val ∧
    a = 21 ∧ b = 12 ∧ c = 3 :=
by sorry

end simplify_expression_l2631_263127


namespace max_a_for_defined_f_l2631_263161

-- Define the function g(x) = |x-2| + |x-a|
def g (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x - a|

-- State the theorem
theorem max_a_for_defined_f :
  (∃ (a_max : ℝ), (∀ (a : ℝ), (∀ (x : ℝ), g a x ≥ 2 * a) → a ≤ a_max) ∧
                  (∀ (x : ℝ), g a_max x ≥ 2 * a_max) ∧
                  a_max = 2/3) :=
sorry

end max_a_for_defined_f_l2631_263161

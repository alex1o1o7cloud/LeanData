import Mathlib

namespace NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_of_120_l2308_230841

theorem sum_two_smallest_prime_factors_of_120 :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧
  p ∣ 120 ∧ q ∣ 120 ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 120 → r = p ∨ r ≥ q) ∧
  p + q = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_of_120_l2308_230841


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_7n_plus_3_l2308_230853

theorem max_gcd_13n_plus_4_7n_plus_3 :
  ∃ (k : ℕ+), ∀ (n : ℕ+), Nat.gcd (13 * n + 4) (7 * n + 3) ≤ k ∧
  ∃ (m : ℕ+), Nat.gcd (13 * m + 4) (7 * m + 3) = k ∧
  k = 11 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_7n_plus_3_l2308_230853


namespace NUMINAMATH_CALUDE_question_arrangement_l2308_230857

/-- Represents the number of ways to arrange 6 questions -/
def arrangement_count : ℕ := 144

/-- The number of multiple-choice questions -/
def total_questions : ℕ := 6

/-- The number of easy questions -/
def easy_questions : ℕ := 2

/-- The number of medium questions -/
def medium_questions : ℕ := 2

/-- The number of difficult questions -/
def difficult_questions : ℕ := 2

theorem question_arrangement :
  (easy_questions = 2) →
  (medium_questions = 2) →
  (difficult_questions = 2) →
  (total_questions = easy_questions + medium_questions + difficult_questions) →
  arrangement_count = 144 := by
  sorry

end NUMINAMATH_CALUDE_question_arrangement_l2308_230857


namespace NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_vertex_l2308_230804

/-- Given a hyperbola with equation x²/16 - y²/9 = 1, 
    prove that the standard equation of a parabola 
    with its focus at the right vertex of this hyperbola is y² = 16x -/
theorem parabola_equation_from_hyperbola_vertex (x y : ℝ) : 
  (x^2 / 16 - y^2 / 9 = 1) → 
  ∃ (x₀ y₀ : ℝ), 
    (x₀ > 0 ∧ y₀ = 0 ∧ x₀^2 / 16 - y₀^2 / 9 = 1) ∧ 
    (∀ (x' y' : ℝ), (y' - y₀)^2 = 16 * (x' - x₀)) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_vertex_l2308_230804


namespace NUMINAMATH_CALUDE_x_range_lower_bound_l2308_230821

theorem x_range_lower_bound (x y : ℝ) (h : x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0) :
  x ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_x_range_lower_bound_l2308_230821


namespace NUMINAMATH_CALUDE_darcy_laundry_theorem_l2308_230822

/-- Given the number of shirts and shorts Darcy has, and the number he has folded,
    calculate the number of remaining pieces to fold. -/
def remaining_to_fold (total_shirts : ℕ) (total_shorts : ℕ) 
                      (folded_shirts : ℕ) (folded_shorts : ℕ) : ℕ :=
  (total_shirts - folded_shirts) + (total_shorts - folded_shorts)

/-- Theorem stating that with 20 shirts and 8 shorts, 
    if 12 shirts and 5 shorts are folded, 
    11 pieces remain to be folded. -/
theorem darcy_laundry_theorem : 
  remaining_to_fold 20 8 12 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_darcy_laundry_theorem_l2308_230822


namespace NUMINAMATH_CALUDE_angle_sine_equivalence_l2308_230833

theorem angle_sine_equivalence (A B C : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  A > B ↔ Real.sin A > Real.sin B :=
sorry

end NUMINAMATH_CALUDE_angle_sine_equivalence_l2308_230833


namespace NUMINAMATH_CALUDE_distinct_bracelets_count_l2308_230830

/-- Represents a bead color -/
inductive BeadColor
| Red
| Blue
| Purple

/-- Represents a bracelet as a circular arrangement of beads -/
def Bracelet := List BeadColor

/-- Checks if two bracelets are equivalent under rotation and reflection -/
def are_equivalent (b1 b2 : Bracelet) : Bool :=
  sorry

/-- Counts the number of beads of each color in a bracelet -/
def count_beads (b : Bracelet) : Nat × Nat × Nat :=
  sorry

/-- Generates all possible bracelets with 2 red, 2 blue, and 2 purple beads -/
def generate_bracelets : List Bracelet :=
  sorry

/-- Counts the number of distinct bracelets -/
def count_distinct_bracelets : Nat :=
  sorry

/-- Theorem: The number of distinct bracelets with 2 red, 2 blue, and 2 purple beads is 11 -/
theorem distinct_bracelets_count :
  count_distinct_bracelets = 11 := by
  sorry

end NUMINAMATH_CALUDE_distinct_bracelets_count_l2308_230830


namespace NUMINAMATH_CALUDE_distance_between_vertices_l2308_230846

-- Define the equation of the parabolas
def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 1) = 5

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -2)

-- Theorem stating the distance between vertices
theorem distance_between_vertices :
  let (x1, y1) := vertex1
  let (x2, y2) := vertex2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l2308_230846


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l2308_230871

theorem sqrt_sum_equality : 
  let a : ℕ := 49
  let b : ℕ := 64
  let c : ℕ := 100
  Real.sqrt a + Real.sqrt b + Real.sqrt c = 
    Real.sqrt (219 + Real.sqrt 10080 + Real.sqrt 12600 + Real.sqrt 35280) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l2308_230871


namespace NUMINAMATH_CALUDE_ellipse_equation_l2308_230834

/-- An ellipse with given foci and major axis length has the specified equation -/
theorem ellipse_equation (F₁ F₂ : ℝ × ℝ) (major_axis_length : ℝ) :
  F₁ = (-1, 0) →
  F₂ = (1, 0) →
  major_axis_length = 10 →
  ∀ x y : ℝ, (x^2 / 25 + y^2 / 24 = 1) ↔ (x, y) ∈ {p : ℝ × ℝ | Real.sqrt ((p.1 - F₁.1)^2 + (p.2 - F₁.2)^2) + Real.sqrt ((p.1 - F₂.1)^2 + (p.2 - F₂.2)^2) = major_axis_length} :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2308_230834


namespace NUMINAMATH_CALUDE_smaller_equals_larger_l2308_230818

/-- A circle with an inscribed rectangle and a smaller rectangle -/
structure InscribedRectangles where
  /-- The radius of the circle -/
  r : ℝ
  /-- Half-width of the larger rectangle -/
  a : ℝ
  /-- Half-height of the larger rectangle -/
  b : ℝ
  /-- Proportion of the smaller rectangle's side to the larger rectangle's side -/
  x : ℝ
  /-- The larger rectangle is inscribed in the circle -/
  inscribed : r^2 = a^2 + b^2
  /-- The smaller rectangle has two vertices on the circle -/
  smaller_on_circle : r^2 = (a*x)^2 + (b*x)^2
  /-- The smaller rectangle's side coincides with the larger rectangle's side -/
  coincide : 0 < x ∧ x ≤ 1

/-- The area of the smaller rectangle is equal to the area of the larger rectangle -/
theorem smaller_equals_larger (ir : InscribedRectangles) : 
  (ir.a * ir.x) * (ir.b * ir.x) = ir.a * ir.b := by
  sorry

end NUMINAMATH_CALUDE_smaller_equals_larger_l2308_230818


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2308_230806

theorem expand_and_simplify (x : ℝ) : 2*x*(x-4) - (2*x-3)*(x+2) = -9*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2308_230806


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l2308_230858

/-- Represents the lengths of external diagonals of a right regular prism -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0

/-- Checks if given lengths can be external diagonals of a right regular prism -/
def isValidExternalDiagonals (d : ExternalDiagonals) : Prop :=
  d.a^2 + d.b^2 > d.c^2 ∧ d.b^2 + d.c^2 > d.a^2 ∧ d.a^2 + d.c^2 > d.b^2

theorem invalid_external_diagonals :
  ¬ isValidExternalDiagonals ⟨4, 5, 7, by norm_num, by norm_num, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l2308_230858


namespace NUMINAMATH_CALUDE_donation_to_first_home_l2308_230896

theorem donation_to_first_home 
  (total_donation : ℝ) 
  (second_home_donation : ℝ) 
  (third_home_donation : ℝ) 
  (h1 : total_donation = 700)
  (h2 : second_home_donation = 225)
  (h3 : third_home_donation = 230) :
  total_donation - second_home_donation - third_home_donation = 245 :=
by sorry

end NUMINAMATH_CALUDE_donation_to_first_home_l2308_230896


namespace NUMINAMATH_CALUDE_piggy_bank_coins_l2308_230897

theorem piggy_bank_coins (quarters dimes nickels : ℕ) : 
  dimes = quarters + 3 →
  nickels = quarters - 6 →
  quarters + dimes + nickels = 63 →
  quarters = 22 := by
sorry

end NUMINAMATH_CALUDE_piggy_bank_coins_l2308_230897


namespace NUMINAMATH_CALUDE_ellipse_equation_l2308_230891

/-- Given an ellipse with one focus at (√3, 0) and a = 2b, its standard equation is x²/4 + y² = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a = 2*b) (h2 : a^2 - b^2 = 3) :
  ∀ x y : ℝ, x^2/4 + y^2 = 1 ↔ x^2/(4*b^2) + y^2/b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2308_230891


namespace NUMINAMATH_CALUDE_hyperbola_symmetry_l2308_230876

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola x² - y² = a² -/
def Hyperbola (a : ℝ) : Set Point :=
  {p : Point | p.x^2 - p.y^2 = a^2}

/-- Represents the line y = x - 2 -/
def SymmetryLine : Set Point :=
  {p : Point | p.y = p.x - 2}

/-- Represents the line 2x + 3y = 6 -/
def TangentLine : Set Point :=
  {p : Point | 2 * p.x + 3 * p.y = 6}

/-- Defines symmetry about a line -/
def SymmetricPoint (p : Point) : Point :=
  ⟨p.y + 2, p.x - 2⟩

/-- Defines the curve C₂ symmetric to C₁ about the symmetry line -/
def C₂ (a : ℝ) : Set Point :=
  {p : Point | SymmetricPoint p ∈ Hyperbola a}

/-- States that the tangent line is tangent to C₂ -/
def IsTangent (a : ℝ) : Prop :=
  ∃ p : Point, p ∈ C₂ a ∧ p ∈ TangentLine

theorem hyperbola_symmetry (a : ℝ) (h : a > 0) (h_tangent : IsTangent a) : 
  a = 8 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_symmetry_l2308_230876


namespace NUMINAMATH_CALUDE_ott_fraction_of_total_l2308_230809

/-- Represents the amount of money each person has -/
structure Money where
  loki : ℚ
  moe : ℚ
  nick : ℚ
  ott : ℚ

/-- The initial state of money distribution -/
def initial_money : Money := {
  loki := 5,
  moe := 5,
  nick := 3,
  ott := 0
}

/-- The amount of money given to Ott -/
def money_given : ℚ := 1

/-- The state of money after giving to Ott -/
def final_money : Money := {
  loki := initial_money.loki - money_given,
  moe := initial_money.moe - money_given,
  nick := initial_money.nick - money_given,
  ott := initial_money.ott + 3 * money_given
}

/-- The theorem to prove -/
theorem ott_fraction_of_total (m : Money := final_money) :
  m.ott / (m.loki + m.moe + m.nick + m.ott) = 3 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ott_fraction_of_total_l2308_230809


namespace NUMINAMATH_CALUDE_no_integer_function_satisfies_condition_l2308_230874

theorem no_integer_function_satisfies_condition :
  ¬ ∃ (f : ℤ → ℤ), ∀ (x y : ℤ), f (x + f y) = f x - y :=
by sorry

end NUMINAMATH_CALUDE_no_integer_function_satisfies_condition_l2308_230874


namespace NUMINAMATH_CALUDE_min_value_of_line_through_point_l2308_230803

/-- Given a line ax + by - 1 = 0 passing through the point (1, 2),
    where a and b are positive real numbers,
    the minimum value of 1/a + 2/b is 9. -/
theorem min_value_of_line_through_point (a b : ℝ) : 
  a > 0 → b > 0 → a + 2*b = 1 → (1/a + 2/b) ≥ 9 := by sorry

end NUMINAMATH_CALUDE_min_value_of_line_through_point_l2308_230803


namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_l2308_230845

/-- Given a polynomial g(x) = 2x^7 - 3x^5 + px^2 + 2x - 6 where g(5) = 10, 
    prove that g(-5) = -301383 -/
theorem polynomial_value_at_negative (p : ℝ) : 
  (fun x : ℝ => 2*x^7 - 3*x^5 + p*x^2 + 2*x - 6) 5 = 10 → 
  (fun x : ℝ => 2*x^7 - 3*x^5 + p*x^2 + 2*x - 6) (-5) = -301383 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_negative_l2308_230845


namespace NUMINAMATH_CALUDE_eg_length_l2308_230816

/-- A quadrilateral with specific side lengths -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℝ)
  (EG : ℕ)

/-- The theorem stating the length of EG in the specific quadrilateral -/
theorem eg_length (q : Quadrilateral) 
  (h1 : q.EF = 7)
  (h2 : q.FG = 13)
  (h3 : q.GH = 7)
  (h4 : q.HE = 11) :
  q.EG = 13 := by
  sorry


end NUMINAMATH_CALUDE_eg_length_l2308_230816


namespace NUMINAMATH_CALUDE_f_is_direct_proportion_l2308_230854

/-- A function f : ℝ → ℝ is a direct proportion function if there exists a constant k such that f(x) = k * x for all x. -/
def IsDirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function f(x) = 2x -/
def f : ℝ → ℝ := fun x ↦ 2 * x

/-- Theorem: The function f(x) = 2x is a direct proportion function -/
theorem f_is_direct_proportion : IsDirectProportion f := by
  sorry

end NUMINAMATH_CALUDE_f_is_direct_proportion_l2308_230854


namespace NUMINAMATH_CALUDE_min_c_value_l2308_230870

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b) (hbc : b < c) (hsum : a + b + c = 1503)
  (hunique : ∃! (x y : ℝ), 2 * x + y = 2008 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 496 ∧ ∃ (a' b' c' : ℕ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' < b' ∧ b' < c' ∧
    a' + b' + c' = 1503 ∧ c' = 496 ∧
    ∃! (x y : ℝ), 2 * x + y = 2008 ∧ y = |x - a'| + |x - b'| + |x - c'| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l2308_230870


namespace NUMINAMATH_CALUDE_expansion_terms_count_l2308_230889

/-- The number of terms in the expansion of a product of two sums -/
def num_terms_expansion (n m : ℕ) : ℕ := n * m

/-- Theorem: The expansion of a product of two sums with 4 and 5 terms respectively has 20 terms -/
theorem expansion_terms_count :
  num_terms_expansion 4 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l2308_230889


namespace NUMINAMATH_CALUDE_power_of_power_l2308_230823

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2308_230823


namespace NUMINAMATH_CALUDE_consecutive_integers_divisibility_l2308_230875

theorem consecutive_integers_divisibility : ∃ (a b c : ℕ), 
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧  -- positive integers
  (b = a + 1) ∧ (c = b + 1) ∧    -- consecutive
  (a % 1 = 0) ∧                  -- a divisible by (b - a)^2
  (a % 4 = 0) ∧                  -- a divisible by (c - a)^2
  (b % 1 = 0) :=                 -- b divisible by (c - b)^2
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_divisibility_l2308_230875


namespace NUMINAMATH_CALUDE_set_intersection_problem_l2308_230856

def A : Set ℝ := {1, 2, 6}
def B : Set ℝ := {2, 4}
def C : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 5}

theorem set_intersection_problem : (A ∪ B) ∩ C = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l2308_230856


namespace NUMINAMATH_CALUDE_right_triangle_sin_A_l2308_230815

theorem right_triangle_sin_A (A B C : Real) (h1 : 3 * Real.sin A = 2 * Real.cos A) 
  (h2 : Real.cos B = 0) : Real.sin A = 2 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_A_l2308_230815


namespace NUMINAMATH_CALUDE_quadratic_rational_root_implies_even_coefficient_l2308_230865

theorem quadratic_rational_root_implies_even_coefficient
  (a b c : ℤ)
  (h_a_nonzero : a ≠ 0)
  (h_rational_root : ∃ (x : ℚ), a * x^2 + b * x + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_implies_even_coefficient_l2308_230865


namespace NUMINAMATH_CALUDE_quadratic_second_difference_constant_l2308_230882

/-- Second difference of a function f at point n -/
def second_difference (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  (f (n + 2) - f (n + 1)) - (f (n + 1) - f n)

/-- A quadratic function with linear and constant terms -/
def quadratic_function (a b : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ)^2 + a * (n : ℝ) + b

theorem quadratic_second_difference_constant (a b : ℝ) :
  ∀ n : ℕ, second_difference (quadratic_function a b) n = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_second_difference_constant_l2308_230882


namespace NUMINAMATH_CALUDE_vector_addition_l2308_230895

def a : Fin 2 → ℝ := ![3, 1]
def b : Fin 2 → ℝ := ![-2, 5]

theorem vector_addition : 2 • a + b = ![4, 7] := by sorry

end NUMINAMATH_CALUDE_vector_addition_l2308_230895


namespace NUMINAMATH_CALUDE_baker_pastries_sold_l2308_230802

/-- The number of cakes sold by the baker -/
def cakes_sold : ℕ := 78

/-- The difference between pastries and cakes sold -/
def pastry_cake_difference : ℕ := 76

/-- The number of pastries sold by the baker -/
def pastries_sold : ℕ := cakes_sold + pastry_cake_difference

theorem baker_pastries_sold : pastries_sold = 154 := by
  sorry

end NUMINAMATH_CALUDE_baker_pastries_sold_l2308_230802


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l2308_230883

theorem simplify_sqrt_sum : 
  (Real.sqrt 418 / Real.sqrt 308) + (Real.sqrt 294 / Real.sqrt 196) = 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l2308_230883


namespace NUMINAMATH_CALUDE_g_composition_result_l2308_230864

noncomputable def g (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^3 else -z^3

theorem g_composition_result :
  g (g (g (g (1 + I)))) = -134217728 - 134217728 * I :=
by sorry

end NUMINAMATH_CALUDE_g_composition_result_l2308_230864


namespace NUMINAMATH_CALUDE_unique_orthocenter_line_l2308_230838

/-- The ellipse with equation x^2/2 + y^2 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- The upper vertex of the ellipse -/
def B : ℝ × ℝ := (0, 1)

/-- The right focus of the ellipse -/
def F : ℝ × ℝ := (1, 0)

/-- A line that intersects the ellipse -/
def line_intersects_ellipse (m b : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    y₁ = m * x₁ + b ∧ y₂ = m * x₂ + b

/-- F is the orthocenter of triangle BMN -/
def F_is_orthocenter (M N : ℝ × ℝ) : Prop :=
  let (xm, ym) := M
  let (xn, yn) := N
  (1 - xn) * xm - yn * (ym - 1) = 0 ∧
  (1 - xm) * xn - ym * (yn - 1) = 0

theorem unique_orthocenter_line :
  ∃! m b : ℝ, 
    line_intersects_ellipse m b ∧
    (∀ M N : ℝ × ℝ, 
      ellipse M.1 M.2 → ellipse N.1 N.2 → 
      M.2 = m * M.1 + b → N.2 = m * N.1 + b →
      F_is_orthocenter M N) ∧
    m = 1 ∧ b = -4/3 :=
sorry

end NUMINAMATH_CALUDE_unique_orthocenter_line_l2308_230838


namespace NUMINAMATH_CALUDE_intersecting_squares_area_difference_l2308_230837

/-- Given four intersecting squares with sides 12, 9, 7, and 3,
    the difference between the sum of the areas of the largest and third largest squares
    and the sum of the areas of the second largest and smallest squares is 103. -/
theorem intersecting_squares_area_difference : 
  let a := 12 -- side length of the largest square
  let b := 9  -- side length of the second largest square
  let c := 7  -- side length of the third largest square
  let d := 3  -- side length of the smallest square
  (a ^ 2 + c ^ 2) - (b ^ 2 + d ^ 2) = 103 := by
sorry

#eval (12 ^ 2 + 7 ^ 2) - (9 ^ 2 + 3 ^ 2) -- This should evaluate to 103

end NUMINAMATH_CALUDE_intersecting_squares_area_difference_l2308_230837


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2308_230820

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point :
  let given_line : Line := { a := 3, b := 1, c := -1 }
  let parallel_line : Line := { a := 3, b := 1, c := -5 }
  let point : (ℝ × ℝ) := (1, 2)
  parallel given_line parallel_line ∧
  point_on_line point.1 point.2 parallel_line :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2308_230820


namespace NUMINAMATH_CALUDE_square_root_problem_l2308_230893

theorem square_root_problem (a b : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (a + 3)^2 = x ∧ (2*a - 6)^2 = x) →
  ((-2)^3 = b) →
  (∃ (y : ℝ), y^2 = a - b ∧ (y = 3 ∨ y = -3)) :=
by sorry

end NUMINAMATH_CALUDE_square_root_problem_l2308_230893


namespace NUMINAMATH_CALUDE_grid_rectangles_l2308_230811

theorem grid_rectangles (h : ℕ) (v : ℕ) (h_eq : h = 5) (v_eq : v = 6) :
  (h.choose 2) * (v.choose 2) = 150 := by
  sorry

end NUMINAMATH_CALUDE_grid_rectangles_l2308_230811


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l2308_230801

/-- The probability of selecting two non-defective pens from a box of 9 pens with 3 defective pens -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 9)
  (h2 : defective_pens = 3)
  (h3 : defective_pens < total_pens) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l2308_230801


namespace NUMINAMATH_CALUDE_no_extreme_points_implies_a_leq_two_l2308_230800

/-- Given a function f(x) = x - 1/x - a*ln(x), if f has no extreme value points for x > 0,
    then a ≤ 2 --/
theorem no_extreme_points_implies_a_leq_two (a : ℝ) :
  (∀ x > 0, ∃ y > 0, (x - 1/x - a * Real.log x) < (y - 1/y - a * Real.log y) ∨
                     (x - 1/x - a * Real.log x) > (y - 1/y - a * Real.log y)) →
  a ≤ 2 := by
sorry


end NUMINAMATH_CALUDE_no_extreme_points_implies_a_leq_two_l2308_230800


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l2308_230899

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- State the theorem
theorem perpendicular_transitivity 
  (h1 : parallel α β) 
  (h2 : parallel β γ) 
  (h3 : perpendicular m α) : 
  perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l2308_230899


namespace NUMINAMATH_CALUDE_another_number_with_remainder_three_l2308_230869

theorem another_number_with_remainder_three (n : ℕ) : 
  n = 1680 → 
  n % 9 = 0 → 
  ∃ m : ℕ, m ≠ n ∧ n % m = 3 → 
  n % 1677 = 3 :=
by sorry

end NUMINAMATH_CALUDE_another_number_with_remainder_three_l2308_230869


namespace NUMINAMATH_CALUDE_properties_of_one_minus_sqrt_two_l2308_230868

theorem properties_of_one_minus_sqrt_two :
  let x : ℝ := 1 - Real.sqrt 2
  (- x = Real.sqrt 2 - 1) ∧
  (|x| = Real.sqrt 2 - 1) ∧
  (x⁻¹ = -1 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_properties_of_one_minus_sqrt_two_l2308_230868


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2308_230859

theorem fraction_to_decimal : (7 : ℚ) / 50 = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2308_230859


namespace NUMINAMATH_CALUDE_inequality_proof_l2308_230861

theorem inequality_proof (a b c d : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (h_sum : a*b + b*c + c*d + d*a = 1) : 
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + 
  (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2308_230861


namespace NUMINAMATH_CALUDE_peach_ratio_l2308_230824

/-- Proves the ratio of peaches in knapsack to one cloth bag is 1:2 --/
theorem peach_ratio (total_peaches : ℕ) (knapsack_peaches : ℕ) (num_cloth_bags : ℕ) :
  total_peaches = 5 * 12 →
  knapsack_peaches = 12 →
  num_cloth_bags = 2 →
  (total_peaches - knapsack_peaches) % num_cloth_bags = 0 →
  (knapsack_peaches : ℚ) / ((total_peaches - knapsack_peaches) / num_cloth_bags) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_peach_ratio_l2308_230824


namespace NUMINAMATH_CALUDE_probability_of_sum_25_l2308_230810

/-- Represents a die with numbered and blank faces -/
structure Die where
  faces : ℕ
  numbered_faces : ℕ
  min_number : ℕ
  max_number : ℕ

/-- The first die with 18 numbered faces (1-18) and 2 blank faces -/
def die1 : Die :=
  { faces := 20
  , numbered_faces := 18
  , min_number := 1
  , max_number := 18 }

/-- The second die with 19 numbered faces (2-20) and 1 blank face -/
def die2 : Die :=
  { faces := 20
  , numbered_faces := 19
  , min_number := 2
  , max_number := 20 }

/-- Calculates the number of ways to roll a specific sum with two dice -/
def waysToRollSum (d1 d2 : Die) (sum : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of possible outcomes when rolling two dice -/
def totalOutcomes (d1 d2 : Die) : ℕ :=
  d1.faces * d2.faces

/-- The main theorem stating the probability of rolling a sum of 25 -/
theorem probability_of_sum_25 :
  (waysToRollSum die1 die2 25 : ℚ) / (totalOutcomes die1 die2 : ℚ) = 7 / 200 :=
sorry

end NUMINAMATH_CALUDE_probability_of_sum_25_l2308_230810


namespace NUMINAMATH_CALUDE_least_perimeter_of_triangle_l2308_230884

theorem least_perimeter_of_triangle (a b x : ℕ) : 
  a = 33 → b = 42 → x > 0 → 
  x + a > b → x + b > a → a + b > x →
  ∀ y : ℕ, y > 0 → y + a > b → y + b > a → a + b > y → x ≤ y →
  a + b + x = 85 := by
sorry

end NUMINAMATH_CALUDE_least_perimeter_of_triangle_l2308_230884


namespace NUMINAMATH_CALUDE_road_with_ten_trees_length_l2308_230827

/-- The length of a road with trees planted at equal intervals -/
def road_length (num_trees : ℕ) (interval : ℝ) : ℝ :=
  (num_trees - 1 : ℝ) * interval

/-- Theorem: The length of a road with 10 trees planted at 10-meter intervals is 90 meters -/
theorem road_with_ten_trees_length :
  road_length 10 10 = 90 := by
  sorry

#eval road_length 10 10

end NUMINAMATH_CALUDE_road_with_ten_trees_length_l2308_230827


namespace NUMINAMATH_CALUDE_bracket_removal_equality_l2308_230828

theorem bracket_removal_equality (a b c : ℝ) : a - 2*(b - c) = a - 2*b + 2*c := by
  sorry

end NUMINAMATH_CALUDE_bracket_removal_equality_l2308_230828


namespace NUMINAMATH_CALUDE_lunch_slices_count_l2308_230852

/-- The number of slices of pie served during lunch today -/
def lunch_slices : ℕ := sorry

/-- The total number of slices of pie served today -/
def total_slices : ℕ := 12

/-- The number of slices of pie served during dinner today -/
def dinner_slices : ℕ := 5

/-- Theorem stating that the number of slices served during lunch today is 7 -/
theorem lunch_slices_count : lunch_slices = total_slices - dinner_slices := by sorry

end NUMINAMATH_CALUDE_lunch_slices_count_l2308_230852


namespace NUMINAMATH_CALUDE_inequality_proof_l2308_230814

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (hd : 0 < d ∧ d < 1) : 
  1 + a * b + b * c + c * d + d * a + a * c + b * d > a + b + c + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2308_230814


namespace NUMINAMATH_CALUDE_equation_solutions_l2308_230850

-- Define the equations as functions
def eqnA (x : ℝ) := (3*x + 1)^2 = 0
def eqnB (x : ℝ) := |2*x + 1| - 6 = 0
def eqnC (x : ℝ) := Real.sqrt (5 - x) + 3 = 0
def eqnD (x : ℝ) := Real.sqrt (4*x + 9) - 7 = 0
def eqnE (x : ℝ) := |5*x - 3| + 2 = -1

-- Define the existence of solutions
def has_solution (f : ℝ → Prop) := ∃ x, f x

-- Theorem statement
theorem equation_solutions :
  (has_solution eqnA) ∧
  (has_solution eqnB) ∧
  (¬ has_solution eqnC) ∧
  (has_solution eqnD) ∧
  (¬ has_solution eqnE) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2308_230850


namespace NUMINAMATH_CALUDE_managers_salary_l2308_230887

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (salary_increase : ℚ) :
  num_employees = 20 →
  avg_salary = 1300 →
  salary_increase = 100 →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase) / (num_employees + 1) - avg_salary = salary_increase →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase) - (num_employees * avg_salary) = 3400 :=
by
  sorry

end NUMINAMATH_CALUDE_managers_salary_l2308_230887


namespace NUMINAMATH_CALUDE_profit_7500_at_65_max_profit_at_70_max_profit_is_8000_l2308_230888

/-- Represents the online store's pricing and sales model -/
structure Store where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ

/-- Calculates the number of items sold based on the current price -/
def items_sold (s : Store) (price : ℝ) : ℝ :=
  s.initial_sales + s.price_sensitivity * (s.initial_price - price)

/-- Calculates the weekly profit based on the current price -/
def weekly_profit (s : Store) (price : ℝ) : ℝ :=
  (price - s.cost_price) * (items_sold s price)

/-- The store's pricing and sales model -/
def children_clothing_store : Store :=
  { cost_price := 50
  , initial_price := 80
  , initial_sales := 200
  , price_sensitivity := 20 }

/-- Theorem: The selling price of 65 yuan achieves a weekly profit of 7500 yuan while maximizing customer benefits -/
theorem profit_7500_at_65 :
  weekly_profit children_clothing_store 65 = 7500 ∧
  ∀ p, p < 65 → weekly_profit children_clothing_store p < 7500 :=
sorry

/-- Theorem: The selling price of 70 yuan maximizes the weekly profit -/
theorem max_profit_at_70 :
  ∀ p, weekly_profit children_clothing_store p ≤ weekly_profit children_clothing_store 70 :=
sorry

/-- Theorem: The maximum weekly profit is 8000 yuan -/
theorem max_profit_is_8000 :
  weekly_profit children_clothing_store 70 = 8000 :=
sorry

end NUMINAMATH_CALUDE_profit_7500_at_65_max_profit_at_70_max_profit_is_8000_l2308_230888


namespace NUMINAMATH_CALUDE_product_of_good_sequences_is_good_l2308_230886

/-- A sequence is a function from natural numbers to real numbers. -/
def Sequence := ℕ → ℝ

/-- The first derivative of a sequence. -/
def firstDerivative (a : Sequence) : Sequence :=
  λ n => a (n + 1) - a n

/-- The k-th derivative of a sequence. -/
def kthDerivative : ℕ → Sequence → Sequence
  | 0, a => a
  | k + 1, a => firstDerivative (kthDerivative k a)

/-- A sequence is good if it and all its derivatives consist of positive numbers. -/
def isGoodSequence (a : Sequence) : Prop :=
  ∀ k n, kthDerivative k a n > 0

/-- The element-wise product of two sequences. -/
def productSequence (a b : Sequence) : Sequence :=
  λ n => a n * b n

/-- Theorem: The element-wise product of two good sequences is also a good sequence. -/
theorem product_of_good_sequences_is_good (a b : Sequence) 
  (ha : isGoodSequence a) (hb : isGoodSequence b) : 
  isGoodSequence (productSequence a b) := by
  sorry

end NUMINAMATH_CALUDE_product_of_good_sequences_is_good_l2308_230886


namespace NUMINAMATH_CALUDE_max_log_sum_l2308_230843

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = 40) :
  ∃ (max : ℝ), max = 2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + 4*b = 40 → Real.log a + Real.log b ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_log_sum_l2308_230843


namespace NUMINAMATH_CALUDE_fixed_costs_calculation_l2308_230862

/-- The fixed monthly costs for a computer manufacturer producing electronic components -/
def fixed_monthly_costs : ℝ := 16699.50

/-- The production cost per component -/
def production_cost : ℝ := 80

/-- The shipping cost per component -/
def shipping_cost : ℝ := 7

/-- The number of components produced and sold per month -/
def monthly_units : ℕ := 150

/-- The lowest selling price per component for break-even -/
def selling_price : ℝ := 198.33

theorem fixed_costs_calculation :
  fixed_monthly_costs = 
    selling_price * monthly_units - 
    (production_cost + shipping_cost) * monthly_units :=
by sorry

end NUMINAMATH_CALUDE_fixed_costs_calculation_l2308_230862


namespace NUMINAMATH_CALUDE_john_painting_area_l2308_230860

/-- The area John needs to paint on a wall -/
def areaToPaint (wallHeight wallLength paintingWidth paintingHeight : ℝ) : ℝ :=
  wallHeight * wallLength - paintingWidth * paintingHeight

/-- Theorem: John needs to paint 135 square feet -/
theorem john_painting_area :
  areaToPaint 10 15 3 5 = 135 := by
sorry

end NUMINAMATH_CALUDE_john_painting_area_l2308_230860


namespace NUMINAMATH_CALUDE_age_difference_l2308_230813

theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 27 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 29 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2308_230813


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l2308_230890

/-- Given a pentagon STARS where four of its angles are congruent and two of these are equal, 
    prove that the measure of one of these angles is 108°. -/
theorem pentagon_angle_measure (S T A R : ℝ) : 
  (S + T + A + R + S = 540) → -- Sum of angles in a pentagon
  (S = T) → (T = A) → (A = R) → -- Four angles are congruent
  (A = S) → -- Two of these angles are equal
  R = 108 := by sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l2308_230890


namespace NUMINAMATH_CALUDE_no_perfect_squares_in_all_ones_sequence_l2308_230832

/-- Represents a number in the sequence 11, 111, 1111, ... -/
def allOnesNumber (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Predicate to check if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

/-- Theorem: There are no perfect squares in the sequence of numbers
    consisting of only the digit 1, starting from 11 -/
theorem no_perfect_squares_in_all_ones_sequence :
  ∀ n : ℕ, n ≥ 2 → ¬ isPerfectSquare (allOnesNumber n) :=
sorry

end NUMINAMATH_CALUDE_no_perfect_squares_in_all_ones_sequence_l2308_230832


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_simplification_and_evaluation_l2308_230848

-- Question 1
theorem factorization_1 (m n : ℝ) : 
  m^2 * (n - 3) + 4 * (3 - n) = (n - 3) * (m + 2) * (m - 2) := by sorry

-- Question 2
theorem factorization_2 (p : ℝ) :
  (p - 3) * (p - 1) + 1 = (p - 2)^2 := by sorry

-- Question 3
theorem simplification_and_evaluation (x : ℝ) 
  (h : x^2 + x + 1/4 = 0) :
  ((2*x + 1) / (x + 1) + x - 1) / ((x + 2) / (x^2 + 2*x + 1)) = -1/4 := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_simplification_and_evaluation_l2308_230848


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l2308_230878

/-- Define the @ operation for real numbers -/
def at_op (p q : ℝ) : ℝ := p + q - p * q

/-- The theorem statement -/
theorem inequality_system_solution_range (m : ℝ) :
  (∃! (a b : ℤ), (a ≠ b) ∧ 
    (at_op 2 (a : ℝ) > 0) ∧ (at_op (a : ℝ) 3 ≤ m) ∧
    (at_op 2 (b : ℝ) > 0) ∧ (at_op (b : ℝ) 3 ≤ m) ∧
    (∀ x : ℤ, x ≠ a ∧ x ≠ b → 
      ¬((at_op 2 (x : ℝ) > 0) ∧ (at_op (x : ℝ) 3 ≤ m))))
  → 3 ≤ m ∧ m < 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l2308_230878


namespace NUMINAMATH_CALUDE_power_calculation_l2308_230825

theorem power_calculation (m n : ℕ) (h1 : 2^m = 3) (h2 : 4^n = 8) :
  2^(3*m - 2*n + 3) = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2308_230825


namespace NUMINAMATH_CALUDE_carol_picked_29_carrots_l2308_230819

/-- The number of carrots Carol picked -/
def carols_carrots (total_carrots good_carrots bad_carrots moms_carrots : ℕ) : ℕ :=
  total_carrots - moms_carrots

/-- Theorem stating that Carol picked 29 carrots -/
theorem carol_picked_29_carrots 
  (total_carrots : ℕ) 
  (good_carrots : ℕ) 
  (bad_carrots : ℕ) 
  (moms_carrots : ℕ) 
  (h1 : total_carrots = good_carrots + bad_carrots)
  (h2 : good_carrots = 38)
  (h3 : bad_carrots = 7)
  (h4 : moms_carrots = 16) :
  carols_carrots total_carrots good_carrots bad_carrots moms_carrots = 29 := by
  sorry

end NUMINAMATH_CALUDE_carol_picked_29_carrots_l2308_230819


namespace NUMINAMATH_CALUDE_cakes_distribution_l2308_230849

theorem cakes_distribution (total_cakes : ℕ) (friends : ℕ) (cakes_per_friend : ℕ) :
  total_cakes = 30 →
  friends = 2 →
  cakes_per_friend = total_cakes / friends →
  cakes_per_friend = 15 := by
sorry

end NUMINAMATH_CALUDE_cakes_distribution_l2308_230849


namespace NUMINAMATH_CALUDE_doll_factory_operation_time_l2308_230807

/-- Calculate the total machine operation time for dolls and accessories -/
theorem doll_factory_operation_time :
  let num_dolls : ℕ := 12000
  let shoes_per_doll : ℕ := 2
  let bags_per_doll : ℕ := 3
  let cosmetics_per_doll : ℕ := 1
  let hats_per_doll : ℕ := 5
  let doll_production_time : ℕ := 45
  let accessory_production_time : ℕ := 10

  let total_accessories : ℕ := num_dolls * (shoes_per_doll + bags_per_doll + cosmetics_per_doll + hats_per_doll)
  let doll_time : ℕ := num_dolls * doll_production_time
  let accessory_time : ℕ := total_accessories * accessory_production_time
  let total_time : ℕ := doll_time + accessory_time

  total_time = 1860000 := by
  sorry

end NUMINAMATH_CALUDE_doll_factory_operation_time_l2308_230807


namespace NUMINAMATH_CALUDE_odd_divisibility_l2308_230894

def sum_of_powers (n : ℕ) : ℕ := (Finset.range (n - 1)).sum (λ k => k^n)

theorem odd_divisibility (n : ℕ) (h : n > 1) :
  n ∣ sum_of_powers n ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_odd_divisibility_l2308_230894


namespace NUMINAMATH_CALUDE_inverse_function_property_l2308_230840

def invertible_function (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

theorem inverse_function_property
  (f : ℝ → ℝ)
  (h_inv : invertible_function f)
  (h_point : 1 - f 1 = 2) :
  ∃ g : ℝ → ℝ, invertible_function g ∧ g = f⁻¹ ∧ g (-1) - (-1) = 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_property_l2308_230840


namespace NUMINAMATH_CALUDE_find_multiplier_l2308_230898

theorem find_multiplier : ∃ (m : ℕ), 
  220050 = m * (555 - 445) * (555 + 445) + 50 ∧ 
  m * (555 - 445) = 220050 / (555 + 445) :=
by sorry

end NUMINAMATH_CALUDE_find_multiplier_l2308_230898


namespace NUMINAMATH_CALUDE_sequence_properties_l2308_230851

def sequence_a (n : ℕ) : ℝ := 2^n - 1

def S (n : ℕ) : ℝ := 2 * sequence_a n - n

theorem sequence_properties :
  (∀ n : ℕ, S n = 2 * sequence_a n - n) →
  (∀ n : ℕ, sequence_a (n + 1) + 1 = 2 * (sequence_a n + 1)) ∧
  (∀ n : ℕ, sequence_a n = 2^n - 1) ∧
  (∀ k : ℕ, 2 * sequence_a (k + 1) ≠ sequence_a k + sequence_a (k + 2)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2308_230851


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2308_230880

theorem algebraic_expression_value (a b : ℝ) (h1 : a * b = 2) (h2 : a + b = 3) :
  2 * a^3 * b - 4 * a^2 * b^2 + 2 * a * b^3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2308_230880


namespace NUMINAMATH_CALUDE_triangle_area_13_13_24_l2308_230885

/-- The area of a triangle with side lengths 13, 13, and 24 is 60 square units. -/
theorem triangle_area_13_13_24 : ∃ (A : ℝ), 
  A = (1/2) * 24 * Real.sqrt (13^2 - 12^2) ∧ A = 60 := by sorry

end NUMINAMATH_CALUDE_triangle_area_13_13_24_l2308_230885


namespace NUMINAMATH_CALUDE_equal_price_sheets_is_12_l2308_230826

/-- The number of sheets for which two photo companies charge the same amount -/
def equal_price_sheets : ℕ :=
  let john_per_sheet : ℚ := 275 / 100
  let john_sitting_fee : ℚ := 125
  let sam_per_sheet : ℚ := 150 / 100
  let sam_sitting_fee : ℚ := 140
  ⌊(sam_sitting_fee - john_sitting_fee) / (john_per_sheet - sam_per_sheet)⌋₊

theorem equal_price_sheets_is_12 : equal_price_sheets = 12 := by
  sorry

#eval equal_price_sheets

end NUMINAMATH_CALUDE_equal_price_sheets_is_12_l2308_230826


namespace NUMINAMATH_CALUDE_system_solution_l2308_230866

theorem system_solution (x y : ℝ) (h1 : 2*x + y = 7) (h2 : x + 2*y = 10) : 
  (x + y) / 3 = 17/9 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2308_230866


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l2308_230873

theorem fraction_sum_simplification :
  8 / 19 - 5 / 57 + 1 / 3 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l2308_230873


namespace NUMINAMATH_CALUDE_perfect_square_difference_l2308_230855

theorem perfect_square_difference : ∃ (x a b : ℤ), 
  (x + 100 = a^2) ∧ 
  (x + 164 = b^2) ∧ 
  (x = 125 ∨ x = -64 ∨ x = -100) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_difference_l2308_230855


namespace NUMINAMATH_CALUDE_M_sufficient_not_necessary_for_N_l2308_230805

def M : Set ℝ := {x | x^2 < 3*x}
def N : Set ℝ := {x | |x - 1| < 2}

theorem M_sufficient_not_necessary_for_N :
  (∀ a : ℝ, a ∈ M → a ∈ N) ∧ (∃ b : ℝ, b ∈ N ∧ b ∉ M) := by sorry

end NUMINAMATH_CALUDE_M_sufficient_not_necessary_for_N_l2308_230805


namespace NUMINAMATH_CALUDE_calculate_expression_l2308_230829

/-- Proves that 8 * 9(2/5) - 3 = 72(1/5) -/
theorem calculate_expression : 8 * (9 + 2/5) - 3 = 72 + 1/5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2308_230829


namespace NUMINAMATH_CALUDE_power_sum_equals_seventeen_l2308_230847

theorem power_sum_equals_seventeen : (-3 : ℤ)^4 + (-4 : ℤ)^3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_seventeen_l2308_230847


namespace NUMINAMATH_CALUDE_fraction_product_equals_one_l2308_230844

theorem fraction_product_equals_one : 
  (36 : ℚ) / 34 * 26 / 48 * 136 / 78 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_product_equals_one_l2308_230844


namespace NUMINAMATH_CALUDE_probability_even_sum_two_wheels_l2308_230863

/-- Represents a wheel with sections labeled as even or odd numbers -/
structure Wheel where
  total_sections : ℕ
  even_sections : ℕ
  odd_sections : ℕ
  sections_sum : even_sections + odd_sections = total_sections

/-- Calculates the probability of getting an even sum when spinning two wheels -/
def probability_even_sum (wheel1 wheel2 : Wheel) : ℚ :=
  let p_even1 := wheel1.even_sections / wheel1.total_sections
  let p_odd1 := wheel1.odd_sections / wheel1.total_sections
  let p_even2 := wheel2.even_sections / wheel2.total_sections
  let p_odd2 := wheel2.odd_sections / wheel2.total_sections
  (p_even1 * p_even2) + (p_odd1 * p_odd2)

theorem probability_even_sum_two_wheels :
  let wheel1 : Wheel := ⟨3, 2, 1, by simp⟩
  let wheel2 : Wheel := ⟨5, 3, 2, by simp⟩
  probability_even_sum wheel1 wheel2 = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_two_wheels_l2308_230863


namespace NUMINAMATH_CALUDE_cloth_cutting_l2308_230867

theorem cloth_cutting (S : ℝ) : 
  S / 2 + S / 4 = 75 → S = 100 := by
sorry

end NUMINAMATH_CALUDE_cloth_cutting_l2308_230867


namespace NUMINAMATH_CALUDE_total_birds_l2308_230879

/-- Given 3 pairs of birds, prove that the total number of birds is 6. -/
theorem total_birds (pairs : ℕ) (h : pairs = 3) : pairs * 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_l2308_230879


namespace NUMINAMATH_CALUDE_multiply_subtract_equal_compute_expression_l2308_230842

theorem multiply_subtract_equal (a b c : ℤ) : a * c - b * c = (a - b) * c := by sorry

theorem compute_expression : 45 * 1313 - 10 * 1313 = 45955 := by sorry

end NUMINAMATH_CALUDE_multiply_subtract_equal_compute_expression_l2308_230842


namespace NUMINAMATH_CALUDE_tina_win_probability_l2308_230812

theorem tina_win_probability (p_lose : ℚ) (h_lose : p_lose = 3/7) (h_no_tie : True) :
  1 - p_lose = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_tina_win_probability_l2308_230812


namespace NUMINAMATH_CALUDE_emily_salary_adjustment_l2308_230839

/-- Calculates Emily's new salary after adjusting employee salaries -/
def emilysNewSalary (initialSalary: ℕ) (numEmployees: ℕ) (initialEmployeeSalary targetEmployeeSalary: ℕ) : ℕ :=
  initialSalary - numEmployees * (targetEmployeeSalary - initialEmployeeSalary)

/-- Proves that Emily's new salary is $850,000 given the initial conditions -/
theorem emily_salary_adjustment :
  emilysNewSalary 1000000 10 20000 35000 = 850000 := by
  sorry

end NUMINAMATH_CALUDE_emily_salary_adjustment_l2308_230839


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2308_230872

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2308_230872


namespace NUMINAMATH_CALUDE_three_digit_primes_ending_in_one_l2308_230881

theorem three_digit_primes_ending_in_one (p : ℕ) : 
  (200 < p ∧ p < 1000 ∧ p % 10 = 1 ∧ Nat.Prime p) → 
  (Finset.filter (λ x => 200 < x ∧ x < 1000 ∧ x % 10 = 1 ∧ Nat.Prime x) (Finset.range 1000)).card = 23 :=
sorry

end NUMINAMATH_CALUDE_three_digit_primes_ending_in_one_l2308_230881


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2308_230817

theorem shaded_area_calculation (R : ℝ) (d : ℝ) (h1 : R = 10) (h2 : d = 8) : 
  let r : ℝ := Real.sqrt (R^2 - d^2)
  let large_circle_area : ℝ := π * R^2
  let small_circle_area : ℝ := 2 * π * r^2
  large_circle_area - small_circle_area = 28 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2308_230817


namespace NUMINAMATH_CALUDE_samson_utility_l2308_230892

/-- Represents the utility function for Samson's activities -/
def utility (math : ℝ) (frisbee : ℝ) : ℝ := math * frisbee

/-- Represents the conditions of the problem -/
theorem samson_utility (t : ℝ) : 
  utility (8 - t) t = utility (t + 3) (2 - t) → t = 2/3 := by
  sorry

#check samson_utility

end NUMINAMATH_CALUDE_samson_utility_l2308_230892


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2308_230836

theorem system_of_equations_solution :
  ∃ (x y : ℝ), x + 2*y = 3 ∧ x - 4*y = 9 → x = 5 ∧ y = -1 := by
  sorry

#check system_of_equations_solution

end NUMINAMATH_CALUDE_system_of_equations_solution_l2308_230836


namespace NUMINAMATH_CALUDE_aaron_cards_total_l2308_230835

/-- Given that Aaron initially has 5 cards and finds 62 more, 
    prove that he ends up with 67 cards in total. -/
theorem aaron_cards_total (initial_cards : ℕ) (found_cards : ℕ) : 
  initial_cards = 5 → found_cards = 62 → initial_cards + found_cards = 67 := by
  sorry

end NUMINAMATH_CALUDE_aaron_cards_total_l2308_230835


namespace NUMINAMATH_CALUDE_hundredth_group_sum_divided_by_100_l2308_230808

/-- The sum of the first n natural numbers -/
def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The last number in the nth group -/
def last_number (n : ℕ) : ℕ := 2 * sum_of_naturals n

/-- The first number in the nth group -/
def first_number (n : ℕ) : ℕ := last_number (n - 1) - 2 * (n - 1)

/-- The sum of numbers in the nth group -/
def group_sum (n : ℕ) : ℕ := n * (first_number n + last_number n) / 2

theorem hundredth_group_sum_divided_by_100 :
  group_sum 100 / 100 = 10001 := by sorry

end NUMINAMATH_CALUDE_hundredth_group_sum_divided_by_100_l2308_230808


namespace NUMINAMATH_CALUDE_smallest_number_divisible_after_increase_l2308_230831

theorem smallest_number_divisible_after_increase : ∃ (k : ℕ), 
  (∀ (n : ℕ), n < 3153 → ¬∃ (m : ℕ), (n + m) % 18 = 0 ∧ (n + m) % 70 = 0 ∧ (n + m) % 25 = 0 ∧ (n + m) % 21 = 0) ∧
  (∃ (m : ℕ), (3153 + m) % 18 = 0 ∧ (3153 + m) % 70 = 0 ∧ (3153 + m) % 25 = 0 ∧ (3153 + m) % 21 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_after_increase_l2308_230831


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_square_l2308_230877

def x : ℕ := 11 * 36 * 54

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_y_for_perfect_square : 
  (∃ y : ℕ, y > 0 ∧ is_perfect_square (x * y)) ∧ 
  (∀ z : ℕ, z > 0 ∧ z < 66 → ¬is_perfect_square (x * z)) ∧
  is_perfect_square (x * 66) :=
sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_square_l2308_230877

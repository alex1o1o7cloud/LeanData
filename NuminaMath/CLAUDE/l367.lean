import Mathlib

namespace NUMINAMATH_CALUDE_root_product_rational_l367_36750

-- Define the polynomial f(z)
def f (a b c d e : ℤ) (z : ℂ) : ℂ := a * z^4 + b * z^3 + c * z^2 + d * z + e

-- Define the roots r1, r2, r3, r4
variable (r1 r2 r3 r4 : ℂ)

-- State the theorem
theorem root_product_rational
  (a b c d e : ℤ)
  (h_a_nonzero : a ≠ 0)
  (h_f_factored : ∀ z, f a b c d e z = a * (z - r1) * (z - r2) * (z - r3) * (z - r4))
  (h_sum_rational : ∃ q : ℚ, (r1 + r2 : ℂ) = q)
  (h_sum_distinct : r1 + r2 ≠ r3 + r4) :
  ∃ q : ℚ, (r1 * r2 : ℂ) = q :=
sorry

end NUMINAMATH_CALUDE_root_product_rational_l367_36750


namespace NUMINAMATH_CALUDE_sum_of_four_repeated_digit_terms_l367_36700

/-- A function that checks if a natural number consists of repeated digits --/
def is_repeated_digit (n : ℕ) : Prop := sorry

/-- A function that returns the number of digits in a natural number --/
def num_digits (n : ℕ) : ℕ := sorry

theorem sum_of_four_repeated_digit_terms : 
  ∃ (a b c d : ℕ), 
    2017 = a + b + c + d ∧ 
    is_repeated_digit a ∧ 
    is_repeated_digit b ∧ 
    is_repeated_digit c ∧ 
    is_repeated_digit d ∧ 
    num_digits a ≠ num_digits b ∧ 
    num_digits a ≠ num_digits c ∧ 
    num_digits a ≠ num_digits d ∧ 
    num_digits b ≠ num_digits c ∧ 
    num_digits b ≠ num_digits d ∧ 
    num_digits c ≠ num_digits d :=
by sorry

end NUMINAMATH_CALUDE_sum_of_four_repeated_digit_terms_l367_36700


namespace NUMINAMATH_CALUDE_equal_roots_implies_k_eq_four_l367_36701

/-- 
A quadratic equation ax^2 + bx + c = 0 has two equal real roots if and only if 
its discriminant b^2 - 4ac is equal to 0.
-/
def has_two_equal_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- 
Given a quadratic equation kx^2 - 2kx + 4 = 0 with two equal real roots,
prove that k = 4.
-/
theorem equal_roots_implies_k_eq_four :
  ∀ k : ℝ, k ≠ 0 → has_two_equal_real_roots k (-2*k) 4 → k = 4 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_implies_k_eq_four_l367_36701


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l367_36784

theorem triangle_angle_proof (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → -- Triangle condition
  a * Real.cos B = 3 * b * Real.cos A → -- Given equation
  B = A - π / 6 → -- Given relation between A and B
  B = π / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l367_36784


namespace NUMINAMATH_CALUDE_fox_can_catch_mole_l367_36787

/-- Represents a mound in the line of 100 mounds. -/
def Mound := Fin 100

/-- Represents the state of the game at any given time. -/
structure GameState where
  molePosition : Mound
  foxPosition : Mound

/-- Represents a strategy for the fox. -/
def FoxStrategy := GameState → Mound

/-- Represents the result of a single move in the game. -/
inductive MoveResult
  | Caught
  | Continue (newState : GameState)

/-- Simulates a single move in the game. -/
def makeMove (state : GameState) (strategy : FoxStrategy) : MoveResult :=
  sorry

/-- Simulates the game for a given number of moves. -/
def playGame (initialState : GameState) (strategy : FoxStrategy) (moves : Nat) : Bool :=
  sorry

/-- The main theorem stating that there exists a strategy for the fox to catch the mole. -/
theorem fox_can_catch_mole :
  ∃ (strategy : FoxStrategy), ∀ (initialState : GameState),
    playGame initialState strategy 200 = true :=
  sorry

end NUMINAMATH_CALUDE_fox_can_catch_mole_l367_36787


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l367_36753

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is the line x - 2y = 0,
    then its eccentricity is √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ x - 2*y = 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l367_36753


namespace NUMINAMATH_CALUDE_equation_system_solution_l367_36779

theorem equation_system_solution (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l367_36779


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l367_36706

theorem least_positive_integer_with_remainders : ∃ M : ℕ, 
  (M > 0) ∧
  (M % 6 = 5) ∧
  (M % 7 = 6) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (M % 10 = 9) ∧
  (M % 11 = 10) ∧
  (∀ n : ℕ, n > 0 ∧ 
    n % 6 = 5 ∧
    n % 7 = 6 ∧
    n % 8 = 7 ∧
    n % 9 = 8 ∧
    n % 10 = 9 ∧
    n % 11 = 10 → n ≥ M) ∧
  M = 27719 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l367_36706


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l367_36722

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : x > -2)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + 1 < 9) :
  ∃ (upper_bound : ℤ), 
    (∀ y : ℤ, (3 < y ∧ y < 10) → 
               (5 < y ∧ y < 18) → 
               (y > -2) → 
               (0 < y ∧ y < 8) → 
               (y + 1 < 9) → 
               y ≤ upper_bound) ∧
    (upper_bound = 8) :=
sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l367_36722


namespace NUMINAMATH_CALUDE_share_of_A_l367_36797

theorem share_of_A (total : ℝ) (a b c : ℝ) : 
  total = 116000 →
  a + b + c = total →
  a / b = 3 / 4 →
  b / c = 5 / 6 →
  a = 116000 * 15 / 59 :=
by sorry

end NUMINAMATH_CALUDE_share_of_A_l367_36797


namespace NUMINAMATH_CALUDE_prob_same_heads_is_five_thirty_seconds_l367_36705

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The probability of getting heads on a single penny toss -/
def prob_heads : ℚ := 1/2

/-- The probability that Ephraim gets the same number of heads as Keiko -/
def prob_same_heads : ℚ := 5/32

/-- Theorem stating that the probability of Ephraim getting the same number of heads as Keiko is 5/32 -/
theorem prob_same_heads_is_five_thirty_seconds :
  prob_same_heads = 5/32 := by sorry

end NUMINAMATH_CALUDE_prob_same_heads_is_five_thirty_seconds_l367_36705


namespace NUMINAMATH_CALUDE_unique_positive_solution_arctan_equation_l367_36791

theorem unique_positive_solution_arctan_equation :
  ∃! y : ℝ, y > 0 ∧ Real.arctan (1 / y) + Real.arctan (1 / y^2) = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_arctan_equation_l367_36791


namespace NUMINAMATH_CALUDE_triangle_obtuse_iff_tangent_product_less_than_one_l367_36741

theorem triangle_obtuse_iff_tangent_product_less_than_one 
  (α β γ : Real) (h_sum : α + β + γ = Real.pi) (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) :
  γ > Real.pi / 2 ↔ Real.tan α * Real.tan β < 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_obtuse_iff_tangent_product_less_than_one_l367_36741


namespace NUMINAMATH_CALUDE_prob_white_both_urns_l367_36745

/-- Represents an urn with a certain number of black and white balls -/
structure Urn :=
  (black : ℕ)
  (white : ℕ)

/-- Calculates the probability of drawing a white ball from an urn -/
def prob_white (u : Urn) : ℚ :=
  u.white / (u.black + u.white)

/-- The probability of drawing white balls from both urns is 7/30 -/
theorem prob_white_both_urns (urn1 urn2 : Urn)
  (h1 : urn1 = Urn.mk 6 4)
  (h2 : urn2 = Urn.mk 5 7) :
  prob_white urn1 * prob_white urn2 = 7 / 30 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_both_urns_l367_36745


namespace NUMINAMATH_CALUDE_jensen_family_mileage_l367_36724

/-- Represents the mileage problem for the Jensen family's road trip -/
theorem jensen_family_mileage
  (total_highway_miles : ℝ)
  (total_city_miles : ℝ)
  (highway_mpg : ℝ)
  (total_gallons : ℝ)
  (h1 : total_highway_miles = 210)
  (h2 : total_city_miles = 54)
  (h3 : highway_mpg = 35)
  (h4 : total_gallons = 9) :
  (total_city_miles / (total_gallons - total_highway_miles / highway_mpg)) = 18 :=
by sorry

end NUMINAMATH_CALUDE_jensen_family_mileage_l367_36724


namespace NUMINAMATH_CALUDE_highest_frequency_count_l367_36702

theorem highest_frequency_count (total_sample : ℕ) (num_groups : ℕ) 
  (cumulative_freq_seven : ℚ) (a : ℕ) (r : ℕ) : 
  total_sample = 100 →
  num_groups = 10 →
  cumulative_freq_seven = 79/100 →
  r > 1 →
  a + a * r + a * r^2 = total_sample - (cumulative_freq_seven * total_sample).num →
  (∃ (max_freq : ℕ), max_freq = max a (max (a * r) (a * r^2)) ∧ max_freq = 12) :=
sorry

end NUMINAMATH_CALUDE_highest_frequency_count_l367_36702


namespace NUMINAMATH_CALUDE_f_at_three_equals_five_l367_36738

/-- A quadratic function f(x) = ax^2 + bx + 2 satisfying f(1) = 4 and f(2) = 5 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

/-- Theorem: Given f(x) = ax^2 + bx + 2 with f(1) = 4 and f(2) = 5, prove that f(3) = 5 -/
theorem f_at_three_equals_five (a b : ℝ) (h1 : f a b 1 = 4) (h2 : f a b 2 = 5) :
  f a b 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_at_three_equals_five_l367_36738


namespace NUMINAMATH_CALUDE_largest_subset_sine_inequality_l367_36789

theorem largest_subset_sine_inequality :
  ∀ y ∈ Set.Icc 0 Real.pi, ∀ x ∈ Set.Icc 0 Real.pi,
  Real.sin (x + y) ≤ Real.sin x + Real.sin y :=
by sorry

end NUMINAMATH_CALUDE_largest_subset_sine_inequality_l367_36789


namespace NUMINAMATH_CALUDE_hyperbola_conditions_exclusive_or_conditions_l367_36767

-- Define proposition p
def p (k : ℝ) : Prop := k^2 - 8*k - 20 ≤ 0

-- Define proposition q
def q (k : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 4 - k > 0 ∧ 1 - k < 0 ∧
  ∀ (x y : ℝ), x^2 / (4-k) + y^2 / (1-k) = 1 ↔ (x/a)^2 - (y/b)^2 = 1

theorem hyperbola_conditions (k : ℝ) : q k ↔ 1 < k ∧ k < 4 := by sorry

theorem exclusive_or_conditions (k : ℝ) : (p k ∨ q k) ∧ ¬(p k ∧ q k) ↔ 
  (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10) := by sorry

end NUMINAMATH_CALUDE_hyperbola_conditions_exclusive_or_conditions_l367_36767


namespace NUMINAMATH_CALUDE_number_line_problem_l367_36776

/-- Given a number line with equally spaced markings, prove that if the starting point is 2,
    the ending point is 34, and there are 8 equal steps between them,
    then the point z reached after 6 steps from 2 is 26. -/
theorem number_line_problem (start end_ : ℝ) (total_steps : ℕ) (steps_to_z : ℕ) :
  start = 2 →
  end_ = 34 →
  total_steps = 8 →
  steps_to_z = 6 →
  let step_length := (end_ - start) / total_steps
  start + steps_to_z * step_length = 26 := by
  sorry

end NUMINAMATH_CALUDE_number_line_problem_l367_36776


namespace NUMINAMATH_CALUDE_circle_radius_proof_l367_36792

theorem circle_radius_proof (num_pencils : ℕ) (pencil_length : ℚ) (inches_per_foot : ℕ) :
  num_pencils = 56 →
  pencil_length = 6 →
  inches_per_foot = 12 →
  (num_pencils * pencil_length / (2 * inches_per_foot) : ℚ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l367_36792


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_45_l367_36761

theorem max_consecutive_integers_sum_45 (n : ℕ) 
  (h : ∃ a : ℤ, (Finset.range n).sum (λ i => a + i) = 45) : n ≤ 90 := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_45_l367_36761


namespace NUMINAMATH_CALUDE_correct_investment_equation_l367_36764

/-- Represents the investment scenario over two years -/
def investment_scenario (initial_investment : ℝ) (total_investment : ℝ) (growth_rate : ℝ) : Prop :=
  initial_investment * (1 + growth_rate) + initial_investment * (1 + growth_rate)^2 = total_investment

/-- Theorem stating that the given equation correctly represents the investment scenario -/
theorem correct_investment_equation :
  investment_scenario 2500 6600 x = true :=
by
  sorry

end NUMINAMATH_CALUDE_correct_investment_equation_l367_36764


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l367_36729

theorem quadratic_roots_transformation (D E F : ℝ) (α β : ℝ) (h1 : D ≠ 0) :
  (D * α^2 + E * α + F = 0) →
  (D * β^2 + E * β + F = 0) →
  ∃ (p q : ℝ), (α^2 + 1)^2 + p * (α^2 + 1) + q = 0 ∧
                (β^2 + 1)^2 + p * (β^2 + 1) + q = 0 ∧
                p = (2 * D * F - E^2 - 2 * D^2) / D^2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l367_36729


namespace NUMINAMATH_CALUDE_subset_sum_equals_A_l367_36783

theorem subset_sum_equals_A (A : ℕ) (a : List ℕ) : 
  (∀ n ∈ Finset.range 9, A % (n + 1) = 0) →
  (∀ x ∈ a, x < 10) →
  (2 * A = a.sum) →
  ∃ s : List ℕ, s.toFinset ⊆ a.toFinset ∧ s.sum = A := by
  sorry

end NUMINAMATH_CALUDE_subset_sum_equals_A_l367_36783


namespace NUMINAMATH_CALUDE_second_number_is_37_l367_36714

theorem second_number_is_37 (a b c d : ℕ) : 
  a + b + c + d = 260 →
  a = 2 * b →
  c = a / 3 →
  d = 2 * (b + c) →
  b = 37 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_37_l367_36714


namespace NUMINAMATH_CALUDE_three_lines_intersection_angles_l367_36794

-- Define a structure for a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define a structure for an intersection point
structure IntersectionPoint where
  point : ℝ × ℝ

-- Define a function to calculate the angle between two lines
def angleBetweenLines (l1 l2 : Line) : ℝ := sorry

-- Theorem statement
theorem three_lines_intersection_angles 
  (l1 l2 l3 : Line) 
  (p : IntersectionPoint) 
  (h1 : l1.point1 = p.point ∨ l1.point2 = p.point)
  (h2 : l2.point1 = p.point ∨ l2.point2 = p.point)
  (h3 : l3.point1 = p.point ∨ l3.point2 = p.point) :
  angleBetweenLines l1 l2 = 120 ∧ 
  angleBetweenLines l2 l3 = 120 ∧ 
  angleBetweenLines l3 l1 = 120 := by sorry

end NUMINAMATH_CALUDE_three_lines_intersection_angles_l367_36794


namespace NUMINAMATH_CALUDE_freddy_is_18_l367_36770

def job_age : ℕ := 5

def stephanie_age (j : ℕ) : ℕ := 4 * j

def freddy_age (s : ℕ) : ℕ := s - 2

theorem freddy_is_18 : freddy_age (stephanie_age job_age) = 18 := by
  sorry

end NUMINAMATH_CALUDE_freddy_is_18_l367_36770


namespace NUMINAMATH_CALUDE_max_value_of_operation_l367_36788

theorem max_value_of_operation : ∃ (n : ℤ), 
  10 ≤ n ∧ n ≤ 99 ∧ 
  (250 - 3*n)^2 = 4 ∧
  ∀ (m : ℤ), 10 ≤ m ∧ m ≤ 99 → (250 - 3*m)^2 ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_operation_l367_36788


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l367_36748

theorem quadratic_rewrite_ratio : 
  ∃ (c p q : ℚ), 
    (∀ j, 8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q) ∧ 
    q / p = -77 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l367_36748


namespace NUMINAMATH_CALUDE_no_solution_exists_l367_36707

theorem no_solution_exists :
  ¬∃ (B C : ℕ+), 
    (Nat.lcm 360 (Nat.lcm B C) = 55440) ∧ 
    (Nat.gcd 360 (Nat.gcd B C) = 15) ∧ 
    (B * C = 2316) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l367_36707


namespace NUMINAMATH_CALUDE_carlton_outfits_l367_36768

/-- The number of outfits Carlton has -/
def number_of_outfits (button_up_shirts : ℕ) : ℕ :=
  (2 * button_up_shirts) * button_up_shirts

/-- Theorem stating that Carlton has 18 outfits -/
theorem carlton_outfits : number_of_outfits 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_carlton_outfits_l367_36768


namespace NUMINAMATH_CALUDE_flowerbed_perimeter_l367_36730

/-- The perimeter of a rectangular flowerbed with given dimensions -/
theorem flowerbed_perimeter : 
  let width : ℝ := 4
  let length : ℝ := 2 * width - 1
  2 * (length + width) = 22 := by sorry

end NUMINAMATH_CALUDE_flowerbed_perimeter_l367_36730


namespace NUMINAMATH_CALUDE_find_M_l367_36717

theorem find_M : ∃ M : ℕ, (992 + 994 + 996 + 998 + 1000 = 5000 - M) ∧ (M = 20) := by
  sorry

end NUMINAMATH_CALUDE_find_M_l367_36717


namespace NUMINAMATH_CALUDE_share_yield_calculation_l367_36721

/-- Calculates the effective interest rate (yield) for a share --/
theorem share_yield_calculation (face_value : ℝ) (dividend_rate : ℝ) (market_value : ℝ) :
  face_value = 60 ∧ dividend_rate = 0.09 ∧ market_value = 45 →
  (face_value * dividend_rate) / market_value = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_share_yield_calculation_l367_36721


namespace NUMINAMATH_CALUDE_not_cylinder_if_triangle_front_view_l367_36737

/-- A type representing geometric bodies -/
inductive GeometricBody
  | Cylinder
  | Cone
  | Tetrahedron
  | TriangularPrism

/-- A type representing possible front views -/
inductive FrontView
  | Triangle
  | Rectangle
  | Circle

/-- A function that returns the front view of a geometric body -/
def frontView (body : GeometricBody) : FrontView :=
  match body with
  | GeometricBody.Cylinder => FrontView.Rectangle
  | GeometricBody.Cone => FrontView.Triangle
  | GeometricBody.Tetrahedron => FrontView.Triangle
  | GeometricBody.TriangularPrism => FrontView.Triangle

/-- Theorem: If a geometric body has a triangle as its front view, it cannot be a cylinder -/
theorem not_cylinder_if_triangle_front_view (body : GeometricBody) :
  frontView body = FrontView.Triangle → body ≠ GeometricBody.Cylinder :=
by
  sorry

end NUMINAMATH_CALUDE_not_cylinder_if_triangle_front_view_l367_36737


namespace NUMINAMATH_CALUDE_parabola_properties_l367_36781

/-- A parabola passing through (-1, 0) and (m, 0) opening downwards -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  h_a_neg : a < 0
  h_m_bounds : 1 < m ∧ m < 2
  h_pass_through : a * (-1)^2 + b * (-1) + c = 0 ∧ a * m^2 + b * m + c = 0

/-- The properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  (p.b > 0) ∧ 
  (∀ x₁ x₂ y₁ y₂ : ℝ, 
    (p.a * x₁^2 + p.b * x₁ + p.c = y₁) → 
    (p.a * x₂^2 + p.b * x₂ + p.c = y₂) → 
    x₁ < x₂ → 
    x₁ + x₂ > 1 → 
    y₁ > y₂) ∧
  (p.a ≤ -1 → 
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    p.a * x₁^2 + p.b * x₁ + p.c = 1 ∧ 
    p.a * x₂^2 + p.b * x₂ + p.c = 1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l367_36781


namespace NUMINAMATH_CALUDE_not_integer_fraction_l367_36758

theorem not_integer_fraction (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  ¬ ∃ (n : ℤ), (a^2 + b^2) / (a^2 - b^2) = n := by
  sorry

end NUMINAMATH_CALUDE_not_integer_fraction_l367_36758


namespace NUMINAMATH_CALUDE_perfect_square_factors_count_l367_36790

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def count_perfect_square_factors (a b c : ℕ) : ℕ :=
  (a + 1) * (b + 1) * (c + 1)

theorem perfect_square_factors_count :
  count_perfect_square_factors 6 7 9 = 560 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_count_l367_36790


namespace NUMINAMATH_CALUDE_colorings_count_l367_36755

/-- The number of ways to color the edges of an m × n rectangle with three colors,
    such that each unit square has two sides of one color and two sides of another color. -/
def colorings (m n : ℕ) : ℕ :=
  18 * 2^(m*n - 1) * 3^(m + n - 2)

/-- Theorem stating that the number of valid colorings for an m × n rectangle
    with three colors is equal to 18 × 2^(mn-1) × 3^(m+n-2). -/
theorem colorings_count (m n : ℕ) :
  colorings m n = 18 * 2^(m*n - 1) * 3^(m + n - 2) :=
by sorry

end NUMINAMATH_CALUDE_colorings_count_l367_36755


namespace NUMINAMATH_CALUDE_probability_of_C_l367_36716

/-- A board game spinner with six regions -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)
  (probE : ℚ)
  (probF : ℚ)

/-- The conditions of the spinner -/
def spinnerConditions (s : Spinner) : Prop :=
  s.probA = 2/9 ∧
  s.probB = 1/6 ∧
  s.probC = s.probD ∧
  s.probC = s.probE ∧
  s.probF = 2 * s.probC ∧
  s.probA + s.probB + s.probC + s.probD + s.probE + s.probF = 1

/-- The theorem stating the probability of region C -/
theorem probability_of_C (s : Spinner) (h : spinnerConditions s) : s.probC = 11/90 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_C_l367_36716


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l367_36796

theorem subtraction_of_fractions : 1 / 210 - 17 / 35 = -101 / 210 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l367_36796


namespace NUMINAMATH_CALUDE_cubic_root_product_l367_36757

theorem cubic_root_product : ∃ (z₁ z₂ : ℂ),
  z₁^3 = -27 ∧ z₂^3 = -27 ∧ 
  (∃ (a₁ b₁ a₂ b₂ : ℝ), z₁ = a₁ + b₁ * I ∧ z₂ = a₂ + b₂ * I ∧ a₁ > 0 ∧ a₂ > 0) ∧
  z₁ * z₂ = 9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l367_36757


namespace NUMINAMATH_CALUDE_time_at_15mph_is_3_hours_l367_36739

/-- Represents the running scenario with three different speeds -/
structure RunningScenario where
  time_at_15mph : ℝ
  time_at_10mph : ℝ
  time_at_8mph : ℝ

/-- The total time of the run is 14 hours -/
def total_time (run : RunningScenario) : ℝ :=
  run.time_at_15mph + run.time_at_10mph + run.time_at_8mph

/-- The total distance covered is 164 miles -/
def total_distance (run : RunningScenario) : ℝ :=
  15 * run.time_at_15mph + 10 * run.time_at_10mph + 8 * run.time_at_8mph

/-- Theorem stating that the time spent running at 15 mph was 3 hours -/
theorem time_at_15mph_is_3_hours :
  ∃ (run : RunningScenario),
    total_time run = 14 ∧
    total_distance run = 164 ∧
    run.time_at_15mph = 3 ∧
    run.time_at_10mph ≥ 0 ∧
    run.time_at_8mph ≥ 0 :=
  sorry

end NUMINAMATH_CALUDE_time_at_15mph_is_3_hours_l367_36739


namespace NUMINAMATH_CALUDE_ceiling_sqrt_200_l367_36785

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_200_l367_36785


namespace NUMINAMATH_CALUDE_walking_problem_l367_36752

/-- The problem of two people walking towards each other on a road -/
theorem walking_problem (total_distance : ℝ) (yolanda_speed : ℝ) (bob_speed : ℝ) 
  (head_start : ℝ) :
  total_distance = 40 ∧ 
  yolanda_speed = 2 ∧ 
  bob_speed = 4 ∧ 
  head_start = 1 →
  ∃ (meeting_time : ℝ),
    meeting_time > 0 ∧
    head_start * yolanda_speed + meeting_time * yolanda_speed + meeting_time * bob_speed = total_distance ∧
    meeting_time * bob_speed = 25 + 1/3 :=
by sorry

end NUMINAMATH_CALUDE_walking_problem_l367_36752


namespace NUMINAMATH_CALUDE_pyramid_volume_theorem_l367_36708

/-- Represents a pyramid with a square base and a vertex -/
structure Pyramid where
  base_area : ℝ
  triangle_abe_area : ℝ
  triangle_cde_area : ℝ

/-- Calculate the volume of a pyramid -/
def pyramid_volume (p : Pyramid) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem pyramid_volume_theorem (p : Pyramid) 
  (h1 : p.base_area = 256)
  (h2 : p.triangle_abe_area = 120)
  (h3 : p.triangle_cde_area = 110) :
  pyramid_volume p = 1152 :=
sorry

end NUMINAMATH_CALUDE_pyramid_volume_theorem_l367_36708


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l367_36774

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 130) 
  (h2 : x * y = 36) : 
  x + y ≤ Real.sqrt 202 ∧ ∃ (a b : ℝ), a^2 + b^2 = 130 ∧ a * b = 36 ∧ a + b = Real.sqrt 202 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l367_36774


namespace NUMINAMATH_CALUDE_sqrt_64_minus_neg_2_cubed_equals_16_l367_36712

theorem sqrt_64_minus_neg_2_cubed_equals_16 : 
  Real.sqrt 64 - (-2)^3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_64_minus_neg_2_cubed_equals_16_l367_36712


namespace NUMINAMATH_CALUDE_min_value_phi_l367_36728

/-- Given real numbers a and b satisfying a^2 + b^2 - 4b + 3 = 0,
    and a function f(x) = a·sin(2x) + b·cos(2x) + 1 with maximum value φ(a,b),
    prove that the minimum value of φ(a,b) is 2. -/
theorem min_value_phi (a b : ℝ) (h : a^2 + b^2 - 4*b + 3 = 0) : 
  let f := fun (x : ℝ) ↦ a * Real.sin (2*x) + b * Real.cos (2*x) + 1
  let φ := fun (a b : ℝ) ↦ Real.sqrt (a^2 + b^2) + 1
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ φ a b ∧ 2 ≤ φ a b :=
by sorry

end NUMINAMATH_CALUDE_min_value_phi_l367_36728


namespace NUMINAMATH_CALUDE_triangle_area_l367_36710

/-- The area of a triangle with base 12 and height 9 is 54 -/
theorem triangle_area : ∀ (base height : ℝ), 
  base = 12 → height = 9 → (1/2 : ℝ) * base * height = 54 := by
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l367_36710


namespace NUMINAMATH_CALUDE_cube_digit_sum_l367_36786

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for nine-digit numbers -/
def is_nine_digit (n : ℕ) : Prop := sorry

theorem cube_digit_sum (N : ℕ) (h1 : is_nine_digit N) (h2 : sum_of_digits N = 3) :
  sum_of_digits (N^3) = 9 ∨ sum_of_digits (N^3) = 18 ∨ sum_of_digits (N^3) = 27 := by sorry

end NUMINAMATH_CALUDE_cube_digit_sum_l367_36786


namespace NUMINAMATH_CALUDE_neg_a_fourth_times_neg_a_squared_l367_36740

theorem neg_a_fourth_times_neg_a_squared (a : ℝ) : -a^4 * (-a)^2 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_neg_a_fourth_times_neg_a_squared_l367_36740


namespace NUMINAMATH_CALUDE_valid_sequences_count_l367_36734

-- Define the square
def Square := {A : ℝ × ℝ | A = (1, 1) ∨ A = (-1, 1) ∨ A = (-1, -1) ∨ A = (1, -1)}

-- Define the transformations
inductive Transform
| L  -- 90° counterclockwise rotation
| R  -- 90° clockwise rotation
| H  -- reflection across x-axis
| V  -- reflection across y-axis

-- Define a sequence of transformations
def TransformSequence := List Transform

-- Function to check if a transformation is a reflection
def isReflection (t : Transform) : Bool :=
  match t with
  | Transform.H => true
  | Transform.V => true
  | _ => false

-- Function to count reflections in a sequence
def countReflections (seq : TransformSequence) : Nat :=
  seq.filter isReflection |>.length

-- Function to check if a sequence maps the square back to itself
def mapsToSelf (seq : TransformSequence) : Bool :=
  sorry  -- Implementation details omitted

-- Theorem statement
theorem valid_sequences_count (n : Nat) :
  (∃ (seqs : List TransformSequence),
    (∀ seq ∈ seqs,
      seq.length = 24 ∧
      mapsToSelf seq ∧
      Even (countReflections seq)) ∧
    seqs.length = n) :=
  sorry

#check valid_sequences_count

end NUMINAMATH_CALUDE_valid_sequences_count_l367_36734


namespace NUMINAMATH_CALUDE_football_game_attendance_l367_36725

/-- Represents the number of adults attending the football game -/
def num_adults : ℕ := sorry

/-- Represents the number of children attending the football game -/
def num_children : ℕ := sorry

/-- The price of an adult ticket in cents -/
def adult_price : ℕ := 60

/-- The price of a child ticket in cents -/
def child_price : ℕ := 25

/-- The total number of attendees -/
def total_attendance : ℕ := 280

/-- The total money collected in cents -/
def total_money : ℕ := 14000

theorem football_game_attendance :
  (num_adults + num_children = total_attendance) ∧
  (num_adults * adult_price + num_children * child_price = total_money) →
  num_adults = 200 := by sorry

end NUMINAMATH_CALUDE_football_game_attendance_l367_36725


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l367_36759

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 + i) / (1 - i)) = 1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l367_36759


namespace NUMINAMATH_CALUDE_clothing_percentage_proof_l367_36711

theorem clothing_percentage_proof (food_percent : ℝ) (other_percent : ℝ) 
  (clothing_tax_rate : ℝ) (food_tax_rate : ℝ) (other_tax_rate : ℝ) 
  (total_tax_rate : ℝ) :
  food_percent = 20 →
  other_percent = 30 →
  clothing_tax_rate = 4 →
  food_tax_rate = 0 →
  other_tax_rate = 8 →
  total_tax_rate = 4.4 →
  (100 - food_percent - other_percent) * clothing_tax_rate / 100 + 
    food_percent * food_tax_rate / 100 + 
    other_percent * other_tax_rate / 100 = total_tax_rate →
  100 - food_percent - other_percent = 50 := by
sorry

end NUMINAMATH_CALUDE_clothing_percentage_proof_l367_36711


namespace NUMINAMATH_CALUDE_marble_drawing_probability_l367_36709

/-- The probability of drawing marbles consecutively by color --/
theorem marble_drawing_probability : 
  let total_marbles : ℕ := 12
  let blue_marbles : ℕ := 4
  let orange_marbles : ℕ := 3
  let green_marbles : ℕ := 5
  let favorable_outcomes : ℕ := Nat.factorial 3 * Nat.factorial blue_marbles * 
                                 Nat.factorial orange_marbles * Nat.factorial green_marbles
  let total_outcomes : ℕ := Nat.factorial total_marbles
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4620 := by
  sorry

end NUMINAMATH_CALUDE_marble_drawing_probability_l367_36709


namespace NUMINAMATH_CALUDE_calories_burned_per_mile_l367_36746

/-- Represents the calories burned per mile walked -/
def calories_per_mile : ℝ := sorry

/-- The total distance walked in miles -/
def total_distance : ℝ := 3

/-- The calories in the candy bar -/
def candy_bar_calories : ℝ := 200

/-- The net calorie deficit -/
def net_deficit : ℝ := 250

theorem calories_burned_per_mile :
  calories_per_mile * total_distance - candy_bar_calories = net_deficit ∧
  calories_per_mile = 150 := by sorry

end NUMINAMATH_CALUDE_calories_burned_per_mile_l367_36746


namespace NUMINAMATH_CALUDE_second_day_excess_is_47_l367_36733

/-- The number of people who swam on each day of a 3-day period at a public pool. -/
structure SwimmingData :=
  (total : ℕ)
  (day1 : ℕ)
  (day3 : ℕ)
  (h_total : total = day1 + day3 + (total - day1 - day3))
  (h_day2_gt_day3 : total - day1 - day3 > day3)

/-- The difference between the number of people who swam on the second day and the third day. -/
def secondDayExcess (data : SwimmingData) : ℕ :=
  data.total - data.day1 - 2 * data.day3

theorem second_day_excess_is_47 (data : SwimmingData)
  (h_total : data.total = 246)
  (h_day1 : data.day1 = 79)
  (h_day3 : data.day3 = 120) :
  secondDayExcess data = 47 := by
  sorry

end NUMINAMATH_CALUDE_second_day_excess_is_47_l367_36733


namespace NUMINAMATH_CALUDE_burning_candle_variables_l367_36799

/-- Represents a burning candle -/
structure BurningCandle where
  a : ℝ  -- Original length in centimeters
  t : ℝ  -- Burning time in minutes
  y : ℝ  -- Remaining length in centimeters

/-- Predicate to check if a quantity is variable in the context of a burning candle -/
def isVariable (candle : BurningCandle) (quantity : ℝ) : Prop :=
  ∃ (candle' : BurningCandle), candle.a = candle'.a ∧ quantity ≠ candle'.t

theorem burning_candle_variables (candle : BurningCandle) :
  (isVariable candle candle.t ∧ isVariable candle candle.y) ∧
  ¬(isVariable candle candle.a) := by
  sorry

#check burning_candle_variables

end NUMINAMATH_CALUDE_burning_candle_variables_l367_36799


namespace NUMINAMATH_CALUDE_southbound_cyclist_speed_l367_36704

/-- 
Given two cyclists starting from the same point and traveling in opposite directions,
with one cyclist traveling north at 10 km/h, prove that the speed of the southbound
cyclist is 15 km/h if they are 50 km apart after 2 hours.
-/
theorem southbound_cyclist_speed 
  (north_speed : ℝ) 
  (time : ℝ) 
  (distance : ℝ) 
  (h1 : north_speed = 10) 
  (h2 : time = 2) 
  (h3 : distance = 50) : 
  ∃ south_speed : ℝ, south_speed = 15 ∧ (north_speed + south_speed) * time = distance :=
sorry

end NUMINAMATH_CALUDE_southbound_cyclist_speed_l367_36704


namespace NUMINAMATH_CALUDE_employee_count_l367_36735

/-- Proves the number of employees given salary information -/
theorem employee_count 
  (avg_salary : ℝ) 
  (salary_increase : ℝ) 
  (manager_salary : ℝ) 
  (h1 : avg_salary = 1700)
  (h2 : salary_increase = 100)
  (h3 : manager_salary = 3800) :
  ∃ (E : ℕ), 
    (E : ℝ) * (avg_salary + salary_increase) = E * avg_salary + manager_salary ∧ 
    E = 20 :=
by sorry

end NUMINAMATH_CALUDE_employee_count_l367_36735


namespace NUMINAMATH_CALUDE_frank_reading_speed_l367_36742

/-- Given a book with a certain number of pages and the number of days to read it,
    calculate the number of pages read per day. -/
def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

/-- Theorem stating that Frank read 102 pages per day. -/
theorem frank_reading_speed :
  pages_per_day 612 6 = 102 := by
  sorry

end NUMINAMATH_CALUDE_frank_reading_speed_l367_36742


namespace NUMINAMATH_CALUDE_hand_mitt_cost_is_14_l367_36713

/-- The cost of cooking gear for Eve's nieces --/
def cooking_gear_cost (hand_mitt_cost : ℝ) : Prop :=
  let apron_cost : ℝ := 16
  let utensils_cost : ℝ := 10
  let knife_cost : ℝ := 2 * utensils_cost
  let total_cost_per_niece : ℝ := hand_mitt_cost + apron_cost + utensils_cost + knife_cost
  let discount_rate : ℝ := 0.75
  let number_of_nieces : ℕ := 3
  let total_spent : ℝ := 135
  discount_rate * (number_of_nieces : ℝ) * total_cost_per_niece = total_spent

theorem hand_mitt_cost_is_14 :
  ∃ (hand_mitt_cost : ℝ), cooking_gear_cost hand_mitt_cost ∧ hand_mitt_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_hand_mitt_cost_is_14_l367_36713


namespace NUMINAMATH_CALUDE_pizza_topping_options_l367_36771

/-- Represents the number of topping options for each category --/
structure ToppingOptions where
  cheese : Nat
  meat : Nat
  vegetable : Nat

/-- Represents the restriction between pepperoni and peppers --/
def hasPepperoniPepperRestriction : Bool := true

/-- Calculates the total number of topping combinations --/
def totalCombinations (options : ToppingOptions) (restriction : Bool) : Nat :=
  if restriction then
    options.cheese * (options.meat - 1) * options.vegetable +
    options.cheese * 1 * (options.vegetable - 1)
  else
    options.cheese * options.meat * options.vegetable

/-- The main theorem to prove --/
theorem pizza_topping_options :
  ∃ (options : ToppingOptions),
    options.cheese = 3 ∧
    options.vegetable = 5 ∧
    hasPepperoniPepperRestriction = true ∧
    totalCombinations options hasPepperoniPepperRestriction = 57 ∧
    options.meat = 4 := by
  sorry


end NUMINAMATH_CALUDE_pizza_topping_options_l367_36771


namespace NUMINAMATH_CALUDE_emmy_and_gerry_apples_l367_36793

/-- The number of apples that can be bought with a given amount of money at a given price per apple -/
def apples_buyable (money : ℕ) (price : ℕ) : ℕ :=
  money / price

theorem emmy_and_gerry_apples : 
  let apple_price : ℕ := 2
  let emmy_money : ℕ := 200
  let gerry_money : ℕ := 100
  apples_buyable emmy_money apple_price + apples_buyable gerry_money apple_price = 150 :=
by sorry

end NUMINAMATH_CALUDE_emmy_and_gerry_apples_l367_36793


namespace NUMINAMATH_CALUDE_train_speed_calculation_l367_36780

/-- Prove that given two trains of equal length, where the faster train travels at a given speed
    and passes the slower train in a given time, the speed of the slower train can be calculated. -/
theorem train_speed_calculation (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 65 →
  faster_speed = 49 →
  passing_time = 36 →
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    2 * train_length = (faster_speed - slower_speed) * (5 / 18) * passing_time :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l367_36780


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l367_36765

/-- Two vectors in R² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (m, 2)
  parallel a b → m = -Real.sqrt 2 ∨ m = Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_parallel_vectors_m_value_l367_36765


namespace NUMINAMATH_CALUDE_exp_greater_than_power_over_factorial_l367_36769

theorem exp_greater_than_power_over_factorial
  (x : ℝ) (n : ℕ) (h1 : x > 1) (h2 : n > 0) :
  Real.exp (x - 1) > x ^ n / n.factorial :=
sorry

end NUMINAMATH_CALUDE_exp_greater_than_power_over_factorial_l367_36769


namespace NUMINAMATH_CALUDE_ellipse_vertices_distance_l367_36715

/-- The distance between the vertices of the ellipse (x^2/144) + (y^2/36) = 1 is 24 -/
theorem ellipse_vertices_distance : 
  let ellipse := {p : ℝ × ℝ | (p.1^2 / 144) + (p.2^2 / 36) = 1}
  ∃ v1 v2 : ℝ × ℝ, v1 ∈ ellipse ∧ v2 ∈ ellipse ∧ 
    (∀ p ∈ ellipse, ‖p.1‖ ≤ ‖v1.1‖) ∧
    ‖v1 - v2‖ = 24 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_vertices_distance_l367_36715


namespace NUMINAMATH_CALUDE_square_difference_1001_999_l367_36751

theorem square_difference_1001_999 : 1001^2 - 999^2 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_1001_999_l367_36751


namespace NUMINAMATH_CALUDE_shirt_cost_l367_36719

theorem shirt_cost (J S : ℝ) 
  (eq1 : 3 * J + 2 * S = 69) 
  (eq2 : 2 * J + 3 * S = 76) : 
  S = 18 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l367_36719


namespace NUMINAMATH_CALUDE_sum_of_roots_even_l367_36747

theorem sum_of_roots_even (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (h_distinct : ∃ (x y : ℤ), x ≠ y ∧ x^2 - 2*p*x + p*q = 0 ∧ y^2 - 2*p*y + p*q = 0) :
  ∃ (k : ℤ), 2 * p = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_even_l367_36747


namespace NUMINAMATH_CALUDE_sonita_stamp_purchase_l367_36720

theorem sonita_stamp_purchase (two_q_stamps : ℕ) 
  (h1 : two_q_stamps > 0)
  (h2 : two_q_stamps < 9)
  (h3 : two_q_stamps % 5 = 0) :
  2 * two_q_stamps + 10 * two_q_stamps + (100 - 12 * two_q_stamps) / 5 = 63 := by
  sorry

#check sonita_stamp_purchase

end NUMINAMATH_CALUDE_sonita_stamp_purchase_l367_36720


namespace NUMINAMATH_CALUDE_complex_product_theorem_l367_36772

theorem complex_product_theorem (a : ℝ) (z₁ z₂ : ℂ) : 
  z₁ = a - 2*I ∧ z₂ = -1 + a*I ∧ (∃ b : ℝ, z₁ + z₂ = b*I) → z₁ * z₂ = 1 + 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l367_36772


namespace NUMINAMATH_CALUDE_min_tiles_needed_l367_36749

/-- Represents the dimensions of a rectangular object -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the area of a rectangle in square inches -/
def areaInSquareInches (rect : Rectangle) : ℕ := rect.length * rect.width

/-- Calculates the number of small rectangles needed to cover a larger rectangle -/
def tilesNeeded (smallRect : Rectangle) (largeRect : Rectangle) : ℕ :=
  (areaInSquareInches largeRect) / (areaInSquareInches smallRect)

theorem min_tiles_needed :
  let tile := Rectangle.mk 2 3
  let room := Rectangle.mk (feetToInches 3) (feetToInches 6)
  tilesNeeded tile room = 432 := by sorry

end NUMINAMATH_CALUDE_min_tiles_needed_l367_36749


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l367_36762

/-- Given an arithmetic sequence with first term 3^2 and third term 3^4, 
    the middle term y is equal to 45. -/
theorem arithmetic_sequence_middle_term : 
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
    a 0 = 3^2 →                                       -- first term
    a 2 = 3^4 →                                       -- third term
    a 1 = 45 :=                                       -- middle term (y)
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l367_36762


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l367_36763

theorem largest_divisor_of_expression (n : ℕ+) : 
  ∃ (m : ℕ), m = 2448 ∧ 
  (∀ k : ℕ+, (9^(2*k.val) - 8^(2*k.val) - 17) % m = 0) ∧
  (∀ m' : ℕ, m' > m → ∃ k : ℕ+, (9^(2*k.val) - 8^(2*k.val) - 17) % m' ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l367_36763


namespace NUMINAMATH_CALUDE_solution_set_equals_target_set_l367_36760

/-- The set of solutions for the system of equations with parameter a -/
def SolutionSet : Set (ℝ × ℝ) :=
  {(x, y) | ∃ a : ℝ, a * x + y = 2 * a + 3 ∧ x - a * y = a + 4}

/-- The circle with center (3, 1) and radius √5, excluding (2, -1) -/
def TargetSet : Set (ℝ × ℝ) :=
  {(x, y) | (x - 3)^2 + (y - 1)^2 = 5 ∧ (x, y) ≠ (2, -1)}

theorem solution_set_equals_target_set : SolutionSet = TargetSet := by sorry

end NUMINAMATH_CALUDE_solution_set_equals_target_set_l367_36760


namespace NUMINAMATH_CALUDE_square_formation_theorem_l367_36756

def sum_of_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

def can_form_square (n : ℕ) : Prop :=
  sum_of_natural_numbers n % 4 = 0

def min_breaks_for_square (n : ℕ) : ℕ :=
  let total := sum_of_natural_numbers n
  let remainder := total % 4
  if remainder = 0 then 0
  else if remainder = 1 || remainder = 3 then 1
  else 2

theorem square_formation_theorem :
  (min_breaks_for_square 12 = 2) ∧
  (can_form_square 15 = true) := by sorry

end NUMINAMATH_CALUDE_square_formation_theorem_l367_36756


namespace NUMINAMATH_CALUDE_total_distance_is_200_l367_36732

/-- Represents the cycling journey of Jack and Peter -/
structure CyclingJourney where
  speed : ℝ
  timeHomeToStore : ℝ
  timeStoreToPeter : ℝ
  distanceStoreToPeter : ℝ

/-- Calculates the total distance cycled by Jack and Peter -/
def totalDistanceCycled (journey : CyclingJourney) : ℝ :=
  let distanceHomeToStore := journey.speed * journey.timeHomeToStore
  let distanceStoreToPeter := journey.distanceStoreToPeter
  distanceHomeToStore + 2 * distanceStoreToPeter

/-- Theorem stating the total distance cycled is 200 miles -/
theorem total_distance_is_200 (journey : CyclingJourney) 
  (h1 : journey.timeHomeToStore = 2 * journey.timeStoreToPeter)
  (h2 : journey.speed > 0)
  (h3 : journey.distanceStoreToPeter = 50) :
  totalDistanceCycled journey = 200 := by
  sorry


end NUMINAMATH_CALUDE_total_distance_is_200_l367_36732


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l367_36727

theorem two_digit_number_problem (n m : ℕ) : 
  10 ≤ m ∧ m < n ∧ n ≤ 99 →  -- n and m are 2-digit numbers, n > m
  n - m = 58 →  -- difference is 58
  n^2 % 100 = m^2 % 100 →  -- last two digits of squares are the same
  m = 21 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l367_36727


namespace NUMINAMATH_CALUDE_min_value_of_z_l367_36731

theorem min_value_of_z (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 12) :
  ∃ (z_min : ℝ), z_min = -5 ∧ ∀ (z : ℝ), z = 2 * x + Real.sqrt 3 * y → z ≥ z_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_z_l367_36731


namespace NUMINAMATH_CALUDE_point_on_y_axis_l367_36736

/-- 
If a point P with coordinates (a-1, a²-9) lies on the y-axis, 
then its coordinates are (0, -8).
-/
theorem point_on_y_axis (a : ℝ) : 
  (a - 1 = 0) → (a - 1, a^2 - 9) = (0, -8) := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l367_36736


namespace NUMINAMATH_CALUDE_qin_jiushao_v3_value_l367_36703

def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

def v_3 (x : ℝ) : ℝ := ((7*x + 6)*x + 5)*x + 4

theorem qin_jiushao_v3_value : v_3 3 = 262 := by
  sorry

end NUMINAMATH_CALUDE_qin_jiushao_v3_value_l367_36703


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l367_36782

theorem roots_of_quadratic (x : ℝ) : x^2 = 5*x ↔ x = 0 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l367_36782


namespace NUMINAMATH_CALUDE_smallest_integer_greater_than_neg_seventeen_thirds_l367_36754

theorem smallest_integer_greater_than_neg_seventeen_thirds :
  Int.ceil (-17 / 3 : ℚ) = -5 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_greater_than_neg_seventeen_thirds_l367_36754


namespace NUMINAMATH_CALUDE_gildas_marbles_l367_36795

theorem gildas_marbles (initial_marbles : ℝ) (initial_marbles_pos : initial_marbles > 0) :
  let remaining_after_pedro := initial_marbles * (1 - 0.25)
  let remaining_after_ebony := remaining_after_pedro * (1 - 0.15)
  let remaining_after_jimmy := remaining_after_ebony * (1 - 0.30)
  (remaining_after_jimmy / initial_marbles) * 100 = 44.625 := by
sorry

end NUMINAMATH_CALUDE_gildas_marbles_l367_36795


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l367_36798

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 49) 
  (h2 : x * y = -8) : 
  x^2 + y^2 = 33 := by sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l367_36798


namespace NUMINAMATH_CALUDE_total_pears_picked_l367_36743

def alyssa_pears : ℕ := 42
def nancy_pears : ℕ := 17

theorem total_pears_picked :
  alyssa_pears + nancy_pears = 59 := by sorry

end NUMINAMATH_CALUDE_total_pears_picked_l367_36743


namespace NUMINAMATH_CALUDE_circle_bisector_properties_l367_36744

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point A
def A : ℝ × ℝ := (6, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define a point P on the circle
def P (x y : ℝ) : Prop := Circle x y

-- Define point M on the bisector of ∠POA and on PA
def M (x y : ℝ) (px py : ℝ) : Prop :=
  P px py ∧ 
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
  x = t * px + (1 - t) * A.1 ∧
  y = t * py + (1 - t) * A.2 ∧
  (x - O.1) * (A.1 - O.1) + (y - O.2) * (A.2 - O.2) = 
  (px - O.1) * (A.1 - O.1) + (py - O.2) * (A.2 - O.2)

-- Theorem statement
theorem circle_bisector_properties 
  (x y px py : ℝ) 
  (h_m : M x y px py) :
  (∃ (ma pm : ℝ), ma / pm = 3 ∧ 
    ma^2 = (x - A.1)^2 + (y - A.2)^2 ∧
    pm^2 = (x - px)^2 + (y - py)^2) ∧
  (x - 2/3)^2 + y^2 = 9/4 :=
sorry

end NUMINAMATH_CALUDE_circle_bisector_properties_l367_36744


namespace NUMINAMATH_CALUDE_min_value_ab_l367_36726

theorem min_value_ab (a b : ℝ) (h : 0 < a ∧ 0 < b) (eq : 1/a + 2/b = Real.sqrt (a*b)) : 
  2 * Real.sqrt 2 ≤ a * b := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l367_36726


namespace NUMINAMATH_CALUDE_mixture_composition_l367_36723

theorem mixture_composition (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 100) :
  0.4 * x + 0.5 * y = 47 → x = 30 :=
by sorry

end NUMINAMATH_CALUDE_mixture_composition_l367_36723


namespace NUMINAMATH_CALUDE_servant_service_duration_l367_36766

/-- Represents the servant's employment contract and actual service --/
structure ServantContract where
  yearlyPayment : ℕ  -- Payment in Rupees for a full year of service
  uniformPrice : ℕ   -- Price of the uniform in Rupees
  actualPayment : ℕ  -- Actual payment received in Rupees
  actualUniform : Bool -- Whether the servant received the uniform

/-- Calculates the number of months served based on the contract and actual payment --/
def monthsServed (contract : ServantContract) : ℚ :=
  let totalYearlyValue := contract.yearlyPayment + contract.uniformPrice
  let actualTotalReceived := contract.actualPayment + 
    (if contract.actualUniform then contract.uniformPrice else 0)
  let fractionWorked := (totalYearlyValue - actualTotalReceived) / contract.yearlyPayment
  12 * (1 - fractionWorked)

/-- Theorem stating that given the problem conditions, the servant served for approximately 3 months --/
theorem servant_service_duration (contract : ServantContract) 
  (h1 : contract.yearlyPayment = 900)
  (h2 : contract.uniformPrice = 100)
  (h3 : contract.actualPayment = 650)
  (h4 : contract.actualUniform = true) :
  ∃ (m : ℕ), m = 3 ∧ abs (monthsServed contract - m) < 1 := by
  sorry

end NUMINAMATH_CALUDE_servant_service_duration_l367_36766


namespace NUMINAMATH_CALUDE_extraordinary_stack_size_l367_36777

/-- An extraordinary stack of cards -/
structure ExtraordinaryStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (pile_a_size : ℕ := n)
  (pile_b_size : ℕ := n)
  (card_57_from_a_position : ℕ := 57)
  (card_200_from_b_position : ℕ := 200)

/-- The number of cards in an extraordinary stack is 198 -/
theorem extraordinary_stack_size :
  ∀ (stack : ExtraordinaryStack),
    stack.card_57_from_a_position % 2 = 1 →
    stack.card_200_from_b_position % 2 = 0 →
    stack.card_57_from_a_position ≤ stack.total_cards →
    stack.card_200_from_b_position ≤ stack.total_cards →
    stack.total_cards = 198 := by
  sorry

end NUMINAMATH_CALUDE_extraordinary_stack_size_l367_36777


namespace NUMINAMATH_CALUDE_sin_beta_value_l367_36718

-- Define acute angles
def is_acute (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- State the theorem
theorem sin_beta_value (α β : Real) 
  (h_acute_α : is_acute α) (h_acute_β : is_acute β)
  (h_sin_α : Real.sin α = (4/7) * Real.sqrt 3)
  (h_cos_sum : Real.cos (α + β) = -11/14) :
  Real.sin β = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_beta_value_l367_36718


namespace NUMINAMATH_CALUDE_solution_set_f_leq_2abs_condition_on_abc_l367_36773

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + |x - 2|

-- Theorem 1: Solution set of f(x) ≤ 2|x|
theorem solution_set_f_leq_2abs (x : ℝ) :
  x ∈ {y : ℝ | f y ≤ 2 * |y|} ↔ x ∈ Set.Icc 1 2 :=
sorry

-- Theorem 2: Condition on a, b, c
theorem condition_on_abc (a b c : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 + 4*b^2 + 5*c^2 - 1/4) → a*c + 4*b*c ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_2abs_condition_on_abc_l367_36773


namespace NUMINAMATH_CALUDE_sum_of_A_and_D_is_six_l367_36775

-- Define single-digit numbers
def SingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

-- Define three-digit number ABX
def ThreeDigitABX (A B X : ℕ) : ℕ := 100 * A + 10 * B + X

-- Define three-digit number CDY
def ThreeDigitCDY (C D Y : ℕ) : ℕ := 100 * C + 10 * D + Y

-- Define four-digit number XYXY
def FourDigitXYXY (X Y : ℕ) : ℕ := 1000 * X + 100 * Y + 10 * X + Y

-- Theorem statement
theorem sum_of_A_and_D_is_six 
  (A B C D X Y : ℕ) 
  (hA : SingleDigit A) (hB : SingleDigit B) (hC : SingleDigit C) 
  (hD : SingleDigit D) (hX : SingleDigit X) (hY : SingleDigit Y)
  (h_sum : ThreeDigitABX A B X + ThreeDigitCDY C D Y = FourDigitXYXY X Y) :
  A + D = 6 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_A_and_D_is_six_l367_36775


namespace NUMINAMATH_CALUDE_geometry_propositions_l367_36778

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) := by
  sorry

end NUMINAMATH_CALUDE_geometry_propositions_l367_36778

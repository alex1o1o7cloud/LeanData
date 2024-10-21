import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_dot_product_OA_OB_l217_21765

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the point P
def P : ℝ × ℝ := (3, 1)

-- Define a point on the parabola
def on_parabola (Q : ℝ × ℝ) : Prop := parabola Q.1 Q.2

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem 1: Minimum sum of distances
theorem min_sum_distances :
  ∃ (m : ℝ), m = 4 ∧ ∀ (Q : ℝ × ℝ), on_parabola Q → distance Q P + distance Q focus ≥ m := by
  sorry

-- Theorem 2: Dot product of OA and OB
theorem dot_product_OA_OB :
  ∀ (A B : ℝ × ℝ), on_parabola A ∧ on_parabola B ∧ 
  (∃ (k : ℝ), A.2 - focus.2 = k * (A.1 - focus.1) ∧ B.2 - focus.2 = k * (B.1 - focus.1)) →
  A.1 * B.1 + A.2 * B.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_dot_product_OA_OB_l217_21765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l217_21775

/-- Hyperbola struct representing the equation (x²/a² - y²/b² = 1) -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define a membership relation for a point on a hyperbola -/
def Point.onHyperbola (P : Point) (C : Hyperbola) : Prop :=
  (P.x^2 / C.a^2) - (P.y^2 / C.b^2) = 1

/-- Theorem: Given a hyperbola C and a point P on C satisfying certain conditions,
    the eccentricity of C is √2 + 1 -/
theorem hyperbola_eccentricity (C : Hyperbola) (P F₁ F₂ : Point)
    (h_on_hyperbola : Point.onHyperbola P C)
    (h_perpendicular : (P.x - F₁.x) * (F₂.x - F₁.x) + (P.y - F₁.y) * (F₂.y - F₁.y) = 0)
    (h_equal_distance : (P.x - F₁.x)^2 + (P.y - F₁.y)^2 = (F₂.x - F₁.x)^2 + (F₂.y - F₁.y)^2) :
    C.a / Real.sqrt (C.a^2 + C.b^2) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l217_21775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l217_21778

noncomputable def f (x : ℝ) : ℝ := 2^(-|x|+1)

theorem range_of_f :
  Set.range f = Set.Ioo 0 2 ∪ {2} := by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l217_21778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l217_21751

/-- Represents a parabola y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a line with slope m passing through point (x₀, y₀) -/
structure Line where
  m : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- The focus of a parabola -/
def focus (para : Parabola) : ℝ × ℝ := (para.p, 0)

/-- The equation of a line -/
def line_eq (l : Line) (x : ℝ) : ℝ := l.m * (x - l.x₀) + l.y₀

/-- Intersection points of a line and a parabola -/
def intersection (para : Parabola) (l : Line) : Set (ℝ × ℝ) :=
  {(x, y) | y^2 = 2 * para.p * x ∧ y = line_eq l x}

/-- Area of a triangle given three points -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

theorem parabola_triangle_area 
  (para : Parabola)
  (h_focus : focus para = (1, 0))
  (l : Line)
  (h_slope : l.m = 2)
  (h_through_focus : l.x₀ = 1 ∧ l.y₀ = 0)
  (A B : ℝ × ℝ)
  (h_AB : A ∈ intersection para l ∧ B ∈ intersection para l ∧ A ≠ B) :
  triangle_area (0, 0) A B = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l217_21751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_distribution_l217_21730

/-- Proves that if Li Ming gives 14 books to Wang Hong, he will have 2 fewer books than Wang Hong,
    given that Li Ming initially had 26 more books than Wang Hong. -/
theorem book_distribution (initial_difference : ℤ) (books_given : ℤ) (final_difference : ℤ) : 
  initial_difference = 26 →
  books_given = 14 →
  final_difference = 2 →
  initial_difference - 2 * books_given = -final_difference := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_distribution_l217_21730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shins_burger_count_indeterminate_l217_21747

/-- Represents the cost of items at the restaurant --/
structure RestaurantCost where
  burger : ℝ
  shake : ℝ
  cola : ℝ

/-- Represents an order at the restaurant --/
structure RestaurantOrder where
  burgers : ℕ
  shakes : ℕ
  colas : ℕ

/-- Calculate the total cost of an order --/
def orderCost (cost : RestaurantCost) (order : RestaurantOrder) : ℝ :=
  cost.burger * order.burgers + cost.shake * order.shakes + cost.cola * order.colas

/-- The theorem stating that Shin's burger count cannot be uniquely determined --/
theorem shins_burger_count_indeterminate (cost : RestaurantCost) :
  ∃ (shinsOrder1 shinsOrder2 : RestaurantOrder),
    shinsOrder1.burgers ≠ shinsOrder2.burgers ∧
    shinsOrder1.shakes = 7 ∧ shinsOrder2.shakes = 7 ∧
    shinsOrder1.colas = 1 ∧ shinsOrder2.colas = 1 ∧
    orderCost cost shinsOrder1 = 120 ∧
    orderCost cost shinsOrder2 = 120 ∧
    orderCost cost { burgers := 4, shakes := 10, colas := 1 } = 158.50 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shins_burger_count_indeterminate_l217_21747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sqrt_three_implies_sin_2theta_over_one_plus_cos_2theta_l217_21794

theorem tan_sqrt_three_implies_sin_2theta_over_one_plus_cos_2theta (θ : ℝ) :
  Real.tan θ = Real.sqrt 3 → (Real.sin (2*θ)) / (1 + Real.cos (2*θ)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sqrt_three_implies_sin_2theta_over_one_plus_cos_2theta_l217_21794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l217_21754

/-- f(n) denotes the number of ways of representing n as a sum of powers of 2 with nonnegative integer exponents -/
def f : ℕ → ℕ := sorry

/-- The main theorem stating the bounds for f(2^n) -/
theorem f_bounds (n : ℕ) (h : n ≥ 3) : 
  (2 : ℝ) ^ (n^2 / 4 : ℝ) < (f (2^n) : ℝ) ∧ (f (2^n) : ℝ) < (2 : ℝ) ^ (n^2 / 2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l217_21754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_unbounded_l217_21734

-- Define the line l passing through (-2, 0) with slope k
def line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + 2)

-- Define the circle x^2 + y^2 = 2x
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 2*x

-- Theorem statement
theorem slope_range_unbounded :
  ∀ k : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    line k x₁ y₁ ∧
    line k x₂ y₂ ∧
    circle_equation x₁ y₁ ∧
    circle_equation x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_unbounded_l217_21734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l217_21713

/-- The area of a triangle given its side lengths using Heron's formula -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 30, 28, and 12 is approximately 110.84 -/
theorem triangle_area_approx :
  ∃ ε > 0, |triangle_area 30 28 12 - 110.84| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l217_21713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_PQRS_equals_7sqrt2_plus_sqrt10_l217_21746

-- Define the points
def P : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (3, 4)
def R : ℝ × ℝ := (6, 1)
def S : ℝ × ℝ := (8, -1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter of PQRS
noncomputable def perimeter_PQRS : ℝ :=
  distance P Q + distance Q R + distance R S + distance S P

-- Theorem statement
theorem perimeter_PQRS_equals_7sqrt2_plus_sqrt10 :
  perimeter_PQRS = 7 * Real.sqrt 2 + Real.sqrt 10 := by
  sorry

#check perimeter_PQRS_equals_7sqrt2_plus_sqrt10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_PQRS_equals_7sqrt2_plus_sqrt10_l217_21746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_matches_l217_21755

theorem cricket_team_matches (initial_win_rate final_win_rate : ℚ) 
  (additional_wins : ℕ) (h1 : initial_win_rate = 28/100) 
  (h2 : final_win_rate = 52/100) (h3 : additional_wins = 60) : ℕ :=
  let initial_matches := 120
  by
    have h4 : initial_win_rate * initial_matches + additional_wins = 
              final_win_rate * (initial_matches + additional_wins) := by sorry
    exact initial_matches

-- Remove the #eval line as it's causing issues
-- #eval cricket_team_matches 0.28 0.52 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_matches_l217_21755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l217_21790

noncomputable def f (x : ℝ) : ℝ := (2*x^4 - 8*x^3 + 12*x^2 - 8*x + 2) / (x^3 - 5*x^2 + 6*x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l217_21790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pebble_distribution_impossibility_l217_21744

theorem pebble_distribution_impossibility :
  ¬ ∃ (n : ℕ), n > 0 ∧
    (let first_share := n / 2 + 1
     let remaining_after_first := n - first_share
     let second_share := remaining_after_first / 3
     let third_share := 2 * second_share
     first_share + second_share + third_share = n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pebble_distribution_impossibility_l217_21744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_symmetry_l217_21712

/-- A quadratic function of the form y = Ax² + C -/
def quadratic_function (A C : ℝ) : ℝ → ℝ := λ x ↦ A * x^2 + C

theorem quadratic_symmetry (A C x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) :
  quadratic_function A C x₁ = quadratic_function A C x₂ →
  quadratic_function A C (x₁ + x₂) = C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_symmetry_l217_21712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_and_range_of_f_l217_21735

-- Define the given conditions
noncomputable def alpha : ℝ := Real.pi / 3

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x * Real.sin (x - alpha)

-- State the theorem
theorem alpha_and_range_of_f :
  (2 * Real.sin alpha * Real.tan alpha = 3) ∧
  (0 < alpha) ∧ (alpha < Real.pi) →
  (alpha = Real.pi / 3) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4),
    f x ∈ Set.Icc (-1) 0) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x = -1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_and_range_of_f_l217_21735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l217_21715

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 6 * (Real.cos x)^2 - Real.sqrt 3 * Real.sin (2 * x)

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) :
  t.B = π / 12 →
  f t.A = 3 - 2 * Real.sqrt 3 →
  (∀ x, f x ≤ 2 * Real.sqrt 3 + 3) ∧
  (∀ ε > 0, ∃ x, x > 0 ∧ x < π + ε ∧ f x = f (x + π)) ∧
  t.a / t.c = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l217_21715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_60_l217_21741

noncomputable def polynomial (x : ℝ) : ℝ := (x - 1) * (x^2 - 2) * (x^3 - 3) * (x^4 - 4) * (x^5 - 5) * 
                               (x^6 - 6) * (x^7 - 7) * (x^8 - 8) * (x^9 - 9) * (x^10 - 10) * 
                               (x^11 - 11) * (x^12 - 12)

theorem coefficient_of_x_60 : 
  ∃ (p : Polynomial ℝ), (∀ x, polynomial x = p.eval x) ∧ p.coeff 60 = 156 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_60_l217_21741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_distance_halfway_l217_21799

/-- A convex polygon represented by its perimeter length -/
structure ConvexPolygon where
  perimeter : ℝ
  perimeter_positive : perimeter > 0

/-- A point on the perimeter of a convex polygon -/
structure PerimeterPoint (p : ConvexPolygon) where
  position : ℝ
  position_on_perimeter : 0 ≤ position ∧ position < p.perimeter

/-- The distance between two points on the perimeter -/
noncomputable def perimeterDistance (p : ConvexPolygon) (a b : PerimeterPoint p) : ℝ :=
  min (abs (b.position - a.position)) (p.perimeter - abs (b.position - a.position))

/-- The theorem stating that the maximum minimum distance is achieved when points start halfway around the perimeter -/
theorem max_min_distance_halfway (p : ConvexPolygon) :
  ∃ (a b : PerimeterPoint p),
    perimeterDistance p a b = p.perimeter / 2 ∧
    ∀ (c d : PerimeterPoint p),
      perimeterDistance p c d ≤ perimeterDistance p a b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_distance_halfway_l217_21799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_lambda_l217_21752

theorem smallest_lambda (n : ℕ) (hn : n > 0) :
  ∃ (lambda : ℝ), lambda > 0 ∧
  (∃ (a : Fin n → ℝ), (∀ i, 0 ≤ a i ∧ a i ≤ 1) ∧
    (∀ (x : Fin n → ℝ), (∀ i j, i ≤ j → x i ≤ x j) →
      (∀ i, 0 ≤ x i ∧ x i ≤ 1) →
        (∃ i, |x i - a i| ≤ lambda))) ∧
  (∀ (lambda' : ℝ), lambda' > 0 →
    (∃ (a : Fin n → ℝ), (∀ i, 0 ≤ a i ∧ a i ≤ 1) ∧
      (∀ (x : Fin n → ℝ), (∀ i j, i ≤ j → x i ≤ x j) →
        (∀ i, 0 ≤ x i ∧ x i ≤ 1) →
          (∃ i, |x i - a i| ≤ lambda'))) →
    lambda ≤ lambda') ∧
  lambda = 1 / (2 * n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_lambda_l217_21752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_vector_sum_implies_60_degree_angle_l217_21768

-- Define the triangle ABC
structure Triangle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] where
  A : V
  B : V
  C : V

-- Define the altitudes of the triangle
noncomputable def altitude {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (t : Triangle V) (v : V) : V := sorry

-- Define IsAcute for a triangle
def IsAcute {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (t : Triangle V) : Prop := sorry

-- Define the angle function
noncomputable def angle {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (A B C : V) : ℝ := sorry

-- State the theorem
theorem altitude_vector_sum_implies_60_degree_angle 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (t : Triangle V) :
  let AD := altitude t t.A
  let BE := altitude t t.B
  let CF := altitude t t.C
  IsAcute t →
  5 • AD + 7 • BE + 2 • CF = 0 →
  angle t.A t.C t.B = 60 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_vector_sum_implies_60_degree_angle_l217_21768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_root_of_unity_exponent_l217_21703

theorem smallest_root_of_unity_exponent : ∃ (n : ℕ), 
  (∀ (z : ℂ), z^6 - z^3 + 1 = 0 → z^n = 1) ∧ 
  (∀ (m : ℕ), 0 < m → m < n → ∃ (w : ℂ), w^6 - w^3 + 1 = 0 ∧ w^m ≠ 1) := by
  -- Define the polynomial
  let p (z : ℂ) := z^6 - z^3 + 1

  -- Claim that n = 9 satisfies the conditions
  use 9

  sorry -- Proof steps would go here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_root_of_unity_exponent_l217_21703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l217_21722

theorem binomial_expansion_properties (x : ℝ) (n : ℕ) :
  n + 1 = 10 →
  (n.choose 4 = 126 ∧ 
   (∃ k : ℕ, k ≤ n ∧ 
    (n.choose k : ℝ) * (1/2)^(n - k) * (-1)^k = -21/16)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l217_21722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2500_is_8_l217_21702

noncomputable def consecutive_integers_decimal (n : ℕ) : ℚ :=
  -- Definition of the decimal number formed by consecutive integers up to n
  sorry

noncomputable def is_nth_digit (q : ℚ) (n : ℕ) (d : ℕ) : Prop :=
  -- Definition to check if d is the nth digit after the decimal point in q
  sorry

theorem digit_2500_is_8 :
  let x := consecutive_integers_decimal 800
  ∃ (d : ℕ), d = 8 ∧ is_nth_digit x 2500 d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2500_is_8_l217_21702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_fourth_quadrant_product_l217_21797

-- Define complex numbers z₁ and z₂
def z₁ (a : ℝ) : ℂ := a - 2*Complex.I
def z₂ : ℂ := 3 + 4*Complex.I

-- Theorem for the first part of the problem
theorem pure_imaginary_product (a : ℝ) :
  (z₁ a * z₂).re = 0 → a = -8/3 :=
by sorry

-- Theorem for the second part of the problem
theorem fourth_quadrant_product (a : ℝ) :
  (z₁ a * z₂).re > 0 ∧ (z₁ a * z₂).im < 0 → -8/3 < a ∧ a < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_fourth_quadrant_product_l217_21797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_lambda_range_l217_21743

def a : ℝ × ℝ := (1, -2)
def b (lambda : ℝ) : ℝ × ℝ := (-1, lambda)

def is_obtuse (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 < 0 ∧ ¬∃ (mu : ℝ), mu < 0 ∧ v = (mu * w.1, mu * w.2)

theorem obtuse_angle_lambda_range :
  ∀ lambda : ℝ, is_obtuse a (b lambda) → lambda ∈ Set.Ioo (-1/2) 2 ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_lambda_range_l217_21743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l217_21728

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := ((x - 3) * (12 - x)) / x

-- State the theorem
theorem max_value_of_f :
  ∀ x : ℝ, 3 < x → x < 12 → f x ≤ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l217_21728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_sum_l217_21720

theorem cookie_sum : 
  (Finset.filter (fun N : ℕ => N < 50 ∧ N % 4 = 1 ∧ N % 6 = 5) (Finset.range 50)).sum id = 208 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_sum_l217_21720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_inequality_l217_21725

/-- The function f(x) = x*e^x + m*x^2 - n*x -/
noncomputable def f (x m n : ℝ) : ℝ := x * Real.exp x + m * x^2 - n * x

/-- The function g(x) = f(x) + e^x -/
noncomputable def g (x : ℝ) : ℝ := f x (-1/2) 2 + Real.exp x

/-- The derivative of f with respect to x -/
noncomputable def f_deriv (x m n : ℝ) : ℝ := (x + 1) * Real.exp x + 2 * m * x - n

theorem f_monotonicity_and_inequality :
  (∀ x : ℝ, x < -2 → HasDerivAt g ((x + 2) * (Real.exp x - 1)) x) ∧ 
  (∀ x : ℝ, -2 < x → x < 0 → HasDerivAt g ((x + 2) * (Real.exp x - 1)) x) ∧
  (∀ x : ℝ, 0 < x → HasDerivAt g ((x + 2) * (Real.exp x - 1)) x) ∧
  (∀ x m n : ℝ, f_deriv x m n ≤ (x + 2) * Real.exp x → m - n / 2 ≤ Real.exp 1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_inequality_l217_21725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_200m_12s_l217_21708

/-- Calculates the speed of a train in km/hr given its length in meters and time to cross a pole in seconds -/
noncomputable def train_speed (length_m : ℝ) (time_s : ℝ) : ℝ :=
  (length_m / 1000) / (time_s / 3600)

/-- Theorem stating that a train with length 200 meters crossing a pole in 12 seconds has a speed of 60 km/hr -/
theorem train_speed_200m_12s :
  train_speed 200 12 = 60 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the arithmetic
  simp [div_div_eq_mul_div]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_200m_12s_l217_21708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_base5_number_l217_21798

def is_two_digit_base5 (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 44

def alternating_sequence (seq : List ℕ) : Prop :=
  ∀ i : ℕ, i + 1 < seq.length → 
    if i % 2 = 0 
    then seq[i]! + 1 = seq[i+1]! 
    else seq[i]! = seq[i+1]! + 1

noncomputable def compose_base5 (seq : List ℕ) : ℕ := 
  seq.foldl (fun acc x => acc * 100 + x) 0

theorem no_special_base5_number : ¬ ∃ (x : ℕ) (seq : List ℕ),
  seq.length = 2021 ∧ 
  (∀ n ∈ seq, is_two_digit_base5 n) ∧
  alternating_sequence seq ∧
  x = compose_base5 seq ∧
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ q = p + 2 ∧ x = p * q :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_base5_number_l217_21798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_polar_coordinates_l217_21709

/-- PolarCoord represents a point in polar coordinates -/
structure PolarCoord where
  r : ℝ
  θ : ℝ

/-- symmetric_point A l returns the symmetric point of A with respect to line l -/
noncomputable def symmetric_point (A : PolarCoord) (l : ℝ → ℝ → Prop) : PolarCoord :=
  sorry

/-- The symmetric point of A (2, π/2) with respect to the line ρcosθ=1 has polar coordinates (2√2, π/4) -/
theorem symmetric_point_polar_coordinates :
  let A : PolarCoord := ⟨2, Real.pi/2⟩
  let l : ℝ → ℝ → Prop := fun ρ θ ↦ ρ * Real.cos θ = 1
  let B : PolarCoord := symmetric_point A l
  B.r = 2 * Real.sqrt 2 ∧ B.θ = Real.pi/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_polar_coordinates_l217_21709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_grade_is_three_l217_21719

def is_standard_grade (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5

def product_of_grades : ℕ := 2007

theorem final_grade_is_three :
  ∃ (grades : List ℕ),
    (∀ g, g ∈ grades → is_standard_grade g) ∧
    (grades.prod = product_of_grades) ∧
    (3 ∈ grades) ∧
    (∀ g, g ∈ grades → g ≠ 3 → g < 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_grade_is_three_l217_21719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_AP_is_quadratic_l217_21705

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- Sum of the first n terms of an arithmetic progression -/
noncomputable def sum_AP (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * ap.a + (n - 1 : ℝ) * ap.d)

/-- Theorem: The sum of the first n terms of an arithmetic progression is a quadratic function of n -/
theorem sum_AP_is_quadratic (ap : ArithmeticProgression) :
  ∃ a b c : ℝ, ∀ n : ℕ, sum_AP ap n = a * (n : ℝ)^2 + b * (n : ℝ) + c := by
  -- We'll provide the coefficients explicitly
  use ap.d / 2, ap.a - ap.d / 2, 0
  intro n
  -- Expand the definition of sum_AP and simplify
  simp [sum_AP]
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_AP_is_quadratic_l217_21705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_inequality_l217_21789

/-- A function representing the allowed operations of the calculator -/
def calculator_op (x : ℕ) : ℕ → ℕ
| 0 => x^2
| 1 => x + 1
| _ => x

/-- A predicate to check if a sequence of operations is valid -/
def valid_ops (ops : List ℕ) : Prop :=
  ∀ i, i + 1 < ops.length → ¬(ops[i]? = some 1 ∧ ops[i+1]? = some 1)

/-- The result of applying a sequence of operations to a number -/
def apply_ops (x : ℕ) (ops : List ℕ) : ℕ :=
  ops.foldl calculator_op x

theorem calculator_inequality (x n S : ℕ) (ops : List ℕ)
  (h_pos_x : x > 0) (h_pos_n : n > 0) (h_pos_S : S > 0)
  (h_valid : valid_ops ops)
  (h_result : apply_ops x ops = S)
  (h_inequality : S > x^n + 1) :
  S > x^n + x - 1 := by
  sorry

#check calculator_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_inequality_l217_21789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_is_534_l217_21784

def digits : List Nat := [0, 4, 5, 6, 7, 8, 9]

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.length = 6 ∧ arr.toFinset ⊆ digits.toFinset

def to_number (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

def sum_of_numbers (arr : List Nat) : Nat :=
  match arr with
  | [a, b, c, d, e, f] => to_number a b c + to_number d e f
  | _ => 0  -- Default case for invalid arrangements

theorem smallest_sum_is_534 :
  ∀ arr : List Nat, is_valid_arrangement arr → sum_of_numbers arr ≥ 534 :=
by
  intro arr h
  sorry  -- Placeholder for the actual proof

#eval sum_of_numbers [4, 5, 6, 0, 7, 8]  -- Should output 534

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_is_534_l217_21784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_score_l217_21760

def total_balls : ℕ := 10
def red_balls : ℕ := 5
def yellow_balls : ℕ := 5
def balls_drawn : ℕ := 5

def score (same_color : ℕ) : ℝ :=
  if same_color = 5 then 100
  else if same_color = 4 then 50
  else 0

noncomputable def prob_all_same : ℝ := 2 / (Nat.choose total_balls balls_drawn)
noncomputable def prob_four_same : ℝ := (2 * Nat.choose red_balls 4 * Nat.choose yellow_balls 1) / (Nat.choose total_balls balls_drawn)

theorem expected_score :
  (prob_all_same * score 5 + prob_four_same * score 4 + (1 - prob_all_same - prob_four_same) * score 3) = 75 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_score_l217_21760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l217_21717

/-- The area of a triangle with vertices at (0,0,0), (2,4,6), and (1,2,1) is 5. -/
theorem triangle_area : ∃ (area : ℝ), area = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l217_21717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_to_combined_assets_ratio_l217_21723

/-- The price of Company KW relative to the assets of Companies A and B -/
noncomputable def price_ratio (price_kw : ℝ) (assets_a : ℝ) (assets_b : ℝ) : ℝ :=
  price_kw / (assets_a + assets_b)

/-- The theorem stating the relationship between the price of Company KW and the combined assets of Companies A and B -/
theorem price_to_combined_assets_ratio 
  (price_kw : ℝ) (assets_a : ℝ) (assets_b : ℝ)
  (h1 : price_kw = 1.4 * assets_a)
  (h2 : price_kw = 2 * assets_b)
  (h3 : assets_a > 0)
  (h4 : assets_b > 0) :
  ∃ ε > 0, |price_ratio price_kw assets_a assets_b - 0.8235| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_to_combined_assets_ratio_l217_21723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_tan_x_value_l217_21782

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x + a * Real.cos x

-- Part 1
theorem max_value_of_f (x : ℝ) : 
  ∃ (M : ℝ), M = 2 ∧ ∀ y, f (Real.sqrt 3) y ≤ M := by sorry

-- Part 2
theorem tan_x_value (x : ℝ) (a : ℝ) :
  f a (π/4) = 0 → 
  f a x = 1/5 → 
  0 < x → 
  x < π → 
  Real.tan x = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_tan_x_value_l217_21782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l217_21707

/-- Calculates simple interest given principal, rate, and time -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_problem (principal : ℝ) (time : ℝ) (actual_rate : ℝ) (additional_interest : ℝ) 
  (supposed_rate : ℝ) :
  principal = 15000 →
  time = 2 →
  actual_rate = 15 →
  additional_interest = 900 →
  simple_interest principal actual_rate time - simple_interest principal supposed_rate time = additional_interest →
  supposed_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l217_21707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_max_min_S_l217_21726

/-- Represents a seat in the classroom --/
structure Seat where
  row : Nat
  col : Nat
  hrow : row ≥ 1 ∧ row ≤ 6
  hcol : col ≥ 1 ∧ col ≤ 8

/-- Calculates the position value for a student's move --/
def positionValue (start finish : Seat) : Int :=
  (start.row - finish.row) + (start.col - finish.col)

/-- The sum of position values for all students --/
def S (initial final : Fin 47 → Seat) : Int :=
  (Finset.sum (Finset.range 47) fun i => positionValue (initial i) (final i))

/-- The theorem stating the difference between max and min S --/
theorem difference_max_min_S :
  ∃ (initial₁ final₁ initial₂ final₂ : Fin 47 → Seat),
    (∀ (initial final : Fin 47 → Seat), S initial final ≤ S initial₁ final₁) ∧
    (∀ (initial final : Fin 47 → Seat), S initial final ≥ S initial₂ final₂) ∧
    S initial₁ final₁ - S initial₂ final₂ = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_max_min_S_l217_21726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_value_f_value_l217_21750

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = (Real.sqrt 3 / 2) * t.b ∧ t.B = t.C

-- Theorem 1: cos B = √3/4
theorem cos_B_value (t : Triangle) (h : triangle_conditions t) :
  Real.cos t.B = Real.sqrt 3 / 4 := by
  sorry

-- Define the function f(x)
noncomputable def f (B : ℝ) (x : ℝ) : ℝ :=
  Real.sin (2 * x + B)

-- Theorem 2: f(π/6) = (3 + √13)/8
theorem f_value (t : Triangle) (h : triangle_conditions t) :
  f t.B (Real.pi / 6) = (3 + Real.sqrt 13) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_value_f_value_l217_21750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_overlap_sum_l217_21761

/-- Represents a point in a 2D coordinate plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line in a 2D coordinate plane -/
structure Line where
  slope : ℚ
  yIntercept : ℚ

/-- Calculates the slope between two points -/
def slopeBetweenPoints (p1 p2 : Point) : ℚ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- Calculates the midpoint between two points -/
def midpointOf (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

/-- Determines if a point is on a given line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

/-- Determines if two points are symmetric about a given line -/
def areSymmetric (p1 p2 : Point) (l : Line) : Prop :=
  slopeBetweenPoints p1 p2 = -1 / l.slope ∧
  isOnLine (midpointOf p1 p2) l

/-- Main theorem: If A(0,2) overlaps with B(4,0), and C(7,3) overlaps with D(m,n) after folding, then m + n = 34/5 -/
theorem fold_overlap_sum :
  ∀ (m n : ℚ),
  let A : Point := { x := 0, y := 2 }
  let B : Point := { x := 4, y := 0 }
  let C : Point := { x := 7, y := 3 }
  let D : Point := { x := m, y := n }
  let foldLine : Line := { slope := 2, yIntercept := -3 }
  areSymmetric A B foldLine ∧ areSymmetric C D foldLine →
  m + n = 34/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_overlap_sum_l217_21761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_min_value_on_interval_min_value_cases_l217_21781

/-- The function f(x) = e^x / x -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

/-- Tangent line theorem -/
theorem tangent_line_at_one :
  ∀ x : ℝ, x > 0 → (deriv f) 1 = 0 ∧ f 1 = Real.exp 1 := by sorry

/-- Minimum value theorem -/
theorem min_value_on_interval (t : ℝ) (h : t > 0) :
  (∀ x ∈ Set.Icc t (t + 1), f t ≤ f x) ∨
  (∀ x ∈ Set.Icc t (t + 1), Real.exp 1 ≤ f x) := by sorry

/-- Specific cases of the minimum value theorem -/
theorem min_value_cases (t : ℝ) (h : t > 0) :
  (t ≥ 1 → ∀ x ∈ Set.Icc t (t + 1), f t ≤ f x) ∧
  (t < 1 → ∀ x ∈ Set.Icc t (t + 1), Real.exp 1 ≤ f x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_min_value_on_interval_min_value_cases_l217_21781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_problem_l217_21767

theorem lcm_gcd_problem (a b : ℕ) : 
  Nat.lcm a b = 2310 → Nat.gcd a b = 30 → a = 210 → b = 330 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_problem_l217_21767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_full_house_60_card_deck_l217_21748

/-- A deck of cards -/
structure Deck where
  total_cards : ℕ
  ranks : ℕ
  cards_per_rank : ℕ
  h_total : total_cards = ranks * cards_per_rank

/-- A full house in a 5-card poker hand -/
def is_full_house (hand : Finset ℕ) : Prop :=
  ∃ (r₁ r₂ : ℕ), r₁ ≠ r₂ ∧ 
    (hand.filter (λ c ↦ c / 4 = r₁)).card = 3 ∧
    (hand.filter (λ c ↦ c / 4 = r₂)).card = 2

/-- The probability of drawing a full house -/
def prob_full_house (d : Deck) : ℚ :=
  (d.ranks.choose 1 * d.cards_per_rank.choose 3 * (d.ranks - 1).choose 1 * d.cards_per_rank.choose 2) /
  d.total_cards.choose 5

/-- Theorem: The probability of drawing a full house from a 60-card deck 
    with 4 cards in each of 15 ranks is 35/38,011 -/
theorem prob_full_house_60_card_deck :
  let d : Deck := ⟨60, 15, 4, rfl⟩
  prob_full_house d = 35 / 38011 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_full_house_60_card_deck_l217_21748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_coeff_sum_l217_21785

open Real

variable (α : ℝ)
variable (m n p : ℤ)

-- Define the cosine equations
noncomputable def cos_2α : ℝ := 2 * (cos α)^2 - 1
noncomputable def cos_4α : ℝ := 8 * (cos α)^4 - 8 * (cos α)^2 + 1
noncomputable def cos_6α : ℝ := 32 * (cos α)^6 - 48 * (cos α)^4 + 18 * (cos α)^2 - 1
noncomputable def cos_8α : ℝ := 128 * (cos α)^8 - 256 * (cos α)^6 + 160 * (cos α)^4 - 32 * (cos α)^2 + 1
noncomputable def cos_10α : ℝ := m * (cos α)^10 - 1280 * (cos α)^8 + 1120 * (cos α)^6 + n * (cos α)^4 + p * (cos α)^2 - 1

-- State the theorem
theorem cos_coeff_sum : m - n + p = 962 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_coeff_sum_l217_21785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_five_solutions_l217_21788

noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

def equation (x : ℝ) : Prop :=
  6 * (frac x)^3 + (frac x)^2 + (frac x) + 2 * x = 2018

theorem equation_has_five_solutions :
  ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x ∈ s, equation x ∧
    ∀ y, equation y → y ∈ s :=
by
  sorry

#check equation_has_five_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_five_solutions_l217_21788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l217_21738

open Real

-- Define the triangle and point
variable (D E F Q : ℝ × ℝ)

-- Define vectors
def vector (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the condition
axiom vector_condition : vector Q D + 3 • (vector Q E) + 2 • (vector Q F) = (0, 0)

-- Define the area of a triangle
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ := 
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- State the theorem
theorem area_ratio (h : Q ≠ D ∧ Q ≠ F) : 
  triangle_area D E F / triangle_area D Q F = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l217_21738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_perp_to_same_line_are_parallel_skew_lines_parallel_planes_l217_21757

-- Define the necessary structures
structure Line
structure Plane

-- Define the relationships
axiom perpendicular : Plane → Line → Prop
axiom parallel : Plane → Plane → Prop
axiom parallel_line_plane : Line → Plane → Prop
axiom contained_in : Line → Plane → Prop
axiom skew : Line → Line → Prop

-- Theorem 1: Two planes perpendicular to the same line are parallel
theorem planes_perp_to_same_line_are_parallel 
  (p₁ p₂ : Plane) (l : Line) 
  (h₁ : perpendicular p₁ l) (h₂ : perpendicular p₂ l) : 
  parallel p₁ p₂ := by sorry

-- Theorem 2: If lines a and b are skew, with a contained in plane α, 
-- b parallel to plane α, b contained in plane β, and a parallel to plane β, 
-- then planes α and β are parallel
theorem skew_lines_parallel_planes 
  (a b : Line) (α β : Plane)
  (h₁ : skew a b)
  (h₂ : contained_in a α)
  (h₃ : parallel_line_plane b α)
  (h₄ : contained_in b β)
  (h₅ : parallel_line_plane a β) :
  parallel α β := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_perp_to_same_line_are_parallel_skew_lines_parallel_planes_l217_21757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_movement_probability_l217_21714

/-- Represents a cube with 8 vertices and 12 edges -/
structure Cube where
  vertices : Fin 8
  edges : Fin 12

/-- Represents a bug's movement on the cube -/
structure BugMovement (cube : Cube) where
  start_vertex : Fin 8
  num_moves : Nat
  move_probabilities : Fin 8 → Fin 3 → ℚ

/-- The probability of the bug visiting all vertices at least once in 9 moves -/
def probability_visit_all_vertices (cube : Cube) (bug : BugMovement cube) : ℚ :=
  sorry

/-- Main theorem: The probability of visiting all vertices in 9 moves is 4/243 -/
theorem bug_movement_probability (cube : Cube) (bug : BugMovement cube) :
  bug.num_moves = 9 →
  (∀ v : Fin 8, ∀ e : Fin 3, bug.move_probabilities v e = 1/3) →
  probability_visit_all_vertices cube bug = 4/243 :=
by
  sorry

#check bug_movement_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_movement_probability_l217_21714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_sum_alpha_beta_l217_21792

-- Part 1
theorem cos_alpha_value (α : ℝ) 
  (h1 : Real.cos (α + π/6) = 1/3) 
  (h2 : π/6 < α) 
  (h3 : α < π/2) : 
  Real.cos α = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 := by sorry

-- Part 2
theorem sum_alpha_beta (α β : ℝ)
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α = Real.sqrt 5 / 5)
  (h4 : Real.cos β = Real.sqrt 10 / 10) :
  α + β = 3*π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_sum_alpha_beta_l217_21792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_well_definedness_l217_21739

/-- Given n ∈ ℕ, prove that certain expressions are well-defined for all real x -/
theorem expression_well_definedness (n : ℕ) :
  (∀ x : ℝ, ∃ y : ℝ, y^4 = ((-4 : ℝ)^(2*n))) ∧ 
  (∀ x : ℝ, ∃ y : ℝ, y^5 = x^2) ∧ 
  (∀ x : ℝ, ∃ y : ℝ, y^5 = -x^2) ∧ 
  ¬(∀ x : ℝ, ∃ y : ℝ, y^4 = ((-4 : ℝ)^(2*n+1))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_well_definedness_l217_21739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_only_owners_l217_21770

theorem scooter_only_owners (total scooter_owners bike_owners : ℕ)
  (h1 : total = 400)
  (h2 : scooter_owners = 370)
  (h3 : bike_owners = 80)
  (h4 : ∀ a, a < total → (a < scooter_owners ∨ a < bike_owners)) :
  scooter_owners - (scooter_owners + bike_owners - total) = 320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_only_owners_l217_21770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solutions_l217_21701

theorem sqrt_equation_solutions : 
  {x : ℝ | (9*x - 8) ≥ 0 ∧ Real.sqrt (9*x - 8) + 18 / Real.sqrt (9*x - 8) = 8} = {17/9, 44/9} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solutions_l217_21701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_properties_l217_21766

noncomputable def f (x : ℝ) : ℝ := x^2 + 4*x + 3

noncomputable def g (x : ℝ) : ℝ := (3 - 2*x^2) / (1 + x^2)

theorem function_and_range_properties :
  (∀ x ≥ -1, f x = (Real.sqrt x + 1)^2 + 2*(Real.sqrt x + 1)) ∧
  (∀ x < -1, ¬ ∃ y, f y = x) ∧
  (∀ y ∈ Set.Ioo (-2) 3, ∃ x, g x = y) ∧
  (∀ y > 3, ¬ ∃ x, g x = y) ∧
  (∀ y < -2, ¬ ∃ x, g x = y) ∧
  (∃ x, g x = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_properties_l217_21766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_to_plane_are_parallel_perpendicular_planes_to_line_are_parallel_l217_21731

-- Define the types for lines and planes
structure Line where

structure Plane where

-- Define the relationships
def perpendicular (a b : Line ⊕ Plane) : Prop := sorry

def parallel (a b : Line ⊕ Plane) : Prop := sorry

-- Theorem for case 1: X and Y are lines, Z is a plane
theorem perpendicular_lines_to_plane_are_parallel
  (X Y : Line) (Z : Plane)
  (h1 : perpendicular (Sum.inl X) (Sum.inr Z))
  (h2 : perpendicular (Sum.inl Y) (Sum.inr Z)) :
  parallel (Sum.inl X) (Sum.inl Y) := by
  sorry

-- Theorem for case 2: Z is a line, X and Y are planes
theorem perpendicular_planes_to_line_are_parallel
  (X Y : Plane) (Z : Line)
  (h1 : perpendicular (Sum.inr X) (Sum.inl Z))
  (h2 : perpendicular (Sum.inr Y) (Sum.inl Z)) :
  parallel (Sum.inr X) (Sum.inr Y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_to_plane_are_parallel_perpendicular_planes_to_line_are_parallel_l217_21731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l217_21796

theorem sum_of_solutions : ∃ (S : Finset ℝ), (∀ x ∈ S, |x^2 - 12*x + 34| = 2) ∧ (S.sum id = 18) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l217_21796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_iff_a_in_range_l217_21772

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

theorem monotonic_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc 4 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_iff_a_in_range_l217_21772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l217_21729

-- Define the power function f
noncomputable def f (m k : ℤ) (x : ℝ) : ℝ := (m^2 - 2*m + 2) * x^(5*k - 2*k^2)

-- Define the conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on_positive (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x < f y

-- Main theorem
theorem power_function_properties (m k : ℤ) (a b : ℝ) :
  is_even (f m k) →
  is_increasing_on_positive (f m k) →
  a > 0 →
  b > 0 →
  a + b = 4 * (m : ℝ) →
  (∃ (g : ℝ → ℝ), (∀ x, f m k x = g x) ∧ (∀ x, g x = x^2)) ∧
  (∀ x, f m k (2*x - 1) < f m k (2 - x) → -1 < x ∧ x < 1) ∧
  (∃ min_val : ℝ, min_val = 3/2 ∧ 
    ∀ a b : ℝ, a > 0 → b > 0 → a + b = 4 * (m : ℝ) → 
      1/(a+1) + 4/(b+1) ≥ min_val) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l217_21729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_center_l217_21795

-- Define the large square
def large_square_side : ℝ := 6

-- Define the small squares
def small_square_side : ℝ := 2

-- Define points W, X, Y, Z on the small squares
structure CornerPoint where
  x : ℝ
  y : ℝ
  is_on_small_square : Bool

-- Define the square ABCD
structure SquareABCD where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the center point P of the large square
def center_P : ℝ × ℝ := (0, 0)

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_to_center 
  (W X Y Z : CornerPoint) 
  (ABCD : SquareABCD) : 
  ∃ (vertex : ℝ × ℝ), 
    (vertex = ABCD.A ∨ vertex = ABCD.B ∨ vertex = ABCD.C ∨ vertex = ABCD.D) ∧
    distance vertex center_P ≤ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_center_l217_21795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l217_21736

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => 3 * a n + 3^(n + 1) - 2^n

def b (n : ℕ) : ℚ := (a n - 2^n) / 3^n

def S (n : ℕ) : ℚ := ((2 * n - 3) * 3^(n + 1) + 2^(n + 3) + 7) / 4

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → b n = n - 1) ∧
  (∀ n : ℕ, n > 0 → a n = (n - 1) * 3^n + 2^n) ∧
  (∀ n : ℕ, n > 0 → (Finset.range n).sum (fun i => a (i + 1)) = S n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l217_21736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotonic_decrease_l217_21780

-- Define the power function
noncomputable def power_function (a : ℝ) : ℝ → ℝ := fun x ↦ x^a

-- Define the condition that the function passes through (2, 1/2)
def passes_through_point (f : ℝ → ℝ) : Prop := f 2 = 1/2

-- Define the intervals of monotonic decrease
def monotonic_decrease_intervals (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y ∧ x < 0 → f x > f y) ∧
  (∀ x y, 0 < x ∧ x < y → f x > f y)

-- Theorem statement
theorem power_function_monotonic_decrease :
  ∃ a : ℝ, passes_through_point (power_function a) →
    monotonic_decrease_intervals (power_function a) := by
  -- Proof goes here
  sorry

#check power_function_monotonic_decrease

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotonic_decrease_l217_21780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_eq_cos_l217_21774

open Real

/-- A sequence of functions defined by repeated differentiation of cosine -/
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => cos
  | n + 1 => deriv (f n)

/-- The 2016th function in the sequence is equal to cosine -/
theorem f_2016_eq_cos : f 2016 = cos := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_eq_cos_l217_21774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_games_played_l217_21787

/-- Represents the outcome of a single game -/
inductive GameOutcome
| A_wins
| B_wins

/-- Represents the state of the match -/
structure MatchState :=
  (score_A : ℕ)
  (score_B : ℕ)
  (games_played : ℕ)

/-- Defines when a match ends -/
def match_ends (state : MatchState) : Prop :=
  (state.score_A ≥ state.score_B + 2) ∨ 
  (state.score_B ≥ state.score_A + 2) ∨ 
  (state.games_played = 6)

/-- Probability of A winning a single game -/
noncomputable def prob_A_wins : ℝ := 2/3

/-- Probability of B winning a single game -/
noncomputable def prob_B_wins : ℝ := 1/3

/-- The random variable representing the number of games played -/
noncomputable def ξ : ℕ → ℝ := sorry

/-- Expected value of ξ -/
noncomputable def E_ξ : ℝ := sorry

theorem expected_games_played :
  E_ξ = 266/81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_games_played_l217_21787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_n_sum_l217_21779

/-- Convert a number from base n to decimal --/
def to_decimal (digits : List Nat) (n : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * n^i) 0

theorem base_n_sum (n d : Nat) : 
  (n > 0) → 
  (d < 10) → 
  (to_decimal [3, 2, d] n = 263) → 
  (to_decimal [3, 2, 4] n = to_decimal [1, 1, d, 1] 8) → 
  n + d = 12 := by
  sorry

#eval to_decimal [3, 2, 4] 10  -- Should output 324
#eval to_decimal [1, 1, 2, 1] 8  -- Should output 577

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_n_sum_l217_21779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_a_eq_one_l217_21724

/-- Two lines in the plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- The first line x + y = 0 -/
noncomputable def line1 : Line :=
  { slope := -1, intercept := 0 }

/-- The second line x - ay = 0 -/
noncomputable def line2 (a : ℝ) : Line :=
  { slope := 1/a, intercept := 0 }

/-- Theorem: The lines are perpendicular if and only if a = 1 -/
theorem perpendicular_iff_a_eq_one (a : ℝ) :
  perpendicular line1 (line2 a) ↔ a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_a_eq_one_l217_21724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l217_21710

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * ((n : ℝ) - 1)

noncomputable def S (n : ℕ) (b : ℕ → ℝ) : ℝ := (n : ℝ) * b n

noncomputable def a (n : ℕ) : ℝ := 4 * (n : ℝ) - 3

noncomputable def c (n : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ) : ℝ := 1 / (a n * (2 * b n + 3))

noncomputable def T (n : ℕ) (c : ℕ → ℝ) : ℝ := (n : ℝ) / (4 * (n : ℝ) + 1)

theorem sequence_properties (n : ℕ) :
  let b := arithmetic_sequence 1 2
  (∀ k, S k b = (k : ℝ) * b k) →
  (a n = 4 * (n : ℝ) - 3) ∧
  (T n (c · a b) = (n : ℝ) / (4 * (n : ℝ) + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l217_21710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_exam_theorem_l217_21759

/-- Represents a class in the ninth-grade school -/
structure SchoolClass where
  students : ℕ
  average_score : ℝ

/-- Represents a teacher in the school -/
inductive Teacher
| Wang
| Li

/-- The four classes in the school -/
def school_classes : Fin 4 → SchoolClass
| 0 => ⟨ 0, 68 ⟩
| 1 => ⟨ 0, 78 ⟩
| 2 => ⟨ 0, 74 ⟩
| 3 => ⟨ 0, 72 ⟩

/-- The teacher of each class -/
def class_teacher : Fin 4 → Teacher
| 0 => Teacher.Wang
| 1 => Teacher.Wang
| 2 => Teacher.Li
| 3 => Teacher.Li

/-- The average score of Teacher Li's classes -/
def li_average : ℝ := 73

theorem school_exam_theorem (a b c : ℕ) :
  li_average = 73 ∧
  ∃ (x : ℝ), x = (68 * a + 78 * b) / (a + b) ∧ (x = li_average ∨ x ≠ li_average) :=
by
  sorry

#check school_exam_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_exam_theorem_l217_21759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_from_tan_A_l217_21742

theorem cos_A_from_tan_A (A : ℝ) (h1 : A ∈ Set.Ioo 0 π) (h2 : Real.tan A = -2) : 
  Real.cos A = - Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_from_tan_A_l217_21742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_volume_l217_21740

/-- A pyramid with a rectangular base and specific lateral face angles -/
structure SpecialPyramid where
  baseArea : ℝ
  perpendicularFaces : Fin 2
  inclinedFaces : Fin 2
  inclinedAngles : Fin 2 → ℝ
  angleCond1 : inclinedAngles 0 = π / 6  -- 30°
  angleCond2 : inclinedAngles 1 = π / 3  -- 60°

/-- The volume of the special pyramid -/
noncomputable def pyramidVolume (p : SpecialPyramid) : ℝ := 
  p.baseArea * Real.sqrt p.baseArea / 3

/-- Theorem: The volume of the special pyramid is S√S / 3 -/
theorem special_pyramid_volume (p : SpecialPyramid) :
  pyramidVolume p = p.baseArea * Real.sqrt p.baseArea / 3 := by
  -- Unfold the definition of pyramidVolume
  unfold pyramidVolume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_volume_l217_21740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l217_21716

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := 1 - (a * Real.exp (x * Real.log 3)) / (Real.exp (x * Real.log 3) + 1)

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (2*b - 6) b → is_odd_function (f a b)) →
  (a = 2 ∧ b = 2) ∧
  (∀ x y, x ∈ Set.Ioo (-4) 2 → y ∈ Set.Ioo (-4) 2 → x < y → f 2 2 x > f 2 2 y) ∧
  (∀ m : ℝ, f 2 2 (m-2) + f 2 2 (2*m+1) > 0 → m ∈ Set.Ioo 0 (1/3)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l217_21716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_twelve_dividing_twenty_factorial_l217_21773

theorem largest_power_of_twelve_dividing_twenty_factorial :
  (∃ n : ℕ, 12^n ∣ Nat.factorial 20 ∧ ∀ m : ℕ, 12^m ∣ Nat.factorial 20 → m ≤ n) ∧
  (∀ k : ℕ, k > 8 → ¬(12^k ∣ Nat.factorial 20)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_twelve_dividing_twenty_factorial_l217_21773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_y_coordinate_is_40_div_3_l217_21771

-- Define the pentagon vertices
noncomputable def P : ℝ × ℝ := (0, 0)
noncomputable def R : ℝ × ℝ := (0, 5)
noncomputable def S : ℝ × ℝ := (6, 5)
noncomputable def T : ℝ × ℝ := (6, 0)
noncomputable def Q : ℝ → ℝ × ℝ := λ y => (3, y)

-- Define the area of a triangle
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((a.1 - c.1) * (b.2 - a.2) - (a.1 - b.1) * (c.2 - a.2))

-- Define the area of the pentagon
noncomputable def pentagonArea (qy : ℝ) : ℝ :=
  triangleArea P R (Q qy) + triangleArea R (Q qy) S + triangleArea P (Q qy) S + triangleArea P S T

-- Theorem statement
theorem q_y_coordinate_is_40_div_3 :
  ∃ qy : ℝ, pentagonArea qy = 50 ∧ qy = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_y_coordinate_is_40_div_3_l217_21771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_theorem_l217_21777

/-- Calculates the speed of a faster train given the conditions of two trains traveling in opposite directions. -/
noncomputable def faster_train_speed (slower_train_speed : ℝ) (passing_time : ℝ) (faster_train_length : ℝ) : ℝ :=
  let slower_speed_ms := slower_train_speed * 1000 / 3600
  let relative_speed := faster_train_length / passing_time
  let faster_speed_ms := relative_speed - slower_speed_ms
  faster_speed_ms * 3600 / 1000

/-- Theorem stating the speed of the faster train under given conditions. -/
theorem faster_train_speed_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ |faster_train_speed 36 10 225.018 - 45.00648| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_theorem_l217_21777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sequence_trig_l217_21700

theorem triangle_geometric_sequence_trig (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- a, b, c are side lengths corresponding to angles A, B, C
  0 < a ∧ 0 < b ∧ 0 < c →
  -- a, b, c form a geometric sequence
  b^2 = a * c →
  -- cos B = 4/5
  Real.cos B = 4/5 →
  -- Theorem to prove
  1 / Real.tan A + 1 / Real.tan C = 5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sequence_trig_l217_21700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l217_21704

theorem inequality_proof (m n : ℕ) (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  let p := m + n
  0 ≤ x^(m+n) + y^(m+n) + z^(m+n) - x^m * y^n - y^m * z^n - z^m * x^n ∧ 
  x^(m+n) + y^(m+n) + z^(m+n) - x^m * y^n - y^m * z^n - z^m * x^n ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l217_21704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_midpoint_area_ratio_l217_21721

/-- Represents a trapezoid with bases of length a and b -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  h_pos : h > 0
  a_gt_b : a > b

/-- The area of the trapezoid -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ := (t.a + t.b) * t.h / 2

/-- The area of the quadrilateral formed by joining the midpoints -/
noncomputable def midpoint_quad_area (t : Trapezoid) : ℝ := (t.a - t.b) * t.h / 4

/-- The theorem stating the relationship between the areas and the base ratio -/
theorem trapezoid_midpoint_area_ratio (t : Trapezoid) : 
  midpoint_quad_area t = trapezoid_area t / 4 → t.a / t.b = 3 := by
  intro h
  -- The proof goes here
  sorry

#check trapezoid_midpoint_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_midpoint_area_ratio_l217_21721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_function_iff_l217_21758

/-- A function is beautiful if it's monotonic and has a specific range property. -/
def is_beautiful_function (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  (Monotone f) ∧
  (∃ a b : ℝ, (Set.Icc (a/2) (b/2) ⊆ D) ∧ 
    (Set.image f (Set.Icc (a/2) (b/2)) = Set.Icc a b))

/-- The logarithmic function we're considering. -/
noncomputable def f (c t : ℝ) (x : ℝ) : ℝ := Real.log (c^x - t) / Real.log c

/-- The main theorem stating when f is a beautiful function. -/
theorem beautiful_function_iff (c : ℝ) (hc : c > 0 ∧ c ≠ 1) :
  ∀ t : ℝ, (∃ D : Set ℝ, is_beautiful_function (f c t) D) ↔ 0 < t ∧ t < 1/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_function_iff_l217_21758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l217_21756

noncomputable def initial_number : ℂ := -4 - 6 * Complex.I

noncomputable def rotation_angle : ℝ := Real.pi / 3  -- 60 degrees in radians

noncomputable def dilation_factor : ℝ := Real.sqrt 2

noncomputable def translation_vector : ℂ := 2 + 2 * Complex.I

noncomputable def transformed_number : ℂ := 
  (initial_number * Complex.exp (Complex.I * rotation_angle) * dilation_factor) + translation_vector

theorem complex_transformation :
  transformed_number = (-5 * Real.sqrt 2 + 2) + (Real.sqrt 6 + 2) * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l217_21756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_honor_students_count_l217_21706

/-- Represents a class of students -/
structure ClassInfo where
  girls : ℕ
  boys : ℕ
  honorGirls : ℕ
  honorBoys : ℕ

/-- The conditions of the problem -/
def validClass (c : ClassInfo) : Prop :=
  c.girls + c.boys < 30 ∧
  c.girls > 0 ∧
  c.boys > 0 ∧
  c.honorGirls * c.girls = 3 * c.girls / 13 ∧
  c.honorBoys * c.boys = 4 * c.boys / 11

/-- The theorem to be proved -/
theorem honor_students_count (c : ClassInfo) (h : validClass c) : 
  c.honorGirls + c.honorBoys = 7 := by
  sorry

#check honor_students_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_honor_students_count_l217_21706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_most_frequent_l217_21753

-- Define the set of numbers
def numbers : Finset ℕ := Finset.range 10 |>.filter (λ n => n ≥ 1)

-- Define a function to calculate the units digit of a sum
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to count occurrences of a specific units digit
def countOccurrences (digit : ℕ) : ℕ :=
  Finset.card (Finset.product numbers numbers |>.filter 
    (λ (i, j) => unitsDigit (i + j) = digit))

-- Theorem statement
theorem zero_most_frequent :
  ∀ d : ℕ, d ≠ 0 → countOccurrences 0 > countOccurrences d := by
  sorry

#eval countOccurrences 0  -- Expected: 20
#eval countOccurrences 1  -- Expected: ~18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_most_frequent_l217_21753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_function_theorem_l217_21749

-- Define the exponential function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exp_function_theorem (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-1) = 1/2 → f a (f a 2) = 16 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_function_theorem_l217_21749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_2order_repeatable_l217_21733

/-- A binary sequence is a finite sequence of 0s and 1s -/
def BinarySequence (n : ℕ) := Fin n → Fin 2

/-- A sequence is k-order repeatable if there exist two distinct indices i and j such that
    the k consecutive terms starting from i match those starting from j -/
def IsKOrderRepeatable (k : ℕ) {n : ℕ} (seq : BinarySequence n) : Prop :=
  ∃ i j, i ≠ j ∧ i + k ≤ n ∧ j + k ≤ n ∧
    ∀ t, t < k → seq ⟨i + t, by sorry⟩ = seq ⟨j + t, by sorry⟩

/-- The minimum length of a binary sequence that guarantees it is 2-order repeatable is 5 -/
theorem min_length_2order_repeatable :
  (∀ n, n ≥ 5 → ∀ seq : BinarySequence n, IsKOrderRepeatable 2 seq) ∧
  (∃ m, m = 4 ∧ ∃ seq : BinarySequence m, ¬IsKOrderRepeatable 2 seq) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_2order_repeatable_l217_21733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_ratio_l217_21745

/-- The ratio of areas between an inscribed square and its encompassing square --/
theorem inscribed_square_area_ratio : 
  ∀ (large_square_side : ℝ), large_square_side > 0 →
  (large_square_side / 2)^2 / large_square_side^2 = 1/4 := by
  intro large_square_side h_positive
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_ratio_l217_21745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_movable_disk_l217_21764

/-- Represents a circular disk on a 2D plane --/
structure Disk where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Checks if two disks overlap --/
def disks_overlap (d1 d2 : Disk) : Prop :=
  let (x1, y1) := d1.center
  let (x2, y2) := d2.center
  (x1 - x2)^2 + (y1 - y2)^2 < (d1.radius + d2.radius)^2

/-- Theorem: In a set of non-overlapping disks, there exists a disk that can be moved upward --/
theorem exists_movable_disk (disks : Set Disk) :
  (∀ d1 d2 : Disk, d1 ∈ disks → d2 ∈ disks → d1 ≠ d2 → ¬disks_overlap d1 d2) →
  ∃ d ∈ disks, ∀ d' ∈ disks, d ≠ d' →
    let (x, y) := d.center
    let (x', y') := d'.center
    y > y' ∨ (y = y' ∧ x ≠ x') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_movable_disk_l217_21764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_dividing_special_sequence_l217_21791

theorem infinitely_many_primes_dividing_special_sequence : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p ∣ (2014^(2^n) + 2014)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_dividing_special_sequence_l217_21791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_minus_3pi_4_l217_21763

theorem cos_2alpha_minus_3pi_4 (α : ℝ) 
  (h1 : Real.sin (α + π / 2) = -Real.sqrt 5 / 5) 
  (h2 : α ∈ Set.Ioo 0 π) : 
  Real.cos (2 * α - 3 * π / 4) = -Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_minus_3pi_4_l217_21763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_roots_l217_21776

theorem simplify_roots (x : ℝ) (hx : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (98 * x) * (250 * x^2) ^ (1/3) =
  525 * x^2 * Real.sqrt (8 * x) * (2 * x) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_roots_l217_21776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_jogger_time_l217_21718

/-- Calculates the time it takes for a train to pass a jogger given their speeds and initial positions. -/
noncomputable def train_passing_time (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) : ℝ :=
  let relative_speed := train_speed - jogger_speed
  let total_distance := initial_distance + train_length
  total_distance / (relative_speed * 1000 / 3600)

/-- Theorem stating that under given conditions, the time for a train to pass a jogger is approximately 35.33 seconds. -/
theorem train_passes_jogger_time :
  let jogger_speed := (6 : ℝ) -- km/hr
  let train_speed := (60 : ℝ) -- km/hr
  let train_length := (250 : ℝ) -- meters
  let initial_distance := (280 : ℝ) -- meters
  let passing_time := train_passing_time jogger_speed train_speed train_length initial_distance
  abs (passing_time - 35.33) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_jogger_time_l217_21718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_by_eight_division_l217_21786

/-- Represents a rectangular part of the grid -/
structure GridPart where
  width : Nat
  height : Nat

/-- Calculates the area of a GridPart -/
def area (p : GridPart) : Nat :=
  p.width * p.height

/-- Calculates the perimeter of a GridPart -/
def perimeter (p : GridPart) : Nat :=
  2 * (p.width + p.height)

/-- Represents the division of the 8x8 grid -/
def gridDivision : List GridPart :=
  [⟨4, 4⟩, ⟨4, 4⟩, ⟨8, 2⟩, ⟨8, 2⟩]

theorem eight_by_eight_division :
  /- The grid division has exactly 4 parts -/
  (gridDivision.length = 4) ∧
  /- All parts fit within an 8x8 grid -/
  (∀ p, p ∈ gridDivision → p.width ≤ 8 ∧ p.height ≤ 8) ∧
  /- The total area of all parts is 64 (8x8) -/
  ((gridDivision.map area).sum = 64) ∧
  /- All parts have equal area -/
  (∀ p q, p ∈ gridDivision → q ∈ gridDivision → area p = area q) ∧
  /- All parts have pairwise different perimeters -/
  (∀ p q, p ∈ gridDivision → q ∈ gridDivision → p ≠ q → perimeter p ≠ perimeter q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_by_eight_division_l217_21786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_equilateral_triangle_l217_21769

/-- An equilateral triangle with side length 4 -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧ 
                (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2
  side_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16

/-- A point inside the triangle -/
def InsidePoint (t : EquilateralTriangle) := 
  { P : ℝ × ℝ // P.1 ∈ Set.Icc (min t.A.1 (min t.B.1 t.C.1)) (max t.A.1 (max t.B.1 t.C.1)) ∧
              P.2 ∈ Set.Icc (min t.A.2 (min t.B.2 t.C.2)) (max t.A.2 (max t.B.2 t.C.2)) }

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The vector from one point to another -/
def vector (a b : ℝ × ℝ) : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)

theorem min_dot_product_equilateral_triangle (t : EquilateralTriangle) :
  ∃ (min : ℝ), min = -6 ∧ ∀ (P : InsidePoint t),
    dot_product (vector P.val t.A) (vector P.val t.B + vector P.val t.C) ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_equilateral_triangle_l217_21769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_to_chord_distance_l217_21783

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  passesThrough : (ℝ × ℝ) → Prop

-- Define the line equation
def LineEquation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the distance from a point to a line
noncomputable def distancePointToLine (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * p.1 + b * p.2 + c| / Real.sqrt (a^2 + b^2)

theorem circle_center_to_chord_distance :
  ∀ (C : Circle),
    C.passesThrough (0, 0) →
    C.passesThrough (4, 2) →
    LineEquation 1 2 (-1) C.center.1 C.center.2 →
    distancePointToLine C.center 1 (-1/2) 0 = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_to_chord_distance_l217_21783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_and_chord_length_l217_21762

/-- Given a sector with arc length 6 and radius 3, prove its area and chord length. -/
theorem sector_area_and_chord_length :
  ∃ (arc_length radius central_angle sector_area chord_length : ℝ),
    arc_length = 6 ∧
    radius = 3 ∧
    central_angle = arc_length / radius ∧
    sector_area = (1/2) * radius^2 * central_angle ∧
    chord_length = 2 * radius * Real.sin (central_angle / 2) ∧
    sector_area = 9 ∧
    chord_length = 6 * Real.sin 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_and_chord_length_l217_21762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l217_21793

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 3 * x^2)

theorem range_sum (a b : ℝ) :
  (∀ y, y ∈ Set.Ioo a b ↔ ∃ x, f x = y) →
  (∀ y, y ≤ b → ∃ x, f x = y) →
  (∀ y, y > b → ∀ x, f x ≠ y) →
  a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l217_21793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l217_21711

/-- The function f(x) = (x^2 + 1) / x -/
noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x

/-- Condition p: x^2 - 4x + 3 ≤ 0 -/
def p (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

/-- Condition q: f(x) has both a maximum and a minimum value -/
def q : Prop := ∃ (max min : ℝ), ∀ x : ℝ, x ≠ 0 → f x ≤ max ∧ min ≤ f x

/-- Theorem: p is sufficient but not necessary for q -/
theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q) ∧ ¬(∀ x : ℝ, q → p x) := by
  sorry

#check p_sufficient_not_necessary_for_q

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l217_21711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_fraction_implies_rate_l217_21732

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_fraction_implies_rate
  (principal : ℝ)
  (time : ℝ)
  (rate : ℝ)
  (h_time : time = 10)
  (h_fraction : simple_interest principal rate time = (1/6) * (principal + simple_interest principal rate time)) :
  rate = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_fraction_implies_rate_l217_21732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pole_height_is_6_8_l217_21737

noncomputable section

-- Define the pole and cable setup
def cable_ground_distance : ℝ := 4
def person_distance_from_base : ℝ := 3
def person_height : ℝ := 1.7

-- Define the function to calculate pole height
def calculate_pole_height (cable_ground_distance person_distance_from_base person_height : ℝ) : ℝ :=
  let remaining_distance := cable_ground_distance - person_distance_from_base
  cable_ground_distance * (person_height / remaining_distance)

-- Theorem statement
theorem pole_height_is_6_8 :
  calculate_pole_height cable_ground_distance person_distance_from_base person_height = 6.8 := by
  -- Unfold the definition of calculate_pole_height
  unfold calculate_pole_height
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pole_height_is_6_8_l217_21737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_1_to_30_l217_21727

def is_prime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else (List.range (n - 1)).all (fun m => m + 2 = n ∨ n % (m + 2) ≠ 0)

def count_primes (n : ℕ) : ℕ :=
  (List.range n).filter (fun m => is_prime (m + 1)) |>.length

theorem probability_prime_1_to_30 : 
  (count_primes 30 : ℚ) / 30 = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_1_to_30_l217_21727

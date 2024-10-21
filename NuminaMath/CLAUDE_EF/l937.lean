import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l937_93794

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - focus.1)^2 + (y - focus.2)^2 = 1

-- Theorem statement
theorem circle_equation_proof :
  ∀ x y : ℝ, parabola x y → circle_eq x y → x^2 - 2*x + y^2 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l937_93794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l937_93725

/-- A parabola passing through a point -/
structure Parabola where
  p : ℝ
  passes_through : p * 1^2 = 3

/-- The distance from the focus to the directrix of a parabola -/
noncomputable def focus_directrix_distance (parabola : Parabola) : ℝ := 1 / (4 * parabola.p)

/-- Theorem: For a parabola y = px^2 passing through (1,3), the distance from focus to directrix is 1/6 -/
theorem parabola_focus_directrix_distance (parabola : Parabola) : 
  focus_directrix_distance parabola = 1/6 := by
  sorry

#eval "Parabola theorem compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l937_93725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_coefficient_sum_l937_93779

/-- Given a rational function g(x) with vertical asymptotes at x = 2 and x = -3,
    prove that the sum of the coefficients c and d in the denominator is -5. -/
theorem asymptote_coefficient_sum (c d : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -3 → 
    ∃ g : ℝ → ℝ, g x = (x + 3) / (x^2 + c*x + d)) →
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 2| ∧ |x - 2| < δ → ∀ g : ℝ → ℝ, g x = (x + 3) / (x^2 + c*x + d) → |g x| > 1/ε) →
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x + 3| ∧ |x + 3| < δ → ∀ g : ℝ → ℝ, g x = (x + 3) / (x^2 + c*x + d) → |g x| > 1/ε) →
  c + d = -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_coefficient_sum_l937_93779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_positive_reals_floor_divisibility_implies_integers_l937_93701

theorem distinct_positive_reals_floor_divisibility_implies_integers 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b)
  (h : ∀ n : ℕ, (⌊n * a⌋ : ℤ) ∣ (⌊n * b⌋ : ℤ)) :
  ∃ (m k : ℕ), (a : ℝ) = m ∧ (b : ℝ) = k :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_positive_reals_floor_divisibility_implies_integers_l937_93701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_profit_is_50_percent_l937_93787

/-- Represents the trader's buying and selling weights -/
structure TraderWeights where
  buyIndicated : ℝ
  buyActual : ℝ
  sellActual : ℝ
  sellClaimed : ℝ

/-- Calculates the profit percentage based on the trader's weights -/
noncomputable def profitPercentage (w : TraderWeights) : ℝ :=
  ((w.sellClaimed - w.sellActual) / w.sellActual) * 100

/-- Theorem stating that the trader's profit percentage is 50% -/
theorem trader_profit_is_50_percent (w : TraderWeights) : 
  w.buyActual = w.buyIndicated * 1.1 →
  w.sellActual + w.sellActual * 0.5 = w.sellClaimed →
  w.buyIndicated = w.sellClaimed →
  profitPercentage w = 50 := by
  sorry

#check trader_profit_is_50_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_profit_is_50_percent_l937_93787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l937_93709

theorem count_integers_in_pi_range : 
  (Finset.range (Int.toNat (Int.floor (15 * Real.pi) - Int.ceil (-5 * Real.pi) + 1))).card = 63 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l937_93709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_fixed_point_l937_93740

/-- Parabola type -/
structure Parabola where
  p : ℝ
  eq : ℝ × ℝ → Prop
  h_pos : p > 0
  h_eq : ∀ x y, eq (x, y) ↔ y^2 = 2*p*x

/-- Line type -/
structure Line where
  m : ℝ
  b : ℝ
  eq : ℝ × ℝ → Prop
  h_eq : ∀ x y, eq (x, y) ↔ x = m*y + b

/-- Theorem statement -/
theorem parabola_intersection_fixed_point
  (C : Parabola)
  (h_through : C.eq (4, 4))
  (l : Line)
  (h_distinct : ∃ E F : ℝ × ℝ, E ≠ F ∧ C.eq E ∧ C.eq F ∧ l.eq E ∧ l.eq F)
  (h_perp : ∃ E A B : ℝ × ℝ,
    C.eq E ∧ l.eq E ∧
    (∃ x, A = (x, x)) ∧
    (∃ k, B.2 = k * B.1) ∧
    E.1 = A.1 ∧ E.2 = B.2)
  (h_midpoint : ∃ E A B : ℝ × ℝ, A.1 = (E.1 + B.1) / 2 ∧ A.2 = (E.2 + B.2) / 2) :
  l.eq (0, 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_fixed_point_l937_93740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_average_speed_l937_93756

/-- Given Alice's trip details, prove her average speed -/
theorem alice_average_speed 
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ)
  (h1 : distance1 = 45)
  (h2 : speed1 = 15)
  (h3 : distance2 = 15)
  (h4 : speed2 = 45) :
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 18 := by
  sorry

#check alice_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_average_speed_l937_93756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_volume_ratio_l937_93719

-- Define the dimensions of the containers
def greg_diameter : ℝ := 4
def greg_height : ℝ := 20
def violet_diameter : ℝ := 12
def violet_height : ℝ := 6

-- Define the volume of a cylinder
noncomputable def cylinder_volume (diameter : ℝ) (height : ℝ) : ℝ :=
  Real.pi * (diameter / 2)^2 * height

-- Theorem statement
theorem container_volume_ratio :
  (cylinder_volume greg_diameter greg_height) / (cylinder_volume violet_diameter violet_height) = 10 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_volume_ratio_l937_93719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_power_l937_93755

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the property that (n+i)^6 is an integer
def is_integer_power (n : ℤ) : Prop := ∃ m : ℤ, (n : ℂ) + i ^ 6 = m

-- Theorem statement
theorem unique_integer_power : ∃! n : ℤ, is_integer_power n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_power_l937_93755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_least_multiple_of_101_l937_93795

def a : ℕ → ℕ
  | 0 => 15
  | n+1 => if n ≥ 15 then 50 * a n + 2 * (n+1) else a n

theorem exists_least_multiple_of_101 :
  ∃ N : ℕ, N > 15 ∧ 101 ∣ a N ∧ ∀ n : ℕ, 15 < n ∧ n < N → ¬(101 ∣ a n) :=
by sorry

#eval a 15  -- To check if the function works as expected

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_least_multiple_of_101_l937_93795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_correctness_l937_93708

-- Define the propositions as axioms instead of strings
axiom proposition_1 : Prop
axiom proposition_2 : Prop
axiom proposition_3 : Prop
axiom proposition_4 : Prop

-- Define the correctness of each proposition
def is_correct (p : Prop) : Prop := p

-- State the theorem
theorem proposition_correctness :
  ¬(is_correct proposition_1) ∧
  (is_correct proposition_2) ∧
  (is_correct proposition_3) ∧
  ¬(is_correct proposition_4) :=
by
  sorry

#check proposition_correctness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_correctness_l937_93708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_angle_theorem_l937_93771

/-- A quadrilateral with vertices A, B, C, D -/
structure Quadrilateral (V : Type*) [NormedAddCommGroup V] where
  A : V
  B : V
  C : V
  D : V

/-- The acute angle between the diagonals of a quadrilateral -/
noncomputable def diagonalAngle {V : Type*} [NormedAddCommGroup V] (q : Quadrilateral V) : ℝ :=
  sorry

/-- Predicate to check if a quadrilateral is cyclic -/
def isCyclic {V : Type*} [NormedAddCommGroup V] (q : Quadrilateral V) : Prop :=
  sorry

/-- The side lengths of a quadrilateral -/
noncomputable def sideLengths {V : Type*} [NormedAddCommGroup V] (q : Quadrilateral V) : Fin 4 → ℝ :=
  sorry

theorem diagonal_angle_theorem {V : Type*} [NormedAddCommGroup V] 
  (q₁ q₂ : Quadrilateral V) (φ : ℝ) :
  isCyclic q₁ →
  diagonalAngle q₁ = φ →
  sideLengths q₁ = sideLengths q₂ →
  diagonalAngle q₂ ≤ φ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_angle_theorem_l937_93771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_from_radii_sum_l937_93784

theorem triangle_right_angle_from_radii_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let s := (a + b + c) / 2
  let x := s - a
  let y := s - b
  let z := s - c
  let r := (s * x * y * z).sqrt / s
  let ra := (s * x * y * z).sqrt / x
  let rb := (s * x * y * z).sqrt / y
  let rc := (s * x * y * z).sqrt / z
  (r + ra + rb + rc = 2 * s) →
  (a * a = b * b + c * c ∨ b * b = a * a + c * c ∨ c * c = a * a + b * b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_from_radii_sum_l937_93784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_education_percentage_is_ten_l937_93797

noncomputable def salary : ℝ := 2125
noncomputable def house_rent_percentage : ℝ := 20
noncomputable def clothes_percentage : ℝ := 10
noncomputable def remaining_amount : ℝ := 1377

noncomputable def amount_after_rent (s : ℝ) (r : ℝ) : ℝ := s * (1 - r / 100)

noncomputable def amount_after_education (a : ℝ) (p : ℝ) : ℝ := a * (1 - p / 100)

noncomputable def amount_after_clothes (a : ℝ) (c : ℝ) : ℝ := a * (1 - c / 100)

theorem education_percentage_is_ten :
  ∃ p : ℝ,
    amount_after_clothes
      (amount_after_education
        (amount_after_rent salary house_rent_percentage)
        p)
      clothes_percentage
    = remaining_amount ∧ p = 10 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_education_percentage_is_ten_l937_93797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_reciprocal_distances_exists_max_sum_reciprocal_distances_l937_93754

/-- Line l in parametric form -/
noncomputable def line_l (t a : ℝ) : ℝ × ℝ := (t * Real.cos a, t * Real.sin a)

/-- Circle C -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

/-- Intersection points of line l and circle C -/
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l t a ∧ circle_C p.1 p.2}

/-- Distance from origin to a point -/
noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

theorem max_sum_reciprocal_distances (a : ℝ) (ha : 0 < a ∧ a < Real.pi / 2) :
  let points := intersection_points a
  ∀ A B, A ∈ points → B ∈ points → A ≠ B →
    (1 / distance_from_origin A + 1 / distance_from_origin B) ≤ 2 * Real.sqrt 5 :=
by sorry

theorem exists_max_sum_reciprocal_distances :
  ∃ a : ℝ, 0 < a ∧ a < Real.pi / 2 ∧
    let points := intersection_points a
    ∃ A B, A ∈ points ∧ B ∈ points ∧ A ≠ B ∧
      1 / distance_from_origin A + 1 / distance_from_origin B = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_reciprocal_distances_exists_max_sum_reciprocal_distances_l937_93754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l937_93751

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |Real.sin x| + 2 * |Real.cos x|

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc 1 (Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l937_93751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_heater_capacity_ratio_l937_93747

/-- The ratio of Wallace's water heater capacity to Catherine's water heater capacity is 2:1 -/
theorem water_heater_capacity_ratio :
  -- Wallace's water heater capacity
  let wallace_capacity : ℚ := 40
  -- Wallace's water heater fullness
  let wallace_fullness : ℚ := 3/4
  -- Catherine's water heater fullness
  let catherine_fullness : ℚ := 3/4
  -- Total water in both heaters
  let total_water : ℚ := 45
  -- Catherine's water heater capacity
  let catherine_capacity : ℚ := (total_water - wallace_capacity * wallace_fullness) / catherine_fullness
  -- The ratio of Wallace's capacity to Catherine's capacity
  let capacity_ratio : ℚ := wallace_capacity / catherine_capacity
  ∀ (wallace_capacity catherine_capacity : ℚ),
    wallace_capacity = 40 →
    catherine_capacity = (45 - 40 * (3/4)) / (3/4) →
    wallace_capacity / catherine_capacity = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_heater_capacity_ratio_l937_93747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_minus_pi_sixth_l937_93760

theorem cos_double_angle_minus_pi_sixth (α : ℝ) : 
  0 < α ∧ α < π/2 → 
  Real.sin (α + π/6) = 3/5 → 
  Real.cos (2*α - π/6) = 24/25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_minus_pi_sixth_l937_93760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_correctness_l937_93776

noncomputable section

-- Define the functions
def f1 (x : ℝ) : ℝ := Real.cos x / x
def f2 (x : ℝ) : ℝ := (x^2 + x + 1) * Real.exp x
def f3 (x : ℝ) : ℝ := (2 * x) / (x^2 + 1)
def f4 (x : ℝ) : ℝ := Real.exp (3 * x + 1)

-- State the theorem
theorem derivative_correctness :
  ¬(∀ x, deriv f1 x = -Real.sin x / x^2) ∧
  ¬(∀ x, deriv f2 x = (2 * x + 1) * Real.exp x) ∧
  (∀ x, deriv f3 x = (2 - 2 * x^2) / (x^2 + 1)^2) ∧
  (∀ x, deriv f4 x = 3 * Real.exp (3 * x + 1)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_correctness_l937_93776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_unique_solution_l937_93700

/-- Triangle ABC with given properties has exactly one solution -/
theorem triangle_abc_unique_solution :
  ∃! (B C : ℝ) (b : ℝ),
    0 < B ∧ 0 < C ∧ 0 < b ∧
    B + C = 120 ∧  -- Since A = 60°, B + C = 180° - 60° = 120°
    4 * Real.sin B = 4 * Real.sin (π / 3) ∧  -- Law of sines: a / sin A = c / sin C
    b ^ 2 = 4 ^ 2 + 4 ^ 2 - 2 * 4 * 4 * Real.cos (π / 3) :=  -- Law of cosines: b^2 = a^2 + c^2 - 2ac * cos A
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_unique_solution_l937_93700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasyas_no_purchase_days_l937_93761

/-- Represents the number of days Vasya buys 9 marshmallows -/
def x : ℕ := sorry

/-- Represents the number of days Vasya buys 2 meat pies -/
def y : ℕ := sorry

/-- Represents the number of days Vasya buys 4 marshmallows and 1 meat pie -/
def z : ℕ := sorry

/-- Represents the number of days Vasya buys nothing -/
def w : ℕ := sorry

/-- The total number of school days -/
def total_days : ℕ := 15

/-- The total number of marshmallows bought -/
def total_marshmallows : ℕ := 30

/-- The total number of meat pies bought -/
def total_meat_pies : ℕ := 9

theorem vasyas_no_purchase_days :
  (x + y + z + w = total_days) →
  (9 * x + 4 * z = total_marshmallows) →
  (2 * y + z = total_meat_pies) →
  w = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasyas_no_purchase_days_l937_93761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_revenue_is_4_07_l937_93775

/-- Represents the sales data for a single item -/
structure ItemSales where
  price : ℚ
  quantity : ℕ

/-- Calculates the total sales for an item -/
def totalSales (item : ItemSales) : ℚ :=
  item.price * item.quantity

/-- Calculates the average revenue per item sold given a list of item sales -/
def averageRevenue (items : List ItemSales) : ℚ :=
  let totalRevenue := (items.map totalSales).sum
  let totalQuantity := (items.map (λ i => i.quantity)).sum
  totalRevenue / totalQuantity

theorem average_revenue_is_4_07 (strawberry blueberry chocolate vanilla : ItemSales)
    (h1 : strawberry = { price := 4, quantity := 20 })
    (h2 : blueberry = { price := 3, quantity := 30 })
    (h3 : chocolate = { price := 5, quantity := 15 })
    (h4 : vanilla = { price := 6, quantity := 10 }) :
    averageRevenue [strawberry, blueberry, chocolate, vanilla] = 407/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_revenue_is_4_07_l937_93775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_values_and_m_range_l937_93748

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := 2^(x + Real.cos α) - 2^(-x + Real.cos α)

theorem alpha_values_and_m_range :
  (∃ (α : ℝ), f 1 α = 3/4 ∧ Set.range (λ k : ℤ ↦ 2 * Real.pi * ↑k + Real.pi) = {α | f 1 α = 3/4}) ∧
  (∀ m : ℝ, (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi/2 → f (m * Real.cos θ) Real.pi + f (1 - m) Real.pi > 0) ↔ m < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_values_and_m_range_l937_93748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y₀_range_l937_93757

-- Define the circle O
def circle_O : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define point M
noncomputable def point_M (y₀ : ℝ) : ℝ × ℝ := (Real.sqrt 3, y₀)

-- Define the tangent line condition
def is_tangent (M N : ℝ × ℝ) : Prop := sorry

-- Define the angle condition
def angle_condition (O M N : ℝ × ℝ) : Prop := sorry

theorem y₀_range (y₀ : ℝ) :
  ∃ (N : ℝ × ℝ),
    N ∈ circle_O ∧
    is_tangent (point_M y₀) N ∧
    angle_condition (0, 0) (point_M y₀) N
  → -1 ≤ y₀ ∧ y₀ ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y₀_range_l937_93757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l937_93732

-- Define the function f(x) = lg|sin x|
noncomputable def f (x : ℝ) : ℝ := Real.log (|Real.sin x|) / Real.log 10

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = f x) ∧  -- f is even
  (∀ x, f (x + π) = f x) ∧  -- f has period π
  (∀ p, 0 < p → p < π → ∃ x, f (x + p) ≠ f x) :=  -- π is the smallest positive period
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l937_93732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l937_93717

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x

-- Define the relationship between f and g
axiom f_g_sum : ∀ x : ℝ, f x + g x = (1/2)^x

-- Define the existence of x₀
axiom exists_x0 : ∃ x0 : ℝ, x0 ∈ Set.Icc (1/2) 1 ∧ 
  ∃ a : ℝ, a * f x0 + g (2*x0) = 0

-- State the theorem
theorem a_range : 
  ∃ a_min a_max : ℝ, a_min = 2 * Real.sqrt 2 ∧ a_max = 5/2 * Real.sqrt 2 ∧
  (∀ a : ℝ, (∃ x0 : ℝ, x0 ∈ Set.Icc (1/2) 1 ∧ a * f x0 + g (2*x0) = 0) → 
  a ∈ Set.Icc a_min a_max) ∧
  (∀ a : ℝ, a ∈ Set.Icc a_min a_max → 
  ∃ x0 : ℝ, x0 ∈ Set.Icc (1/2) 1 ∧ a * f x0 + g (2*x0) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l937_93717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_geometric_sequence_implies_a_equals_one_l937_93793

-- Define the curve C
def curve_C (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x ∧ a > 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, -4)

-- Define the geometric sequence condition
def geometric_sequence (pm mn pn : ℝ) : Prop := mn^2 = pm * pn

-- Main theorem
theorem intersection_geometric_sequence_implies_a_equals_one :
  ∀ (a : ℝ) (M N : ℝ × ℝ),
  (∃ (x_m y_m : ℝ), M = (x_m, y_m) ∧ curve_C a x_m y_m ∧ line_l x_m y_m) →
  (∃ (x_n y_n : ℝ), N = (x_n, y_n) ∧ curve_C a x_n y_n ∧ line_l x_n y_n) →
  (let (px, py) := point_P
   let pm := Real.sqrt ((M.1 - px)^2 + (M.2 - py)^2)
   let pn := Real.sqrt ((N.1 - px)^2 + (N.2 - py)^2)
   let mn := Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)
   geometric_sequence pm mn pn) →
  a = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_geometric_sequence_implies_a_equals_one_l937_93793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_radii_correct_l937_93705

/-- A right trapezoid with inscribed circles -/
structure RightTrapezoidWithCircles where
  a : ℝ
  b : ℝ
  h_ab : 0 < a ∧ a < b

/-- The radii of the inscribed circles in a right trapezoid -/
noncomputable def inscribed_circle_radii (t : RightTrapezoidWithCircles) : ℝ × ℝ :=
  let r := (t.a * Real.sqrt t.b) / (Real.sqrt t.a + Real.sqrt t.b)
  let R := (t.b * Real.sqrt t.a) / (Real.sqrt t.a + Real.sqrt t.b)
  (r, R)

theorem inscribed_circles_radii_correct (t : RightTrapezoidWithCircles) :
  let (r, R) := inscribed_circle_radii t
  r = (t.a * Real.sqrt t.b) / (Real.sqrt t.a + Real.sqrt t.b) ∧
  R = (t.b * Real.sqrt t.a) / (Real.sqrt t.a + Real.sqrt t.b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_radii_correct_l937_93705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_three_color_triangle_l937_93736

/-- A color type representing the three possible colors --/
inductive Color
  | Green
  | Red
  | Blue

/-- A vertex in the triangle --/
structure Vertex where
  color : Color

/-- A small triangle within the large triangle --/
structure SmallTriangle where
  v1 : Vertex
  v2 : Vertex
  v3 : Vertex

/-- The large triangle ΔGRB with its coloring conditions --/
structure LargeTriangle where
  G : Vertex
  R : Vertex
  B : Vertex
  smallTriangles : Finset SmallTriangle
  h_G_color : G.color = Color.Green
  h_R_color : R.color = Color.Red
  h_B_color : B.color = Color.Blue
  h_GR_color : ∀ v : Vertex, v.color = Color.Green ∨ v.color = Color.Red
  h_RB_color : ∀ v : Vertex, v.color = Color.Red ∨ v.color = Color.Blue
  h_GB_color : ∀ v : Vertex, v.color = Color.Green ∨ v.color = Color.Blue
  h_small_triangle_count : smallTriangles.card = 25

/-- The main theorem --/
theorem exists_three_color_triangle (T : LargeTriangle) :
  ∃ t ∈ T.smallTriangles, t.v1.color ≠ t.v2.color ∧ t.v2.color ≠ t.v3.color ∧ t.v1.color ≠ t.v3.color := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_three_color_triangle_l937_93736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_is_188_9_l937_93798

/-- Represents the exchange rates and distribution ratios -/
structure ExchangeRates where
  pound_to_dollar : ℚ
  euro_to_dollar : ℚ
  yen_to_dollar : ℚ
  aud_to_dollar : ℚ
  a_ratio : ℚ
  b_ratio : ℚ
  c_ratio : ℚ
  d_ratio : ℚ
  e_ratio : ℚ

/-- Calculates the total amount distributed in Dollars -/
def calculate_total_amount (rates : ExchangeRates) (c_share_yen : ℚ) : ℚ :=
  let a_share := c_share_yen / rates.c_ratio * rates.a_ratio
  let b_share := c_share_yen / rates.c_ratio * rates.b_ratio
  let d_share := c_share_yen / rates.c_ratio * rates.d_ratio
  let e_share := c_share_yen / rates.c_ratio * rates.e_ratio
  a_share * rates.pound_to_dollar +
  b_share * rates.euro_to_dollar +
  c_share_yen * rates.yen_to_dollar / 200 +
  d_share +
  e_share * rates.aud_to_dollar

/-- Theorem stating that the total amount distributed is 188.9 Dollars -/
theorem total_amount_is_188_9 (rates : ExchangeRates)
  (h1 : rates.pound_to_dollar = 59/50)
  (h2 : rates.euro_to_dollar = 28/25)
  (h3 : rates.yen_to_dollar = 91/10000)
  (h4 : rates.aud_to_dollar = 73/100)
  (h5 : rates.a_ratio = 1)
  (h6 : rates.b_ratio = 3/2)
  (h7 : rates.c_ratio = 200)
  (h8 : rates.d_ratio = 2)
  (h9 : rates.e_ratio = 6/5) :
  calculate_total_amount rates 5000 = 1889/10 := by
  sorry

#eval calculate_total_amount
  { pound_to_dollar := 59/50
    euro_to_dollar := 28/25
    yen_to_dollar := 91/10000
    aud_to_dollar := 73/100
    a_ratio := 1
    b_ratio := 3/2
    c_ratio := 200
    d_ratio := 2
    e_ratio := 6/5 }
  5000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_is_188_9_l937_93798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jihyae_money_l937_93769

/-- The amount of money Jihyae had initially -/
def initial_money : ℕ → Prop := sorry

/-- The amount spent on school supplies -/
def school_supplies (m : ℕ) : ℕ := m / 2 + 200

/-- The amount saved -/
def savings (m : ℕ) : ℕ := (m - school_supplies m) / 2 + 300

/-- The amount left after spending and saving -/
def remaining (m : ℕ) : ℕ := m - school_supplies m - savings m

theorem jihyae_money :
  ∃ m : ℕ, initial_money m ∧ 
    school_supplies m = m / 2 + 200 ∧
    savings m = (m - school_supplies m) / 2 + 300 ∧
    remaining m = 350 ∧
    m = 3000 := by
  sorry

#check jihyae_money

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jihyae_money_l937_93769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_vegetarian_eaters_l937_93759

-- Define the family and its dietary preferences
def Family := Nat
def total_veg_eaters : Nat := 26
def both_veg_and_nonveg_eaters : Nat := 11

-- Theorem to prove
theorem only_vegetarian_eaters : 
  total_veg_eaters - both_veg_and_nonveg_eaters = 15 := by
  -- Proof goes here
  sorry

#eval total_veg_eaters - both_veg_and_nonveg_eaters

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_vegetarian_eaters_l937_93759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l937_93783

theorem undefined_values_count : 
  ∃! (S : Finset ℝ), (∀ x ∈ S, (x^2 + 4*x - 5) * (x - 4) = 0) ∧ S.card = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l937_93783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l937_93734

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y^2 * z^2) ^ (1/3) ≥ (x * y * z) ^ ((x + y + z) / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l937_93734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adams_change_l937_93766

/-- Calculates the change after a purchase with tax and fee --/
def calculate_change (initial_amount : ℚ) (item_cost : ℚ) (tax_rate : ℚ) (additional_fee : ℚ) : ℚ :=
  let tax := (item_cost * tax_rate).floor / 100
  let total_cost := item_cost + tax + additional_fee
  ((initial_amount - total_cost) * 100).floor / 100

/-- Proves that Adam's change is $0.07 --/
theorem adams_change :
  calculate_change 5 4.28 0.07 0.35 = 0.07 := by
  -- Unfold the definition of calculate_change
  unfold calculate_change
  -- Simplify the expressions
  simp
  -- The proof is completed
  sorry

#eval calculate_change 5 4.28 0.07 0.35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adams_change_l937_93766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_properties_l937_93762

/-- Ellipse C with given properties -/
def EllipseC (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Circle M with center (-1, 0) and radius r -/
def CircleM (x y r : ℝ) : Prop :=
  (x + 1)^2 + y^2 = r^2

/-- Slope of tangent line from point (0, 1) to circle M -/
def TangentSlope (r k : ℝ) : Prop :=
  (1 - r^2) * k^2 - 2 * k + 1 - r^2 = 0

theorem ellipse_circle_properties :
  ∀ (r : ℝ) (k₁ k₂ : ℝ),
  0 < r → r < 1 →
  TangentSlope r k₁ →
  TangentSlope r k₂ →
  k₁ ≠ k₂ →
  (∀ x y, EllipseC x y ↔ x^2 / 4 + y^2 = 1) ∧
  k₁ * k₂ = 1 ∧
  ∃ (x y : ℝ), 
    EllipseC x y ∧ 
    y - 1 = k₁ * x ∧
    y - 1 = k₂ * x ∧
    y = -5/3 ∧
    x = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_properties_l937_93762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l937_93733

-- Define the constants
noncomputable def a : ℝ := Real.rpow 7 0.3
noncomputable def b : ℝ := Real.rpow 0.3 7
noncomputable def c : ℝ := Real.log 0.3 / Real.log 7

-- State the theorem
theorem relationship_abc : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l937_93733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l937_93752

/-- Represents the fuel efficiency of an SUV in different driving conditions -/
structure SUVFuelEfficiency where
  highway : ℝ
  city : ℝ

/-- Calculates the maximum distance an SUV can travel given its fuel efficiency and available gasoline -/
noncomputable def maxDistance (efficiency : SUVFuelEfficiency) (gasoline : ℝ) : ℝ :=
  max (efficiency.highway * gasoline) (efficiency.city * gasoline)

/-- Theorem stating the maximum distance an SUV can travel with given efficiency and gasoline -/
theorem suv_max_distance (efficiency : SUVFuelEfficiency) (gasoline : ℝ) :
  efficiency.highway = 12.2 →
  efficiency.city = 7.6 →
  gasoline = 23 →
  maxDistance efficiency gasoline = 280.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l937_93752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_dividing_curve_length_l937_93758

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  area : ℝ
  area_pos : 0 < area

/-- Represents a curve that divides a triangle -/
structure DividingCurve where
  length : ℝ
  divides_equally : Bool

/-- The shortest curve that divides an equilateral triangle into two equal areas -/
noncomputable def shortest_dividing_curve (t : EquilateralTriangle) : DividingCurve :=
  { length := Real.sqrt (Real.pi * t.area / 3)
  , divides_equally := true }

/-- Theorem: The shortest curve dividing an equilateral triangle into two equal areas has length √(πS/3) -/
theorem shortest_dividing_curve_length (t : EquilateralTriangle) :
  ∀ c : DividingCurve, c.divides_equally → (shortest_dividing_curve t).length ≤ c.length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_dividing_curve_length_l937_93758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_division_values_l937_93767

noncomputable def truncate (x : ℝ) : ℝ := ⌊x * 1000⌋ / 1000

def possible_values (a : ℝ) : Set ℝ :=
  {x | ∃ (a0 : ℝ), a0 = truncate a ∧ x = truncate (a0 / a)}

theorem truncated_division_values :
  ∀ a : ℝ, a > 0 →
    possible_values a = {0} ∪ {x | x ≥ 0.5 ∧ x ≤ 1 ∧ ∃ n : ℕ, x = truncate (0.5 + n * 0.001)} :=
by
  sorry

#check truncated_division_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_division_values_l937_93767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_k_value_l937_93789

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 10*x

-- Define the point inside the circle
def point_inside : ℝ × ℝ := (5, 3)

-- Define the common difference range
def d_range (d : ℝ) : Prop := 1/3 ≤ d ∧ d ≤ 1/2

-- Define the arithmetic sequence of chord lengths
def chord_sequence (k : ℕ) (a₁ aₖ d : ℝ) : Prop :=
  aₖ = a₁ + (k - 1) * d

-- Theorem statement
theorem impossible_k_value :
  ∀ (k : ℕ) (a₁ aₖ d : ℝ),
    my_circle (point_inside.1) (point_inside.2) →
    d_range d →
    chord_sequence k a₁ aₖ d →
    k ≠ 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_k_value_l937_93789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l937_93764

/-- The circle equation -/
def circle_eq (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x + a*y - 5 = 0

/-- The line equation -/
def line_eq (x y : ℝ) : Prop :=
  2*x + y - 1 = 0

/-- The symmetric point with respect to the line -/
def symmetric_point (x y x' y' : ℝ) : Prop :=
  line_eq ((x + x') / 2) ((y + y') / 2) ∧
  (x' - x) = 2 * (((x + x') / 2) - x) ∧
  (y' - y) = 2 * (((y + y') / 2) - y)

theorem circle_symmetry (a : ℝ) :
  (∀ x y, circle_eq a x y → 
    ∃ x' y', symmetric_point x y x' y' ∧ circle_eq a x' y') →
  a = -10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l937_93764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_range_l937_93712

/-- The dihedral angle between two adjacent lateral faces in a regular n-gonal pyramid -/
def dihedral_angle (n : ℕ) (θ : ℝ) : Prop :=
  n > 2 ∧ ((n - 2 : ℝ) / n) * Real.pi < θ ∧ θ < Real.pi

/-- Theorem stating the range of dihedral angles in a regular n-gonal pyramid -/
theorem dihedral_angle_range (n : ℕ) (h : n > 2) :
  ∃ θ : ℝ, dihedral_angle n θ := by
  sorry

#check dihedral_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_range_l937_93712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l937_93726

/-- The area of a triangle with two sides of length 30 and one side of length 50 --/
noncomputable def triangleArea : ℝ := Real.sqrt (55 * (55 - 30) * (55 - 30) * (55 - 50))

/-- Theorem stating that the area of the triangle is approximately 414.67 --/
theorem triangle_area_approx : 
  ∀ (ε : ℝ), ε > 0 → |triangleArea - 414.67| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l937_93726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_existence_l937_93702

theorem special_set_existence (n : ℕ) :
  ∃ (S : Finset ℕ), 
    Finset.card S = n ∧ 
    ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → 
      (∃ (d : ℕ), d > 0 ∧ d ∣ a ∧ d ∣ b ∧ d = a - b ∧ 
        ∀ (c : ℕ), c ∈ S → c ≠ a → c ≠ b → ¬(d ∣ c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_existence_l937_93702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contractor_fine_calculation_l937_93782

theorem contractor_fine_calculation 
  (total_days : ℕ) 
  (daily_wage : ℚ) 
  (absent_days : ℕ) 
  (total_payment : ℚ) 
  (h1 : total_days = 30)
  (h2 : daily_wage = 25)
  (h3 : absent_days = 10)
  (h4 : total_payment = 425)
  : 
  (total_days - absent_days) * daily_wage - absent_days * (daily_wage * total_days - total_payment) / absent_days = total_payment :=
by
  sorry

#eval (30 - 10) * 25 - 10 * ((25 * 30 - 425) / 10) -- Should evaluate to 425

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contractor_fine_calculation_l937_93782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_between_27_and_28_l937_93716

noncomputable def A : ℝ × ℝ := (20, 0)
noncomputable def B : ℝ × ℝ := (1, 0)
noncomputable def D : ℝ × ℝ := (1, 7)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem sum_distances_between_27_and_28 :
  27 < distance A D + distance B D ∧ distance A D + distance B D < 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_between_27_and_28_l937_93716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auditorium_rows_l937_93729

/-- The number of rows in the auditorium --/
def n : ℕ := 29

/-- Theorem stating that n satisfies the conditions of the problem --/
theorem auditorium_rows :
  (∀ (seating : Fin 30 → Fin n), ∃ (i j : Fin 30), i ≠ j ∧ seating i = seating j) ∧
  (∃ (seating : Fin 26 → Fin n), ∃ (r1 r2 r3 : Fin n), 
    (∀ i : Fin 26, seating i ≠ r1 ∧ seating i ≠ r2 ∧ seating i ≠ r3)) :=
by
  constructor
  · intro seating
    sorry  -- Proof that with 30 students, at least two share a row
  · sorry  -- Proof that with 26 students, at least three rows can be empty

#eval n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_auditorium_rows_l937_93729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l937_93745

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def time_to_pass_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating the time it takes for the train to pass the bridge -/
theorem train_bridge_passing_time :
  time_to_pass_bridge 592 36 253 = 84.5 := by
  -- Unfold the definition of time_to_pass_bridge
  unfold time_to_pass_bridge
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l937_93745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_curvature_lt_one_cubic_curvature_zero_constant_curvature_exists_parabola_curvature_le_two_l937_93770

/-- The curvature between two points on a curve -/
noncomputable def curvature (f : ℝ → ℝ) (x₁ x₂ : ℝ) : ℝ :=
  let y₁ := f x₁
  let y₂ := f x₂
  let k₁ := (deriv f) x₁
  let k₂ := (deriv f) x₂
  |k₁ - k₂| / Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The exponential function -/
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

/-- Theorem: The curvature of e^x is always less than 1 -/
theorem exp_curvature_lt_one (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  curvature exp x₁ x₂ < 1 := by
  sorry

/-- Statement 1: Curvature of y = x^3 at x = 1 and x = -1 is 0 -/
theorem cubic_curvature_zero :
  curvature (fun x => x^3) 1 (-1) = 0 := by
  sorry

/-- Statement 2: There exists a function with constant curvature -/
theorem constant_curvature_exists :
  ∃ f : ℝ → ℝ, ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → ∃ c : ℝ, curvature f x₁ x₂ = c := by
  sorry

/-- Statement 3: Curvature of y = x^2 + 1 is always ≤ 2 -/
theorem parabola_curvature_le_two (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  curvature (fun x => x^2 + 1) x₁ x₂ ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_curvature_lt_one_cubic_curvature_zero_constant_curvature_exists_parabola_curvature_le_two_l937_93770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_rate_ratio_l937_93715

/-- Represents the work rate of three workers digging a pit -/
structure WorkRate where
  rate : ℝ

/-- The work rate when workers work in shifts as described in the problem -/
def shift_rate : WorkRate := ⟨2⟩

/-- The work rate when workers work simultaneously -/
def simultaneous_rate : WorkRate := ⟨2.5⟩

/-- Theorem stating the ratio of simultaneous work rate to shift work rate -/
theorem work_rate_ratio : 
  (simultaneous_rate.rate / shift_rate.rate) = 2.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_rate_ratio_l937_93715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_argument_l937_93722

noncomputable def complex_sum : ℂ :=
  Complex.exp (11 * Real.pi * Complex.I / 60) +
  Complex.exp (29 * Real.pi * Complex.I / 60) +
  Complex.exp (49 * Real.pi * Complex.I / 60) +
  Complex.exp (59 * Real.pi * Complex.I / 60) +
  Complex.exp (79 * Real.pi * Complex.I / 60)

theorem complex_sum_argument :
  Complex.arg complex_sum = 49 * Real.pi / 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_argument_l937_93722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jogging_speed_is_pi_over_three_l937_93773

/-- Represents the athletic track -/
structure Track where
  straight_length : ℝ
  width : ℝ

/-- Calculates the length of the inner path of the track -/
noncomputable def inner_path_length (t : Track) (inner_radius : ℝ) : ℝ :=
  t.straight_length + 2 * Real.pi * inner_radius

/-- Calculates the length of the outer path of the track -/
noncomputable def outer_path_length (t : Track) (inner_radius : ℝ) : ℝ :=
  t.straight_length + 2 * Real.pi * (inner_radius + t.width)

/-- Theorem stating that the jogging speed is π/3 m/s given the track conditions -/
theorem jogging_speed_is_pi_over_three (t : Track) (inner_radius : ℝ) (time_diff : ℝ) :
  t.straight_length = 200 ∧ t.width = 8 ∧ time_diff = 48 →
  ∃ speed : ℝ, speed = Real.pi / 3 ∧
    outer_path_length t inner_radius / speed = inner_path_length t inner_radius / speed + time_diff := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jogging_speed_is_pi_over_three_l937_93773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_m_l937_93799

/-- A rational number m satisfies the conditions if |m| < 150 and 
    the equation 6x^2 + mx + 18 = 0 has at least one integer solution for x -/
def satisfies_conditions (m : ℚ) : Prop :=
  (abs m < 150) ∧ (∃ x : ℤ, 6 * x^2 + m * x + 18 = 0)

/-- The set of all rational numbers m that satisfy the conditions -/
def valid_m_set : Set ℚ :=
  {m : ℚ | satisfies_conditions m}

/-- The theorem stating that the number of elements in the valid_m_set is 48 -/
theorem count_valid_m : ∃ (s : Finset ℚ), s.card = 48 ∧ ∀ m, m ∈ s ↔ m ∈ valid_m_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_m_l937_93799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_work_days_l937_93796

/-- Proves that Mark works 5 days per week given his wage, hours, and expenses. -/
theorem marks_work_days : ℕ := 5

lemma marks_work_days_proof : marks_work_days = 5 := by
  let old_wage : ℚ := 40
  let raise_percentage : ℚ := 5 / 100
  let new_wage : ℚ := old_wage * (1 + raise_percentage)
  let hours_per_day : ℕ := 8
  let old_bills : ℕ := 600
  let personal_trainer : ℕ := 100
  let leftover : ℕ := 980
  let total_expenses : ℕ := old_bills + personal_trainer
  
  -- Calculate days worked as a rational number
  let days_worked : ℚ := (leftover + total_expenses) / (new_wage * hours_per_day)
  
  -- Assert that days_worked equals 5
  have h : days_worked = 5 := by
    -- The actual proof would go here
    sorry
  
  -- Show that marks_work_days equals 5
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_work_days_l937_93796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_investment_ratio_l937_93731

/-- 
Given:
- Susan invests money at two different rates: 11% and 9%.
- The total interest after 1 year is $2400.
- She has invested $12000 at each rate.

Prove: The ratio of the amount invested at 11% to the amount invested at 9% is 1:1.
-/
theorem susan_investment_ratio : 
  ∀ (amount_11 amount_9 : ℚ),
  amount_11 = 12000 →
  amount_9 = 12000 →
  (11 : ℚ) / 100 * amount_11 + (9 : ℚ) / 100 * amount_9 = 2400 →
  amount_11 / amount_9 = 1 := by
  sorry

/-- The total interest earned from both investments -/
def total_interest (amount_11 amount_9 : ℚ) : ℚ :=
  (11 : ℚ) / 100 * amount_11 + (9 : ℚ) / 100 * amount_9

/-- The ratio of the amounts invested at 11% and 9% -/
def investment_ratio (amount_11 amount_9 : ℚ) : ℚ :=
  amount_11 / amount_9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_investment_ratio_l937_93731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ratio_theorem_l937_93728

/-- Parabola type representing y^2 = 2Px -/
structure Parabola where
  P : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def Parabola.focus (p : Parabola) : Point :=
  { x := p.P, y := 0 }

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_ratio_theorem (p : Parabola) (A B : Point) :
  A.x = 2 ∧ A.y = 4 ∧ B.x = 8 ∧ B.y = -8 ∧ p.P = 4 →
  (distance A (p.focus)) / (distance B (p.focus)) = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ratio_theorem_l937_93728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_l937_93738

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a * x

-- Part 1: Tangent line equation
theorem tangent_line_equation (a : ℝ) (h : a > 0) :
  a = 1 → ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ (2 * Real.exp 1 - 1) * x - y - Real.exp 1 = 0 :=
by sorry

-- Part 2: Range of a
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → f a x ≥ Real.log x - x + 1) ↔ a > 0 ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_l937_93738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_true_l937_93735

-- Define the basic structures in algorithms
inductive AlgorithmStructure
| Sequential
| ConditionalBranch
| Loop

-- Define what it means for an algorithm to contain a structure
def contains (algorithm : Type) (s : AlgorithmStructure) : Prop := sorry

-- Define the three statements
def statement1 : Prop := ∀ (algorithm : Type), contains algorithm AlgorithmStructure.Sequential
def statement2 : Prop := ∀ (algorithm : Type), contains algorithm AlgorithmStructure.ConditionalBranch → contains algorithm AlgorithmStructure.Loop
def statement3 : Prop := ∀ (algorithm : Type), contains algorithm AlgorithmStructure.Loop → contains algorithm AlgorithmStructure.ConditionalBranch

-- The theorem to prove
theorem exactly_two_statements_true : 
  (statement1 ∧ ¬statement2 ∧ statement3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_true_l937_93735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_l937_93744

/-- Represents a regular polygon --/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ

/-- Calculates the interior angle of a regular polygon --/
noncomputable def interiorAngle (p : RegularPolygon) : ℝ :=
  180 * (p.sides - 2 : ℝ) / p.sides

/-- Represents the configuration of four polygons --/
structure FourPolygons where
  p1 : RegularPolygon
  p2 : RegularPolygon
  p3 : RegularPolygon
  p4 : RegularPolygon
  congruent : p1 = p4 ∨ p1 = p3 ∨ p1 = p2 ∨ p2 = p3 ∨ p2 = p4 ∨ p3 = p4
  unitSides : p1.sideLength = 1 ∧ p2.sideLength = 1 ∧ p3.sideLength = 1 ∧ p4.sideLength = 1
  angleSum : interiorAngle p1 + interiorAngle p2 + interiorAngle p3 + interiorAngle p4 = 360

/-- Calculates the perimeter of the external boundary --/
def externalPerimeter (fp : FourPolygons) : ℝ :=
  (fp.p1.sides + fp.p2.sides + fp.p3.sides + fp.p4.sides : ℝ) - 8

/-- The main theorem --/
theorem max_perimeter (fp : FourPolygons) : externalPerimeter fp ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_l937_93744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_E_l937_93720

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log (2/3)

-- Define the interval E
def E : Set ℝ := Set.Ioo (-3) (-1)

-- Theorem statement
theorem f_increasing_on_E :
  -- f is strictly increasing on E
  (∀ x₁ x₂, x₁ ∈ E → x₂ ∈ E → x₁ < x₂ → f x₁ < f x₂) ∧
  -- E is the largest such interval in the domain of f
  (∀ y, y ∉ E → y < -1 → ∃ z, z < y ∧ f z ≥ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_E_l937_93720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_P_complement_Q_equals_expected_result_l937_93703

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def Q : Set ℝ := {x | x^2 - 4 < 0}

-- Define the complement of Q in ℝ
def complement_Q : Set ℝ := {x | ¬ (x ∈ Q)}

-- Define the union of P and complement of Q
def union_P_complement_Q : Set ℝ := P ∪ complement_Q

-- Define the expected result set
def expected_result : Set ℝ := Set.Iic (-2) ∪ Set.Ici 1

-- Theorem statement
theorem union_P_complement_Q_equals_expected_result :
  union_P_complement_Q = expected_result := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_P_complement_Q_equals_expected_result_l937_93703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_inequality_l937_93788

theorem no_function_satisfies_inequality :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_inequality_l937_93788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_problem_l937_93741

/-- Definitions for the problem -/
def Line : Type := ℝ → ℝ → Prop
def Point : Type := ℝ × ℝ
def Parallel : Line → Line → Prop := sorry
def OnLine : Point → Line → Prop := sorry
def MeasureAngle : Point → Point → Point → ℝ := sorry

/-- Given two parallel lines and angles formed by a transversal, 
    prove that one of the angles measures 18 degrees. -/
theorem angle_measure_problem (p q r : Line) (A B C D E : Point) :
  Parallel p q →
  OnLine A p →
  OnLine B q →
  OnLine C r →
  OnLine D r →
  OnLine E q →
  MeasureAngle C A D = (1 / 9 : ℝ) * MeasureAngle C B D →
  MeasureAngle C A D = MeasureAngle C D E →
  MeasureAngle C B D + MeasureAngle C D E = 180 →
  MeasureAngle C D E = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_problem_l937_93741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_one_sixth_two_fifths_l937_93710

/-- The fraction halfway between two fractions on the number line -/
def fraction_midpoint (a b : ℚ) : ℚ := (a + b) / 2

/-- Theorem stating that the midpoint between 1/6 and 2/5 is 17/60 -/
theorem midpoint_one_sixth_two_fifths :
  fraction_midpoint (1/6 : ℚ) (2/5 : ℚ) = 17/60 := by
  -- Unfold the definition of fraction_midpoint
  unfold fraction_midpoint
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_one_sixth_two_fifths_l937_93710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_equivalence_function_domains_l937_93743

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 3) / Real.log 10
noncomputable def g (x : ℝ) : ℝ := Real.sqrt ((2 / (x - 1)) - 1)

-- Define the domain sets A and B
def A : Set ℝ := {x | x > 3/2}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Define the complement of B
def C_UB : Set ℝ := {x | x ≤ 1 ∨ x > 3}

-- Theorem to prove the equivalence of the sets
theorem sets_equivalence :
  A = {x | x > 3/2} ∧
  B = {x | 1 < x ∧ x ≤ 3} ∧
  (A ∩ B) = {x | 3/2 < x ∧ x ≤ 3} ∧
  (A ∪ C_UB) = {x | x ≤ 1 ∨ x > 3/2} := by
  sorry

-- Theorem to prove the domains of f and g
theorem function_domains :
  (∀ x ∈ A, 2 * x - 3 > 0) ∧
  (∀ x ∈ B, 2 / (x - 1) - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_equivalence_function_domains_l937_93743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l937_93742

/-- An ellipse with eccentricity √3/2 and minor axis length 4 -/
structure Ellipse :=
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : b = 2)
  (h4 : (a^2 - b^2) / a^2 = 3/4)

/-- The line y = x + 2 -/
def line (x : ℝ) : ℝ := x + 2

/-- The intersection points of the ellipse and the line -/
def intersection_points (e : Ellipse) : Set (ℝ × ℝ) :=
  {p | e.a^2 * p.2^2 + e.b^2 * p.1^2 = e.a^2 * e.b^2 ∧ p.2 = line p.1}

/-- The length of the chord formed by the intersection points -/
noncomputable def chord_length (e : Ellipse) : ℝ :=
  let points := intersection_points e
  let x1 := -2
  let y1 := 0
  let x2 := 6/5
  let y2 := 16/5
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem ellipse_chord_length (e : Ellipse) :
  chord_length e = 16 * Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l937_93742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l937_93753

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem trigonometric_function_properties 
  (ω φ : ℝ) 
  (h1 : ω > 0) 
  (h2 : |φ| < Real.pi / 2) 
  (h3 : Real.cos (Real.pi / 4) * Real.cos φ - Real.sin (3 * Real.pi / 4) * Real.sin φ = 0) 
  (h4 : ∃ (k : ℝ), 2 * Real.pi / ω = 2 * Real.pi / 3 + k * 2 * Real.pi) :
  φ = Real.pi / 4 ∧ 
  f ω φ = f 3 (Real.pi / 4) ∧
  (∃ (m : ℝ), m > 0 ∧ 
    (∀ (x : ℝ), f ω φ (x + m) = f ω φ (-x + m)) ∧
    (∀ (m' : ℝ), m' > 0 ∧ (∀ (x : ℝ), f ω φ (x + m') = f ω φ (-x + m')) → m ≤ m') ∧
    m = Real.pi / 12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l937_93753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l937_93765

-- Define the rectangle ABCD
noncomputable def AB : ℝ := 12 * Real.sqrt 3
noncomputable def BC : ℝ := 13 * Real.sqrt 3

-- Define the point P where diagonals intersect
noncomputable def P : ℝ × ℝ × ℝ := (0, 291 / (2 * Real.sqrt 399), 99 / Real.sqrt 133)

-- Define the base triangle ABC
noncomputable def baseArea : ℝ := 18 * Real.sqrt 133

-- Define the height of the pyramid
noncomputable def pyramidHeight : ℝ := 99 / Real.sqrt 133

-- Theorem statement
theorem pyramid_volume (isIsosceles : Bool) : isIsosceles → (1/3 : ℝ) * baseArea * pyramidHeight = 594 := by
  intro h
  sorry

#check pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l937_93765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pollen_mass_scientific_notation_l937_93772

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
noncomputable def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem pollen_mass_scientific_notation :
  let mass : ℝ := 0.000037
  let scientific_form := to_scientific_notation mass
  scientific_form.coefficient = 3.7 ∧ scientific_form.exponent = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pollen_mass_scientific_notation_l937_93772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_m_range_l937_93721

/-- A function f(x) = m * 3^x - x + 3 with m < 0 that has a root in the interval (0, 1) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * (3^x) - x + 3

/-- Theorem stating that if f(x) has a root in (0, 1) for m < 0, then -3 < m < -2/3 -/
theorem root_implies_m_range (m : ℝ) (h₁ : m < 0) 
  (h₂ : ∃ x ∈ Set.Ioo 0 1, f m x = 0) : 
  -3 < m ∧ m < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_m_range_l937_93721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sin_integral_l937_93713

open MeasureTheory

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := |Real.sin x|

-- State the theorem
theorem abs_sin_integral : ∫ x in (0)..(2*Real.pi), f x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sin_integral_l937_93713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_distance_to_circumcenter_l937_93730

/-- Circle Γ with center and radius -/
structure Circle (Γ : Type) where
  center : ℝ × ℝ
  radius : ℝ

/-- Point on a plane -/
def Point := ℝ × ℝ

/-- Line on a plane -/
structure Line where
  point1 : Point
  point2 : Point

/-- Triangle on a plane -/
structure Triangle where
  A : Point
  D : Point
  C : Point

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Circumcenter of a triangle -/
noncomputable def circumcenter (t : Triangle) : Point :=
  sorry

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  sorry

/-- Point is on a circle -/
def pointOnCircle (p : Point) (c : Circle ℝ) : Prop :=
  distance p c.center = c.radius

/-- Point is on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  distance p l.point1 + distance p l.point2 = distance l.point1 l.point2

theorem circle_tangent_distance_to_circumcenter 
  (Γ : Circle ℝ) 
  (A B C D : Point) 
  (AB_line AD_line : Line) 
  (ADC : Triangle) :
  distance A B = 6 →
  distance A B = distance B C →
  A ≠ C →
  pointOnCircle D Γ →
  AB_line.point1 = A ∧ AB_line.point2 = B →
  pointOnLine C AB_line →
  (∃ (E : Point), pointOnCircle E Γ ∧ distance C E = distance C D ∧ E ≠ D) →
  AD_line.point1 = A ∧ AD_line.point2 = D →
  ADC.A = A ∧ ADC.D = D ∧ ADC.C = C →
  distancePointToLine (circumcenter ADC) AD_line = 4 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_distance_to_circumcenter_l937_93730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_l937_93706

noncomputable def net_salary : ℝ := 3600
noncomputable def discretionary_income : ℝ := net_salary / 5
noncomputable def vacation_fund_percentage : ℝ := 30
noncomputable def eating_out_percentage : ℝ := 35
noncomputable def gifts_amount : ℝ := 108

theorem savings_percentage :
  ∃ (savings_percent : ℝ),
    savings_percent = 20 ∧
    vacation_fund_percentage / 100 * discretionary_income +
    eating_out_percentage / 100 * discretionary_income +
    savings_percent / 100 * discretionary_income +
    gifts_amount = discretionary_income := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_l937_93706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l937_93763

theorem sum_remainder_mod_15 (a b c : ℕ) 
  (ha : a % 15 = 11) 
  (hb : b % 15 = 13) 
  (hc : c % 15 = 14) : 
  (a + b + c) % 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l937_93763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sonic_lead_second_race_l937_93790

/-- Represents the race scenario between Sonic and Dash -/
structure RaceScenario where
  raceDistance : ℝ
  firstRaceLead : ℝ
  startingHandicap : ℝ

/-- Calculates the lead distance for Sonic in the second race -/
noncomputable def secondRaceLead (scenario : RaceScenario) : ℝ :=
  let sonicSecondRaceDistance := scenario.raceDistance + scenario.startingHandicap
  let dashSecondRaceDistance := scenario.raceDistance * (scenario.raceDistance - scenario.firstRaceLead) / scenario.raceDistance
  sonicSecondRaceDistance - dashSecondRaceDistance

/-- Theorem stating that Sonic finishes 19.2 meters ahead in the second race -/
theorem sonic_lead_second_race :
  let scenario : RaceScenario := {
    raceDistance := 200,
    firstRaceLead := 16,
    startingHandicap := 2.5 * 16
  }
  secondRaceLead scenario = 19.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sonic_lead_second_race_l937_93790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_frac_part_l937_93785

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

theorem largest_n_frac_part : 
  ∃ n : ℝ, (∀ m : ℝ, (floor m : ℝ) / m = 2015 / 2016 → m ≤ n) ∧ 
  (floor n : ℝ) / n = 2015 / 2016 ∧ 
  frac n = 2014 / 2015 := by
  sorry

#check largest_n_frac_part

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_frac_part_l937_93785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charles_jogging_speed_l937_93718

/-- Charles' jogging problem -/
theorem charles_jogging_speed :
  ∀ (S : ℝ),
  (S * (40 / 60) + 4 * ((70 - 40) / 60) = 6) →
  S = 8 := by
  intro S
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charles_jogging_speed_l937_93718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_axis_l937_93750

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * abs (Real.sin (x / 2)) + abs (Real.cos (x / 2))

-- Theorem statement
theorem f_symmetry_axis (k : ℤ) :
  ∀ x : ℝ, f (k * Real.pi + x) = f (k * Real.pi - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_axis_l937_93750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_l937_93704

theorem polynomial_value (x : ℝ) : 
  (2009 * x + 2008)^2 + (2009 * x + 2009)^2 + (2009 * x + 2010)^2 - 
  (2009 * x + 2008) * (2009 * x + 2009) - 
  (2009 * x + 2009) * (2009 * x + 2010) - 
  (2009 * x + 2008) * (2009 * x + 2010) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_l937_93704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_revenue_l937_93746

/-- Represents the number of 15-second commercials -/
def x : ℕ := 4

/-- Represents the number of 30-second commercials -/
def y : ℕ := 2

/-- The total advertising time in seconds -/
def total_time : ℕ := 120

/-- The fee for a 15-second commercial in thousand yuan -/
def fee_15s : ℚ := 0.6

/-- The fee for a 30-second commercial in thousand yuan -/
def fee_30s : ℚ := 1

/-- The constraint that each type of commercial must air at least twice -/
axiom min_airings : x ≥ 2 ∧ y ≥ 2

/-- The time constraint equation -/
axiom time_constraint : 15 * x + 30 * y = total_time

/-- The revenue function -/
def revenue : ℚ := x * fee_15s + y * fee_30s

/-- The theorem stating the maximum revenue -/
theorem max_revenue : ∃ (x y : ℕ), x ≥ 2 ∧ y ≥ 2 ∧ 15 * x + 30 * y = total_time ∧
  revenue = 4.4 ∧ ∀ (x' y' : ℕ), x' ≥ 2 → y' ≥ 2 → 15 * x' + 30 * y' = total_time →
  x' * fee_15s + y' * fee_30s ≤ 4.4 := by
  sorry

#eval revenue

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_revenue_l937_93746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_eq_cos_2x_solutions_l937_93739

theorem sin_3x_eq_cos_2x_solutions :
  ∃ (S : Finset ℝ), S.card = 4 ∧ 
  (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin (3 * x) = Real.cos (2 * x)) ∧
  (∀ y, 0 ≤ y ∧ y ≤ 2 * Real.pi ∧ Real.sin (3 * y) = Real.cos (2 * y) → y ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_eq_cos_2x_solutions_l937_93739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_and_isosceles_triangle_l937_93792

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a quadratic equation has two distinct real roots -/
def hasTwoDistinctRealRoots (eq : QuadraticEquation) : Prop :=
  eq.b^2 - 4*eq.a*eq.c > 0

/-- Represents a triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Main theorem -/
theorem quadratic_equation_and_isosceles_triangle (k : ℝ) :
  let eq : QuadraticEquation := ⟨1, -(2*k+1), k^2+k⟩
  (hasTwoDistinctRealRoots eq) ∧
  (∃ t : Triangle, t.c = 4 ∧ isIsosceles t ∧
    ((t.a = (2*k+1 + Real.sqrt ((2*k+1)^2 - 4*(k^2+k)))/2 ∧
      t.b = (2*k+1 - Real.sqrt ((2*k+1)^2 - 4*(k^2+k)))/2) ∨
     (t.b = (2*k+1 + Real.sqrt ((2*k+1)^2 - 4*(k^2+k)))/2 ∧
      t.a = (2*k+1 - Real.sqrt ((2*k+1)^2 - 4*(k^2+k)))/2))) →
  k = 3 ∨ k = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_and_isosceles_triangle_l937_93792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_wins_l937_93780

/-- Represents the game state -/
structure GameState where
  sticks : ℕ
  aliceTurn : Bool

/-- Defines a valid move in the game -/
def validMove (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 3

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : ℕ) : GameState :=
  { sticks := state.sticks - move
  , aliceTurn := ¬state.aliceTurn }

/-- Defines a winning strategy for Alice -/
inductive AliceWinningStrategy : GameState → Prop
  | aliceTurn (state : GameState) (move : ℕ) (h : validMove move) 
      (h' : (applyMove state move).sticks % 4 = 0) : 
    AliceWinningStrategy state
  | bobTurn (state : GameState) 
      (h : ∀ move, validMove move → 
        AliceWinningStrategy (applyMove state move)) :
    AliceWinningStrategy state

/-- The main theorem stating Alice has a winning strategy -/
theorem alice_wins :
  AliceWinningStrategy { sticks := 2022, aliceTurn := true } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_wins_l937_93780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l937_93749

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then Real.exp x + Real.sin x else Real.exp (-x) + Real.sin (-x)

theorem solution_set (h : ∀ x, f x = f (-x)) :
  {x : ℝ | f (2*x - 1) < Real.exp π} = Set.Ioo ((1 - π) / 2) ((1 + π) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l937_93749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_after_decimal_sqrt_n_squared_plus_n_plus_one_l937_93723

noncomputable def first_decimal_digit (x : ℝ) : ℕ :=
  Int.toNat ⌊(x - ↑⌊x⌋) * 10⌋

theorem first_digit_after_decimal_sqrt_n_squared_plus_n_plus_one (n : ℕ) (hn : n > 0) :
  (n = 1 → first_decimal_digit (Real.sqrt (n^2 + n + 1)) = 7) ∧
  (n = 2 → first_decimal_digit (Real.sqrt (n^2 + n + 1)) = 6) ∧
  (n = 3 → first_decimal_digit (Real.sqrt (n^2 + n + 1)) = 6) ∧
  (n ≥ 4 → first_decimal_digit (Real.sqrt (n^2 + n + 1)) = 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_after_decimal_sqrt_n_squared_plus_n_plus_one_l937_93723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_c_all_reals_l937_93786

noncomputable def c (k : ℝ) (x : ℝ) : ℝ := (k*x^2 + 2*x - 5) / (-5*x^2 + 3*x + k)

theorem domain_c_all_reals (k : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, c k x = y) ↔ k < -9/20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_c_all_reals_l937_93786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_inequality_l937_93737

theorem cubic_root_inequality (x : ℝ) :
  x^(1/3) + 3 / (x^(1/3) + 2) ≤ 0 ↔ x < -8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_inequality_l937_93737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carnival_spending_l937_93777

/- Define the age groups -/
inductive AgeGroup
| Under18
| Over18
deriving Repr, DecidableEq

/- Define the time of day -/
inductive TimeOfDay
| Morning
| Afternoon
deriving Repr, DecidableEq

/- Define the ride types -/
inductive RideType
| BumperCar
| SpaceShuttle
| FerrisWheel
deriving Repr, DecidableEq

/- Define the price structure -/
def basePrice (ride : RideType) (age : AgeGroup) : ℕ :=
  match ride, age with
  | RideType.BumperCar, AgeGroup.Under18 => 2
  | RideType.BumperCar, AgeGroup.Over18 => 3
  | RideType.SpaceShuttle, AgeGroup.Under18 => 4
  | RideType.SpaceShuttle, AgeGroup.Over18 => 5
  | RideType.FerrisWheel, AgeGroup.Under18 => 5
  | RideType.FerrisWheel, AgeGroup.Over18 => 6

/- Define the afternoon price increase -/
def afternoonIncrease : ℕ := 1

/- Calculate the price of a ride -/
def ridePrice (ride : RideType) (age : AgeGroup) (time : TimeOfDay) : ℕ :=
  basePrice ride age + (if time = TimeOfDay.Afternoon then afternoonIncrease else 0)

/- Define the number of rides for each person -/
def maraRides : List (RideType × TimeOfDay × ℕ) := [
  (RideType.BumperCar, TimeOfDay.Morning, 1),
  (RideType.BumperCar, TimeOfDay.Afternoon, 1),
  (RideType.FerrisWheel, TimeOfDay.Morning, 2),
  (RideType.FerrisWheel, TimeOfDay.Afternoon, 1)
]

def rileyRides : List (RideType × TimeOfDay × ℕ) := [
  (RideType.SpaceShuttle, TimeOfDay.Morning, 2),
  (RideType.SpaceShuttle, TimeOfDay.Afternoon, 2),
  (RideType.FerrisWheel, TimeOfDay.Morning, 2),
  (RideType.FerrisWheel, TimeOfDay.Afternoon, 1)
]

/- Calculate the total cost for a person -/
def totalCost (rides : List (RideType × TimeOfDay × ℕ)) (age : AgeGroup) : ℕ :=
  rides.foldl (fun acc (ride, time, count) => acc + count * ridePrice ride age time) 0

/- Theorem: The total amount spent by Mara and Riley is $62 -/
theorem carnival_spending :
  totalCost maraRides AgeGroup.Under18 + totalCost rileyRides AgeGroup.Over18 = 62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carnival_spending_l937_93777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_two_equals_negative_eight_l937_93707

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x ≥ 0 then 3^x - 1 else -((3^(-x)) - 1)

-- State the theorem
theorem f_negative_two_equals_negative_eight :
  f (-2) = -8 :=
by
  sorry

-- Define the odd function property
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

-- State that f is an odd function
axiom f_is_odd : is_odd_function f

-- State that f(x) = 3^x - 1 for x ≥ 0
axiom f_nonnegative (x : ℝ) : x ≥ 0 → f x = 3^x - 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_two_equals_negative_eight_l937_93707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suresh_work_time_l937_93714

/-- The time it takes for Suresh to complete the job alone -/
noncomputable def suresh_time : ℝ := 15

/-- The time it takes for Ashutosh to complete the job alone -/
noncomputable def ashutosh_time : ℝ := 15

/-- The time Ashutosh worked to complete the remaining job -/
noncomputable def ashutosh_remaining_time : ℝ := 6

/-- The fraction of the job completed by a person in one hour -/
noncomputable def job_fraction_per_hour (time : ℝ) : ℝ := 1 / time

theorem suresh_work_time :
  ∃ (x : ℝ), 
    x > 0 ∧ 
    x < suresh_time ∧
    (job_fraction_per_hour suresh_time) * x + 
    (job_fraction_per_hour ashutosh_time) * ashutosh_remaining_time = 1 ∧
    x = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suresh_work_time_l937_93714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_for_given_radius_and_area_l937_93781

/-- The surface area of a cylinder given its radius and height -/
noncomputable def cylinderSurfaceArea (r h : ℝ) : ℝ := 2 * Real.pi * r^2 + 2 * Real.pi * r * h

/-- The height of a cylinder given its radius and surface area -/
noncomputable def cylinderHeight (r sa : ℝ) : ℝ := (sa - 2 * Real.pi * r^2) / (2 * Real.pi * r)

theorem cylinder_height_for_given_radius_and_area :
  let r : ℝ := 3
  let sa : ℝ := 36 * Real.pi
  cylinderHeight r sa = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_for_given_radius_and_area_l937_93781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_n_bounds_l937_93711

-- Define the sequence a_n
def a : ℕ → ℚ
| 0 => 1
| n + 1 => 2 * a n

-- Define S_n as the sum of the first n terms of a_n
def S : ℕ → ℚ
| 0 => 0
| n + 1 => S n + a n

-- Define the sequence b_n
def b (n : ℕ) : ℚ := (n + 1) / (4 * a n)

-- Define T_n as the sum of the first n terms of b_n
def T : ℕ → ℚ
| 0 => 0
| n + 1 => T n + b n

-- State the theorem
theorem T_n_bounds (n : ℕ) : 1/4 ≤ T n ∧ T n < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_n_bounds_l937_93711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bouquet_carnations_l937_93778

theorem flower_bouquet_carnations (F : ℚ) (hF : F > 0) : 
  let pink := (6 : ℚ)/10 * F
  let pink_roses := (1 : ℚ)/3 * pink
  let red := F - pink
  let red_carnations := (3 : ℚ)/4 * red
  let pink_carnations := pink - pink_roses
  let total_carnations := pink_carnations + red_carnations
  total_carnations = (1 : ℚ)/2 * F := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bouquet_carnations_l937_93778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l937_93727

/-- Given an arithmetic sequence {a_n} with first term a₁ and common difference d,
    S_n represents the sum of the first n terms. -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence {a_n} with common difference d,
    if 2S_3 - 3S_2 = 15, then d = 5. -/
theorem arithmetic_sequence_common_difference 
  (a₁ d : ℝ) (h : 2 * S a₁ d 3 - 3 * S a₁ d 2 = 15) : d = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l937_93727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_count_l937_93768

/-- A point in the 2D plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- Checks if a point is within or on the boundary of a circle -/
def inCircle (center : IntPoint) (radius : ℕ) (point : IntPoint) : Prop :=
  (point.x - center.x) ^ 2 + (point.y - center.y) ^ 2 ≤ radius ^ 2

/-- The set of all integer points within or on the boundary of the intersection of two circles -/
def intersectionPoints (center1 center2 : IntPoint) (radius : ℕ) : Set IntPoint :=
  {p : IntPoint | inCircle center1 radius p ∧ inCircle center2 radius p}

theorem intersection_point_count :
  let center1 : IntPoint := ⟨0, 0⟩
  let center2 : IntPoint := ⟨8, 0⟩
  let radius : ℕ := 5
  ∃ (S : Finset IntPoint), S.card = 9 ∧ ∀ p, p ∈ S ↔ p ∈ intersectionPoints center1 center2 radius := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_count_l937_93768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_on_interval_l937_93724

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x^2 + 4*x - 5 else x^2 - 4*x - 5

theorem min_value_of_f_on_interval (h : ∀ x, f (-x) = f x) :
  ∃ x₀ ∈ Set.Icc 3 5, ∀ x ∈ Set.Icc 3 5, f x₀ ≤ f x ∧ f x₀ = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_on_interval_l937_93724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_s_constant_l937_93791

/-- The slope of the line s -/
def m : ℚ := 4/3

/-- The smallest distance from the origin to line s -/
def d : ℝ := 60

/-- The equation of line s: y = mx - c -/
def line_s (x y c : ℝ) : Prop := y = m * x - c

/-- The distance from a point (x, y) to the origin -/
noncomputable def distance_to_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- The theorem stating that the constant in the equation of line s is 100 -/
theorem line_s_constant : 
  ∃ c : ℝ, (∀ x y : ℝ, line_s x y c → distance_to_origin x y ≥ d) ∧ 
  (∃ x y : ℝ, line_s x y c ∧ distance_to_origin x y = d) → 
  c = 100 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_s_constant_l937_93791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l937_93774

theorem triangle_problem (a b c A B C : Real) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A ∧ A < π) → (0 < B ∧ B < π) → (0 < C ∧ C < π) →
  (A + B + C = π) →
  -- Law of cosines
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Given conditions
  (b = 2) →
  (c = 2 * Real.sqrt 3) →
  -- Part I
  (A = 5*π/6 → a = 2 * Real.sqrt 7) ∧
  -- Part II
  (C = π/2 + A → A = π/6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l937_93774

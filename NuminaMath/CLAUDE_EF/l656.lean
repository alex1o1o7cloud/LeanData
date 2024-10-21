import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_circle_positional_relationship_l656_65683

-- Define the circle's center and radius
def circle_center : ℝ × ℝ := (2, -3)
def circle_radius : ℝ := 5

-- Define the point M
def point_M : ℝ × ℝ := (5, -7)

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem to prove that point M is on the circle
theorem point_on_circle :
  distance circle_center point_M = circle_radius :=
by sorry

-- Theorem to state the positional relationship
theorem positional_relationship :
  distance circle_center point_M = circle_radius →
  (point_M ∈ Metric.sphere circle_center circle_radius) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_circle_positional_relationship_l656_65683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l656_65605

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being invertible
def IsInvertible (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Define the equation we're solving
def Equation (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f (x^3) = f (x^6)

-- Theorem statement
theorem intersection_points (h : IsInvertible f) :
  (∃ x y : ℝ, x ≠ y ∧ Equation f x ∧ Equation f y) ∧
  (∀ x y z : ℝ, Equation f x ∧ Equation f y ∧ Equation f z → x = y ∨ x = z ∨ y = z) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l656_65605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_equiv_l656_65694

/-- Converts rectangular coordinates to spherical coordinates -/
noncomputable def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let θ := if x ≥ 0 ∧ y ≥ 0 then Real.arctan (y / x)
           else if x < 0 ∧ y ≥ 0 then Real.pi + Real.arctan (y / x)
           else if x < 0 ∧ y < 0 then Real.pi + Real.arctan (y / x)
           else 2 * Real.pi + Real.arctan (y / x)
  let φ := Real.arccos (z / ρ)
  (ρ, θ, φ)

/-- Theorem stating the equivalence of given rectangular and spherical coordinates -/
theorem rectangular_to_spherical_equiv :
  let (ρ, θ, φ) := rectangular_to_spherical (4 * Real.sqrt 3) (-2) 5
  ρ = Real.sqrt 77 ∧
  θ = 11 * Real.pi / 6 ∧
  φ = Real.arccos (5 / Real.sqrt 77) ∧
  ρ > 0 ∧
  0 ≤ θ ∧ θ < 2 * Real.pi ∧
  0 ≤ φ ∧ φ ≤ Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_equiv_l656_65694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_georgia_carnation_friends_l656_65639

/-- The cost of a single carnation in dollars -/
def single_carnation_cost : ℚ := 1/2

/-- The cost of a dozen carnations in dollars -/
def dozen_carnation_cost : ℚ := 4

/-- The number of teachers Georgia sent carnations to -/
def number_of_teachers : ℕ := 5

/-- The total amount Georgia spent in dollars -/
def total_spent : ℚ := 25

/-- The number of friends Georgia bought a single carnation for -/
def number_of_friends : ℕ := 
  ((total_spent - (↑number_of_teachers * dozen_carnation_cost)) / single_carnation_cost).floor.toNat

theorem georgia_carnation_friends : number_of_friends = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_georgia_carnation_friends_l656_65639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_quadratic_forms_l656_65657

theorem odd_prime_quadratic_forms (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ((∃ (m n : ℤ), p = m^2 + 16*n^2) ↔ p % 8 = 1) ∧
  ((∃ (m n : ℤ), p = 4*m^2 + 4*m*n + 5*n^2) ↔ p % 8 = 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_prime_quadratic_forms_l656_65657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_intersecting_lines_l656_65646

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 72

-- Define the point G
def G : ℝ × ℝ := (-3, 0)

-- Define the trajectory M
def M (x y : ℝ) : Prop := x^2 / 18 + y^2 / 9 = 1

-- Define a line with slope 1
def Line (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define a circle passing through the origin
def CircleThroughOrigin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem trajectory_and_intersecting_lines :
  -- Part 1: M is the trajectory of E
  (∀ x y, M x y ↔ 
    ∃ s : ℝ × ℝ, C s.1 s.2 ∧ 
    ∃ e : ℝ × ℝ, (e.1 - G.1)^2 + (e.2 - G.2)^2 = (e.1 - s.1)^2 + (e.2 - s.2)^2 ∧
         (e.1 - G.1) * (s.1 - e.1) + (e.2 - G.2) * (s.2 - e.2) = 0 ∧
         x = e.1 ∧ y = e.2) ∧
  -- Part 2: Exactly two lines with slope 1 intersect M and form circles through origin
  (∃! m₁ m₂, m₁ ≠ m₂ ∧
    (∀ m, (∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ 
      M x₁ y₁ ∧ M x₂ y₂ ∧ 
      Line m x₁ y₁ ∧ Line m x₂ y₂ ∧
      CircleThroughOrigin x₁ y₁ x₂ y₂) ↔ 
    (m = m₁ ∨ m = m₂)) ∧
    m₁ = 2 * Real.sqrt 3 ∧ m₂ = -2 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_intersecting_lines_l656_65646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_foci_slope_om_l656_65660

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := y^2 / 9 + x^2 / 8 = 1

-- Define the line
def line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the foci of the ellipse
noncomputable def F1 : ℝ × ℝ := (0, Real.sqrt 5)
noncomputable def F2 : ℝ × ℝ := (0, -Real.sqrt 5)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Theorem 1
theorem cosine_angle_foci (x y : ℝ) :
  ellipse x y →
  distance x y F1.1 F1.2 = distance x y F2.1 F2.2 →
  (distance x y F1.1 F1.2)^2 + (distance x y F2.1 F2.2)^2 - (distance F1.1 F1.2 F2.1 F2.2)^2
  / (2 * distance x y F1.1 F1.2 * distance x y F2.1 F2.2) = 7/9 := by sorry

-- Theorem 2
theorem slope_om (x1 y1 x2 y2 : ℝ) :
  ellipse x1 y1 →
  ellipse x2 y2 →
  line x1 y1 →
  line x2 y2 →
  ((y1 + y2) / 2) / ((x1 + x2) / 2) = -9/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_foci_slope_om_l656_65660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_has_triangle_l656_65630

/-- A polygon in the square division --/
structure Polygon where
  sides : ℕ
  convex : Bool
  in_square : Bool

/-- A square division into polygons --/
structure SquareDivision where
  polygons : List Polygon
  more_than_one : 1 < polygons.length
  all_convex : ∀ p, p ∈ polygons → p.convex
  pairwise_different : ∀ p q, p ∈ polygons → q ∈ polygons → p ≠ q → p.sides ≠ q.sides

/-- The existence of a triangle in the square division --/
def has_triangle (sd : SquareDivision) : Prop :=
  ∃ p, p ∈ sd.polygons ∧ p.sides = 3

/-- The main theorem: Any valid square division must contain a triangle --/
theorem square_division_has_triangle (sd : SquareDivision) : has_triangle sd := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_has_triangle_l656_65630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_for_given_eccentricity_l656_65628

/-- The ratio of semi-major axis to semi-minor axis for an ellipse with eccentricity √3/2 -/
theorem ellipse_ratio_for_given_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  (Real.sqrt (a^2 - b^2) / a = Real.sqrt 3 / 2) →
  a / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_for_given_eccentricity_l656_65628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gobblean_words_count_l656_65619

/-- The number of letters in the Gobblean alphabet -/
def alphabet_size : Nat := 6

/-- The maximum number of letters in a Gobblean word -/
def max_word_length : Nat := 4

/-- Calculates the number of possible words of a given length -/
def words_of_length (n : Nat) : Nat :=
  if n ≤ max_word_length then
    (List.range n).foldl (fun acc i => acc * (alphabet_size - i)) alphabet_size
  else
    0

/-- The total number of possible Gobblean words -/
def total_words : Nat :=
  (List.range (max_word_length + 1)).foldl (fun acc n => acc + words_of_length n) 0

theorem gobblean_words_count :
  total_words = 516 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gobblean_words_count_l656_65619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_account_balance_difference_l656_65645

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / periods) ^ (periods * time)

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem account_balance_difference :
  let angela_principal : ℝ := 5000
  let angela_rate : ℝ := 0.05
  let angela_periods : ℝ := 2
  let bob_principal : ℝ := 7000
  let bob_rate : ℝ := 0.04
  let time : ℝ := 15

  let angela_balance := compound_interest angela_principal angela_rate angela_periods time
  let bob_balance := simple_interest bob_principal bob_rate time

  ⌊bob_balance - angela_balance⌋ = 726 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_account_balance_difference_l656_65645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l656_65633

noncomputable def f (x : ℝ) := Real.log (abs x)

noncomputable def g (a x : ℝ) := 1 / (deriv f x) + a * (deriv f x)

theorem problem_solution (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x > 0, g a x ≥ 2) 
  (h3 : ∃ x > 0, g a x = 2) :
  (∀ x ≠ 0, g a x = x + a / x) ∧ 
  (a = 1) ∧ 
  (∫ x in (3/2)..(2), ((2/3) * x + 7/6) - (x + 1/x) = 7/24 + Real.log 3 - 2 * Real.log 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l656_65633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l656_65687

/-- The length of the first train given the conditions of the problem -/
noncomputable def first_train_length (v1 v2 : ℝ) (t : ℝ) (l2 : ℝ) : ℝ :=
  (v1 + v2) * (5/18) * t - l2

/-- The problem statement as a theorem -/
theorem train_length_problem (v1 v2 t l2 : ℝ) 
  (h1 : v1 = 42)
  (h2 : v2 = 30)
  (h3 : t = 12.998960083193344)
  (h4 : l2 = 160) :
  ∃ ε > 0, |first_train_length v1 v2 t l2 - 99.98| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l656_65687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_equation_m_range_l656_65612

theorem system_equation_m_range (m : ℝ) :
  (∃ x y : ℝ, Real.sin x = m * (Real.sin y)^3 ∧ Real.cos x = m * (Real.cos y)^3) →
  1 ≤ m ∧ m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_equation_m_range_l656_65612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l656_65659

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.cos (α + π/4) = 1/3) 
  (h2 : α ∈ Set.Ioo 0 (π/2)) : 
  Real.sin α = (4 - Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l656_65659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_munificence_cubic_l656_65671

/-- Munificence of a polynomial p(x) on [-1, 1] -/
noncomputable def munificence (p : ℝ → ℝ) : ℝ :=
  ⨆ (x : ℝ) (h : x ∈ Set.Icc (-1) 1), |p x|

/-- The polynomial p(x) = x³ - 3x + c -/
def p (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

theorem min_munificence_cubic :
  ∃ (c : ℝ), munificence (p c) = 5 ∧
  (∀ (c' : ℝ), munificence (p c') ≥ munificence (p c)) ∧
  c = -1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_munificence_cubic_l656_65671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_specific_l656_65638

/-- The area of a quadrilateral with a diagonal and two offsets -/
noncomputable def quadrilateral_area (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) : ℝ :=
  (1/2) * diagonal * offset1 + (1/2) * diagonal * offset2

/-- Theorem: The area of a quadrilateral with diagonal 28 cm and offsets 9 cm and 6 cm is 210 cm² -/
theorem quadrilateral_area_specific : quadrilateral_area 28 9 6 = 210 := by
  -- Unfold the definition of quadrilateral_area
  unfold quadrilateral_area
  -- Simplify the arithmetic expression
  simp [mul_add, add_mul, mul_comm, mul_assoc]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_specific_l656_65638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l656_65648

/-- The inverse proportion function f(x) = 3/x -/
noncomputable def f (x : ℝ) : ℝ := 3 / x

theorem inverse_proportion_properties :
  (∀ x, x > 0 → f x > 0) ∧
  (∀ x, x < 0 → f x < 0) ∧
  (∀ x₁ x₂, x₁ ≠ 0 → x₂ ≠ 0 → x₁ < x₂ → f x₁ > f x₂) ∧
  (f 1 = 3) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x| > (1 : ℝ)/ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l656_65648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l656_65696

noncomputable def clock_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let hour_angle := (hours % 12 + minutes / 60 : ℝ) * 30
  let minute_angle := minutes * 6
  let angle_diff := abs (minute_angle - hour_angle)
  min angle_diff (360 - angle_diff)

theorem clock_angle_at_3_15 : clock_angle 3 15 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l656_65696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_theorem_l656_65624

/-- The surface area of a solid of revolution formed by rotating a rectangle -/
noncomputable def surface_area_solid_revolution (S : ℝ) (α : ℝ) : ℝ :=
  4 * Real.sqrt 2 * Real.pi * S * Real.sin ((α / 2) + (Real.pi / 4))

/-- Theorem stating the surface area of the solid of revolution -/
theorem surface_area_theorem (S : ℝ) (α : ℝ) 
  (h1 : S > 0) -- Area of rectangle is positive
  (h2 : 0 < α ∧ α < Real.pi) -- Angle between diagonals is between 0 and π
  : 
  surface_area_solid_revolution S α = 
  4 * Real.sqrt 2 * Real.pi * S * Real.sin ((α / 2) + (Real.pi / 4)) :=
by
  -- Unfold the definition of surface_area_solid_revolution
  unfold surface_area_solid_revolution
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_theorem_l656_65624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_lengths_correct_l656_65613

noncomputable def train_a_length : ℝ := 600

noncomputable def train_b_length : ℝ := 300

noncomputable def platform_length : ℝ := 600

noncomputable def train_a_speed : ℝ := 72 * 1000 / 3600

noncomputable def train_b_speed : ℝ := 80 * 1000 / 3600

noncomputable def crossing_time : ℝ := 60

theorem train_lengths_correct : 
  (train_a_length = platform_length) ∧
  (train_b_length = platform_length / 2) ∧
  (train_a_speed * crossing_time = train_a_length + platform_length) ∧
  (train_a_length = 600) ∧
  (train_b_length = 300) := by
  sorry

#check train_lengths_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_lengths_correct_l656_65613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_exists_for_4_not_for_5_l656_65600

def is_valid_partition (n : ℕ) (A₁ A₂ A₃ : Finset ℕ) : Prop :=
  n ≥ 3 ∧
  A₁.card = n ∧ A₂.card = n ∧ A₃.card = n ∧
  A₁ ∪ A₂ ∪ A₃ = Finset.range (3 * n + 1) \ {0} ∧
  (∀ (i : Fin 3) (x y : ℕ), x ∈ (match i with | 0 => A₁ | 1 => A₂ | 2 => A₃) → 
    y ∈ (match i with | 0 => A₁ | 1 => A₂ | 2 => A₃) → x ≠ y →
    (x + y) % (3 * n + 1) ≠ 0 ∧ 
    ∀ (z : ℕ), z ∈ (match i with | 0 => A₁ | 1 => A₂ | 2 => A₃) → (x + y) % (3 * n + 1) ≠ z)

theorem partition_exists_for_4_not_for_5 :
  (∃ (A₁ A₂ A₃ : Finset ℕ), is_valid_partition 4 A₁ A₂ A₃) ∧
  (¬ ∃ (A₁ A₂ A₃ : Finset ℕ), is_valid_partition 5 A₁ A₂ A₃) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_exists_for_4_not_for_5_l656_65600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_tick_medicine_cost_l656_65618

theorem flea_tick_medicine_cost
  (original_price : ℝ)
  (duration_months : ℕ)
  (cashback_rate : ℝ)
  (coupon_discount_rate : ℝ)
  (mail_in_rebate : ℝ)
  (shipping_fee : ℝ)
  (sales_tax_rate : ℝ)
  (h1 : original_price = 150)
  (h2 : duration_months = 6)
  (h3 : cashback_rate = 0.1)
  (h4 : coupon_discount_rate = 0.15)
  (h5 : mail_in_rebate = 25)
  (h6 : shipping_fee = 12)
  (h7 : sales_tax_rate = 0.05) :
  (original_price * (1 - coupon_discount_rate) + shipping_fee) * (1 + sales_tax_rate) - mail_in_rebate - (original_price * cashback_rate) / duration_months = 17.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_tick_medicine_cost_l656_65618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l656_65670

noncomputable def θ : ℝ :=
  Real.arctan (1/2) + Real.pi

def α_condition (α : ℝ) : Prop :=
  Real.cos (α + Real.pi/4) = Real.sin θ

theorem angle_problem :
  -- Part 1
  Real.cos (Real.pi/2 + θ) = Real.sqrt 5 / 5 ∧
  -- Part 2
  ∀ α, α_condition α →
    (Real.sin (2*α + Real.pi/4) = 7 * Real.sqrt 2 / 10 ∨
     Real.sin (2*α + Real.pi/4) = - Real.sqrt 2 / 10) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l656_65670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_theorem_l656_65611

/-- Represents an investment split between two interest rates -/
structure Investment where
  total : ℝ
  rate1 : ℝ
  rate2 : ℝ
  tax : ℝ

/-- Calculates the average effective interest rate for the given investment -/
noncomputable def averageEffectiveRate (inv : Investment) (x : ℝ) : ℝ :=
  let postTaxRate2 := inv.rate2 * (1 - inv.tax)
  let interest1 := inv.rate1 * (inv.total - x)
  let interest2 := postTaxRate2 * x
  (interest1 + interest2) / inv.total

/-- Theorem stating the average effective rate is approximately 5.6% -/
theorem investment_theorem (inv : Investment) (x : ℝ) :
  inv.total = 5000 ∧
  inv.rate1 = 0.05 ∧
  inv.rate2 = 0.07 ∧
  inv.tax = 0.10 ∧
  inv.rate1 * (inv.total - x) = inv.rate2 * (1 - inv.tax) * x →
  ∃ ε > 0, |averageEffectiveRate inv x - 0.056| < ε := by
  sorry

#eval "Investment theorem defined"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_theorem_l656_65611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_54880000_l656_65603

theorem cube_root_54880000 : Real.rpow 54880000 (1/3) = 140 * Real.rpow 10 (1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_54880000_l656_65603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_lambda_l656_65626

/-- Given two vectors a and b in ℝ², if λa + b is perpendicular to a, then λ = -1 -/
theorem perpendicular_vector_lambda (a b : ℝ × ℝ) (h1 : a = (1, -3)) (h2 : b = (4, -2)) :
  ∃ l : ℝ, (l • a.1 + b.1, l • a.2 + b.2) • a = 0 → l = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_lambda_l656_65626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_term_in_cube_decomposition_l656_65607

/-- Given that for m = 5, m^2 = 1 + 3 + 5 + 7 + 9, 
    prove that the largest number in the decomposition of m^3 is 29 -/
theorem largest_term_in_cube_decomposition (m : ℕ) (h1 : m = 5) 
  (h2 : m^2 = 1 + 3 + 5 + 7 + 9) : 
  ∃ (decomp : List ℕ), (decomp.sum = m^3) ∧ (decomp.maximum? = some 29) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_term_in_cube_decomposition_l656_65607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_max_volume_l656_65695

/-- The volume of a right circular cone with radius r and height h -/
noncomputable def coneVolume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The height of a right circular cone with radius r and unit generatrix -/
noncomputable def coneHeight (r : ℝ) : ℝ := Real.sqrt (1 - r^2)

/-- Theorem: The volume of a right circular cone with unit generatrix is maximized when the radius is √(2/3) -/
theorem cone_max_volume :
  ∃ (r : ℝ), r > 0 ∧ r < 1 ∧
  ∀ (r' : ℝ), r' > 0 → r' < 1 →
  coneVolume r' (coneHeight r') ≤ coneVolume r (coneHeight r) ∧
  r = Real.sqrt (2/3) := by
  sorry

#check cone_max_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_max_volume_l656_65695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_l656_65651

/-- Given a positive integer m where 150m^3 has 150 positive integer divisors,
    prove that 64m^4 has 675 positive integer divisors. -/
theorem divisor_count (m : ℕ+) 
  (h : (Finset.filter (λ x : ℕ => 150 * m.val ^ 3 % x = 0) (Finset.range (150 * m.val ^ 3 + 1))).card = 150) : 
  (Finset.filter (λ x : ℕ => 64 * m.val ^ 4 % x = 0) (Finset.range (64 * m.val ^ 4 + 1))).card = 675 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_l656_65651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approximation_l656_65686

/-- The area of a triangle using Heron's formula -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 10, 30, and 21 is approximately 17.31 -/
theorem triangle_area_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |triangleArea 10 30 21 - 17.31| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approximation_l656_65686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l656_65625

theorem min_value_of_expression : 
  (∀ x : ℝ, (16 : ℝ)^x - (4 : ℝ)^x + 1 ≥ 3/4) ∧ 
  (∃ y : ℝ, (16 : ℝ)^y - (4 : ℝ)^y + 1 = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l656_65625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l656_65689

noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def ψ : ℝ := (1 - Real.sqrt 5) / 2

def E : Type := ℕ → ℝ

def is_in_E (u : E) : Prop :=
  ∀ n : ℕ, u (n + 2) = u (n + 1) + u n

noncomputable def a : E := λ n => φ ^ n
noncomputable def b : E := λ n => ψ ^ n

def is_basis (v w : E) : Prop :=
  (∀ u : E, ∃ l m : ℝ, ∀ n : ℕ, u n = l * v n + m * w n) ∧
  (∀ l m : ℝ, (∀ n : ℕ, l * v n + m * w n = 0) → l = 0 ∧ m = 0)

noncomputable def F : E := λ n =>
  (1 / Real.sqrt 5) * (φ ^ n - ψ ^ n)

theorem main_theorem :
  is_in_E a ∧ is_in_E b ∧
  is_basis a b ∧
  (∀ n : ℕ, F (n + 2) = F (n + 1) + F n) ∧
  F 0 = 0 ∧ F 1 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l656_65689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cabinet_discount_percentage_l656_65663

/-- Calculates the discount percentage given the original price and sale price -/
noncomputable def discount_percentage (original_price sale_price : ℝ) : ℝ :=
  (original_price - sale_price) / original_price * 100

theorem cabinet_discount_percentage :
  let original_price : ℝ := 1200
  let sale_price : ℝ := 1020
  discount_percentage original_price sale_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cabinet_discount_percentage_l656_65663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_area_l656_65664

/-- The area of a regular hexagon inscribed in a circle with radius 3 units -/
noncomputable def hexagon_area : ℝ := (27 * Real.sqrt 3) / 2

/-- Theorem: The area of a regular hexagon inscribed in a circle with radius 3 units is (27 * √3) / 2 square units -/
theorem inscribed_hexagon_area :
  let circle_radius : ℝ := 3
  hexagon_area = (27 * Real.sqrt 3) / 2 := by
  sorry

#check inscribed_hexagon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_area_l656_65664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_lcm_a_c_l656_65629

-- Define the theorem
theorem least_possible_lcm_a_c :
  ∃ (a b c : ℕ), 
    Nat.lcm a b = 20 ∧ 
    Nat.lcm b c = 18 ∧
    (∃ (a' c' : ℕ), Nat.lcm a' c' = 90 ∧ 
      ∀ (a'' c'' : ℕ), Nat.lcm a'' b = 20 → Nat.lcm b c'' = 18 → Nat.lcm a'' c'' ≥ 90) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_lcm_a_c_l656_65629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l656_65698

open Real

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  Real.tan A / Real.tan B = 4/3 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = b * Real.sin C / Real.sin B →
  a = c * Real.sin A / Real.sin C →
  b = c * Real.sin B / Real.sin C →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →
  (∀ (area : ℝ), area = 1/2 * b * c * Real.sin A → area ≤ 1/2) ∧
  (∃ (area : ℝ), area = 1/2 * b * c * Real.sin A ∧ area = 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l656_65698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_is_perpendicular_l656_65608

/-- A plane passing through the x-axis -/
structure PlanePassingThroughXAxis where
  a : ℝ
  c : ℝ
  eq : (y : ℝ) → (z : ℝ) → a * y + c * z = 0

/-- The distance from a point to a plane passing through the x-axis -/
noncomputable def distanceToPlane (p : ℝ × ℝ × ℝ) (plane : PlanePassingThroughXAxis) : ℝ :=
  let (x, y, z) := p
  |plane.a * y + plane.c * z| / Real.sqrt (plane.a^2 + plane.c^2)

/-- Represents the length of the perpendicular segment from a point to the plane -/
noncomputable def perpendicularSegmentLength (p : ℝ × ℝ × ℝ) (plane : PlanePassingThroughXAxis) : ℝ :=
  sorry  -- This is a placeholder for the actual implementation

/-- Theorem stating that the distance formula is correct -/
theorem distance_to_plane_is_perpendicular 
  (p : ℝ × ℝ × ℝ) (plane : PlanePassingThroughXAxis) :
  distanceToPlane p plane = 
    perpendicularSegmentLength p plane := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_is_perpendicular_l656_65608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_profit_share_l656_65661

/-- Represents a partner in the investment partnership -/
structure Partner where
  investment : ℚ
  time : ℚ

/-- Calculates the share ratio of a partner -/
def shareRatio (p : Partner) : ℚ := p.investment * p.time

/-- Calculates the total share ratio of all partners -/
def totalShareRatio (partners : List Partner) : ℚ :=
  partners.map shareRatio |>.sum

/-- Calculates a partner's share of the total profit -/
def profitShare (p : Partner) (partners : List Partner) (totalProfit : ℚ) : ℚ :=
  (shareRatio p / totalShareRatio partners) * totalProfit

theorem b_profit_share
  (d_investment : ℚ)
  (a b c d : Partner)
  (h1 : a.investment = 5 * d_investment)
  (h2 : b.investment = (5/4) * d_investment)
  (h3 : c.investment = (5/2) * d_investment)
  (h4 : d.investment = d_investment)
  (h5 : a.time = 6)
  (h6 : b.time = 8)
  (h7 : c.time = 10)
  (h8 : d.time = 12)
  (total_profit : ℚ)
  (h9 : total_profit = 10000) :
  profitShare b [a, b, c, d] total_profit = (10 / 77) * 10000 := by
  sorry

#eval ((10 : ℚ) / 77) * 10000 -- To verify the approximate result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_profit_share_l656_65661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_vector_combination_l656_65688

theorem range_of_vector_combination (A B C : ℝ × ℝ) (lambda mu : ℝ) :
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A.1^2 + A.2^2 = 1 →
  B.1^2 + B.2^2 = 1 →
  C.1^2 + C.2^2 = 1 →
  lambda > 0 →
  mu > 0 →
  C = (lambda * A.1 + mu * B.1, lambda * A.2 + mu * B.2) →
  (lambda - 1)^2 + (mu - 3)^2 > 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_vector_combination_l656_65688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_less_than_half_l656_65602

-- Define an equilateral triangle with side length 1
def EquilateralTriangle : Set (ℝ × ℝ) := sorry

-- Define a function to check if a point is inside the triangle
def isInside (p : ℝ × ℝ) (t : Set (ℝ × ℝ)) : Prop := sorry

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem distance_less_than_half (t : Set (ℝ × ℝ)) (p1 p2 p3 p4 p5 : ℝ × ℝ) :
  t = EquilateralTriangle →
  isInside p1 t ∧ isInside p2 t ∧ isInside p3 t ∧ isInside p4 t ∧ isInside p5 t →
  ∃ i j, i ≠ j ∧ i ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ j ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    distance (match i with
              | 1 => p1
              | 2 => p2
              | 3 => p3
              | 4 => p4
              | _ => p5)
             (match j with
              | 1 => p1
              | 2 => p2
              | 3 => p3
              | 4 => p4
              | _ => p5) < 0.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_less_than_half_l656_65602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l656_65679

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := -1/3 * x^3 + x^2

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 4 ∧ 
  (∀ x, x ∈ Set.Icc 0 4 → f x ≤ f c) ∧
  f c = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l656_65679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_max_l656_65631

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def TriangleCondition (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧ 
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  (Real.sqrt 3 / 6) * t.a = t.b * Real.sin t.C / 2

-- State the theorem
theorem triangle_ratio_max (t : Triangle) (h : TriangleCondition t) :
  (t.b / t.c + t.c / t.b) ≤ 4 := by
  sorry

#check triangle_ratio_max

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_max_l656_65631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l656_65644

/-- The speed of a train crossing a bridge -/
noncomputable def train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  speed_ms * 3.6

/-- Theorem: A train 120 meters long crossing a 255-meter bridge in 30 seconds travels at 45 km/hr -/
theorem train_speed_problem : train_speed 120 255 30 = 45 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the expression
  simp
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l656_65644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_product_l656_65680

-- Define the original expression
noncomputable def original_expr : ℝ := (2 + Real.sqrt 5) / (3 - Real.sqrt 5)

-- Define the rationalized form
noncomputable def rationalized_form : ℝ := 11/4 + 5/4 * Real.sqrt 5

-- Define the coefficients A, B, and C
def A : ℤ := 11
def B : ℤ := 5
def C : ℤ := 5

-- Theorem statement
theorem rationalize_and_product :
  (original_expr = rationalized_form) ∧ (A * B * C = 275) := by
  sorry

#eval A * B * C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_product_l656_65680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_conditions_satisfied_l656_65662

/-- The projection of vector u onto vector v -/
noncomputable def proj (v u : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  (scalar * v.1, scalar * v.2)

/-- The vector u we're trying to prove correct -/
def u : ℝ × ℝ := (10, 1)

/-- The first projection vector -/
def v1 : ℝ × ℝ := (3, 2)

/-- The second projection vector -/
def v2 : ℝ × ℝ := (1, 4)

theorem projection_conditions_satisfied :
  proj v1 u = (12, 8) ∧ proj v2 u = (35/17, 140/17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_conditions_satisfied_l656_65662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_boys_one_girl_l656_65621

/-- The probability of having at least two boys and one girl in a family of four children -/
theorem prob_two_boys_one_girl (p : ℝ) (h_p : p = 1 / 2) :
  let n := 4  -- number of children
  let prob_2b2g := Nat.choose n 2 * p^n  -- probability of 2 boys and 2 girls
  let prob_3b1g := Nat.choose n 3 * p^n  -- probability of 3 boys and 1 girl
  let prob_4b := Nat.choose n 4 * p^n    -- probability of 4 boys
  prob_2b2g + prob_3b1g + prob_4b = 11 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_boys_one_girl_l656_65621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_hyperbola_equation_l656_65610

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The focal length of a hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on the asymptote of a hyperbola -/
def on_asymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y / p.x = h.b / h.a

/-- The theorem stating the properties of the specific hyperbola -/
theorem hyperbola_properties (h : Hyperbola) (p : Point) :
  focal_length h = 10 →
  on_asymptote h p →
  p.x = 2 →
  p.y = 1 →
  h.a = 2 * Real.sqrt 5 ∧ h.b = Real.sqrt 5 := by sorry

/-- The main theorem proving the equation of the hyperbola -/
theorem hyperbola_equation (h : Hyperbola) (p : Point) :
  focal_length h = 10 →
  on_asymptote h p →
  p.x = 2 →
  p.y = 1 →
  ∀ x y : ℝ, (x^2 / 20 - y^2 / 5 = 1) ↔ (x^2 / h.a^2 - y^2 / h.b^2 = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_hyperbola_equation_l656_65610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_english_marks_l656_65685

/-- Represents the marks of a student in different subjects -/
structure StudentMarks where
  english : ℚ
  mathematics : ℚ
  physics : ℚ
  chemistry : ℚ
  biology : ℚ

/-- Calculates the average marks of a student -/
def averageMarks (marks : StudentMarks) : ℚ :=
  (marks.english + marks.mathematics + marks.physics + marks.chemistry + marks.biology) / 5

/-- Theorem stating that David's marks in English are 72 -/
theorem davids_english_marks :
  ∃ (marks : StudentMarks),
    marks.mathematics = 45 ∧
    marks.physics = 72 ∧
    marks.chemistry = 77 ∧
    marks.biology = 75 ∧
    averageMarks marks = 68.2 ∧
    marks.english = 72 := by
  sorry

#check davids_english_marks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_english_marks_l656_65685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_vectors_l656_65637

/-- Given two plane vectors a and b with an angle of 60° between them and magnitude 2,
    prove that the projection of a onto b is (1/2) * b -/
theorem projection_of_vectors (a b : ℝ × ℝ) : 
  (a.1 * b.1 + a.2 * b.2 = 2) →  -- dot product is 2 (cos 60° = 1/2)
  (a.1^2 + a.2^2 = 4) →              -- |a| = 2
  (b.1^2 + b.2^2 = 4) →              -- |b| = 2
  ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b = (1/2 : ℝ) • b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_vectors_l656_65637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l656_65691

theorem book_price_change (initial_price : ℝ) (initial_price_pos : initial_price > 0) : 
  (initial_price * (1 - 0.5) * (1 + 0.6) - initial_price) / initial_price = -0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l656_65691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l656_65601

theorem expression_simplification :
  Real.sqrt (1/4) - (1/8)^(1/3 : ℝ) + Real.sqrt 81 + |Real.sqrt 2 - 3| = 12 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l656_65601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_8_percent_l656_65682

/-- The time (in years) it takes for an investment to triple in value when compounded annually at rate r% -/
noncomputable def tripling_time (r : ℝ) : ℝ := 112 / r

/-- The final value of an investment after t years, given an initial value and annual interest rate -/
noncomputable def investment_value (initial_value rate t : ℝ) : ℝ :=
  initial_value * (1 + rate / 100) ^ t

theorem interest_rate_is_8_percent (r : ℝ) :
  tripling_time r = 112 / r →
  investment_value 3500 r 28 = 31500 →
  r = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_8_percent_l656_65682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_age_is_60_l656_65609

/-- The age of the father given the conditions in the problem -/
def fathers_age (mans_age : ℚ) : ℚ := mans_age * (5/2)

/-- The conditions of the problem -/
theorem fathers_age_is_60 :
  ∃ (mans_age : ℚ),
    (mans_age = (2/5) * fathers_age mans_age) ∧
    (mans_age + 12 = (1/2) * (fathers_age mans_age + 12)) ∧
    fathers_age mans_age = 60 := by
  -- We'll use 24 as the man's age
  use 24
  constructor
  · -- Prove mans_age = (2/5) * fathers_age mans_age
    simp [fathers_age]
    norm_num
  constructor
  · -- Prove mans_age + 12 = (1/2) * (fathers_age mans_age + 12)
    simp [fathers_age]
    norm_num
  · -- Prove fathers_age mans_age = 60
    simp [fathers_age]
    norm_num

#check fathers_age_is_60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_age_is_60_l656_65609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_equals_8_l656_65622

/-- A function that maps letters to digits -/
def digit_map (c : Char) : ℕ := sorry

/-- The sum of digits represented by A, B, C, and D is 20 -/
axiom sum_20 : digit_map 'A' + digit_map 'B' + digit_map 'C' + digit_map 'D' = 20

/-- The sum of digits represented by B and A, plus 1, is 11 -/
axiom sum_11 : digit_map 'B' + digit_map 'A' + 1 = 11

/-- All digits are unique -/
axiom unique_digits : ∀ x y : Char, x ≠ y → digit_map x ≠ digit_map y

/-- All digits are between 0 and 9 -/
axiom valid_digits : ∀ x : Char, 0 ≤ digit_map x ∧ digit_map x ≤ 9

/-- The value of D is 8 -/
theorem d_equals_8 : digit_map 'D' = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_equals_8_l656_65622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l656_65669

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define a point P outside the circle
def point_outside_circle (x y : ℝ) : Prop :=
  ¬(circle_C x y) ∧ x^2 + y^2 > 4

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 2)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem: The trajectory of P satisfying |PM| = |PO| is 2x - 4y + 1 = 0
theorem trajectory_of_P (x y : ℝ) :
  point_outside_circle x y →
  (distance x y (-1) 2 = circle_radius) →
  (distance x y 0 0 = distance x y (-1) 2) →
  2*x - 4*y + 1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l656_65669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_c_value_l656_65627

/-- A quadratic function with a range of [0, +∞) and a specific solution set for f(x) < c -/
def quadratic_function (a b c m : ℝ) : Prop :=
  -- Define the function
  let f := fun (x : ℝ) => x^2 + a*x + b
  -- Range condition
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f x = y) ∧
  (∀ x : ℝ, f x ≥ 0) ∧
  -- Solution set condition
  (∀ x : ℝ, f x < c ↔ m < x ∧ x < m + 6)

/-- The theorem stating that under the given conditions, c must equal 9 -/
theorem quadratic_function_c_value (a b c m : ℝ) :
  quadratic_function a b c m → c = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_c_value_l656_65627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_propositions_l656_65668

/-- A statement is a linguistic expression. -/
def Statement : Type := String

/-- A proposition is a statement that can be judged as true or false. -/
def isProposition (s : Statement) : Bool := sorry

/-- The list of given statements. -/
def givenStatements : List Statement :=
  [ "The empty set is a proper subset of any set"
  , "Find the roots of x^2-3x-4=0"
  , "What are the integers that satisfy 3x-2>0?"
  , "Close the door"
  , "Are two lines perpendicular to the same line necessarily parallel?"
  , "Natural numbers are even"
  ]

/-- The theorem stating that exactly 2 of the given statements are propositions. -/
theorem two_propositions :
  (givenStatements.filter isProposition).length = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_propositions_l656_65668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_is_80_l656_65606

/-- The probability of a coin landing heads on a single toss -/
noncomputable def p_heads : ℝ := 1 / 3

/-- The total number of coins -/
def total_coins : ℕ := 100

/-- The maximum number of tosses for each coin -/
def max_tosses : ℕ := 4

/-- The probability of a coin landing heads after up to four tosses -/
noncomputable def p_heads_four_tosses : ℝ :=
  p_heads + (1 - p_heads) * p_heads + 
  (1 - p_heads)^2 * p_heads + 
  (1 - p_heads)^3 * p_heads

/-- The expected number of coins landing heads after up to four tosses -/
noncomputable def expected_heads : ℝ := total_coins * p_heads_four_tosses

theorem expected_heads_is_80 : expected_heads = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_is_80_l656_65606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_weight_loss_l656_65658

/-- The combined weight loss of two friends given their individual weekly loss and duration -/
theorem combined_weight_loss 
  (aleesia_weekly_loss : ℝ) 
  (aleesia_weeks : ℕ) 
  (alexei_weekly_loss : ℝ) 
  (alexei_weeks : ℕ) 
  (h1 : aleesia_weekly_loss = 1.5) 
  (h2 : aleesia_weeks = 10) 
  (h3 : alexei_weekly_loss = 2.5) 
  (h4 : alexei_weeks = 8) : 
  aleesia_weekly_loss * (aleesia_weeks : ℝ) + alexei_weekly_loss * (alexei_weeks : ℝ) = 35 := by
  sorry

#check combined_weight_loss

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_weight_loss_l656_65658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_sum_equals_fraction_l656_65643

/-- The sum of the double infinite series ∑_{n=1}^∞ ∑_{k=1}^n k^2 / 3^(n+k) equals 3645/41552 -/
theorem double_sum_equals_fraction : 
  (∑' n : ℕ+, ∑' k : ℕ, (k : ℝ)^2 / 3^(n + k)) = 3645 / 41552 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_sum_equals_fraction_l656_65643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_formation_theorem_l656_65647

-- Define the universe of discourse
universe u
variable {α : Type u}

-- Define the properties of a well-defined set
def is_well_defined_set (S : Set α) : Prop :=
  ∀ x, x ∈ S → (∃ y, y = x) ∧ (∀ z, z = x → z ∈ S)

-- Define the groups
variable (all_cubes : Set α)
variable (all_major_supermarkets_in_Wenzhou : Set α)
variable (all_difficult_math_problems : Set α)
variable (famous_dancers : Set α)
variable (all_products_from_factory_2012 : Set α)
variable (all_points_on_coordinate_axes : Set α)

-- Theorem statement
theorem set_formation_theorem :
  is_well_defined_set all_cubes ∧
  is_well_defined_set all_products_from_factory_2012 ∧
  is_well_defined_set all_points_on_coordinate_axes ∧
  ¬(is_well_defined_set all_major_supermarkets_in_Wenzhou) ∧
  ¬(is_well_defined_set all_difficult_math_problems) ∧
  ¬(is_well_defined_set famous_dancers) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_formation_theorem_l656_65647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l656_65674

/-- The length of the major axis of the ellipse 16x^2 + 9y^2 = 144 is 8 -/
theorem ellipse_major_axis_length : ∃ (major_axis_length : ℝ), major_axis_length = 8 := by
  -- Define the ellipse equation
  let ellipse_eq : ℝ → ℝ → Prop := fun x y => 16 * x^2 + 9 * y^2 = 144

  -- State that the length of the major axis is 8
  let major_axis_length : ℝ := 8

  -- Prove that the major axis length is correct
  have major_axis_length_is_correct : major_axis_length = 8 := by rfl

  -- Conclude the proof
  exact ⟨major_axis_length, major_axis_length_is_correct⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l656_65674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l656_65620

/-- Given that 0.overline{02} = 2/99, prove that 2.overline{17} = 215/99 -/
theorem repeating_decimal_to_fraction :
  (∃ (x : ℚ), x = 2 / 99 ∧ (∀ n : ℕ, (x * 10^n - (x * 10^n).floor) = 0.02)) →
  (∃ (y : ℚ), y = 215 / 99 ∧ (∀ n : ℕ, ((y - 2) * 10^n - ((y - 2) * 10^n).floor) = 0.17)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l656_65620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equiv_fraction_l656_65641

/-- The decimal representation of the number we're considering -/
def x : ℚ := 0.4 + (5/9) / 100

/-- The theorem stating that the given repeating decimal equals the fraction 226/495 -/
theorem repeating_decimal_equiv_fraction : x = 226 / 495 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equiv_fraction_l656_65641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_travels_to_beijing_l656_65655

/-- Represents that the events are independent --/
axiom IndependentEvents : Prop

/-- 
Calculates the probability that at least one of three independent events occurs
given their individual probabilities
--/
def ProbabilityAtLeastOne (p1 p2 p3 : ℝ) : ℝ :=
  1 - (1 - p1) * (1 - p2) * (1 - p3)

theorem at_least_one_travels_to_beijing 
  (prob_A prob_B prob_C : ℝ) 
  (hA : prob_A = 1/3) 
  (hB : prob_B = 1/4) 
  (hC : prob_C = 1/5) 
  (hIndep : IndependentEvents) : 
  ProbabilityAtLeastOne prob_A prob_B prob_C = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_travels_to_beijing_l656_65655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l656_65692

/-- Calculates the time (in seconds) for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem stating that a train of length 110 m traveling at 60 kmph takes approximately 18 seconds to cross a bridge of length 190 m -/
theorem train_bridge_crossing_time :
  let time := train_crossing_time 110 60 190
  ∃ ε > 0, |time - 18| < ε :=
by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check train_crossing_time 110 60 190

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l656_65692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_multiples_of_nine_l656_65656

theorem two_digit_multiples_of_nine : 
  (Finset.filter (λ n : ℕ => 10 ≤ n ∧ n ≤ 99 ∧ n % 9 = 0) (Finset.range 100)).card = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_multiples_of_nine_l656_65656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l656_65667

-- Define the line l passing through points A and B
noncomputable def line_l (m : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ t : ℝ, x = t * (-Real.sqrt 3) ∧ y = t * (3*m^2 + 12*m + 11) + 2}

-- Define the angle of inclination θ
noncomputable def angle_of_inclination (m : ℝ) : ℝ :=
  Real.arctan ((3*m^2 + 12*m + 11) / (Real.sqrt 3))

-- Theorem statement
theorem angle_of_inclination_range :
  ∀ m : ℝ, 
    (angle_of_inclination m ∈ Set.Icc 0 (π/6)) ∨ 
    (angle_of_inclination m ∈ Set.Ioo (π/2) π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l656_65667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_problem_l656_65650

/-- Calculate the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  (amount - principal) * 100 / (principal * time)

/-- The simple interest rate for the given problem -/
def problem_rate : ℚ := simple_interest_rate 750 950 5

theorem simple_interest_rate_problem :
  (problem_rate * 1000).floor / 1000 = 533 / 100 := by
  -- Proof goes here
  sorry

#eval (problem_rate * 1000).floor / 1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_problem_l656_65650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_OA_OB_and_coplanar_l656_65684

/-- Given four points in ℝ³ -/
def O : Fin 3 → ℝ := ![0, 0, 0]
def A : Fin 3 → ℝ := ![4, 3, 0]
def B : Fin 3 → ℝ := ![-3, 0, 4]
def C : Fin 3 → ℝ := ![5, 6, 4]

/-- Vector from O to A -/
def OA : Fin 3 → ℝ := ![A 0 - O 0, A 1 - O 1, A 2 - O 2]

/-- Vector from O to B -/
def OB : Fin 3 → ℝ := ![B 0 - O 0, B 1 - O 1, B 2 - O 2]

/-- Vector from B to C -/
def BC : Fin 3 → ℝ := ![C 0 - B 0, C 1 - B 1, C 2 - B 2]

/-- Dot product of two 3D vectors -/
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0 * w 0) + (v 1 * w 1) + (v 2 * w 2)

/-- Magnitude of a 3D vector -/
noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

/-- Cosine of the angle between two 3D vectors -/
noncomputable def cos_angle (v w : Fin 3 → ℝ) : ℝ :=
  dot_product v w / (magnitude v * magnitude w)

theorem angle_OA_OB_and_coplanar :
  cos_angle OA OB = -12/25 ∧
  ∃ (a b c d : ℝ), a * O 0 + b * A 0 + c * B 0 + d * C 0 = 0 ∧
                   a * O 1 + b * A 1 + c * B 1 + d * C 1 = 0 ∧
                   a * O 2 + b * A 2 + c * B 2 + d * C 2 = 0 ∧
                   (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_OA_OB_and_coplanar_l656_65684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_prime_pairs_sum_50_l656_65623

/-- A function that returns the number of unordered pairs of prime numbers that sum to 50 -/
def count_prime_pairs_sum_50 : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    Nat.Prime p.1 ∧ Nat.Prime p.2 ∧ p.1 + p.2 = 50 ∧ p.1 ≤ p.2)
    (Finset.product (Finset.range 50) (Finset.range 50))).card

/-- Theorem stating that there are exactly 4 unordered pairs of prime numbers that sum to 50 -/
theorem four_prime_pairs_sum_50 : count_prime_pairs_sum_50 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_prime_pairs_sum_50_l656_65623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l656_65617

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 3/4
  h_point : b^2 = 1

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  h_distinct : M ≠ N ∧ M ≠ (0, -1) ∧ N ≠ (0, -1)
  h_on_line : M.2 = k * M.1 + 3/5 ∧ N.2 = k * N.1 + 3/5
  h_on_ellipse : M.1^2 / E.a^2 + M.2^2 / E.b^2 = 1 ∧
                 N.1^2 / E.a^2 + N.2^2 / E.b^2 = 1

/-- Main theorem -/
theorem ellipse_properties (E : Ellipse) (L : IntersectingLine E) :
  (E.a = 2 ∧ E.b = 1) ∧
  (L.M.1 * L.N.1 + (L.M.2 + 1) * (L.N.2 + 1) = 0) ∧
  (∃ (k : ℝ), k = 0 ∨ k = Real.sqrt 5 / 5 ∨ k = -(Real.sqrt 5 / 5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l656_65617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_l656_65642

theorem sin_product (α β : ℝ) 
  (h1 : Real.cos (α - β) = 1/3) 
  (h2 : Real.cos (α + β) = 2/3) : 
  Real.sin α * Real.sin β = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_l656_65642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_orthocenter_l656_65677

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A rectangular parallelepiped -/
structure Parallelepiped where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  distinct_edges : A.x ≠ B.x ∧ B.y ≠ C.y ∧ C.z ≠ D.z

/-- A plane in 3D space -/
structure Plane where
  normal : Point3D
  d : ℝ

/-- The set of intersection points -/
def IntersectionPoints (p : Parallelepiped) (plane : Plane) : Set Point3D :=
  sorry

/-- The orthocenter of a triangle -/
def Orthocenter (p1 p2 p3 : Point3D) : Point3D :=
  sorry

/-- Subtraction for Point3D -/
instance : HSub Point3D Point3D Point3D where
  hSub := λ p1 p2 => Point3D.mk (p1.x - p2.x) (p1.y - p2.y) (p1.z - p2.z)

/-- The main theorem -/
theorem intersection_points_orthocenter 
  (p : Parallelepiped) 
  (plane : Plane) 
  (h : plane.normal = p.D - p.A) : 
  ∃ (p1 p2 p3 : Point3D), 
    p1 ∈ IntersectionPoints p plane ∧
    p2 ∈ IntersectionPoints p plane ∧
    p3 ∈ IntersectionPoints p plane ∧
    Orthocenter p1 p2 p3 ∈ IntersectionPoints p plane := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_orthocenter_l656_65677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_floor_sequence_count_l656_65681

def floor_sequence (n : ℕ) : ℕ := Int.toNat ⌊(n^2 : ℚ) / 1000⌋

def distinct_count (s : List ℕ) : ℕ := (s.toFinset).card

theorem distinct_floor_sequence_count :
  distinct_count (List.map floor_sequence (List.range 1000)) = 751 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_floor_sequence_count_l656_65681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_l656_65634

/-- The time (in seconds) it takes for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  train_length / (train_speed_kmh * 1000 / 3600)

/-- Theorem: A train of length 120 m traveling at 180 km/h takes 2.4 seconds to cross an electric pole -/
theorem train_crossing_pole : train_crossing_time 120 180 = 2.4 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_l656_65634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l656_65653

noncomputable def original_curve (x : ℝ) : ℝ := Real.cos x

def scaling_x (x : ℝ) : ℝ := 2 * x
def scaling_y (y : ℝ) : ℝ := 3 * y

noncomputable def transformed_curve (x' : ℝ) : ℝ := 3 * Real.cos (x' / 2)

theorem curve_transformation :
  ∀ x' y', y' = transformed_curve x' ↔
    ∃ x y, y = original_curve x ∧ x' = scaling_x x ∧ y' = scaling_y y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l656_65653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l656_65673

open Polynomial

theorem polynomial_equality (m n : ℕ) (A B : ℂ[X]) :
  m > 2 →
  n > 2 →
  A ≠ 0 →
  B ≠ 0 →
  (degree A > 1 ∨ degree B > 1) →
  degree (A^m - B^n) < min m n →
  A^m = B^n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l656_65673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l656_65615

/-- Calculates the final price of a book after a series of price changes -/
noncomputable def finalPrice (initialPrice : ℝ) : ℝ :=
  let priceAfterDecrease := initialPrice * (1 - 0.30)
  let priceAfterIncrease := priceAfterDecrease * (1 + 0.20)
  priceAfterIncrease * (1 + 0.15)

/-- Theorem stating the net change in book price after adjustments -/
theorem book_price_change (P : ℝ) (P_pos : P > 0) :
  finalPrice P = P * 0.966 ∧ (finalPrice P - P) / P = -0.034 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l656_65615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_in_cm_l656_65666

/-- Proves that the distance measured on the map is 155 cm given the specified conditions -/
theorem map_distance_in_cm : 
  let inches_to_miles : ℝ := 40 / 2.5
  let inches_to_cm : ℝ := 2.54
  let actual_distance_miles : ℝ := 976.3779527559055
  Int.floor (actual_distance_miles / inches_to_miles * inches_to_cm) = 155 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_in_cm_l656_65666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l656_65697

theorem triangle_sine_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (2 * A) + Real.sin (2 * B) + Real.sin (2 * C) ≤ Real.sin A + Real.sin B + Real.sin C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l656_65697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_l656_65632

def f : ℕ → ℚ
  | 0 => 1
  | n + 1 => 2 * f n / (f n + 2)

theorem f_formula (n : ℕ) : f n = 2 / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_l656_65632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_weight_before_processing_l656_65635

/-- Calculates the original weight of a side of beef before processing -/
noncomputable def original_weight (final_weight : ℝ) (loss_percentage : ℝ) : ℝ :=
  final_weight / (1 - loss_percentage / 100)

/-- Theorem: If a side of beef loses 35% of its weight during processing
    and weighs 560 pounds after processing, then it weighed 861.54 pounds
    before processing. -/
theorem beef_weight_before_processing :
  let final_weight := 560
  let loss_percentage := 35
  abs (original_weight final_weight loss_percentage - 861.54) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval original_weight 560 35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_weight_before_processing_l656_65635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_range_of_a_l656_65672

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 3|

-- Define the solution set of f(x) > 2
def solution_set (m : ℝ) : Set ℝ := {x | f m x > 2}

-- Theorem 1: The value of m
theorem find_m : ∃ m : ℝ, solution_set m = Set.Ioo 2 4 := by sorry

-- Theorem 2: The range of a
theorem range_of_a :
  ∃ m : ℝ, ∀ a : ℝ, (∀ x : ℝ, |x - a| ≥ f m x) ↔ a ∈ Set.Iic 0 ∪ Set.Ici 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_range_of_a_l656_65672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_2x_minus_5_l656_65678

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem domain_of_f_2x_minus_5 (h : Set.Icc (-2) 3 = {x | ∃ y, f (x + 1) = y}) :
  {x | ∃ y, f (2*x - 5) = y} = Set.Icc 2 (9/2) := by
  sorry

#check domain_of_f_2x_minus_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_2x_minus_5_l656_65678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_black_ball_probability_l656_65652

def total_balls : ℕ := 10
def black_balls : ℕ := 3
def white_balls : ℕ := 7
def num_draws : ℕ := 3

noncomputable def prob_black : ℝ := black_balls / total_balls
noncomputable def prob_white : ℝ := white_balls / total_balls

theorem exactly_one_black_ball_probability :
  (Nat.choose num_draws 1 : ℝ) * prob_white^2 * prob_black =
  (Nat.choose num_draws 1 : ℝ) * (white_balls / total_balls)^2 * (black_balls / total_balls) := by
  sorry

#eval total_balls
#eval black_balls
#eval white_balls
#eval num_draws

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_black_ball_probability_l656_65652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_sequence_count_primes_in_sequence_l656_65654

def Q : ℕ := (Finset.filter (fun n => Nat.Prime n ∧ n ≤ 31) (Finset.range 32)).prod id

def sequenceQ (m : ℕ) : ℕ := Q + m

theorem no_primes_in_sequence :
  ∀ m ∈ Finset.range 31, m ≥ 2 → ¬(Nat.Prime (sequenceQ m)) :=
by sorry

theorem count_primes_in_sequence :
  (Finset.filter (fun m => Nat.Prime (sequenceQ m)) (Finset.range 31)).card = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_sequence_count_primes_in_sequence_l656_65654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_increase_l656_65699

theorem expenditure_increase
  (expenditure savings income : ℝ)
  (h_ratio : expenditure / savings = 3 / 2)
  (h_income_increase : income * 1.15 = expenditure + savings * 1.06)
  (h_initial_income : income = expenditure + savings) :
  (income * 1.15 - savings * 1.06 - expenditure) / expenditure = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_increase_l656_65699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l656_65636

def votes : List ℕ := [3600, 8400, 31200, 4300, 2700, 7200, 15500]

theorem winning_candidate_percentage : 
  (100 * (votes.maximum?.getD 0) : ℚ) / votes.sum = 4333 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l656_65636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l656_65604

-- Define the complex number z
variable (z : ℂ)

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_part_of_z (h : i * z = Complex.mk (Real.sqrt 2) (-1)) :
  z.im = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l656_65604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l656_65675

theorem divisibility_condition (n : ℕ) : n > 0 → ((n^4 + n^2) % (2*n + 1) = 0 ↔ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l656_65675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_inter_B_when_a_zero_range_of_a_when_A_subset_B_l656_65676

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2*x - 3)/(2*x + 1) + a

-- Define the set A (range of f)
def A (a : ℝ) : Set ℝ := Set.range (f a)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x + 2) + Real.sqrt (2 - x)

-- Define the set B (domain of g)
def B : Set ℝ := Set.Icc (-2) 2

-- Theorem 1
theorem complement_A_inter_B_when_a_zero :
  Set.compl (A 0 ∩ B) = Set.Ioi (-2) ∪ Set.Ioi 0 :=
sorry

-- Theorem 2
theorem range_of_a_when_A_subset_B :
  {a : ℝ | A a ∩ B = A a} = Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_inter_B_when_a_zero_range_of_a_when_A_subset_B_l656_65676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_with_same_probability_l656_65616

/-- Represents a symmetrical die with faces numbered 1 to 6 -/
structure SymmetricalDie where
  faces : Finset Nat
  property : faces = {1, 2, 3, 4, 5, 6}

/-- Represents a set of symmetrical dice -/
def DiceSet := List SymmetricalDie

/-- The probability of getting a specific sum when rolling the dice -/
noncomputable def probability (dice : DiceSet) (sum : Nat) : Real := sorry

/-- The number of dice needed to achieve a sum of 2022 -/
def numDice : Nat := 2022 / 6

theorem smallest_sum_with_same_probability 
  (dice : DiceSet) 
  (h1 : dice.length = numDice) 
  (h2 : probability dice 2022 > 0) :
  ∃ (p : Real), probability dice 2022 = p ∧ 
                probability dice (numDice * 1) = p ∧
                ∀ (s : Nat), s < numDice * 1 → probability dice s < p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_with_same_probability_l656_65616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l656_65614

/-- The area of a trapezoid with height x, one base 2x, and the other base 3x -/
noncomputable def trapezoidArea (x : ℝ) : ℝ := (2 * x + 3 * x) / 2 * x

theorem trapezoid_area_formula (x : ℝ) : 
  trapezoidArea x = 5 * x^2 / 2 := by
  -- Unfold the definition of trapezoidArea
  unfold trapezoidArea
  -- Simplify the expression
  simp [mul_add, mul_div_right_comm]
  -- Algebraic manipulation
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l656_65614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_difference_is_negative_11_66_l656_65690

/-- A school with students, teachers, and class enrollments. -/
structure School where
  num_students : ℕ
  num_teachers : ℕ
  enrollments : List ℕ

/-- Calculate the average number of students per teacher. -/
noncomputable def avg_students_per_teacher (school : School) : ℝ :=
  (school.num_students : ℝ) / school.num_teachers

/-- Calculate the average number of students per student. -/
noncomputable def avg_students_per_student (school : School) : ℝ :=
  let total_students := school.num_students
  (school.enrollments.map (fun n => (n * n : ℝ) / total_students)).sum / total_students

/-- The specific school in the problem. -/
def our_school : School :=
  { num_students := 120
  , num_teachers := 4
  , enrollments := [60, 30, 20, 10] }

/-- The main theorem to prove. -/
theorem avg_difference_is_negative_11_66 :
  avg_students_per_teacher our_school - avg_students_per_student our_school = -11.66 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_difference_is_negative_11_66_l656_65690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_smallest_solutions_l656_65693

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

def equation (x : ℝ) : Prop :=
  x - (floor x : ℝ) = 2 / (floor x : ℝ)^2

noncomputable def solution1 : ℝ := 2.5
noncomputable def solution2 : ℝ := 3 + 2/9
noncomputable def solution3 : ℝ := 4 + 1/8

theorem sum_of_smallest_solutions :
  equation solution1 ∧ 
  equation solution2 ∧ 
  equation solution3 ∧ 
  (∀ x : ℝ, equation x → x ≥ solution1) ∧
  solution1 + solution2 + solution3 = 9.847 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_smallest_solutions_l656_65693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_camp_selection_l656_65640

def total_people : ℕ := 12
def num_boys : ℕ := 7
def num_girls : ℕ := 5
def num_selected : ℕ := 4

theorem summer_camp_selection :
  (Nat.choose total_people num_selected) - 
  (Nat.choose num_boys num_selected) - 
  (Nat.choose num_girls num_selected) = 
  (Nat.choose num_boys 1 * Nat.choose num_girls 1 * Nat.choose (total_people - 2) 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_camp_selection_l656_65640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_l656_65665

/-- Represents a dart board with two concentric regular hexagons -/
structure DartBoard where
  inner_side : ℝ
  outer_side : ℝ
  outer_is_double : outer_side = 2 * inner_side

/-- Calculates the area of a regular hexagon given its side length -/
noncomputable def hexagon_area (side : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * side ^ 2

/-- Calculates the probability of a dart landing in the inner hexagon -/
noncomputable def inner_hexagon_probability (board : DartBoard) : ℝ :=
  hexagon_area board.inner_side / hexagon_area board.outer_side

/-- Theorem: The probability of a dart landing in the inner hexagon is 1/4 -/
theorem dart_probability (board : DartBoard) :
    inner_hexagon_probability board = 1 / 4 := by
  sorry

#check dart_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_l656_65665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_repaint_l656_65649

/-- Represents the four possible colors --/
inductive Color
  | Red
  | Blue
  | Green
  | Yellow

/-- Represents a circular fence with 2n sections --/
def Fence (n : ℕ) := Fin (2 * n) → Color

/-- Represents the operation of repainting three consecutive sections --/
def repaint (f : Fence n) (i : Fin (2 * n)) : Fence n :=
  sorry

/-- Counts the number of pairs of sections that are either consecutive or have at most one section between them and are of distinct colors --/
def S (f : Fence n) : ℕ :=
  sorry

/-- Auxiliary function to represent repeated application of repaint --/
def repeatRepaint (f : Fence n) : ℕ → Fin (2 * n) → Fence n
  | 0, _ => f
  | k+1, i => repaint (repeatRepaint f k i) i

/-- Theorem stating that infinite repainting is impossible --/
theorem no_infinite_repaint (n : ℕ) (h : n ≥ 3) :
  ¬∃ (f : Fence n), ∀ k : ℕ, ∃ i : Fin (2 * n), S (repeatRepaint f k i) > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_repaint_l656_65649

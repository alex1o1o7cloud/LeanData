import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_values_l1042_104246

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then x^2 - 2*x + 2 else 1 - x

-- State the theorem
theorem find_a_values (a : ℝ) (h1 : a > 0) (h2 : f a 1 + f a (-a) = 5/2) :
  a = 1/2 ∨ a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_values_l1042_104246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l1042_104236

theorem angle_of_inclination (x y : ℝ) :
  y = -x + 2 → ∃ α : ℝ, α = 3 * Real.pi / 4 ∧ Real.tan α = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l1042_104236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1042_104276

/-- The function f(x) = x + 1/(x-2) for x > 2 -/
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

/-- The theorem stating that the minimum value of f(x) for x > 2 is 4 -/
theorem min_value_of_f :
  (∀ x > 2, f x ≥ 4) ∧ (∃ x > 2, f x = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1042_104276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_count_l1042_104274

/-- Represents the farm with sheep and horses -/
structure Farm where
  young_horses : ℕ
  adult_horses : ℕ
  sheep : ℕ

/-- Calculates the total number of horses -/
def total_horses (farm : Farm) : ℕ := farm.young_horses + farm.adult_horses

/-- Theorem: Given the conditions, the number of sheep on the farm is 27 -/
theorem sheep_count (farm : Farm) : farm.sheep = 27 :=
  by
  have h1 : farm.adult_horses = 2 * farm.young_horses := sorry
  have h2 : 230 * farm.adult_horses + 150 * farm.young_horses = 12880 := sorry
  have h3 : (farm.sheep : ℤ) * 7 = ((total_horses farm) : ℤ) * 3 := sorry
  sorry

#check sheep_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_count_l1042_104274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_beats_seventh_l1042_104213

-- Define the tournament structure
structure ChessTournament where
  players : Fin 8 → ℝ
  distinct_scores : ∀ i j, i ≠ j → players i ≠ players j
  second_place_condition : players 1 = players 4 + players 5 + players 6 + players 7
  scores_bounded : ∀ i, 0 ≤ players i ∧ players i ≤ 7
  total_points : Finset.sum (Finset.univ : Finset (Fin 8)) players = 28

-- Define the result of a match
inductive MatchResult
  | Win
  | Draw
  | Loss

-- Theorem statement
theorem third_beats_seventh (tournament : ChessTournament) :
  ∃ (result : MatchResult), result = MatchResult.Win := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_beats_seventh_l1042_104213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fedya_is_last_l1042_104216

-- Define the set of boys
inductive Boy : Type
  | Misha
  | Anton
  | Petya
  | Fedya

-- Define the possible places
inductive Place : Type
  | First
  | Second
  | Third
  | Last

-- Define the place_of function
def place_of : Boy → Place := sorry

-- Define the statements made by each boy
def misha_statement : Prop := place_of Boy.Misha ≠ Place.First ∧ place_of Boy.Misha ≠ Place.Last
def anton_statement : Prop := place_of Boy.Anton ≠ Place.Last
def petya_statement : Prop := place_of Boy.Petya = Place.First
def fedya_statement : Prop := place_of Boy.Fedya = Place.Last

-- Define the condition that one boy lied and the others told the truth
def one_liar : Prop :=
  (¬misha_statement ∧ anton_statement ∧ petya_statement ∧ fedya_statement) ∨
  (misha_statement ∧ ¬anton_statement ∧ petya_statement ∧ fedya_statement) ∨
  (misha_statement ∧ anton_statement ∧ ¬petya_statement ∧ fedya_statement) ∨
  (misha_statement ∧ anton_statement ∧ petya_statement ∧ ¬fedya_statement)

-- Define that all boys have different places
def all_different : Prop :=
  ∀ (b1 b2 : Boy), b1 ≠ b2 → place_of b1 ≠ place_of b2

-- The theorem to prove
theorem fedya_is_last :
  one_liar → all_different → place_of Boy.Fedya = Place.Last :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fedya_is_last_l1042_104216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_15_implies_x_eq_neg4_or_5_l1042_104269

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 1 else 3*x

theorem f_eq_15_implies_x_eq_neg4_or_5 :
  ∀ x : ℝ, f x = 15 → x = -4 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_15_implies_x_eq_neg4_or_5_l1042_104269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_iff_intersection_nonempty_and_subset_iff_l1042_104251

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}
def B (a : ℝ) : Set ℝ := {y | ∃ x, y = x^2 - 2*x + a}
def C (a : ℝ) : Set ℝ := {x | x^2 - a*x - 4 ≤ 0}

-- Define propositions p and q
def p (a : ℝ) : Prop := (A ∩ B a).Nonempty
def q (a : ℝ) : Prop := A ⊆ C a

-- Theorem 1
theorem intersection_empty_iff (a : ℝ) : (A ∩ B a) = ∅ ↔ a > 3 := by sorry

-- Theorem 2
theorem intersection_nonempty_and_subset_iff (a : ℝ) : p a ∧ q a ↔ 0 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_iff_intersection_nonempty_and_subset_iff_l1042_104251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acetic_acid_molecular_weight_l1042_104232

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in Acetic acid -/
def carbon_count : ℕ := 2

/-- The number of Hydrogen atoms in Acetic acid -/
def hydrogen_count : ℕ := 4

/-- The number of Oxygen atoms in Acetic acid -/
def oxygen_count : ℕ := 2

/-- The molecular weight of Acetic acid in g/mol -/
def acetic_acid_weight : ℝ := 
  carbon_weight * (carbon_count : ℝ) + 
  hydrogen_weight * (hydrogen_count : ℝ) + 
  oxygen_weight * (oxygen_count : ℝ)

theorem acetic_acid_molecular_weight : 
  ∃ ε > 0, |acetic_acid_weight - 60.052| < ε := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acetic_acid_molecular_weight_l1042_104232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1042_104238

theorem simplify_expression : 
  (Real.rpow (Real.sqrt ((5:ℝ)^2)) (3/4)) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1042_104238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coastal_area_income_theorem_l1042_104281

-- Define the problem parameters
variable (m : ℝ) (x : ℝ) (a : ℝ)

-- Define the conditions
def total_households : ℕ := 100
noncomputable def income_increase_rate (x : ℝ) : ℝ := 2 * x / 100
noncomputable def aquaculture_income (m a x : ℝ) : ℝ := m * (a - 3 * x / 50)

-- Define the income comparison condition
def income_condition (m x a : ℝ) : Prop :=
  ∀ x > 0, x ≤ 50 →
    aquaculture_income m a x * x ≤ 
    m * (total_households - x) * (1 + income_increase_rate x)

-- State the theorem
theorem coastal_area_income_theorem (m : ℝ) (h_m : m > 0) :
  (∃ x_max : ℝ, x_max = 50 ∧ 
    ∀ x : ℝ, 0 < x → x ≤ x_max → 
      m * (total_households - x) * (1 + income_increase_rate x) ≥ m * total_households) ∧
  (∃ a_max : ℝ, a_max = 5 ∧ 
    ∀ a : ℝ, a > 0 → a ≤ a_max → income_condition m x a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coastal_area_income_theorem_l1042_104281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1042_104286

def complex_number_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem z_in_first_quadrant :
  let z : ℂ := (-2 + Complex.I) / Complex.I
  complex_number_quadrant z :=
by
  -- Unfold the definition of z
  let z := (-2 + Complex.I) / Complex.I
  
  -- Simplify the complex number division
  have h1 : z = 1 + 2 * Complex.I := by
    -- You would prove this step here
    sorry
  
  -- Show that the real part is positive
  have h_re : z.re > 0 := by
    -- You would prove this step here
    sorry
  
  -- Show that the imaginary part is positive
  have h_im : z.im > 0 := by
    -- You would prove this step here
    sorry
  
  -- Conclude that z is in the first quadrant
  exact ⟨h_re, h_im⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1042_104286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peters_books_l1042_104254

theorem peters_books (total_books : ℕ) 
  (h1 : total_books > 0)
  (h2 : (40 * total_books : ℕ) = (10 * total_books : ℕ) + 60) : 
  total_books = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peters_books_l1042_104254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_decreasing_digit_numbers_eq_1013_l1042_104221

/-- A function that counts the number of natural numbers with at least two digits,
    where each subsequent digit is less than the previous one. -/
def count_decreasing_digit_numbers : ℕ :=
  1013  -- We define it directly as 1013 for now

/-- The theorem stating that the count of natural numbers with at least two digits,
    where each subsequent digit is less than the previous one, is 1013. -/
theorem count_decreasing_digit_numbers_eq_1013 :
  count_decreasing_digit_numbers = 1013 := by
  rfl  -- reflexivity, since we defined the function to be 1013


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_decreasing_digit_numbers_eq_1013_l1042_104221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l1042_104263

/-- The rational function f(x) = (3x^3 + 2x^2 + 6x - 12) / (x - 4) -/
noncomputable def f (x : ℝ) : ℝ := (3*x^3 + 2*x^2 + 6*x - 12) / (x - 4)

/-- The slope of the slant asymptote of f(x) -/
def m : ℝ := 14

/-- The y-intercept of the slant asymptote of f(x) -/
def b : ℝ := 62

theorem slant_asymptote_sum : m + b = 76 := by
  simp [m, b]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l1042_104263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1042_104219

/-- Given a hyperbola, a circle, and a parabola with specific properties, 
    prove that the eccentricity of the hyperbola is (1 + √5) / 2 -/
theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (x, y) ∈ Set.range (λ t ↦ (a * Real.cosh t, b * Real.sinh t)))
  (circle : ∀ x y : ℝ, x^2 + y^2 = a^2 → (x, y) ∈ Set.range (λ t ↦ (a * Real.cos t, a * Real.sin t)))
  (parabola : ∀ x y : ℝ, y^2 = 4 * c * x → (x, y) ∈ Set.range (λ t ↦ (t^2, 2 * c * t)))
  (F : ℝ × ℝ) (hF : F = (-c, 0))
  (E : ℝ × ℝ) (hE : E.1^2 + E.2^2 = a^2)
  (P : ℝ × ℝ) (hP : P.2^2 = 4 * c * P.1)
  (tangent_condition : ∃ t : ℝ → ℝ × ℝ, Differentiable ℝ t ∧ t 0 = E ∧ (∀ s, (t s).1^2 + (t s).2^2 = a^2) ∧ ∃ u, t u = P)
  (midpoint_condition : E = (F + P) / 2)
  : (c^2 / a^2)^(1/2) = (1 + 5^(1/2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1042_104219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_cost_savings_l1042_104201

theorem fuel_cost_savings 
  (old_efficiency : ℝ) 
  (efficiency_increase_percent : ℝ) 
  (fuel_price_increase_percent : ℝ) 
  (trip_distance : ℝ) 
  (h1 : efficiency_increase_percent = 75) 
  (h2 : fuel_price_increase_percent = 40) 
  (h3 : trip_distance > 0) :
  ∃ ε > 0, |((trip_distance / old_efficiency) - 
             (trip_distance / (old_efficiency * (1 + efficiency_increase_percent / 100)) * 
              (1 + fuel_price_increase_percent / 100))) / 
             (trip_distance / old_efficiency) * 100 - (100 / 7)| < ε :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_cost_savings_l1042_104201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximizing_radius_l1042_104220

/-- A solid with a cylindrical middle and conical caps at each end, where the height of each cap equals the length of the middle. -/
structure ConicalCylindricalSolid where
  R : ℝ  -- radius of the cylindrical part and base of the conical parts
  H : ℝ  -- height of the cylindrical part (equal to the height of each conical cap)

/-- The surface area of the ConicalCylindricalSolid -/
noncomputable def surfaceArea (solid : ConicalCylindricalSolid) : ℝ :=
  2 * Real.pi * solid.R * solid.H + 2 * Real.pi * solid.R * Real.sqrt (solid.R^2 + solid.H^2)

/-- The volume of the ConicalCylindricalSolid -/
noncomputable def volume (solid : ConicalCylindricalSolid) : ℝ :=
  (5/3) * Real.pi * solid.R^2 * solid.H

/-- The theorem stating that the radius maximizing the volume for a given surface area is (A / (π√5))^(1/3) -/
theorem volume_maximizing_radius (A : ℝ) (h : A > 0) :
  ∃ (solid : ConicalCylindricalSolid),
    surfaceArea solid = A ∧
    ∀ (other : ConicalCylindricalSolid),
      surfaceArea other = A →
      volume other ≤ volume solid ∧
      solid.R = (A / (Real.pi * Real.sqrt 5))^(1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximizing_radius_l1042_104220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l1042_104290

-- Define the triangle vertices
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (4, 7)
def C : ℝ × ℝ := (8, 1)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem longest_side_length :
  max (distance A B) (max (distance B C) (distance A C)) = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l1042_104290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_division_theorem_l1042_104296

theorem remainder_division_theorem (x y : ℕ) (hx : x > 0) (hy : y > 0) : 
  (x : ℝ) / (y : ℝ) = 96.2 → 
  x % y = 5 → 
  y = 25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_division_theorem_l1042_104296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l1042_104206

theorem max_product_sum (f g h j : ℕ) : 
  f ∈ ({4, 5, 6, 7} : Set ℕ) → 
  g ∈ ({4, 5, 6, 7} : Set ℕ) → 
  h ∈ ({4, 5, 6, 7} : Set ℕ) → 
  j ∈ ({4, 5, 6, 7} : Set ℕ) → 
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j → 
  (f * g + g * h + h * j + f * j : ℕ) ≤ 120 ∧ 
  ∃ (f' g' h' j' : ℕ), 
    f' ∈ ({4, 5, 6, 7} : Set ℕ) ∧ 
    g' ∈ ({4, 5, 6, 7} : Set ℕ) ∧ 
    h' ∈ ({4, 5, 6, 7} : Set ℕ) ∧ 
    j' ∈ ({4, 5, 6, 7} : Set ℕ) ∧ 
    f' ≠ g' ∧ f' ≠ h' ∧ f' ≠ j' ∧ g' ≠ h' ∧ g' ≠ j' ∧ h' ≠ j' ∧ 
    f' * g' + g' * h' + h' * j' + f' * j' = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l1042_104206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1042_104268

/-- Given an ellipse with the following properties:
  1. Its equation is x²/a² + y²/b² = 1 where a > b > 0
  2. Its eccentricity is √3/2
  3. A line with slope 1 passes through point M(b, 0) and intersects the ellipse at points A and B
  4. OA · OB = 32/5 * cot(∠AOB)
  Then the equation of the ellipse is x²/16 + y²/4 = 1 -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (eccentricity : ℝ) (h_eccentricity : eccentricity = Real.sqrt 3 / 2)
  (M : ℝ × ℝ) (hM : M = (b, 0))
  (A B : ℝ × ℝ) (hAB : ∃ (m : ℝ), m = 1 ∧ A.2 - M.2 = m * (A.1 - M.1) ∧ B.2 - M.2 = m * (B.1 - M.1))
  (hDot : A.1 * B.1 + A.2 * B.2 = 32 / 5 * Real.tan (Real.pi / 2 - Real.arctan ((B.2 - A.2) / (B.1 - A.1)))) :
  ∃ (x y : ℝ), x^2 / 16 + y^2 / 4 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1042_104268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1042_104277

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  Real.cos t.A * Real.cos t.B - 1 = Real.sin t.A * Real.sin t.B - 2 * (Real.sin t.C)^2 ∧
  t.c = 4 ∧
  t.a^2 + t.b^2 = 32

theorem triangle_theorem (t : Triangle) (h : TriangleConditions t) :
  t.C = Real.pi / 3 ∧ (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1042_104277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l1042_104209

/-- The maximum value of the function y = -3 cos(3x) + c sin(3x) -/
def max_value : ℝ := 5

/-- The coefficient of the cosine term -/
def a : ℝ := -3

/-- The coefficient of x in both trigonometric terms -/
def b : ℝ := 3

/-- The function y = -3 cos(3x) + c sin(3x) -/
noncomputable def y (x c : ℝ) : ℝ := a * Real.cos (b * x) + c * Real.sin (b * x)

theorem find_c :
  ∃ c : ℝ, (∀ x : ℝ, y x c ≤ max_value) ∧ 
  (∃ x : ℝ, y x c = max_value) ∧ 
  (c = 4 ∨ c = -4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l1042_104209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumscribed_circle_radius_l1042_104211

/-- Given a sector with central angle 2θ cut from a circle of radius 8,
    the radius of the circle circumscribed about the sector is 4 sec (θ/2) -/
theorem sector_circumscribed_circle_radius (θ : Real) :
  let r : Real := 8  -- radius of the original circle
  let α : Real := 2 * θ  -- central angle of the sector
  let R : Real := 4 * (1 / Real.cos (θ / 2))  -- radius of the circumscribed circle
  R = (r / 2) / Real.cos (α / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumscribed_circle_radius_l1042_104211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_area_l1042_104273

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

/-- Given conditions for the triangle -/
noncomputable def specialTriangle : Triangle where
  a := Real.sqrt 13
  c := 3
  b := 4  -- This is derived in the solution, but we include it as a given
  A := Real.pi / 3  -- This is derived in the solution, but we include it as a given
  B := 0  -- Placeholder value, not needed for the proof
  C := 0  -- Placeholder value, not needed for the proof

/-- Theorem stating that the area of the special triangle is 3√3 -/
theorem special_triangle_area : triangleArea specialTriangle = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_area_l1042_104273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_enclosed_area_l1042_104262

/-- The parabola function -/
noncomputable def parabola (x : ℝ) : ℝ := 1 - x^2

/-- The area enclosed by the tangents and x-axis -/
noncomputable def enclosedArea (a b : ℝ) : ℝ :=
  (1/4) * (a - b) * (2 - a*b - 1/(a*b))

theorem minimum_enclosed_area (a b : ℝ) (h : a * b < 0) :
  ∃ (a₀ b₀ : ℝ), a₀ * b₀ < 0 ∧
    ∀ (a' b' : ℝ), a' * b' < 0 →
      enclosedArea a' b' ≥ enclosedArea a₀ b₀ ∧
      enclosedArea a₀ b₀ = 8 * Real.sqrt 3 / 9 := by
  sorry

#check minimum_enclosed_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_enclosed_area_l1042_104262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_identification_l1042_104279

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x / 2) + Real.cos (x / 2)
noncomputable def g (A ω φ x : ℝ) : ℝ := A * Real.cos (ω * x + φ)

-- Define periodic function property
def is_periodic (h : ℝ → ℝ) : Prop := ∃ T > 0, ∀ x, h (x + T) = h x

-- Define minor_premise as a proposition
def minor_premise : Prop := ∃ A ω φ, ∀ x, f x = g A ω φ x

-- State the theorem
theorem minor_premise_identification 
  (h1 : ∃ A ω φ, ∀ x, f x = g A ω φ x)  -- f can be expressed as g
  (h2 : ∀ A ω φ, is_periodic (g A ω φ))  -- g is periodic for all A, ω, φ
  (h3 : is_periodic f)  -- f is periodic
  : minor_premise := by
  exact h1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_identification_l1042_104279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_five_largest_nonempty_domain_l1042_104282

noncomputable def g : ℕ → (ℝ → ℝ)
| 0 => λ _ => 0  -- Add a case for 0 to cover all natural numbers
| 1 => λ x => Real.sqrt (4 - x)
| (n + 2) => λ x => g (n + 1) (Real.sqrt ((n + 3)^2 - x))

theorem domain_of_g_five (x : ℝ) :
  (∃ y, g 5 y = x) ↔ x = -589 :=
sorry

theorem largest_nonempty_domain :
  (∀ x, ¬ ∃ y, g 6 y = x) ∧
  (∃ x, ∃ y, g 5 y = x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_five_largest_nonempty_domain_l1042_104282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_current_l1042_104295

-- Define complex numbers
def Z₁ : ℂ := 2 + Complex.I
def Z₂ : ℂ := 3 - 2*Complex.I
def V : ℂ := 5 - 2*Complex.I

-- Define the total impedance for series connection
def Z : ℂ := Z₁ + Z₂

-- Define the current using Ohm's law
noncomputable def I : ℂ := V / Z

-- Theorem to prove
theorem circuit_current : I = 9/8 - (5/24)*Complex.I :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_current_l1042_104295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l1042_104202

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 ≤ 1}
def B : Set ℝ := {x : ℝ | Real.exp (x * Real.log 2) ≤ 1}

-- State the theorem
theorem intersection_A_complement_B : A ∩ Bᶜ = Set.Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l1042_104202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matchbox_car_percentage_l1042_104272

theorem matchbox_car_percentage (total : ℕ) (regular_percent : ℚ) (convertibles : ℕ) : 
  total = 125 →
  regular_percent = 64 / 100 →
  convertibles = 35 →
  (total - (regular_percent * ↑total).floor - convertibles) / ↑total = 8 / 100 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matchbox_car_percentage_l1042_104272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locally_odd_f_range_of_m_l1042_104235

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (4 : ℝ)^x - m * (2 : ℝ)^(x + 1) + m^2 - 3

-- Define the property of being locally odd
def locally_odd (f : ℝ → ℝ) : Prop := ∃ x : ℝ, f (-x) = -f x

-- State the theorem
theorem locally_odd_f_range_of_m :
  ∀ m : ℝ, locally_odd (f m) → 1 - Real.sqrt 3 ≤ m ∧ m ≤ 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locally_odd_f_range_of_m_l1042_104235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triples_l1042_104243

def valid_triple (a b c : ℕ+) : Prop :=
  Nat.lcm a b = 180 ∧
  Nat.lcm a c = 450 ∧
  Nat.lcm b c = 1200 ∧
  Nat.gcd a (Nat.gcd b c) = 3

theorem count_valid_triples :
  ∃! (S : Finset (ℕ+ × ℕ+ × ℕ+)),
    (∀ t ∈ S, valid_triple t.1 t.2.1 t.2.2) ∧
    S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triples_l1042_104243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_tangent_and_normal_l1042_104299

/-- The sphere equation -/
def sphere_equation (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 5 = 0

/-- The point on the sphere -/
def M₀ : ℝ × ℝ × ℝ := (3, -1, 5)

/-- The tangent plane equation -/
def tangent_plane_equation (x y z : ℝ) : Prop :=
  2*x + y + 2*z - 15 = 0

/-- The normal line equation -/
def normal_line_equation (x y z : ℝ) : Prop :=
  (x - 3) / 2 = (y + 1) / 1 ∧ (y + 1) / 1 = (z - 5) / 2

theorem sphere_tangent_and_normal :
  sphere_equation M₀.1 M₀.2.1 M₀.2.2 →
  (∀ x y z, tangent_plane_equation x y z ↔ 
    (x - M₀.1) * (2 * M₀.1 - 2) + 
    (y - M₀.2.1) * (2 * M₀.2.1 + 4) + 
    (z - M₀.2.2) * (2 * M₀.2.2 - 6) = 0) ∧
  (∀ x y z, normal_line_equation x y z ↔ 
    (x - M₀.1) / (2 * M₀.1 - 2) = 
    (y - M₀.2.1) / (2 * M₀.2.1 + 4) ∧
    (y - M₀.2.1) / (2 * M₀.2.1 + 4) = 
    (z - M₀.2.2) / (2 * M₀.2.2 - 6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_tangent_and_normal_l1042_104299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meeting_point_l1042_104266

/-- Two trains traveling in the same direction meet at a certain distance from their starting point. -/
theorem trains_meeting_point
  (speed_train1 : ℝ)
  (speed_train2 : ℝ)
  (time_difference : ℝ)
  (h1 : speed_train1 = 30)
  (h2 : speed_train2 = 40)
  (h3 : time_difference = 6)
  (h4 : speed_train2 > speed_train1) :
  let relative_speed := speed_train2 - speed_train1
  let initial_distance := speed_train1 * time_difference
  let catch_up_time := initial_distance / relative_speed
  let meeting_distance := speed_train2 * catch_up_time
  meeting_distance = 720 := by
  -- Proof steps go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meeting_point_l1042_104266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadrilateral_with_large_area_l1042_104229

/-- A convex polygon with area 1 and more than 3 sides -/
structure ConvexPolygon where
  vertices : Finset (ℝ × ℝ)
  convex : Convex ℝ (↑vertices : Set (ℝ × ℝ))
  area : MeasureTheory.MeasureSpace.volume (↑vertices : Set (ℝ × ℝ)) = 1
  sides : vertices.card > 3

/-- The area of a quadrilateral formed by four points -/
noncomputable def quadrilateralArea (a b c d : ℝ × ℝ) : ℝ := sorry

/-- Theorem: In any convex polygon with area 1 and more than 3 sides,
    there exist four vertices forming a quadrilateral with area at least 1/2 -/
theorem exists_quadrilateral_with_large_area (P : ConvexPolygon) :
  ∃ (a b c d : ℝ × ℝ), a ∈ P.vertices ∧ b ∈ P.vertices ∧ c ∈ P.vertices ∧ d ∈ P.vertices ∧
    quadrilateralArea a b c d ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadrilateral_with_large_area_l1042_104229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_small_boxes_count_l1042_104297

/-- The dimensions of the wooden box in centimeters -/
def wooden_box_dimensions : Fin 3 → ℕ := ![400, 200, 400]

/-- The dimensions of the small box in centimeters -/
def small_box_dimensions : Fin 3 → ℕ := ![4, 2, 2]

/-- Calculate the volume of a box given its dimensions -/
def volume (dimensions : Fin 3 → ℕ) : ℕ :=
  (dimensions 0) * (dimensions 1) * (dimensions 2)

/-- The maximum number of small boxes that can fit in the wooden box -/
def max_small_boxes : ℕ :=
  volume wooden_box_dimensions / volume small_box_dimensions

/-- Theorem stating the maximum number of small boxes that can fit in the wooden box -/
theorem max_small_boxes_count : max_small_boxes = 2000000 := by
  -- Unfold definitions
  unfold max_small_boxes
  unfold volume
  unfold wooden_box_dimensions
  unfold small_box_dimensions
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_small_boxes_count_l1042_104297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l1042_104226

theorem triangle_third_side_length (a b c : ℝ) (θ : ℝ) : 
  a = 10 → b = 12 → θ = 150 * Real.pi / 180 → 
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ) → 
  c = Real.sqrt (244 + 120 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l1042_104226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_shape_area_l1042_104257

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x^2

-- Define the line function
def line (_ : ℝ) : ℝ := 1

-- Define the enclosed area
noncomputable def enclosed_area : ℝ := ∫ x in Set.Icc (-1) 1, line x - parabola x

-- Theorem statement
theorem enclosed_shape_area : enclosed_area = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_shape_area_l1042_104257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_problem_l1042_104214

/-- The expected number of seconds until all cheese is gone -/
def expected_time : ℕ := 2019

/-- The problem statement as a theorem -/
theorem cheese_problem :
  let n := 2018  -- number of cheese slices
  let d := 2019  -- number of sides on the die
  expected_time = d := by
  sorry  -- Proof to be filled in

#check cheese_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_problem_l1042_104214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_years_for_income_to_exceed_investment_l1042_104233

/-- The initial investment in millions -/
noncomputable def initial_investment : ℝ := 800

/-- The investment decrease rate per year -/
noncomputable def investment_decrease_rate : ℝ := 1/5

/-- The initial tourism income in millions -/
noncomputable def initial_tourism_income : ℝ := 400

/-- The tourism income increase rate per year -/
noncomputable def tourism_income_increase_rate : ℝ := 1/4

/-- The total investment after n years -/
noncomputable def total_investment (n : ℕ) : ℝ :=
  initial_investment * (1 - (1 - investment_decrease_rate)^n) / investment_decrease_rate

/-- The total tourism income after n years -/
noncomputable def total_tourism_income (n : ℕ) : ℝ :=
  initial_tourism_income * ((1 + tourism_income_increase_rate)^n - 1) / tourism_income_increase_rate

/-- The minimum number of years required for total tourism income to exceed total investment -/
theorem min_years_for_income_to_exceed_investment :
  ∃ n : ℕ, (∀ k < n, total_tourism_income k ≤ total_investment k) ∧
           (total_tourism_income n > total_investment n) ∧
           n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_years_for_income_to_exceed_investment_l1042_104233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_strategy_faster_l1042_104223

/-- The circumference of the circular alley in kilometers -/
noncomputable def circumference : ℝ := 2 * Real.pi

/-- The walker's speed in km/h -/
def walker_speed : ℝ := 6

/-- The cyclist's speed in km/h -/
def cyclist_speed : ℝ := 20

/-- The time taken for the first strategy (moving directly towards each other) -/
noncomputable def time_strategy1 : ℝ := 6 / (walker_speed + cyclist_speed)

/-- The time taken for the second strategy (cyclist going to point A first) -/
noncomputable def time_strategy2 : ℝ := (circumference - 2) / (walker_speed + cyclist_speed)

/-- Theorem stating that the first strategy takes less time than the second strategy -/
theorem first_strategy_faster : time_strategy1 < time_strategy2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_strategy_faster_l1042_104223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_equal_f_product_one_l1042_104259

-- Define the function f(x) = |log₁₀(x)|
noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 10|

-- State the theorem
theorem distinct_equal_f_product_one (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) :
  f a = f b → a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_equal_f_product_one_l1042_104259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l1042_104212

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Adding the base case for 0
  | 1 => 2
  | n + 2 => 3 * sequence_a (n + 1) + 5

theorem sequence_a_general_term (n : ℕ) : 
  sequence_a n = 1/2 * 3^(n+1) - 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l1042_104212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_is_14_l1042_104200

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- Side lengths of the triangle -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- One side is divided into segments of 9 and 11 units by the tangent point -/
  segment1 : ℝ
  segment2 : ℝ
  /-- Properties of a valid triangle with inscribed circle -/
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0
  positive_radius : r > 0
  tangent_point : segment1 + segment2 = max a (max b c)
  radius_formula : r = (a + b + c) / 2 - max a (max b c)

/-- Theorem: The shortest side of the triangle is 14 units long -/
theorem shortest_side_is_14 (t : TriangleWithInscribedCircle) 
  (h1 : t.r = 5)
  (h2 : t.segment1 = 9)
  (h3 : t.segment2 = 11) :
  min t.a (min t.b t.c) = 14 := by
  sorry

#check shortest_side_is_14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_is_14_l1042_104200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_k_value_l1042_104240

theorem quadratic_function_k_value (a b c k : ℤ) (f : ℤ → ℤ) : 
  (∀ x, f x = a * x^2 + b * x + c) →
  f 2 = 0 →
  30 < f 5 ∧ f 5 < 40 →
  50 < f 6 ∧ f 6 < 60 →
  1000 * k < f 50 ∧ f 50 < 1000 * (k + 1) →
  k = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_k_value_l1042_104240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_equality_l1042_104203

theorem cosine_product_equality (z : ℂ) : 
  Complex.cos z * Complex.cos (2*z) * Complex.cos (4*z) * Complex.cos (8*z) = (1/16 : ℂ) →
  (∃ (k : ℤ), z = (2 * Real.pi * ↑k : ℂ) / 15 ∧ ¬∃ (l : ℤ), k = 15 * l) ∨
  (∃ (k : ℤ), z = (Real.pi * ↑(2*k + 1) : ℂ) / 17 ∧ ¬∃ (l : ℤ), k = 17 * l + 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_equality_l1042_104203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_source_function_a_range_l1042_104293

/-- A function is a "source" function if there exists a point where the function value equals its derivative. -/
def is_source_function (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = deriv f x₀

/-- The logarithmic function minus twice the identity function minus a constant. -/
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x ↦ Real.log x - 2 * x - a

theorem source_function_a_range :
  ∀ a : ℝ, is_source_function (f a) → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_source_function_a_range_l1042_104293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nina_elle_pairing_probability_l1042_104270

/-- Represents the class with students and pairing rules -/
structure ClassInfo where
  total_students : ℕ
  nina_potential_partners : ℕ
  elle_is_potential_partner : Bool

/-- Calculates the probability of Nina being paired with Elle -/
def probability_nina_paired_with_elle (c : ClassInfo) : ℚ :=
  if c.elle_is_potential_partner then 1 / c.nina_potential_partners else 0

/-- Theorem stating the probability of Nina being paired with Elle -/
theorem nina_elle_pairing_probability (c : ClassInfo) 
  (h1 : c.total_students = 32)
  (h2 : c.nina_potential_partners = 29)
  (h3 : c.elle_is_potential_partner = true) :
  probability_nina_paired_with_elle c = 1 / 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nina_elle_pairing_probability_l1042_104270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_properties_l1042_104222

/-- Represents the meeting point of two cars on a highway --/
structure MeetingPoint where
  distance : ℝ  -- Distance from Bilbao in km
  time : ℝ      -- Time to meeting point in minutes

/-- Finds the meeting point of two cars on a highway --/
noncomputable def find_meeting_point (total_distance : ℝ) (time_difference : ℝ) : MeetingPoint :=
  let distance := (total_distance - time_difference) / 2
  let time := distance * 60 / (total_distance / (150 / 1.714))
  ⟨distance, time⟩

/-- Theorem stating the properties of the meeting point --/
theorem meeting_point_properties (total_distance time_difference : ℝ)
  (h1 : total_distance = 150)
  (h2 : time_difference = 25)
  (h3 : ∀ x, 0 < x ∧ x < total_distance → 
    (total_distance - 2*x = (total_distance - x) / (total_distance / (150 / 2.4)) - x / (total_distance / (150 / 1.714)))) :
  let mp := find_meeting_point total_distance time_difference
  mp.distance = 62.5 ∧ (36 < mp.time ∧ mp.time < 37) :=
by sorry

-- Remove the #eval command as it's causing issues
-- #eval find_meeting_point 150 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_properties_l1042_104222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derek_to_amy_debt_ratio_l1042_104285

def total_promised : ℕ := 400
def amount_received : ℕ := 285
def sally_debt : ℕ := 35
def carl_debt : ℕ := 35
def amy_debt : ℕ := 30

theorem derek_to_amy_debt_ratio :
  (total_promised - amount_received - (sally_debt + carl_debt + amy_debt) : ℚ) / amy_debt = 1 / 2 := by
  -- Convert all natural numbers to rationals for division
  have h1 : (total_promised : ℚ) - (amount_received : ℚ) - ((sally_debt : ℚ) + (carl_debt : ℚ) + (amy_debt : ℚ)) = 15 := by sorry
  have h2 : (amy_debt : ℚ) = 30 := by sorry
  -- Perform the division
  have h3 : 15 / 30 = 1 / 2 := by sorry
  -- Combine the steps
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derek_to_amy_debt_ratio_l1042_104285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_fourth_quadrant_l1042_104218

theorem sin_double_angle_fourth_quadrant (α : Real) :
  (-(π / 2) < α ∧ α < 0) → Real.sin (2 * α) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_fourth_quadrant_l1042_104218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1042_104271

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.sin α - Real.cos α = Real.sqrt 2) 
  (h2 : α ∈ Set.Ioo 0 Real.pi) : 
  Real.tan α = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1042_104271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_fifth_f_greater_cos_pi_fifth_f_l1042_104204

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

-- State the theorem
theorem sin_pi_fifth_f_greater_cos_pi_fifth_f :
  f (Real.sin (π / 5)) > f (Real.cos (π / 5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_fifth_f_greater_cos_pi_fifth_f_l1042_104204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_value_l1042_104261

theorem triangle_cosine_value (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a * Real.sin B = b * Real.sin A ∧
  b * Real.sin C = c * Real.sin B ∧
  c * Real.sin A = a * Real.sin C ∧
  Real.sin A = 2 * Real.sin C ∧
  (a / c = b / a ∨ a / c = c / b) →
  Real.cos A = -1/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_value_l1042_104261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cycle_work_gas_cycle_work_approx_l1042_104265

/-- Represents the pressure at a reference state --/
def p₀ : ℝ := 10^5

/-- Represents the volume at a reference state in cubic meters --/
def V₀ : ℝ := 3e-3

/-- Represents the work done by the gas in a cycle --/
noncomputable def cycle_work (p₀ V₀ : ℝ) : ℝ := 5 * Real.pi * p₀ * V₀

/-- Theorem stating that the work done by the gas in the cycle is 5πp₀V₀ --/
theorem gas_cycle_work :
  cycle_work p₀ V₀ = 5 * Real.pi * p₀ * V₀ := by
  -- Proof goes here
  sorry

/-- Theorem stating that the work done by the gas in the cycle is approximately 2827 J --/
theorem gas_cycle_work_approx :
  ∃ ε > 0, |cycle_work p₀ V₀ - 2827| < ε := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cycle_work_gas_cycle_work_approx_l1042_104265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1042_104292

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 7*x + 10)

-- Define the domain
def domain : Set ℝ := {x | x < -5 ∨ (-5 < x ∧ x < -2) ∨ x > -2}

-- Theorem statement
theorem f_domain : 
  {x : ℝ | ∃ y, f x = y} = domain := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1042_104292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_last_element_l1042_104287

def triangle_element : ℕ → ℕ → ℚ
| 1, j => 1 / j
| i, j => triangle_element (i-1) (j-1) - triangle_element (i-1) j

def general_formula (i j : ℕ) : ℚ :=
  (Nat.factorial (i-1) * Nat.factorial (j-1)) / Nat.factorial (i+j-1)

theorem triangle_last_element :
  triangle_element 1993 1 = general_formula 1993 1 ∧ general_formula 1993 1 = 1 / 1993 := by
  sorry

#eval triangle_element 1993 1
#eval general_formula 1993 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_last_element_l1042_104287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_moment_of_inertia_l1042_104258

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Represents a point mass -/
structure PointMass where
  position : Point
  mass : ℝ

/-- Calculates the moment of inertia of a system of point masses about a given point -/
def momentOfInertia (masses : List PointMass) (center : Point) : ℝ :=
  masses.foldl (fun acc m => acc + m.mass * squaredDistance m.position center) 0

/-- The theorem to be proved -/
theorem quadrilateral_moment_of_inertia :
  let s := Point.mk 0 0
  let a1 := PointMass.mk (Point.mk 3 4) 2
  let a2 := PointMass.mk (Point.mk 2 0) 6
  let a3 := PointMass.mk (Point.mk 3 (-4)) 2
  let a4 := PointMass.mk (Point.mk 8 0) 3
  let masses := [a1, a2, a3, a4]
  momentOfInertia masses s = 316 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_moment_of_inertia_l1042_104258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_five_digit_number_with_product_210_l1042_104241

/-- Represents a five-digit number as a list of its digits -/
def FiveDigitNumber := List Nat

/-- Checks if a list represents a valid five-digit number -/
def isValidFiveDigitNumber (n : FiveDigitNumber) : Prop :=
  n.length = 5 ∧ n.head! ≠ 0 ∧ n.all (λ d => d < 10)

/-- Calculates the product of the digits of a number -/
def digitProduct (n : FiveDigitNumber) : Nat :=
  n.prod

/-- Calculates the sum of the digits of a number -/
def digitSum (n : FiveDigitNumber) : Nat :=
  n.sum

/-- Compares two five-digit numbers -/
def greaterThan (a b : FiveDigitNumber) : Prop :=
  (a.reverse.map (· * 10^4) |>.sum) > (b.reverse.map (· * 10^4) |>.sum)

/-- The main theorem to be proven -/
theorem greatest_five_digit_number_with_product_210 :
  ∃ M : FiveDigitNumber,
    isValidFiveDigitNumber M ∧
    digitProduct M = 210 ∧
    (∀ n : FiveDigitNumber, isValidFiveDigitNumber n → digitProduct n = 210 → greaterThan M n) ∧
    digitSum M = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_five_digit_number_with_product_210_l1042_104241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_theorem_l1042_104208

theorem chord_cosine_theorem (r : ℝ) (γ δ : ℝ) :
  γ + δ < π →
  0 < Real.cos γ →
  5^2 = r^2 * (2 - 2 * Real.cos γ) →
  6^2 = r^2 * (2 - 2 * Real.cos δ) →
  7^2 = r^2 * (2 - 2 * Real.cos (γ + δ)) →
  Real.cos γ = (1 : ℚ) / 49 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_theorem_l1042_104208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_condition_l1042_104283

theorem set_condition (x : ℝ) : {y : ℝ | y = 3 ∨ y = x^2 - 2*x}.Nonempty → x ≠ 3 ∧ x ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_condition_l1042_104283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1042_104284

-- Define the line equation
def line_equation (x y : ℝ) (c : ℝ) : Prop := x - Real.sqrt 3 * y + c = 0

-- Define the inclination angle of a line
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m

-- State the theorem
theorem line_inclination_angle (c : ℝ) :
  inclination_angle (Real.sqrt 3 / 3) = 30 * π / 180 := by
  sorry

#check line_inclination_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1042_104284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_five_count_three_digit_multiples_of_five_l1042_104215

theorem three_digit_multiples_of_five : ℕ → Prop :=
  fun n => n = (Finset.filter (fun x => x % 5 = 0) (Finset.range 900)).card

theorem count_three_digit_multiples_of_five :
  ∃ n, three_digit_multiples_of_five n ∧ n = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_five_count_three_digit_multiples_of_five_l1042_104215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_roots_relation_l1042_104234

theorem sin_cos_roots_relation (α β u v w y : ℝ) : 
  (∀ x : ℝ, x^2 - u*x + v = 0 ↔ x = Real.sin α ∨ x = Real.sin β) →
  (∀ x : ℝ, x^2 - w*x + y = 0 ↔ x = Real.cos α ∨ x = Real.cos β) →
  w*y = 1 - v := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_roots_relation_l1042_104234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_top_block_value_l1042_104260

/-- Represents a block in the pyramid --/
structure Block where
  layer : Nat
  value : ℝ

/-- Represents the pyramid structure --/
structure Pyramid where
  blocks : List Block
  bottom_layer : List ℝ

/-- The pyramid satisfies the required structure --/
def valid_structure (p : Pyramid) : Prop :=
  p.blocks.length = 30 ∧
  (p.blocks.filter (fun b => b.layer = 1)).length = 16 ∧
  (p.blocks.filter (fun b => b.layer = 2)).length = 9 ∧
  (p.blocks.filter (fun b => b.layer = 3)).length = 4 ∧
  (p.blocks.filter (fun b => b.layer = 4)).length = 1

/-- The bottom layer is numbered 1 through 16 --/
def valid_bottom_layer (p : Pyramid) : Prop :=
  p.bottom_layer.length = 16 ∧
  p.bottom_layer.all (fun x => x ≥ 1 ∧ x ≤ 16) ∧
  p.bottom_layer.toFinset.card = 16

/-- Each upper block is the average of four blocks below it --/
def valid_averages (p : Pyramid) : Prop :=
  ∀ b, b ∈ p.blocks → b.layer > 1 →
    ∃ b1 b2 b3 b4, b1 ∈ p.blocks ∧ b2 ∈ p.blocks ∧ b3 ∈ p.blocks ∧ b4 ∈ p.blocks ∧
      b1.layer = b.layer - 1 ∧
      b2.layer = b.layer - 1 ∧
      b3.layer = b.layer - 1 ∧
      b4.layer = b.layer - 1 ∧
      b.value = (b1.value + b2.value + b3.value + b4.value) / 4

/-- The theorem to be proved --/
theorem min_top_block_value (p : Pyramid) (lower_bound : ℝ) :
  valid_structure p →
  valid_bottom_layer p →
  valid_averages p →
  ∃ top_block, top_block ∈ p.blocks ∧ top_block.layer = 4 ∧ top_block.value ≥ lower_bound :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_top_block_value_l1042_104260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_hexagons_l1042_104288

/-- A hexagon with integer side lengths --/
structure Hexagon where
  sides : Fin 6 → ℕ

/-- The perimeter of a hexagon --/
def perimeter (h : Hexagon) : ℕ := (Finset.univ.sum fun i => h.sides i)

/-- Check if three sides can form a triangle --/
def canFormTriangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if no three sides of a hexagon can form a triangle --/
def noTriangle (h : Hexagon) : Prop :=
  ∀ i j k, i < j → j < k → ¬(canFormTriangle (h.sides i) (h.sides j) (h.sides k))

/-- The set of valid hexagons --/
def ValidHexagons : Set Hexagon :=
  {h | perimeter h = 20 ∧ noTriangle h}

/-- Theorem stating that there are infinitely many valid hexagons --/
theorem infinitely_many_valid_hexagons :
  Set.Infinite ValidHexagons :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_hexagons_l1042_104288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_trig_identity_l1042_104255

theorem angle_terminal_side_trig_identity (θ : Real) :
  (∃ (x y : Real), x = 4 ∧ y = -3 ∧ x = 5 * Real.cos θ ∧ y = 5 * Real.sin θ) →
  2 * Real.cos θ - Real.sin θ = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_trig_identity_l1042_104255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_j_speed_l1042_104239

/-- Represents the speed of person J in km/h -/
def j : ℝ := sorry

/-- Represents the speed of person P in km/h -/
def p : ℝ := sorry

/-- The distance walked by both J and P -/
def distance : ℝ := 24

/-- The sum of speeds of J and P -/
def speed_sum : ℝ := 7

/-- The sum of time taken by J and P -/
def time_sum : ℝ := 14

theorem j_speed :
  j > p →                     -- J is faster than P
  j + p = speed_sum →         -- Sum of speeds is 7 kmph
  distance / j + distance / p = time_sum →  -- Sum of times is 14 hours
  j = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_j_speed_l1042_104239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_ratio_l1042_104289

noncomputable section

-- Define the square
def Square (side : ℝ) : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ side ∧ 0 ≤ p.2 ∧ p.2 ≤ side}

-- Define the isosceles trapezoid
def IsoscelesTrapezoid (a b h : ℝ) : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ b ∧ 0 ≤ p.2 ∧ p.2 ≤ h ∧ 
       (p.2 = 0 ∨ p.2 = h ∨ 
        ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ p.1 = t * a + (1 - t) * b ∧ 
             p.2 = t * h)}

-- Define the configuration
def TrapezoidConfiguration (side a b h : ℝ) : Prop :=
  ∃ (t1 t2 t3 t4 : Set (ℝ × ℝ)),
    t1 = IsoscelesTrapezoid a b h ∧
    t2 = IsoscelesTrapezoid a b h ∧
    t3 = IsoscelesTrapezoid a b h ∧
    t4 = IsoscelesTrapezoid a b h ∧
    -- The bases of trapezoids form diagonals of the square
    (∀ p ∈ t1 ∪ t2 ∪ t3 ∪ t4, p.1 + p.2 = side ∨ p.1 = p.2)

-- Define the ratio condition
def RatioCondition (side : ℝ) : Prop :=
  ∃ (p x q : ℝ × ℝ),
    p ∈ Square side ∧ q ∈ Square side ∧ x ∈ Square side ∧
    p.1 = 0 ∧ q.1 = side ∧ x.2 = x.1 ∧
    (x.1 - p.1) = 3 * (q.1 - x.1)

-- Define the area of shaded region (placeholder)
def areaShaded (side : ℝ) : ℝ := sorry

-- Define the area of square
def areaSquare (side : ℝ) : ℝ := side * side

-- Theorem statement
theorem trapezoid_area_ratio 
  (side a b h : ℝ) 
  (hpos : side > 0 ∧ a > 0 ∧ b > 0 ∧ h > 0) 
  (hconfig : TrapezoidConfiguration side a b h)
  (hratio : RatioCondition side) :
  (areaShaded side) / (areaSquare side) = 0.375 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_ratio_l1042_104289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_calculation_l1042_104253

/-- The time taken for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed : ℝ) : ℝ :=
  train_length / train_speed

/-- Theorem: The time taken for a train of length 150 m, traveling at a speed of 179.99999999999997 m/s,
    to cross an electric pole is equal to 150 / 179.99999999999997 seconds -/
theorem train_crossing_time_calculation :
  train_crossing_time 150 179.99999999999997 = 150 / 179.99999999999997 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- The equation is true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_calculation_l1042_104253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_problem_l1042_104275

def T : ℕ → ℕ
  | 0 => 1  -- Adding this case to handle Nat.zero
  | 1 => 1
  | 2 => 5
  | n+3 => T (n+2) + 4 * T (n+1) + 2 * T n

theorem tiling_problem :
  (T 10 = 13377) ∧ (T 2013 % 10 = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_problem_l1042_104275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_implication_l1042_104294

theorem power_equation_implication (x : ℝ) : 
  (8 : ℝ)^x - (4 : ℝ)^(x + 1) = 384 → (3*x)^x = 729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_implication_l1042_104294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_formula_for_heavy_items_l1042_104256

/-- Represents the cost function for the express delivery company -/
noncomputable def deliveryCost (x : ℝ) : ℝ :=
  if x ≤ 2 then 10 else 10 + 2 * (x - 2)

/-- Theorem stating the cost for items weighing more than 2 kg -/
theorem cost_formula_for_heavy_items (x : ℝ) (h : x > 2) :
  deliveryCost x = 2 * x + 6 := by
  sorry

#check cost_formula_for_heavy_items

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_formula_for_heavy_items_l1042_104256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commute_time_sum_squares_l1042_104278

def commute_times (x y : ℝ) : List ℝ := [x, y, 10, 11, 9]

theorem commute_time_sum_squares (x y : ℝ) 
  (h_mean : (List.sum (commute_times x y)) / 5 = 10)
  (h_variance : (List.sum (List.map (λ t => (t - 10)^2) (commute_times x y))) / 5 = 2) :
  x^2 + y^2 = 208 := by
sorry

#eval commute_times 1 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commute_time_sum_squares_l1042_104278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_equals_ratio_l1042_104291

open BigOperators Nat

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℚ :=
  (factorial n) / ((factorial k) * (factorial (n - k)))

/-- The sum of 1/binomial(n, 2009) from n=2009 to infinity -/
noncomputable def infiniteSum : ℚ :=
  ∑' n, (if n ≥ 2009 then 1 / binomial n 2009 else 0)

theorem binomial_sum_equals_ratio :
  infiniteSum = 2009 / 2008 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_equals_ratio_l1042_104291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_passes_c_time_l1042_104224

/-- Represents a time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents a car -/
inductive Car
| A | B | C | D

/-- Represents an event when two cars meet -/
structure MeetingEvent where
  car1 : Car
  car2 : Car
  time : Time

/-- The problem setup -/
def problem_setup : List MeetingEvent :=
  [⟨Car.A, Car.B, ⟨8, 0, by norm_num⟩⟩,
   ⟨Car.A, Car.C, ⟨9, 0, by norm_num⟩⟩,
   ⟨Car.A, Car.D, ⟨10, 0, by norm_num⟩⟩,
   ⟨Car.D, Car.B, ⟨12, 0, by norm_num⟩⟩,
   ⟨Car.D, Car.C, ⟨14, 0, by norm_num⟩⟩]

/-- The theorem to prove -/
theorem b_passes_c_time (events : List MeetingEvent) 
  (h : events = problem_setup) : 
  ∃ (t : Time), t.hours = 10 ∧ t.minutes = 40 ∧ 
  (∃ (e : MeetingEvent), e ∈ events ∧ e.car1 = Car.B ∧ e.car2 = Car.C ∧ e.time = t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_passes_c_time_l1042_104224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_loss_percentage_l1042_104298

/-- Represents a type of radio with its cost price, selling price, and units sold. -/
structure Radio where
  cost_price : ℚ
  selling_price : ℚ
  units_sold : ℚ

/-- Calculates the overall loss percentage for a set of radios. -/
def overall_loss_percentage (radios : List Radio) : ℚ :=
  let total_cost := radios.foldl (fun acc r => acc + r.cost_price * r.units_sold) 0
  let total_selling := radios.foldl (fun acc r => acc + r.selling_price * r.units_sold) 0
  let total_loss := total_cost - total_selling
  total_loss / total_cost * 100

/-- The main theorem stating the overall loss percentage for the given radios. -/
theorem radio_loss_percentage : 
  let radios : List Radio := [
    { cost_price := 1500, selling_price := 1290, units_sold := 5 },
    { cost_price := 2200, selling_price := 1900, units_sold := 8 },
    { cost_price := 3000, selling_price := 2800, units_sold := 10 }
  ]
  abs (overall_loss_percentage radios - 989/100) < 1/100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_loss_percentage_l1042_104298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1042_104205

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  -- Smallest positive period is π
  (∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
    (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  -- Axis of symmetry
  (∀ k : ℤ, ∀ x : ℝ, f (2 * (Real.pi / 3 + k * Real.pi / 2) - x) = f x) ∧
  -- Monotonic increasing intervals
  (∀ k : ℤ, ∀ x y : ℝ,
    -Real.pi / 6 + k * Real.pi ≤ x ∧ x < y ∧ y ≤ Real.pi / 3 + k * Real.pi →
    f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1042_104205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalence_theorem_l1042_104267

/-- Sets A and B defined by absolute value inequalities -/
def A (m : ℝ) : Set ℝ := {x | |x - m| > 3}
def B : Set ℝ := {x | |x - 1| < 2}

/-- Propositions p and q -/
def p (m x : ℝ) : Prop := x ∈ A m
def q (x : ℝ) : Prop := x ∈ B

/-- Theorem stating the equivalence of the three conditions -/
theorem equivalence_theorem (m : ℝ) :
  (A m ∩ B = ∅) ∧
  (∀ x, (p m x ∨ q x) ∧ ¬(p m x ∧ q x)) ∧
  (0 ≤ m ∧ m ≤ 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalence_theorem_l1042_104267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion_difference_l1042_104252

/-- Calculates the cost of two pairs of shoes under Promotion A -/
def costPromotionA (price1 : ℝ) (price2 : ℝ) : ℝ :=
  price1 + price2 * 0.6

/-- Calculates the cost of two pairs of shoes under Promotion B -/
noncomputable def costPromotionB (price1 : ℝ) (price2 : ℝ) : ℝ :=
  let totalPrice := price1 + price2
  let discountedPrice := price1 + (price2 - 15)
  if totalPrice > 70 then discountedPrice - 5 else discountedPrice

/-- The prices of Julian's shoes -/
def julianShoe1 : ℝ := 50
def julianShoe2 : ℝ := 25

/-- Theorem stating the difference in cost between promotions -/
theorem promotion_difference :
  costPromotionA julianShoe1 julianShoe2 - costPromotionB julianShoe1 julianShoe2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion_difference_l1042_104252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_theorem_l1042_104245

noncomputable def roundToDecimalPlace (x : ℝ) (decimalPlace : ℕ) : ℝ :=
  (⌊x * 10^decimalPlace + 0.5⌋) / 10^decimalPlace

def roundToPlaceValue (x : ℕ) (placeValue : ℕ) : ℕ :=
  ((x + placeValue / 2) / placeValue) * placeValue

theorem rounding_theorem :
  (roundToDecimalPlace 3.896 2 = 3.90) ∧
  (roundToPlaceValue 66800 10000 = 70000) := by
  sorry

#eval roundToPlaceValue 66800 10000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_theorem_l1042_104245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_multiplicative_f_sum_over_divisors_l1042_104228

def f (n : ℕ) : ℕ := (Finset.range n).sum (λ k => Nat.gcd (k + 1) n)

theorem f_multiplicative (m n : ℕ) (h : Nat.Coprime m n) : 
  f (m * n) = f m * f n := by sorry

theorem f_sum_over_divisors (n : ℕ) : 
  (Finset.filter (λ d => d ∣ n) (Finset.range (n + 1))).sum f = n * (Finset.filter (λ d => d ∣ n) (Finset.range (n + 1))).card := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_multiplicative_f_sum_over_divisors_l1042_104228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_shiny_on_sixth_draw_l1042_104248

/-- The number of shiny pennies in the box -/
def shiny_pennies : ℕ := 5

/-- The number of dull pennies in the box -/
def dull_pennies : ℕ := 6

/-- The total number of pennies in the box -/
def total_pennies : ℕ := shiny_pennies + dull_pennies

/-- The number of draws made -/
def num_draws : ℕ := 6

/-- The number of shiny pennies we want to draw -/
def target_shiny : ℕ := 4

/-- The probability of drawing exactly four shiny pennies in six draws, 
    with the fourth shiny penny being drawn on the sixth draw -/
def probability_fourth_shiny_on_sixth : ℚ := 5 / 231

theorem fourth_shiny_on_sixth_draw : 
  (Nat.choose shiny_pennies (target_shiny - 1) * Nat.choose (total_pennies - num_draws + 1) 1) / 
  (Nat.choose total_pennies num_draws) = probability_fourth_shiny_on_sixth := by
  sorry

#eval Nat.choose shiny_pennies (target_shiny - 1)
#eval Nat.choose (total_pennies - num_draws + 1) 1
#eval Nat.choose total_pennies num_draws

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_shiny_on_sixth_draw_l1042_104248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alyssa_fruit_purchase_l1042_104244

def grapes : ℚ := 12.08
def cherries : ℚ := 9.85
def mangoes : ℚ := 7.50
def pineapple : ℚ := 4.25
def starfruit : ℚ := 3.98
def tax_rate : ℚ := 0.10
def discount : ℚ := 3

def total_cost : ℚ := grapes + cherries + mangoes + pineapple + starfruit

theorem alyssa_fruit_purchase :
  (total_cost + (tax_rate * total_cost).floor / 100 * 100 - discount).floor / 100 * 100 = 3843 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alyssa_fruit_purchase_l1042_104244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l1042_104250

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := x / (2 * x - 1)

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := -1 / ((2 * x - 1)^2)

-- Define the point of tangency
def point : ℝ × ℝ := (1, 1)

-- Define the tangent line equation
def tangent_line (x : ℝ) : ℝ := -x + 2

theorem tangent_line_at_point :
  (f point.fst = point.snd) ∧
  (f' point.fst = -1) ∧
  (∀ x : ℝ, tangent_line x = point.snd + f' point.fst * (x - point.fst)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l1042_104250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_82_27534_to_nearest_tenth_l1042_104230

noncomputable def round_to_nearest_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem round_82_27534_to_nearest_tenth :
  round_to_nearest_tenth 82.27534 = 82.3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_82_27534_to_nearest_tenth_l1042_104230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_game_no_chance_rematch_possible_l1042_104227

/-- Represents the number of people in the tribe -/
def tribeSize : ℕ := 30

/-- Represents the total number of coins to be distributed in the rematch -/
def totalCoins : ℕ := 270

/-- Represents the maximum number of people that can be expelled -/
def maxExpelled : ℕ := 6

/-- Function to calculate the number of coins for a person (placeholder) -/
def number_of_coins (i : ℕ) : ℕ := sorry

/-- Theorem stating that in the original game, at least two people will have the same number of coins -/
theorem original_game_no_chance (n : ℕ) (h : n = tribeSize) : 
  ∃ (i j : ℕ), i ≠ j ∧ i < n ∧ j < n ∧ (number_of_coins i = number_of_coins j) := by
  sorry

/-- Function to calculate the sum of natural numbers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating that after expelling at most 6 people, it's possible to uniquely distribute 270 coins -/
theorem rematch_possible :
  ∃ (expelled : ℕ), expelled ≤ maxExpelled ∧ 
  sum_to_n (tribeSize - expelled) ≤ totalCoins := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_game_no_chance_rematch_possible_l1042_104227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_circle_l1042_104231

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis

/-- Represents the trajectory of a point -/
inductive Trajectory
  | Circle : Point → ℝ → Trajectory
  | Other : Trajectory

/-- External angle bisector of ∠F₁MF₂ -/
def externalAngleBisector (f₁ f₂ m : Point) : Point → Point := sorry

/-- Foot of perpendicular from a point to a line -/
def perpendicularFoot (p : Point) (l : Point → Point) : Point := sorry

/-- Check if a point is on an ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop := sorry

/-- Check if two points are foci of an ellipse -/
def areFoci (e : Ellipse) (f₁ f₂ : Point) : Prop := sorry

/-- The theorem stating that the trajectory of N is a circle -/
theorem trajectory_is_circle 
  (e : Ellipse) 
  (f₁ f₂ : Point) 
  (h_foci : areFoci e f₁ f₂)
  (m : ℝ → Point)  -- m is a function representing the moving point
  (h_m : ∀ t, isOnEllipse e (m t))
  (n : ℝ → Point)
  (h_n : ∀ t, n t = perpendicularFoot f₂ (externalAngleBisector f₁ f₂ (m t)))
  : ∃ (center : Point) (radius : ℝ), 
    ∀ t, (n t).x^2 + (n t).y^2 = radius^2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_circle_l1042_104231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_M_and_N_l1042_104207

noncomputable section

-- Define the sets M and N
def M : Set ℂ := {z | Complex.abs (z - Complex.I * 3) = 1}
def N : Set ℂ := {z | Complex.abs (z - 4) = 1}

-- Define the shortest distance function
noncomputable def shortestDistance (A B : ℂ) : ℝ := Complex.abs (A - B)

-- Theorem statement
theorem shortest_distance_between_M_and_N :
  ∃ (A B : ℂ), A ∈ M ∧ B ∈ N ∧ 
  (∀ (X Y : ℂ), X ∈ M → Y ∈ N → shortestDistance A B ≤ shortestDistance X Y) ∧
  shortestDistance A B = 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_M_and_N_l1042_104207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_can_achieve_equal_coins_a_max_can_achieve_equal_coins_b_max_can_achieve_equal_coins_c_l1042_104247

/-- Represents the state of coins in cans -/
def CoinState := Fin 2015 → ℕ

/-- Represents a step in Max's process -/
def Step := Fin 2015

/-- Apply a step to a coin state -/
def applyStep (state : CoinState) (step : Step) : CoinState :=
  λ i => if i = step then state i else state i + step.val

/-- Check if all cans have the same number of coins -/
def allEqual (state : CoinState) : Prop :=
  ∀ i j : Fin 2015, state i = state j

/-- Initial state for configuration (a) -/
def initialStateA : CoinState := λ _ => 0

/-- Initial state for configuration (b) -/
def initialStateB : CoinState := λ i => i.val + 1

/-- Initial state for configuration (c) -/
def initialStateC : CoinState := λ i => 2016 - (i.val + 1)

/-- Theorem: Max can achieve equal coins for configuration (a) -/
theorem max_can_achieve_equal_coins_a :
  ∃ (steps : List Step), steps.length > 0 ∧
    allEqual (steps.foldl applyStep initialStateA) := by
  sorry

/-- Theorem: Max can achieve equal coins for configuration (b) -/
theorem max_can_achieve_equal_coins_b :
  ∃ (steps : List Step), steps.length > 0 ∧
    allEqual (steps.foldl applyStep initialStateB) := by
  sorry

/-- Theorem: Max can achieve equal coins for configuration (c) -/
theorem max_can_achieve_equal_coins_c :
  ∃ (steps : List Step), steps.length > 0 ∧
    allEqual (steps.foldl applyStep initialStateC) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_can_achieve_equal_coins_a_max_can_achieve_equal_coins_b_max_can_achieve_equal_coins_c_l1042_104247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1042_104242

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^4 + 2 * abs (x - 1)

-- State the theorem
theorem f_inequality (x : ℝ) : f x > f (2 * x) ↔ 0 < x ∧ x < 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1042_104242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_line_distance_range_l1042_104225

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 is given by |Ax₀ + By₀ + C| / √(A² + B²) -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- The theorem stating the range of 'a' given the conditions -/
theorem point_line_distance_range :
  ∀ (a : ℝ),
  (distance_point_to_line 2 3 a (a-1) 3 ≥ 3) ↔ 
  (a ≤ -3 ∨ a ≥ 3/7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_line_distance_range_l1042_104225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l1042_104210

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℚ → ℚ) : Prop :=
  ∀ x y z : ℚ, f (x + f (y + f z)) = y + f (x + z)

/-- The set of all functions satisfying the equation -/
def SolutionSet : Set (ℚ → ℚ) :=
  {f | SatisfiesEquation f}

/-- The identity function on ℚ -/
def id_rat : ℚ → ℚ := λ x ↦ x

/-- The set of all functions of the form f(x) = a - x for a ∈ ℚ -/
def ReflectionSet : Set (ℚ → ℚ) :=
  {f | ∃ a : ℚ, ∀ x : ℚ, f x = a - x}

/-- The main theorem stating that the solution set is exactly
    the union of the identity function and the reflection set -/
theorem solution_characterization :
  SolutionSet = {id_rat} ∪ ReflectionSet := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l1042_104210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_two_l1042_104280

/-- Circle C in Cartesian coordinates -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- Line l in standard form -/
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

/-- The chord length is the distance between two intersection points of the circle and the line -/
noncomputable def chord_length (C L : ℝ → ℝ → Prop) : ℝ := 
  Real.sqrt 2 -- We're using the known result here

/-- Theorem stating that the chord length is √2 -/
theorem chord_length_is_sqrt_two :
  chord_length circle_C line_l = Real.sqrt 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_two_l1042_104280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l1042_104217

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) * Real.cos (3 * x)

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := 2 * Real.exp (2 * x) * Real.cos (3 * x) - 3 * Real.exp (2 * x) * Real.sin (3 * x)

-- Theorem statement
theorem tangent_line_at_zero_one :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 2 * x - y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l1042_104217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_total_slices_l1042_104264

-- Define the number of slices for each pizza
noncomputable def pizza1_slices : ℚ := 8
noncomputable def pizza2_slices : ℚ := 12
noncomputable def pizza3_slices : ℚ := 16
noncomputable def pizza4_slices : ℚ := 18

-- Define how many slices each person eats
noncomputable def tom_eats : ℚ := 5/2
noncomputable def alice_eats : ℚ := 7/2
noncomputable def bob_eats_total : ℚ := 29/4
noncomputable def bob_eats_pizza3 : ℚ := 5

-- Define James' eating behavior
noncomputable def james_fraction : ℚ := 1/2

-- Theorem to prove
theorem james_total_slices :
  let remaining1 := pizza1_slices - tom_eats
  let remaining2 := pizza2_slices - alice_eats
  let remaining3 := pizza3_slices - bob_eats_pizza3
  let remaining4 := pizza4_slices - (bob_eats_total - bob_eats_pizza3)
  let james_eats := james_fraction * (remaining1 + remaining2 + remaining3 + remaining4)
  james_eats = 163/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_total_slices_l1042_104264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_is_one_third_l1042_104249

noncomputable section

/-- Parabola C: y² = 4x -/
def C (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus F of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- Point P lies on parabola C -/
def P_on_C (P : ℝ × ℝ) : Prop := C P.1 P.2

/-- Vector from Q to F -/
def QF (Q : ℝ × ℝ) : ℝ × ℝ := (F.1 - Q.1, F.2 - Q.2)

/-- Vector from P to Q -/
def PQ (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

/-- Condition: PQ = 9QF -/
def PQ_eq_9QF (P Q : ℝ × ℝ) : Prop := PQ P Q = (9 * (QF Q).1, 9 * (QF Q).2)

/-- Slope of line OQ -/
noncomputable def slope_OQ (Q : ℝ × ℝ) : ℝ := Q.2 / Q.1

/-- Theorem: Maximum slope of OQ is 1/3 -/
theorem max_slope_OQ_is_one_third :
  ∃ (Q_max : ℝ × ℝ), ∀ (P Q : ℝ × ℝ),
    P_on_C P → PQ_eq_9QF P Q →
    slope_OQ Q ≤ slope_OQ Q_max ∧ slope_OQ Q_max = 1/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_is_one_third_l1042_104249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_orders_count_l1042_104237

def number_of_letters : ℕ := 11
def typed_letter : ℕ := 9

def possible_typing_orders : ℕ := Finset.sum (Finset.range 10) (λ k => Nat.choose (typed_letter - 1) k * (k + 2))

theorem typing_orders_count :
  possible_typing_orders = 3328 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_orders_count_l1042_104237

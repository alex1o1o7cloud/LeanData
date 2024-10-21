import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l1024_102484

/-- Calculates the average speed of a journey with varying speeds and a rest period. -/
noncomputable def average_speed (total_distance : ℝ) (speed1 speed2 speed3 : ℝ) (rest_time : ℝ) : ℝ :=
  let distance1 := total_distance / 3
  let distance2 := total_distance / 4
  let distance3 := total_distance - distance1 - distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let total_time := time1 + time2 + time3 + rest_time
  total_distance / total_time

/-- Theorem stating that the average speed for the given journey is approximately 2.057 mph. -/
theorem journey_average_speed :
  let total_distance := (12 : ℝ)
  let speed1 := (3 : ℝ)
  let speed2 := (1.5 : ℝ)
  let speed3 := (2.5 : ℝ)
  let rest_time := (0.5 : ℝ)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
    |average_speed total_distance speed1 speed2 speed3 rest_time - 2.057| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l1024_102484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_for_complete_circle_l1024_102419

open Real

/-- The set of points (r cos θ, r sin θ) where r = sin θ and 0 ≤ θ ≤ t -/
def CircleSet (t : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ θ, 0 ≤ θ ∧ θ ≤ t ∧ p = ((sin θ) * (cos θ), (sin θ) * (sin θ))}

/-- A complete circle is the set of all points (x, y) such that x² + y² = 1 -/
def CompleteCircle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 1}

/-- π is the smallest positive real number t such that CircleSet t equals CompleteCircle -/
theorem smallest_t_for_complete_circle :
  ∀ t > 0, CircleSet t = CompleteCircle ↔ t ≥ π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_for_complete_circle_l1024_102419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l1024_102407

theorem cube_root_equation_solution :
  ∃! x : ℚ, (5 - x) = -125/27 ∧ x = 260/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l1024_102407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l1024_102418

theorem angle_in_second_quadrant (α : Real) : 
  (Real.sin α * Real.cos α < 0) → (Real.cos α - Real.sin α < 0) → 
  0 < α ∧ α < Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l1024_102418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_inverse_l1024_102470

theorem power_of_three_inverse (a b : ℕ) : 
  (2^a * 5^b = 200) →
  (∀ k, 2^k ∣ 200 → k ≤ a) →
  (∀ k, 5^k ∣ 200 → k ≤ b) →
  (1/3 : ℚ)^(b - a) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_inverse_l1024_102470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_cost_l1024_102433

/-- The cost of flour in Laura's cake recipe -/
theorem flour_cost (sugar_eggs_butter_cost : ℝ) 
                   (total_slices : ℕ) 
                   (mother_slices : ℕ) 
                   (dog_slices : ℕ) 
                   (dog_eaten_cost : ℝ) 
                   (h1 : sugar_eggs_butter_cost = 5)
                   (h2 : total_slices = 6)
                   (h3 : mother_slices = 2)
                   (h4 : dog_slices = total_slices - mother_slices)
                   (h5 : dog_eaten_cost = 6)
                   (h6 : dog_slices > 0)
                   (h7 : (dog_eaten_cost / (dog_slices : ℝ)) * (total_slices : ℝ) - sugar_eggs_butter_cost = 4)
                   : 4 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_cost_l1024_102433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_salt_price_l1024_102488

/-- Given the price of sugar and salt, calculate the price of a different quantity -/
theorem sugar_salt_price
  (price_2sugar_5salt : ℝ)
  (price_1sugar : ℝ)
  (h1 : price_2sugar_5salt = 5.50)
  (h2 : price_1sugar = 1.50) :
  3 * price_1sugar + (price_2sugar_5salt - 2 * price_1sugar) / 5 = 5.00 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_salt_price_l1024_102488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_divisible_by_18_with_sqrt_between_22_and_22_5_l1024_102416

theorem integer_divisible_by_18_with_sqrt_between_22_and_22_5 :
  ∃ (n : ℕ), 
    (n : ℝ).sqrt > 22 ∧ 
    (n : ℝ).sqrt < 22.5 ∧ 
    n % 18 = 0 ∧
    (n = 486 ∨ n = 504) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_divisible_by_18_with_sqrt_between_22_and_22_5_l1024_102416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_is_0_6_l1024_102451

/-- Represents a brand of flour with a nominal mass and tolerance -/
structure FlourBrand where
  nominal_mass : ℝ
  tolerance : ℝ

/-- The set of flour brands available in the store -/
def flour_brands : List FlourBrand :=
  [⟨2.5, 0.1⟩, ⟨2.5, 0.2⟩, ⟨2.5, 0.3⟩]

/-- The maximum mass of any flour bag -/
noncomputable def max_mass (brands : List FlourBrand) : ℝ :=
  brands.map (fun b => b.nominal_mass + b.tolerance)
    |>.maximum?
    |>.getD 0

/-- The minimum mass of any flour bag -/
noncomputable def min_mass (brands : List FlourBrand) : ℝ :=
  brands.map (fun b => b.nominal_mass - b.tolerance)
    |>.minimum?
    |>.getD 0

/-- The maximum difference in masses between any two bags -/
noncomputable def max_difference (brands : List FlourBrand) : ℝ :=
  max_mass brands - min_mass brands

theorem max_difference_is_0_6 :
  max_difference flour_brands = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_is_0_6_l1024_102451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1024_102405

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

theorem function_properties (ω φ : ℝ) (h1 : ω > 0) (h2 : -π/2 ≤ φ) (h3 : φ < π/2)
  (h4 : ∀ x, f ω φ (x + π/3) = f ω φ (π/3 - x))  -- Symmetry about x = π/3
  (h5 : ∀ x, f ω φ (x + π) = f ω φ x)  -- Period π
  : 
  (∀ x, f ω φ x = Real.sqrt 3 * Real.sin (2*x - π/6)) ∧   -- 1
  (∀ k : ℤ, ∃ x, x = k * π / 2 + π / 12 ∧ ∀ y, f ω φ (x + y) = f ω φ (x - y)) ∧   -- 2
  (∀ k : ℤ, ∀ x, k * π - π/6 ≤ x ∧ x ≤ k * π + π/3 → 
    ∀ y, x < y → f ω φ x < f ω φ y)   -- 3
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1024_102405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l1024_102450

theorem exponential_equation_solution (x y : ℝ) :
  (7 : ℝ)^(3*x - 1) * (3 : ℝ)^(4*y - 3) = (49 : ℝ)^x * (27 : ℝ)^y → x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l1024_102450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1024_102471

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (ω * x) + Real.sqrt 2 * Real.cos (ω * x)

theorem omega_range (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x₁ x₂ : ℝ, -π/3 < x₁ ∧ x₁ < x₂ ∧ x₂ < π/4 → f ω x₁ < f ω x₂) :
  ω ∈ Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1024_102471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l1024_102480

theorem binomial_expansion_properties : 
  ∃ (x : ℤ), 
  let n : ℕ := 1999
  -- Define the binomial coefficient
  let binomial_coef (n k : ℕ) := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))
  -- Define the 1000th term of the expansion
  let term_1000 := -Int.ofNat (binomial_coef n 1000) * x^999
  -- Define the binomial expansion
  let binomial_expansion := (1 - x)^n
  (
    -- Proposition 1: The 1000th term is correct
    term_1000 = -Int.ofNat (binomial_coef n 1000) * x^999 ∧
    -- Proposition 4: The remainder when x = 2000 is 1
    binomial_expansion % 2000 = 1 
  ) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l1024_102480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_grandeur_monic_quadratic_l1024_102420

/-- The grandeur of a polynomial q(x) on the interval [-2, 2] -/
noncomputable def grandeur (q : ℝ → ℝ) : ℝ :=
  ⨆ (x : ℝ) (hx : x ∈ Set.Icc (-2) 2), |q x|

/-- A monic quadratic polynomial -/
def monicQuadratic (b c : ℝ) (x : ℝ) : ℝ :=
  x^2 + b*x + c

theorem smallest_grandeur_monic_quadratic :
  ∃ (b₀ c₀ : ℝ), ∀ (b c : ℝ),
    grandeur (monicQuadratic b c) ≥ grandeur (monicQuadratic b₀ c₀) ∧
    grandeur (monicQuadratic b₀ c₀) = 3 := by
  sorry

#check smallest_grandeur_monic_quadratic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_grandeur_monic_quadratic_l1024_102420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_when_area_maximized_l1024_102494

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of the foci -/
noncomputable def left_focus : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 3, 0)

/-- Definition of a point on the ellipse -/
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  is_on_ellipse P.1 P.2

/-- Definition of a line passing through the center -/
def line_through_center (P Q : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), P = (t, 0) ∧ Q = (-t, 0)

/-- Definition of the area of quadrilateral being maximized -/
def area_maximized (P Q : ℝ × ℝ) : Prop :=
  P = (0, 1) ∧ Q = (0, -1)

/-- The main theorem -/
theorem dot_product_when_area_maximized 
  (P Q : ℝ × ℝ) 
  (h1 : point_on_ellipse P)
  (h2 : line_through_center P Q)
  (h3 : point_on_ellipse Q)
  (h4 : area_maximized P Q) :
  (P.1 - left_focus.1) * (P.1 - right_focus.1) + 
  (P.2 - left_focus.2) * (P.2 - right_focus.2) = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_when_area_maximized_l1024_102494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l1024_102459

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -1/x + a
  else Real.sqrt x * (x - a) - 1

-- State the theorem
theorem f_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → f a x > x - 1) →
  -3 < a ∧ a < -1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l1024_102459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OM_l1024_102427

/-- The parabola y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- A point on the parabola -/
structure ParabolaPoint (para : Parabola) where
  y : ℝ
  eq : y^2 = 2 * para.p * (y^2 / (2 * para.p))

/-- The focus of the parabola -/
noncomputable def focus (para : Parabola) : ℝ × ℝ := (para.p / 2, 0)

/-- Point M on the line segment PF such that |PM| = 2|MF| -/
noncomputable def pointM (para : Parabola) (P : ParabolaPoint para) : ℝ × ℝ := by
  let F := focus para
  let P_coords := (P.y^2 / (2 * para.p), P.y)
  exact ((2 * F.1 + P_coords.1) / 3, P_coords.2 / 3)

/-- The slope of line OM -/
noncomputable def slopeOM (para : Parabola) (P : ParabolaPoint para) : ℝ := by
  let M := pointM para P
  exact M.2 / M.1

theorem max_slope_OM (para : Parabola) :
  ∃ (max : ℝ), max = Real.sqrt 2 / 2 ∧
    ∀ (P : ParabolaPoint para), |slopeOM para P| ≤ max := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OM_l1024_102427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1024_102467

open Real
open BigOperators

/-- The sum of the infinite series ∑(n=1 to ∞) (3n + 2) / (n(n + 1)(n + 3)) is equal to 71/240 -/
theorem series_sum : ∑' (n : ℕ), (3 * (n + 1 : ℝ) + 2) / ((n + 1) * (n + 2) * (n + 4)) = 71 / 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1024_102467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_length_l1024_102489

/-- Represents a rhombus with given side length and shorter diagonal -/
structure Rhombus where
  side_length : ℝ
  shorter_diagonal : ℝ

/-- Calculates the length of the longer diagonal of a rhombus -/
noncomputable def longer_diagonal (r : Rhombus) : ℝ :=
  2 * Real.sqrt (r.side_length ^ 2 - (r.shorter_diagonal / 2) ^ 2)

theorem rhombus_longer_diagonal_length :
  let r : Rhombus := { side_length := 65, shorter_diagonal := 60 }
  longer_diagonal r = 116 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_length_l1024_102489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_intersection_equidistant_l1024_102412

/-- The equation of a line passing through the intersection of two lines and equidistant from two points -/
theorem line_equation_through_intersection_equidistant :
  ∃ l : ℝ × ℝ → ℝ,
    (∀ p : ℝ × ℝ, (p.1 - 2 * p.2 + 4 = 0 ∧ 2 * p.1 - p.2 - 1 = 0) → l p = 0) ∧
    (∀ p : ℝ × ℝ, l p = 0 → (p.1 - 0)^2 + (p.2 - 4)^2 = (p.1 - 4)^2 + (p.2 - 0)^2) ∧
    ((∀ p : ℝ × ℝ, l p = 0 ↔ p.1 + p.2 - 5 = 0) ∨ (∀ p : ℝ × ℝ, l p = 0 ↔ p.1 = 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_intersection_equidistant_l1024_102412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_3_eq_3_l1024_102429

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  (x^(2^6 - 1) - 1)⁻¹ * ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) * (x^32 + 1) - 1)

-- Theorem statement
theorem f_of_3_eq_3 : f 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_3_eq_3_l1024_102429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_z_headstart_l1024_102406

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ

/-- Represents a race with three runners -/
structure Race where
  distance : ℝ
  x : Runner
  y : Runner
  z : Runner
  xy_headstart : ℝ
  xz_headstart : ℝ
  hxy : x.speed * distance = y.speed * distance + xy_headstart
  hxz : x.speed * distance = z.speed * distance + xz_headstart

/-- Theorem stating that if X can give Y a 100m head start and Z a 200m head start,
    then Y can give Z a 200m head start -/
theorem y_z_headstart (race : Race) (h1 : race.xy_headstart = 100) (h2 : race.xz_headstart = 200) :
  ∃ (yz_headstart : ℝ), yz_headstart = 200 ∧ 
  race.y.speed * race.distance = race.z.speed * race.distance + yz_headstart := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_z_headstart_l1024_102406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_sector_l1024_102431

/-- Given a cone formed from a 270-degree sector of a circle with radius 20,
    prove that the volume of the cone divided by π is equal to 1125√7. -/
theorem cone_volume_from_sector (r : ℝ) (angle : ℝ) :
  r = 20 →
  angle = 270 * π / 180 →
  (1/3) * π * (r * angle / (2 * π))^2 * Real.sqrt (r^2 - (r * angle / (2 * π))^2) / π = 1125 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_sector_l1024_102431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_water_gallons_l1024_102454

-- Define the capacities of the water heaters
noncomputable def wallace_capacity : ℝ := 40
noncomputable def catherine_capacity : ℝ := wallace_capacity / 2
noncomputable def albert_capacity : ℝ := wallace_capacity * 1.5

-- Define the fullness of each water heater
noncomputable def wallace_fullness : ℝ := 3 / 4
noncomputable def catherine_fullness : ℝ := 3 / 4
noncomputable def albert_initial_fullness : ℝ := 2 / 3

-- Define the leak in Albert's water heater
noncomputable def albert_leak : ℝ := 5

-- Theorem statement
theorem total_water_gallons :
  wallace_capacity * wallace_fullness +
  catherine_capacity * catherine_fullness +
  (albert_capacity * albert_initial_fullness - albert_leak) = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_water_gallons_l1024_102454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_evaluation_l1024_102448

theorem fraction_evaluation (a b c : ℝ) (h : a ≠ b) :
  c * (a^(-6 : ℤ) - b^(-6 : ℤ)) / (a^(-3 : ℤ) - b^(-3 : ℤ)) = c * (a^(-6 : ℤ) + a^(-3 : ℤ) * b^(-3 : ℤ) + b^(-6 : ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_evaluation_l1024_102448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_20_l1024_102432

-- Define the coordinates of the opposite vertices
def point1 : ℚ × ℚ := (1, 2)
def point2 : ℚ × ℚ := (7, 4)

-- Define the square area function
noncomputable def squareArea (p1 p2 : ℚ × ℚ) : ℚ :=
  let dx := p2.1 - p1.1
  let dy := p2.2 - p1.2
  (dx * dx + dy * dy) / 2

-- Theorem statement
theorem square_area_is_20 : squareArea point1 point2 = 20 := by
  -- Unfold the definition of squareArea
  unfold squareArea
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_20_l1024_102432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_transformation_l1024_102424

/-- Prove that shifting sin(2x + π/6) left by 3π/4 and stretching vertically by 2 yields sin2x - √3cos2x -/
theorem sin_cos_transformation (x : ℝ) : 
  2 * Real.sin (2 * (x + 3 * Real.pi / 4) + Real.pi / 6) = 
    Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_transformation_l1024_102424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l1024_102449

noncomputable section

open Real

-- Define the line y = √3x
def line (x : ℝ) : ℝ := sqrt 3 * x

-- Define the set of angles whose terminal side lies on the line
def angleset : Set ℝ := {α | ∃ k : ℤ, α = π / 3 + k * π}

-- Define the third quadrant
def third_quadrant (α : ℝ) : Prop := π < α ∧ α < 3 * π / 2

-- Define the first or second quadrant
def first_or_second_quadrant (α : ℝ) : Prop := 0 < α ∧ α < π

theorem angle_problem (α : ℝ) (h : third_quadrant α) :
  angleset = {α | ∃ k : ℤ, α = π / 3 + k * π} ∧
  first_or_second_quadrant (2 * α) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l1024_102449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_number_l1024_102460

theorem imaginary_part_of_complex_number :
  ∃ (z : ℂ), (z.im ≠ 2 ∧ z.im ≠ 1) ∧ z.im ∈ Set.univ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_number_l1024_102460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_salary_difference_l1024_102436

def CircularSequence (n : ℕ) := Fin n → ℤ

def ValidCircularSequence (seq : CircularSequence 2002) : Prop :=
  (∀ i : Fin 2002, (seq i - seq (i.succ)).natAbs ∈ ({2, 3} : Set ℕ)) ∧
  (∀ i j : Fin 2002, i ≠ j → seq i ≠ seq j)

theorem max_salary_difference (seq : CircularSequence 2002) 
  (h : ValidCircularSequence seq) : 
  (∀ i j : Fin 2002, (seq i - seq j).natAbs ≤ 3002) ∧ 
  (∃ i j : Fin 2002, (seq i - seq j).natAbs = 3002) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_salary_difference_l1024_102436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_transform_involution_l1024_102478

def board_transform (a : List ℕ) : List ℕ :=
  let b := List.map (λ k => (a.filter (λ x => x > k)).length) (List.range ((a.maximum.getD 0) + 1))
  b.filter (λ x => x > 0)

theorem board_transform_involution (a : List ℕ) (h : a.all (λ x => x > 0)) :
  board_transform (board_transform a) = a := by
  sorry

#check board_transform_involution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_transform_involution_l1024_102478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_multiple_l1024_102409

/-- Anne's savings -/
def A : ℝ := sorry

/-- Katherine's savings -/
def K : ℝ := sorry

/-- The multiple of Katherine's savings -/
def m : ℝ := sorry

/-- Proof of the problem -/
theorem savings_multiple : 
  (A - 150 = (1/3) * K) →  -- If Anne had $150 less, she would have exactly 1/3 as much as Katherine
  (m * K = 3 * A) →        -- If Katherine had a certain multiple of her savings, she would have exactly 3 times as much as Anne
  (A + K = 750) →          -- They have saved together $750
  m = 2                    -- The multiple of Katherine's savings that would make her have 3 times Anne's savings is 2
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_multiple_l1024_102409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_balls_three_boxes_l1024_102464

/-- Stirling number of the second kind -/
def stirling2 : Nat → Nat → Nat
  | 0, 0 => 1
  | 0, _ => 0
  | _, 0 => 0
  | n + 1, k + 1 => (k + 1) * stirling2 n (k + 1) + stirling2 n k

/-- Number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ballsInBoxes (n k : Nat) : Nat := stirling2 n k

theorem five_balls_three_boxes :
  ballsInBoxes 5 3 = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_balls_three_boxes_l1024_102464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_theorem_l1024_102442

open BigOperators

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def binomial_sum (n : ℕ) : ℕ := 
  (List.range 11).foldl (fun acc k => acc + Nat.choose n k) 0

theorem fraction_sum_theorem :
  let M : ℕ := 524287
  let n : ℕ := 20
  (((List.range 10).map (fun k => (1 : ℚ) / ((factorial k) * (factorial (19 - k))))).sum
    = (M : ℚ) / (factorial n)) ∧
  (Int.floor ((M : ℚ) / 100) = 5242) := by
  sorry

#eval binomial_sum 20
#eval Int.floor ((524287 : ℚ) / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_theorem_l1024_102442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pairs_satisfy_equation_l1024_102447

theorem two_pairs_satisfy_equation : 
  ∃! n : ℕ, n = (Finset.filter 
    (λ p : ℕ × ℕ ↦ p.1 ^ 2 - p.2 ^ 2 = 171 ∧ p.1 > 0 ∧ p.2 > 0) 
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pairs_satisfy_equation_l1024_102447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1024_102425

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1024_102425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1024_102490

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi/3 + x) * Real.cos (Real.pi/3 - x) - Real.sin x * Real.cos x + 1/4

theorem f_properties :
  -- The smallest positive period of f is π
  (∀ (x : ℝ), f (x + Real.pi) = f x) ∧
  (∀ (T : ℝ), T > 0 → (∀ (x : ℝ), f (x + T) = f x) → T ≥ Real.pi) ∧
  -- The maximum value of f is √2/2
  (∀ (x : ℝ), f x ≤ Real.sqrt 2 / 2) ∧
  (∃ (x : ℝ), f x = Real.sqrt 2 / 2) ∧
  -- f is monotonically decreasing on [0, 3π/8] and [7π/8, π]
  (∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ 3*Real.pi/8 → f y < f x) ∧
  (∀ (x y : ℝ), 7*Real.pi/8 ≤ x ∧ x < y ∧ y ≤ Real.pi → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1024_102490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1024_102422

-- Define the constants
noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 5 / Real.log 0.2
noncomputable def c : ℝ := (1/2)^2

-- Theorem statement
theorem relationship_abc : b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1024_102422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1024_102434

theorem problem_solution : 
  ((-1)^4 - (2 - 3)^2 * (-2)^3 = 7) ∧ 
  (|Real.sqrt 2 - 2| + Real.sqrt (4/9) - (8 : ℝ)^(1/3) = 2/3 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1024_102434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l1024_102461

theorem repeating_decimal_sum : 
  (1/9 : ℚ) + (1/99 : ℚ) + (1/9999 : ℚ) = 1213 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l1024_102461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1024_102440

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x / Real.exp x - a * x * Real.log x

-- Define the tangent line
def tangentLine (b : ℝ) (x : ℝ) : ℝ := b * x + 1 + 1 / Real.exp 1

-- Theorem statement
theorem function_properties (a b : ℝ) :
  (∀ x, x > 0 → (deriv (f a)) x = (deriv (tangentLine b)) x) →
  (f a 1 = tangentLine b 1) →
  (a = 1 ∧ b = -1) ∧
  (∀ x, x > 0 → f a x < 2 / Real.exp 1) ∧
  (∀ m n, m > 0 → n > 0 → m * n = 1 → 
    1 / Real.exp (m - 1) + 1 / Real.exp (n - 1) < 2 * (m + n)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1024_102440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l1024_102455

-- Define the side length of a small pen
variable (s : ℝ)

-- Define the perimeter of a small pen
noncomputable def small_perimeter (s : ℝ) : ℝ := 3 * s

-- Define the perimeter of the large pen
noncomputable def large_perimeter (s : ℝ) : ℝ := 3 * small_perimeter s

-- Define the side length of the large pen
noncomputable def large_side (s : ℝ) : ℝ := large_perimeter s / 3

-- Define the area of a small pen
noncomputable def small_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

-- Define the area of the large pen
noncomputable def large_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * (large_side s)^2

-- Define the total area of four small pens
noncomputable def total_small_area (s : ℝ) : ℝ := 4 * small_area s

-- Theorem statement
theorem area_ratio (s : ℝ) (h : s > 0) : 
  total_small_area s / large_area s = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l1024_102455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_400g_instrument_l1024_102482

/-- The cost in dollars per kilogram for transporting equipment to a lunar base. -/
noncomputable def cost_per_kg : ℚ := 18000

/-- The weight of the scientific instrument in grams. -/
def instrument_weight_g : ℚ := 400

/-- Converts grams to kilograms. -/
def grams_to_kg (g : ℚ) : ℚ := g / 1000

/-- Calculates the cost of transporting a given weight in kilograms. -/
noncomputable def transport_cost (weight_kg : ℚ) : ℚ := weight_kg * cost_per_kg

/-- Theorem: The cost of transporting a 400 g scientific instrument is $7,200. -/
theorem transport_cost_400g_instrument :
  transport_cost (grams_to_kg instrument_weight_g) = 7200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_400g_instrument_l1024_102482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1024_102495

theorem diophantine_equation_solutions (m n : ℤ) :
  (m^2 + n) * (m + n^2) = (m + n)^3 ↔ 
  (m = 0) ∨
  (n = 0) ∨
  (m = -5 ∧ n = 2) ∨
  (m = -1 ∧ n = 1) ∨
  (m = 1 ∧ n = -1) ∨
  (m = 2 ∧ n = -5) ∨
  (m = 4 ∧ n = 11) ∨
  (m = 5 ∧ n = 7) ∨
  (m = 7 ∧ n = 5) ∨
  (m = 11 ∧ n = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1024_102495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_true_statements_even_proposition_relationships_l1024_102466

/-- Represents the truth value of a proposition -/
inductive PropValue
| True
| False

/-- Represents a proposition and its related forms -/
structure Proposition where
  original : PropValue
  inverse : PropValue
  negation : PropValue
  contrapositive : PropValue

/-- Counts the number of true statements in a Proposition -/
def countTrueStatements (p : Proposition) : Nat :=
  let count (v : PropValue) : Nat := match v with
    | PropValue.True => 1
    | PropValue.False => 0
  count p.original + count p.inverse + count p.negation + count p.contrapositive

/-- Theorem stating that the number of true statements is always even -/
theorem num_true_statements_even (p : Proposition) :
  ∃ n : Nat, n ∈ ({0, 2, 4} : Set Nat) ∧ countTrueStatements p = n :=
by
  sorry

/-- Theorem stating the logical relationships between the proposition forms -/
theorem proposition_relationships (p : Proposition) :
  (p.original = p.contrapositive) ∧
  (p.negation ≠ p.original) ∧
  (p.inverse = p.negation) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_true_statements_even_proposition_relationships_l1024_102466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_with_remainder_l1024_102499

theorem binomial_expansion_with_remainder (a b m : ℕ) (hm : m > 0) :
  ∃ S : ℤ, (a + b : ℤ) ^ m = a ^ m + m * a ^ (m - 1) * b + b ^ 2 * S :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_with_remainder_l1024_102499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_cos_2alpha_l1024_102445

theorem sin_2alpha_plus_cos_2alpha (α : Real) 
  (h1 : Real.sin α = 1/3) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.sin (2*α) + Real.cos (2*α) = (7 - 4*Real.sqrt 2) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_cos_2alpha_l1024_102445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1024_102428

-- Define the logarithm function
noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

-- Define a, b, and c as noncomputable
noncomputable def a : ℝ := log 11 10
noncomputable def b : ℝ := (log 11 9) ^ 2
noncomputable def c : ℝ := log 10 11

-- Theorem statement
theorem log_inequality : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1024_102428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_fractions_theorem_l1024_102493

/-- A type representing infinite decimal fractions -/
def InfiniteDecimalFraction := ℕ → Fin 10

/-- A function that checks if a sequence has an infinite number of a specific digit -/
def has_infinite_digit (seq : ℕ → Fin 10) (d : Fin 10) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ seq m = d

/-- The difference between two infinite decimal fractions -/
def difference (a b : InfiniteDecimalFraction) : ℕ → Fin 10 :=
  λ n ↦ (a n - b n) % 10

theorem eleven_fractions_theorem (fractions : Fin 11 → InfiniteDecimalFraction) :
  ∃ i j : Fin 11, i ≠ j ∧
    (has_infinite_digit (difference (fractions i) (fractions j)) 0 ∨
     has_infinite_digit (difference (fractions i) (fractions j)) 9) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_fractions_theorem_l1024_102493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_standard_deviation_l1024_102472

noncomputable def data : List ℝ := [1, 3, 2, 5, 4]

noncomputable def mean (l : List ℝ) : ℝ := (l.sum) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (fun x => (x - m) ^ 2)).sum / l.length

noncomputable def standardDeviation (l : List ℝ) : ℝ :=
  Real.sqrt (variance l)

theorem data_standard_deviation :
  mean data = 3 → standardDeviation data = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_standard_deviation_l1024_102472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_p_is_trisection_point_l1024_102413

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

-- Define points A, B, C, O, and P
variable (A B C O P : V)

-- Define the property of non-collinearity
def NonCollinear (A B C : V) : Prop :=
  ∀ (t : ℝ), A - B ≠ t • (C - B)

-- Define the centroid property
def Centroid (O A B C : V) : Prop :=
  O = (1/3 : ℝ) • (A + B + C)

-- Define the given condition for point P
def PointPCondition (O P A B C : V) : Prop :=
  P - O = (1/3 : ℝ) • ((1/2 : ℝ) • (A - O) + (1/2 : ℝ) • (B - O) + 2 • (C - O))

-- Define the trisection point of the median
def TrisectionPoint (P O C : V) : Prop :=
  P - O = (1/2 : ℝ) • (C - O)

-- State the theorem
theorem point_p_is_trisection_point
  (h_non_collinear : NonCollinear A B C)
  (h_centroid : Centroid O A B C)
  (h_p_condition : PointPCondition O P A B C) :
  TrisectionPoint P O C ∧ P ≠ O :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_p_is_trisection_point_l1024_102413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_specific_lines_l1024_102474

/-- Line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The first quadrant -/
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- Area below a line in the first quadrant -/
noncomputable def AreaBelow (l : Line) : ℝ :=
  (l.intercept ^ 2) / (2 * (-l.slope))

/-- Area between two lines in the first quadrant -/
noncomputable def AreaBetween (l1 l2 : Line) : ℝ :=
  AreaBelow l1 - AreaBelow l2

/-- The probability of a point falling between two lines -/
noncomputable def ProbabilityBetweenLines (l1 l2 : Line) : ℝ :=
  (AreaBetween l1 l2) / (AreaBelow l1)

theorem probability_between_specific_lines :
  let p : Line := { slope := -2, intercept := 8 }
  let q : Line := { slope := -3, intercept := 8 }
  ProbabilityBetweenLines p q = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_specific_lines_l1024_102474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_max_l1024_102404

theorem triangle_sine_sum_max (A B C : ℝ) : 
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  ConvexOn ℝ (Set.Ioo 0 π) Real.sin →
  Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_max_l1024_102404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_eccentricity_l1024_102475

-- Define the conic sections
def C₁ (m n x y : ℝ) : Prop := m * x^2 + n * y^2 = 1
def C₂ (p q x y : ℝ) : Prop := p * x^2 - q * y^2 = 1

-- Define the theorem
theorem conic_sections_eccentricity
  (m n p q : ℝ)
  (hm : m > 0)
  (hn : n > m)
  (hp : p > 0)
  (hq : q > 0)
  (hF : ∃ F₁ F₂ : ℝ × ℝ, ∀ x y : ℝ, C₁ m n x y ↔ C₂ p q x y)
  (hM : ∃ M : ℝ × ℝ, C₁ m n M.1 M.2 ∧ C₂ p q M.1 M.2)
  (hAngle : ∀ (M F₁ F₂ : ℝ × ℝ), C₁ m n M.1 M.2 → C₂ p q M.1 M.2 → 
    (M.1 - F₁.1) * (M.1 - F₂.1) + (M.2 - F₁.2) * (M.2 - F₂.2) = 0)
  (hEcc₁ : Real.sqrt (1 - m / n) = 3 / 4) :
  Real.sqrt (1 + q / p) = 3 * Real.sqrt 2 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_eccentricity_l1024_102475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_production_equation_l1024_102485

/-- Represents the production rate of medical masks in millions per week -/
def production_rate : ℝ → Prop := sorry

/-- Represents the total number of masks to be produced in millions -/
def total_masks : ℝ := 180

/-- Represents the increased production rate after the first week -/
def increased_rate (x : ℝ) : ℝ := 1.5 * x

/-- Represents the condition that the task is completed one week ahead of schedule -/
def ahead_of_schedule (x : ℝ) : Prop :=
  (total_masks - x) / x = (total_masks - x) / (increased_rate x) + 1

theorem mask_production_equation :
  ∀ x : ℝ,
  production_rate x →
  ahead_of_schedule x →
  (180 - x) / x = (180 - x) / (1.5 * x) + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_production_equation_l1024_102485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lighthouse_problem_l1024_102423

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

theorem lighthouse_problem (A B C D : Point) 
  (h1 : distance A B = 5)
  (h2 : distance B C = 12)
  (h3 : distance A C = 13)
  (h4 : angle B A D = angle C A D)
  (h5 : angle A C B = angle D C B) :
  ∃ (p q r : ℕ), 
    distance A D = (p * Real.sqrt q) / r ∧ 
    Nat.Coprime p q ∧
    Nat.Coprime p r ∧
    Nat.Coprime q r ∧
    (∀ (prime : ℕ), Nat.Prime prime → ¬(r % (prime * prime) = 0)) ∧
    p = 360 ∧
    q = 13 ∧
    r = 391 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lighthouse_problem_l1024_102423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_tangent_to_truncated_cone_l1024_102410

/-- The radius of a sphere tangent to a truncated cone -/
theorem sphere_radius_tangent_to_truncated_cone (r₁ r₂ : ℝ) (h₁ : r₁ = 24) (h₂ : r₂ = 4) :
  Real.sqrt 96 = Real.sqrt ((r₁ - r₂)^2 + (r₁ + r₂)^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_tangent_to_truncated_cone_l1024_102410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_millionth_digit_of_one_forty_first_l1024_102435

/-- The period of the decimal expansion of 1/41 -/
def period : ℕ := 5

/-- The repeating sequence in the decimal expansion of 1/41 -/
def repeating_sequence : List ℕ := [0, 2, 4, 3, 9]

/-- The millionth digit after the decimal point in 1/41 -/
def millionth_digit : ℕ := 9

/-- Theorem stating that the millionth digit after the decimal point in 1/41 is 9 -/
theorem millionth_digit_of_one_forty_first :
  (1000000 % period = 0) ∧ 
  (repeating_sequence.length = period) ∧
  (repeating_sequence.getLast? = some millionth_digit) →
  millionth_digit = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_millionth_digit_of_one_forty_first_l1024_102435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_theorem_l1024_102438

noncomputable section

open Real

def sequence_property (a b : ℕ → ℝ) : Prop :=
  ∀ n, sin (a (n + 1)) = sin (a n) + cos (b n) ∧
       cos (b (n + 1)) = cos (b n) - sin (a n)

theorem sequence_theorem (a b : ℕ → ℝ) (h : sequence_property a b) :
  (∀ n, (sin (a (n + 1)))^2 + (cos (b (n + 1)))^2 = 2 * ((sin (a n))^2 + (cos (b n))^2)) ∧
  ¬ ∃ (a₁ b₁ : ℝ) (r : ℝ), r ≠ 2 ∧
    (∀ n, (sin (a n))^2 + (cos (b n))^2 = r^(n - 1) * ((sin a₁)^2 + (cos b₁)^2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_theorem_l1024_102438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_lines_intersecting_plane_l1024_102421

/-- A plane in 3D space -/
structure Plane where

/-- A line in 3D space -/
structure Line where

/-- The angle between a line and a plane -/
def angle_line_plane (l : Line) (p : Plane) : ℝ := sorry

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- A line lies in a plane -/
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem three_lines_intersecting_plane :
  ∀ (S₁ S₂ : Plane) (e₁ e₂ e₃ : Line),
    (e₁ ≠ e₂ ∧ e₁ ≠ e₃ ∧ e₂ ≠ e₃) →  -- Three distinct lines
    (line_in_plane e₁ S₁ ∧ line_in_plane e₂ S₁ ∧ line_in_plane e₃ S₁) →  -- Lines lie in plane S₁
    (angle_line_plane e₁ S₂ = 45 ∧ 
     angle_line_plane e₂ S₂ = 45 ∧ 
     angle_line_plane e₃ S₂ = 45) →  -- Each line intersects S₂ at 45°
    (parallel e₁ e₂ ∨ parallel e₁ e₃ ∨ parallel e₂ e₃) :=  -- Two lines are parallel
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_lines_intersecting_plane_l1024_102421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_position_after_120_moves_l1024_102468

/-- The initial position of the particle --/
def initial_position : ℂ := 3

/-- The rotation factor for each move --/
noncomputable def omega : ℂ := Complex.exp (Complex.I * (Real.pi / 3))

/-- The translation distance for each move --/
def translation : ℝ := 8

/-- The number of moves --/
def num_moves : ℕ := 120

/-- The position after a single move --/
noncomputable def move (z : ℂ) : ℂ := omega * z + translation

/-- The position after n moves --/
noncomputable def position (n : ℕ) : ℂ :=
  (move^[n]) initial_position

/-- Theorem stating that the position after 120 moves is the same as the initial position --/
theorem position_after_120_moves :
  position num_moves = initial_position := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_position_after_120_moves_l1024_102468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_addition_problem_l1024_102476

/-- Represents a digit in base 6 --/
def Base6Digit := Fin 6

/-- Addition in base 6 without carry --/
def add_base6 (a b : Base6Digit) : Base6Digit :=
  ⟨(a.val + b.val) % 6, by sorry⟩

/-- Conversion from Base6Digit to ℕ --/
def to_nat (d : Base6Digit) : ℕ := d.val

/-- Helper function to create a Base6Digit from a Nat --/
def nat_to_base6 (n : ℕ) : Base6Digit :=
  ⟨n % 6, by sorry⟩

theorem base6_addition_problem (X Y : Base6Digit) :
  (add_base6 Y (nat_to_base6 3) = X) →
  (add_base6 X (nat_to_base6 5) = nat_to_base6 2) →
  (to_nat X + to_nat Y = 3) :=
by sorry

#eval nat_to_base6 3
#eval nat_to_base6 5
#eval nat_to_base6 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_addition_problem_l1024_102476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bev_distance_to_halfway_l1024_102453

/-- The distance Bev needs to drive to reach the halfway point of her journey -/
noncomputable def distance_to_halfway (driven : ℝ) (remaining : ℝ) : ℝ :=
  (driven + remaining) / 2 - driven

/-- Theorem stating the additional distance Bev needs to drive to reach the halfway point -/
theorem bev_distance_to_halfway :
  distance_to_halfway 312 858 = 273 := by
  -- Unfold the definition of distance_to_halfway
  unfold distance_to_halfway
  -- Simplify the arithmetic expression
  simp [add_div]
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bev_distance_to_halfway_l1024_102453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1024_102408

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the center and radius of C1
def center_C1 : ℝ × ℝ := (-1, -1)
def radius_C1 : ℝ := 2

-- Define the center and radius of C2
def center_C2 : ℝ × ℝ := (2, 1)
def radius_C2 : ℝ := 2

-- Calculate the distance between the centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 13

-- Theorem statement
theorem circles_intersect :
  0 < distance_between_centers ∧
  distance_between_centers < radius_C1 + radius_C2 ∧
  distance_between_centers > |radius_C1 - radius_C2| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1024_102408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_no_lattice_points_l1024_102456

/-- A line passes through a lattice point if there exist integers x and y satisfying the line equation -/
def passes_through_lattice_point (m : ℚ) : Prop :=
  ∃ x y : ℤ, 1 ≤ x ∧ x ≤ 50 ∧ y = ⌊m * x + 3⌋

/-- The maximum value of a such that no line y = mx + 3 with 2/5 < m < a
    passes through a lattice point for 1 ≤ x ≤ 50 -/
theorem max_a_no_lattice_points :
  (∀ m : ℚ, 2/5 < m → m < 22/51 → ¬ passes_through_lattice_point m) ∧
  (∀ ε > 0, ∃ m : ℚ, 2/5 < m ∧ m < 22/51 + ε ∧ passes_through_lattice_point m) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_no_lattice_points_l1024_102456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_l1024_102402

noncomputable section

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2) / a

-- Define the minor axis length
def minor_axis_length (b : ℝ) : ℝ := 2 * b

-- Define the line l passing through M(-2, 0)
def line_through_M (m x y : ℝ) : Prop := x = m * y - 2

-- Define the area of triangle OPQ
noncomputable def area_OPQ (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (1/2) * abs (x₁ * y₂ - x₂ * y₁)

theorem ellipse_and_max_area 
  (a b : ℝ) 
  (h_positive : a > b ∧ b > 0) 
  (h_minor_axis : minor_axis_length b = 2) 
  (h_eccentricity : eccentricity a b = Real.sqrt 2 / 2) :
  (∀ x y, ellipse x y a b ↔ ellipse x y (Real.sqrt 2) 1) ∧ 
  (∃ max_area : ℝ, max_area = Real.sqrt 2 / 2 ∧ 
    ∀ m x₁ y₁ x₂ y₂, 
      line_through_M m x₁ y₁ → 
      line_through_M m x₂ y₂ → 
      ellipse x₁ y₁ (Real.sqrt 2) 1 → 
      ellipse x₂ y₂ (Real.sqrt 2) 1 → 
      area_OPQ x₁ y₁ x₂ y₂ ≤ max_area) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_l1024_102402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l1024_102463

-- Define the functions for the two curves
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x
def g (x : ℝ) : ℝ := x^2

-- State the theorem
theorem area_between_curves : 
  (∫ x in (Set.Icc 0 1), f x - g x) = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l1024_102463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_calculation_l1024_102479

theorem restaurant_bill_calculation (adults children : ℕ) 
  (adult_meal_cost child_meal_cost dessert_cost discount_rate : ℚ) :
  adults = 2 →
  children = 5 →
  adult_meal_cost = 7 →
  child_meal_cost = 3 →
  dessert_cost = 2 →
  discount_rate = 15/100 →
  let total_before_discount := adults * adult_meal_cost + children * child_meal_cost + 
    (adults + children) * dessert_cost
  let discount_amount := total_before_discount * discount_rate
  let total_after_discount := total_before_discount - discount_amount
  total_after_discount = 3655/100 := by
  sorry

#eval (2 : ℕ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_calculation_l1024_102479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_monotonic_subsequence_l1024_102439

/-- Given an infinite sequence of real numbers, there exists an infinite monotonic subsequence. -/
theorem infinite_monotonic_subsequence (s : ℕ → ℝ) :
  (∃ f : ℕ → ℕ, StrictMono f ∧ StrictMono (s ∘ f)) ∨
  (∃ f : ℕ → ℕ, StrictMono f ∧ Antitone (s ∘ f)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_monotonic_subsequence_l1024_102439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_squares_l1024_102430

/-- Given a square ABCD with side length 4, and points E, F, G, H as centers of right-angled 
    isosceles triangles with hypotenuses AB, BC, CD, and DA respectively, exterior to the square,
    prove that the ratio of the area of square EFGH to the area of square ABCD is (3 + 2√2) / 4 -/
theorem area_ratio_squares (A B C D E F G H : ℝ × ℝ) : 
  let s := 4
  -- ABCD is a square with side length 4
  (B.1 - A.1 = s ∧ B.2 = A.2) →
  (C.1 = B.1 ∧ C.2 - B.2 = s) →
  (D.1 = A.1 ∧ D.2 = C.2) →
  (A.1 = D.1 ∧ A.2 = B.2) →
  -- E, F, G, H are centers of right-angled isosceles triangles
  (E.1 = (A.1 + B.1) / 2 ∧ E.2 = A.2 - s / Real.sqrt 2) →
  (F.1 = B.1 + s / Real.sqrt 2 ∧ F.2 = (B.2 + C.2) / 2) →
  (G.1 = (C.1 + D.1) / 2 ∧ G.2 = C.2 + s / Real.sqrt 2) →
  (H.1 = A.1 - s / Real.sqrt 2 ∧ H.2 = (A.2 + D.2) / 2) →
  -- Ratio of areas
  ∃ (area_EFGH area_ABCD : ℝ), area_EFGH / area_ABCD = (3 + 2 * Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_squares_l1024_102430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_mowing_hours_l1024_102487

/-- Calculates the number of hours Jerry spent mowing the lawn given the following conditions:
  * Jerry charges $15 per hour
  * Jerry spent 8 hours painting the house
  * Time to fix the kitchen counter was 3 times longer than painting
  * Miss Stevie paid $570 in total
-/
noncomputable def hours_mowing_lawn (hourly_rate : ℚ) (painting_hours : ℚ) (counter_multiplier : ℚ) (total_paid : ℚ) : ℚ :=
  (total_paid - hourly_rate * (painting_hours + counter_multiplier * painting_hours)) / hourly_rate

/-- Proves that Jerry spent 6 hours mowing the lawn given the specified conditions -/
theorem jerry_mowing_hours :
  hours_mowing_lawn 15 8 3 570 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_mowing_hours_l1024_102487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l1024_102417

-- Define points E and F
def E : ℝ × ℝ := (-3, -2)
def F : ℝ × ℝ := (5, 10)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define point D
def D : ℝ × ℝ := (3, 7)

-- Theorem statement
theorem point_D_coordinates :
  ∃ (D : ℝ × ℝ), D.1 ∈ Set.Icc E.1 F.1 ∧ D.2 ∈ Set.Icc E.2 F.2 ∧
  distance E D = 2 * distance D F ∧
  D = (3, 7) := by
  use D
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l1024_102417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l1024_102473

-- Define the universal set U
def U : Set ℕ := {x | x ≤ 5}

-- Define set A
def A : Set ℕ := {x | 2 * x - 5 < 0}

-- Theorem statement
theorem complement_of_A : (U \ A) = {3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l1024_102473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_major_axis_length_focal_distance_value_eccentricity_value_l1024_102491

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the semi-major and semi-minor axes
def semi_major_axis : ℝ := 4
def semi_minor_axis : ℝ := 2

-- Theorem for the length of the major axis
theorem major_axis_length : 
  2 * semi_major_axis = 8 := by sorry

-- Define the focal distance
noncomputable def focal_distance : ℝ := Real.sqrt (semi_major_axis^2 - semi_minor_axis^2)

-- Theorem for the focal distance
theorem focal_distance_value : 
  2 * focal_distance = 4 * Real.sqrt 3 := by sorry

-- Define the eccentricity
noncomputable def eccentricity : ℝ := focal_distance / semi_major_axis

-- Theorem for the eccentricity
theorem eccentricity_value : 
  eccentricity = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_major_axis_length_focal_distance_value_eccentricity_value_l1024_102491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_removable_count_l1024_102437

/-- Given a natural number n, returns the set of natural numbers from 1 to n -/
def numberSet (n : ℕ) : Finset ℕ := Finset.range n

/-- Checks if there exist two numbers in the set where one divides the other -/
def hasDivisiblePair (s : Finset ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ (a ∣ b ∨ b ∣ a)

/-- The main theorem stating that 499 is the largest number satisfying the condition -/
theorem largest_removable_count : 
  (∀ (s : Finset ℕ), s ⊆ numberSet 1000 → s.card = 501 → hasDivisiblePair s) ∧
  (∀ (m : ℕ), m > 499 → ∃ (s : Finset ℕ), s ⊆ numberSet 1000 ∧ s.card = 1000 - m ∧ ¬hasDivisiblePair s) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_removable_count_l1024_102437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bonus_is_120_l1024_102458

/-- Calculates the bonus received by a salesman given the total commission, base commission rate, bonus rate, and threshold for bonus. -/
noncomputable def calculate_bonus (total_commission : ℝ) (base_rate : ℝ) (bonus_rate : ℝ) (threshold : ℝ) : ℝ :=
  let total_sales := total_commission / (base_rate + bonus_rate) + threshold * bonus_rate / (base_rate + bonus_rate)
  bonus_rate * (total_sales - threshold)

/-- Theorem stating that under given conditions, the bonus received is 120. -/
theorem bonus_is_120 :
  calculate_bonus 1380 0.09 0.03 10000 = 120 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bonus_is_120_l1024_102458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_residue_system_moduli_l1024_102441

def is_valid_modulus (n : ℕ) : Prop :=
  Nat.Coprime 1 n ∧ 
  Nat.Coprime 5 n ∧ 
  (1 % n ≠ 5 % n) ∧ 
  (Nat.totient n = 2)

theorem reduced_residue_system_moduli : 
  ∀ n : ℕ, n > 0 → (is_valid_modulus n ↔ (n = 3 ∨ n = 6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_residue_system_moduli_l1024_102441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1024_102403

/-- A circle with center on the y-axis, radius 1, passing through (1,2) has equation x² + (y-2)² = 1 -/
theorem circle_equation : ∃ b : ℝ, 
  (∀ x y : ℝ, x^2 + (y - b)^2 = 1 ↔ (x, y) ∈ Metric.sphere (0 : ℝ × ℝ) 1) ∧ 
  (1, 2) ∈ Metric.sphere (0, b) 1 ∧
  (∀ x y : ℝ, x^2 + (y - 2)^2 = 1 ↔ (x, y) ∈ Metric.sphere (0, b) 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1024_102403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1024_102457

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x) / 2 - a / (Real.exp x)

theorem range_of_a :
  ∀ a : ℝ,
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → x₁ ≠ x₂ →
    (abs (f a x₁) - abs (f a x₂)) * (x₁ - x₂) > 0) →
  a ∈ Set.Icc (-(Real.exp 2) / 2) ((Real.exp 2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1024_102457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l1024_102483

/-- The time (in seconds) it takes for a train to pass a man moving in the opposite direction -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  train_length / ((train_speed + man_speed) * (5/18))

/-- Theorem: A 140 m long train moving at 50 km/hr passes a man moving at 4 km/hr 
    in the opposite direction in approximately 9.33 seconds -/
theorem train_passing_man_time :
  ∃ ε > 0, |train_passing_time 140 50 4 - 9.33| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l1024_102483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_exponential_difference_l1024_102469

theorem max_value_of_exponential_difference :
  (∀ x : ℝ, 2^x - 4^x ≤ (1/4 : ℝ)) ∧ (2^(-1 : ℝ) - 4^(-1 : ℝ) = (1/4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_exponential_difference_l1024_102469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_C_in_rectangle_Y_l1024_102498

-- Define the tiles and rectangles
inductive Tile | A | B | C | D
inductive Rectangle | W | X | Y | Z

-- Define the sides of a tile
inductive Side | Top | Right | Bottom | Left

-- Define the number on each side of each tile
def tileNumber (t : Tile) (s : Side) : Nat :=
  match t, s with
  | Tile.A, Side.Top => 6
  | Tile.A, Side.Right => 4
  | Tile.A, Side.Bottom => 1
  | Tile.A, Side.Left => 3
  | Tile.B, Side.Top => 1
  | Tile.B, Side.Right => 2
  | Tile.B, Side.Bottom => 5
  | Tile.B, Side.Left => 6
  | Tile.C, Side.Top => 5
  | Tile.C, Side.Right => 6
  | Tile.C, Side.Bottom => 3
  | Tile.C, Side.Left => 4
  | Tile.D, Side.Top => 4
  | Tile.D, Side.Right => 5
  | Tile.D, Side.Bottom => 2
  | Tile.D, Side.Left => 1

-- Define the arrangement of tiles
def arrangement : Tile → Rectangle := sorry

-- Define adjacency of rectangles
def adjacent : Rectangle → Rectangle → Prop := sorry

-- Axioms
axiom arrangement_injective : Function.Injective arrangement

axiom adjacent_numbers_match (t1 t2 : Tile) (r1 r2 : Rectangle) (s1 s2 : Side) :
  adjacent r1 r2 → arrangement t1 = r1 → arrangement t2 = r2 →
  ((s1 = Side.Right ∧ s2 = Side.Left) ∨ (s1 = Side.Left ∧ s2 = Side.Right) ∨
   (s1 = Side.Top ∧ s2 = Side.Bottom) ∨ (s1 = Side.Bottom ∧ s2 = Side.Top)) →
  tileNumber t1 s1 = tileNumber t2 s2

-- Theorem: Tile C is placed in Rectangle Y
theorem tile_C_in_rectangle_Y : arrangement Tile.C = Rectangle.Y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_C_in_rectangle_Y_l1024_102498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_pi_over_four_l1024_102426

noncomputable section

/-- The area of the region bounded by r = sin(φ), r = √2 * cos(φ - π/4), and 0 ≤ φ ≤ 3π/4 -/
def bounded_area : ℝ := Real.pi / 4

/-- First curve in polar coordinates -/
def r₁ (φ : ℝ) : ℝ := Real.sin φ

/-- Second curve in polar coordinates -/
def r₂ (φ : ℝ) : ℝ := Real.sqrt 2 * Real.cos (φ - Real.pi / 4)

/-- Lower bound of φ -/
def φ_lower : ℝ := 0

/-- Upper bound of φ -/
def φ_upper : ℝ := 3 * Real.pi / 4

theorem area_equals_pi_over_four :
  bounded_area = ∫ φ in φ_lower..φ_upper, (1 / 2) * (min (r₁ φ) (r₂ φ))^2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_pi_over_four_l1024_102426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_extreme_point_property_l1024_102492

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - a*x - b

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3*x^2 - a

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a b : ℝ) (ha : a > 0) :
  ∀ x : ℝ, 
    (x < -Real.sqrt (3*a) / 3 → f' a x > 0) ∧
    (-Real.sqrt (3*a) / 3 < x ∧ x < Real.sqrt (3*a) / 3 → f' a x < 0) ∧
    (x > Real.sqrt (3*a) / 3 → f' a x > 0) :=
sorry

-- Theorem for extreme points
theorem extreme_point_property (a b : ℝ) (x₀ x₁ : ℝ) 
  (h_extreme : f' a x₀ = 0)
  (h_distinct : x₁ ≠ x₀)
  (h_equal_value : f a b x₁ = f a b x₀) :
  x₁ + 2*x₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_extreme_point_property_l1024_102492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationship_possibilities_l1024_102444

-- Define a type for lines in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  -- But for this abstract problem, we'll just use an opaque type
  mk :: (dummy : Unit)

-- Define a type for points in 3D space
structure Point3D where
  mk :: (dummy : Unit)

-- Define a type for vectors in 3D space
structure Vector3D where
  mk :: (dummy : Unit)

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  -- In reality, this would involve a more complex definition
  -- For now, we'll leave it as an axiom
  sorry

-- Define membership for points in lines
def point_on_line (p : Point3D) (l : Line3D) : Prop :=
  sorry

-- Define vector addition for points
def point_add_vector (p : Point3D) (v : Vector3D) : Point3D :=
  sorry

-- Define zero vector
def zero_vector : Vector3D :=
  sorry

-- Main theorem
theorem line_relationship_possibilities (a b c : Line3D) 
  (h1 : are_skew a b) (h2 : are_skew b c) :
  (are_skew a c ∨ ∃ (p : Point3D), point_on_line p a ∧ point_on_line p c) ∨ 
  ∃ (v : Vector3D), v ≠ zero_vector ∧ ∀ (p : Point3D), point_on_line p a ↔ point_on_line (point_add_vector p v) c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationship_possibilities_l1024_102444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_eq_neg_one_l1024_102446

noncomputable def f (x m : ℝ) : ℝ := Real.log ((x + 1) / (x - 1)) + m + 1

theorem odd_function_implies_m_eq_neg_one (m : ℝ) :
  (∀ x, x ∈ Set.Ioi 1 ∪ Set.Iio (-1) → f (-x) m = -f x m) →
  m = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_eq_neg_one_l1024_102446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_has_twelve_edges_l1024_102401

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where

/-- The number of edges in a geometric shape -/
def num_edges (shape : Type) : ℕ := 
  sorry -- We'll leave this unimplemented for now

/-- Theorem: A cube has 12 edges -/
theorem cube_has_twelve_edges (c : Cube) : num_edges Cube = 12 := by
  sorry

#check cube_has_twelve_edges

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_has_twelve_edges_l1024_102401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_resolvable_debt_l1024_102496

/-- The value of a pig in dollars -/
def pig_value : ℕ := 350

/-- The value of a goat in dollars -/
def goat_value : ℕ := 240

/-- The smallest resolvable debt in dollars -/
def smallest_debt : ℕ := 10

/-- Theorem stating that the smallest resolvable debt is correct -/
theorem smallest_resolvable_debt :
  ∃ (p g : ℤ), (smallest_debt : ℤ) = pig_value * p + goat_value * g ∧
  ∀ (d : ℕ) (p' g' : ℤ), d > 0 ∧ d < smallest_debt →
    (d : ℤ) ≠ pig_value * p' + goat_value * g' :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_resolvable_debt_l1024_102496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l1024_102462

theorem no_integer_solution (m n : ℤ) : (10 : ℝ) ^ m ≠ 25 ^ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l1024_102462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_ohara_triple_16_64_9_l1024_102477

/-- Modified O'Hara triple -/
def is_modified_ohara_triple (a b c x : ℕ) : Prop :=
  Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) + Real.sqrt (c : ℝ) = x

/-- Theorem: If (16, 64, 9, x) is a Modified O'Hara triple, then x = 15 -/
theorem modified_ohara_triple_16_64_9 (x : ℕ) :
  is_modified_ohara_triple 16 64 9 x → x = 15 := by
  sorry

#check modified_ohara_triple_16_64_9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_ohara_triple_16_64_9_l1024_102477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_theorem_l1024_102411

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with parametric equations -/
structure Ellipse where
  center : Point
  a : ℝ
  b : ℝ

/-- Calculates the foci of an ellipse -/
noncomputable def calculateFoci (e : Ellipse) : (Point × Point) :=
  let c := (e.a^2 - e.b^2).sqrt
  let focus1 := Point.mk (e.center.x - c) e.center.y
  let focus2 := Point.mk (e.center.x + c) e.center.y
  (focus1, focus2)

theorem ellipse_foci_theorem (e : Ellipse) : 
  e.center = Point.mk 2 (-3) → e.a = 5 → e.b = 4 → 
  calculateFoci e = (Point.mk (-1) (-3), Point.mk 5 (-3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_theorem_l1024_102411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_set_with_division_property_l1024_102465

theorem integer_set_with_division_property (n : ℕ) (h : n ≥ 2) :
  ∃ (S : Finset ℤ), (Finset.card S = n) ∧
  (∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → (a - b)^2 ∣ (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_set_with_division_property_l1024_102465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_luggage_capacity_proof_l1024_102481

/-- Given an airplane trip with the following conditions:
  * There are 6 people
  * Each person has 5 bags
  * Each bag weighs 50 pounds (maximum weight)
  * The airplane can hold a total luggage weight of 6000 pounds
  Prove that the plane can hold 90 more bags at maximum weight. -/
def airplane_luggage_capacity 
  (num_people : Nat) 
  (bags_per_person : Nat) 
  (bag_weight : Nat) 
  (total_capacity : Nat) : Nat :=
  let total_bags := num_people * bags_per_person
  let current_weight := total_bags * bag_weight
  let remaining_capacity := total_capacity - current_weight
  remaining_capacity / bag_weight

theorem airplane_luggage_capacity_proof 
  (num_people : Nat) 
  (bags_per_person : Nat) 
  (bag_weight : Nat) 
  (total_capacity : Nat) :
  airplane_luggage_capacity num_people bags_per_person bag_weight total_capacity = 90 :=
by
  -- Proof goes here
  sorry

#eval airplane_luggage_capacity 6 5 50 6000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_luggage_capacity_proof_l1024_102481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parabolas_properties_l1024_102486

/-- Two parabolas with a common focus and perpendicular axes -/
structure PerpendicularParabolas where
  /-- Parameter of the first parabola -/
  p : ℝ
  /-- Parameter of the second parabola -/
  q : ℝ
  /-- Assumption that p and q are positive -/
  p_pos : p > 0
  q_pos : q > 0

/-- Point on the common tangent -/
structure CommonTangentPoint where
  x : ℝ
  y : ℝ
  /-- The point satisfies the equation of the common tangent -/
  on_common_tangent : q * x + p * y + (p^2 + q^2) / 2 = 0

/-- Secondary tangent point on the first parabola -/
noncomputable def secondary_tangent_point1 (pp : PerpendicularParabolas) (ctp : CommonTangentPoint) : ℝ × ℝ :=
  ((ctp.x * pp.p + pp.q * ctp.y) / pp.p, (ctp.x^2 - pp.p^2) / (2 * pp.p))

/-- Secondary tangent point on the second parabola -/
noncomputable def secondary_tangent_point2 (pp : PerpendicularParabolas) (ctp : CommonTangentPoint) : ℝ × ℝ :=
  ((ctp.y^2 - pp.q^2) / (2 * pp.q), (ctp.y * pp.q + pp.p * ctp.x) / pp.q)

/-- Main theorem: Common tangent and perpendicular secondary tangents -/
theorem perpendicular_parabolas_properties (pp : PerpendicularParabolas) :
  (∃ (a b c : ℝ), ∀ (x y : ℝ), a * x + b * y + c = 0 →
    (x^2 = 2 * pp.p * y + pp.p^2 ∨ y^2 = 2 * pp.q * x + pp.q^2)) ∧
  (∀ (ctp : CommonTangentPoint),
    let (x1, y1) := secondary_tangent_point1 pp ctp
    let (x2, y2) := secondary_tangent_point2 pp ctp
    (x1 - ctp.x) * (x2 - ctp.x) + (y1 - ctp.y) * (y2 - ctp.y) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parabolas_properties_l1024_102486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_m_value_l1024_102443

theorem inequality_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, x > 0 → x^2 - 2*(m^2 + m + 1)*Real.log x ≥ 1) → 
  (m = 0 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_m_value_l1024_102443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l1024_102414

/-- The sum of the infinite series Σ(n/5^n) from n=1 to ∞ -/
noncomputable def series_sum : ℝ := ∑' n, n / (5 ^ n : ℝ)

/-- The theorem stating that the sum of the series is 5/16 -/
theorem series_sum_value : series_sum = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l1024_102414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1024_102400

/-- An ellipse with specific eccentricity and focal distance -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h0 : a > 0
  h1 : b > 0
  h2 : a > b
  h3 : (a^2 - b^2) / a^2 = 6 / 9  -- eccentricity squared
  h4 : a - Real.sqrt (a^2 - b^2) = Real.sqrt 3     -- distance from minor axis end to right focus

/-- A line passing through the left focus of the ellipse with slope 1 -/
def leftFocusLine (e : SpecialEllipse) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + Real.sqrt (e.a^2 - e.b^2)}

/-- The intersection points of the line with the ellipse -/
def intersectionPoints (e : SpecialEllipse) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ leftFocusLine e ∧ p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1}

/-- The theorem stating the length of the chord -/
theorem chord_length (e : SpecialEllipse) :
  let points := intersectionPoints e
  ∃ p q : ℝ × ℝ, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1024_102400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_to_even_function_l1024_102452

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + 2 * Real.pi / 3)

-- Define the shifted function
noncomputable def g (x : ℝ) : ℝ := f (x + 5 * Real.pi / 12)

-- Theorem statement
theorem shift_to_even_function :
  ∀ x : ℝ, g x = g (-x) := by
  intro x
  -- Expand the definitions of g and f
  unfold g f
  -- Use trigonometric identities to simplify
  simp [Real.sin_add, Real.cos_add]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_to_even_function_l1024_102452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_equivalence_l1024_102415

-- Define the types for our variables
variable (a b : ℝ)

-- Define the solution set of ax - b > 0
def solution_set_1 (a b : ℝ) : Set ℝ := {x | a * x - b > 0}

-- Define the solution set of ax^2 + bx > 0
def solution_set_2 (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x > 0}

-- State the theorem
theorem solution_sets_equivalence (a b : ℝ) :
  solution_set_1 a b = Set.Ioi (-1) →
  solution_set_2 a b = Set.Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_equivalence_l1024_102415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equations_max_area_triangle_CPQ_line_equations_max_area_l1024_102497

-- Define the circle C
noncomputable def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define point A
def point_A : ℝ × ℝ := (1, 0)

-- Define a line passing through point A
noncomputable def line_through_A (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ k : ℝ) : ℝ :=
  |k * x₀ - y₀ - k| / Real.sqrt (k^2 + 1)

-- Theorem for tangent line equations
theorem tangent_line_equations :
  ∃ (k : ℝ), (∀ x y, line_through_A k x y → distance_point_to_line 3 4 k = 2) ∧
  (k = 0 ∨ k = 3/4) := by sorry

-- Theorem for maximum area of triangle CPQ
theorem max_area_triangle_CPQ :
  ∃ (k : ℝ), (∀ x y, line_through_A k x y →
    ∃ (d : ℝ), d = distance_point_to_line 3 4 k ∧
    d * Real.sqrt (4 - d^2) ≤ 2) := by sorry

-- Theorem for line equations when area is maximum
theorem line_equations_max_area :
  ∃ (k₁ k₂ : ℝ), (k₁ = 1 ∧ k₂ = 7) ∧
  (∀ x y, line_through_A k₁ x y ∨ line_through_A k₂ x y →
    ∃ (d : ℝ), d = distance_point_to_line 3 4 k₁ ∧
    d * Real.sqrt (4 - d^2) = 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equations_max_area_triangle_CPQ_line_equations_max_area_l1024_102497

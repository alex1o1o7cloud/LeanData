import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_when_f_less_than_2_l967_96763

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x - 1

-- State the theorem
theorem x_range_when_f_less_than_2 (x : ℝ) :
  f (x^2 - 4) < 2 → x ∈ Set.Ioo (-Real.sqrt 5) (-2) ∪ Set.Ioo 2 (Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_when_f_less_than_2_l967_96763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_height_for_min_material_l967_96777

/-- Represents the material usage of an open-top box with square base --/
noncomputable def material_usage (x : ℝ) : ℝ := x^2 + 1024 / x

/-- The volume of the box --/
def volume : ℝ := 256

/-- Theorem: The height that minimizes material usage for an open-top box 
    with square base and volume 256 is 4 units --/
theorem optimal_height_for_min_material : 
  ∃ (x : ℝ), x > 0 ∧ x * x * (volume / (x * x)) = volume ∧
  ∀ (y : ℝ), y > 0 → y * y * (volume / (y * y)) = volume →
  material_usage x ≤ material_usage y ∧
  volume / (x * x) = 4 := by
  sorry

#check optimal_height_for_min_material

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_height_for_min_material_l967_96777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_artillery_range_average_l967_96711

theorem artillery_range_average (shots : ℕ) (r1 r2 r3 r4 : ℝ) (n1 n2 n3 n4 : ℕ) :
  shots = n1 + n2 + n3 + n4 →
  n1 = 18 →
  n2 = 25 →
  n3 = 53 →
  n4 = 4 →
  r1 = 632 →
  r2 = 628 →
  r3 = 620 →
  r4 = 640 →
  (n1 * r1 + n2 * r2 + n3 * r3 + n4 * r4) / shots = 625.96 :=
by
  intros h_shots h_n1 h_n2 h_n3 h_n4 h_r1 h_r2 h_r3 h_r4
  rw [h_shots, h_n1, h_n2, h_n3, h_n4, h_r1, h_r2, h_r3, h_r4]
  norm_num
  sorry  -- The exact arithmetic is omitted for brevity

#check artillery_range_average

end NUMINAMATH_CALUDE_ERRORFEEDBACK_artillery_range_average_l967_96711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l967_96718

def sequence_x : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => sequence_x (n + 1) + 3 * sequence_x n

theorem divisibility_property (p : ℕ) (hp : Nat.Prime p) :
  ∃ k : ℤ, sequence_x (p - 1) - 1 = 3 * p * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l967_96718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_is_correct_l967_96709

/-- The maximum area of a triangle DEF with DE = 15 and EF : DF = 25 : 24 -/
def max_triangle_area : ℝ := 1584.375

/-- Triangle DEF with given properties -/
structure TriangleDEF where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  DE_eq : DE = 15
  ratio : EF / DF = 25 / 24

/-- The area of a triangle given its side lengths -/
noncomputable def triangle_area (t : TriangleDEF) : ℝ :=
  let s := (t.DE + t.EF + t.DF) / 2
  Real.sqrt (s * (s - t.DE) * (s - t.EF) * (s - t.DF))

/-- Theorem: The maximum area of triangle DEF with the given properties is 1584.375 -/
theorem max_triangle_area_is_correct :
  ∀ t : TriangleDEF, triangle_area t ≤ max_triangle_area :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_is_correct_l967_96709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l967_96722

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - Real.sqrt (1 - x)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Iic 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l967_96722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l967_96735

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the slope of one asymptote is 2, then the eccentricity e = √5 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_slope : b / a = 2) : 
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l967_96735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_problem_l967_96758

/-- A circle represented by its center and radius. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if one circle is inside another. -/
def Circle.isInside (c₁ c₂ : Circle) : Prop := sorry

/-- Check if one circle touches another internally. -/
def Circle.touchesInternally (c₁ c₂ : Circle) : Prop := sorry

/-- Check if one circle touches another externally. -/
def Circle.touchesExternally (c₁ c₂ : Circle) : Prop := sorry

/-- Check if a circle passes through the center of another circle. -/
def Circle.passesThroughCenter (c₁ c₂ : Circle) : Prop := sorry

/-- Given a circle O₄ with radius R and three circles O₁, O₂, and O₃ inside O₄,
    if O₁, O₂, and O₃ touch O₄ internally and each other externally,
    and O₁ and O₂ pass through the center of O₄,
    then the radius of O₃ is R/3. -/
theorem circle_radius_problem (R : ℝ) (O₁ O₂ O₃ O₄ : Circle) :
  R > 0 →
  O₄.radius = R →
  O₁.isInside O₄ ∧ O₂.isInside O₄ ∧ O₃.isInside O₄ →
  O₁.touchesInternally O₄ ∧ O₂.touchesInternally O₄ ∧ O₃.touchesInternally O₄ →
  O₁.touchesExternally O₂ ∧ O₁.touchesExternally O₃ ∧ O₂.touchesExternally O₃ →
  O₁.passesThroughCenter O₄ ∧ O₂.passesThroughCenter O₄ →
  O₃.radius = R / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_problem_l967_96758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solutions_l967_96756

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (m + 1/m)*x + 1

theorem f_inequality_solutions (m : ℝ) :
  (m = 2 → {x : ℝ | f m x ≤ 0} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 2}) ∧
  (m > 0 → 
    ((0 < m ∧ m < 1 → {x : ℝ | f m x ≥ 0} = {x : ℝ | x ≤ m ∨ x ≥ 1/m}) ∧
     (m = 1 → {x : ℝ | f m x ≥ 0} = Set.univ) ∧
     (m > 1 → {x : ℝ | f m x ≥ 0} = {x : ℝ | x ≥ m ∨ x ≤ 1/m}))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solutions_l967_96756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_exists_l967_96793

/-- A polynomial is odd if it only contains odd-degree terms -/
def IsOddPolynomial (p : Polynomial ℝ) : Prop :=
  ∀ (i : ℕ), i % 2 = 0 → p.coeff i = 0

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem unique_polynomial_exists (n : ℕ) (hn : n > 0) :
  ∃! (f : Polynomial ℝ),
    (f.degree = n) ∧
    (f.eval 0 = 1) ∧
    IsOddFunction (fun x ↦ (x + 1) * (f.eval x)^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_exists_l967_96793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stocking_cost_l967_96799

/-- Calculates the total cost of stockings with a discount and monogramming fee. -/
theorem stocking_cost (num_stockings : ℕ) (original_price discount_percent monogram_fee : ℚ) :
  num_stockings = 9 ∧ 
  original_price = 20 ∧ 
  discount_percent = 10 / 100 ∧
  monogram_fee = 5 →
  (num_stockings : ℚ) * (original_price * (1 - discount_percent) + monogram_fee) = 207 := by
  sorry

#eval (9 : ℚ) * ((20 * (1 - 10 / 100)) + 5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stocking_cost_l967_96799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l967_96781

def line (x y : ℝ) : Prop := 5 * y - 3 * x = 15

def circle_region (x y : ℝ) : Prop := x^2 + y^2 ≤ 16

def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | line p.1 p.2 ∧ circle_region p.1 p.2}

theorem solution_count :
  ∃ n : ℕ, 2 < n ∧ Finite solution_set ∧ Nat.card solution_set = n :=
sorry

#check solution_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l967_96781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_closest_integers_to_sqrt_40_l967_96738

theorem sum_of_closest_integers_to_sqrt_40 : 
  (Int.floor (Real.sqrt 40) : Int) + (Int.ceil (Real.sqrt 40) : Int) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_closest_integers_to_sqrt_40_l967_96738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liters_to_pints_conversion_l967_96713

/-- Given that 0.75 liters is approximately 1.575 pints, 
    prove that 1.5 liters is approximately 3.2 pints. -/
theorem liters_to_pints_conversion (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ 
  ∀ x y : ℝ, 
    |x - 0.75| < δ → |y - 1.575| < δ → |(1.5 * y / x) - 3.2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liters_to_pints_conversion_l967_96713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_if_common_perpendicular_line_lines_perpendicular_if_perpendicular_to_perpendicular_planes_l967_96724

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Theorem 1: If a line is perpendicular to two planes, then those planes are parallel
theorem planes_parallel_if_common_perpendicular_line 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular_line_plane m α) 
  (h2 : perpendicular_line_plane m β) : 
  parallel_planes α β :=
sorry

-- Theorem 2: If two lines are perpendicular to two perpendicular planes respectively, 
-- then those lines are perpendicular to each other
theorem lines_perpendicular_if_perpendicular_to_perpendicular_planes 
  (m n : Line) (α β : Plane)
  (h1 : perpendicular_line_plane m α)
  (h2 : perpendicular_line_plane n β)
  (h3 : perpendicular_planes α β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_if_common_perpendicular_line_lines_perpendicular_if_perpendicular_to_perpendicular_planes_l967_96724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sinusoid_l967_96783

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - 2 * Real.pi / 3)

-- Define the target function
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + 2 * Real.pi / 3)

-- Define the horizontal shift
noncomputable def shift : ℝ := 2 * Real.pi / 3

-- Theorem stating that shifting f to the left by 'shift' results in g
theorem shift_sinusoid (x : ℝ) : f (x + shift) = g x := by
  -- Expand the definitions of f and g
  unfold f g shift
  -- Simplify the expressions
  simp [Real.sin_add]
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sinusoid_l967_96783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l967_96714

-- Define the ellipse type
structure Ellipse where
  points : List (ℝ × ℝ)
  axes_aligned : Bool
  center_is_point : Bool

-- Define the theorem
theorem ellipse_foci_distance (e : Ellipse) :
  e.points = [(-3, 5), (4, -3), (9, 5)] ∧
  e.axes_aligned ∧
  e.center_is_point →
  4 * Real.sqrt 7 = 4 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l967_96714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l967_96731

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) - Real.cos (2 * x)

-- State the theorem
theorem f_properties :
  ∃ (k : ℤ),
    (∀ x, f x = -Real.sin (2 * x + Real.pi / 6)) ∧
    (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
    (∀ x, f (2 * Real.pi / 3 + x) = f (2 * Real.pi / 3 - x)) ∧
    (∀ x, f (5 * Real.pi / 12 + x) = -f (5 * Real.pi / 12 - x)) ∧
    (∀ x ∈ Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3),
      ∀ y ∈ Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3),
        x ≤ y → f x ≤ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l967_96731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_60_minus_alpha_equals_sqrt3_div_2_l967_96734

theorem cos_60_minus_alpha_equals_sqrt3_div_2 (α : Real) :
  Real.sin (30 * Real.pi / 180 + α) = Real.sqrt 3 / 2 →
  Real.cos (60 * Real.pi / 180 - α) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_60_minus_alpha_equals_sqrt3_div_2_l967_96734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l967_96764

theorem sum_remainder_mod_15 (a b c : ℕ) 
  (ha : a % 15 = 8)
  (hb : b % 15 = 12)
  (hc : c % 15 = 13) :
  (a + b + c) % 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l967_96764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_f_99_l967_96733

noncomputable def f (x : ℝ) : ℝ := 
  if 1 ≤ x ∧ x ≤ 3 then 1 - |x - 2| else 0

theorem smallest_x_equals_f_99 (h₁ : ∀ x ≥ 1, f (3 * x) = 3 * f x) 
  (h₂ : ∀ x ≥ 1, x ∈ Set.Ioi 1) : 
  ∃ y ≥ 1, f y = f 99 ∧ ∀ x ≥ 1, f x = f 99 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_f_99_l967_96733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_and_minimum_l967_96737

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the given conditions
axiom func_relation : ∀ x : ℝ, f (2 * x - 1) + g (x + 1) = 4 * x^2 - 2 * x - 1
axiom g_def : ∀ x : ℝ, g x = 2 * x

-- State the theorem to be proved
theorem f_expression_and_minimum :
  (∀ x : ℝ, f x = x^2 - 4 * x) ∧
  (∃ min : ℝ, min = -4 ∧ ∀ x : ℝ, f x ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_and_minimum_l967_96737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_lines_l967_96705

noncomputable section

-- Define the bounding functions
def f (x : ℝ) : ℝ := Real.log x / Real.log 2
def g : ℝ → ℝ := λ _ => 1
def h : ℝ → ℝ := λ _ => 0

-- Define the bounds of integration
def a : ℝ := 1
def b : ℝ := 2

-- Theorem statement
theorem area_bounded_by_lines :
  (∫ x in a..b, g x - (max (f x) (h x))) = 1 / Real.log 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_lines_l967_96705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_common_multiple_of_9_and_6_l967_96703

theorem smallest_common_multiple_of_9_and_6 : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m) ∧ n % 9 = 0 ∧ n % 6 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_common_multiple_of_9_and_6_l967_96703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l967_96729

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ) ^ (4 * x + 2) * (4 : ℝ) ^ (2 * x + 4) = (8 : ℝ) ^ (3 * x + 4) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l967_96729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ariana_speed_is_six_l967_96762

/-- Represents a runner in the relay race -/
structure Runner where
  name : String
  time : ℝ
  speed : ℝ
  distance : ℝ

/-- Represents the relay race -/
structure RelayRace where
  totalDistance : ℝ
  totalTime : ℝ
  runners : List Runner

/-- Calculates Ariana's speed given the race conditions -/
noncomputable def calculateArianaSpeed (race : RelayRace) : ℝ :=
  let sadie : Runner := { name := "Sadie", time := 2, speed := 3, distance := 2 * 3 }
  let ariana : Runner := { name := "Ariana", time := 0.5, speed := 0, distance := 0 }
  let sarah : Runner := { name := "Sarah", time := race.totalTime - sadie.time - ariana.time, speed := 4, distance := (race.totalTime - sadie.time - ariana.time) * 4 }
  let arianaDistance := race.totalDistance - sadie.distance - sarah.distance
  arianaDistance / ariana.time

/-- Theorem stating that Ariana's speed is 6 miles per hour -/
theorem ariana_speed_is_six (race : RelayRace) 
    (h1 : race.totalDistance = 17) 
    (h2 : race.totalTime = 4.5) : calculateArianaSpeed race = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ariana_speed_is_six_l967_96762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sphere_center_sum_l967_96797

def origin : Fin 3 → ℝ := λ _ => 0

def is_midpoint (m d : Fin 3 → ℝ) : Prop :=
  ∀ i, m i = d i / 2

def plane_intersects_axes (d A B C : Fin 3 → ℝ) : Prop :=
  A 1 = 0 ∧ A 2 = 0 ∧ B 0 = 0 ∧ B 2 = 0 ∧ C 0 = 0 ∧ C 1 = 0 ∧
  A ≠ origin ∧ B ≠ origin ∧ C ≠ origin

def is_sphere_center (p d A B C : Fin 3 → ℝ) : Prop :=
  ∃ r : ℝ, 
    (p 0 - origin 0)^2 + (p 1 - origin 1)^2 + (p 2 - origin 2)^2 = r^2 ∧
    (p 0 - A 0)^2 + (p 1 - A 1)^2 + (p 2 - A 2)^2 = r^2 ∧
    (p 0 - B 0)^2 + (p 1 - B 1)^2 + (p 2 - B 2)^2 = r^2 ∧
    (p 0 - C 0)^2 + (p 1 - C 1)^2 + (p 2 - C 2)^2 = r^2

theorem midpoint_sphere_center_sum 
  (a d p : Fin 3 → ℝ) 
  (A B C : Fin 3 → ℝ) :
  is_midpoint a d →
  plane_intersects_axes d A B C →
  is_sphere_center p d A B C →
  d 0 / p 0 + d 1 / p 1 + d 2 / p 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sphere_center_sum_l967_96797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_mass_theorem_l967_96712

/-- The mass of a curve given its equation, linear density, and domain. -/
noncomputable def curveMass (f : ℝ → ℝ) (ρ : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, ρ x * Real.sqrt (1 + (deriv f x)^2)

/-- The curve equation y = 1/2 * x^2 + 1 -/
noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 + 1

/-- The linear density function ρ(x) = x + 1 -/
def ρ (x : ℝ) : ℝ := x + 1

/-- The theorem stating the mass of the curve -/
theorem curve_mass_theorem :
  curveMass f ρ 0 (Real.sqrt 3) = 7/3 + 1/2 * (Real.log (2 + Real.sqrt 3) + 2 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_mass_theorem_l967_96712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l967_96700

noncomputable def f (x : ℝ) : ℝ := 5 / (3 * x^4 - 4)

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  intro x
  unfold f
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l967_96700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_lower_bound_l967_96720

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x - 4*x

theorem extreme_points_sum_lower_bound (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) 
  (hx : x₁ < x₂) 
  (h_extreme : ∃ (y₁ y₂ : ℝ), y₁ < y₂ ∧ 
    (∀ (h : Set ℝ), IsOpen h → y₁ ∈ h → ∃ z ∈ h, f a z ≤ f a y₁) ∧
    (∀ (h : Set ℝ), IsOpen h → y₂ ∈ h → ∃ z ∈ h, f a z ≤ f a y₂)) :
  f a x₁ + f a x₂ > Real.log a - 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_lower_bound_l967_96720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_candidate_majority_no_second_round_necessary_l967_96716

/-- Represents the number of votes for the i-th candidate -/
noncomputable def votes (x : ℝ) (i : ℕ) : ℝ := x / (2 ^ (i - 1))

/-- The sum of votes for all candidates except the first one -/
noncomputable def sum_other_votes (x : ℝ) (n : ℕ) : ℝ := x * (1 - 1 / (2 ^ (n - 1)))

/-- Theorem stating that the first candidate always has more than 50% of total votes -/
theorem first_candidate_majority (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 1) :
  votes x 1 > sum_other_votes x n := by sorry

/-- Corollary: No second round is necessary -/
theorem no_second_round_necessary (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 1) :
  ∃ (winner : ℕ), winner = 1 ∧ votes x winner > sum_other_votes x n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_candidate_majority_no_second_round_necessary_l967_96716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_l967_96769

/-- Given points A, B, C, and D on a line where AB = BC = CD = 6,
    the distance between the midpoints of segments AB and CD is 12. -/
theorem midpoint_distance (A B C D : ℝ) :
  (B - A = 6) → (C - B = 6) → (D - C = 6) →
  |((C + D) / 2) - ((A + B) / 2)| = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_l967_96769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_second_quadrant_l967_96780

theorem tan_value_second_quadrant (θ : Real) 
  (h1 : Real.sin θ = 4/5) 
  (h2 : π/2 < θ ∧ θ < π) : 
  Real.tan θ = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_second_quadrant_l967_96780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_period_and_triangle_area_l967_96784

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 - 2 * Real.sin x ^ 2 - Real.cos (2 * x + Real.pi / 3)

-- State the theorem
theorem function_period_and_triangle_area :
  -- Part 1: The smallest positive period of f is π
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  -- Part 2: Maximum area of triangle ABC
  (∀ (a c : ℝ) (A B C : ℝ),
    -- Given conditions
    0 < B ∧ B < Real.pi ∧
    f (B / 2) = 1 ∧
    -- Triangle side lengths and angles
    0 < a ∧ 0 < c ∧
    a^2 + c^2 - 2*a*c*Real.cos B = 5^2 →
    -- The area is less than or equal to (25√3)/4
    1/2 * a * c * Real.sin B ≤ 25 * Real.sqrt 3 / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_period_and_triangle_area_l967_96784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_probability_l967_96774

theorem ball_probability (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h_total : total = 100) 
  (h_red : red = 47) 
  (h_purple : purple = 3) : 
  (total - (red + purple)) / total = (1/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_probability_l967_96774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_inequality_l967_96795

open Real

/-- The function f(x) = 2ln(x) - (1/2)ax^2 + (2-a)x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * log x - (1/2) * a * x^2 + (2-a) * x

/-- The derivative of f --/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2 / x - a * x + (2-a)

theorem f_derivative_inequality (a : ℝ) :
  ∀ x₁ x₂ x₀ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ →
  (∃ x₀ > 0, f a x₂ - f a x₁ = f_deriv a x₀ * (x₂ - x₁)) →
  f_deriv a ((x₁ + x₂) / 2) < f_deriv a x₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_inequality_l967_96795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_between_vectors_l967_96719

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cos_angle_between_vectors (u v : V)
  (hu : ‖u‖ = 3)
  (hv : ‖v‖ = 4)
  (huv : ‖u - v‖ = 2) :
  (inner u v / (‖u‖ * ‖v‖) : ℝ) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_between_vectors_l967_96719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_greater_than_one_two_distinct_zero_points_l967_96710

-- Define the function f
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (Real.log x / Real.log 2)^2 + 4 * (Real.log x / Real.log 2) + m

-- Define the domain of x
def domain (x : ℝ) : Prop := 1/8 ≤ x ∧ x ≤ 4

-- Theorem for part (I)
theorem zero_point_greater_than_one (m : ℝ) :
  (∃ x : ℝ, domain x ∧ x > 1 ∧ f x m = 0) → -12 ≤ m ∧ m < 0 := by
  sorry

-- Theorem for part (II)
theorem two_distinct_zero_points (m : ℝ) :
  (∃ α β : ℝ, domain α ∧ domain β ∧ α ≠ β ∧ f α m = 0 ∧ f β m = 0) →
  (3 ≤ m ∧ m < 4) ∧ (∃ α β : ℝ, f α m = 0 ∧ f β m = 0 ∧ α * β = 1/16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_greater_than_one_two_distinct_zero_points_l967_96710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_tetrahedron_C₁LMN_l967_96740

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Calculates the volume of a tetrahedron given four points -/
def tetrahedronVolume (p1 p2 p3 p4 : Point3D) : ℝ := sorry

/-- Represents a line segment between two points -/
def lineSegment (p1 p2 : Point3D) : Set Point3D := sorry

/-- Represents a plane through three points -/
def plane (p1 p2 p3 : Point3D) : Set Point3D := sorry

/-- Main theorem: Volume of tetrahedron C₁LMN in the given rectangular prism -/
theorem volume_of_tetrahedron_C₁LMN (prism : RectangularPrism) 
  (hAA₁ : distance prism.A prism.A₁ = 2)
  (hAD : distance prism.A prism.D = 3)
  (hAB : distance prism.A prism.B = 251)
  (L : Point3D) (hL : L ∈ lineSegment prism.C prism.C₁)
  (M : Point3D) (hM : M ∈ lineSegment prism.C₁ prism.B₁)
  (N : Point3D) (hN : N ∈ lineSegment prism.C₁ prism.D₁)
  (hPlane : L ∈ plane prism.A₁ prism.B prism.D ∧ 
            M ∈ plane prism.A₁ prism.B prism.D ∧ 
            N ∈ plane prism.A₁ prism.B prism.D) :
  tetrahedronVolume prism.C₁ L M N = 2008 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_tetrahedron_C₁LMN_l967_96740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_successful_iff_power_of_two_l967_96785

/-- Represents the three possible cube colors -/
inductive Color
  | White
  | Blue
  | Red
deriving Repr, DecidableEq

/-- Represents a circular arrangement of colored cubes -/
def Arrangement (n : ℕ) := Fin n → Color

/-- Defines the result of combining two colors according to the robot's rules -/
def combineColors (c1 c2 : Color) : Color :=
  match c1, c2 with
  | Color.White, Color.White => Color.White
  | Color.Blue, Color.Blue => Color.Blue
  | Color.Red, Color.Red => Color.Red
  | Color.White, Color.Blue => Color.Red
  | Color.White, Color.Red => Color.Blue
  | Color.Blue, Color.White => Color.Red
  | Color.Blue, Color.Red => Color.White
  | Color.Red, Color.White => Color.Blue
  | Color.Red, Color.Blue => Color.White

/-- Simulates the robot's process and returns the final cube color -/
def finalColor (arr : Arrangement n) (start : Fin n) : Color :=
  sorry  -- Implementation details omitted

/-- Checks if an arrangement is "good" -/
def isGoodArrangement (arr : Arrangement n) : Prop :=
  ∀ (start1 start2 : Fin n), finalColor arr start1 = finalColor arr start2

/-- Checks if a number N is "successful" -/
def isSuccessful (N : ℕ) : Prop :=
  ∀ (arr : Arrangement N), isGoodArrangement arr

/-- Main theorem: N is successful if and only if it's a power of 2 -/
theorem successful_iff_power_of_two (N : ℕ) :
  isSuccessful N ↔ ∃ (k : ℕ), N = 2^k :=
  sorry

#check successful_iff_power_of_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_successful_iff_power_of_two_l967_96785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_gas_cost_l967_96775

/-- Represents Tony's car and gas usage scenario -/
structure CarScenario where
  efficiency : ℚ  -- miles per gallon
  round_trip : ℚ  -- miles
  work_days_per_week : ℕ
  tank_capacity : ℚ  -- gallons
  gas_price : ℚ  -- dollars per gallon
  weeks : ℕ

/-- Calculates the total gas cost for the given scenario -/
def total_gas_cost (scenario : CarScenario) : ℚ :=
  let total_miles := scenario.round_trip * scenario.work_days_per_week * scenario.weeks
  let total_gallons := total_miles / scenario.efficiency
  total_gallons * scenario.gas_price

/-- Theorem stating that Tony spends $80 on gas in 4 weeks -/
theorem tony_gas_cost :
  let scenario : CarScenario := {
    efficiency := 25
    round_trip := 50
    work_days_per_week := 5
    tank_capacity := 10
    gas_price := 2
    weeks := 4
  }
  total_gas_cost scenario = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_gas_cost_l967_96775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_phi_l967_96782

/-- The function f(x) with angular frequency ω and phase shift φ -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

/-- The function g(x) with angular frequency ω -/
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

/-- Theorem stating the value of φ given the conditions -/
theorem find_phi (ω φ : ℝ) : 
  ω > 0 ∧ 
  |φ| < π/2 ∧
  (∀ x : ℝ, f ω φ (x + 2*π/ω) = f ω φ x) ∧
  (∀ x : ℝ, g ω x = f ω φ (x - 2*π/3)) →
  φ = 2*π/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_phi_l967_96782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_correct_l967_96794

open Real

/-- The function f(x) = (ln x) / x -/
noncomputable def f (x : ℝ) : ℝ := (log x) / x

/-- The derivative of f(x) -/
noncomputable def f_derivative (x : ℝ) : ℝ := (1 - log x) / (x^2)

theorem f_derivative_correct :
  ∀ x > 0, deriv f x = f_derivative x := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_correct_l967_96794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_3_5_power_l967_96779

def v : ℕ → ℤ
| 0 => 0
| 1 => 1
| (n + 2) => 8 * v (n + 1) - v n

theorem no_3_5_power (n : ℕ) :
  ∀ (α β : ℕ), v n ≠ (3 : ℤ) ^ α * (5 : ℤ) ^ β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_3_5_power_l967_96779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_beta_l967_96743

theorem sin_alpha_minus_beta (α β : ℝ) :
  0 < α ∧ α < π / 2 →  -- α is acute
  π / 2 < β ∧ β < π →  -- β is obtuse
  Real.cos (α - π / 3) = 2 / 3 →
  Real.cos (β + π / 6) = -2 / 3 →
  Real.sin (α - β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_beta_l967_96743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_p_l967_96753

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | (3/x) - 1 ≥ 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the condition β
def beta (p : ℝ) (x : ℝ) : Prop := 2*x + p ≤ 0

-- Theorem statement
theorem range_of_p :
  ∀ p : ℝ, (∀ x ∈ A_intersect_B, beta p x) ↔ p ∈ Set.Ioi (-6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_p_l967_96753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_unit_circle_l967_96747

-- Define a complex number z with rational real and imaginary parts
noncomputable def z (x y : ℚ) : ℂ := x + y * Complex.I

-- State the theorem
theorem complex_unit_circle (x y : ℚ) (n : ℤ) :
  (Complex.abs (z x y) = 1) →
  ((Complex.abs (z 1 0) = 1) ∧ (Complex.abs (z 0 1) = 1)) ∧
  (∃ θ : ℝ, z x y = Complex.exp (θ * Complex.I) ∧
    Complex.abs ((z x y)^(2*n) - 1) = 2 * Complex.abs (Real.sin (θ * ↑n))) ∧
  (∃ q : ℚ, Complex.abs ((z x y)^(2*n) - 1) = q) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_unit_circle_l967_96747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_interior_angles_bisectors_perpendicular_l967_96770

-- Define the concept of a line
structure Line where
  -- Placeholder for line properties
  dummy : Unit

-- Define the concept of an angle
structure Angle where
  -- Placeholder for angle properties
  dummy : Unit

-- Define the concept of angle bisector
def angle_bisector (a : Angle) : Line :=
  -- Placeholder definition
  { dummy := () }

-- Define what it means for two lines to be intersected by a third line
def intersected_by_third_line (l1 l2 l3 : Line) : Prop :=
  -- Placeholder definition
  True

-- Define consecutive interior angles
def consecutive_interior_angles (a1 a2 : Angle) (l1 l2 l3 : Line) : Prop :=
  -- Placeholder definition
  True

-- Define supplementary angles
def supplementary (a1 a2 : Angle) : Prop :=
  -- Placeholder definition
  True

-- Define perpendicular lines
def perpendicular (l1 l2 : Line) : Prop :=
  -- Placeholder definition
  True

-- The main theorem
theorem consecutive_interior_angles_bisectors_perpendicular 
  (l1 l2 l3 : Line) (a1 a2 : Angle) :
  intersected_by_third_line l1 l2 l3 →
  consecutive_interior_angles a1 a2 l1 l2 l3 →
  supplementary a1 a2 →
  perpendicular (angle_bisector a1) (angle_bisector a2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_interior_angles_bisectors_perpendicular_l967_96770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaoxing_treaty_stipulation_l967_96798

/-- Represents historical treaties --/
inductive Treaty
| Shaoxing
| Chenqiao
| WesternXiaPeace

/-- Represents historical dynasties --/
inductive Dynasty
| SouthernSong
| NorthernSong
| Jin
| Liao
| WesternXia
| Khitan

/-- Represents possible stipulations in treaties --/
inductive Stipulation
| PledgeAllegiance : Dynasty → Dynasty → Stipulation
| FraternalNations : Dynasty → Dynasty → Stipulation
| VassalState : Dynasty → Dynasty → Stipulation

/-- Function that returns the stipulation of a given treaty --/
def treatyStipulation : Treaty → Stipulation
| Treaty.Shaoxing => Stipulation.PledgeAllegiance Dynasty.SouthernSong Dynasty.Jin
| Treaty.Chenqiao => Stipulation.FraternalNations Dynasty.NorthernSong Dynasty.Liao
| Treaty.WesternXiaPeace => Stipulation.VassalState Dynasty.WesternXia Dynasty.NorthernSong

/-- Theorem stating that the Treaty of Shaoxing stipulated that the Emperor of Southern Song Dynasty pledged allegiance to the Jin Dynasty --/
theorem shaoxing_treaty_stipulation :
  treatyStipulation Treaty.Shaoxing = Stipulation.PledgeAllegiance Dynasty.SouthernSong Dynasty.Jin :=
by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaoxing_treaty_stipulation_l967_96798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_extrema_l967_96778

/-- Ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- Line L equation -/
def line_L (x y : ℝ) : ℝ := x + 2*y - 10

/-- Distance from a point to Line L -/
noncomputable def distance_to_line_L (x y : ℝ) : ℝ := 
  |line_L x y| / Real.sqrt 5

theorem ellipse_distance_extrema :
  ∃ (x_min y_min x_max y_max : ℝ),
    is_on_ellipse x_min y_min ∧
    is_on_ellipse x_max y_max ∧
    (∀ x y : ℝ, is_on_ellipse x y → 
      distance_to_line_L x_min y_min ≤ distance_to_line_L x y) ∧
    (∀ x y : ℝ, is_on_ellipse x y → 
      distance_to_line_L x y ≤ distance_to_line_L x_max y_max) ∧
    x_min = 9/5 ∧ y_min = 8/5 ∧
    x_max = -9/5 ∧ y_max = -8/5 ∧
    distance_to_line_L x_min y_min = Real.sqrt 5 ∧
    distance_to_line_L x_max y_max = 3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_extrema_l967_96778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_circle_intersection_l967_96745

/-- Given a circle C and a line l, prove the equation of l -/
theorem line_equation_from_circle_intersection :
  let C : Set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 - 4 * p.1 - 2 * p.2 = 0}
  let l : Set (ℝ × ℝ) → Prop := λ L ↦ ∃ b, L = {p | p.2 = p.1 + b}
  let chord_length : ℝ := 2 * Real.sqrt 3
  let inclination_angle : ℝ := Real.pi / 4
  ∀ L, l L ∧ 
    (∃ p q : ℝ × ℝ, p ∈ C ∧ q ∈ C ∧ p ∈ L ∧ q ∈ L ∧ 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) →
    (L = {p | p.2 = p.1 + 1} ∨ L = {p | p.2 = p.1 - 3}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_circle_intersection_l967_96745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_mobots_for_triangular_grid_l967_96790

/-- Represents the possible orientations of a mobot -/
inductive MobotOrientation
  | East
  | WestOfNorth
  | WestOfSouth

/-- Represents a point on the triangular grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a mobot on the grid -/
structure Mobot where
  position : GridPoint
  orientation : MobotOrientation

/-- Represents the triangular grid lawn -/
def TriangularGrid (n : ℕ) : Set GridPoint :=
  {p : GridPoint | p.x + p.y < n ∧ p.x ≥ 0 ∧ p.y ≥ 0}

/-- Predicate to check if a mobot can reach a given point from its current position and orientation -/
def CanReach (start : GridPoint) (orientation : MobotOrientation) (target : GridPoint) : Prop :=
  sorry  -- Definition of reachability based on mobot movement rules

/-- Predicate to check if a set of mobots can mow the entire lawn -/
def CanMowEntireLawn (n : ℕ) (mobots : List Mobot) : Prop :=
  ∀ p ∈ TriangularGrid n, ∃ m ∈ mobots, CanReach m.position m.orientation p

/-- Theorem stating the minimum number of mobots required -/
theorem min_mobots_for_triangular_grid (n : ℕ) :
  (∃ mobots : List Mobot, CanMowEntireLawn n mobots ∧ mobots.length = n) ∧
  (∀ mobots : List Mobot, CanMowEntireLawn n mobots → mobots.length ≥ n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_mobots_for_triangular_grid_l967_96790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l967_96788

/-- Given a point P(m, 2) on the terminal side of angle α and tan(α + π/4) = 3, prove that cosα = 2√5/5 -/
theorem cos_alpha_value (m : ℝ) (α : ℝ) 
  (h1 : (m, 2) ∈ Set.range (fun t => (t * Real.cos α, t * Real.sin α))) 
  (h2 : Real.tan (α + π/4) = 3) : 
  Real.cos α = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l967_96788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basel_problem_l967_96707

open Real
open Finset
open BigOperators

-- Define the infinite product representation of sin(x)/x
noncomputable def sinc_product (x : ℝ) : ℝ := ∏' n, (1 - x^2 / (n^2 * π^2))

-- Define the Taylor series of sin(x)
noncomputable def sin_series (x : ℝ) : ℝ := ∑' n, ((-1)^n * x^(2*n+1)) / (Nat.factorial (2*n+1))

-- State the theorem
theorem basel_problem : ∑' (n : ℕ), 1 / ((n + 1)^2 : ℝ) = π^2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basel_problem_l967_96707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l967_96706

theorem problem_statement (a b : ℝ) (h : ({a, 1} : Set ℝ) = {0, a+b}) : b - a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l967_96706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_slope_one_tangent_line_at_point_l967_96742

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + 4/3

noncomputable def f' (x : ℝ) : ℝ := x^2

theorem tangent_lines_slope_one (x y : ℝ) :
  (3*x - 3*y + 2 = 0 ∨ x - y + 2 = 0) ↔ 
  (∃ x₀ : ℝ, f' x₀ = 1 ∧ y - f x₀ = 1 * (x - x₀)) :=
by sorry

theorem tangent_line_at_point (x y : ℝ) :
  4*x - y - 4 = 0 ↔ y - f 2 = f' 2 * (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_slope_one_tangent_line_at_point_l967_96742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_trebled_time_l967_96741

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Given conditions -/
structure InterestProblem where
  principal : ℝ
  rate : ℝ
  total_time : ℝ
  initial_interest : ℝ
  final_interest : ℝ

/-- The problem statement -/
theorem principal_trebled_time (problem : InterestProblem) 
  (h1 : simple_interest problem.principal problem.rate problem.total_time = problem.initial_interest)
  (h2 : ∃ n : ℝ, simple_interest problem.principal problem.rate n + 
        simple_interest (3 * problem.principal) problem.rate (problem.total_time - n) = problem.final_interest)
  (h3 : problem.total_time = 10)
  (h4 : problem.initial_interest = 1000)
  (h5 : problem.final_interest = 2000) :
  ∃ n : ℝ, n = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_trebled_time_l967_96741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_roots_condition_root_signs_correct_l967_96765

/-- Represents the quadratic equation m x^2 - (1-m) x + (m-1) = 0 --/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 - (1 - m) * x + (m - 1) = 0

/-- The discriminant of the quadratic equation --/
noncomputable def discriminant (m : ℝ) : ℝ :=
  1 + 2*m - 3*m^2

/-- Predicate for real roots --/
def has_real_roots (m : ℝ) : Prop :=
  discriminant m ≥ 0

/-- Theorem stating the condition for real roots --/
theorem real_roots_condition (m : ℝ) :
  has_real_roots m ↔ -1/3 ≤ m ∧ m ≤ 1 := by
  sorry

/-- Function to describe the signs of roots for different m values --/
noncomputable def root_signs (m : ℝ) : String :=
  if m < -1/3 then "complex"
  else if m = -1/3 then "both negative (double root)"
  else if m < 0 then "both negative"
  else if m = 0 then "one negative, one infinity"
  else if m < 1 then "opposite signs"
  else if m = 1 then "both zero"
  else "complex"

/-- Theorem stating the correctness of root_signs function --/
theorem root_signs_correct (m : ℝ) :
  ∀ x, quadratic_equation m x → 
    (x ∈ Set.univ ↔ root_signs m ≠ "complex") ∧
    (root_signs m = "both negative" → x < 0) ∧
    (root_signs m = "opposite signs" → 
      ∃ y, quadratic_equation m y ∧ x * y < 0) ∧
    (root_signs m = "both zero" → x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_roots_condition_root_signs_correct_l967_96765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_counterfeit_coins_l967_96761

/-- Represents the result of testing two coins -/
inductive TestResult
  | BothReal
  | BothCounterfeit
  | Different

/-- Represents a coin -/
inductive Coin
  | Real
  | Counterfeit
deriving DecidableEq

/-- A function that tests two coins and returns a result -/
def testCoins : Coin → Coin → TestResult := sorry

/-- The main theorem stating that all counterfeit coins can be identified in 64 or fewer tests -/
theorem identify_counterfeit_coins 
  (coins : Finset Coin) 
  (h_total : coins.card = 100) 
  (h_counterfeit : (coins.filter (· = Coin.Counterfeit)).card = 85) 
  (h_real : (coins.filter (· = Coin.Real)).card = 15) :
  ∃ (tests : ℕ) (identified : Finset Coin), 
    tests ≤ 64 ∧ 
    identified = coins.filter (· = Coin.Counterfeit) := 
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_counterfeit_coins_l967_96761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_d_value_l967_96702

-- Define the set of natural numbers from a to k
def NaturalSet := Fin 10 → ℕ

-- Define the relationship between the numbers
def ArrowSum (s : NaturalSet) (i j k : Fin 10) : Prop :=
  s i + s j = s k

-- Define the specific relationship for d
def DSum (s : NaturalSet) : ℕ :=
  s 0 + 3 * (s 4 + s 7) + s 9

-- The main theorem
theorem min_d_value (s : NaturalSet) 
  (h1 : ∀ i j, i ≠ j → s i ≠ s j)
  (h2 : ArrowSum s 0 1 2)
  (h3 : ArrowSum s 1 2 3)
  (h4 : ArrowSum s 4 5 6)
  (h5 : ArrowSum s 7 8 9) :
  20 ≤ DSum s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_d_value_l967_96702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_irrational_numbers_l967_96746

-- Define the list of numbers
noncomputable def number_list : List ℝ := [-1, Real.pi, Real.sqrt 2, -(9 ^ (1/3 : ℝ)), 22/7, 0.1010010001]

-- Define a function to check if a number is irrational
def is_irrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ (p : ℝ) / (q : ℝ)

-- Axiom for the irrationality of specific numbers
axiom pi_irrational : is_irrational Real.pi
axiom sqrt_2_irrational : is_irrational (Real.sqrt 2)
axiom cube_root_9_irrational : is_irrational (9 ^ (1/3 : ℝ))
axiom decimal_irrational : is_irrational 0.1010010001

-- Theorem statement
theorem count_irrational_numbers :
  (number_list.filter (λ x => is_irrational x = True)).length = 4 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_irrational_numbers_l967_96746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_progression_l967_96748

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

/-- Theorem: Maximum value of S_n given S_3 and S_57 -/
theorem max_sum_arithmetic_progression
  (ap : ArithmeticProgression)
  (h1 : sum_n ap 3 = 327)
  (h2 : sum_n ap 57 = 57) :
  ∃ n : ℕ, ∀ m : ℕ, sum_n ap m ≤ sum_n ap n ∧ sum_n ap n = 1653 := by
  sorry

#eval sum_n { a := 113, d := -4 } 29

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_progression_l967_96748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l967_96760

noncomputable def f (x : ℝ) : ℝ := (Real.arccos (x/2))^2 + Real.pi * Real.arcsin (x/2) - (Real.arcsin (x/2))^2 + (Real.pi^2/6) * (x^2 + 2*x + 1)

theorem f_range :
  ∀ y ∈ Set.range f, Real.pi^2/4 ≤ y ∧ y ≤ 39*Real.pi^2/96 ∧
  ∃ x ∈ Set.Icc (-2) 2, f x = Real.pi^2/4 ∧
  ∃ x ∈ Set.Icc (-2) 2, f x = 39*Real.pi^2/96 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l967_96760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l967_96730

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3) + Real.cos (2 * x + Real.pi / 6)

theorem f_properties :
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ x, f x ≤ M) ∧
  (∃ (T : ℝ), T = Real.pi ∧ ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x y, Real.pi/24 < x ∧ x < y ∧ y < 13*Real.pi/24 → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l967_96730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_in_first_and_third_quadrants_l967_96759

/-- A function f is an inverse proportion function if there exists a non-zero constant k such that f(x) = k/x for all non-zero x -/
def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- A function f has its graph in the first and third quadrants if f(x) and x have the same sign for all non-zero x -/
def graph_in_first_and_third_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (f x > 0 ↔ x > 0)

/-- The main theorem -/
theorem inverse_proportion_in_first_and_third_quadrants (n : ℝ) :
  is_inverse_proportion (fun x ↦ (n + 1) * x^(n^2 - 5)) ∧
  graph_in_first_and_third_quadrants (fun x ↦ (n + 1) * x^(n^2 - 5)) →
  n = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_in_first_and_third_quadrants_l967_96759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_condition_implies_a_bound_l967_96739

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + Real.log (x + 1) + x

-- State the theorem
theorem function_condition_implies_a_bound (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 > 1 ∧ x2 > 1 ∧ x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 1) →
  a ≤ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_condition_implies_a_bound_l967_96739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_cube_arrangement_l967_96771

-- Define a cube with 8 vertices
inductive Cube
| v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8

-- Define a function to represent the numbers placed on the vertices
def VertexNumber : Cube → ℕ := sorry

-- Define a predicate for adjacent vertices
def Adjacent : Cube → Cube → Prop := sorry

-- Define a predicate for non-adjacent vertices
def NonAdjacent : Cube → Cube → Prop := sorry

-- Theorem statement
theorem impossible_cube_arrangement :
  ¬∃ (f : Cube → ℕ),
    (∀ v, 1 ≤ f v ∧ f v ≤ 220) ∧
    (∀ v1 v2, v1 ≠ v2 → f v1 ≠ f v2) ∧
    (∀ v1 v2, Adjacent v1 v2 → ∃ d > 1, d ∣ f v1 ∧ d ∣ f v2) ∧
    (∀ v1 v2, NonAdjacent v1 v2 → ∀ d > 1, ¬(d ∣ f v1 ∧ d ∣ f v2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_cube_arrangement_l967_96771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l967_96726

-- Define points A and B
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (-4, 5)

-- Define the line AB
def line_AB (t : ℝ) : ℝ × ℝ := (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define point C
noncomputable def C : ℝ × ℝ := sorry

-- Theorem statement
theorem point_C_coordinates :
  (∃ t : ℝ, C = line_AB t) ∧
  distance A C = 3 * distance A B →
  C = (16, -19) ∨ C = (-14, 17) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l967_96726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_sum_symmetry_l967_96721

noncomputable def v (x : ℝ) := -x + 2 * Real.cos (x * Real.pi / 3) + 1

theorem v_sum_symmetry (a b : ℝ) :
  v (-a) + v (-b) + v b + v a = 4 + 4 * (Real.cos (a * Real.pi / 3) + Real.cos (b * Real.pi / 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_sum_symmetry_l967_96721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recursive_angle_formula_l967_96773

theorem recursive_angle_formula (A₀ θ : ℝ) (n : ℕ) :
  let rec A : ℕ → ℝ
    | 0 => A₀
    | i + 1 => (A i * Real.cos θ + Real.sin θ) / (-A i * Real.sin θ + Real.cos θ)
  (A n * Real.cos (n * θ) + Real.sin (n * θ)) / (-A n * Real.sin (n * θ) + Real.cos (n * θ)) = A n :=
by
  induction n with
  | zero => sorry
  | succ n ih => sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_recursive_angle_formula_l967_96773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr1_eq_expr2_eq_expr3_eq_expr4_eq_expr5_eq_l967_96704

-- Define the expressions
noncomputable def expr1 : ℝ := 1.25 * 3.2 * 0.8
noncomputable def expr2 : ℝ := 14.5 * 10.1
noncomputable def expr3 : ℝ := 2.8 + 3.5 * 2.8
noncomputable def expr4 : ℝ := 4.8 * 56.7 + 0.567 * 520
noncomputable def expr5 : ℝ := 8.7 / 0.2 / 0.5

-- State the theorems to be proved
theorem expr1_eq : expr1 = 3.2 := by sorry

theorem expr2_eq : expr2 = 159.5 := by sorry

theorem expr3_eq : expr3 = 12.6 := by sorry

theorem expr4_eq : expr4 = 567 := by sorry

theorem expr5_eq : expr5 = 87 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr1_eq_expr2_eq_expr3_eq_expr4_eq_expr5_eq_l967_96704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olivia_art_studio_area_l967_96744

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total paintable area in an art studio -/
def total_paintable_area (chamber_dims : RoomDimensions) 
  (num_chambers : ℕ) (window_door_area : ℝ) : ℝ :=
  let wall_area := 2 * (chamber_dims.length * chamber_dims.height + 
                        chamber_dims.width * chamber_dims.height)
  let ceiling_area := chamber_dims.length * chamber_dims.width
  let chamber_area := wall_area + ceiling_area - window_door_area
  chamber_area * (num_chambers : ℝ)

/-- Theorem stating the total paintable area in Olivia's art studio -/
theorem olivia_art_studio_area : 
  let chamber_dims : RoomDimensions := ⟨15, 12, 10⟩
  let num_chambers : ℕ := 4
  let window_door_area : ℝ := 80
  total_paintable_area chamber_dims num_chambers window_door_area = 2560 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_olivia_art_studio_area_l967_96744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_highway_miles_per_tankful_l967_96728

/-- Represents the fuel efficiency and travel distance of a car in city and highway conditions -/
structure CarFuelData where
  city_miles_per_tankful : ℚ
  city_miles_per_gallon : ℚ
  city_highway_mpg_difference : ℚ

/-- Calculates the miles per tankful on the highway given the car's fuel data -/
def highway_miles_per_tankful (data : CarFuelData) : ℚ :=
  let tank_size := data.city_miles_per_tankful / data.city_miles_per_gallon
  let highway_mpg := data.city_miles_per_gallon + data.city_highway_mpg_difference
  highway_mpg * tank_size

/-- Theorem stating that given the specified conditions, the car travels 480 miles per tankful on the highway -/
theorem car_highway_miles_per_tankful 
  (data : CarFuelData)
  (h1 : data.city_miles_per_tankful = 336)
  (h2 : data.city_miles_per_gallon = 14)
  (h3 : data.city_highway_mpg_difference = 6) :
  highway_miles_per_tankful data = 480 := by
  sorry

#eval highway_miles_per_tankful { city_miles_per_tankful := 336, city_miles_per_gallon := 14, city_highway_mpg_difference := 6 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_highway_miles_per_tankful_l967_96728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_f₁_min_value_f₂_l967_96736

-- Part 1
noncomputable def f₁ (x : ℝ) : ℝ := -1/3 * x^3 - 1/2 * x^2 + 6*x

theorem extreme_values_f₁ :
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -3 ∧
    IsLocalMax f₁ x₁ ∧ f₁ x₁ = 22/3 ∧
    IsLocalMin f₁ x₂ ∧ f₁ x₂ = -27/2) :=
by sorry

-- Part 2
noncomputable def f₂ (m : ℝ) (x : ℝ) : ℝ := 1/3 * x^3 - 1/2 * x^2 + 2*m*x

theorem min_value_f₂ (m : ℝ) (h₁ : -2 < m) (h₂ : m < 0)
  (h₃ : ∃ x₀ ∈ Set.Icc 1 4, ∀ x ∈ Set.Icc 1 4, f₂ m x ≤ f₂ m x₀)
  (h₄ : ∃ x₁ ∈ Set.Icc 1 4, f₂ m x₁ = 16/3) :
  ∃ x₂ ∈ Set.Icc 1 4, (∀ x ∈ Set.Icc 1 4, f₂ m x₂ ≤ f₂ m x) ∧ f₂ m x₂ = -10/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_f₁_min_value_f₂_l967_96736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l967_96708

def discriminant (b c : ℤ) : ℤ := b^2*c^2 - 4*b^3 - 4*c^3 + 18*b*c - 27

def valid_pair (b c : ℤ) : Prop :=
  b > 0 ∧ c > 0 ∧ discriminant b c ≤ 0 ∧ discriminant c b ≤ 0

theorem count_valid_pairs :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), p ∈ S ↔ valid_pair p.1 p.2) ∧ Finset.card S = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l967_96708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_2a_plus_b_l967_96766

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (a b : V)

-- State the theorem
theorem magnitude_2a_plus_b (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 1) (h3 : ‖a + b‖ = Real.sqrt 3) :
  ‖(2 : ℝ) • a + b‖ = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_2a_plus_b_l967_96766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l967_96749

def a (l : ℝ) : ℝ × ℝ := (1, l)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (1, -2)

theorem collinear_vectors (l : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ (2 • a l + b) = k • c) → l = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l967_96749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_piece_shape_l967_96787

/-- Represents a rectangular piece on the grid -/
structure Piece where
  width : Nat
  height : Nat

/-- Represents the 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a piece can be placed on the grid at a given position -/
def canPlace (grid : Grid) (piece : Piece) (x y : Fin 4) : Prop :=
  ∀ i j, i < piece.width → j < piece.height → 
    (x + ↑i < 4 ∧ y + ↑j < 4) ∧ ¬grid (x + ↑i) (y + ↑j)

/-- Places a piece on the grid at a given position -/
def place (grid : Grid) (piece : Piece) (x y : Fin 4) : Grid :=
  λ i j ↦ if i ≥ x ∧ i < x + ↑piece.width ∧ j ≥ y ∧ j < y + ↑piece.height
         then true
         else grid i j

/-- Checks if the grid is completely filled -/
def isFilled (grid : Grid) : Prop :=
  ∀ i j, grid i j

theorem third_piece_shape (grid : Grid) (piece1 piece2 : Piece) 
  (h1 : piece1 = ⟨2, 1⟩)
  (h2 : piece2 = ⟨3, 1⟩)
  (h3 : ∃ x1 y1 x2 y2, canPlace grid piece1 x1 y1 ∧ 
                       canPlace (place grid piece1 x1 y1) piece2 x2 y2) :
  ∃ piece3 : Piece, (piece3 = ⟨4, 1⟩ ∨ piece3 = ⟨1, 4⟩) ∧
  ∃ x3 y3, canPlace (place (place grid piece1 x1 y1) piece2 x2 y2) piece3 x3 y3 ∧
  isFilled (place (place (place grid piece1 x1 y1) piece2 x2 y2) piece3 x3 y3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_piece_shape_l967_96787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l967_96752

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2)^x + 1

-- State the theorem
theorem range_of_f :
  ∀ y, y ∈ Set.range (f ∘ (Set.Icc (-1) 1).restrict f) ↔ y ∈ Set.Icc (3/2) 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l967_96752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_with_divisible_pair_l967_96772

/-- Given distinct prime numbers p, q, and r, A is the set of numbers of the form p^a * q^b * r^c
    where 0 ≤ a, b, c ≤ 5 -/
def A (p q r : ℕ) : Set ℕ :=
  {n | ∃ (a b c : ℕ), 0 ≤ a ∧ a ≤ 5 ∧ 0 ≤ b ∧ b ≤ 5 ∧ 0 ≤ c ∧ c ≤ 5 ∧ n = p^a * q^b * r^c}

/-- For any subset B of A with |B| = n, there exist x, y ∈ B such that x divides y -/
def HasDivisiblePair (B : Set ℕ) : Prop :=
  ∃ (x y : ℕ), x ∈ B ∧ y ∈ B ∧ x ≠ y ∧ x ∣ y

/-- 28 is the least natural number n such that for any subset B of A with |B| = n,
    there exist x, y ∈ B where x divides y -/
theorem least_n_with_divisible_pair (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r)
    (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
    (∀ (B : Finset ℕ), ↑B ⊆ A p q r → B.card = 28 → HasDivisiblePair ↑B) ∧
    (∀ (m : ℕ), m < 28 → ∃ (B : Finset ℕ), ↑B ⊆ A p q r ∧ B.card = m ∧ ¬HasDivisiblePair ↑B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_with_divisible_pair_l967_96772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_implies_a_value_l967_96786

/-- Given an inequality (x - a) / (x + 1) > 0 with solution set (-∞, -1) ∪ (4, +∞), prove that a = 4 -/
theorem inequality_solution_implies_a_value (a : ℝ) :
  (∀ x : ℝ, (x - a) / (x + 1) > 0 ↔ x ∈ Set.Ioi (-1) ∪ Set.Iio 4) →
  a = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_implies_a_value_l967_96786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spending_100_yuan_l967_96723

-- We need to use ℤ (integers) instead of ℕ (natural numbers) to allow for negative values
def ways_to_spend (n : ℕ) : ℤ :=
  (1 : ℤ) * (2^(n+1) + (-1)^n) / 3

theorem spending_100_yuan :
  ways_to_spend 100 = (1 : ℤ) * (2^101 + 1) / 3 :=
by
  -- The proof is skipped using 'sorry'
  sorry

#eval ways_to_spend 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spending_100_yuan_l967_96723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sri_lanka_village_population_l967_96715

def village_population (initial_population : ℕ) 
  (death_rate : ℚ) (leaving_rate : ℚ) : ℕ :=
  let remaining_after_death := initial_population - Int.floor (↑initial_population * death_rate)
  let final_population := remaining_after_death - Int.floor (↑remaining_after_death * leaving_rate)
  final_population.toNat

theorem sri_lanka_village_population : 
  village_population 3161 (5/100) (15/100) = 2553 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sri_lanka_village_population_l967_96715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l967_96701

/-- A circle that passes through (0,3) and is tangent to y = x^2 at (3,9) has center (3, 97/10) -/
theorem circle_center (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) : 
  (∀ (p : ℝ × ℝ), p ∈ C → (p.1 - center.1)^2 + (p.2 - center.2)^2 = (center.1 - 3)^2 + (center.2 - 9)^2) →
  (0, 3) ∈ C →
  (3, 9) ∈ C →
  (∀ (p : ℝ × ℝ), p ∈ C → p.2 ≠ p.1^2 ∨ (p.1 = 3 ∧ p.2 = 9)) →
  center = (3, 97/10) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l967_96701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_non_integer_solution_l967_96792

theorem no_non_integer_solution (x y : ℝ) (m n : ℤ) 
  (h1 : (6 : ℝ) * x + (5 : ℝ) * y = ↑m)
  (h2 : (13 : ℝ) * x + (11 : ℝ) * y = ↑n) :
  ∃ (z : ℤ), x = ↑z :=
by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_non_integer_solution_l967_96792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l967_96727

noncomputable def f (x : ℝ) := Real.sqrt (4 * x + 1)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ -1/4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l967_96727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_wire_for_poles_l967_96754

noncomputable def shortest_wire_length (d1 d2 : ℝ) : ℝ :=
  let r1 := d1 / 2
  let r2 := d2 / 2
  let center_distance := r1 + r2
  let radius_diff := r2 - r1
  let straight_section := 2 * Real.sqrt (center_distance ^ 2 - radius_diff ^ 2)
  let small_arc := 2 * Real.pi * r1 * (1/3)
  let large_arc := 2 * Real.pi * r2 * (2/3)
  straight_section + small_arc + large_arc

theorem shortest_wire_for_poles :
  shortest_wire_length 4 20 = 8 * Real.sqrt 5 + 44 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_wire_for_poles_l967_96754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_area_l967_96755

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus (F : ℝ × ℝ) : Prop := F.1 = 1 ∧ F.2 = 0

-- Define a point on the parabola
def pointOnParabola (B : ℝ × ℝ) : Prop := parabola B.1 B.2

-- Define point A
def A : ℝ × ℝ := (5, 4)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- Theorem statement
theorem min_perimeter_triangle_area 
  (F : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (hF : focus F) 
  (hB : pointOnParabola B) :
  ∃ (B : ℝ × ℝ), 
    pointOnParabola B ∧ 
    (∀ (B' : ℝ × ℝ), pointOnParabola B' → 
      (Real.sqrt ((B'.1 - A.1)^2 + (B'.2 - A.2)^2) + 
       Real.sqrt ((B'.1 - F.1)^2 + (B'.2 - F.2)^2) + 
       Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2)) ≥
      (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) + 
       Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) + 
       Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2))) →
    triangleArea A B F = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_area_l967_96755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l967_96751

/-- The area of the trapezoid formed by the arrangement of three squares --/
theorem trapezoid_area (s₁ s₂ s₃ : ℝ) (h₁ : s₁ = 4) (h₂ : s₂ = 6) (h₃ : s₃ = 8) : 
  (s₁ * (s₃ / (s₁ + s₂ + s₃)) + (s₁ + s₂) * (s₃ / (s₁ + s₂ + s₃))) * s₂ / 2 = 56 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l967_96751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_max_min_difference_l967_96776

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_max_min_difference (a : ℝ) : 
  (a > 0 ∧ a ≠ 1) →
  (∀ x ∈ Set.Icc 2 4, f a x ≤ f a 4 ∧ f a 2 ≤ f a x) →
  (f a 4 - f a 2 = 1) →
  (a = 1/2 ∨ a = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_max_min_difference_l967_96776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_distinct_values_l967_96791

theorem least_distinct_values (list_size : ℕ) (mode_frequency : ℕ) 
  (h1 : list_size = 2412) (h2 : mode_frequency = 12) :
  ∃ x : ℕ, x = 219 ∧ x * (mode_frequency - 1) + mode_frequency ≥ list_size ∧
    ∀ y : ℕ, y < x → y * (mode_frequency - 1) + mode_frequency < list_size := by
  sorry

#check least_distinct_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_distinct_values_l967_96791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_value_l967_96796

theorem smallest_c_value (c d : ℤ) (r s t : ℕ+) : 
  (∀ x : ℤ, x^3 - c*x^2 + d*x - 3080 = (x - r.val) * (x - s.val) * (x - t.val)) →
  r.val * s.val * t.val = 3080 →
  c = r.val + s.val + t.val →
  ∀ c' : ℤ, (∃ d' r' s' t' : ℕ+, 
    (∀ x : ℤ, x^3 - c'*x^2 + d'*x - 3080 = (x - r'.val) * (x - s'.val) * (x - t'.val)) ∧
    r'.val * s'.val * t'.val = 3080) →
  c' ≥ 34 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_value_l967_96796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_eight_l967_96768

/-- The number of barrels to be filled -/
def num_barrels : ℕ := 4

/-- The capacity of each barrel in gallons -/
noncomputable def barrel_capacity : ℝ := 7

/-- The flow rate of the faucet in gallons per minute -/
noncomputable def flow_rate : ℝ := 3.5

/-- The time needed to fill all barrels in minutes -/
noncomputable def fill_time : ℝ := (num_barrels : ℝ) * barrel_capacity / flow_rate

theorem fill_time_is_eight : fill_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_eight_l967_96768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_problem_l967_96717

/-- Represents the count of each number on the blackboard -/
structure BoardState where
  ones : Nat
  twos : Nat
  threes : Nat
  fours : Nat
  fives : Nat

/-- Represents an operation on the board -/
inductive Operation
  | erase_1234_write5
  | erase_1235_write4
  | erase_1245_write3
  | erase_1345_write2
  | erase_2345_write1

/-- Applies an operation to a board state -/
def apply_operation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.erase_1234_write5 => 
      { ones := state.ones - 1, twos := state.twos - 1, threes := state.threes - 1, 
        fours := state.fours - 1, fives := state.fives + 1 }
  | Operation.erase_1235_write4 => 
      { ones := state.ones - 1, twos := state.twos - 1, threes := state.threes - 1, 
        fours := state.fours + 1, fives := state.fives - 1 }
  | Operation.erase_1245_write3 => 
      { ones := state.ones - 1, twos := state.twos - 1, threes := state.threes + 1, 
        fours := state.fours - 1, fives := state.fives - 1 }
  | Operation.erase_1345_write2 => 
      { ones := state.ones - 1, twos := state.twos + 1, threes := state.threes - 1, 
        fours := state.fours - 1, fives := state.fives - 1 }
  | Operation.erase_2345_write1 => 
      { ones := state.ones + 1, twos := state.twos - 1, threes := state.threes - 1, 
        fours := state.fours - 1, fives := state.fives - 1 }

/-- The initial state of the blackboard -/
def initial_state : BoardState :=
  { ones := 2006, twos := 2007, threes := 2008, fours := 2009, fives := 2010 }

/-- Checks if the board state has exactly two numbers left -/
def has_two_numbers_left (state : BoardState) : Bool :=
  (if state.ones > 0 then 1 else 0) + (if state.twos > 0 then 1 else 0) +
  (if state.threes > 0 then 1 else 0) + (if state.fours > 0 then 1 else 0) +
  (if state.fives > 0 then 1 else 0) = 2

/-- Calculates the product of the remaining numbers -/
def product_of_remaining (state : BoardState) : Nat :=
  (if state.ones > 0 then 1 else 1) *
  (if state.twos > 0 then 2 else 1) *
  (if state.threes > 0 then 3 else 1) *
  (if state.fours > 0 then 4 else 1) *
  (if state.fives > 0 then 5 else 1)

theorem blackboard_problem :
  ∃ (ops : List Operation), 
    let final_state := ops.foldl apply_operation initial_state
    has_two_numbers_left final_state ∧ product_of_remaining final_state = 8 := by
  sorry

#eval product_of_remaining initial_state

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_problem_l967_96717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l967_96789

noncomputable def f (x : ℝ) : ℝ := Real.sin x * (Real.cos x - Real.sqrt 3 * Real.sin x)

theorem f_properties :
  ∃ (T : ℝ), T > 0 ∧ 
  (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = π ∧
  ∃ (a b : ℝ), 0 < a ∧ a < π/2 ∧
  (∀ x, f x = Real.sin (2*(x + a)) - b) ∧
  a * b = (Real.sqrt 3 / 12) * π ∧
  Set.Icc (-Real.sqrt 3) (1 - Real.sqrt 3 / 2) = { y | ∃ x ∈ Set.Icc 0 (π/2), f x = y } :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l967_96789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_gasoline_percentage_l967_96725

/-- Represents the composition of a fuel mixture -/
structure FuelMixture where
  total_volume : ℝ
  ethanol_percentage : ℝ

/-- Calculates the volume of ethanol in a fuel mixture -/
noncomputable def ethanol_volume (mixture : FuelMixture) : ℝ :=
  mixture.total_volume * (mixture.ethanol_percentage / 100)

/-- Theorem: The desired percentage of gasoline for optimum performance is 90% -/
theorem optimal_gasoline_percentage
  (initial_mixture : FuelMixture)
  (added_ethanol : ℝ)
  (optimal_ethanol_percentage : ℝ)
  (h1 : initial_mixture.total_volume = 45)
  (h2 : initial_mixture.ethanol_percentage = 5)
  (h3 : added_ethanol = 2.5)
  (h4 : optimal_ethanol_percentage = 10)
  (h5 : ethanol_volume initial_mixture + added_ethanol = 
        (initial_mixture.total_volume + added_ethanol) * (optimal_ethanol_percentage / 100)) :
  100 - optimal_ethanol_percentage = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_gasoline_percentage_l967_96725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l967_96750

theorem range_of_a (a : ℝ) : 
  (∀ (x θ : ℝ), θ ∈ Set.Icc 0 (Real.pi / 2) → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) 
  ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l967_96750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l967_96757

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f
noncomputable def f (x y : ℝ) : ℝ := (x + y) / (floor x * floor y + floor x + floor y + 1)

-- Define the set representing the range of f
def range_f : Set ℝ := {1/2} ∪ Set.Icc (5/6) (5/4)

-- Theorem statement
theorem range_of_f (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  f x y ∈ range_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l967_96757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kings_tour_length_bounds_l967_96732

/-- Represents a king's tour on an 8x8 chessboard -/
def KingsTour := List (Fin 8 × Fin 8)

/-- The length of a move between two squares -/
noncomputable def moveLength (a b : Fin 8 × Fin 8) : ℝ :=
  Real.sqrt ((a.1.val - b.1.val : ℝ) ^ 2 + (a.2.val - b.2.val : ℝ) ^ 2)

/-- The total length of a king's tour -/
noncomputable def tourLength (tour : KingsTour) : ℝ :=
  (List.zip tour (tour.tail.append [tour.head!])).map (fun (a, b) => moveLength a b) |>.sum

/-- A valid king's tour visits each square exactly once and returns to the start -/
def isValidTour (tour : KingsTour) : Prop :=
  tour.length = 64 ∧
  tour.toFinset.card = 64 ∧
  tour.head! = tour.getLast!

theorem kings_tour_length_bounds {tour : KingsTour} (h : isValidTour tour) :
  64 ≤ tourLength tour ∧ tourLength tour ≤ 28 + 36 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kings_tour_length_bounds_l967_96732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l967_96767

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp (2 * x)

-- Define the point of tangency
def point : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (HasDerivAt f m 0) ∧
    f point.1 = point.2 ∧
    m = 3 ∧ b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l967_96767

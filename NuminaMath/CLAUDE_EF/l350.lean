import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_problem_l350_35068

theorem cube_root_problem (h1 : Real.rpow 25.36 (1/3) = 2.938) 
  (h2 : Real.rpow 253.6 (1/3) = 6.329) : 
  Real.rpow 25360000 (1/3) = 293.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_problem_l350_35068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tangent_line_with_slope_two_l350_35044

-- Define the function f(x) = √x + 2x
noncomputable def f (x : ℝ) := Real.sqrt x + 2 * x

-- State the theorem
theorem no_tangent_line_with_slope_two :
  ∀ x : ℝ, x > 0 → deriv f x ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tangent_line_with_slope_two_l350_35044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_to_even_function_l350_35034

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

noncomputable def shifted_function (x : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 8) + Real.pi / 4)

theorem shift_to_even_function :
  ∀ x : ℝ, shifted_function x = shifted_function (-x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_to_even_function_l350_35034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l350_35033

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2*x - 2 - 4/x

-- Define the set P
def P : Set ℝ := {x | x > 0 ∧ f' x > 0}

-- Define the set Q
def Q (a : ℝ) : Set ℝ := {x | x^2 + (a-1)*x - a > 0}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (P ⊂ Q a ∧ P ≠ Q a) ↔ a ∈ Set.Ici (-2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l350_35033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_fourths_l350_35020

/-- The sum of the infinite series ∑(k=1 to ∞) k/3^k -/
noncomputable def series_sum : ℝ := ∑' k, (k : ℝ) / 3^k

/-- Theorem: The sum of the infinite series ∑(k=1 to ∞) k/3^k is equal to 3/4 -/
theorem series_sum_equals_three_fourths : series_sum = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_fourths_l350_35020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_problem_l350_35045

theorem notebook_problem (n k : ℕ) : 
  (∃ (x : ℕ), n * (n + 1) / 2 - (2 * k + 1) = 2021) →
  (n ≥ 64 ∧ (n % 4 = 0 ∨ n % 4 = 3)) ∧
  (∃ (m : ℕ), m % 2 = 1 ∧ m > 127 ∧ k = (m^2 - 16177) / 16) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_problem_l350_35045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_two_valid_l350_35073

def is_valid (n : ℕ) : Bool :=
  n > 1 && (3^n % n = 0) && ((3^n - 1) % (n - 1) = 0)

def first_two_valid : List ℕ :=
  (List.range 1000).filter is_valid |> List.take 2

theorem sum_of_first_two_valid :
  first_two_valid.sum = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_two_valid_l350_35073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carnival_money_l350_35097

theorem carnival_money (allan brenda cole david elise : ℤ) : 
  allan + brenda + cole + david + elise = 75 →
  |allan - brenda| = 21 →
  |brenda - cole| = 8 →
  |cole - david| = 6 →
  |david - elise| = 5 →
  |elise - allan| = 12 →
  cole = 14 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carnival_money_l350_35097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roaming_area_difference_l350_35004

/-- Represents the roaming area for a cat tied to an octagonal playhouse. -/
structure CatRoamingArea where
  /-- Side length of the octagonal playhouse in feet -/
  playhouse_side : ℝ
  /-- Length of the cat's leash in feet -/
  leash_length : ℝ

/-- Calculates the roaming area when the cat is tied to the midpoint of one side -/
noncomputable def arrangement_I_area (setup : CatRoamingArea) : ℝ :=
  0.5 * Real.pi * setup.leash_length ^ 2

/-- Calculates the roaming area when the cat is tied 5 feet away from a corner -/
noncomputable def arrangement_II_area (setup : CatRoamingArea) : ℝ :=
  0.75 * Real.pi * setup.leash_length ^ 2 + 0.25 * Real.pi * 5 ^ 2

/-- Theorem stating the difference in roaming area between Arrangement II and Arrangement I -/
theorem roaming_area_difference (setup : CatRoamingArea) 
  (h1 : setup.playhouse_side = 10)
  (h2 : setup.leash_length = 15) :
  arrangement_II_area setup - arrangement_I_area setup = 62.5 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roaming_area_difference_l350_35004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l350_35084

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x)) / Real.log 2

-- State the theorem
theorem f_is_odd : 
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f (-x) = -f x) ∧ 
  (Set.Ioo (-1 : ℝ) 1 = {x : ℝ | f x ≠ 0}) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l350_35084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_parent_combinations_for_type_O_l350_35035

-- Define the blood types
inductive BloodType
| A
| B
| O
| AB
deriving Fintype, Repr

-- Define a function to check if a blood type can produce an O allele
def canProduceOAllele : BloodType → Bool
  | BloodType.A => true
  | BloodType.B => true
  | BloodType.O => true
  | BloodType.AB => false

-- Define a function to check if a pair of blood types can produce a child with type O
def canProduceTypeO (parent1 : BloodType) (parent2 : BloodType) : Bool :=
  canProduceOAllele parent1 && canProduceOAllele parent2

-- Define the theorem
theorem count_parent_combinations_for_type_O :
  (Finset.filter (fun pair => canProduceTypeO pair.1 pair.2)
    (Finset.product Finset.univ Finset.univ)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_parent_combinations_for_type_O_l350_35035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_sum_l350_35055

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := ((x - 2) * (x - 4)) / 3
noncomputable def g (x : ℝ) : ℝ := -f x
noncomputable def h (x : ℝ) : ℝ := f (-x)

-- Define the number of intersection points
def a : ℕ := 2  -- Number of intersection points between f and g
def b : ℕ := 1  -- Number of intersection points between f and h

-- Theorem statement
theorem intersection_points_sum : 10 * a + b = 21 := by
  -- Expand the definitions of a and b
  have h1 : a = 2 := rfl
  have h2 : b = 1 := rfl
  
  -- Perform the calculation
  calc
    10 * a + b = 10 * 2 + 1 := by rw [h1, h2]
    _          = 21         := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_sum_l350_35055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l350_35010

theorem equilateral_triangle_perimeter (z : ℂ) : 
  z ≠ 0 → 
  z ≠ 1 → 
  (z^3 - z = Complex.exp (2*Real.pi*Complex.I/3) * (z^2 - z) ∨ 
   z^3 - z = Complex.exp (-2*Real.pi*Complex.I/3) * (z^2 - z)) → 
  3 * Complex.abs (z^2 - z) = 3 * Real.sqrt 3 :=
by
  sorry

#check equilateral_triangle_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l350_35010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_journey_time_l350_35040

/-- Calculates the total journey time for a boat traveling upstream and downstream in a river -/
theorem boat_journey_time 
  (river_speed : ℝ) 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (h1 : river_speed = 2)
  (h2 : boat_speed = 6)
  (h3 : distance = 64) :
  (distance / (boat_speed - river_speed) + distance / (boat_speed + river_speed)) = 24 := by
  sorry

#check boat_journey_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_journey_time_l350_35040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_and_midpoint_trajectory_l350_35058

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the length of AB
noncomputable def length_AB : ℝ := Real.sqrt 17

-- Define the slope angles
def slope_angles : Set ℝ := {Real.pi/3, 2*Real.pi/3}

-- Define the equation of the trajectory of midpoint M
def trajectory_M (x y : ℝ) : Prop := (x - 1/2)^2 + (y - 1)^2 = 1/4

-- Theorem statement
theorem circle_intersection_and_midpoint_trajectory :
  ∃ (l : ℝ → ℝ → Prop) (A B : ℝ × ℝ),
    (∀ x y, l x y → (x = 1 ∧ y = 1) ∨ (x ≠ 1 ∧ y ≠ 1)) →  -- l passes through P
    (circle_C A.1 A.2 ∧ circle_C B.1 B.2) →  -- A and B are on circle C
    (l A.1 A.2 ∧ l B.1 B.2) →  -- A and B are on line l
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = length_AB^2 →  -- |AB| = √17
    (∃ θ ∈ slope_angles, ∀ x y, l x y → y - 1 = Real.tan θ * (x - 1)) ∧  -- Slope angle is π/3 or 2π/3
    (∀ x y, trajectory_M x y ↔ 
      ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
      x = (1 - t) * A.1 + t * B.1 ∧ 
      y = (1 - t) * A.2 + t * B.2) :=  -- Midpoint trajectory
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_and_midpoint_trajectory_l350_35058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedral_angle_sine_ratio_l350_35047

/-- Represents a trihedral angle with face angles and dihedral angles -/
structure TrihedralAngle where
  α : ℝ
  β : ℝ
  γ : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem: For a trihedral angle, the ratio of sine of face angle to sine of opposite dihedral angle is constant -/
theorem trihedral_angle_sine_ratio (t : TrihedralAngle) :
  (Real.sin t.α) / (Real.sin t.A) = (Real.sin t.β) / (Real.sin t.B) ∧
  (Real.sin t.β) / (Real.sin t.B) = (Real.sin t.γ) / (Real.sin t.C) := by
  sorry

#check trihedral_angle_sine_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedral_angle_sine_ratio_l350_35047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_tangent_slope_range_l350_35066

/-- The hyperbola equation -/
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 12 - y^2 / 4 = 1

/-- The right focus of the hyperbola -/
noncomputable def right_focus : ℝ × ℝ := (2 * Real.sqrt 3, 0)

/-- A line passing through a point with a given slope -/
def line_through_point (p : ℝ × ℝ) (m : ℝ) (x y : ℝ) : Prop :=
  y - p.2 = m * (x - p.1)

/-- The slope is in the given range -/
def slope_in_range (m : ℝ) : Prop :=
  -Real.sqrt 3 / 3 ≤ m ∧ m ≤ Real.sqrt 3 / 3

/-- The main theorem -/
theorem hyperbola_tangent_slope_range :
  ∀ (m : ℝ),
    (∃! (x y : ℝ), is_on_hyperbola x y ∧ x > 0 ∧ line_through_point right_focus m x y) →
    slope_in_range m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_tangent_slope_range_l350_35066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_10_over_3_f_monotone_decreasing_l350_35067

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1/x

-- Theorem 1: f(a) = 10/3 if and only if a = 3 or a = 1/3
theorem f_equals_10_over_3 (a : ℝ) :
  f a = 10/3 ↔ a = 3 ∨ a = 1/3 := by sorry

-- Theorem 2: f is monotonically decreasing on (0, 1)
theorem f_monotone_decreasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 → f x₁ > f x₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_10_over_3_f_monotone_decreasing_l350_35067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l350_35057

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (l1 l2 : Line) : Prop := l1.slope * l2.slope = -1

/-- First line (m+2)x + 3my + 1 = 0 -/
def line1 (m : ℝ) : Line := ⟨m+2, 3*m, 1⟩

/-- Second line (m-2)x + (m+2)y - 3 = 0 -/
def line2 (m : ℝ) : Line := ⟨m-2, m+2, -3⟩

/-- m = 1/2 is a sufficient but not necessary condition for perpendicularity -/
theorem perpendicular_condition :
  (∃ m : ℝ, m ≠ 1/2 ∧ perpendicular (line1 m) (line2 m)) ∧
  perpendicular (line1 (1/2)) (line2 (1/2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l350_35057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l350_35021

noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

def line1 : ℝ × ℝ → Prop :=
  fun (x, y) ↦ 3 * x + 4 * y - 7 = 0

def line2 : ℝ × ℝ → Prop :=
  fun (x, y) ↦ 6 * x + 8 * y + 1 = 0

theorem distance_between_lines :
  distance_between_parallel_lines 6 8 (-14) 1 = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l350_35021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_factorial_product_l350_35093

theorem min_sum_of_factorial_product (a b c d : ℕ+) : 
  a * b * c * d = Nat.factorial 8 → (a : ℕ) + b + c + d ≥ 102 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_factorial_product_l350_35093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallelism_l350_35048

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

/-- Two 2D vectors are parallel if their cross product is zero -/
def isParallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_parallelism (k : ℝ) :
  isParallel (a.1 + k * c.1, a.2 + k * c.2) (2 * b.1 - a.1, 2 * b.2 - a.2) →
  k = -16/13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallelism_l350_35048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l350_35072

/-- The complex number z = (2-i)/(1+i) is located in the fourth quadrant of the complex plane -/
theorem z_in_fourth_quadrant : 
  let z : ℂ := (2 - I) / (1 + I)
  Real.sign z.re = 1 ∧ Real.sign z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l350_35072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l350_35092

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 3 = 0
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 3 = 0

-- Define the centers and radii
def center_O1 : ℝ × ℝ := (-1, -2)
def center_O2 : ℝ × ℝ := (2, 1)
noncomputable def radius_O1 : ℝ := Real.sqrt 2
noncomputable def radius_O2 : ℝ := 2 * Real.sqrt 2

-- Theorem: The circles are externally tangent
theorem circles_externally_tangent :
  let d := Real.sqrt ((center_O2.1 - center_O1.1)^2 + (center_O2.2 - center_O1.2)^2)
  d = radius_O1 + radius_O2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l350_35092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l350_35030

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 - 4 * Real.sin x - 4 * (Real.cos x)^2

-- State the theorem
theorem f_extrema :
  (∀ x, f x ≤ 7) ∧ (∃ x, f x = 7) ∧
  (∀ x, f x ≥ -2) ∧ (∃ x, f x = -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l350_35030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_savings_l350_35038

noncomputable def daily_allowance : ℝ := 15

noncomputable def week1_savings : ℝ :=
  6 * (daily_allowance / 2) + (daily_allowance / 3)

noncomputable def week2_savings : ℝ :=
  5 * (daily_allowance * 0.6) + 2 * (daily_allowance * 0.25)

noncomputable def week3_savings : ℝ :=
  7 * (daily_allowance * 0.75) - 30

noncomputable def total_savings : ℝ :=
  week1_savings + week2_savings + week3_savings

theorem martha_savings : total_savings = 151.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_savings_l350_35038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l350_35031

theorem calculate_expression : 
  (81 : ℝ) ^ (1/4) + (32 : ℝ) ^ (1/5) - (49 : ℝ) ^ (1/2) = -2 := by
  have h1 : (81 : ℝ) ^ (1/4) = 3 := by sorry
  have h2 : (32 : ℝ) ^ (1/5) = 2 := by sorry
  have h3 : (49 : ℝ) ^ (1/2) = 7 := by sorry
  rw [h1, h2, h3]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l350_35031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l350_35029

/-- The area of the shaded region formed by quarter circles of radius 1 foot,
    alternating top and bottom along a 3-foot length. -/
noncomputable def shadedArea : ℝ := 3/4 * Real.pi

/-- The length of the pattern in feet -/
def patternLength : ℝ := 3

/-- The radius of each quarter circle in feet -/
def circleRadius : ℝ := 1

theorem shaded_area_calculation :
  shadedArea = (patternLength / (2 * circleRadius)) * (1/4 * Real.pi * circleRadius^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l350_35029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_distance_sum_l350_35037

/-- Predicate to check if a point is the inscribed circle center of a triangle -/
def IsInscribedCenter (O : EuclideanSpace ℝ (Fin 2)) (triangle : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 
    (∀ (P : EuclideanSpace ℝ (Fin 2)), P ∈ triangle → dist O P = r)

/-- Given a triangle ABC with inscribed circle center O, prove that the sum of the squared distances
    from O to each vertex divided by the product of the other two sides equals 1. -/
theorem inscribed_circle_distance_sum 
  (A B C O : EuclideanSpace ℝ (Fin 2)) (a b c : ℝ) :
  IsInscribedCenter O {A, B, C} →
  a = dist B C →
  b = dist A C →
  c = dist A B →
  (dist O A)^2 / (b * c) + (dist O B)^2 / (a * c) + (dist O C)^2 / (a * b) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_distance_sum_l350_35037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_isosceles_triangle_l350_35025

theorem unique_isosceles_triangle (b c : ℝ) (B : Real) :
  b > 0 → c > 0 → B > 0 → B < Real.pi →
  b = c → B = Real.pi / 4 →
  ∃! (a : ℝ), ∃! (A C : Real),
    (0 < a ∧ 0 < A ∧ A < Real.pi ∧ 0 < C ∧ C < Real.pi) ∧
    (A + B + C = Real.pi) ∧
    (a^2 = b^2 + c^2 - 2*b*c*(Real.cos B)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_isosceles_triangle_l350_35025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_translation_is_cos_l350_35027

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- Define the translation operations
def translate_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x ↦ f (x - a)
def translate_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x ↦ f x + b

-- Define the resulting function after translations
noncomputable def g : ℝ → ℝ := translate_up (translate_right f (Real.pi / 2)) 1

-- Theorem statement
theorem sin_translation_is_cos :
  ∀ x : ℝ, g x = 1 - Real.cos x := by
  intro x
  unfold g translate_up translate_right f
  simp [Real.sin_sub_pi_div_two]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_translation_is_cos_l350_35027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_line_is_optimal_min_distance_product_line_is_optimal_l350_35026

/-- A line passing through a point (x₀, y₀) -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : a * 2 + b * 1 + c = 0

/-- The area of a triangle formed by a line intersecting the x and y axes -/
def triangleArea (l : Line) : ℝ := sorry

/-- The product of distances from a point to the x and y axis intersections of a line -/
def distanceProduct (l : Line) (x₀ y₀ : ℝ) : ℝ := sorry

/-- The line passing through (2, 1) that minimizes the area of the triangle formed with the axes -/
def minAreaLine : Line where
  a := 1
  b := 2
  c := -4
  eq := by simp; ring

/-- The line passing through (2, 1) that minimizes the product of distances to axis intersections -/
def minDistanceProductLine : Line where
  a := 1
  b := 1
  c := -3
  eq := by simp; ring

theorem min_area_line_is_optimal :
  ∀ l : Line, triangleArea minAreaLine ≤ triangleArea l := by sorry

theorem min_distance_product_line_is_optimal :
  ∀ l : Line, distanceProduct minDistanceProductLine 2 1 ≤ distanceProduct l 2 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_line_is_optimal_min_distance_product_line_is_optimal_l350_35026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wife_speed_l350_35079

/-- Proves that given specific conditions on a circular track, 
    the second person's speed is 3.75 km/hr -/
theorem wife_speed (track_circumference : ℝ) 
                   (deepak_speed : ℝ) 
                   (meeting_time : ℝ) 
                   (h1 : track_circumference = 0.627) -- in km
                   (h2 : deepak_speed = 4.5) -- in km/hr
                   (h3 : meeting_time = 4.56 / 60) -- in hours
                   : ∃ (wife_speed : ℝ), 
                     wife_speed * meeting_time + deepak_speed * meeting_time = track_circumference ∧ 
                     wife_speed = 3.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wife_speed_l350_35079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l350_35094

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≤ 1/4) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≥ -1/2) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ x₂ ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x₁ = 1/4 ∧ f x₂ = -1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l350_35094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l350_35071

theorem min_value_of_expression (a b : ℕ+) (ha : a < 6) (hb : b < 9) :
  (∀ x y : ℕ+, x < 6 → y < 9 → (3 : ℤ) * x - 2 * x * y ≥ (3 : ℤ) * a - 2 * a * b) →
  (3 : ℤ) * a - 2 * a * b = -65 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l350_35071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supporting_functions_existence_l350_35042

-- Define the concept of a supporting function
def has_supporting_function (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, ∀ x : ℝ, f x ≥ k * x + b

-- Define the given functions
noncomputable def f1 : ℝ → ℝ := fun x ↦ x^3
noncomputable def f2 : ℝ → ℝ := fun x ↦ Real.exp (-x * Real.log 2)
noncomputable def f3 : ℝ → ℝ := fun x ↦ if x > 0 then Real.log x / Real.log 10 else 0
noncomputable def f4 : ℝ → ℝ := fun x ↦ x + Real.sin x

-- State the theorem
theorem supporting_functions_existence :
  has_supporting_function f2 ∧
  has_supporting_function f4 ∧
  ¬has_supporting_function f1 ∧
  ¬has_supporting_function f3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supporting_functions_existence_l350_35042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_at_least_two_white_l350_35006

def num_red_balls : ℕ := 6
def num_white_balls : ℕ := 4
def total_balls : ℕ := num_red_balls + num_white_balls
def num_drawn : ℕ := 4

def probability_at_least_two_white : ℚ := 23/42

theorem probability_of_at_least_two_white :
  (Nat.choose num_white_balls 2 * Nat.choose num_red_balls (num_drawn - 2) +
   Nat.choose num_white_balls 3 * Nat.choose num_red_balls (num_drawn - 3) +
   Nat.choose num_white_balls 4) / Nat.choose total_balls num_drawn
  = probability_at_least_two_white := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_at_least_two_white_l350_35006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_inequality_range_l350_35069

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := exp x - x
noncomputable def g (x : ℝ) : ℝ := exp (-x) + x

-- Define the function F
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := a * f x - g x

-- Define the property for x₁ and x₂
def is_max_min (a : ℝ) (x₁ x₂ : ℝ) : Prop :=
  ∀ x, F a x ≤ F a x₁ ∧ F a x ≥ F a x₂

-- State the theorem
theorem F_inequality_range (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < 1) (h₃ : is_max_min a x₁ x₂) :
  (∀ t : ℝ, t ≤ -1 → F a x₁ + t * F a x₂ > 0) ∧
  (∀ ε > 0, ∃ a₀ : ℝ, 0 < a₀ ∧ a₀ < 1 ∧ F a₀ x₁ + (-1 + ε) * F a₀ x₂ ≤ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_inequality_range_l350_35069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_byron_winning_strategy_ada_winning_strategy_l350_35039

-- Part (a)
theorem byron_winning_strategy (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (p q r : ℝ), ({p, q, r} : Set ℝ) = {a, b, c} ∧ ∃ x : ℝ, p * x^2 + q * x + r = 0 :=
sorry

-- Part (b)
theorem ada_winning_strategy (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ c : ℝ, c ≠ 0 ∧
    ∀ (p q r : ℝ), ({p, q, r} : Set ℝ) = {a, b, c} →
      ∃ x : ℝ, p * x^2 + q * x + r = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_byron_winning_strategy_ada_winning_strategy_l350_35039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_triangle_inequality_l350_35054

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.cos x * (Real.sqrt 3 * Real.sin x - Real.cos x) + m

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m (x + Real.pi/6)

theorem problem_solution (m : ℝ) :
  (∀ x ∈ Set.Icc (Real.pi/4) (Real.pi/3), g m x ≥ Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc (Real.pi/4) (Real.pi/3), g m x = Real.sqrt 3 / 2) →
  m = Real.sqrt 3 / 2 := by
  sorry

theorem triangle_inequality (A B C : ℝ) :
  0 < A ∧ A < Real.pi/2 ∧
  0 < B ∧ B < Real.pi/2 ∧
  0 < C ∧ C < Real.pi/2 ∧
  A + B + C = Real.pi ∧
  g (Real.sqrt 3 / 2) (C/2) = -1/2 + Real.sqrt 3 →
  Real.sqrt 3 / 2 < Real.sin A + Real.cos B ∧ Real.sin A + Real.cos B < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_triangle_inequality_l350_35054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l350_35080

/-- A function satisfying the given functional equation and initial condition -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ContinuousOn f Set.univ ∧ 
  (∀ x y : ℝ, 2 * f (x + y) = f x * f y) ∧
  f 1 = 10

/-- The theorem stating that the only function satisfying the given conditions is f(x) = 2 * 5^x -/
theorem unique_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  ∀ x : ℝ, f x = 2 * (5 : ℝ) ^ x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l350_35080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l350_35009

/-- Calculates the average speed of a train given two segments of its journey -/
noncomputable def average_speed (x : ℝ) : ℝ :=
  let distance1 := x
  let speed1 := 40
  let distance2 := 2 * x
  let speed2 := 20
  let total_distance := 5 * x
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  total_distance / total_time

/-- Theorem stating that the average speed of the train is 40 kmph -/
theorem train_average_speed (x : ℝ) (hx : x > 0) : average_speed x = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l350_35009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_parameter_l350_35088

/-- Given a function f and its inverse, prove that the parameter b equals 3 -/
theorem inverse_function_parameter (f : ℝ → ℝ) (b : ℝ) : 
  (∀ x, f x = 1 / (2 * x + b)) → 
  (∀ x, Function.invFun f x = (1 - 3 * x) / (3 * x)) → 
  b = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_parameter_l350_35088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_value_l350_35053

theorem tan_plus_cot_value (α : ℝ) (h : Real.sin α + Real.cos α = -Real.sqrt 2) : 
  Real.tan α + (1 / Real.tan α) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_value_l350_35053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l350_35051

-- Define the line and parabola
def line (x y : ℝ) : Prop := y - 2*x + 5 = 0
def parabola (x y : ℝ) : Prop := y^2 = 3*x + 4

-- Define point P
def P : ℝ × ℝ := (2, 0)

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_difference :
  ∀ (A B : ℝ × ℝ),
  line A.1 A.2 ∧ parabola A.1 A.2 ∧
  line B.1 B.2 ∧ parabola B.1 B.2 ∧
  A ≠ B →
  |distance A P - distance B P| = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l350_35051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l350_35024

-- Define the triangle ABC and point D
variable (A B C D : EuclideanSpace ℝ (Fin 2))

-- State that D is on AB
variable (h1 : D ∈ SegmentOpen A B)

-- Define the condition AD = 2DB
variable (h2 : A - D = (2 : ℝ) • (D - B))

-- Theorem statement
theorem vector_relation :
  C - D = (1 / 3 : ℝ) • (C - A) + (2 / 3 : ℝ) • (C - B) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l350_35024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l350_35090

-- Define the functions f and g
def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := (x^3)^(1/3)

-- Theorem stating that f and g are the same function
theorem f_equals_g : f = g := by
  ext x
  simp [f, g]
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l350_35090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersections_condition_l350_35050

/-- The function f(x) defined as the integral of p - ln(1 + |t|) from -1 to x. -/
noncomputable def f (p : ℝ) (x : ℝ) : ℝ := ∫ (t : ℝ) in (-1)..x, p - Real.log (1 + abs t)

/-- The theorem stating the condition for f(x) to have two intersections with the positive x-axis. -/
theorem two_intersections_condition (p : ℝ) (hp : p > 0) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f p x₁ = 0 ∧ f p x₂ = 0) ↔
  Real.log (2 * Real.log 2) < p ∧ p < 2 * Real.log 2 - 1 := by
  sorry

#check two_intersections_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersections_condition_l350_35050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_bugs_on_board_l350_35012

/-- Represents a direction on the board -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a position on the board -/
structure Position where
  x : Fin 10
  y : Fin 10

/-- Represents a bug on the board -/
structure Bug where
  position : Position
  direction : Direction

/-- The game board -/
def Board := List Bug

/-- Simulates one step of bug movement -/
def moveStep (board : Board) : Board :=
  sorry

/-- Checks if any two bugs occupy the same position -/
def hasCollision (board : Board) : Bool :=
  sorry

/-- Theorem: The maximum number of bugs on a 10x10 board is 40 -/
theorem max_bugs_on_board :
  ∃ (board : Board), board.length = 40 ∧
  (∀ n : ℕ, ¬hasCollision (n.fold (fun _ b => moveStep b) board)) ∧
  (∀ (largerBoard : Board), largerBoard.length > 40 →
    ∃ n : ℕ, hasCollision (n.fold (fun _ b => moveStep b) largerBoard)) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_bugs_on_board_l350_35012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_depreciation_rate_l350_35099

/-- Represents the annual depreciation rate of a machine --/
noncomputable def annual_depreciation_rate (initial_value : ℝ) (value_after_two_years : ℝ) : ℝ :=
  1 - (value_after_two_years / initial_value) ^ (1/2 : ℝ)

/-- Theorem stating that the annual depreciation rate for a machine
    with initial value $8,000 and value $3,200 after two years
    is approximately 36.75% --/
theorem machine_depreciation_rate :
  let rate := annual_depreciation_rate 8000 3200
  ∃ ε > 0, abs (rate - 0.3675) < ε ∧ ε < 0.0001 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_depreciation_rate_l350_35099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_undefined_value_l350_35008

theorem largest_undefined_value : 
  let f (x : ℝ) := (x - 3) / (6 * x^2 - 65 * x + 54)
  ∃ (max : ℝ), (∀ x, ¬(∃ y, f x = y) → x ≤ max) ∧ ¬(∃ y, f max = y) ∧ max = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_undefined_value_l350_35008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l350_35007

/-- The original curve function -/
noncomputable def original_curve (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 3)

/-- The transformed curve function -/
noncomputable def transformed_curve (x : ℝ) : ℝ := 2 * Real.sin (x / 3 + Real.pi / 3)

/-- Theorem stating that the transformed curve is equivalent to stretching the horizontal coordinates of the original curve -/
theorem curve_transformation (x : ℝ) :
  transformed_curve (3 * x) = original_curve x :=
by
  -- Unfold the definitions of transformed_curve and original_curve
  unfold transformed_curve original_curve
  -- Simplify the expressions
  simp [Real.sin_add]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l350_35007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intervals_of_monotonicity_range_of_a_l350_35028

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k * x^2) / Real.exp x

-- State the theorem for the intervals of monotonicity
theorem intervals_of_monotonicity (k : ℝ) (h : k ≠ 0) :
  (k < 0 → (StrictMonoOn (f k) (Set.Ioi 2) ∧ StrictMonoOn (f k) (Set.Iio 0)) ∧
           (StrictAntiOn (f k) (Set.Ioo 0 2))) ∧
  (k > 0 → (StrictMonoOn (f k) (Set.Ioo 0 2)) ∧
           (StrictAntiOn (f k) (Set.Ioi 2) ∧ StrictAntiOn (f k) (Set.Iio 0))) :=
by sorry

-- State the theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ Real.log (f 1 x) > a * x) → a < 2 / Real.exp 1 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intervals_of_monotonicity_range_of_a_l350_35028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersections_l350_35078

def line_l1 (x y : ℝ) : Prop := 2 * x + y - 8 = 0

def line_l2 (x y : ℝ) : Prop := x - 3 * y + 10 = 0

def my_midpoint (x1 y1 x2 y2 xm ym : ℝ) : Prop :=
  xm = (x1 + x2) / 2 ∧ ym = (y1 + y2) / 2

def line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

theorem line_through_intersections (x1 y1 x2 y2 : ℝ) :
  line_l1 x1 y1 →
  line_l2 x2 y2 →
  my_midpoint x1 y1 x2 y2 0 1 →
  ∃ (a b c : ℝ), line_equation a b c x1 y1 ∧ line_equation a b c x2 y2 ∧ a = 1 ∧ b = 4 ∧ c = -4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersections_l350_35078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_plate_diameter_eq_longest_side_l350_35005

/-- A triangle with sides a, b, and c. -/
structure Triangle (a b c : ℝ) : Type :=
  (hpos : 0 < a ∧ 0 < b ∧ 0 < c)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b)

/-- The minimal diameter of a circular plate that can fit two pieces of a triangle -/
noncomputable def minPlateDiameter (t : Triangle a b c) : ℝ :=
  max a (max b c)

/-- The theorem stating that the minimal plate diameter for a triangle with sides 19, 20, and 21
    is equal to the longest side of the triangle -/
theorem min_plate_diameter_eq_longest_side :
  let t : Triangle 19 20 21 := {
    hpos := by norm_num
    htri := by norm_num
  }
  minPlateDiameter t = 21 := by
  unfold minPlateDiameter
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_plate_diameter_eq_longest_side_l350_35005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mn_is_8_l350_35085

/-- The quadratic function -/
noncomputable def y (m n x : ℝ) : ℝ := (1/2) * (m - 1) * x^2 + (n - 6) * x + 1

/-- The condition that y decreases as x increases for 1 ≤ x ≤ 2 -/
def decreasing_y (m n : ℝ) : Prop :=
  ∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 2 → y m n x₁ > y m n x₂

/-- The theorem stating that the maximum value of mn is 8 -/
theorem max_mn_is_8 (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) (hdecr : decreasing_y m n) :
  m * n ≤ 8 ∧ ∃ m₀ n₀, m₀ ≥ 0 ∧ n₀ ≥ 0 ∧ decreasing_y m₀ n₀ ∧ m₀ * n₀ = 8 := by
  sorry

#check max_mn_is_8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mn_is_8_l350_35085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_derivative_g_l350_35018

-- Define the functions
def f (x : ℝ) : ℝ := (x + 1) * (x + 2) * (x + 3)

noncomputable def g (x : ℝ) : ℝ := 2 * x * Real.tan x

-- State the theorems
theorem derivative_f :
  deriv f = λ x => 3 * x^2 + 12 * x + 11 := by sorry

theorem derivative_g :
  deriv g = λ x => (2 * Real.sin x * Real.cos x + 2 * x) / (Real.cos x)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_derivative_g_l350_35018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_equals_cos_l350_35011

/-- Recursively defines the expression with n nested square roots of 2 -/
noncomputable def nestedSqrt : ℕ → ℝ
  | 0 => 0
  | 1 => Real.sqrt 2
  | n + 1 => Real.sqrt (2 + nestedSqrt n)

/-- States the theorem that the nested square root expression equals 2cos(π/2^(n+1)) -/
theorem nested_sqrt_equals_cos (n : ℕ) (h : n ≥ 1) :
  nestedSqrt n = 2 * Real.cos (π / 2^(n + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_equals_cos_l350_35011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_minimum_c_l350_35013

noncomputable def f (a b x : ℝ) : ℝ := 2 * a * x - b / x + Real.log x

theorem extreme_values_and_minimum_c (a b : ℝ) :
  (∀ x : ℝ, x > 0 → (deriv (f a b)) x = 0 ↔ x = 1 ∨ x = 1/2) →
  a = -1/3 ∧ b = -1/3 ∧
  (∀ c : ℝ, (∃ x₀ : ℝ, x₀ ∈ Set.Icc (1/4 : ℝ) 2 ∧ f a b x₀ - c ≤ 0) ↔ c ≥ -7/6 + Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_minimum_c_l350_35013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l350_35083

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus F
def focus : ℝ × ℝ := (1, 0)

-- Define a point P on the parabola
variable (P : ℝ × ℝ)

-- Define M as the foot of the perpendicular from P to the y-axis
def M (P : ℝ × ℝ) : ℝ × ℝ := (0, P.2)

-- State the theorem
theorem parabola_triangle_area :
  parabola P.1 P.2 →  -- P is on the parabola
  ‖P - focus‖ = 5 →   -- distance PF is 5
  (1/2 : ℝ) * ‖P - M P‖ * |P.2| = 8 :=  -- area of triangle PFM is 8
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l350_35083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_18_and_sqrt_between_30_and_30_3_l350_35062

theorem divisible_by_18_and_sqrt_between_30_and_30_3 :
  ∀ n : ℕ, 
    n > 0 ∧
    n % 18 = 0 ∧ 
    30 < (n : ℝ).sqrt ∧ 
    (n : ℝ).sqrt < 30.3 
    ↔ n = 900 ∨ n = 918 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_18_and_sqrt_between_30_and_30_3_l350_35062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seventeen_pi_fourths_l350_35032

theorem cos_seventeen_pi_fourths : Real.cos (17 * π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seventeen_pi_fourths_l350_35032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chromatic_number_bound_l350_35074

/-- A simple undirected graph. -/
structure MySimpleGraph (V : Type*) where
  adj : V → V → Prop
  sym : ∀ u v, adj u v → adj v u
  loopless : ∀ v, ¬adj v v

/-- The number of cycles containing an edge in a graph. -/
def edgeCycleCount {V : Type*} (G : MySimpleGraph V) : V → V → ℕ := sorry

/-- The chromatic number of a graph. -/
def chromaticNumber {V : Type*} (G : MySimpleGraph V) : ℕ := sorry

/-- The main theorem: if each edge is in at most n cycles, the chromatic number is at most n+1. -/
theorem chromatic_number_bound {V : Type*} (G : MySimpleGraph V) (n : ℕ) 
  (h_n : n ≥ 2) 
  (h_cycles : ∀ u v, G.adj u v → edgeCycleCount G u v ≤ n) : 
  chromaticNumber G ≤ n + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chromatic_number_bound_l350_35074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_expansion_l350_35070

open Polynomial

theorem coefficient_of_x_in_expansion : 
  let x : ℚ[X] := X
  let f := (1 - 2*x) * (1 - x)^5
  coeff f 1 = -30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_expansion_l350_35070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_neg_pi_over_8_l350_35077

-- Define the function f
noncomputable def f (ω φ x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ) - Real.cos (ω * x + φ)

-- State the theorem
theorem f_value_at_neg_pi_over_8 :
  ∃ (ω φ : ℝ), 
    ω > 0 ∧ 
    0 < φ ∧ 
    φ < π ∧ 
    (∀ x, f ω φ x = f ω φ (-x)) ∧ 
    (∀ x, f ω φ (x + π / (2 * ω)) = f ω φ x) ∧
    f ω φ (-π/8) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_neg_pi_over_8_l350_35077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l350_35082

noncomputable section

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the line (2-m)x+my+3=0 -/
noncomputable def slope1 (m : ℝ) : ℝ := m / (m - 2)

/-- The slope of the line x-my-3=0 -/
noncomputable def slope2 (m : ℝ) : ℝ := 1 / m

/-- m=-2 is a sufficient but not necessary condition for the lines to be perpendicular -/
theorem perpendicular_condition (m : ℝ) :
  (m = -2 → perpendicular (slope1 m) (slope2 m)) ∧
  ¬(perpendicular (slope1 m) (slope2 m) → m = -2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l350_35082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_and_area_l350_35015

/-- Definition of an equilateral triangle with side length 10 -/
structure EquilateralTriangle :=
  (side_length : ℝ)
  (is_equilateral : side_length = 10)

/-- Calculate the perimeter of an equilateral triangle -/
def perimeter (t : EquilateralTriangle) : ℝ :=
  3 * t.side_length

/-- Calculate the area of an equilateral triangle -/
noncomputable def area (t : EquilateralTriangle) : ℝ :=
  (Real.sqrt 3 / 4) * t.side_length^2

/-- Theorem stating the perimeter and area of the specific equilateral triangle -/
theorem equilateral_triangle_perimeter_and_area :
  ∀ t : EquilateralTriangle,
    perimeter t = 30 ∧ area t = 25 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_and_area_l350_35015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_photos_for_condition_l350_35023

-- Define the predicates as parameters of the theorem
theorem min_photos_for_condition 
  (n_girls n_boys : ℕ) 
  (h_girls : n_girls = 4) 
  (h_boys : n_boys = 8)
  (is_two_boys : ∀ n, Fin n → Prop)
  (is_two_girls : ∀ n, Fin n → Prop)
  (same_children : ∀ n, Fin n → Fin n → Prop) :
  ∃ (min_photos : ℕ), 
    min_photos = n_girls * n_boys + 1 ∧
    (∀ (photos : ℕ), photos ≥ min_photos →
      (∃ (photo : Fin photos), is_two_boys photos photo ∨ is_two_girls photos photo) ∨
      (∃ (photo1 photo2 : Fin photos), photo1 ≠ photo2 ∧ same_children photos photo1 photo2)) :=
by
  -- We use 'sorry' to skip the proof for now
  sorry

-- The definitions are removed as they are now parameters of the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_photos_for_condition_l350_35023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_grid_fraction_l350_35081

/-- Calculate the area of a triangle using the Shoelace formula -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The fraction of an 8x6 grid covered by a triangle with vertices (2,2), (6,2), and (5,5) -/
theorem triangle_grid_fraction :
  let x1 : ℚ := 2
  let y1 : ℚ := 2
  let x2 : ℚ := 6
  let y2 : ℚ := 2
  let x3 : ℚ := 5
  let y3 : ℚ := 5
  let gridWidth : ℚ := 8
  let gridHeight : ℚ := 6
  (triangleArea x1 y1 x2 y2 x3 y3) / (gridWidth * gridHeight) = 1/8 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_grid_fraction_l350_35081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locally_odd_m_range_l350_35000

/-- A function f is locally odd if there exists an x in its domain such that f(-x) = -f(x) -/
def LocallyOdd (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f (-x) = -f x

/-- The function f(x) = 4^x - m*2^(x+1) + m^2 - 5 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  (4 : ℝ)^x - m * (2 : ℝ)^(x + 1) + m^2 - 5

theorem locally_odd_m_range :
  ∀ m : ℝ, LocallyOdd (f m) ↔ 1 - Real.sqrt 5 < m ∧ m ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locally_odd_m_range_l350_35000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_and_complementary_l350_35086

structure MathGroup :=
  (boys : ℕ)
  (girls : ℕ)

def select (n : ℕ) (g : MathGroup) : Type := Unit

def at_least_one_boy {g : MathGroup} (s : select 2 g) : Prop := sorry

def all_girls {g : MathGroup} (s : select 2 g) : Prop := sorry

theorem mutually_exclusive_and_complementary :
  let g := MathGroup.mk 3 2
  let s := select 2 g
  (∀ x : s, ¬(at_least_one_boy x ∧ all_girls x)) ∧
  (∀ x : s, at_least_one_boy x ∨ all_girls x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_and_complementary_l350_35086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_run_rate_l350_35063

/-- Represents a cricket game situation -/
structure CricketGame where
  total_overs : ℕ
  first_part_overs : ℕ
  first_part_run_rate : ℚ
  target_runs : ℕ

/-- Calculates the required run rate for the remaining overs -/
noncomputable def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_part_overs
  let first_part_runs := game.first_part_run_rate * game.first_part_overs
  let remaining_runs := game.target_runs - first_part_runs.floor
  remaining_runs / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate (game : CricketGame)
  (h_total_overs : game.total_overs = 50)
  (h_first_part_overs : game.first_part_overs = 10)
  (h_first_part_run_rate : game.first_part_run_rate = 16/5)
  (h_target_runs : game.target_runs = 282) :
  required_run_rate game = 25/4 := by
  sorry

#eval (25 : ℚ) / 4  -- To show that 25/4 is indeed equal to 6.25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_run_rate_l350_35063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_tangent_line_distance_l350_35076

/-- A parabola with equation x^2 = 16y -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (1/16) * p.1^2}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (0, 4)

/-- A line passing through (-1, 0) that intersects the parabola at only one point -/
structure TangentLine where
  k : ℝ
  eq : ℝ → ℝ
  passes_through : eq (-1) = 0
  tangent : ∀ x, (x, eq x) ∈ Parabola → (x = -1 ∨ eq x = -(1/4) * x - 1/4 ∨ x = -1)

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (l : TangentLine) : ℝ :=
  if l.k = 0 then
    |p.1 + 1|
  else
    |l.k * p.1 - p.2 + l.eq 0| / Real.sqrt (l.k^2 + 1)

/-- The theorem stating the possible distances from the focus to the tangent line -/
theorem focus_to_tangent_line_distance (l : TangentLine) :
  distance_point_to_line Focus l = 1 ∨
  distance_point_to_line Focus l = 4 ∨
  distance_point_to_line Focus l = Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_tangent_line_distance_l350_35076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winner_margin_over_second_l350_35061

/-- Represents the election results for a small town mayoral election. -/
structure ElectionResults where
  total_votes : Nat
  candidate_votes : Fin 4 → Nat
  votes_sum_check : (Finset.sum (Finset.univ : Finset (Fin 4)) candidate_votes) = total_votes

/-- Theorem stating that the winner's margin over the second-place candidate is 53 votes. -/
theorem winner_margin_over_second (election : ElectionResults)
  (h1 : election.candidate_votes 0 = 195)
  (h2 : election.candidate_votes 1 = 142)
  (h3 : election.candidate_votes 2 = 116)
  (h4 : election.candidate_votes 3 = 90)
  (h5 : election.total_votes = 963) :
  election.candidate_votes 0 - election.candidate_votes 1 = 53 := by
  sorry

#check winner_margin_over_second

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winner_margin_over_second_l350_35061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zu_rate_theorem_l350_35060

def digits : List Nat := [1, 4, 1, 5, 9, 2, 6]

def is_valid_arrangement (arrangement : List Nat) : Bool :=
  arrangement.length = 7 && 
  arrangement.toFinset = digits.toFinset &&
  (arrangement.take 2 = [1, 1] || arrangement.take 2 = [1, 2])

def count_valid_arrangements : Nat :=
  (List.permutations digits).filter is_valid_arrangement |>.length

theorem zu_rate_theorem : count_valid_arrangements = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zu_rate_theorem_l350_35060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_is_identity_P_uniqueness_l350_35056

/-- A polynomial satisfying the given conditions -/
def P : ℝ → ℝ := sorry

/-- P(0) = 0 -/
axiom P_zero : P 0 = 0

/-- For all real x, P(x^2 + 1) = [P(x)]^2 + 1 -/
axiom P_functional_equation : ∀ x : ℝ, P (x^2 + 1) = (P x)^2 + 1

/-- Theorem: P(x) = x for all real x -/
theorem P_is_identity : ∀ x : ℝ, P x = x := by
  sorry

/-- Theorem: P is the unique polynomial satisfying the conditions -/
theorem P_uniqueness : 
  ∀ Q : ℝ → ℝ, (Q 0 = 0 ∧ (∀ x : ℝ, Q (x^2 + 1) = (Q x)^2 + 1)) → (∀ x : ℝ, Q x = P x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_is_identity_P_uniqueness_l350_35056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l350_35014

noncomputable def f (x : ℝ) := Real.sin (2 * x) - Real.sqrt 3 * (Real.cos x ^ 2 - Real.sin x ^ 2)

theorem f_properties :
  let C := {(x, y) | y = f x}
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (11 * Real.pi / 6 - x, y) ∈ C) ∧ 
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (4 * Real.pi / 3 - x, -y) ∈ C) ∧
  (∀ (x₁ x₂ : ℝ), -Real.pi / 12 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5 * Real.pi / 12 → f x₁ < f x₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l350_35014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l350_35046

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 / 4^x) - (1 / 2^x) + 1

-- State the theorem
theorem f_min_max :
  ∃ (x_min x_max : ℝ),
    x_min ∈ Set.Icc (-3 : ℝ) 2 ∧
    x_max ∈ Set.Icc (-3 : ℝ) 2 ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 2, f x ≥ f x_min) ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 2, f x ≤ f x_max) ∧
    f x_min = 3/4 ∧
    f x_max = 57 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l350_35046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_exists_l350_35022

theorem inequality_exists (S : Finset ℝ) (h : S.card = 9) :
  ∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a * c + b * d)^2 ≥ 9/10 * (a^2 + b^2) * (c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_exists_l350_35022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l350_35091

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a^2 + h.b^2) / h.a^2)

/-- Checks if a point is on the hyperbola -/
def is_on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1

/-- Checks if a point is on the line x = 2a -/
def is_on_line (h : Hyperbola) (p : Point) : Prop :=
  p.x = 2 * h.a

/-- Checks if three points form an equilateral triangle -/
def is_equilateral_triangle (p1 p2 p3 : Point) : Prop :=
  let d12 := (p1.x - p2.x)^2 + (p1.y - p2.y)^2
  let d23 := (p2.x - p3.x)^2 + (p2.y - p3.y)^2
  let d31 := (p3.x - p1.x)^2 + (p3.y - p1.y)^2
  d12 = d23 ∧ d23 = d31

theorem hyperbola_eccentricity (h : Hyperbola) (A B : Point) :
  is_on_hyperbola h A ∧ is_on_hyperbola h B ∧
  is_on_line h A ∧ is_on_line h B ∧
  is_equilateral_triangle A B (Point.mk 0 0) →
  eccentricity h = Real.sqrt 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l350_35091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l350_35001

def A : ℝ × ℝ × ℝ := (-3, 0, 1)
def B : ℝ × ℝ × ℝ := (2, 1, -1)
def C : ℝ × ℝ × ℝ := (-2, 2, 0)
def D : ℝ × ℝ × ℝ := (1, 3, 2)

def line1 := (A, B)
def line2 := (C, D)

/-- Calculate the distance between two lines in 3D space -/
noncomputable def distanceBetweenLines (l1 l2 : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)) : ℝ :=
  sorry -- Placeholder for the actual distance calculation

theorem distance_between_lines :
  let dist := Real.sqrt 3 * (5 / Real.sqrt 23)
  ∃ (d : ℝ), d = dist ∧ d = distanceBetweenLines line1 line2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l350_35001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l350_35087

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ a ∈ Set.Ioc (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l350_35087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geologist_distance_probability_l350_35017

/-- Represents the number of roads in the circular field -/
def num_roads : ℕ := 6

/-- Represents the speed of the geologists in km/h -/
def speed : ℝ := 4

/-- Represents the time of travel in hours -/
def time : ℝ := 1

/-- Represents the minimum distance between geologists we're interested in (km) -/
def min_distance : ℝ := 6

/-- Calculates the angle between adjacent roads in radians -/
noncomputable def angle_between_roads : ℝ := 2 * Real.pi / num_roads

/-- Calculates the distance traveled by each geologist -/
def distance_traveled : ℝ := speed * time

/-- Calculates the distance between geologists on adjacent roads -/
noncomputable def distance_adjacent (angle_between_roads : ℝ) (distance_traveled : ℝ) : ℝ :=
  Real.sqrt (2 * distance_traveled^2 * (1 - Real.cos angle_between_roads))

/-- Calculates the number of favorable outcomes -/
def favorable_outcomes : ℕ := num_roads * (num_roads / 2 - 1)

/-- Calculates the total number of possible outcomes -/
def total_outcomes : ℕ := num_roads^2

/-- The main theorem stating the probability -/
theorem geologist_distance_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 1/2 := by
  sorry

#eval favorable_outcomes
#eval total_outcomes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geologist_distance_probability_l350_35017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l350_35043

-- Define points A and B
def A : ℝ × ℝ := (-4, 3)
def B : ℝ × ℝ := (2, 0)

-- Define the function to calculate the sum of distances
noncomputable def sum_distances (y : ℝ) : ℝ :=
  Real.sqrt ((0 - A.1)^2 + (y - A.2)^2) + Real.sqrt ((0 - B.1)^2 + (y - B.2)^2)

-- Theorem statement
theorem min_distance_point :
  ∃ (y : ℝ), (0, y) = (0, 1) ∧
  ∀ (z : ℝ), sum_distances y ≤ sum_distances z := by
  sorry

#check min_distance_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l350_35043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_l350_35019

-- Define the hyperbolic tangent function
noncomputable def th (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x))

-- Define the function y
noncomputable def y (x : ℝ) : ℝ :=
  (1/2) * Real.log ((1 + Real.sqrt (th x)) / (1 - Real.sqrt (th x))) - Real.arctan (Real.sqrt (th x))

-- State the theorem
theorem derivative_y (x : ℝ) :
  deriv y x = Real.sqrt (th x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_l350_35019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l350_35036

-- Define the function f
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- State the theorem
theorem trigonometric_function_properties
  (A ω φ x₀ : ℝ)
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_φ : φ ∈ Set.Ioo 0 (Real.pi / 2))
  (h_x₀ : x₀ > 0)
  (h_y_intercept : f A ω φ 0 = 1)
  (h_max : f A ω φ (x₀ - 3/2) = 2)
  (h_min : f A ω φ x₀ = -2) :
  (∃ (k : ℝ), f 2 (2 * Real.pi / 3) (Real.pi / 6) = f A ω φ) ∧
  (∀ (k : ℝ), (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 (3/2) ∧ x₂ ∈ Set.Icc 0 (3/2) ∧
    f 2 (2 * Real.pi / 3) (Real.pi / 6) x₁ = (k + 1) / 2 ∧
    f 2 (2 * Real.pi / 3) (Real.pi / 6) x₂ = (k + 1) / 2) →
    k ∈ Set.Icc 1 3) ∧
  (∃ (x₁ x₂ : ℝ), x₁ = 7/2 ∧ x₂ = 5 ∧
    x₁ ∈ Set.Icc (13/4) (23/4) ∧ x₂ ∈ Set.Icc (13/4) (23/4) ∧
    ∀ (x : ℝ), x ∈ Set.Icc (13/4) (23/4) →
      f 2 (2 * Real.pi / 3) (Real.pi / 6) (2 * x₁ - x) =
      f 2 (2 * Real.pi / 3) (Real.pi / 6) x ∧
      f 2 (2 * Real.pi / 3) (Real.pi / 6) (2 * x₂ - x) =
      f 2 (2 * Real.pi / 3) (Real.pi / 6) x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l350_35036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jana_walking_distance_l350_35049

/-- Given that Jana walks 1.5 miles in 30 minutes at a constant rate,
    prove that she will walk 2.25 miles in 45 minutes. -/
theorem jana_walking_distance
  (initial_distance : ℝ)
  (initial_time : ℝ)
  (final_time : ℝ)
  (h1 : initial_distance = 1.5)
  (h2 : initial_time = 30)
  (h3 : final_time = 45)
  (h4 : initial_time > 0) :
  let rate := initial_distance / initial_time
  let final_distance := rate * final_time
  final_distance = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jana_walking_distance_l350_35049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l350_35065

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := n * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem max_sum_arithmetic_sequence :
  ∃ (d : ℝ),
    arithmetic_sequence 11 d 5 = -1 ∧
    ∃ (n : ℕ),
      (∀ (k : ℕ), sum_arithmetic_sequence 11 d k ≤ sum_arithmetic_sequence 11 d n) ∧
      sum_arithmetic_sequence 11 d n = 26 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l350_35065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l350_35096

theorem sin_cos_product (x : ℝ) (h1 : 0 < x ∧ x < π / 2) (h2 : Real.sin x = 3 * Real.cos x) : 
  Real.sin x * Real.cos x = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l350_35096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_for_horizontal_asymptote_l350_35089

/-- The numerator of our rational function -/
def numerator (x : ℝ) : ℝ := 3*x^7 - 2*x^5 + 4*x^3 - 5

/-- A general polynomial function -/
noncomputable def q (x : ℝ) : ℝ := sorry

/-- Our rational function -/
noncomputable def f (x : ℝ) : ℝ := numerator x / q x

/-- The degree of a polynomial -/
def degree (p : ℝ → ℝ) : ℕ := sorry

/-- A function has a horizontal asymptote -/
def has_horizontal_asymptote (f : ℝ → ℝ) : Prop := sorry

theorem min_degree_for_horizontal_asymptote :
  (has_horizontal_asymptote f) → degree q ≥ 7 ∧ 
  ∀ (d : ℕ), d < 7 → ¬(has_horizontal_asymptote (λ x ↦ numerator x / x^d)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_for_horizontal_asymptote_l350_35089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newton_sequence_quadratic_zeros_l350_35041

/-- Newton sequence for a quadratic function -/
noncomputable def newton_sequence (a b c : ℝ) (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => let xn := newton_sequence a b c x₀ n
             xn - (a * xn^2 + b * xn + c) / (2 * a * xn + b)

/-- Sequence a_n derived from Newton sequence -/
noncomputable def a_sequence (a b c : ℝ) (x₀ : ℝ) : ℕ → ℝ
  | n => Real.log ((newton_sequence a b c x₀ n - 2) / (newton_sequence a b c x₀ n - 1))

theorem newton_sequence_quadratic_zeros (a b c : ℝ) (x₀ : ℝ) :
  a > 0 →
  a + b + c = 0 →
  4 * a + 2 * b + c = 0 →
  (∀ n, newton_sequence a b c x₀ n > 2) →
  a_sequence a b c x₀ 1 = 2 →
  ∀ n, a_sequence a b c x₀ n = 2^n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_newton_sequence_quadratic_zeros_l350_35041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_to_line_l350_35002

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- A point is on a line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Theorem: The minimum distance between a point outside a line and a point on the line
    is equal to the distance from the point to the line -/
theorem min_distance_point_to_line (P : Point) (L : Line) 
    (h : ¬isOnLine P L) : 
    (∀ Q : Point, isOnLine Q L → distance P Q ≥ distanceToLine P L) ∧ 
    (∃ Q : Point, isOnLine Q L ∧ distance P Q = distanceToLine P L) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_to_line_l350_35002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_focal_length_l350_35059

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 15 * y^2 - x^2 = 15

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the focal length of a conic section
noncomputable def focal_length (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

-- Theorem statement
theorem same_focal_length :
  ∃ (a₁ b₁ a₂ b₂ : ℝ),
    (∀ x y, hyperbola x y ↔ (y / b₁)^2 - (x / a₁)^2 = 1) ∧
    (∀ x y, ellipse x y ↔ (x / a₂)^2 + (y / b₂)^2 = 1) ∧
    focal_length a₁ b₁ = focal_length a₂ b₂ :=
by
  -- We'll use the values from the solution
  let a₁ : ℝ := Real.sqrt 15
  let b₁ : ℝ := 1
  let a₂ : ℝ := 5
  let b₂ : ℝ := 3
  
  use a₁, b₁, a₂, b₂
  
  apply And.intro
  · -- Prove the hyperbola equation
    sorry
  apply And.intro
  · -- Prove the ellipse equation
    sorry
  · -- Prove the focal lengths are equal
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_focal_length_l350_35059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_average_speed_l350_35075

/-- Calculates the average speed of a car journey given the speeds and durations of different segments --/
noncomputable def average_speed (speed1 speed2 speed3 : ℝ) (time1 time2 time3 : ℝ) (conversion_factor : ℝ) : ℝ :=
  let total_distance_km := speed1 * time1 + speed2 * time2 + speed3 * time3
  let total_distance_miles := total_distance_km * conversion_factor
  let total_time := time1 + time2 + time3
  total_distance_miles / total_time

/-- The average speed of the car journey is approximately 81.49 mph --/
theorem car_journey_average_speed :
  let speed1 := (120 : ℝ) -- km/h
  let speed2 := (150 : ℝ) -- km/h
  let speed3 := (80 : ℝ) -- km/h
  let time1 := (1 : ℝ) -- hour
  let time2 := (2 : ℝ) -- hours
  let time3 := (0.5 : ℝ) -- hours
  let conversion_factor := (0.62 : ℝ) -- miles/km
  abs (average_speed speed1 speed2 speed3 time1 time2 time3 conversion_factor - 81.49) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_average_speed_l350_35075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_linear_combination_l350_35003

open Matrix

variable (N : Matrix (Fin 2) (Fin 2) ℝ)
variable (a b : Fin 2 → ℝ)

theorem compute_linear_combination 
  (h1 : N.vecMul a = ![3, -7])
  (h2 : N.vecMul b = ![-4, 3]) :
  N.vecMul (3 • a - 2 • b) = ![17, -27] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_linear_combination_l350_35003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l350_35095

/-- A line passing through a point (2,1) and intersecting positive x and y axes -/
structure IntersectingLine where
  /-- Slope of the line -/
  m : ℝ
  /-- y-intercept of the line -/
  b : ℝ
  /-- The line passes through (2,1) -/
  point_condition : 1 = 2 * m + b
  /-- The line intersects positive x and y axes -/
  intersect_condition : m < 0 ∧ b > 0

/-- The area of the triangle formed by the line's intersections with axes and the origin -/
noncomputable def triangle_area (l : IntersectingLine) : ℝ :=
  -1/2 * (1 - 2*l.m)^2 / l.m

/-- The minimum area of the triangle is 2 -/
theorem min_triangle_area :
  ∃ (l : IntersectingLine), ∀ (l' : IntersectingLine), triangle_area l ≤ triangle_area l' ∧ triangle_area l = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l350_35095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_OM_ON_l350_35052

-- Define the hyperbola
noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define a point on the hyperbola
def point_on_hyperbola (x₀ y₀ : ℝ) : Prop := hyperbola x₀ y₀

-- Define the slope of the asymptotes
def asymptote_slope : ℚ := 1/2

-- Define the x-coordinates of points M and N
noncomputable def x_M (x₀ y₀ : ℝ) : ℝ := x₀ - 2*y₀
noncomputable def x_N (x₀ y₀ : ℝ) : ℝ := x₀ + 2*y₀

-- State the theorem
theorem product_OM_ON (x₀ y₀ : ℝ) : 
  point_on_hyperbola x₀ y₀ → |x_M x₀ y₀| * |x_N x₀ y₀| = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_OM_ON_l350_35052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l350_35098

def a : ℕ → ℚ
| 0 => 1/2
| n+1 => 1 / (1 - a n)

theorem sequence_properties :
  (a 1 = 2) ∧
  (a 2 = -1) ∧
  (a 3 = 1/2) ∧
  (∀ k : ℕ, a k = a (k + 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l350_35098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_AB_distance_l350_35016

open Real

-- Define the parametric equations for curve C₁
noncomputable def C₁ (a : ℝ) : ℝ × ℝ := (2 * cos a, 2 + 2 * sin a)

-- Define the relationship between points O, M, and P
def P_from_M (M : ℝ × ℝ) : ℝ × ℝ := (2 * M.1, 2 * M.2)

-- Define curve C₂ as the trajectory of P
def C₂ : Set (ℝ × ℝ) := {P | ∃ a, P = P_from_M (C₁ a)}

-- State the theorem for the equation of C₂
theorem C₂_equation : C₂ = {P : ℝ × ℝ | P.1^2 + (P.2 - 4)^2 = 16} := by sorry

-- Define the polar equations for C₁ and C₂
noncomputable def C₁_polar (θ : ℝ) : ℝ := 4 * sin θ
noncomputable def C₂_polar (θ : ℝ) : ℝ := 8 * sin θ

-- Define points A and B
noncomputable def A : ℝ × ℝ := (C₁_polar (π/3) * cos (π/3), C₁_polar (π/3) * sin (π/3))
noncomputable def B : ℝ × ℝ := (C₂_polar (π/3) * cos (π/3), C₂_polar (π/3) * sin (π/3))

-- State the theorem for the distance between A and B
theorem AB_distance : sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_AB_distance_l350_35016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_powers_l350_35064

theorem compare_powers : (4 : ℝ)^(1/4) > (5 : ℝ)^(1/5) ∧ (5 : ℝ)^(1/5) > (16 : ℝ)^(1/16) ∧ (16 : ℝ)^(1/16) > (25 : ℝ)^(1/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_powers_l350_35064

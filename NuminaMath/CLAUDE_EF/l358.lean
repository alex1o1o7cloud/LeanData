import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_duration_is_six_hours_l358_35825

/-- Represents the duration of a car trip in hours -/
structure TripDuration where
  initial : ℝ
  additional : ℝ

/-- Calculates the average speed of a trip given the initial and additional durations -/
noncomputable def averageSpeed (d : TripDuration) (v1 v2 : ℝ) : ℝ :=
  (v1 * d.initial + v2 * d.additional) / (d.initial + d.additional)

/-- Theorem stating that under given conditions, the trip duration is 6 hours -/
theorem trip_duration_is_six_hours :
  ∀ (d : TripDuration),
    d.initial = 4 →
    averageSpeed d 55 70 = 60 →
    d.initial + d.additional = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_duration_is_six_hours_l358_35825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l358_35859

def line1 (t : ℝ) : ℝ × ℝ := (3 + 2*t, -4 - 3*t)
def line2 (s : ℝ) : ℝ × ℝ := (2*s, -3*s)

def direction_vector : ℝ × ℝ := (2, -3)

theorem parallel_lines_distance :
  let v : ℝ × ℝ := (3, -4)
  let proj := ((v.1 * direction_vector.1 + v.2 * direction_vector.2) / (direction_vector.1^2 + direction_vector.2^2))
  let distance := Real.sqrt ((v.1 - proj * direction_vector.1)^2 + (v.2 - proj * direction_vector.2)^2)
  distance = Real.sqrt 13 / 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l358_35859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_molecules_to_moles_approx_l358_35804

/-- Avogadro's number (molecules per mole) -/
noncomputable def avogadro : ℝ := 6.022e23

/-- Number of molecules -/
noncomputable def num_molecules : ℝ := 6e26

/-- Number of moles -/
noncomputable def num_moles : ℝ := num_molecules / avogadro

/-- Approximate equality for real numbers -/
def approx_equal (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

notation x " ≈ " y => approx_equal x y 0.01

theorem molecules_to_moles_approx :
  num_moles ≈ 1000 := by
  sorry

#check molecules_to_moles_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_molecules_to_moles_approx_l358_35804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_max_interval_ratio_l358_35891

-- Define the maximum value of sin x on a closed interval
noncomputable def M (I : Set ℝ) : ℝ := ⨆ (x ∈ I), Real.sin x

-- State the theorem
theorem sin_max_interval_ratio (a : ℝ) (h1 : a > 0) :
  M (Set.Icc 0 a) = 2 * M (Set.Icc a (2 * a)) →
  a = 5 * Real.pi / 6 ∨ a = 13 * Real.pi / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_max_interval_ratio_l358_35891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l358_35895

-- Define the power function as noncomputable
noncomputable def powerFunction (n : ℝ) : ℝ → ℝ := fun x ↦ x^n

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  (∃ n : ℝ, f = powerFunction n) →
  f 4 = (1 : ℝ) / 2 →
  f (1 / 4) = 2 := by
  intros h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l358_35895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_distance_before_karen_wins_l358_35806

/-- Proves the distance Tom drives before Karen wins the bet -/
theorem tom_distance_before_karen_wins 
  (karen_speed tom_speed karen_delay karen_lead : ℝ) 
  (h1 : karen_speed = 60)
  (h2 : tom_speed = 45)
  (h3 : karen_delay = 4 / 60)  -- 4 minutes in hours
  (h4 : karen_lead = 4)
  : tom_speed * (karen_lead + tom_speed * karen_delay) / (karen_speed - tom_speed) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_distance_before_karen_wins_l358_35806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_metal_mass_l358_35802

/-- Proves that the mass of the fourth metal in an alloy is approximately 8.08 kg given specific conditions -/
theorem fourth_metal_mass (m₁ m₂ m₃ m₄ : ℝ) 
  (total_mass : m₁ + m₂ + m₃ + m₄ = 35)
  (first_second_relation : m₁ = 1.5 * m₂)
  (second_third_ratio : m₂ / m₃ = 3 / 4)
  (third_fourth_ratio : m₃ / m₄ = 5 / 6) :
  ∃ ε > 0, |m₄ - 8.08| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_metal_mass_l358_35802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cooling_time_is_two_hours_l358_35840

/-- Represents the characteristics and performance of an old car -/
structure OldCar where
  speed : ℚ  -- Speed in miles per hour
  drive_time : ℚ  -- Time of constant driving before cooling down in hours
  total_distance : ℚ  -- Total distance covered in miles
  total_time : ℚ  -- Total time including driving and cooling down in hours

/-- Calculates the cooling down time for the old car -/
def cooling_down_time (car : OldCar) : ℚ :=
  car.total_time - (car.total_distance / car.speed)

/-- Theorem stating that the cooling down time for the given car is 2 hours -/
theorem cooling_time_is_two_hours :
  let car : OldCar := {
    speed := 8,
    drive_time := 5,
    total_distance := 88,
    total_time := 13
  }
  cooling_down_time car = 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cooling_time_is_two_hours_l358_35840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_surface_area_l358_35844

/-- The total surface area of a regular hexagonal pyramid -/
noncomputable def total_surface_area (h : ℝ) : ℝ := (3 * h^2 * Real.sqrt 3) / 2

/-- Theorem: The total surface area of a regular hexagonal pyramid with apothem h 
    and dihedral angle at the base 60° is equal to (3h² √3) / 2 -/
theorem hexagonal_pyramid_surface_area (h : ℝ) (h_pos : h > 0) :
  let apothem := h
  let dihedral_angle := 60
  total_surface_area apothem = (3 * h^2 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_surface_area_l358_35844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_2x_plus_exp_x_l358_35890

theorem definite_integral_2x_plus_exp_x (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x + Real.exp x) :
  ∫ x in Set.Icc 0 1, f x = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_2x_plus_exp_x_l358_35890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l358_35858

/-- The time taken for a person to cover the entire length of an escalator -/
noncomputable def escalator_time (escalator_speed : ℝ) (escalator_length : ℝ) (person_speed : ℝ) : ℝ :=
  escalator_length / (escalator_speed + person_speed)

/-- Theorem: The time taken for a person to cover the entire length of the escalator is 10 seconds -/
theorem escalator_problem :
  let escalator_speed : ℝ := 15
  let escalator_length : ℝ := 180
  let person_speed : ℝ := 3
  escalator_time escalator_speed escalator_length person_speed = 10 := by
  -- Unfold the definition of escalator_time
  unfold escalator_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l358_35858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarfs_truthfulness_l358_35849

/-- Represents a dwarf's truthfulness -/
inductive Truthfulness
| Truthful
| Liar
deriving DecidableEq

/-- Represents an ice cream flavor -/
inductive IceCream
| Vanilla
| Chocolate
| Fruit
deriving DecidableEq

/-- Represents a dwarf with their truthfulness and preferred ice cream -/
structure Dwarf where
  truthfulness : Truthfulness
  favoriteFlavor : IceCream
deriving DecidableEq

/-- The problem statement -/
theorem dwarfs_truthfulness 
  (dwarfs : Finset Dwarf)
  (total_count : Nat)
  (h_total : dwarfs.card = total_count)
  (h_total_10 : total_count = 10)
  (h_vanilla : (dwarfs.filter (fun d => d.truthfulness = Truthfulness.Truthful ∧ d.favoriteFlavor = IceCream.Vanilla)).card +
               (dwarfs.filter (fun d => d.truthfulness = Truthfulness.Liar ∧ d.favoriteFlavor ≠ IceCream.Vanilla)).card = 10)
  (h_chocolate : (dwarfs.filter (fun d => d.truthfulness = Truthfulness.Truthful ∧ d.favoriteFlavor = IceCream.Chocolate)).card +
                 (dwarfs.filter (fun d => d.truthfulness = Truthfulness.Liar ∧ d.favoriteFlavor ≠ IceCream.Chocolate)).card = 5)
  (h_fruit : (dwarfs.filter (fun d => d.truthfulness = Truthfulness.Truthful ∧ d.favoriteFlavor = IceCream.Fruit)).card +
             (dwarfs.filter (fun d => d.truthfulness = Truthfulness.Liar ∧ d.favoriteFlavor ≠ IceCream.Fruit)).card = 1) :
  (dwarfs.filter (fun d => d.truthfulness = Truthfulness.Truthful)).card = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarfs_truthfulness_l358_35849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l358_35855

open Real

-- Define the triangle ABC
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  c^2 = a^2 + b^2 - 2*a*b*(cos C) ∧
  sin A / a = sin B / b ∧ sin B / b = sin C / c

-- State the theorem
theorem triangle_properties :
  ∀ (a b c A B C : ℝ),
  Triangle a b c A B C →
  a = 4 →
  b = 5 →
  cos C = 1/8 →
  (1/2 * a * b * sin C = 15 * sqrt 7 / 4) ∧
  (c = 6) ∧
  (sin A = sqrt 7 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l358_35855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_A_l358_35832

/-- The length of the segment connecting a point to its reflection over the y-axis --/
noncomputable def reflection_distance (x y : ℝ) : ℝ :=
  Real.sqrt ((2 * x)^2)

/-- Theorem: The length of the segment connecting A(3, -4) to its reflection A' over the y-axis is 6 --/
theorem reflection_distance_A : reflection_distance 3 (-4) = 6 := by
  -- Unfold the definition of reflection_distance
  unfold reflection_distance
  -- Simplify the expression
  simp
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_A_l358_35832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l358_35808

theorem vector_equation_solution (a b : ℝ × ℝ) (lambda : ℝ) :
  a = (1, Real.sqrt 3) →
  (b.1 * b.1 + b.2 * b.2 = 1) →
  a.1 + lambda * b.1 = 0 ∧ a.2 + lambda * b.2 = 0 →
  lambda = 2 ∨ lambda = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l358_35808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_when_a_is_one_subset_iff_a_in_range_l358_35873

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}

-- Theorem 1: When a = 1, A ∪ B = [0, 4]
theorem union_when_a_is_one : 
  A 1 ∪ B = Set.Icc 0 4 := by sorry

-- Theorem 2: A ⊆ B if and only if a ∈ [2, 3]
theorem subset_iff_a_in_range : 
  ∀ a : ℝ, A a ⊆ B ↔ a ∈ Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_when_a_is_one_subset_iff_a_in_range_l358_35873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_inscribed_iff_rectangle_parallelogram_circumscribe_iff_rhombus_parallelogram_inscribed_and_circumscribe_iff_square_l358_35826

/-- A parallelogram with sides a, b and angles α, β -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  α : ℝ
  β : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_angles : 0 < α ∧ 0 < β ∧ α + β = Real.pi

/-- A parallelogram is a rectangle if all its angles are right angles -/
def Parallelogram.isRectangle (p : Parallelogram) : Prop :=
  p.α = Real.pi/2 ∧ p.β = Real.pi/2

/-- A parallelogram is a rhombus if all its sides are equal -/
def Parallelogram.isRhombus (p : Parallelogram) : Prop :=
  p.a = p.b

/-- A parallelogram is a square if it's both a rectangle and a rhombus -/
def Parallelogram.isSquare (p : Parallelogram) : Prop :=
  p.isRectangle ∧ p.isRhombus

/-- A parallelogram can be inscribed in a circle if its opposite angles are supplementary -/
def Parallelogram.canBeInscribed (p : Parallelogram) : Prop :=
  p.α + p.α = Real.pi ∧ p.β + p.β = Real.pi

/-- A parallelogram can circumscribe a circle if the sum of its opposite sides are equal -/
def Parallelogram.canCircumscribe (p : Parallelogram) : Prop :=
  p.a + p.a = p.b + p.b

theorem parallelogram_inscribed_iff_rectangle (p : Parallelogram) :
  p.canBeInscribed ↔ p.isRectangle := by sorry

theorem parallelogram_circumscribe_iff_rhombus (p : Parallelogram) :
  p.canCircumscribe ↔ p.isRhombus := by sorry

theorem parallelogram_inscribed_and_circumscribe_iff_square (p : Parallelogram) :
  (p.canBeInscribed ∧ p.canCircumscribe) ↔ p.isSquare := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_inscribed_iff_rectangle_parallelogram_circumscribe_iff_rhombus_parallelogram_inscribed_and_circumscribe_iff_square_l358_35826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_domain_of_f_l358_35885

-- Define the function f(x) = 3^x
noncomputable def f (x : ℝ) : ℝ := 3^x

-- Define the domain of f
def domain_f : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Define the range of f over its domain
def range_f : Set ℝ := {y | ∃ x ∈ domain_f, f x = y}

-- Theorem stating that the domain of the inverse function is (1, 9]
theorem inverse_domain_of_f : range_f = Set.Ioc 1 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_domain_of_f_l358_35885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cesaro_sum_extended_l358_35857

noncomputable def cesaro_sum (s : List ℝ) : ℝ :=
  (s.scanl (· + ·) 0).sum / s.length

theorem cesaro_sum_extended (b : List ℝ) :
  b.length = 50 →
  cesaro_sum b = 500 →
  cesaro_sum (2 :: b) = 492 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cesaro_sum_extended_l358_35857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_properties_floor_add_integer_floor_sum_inequality_floor_product_equality_l358_35813

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorems to be proved
theorem floor_properties :
  (∀ x : ℝ, floor (x + 2) = floor x + 2) ∧
  (∀ x y : ℝ, floor (x + y) ≤ floor x + floor y + 1) ∧
  (∀ x y : ℝ, floor ((x + 1) * (y + 1)) = floor x * floor y + floor x + floor y + 1) :=
by
  sorry -- Proof skipped for now

-- You can add individual theorems for each property if needed
theorem floor_add_integer (x : ℝ) : floor (x + 2) = floor x + 2 :=
by
  sorry

theorem floor_sum_inequality (x y : ℝ) : floor (x + y) ≤ floor x + floor y + 1 :=
by
  sorry

theorem floor_product_equality (x y : ℝ) :
  floor ((x + 1) * (y + 1)) = floor x * floor y + floor x + floor y + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_properties_floor_add_integer_floor_sum_inequality_floor_product_equality_l358_35813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_l358_35838

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 5) + Real.sin (x / 7)

-- Define the property of x being the smallest positive value where f(x) is maximum
def is_smallest_max (x : ℝ) : Prop :=
  (∀ y : ℝ, y > 0 → f y ≤ f x) ∧ 
  (∀ z : ℝ, 0 < z ∧ z < x → f z < f x)

-- State the theorem
theorem smallest_max_value : 
  ∃ x : ℝ, x = 11100 ∧ is_smallest_max x := by
  sorry

#check smallest_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_l358_35838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consumption_reduction_equals_price_increase_white_sugar_reduction_brown_sugar_reduction_powdered_sugar_reduction_l358_35899

/-- Represents the price and consumption of a type of sugar -/
structure Sugar where
  initialPrice : ℚ
  newPrice : ℚ
  consumption : ℚ

/-- Calculates the percentage increase in price -/
noncomputable def priceIncrease (s : Sugar) : ℚ :=
  (s.newPrice - s.initialPrice) / s.initialPrice * 100

/-- Calculates the new consumption to maintain the same expenditure -/
noncomputable def newConsumption (s : Sugar) : ℚ :=
  s.consumption * (1 - priceIncrease s / 100)

/-- Theorem: The percentage reduction in consumption equals the percentage increase in price -/
theorem consumption_reduction_equals_price_increase (s : Sugar) :
  (s.consumption - newConsumption s) / s.consumption * 100 = priceIncrease s := by sorry

/-- Define the three types of sugar -/
def whiteSugar : Sugar := { initialPrice := 6, newPrice := 15/2, consumption := 1 }
def brownSugar : Sugar := { initialPrice := 8, newPrice := 39/4, consumption := 1 }
def powderedSugar : Sugar := { initialPrice := 10, newPrice := 23/2, consumption := 1 }

/-- Theorems for each type of sugar -/
theorem white_sugar_reduction :
  (whiteSugar.consumption - newConsumption whiteSugar) / whiteSugar.consumption * 100 = 25 := by sorry

theorem brown_sugar_reduction :
  (brownSugar.consumption - newConsumption brownSugar) / brownSugar.consumption * 100 = 21875/1000 := by sorry

theorem powdered_sugar_reduction :
  (powderedSugar.consumption - newConsumption powderedSugar) / powderedSugar.consumption * 100 = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consumption_reduction_equals_price_increase_white_sugar_reduction_brown_sugar_reduction_powdered_sugar_reduction_l358_35899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_radius_proof_l358_35862

/-- The radius of a circle that is internally tangent to four externally tangent circles of radius 2 -/
noncomputable def large_circle_radius : ℝ := 2 * (Real.sqrt 2 + 1)

/-- The four smaller circles, each with radius 2 -/
def small_circle_radius : ℝ := 2

/-- Theorem: The radius of a circle that is internally tangent to four externally tangent circles of radius 2 is equal to 2(√2 + 1) -/
theorem large_circle_radius_proof :
  ∃ (R : ℝ), R = large_circle_radius ∧
  ∃ (c1 c2 c3 c4 : ℝ × ℝ),
    (∀ (i j : Fin 4), i ≠ j → ‖c1 - c2‖ = 2 * small_circle_radius) ∧
    (∀ (i : Fin 4), ‖c1 - (0, 0)‖ = R - small_circle_radius) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_radius_proof_l358_35862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l358_35892

theorem problem_solution : (10 : ℚ) * ((1/2 + 1/5 + 1/10) : ℚ)⁻¹ = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l358_35892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l358_35842

theorem trig_inequality : 
  Real.cos (1/2 : ℝ) > Real.sin (3/2 : ℝ) - Real.sin (1/2 : ℝ) ∧ 
  Real.sin (3/2 : ℝ) - Real.sin (1/2 : ℝ) > Real.cos (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l358_35842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_trajectory_l358_35850

noncomputable def C₁ (α t : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, t * Real.sin α)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

noncomputable def A (α : ℝ) : ℝ × ℝ := (Real.sin α * Real.sin α, -Real.cos α * Real.sin α)

noncomputable def P (α : ℝ) : ℝ × ℝ := ((Real.sin α * Real.sin α) / 2, -(Real.cos α * Real.sin α) / 2)

theorem intersection_points_and_trajectory :
  (∃ t θ, C₁ (Real.pi/4) t = C₂ θ ∧ (C₁ (Real.pi/4) t = (1, 0) ∨ C₁ (Real.pi/4) t = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2))) ∧
  (∃ c r, ∀ α, ∃ x y, P α = (x, y) ∧ (x - c.1)^2 + y^2 = r^2 ∧ c = (1/2, 0) ∧ r = 1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_trajectory_l358_35850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equation_solution_l358_35879

-- State the theorem using the built-in max function
theorem max_equation_solution (x : ℝ) :
  max 1 x = x^2 - 6 ↔ x = 3 ∨ x = -Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equation_solution_l358_35879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_inclination_angle_l358_35803

/-- The inclination angle of a line with slope m -/
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m

/-- The line l defined by parametric equations -/
noncomputable def line_l (s : ℝ) : ℝ × ℝ := (s + 1, Real.sqrt 3 * s)

theorem line_l_inclination_angle :
  inclination_angle ((line_l 1).2 - (line_l 0).2) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_inclination_angle_l358_35803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_slope_l358_35845

/-- Given a hyperbola with eccentricity 2 and equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    where the right focus F₂ is the focus of the parabola y² = 8x,
    and a line l passing through F₂ intersects the right branch of the hyperbola at points P and Q,
    if PF₁ ⊥ QF₁ (where F₁ is the left focus), then the slope of line l is ± 3√7/7. -/
theorem hyperbola_parabola_intersection_slope (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2  -- eccentricity
  let hyperbola := λ (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let parabola := λ (x y : ℝ) ↦ y^2 = 8*x
  let F₂ := (2, 0)  -- right focus (focus of parabola)
  let F₁ := (-2, 0)  -- left focus
  ∀ P Q : ℝ × ℝ,
    hyperbola P.1 P.2 →
    hyperbola Q.1 Q.2 →
    (∃ t : ℝ, P.2 = t * (P.1 - F₂.1) ∧ Q.2 = t * (Q.1 - F₂.1)) →  -- P and Q on line l through F₂
    ((P.1 - F₁.1) * (Q.1 - F₁.1) + (P.2 - F₁.2) * (Q.2 - F₁.2) = 0) →  -- PF₁ ⊥ QF₁
    (∃ k : ℝ, k = 3 * Real.sqrt 7 / 7 ∨ k = -3 * Real.sqrt 7 / 7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_slope_l358_35845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_inclination_angle_l358_35856

/-- The circle equation: x² + y² - 4x - 6y + 9 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 9 = 0

/-- The line equation: y = kx + 3 -/
def line_equation (k x y : ℝ) : Prop := y = k*x + 3

/-- The chord length is 2√3 -/
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

/-- The inclination angle of the line -/
def inclination_angle (θ : ℝ) : Prop := θ = Real.pi/6 ∨ θ = 5*Real.pi/6

theorem chord_inclination_angle 
  (k : ℝ) 
  (h1 : ∃ x y : ℝ, circle_equation x y ∧ line_equation k x y)
  (h2 : ∃ x1 y1 x2 y2 : ℝ, 
    circle_equation x1 y1 ∧ circle_equation x2 y2 ∧ 
    line_equation k x1 y1 ∧ line_equation k x2 y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = chord_length^2) :
  ∃ θ : ℝ, inclination_angle θ ∧ k = Real.tan θ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_inclination_angle_l358_35856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_increase_percentage_l358_35851

/-- Given Elaine's rent spending patterns over two years, prove that this year's rent is 202.5% of last year's rent. -/
theorem rent_increase_percentage (E : ℝ) : 
  (0.30 * (1.35 * E)) / (0.20 * E) * 100 = 202.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_increase_percentage_l358_35851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_15_l358_35884

/-- Represents the number of radios purchased by the dealer -/
def n : ℕ := 15  -- We'll use 15 as our example value

/-- The total cost of all radios -/
def total_cost : ℚ := 200

/-- The profit per radio sold (not donated) -/
def profit_per_radio : ℚ := 10

/-- The number of radios donated -/
def donated_radios : ℕ := 3

/-- The total profit made by the dealer -/
def total_profit : ℚ := 100

/-- The cost per radio -/
def cost_per_radio : ℚ := total_cost / n

/-- The income from donated radios -/
def donated_income : ℚ := (donated_radios : ℚ) * (cost_per_radio / 2)

/-- The income from selling the remaining radios -/
def selling_income : ℚ := (n - donated_radios : ℚ) * (cost_per_radio + profit_per_radio)

/-- The profit equation -/
def profit_equation (m : ℕ) : Prop :=
  let cost_per_radio_m : ℚ := total_cost / m
  let donated_income_m : ℚ := (donated_radios : ℚ) * (cost_per_radio_m / 2)
  let selling_income_m : ℚ := (m - donated_radios : ℚ) * (cost_per_radio_m + profit_per_radio)
  selling_income_m + donated_income_m - total_cost = total_profit

/-- The theorem stating that 15 is the smallest positive integer satisfying the profit equation -/
theorem smallest_n_is_15 : 
  (∀ m : ℕ, m > 0 ∧ m < 15 → ¬profit_equation m) ∧ profit_equation 15 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_15_l358_35884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_f_definition_f_order_l358_35810

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Ioo (-Real.pi/2) (Real.pi/2) then x + Real.tan x else 0

theorem f_symmetry (x : ℝ) : f x = f (Real.pi - x) := by sorry

theorem f_definition (x : ℝ) (h : x ∈ Set.Ioo (-Real.pi/2) (Real.pi/2)) : 
  f x = x + Real.tan x := by sorry

theorem f_order : f 3 < f 1 ∧ f 1 < f 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_f_definition_f_order_l358_35810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_l358_35894

-- Define circle O
def circle_O (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 1

-- Define the center of circle C
def center_C : ℝ × ℝ := (3, -4)

-- Define tangency between two circles
def tangent (c1 c2 : (ℝ × ℝ) → Prop) : Prop :=
  ∃ (p : ℝ × ℝ), c1 p ∧ c2 p

-- Define circle C (we'll prove this is correct)
def circle_C (p : ℝ × ℝ) : Prop :=
  (p.1 - 3)^2 + (p.2 + 4)^2 = 16 ∨ (p.1 - 3)^2 + (p.2 + 4)^2 = 36

-- Theorem statement
theorem circle_C_equation :
  tangent circle_O circle_C ∧ 
  (∀ (p : ℝ × ℝ), circle_C p → (p.1 - center_C.1)^2 + (p.2 - center_C.2)^2 = 16 ∨
                               (p.1 - center_C.1)^2 + (p.2 - center_C.2)^2 = 36) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_l358_35894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_with_tan_l358_35889

theorem cos_double_angle_with_tan (a : ℝ) (h : Real.tan a = 2) : Real.cos (2 * a) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_with_tan_l358_35889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l358_35848

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + x + a)

theorem function_properties :
  ∃ (a : ℝ),
    (deriv (f a) 0 = 2) ∧
    (a = 1) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ Real.exp (-1)) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ 7 * Real.exp 2) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = Real.exp (-1)) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 7 * Real.exp 2) :=
by
  -- We choose a = 1 as per the solution
  use 1
  
  -- Prove each part of the conjunction
  constructor
  · sorry -- Proof that deriv (f 1) 0 = 2
  
  constructor
  · rfl -- a = 1 is true by reflexivity
  
  constructor
  · sorry -- Proof that ∀ x ∈ [-2, 2], f 1 x ≥ e^(-1)
  
  constructor
  · sorry -- Proof that ∀ x ∈ [-2, 2], f 1 x ≤ 7e^2
  
  constructor
  · sorry -- Proof that ∃ x ∈ [-2, 2], f 1 x = e^(-1)
  
  · sorry -- Proof that ∃ x ∈ [-2, 2], f 1 x = 7e^2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l358_35848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l358_35830

open Real

theorem triangle_angle_theorem (a b c A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  2 * b * Real.cos B = a * Real.cos C + c * Real.cos A →
  B = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l358_35830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_square_height_is_two_l358_35886

/-- The vertical distance from the highest point of a rotated square to the original base line -/
noncomputable def rotated_square_height (side_length : ℝ) : ℝ :=
  let diagonal := side_length * Real.sqrt 2
  let center_height := side_length / 2
  let drop_distance := diagonal / 2 - center_height
  let top_point_height := Real.sqrt 2 * side_length / 2
  center_height + (top_point_height - drop_distance)

/-- Theorem stating that for squares of side length 2, the height of the rotated square is 2 -/
theorem rotated_square_height_is_two :
  rotated_square_height 2 = 2 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval rotated_square_height 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_square_height_is_two_l358_35886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_fifty_percent_l358_35839

/-- A rectangle tiled with squares and hexagons -/
structure TiledRectangle where
  width : ℚ
  height : ℚ
  num_squares : ℕ
  square_side : ℚ

/-- Calculate the area of the rectangle -/
def rectangle_area (r : TiledRectangle) : ℚ :=
  r.width * r.height

/-- Calculate the area covered by squares -/
def squares_area (r : TiledRectangle) : ℚ :=
  r.num_squares * (r.square_side ^ 2)

/-- Calculate the area covered by hexagons -/
def hexagons_area (r : TiledRectangle) : ℚ :=
  rectangle_area r - squares_area r

/-- Calculate the percentage of area covered by hexagons -/
def hexagon_percentage (r : TiledRectangle) : ℚ :=
  (hexagons_area r / rectangle_area r) * 100

/-- Theorem: The area covered by hexagons is 50% of the total area -/
theorem hexagon_area_fifty_percent (r : TiledRectangle) 
  (h1 : r.width = 4)
  (h2 : r.height = 3)
  (h3 : r.num_squares = 6)
  (h4 : r.square_side = 1) :
  hexagon_percentage r = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_fifty_percent_l358_35839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_of_a_l358_35836

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + (1 - a) * x - Real.log (a * x)

theorem tangent_line_and_range_of_a (a : ℝ) (h_a : a > 0) :
  (∀ x > 0, f 1 x = Real.exp x - Real.log x) ∧
  (∃ m b : ℝ, ∀ x : ℝ, m * x + b = (Real.exp 1 - 1) * x - f 1 1 + 1) ∧
  (∀ x > 0, f a x ≥ 0 ↔ 0 < a ∧ a ≤ Real.exp 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_of_a_l358_35836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l358_35835

/-- Calculate compound interest given principal, rate, and time -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Calculate simple interest given principal, rate, and time -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * (time : ℝ) / 100

theorem compound_interest_calculation (P : ℝ) :
  simple_interest P 10 6 = 1800 →
  ∃ (x : ℝ), abs (compound_interest P 10 6 - x) < 0.01 ∧ x = 2314.68 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l358_35835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_202_l358_35807

/-- Converts a base-10 number to its base-9 representation -/
def toBase9 (n : ℕ) : ℕ := sorry

/-- Converts a base-9 number to its base-10 representation -/
def fromBase9 (n : ℕ) : ℕ := sorry

theorem base_conversion_202 :
  toBase9 202 = 244 ∧ fromBase9 244 = 202 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_202_l358_35807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_items_percentage_l358_35831

/-- Represents the sales data for Toby's garage sale -/
structure GarageSale where
  treadmill_price : ℚ
  chest_price : ℚ
  tv_price : ℚ
  total_sales : ℚ

/-- Calculates the percentage of total sales from the three specified items -/
def percentage_from_items (sale : GarageSale) : ℚ :=
  ((sale.treadmill_price + sale.chest_price + sale.tv_price) / sale.total_sales) * 100

/-- Theorem stating that the percentage of sales from the three items is 75% -/
theorem three_items_percentage (sale : GarageSale) 
  (h1 : sale.treadmill_price = 100)
  (h2 : sale.chest_price = sale.treadmill_price / 2)
  (h3 : sale.tv_price = sale.treadmill_price * 3)
  (h4 : sale.total_sales = 600) :
  percentage_from_items sale = 75 := by
  sorry

#eval percentage_from_items { treadmill_price := 100, chest_price := 50, tv_price := 300, total_sales := 600 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_items_percentage_l358_35831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_squares_l358_35801

/-- An arithmetic progression is represented by its first term and common difference -/
structure ArithmeticProgression where
  first_term : ℤ
  common_difference : ℤ

/-- A term in an arithmetic progression -/
def term (ap : ArithmeticProgression) (n : ℕ) : ℤ :=
  ap.first_term + n * ap.common_difference

/-- Predicate to check if a number is a square -/
def is_square (x : ℤ) : Prop := ∃ y : ℤ, y * y = x

/-- Predicate to check if an arithmetic progression contains a square -/
def contains_square (ap : ArithmeticProgression) : Prop :=
  ∃ n : ℕ, is_square (term ap n)

/-- Main theorem: If an infinite arithmetic progression contains a square,
    it contains infinitely many squares -/
theorem infinitely_many_squares (ap : ArithmeticProgression) :
  contains_square ap → ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ n, is_square (term ap (f n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_squares_l358_35801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wx_xy_ratio_l358_35833

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℚ
  height : ℚ

/-- The large rectangle WXYZ -/
def largeRectangle : Rectangle := ⟨12, 7⟩

/-- One of the seven identical smaller rectangles -/
def smallRectangle : Rectangle := ⟨3, 4⟩

/-- The number of small rectangles horizontally aligned along WX -/
def horizontalCount : ℕ := 4

/-- The number of small rectangles vertically stacked along XY -/
def verticalCount : ℕ := 3

/-- The total number of small rectangles -/
def totalCount : ℕ := 7

theorem wx_xy_ratio :
  largeRectangle.width / largeRectangle.height = 12 / 7 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wx_xy_ratio_l358_35833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l358_35860

/-- Represents the average speed of a car under specific conditions. -/
def average_speed (v : ℝ) : Prop :=
  let first_half_speed := v + 20
  let second_half_speed := 0.8 * v
  let total_time := (1 / (2 * first_half_speed)) + (1 / (2 * second_half_speed))
  v = 1 / total_time

/-- 
Theorem: The average speed of a car is 60 km/h when it travels:
- half the distance at 20 km/h faster than its average speed
- the other half at 20% less than its average speed
-/
theorem car_average_speed : average_speed 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l358_35860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l358_35861

noncomputable def f (x : ℝ) := 5 * Real.sin (2 * x - Real.pi / 6)

theorem function_properties (k : ℤ) :
  f (Real.pi / 12) = 0 ∧
  f (Real.pi / 3) = 5 ∧
  HasDerivAt f 0 (Real.pi / 3) ∧
  (∀ x, |f x| ≤ 5) →
  StrictMonoOn f (Set.Icc ((-Real.pi / 6) + k * Real.pi) ((Real.pi / 3) + k * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l358_35861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l358_35874

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((1 + a * x) / (1 + x))

-- State the theorem
theorem odd_function_properties (a b : ℝ) (h1 : a ≠ 1) 
  (h2 : ∀ x ∈ Set.Ioo (-b) b, f a x = -f a (-x)) : 
  (a = -1) ∧ 
  (b ∈ Set.Ioo 0 1) ∧ 
  (∀ x : ℝ, f a x > 0 ↔ x ∈ Set.Ioo (-1) 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l358_35874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_cost_theorem_l358_35898

/-- Parking cost structure and calculation -/
structure ParkingCost where
  baseCost : ℚ  -- Base cost for up to 2 hours
  baseHours : ℚ  -- Number of hours covered by base cost
  additionalCost : ℚ  -- Cost per hour after base hours
  totalHours : ℚ  -- Total parking duration

/-- Calculate the total parking cost -/
def totalCost (p : ParkingCost) : ℚ :=
  p.baseCost + (p.totalHours - p.baseHours) * p.additionalCost

/-- Calculate the average cost per hour -/
def averageCostPerHour (p : ParkingCost) : ℚ :=
  totalCost p / p.totalHours

/-- Theorem: The average cost per hour for 9 hours of parking is $3.58 -/
theorem parking_cost_theorem (p : ParkingCost) 
  (h1 : p.baseCost = 20)
  (h2 : p.baseHours = 2)
  (h3 : p.additionalCost = 7/4)
  (h4 : p.totalHours = 9) :
  averageCostPerHour p = 179/50 := by
  sorry

#eval averageCostPerHour { baseCost := 20, baseHours := 2, additionalCost := 7/4, totalHours := 9 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_cost_theorem_l358_35898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_number_solutions_rational_number_solutions_l358_35869

-- Define the equation using Real numbers instead of Rational
def equation (x y : ℝ) : Prop := x^y = y^x

-- Theorem for natural number solutions
theorem natural_number_solutions :
  ∀ x y : ℕ, equation (x : ℝ) (y : ℝ) ↔ (x = y ∨ (x = 4 ∧ y = 2)) :=
by sorry

-- Theorem for rational number solutions
theorem rational_number_solutions :
  ∀ x y : ℚ, equation (x : ℝ) (y : ℝ) ↔ 
    ∃ p : ℝ, x = (1 + 1/p)^(p+1) ∧ y = (1 + 1/p)^p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_number_solutions_rational_number_solutions_l358_35869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_and_circles_with_specific_areas_l358_35805

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an equilateral triangle in a 2D plane -/
structure EquilateralTriangle where
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ
  vertex3 : ℝ × ℝ

/-- Checks if a point is inside a circle -/
def isInside (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

/-- Calculates the area of points satisfying a given condition -/
noncomputable def areaOfPoints (condition : ℝ × ℝ → Prop) : ℝ := sorry

/-- Main theorem stating the existence of the required configuration -/
theorem exists_triangle_and_circles_with_specific_areas :
  ∃ (t : EquilateralTriangle) (c1 c2 c3 : Circle),
    c1.center = t.vertex1 ∧ c2.center = t.vertex2 ∧ c3.center = t.vertex3 ∧
    areaOfPoints (λ p ↦ (isInside p c1 ∨ isInside p c2 ∨ isInside p c3) ∧
                       ¬(isInside p c1 ∧ isInside p c2) ∧
                       ¬(isInside p c1 ∧ isInside p c3) ∧
                       ¬(isInside p c2 ∧ isInside p c3)) = 100 ∧
    areaOfPoints (λ p ↦ (isInside p c1 ∧ isInside p c2 ∧ ¬isInside p c3) ∨
                       (isInside p c1 ∧ isInside p c3 ∧ ¬isInside p c2) ∨
                       (isInside p c2 ∧ isInside p c3 ∧ ¬isInside p c1)) = 10 ∧
    areaOfPoints (λ p ↦ isInside p c1 ∧ isInside p c2 ∧ isInside p c3) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_and_circles_with_specific_areas_l358_35805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sqrt5_minus3_l358_35871

theorem quadratic_root_sqrt5_minus3 :
  ∃ (a b : ℚ), (a = 6 ∧ b = -4) ∧
  (Real.sqrt 5 - 3)^2 + a * (Real.sqrt 5 - 3) + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sqrt5_minus3_l358_35871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_derivative_condition_l358_35827

theorem monotonic_increasing_derivative_condition (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, HasDerivAt f (f' x) x) →
  ((∀ x, f' x ≥ 0) → Monotone f) ∧
  ¬ (Monotone f → (∀ x, f' x ≥ 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_derivative_condition_l358_35827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_coefficient_coefficient_is_negative_41_l358_35893

/-- The expansion of (3x^3 + 4x^2 + 5x)(2x^2 - 9x + 1) -/
def expansion (x : ℝ) : ℝ :=
  6*x^5 - 27*x^4 + 3*x^3 + 8*x^4 - 36*x^3 + 4*x^2 + 10*x^3 - 45*x^2 + 5*x

/-- The coefficient of x^2 in the expansion of (3x^3 + 4x^2 + 5x)(2x^2 - 9x + 1) is -41 -/
theorem x_squared_coefficient (x : ℝ) : 
  (3*x^3 + 4*x^2 + 5*x) * (2*x^2 - 9*x + 1) = expansion x :=
by sorry

/-- The coefficient of x^2 in the expansion is -41 -/
theorem coefficient_is_negative_41 : 
  ∃ p : Polynomial ℝ, (p.coeff 2 = -41 ∧ ∀ x, p.eval x = expansion x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_coefficient_coefficient_is_negative_41_l358_35893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_minus_alpha_l358_35878

theorem sin_pi_half_minus_alpha (α : ℝ) : Real.sin (π / 2 - α) = Real.cos α := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_minus_alpha_l358_35878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_fee_proof_l358_35800

theorem workshop_fee_proof (initial_people : ℕ) (final_people : ℕ) (cost_difference : ℚ) :
  initial_people = 4 →
  final_people = 7 →
  cost_difference = 15 →
  ∃ (total_cost : ℚ), 
    total_cost / initial_people - total_cost / final_people = cost_difference ∧
    total_cost = 140 :=
by
  intro h_initial h_final h_diff
  use 140
  constructor
  · -- Proof of the equation
    rw [h_initial, h_final, h_diff]
    norm_num
  · -- Proof of total_cost = 140
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_fee_proof_l358_35800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l358_35864

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line with a given slope -/
structure Line where
  slope : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ := sorry

/-- Checks if a point is on the ellipse -/
def on_ellipse (p : Point) (e : Ellipse a b) : Prop := sorry

theorem ellipse_eccentricity_special_case
  (a b c : ℝ)
  (e : Ellipse a b)
  (A : Point)
  (F : Point)
  (l : Line)
  (B : Point)
  (h_A : A.x = -a ∧ A.y = 0)
  (h_F : F.x = c ∧ F.y = 0)
  (h_l : l.slope = Real.tan (30 * π / 180))
  (h_intersect : on_ellipse B e)
  (h_perpendicular : (B.x - F.x) * (A.x - F.x) + (B.y - F.y) * (A.y - F.y) = 0) :
  eccentricity e = (3 - Real.sqrt 3) / 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l358_35864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_equal_sugar_distribution_l358_35875

/-- Represents a jar with its volume and sugar content -/
structure Jar where
  volume : ℚ
  sugar : ℚ

/-- Represents the state of the three jars -/
structure JarState where
  jar1 : Jar
  jar2 : Jar
  jar3 : Jar

/-- Initial state of the jars -/
def initial_state : JarState :=
  { jar1 := { volume := 0, sugar := 0 }
  , jar2 := { volume := 700, sugar := 50 }
  , jar3 := { volume := 800, sugar := 60 }
  }

/-- The volume of the measuring cup -/
def measuring_cup_volume : ℚ := 100

/-- Function to apply a transfer between jars -/
def apply_transfer (state : JarState) (transfer_volume : ℚ) : JarState :=
  sorry

/-- Theorem stating the impossibility of achieving the desired state -/
theorem impossible_equal_sugar_distribution (state : JarState) :
  state.jar1.volume = 0 ∧ state.jar2.sugar = state.jar3.sugar →
  ¬ (∃ (n : ℕ), ∃ (transfers : Fin (n + 1) → JarState), 
     transfers 0 = initial_state ∧
     (∀ i : Fin n, transfers i.succ = apply_transfer (transfers i) measuring_cup_volume) ∧
     transfers (Fin.last n) = state) :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_equal_sugar_distribution_l358_35875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_assignments_for_passing_l358_35828

/-- The number of assignments required for the nth point -/
def assignments_for_point (n : ℕ) : ℕ := (Int.ceil (n / 4 : ℚ)).toNat

/-- The total number of assignments required for the first n points -/
def total_assignments (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => assignments_for_point (i + 1))

/-- The minimum number of points required (80% of 20) -/
def min_points : ℕ := 16

theorem min_assignments_for_passing :
  total_assignments min_points = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_assignments_for_passing_l358_35828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_mathematicians_asleep_l358_35863

/-- Represents a mathematician at the conference -/
structure Mathematician where
  id : Nat
deriving Repr, DecidableEq

/-- Represents a sleep instance -/
structure SleepInstance where
  mathematician : Mathematician
  time : Nat
deriving Repr, DecidableEq

/-- The conference setup -/
structure Conference where
  mathematicians : Finset Mathematician
  sleepInstances : Finset SleepInstance

def Conference.validSetup (conf : Conference) : Prop :=
  (conf.mathematicians.card = 9) ∧
  (∀ m : Mathematician, m ∈ conf.mathematicians →
    (conf.sleepInstances.filter (λ si : SleepInstance => si.mathematician = m)).card ≤ 4) ∧
  (∀ m1 m2 : Mathematician, m1 ∈ conf.mathematicians → m2 ∈ conf.mathematicians → m1 ≠ m2 →
    ∃ t : Nat, (SleepInstance.mk m1 t) ∈ conf.sleepInstances ∧
              (SleepInstance.mk m2 t) ∈ conf.sleepInstances)

/-- The main theorem -/
theorem three_mathematicians_asleep (conf : Conference) (h : conf.validSetup) :
  ∃ t : Nat, ∃ m1 m2 m3 : Mathematician,
    m1 ∈ conf.mathematicians ∧ m2 ∈ conf.mathematicians ∧ m3 ∈ conf.mathematicians ∧
    m1 ≠ m2 ∧ m1 ≠ m3 ∧ m2 ≠ m3 ∧
    (SleepInstance.mk m1 t) ∈ conf.sleepInstances ∧
    (SleepInstance.mk m2 t) ∈ conf.sleepInstances ∧
    (SleepInstance.mk m3 t) ∈ conf.sleepInstances :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_mathematicians_asleep_l358_35863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harris_walking_time_harris_walking_time_proof_l358_35897

/-- Represents the walking speed of a person relative to Mr. Harris's speed -/
def RelativeSpeed : Type := ℝ

/-- Represents the distance to a destination relative to the distance to Mr. Harris's store -/
def RelativeDistance : Type := ℝ

/-- Represents time in hours -/
def Time : Type := ℝ

/-- The theorem statement -/
theorem harris_walking_time 
  (your_speed : RelativeSpeed)
  (your_destination : RelativeDistance)
  (your_time : Time)
  (h1 : your_speed = (2 : ℝ))
  (h2 : your_destination = (3 : ℝ))
  (h3 : your_time = (3 : ℝ))
  : Time := (2 : ℝ)

/-- The proof of the theorem -/
theorem harris_walking_time_proof
  (your_speed : RelativeSpeed)
  (your_destination : RelativeDistance)
  (your_time : Time)
  (h1 : your_speed = (2 : ℝ))
  (h2 : your_destination = (3 : ℝ))
  (h3 : your_time = (3 : ℝ))
  : harris_walking_time your_speed your_destination your_time h1 h2 h3 = (2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harris_walking_time_harris_walking_time_proof_l358_35897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l358_35880

/-- Represents the speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 72

/-- Represents the length of the train in meters -/
noncomputable def train_length : ℝ := 180

/-- Converts km/hr to m/s -/
noncomputable def km_hr_to_m_s (speed : ℝ) : ℝ := speed * 1000 / 3600

/-- Calculates the time taken for the train to cross the pole -/
noncomputable def time_to_cross (speed : ℝ) (length : ℝ) : ℝ :=
  length / (km_hr_to_m_s speed)

/-- Theorem stating that the train takes 9 seconds to cross the pole -/
theorem train_crossing_time :
  time_to_cross train_speed train_length = 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l358_35880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l358_35870

/-- The constant term in the expansion of (4x + 2/x)^8 -/
def constant_term : ℕ := 286720

/-- The binomial expansion of (4x + 2/x)^8 -/
noncomputable def expansion (x : ℝ) : ℝ := (4*x + 2/x)^8

/-- Theorem stating the existence of a function f such that
    expansion x = constant_term + x * f x for all non-zero x -/
theorem constant_term_proof :
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, expansion x = constant_term + x * f x) := by
  sorry

#eval constant_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l358_35870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l358_35887

def determinant (a1 a2 a3 a4 : ℝ) : ℝ := a1 * a4 - a2 * a3

noncomputable def f (x : ℝ) : ℝ := 
  determinant (Real.sin (2 * x)) (Real.cos (2 * x)) 1 (Real.sqrt 3)

noncomputable def shifted_f (x : ℝ) : ℝ := f (x - Real.pi / 3)

theorem axis_of_symmetry :
  ∃ k : ℤ, shifted_f (Real.pi / 6 + k * Real.pi / 2) = shifted_f (Real.pi / 6 - k * Real.pi / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l358_35887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_draw_all_colors_l358_35816

def white_balls : ℕ := 5
def red_balls : ℕ := 4
def black_balls : ℕ := 3
def total_balls : ℕ := white_balls + red_balls + black_balls
def drawn_balls : ℕ := 10

theorem certain_draw_all_colors :
  drawn_balls ≥ total_balls - white_balls + 1 ∧
  drawn_balls ≥ total_balls - red_balls + 1 ∧
  drawn_balls ≥ total_balls - black_balls + 1 →
  drawn_balls = total_balls := by
  sorry

#check certain_draw_all_colors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_draw_all_colors_l358_35816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_circumradius_times_half_pedal_perimeter_l358_35829

/-- An acute-angled triangle with vertices A, B, and C -/
structure AcuteTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  isAcute : Prop

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : AcuteTriangle) : ℝ :=
  sorry

/-- The pedal triangle formed by the feet of the altitudes -/
noncomputable def pedalTriangle (t : AcuteTriangle) : AcuteTriangle :=
  sorry

/-- The perimeter of a triangle -/
noncomputable def perimeter (t : AcuteTriangle) : ℝ :=
  sorry

/-- The area of a triangle -/
noncomputable def area (t : AcuteTriangle) : ℝ :=
  sorry

/-- The main theorem -/
theorem area_equals_circumradius_times_half_pedal_perimeter (t : AcuteTriangle) :
  area t = (circumradius t * perimeter (pedalTriangle t)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_circumradius_times_half_pedal_perimeter_l358_35829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_days_to_triple_repayment_l358_35853

/-- The least number of days to repay at least three times the borrowed amount -/
theorem least_days_to_triple_repayment : ℕ := by
  let borrowed_amount : ℝ := 50
  let daily_interest_rate : ℝ := 0.1
  let repayment_amount (x : ℝ) : ℝ := borrowed_amount + borrowed_amount * daily_interest_rate * x
  
  have h : ∀ y : ℕ, y < 20 → repayment_amount y < 3 * borrowed_amount := by sorry
  have h' : repayment_amount 20 ≥ 3 * borrowed_amount := by sorry
  
  exact 20

#check least_days_to_triple_repayment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_days_to_triple_repayment_l358_35853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_k_l358_35824

def k : ℕ := 10^30 - 36

theorem sum_of_digits_k : (Nat.digits 10 k).sum = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_k_l358_35824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_symmetric_curve_l358_35882

/-- Given a curve y = sin(ωx + π/3) with ω > 0, if the curve is symmetric about the line x = π,
    then the minimum value of ω is 1/6 -/
theorem min_omega_symmetric_curve (ω : ℝ) (h1 : ω > 0) :
  (∀ x : ℝ, Real.sin (ω * x + π / 3) = Real.sin (ω * (2 * π - x) + π / 3)) →
  ω ≥ 1 / 6 ∧ ∃ (ω₀ : ℝ), ω₀ = 1 / 6 ∧ (∀ x : ℝ, Real.sin (ω₀ * x + π / 3) = Real.sin (ω₀ * (2 * π - x) + π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_symmetric_curve_l358_35882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_in_circle_l358_35821

theorem parabola_vertex_in_circle (a b : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*a*x₁ + b = 0 ∧ x₂^2 + 2*a*x₂ + b = 0) →
  (∀ x y : ℝ, y = x^2 + 2*a*x + b → (x + a)^2 + (y - (b - a^2))^2 ≤ ((x₁ - x₂)/2)^2) →
  a^2 - 1 < b ∧ b < a^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_in_circle_l358_35821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_inverse_l358_35883

/-- The function f(x) = (x-1)/(x+1) -/
noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

/-- Theorem: For all non-zero real x, f(x) + f(1/x) = 0 -/
theorem f_sum_inverse (x : ℝ) (hx : x ≠ 0) : f x + f (1/x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_inverse_l358_35883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_comparison_most_cost_effective_plan_l358_35868

/-- Represents a supermarket with pricing and discount information -/
structure Supermarket where
  racket_price : ℚ
  ball_price : ℚ
  discount : ℚ
  free_balls : ℕ

/-- Calculates the cost of purchasing from a supermarket -/
def purchase_cost (s : Supermarket) (n : ℕ) (k : ℕ) : ℚ :=
  (s.racket_price * n + s.ball_price * (n * k - n * s.free_balls)) * (1 - s.discount)

/-- Theorem stating the cost-effectiveness comparison between supermarkets A and B -/
theorem supermarket_comparison (n : ℕ) (k : ℕ) : 
  let A : Supermarket := ⟨50, 2, 1/10, 0⟩
  let B : Supermarket := ⟨50, 2, 0, 4⟩
  (n = 1 ∧ k = 10 → purchase_cost B n k < purchase_cost A n k) ∧
  (k < 15 → purchase_cost B n k < purchase_cost A n k) ∧
  (k = 15 → purchase_cost B n k = purchase_cost A n k) ∧
  (k > 15 → purchase_cost B n k > purchase_cost A n k) :=
sorry

/-- Theorem stating the most cost-effective plan for n rackets and 20 balls per racket -/
theorem most_cost_effective_plan (n : ℕ) :
  let A : Supermarket := ⟨50, 2, 1/10, 0⟩
  let B : Supermarket := ⟨50, 2, 0, 4⟩
  let optimal_cost := n * 50 + (20 * n - 4 * n) * 2 * (9/10)
  optimal_cost = 788/10 * n ∧
  ∀ (plan : ℕ → ℕ), 
    plan 0 + plan 1 = n → 
    purchase_cost B (plan 0) 20 + purchase_cost A (plan 1) 20 ≥ optimal_cost :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_comparison_most_cost_effective_plan_l358_35868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_used_l358_35822

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | Stratified
  | RandomNumberTable
  | Systematic

/-- Represents a class of students -/
structure StudentClass where
  size : Nat
  numbering : Finset Nat

/-- Represents a grade with multiple classes -/
structure Grade where
  classes : List StudentClass

/-- The sampling method used in the given scenario -/
def samplingMethodUsed (g : Grade) (selectedNumber : Nat) : SamplingMethod :=
  SamplingMethod.Systematic

/-- Theorem stating that the sampling method used is systematic sampling -/
theorem systematic_sampling_used (g : Grade) 
    (h1 : g.classes.length = 12)
    (h2 : ∀ c ∈ g.classes, c.size = 50)
    (h3 : ∀ c ∈ g.classes, c.numbering = Finset.range 50)
    (h4 : ∀ c ∈ g.classes, 40 ∈ c.numbering) :
    samplingMethodUsed g 40 = SamplingMethod.Systematic := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_used_l358_35822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_3_seconds_l358_35817

/-- The motion equation of an object -/
noncomputable def s (t : ℝ) : ℝ := 1 - t + t^2

/-- The instantaneous velocity at time t -/
noncomputable def v (t : ℝ) : ℝ := deriv s t

theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_3_seconds_l358_35817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_side_length_formula_l358_35834

/-- The side length of a regular decagon inscribed in a circle -/
noncomputable def regular_decagon_side_length (r : ℝ) : ℝ := r * (Real.sqrt 5 - 1) / 2

/-- Theorem: The side length of a regular decagon inscribed in a circle with radius r
    is equal to r * (√5 - 1) / 2, given that the central angle is 36° -/
theorem regular_decagon_side_length_formula (r : ℝ) (h : r > 0) :
  let x := regular_decagon_side_length r
  let central_angle := 36 * π / 180
  2 * r * Real.sin (central_angle / 2) = x := by
  sorry

#check regular_decagon_side_length
#check regular_decagon_side_length_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_side_length_formula_l358_35834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_domain_is_real_f_symmetric_f_period_one_f_not_increasing_l358_35812

-- Definition of the nearest integer function
noncomputable def nearest_integer (x : ℝ) : ℤ :=
  if x - (⌊x⌋ : ℝ) ≤ 1/2 then ⌊x⌋ else ⌈x⌉

-- Definition of the function f(x)
noncomputable def f (x : ℝ) : ℝ := |x - (nearest_integer x)|

-- Theorem stating the properties of f(x)
theorem f_properties :
  (∀ x, f x ∈ Set.Icc 0 (1/2)) ∧
  (∀ k : ℤ, ∀ x, f (k - x) = f (-x)) ∧
  (∀ x, f (x + 1) = f x) :=
by
  sorry

-- Additional properties
theorem f_domain_is_real : ∀ x : ℝ, ∃ y, y = f x :=
by
  sorry

theorem f_symmetric : ∀ k : ℤ, ∀ x : ℝ, f (k/2 + x) = f (k/2 - x) :=
by
  sorry

theorem f_period_one : ∀ x : ℝ, f (x + 1) = f x :=
by
  sorry

theorem f_not_increasing : ¬(∀ x y : ℝ, x ∈ Set.Icc (-1/2) (1/2) → 
                             y ∈ Set.Icc (-1/2) (1/2) → x < y → f x < f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_domain_is_real_f_symmetric_f_period_one_f_not_increasing_l358_35812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l358_35823

-- Define the fractions
noncomputable def f1 (x : ℝ) : ℝ := (2*x^3 + x^2 - 8*x + 5) / (7*x^2 - 12*x + 5)
noncomputable def f2 (x : ℝ) : ℝ := (2*x^3 + 3*x^2 + x) / (x^3 - x^2 - 2*x)

-- Define the simplified fractions
noncomputable def s1 (x : ℝ) : ℝ := (2*x^2 + 3*x - 5) / (7*x - 5)
noncomputable def s2 (x : ℝ) : ℝ := (2*x + 1) / (x - 2)

-- Theorem stating the equality of the original and simplified fractions
theorem fraction_simplification (x : ℝ) 
  (h1 : 7*x^2 - 12*x + 5 ≠ 0) 
  (h2 : x^3 - x^2 - 2*x ≠ 0) 
  (h3 : 7*x - 5 ≠ 0) 
  (h4 : x - 2 ≠ 0) : 
  f1 x = s1 x ∧ f2 x = s2 x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l358_35823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_after_two_years_proof_l358_35872

/-- The amount after n years given an initial amount and annual rate of increase -/
def amount_after_n_years (initial_amount : ℝ) (rate : ℝ) (n : ℕ) : ℝ :=
  initial_amount * (1 + rate) ^ n

/-- Rounds a real number to two decimal places -/
noncomputable def round_to_two_decimals (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem amount_after_two_years_proof :
  let initial_amount : ℝ := 65000
  let rate : ℝ := 1/8
  let years : ℕ := 2
  let final_amount := amount_after_n_years initial_amount rate years
  round_to_two_decimals final_amount = 82265.63 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_after_two_years_proof_l358_35872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l358_35847

theorem quadratic_equation_solution (x : ℝ) :
  2 * x^2 - 8 * x + 3 = 0 ↔ x = 2 + (Real.sqrt 10) / 2 ∨ x = 2 - (Real.sqrt 10) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l358_35847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_treasure_signs_l358_35811

/-- Represents the number of palm trees with signs -/
def total_trees : ℕ := 30

/-- Represents the number of signs saying "Exactly under 15 signs a treasure is buried" -/
def signs_15 : ℕ := 15

/-- Represents the number of signs saying "Exactly under 8 signs a treasure is buried" -/
def signs_8 : ℕ := 8

/-- Represents the number of signs saying "Exactly under 4 signs a treasure is buried" -/
def signs_4 : ℕ := 4

/-- Represents the number of signs saying "Exactly under 3 signs a treasure is buried" -/
def signs_3 : ℕ := 3

/-- Represents that only signs under which there is no treasure are truthful -/
def truthful_signs (n : ℕ) : Prop := 
  n ≤ total_trees ∧ (n ≠ signs_15 ∧ n ≠ signs_8 ∧ n ≠ signs_4 ∧ n ≠ signs_3)

/-- The theorem stating that the minimum number of signs under which treasures can be buried is 15 -/
theorem min_treasure_signs : ∃ (n : ℕ), n = 15 ∧ ∀ (m : ℕ), m < n → ¬(truthful_signs m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_treasure_signs_l358_35811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_r_l358_35837

-- Define the polynomials f and g
def f (r a : ℝ) : ℝ → ℝ := λ x => (x - r) * (x - (r + 6)) * (x - a)
def g (r b : ℝ) : ℝ → ℝ := λ x => (x - (r + 2)) * (x - (r + 8)) * (x - b)

-- State the theorem
theorem find_r (r a b : ℝ) : 
  (∃ c d e, ∀ x, f r a x = x^3 + c * x^2 + d * x + e) →  -- f is monic cubic
  (∃ p q s, ∀ x, g r b x = x^3 + p * x^2 + q * x + s) →  -- g is monic cubic
  f r a (r + 5) = g r b (r + 5) →               -- f(r+5) = g(r+5)
  r = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_r_l358_35837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l358_35866

theorem roots_of_equation : 
  ∀ x : ℝ, (20 / (x^2 - 9) - 3 / (x + 3) = 1) ↔ (x = -8 ∨ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l358_35866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l358_35881

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof steps would go here
  sorry

#check g_is_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l358_35881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_modified_hemisphere_area_l358_35843

/-- The total surface area of a modified hemisphere with a cylindrical hole -/
noncomputable def modified_hemisphere_surface_area (R r : ℝ) : ℝ :=
  2 * Real.pi * R^2 - 2 * Real.pi * r * R + Real.pi * (R^2 - r^2)

/-- Theorem stating the surface area of the specific modified hemisphere -/
theorem specific_modified_hemisphere_area :
  modified_hemisphere_surface_area 15 3 = 576 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_modified_hemisphere_area_l358_35843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_in_unit_cube_l358_35877

/-- The maximum radius of a sphere inside a unit cube, tangent to a diagonal of the cube -/
noncomputable def max_sphere_radius : ℝ := (4 - Real.sqrt 6) / 5

/-- A cube with side length 1 -/
structure UnitCube where
  side_length : ℝ
  side_length_eq_one : side_length = 1

/-- A sphere inside a unit cube, tangent to a diagonal -/
structure SphereInCube (cube : UnitCube) where
  center : Fin 3 → ℝ
  radius : ℝ
  inside_cube : radius ≤ cube.side_length / 2
  tangent_to_diagonal : ∃ (p q : Fin 3 → ℝ), 
    p 0 = 0 ∧ p 1 = 1 ∧ p 2 = 1 ∧
    q 0 = 1 ∧ q 1 = 0 ∧ q 2 = 0 ∧
    (center 0 - p 0) * (q 0 - p 0) + 
    (center 1 - p 1) * (q 1 - p 1) + 
    (center 2 - p 2) * (q 2 - p 2) = 
    radius * Real.sqrt ((q 0 - p 0)^2 + (q 1 - p 1)^2 + (q 2 - p 2)^2)

/-- The theorem stating that the maximum radius of a sphere inside a unit cube, 
    tangent to a diagonal, is equal to max_sphere_radius -/
theorem max_sphere_radius_in_unit_cube (cube : UnitCube) :
  ∀ (s : SphereInCube cube), s.radius ≤ max_sphere_radius := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_in_unit_cube_l358_35877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_20_at_30_div_7_l358_35852

-- Define the functions h and f
noncomputable def f (x : ℝ) : ℝ := 30 / (x + 2)
noncomputable def h (x : ℝ) : ℝ := 4 * (f⁻¹ x)

-- State the theorem
theorem h_equals_20_at_30_div_7 :
  ∃ x : ℝ, h x = 20 ∧ x = 30 / 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_20_at_30_div_7_l358_35852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_2x_l358_35896

theorem definite_integral_exp_plus_2x : ∫ x in (Set.Icc 0 1), (Real.exp x + 2 * x) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_2x_l358_35896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_fruit_is_raspberry_l358_35814

/-- Represents the types of fruit in the bowl -/
inductive Fruit
  | Strawberry
  | Raspberry

/-- Represents the possible actions Emma can take -/
inductive EmmaAction
  | TwoRaspberries
  | TwoStrawberries
  | OneEach

/-- Represents the state of the bowl -/
structure BowlState where
  strawberries : Nat
  raspberries : Nat

/-- Performs an action on the bowl state -/
def performAction (state : BowlState) (action : EmmaAction) : BowlState :=
  match action with
  | EmmaAction.TwoRaspberries => { strawberries := state.strawberries + 1, raspberries := state.raspberries - 2 }
  | EmmaAction.TwoStrawberries => { strawberries := state.strawberries - 1, raspberries := state.raspberries }
  | EmmaAction.OneEach => { strawberries := state.strawberries - 1, raspberries := state.raspberries }

/-- Theorem: The last fruit in the bowl is always a raspberry -/
theorem last_fruit_is_raspberry (initialStrawberries initialRaspberries : Nat)
  (h_odd : Odd initialRaspberries)
  (h_positive : initialRaspberries > 0)
  (actions : List EmmaAction) :
  let finalState := actions.foldl performAction { strawberries := initialStrawberries, raspberries := initialRaspberries }
  (finalState.strawberries + finalState.raspberries = 1) → finalState.raspberries = 1 :=
by sorry

#check last_fruit_is_raspberry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_fruit_is_raspberry_l358_35814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l358_35876

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) (x : ℝ) : Prop :=
  t.C = Real.pi/4 ∧ t.c = Real.sqrt 2 ∧ t.a = x

-- Define the existence of two distinct triangles
def twoDistinctTriangles (x : ℝ) : Prop :=
  ∃ t1 t2 : Triangle, t1 ≠ t2 ∧ satisfiesConditions t1 x ∧ satisfiesConditions t2 x

-- Theorem statement
theorem range_of_x :
  ∀ x : ℝ, (twoDistinctTriangles x) ↔ (Real.sqrt 2 < x ∧ x < 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l358_35876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formulas_l358_35846

theorem sequence_formulas (θ : ℝ) (a b : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → a n ≠ 0 ∧ b n ≠ 0) →
  (∀ n : ℕ, n > 1 → a n = a (n-1) * Real.cos θ - b (n-1) * Real.sin θ) →
  (∀ n : ℕ, n > 1 → b n = a (n-1) * Real.sin θ + b (n-1) * Real.cos θ) →
  a 1 = 1 →
  b 1 = Real.tan θ →
  (∀ n : ℕ, n ≥ 1 → a n = 1 / Real.cos θ * Real.cos (n * θ) ∧ 
                     b n = 1 / Real.cos θ * Real.sin (n * θ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formulas_l358_35846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_example_theorem_l358_35819

theorem example_theorem (n : ℕ) : n + 0 = n := by
  rw [add_zero]

#check example_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_example_theorem_l358_35819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l358_35820

noncomputable section

open Real

def angle (A B C : ℝ × ℝ) : ℝ :=
  arccos (((A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2)) /
    (sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)))

theorem right_triangle_side_length 
  (A B C D : ℝ × ℝ) 
  (right_angle : (A.1 - C.1) * (A.2 - D.2) = (A.2 - C.2) * (A.1 - D.1))
  (angle_ACD : cos (angle A C D) = cos (50 * π / 180))
  (AD_length : sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 5 * sqrt 3) :
  sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 5 * sqrt (3 * (1 - cos (50 * π / 180))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l358_35820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l358_35865

theorem trigonometric_simplification (α : ℝ) 
  (h : 2 * Real.cos (π / 6 - 2 * α) + Real.sqrt 3 * Real.cos (2 * α - 3 * π) ≠ 0) : 
  (Real.sin (2 * α - 3 * π) + 2 * Real.cos (7 * π / 6 + 2 * α)) / 
  (2 * Real.cos (π / 6 - 2 * α) + Real.sqrt 3 * Real.cos (2 * α - 3 * π)) = 
  -Real.sqrt 3 * (Real.cos (2 * α) / Real.sin (2 * α)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l358_35865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_theorem_l358_35854

-- Define the polynomials
def P (a b : ℝ) (x : ℝ) : ℝ := x^3 - x^2 - a*x - b
def Q (a b : ℝ) (x : ℂ) : ℂ := x^3 - x^2 + b*x + a

-- State the theorem
theorem cubic_root_theorem (a b : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ 
    P a b x₁ = 0 ∧ P a b x₂ = 0 ∧ P a b x₃ = 0) →
  (∃ y : ℝ, y > 0 ∧ Q a b y = 0) ∧
  (∃ z w : ℂ, z ≠ w ∧ z.im ≠ 0 ∧ w.im ≠ 0 ∧ Q a b z = 0 ∧ Q a b w = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_theorem_l358_35854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_sum_l358_35809

/-- Parabola type -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2*a*x

/-- Line type -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : ℝ → ℝ → Prop := fun x y => a*x + b*y + c = 0

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x y : ℝ) (l : Line) : ℝ :=
  (abs (l.a * x + l.b * y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

theorem parabola_min_distance_sum (C : Parabola) (l2 : Line) :
  C.a = 4 →
  l2.a = 3 ∧ l2.b = -4 ∧ l2.c = 24 →
  (∀ x y : ℝ, C.equation x y →
    ∃ d : ℝ, d ≥ 6 ∧
      d = distance_point_to_line x y {a := 1, b := 0, c := 2} +
          distance_point_to_line x y l2) ∧
  (∃ x y : ℝ, C.equation x y ∧
    distance_point_to_line x y {a := 1, b := 0, c := 2} +
    distance_point_to_line x y l2 = 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_sum_l358_35809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_areas_l358_35818

/-- Properties of a right cylinder -/
structure RightCylinder where
  height : ℝ
  radius : ℝ

/-- Calculate the lateral surface area of a right cylinder -/
noncomputable def lateralSurfaceArea (c : RightCylinder) : ℝ :=
  2 * Real.pi * c.radius * c.height

/-- Calculate the total surface area of a right cylinder -/
noncomputable def totalSurfaceArea (c : RightCylinder) : ℝ :=
  2 * Real.pi * c.radius * c.height + 2 * Real.pi * c.radius^2

/-- Theorem: Lateral and total surface areas of a specific cylinder -/
theorem cylinder_surface_areas :
  let c : RightCylinder := ⟨10, 3⟩
  (lateralSurfaceArea c = 60 * Real.pi) ∧
  (totalSurfaceArea c = 78 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_areas_l358_35818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l358_35867

noncomputable def f (x : Real) := Real.sqrt 3 * Real.sin x * Real.cos x + Real.sin x ^ 2

theorem triangle_cosine_theorem (A B C : Real) (a b c : Real) (D : Real) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  f A = 3 / 2 →
  (∃ (AD BD : Real), AD = Real.sqrt 2 * BD ∧ AD = 2) →
  Real.cos c = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l358_35867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_washing_theorem_l358_35815

/-- Represents the dimensions of a bar of soap -/
structure SoapDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a bar of soap given its dimensions -/
noncomputable def soapVolume (d : SoapDimensions) : ℝ := d.length * d.width * d.height

/-- Represents the state of a bar of soap after some time of washing -/
structure SoapState where
  dimensions : SoapDimensions
  washTime : ℝ

/-- Defines how the soap dimensions change after 7 hours of washing -/
noncomputable def washFor7Hours (initial : SoapDimensions) : SoapDimensions :=
  { length := initial.length / 2
  , width := initial.width / 2
  , height := initial.height / 2 }

theorem soap_washing_theorem (initial : SoapDimensions) :
  let initialVolume := soapVolume initial
  let afterWashingVolume := soapVolume (washFor7Hours initial)
  let remainingWashTime := initialVolume / 7 / (initialVolume / 8)
  remainingWashTime = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_washing_theorem_l358_35815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_red_given_upper_red_is_two_thirds_l358_35888

/-- Represents a card with two sides -/
inductive Card
| RR  -- Red on both sides
| RB  -- Red on one side, Blue on the other

/-- The probability of choosing each card -/
noncomputable def card_prob : Card → ℝ
| Card.RR => 1/2
| Card.RB => 1/2

/-- The probability of seeing a red side on a given card -/
noncomputable def red_side_prob : Card → ℝ
| Card.RR => 1
| Card.RB => 1/2

/-- The total probability of seeing a red side -/
noncomputable def total_red_prob : ℝ := 
  card_prob Card.RR * red_side_prob Card.RR + card_prob Card.RB * red_side_prob Card.RB

/-- The probability that both sides are red given that the upper side is red -/
noncomputable def prob_both_red_given_upper_red : ℝ := 
  (card_prob Card.RR * red_side_prob Card.RR) / total_red_prob

theorem prob_both_red_given_upper_red_is_two_thirds : 
  prob_both_red_given_upper_red = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_red_given_upper_red_is_two_thirds_l358_35888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_foldable_position_l358_35841

/-- Represents a position where an additional triangle can be attached --/
inductive AttachmentPosition
| Left
| MiddleLeft
| Center
| MiddleRight
| Right

/-- Represents a figure composed of equilateral triangles --/
structure TriangleFigure where
  base_triangles : Fin 4 → Unit  -- Represents the 4 base triangles
  attachment : Option AttachmentPosition

/-- Predicate to check if a figure can be folded into a triangular pyramid with one face missing --/
def can_fold_to_pyramid (figure : TriangleFigure) : Prop :=
  sorry

/-- The main theorem to be proved --/
theorem unique_foldable_position :
  ∃! pos : AttachmentPosition, 
    can_fold_to_pyramid ⟨λ _ => (), some pos⟩ :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_foldable_position_l358_35841

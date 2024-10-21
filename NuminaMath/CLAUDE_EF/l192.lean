import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sauroposeidon_model_height_l192_19205

/-- The height of a scale model dinosaur -/
noncomputable def model_height (actual_height : ℝ) (scale : ℝ) : ℝ :=
  actual_height / scale

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem sauroposeidon_model_height : 
  round_to_nearest (model_height 60 30) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sauroposeidon_model_height_l192_19205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_problem_l192_19294

theorem lcm_problem : 
  (Nat.lcm (Nat.lcm (12^2) (16^3)) (Nat.lcm (18^2) (24^3))) % 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_problem_l192_19294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bf_de_equal_half_yard_l192_19299

/-- A rectangle with a rhombus inscribed in it -/
structure RhombusInRectangle where
  width : ℝ  -- Width of the rectangle
  length : ℝ  -- Length of the rectangle
  rhombus_perimeter : ℝ  -- Perimeter of the inscribed rhombus
  width_positive : 0 < width
  length_positive : 0 < length
  rhombus_perimeter_positive : 0 < rhombus_perimeter

/-- The length of BF in the RhombusInRectangle configuration -/
noncomputable def bf_length (r : RhombusInRectangle) : ℝ :=
  r.rhombus_perimeter / 4 - r.width

/-- The length of DE in the RhombusInRectangle configuration -/
noncomputable def de_length (r : RhombusInRectangle) : ℝ :=
  r.rhombus_perimeter / 4 - r.width

/-- Theorem stating that BF and DE are equal and have length 0.5 yards -/
theorem bf_de_equal_half_yard (r : RhombusInRectangle) 
    (h_width : r.width = 20)
    (h_length : r.length = 25)
    (h_perimeter : r.rhombus_perimeter = 82) :
    bf_length r = de_length r ∧ bf_length r = 0.5 := by
  sorry

#check bf_de_equal_half_yard

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bf_de_equal_half_yard_l192_19299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_on_parabola_height_l192_19297

/-- A right triangle with vertices on a parabola and hypotenuse parallel to x-axis has height 1 -/
theorem right_triangle_on_parabola_height (A B C : ℝ × ℝ) : 
  (∀ P ∈ ({A, B, C} : Set (ℝ × ℝ)), P.2 = P.1^2) →  -- All vertices lie on y = x^2
  (A.2 = B.2) →  -- Hypotenuse AB is parallel to x-axis
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 →  -- ABC is a right triangle
  ∃ D : ℝ × ℝ, D.1 = C.1 ∧ D.2 = A.2 ∧ |C.2 - D.2| = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_on_parabola_height_l192_19297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l192_19244

/-- The circle in polar coordinates -/
def circle_eq (ρ θ : ℝ) : Prop := ρ = -4 * Real.sin θ

/-- The line in polar coordinates -/
def line_eq (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi/4) = Real.sqrt 2

/-- The shortest distance from the circle to the line -/
noncomputable def shortest_distance : ℝ := 2 * Real.sqrt 2 - 2

theorem circle_line_distance :
  ∀ (ρ θ : ℝ), circle_eq ρ θ →
  (∃ (ρ' θ' : ℝ), line_eq ρ' θ' ∧
    ∀ (ρ'' θ'' : ℝ), line_eq ρ'' θ'' →
      (ρ - ρ' * Real.cos (θ' - θ))^2 + (ρ * Real.sin θ - ρ' * Real.sin θ')^2 ≤
      (ρ - ρ'' * Real.cos (θ'' - θ))^2 + (ρ * Real.sin θ - ρ'' * Real.sin θ'')^2) →
  Real.sqrt ((ρ - ρ' * Real.cos (θ' - θ))^2 + (ρ * Real.sin θ - ρ' * Real.sin θ')^2) = shortest_distance :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l192_19244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_sum_diff_permutations_l192_19275

open BigOperators Finset

def permutations (n : ℕ) : Finset (Equiv.Perm (Fin n)) :=
  Finset.filter (fun _ => true) Finset.univ

def sum_diff (σ : Equiv.Perm (Fin 5)) : ℚ :=
  |σ 0 - σ 1| + |σ 2 - σ 3|

theorem average_sum_diff_permutations :
  (∑ σ in permutations 5, sum_diff σ) / (permutations 5).card = 2 := by
  sorry

#eval (∑ σ in permutations 5, sum_diff σ) / (permutations 5).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_sum_diff_permutations_l192_19275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_eq_45_l192_19226

/-- The smallest positive angle (in degrees) that satisfies the given equation is 45 degrees. -/
theorem smallest_angle_eq_45 : 
  ∃ θ : ℝ, θ > 0 ∧ θ < 360 ∧
    Real.cos (θ * π / 180) = 
      Real.sin (45 * π / 180) + Real.cos (48 * π / 180) - 
      Real.sin (18 * π / 180) - Real.cos (12 * π / 180) ∧
    (∀ φ : ℝ, φ > 0 ∧ φ < θ → 
      Real.cos (φ * π / 180) ≠ 
        Real.sin (45 * π / 180) + Real.cos (48 * π / 180) - 
        Real.sin (18 * π / 180) - Real.cos (12 * π / 180)) ∧
    θ = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_eq_45_l192_19226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vehicles_l192_19214

-- Define the speeds of the truck and car in km/h
noncomputable def truck_speed : ℝ := 65
noncomputable def car_speed : ℝ := 85

-- Define the time elapsed in hours
noncomputable def time_elapsed : ℝ := 3 / 60

-- Theorem to prove
theorem distance_between_vehicles : 
  let truck_distance := truck_speed * time_elapsed
  let car_distance := car_speed * time_elapsed
  car_distance - truck_distance = 1 := by
  -- Unfold the definitions
  unfold truck_speed car_speed time_elapsed
  -- Simplify the expressions
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vehicles_l192_19214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_sqrt_two_l192_19233

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 1) * x^m

-- State the theorem
theorem power_function_increasing_sqrt_two (m : ℝ) :
  (∀ x > 0, ∃ k > 0, f m x = k * x^m) →  -- f is a power function
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) →  -- f is increasing on (0, +∞)
  m = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_sqrt_two_l192_19233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_division_theorem_l192_19236

theorem group_division_theorem :
  ∀ (A B : Finset ℕ),
    A ∪ B = Finset.range 12 →
    A ∩ B = ∅ →
    A.card + B.card = 12 →
    (A.sum (λ x => (x : ℚ))) / A.card = (B.sum (λ x => (x : ℚ))) / B.card + 2 →
    A.card = 3 ∨ A.card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_division_theorem_l192_19236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_time_for_given_tank_l192_19258

/-- Represents a tank with inlet and outlet pipes and a leak -/
structure Tank where
  inlet_time : ℝ  -- Time to fill the tank using only the inlet pipe
  outlet_time : ℝ  -- Time to empty the tank using only the outlet pipe
  fill_time_with_leak : ℝ  -- Time to fill the tank with both pipes open and leak present

/-- Calculates the time for a full tank to completely leak out -/
noncomputable def leak_time (t : Tank) : ℝ :=
  (t.inlet_time * t.outlet_time * t.fill_time_with_leak) /
  (t.inlet_time * t.outlet_time - t.inlet_time * t.fill_time_with_leak - t.outlet_time * t.fill_time_with_leak)

theorem leak_time_for_given_tank :
  let t : Tank := { inlet_time := 3, outlet_time := 4, fill_time_with_leak := 14 }
  leak_time t = 84 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_time_for_given_tank_l192_19258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_largest_least_l192_19283

def digits : List Nat := [7, 3, 1, 4]

def largest_number (ds : List Nat) : Nat :=
  (ds.toArray.qsort (· > ·)).toList.foldl (fun acc d => acc * 10 + d) 0

def least_number (ds : List Nat) : Nat :=
  (ds.toArray.qsort (· < ·)).toList.foldl (fun acc d => acc * 10 + d) 0

theorem difference_largest_least :
  largest_number digits - least_number digits = 6084 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_largest_least_l192_19283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_has_odd_point_trig_odd_point_iff_l192_19216

-- Definition of an "odd point"
def is_odd_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f (-x₀) = -f x₀

-- Statement 1
theorem cubic_has_odd_point (a b c : ℝ) :
  ∃ x₀ : ℝ, is_odd_point (fun x ↦ a * x^3 + b * x^2 + c * x - 2 * b) x₀ :=
sorry

-- Statement 2
theorem trig_odd_point_iff (m : ℝ) :
  (∃ x₀ : ℝ, is_odd_point (fun x ↦ Real.sqrt 2 * Real.sin x - Real.cos x ^ 2 + m) x₀) ↔
  (0 ≤ m ∧ m ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_has_odd_point_trig_odd_point_iff_l192_19216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hamiltonian_cycle_for_all_grids_l192_19228

/-- Represents a grid point in a rectangular grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a rectangular grid -/
structure RectangularGrid where
  m : ℕ
  n : ℕ

/-- A path on the grid is a list of grid points -/
def GridPath := List GridPoint

/-- Checks if a path is closed (starts and ends at the same point) -/
def isClosed (path : GridPath) : Prop :=
  path.head? = path.getLast?

/-- Checks if a path visits all points in a grid exactly once -/
def visitsAllPointsOnce (grid : RectangularGrid) (path : GridPath) : Prop :=
  ∀ x y, x ≤ grid.m ∧ y ≤ grid.n →
    ∃! i, path.get? i = some ⟨x, y⟩

/-- The main theorem: It's impossible to construct a closed path that visits
    all points exactly once for all rectangular grids -/
theorem no_hamiltonian_cycle_for_all_grids :
  ¬ ∀ (grid : RectangularGrid), ∃ (path : GridPath),
    isClosed path ∧ visitsAllPointsOnce grid path :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hamiltonian_cycle_for_all_grids_l192_19228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_is_16_l192_19235

/-- Represents a cylinder with a rectangular side surface. -/
structure Cylinder where
  base_perimeter : ℝ
  diagonal : ℝ

/-- Calculates the height of a cylinder given its base perimeter and diagonal. -/
noncomputable def cylinder_height (c : Cylinder) : ℝ :=
  Real.sqrt (c.diagonal ^ 2 - c.base_perimeter ^ 2)

theorem cylinder_height_is_16 (c : Cylinder) 
  (h_perimeter : c.base_perimeter = 12)
  (h_diagonal : c.diagonal = 20) : 
  cylinder_height c = 16 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_is_16_l192_19235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l192_19257

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + Real.sqrt (1 - x^2)) - Real.sqrt (1 + x)

theorem f_properties :
  (∀ x, f x ≠ 0 → x ∈ Set.Icc (-1 : ℝ) 1) ∧
  (Set.range f = Set.Icc (1 - Real.sqrt 2) 1) ∧
  (∀ x y, x ∈ Set.Icc (-1 : ℝ) 1 → y ∈ Set.Icc (-1 : ℝ) 1 → x < y → f y < f x) ∧
  (Set.Ioc (-1 : ℝ) ((3 * Real.sqrt 2 - 1 - Real.sqrt (13 + 6 * Real.sqrt 2)) / 8) =
   {x | x ∈ Set.Icc (-1 : ℝ) 1 ∧ f x > 1/2}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l192_19257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_spheres_radii_relation_l192_19239

/-- The equation of a sphere with center c and radius r, containing point p. -/
def sphere_equation (p c : ℝ × ℝ × ℝ) (r : ℝ) : Prop :=
  (p.1 - c.1)^2 + (p.2.1 - c.2.1)^2 + (p.2.2 - c.2.2)^2 = r^2

/-- A point is a tangent point for three spheres if it lies on the surface of all three spheres. -/
def is_tangent_point (p : ℝ × ℝ × ℝ) (r₁ r₂ r₃ : ℝ) : Prop :=
  ∃ (c₁ c₂ c₃ : ℝ × ℝ × ℝ), 
    sphere_equation p c₁ r₁ ∧ 
    sphere_equation p c₂ r₂ ∧ 
    sphere_equation p c₃ r₃

theorem tangent_spheres_radii_relation (r₁ r₂ r₃ : ℝ) 
  (h₁ : 0 < r₃) (h₂ : r₃ ≤ r₂) (h₃ : r₂ ≤ r₁) 
  (h₄ : ∃ (p : ℝ × ℝ × ℝ), is_tangent_point p r₁ r₂ r₃) :
  r₃ ≥ (r₁ * r₂) / (Real.sqrt r₁ + Real.sqrt r₂)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_spheres_radii_relation_l192_19239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l192_19248

def f : ℕ → ℤ
  | 0 => 60  -- Add a case for 0
  | 1 => 60
  | n + 2 => if n % 2 = 0 then f (n + 1) + 3 else f (n + 1) - 2

def is_valid (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≥ n → f m ≥ 63

theorem smallest_valid_n :
  (∃ n, is_valid n) ∧ (∀ k, k < 11 → ¬is_valid k) ∧ is_valid 11 := by
  sorry

#eval f 11  -- This line is added to check the value of f 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l192_19248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_problem_l192_19288

theorem cube_volume_problem (reference_volume : ℝ) (new_cube_side : ℝ) : 
  reference_volume = 8 →
  (6 * new_cube_side^2) = 4 * (6 * reference_volume^(2/3)) →
  new_cube_side^3 = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_problem_l192_19288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_calculation_semicircle_perimeter_formula_l192_19204

/-- The perimeter of a semicircle in centimeters -/
noncomputable def semicircle_perimeter : ℝ := 144

/-- The radius of a semicircle in centimeters -/
noncomputable def semicircle_radius : ℝ := semicircle_perimeter / (Real.pi + 2)

theorem semicircle_radius_calculation :
  semicircle_radius = semicircle_perimeter / (Real.pi + 2) := by
  -- Unfold the definitions
  unfold semicircle_radius semicircle_perimeter
  -- The equation now holds by reflexivity
  rfl

theorem semicircle_perimeter_formula (r : ℝ) :
  semicircle_perimeter = r * Real.pi + 2 * r := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_calculation_semicircle_perimeter_formula_l192_19204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l192_19242

theorem solve_exponential_equation (x : ℝ) :
  (3 : ℝ)^(2*x) * (3 : ℝ)^(2*x) * (3 : ℝ)^(2*x) * (3 : ℝ)^(2*x) = (27 : ℝ)^4 → x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l192_19242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_sum_upper_bound_l192_19243

theorem sine_cosine_sum_upper_bound (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  Real.sin α + Real.cos α ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_sum_upper_bound_l192_19243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_with_more_than_five_factors_eq_four_l192_19292

/-- The number of positive integer factors of 3080 that have more than 5 factors -/
noncomputable def factors_with_more_than_five_factors : ℕ :=
  (Finset.filter (fun d => (Nat.divisors d).card > 5) (Nat.divisors 3080)).card

/-- Theorem stating that the number of positive integer factors of 3080 
    that have more than 5 factors is equal to 4 -/
theorem factors_with_more_than_five_factors_eq_four :
  factors_with_more_than_five_factors = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_with_more_than_five_factors_eq_four_l192_19292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_12_pretty_iff_decidable_sum_12_pretty_div_12_l192_19267

def is_12_pretty (n : ℕ) : Prop :=
  (n > 0) ∧ 
  (Finset.card (Nat.divisors n) = 12) ∧ 
  (n % 12 = 0)

-- Define a decidable predicate for is_12_pretty
def is_12_pretty_decidable (n : ℕ) : Bool :=
  (n > 0) && 
  (Finset.card (Nat.divisors n) = 12) && 
  (n % 12 = 0)

-- Prove that the decidable predicate is equivalent to the original predicate
theorem is_12_pretty_iff_decidable (n : ℕ) :
  is_12_pretty n ↔ is_12_pretty_decidable n = true := by sorry

def sum_12_pretty : ℕ := 
  Finset.sum (Finset.filter (λ n => is_12_pretty_decidable n) (Finset.range 1000)) id

theorem sum_12_pretty_div_12 : 
  (sum_12_pretty : ℚ) / 12 = 55.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_12_pretty_iff_decidable_sum_12_pretty_div_12_l192_19267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friction_coefficient_spring_blocks_l192_19266

/-- Given two identical blocks on a horizontal plane, connected by a thread with a compressed spring between them, this theorem proves the relationship between the coefficient of friction and other parameters when the thread is cut and the blocks move apart. -/
theorem friction_coefficient_spring_blocks (m : ℝ) (g : ℝ) (PE : ℝ) (dL : ℝ) (μ : ℝ) :
  m > 0 → g > 0 → PE > 0 → dL > 0 →
  (μ = PE / (m * g * dL)) ↔ 
  (∃ (spring_force : ℝ), 
    spring_force > 0 ∧ 
    PE = spring_force * dL ∧
    spring_force = 2 * μ * m * g) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friction_coefficient_spring_blocks_l192_19266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_payment_days_l192_19249

/-- The number of days the sum can pay all workers together -/
noncomputable def combined_days (S A B C : ℝ) : ℝ :=
  S / (A + B + C)

/-- Theorem stating the combined days of payment -/
theorem combined_payment_days
  (S A B C : ℝ)
  (hA : S = 21 * A)
  (hB : S = 28 * B)
  (hC : S = 35 * C)
  (hpos : A > 0 ∧ B > 0 ∧ C > 0) :
  ⌊combined_days S A B C⌋ = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_payment_days_l192_19249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l192_19208

-- Define set M
def M : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

-- Define set N
def N : Set ℝ := {x | Real.exp (x * Real.log 2) > 1/2}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l192_19208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_l192_19282

/-- The function f(x) = x^2 + 2x + a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

/-- The function g(x) = 1 / e^x -/
noncomputable def g (x : ℝ) : ℝ := 1 / Real.exp x

/-- The interval [1/2, 2] -/
def I : Set ℝ := Set.Icc (1/2) 2

theorem max_value_of_a (a : ℝ) :
  (∀ x₁ ∈ I, ∃ x₂ ∈ I, f a x₁ ≤ g x₂) →
  a ≤ Real.sqrt (Real.exp 1) / Real.exp 1 - 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_l192_19282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_blocks_needed_l192_19241

-- Define the dimensions of the clay block
def block_length : ℝ := 8
def block_width : ℝ := 3
def block_height : ℝ := 2

-- Define the dimensions of the cylindrical sculpture
def cylinder_height : ℝ := 9
def cylinder_diameter : ℝ := 6

-- Calculate the volume of one clay block
def block_volume : ℝ := block_length * block_width * block_height

-- Calculate the volume of the cylindrical sculpture
noncomputable def cylinder_volume : ℝ := Real.pi * (cylinder_diameter / 2)^2 * cylinder_height

-- Define the function to calculate the number of whole blocks needed
noncomputable def blocks_needed : ℕ := Int.toNat (Int.ceil (cylinder_volume / block_volume))

-- The theorem to prove
theorem min_blocks_needed : blocks_needed = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_blocks_needed_l192_19241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l192_19256

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.log (x + Real.sqrt (4 + x^2))

-- Theorem statement
theorem g_neither_even_nor_odd :
  (∃ x : ℝ, g (-x) ≠ g x) ∧ (∃ x : ℝ, g (-x) ≠ -g x) := by
  -- We'll use x = 1 to prove both parts
  let x := 1

  -- Prove g(-x) ≠ g(x)
  have h1 : g (-x) ≠ g x := by
    -- We could provide a detailed proof here, but for now we'll use sorry
    sorry

  -- Prove g(-x) ≠ -g(x)
  have h2 : g (-x) ≠ -g x := by
    -- We could provide a detailed proof here, but for now we'll use sorry
    sorry

  -- Combine the two parts
  exact ⟨⟨x, h1⟩, ⟨x, h2⟩⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l192_19256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sums_iff_twelve_l192_19262

/-- The sum of the first n terms of an arithmetic sequence with first term a and common difference d -/
noncomputable def arithmetic_sum (a d n : ℝ) : ℝ := n / 2 * (2 * a + (n - 1) * d)

/-- The proposition that the sums of the first n terms of two specific arithmetic sequences are equal -/
def sums_equal (n : ℝ) : Prop :=
  arithmetic_sum 9 5 n = arithmetic_sum 20 3 n

theorem equal_sums_iff_twelve (n : ℝ) (hn : n ≠ 0) :
  sums_equal n ↔ n = 12 := by
  sorry

#check equal_sums_iff_twelve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sums_iff_twelve_l192_19262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_Q_less_than_threshold_l192_19254

-- Define the number of boxes
def num_boxes : ℕ := 3000

-- Define the probability function Q(n)
noncomputable def Q (n : ℕ) : ℝ := 1 / (n * (n^2 + 1))

-- Theorem statement
theorem smallest_n_for_Q_less_than_threshold :
  (Q 15 < 1 / (num_boxes : ℝ)) ∧
  (∀ m : ℕ, m < 15 → Q m ≥ 1 / (num_boxes : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_Q_less_than_threshold_l192_19254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_R2_l192_19298

/-- Rectangle with width, height, and area -/
structure Rectangle where
  width : ℝ
  height : ℝ
  area : ℝ

/-- The diagonal length of a rectangle -/
noncomputable def diagonal (r : Rectangle) : ℝ := Real.sqrt (r.width^2 + r.height^2)

/-- R1 is a rectangle with one side 3 and area 24 -/
def R1 : Rectangle := { width := 3, height := 8, area := 24 }

/-- R2 is similar to R1 with diagonal twice as long -/
def R2 : Rectangle :=
  { width := 6,
    height := 16,
    area := 96 }

theorem area_of_R2 :
  R2.width * R2.height = 96 ∧
  R2.width / R2.height = R1.width / R1.height ∧
  diagonal R2 = 2 * diagonal R1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_R2_l192_19298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_l192_19227

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot := v.1 * w.1 + v.2 * w.2
  let norm_squared := w.1^2 + w.2^2
  (dot / norm_squared * w.1, dot / norm_squared * w.2)

theorem projection_line : ∀ (x y : ℝ),
  projection (x, y) (3, -1) = (3/2, -1/2) → y = 3*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_l192_19227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l192_19272

noncomputable section

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given condition
def condition (t : Triangle) : Prop :=
  2 * t.b * Real.cos t.A = 2 * t.c - Real.sqrt 3 * t.a

-- Define the function f
def f (x : Real) : Real :=
  Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 / 4

-- Theorem statement
theorem triangle_property (t : Triangle) (h : condition t) :
  t.B = Real.pi/6 ∧ ∃ (max : Real), max = 1/2 ∧ ∀ x, f x ≤ max := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l192_19272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l192_19278

open Real

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

-- Define the line l
def line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + (3/2)

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem ellipse_and_line_theorem (a b : ℝ) :
  a > b ∧ b > 0 ∧
  ellipse a b 1 (Real.sqrt 6 / 3) ∧
  eccentricity a b = Real.sqrt 6 / 3 →
  (∃ k : ℝ,
    (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 3 + y^2 = 1) ∧
    (line k 0 (3/2)) ∧
    (∃ x₁ y₁ x₂ y₂ : ℝ,
      x₁ ≠ x₂ ∧
      ellipse a b x₁ y₁ ∧
      ellipse a b x₂ y₂ ∧
      line k x₁ y₁ ∧
      line k x₂ y₂ ∧
      distance x₁ y₁ 0 (-1) = distance x₂ y₂ 0 (-1)) ∧
    (k = Real.sqrt 6 / 3 ∨ k = -Real.sqrt 6 / 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l192_19278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_leap_years_l192_19219

def is_leap_year (year : ℕ) : Bool :=
  ((year % 4 = 0 && year % 100 ≠ 0) || year % 400 = 0)

def years : List ℕ := [1964, 1978, 1996, 2001, 2100]

theorem two_leap_years :
  (years.filter is_leap_year).length = 2 := by
  -- Proof goes here
  sorry

#eval years.filter is_leap_year
#eval (years.filter is_leap_year).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_leap_years_l192_19219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_one_l192_19261

noncomputable def f (x : ℝ) : ℝ := 
  if x < 1 then -x else (x - 1)^2

theorem f_inverse_of_one (a : ℝ) : f a = 1 ↔ a = -1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_one_l192_19261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rect_and_translate_specific_l192_19223

noncomputable def polar_to_rect_and_translate (r : ℝ) (θ : ℝ) (tx : ℝ) (ty : ℝ) : ℝ × ℝ :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x + tx, y + ty)

theorem polar_to_rect_and_translate_specific :
  polar_to_rect_and_translate 4 (Real.pi / 6) 2 (-1) = (2 * Real.sqrt 3 + 2, 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rect_and_translate_specific_l192_19223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l192_19201

/-- Definition of the complex number z in terms of a real number a -/
def z (a : ℝ) : ℂ := (a^2 - 1 : ℝ) + (a - 2 : ℝ) * Complex.I

/-- A complex number is purely imaginary if its real part is zero -/
def is_purely_imaginary (c : ℂ) : Prop := c.re = 0

theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → is_purely_imaginary (z a)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ is_purely_imaginary (z a)) :=
by
  sorry

#check a_eq_one_sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l192_19201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_OA_OB_l192_19277

/-- Parabola defined by y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- Origin -/
def O : ℝ × ℝ := (0, 0)

/-- A line passing through the focus F -/
structure LineThroughFocus where
  slope : ℝ ⊕ Unit
  intersectsParabola : ∃ (A B : ℝ × ℝ), A ≠ B ∧ A ∈ Parabola ∧ B ∈ Parabola ∧
    (slope = Sum.inr Unit.unit → A.1 = 1 ∧ B.1 = 1) ∧
    (∀ k, slope = Sum.inl k → A.2 = k * (A.1 - 1) ∧ B.2 = k * (B.1 - 1))

/-- Theorem stating that the dot product of OA and OB is always -3 -/
theorem dot_product_OA_OB (l : LineThroughFocus) :
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ A ∈ Parabola ∧ B ∈ Parabola ∧
  (l.slope = Sum.inr Unit.unit → A.1 = 1 ∧ B.1 = 1) ∧
  (∀ k, l.slope = Sum.inl k → A.2 = k * (A.1 - 1) ∧ B.2 = k * (B.1 - 1)) ∧
  (A.1 * B.1 + A.2 * B.2 = -3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_OA_OB_l192_19277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reforestation_and_subsidy_l192_19217

-- Constants
def total_sloping_land : ℝ := 91000000
def initial_reforestation : ℝ := 5150000
def annual_increase : ℝ := 0.12
def grain_subsidy : ℝ := 300
def grain_price : ℝ := 0.7
def annual_subsidy : ℝ := 20

-- Theorem statement
theorem reforestation_and_subsidy :
  ∃ (year : ℕ) (total_subsidy : ℝ),
    -- Part 1: Year when reforestation is essentially solved
    year = 2009 ∧
    initial_reforestation * (1 - (1 + annual_increase)^(year - 2002)) / (1 - (1 + annual_increase))
      ≥ 0.7 * total_sloping_land ∧
    -- Part 2: Total government spending on subsidies
    (total_subsidy ≥ 1.43 * 10^9 ∧ total_subsidy ≤ 1.45 * 10^9) ∧
    total_subsidy = (grain_subsidy * grain_price + annual_subsidy) *
      initial_reforestation * (1 - (1 + annual_increase)^(year - 2002 - 1)) / (1 - (1 + annual_increase)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reforestation_and_subsidy_l192_19217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_from_tan_l192_19271

theorem cos_value_from_tan (α : Real) :
  α ∈ Set.Ioo (π / 2) π →
  Real.tan α = -Real.sqrt 3 / 3 →
  Real.cos α = -Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_from_tan_l192_19271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l192_19238

def u : ℝ × ℝ × ℝ := (4, 2, -3)
def v : ℝ × ℝ × ℝ := (2, -4, 5)

theorem parallelogram_area (u v : ℝ × ℝ × ℝ) : 
  Real.sqrt (
    ((u.fst * v.snd.snd - u.snd.fst * v.snd.snd) ^ 2) +
    ((u.snd.fst * v.snd.snd - u.snd.snd * v.snd.fst) ^ 2) +
    ((u.snd.snd * v.fst - u.fst * v.snd.snd) ^ 2)
  ) = 10 * Real.sqrt 27 := by
  sorry

#check parallelogram_area u v

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l192_19238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l192_19221

-- Define the function g
noncomputable def g (d : ℝ) (x : ℝ) : ℝ := 1 / (3 * x + d)

-- Define the inverse function g⁻¹
noncomputable def g_inv (x : ℝ) : ℝ := (1 - 3 * x) / (3 * x)

-- Theorem statement
theorem inverse_function_condition (d : ℝ) : 
  (∀ x, g d x ≠ 0 → g_inv (g d x) = x) ↔ 
  (d = (3 + Real.sqrt 13) / 2 ∨ d = (3 - Real.sqrt 13) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l192_19221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_a_l192_19268

def is_multiple_of_three (n : Nat) : Prop :=
  n % 3 = 0

def digit_sum (a : Nat) : Nat :=
  2 + 6 + a + a + 2

def is_valid_a (a : Nat) : Bool :=
  a < 10 && (digit_sum a) % 3 = 0

theorem count_valid_a :
  (Finset.filter (fun a => is_valid_a a) (Finset.range 10)).card = 3 := by
  sorry

#eval (Finset.filter (fun a => is_valid_a a) (Finset.range 10)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_a_l192_19268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l192_19289

/-- An acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π
  opposite_sides : a > 0 ∧ b > 0 ∧ c > 0

/-- Vector m defined as (√3*a, c) -/
noncomputable def m (t : AcuteTriangle) : ℝ × ℝ :=
  (Real.sqrt 3 * t.a, t.c)

/-- Vector n defined as (sin A, cos C) -/
noncomputable def n (t : AcuteTriangle) : ℝ × ℝ :=
  (Real.sin t.A, Real.cos t.C)

/-- Theorem stating properties of the acute triangle given the conditions -/
theorem acute_triangle_properties (t : AcuteTriangle) 
  (h : m t = 3 • (n t)) : 
  t.C = π / 3 ∧ 
  (3 * Real.sqrt 3 + 3) / 2 < t.a + t.b + t.c ∧ 
  t.a + t.b + t.c ≤ 9 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l192_19289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_intersection_property_l192_19251

/-- Parabola with equation y² = 4x -/
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Ellipse with equation y²/2 + x² = 1 -/
def Ellipse (x y : ℝ) : Prop := y^2/2 + x^2 = 1

/-- Focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- Line passing through F and intersecting y-axis at N -/
def Line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

/-- N is the point where the line intersects the y-axis -/
def N (k : ℝ) : ℝ × ℝ := (0, -k)

/-- A is one of the intersection points of the line and the parabola -/
def A (k x₁ y₁ : ℝ) : Prop := Parabola x₁ y₁ ∧ Line k x₁ y₁

/-- B is the other intersection point of the line and the parabola -/
def B (k x₂ y₂ : ℝ) : Prop := Parabola x₂ y₂ ∧ Line k x₂ y₂

/-- lambda is defined by NA = lambda*AF -/
noncomputable def lambda (x₁ : ℝ) : ℝ := x₁ / (1 - x₁)

/-- mu is defined by NB = mu*BF -/
noncomputable def mu (x₂ : ℝ) : ℝ := x₂ / (1 - x₂)

theorem parabola_ellipse_intersection_property 
  (k x₁ y₁ x₂ y₂ : ℝ) 
  (hA : A k x₁ y₁) 
  (hB : B k x₂ y₂) 
  (hDiff : x₁ ≠ x₂) : 
  lambda x₁ + mu x₂ = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_intersection_property_l192_19251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_child_age_l192_19291

/-- Represents the ages of children in an arithmetic sequence -/
def ChildrenAges (n : ℕ) (avg : ℚ) (d : ℕ) : List ℚ :=
  let first := avg - (d * ((n - 1) / 2 : ℚ))
  List.range n |>.map (fun i => first + (i : ℚ) * d)

theorem oldest_child_age
  (n : ℕ) (avg : ℚ) (d : ℕ)
  (h_n : n = 7)
  (h_avg : avg = 8)
  (h_d : d = 3) :
  (ChildrenAges n avg d).getLast? = some 17 := by
  sorry

#eval ChildrenAges 7 8 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_child_age_l192_19291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_matching_indices_l192_19290

noncomputable def sequence_a : ℕ → ℝ
| 0 => 0.2023
| k+1 => if k % 2 = 0 then
           (0.202301 + (k+4 : ℝ) * 10^(-(k+4 : ℝ)))^(sequence_a k)
         else
           (0.202301 + (k+4 : ℝ) * 10^(-(k+4 : ℝ)) + 10^(-(k+7 : ℝ)))^(sequence_a k)

noncomputable def sequence_b : ℕ → ℝ := sorry

theorem sum_of_matching_indices : 
  (Finset.range 1011).sum (fun i => 2*i + 2) = 1023112 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_matching_indices_l192_19290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_fourth_quadrant_l192_19212

theorem tan_value_fourth_quadrant (α : ℝ) 
  (h1 : Real.sin α = -5/13) 
  (h2 : 3*π/2 < α ∧ α < 2*π) : 
  Real.tan α = -5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_fourth_quadrant_l192_19212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_FPQ_l192_19293

/-- The area of triangle FPQ given specific ellipse and parabola conditions -/
theorem area_triangle_FPQ (a b c p : ℝ) (F P Q : ℝ × ℝ) : 
  a > 0 → b > 0 → a > b →  -- ellipse conditions
  c = 2 * Real.sqrt 2 →  -- focal length
  c / a = Real.sqrt 6 / 3 →  -- eccentricity
  p > 0 →  -- parabola condition
  F.2 = b →  -- F is upper vertex of ellipse
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) →  -- P is on ellipse
  (Q.1^2 / a^2 + Q.2^2 / b^2 = 1) →  -- Q is on ellipse
  P ≠ F → Q ≠ F →  -- P and Q are different from F
  (P.1 - F.1) * (Q.1 - F.1) + (P.2 - F.2) * (Q.2 - F.2) = 0 →  -- FP ⋅ FQ = 0
  ∃ (k m : ℝ), P.2 = k * P.1 + m ∧ Q.2 = k * Q.1 + m ∧ 
    ∀ (x : ℝ), x^2 = 2 * p * (k * x + m) →  -- PQ is tangent to parabola
  (1/2) * b * abs (P.1 - Q.1) = (18 * Real.sqrt 3) / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_FPQ_l192_19293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l192_19280

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

theorem monotonic_increase_interval
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : -π / 2 < φ ∧ φ < π / 2)
  (h_symmetry : ∀ x, f ω φ (2 / 3 - x) = -f ω φ (x))
  (h_distance : ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧
    (∀ x, f ω φ x ≤ f ω φ x₁) ∧
    (∀ x, f ω φ x ≥ f ω φ x₂) ∧
    f ω φ x₁ - f ω φ x₂ = 4) :
  ∀ k : ℤ, StrictMonoOn (f ω φ) (Set.Ioo (4 * k * π - 2 * π / 3) (4 * k * π + 4 * π / 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l192_19280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_caught_sampling_theorem_l192_19202

/-- The percentage of customers who sample candy -/
noncomputable def total_sampling_percentage : ℝ := 25

/-- The percentage of sampling customers who are not caught -/
noncomputable def not_caught_percentage : ℝ := 12

/-- The percentage of customers caught sampling candy -/
noncomputable def caught_sampling_percentage : ℝ := total_sampling_percentage * (100 - not_caught_percentage) / 100

theorem caught_sampling_theorem : caught_sampling_percentage = 22 := by
  -- Unfold the definitions
  unfold caught_sampling_percentage total_sampling_percentage not_caught_percentage
  
  -- Simplify the arithmetic
  simp [mul_div_assoc, sub_mul, mul_comm, mul_assoc]
  
  -- The proof is completed by normalization of real number arithmetic
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_caught_sampling_theorem_l192_19202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_AB_distance_l192_19274

-- Define the curve C₁
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 + 2 * Real.sin α)

-- Define the transformation from M to P
def M_to_P (M : ℝ × ℝ) : ℝ × ℝ := (2 * M.1, 2 * M.2)

-- Define the curve C₂
noncomputable def C₂ (α : ℝ) : ℝ × ℝ := M_to_P (C₁ α)

-- Theorem for the equation of C₂
theorem C₂_equation (x y : ℝ) :
  (∃ α, C₂ α = (x, y)) ↔ x^2 + (y - 4)^2 = 16 := by sorry

-- Define points A and B
noncomputable def A : ℝ × ℝ := C₁ (Real.pi/3)
noncomputable def B : ℝ × ℝ := C₂ (Real.pi/3)

-- Theorem for the distance between A and B
theorem AB_distance : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_AB_distance_l192_19274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_properties_l192_19237

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2 + 2 * a * x - 1

-- Define the derivative of f
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x + 2 * a

-- State the theorem
theorem critical_points_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : f_deriv a x₁ = 0) 
  (h2 : f_deriv a x₂ = 0) 
  (h3 : x₁ ≠ x₂) :
  a > 0.5 * Real.exp 2 ∧ Real.sqrt (x₁ - 1) + Real.sqrt (x₂ - 1) > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_properties_l192_19237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_polar_equation_l192_19286

/-- The area of the circle represented by the polar equation r = 3 cos θ - 4 sin θ -/
theorem circle_area_from_polar_equation : 
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ : ℝ, (3 * Real.cos θ - 4 * Real.sin θ)^2 = 
      (3 * Real.cos θ - 4 * Real.sin θ * Real.cos θ - center.1)^2 + 
      (3 * Real.cos θ - 4 * Real.sin θ * Real.sin θ - center.2)^2) ∧
    Real.pi * radius^2 = 25 * Real.pi / 4 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_polar_equation_l192_19286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_problems_l192_19263

theorem math_problems :
  (∃ x : ℝ, x = Real.sqrt 2 - Real.sqrt 8 + Real.sqrt 32 ∧ x = 3 * Real.sqrt 2) ∧
  (∃ y : ℝ, y = (Real.sqrt 3 - Real.sqrt 2)^2 - (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) ∧ y = 4 - 2 * Real.sqrt 6) ∧
  (∃ z : ℝ, z = (Real.sqrt 6 - 2 * Real.sqrt 15) * Real.sqrt 3 - Real.sqrt 36 / Real.sqrt 2 ∧ z = -6 * Real.sqrt 5) ∧
  (∃ w : ℝ, w = (Real.sqrt 8 - Real.sqrt 6) / Real.sqrt 2 + 3 * Real.sqrt (1/3) ∧ w = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_problems_l192_19263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_10_common_terms_l192_19270

def arithmetic_progression (n : ℕ) : ℕ := 5 + 3 * n

def geometric_progression (k : ℕ) : ℕ := 20 * 2^k

def is_common_term (x : ℕ) : Bool :=
  (List.range 1000).any (fun n => 
    (List.range 1000).any (fun k => 
      arithmetic_progression n = x ∧ geometric_progression k = x))

def common_terms : List ℕ :=
  (List.range 1000).filter is_common_term

theorem sum_of_first_10_common_terms :
  (common_terms.take 10).sum = 6990500 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_10_common_terms_l192_19270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l192_19247

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem monotonic_increasing_interval :
  ∀ x ∈ Set.Icc (-Real.pi) 0,
    (∀ y ∈ Set.Icc (-Real.pi) 0, x < y → f x < f y) ↔ 
    x ∈ Set.Icc (-(3 * Real.pi / 4)) (-(Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l192_19247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_j_is_inverse_of_h_l192_19225

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := 6 - 3 * x

-- Define the proposed inverse function j
noncomputable def j (x : ℝ) : ℝ := (6 - x) / 3

-- Theorem stating that j is the inverse of h
theorem j_is_inverse_of_h : 
  (∀ x : ℝ, h (j x) = x) ∧ (∀ x : ℝ, j (h x) = x) := by
  constructor
  · intro x
    simp [h, j]
    ring
  · intro x
    simp [h, j]
    ring
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_j_is_inverse_of_h_l192_19225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_third_l192_19269

/-- The sum of the infinite series ∑(n=1 to ∞) (4n-3)/3^n is equal to 1/3 -/
theorem series_sum_equals_one_third :
  ∑' (n : ℕ), (4 * (n + 1) - 3 : ℝ) / (3 : ℝ) ^ (n + 1) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_third_l192_19269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_zero_functions_range_l192_19295

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1) + x - 2
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a + 8

-- Define what it means for functions to be "adjacent zero functions"
def adjacent_zero_functions (f g : ℝ → ℝ) : Prop :=
  ∃ (α β : ℝ), f α = 0 ∧ g β = 0 ∧ |α - β| ≤ 1

-- State the theorem
theorem adjacent_zero_functions_range :
  ∀ a : ℝ, adjacent_zero_functions f (g a) → a ∈ Set.Icc 4 (9/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_zero_functions_range_l192_19295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l192_19230

theorem diophantine_equation_solutions :
  let solution_count : ℕ := 
    (Finset.filter 
      (fun t : ℕ => 
        let x : ℕ := 380 - 3 * t
        let y : ℕ := 1 + 2 * t
        let z : ℕ := 37
        x > 0 ∧ y > 0 ∧ z > 0 ∧ 2 * x + 3 * y + z = 800)
      (Finset.range 381)).card
  solution_count = 127 := by
  sorry

#check diophantine_equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l192_19230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_zero_l192_19234

noncomputable def projection_matrix (v : Fin 2 → ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm_squared := (v 0) ^ 2 + (v 1) ^ 2
  Matrix.of (λ i j ↦ (v i * v j) / norm_squared)

def v : Fin 2 → ℝ := ![3, 5]

theorem det_projection_zero :
  Matrix.det (projection_matrix v) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_zero_l192_19234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_division_by_positive_leading_negative_constant_l192_19284

/-- A polynomial with all non-negative coefficients -/
def NonNegativePolynomial (P : Polynomial ℝ) : Prop :=
  ∀ i, P.coeff i ≥ 0

/-- A polynomial with positive leading coefficient and negative constant term -/
def PositiveLeadingNegativeConstant (Q : Polynomial ℝ) : Prop :=
  Q.leadingCoeff > 0 ∧ Q.coeff 0 < 0

/-- A non-constant polynomial -/
def NonConstantPolynomial (P : Polynomial ℝ) : Prop :=
  ∃ i > 0, P.coeff i ≠ 0

theorem no_division_by_positive_leading_negative_constant
  (P : Polynomial ℝ) (hP : NonNegativePolynomial P) (hP_nonconstant : NonConstantPolynomial P) :
  ¬ ∃ Q : Polynomial ℝ, PositiveLeadingNegativeConstant Q ∧ Q ∣ P :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_division_by_positive_leading_negative_constant_l192_19284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_point_f1_stationary_point_f2_l192_19231

-- Define the functions
def f1 (x y : ℝ) : ℝ := (x - 3)^2 + (y - 2)^2
def f2 (x y z : ℝ) : ℝ := x^2 + 4*y^2 + 9*z^2 - 4*x + 16*y + 18*z + 1

-- Define the concept of a stationary point
def IsStationaryPoint (f : ℝ → ℝ → ℝ) (x y : ℝ) : Prop :=
  (deriv (fun x => f x y) x = 0) ∧ (deriv (fun y => f x y) y = 0)

def IsStationaryPoint3D (f : ℝ → ℝ → ℝ → ℝ) (x y z : ℝ) : Prop :=
  (deriv (fun x => f x y z) x = 0) ∧ 
  (deriv (fun y => f x y z) y = 0) ∧ 
  (deriv (fun z => f x y z) z = 0)

-- State the theorems
theorem stationary_point_f1 : IsStationaryPoint f1 3 2 := by sorry

theorem stationary_point_f2 : IsStationaryPoint3D f2 2 (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_point_f1_stationary_point_f2_l192_19231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_temperature_l192_19273

/-- Temperatures for each day of the week -/
structure WeekTemperatures where
  monday : ℚ
  tuesday : ℚ
  wednesday : ℚ
  thursday : ℚ
  friday : ℚ

/-- The average temperature of four consecutive days -/
def average_temp (t1 t2 t3 t4 : ℚ) : ℚ := (t1 + t2 + t3 + t4) / 4

theorem monday_temperature (temps : WeekTemperatures) : 
  average_temp temps.monday temps.tuesday temps.wednesday temps.thursday = 48 →
  average_temp temps.tuesday temps.wednesday temps.thursday temps.friday = 46 →
  temps.friday = 33 →
  (temps.monday = 41 ∨ temps.tuesday = 41 ∨ temps.wednesday = 41 ∨ temps.thursday = 41 ∨ temps.friday = 41) →
  temps.monday = 41 := by
  sorry

#check monday_temperature

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_temperature_l192_19273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l192_19285

noncomputable section

/-- The curve function y = (1/3)x³ - x --/
def f (x : ℝ) : ℝ := (1/3) * x^3 - x

/-- The derivative of the curve function --/
def f' (x : ℝ) : ℝ := x^2 - 1

/-- A point on the curve where the line is tangent --/
structure TangentPoint where
  x : ℝ
  y : ℝ
  on_curve : y = f x
  non_zero : x ≠ 0

/-- The slope of the tangent line at a given point --/
def tangent_slope (p : TangentPoint) : ℝ := f' p.x

/-- The equation of a line passing through (2,-2) with slope m --/
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * (x - 2) - 2

/-- The theorem stating that the tangent line passes through (2,-2) and is tangent to the curve --/
theorem tangent_line_equation : 
  ∃ (p : TangentPoint), (line_equation (tangent_slope p) p.x = p.y) ∧ 
  ((line_equation (tangent_slope p) = λ x ↦ -x) ∨ (line_equation (tangent_slope p) = λ x ↦ 8*x - 18)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l192_19285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unusual_arithmetic_l192_19259

/-- In a country with unusual arithmetic, 1/5 of 8 equals 4 -/
def country_arithmetic : Prop := (1/5 : ℝ) * 8 = 4

/-- The multiplier used in this country's arithmetic -/
noncomputable def country_multiplier : ℝ := 4 / ((1/5 : ℝ) * 8)

/-- The theorem stating that if 1/4 of X equals 10 in this country, then X must be 16 -/
theorem unusual_arithmetic (X : ℝ) (h : country_arithmetic) : 
  (1/4 : ℝ) * X * country_multiplier = 10 → X = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unusual_arithmetic_l192_19259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_expected_gain_l192_19276

/-- A game where player A conceals a coin and player B guesses its value -/
structure CoinGame where
  /-- The probability that player A chooses the 10 copeck coin -/
  p₁ : ℝ
  /-- The probability that player B guesses the coin is 10 copecks -/
  p₂ : ℝ
  /-- The probability p₁ is between 0 and 1 -/
  h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1
  /-- The probability p₂ is between 0 and 1 -/
  h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1

/-- The expected gain for player B given the probabilities -/
noncomputable def expectedGain (g : CoinGame) : ℝ :=
  60 * g.p₁ * g.p₂ - 35 * g.p₁ - 35 * g.p₂ + 20

/-- The optimal strategy for player A -/
noncomputable def optimalStrategyA : ℝ := 35 / 60

/-- Theorem: The expected gain for player B is -5/12 when both players use optimal strategies -/
theorem optimal_expected_gain (g : CoinGame) :
  g.p₁ = optimalStrategyA → expectedGain g = -5/12 := by
  sorry

#check optimal_expected_gain

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_expected_gain_l192_19276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_totally_convex_is_real_analytic_l192_19213

open Set
open Function

/-- A function is totally convex if (-1)^k * f^(k)(t) > 0 for all t and k > 0 -/
def TotallyConvex (f : ℝ → ℝ) : Prop :=
  ∀ (t : ℝ) (k : ℕ), k > 0 → ((-1 : ℝ) ^ k) * (iteratedDeriv k f t) > 0

/-- Main theorem: Every totally convex function on (0,+∞) is real analytic -/
theorem totally_convex_is_real_analytic
  (f : ℝ → ℝ)
  (hf : DifferentiableOn ℝ f (Set.Ioi 0))
  (h_smooth : ∀ (n : ℕ), DifferentiableOn ℝ (iteratedDeriv n f) (Set.Ioi 0))
  (h_convex : TotallyConvex f) :
  AnalyticOn ℝ f (Set.Ioi 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_totally_convex_is_real_analytic_l192_19213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_existence_l192_19210

theorem partition_existence : ∃ S : Set ℕ, Set.Infinite S ∧ ∀ m ∈ S, ∃ A B C : Finset ℕ,
  (A ∪ B ∪ C = Finset.range (3 * m)) ∧
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
  (A.card = m) ∧ (B.card = m) ∧ (C.card = m) ∧
  ∃ f g h : Fin m → ℕ,
    (∀ i, f i ∈ A) ∧ (∀ i, g i ∈ B) ∧ (∀ i, h i ∈ C) ∧
    (∀ i, f i + g i = h i) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_existence_l192_19210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l192_19209

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (2*x + 3)^0 / Real.sqrt (|x| - x)

-- State the theorem
theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | x < 0 ∧ x ≠ -3/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l192_19209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_negative_sufficient_not_necessary_l192_19245

noncomputable section

/-- The quadratic function f(x) = x^2 + bx -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

/-- The minimum value of f(x) -/
def f_min (b : ℝ) : ℝ := -(b^2/4)

/-- Theorem stating that b < 0 is a sufficient but not necessary condition -/
theorem b_negative_sufficient_not_necessary (b : ℝ) :
  (b < 0 → ∀ x, f b (f b x) ≥ f_min b) ∧
  (∃ c ≥ 0, ∀ x, f c (f c x) ≥ f_min c) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_negative_sufficient_not_necessary_l192_19245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_prime_factorial_sum_divisible_l192_19296

/-- Given positive integers x and y, if their average is prime and (x! + y!) / (x + y) is an integer, then x = y and x is prime -/
theorem average_prime_factorial_sum_divisible (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  (∃ p : ℕ, Nat.Prime p ∧ (x + y : ℚ) / 2 = p) →
  (∃ k : ℕ, k * (x + y) = Nat.factorial x + Nat.factorial y) →
  x = y ∧ Nat.Prime x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_prime_factorial_sum_divisible_l192_19296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l192_19255

def sequence_a : ℕ → ℚ
  | 0 => -2  -- Define for 0 to cover all natural numbers
  | n + 1 => 2 + (2 * sequence_a n) / (1 - sequence_a n)

theorem sixth_term_value : sequence_a 5 = -14/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l192_19255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l192_19250

-- Define the sets P and Q
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | (x - 1)^2 ≤ 4}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l192_19250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_of_C_l192_19218

-- Define points A and B
def A : ℝ × ℝ := (-3, 2)
def B : ℝ × ℝ := (5, 10)

-- Define C as a function of x
def C (x : ℝ) : ℝ × ℝ := (x, 8)  -- y-coordinate is given as 8

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem x_coordinate_of_C :
  -- C is on line segment AB
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C (7/3) = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2)) →
  -- C is twice as far from A as it is from B
  distance A (C (7/3)) = 2 * distance B (C (7/3)) →
  -- The x-coordinate of C is 7/3
  (C (7/3)).1 = 7/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_of_C_l192_19218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_sum_l192_19252

theorem cos_double_sum (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) : 
  Real.cos (2*α + 2*β) = 1/9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_sum_l192_19252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_polynomials_l192_19203

noncomputable def is_polynomial (e : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∃ (p : Polynomial ℝ), ∀ x y a, e x y a = p.eval (x + y + a)

noncomputable def expr1 (x y a : ℝ) : ℝ := 1 / x
noncomputable def expr2 (x y a : ℝ) : ℝ := 2*x + y
noncomputable def expr3 (x y a : ℝ) : ℝ := (1/3) * a^2
noncomputable def expr4 (x y a : ℝ) : ℝ := (x - y) / Real.pi
noncomputable def expr5 (x y a : ℝ) : ℝ := (5*y) / (4*x)
def expr6 (x y a : ℝ) : ℝ := 0

theorem four_polynomials :
  ¬(is_polynomial expr1) ∧
  is_polynomial expr2 ∧
  is_polynomial expr3 ∧
  is_polynomial expr4 ∧
  ¬(is_polynomial expr5) ∧
  is_polynomial expr6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_polynomials_l192_19203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_distance_center_line_l192_19265

/-- The distance between a point (x₀, y₀) and a line ax + by + c = 0 -/
noncomputable def distance_point_line (x₀ y₀ a b c : ℝ) : ℝ :=
  (|a * x₀ + b * y₀ + c|) / Real.sqrt (a^2 + b^2)

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop :=
  3 * x + 4 * y = 5

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 5

/-- The theorem stating that the line intersects the circle -/
theorem line_intersects_circle :
  ∃ x y : ℝ, line_equation x y ∧ circle_equation x y := by
  sorry

/-- The distance between the center of the circle and the line -/
theorem distance_center_line :
  distance_point_line 1 (-2) 3 4 (-5) < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_distance_center_line_l192_19265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_student_weight_l192_19200

/-- Given 29 students with an average weight of 28 kg, if a new student is admitted
    and the new average weight becomes 27.4 kg, then the weight of the new student is 10 kg. -/
theorem new_student_weight (initial_count : ℕ) (initial_avg : ℚ) (new_avg : ℚ) (new_student : ℚ) :
  initial_count = 29 →
  initial_avg = 28 →
  new_avg = 27.4 →
  (initial_count : ℚ) * initial_avg + new_student = (initial_count + 1 : ℚ) * new_avg →
  new_student = 10 := by
  sorry

#check new_student_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_student_weight_l192_19200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_x_squared_at_one_l192_19206

/-- The equation of the tangent line to y = x^2 at (1, 1) is 2x - y - 1 = 0 -/
theorem tangent_line_x_squared_at_one : 
  let f : ℝ → ℝ := fun x ↦ x^2
  let point : ℝ × ℝ := (1, 1)
  let tangent_line : ℝ → ℝ → Prop := fun x y ↦ 2*x - y - 1 = 0
  (∀ x y, tangent_line x y ↔ y - f point.1 = (deriv f) point.1 * (x - point.1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_x_squared_at_one_l192_19206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_arcs_implies_sum_of_squares_l192_19232

/-- Two lines dividing a unit circle into four equal arcs implies a²+b² = 2 -/
theorem equal_arcs_implies_sum_of_squares (a b : ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 1 → 
    (y = x + a ∨ y = x + b) → 
    (∃ (θ : ℝ), θ ∈ Set.Icc 0 (2 * Real.pi) ∧ 
      (Real.cos θ = x ∧ Real.sin θ = y) ∧ 
      (∃ (k : ℕ), k < 4 ∧ θ = (k : ℝ) * Real.pi / 2))) →
  a^2 + b^2 = 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_arcs_implies_sum_of_squares_l192_19232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l192_19279

theorem max_value_expression (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 1) :
  a + Real.sqrt (a * b) + (a * b * c) ^ (1/3) ≤ 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l192_19279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_fourth_power_l192_19287

theorem coefficient_x_fourth_power : 
  (Polynomial.coeff (Polynomial.X * (2 * Polynomial.X - 1)^6) 4) = -160 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_fourth_power_l192_19287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_roots_l192_19260

theorem sine_cosine_roots (θ : ℝ) (m : ℝ) : 
  (4 * (Real.sin θ)^2 + 2 * m * Real.sin θ + m = 0) → 
  (4 * (Real.cos θ)^2 + 2 * m * Real.cos θ + m = 0) → 
  ((2 * m)^2 - 16 * m ≥ 0) →
  m = 1 - Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_roots_l192_19260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_points_theorem_l192_19229

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the line l
def line_l (x : ℝ) : Prop := x = 25 / 4

-- Define the distance from a point to line l
def distance_to_l (x : ℝ) : ℝ := 25 / 4 - x

-- Define the arithmetic sequence property
def arithmetic_sequence (d₁ d₂ d₃ : ℝ) : Prop := 2 * d₂ = d₁ + d₃

theorem ellipse_points_theorem (x₁ y₁ x₂ y₂ : ℝ) :
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
  arithmetic_sequence (distance_to_l x₁) (distance_to_l 4) (distance_to_l x₂) →
  (x₁ + x₂ = 8 ∧
   ∃ x₀ : ℝ, 25 * x₀ - 20 * 0 - 64 = 0 ∧
              (y₁ + y₂) / 2 = (x₁ - x₂) / (y₁ - y₂) * (x₀ - 4)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_points_theorem_l192_19229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thursday_sales_l192_19211

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def friday_sales : ℕ := 135
def unsold_percentage : ℚ := 71.28571428571429 / 100

theorem thursday_sales :
  ∃ (thursday_sales : ℕ),
    thursday_sales = initial_stock - 
      (Int.toNat ⌊(initial_stock : ℚ) * unsold_percentage⌋) - 
      (monday_sales + tuesday_sales + wednesday_sales + friday_sales) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thursday_sales_l192_19211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_lengths_equal_l192_19224

/-- Regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n) → ℝ × ℝ

/-- Coloring of vertices -/
def Coloring (n : ℕ) := Fin (2*n) → Bool

/-- Segment lengths between pairs of vertices of the same color -/
def SegmentLengths (n : ℕ) (p : RegularPolygon n) (c : Coloring n) (color : Bool) : Multiset ℝ :=
  sorry

theorem segment_lengths_equal (n : ℕ) (p : RegularPolygon n) (c : Coloring n) 
  (h : (Finset.filter (fun i => c i = true) (Finset.univ : Finset (Fin (2*n)))).card = n) :
  SegmentLengths n p c true = SegmentLengths n p c false :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_lengths_equal_l192_19224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_difference_l192_19207

/-- Represents the scenario of a truck and a car meeting on a road --/
structure MeetingScenario where
  S : ℝ  -- Distance between village and city
  x : ℝ  -- Speed of the truck
  y : ℝ  -- Speed of the car
  h : x > 0 ∧ y > 0 ∧ S > 0  -- Speeds and distance are positive

/-- The distance from the village to the meeting point when the truck starts 45 minutes earlier --/
noncomputable def earlier_truck_meeting (scenario : MeetingScenario) : ℝ :=
  scenario.S * scenario.x / (scenario.x + scenario.y) + 0.75 * scenario.x + 
  (scenario.S - 0.75 * scenario.x) * scenario.x / (scenario.x + scenario.y)

/-- The distance from the village to the meeting point when the car starts 20 minutes earlier --/
noncomputable def earlier_car_meeting (scenario : MeetingScenario) : ℝ :=
  (scenario.S - 1/3 * scenario.y) * scenario.x / (scenario.x + scenario.y)

/-- The theorem stating that under the given conditions, k = 8 --/
theorem meeting_point_difference (scenario : MeetingScenario) 
  (h1 : earlier_truck_meeting scenario = scenario.S * scenario.x / (scenario.x + scenario.y) + 18) :
  scenario.S * scenario.x / (scenario.x + scenario.y) - earlier_car_meeting scenario = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_difference_l192_19207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_iff_continuous_composition_l192_19246

/-- Definition of the function k_n -/
noncomputable def k_n (n : ℝ) (x : ℝ) : ℝ :=
  if x ≤ -n then -n
  else if x < n then x
  else n

/-- Theorem: A real-valued function f is continuous if and only if
    for all n, the composition of k_n and f is continuous -/
theorem continuous_iff_continuous_composition (f : ℝ → ℝ) :
  Continuous f ↔ ∀ n, Continuous (k_n n ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_iff_continuous_composition_l192_19246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l192_19220

/-- A type representing a line in a plane -/
structure Line where
  id : ℕ

/-- A type representing a point in a plane -/
structure Point where

/-- The set of all lines -/
def all_lines : Finset Line := sorry

/-- The number of lines -/
def num_lines : ℕ := 100

/-- The common point through which some lines pass -/
def point_A : Point := sorry

/-- Predicate to check if a line is parallel to every 4th line -/
def is_parallel_to_every_4th (l : Line) : Prop := sorry

/-- Predicate to check if a line passes through point A -/
def passes_through_A (l : Line) : Prop := sorry

/-- The set of intersection points between pairs of lines -/
noncomputable def intersection_points : Finset Point := sorry

/-- The main theorem -/
theorem max_intersection_points : 
  (∀ l ∈ all_lines, l.id ≤ num_lines) →
  (∀ l ∈ all_lines, l.id % 4 = 0 → is_parallel_to_every_4th l) →
  (∀ l ∈ all_lines, l.id % 4 = 1 → passes_through_A l) →
  Finset.card intersection_points = 4351 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l192_19220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elina_donut_holes_l192_19215

noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

def elina_radius : ℝ := 5
def marco_radius : ℝ := 7
def priya_radius : ℝ := 9

noncomputable def elina_area : ℝ := sphere_surface_area elina_radius
noncomputable def marco_area : ℝ := sphere_surface_area marco_radius
noncomputable def priya_area : ℝ := sphere_surface_area priya_radius

theorem elina_donut_holes :
  ∃ (n : ℕ), n * elina_area = n * marco_area ∧ n * elina_area = n * priya_area ∧ n = 441 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elina_donut_holes_l192_19215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l192_19253

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + 3 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-2, 1)

-- Define the radius of the circle
noncomputable def circle_radius : ℝ := Real.sqrt 2

-- Theorem statement
theorem circle_properties :
  ∀ (x y : ℝ),
    circle_equation x y ↔ 
      (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
by
  -- The proof is skipped for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l192_19253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_l192_19222

/-- Converts kilometers per hour to meters per second -/
noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

theorem speed_conversion (speed_kmph : ℝ) (speed_mps : ℝ) 
  (h : speed_kmph = 18) (h2 : speed_mps = kmph_to_mps speed_kmph) : 
  speed_mps = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_l192_19222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l192_19281

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |2 - 1/x|

-- State the theorem
theorem function_properties (a b : ℝ) (ha : 0 < a) (hb : a < b) (hf : f a = f b) :
  (1/a + 1/b = 4) ∧
  (32/3 ≤ 1/a^2 + 2/b^2) ∧
  (1/a^2 + 2/b^2 < 16) ∧
  ¬∃ (m n : ℝ), (0 < m) ∧ (m < n) ∧ (∀ x, m ≤ x ∧ x ≤ n → m ≤ f x ∧ f x ≤ n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l192_19281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_to_circumscribed_sphere_volume_ratio_cube_l192_19264

/-- The ratio of the volume of the inscribed sphere to the circumscribed sphere of a cube -/
theorem inscribed_to_circumscribed_sphere_volume_ratio_cube :
  ∀ (a : ℝ), a > 0 →
  (4/3 * Real.pi * a^3) / (4/3 * Real.pi * (Real.sqrt 3 * a)^3) = 1 / (3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_to_circumscribed_sphere_volume_ratio_cube_l192_19264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_value_l192_19240

/-- Represents the value of a letter in the alphabet -/
structure LetterValue where
  val : ℕ

/-- Represents a word as a list of letter values -/
def Word := List LetterValue

/-- The point value of a word is the sum of its letters' values -/
def wordValue (w : Word) : ℕ := w.map (·.val) |>.sum

/-- Given conditions -/
axiom distinct_values : ∀ (x y : LetterValue), x ≠ y → x.val ≠ y.val

def F : LetterValue := ⟨23⟩

axiom FORMED_value : wordValue [F, ⟨0⟩, ⟨0⟩, ⟨0⟩, ⟨0⟩, ⟨0⟩] = 63

axiom DEMO_value : wordValue [⟨0⟩, ⟨0⟩, ⟨0⟩, ⟨0⟩] = 30

axiom MODE_value : wordValue [⟨0⟩, ⟨0⟩, ⟨0⟩, ⟨0⟩] = 41

/-- Theorem: The value of R is 10 -/
theorem R_value : ∃ (R : LetterValue), R.val = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_value_l192_19240

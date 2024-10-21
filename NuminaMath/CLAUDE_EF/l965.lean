import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l965_96571

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Checks if a point lies on the parabola y^2 = √3x -/
def onParabola (p : Point) : Prop :=
  p.y^2 = Real.sqrt 3 * p.x

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The side length of the equilateral triangle is 2√3 ± 3 -/
theorem equilateral_triangle_side_length 
  (t : EquilateralTriangle) 
  (h1 : t.v1 = ⟨Real.sqrt 3 / 4, 0⟩) 
  (h2 : onParabola t.v2) 
  (h3 : onParabola t.v3) 
  (h4 : distance t.v1 t.v2 = distance t.v2 t.v3) 
  (h5 : distance t.v2 t.v3 = distance t.v3 t.v1) : 
  ∃ (sign : ℝ), sign^2 = 1 ∧ distance t.v1 t.v2 = 2 * Real.sqrt 3 + sign * 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l965_96571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_5_27_l965_96523

/-- Represents a tetrahedron with inscribed, circumscribed, and externally tangent spheres -/
structure Tetrahedron where
  R : ℝ
  inscribed_radius : ℝ
  circumscribed_radius : ℝ
  external_radius : ℝ
  inscribed_radius_eq : inscribed_radius = 2 * R / 3
  circumscribed_radius_eq : circumscribed_radius = 2 * R
  external_radius_eq : external_radius = inscribed_radius

/-- The probability of a randomly chosen point in the circumscribed sphere
    being inside either the inscribed sphere or one of the four externally tangent spheres -/
noncomputable def probability (t : Tetrahedron) : ℝ :=
  (5 * (4 / 3) * Real.pi * t.inscribed_radius ^ 3) /
  ((4 / 3) * Real.pi * t.circumscribed_radius ^ 3)

/-- Theorem stating that the probability is equal to 5/27 -/
theorem probability_is_5_27 (t : Tetrahedron) : probability t = 5 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_5_27_l965_96523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_plot_seeds_l965_96511

/-- Proves that the number of seeds planted in the second plot is 200 --/
theorem second_plot_seeds (seeds_first_plot : ℕ) (germination_rate_first : ℚ) 
  (germination_rate_second : ℚ) (total_germination_rate : ℚ) (seeds_second_plot : ℕ) : 
  seeds_first_plot = 300 →
  germination_rate_first = 1/5 →
  germination_rate_second = 7/20 →
  total_germination_rate = 13/50 →
  (germination_rate_first * (seeds_first_plot : ℚ) + 
   germination_rate_second * (seeds_second_plot : ℚ)) / 
  ((seeds_first_plot : ℚ) + (seeds_second_plot : ℚ)) = total_germination_rate →
  seeds_second_plot = 200 := by
  sorry

#check second_plot_seeds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_plot_seeds_l965_96511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_form_set_l965_96591

-- Define the property of being a square number
def IsSquareNum (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- Define the set of all squares
def SquareSet : Set ℕ := {n : ℕ | IsSquareNum n}

-- Theorem statement
theorem squares_form_set :
  -- Definiteness: For any natural number, we can determine if it's in the set or not
  (∀ n : ℕ, IsSquareNum n ∨ ¬IsSquareNum n) ∧
  -- Unorderedness: The order of elements doesn't matter (this is inherent in the definition of sets)
  True ∧
  -- Uniqueness: Each element appears only once (also inherent in the definition of sets)
  True := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_form_set_l965_96591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_credit_is_76_l965_96596

/-- Calculates the remaining credit to be paid given the credit limit, additional purchase, discount rate, and payments made. -/
noncomputable def remaining_credit (credit_limit : ℝ) (additional_purchase : ℝ) (discount_rate : ℝ) (payment1 : ℝ) (payment2 : ℝ) : ℝ :=
  let total_spent := credit_limit + additional_purchase
  let discount := if total_spent ≥ 50 then discount_rate * total_spent else 0
  let total_after_discount := total_spent - discount
  let total_paid := payment1 + payment2
  total_after_discount - total_paid

/-- Theorem stating that under the given conditions, the remaining credit to be paid is $76. -/
theorem remaining_credit_is_76 :
  remaining_credit 100 20 0.05 15 23 = 76 := by
  -- Unfold the definition of remaining_credit
  unfold remaining_credit
  -- Simplify the expressions
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_credit_is_76_l965_96596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_l965_96516

open Real

theorem limit_of_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |((n + 1 : ℝ)^(1/2) - (n^3 + 1 : ℝ)^(1/3)) / ((n + 1 : ℝ)^(1/4) - (n^5 + 1 : ℝ)^(1/5)) - 1| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_l965_96516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l965_96554

theorem quadratic_equation_solution : 
  let x₁ : ℝ := 1 + Real.sqrt 6 / 2
  let x₂ : ℝ := 1 - Real.sqrt 6 / 2
  (2 * x₁^2 - 4 * x₁ - 1 = 0) ∧ (2 * x₂^2 - 4 * x₂ - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l965_96554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_matches_given_condition_f_complete_definition_l965_96508

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then x^2 - 2*x + 3
  else if x < 0 then -x^2 - 2*x - 3
  else 0

theorem f_is_odd_and_matches_given_condition : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x > 0, f x = x^2 - 2*x + 3) := by
  sorry

theorem f_complete_definition :
  ∀ x, f x = if x > 0 then x^2 - 2*x + 3
            else if x < 0 then -x^2 - 2*x - 3
            else 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_matches_given_condition_f_complete_definition_l965_96508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l965_96562

/-- The function f(x) defined in the problem -/
noncomputable def f (x m : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.sin x) ^ 2 + m

/-- Theorem stating that if the sum of max and min values of f is 0 and f(π/2) = 0, then m = 1 -/
theorem problem_solution (m : ℝ) :
  (∃ (max min : ℝ), (∀ x, f x m ≤ max) ∧ (∀ x, f x m ≥ min) ∧ max + min = 0) →
  f (Real.pi / 2) m = 0 →
  m = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l965_96562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_49_20_l965_96509

/-- The area of the triangle formed by the intersection of three lines -/
noncomputable def triangleArea (line1 line2 line3 : ℝ → ℝ) : ℝ :=
  sorry

/-- The first line equation: y = 2x + 1 -/
noncomputable def line1 (x : ℝ) : ℝ := 2 * x + 1

/-- The second line equation: y = -1/2x + 4 -/
noncomputable def line2 (x : ℝ) : ℝ := -1/2 * x + 4

/-- The third line equation: y = 2 -/
noncomputable def line3 (_ : ℝ) : ℝ := 2

theorem triangle_area_is_49_20 :
  triangleArea line1 line2 line3 = 49/20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_49_20_l965_96509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_symmetric_points_l965_96555

noncomputable def f (x : ℝ) := 2 * Real.sin (Real.pi * x / 6 + Real.pi / 3)

theorem dot_product_symmetric_points
  (A B C : ℝ × ℝ)
  (h_domain : -2 < A.1 ∧ A.1 < 10)
  (h_A : f A.1 = 0 ∧ A.2 = 0)
  (h_BC : f B.1 = B.2 ∧ f C.1 = C.2)
  (h_symmetric : B.1 + C.1 = 2 * A.1 ∧ B.2 + C.2 = 0) :
  (B.1 + C.1, B.2 + C.2) • A = 32 := by
  sorry

#check dot_product_symmetric_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_symmetric_points_l965_96555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l965_96531

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x + 6 else x + 6

-- Define the solution set
def solution_set : Set ℝ := Set.Ioo (-3) 1 ∪ Set.Ioi 3

-- Theorem statement
theorem f_inequality_solution_set :
  {x : ℝ | f x > f 1} = solution_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l965_96531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_store_theorem_l965_96594

/-- Represents the purchase and sales data for a toy store -/
structure ToyStore where
  first_purchase_cost : ℕ
  second_purchase_cost : ℕ
  price_difference : ℕ
  discounted_quantity : ℕ
  discount_percentage : Rat
  minimum_profit : ℕ

/-- Calculates the first purchase quantity and minimum retail price -/
noncomputable def calculate_toy_data (store : ToyStore) : ℕ × ℕ :=
  let first_quantity := 400 / 2
  let total_quantity := first_quantity * 3
  let full_price_quantity := total_quantity - store.discounted_quantity
  let y := (store.minimum_profit + store.first_purchase_cost + store.second_purchase_cost : ℚ)
  let y := y / (full_price_quantity + store.discounted_quantity * (1 - store.discount_percentage))
  (first_quantity, y.ceil.toNat)

theorem toy_store_theorem (store : ToyStore) 
  (h1 : store.first_purchase_cost = 7200)
  (h2 : store.second_purchase_cost = 14800)
  (h3 : store.price_difference = 2)
  (h4 : store.discounted_quantity = 80)
  (h5 : store.discount_percentage = 2/5)
  (h6 : store.minimum_profit = 4800) :
  calculate_toy_data store = (200, 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_store_theorem_l965_96594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_per_flat_l965_96503

/-- Proves the number of roses in each flat given the conditions of Seymour's plant shop --/
theorem roses_per_flat 
  (petunia_flats : ℕ) 
  (petunias_per_flat : ℕ) 
  (rose_flats : ℕ) 
  (venus_flytraps : ℕ) 
  (petunia_fertilizer : ℕ) 
  (rose_fertilizer : ℕ) 
  (venus_flytrap_fertilizer : ℕ) 
  (total_fertilizer : ℕ)
  (h1 : petunia_flats = 4)
  (h2 : petunias_per_flat = 8)
  (h3 : rose_flats = 3)
  (h4 : venus_flytraps = 2)
  (h5 : petunia_fertilizer = 8)
  (h6 : rose_fertilizer = 3)
  (h7 : venus_flytrap_fertilizer = 2)
  (h8 : total_fertilizer = 314) : 
  ∃ (roses_per_flat : ℕ), 
    roses_per_flat * rose_flats * rose_fertilizer + 
    petunia_flats * petunias_per_flat * petunia_fertilizer + 
    venus_flytraps * venus_flytrap_fertilizer = total_fertilizer ∧ 
    roses_per_flat = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_per_flat_l965_96503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_fourth_plus_a_negative_fourth_l965_96549

theorem a_fourth_plus_a_negative_fourth (a : ℝ) (h : 5 = a + a⁻¹) : a^4 + a⁻¹^4 = 527 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_fourth_plus_a_negative_fourth_l965_96549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_relation_l965_96579

theorem sin_angle_relation (α : ℝ) (h1 : α ∈ Set.Ioo (π/2) π) (h2 : Real.sin (α + π/12) = 1/3) :
  Real.sin (α + 7*π/12) = -2*Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_relation_l965_96579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_factorial_sum_l965_96553

theorem greatest_prime_factor_of_factorial_sum : 
  (List.maximum? (Nat.factors (Nat.factorial 15 + Nat.factorial 17))).isSome ∧
  (List.maximum? (Nat.factors (Nat.factorial 15 + Nat.factorial 17))).get! = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_factorial_sum_l965_96553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concrete_path_volume_l965_96500

/-- Represents the dimensions of a concrete path -/
structure PathDimensions where
  width : ℝ  -- in feet
  length : ℝ  -- in feet
  thickness : ℝ  -- in inches

/-- Converts feet to yards -/
noncomputable def feetToYards (feet : ℝ) : ℝ := feet / 3

/-- Converts inches to yards -/
noncomputable def inchesToYards (inches : ℝ) : ℝ := inches / 36

/-- Calculates the volume of concrete needed in cubic yards -/
noncomputable def concreteVolume (d : PathDimensions) : ℝ :=
  feetToYards d.width * feetToYards d.length * inchesToYards d.thickness

/-- Rounds up to the nearest whole number -/
noncomputable def ceilToInt (x : ℝ) : ℤ := Int.ceil x

theorem concrete_path_volume (d : PathDimensions) 
  (h1 : d.width = 4)
  (h2 : d.length = 150)
  (h3 : d.thickness = 4) :
  ceilToInt (concreteVolume d) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concrete_path_volume_l965_96500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_milk_percentage_l965_96592

theorem initial_milk_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_milk_percentage : ℝ)
  (initial_milk_percentage : ℝ)
  (h1 : initial_volume = 60)
  (h2 : added_water = 33.33333333333333)
  (h3 : final_milk_percentage = 54)
  (h4 : initial_volume * (initial_milk_percentage / 100) = 
        (initial_volume + added_water) * (final_milk_percentage / 100)) :
  initial_milk_percentage = 84 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_milk_percentage_l965_96592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_running_speed_l965_96564

-- Define the side length of the square field in meters
noncomputable def side_length : ℝ := 50

-- Define the time taken to run around the field in seconds
noncomputable def time_taken : ℝ := 80

-- Define the perimeter of the field
noncomputable def perimeter : ℝ := 4 * side_length

-- Define the speed conversion factor from m/s to km/hr
noncomputable def speed_conversion : ℝ := 3600 / 1000

-- Theorem to prove
theorem boys_running_speed :
  (perimeter / time_taken) * speed_conversion = 9 := by
  -- Unfold the definitions
  unfold perimeter speed_conversion side_length time_taken
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_running_speed_l965_96564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_intersecting_lines_l965_96556

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  -- We don't need to specify fields for this problem
  mk :: -- Empty constructor

/-- A line that intersects the interior of a regular tetrahedron -/
structure IntersectingLine (T : RegularTetrahedron) where
  intersects_at_midpoints : Bool

/-- The set of all valid intersecting lines for a given tetrahedron -/
def validLines (T : RegularTetrahedron) : Set (IntersectingLine T) :=
  {l : IntersectingLine T | l.intersects_at_midpoints}

/-- The set of cardinalities of finite subsets of a given set -/
def finiteSubsetCards {α : Type*} (S : Set α) : Set ℕ :=
  {n : ℕ | ∃ (F : Finset α), F.toSet ⊆ S ∧ F.card = n}

theorem tetrahedron_intersecting_lines (T : RegularTetrahedron) :
  let S := validLines T
  ∃ (min max : ℕ), 
    (∀ m : ℕ, m ∈ finiteSubsetCards S → min ≤ m ∧ m ≤ max) ∧
    max - min = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_intersecting_lines_l965_96556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l965_96534

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  2 * x^2 - 3 * y^2 + 8 * x - 12 * y - 23 = 0

-- Define the focus coordinates
noncomputable def focus : ℝ × ℝ := (-2 - Real.sqrt (5/6), -2)

-- Theorem statement
theorem focus_of_hyperbola :
  let (fx, fy) := focus
  ∃ c, c > 0 ∧
    ∀ x y, hyperbola_eq x y →
      (x - fx)^2 + (y - fy)^2 = 
      ((x + 2)^2 / (1/2) - (y + 2)^2 / (1/3) + c^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l965_96534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_given_equation_l965_96517

theorem tan_value_for_given_equation (α : ℝ) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 2 →
  Real.tan α = 1 ∧ α = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_given_equation_l965_96517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l965_96501

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_lambda : ∀ l : ℝ,
  let a : ℝ × ℝ := (2, 6)
  let b : ℝ × ℝ := (-1, l)
  are_parallel a b → l = -3 := by
  intro l
  simp [are_parallel]
  intro h
  -- The rest of the proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l965_96501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_l965_96528

-- Define the triangle structure
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the congruence relation between triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Theorem statement
theorem angle_C_value (ABC DEF : Triangle) :
  congruent ABC DEF →
  ABC.A = 100 →
  ABC.B = 60 →  -- Changed from DEF.E to ABC.B
  ABC.C = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_l965_96528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_B_to_center_l965_96524

-- Define the circle and points
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

variable (PointA PointB PointC : ℝ × ℝ)

-- Define the conditions
axiom circle_radius : (11 : ℝ) = 11
axiom points_on_circle : PointA ∈ Circle (0, 0) 11 ∧ PointB ∈ Circle (0, 0) 11 ∧ PointC ∈ Circle (0, 0) 11
axiom distance_AB : ((PointA.1 - PointB.1)^2 + (PointA.2 - PointB.2)^2 : ℝ) = 7^2
axiom distance_BC : ((PointB.1 - PointC.1)^2 + (PointB.2 - PointC.2)^2 : ℝ) = 3^2
axiom right_angle_ABC : (PointA.1 - PointB.1) * (PointC.1 - PointB.1) + (PointA.2 - PointB.2) * (PointC.2 - PointB.2) = 0

-- Theorem statement
theorem distance_B_to_center : 
  PointB.1^2 + PointB.2^2 = 82 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_B_to_center_l965_96524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ms_warren_total_distance_l965_96587

/-- Calculates the distance traveled given speed in mph and time in minutes -/
noncomputable def distance (speed : ℝ) (time : ℝ) : ℝ := speed * (time / 60)

/-- Represents Ms. Warren's running and walking activity -/
def ms_warren_activity : Prop :=
  let run_speed : ℝ := 6
  let run_time : ℝ := 20
  let walk_speed : ℝ := 2
  let walk_time : ℝ := 30
  let total_distance : ℝ := distance run_speed run_time + distance walk_speed walk_time
  total_distance = 3

theorem ms_warren_total_distance : ms_warren_activity := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ms_warren_total_distance_l965_96587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l965_96586

theorem power_inequality :
  (0.1 : ℝ) ^ (0.8 : ℝ) < (0.2 : ℝ) ^ (0.8 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l965_96586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_k_value_l965_96527

def main (e₁ e₂ : ℝ × ℝ) (k : ℝ) : Prop :=
  let a := (2 * e₁.1, 2 * e₁.2) - e₂
  let b := (k * e₁.1, k * e₁.2) + e₂
  e₁ ≠ (0, 0) ∧ 
  e₂ ≠ (0, 0) ∧ 
  (∀ (t : ℝ), e₁ ≠ (t * e₂.1, t * e₂.2)) ∧ 
  (∃ (l : ℝ), a = (l * b.1, l * b.2)) →
  k = -2

theorem prove_k_value : ∀ (e₁ e₂ : ℝ × ℝ) (k : ℝ), main e₁ e₂ k := by
  sorry

#check prove_k_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_k_value_l965_96527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_and_min_distance_l965_96525

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + t/2, Real.sqrt 3/2 * t)

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

noncomputable def curve_C2 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sqrt 3 * Real.sin θ)

noncomputable def dist_to_line_l (p : ℝ × ℝ) : ℝ :=
  (|Real.sqrt 3 * p.1 - Real.sqrt 3 * p.2 - 2 * Real.sqrt 3|) / 2

theorem intersection_length_and_min_distance :
  (∃ A B : ℝ × ℝ, ∃ t₁ t₂ θ₁ θ₂ : ℝ, 
    line_l t₁ = curve_C1 θ₁ ∧ 
    line_l t₂ = curve_C1 θ₂ ∧ 
    A = line_l t₁ ∧ 
    B = line_l t₂ ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2) ∧
  (∀ θ : ℝ, dist_to_line_l (curve_C2 θ) ≥ Real.sqrt 6/2 * (Real.sqrt 2 - 1)) ∧
  (∃ θ : ℝ, dist_to_line_l (curve_C2 θ) = Real.sqrt 6/2 * (Real.sqrt 2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_and_min_distance_l965_96525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l965_96552

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.rpow x (1/4) = 15 / (8 - (Real.rpow x (1/4))^2)

-- Define the approximate solutions
def solution1 : ℝ := 5.0625
def solution2 : ℝ := 39.0625

-- Theorem statement
theorem equation_solutions :
  ∃ (ε : ℝ), ε > 0 ∧
  (∀ (x : ℝ), equation x → 
    (|x - solution1| < ε ∨ |x - solution2| < ε)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l965_96552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l965_96561

noncomputable section

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (x + m) * Real.log x - (m + 1 + 1 / Real.exp 1) * x

-- Theorem statement
theorem problem_solution :
  -- Part I
  (∃ (m : ℝ), ∀ (x : ℝ), x > 0 → (deriv (f m)) x = 0 ↔ x = Real.exp 1) ∧
  -- Part II
  (∀ (x : ℝ), x > 1 → f 1 x + (2 + 1 / Real.exp 1) * x > 2 * x - 2) ∧
  -- Part III
  (∀ (a x : ℝ), a ≥ 2 → x ≥ 1 → 
    |f 1 x - Real.exp 1 / x| < |f 1 x - (Real.exp (x - 1) + a)|) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l965_96561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_muffin_payment_l965_96566

/-- The problem of calculating Janet's initial payment for muffins -/
theorem janet_muffin_payment
  (muffin_price : ℚ)
  (muffin_count : ℕ)
  (change_received : ℚ)
  (h1 : muffin_price = 75 / 100)
  (h2 : muffin_count = 12)
  (h3 : change_received = 11) :
  muffin_price * (muffin_count : ℚ) + change_received = 20 := by
  sorry

#check janet_muffin_payment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_muffin_payment_l965_96566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_decreasing_l965_96559

-- Define the function f(x) = 1/|x|
noncomputable def f (x : ℝ) : ℝ := 1 / abs x

-- State the theorem
theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧  -- f is even
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) -- f is decreasing on (0, +∞)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_decreasing_l965_96559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_vectors_l965_96546

/-- The cosine of the angle between two 2D vectors -/
noncomputable def cos_angle (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))

theorem cos_angle_specific_vectors :
  let a : ℝ × ℝ := (3, 4)
  let b : ℝ × ℝ := (5, 12)
  cos_angle a b = 63 / 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_vectors_l965_96546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_model_correct_l965_96550

/-- A linear function modeling the relationship between price and items sold --/
noncomputable def sales_model (x : ℝ) : ℝ := -1/4 * x + 50

/-- The domain of the sales model --/
def valid_price (x : ℝ) : Prop := 0 < x ∧ x < 200

theorem sales_model_correct (p₁ p₂ q₁ q₂ : ℝ) 
  (h₁ : p₁ = 80 ∧ q₁ = 30)
  (h₂ : p₂ = 120 ∧ q₂ = 20)
  (h₃ : valid_price p₁ ∧ valid_price p₂) :
  sales_model p₁ = q₁ ∧ sales_model p₂ = q₂ := by
  sorry

#check sales_model_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_model_correct_l965_96550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_difference_l965_96593

theorem quadratic_roots_difference (k : ℝ) : 
  ∃ r₁ r₂ : ℝ, (r₁^2 - (k + 4) * r₁ + k = 0) ∧ 
               (r₂^2 - (k + 4) * r₂ + k = 0) ∧ 
               |r₁ - r₂| = Real.sqrt (k^2 + 4*k + 16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_difference_l965_96593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_90_hours_l965_96522

/-- Calculates the total time for a round trip by boat given the boat's speed in still water, 
    the stream's speed, and the distance to the destination. -/
noncomputable def total_time_round_trip (boat_speed : ℝ) (stream_speed : ℝ) (distance : ℝ) : ℝ :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  distance / downstream_speed + distance / upstream_speed

/-- Theorem stating that for a boat with speed 12 km/hr in still water, 
    a stream with speed 4 km/hr, and a destination 480 km away, 
    the total time taken for a round trip is 90 hours. -/
theorem round_trip_time_is_90_hours :
  total_time_round_trip 12 4 480 = 90 := by
  sorry

-- Use #eval only for computable functions
/-- Approximate calculation of the total time for the round trip -/
def approx_total_time : Float :=
  let boat_speed : Float := 12
  let stream_speed : Float := 4
  let distance : Float := 480
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  distance / downstream_speed + distance / upstream_speed

#eval approx_total_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_90_hours_l965_96522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_labeled_complete_graph_divisible_cycle_exists_l965_96541

/-- A complete graph with edge labels -/
structure LabeledCompleteGraph (α : Type*) (β : Type*) where
  vertices : Finset α
  label : α → α → β

/-- The sum of labels along a cycle in a labeled complete graph -/
def cycleLabelSum {α β : Type*} [AddCommGroup β] (G : LabeledCompleteGraph α β) (cycle : List α) : β :=
  let pairs := cycle.zip (cycle.rotateLeft 1)
  pairs.map (fun (a, b) => G.label a b) |>.sum

theorem labeled_complete_graph_divisible_cycle_exists (p : ℕ) (hp : Nat.Prime p) :
  ∀ (G : LabeledCompleteGraph (Fin (1000 * p)) ℤ),
  ∃ (cycle : List (Fin (1000 * p))),
    cycleLabelSum G cycle % p = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_labeled_complete_graph_divisible_cycle_exists_l965_96541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangements_count_l965_96572

/-- The number of different arrangements of 4 boys and 3 girls in a row,
    where exactly 2 of the 3 girls stand together. -/
def arrangements_count : ℕ := 2880

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 3

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_arrangements_count :
  arrangements_count = 
    (Nat.factorial (num_boys + num_girls)) / 
    (Nat.factorial num_boys * Nat.factorial num_girls) * 
    (num_boys + 1) * (num_girls - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangements_count_l965_96572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l965_96530

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 4)

-- State the theorem about the range of the function
theorem f_range : Set.range f = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l965_96530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_f_greater_max_g_l965_96539

open Real

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := exp x / (x^2 - m*x + 1)
def g (x : ℝ) : ℝ := x

-- State the theorem
theorem min_f_greater_max_g (m : ℝ) (hm : m ∈ Set.Ioo 0 (1/2)) :
  ∃ (M N : ℝ), (∀ x ∈ Set.Icc 1 (m+1), f m x ≥ M) ∧
                (∀ x ∈ Set.Icc 1 (m+1), g x ≤ N) ∧
                M > N := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_f_greater_max_g_l965_96539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l965_96535

-- Define the function f(x) = e^x cos(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

-- State the theorem
theorem f_increasing_interval :
  ∀ x : ℝ, x ∈ Set.Ioo 0 Real.pi →
  (∀ y : ℝ, y ∈ Set.Ioo 0 (Real.pi / 4) → y < x → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l965_96535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l965_96565

def x : ℝ × ℝ × ℝ := (6, 12, -1)
def p : ℝ × ℝ × ℝ := (1, 3, 0)
def q : ℝ × ℝ × ℝ := (2, -1, 1)
def r : ℝ × ℝ × ℝ := (0, -1, 2)

theorem vector_decomposition : x = 4 • p + q - r := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l965_96565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_range_l965_96547

-- Define the function f with the given range
noncomputable def f : ℝ → ℝ := sorry

-- Define the range of f
axiom f_range : Set.range f = Set.Icc (1/2) 3

-- Define the function F
noncomputable def F (x : ℝ) : ℝ := f x + 1 / (f x)

-- Theorem statement
theorem F_range : Set.range F = Set.Icc 2 (10/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_range_l965_96547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_calculation_l965_96563

noncomputable def item1_cost : ℝ := 1200
noncomputable def item2_cost : ℝ := 1500
noncomputable def item3_cost : ℝ := 1800

noncomputable def discount1 : ℝ := 0.10
noncomputable def discount2 : ℝ := 0.15
noncomputable def discount3 : ℝ := 0.20

noncomputable def sales_tax : ℝ := 0.05

noncomputable def total_cost : ℝ := item1_cost + item2_cost + item3_cost

noncomputable def discounted_price1 : ℝ := item1_cost * (1 - discount1)
noncomputable def discounted_price2 : ℝ := item2_cost * (1 - discount2)
noncomputable def discounted_price3 : ℝ := item3_cost * (1 - discount3)

noncomputable def final_price1 : ℝ := discounted_price1 * (1 + sales_tax)
noncomputable def final_price2 : ℝ := discounted_price2 * (1 + sales_tax)
noncomputable def final_price3 : ℝ := discounted_price3 * (1 + sales_tax)

noncomputable def total_selling_price : ℝ := final_price1 + final_price2 + final_price3

noncomputable def overall_loss : ℝ := total_cost - total_selling_price

noncomputable def loss_percentage : ℝ := (overall_loss / total_cost) * 100

theorem loss_percentage_calculation :
  (loss_percentage ≥ 11.44) ∧ (loss_percentage ≤ 11.46) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_calculation_l965_96563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balls_in_cube_l965_96513

theorem max_balls_in_cube :
  let cube_side : ℝ := 6
  let ball_radius : ℝ := 2
  let cube_volume : ℝ := cube_side ^ 3
  let ball_volume : ℝ := (4 / 3) * Real.pi * ball_radius ^ 3
  let max_balls : ℤ := ⌊cube_volume / ball_volume⌋
  max_balls = 6 := by
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balls_in_cube_l965_96513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l965_96514

/-- The area of the triangle formed by the lines y = x, x = -5, and the x-axis -/
theorem triangle_area : ∃ (area : Real), area = 12.5 := by
  -- Define the lines and x-axis
  let line1 : Real → Real := λ x ↦ x
  let line2 : Real := -5
  let x_axis : Real → Real := λ _ ↦ 0

  -- Define the intersection point
  let intersection_x : Real := -5
  let intersection_y : Real := -5

  -- Calculate the base and height of the triangle
  let base : Real := 5 -- from -5 to 0
  let height : Real := 5 -- from -5 to 0

  -- Calculate the area
  let area : Real := (1/2) * base * height

  -- Assert that the area is 12.5
  have h : area = 12.5 := by
    -- Proof goes here
    sorry

  -- Conclude the theorem
  exact ⟨area, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l965_96514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_expansion_l965_96568

theorem coefficient_x_squared_expansion : 
  let f : Polynomial ℤ := X^2 + X + 1
  let g : Polynomial ℤ := (1 - X)^6
  (f * g).coeff 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_expansion_l965_96568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_count_l965_96570

theorem strawberry_count (total_fruit : ℕ) (kiwi_fraction : ℚ) (strawberry_count : ℕ) : 
  total_fruit = 78 →
  kiwi_fraction = 1/3 →
  strawberry_count = total_fruit - (kiwi_fraction * ↑total_fruit).floor →
  strawberry_count = 52 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_count_l965_96570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_resultant_shape_area_l965_96599

/-- Represents a square with a given side length and rotation angle. -/
structure RotatedSquare where
  side : ℝ
  angle : ℝ

/-- Calculates the area of overlap between two rotated squares. -/
noncomputable def overlapArea (s1 s2 : RotatedSquare) : ℝ :=
  s1.side * s2.side * Real.sin (s2.angle - s1.angle)

/-- Calculates the total area of the resultant shape formed by overlapping rotated squares. -/
noncomputable def resultantArea (squares : List RotatedSquare) : ℝ :=
  (squares.map (λ s => s.side ^ 2)).sum -
  (List.zipWith overlapArea squares (squares.tail!)).sum

/-- The main theorem stating the area of the resultant shape. -/
theorem resultant_shape_area : 
  let squares := [
    ⟨6, 0⟩,
    ⟨7, 20 * π / 180⟩,
    ⟨8, 40 * π / 180⟩,
    ⟨9, 60 * π / 180⟩
  ]
  resultantArea squares = 220 - 15 * Real.sin (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_resultant_shape_area_l965_96599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinate_rotation_l965_96543

/-- Given a point with rectangular coordinates (-3, -4, 2) and corresponding spherical coordinates
    (ρ, θ, φ), prove that the point with spherical coordinates (ρ, θ + π, φ) has rectangular
    coordinates (3, 4, 2). -/
theorem spherical_coordinate_rotation (ρ θ φ : ℝ) :
  (-3 = ρ * Real.sin φ * Real.cos θ) →
  (-4 = ρ * Real.sin φ * Real.sin θ) →
  (2 = ρ * Real.cos φ) →
  (3 = ρ * Real.sin φ * Real.cos (θ + π)) ∧
  (4 = ρ * Real.sin φ * Real.sin (θ + π)) ∧
  (2 = ρ * Real.cos φ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinate_rotation_l965_96543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l965_96506

/-- The eccentricity of a hyperbola with given equation and asymptotes -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ Set.range (λ t : ℝ × ℝ ↦ t))
  (h_asymptote : ∀ x y : ℝ, (4 * a * x = b * y ∨ 4 * a * x = -b * y) ↔ 
    (x, y) ∈ Set.range (λ t : ℝ × ℝ ↦ t)) :
  Real.sqrt 5 = (Real.sqrt (a^2 + b^2)) / a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l965_96506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_town_game_each_round_l965_96557

/-- Represents a chess tournament with the given properties -/
structure ChessTournament where
  /-- The number of participants in the tournament -/
  num_participants : ℕ
  /-- The number of games played between participants from the same town -/
  same_town_games : ℕ
  /-- Proof that the number of participants is 10 -/
  participant_count : num_participants = 10
  /-- Proof that at least half of the total games are played by players from the same town -/
  half_same_town : same_town_games ≥ (num_participants.choose 2) / 2

/-- Predicate indicating that a game is played in a specific round -/
def game_in_round (round : ℕ) (game : ℕ) : Prop := sorry

/-- Predicate indicating that a game is played by players from the same town -/
def same_town_players (game : ℕ) : Prop := sorry

/-- Theorem stating that in each round of the tournament, there is a game played by two participants from the same town -/
theorem same_town_game_each_round (t : ChessTournament) :
  ∀ (round : ℕ), ∃ (game : ℕ), game_in_round round game ∧ same_town_players game :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_town_game_each_round_l965_96557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_D_l965_96532

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the area of a triangle
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  sorry

-- Define a point D
noncomputable def pointD (t : Triangle) : ℝ × ℝ := 
  sorry

-- Theorem statement
theorem exists_point_D (t : Triangle) :
  ∃ D : ℝ × ℝ,
    triangleArea t.A t.B D = (1/2) * triangleArea t.A t.B t.C ∧
    triangleArea t.B t.C D = (1/6) * triangleArea t.A t.B t.C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_D_l965_96532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fourth_term_l965_96518

/-- Given that x, 2x+2, and 3x+3 are the first three terms of a geometric sequence,
    prove that the fourth term of the sequence is -27/2 -/
theorem geometric_sequence_fourth_term (x : ℝ) 
  (h1 : x ≠ 0)
  (h2 : 2*x + 2 ≠ 0)
  (h3 : 3*x + 3 ≠ 0)
  (h4 : (2*x + 2)^2 = x * (3*x + 3)) : 
  (3*x + 3) * ((2*x + 2) / x) = -27/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fourth_term_l965_96518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_schedule_count_is_312_l965_96551

/-- Represents the six classes that need to be scheduled -/
inductive ScheduledClass
| Chinese
| Mathematics
| Politics
| English
| PE
| Art

/-- A schedule is a permutation of the six classes -/
def Schedule := Fin 6 → ScheduledClass

/-- Checks if a schedule is valid according to the given conditions -/
def isValidSchedule (s : Schedule) : Prop :=
  (∃ i < 3, s i = ScheduledClass.Mathematics) ∧ 
  (s 0 ≠ ScheduledClass.PE)

/-- The number of valid schedules -/
def validScheduleCount : ℕ := sorry

/-- Theorem stating that the number of valid schedules is 312 -/
theorem valid_schedule_count_is_312 : validScheduleCount = 312 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_schedule_count_is_312_l965_96551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centralBankInterest_bankBenefits_registrationReasoning_l965_96576

structure LoyaltyProgram where
  name : String
  requiresRegistration : Bool

structure Bank where
  name : String
  offersLoyaltyProgram : Bool

structure CentralBank where
  name : String
  nationalPaymentSystem : String

def mirLoyaltyProgram : LoyaltyProgram :=
  { name := "Mir Loyalty Program"
  , requiresRegistration := true }

def russianCentralBank : CentralBank :=
  { name := "Central Bank of the Russian Federation"
  , nationalPaymentSystem := "Mir" }

#check mirLoyaltyProgram
#check russianCentralBank

-- Placeholder for arguments
def argumentForCentralBank (cb : CentralBank) (lp : LoyaltyProgram) : Prop := sorry

def argumentForParticipatingBanks (b : Bank) (lp : LoyaltyProgram) : Prop := sorry

def reasonForRegistration (lp : LoyaltyProgram) : Prop := sorry

-- These would be filled in with actual arguments in a real implementation
theorem centralBankInterest : argumentForCentralBank russianCentralBank mirLoyaltyProgram := sorry

theorem bankBenefits : ∀ b : Bank, b.offersLoyaltyProgram → argumentForParticipatingBanks b mirLoyaltyProgram := sorry

theorem registrationReasoning : reasonForRegistration mirLoyaltyProgram := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centralBankInterest_bankBenefits_registrationReasoning_l965_96576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l965_96589

theorem tan_alpha_value (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l965_96589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l965_96507

theorem cos_beta_value (α β : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2))
  (h2 : β ∈ Set.Ioo (π/2) π)
  (h3 : Real.cos α = 1/3)
  (h4 : Real.sin (α + β) = -3/5) : 
  Real.cos β = -(4 + 6*Real.sqrt 2)/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l965_96507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2016_equals_1008_l965_96540

def sequence_a : ℕ → ℚ
  | 0 => 3/5  -- Add this case to handle Nat.zero
  | 1 => 3/5
  | (n+1) => let a_n := sequence_a n
              if 0 ≤ a_n ∧ a_n ≤ 1/2 then 2*a_n
              else if 1/2 < a_n ∧ a_n < 1 then 2*a_n - 1
              else 0  -- This case should never occur based on the problem definition

def S (n : ℕ) : ℚ := (List.range n).map sequence_a |>.sum

theorem sum_2016_equals_1008 : S 2016 = 1008 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2016_equals_1008_l965_96540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l965_96574

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ := Real.sqrt (1 - E.b^2 / E.a^2)

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem about the eccentricity range of an ellipse under specific conditions -/
theorem eccentricity_range (E : Ellipse) 
  (P Q : PointOnEllipse E) 
  (F₁ F₂ : ℝ × ℝ) -- Foci of the ellipse
  (h_symmetric : P.x = -Q.x ∧ P.y = -Q.y) -- P and Q are symmetric w.r.t. origin
  (h_dist_eq : distance (P.x, P.y) (Q.x, Q.y) = distance F₁ F₂)
  (h_area : ∃ S, S ≥ 1/8 * (distance (P.x, P.y) (Q.x, Q.y))^2 ∧ 
    S = 1/2 * |Matrix.det !![P.x - F₂.1, Q.x - F₂.1; P.y - F₂.2, Q.y - F₂.2]|) :
  Real.sqrt 2 / 2 ≤ eccentricity E ∧ eccentricity E ≤ Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l965_96574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evalAt2_invariant_no_strictly_greater_polynomial_l965_96538

/-- Represents a polynomial of degree 200 --/
def Polynomial200 := Fin 201 → ℝ

/-- The transformation operation on a polynomial --/
def transform (p : Polynomial200) (k : Fin 200) (a : ℝ) : Polynomial200 :=
  fun i => 
    if i = k then p i + 2 * a
    else if i = k.succ then p i - a
    else p i

/-- Evaluates a polynomial at x = 2 --/
def evalAt2 (p : Polynomial200) : ℝ :=
  Finset.sum (Finset.univ : Finset (Fin 201)) fun i => p i * (2 ^ i.val)

/-- States that the evaluation at x = 2 is invariant under transformation --/
theorem evalAt2_invariant (p : Polynomial200) (k : Fin 200) (a : ℝ) :
  evalAt2 (transform p k a) = evalAt2 p := by sorry

/-- Evaluates a polynomial at a given real number --/
def evalAt (p : Polynomial200) (x : ℝ) : ℝ :=
  Finset.sum (Finset.univ : Finset (Fin 201)) fun i => p i * (x ^ i.val)

/-- The main theorem stating that no polynomial can be strictly greater than another --/
theorem no_strictly_greater_polynomial (p q : Polynomial200) :
  ¬∀ (x : ℝ), evalAt p x > evalAt q x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evalAt2_invariant_no_strictly_greater_polynomial_l965_96538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_and_range_l965_96526

-- Define the power function f
noncomputable def f (k : ℤ) (x : ℝ) : ℝ := x^(-k^2 + k + 2)

-- Define the function g
noncomputable def g (q : ℝ) (x : ℝ) : ℝ := 1 - q * x^2 + (2*q - 1) * x

-- State the theorem
theorem power_function_and_range :
  ∃! k : ℤ, (∀ x : ℝ, f k x = x^2) ∧
  (f k 2 < f k 3) ∧
  (∃ q : ℝ, q > 0 ∧
    (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → -4 ≤ g q x ∧ g q x ≤ 17/8) ∧
    q = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_and_range_l965_96526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l965_96558

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

-- State the theorems to be proved
theorem f_properties :
  -- 1. f is an odd function
  (∀ x, f (-x) = -f x) ∧
  -- 2. f is monotonic on (-∞, +∞)
  (∀ x y, x < y → f x < f y) ∧
  -- 3. f(x) > 0 when x > 0, and f(x) < 0 when x < 0
  (∀ x, x > 0 → f x > 0) ∧ (∀ x, x < 0 → f x < 0) ∧
  -- 4. f is not a periodic function
  ¬(∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x) ∧
  -- 5. The graph of f is not symmetric about the line y = -x
  ¬(∀ x, f (-f x) = -x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l965_96558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_and_inequality_l965_96512

noncomputable section

-- Define the quadratic function g(x)
def g (m n : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + n + 1

-- Define the function f(x)
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := (g x - 2 * x) / x

theorem quadratic_function_and_inequality 
  (m n : ℝ) 
  (h_m : m > 0) 
  (h_max : ∀ x ∈ Set.Icc 0 3, g m n x ≤ 4) 
  (h_min : ∀ x ∈ Set.Icc 0 3, g m n x ≥ 0) 
  (h_max_achieved : ∃ x ∈ Set.Icc 0 3, g m n x = 4) 
  (h_min_achieved : ∃ x ∈ Set.Icc 0 3, g m n x = 0) :
  (∀ x, g m n x = x^2 - 2*x + 1) ∧ 
  (∀ k, (∀ x ∈ Set.Icc (-3 : ℝ) 3, f (g m n) (2^x) - k * 2^x ≤ 0) → k ≥ 55.5) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_and_inequality_l965_96512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l965_96510

/-- The area of a quadrilateral given its diagonal and offsets -/
noncomputable def quadrilateral_area (d h₁ h₂ : ℝ) : ℝ := (1/2) * d * (h₁ + h₂)

/-- Theorem: The area of a quadrilateral with diagonal 15 cm and offsets 6 cm and 4 cm is 75 cm² -/
theorem quadrilateral_area_example : quadrilateral_area 15 6 4 = 75 := by
  -- Unfold the definition of quadrilateral_area
  unfold quadrilateral_area
  -- Simplify the arithmetic expression
  simp [mul_add, mul_assoc]
  -- Check that the result is equal to 75
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l965_96510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_ratio_l965_96548

/-- The ratio of areas between an inscribed square and its containing square -/
theorem inscribed_square_area_ratio :
  ∀ (outer_side_length : ℝ) (inner_side_length : ℝ),
    outer_side_length > 0 →
    inner_side_length = outer_side_length * (Real.sqrt 2 / 2) →
    (inner_side_length ^ 2) / (outer_side_length ^ 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_ratio_l965_96548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l965_96569

-- Part 1
theorem simplify_expression_1 (a b : ℝ) (hb : b > 0) :
  (a^(3/2) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / ((1/3) * a * b^(5/6)) = -9 * a :=
sorry

-- Part 2
theorem simplify_expression_2 :
  Real.log 3 / Real.log 4 * Real.log 2 / Real.log 9 - Real.log (32^(1/4)) / Real.log (1/2) = 11/8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l965_96569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l965_96536

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- Define the slope of the tangent line
noncomputable def tangent_slope : ℝ := 3 * point.1^2 - 2

-- Define the equation of the tangent line
noncomputable def tangent_line (x : ℝ) : ℝ := tangent_slope * (x - point.1) + point.2

-- Theorem statement
theorem tangent_line_equation : 
  ∀ x : ℝ, tangent_line x = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l965_96536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l965_96581

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x - 3)

-- State the theorem
theorem range_of_f :
  Set.range f = {y : ℝ | y ≠ 2} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l965_96581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_length_for_given_conditions_l965_96505

/-- A race between two runners where one is faster and gives the other a head start -/
structure Race where
  /-- The speed ratio of the faster runner to the slower runner -/
  speed_ratio : ℝ
  /-- The head start distance given to the slower runner -/
  head_start : ℝ

/-- The length of the race course -/
noncomputable def race_length (r : Race) : ℝ :=
  (r.speed_ratio * r.head_start) / (r.speed_ratio - 1)

/-- Theorem stating that for the given conditions, the race length is 92 meters -/
theorem race_length_for_given_conditions :
  let r : Race := { speed_ratio := 4, head_start := 69 }
  race_length r = 92 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_length_for_given_conditions_l965_96505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l965_96502

/-- Given an isosceles triangle with two sides of length 12 and one side of length 15,
    and a similar triangle with longest side 30, the perimeter of the larger triangle is 78. -/
theorem similar_triangle_perimeter
  (small_equal_side : ℝ)
  (small_long_side : ℝ)
  (large_long_side : ℝ)
  (perimeter : ℝ)
  (h1 : small_equal_side = 12)
  (h2 : small_long_side = 15)
  (h3 : large_long_side = 30)
  (h4 : small_long_side > small_equal_side)
  (h5 : perimeter = 2 * (large_long_side * small_equal_side / small_long_side) + large_long_side) :
  perimeter = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l965_96502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_5_6_in_terms_of_a_and_b_l965_96578

-- Define the given conditions
noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.log 10 / Real.log 3

-- State the theorem
theorem log_5_6_in_terms_of_a_and_b : Real.log 6 / Real.log 5 = (a * b + 1) / (b - a * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_5_6_in_terms_of_a_and_b_l965_96578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barbed_wire_cost_theorem_l965_96598

/-- Calculates the cost of drawing barbed wire around a square field -/
noncomputable def barbed_wire_cost (area : ℝ) (gate_width : ℝ) (num_gates : ℕ) (wire_cost_per_meter : ℝ) : ℝ :=
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let wire_length := perimeter - (↑num_gates * gate_width)
  wire_length * wire_cost_per_meter

/-- The cost of drawing barbed wire around a square field with given specifications is 777 -/
theorem barbed_wire_cost_theorem :
  barbed_wire_cost 3136 1 2 3.5 = 777 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barbed_wire_cost_theorem_l965_96598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l965_96597

/-- The length of a train in meters, given its speed in km/h and time in seconds to cross a pole -/
noncomputable def train_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time * 1000 / 3600

/-- Theorem: A train traveling at 70 km/hr that crosses a pole in 36 seconds has a length of 700 meters -/
theorem train_length_calculation :
  train_length 70 36 = 700 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l965_96597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_parameter_product_l965_96529

/-- Given two vectors a and b in R^3 that are parallel, prove that the product of their parameters l and m is 1/10. -/
theorem parallel_vectors_parameter_product (l m : ℝ) :
  let a : Fin 3 → ℝ := ![l + 1, 0, 2*l]
  let b : Fin 3 → ℝ := ![6, 2*m - 1, 2]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  l * m = 1/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_parameter_product_l965_96529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_subinterval_l965_96515

/-- The probability of a randomly chosen real number from [-5,5] being in (0,1) is 1/10 -/
theorem probability_in_subinterval :
  let total_interval : Set ℝ := Set.Icc (-5) 5
  let subinterval : Set ℝ := Set.Ioo 0 1
  (MeasureTheory.volume subinterval) / (MeasureTheory.volume total_interval) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_subinterval_l965_96515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_comparison_l965_96580

structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

def box_lt (a b : Box) : Prop :=
  (a.x ≤ b.x ∧ a.y ≤ b.y ∧ a.z ≤ b.z) ∨
  (a.x ≤ b.x ∧ a.y ≤ b.z ∧ a.z ≤ b.y) ∨
  (a.x ≤ b.y ∧ a.y ≤ b.x ∧ a.z ≤ b.z) ∨
  (a.x ≤ b.y ∧ a.y ≤ b.z ∧ a.z ≤ b.x) ∨
  (a.x ≤ b.z ∧ a.y ≤ b.x ∧ a.z ≤ b.y) ∨
  (a.x ≤ b.z ∧ a.y ≤ b.y ∧ a.z ≤ b.x)

infix:50 " <ᵇ " => box_lt

def box_eq (a b : Box) : Prop :=
  (a.x = b.x ∧ a.y = b.y ∧ a.z = b.z) ∨
  (a.x = b.x ∧ a.y = b.z ∧ a.z = b.y) ∨
  (a.x = b.y ∧ a.y = b.x ∧ a.z = b.z) ∨
  (a.x = b.y ∧ a.y = b.z ∧ a.z = b.x) ∨
  (a.x = b.z ∧ a.y = b.x ∧ a.z = b.y) ∨
  (a.x = b.z ∧ a.y = b.y ∧ a.z = b.x)

infix:50 " =ᵇ " => box_eq

def A : Box := ⟨6, 5, 3⟩
def B : Box := ⟨5, 4, 1⟩
def C : Box := ⟨3, 2, 2⟩

theorem box_comparison :
  (A <ᵇ B) = false ∧ (C <ᵇ A) = true := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_comparison_l965_96580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_triangle_properties_l965_96545

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents an extended triangle with vertices M, N, and P -/
structure ExtendedTriangle where
  M : ℝ × ℝ
  N : ℝ × ℝ
  P : ℝ × ℝ

/-- The center of a triangle -/
noncomputable def TriangleCenter (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateral (t : Triangle) : Prop := sorry

/-- Function to calculate the side length of a triangle -/
noncomputable def SideLength (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is extended by a given length to form another triangle -/
def IsExtendedBy (t1 : Triangle) (t2 : ExtendedTriangle) (length : ℝ) : Prop := sorry

/-- Given an equilateral triangle ABC with side length a and its sides extended by distance a 
    in the same rotational direction to form triangle MNP, prove the following properties -/
theorem extended_triangle_properties 
  (ABC : Triangle) 
  (MNP : ExtendedTriangle) 
  (a : ℝ) 
  (h_equilateral : IsEquilateral ABC)
  (h_side_length : SideLength ABC = a)
  (h_extension : IsExtendedBy ABC MNP a) :
  IsEquilateral (Triangle.mk MNP.M MNP.N MNP.P) ∧ 
  SideLength (Triangle.mk MNP.M MNP.N MNP.P) = a * Real.sqrt 7 ∧
  TriangleCenter ABC = TriangleCenter (Triangle.mk MNP.M MNP.N MNP.P) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_triangle_properties_l965_96545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_postal_service_revenue_l965_96542

/-- Calculate the total revenue from stamp sales --/
theorem postal_service_revenue (colored_price : ℚ) (bw_price : ℚ) (golden_price : ℚ)
  (colored_sold : ℕ) (bw_sold : ℕ) (golden_sold : ℕ) :
  colored_price = 1/2 →
  bw_price = 7/20 →
  golden_price = 2 →
  colored_sold = 578833 →
  bw_sold = 523776 →
  golden_sold = 120456 →
  (colored_price * colored_sold + bw_price * bw_sold + golden_price * golden_sold : ℚ) = 71365010/100 := by
  sorry

#eval (1/2 : ℚ) * 578833 + (7/20 : ℚ) * 523776 + 2 * 120456

end NUMINAMATH_CALUDE_ERRORFEEDBACK_postal_service_revenue_l965_96542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_99_99_less_than_1_99_l965_96584

-- Define the function f
noncomputable def f : ℕ → ℕ → ℝ
| 0, k => 2^k
| m, 0 => 1
| m+1, n+1 => (2 * f m (n+1) * f (m+1) n) / (f m (n+1) + f (m+1) n)

-- State the theorem
theorem f_99_99_less_than_1_99 : f 99 99 < 1.99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_99_99_less_than_1_99_l965_96584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l965_96521

noncomputable def f (x : ℝ) (a : ℝ) (g : ℝ → ℝ) : ℝ :=
  if x ≥ 0 then Real.exp (x * Real.log 3) + a else g x

theorem f_range_theorem (a : ℝ) (g : ℝ → ℝ) :
  (∀ x, f x a g = f (-x) a g) →  -- f is even
  (∀ x y, 0 ≤ x ∧ x < y → f x a g < f y a g) →  -- f is increasing for x ≥ 0
  {x : ℝ | f (x - 1) a g < f 2 a g} = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l965_96521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_eccentricity_l965_96590

/-- Given a hyperbola mx^2 - ny^2 = 1 (m > 0, n > 0) with eccentricity 2,
    the eccentricity of the ellipse mx^2 + ny^2 = 1 is √6/3 -/
theorem hyperbola_ellipse_eccentricity (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∃ (x y : ℝ), m * x^2 - n * y^2 = 1) →
  (Real.sqrt ((1/m + 1/n) / (1/m)) = 2) →
  Real.sqrt ((1/n - 1/m) / (1/n)) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_eccentricity_l965_96590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sum_6_l965_96573

/-- An arithmetic sequence with common difference 2 where a₁, a₃, a₄ form a geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℚ) : Prop :=
  (∀ n, a (n + 1) = a n + 2) ∧ 
  (a 3)^2 = a 1 * a 4

/-- Sum of first n terms of an arithmetic sequence -/
def ArithmeticSum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  n * a 1 + n * (n - 1) / 2 * 2

theorem arithmetic_geometric_sum_6 (a : ℕ → ℚ) :
  ArithmeticGeometricSequence a → ArithmeticSum a 6 = -18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sum_6_l965_96573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_secant_l965_96560

/-- Given a circle with center O and a point P outside the circle, with a secant from P
    intersecting the circle at Q and R, this theorem proves that if OP = 15, PQ = 11, 
    and QR = 5, then the radius of the circle is 7. -/
theorem circle_radius_from_secant (O P Q R : EuclideanSpace ℝ (Fin 2)) (r : ℝ) : 
  let d := 15
  let pq := 11
  let qr := 5
  let pr := pq + qr
  dist O P = d →
  dist P Q = pq →
  dist Q R = qr →
  (∃ c : Set (EuclideanSpace ℝ (Fin 2)), Metric.sphere O r = c ∧ Q ∈ c ∧ R ∈ c) →
  pq * pr = (d - r) * (d + r) →
  r = 7 := by
  sorry

#check circle_radius_from_secant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_secant_l965_96560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_y_axis_l965_96585

-- Define the points A and B
def A : ℝ × ℝ := (-3, -1)
def B : ℝ × ℝ := (2, 3)

-- Define the function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the point on the y-axis
def pointOnYAxis (y : ℝ) : ℝ × ℝ := (0, y)

-- State the theorem
theorem equidistant_point_on_y_axis :
  ∃ y : ℝ, distance (pointOnYAxis y) A = distance (pointOnYAxis y) B ∧ y = 3/8 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_y_axis_l965_96585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_for_quadratic_equation_l965_96582

-- Define the equation
def equation (k : ℤ) (x : ℝ) : ℝ := (k - 1) * x^(k.natAbs + 1) - x + 5

-- Define what it means for the equation to be quadratic
def is_quadratic (k : ℤ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, equation k x = a * x^2 + b * x + c

-- Theorem statement
theorem unique_k_for_quadratic_equation :
  ∃! k : ℤ, is_quadratic k ∧ k = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_for_quadratic_equation_l965_96582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_r_minus_p_for_factorial_nine_l965_96577

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem smallest_r_minus_p_for_factorial_nine (p q r : ℕ) : 
  p * q * r = factorial 9 → 0 < p → p < q → q < r → 
  ∀ (p' q' r' : ℕ), p' * q' * r' = factorial 9 → 0 < p' → p' < q' → q' < r' → 
  r - p ≤ r' - p' :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_r_minus_p_for_factorial_nine_l965_96577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_coordinate_l965_96544

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- Define symmetry_centers function
def symmetry_centers (g : ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∀ x, g (p.1 + x) = g (p.1 - x)}

theorem symmetry_center_coordinate 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : |φ| < π / 2) 
  (h_period : ∀ x, f ω φ (x + 4 * π) = f ω φ x) 
  (h_max : ∀ x, f ω φ x ≤ f ω φ (π / 3)) :
  ∃ y, ((-2 * π / 3, y) : ℝ × ℝ) ∈ symmetry_centers (f ω φ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_coordinate_l965_96544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l965_96533

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Predicate for an acute triangle -/
def is_acute_triangle (p1 p2 p3 : Point) : Prop :=
  (distance p1 p2)^2 + (distance p2 p3)^2 > (distance p1 p3)^2 ∧
  (distance p2 p3)^2 + (distance p1 p3)^2 > (distance p1 p2)^2 ∧
  (distance p1 p2)^2 + (distance p1 p3)^2 > (distance p2 p3)^2

theorem ellipse_eccentricity_half (e : Ellipse) (F1 F2 B : Point)
  (h_foci : distance F1 F2 = 2 * Real.sqrt (e.a^2 - e.b^2))
  (h_minor : distance B ⟨0, 0⟩ = e.b)
  (h_acute : is_acute_triangle F1 F2 B) :
  eccentricity e = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l965_96533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_route_malfunction_l965_96575

/-- Represents a seven-segment display digit -/
structure SevenSegmentDigit where
  segments : Fin 7 → Bool

/-- Represents a three-digit number on a seven-segment display -/
structure ThreeDigitDisplay where
  hundreds : SevenSegmentDigit
  tens : SevenSegmentDigit
  ones : SevenSegmentDigit

/-- Checks if a given ThreeDigitDisplay represents the number 351 -/
def is_351 (display : ThreeDigitDisplay) : Prop :=
  sorry

/-- Checks if a given ThreeDigitDisplay has exactly one segment malfunctioning -/
def has_one_malfunction (display : ThreeDigitDisplay) : Prop :=
  sorry

/-- The set of all possible ThreeDigitDisplays -/
def all_displays : Set ThreeDigitDisplay :=
  sorry

/-- The theorem to be proved -/
theorem bus_route_malfunction :
  ∃ (S : Finset ThreeDigitDisplay),
    (∀ d ∈ S, is_351 d ∧ has_one_malfunction d) ∧
    (∀ d, is_351 d ∧ has_one_malfunction d → d ∈ S) ∧
    S.card = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_route_malfunction_l965_96575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_for_A_l965_96595

theorem exists_k_for_A (n m : ℤ) (hn : n > 1) (hm : m > 1) :
  ∃ k : ℤ, k ≥ 2 ∧ 
    ((n + Real.sqrt (n^2 - 4 : ℝ)) / 2) ^ m = (k + Real.sqrt (k^2 - 4 : ℝ)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_for_A_l965_96595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_problem_l965_96588

open Real MeasureTheory

theorem definite_integral_problem :
  (∫ x in Set.Icc (-1) 1, (2 * Real.sqrt (1 - x^2) - Real.sin x)) = π + 2 * Real.cos 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_problem_l965_96588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l965_96504

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

-- Define the given line
def given_line (a b : ℝ) : Line := { slope := a, intercept := b }

-- Define the line y = 2x - 1
def reference_line : Line := { slope := 2, intercept := -1 }

-- Theorem statement
theorem line_equation (a b : ℝ)
  (h1 : parallel (given_line a b) reference_line) 
  (h2 : (given_line a b).intercept = 3) : 
  given_line a b = { slope := 2, intercept := 3 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l965_96504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_possibilities_l965_96583

-- Define the set S
def S : Finset Int := sorry

-- Define the given seven integers
def given_integers : Finset Int := {1, 5, 7, 11, 13, 18, 21}

-- State the properties of S
axiom S_size : S.card = 11
axiom S_distinct : ∀ x y, x ∈ S → y ∈ S → x ≠ y
axiom S_contains_given : given_integers ⊆ S

-- Define the median of a set
noncomputable def median (T : Finset Int) : Int := sorry

-- Define the set of possible medians
def possible_medians : Finset Int := sorry

-- State the theorem
theorem median_possibilities :
  possible_medians.card = 5 := by
  sorry

#check median_possibilities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_possibilities_l965_96583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_B_independent_l965_96520

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the probability measure
variable (P : Set Ω → ℝ)

-- Define events A and B
def A : Set Ω := {ω : Ω | ω.fst = true}
def B : Set Ω := {ω : Ω | ω.snd = false}

-- Axioms for probability measure
axiom prob_nonneg : ∀ (S : Set Ω), P S ≥ 0
axiom prob_total : P (Set.univ : Set Ω) = 1
axiom prob_additive : ∀ (S T : Set Ω), Disjoint S T → P (S ∪ T) = P S + P T

-- Fairness of coins
axiom fair_coins : P A = 1/2 ∧ P B = 1/2

-- Theorem: A and B are independent
theorem A_B_independent : P (A ∩ B) = P A * P B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_B_independent_l965_96520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abs_sum_solution_l965_96537

/-- The Diophantine equation we're working with -/
def equation (x y : ℤ) : Prop :=
  6 * x^2 + 5 * x * y + y^2 = 6 * x + 2 * y + 7

/-- The absolute value sum we're trying to maximize -/
def abs_sum (x y : ℤ) : ℤ :=
  x.natAbs + y.natAbs

/-- Theorem stating that (-8, 25) is a solution that maximizes |x| + |y| -/
theorem max_abs_sum_solution :
  equation (-8) 25 ∧
  ∀ x y : ℤ, equation x y → abs_sum x y ≤ abs_sum (-8) 25 := by
  sorry

#eval abs_sum (-8) 25  -- This will evaluate to 33

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abs_sum_solution_l965_96537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l965_96567

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < Real.pi/2) (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.tan α = 3) (h4 : Real.sin (α + β) = 3/5) : Real.cos β = Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l965_96567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_bound_l965_96519

open Real

theorem function_inequality_implies_a_bound
  (f : ℝ → ℝ)
  (a : ℝ)
  (h_f : ∀ x, f x = x + a * log x)
  (h_a : a > 0)
  (h_ineq : ∀ x₁ x₂, x₁ ∈ Set.Ioo (1/2 : ℝ) 1 → x₂ ∈ Set.Ioo (1/2 : ℝ) 1 → x₁ ≠ x₂ →
            |f x₁ - f x₂| > |1/x₁ - 1/x₂|) :
  a ≥ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_bound_l965_96519

import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_pairs_sum_50_l251_25193

theorem prime_pairs_sum_50 : 
  ∃! n : ℕ, n = (Finset.filter (λ p : ℕ × ℕ => Nat.Prime p.1 ∧ Nat.Prime p.2 ∧ p.1 + p.2 = 50 ∧ p.1 ≤ p.2) (Finset.product (Finset.range 51) (Finset.range 51))).card ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_pairs_sum_50_l251_25193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l251_25111

/-- Calculates the speed of a train given its length and time to cross a point. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem: The speed of a train that is 3000 meters long and crosses an electric pole in 120 seconds is 25 meters per second. -/
theorem train_speed_theorem :
  train_speed 3000 120 = 25 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l251_25111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_louies_pie_share_l251_25141

theorem louies_pie_share (total_pie : ℚ) (louie_share dewey_share huey_share : ℚ) : 
  total_pie = 8/9 →
  louie_share = 2 * dewey_share →
  louie_share = 2 * huey_share →
  dewey_share = huey_share →
  total_pie = louie_share + dewey_share + huey_share →
  louie_share = 4/9 := by
  intros h1 h2 h3 h4 h5
  -- Proof steps would go here
  sorry

#check louies_pie_share

end NUMINAMATH_CALUDE_ERRORFEEDBACK_louies_pie_share_l251_25141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_solid_sum_of_edges_l251_25196

/-- Represents a rectangular solid with dimensions in geometric progression -/
structure GeometricProgressionSolid where
  a : ℝ
  r : ℝ

/-- The volume of the solid -/
noncomputable def volume (s : GeometricProgressionSolid) : ℝ :=
  s.a * s.a * s.a

/-- The surface area of the solid -/
noncomputable def surfaceArea (s : GeometricProgressionSolid) : ℝ :=
  2 * (s.a^2 / s.r + s.a^2 + s.a^2 * s.r)

/-- The sum of the lengths of all edges of the solid -/
noncomputable def sumOfEdges (s : GeometricProgressionSolid) : ℝ :=
  4 * (s.a / s.r + s.a + s.a * s.r)

/-- Theorem: For a rectangular solid with volume 512 cm³, surface area 384 cm², 
    and dimensions in geometric progression, the sum of all edge lengths is 112 cm -/
theorem geometric_progression_solid_sum_of_edges :
  ∃ s : GeometricProgressionSolid,
    volume s = 512 ∧
    surfaceArea s = 384 ∧
    sumOfEdges s = 112 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_solid_sum_of_edges_l251_25196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_braking_problem_l251_25161

-- Define the initial speed in m/s
noncomputable def initial_speed : ℝ := 108 * 1000 / 3600

-- Define the acceleration (deceleration) in m/s²
def acceleration : ℝ := -0.5

-- Define the time to stop in seconds
def time_to_stop : ℝ := 60

-- Define the distance covered in meters
def distance_covered : ℝ := 900

-- Theorem statement
theorem train_braking_problem :
  initial_speed + acceleration * time_to_stop = 0 ∧
  distance_covered = initial_speed * time_to_stop + 
    (1/2) * acceleration * time_to_stop^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_braking_problem_l251_25161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l251_25113

def sequenceA (n : ℕ+) : ℝ := 3 * n.val - 1

theorem sequence_formula (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, S n > 1) →
  (∀ n, 6 * S n = (a n + 1) * (a n + 2)) →
  a = sequenceA :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l251_25113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l251_25105

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side length opposite to A
  b : ℝ  -- Side length opposite to B
  c : ℝ  -- Side length opposite to C
  is_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the theorem
theorem triangle_side_length (t : Triangle)
  (h1 : Real.tan t.A = 2 * Real.tan t.B)
  (h2 : t.a^2 - t.b^2 = (1/3) * t.c) :
  t.c = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l251_25105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pituitary_secretes_growth_hormone_other_glands_dont_secrete_growth_hormone_l251_25191

structure Gland where
  name : String
  secretes_growth_hormone : Bool

def pancreas : Gland := { name := "Pancreas", secretes_growth_hormone := false }
def thyroid : Gland := { name := "Thyroid", secretes_growth_hormone := false }
def pituitary : Gland := { name := "Pituitary", secretes_growth_hormone := true }
def salivary : Gland := { name := "Salivary gland", secretes_growth_hormone := false }

def correct_answer (g : Gland) : Prop :=
  g.secretes_growth_hormone = true

theorem pituitary_secretes_growth_hormone :
  correct_answer pituitary := by
  simp [correct_answer, pituitary]

theorem other_glands_dont_secrete_growth_hormone :
  ¬ (correct_answer pancreas ∨ correct_answer thyroid ∨ correct_answer salivary) := by
  simp [correct_answer, pancreas, thyroid, salivary]
  
#check pituitary_secretes_growth_hormone
#check other_glands_dont_secrete_growth_hormone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pituitary_secretes_growth_hormone_other_glands_dont_secrete_growth_hormone_l251_25191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_90_l251_25131

/-- The speed of a train in km/hr -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: The speed of the train is 90 km/hr -/
theorem train_speed_is_90 (length : ℝ) (time : ℝ)
  (h1 : length = 300) -- The length of the train is 300 meters
  (h2 : time = 12)    -- The train crosses a pole in 12 seconds
  : train_speed length time = 90 := by
  sorry

#check train_speed_is_90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_90_l251_25131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l251_25173

/-- The area of a quadrilateral with a diagonal of length d and offsets h1 and h2 -/
noncomputable def quadrilateralArea (d h1 h2 : ℝ) : ℝ := (1/2 * d * h1) + (1/2 * d * h2)

/-- Theorem: The area of a quadrilateral with diagonal 30 and offsets 9 and 6 is 225 -/
theorem quadrilateral_area_example : quadrilateralArea 30 9 6 = 225 := by
  -- Unfold the definition of quadrilateralArea
  unfold quadrilateralArea
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num

-- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l251_25173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_volume_and_icing_area_is_21_l251_25151

/-- Represents a triangular piece cut from a cube --/
structure TriangularPiece where
  edge_length : ℝ
  volume : ℝ
  icing_area : ℝ

/-- Calculates the volume and icing area of the triangular piece --/
noncomputable def calculate_triangular_piece (edge_length : ℝ) : TriangularPiece :=
  { edge_length := edge_length
  , volume := (9 / 4) * edge_length
  , icing_area := (9 / 4) + 4 * edge_length }

/-- Theorem stating that the sum of volume and icing area is 21 for a cube with edge length 3 --/
theorem sum_volume_and_icing_area_is_21 :
  let piece := calculate_triangular_piece 3
  piece.volume + piece.icing_area = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_volume_and_icing_area_is_21_l251_25151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payback_period_is_165_days_l251_25142

/-- Represents the mining setup and market conditions for Ethereum mining -/
structure MiningSetup where
  system_unit_cost : ℚ
  graphics_card_cost : ℚ
  num_graphics_cards : ℕ
  system_unit_power : ℚ
  graphics_card_power : ℚ
  ethereum_per_card_per_day : ℚ
  ethereum_to_rubles : ℚ
  electricity_cost_per_kwh : ℚ

/-- Calculates the number of days required for the investment to pay off -/
noncomputable def payback_period (setup : MiningSetup) : ℚ :=
  let total_cost := setup.system_unit_cost + setup.num_graphics_cards * setup.graphics_card_cost
  let daily_ethereum := setup.num_graphics_cards * setup.ethereum_per_card_per_day
  let daily_earnings := daily_ethereum * setup.ethereum_to_rubles
  let total_power := setup.system_unit_power + setup.num_graphics_cards * setup.graphics_card_power
  let daily_power_kwh := total_power * 24 / 1000
  let daily_electricity_cost := daily_power_kwh * setup.electricity_cost_per_kwh
  let daily_profit := daily_earnings - daily_electricity_cost
  total_cost / daily_profit

/-- Theorem stating that the payback period for the given setup is approximately 165 days -/
theorem payback_period_is_165_days :
  let setup := MiningSetup.mk 9499 31431 2 120 125 (877/100000) 27790.37 5.38
  ⌈(payback_period setup : ℚ)⌉ = 165 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_payback_period_is_165_days_l251_25142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l251_25152

noncomputable def f (x : ℝ) := 4 * Real.sin (2 * x + Real.pi / 2)

theorem f_properties :
  (∀ x : ℝ, f x = 4 * Real.cos (2 * x - Real.pi / 2)) ∧
  (∀ x : ℝ, f (-Real.pi / 4 + x) = f (-Real.pi / 4 - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l251_25152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l251_25123

-- Define the line
def line (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 4 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 25

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ circle_eq A.1 A.2 ∧
  line B.1 B.2 ∧ circle_eq B.1 B.2 ∧
  A ≠ B

-- Define point P on the circle
def point_on_circle (P : ℝ × ℝ) : Prop :=
  circle_eq P.1 P.2

-- Define P as distinct from A and B
def P_distinct (P A B : ℝ × ℝ) : Prop :=
  P ≠ A ∧ P ≠ B

-- Define the area of a triangle
noncomputable def area_triangle (A B P : ℝ × ℝ) : ℝ :=
  sorry -- Placeholder for the actual area calculation

-- Theorem statement
theorem max_area_triangle (A B P : ℝ × ℝ) :
  intersection_points A B →
  point_on_circle P →
  P_distinct P A B →
  ∃ (max_area : ℝ), max_area = 32 ∧
    ∀ (Q : ℝ × ℝ), point_on_circle Q → P_distinct Q A B →
      area_triangle A B Q ≤ max_area :=
by
  sorry -- Placeholder for the proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l251_25123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l251_25194

theorem tan_difference (α β : ℝ) (h1 : Real.tan α = 9) (h2 : Real.tan β = 5) :
  Real.tan (α - β) = 2 / 23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l251_25194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_fraction_sum_l251_25118

theorem coprime_fraction_sum (N a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c) →
  ((1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = N / (a + b + c)) ↔ (N = 9 ∨ N = 10 ∨ N = 11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_fraction_sum_l251_25118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_intersection_iff_product_equality_l251_25199

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the circle
variable (Circle : Type)

-- Define the center of a circle
variable (center : Circle → Point)

-- Define the midpoint of a line segment
variable (midpoint : Point → Point → Point)

-- Define the property of a quadrilateral circumscribing a circle
variable (circumscribes : Point → Point → Point → Point → Circle → Prop)

-- Define the property of a point lying on a line
variable (on_line : Point → Line → Prop)

-- Define the intersection of two lines
variable (intersection : Line → Line → Point)

-- Define the line passing through two points
variable (line_through : Point → Point → Line)

-- Define the distance between two points
variable (distance : Point → Point → ℝ)

-- Define the property of being a parallelogram
variable (is_parallelogram : Point → Point → Point → Point → Prop)

-- State the theorem
theorem midpoint_intersection_iff_product_equality
  (A B C D O : Point) (circle : Circle) :
  ¬ is_parallelogram A B C D →
  circumscribes A B C D circle →
  O = center circle →
  (O = intersection (line_through (midpoint A B) (midpoint C D)) 
                    (line_through (midpoint A D) (midpoint B C))) ↔
  (distance O A * distance O C = distance O B * distance O D) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_intersection_iff_product_equality_l251_25199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_reflected_arcs_area_l251_25168

/-- The area of the region bounded by 6 reflected arcs of a circle with an inscribed regular hexagon of side length 1 -/
theorem hexagon_reflected_arcs_area : 
  ∃ (r : ℝ) (hexagon_area arc_area : ℝ),
    -- The radius of the circle
    r = Real.sqrt 2 / 3 ∧ 
    -- The area of the regular hexagon
    hexagon_area = 3 * Real.sqrt 3 / 2 ∧ 
    -- The area of each reflected arc
    arc_area = 4 * Real.pi / 27 - Real.sqrt 3 / 4 ∧ 
    -- The total area of the bounded region
    hexagon_area - 6 * arc_area = 3 * Real.sqrt 3 - 8 * Real.pi / 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_reflected_arcs_area_l251_25168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_coordinate_l251_25136

/-- The centroid of a triangle with vertices (x₁, y₁), (x₂, y₂), (x₃, y₃) -/
noncomputable def centroid (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ × ℝ :=
  ((x₁ + x₂ + x₃) / 3, (y₁ + y₂ + y₃) / 3)

/-- Theorem: The centroid of a triangle is located at the average of its vertices' coordinates -/
theorem centroid_coordinate (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  centroid x₁ y₁ x₂ y₂ x₃ y₃ = ((x₁ + x₂ + x₃) / 3, (y₁ + y₂ + y₃) / 3) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_coordinate_l251_25136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_hemisphere_radius_l251_25124

/-- The radius of a sphere that transforms into a hemisphere with radius 4∛2 cm while maintaining its volume -/
noncomputable def original_sphere_radius : ℝ := 2 * Real.rpow 2 (1/3)

/-- The radius of the hemisphere formed after the transformation -/
noncomputable def hemisphere_radius : ℝ := 4 * Real.rpow 2 (1/3)

theorem sphere_to_hemisphere_radius :
  (4/3 * Real.pi * original_sphere_radius^3 = 2/3 * Real.pi * hemisphere_radius^3) →
  original_sphere_radius = 2 * Real.rpow 2 (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_hemisphere_radius_l251_25124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l251_25189

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Divisibility by 3 -/
def divisible_by_three (n : ℤ) : Prop := ∃ m : ℤ, n = 3 * m

theorem polynomial_divisibility 
  (f : IntPolynomial) 
  (k : ℤ) 
  (h1 : divisible_by_three (f.eval k))
  (h2 : divisible_by_three (f.eval (k + 1)))
  (h3 : divisible_by_three (f.eval (k + 2))) :
  ∀ m : ℤ, divisible_by_three (f.eval m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l251_25189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_odd_function_theorem_l251_25188

-- Define the exponential function g
noncomputable def g : ℝ → ℝ := fun x ↦ 2^x

-- Define the function f
noncomputable def f (m n : ℝ) : ℝ → ℝ := fun x ↦ (-g x + n) / (2 * g x + m)

-- Main theorem
theorem exponential_odd_function_theorem (h1 : g 2 = 4)
  (h2 : ∀ x, f 2 1 x = -f 2 1 (-x)) :
  (∃ m n : ℝ, m = 2 ∧ n = 1) ∧
  (∀ t k : ℝ, f 2 1 (t^2 - 2*t) + f 2 1 (2*t^2 - k) > 0 → k > -1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_odd_function_theorem_l251_25188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l251_25100

theorem cube_root_sum_equals_one :
  (5 + 2 * Real.sqrt 13) ^ (1/3 : ℝ) + (5 - 2 * Real.sqrt 13) ^ (1/3 : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l251_25100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milkman_problem_l251_25167

/-- The amount of pure milk the milkman started with -/
def pure_milk_amount : ℝ → Prop := λ x => x > 0

/-- The cost of pure milk per litre -/
def pure_milk_cost : ℝ := 3.60

/-- The selling price of the mixture per litre -/
def mixture_price : ℝ := 3

/-- The amount of water added to the pure milk -/
def water_added : ℝ := 5

theorem milkman_problem (x : ℝ) :
  pure_milk_amount x →
  pure_milk_cost * x = mixture_price * (x + water_added) →
  x = 25 := by
    intro h1 h2
    -- The proof steps would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milkman_problem_l251_25167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_six_div_eighteen_equals_one_div_cube_root_three_l251_25138

theorem cube_root_six_div_eighteen_equals_one_div_cube_root_three :
  (6 / 18 : ℝ) ^ (1/3 : ℝ) = 1 / (3 : ℝ)^(1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_six_div_eighteen_equals_one_div_cube_root_three_l251_25138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l251_25143

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the second derivative of f
def f'' : ℝ → ℝ := sorry

-- Axiom for the first condition
axiom condition1 : ∀ x : ℝ, f x + f'' x > 1

-- Axiom for the second condition
axiom condition2 : f 1 = 0

-- Define the solution set
def solution_set : Set ℝ := Set.Iic 1

-- Theorem statement
theorem inequality_solution :
  ∀ x : ℝ, x ∈ solution_set ↔ f x - 1 + (Real.exp (x - 1))⁻¹ ≤ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l251_25143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l251_25135

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  hd : d ≠ 0 -- d is not zero
  ha : ∀ n, a (n + 1) = a n + d  -- Arithmetic sequence property

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + (n - 1) / 2 * seq.d)

theorem arithmetic_sequence_property
  (seq : ArithmeticSequence)
  (h1 : S seq 3 = 9)
  (h2 : ∃ r, seq.a 5 = seq.a 3 * r ∧ seq.a 8 = seq.a 5 * r) :
  seq.d = 1 ∧ ∀ n, S seq n = (n^2 + 3*n) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l251_25135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l251_25154

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (1/2) * Real.log x

theorem a_range (a : ℝ) :
  (∀ x y, 1 ≤ x ∧ x < y → f a y ≤ f a x) ∧
  (∃ x > 0, (2:ℝ)^x * (x - a) < 1) →
  -1 < a ∧ a ≤ -1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l251_25154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_tank_used_is_five_twelfths_l251_25148

/-- Represents the characteristics and journey of a car -/
structure Car where
  speed : ℚ  -- Speed in miles per hour
  fuelEfficiency : ℚ  -- Miles per gallon
  tankCapacity : ℚ  -- Capacity in gallons
  travelTime : ℚ  -- Travel time in hours

/-- Calculates the fraction of a full tank used by a car during its journey -/
def fractionOfTankUsed (c : Car) : ℚ :=
  (c.speed * c.travelTime) / (c.fuelEfficiency * c.tankCapacity)

/-- Theorem stating that for a car with given characteristics, 
    the fraction of tank used is 5/12 -/
theorem fraction_of_tank_used_is_five_twelfths :
  let c : Car := {
    speed := 50,
    fuelEfficiency := 30,
    tankCapacity := 20,
    travelTime := 5
  }
  fractionOfTankUsed c = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_tank_used_is_five_twelfths_l251_25148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_shape_l251_25175

noncomputable section

-- Define the bounds of integration
def lower_bound : ℝ := 1
def upper_bound : ℝ := 2

-- Define the function representing the curve xy=1
def f (y : ℝ) : ℝ := 1 / y

-- State the theorem
theorem area_enclosed_shape :
  (∫ y in lower_bound..upper_bound, f y) = Real.log 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_shape_l251_25175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l251_25137

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

-- State the theorem
theorem sin_graph_shift :
  ∀ x : ℝ, f x = g (x + Real.pi / 8) :=
by
  intro x
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l251_25137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_properties_l251_25184

/-- A regular octagon inscribed in a circle -/
structure RegularOctagonInCircle where
  /-- The side length of the octagon -/
  side_length : ℝ
  /-- Side length is positive -/
  side_length_pos : side_length > 0

/-- Calculate the arc length corresponding to one side of the octagon -/
noncomputable def arc_length (octagon : RegularOctagonInCircle) : ℝ :=
  (Real.pi / 4) * octagon.side_length

/-- Calculate the area of one triangular sector -/
noncomputable def triangular_sector_area (octagon : RegularOctagonInCircle) : ℝ :=
  (Real.pi / 32) * octagon.side_length^2

/-- Theorem stating the correct arc length and triangular sector area for a regular octagon with side length 5 -/
theorem octagon_properties :
  ∃ (octagon : RegularOctagonInCircle),
    octagon.side_length = 5 ∧
    arc_length octagon = 5 * Real.pi / 4 ∧
    triangular_sector_area octagon = 25 * Real.pi / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_properties_l251_25184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_2_eq_neg_3_l251_25125

/-- The function g defined on real numbers -/
noncomputable def g (x y : ℝ) : ℝ := (x^3 - 3*x^2*y + x*y^2) / (x^2 - y^2)

/-- Theorem stating that g(3, 2) = -3 -/
theorem g_3_2_eq_neg_3 : g 3 2 = -3 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the numerator and denominator
  simp [pow_two, pow_three]
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_2_eq_neg_3_l251_25125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_piles_for_660_stones_l251_25132

/-- Represents the stone splitting game -/
structure StoneSplittingGame where
  totalStones : ℕ
  maxPiles : ℕ

/-- Checks if a list of pile sizes is valid according to the game rules -/
def isValidDistribution (piles : List ℕ) : Prop :=
  ∀ i j, i < piles.length → j < piles.length → 2 * piles[i]! > piles[j]!

/-- The main theorem stating the maximum number of piles for 660 stones -/
theorem max_piles_for_660_stones :
  ∃ (game : StoneSplittingGame) (distribution : List ℕ),
    game.totalStones = 660 ∧
    game.maxPiles = 30 ∧
    distribution.length = game.maxPiles ∧
    distribution.sum = game.totalStones ∧
    isValidDistribution distribution ∧
    ∀ (otherGame : StoneSplittingGame) (otherDist : List ℕ),
      otherGame.totalStones = 660 →
      otherDist.sum = otherGame.totalStones →
      isValidDistribution otherDist →
      otherDist.length ≤ game.maxPiles :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_piles_for_660_stones_l251_25132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cell_division_l251_25171

theorem cell_division (x : ℕ) : ∃ y : ℕ, y = 2^x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cell_division_l251_25171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_of_ellipse_l251_25157

/-- The distance between the foci of an ellipse -/
noncomputable def distance_between_foci (a b c d : ℝ) : ℝ :=
  Real.sqrt ((a + c)^2 + (b - d)^2)

/-- Definition of the ellipse equation -/
def is_ellipse (x y : ℝ) : Prop :=
  Real.sqrt ((2*x - 6)^2 + (3*y + 12)^2) + Real.sqrt ((2*x + 10)^2 + (3*y - 24)^2) = 30

/-- Theorem stating the distance between foci of the given ellipse -/
theorem distance_between_foci_of_ellipse :
  distance_between_foci (3/2) (-4) (-5/2) 8 = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_of_ellipse_l251_25157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shelves_for_five_books_together_no_larger_k_works_l251_25104

/-- The total number of books --/
def total_books : ℕ := 1300

/-- The number of bookshelves --/
def num_shelves : ℕ := 18

/-- The minimum number of books that must remain together --/
def min_together : ℕ := 5

theorem max_shelves_for_five_books_together :
  ∀ (arrangement : Fin total_books → Fin num_shelves),
  ∃ (shelf : Fin num_shelves) (books : Finset (Fin total_books)),
    books.card ≥ min_together ∧
    (∀ book ∈ books, arrangement book = shelf) :=
by sorry

theorem no_larger_k_works (k : ℕ) (h : k > num_shelves) :
  ∃ (arrangement : Fin total_books → Fin k),
  ∀ (shelf : Fin k) (books : Finset (Fin total_books)),
    books.card ≥ min_together →
    ∃ book ∈ books, arrangement book ≠ shelf :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shelves_for_five_books_together_no_larger_k_works_l251_25104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_l251_25128

theorem sin_cos_sum (α : ℝ) 
  (h1 : Real.sin (2 * α) = 3 / 4)
  (h2 : π < α)
  (h3 : α < 3 * π / 2) :
  Real.sin α + Real.cos α = -Real.sqrt 7 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_l251_25128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_y_axis_l251_25187

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + Real.log x

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1 / x

-- Theorem statement
theorem tangent_perpendicular_to_y_axis (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ ¬ ∃ y : ℝ, |f_derivative a x| ≤ y) → a < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_y_axis_l251_25187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_selection_sum_l251_25134

def red_balls : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 100) (Finset.range 101)
def blue_balls : Finset ℕ := Finset.filter (λ n => 101 ≤ n ∧ n ≤ 200) (Finset.range 201)

def valid_selection (a b : ℕ) : Prop :=
  a ≤ 100 ∧ b ≤ 100 ∧
  ∀ (S : Finset ℕ) (T : Finset ℕ),
    S ⊆ red_balls ∧ T ⊆ blue_balls ∧ S.card = a ∧ T.card = b →
    ∃ (x y : ℕ) (z : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ T ∧ x + y = z

theorem min_selection_sum :
  ∃ (a b : ℕ), valid_selection a b ∧ a + b = 115 ∧
  ∀ (c d : ℕ), valid_selection c d → c + d ≥ 115 :=
by
  -- The proof goes here
  sorry

#check min_selection_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_selection_sum_l251_25134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l251_25145

theorem function_property (f : ℤ → ℤ) :
  (∀ m n : ℤ, f (f (m + n)) = f m + f n) →
  (∃ a : ℤ, ∀ n : ℤ, f n = n + a) ∨ (∀ n : ℤ, f n = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l251_25145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_AB_l251_25144

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-1, 0)

-- Define a line passing through a point
def line_through (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  ∃ k : ℝ, y - p.2 = k * (x - p.1)

-- Define perpendicularity of two vectors
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Main theorem
theorem distance_to_line_AB : 
  ∃ A B : ℝ × ℝ,
  ellipse A.1 A.2 ∧
  ellipse B.1 B.2 ∧
  line_through left_focus A.1 A.2 ∧
  line_through left_focus B.1 B.2 ∧
  perpendicular A B ∧
  (∃ l : ℝ × ℝ → ℝ, l (0, 0) = Real.sqrt 6 / 3 ∧ 
    ∀ x y : ℝ, line_through A x y ∧ line_through B x y → l (x, y) = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_AB_l251_25144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_mean_after_removal_l251_25102

/-- The arithmetic mean of a list of numbers -/
noncomputable def arithmeticMean (numbers : List ℝ) : ℝ :=
  numbers.sum / numbers.length

theorem new_mean_after_removal (originalSet : List ℝ) : 
  originalSet.length = 60 → 
  arithmeticMean originalSet = 42 →
  48 ∈ originalSet →
  52 ∈ originalSet →
  60 ∈ originalSet →
  let newSet := originalSet.filter (λ x => x ≠ 48 ∧ x ≠ 52 ∧ x ≠ 60)
  abs (arithmeticMean newSet - 41.404) < 0.001 := by
  sorry

#check new_mean_after_removal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_mean_after_removal_l251_25102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_hours_worked_l251_25179

/-- Represents the compensation structure and work hours of a bus driver -/
structure BusDriverCompensation where
  regularRate : ℚ
  regularHours : ℚ
  overtimeRate : ℚ
  totalCompensation : ℚ

/-- Calculates the total hours worked by a bus driver given their compensation structure -/
def totalHoursWorked (bdc : BusDriverCompensation) : ℚ :=
  bdc.regularHours + (bdc.totalCompensation - bdc.regularRate * bdc.regularHours) / bdc.overtimeRate

/-- Theorem stating that the bus driver worked 58 hours given the specified conditions -/
theorem bus_driver_hours_worked :
  let bdc : BusDriverCompensation := {
    regularRate := 14,
    regularHours := 40,
    overtimeRate := 14 * (1 + 3/4),
    totalCompensation := 998
  }
  totalHoursWorked bdc = 58 := by
    -- The proof goes here
    sorry

#eval totalHoursWorked {
  regularRate := 14,
  regularHours := 40,
  overtimeRate := 14 * (1 + 3/4),
  totalCompensation := 998
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_hours_worked_l251_25179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l251_25120

/-- Given a function f : ℝ → ℝ satisfying f(x+1) = x^2 - 3x + 2 for all x,
    prove that f(x) = x^2 - 5x + 6 for all x. -/
theorem function_equality (f : ℝ → ℝ) 
    (h : ∀ x, f (x + 1) = x^2 - 3*x + 2) : 
    ∀ x, f x = x^2 - 5*x + 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l251_25120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_has_infinite_perpendicular_tangents_l251_25110

open Real

-- Define the functions
noncomputable def f₁ (x : ℝ) := Real.exp x
def f₂ (x : ℝ) := x^3
noncomputable def f₃ (x : ℝ) := Real.log x
noncomputable def f₄ (x : ℝ) := Real.sin x

-- Define the derivative of each function
noncomputable def f₁' (x : ℝ) := Real.exp x
def f₂' (x : ℝ) := 3 * x^2
noncomputable def f₃' (x : ℝ) := 1 / x
noncomputable def f₄' (x : ℝ) := Real.cos x

-- Define the property of having infinitely many perpendicular tangents
def has_infinite_perpendicular_tangents (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∃ S : Set (ℝ × ℝ), Set.Infinite S ∧ ∀ (x₁ x₂ : ℝ), (x₁, x₂) ∈ S → f' x₁ * f' x₂ = -1

-- State the theorem
theorem sin_has_infinite_perpendicular_tangents :
  has_infinite_perpendicular_tangents f₄ f₄' ∧
  ¬has_infinite_perpendicular_tangents f₁ f₁' ∧
  ¬has_infinite_perpendicular_tangents f₂ f₂' ∧
  ¬has_infinite_perpendicular_tangents f₃ f₃' :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_has_infinite_perpendicular_tangents_l251_25110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l251_25149

-- Define the function f(x) = (x + 2) / x
noncomputable def f (x : ℝ) : ℝ := (x + 2) / x

-- Theorem statement
theorem f_properties :
  -- 1. Domain is all real numbers except 0
  (∀ x : ℝ, x ≠ 0 → f x ∈ Set.univ) ∧
  -- 2. Range is all real numbers except 1
  (∀ y : ℝ, y ≠ 1 → ∃ x : ℝ, f x = y) ∧
  -- 3. f is strictly decreasing on (0, +∞)
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ > f x₂) ∧
  -- 4. Range on [2, 8] is [5/4, 2]
  (∀ x : ℝ, 2 ≤ x → x ≤ 8 → 5/4 ≤ f x ∧ f x ≤ 2) ∧
  (f 8 = 5/4) ∧ (f 2 = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l251_25149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_same_color_points_with_min_distance_l251_25174

-- Define the color type
inductive Color
| Red
| Yellow
| Green
| Blue

-- Define the triangle type
structure RightIsoscelesTriangle where
  -- The legs have length 1
  leg_length : ℝ
  leg_length_eq : leg_length = 1

-- Define a point in the triangle
structure Point where
  x : ℝ
  y : ℝ
  in_triangle : x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 1

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Theorem statement
theorem exists_same_color_points_with_min_distance 
  (t : RightIsoscelesTriangle) (c : Point → Color) :
  ∃ (p1 p2 : Point), c p1 = c p2 ∧ distance p1 p2 ≥ 2 - Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_same_color_points_with_min_distance_l251_25174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_valid_numbers_l251_25177

def is_valid_number (n : ℕ) : Prop :=
  let digits := [1, 2, 3, 4, 5, 6]
  (∃ a b c d e f : ℕ, n = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f
    ∧ a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧ f ∈ digits
    ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f
    ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f
    ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f
    ∧ d ≠ e ∧ d ≠ f
    ∧ e ≠ f)
  ∧ n % 6 = 0
  ∧ (n / 10) % 5 = 0
  ∧ (n / 100) % 4 = 0
  ∧ (n / 1000) % 3 = 0
  ∧ (n / 10000) % 2 = 0

theorem exactly_two_valid_numbers :
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ n ∈ s, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_valid_numbers_l251_25177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l251_25158

/-- The length of a platform given a train's characteristics and crossing time --/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmph = 52 →
  crossing_time = 30 →
  ∃ platform_length : ℝ,
    platform_length = (train_speed_kmph * 1000 / 3600 * crossing_time) - train_length ∧
    abs (platform_length - 323.2) < 0.1 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l251_25158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convert_105_degrees_to_radians_l251_25163

/-- Converts degrees to radians -/
noncomputable def degrees_to_radians (degrees : ℝ) : ℝ := degrees * (Real.pi / 180)

theorem convert_105_degrees_to_radians :
  degrees_to_radians 105 = (7 * Real.pi) / 12 := by
  -- Unfold the definition of degrees_to_radians
  unfold degrees_to_radians
  -- Simplify the expression
  simp [mul_div_assoc, mul_comm, mul_assoc]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convert_105_degrees_to_radians_l251_25163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_in_square_min_area_l251_25150

open Real

/-- The area not covered by a circle of diameter x after one full revolution inside a unit square -/
noncomputable def S (x : ℝ) : ℝ :=
  if x < 1/4 then
    (20 - π) * (x - 4 / (20 - π))^2 + (4 - π) / (20 - π)
  else
    (4 - π) * x^2

/-- The value of x that minimizes S -/
noncomputable def x_min : ℝ := 4 / (20 - π)

theorem circle_in_square_min_area :
  ∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → S x ≥ S x_min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_in_square_min_area_l251_25150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_body_volume_l251_25117

/-- A convex body formed by planes on the edges of a unit cube -/
structure CustomConvexBody where
  /-- The underlying cube is a unit cube -/
  is_unit_cube : Bool
  /-- There are 12 planes, one for each edge of the cube -/
  num_planes : Nat
  /-- Each plane forms a 45° angle with adjacent faces -/
  plane_angle : ℝ
  /-- The planes do not intersect the cube -/
  no_intersection : Bool

/-- The volume of the convex body -/
noncomputable def volume (body : CustomConvexBody) : ℝ :=
  sorry

/-- Theorem stating that the volume of the described convex body is 2 -/
theorem convex_body_volume (body : CustomConvexBody) :
    body.is_unit_cube ∧
    body.num_planes = 12 ∧
    body.plane_angle = Real.pi / 4 ∧
    body.no_intersection
    → volume body = 2 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_body_volume_l251_25117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_general_term_l251_25139

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  property1 : a 3 * a 7 = -16
  property2 : a 4 + a 6 = 0

/-- The general term formula for the arithmetic sequence -/
def general_term (n : ℕ) (x : ℝ) : Prop :=
  x = 2 * n - 10 ∨ x = -2 * n + 10

theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∃ f : ℕ → ℝ, (∀ n, general_term n (f n)) ∧ (∀ n, seq.a n = f n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_general_term_l251_25139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_zero_f_leq_g_iff_m_geq_one_l251_25115

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x + 1
noncomputable def g (x : ℝ) : ℝ := x * (Real.exp x - 2)

-- Part 1: Maximum value of f is 0 when m = 1
theorem max_value_f_zero (x : ℝ) (hx : x > 0) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (y : ℝ) (hy : y > 0), f m y ≤ f m x :=
by sorry

-- Part 2: f(x) ≤ g(x) for all x > 0 iff m ≥ 1
theorem f_leq_g_iff_m_geq_one :
  ∀ (m : ℝ), (∀ (x : ℝ) (hx : x > 0), f m x ≤ g x) ↔ m ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_zero_f_leq_g_iff_m_geq_one_l251_25115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_union_N_equals_nonnegative_reals_l251_25159

-- Define the set of real numbers
variable (r : Set ℝ)

-- Define set M
def M : Set ℝ := {x : ℝ | ∃ y, y = Real.log (1 - 2/x)}

-- Define set N
def N : Set ℝ := {x : ℝ | x ≥ 1}

-- State the theorem
theorem complement_M_union_N_equals_nonnegative_reals :
  (Set.univ \ M) ∪ N = Set.Ici (0 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_union_N_equals_nonnegative_reals_l251_25159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l251_25160

/-- The eccentricity of an ellipse with the given properties -/
theorem ellipse_eccentricity (a b : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧
  ((P.1 - F₂.1) * (F₁.1 - F₂.1) + (P.2 - F₂.2) * (F₁.2 - F₂.2) = 0) ∧
  (Real.cos (Real.pi/6) * (P.1 - F₁.1) = Real.sin (Real.pi/6) * (P.2 - F₁.2)) →
  Real.sqrt (a^2 - b^2) / a = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l251_25160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_percentage_rounded_to_23_percent_l251_25127

/-- Represents the duration of a work day in hours -/
def work_day_hours : ℚ := 10

/-- Represents the duration of the first meeting in minutes -/
def meeting1_minutes : ℚ := 35

/-- Represents the duration of the second meeting in minutes -/
def meeting2_minutes : ℚ := 105

/-- Calculates the percentage of work day spent in meetings -/
noncomputable def meeting_percentage : ℚ :=
  let work_day_minutes := work_day_hours * 60
  let total_meeting_minutes := meeting1_minutes + meeting2_minutes
  (total_meeting_minutes / work_day_minutes) * 100

theorem meeting_percentage_rounded_to_23_percent :
  Int.floor (meeting_percentage + 0.5) = 23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_percentage_rounded_to_23_percent_l251_25127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_glasses_count_l251_25130

theorem restaurant_glasses_count :
  ∀ (small_boxes large_boxes : ℕ),
  small_boxes + large_boxes > 0 →
  large_boxes = small_boxes + 16 →
  (12 * small_boxes + 16 * large_boxes) / (small_boxes + large_boxes) = 15 →
  12 * small_boxes + 16 * large_boxes = 480 :=
fun small_boxes large_boxes h_positive h_box_diff h_average =>
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_glasses_count_l251_25130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_half_and_seven_halves_l251_25107

/-- A random variable X with probability distribution P(X=i) = i/a for i = 1, 2, 3, 4 -/
noncomputable def X (a : ℝ) : ℕ → ℝ
| 1 => 1 / a
| 2 => 2 / a
| 3 => 3 / a
| 4 => 4 / a
| _ => 0

/-- The sum of probabilities for X equals 1 -/
axiom sum_prob_one (a : ℝ) : (X a 1) + (X a 2) + (X a 3) + (X a 4) = 1

/-- Theorem: P(1/2 < X < 7/2) = 0.6 -/
theorem probability_between_half_and_seven_halves (a : ℝ) :
  (X a 1) + (X a 2) + (X a 3) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_half_and_seven_halves_l251_25107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_chemistry_marks_correct_l251_25133

/-- Calculates David's marks in Chemistry given his other marks and average --/
def davids_chemistry_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (biology : ℕ)
  (average : ℕ)
  (total_subjects : ℕ) : ℕ :=
  let total_marks := average * total_subjects
  let known_marks := english + mathematics + physics + biology
  total_marks - known_marks

theorem davids_chemistry_marks_correct
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (biology : ℕ)
  (average : ℕ)
  (total_subjects : ℕ)
  (h_english : english = 96)
  (h_mathematics : mathematics = 95)
  (h_physics : physics = 82)
  (h_biology : biology = 95)
  (h_average : average = 93)
  (h_total_subjects : total_subjects = 5)
  : davids_chemistry_marks english mathematics physics biology average total_subjects = 97 := by
  sorry

#eval davids_chemistry_marks 96 95 82 95 93 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_chemistry_marks_correct_l251_25133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_l251_25108

/-- Represents the number of volunteers -/
def num_volunteers : ℕ := 6

/-- Represents the number of days -/
def num_days : ℕ := 6

/-- Represents the number of ways to arrange B and C consecutively -/
def bc_arrangements : ℕ := 2

/-- Calculates the total number of arrangements without restrictions -/
def total_arrangements : ℕ := bc_arrangements * Nat.factorial (num_volunteers - 1)

/-- Calculates the number of arrangements with A on the first day -/
def a_first_arrangements : ℕ := bc_arrangements * Nat.factorial (num_volunteers - 2)

/-- The main theorem stating the number of valid arrangements -/
theorem valid_arrangements : 
  total_arrangements - a_first_arrangements = 192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_l251_25108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l251_25178

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - x^2 + a*x

-- State the theorem
theorem f_properties (a : ℝ) :
  (∀ x > 0, f 1 x < f 1 1 → x > 1) ∧ 
  (∀ x > 0, f 1 x > f 1 1 → 0 < x ∧ x < 1) ∧
  (a ≤ 1 → ∀ x > 0, f a x ≤ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l251_25178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l251_25101

theorem triangle_problem (A B C : ℝ) (AB : ℝ) 
  (h1 : Real.cos A = -5/13)
  (h2 : Real.cos B = 4/5)
  (h3 : AB = 11)
  (h4 : 0 < A ∧ A < Real.pi)
  (h5 : 0 < B ∧ B < Real.pi)
  (h6 : 0 < C ∧ C < Real.pi)
  (h7 : A + B + C = Real.pi) : 
  Real.sin C = 33/65 ∧ (1/2 * AB * AB * Real.sin A * Real.sin B / Real.sin C = 234) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l251_25101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_d_and_q_is_nine_l251_25197

/-- An arithmetic sequence with first term 1 and common difference d -/
def arithmetic_sequence (d : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => arithmetic_sequence d n + d

/-- A geometric sequence with first term b₁ and common ratio q -/
def geometric_sequence (b₁ q : ℝ) : ℕ → ℝ
  | 0 => b₁
  | n + 1 => geometric_sequence b₁ q n * q

/-- The sum of the arithmetic and geometric sequences -/
def combined_sequence (d q b₁ : ℝ) (n : ℕ) : ℝ :=
  arithmetic_sequence d n + geometric_sequence b₁ q n

theorem sum_of_d_and_q_is_nine (d q : ℝ) :
  (∃ b₁ : ℝ,
    combined_sequence d q b₁ 0 = 3 ∧
    combined_sequence d q b₁ 1 = 12 ∧
    combined_sequence d q b₁ 2 = 23) →
  d + q = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_d_and_q_is_nine_l251_25197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_angles_l251_25109

/-- A triangular pyramid with base ABC and apex S -/
structure TriangularPyramid where
  S : Real × Real × Real
  A : Real × Real × Real
  B : Real × Real × Real
  C : Real × Real × Real

/-- Predicate to check if a triangle is isosceles right with right angle at A -/
def IsIsoscelesRightTriangle (A B C : Real × Real × Real) : Prop := sorry

/-- Predicate to check if four angles form an arithmetic progression with non-zero common difference -/
def IsArithmeticProgression (a b c d : Real) : Prop := sorry

/-- Predicate to check if three areas form a geometric progression -/
def IsGeometricProgression (a b c : Real) : Prop := sorry

/-- Function to calculate the angle between two edges of a pyramid -/
noncomputable def PyramidAngle (S A B : Real × Real × Real) : Real := sorry

/-- Function to calculate the area of a triangle face of a pyramid -/
noncomputable def FaceArea (S A B : Real × Real × Real) : Real := sorry

/-- Main theorem -/
theorem triangular_pyramid_angles (SABC : TriangularPyramid) :
  IsIsoscelesRightTriangle SABC.A SABC.B SABC.C →
  IsArithmeticProgression 
    (PyramidAngle SABC.S SABC.A SABC.B)
    (PyramidAngle SABC.S SABC.C SABC.A)
    (PyramidAngle SABC.S SABC.A SABC.C)
    (PyramidAngle SABC.S SABC.B SABC.A) →
  IsGeometricProgression
    (FaceArea SABC.S SABC.A SABC.B)
    (FaceArea SABC.A SABC.B SABC.C)
    (FaceArea SABC.S SABC.A SABC.C) →
  ∃ φ : Real,
    φ = (1/2) * Real.arccos (Real.sqrt 2 - 1) ∧
    PyramidAngle SABC.S SABC.A SABC.B = π/2 - 2*φ ∧
    PyramidAngle SABC.S SABC.C SABC.A = π/2 - φ ∧
    PyramidAngle SABC.S SABC.A SABC.C = π/2 ∧
    PyramidAngle SABC.S SABC.B SABC.A = π/2 + φ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_angles_l251_25109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fan_sales_theorem_l251_25147

/-- Represents the fan sales and profit scenario -/
structure FanSales where
  initial_sales : ℝ
  initial_profit : ℝ
  price_reduction : ℝ → ℝ
  sales_increase : ℝ → ℝ

/-- The specific fan sales scenario given in the problem -/
def given_scenario : FanSales := {
  initial_sales := 24
  initial_profit := 60
  price_reduction := λ x => 5 * x
  sales_increase := λ x => 4 * x
}

/-- New average daily sales after price reduction -/
def new_sales (s : FanSales) (x : ℝ) : ℝ :=
  s.initial_sales + s.sales_increase x

/-- New profit per unit after price reduction -/
def new_profit (s : FanSales) (x : ℝ) : ℝ :=
  s.initial_profit - s.price_reduction x

/-- Daily profit function -/
def daily_profit (s : FanSales) (x : ℝ) : ℝ :=
  new_sales s x * new_profit s x

theorem fan_sales_theorem (s : FanSales) :
  (new_sales s = λ x => s.initial_sales + 4 * x) ∧
  (new_profit s = λ x => s.initial_profit - 5 * x) ∧
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ daily_profit s x₁ = 1540 ∧ daily_profit s x₂ = 1540) ∧
  (∀ x, daily_profit s x ≠ 2000) := by
  sorry

#check fan_sales_theorem given_scenario

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fan_sales_theorem_l251_25147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_coterminal_angle_proof_l251_25198

/-- The smallest positive angle (in radians) coterminal with -560 degrees -/
noncomputable def smallest_coterminal_angle : ℝ := 8 * Real.pi / 9

/-- Conversion factor from degrees to radians -/
noncomputable def deg_to_rad : ℝ := Real.pi / 180

theorem smallest_coterminal_angle_proof :
  smallest_coterminal_angle > 0 ∧
  ∃ (k : ℤ), smallest_coterminal_angle = -560 * deg_to_rad + 2 * Real.pi * k ∧
  ∀ (θ : ℝ), θ > 0 → (∃ (m : ℤ), θ = -560 * deg_to_rad + 2 * Real.pi * m) →
    θ ≥ smallest_coterminal_angle :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_coterminal_angle_proof_l251_25198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_percent_problem_l251_25195

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem rate_percent_problem (principal interest time : ℝ) 
  (h1 : principal = 15000)
  (h2 : interest = 6000)
  (h3 : time = 8)
  (h4 : simple_interest principal 5 time = interest) : 
  simple_interest principal 5 time = interest := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_percent_problem_l251_25195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l251_25112

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define the function F
noncomputable def F (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then f a b x else -f a b x

-- Define the function g
def g (a b k : ℝ) (x : ℝ) : ℝ := f a b x - k * x

theorem problem_solution (a b m n : ℝ) :
  f a b (-1) = 0 →
  (∀ x, f a b x ≥ 0) →
  m > 0 →
  n < 0 →
  m + n > 0 →
  a > 0 →
  (∀ x, f a b x = f a b (-x)) →
  (F a b = λ x ↦ if x > 0 then (x + 1)^2 else -(x + 1)^2) ∧
  (∀ k, (∀ x ∈ Set.Icc (-2) 2, Monotone (g a b k)) → (k ≥ 6 ∨ k ≤ -2)) ∧
  F a b m + F a b n > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l251_25112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_limit_proof_l251_25121

/-- Represents a circle with center O and radius r = a + x -/
structure Circle (a : ℝ) (x : ℝ) where
  center : ℝ × ℝ
  radius : ℝ := a + x

/-- Represents a chord in the circle -/
structure Chord (circle : Circle a x) where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ
  length : ℝ

/-- The ratio of areas as x approaches 0 -/
noncomputable def areaRatioLimit (a d : ℝ) : ℝ := (2 * a + d) / d

theorem area_ratio_limit_proof 
  (a : ℝ) 
  (ha : a > 0)
  (x : ℝ) 
  (hx : x ≤ a)
  (circle : Circle a x)
  (AB : Chord circle)
  (hAB : AB.length = 2 * circle.radius) -- AB is a diameter
  (CD : Chord circle)
  (hCD : CD.length = d) -- CD maintains its length d
  (E : ℝ × ℝ) -- Midpoint of CD
  (hE : E = ((CD.start.1 + CD.endpoint.1) / 2, (CD.start.2 + CD.endpoint.2) / 2))
  (b : ℝ)
  (hOE : dist circle.center E = b) -- OE = b
  : 
  ∃ (K R : ℝ → ℝ), 
    (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → 
      |K x / R x - areaRatioLimit a d| < ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_limit_proof_l251_25121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unfactorable_expression_l251_25185

theorem unfactorable_expression :
  ∃! f : ℝ → ℝ → ℝ, 
    (f = λ a b ↦ a^2 + 2*a*b - b^2) ∧
    (∀ x y, ∃ g h : ℝ → ℝ, g x * h x = 9*x^2 + 3*x*y^2) ∧
    (∀ x y, ∃ g h : ℝ → ℝ, g x * h x = -x^2 + 25*y^2) ∧
    (∀ x, ∃ g h : ℝ → ℝ, g x * h x = x^2 - x + 1/4) ∧
    (∀ a b, ¬∃ g h : ℝ → ℝ, g a * h a = f a b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unfactorable_expression_l251_25185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_partition_exists_l251_25181

/-- A binary sequence of length 2022 with 1011 zeros and 1011 ones -/
def BinarySequence : Type := Fin 2022 → Fin 2

/-- The set of all binary sequences of length 2022 with 1011 zeros and 1011 ones -/
def AllSequences : Set BinarySequence :=
  {s : BinarySequence | (Finset.filter (fun i => s i = 0) Finset.univ).card = 1011 ∧
                        (Finset.filter (fun i => s i = 1) Finset.univ).card = 1011}

/-- Two sequences are compatible if they match in exactly 4 positions -/
def compatible (s1 s2 : BinarySequence) : Prop :=
  (Finset.filter (fun i => s1 i = s2 i) Finset.univ).card = 4

/-- A partition of AllSequences into 20 groups -/
def Partition := Fin 20 → Set BinarySequence

/-- The partition covers all sequences and no two compatible sequences are in the same group -/
def ValidPartition (p : Partition) : Prop :=
  (∀ s, s ∈ AllSequences → ∃ i, s ∈ p i) ∧
  (∀ i : Fin 20, ∀ s1 s2, s1 ∈ p i → s2 ∈ p i → compatible s1 s2 → s1 = s2)

theorem sequence_partition_exists : ∃ p : Partition, ValidPartition p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_partition_exists_l251_25181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_theorem_l251_25153

/-- Basketball shooting game between two players -/
structure BasketballGame where
  playerA_shooting_percentage : ℝ
  playerB_shooting_percentage : ℝ
  first_shot_probability : ℝ

/-- The probability that player B takes the second shot -/
noncomputable def prob_B_second_shot (game : BasketballGame) : ℝ :=
  game.first_shot_probability * (1 - game.playerA_shooting_percentage) +
  game.first_shot_probability * game.playerB_shooting_percentage

/-- The probability that player A takes the i-th shot -/
noncomputable def prob_A_ith_shot (game : BasketballGame) (i : ℕ) : ℝ :=
  1/3 + (1/6) * (2/5)^(i-1)

/-- The expected number of times player A shoots in the first n shots -/
noncomputable def expected_A_shots (game : BasketballGame) (n : ℕ) : ℝ :=
  (5/18) * (1 - (2/5)^n) + n/3

/-- Main theorem about the basketball game -/
theorem basketball_game_theorem (game : BasketballGame) 
  (h1 : game.playerA_shooting_percentage = 0.6)
  (h2 : game.playerB_shooting_percentage = 0.8)
  (h3 : game.first_shot_probability = 0.5) :
  (prob_B_second_shot game = 0.6) ∧
  (∀ i : ℕ, prob_A_ith_shot game i = 1/3 + (1/6) * (2/5)^(i-1)) ∧
  (∀ n : ℕ, expected_A_shots game n = (5/18) * (1 - (2/5)^n) + n/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_theorem_l251_25153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l251_25162

-- Define the propositions
def proposition_1 (m : ℝ) : Prop :=
  (m > 0) ↔ ∃ (a b : ℝ), a^2 + b^2 = 1 ∧ ∀ (x y : ℝ), x^2 + m*y^2 = 1

def proposition_2 (a : ℝ) : Prop :=
  (a = 1 → ∀ (x y : ℝ), a*x + y - 1 = 0 ↔ x + a*y - 2 = 0) ∧
  ¬(∀ (x y : ℝ), a*x + y - 1 = 0 ↔ x + a*y - 2 = 0 → a = 1)

def proposition_3 (m : ℝ) : Prop :=
  (∀ (x₁ x₂ : ℝ), x₁ < x₂ → x₁^3 + m*x₁ < x₂^3 + m*x₂) ↔ m > 0

def proposition_4 (p q : Prop) : Prop :=
  (p ∨ q → p ∧ q) ∧ ¬(p ∧ q → p ∨ q)

-- Theorem statement
theorem propositions_truth : 
  (∃ (a : ℝ), proposition_2 a) ∧ 
  (∃ (p q : Prop), proposition_4 p q) ∧ 
  (∀ (m : ℝ), ¬proposition_1 m) ∧ 
  (∀ (m : ℝ), ¬proposition_3 m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l251_25162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_problem_l251_25192

/-- Given a solution with initial volume, initial alcohol percentage, added water volume,
    and final alcohol percentage, calculate the volume of alcohol added to achieve the final percentage. -/
noncomputable def alcohol_added (initial_volume : ℝ) (initial_percent : ℝ) (added_water : ℝ) (final_percent : ℝ) : ℝ :=
  let initial_alcohol := initial_volume * initial_percent
  let total_volume := initial_volume + added_water
  ((final_percent * total_volume) - initial_alcohol) / (1 - final_percent)

/-- The amount of alcohol added to the solution is approximately 4.5 liters. -/
theorem alcohol_solution_problem :
  let initial_volume : ℝ := 40
  let initial_percent : ℝ := 0.05
  let added_water : ℝ := 5.5
  let final_percent : ℝ := 0.13
  abs (alcohol_added initial_volume initial_percent added_water final_percent - 4.5) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_problem_l251_25192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_dog_hamburger_difference_l251_25126

noncomputable section

-- Define the weights of different food items
def chicken_weight : ℝ := 16
def hamburger_weight : ℝ := chicken_weight / 2
def total_weight : ℝ := 39

-- Define the weight of hot dogs as a variable
def hot_dog_weight : ℝ → ℝ := λ x => x

-- Define the weight of sides as half the weight of hot dogs
def side_weight : ℝ → ℝ := λ x => x / 2

-- Theorem statement
theorem hot_dog_hamburger_difference :
  ∃ x : ℝ, 
    x > hamburger_weight ∧ 
    chicken_weight + hamburger_weight + hot_dog_weight x + side_weight x = total_weight ∧
    x - hamburger_weight = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_dog_hamburger_difference_l251_25126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_commute_speed_l251_25186

/-- Represents the travel scenario for Alice's commute --/
structure TravelScenario where
  distance : ℝ  -- Distance in miles
  time : ℝ      -- Time in hours

/-- Calculates the speed given a TravelScenario --/
noncomputable def speed (s : TravelScenario) : ℝ := s.distance / s.time

/-- Alice's travel scenario when driving at 45 mph --/
noncomputable def scenario45 : TravelScenario := {
  distance := 45 * (1 - 5/60),
  time := 1 - 5/60
}

/-- Alice's travel scenario when driving at 65 mph --/
noncomputable def scenario65 : TravelScenario := {
  distance := 65 * (1 - 7/60),
  time := 1 - 7/60
}

/-- The exact speed Alice needs to drive to arrive on time --/
noncomputable def exactSpeed : ℝ := scenario45.distance / 1

theorem alice_commute_speed :
  ∃ ε > 0, abs (exactSpeed - 25) < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_commute_speed_l251_25186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_numbers_with_ratio_l251_25106

theorem product_of_numbers_with_ratio (x y : ℚ) : 
  (x - y : ℚ) / 1 = (x + y : ℚ) / 8 ∧ (x - y : ℚ) / 1 = (x * y : ℚ) / 15 → x * y = 100 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_numbers_with_ratio_l251_25106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_total_calories_l251_25103

/-- Calculates the total calories Tom ate given his consumption of carrots and broccoli -/
theorem tom_total_calories : 85 = (
  let carrot_weight : ℕ := 1  -- Tom eats 1 pound of carrots
  let broccoli_weight : ℕ := 2 * carrot_weight  -- Tom eats twice as much broccoli as carrots
  let carrot_calories_per_pound : ℕ := 51  -- Carrots have 51 calories per pound
  let broccoli_calories_per_pound : ℕ := carrot_calories_per_pound / 3  -- Broccoli has 1/3 the calories of carrots per pound
  carrot_weight * carrot_calories_per_pound + broccoli_weight * broccoli_calories_per_pound
) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_total_calories_l251_25103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_consumption_rate_calculation_l251_25122

/-- Represents the rate of fuel consumption in gallons per hour -/
noncomputable def fuelConsumptionRate (totalFuel : ℝ) (totalHours : ℝ) : ℝ :=
  totalFuel / totalHours

/-- Proves that the fuel consumption rate is equal to the total fuel divided by total hours -/
theorem fuel_consumption_rate_calculation (totalFuel : ℝ) (totalHours : ℝ) 
    (hFuel : totalFuel = 100) 
    (hHours : totalHours = 175) :
    fuelConsumptionRate totalFuel totalHours = totalFuel / totalHours :=
by
  -- Unfold the definition of fuelConsumptionRate
  unfold fuelConsumptionRate
  -- The result follows directly from the definition
  rfl

/-- Computes the approximate fuel consumption rate -/
def approximate_fuel_consumption_rate : ℚ :=
  100 / 175

#eval approximate_fuel_consumption_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_consumption_rate_calculation_l251_25122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_max_value_is_attainable_l251_25180

theorem max_value_of_expression (x y z w : ℕ) : 
  x ∈ ({2, 3, 4, 5} : Set ℕ) → y ∈ ({2, 3, 4, 5} : Set ℕ) → 
  z ∈ ({2, 3, 4, 5} : Set ℕ) → w ∈ ({2, 3, 4, 5} : Set ℕ) →
  x ≠ y → x ≠ z → x ≠ w → y ≠ z → y ≠ w → z ≠ w →
  x * y + y * z + z * w + w * x + 10 ≤ 59 :=
by sorry

theorem max_value_is_attainable : 
  ∃ (x y z w : ℕ), x ∈ ({2, 3, 4, 5} : Set ℕ) ∧ y ∈ ({2, 3, 4, 5} : Set ℕ) ∧ 
  z ∈ ({2, 3, 4, 5} : Set ℕ) ∧ w ∈ ({2, 3, 4, 5} : Set ℕ) ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
  x * y + y * z + z * w + w * x + 10 = 59 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_max_value_is_attainable_l251_25180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l251_25182

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem function_equality (f : ℝ → ℝ) :
  (∀ x : ℝ, f (g x) ≤ x ∧ x ≤ g (f x)) →
  ∃! h : ℝ → ℝ, (∀ x : ℝ, g (h x) = x ∧ h (g x) = x) ∧ f = h :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l251_25182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_cosine_and_value_l251_25170

theorem angle_sum_cosine_and_value (α β : Real) 
  (h1 : Real.sin α = Real.sqrt 5 / 5)
  (h2 : Real.sin β = Real.sqrt 10 / 10)
  (h3 : π / 2 < α ∧ α < π)
  (h4 : π / 2 < β ∧ β < π) :
  Real.cos (α + β) = Real.sqrt 2 / 2 ∧ α + β = 7 * π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_cosine_and_value_l251_25170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_zero_l251_25146

/-- The function representing the distance from a point (x, y) to the line 2x + y = 2√m -/
noncomputable def distance_to_line (m : ℝ) (x y : ℝ) : ℝ :=
  abs (2*x + y - 2*Real.sqrt m) / Real.sqrt 5

/-- The function representing the circle equation x^2 + y^2 = 4m -/
def circle_equation (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = 4*m

/-- The function representing the line equation 2x + y = 2√m -/
def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  2*x + y = 2*Real.sqrt m

/-- The theorem stating that if the circle is tangent to the line, then m = 0 -/
theorem tangent_implies_zero (m : ℝ) : 
  (∃ x y : ℝ, circle_equation m x y ∧ line_equation m x y ∧ 
    distance_to_line m x y = 2 * Real.sqrt m) → m = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_zero_l251_25146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l251_25166

def is_valid_triangle (x : ℝ) : Prop :=
  x > 0 ∧ x + 15 > 40 ∧ x + 40 > 15 ∧ 15 + 40 > x

def count_valid_integers (lower upper : ℤ) : ℕ :=
  (upper - lower + 1).toNat

theorem triangle_side_count :
  count_valid_integers 26 54 = 29 ∧
  ∀ x : ℤ, x ≥ 26 ∧ x ≤ 54 → is_valid_triangle (x : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l251_25166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l251_25114

noncomputable def triangle_ABC (A B C : Real) (a b c : Real) : Prop :=
  A + B + C = Real.pi ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

theorem triangle_theorem (A B C : Real) (a b c : Real) 
  (h_triangle : triangle_ABC A B C a b c)
  (h_equation : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos B) :
  B = Real.pi / 3 ∧
  ∀ M : Real, (∃ A', triangle_ABC A' B C a b c ∧ 
            M = Real.sin A' * (Real.sqrt 3 * Real.cos A' - Real.sin A')) →
           -3/2 < M ∧ M ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l251_25114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_y_value_l251_25129

/-- Represents a rectangle in a rectangular coordinate system -/
structure Rectangle where
  x1 : ℝ
  x2 : ℝ
  y1 : ℝ
  y2 : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := |r.x2 - r.x1| * |r.y2 - r.y1|

/-- Theorem: If a rectangle has vertices (-7, y), (1, y), (1, -6), and (-7, -6),
    and its area is 56, then y = 1 -/
theorem rectangle_y_value (y : ℝ) :
  let r := Rectangle.mk (-7) 1 y (-6)
  r.area = 56 → y = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_y_value_l251_25129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rental_cost_l251_25156

/-- Represents the number of buses of type A -/
def x : ℕ := sorry

/-- Represents the number of buses of type B -/
def y : ℕ := sorry

/-- The total number of passengers -/
def total_passengers : ℕ := 900

/-- The passenger capacity of bus type A -/
def capacity_A : ℕ := 36

/-- The passenger capacity of bus type B -/
def capacity_B : ℕ := 60

/-- The rental cost of bus type A in yuan -/
def cost_A : ℕ := 1600

/-- The rental cost of bus type B in yuan -/
def cost_B : ℕ := 2400

/-- The maximum total number of buses that can be rented -/
def max_buses : ℕ := 21

/-- The maximum difference between the number of type B and type A buses -/
def max_diff : ℕ := 7

/-- The total rental cost in yuan -/
def total_cost : ℕ := cost_A * x + cost_B * y

/-- Theorem stating that the minimum total rental cost is 36800 yuan -/
theorem min_rental_cost :
  (capacity_A * x + capacity_B * y ≥ total_passengers) →
  (x + y ≤ max_buses) →
  (y - x ≤ max_diff) →
  (∀ x' y', 
    (capacity_A * x' + capacity_B * y' ≥ total_passengers) →
    (x' + y' ≤ max_buses) →
    (y' - x' ≤ max_diff) →
    (cost_A * x' + cost_B * y' ≥ total_cost)) →
  total_cost = 36800 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rental_cost_l251_25156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_graph_l251_25119

theorem point_on_graph (c : ℝ) : 2^2 + c = 5 → c = 1 := by
  intro h
  rw [pow_two] at h
  linarith

#eval (2 : ℝ)^2 + 1 -- Evaluates to 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_graph_l251_25119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jam_probability_l251_25155

/-- The probability that the sum of half of a random number in [0,1] and another random number in [0,1] is at least 1 -/
theorem jam_probability : 
  let F := Set.Icc (0 : ℝ) 1 ×ˢ Set.Icc (0 : ℝ) 1
  let μ := MeasureTheory.volume F
  let A := {p : ℝ × ℝ | p ∈ F ∧ p.1 / 2 + p.2 ≥ 1}
  (MeasureTheory.volume A) / μ = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jam_probability_l251_25155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_who_left_l251_25140

/-- Given a class with an initial ratio of girls to boys as 5:6, 
    which changes to 2:3 after some girls leave, and there are 120 boys,
    prove that 20 girls left the class. -/
theorem girls_who_left (initial_girls : ℕ) (initial_boys : ℕ) 
  (final_girls : ℕ) (final_boys : ℕ) : 
  initial_girls * 6 = initial_boys * 5 →
  final_girls * 3 = final_boys * 2 →
  initial_boys = 120 →
  final_boys = 120 →
  initial_girls - final_girls = 20 := by
  intros h1 h2 h3 h4
  sorry

#check girls_who_left

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_who_left_l251_25140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_negative_reciprocal_l251_25183

/-- The function f(x) = (x + a) / (x + b) -/
noncomputable def f (a b x : ℝ) : ℝ := (x + a) / (x + b)

/-- Theorem: f(f(x)) = -1/x for all x ≠ 0 if and only if (a, b) = (-1, 1) -/
theorem f_composition_equals_negative_reciprocal (a b : ℝ) :
  (∀ x : ℝ, x ≠ 0 → x ≠ -b → f a b (f a b x) = -1/x) ↔ (a = -1 ∧ b = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_negative_reciprocal_l251_25183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_evaluation_square_root_difference_l251_25169

-- Part 1
theorem complex_expression_evaluation :
  (2 * (7 / 9 : ℝ)) ^ (1 / 2 : ℝ) - (2 * Real.sqrt 3 - Real.pi) ^ (0 : ℝ) - 
  (2 * (10 / 27 : ℝ)) ^ (-(2 / 3 : ℝ)) + (1 / 4 : ℝ) ^ (-(3 / 2 : ℝ)) = 8 + (5 / 48 : ℝ) := by sorry

-- Part 2
theorem square_root_difference (x : ℝ) (h1 : 0 < x) (h2 : x < 1) (h3 : x + x⁻¹ = 3) :
  x ^ ((1 : ℝ) / 2) - x ^ (-(1 : ℝ) / 2) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_evaluation_square_root_difference_l251_25169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_values_l251_25176

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of a line in the form y = mx + b is m -/
def slope_of_line1 (a : ℝ) : ℝ := a

/-- The slope of a line in the form Ax + By + C = 0 is -A/B -/
noncomputable def slope_of_line2 (a : ℝ) : ℝ := 3 / (a + 2)

theorem parallel_lines_a_values (a : ℝ) : 
  are_parallel (slope_of_line1 a) (slope_of_line2 a) → a = 1 ∨ a = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_values_l251_25176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_implies_m_bound_l251_25164

-- Define the function
noncomputable def f (x m : ℝ) : ℝ := (9 : ℝ)^x + m * (3 : ℝ)^x - 3

-- State the theorem
theorem monotone_decreasing_implies_m_bound (m : ℝ) :
  (∀ x₁ x₂ : ℝ, -2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 → f x₂ m < f x₁ m) →
  m ≤ -18 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_implies_m_bound_l251_25164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l251_25116

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a trapezoid -/
structure Trapezoid where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  let base1 := distance t.E t.F
  let base2 := distance t.G t.H
  let height := t.G.x - t.F.x
  (base1 + base2) * height / 2

/-- The theorem stating that the area of the specific trapezoid is 2(√13 + 4) -/
theorem trapezoid_area_theorem : 
  let t := Trapezoid.mk 
    (Point.mk 0 0) 
    (Point.mk 2 (-3)) 
    (Point.mk 6 0) 
    (Point.mk 6 4)
  trapezoidArea t = 2 * (Real.sqrt 13 + 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l251_25116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_k_range_l251_25190

noncomputable section

variable (f : ℝ → ℝ)

axiom f_increasing : ∀ x y : ℝ, x < y → f x < f y
axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem k_range (k : ℝ) : 
  (∀ x : ℝ, f (k * 3^x) + f (3^x - 9^x - 2) < 0) → 
  k < -1 + 2 * Real.sqrt 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_k_range_l251_25190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_m_min_value_of_Z_min_Z_is_9_l251_25172

noncomputable section

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ := {x | f m (x + 2) ≥ 0}

-- Theorem 1: Value of m
theorem value_of_m (m : ℝ) (h : solution_set m = Set.Icc (-1) 1) : m = 1 := by
  sorry

-- Define Z
def Z (a b c : ℝ) : ℝ := a + 2*b + 3*c

-- Theorem 2: Minimum value of Z
theorem min_value_of_Z (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/(2*b) + 1/(3*c) = 1) : 
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → 1/x + 1/(2*y) + 1/(3*z) = 1 → Z a b c ≤ Z x y z := by
  sorry

-- Corollary: The minimum value of Z is 9
theorem min_Z_is_9 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/(2*b) + 1/(3*c) = 1) : Z a b c ≥ 9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_m_min_value_of_Z_min_Z_is_9_l251_25172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_afternoon_boys_count_l251_25165

/-- Represents the number of boys a girl danced with -/
def DancingSequence (n : ℕ) : ℕ := n + 6

theorem club_afternoon_boys_count 
  (total_attendees : ℕ) 
  (first_girl_danced : ℕ) 
  (last_girl_danced : ℕ → ℕ) 
  (h1 : total_attendees = 31)
  (h2 : first_girl_danced = 7)
  (h3 : ∀ n : ℕ, DancingSequence n = n + 6)
  (h4 : ∃ k : ℕ, last_girl_danced k = DancingSequence k ∧ last_girl_danced k = k - 3)
  : ∃ boys : ℕ, boys = 20 ∧ boys + (DancingSequence boys - 6) = total_attendees :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_afternoon_boys_count_l251_25165

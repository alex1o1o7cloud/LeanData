import Mathlib

namespace NUMINAMATH_CALUDE_pirate_gold_distribution_l3492_349225

theorem pirate_gold_distribution (total : ℕ) (jack jimmy tom sanji : ℕ) : 
  total = 280 ∧ 
  jimmy = jack + 11 ∧ 
  tom = jack - 15 ∧ 
  sanji = jack + 20 ∧ 
  total = jack + jimmy + tom + sanji → 
  sanji = 86 := by
sorry

end NUMINAMATH_CALUDE_pirate_gold_distribution_l3492_349225


namespace NUMINAMATH_CALUDE_sum_and_product_zero_l3492_349281

theorem sum_and_product_zero (a b : ℝ) 
  (h1 : 2*a + 2*b + a*b = 1) 
  (h2 : a + b + 3*a*b = -2) : 
  a + b + a*b = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_and_product_zero_l3492_349281


namespace NUMINAMATH_CALUDE_sugar_for_40_cookies_l3492_349227

/-- The amount of sugar needed to make a given number of cookies -/
def sugar_needed (cookies : ℕ) : ℚ :=
  (2 / 3) * (cookies / 8 : ℚ)

/-- Proof that 40 cookies require 10/3 cups of sugar -/
theorem sugar_for_40_cookies :
  sugar_needed 40 = 10 / 3 := by
  sorry

#eval sugar_needed 40

end NUMINAMATH_CALUDE_sugar_for_40_cookies_l3492_349227


namespace NUMINAMATH_CALUDE_parking_cost_is_10_l3492_349256

-- Define the given conditions
def saved : ℕ := 28
def entry_cost : ℕ := 55
def meal_pass_cost : ℕ := 25
def distance : ℕ := 165
def fuel_efficiency : ℕ := 30
def gas_price : ℕ := 3
def additional_savings : ℕ := 95

-- Define the function to calculate parking cost
def parking_cost : ℕ :=
  let total_needed := saved + additional_savings
  let round_trip_distance := 2 * distance
  let gas_needed := round_trip_distance / fuel_efficiency
  let gas_cost := gas_needed * gas_price
  let total_cost_without_parking := gas_cost + entry_cost + meal_pass_cost
  total_needed - total_cost_without_parking

-- Theorem to prove
theorem parking_cost_is_10 : parking_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_parking_cost_is_10_l3492_349256


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_ten_l3492_349228

theorem units_digit_of_seven_to_ten (n : ℕ) : 7^10 ≡ 9 [MOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_ten_l3492_349228


namespace NUMINAMATH_CALUDE_dodecahedron_outer_rectangle_property_l3492_349286

/-- Regular dodecahedron with side length a -/
structure RegularDodecahedron (a : ℝ) where
  side_length : a > 0

/-- Point on a line outside a face of the dodecahedron -/
structure OuterPoint (a m : ℝ) where
  distance : m > 0

/-- Rectangle formed by four outer points -/
structure OuterRectangle (a m : ℝ) where
  A : OuterPoint a m
  B : OuterPoint a m
  C : OuterPoint a m
  D : OuterPoint a m

theorem dodecahedron_outer_rectangle_property 
  (a m : ℝ) 
  (d : RegularDodecahedron a) 
  (r : OuterRectangle a m) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
  y / x = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_outer_rectangle_property_l3492_349286


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3492_349292

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

theorem f_increasing_on_interval :
  ∀ a b, 0 < a ∧ b < π/2 ∧ a < b →
  StrictMonoOn f (Set.Ioo a b) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3492_349292


namespace NUMINAMATH_CALUDE_smallest_circular_sequence_l3492_349296

def is_valid_sequence (s : List Nat) : Prop :=
  ∀ x ∈ s, x = 1 ∨ x = 2

def contains_all_four_digit_sequences (s : List Nat) : Prop :=
  ∀ seq : List Nat, seq.length = 4 → is_valid_sequence seq →
    ∃ i, List.take 4 (List.rotateLeft s i ++ List.rotateLeft s i) = seq ∨
         List.take 4 (List.rotateRight s i ++ List.rotateRight s i) = seq

theorem smallest_circular_sequence :
  ∃ (N : Nat) (s : List Nat),
    N = s.length ∧
    is_valid_sequence s ∧
    contains_all_four_digit_sequences s ∧
    (∀ M < N, ¬∃ t : List Nat, M = t.length ∧ is_valid_sequence t ∧ contains_all_four_digit_sequences t) ∧
    N = 14 := by
  sorry

end NUMINAMATH_CALUDE_smallest_circular_sequence_l3492_349296


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3492_349249

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2 * x - 4| = x + 3 :=
by
  -- The unique solution is x = 7
  use 7
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3492_349249


namespace NUMINAMATH_CALUDE_right_triangle_geometric_mean_l3492_349208

theorem right_triangle_geometric_mean (a c : ℝ) (h₁ : 0 < a) (h₂ : 0 < c) :
  (c * c = a * c) → (a = (c * (Real.sqrt 5 - 1)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_geometric_mean_l3492_349208


namespace NUMINAMATH_CALUDE_positive_integer_pairs_satisfying_equation_l3492_349206

theorem positive_integer_pairs_satisfying_equation :
  ∀ x y : ℕ+, 
    (x * y * Nat.gcd x.val y.val = x + y + (Nat.gcd x.val y.val)^2) ↔ 
    ((x = 2 ∧ y = 2) ∨ (x = 2 ∧ y = 3) ∨ (x = 3 ∧ y = 2)) := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_pairs_satisfying_equation_l3492_349206


namespace NUMINAMATH_CALUDE_triangle_side_length_l3492_349262

theorem triangle_side_length (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  a + c = 2 * b →          -- Given condition
  a * c = 6 →              -- Given condition
  Real.cos (60 * π / 180) = (a^2 + c^2 - b^2) / (2 * a * c) →  -- Cosine theorem for 60°
  b = Real.sqrt 6 := by
sorry

-- Note: We use Real.cos and Real.sqrt to represent cosine and square root functions

end NUMINAMATH_CALUDE_triangle_side_length_l3492_349262


namespace NUMINAMATH_CALUDE_max_cd_length_l3492_349276

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    where c = 4 and CD⊥AB, prove that the maximum value of CD is 2√3 under the given condition. -/
theorem max_cd_length (a b : ℝ) (A B C : ℝ) :
  let c : ℝ := 4
  (c * Real.cos C * Real.cos (A - B) + c = c * Real.sin C ^ 2 + b * Real.sin A * Real.sin C) →
  (∃ (D : ℝ), D ≤ 2 * Real.sqrt 3 ∧
    ∀ (E : ℝ), (c * Real.cos C * Real.cos (A - B) + c = c * Real.sin C ^ 2 + b * Real.sin A * Real.sin C) →
      E ≤ D) :=
by sorry

end NUMINAMATH_CALUDE_max_cd_length_l3492_349276


namespace NUMINAMATH_CALUDE_juice_price_proof_l3492_349275

def total_paid : ℚ := 370 / 100
def muffin_price : ℚ := 75 / 100
def muffin_count : ℕ := 3

theorem juice_price_proof :
  total_paid - (muffin_price * muffin_count) = 145 / 100 := by
  sorry

end NUMINAMATH_CALUDE_juice_price_proof_l3492_349275


namespace NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_l3492_349246

-- Equation 1
theorem solution_equation_one : 
  ∀ x : ℝ, 2 * x^2 - 4 * x - 1 = 0 ↔ x = 1 + Real.sqrt 6 / 2 ∨ x = 1 - Real.sqrt 6 / 2 := by
sorry

-- Equation 2
theorem solution_equation_two :
  ∀ x : ℝ, (x - 1) * (x + 2) = 28 ↔ x = -6 ∨ x = 5 := by
sorry

end NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_l3492_349246


namespace NUMINAMATH_CALUDE_wood_rope_problem_l3492_349277

/-- Represents the system of equations for the wood and rope problem -/
def wood_rope_equations (x y : ℝ) : Prop :=
  (y - x = 4.5) ∧ (x - y/2 = 1)

/-- Theorem stating that the equations correctly represent the given conditions -/
theorem wood_rope_problem (x y : ℝ) :
  wood_rope_equations x y →
  (y - x = 4.5 ∧ x - y/2 = 1) :=
by
  sorry

#check wood_rope_problem

end NUMINAMATH_CALUDE_wood_rope_problem_l3492_349277


namespace NUMINAMATH_CALUDE_cube_lines_properties_l3492_349232

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  a : ℝ
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Defines a cube with given edge length and correct vertex positions -/
def makeCube (a : ℝ) : Cube := {
  a := a,
  A := ⟨0, 0, 0⟩,
  B := ⟨a, 0, 0⟩,
  C := ⟨a, a, 0⟩,
  D := ⟨0, a, 0⟩,
  A₁ := ⟨0, 0, a⟩,
  B₁ := ⟨a, 0, a⟩,
  C₁ := ⟨a, a, a⟩,
  D₁ := ⟨0, a, a⟩
}

/-- Calculates the angle between two lines in the cube -/
def angleBetweenLines (cube : Cube) : ℝ := sorry

/-- Calculates the distance between two lines in the cube -/
def distanceBetweenLines (cube : Cube) : ℝ := sorry

theorem cube_lines_properties (a : ℝ) (h : a > 0) :
  let cube := makeCube a
  angleBetweenLines cube = 90 ∧ 
  distanceBetweenLines cube = a * Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_lines_properties_l3492_349232


namespace NUMINAMATH_CALUDE_fraction_equality_l3492_349226

theorem fraction_equality (x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 8 * y) = -1) :
  (2 * x + 8 * y) / (4 * x - 2 * y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3492_349226


namespace NUMINAMATH_CALUDE_A_intersect_B_l3492_349204

def A : Set ℝ := {-2, 0, 1, 2}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem A_intersect_B : A ∩ B = {-2, 0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l3492_349204


namespace NUMINAMATH_CALUDE_division_problem_l3492_349248

theorem division_problem (dividend quotient remainder : ℕ) (divisor : ℕ) :
  dividend = 760 →
  quotient = 21 →
  remainder = 4 →
  dividend = divisor * quotient + remainder →
  divisor = 36 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3492_349248


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l3492_349290

theorem cube_volume_ratio : 
  let cube1_edge_length : ℚ := 10  -- in inches
  let cube2_edge_length : ℚ := 5 * 12  -- 5 feet converted to inches
  let volume_ratio := (cube1_edge_length / cube2_edge_length) ^ 3
  volume_ratio = 1 / 216 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l3492_349290


namespace NUMINAMATH_CALUDE_photo_arrangements_l3492_349217

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange k objects out of n distinct objects. -/
def permutations (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

theorem photo_arrangements (teachers students : ℕ) 
  (h1 : teachers = 4) (h2 : students = 4) : 
  /- Students stand together -/
  (arrangements students * arrangements (teachers + 1) = 2880) ∧ 
  /- No two students are adjacent -/
  (arrangements teachers * permutations (teachers + 1) students = 2880) ∧
  /- Teachers and students alternate -/
  (2 * arrangements teachers * arrangements students = 1152) := by
  sorry

#check photo_arrangements

end NUMINAMATH_CALUDE_photo_arrangements_l3492_349217


namespace NUMINAMATH_CALUDE_inequality_proof_l3492_349210

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^2 / 2 + (a + b) / 4 ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3492_349210


namespace NUMINAMATH_CALUDE_money_sharing_ratio_l3492_349272

theorem money_sharing_ratio (total : ℕ) (ken_amount : ℕ) : 
  total = 5250 → ken_amount = 1750 → 
  (total - ken_amount) / ken_amount = 2 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_ratio_l3492_349272


namespace NUMINAMATH_CALUDE_trumpet_players_count_l3492_349239

def orchestra_size : ℕ := 21
def drummer_count : ℕ := 1
def trombone_count : ℕ := 4
def french_horn_count : ℕ := 1
def violinist_count : ℕ := 3
def cellist_count : ℕ := 1
def contrabassist_count : ℕ := 1
def clarinet_count : ℕ := 3
def flute_count : ℕ := 4
def maestro_count : ℕ := 1

theorem trumpet_players_count :
  orchestra_size - (drummer_count + trombone_count + french_horn_count + 
    violinist_count + cellist_count + contrabassist_count + 
    clarinet_count + flute_count + maestro_count) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trumpet_players_count_l3492_349239


namespace NUMINAMATH_CALUDE_rectangle_area_l3492_349216

/-- The area of a rectangle given its perimeter and width -/
theorem rectangle_area (perimeter width : ℝ) (h1 : perimeter = 56) (h2 : width = 16) :
  let length := (perimeter - 2 * width) / 2
  width * length = 192 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3492_349216


namespace NUMINAMATH_CALUDE_range_of_function_l3492_349214

theorem range_of_function :
  ∀ (x : ℝ), -2/3 ≤ (Real.sin x - 1) / (2 - Real.sin x) ∧ 
             (Real.sin x - 1) / (2 - Real.sin x) ≤ 0 ∧
  (∃ (y : ℝ), (Real.sin y - 1) / (2 - Real.sin y) = -2/3) ∧
  (∃ (z : ℝ), (Real.sin z - 1) / (2 - Real.sin z) = 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_function_l3492_349214


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l3492_349240

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define what it means for a circle to be tangent to the x-axis
def tangentToXAxis (c : Circle) : Prop :=
  c.center.2 = c.radius

-- Define the equation of a circle
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_tangent_to_x_axis :
  ∀ (c : Circle),
    c.center = (5, 4) →
    tangentToXAxis c →
    ∀ (x y : ℝ), circleEquation c x y ↔ (x - 5)^2 + (y - 4)^2 = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l3492_349240


namespace NUMINAMATH_CALUDE_bike_rental_fixed_fee_bike_rental_fixed_fee_proof_l3492_349283

/-- The fixed fee for renting a bike, given the total cost formula and a specific rental case. -/
theorem bike_rental_fixed_fee : ℝ → Prop :=
  fun fixed_fee =>
    let total_cost := fun (hours : ℝ) => fixed_fee + 7 * hours
    total_cost 9 = 80 → fixed_fee = 17

/-- Proof of the bike rental fixed fee theorem -/
theorem bike_rental_fixed_fee_proof : bike_rental_fixed_fee 17 := by
  sorry

end NUMINAMATH_CALUDE_bike_rental_fixed_fee_bike_rental_fixed_fee_proof_l3492_349283


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3492_349279

theorem square_sum_geq_product_sum (x y z : ℝ) : x^2 + y^2 + z^2 ≥ x*y + y*z + z*x := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3492_349279


namespace NUMINAMATH_CALUDE_line_parallel_from_plane_parallel_l3492_349222

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the parallelism relation for lines and planes
variable (parallel : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_from_plane_parallel
  (a b : Line) (α β γ δ : Plane)
  (h_distinct_lines : a ≠ b)
  (h_distinct_planes : α ≠ β ∧ α ≠ γ ∧ α ≠ δ ∧ β ≠ γ ∧ β ≠ δ ∧ γ ≠ δ)
  (h_intersect_ab : intersect α β = a)
  (h_intersect_gd : intersect γ δ = b)
  (h_parallel_ag : planeParallel α γ)
  (h_parallel_bd : planeParallel β δ) :
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_from_plane_parallel_l3492_349222


namespace NUMINAMATH_CALUDE_bus_speed_calculation_l3492_349265

/-- Proves that a bus stopping for 12 minutes per hour with an average speed of 40 km/hr including stoppages has an average speed of 50 km/hr excluding stoppages. -/
theorem bus_speed_calculation (stop_time : ℝ) (avg_speed_with_stops : ℝ) :
  stop_time = 12 →
  avg_speed_with_stops = 40 →
  let moving_time : ℝ := 60 - stop_time
  let speed_ratio : ℝ := moving_time / 60
  (speed_ratio * (60 / moving_time) * avg_speed_with_stops) = 50 := by
  sorry

#check bus_speed_calculation

end NUMINAMATH_CALUDE_bus_speed_calculation_l3492_349265


namespace NUMINAMATH_CALUDE_area_equality_in_divided_triangle_l3492_349266

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (A B C : Point) : ℝ := sorry

/-- Represents a triangle with its three vertices -/
structure Triangle :=
  (A B C : Point)

/-- Given a triangle and a ratio, returns a point on one of its sides -/
def pointOnSide (T : Triangle) (ratio : ℝ) (side : Fin 3) : Point := sorry

theorem area_equality_in_divided_triangle (ABC : Triangle) :
  let D := pointOnSide ABC (1/3) 0
  let E := pointOnSide ABC (1/3) 1
  let F := pointOnSide ABC (1/3) 2
  let G := pointOnSide (Triangle.mk D E F) (1/2) 0
  let H := pointOnSide (Triangle.mk D E F) (1/2) 1
  let I := pointOnSide (Triangle.mk D E F) (1/2) 2
  triangleArea D A G + triangleArea E B H + triangleArea F C I = triangleArea G H I :=
by sorry

end NUMINAMATH_CALUDE_area_equality_in_divided_triangle_l3492_349266


namespace NUMINAMATH_CALUDE_exists_fib_divisible_by_2014_l3492_349213

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Theorem: There exists a positive integer n such that F_n is divisible by 2014 -/
theorem exists_fib_divisible_by_2014 : ∃ n : ℕ, n > 0 ∧ 2014 ∣ fib n := by
  sorry

end NUMINAMATH_CALUDE_exists_fib_divisible_by_2014_l3492_349213


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l3492_349269

theorem triangle_perimeter_bound :
  ∀ (a b c : ℝ),
  a = 7 →
  b ≥ 14 →
  a + b > c →
  a + c > b →
  b + c > a →
  a + b + c < 42 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l3492_349269


namespace NUMINAMATH_CALUDE_function_bound_implies_parameter_range_l3492_349207

-- Define the function f
def f (a d x : ℝ) : ℝ := a * x^3 + x^2 + x + d

-- State the theorem
theorem function_bound_implies_parameter_range :
  ∀ (a d : ℝ),
  (∀ x : ℝ, |x| ≤ 1 → |f a d x| ≤ 1) →
  (a ∈ Set.Icc (-2) 0 ∧ d ∈ Set.Icc (-2) 0) :=
by sorry

end NUMINAMATH_CALUDE_function_bound_implies_parameter_range_l3492_349207


namespace NUMINAMATH_CALUDE_seventh_group_draw_l3492_349218

/-- Represents the systematic sampling method for a population -/
structure SystematicSampling where
  populationSize : Nat
  groupCount : Nat
  sampleSize : Nat
  firstDrawn : Nat

/-- Calculates the number drawn in a specific group -/
def SystematicSampling.numberDrawnInGroup (s : SystematicSampling) (groupNumber : Nat) : Nat :=
  let groupSize := s.populationSize / s.groupCount
  let baseNumber := (groupNumber - 1) * groupSize
  baseNumber + (s.firstDrawn + groupNumber - 1) % 10

theorem seventh_group_draw (s : SystematicSampling) 
  (h1 : s.populationSize = 100)
  (h2 : s.groupCount = 10)
  (h3 : s.sampleSize = 10)
  (h4 : s.firstDrawn = 6) :
  s.numberDrawnInGroup 7 = 63 := by
  sorry

#check seventh_group_draw

end NUMINAMATH_CALUDE_seventh_group_draw_l3492_349218


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3492_349287

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3492_349287


namespace NUMINAMATH_CALUDE_stock_price_change_l3492_349223

def down_limit : ℝ := 0.9
def up_limit : ℝ := 1.1
def num_limits : ℕ := 3

theorem stock_price_change (initial_price : ℝ) (initial_price_pos : initial_price > 0) :
  initial_price * (down_limit ^ num_limits) * (up_limit ^ num_limits) < initial_price :=
by sorry

end NUMINAMATH_CALUDE_stock_price_change_l3492_349223


namespace NUMINAMATH_CALUDE_vector_on_line_k_value_l3492_349250

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def line_through (a b : V) : ℝ → V :=
  λ t => a + t • (b - a)

theorem vector_on_line_k_value
  (a b : V) (ha_ne_b : a ≠ b) (k : ℝ) :
  (∃ t : ℝ, line_through a b t = k • a + (5/7 : ℝ) • b) →
  k = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_vector_on_line_k_value_l3492_349250


namespace NUMINAMATH_CALUDE_quadratic_linear_intersection_l3492_349229

theorem quadratic_linear_intersection (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 2 = -3 * x - 2) ↔ a = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_linear_intersection_l3492_349229


namespace NUMINAMATH_CALUDE_hadley_walk_distance_l3492_349242

/-- The total distance Hadley walked in his boots -/
def total_distance (grocery_store_distance pet_store_distance home_distance : ℕ) : ℕ :=
  grocery_store_distance + pet_store_distance + home_distance

/-- Theorem stating the total distance Hadley walked -/
theorem hadley_walk_distance :
  ∃ (grocery_store_distance pet_store_distance home_distance : ℕ),
    grocery_store_distance = 2 ∧
    pet_store_distance = 2 - 1 ∧
    home_distance = 4 - 1 ∧
    total_distance grocery_store_distance pet_store_distance home_distance = 6 := by
  sorry

end NUMINAMATH_CALUDE_hadley_walk_distance_l3492_349242


namespace NUMINAMATH_CALUDE_april_order_proof_l3492_349241

/-- The number of cases of soda ordered in April -/
def april_cases : ℕ := sorry

/-- The number of cases of soda ordered in May -/
def may_cases : ℕ := 30

/-- The number of bottles per case -/
def bottles_per_case : ℕ := 20

/-- The total number of bottles ordered in April and May -/
def total_bottles : ℕ := 1000

theorem april_order_proof :
  april_cases = 20 ∧
  april_cases * bottles_per_case + may_cases * bottles_per_case = total_bottles :=
by sorry

end NUMINAMATH_CALUDE_april_order_proof_l3492_349241


namespace NUMINAMATH_CALUDE_seongmin_completion_time_l3492_349224

/-- The number of days it takes Seongmin to complete the task alone -/
def seongmin_days : ℚ := 32

/-- The fraction of work Jinwoo and Seongmin complete together in 8 days -/
def work_together : ℚ := 7/12

/-- The number of days Jinwoo and Seongmin work together -/
def days_together : ℚ := 8

/-- The number of days Jinwoo works alone to complete the remaining work -/
def jinwoo_alone_days : ℚ := 10

theorem seongmin_completion_time :
  let total_work : ℚ := 1
  let work_rate_together : ℚ := work_together / days_together
  let jinwoo_alone_work : ℚ := total_work - work_together
  let jinwoo_work_rate : ℚ := jinwoo_alone_work / jinwoo_alone_days
  let seongmin_work_rate : ℚ := work_rate_together - jinwoo_work_rate
  seongmin_days = total_work / seongmin_work_rate :=
by sorry

end NUMINAMATH_CALUDE_seongmin_completion_time_l3492_349224


namespace NUMINAMATH_CALUDE_total_area_is_36_l3492_349255

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- The size of the square grid -/
def gridSize : ℕ := 6

/-- The center point of the grid -/
def gridCenter : Point := { x := 3, y := 3 }

/-- Calculates the area of a triangle given its three points -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Generates all triangles formed by connecting the center to adjacent perimeter points -/
def perimeterTriangles : List Triangle := sorry

/-- Theorem: The total area of triangles formed by connecting the center of a 6x6 square grid
    to each pair of adjacent vertices along the perimeter is equal to 36 -/
theorem total_area_is_36 : 
  (perimeterTriangles.map triangleArea).sum = 36 := by sorry

end NUMINAMATH_CALUDE_total_area_is_36_l3492_349255


namespace NUMINAMATH_CALUDE_more_I_than_P_l3492_349235

/-- Sum of digits of a natural number -/
def S (n : ℕ) : ℕ := sorry

/-- Property P: all terms in the sequence n, S(n), S(S(n)),... are even -/
def has_property_P (n : ℕ) : Prop := sorry

/-- Property I: all terms in the sequence n, S(n), S(S(n)),... are odd -/
def has_property_I (n : ℕ) : Prop := sorry

/-- Count of numbers with property P in the range 1 to 2017 -/
def count_P : ℕ := sorry

/-- Count of numbers with property I in the range 1 to 2017 -/
def count_I : ℕ := sorry

theorem more_I_than_P : count_I > count_P := by sorry

end NUMINAMATH_CALUDE_more_I_than_P_l3492_349235


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_area_ratio_l3492_349209

/-- The ratio of the perimeter to the area of an equilateral triangle with side length 8 units is √3/2 -/
theorem equilateral_triangle_perimeter_area_ratio :
  let s : ℝ := 8
  let perimeter : ℝ := 3 * s
  let area : ℝ := s^2 * Real.sqrt 3 / 4
  perimeter / area = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_area_ratio_l3492_349209


namespace NUMINAMATH_CALUDE_function_extrema_l3492_349221

open Real

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x - log x

theorem function_extrema (a b : ℝ) :
  (a = -1 ∧ b = 3 →
    (∃ (max min : ℝ),
      (∀ x ∈ Set.Icc (1/2) 2, f a b x ≤ max) ∧
      (∃ x ∈ Set.Icc (1/2) 2, f a b x = max) ∧
      (∀ x ∈ Set.Icc (1/2) 2, f a b x ≥ min) ∧
      (∃ x ∈ Set.Icc (1/2) 2, f a b x = min) ∧
      max = 2 ∧
      min = log 2 + 5/4)) ∧
  (a = 0 →
    (∃! b : ℝ,
      b > 0 ∧
      (∃ min : ℝ,
        (∀ x ∈ Set.Ioo 0 (exp 1), f a b x ≥ min) ∧
        (∃ x ∈ Set.Ioo 0 (exp 1), f a b x = min) ∧
        min = 3) ∧
      b = exp 2)) :=
sorry

end NUMINAMATH_CALUDE_function_extrema_l3492_349221


namespace NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l3492_349288

theorem tangent_line_to_logarithmic_curve (a : ℝ) : 
  (∃ x₀ y₀ : ℝ, 
    y₀ = x₀ + 1 ∧ 
    y₀ = Real.log (x₀ + a) ∧ 
    (Real.exp y₀)⁻¹ = 1) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l3492_349288


namespace NUMINAMATH_CALUDE_remainder_thirteen_plus_x_l3492_349299

theorem remainder_thirteen_plus_x (x : ℕ+) (h : 8 * x.val ≡ 1 [MOD 29]) :
  (13 + x.val) % 29 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_thirteen_plus_x_l3492_349299


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l3492_349285

/-- The volume of a regular tetrahedron with given base side length and lateral face angle -/
theorem tetrahedron_volume 
  (base_side : ℝ) 
  (lateral_angle : ℝ) 
  (h_base : base_side = Real.sqrt 3) 
  (h_angle : lateral_angle = π / 3) : 
  (1 / 3 : ℝ) * base_side ^ 2 * (base_side / 2) / Real.tan lateral_angle = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l3492_349285


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_million_l3492_349231

theorem multiplicative_inverse_modulo_million : ∃ N : ℕ, 
  N > 0 ∧ 
  N < 1000000 ∧ 
  (N * ((222222 : ℕ) * 476190)) % 1000000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_million_l3492_349231


namespace NUMINAMATH_CALUDE_inequality_solution_l3492_349236

theorem inequality_solution (x : ℝ) : x^2 - x - 5 > 3*x ↔ x > 5 ∨ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3492_349236


namespace NUMINAMATH_CALUDE_find_m_l3492_349212

theorem find_m : ∃ m : ℝ, 
  (∃ y : ℝ, 2 - 3 * (1 - y) = 2 * y) ∧ 
  (∃ x : ℝ, m * (x - 3) - 2 = -8) ∧ 
  (∀ y x : ℝ, 2 - 3 * (1 - y) = 2 * y ↔ m * (x - 3) - 2 = -8) → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_find_m_l3492_349212


namespace NUMINAMATH_CALUDE_flag_movement_theorem_l3492_349203

/-- Calculates the total distance a flag moves on a flagpole given the pole height and a sequence of movements. -/
def totalFlagMovement (poleHeight : ℝ) (movements : List ℝ) : ℝ :=
  movements.map (abs) |>.sum

/-- Theorem stating the total distance a flag moves on a 60-foot flagpole when raised to the top, 
    lowered halfway, raised to the top again, and then lowered completely is 180 feet. -/
theorem flag_movement_theorem :
  let poleHeight : ℝ := 60
  let movements : List ℝ := [poleHeight, -poleHeight/2, poleHeight/2, -poleHeight]
  totalFlagMovement poleHeight movements = 180 := by
  sorry

#eval totalFlagMovement 60 [60, -30, 30, -60]

end NUMINAMATH_CALUDE_flag_movement_theorem_l3492_349203


namespace NUMINAMATH_CALUDE_unique_two_digit_number_exists_l3492_349254

/-- A two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Get the tens digit of a two-digit number -/
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10

/-- Get the units digit of a two-digit number -/
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

/-- Reverse the digits of a two-digit number -/
def reverse_digits (n : TwoDigitNumber) : TwoDigitNumber :=
  ⟨10 * (units_digit n) + (tens_digit n), by sorry⟩

theorem unique_two_digit_number_exists :
  ∃! (X : TwoDigitNumber),
    (tens_digit X) * (units_digit X) = 24 ∧
    (reverse_digits X).val = X.val + 18 ∧
    X.val = 46 := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_exists_l3492_349254


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_five_l3492_349257

theorem fraction_zero_implies_x_negative_five (x : ℝ) :
  (x + 5) / (x - 2) = 0 → x = -5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_five_l3492_349257


namespace NUMINAMATH_CALUDE_science_club_enrollment_l3492_349233

theorem science_club_enrollment (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) :
  total = 150 →
  math = 80 →
  physics = 60 →
  both = 20 →
  total - (math + physics - both) = 30 := by
sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l3492_349233


namespace NUMINAMATH_CALUDE_shielas_paint_colors_l3492_349237

theorem shielas_paint_colors (total_blocks : ℕ) (blocks_per_color : ℕ) (h1 : total_blocks = 196) (h2 : blocks_per_color = 14) : 
  total_blocks / blocks_per_color = 14 := by
  sorry

end NUMINAMATH_CALUDE_shielas_paint_colors_l3492_349237


namespace NUMINAMATH_CALUDE_shorter_leg_of_second_triangle_l3492_349238

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ
  hypotenuse_eq : hypotenuse = 2 * shorterLeg
  longerLeg_eq : longerLeg = shorterLeg * Real.sqrt 3

/-- A sequence of two 30-60-90 triangles where the hypotenuse of the first is the longer leg of the second -/
def TwoTriangles (t1 t2 : Triangle30_60_90) :=
  t1.hypotenuse = 12 ∧ t1.longerLeg = t2.hypotenuse

theorem shorter_leg_of_second_triangle (t1 t2 : Triangle30_60_90) 
  (h : TwoTriangles t1 t2) : t2.shorterLeg = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_shorter_leg_of_second_triangle_l3492_349238


namespace NUMINAMATH_CALUDE_inequality_range_l3492_349291

theorem inequality_range (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, |2*a - b| + |a + b| ≥ |a| * (|x - 1| + |x + 1|)) →
  x ∈ Set.Icc (-3/2) (3/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3492_349291


namespace NUMINAMATH_CALUDE_trigonometric_problem_l3492_349294

open Real

theorem trigonometric_problem (α : Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : sin α = 2 * Real.sqrt 5 / 5) :
  (tan α = 2) ∧ 
  ((4 * sin (π - α) + 2 * cos (2 * π - α)) / (sin (π/2 - α) - sin α) = -10) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l3492_349294


namespace NUMINAMATH_CALUDE_square_mod_four_l3492_349284

theorem square_mod_four (n : ℤ) : (n^2) % 4 = 0 ∨ (n^2) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_mod_four_l3492_349284


namespace NUMINAMATH_CALUDE_units_digit_of_five_to_ten_l3492_349202

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 5^10 is 5 -/
theorem units_digit_of_five_to_ten : unitsDigit (5^10) = 5 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_five_to_ten_l3492_349202


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3492_349282

theorem polynomial_evaluation : 
  let x : ℤ := -2
  2 * x^4 + 3 * x^3 - x^2 + 2 * x + 5 = 5 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3492_349282


namespace NUMINAMATH_CALUDE_liz_jump_shots_liz_jump_shots_correct_l3492_349260

theorem liz_jump_shots (initial_deficit : ℕ) (free_throws : ℕ) (three_pointers : ℕ) 
  (opponent_points : ℕ) (final_deficit : ℕ) : ℕ :=
  let free_throw_points := free_throws * 1
  let three_pointer_points := three_pointers * 3
  let total_deficit := initial_deficit + opponent_points
  let points_needed := total_deficit - final_deficit
  let jump_shot_points := points_needed - free_throw_points - three_pointer_points
  jump_shot_points / 2

theorem liz_jump_shots_correct :
  liz_jump_shots 20 5 3 10 8 = 4 := by sorry

end NUMINAMATH_CALUDE_liz_jump_shots_liz_jump_shots_correct_l3492_349260


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3492_349252

theorem triangle_angle_measure (A B C : ℝ) : 
  -- Triangle ABC
  A + B + C = 180 →
  -- Angle C is triple angle B
  C = 3 * B →
  -- Angle B is 15°
  B = 15 →
  -- Then angle A is 120°
  A = 120 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3492_349252


namespace NUMINAMATH_CALUDE_trajectory_of_P_max_distance_to_L_min_distance_to_L_l3492_349205

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point M on circle C
def M (x₀ y₀ : ℝ) : Prop := C x₀ y₀

-- Define point N
def N : ℝ × ℝ := (4, 0)

-- Define point P as midpoint of MN
def P (x y x₀ y₀ : ℝ) : Prop := x = (x₀ + 4) / 2 ∧ y = y₀ / 2

-- Theorem for the trajectory of P
theorem trajectory_of_P (x y : ℝ) : 
  (∃ x₀ y₀, M x₀ y₀ ∧ P x y x₀ y₀) → (x - 2)^2 + y^2 = 1 :=
sorry

-- Define the line L: 3x + 4y - 26 = 0
def L (x y : ℝ) : Prop := 3*x + 4*y - 26 = 0

-- Theorem for maximum distance
theorem max_distance_to_L (x y : ℝ) :
  (∃ x₀ y₀, M x₀ y₀ ∧ P x y x₀ y₀) → 
  (∀ x' y', (∃ x₀' y₀', M x₀' y₀' ∧ P x' y' x₀' y₀') → 
    |3*x + 4*y - 26| / Real.sqrt 25 ≤ 5) :=
sorry

-- Theorem for minimum distance
theorem min_distance_to_L (x y : ℝ) :
  (∃ x₀ y₀, M x₀ y₀ ∧ P x y x₀ y₀) → 
  (∀ x' y', (∃ x₀' y₀', M x₀' y₀' ∧ P x' y' x₀' y₀') → 
    |3*x + 4*y - 26| / Real.sqrt 25 ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_max_distance_to_L_min_distance_to_L_l3492_349205


namespace NUMINAMATH_CALUDE_min_abs_z_complex_l3492_349295

theorem min_abs_z_complex (z : ℂ) (h : Complex.abs (z - 2*I) + Complex.abs (z - 5) = 7) :
  ∃ (w : ℂ), Complex.abs w = 10/7 ∧ ∀ z', Complex.abs (z' - 2*I) + Complex.abs (z' - 5) = 7 → Complex.abs w ≤ Complex.abs z' :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_complex_l3492_349295


namespace NUMINAMATH_CALUDE_expression_value_l3492_349263

theorem expression_value (x : ℝ) (h : 4 * x^2 - 2 * x + 5 = 7) :
  2 * (x^2 - x) - (x - 1) + (2 * x + 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3492_349263


namespace NUMINAMATH_CALUDE_infinitely_many_real_roots_l3492_349219

theorem infinitely_many_real_roots : Set.Infinite {x : ℝ | ∃ y : ℝ, y^2 = -(x+1)^3} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_real_roots_l3492_349219


namespace NUMINAMATH_CALUDE_perfect_square_sum_in_partition_l3492_349293

theorem perfect_square_sum_in_partition (n : ℕ) (A B : Set ℕ) 
  (h1 : n ≥ 15)
  (h2 : A ⊆ Finset.range (n + 1))
  (h3 : B ⊆ Finset.range (n + 1))
  (h4 : A ∩ B = ∅)
  (h5 : A ∪ B = Finset.range (n + 1))
  (h6 : A ≠ Finset.range (n + 1))
  (h7 : B ≠ Finset.range (n + 1)) :
  ∃ (x y : ℕ), (x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ ∃ (k : ℕ), x + y = k^2) ∨
               (x ∈ B ∧ y ∈ B ∧ x ≠ y ∧ ∃ (k : ℕ), x + y = k^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_sum_in_partition_l3492_349293


namespace NUMINAMATH_CALUDE_cylinder_height_l3492_349274

/-- The height of a right cylinder with radius 2 feet and surface area 12π square feet is 1 foot. -/
theorem cylinder_height (π : ℝ) (h : ℝ) : 
  (2 * π * 2^2 + 2 * π * 2 * h = 12 * π) → h = 1 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_l3492_349274


namespace NUMINAMATH_CALUDE_turkey_roasting_problem_l3492_349271

/-- Represents the turkey roasting problem --/
structure TurkeyRoasting where
  turkeyWeight : ℕ
  roastingTimePerPound : ℕ
  roastingStartTime : ℕ
  dinnerTime : ℕ

/-- Calculates the maximum number of turkeys that can be roasted --/
def maxTurkeys (tr : TurkeyRoasting) : ℕ :=
  let totalRoastingTime := tr.dinnerTime - tr.roastingStartTime
  let roastingTimePerTurkey := tr.turkeyWeight * tr.roastingTimePerPound
  totalRoastingTime / roastingTimePerTurkey

/-- Theorem statement for the turkey roasting problem --/
theorem turkey_roasting_problem :
  let tr : TurkeyRoasting := {
    turkeyWeight := 16,
    roastingTimePerPound := 15,
    roastingStartTime := 10 * 60,  -- 10:00 am in minutes
    dinnerTime := 18 * 60  -- 6:00 pm in minutes
  }
  maxTurkeys tr = 2 := by
  sorry


end NUMINAMATH_CALUDE_turkey_roasting_problem_l3492_349271


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3492_349230

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 : 
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3492_349230


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3492_349278

theorem intersection_of_sets :
  let P : Set ℕ := {1, 3, 5}
  let Q : Set ℕ := {x | 2 ≤ x ∧ x ≤ 5}
  P ∩ Q = {3, 5} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3492_349278


namespace NUMINAMATH_CALUDE_tetrahedron_count_is_twelve_l3492_349268

/-- A regular triangular prism -/
structure RegularTriangularPrism where
  vertices : Finset (Fin 6)
  vertex_count : vertices.card = 6

/-- The number of ways to choose 4 vertices from 6 -/
def choose_four (prism : RegularTriangularPrism) : Nat :=
  Nat.choose 6 4

/-- The number of cases where 4 chosen points are coplanar -/
def coplanar_cases : Nat := 3

/-- The number of tetrahedrons that can be formed -/
def tetrahedron_count (prism : RegularTriangularPrism) : Nat :=
  choose_four prism - coplanar_cases

/-- Theorem: The number of tetrahedrons is 12 -/
theorem tetrahedron_count_is_twelve (prism : RegularTriangularPrism) :
  tetrahedron_count prism = 12 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_count_is_twelve_l3492_349268


namespace NUMINAMATH_CALUDE_spinner_probability_l3492_349264

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_D = 1/6 → p_A + p_B + p_C + p_D = 1 → p_C = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3492_349264


namespace NUMINAMATH_CALUDE_match_box_dozens_l3492_349261

theorem match_box_dozens (total_matches : ℕ) (matches_per_box : ℕ) (boxes_per_dozen : ℕ) : 
  total_matches = 1200 →
  matches_per_box = 20 →
  boxes_per_dozen = 12 →
  (total_matches / matches_per_box) / boxes_per_dozen = 5 :=
by sorry

end NUMINAMATH_CALUDE_match_box_dozens_l3492_349261


namespace NUMINAMATH_CALUDE_condition_relationship_l3492_349298

theorem condition_relationship : 
  (∀ x : ℝ, (0 < x ∧ x < 1) → x^2 < 1) ∧ 
  (∃ x : ℝ, x^2 < 1 ∧ ¬(0 < x ∧ x < 1)) := by
sorry

end NUMINAMATH_CALUDE_condition_relationship_l3492_349298


namespace NUMINAMATH_CALUDE_city_population_l3492_349297

theorem city_population (known_percentage : ℝ) (known_population : ℕ) (total_population : ℕ) : 
  known_percentage = 96 / 100 →
  known_population = 23040 →
  (known_percentage * total_population : ℝ) = known_population →
  total_population = 24000 := by
  sorry

end NUMINAMATH_CALUDE_city_population_l3492_349297


namespace NUMINAMATH_CALUDE_super_bowl_commercial_break_l3492_349273

/-- The duration of a commercial break with a given number of 5-minute and 2-minute commercials -/
def commercial_break_duration (five_min_count : ℕ) (two_min_count : ℕ) : ℕ :=
  5 * five_min_count + 2 * two_min_count

/-- Theorem stating that a commercial break with three 5-minute commercials and eleven 2-minute commercials lasts 37 minutes -/
theorem super_bowl_commercial_break :
  commercial_break_duration 3 11 = 37 := by
  sorry

end NUMINAMATH_CALUDE_super_bowl_commercial_break_l3492_349273


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3492_349220

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence where a_4 = 5 and a_8 = 6, a_2 * a_10 = 30 -/
theorem geometric_sequence_product (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_a4 : a 4 = 5) 
    (h_a8 : a 8 = 6) : 
  a 2 * a 10 = 30 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l3492_349220


namespace NUMINAMATH_CALUDE_total_onions_grown_l3492_349267

theorem total_onions_grown (sara_onions sally_onions fred_onions : ℕ)
  (h1 : sara_onions = 4)
  (h2 : sally_onions = 5)
  (h3 : fred_onions = 9) :
  sara_onions + sally_onions + fred_onions = 18 := by
sorry

end NUMINAMATH_CALUDE_total_onions_grown_l3492_349267


namespace NUMINAMATH_CALUDE_value_of_a_l3492_349201

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2

-- Theorem statement
theorem value_of_a (a : ℝ) : f_derivative a (-1) = 3 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3492_349201


namespace NUMINAMATH_CALUDE_polynomial_difference_l3492_349259

/-- A polynomial of degree 5 with specific properties -/
def f (a₁ a₂ a₃ a₄ a₅ : ℝ) (x : ℝ) : ℝ :=
  x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅

/-- The theorem statement -/
theorem polynomial_difference (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ m : ℝ, m ∈ ({1, 2, 3, 4} : Set ℝ) → f a₁ a₂ a₃ a₄ a₅ m = 2017 * m) →
  f a₁ a₂ a₃ a₄ a₅ 10 - f a₁ a₂ a₃ a₄ a₅ (-5) = 75615 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_difference_l3492_349259


namespace NUMINAMATH_CALUDE_min_distinct_values_l3492_349243

/-- Given a list of 3000 positive integers with a unique mode occurring exactly 12 times,
    the minimum number of distinct values in the list is 273. -/
theorem min_distinct_values (L : List ℕ+) : 
  L.length = 3000 →
  ∃! m : ℕ+, (L.count m = 12 ∧ ∀ n : ℕ+, L.count n ≤ L.count m) →
  L.toFinset.card ≥ 273 :=
by sorry

end NUMINAMATH_CALUDE_min_distinct_values_l3492_349243


namespace NUMINAMATH_CALUDE_max_overtakes_relay_race_l3492_349270

/-- Represents a relay race between two teams -/
structure RelayRace where
  num_runners : ℕ
  num_segments : ℕ
  runners_per_team : ℕ

/-- Represents the maximum number of overtakes in a relay race -/
def max_overtakes (race : RelayRace) : ℕ :=
  2 * (race.num_runners - 1)

/-- Theorem stating the maximum number of overtakes in the specific relay race scenario -/
theorem max_overtakes_relay_race :
  ∀ (race : RelayRace),
    race.num_runners = 20 →
    race.num_segments = 20 →
    race.runners_per_team = 20 →
    max_overtakes race = 38 := by
  sorry

end NUMINAMATH_CALUDE_max_overtakes_relay_race_l3492_349270


namespace NUMINAMATH_CALUDE_rectangle_square_ratio_l3492_349253

/-- Represents a right triangle with a rectangle and square inscribed as described in the problem -/
structure TriangleWithInscriptions where
  /-- Side lengths of the right triangle -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- Side lengths of the inscribed rectangle -/
  rect_side1 : ℝ
  rect_side2 : ℝ
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- Conditions for the triangle -/
  triangle_right : a^2 + b^2 = c^2
  triangle_sides : a = 5 ∧ b = 12 ∧ c = 13
  /-- Conditions for the rectangle -/
  rectangle_sides : rect_side1 = 5 ∧ rect_side2 = 12
  /-- Condition for the square -/
  square_formula : square_side = (a * b) / c

/-- The main theorem stating the ratio of the longer rectangle side to the square side -/
theorem rectangle_square_ratio (t : TriangleWithInscriptions) :
  t.rect_side2 / t.square_side = 13 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_ratio_l3492_349253


namespace NUMINAMATH_CALUDE_x_value_l3492_349280

theorem x_value (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3492_349280


namespace NUMINAMATH_CALUDE_partition_scores_with_equal_average_l3492_349289

theorem partition_scores_with_equal_average 
  (N : ℕ) 
  (scores : List ℤ) 
  (h_length : scores.length = 3 * N)
  (h_range : ∀ s ∈ scores, 60 ≤ s ∧ s ≤ 100)
  (h_freq : ∀ s ∈ scores, (scores.filter (· = s)).length ≥ 2)
  (h_avg : scores.sum / (3 * N) = 824 / 10) :
  ∃ (class1 class2 class3 : List ℤ),
    class1.length = N ∧ 
    class2.length = N ∧ 
    class3.length = N ∧
    scores = class1 ++ class2 ++ class3 ∧
    class1.sum / N = 824 / 10 ∧
    class2.sum / N = 824 / 10 ∧
    class3.sum / N = 824 / 10 :=
by sorry

end NUMINAMATH_CALUDE_partition_scores_with_equal_average_l3492_349289


namespace NUMINAMATH_CALUDE_union_equals_M_l3492_349247

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 0}
def N : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≠ 0}

theorem union_equals_M : M ∪ N = M := by sorry

end NUMINAMATH_CALUDE_union_equals_M_l3492_349247


namespace NUMINAMATH_CALUDE_prime_factorization_sum_l3492_349215

theorem prime_factorization_sum (w x y z k : ℕ) : 
  2^w * 3^x * 5^y * 7^z * 11^k = 2310 → 2*w + 3*x + 5*y + 7*z + 11*k = 28 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_sum_l3492_349215


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3492_349200

theorem quadratic_root_property (a : ℝ) : 
  (2 * a^2 = 6 * a - 4) → (a^2 - 3 * a + 2024 = 2022) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3492_349200


namespace NUMINAMATH_CALUDE_desert_area_changes_l3492_349251

/-- Represents the desert area problem -/
structure DesertArea where
  initial_area : ℝ  -- Initial desert area in 1997
  annual_increase : ℝ  -- Annual increase in desert area
  afforestation_rate : ℝ  -- Annual reduction due to afforestation measures

/-- Calculates the desert area after a given number of years without afforestation -/
def area_after_years (d : DesertArea) (years : ℕ) : ℝ :=
  d.initial_area + d.annual_increase * years

/-- Calculates the desert area after a given number of years with afforestation -/
def area_with_afforestation (d : DesertArea) (years : ℕ) : ℝ :=
  d.initial_area + d.annual_increase * years - d.afforestation_rate * years

/-- Main theorem about desert area changes -/
theorem desert_area_changes (d : DesertArea) 
    (h1 : d.initial_area = 9e5)
    (h2 : d.annual_increase = 2000)
    (h3 : d.afforestation_rate = 8000) :
    area_after_years d 23 = 9.46e5 ∧ 
    (∃ (y : ℕ), y ≤ 19 ∧ area_with_afforestation d y < 8e5 ∧ 
                ∀ (z : ℕ), z < y → area_with_afforestation d z ≥ 8e5) :=
  sorry


end NUMINAMATH_CALUDE_desert_area_changes_l3492_349251


namespace NUMINAMATH_CALUDE_six_power_plus_one_all_digits_same_l3492_349258

/-- A number has all digits the same in its decimal representation -/
def all_digits_same (m : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ ∀ k : ℕ, (m / 10^k) % 10 = d ∨ m / 10^k = 0

/-- The set of positive integers n for which 6^n + 1 has all digits the same -/
def S : Set ℕ := {n : ℕ | n > 0 ∧ all_digits_same (6^n + 1)}

theorem six_power_plus_one_all_digits_same :
  S = {1, 5} :=
sorry

end NUMINAMATH_CALUDE_six_power_plus_one_all_digits_same_l3492_349258


namespace NUMINAMATH_CALUDE_total_shaded_area_l3492_349245

/-- Given a square carpet with the following properties:
  * Total side length of 16 feet
  * Contains one large shaded square and twelve smaller congruent shaded squares
  * Ratio of carpet side to large shaded square side (S) is 4:1
  * Ratio of large shaded square side (S) to smaller shaded square side (T) is 2:1
  The total shaded area is 64 square feet. -/
theorem total_shaded_area (carpet_side : ℝ) (S T : ℝ) : 
  carpet_side = 16 ∧ 
  carpet_side / S = 4 ∧ 
  S / T = 2 → 
  S^2 + 12 * T^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_shaded_area_l3492_349245


namespace NUMINAMATH_CALUDE_amusement_park_tickets_l3492_349211

theorem amusement_park_tickets (total_cost : ℕ) (adult_price child_price : ℕ) (adult_child_diff : ℕ) : 
  total_cost = 720 →
  adult_price = 15 →
  child_price = 8 →
  adult_child_diff = 25 →
  ∃ (num_children : ℕ), 
    num_children * child_price + (num_children + adult_child_diff) * adult_price = total_cost ∧ 
    num_children = 15 :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_tickets_l3492_349211


namespace NUMINAMATH_CALUDE_hexagon_area_is_32_l3492_349244

/-- A hexagon surrounded by triangles forming a rectangle -/
structure HexagonWithTriangles where
  num_triangles : ℕ
  triangle_area : ℝ
  rectangle_area : ℝ

/-- The area of the hexagon -/
def hexagon_area (h : HexagonWithTriangles) : ℝ :=
  h.rectangle_area - h.num_triangles * h.triangle_area

/-- Theorem: The area of the hexagon is 32 square units -/
theorem hexagon_area_is_32 (h : HexagonWithTriangles) 
    (h_num_triangles : h.num_triangles = 4)
    (h_triangle_area : h.triangle_area = 2)
    (h_rectangle_area : h.rectangle_area = 40) : 
  hexagon_area h = 32 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_is_32_l3492_349244


namespace NUMINAMATH_CALUDE_abs_neg_three_halves_l3492_349234

theorem abs_neg_three_halves : |(-3/2 : ℚ)| = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_halves_l3492_349234

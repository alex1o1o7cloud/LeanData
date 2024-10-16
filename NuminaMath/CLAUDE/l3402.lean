import Mathlib

namespace NUMINAMATH_CALUDE_min_white_surface_area_l3402_340260

/-- Represents a cube with some faces painted gray and others white -/
structure PaintedCube where
  grayFaces : Fin 6 → Bool

/-- The number of identical structures -/
def numStructures : Nat := 7

/-- The number of cubes in each structure -/
def cubesPerStructure : Nat := 8

/-- The number of additional white cubes -/
def additionalWhiteCubes : Nat := 8

/-- The edge length of each small cube in cm -/
def smallCubeEdgeLength : ℝ := 1

/-- The total number of cubes used to construct the large cube -/
def totalCubes : Nat := numStructures * cubesPerStructure + additionalWhiteCubes

/-- The edge length of the large cube in terms of small cubes -/
def largeCubeEdgeLength : Nat := 4

/-- The surface area of the large cube in cm² -/
def largeCubeSurfaceArea : ℝ := 6 * (largeCubeEdgeLength * largeCubeEdgeLength : ℝ) * smallCubeEdgeLength ^ 2

/-- A function to calculate the maximum possible gray surface area -/
def maxGraySurfaceArea : ℝ := 84

/-- Theorem stating that the minimum white surface area is 12 cm² -/
theorem min_white_surface_area :
  largeCubeSurfaceArea - maxGraySurfaceArea = 12 := by sorry

end NUMINAMATH_CALUDE_min_white_surface_area_l3402_340260


namespace NUMINAMATH_CALUDE_vector_collinearity_l3402_340212

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2) ∨ w = (k * v.1, k * v.2)

theorem vector_collinearity :
  let m : ℝ × ℝ := (0, -2)
  let n : ℝ × ℝ := (Real.sqrt 3, 1)
  let v : ℝ × ℝ := (-1, Real.sqrt 3)
  collinear (2 * m.1 + n.1, 2 * m.2 + n.2) v := by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l3402_340212


namespace NUMINAMATH_CALUDE_square_triangle_area_ratio_l3402_340203

theorem square_triangle_area_ratio : 
  ∀ (s t : ℝ), s > 0 → t > 0 → 
  s^2 = (t^2 * Real.sqrt 3) / 4 → 
  s / t = (Real.sqrt (Real.sqrt 3)) / 2 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_area_ratio_l3402_340203


namespace NUMINAMATH_CALUDE_smallest_N_l3402_340286

theorem smallest_N (k : ℕ) (hk : k ≥ 1) :
  let N := 2 * k^3 + 3 * k^2 + k
  ∀ (a : Fin (2 * k + 1) → ℕ),
    (∀ i, a i ≥ 1) →
    (∀ i j, i ≠ j → a i ≠ a j) →
    (Finset.sum Finset.univ a ≥ N) →
    (∀ s : Finset (Fin (2 * k + 1)), s.card = k → Finset.sum s a ≤ N / 2) →
    ∀ M : ℕ, M < N →
      ¬∃ (b : Fin (2 * k + 1) → ℕ),
        (∀ i, b i ≥ 1) ∧
        (∀ i j, i ≠ j → b i ≠ b j) ∧
        (Finset.sum Finset.univ b ≥ M) ∧
        (∀ s : Finset (Fin (2 * k + 1)), s.card = k → Finset.sum s b ≤ M / 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_N_l3402_340286


namespace NUMINAMATH_CALUDE_game_savings_ratio_l3402_340245

theorem game_savings_ratio (game_cost : ℝ) (tax_rate : ℝ) (weekly_allowance : ℝ) (weeks_to_save : ℕ) :
  game_cost = 50 →
  tax_rate = 0.1 →
  weekly_allowance = 10 →
  weeks_to_save = 11 →
  (weekly_allowance / weekly_allowance : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_game_savings_ratio_l3402_340245


namespace NUMINAMATH_CALUDE_square_gt_one_vs_cube_gt_one_l3402_340218

theorem square_gt_one_vs_cube_gt_one :
  {a : ℝ | a^2 > 1} ⊃ {a : ℝ | a^3 > 1} ∧ {a : ℝ | a^2 > 1} ≠ {a : ℝ | a^3 > 1} :=
by sorry

end NUMINAMATH_CALUDE_square_gt_one_vs_cube_gt_one_l3402_340218


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l3402_340271

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  side_count : ℕ
  total_cubes : ℕ
  inner_cubes : ℕ

/-- The number of smaller cubes with no faces colored in a painted cube cut into 64 equal parts -/
def painted_cube_inner_count : ℕ := 8

/-- Theorem: In a cube cut into 64 equal smaller cubes, 
    the number of smaller cubes with no faces touching the original cube's surface is 8 -/
theorem painted_cube_theorem (c : CutCube) 
  (h1 : c.side_count = 4)
  (h2 : c.total_cubes = 64)
  (h3 : c.inner_cubes = (c.side_count - 2)^3) :
  c.inner_cubes = painted_cube_inner_count := by sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l3402_340271


namespace NUMINAMATH_CALUDE_impossible_arrangement_l3402_340242

def numbers : List ℕ := [1, 4, 9, 16, 25, 36, 49, 64, 81]

def radial_lines : ℕ := 6

def appears_twice (n : ℕ) : Prop := ∃ (l₁ l₂ : List ℕ), l₁ ≠ l₂ ∧ n ∈ l₁ ∧ n ∈ l₂

theorem impossible_arrangement : 
  ¬∃ (arrangement : List (List ℕ)), 
    (∀ n ∈ numbers, appears_twice n) ∧ 
    (arrangement.length = radial_lines) ∧
    (∃ (s : ℕ), ∀ l ∈ arrangement, l.sum = s) :=
sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l3402_340242


namespace NUMINAMATH_CALUDE_final_position_16_meters_l3402_340296

/-- Represents a back-and-forth race between two runners -/
structure Race where
  distance : ℝ  -- Total distance of the race (one way)
  meetPoint : ℝ  -- Distance from B to meeting point
  catchPoint : ℝ  -- Distance from finish when B catches A

/-- Calculates the final position of runner A when B finishes the race -/
def finalPositionA (race : Race) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the final position of A in the given race scenario -/
theorem final_position_16_meters (race : Race) 
  (h1 : race.meetPoint = 24)
  (h2 : race.catchPoint = 48) :
  finalPositionA race = 16 := by
  sorry

end NUMINAMATH_CALUDE_final_position_16_meters_l3402_340296


namespace NUMINAMATH_CALUDE_shifted_function_eq_minus_three_x_minus_four_l3402_340266

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Shifts a linear function vertically by a given amount -/
def shift_vertical (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + shift }

/-- The original linear function y = -3x -/
def original_function : LinearFunction :=
  { m := -3, b := 0 }

/-- The amount to shift the function down -/
def shift_amount : ℝ := -4

theorem shifted_function_eq_minus_three_x_minus_four :
  shift_vertical original_function shift_amount = { m := -3, b := -4 } := by
  sorry

end NUMINAMATH_CALUDE_shifted_function_eq_minus_three_x_minus_four_l3402_340266


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l3402_340206

theorem tan_45_degrees_equals_one :
  let θ : Real := 45 * π / 180  -- Convert 45 degrees to radians
  let tan_θ := Real.tan θ
  let sin_θ := Real.sin θ
  let cos_θ := Real.cos θ
  (∀ α, Real.tan α = Real.sin α / Real.cos α) →  -- General tangent identity
  sin_θ = Real.sqrt 2 / 2 →  -- Given value of sin 45°
  cos_θ = Real.sqrt 2 / 2 →  -- Given value of cos 45°
  tan_θ = 1 := by
sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l3402_340206


namespace NUMINAMATH_CALUDE_car_speed_problem_l3402_340244

/-- The speed of Car B in km/h -/
def speed_B : ℝ := 35

/-- The time it takes Car A to catch up with Car B when traveling at 50 km/h -/
def time_1 : ℝ := 6

/-- The time it takes Car A to catch up with Car B when traveling at 80 km/h -/
def time_2 : ℝ := 2

/-- The speed of Car A in the first scenario (km/h) -/
def speed_A_1 : ℝ := 50

/-- The speed of Car A in the second scenario (km/h) -/
def speed_A_2 : ℝ := 80

theorem car_speed_problem :
  speed_B * time_1 = speed_A_1 * time_1 - (time_1 - time_2) * speed_B ∧
  speed_B * time_2 = speed_A_2 * time_2 - (time_1 - time_2) * speed_B :=
by
  sorry

#check car_speed_problem

end NUMINAMATH_CALUDE_car_speed_problem_l3402_340244


namespace NUMINAMATH_CALUDE_x_value_in_set_A_l3402_340250

-- Define the set A
def A (x : ℝ) : Set ℝ := {0, -1, x}

-- Theorem statement
theorem x_value_in_set_A (x : ℝ) (h1 : x^2 ∈ A x) (h2 : 0 ≠ -1 ∧ 0 ≠ x ∧ -1 ≠ x) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_set_A_l3402_340250


namespace NUMINAMATH_CALUDE_range_of_a_l3402_340214

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 0}
def B (a : ℝ) : Set ℝ := {1, 3, a}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a ≠ ∅ → a ∈ A := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3402_340214


namespace NUMINAMATH_CALUDE_tank_filling_capacity_l3402_340281

/-- Given a tank that can be filled with 28 buckets of 13.5 litres each,
    prove that if the same tank can be filled with 42 buckets of equal capacity,
    then the capacity of each bucket in the second case is 9 litres. -/
theorem tank_filling_capacity (tank_volume : ℝ) (bucket_count_1 bucket_count_2 : ℕ) 
    (bucket_capacity_1 : ℝ) :
  tank_volume = bucket_count_1 * bucket_capacity_1 →
  bucket_count_1 = 28 →
  bucket_capacity_1 = 13.5 →
  bucket_count_2 = 42 →
  ∃ bucket_capacity_2 : ℝ, 
    tank_volume = bucket_count_2 * bucket_capacity_2 ∧
    bucket_capacity_2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_tank_filling_capacity_l3402_340281


namespace NUMINAMATH_CALUDE_debate_team_max_groups_l3402_340233

/-- Given a debate team with boys and girls, calculate the maximum number of groups
    that can be formed with a minimum number of boys and girls per group. -/
def max_groups (num_boys num_girls min_boys_per_group min_girls_per_group : ℕ) : ℕ :=
  min (num_boys / min_boys_per_group) (num_girls / min_girls_per_group)

/-- Theorem stating that for a debate team with 31 boys and 32 girls,
    where each group must have at least 2 boys and 3 girls,
    the maximum number of groups that can be formed is 10. -/
theorem debate_team_max_groups :
  max_groups 31 32 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_max_groups_l3402_340233


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3402_340280

theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 26) (h2 : DF = 15) (h3 : EF = 17) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3402_340280


namespace NUMINAMATH_CALUDE_frank_final_position_l3402_340284

/-- Represents Frank's position relative to his starting point -/
def dance_position (back1 forward1 back2 : ℤ) : ℤ :=
  -back1 + forward1 - back2 + 2 * back2

/-- Proves that Frank's final position is 7 steps forward from his starting point -/
theorem frank_final_position :
  dance_position 5 10 2 = 7 := by sorry

end NUMINAMATH_CALUDE_frank_final_position_l3402_340284


namespace NUMINAMATH_CALUDE_given_equation_is_quadratic_l3402_340254

/-- Represents a quadratic equation in standard form -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0

/-- The given equation: 3(x+1)^2 = 2(x+1) -/
def given_equation (x : ℝ) : Prop :=
  3 * (x + 1)^2 = 2 * (x + 1)

/-- Theorem stating that the given equation is equivalent to a quadratic equation -/
theorem given_equation_is_quadratic :
  ∃ (q : QuadraticEquation), ∀ x, given_equation x ↔ q.a * x^2 + q.b * x + q.c = 0 :=
sorry

end NUMINAMATH_CALUDE_given_equation_is_quadratic_l3402_340254


namespace NUMINAMATH_CALUDE_probability_one_good_product_l3402_340294

def total_products : ℕ := 5
def good_products : ℕ := 3
def defective_products : ℕ := 2
def selections : ℕ := 2

theorem probability_one_good_product : 
  (Nat.choose good_products 1 * Nat.choose defective_products 1) / 
  Nat.choose total_products selections = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_one_good_product_l3402_340294


namespace NUMINAMATH_CALUDE_integer_sum_l3402_340285

theorem integer_sum (x y : ℕ+) (h1 : x.val - y.val = 8) (h2 : x.val * y.val = 288) : 
  x.val + y.val = 35 := by
sorry

end NUMINAMATH_CALUDE_integer_sum_l3402_340285


namespace NUMINAMATH_CALUDE_max_value_of_g_l3402_340269

/-- The function g(x) = 4x - x^4 --/
def g (x : ℝ) : ℝ := 4 * x - x^4

/-- The theorem stating that the maximum value of g(x) on [0, √4] is 3 --/
theorem max_value_of_g :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.sqrt 4 ∧
  g x = 3 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ Real.sqrt 4 → g y ≤ g x :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3402_340269


namespace NUMINAMATH_CALUDE_digit_equality_l3402_340200

theorem digit_equality (n k : ℕ) : 
  (10^(k-1) ≤ n^n ∧ n^n < 10^k) ∧ 
  (10^(n-1) ≤ k^k ∧ k^k < 10^n) ↔ 
  ((n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9)) :=
by sorry

end NUMINAMATH_CALUDE_digit_equality_l3402_340200


namespace NUMINAMATH_CALUDE_distance_traveled_l3402_340291

/-- Given a car's fuel efficiency and the amount of fuel used, calculate the distance traveled. -/
theorem distance_traveled (efficiency : ℝ) (fuel_used : ℝ) (h1 : efficiency = 20) (h2 : fuel_used = 3) :
  efficiency * fuel_used = 60 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3402_340291


namespace NUMINAMATH_CALUDE_hair_reaches_floor_simultaneously_l3402_340288

/-- Represents the growth rate of a person or their hair -/
structure GrowthRate where
  rate : ℝ

/-- Represents a person with their growth rate and hair growth rate -/
structure Person where
  growth : GrowthRate
  hairGrowth : GrowthRate

/-- The rate at which the distance from hair to floor decreases -/
def hairToFloorRate (p : Person) : ℝ :=
  p.hairGrowth.rate - p.growth.rate

theorem hair_reaches_floor_simultaneously
  (katya alena : Person)
  (h1 : katya.hairGrowth.rate = 2 * katya.growth.rate)
  (h2 : alena.growth.rate = katya.hairGrowth.rate)
  (h3 : alena.hairGrowth.rate = 1.5 * alena.growth.rate) :
  hairToFloorRate katya = hairToFloorRate alena :=
sorry

end NUMINAMATH_CALUDE_hair_reaches_floor_simultaneously_l3402_340288


namespace NUMINAMATH_CALUDE_great_circle_bisects_angle_l3402_340264

-- Define the sphere
def Sphere : Type := ℝ × ℝ × ℝ

-- Define the north pole
def N : Sphere := (0, 0, 1)

-- Define a great circle
def GreatCircle (p q : Sphere) : Type := sorry

-- Define a point on the equator
def OnEquator (p : Sphere) : Prop := sorry

-- Define equidistance from a point
def Equidistant (a b c : Sphere) : Prop := sorry

-- Define angle bisection on a sphere
def AngleBisector (a b c d : Sphere) : Prop := sorry

-- Theorem statement
theorem great_circle_bisects_angle 
  (A B C : Sphere) 
  (h1 : GreatCircle N A)
  (h2 : GreatCircle N B)
  (h3 : Equidistant N A B)
  (h4 : OnEquator C) :
  AngleBisector C N A B :=
sorry

end NUMINAMATH_CALUDE_great_circle_bisects_angle_l3402_340264


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3402_340240

theorem complex_equation_solution (x y : ℝ) :
  (2 * x - 1 : ℂ) + I = -y - (3 - y) * I →
  x = -3/2 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3402_340240


namespace NUMINAMATH_CALUDE_triangle_area_l3402_340297

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  b + c = 5 →
  Real.tan B + Real.tan C + Real.sqrt 3 = Real.sqrt 3 * Real.tan B * Real.tan C →
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3402_340297


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l3402_340231

/-- The value of p for a parabola y^2 = 2px where the distance between (-2, 3) and the focus is 5 -/
theorem parabola_focus_distance (p : ℝ) : 
  p > 0 → -- Condition that p is positive
  let focus : ℝ × ℝ := (p/2, 0) -- Definition of focus for parabola y^2 = 2px
  (((-2 : ℝ) - p/2)^2 + 3^2).sqrt = 5 → -- Distance formula between (-2, 3) and focus is 5
  p = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l3402_340231


namespace NUMINAMATH_CALUDE_sam_money_value_l3402_340299

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The number of pennies Sam has -/
def num_pennies : ℕ := 9

/-- The number of quarters Sam has -/
def num_quarters : ℕ := 7

/-- The total value of Sam's money in dollars -/
def total_value : ℚ := num_pennies * penny_value + num_quarters * quarter_value

theorem sam_money_value : total_value = 184 / 100 := by sorry

end NUMINAMATH_CALUDE_sam_money_value_l3402_340299


namespace NUMINAMATH_CALUDE_gear_rpm_problem_l3402_340265

/-- The number of revolutions per minute for gear q -/
def q_rpm : ℝ := 40

/-- The duration in minutes -/
def duration : ℝ := 0.5

/-- The difference in revolutions between gear q and gear p after 30 seconds -/
def revolution_difference : ℝ := 15

/-- The number of revolutions per minute for gear p -/
def p_rpm : ℝ := 10

theorem gear_rpm_problem :
  q_rpm * duration - revolution_difference = p_rpm * duration :=
by sorry

end NUMINAMATH_CALUDE_gear_rpm_problem_l3402_340265


namespace NUMINAMATH_CALUDE_train_speed_train_speed_approximately_60_l3402_340293

/-- The speed of a train given its length, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_speed (train_length : ℝ) (man_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_length / passing_time
  let train_speed_ms := relative_speed - (man_speed * 1000 / 3600)
  let train_speed_kmh := train_speed_ms * 3600 / 1000
  train_speed_kmh

/-- The speed of the train is approximately 60 km/hr given the specified conditions. -/
theorem train_speed_approximately_60 :
  ∃ ε > 0, abs (train_speed 220 6 11.999040076793857 - 60) < ε :=
sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_approximately_60_l3402_340293


namespace NUMINAMATH_CALUDE_red_balls_count_l3402_340282

theorem red_balls_count (total : ℕ) (prob : ℚ) : 
  total = 15 →
  prob = 1 / 21 →
  ∃ (r : ℕ), r ≤ total ∧ 
    (r : ℚ) / total * ((r : ℚ) - 1) / (total - 1 : ℚ) = prob ∧
    r = 5 :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l3402_340282


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3402_340207

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  (∃ a b : ℝ, a + b > 0 ∧ a * b ≤ 0) ∧ 
  (∃ a b : ℝ, a * b > 0 ∧ a + b ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3402_340207


namespace NUMINAMATH_CALUDE_total_cost_is_2000_l3402_340289

/-- The cost of buying two laptops, where the first laptop costs $500 and the second laptop is 3 times as costly as the first laptop. -/
def total_cost (first_laptop_cost : ℕ) (cost_multiplier : ℕ) : ℕ :=
  first_laptop_cost + (cost_multiplier * first_laptop_cost)

/-- Theorem stating that the total cost of buying both laptops is $2000. -/
theorem total_cost_is_2000 : total_cost 500 3 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_2000_l3402_340289


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_ratio_l3402_340246

noncomputable def f₁ (a : ℝ) (x : ℝ) : ℝ := x^2 - x + 2*a
noncomputable def f₂ (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x + 3
noncomputable def f₃ (a b : ℝ) (x : ℝ) : ℝ := 4*x^2 + (2*b-3)*x + 6*a + 3
noncomputable def f₄ (a b : ℝ) (x : ℝ) : ℝ := 4*x^2 + (6*b-1)*x + 9 + 2*a

noncomputable def A (a : ℝ) : ℝ := Real.sqrt (1 - 8*a)
noncomputable def B (b : ℝ) : ℝ := Real.sqrt (4*b^2 - 12)
noncomputable def C (a b : ℝ) : ℝ := (1/4) * Real.sqrt ((2*b - 3)^2 - 64*(6*a + 3))
noncomputable def D (a b : ℝ) : ℝ := (1/4) * Real.sqrt ((6*b - 1)^2 - 64*(9 + 2*a))

theorem quadratic_roots_difference_ratio (a b : ℝ) (h : A a ≠ B b) :
  (C a b^2 - D a b^2) / (A a^2 - B b^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_ratio_l3402_340246


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3402_340248

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define a geometric sequence
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  arithmetic_sequence a d →
  a 3 = 7 →
  geometric_sequence (λ n => a n - 1) →
  a 10 = 21 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3402_340248


namespace NUMINAMATH_CALUDE_kenneth_earnings_l3402_340223

/-- Kenneth's earnings problem -/
theorem kenneth_earnings (earnings : ℝ) 
  (h1 : earnings * 0.1 + earnings * 0.15 + 75 + 80 + 405 = earnings) : 
  earnings = 746.67 := by
sorry

end NUMINAMATH_CALUDE_kenneth_earnings_l3402_340223


namespace NUMINAMATH_CALUDE_trig_identity_l3402_340261

theorem trig_identity (θ : Real) (h : Real.sin θ + Real.cos θ = Real.sqrt 2) :
  Real.tan θ + (Real.tan θ)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3402_340261


namespace NUMINAMATH_CALUDE_solve_refrigerator_problem_l3402_340251

def refrigerator_problem (refrigerator_price : ℝ) : Prop :=
  let mobile_price : ℝ := 8000
  let refrigerator_sold : ℝ := refrigerator_price * 0.96
  let mobile_sold : ℝ := mobile_price * 1.1
  let total_bought : ℝ := refrigerator_price + mobile_price
  let total_sold : ℝ := refrigerator_sold + mobile_sold
  let profit : ℝ := 200
  total_sold = total_bought + profit

theorem solve_refrigerator_problem :
  ∃ (price : ℝ), refrigerator_problem price ∧ price = 15000 := by
  sorry

end NUMINAMATH_CALUDE_solve_refrigerator_problem_l3402_340251


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_leg_length_l3402_340237

/-- An isosceles right triangle with a median to the hypotenuse of length 15 units has legs of length 15√2 units. -/
theorem isosceles_right_triangle_leg_length :
  ∀ (a b c m : ℝ),
  a = b →                          -- The triangle is isosceles
  a^2 + b^2 = c^2 →                -- The triangle is right-angled (Pythagorean theorem)
  m = 15 →                         -- The median to the hypotenuse is 15 units
  m = c / 2 →                      -- The median to the hypotenuse is half the hypotenuse length
  a = 15 * Real.sqrt 2 :=           -- The leg length is 15√2
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_leg_length_l3402_340237


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3402_340256

theorem fraction_to_decimal : (49 : ℚ) / 160 = 0.30625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3402_340256


namespace NUMINAMATH_CALUDE_find_a_l3402_340238

theorem find_a (b w : ℝ) (h1 : b = 2120) (h2 : w = 0.5) : ∃ a : ℝ, w = a / b ∧ a = 1060 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3402_340238


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_4_8_l3402_340210

theorem gcf_lcm_sum_4_8 : Nat.gcd 4 8 + Nat.lcm 4 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_4_8_l3402_340210


namespace NUMINAMATH_CALUDE_simplified_root_expression_l3402_340263

theorem simplified_root_expression : 
  ∃ (a b : ℕ), (a > 0 ∧ b > 0) ∧ 
  (3^5 * 5^4)^(1/4) = a * b^(1/4) ∧ 
  a + b = 18 := by sorry

end NUMINAMATH_CALUDE_simplified_root_expression_l3402_340263


namespace NUMINAMATH_CALUDE_seth_oranges_l3402_340247

def boxes_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  let remaining := initial - given_away
  remaining - (remaining / 2)

theorem seth_oranges :
  boxes_left 9 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_seth_oranges_l3402_340247


namespace NUMINAMATH_CALUDE_sequence_ratio_l3402_340272

/-- Given an arithmetic sequence and a geometric sequence with specific properties, 
    prove that the ratio of the sum of certain terms to another term equals 5/2. -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (1 : ℝ) < a₁ ∧ a₁ < a₂ ∧ a₂ < 4 ∧  -- arithmetic sequence condition
  (∃ r : ℝ, r > 0 ∧ b₁ = r ∧ b₂ = r^2 ∧ b₃ = r^3 ∧ 4 = r^4) →  -- geometric sequence condition
  (a₁ + a₂) / b₂ = 5/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_ratio_l3402_340272


namespace NUMINAMATH_CALUDE_collinear_vectors_l3402_340225

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem collinear_vectors (h1 : ¬ Collinear ℝ ({0, a, b} : Set V))
    (h2 : Collinear ℝ ({0, 2 • a + k • b, a - b} : Set V)) :
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l3402_340225


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l3402_340273

theorem largest_divisor_of_expression (x : ℤ) (h : Even x) :
  ∃ (k : ℤ), (8*x + 2) * (8*x + 4) * (4*x + 2) = 240 * k ∧
  ∀ (m : ℤ), m > 240 → ∃ (y : ℤ), Even y ∧ ¬∃ (l : ℤ), (8*y + 2) * (8*y + 4) * (4*y + 2) = m * l :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l3402_340273


namespace NUMINAMATH_CALUDE_race_time_calculation_l3402_340241

/-- 
Given a 100-meter race where:
- Runner A beats runner B by 20 meters
- Runner B finishes the race in 45 seconds

This theorem proves that runner A finishes the race in 36 seconds.
-/
theorem race_time_calculation (race_distance : ℝ) (b_time : ℝ) (distance_difference : ℝ) 
  (h1 : race_distance = 100)
  (h2 : b_time = 45)
  (h3 : distance_difference = 20) : 
  ∃ (a_time : ℝ), a_time = 36 ∧ 
  (race_distance / a_time = (race_distance - distance_difference) / a_time) ∧
  ((race_distance - distance_difference) / a_time = race_distance / b_time) :=
sorry

end NUMINAMATH_CALUDE_race_time_calculation_l3402_340241


namespace NUMINAMATH_CALUDE_sum_of_f_l3402_340276

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

theorem sum_of_f (x : ℝ) : f (-x) + f (1 + x) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_l3402_340276


namespace NUMINAMATH_CALUDE_triangle_area_l3402_340228

/-- Given a triangle ABC where cos A = 4/5 and AB · AC = 8, prove that its area is 3 -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let cos_A : ℝ := 4/5
  let dot_product : ℝ := (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)
  dot_product = 8 →
  (1/2) * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) * Real.sqrt (1 - cos_A^2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3402_340228


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l3402_340213

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2 + 1
def parabola2 (x y : ℝ) : Prop := x - 1 = (y + 2)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem intersection_sum_zero :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧
    (x₁, y₁) ≠ (x₃, y₃) ∧
    (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧
    (x₂, y₂) ≠ (x₄, y₄) ∧
    (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l3402_340213


namespace NUMINAMATH_CALUDE_calculate_expression_l3402_340202

theorem calculate_expression : (-3)^25 + 2^(4^2 + 5^2 - 7^2) + 3^3 = -3^25 + 27 + 1/256 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3402_340202


namespace NUMINAMATH_CALUDE_max_value_of_function_l3402_340239

theorem max_value_of_function (x : ℝ) : 
  (∀ x, -1 ≤ Real.cos x ∧ Real.cos x ≤ 1) → 
  ∃ y_max : ℝ, y_max = 4 ∧ ∀ x, 3 - Real.cos (x / 2) ≤ y_max := by
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3402_340239


namespace NUMINAMATH_CALUDE_max_l_pieces_theorem_max_l_pieces_5x10_max_l_pieces_5x9_l3402_340275

/-- Represents an L-shaped piece consisting of 3 cells -/
structure LPiece where
  size : Nat
  size_eq : size = 3

/-- Represents a rectangular grid -/
structure Grid where
  rows : Nat
  cols : Nat

/-- Calculates the maximum number of L-shaped pieces that can be cut from a grid -/
def maxLPieces (g : Grid) (l : LPiece) : Nat :=
  (g.rows * g.cols) / l.size

/-- Theorem: The maximum number of L-shaped pieces in a grid is the floor of total cells divided by piece size -/
theorem max_l_pieces_theorem (g : Grid) (l : LPiece) :
  maxLPieces g l = ⌊(g.rows * g.cols : ℚ) / l.size⌋ :=
sorry

/-- Corollary: For a 5x10 grid, the maximum number of L-shaped pieces is 16 -/
theorem max_l_pieces_5x10 :
  maxLPieces { rows := 5, cols := 10 } { size := 3, size_eq := rfl } = 16 :=
sorry

/-- Corollary: For a 5x9 grid, the maximum number of L-shaped pieces is 15 -/
theorem max_l_pieces_5x9 :
  maxLPieces { rows := 5, cols := 9 } { size := 3, size_eq := rfl } = 15 :=
sorry

end NUMINAMATH_CALUDE_max_l_pieces_theorem_max_l_pieces_5x10_max_l_pieces_5x9_l3402_340275


namespace NUMINAMATH_CALUDE_linear_equation_condition_l3402_340279

theorem linear_equation_condition (m : ℤ) : 
  (∃ a b : ℝ, ∀ x : ℝ, (m + 1 : ℝ) * x^(|m|) + 3 = a * x + b) ↔ m = 1 :=
sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l3402_340279


namespace NUMINAMATH_CALUDE_diameter_eq_hypotenuse_l3402_340232

/-- Triangle PQR with sides PQ = 15, QR = 36, and RP = 39 -/
structure RightTriangle where
  PQ : ℝ
  QR : ℝ
  RP : ℝ
  pq_eq : PQ = 15
  qr_eq : QR = 36
  rp_eq : RP = 39
  right_angle : PQ^2 + QR^2 = RP^2

/-- The diameter of the circumscribed circle of a right triangle is equal to the length of its hypotenuse -/
theorem diameter_eq_hypotenuse (t : RightTriangle) : 
  2 * (t.RP / 2) = t.RP := by sorry

end NUMINAMATH_CALUDE_diameter_eq_hypotenuse_l3402_340232


namespace NUMINAMATH_CALUDE_prob_queen_then_diamond_correct_l3402_340216

/-- The probability of drawing a Queen first and a diamond second from a standard 52-card deck, without replacement -/
def prob_queen_then_diamond : ℚ := 18 / 221

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The number of diamond cards in a standard deck -/
def num_diamonds : ℕ := 13

/-- The number of non-diamond Queens in a standard deck -/
def num_non_diamond_queens : ℕ := 3

theorem prob_queen_then_diamond_correct :
  prob_queen_then_diamond = 
    (1 / deck_size * num_diamonds / (deck_size - 1)) + 
    (num_non_diamond_queens / deck_size * num_diamonds / (deck_size - 1)) := by
  sorry

end NUMINAMATH_CALUDE_prob_queen_then_diamond_correct_l3402_340216


namespace NUMINAMATH_CALUDE_intersection_at_most_one_point_f_composition_half_l3402_340226

-- Statement B
theorem intersection_at_most_one_point (f : ℝ → ℝ) :
  ∃ (y : ℝ), ∀ (y' : ℝ), f 1 = y' → y = y' :=
sorry

-- Statement D
def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem f_composition_half : f (f (1/2)) = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_at_most_one_point_f_composition_half_l3402_340226


namespace NUMINAMATH_CALUDE_harrison_elementary_students_l3402_340215

/-- The number of students in Harrison Elementary School -/
def total_students : ℕ := 1060

/-- The fraction of students remaining at Harrison Elementary School -/
def remaining_fraction : ℚ := 3/5

/-- The number of grade levels -/
def grade_levels : ℕ := 3

/-- The number of students in each advanced class -/
def advanced_class_size : ℕ := 20

/-- The number of normal classes per grade level -/
def normal_classes_per_grade : ℕ := 6

/-- The number of students in each normal class -/
def normal_class_size : ℕ := 32

/-- Theorem stating the total number of students in Harrison Elementary School -/
theorem harrison_elementary_students :
  total_students = 
    (grade_levels * advanced_class_size + 
     grade_levels * normal_classes_per_grade * normal_class_size) / remaining_fraction :=
by sorry

end NUMINAMATH_CALUDE_harrison_elementary_students_l3402_340215


namespace NUMINAMATH_CALUDE_obtuse_angle_equation_l3402_340230

theorem obtuse_angle_equation (α : Real) : 
  α > π / 2 ∧ α < π →
  Real.sin α * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 →
  α = 140 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_obtuse_angle_equation_l3402_340230


namespace NUMINAMATH_CALUDE_min_max_f_l3402_340277

def a : ℕ := 2001

def A : Set (ℕ × ℕ) :=
  {p | let m := p.1
       let n := p.2
       m < 2 * a ∧
       (2 * n) ∣ (2 * a * m - m^2 + n^2) ∧
       n^2 - m^2 + 2 * m * n ≤ 2 * a * (n - m)}

def f (p : ℕ × ℕ) : ℚ :=
  let m := p.1
  let n := p.2
  (2 * a * m - m^2 - m * n) / n

theorem min_max_f :
  ∃ (min max : ℚ), min = 2 ∧ max = 3750 ∧
  (∀ p ∈ A, min ≤ f p ∧ f p ≤ max) ∧
  (∃ p₁ ∈ A, f p₁ = min) ∧
  (∃ p₂ ∈ A, f p₂ = max) :=
sorry

end NUMINAMATH_CALUDE_min_max_f_l3402_340277


namespace NUMINAMATH_CALUDE_number_division_l3402_340292

theorem number_division (x : ℝ) : x + 8 = 88 → x / 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l3402_340292


namespace NUMINAMATH_CALUDE_sum_of_squares_l3402_340201

theorem sum_of_squares (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) : a^2 + b^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3402_340201


namespace NUMINAMATH_CALUDE_local_minimum_condition_l3402_340221

/-- The function f(x) = x(x - m)² has a local minimum at x = 2 if and only if m = 6 -/
theorem local_minimum_condition (m : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), x * (x - m)^2 ≥ 2 * (2 - m)^2) ↔ m = 6 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_condition_l3402_340221


namespace NUMINAMATH_CALUDE_square_area_perimeter_ratio_l3402_340243

theorem square_area_perimeter_ratio : 
  ∀ (s₁ s₂ : ℝ), s₁ > 0 → s₂ > 0 → 
  (s₁^2 / s₂^2 = 16 / 49) → 
  ((4 * s₁) / (4 * s₂) = 4 / 7) := by
sorry

end NUMINAMATH_CALUDE_square_area_perimeter_ratio_l3402_340243


namespace NUMINAMATH_CALUDE_series_growth_l3402_340252

theorem series_growth (n : ℕ) (h : n > 1) :
  (Finset.range (2^(n+1) - 1)).card - (Finset.range (2^n - 1)).card = 2^n :=
sorry

end NUMINAMATH_CALUDE_series_growth_l3402_340252


namespace NUMINAMATH_CALUDE_range_of_a_l3402_340220

theorem range_of_a (x : ℝ) (a : ℝ) : 
  x ∈ Set.Ioo 0 π → 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 2 * Real.sin (x₁ + π/3) = a ∧ 2 * Real.sin (x₂ + π/3) = a) → 
  a ∈ Set.Ioo (Real.sqrt 3) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3402_340220


namespace NUMINAMATH_CALUDE_infinite_solutions_equation_l3402_340259

theorem infinite_solutions_equation :
  ∃ (x y z : ℕ → ℕ), ∀ n : ℕ,
    (x n)^2 + (x n + 1)^2 = (y n)^2 ∧
    z n = 2 * (x n) + 1 ∧
    (z n)^2 = 2 * (y n)^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_equation_l3402_340259


namespace NUMINAMATH_CALUDE_maple_trees_cut_down_l3402_340229

-- Define the initial number of maple trees
def initial_maple_trees : ℝ := 9.0

-- Define the final number of maple trees
def final_maple_trees : ℕ := 7

-- Theorem to prove the number of maple trees cut down
theorem maple_trees_cut_down :
  initial_maple_trees - final_maple_trees = 2 := by
  sorry


end NUMINAMATH_CALUDE_maple_trees_cut_down_l3402_340229


namespace NUMINAMATH_CALUDE_shortest_side_of_similar_triangle_l3402_340209

/-- Given two similar right triangles, where the first triangle has a side length of 30 inches
    and a hypotenuse of 34 inches, and the second triangle has a hypotenuse of 102 inches,
    the shortest side of the second triangle is 48 inches. -/
theorem shortest_side_of_similar_triangle (a b c : ℝ) : 
  a^2 + b^2 = 34^2 →  -- Pythagorean theorem for the first triangle
  a = 30 →            -- Given side length of the first triangle
  b ≤ a →             -- b is the shortest side of the first triangle
  c^2 + (3*b)^2 = 102^2 →  -- Pythagorean theorem for the second triangle
  3*b = 48 :=         -- The shortest side of the second triangle is 48 inches
by sorry

end NUMINAMATH_CALUDE_shortest_side_of_similar_triangle_l3402_340209


namespace NUMINAMATH_CALUDE_sales_price_calculation_l3402_340217

theorem sales_price_calculation (C S : ℝ) 
  (h1 : 1.7 * C = 51)  -- Gross profit is 170% of cost and equals $51
  (h2 : S = C + 1.7 * C)  -- Sales price is cost plus gross profit
  : S = 81 := by
  sorry

end NUMINAMATH_CALUDE_sales_price_calculation_l3402_340217


namespace NUMINAMATH_CALUDE_speed_of_sound_l3402_340235

/-- The speed of sound given specific conditions --/
theorem speed_of_sound (travel_time : Real) (blast_interval : Real) (distance : Real) :
  travel_time = 30.0 + 25.0 / 60 → -- 30 minutes and 25 seconds in hours
  blast_interval = 0.5 → -- 30 minutes in hours
  distance = 8250 → -- distance in meters
  (distance / (travel_time - blast_interval)) * (1 / 3600) = 330 := by
  sorry

end NUMINAMATH_CALUDE_speed_of_sound_l3402_340235


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3402_340208

/-- 
If 4x² + (m-3)x + 1 is a perfect square trinomial, then m = 7 or m = -1.
-/
theorem perfect_square_trinomial_condition (m : ℝ) : 
  (∃ (a b : ℝ), ∀ (x : ℝ), 4*x^2 + (m-3)*x + 1 = (a*x + b)^2) → 
  (m = 7 ∨ m = -1) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3402_340208


namespace NUMINAMATH_CALUDE_units_digit_sum_powers_l3402_340274

theorem units_digit_sum_powers : (2^20 + 3^21 + 7^20) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_powers_l3402_340274


namespace NUMINAMATH_CALUDE_triangle_angle_C_l3402_340262

theorem triangle_angle_C (a b c A B C : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  10 * a * Real.cos B = 3 * b * Real.cos A →
  Real.cos A = (5 * Real.sqrt 26) / 26 →
  C = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l3402_340262


namespace NUMINAMATH_CALUDE_ab_value_l3402_340211

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3402_340211


namespace NUMINAMATH_CALUDE_odd_reciprocal_sum_diverges_exists_rearrangement_alternating_harmonic_diverges_l3402_340268

open Set
open Function
open BigOperators
open Filter

def diverges_to_infinity (s : ℕ → ℝ) : Prop :=
  ∀ M : ℝ, M > 0 → ∃ N : ℕ, ∀ n : ℕ, n > N → s n > M

theorem odd_reciprocal_sum_diverges :
  diverges_to_infinity (λ n : ℕ => ∑ k in Finset.range n, 1 / (2 * k + 1 : ℝ)) :=
sorry

theorem exists_rearrangement_alternating_harmonic_diverges :
  ∃ f : ℕ → ℕ, Bijective f ∧
    diverges_to_infinity (λ n : ℕ => ∑ k in Finset.range n, (-1 : ℝ)^(f.invFun k - 1) / f.invFun k) :=
sorry

end NUMINAMATH_CALUDE_odd_reciprocal_sum_diverges_exists_rearrangement_alternating_harmonic_diverges_l3402_340268


namespace NUMINAMATH_CALUDE_equation_solution_l3402_340222

theorem equation_solution : ∃ x : ℝ, (5 + 3.5 * x = 2.5 * x - 25) ∧ (x = -30) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3402_340222


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3402_340224

theorem sufficient_not_necessary (x : ℝ) : 
  (x ≥ (1/2) → 2*x^2 + x - 1 ≥ 0) ∧ 
  ¬(2*x^2 + x - 1 ≥ 0 → x ≥ (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3402_340224


namespace NUMINAMATH_CALUDE_triangle_cosine_value_l3402_340205

theorem triangle_cosine_value (A B C : ℝ) (h : 7 * Real.sin B ^ 2 + 3 * Real.sin C ^ 2 = 2 * Real.sin A ^ 2 + 2 * Real.sin A * Real.sin B * Real.sin C) :
  Real.cos (A - π / 4) = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_value_l3402_340205


namespace NUMINAMATH_CALUDE_vector_equality_implies_x_coordinate_l3402_340267

/-- Given vectors a and b in ℝ², if |a + b| = |a - b|, then the x-coordinate of b is 1. -/
theorem vector_equality_implies_x_coordinate (a b : ℝ × ℝ) 
  (h : a = (-2, 1)) (h' : b.2 = 2) :
  ‖a + b‖ = ‖a - b‖ → b.1 = 1 := by
  sorry

#check vector_equality_implies_x_coordinate

end NUMINAMATH_CALUDE_vector_equality_implies_x_coordinate_l3402_340267


namespace NUMINAMATH_CALUDE_division_multiplication_order_l3402_340219

theorem division_multiplication_order : (120 / 6) / 2 * 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_order_l3402_340219


namespace NUMINAMATH_CALUDE_maxwell_twice_sister_age_sister_current_age_l3402_340227

/-- Maxwell's current age -/
def maxwell_age : ℕ := 6

/-- Maxwell's sister's current age -/
def sister_age : ℕ := 2

/-- In 2 years, Maxwell will be twice his sister's age -/
theorem maxwell_twice_sister_age : 
  maxwell_age + 2 = 2 * (sister_age + 2) := by sorry

/-- Proof that Maxwell's sister is currently 2 years old -/
theorem sister_current_age : sister_age = 2 := by sorry

end NUMINAMATH_CALUDE_maxwell_twice_sister_age_sister_current_age_l3402_340227


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3402_340236

theorem cubic_root_sum_cubes (a b c r s t : ℝ) : 
  r^3 - a*r^2 + b*r - c = 0 →
  s^3 - a*s^2 + b*s - c = 0 →
  t^3 - a*t^2 + b*t - c = 0 →
  r^3 + s^3 + t^3 = a^3 - 3*a*b + 3*c :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3402_340236


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3402_340255

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y : ℝ, x^2 / 12 + y^2 / 4 = 1) →
  (∃ c : ℝ, c > 0 ∧ c^2 = a^2 - b^2 ∧ c^2 = 12 - 4) →
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt 3 * x ∧ x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y : ℝ, x^2 / 2 - y^2 / 6 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3402_340255


namespace NUMINAMATH_CALUDE_second_smallest_sum_of_two_cubes_l3402_340253

-- Define a function to check if a number is the sum of two cubes
def isSumOfTwoCubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ a^3 + b^3 = n

-- Define a function to check if a number can be written as the sum of two cubes in two different ways
def hasTwoDifferentCubeRepresentations (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a^3 + b^3 = n ∧ c^3 + d^3 = n ∧
    (a ≠ c ∨ b ≠ d) ∧ (a ≠ d ∨ b ≠ c)

-- Define the theorem
theorem second_smallest_sum_of_two_cubes : 
  (∃ m : ℕ, m < 4104 ∧ hasTwoDifferentCubeRepresentations m) ∧
  (∀ k : ℕ, k < 4104 → k ≠ 1729 → ¬hasTwoDifferentCubeRepresentations k) ∧
  hasTwoDifferentCubeRepresentations 4104 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_sum_of_two_cubes_l3402_340253


namespace NUMINAMATH_CALUDE_square_root_of_four_l3402_340295

theorem square_root_of_four :
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l3402_340295


namespace NUMINAMATH_CALUDE_range_of_a_l3402_340283

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(5-2*a))^x > (-(5-2*a))^y

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Iic (-2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3402_340283


namespace NUMINAMATH_CALUDE_james_pays_37_50_l3402_340298

/-- Calculates the amount James pays for singing lessons -/
def james_payment (total_lessons : ℕ) (free_lessons : ℕ) (full_price_lessons : ℕ) 
  (lesson_cost : ℚ) (uncle_payment_fraction : ℚ) : ℚ :=
  let paid_lessons := total_lessons - free_lessons
  let discounted_lessons := paid_lessons - full_price_lessons
  let half_price_lessons := (discounted_lessons + 1) / 2
  let total_paid_lessons := full_price_lessons + half_price_lessons
  let total_cost := total_paid_lessons * lesson_cost
  (1 - uncle_payment_fraction) * total_cost

/-- Theorem stating that James pays $37.50 for his singing lessons -/
theorem james_pays_37_50 : 
  james_payment 20 1 10 5 (1/2) = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_james_pays_37_50_l3402_340298


namespace NUMINAMATH_CALUDE_sin_negative_1560_degrees_l3402_340290

theorem sin_negative_1560_degrees : Real.sin ((-1560 : ℝ) * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_1560_degrees_l3402_340290


namespace NUMINAMATH_CALUDE_gingerbread_problem_l3402_340204

theorem gingerbread_problem (trays_with_25 : ℕ) (additional_trays : ℕ) (total_gingerbreads : ℕ) :
  trays_with_25 = 4 →
  additional_trays = 3 →
  total_gingerbreads = 160 →
  (trays_with_25 * 25 + additional_trays * ((total_gingerbreads - trays_with_25 * 25) / additional_trays) = total_gingerbreads) →
  (total_gingerbreads - trays_with_25 * 25) / additional_trays = 20 := by
sorry

end NUMINAMATH_CALUDE_gingerbread_problem_l3402_340204


namespace NUMINAMATH_CALUDE_sheepdog_speed_l3402_340234

/-- Proves that a sheepdog running at the specified speed can catch a sheep in the given time --/
theorem sheepdog_speed (sheep_speed : ℝ) (initial_distance : ℝ) (catch_time : ℝ) 
  (h1 : sheep_speed = 12)
  (h2 : initial_distance = 160)
  (h3 : catch_time = 20) :
  (initial_distance + sheep_speed * catch_time) / catch_time = 20 := by
  sorry

#check sheepdog_speed

end NUMINAMATH_CALUDE_sheepdog_speed_l3402_340234


namespace NUMINAMATH_CALUDE_consecutive_natural_numbers_sum_l3402_340270

theorem consecutive_natural_numbers_sum (a : ℕ) : 
  (∃ (x : ℕ), x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 50) → 
  (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = 50) → 
  (a + 2 = 10) := by
  sorry

#check consecutive_natural_numbers_sum

end NUMINAMATH_CALUDE_consecutive_natural_numbers_sum_l3402_340270


namespace NUMINAMATH_CALUDE_solve_salary_problem_l3402_340257

def salary_problem (a b : ℝ) : Prop :=
  a + b = 4000 ∧
  0.05 * a = 0.15 * b ∧
  a = 3000

theorem solve_salary_problem :
  ∃ (a b : ℝ), salary_problem a b :=
sorry

end NUMINAMATH_CALUDE_solve_salary_problem_l3402_340257


namespace NUMINAMATH_CALUDE_speed_of_boat_in_still_water_l3402_340249

/-- Theorem: Speed of boat in still water
Given:
- The rate of the current is 15 km/hr
- The boat traveled downstream for 25 minutes
- The boat covered a distance of 33.33 km downstream

Prove that the speed of the boat in still water is approximately 64.992 km/hr
-/
theorem speed_of_boat_in_still_water
  (current_speed : ℝ)
  (travel_time : ℝ)
  (distance_covered : ℝ)
  (h1 : current_speed = 15)
  (h2 : travel_time = 25 / 60)
  (h3 : distance_covered = 33.33) :
  ∃ (boat_speed : ℝ), abs (boat_speed - 64.992) < 0.001 ∧
    distance_covered = (boat_speed + current_speed) * travel_time :=
by sorry

end NUMINAMATH_CALUDE_speed_of_boat_in_still_water_l3402_340249


namespace NUMINAMATH_CALUDE_trapezoid_area_l3402_340258

/-- A trapezoid ABCD with the following properties:
  * BC = 5
  * Distance from A to line BC is 3
  * Distance from D to line BC is 7
-/
structure Trapezoid where
  BC : ℝ
  dist_A_to_BC : ℝ
  dist_D_to_BC : ℝ
  h_BC : BC = 5
  h_dist_A : dist_A_to_BC = 3
  h_dist_D : dist_D_to_BC = 7

/-- The area of the trapezoid ABCD -/
def area (t : Trapezoid) : ℝ := 25

/-- Theorem stating that the area of the trapezoid ABCD is 25 -/
theorem trapezoid_area (t : Trapezoid) : area t = 25 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3402_340258


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3402_340278

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 10*x ≥ -25) ∧ (∃ x : ℝ, x^2 + 10*x = -25) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3402_340278


namespace NUMINAMATH_CALUDE_nigels_winnings_l3402_340287

/-- The amount of money Nigel won initially -/
def initial_winnings : ℝ := sorry

/-- The amount Nigel gave away -/
def amount_given_away : ℝ := 25

/-- The amount Nigel's mother gave him -/
def amount_from_mother : ℝ := 80

/-- The extra amount Nigel has compared to twice his initial winnings -/
def extra_amount : ℝ := 10

theorem nigels_winnings :
  initial_winnings - amount_given_away + amount_from_mother = 
  2 * initial_winnings + extra_amount ∧ initial_winnings = 45 := by
  sorry

end NUMINAMATH_CALUDE_nigels_winnings_l3402_340287

import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_example_l253_25331

/-- Calculates the profit percentage given the cost price and selling price. -/
noncomputable def profit_percentage (cost_price selling_price : ℝ) : ℝ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem: The profit percentage for an article bought at Rs. 500 and sold at Rs. 550 is 10%. -/
theorem profit_percentage_example : profit_percentage 500 550 = 10 := by
  -- Unfold the definition of profit_percentage
  unfold profit_percentage
  -- Simplify the arithmetic expression
  simp [div_mul_eq_mul_div]
  -- Perform the calculation
  norm_num
  -- QED


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_example_l253_25331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l253_25304

/-- Given a distance in kilometers and a time in seconds, calculate the speed in meters per second -/
noncomputable def calculate_speed (distance_km : ℝ) (time_s : ℝ) : ℝ :=
  (distance_km * 1000) / time_s

/-- Theorem stating that covering 17.138 km in 38 seconds results in a speed of 451 m/s -/
theorem speed_calculation :
  let distance_km : ℝ := 17.138
  let time_s : ℝ := 38
  Int.floor (calculate_speed distance_km time_s) = 451 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l253_25304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_is_13_l253_25382

/-- Represents the age difference between two ages -/
def age_difference (a b : ℕ) : ℕ := 
  if a ≥ b then a - b else 0

/-- Represents the age of a person -/
structure Person where
  age : ℕ

/-- Represents the family with four members -/
structure Family where
  raj : Person
  ravi : Person
  hema : Person
  rahul : Person

/-- The conditions given in the problem -/
def family_conditions (f : Family) : Prop :=
  f.raj.age > f.ravi.age ∧
  f.hema.age = f.ravi.age - 2 ∧
  f.raj.age = 3 * f.rahul.age ∧
  3 * f.rahul.age = 2 * f.hema.age ∧
  20 = f.hema.age + (f.hema.age / 3)

theorem age_difference_is_13 (f : Family) (h : family_conditions f) : 
  age_difference f.raj.age f.ravi.age = 13 := by
  sorry

#check age_difference_is_13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_is_13_l253_25382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_ways_count_l253_25330

-- Define the set of allowed digits
def allowed_digits : Finset Nat := {0, 2, 4, 5, 6, 7}

-- Define the structure of the number
def number_structure : List Nat := [2, 0, 1, 6, 0, 7]

-- Define a function to check if a number is divisible by 75
def is_divisible_by_75 (n : Nat) : Bool := n % 75 = 0

-- Define a function to generate all possible 11-digit numbers
def generate_numbers : List Nat :=
  sorry

-- Define the count of valid numbers
def valid_count : Nat :=
  (generate_numbers.filter is_divisible_by_75).length

-- Theorem statement
theorem valid_ways_count :
  valid_count = 432 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_ways_count_l253_25330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l253_25369

open Nat

/-- Sum of digits in the decimal representation of n -/
def S (n : ℕ) : ℕ := (digits 10 n).sum

/-- Sum of all stumps of n -/
def T (n : ℕ) : ℕ := sorry

/-- Main theorem: n = S(n) + 9*T(n) -/
theorem main_theorem (n : ℕ) : n = S n + 9 * T n := by
  sorry

#check main_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l253_25369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_package_contains_twenty_car_washes_l253_25318

-- Define the given conditions
noncomputable def normal_price : ℚ := 15
noncomputable def package_price_ratio : ℚ := 60 / 100
noncomputable def total_paid : ℚ := 180

-- Define the number of car washes as a function of the given conditions
noncomputable def num_car_washes (normal_price package_price_ratio total_paid : ℚ) : ℚ :=
  total_paid / (normal_price * package_price_ratio)

-- Theorem statement
theorem package_contains_twenty_car_washes :
  num_car_washes normal_price package_price_ratio total_paid = 20 := by
  -- Unfold the definition of num_car_washes
  unfold num_car_washes
  -- Simplify the expression
  simp [normal_price, package_price_ratio, total_paid]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_package_contains_twenty_car_washes_l253_25318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l253_25313

noncomputable def f (x : ℝ) : ℝ := x + 2 / x

theorem f_properties :
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f x) ∧
  (∀ x₁ x₂ : ℝ, Real.sqrt 2 ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l253_25313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_num_and_den_l253_25375

/-- The repeating decimal 5.1717171717... -/
def x : ℚ := 5 + 17 / 99

/-- The sum of the numerator and denominator of x when expressed as a fraction in lowest terms -/
def sum_num_den : ℕ := (x.num.natAbs + x.den)

theorem sum_of_num_and_den : sum_num_den = 611 := by
  -- Proof steps would go here
  sorry

#eval sum_num_den

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_num_and_den_l253_25375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_amount_during_test_l253_25308

/-- The amount of liquid (in liters) that triggers the need to use the bathroom -/
def bathroom_threshold : ℝ := 1.2

/-- Conversion factor from ounces to milliliters -/
def oz_to_ml : ℝ := 29.57

/-- Conversion factor from liters to milliliters -/
def l_to_ml : ℝ := 1000

/-- Amount of milk consumed in milliliters -/
def milk_consumed : ℝ := 250

/-- Amount of orange juice consumed in ounces -/
def orange_juice_consumed : ℝ := 5

/-- Amount of grape juice consumed in ounces -/
def grape_juice_consumed : ℝ := 10

/-- Amount of soda consumed in liters -/
def soda_consumed : ℝ := 0.1

/-- Theorem stating the amount of water Jamie can drink during the test -/
theorem water_amount_during_test :
  ∃ (water_oz : ℝ),
    (water_oz ≥ 13.73 ∧ water_oz ≤ 13.75) ∧
    (milk_consumed + orange_juice_consumed * oz_to_ml + grape_juice_consumed * oz_to_ml + soda_consumed * l_to_ml + water_oz * oz_to_ml) / l_to_ml ≤ bathroom_threshold :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_amount_during_test_l253_25308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_uphill_speed_l253_25340

/-- Calculates the uphill speed given the conditions of the motorcycle journey -/
noncomputable def uphill_speed (downhill_speed : ℝ) (total_time : ℝ) (total_distance : ℝ) : ℝ :=
  let one_way_distance := total_distance / 2
  let downhill_time := one_way_distance / downhill_speed
  let uphill_time := total_time - downhill_time
  one_way_distance / uphill_time

/-- Theorem stating that under the given conditions, the uphill speed is 50 kmph -/
theorem motorcycle_uphill_speed :
  uphill_speed 100 12 800 = 50 := by
  -- Unfold the definition of uphill_speed
  unfold uphill_speed
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_uphill_speed_l253_25340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_speed_l253_25306

/-- Calculates the average speed for a round trip given the distance and speeds. -/
noncomputable def average_speed_round_trip (distance : ℝ) (speed_ab speed_ba : ℝ) : ℝ :=
  (2 * distance) / (distance / speed_ab + distance / speed_ba)

/-- Theorem: The average speed for the given round trip is 19.2 km/h. -/
theorem round_trip_average_speed :
  let distance := (48 : ℝ)
  let speed_ab := (16 : ℝ)
  let speed_ba := (24 : ℝ)
  average_speed_round_trip distance speed_ab speed_ba = 19.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_speed_l253_25306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_proof_l253_25345

/-- An ellipse with equation x^2/a^2 + y^2/b^2 = 1 --/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of an ellipse --/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point in ℝ² --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on an ellipse --/
def Ellipse.contains (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

theorem ellipse_equation_proof (e1 e2 : Ellipse) (p : Point) :
  e1.a^2 = 4 ∧ e1.b^2 = 1 ∧
  e2.a^2 = 6 ∧ e2.b^2 = 3 ∧
  p.x = 2 ∧ p.y = 1 →
  e1.eccentricity = e2.eccentricity ∧
  e2.contains p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_proof_l253_25345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_of_M_l253_25374

/-- An angle α in a rectangular coordinate system with its vertex at the origin and initial side along the positive x-axis -/
structure Angle (α : Real) where
  vertex_origin : True
  initial_side_x_axis : True

/-- A point M on the terminal side of angle α -/
structure TerminalPoint (α : Real) where
  x : Real
  y : Real
  on_terminal_side : True

/-- The theorem stating the x-coordinate of point M given sin α = 1/3 -/
theorem x_coordinate_of_M (α : Real) (M : TerminalPoint α) 
  (h_sin : Real.sin α = 1/3) (h_y : M.y = 1) :
  M.x = 2 * Real.sqrt 2 ∨ M.x = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_of_M_l253_25374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_in_sets_l253_25321

/-- Checks if three numbers can form a right-angled triangle -/
noncomputable def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The sets of numbers to check -/
noncomputable def number_sets : List (ℝ × ℝ × ℝ) :=
  [(1, 1, 2), (2, Real.sqrt 7, Real.sqrt 3), (4, 6, 8), (5, 12, 11)]

theorem right_triangle_in_sets :
  ∃! (a b c : ℝ), (a, b, c) ∈ number_sets ∧ is_right_triangle a b c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_in_sets_l253_25321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_type_l253_25319

theorem triangle_type (A B C : ℝ) (h : A + B + C = 180) 
  (h2 : ∃ (x : ℝ), A = 2*x ∧ B = 3*x ∧ C = 5*x) :
  C = 90 :=
by
  rcases h2 with ⟨x, hA, hB, hC⟩
  have h3 : 10 * x = 180 := by
    rw [← h, hA, hB, hC]
    ring
  have h4 : x = 18 := by
    linarith
  rw [hC, h4]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_type_l253_25319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l253_25310

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x) ^ 2

theorem f_properties :
  (∀ x : ℝ, f x ≤ 3) ∧
  (∀ k : ℤ, f (π / 6 + k * π) = 3) ∧
  (∀ k : ℤ, ∀ x : ℝ, -π / 3 + k * π ≤ x ∧ x ≤ π / 6 + k * π →
    ∀ y : ℝ, -π / 3 + k * π ≤ y ∧ y ≤ x → f y ≤ f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l253_25310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l253_25362

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the properties of the specific triangle ABC -/
theorem triangle_abc_properties (t : Triangle) 
  (h1 : (2 * t.a - t.b) * Real.cos t.C - t.c * Real.cos t.B = 0)
  (h2 : t.a + t.b = 13)
  (h3 : t.c = 7) :
  t.C = π / 3 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = 10 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l253_25362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_theorem_l253_25315

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℚ
  breadth : ℚ
  total_fencing_cost : ℚ

/-- Calculates the perimeter of a rectangular plot. -/
def perimeter (plot : RectangularPlot) : ℚ :=
  2 * (plot.length + plot.breadth)

/-- Calculates the cost of fencing per meter for a rectangular plot. -/
def fencing_cost_per_meter (plot : RectangularPlot) : ℚ :=
  plot.total_fencing_cost / perimeter plot

/-- Theorem stating that for a rectangular plot with given dimensions and total fencing cost,
    the fencing cost per meter is 26.5. -/
theorem fencing_cost_theorem (plot : RectangularPlot) 
    (h1 : plot.length = 65)
    (h2 : plot.breadth = 35)
    (h3 : plot.total_fencing_cost = 5300) :
    fencing_cost_per_meter plot = 265 / 10 := by
  sorry

/-- Compute the fencing cost per meter for the given plot dimensions. -/
def main : IO Unit := do
  let plot : RectangularPlot := { length := 65, breadth := 35, total_fencing_cost := 5300 }
  IO.println s!"Fencing cost per meter: {fencing_cost_per_meter plot}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_theorem_l253_25315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_formula_l253_25361

/-- The volume of a cone with given parameters. -/
noncomputable def cone_volume (h α β : ℝ) : ℝ :=
  (Real.pi * h^3 * (Real.cos β^2 + Real.tan (α/2)^2)) / (3 * Real.sin β^2)

/-- Theorem stating the volume of a cone with specific geometric properties. -/
theorem cone_volume_formula (h α β : ℝ) (h_pos : h > 0) (α_pos : α > 0) (β_pos : β > 0) (β_lt_pi_2 : β < Real.pi/2) :
  ∃ V : ℝ, V = cone_volume h α β ∧ V > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_formula_l253_25361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l253_25397

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := 1 - (Real.sin x) / (x^4 + 2*x^2 + 1)

-- State the theorem
theorem sum_of_max_min_f : 
  ∃ (max min : ℝ), (∀ x : ℝ, f x ≤ max ∧ min ≤ f x) ∧ max + min = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l253_25397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l253_25372

noncomputable def f (x : ℝ) := 6 * (Real.cos x)^2 - Real.sin (2 * x)

theorem f_properties :
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ M = 6) ∧
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    (∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f (y + q) ≠ f y) ∧ 
    p = Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l253_25372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_excluding_stoppages_l253_25368

/-- The speed of a bus excluding stoppages, given its speed including stoppages and stoppage time. -/
theorem bus_speed_excluding_stoppages 
  (speed_including_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (h1 : speed_including_stoppages = 45) 
  (h2 : stoppage_time = 15) : 
  speed_including_stoppages * (60 / (60 - stoppage_time)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_excluding_stoppages_l253_25368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_velocity_is_three_l253_25334

/-- The displacement function of an object in linear motion -/
def displacement (t : ℝ) : ℝ := 3 * t - t^2

/-- The velocity function of the object -/
noncomputable def velocity (t : ℝ) : ℝ := deriv displacement t

/-- The initial velocity of the object -/
noncomputable def initialVelocity : ℝ := velocity 0

theorem initial_velocity_is_three :
  initialVelocity = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_velocity_is_three_l253_25334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l253_25386

def ellipse_equation (x y : ℝ) : Prop := x^2 / 4 + y^2 / 9 = 1

noncomputable def major_axis_length : ℝ :=
  2 * Real.sqrt (max 9 4)

theorem ellipse_major_axis_length :
  major_axis_length = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l253_25386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l253_25338

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt 3 * Real.sin (2 * x) + 2) * 1 + Real.cos x * (2 * Real.cos x)

theorem triangle_side_length (A B C : ℝ) (h1 : f A = 4) (h2 : 0 < A ∧ A < Real.pi)
  (h3 : (1 : ℝ) * C * Real.sin A / 2 = Real.sqrt 3 / 2) :
  ∃ a : ℝ, a^2 = 3 ∧ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l253_25338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l253_25343

noncomputable def intersection_point (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ × ℝ :=
  ((b₁ * c₂ - b₂ * c₁) / (a₁ * b₂ - a₂ * b₁), (a₂ * c₁ - a₁ * c₂) / (a₁ * b₂ - a₂ * b₁))

noncomputable def line_slope (a b : ℝ) : ℝ := -a / b

noncomputable def perpendicular_slope (m : ℝ) : ℝ := -1 / m

def line_equation (x₀ y₀ m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ y - y₀ = m * (x - x₀)

theorem line_equation_proof (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  let p := intersection_point a₁ b₁ c₁ a₂ b₂ c₂
  let m := perpendicular_slope (line_slope 2 1)
  (∀ x y, line_equation p.1 p.2 m x y ↔ x + 2*y - 11 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l253_25343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_permutations_l253_25392

/-- Represents the number of times a player plays on a lane -/
def player_plays (p : ℕ) (l : ℕ) (times : ℕ) : Prop :=
  sorry

/-- Calculates the number of permutations for the given setup -/
def number_of_permutations (n : ℕ) (m : ℕ) : ℕ :=
  sorry

/-- The number of possible player permutations for a bowling game setup -/
theorem bowling_permutations (n m : ℕ) : n = 8 → m = 4 → 
  (∀ p, p ≤ n → ∀ l, l ≤ m → (l = m → player_plays p l 1) ∧ (l < m → player_plays p l 2)) →
  number_of_permutations n m = 120960 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_permutations_l253_25392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_value_from_arguments_l253_25325

theorem complex_value_from_arguments (z : ℂ) 
  (h1 : Complex.arg (z^2 - 4) = 5*π/6) 
  (h2 : Complex.arg (z^2 + 4) = π/3) : 
  z = 1 + Complex.I * Real.sqrt 3 ∨ z = -1 - Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_value_from_arguments_l253_25325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_activation_function_properties_l253_25344

noncomputable def f (x : ℝ) : ℝ := 2 / (1 + Real.exp (-2 * x)) - 1

theorem activation_function_properties :
  -- The function is odd
  (∀ x, f (-x) = -f x) ∧
  -- The function is increasing
  (∀ x y, x < y → f x < f y) ∧
  -- No tangent line perpendicular to x + √2y = 0
  (∀ x, ¬(∃ y, f y = x ∧ ((deriv f) y) * ((deriv f) y) = 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_activation_function_properties_l253_25344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_b_sigma_b_l253_25327

/-- Sum of positive divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Theorem: Given N = 2^r * b where r and b are positive integers, b is odd, 
    and σ(N) = 2N - 1, then b and σ(b) are coprime -/
theorem coprime_b_sigma_b 
  (r b : ℕ) 
  (hr : r > 0)
  (hb : b > 0)
  (h_b_odd : Odd b)
  (h_sigma : sigma (2^r * b) = 2 * (2^r * b) - 1) : 
  Nat.Coprime b (sigma b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_b_sigma_b_l253_25327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_pyramid_volumes_theorem_l253_25373

/-- A rectangular parallelepiped with specific properties -/
structure RectParallelepiped where
  V : ℝ  -- Volume
  R : ℝ  -- Radius of circumscribed sphere
  m : ℝ  -- Sum of lengths of edges AA₁, AB, and AD
  volume_positive : 0 < V
  radius_positive : 0 < R
  edge_sum_positive : 0 < m

/-- The sum of volumes of three specific pyramids in the parallelepiped -/
noncomputable def sum_pyramid_volumes (p : RectParallelepiped) : ℝ :=
  (2 * p.V / 3) - (4 * p.V^2 * p.R^2 * (p.m^2 - 4*p.R^2 - 2*p.V*p.m + 8*p.V^2)) / (p.m^2 - 4*p.R^2 - 2*p.V*p.m)

/-- Theorem stating the sum of volumes of three specific pyramids in a rectangular parallelepiped -/
theorem sum_pyramid_volumes_theorem (p : RectParallelepiped) :
  ∃ (V₁ V₂ V₃ : ℝ), V₁ + V₂ + V₃ = sum_pyramid_volumes p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_pyramid_volumes_theorem_l253_25373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_calculation_l253_25329

/-- The volume of a cone in cubic inches. -/
noncomputable def cone_volume : ℝ := 15552 * Real.pi

/-- The vertex angle of the vertical cross section in degrees. -/
def vertex_angle : ℝ := 45

/-- The height of the cone in inches. -/
def cone_height : ℝ := 20.0

/-- Theorem stating that a cone with the given volume and vertex angle has the specified height. -/
theorem cone_height_calculation (ε : ℝ) (h_ε : ε > 0) :
  ∃ (h : ℝ), abs (h - cone_height) < ε ∧
  cone_volume = (1/3) * Real.pi * (h / Real.tan (vertex_angle / 2 * Real.pi / 180))^2 * h :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_calculation_l253_25329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_through_point_with_same_foci_l253_25335

/-- The standard equation of an ellipse -/
def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  y^2 / a^2 + x^2 / b^2 = 1

/-- The foci of an ellipse given its semi-major and semi-minor axes -/
noncomputable def ellipse_foci (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

theorem ellipse_through_point_with_same_foci :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    ellipse_equation a b (Real.sqrt 3) (-Real.sqrt 5) ∧
    ellipse_foci a b = ellipse_foci 5 3 ∧
    a^2 = 20 ∧ b^2 = 4 :=
by sorry

#check ellipse_through_point_with_same_foci

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_through_point_with_same_foci_l253_25335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_property_l253_25376

/-- Function to get the units digit of a number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Function to get the sum of non-units digits of a number -/
def sumOfOtherDigits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10)

/-- Predicate to check if a number satisfies the property -/
def satisfiesProperty (n : ℕ) : Bool :=
  unitsDigit n = sumOfOtherDigits n

/-- The main theorem to be proved -/
theorem count_numbers_with_property :
  (Finset.filter (fun n => satisfiesProperty n) (Finset.range 3023 \ Finset.range 1000)).card = 109 := by
  sorry

#eval (Finset.filter (fun n => satisfiesProperty n) (Finset.range 3023 \ Finset.range 1000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_property_l253_25376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_range_l253_25324

noncomputable section

-- Define the line l
def line_l (α t : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 1/2 + t * Real.sin α)

-- Define the curve C in polar coordinates
def curve_C_polar (θ : ℝ) : ℝ := Real.sqrt (12 / (4 * Real.sin θ^2 + 3 * Real.cos θ^2))

-- Define the curve C in Cartesian coordinates
def curve_C_cartesian (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define point P
def point_P : ℝ × ℝ := (1, 1/2)

-- Theorem statement
theorem intersection_product_range :
  ∀ α : ℝ,
  ∃ A B : ℝ × ℝ,
  (∃ t₁ t₂ : ℝ, line_l α t₁ = A ∧ line_l α t₂ = B) ∧
  curve_C_cartesian A.1 A.2 ∧
  curve_C_cartesian B.1 B.2 ∧
  A ≠ B ∧
  2 ≤ ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2).sqrt * ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2).sqrt ∧
  ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2).sqrt * ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2).sqrt ≤ 8/3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_range_l253_25324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_composite_l253_25328

/-- A sequence of natural numbers where each subsequent number is obtained by adding a proper divisor to the previous number -/
def ValidSequence (seq : List ℕ) : Prop :=
  seq.head? = some 4 ∧
  ∀ i, i + 1 < seq.length →
    ∃ d, d ∣ seq[i]! ∧ d ≠ 1 ∧ d ≠ seq[i]! ∧ seq[i + 1]! = seq[i]! + d

/-- A natural number is composite if it has a proper divisor greater than 1 -/
def IsComposite (n : ℕ) : Prop :=
  ∃ d, d ∣ n ∧ 1 < d ∧ d < n

theorem reach_composite (n : ℕ) (h : IsComposite n) :
  ∃ seq : List ℕ, ValidSequence seq ∧ seq.getLast? = some n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_composite_l253_25328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l253_25381

/-- Represents the time taken to fill a cistern when two pipes are opened simultaneously -/
noncomputable def fillTime (fillRate emptyRate : ℝ) : ℝ :=
  1 / (fillRate - emptyRate)

/-- Theorem stating the time taken to fill the cistern under given conditions -/
theorem cistern_fill_time :
  let fillRate := (1 : ℝ) / 20  -- Rate at which pipe A fills the cistern
  let emptyRate := (1 : ℝ) / 25 -- Rate at which pipe B empties the cistern
  fillTime fillRate emptyRate = 100 := by
  unfold fillTime
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l253_25381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_vector_dot_product_l253_25398

/-- Predicate to check if six points form a regular hexagon -/
def IsRegularHexagon (A B C D E F : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if the side length of a hexagon is 1 -/
def HexagonSideLength (A B C D E F : ℝ × ℝ) (length : ℝ) : Prop := sorry

/-- Given a regular hexagon ABCDEF with side length 1, 
    prove that the dot product of (AB + DC) and (AD + BE) equals -3 -/
theorem hexagon_vector_dot_product (A B C D E F : ℝ × ℝ) : 
  IsRegularHexagon A B C D E F → 
  HexagonSideLength A B C D E F 1 →
  (B - A + C - D) • (D - A + E - B) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_vector_dot_product_l253_25398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l253_25383

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

-- State the theorem
theorem f_properties :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), ∀ y ∈ Set.Icc 0 (Real.pi / 2), x < y → f x < f y) ∧
  (∀ x : ℝ, f ((-Real.pi / 4) - x) = f ((-Real.pi / 4) + x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l253_25383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_equals_36_l253_25307

noncomputable section

-- Define the side lengths of the triangle
def triangle_side1 : ℝ := 5.5
def triangle_side2 : ℝ := 7.5
def triangle_side3 : ℝ := 11

-- Define the perimeter of the triangle
def triangle_perimeter : ℝ := triangle_side1 + triangle_side2 + triangle_side3

-- Define the side length of the square
def square_side : ℝ := triangle_perimeter / 4

-- Theorem statement
theorem square_area_equals_36 : square_side^2 = 36 := by
  -- Expand the definition of square_side
  unfold square_side
  -- Expand the definition of triangle_perimeter
  unfold triangle_perimeter
  -- Simplify the expression
  simp [triangle_side1, triangle_side2, triangle_side3]
  -- The proof is complete
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_equals_36_l253_25307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l253_25323

noncomputable def f (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) * (2 * x^2 + 1)

theorem f_inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 3 4, f (a * x + 1) ≤ f (x - 2)) →
  -2/3 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l253_25323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_integral_bounded_l253_25349

/-- A function with an absolutely convergent Fourier series satisfying certain conditions -/
structure FourierFunction where
  f : ℝ → ℝ
  a : ℕ → ℝ
  b : ℕ → ℝ
  fourier_series : ∀ x, f x = (a 0) / 2 + ∑' k, a k * Real.cos (k * x) + b k * Real.sin (k * x)
  abs_convergent : Summable (fun k ↦ |a k| + |b k|)
  coeff_condition : ∀ k : ℕ, a k ^ 2 + b k ^ 2 ≥ a (k + 1) ^ 2 + b (k + 1) ^ 2

/-- The main theorem stating the uniform boundedness of the integral expression -/
theorem fourier_integral_bounded (ff : FourierFunction) :
  ∃ C : ℝ, C > 0 ∧ ∀ h : ℝ, h > 0 →
    (1 / h) * ∫ x in (0 : ℝ)..(2 * Real.pi), (ff.f (x + h) - ff.f (x - h))^2 ≤ C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_integral_bounded_l253_25349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_iff_a_in_range_l253_25326

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2) * x + 2

theorem monotonic_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (4 ≤ a ∧ a < 8) :=
by sorry

#check monotonic_increasing_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_iff_a_in_range_l253_25326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l253_25389

/-- Calculates the interest rate for a loan given the monthly payment, number of months, and total amount owed with interest. -/
noncomputable def calculate_interest_rate (monthly_payment : ℝ) (num_months : ℕ) (total_with_interest : ℝ) : ℝ :=
  let principal := monthly_payment * (num_months : ℝ)
  let interest_amount := total_with_interest - principal
  (interest_amount / principal) * 100

/-- Theorem stating that for a loan with $100 monthly payments for 12 months and a total owed of $1320, the interest rate is 10%. -/
theorem interest_rate_is_ten_percent :
  calculate_interest_rate 100 12 1320 = 10 := by
  unfold calculate_interest_rate
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l253_25389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_satisfies_conditions_mySequence_is_unique_l253_25370

noncomputable def mySequence (n : ℕ) : ℚ :=
  1 / n.factorial + 1

theorem mySequence_satisfies_conditions :
  mySequence 1 = 2 ∧
  ∀ n : ℕ, n ≥ 1 → (n + 1 : ℚ) * mySequence (n + 1) = mySequence n + n :=
by sorry

theorem mySequence_is_unique :
  ∀ a : ℕ → ℚ,
  (a 1 = 2 ∧ ∀ n : ℕ, n ≥ 1 → (n + 1 : ℚ) * a (n + 1) = a n + n) →
  ∀ n : ℕ, a n = mySequence n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_satisfies_conditions_mySequence_is_unique_l253_25370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_collinearity_l253_25322

-- Define the ellipse C
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle x^2 + y^2 = b^2
def circle_eq (x y b : ℝ) : Prop :=
  x^2 + y^2 = b^2

-- Define collinearity of three points
def collinear (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Main theorem
theorem ellipse_tangent_collinearity 
  (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hf : a^2 - b^2 = 2) -- Condition for right focus at (√2, 0)
  (he : (a^2 - b^2) / a^2 = 2/3) -- Condition for eccentricity √6/3
  (x1 y1 x2 y2 : ℝ)
  (hm : ellipse x1 y1 a b) (hn : ellipse x2 y2 a b)
  (ht : ∃ (xt yt : ℝ), circle_eq xt yt b ∧ 
        ((y2 - y1) * (xt - x1) = (yt - y1) * (x2 - x1)) ∧ 
        xt > 0) :
  (a^2 = 3 ∧ b^2 = 1) ∧
  (collinear x1 y1 x2 y2 (Real.sqrt 2) 0 ↔ distance x1 y1 x2 y2 = Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_collinearity_l253_25322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_imply_x_is_sqrt_2_l253_25396

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

/-- The first vector -/
noncomputable def a : ℝ × ℝ := (1, Real.sqrt (1 + Real.sin (40 * Real.pi / 180)))

/-- The second vector -/
noncomputable def b (x : ℝ) : ℝ × ℝ := (1 / Real.sin (65 * Real.pi / 180), x)

theorem collinear_vectors_imply_x_is_sqrt_2 :
  collinear a (b x) → x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_imply_x_is_sqrt_2_l253_25396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_is_6_5_l253_25357

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  totalCost : ℝ
  lengthWidthRelation : length = width + 10
  perimeterFormula : perimeter = 2 * (length + width)
  perimeterValue : perimeter = 140
  totalCostValue : totalCost = 910

/-- Calculates the rate of fencing per meter for a given rectangular plot -/
noncomputable def fencingRate (plot : RectangularPlot) : ℝ :=
  plot.totalCost / plot.perimeter

/-- Theorem stating that the fencing rate for the given plot is 6.5 -/
theorem fencing_rate_is_6_5 (plot : RectangularPlot) : fencingRate plot = 6.5 := by
  sorry

#check fencing_rate_is_6_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_is_6_5_l253_25357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_munificence_monic_quadratic_l253_25348

/-- Definition of munificence for a polynomial on [-2, 2] -/
noncomputable def munificence (p : ℝ → ℝ) : ℝ :=
  ⨆ (x : ℝ) (h : x ∈ Set.Icc (-2) 2), |p x|

/-- A monic quadratic polynomial -/
def monic_quadratic (b c : ℝ) (x : ℝ) : ℝ :=
  x^2 + b*x + c

/-- The smallest possible munificence for a monic quadratic polynomial is 2 -/
theorem min_munificence_monic_quadratic :
  ∃ (b c : ℝ), munificence (monic_quadratic b c) = 2 ∧
  ∀ (b' c' : ℝ), munificence (monic_quadratic b' c') ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_munificence_monic_quadratic_l253_25348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_sum_of_digits_binomial_l253_25384

def binomial (n k : ℕ) : ℕ := n.choose k

def sum_of_digits (n : ℕ) : ℕ := sorry

def num_digits (n : ℕ) : ℕ := sorry

-- Define an approximation relation
def approx (a b : ℕ) : Prop := sorry

-- Use notation for approximation
notation:50 a " ≈ " b => approx a b

theorem estimate_sum_of_digits_binomial :
  ∃ (N : ℕ), 
    (sum_of_digits (binomial 1000 100) ≈ N) ∧ 
    (N ≈ 621) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_sum_of_digits_binomial_l253_25384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_male_height_is_182_l253_25388

/-- Calculates the average male height given the overall average height,
    average female height, and the ratio of men to women. -/
noncomputable def average_male_height (overall_avg : ℝ) (female_avg : ℝ) (ratio : ℝ) : ℝ :=
  (overall_avg * (ratio + 1) - female_avg) / ratio

/-- Theorem stating that given the specific conditions,
    the average male height is 182 cm. -/
theorem average_male_height_is_182
  (overall_avg : ℝ)
  (female_avg : ℝ)
  (ratio : ℝ)
  (h1 : overall_avg = 180)
  (h2 : female_avg = 170)
  (h3 : ratio = 5) :
  average_male_height overall_avg female_avg ratio = 182 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and can cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_male_height_is_182_l253_25388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l253_25353

open Real

-- Define the function f(x) = (e^x + 3) / x
noncomputable def f (x : ℝ) : ℝ := (exp x + 3) / x

-- State the theorem
theorem max_a_value (a : ℤ) :
  (∀ x > 0, f x ≥ exp (↑a)) →
  a ≤ 1 ∧ (∃ x > 0, f x ≥ exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l253_25353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_and_evaluation_l253_25312

-- Define the original expression
noncomputable def f (x : ℝ) : ℝ := (((3 * x + 2) / (x + 1) - 2) * ((x^2 - 1) / x))

-- Define the specific x value
noncomputable def specific_x : ℝ := Real.sqrt 16 - (1 / 4)⁻¹ - (Real.pi - 3)^0

-- Theorem statement
theorem expression_simplification_and_evaluation :
  (∀ x : ℝ, x ≠ -1 ∧ x ≠ 0 → f x = x - 1) ∧
  f specific_x = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_and_evaluation_l253_25312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_progression_sum_l253_25305

/-- Sum of a geometric progression with n terms, first term a, and common ratio r -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- Sum of the reciprocal progression -/
noncomputable def reciprocal_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  (1/a) * (1 - (1/r)^n) / (1 - 1/r)

theorem reciprocal_progression_sum 
  (n : ℕ) (r : ℝ) (s' : ℝ) 
  (h1 : r ≠ 0) 
  (h2 : r ≠ 1/2) 
  (h3 : s' = geometric_sum 2 (2*r) n) :
  reciprocal_sum 2 (2*r) n = s' / (2*r)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_progression_sum_l253_25305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_converges_to_sqrt_two_l253_25356

noncomputable def sequence_x (x₁ : ℝ) : ℕ → ℝ
  | 0 => x₁
  | 1 => x₁
  | n + 2 => 1 + sequence_x x₁ (n + 1) - (1/2) * (sequence_x x₁ (n + 1))^2

theorem x_converges_to_sqrt_two (x₁ : ℝ) (h₁ : 1 < x₁) (h₂ : x₁ < 2) :
  ∀ n ≥ 3, |sequence_x x₁ n - Real.sqrt 2| < (1/2)^n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_converges_to_sqrt_two_l253_25356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_match_theorem_l253_25360

/-- A graph representing a tennis club's match arrangement -/
structure TennisGraph where
  V : Finset ℕ
  E : Finset (Finset ℕ)
  card_V : V.card = 20
  card_E : E.card = 14
  edge_size : ∀ e ∈ E, e.card = 2
  vertex_in_edge : ∀ v ∈ V, ∃ e ∈ E, v ∈ e

/-- Definition of pairwise disjoint edges -/
def PairwiseDisjoint (G : TennisGraph) (S : Finset (Finset ℕ)) : Prop :=
  ∀ e₁ e₂, e₁ ∈ S → e₂ ∈ S → e₁ ≠ e₂ → e₁ ∩ e₂ = ∅

/-- Main theorem: There exist at least 6 pairwise disjoint edges -/
theorem tennis_match_theorem (G : TennisGraph) :
  ∃ S : Finset (Finset ℕ), S ⊆ G.E ∧ S.card ≥ 6 ∧ PairwiseDisjoint G S := by
  sorry

#check tennis_match_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_match_theorem_l253_25360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_degrees_l253_25395

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![cos θ, -sin θ;
     sin θ,  cos θ]

theorem rotation_150_degrees :
  rotation_matrix (150 * π / 180) = !![-(Real.sqrt 3) / 2, -1 / 2;
                                       1 / 2, -(Real.sqrt 3) / 2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_degrees_l253_25395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cab_time_theorem_l253_25364

/-- The usual time for a cab to cover a distance d, given specific speed changes and delay -/
theorem cab_time_theorem (d t : ℝ) : 
  t > 0 → 
  d > 0 → 
  (let v := d / t
   let v1 := (5 / 6) * v
   let v2 := (2 / 3) * v
   let t1 := d / v1
   let t2 := d / v2
   t1 + t2 = 2 * t + 5) →
  t = 50 / 7 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cab_time_theorem_l253_25364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisibility_l253_25350

theorem subset_sum_divisibility (p : ℕ) (hp : Nat.Prime p) (ho : Odd p) :
  let F := Finset.range (p - 1)
  let s (T : Finset ℕ) := T.sum id
  let count := (Finset.filter (λ T => T.Nonempty ∧ s T % p = 0) (Finset.powerset F)).card
  count = (2^(p-1) - 1) / p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisibility_l253_25350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l253_25337

noncomputable def x : ℝ := (2 + Real.sqrt 3) ^ 500
noncomputable def n : ℤ := ⌊x⌋
noncomputable def f : ℝ := x - n

theorem x_times_one_minus_f_equals_one : x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l253_25337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l253_25333

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^2
def curve2 (x : ℝ) : ℝ := x

-- Define the area of the closed figure
noncomputable def area : ℝ := ∫ x in Set.Icc 0 1, curve2 x - curve1 x

-- Theorem statement
theorem area_between_curves : area = 1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l253_25333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l253_25358

/-- The perimeter of a right triangle ABC with vertices A(0,0), B(1,0), and C(1,4) is 5 + √17. -/
theorem triangle_perimeter : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (1, 4)
  let AB : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC : ℝ := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let perimeter : ℝ := AB + BC + AC
  perimeter = 5 + Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l253_25358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l253_25302

-- Define the ellipse C
noncomputable def Ellipse (a b : ℝ) := {(x, y) : ℝ × ℝ | x^2/a^2 + y^2/b^2 = 1}

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

-- Define minimum distance from a point on C to the left focus
noncomputable def min_distance_to_left_focus (a b : ℝ) : ℝ := a - Real.sqrt (a^2 - b^2)

-- Theorem statement
theorem ellipse_properties (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : eccentricity a b = Real.sqrt 2 / 2)
  (h5 : min_distance_to_left_focus a b = Real.sqrt 2 - 1) :
  (∃ C : Set (ℝ × ℝ), C = Ellipse (Real.sqrt 2) 1) ∧
  (∃ B₂ F₁ F₂ : ℝ × ℝ, ∃ θ : ℝ, 
    B₂ ∈ Ellipse (Real.sqrt 2) 1 ∧ 
    F₁ ∈ Ellipse (Real.sqrt 2) 1 ∧ 
    F₂ ∈ Ellipse (Real.sqrt 2) 1 ∧
    θ = Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l253_25302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_dog_cost_l253_25300

/-- The cost of a hamburger -/
def h : ℚ := sorry

/-- The cost of a hot dog -/
def d : ℚ := sorry

/-- Condition from day 1 purchase -/
axiom day1 : 3 * h + 4 * d = 10

/-- Condition from day 2 purchase -/
axiom day2 : 2 * h + 3 * d = 7

/-- Theorem: The cost of a hot dog is 1 dollar -/
theorem hot_dog_cost : d = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_dog_cost_l253_25300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_n_digit_sequence_l253_25399

/-- The number of positive integers in base n representation with all different digits
    and every digit except the leftmost one differs from some digit to its left by ±1. -/
def F (n : ℕ) : ℕ := 2^(n+1) - 2*n - 2

theorem base_n_digit_sequence (n : ℕ) (hn : n > 0) :
  F n = 2^(n+1) - 2*n - 2 := by
  sorry

#eval F 3  -- Expected output: 11
#eval F 4  -- Expected output: 26
#eval F 5  -- Expected output: 57

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_n_digit_sequence_l253_25399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_theorem_l253_25317

/-- The number of boys in the line -/
def num_boys : ℕ := 8

/-- The number of girls in the line -/
def num_girls : ℕ := 12

/-- The total number of people in the line -/
def total_people : ℕ := num_boys + num_girls

/-- The number of adjacent pairs in the line -/
def num_pairs : ℕ := total_people - 1

/-- T represents the number of places where a boy and a girl are standing next to each other -/
def T : ℕ → ℕ := sorry

/-- The probability of a boy-girl or girl-boy pair at any given position -/
def prob_boy_girl_pair : ℚ := 
  (num_boys : ℚ) / total_people * (num_girls : ℚ) / (total_people - 1) +
  (num_girls : ℚ) / total_people * (num_boys : ℚ) / (total_people - 1)

/-- The expected value of T -/
def expected_value_T : ℚ := (num_pairs : ℚ) * prob_boy_girl_pair

theorem expected_value_theorem : 
  expected_value_T = 912 / 95 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_theorem_l253_25317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_negative_456_l253_25336

/-- The set of angles with the same terminal side as a given angle -/
def SameTerminalSide (θ : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = θ + k * 360}

/-- 264° is coterminal with -456° -/
axiom coterminal : (264 : ℝ) = -456 + 360

theorem same_terminal_side_negative_456 :
  SameTerminalSide (-456) = {α : ℝ | ∃ k : ℤ, α = k * 360 + 264} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_negative_456_l253_25336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_1023_l253_25377

def sequence_a : ℕ → ℕ
  | 0 => 1
  | (n + 1) => 2 * sequence_a n + 1

theorem a_10_equals_1023 : sequence_a 10 = 1023 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_1023_l253_25377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_in_trisected_angle_l253_25366

-- Define the necessary types and functions
variable (Point : Type)
variable (Trisects : Point → Point → Point → Point → Point → Prop)
variable (Bisects : Point → Point → Point → Point → Prop)
variable (angle : Point → Point → Point → ℝ)

-- The main theorem
theorem angle_ratio_in_trisected_angle (A B C P Q M : Point) :
  Trisects C P Q A B →
  Bisects C M P Q →
  (angle M C Q) / (angle A C Q) = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_in_trisected_angle_l253_25366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoya_has_more_books_l253_25385

/-- The number of science books Xiao Ya has -/
def xiaoya_science : ℕ := sorry

/-- The number of storybooks Xiao Ya has -/
def xiaoya_story : ℕ := sorry

/-- The total number of science books -/
def total_science : ℕ := 66

/-- The total number of storybooks -/
def total_story : ℕ := 92

/-- Xiao Pang's science books are twice Xiao Ya's -/
axiom xiaopang_science : xiaoya_science * 2 = total_science - xiaoya_science

/-- Xiao Ya's storybooks are three times Xiao Pang's -/
axiom xiaoya_story_triple : xiaoya_story = (total_story - xiaoya_story) * 3

/-- The difference in the number of books between Xiao Ya and Xiao Pang -/
def book_difference : ℤ := (xiaoya_science + xiaoya_story : ℤ) - 
  ((total_science - xiaoya_science) + (total_story - xiaoya_story) : ℤ)

theorem xiaoya_has_more_books : book_difference = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoya_has_more_books_l253_25385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_medians_l253_25393

/-- The sum of squares of medians of a triangle with sides 13, 14, and 15 is 442.5 -/
theorem sum_of_squares_of_medians (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let m₁ := ((2 * b^2 + 2 * c^2 - a^2) / 4 : ℝ)
  let m₂ := ((2 * c^2 + 2 * a^2 - b^2) / 4 : ℝ)
  let m₃ := ((2 * a^2 + 2 * b^2 - c^2) / 4 : ℝ)
  m₁ + m₂ + m₃ = 442.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_medians_l253_25393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_lambda_bound_l253_25342

def a (n : ℕ+) (lambda : ℝ) : ℝ := n.val ^ 2 + lambda * n.val

theorem increasing_sequence_lambda_bound (lambda : ℝ) :
  (∀ n m : ℕ+, n < m → a n lambda < a m lambda) → lambda > -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_lambda_bound_l253_25342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_agency_daily_charge_value_l253_25367

/-- The daily charge of the first agency -/
def first_agency_daily_charge : ℝ := 20.25

/-- The per-mile charge of the first agency -/
def first_agency_mile_charge : ℝ := 0.14

/-- The daily charge of the second agency -/
def second_agency_daily_charge : ℝ := 18.25

/-- The per-mile charge of the second agency -/
def second_agency_mile_charge : ℝ := 0.22

/-- The distance at which the first agency becomes less expensive -/
def crossover_distance : ℝ := 25.0

theorem first_agency_daily_charge_value :
  first_agency_daily_charge = 20.25 :=
by
  -- The proof is trivial since we defined first_agency_daily_charge as 20.25
  rfl

#check first_agency_daily_charge_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_agency_daily_charge_value_l253_25367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_y_value_l253_25379

theorem terminal_side_y_value (α : ℝ) (y : ℝ) :
  (∃ (P : ℝ × ℝ), P = (-3, y) ∧ P.1 = -3) →
  Real.sin α = 4 / 5 →
  y = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_y_value_l253_25379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l253_25303

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x + Real.sin x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x + m)

theorem min_translation_for_symmetry :
  ∃ m : ℝ, m > 0 ∧
  (∀ x : ℝ, g m x = -g m (-x)) ∧
  (∀ m' : ℝ, m' > 0 ∧ (∀ x : ℝ, g m' x = -g m' (-x)) → m ≤ m') ∧
  m = 2 * Real.pi / 3 := by
  sorry

#check min_translation_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l253_25303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_theorem_l253_25311

/-- The time taken for two workers to complete a job together -/
noncomputable def time_together (time_a time_b : ℝ) : ℝ :=
  1 / (1 / time_a + 1 / time_b)

/-- Theorem: Given the individual work times, prove the combined work time -/
theorem work_time_theorem (time_a time_b : ℝ) 
  (ha : time_a = 12) 
  (hb : time_b = 14) : 
  time_together time_a time_b = 84 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_theorem_l253_25311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_overall_rate_l253_25363

/-- Represents the correctness rate for homework problems -/
structure HomeworkStats where
  individual_rate : ℚ
  total_rate : ℚ

/-- Calculates the overall correctness rate for a student -/
def calculate_overall_rate (individual_rate : ℚ) (shared_rate : ℚ) : ℚ :=
  (2/3 * individual_rate) + (1/3 * shared_rate)

/-- Theorem stating Mia's overall correctness rate -/
theorem mia_overall_rate (liam : HomeworkStats) (mia_individual_rate : ℚ) 
  (h1 : liam.individual_rate = 70/100)
  (h2 : liam.total_rate = 82/100)
  (h3 : mia_individual_rate = 85/100) :
  calculate_overall_rate mia_individual_rate ((3 * liam.total_rate - 2 * liam.individual_rate)) = 92/100 := by
  sorry

#eval calculate_overall_rate (85/100) ((3 * 82/100 - 2 * 70/100))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_overall_rate_l253_25363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_cut_theorem_l253_25394

structure CircleCut where
  parts : Set (Set (EuclideanSpace ℝ (Fin 2)))
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

def is_valid_cut (cut : CircleCut) : Prop :=
  ∀ part ∈ cut.parts, cut.center ∈ (frontier part)

def can_form_hexagon (cut : CircleCut) : Prop :=
  ∃ hexagon_parts : Set (Set (EuclideanSpace ℝ (Fin 2))),
    hexagon_parts ⊆ cut.parts ∧
    ∃ vertices : Fin 6 → EuclideanSpace ℝ (Fin 2),
      (∀ i : Fin 6, ‖vertices i - cut.center‖ = cut.radius) ∧
      (∀ i : Fin 6, ‖vertices i - vertices (i.succ)‖ = cut.radius)

theorem circle_cut_theorem :
  ∃ cut : CircleCut, is_valid_cut cut ∧ can_form_hexagon cut :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_cut_theorem_l253_25394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_difference_l253_25301

-- Define the points for lines p and q
def p1 : ℝ × ℝ := (0, 8)
def p2 : ℝ × ℝ := (4, 0)
def q1 : ℝ × ℝ := (0, 5)
def q2 : ℝ × ℝ := (10, 0)

-- Define the slope of a line given two points
noncomputable def lineSlope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the y-intercept of a line given a point and slope
def yIntercept (p : ℝ × ℝ) (m : ℝ) : ℝ :=
  p.2 - m * p.1

-- Define the x-coordinate where a line reaches y = 20
noncomputable def xAt20 (m : ℝ) (b : ℝ) : ℝ :=
  (20 - b) / m

-- Theorem statement
theorem x_coordinate_difference : 
  let mp := lineSlope p1 p2
  let mq := lineSlope q1 q2
  let bp := yIntercept p1 mp
  let bq := yIntercept q1 mq
  let xp := xAt20 mp bp
  let xq := xAt20 mq bq
  |xp - xq| = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_difference_l253_25301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_and_phase_shift_l253_25380

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (3 * x + Real.pi / 4)

theorem amplitude_and_phase_shift (x : ℝ) :
  (∃ A B C, f x = A * Real.sin (B * x + C)) →
  (∃ A, ∀ x, |f x| ≤ A ∧ ∃ x₀, |f x₀| = A) ∧
  (∃ φ, ∀ x, f x = f (x + φ) ∧ ∀ ψ, 0 < ψ ∧ ψ < |φ| → ∃ x, f x ≠ f (x + ψ)) →
  (∃ A φ, A = 3 ∧ φ = -Real.pi/12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_and_phase_shift_l253_25380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_circle_l253_25378

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 1 / (1 + Real.sin θ)

-- Define a circle in Cartesian coordinates
def is_circle (x y : ℝ → ℝ) : Prop :=
  ∃ (h k r : ℝ), ∀ t, (x t - h)^2 + (y t - k)^2 = r^2

-- Theorem statement
theorem polar_equation_is_circle :
  ∃ (x y : ℝ → ℝ), (∀ t, polar_equation (Real.sqrt (x t^2 + y t^2)) (Real.arctan (y t / x t))) →
    is_circle x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_circle_l253_25378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_correct_l253_25355

/-- A right triangle with sides 5, 12, and 13 inches -/
structure RightTriangle where
  a : ℚ
  b : ℚ
  c : ℚ
  right_triangle : a^2 + b^2 = c^2
  a_eq : a = 5
  b_eq : b = 12
  c_eq : c = 13

/-- The length of the crease when folding vertex A to vertex C -/
noncomputable def crease_length (t : RightTriangle) : ℝ :=
  Real.sqrt (7336806 / 1000000)

/-- Theorem stating that the crease length is correct -/
theorem crease_length_correct (t : RightTriangle) :
  crease_length t = Real.sqrt (7.336806) := by
  sorry

#eval (7336806 : ℚ) / 1000000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_correct_l253_25355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_equation_solution_l253_25371

theorem square_root_equation_solution : ∃ y : ℝ, y > 0 ∧ 
  Real.sqrt ((7 * 2.3333333333333335) / y) = 2.3333333333333335 ∧ 
  |y - 3| < 0.0000001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_equation_solution_l253_25371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_AP_and_AB_l253_25387

noncomputable def AP : Fin 2 → ℝ := ![1, Real.sqrt 3]
noncomputable def PB : Fin 2 → ℝ := ![-Real.sqrt 3, 1]

noncomputable def AB : Fin 2 → ℝ := ![1 - Real.sqrt 3, Real.sqrt 3 + 1]

theorem angle_between_AP_and_AB :
  let dot_product := (AP 0) * (AB 0) + (AP 1) * (AB 1)
  let magnitude_AP := Real.sqrt ((AP 0)^2 + (AP 1)^2)
  let magnitude_AB := Real.sqrt ((AB 0)^2 + (AB 1)^2)
  let cos_theta := dot_product / (magnitude_AP * magnitude_AB)
  Real.arccos cos_theta = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_AP_and_AB_l253_25387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l253_25346

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_properties 
  (a₁ d : ℝ) 
  (h_d : d ≠ 0) 
  (h_a₃_eq_S₅ : arithmetic_sequence a₁ d 3 = arithmetic_sum a₁ d 5)
  (h_a₂a₄_eq_S₄ : arithmetic_sequence a₁ d 2 * arithmetic_sequence a₁ d 4 = arithmetic_sum a₁ d 4) :
  (∀ n : ℕ, arithmetic_sequence a₁ d n = 2 * n - 6) ∧
  (∀ n : ℕ, n ≥ 7 ↔ arithmetic_sum a₁ d n > arithmetic_sequence a₁ d n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l253_25346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_theorem_l253_25365

/-- Represents a class in the study -/
structure StudyClass where
  size : ℕ
  selected : ℕ

/-- Represents the study group -/
def StudyGroup (class1 class2 : StudyClass) : ℕ := class1.selected + class2.selected

/-- Calculates the probability of selecting students from different classes in two draws -/
def probability_different_classes (class1 class2 : StudyClass) : ℚ :=
  (class1.selected * class2.selected * 2) / (StudyGroup class1 class2)^2

theorem stratified_sampling_theorem (class1 class2 : StudyClass) 
  (h1 : class1.size = 18)
  (h2 : class2.size = 27)
  (h3 : class2.selected = 3)
  (h4 : class1.selected * class2.size = class2.selected * class1.size) : 
  StudyGroup class1 class2 = 5 ∧ 
  probability_different_classes class1 class2 = 12/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_theorem_l253_25365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_first_1500_even_integers_l253_25332

def count_digits (n : ℕ) : ℕ := 
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

def sum_digits_even (upper_bound : ℕ) : ℕ :=
  (List.range (upper_bound/2)).map (fun i => count_digits ((i+1)*2)) |>.sum

theorem sum_digits_first_1500_even_integers : 
  sum_digits_even 3000 = 5448 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_first_1500_even_integers_l253_25332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_purchasing_problem_l253_25309

theorem book_purchasing_problem :
  -- Define the cost of books A and B
  let cost_A : ℚ := 30
  let cost_B : ℚ := 50

  -- Define the conditions
  let condition1 : Prop := cost_A + 3 * cost_B = 180
  let condition2 : Prop := 3 * cost_A + cost_B = 140
  let budget_constraint (m : ℕ) : Prop := (cost_A * m + cost_B * (3/2 * ↑m)) ≤ 700
  let integer_constraint (m : ℕ) : Prop := ∃ k : ℕ, 3 * m = 2 * k

  -- Define the possible scenarios
  let scenarios : List (ℕ × ℕ) := [(2, 3), (4, 6), (6, 9)]

  -- Theorem statement
  (condition1 ∧ condition2) →
  (∀ m : ℕ, budget_constraint m ∧ integer_constraint m →
    (m, (3 * m / 2)) ∈ scenarios) ∧
  (∀ s ∈ scenarios, let (a, b) := s; budget_constraint a ∧ integer_constraint a ∧ b = 3 * a / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_purchasing_problem_l253_25309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_in_specific_quadrilateral_l253_25390

/-- The radius of the largest inscribed circle in a quadrilateral with given side lengths -/
noncomputable def largest_inscribed_circle_radius (a b c d : ℝ) : ℝ :=
  let s := (a + b + c + d) / 2
  Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d) / s)

/-- Theorem: The radius of the largest inscribed circle in a quadrilateral with sides 10, 15, 8, and 13 -/
theorem largest_circle_in_specific_quadrilateral :
  largest_inscribed_circle_radius 10 15 8 13 = Real.sqrt (15600 / 23) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_in_specific_quadrilateral_l253_25390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_x_minus_abs_y_l253_25316

theorem min_value_x_minus_abs_y (x y : ℝ) 
  (h : Real.log (x + 2*y) / Real.log 4 + Real.log (x - 2*y) / Real.log 4 = 1) :
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ ∀ (z : ℝ), z = x - abs y → z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_x_minus_abs_y_l253_25316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_reciprocal_f_l253_25320

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (Real.cos (2 * x) - (a + 2) * Real.cos x + a + 1) / Real.sin x

theorem integral_of_reciprocal_f (a : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |f a x / x - 1/2| < ε) →
  ∫ x in (π / 3)..(π / 2), 1 / f a x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_reciprocal_f_l253_25320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dining_out_savings_l253_25359

/-- Represents the relationship between savings and months of not dining out -/
def SavingsRelation (savings : ℝ) (months : ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ savings = k * months

theorem dining_out_savings 
  (h : SavingsRelation 150 5) : 
  SavingsRelation 240 8 := by
  sorry

#check dining_out_savings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dining_out_savings_l253_25359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l253_25314

noncomputable def projection (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let v_norm_squared := v.1 * v.1 + v.2 * v.2
  (dot_product / v_norm_squared * v.1, dot_product / v_norm_squared * v.2)

theorem projection_property (v : ℝ × ℝ) :
  projection v (3, -1) = (24/5, -8/5) →
  projection v (1, 4) = (-3/10, 1/10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l253_25314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_three_numbers_l253_25391

theorem existence_of_three_numbers (S : Finset ℝ) (h : Finset.card S = 1400) :
  ∃ x y z, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    |((x - y) * (y - z) * (z - x)) / (x^4 + y^4 + z^4 + 1)| < 0.009 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_three_numbers_l253_25391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l253_25354

/-- A line in the form 2x - y + a = 0 -/
structure Line where
  a : ℝ
  eq : ℝ → ℝ → Prop := fun x y ↦ 2 * x - y + a = 0

/-- A circle in the form x^2 + y^2 - 4x + 6y - 12 = 0 -/
def Circle : ℝ → ℝ → Prop :=
  fun x y ↦ x^2 + y^2 - 4*x + 6*y - 12 = 0

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem line_circle_intersection (l : Line) (h : l.a > -5) :
  ∃ M N : ℝ × ℝ,
    l.eq M.1 M.2 ∧
    l.eq N.1 N.2 ∧
    Circle M.1 M.2 ∧
    Circle N.1 N.2 ∧
    distance M.1 M.2 N.1 N.2 = 4 * Real.sqrt 5 →
    l.a = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l253_25354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_M_l253_25351

/-- Converts polar coordinates to rectangular coordinates -/
noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- The point M in polar coordinates -/
noncomputable def M : ℝ × ℝ := (5, Real.pi/3)

theorem polar_to_rectangular_M :
  polar_to_rectangular M.1 M.2 = (5/2, 5*Real.sqrt 3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_M_l253_25351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_obtuse_triangle_is_three_sixty_fourth_l253_25339

/-- The probability that no three out of four randomly chosen points on a circle
    form an obtuse triangle with the circle's center -/
noncomputable def probability_no_obtuse_triangle : ℝ := 3 / 64

/-- Theorem stating that the probability of no three out of four randomly chosen points
    on a circle forming an obtuse triangle with the circle's center is 3/64 -/
theorem probability_no_obtuse_triangle_is_three_sixty_fourth :
  probability_no_obtuse_triangle = 3 / 64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_obtuse_triangle_is_three_sixty_fourth_l253_25339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_between_factorial_minus_m_and_minus_one_l253_25347

theorem no_primes_between_factorial_minus_m_and_minus_one (m : ℕ) (h : m > 2) :
  ∀ k, Nat.factorial m - m ≤ k ∧ k ≤ Nat.factorial m - 1 → ¬ Nat.Prime k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_between_factorial_minus_m_and_minus_one_l253_25347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_perfect_square_differences_l253_25341

/-- A quadratic polynomial -/
def QuadraticPolynomial (α : Type*) [Ring α] := α → α

/-- The property that P(a) - P(b) is a perfect square for some integers a and b -/
def HasPerfectSquareDifference (P : QuadraticPolynomial ℤ) : Prop :=
  ∃ (a b : ℤ) (n : ℕ), P a - P b = n^2

/-- The theorem to be proved -/
theorem quadratic_polynomial_perfect_square_differences
  (P : QuadraticPolynomial ℤ)
  (h : HasPerfectSquareDifference P) :
  ∃ (S : Set (ℤ × ℤ)), (∀ (c d : ℤ), (c, d) ∈ S → ∃ (m : ℕ), P c - P d = m^2) ∧ Infinite S :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_perfect_square_differences_l253_25341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hamburger_sales_theorem_l253_25352

/-- Represents the days Martin works --/
inductive WorkDay
| Monday
| Tuesday
| Wednesday

/-- Returns the number of hamburgers sold before 6 for a given day --/
def hamburgers_before_6 (day : WorkDay) : ℕ :=
  match day with
  | .Monday => 48
  | .Tuesday => 36
  | .Wednesday => 52

/-- Returns the additional number of hamburgers sold after 6 for a given day --/
def additional_hamburgers_after_6 (day : WorkDay) : ℕ :=
  match day with
  | .Monday => 28
  | .Tuesday => 15
  | .Wednesday => 21

/-- Returns the price of hamburgers for a given day --/
def hamburger_price (day : WorkDay) : ℚ :=
  match day with
  | .Monday => 3.5
  | .Tuesday => 4.25
  | .Wednesday => 3.75

/-- Calculates the total number of hamburgers sold after 6 during the work week --/
def total_hamburgers_after_6 : ℕ :=
  (hamburgers_before_6 WorkDay.Monday + additional_hamburgers_after_6 WorkDay.Monday) +
  (hamburgers_before_6 WorkDay.Tuesday + additional_hamburgers_after_6 WorkDay.Tuesday) +
  (hamburgers_before_6 WorkDay.Wednesday + additional_hamburgers_after_6 WorkDay.Wednesday)

/-- Calculates the total revenue from hamburgers sold after 6 during the work week --/
def total_revenue_after_6 : ℚ :=
  (hamburger_price WorkDay.Monday * (hamburgers_before_6 WorkDay.Monday + additional_hamburgers_after_6 WorkDay.Monday : ℚ)) +
  (hamburger_price WorkDay.Tuesday * (hamburgers_before_6 WorkDay.Tuesday + additional_hamburgers_after_6 WorkDay.Tuesday : ℚ)) +
  (hamburger_price WorkDay.Wednesday * (hamburgers_before_6 WorkDay.Wednesday + additional_hamburgers_after_6 WorkDay.Wednesday : ℚ))

theorem hamburger_sales_theorem :
  total_hamburgers_after_6 = 200 ∧ total_revenue_after_6 = 756.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hamburger_sales_theorem_l253_25352

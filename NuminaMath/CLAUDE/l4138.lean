import Mathlib

namespace initial_water_percentage_l4138_413853

theorem initial_water_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_volume = 125)
  (h2 : added_water = 8.333333333333334)
  (h3 : final_water_percentage = 25)
  (h4 : (initial_volume * x + added_water) / (initial_volume + added_water) * 100 = final_water_percentage) :
  x * 100 = 20 :=
by
  sorry

#check initial_water_percentage

end initial_water_percentage_l4138_413853


namespace binary_10010_is_18_l4138_413844

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10010_is_18 : 
  binary_to_decimal [false, true, false, false, true] = 18 := by
  sorry

end binary_10010_is_18_l4138_413844


namespace lawn_area_l4138_413825

/-- Calculates the area of a lawn in a rectangular park with crossroads -/
theorem lawn_area (park_length park_width road_width : ℝ) 
  (h1 : park_length = 60)
  (h2 : park_width = 40)
  (h3 : road_width = 3) : 
  park_length * park_width - 
  (park_length * road_width + park_width * road_width - road_width * road_width) = 2109 :=
by sorry

end lawn_area_l4138_413825


namespace symmetric_points_solution_l4138_413880

/-- 
Given two points P and Q that are symmetric about the x-axis,
prove that their coordinates satisfy the given conditions and
result in specific values for a and b.
-/
theorem symmetric_points_solution :
  ∀ (a b : ℝ),
  let P : ℝ × ℝ := (-a + 3*b, 3)
  let Q : ℝ × ℝ := (-5, a - 2*b)
  -- P and Q are symmetric about the x-axis
  (P.1 = Q.1 ∧ P.2 = -Q.2) →
  (a = -19 ∧ b = -8) :=
by sorry

end symmetric_points_solution_l4138_413880


namespace last_digit_2014_power_2014_l4138_413866

/-- The last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Exponentiation modulo 10 -/
def powerMod10 (base exponent : ℕ) : ℕ :=
  (base ^ exponent) % 10

theorem last_digit_2014_power_2014 :
  lastDigit (powerMod10 2014 2014) = 6 := by
  sorry

end last_digit_2014_power_2014_l4138_413866


namespace correct_average_after_misreading_l4138_413874

theorem correct_average_after_misreading (n : ℕ) (incorrect_avg : ℚ) 
  (misread_numbers : List (ℚ × ℚ)) :
  n = 20 ∧ 
  incorrect_avg = 85 ∧ 
  misread_numbers = [(90, 30), (120, 60), (75, 25), (150, 50), (45, 15)] →
  (n : ℚ) * incorrect_avg + (misread_numbers.map (λ p => p.1 - p.2)).sum = n * 100 := by
  sorry

#check correct_average_after_misreading

end correct_average_after_misreading_l4138_413874


namespace empire_state_height_is_443_l4138_413858

/-- The height of the Petronas Towers in meters -/
def petronas_height : ℝ := 452

/-- The height difference between the Empire State Building and the Petronas Towers in meters -/
def height_difference : ℝ := 9

/-- The height of the Empire State Building in meters -/
def empire_state_height : ℝ := petronas_height - height_difference

theorem empire_state_height_is_443 : empire_state_height = 443 := by
  sorry

end empire_state_height_is_443_l4138_413858


namespace share_distribution_l4138_413837

theorem share_distribution (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) (h1 : total = 6600) (h2 : ratio1 = 2) (h3 : ratio2 = 4) (h4 : ratio3 = 6) :
  (total * ratio1) / (ratio1 + ratio2 + ratio3) = 1100 := by
sorry

end share_distribution_l4138_413837


namespace fib_100_mod_7_l4138_413875

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fib_100_mod_7 : fib 100 % 7 = 3 := by
  sorry

end fib_100_mod_7_l4138_413875


namespace z_min_max_in_D_l4138_413884

-- Define the function z
def z (x y : ℝ) : ℝ := 4 * x^2 + y^2 - 16 * x - 4 * y + 20

-- Define the region D
def D : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.1 - 2 * p.2 ≤ 0 ∧ p.1 + p.2 - 6 ≤ 0}

-- Theorem statement
theorem z_min_max_in_D :
  (∃ p ∈ D, ∀ q ∈ D, z p.1 p.2 ≤ z q.1 q.2) ∧
  (∃ p ∈ D, ∀ q ∈ D, z p.1 p.2 ≥ z q.1 q.2) ∧
  (∃ p ∈ D, z p.1 p.2 = 0) ∧
  (∃ p ∈ D, z p.1 p.2 = 32) :=
sorry

end z_min_max_in_D_l4138_413884


namespace school_travel_time_l4138_413891

theorem school_travel_time (usual_rate : ℝ) (usual_time : ℝ) : 
  (usual_time > 0) →
  (17 / 13 * usual_rate * (usual_time - 7) = usual_rate * usual_time) →
  usual_time = 119 / 4 := by
  sorry

end school_travel_time_l4138_413891


namespace tangent_points_focus_slope_l4138_413832

/-- The slope of the line connecting the tangent points and the focus of a parabola -/
theorem tangent_points_focus_slope (x₀ y₀ : ℝ) : 
  x₀ = -1 → y₀ = 2 → 
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- Tangent points satisfy the parabola equation
    y₁^2 = 4*x₁ ∧ y₂^2 = 4*x₂ ∧
    -- Tangent lines pass through (x₀, y₀)
    (∃ k₁ k₂ : ℝ, y₁ - y₀ = k₁*(x₁ - x₀) ∧ y₂ - y₀ = k₂*(x₂ - x₀)) →
    -- Slope of the line connecting tangent points and focus
    (y₁ - 1/4) / (x₁ - 1/4) = 1 ∧ (y₂ - 1/4) / (x₂ - 1/4) = 1 :=
by sorry

end tangent_points_focus_slope_l4138_413832


namespace hyperbolas_same_asymptotes_l4138_413887

/-- Given two hyperbolas with equations (x²/9) - (y²/16) = 1 and (y²/25) - (x²/M) = 1
    that have the same asymptotes, M equals 225/16. -/
theorem hyperbolas_same_asymptotes (M : ℝ) : 
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / M = 1) → M = 225 / 16 :=
by sorry

end hyperbolas_same_asymptotes_l4138_413887


namespace original_cube_side_length_l4138_413816

/-- Given a cube of side length s that is painted and cut into smaller cubes of side 3,
    if there are exactly 12 smaller cubes with paint on 2 sides, then s = 6 -/
theorem original_cube_side_length (s : ℕ) : 
  s > 0 →  -- ensure the side length is positive
  (12 * (s / 3 - 1) = 12) →  -- condition for 12 smaller cubes with paint on 2 sides
  s = 6 := by
sorry

end original_cube_side_length_l4138_413816


namespace negation_equivalence_not_always_greater_product_quadratic_roots_condition_l4138_413859

-- Statement 1
theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) :=
sorry

-- Statement 2
theorem not_always_greater_product :
  ∃ a b c d : ℝ, a > b ∧ c > d ∧ a * c ≤ b * d :=
sorry

-- Statement 3
theorem quadratic_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + (a - 3) * x + a = 0 ∧ y^2 + (a - 3) * y + a = 0) →
  a < 0 :=
sorry

end negation_equivalence_not_always_greater_product_quadratic_roots_condition_l4138_413859


namespace fred_grew_nine_onions_l4138_413808

/-- The number of onions Sally grew -/
def sally_onions : ℕ := 5

/-- The number of onions Sally and Fred gave away -/
def onions_given_away : ℕ := 4

/-- The number of onions Sally and Fred have remaining -/
def onions_remaining : ℕ := 10

/-- The number of onions Fred grew -/
def fred_onions : ℕ := sally_onions + onions_given_away + onions_remaining - sally_onions - onions_given_away

theorem fred_grew_nine_onions : fred_onions = 9 := by
  sorry

end fred_grew_nine_onions_l4138_413808


namespace f_max_min_on_interval_l4138_413864

-- Define the function f(x) = x³ - 2x² + 5
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5

-- Define the interval [-2, 2]
def interval : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

-- Theorem stating the maximum and minimum values of f(x) on the interval [-2, 2]
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 5 ∧ min = -11 := by
  sorry

end f_max_min_on_interval_l4138_413864


namespace land_area_decreases_l4138_413826

theorem land_area_decreases (a : ℝ) (h : a > 4) : a^2 > (a+4)*(a-4) := by
  sorry

end land_area_decreases_l4138_413826


namespace max_integer_value_x_l4138_413898

theorem max_integer_value_x (x : ℤ) : 
  (3 : ℚ) * x - 1/4 ≤ 1/3 * x - 2 → x ≤ -1 :=
by sorry

end max_integer_value_x_l4138_413898


namespace set_one_two_three_not_triangle_l4138_413802

/-- Triangle Inequality Theorem: A set of three positive real numbers a, b, and c can form a triangle
    if and only if the sum of any two is greater than the third. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The set {1, 2, 3} cannot form a triangle. -/
theorem set_one_two_three_not_triangle :
  ¬ can_form_triangle 1 2 3 := by
  sorry

#check set_one_two_three_not_triangle

end set_one_two_three_not_triangle_l4138_413802


namespace matrix_N_property_l4138_413829

open Matrix

theorem matrix_N_property (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec ![3, -2] = ![4, 1])
  (h2 : N.mulVec ![-4, 6] = ![-2, 0]) :
  N.mulVec ![7, 0] = ![6, 2] := by
  sorry

end matrix_N_property_l4138_413829


namespace complex_arithmetic_result_l4138_413833

theorem complex_arithmetic_result : 
  ((2 - 3*I) + (4 + 6*I)) * (-1 + 2*I) = -12 + 9*I :=
by sorry

end complex_arithmetic_result_l4138_413833


namespace sum_of_digits_888_base8_l4138_413851

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the sum of digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_888_base8 : sumDigits (toBase8 888) = 13 := by
  sorry

end sum_of_digits_888_base8_l4138_413851


namespace increasing_cubic_function_l4138_413804

/-- A function f(x) = x^3 - ax^2 - 3x is increasing on [1, +∞) if and only if a ≤ 0 -/
theorem increasing_cubic_function (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → (deriv (fun x => x^3 - a*x^2 - 3*x)) x ≥ 0) ↔ a ≤ 0 := by
  sorry

end increasing_cubic_function_l4138_413804


namespace f_minus_three_halves_value_l4138_413872

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def f_squared_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x ∧ x < 1 → f x = x^2

theorem f_minus_three_halves_value
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_period : has_period_two f)
  (h_squared : f_squared_on_unit_interval f) :
  f (-3/2) = -1/4 := by
  sorry

end f_minus_three_halves_value_l4138_413872


namespace projectile_max_height_l4138_413897

/-- The height function of the projectile --/
def h (t : ℝ) : ℝ := -12 * t^2 + 48 * t + 25

/-- The maximum height reached by the projectile --/
theorem projectile_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 73 :=
sorry

end projectile_max_height_l4138_413897


namespace sin_arctan_equality_l4138_413819

theorem sin_arctan_equality : ∃ (x : ℝ), x > 0 ∧ Real.sin (Real.arctan x) = x := by
  let x := Real.sqrt ((-1 + Real.sqrt 5) / 2)
  use x
  have h1 : x > 0 := sorry
  have h2 : Real.sin (Real.arctan x) = x := sorry
  exact ⟨h1, h2⟩

#check sin_arctan_equality

end sin_arctan_equality_l4138_413819


namespace subset_condition_intersection_condition_l4138_413869

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 ≤ x ∧ x ≤ m + 1}

-- Theorem 1: B ⊆ A iff m ∈ [-1, +∞)
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≥ -1 := by sorry

-- Theorem 2: ∃x ∈ A such that x ∈ B iff m ∈ [-4, 2]
theorem intersection_condition (m : ℝ) : (∃ x, x ∈ A ∧ x ∈ B m) ↔ -4 ≤ m ∧ m ≤ 2 := by sorry

end subset_condition_intersection_condition_l4138_413869


namespace base3_sum_equality_l4138_413893

/-- Converts a base 3 number represented as a list of digits to a natural number. -/
def base3ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 3 * acc) 0

/-- The sum of 2₃, 121₃, 1212₃, and 12121₃ equals 2111₃ in base 3. -/
theorem base3_sum_equality : 
  base3ToNat [2] + base3ToNat [1, 2, 1] + base3ToNat [2, 1, 2, 1] + base3ToNat [1, 2, 1, 2, 1] = 
  base3ToNat [1, 1, 1, 2] := by
  sorry

#eval base3ToNat [2] + base3ToNat [1, 2, 1] + base3ToNat [2, 1, 2, 1] + base3ToNat [1, 2, 1, 2, 1]
#eval base3ToNat [1, 1, 1, 2]

end base3_sum_equality_l4138_413893


namespace diophantine_equation_solution_l4138_413861

/-- Given positive integers a, b, c with (a,b,c) = 1 and (a,b) = d, 
    if n > (ab/d) + cd - a - b - c, then there exist nonnegative integers x, y, z 
    such that ax + by + cz = n -/
theorem diophantine_equation_solution 
  (a b c d : ℕ+) (n : ℕ) 
  (h1 : Nat.gcd a.val (Nat.gcd b.val c.val) = 1)
  (h2 : Nat.gcd a.val b.val = d.val)
  (h3 : n > a.val * b.val / d.val + c.val * d.val - a.val - b.val - c.val) :
  ∃ x y z : ℕ, a.val * x + b.val * y + c.val * z = n :=
sorry

end diophantine_equation_solution_l4138_413861


namespace initial_balloons_l4138_413886

theorem initial_balloons (initial : ℕ) : initial + 2 = 11 → initial = 9 := by
  sorry

end initial_balloons_l4138_413886


namespace probability_most_expensive_chosen_l4138_413862

def num_computers : ℕ := 10
def num_display : ℕ := 3

theorem probability_most_expensive_chosen :
  (Nat.choose (num_computers - 2) (num_display - 2)) / (Nat.choose num_computers num_display) = 1 / 15 := by
sorry

end probability_most_expensive_chosen_l4138_413862


namespace point_quadrant_relation_l4138_413806

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Checks if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem stating that if P(a+b, ab) is in the second quadrant, 
    then Q(-a, b) is in the fourth quadrant -/
theorem point_quadrant_relation (a b : ℝ) :
  isInSecondQuadrant (Point.mk (a + b) (a * b)) →
  isInFourthQuadrant (Point.mk (-a) b) :=
by
  sorry


end point_quadrant_relation_l4138_413806


namespace correct_system_of_equations_l4138_413818

/-- Represents the number of workers in the workshop -/
def total_workers : ℕ := 26

/-- Represents the number of screws a worker can produce per day -/
def screws_per_worker : ℕ := 800

/-- Represents the number of nuts a worker can produce per day -/
def nuts_per_worker : ℕ := 1000

/-- Represents the number of nuts needed to match one screw -/
def nuts_per_screw : ℕ := 2

/-- Theorem stating the correct system of equations for matching screws and nuts -/
theorem correct_system_of_equations (x y : ℕ) :
  (x + y = total_workers) ∧
  (nuts_per_worker * y = nuts_per_screw * screws_per_worker * x) →
  (x + y = total_workers) ∧
  (1000 * y = 2 * 800 * x) :=
by sorry

end correct_system_of_equations_l4138_413818


namespace bus_ride_difference_l4138_413899

/-- Proves that 15 more children got on the bus than got off during the entire ride -/
theorem bus_ride_difference (initial : ℕ) (final : ℕ) 
  (got_off_first : ℕ) (got_off_second : ℕ) (got_off_third : ℕ) 
  (h1 : initial = 20)
  (h2 : final = 35)
  (h3 : got_off_first = 54)
  (h4 : got_off_second = 30)
  (h5 : got_off_third = 15) :
  ∃ (got_on_total : ℕ),
    got_on_total = final - initial + got_off_first + got_off_second + got_off_third ∧
    got_on_total - (got_off_first + got_off_second + got_off_third) = 15 := by
  sorry


end bus_ride_difference_l4138_413899


namespace gcd_lcm_problem_l4138_413835

theorem gcd_lcm_problem (a b : ℕ) : 
  a > 0 → b > 0 → Nat.gcd a b = 45 → Nat.lcm a b = 1260 → a = 180 → b = 315 := by
  sorry

end gcd_lcm_problem_l4138_413835


namespace quadratic_roots_sum_of_squares_l4138_413882

theorem quadratic_roots_sum_of_squares (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (2*k - 1)*x₁ + k^2 - 1 = 0 ∧
    x₂^2 + (2*k - 1)*x₂ + k^2 - 1 = 0 ∧
    x₁^2 + x₂^2 = 19) →
  k = -2 :=
by sorry

end quadratic_roots_sum_of_squares_l4138_413882


namespace volume_of_sphere_wedge_l4138_413827

/-- The volume of a wedge from a sphere cut into six congruent parts, given the sphere's circumference --/
theorem volume_of_sphere_wedge (circumference : ℝ) :
  circumference = 18 * Real.pi →
  (1 / 6) * (4 / 3) * Real.pi * (circumference / (2 * Real.pi))^3 = 162 * Real.pi := by
  sorry

end volume_of_sphere_wedge_l4138_413827


namespace triangle_side_length_l4138_413836

theorem triangle_side_length (a b : ℝ) (A B : ℝ) :
  b = 4 * Real.sqrt 6 →
  B = π / 3 →
  A = π / 4 →
  a = (4 * Real.sqrt 6) * (Real.sin (π / 4)) / (Real.sin (π / 3)) →
  a = 8 := by
  sorry

end triangle_side_length_l4138_413836


namespace x_percent_of_z_l4138_413810

theorem x_percent_of_z (x y z : ℝ) (h1 : x = 1.30 * y) (h2 : y = 0.50 * z) : x = 0.65 * z := by
  sorry

end x_percent_of_z_l4138_413810


namespace rectangle_triangle_area_ratio_l4138_413809

/-- The ratio of the area of a rectangle to the area of a triangle formed with one side of the rectangle as base --/
theorem rectangle_triangle_area_ratio 
  (L W : ℝ) 
  (θ : ℝ) 
  (h_pos : L > 0 ∧ W > 0)
  (h_angle : 0 < θ ∧ θ < π / 2) :
  (L * W) / ((1/2) * L * W * Real.sin θ) = 2 / Real.sin θ :=
by sorry

end rectangle_triangle_area_ratio_l4138_413809


namespace binary_multiplication_theorem_l4138_413883

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n |>.reverse

def binary_1101101 : List Bool := [true, true, false, true, true, false, true]
def binary_1101 : List Bool := [true, true, false, true]
def binary_result : List Bool := [true, false, false, true, false, true, false, true, false, false, false, true]

theorem binary_multiplication_theorem :
  nat_to_binary ((binary_to_nat binary_1101101) * (binary_to_nat binary_1101)) = binary_result := by
  sorry

end binary_multiplication_theorem_l4138_413883


namespace equality_check_l4138_413834

theorem equality_check : 
  (3^2 ≠ 2^3) ∧ 
  ((-2)^3 = -2^3) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  (-(-2) ≠ -|-2|) := by
  sorry

end equality_check_l4138_413834


namespace rad_divides_theorem_l4138_413879

-- Define the rad function
def rad : ℕ → ℕ
| 0 => 1
| 1 => 1
| n+2 => (Finset.prod (Nat.factors (n+2)).toFinset id)

-- Define a polynomial with nonnegative integer coefficients
def NonnegIntPoly := {f : Polynomial ℕ // ∀ i, 0 ≤ f.coeff i}

theorem rad_divides_theorem (f : NonnegIntPoly) :
  (∀ n : ℕ, rad (f.val.eval n) ∣ rad (f.val.eval (n^(rad n)))) →
  ∃ a m : ℕ, f.val = Polynomial.monomial m a :=
sorry

end rad_divides_theorem_l4138_413879


namespace probability_consecutive_dali_prints_l4138_413871

/-- The probability of consecutive Dali prints in a random arrangement --/
theorem probability_consecutive_dali_prints
  (total_pieces : ℕ)
  (dali_prints : ℕ)
  (h1 : total_pieces = 12)
  (h2 : dali_prints = 4)
  (h3 : dali_prints ≤ total_pieces) :
  (dali_prints.factorial * (total_pieces - dali_prints + 1).factorial) /
    total_pieces.factorial = 1 / 55 :=
by sorry

end probability_consecutive_dali_prints_l4138_413871


namespace units_digit_plus_two_l4138_413823

/-- Given a positive even integer with a positive units digit, 
    if the units digit of its cube minus the units digit of its square is 0, 
    then the units digit of the number plus 2 is 8. -/
theorem units_digit_plus_two (p : ℕ) : 
  p > 0 → 
  Even p → 
  (p % 10 > 0) → 
  ((p^3 % 10) - (p^2 % 10) = 0) → 
  ((p + 2) % 10 = 8) := by
sorry

end units_digit_plus_two_l4138_413823


namespace slope_equals_twelve_implies_m_equals_negative_two_l4138_413850

/-- Given two points A(-m, 6) and B(1, 3m), prove that m = -2 when the slope of the line passing through these points is 12. -/
theorem slope_equals_twelve_implies_m_equals_negative_two (m : ℝ) : 
  (let A : ℝ × ℝ := (-m, 6)
   let B : ℝ × ℝ := (1, 3*m)
   (3*m - 6) / (1 - (-m)) = 12) → m = -2 := by
sorry

end slope_equals_twelve_implies_m_equals_negative_two_l4138_413850


namespace jack_stair_step_height_l4138_413831

/-- Given Jack's stair climbing scenario, prove the height of each step. -/
theorem jack_stair_step_height :
  -- Net flights descended
  ∀ (net_flights : ℕ),
  -- Steps per flight
  ∀ (steps_per_flight : ℕ),
  -- Total descent in inches
  ∀ (total_descent : ℕ),
  -- Given conditions
  net_flights = 3 →
  steps_per_flight = 12 →
  total_descent = 288 →
  -- Prove that the height of each step is 8 inches
  (total_descent : ℚ) / (net_flights * steps_per_flight : ℚ) = 8 := by
  sorry

end jack_stair_step_height_l4138_413831


namespace unique_prime_sum_diff_l4138_413894

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem unique_prime_sum_diff :
  ∃! p : ℕ, is_prime p ∧ 
    (∃ a b : ℕ, is_prime a ∧ is_prime b ∧ p = a + b) ∧
    (∃ c d : ℕ, is_prime c ∧ is_prime d ∧ p = c - d) :=
by
  use 5
  sorry

end unique_prime_sum_diff_l4138_413894


namespace cube_split_theorem_l4138_413843

/-- The sum of consecutive integers from 2 to n -/
def consecutiveSum (n : ℕ) : ℕ := (n + 2) * (n - 1) / 2

/-- The nth odd number starting from 3 -/
def nthOddFrom3 (n : ℕ) : ℕ := 2 * n + 1

theorem cube_split_theorem (m : ℕ) (h1 : m > 1) :
  (∃ k, k ∈ Finset.range m ∧ nthOddFrom3 (consecutiveSum m - k) = 333) ↔ m = 18 := by
  sorry

end cube_split_theorem_l4138_413843


namespace smallest_k_coprime_subset_l4138_413842

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_k_coprime_subset : ∃ (k : ℕ),
  (k = 51) ∧ 
  (∀ (S : Finset ℕ), S ⊆ Finset.range 100 → S.card ≥ k → 
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ is_coprime a b) ∧
  (∀ (k' : ℕ), k' < k → 
    ∃ (S : Finset ℕ), S ⊆ Finset.range 100 ∧ S.card = k' ∧
      ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → ¬is_coprime a b) :=
by sorry

end smallest_k_coprime_subset_l4138_413842


namespace fibonacci_geometric_sequence_l4138_413807

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the theorem
theorem fibonacci_geometric_sequence (a k p : ℕ) :
  (∃ r : ℚ, r > 1 ∧ fib k = r * fib a ∧ fib p = r * fib k) →  -- Geometric sequence condition
  (a < k ∧ k < p) →  -- Increasing order condition
  (a + k + p = 2010) →  -- Sum condition
  a = 669 := by
  sorry

end fibonacci_geometric_sequence_l4138_413807


namespace quadratic_roots_equivalence_l4138_413847

/-- A quadratic function f(x) = ax^2 + bx + c where a > 0 -/
def QuadraticFunction (a b c : ℝ) (h : a > 0) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_roots_equivalence (a b c : ℝ) (h : a > 0) :
  let f := QuadraticFunction a b c h
  (f (f (-b / (2 * a))) < 0) ↔
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ f (f y₁) = 0 ∧ f (f y₂) = 0) :=
sorry

end quadratic_roots_equivalence_l4138_413847


namespace min_value_expression_l4138_413840

theorem min_value_expression (x : ℝ) : 
  (8 - x) * (6 - x) * (8 + x) * (6 + x) ≥ -196 ∧ 
  ∃ y : ℝ, (8 - y) * (6 - y) * (8 + y) * (6 + y) = -196 :=
by sorry

end min_value_expression_l4138_413840


namespace range_of_a_l4138_413890

-- Define the property that the inequality holds for all real x
def inequality_holds_for_all (a : ℝ) : Prop :=
  ∀ x : ℝ, |a * x - 1| + |a * x - a| ≥ 1

-- Define the theorem
theorem range_of_a (a : ℝ) :
  inequality_holds_for_all a ∧ a > 0 → a ≥ 2 :=
by sorry

end range_of_a_l4138_413890


namespace lcm_problem_l4138_413868

theorem lcm_problem (a b c : ℕ) 
  (h1 : Nat.lcm a b = 60) 
  (h2 : Nat.lcm a c = 270) : 
  Nat.lcm b c = 540 := by
sorry

end lcm_problem_l4138_413868


namespace multiple_is_two_l4138_413828

-- Define the variables
def mother_age : ℕ := 40
def daughter_age : ℕ := 30 -- This is derived, not given directly
def multiple : ℚ := 2 -- This is what we want to prove

-- Define the conditions
def condition1 (m : ℕ) (d : ℕ) (x : ℚ) : Prop :=
  m + x * d = 70

def condition2 (m : ℕ) (d : ℕ) (x : ℚ) : Prop :=
  d + x * m = 95

-- Theorem statement
theorem multiple_is_two :
  condition1 mother_age daughter_age multiple ∧
  condition2 mother_age daughter_age multiple ∧
  multiple = 2 := by sorry

end multiple_is_two_l4138_413828


namespace scientific_notation_of_600000_l4138_413857

theorem scientific_notation_of_600000 : ∃ (a : ℝ) (n : ℤ), 
  1 ≤ a ∧ a < 10 ∧ 600000 = a * (10 : ℝ) ^ n :=
by
  -- Proof goes here
  sorry

end scientific_notation_of_600000_l4138_413857


namespace sum_of_herds_equals_total_l4138_413863

/-- The total number of sheep on the farm -/
def total_sheep : ℕ := 149

/-- The number of herds on the farm -/
def num_herds : ℕ := 5

/-- The number of sheep in each herd -/
def herd_sizes : Fin num_herds → ℕ
  | ⟨0, _⟩ => 23
  | ⟨1, _⟩ => 37
  | ⟨2, _⟩ => 19
  | ⟨3, _⟩ => 41
  | ⟨4, _⟩ => 29
  | ⟨n+5, h⟩ => absurd h (Nat.not_lt_of_ge (Nat.le_add_left 5 n))

/-- The theorem stating that the sum of sheep in all herds equals the total number of sheep -/
theorem sum_of_herds_equals_total :
  (Finset.univ.sum fun i => herd_sizes i) = total_sheep := by
  sorry

end sum_of_herds_equals_total_l4138_413863


namespace rearrange_3622_l4138_413801

def digits : List ℕ := [3, 6, 2, 2]

theorem rearrange_3622 : (List.permutations digits).length = 12 := by
  sorry

end rearrange_3622_l4138_413801


namespace pentagonal_tiles_count_l4138_413895

theorem pentagonal_tiles_count (t p : ℕ) : 
  t + p = 30 →  -- Total number of tiles
  3 * t + 5 * p = 100 →  -- Total number of edges
  p = 5  -- Number of pentagonal tiles
  := by sorry

end pentagonal_tiles_count_l4138_413895


namespace arrangements_with_specific_people_at_ends_l4138_413824

/-- The number of permutations of n distinct objects. -/
def permutations (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange m objects out of n distinct objects. -/
def arrangements (n m : ℕ) : ℕ := 
  if m ≤ n then
    permutations n / permutations (n - m)
  else
    0

theorem arrangements_with_specific_people_at_ends (total_people : ℕ) 
  (specific_people : ℕ) (h : total_people = 6 ∧ specific_people = 2) : 
  permutations total_people - 
  (arrangements (total_people - 2) specific_people * permutations (total_people - specific_people)) = 432 := by
  sorry

end arrangements_with_specific_people_at_ends_l4138_413824


namespace tree_height_after_three_good_years_l4138_413821

/-- Represents the growth factor of a tree in different conditions -/
inductive GrowthCondition
| Good
| Bad

/-- Calculates the height of a tree after a given number of years -/
def treeHeight (initialHeight : ℝ) (years : ℕ) (conditions : List GrowthCondition) : ℝ :=
  match years, conditions with
  | 0, _ => initialHeight
  | n+1, [] => initialHeight  -- Default to initial height if no conditions are specified
  | n+1, c::cs => 
    let newHeight := 
      match c with
      | GrowthCondition.Good => 3 * initialHeight
      | GrowthCondition.Bad => 2 * initialHeight
    treeHeight newHeight n cs

/-- Theorem stating the height of the tree after 3 years of good growth -/
theorem tree_height_after_three_good_years :
  let initialHeight : ℝ := treeHeight 1458 3 [GrowthCondition.Bad, GrowthCondition.Bad, GrowthCondition.Bad]
  treeHeight initialHeight 3 [GrowthCondition.Good, GrowthCondition.Good, GrowthCondition.Good] = 1458 :=
by sorry

#eval treeHeight 1458 3 [GrowthCondition.Bad, GrowthCondition.Bad, GrowthCondition.Bad]

end tree_height_after_three_good_years_l4138_413821


namespace square_sum_theorem_l4138_413873

theorem square_sum_theorem (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 8) : x^2 + y^2 = 33 := by
  sorry

end square_sum_theorem_l4138_413873


namespace no_solution_for_square_free_l4138_413852

/-- A positive integer is square-free if its prime factorization contains no repeated factors. -/
def IsSquareFree (n : ℕ) : Prop :=
  ∀ (p : ℕ), Nat.Prime p → (p ^ 2 ∣ n) → p = 1

/-- Two natural numbers are relatively prime if their greatest common divisor is 1. -/
def RelativelyPrime (x y : ℕ) : Prop :=
  Nat.gcd x y = 1

theorem no_solution_for_square_free (n : ℕ) (hn : IsSquareFree n) :
  ¬∃ (x y : ℕ), RelativelyPrime x y ∧ ((x + y) ^ 3 ∣ x ^ n + y ^ n) :=
sorry

end no_solution_for_square_free_l4138_413852


namespace max_value_7b_5c_l4138_413820

/-- The function f(x) = ax^2 + bx + c -/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the maximum value of 7b+5c given the conditions -/
theorem max_value_7b_5c (a b c : ℝ) : 
  (∃ a' ∈ Set.Icc 1 2, ∀ x ∈ Set.Icc 1 2, f a' b c x ≤ 1) →
  (∀ y : ℝ, 7 * b + 5 * c ≤ y) → y = -6 := by
  sorry

end max_value_7b_5c_l4138_413820


namespace sum_in_base3_l4138_413845

/-- Represents a number in base 3 --/
def Base3 : Type := List (Fin 3)

/-- Converts a natural number to its base 3 representation --/
def toBase3 (n : ℕ) : Base3 := sorry

/-- Adds two Base3 numbers --/
def addBase3 (a b : Base3) : Base3 := sorry

/-- Theorem: The sum of 2₃, 21₃, 110₃, and 2202₃ in base 3 is 11000₃ --/
theorem sum_in_base3 :
  addBase3 (toBase3 2)
    (addBase3 (toBase3 7)
      (addBase3 (toBase3 12)
        (toBase3 72))) = [1, 1, 0, 0, 0] := by sorry

end sum_in_base3_l4138_413845


namespace range_of_m_l4138_413867

def A : Set ℝ := {x | |x - 1| < 2}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

theorem range_of_m :
  (∀ m : ℝ, (∀ x : ℝ, x ∈ B m → x ∈ A) ∧ (∃ x : ℝ, x ∈ A ∧ x ∉ B m)) ↔ 
  (∀ m : ℝ, m > 2) :=
sorry

end range_of_m_l4138_413867


namespace smallest_positive_linear_combination_l4138_413815

theorem smallest_positive_linear_combination : 
  ∃ (k : ℕ), k > 0 ∧ (∃ (m n : ℤ), k = 1205 * m + 27090 * n) ∧ 
  (∀ (j : ℕ), j > 0 → (∃ (x y : ℤ), j = 1205 * x + 27090 * y) → j ≥ k) ∧
  k = 5 := by
sorry

end smallest_positive_linear_combination_l4138_413815


namespace negation_of_implication_l4138_413822

theorem negation_of_implication :
  (¬(∀ x : ℝ, x > 5 → x > 0)) ↔ (∀ x : ℝ, x ≤ 5 → x ≤ 0) :=
by sorry

end negation_of_implication_l4138_413822


namespace exists_m_with_x_squared_leq_eight_l4138_413817

theorem exists_m_with_x_squared_leq_eight : ∃ m : ℝ, m ≤ 2 ∧ ∃ x > m, x^2 ≤ 8 := by
  sorry

end exists_m_with_x_squared_leq_eight_l4138_413817


namespace sine_cosine_transformation_l4138_413860

open Real

theorem sine_cosine_transformation (x : ℝ) :
  sin (2 * x) - Real.sqrt 3 * cos (2 * x) = 2 * sin (2 * x - π / 3) := by
  sorry

end sine_cosine_transformation_l4138_413860


namespace quadratic_real_roots_iff_k_le_4_l4138_413881

/-- The quadratic function f(x) = (k - 3)x² + 2x + 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 3) * x^2 + 2 * x + 1

/-- The discriminant of the quadratic function f -/
def discriminant (k : ℝ) : ℝ := 4 - 4 * k + 12

theorem quadratic_real_roots_iff_k_le_4 :
  ∀ k : ℝ, (∃ x : ℝ, f k x = 0) ↔ k ≤ 4 := by sorry

end quadratic_real_roots_iff_k_le_4_l4138_413881


namespace root_product_l4138_413846

theorem root_product (n r : ℝ) (c d : ℝ) : 
  (c^2 - n*c + 3 = 0) → 
  (d^2 - n*d + 3 = 0) → 
  ∃ s : ℝ, ((c + 1/d)^2 - r*(c + 1/d) + s = 0) ∧ 
           ((d + 1/c)^2 - r*(d + 1/c) + s = 0) ∧ 
           (s = 16/3) := by
  sorry

end root_product_l4138_413846


namespace mod_twelve_six_nine_l4138_413812

theorem mod_twelve_six_nine (n : ℕ) : 12^6 ≡ n [ZMOD 9] → 0 ≤ n → n < 9 → n = 0 := by
  sorry

end mod_twelve_six_nine_l4138_413812


namespace distribute_seven_balls_three_boxes_l4138_413889

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 16 ways to distribute 7 indistinguishable balls
    into 3 distinguishable boxes, with each box containing at least one ball -/
theorem distribute_seven_balls_three_boxes :
  distribute_balls 7 3 = 16 := by
  sorry

end distribute_seven_balls_three_boxes_l4138_413889


namespace four_digit_sum_problem_l4138_413878

theorem four_digit_sum_problem :
  ∃ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    0 ≤ d ∧ d ≤ 9 ∧
    a > b ∧ b > c ∧ c > d ∧
    1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a = 10477 :=
by sorry

end four_digit_sum_problem_l4138_413878


namespace some_number_value_l4138_413814

theorem some_number_value (some_number : ℝ) : 
  (some_number * 10) / 100 = 0.032420000000000004 → 
  some_number = 0.32420000000000004 := by
sorry

end some_number_value_l4138_413814


namespace taxi_ride_distance_l4138_413855

/-- Calculates the distance of a taxi ride given the fare structure and total fare -/
theorem taxi_ride_distance
  (initial_fare : ℚ)
  (initial_distance : ℚ)
  (additional_fare : ℚ)
  (additional_distance : ℚ)
  (total_fare : ℚ)
  (h1 : initial_fare = 8)
  (h2 : initial_distance = 1/5)
  (h3 : additional_fare = 4/5)
  (h4 : additional_distance = 1/5)
  (h5 : total_fare = 39.2) :
  ∃ (distance : ℚ), distance = 8 ∧ 
    total_fare = initial_fare + (distance - initial_distance) / additional_distance * additional_fare :=
by sorry

end taxi_ride_distance_l4138_413855


namespace range_of_a_for_subset_intersection_when_a_is_4_l4138_413830

/-- The set A depending on the parameter a -/
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - 2*a - 5) < 0}

/-- The set B depending on the parameter a -/
def B (a : ℝ) : Set ℝ := {x | x > 2*a ∧ x < a^2 + 2}

/-- The theorem stating the range of a -/
theorem range_of_a_for_subset (a : ℝ) : 
  (a > -3/2) → (B a ⊆ A a) → (1 ≤ a ∧ a ≤ 3) :=
sorry

/-- The theorem for the specific case when a = 4 -/
theorem intersection_when_a_is_4 : 
  A 4 ∩ B 4 = {x | 8 < x ∧ x < 13} :=
sorry

end range_of_a_for_subset_intersection_when_a_is_4_l4138_413830


namespace josh_spending_l4138_413896

/-- Josh's spending problem -/
theorem josh_spending (x y : ℝ) : 
  (x - 1.75 - y = 6) → y = x - 7.75 := by
sorry

end josh_spending_l4138_413896


namespace imaginary_power_sum_l4138_413854

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^13 + i^18 + i^23 + i^28 + i^33 = i :=
by
  sorry

end imaginary_power_sum_l4138_413854


namespace range_of_dot_product_line_passes_fixed_point_l4138_413838

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the focus and vertex
def F : ℝ × ℝ := (-1, 0)
def A : ℝ × ℝ := (-2, 0)

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the vector from a point to another
def vector_to (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Theorem 1: Range of PF · PA
theorem range_of_dot_product :
  ∀ P : ℝ × ℝ, ellipse P.1 P.2 →
  0 ≤ dot_product (vector_to P F) (vector_to P A) ∧
  dot_product (vector_to P F) (vector_to P A) ≤ 12 :=
sorry

-- Define the line
def line (k m : ℝ) (x : ℝ) : ℝ := k * x + m

-- Theorem 2: Line passes through fixed point
theorem line_passes_fixed_point :
  ∀ k m : ℝ, ∀ M N : ℝ × ℝ,
  M ≠ N →
  ellipse M.1 M.2 →
  ellipse N.1 N.2 →
  M.2 = line k m M.1 →
  N.2 = line k m N.1 →
  (∃ H : ℝ × ℝ, 
    dot_product (vector_to A H) (vector_to M N) = 0 ∧
    dot_product (vector_to A H) (vector_to A H) = 
    dot_product (vector_to M H) (vector_to H N)) →
  line k m (-2/7) = 0 :=
sorry

end range_of_dot_product_line_passes_fixed_point_l4138_413838


namespace coefficient_x_fourth_power_l4138_413839

theorem coefficient_x_fourth_power (n : ℕ) (k : ℕ) : 
  n = 6 → k = 4 → (Nat.choose n k) * (2^k) = 240 := by
  sorry

end coefficient_x_fourth_power_l4138_413839


namespace unique_valid_number_l4138_413813

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000 ∧ n < 10000000) ∧
  (∀ d : ℕ, d < 7 → (∃! i : ℕ, i < 7 ∧ (n / 10^i) % 10 = d)) ∧
  (n % 100 % 2 = 0 ∧ (n / 100000) % 100 % 2 = 0) ∧
  (n % 1000 % 3 = 0 ∧ (n / 10000) % 1000 % 3 = 0) ∧
  (n % 10000 % 4 = 0 ∧ (n / 1000) % 10000 % 4 = 0) ∧
  (n % 100000 % 5 = 0 ∧ (n / 100) % 100000 % 5 = 0) ∧
  (n % 1000000 % 6 = 0 ∧ (n / 10) % 1000000 % 6 = 0)

theorem unique_valid_number : ∃! n : ℕ, is_valid_number n ∧ n = 3216540 := by
  sorry

end unique_valid_number_l4138_413813


namespace problem_statement_l4138_413870

theorem problem_statement : 
  let p := ∀ x : ℤ, x^2 > x
  let q := ∃ x : ℝ, x > 0 ∧ x + 2/x > 4
  (¬p) ∨ q := by sorry

end problem_statement_l4138_413870


namespace residue_problem_l4138_413865

theorem residue_problem : Int.mod (Int.mod (-1043) 36) 10 = 1 := by sorry

end residue_problem_l4138_413865


namespace tangent_line_to_exponential_curve_l4138_413848

/-- The line y = kx is tangent to the curve y = 2e^x if and only if k = 2e -/
theorem tangent_line_to_exponential_curve (k : ℝ) :
  (∃ x₀ : ℝ, k * x₀ = 2 * Real.exp x₀ ∧
             k = 2 * Real.exp x₀) ↔ k = 2 * Real.exp 1 :=
by sorry

end tangent_line_to_exponential_curve_l4138_413848


namespace equation_solution_l4138_413805

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (x - 20))) = 59 ∧ x = 15 := by
  sorry

end equation_solution_l4138_413805


namespace similar_triangles_leg_sum_l4138_413841

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (1/2) * a * b = 10 →
  a^2 + b^2 = 100 →
  (1/2) * c * d = 250 →
  c/a = d/b →
  c + d = 30 * Real.sqrt 5 :=
by sorry

end similar_triangles_leg_sum_l4138_413841


namespace parabola_properties_l4138_413876

/-- Represents a parabola of the form y = ax^2 + bx - 4 -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x - 4

theorem parabola_properties (p : Parabola) 
  (h1 : p.contains (-2) 0)
  (h2 : p.contains (-1) (-4))
  (h3 : p.contains 0 (-4))
  (h4 : p.contains 1 0)
  (h5 : p.contains 2 8) :
  (p.contains 0 (-4)) ∧ 
  (p.a = 2 ∧ p.b = 2) ∧ 
  (p.contains (-3) 8) := by
  sorry

end parabola_properties_l4138_413876


namespace cistern_emptying_time_l4138_413885

/-- Given a cistern with two taps, prove the emptying time of the second tap -/
theorem cistern_emptying_time (fill_time : ℝ) (combined_time : ℝ) (empty_time : ℝ) : 
  fill_time = 4 → combined_time = 44 / 7 → empty_time = 11 → 
  1 / fill_time - 1 / empty_time = 1 / combined_time := by
sorry

end cistern_emptying_time_l4138_413885


namespace divisibility_by_twelve_l4138_413849

theorem divisibility_by_twelve (m : Nat) : m ≤ 9 → (915 * 10 + m) % 12 = 0 ↔ m = 6 := by
  sorry

end divisibility_by_twelve_l4138_413849


namespace tomato_cucumber_price_difference_l4138_413892

theorem tomato_cucumber_price_difference :
  ∀ (tomato_price cucumber_price : ℝ),
  tomato_price < cucumber_price →
  cucumber_price = 5 →
  2 * tomato_price + 3 * cucumber_price = 23 →
  (cucumber_price - tomato_price) / cucumber_price = 0.2 :=
by
  sorry

end tomato_cucumber_price_difference_l4138_413892


namespace parallel_lines_m_equals_one_l4138_413803

/-- Two lines are parallel if their slopes are equal -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

/-- Line l₁ with equation x + (1+m)y = 2 - m -/
def l₁ (m : ℝ) (x y : ℝ) : Prop :=
  x + (1+m)*y = 2 - m

/-- Line l₂ with equation 2mx + 4y + 16 = 0 -/
def l₂ (m : ℝ) (x y : ℝ) : Prop :=
  2*m*x + 4*y + 16 = 0

theorem parallel_lines_m_equals_one :
  ∀ m : ℝ, parallel 1 (1+m) (2*m) 4 → m = 1 :=
by sorry

end parallel_lines_m_equals_one_l4138_413803


namespace complement_P_intersect_Q_l4138_413811

def P : Set ℝ := {x | x ≤ 0 ∨ x > 3}
def Q : Set ℝ := {0, 1, 2, 3}

theorem complement_P_intersect_Q :
  (Set.compl P) ∩ Q = {1, 2, 3} := by sorry

end complement_P_intersect_Q_l4138_413811


namespace max_value_quadratic_l4138_413800

/-- The function f(x) = -9x^2 + 27x + 15 has a maximum value of 141/4. -/
theorem max_value_quadratic : ∃ (M : ℝ), M = (141 : ℝ) / 4 ∧ 
  ∀ (x : ℝ), -9 * x^2 + 27 * x + 15 ≤ M :=
by sorry

end max_value_quadratic_l4138_413800


namespace ones_digit_of_prime_arithmetic_sequence_l4138_413856

theorem ones_digit_of_prime_arithmetic_sequence (a b c d : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧  -- Four prime numbers
  a > 5 ∧                                  -- a is greater than 5
  b = a + 6 ∧ c = b + 6 ∧ d = c + 6 ∧      -- Arithmetic sequence with common difference 6
  a < b ∧ b < c ∧ c < d →                  -- Increasing sequence
  a % 10 = 1 :=                            -- The ones digit of a is 1
by sorry

end ones_digit_of_prime_arithmetic_sequence_l4138_413856


namespace product_of_roots_quartic_l4138_413888

theorem product_of_roots_quartic (p q r s : ℂ) : 
  (3 * p^4 - 8 * p^3 + p^2 - 10 * p - 24 = 0) →
  (3 * q^4 - 8 * q^3 + q^2 - 10 * q - 24 = 0) →
  (3 * r^4 - 8 * r^3 + r^2 - 10 * r - 24 = 0) →
  (3 * s^4 - 8 * s^3 + s^2 - 10 * s - 24 = 0) →
  p * q * r * s = -8 := by
sorry

end product_of_roots_quartic_l4138_413888


namespace greatest_four_digit_multiple_of_17_l4138_413877

theorem greatest_four_digit_multiple_of_17 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 17 ∣ n → n ≤ 9996 ∧ 17 ∣ 9996 := by
  sorry

end greatest_four_digit_multiple_of_17_l4138_413877

import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_l179_17921

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a point in Cartesian coordinates -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- The polar equation of the curve -/
def polarEquation (p : PolarPoint) : Prop :=
  p.ρ^2 * Real.cos (2 * p.θ) - 2 * p.ρ * Real.cos p.θ = 1

/-- Conversion from polar to Cartesian coordinates -/
noncomputable def polarToCartesian (p : PolarPoint) : CartesianPoint :=
  { x := p.ρ * Real.cos p.θ
    y := p.ρ * Real.sin p.θ }

/-- Definition of a hyperbola in Cartesian coordinates -/
def isHyperbola (f : CartesianPoint → Prop) : Prop :=
  ∃ a b h k : ℝ, ∀ p : CartesianPoint,
    f p ↔ (p.x - h)^2 / a^2 - (p.y - k)^2 / b^2 = 1

/-- The main theorem: The curve is a hyperbola -/
theorem curve_is_hyperbola :
  isHyperbola (fun c => ∃ p : PolarPoint, polarToCartesian p = c ∧ polarEquation p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_l179_17921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_ratio_l179_17984

theorem age_difference_ratio : ∀ (R J K : ℕ),
  R = J + 6 →
  R + 4 = 2 * (J + 4) →
  (R + 4) * (K + 4) = 108 →
  (R - J : ℚ) / (R - K) = 2 :=
λ R J K h1 h2 h3 =>
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_ratio_l179_17984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l179_17923

theorem sequence_limit (a b : ℝ) : 
  ∃ (x : ℕ → ℝ) (L : ℝ),
    x 1 = a ∧
    x 2 = b ∧
    (∀ n ≥ 3, x n = (x (n - 1) + x (n - 2)) / 2) ∧
    L = (a + 2 * b) / 3 ∧
    ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - L| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l179_17923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_ten_l179_17978

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 : ℝ)^x - (2 : ℝ)^(-x) * Real.log a

-- State the theorem
theorem odd_function_implies_a_equals_ten (a : ℝ) :
  (∀ x, f a x = -(f a (-x))) → a = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_ten_l179_17978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l179_17955

theorem remainder_problem (x : ℕ) (h : (7 * x) % 31 = 1) :
  (13 + x) % 31 = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l179_17955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_velocity_at_3s_l179_17975

/-- The motion equation of a particle, where S is displacement in meters and t is time in seconds -/
noncomputable def S (t : ℝ) : ℝ := 1 / t^2

/-- The instantaneous velocity of the particle at time t -/
noncomputable def instantaneous_velocity (t : ℝ) : ℝ := deriv S t

theorem particle_velocity_at_3s :
  instantaneous_velocity 3 = -2/27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_velocity_at_3s_l179_17975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l179_17925

/-- Represents the floor dimensions -/
def floor_length : ℝ := 12
def floor_width : ℝ := 15

/-- Represents the tile dimensions -/
def tile_size : ℝ := 2

/-- Represents the radius of each quarter circle on a tile -/
def circle_radius : ℝ := 1

/-- Calculates the total shaded area of the floor -/
noncomputable def total_shaded_area : ℝ :=
  let num_tiles : ℝ := (floor_length / tile_size) * (floor_width / tile_size)
  let tile_area : ℝ := tile_size * tile_size
  let white_area_per_tile : ℝ := Real.pi * circle_radius^2
  let shaded_area_per_tile : ℝ := tile_area - white_area_per_tile
  num_tiles * shaded_area_per_tile

/-- Theorem stating the total shaded area of the floor -/
theorem shaded_area_calculation :
  total_shaded_area = 180 - 45 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l179_17925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l179_17996

theorem triangle_side_calculation (a b c A B C : ℝ) : 
  a = 5 → A = π/4 → Real.cos B = 3/5 → b = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l179_17996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_220_l179_17917

-- Define the given parameters
def train_length : ℚ := 155
def train_speed_kmh : ℚ := 45
def crossing_time : ℚ := 30

-- Define the function to calculate bridge length
def calculate_bridge_length (train_length speed_kmh crossing_time : ℚ) : ℚ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  let total_distance := speed_ms * crossing_time
  total_distance - train_length

-- State the theorem
theorem bridge_length_is_220 :
  calculate_bridge_length train_length train_speed_kmh crossing_time = 220 := by
  -- Unfold the definition of calculate_bridge_length
  unfold calculate_bridge_length
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

#eval calculate_bridge_length train_length train_speed_kmh crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_220_l179_17917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_regions_area_l179_17953

noncomputable section

/-- The side length of the equilateral triangle -/
def side_length : ℝ := 10

/-- The radius of the larger circle circumscribing the equilateral triangle -/
noncomputable def R : ℝ := side_length / Real.sqrt 3

/-- The radius of the smaller circle inscribed in the equilateral triangle -/
noncomputable def r : ℝ := side_length * Real.sqrt 3 / 6

/-- The area of one shaded region -/
noncomputable def shaded_area : ℝ := Real.pi * R^2 / 6 - Real.pi * r^2 / 6

theorem shaded_regions_area : 3 * shaded_area = 25 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_regions_area_l179_17953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l179_17914

def A (a : ℝ) : Set ℝ := {x : ℝ | 6 * x + a > 0}

theorem range_of_a (a : ℝ) : 1 ∉ A a → a ∈ Set.Iic (-6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l179_17914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l179_17971

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - 2 * x) + Real.log (1 + 3 * x)

def domain_f : Set ℝ := {x | x > -1/3 ∧ x ≠ 1/2}

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l179_17971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_exists_l179_17939

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a^(x-2) + 1

-- State the theorem
theorem fixed_point_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x : ℝ, g a (f a x) = x := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_exists_l179_17939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_five_l179_17932

def sequence_nums : List ℕ := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

theorem product_remainder_mod_five :
  (sequence_nums.prod : ℤ) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_five_l179_17932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_model4_best_fit_l179_17905

-- Define the structure for a regression model
structure RegressionModel where
  r_squared : ℝ

-- Define the criterion for better fit
def better_fit (m1 m2 : RegressionModel) : Prop :=
  m1.r_squared > m2.r_squared

-- Define our four models
def model1 : RegressionModel := ⟨0.25⟩
def model2 : RegressionModel := ⟨0.50⟩
def model3 : RegressionModel := ⟨0.80⟩
def model4 : RegressionModel := ⟨0.98⟩

-- Theorem: Model 4 has the best fit
theorem model4_best_fit :
  better_fit model4 model1 ∧
  better_fit model4 model2 ∧
  better_fit model4 model3 := by
  -- Prove each part of the conjunction
  apply And.intro
  · -- Prove model4 is better than model1
    unfold better_fit
    simp [model4, model1]
    norm_num
  · apply And.intro
    · -- Prove model4 is better than model2
      unfold better_fit
      simp [model4, model2]
      norm_num
    · -- Prove model4 is better than model3
      unfold better_fit
      simp [model4, model3]
      norm_num

#check model4_best_fit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_model4_best_fit_l179_17905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l179_17938

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) * (x^2 - x + 1)

-- State the theorem
theorem derivative_of_f :
  deriv f = λ x ↦ 3 * x^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l179_17938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_cranes_remaining_l179_17968

theorem paper_cranes_remaining (total : ℕ) (alice_fraction : ℚ) (friend_fraction : ℚ) : 
  total = 1000 → 
  alice_fraction = 1/2 → 
  friend_fraction = 1/5 → 
  total - (alice_fraction * ↑total).floor - (friend_fraction * ↑(total - (alice_fraction * ↑total).floor)).floor = 400 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_cranes_remaining_l179_17968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_difference_l179_17976

theorem savings_difference (last_year_salary : ℝ) (savings_rate_last_year : ℝ) 
  (bonus_rate : ℝ) (bonus_tax_rate : ℝ) (bonus_savings_rate : ℝ)
  (salary_increase_rate : ℝ) (savings_rate_this_year : ℝ)
  (part_time_income : ℝ) (part_time_tax_rate : ℝ) (part_time_savings_rate : ℝ) :
  last_year_salary = 45000 →
  savings_rate_last_year = 0.083 →
  bonus_rate = 0.03 →
  bonus_tax_rate = 0.20 →
  bonus_savings_rate = 0.70 →
  salary_increase_rate = 0.115 →
  savings_rate_this_year = 0.056 →
  part_time_income = 3200 →
  part_time_tax_rate = 0.15 →
  part_time_savings_rate = 0.50 →
  let last_year_savings := last_year_salary * savings_rate_last_year + 
    last_year_salary * bonus_rate * (1 - bonus_tax_rate) * bonus_savings_rate
  let this_year_salary := last_year_salary * (1 + salary_increase_rate)
  let this_year_savings := this_year_salary * savings_rate_this_year + 
    part_time_income * (1 - part_time_tax_rate) * part_time_savings_rate
  last_year_savings - this_year_savings = 321.20 := by
  sorry

#check savings_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_difference_l179_17976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_electronic_device_l179_17927

/-- Additional cost function for x < 80 -/
noncomputable def C_less_80 (x : ℝ) : ℝ := (1/2) * x^2 + 40 * x

/-- Additional cost function for x ≥ 80 -/
noncomputable def C_ge_80 (x : ℝ) : ℝ := 101 * x + 8100 / x - 2180

/-- Annual profit function for x < 80 -/
noncomputable def profit_less_80 (x : ℝ) : ℝ := 100 * x - C_less_80 x - 500

/-- Annual profit function for x ≥ 80 -/
noncomputable def profit_ge_80 (x : ℝ) : ℝ := 100 * x - C_ge_80 x - 500

/-- Theorem stating the maximum profit and optimal production quantity -/
theorem max_profit_electronic_device :
  ∃ (x : ℝ), x = 90 ∧ profit_ge_80 x = 1500 ∧
  ∀ (y : ℝ), y > 0 → profit_less_80 y ≤ 1500 ∧ profit_ge_80 y ≤ 1500 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_electronic_device_l179_17927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meet_at_2000_seconds_l179_17960

/-- Represents a cyclist on a circular track -/
structure Cyclist where
  speed : ℚ
  deriving Repr

/-- The problem setup -/
def cyclistProblem : ℚ × List Cyclist :=
  let trackLength : ℚ := 600
  let cyclists : List Cyclist := [⟨3.6⟩, ⟨3.9⟩, ⟨4.2⟩]
  (trackLength, cyclists)

/-- Function to calculate the time when cyclists meet again -/
noncomputable def meetingTime (problem : ℚ × List Cyclist) : ℚ :=
  sorry

/-- Theorem stating that the cyclists meet after 2000 seconds -/
theorem cyclists_meet_at_2000_seconds :
  meetingTime cyclistProblem = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meet_at_2000_seconds_l179_17960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_minimum_l179_17931

/-- The quadratic function y = m²x² - 4x + 1 has a minimum value of -3. -/
theorem quadratic_minimum (m : ℝ) : 
  (∃ (y : ℝ → ℝ), y = (λ x => m^2 * x^2 - 4*x + 1) ∧ 
   ∃ (min_value : ℝ), min_value = -3 ∧ 
   ∀ x, y x ≥ min_value) → 
  m = 1 ∨ m = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_minimum_l179_17931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_BC_isosceles_l179_17987

/-- Triangle with fixed angle and area -/
structure Triangle where
  α : ℝ
  S : ℝ
  b : ℝ
  c : ℝ

/-- The length of side BC in a triangle -/
noncomputable def side_BC_length (t : Triangle) : ℝ :=
  Real.sqrt (t.b^2 + t.c^2 - 2 * t.b * t.c * Real.cos t.α)

/-- Theorem: The length of side BC is minimized when the triangle is isosceles -/
theorem min_side_BC_isosceles (t : Triangle) :
  ∀ t' : Triangle, t'.α = t.α → t'.S = t.S →
  side_BC_length t ≤ side_BC_length t' ↔ t.b = t.c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_BC_isosceles_l179_17987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_largest_and_smallest_angles_l179_17903

-- Define a triangle with sides a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the specific triangle from the problem
noncomputable def specificTriangle : Triangle where
  a := 1
  b := Real.sqrt 5
  c := 2 * Real.sqrt 2
  h_positive := by sorry
  h_triangle := by sorry

-- Function to calculate the angle opposite to a side using the cosine law
noncomputable def cosineAngle (t : Triangle) (side : ℝ) : ℝ :=
  Real.arccos ((t.a^2 + t.b^2 + t.c^2 - 2 * side^2) / (2 * (t.a * t.b * t.c / side)))

-- Theorem statement
theorem sum_of_largest_and_smallest_angles (t : Triangle) (h : t = specificTriangle) :
  let angles := [cosineAngle t t.a, cosineAngle t t.b, cosineAngle t t.c]
  (List.maximum? angles).getD 0 + (List.minimum? angles).getD 0 = 135 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_largest_and_smallest_angles_l179_17903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_inequality_holds_l179_17920

-- Define the function f
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := 2^(x + Real.cos α) - 2^(-x + Real.cos α)

-- Theorem 1
theorem alpha_value (α : ℝ) (h1 : 0 ≤ α) (h2 : α ≤ Real.pi) 
  (h3 : f α 1 = (3 * Real.sqrt 2) / 4) : α = 2 * Real.pi / 3 := by
  sorry

-- Theorem 2
theorem inequality_holds (m : ℝ) (θ : ℝ) (h : m < 1) :
  f (2 * Real.pi / 3) (m * |Real.cos θ|) + f (2 * Real.pi / 3) (1 - m) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_inequality_holds_l179_17920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_minimum_f_inverse_l179_17900

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * cos x * sin (x + π/3) - sqrt 3 * (sin x)^2 + sin x * cos x

-- Theorem for the smallest positive period
theorem f_period : ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') := by
  sorry

-- Theorem for the minimum value and its occurrence
theorem f_minimum : (∃ (k : ℤ), f (k * π - 5*π/12) = -2) ∧ (∀ (x : ℝ), f x ≥ -2) := by
  sorry

-- Theorem for the inverse function value
theorem f_inverse : ∃ (f_inv : ℝ → ℝ), (∀ (x : ℝ), x ∈ Set.Icc (π/12) (7*π/12) → f (f_inv x) = x) ∧ f_inv 1 = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_minimum_f_inverse_l179_17900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l179_17981

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x - Real.pi / 3)

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l179_17981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cassie_height_cm_l179_17970

-- Define the conversion factor
def inch_to_cm : ℚ := 2.54

-- Define Cassie's height in inches
def cassie_height_inches : ℚ := 68

-- Define the function to convert inches to centimeters
def inches_to_cm (inches : ℚ) : ℚ := inches * inch_to_cm

-- Define the function to round to the nearest tenth
noncomputable def round_to_tenth (x : ℚ) : ℚ := 
  ⌊(x * 10 + 1/2)⌋ / 10

-- Theorem statement
theorem cassie_height_cm :
  round_to_tenth (inches_to_cm cassie_height_inches) = 172.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cassie_height_cm_l179_17970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_conditions_has_three_digits_l179_17991

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log n 10 + 1

theorem smallest_n_with_conditions_has_three_digits :
  ∃ n : ℕ,
    (∀ m : ℕ, m < n →
      ¬(is_divisible_by m 24 ∧
        is_perfect_square m ∧
        is_perfect_square (m^2))) ∧
    is_divisible_by n 24 ∧
    is_perfect_square n ∧
    is_perfect_square (n^2) ∧
    num_digits n = 3 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_conditions_has_three_digits_l179_17991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transistors_in_2010_l179_17941

/-- Calculates the number of transistors on a CPU after a given number of years,
    given an initial number of transistors and a doubling period. -/
def transistors_after_years (initial_transistors : ℕ) (doubling_period : ℚ) (years : ℕ) : ℕ :=
  initial_transistors * 2 ^ (((years : ℚ) / doubling_period).floor.toNat)

/-- The number of transistors on a typical CPU in 2010, given the initial conditions from 1990. -/
theorem transistors_in_2010 :
  transistors_after_years 2000000 (3/2) 20 = 16384000000 := by
  sorry

#eval transistors_after_years 2000000 (3/2) 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transistors_in_2010_l179_17941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l179_17961

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then 2^x - (1/2)*x - 1 else -(2^(-x) - (1/2)*(-x) - 1)

theorem f_has_three_zeros :
  (∀ x, f (-x) = -f x) ∧ 
  (∃! a b c, a < b ∧ b < c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) := by
  sorry

#check f_has_three_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l179_17961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_false_l179_17966

-- Define the functions
noncomputable def f1 (x : ℝ) : ℝ := Real.sqrt (x^2)
noncomputable def f2 (x : ℝ) : ℝ := (Real.sqrt x)^2

-- Define the domain of f(x-1)
def domain_f (x : ℝ) : Prop := x ∈ Set.Icc 1 2

-- Define the logarithm function
noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the composite function
noncomputable def g (x : ℝ) : ℝ := log_base_2 (x^2 + 2*x - 3)

-- State the theorem
theorem all_statements_false :
  (∃ x, f1 x ≠ f2 x) ∧
  (∃ f : ℝ → ℝ, (∀ x, domain_f x → f (x-1) ∈ Set.Icc 0 1) ∧
    {x | f (3*x^2) ∈ Set.Icc 0 1} ≠ Set.Icc 0 (Real.sqrt 3 / 3)) ∧
  (∃ x y, x < y ∧ x > 1 ∧ g x ≥ g y) ∧
  (∃ x, x ∈ Set.Ioi (-1) ∧ x ≤ 1 ∧ ¬(∀ y, y > x → g x < g y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_false_l179_17966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_parallelism_assumption_l179_17924

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relation for a line being in a plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the relation for a line being parallel to a plane
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the relation for two lines being parallel
def lines_parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem incorrect_parallelism_assumption :
  ¬(∀ (l : Line) (p : Plane),
    line_parallel_to_plane l p →
    (∀ (l' : Line), line_in_plane l' p → lines_parallel l l')) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_parallelism_assumption_l179_17924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cucumber_tomato_ratio_two_to_one_l179_17934

/-- Represents the garden planting scenario --/
structure GardenPlanting where
  totalRows : ℕ
  tomatoPlantsPerRow : ℕ
  totalTomatoes : ℕ
  tomatoesPerPlant : ℕ

/-- Calculates the ratio of cucumber rows to tomato rows --/
def cucumberToTomatoRowRatio (g : GardenPlanting) : Rat :=
  let tomatoPlants := g.totalTomatoes / g.tomatoesPerPlant
  let tomatoRows := tomatoPlants / g.tomatoPlantsPerRow
  let cucumberRows := g.totalRows - tomatoRows
  cucumberRows / tomatoRows

/-- Theorem stating that the ratio of cucumber rows to tomato rows is 2:1 --/
theorem cucumber_tomato_ratio_two_to_one (g : GardenPlanting) 
  (h1 : g.totalRows = 15)
  (h2 : g.tomatoPlantsPerRow = 8)
  (h3 : g.totalTomatoes = 120)
  (h4 : g.tomatoesPerPlant = 3) :
  cucumberToTomatoRowRatio g = 2 := by
  sorry

#eval cucumberToTomatoRowRatio { totalRows := 15, tomatoPlantsPerRow := 8, totalTomatoes := 120, tomatoesPerPlant := 3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cucumber_tomato_ratio_two_to_one_l179_17934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bound_sequence_a_difference_l179_17963

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 8
  | (n + 3) => sequence_a (n + 1) + (4 / (n + 2 : ℝ)) * sequence_a (n + 2)

theorem sequence_a_bound (n : ℕ) : sequence_a n ≤ 2 * (n^2 : ℝ) := by
  sorry

theorem sequence_a_difference (n : ℕ) : 
  sequence_a (n + 1) - sequence_a n ≤ 4 * n + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bound_sequence_a_difference_l179_17963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l179_17940

def U (a : ℝ) : Set ℝ := {1, 2, a^2 + 2*a - 3}

def A (a : ℝ) : Set ℝ := {|a - 2|, 2}

theorem find_a : 
  ∃ a : ℝ, 
  (U a) \ (A a) = {0} → 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l179_17940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_monotonicity_and_bound_l179_17901

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + log x) / x

-- Theorem statement
theorem extreme_value_and_monotonicity_and_bound :
  -- Part 1: Extreme value at x = 1
  (∃ a : ℝ, ∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 ↔ x = 1) ∧
  -- Part 2: Monotonicity of f
  (let a := 1
   (∀ x : ℝ, 0 < x → x < 1 → (deriv (f a)) x > 0) ∧
   (∀ x : ℝ, x > 1 → (deriv (f a)) x < 0)) ∧
  -- Part 3: Upper bound for m
  (∀ m : ℝ, (∀ x : ℝ, x ≥ 1 → f 1 x ≥ m / (1 + x)) ↔ m ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_monotonicity_and_bound_l179_17901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l179_17964

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A circle centered at the origin -/
structure Circle where
  r : ℝ
  h_positive : 0 < r

theorem ellipse_equation (e : Ellipse) 
  (h_eccentricity : e.eccentricity = 1/2)
  (h_tangent : ∃ (c : Circle), c.r = e.b ∧ 
    (∀ (x y : ℝ), x^2 + y^2 = c.r^2 → x - y + Real.sqrt 6 ≥ 0)) :
  e.a^2 = 4 ∧ e.b^2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l179_17964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proj_w_v_equals_l179_17906

noncomputable def v : ℝ × ℝ := (4, 2)
noncomputable def w : ℝ × ℝ := (10, -5)

noncomputable def proj_vector (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_squared := v.1 * v.1 + v.2 * v.2
  ((dot_product / norm_squared) * v.1, (dot_product / norm_squared) * v.2)

theorem proj_w_v_equals : proj_vector v w = (12/5, -6/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proj_w_v_equals_l179_17906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l179_17930

theorem rectangle_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + 5) * (c + d + 5) = a * c + a * d + b * c + b * d + 5 * a + 5 * b + 5 * c + 5 * d + 25 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l179_17930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_l179_17909

-- Define the angles A and B as variables
variable (A B : ℝ)

-- Define the condition that the sides of angles A and B are parallel
axiom parallel_sides : (A = B) ∨ (A + B = 180)

-- Define the given equation
axiom given_equation : 3 * A - B = 80

-- Theorem to prove
theorem angle_B_value : B = 40 ∨ B = 115 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_l179_17909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_intersection_point_l179_17910

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
structure GeometricSetup where
  O : Circle
  A : ℝ × ℝ
  is_outside : ‖A - O.center‖ > O.radius

-- Define a line through A intersecting the circle
def intersecting_line (setup : GeometricSetup) : Type :=
  { l : Set (ℝ × ℝ) // 
    ∃ (B C : ℝ × ℝ), 
      A ∈ l ∧ B ∈ l ∧ C ∈ l ∧
      ‖B - setup.O.center‖ = setup.O.radius ∧
      ‖C - setup.O.center‖ = setup.O.radius ∧
      ∃ t : ℝ, 0 < t ∧ t < 1 ∧ B = (1 - t) • setup.A + t • C }
  where A := setup.A

-- Define the symmetric line
def symmetric_line (setup : GeometricSetup) (l : intersecting_line setup) : 
  { l' : Set (ℝ × ℝ) // 
    ∃ (D E : ℝ × ℝ), 
      setup.A ∈ l' ∧ D ∈ l' ∧ E ∈ l' ∧
      ‖D - setup.O.center‖ = setup.O.radius ∧
      ‖E - setup.O.center‖ = setup.O.radius ∧
      ∃ t : ℝ, 0 < t ∧ t < 1 ∧ E = (1 - t) • setup.A + t • D } :=
  sorry

-- Define the intersection point of diagonals
noncomputable def intersection_point (setup : GeometricSetup) (l : intersecting_line setup) : ℝ × ℝ :=
  sorry

-- The main theorem
theorem fixed_intersection_point (setup : GeometricSetup) :
  ∀ (l : intersecting_line setup), 
    let P := intersection_point setup l
    ∃ (t : ℝ), P = (1 - t) • setup.A + t • setup.O.center ∧
                 t = (‖setup.A - setup.O.center‖^2 - setup.O.radius^2) / ‖setup.A - setup.O.center‖^2 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_intersection_point_l179_17910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l179_17957

/-- Given a point P and the midpoint M of line segment PQ, 
    prove that if vector PQ is collinear with vector a = (lambda, 1),
    then lambda = -2/3 -/
theorem collinear_vectors (P M Q : ℝ × ℝ) (lambda : ℝ) : 
  P = (-1, 2) →
  M = (1, -1) →
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  ∃ (k : ℝ), k ≠ 0 ∧ (Q.1 - P.1, Q.2 - P.2) = (k * lambda, k * 1) →
  lambda = -2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l179_17957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_painting_possibilities_l179_17928

theorem floor_painting_possibilities :
  let count := Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    b > a ∧ 
    a * b - 8 * a - 8 * b + 32 = 0) (Finset.range 41 ×ˢ Finset.range 41)
  Finset.card count = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_painting_possibilities_l179_17928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_interest_rate_l179_17983

/-- Calculates simple interest -/
noncomputable def simpleInterest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem lending_interest_rate 
  (borrowed_amount : ℝ)
  (borrowed_rate : ℝ)
  (time : ℝ)
  (yearly_gain : ℝ)
  (h1 : borrowed_amount = 8000)
  (h2 : borrowed_rate = 4)
  (h3 : time = 2)
  (h4 : yearly_gain = 160) :
  let borrowed_interest := simpleInterest borrowed_amount borrowed_rate time
  let total_gain := yearly_gain * time
  let total_interest_earned := borrowed_interest + total_gain
  let lending_rate := (total_interest_earned * 100) / (borrowed_amount * time)
  lending_rate = 6 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_interest_rate_l179_17983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_l179_17969

def sum_series (n : ℕ) : ℕ := (n - 1) * 2^n + 1

theorem perfect_square_sum (n : ℕ) : 
  (∃ k : ℕ, sum_series n = k^2) ↔ n = 1 ∨ n = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_l179_17969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_knights_in_village_l179_17952

theorem min_knights_in_village (total_residents total_statements liar_statements : ℕ)
  (h1 : total_residents = 7)
  (h2 : total_statements = total_residents * (total_residents - 1))
  (h3 : total_statements = 42)
  (h4 : liar_statements = 24) :
  ∃ k : ℕ, k ≥ 3 ∧ 
  (∀ n : ℕ, n < k → ¬(n * (total_residents - n) = liar_statements / 2)) ∧
  k * (total_residents - k) = liar_statements / 2 :=
by
  -- The proof goes here
  sorry

#check min_knights_in_village

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_knights_in_village_l179_17952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_price_increase_is_200_percent_l179_17994

/-- Represents the price change of coffee and milk powder from June to July -/
structure PriceChange where
  june_price : ℝ  -- Price of both coffee and milk powder in June
  july_milk_price : ℝ  -- Price of milk powder in July
  july_mixture_price : ℝ  -- Price of the mixture in July
  mixture_weight : ℝ  -- Weight of the mixture in pounds

/-- Calculates the percentage increase in coffee price -/
noncomputable def coffee_price_increase (pc : PriceChange) : ℝ :=
  let july_coffee_price := (2 * pc.july_mixture_price - pc.july_milk_price * pc.mixture_weight) / pc.mixture_weight
  (july_coffee_price / pc.june_price - 1) * 100

/-- Theorem stating that the coffee price increase is 200% -/
theorem coffee_price_increase_is_200_percent (pc : PriceChange) :
  pc.june_price > 0 ∧
  pc.july_milk_price = 0.4 ∧
  pc.july_mixture_price = 5.1 ∧
  pc.mixture_weight = 3 ∧
  pc.july_milk_price = 0.4 * pc.june_price →
  coffee_price_increase pc = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_price_increase_is_200_percent_l179_17994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_square_distance_l179_17918

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line
def line (m : ℝ) (x y : ℝ) : Prop :=
  y = (1/2) * x + m

-- Define the theorem
theorem ellipse_square_distance (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∀ m : ℝ, -Real.sqrt 2 < m ∧ m < Real.sqrt 2 →
  ∃ A C : ℝ × ℝ,
    ellipse a b A.1 A.2 ∧
    ellipse a b C.1 C.2 ∧
    line m A.1 A.2 ∧
    line m C.1 C.2 ∧
    ellipse a 1 0 1 ∧
    (a^2 - b^2) / a^2 = 3/4 →
    let B := ((A.1 + C.1 + A.2 - C.2) / 2, (A.2 + C.2 - A.1 + C.1) / 2)
    let N := (-2*m, 0)
    (B.1 - N.1)^2 + (B.2 - N.2)^2 = 5/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_square_distance_l179_17918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l179_17997

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

noncomputable def tangent_line (t : ℝ) (x y : ℝ) : Prop := x + Real.exp t * y = t + 1

noncomputable def triangle_area (t : ℝ) : ℝ := (t + 1)^2 / (2 * Real.exp t)

theorem max_triangle_area :
  ∀ t : ℝ, t ≥ 0 → triangle_area t ≤ 2 / Real.exp 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l179_17997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_exposed_surface_area_l179_17965

-- Define the structure for a rectangular solid
structure RectangularSolid where
  volume : ℝ
  face_area1 : ℝ
  face_area2 : ℝ

-- Define the three solids
noncomputable def solid1 : RectangularSolid := { volume := 128, face_area1 := 4, face_area2 := 32 }
noncomputable def solid2 : RectangularSolid := { volume := 128, face_area1 := 64, face_area2 := 16 }
noncomputable def solid3 : RectangularSolid := { volume := 128, face_area1 := 8, face_area2 := 32 }

-- Function to calculate the third face area
noncomputable def third_face_area (s : RectangularSolid) : ℝ :=
  s.volume^2 / (s.face_area1 * s.face_area2)

-- Function to calculate the side lengths of a solid
noncomputable def side_lengths (s : RectangularSolid) : (ℝ × ℝ × ℝ) :=
  (s.volume / s.face_area1, s.volume / s.face_area2, s.volume / (third_face_area s))

-- Function to calculate the lateral surface area of a solid
noncomputable def lateral_surface_area (s : RectangularSolid) : ℝ :=
  let (x, y, z) := side_lengths s
  2 * (x * y + y * z)

-- Theorem stating the minimum exposed surface area
theorem min_exposed_surface_area :
  lateral_surface_area solid1 + lateral_surface_area solid2 + lateral_surface_area solid3 +
  min (third_face_area solid1) (min (third_face_area solid2) (third_face_area solid3)) +
  min solid1.face_area1 (min solid1.face_area2 (min solid2.face_area1 (min solid2.face_area2 (min solid3.face_area1 solid3.face_area2)))) = 688 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_exposed_surface_area_l179_17965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_l179_17951

theorem triangle_side_ratio (A B C : ℝ) (a b c : ℝ) :
  A > 0 → B > 0 → C > 0 →
  a > 0 → b > 0 → c > 0 →
  A + B + C = π →
  A = B ∧ C = 2*A →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  a = b ∧ c = Real.sqrt 2 * a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_l179_17951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_l179_17949

/-- The function f(x) = ax³ + x² has an extreme value at x = -4/3 if and only if a = 1/2 -/
theorem extreme_value_condition (a : ℝ) : 
  (∃ f : ℝ → ℝ, f = (λ x ↦ a * x^3 + x^2) ∧ 
   ∃ε > 0, ∀ x ∈ Set.Ioo (-4/3 - ε) (-4/3 + ε), f x ≤ f (-4/3) ∨ f x ≥ f (-4/3)) ↔ 
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_l179_17949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_k_l179_17904

def k : ℕ := 10^45 - 46

theorem sum_of_digits_k : (Nat.digits 10 k).sum = 414 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_k_l179_17904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_ln2_l179_17948

open Real MeasureTheory

theorem integral_equals_ln2 :
  ∫ x in Set.Icc π (2*π), (x + cos x) / (x^2 + 2 * sin x) = log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_ln2_l179_17948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positivist_power_implies_positivist_l179_17985

/-- A polynomial is positivist if it can be written as a product of two non-constant polynomials
    with non-negative real coefficients. -/
def IsPositivist (p : Polynomial ℝ) : Prop :=
  ∃ (q r : Polynomial ℝ), p = q * r ∧ q ≠ 0 ∧ r ≠ 0 ∧
    (∀ i, q.coeff i ≥ 0) ∧ (∀ i, r.coeff i ≥ 0)

/-- Given a polynomial f(x) of degree greater than one such that f(x^n) is positivist
    for some positive integer n, prove that f(x) is positivist. -/
theorem positivist_power_implies_positivist
  (f : Polynomial ℝ) (n : ℕ+) (h_deg : f.degree > 1)
  (h_pos : IsPositivist (f.comp (Polynomial.X ^ n.val))) :
  IsPositivist f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positivist_power_implies_positivist_l179_17985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l179_17977

/-- The ellipse C defined by x = cos θ and y = 2 sin θ (θ ∈ ℝ) -/
noncomputable def ellipse_C (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 2 * Real.sin θ)

/-- A point (m, 1/2) lies on the ellipse C -/
def point_on_ellipse (m : ℝ) : Prop :=
  ∃ θ : ℝ, ellipse_C θ = (m, 1/2)

/-- The eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_properties (m : ℝ) (h : point_on_ellipse m) :
  (m = Real.sqrt 15 / 4 ∨ m = -Real.sqrt 15 / 4) ∧
  eccentricity 2 1 = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l179_17977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_length_one_solution_l179_17959

-- Define the vectors a and b
noncomputable def a (x : Real) : Fin 2 → Real := ![Real.cos (3*x/2), Real.sin (3*x/2)]
noncomputable def b (x : Real) : Fin 2 → Real := ![Real.cos (x/2), -Real.sin (x/2)]

-- Define the theorem
theorem vector_sum_length_one_solution (x : Real) 
  (h1 : x ∈ Set.Icc 0 Real.pi) 
  (h2 : Real.sqrt ((a x 0 + b x 0)^2 + (a x 1 + b x 1)^2) = 1) : 
  x = Real.pi/3 ∨ x = 2*Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_length_one_solution_l179_17959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equation_solvability_l179_17999

-- Define the integral equation
def integral_equation (φ : ℝ → ℝ) (α β : ℝ) : Prop :=
  ∀ x, φ x = 4 * ∫ t in Set.Icc 0 1, x * t^2 * φ t + α * x + β

-- Define the solvability condition
def solvability_condition (α β : ℝ) : Prop :=
  3 * α + 4 * β = 0

-- State the theorem
theorem integral_equation_solvability (α β : ℝ) :
  (∃ φ : ℝ → ℝ, integral_equation φ α β) ↔ solvability_condition α β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equation_solvability_l179_17999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_all_quadrants_l179_17922

/-- The cubic function f(x) defined with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 + (1/2) * a * x^2 - 2 * a * x + 2 * a + 1

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 2 * a

/-- The condition for f(x) to pass through all four quadrants -/
def passes_through_all_quadrants (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ : ℝ, 
    f a x₁ > 0 ∧ f a x₂ < 0 ∧ f a x₃ > 0 ∧ f a x₄ < 0 ∧
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄

/-- The theorem stating the range of a for which f(x) passes through all four quadrants -/
theorem a_range_for_all_quadrants :
  ∀ a : ℝ, passes_through_all_quadrants a ↔ -6/5 < a ∧ a < -3/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_all_quadrants_l179_17922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l179_17937

/-- A line passing through a point with a given inclination angle -/
structure Line where
  point : ℝ × ℝ
  angle : ℝ

/-- A circle with center at origin -/
structure Circle where
  radius : ℝ

/-- The intersection points of a line and a circle -/
def intersection (l : Line) (c : Circle) : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = c.radius^2 ∧ 
       ∃ t : ℝ, p.1 = l.point.1 + t * Real.cos l.angle ∧
              p.2 = l.point.2 + t * Real.sin l.angle}

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_product (l : Line) (c : Circle) :
  l.point = (1, 1) →
  l.angle = π/6 →
  c.radius^2 = 4 →
  let points := intersection l c
  ∃ A B : ℝ × ℝ, A ∈ points ∧ B ∈ points ∧ A ≠ B ∧
    (distance l.point A) * (distance l.point B) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l179_17937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l179_17958

/-- Sequence of points on the positive y-axis -/
noncomputable def A (n : ℕ) : ℝ × ℝ := (0, 1 / n)

/-- Sequence of points on the curve y = √(2x) -/
noncomputable def B (n : ℕ) : ℝ × ℝ := 
  let b := Real.sqrt ((1 / n)^2 + 1) - 1
  (b, Real.sqrt (2 * b))

/-- x-intercept of line A_nB_n -/
noncomputable def a (n : ℕ) : ℝ :=
  let b := (B n).1
  b / (1 - n * Real.sqrt (2 * b))

/-- x-coordinate of point B_n -/
noncomputable def b (n : ℕ) : ℝ := (B n).1

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → a n > a (n + 1) ∧ a (n + 1) > 4) ∧
  (∃ n₀ : ℕ, ∀ n > n₀, 
    (Finset.range n).sum (λ i => b (i + 2) / b (i + 1)) + b (n + 1) / b n < n - 2004) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l179_17958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l179_17907

theorem power_difference (x y : ℝ) (h1 : (10 : ℝ)^x = 3) (h2 : (10 : ℝ)^y = 4) : 
  (10 : ℝ)^(x-2*y) = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l179_17907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_scrumptious_l179_17995

def IsScrumptious (n : Int) : Prop :=
  ∃ (a b : Int), a ≤ n ∧ n ≤ b ∧ (∀ k, a ≤ k ∧ k ≤ b → k ∈ Finset.Icc a b) ∧
    (Finset.sum (Finset.Icc a b) id) = 2021

theorem smallest_scrumptious : 
  IsScrumptious (-2020) ∧ ∀ m : Int, m < -2020 → ¬IsScrumptious m := by
  sorry

#check smallest_scrumptious

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_scrumptious_l179_17995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l179_17982

-- Define the function g
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := 1 / (3 * x + b)

-- Define the inverse function g^(-1)
noncomputable def g_inv (x : ℝ) : ℝ := (1 - 3 * x) / (3 * x)

-- Theorem statement
theorem inverse_function_condition (b : ℝ) :
  (∀ x, g_inv (g b x) = x) → b = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l179_17982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivatives_of_f_and_g_l179_17992

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.tan x

-- State the theorem
theorem derivatives_of_f_and_g :
  (∀ x, HasDerivAt f (2 * x * Real.sin x + x^2 * Real.cos x) x) ∧
  (∀ x, HasDerivAt g (1 / (Real.cos x)^2) x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivatives_of_f_and_g_l179_17992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_cubic_function_m_bound_l179_17935

/-- If f(x) = -x³ + x² + mx + m is an increasing function on (-1, 1), then m ≥ 5 -/
theorem increasing_cubic_function_m_bound 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ x, f x = -x^3 + x^2 + m*x + m) 
  (h2 : StrictMonoOn f (Set.Ioo (-1) 1)) : 
  m ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_cubic_function_m_bound_l179_17935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reasoning_methods_properties_l179_17919

-- Define the set of statement indices
def StatementIndex : Type := Fin 5

-- Define the properties of reasoning methods
def is_cause_and_effect (method : String) : Prop := sorry
def is_forward_reasoning (method : String) : Prop := sorry
def is_cause_seeking_from_effect (method : String) : Prop := sorry
def is_indirect_proof (method : String) : Prop := sorry
def is_reverse_reasoning (method : String) : Prop := sorry

-- Define the reasoning methods
def synthetic_method : String := "synthetic"
def analytical_method : String := "analytical"
def contradiction_method : String := "contradiction"

-- Define the set of correct statements
def correct_statements : Set (Fin 5) := {0, 1, 2}

-- Theorem statement
theorem reasoning_methods_properties :
  (is_cause_and_effect synthetic_method) ∧
  (is_forward_reasoning synthetic_method) ∧
  (is_cause_seeking_from_effect analytical_method) ∧
  (¬ is_indirect_proof analytical_method) ∧
  (¬ is_reverse_reasoning contradiction_method) →
  correct_statements = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reasoning_methods_properties_l179_17919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_not_power_of_three_sum_l179_17954

theorem max_subset_size_not_power_of_three_sum : ∃ (T : Finset ℕ), 
  (∀ x, x ∈ T → x ≥ 1 ∧ x ≤ 242) ∧ 
  (∀ a b, a ∈ T → b ∈ T → ¬ ∃ n : ℕ, a + b = 3^n) ∧
  T.card = 121 ∧
  (∀ S : Finset ℕ, (∀ x, x ∈ S → x ≥ 1 ∧ x ≤ 242) → 
    (∀ a b, a ∈ S → b ∈ S → ¬ ∃ n : ℕ, a + b = 3^n) → S.card ≤ 121) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_not_power_of_three_sum_l179_17954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l179_17967

/-- The distance from a point to a line -/
noncomputable def distancePointToLine (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- Theorem: If the distance from the focus of the parabola y² = 2px (p > 0) 
    to the line y = x + 1 is √2, then p = 2 -/
theorem parabola_focus_distance (p : ℝ) (hp : p > 0) :
  distancePointToLine (p/2) 0 (-1) 1 (-1) = Real.sqrt 2 → p = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l179_17967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l179_17945

/-- Arithmetic sequence with common difference d and first term d -/
def arithmetic_sequence (d : ℚ) (n : ℕ) : ℚ := n * d

/-- Sum of first n terms of the sequence {a_n²} -/
def S (d : ℚ) (n : ℕ) : ℚ := (n * (n + 1) * (2 * n + 1) / 6) * d^2

/-- Geometric sequence with common ratio q and first term d² -/
def geometric_sequence (d q : ℚ) (n : ℕ) : ℚ := d^2 * q^(n - 1)

/-- Sum of first n terms of the geometric sequence -/
def T (d q : ℚ) (n : ℕ) : ℚ := d^2 * (1 - q^n) / (1 - q)

theorem arithmetic_geometric_sequence_ratio (d q : ℚ) :
  d ≠ 0 →
  q < 1 →
  q > 0 →
  (∀ n : ℕ, ∃ k : ℚ, geometric_sequence d q n = k * Real.sin k) →
  (∃ m : ℕ, S d 3 / T d q 3 = m) →
  q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l179_17945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l179_17926

theorem angle_sum_proof (α β : ℝ) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.sin α = 2 * Real.sqrt 5 / 5 →
  Real.sin β = 3 * Real.sqrt 10 / 10 →
  α + β = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l179_17926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sin_property_l179_17915

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_property (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 1 + a 7 + a 13 = 4 * Real.pi) : 
  Real.sin (a 2 + a 12) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sin_property_l179_17915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_of_8_eq_5_l179_17986

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ+) : ℕ := sumOfDigits ((n : ℕ)^2 + 1)

/-- Recursive definition of fₖ -/
def f_k : ℕ → ℕ+ → ℕ
  | 0, n => (n : ℕ)
  | k+1, n => f ⟨f_k k n, sorry⟩

/-- Main theorem to prove -/
theorem f_2018_of_8_eq_5 : f_k 2018 8 = 5 := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_of_8_eq_5_l179_17986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l179_17933

-- Define a, b, and c as noncomputable real numbers
noncomputable def a : ℝ := Real.rpow 2 0.6
noncomputable def b : ℝ := Real.rpow 4 0.4
noncomputable def c : ℝ := Real.rpow 3 0.8

-- State the theorem
theorem problem_statement :
  a < b ∧ a * b < c^2 ∧ b^2 < a * c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l179_17933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_op_with_i_l179_17950

-- Define the operation
def matrix_op (a b c d : ℂ) : ℂ := a * d - b * c

-- State the theorem
theorem matrix_op_with_i : 
  matrix_op Complex.I 1 2 Complex.I * Complex.I = -3 * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_op_with_i_l179_17950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athena_fruit_drinks_l179_17912

-- Define the problem parameters
def sandwiches : ℕ := 3
def sandwich_price : ℚ := 3
def fruit_drink_price : ℚ := 5/2
def total_spent : ℚ := 14

-- Define the function to calculate the number of fruit drinks
noncomputable def num_fruit_drinks : ℚ := 
  (total_spent - (sandwiches : ℚ) * sandwich_price) / fruit_drink_price

-- Theorem statement
theorem athena_fruit_drinks : num_fruit_drinks = 2 := by
  -- Unfold the definition of num_fruit_drinks
  unfold num_fruit_drinks
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_athena_fruit_drinks_l179_17912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l179_17943

/-- The function f(x) defined as ln(ax+1) + x³ - x² - ax --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + x^3 - x^2 - a * x

/-- The statement that f is increasing on [2, +∞) --/
def f_increasing (a : ℝ) : Prop :=
  ∀ x y, 2 ≤ x → x < y → f a x < f a y

/-- The main theorem: f is increasing on [2, +∞) iff a ∈ [0, 4+2√5] --/
theorem f_increasing_iff_a_in_range :
  ∀ a : ℝ, f_increasing a ↔ 0 ≤ a ∧ a ≤ 4 + 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l179_17943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_is_37_l179_17973

def is_valid_arrangement (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ∈ ({1, 2, 3, 4} : Set ℕ) ∧ b ∈ ({1, 2, 3, 4} : Set ℕ) ∧ 
  c ∈ ({1, 2, 3, 4} : Set ℕ) ∧ d ∈ ({1, 2, 3, 4} : Set ℕ)

def sum_of_arrangement (a b c d : ℕ) : ℕ :=
  10 * a + b + 10 * c + d

theorem smallest_sum_is_37 :
  ∀ a b c d : ℕ, is_valid_arrangement a b c d →
  sum_of_arrangement a b c d ≥ 37 ∧
  ∃ w x y z : ℕ, is_valid_arrangement w x y z ∧ sum_of_arrangement w x y z = 37 :=
by sorry

#check smallest_sum_is_37

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_is_37_l179_17973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salad_fries_ratio_is_three_l179_17902

/-- Represents the prices of items at McDonald's --/
structure McdonaldsPrices where
  total : ℚ
  fries : ℚ
  burger : ℚ

/-- Calculates the ratio of salad price to fries price --/
def saladToFriesRatio (p : McdonaldsPrices) : ℚ :=
  (p.total - p.burger - 2 * p.fries) / p.fries

/-- Theorem stating that the ratio of salad price to fries price is 3 --/
theorem salad_fries_ratio_is_three (p : McdonaldsPrices) 
  (h1 : p.total = 15)
  (h2 : p.fries = 2)
  (h3 : p.burger = 5) :
  saladToFriesRatio p = 3 := by
  sorry

#check salad_fries_ratio_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salad_fries_ratio_is_three_l179_17902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_match_result_l179_17993

/-- Represents a soccer team --/
inductive Team
| North
| South

/-- Represents the result of a soccer match --/
structure MatchResult where
  northGoals : Nat
  southGoals : Nat

/-- Represents a prediction about the match --/
inductive Prediction
| NoDraw
| SouthConcedes
| NorthWins
| NorthNotLose
| ExactlyThreeGoals

/-- Checks if a prediction is correct given a match result --/
def isPredictionCorrect (p : Prediction) (result : MatchResult) : Bool :=
  match p with
  | Prediction.NoDraw => result.northGoals ≠ result.southGoals
  | Prediction.SouthConcedes => result.northGoals > 0
  | Prediction.NorthWins => result.northGoals > result.southGoals
  | Prediction.NorthNotLose => result.northGoals ≥ result.southGoals
  | Prediction.ExactlyThreeGoals => result.northGoals + result.southGoals = 3

/-- Counts the number of correct predictions for a given match result --/
def countPredictions (result : MatchResult) : Nat :=
  let predictions := [Prediction.NoDraw, Prediction.SouthConcedes, Prediction.NorthWins,
                      Prediction.NorthNotLose, Prediction.ExactlyThreeGoals]
  predictions.countP (λ p => isPredictionCorrect p result)

/-- The main theorem to prove --/
theorem soccer_match_result :
  ∃! result : MatchResult,
    (countPredictions result = 3) ∧
    (result.northGoals = 2 ∧ result.southGoals = 1) :=
by sorry

#eval countPredictions { northGoals := 2, southGoals := 1 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_match_result_l179_17993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_for_smaller_x_l179_17936

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - Real.log x / Real.log 3

theorem f_positive_for_smaller_x (x₀ x₁ : ℝ) 
  (h₀ : f x₀ = 0) 
  (h₁ : 0 < x₁) 
  (h₂ : x₁ < x₀) : 
  f x₁ > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_for_smaller_x_l179_17936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l179_17989

/-- Represents a convex pentagon in 2D space -/
def ConvexPentagon (A B C D E : ℝ × ℝ) : Prop := sorry

/-- Calculates the area of a triangle given its three vertices -/
def AreaTriangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Calculates the area of a pentagon given its five vertices -/
def AreaPentagon (A B C D E : ℝ × ℝ) : ℝ := sorry

/-- Given a convex pentagon ABCDE where the areas of triangles ABC, BCD, CDE, DEA, and EAB are all equal to 1, 
    the area of pentagon ABCDE is (5 + √5) / 2. -/
theorem pentagon_area (A B C D E : ℝ × ℝ) 
  (h_convex : ConvexPentagon A B C D E)
  (h_abc : AreaTriangle A B C = 1)
  (h_bcd : AreaTriangle B C D = 1)
  (h_cde : AreaTriangle C D E = 1)
  (h_dea : AreaTriangle D E A = 1)
  (h_eab : AreaTriangle E A B = 1) :
  AreaPentagon A B C D E = (5 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l179_17989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_prob_after_three_turns_l179_17998

/-- Represents the player who has the ball -/
inductive Player
  | Alice
  | Bob

/-- The game state after each turn -/
structure GameState where
  turn : ℕ
  player : Player

/-- The probability of Alice having the ball after three turns -/
def prob_alice_after_three_turns : ℚ := 5/24

/-- The transition probability from one player to another -/
def transition_prob (f t : Player) : ℚ :=
  match f, t with
  | Player.Alice, Player.Bob => 1/2
  | Player.Alice, Player.Alice => 1/2
  | Player.Bob, Player.Alice => 1/3
  | Player.Bob, Player.Bob => 2/3

/-- The initial state of the game -/
def initial_state : GameState :=
  { turn := 0, player := Player.Alice }

/-- The probability of reaching a given state after n turns -/
noncomputable def prob_reach_state (n : ℕ) (player : Player) : ℚ :=
  sorry

theorem alice_prob_after_three_turns :
  prob_reach_state 3 Player.Alice = prob_alice_after_three_turns := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_prob_after_three_turns_l179_17998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_density_properties_l179_17944

/-- The normal distribution density function -/
noncomputable def normal_density (μ σ x : ℝ) : ℝ :=
  1 / (σ * Real.sqrt (2 * Real.pi)) * Real.exp (-(x - μ)^2 / (2 * σ^2))

theorem normal_density_properties (μ σ : ℝ) (hσ : σ > 0) :
  (∀ x : ℝ, normal_density μ σ (2*μ - x) = normal_density μ σ x) ∧
  (∀ x : ℝ, normal_density μ σ x > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_density_properties_l179_17944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l179_17946

/-- Given two vectors v1 and v2 in ℝ², prove that there exists a unique vector q
    such that both v1 and v2 project onto q when projected onto some vector u. -/
theorem projection_equality (v1 v2 : ℝ × ℝ) (h : v1 ≠ v2) :
  ∃! q : ℝ × ℝ, ∃ u : ℝ × ℝ,
    (v1 - q) • u = 0 ∧ (v2 - q) • u = 0 :=
by
  -- Instantiate the given vectors
  let v1 : ℝ × ℝ := (-3, 5)
  let v2 : ℝ × ℝ := (4, 1)
  -- Define the proposed solution
  let q : ℝ × ℝ := (-92/33, 161/33)
  sorry

/-- The unique vector q that satisfies the projection equality for the given vectors. -/
noncomputable def projection_vector : ℝ × ℝ := (-92/33, 161/33)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l179_17946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_side_length_from_area_l179_17942

-- Define the hexagon
structure Hexagon where
  side_length : ℝ
  area : ℝ
  angles_60 : Fin 3 → ℝ
  angles_120 : Fin 3 → ℝ

-- Define the properties of the hexagon
def hexagon_properties (h : Hexagon) : Prop :=
  h.area = 18 ∧
  (∀ i : Fin 3, h.angles_60 i = 60 ∧ h.angles_120 i = 120)

-- Theorem statement
theorem hexagon_perimeter (h : Hexagon) 
  (hp : hexagon_properties h) : 
  6 * h.side_length = 12 * Real.rpow 3 (1/4) := by
  sorry

-- Additional helper theorem to show the relationship between side length and area
theorem side_length_from_area (h : Hexagon) 
  (hp : hexagon_properties h) :
  h.side_length = 2 * Real.rpow 3 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_side_length_from_area_l179_17942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_pointed_star_angle_sum_l179_17929

/-- A star polygon with n points evenly spaced on a circle -/
structure StarPolygon (n : ℕ) where
  n_ge_5 : n ≥ 5

/-- The angle at each tip of a star polygon -/
noncomputable def tip_angle (n : ℕ) (star : StarPolygon n) : ℝ :=
  (360 / n) * 2

/-- The sum of angles at all tips of a star polygon -/
noncomputable def sum_of_tip_angles (n : ℕ) (star : StarPolygon n) : ℝ :=
  n * tip_angle n star

/-- Theorem: The sum of angles at all tips of a 9-pointed star is 720 degrees -/
theorem nine_pointed_star_angle_sum :
  ∀ (star : StarPolygon 9), sum_of_tip_angles 9 star = 720 := by
  sorry

#check nine_pointed_star_angle_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_pointed_star_angle_sum_l179_17929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l179_17988

-- Define the vectors and function
noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 2 * Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - Real.sqrt 3

-- State the theorem
theorem f_properties :
  (∃ (k : ℤ), ∀ (x : ℝ), f (x + π) = f x) ∧
  (∃ (α : ℝ), π/2 < α ∧ α < π ∧
    (f (α/2 - π/6) - f (α/2 + π/12) = Real.sqrt 6) ∧
    (α = 7*π/12 ∨ α = 11*π/12)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l179_17988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l179_17913

-- Define n as 2^0.1
noncomputable def n : ℝ := 2^(1/10)

-- Theorem stating that if n^b = 16, then b = 40
theorem power_equality (b : ℝ) (h : n^b = 16) : b = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l179_17913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l179_17974

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (1 + Real.cos θ, Real.sin θ)

-- Define the line L
def line_L (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 2 * Real.sqrt 3

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  A ≠ B ∧ 
  ∃ θ₁ θ₂, curve_C θ₁ = A ∧ curve_C θ₂ = B ∧
  line_L A.1 A.2 ∧ line_L B.1 B.2

-- Theorem statement
theorem distance_between_intersection_points 
  (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l179_17974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_from_floor_division_l179_17947

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem integers_from_floor_division (a b : ℝ) : 
  a > 0 → b > 0 → a ≠ b → 
  (∀ n : ℕ+, (floor (n * a) ∣ floor (n * b))) → 
  (∃ m : ℤ, a = m) ∧ (∃ k : ℤ, b = k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_from_floor_division_l179_17947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_bc_distance_l179_17916

/-- Two internally tangent circles with centers A and B, and a line tangent to both -/
structure TangentCircles where
  A : EuclideanSpace ℝ (Fin 2) -- Center of first circle
  B : EuclideanSpace ℝ (Fin 2) -- Center of second circle
  C : EuclideanSpace ℝ (Fin 2) -- Point where tangent line intersects AB
  r₁ : ℝ     -- Radius of first circle
  r₂ : ℝ     -- Radius of second circle
  h₁ : r₁ = 7
  h₂ : r₂ = 4
  h₃ : dist A B = r₁ - r₂  -- Internally tangent condition
  h₄ : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B  -- C lies on AB

/-- The distance BC in the TangentCircles configuration is 4 -/
theorem tangent_circles_bc_distance (tc : TangentCircles) : dist tc.B tc.C = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_bc_distance_l179_17916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_mod_1000_l179_17979

/-- The number of As, Bs, and Cs in the original string -/
def numA : ℕ := 5
def numB : ℕ := 6
def numC : ℕ := 7

/-- The total length of the string -/
def totalLength : ℕ := numA + numB + numC

/-- The number of permutations satisfying the conditions -/
def N : ℕ := Finset.sum (Finset.range 5) (fun k => 
  (numA.choose (k+1)) * (numB.choose k) * (numC.choose (k+2)))

/-- The main theorem stating that N is congruent to 996 modulo 1000 -/
theorem permutation_count_mod_1000 : N ≡ 996 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_mod_1000_l179_17979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_triangle_property_l179_17972

/-- A set has the triangle property if it includes three distinct elements
    that are side lengths of a triangle with positive area. -/
def has_triangle_property (s : Finset ℕ) : Prop :=
  ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of consecutive integers from 7 to n, inclusive. -/
def consecutive_set (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => 7 ≤ x ∧ x ≤ n) (Finset.range (n + 1))

/-- All ten-element subsets of a set have the triangle property. -/
def all_ten_subsets_have_triangle_property (s : Finset ℕ) : Prop :=
  ∀ t ⊆ s, t.card = 10 → has_triangle_property t

/-- 258 is the largest integer n such that for the set {7, 8, 9, ..., n},
    all ten-element subsets have the triangle property. -/
theorem largest_n_with_triangle_property :
  (∀ n ≤ 258, all_ten_subsets_have_triangle_property (consecutive_set n)) ∧
  ¬(all_ten_subsets_have_triangle_property (consecutive_set 259)) := by
  sorry

#check largest_n_with_triangle_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_triangle_property_l179_17972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_36km_l179_17908

/-- The distance between Maxwell and Brad's homes -/
noncomputable def distance_between_homes (maxwell_speed brad_speed maxwell_distance : ℝ) : ℝ :=
  maxwell_distance * (1 + brad_speed / maxwell_speed)

theorem distance_is_36km (maxwell_speed brad_speed maxwell_distance : ℝ) 
  (h1 : maxwell_speed = 3)
  (h2 : brad_speed = 6)
  (h3 : maxwell_distance = 12) :
  distance_between_homes maxwell_speed brad_speed maxwell_distance = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_36km_l179_17908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_dimensions_l179_17911

/-- Represents a rectangle with given perimeter and length-width relationship --/
structure Rectangle where
  perimeter : ℕ
  length_width_diff : ℕ

/-- Calculates the width of the rectangle given its perimeter and length-width difference --/
def calculate_width (rect : Rectangle) : ℕ :=
  (rect.perimeter - 2 * rect.length_width_diff) / 4

/-- Calculates the length of the rectangle given its width and length-width difference --/
def calculate_length (rect : Rectangle) (width : ℕ) : ℕ :=
  width + rect.length_width_diff

/-- Theorem stating the dimensions of the rectangle --/
theorem rectangle_dimensions (rect : Rectangle) 
  (h_perimeter : rect.perimeter = 28)
  (h_length_width_diff : rect.length_width_diff = 4) :
  calculate_width rect = 5 ∧ calculate_length rect (calculate_width rect) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_dimensions_l179_17911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rooks_on_black_squares_even_l179_17990

/-- A chessboard is represented as a function from pairs of integers (1 to 8) to Booleans,
    where True represents a rook's presence. -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- A valid chessboard configuration has exactly one rook in each row and column. -/
def is_valid_configuration (board : Chessboard) : Prop :=
  (∀ i : Fin 8, ∃! j : Fin 8, board i j) ∧
  (∀ j : Fin 8, ∃! i : Fin 8, board i j)

/-- A square is black if the sum of its row and column indices is even. -/
def is_black_square (i j : Fin 8) : Bool :=
  (i.val + j.val) % 2 = 0

/-- Count the number of rooks on black squares. -/
def count_rooks_on_black_squares (board : Chessboard) : ℕ :=
  (Finset.univ : Finset (Fin 8)).sum fun i =>
    (Finset.univ : Finset (Fin 8)).sum fun j =>
      if board i j ∧ is_black_square i j then 1 else 0

theorem rooks_on_black_squares_even (board : Chessboard) 
  (h : is_valid_configuration board) :
  Even (count_rooks_on_black_squares board) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rooks_on_black_squares_even_l179_17990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l179_17962

/-- A quadratic function with specific properties -/
def quadratic_function (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧
  f 2 = -1 ∧ f (-1) = -1 ∧ (∃ x, ∀ y, f y ≤ f x) ∧ (∃ x, f x = 8)

/-- The analytical expression of the quadratic function -/
noncomputable def analytical_expression (x : ℝ) : ℝ :=
  -9/4 * x^2 + 9/2 * x + 23/4

/-- The minimum value of the function on the interval [m, 3] -/
noncomputable def min_value (m : ℝ) : ℝ :=
  if m ≤ -1/2 then -9/4 * m^2 + 9/2 * m + 23/4 else -33/4

theorem quadratic_function_properties :
  ∀ f : ℝ → ℝ, quadratic_function f →
    (∀ x, f x = analytical_expression x) ∧
    (∀ m, m < 3 → (∀ x, m ≤ x ∧ x ≤ 3 → min_value m ≤ f x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l179_17962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_1728000_l179_17980

theorem cube_root_1728000 : (1728000 : ℝ) ^ (1/3 : ℝ) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_1728000_l179_17980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_and_chord_length_l179_17956

-- Define the circles and line
def circle_C (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def circle_M (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - Real.sqrt 15)^2 = r^2}

def tangent_line (n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + Real.sqrt 3 * p.2 + n = 0}

def intersection_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.sqrt 3 * p.1 - Real.sqrt 2 * p.2 = 0}

-- Define the theorem
theorem circle_tangency_and_chord_length 
  (center_C : ℝ × ℝ) 
  (radius_C : ℝ) 
  (n : ℝ) 
  (r : ℝ) :
  (center_C.2 = 0) →  -- Center of C is on x-axis
  ((3/2, Real.sqrt 3/2) ∈ circle_C center_C radius_C) →  -- Point is on circle C
  ((3/2, Real.sqrt 3/2) ∈ tangent_line n) →  -- Point is on tangent line
  (∃ p, p ∈ circle_C center_C radius_C ∧ p ∈ circle_M r) →  -- Circles are tangent
  (r > 0) →
  let chord_length := Real.sqrt (4 * (r^2 - 6))
  chord_length = 2 * Real.sqrt 3 ∨ chord_length = 2 * Real.sqrt 19 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_and_chord_length_l179_17956

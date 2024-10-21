import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_todd_spent_amount_l118_11809

def wallet_amount : ℚ := 37.25
def candy_bar_price : ℚ := 1.14
def cookies_price : ℚ := 2.39
def soda_price : ℚ := 1.75
def chips_price : ℚ := 1.85
def juice_price : ℚ := 2.69
def hamburger_price : ℚ := 3.99
def candy_hamburger_discount : ℚ := 0.12
def cookies_discount : ℚ := 0.15
def sales_tax : ℚ := 0.085

def total_spent : ℚ := 13.93

theorem todd_spent_amount :
  let candy_bar_discounted := candy_bar_price * (1 - candy_hamburger_discount)
  let hamburger_discounted := hamburger_price * (1 - candy_hamburger_discount)
  let cookies_discounted := cookies_price * (1 - cookies_discount)
  let subtotal := candy_bar_discounted + hamburger_discounted + cookies_discounted + soda_price + chips_price + juice_price
  let total_with_tax := subtotal * (1 + sales_tax)
  (Int.floor (total_with_tax * 100 + 0.5) : ℚ) / 100 = total_spent := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_todd_spent_amount_l118_11809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_sevens_in_hundred_l118_11879

/-- Count the occurrences of a digit in a number -/
def countDigit (n : Nat) (d : Nat) : Nat :=
  sorry

/-- Count the occurrences of a digit in a range of numbers -/
def countDigitInRange (start : Nat) (stop : Nat) (d : Nat) : Nat :=
  sorry

/-- The main theorem stating that the count of 7s in numbers from 1 to 100 is 14 -/
theorem count_sevens_in_hundred : countDigitInRange 1 100 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_sevens_in_hundred_l118_11879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_problem_l118_11877

theorem pencil_problem (x y : ℤ) : 
  (∀ m : ℤ, (y : ℚ) / x - 1 / 15 = 2 / (x + 10)) → 
  (∀ n : ℤ, n > 0 → (y : ℚ) / x - 2 / (30 * n) = y / (x + 30 * n)) →
  x = 5 ∧ y = 1 := by
  sorry

#check pencil_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_problem_l118_11877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_tangent_line_l118_11815

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

-- State the theorem
theorem function_and_tangent_line 
  (h : HasDerivAt (f 1) 1 0) : 
  (∀ x, f 1 x = x^2 + x) ∧ 
  (∀ y, x + y + 1 = 0 ↔ y = (f 1 (-1) + (deriv (f 1) (-1))*(x - (-1)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_tangent_line_l118_11815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_is_14_l118_11821

-- Define the binomial expansion function
def binomialCoeff (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expansion of (2-√x)^7
noncomputable def expansion (x : ℝ) : ℝ := (2 - Real.sqrt x)^7

-- Theorem statement
theorem coefficient_x_cubed_is_14 :
  ∃ (f : ℝ → ℝ), (∀ x, expansion x = 14 * x^3 + f x) ∧ (∀ x, x^3 ∣ f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_is_14_l118_11821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_g_4_l118_11861

-- Define the functions s and g as noncomputable
noncomputable def s (x : ℝ) : ℝ := Real.sqrt (4 * x + 2)
noncomputable def g (x : ℝ) : ℝ := 4 - s x

-- State the theorem
theorem s_of_g_4 : s (g 4) = Real.sqrt (18 - 12 * Real.sqrt 2) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_g_4_l118_11861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_walk_distance_l118_11875

/-- Represents the initial setup and movement of Jack, Christina, and Lindy --/
structure DogWalk where
  jack_speed : ℚ
  christina_speed : ℚ
  lindy_speed : ℚ
  lindy_distance : ℚ

/-- Calculates the initial distance between Jack and Christina --/
def initial_distance (w : DogWalk) : ℚ :=
  (w.jack_speed + w.christina_speed) * (w.lindy_distance / w.lindy_speed)

/-- Theorem stating that the initial distance between Jack and Christina is 270 feet --/
theorem dog_walk_distance (w : DogWalk) 
    (h1 : w.jack_speed = 4)
    (h2 : w.christina_speed = 5)
    (h3 : w.lindy_speed = 8)
    (h4 : w.lindy_distance = 240) :
    initial_distance w = 270 := by
  sorry

/-- Example calculation --/
def example_walk : DogWalk := {
  jack_speed := 4,
  christina_speed := 5,
  lindy_speed := 8,
  lindy_distance := 240
}

#eval initial_distance example_walk

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_walk_distance_l118_11875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equals_M_expression_l118_11807

/-- The side length of the cube -/
def cube_side : ℝ := 3

/-- The surface area of the cube -/
noncomputable def cube_surface_area : ℝ := 6 * cube_side^2

/-- The surface area of the sphere -/
noncomputable def sphere_surface_area : ℝ := cube_surface_area

/-- The radius of the sphere -/
noncomputable def sphere_radius : ℝ := (3 * Real.sqrt 3) / Real.sqrt Real.pi

/-- The volume of the sphere -/
noncomputable def sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius^3

/-- The value M in the problem -/
def M : ℝ := 36

theorem sphere_volume_equals_M_expression :
  sphere_volume = (M * Real.sqrt 3) / Real.sqrt Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equals_M_expression_l118_11807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l118_11804

theorem hyperbola_eccentricity_range (a b c : ℝ) :
  a > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / (5 - a^2) = 1) →
  2 < b / a →
  b / a < 3 →
  c / a = Real.sqrt (1 + (b / a)^2) →
  Real.sqrt 5 < c / a ∧ c / a < Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l118_11804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l118_11849

def A : Set ℝ := {0, 1, 2, 3}

def B : Set ℝ := {x : ℝ | x^2 - x - 2 < 0}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l118_11849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_comparison_l118_11856

noncomputable section

structure School where
  soccer_prob : ℝ
  running_prob : ℝ
  jump_prob : ℝ

def school_A : School := ⟨2/3, 2/3, 2/3⟩

def school_B (m : ℝ) : School := ⟨3/4, 2/3, m⟩

noncomputable def prob_exactly_two (s : School) : ℝ :=
  3 * s.soccer_prob * s.running_prob * (1 - s.jump_prob)

noncomputable def prob_at_most_two (s : School) : ℝ :=
  1 - s.soccer_prob * s.running_prob * s.jump_prob

noncomputable def expected_passes (s : School) : ℝ :=
  s.soccer_prob + s.running_prob + s.jump_prob

theorem school_comparison (m : ℝ) (h1 : 0 < m) (h2 : m < 1) :
  (prob_exactly_two school_A = 4/9) ∧
  (prob_at_most_two (school_B (3/4)) = 5/8) ∧
  (∀ m, 0 < m ∧ m < 7/12 → expected_passes school_A > expected_passes (school_B m)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_comparison_l118_11856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l118_11868

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 110 m, traveling at 72 km/hr, 
    takes 14.25 seconds to cross a bridge of length 175 m -/
theorem train_crossing_bridge_time :
  train_crossing_time 110 72 175 = 14.25 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_crossing_time 110 72 175

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l118_11868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_standard_right_triangles_l118_11851

/-- A right triangle with integer side lengths where the perimeter equals the area -/
structure StandardRightTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  right_triangle : a ^ 2 + b ^ 2 = c ^ 2
  perimeter_equals_area : (a : ℕ) + (b : ℕ) + (c : ℕ) = (a * b) / 2

/-- The number of standard right triangles -/
def count_standard_right_triangles : ℕ := 2

/-- Theorem: There are exactly two standard right triangles -/
theorem two_standard_right_triangles :
  ∃ (t₁ t₂ : StandardRightTriangle), ∀ (t : StandardRightTriangle), t = t₁ ∨ t = t₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_standard_right_triangles_l118_11851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_negation_equal_angles_sides_l118_11843

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the property of equal sides
def equalSides (t : Triangle) (s1 s2 : ℝ) : Prop :=
  (s1 = t.a ∧ s2 = t.b) ∨ (s1 = t.b ∧ s2 = t.c) ∨ (s1 = t.a ∧ s2 = t.c)

-- Define the property of equal angles
def equalAngles (t : Triangle) (α1 α2 : ℝ) : Prop :=
  (α1 = t.α ∧ α2 = t.β) ∨ (α1 = t.β ∧ α2 = t.γ) ∨ (α1 = t.α ∧ α2 = t.γ)

-- State the theorem
theorem inverse_negation_equal_angles_sides (t : Triangle) :
  (¬(∃ s1 s2 α1 α2, equalSides t s1 s2 ∧ equalAngles t α1 α2)) →
  (∀ s1 s2 α1 α2, ¬(equalAngles t α1 α2) → ¬(equalSides t s1 s2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_negation_equal_angles_sides_l118_11843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_inequality_degenerate_triangle_inequality_l118_11837

theorem isosceles_triangle_inequality (a b k : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : k = a + 2*b) :
  k/2 < a + b ∧ a + b < 3*k/4 := by
  constructor
  · -- Proof for k/2 < a + b
    sorry
  · -- Proof for a + b < 3*k/4
    sorry

-- For the degenerate case
theorem degenerate_triangle_inequality (b k : ℝ) 
  (h1 : b > 0) (h2 : k = 2*b) :
  k/2 ≤ b ∧ b ≤ 3*k/4 := by
  constructor
  · -- Proof for k/2 ≤ b
    sorry
  · -- Proof for b ≤ 3*k/4
    sorry

#check isosceles_triangle_inequality
#check degenerate_triangle_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_inequality_degenerate_triangle_inequality_l118_11837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l118_11822

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

-- State the theorem
theorem monotonic_decreasing_interval_of_f :
  ∃ (a b : ℝ), a = 0 ∧ b = 1/2 ∧
  (∀ x, x ∈ Set.Ioo a b → x ∈ Set.Ioi 0) ∧
  (∀ x y, x ∈ Set.Ioo a b → y ∈ Set.Ioo a b → x < y → f y < f x) ∧
  (∀ ε > 0, ∃ x y, x ∈ Set.Ioi 0 ∧ y ∈ Set.Ioi 0 ∧ x < a + ε ∧ y > b - ε ∧ f x ≤ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l118_11822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_period_50_days_l118_11897

/-- Given a bowl of water with initial amount and daily evaporation rate,
    calculate the number of days for a specific percentage to evaporate. -/
noncomputable def evaporation_days (initial_amount : ℝ) (daily_rate : ℝ) (evaporation_percent : ℝ) : ℝ :=
  (initial_amount * evaporation_percent / 100) / daily_rate

/-- Theorem stating that it takes 50 days for 0.4% of 10 ounces to evaporate
    at a rate of 0.0008 ounce per day. -/
theorem evaporation_period_50_days :
  evaporation_days 10 0.0008 0.4 = 50 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_period_50_days_l118_11897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l118_11876

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

theorem geometric_series_sum :
  let a : ℝ := 3
  let r : ℝ := -2
  let n : ℕ := 10
  geometric_sum a r n = -1023 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l118_11876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l118_11859

noncomputable def f (x : ℝ) := 1 / Real.log (x + 1) + Real.sqrt (4 - x^2)

def domain (f : ℝ → ℝ) := {x : ℝ | ∃ y, f x = y}

theorem f_domain :
  domain f = {x : ℝ | -1 < x ∧ x ≤ 2 ∧ x ≠ 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l118_11859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_one_l118_11803

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.c / Real.sin t.B + t.b / Real.sin t.C = 2 * t.a

-- Theorem statement
theorem triangle_area_is_one (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.b = Real.sqrt 2) : 
  (1 / 2) * t.a * t.b * Real.sin t.C = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_one_l118_11803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carnation_percentage_l118_11864

def flower_bouquet (total : ℕ) (pink yellow blue red : ℕ) : Prop :=
  pink + yellow + blue + red = total ∧
  (pink + yellow : ℚ) / total = 7 / 10 ∧
  (blue : ℚ) / total = 3 / 10

def pink_roses_ratio (pink_roses pink : ℕ) : Prop :=
  (pink_roses : ℚ) / pink = 1 / 5

def yellow_orchids_ratio (yellow_orchids yellow : ℕ) : Prop :=
  (yellow_orchids : ℚ) / yellow = 4 / 5

theorem carnation_percentage
  (total pink yellow blue red pink_roses yellow_orchids : ℕ)
  (h1 : flower_bouquet total pink yellow blue red)
  (h2 : pink_roses_ratio pink_roses pink)
  (h3 : yellow_orchids_ratio yellow_orchids yellow) :
  (blue : ℚ) / total = 3 / 10 :=
by
  sorry

#check carnation_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carnation_percentage_l118_11864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_4_l118_11855

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 18 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 3 * x^2 - 3 * x - 4

-- State the theorem
theorem f_of_g_of_4 : f (g 4) = 14.25 * Real.sqrt 2 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_4_l118_11855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_rectangle_poster_length_l118_11898

/-- The golden ratio -/
noncomputable def golden_ratio : ℝ := (Real.sqrt 5 - 1) / 2

/-- The width of the poster in cm -/
noncomputable def poster_width : ℝ := 20 + 2 * Real.sqrt 5

/-- Theorem: The length of a golden rectangle poster with width (20 + 2√5) cm is (15 + 11√5) cm -/
theorem golden_rectangle_poster_length :
  let length := poster_width / golden_ratio
  length = 15 + 11 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_rectangle_poster_length_l118_11898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_circle_ratio_l118_11806

/-- A rhombus with an inscribed circle. -/
structure RhombusWithCircle where
  -- The side length of the rhombus
  a : ℝ
  -- The radius of the inscribed circle
  r : ℝ
  -- Assumption that a and r are positive
  a_pos : 0 < a
  r_pos : 0 < r
  -- The angle ABC is 60°
  angle_abc : Real.cos (60 * π / 180) = 1/2
  -- The circle is tangent to AD at A
  tangent_at_a : True
  -- The center of the circle is inside the rhombus
  center_inside : True
  -- Tangents from C to the circle are perpendicular
  perpendicular_tangents : True
  -- The relationship between a and r derived from the problem constraints
  a_r_relation : r^2 + a*r*(Real.sqrt 3) - a^2 = 0

/-- The theorem stating the ratio of rhombus perimeter to circle circumference. -/
theorem rhombus_circle_ratio (rc : RhombusWithCircle) :
  (4 * rc.a) / (2 * Real.pi * rc.r) = (Real.sqrt 3 + Real.sqrt 7) / Real.pi := by
  sorry

#check rhombus_circle_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_circle_ratio_l118_11806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_three_heads_ten_coins_l118_11811

theorem probability_at_most_three_heads_ten_coins : 
  let n : ℕ := 10  -- total number of coins
  let k : ℕ := 3   -- maximum number of heads
  let favorable_outcomes : ℕ := (Finset.range (k + 1)).sum (λ i ↦ n.choose i)
  let total_outcomes : ℕ := 2^n
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_three_heads_ten_coins_l118_11811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_implies_a_range_l118_11801

/-- The function f(x) defined as 2^x - 2/x - a --/
noncomputable def f (x a : ℝ) : ℝ := 2^x - 2/x - a

/-- Theorem: If f(x) has a zero point in (1,2), then a is in (0,3) --/
theorem zero_point_implies_a_range (a : ℝ) :
  (∃ x, x ∈ Set.Ioo 1 2 ∧ f x a = 0) → a ∈ Set.Ioo 0 3 :=
by
  sorry

#check zero_point_implies_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_implies_a_range_l118_11801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l118_11802

-- Define the complex number z
noncomputable def z : ℂ := (3 + Complex.I) / (1 + Complex.I)

-- Theorem statement
theorem z_in_fourth_quadrant : 
  Real.sign z.re = 1 ∧ Real.sign z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l118_11802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_polar_circle_l118_11800

-- Define the circle in polar coordinates
noncomputable def polar_circle (ρ θ : ℝ) : Prop := ρ = Real.sin θ

-- Define the center of the circle in polar coordinates
noncomputable def circle_center : ℝ × ℝ := (1/2, Real.pi/2)

-- Theorem statement
theorem center_of_polar_circle :
  ∀ ρ θ : ℝ, polar_circle ρ θ → 
  ∃ (center : ℝ × ℝ), center = circle_center :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_polar_circle_l118_11800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_managers_salary_l118_11818

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (salary_increase : ℚ) (managers_salary : ℚ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 150 →
  (num_employees * avg_salary + managers_salary) / (num_employees + 1) = avg_salary + salary_increase →
  managers_salary = 4650 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_managers_salary_l118_11818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l118_11896

/-- The area of a sector of a circle with given radius and arc length -/
noncomputable def sectorArea (radius : ℝ) (arcLength : ℝ) : ℝ :=
  (arcLength / (2 * Real.pi * radius)) * (Real.pi * radius^2)

/-- Theorem: The area of a sector of a circle with radius 5 cm and arc length 3.5 cm is 8.75 cm² -/
theorem sector_area_example : sectorArea 5 3.5 = 8.75 := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l118_11896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l118_11873

/-- Predicate to check if a digit is less than 5 -/
def isLessThan5 (d : Nat) : Bool :=
  d < 5

/-- Predicate to check if a digit is greater than 5 -/
def isGreaterThan5 (d : Nat) : Bool :=
  d > 5

/-- Predicate to check if two digits satisfy the condition -/
def satisfyCondition (d1 d2 : Nat) : Bool :=
  (isLessThan5 d1 ∧ isLessThan5 d2) ∨ (isGreaterThan5 d1 ∧ isGreaterThan5 d2)

/-- Function to count valid four-digit numbers -/
def countValidNumbers : Nat :=
  (Finset.filter (fun n => 
    let d1 := n / 1000
    let d2 := (n / 100) % 10
    let d3 := (n / 10) % 10
    let d4 := n % 10
    satisfyCondition d1 d2 ∧ satisfyCondition d3 d4
  ) (Finset.range 9000)).card

/-- Theorem stating the count of valid four-digit numbers -/
theorem count_valid_numbers : countValidNumbers = 1476 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l118_11873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_problem_solution_l118_11839

/-- Represents a walker with a starting time and speed. -/
structure Walker where
  startTime : ℚ
  speed : ℚ

/-- The problem setup -/
def walkingProblem (totalDistance : ℚ) (adrienne harold christine : Walker) : Prop :=
  totalDistance = 60 ∧
  adrienne.startTime = 0 ∧
  adrienne.speed = 3 ∧
  harold.startTime = 1 ∧
  harold.speed = adrienne.speed + 1 ∧
  christine.startTime = 1 ∧
  christine.speed = harold.speed - (1/2)

/-- The meeting point of two walkers -/
def meetingPoint (w1 w2 : Walker) : ℚ :=
  (w1.speed * w1.startTime + w2.speed * w2.startTime) / (w1.speed + w2.speed)

/-- The theorem to be proved -/
theorem walking_problem_solution 
  (totalDistance : ℚ) (adrienne harold christine : Walker) 
  (h : walkingProblem totalDistance adrienne harold christine) :
  meetingPoint adrienne harold = 12 ∧ 
  meetingPoint adrienne christine ≠ meetingPoint adrienne harold := by
  sorry

#eval meetingPoint ⟨0, 3⟩ ⟨1, 4⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_problem_solution_l118_11839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isocline_through_origin_special_isoclines_vertical_tangents_l118_11850

/-- The differential equation dy/dx = (y - x) / (y + x) -/
noncomputable def diff_eq (x y : ℝ) : ℝ := (y - x) / (y + x)

/-- An isocline of the differential equation -/
def isocline (k : ℝ) (x y : ℝ) : Prop := diff_eq x y = k

theorem isocline_through_origin (k : ℝ) (h : k ≠ 1) :
  ∃ m : ℝ, ∀ x y : ℝ, isocline k x y → y = m * x :=
by sorry

theorem special_isoclines :
  (∀ x : ℝ, isocline (-1) x 0) ∧
  (∀ x : ℝ, isocline 0 x x) ∧
  (∀ y : ℝ, isocline 1 0 y) :=
by sorry

theorem vertical_tangents :
  ∀ x : ℝ, (diff_eq x (-x))⁻¹ = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isocline_through_origin_special_isoclines_vertical_tangents_l118_11850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_with_gcd_conditions_l118_11812

theorem least_n_with_gcd_conditions : 
  ∃ n : ℕ, n > 2000 ∧ 
    Nat.gcd 75 (n + 135) = 15 ∧ 
    Nat.gcd (n + 75) 135 = 45 ∧
    (∀ m : ℕ, m > 2000 → 
      Nat.gcd 75 (m + 135) = 15 → 
      Nat.gcd (m + 75) 135 = 45 → 
      n ≤ m) ∧
    n = 2025 :=
by sorry

def sumDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumDigits (n / 10)

#eval sumDigits 2025

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_with_gcd_conditions_l118_11812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_and_sum_l118_11860

theorem cube_root_sum_and_sum (x y : ℝ) :
  x^(1/3) + y^(1/3) = 4 ∧ x + y = 28 →
  (x = 1 ∧ y = 27) ∨ (x = 27 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_and_sum_l118_11860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_and_minimum_distance_l118_11870

-- Define the points and the line
def A : ℝ × ℝ := (-3, 5)
def B : ℝ × ℝ := (2, 15)
def line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the symmetric point A'
def A' : ℝ × ℝ := (4, -2)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem symmetric_point_and_minimum_distance :
  (∀ x y : ℝ, line x y → distance (x, y) A + distance (x, y) A' = distance (x, y) A + distance (x, y) B) ∧
  (∀ x y : ℝ, line x y → distance (x, y) A + distance (x, y) B ≥ Real.sqrt 293) ∧
  (∃ x y : ℝ, line x y ∧ distance (x, y) A + distance (x, y) B = Real.sqrt 293) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_and_minimum_distance_l118_11870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l118_11882

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the asymptote of the hyperbola
def asymptote (a b x y : ℝ) : Prop := b * x + a * y = 0

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x y : ℝ, hyperbola a b x y ∧ circle_equation x y ∧ asymptote a b x y) →
  eccentricity a b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l118_11882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_with_one_repeat_is_360_l118_11842

def four_digit_numbers_with_one_repeat : ℕ :=
  let digits : Finset ℕ := {1, 2, 3, 4, 5}
  let total_positions : ℕ := 4
  let repeat_count : ℕ := 1
  
  -- Number of ways to choose the repeated digit
  let repeated_digit_choices : ℕ := Finset.card digits
  
  -- Number of ways to place the repeated digit
  let repeated_digit_positions : ℕ := Nat.choose total_positions 2
  
  -- Number of ways to choose remaining digits
  let remaining_digit_choices : ℕ := Nat.choose (Finset.card digits - 1) (total_positions - repeat_count - 1)
  
  -- Number of ways to arrange remaining digits
  let remaining_digit_arrangements : ℕ := Nat.factorial (total_positions - repeat_count - 1)
  
  -- Total number of distinct four-digit numbers
  repeated_digit_choices * repeated_digit_positions * remaining_digit_choices * remaining_digit_arrangements

-- Evaluate the result
#eval four_digit_numbers_with_one_repeat

theorem four_digit_numbers_with_one_repeat_is_360 :
  four_digit_numbers_with_one_repeat = 360 := by
  -- The proof of this theorem
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_with_one_repeat_is_360_l118_11842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l118_11852

/-- The curve C in Cartesian coordinates -/
def curve (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

/-- The line L in Cartesian coordinates -/
def line (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 5 = 0

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem intersection_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    curve x₁ y₁ ∧ curve x₂ y₂ ∧
    line x₁ y₁ ∧ line x₂ y₂ ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l118_11852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_cross_section_count_l118_11836

-- Define the type for geometric bodies
inductive GeometricBody
  | Cylinder
  | Cube
  | Prism
  | Sphere
  | Cone
  | Cuboid

-- Define a function to check if a geometric body can have a circular cross-section
def hasCircularCrossSection (body : GeometricBody) : Bool :=
  match body with
  | GeometricBody.Cylinder => true
  | GeometricBody.Sphere => true
  | GeometricBody.Cone => true
  | _ => false

-- Define the list of geometric bodies
def geometricBodies : List GeometricBody :=
  [GeometricBody.Cylinder, GeometricBody.Cube, GeometricBody.Prism,
   GeometricBody.Sphere, GeometricBody.Cone, GeometricBody.Cuboid]

-- Theorem: The number of geometric bodies that can have a circular cross-section is 3
theorem circular_cross_section_count :
  (geometricBodies.filter hasCircularCrossSection).length = 3 := by
  sorry

#eval (geometricBodies.filter hasCircularCrossSection).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_cross_section_count_l118_11836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l118_11828

-- Define the semicircle and rectangle
def Semicircle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 ∧ p.2 ≥ center.2}

def Rectangle (bottomLeft topRight : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | bottomLeft.1 ≤ p.1 ∧ p.1 ≤ topRight.1 ∧ bottomLeft.2 ≤ p.2 ∧ p.2 ≤ topRight.2}

-- Define the theorem
theorem inscribed_rectangle_area 
  (G H M N O P : ℝ × ℝ)
  (h_semicircle : Semicircle ((G.1 + H.1) / 2, 0) (‖H - G‖ / 2))
  (h_rectangle : Rectangle M P)
  (h_inscribed : Rectangle M P ⊆ Semicircle ((G.1 + H.1) / 2, 0) (‖H - G‖ / 2))
  (h_MN : ‖N - M‖ = 18)
  (h_MG : ‖M - G‖ = 6)
  (h_PH : ‖P - H‖ = 6) :
  ‖N - M‖ * ‖P - M‖ = 54 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l118_11828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_two_equals_two_thirds_l118_11889

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (1/x + 1)

-- State the theorem
theorem f_of_two_equals_two_thirds : f 2 = 2/3 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [div_add_one, div_eq_mul_inv]
  -- Prove the equality
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_two_equals_two_thirds_l118_11889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circle_centers_l118_11820

theorem distance_between_circle_centers (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_sides : a = 8 ∧ b = 15 ∧ c = 17) : 
  let s := (a + b + c) / 2;
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c));
  let inradius := area / s;
  let circumradius := c / 2;
  Real.sqrt (inradius^2 + (circumradius - (c - a - b) / 2)^2) = 13 / 2 := by
  sorry

#check distance_between_circle_centers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circle_centers_l118_11820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terms_not_like_l118_11894

/-- Definition of like terms -/
def are_like_terms (term1 term2 : Polynomial ℚ) : Prop :=
  term1.support = term2.support

/-- The terms we're comparing -/
noncomputable def term1 : Polynomial ℚ := 2 * Polynomial.X^2 * Polynomial.X
noncomputable def term2 : Polynomial ℚ := 2 * Polynomial.X * Polynomial.X^2

/-- Theorem stating that term1 and term2 are not like terms -/
theorem terms_not_like : ¬(are_like_terms term1 term2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_terms_not_like_l118_11894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_BC_l118_11805

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line y = -2
def is_on_line_P (x y : ℝ) : Prop := y = -2

-- Define point A
def point_A : ℝ × ℝ := (2, 1)

-- Define the line BC as a function of t (parameter for point P)
def is_on_line_BC (t x y : ℝ) : Prop := t * x - 2 * y = 1

-- Define the distance function from a point to a line
noncomputable def distance_to_line (px py t : ℝ) : ℝ :=
  |t * px - 2 * py - 1| / Real.sqrt (t^2 + 4)

-- Theorem statement
theorem max_distance_to_BC :
  ∃ (max_dist : ℝ), max_dist = 5/2 ∧
  ∀ (t : ℝ), distance_to_line point_A.1 point_A.2 t ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_BC_l118_11805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l118_11885

/-- The number of solutions to the equation 3cos³x - 7cos²x + 3cosx = 0 in the range [0, 2π] -/
def num_solutions : ℕ := 4

/-- The equation we're solving -/
def equation (x : ℝ) : Prop :=
  3 * (Real.cos x)^3 - 7 * (Real.cos x)^2 + 3 * (Real.cos x) = 0

/-- The range of x we're considering -/
def in_range (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2 * Real.pi

theorem solution_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, equation x ∧ in_range x) ∧ S.card = num_solutions :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l118_11885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l118_11823

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_theorem (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = -1) :
  ∀ x, (fun y ↦ (1/2 : ℝ)^y) x = Function.invFun (f a) x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l118_11823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l118_11895

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then Real.log x / Real.log (1/2)
  else 2 + (4 ^ x)

-- State the theorem
theorem f_composition_value :
  f (f (1/2)) = -2 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l118_11895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_theta_l118_11832

-- Define the vectors a and b
noncomputable def a (θ : Real) : Real × Real := (2, 1 - Real.cos θ)
noncomputable def b (θ : Real) : Real × Real := (1 + Real.cos θ, 1/4)

-- Define the condition for parallel vectors
def are_parallel (v w : Real × Real) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Define an obtuse angle
def is_obtuse (θ : Real) : Prop :=
  Real.pi/2 < θ ∧ θ < Real.pi

-- State the theorem
theorem parallel_vectors_theta :
  ∀ θ : Real,
    are_parallel (a θ) (b θ) →
    is_obtuse θ →
    θ = Real.pi/4*3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_theta_l118_11832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equipment_calculations_l118_11810

/-- Equipment data and production information --/
structure EquipmentData where
  initialInvestment : ℚ
  equipmentCost : ℚ
  maxCapacity : ℚ
  julyProduction : ℚ
  augustProduction : ℚ
  septemberProduction : ℚ
  desiredProfit : ℚ

/-- Calculate depreciation per bottle --/
def depreciationPerBottle (data : EquipmentData) : ℚ :=
  data.equipmentCost / data.maxCapacity

/-- Calculate total depreciation --/
def totalDepreciation (data : EquipmentData) : ℚ :=
  (data.julyProduction + data.augustProduction + data.septemberProduction) * depreciationPerBottle data

/-- Calculate residual value --/
def residualValue (data : EquipmentData) : ℚ :=
  data.equipmentCost - totalDepreciation data

/-- Calculate required sales price --/
def requiredSalesPrice (data : EquipmentData) : ℚ :=
  residualValue data + data.desiredProfit

/-- Theorem stating the correctness of the calculations --/
theorem equipment_calculations (data : EquipmentData) 
  (h1 : data.initialInvestment = 1500000)
  (h2 : data.equipmentCost = 500000)
  (h3 : data.maxCapacity = 100000)
  (h4 : data.julyProduction = 200)
  (h5 : data.augustProduction = 15000)
  (h6 : data.septemberProduction = 12300)
  (h7 : data.desiredProfit = 10000) :
  totalDepreciation data = 137500 ∧ 
  residualValue data = 362500 ∧ 
  requiredSalesPrice data = 372500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equipment_calculations_l118_11810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l118_11833

noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

def left_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x => f (x + shift)

def shorten_abscissa (f : ℝ → ℝ) (factor : ℝ) : ℝ → ℝ :=
  λ x => f (factor * x)

theorem transformation_result :
  let f := original_function
  let g := left_shift f (π/3)
  let h := shorten_abscissa g 2
  ∀ x, h x = Real.sin (2*x + 2*π/3) := by
  intro x
  simp [original_function, left_shift, shorten_abscissa]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l118_11833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l118_11880

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 16*y

-- Define the focus and directrix
def focus : ℝ × ℝ := (0, 4)
def directrix (y : ℝ) : Prop := y = -4

-- Define a point on the directrix
variable (M : ℝ × ℝ)

-- Define a point on the parabola
variable (P : ℝ × ℝ)

-- State the theorem
theorem parabola_distance_theorem 
  (h1 : directrix M.2)
  (h2 : parabola P.1 P.2)
  (h3 : ∃ t : ℝ, P = M + t • (focus - M))  -- P is on line MF
  (h4 : focus - M = 3 • (focus - P))       -- FM = 3FP
  : ‖focus - P‖ = 16/3 := by
  sorry

#check parabola_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l118_11880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_mod_500_l118_11891

/-- A sequence of positive integers whose binary representation has exactly 6 ones -/
def T : ℕ → ℕ := sorry

/-- The 500th number in the sequence T -/
def M : ℕ := T 500

/-- T is an increasing sequence -/
axiom T_increasing : ∀ n m, n < m → T n < T m

/-- Each number in T has exactly 6 ones in its binary representation -/
axiom T_binary_ones : ∀ n, (Nat.digits 2 (T n)).count 1 = 6

theorem M_mod_500 : M % 500 = 204 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_mod_500_l118_11891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l118_11893

def A : Set ℕ := {x : ℕ | x > 0 ∧ x < 3}
def B : Set ℕ := {x : ℕ | x < 2}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l118_11893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_6x5_divisible_l118_11872

/-- Represents a shape made of unit squares --/
structure Shape where
  width : ℕ
  height : ℕ

/-- Represents an L-shaped piece --/
def L_shape : Shape :=
  { width := 2, height := 2 }

/-- Represents a cross-shaped piece --/
def cross_shape : Shape :=
  { width := 3, height := 3 }

/-- Checks if a shape can be divided into a given number of pieces of a specific type --/
def can_divide (s : Shape) (piece : Shape) (n : ℕ) : Prop :=
  ∃ (arrangement : List (ℕ × ℕ)), 
    arrangement.length = n ∧
    ∀ (pos : ℕ × ℕ), pos ∈ arrangement → 
      pos.1 + piece.width ≤ s.width ∧
      pos.2 + piece.height ≤ s.height

/-- The main theorem stating that a 6x5 rectangle can be divided into both L-shapes and cross-shapes --/
theorem rectangle_6x5_divisible :
  let rect : Shape := { width := 6, height := 5 }
  can_divide rect L_shape 4 ∧ can_divide rect cross_shape 5 := by
  sorry

#check rectangle_6x5_divisible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_6x5_divisible_l118_11872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_intersection_of_sequences_l118_11841

/-- The sequence S(x) defined as {[nx]}_{n ∈ ℕ} -/
def S (x : ℝ) : Set ℕ := {n : ℕ | ∃ k : ℕ, k = ⌊n * x⌋}

/-- The polynomial f(x) = x³ - 10x² + 29x - 25 -/
def f (x : ℝ) : ℝ := x^3 - 10*x^2 + 29*x - 25

theorem infinite_intersection_of_sequences :
  ∃ α β : ℝ, α ≠ β ∧ f α = 0 ∧ f β = 0 ∧ (S α ∩ S β).Infinite :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_intersection_of_sequences_l118_11841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l118_11884

-- Define the necessary structures and predicates
structure Point
structure Line
structure Circle

def SameSide : Point → Point → Line → Prop := sorry
def PassesThrough : Circle → Point → Prop := sorry
def TangentTo : Circle → Line → Prop := sorry

theorem circle_tangent_line (P Q : Point) (l : Line) : 
  P ≠ Q → 
  SameSide P Q l → 
  ∃ (c1 c2 : Circle), c1 ≠ c2 ∧ 
    PassesThrough c1 P ∧ PassesThrough c1 Q ∧ TangentTo c1 l ∧
    PassesThrough c2 P ∧ PassesThrough c2 Q ∧ TangentTo c2 l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l118_11884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_integer_T_l118_11831

-- Define the hexadecimal system
def hexDigits : Finset ℕ := Finset.range 15 \ {0}

-- Define the sum of reciprocals of hexadecimal digits
noncomputable def L : ℚ := (hexDigits.sum (λ i => 1 / ↑i))

-- Define T_n
noncomputable def T (n : ℕ) : ℚ := n * 16^(n-1) * L + 1

-- Theorem statement
theorem smallest_n_for_integer_T :
  ∃ (n : ℕ), n > 0 ∧ (T n).isInt ∧ ∀ (m : ℕ), m > 0 ∧ m < n → ¬(T m).isInt :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_integer_T_l118_11831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_variance_with_zero_and_one_l118_11853

/-- A set of n real numbers containing 0 and 1 -/
def NumberSet (n : ℕ) := { s : Finset ℝ // s.card = n ∧ 0 ∈ s ∧ 1 ∈ s }

/-- The variance of a finite set of real numbers -/
noncomputable def variance (s : Finset ℝ) : ℝ :=
  let mean := s.sum id / s.card
  s.sum (fun x => (x - mean) ^ 2) / s.card

/-- The theorem stating the minimum variance and the condition to achieve it -/
theorem min_variance_with_zero_and_one (n : ℕ) (h : 2 ≤ n) :
  (∃ (s : NumberSet n), ∀ (t : NumberSet n), variance s.val ≤ variance t.val) ∧
  (∃ (s : NumberSet n), variance s.val = 1 / (2 * n) ∧
    ∀ (x : ℝ), x ∈ s.val → x = 0 ∨ x = 1 ∨ x = 1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_variance_with_zero_and_one_l118_11853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_intersection_l118_11886

noncomputable section

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

theorem ellipse_fixed_point_intersection :
  ∀ (a b : ℝ),
  a > b ∧ b > 0 →
  eccentricity a b = 1/2 →
  ellipse (Real.sqrt 3) (Real.sqrt 3 / 2) a b →
  ellipse (-a) 0 a b →
  ∃ (P Q : ℝ × ℝ),
    ellipse P.1 P.2 a b ∧
    ellipse Q.1 Q.2 a b ∧
    (P.1 + a) * (Q.1 + a) + P.2 * Q.2 = 0 →
    ∃ (t : ℝ), P.2 = (t + 2/7) * P.1 ∧ Q.2 = (t + 2/7) * Q.1 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_intersection_l118_11886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_last_player_same_l118_11838

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament (n : ℕ) where
  /-- Total number of players -/
  num_players : ℕ
  /-- Total number of matches -/
  num_matches : ℕ
  /-- Condition that there are 2n+3 players -/
  player_count : num_players = 2*n + 3
  /-- Condition that the number of matches is correct -/
  match_count : num_matches = (num_players * (num_players - 1)) / 2
  /-- Function representing the schedule of matches -/
  schedule : Fin num_matches → Fin num_players × Fin num_players
  /-- Condition that every pair of players plays exactly once -/
  one_match_per_pair : ∀ i j, i < j → ∃! k, (schedule k = ⟨i, j⟩ ∨ schedule k = ⟨j, i⟩)
  /-- Condition that each player rests for at least n consecutive matches after playing -/
  rest_period : ∀ i k, (schedule k).1 = i ∨ (schedule k).2 = i →
    ∀ m, k < m → m < k + n → (schedule m).1 ≠ i ∧ (schedule m).2 ≠ i

/-- Theorem stating that one player from the first match also plays in the last match -/
theorem first_last_player_same (n : ℕ) (tournament : ChessTournament n) :
  ∃ i : Fin tournament.num_players,
    ((tournament.schedule ⟨0, sorry⟩).1 = i ∨ (tournament.schedule ⟨0, sorry⟩).2 = i) ∧
    ((tournament.schedule ⟨tournament.num_matches - 1, sorry⟩).1 = i ∨
     (tournament.schedule ⟨tournament.num_matches - 1, sorry⟩).2 = i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_last_player_same_l118_11838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l118_11858

noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 2) * (Real.sin (x / 2) - Real.sqrt 3 * Real.cos (x / 2))

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l118_11858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l118_11846

/-- The circle defined by x^2 + y^2 = 16 -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- The parabola defined by y = x^2 - 4 -/
def parabolaEq (x y : ℝ) : Prop := y = x^2 - 4

/-- A point (x, y) is an intersection point if it satisfies both equations -/
def isIntersectionPoint (x y : ℝ) : Prop := circleEq x y ∧ parabolaEq x y

/-- The set of all intersection points -/
def intersectionPoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | isIntersectionPoint p.1 p.2}

theorem intersection_count :
  ∃ (s : Finset (ℝ × ℝ)), s.card = 3 ∧ ∀ p, p ∈ s ↔ p ∈ intersectionPoints := by
  sorry

#check intersection_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l118_11846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_arrangement_count_l118_11847

theorem chemical_arrangement_count :
  let total_substances : ℕ := 8
  let selected_substances : ℕ := 4
  let restricted_substances : ℕ := 2
  let total_bottles : ℕ := 4
  
  (total_substances - restricted_substances).choose 1 *
  (total_substances - 1).descFactorial (selected_substances - 1) = 1260 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_arrangement_count_l118_11847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l118_11848

theorem equation_solutions (k r s : ℕ) (hk : k > 0) (hr : r > 0) (hs : s > 0) :
  (k^2 - 6*k + 11)^(r - 1) = (2*k - 7)^s ↔ k ∈ ({2, 3, 4, 8} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l118_11848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_diagonal_ratio_l118_11892

/-- A regular octagon -/
structure RegularOctagon where
  side : ℝ
  side_pos : side > 0

/-- Diagonal connecting two vertices skipping two sides -/
noncomputable def diagonal_skip_two (o : RegularOctagon) : ℝ := o.side * Real.sqrt 2

/-- Diagonal connecting two vertices skipping three sides -/
def diagonal_skip_three (o : RegularOctagon) : ℝ := o.side

/-- The ratio of the two diagonals -/
noncomputable def diagonal_ratio (o : RegularOctagon) : ℝ :=
  diagonal_skip_two o / diagonal_skip_three o

theorem regular_octagon_diagonal_ratio (o : RegularOctagon) :
  diagonal_ratio o = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_diagonal_ratio_l118_11892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_property_l118_11862

/-- Represents a geometric sequence with first term a₁ and common ratio r -/
structure GeometricSequence where
  a₁ : ℝ
  r : ℝ

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.a₁ * (1 - g.r^n) / (1 - g.r)

/-- Theorem: For a geometric sequence with r = 3, if S_n = 3^n - b, then b = 1 -/
theorem geometric_sequence_sum_property (g : GeometricSequence) (b : ℝ) :
  g.r = 3 → (∀ n : ℕ, geometricSum g n = 3^n - b) → b = 1 := by
  sorry

#check geometric_sequence_sum_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_property_l118_11862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_correctness_l118_11874

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ := (x^3 + 6*x^2 - 10*x + 52) / ((x-2)*(x+2)^3)

-- Define the antiderivative function
noncomputable def F (x : ℝ) : ℝ := Real.log (abs (x - 2)) + 11 / (x + 2)^2

-- State the theorem
theorem integral_correctness (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_correctness_l118_11874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l118_11819

-- Define the function f as noncomputable
noncomputable def f (t : ℝ) : ℝ := t / (1 - t^2)

-- State the theorem
theorem inverse_function_theorem (x y : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  y = f x → x = y / (1 + y^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l118_11819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l118_11878

-- Define the angle measure type
def AngleMeasure : Type := ℝ

-- Define the parallel relation between lines
def Parallel (l k : Type) : Prop := sorry

-- Define the lines l and k
variable (l k : Type)

-- Define the angles A, B, and C
variable (A B C : AngleMeasure)

-- State the theorem
theorem angle_B_measure
  (h1 : Parallel l k)
  (h2 : A = (100 : ℝ))
  (h3 : C = (60 : ℝ)) :
  B = (100 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l118_11878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_touch_distance_l118_11816

/-- 
Given a triangle ABC with sides a, b, and c, and a median BM to side AC,
this theorem states that the distance between the points where the incircles 
of triangles ABM and BCM touch BM is equal to |a-c|/2.
-/
theorem incircle_touch_distance (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  ∃ (d : ℝ), d = |a - c| / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_touch_distance_l118_11816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimizes_distance_sum_minimizing_point_on_parabola_l118_11869

/-- The function representing the sum of distances from (t, t²) to (0, 1) and (14, 46) -/
noncomputable def distance_sum (t : ℝ) : ℝ :=
  (t^2 + (t^2 - 1)^2).sqrt + ((t - 14)^2 + (t^2 - 46)^2).sqrt

/-- The theorem stating that 7/2 minimizes the distance sum -/
theorem minimizes_distance_sum :
  ∀ t : ℝ, distance_sum (7/2) ≤ distance_sum t := by
  sorry

/-- The theorem stating that the minimizing point lies on y = x² -/
theorem minimizing_point_on_parabola :
  (7/2)^2 = (7/2)^2 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimizes_distance_sum_minimizing_point_on_parabola_l118_11869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_area_l118_11844

-- Define the ellipse E
noncomputable def E (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line l
def l (x y m : ℝ) : Prop := y = x + m

-- Define the point M
def M (m : ℝ) : ℝ × ℝ := (0, 3 * m)

-- Define the area of triangle CDM
noncomputable def area_CDM (m : ℝ) : ℝ := (4 * Real.sqrt 3 / 7) * Real.sqrt ((7 - m^2) * m^2)

theorem ellipse_max_area :
  ∀ m : ℝ, area_CDM m ≤ 2 * Real.sqrt 3 ∧
  (area_CDM m = 2 * Real.sqrt 3 ↔ m = Real.sqrt 14 / 2 ∨ m = -Real.sqrt 14 / 2) :=
by
  sorry

#check ellipse_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_area_l118_11844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_puzzle_l118_11883

noncomputable def M (x y : ℝ) : ℝ := max x y

noncomputable def m (x y : ℝ) : ℝ := min x y

theorem max_min_puzzle (a b c d e : ℝ) (h : a < b ∧ b < c ∧ c < d ∧ d < e) :
  M (M a (m b c)) (m d (m a e)) = b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_puzzle_l118_11883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l118_11830

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square with side length 1 -/
def UnitSquare : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

/-- The corners of the unit square -/
def UnitSquareCorners : Set Point :=
  {⟨0, 0⟩, ⟨1, 0⟩, ⟨0, 1⟩, ⟨1, 1⟩}

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)) / 2

theorem triangle_area_bound (n : ℕ) (points : Finset Point) 
    (h : ∀ p ∈ points, p ∈ UnitSquare) (h_card : points.card = n) :
  (∃ p1 p2 p3 : Point, (p1 ∈ points ∨ p1 ∈ UnitSquareCorners) ∧
                       (p2 ∈ points ∨ p2 ∈ UnitSquareCorners) ∧
                       (p3 ∈ points ∨ p3 ∈ UnitSquareCorners) ∧
                       triangleArea p1 p2 p3 ≤ 1 / (2 * (n + 1))) ∧
  (∃ p1 p2 p3 : Point, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
                       triangleArea p1 p2 p3 ≤ 1 / (n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l118_11830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_ngon_rollable_l118_11881

/-- A regular n-gon on a plane -/
structure RegularNGon where
  n : ℕ
  center : ℝ × ℝ
  sideLength : ℝ

/-- A circle on a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Function to flip a regular n-gon over one of its sides -/
def flipNGon (ngon : RegularNGon) : RegularNGon :=
  sorry

/-- Predicate to check if a point is inside a circle -/
def isInside (point : ℝ × ℝ) (circle : Circle) : Prop :=
  sorry

/-- Theorem stating that a regular n-gon (n ≠ 3, 4, 6) can be rolled into any circle -/
theorem regular_ngon_rollable (n : ℕ) (ngon : RegularNGon) (circle : Circle) :
  n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 6 →
  ∃ (flippedNGon : RegularNGon), (∃ (k : ℕ), flippedNGon = (flipNGon^[k] ngon)) ∧
    isInside flippedNGon.center circle :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_ngon_rollable_l118_11881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_puppy_cost_l118_11863

/-- Calculates the total first-year cost for Julia's new puppy --/
noncomputable def total_first_year_cost (
  puppy_cost : ℚ)
  (dog_food_cost : ℚ)
  (treat_cost : ℚ)
  (treat_quantity : ℕ)
  (toy_box_cost : ℚ)
  (toy_box_quantity : ℕ)
  (crate_cost : ℚ)
  (bed_cost : ℚ)
  (collar_leash_cost : ℚ)
  (grooming_tools_cost : ℚ)
  (training_class_cost : ℚ)
  (training_class_quantity : ℕ)
  (discount_percentage : ℚ)
  (monthly_insurance_cost : ℚ)
  (insurance_months : ℕ) : ℚ :=
  let initial_items_cost := 
    dog_food_cost + 
    treat_cost * treat_quantity + 
    toy_box_cost * toy_box_quantity + 
    crate_cost + 
    bed_cost + 
    collar_leash_cost + 
    grooming_tools_cost + 
    training_class_cost * training_class_quantity
  let discounted_cost := initial_items_cost * (1 - discount_percentage / 100)
  let total_with_puppy := discounted_cost + puppy_cost
  let insurance_cost := monthly_insurance_cost * insurance_months
  total_with_puppy + insurance_cost

/-- Theorem stating that the total first-year cost for Julia's new puppy is $984.25 --/
theorem julia_puppy_cost : 
  total_first_year_cost 150 40 5 3 25 2 120 80 35 45 60 5 15 21 12 = 984.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_puppy_cost_l118_11863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_theorem_l118_11817

/-- A linear function y = kx + b + 2 intersecting positive x and y axes -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0
  b_positive : b > 0

/-- The area of the triangle formed by the linear function and the axes -/
noncomputable def triangleArea (f : LinearFunction) : ℝ := -(f.b + 2)^2 / (2 * f.k)

/-- The sum of the lengths of the triangle's legs -/
noncomputable def legSum (f : LinearFunction) : ℝ := -(f.b + 2) / f.k + f.b + 2

/-- Theorem: If the area equals the sum of leg lengths plus 3, 
    the minimum area is 7 + 2√10 -/
theorem min_area_theorem (f : LinearFunction) 
  (h : triangleArea f = legSum f + 3) : 
  ∃ (min_area : ℝ), 
    (∀ (g : LinearFunction), triangleArea g ≥ min_area) ∧ 
    min_area = 7 + 2 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_theorem_l118_11817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_integer_roots_l118_11834

def is_valid_pair (m n : ℕ) : Prop :=
  ∃ (a b c : ℤ), 
    a + b + c = 17 ∧
    a * b + b * c + c * a = m ∧
    a * b * c = n^2 ∧
    (∀ x : ℝ, x^3 - 17*x^2 + (m : ℝ)*x - (n^2 : ℝ) = 0 → ∃ (y : ℤ), x = y)

theorem cubic_integer_roots :
  ∀ m n : ℕ, is_valid_pair m n ↔ 
    ((m = 80 ∧ n = 10) ∨ 
     (m = 88 ∧ n = 12) ∨ 
     (m = 80 ∧ n = 8) ∨ 
     (m = 90 ∧ n = 12)) :=
by
  sorry

#check cubic_integer_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_integer_roots_l118_11834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_A_l118_11827

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n * a (n + 1) * a (n + 2) = a n + a (n + 1) + a (n + 2)

def not_one (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n * a (n + 1) ≠ 1

def initial_values (a : ℕ+ → ℝ) : Prop :=
  a 1 = 1 ∧ a 2 = 2

def general_term (A ω φ c : ℝ) (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n = A * Real.sin (ω * ↑n + φ) + c

theorem sequence_value_A 
  (a : ℕ+ → ℝ) (A ω φ c : ℝ) 
  (h_seq : sequence_property a)
  (h_not_one : not_one a)
  (h_init : initial_values a)
  (h_general : general_term A ω φ c a)
  (h_ω_pos : ω > 0)
  (h_φ_bound : abs φ < Real.pi / 2) :
  A = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_A_l118_11827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monica_expected_winnings_l118_11808

/-- Represents the outcome of a die roll -/
inductive DieOutcome
  | Prime (n : Nat)
  | CompositeOrOne
  | Other

/-- Defines the winnings for each outcome -/
def winnings (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Prime n => n
  | DieOutcome.CompositeOrOne => 0
  | DieOutcome.Other => -5

/-- Classifies a number as prime, composite/one, or other -/
def classifyOutcome (n : Nat) : DieOutcome :=
  if n ∈ [2, 3, 5, 7] then DieOutcome.Prime n
  else if n ∈ [1, 4, 6, 8] then DieOutcome.CompositeOrOne
  else DieOutcome.Other

/-- The probability of each outcome on a fair 8-sided die -/
def prob (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Prime _ => 1/8
  | DieOutcome.CompositeOrOne => 1/2
  | DieOutcome.Other => 0

/-- The expected value of Monica's winnings -/
noncomputable def expectedValue : ℚ :=
  (List.range 9).map (λ n => prob (classifyOutcome n) * winnings (classifyOutcome n))
  |> List.sum

theorem monica_expected_winnings :
  expectedValue = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monica_expected_winnings_l118_11808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_quadratic_factors_irreducible_l118_11887

noncomputable section

open Real

theorem polynomial_factorization (x : ℝ) :
  x^6 - 16 * sqrt 5 * x^3 + 64 =
  (x - (sqrt 5 + 1)) * (x^2 + x * (sqrt 5 + 1) + 6 + 2 * sqrt 5) *
  (x - (sqrt 5 - 1)) * (x^2 + x * (sqrt 5 - 1) + 6 - 2 * sqrt 5) :=
by sorry

theorem quadratic_factors_irreducible :
  ∀ (a b c : ℝ), (∀ x : ℝ, a * x^2 + b * x + c ≠ 0) →
  (∀ x : ℝ, x^2 + x * (sqrt 5 + 1) + 6 + 2 * sqrt 5 ≠ 0) ∧
  (∀ x : ℝ, x^2 + x * (sqrt 5 - 1) + 6 - 2 * sqrt 5 ≠ 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_quadratic_factors_irreducible_l118_11887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_factorial_l118_11867

theorem smallest_difference_factorial (a b c : ℕ+) : 
  a * b * c = Nat.factorial 9 → a < b → b < c → 
  ∀ x y z : ℕ+, x * y * z = Nat.factorial 9 → x < y → y < z → c - a ≤ z - x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_factorial_l118_11867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l118_11854

theorem sin_alpha_value (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.cos (α + π/3) = -4/5) : 
  Real.sin α = (3 + 4 * Real.sqrt 3) / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l118_11854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l118_11829

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define point P
def P : ℝ × ℝ := (2, -1)

-- Define that P is the midpoint of chord AB
def is_midpoint_of_chord (P : ℝ × ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ,
    circle A.1 A.2 ∧
    circle B.1 B.2 ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the equation of line AB
def line_AB (x y : ℝ) : Prop := x - y - 3 = 0

-- Theorem statement
theorem chord_equation :
  is_midpoint_of_chord P circle_eq →
  ∀ x y : ℝ, line_AB x y ↔ ∃ t : ℝ, (x, y) = (P.1 + t, P.2 + t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l118_11829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l118_11865

theorem least_number_with_remainder : 
  ∃ n : ℕ, n = 12611 ∧ 
  (∀ d : ℕ, d ∈ ({18, 24, 35, 45, 50} : Finset ℕ) → n % d = 11) ∧
  (∀ m : ℕ, m < n → ∃ d ∈ ({18, 24, 35, 45, 50} : Finset ℕ), m % d ≠ 11) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l118_11865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liu_xiang_hurdles_metrics_l118_11888

/-- Represents the 110m hurdles race parameters and performance metrics -/
structure HurdlesRace where
  totalDistance : ℝ
  startToFirstHurdle : ℝ
  lastHurdleToFinish : ℝ
  numHurdles : ℕ
  bestFirstSegmentTime : ℝ
  bestLastSegmentTime : ℝ
  fastestHurdleCycleTime : ℝ

/-- Calculates the distance between consecutive hurdles and the theoretical best time -/
noncomputable def calculateHurdlesMetrics (race : HurdlesRace) : ℝ × ℝ :=
  let distanceBetweenHurdles := (race.totalDistance - race.startToFirstHurdle - race.lastHurdleToFinish) / (race.numHurdles.pred : ℝ)
  let theoreticalBestTime := race.bestFirstSegmentTime + (race.numHurdles.pred : ℝ) * race.fastestHurdleCycleTime + race.bestLastSegmentTime
  (distanceBetweenHurdles, theoreticalBestTime)

theorem liu_xiang_hurdles_metrics :
  let race : HurdlesRace := {
    totalDistance := 110
    startToFirstHurdle := 13.72
    lastHurdleToFinish := 14.02
    numHurdles := 10
    bestFirstSegmentTime := 2.5
    bestLastSegmentTime := 1.4
    fastestHurdleCycleTime := 0.96
  }
  let (distanceBetweenHurdles, theoreticalBestTime) := calculateHurdlesMetrics race
  abs (distanceBetweenHurdles - 9.14) < 0.01 ∧ abs (theoreticalBestTime - 12.54) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liu_xiang_hurdles_metrics_l118_11888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_major_axis_l118_11825

noncomputable section

-- Define the ellipse parameters
def b : ℝ := 4

-- Define the eccentricity
def e : ℝ := 3/4

-- Theorem statement
theorem ellipse_semi_major_axis :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 → 
    e = (Real.sqrt (a^2 - b^2))/a) ∧
  a = 7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_major_axis_l118_11825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l118_11824

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3^x / (3^x + 1) - a

theorem odd_function_properties (a : ℝ) 
  (h : ∀ x : ℝ, f a (-x) = -(f a x)) :
  (a = 1/2) ∧ 
  (∀ x y : ℝ, x < y → f a x < f a y) ∧ 
  (∀ m : ℝ, (∀ x : ℝ, f a x < m - 1) ↔ m ≥ 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l118_11824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gauss_function_root_floor_l118_11835

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem gauss_function_root_floor :
  ∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ = 0 ∧ floor x₀ = 2 := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma f_continuous : Continuous f := by
  sorry

lemma f_two_neg : f 2 < 0 := by
  sorry

lemma f_three_pos : f 3 > 0 := by
  sorry

lemma f_monotone : StrictMono f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gauss_function_root_floor_l118_11835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_bound_l118_11813

theorem function_bound (a b : ℝ) (h : a * b ≠ 0) :
  let f : ℝ → ℝ := fun x ↦ a / x + b / (x^2)
  let m : ℝ := max (abs a) (max (abs b) 1)
  ∀ x : ℝ, abs x > m → abs (f x) < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_bound_l118_11813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corrected_mean_l118_11871

theorem corrected_mean (initial_mean : ℝ) (initial_count : ℕ) 
  (removed_score1 removed_score2 : ℝ) : 
  initial_mean = 42 →
  initial_count = 60 →
  removed_score1 = 50 →
  removed_score2 = 60 →
  abs ((initial_mean * initial_count - (removed_score1 + removed_score2)) / (initial_count - 2) - 41.55) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_corrected_mean_l118_11871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_2008_is_sum_of_two_squares_l118_11890

def x : ℕ → ℚ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => (x (n + 2))^2 / x (n + 1) + 3 / x (n + 1)

theorem x_2008_is_sum_of_two_squares :
  ∃ (a b : ℤ), x 2008 = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_2008_is_sum_of_two_squares_l118_11890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sphere_volume_equals_target_l118_11845

/-- Regular triangular pyramid with specific properties -/
structure RegularTriangularPyramid where
  height : ℝ
  base_area : ℝ
  lateral_face_area : ℝ
  base_lateral_ratio : base_area = (1/2) * lateral_face_area

/-- Sequence of spheres in the pyramid -/
noncomputable def sphere_sequence (p : RegularTriangularPyramid) : ℕ → ℝ
  | 0 => 1  -- Assuming the first sphere has radius 1 (we'll adjust this later)
  | n + 1 => (1/2) * sphere_sequence p n

/-- Total volume of all spheres in the sequence -/
noncomputable def total_sphere_volume (p : RegularTriangularPyramid) : ℝ :=
  (4/3) * Real.pi * ∑' n, (sphere_sequence p n)^3

/-- Main theorem: The total volume of spheres equals (686/327)π -/
theorem total_sphere_volume_equals_target (p : RegularTriangularPyramid)
  (h : p.height = 70) :
  total_sphere_volume p = (686/327) * Real.pi := by
  sorry

#check total_sphere_volume_equals_target

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sphere_volume_equals_target_l118_11845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contains_all_nonnegative_integers_l118_11899

theorem contains_all_nonnegative_integers (S : Set ℕ) 
  (h1 : ∃ a b, a ∈ S ∧ b ∈ S ∧ a > 1 ∧ b > 1 ∧ Nat.gcd a b = 1)
  (h2 : ∀ x y, x ∈ S → y ∈ S → y ≠ 0 → x * y ∈ S ∧ x % y ∈ S) :
  ∀ n : ℕ, n ∈ S :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contains_all_nonnegative_integers_l118_11899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robinson_can_escape_l118_11826

/-- Represents the lake and the speeds of Robinson and the cannibal -/
structure EscapeScenario where
  R : ℝ  -- Radius of the circular lake
  v_swim : ℝ  -- Robinson's swimming speed
  h_R_pos : 0 < R  -- The radius is positive
  h_v_swim_pos : 0 < v_swim  -- The swimming speed is positive

/-- Defines the existence of an escape strategy for Robinson -/
def has_escape_strategy (scenario : EscapeScenario) : Prop :=
  ∃ r : ℝ, 
    scenario.R / 4 > r ∧ 
    r > (1 - Real.pi / 4) * scenario.R ∧
    scenario.v_swim / r > (4 * scenario.v_swim) / scenario.R ∧
    Real.pi * scenario.R / (4 * scenario.v_swim) > (scenario.R - r) / scenario.v_swim

/-- Theorem stating that Robinson can always escape -/
theorem robinson_can_escape (scenario : EscapeScenario) : 
  has_escape_strategy scenario := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_robinson_can_escape_l118_11826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_six_l118_11866

theorem sum_equals_six (x n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_eq : 2 * x^3 + x^2 + 10 * x + 5 = 2 * p^(n:ℕ)) 
  (h_pos : x > 0 ∧ n > 0) : 
  x + n + p = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_six_l118_11866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l118_11814

/-- Definition of the sequence a_n -/
noncomputable def a (n : ℕ+) : ℝ := n

/-- Definition of the sum S_n -/
noncomputable def S (n : ℕ+) : ℝ := (1/2) * (a n)^2 + (1/2) * (a n)

/-- Definition of the function f -/
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + (1/2) * x

/-- Definition of the sequence b_n -/
noncomputable def b (n : ℕ+) : ℝ := (2 : ℝ)^(n : ℝ) * (2 * (a n) - 1)

/-- Definition of the sum T_n -/
noncomputable def T (n : ℕ+) : ℝ := 6 + (2 * n - 3) * (2 : ℝ)^((n + 1) : ℝ)

/-- Definition of the sequence c_n -/
noncomputable def c (n : ℕ+) : ℝ := (4 * n - 6) / (T n - 6) - 1 / ((a n) * (a (n + 1)))

/-- Sum of the first n terms of c_n -/
noncomputable def c_sum (n : ℕ+) : ℝ := (Finset.range n).sum (λ i => c ⟨i + 1, Nat.succ_pos i⟩)

/-- The main theorem -/
theorem max_a_value (a : ℝ) :
  (∀ n : ℕ+, ∃ x : ℝ, x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ) ∧ c_sum n ≤ f x - a) →
  a ≤ 19/80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l118_11814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_radii_order_l118_11857

-- Define the circles and their properties
noncomputable def circle_X_radius : ℝ := 2 * Real.pi
noncomputable def circle_Y_circumference : ℝ := 12 * Real.pi
noncomputable def circle_Z_area : ℝ := 16 * Real.pi
noncomputable def half_circle_W_area : ℝ := 8 * Real.pi

-- Define functions to calculate radii
noncomputable def radius_from_circumference (c : ℝ) : ℝ := c / (2 * Real.pi)
noncomputable def radius_from_area (a : ℝ) : ℝ := Real.sqrt (a / Real.pi)
noncomputable def radius_from_half_circle_area (a : ℝ) : ℝ := Real.sqrt (2 * a / Real.pi)

-- Calculate radii for each circle
noncomputable def circle_Y_radius : ℝ := radius_from_circumference circle_Y_circumference
noncomputable def circle_Z_radius : ℝ := radius_from_area circle_Z_area
noncomputable def half_circle_W_radius : ℝ := radius_from_half_circle_area half_circle_W_area

-- Theorem statement
theorem circles_radii_order :
  circle_Z_radius ≤ half_circle_W_radius ∧
  half_circle_W_radius ≤ circle_Y_radius ∧
  circle_Y_radius ≤ circle_X_radius :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_radii_order_l118_11857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BXC_eq_1200_div_11_l118_11840

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  -- Base lengths
  ab : ℝ
  cd : ℝ
  -- Total area
  area : ℝ
  -- Conditions
  ab_positive : 0 < ab
  cd_positive : 0 < cd
  area_positive : 0 < area

/-- Calculates the area of triangle BXC in the trapezoid -/
noncomputable def area_BXC (t : Trapezoid) : ℝ :=
  (8 / 11) * (t.area - (1 / 2) * t.cd * ((2 * t.area) / (t.ab + t.cd)))

/-- Theorem stating the area of triangle BXC in the given trapezoid -/
theorem area_BXC_eq_1200_div_11 (t : Trapezoid) 
    (h1 : t.ab = 15) (h2 : t.cd = 40) (h3 : t.area = 550) : 
    area_BXC t = 1200 / 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BXC_eq_1200_div_11_l118_11840

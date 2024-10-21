import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_four_l1118_111853

-- Define the total cost function
def G (x : ℝ) : ℝ := 2.8 + x

-- Define the sales revenue function
noncomputable def R (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x else 11

-- Define the profit function
noncomputable def f (x : ℝ) : ℝ := R x - G x

-- Theorem statement
theorem max_profit_at_four :
  ∃ (max_profit : ℝ), max_profit = f 4 ∧
  ∀ (x : ℝ), f x ≤ max_profit ∧
  max_profit = 3.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_four_l1118_111853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_adorable_seven_digit_integer_l1118_111837

def is_adorable (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ),
    n = d₁ * 1000000 + d₂ * 100000 + d₃ * 10000 + d₄ * 1000 + d₅ * 100 + d₆ * 10 + d₇ ∧
    Finset.toSet {d₁, d₂, d₃, d₄, d₅, d₆, d₇} = Finset.toSet {2, 4, 6, 8, 10, 12, 14} ∧
    (∀ k : ℕ, k ∈ Finset.range 7 →
      (d₁ * 10^(k-1) + d₂ * 10^(k-2) + d₃ * 10^(k-3) + d₄ * 10^(k-4) + 
       d₅ * 10^(k-5) + d₆ * 10^(k-6) + d₇ * 10^(k-7)) % (k+1) = 0)

theorem unique_adorable_seven_digit_integer :
  ∃! n : ℕ, is_adorable n ∧ n = 4261210814 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_adorable_seven_digit_integer_l1118_111837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximize_segment_area_l1118_111812

noncomputable def optimal_chord_length : ℝ := 16 * Real.pi / (16 + Real.pi^2)

-- Helper function to calculate the area of the segment outside the inscribed circle
noncomputable def area_of_segment (R chord_length : ℝ) : ℝ :=
  let α := Real.arcsin (chord_length / (2 * R))
  let segment_area := R^2 * (Real.pi - (α - Real.sin α * Real.cos α))
  let inscribed_circle_area := Real.pi * (R * (1 + Real.cos α))^2 / 4
  segment_area - inscribed_circle_area

theorem maximize_segment_area (R : ℝ) (h : R = 1) :
  ∃ (chord_length : ℝ),
    chord_length = optimal_chord_length ∧
    ∀ (other_chord : ℝ),
      area_of_segment R other_chord ≤ area_of_segment R chord_length :=
by sorry

#check maximize_segment_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximize_segment_area_l1118_111812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_digit_is_4_l1118_111819

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def expression (n : ℕ) : ℚ :=
  (factorial 3 * factorial 5 + factorial 4 * factorial 3) / n

def has_100th_digit_4 (q : ℚ) : Prop :=
  ∃ (k m : ℕ), q = (k * 10^99 + 4 * 10^98 + m) / 10^99 ∧ m < 10^98

theorem hundredth_digit_is_4 :
  has_100th_digit_4 (expression 1944) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_digit_is_4_l1118_111819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l1118_111882

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0
  | n + 1 => (10 / 6) * sequence_a n + (8 / 6) * Real.sqrt (4^n - (sequence_a n)^2)

theorem a_5_value : sequence_a 5 = 32000 / 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l1118_111882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_theorem_l1118_111895

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  stillWater : ℝ
  stream : ℝ

/-- Given the downstream and upstream distances and time, calculates the swimmer's speed. -/
noncomputable def calculateSwimmerSpeed (downstreamDistance upstreamDistance time : ℝ) : SwimmerSpeed :=
  { stillWater := (downstreamDistance + upstreamDistance) / (2 * time),
    stream := (downstreamDistance - upstreamDistance) / (2 * time) }

/-- Theorem: A swimmer who covers 91 km downstream and 21 km upstream in 7 hours each has a speed of 8 km/h in still water. -/
theorem swimmer_speed_theorem (s : SwimmerSpeed) :
  s = calculateSwimmerSpeed 91 21 7 →
  s.stillWater = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_theorem_l1118_111895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_installation_equation_l1118_111866

/-- Represents the number of devices installed per hour by person B -/
def x : ℝ := sorry

/-- The number of devices installed by each person -/
def devices_installed : ℕ := 10

/-- The difference in devices installed per hour between A and B -/
def installation_rate_difference : ℕ := 2

/-- The time difference in completion between A and B -/
def time_difference : ℝ := 1

/-- Theorem stating that the given equation correctly represents the installation situation -/
theorem installation_equation : 
  (devices_installed : ℝ) / x - (devices_installed : ℝ) / (x + installation_rate_difference) = time_difference := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_installation_equation_l1118_111866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1118_111847

/-- An ellipse with center at the origin, axis of symmetry coinciding with coordinate axes,
    passing through (4,0), and eccentricity √3/2 -/
structure Ellipse where
  -- Point on ellipse
  point : ℝ × ℝ
  -- Eccentricity
  e : ℝ
  -- Conditions
  center_origin : point.1^2 / 16 + point.2^2 / 4 = 1 ∨ point.2^2 / 64 + point.1^2 / 16 = 1
  passes_through_4_0 : point = (4, 0)
  eccentricity : e = Real.sqrt 3 / 2

/-- The standard equation of the ellipse is either (x²/16) + (y²/4) = 1 or (y²/64) + (x²/16) = 1 -/
theorem ellipse_equation (E : Ellipse) : 
  (∀ x y : ℝ, x^2 / 16 + y^2 / 4 = 1) ∨ 
  (∀ x y : ℝ, y^2 / 64 + x^2 / 16 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1118_111847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polygon_with_perimeter_24_area_16_l1118_111886

/-- A polygon represented by its sides --/
structure Polygon where
  sides : List ℝ
  positive_sides : ∀ s ∈ sides, s > 0

/-- The perimeter of a polygon --/
def perimeter (p : Polygon) : ℝ := p.sides.sum

/-- The area of a polygon --/
noncomputable def area (p : Polygon) : ℝ := sorry

/-- Theorem: There exists a polygon with perimeter 24 and area 16 --/
theorem exists_polygon_with_perimeter_24_area_16 :
  ∃ (p : Polygon), perimeter p = 24 ∧ area p = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polygon_with_perimeter_24_area_16_l1118_111886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_negative_iff_a_in_range_l1118_111899

noncomputable def f (x a : ℝ) : ℝ := x^2 - Real.log (x + 1) / Real.log a - 4*x + 4

theorem function_negative_iff_a_in_range :
  ∀ a : ℝ, a > 0 →
  (∀ x : ℝ, x > 1 ∧ x < 2 → f x a < 0) ↔ (a > 1 ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_negative_iff_a_in_range_l1118_111899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_for_inequality_l1118_111861

theorem smallest_c_for_inequality : 
  ∃ c : ℝ, c > 0 ∧ 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → (x * y) ^ (1/3) + c * abs (x - y) ≥ (x + y) / 3) ∧
  (∀ c' : ℝ, c' > 0 → 
    (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → (x * y) ^ (1/3) + c' * abs (x - y) ≥ (x + y) / 3) → 
    c' ≥ c) ∧
  c = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_for_inequality_l1118_111861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_first_and_last_l1118_111824

def numbers : List Int := [-3, 2, 5, 10, 15]

def is_valid_arrangement (arr : List Int) : Prop :=
  arr.length = 5 ∧
  arr.toFinset = numbers.toFinset ∧
  (∃ i, i ∈ [2, 3, 4] ∧ arr[i]! = 15) ∧
  (∃ i, i ∈ [1, 2, 3] ∧ arr[i]! = -3) ∧
  (∃ i, i ∈ [1, 2, 3, 4] ∧ arr[i]! = 10) ∧
  arr[4]! ≠ 5

theorem average_of_first_and_last (arr : List Int) :
  is_valid_arrangement arr → (arr[0]! + arr[4]!) / 2 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_first_and_last_l1118_111824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_consumption_rate_l1118_111846

/-- Represents the typing speed in words per minute -/
def typing_speed : ℕ := 50

/-- Represents the number of pages in the paper -/
def num_pages : ℕ := 5

/-- Represents the number of words per page -/
def words_per_page : ℕ := 400

/-- Represents the total amount of water needed for the entire paper in ounces -/
def total_water : ℝ := 10

/-- Calculates the water consumption rate in ounces per hour -/
noncomputable def water_per_hour : ℝ :=
  let total_words := num_pages * words_per_page
  let typing_time_hours := (total_words : ℝ) / (typing_speed * 60 : ℝ)
  total_water / typing_time_hours

/-- Theorem stating that the water consumption rate is approximately 15 ounces per hour -/
theorem water_consumption_rate : 
  |water_per_hour - 15| < 0.1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_consumption_rate_l1118_111846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_three_zeros_iff_k_range_l1118_111893

open Real

/-- The function f(x) = 2x³ - 3x² + 1 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

/-- The function g(x) = kx + 1 - ln x -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := k * x + 1 - Real.log x

/-- The function h(x) = min{f(x), g(x)} -/
noncomputable def h (k : ℝ) (x : ℝ) : ℝ := min (f x) (g k x)

/-- The number of zeros of h(x) in (0, +∞) -/
noncomputable def num_zeros (k : ℝ) : ℕ := sorry

theorem h_three_zeros_iff_k_range :
  ∀ k : ℝ, num_zeros k = 3 ↔ 0 < k ∧ k < Real.exp (-2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_three_zeros_iff_k_range_l1118_111893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_through_origin_l1118_111889

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := Real.log x + b

theorem common_tangent_through_origin (b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧
    (deriv f x₁ : ℝ) * x₁ = f x₁ ∧
    (deriv (g b) x₂ : ℝ) * x₂ = g b x₂ ∧
    deriv f x₁ = deriv (g b) x₂) →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_through_origin_l1118_111889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_of_omega_l1118_111894

noncomputable def ω : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I

theorem sum_of_powers_of_omega : 1 + ω + ω^2 + ω^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_of_omega_l1118_111894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1118_111896

/-- Represents the time (in hours) it takes to fill a cistern -/
noncomputable def fill_time_with_leak : ℝ := 12

/-- Represents the time (in hours) it takes for the leak to empty a full cistern -/
noncomputable def empty_time : ℝ := 24

/-- Calculates the time (in hours) it takes to fill the cistern without the leak -/
noncomputable def fill_time_without_leak : ℝ :=
  let leak_rate : ℝ := 1 / empty_time
  let fill_rate : ℝ := (1 / fill_time_with_leak) + leak_rate
  1 / fill_rate

theorem cistern_fill_time :
  fill_time_without_leak = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1118_111896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_markup_theorem_l1118_111848

/-- Calculate the percentage markup given the selling price and cost price -/
noncomputable def percentage_markup (selling_price : ℝ) (cost_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that the percentage markup is 100% for the given prices -/
theorem furniture_markup_theorem (selling_price cost_price : ℝ) 
  (h1 : selling_price = 1000)
  (h2 : cost_price = 500) :
  percentage_markup selling_price cost_price = 100 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_markup_theorem_l1118_111848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_image_l1118_111887

noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

theorem dilation_image :
  let center := (-2 : ℝ) + 2*I
  let scale := (4 : ℝ)
  let z := (1 : ℝ) + 2*I
  dilation center scale z = (10 : ℝ) + 2*I := by
  -- Unfold the definition of dilation
  unfold dilation
  -- Simplify the expression
  simp [Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_image_l1118_111887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_maximized_l1118_111851

/-- The slant height of the cone -/
noncomputable def slant_height : ℝ := 18

/-- The height of the cone -/
noncomputable def cone_height : ℝ := 6 * Real.sqrt 3

/-- The volume of the cone as a function of its height -/
noncomputable def volume (h : ℝ) : ℝ := (1/3) * Real.pi * (324 * h - h^3)

/-- The derivative of the volume function -/
noncomputable def volume_derivative (h : ℝ) : ℝ := Real.pi * (108 - h^2)

theorem cone_volume_maximized :
  (∀ h : ℝ, 0 < h ∧ h < slant_height → volume h ≤ volume cone_height) ∧
  volume_derivative cone_height = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_maximized_l1118_111851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_first_group_is_13_l1118_111850

/-- The number of days it takes the first group to complete the work -/
def days_first_group : ℕ := 80

/-- The number of men in the second group -/
def men_second_group : ℕ := 20

/-- The number of days it takes the second group to complete the work -/
def days_second_group : ℕ := 52

/-- The amount of work is the product of the number of men and the number of days -/
def work_product (men : ℕ) (days : ℕ) : ℕ := men * days

/-- The theorem stating that the number of men in the first group is 13 -/
theorem men_first_group_is_13 : 
  ∃ (men : ℕ), work_product men days_first_group = work_product men_second_group days_second_group ∧ men = 13 := by
  sorry

#eval work_product 13 days_first_group
#eval work_product men_second_group days_second_group

end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_first_group_is_13_l1118_111850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_distance_equals_pyramid_height_l1118_111878

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Calculates the area of a triangle given its three sides -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Placeholder for the perpendicular_distance function -/
noncomputable def perpendicular_distance (p : Point3D) (plane : Point3D → Point3D → Point3D → Prop) : ℝ :=
  sorry

/-- Placeholder for the plane_through function -/
def plane_through (A B C : Point3D) : Point3D → Point3D → Point3D → Prop :=
  fun _ _ _ => True

/-- Theorem: The perpendicular distance from D to plane ABC is equal to the height of a pyramid -/
theorem perpendicular_distance_equals_pyramid_height 
  (D : Point3D) 
  (A : Point3D) 
  (B : Point3D) 
  (C : Point3D) 
  (h : D = ⟨0, 0, 0⟩) 
  (hA : A = ⟨5, 0, 0⟩) 
  (hB : B = ⟨0, 6, 0⟩) 
  (hC : C = ⟨0, 0, 4⟩) :
  ∃ (dist : ℝ), 
    dist = (3 * 40) / triangleArea (distance A B) (distance A C) (distance B C) ∧
    dist = (perpendicular_distance D (plane_through A B C)) :=
by sorry

#check perpendicular_distance_equals_pyramid_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_distance_equals_pyramid_height_l1118_111878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l1118_111874

/-- Represents the length of a train in meters. -/
noncomputable def train_length : ℝ := 120

/-- Represents the time taken by the train to cross a telegraph post in seconds. -/
noncomputable def crossing_time : ℝ := 6

/-- Represents the speed of the train in km/hr. -/
noncomputable def train_speed : ℝ := 72

/-- Conversion factor from km/hr to m/s. -/
noncomputable def km_hr_to_m_s : ℝ := 5 / 18

theorem train_length_proof : 
  train_length = (train_speed * km_hr_to_m_s) * crossing_time :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l1118_111874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_property_l1118_111810

theorem sum_digits_property (k : ℕ) :
  ∃ (n : ℕ) (d : ℕ), n % 9 = 1 ∧ d < 10^k ∧ n * 10^k + d = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_property_l1118_111810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prosecutor_conclusion_l1118_111809

variable (X Y : Prop)

theorem prosecutor_conclusion (h1 : X ∨ Y) (h2 : ¬X) : Y :=
  by
    cases h1
    . contradiction
    . assumption


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prosecutor_conclusion_l1118_111809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_PA_equation_dot_product_bounds_l1118_111868

-- Define the ellipse parameters
noncomputable def a : ℝ := Real.sqrt 15
noncomputable def b : ℝ := Real.sqrt 10
noncomputable def c : ℝ := Real.sqrt 5

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 8)^2 + (y - 6)^2 = 4

-- Define the conditions
axiom a_gt_b : a > b
axiom b_gt_zero : b > 0
axiom ellipse_point : ellipse_eq (-3) 2
axiom ellipse_eccentricity : c / a = Real.sqrt 3 / 3
axiom circle_O_diameter : 2 * b = Real.sqrt 10

-- Define the theorems to be proved
theorem ellipse_equation : a^2 = 15 ∧ b^2 = 10 := by sorry

theorem line_PA_equation (k : ℝ) : 
  (k = 1/3 ∨ k = 13/9) → 
  ∀ x y, y - 6 = k * (x - 8) ∧ (x - 8)^2 + (y - 6)^2 = 4 → 
  (x - 3*y + 10 = 0 ∨ 13*x - 9*y - 50 = 0) := by sorry

theorem dot_product_bounds : 
  ∃ (OA OB : ℝ × ℝ),
    (∀ P : ℝ × ℝ, circle_M P.1 P.2 → 
      let PA := (P.1 - OA.1, P.2 - OA.2)
      let PB := (P.1 - OB.1, P.2 - OB.2)
      PA.1^2 + PA.2^2 = 10 ∧ PB.1^2 + PB.2^2 = 10) →
    -55/8 ≥ (OA.1 * OB.1 + OA.2 * OB.2) ∧ 
    (OA.1 * OB.1 + OA.2 * OB.2) ≥ -155/18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_PA_equation_dot_product_bounds_l1118_111868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1118_111838

/-- Circle C₁ with center (-1, -4) and radius 4 -/
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + (y + 4)^2 = 16

/-- Circle C₂ with center (2, -2) and radius 3 -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 9

/-- The distance between the centers of C₁ and C₂ -/
noncomputable def distance_between_centers : ℝ := Real.sqrt 13

/-- Theorem stating that C₁ and C₂ are intersecting -/
theorem circles_intersect : 
  (4 : ℝ) - 3 < distance_between_centers ∧ 
  distance_between_centers < (4 : ℝ) + 3 := by
  sorry

#check circles_intersect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1118_111838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_tax_calculation_l1118_111862

/-- Represents the percentage tax on clothing -/
def clothing_tax_percentage (x : ℝ) : Prop := x = 4

/-- Represents the given conditions of Rose's shopping trip -/
structure ShoppingTrip where
  clothing_percent : ℝ
  food_percent : ℝ
  other_percent : ℝ
  other_tax_rate : ℝ
  total_tax_rate : ℝ

/-- The theorem statement -/
theorem shopping_tax_calculation (trip : ShoppingTrip) 
  (h1 : trip.clothing_percent = 50)
  (h2 : trip.food_percent = 20)
  (h3 : trip.other_percent = 30)
  (h4 : trip.clothing_percent + trip.food_percent + trip.other_percent = 100)
  (h5 : trip.other_tax_rate = 8)
  (h6 : trip.total_tax_rate = 4.4) :
  clothing_tax_percentage 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_tax_calculation_l1118_111862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_equality_expression_two_equality_l1118_111801

-- Expression 1
theorem expression_one_equality : 
  Real.cos (-11 * π / 6) + Real.sin (12 * π / 5) * Real.tan (6 * π) = Real.sqrt 3 / 2 := by
  sorry

-- Expression 2
theorem expression_two_equality : 
  Real.sin (420 * π / 180) * Real.cos (750 * π / 180) + 
  Real.sin (-330 * π / 180) * Real.cos (-660 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_equality_expression_two_equality_l1118_111801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extrema_bounded_difference_area_for_three_tangents_l1118_111802

def f (x : ℝ) := x^3 - 3*x

theorem odd_function_extrema (b c d : ℝ) :
  (∀ x, f (-x) = -f x) →
  (∃ y, ∀ x, f x ≤ f y ∧ f (-y) ≤ f x) →
  (∃ g : ℝ → ℝ, g = λ x ↦ x^3 + b*x^2 + c*x + d) →
  (g = f) :=
sorry

theorem bounded_difference :
  ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → |f x₁ - f x₂| ≤ 4 :=
sorry

theorem area_for_three_tangents :
  ∃ A : ℝ, A = 8 ∧
  ∀ m n : ℝ, |m| < 2 →
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧
    (n - f x₁ = (deriv f x₁) * (m - x₁)) ∧
    (n - f x₂ = (deriv f x₂) * (m - x₂)) ∧
    (n - f x₃ = (deriv f x₃) * (m - x₃))) →
  A = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extrema_bounded_difference_area_for_three_tangents_l1118_111802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_area_theorem_l1118_111890

/-- Calculates the total area of a plot consisting of a rectangle and an attached triangle -/
noncomputable def totalArea (rectangleArea : ℝ) (triangleHeight : ℝ) : ℝ :=
  let rectangleLength := Real.sqrt (3 * rectangleArea)
  let rectangleBreadth := rectangleLength / 3
  let triangleArea := (1 / 2) * rectangleBreadth * triangleHeight
  rectangleArea + triangleArea

theorem plot_area_theorem (rectangleArea triangleHeight : ℝ) 
    (h1 : rectangleArea = 972)
    (h2 : triangleHeight = 7) : 
  totalArea rectangleArea triangleHeight = 1035 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_area_theorem_l1118_111890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1118_111817

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the problem setup
def problem_setup (ABCD AEFG : Square) : Prop :=
  -- Squares are in the same plane (implicitly true by using the same coordinate system)
  -- Squares have the same orientation (we'll assume this without explicit definition)
  true

-- Define a line passing through two points
def Line (P Q : Point) : Set Point :=
  {R : Point | ∃ t : ℝ, R.x = P.x + t * (Q.x - P.x) ∧ R.y = P.y + t * (Q.y - P.y)}

-- State the theorem
theorem intersection_point (ABCD AEFG : Square) 
  (h : problem_setup ABCD AEFG) : 
  ∃ M : Point, 
    M ∈ Line ABCD.B AEFG.A ∧ 
    M ∈ Line ABCD.C AEFG.A ∧ 
    M ∈ Line ABCD.D AEFG.A :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1118_111817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_distance_l1118_111898

/-- The distance between Chennai and Hyderabad in miles -/
noncomputable def distance_between_cities : ℝ := 350

/-- David's speed in miles per hour -/
noncomputable def david_speed : ℝ := 50

/-- Lewis's speed in miles per hour -/
noncomputable def lewis_speed : ℝ := 70

/-- The distance from Chennai where David and Lewis meet -/
noncomputable def meeting_point : ℝ := distance_between_cities * david_speed / (lewis_speed + david_speed)

/-- Theorem stating that David and Lewis meet approximately 145.83 miles from Chennai -/
theorem meeting_point_distance : 
  ∃ ε > 0, |meeting_point - 145.83| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_distance_l1118_111898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_hyperbola_l1118_111823

/-- Represents the equation of a conic section -/
structure ConicSection where
  x_coeff : ℝ
  y_coeff : ℝ
  rhs : ℝ

/-- Checks if a conic section is a hyperbola with foci on the x-axis -/
def is_hyperbola_x_axis (c : ConicSection) : Prop :=
  c.x_coeff > 0 ∧ c.y_coeff < 0

/-- Constructs a conic section from the given equation -/
noncomputable def make_conic_section (m : ℝ) : ConicSection :=
  { x_coeff := 1 / (m - 1)
  , y_coeff := 1 / (4 - m)
  , rhs := 1 }

/-- Theorem: m > 5 is a sufficient condition for the equation to represent a hyperbola with foci on the x-axis -/
theorem sufficient_condition_hyperbola (m : ℝ) : 
  m > 5 → is_hyperbola_x_axis (make_conic_section m) :=
by
  intro h
  unfold is_hyperbola_x_axis make_conic_section
  apply And.intro
  · -- Prove x_coeff > 0
    sorry
  · -- Prove y_coeff < 0
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_hyperbola_l1118_111823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tian_du_peak_temp_second_mountain_height_l1118_111844

-- Define the temperature drop rate
noncomputable def temp_drop_rate : ℝ := 0.6

-- Define the height increment for which the temperature drop rate applies
noncomputable def height_increment : ℝ := 100

-- Function to calculate temperature at a given height
noncomputable def temp_at_height (base_temp : ℝ) (height : ℝ) : ℝ :=
  base_temp - (height / height_increment) * temp_drop_rate

-- Function to calculate height given temperature difference
noncomputable def height_from_temp_diff (temp_diff : ℝ) : ℝ :=
  (temp_diff / temp_drop_rate) * height_increment

-- Theorem for Tian Du Peak
theorem tian_du_peak_temp :
  let base_temp : ℝ := 18
  let peak_height : ℝ := 1800
  temp_at_height base_temp peak_height = 7.2 := by sorry

-- Theorem for the second mountain
theorem second_mountain_height :
  let foot_temp : ℝ := 10
  let peak_temp : ℝ := -8
  height_from_temp_diff (foot_temp - peak_temp) = 3000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tian_du_peak_temp_second_mountain_height_l1118_111844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_bridge_time_l1118_111892

/-- Calculates the time (in seconds) for a train to pass a bridge -/
noncomputable def time_to_pass_bridge (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 360m traveling at 60 km/h takes approximately 30 seconds to pass a bridge of length 140m -/
theorem train_passing_bridge_time :
  let train_length : ℝ := 360
  let bridge_length : ℝ := 140
  let train_speed_kmh : ℝ := 60
  let calculated_time := time_to_pass_bridge train_length bridge_length train_speed_kmh
  abs (calculated_time - 30) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_bridge_time_l1118_111892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_l1118_111835

-- Define the parabola
def Parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus of the parabola
def Focus : ℝ × ℝ := (0, 1)

-- Define a point on the parabola
def PointOnParabola (p : ℝ × ℝ) : Prop := Parabola p.1 p.2

-- Define the vector from focus to a point
def VectorFromFocus (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - Focus.1, p.2 - Focus.2)

-- Define the distance from focus to a point
noncomputable def DistanceFromFocus (p : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - Focus.1)^2 + (p.2 - Focus.2)^2)

-- Theorem statement
theorem parabola_focus_property (A B C : ℝ × ℝ) 
  (hA : PointOnParabola A) (hB : PointOnParabola B) (hC : PointOnParabola C)
  (h_sum : VectorFromFocus A + VectorFromFocus B + VectorFromFocus C = (0, 0)) :
  DistanceFromFocus A + DistanceFromFocus B + DistanceFromFocus C = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_l1118_111835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l1118_111808

-- Define the line
noncomputable def line (x y : ℝ) : Prop := y = -2/3 * x + 4

-- Define the vector parameterizations
noncomputable def param_A (t : ℝ) : ℝ × ℝ := (3 + 3*t, 4 - 2*t)
noncomputable def param_B (t : ℝ) : ℝ × ℝ := (1.5*t, 4 - t)
noncomputable def param_C (t : ℝ) : ℝ × ℝ := (1 - 6*t, 3.33 - 4*t)
noncomputable def param_D (t : ℝ) : ℝ × ℝ := (5 + 1.5*t, 2/3 - t)
noncomputable def param_E (t : ℝ) : ℝ × ℝ := (-6 + 9*t, 8 - 6*t)

-- Theorem stating which parameterizations are valid
theorem valid_parameterizations :
  (∀ t, line (param_B t).1 (param_B t).2) ∧
  (∀ t, line (param_D t).1 (param_D t).2) ∧
  (∃ t, ¬ line (param_A t).1 (param_A t).2) ∧
  (∃ t, ¬ line (param_C t).1 (param_C t).2) ∧
  (∃ t, ¬ line (param_E t).1 (param_E t).2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l1118_111808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_baking_days_l1118_111834

/-- The number of days Sara bakes cakes -/
def days_baking : ℕ := 9

/-- The number of cakes Sara bakes per day -/
def cakes_per_day : ℕ := 10

/-- The number of cakes Carol eats -/
def cakes_eaten : ℕ := 12

/-- The number of cans of frosting needed for the remaining cakes -/
def frosting_cans : ℕ := 76

/-- Each cake requires one can of frosting -/
axiom one_can_per_cake : frosting_cans = cakes_per_day * days_baking - cakes_eaten

theorem sara_baking_days : days_baking = 9 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_baking_days_l1118_111834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_isosceles_right_triangle_l1118_111841

/-- Given an isosceles right triangle with leg length a and a circle that
    touches one leg, passes through the opposite acute angle vertex, and
    has its center on the hypotenuse, the radius of the circle is a(2 - √2). -/
theorem circle_radius_isosceles_right_triangle (a : ℝ) (h : a > 0) :
  ∃ (r : ℝ), r = a * (2 - Real.sqrt 2) ∧
  ∃ (triangle : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)),
    -- The triangle is isosceles and right-angled
    (∃ A B C : ℝ × ℝ, triangle = {A, B, C} ∧
      dist A B = dist A C ∧ dist A B = a ∧ dist B C = a * Real.sqrt 2) ∧
    -- The circle touches one leg of the triangle
    (∃ K : ℝ × ℝ, K ∈ circle ∧ K ∈ triangle) ∧
    -- The circle passes through the vertex of the opposite acute angle
    (∃ V : ℝ × ℝ, V ∈ circle ∧ V ∈ triangle) ∧
    -- The center of the circle lies on the hypotenuse of the triangle
    (∃ O : ℝ × ℝ, O ∈ circle ∧ ∃ P Q : ℝ × ℝ, P ∈ triangle ∧ Q ∈ triangle ∧ 
      ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ O = (1 - t) • P + t • Q) ∧
    -- The circle has radius r
    (∀ X : ℝ × ℝ, X ∈ circle ↔ ∃ C : ℝ × ℝ, dist X C = r ∧ (∀ Y : ℝ × ℝ, Y ∈ circle → dist Y C ≤ dist X C)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_isosceles_right_triangle_l1118_111841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sequence_sine_l1118_111852

theorem triangle_arithmetic_sequence_sine (a b c : ℝ) (A B C : ℝ) :
  a > b ∧ b > c ∧ c > 0 →  -- side lengths in descending order
  a - b = 2 ∧ b - c = 2 →  -- arithmetic sequence with common difference 2
  A > B ∧ B > C →  -- angles in descending order
  Real.sin A = Real.sqrt 3 / 2 →  -- sine of largest angle
  0 < C ∧ C < Real.pi →  -- C is a valid angle in radians
  Real.sin C = 3 * Real.sqrt 3 / 14 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sequence_sine_l1118_111852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_call_problem_l1118_111804

theorem phone_call_problem (n : ℕ) (k : ℕ+) : 
  (∀ (subset : Finset (Fin n)), subset.card = n - 2 → 
    (subset.powerset.filter (λ s ↦ s.card = 2)).card = 3^k.val) →
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_call_problem_l1118_111804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_selling_price_approximately_343_l1118_111822

/-- Calculates the lower selling price of an article given its cost price and a higher selling price that yields 5% more profit -/
noncomputable def lower_selling_price (cost_price : ℝ) (higher_price : ℝ) : ℝ :=
  let profit_at_higher_price := higher_price - cost_price
  let profit_at_lower_price := profit_at_higher_price / 1.05
  cost_price + profit_at_lower_price

theorem lower_selling_price_approximately_343 :
  ∃ ε > 0, |lower_selling_price 200 350 - 343| < ε :=
by
  -- The proof is omitted for now
  sorry

-- We can't use #eval for noncomputable functions, so we'll use #check instead
#check lower_selling_price 200 350

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_selling_price_approximately_343_l1118_111822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_symmetric_l1118_111854

-- Define the curves C and C1
noncomputable def C (x y : ℝ) : Prop := y = x^3 - x
noncomputable def C1 (x y t s : ℝ) : Prop := y = (x - t)^3 - (x - t) + s

-- Define the symmetry point A
noncomputable def A (t s : ℝ) : ℝ × ℝ := (t/2, s/2)

-- Theorem statement
theorem curves_symmetric (t s : ℝ) :
  ∀ (x1 y1 x2 y2 : ℝ),
    C x1 y1 → C1 x2 y2 t s →
    (x1 + x2 = t ∧ y1 + y2 = s) ↔
    (x2, y2) = (t - x1, s - y1) := by
  sorry

#check curves_symmetric

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_symmetric_l1118_111854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_problem_l1118_111807

theorem rotation_problem (y : ℝ) (h : y < 360) : 
  (780 % 360 = 60) ∧ (y = 300) := by
  -- Split the goal into two parts
  constructor
  
  -- Prove that 780 % 360 = 60
  · norm_num
  
  -- Prove that y = 300
  · -- We know that clockwise rotation of 60° is equivalent to
    -- counterclockwise rotation of 300°
    have h1 : 360 - 60 = 300 := by norm_num
    
    -- Given that y < 360 and y lands at the same point as 60° clockwise,
    -- y must be equal to 300
    sorry -- Full proof would require more advanced rotation properties
  
#check rotation_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_problem_l1118_111807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_acronym_length_l1118_111815

/-- The length of a diagonal in a unit square --/
noncomputable def unitDiagonalLength : ℝ := Real.sqrt 2

/-- The circumference of a circle with unit diameter --/
noncomputable def unitCircleCircumference : ℝ := Real.pi

/-- The total length of straight segments in the acronym XYZ --/
def straightSegmentsLength : ℝ := 6

/-- The total length of diagonal segments in the acronym XYZ --/
noncomputable def diagonalSegmentsLength : ℝ := 3 * unitDiagonalLength

/-- The total length of all segments forming the acronym XYZ plus the circumference of a unit-diameter circular hole --/
noncomputable def totalLength : ℝ := straightSegmentsLength + diagonalSegmentsLength + unitCircleCircumference

/-- Theorem stating the total length of the XYZ acronym --/
theorem xyz_acronym_length :
  totalLength = 6 + 3 * Real.sqrt 2 + Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_acronym_length_l1118_111815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_initial_male_workers_l1118_111883

theorem fraction_of_initial_male_workers :
  let initial_total : ℕ := 90
  let new_female_hires : ℕ := 10
  let final_female_percentage : ℚ := 2/5
  let fraction_of_initial_male_workers : ℚ := 
    (initial_total - (final_female_percentage * (initial_total + new_female_hires) - new_female_hires)) / initial_total
  fraction_of_initial_male_workers = 2/3 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_initial_male_workers_l1118_111883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_share_l1118_111885

/-- Represents the profit shares of the five partners -/
def profit_shares : List ℕ := [2, 4, 4, 6, 7]

/-- The total profit to be distributed -/
def total_profit : ℕ := 46000

/-- Calculates the maximum share value -/
def max_share_value (shares : List ℕ) (profit : ℕ) : ℕ :=
  let total_shares := shares.sum
  let share_value := profit / total_shares
  (shares.maximum?.getD 0) * share_value

/-- Theorem stating that the maximum profit share is $14,000 -/
theorem max_profit_share :
  max_share_value profit_shares total_profit = 14000 := by
  sorry

#eval max_share_value profit_shares total_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_share_l1118_111885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_students_in_section_A_l1118_111860

/-- Calculate the number of students in section A based on given conditions. -/
def students_in_section_A : ℕ :=
  let section_B_students : ℕ := 20
  let section_A_avg_weight : ℚ := 40
  let section_B_avg_weight : ℚ := 35
  let total_avg_weight : ℚ := 38
  
  let section_A_students : ℚ := 
    (section_A_avg_weight * section_B_students * (section_B_avg_weight - total_avg_weight)) /
    (total_avg_weight * (section_A_avg_weight - section_B_avg_weight))

  ⌊section_A_students⌋.toNat

/-- Given:
    - There are 2 sections, A and B
    - Section B has 20 students
    - Average weight of section A is 40 kg
    - Average weight of section B is 35 kg
    - Average weight of the whole class is 38 kg
    Prove that the number of students in section A is 30. -/
theorem prove_students_in_section_A : students_in_section_A = 30 := by
  sorry

#eval students_in_section_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_students_in_section_A_l1118_111860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_for_arithmetic_progression_with_non_real_roots_l1118_111876

-- Define the polynomial
def polynomial (b : ℝ) (x : ℂ) : ℂ := x^3 - 9*x^2 + 39*x + b

-- Define the property of roots forming an arithmetic progression
def roots_form_arithmetic_progression (p : ℝ → ℂ → ℂ) (b : ℝ) : Prop :=
  ∃ (r d : ℂ), (p b r = 0 ∧ p b (r-d) = 0 ∧ p b (r+d) = 0)

-- Define the property of having a pair of non-real roots
def has_non_real_roots (p : ℝ → ℂ → ℂ) (b : ℝ) : Prop :=
  ∃ (z : ℂ), (p b z = 0 ∧ z.im ≠ 0)

-- The main theorem
theorem unique_b_for_arithmetic_progression_with_non_real_roots :
  ∃! (b : ℝ), roots_form_arithmetic_progression polynomial b ∧ 
              has_non_real_roots polynomial b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_for_arithmetic_progression_with_non_real_roots_l1118_111876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1118_111867

/-- An ellipse with center at the origin, foci on the x-axis, eccentricity 1/2, 
    and maximum distance from a point on the ellipse to a focus is 3 -/
structure Ellipse where
  eccentricity : ℝ
  max_distance : ℝ
  h_ecc : eccentricity = 1/2
  h_max : max_distance = 3

/-- A line passing through a point P(0, m) and intersecting the ellipse at two distinct points A and B -/
structure IntersectingLine (E : Ellipse) where
  m : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_distinct : A ≠ B
  h_on_ellipse : (A.1^2/4 + A.2^2/3 = 1) ∧ (B.1^2/4 + B.2^2/3 = 1)
  h_ratio : (A.1 - 0)^2 + (A.2 - m)^2 = 9 * ((0 - B.1)^2 + (m - B.2)^2)

/-- The theorem to be proved -/
theorem ellipse_properties (E : Ellipse) :
  (∀ x y : ℝ, x^2/4 + y^2/3 = 1 ↔ (x, y) ∈ Set.range (λ (p : ℝ × ℝ) => p)) ∧
  (∀ l : IntersectingLine E, -Real.sqrt 3 < l.m ∧ l.m < -Real.sqrt 3/2 ∨
                             Real.sqrt 3/2 < l.m ∧ l.m < Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1118_111867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1118_111871

/-- Given an ellipse with semi-major axis a and semi-minor axis b, foci F₁ and F₂,
    point A on the ellipse, and point B on the y-axis, prove that if AF₁ ⊥ BF₁ and
    AF₂ = (2/3)F₂B, then the eccentricity of the ellipse is √5/5 -/
theorem ellipse_eccentricity (a b : ℝ) (F₁ F₂ A B : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧
  (B.1 = 0) ∧
  (((A.1 - F₁.1) * (B.1 - F₁.1) + (A.2 - F₁.2) * (B.2 - F₁.2)) = 0) ∧
  ((A.1 - F₂.1, A.2 - F₂.2) = (2/3) * (F₂.1 - B.1, F₂.2 - B.2)) →
  let c := Real.sqrt (a^2 - b^2)
  let e := c / a
  e = Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1118_111871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_light_change_probability_l1118_111881

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  duration : ℕ
  changePoints : List ℕ

/-- Calculates the probability of observing a change in a traffic light cycle -/
def probabilityOfObservingChange (cycle : TrafficLightCycle) (observationDuration : ℕ) : ℚ :=
  let totalChangeIntervals := (cycle.changePoints.map (fun p => min observationDuration p)).sum
  (totalChangeIntervals : ℚ) / (cycle.duration : ℚ)

/-- The traffic light cycle problem -/
theorem traffic_light_change_probability :
  let cycle := TrafficLightCycle.mk 100 [45, 50, 55, 100]
  probabilityOfObservingChange cycle 4 = 4 / 25 := by
  sorry

#eval probabilityOfObservingChange (TrafficLightCycle.mk 100 [45, 50, 55, 100]) 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_light_change_probability_l1118_111881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l1118_111836

/-- Complex number type -/
def C : Type := ℝ × ℝ

/-- Imaginary unit -/
def i : C := (0, 1)

/-- Addition for complex numbers -/
def add (a b : C) : C := (a.1 + b.1, a.2 + b.2)

/-- Multiplication for complex numbers -/
def mul (a b : C) : C := (a.1 * b.1 - a.2 * b.2, a.1 * b.2 + a.2 * b.1)

/-- Scalar multiplication for complex numbers -/
def smul (r : ℝ) (z : C) : C := (r * z.1, r * z.2)

/-- Equality for complex numbers -/
def eq (a b : C) : Prop := a.1 = b.1 ∧ a.2 = b.2

theorem complex_equation_solution :
  let z : C := (0, -3/2)  -- Representing -3i/2
  eq (add (3, 0) (mul (smul (-2) i) z))
     (add (-3, 0) (mul (smul 2 i) z)) ∧
  eq (mul i i) (-1, 0) := by
  sorry

#check complex_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l1118_111836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_circle_from_polar_equation_l1118_111818

-- Define the polar equation
noncomputable def polarEquation (θ : ℝ) : ℝ := 3 * Real.cos θ - 4 * Real.sin θ

-- Define the area of the circle
noncomputable def circleArea (r : ℝ) : ℝ := Real.pi * r^2

-- Theorem statement
theorem area_of_circle_from_polar_equation :
  ∃ (r : ℝ), (∀ θ, polarEquation θ ≤ r) ∧ circleArea r = (25 / 4) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_circle_from_polar_equation_l1118_111818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l1118_111811

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - ((a + 1) / a) * x + 1

-- State the theorem
theorem function_analysis (a : ℝ) (h : a > 0) :
  -- Part 1
  (a = 1/2 → ∀ x, f a x ≤ 0 ↔ (3 - Real.sqrt 5) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 5) / 2) ∧
  -- Part 2
  ((0 < a ∧ a < 1 → a < 1/a) ∧ (a > 1 → a > 1/a) ∧ (a = 1 → a = 1/a)) ∧
  -- Part 3
  ((0 < a ∧ a < 1 → ∀ x, f a x ≤ 0 ↔ a < x ∧ x < 1/a) ∧
   (a > 1 → ∀ x, f a x ≤ 0 ↔ 1/a < x ∧ x < a) ∧
   (a = 1 → ∀ x, f a x ≤ 0 ↔ x = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l1118_111811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1118_111839

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x - 1/x - b * Real.log x

-- Define the function g
def g (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem function_properties :
  (∃ b : ℝ, (∀ x : ℝ, x > 0 → (deriv (f b)) 1 = 0) ∧ b = 2) ∧
  (∀ x : ℝ, x > 0 → g x > f 2 x - 2 * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1118_111839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_x_equals_pi_minus_two_over_four_l1118_111833

theorem integral_sqrt_minus_x_equals_pi_minus_two_over_four :
  ∫ x in (0:ℝ)..1, (Real.sqrt (1 - x^2) - x) = (π - 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_x_equals_pi_minus_two_over_four_l1118_111833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1118_111825

noncomputable section

-- Define the function f on [-1,0) ∪ (0,1]
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2*x + 1/x^2
  else if x > 0 then 2*x - 1/x^2
  else 0  -- undefined at x = 0

-- State the theorem
theorem f_properties :
  -- f is odd
  (∀ x, x ∈ Set.Icc (-1) 1 \ {0} → f (-x) = -f x) ∧
  -- f(x) = 2x + 1/x² for x ∈ [-1,0)
  (∀ x, x ∈ Set.Ioc (-1) 0 → f x = 2*x + 1/x^2) →
  -- Conclusion 1: f(x) = 2x - 1/x² for x ∈ (0,1]
  (∀ x, x ∈ Set.Ioc 0 1 → f x = 2*x - 1/x^2) ∧
  -- Conclusion 2: f is monotonically increasing on (0,1]
  (∀ x y, x ∈ Set.Ioc 0 1 → y ∈ Set.Ioc 0 1 → x < y → f x < f y) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1118_111825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_l1118_111832

theorem complex_expression_equality : 
  (1 - Complex.I)^2 - (4 + 2*Complex.I)/(1 - 2*Complex.I) - 4*Complex.I^2014 = 4 - 4*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_l1118_111832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_hyperbola_equations_l1118_111879

/-- Definition of an ellipse with given properties -/
def Ellipse (e : ℝ) (d : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (f₁ f₂ : ℝ × ℝ), 
    f₁.2 = 0 ∧ f₂.2 = 0 ∧ 
    Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = d ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) / d = e}

/-- Definition of a hyperbola with given properties -/
def Hyperbola (c : ℝ) (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a b : ℝ),
    (p.1 / a)^2 - (p.2 / b)^2 = 1 ∧
    b / a = m ∧
    a^2 - c^2 = b^2}

theorem ellipse_and_hyperbola_equations :
  ∃ (e : Set (ℝ × ℝ)) (h : Set (ℝ × ℝ)),
    e = Ellipse (1/2) 8 ∧
    h = Hyperbola 2 (Real.sqrt 3) ∧
    (∀ p ∈ e, p.1^2 / 16 + p.2^2 / 12 = 1) ∧
    (∀ p ∈ h, p.1^2 - p.2^2 / 3 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_hyperbola_equations_l1118_111879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_piece_length_l1118_111884

/-- Represents the continued fraction [2; 3, 4] -/
def ratio : ℚ := 2 + 1 / (3 + 1 / 4)

/-- The total length of the wire in centimeters -/
noncomputable def total_length : ℝ := 100

/-- The length of the shorter piece of wire in centimeters -/
noncomputable def shorter_piece : ℝ := total_length / (1 + ratio)

/-- Theorem stating that the shorter piece is approximately 30.23 cm -/
theorem shorter_piece_length : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |shorter_piece - 30.23| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_piece_length_l1118_111884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l1118_111875

theorem triangle_side_ratio_range (a b c A B C : ℝ) : 
  0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2 →  -- acute triangle
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  A + B + C = Real.pi →  -- angle sum in a triangle
  a / Real.sin A = b / Real.sin B →  -- law of sines
  b / Real.sin B = c / Real.sin C →  -- law of sines
  2 * Real.sin A * (a * Real.cos C + c * Real.cos A) = Real.sqrt 3 * a →
  Real.sqrt 3 / 3 < c / b ∧ c / b < 2 * Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l1118_111875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_two_points_l1118_111827

/-- Definition of a line through two points -/
def line_through (p₁ p₂ : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • p₁ + t • p₂}

/-- 
Theorem: For any two distinct points (x₁, y₁) and (x₂, y₂) in a 2D plane, 
the equation (x₂ - x₁)(y - y₁) = (y₂ - y₁)(x - x₁) represents all lines 
passing through these points.
-/
theorem line_equation_through_two_points 
  (x₁ y₁ x₂ y₂ : ℝ) (h : (x₁, y₁) ≠ (x₂, y₂)) :
  ∀ (x y : ℝ), (x, y) ∈ line_through (x₁, y₁) (x₂, y₂) ↔ 
    (x₂ - x₁) * (y - y₁) = (y₂ - y₁) * (x - x₁) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_two_points_l1118_111827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_symmetry_l1118_111830

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_properties_and_symmetry 
  (A ω φ : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : |φ| < π/2) 
  (h4 : ∀ x, f A ω φ (x + 2) = f A ω φ x) 
  (h5 : f A ω φ (1/3) = 2) 
  (h6 : ∀ x, f A ω φ x ≤ 2) :
  (∀ x, f A ω φ x = 2 * Real.sin (π * x + π/6)) ∧ 
  (∃ y ∈ Set.Icc (21/4) (23/4), ∀ x, f A ω φ (2*y - x) = f A ω φ x) ∧
  (∀ x, f A ω φ (32/3 - x) = f A ω φ x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_symmetry_l1118_111830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1118_111858

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : 0 < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a hyperbola with given properties, |PF₂| = 1 or 9 -/
theorem hyperbola_focal_distance (h : Hyperbola) (p f1 f2 : Point) : 
  h.b / h.a = 1 / 2 →  -- Condition from asymptote equation
  distance p f1 = 5 →  -- Given |PF₁| = 5
  (∃ (x y : ℝ), x^2 / h.a^2 - y^2 / h.b^2 = 1 ∧ p.x = x ∧ p.y = y) →  -- P is on the hyperbola
  distance p f2 = 1 ∨ distance p f2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1118_111858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1118_111805

theorem trigonometric_identities (x : Real) 
  (h1 : Real.cos (x - Real.pi/4) = Real.sqrt 2/10)
  (h2 : x ∈ Set.Ioo (Real.pi/2) (3*Real.pi/4)) :
  Real.sin x = 4/5 ∧ Real.cos (2*x - Real.pi/3) = -(7 + 24*Real.sqrt 3)/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1118_111805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l1118_111865

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_approx :
  let a := (26 : ℝ)
  let b := (25 : ℝ)
  let c := (10 : ℝ)
  abs (triangle_area a b c - 95) < 0.5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l1118_111865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_z_value_l1118_111891

theorem smallest_z_value (w x y z : ℕ) : 
  w^3 + x^3 + y^3 = z^3 →
  w^3 < x^3 ∧ x^3 < y^3 ∧ y^3 < z^3 →
  (∃ n : ℕ, w = 2*n ∧ x = 2*(n+1) ∧ y = 2*(n+2) ∧ z = 2*(n+3)) →
  z ≥ 12 := by
  sorry

#check smallest_z_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_z_value_l1118_111891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_ratio_l1118_111857

/-- Represents a soccer team composition -/
structure SoccerTeam where
  total : Nat
  goalies : Nat
  defenders : Nat
  strikers : Nat
  midfielders : Nat
  comp_eq : total = goalies + defenders + midfielders + strikers

/-- The ratio of two natural numbers -/
def ratio (a b : Nat) : ℚ :=
  ↑a / ↑b

theorem soccer_team_ratio (team : SoccerTeam)
  (h1 : team.total = 40)
  (h2 : team.goalies = 3)
  (h3 : team.defenders = 10)
  (h4 : team.strikers = 7) :
  ratio team.midfielders team.defenders = 2 / 1 := by
  sorry

#eval ratio 20 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_ratio_l1118_111857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_increase_l1118_111880

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt 3 * (Real.cos (x/2) - Real.sin (x/2)) * (Real.cos (x/2) + Real.sin (x/2)) + 2 * Real.sin (x/2) * Real.cos (x/2)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi/6)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ p > 0 ∧ ∀ q, is_periodic f q ∧ q > 0 → p ≤ q

def interval_of_increase (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_period_and_g_increase :
  (smallest_positive_period f (2 * Real.pi)) ∧
  (∀ k : ℤ, interval_of_increase g (-2*Real.pi/3 + 2*k*Real.pi) (Real.pi/3 + 2*k*Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_increase_l1118_111880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1118_111869

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X : Polynomial ℝ)^4 - 3*(X^3) + 2*(X^2) + 4*X - 1 =
  ((X^2 - 1) * (X - 2)) * q + (X^2 + X + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1118_111869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_term_at_12_or_13_l1118_111849

/-- The sequence a_n = n / (n^2 + 156) -/
noncomputable def a (n : ℕ) : ℝ := n / (n^2 + 156)

/-- The function f(x) = x / (x^2 + 156) -/
noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 156)

theorem max_term_at_12_or_13 :
  ∃ (m : ℕ), (m = 12 ∨ m = 13) ∧
  ∀ (n : ℕ), n ≠ 0 → a n ≤ a m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_term_at_12_or_13_l1118_111849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abs_x_plus_y_l1118_111855

-- Define the complex numbers z₁ and z₂
def z₁ (x y : ℝ) : ℂ := x + (y + 2) * Complex.I
def z₂ (x y : ℝ) : ℂ := (x - 2) + y * Complex.I

-- State the theorem
theorem max_abs_x_plus_y (x y : ℝ) 
  (h : Complex.abs (z₁ x y) + Complex.abs (z₂ x y) = 4) :
  ∃ (x₀ y₀ : ℝ), Complex.abs (z₁ x₀ y₀) + Complex.abs (z₂ x₀ y₀) = 4 ∧
                 (∀ (x' y' : ℝ), Complex.abs (z₁ x' y') + Complex.abs (z₂ x' y') = 4 →
                                |x' + y'| ≤ |x₀ + y₀|) ∧
                 |x₀ + y₀| = 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abs_x_plus_y_l1118_111855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_figure_l1118_111888

-- Define the curve
def f (x : ℝ) : ℝ := x^2

-- Define the point A
def A : ℝ × ℝ := (2, 4)

-- Define the tangent line at A
def tangent_line (x : ℝ) : ℝ := 4*x - 4

-- Define the area of the figure
noncomputable def area : ℝ :=
  (∫ x in Set.Icc 0 1, f x) + (∫ x in Set.Icc 1 2, (f x - tangent_line x))

-- Theorem statement
theorem area_of_figure : area = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_figure_l1118_111888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1118_111870

def circle_center : ℝ × ℝ := (-3, 2)
def circle_radius : ℝ := 2
def point_p : ℝ × ℝ := (-1, 6)

def is_tangent_line (a b c : ℝ) : Prop :=
  let d := |a * circle_center.1 + b * circle_center.2 + c| / Real.sqrt (a^2 + b^2)
  d = circle_radius

theorem tangent_line_equation :
  (∃ (k : ℝ), is_tangent_line 3 (-4) 27 ∧ 3 * point_p.1 - 4 * point_p.2 + 27 = 0) ∨
  (is_tangent_line 1 0 1 ∧ point_p.1 = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1118_111870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_point_slope_equals_pi_over_four_l1118_111864

-- Define the curve (marked as noncomputable due to dependency on Real)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

-- Define the derivative of the curve (marked as noncomputable due to dependency on Real)
noncomputable def f_prime (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem tangent_slope_at_point :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -5/3
  f x₀ = y₀ ∧ f_prime x₀ = 1 := by
  -- Proof steps would go here, but we'll use sorry to skip the proof
  sorry

-- Additional theorem to connect the slope to the angle
theorem slope_equals_pi_over_four :
  f_prime 1 = Real.tan (π / 4) := by
  -- Proof steps would go here, but we'll use sorry to skip the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_point_slope_equals_pi_over_four_l1118_111864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_repeated_letter_probability_l1118_111842

theorem adjacent_repeated_letter_probability :
  let n : ℕ := 10  -- Total number of letters in the code
  let r : ℕ := 2   -- Number of times the repeated letter appears
  let p : ℚ := 1 / 5  -- Given probability
  ∀ (total_arrangements : ℕ) (favorable_arrangements : ℕ),
    total_arrangements = n.factorial / r.factorial →
    favorable_arrangements = (n - r + 1).factorial →
    (favorable_arrangements : ℚ) / total_arrangements = p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_repeated_letter_probability_l1118_111842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_l1118_111800

/-- The trajectory of point P satisfying |PA| = 2|PB| -/
def trajectory (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

/-- The area enclosed by the trajectory -/
noncomputable def enclosed_area : ℝ := 4 * Real.pi

/-- Theorem stating the properties of the trajectory and its enclosed area -/
theorem trajectory_and_area :
  ∀ (x y : ℝ),
  (x + 2)^2 + y^2 = 4 * ((x - 1)^2 + y^2) →
  trajectory x y ∧ enclosed_area = 4 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_l1118_111800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_properties_l1118_111816

theorem quadratic_roots_properties (k : ℝ) (x₁ x₂ : ℝ) 
  (h_pos : k > 0) 
  (h_roots : x₁^2 - 2*k*x₁ - 2*k^2 = 0 ∧ x₂^2 - 2*k*x₂ - 2*k^2 = 0) :
  x₁ * x₂ = -2*k^2 ∧ |x₁ - x₂| = 2*Real.sqrt 3*k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_properties_l1118_111816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_7_8_to_2_l1118_111877

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 - Real.sin x - 2 * (Real.cos x)^2

-- Define the domain
def D : Set ℝ := {x | Real.pi/6 ≤ x ∧ x ≤ 7*Real.pi/6}

-- Theorem statement
theorem f_range_is_7_8_to_2 :
  {y | ∃ x ∈ D, f x = y} = {y | 7/8 ≤ y ∧ y ≤ 2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_7_8_to_2_l1118_111877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_cylinders_to_cone_percentage_approx_l1118_111826

/-- The volume of a right circular cylinder -/
noncomputable def cylinderVolume (height : ℝ) (circumference : ℝ) : ℝ :=
  (circumference^2 * height) / (4 * Real.pi)

/-- The volume of a right circular cone -/
noncomputable def coneVolume (height : ℝ) (baseCircumference : ℝ) : ℝ :=
  (baseCircumference^2 * height) / (12 * Real.pi)

/-- The combined volume of two cylinders as a percentage of a cone's volume -/
noncomputable def combinedCylindersToConePercentage 
  (cylinderAHeight : ℝ) (cylinderACircumference : ℝ)
  (cylinderBHeight : ℝ) (cylinderBCircumference : ℝ)
  (coneHeight : ℝ) (coneBaseCircumference : ℝ) : ℝ :=
  let cylinderAVolume := cylinderVolume cylinderAHeight cylinderACircumference
  let cylinderBVolume := cylinderVolume cylinderBHeight cylinderBCircumference
  let coneVolume := coneVolume coneHeight coneBaseCircumference
  ((cylinderAVolume + cylinderBVolume) / coneVolume) * 100

theorem combined_cylinders_to_cone_percentage_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (combinedCylindersToConePercentage 6 8 8 10 10 12 - 246.67) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_cylinders_to_cone_percentage_approx_l1118_111826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_postal_rate_correct_l1118_111863

noncomputable def postalRate (W : ℝ) : ℕ :=
  if W ≤ 5 then
    (10 * ⌈W⌉).toNat
  else
    (50 + 3 * ⌈W - 5⌉).toNat

theorem postal_rate_correct (W : ℝ) :
  postalRate W = if W ≤ 5 then (10 * ⌈W⌉).toNat else (50 + 3 * ⌈W - 5⌉).toNat := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_postal_rate_correct_l1118_111863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_consecutive_sum_of_squares_l1118_111856

theorem no_four_consecutive_sum_of_squares (n : ℕ) : 
  ∃ k ∈ Finset.range 4, ¬∃ (a b : ℕ), n + k = a^2 + b^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_consecutive_sum_of_squares_l1118_111856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abcd_plus_abcd_eq_5472_implies_d_eq_6_l1118_111814

/-- Represents a four-digit number ABCD --/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_range : a ≥ 1 ∧ a ≤ 9
  b_range : b ≥ 0 ∧ b ≤ 9
  c_range : c ≥ 0 ∧ c ≤ 9
  d_range : d ≥ 0 ∧ d ≤ 9
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Converts a FourDigitNumber to its numerical value --/
def fourDigitToNat (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

theorem abcd_plus_abcd_eq_5472_implies_d_eq_6 (n : FourDigitNumber) 
  (h : fourDigitToNat n + fourDigitToNat n = 5472) : n.d = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abcd_plus_abcd_eq_5472_implies_d_eq_6_l1118_111814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_two_l1118_111821

-- Define the functions representing the conditions
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x
def g (x : ℝ) : ℝ := x - 2

-- Define the distance function
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (x₁ - x₂)^2 + (y₁ - y₂)^2

-- State the theorem
theorem min_distance_is_two :
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    f x₁ = y₁ ∧
    g x₂ = y₂ ∧
    (∀ a b c d : ℝ, f a = b → g c = d → distance x₁ y₁ x₂ y₂ ≤ distance a b c d) ∧
    distance x₁ y₁ x₂ y₂ = 2 :=
by
  sorry

#check min_distance_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_two_l1118_111821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2019_l1118_111831

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | n+2 => sequence_a (n+1) - sequence_a n + (n+2)

theorem sequence_a_2019 : sequence_a 2019 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2019_l1118_111831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_calculations_l1118_111845

/-- Given a point P(-1, 2) on the terminal side of angle α, prove the following statements -/
theorem angle_calculations (α : ℝ) (h : ∃ (P : ℝ × ℝ), P = (-1, 2) ∧ P.1 = -Real.cos α ∧ P.2 = Real.sin α) : 
  Real.tan α = -2 ∧ (Real.sin α + Real.cos α) / (Real.cos α - Real.sin α) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_calculations_l1118_111845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_triangle_area_l1118_111843

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  area : ℝ
  area_eq : area = (Real.sqrt 3 / 4) * side_length ^ 2

/-- Represents the result of folding an equilateral triangle twice -/
structure FoldedTriangle where
  original : EquilateralTriangle
  folded_area : ℝ

/-- 
If an equilateral triangle is folded twice in a specific manner 
resulting in a shape with an area of 12 cm², then the area of 
the original triangle is 36 cm².
-/
theorem folded_triangle_area (t : FoldedTriangle) 
  (h : t.folded_area = 12) : t.original.area = 36 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_triangle_area_l1118_111843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_to_negative_x_l1118_111803

noncomputable def f (x : ℝ) : ℝ := 1 / Real.exp x

theorem tangent_line_parallel_to_negative_x :
  ∃ (a b c : ℝ), 
    (∀ x, (deriv f x = -1) → 
      (a * x + b * f x + c = 0 ∧ 
       ∀ y, a * x + b * y + c = 0 → y = -x + f x)) ∧ 
    a = 1 ∧ b = 1 ∧ c = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_to_negative_x_l1118_111803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_iff_sum_squares_bound_l1118_111829

/-- Represents a round-robin tournament result -/
structure Tournament (n : ℕ) where
  scores : Fin n → ℕ
  valid_scores : ∀ i, scores i ≤ n - 1

/-- Checks if there's a cyclic relationship between three players -/
def has_cycle (t : Tournament n) : Prop :=
  ∃ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
    (t.scores a > t.scores b) ∧ 
    (t.scores b > t.scores c) ∧ 
    (t.scores c > t.scores a)

/-- Sum of squares of scores -/
def sum_of_squares (t : Tournament n) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin n)) fun i => (t.scores i) ^ 2

/-- The main theorem -/
theorem cycle_iff_sum_squares_bound {n : ℕ} (t : Tournament n) :
  has_cycle t ↔ sum_of_squares t < (n - 1) * n * (2 * n - 1) / 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_iff_sum_squares_bound_l1118_111829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l1118_111813

/-- Calculates the position of the hour hand at a given time -/
noncomputable def hourHandPosition (hours : ℕ) (minutes : ℕ) : ℝ :=
  (hours % 12 : ℝ) * 30 + (minutes : ℝ) * 0.5

/-- Calculates the position of the minute hand at a given time -/
noncomputable def minuteHandPosition (minutes : ℕ) : ℝ :=
  (minutes : ℝ) * 6

/-- Calculates the smaller angle between two positions on a clock face -/
noncomputable def smallerAngle (pos1 : ℝ) (pos2 : ℝ) : ℝ :=
  min (abs (pos1 - pos2)) (360 - abs (pos1 - pos2))

theorem clock_angle_at_3_15 :
  smallerAngle (hourHandPosition 3 15) (minuteHandPosition 15) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l1118_111813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_shadow_boundary_l1118_111828

def sphere_radius : ℝ := 2
def sphere_center : Fin 3 → ℝ := ![0, 0, 2]
def light_source : Fin 3 → ℝ := ![0, -2, 3]

noncomputable def shadow_boundary (x : ℝ) : ℝ := -2 - Real.sqrt (4 - x^2)

theorem sphere_shadow_boundary :
  ∀ x : ℝ, x^2 ≤ 4 →
  let y := shadow_boundary x
  (![x, y, 0] : Fin 3 → ℝ) ∈ {p : Fin 3 → ℝ | ∃ t : Fin 3 → ℝ, 
    (t 0 - sphere_center 0)^2 + (t 1 - sphere_center 1)^2 + (t 2 - sphere_center 2)^2 = sphere_radius^2 ∧
    (p 0 - light_source 0) * (t 0 - sphere_center 0) + 
    (p 1 - light_source 1) * (t 1 - sphere_center 1) + 
    (p 2 - light_source 2) * (t 2 - sphere_center 2) = 0 ∧
    ∃ k : ℝ, p = fun i => k * (t i - light_source i) + light_source i} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_shadow_boundary_l1118_111828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_share_price_change_approx_l1118_111859

noncomputable def share_price_change (P : ℝ) : ℝ :=
  let q1 := P * 1.30
  let q2 := q1 * 0.80
  let q3 := q2 * 1.40
  let q4 := q3 * 0.90
  ((q4 - q1) / q1) * 100

theorem share_price_change_approx (P : ℝ) (P_pos : P > 0) :
  abs (share_price_change P - 0.8) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_share_price_change_approx_l1118_111859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_price_check_result_l1118_111897

/-- The cost price of an item given its selling price and markup percentage. -/
noncomputable def costPrice (sellingPrice : ℝ) (markupPercentage : ℝ) : ℝ :=
  sellingPrice / (1 + markupPercentage / 100)

/-- Theorem: The cost price of a computer table sold for Rs. 8600 with a 20% markup is approximately Rs. 7166.67. -/
theorem computer_table_cost_price :
  let sellingPrice : ℝ := 8600
  let markupPercentage : ℝ := 20
  abs (costPrice sellingPrice markupPercentage - 7166.67) < 0.01 := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use a theorem to check the result
theorem check_result :
  abs (costPrice 8600 20 - 7166.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_price_check_result_l1118_111897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_formula_correct_l1118_111820

/-- An isosceles trapezoid with one acute base angle of 45° -/
structure IsoscelesTrapezoid where
  /-- Length of the longer parallel side -/
  a : ℝ
  /-- Length of the shorter parallel side -/
  b : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- One of the acute base angles is 45° -/
  hasFortyFiveDegreeAngle : True
  /-- The longer side is indeed longer than the shorter side -/
  a_gt_b : a > b

/-- The area of an isosceles trapezoid with one 45° acute base angle -/
noncomputable def area (t : IsoscelesTrapezoid) : ℝ :=
  (t.a^2 - t.b^2) / 4

/-- Theorem stating that the area formula is correct -/
theorem area_formula_correct (t : IsoscelesTrapezoid) : 
  area t = (t.a^2 - t.b^2) / 4 := by
  -- Unfold the definition of area
  unfold area
  -- The equality follows directly from the definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_formula_correct_l1118_111820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_order_l1118_111840

-- Define the functions as noncomputable
noncomputable def f (x : ℝ) := x^x
noncomputable def g (x : ℝ) := x^(x^x)
noncomputable def h (x : ℝ) := x^(x^(x^x))

-- State the theorem
theorem increasing_order (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) : 
  x < h x ∧ h x < g x := by
  -- The proof is skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_order_l1118_111840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_ratio_l1118_111872

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the focal length
noncomputable def focal_length : ℝ := 2 * Real.sqrt 2

-- Define point M
noncomputable def M : ℝ × ℝ := (Real.sqrt 2, 1)

-- Define point P
def P : ℝ × ℝ := (0, 1)

-- Define point Q
def Q : ℝ × ℝ := (0, 2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem ellipse_fixed_point_ratio :
  ∀ (A B : ℝ × ℝ),
  E A.1 A.2 →
  E B.1 B.2 →
  ∃ (k : ℝ), A.2 - P.2 = k * (A.1 - P.1) ∧ B.2 - P.2 = k * (B.1 - P.1) →
  distance Q A / distance Q B = distance P A / distance P B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_ratio_l1118_111872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1118_111806

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x, g (-x) = -g x := by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1118_111806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_plane_perpendicular_planes_l1118_111873

-- Define the basic types
structure Line
structure Plane

-- Define the relationships as axioms
axiom perpendicular_line_plane : Line → Plane → Prop
axiom perpendicular_line_line : Line → Line → Prop
axiom perpendicular_plane_plane : Plane → Plane → Prop
axiom parallel_line_plane : Line → Plane → Prop
axiom contained_in : Line → Plane → Prop
axiom intersecting : Line → Line → Prop

-- Theorem 1
theorem perpendicular_to_plane (l : Line) (α : Plane) 
  (h1 : ∃ l1 l2 : Line, contained_in l1 α ∧ contained_in l2 α ∧ 
        intersecting l1 l2 ∧ perpendicular_line_line l l1 ∧ perpendicular_line_line l l2) :
  perpendicular_line_plane l α :=
sorry

-- Theorem 4
theorem perpendicular_planes (l : Line) (α β : Plane)
  (h1 : contained_in l β)
  (h2 : perpendicular_line_plane l α) :
  perpendicular_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_plane_perpendicular_planes_l1118_111873

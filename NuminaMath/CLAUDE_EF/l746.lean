import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_with_odd_factors_l746_74635

theorem three_digit_integers_with_odd_factors : 
  (Finset.filter (λ n : ℕ => 100 ≤ n ∧ n ≤ 999 ∧ 
    (Finset.filter (λ d : ℕ => d ∣ n) (Finset.range (n + 1))).card % 2 = 1) 
    (Finset.range 1000)).card = 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_with_odd_factors_l746_74635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_slices_l746_74609

/-- Represents the number of slices in each pizza -/
def slices_per_pizza : ℕ := 12

/-- The number of slices remaining after Dean ate half the Hawaiian pizza -/
def remaining_hawaiian_after_dean : ℚ := slices_per_pizza / 2

/-- The number of slices remaining after Frank ate 3 slices of Hawaiian pizza -/
def remaining_hawaiian_after_frank : ℚ := remaining_hawaiian_after_dean - 3

/-- The number of slices remaining of the cheese pizza after Sammy ate a third -/
def remaining_cheese : ℚ := (2 : ℚ) / 3 * slices_per_pizza

/-- The total number of slices remaining -/
def total_remaining : ℕ := 11

theorem pizza_slices : 
  remaining_hawaiian_after_frank + remaining_cheese = total_remaining ∧ 
  slices_per_pizza = 12 := by
  sorry

#eval slices_per_pizza

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_slices_l746_74609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relationship_l746_74611

-- Define the sets
def S : Set ℝ := {x | (1/3:ℝ)^x < 1}
def T : Set ℝ := {x | 1/x > 1}

-- State the theorem
theorem condition_relationship : T ⊆ S ∧ ¬(S ⊆ T) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relationship_l746_74611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_complex_ratio_l746_74677

theorem min_complex_ratio (a₁ a₂ a₃ : ℂ) 
  (h₁ : a₁ ≠ 0) (h₂ : a₂ ≠ 0) (h₃ : a₃ ≠ 0)
  (nonneg_re_a₁ : 0 ≤ a₁.re) (nonneg_im_a₁ : 0 ≤ a₁.im)
  (nonneg_re_a₂ : 0 ≤ a₂.re) (nonneg_im_a₂ : 0 ≤ a₂.im)
  (nonneg_re_a₃ : 0 ≤ a₃.re) (nonneg_im_a₃ : 0 ≤ a₃.im) :
  Complex.abs (a₁ + a₂ + a₃) / (Complex.abs (a₁ * a₂ * a₃)) ^ (1/3 : ℝ) ≥ Real.sqrt 3 * 2 ^ (1/3 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_complex_ratio_l746_74677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_area_calculation_l746_74678

def gino_stick_count : ℕ := 63
def gino_stick_length : ℚ := 9/2
def gino_stick_width : ℚ := 2/5

def my_stick_count : ℕ := 50
def my_stick_length : ℚ := 6
def my_stick_width : ℚ := 3/5

def gino_square_side_sticks : ℕ := 15
def my_rectangle_length_sticks : ℕ := 25
def my_rectangle_width_sticks : ℕ := 25

theorem combined_area_calculation :
  (gino_square_side_sticks * gino_stick_length) * (gino_square_side_sticks * gino_stick_length) +
  (my_rectangle_length_sticks * my_stick_length) * (my_rectangle_width_sticks * my_stick_width) = 6806.25 := by
  -- Proof steps would go here
  sorry

#eval (gino_square_side_sticks * gino_stick_length) * (gino_square_side_sticks * gino_stick_length) +
      (my_rectangle_length_sticks * my_stick_length) * (my_rectangle_width_sticks * my_stick_width)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_area_calculation_l746_74678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_or_five_scores_l746_74694

def student_score : ℕ → ℕ := sorry

theorem no_four_or_five_scores (n : ℕ) (x : ℕ) : 
  (n > 0) →  -- There is at least one student
  (9 + 7 + 5 + 3 + 1 = 25) →  -- Total problems solved
  (n * x + 1 = 25) →  -- Equation relating n and x
  (∀ i : ℕ, i < n - 1 → i ≠ 0 → student_score i = x) →  -- All students except Petya solved x problems
  (student_score (n - 1) = x + 1) →  -- Petya solved x + 1 problems
  (∀ i : ℕ, i < n → student_score i ≤ 3) :=  -- No student received a score of 4 or 5
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_or_five_scores_l746_74694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_average_x_coordinate_l746_74646

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^4 + 7 * x^3 + 3 * x - 5

-- Define collinearity for four points
def collinear (p₁ p₂ p₃ p₄ : ℝ × ℝ) : Prop :=
  ∃ (m c : ℝ), ∀ (p : ℝ × ℝ), p = p₁ ∨ p = p₂ ∨ p = p₃ ∨ p = p₄ → p.2 = m * p.1 + c

theorem collinear_points_average_x_coordinate 
  (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : f x₁ = 2 * x₁^4 + 7 * x₁^3 + 3 * x₁ - 5)
  (h₂ : f x₂ = 2 * x₂^4 + 7 * x₂^3 + 3 * x₂ - 5)
  (h₃ : f x₃ = 2 * x₃^4 + 7 * x₃^3 + 3 * x₃ - 5)
  (h₄ : f x₄ = 2 * x₄^4 + 7 * x₄^3 + 3 * x₄ - 5)
  (h_distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)
  (h_collinear : collinear (x₁, f x₁) (x₂, f x₂) (x₃, f x₃) (x₄, f x₄)) :
  (x₁ + x₂ + x₃ + x₄) / 4 = -7/8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_average_x_coordinate_l746_74646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_functions_l746_74659

-- Define the functions
noncomputable def f1 (x : ℝ) : ℝ := x^2
noncomputable def f2 (x : ℝ) : ℝ := |x^2|
noncomputable def f3 (x : ℝ) : ℝ := 2/x
noncomputable def f4 (x : ℝ) : ℝ := |2/x|

-- Define symmetry properties
def axis_symmetric (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def origin_symmetric (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem symmetry_of_functions :
  (axis_symmetric f3 ∧ origin_symmetric f3) ∧
  (¬(axis_symmetric f1 ∧ origin_symmetric f1)) ∧
  (¬(axis_symmetric f2 ∧ origin_symmetric f2)) ∧
  (¬(axis_symmetric f4 ∧ origin_symmetric f4)) := by
  sorry

#check symmetry_of_functions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_functions_l746_74659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_is_two_l746_74698

/-- The function f(x) = (x^2 + 2) / sqrt(x^2 + 1) -/
noncomputable def f (x : ℝ) : ℝ := (x^2 + 2) / Real.sqrt (x^2 + 1)

/-- The minimum value of f(x) is 2 -/
theorem f_min_value_is_two : 
  ∀ x : ℝ, f x ≥ 2 ∧ ∃ x₀ : ℝ, f x₀ = 2 := by
  sorry

#check f_min_value_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_is_two_l746_74698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sum_theorem_l746_74626

theorem integral_sum_theorem : 
  let a : ℝ := ∫ x in (-1 : ℝ)..1, x
  let b : ℝ := ∫ x in (0 : ℝ)..Real.pi, Real.sin x
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sum_theorem_l746_74626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l746_74656

/-- Calculates the length of a platform given train length, speed, and crossing time. -/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  let platform_length := total_distance - train_length
  train_length = 140 ∧ train_speed_kmh = 55 ∧ crossing_time = 43.196544276457885 →
  abs (platform_length - 519.4444444444443) < 0.0001 := by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l746_74656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l746_74644

/-- Represents a point in the triangle --/
structure TrianglePoint where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the triangle ABC --/
structure EquilateralTriangle where
  A : TrianglePoint
  B : TrianglePoint
  C : TrianglePoint
  side_length : ℝ
  is_equilateral : side_length = 1

/-- Represents the points on the sides of the triangle --/
structure TrianglePoints where
  A₁ : TrianglePoint
  A₂ : TrianglePoint
  B₁ : TrianglePoint
  B₂ : TrianglePoint
  C₁ : TrianglePoint
  C₂ : TrianglePoint
  on_sides : (A₁.y + A₁.z = 1) ∧ (A₂.y + A₂.z = 1) ∧
             (B₁.x + B₁.z = 1) ∧ (B₂.x + B₂.z = 1) ∧
             (C₁.x + C₁.y = 1) ∧ (C₂.x + C₂.y = 1)
  order : (A₁.y < A₂.y) ∧ (B₁.x < B₂.x) ∧ (C₁.x < C₂.x)

/-- Checks if three lines are concurrent --/
def are_concurrent (p₁ q₁ p₂ q₂ p₃ q₃ : TrianglePoint) : Prop :=
  let det := (p₁.x * q₁.y - p₁.y * q₁.x) * (p₂.z - q₂.z) +
             (p₁.y * q₁.z - p₁.z * q₁.y) * (p₂.x - q₂.x) +
             (p₁.z * q₁.x - p₁.x * q₁.z) * (p₂.y - q₂.y)
  det = 0

/-- Calculates the perimeter of a triangle --/
noncomputable def triangle_perimeter (p₁ p₂ p₃ : TrianglePoint) : ℝ :=
  let d₁ := Real.sqrt ((p₁.x - p₂.x)^2 + (p₁.y - p₂.y)^2 + (p₁.z - p₂.z)^2)
  let d₂ := Real.sqrt ((p₂.x - p₃.x)^2 + (p₂.y - p₃.y)^2 + (p₂.z - p₃.z)^2)
  let d₃ := Real.sqrt ((p₃.x - p₁.x)^2 + (p₃.y - p₁.y)^2 + (p₃.z - p₁.z)^2)
  d₁ + d₂ + d₃

/-- The main theorem --/
theorem equilateral_triangle_perimeter
  (triangle : EquilateralTriangle)
  (points : TrianglePoints)
  (h_concurrent : are_concurrent points.B₁ points.C₂ points.C₁ points.A₂ points.A₁ points.B₂)
  (h_equal_perimeters : 
    triangle_perimeter triangle.A points.B₂ points.C₁ =
    triangle_perimeter triangle.B points.C₂ points.A₁ ∧
    triangle_perimeter triangle.B points.C₂ points.A₁ =
    triangle_perimeter triangle.C points.A₂ points.B₁) :
  triangle_perimeter triangle.A points.B₂ points.C₁ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l746_74644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l746_74653

theorem expression_equality : (3 - Real.pi) ^ 0 - (1/3 : Real) ^ (-1 : Int) + |2 - Real.sqrt 8| + 2 * Real.cos (45 * π / 180) = 3 * Real.sqrt 2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l746_74653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_by_8_and_next_by_25_l746_74689

theorem infinitely_many_divisible_by_8_and_next_by_25 :
  ∃ f : ℕ → ℕ, StrictMono f ∧ 
  (∀ n, (f n) % 8 = 0 ∧ ((f n) + 1) % 25 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_by_8_and_next_by_25_l746_74689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_gain_percentage_l746_74612

/-- A shop owner sells cloth and makes a profit. -/
structure ClothSale where
  /-- The amount of cloth sold in meters -/
  sold : ℚ
  /-- The amount of cloth whose selling price equals the profit, in meters -/
  profit_equiv : ℚ

/-- Calculate the gain percentage for a cloth sale -/
def gain_percentage (sale : ClothSale) : ℚ :=
  (sale.profit_equiv / sale.sold) * 100

/-- Theorem: The gain percentage for selling 40 meters of cloth with a profit 
    equivalent to the selling price of 10 meters is 25% -/
theorem cloth_sale_gain_percentage : 
  gain_percentage ⟨40, 10⟩ = 25 := by
  -- Unfold the definition of gain_percentage
  unfold gain_percentage
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_gain_percentage_l746_74612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l746_74645

-- Define the train parameters
noncomputable def train1_length : ℝ := 315
noncomputable def train2_length : ℝ := 270
noncomputable def train1_speed : ℝ := 80
noncomputable def train2_speed : ℝ := 50

-- Define the conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℝ := 1000 / 3600

-- Define the theorem
theorem trains_crossing_time :
  let total_length := train1_length + train2_length
  let relative_speed := (train1_speed + train2_speed) * km_hr_to_m_s
  let crossing_time := total_length / relative_speed
  ∃ ε > 0, |crossing_time - 16.2| < ε := by
  sorry

-- Note: We use ∃ ε > 0, |crossing_time - 16.2| < ε to represent "approximately equal"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l746_74645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_sets_between_l746_74664

open Finset

theorem number_of_sets_between : 
  ((Finset.powerset {1, 2, 3, 4, 5, 6}).filter (λ A => {1, 2} ⊂ A)).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_sets_between_l746_74664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_rotation_at_four_oclock_l746_74687

/-- Represents a circular clock face with a smaller disk rolling around it. -/
structure ClockWithDisk where
  clockRadius : ℝ
  diskRadius : ℝ
  initialPosition : ℝ  -- In radians, 0 represents 12 o'clock

/-- Calculates the angle traveled by the disk when it completes one full rotation. -/
noncomputable def angleTraveledForFullRotation (c : ClockWithDisk) : ℝ :=
  2 * Real.pi * c.diskRadius / (c.clockRadius - c.diskRadius)

/-- Theorem stating that for the given clock and disk dimensions, 
    the disk completes one full rotation after traveling 120 degrees. -/
theorem disk_rotation_at_four_oclock (c : ClockWithDisk) 
  (h1 : c.clockRadius = 20)
  (h2 : c.diskRadius = 10)
  (h3 : c.initialPosition = 0) :
  angleTraveledForFullRotation c = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_rotation_at_four_oclock_l746_74687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_keenan_essay_time_l746_74652

/-- Calculates the time needed to write an essay given the specified conditions -/
def essay_writing_time (essay_length : ℕ) (initial_speed : ℕ) (reduced_speed : ℕ) 
  (initial_hours : ℕ) (break_time : ℚ) (distraction_time : ℚ) : ℚ :=
  let initial_words := initial_speed * initial_hours
  let remaining_words := essay_length - initial_words
  let reduced_writing_time : ℚ := (remaining_words : ℚ) / (reduced_speed : ℚ)
  let total_writing_time : ℚ := (initial_hours : ℚ) + reduced_writing_time
  let break_count := ⌈total_writing_time⌉
  let total_break_time := break_count * break_time
  let total_time := total_writing_time + total_break_time + distraction_time
  total_time

/-- Proves that Keenan needs 5.5 hours to complete her essay -/
theorem keenan_essay_time : 
  essay_writing_time 1200 400 200 2 (1/4) (1/2) = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_keenan_essay_time_l746_74652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_routes_between_plains_cities_l746_74697

theorem routes_between_plains_cities 
  (total_cities : Nat) 
  (mountainous_cities : Nat) 
  (plains_cities : Nat) 
  (total_routes : Nat) 
  (mountainous_routes : Nat) 
  (h1 : total_cities = 100)
  (h2 : mountainous_cities = 30)
  (h3 : plains_cities = 70)
  (h4 : total_routes = 150)
  (h5 : mountainous_routes = 21)
  (h6 : total_cities = mountainous_cities + plains_cities)
  (h7 : ∀ (city : Nat), city < total_cities → ∃! (routes : Finset Nat), routes.card = 3) :
  ∃ plains_routes : Nat, plains_routes = 81 ∧ 
    total_routes = mountainous_routes + plains_routes + (total_routes - mountainous_routes - plains_routes) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_routes_between_plains_cities_l746_74697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digging_from_both_ends_is_faster_l746_74684

/-- Represents a tunnel digging scenario with two diggers -/
structure TunnelDigging where
  tunnel_length : ℝ
  slow_digger_rate : ℝ
  fast_digger_rate : ℝ
  (fast_is_faster : fast_digger_rate = 1.5 * slow_digger_rate)

/-- Calculates the total time for both diggers to complete half the tunnel each -/
noncomputable def time_half_each (scenario : TunnelDigging) : ℝ :=
  scenario.tunnel_length / (2 * scenario.slow_digger_rate) +
  scenario.tunnel_length / (2 * scenario.fast_digger_rate)

/-- Calculates the total time for diggers to meet in the middle -/
noncomputable def time_meet_in_middle (scenario : TunnelDigging) : ℝ :=
  2 * scenario.tunnel_length / (scenario.slow_digger_rate + scenario.fast_digger_rate)

/-- Theorem stating that digging from both ends is faster (and thus cheaper) -/
theorem digging_from_both_ends_is_faster (scenario : TunnelDigging) :
  time_meet_in_middle scenario < time_half_each scenario := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digging_from_both_ends_is_faster_l746_74684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l746_74666

open Nat Finset

-- Define a function to check if a number is a valid four-digit number greater than 3999
def isValidNumber (n : ℕ) : Bool :=
  n ≥ 4000 && n ≤ 9999

-- Define a function to get the middle two digits of a four-digit number
def middleDigits (n : ℕ) : ℕ × ℕ :=
  let d3 := (n / 10) % 10
  let d2 := (n / 100) % 10
  (d2, d3)

-- Define a function to check if the product of two digits exceeds 10
def productExceedsTen (d1 d2 : ℕ) : Bool :=
  d1 * d2 > 10

-- Main theorem
theorem count_valid_numbers : 
  (filter (fun n => 
    isValidNumber n && 
    let (d2, d3) := middleDigits n
    productExceedsTen d2 d3
  ) (range 10000)).card = 3480 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l746_74666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_x_production_l746_74610

/-- The number of widgets machine X produces per hour -/
def widgets_x : ℝ := sorry

/-- The number of widgets machine Y produces per hour -/
def widgets_y : ℝ := sorry

/-- The time it takes for machine X to produce 1080 widgets -/
def time_x : ℝ := sorry

/-- The time it takes for machine Y to produce 1080 widgets -/
def time_y : ℝ := sorry

/-- Machine Y produces 20% more widgets per hour than machine X -/
axiom production_rate : widgets_y = 1.2 * widgets_x

/-- Machine X takes 60 hours longer than machine Y to produce 1080 widgets -/
axiom time_difference : time_x = time_y + 60

/-- Both machines produce 1080 widgets -/
axiom total_widgets_x : 1080 = widgets_x * time_x
axiom total_widgets_y : 1080 = widgets_y * time_y

theorem machine_x_production : widgets_x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_x_production_l746_74610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mix_price_is_three_dollars_l746_74675

/-- Calculates the price per pound of a mix given the weights and prices of its components -/
noncomputable def price_per_pound_of_mix (total_weight : ℝ) (cashew_weight : ℝ) (peanut_weight : ℝ) 
  (cashew_price : ℝ) (peanut_price : ℝ) : ℝ :=
  ((cashew_weight * cashew_price) + (peanut_weight * peanut_price)) / total_weight

/-- Theorem: The price per pound of the mix is $3.00 -/
theorem mix_price_is_three_dollars :
  let total_weight : ℝ := 60
  let cashew_weight : ℝ := 10
  let peanut_weight : ℝ := 50
  let cashew_price : ℝ := 6
  let peanut_price : ℝ := 2.4
  price_per_pound_of_mix total_weight cashew_weight peanut_weight cashew_price peanut_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mix_price_is_three_dollars_l746_74675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_is_105_l746_74623

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a trapezium given its four vertices -/
noncomputable def trapeziumArea (a b c d : Point) : ℝ :=
  let ab := b.x - a.x
  let cd := c.x - d.x
  let height := c.y - a.y
  (1 / 2) * (ab + cd) * height

/-- Theorem stating that the area of the given trapezium is 105 square units -/
theorem trapezium_area_is_105 :
  let a : Point := ⟨2, 3⟩
  let b : Point := ⟨14, 3⟩
  let c : Point := ⟨18, 10⟩
  let d : Point := ⟨0, 10⟩
  trapeziumArea a b c d = 105 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_is_105_l746_74623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_section_dimensions_l746_74673

/-- Represents the side length of the square section -/
def x : ℕ := sorry

/-- Represents the length of the rectangular section -/
def rectLength : ℕ := sorry

/-- The total area of the field -/
def totalArea : ℕ := x^2 + x * rectLength

/-- Theorem stating the possible dimensions of the square section -/
theorem square_section_dimensions :
  (25 ≤ rectLength ∧ rectLength ≤ 30) →
  (250 ≤ totalArea ∧ totalArea ≤ 300) →
  (x = 7 ∨ x = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_section_dimensions_l746_74673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l746_74651

noncomputable def f (x : ℝ) : ℝ := 5 / x^2 - 3 * x^2 + 2

theorem f_inequality_range (x : ℝ) :
  f 1 < f (Real.log x / Real.log 3) ↔ x ∈ Set.Ioo (1/3 : ℝ) 1 ∪ Set.Ioo 1 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l746_74651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_first_two_numbers_l746_74605

theorem product_of_first_two_numbers 
  (a b c : ℕ+) 
  (coprime_abc : Nat.Coprime a.val (Nat.gcd b.val c.val))
  (product_bc : b * c = 1073)
  (sum_abc : a + b + c = 85) :
  a * b = 703 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_first_two_numbers_l746_74605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_cells_disappear_l746_74688

-- Define the grid
def Grid := ℕ → ℕ → Bool

-- Define the recoloring rule
def recolor (g : Grid) (i j : ℕ) : Bool :=
  let count := (g i j).toNat + (g (i+1) j).toNat + (g i (j+1)).toNat
  count ≥ 2

-- Define the next state of the grid after one step
def next_state (g : Grid) : Grid :=
  λ i j ↦ recolor g i j

-- Define the state of the grid after t steps
def state_after (g : Grid) : ℕ → Grid
  | 0 => g
  | t+1 => next_state (state_after g t)

-- Count the number of black cells in the grid
noncomputable def count_black (g : Grid) : ℕ :=
  sorry

-- The main theorem
theorem black_cells_disappear (g : Grid) (n : ℕ) 
  (h : count_black g = n) : 
  ∃ t : ℕ, t ≤ n ∧ count_black (state_after g t) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_cells_disappear_l746_74688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equality_iff_angle_equality_l746_74627

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- Define the theorem
theorem sin_equality_iff_angle_equality (t : Triangle) : 
  Real.sin t.A = Real.sin t.B ↔ t.A = t.B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equality_iff_angle_equality_l746_74627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_relation_l746_74604

/-- The eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity_relation (a : ℝ) :
  a > 1 →
  let e₁ := eccentricity a 1
  let e₂ := eccentricity 2 1
  e₂ = Real.sqrt 3 * e₁ →
  a = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_relation_l746_74604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janice_movie_time_l746_74658

/-- The number of hours Janice has before the movie starts -/
noncomputable def time_before_movie (minutes_left : ℚ) : ℚ :=
  minutes_left / 60

/-- Theorem: Given 35 minutes left, Janice has 35/60 hours before the movie starts -/
theorem janice_movie_time : time_before_movie 35 = 35 / 60 := by
  -- Unfold the definition of time_before_movie
  unfold time_before_movie
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_janice_movie_time_l746_74658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outfit_count_l746_74620

/-- The number of shirts available -/
def num_shirts : ℕ := 5

/-- The number of pairs of pants available -/
def num_pants : ℕ := 3

/-- The number of pairs of shoes available -/
def num_shoes : ℕ := 2

/-- An outfit consists of one shirt, one pair of pants, and one pair of shoes -/
def outfit := Fin num_shirts × Fin num_pants × Fin num_shoes

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_shirts * num_pants * num_shoes

theorem outfit_count : total_outfits = 30 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_outfit_count_l746_74620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_sum_c_and_d_l746_74662

-- Define the vertices of the quadrilateral
def v1 : ℝ × ℝ := (1, 2)
def v2 : ℝ × ℝ := (5, 6)
def v3 : ℝ × ℝ := (6, 5)
def v4 : ℝ × ℝ := (2, 1)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the perimeter of the quadrilateral
noncomputable def perimeter : ℝ :=
  distance v1 v2 + distance v2 v3 + distance v3 v4 + distance v4 v1

-- Theorem statement
theorem quadrilateral_perimeter :
  perimeter = 10 * Real.sqrt 2 + 0 * Real.sqrt 10 := by
  sorry

-- Theorem for the sum of c and d
theorem sum_c_and_d : ∃ (c d : ℤ), perimeter = c * Real.sqrt 2 + d * Real.sqrt 10 ∧ c + d = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_sum_c_and_d_l746_74662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alan_coins_theorem_l746_74649

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  five_cent : ℕ
  ten_cent : ℕ
  twenty_cent : ℕ

/-- Calculates the number of different values that can be obtained from a given coin configuration -/
def different_values (coins : CoinCounts) : ℕ :=
  79 - 3 * coins.five_cent - 2 * coins.ten_cent

theorem alan_coins_theorem (coins : CoinCounts) :
  coins.five_cent + coins.ten_cent + coins.twenty_cent = 20 →
  different_values coins = 24 →
  coins.twenty_cent = 5 := by
  intro h1 h2
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alan_coins_theorem_l746_74649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_square_area_ratio_l746_74667

/-- Given a square S and a rectangle R where:
    - The longer side of R is 20% more than a side of S
    - The shorter side of R is 20% less than a side of S
    Prove that the ratio of the area of R to the area of S is 24/25 -/
theorem rectangle_square_area_ratio (s : ℝ) (h : s > 0) :
  (1.2 * s) * (0.8 * s) / (s * s) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_square_area_ratio_l746_74667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_coordinates_l746_74654

-- Define the plane figure
def planeFigure (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ a * p.1^3 ≤ p.2 ∧ p.2 ≤ a}

-- Define the density function (constant ρ = 1)
def density : ℝ × ℝ → ℝ := λ _ ↦ 1

-- Define the centroid function (placeholder)
noncomputable def centroid (S : Set (ℝ × ℝ)) (ρ : ℝ × ℝ → ℝ) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem centroid_coordinates (a : ℝ) (h : a > 0) :
  ∃ c : ℝ × ℝ, c = centroid (planeFigure a) density ∧ c = (0.4, 4*a/7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_coordinates_l746_74654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l746_74640

theorem repeating_decimal_to_fraction : 
  ∀ (x : ℚ), x = 36/99 → x = 4/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l746_74640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peanuts_for_two_oz_oil_l746_74679

/-- Represents the recipe for peanut butter -/
structure PeanutButterRecipe where
  totalWeight : ℚ
  oilWeight : ℚ
  oilRatio : ℚ

/-- Calculates the amount of peanuts used for a given amount of oil -/
def peanutsForOil (recipe : PeanutButterRecipe) (oil : ℚ) : ℚ :=
  let peanutWeight := recipe.totalWeight - recipe.oilWeight
  (peanutWeight / recipe.oilWeight) * oil

/-- Theorem stating the amount of peanuts used for 2 ounces of oil -/
theorem peanuts_for_two_oz_oil (recipe : PeanutButterRecipe)
    (h1 : recipe.totalWeight = 20)
    (h2 : recipe.oilWeight = 4)
    (h3 : recipe.oilRatio = 2) :
  peanutsForOil recipe 2 = 8 := by
  sorry

#eval peanutsForOil { totalWeight := 20, oilWeight := 4, oilRatio := 2 } 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peanuts_for_two_oz_oil_l746_74679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_ten_l746_74670

-- Define the custom operations
noncomputable def custom_div (a b : ℝ) : ℝ := a - b
noncomputable def custom_sub (a b : ℝ) : ℝ := a / b
noncomputable def custom_mul (a b : ℝ) : ℝ := a * b
noncomputable def custom_add (a b : ℝ) : ℝ := a + b

-- Define the equation
noncomputable def equation (x : ℝ) : ℝ :=
  custom_add (custom_mul (custom_sub 9 (custom_div 8 7)) 5) x

-- Theorem statement
theorem solution_is_ten :
  ∃ (x : ℝ), equation x = 13.285714285714286 ∧ x = 10 := by
  -- Provide the value of x
  let x : ℝ := 10
  -- Assert that this x satisfies the equation
  have h1 : equation x = 13.285714285714286 := by sorry
  -- Assert that x is equal to 10
  have h2 : x = 10 := rfl
  -- Combine the two assertions to prove the existence
  exact ⟨x, ⟨h1, h2⟩⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_ten_l746_74670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_base_range_l746_74686

theorem decreasing_exponential_base_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2 - a) ^ x > (2 - a) ^ y) → 
  1 < a ∧ a < 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_base_range_l746_74686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_curve_l746_74613

-- Define the curve C
noncomputable def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- Define the function we want to maximize
noncomputable def f (x y : ℝ) : ℝ := (y + 1) / (x + 1)

-- State the theorem
theorem max_value_on_curve :
  ∃ (M : ℝ), M = 2 + Real.sqrt 6 ∧
  (∀ (x y : ℝ), C x y → f x y ≤ M) ∧
  (∃ (x y : ℝ), C x y ∧ f x y = M) := by
  sorry

#check max_value_on_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_curve_l746_74613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_max_area_line_l746_74618

/-- A parabola with equation y^2 = 8x -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y : ℝ, eq x y ↔ y^2 = 8*x

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.eq x y

/-- The angle bisector of ∠APB is perpendicular to the x-axis -/
def angle_bisector_perpendicular (p : Parabola) (P A B : PointOnParabola p) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ 
    (A.y - P.y) / (A.x - P.x) = k ∧
    (B.y - P.y) / (B.x - P.x) = -1/k

/-- Area of a triangle given three points -/
noncomputable def area_triangle (P A B : PointOnParabola p) : ℝ :=
  (1/2) * abs ((A.x - P.x) * (B.y - P.y) - (B.x - P.x) * (A.y - P.y))

theorem parabola_max_area_line 
  (p : Parabola)
  (P : PointOnParabola p)
  (h_P : P.x = 2 ∧ P.y = 4)
  (A B : PointOnParabola p)
  (h_A : A.y ≤ 0)
  (h_B : B.y ≤ 0)
  (h_bisector : angle_bisector_perpendicular p P A B) :
  ∃ b : ℝ, 
    (∀ x y : ℝ, y = -x + b → (A.x = x ∧ A.y = y) ∨ (B.x = x ∧ B.y = y)) ∧
    (∀ b' : ℝ, 
      (∀ x y : ℝ, y = -x + b' → (A.x = x ∧ A.y = y) ∨ (B.x = x ∧ B.y = y)) →
      area_triangle P A B ≤ area_triangle P 
        { x := A.x, y := -A.x, on_parabola := by sorry }
        { x := B.x, y := -B.x, on_parabola := by sorry }) :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_max_area_line_l746_74618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_annual_income_is_403200_l746_74647

/-- Proves that A's annual income is 403200 given the specified conditions -/
def a_annual_income (c_income : ℕ) : ℕ :=
  let b_income := c_income + c_income / 100 * 12
  let part := b_income / 2
  let a_monthly := part * 5
  a_monthly * 12

theorem a_annual_income_is_403200 : a_annual_income 12000 = 403200 := by
  rfl

#eval a_annual_income 12000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_annual_income_is_403200_l746_74647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_from_series_sum_l746_74661

-- Define the series sum
noncomputable def seriesSum (θ : Real) : Real := ∑' n, (Real.cos θ) ^ (2 * n)

-- State the theorem
theorem cos_double_angle_from_series_sum (θ : Real) 
  (h : seriesSum θ = 7) : Real.cos (2 * θ) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_from_series_sum_l746_74661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l746_74641

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem equation_solutions :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
  (∀ x ∈ s, 9 * x^2 - 36 * (floor x) + 20 = 0) ∧
  (∀ x : ℝ, 9 * x^2 - 36 * (floor x) + 20 = 0 → x ∈ s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l746_74641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_when_proposition_is_false_l746_74625

-- Define the function f(x)
noncomputable def f (k x : ℝ) : ℝ := k * (4^x) - k * (2^(x+1)) + 6*(k-5)

-- State the theorem
theorem range_of_k_when_proposition_is_false :
  (∃ x ∈ Set.Icc 0 1, f k x = 0) → k ∈ Set.Icc 5 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_when_proposition_is_false_l746_74625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_lengths_equal_l746_74696

-- Define the parameters of the ellipse
noncomputable def a : ℝ := 4
noncomputable def b : ℝ := 2 * Real.sqrt 3

-- Define the parameter k for the hyperbola
variable (k : ℝ)

-- Define the focal length of the ellipse
noncomputable def focal_length_ellipse : ℝ := Real.sqrt (a^2 - b^2)

-- Define the focal length of the hyperbola
noncomputable def focal_length_hyperbola (k : ℝ) : ℝ := Real.sqrt ((16 - k) - (12 - k))

-- State the theorem
theorem focal_lengths_equal (h : 12 < k ∧ k < 16) :
  focal_length_ellipse = focal_length_hyperbola k := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_lengths_equal_l746_74696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l746_74695

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and focal distance c -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_c_eq : c^2 = a^2 - b^2

/-- The point where the line through F₂ intersects 2bx + ay = 0 -/
noncomputable def intersection_point (e : Ellipse) : ℝ × ℝ := (e.c / 2, -e.b * e.c / e.a)

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

theorem ellipse_eccentricity_half (e : Ellipse) 
  (h_on_circle : (e.c / 2)^2 + (-e.b * e.c / e.a)^2 = e.c^2) :
  eccentricity e = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l746_74695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_to_third_ratio_l746_74616

-- Define the three numbers
def A : ℚ := sorry
def B : ℚ := 30
def C : ℚ := sorry

-- State the conditions
axiom sum_eq_98 : A + B + C = 98
axiom ratio_A_B : A / B = 2 / 3

-- Define the ratio of B to C
def ratio_B_C : ℚ × ℚ := (B, C)

-- Theorem to prove
theorem second_to_third_ratio :
  ratio_B_C = (5, 8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_to_third_ratio_l746_74616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l746_74657

def P : Set ℝ := {x | 1 < x ∧ x < 3}
def Q : Set ℝ := {x | 2 < x}

theorem intersection_of_P_and_Q : P ∩ Q = Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l746_74657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_to_asymptote_distance_l746_74634

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (positive_a : a > 0)
  (positive_b : b > 0)

/-- The distance from the vertex of a hyperbola to its asymptote -/
noncomputable def vertex_to_asymptote_distance (h : Hyperbola) : ℝ :=
  h.a * h.b / Real.sqrt (h.a^2 + h.b^2)

/-- Theorem: For the hyperbola x²/16 - y²/9 = 1, the distance from its vertex to its asymptote is 12/5 -/
theorem hyperbola_vertex_to_asymptote_distance :
  let h : Hyperbola := ⟨4, 3, by norm_num, by norm_num⟩
  vertex_to_asymptote_distance h = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_to_asymptote_distance_l746_74634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l746_74607

-- Define set A
def A : Set ℝ := {x | x < 1}

-- Define set B
def B : Set ℝ := {y | y ≥ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l746_74607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_three_l746_74602

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x - 1 else -(2^(-x) + 2*(-x) - 1)

-- State the theorem
theorem f_neg_one_eq_neg_three :
  f (-1) = -3 :=
by
  -- Unfold the definition of f
  unfold f
  -- Simplify the if-then-else expression
  simp
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_three_l746_74602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_51_eq_101_l746_74663

/-- A sequence {aₙ} with a₁ = 1 and aₙ₊₁ - aₙ = 2 for all n ≥ 1 -/
def a : ℕ → ℤ
  | 0 => -1  -- Add a case for 0 to make the function total
  | 1 => 1
  | n + 2 => a (n + 1) + 2

/-- The 51st term of the sequence {aₙ} is 101 -/
theorem a_51_eq_101 : a 51 = 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_51_eq_101_l746_74663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_ratio_l746_74693

def admission_problem (adult_fee child_fee total : ℕ) : Prop :=
  ∃ (adults children : ℕ),
    adults > 0 ∧
    children > 0 ∧
    adult_fee * adults + child_fee * children = total ∧
    ∀ (a c : ℕ),
      a > 0 → c > 0 →
      adult_fee * a + child_fee * c = total →
      |(a : ℚ) / c - 1| ≥ |(adults : ℚ) / children - 1|

theorem closest_ratio :
  admission_problem 30 10 2400 →
  ∃ (adults children : ℕ),
    adults = 40 ∧
    children = 120 ∧
    admission_problem 30 10 2400 :=
by
  sorry

#check closest_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_ratio_l746_74693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_product_l746_74648

noncomputable def f (x : ℝ) : ℝ := (x^2 - 19) / (x^3 - 3*x^2 - 4*x + 12)

noncomputable def partial_fraction (A B C : ℝ) (x : ℝ) : ℝ := A / (x - 1) + B / (x + 3) + C / (x - 4)

theorem partial_fraction_decomposition_product :
  ∃ A B C : ℝ, 
    (∀ x : ℝ, x ≠ 1 ∧ x ≠ -3 ∧ x ≠ 4 → f x = partial_fraction A B C x) ∧
    A * B * C = 15 / 196 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_product_l746_74648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_zeros_difference_l746_74691

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
noncomputable def vertex (p : Parabola) : ℝ × ℝ := (3, -3)

/-- The parabola passes through the point (5, 15) -/
def passes_through (p : Parabola) : Prop :=
  15 = p.a * 5^2 + p.b * 5 + p.c

/-- The zeros of the quadratic ax^2 + bx + c -/
noncomputable def zeros (p : Parabola) : ℝ × ℝ :=
  let x₁ := (-p.b + Real.sqrt (p.b^2 - 4*p.a*p.c)) / (2*p.a)
  let x₂ := (-p.b - Real.sqrt (p.b^2 - 4*p.a*p.c)) / (2*p.a)
  (max x₁ x₂, min x₁ x₂)

theorem parabola_zeros_difference (p : Parabola) :
  vertex p = (3, -3) →
  passes_through p →
  let (q, r) := zeros p
  q - r = 2 * Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_zeros_difference_l746_74691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_channel_flow_rates_l746_74665

/-- Represents a water channel system with nodes A, B, C, D, E, F, G, H -/
structure WaterChannelSystem where
  /-- Flow rate in channel BC -/
  q₀ : ℝ
  /-- Flow rate is conserved along any path -/
  flow_conservation : ∀ (path : List (Fin 8)), path.length ≥ 2 → 
    (path.map (fun i => match i with
      | 0 => (1 : ℝ) | 1 => (1 : ℝ) | 2 => (1 : ℝ) | 3 => (1 : ℝ)
      | 4 => (1 : ℝ) | 5 => (1 : ℝ) | 6 => (1 : ℝ) | 7 => (1 : ℝ))).sum = 0

/-- The flow rates in the water channel system satisfy the given conditions -/
theorem water_channel_flow_rates (system : WaterChannelSystem) :
  ∃ (q_AB q_AH q_A : ℝ),
    q_AB = (1/2) * system.q₀ ∧
    q_AH = (3/4) * system.q₀ ∧
    q_A = (7/4) * system.q₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_channel_flow_rates_l746_74665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_A_has_winning_strategy_l746_74674

/-- Represents a player in the game -/
inductive Player : Type where
  | A : Player
  | B : Player
deriving Repr, DecidableEq

/-- Represents the game state -/
structure GameState where
  total_people : Nat
  current_player : Player
  distance_between_players : Nat
deriving Repr

/-- Defines a valid initial game state -/
def valid_initial_state (s : GameState) : Prop :=
  s.total_people = 2003 ∧ s.current_player = Player.A ∧ s.distance_between_players ≥ 1

/-- Defines a move in the game -/
def move (s : GameState) : GameState :=
  { total_people := s.total_people - 1,
    current_player := if s.current_player = Player.A then Player.B else Player.A,
    distance_between_players := s.distance_between_players - 1 }

/-- Defines when a player wins -/
def wins (p : Player) (s : GameState) : Prop :=
  s.distance_between_players = 0 ∧ s.current_player = p

/-- Theorem stating that player A has a winning strategy -/
theorem player_A_has_winning_strategy :
  ∀ (s : GameState), valid_initial_state s → ∃ (n : Nat), wins Player.A (n.iterate move s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_A_has_winning_strategy_l746_74674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_function_characterization_l746_74682

def GoodFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc 0 1 → f x ∈ Set.Icc 0 1) ∧
  (∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → 
    (x - y)^2 ≤ |f x - f y| ∧ |f x - f y| ≤ |x - y|)

theorem good_function_characterization (f : ℝ → ℝ) :
  GoodFunction f →
  ∃ c : ℝ, (∀ x ∈ Set.Icc 0 1, f x = x + c) ∨ 
           (∀ x ∈ Set.Icc 0 1, f x = -x + c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_function_characterization_l746_74682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_is_2017_l746_74628

def f (x : ℝ) : ℝ := 2017 * x^2 - 2018 * x + 2019 * 2020

def interval (t : ℝ) : Set ℝ := { x | t ≤ x ∧ x ≤ t + 2 }

noncomputable def f_max (t : ℝ) : ℝ := ⨆ (x ∈ interval t), f x

noncomputable def f_min (t : ℝ) : ℝ := ⨅ (x ∈ interval t), f x

theorem min_difference_is_2017 :
  ⨅ (t : ℝ), (f_max t - f_min t) = 2017 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_is_2017_l746_74628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_level_curves_of_f_l746_74639

/-- The function whose level curves we're finding -/
noncomputable def f (x y : ℝ) : ℝ := Real.sqrt (8 - |x - 8| - |y|)

/-- Predicate for a point (x, y) being on a level curve with parameter C -/
def on_level_curve (x y C : ℝ) : Prop :=
  (y = 8 - |x - 8| - C^2 ∨ y = -(8 - |x - 8| - C^2)) ∧ 0 ≤ C ∧ C ≤ Real.sqrt 8

/-- Theorem stating that the level curves of f are correctly described by on_level_curve -/
theorem level_curves_of_f :
  ∀ (x y C : ℝ), f x y = C ↔ on_level_curve x y C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_level_curves_of_f_l746_74639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_calculation_l746_74622

theorem class_size_calculation (n : ℕ) 
  (h1 : n > 7) 
  (h2 : (4 * 95 + (n - 7) * 45 : ℚ) / n = 47.32142857142857) : n = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_calculation_l746_74622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l746_74676

-- Define the function f(x) = x + 1/x
noncomputable def f (x : ℝ) : ℝ := x + 1/x

-- Theorem statement
theorem f_properties :
  -- Part 1: f is strictly increasing on (1, +∞)
  (∀ x y : ℝ, 1 < x ∧ x < y → f x < f y) ∧
  -- Part 2: Minimum value on [2, 6]
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 6 → f 2 ≤ f x) ∧
  f 2 = 5/2 ∧
  -- Part 3: Maximum value on [2, 6]
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 6 → f x ≤ f 6) ∧
  f 6 = 37/6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l746_74676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l746_74629

/-- The time (in seconds) for a train to pass an electric pole -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  train_length / (train_speed_kmh * 1000 / 3600)

/-- Theorem stating that a 200m train traveling at 80 km/h passes an electric pole in approximately 9 seconds -/
theorem train_passing_time_approx :
  ∃ ε > 0, |train_passing_time 200 80 - 9| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l746_74629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_formula_l746_74633

theorem cosine_difference_formula (α β : ℝ) : 
  Real.cos (α - β) = Real.cos α * Real.cos β + Real.sin α * Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_formula_l746_74633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l746_74643

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin x * Real.cos x - (Real.sqrt 3 / 2) * (Real.cos x)^2 + Real.sqrt 3 / 4

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  ∃ T, T = Real.pi ∧
  ∀ x₀ : ℝ, x₀ ∈ Set.Icc 0 (Real.pi / 2) → f x₀ = 1/2 → f (2 * x₀) = -(Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l746_74643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l746_74672

/-- A hyperbola with given properties has asymptotes y = ±√3x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- hyperbola equation
  (a^2 + b^2) / a^2 = 4 →                   -- eccentricity is 2
  (1, 0) ∈ {(x, y) | x^2 / a^2 - y^2 / b^2 = 1} →  -- focus at (1,0)
  (∀ x y : ℝ, y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) →   -- asymptotes equation
  True :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l746_74672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carolyn_sum_is_24_l746_74699

def game_numbers : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def has_divisor_in_list (n : Nat) (l : List Nat) : Bool :=
  l.any (fun m => m ≠ n ∧ n % m = 0)

def remove_divisors (n : Nat) (l : List Nat) : List Nat :=
  l.filter (fun m => n % m ≠ 0 ∨ m = n)

def carolyn_move (l : List Nat) : Option Nat :=
  l.find? (fun n => has_divisor_in_list n l)

def game_step (l : List Nat) : List Nat :=
  match carolyn_move l with
  | none => []
  | some n => remove_divisors n (l.filter (· ≠ n))

def game_play (l : List Nat) : List Nat :=
  let rec aux (current : List Nat) (removed : List Nat) (fuel : Nat) : List Nat :=
    match fuel with
    | 0 => removed
    | fuel'+1 => 
      match carolyn_move current with
      | none => removed
      | some n => aux (game_step current) (n :: removed) fuel'
  aux l [] 10

theorem carolyn_sum_is_24 :
  (game_play (game_step (game_numbers.filter (· ≠ 5)))).sum = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carolyn_sum_is_24_l746_74699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_M_l746_74692

def is_valid (M : ℕ) : Prop :=
  (∃ i ∈ ({M, M+1, M+2, M+3} : Set ℕ), i % 8 = 0) ∧
  (∃ i ∈ ({M, M+1, M+2, M+3} : Set ℕ), i % 9 = 0) ∧
  (∃ i ∈ ({M, M+1, M+2, M+3} : Set ℕ), i % 25 = 0) ∧
  (∃ i ∈ ({M, M+1, M+2, M+3} : Set ℕ), i % 49 = 0)

theorem smallest_valid_M :
  is_valid 196 ∧ ∀ M < 196, ¬is_valid M := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_M_l746_74692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_four_to_eighth_power_l746_74630

theorem fourth_root_of_four_to_eighth_power : (4 : ℝ) ^ (1/4) ^ 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_four_to_eighth_power_l746_74630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_above_x_axis_l746_74601

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^2 + a*x + b^2

-- Define the sample space
def sample_space : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

-- Define the event where the graph of f(x) lies entirely above the x-axis
def event : Set (ℝ × ℝ) := {p ∈ sample_space | ∀ x, f p.1 p.2 x > 0}

-- State the theorem
theorem probability_above_x_axis :
  (MeasureTheory.volume event) / (MeasureTheory.volume sample_space) = 5/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_above_x_axis_l746_74601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_line_l746_74683

noncomputable section

-- Define the ellipses and constants
def C1 (x y : ℝ) : Prop := y^2 / 9 + x^2 / 4 = 1

def C2 (x y : ℝ) (lambda : ℝ) : Prop := y^2 / (9 * lambda^2) + x^2 / (4 * lambda^2) = 1

-- Define the line
def line (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

-- Define the area of triangle OAB
def area (k : ℝ) : ℝ := (27 * abs k) / (9 + 4 * k^2)

-- Theorem statement
theorem max_area_line (lambda : ℝ) (h1 : lambda > 1) :
  ∃ k : ℝ, (k = 3/2 ∨ k = -3/2) ∧ 
  ∀ m : ℝ, area k ≥ area m :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_line_l746_74683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_increasing_and_bounded_l746_74655

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => Real.sqrt 2 * a n

theorem a_increasing_and_bounded : 
  (∀ n : ℕ, a n < a (n + 1)) ∧ 
  (∀ n : ℕ, a n < 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_increasing_and_bounded_l746_74655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_symmetrical_about_center_l746_74606

/-- A parallelogram is a quadrilateral with opposite sides parallel -/
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ
  parallel_sides : (vertices 0 - vertices 1) = (vertices 3 - vertices 2) ∧
                   (vertices 1 - vertices 2) = (vertices 0 - vertices 3)

/-- The center of a parallelogram -/
noncomputable def center (p : Parallelogram) : ℝ × ℝ :=
  ((p.vertices 0 + p.vertices 2) : ℝ × ℝ) / 2

/-- A figure is symmetrical about a point if rotating it 180° around that point
    results in the same figure -/
def symmetrical_about (p : Parallelogram) (c : ℝ × ℝ) : Prop :=
  ∀ (i : Fin 4), p.vertices i = 2 * c - p.vertices i

/-- Theorem: A parallelogram is symmetrical about its center -/
theorem parallelogram_symmetrical_about_center (p : Parallelogram) :
  symmetrical_about p (center p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_symmetrical_about_center_l746_74606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_ducks_percentage_l746_74631

/-- The percentage of green ducks in both ponds combined -/
def percentage_green_ducks (small_pond_ducks large_pond_ducks : ℕ)
  (small_pond_green_percent large_pond_green_percent : ℚ) : ℚ :=
  let total_ducks := small_pond_ducks + large_pond_ducks
  let small_pond_green := (small_pond_green_percent * small_pond_ducks) / 100
  let large_pond_green := (large_pond_green_percent * large_pond_ducks) / 100
  let total_green := small_pond_green + large_pond_green
  (total_green / total_ducks) * 100

theorem green_ducks_percentage :
  percentage_green_ducks 20 80 20 15 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_ducks_percentage_l746_74631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_geometric_sequence_l746_74608

def geometric_sequence (a₁ : ℕ) (r : ℚ) : ℕ → ℚ
  | n => a₁ * r^(n - 1)

theorem fourth_term_of_geometric_sequence
  (a₁ : ℕ)
  (a₅ : ℕ)
  (h₁ : a₁ = 5)
  (h₂ : a₅ = 1280)
  (h₃ : ∃ r : ℚ, geometric_sequence a₁ r 5 = a₅) :
  geometric_sequence a₁ (Classical.choose h₃) 4 = 320 := by
  sorry

#check fourth_term_of_geometric_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_geometric_sequence_l746_74608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_function_l746_74680

noncomputable def f (x : ℝ) : ℝ := Real.sin (-3 * x + Real.pi / 6)

theorem transformed_function (x : ℝ) : 
  Real.sin (2 * Real.pi / 3 - 3 * x / 2) = 
  f ((x - Real.pi / 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_function_l746_74680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_percentage_l746_74603

/-- Represents a faulty meter with its weight in grams -/
structure FaultyMeter where
  weight : ℕ

/-- Represents the usage ratio of a meter -/
structure UsageRatio where
  ratio : ℕ

/-- Calculates the profit percentage for a single meter -/
def profitPercentage (meter : FaultyMeter) : ℚ :=
  (1000 - meter.weight : ℚ) / 10

/-- Calculates the total profit percentage given a list of meters and their usage ratios -/
def totalProfitPercentage (meters : List FaultyMeter) (ratios : List UsageRatio) : ℚ :=
  let totalRatio := (ratios.map (λ r => r.ratio)).sum
  let weightedProfits := (List.zip meters ratios).map (λ (m, r) => profitPercentage m * r.ratio)
  weightedProfits.sum / totalRatio

/-- The main theorem stating the total profit percentage for the given scenario -/
theorem shopkeeper_profit_percentage :
  let meters := [FaultyMeter.mk 900, FaultyMeter.mk 850, FaultyMeter.mk 950]
  let ratios := [UsageRatio.mk 3, UsageRatio.mk 2, UsageRatio.mk 1]
  totalProfitPercentage meters ratios = 65/600 := by
  sorry

#eval totalProfitPercentage 
  [FaultyMeter.mk 900, FaultyMeter.mk 850, FaultyMeter.mk 950]
  [UsageRatio.mk 3, UsageRatio.mk 2, UsageRatio.mk 1]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_percentage_l746_74603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_monotonic_subsequence_l746_74617

theorem infinite_monotonic_subsequence
  (u : ℕ → ℝ) :
  ∃ (φ : ℕ → ℕ), StrictMono φ ∧
    ((∀ n, u (φ n) < u (φ (n + 1))) ∨
     (∀ n, u (φ n) > u (φ (n + 1)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_monotonic_subsequence_l746_74617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_network_construction_l746_74638

/-- A type representing a city --/
structure City where
  id : Nat

/-- A type representing a flight route between two cities --/
structure FlightRoute where
  source : City
  destination : City

/-- A type representing a flight network --/
structure FlightNetwork where
  cities : Finset City
  routes : Finset FlightRoute

/-- A function to check if there's a direct flight between two cities --/
def hasDirectFlight (network : FlightNetwork) (source destination : City) : Prop :=
  ∃ (route : FlightRoute), route ∈ network.routes ∧ route.source = source ∧ route.destination = destination

/-- A function to check if there's a path with at most one stopover between two cities --/
def hasPathWithAtMostOneStopover (network : FlightNetwork) (source destination : City) : Prop :=
  hasDirectFlight network source destination ∨
  ∃ (intermediate : City), hasDirectFlight network source intermediate ∧ hasDirectFlight network intermediate destination

/-- The main theorem stating that for any number of cities n ≥ 5, 
    it's possible to construct a flight network with the desired property --/
theorem flight_network_construction (n : Nat) (h : n ≥ 5) :
  ∃ (network : FlightNetwork),
    (network.cities.card = n) ∧
    (∀ (city1 city2 : City), city1 ∈ network.cities → city2 ∈ network.cities → city1 ≠ city2 →
      hasPathWithAtMostOneStopover network city1 city2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_network_construction_l746_74638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_8pi_div_3_l746_74600

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < Real.pi/2 then Real.sqrt 3 * Real.tan x - 1
  else 0  -- We define it as 0 outside the given interval, as we don't have information about it

-- State the theorem
theorem f_value_at_8pi_div_3 :
  -- Conditions
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x, f (x + Real.pi) = f x) →  -- f has period π
  -- Conclusion
  f (8*Real.pi/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_8pi_div_3_l746_74600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_distance_range_l746_74642

/-- The distance between a point and a line -/
noncomputable def distancePointLine (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

theorem tangent_circle_distance_range :
  ∀ t : ℝ,
  let P := (t, 2 - t)
  let Q := (t / (2 * t^2 - 4 * t + 4), (2 - t) / (2 * t^2 - 4 * t + 4))
  let d := distancePointLine Q.1 Q.2 1 1 (-2)
  Real.sqrt 2 / 2 < d ∧ d ≤ Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_distance_range_l746_74642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_configuration_l746_74690

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a function to calculate the area of a triangle given three points
noncomputable def triangle_area (A B C : Point) : ℝ := 
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Define the problem statement
theorem unique_point_configuration (n : ℕ) : 
  (n > 3) →
  (∃ (points : Fin n → Point) (r : Fin n → ℝ),
    (∀ (i j k : Fin n), i.val < j.val ∧ j.val < k.val → 
      ¬(∃ (t : ℝ), (1 - t) • (points i).1 + t • (points j).1 = (points k).1 ∧
                    (1 - t) • (points i).2 + t • (points j).2 = (points k).2)) ∧
    (∀ (i j k : Fin n), i.val < j.val ∧ j.val < k.val → 
      triangle_area (points i) (points j) (points k) = r i + r j + r k)) →
  n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_configuration_l746_74690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ad_cost_theorem_l746_74671

/-- Represents the dimensions of an ad space -/
structure AdSpaceDimensions where
  length : ℝ
  width : ℝ

/-- Represents the details of an ad purchase -/
structure AdPurchase where
  numSpaces : ℕ
  dimensions : AdSpaceDimensions
  costPerSquareFoot : ℝ

/-- Calculates the total cost for multiple companies with identical ad purchases -/
def totalCost (purchase : AdPurchase) (numCompanies : ℕ) : ℝ :=
  let areaPerSpace := purchase.dimensions.length * purchase.dimensions.width
  let totalArea := areaPerSpace * (purchase.numSpaces : ℝ)
  let costPerCompany := totalArea * purchase.costPerSquareFoot
  costPerCompany * (numCompanies : ℝ)

/-- Theorem statement for the ad cost problem -/
theorem ad_cost_theorem (purchase : AdPurchase) (numCompanies : ℕ) :
  purchase.dimensions.length = 12 →
  purchase.dimensions.width = 5 →
  purchase.numSpaces = 10 →
  purchase.costPerSquareFoot = 60 →
  numCompanies = 3 →
  totalCost purchase numCompanies = 108000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ad_cost_theorem_l746_74671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_scenario_rate_l746_74621

/-- Calculates the compound interest rate given initial investment, final amount, and time period. -/
noncomputable def compound_interest_rate (initial_investment : ℝ) (final_amount : ℝ) (years : ℝ) : ℝ :=
  ((final_amount / initial_investment) ^ (1 / years)) - 1

/-- Theorem stating that for the given investment scenario, the compound interest rate is approximately 0.1 -/
theorem investment_scenario_rate : 
  let initial_investment := (1000 : ℝ)
  let final_amount := (1331 : ℝ)
  let years := (3 : ℝ)
  let rate := compound_interest_rate initial_investment final_amount years
  abs (rate - 0.1) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_scenario_rate_l746_74621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_P_trajectory_and_intersection_l746_74660

-- Define the circle A
noncomputable def circle_A (x y : ℝ) : Prop := (x + Real.sqrt 2)^2 + y^2 = 12

-- Define the fixed point B
noncomputable def point_B : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the trajectory of the center of circle P
def trajectory_P (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line that intersects the trajectory
def intersecting_line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the constant k
noncomputable def k : ℝ := Real.sqrt 39 / 3

-- Theorem statement
theorem circle_P_trajectory_and_intersection :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- The trajectory passes through two points (x₁, y₁) and (x₂, y₂)
    trajectory_P x₁ y₁ ∧ trajectory_P x₂ y₂ ∧
    -- These points lie on the intersecting line
    intersecting_line k x₁ y₁ ∧ intersecting_line k x₂ y₂ ∧
    -- The circle with diameter CD passes through the origin
    x₁ * x₂ + y₁ * y₂ = 0 ∧
    -- The constant k can be positive or negative
    (k = Real.sqrt 39 / 3 ∨ k = -Real.sqrt 39 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_P_trajectory_and_intersection_l746_74660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l746_74650

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem axis_of_symmetry (k : ℤ) :
  ∀ x : ℝ, f (Real.pi/12 + k*Real.pi/2 + x) = f (Real.pi/12 + k*Real.pi/2 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l746_74650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_B_radius_l746_74624

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define helper functions
def externally_tangent (C1 C2 : Circle) : Prop := sorry
def internally_tangent (C1 C2 : Circle) : Prop := sorry
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the problem setup
def problem_setup (A B C D : Circle) : Prop :=
  -- Circles A, B, and C are externally tangent to each other
  externally_tangent A B ∧ externally_tangent A C ∧ externally_tangent B C
  -- Circles A, B, and C are internally tangent to circle D
  ∧ internally_tangent A D ∧ internally_tangent B D ∧ internally_tangent C D
  -- Circles B and C are congruent
  ∧ B.radius = C.radius
  -- Circle A has radius 2
  ∧ A.radius = 2
  -- Circle A passes through the center of D
  ∧ distance A.center D.center = A.radius + D.radius

-- Define the theorem
theorem circle_B_radius (A B C D : Circle) :
  problem_setup A B C D → B.radius = 32 / 25 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_B_radius_l746_74624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l746_74668

-- Define z as a complex number
variable (z : ℂ)

-- Define the condition |2z+5| = |z+10|
def condition (z : ℂ) : Prop := Complex.abs (2 * z + 5) = Complex.abs (z + 10)

-- Theorem statement
theorem complex_number_properties (h : condition z) :
  (Complex.abs z = 5) ∧
  (∃ m : ℝ, m = 5 ∨ m = -5) ∧
  (∃ m : ℝ, (m = 5 ∨ m = -5) → (z / m + m / z).im = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l746_74668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_is_negative_one_l746_74614

/-- A point on the curve y = x + 2/x where x > 0 -/
structure PointOnCurve where
  x : ℝ
  y : ℝ
  x_pos : x > 0
  on_curve : y = x + 2/x

/-- The foot of the perpendicular from a point to the line y = x -/
noncomputable def footOnDiagonal (p : PointOnCurve) : ℝ × ℝ :=
  let t := (p.x + p.y) / 2
  (t, t)

/-- The foot of the perpendicular from a point to the y-axis -/
def footOnYAxis (p : PointOnCurve) : ℝ × ℝ :=
  (0, p.y)

/-- The dot product of vectors PA and PB -/
noncomputable def dotProductPAPB (p : PointOnCurve) : ℝ :=
  let a := footOnDiagonal p
  let b := footOnYAxis p
  (a.1 - p.x) * (b.1 - p.x) + (a.2 - p.y) * (b.2 - p.y)

theorem dot_product_is_negative_one (p : PointOnCurve) :
  dotProductPAPB p = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_is_negative_one_l746_74614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_hits_lower_bound_l746_74636

/-- The expected number of hit targets when n boys randomly choose from n targets -/
noncomputable def expected_hits (n : ℕ) : ℝ := n * (1 - (1 - 1 / n) ^ n)

/-- Theorem: The expected number of hit targets is always greater than or equal to n/2 -/
theorem expected_hits_lower_bound (n : ℕ) (hn : n > 0) :
  expected_hits n ≥ n / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_hits_lower_bound_l746_74636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l746_74632

-- Define the parabola C
def C (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (0, -1)

-- Define the line through two points
def line_through (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_properties (p : ℝ) (P Q : ℝ × ℝ) 
  (h1 : C p A.1 A.2)
  (h2 : line_through B P P.1 P.2 ∧ C p P.1 P.2)
  (h3 : line_through B Q Q.1 Q.2 ∧ C p Q.1 Q.2)
  (h4 : P ≠ Q) :
  (line_through A B 1 1) ∧  -- AB is tangent to C
  (distance O P * distance O Q > distance O A * distance O A) ∧  -- |OP| · |OQ| > |OA|^2
  (distance B P * distance B Q > distance B A * distance B A) :=  -- |BP| · |BQ| > |BA|^2
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l746_74632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_half_angle_inequality_l746_74637

theorem triangle_sine_half_angle_inequality (R r : ℝ) (h : R > 0 ∧ r > 0) :
  ∀ A : ℝ, 0 < A ∧ A < π →
  (R - Real.sqrt (R^2 - 2*R*r)) / (2*R) ≤ Real.sin (A/2) ∧
  Real.sin (A/2) ≤ (R + Real.sqrt (R^2 - 2*R*r)) / (2*R) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_half_angle_inequality_l746_74637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_correct_statements_l746_74615

theorem count_correct_statements : ℕ := by
  -- Define the five statements as boolean expressions
  let statement1 := (20 / 100 * 40 = 8)
  let statement2 := (2^3 = 8)
  let statement3 := (7 - 3 * 2 = 8)
  let statement4 := (3^2 - 1^2 = 8)
  let statement5 := (2 * (6 - 4)^2 = 8)

  -- Define a function to count true statements
  let count_true (s1 s2 s3 s4 s5 : Bool) : ℕ :=
    (if s1 then 1 else 0) +
    (if s2 then 1 else 0) +
    (if s3 then 1 else 0) +
    (if s4 then 1 else 0) +
    (if s5 then 1 else 0)

  -- Calculate the count of correct statements
  let result := count_true statement1 statement2 statement3 statement4 statement5

  -- Prove that the count of correct statements is 4
  have h : result = 4 := by sorry

  -- Return the result
  exact result


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_correct_statements_l746_74615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_simplification_l746_74669

theorem exponent_simplification (x y : ℝ) (h : x * y = 1) :
  (4 : ℝ) ^ ((x^3 + y^3)^2) / (4 : ℝ) ^ ((x^3 - y^3)^2) = 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_simplification_l746_74669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_other_side_is_correct_l746_74681

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  side1 : ℝ
  side2 : ℝ
  is_valid : side1 = 1 ∧ side2 > 0
  has_division : ∃ (x y : ℝ), 0 < x ∧ x < side2 ∧ 0 < y ∧ y < side2 ∧
    (min x (side2 - x) ≥ 1) ∧ (min y (side2 - y) ≥ 1) ∧ 
    (max x y ≥ 2)

/-- The minimum length of the other side of the special rectangle -/
noncomputable def minOtherSide (r : SpecialRectangle) : ℝ := 3 + 2 * Real.sqrt 2

/-- Theorem stating the minimum length of the other side -/
theorem min_other_side_is_correct (r : SpecialRectangle) :
  r.side2 ≥ minOtherSide r := by
  sorry

#check min_other_side_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_other_side_is_correct_l746_74681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_attention_time_is_41_4_l746_74685

/-- Represents the duration of the conference session in minutes -/
noncomputable def session_duration : ℝ := 90

/-- Represents the fraction of the audience that listened to the entire talk -/
noncomputable def full_attention_fraction : ℝ := 0.25

/-- Represents the fraction of the audience that did not pay attention at any point -/
noncomputable def no_attention_fraction : ℝ := 0.15

/-- Represents the fraction of the remainder that caught half of the talk -/
noncomputable def half_attention_fraction : ℝ := 0.4

/-- Calculates the average time the talk was heard by the audience members -/
noncomputable def average_attention_time : ℝ :=
  let remainder_fraction := 1 - full_attention_fraction - no_attention_fraction
  let quarter_attention_fraction := remainder_fraction * (1 - half_attention_fraction)
  (full_attention_fraction * session_duration +
   remainder_fraction * half_attention_fraction * (session_duration / 2) +
   quarter_attention_fraction * (session_duration / 4))

/-- Theorem stating that the average attention time is 41.4 minutes -/
theorem average_attention_time_is_41_4 :
  average_attention_time = 41.4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_attention_time_is_41_4_l746_74685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l746_74619

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + 2^x)

theorem sum_of_f_values :
  f (-1/3) + f (-1) + f 0 + f 1 + f (1/3) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l746_74619

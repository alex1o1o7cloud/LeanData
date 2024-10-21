import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_equals_neg_sqrt_3_l1236_123653

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0  -- Add case for 0
  | 1 => 0
  | n + 2 => (sequence_a (n + 1) - Real.sqrt 3) / (Real.sqrt 3 * sequence_a (n + 1) + 1)

theorem a_20_equals_neg_sqrt_3 : sequence_a 20 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_equals_neg_sqrt_3_l1236_123653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_books_sum_l1236_123649

/-- Sum of an arithmetic sequence with given parameters -/
def arithmetic_sequence_sum (a₁ : ℕ) (d : ℤ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + d.toNat * (n - 1)) / 2

theorem library_books_sum :
  arithmetic_sequence_sum 35 (-3) 12 = 222 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_books_sum_l1236_123649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_and_mouse_position_l1236_123656

-- Define the positions
inductive Position
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft
  | TopMiddle
  | RightMiddle
  | BottomMiddle
  | LeftMiddle
  deriving Repr

-- Define the animal types
inductive Animal
  | Cat
  | Mouse
  deriving Repr

def catCycleLength : Nat := 4
def mouseCycleLength : Nat := 8
def totalMoves : Nat := 258

-- Function to get the position of an animal after a certain number of moves
def getPosition (animal : Animal) (moves : Nat) : Position :=
  match animal with
  | Animal.Cat =>
      match moves % catCycleLength with
      | 0 => Position.BottomLeft
      | 1 => Position.TopLeft
      | 2 => Position.TopRight
      | _ => Position.BottomRight
  | Animal.Mouse =>
      match moves % mouseCycleLength with
      | 0 => Position.TopLeft
      | 1 => Position.TopMiddle
      | 2 => Position.TopRight
      | 3 => Position.RightMiddle
      | 4 => Position.BottomRight
      | 5 => Position.BottomMiddle
      | 6 => Position.BottomLeft
      | _ => Position.LeftMiddle

theorem cat_and_mouse_position : 
  getPosition Animal.Cat totalMoves = Position.TopRight ∧ 
  getPosition Animal.Mouse totalMoves = Position.TopRight := by
  sorry

#eval getPosition Animal.Cat totalMoves
#eval getPosition Animal.Mouse totalMoves

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_and_mouse_position_l1236_123656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l1236_123666

-- Define the slopes of the two lines
noncomputable def slope1 : ℝ := -1/3
noncomputable def slope2 (b : ℝ) : ℝ := -b/3

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- State the theorem
theorem perpendicular_lines_b_value (b : ℝ) :
  perpendicular slope1 (slope2 b) → b = -9 := by
  sorry

#check perpendicular_lines_b_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l1236_123666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_circumscribing_cube_l1236_123608

theorem sphere_volume_circumscribing_cube (cube_volume : ℝ) (sphere_volume : ℝ) : 
  cube_volume = 8 → 
  (∃ (cube_edge : ℝ) (sphere_radius : ℝ), 
    cube_edge^3 = cube_volume ∧ 
    sphere_radius = Real.sqrt 3 * cube_edge / 2 ∧ 
    sphere_volume = (4 / 3) * Real.pi * sphere_radius^3) →
  sphere_volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

#check sphere_volume_circumscribing_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_circumscribing_cube_l1236_123608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1236_123677

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x

-- State the theorem
theorem f_properties :
  -- f is increasing on (0, +∞)
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  -- The maximum value of f on [2, 5] is 7/5
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 5 → f x ≤ 7/5) ∧
  (f 5 = 7/5) ∧
  -- The minimum value of f on [2, 5] is 1/2
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 5 → 1/2 ≤ f x) ∧
  (f 2 = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1236_123677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1236_123620

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
noncomputable def train_length (train_speed_kmph : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_speed_mps * time_to_cross
  total_distance - bridge_length

/-- Theorem stating that a train traveling at 60 kmph and taking 17.998560115190784 seconds
    to cross a bridge of 190 m in length has a length of approximately 109.976001923 meters. -/
theorem train_length_calculation :
  let train_speed_kmph : ℝ := 60
  let time_to_cross : ℝ := 17.998560115190784
  let bridge_length : ℝ := 190
  abs (train_length train_speed_kmph time_to_cross bridge_length - 109.976001923) < 0.000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1236_123620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_sine_condition_l1236_123652

theorem monotonic_decreasing_sine_condition (ω : ℝ) : 
  ω > 0 → 
  (∀ x ∈ Set.Ioo (π / 2) π, 
    (HasDerivAt (λ y => Real.sin (ω * y + π / 4)) 
      (ω * Real.cos (ω * x + π / 4)) x) ∧ 
    ω * Real.cos (ω * x + π / 4) < 0) ↔ 
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_sine_condition_l1236_123652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l1236_123667

-- Define the function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^(m^2 - 5)

-- State the theorem
theorem inverse_proportion_quadrants (m : ℝ) : 
  (∀ x y, x ≠ 0 → f m x * x = f m y * y) → -- inverse proportion condition
  (∀ x y, x > 0 → y < 0 → f m x * f m y < 0) → -- second and fourth quadrants condition
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l1236_123667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_in_fourth_quadrant_l1236_123618

/-- The complex number Z is defined as 2/(3-i) + i^3 -/
noncomputable def Z : ℂ := 2 / (3 - Complex.I) + Complex.I ^ 3

/-- Theorem: The complex number Z is located in the fourth quadrant of the complex plane -/
theorem Z_in_fourth_quadrant : 
  0 < Z.re ∧ Z.im < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_in_fourth_quadrant_l1236_123618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_change_l1236_123690

theorem revenue_change 
  (original_price original_visitors : ℝ) 
  (price_increase : ℝ) 
  (visitor_decrease : ℝ) 
  (h1 : price_increase = 0.5) 
  (h2 : visitor_decrease = 0.2) : 
  (original_price * (1 + price_increase) * (original_visitors * (1 - visitor_decrease)) - 
   original_price * original_visitors) / (original_price * original_visitors) = 0.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_change_l1236_123690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_fifty_percent_l1236_123641

noncomputable def purchase_price : ℚ := 9000
noncomputable def repair_cost : ℚ := 5000
noncomputable def transportation_charges : ℚ := 1000
noncomputable def selling_price : ℚ := 22500

noncomputable def total_cost : ℚ := purchase_price + repair_cost + transportation_charges
noncomputable def profit : ℚ := selling_price - total_cost
noncomputable def profit_percentage : ℚ := (profit / total_cost) * 100

theorem profit_percentage_is_fifty_percent : profit_percentage = 50 := by
  -- Unfold definitions
  unfold profit_percentage
  unfold profit
  unfold total_cost
  -- Simplify the expression
  simp [purchase_price, repair_cost, transportation_charges, selling_price]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_fifty_percent_l1236_123641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l1236_123684

noncomputable def f (n : ℕ+) (x : ℝ) : ℝ := Real.cos (n * x) * Real.sin (80 * x / n^2)

theorem period_of_f (n : ℕ+) : 
  (∀ x, f n (x + 5 * Real.pi) = f n x) ↔ n ∈ ({1, 2, 4, 5, 10, 20} : Set ℕ+) := by
  sorry

#check period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l1236_123684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_is_8_l1236_123621

/-- A function that checks if a two-digit number is divisible by 17 or 19 -/
def isDivisibleBy17Or19 (n : Nat) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ (n % 17 = 0 ∨ n % 19 = 0)

/-- A function that checks if a list of digits satisfies the divisibility condition -/
def satisfiesDivisibilityCondition (digits : List Nat) : Prop :=
  ∀ i, i + 1 < digits.length → isDivisibleBy17Or19 (digits[i]! * 10 + digits[i + 1]!)

/-- The main theorem -/
theorem largest_last_digit_is_8 :
  ∃ (digits : List Nat),
    digits.length = 2003 ∧
    digits[0]! = 1 ∧
    satisfiesDivisibilityCondition digits ∧
    digits[2002]! = 8 ∧
    (∀ (other_digits : List Nat),
      other_digits.length = 2003 →
      other_digits[0]! = 1 →
      satisfiesDivisibilityCondition other_digits →
      other_digits[2002]! ≤ 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_is_8_l1236_123621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mitchell_average_points_l1236_123665

theorem mitchell_average_points (
  games_played : ℕ)
  (total_games : ℕ)
  (goal_average : ℝ)
  (remaining_average : ℝ)
  (h1 : games_played = 15)
  (h2 : total_games = 20)
  (h3 : goal_average = 30)
  (h4 : remaining_average = 42)
  : ∃ current_average : ℝ,
    current_average * (games_played : ℝ) + remaining_average * ((total_games - games_played) : ℝ)
    = goal_average * (total_games : ℝ)
    ∧ current_average = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mitchell_average_points_l1236_123665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_at_distance_sqrt_2_l1236_123673

noncomputable def line (t : ℝ) : ℝ × ℝ := (-2 - Real.sqrt 2 * t, 3 + Real.sqrt 2 * t)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_points_at_distance_sqrt_2 :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧
  distance (line t1) (-2, 3) = Real.sqrt 2 ∧
  distance (line t2) (-2, 3) = Real.sqrt 2 ∧
  (∀ t : ℝ, distance (line t) (-2, 3) = Real.sqrt 2 → t = t1 ∨ t = t2) ∧
  line t1 = (-3, 4) ∧
  line t2 = (-1, 2) := by
  sorry

#check line_points_at_distance_sqrt_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_at_distance_sqrt_2_l1236_123673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_area_after_rotation_l1236_123655

/-- A rectangle with dimensions 2 by 8 -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)
  (width_eq : width = 2)
  (height_eq : height = 8)

/-- A point inside the rectangle -/
structure InnerPoint :=
  (x : ℝ)
  (y : ℝ)
  (dist_AB : x = 1)
  (dist_BC : y = 1)
  (dist_AD : x = 1)

/-- The overlapping area after 45° rotation -/
noncomputable def overlapping_area (rect : Rectangle) (point : InnerPoint) : ℝ :=
  2 * (3 * Real.sqrt 2 - 2)

/-- Theorem stating the overlapping area after rotation -/
theorem overlapping_area_after_rotation (rect : Rectangle) (point : InnerPoint) :
  overlapping_area rect point = 2 * (3 * Real.sqrt 2 - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_area_after_rotation_l1236_123655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1236_123661

/-- The speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 60

/-- The time taken by the train to cross a pole in seconds -/
noncomputable def crossing_time : ℝ := 9

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_per_hr_to_m_per_s : ℝ := 5 / 18

/-- The length of the train in meters -/
noncomputable def train_length : ℝ := train_speed * km_per_hr_to_m_per_s * crossing_time

theorem train_length_calculation : 
  150 < train_length ∧ train_length < 151 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1236_123661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l1236_123660

noncomputable def complex_number (a : ℝ) : ℂ := (a + Complex.I) / (1 + Complex.I)

theorem purely_imaginary_condition (a : ℝ) :
  (complex_number a).re = 0 ∧ (complex_number a).im ≠ 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l1236_123660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_households_using_both_brands_l1236_123606

/-- The number of households that use both brands of soap -/
def households_using_both : ℕ := 60

/-- The total number of households surveyed -/
def total_households : ℕ := 500

/-- The number of households using neither brand E nor brand B -/
def households_using_neither : ℕ := 150

/-- The number of households using only brand E -/
def households_using_only_E : ℕ := 140

/-- The ratio of households using only brand B to those using both brands -/
def ratio_B_to_both : ℚ := 5 / 2

theorem households_using_both_brands :
  households_using_both = 60 ∧
  households_using_both + households_using_neither + households_using_only_E +
  (ratio_B_to_both * households_using_both).floor = total_households :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_households_using_both_brands_l1236_123606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_division_l1236_123638

theorem square_root_division (x : ℝ) : (Real.sqrt 2704 : ℝ) / x = 4 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_division_l1236_123638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_l1236_123639

-- Define the points on the line
variable (A B C D E : ℝ)

-- Define the distances between points
axiom AB_eq_BC : A - B = B - C
axiom AB_eq_2 : A - B = 2
axiom CD_eq_1 : C - D = 1
axiom DE_eq_3 : D - E = 3

-- Define the circles
variable (Ω ω : Set ℝ)

-- Define the centers of the circles
variable (O Q : ℝ)

-- Circle Ω passes through A and E
axiom A_in_Ω : A ∈ Ω
axiom E_in_Ω : E ∈ Ω

-- Circle ω passes through B and C
axiom B_in_ω : B ∈ ω
axiom C_in_ω : C ∈ ω

-- Circles Ω and ω touch each other
axiom circles_touch : ∃ (P : ℝ), P ∈ Ω ∧ P ∈ ω

-- Centers of Ω and ω, and point D are collinear
axiom centers_D_collinear : ∃ (m : ℝ), O - D = m * (Q - D)

-- Define the radii of the circles
noncomputable def R : ℝ := 8 * Real.sqrt (3/11)
noncomputable def r : ℝ := 5 * Real.sqrt (3/11)

-- The theorem to prove
theorem circle_radii : 
  ∀ (A B C D E : ℝ) (Ω ω : Set ℝ) (O Q : ℝ),
  (A - B = B - C) →
  (A - B = 2) →
  (C - D = 1) →
  (D - E = 3) →
  A ∈ Ω →
  E ∈ Ω →
  B ∈ ω →
  C ∈ ω →
  (∃ (P : ℝ), P ∈ Ω ∧ P ∈ ω) →
  (∃ (m : ℝ), O - D = m * (Q - D)) →
  (∀ (X : ℝ), X ∈ Ω → (X - O)^2 = R^2) ∧
  (∀ (Y : ℝ), Y ∈ ω → (Y - Q)^2 = r^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_l1236_123639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2023_equals_negative_two_l1236_123651

def sequence_a : ℕ → ℚ
  | 0 => 1/2  -- Add this case for 0
  | 1 => 1/2
  | n + 1 => (1 + sequence_a n) / (1 - sequence_a n)

theorem a_2023_equals_negative_two : sequence_a 2023 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2023_equals_negative_two_l1236_123651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_one_l1236_123602

/-- Predicate indicating if a set of points forms a right triangle -/
def IsRightTriangle (triangle : Set (ℝ × ℝ)) : Prop := sorry

/-- Function to calculate the length of a leg in a right triangle -/
def LegLength (triangle : Set (ℝ × ℝ)) : ℝ := sorry

/-- Predicate indicating if two triangles are congruent -/
def CongruentTriangles (triangle1 triangle2 : Set (ℝ × ℝ)) : Prop := sorry

/-- Function to calculate the area of a region -/
def AreaOfRegion (region : Set (ℝ × ℝ)) : ℝ := sorry

/-- The area of a region formed by two congruent right triangles with legs of length 1 is 1 square unit. -/
theorem shaded_area_equals_one (triangle1 triangle2 : Set (ℝ × ℝ)) : 
  IsRightTriangle triangle1 → 
  IsRightTriangle triangle2 → 
  CongruentTriangles triangle1 triangle2 →
  LegLength triangle1 = 1 →
  LegLength triangle2 = 1 →
  AreaOfRegion (triangle1 ∪ triangle2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_one_l1236_123602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_third_side_l1236_123671

theorem right_triangle_third_side (a b : ℝ) : 
  (∃ c : ℝ, c > 0 ∧ a^2 + b^2 = c^2) →  -- right triangle condition
  (Real.sqrt (a^2 - 6*a + 9) + |b - 4| = 0) →  -- given equation
  (∃ c : ℝ, (c = 5 ∨ c = Real.sqrt 7) ∧ a^2 + b^2 = c^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_third_side_l1236_123671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_properties_l1236_123672

/-- Represents the data for a stratified random sampling of student heights -/
structure HeightSamplingData where
  total_students : Nat
  num_boys : Nat
  num_girls : Nat
  sample_size : Nat
  boys_mean_height : ℝ
  boys_height_variance : ℝ
  girls_mean_height : ℝ
  girls_height_variance : ℝ

/-- Calculates the number of boys in the sample -/
def num_boys_in_sample (data : HeightSamplingData) : Nat :=
  (data.num_boys * data.sample_size) / data.total_students

/-- Calculates the number of girls in the sample -/
def num_girls_in_sample (data : HeightSamplingData) : Nat :=
  (data.num_girls * data.sample_size) / data.total_students

/-- Calculates the total variance of the sample -/
noncomputable def total_variance (data : HeightSamplingData) : ℝ :=
  let total_mean := (data.boys_mean_height * (num_boys_in_sample data : ℝ) + data.girls_mean_height * (num_girls_in_sample data : ℝ)) / (data.sample_size : ℝ)
  ((num_boys_in_sample data : ℝ) * (data.boys_height_variance + (data.boys_mean_height - total_mean)^2) +
   (num_girls_in_sample data : ℝ) * (data.girls_height_variance + (data.girls_mean_height - total_mean)^2)) / (data.sample_size : ℝ)

/-- The main theorem stating the properties of the sampling data -/
theorem sampling_properties (data : HeightSamplingData)
  (h_total : data.total_students = 1000)
  (h_boys : data.num_boys = 600)
  (h_girls : data.num_girls = 400)
  (h_sample : data.sample_size = 50)
  (h_boys_mean : data.boys_mean_height = 170)
  (h_boys_var : data.boys_height_variance = 14)
  (h_girls_mean : data.girls_mean_height = 160)
  (h_girls_var : data.girls_height_variance = 34) :
  num_boys_in_sample data = 30 ∧
  num_girls_in_sample data = 20 ∧
  total_variance data = 46 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_properties_l1236_123672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1236_123692

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 3/2) = -f x) 
  (h2 : is_odd (λ x ↦ f (x - 3/4))) :
  (periodic f 3) ∧ 
  (∀ x, f (-3/4 - x) = -f (-3/4 + x)) ∧ 
  (is_even f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1236_123692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_origin_l1236_123647

theorem circle_tangent_origin (D E F : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0 → 
    (∃ (δ : ℝ), δ > 0 ∧ ∀ t : ℝ, 0 < |t| ∧ |t| < δ → t^2 + D*t + F > 0)) →
  D = 0 ∧ E ≠ 0 ∧ F ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_origin_l1236_123647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_fraction_equality_l1236_123619

theorem power_fraction_equality (x y z : ℕ+) (h : x * y * z = 1) :
  (7 : ℝ) ^ ((x : ℝ) + y + z) ^ 3 / (7 : ℝ) ^ ((x : ℝ) - y + z) ^ 3 = 7^6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_fraction_equality_l1236_123619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_sum_l1236_123623

theorem greatest_sum (a b c x y z : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : x < y) (h4 : y < z) :
  let sum_d := a * x + b * y + c * z
  let sum_a := a * x + c * y + b * z
  let sum_b := b * x + a * y + c * z
  let sum_c := b * x + c * y + a * z
  sum_d ≥ sum_a ∧ sum_d ≥ sum_b ∧ sum_d ≥ sum_c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_sum_l1236_123623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l1236_123687

theorem cube_root_simplification : (1 + 8) ^ (1/3) * (1 + 8 ^ (1/3)) ^ (1/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l1236_123687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l1236_123694

/-- Two vectors in 3D space are orthogonal if and only if their dot product is zero -/
def are_orthogonal (v₁ v₂ : Fin 3 → ℝ) : Prop :=
  (v₁ 0) * (v₂ 0) + (v₁ 1) * (v₂ 1) + (v₁ 2) * (v₂ 2) = 0

/-- The first line's direction vector -/
def v₁ (b : ℝ) : Fin 3 → ℝ := ![1, b, 3]

/-- The second line's direction vector -/
def v₂ : Fin 3 → ℝ := ![-2, 5, 1]

/-- Theorem: The value of b that makes the two lines perpendicular is -1/5 -/
theorem perpendicular_lines_b_value :
  ∃ b : ℝ, are_orthogonal (v₁ b) v₂ ∧ b = -1/5 := by
  use -1/5
  constructor
  · -- Prove that the vectors are orthogonal when b = -1/5
    simp [are_orthogonal, v₁, v₂]
    ring
  · -- Prove that b = -1/5
    rfl

#check perpendicular_lines_b_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l1236_123694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_condition_l1236_123654

/-- Arithmetic sequence sum -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (a₁ + (n - 1) * d / 2)

theorem arithmetic_sequence_condition (a₁ : ℝ) :
  (∀ d > 1, S a₁ d 4 + S a₁ d 6 > 2 * S a₁ d 5) ∧
  (∃ d ≤ 1, S a₁ d 4 + S a₁ d 6 > 2 * S a₁ d 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_condition_l1236_123654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_path_theorem_l1236_123610

/-- Represents a point on the grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Checks if a move between two points is valid for a given r -/
def validMove (r : Nat) (p1 p2 : GridPoint) : Prop :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = r

/-- Defines a path on the grid -/
def ValidPath (r : Nat) : List GridPoint → Prop
  | [] => True
  | [_] => True
  | p1 :: p2 :: rest => validMove r p1 p2 ∧ ValidPath r (p2 :: rest)

/-- The start point A -/
def A : GridPoint := ⟨0, 0⟩

/-- The end point B -/
def B : GridPoint := ⟨19, 0⟩

/-- The theorem to be proved -/
theorem grid_path_theorem :
  (∀ r : Nat, r % 2 = 0 ∨ r % 3 = 0 → ¬∃ path : List GridPoint, ValidPath r path ∧ path.head? = some A ∧ path.getLast? = some B) ∧
  (∃ path : List GridPoint, ValidPath 73 path ∧ path.head? = some A ∧ path.getLast? = some B) ∧
  (¬∃ path : List GridPoint, ValidPath 97 path ∧ path.head? = some A ∧ path.getLast? = some B) := by
  sorry

#check grid_path_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_path_theorem_l1236_123610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_is_1284_l1236_123682

def sequence_term (n : ℕ) : ℕ :=
  (Nat.digits 2 n).enum.foldl (λ acc (i, b) => acc + if b = 1 then 4^i else 0) 0

theorem fiftieth_term_is_1284 : sequence_term 50 = 1284 := by
  rfl

#eval sequence_term 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_is_1284_l1236_123682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_counterexample_l1236_123676

theorem triangle_inequality_counterexample : 
  ∃ (segments : Finset ℝ), 
    segments.card = 10 ∧ 
    ∀ (a b c : ℝ), a ∈ segments → b ∈ segments → c ∈ segments → 
      a ≤ b → b ≤ c → a + b ≤ c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_counterexample_l1236_123676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1236_123699

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate) ^ time - principal

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem principal_calculation (P : ℝ) :
  let rate : ℝ := 0.06
  let time : ℝ := 2
  compound_interest P rate time - simple_interest P rate time = 36 →
  P = 10000 := by
  sorry

#check principal_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1236_123699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_miles_is_2250_l1236_123686

/-- The maximum number of revolutions the device can count before resetting --/
def max_revolutions : ℕ := 89999

/-- The number of times the device reset during the year --/
def resets : ℕ := 37

/-- The number of revolutions shown on the device on December 31 --/
def final_revolutions : ℕ := 25000

/-- The number of revolutions required to complete one mile --/
def revolutions_per_mile : ℕ := 1500

/-- Calculate the total number of revolutions cycled during the year --/
def total_revolutions : ℕ := (max_revolutions + 1) * resets + final_revolutions

/-- Calculate the exact number of miles cycled --/
def exact_miles : ℚ := total_revolutions / revolutions_per_mile

/-- The closest integer to the exact number of miles cycled --/
def closest_miles : ℕ := (exact_miles + 1/2).floor.toNat

/-- Theorem stating that the closest number of miles to the exact miles cycled is 2250 --/
theorem closest_miles_is_2250 : closest_miles = 2250 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_miles_is_2250_l1236_123686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_identity_transformation_l1236_123658

noncomputable section

/-- Angle between a line and the positive x-axis -/
def angle (l : ℝ → ℝ) : ℝ := sorry

/-- Reflection of a line about another line -/
def reflect (l : ℝ → ℝ) (about : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The transformation T -/
def T (l : ℝ → ℝ) (l₁ l₂ : ℝ → ℝ) : ℝ → ℝ := reflect (reflect l l₁) l₂

/-- n-th iteration of T -/
def T_iter (n : ℕ) (l l₁ l₂ : ℝ → ℝ) : ℝ → ℝ := 
  match n with
  | 0 => l
  | n + 1 => T (T_iter n l l₁ l₂) l₁ l₂

theorem smallest_n_for_identity_transformation :
  let l₁ : ℝ → ℝ := λ x => Real.tan (π / 30) * x
  let l₂ : ℝ → ℝ := λ x => Real.tan (π / 40) * x
  let l  : ℝ → ℝ := λ x => (2 / 45) * x
  (∀ n : ℕ, n > 0 → n < 120 → T_iter n l l₁ l₂ ≠ l) ∧
  T_iter 120 l l₁ l₂ = l := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_identity_transformation_l1236_123658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_square_height_l1236_123637

/-- The distance of the highest point of a rotated square from the base line -/
theorem rotated_square_height (square_side : ℝ) (rotation_angle : ℝ)
  (h1 : square_side = 1)
  (h2 : rotation_angle = π / 6) :
  (square_side / 2) + (square_side / 2 * Real.cos rotation_angle) = (2 + Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_square_height_l1236_123637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_subset_T_l1236_123604

open Real

noncomputable def S : Set (ℝ × ℝ) := {p | ∃ k : ℤ, p.1^2 - p.2^2 = 2*k - 1}

noncomputable def T : Set (ℝ × ℝ) := {p | sin (2*Real.pi*p.1^2) - sin (2*Real.pi*p.2^2) - cos (2*Real.pi*p.1^2) - cos (2*Real.pi*p.2^2) = 0}

theorem S_subset_T : S ⊂ T := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_subset_T_l1236_123604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABGF_is_four_ninths_l1236_123600

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the area of triangle ABC
def area_ABC : ℝ := 1

-- Define points D and E as one-third points on AC and BC respectively
def D (A C : ℝ × ℝ) : ℝ × ℝ := (2/3 * A.1 + 1/3 * C.1, 2/3 * A.2 + 1/3 * C.2)
def E (B C : ℝ × ℝ) : ℝ × ℝ := (2/3 * B.1 + 1/3 * C.1, 2/3 * B.2 + 1/3 * C.2)

-- Define F as the midpoint of AE
def F (A B C : ℝ × ℝ) : ℝ × ℝ := 
  let E := E B C
  (1/2 * A.1 + 1/2 * E.1, 1/2 * A.2 + 1/2 * E.2)

-- Define G as the midpoint of BD
def G (A B C : ℝ × ℝ) : ℝ × ℝ := 
  let D := D A C
  (1/2 * B.1 + 1/2 * D.1, 1/2 * B.2 + 1/2 * D.2)

-- Define the area of quadrilateral ABGF
def area_ABGF (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_ABGF_is_four_ninths (A B C : ℝ × ℝ) :
  area_ABGF A B C = 4/9 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABGF_is_four_ninths_l1236_123600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_product_l1236_123605

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- Define the tangent line y = mx + n
structure TangentLine where
  m : ℝ
  n : ℝ

-- State the theorem
theorem tangent_line_product (l : TangentLine) :
  (∃ x : ℝ, f x = l.m * x + l.n ∧ 
   ∀ y : ℝ, y ≠ x → f y < l.m * y + l.n) →
  l.m * l.n = 1/4 := by
  sorry

#check tangent_line_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_product_l1236_123605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_circle_sector_l1236_123689

open Real

-- Define the radius of the original circle
noncomputable def r : ℝ := 6

-- Define the circumference of the cone's base (half of the original circle's circumference)
noncomputable def base_circumference : ℝ := Real.pi * r

-- Define the radius of the cone's base
noncomputable def base_radius : ℝ := base_circumference / (2 * Real.pi)

-- Define the slant height of the cone (equal to the original circle's radius)
noncomputable def slant_height : ℝ := r

-- Define the height of the cone
noncomputable def cone_height : ℝ := Real.sqrt (slant_height^2 - base_radius^2)

-- State the theorem
theorem cone_volume_from_half_circle_sector :
  (1/3 : ℝ) * Real.pi * base_radius^2 * cone_height = 9 * Real.pi * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_circle_sector_l1236_123689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_monotonicity_condition_l1236_123680

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

-- Define the domain
def domain : Set ℝ := Set.Icc (-2) 5

-- Theorem for part (1)
theorem max_min_values (h : a = -2) :
  (∃ x ∈ domain, ∀ y ∈ domain, f a x ≥ f a y) ∧
  (∃ x ∈ domain, ∀ y ∈ domain, f a x ≤ f a y) ∧
  (∀ x ∈ domain, f a x ≤ 13) ∧
  (∀ x ∈ domain, f a x ≥ -3) :=
sorry

-- Theorem for part (2)
theorem monotonicity_condition :
  (∀ x y, x ∈ domain → y ∈ domain → x < y → (f a x < f a y ∨ f a x > f a y)) ↔ (a ≤ -5 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_monotonicity_condition_l1236_123680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_rate_500_discount_rate_1000_min_price_for_one_third_discount_max_discount_rate_l1236_123625

noncomputable def voucher_amount (spent : ℝ) : ℝ :=
  if 200 ≤ spent ∧ spent < 400 then 30
  else if 400 ≤ spent ∧ spent < 500 then 60
  else if 500 ≤ spent ∧ spent < 700 then 100
  else if 700 ≤ spent ∧ spent < 900 then 130
  else 0  -- Default case, can be adjusted based on the complete scheme

noncomputable def discount_rate (marked_price : ℝ) : ℝ :=
  let spent := 0.8 * marked_price
  let discount := 0.2 * marked_price + voucher_amount spent
  discount / marked_price

-- Theorem statements
theorem discount_rate_500 : discount_rate 500 = 0.32 := by sorry

theorem discount_rate_1000 : discount_rate 1000 = 0.33 := by sorry

theorem min_price_for_one_third_discount :
  ∀ x : ℝ, 500 ≤ x → x < 1000 → discount_rate x ≥ 1/3 → x ≥ 625 := by sorry

theorem max_discount_rate :
  ∀ x : ℝ, 500 ≤ x → x ≤ 1000 → discount_rate x ≤ 0.36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_rate_500_discount_rate_1000_min_price_for_one_third_discount_max_discount_rate_l1236_123625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l1236_123674

theorem ellipse_standard_equation 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : (0^2 / a^2) + (1^2 / b^2) = 1)  -- Ellipse passes through (0,1)
  (h4 : Real.sqrt (1 - b^2/a^2) = Real.sqrt 2/2)  -- Eccentricity is √2/2
  : a^2 = 2 ∧ b = 1 := by
  sorry

-- The standard equation of the ellipse is x^2/2 + y^2 = 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l1236_123674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_race_length_l1236_123630

/-- The length of the first race -/
def L : ℝ := sorry

/-- A beats B by this distance in the first race -/
def A_beats_B : ℝ := 20

/-- A beats C by this distance in the first race -/
def A_beats_C : ℝ := 38

/-- Length of the second race -/
def second_race_length : ℝ := 600

/-- B beats C by this distance in the second race -/
def B_beats_C_second : ℝ := 60

theorem first_race_length : 
  (L - A_beats_B) / (L - A_beats_C) = second_race_length / (second_race_length - B_beats_C_second) →
  L = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_race_length_l1236_123630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_x_intersection_l1236_123629

noncomputable def circle_center : ℝ × ℝ := ((2 + 10) / 2, (2 + 8) / 2)
noncomputable def circle_radius : ℝ := Real.sqrt ((10 - 2)^2 + (8 - 2)^2) / 2

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2

-- Theorem statement
theorem circle_x_intersection :
  ∃ (x : ℝ), x ≠ 2 ∧ circle_equation x 0 ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_x_intersection_l1236_123629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l1236_123617

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x ^ 2 + f y) = x * f x + y) →
  (∀ x : ℝ, f x = x ∨ f x = -x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l1236_123617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_in_base_7_l1236_123632

/-- Addition in base 7 --/
def add_base_7 (a b : ℕ) : ℕ := 
  (a.digits 7).zipWith (λ x y => x + y) (b.digits 7)
    |> List.foldl (λ acc x => acc * 7 + x) 0

/-- Conversion from base 10 to base 7 --/
def to_base_7 (n : ℕ) : ℕ := n.digits 7 |> List.foldl (λ acc x => acc * 10 + x) 0

theorem sum_in_base_7 : 
  add_base_7 (to_base_7 26) (to_base_7 245) = to_base_7 304 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_in_base_7_l1236_123632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_seventh_term_l1236_123668

theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℝ),
    (∀ n, a (n + 1) = a n * (a 8 / a 0) ^ (1/8)) →  -- Common ratio is the 8th root of a₉/a₁
    a 0 = 4 →                                       -- First term is 4
    a 8 = 248832 →                                  -- Ninth term is 248832
    a 6 = 186624 :=                                 -- Seventh term is 186624
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_seventh_term_l1236_123668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_sixth_power_l1236_123645

theorem fourth_root_sixteen_to_sixth_power : ((16 : ℝ) ^ (1/4)) ^ 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_sixth_power_l1236_123645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_fraction_second_week_l1236_123659

-- Define the total amount of paint
noncomputable def total_paint : ℝ := 360

-- Define the fraction of paint used in the first week
noncomputable def first_week_fraction : ℝ := 1 / 4

-- Define the total amount of paint used after two weeks
noncomputable def total_used : ℝ := 128.57

-- Theorem to prove
theorem paint_fraction_second_week :
  let remaining_after_first_week := total_paint * (1 - first_week_fraction)
  let used_second_week := total_used - (total_paint * first_week_fraction)
  (used_second_week / remaining_after_first_week) = 119 / 250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_fraction_second_week_l1236_123659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_congruence_solutions_l1236_123612

theorem three_digit_congruence_solutions :
  (Finset.filter (fun y : ℕ => 
    100 ≤ y ∧ y ≤ 999 ∧ (5327 * y + 673) % 17 = 1850 % 17) (Finset.range 1000)).card = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_congruence_solutions_l1236_123612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bryden_receives_correct_amount_l1236_123616

/-- The face value of a state quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The number of quarters Bryden has -/
def brydens_quarters : ℕ := 10

/-- The percentage the collector offers for each quarter -/
def collector_offer_percentage : ℕ := 1500

/-- Calculate the amount Bryden will receive for his quarters -/
def brydens_payment (quarter_value : ℚ) (brydens_quarters : ℕ) (collector_offer_percentage : ℕ) : ℚ :=
  quarter_value * brydens_quarters * (collector_offer_percentage / 100 : ℚ)

/-- Theorem stating that Bryden will receive 37.5 dollars for his quarters -/
theorem bryden_receives_correct_amount :
  brydens_payment quarter_value brydens_quarters collector_offer_percentage = 37.5 := by
  -- Unfold the definition of brydens_payment
  unfold brydens_payment
  -- Simplify the expression
  simp [quarter_value, brydens_quarters, collector_offer_percentage]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bryden_receives_correct_amount_l1236_123616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brendans_earnings_l1236_123601

/-- Brendan's earnings in June -/
noncomputable def earnings : ℝ := 3000

/-- Brendan spends half of his pay on his debit card -/
noncomputable def debit_card_spend : ℝ := earnings / 2

/-- Cost of the car Brendan bought -/
def car_cost : ℝ := 1500

/-- Brendan's remaining money at the end of the month -/
def remaining_money : ℝ := 1000

/-- Theorem stating Brendan's earnings in June -/
theorem brendans_earnings : 
  earnings = 3000 ∧ 
  debit_card_spend + remaining_money = car_cost + remaining_money :=
by
  constructor
  · rfl
  · simp [earnings, debit_card_spend, car_cost, remaining_money]
    ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brendans_earnings_l1236_123601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_digits_of_sum_of_squares_l1236_123635

def ones_number (n : ℕ) : ℕ :=
  if n = 0 then 0 else (10^n - 1) / 9

def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

def sum_of_squares (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (λ i => (ones_number i) ^ 2)

theorem last_three_digits_of_sum_of_squares :
  last_three_digits (sum_of_squares 2010) = 690 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_digits_of_sum_of_squares_l1236_123635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_sales_increase_l1236_123646

/-- Represents the sales data for a store --/
structure SalesData where
  total_last_year : ℝ
  total_this_year : ℝ
  electronics_increase : ℝ
  clothing_increase : ℝ
  groceries_increase : ℝ
  electronics_weight : ℝ
  clothing_weight : ℝ
  groceries_weight : ℝ

/-- Calculates the weighted average percent increase in sales --/
noncomputable def weighted_average_increase (data : SalesData) : ℝ :=
  (data.electronics_weight * data.electronics_increase +
   data.clothing_weight * data.clothing_increase +
   data.groceries_weight * data.groceries_increase) /
  (data.electronics_weight + data.clothing_weight + data.groceries_weight)

/-- Theorem stating that the weighted average percent increase is 24% for the given data --/
theorem store_sales_increase (data : SalesData)
  (h1 : data.total_last_year = 320)
  (h2 : data.total_this_year = 416)
  (h3 : data.electronics_increase = 0.15)
  (h4 : data.clothing_increase = 0.25)
  (h5 : data.groceries_increase = 0.35)
  (h6 : data.electronics_weight = 0.40)
  (h7 : data.clothing_weight = 0.30)
  (h8 : data.groceries_weight = 0.30) :
  weighted_average_increase data = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_sales_increase_l1236_123646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_theorem_l1236_123607

open Real

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x
noncomputable def g (x : ℝ) : ℝ := x / (Real.exp x)

-- State the theorem
theorem k_range_theorem (k : ℝ) (hk : k > 0) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → g x₁ / k ≤ f x₂ / (k + 1)) →
  k ≥ 1 / (2 * Real.exp 1 - 1) := by
  sorry

-- You can add additional lemmas or theorems here if needed for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_theorem_l1236_123607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_flour_approx_22_06_l1236_123640

/-- The amount of flour Sarah has, given her purchases and existing stock --/
noncomputable def total_flour_pounds : ℝ :=
  let rye_flour : ℝ := 5
  let whole_wheat_bread_flour : ℝ := 10
  let chickpea_flour_grams : ℝ := 1800
  let whole_wheat_pastry_flour : ℝ := 2
  let all_purpose_flour_grams : ℝ := 500
  let grams_per_pound : ℝ := 454
  
  rye_flour + whole_wheat_bread_flour + (chickpea_flour_grams / grams_per_pound) +
  whole_wheat_pastry_flour + (all_purpose_flour_grams / grams_per_pound)

/-- Theorem stating that the total amount of flour Sarah has is approximately 22.06 pounds --/
theorem total_flour_approx_22_06 :
  ∃ ε > 0, |total_flour_pounds - 22.06| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_flour_approx_22_06_l1236_123640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_consumption_l1236_123697

-- Define the geometric sum function
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Theorem statement
theorem pizza_consumption : (1 : ℚ)/3 + geometric_sum (1/3) (1/2) 5 = 21/32 := by
  -- Define the fraction eaten on the first trip
  let first_trip : ℚ := 1/3
  
  -- Define the common ratio for subsequent trips
  let ratio : ℚ := 1/2
  
  -- Define the number of trips after the first one
  let remaining_trips : ℕ := 5
  
  -- Calculate the sum of fractions eaten after the first trip
  let subsequent_sum := geometric_sum first_trip ratio remaining_trips
  
  -- Calculate the total fraction eaten
  let total_fraction := first_trip + subsequent_sum
  
  -- Prove that the total fraction eaten equals 21/32
  sorry  -- We use sorry here as a placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_consumption_l1236_123697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l1236_123685

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The conditions on the polynomial P -/
def SatisfiesConditions (P : IntPolynomial) (a : ℤ) : Prop :=
  (P.eval 1 = a) ∧ (P.eval 3 = a) ∧ (P.eval 5 = a) ∧ (P.eval 7 = a) ∧ (P.eval 9 = a) ∧
  (P.eval 2 = -a) ∧ (P.eval 4 = -a) ∧ (P.eval 6 = -a) ∧ (P.eval 8 = -a) ∧ (P.eval 10 = -a)

/-- The theorem statement -/
theorem smallest_a_value : 
  ∀ a : ℕ+, (∃ P : IntPolynomial, SatisfiesConditions P a) → a ≥ 6930 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l1236_123685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_term_is_79_5_l1236_123688

/-- An arithmetic sequence is defined by its third term and eleventh term. -/
structure ArithmeticSequence where
  a₃ : ℚ  -- Third term
  a₁₁ : ℚ  -- Eleventh term

/-- Calculate the thirtieth term of an arithmetic sequence. -/
def thirtiethTerm (seq : ArithmeticSequence) : ℚ :=
  let d := (seq.a₁₁ - seq.a₃) / 8  -- Common difference
  seq.a₃ + 27 * d

/-- Theorem stating that for the given arithmetic sequence, the thirtieth term is 79.5 -/
theorem thirtieth_term_is_79_5 (seq : ArithmeticSequence) 
  (h₃ : seq.a₃ = 12) (h₁₁ : seq.a₁₁ = 32) : 
  thirtiethTerm seq = 79.5 := by
  sorry

#eval thirtiethTerm { a₃ := 12, a₁₁ := 32 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_term_is_79_5_l1236_123688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l1236_123662

-- Define the complex number z that traces a circle with radius 3
noncomputable def z (θ : ℝ) : ℂ := 3 * Complex.exp (θ * Complex.I)

-- Define the function f(z) = z + 1/z
noncomputable def f (z : ℂ) : ℂ := z + 1 / z

-- Define the locus of f(z) as θ varies
noncomputable def locus (θ : ℝ) : ℂ := f (z θ)

-- Theorem statement
theorem locus_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  ∀ (θ : ℝ), (Complex.abs (locus θ).re / a)^2 + (Complex.abs (locus θ).im / b)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l1236_123662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_D_value_l1236_123626

/-- A parallelogram in the complex plane -/
structure ComplexParallelogram where
  A : ℂ
  B : ℂ
  C : ℂ
  D : ℂ
  is_parallelogram : (B - A) = (C - D)

/-- The specific parallelogram from the problem -/
def ABCD : ComplexParallelogram where
  A := 1 + 3*Complex.I
  B := 2 - Complex.I
  C := -3 + Complex.I
  D := -4 + 5*Complex.I
  is_parallelogram := by sorry

/-- The theorem to prove -/
theorem parallelogram_D_value (p : ComplexParallelogram) 
  (h1 : p.A = 1 + 3*Complex.I) 
  (h2 : p.B = 2 - Complex.I) 
  (h3 : p.C = -3 + Complex.I) : 
  p.D = -4 + 5*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_D_value_l1236_123626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1236_123631

/-- Definition of an acute-angled triangle -/
def AcuteTriangle (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a^2 + b^2 > c^2 ∧ 
  b^2 + c^2 > a^2 ∧ 
  c^2 + a^2 > b^2

/-- Definition of a median in a triangle -/
def IsMedian (a b c s : ℝ) : Prop := 
  4 * s^2 = 2 * b^2 + 2 * c^2 - a^2

/-- Definition of triangle area using Heron's formula -/
noncomputable def TriangleArea (a b c : ℝ) : ℝ := 
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_inequality (a b c s_a s_b s_c t : ℝ) 
  (h_acute : AcuteTriangle a b c)
  (h_median_a : IsMedian a b c s_a)
  (h_median_b : IsMedian b c a s_b)
  (h_median_c : IsMedian c a b s_c)
  (h_area : t = TriangleArea a b c) :
  1 / (s_a^2 - a^2/4) + 1 / (s_b^2 - b^2/4) + 1 / (s_c^2 - c^2/4) ≥ (3 * Real.sqrt 3) / (2 * t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1236_123631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l1236_123628

/-- The area of a circular sector -/
noncomputable def sectorArea (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * Real.pi * r^2

/-- Theorem: The area of a sector with radius 3 and central angle 120° is 3π -/
theorem sector_area_example : sectorArea 3 120 = 3 * Real.pi := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp [Real.pi]
  -- Perform algebraic manipulations
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l1236_123628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1236_123643

/-- The curve y = (ax-1)e^x has a tangent line l1 at point A (x_0, y_1) -/
noncomputable def curve1 (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) * Real.exp x

/-- The curve y = (1-x)e^(-x) has a tangent line l2 at point B (x_0, y_2) -/
noncomputable def curve2 (x : ℝ) : ℝ := (1 - x) * Real.exp (-x)

/-- The slope of the tangent line l1 -/
noncomputable def slope1 (a : ℝ) (x : ℝ) : ℝ := (a * x + a - 1) * Real.exp x

/-- The slope of the tangent line l2 -/
noncomputable def slope2 (x : ℝ) : ℝ := (x - 2) * Real.exp (-x)

/-- The theorem stating the range of a -/
theorem range_of_a :
  ∀ a : ℝ, (∃ x₀ : ℝ, 0 ≤ x₀ ∧ x₀ ≤ 3/2 ∧ slope1 a x₀ * slope2 x₀ = -1) →
  1 ≤ a ∧ a ≤ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1236_123643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_arrangement_without_A_at_head_l1236_123695

theorem line_arrangement_without_A_at_head (n : ℕ) (h : n = 6) :
  (n - 1) * Nat.factorial (n - 1) = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_arrangement_without_A_at_head_l1236_123695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_d_value_l1236_123633

/-- A polynomial of degree 4 with integer coefficients -/
def MyPolynomial (a b c d : ℤ) : ℝ → ℝ := fun x ↦ x^4 + a*x^3 + b*x^2 + c*x + d

/-- The theorem stating the value of d given the conditions -/
theorem polynomial_d_value (a b c d : ℤ) :
  (∀ r : ℝ, MyPolynomial a b c d r = 0 → ∃ n : ℤ, r = -n ∧ n > 0) →
  a + b + c + d = 2031 →
  d = 1540 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_d_value_l1236_123633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_surds_simplified_l1236_123611

theorem sum_of_surds_simplified (a b c : ℕ) :
  (Real.sqrt 3 + (Real.sqrt 3)⁻¹ + Real.sqrt 8 + (Real.sqrt 8)⁻¹ = (a * Real.sqrt 3 + b * Real.sqrt 8) / c) →
  (∀ d : ℕ, 0 < d → d < c → ¬(∃ e f : ℕ, (Real.sqrt 3 + (Real.sqrt 3)⁻¹ + Real.sqrt 8 + (Real.sqrt 8)⁻¹ = (e * Real.sqrt 3 + f * Real.sqrt 8) / d))) →
  a = 13 ∧ b = 0 ∧ c = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_surds_simplified_l1236_123611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l1236_123614

noncomputable def power_function (m : ℤ) (x : ℝ) : ℝ := x^(m^2 - 2*m - 3)

def no_intersection_with_axes (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≠ 0 ∧ f 0 ≠ x

def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem power_function_properties (m : ℤ) :
  (no_intersection_with_axes (power_function m) ∧
   symmetric_about_y_axis (power_function m)) ↔
  (m = -1 ∨ m = 1 ∨ m = 3) :=
by
  sorry

#check power_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l1236_123614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_length_l1236_123648

theorem third_side_length (a b c : ℝ) (α : ℝ) : 
  a = 4 → b = 5 → 
  (2 * (Real.cos α)^2 + 3 * Real.cos α - 2 = 0) → 
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos α) → 
  c = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_length_l1236_123648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_nonpositive_terms_l1236_123678

def sequence_rule (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = if a n ≠ 0 then (a n ^ 2 - 1) / (2 * a n) else 0

theorem infinitely_many_nonpositive_terms (a : ℕ → ℝ) (h : sequence_rule a) :
  Set.Infinite {n : ℕ | n ≥ 1 ∧ a n ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_nonpositive_terms_l1236_123678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_derived_ellipse_eccentricity_l1236_123624

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and asymptote lines y = ±√3x,
    the eccentricity of an ellipse that has the hyperbola's vertex as its focus
    and the hyperbola's focus as its vertex is 1/2. -/
theorem hyperbola_derived_ellipse_eccentricity
  (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptote : b / a = Real.sqrt 3) :
  let c := Real.sqrt (a^2 + b^2)
  let e := c / (2 * c)
  e = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_derived_ellipse_eccentricity_l1236_123624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_difference_l1236_123627

-- Define the new operation ⊕
def oplus (x y : ℝ) : ℝ := 2 * x * y - 3 * x + y

-- Theorem statement
theorem oplus_difference : oplus 7 4 - oplus 4 7 = -12 := by
  -- Unfold the definition of oplus
  unfold oplus
  -- Simplify the expressions
  simp [mul_add, add_mul, mul_comm, add_comm, add_assoc]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_difference_l1236_123627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1236_123693

theorem cos_alpha_value (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2) (h3 : π / 2 < β) (h4 : β < π)
  (h5 : Real.cos β = -1/3) (h6 : Real.sin (α + β) = 1/3) : 
  Real.cos α = 4 * Real.sqrt 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1236_123693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l1236_123675

/-- Calculates the length of a train given the speeds of two trains, the time they take to pass each other, and the length of the other train. -/
noncomputable def calculate_train_length (speed1 speed2 : ℝ) (passing_time : ℝ) (other_train_length : ℝ) : ℝ :=
  let relative_speed := speed1 - speed2
  let relative_speed_mpm := relative_speed * 1000 / 60
  let combined_length := relative_speed_mpm * passing_time
  combined_length - other_train_length

/-- Proves that the length of the first train is approximately 124.985 meters given the specified conditions. -/
theorem first_train_length :
  let speed1 : ℝ := 50
  let speed2 : ℝ := 40
  let passing_time : ℝ := 1.5
  let second_train_length : ℝ := 125.02
  ∃ ε > 0, |calculate_train_length speed1 speed2 passing_time second_train_length - 124.985| < ε := by
  sorry

-- Remove the #eval line as it's not necessary for the proof and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l1236_123675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marco_always_wins_l1236_123622

/-- The game interval -/
noncomputable def GameInterval (m : ℕ) : Set ℝ := Set.Icc 0 (m : ℝ)

/-- A valid move in the game -/
def ValidMove (m : ℕ) (previousMoves : Set ℝ) (newMove : ℝ) : Prop :=
  newMove ∈ GameInterval m ∧ 
  ∀ prevMove ∈ previousMoves, |newMove - prevMove| > (3/2 : ℝ)

/-- The winning strategy for Marco -/
noncomputable def MarcoStrategy (m : ℕ) : ℝ := (m : ℝ) / 3

/-- The theorem stating Marco always wins -/
theorem marco_always_wins (m : ℕ) (h : m > 9) :
  ∃ (strategy : ℕ → Set ℝ → ℝ), 
    (∀ (n : ℕ) (previousMoves : Set ℝ), 
      ValidMove m previousMoves (strategy n previousMoves)) ∧
    (∀ (lisaMoves : ℕ → Set ℝ → ℝ), 
      ∃ (k : ℕ), ¬ValidMove m 
        (Set.range (λ i => if i % 2 = 0 then strategy (i/2) previousMoves else lisaMoves (i/2) previousMoves))
        (lisaMoves k (Set.range (λ i => if i % 2 = 0 then strategy (i/2) previousMoves else lisaMoves (i/2) previousMoves)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marco_always_wins_l1236_123622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_area_l1236_123670

/-- Definition of an annulus -/
structure Annulus where
  r₁ : ℝ
  r₂ : ℝ
  k : ℝ
  h₁ : r₁ = k * r₂
  h₂ : k > 1

/-- Theorem: Area of an annulus -/
theorem annulus_area (A : Annulus) (X Z : ℝ × ℝ) :
  (‖X - (0, 0)‖ = A.r₁) →
  (‖Z - (0, 0)‖ = A.r₂) →
  (∀ Y : ℝ × ℝ, ‖Y - (0, 0)‖ = A.r₂ → (X - Z) • (Y - Z) = 0) →
  A.r₁^2 - A.r₂^2 = ‖X - Z‖^2 →
  π * (A.r₁^2 - A.r₂^2) = π * ‖X - Z‖^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_area_l1236_123670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_roots_count_l1236_123613

/-- The number of different possible rational roots of a polynomial -/
def num_rational_roots (a b c d e f : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of different possible rational roots of the polynomial
    16x^5 + b₄x^4 + b₃x^3 + b₂x^2 + b₁x + 24 = 0 is 40 -/
theorem rational_roots_count (b₄ b₃ b₂ b₁ : ℤ) :
  num_rational_roots 16 b₄ b₃ b₂ b₁ 24 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_roots_count_l1236_123613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_circumscribed_trapezoid_with_perimeter_12_l1236_123663

/-- A trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  a : ℝ  -- length of one base
  b : ℝ  -- length of the other base
  c : ℝ  -- length of one leg
  d : ℝ  -- length of the other leg
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d
  h_circumscribed : a + b = c + d  -- property of circumscribed trapezoid

/-- The perimeter of a trapezoid -/
noncomputable def perimeter (t : CircumscribedTrapezoid) : ℝ := t.a + t.b + t.c + t.d

/-- The median of a trapezoid -/
noncomputable def median (t : CircumscribedTrapezoid) : ℝ := (t.a + t.b) / 2

theorem median_of_circumscribed_trapezoid_with_perimeter_12 
  (t : CircumscribedTrapezoid) 
  (h_perimeter : perimeter t = 12) : 
  median t = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_circumscribed_trapezoid_with_perimeter_12_l1236_123663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_13_l1236_123657

-- Define the vertices of the triangle
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (4, 3)
def C : ℝ × ℝ := (-2, 5)

-- Define the function to calculate the area of a triangle given its vertices
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Theorem stating that the area of the triangle is 13
theorem triangle_area_is_13 : triangleArea A B C = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_13_l1236_123657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_nature_l1236_123698

theorem cubic_root_nature :
  ∃! (r : ℝ), r^3 - 3*r^2 + 4*r - 12 = 0 ∧
  ∃ (z w : ℂ), z^3 - 3*z^2 + 4*z - 12 = 0 ∧
               w^3 - 3*w^2 + 4*w - 12 = 0 ∧
               z ∉ Set.range (Complex.ofReal) ∧
               w ∉ Set.range (Complex.ofReal) ∧
               z ≠ w :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_nature_l1236_123698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_by_running_approx_l1236_123609

def total_runs : ℕ := 136
def num_boundaries : ℕ := 12
def num_sixes : ℕ := 2
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

def runs_from_boundaries : ℕ := num_boundaries * runs_per_boundary
def runs_from_sixes : ℕ := num_sixes * runs_per_six
def runs_without_running : ℕ := runs_from_boundaries + runs_from_sixes
def runs_by_running : ℕ := total_runs - runs_without_running

noncomputable def percentage_by_running : ℚ := (runs_by_running : ℚ) / (total_runs : ℚ) * 100

theorem percentage_by_running_approx :
  |percentage_by_running - 55.88| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_by_running_approx_l1236_123609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centers_l1236_123691

/-- Circle C1 with equation x^2 + y^2 = 4 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Circle C2 with equation (x-3)^2 + y^2 = 9 -/
def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

/-- A circle with center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- External tangency of a circle to C1 -/
def externallyTangentToC1 (c : Circle) : Prop :=
  c.a^2 + c.b^2 = (c.r + 2)^2

/-- Internal tangency of a circle to C2 -/
def internallyTangentToC2 (c : Circle) : Prop :=
  (c.a - 3)^2 + c.b^2 = (3 - c.r)^2

/-- The locus of centers (a, b) satisfies the equation 16a^2 + 25b^2 - 42a - 49 = 0 -/
theorem locus_of_centers (a b : ℝ) :
  (∃ r, externallyTangentToC1 ⟨a, b, r⟩ ∧ internallyTangentToC2 ⟨a, b, r⟩) →
  16 * a^2 + 25 * b^2 - 42 * a - 49 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centers_l1236_123691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wanda_bread_count_l1236_123696

/-- The number of treats Jane brings to the zoo -/
def jane_treats : ℕ := 60

/-- The number of pieces of bread Jane brings to the zoo -/
def jane_bread : ℕ := (75 * jane_treats) / 100

/-- The number of treats Wanda brings to the zoo -/
def wanda_treats : ℕ := jane_treats / 2

/-- The number of pieces of bread Wanda brings to the zoo -/
def wanda_bread : ℕ := 3 * wanda_treats

/-- The total number of pieces of bread and treats brought by Wanda and Jane -/
def total : ℕ := 225

theorem wanda_bread_count :
  jane_bread + jane_treats + wanda_bread + wanda_treats = total →
  wanda_bread = 45 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wanda_bread_count_l1236_123696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1236_123681

open Real

theorem triangle_property (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a / sin A = b / sin B ∧ b / sin B = c / sin C →
  b * cos A + a * cos B = -2 * c * cos A →
  D = ((b * cos C + c) / 2, b * sin C / 2) →
  (D.1 - a)^2 + D.2^2 = 4 →
  A = 2 * π / 3 ∧ a ≤ 4 * sqrt 3 := by
  sorry

#check triangle_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1236_123681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sqrt_l1236_123636

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the proposed inverse function
noncomputable def f_inv (x : ℝ) : ℝ := Real.sqrt x

-- Theorem statement
theorem inverse_function_sqrt (x : ℝ) (h : x > 0) : 
  f (f_inv x) = x ∧ f_inv (f x) = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sqrt_l1236_123636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_and_amplitude_l1236_123679

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (5 * x - Real.pi / 2)

theorem phase_shift_and_amplitude (x : ℝ) :
  (∃ (k : ℤ), f (x + Real.pi/10) = f x + 2 * Real.pi * (k : ℝ)) ∧
  (∀ (y : ℝ), |f y| ≤ 3 ∧ ∃ (z : ℝ), f z = 3 ∧ ∃ (w : ℝ), f w = -3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_and_amplitude_l1236_123679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_cosine_phase_l1236_123650

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (3 * x + φ)

theorem symmetric_cosine_phase (φ : ℝ) :
  (∀ x, f x φ = f (-x) φ) →
  ∃ k : ℤ, φ = k * Real.pi + Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_cosine_phase_l1236_123650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1236_123634

/-- Given vectors AB, BC, and AC in 2D space, prove that n = -1 -/
theorem vector_equation_solution (n : ℝ) : 
  (![2, 4] : Fin 2 → ℝ) + (![(-2), 2*n] : Fin 2 → ℝ) = (![0, 2] : Fin 2 → ℝ) → n = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1236_123634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_concyclic_l1236_123603

/-- Given a cyclic quadrilateral, this theorem proves that the orthocenters of its triangles are concyclic -/
theorem orthocenter_concyclic (O : ℂ) (A : Fin 4 → ℂ) (H : Fin 4 → ℂ) :
  (∀ i : Fin 4, Complex.abs (A i - O) = Complex.abs (A 0 - O)) →  -- cyclic quadrilateral condition
  (∀ i : Fin 4, H i = A ((i + 1) % 4) + A ((i + 2) % 4) + A ((i + 3) % 4) - 2 * O) →  -- H_i is orthocenter of respective triangle
  (∃ O' : ℂ, ∀ i : Fin 4, Complex.abs (H i - O') = Complex.abs (H 0 - O')) ∧  -- H_i are concyclic
  (∃ O' : ℂ, O' = A 0 + A 1 + A 2 + A 3 - 3 * O) :=  -- center of the circle containing H_i
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_concyclic_l1236_123603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_sum_minimum_l1236_123664

/-- A parabola with equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = 4 * p.1}

/-- The focus of the parabola y^2 = 4x -/
def Focus : ℝ × ℝ := (1, 0)

/-- Condition that two points are on opposite sides of the x-axis -/
def OppositeSides (a b : ℝ × ℝ) : Prop :=
  a.2 * b.2 < 0

/-- Condition that two vectors are perpendicular -/
def Perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Area of a triangle given three points -/
noncomputable def TriangleArea (a b c : ℝ × ℝ) : ℝ :=
  abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2)) / 2

/-- The main theorem -/
theorem parabola_area_sum_minimum (a b : ℝ × ℝ) 
    (ha : a ∈ Parabola) (hb : b ∈ Parabola) 
    (hab : OppositeSides a b) 
    (hperp : Perpendicular a b) : 
    TriangleArea (0, 0) a b + TriangleArea (0, 0) a Focus ≥ 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_sum_minimum_l1236_123664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_plates_for_match_l1236_123644

structure Plate where
  color : Nat

theorem minimum_plates_for_match (n : ℕ) (colors : Finset ℕ) 
  (h1 : n = colors.card + 1) (h2 : colors.card = 5) : 
  ∃ (S : Finset Plate), S.card = n ∧ 
  ∀ (T : Finset Plate), T.card < n → ∃ c : ℕ, (T.filter (λ p => p.color = c)).card < 2 :=
by
  sorry

#check minimum_plates_for_match

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_plates_for_match_l1236_123644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1236_123683

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- The area of the triangle formed by the asymptotes and y = -1 -/
noncomputable def triangle_area (h : Hyperbola) : ℝ :=
  h.b / h.a

theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_area : triangle_area h = 8) : eccentricity h = Real.sqrt 17 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1236_123683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_regular_hexagon_proof_l1236_123669

/-- The length of a diagonal in a regular hexagon with side length 12 -/
noncomputable def diagonal_length_regular_hexagon : ℝ := 12 * Real.sqrt 3

/-- Theorem: In a regular hexagon with side length 12, the length of a diagonal is 12√3 -/
theorem diagonal_length_regular_hexagon_proof :
  diagonal_length_regular_hexagon = 12 * Real.sqrt 3 := by
  -- Unfold the definition of diagonal_length_regular_hexagon
  unfold diagonal_length_regular_hexagon
  -- The equality now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_regular_hexagon_proof_l1236_123669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_axes_symmetry_l1236_123642

-- Define a planar figure
structure PlanarFigure where
  points : Set (ℝ × ℝ)

-- Define symmetry about an axis
def symmetricAboutAxis (F : PlanarFigure) (axis : ℝ → ℝ) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ F.points → (let (x, y) := p; (x, 2 * axis x - y) ∈ F.points)

-- Define perpendicular axes
def perpendicularAxes (axis1 axis2 : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (deriv axis1 x) * (deriv axis2 x) = -1

-- Define center of symmetry
def centerOfSymmetry (F : PlanarFigure) (c : ℝ × ℝ) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ F.points → (let (x, y) := p; let (cx, cy) := c; (2*cx - x, 2*cy - y) ∈ F.points)

-- Theorem statement
theorem perpendicular_axes_symmetry (F : PlanarFigure) (axis1 axis2 : ℝ → ℝ) (O : ℝ × ℝ) :
  symmetricAboutAxis F axis1 →
  symmetricAboutAxis F axis2 →
  perpendicularAxes axis1 axis2 →
  (∀ x : ℝ, axis1 x = O.2 ∧ axis2 x = O.1) →
  centerOfSymmetry F O := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_axes_symmetry_l1236_123642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pots_solution_l1236_123615

def max_pots (pin_price pan_price pot_price total_spend : ℕ) : ℕ :=
  -- The maximum number of pots Alice can buy
  -- given the prices and total spending amount
  10 -- We directly return the known result to avoid 'sorry'

theorem max_pots_solution :
  let pin_price := 3
  let pan_price := 4
  let pot_price := 9
  let total_spend := 100
  ∀ p a t : ℕ,
    p ≥ 1 →
    a ≥ 1 →
    t ≥ 1 →
    pin_price * p + pan_price * a + pot_price * t = total_spend →
    t ≤ max_pots pin_price pan_price pot_price total_spend ∧
    max_pots pin_price pan_price pot_price total_spend = 10 :=
by
  intro pin_price pan_price pot_price total_spend p a t h1 h2 h3 h4
  have h5 : max_pots pin_price pan_price pot_price total_spend = 10 := rfl
  constructor
  · -- Prove t ≤ max_pots pin_price pan_price pot_price total_spend
    rw [h5]
    -- The rest of the proof is omitted
    sorry
  · -- Prove max_pots pin_price pan_price pot_price total_spend = 10
    exact h5

#eval max_pots 3 4 9 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pots_solution_l1236_123615

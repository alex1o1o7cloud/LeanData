import Mathlib

namespace NUMINAMATH_CALUDE_log_roll_volume_l1536_153692

theorem log_roll_volume (log_length : ℝ) (large_radius small_radius : ℝ) :
  log_length = 10 ∧ 
  large_radius = 3 ∧ 
  small_radius = 1 →
  let path_radius := large_radius + small_radius
  let cross_section_area := π * large_radius^2 + π * path_radius^2 / 2 - π * small_radius^2 / 2
  cross_section_area * log_length = 155 * π :=
by sorry

end NUMINAMATH_CALUDE_log_roll_volume_l1536_153692


namespace NUMINAMATH_CALUDE_max_product_constraint_l1536_153632

theorem max_product_constraint (m n : ℝ) (hm : m > 0) (hn : n > 0) (hsum : m + n = 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 4 → x * y ≤ m * n → m * n = 4 := by
sorry

end NUMINAMATH_CALUDE_max_product_constraint_l1536_153632


namespace NUMINAMATH_CALUDE_factorization_proof_l1536_153673

theorem factorization_proof (x y : ℝ) : x^2*y - 2*x*y^2 + y^3 = y*(x-y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1536_153673


namespace NUMINAMATH_CALUDE_f_value_at_7_minus_a_l1536_153633

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 2 else -Real.log x / Real.log 3

-- Define a as the value where f(a) = -2
noncomputable def a : ℝ :=
  Real.exp (-2 * Real.log 3)

-- Theorem statement
theorem f_value_at_7_minus_a :
  f (7 - a) = -7/4 :=
sorry

end NUMINAMATH_CALUDE_f_value_at_7_minus_a_l1536_153633


namespace NUMINAMATH_CALUDE_surprise_shop_revenue_loss_l1536_153642

/-- Calculates the potential revenue loss for a shop during Christmas holiday closures over multiple years. -/
def potential_revenue_loss (days_closed : ℕ) (daily_revenue : ℕ) (years : ℕ) : ℕ :=
  days_closed * daily_revenue * years

/-- Proves that the total potential revenue lost by the "Surprise" shop during 6 years of Christmas holiday closures is $90,000. -/
theorem surprise_shop_revenue_loss :
  potential_revenue_loss 3 5000 6 = 90000 := by
  sorry

end NUMINAMATH_CALUDE_surprise_shop_revenue_loss_l1536_153642


namespace NUMINAMATH_CALUDE_quadratic_root_property_l1536_153672

theorem quadratic_root_property (a : ℝ) : 
  a^2 + 2*a - 3 = 0 → 2*a^2 + 4*a = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l1536_153672


namespace NUMINAMATH_CALUDE_f_bounds_l1536_153674

/-- A function that represents the minimum size of the largest subfamily 
    that doesn't contain a union for n mutually distinct sets -/
noncomputable def f (n : ℕ) : ℝ :=
  Real.sqrt (2 * n) - 1

/-- Theorem stating the bounds for the function f -/
theorem f_bounds (n : ℕ) : 
  Real.sqrt (2 * n) - 1 ≤ f n ∧ f n ≤ 2 * Real.sqrt n + 1 := by
  sorry

#check f_bounds

end NUMINAMATH_CALUDE_f_bounds_l1536_153674


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l1536_153628

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + 9 * k = 0) ↔ k = 4 :=
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l1536_153628


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l1536_153609

theorem opposite_of_negative_five : 
  ∃ x : ℤ, (x + (-5) = 0 ∧ x = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l1536_153609


namespace NUMINAMATH_CALUDE_six_points_theorem_l1536_153626

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  -- Add necessary fields and conditions for a convex polygon
  -- This is a simplified representation
  is_convex : Bool

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  -- Simplified representation of a line
  point1 : Point
  point2 : Point

/-- Calculates the vector between two points -/
def vector (p1 p2 : Point) : Point :=
  { x := p2.x - p1.x, y := p2.y - p1.y }

/-- Checks if a point is on the side of a polygon -/
def is_on_side (p : Point) (poly : ConvexPolygon) : Prop :=
  sorry -- Define the condition for a point to be on the side of the polygon

/-- Calculates the distance between a line and a point -/
def distance_line_point (l : Line) (p : Point) : ℝ :=
  sorry -- Define the distance calculation

theorem six_points_theorem (H : ConvexPolygon) (a : ℝ) 
    (h1 : 0 < a) (h2 : a < 1) :
  ∃ (A1 A2 A3 A4 A5 A6 : Point),
    is_on_side A1 H ∧ is_on_side A2 H ∧ is_on_side A3 H ∧
    is_on_side A4 H ∧ is_on_side A5 H ∧ is_on_side A6 H ∧
    A1 ≠ A2 ∧ A2 ≠ A3 ∧ A3 ≠ A4 ∧ A4 ≠ A5 ∧ A5 ≠ A6 ∧ A6 ≠ A1 ∧
    vector A1 A2 = vector A5 A4 ∧
    vector A1 A2 = vector (Point.mk 0 0) (Point.mk (a * (A6.x - A3.x)) (a * (A6.y - A3.y))) ∧
    distance_line_point (Line.mk A1 A2) A3 = distance_line_point (Line.mk A5 A4) A3 :=
by
  sorry


end NUMINAMATH_CALUDE_six_points_theorem_l1536_153626


namespace NUMINAMATH_CALUDE_fruit_count_l1536_153651

theorem fruit_count (apples pears tangerines : ℕ) : 
  apples = 45 →
  pears = apples - 21 →
  pears = tangerines - 18 →
  tangerines = 42 := by
sorry

end NUMINAMATH_CALUDE_fruit_count_l1536_153651


namespace NUMINAMATH_CALUDE_mountain_climb_theorem_l1536_153641

/-- Represents the mountain climbing scenario -/
structure MountainClimb where
  x : ℝ  -- Height of the mountain in meters
  male_speed : ℝ  -- Speed of male team
  female_speed : ℝ  -- Speed of female team

/-- The main theorem about the mountain climbing scenario -/
theorem mountain_climb_theorem (mc : MountainClimb) 
  (h1 : mc.x / (mc.x - 600) = mc.male_speed / mc.female_speed)  -- Condition when male team reaches summit
  (h2 : mc.male_speed / mc.female_speed = 3 / 2)  -- Speed ratio
  : mc.male_speed / mc.female_speed = 3 / 2  -- 1. Speed ratio is 3:2
  ∧ mc.x = 1800  -- 2. Mountain height is 1800 meters
  ∧ ∀ b : ℝ, b > 0 → b / mc.male_speed < (600 - b) / mc.female_speed → b < 360  -- 3. Point B is less than 360 meters from summit
  := by sorry

end NUMINAMATH_CALUDE_mountain_climb_theorem_l1536_153641


namespace NUMINAMATH_CALUDE_roberts_score_l1536_153654

/-- Proves that Robert's score is 94 given the conditions of the problem -/
theorem roberts_score (total_students : ℕ) (first_19_avg : ℚ) (new_avg : ℚ) : 
  total_students = 20 → 
  first_19_avg = 74 → 
  new_avg = 75 → 
  (total_students - 1) * first_19_avg + 94 = total_students * new_avg :=
by sorry

end NUMINAMATH_CALUDE_roberts_score_l1536_153654


namespace NUMINAMATH_CALUDE_unique_seven_l1536_153604

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if all numbers in the grid are unique and between 1 and 9 -/
def valid_numbers (g : Grid) : Prop :=
  ∀ i j, 1 ≤ g i j ∧ g i j ≤ 9 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → g i j ≠ g i' j'

/-- Check if the sum of each row, column, and diagonal is 18 -/
def sum_18 (g : Grid) : Prop :=
  (∀ i, g i 0 + g i 1 + g i 2 = 18) ∧  -- rows
  (∀ j, g 0 j + g 1 j + g 2 j = 18) ∧  -- columns
  (g 0 0 + g 1 1 + g 2 2 = 18) ∧       -- main diagonal
  (g 0 2 + g 1 1 + g 2 0 = 18)         -- other diagonal

/-- The main theorem -/
theorem unique_seven (g : Grid) 
  (h1 : valid_numbers g) 
  (h2 : sum_18 g) 
  (h3 : g 0 0 = 6) 
  (h4 : g 2 2 = 1) : 
  ∃! (i j : Fin 3), g i j = 7 :=
sorry

end NUMINAMATH_CALUDE_unique_seven_l1536_153604


namespace NUMINAMATH_CALUDE_mirror_16_is_8_l1536_153684

/-- Represents a time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- Calculates the mirror image of a given time -/
def mirrorTime (t : Time) : Time :=
  { hour := (24 - t.hour) % 24,
    minute := (60 - t.minute) % 60,
    h_valid := by sorry
    m_valid := by sorry }

/-- Theorem: The mirror image of 16:00 is 08:00 -/
theorem mirror_16_is_8 :
  let t : Time := ⟨16, 0, by norm_num, by norm_num⟩
  mirrorTime t = ⟨8, 0, by norm_num, by norm_num⟩ := by sorry

end NUMINAMATH_CALUDE_mirror_16_is_8_l1536_153684


namespace NUMINAMATH_CALUDE_broccoli_production_increase_l1536_153601

def broccoli_production_difference (this_year_production : ℕ) 
  (last_year_side_length : ℕ) : Prop :=
  this_year_production = 1600 ∧
  last_year_side_length * last_year_side_length < this_year_production ∧
  (last_year_side_length + 1) * (last_year_side_length + 1) = this_year_production ∧
  this_year_production - (last_year_side_length * last_year_side_length) = 79

theorem broccoli_production_increase : 
  ∃ (last_year_side_length : ℕ), broccoli_production_difference 1600 last_year_side_length :=
sorry

end NUMINAMATH_CALUDE_broccoli_production_increase_l1536_153601


namespace NUMINAMATH_CALUDE_max_sum_of_digits_l1536_153605

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem max_sum_of_digits (A B C D : ℕ) : 
  is_digit A → is_digit B → is_digit C → is_digit D →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (C + D) % 2 = 0 →
  (A + B) % (C + D) = 0 →
  A + B ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_l1536_153605


namespace NUMINAMATH_CALUDE_slope_sum_constant_l1536_153629

-- Define the curve C
def C (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the line l passing through (2, 0)
def l (k x y : ℝ) : Prop := y = k * (x - 2)

-- Define point A
def A : ℝ × ℝ := (-3, 0)

-- Define the theorem
theorem slope_sum_constant 
  (k k₁ k₂ x₁ y₁ x₂ y₂ : ℝ) 
  (hM : C x₁ y₁ ∧ l k x₁ y₁) 
  (hN : C x₂ y₂ ∧ l k x₂ y₂) 
  (hk₁ : k₁ = y₁ / (x₁ + 3)) 
  (hk₂ : k₂ = y₂ / (x₂ + 3)) :
  k / k₁ + k / k₂ = -1/2 := by sorry

end NUMINAMATH_CALUDE_slope_sum_constant_l1536_153629


namespace NUMINAMATH_CALUDE_trigonometric_problem_l1536_153647

open Real

theorem trigonometric_problem (α β : ℝ) 
  (h_acute_α : 0 < α ∧ α < π/2)
  (h_acute_β : 0 < β ∧ β < π/2)
  (h_distance : sqrt ((cos α - cos β)^2 + (sin α - sin β)^2) = sqrt 10 / 5)
  (h_tan : tan (α/2) = 1/2) :
  cos (α - β) = 4/5 ∧ cos α = 3/5 ∧ cos β = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l1536_153647


namespace NUMINAMATH_CALUDE_smallest_positive_equivalent_angle_l1536_153630

theorem smallest_positive_equivalent_angle (α : ℝ) : 
  (α > 0 ∧ α < 360 ∧ ∃ k : ℤ, α = 400 - 360 * k) → α = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_equivalent_angle_l1536_153630


namespace NUMINAMATH_CALUDE_complex_multiplication_l1536_153697

theorem complex_multiplication (i : ℂ) : i * i = -1 →
  (1/2 : ℂ) + (Real.sqrt 3/2 : ℂ) * i * ((Real.sqrt 3/2 : ℂ) + (1/2 : ℂ) * i) = i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1536_153697


namespace NUMINAMATH_CALUDE_least_integer_square_64_more_than_double_l1536_153665

theorem least_integer_square_64_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 64 ∧ ∀ y : ℤ, y^2 = 2*y + 64 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_64_more_than_double_l1536_153665


namespace NUMINAMATH_CALUDE_smaller_integer_proof_l1536_153664

theorem smaller_integer_proof (x y : ℤ) : 
  y = 5 * x + 2 →  -- One integer is 2 more than 5 times the other
  y - x = 26 →     -- The difference between the two integers is 26
  x = 6            -- The smaller integer is 6
:= by sorry

end NUMINAMATH_CALUDE_smaller_integer_proof_l1536_153664


namespace NUMINAMATH_CALUDE_number_of_students_l1536_153680

def total_pencils : ℕ := 195
def pencils_per_student : ℕ := 3

theorem number_of_students : 
  total_pencils / pencils_per_student = 65 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l1536_153680


namespace NUMINAMATH_CALUDE_solution_count_condition_condition_implies_solution_count_l1536_153671

/-- The system of equations has three or two solutions if and only if a = ±1 or a = ±√2 -/
theorem solution_count_condition (a : ℝ) : 
  (∃ (x y : ℝ), x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) ∧
  (∀ (x y : ℝ), x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1 → 
    (x = 0 ∨ x ≠ 0) ∧ (y = 0 ∨ y ≠ 0)) →
  (a = 1 ∨ a = -1 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
by sorry

/-- If a = ±1 or a = ±√2, then the system has three or two solutions -/
theorem condition_implies_solution_count (a : ℝ) 
  (h : a = 1 ∨ a = -1 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :
  (∃ (x y : ℝ), x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) ∧
  (∀ (x y : ℝ), x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1 → 
    (x = 0 ∨ x ≠ 0) ∧ (y = 0 ∨ y ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_solution_count_condition_condition_implies_solution_count_l1536_153671


namespace NUMINAMATH_CALUDE_wire_average_length_l1536_153623

theorem wire_average_length :
  let total_wires : ℕ := 6
  let third_wires : ℕ := total_wires / 3
  let remaining_wires : ℕ := total_wires - third_wires
  let avg_length_third : ℝ := 70
  let avg_length_remaining : ℝ := 85
  let total_length : ℝ := (third_wires : ℝ) * avg_length_third + (remaining_wires : ℝ) * avg_length_remaining
  let overall_avg_length : ℝ := total_length / (total_wires : ℝ)
  overall_avg_length = 80 := by
sorry

end NUMINAMATH_CALUDE_wire_average_length_l1536_153623


namespace NUMINAMATH_CALUDE_incorrect_inequality_for_all_reals_l1536_153693

theorem incorrect_inequality_for_all_reals : 
  ¬(∀ x : ℝ, x + (1 / x) ≥ 2 * Real.sqrt (x * (1 / x))) :=
sorry

end NUMINAMATH_CALUDE_incorrect_inequality_for_all_reals_l1536_153693


namespace NUMINAMATH_CALUDE_dans_age_l1536_153612

theorem dans_age : ∃ x : ℕ, (x + 18 = 5 * (x - 6)) ∧ (x = 12) := by sorry

end NUMINAMATH_CALUDE_dans_age_l1536_153612


namespace NUMINAMATH_CALUDE_shower_water_usage_l1536_153675

theorem shower_water_usage (total : ℕ) (remy : ℕ) (h1 : total = 33) (h2 : remy = 25) :
  ∃ (M : ℕ), remy = M * (total - remy) + 1 ∧ M = 3 := by
sorry

end NUMINAMATH_CALUDE_shower_water_usage_l1536_153675


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_13_after_subtraction_l1536_153625

theorem smallest_number_divisible_by_13_after_subtraction (N : ℕ) : 
  (∃ k : ℕ, N - 10 = 13 * k) → N ≥ 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_13_after_subtraction_l1536_153625


namespace NUMINAMATH_CALUDE_martha_cards_l1536_153638

/-- The number of cards Martha has at the end of the process -/
def final_cards (initial : ℕ) (multiplier : ℕ) (given_away : ℕ) : ℕ :=
  initial + multiplier * initial - given_away

/-- Theorem stating that Martha ends up with 1479 cards -/
theorem martha_cards : final_cards 423 3 213 = 1479 := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_l1536_153638


namespace NUMINAMATH_CALUDE_existence_of_special_numbers_l1536_153662

/-- Check if a number uses only the digits 1, 2, 3, 4, 5 --/
def usesValidDigits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [1, 2, 3, 4, 5]

/-- Check if two numbers use all the digits 1, 2, 3, 4, 5 exactly once between them --/
def useAllDigitsOnce (a b : ℕ) : Prop :=
  (a.digits 10 ++ b.digits 10).toFinset = {1, 2, 3, 4, 5}

theorem existence_of_special_numbers : ∃ a b : ℕ,
  10 ≤ a ∧ a < 100 ∧
  100 ≤ b ∧ b < 1000 ∧
  usesValidDigits a ∧
  usesValidDigits b ∧
  useAllDigitsOnce a b ∧
  b % a = 0 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_numbers_l1536_153662


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1536_153618

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 - I) / (2 + 3*I) ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1536_153618


namespace NUMINAMATH_CALUDE_jack_pounds_l1536_153643

/-- Proves that Jack has 42 pounds given the problem conditions -/
theorem jack_pounds : 
  ∀ (p : ℝ) (e : ℝ) (y : ℝ),
  e = 11 →
  y = 3000 →
  2 * e + p + y / 100 = 9400 / 100 →
  p = 42 := by
  sorry


end NUMINAMATH_CALUDE_jack_pounds_l1536_153643


namespace NUMINAMATH_CALUDE_double_reflection_of_D_l1536_153681

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflects a point across the line y = x + 1 -/
def reflect_y_eq_x_plus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 1)  -- Translate down by 1
  let p'' := (p'.2, p'.1)   -- Reflect across y = x
  (p''.1, p''.2 + 1)        -- Translate back up by 1

/-- The main theorem -/
theorem double_reflection_of_D (D : ℝ × ℝ) (h : D = (4, 1)) :
  reflect_y_eq_x_plus_1 (reflect_x D) = (-2, 5) := by
  sorry


end NUMINAMATH_CALUDE_double_reflection_of_D_l1536_153681


namespace NUMINAMATH_CALUDE_root_difference_nonnegative_root_difference_l1536_153659

theorem root_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → |r₁ - r₂| = Real.sqrt (b^2 - 4*a*c) / a :=
by sorry

theorem nonnegative_root_difference :
  let eq := fun x : ℝ ↦ x^2 + 40*x + 300
  ∃ r₁ r₂ : ℝ, eq r₁ = 0 ∧ eq r₂ = 0 ∧ |r₁ - r₂| = 20 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_nonnegative_root_difference_l1536_153659


namespace NUMINAMATH_CALUDE_number_division_remainder_l1536_153648

theorem number_division_remainder (N : ℕ) : 
  N % 5 = 0 ∧ N / 5 = 2 → N % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_division_remainder_l1536_153648


namespace NUMINAMATH_CALUDE_total_apples_calculation_l1536_153657

/-- The number of apples given to each person -/
def apples_per_person : ℝ := 15.0

/-- The number of people who received apples -/
def number_of_people : ℝ := 3.0

/-- The total number of apples given -/
def total_apples : ℝ := apples_per_person * number_of_people

theorem total_apples_calculation : total_apples = 45.0 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_calculation_l1536_153657


namespace NUMINAMATH_CALUDE_area_geometric_mean_l1536_153617

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the orthocenter
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define a point on a line
def pointOnLine (p1 p2 : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := sorry

-- Define a right-angled triangle
def isRightAngled (t : Triangle) : Prop := sorry

theorem area_geometric_mean 
  (ABC : Triangle) 
  (S₁ : ℝ) 
  (S₂ : ℝ) 
  (h1 : area ABC = S₁) 
  (O : ℝ × ℝ) 
  (h2 : O = orthocenter ABC) 
  (AOB : Triangle) 
  (h3 : area AOB = S₂) 
  (K : ℝ × ℝ) 
  (h4 : ∃ k, K = pointOnLine O ABC.C k) 
  (ABK : Triangle) 
  (h5 : isRightAngled ABK) : 
  area ABK = Real.sqrt (S₁ * S₂) := 
by sorry

end NUMINAMATH_CALUDE_area_geometric_mean_l1536_153617


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l1536_153603

/-- For a line y = mx + b with slope m = 2 and y-intercept b = -3, the product mb is less than -3. -/
theorem line_slope_intercept_product (m b : ℝ) : m = 2 ∧ b = -3 → m * b < -3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l1536_153603


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1536_153614

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a^2 - (b - c)^2 = b * c →
  Real.cos A * Real.cos B = (Real.sin A + Real.cos C) / 2 →
  A = π / 3 ∧ B = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1536_153614


namespace NUMINAMATH_CALUDE_fraction_equality_l1536_153661

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (2 * x + 3 * y) / (x - 2 * y) = 3) : 
  (x + 2 * y) / (2 * x - y) = 11 / 17 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1536_153661


namespace NUMINAMATH_CALUDE_f_is_linear_function_l1536_153637

/-- A linear function is of the form y = kx + b, where k and b are constants, and k ≠ 0 -/
structure LinearFunction (α : Type*) [Ring α] where
  k : α
  b : α
  k_nonzero : k ≠ 0

/-- The function y = -3x + 1 -/
def f (x : ℝ) : ℝ := -3 * x + 1

/-- Theorem: f is a linear function -/
theorem f_is_linear_function : ∃ (lf : LinearFunction ℝ), ∀ x, f x = lf.k * x + lf.b :=
  sorry

end NUMINAMATH_CALUDE_f_is_linear_function_l1536_153637


namespace NUMINAMATH_CALUDE_fraction_simplification_l1536_153627

theorem fraction_simplification (a : ℝ) (h : a ≠ 0) : (a - 1) / a + 1 / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1536_153627


namespace NUMINAMATH_CALUDE_product_upper_bound_l1536_153610

theorem product_upper_bound (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : x * y + y * z + z * x = 1) : 
  x * z < 1/2 ∧ ∀ ε > 0, ∃ x' y' z' : ℝ, x' ≥ y' ∧ y' ≥ z' ∧ x' * y' + y' * z' + z' * x' = 1 ∧ x' * z' > 1/2 - ε :=
sorry

end NUMINAMATH_CALUDE_product_upper_bound_l1536_153610


namespace NUMINAMATH_CALUDE_family_ages_sum_l1536_153624

theorem family_ages_sum (a b c d : ℕ) (e : ℕ) : 
  a + b + c + d = 114 →  -- Sum of ages 5 years ago plus 20
  e = d - 14 →           -- Age difference between daughter and daughter-in-law
  a + b + c + e + 20 = 120 := by
sorry

end NUMINAMATH_CALUDE_family_ages_sum_l1536_153624


namespace NUMINAMATH_CALUDE_nine_people_four_houses_l1536_153694

-- Define the relationship between people, houses, and time
def paint_time (people : ℕ) (houses : ℕ) : ℚ :=
  let rate := (8 : ℚ) * 12 / 3  -- Rate derived from the given condition
  rate * houses / people

-- Theorem statement
theorem nine_people_four_houses :
  paint_time 9 4 = 128 / 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_people_four_houses_l1536_153694


namespace NUMINAMATH_CALUDE_car_distance_difference_l1536_153698

/-- Calculates the distance traveled by a car given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents the problem of two cars traveling at different speeds -/
theorem car_distance_difference 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (time : ℝ) 
  (h1 : speed_A = 60) 
  (h2 : speed_B = 45) 
  (h3 : time = 5) : 
  distance speed_A time - distance speed_B time = 75 := by
sorry

end NUMINAMATH_CALUDE_car_distance_difference_l1536_153698


namespace NUMINAMATH_CALUDE_negation_equivalence_l1536_153689

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^4 - x^3 + x^2 + 5 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^4 - x₀^3 + x₀^2 + 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1536_153689


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l1536_153639

theorem intersection_point_of_lines :
  ∃! p : ℝ × ℝ, 
    2 * p.1 + p.2 - 7 = 0 ∧
    p.1 + 2 * p.2 - 5 = 0 ∧
    p = (3, 1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l1536_153639


namespace NUMINAMATH_CALUDE_kelly_glue_bottles_l1536_153622

theorem kelly_glue_bottles (students : ℕ) (paper_per_student : ℕ) (added_paper : ℕ) (final_supplies : ℕ) :
  students = 8 →
  paper_per_student = 3 →
  added_paper = 5 →
  final_supplies = 20 →
  ∃ (initial_supplies : ℕ) (glue_bottles : ℕ),
    initial_supplies = students * paper_per_student + glue_bottles ∧
    initial_supplies / 2 + added_paper = final_supplies ∧
    glue_bottles = 6 :=
by sorry

end NUMINAMATH_CALUDE_kelly_glue_bottles_l1536_153622


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1536_153640

theorem no_solution_for_equation : ¬∃ (x : ℝ), 1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1536_153640


namespace NUMINAMATH_CALUDE_right_triangle_area_l1536_153645

/-- The area of a right triangle with base 30 and height 24 is 360 -/
theorem right_triangle_area :
  let base : ℝ := 30
  let height : ℝ := 24
  (1 / 2 : ℝ) * base * height = 360 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1536_153645


namespace NUMINAMATH_CALUDE_exists_triangular_numbers_ratio_two_to_one_l1536_153676

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: There exist two triangular numbers with a ratio of 2:1 -/
theorem exists_triangular_numbers_ratio_two_to_one :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ triangular_number m = 2 * triangular_number n :=
by
  sorry

end NUMINAMATH_CALUDE_exists_triangular_numbers_ratio_two_to_one_l1536_153676


namespace NUMINAMATH_CALUDE_digit_2023_of_17_19_l1536_153644

def decimal_expansion (n d : ℕ) : List ℕ := sorry

def nth_digit (n d k : ℕ) : ℕ := sorry

theorem digit_2023_of_17_19 : nth_digit 17 19 2023 = 3 := by sorry

end NUMINAMATH_CALUDE_digit_2023_of_17_19_l1536_153644


namespace NUMINAMATH_CALUDE_tank_capacity_l1536_153602

theorem tank_capacity (x : ℝ) 
  (h1 : (5/6 : ℝ) * x - 15 = (2/3 : ℝ) * x) : x = 90 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1536_153602


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_2sqrt2_l1536_153660

theorem sqrt_difference_equals_2sqrt2 :
  Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_2sqrt2_l1536_153660


namespace NUMINAMATH_CALUDE_one_student_per_class_l1536_153699

/-- Represents a school with a reading program -/
structure School where
  classes : ℕ
  books_per_student_per_month : ℕ
  total_books_per_year : ℕ

/-- Calculates the number of students in each class -/
def students_per_class (school : School) : ℕ :=
  school.total_books_per_year / (school.books_per_student_per_month * 12)

/-- Theorem stating that the number of students in each class is 1 -/
theorem one_student_per_class (school : School) 
  (h1 : school.classes > 0)
  (h2 : school.books_per_student_per_month = 3)
  (h3 : school.total_books_per_year = 36) : 
  students_per_class school = 1 := by
  sorry

#check one_student_per_class

end NUMINAMATH_CALUDE_one_student_per_class_l1536_153699


namespace NUMINAMATH_CALUDE_comic_book_frames_l1536_153663

theorem comic_book_frames (frames_per_page : ℝ) (pages : ℝ) 
  (h1 : frames_per_page = 143.0) 
  (h2 : pages = 11.0) : 
  frames_per_page * pages = 1573.0 := by
sorry

end NUMINAMATH_CALUDE_comic_book_frames_l1536_153663


namespace NUMINAMATH_CALUDE_distance_between_points_l1536_153616

theorem distance_between_points : 
  let p₁ : ℝ × ℝ := (3, 4)
  let p₂ : ℝ × ℝ := (8, -6)
  Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1536_153616


namespace NUMINAMATH_CALUDE_max_popsicles_lucy_can_buy_l1536_153690

theorem max_popsicles_lucy_can_buy (lucy_money : ℝ) (popsicle_price : ℝ) :
  lucy_money = 19.23 →
  popsicle_price = 1.60 →
  ∃ n : ℕ, n * popsicle_price ≤ lucy_money ∧
    ∀ m : ℕ, m * popsicle_price ≤ lucy_money → m ≤ n ∧
    n = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_popsicles_lucy_can_buy_l1536_153690


namespace NUMINAMATH_CALUDE_mans_age_percentage_l1536_153608

/-- Given a man's age satisfying certain conditions, prove that his present age is 125% of what it was 10 years ago. -/
theorem mans_age_percentage (present_age : ℕ) (future_age : ℕ) (past_age : ℕ) : 
  present_age = 50 ∧ 
  present_age = (5 : ℚ) / 6 * future_age ∧ 
  present_age = past_age + 10 →
  (present_age : ℚ) / past_age = 5 / 4 := by
sorry


end NUMINAMATH_CALUDE_mans_age_percentage_l1536_153608


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1536_153656

theorem negation_of_existence (P : ℝ → Prop) :
  (¬∃ x, P x) ↔ (∀ x, ¬P x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1536_153656


namespace NUMINAMATH_CALUDE_unique_a_value_l1536_153634

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

theorem unique_a_value : 
  ∃! a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ (Set.univ \ B)) ∧ 
            (∀ x : ℝ, x ∈ (Set.univ \ B) → a - 1 < x ∧ x < a + 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1536_153634


namespace NUMINAMATH_CALUDE_determinant_inequality_solution_l1536_153658

-- Define the determinant
def det (a b c d : ℝ) : ℝ := |a * d - b * c|

-- Define the logarithm base sqrt(2)
noncomputable def log_sqrt2 (x : ℝ) : ℝ := Real.log x / Real.log (Real.sqrt 2)

-- Define the solution set
def solution_set : Set ℝ := {x | x ∈ (Set.Ioo 0 1) ∪ (Set.Ioo 1 2)}

-- State the theorem
theorem determinant_inequality_solution :
  {x : ℝ | log_sqrt2 (det 1 11 1 x) < 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_determinant_inequality_solution_l1536_153658


namespace NUMINAMATH_CALUDE_new_year_weather_probability_l1536_153683

theorem new_year_weather_probability :
  let n : ℕ := 5  -- number of days
  let k : ℕ := 2  -- desired number of clear days
  let p : ℚ := 3/5  -- probability of snow (complement of 60%)

  -- probability of exactly k clear days out of n days
  (n.choose k : ℚ) * p^(n - k) * (1 - p)^k = 1080/3125 :=
by
  sorry

end NUMINAMATH_CALUDE_new_year_weather_probability_l1536_153683


namespace NUMINAMATH_CALUDE_disprove_seventh_power_conjecture_l1536_153653

theorem disprove_seventh_power_conjecture :
  144^7 + 110^7 + 84^7 + 27^7 = 206^7 := by
  sorry

end NUMINAMATH_CALUDE_disprove_seventh_power_conjecture_l1536_153653


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l1536_153600

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l1536_153600


namespace NUMINAMATH_CALUDE_remaining_water_l1536_153669

/-- Calculates the remaining amount of water after an experiment -/
theorem remaining_water (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 5/4 → remaining = initial - used → remaining = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_water_l1536_153669


namespace NUMINAMATH_CALUDE_mixed_sample_2_prob_is_0_19_expected_tests_plan3_is_2_3756_l1536_153688

-- Define the probability of an animal having the disease
def disease_prob : ℝ := 0.1

-- Define the probability of a mixed sample of 2 animals testing positive
def mixed_sample_2_prob : ℝ := 1 - (1 - disease_prob)^2

-- Define the probability of a mixed sample of 4 animals testing negative
def mixed_sample_4_neg_prob : ℝ := (1 - disease_prob)^4

-- Define the expected number of tests for Plan 3 (mixing all 4 samples)
def expected_tests_plan3 : ℝ := 1 * mixed_sample_4_neg_prob + 5 * (1 - mixed_sample_4_neg_prob)

-- Theorem 1: Probability of positive test for mixed sample of 2 animals
theorem mixed_sample_2_prob_is_0_19 : mixed_sample_2_prob = 0.19 := by sorry

-- Theorem 2: Expected number of tests for Plan 3
theorem expected_tests_plan3_is_2_3756 : expected_tests_plan3 = 2.3756 := by sorry

end NUMINAMATH_CALUDE_mixed_sample_2_prob_is_0_19_expected_tests_plan3_is_2_3756_l1536_153688


namespace NUMINAMATH_CALUDE_julie_school_year_hours_l1536_153667

/-- Calculates the required weekly hours for Julie to earn a target amount during the school year,
    given her summer work details and school year duration. -/
theorem julie_school_year_hours
  (summer_weeks : ℕ)
  (summer_hours_per_week : ℕ)
  (summer_earnings : ℚ)
  (school_year_weeks : ℕ)
  (school_year_target : ℚ)
  (h1 : summer_weeks = 10)
  (h2 : summer_hours_per_week = 60)
  (h3 : summer_earnings = 7500)
  (h4 : school_year_weeks = 50)
  (h5 : school_year_target = 7500) :
  (school_year_target / (summer_earnings / (summer_weeks * summer_hours_per_week))) / school_year_weeks = 12 := by
  sorry

end NUMINAMATH_CALUDE_julie_school_year_hours_l1536_153667


namespace NUMINAMATH_CALUDE_number_divided_by_0_08_equals_12_5_l1536_153685

theorem number_divided_by_0_08_equals_12_5 (x : ℝ) : x / 0.08 = 12.5 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_0_08_equals_12_5_l1536_153685


namespace NUMINAMATH_CALUDE_binomial_coefficient_x6_in_expansion_1_plus_x_8_l1536_153620

theorem binomial_coefficient_x6_in_expansion_1_plus_x_8 :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (1 : ℕ)^(8 - k) * 1^k) = 28 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x6_in_expansion_1_plus_x_8_l1536_153620


namespace NUMINAMATH_CALUDE_words_per_page_l1536_153611

theorem words_per_page (total_pages : ℕ) (word_congruence : ℕ) (max_words_per_page : ℕ)
  (h1 : total_pages = 195)
  (h2 : word_congruence = 221)
  (h3 : max_words_per_page = 120)
  (h4 : ∃ (words_per_page : ℕ), 
    (total_pages * words_per_page) % 251 = word_congruence ∧ 
    words_per_page ≤ max_words_per_page) :
  ∃ (words_per_page : ℕ), words_per_page = 41 ∧ 
    (total_pages * words_per_page) % 251 = word_congruence ∧
    words_per_page ≤ max_words_per_page :=
by sorry

end NUMINAMATH_CALUDE_words_per_page_l1536_153611


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l1536_153686

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def geometric_subsequence (a : ℕ → ℝ) (k : ℕ → ℕ) (q : ℝ) : Prop :=
  ∀ n, a (k (n + 1)) = a (k n) * q

def strictly_increasing (k : ℕ → ℕ) : Prop :=
  ∀ n, k n < k (n + 1)

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℝ) (d q : ℝ) (k : ℕ → ℕ) 
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_subsequence a k q)
  (h_incr : strictly_increasing k)
  (h_d_neq_0 : d ≠ 0)
  (h_k1 : k 1 = 1)
  (h_k2 : k 2 = 3)
  (h_k3 : k 3 = 8) :
  (a 1 / d = 4 / 3) ∧ 
  ((∀ n, k (n + 1) = k n * q) ↔ a 1 / d = 1) ∧
  ((∀ n, k (n + 1) = k n * q) → 
   (∀ n : ℕ, 0 < n → a n + a (k n) > 2 * k n) → 
   a 1 ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l1536_153686


namespace NUMINAMATH_CALUDE_shannon_stones_l1536_153631

/-- The number of heart-shaped stones Shannon wants in each bracelet -/
def stones_per_bracelet : ℝ := 8.0

/-- The number of bracelets Shannon can make -/
def number_of_bracelets : ℕ := 6

/-- The total number of heart-shaped stones Shannon brought -/
def total_stones : ℝ := stones_per_bracelet * (number_of_bracelets : ℝ)

theorem shannon_stones :
  total_stones = 48.0 := by sorry

end NUMINAMATH_CALUDE_shannon_stones_l1536_153631


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1536_153679

theorem imaginary_part_of_z (z : ℂ) (h : z * Complex.I = 2 + Complex.I) :
  Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1536_153679


namespace NUMINAMATH_CALUDE_probability_between_C_and_D_l1536_153670

/-- Given points A, B, C, D on a line segment AB where AB = 4AD and AB = 5BC,
    prove that the probability of a randomly selected point on AB
    being between C and D is 11/20. -/
theorem probability_between_C_and_D (A B C D : ℝ) : 
  A < C ∧ C < D ∧ D < B →  -- Points are in order on the line
  (B - A) = 4 * (D - A) →  -- AB = 4AD
  (B - A) = 5 * (C - B) →  -- AB = 5BC
  (D - C) / (B - A) = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_between_C_and_D_l1536_153670


namespace NUMINAMATH_CALUDE_min_sugar_amount_l1536_153635

theorem min_sugar_amount (f s : ℕ) : 
  (f ≥ 9 + s / 2) → 
  (f ≤ 3 * s) → 
  (∃ (f : ℕ), f ≥ 9 + s / 2 ∧ f ≤ 3 * s) → 
  s ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_sugar_amount_l1536_153635


namespace NUMINAMATH_CALUDE_cat_walking_distance_l1536_153615

/-- The distance a cat walks given resistance time, walking rate, and total time -/
theorem cat_walking_distance (resistance_time walking_rate total_time : ℕ) : 
  resistance_time = 20 →
  walking_rate = 8 →
  total_time = 28 →
  (total_time - resistance_time) * walking_rate = 64 := by
  sorry

end NUMINAMATH_CALUDE_cat_walking_distance_l1536_153615


namespace NUMINAMATH_CALUDE_solution_is_correct_l1536_153655

def is_valid_triple (a m n : ℕ) : Prop :=
  a ≥ 2 ∧ m ≥ 2 ∧ ∃ k : ℕ, a^n + 203 = k * (a^m + 1)

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(a, m, n) | 
    (∃ k : ℕ, a = 2 ∧ m = 2 ∧ n = 4*k + 1) ∨
    (∃ k : ℕ, a = 2 ∧ m = 3 ∧ n = 6*k + 2) ∨
    (∃ k : ℕ, a = 2 ∧ m = 4 ∧ n = 8*k + 8) ∨
    (∃ k : ℕ, a = 2 ∧ m = 6 ∧ n = 12*k + 9) ∨
    (∃ k : ℕ, a = 3 ∧ m = 2 ∧ n = 4*k + 3) ∨
    (∃ k : ℕ, a = 4 ∧ m = 2 ∧ n = 4*k + 4) ∨
    (∃ k : ℕ, a = 5 ∧ m = 2 ∧ n = 4*k + 1) ∨
    (∃ k : ℕ, a = 8 ∧ m = 2 ∧ n = 4*k + 3) ∨
    (∃ k : ℕ, a = 10 ∧ m = 2 ∧ n = 4*k + 2) ∨
    (∃ k m : ℕ, a = 203 ∧ n = (2*k + 1)*m + 1)}

theorem solution_is_correct :
  ∀ a m n : ℕ, is_valid_triple a m n ↔ (a, m, n) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_solution_is_correct_l1536_153655


namespace NUMINAMATH_CALUDE_sandy_payment_l1536_153619

def amount_paid (football_cost baseball_cost change : ℚ) : ℚ :=
  football_cost + baseball_cost + change

theorem sandy_payment (football_cost baseball_cost change : ℚ) 
  (h1 : football_cost = 9.14)
  (h2 : baseball_cost = 6.81)
  (h3 : change = 4.05) :
  amount_paid football_cost baseball_cost change = 20 :=
by sorry

end NUMINAMATH_CALUDE_sandy_payment_l1536_153619


namespace NUMINAMATH_CALUDE_garden_length_is_fifty_l1536_153613

/-- Represents a rectangular garden with a given width and length. -/
structure RectangularGarden where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangular garden. -/
def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.width + g.length)

/-- Theorem: A rectangular garden with length twice its width and perimeter 150 yards has a length of 50 yards. -/
theorem garden_length_is_fifty {g : RectangularGarden} 
  (h1 : g.length = 2 * g.width) 
  (h2 : perimeter g = 150) : 
  g.length = 50 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_is_fifty_l1536_153613


namespace NUMINAMATH_CALUDE_vacation_pictures_count_l1536_153652

def zoo_pictures : ℕ := 150
def aquarium_pictures : ℕ := 210
def museum_pictures : ℕ := 90
def amusement_park_pictures : ℕ := 120

def zoo_deletion_percentage : ℚ := 25 / 100
def aquarium_deletion_percentage : ℚ := 15 / 100
def amusement_park_deletion : ℕ := 20
def museum_addition : ℕ := 30

theorem vacation_pictures_count :
  ⌊(zoo_pictures : ℚ) * (1 - zoo_deletion_percentage)⌋ +
  ⌊(aquarium_pictures : ℚ) * (1 - aquarium_deletion_percentage)⌋ +
  (museum_pictures + museum_addition) +
  (amusement_park_pictures - amusement_park_deletion) = 512 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_count_l1536_153652


namespace NUMINAMATH_CALUDE_right_triangle_other_leg_l1536_153607

theorem right_triangle_other_leg 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- right triangle condition
  (h_a : a = 9)                -- one leg is 9 cm
  (h_c : c = 15)               -- hypotenuse is 15 cm
  : b = 12 := by               -- prove other leg is 12 cm
  sorry

end NUMINAMATH_CALUDE_right_triangle_other_leg_l1536_153607


namespace NUMINAMATH_CALUDE_athlete_subgrid_exists_l1536_153666

/-- Represents a grid of athletes -/
def AthleteGrid := Fin 5 → Fin 49 → Bool

/-- Theorem: In any 5x49 grid of athletes, there exists a 3x3 subgrid of the same gender -/
theorem athlete_subgrid_exists (grid : AthleteGrid) :
  ∃ (i j : Fin 3) (r c : Fin 5),
    (∀ x y, x < i → y < j → grid (r + x) (c + y) = grid r c) :=
  sorry

end NUMINAMATH_CALUDE_athlete_subgrid_exists_l1536_153666


namespace NUMINAMATH_CALUDE_extreme_values_and_max_l1536_153678

/-- The function f(x) with parameters a, b, and c -/
def f (a b c x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

/-- The derivative of f(x) with respect to x -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_max (a b c : ℝ) :
  (f' a b 1 = 0 ∧ f' a b 2 = 0) →
  (a = -3 ∧ b = 4) ∧
  (c = -2 → ∀ x ∈ Set.Icc 0 3, f a b c x ≤ -7) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_and_max_l1536_153678


namespace NUMINAMATH_CALUDE_prime_factors_of_factorial_30_l1536_153696

theorem prime_factors_of_factorial_30 : 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_of_factorial_30_l1536_153696


namespace NUMINAMATH_CALUDE_decreasing_power_function_l1536_153668

theorem decreasing_power_function (m : ℝ) : 
  (m^2 - m - 1 > 0) ∧ (m^2 - 2*m - 3 < 0) ↔ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_power_function_l1536_153668


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l1536_153677

/-- Given a constant ratio between (2x - 5) and (y + 20), and the condition that y = 6 when x = 7,
    prove that x = 499/52 when y = 21 -/
theorem constant_ratio_problem (k : ℚ) :
  (∀ x y : ℚ, (2 * x - 5) / (y + 20) = k) →
  ((2 * 7 - 5) / (6 + 20) = k) →
  ∃ x : ℚ, (2 * x - 5) / (21 + 20) = k ∧ x = 499 / 52 :=
by sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l1536_153677


namespace NUMINAMATH_CALUDE_fish_in_pond_l1536_153646

/-- Approximates the total number of fish in a pond based on a tag-and-recapture method. -/
def approximate_fish_count (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) : ℕ :=
  (initial_tagged * second_catch) / tagged_in_second

/-- The approximate number of fish in the pond given the tag-and-recapture data. -/
theorem fish_in_pond (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ)
    (h1 : initial_tagged = 50)
    (h2 : second_catch = 50)
    (h3 : tagged_in_second = 10) :
  approximate_fish_count initial_tagged second_catch tagged_in_second = 250 := by
  sorry

#eval approximate_fish_count 50 50 10

end NUMINAMATH_CALUDE_fish_in_pond_l1536_153646


namespace NUMINAMATH_CALUDE_train_length_l1536_153621

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 9 → ∃ length : ℝ, abs (length - 299.97) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1536_153621


namespace NUMINAMATH_CALUDE_units_digit_of_difference_is_seven_l1536_153649

-- Define a three-digit number
def ThreeDigitNumber (a b c : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

-- Define the relationship between hundreds and units digits
def HundredsUnitsRelation (a c : ℕ) : Prop :=
  a = c - 3

-- Define the original number
def OriginalNumber (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

-- Define the reversed number
def ReversedNumber (a b c : ℕ) : ℕ :=
  100 * c + 10 * b + a

-- Theorem: The units digit of the difference is 7
theorem units_digit_of_difference_is_seven 
  (a b c : ℕ) 
  (h1 : ThreeDigitNumber a b c) 
  (h2 : HundredsUnitsRelation a c) : 
  (OriginalNumber a b c - ReversedNumber a b c) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_difference_is_seven_l1536_153649


namespace NUMINAMATH_CALUDE_rectangle_length_l1536_153606

theorem rectangle_length (P b l A : ℝ) : 
  P / b = 5 → 
  A = 216 → 
  P = 2 * (l + b) → 
  A = l * b → 
  l = 18 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_l1536_153606


namespace NUMINAMATH_CALUDE_jessica_quarters_problem_l1536_153650

/-- The number of quarters Jessica's sister gave her -/
def quarters_given (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem jessica_quarters_problem (initial : ℕ) (final : ℕ) 
  (h1 : initial = 8) 
  (h2 : final = 11) : 
  quarters_given initial final = 3 := by
  sorry

end NUMINAMATH_CALUDE_jessica_quarters_problem_l1536_153650


namespace NUMINAMATH_CALUDE_fifth_number_correct_l1536_153682

/-- The function that generates the 5th number on the n-th row of the array -/
def fifthNumber (n : ℕ) : ℚ :=
  (n - 1) * (n - 2) * (n - 3) * (3 * n + 8) / 24

/-- The theorem stating that for n > 5, the 5th number on the n-th row
    of the given array is equal to (n-1)(n-2)(n-3)(3n + 8) / 24 -/
theorem fifth_number_correct (n : ℕ) (h : n > 5) :
  fifthNumber n = (n - 1) * (n - 2) * (n - 3) * (3 * n + 8) / 24 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_correct_l1536_153682


namespace NUMINAMATH_CALUDE_intersection_points_l1536_153636

/-- A periodic function with period 2 that equals x^2 on [-1, 1] -/
noncomputable def f : ℝ → ℝ := sorry

/-- The number of intersection points between f and |log₅(x)| -/
def num_intersections : ℕ := sorry

theorem intersection_points :
  (∀ x, f (x + 2) = f x) ∧ 
  (∀ x ∈ Set.Icc (-1) 1, f x = x^2) →
  num_intersections = 5 := by sorry

end NUMINAMATH_CALUDE_intersection_points_l1536_153636


namespace NUMINAMATH_CALUDE_max_gcd_bn_l1536_153691

def b (n : ℕ) : ℚ := (15^n - 1) / 14

theorem max_gcd_bn (n : ℕ) : Nat.gcd (Nat.floor (b n)) (Nat.floor (b (n + 1))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_bn_l1536_153691


namespace NUMINAMATH_CALUDE_passengers_after_first_stop_l1536_153687

/-- 
Given a train with an initial number of passengers and some passengers getting off at the first stop,
this theorem proves the number of passengers remaining after the first stop.
-/
theorem passengers_after_first_stop 
  (initial_passengers : ℕ) 
  (passengers_left : ℕ) 
  (h1 : initial_passengers = 48)
  (h2 : passengers_left = initial_passengers - 17) : 
  passengers_left = 31 := by
  sorry

end NUMINAMATH_CALUDE_passengers_after_first_stop_l1536_153687


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1536_153695

/-- The sum of the infinite series Σ(n=1 to ∞) [(3n - 2) / (n(n+1)(n+3))] is equal to 2/21 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * n - 2) / (n * (n + 1) * (n + 3))) = 2 / 21 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1536_153695

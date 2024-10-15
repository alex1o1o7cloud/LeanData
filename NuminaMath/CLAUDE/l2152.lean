import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l2152_215267

theorem problem_statement (a b m : ℝ) : 
  2^a = m ∧ 3^b = m ∧ a * b ≠ 0 ∧ 
  ∃ (k : ℝ), a + k = a * b ∧ a * b + k = b → 
  m = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2152_215267


namespace NUMINAMATH_CALUDE_total_scholarship_amount_l2152_215248

-- Define the scholarship amounts
def wendy_scholarship : ℕ := 20000
def kelly_scholarship : ℕ := 2 * wendy_scholarship
def nina_scholarship : ℕ := kelly_scholarship - 8000

-- Theorem statement
theorem total_scholarship_amount :
  wendy_scholarship + kelly_scholarship + nina_scholarship = 92000 := by
  sorry

end NUMINAMATH_CALUDE_total_scholarship_amount_l2152_215248


namespace NUMINAMATH_CALUDE_cousins_age_sum_l2152_215263

theorem cousins_age_sum : ∀ (a b c : ℕ),
  a < 10 ∧ b < 10 ∧ c < 10 →  -- single-digit positive integers
  a ≠ b ∧ b ≠ c ∧ a ≠ c →     -- distinct
  (a < c ∧ b < c) →           -- one cousin is older than the other two
  a * b = 18 →                -- product of younger two
  c * min a b = 28 →          -- product of oldest and youngest
  a + b + c = 18 :=           -- sum of all three
by sorry

end NUMINAMATH_CALUDE_cousins_age_sum_l2152_215263


namespace NUMINAMATH_CALUDE_value_of_t_l2152_215220

-- Define variables
variable (p j t : ℝ)

-- Define the conditions
def condition1 : Prop := j = 0.75 * p
def condition2 : Prop := j = 0.80 * t
def condition3 : Prop := t = p * (1 - t / 100)

-- Theorem statement
theorem value_of_t (h1 : condition1 p j) (h2 : condition2 j t) (h3 : condition3 p t) : t = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_value_of_t_l2152_215220


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2152_215258

/-- Two vectors in R² -/
def Vector2 := Fin 2 → ℝ

/-- Dot product of two vectors in R² -/
def dot_product (v w : Vector2) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

/-- First direction vector -/
def v1 : Vector2 := ![- 6, 2]

/-- Second direction vector -/
def v2 (b : ℝ) : Vector2 := ![b, 3]

/-- Theorem: The value of b that makes the vectors perpendicular is 1 -/
theorem perpendicular_vectors : 
  ∃ b : ℝ, dot_product v1 (v2 b) = 0 ∧ b = 1 := by
sorry


end NUMINAMATH_CALUDE_perpendicular_vectors_l2152_215258


namespace NUMINAMATH_CALUDE_annual_growth_rate_for_doubling_l2152_215240

theorem annual_growth_rate_for_doubling (x : ℝ) (y : ℝ) (h : x > 0) :
  x * (1 + y)^2 = 2*x → y = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_annual_growth_rate_for_doubling_l2152_215240


namespace NUMINAMATH_CALUDE_union_of_sets_l2152_215257

def A (m : ℝ) : Set ℝ := {2, 2^m}
def B (m n : ℝ) : Set ℝ := {m, n}

theorem union_of_sets (m n : ℝ) (h : A m ∩ B m n = {1/4}) : 
  A m ∪ B m n = {2, -2, 1/4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2152_215257


namespace NUMINAMATH_CALUDE_small_triangles_to_large_triangle_area_ratio_l2152_215231

theorem small_triangles_to_large_triangle_area_ratio :
  let small_side_length : ℝ := 2
  let small_triangle_count : ℕ := 8
  let small_triangle_perimeter : ℝ := 3 * small_side_length
  let large_triangle_perimeter : ℝ := small_triangle_count * small_triangle_perimeter
  let large_side_length : ℝ := large_triangle_perimeter / 3
  let triangle_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2
  let small_triangle_area : ℝ := triangle_area small_side_length
  let large_triangle_area : ℝ := triangle_area large_side_length
  (small_triangle_count * small_triangle_area) / large_triangle_area = 1 / 8 := by
  sorry

#check small_triangles_to_large_triangle_area_ratio

end NUMINAMATH_CALUDE_small_triangles_to_large_triangle_area_ratio_l2152_215231


namespace NUMINAMATH_CALUDE_book_price_increase_l2152_215218

theorem book_price_increase (original_price : ℝ) (new_price : ℝ) (increase_percentage : ℝ) : 
  new_price = 330 ∧ 
  increase_percentage = 10 ∧ 
  new_price = original_price * (1 + increase_percentage / 100) →
  original_price = 300 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l2152_215218


namespace NUMINAMATH_CALUDE_random_events_identification_l2152_215281

-- Define the type for events
inductive Event : Type
  | draw_glasses : Event
  | guess_digit : Event
  | electric_charges : Event
  | lottery_win : Event

-- Define what it means for an event to be random
def is_random_event (e : Event) : Prop :=
  ∀ (outcome : Prop), ¬(outcome ∧ ¬outcome)

-- State the theorem
theorem random_events_identification :
  (is_random_event Event.draw_glasses) ∧
  (is_random_event Event.guess_digit) ∧
  (is_random_event Event.lottery_win) ∧
  (¬is_random_event Event.electric_charges) := by
  sorry

end NUMINAMATH_CALUDE_random_events_identification_l2152_215281


namespace NUMINAMATH_CALUDE_arithmetic_mean_neg6_to_8_l2152_215269

def arithmetic_mean (a b : Int) : ℚ :=
  let n := b - a + 1
  let sum := (n * (a + b)) / 2
  sum / n

theorem arithmetic_mean_neg6_to_8 :
  arithmetic_mean (-6) 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_neg6_to_8_l2152_215269


namespace NUMINAMATH_CALUDE_hiker_distance_l2152_215266

/-- Hiker's walking problem -/
theorem hiker_distance 
  (x y : ℝ) 
  (h1 : x * y = 18) 
  (D2 : ℝ := (y - 1) * (x + 1))
  (D3 : ℝ := 5 * 3)
  (D_total : ℝ := 18 + D2 + D3)
  (T_total : ℝ := y + (y - 1) + 3)
  (Z : ℝ)
  (h2 : Z = D_total / T_total) :
  D_total = x * y + y - x + 32 := by
sorry

end NUMINAMATH_CALUDE_hiker_distance_l2152_215266


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l2152_215254

/-- Proves the number of boys in a class given certain height information -/
theorem number_of_boys_in_class 
  (initial_average : ℝ)
  (wrong_height : ℝ)
  (actual_height : ℝ)
  (actual_average : ℝ)
  (h1 : initial_average = 183)
  (h2 : wrong_height = 166)
  (h3 : actual_height = 106)
  (h4 : actual_average = 181) :
  ∃ n : ℕ, n * initial_average - (wrong_height - actual_height) = n * actual_average ∧ n = 30 :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_l2152_215254


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2152_215204

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) ↔ a ∈ Set.Ioi 3 ∪ Set.Iio (-1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2152_215204


namespace NUMINAMATH_CALUDE_mushroom_pickers_l2152_215280

theorem mushroom_pickers (n : ℕ) (A V S R : ℚ) : 
  (∀ i : Fin n, i.val ≠ 0 ∧ i.val ≠ 1 ∧ i.val ≠ 2 → A / 2 = V + A / 2) →  -- Condition 1
  (S + A = R + V + A) →                                                   -- Condition 2
  (A > 0) →                                                               -- Anya has mushrooms
  (n > 3) →                                                               -- At least 4 children
  (n : ℚ) * (A / 2) = A + V + S + R →                                     -- Total mushrooms
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_mushroom_pickers_l2152_215280


namespace NUMINAMATH_CALUDE_vector_inequality_iff_positive_dot_product_l2152_215296

variable (n : ℕ)
variable (a b : Fin n → ℝ)

theorem vector_inequality_iff_positive_dot_product :
  ‖a + b‖ > ‖a - b‖ ↔ a • b > 0 := by sorry

end NUMINAMATH_CALUDE_vector_inequality_iff_positive_dot_product_l2152_215296


namespace NUMINAMATH_CALUDE_complex_number_problem_l2152_215274

theorem complex_number_problem (z : ℂ) (m n : ℝ) :
  (Complex.abs z = 2 * Real.sqrt 10) →
  (Complex.im ((3 - Complex.I) * z) = 0) →
  (Complex.re z < 0) →
  (2 * z^2 + m * z - n = 0) →
  (∃ (a b : ℝ), z = Complex.mk a b ∧ ((a = 2 ∧ b = -6) ∨ (a = -2 ∧ b = 6))) ∧
  (m + n = -72) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2152_215274


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2152_215230

theorem polynomial_divisibility (A B : ℝ) : 
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^205 + A*x + B = 0) → 
  A + B = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2152_215230


namespace NUMINAMATH_CALUDE_tims_change_l2152_215210

/-- The amount of change Tim will get after buying a candy bar -/
def change (initial_amount : ℕ) (price : ℕ) : ℕ :=
  initial_amount - price

/-- Theorem: Tim's change is 5 cents -/
theorem tims_change : change 50 45 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tims_change_l2152_215210


namespace NUMINAMATH_CALUDE_largest_divisible_by_eight_l2152_215285

theorem largest_divisible_by_eight (A B C : ℕ) : 
  A = 8 * B + C → 
  B = C → 
  C < 8 → 
  (∃ k : ℕ, A = 8 * k) → 
  A ≤ 63 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_by_eight_l2152_215285


namespace NUMINAMATH_CALUDE_boxes_with_neither_l2152_215211

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (crayons : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : markers = 8)
  (h3 : crayons = 5)
  (h4 : both = 3) :
  total - (markers + crayons - both) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l2152_215211


namespace NUMINAMATH_CALUDE_arnold_jellybean_count_l2152_215288

/-- Given the following conditions about jellybean counts:
  - Tino has 24 more jellybeans than Lee
  - Arnold has half as many jellybeans as Lee
  - Tino has 34 jellybeans
Prove that Arnold has 5 jellybeans. -/
theorem arnold_jellybean_count (tino lee arnold : ℕ) : 
  tino = lee + 24 →
  arnold = lee / 2 →
  tino = 34 →
  arnold = 5 := by
  sorry

end NUMINAMATH_CALUDE_arnold_jellybean_count_l2152_215288


namespace NUMINAMATH_CALUDE_door_blocked_time_l2152_215255

/-- Represents a clock with a door near its center -/
structure Clock :=
  (door_blocked_by_minute_hand : ℕ → Bool)
  (door_blocked_by_hour_hand : ℕ → Bool)

/-- The duration of a day in minutes -/
def day_minutes : ℕ := 24 * 60

/-- Checks if the door is blocked at a given minute -/
def is_door_blocked (clock : Clock) (minute : ℕ) : Bool :=
  clock.door_blocked_by_minute_hand minute ∨ clock.door_blocked_by_hour_hand minute

/-- Counts the number of minutes the door is blocked in a day -/
def blocked_minutes (clock : Clock) : ℕ :=
  (List.range day_minutes).filter (is_door_blocked clock) |>.length

/-- The theorem stating that the door is blocked for 498 minutes per day -/
theorem door_blocked_time (clock : Clock) 
  (h1 : ∀ (hour : ℕ) (minute : ℕ), hour < 24 → minute < 60 → 
    clock.door_blocked_by_minute_hand (hour * 60 + minute) = (9 ≤ minute ∧ minute < 21))
  (h2 : ∀ (minute : ℕ), minute < day_minutes → 
    clock.door_blocked_by_hour_hand minute = 
      ((108 ≤ minute % 720 ∧ minute % 720 < 252) ∨ 
       (828 ≤ minute % 720 ∧ minute % 720 < 972))) :
  blocked_minutes clock = 498 := by
  sorry


end NUMINAMATH_CALUDE_door_blocked_time_l2152_215255


namespace NUMINAMATH_CALUDE_coconut_oil_needed_l2152_215262

/-- Calculates the amount of coconut oil needed for baking brownies --/
theorem coconut_oil_needed
  (butter_per_cup : ℝ)
  (coconut_oil_per_cup : ℝ)
  (butter_available : ℝ)
  (total_baking_mix : ℝ)
  (h1 : butter_per_cup = 2)
  (h2 : coconut_oil_per_cup = 2)
  (h3 : butter_available = 4)
  (h4 : total_baking_mix = 6) :
  (total_baking_mix - butter_available / butter_per_cup) * coconut_oil_per_cup = 8 :=
by sorry

end NUMINAMATH_CALUDE_coconut_oil_needed_l2152_215262


namespace NUMINAMATH_CALUDE_pascals_theorem_l2152_215295

-- Define a circle
def Circle : Type := {p : ℝ × ℝ // (p.1^2 + p.2^2 = 1)}

-- Define a line
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

-- Define the intersection of two lines
def intersect (l1 l2 : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  l1 ∩ l2

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, q - p = t₁ • (r - p) ∧ r - p = t₂ • (q - p)

-- State Pascal's Theorem
theorem pascals_theorem 
  (A B C D E F : Circle) 
  (P : intersect (Line A.val B.val) (Line D.val E.val))
  (Q : intersect (Line B.val C.val) (Line E.val F.val))
  (R : intersect (Line C.val D.val) (Line F.val A.val)) :
  collinear P Q R :=
sorry

end NUMINAMATH_CALUDE_pascals_theorem_l2152_215295


namespace NUMINAMATH_CALUDE_harrison_elementary_students_l2152_215238

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

end NUMINAMATH_CALUDE_harrison_elementary_students_l2152_215238


namespace NUMINAMATH_CALUDE_third_term_of_sequence_l2152_215214

/-- Given a sequence {a_n} with S_n as the sum of the first n terms, and S_n = n^2 + n, prove a_3 = 6 -/
theorem third_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n^2 + n) : a 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_sequence_l2152_215214


namespace NUMINAMATH_CALUDE_max_area_rectangle_l2152_215290

/-- A rectangle with a given perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 20

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: The rectangle with maximum area among all rectangles with perimeter 40 is a square with sides 10 -/
theorem max_area_rectangle :
  ∀ r : Rectangle, area r ≤ area { length := 10, width := 10, perimeter_constraint := by norm_num } :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l2152_215290


namespace NUMINAMATH_CALUDE_bottle_caps_remaining_l2152_215219

theorem bottle_caps_remaining (initial : Nat) (removed : Nat) (h1 : initial = 16) (h2 : removed = 6) :
  initial - removed = 10 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_remaining_l2152_215219


namespace NUMINAMATH_CALUDE_problem_1_l2152_215271

theorem problem_1 (x : ℝ) : 4 * (x + 1)^2 = 49 → x = 5/2 ∨ x = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2152_215271


namespace NUMINAMATH_CALUDE_food_percentage_is_ten_percent_l2152_215283

-- Define the total amount spent
variable (T : ℝ)

-- Define the percentage spent on food
variable (F : ℝ)

-- Define the conditions
axiom clothing_percentage : 0.60 * T = T * 0.60
axiom other_items_percentage : 0.30 * T = T * 0.30
axiom food_percentage : F * T = T - (0.60 * T + 0.30 * T)

axiom tax_clothing : 0.04 * (0.60 * T) = 0.024 * T
axiom tax_other_items : 0.08 * (0.30 * T) = 0.024 * T
axiom total_tax : 0.048 * T = 0.024 * T + 0.024 * T

-- Theorem to prove
theorem food_percentage_is_ten_percent : F = 0.10 := by
  sorry

end NUMINAMATH_CALUDE_food_percentage_is_ten_percent_l2152_215283


namespace NUMINAMATH_CALUDE_first_box_contacts_l2152_215259

/-- Given two boxes of contacts, prove that the first box contains 75 contacts. -/
theorem first_box_contacts (price1 : ℚ) (quantity2 : ℕ) (price2 : ℚ) 
  (chosen_price : ℚ) (chosen_quantity : ℕ) :
  price1 = 25 →
  quantity2 = 99 →
  price2 = 33 →
  chosen_price = 1 →
  chosen_quantity = 3 →
  ∃ quantity1 : ℕ, quantity1 = 75 ∧ 
    price1 / quantity1 = min (price1 / quantity1) (price2 / quantity2) ∧
    price1 / quantity1 = chosen_price / chosen_quantity :=
by sorry


end NUMINAMATH_CALUDE_first_box_contacts_l2152_215259


namespace NUMINAMATH_CALUDE_salary_restoration_l2152_215278

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : original_salary > 0) : 
  let reduced_salary := original_salary * (1 - 0.2)
  let restoration_factor := reduced_salary * (1 + 0.25)
  restoration_factor = original_salary := by
sorry

end NUMINAMATH_CALUDE_salary_restoration_l2152_215278


namespace NUMINAMATH_CALUDE_point_coordinates_theorem_l2152_215253

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between a point and the y-axis -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- The distance between a point and the x-axis -/
def distanceToXAxis (p : Point) : ℝ := |p.y|

/-- Predicate to check if a point is left of the y-axis -/
def isLeftOfYAxis (p : Point) : Prop := p.x < 0

theorem point_coordinates_theorem (B : Point) 
  (h1 : isLeftOfYAxis B)
  (h2 : distanceToXAxis B = 4)
  (h3 : distanceToYAxis B = 5) :
  (B.x = -5 ∧ B.y = 4) ∨ (B.x = -5 ∧ B.y = -4) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_theorem_l2152_215253


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l2152_215209

def f (a : ℝ) (x : ℝ) : ℝ := 2 - |x + a|

theorem even_function_implies_a_zero :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l2152_215209


namespace NUMINAMATH_CALUDE_probability_point_in_circle_l2152_215247

/-- The probability of a randomly selected point in a square with side length 6 
    being within 2 units of the center is π/9 -/
theorem probability_point_in_circle (square_side : ℝ) (circle_radius : ℝ) : 
  square_side = 6 → 
  circle_radius = 2 → 
  (π * circle_radius^2) / (square_side^2) = π / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_circle_l2152_215247


namespace NUMINAMATH_CALUDE_pebble_throwing_difference_l2152_215227

/-- The number of pebbles Candy throws -/
def candy_pebbles : ℕ := 4

/-- The number of pebbles Lance throws -/
def lance_pebbles : ℕ := 3 * candy_pebbles

/-- The difference between Lance's pebbles and Candy's pebbles -/
def pebble_difference : ℕ := lance_pebbles - candy_pebbles

theorem pebble_throwing_difference :
  pebble_difference = 8 := by sorry

end NUMINAMATH_CALUDE_pebble_throwing_difference_l2152_215227


namespace NUMINAMATH_CALUDE_orchestra_only_females_l2152_215272

theorem orchestra_only_females (
  band_females : ℕ) (band_males : ℕ) 
  (orchestra_females : ℕ) (orchestra_males : ℕ)
  (both_females : ℕ) (both_males : ℕ)
  (total_students : ℕ) :
  band_females = 120 →
  band_males = 110 →
  orchestra_females = 100 →
  orchestra_males = 130 →
  both_females = 90 →
  both_males = 80 →
  total_students = 280 →
  total_students = band_females + band_males + orchestra_females + orchestra_males - both_females - both_males →
  orchestra_females - both_females = 10 := by
sorry

end NUMINAMATH_CALUDE_orchestra_only_females_l2152_215272


namespace NUMINAMATH_CALUDE_line_through_C_parallel_to_AB_area_of_triangle_OMN_l2152_215236

-- Define the points
def A : ℝ × ℝ := (1, 4)
def B : ℝ × ℝ := (3, 2)
def C : ℝ × ℝ := (1, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the perpendicular bisector
def perp_bisector (x y : ℝ) : Prop := x - y + 1 = 0

-- Define points M and N
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (0, 1)

-- Theorem for the line equation
theorem line_through_C_parallel_to_AB :
  line_equation C.1 C.2 ∧
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1) := by sorry

-- Theorem for the area of triangle OMN
theorem area_of_triangle_OMN :
  perp_bisector ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧
  M.2 = 0 ∧ N.1 = 0 ∧
  (1 / 2 : ℝ) * abs M.1 * abs N.2 = (1 / 2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_line_through_C_parallel_to_AB_area_of_triangle_OMN_l2152_215236


namespace NUMINAMATH_CALUDE_spelling_contest_result_l2152_215234

/-- In a spelling contest, given the following conditions:
  * There were 52 total questions
  * Drew got 20 questions correct
  * Drew got 6 questions wrong
  * Carla got twice as many questions wrong as Drew
Prove that Carla got 40 questions correct. -/
theorem spelling_contest_result (total_questions : Nat) (drew_correct : Nat) (drew_wrong : Nat) (carla_wrong_multiplier : Nat) :
  total_questions = 52 →
  drew_correct = 20 →
  drew_wrong = 6 →
  carla_wrong_multiplier = 2 →
  total_questions - (carla_wrong_multiplier * drew_wrong) = 40 := by
  sorry

#check spelling_contest_result

end NUMINAMATH_CALUDE_spelling_contest_result_l2152_215234


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l2152_215284

def double_factorial (n : ℕ) : ℕ := 
  if n ≤ 1 then 1 else n * double_factorial (n - 2)

theorem greatest_prime_factor_of_sum (n : ℕ) : 
  ∃ p : ℕ, Nat.Prime p ∧ 
    p = Nat.gcd (double_factorial 22 + double_factorial 20) p ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (double_factorial 22 + double_factorial 20) → q ≤ p :=
by
  -- The proof goes here
  sorry

#check greatest_prime_factor_of_sum

end NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l2152_215284


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2152_215265

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 3 - x) ↔ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2152_215265


namespace NUMINAMATH_CALUDE_jake_car_soap_cost_l2152_215200

/-- Represents the cost of car soap for Jake's car washing schedule -/
def car_soap_cost (washes_per_bottle : ℕ) (bottle_cost : ℚ) (total_washes : ℕ) : ℚ :=
  (total_washes / washes_per_bottle : ℚ) * bottle_cost

/-- Theorem: Jake spends $20.00 on car soap for washing his car once a week for 20 weeks -/
theorem jake_car_soap_cost :
  car_soap_cost 4 4 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jake_car_soap_cost_l2152_215200


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l2152_215277

theorem pentagon_largest_angle (P Q R S T : ℝ) : 
  P = 70 ∧ 
  Q = 100 ∧ 
  R = S ∧ 
  T = 2 * R + 20 ∧ 
  P + Q + R + S + T = 540 → 
  max P (max Q (max R (max S T))) = 195 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l2152_215277


namespace NUMINAMATH_CALUDE_symmetry_and_monotonicity_l2152_215235

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f (2 - x) = f x

def increasing_on_zero_one (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < 1 ∧ 0 < x₂ ∧ x₂ < 1 → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem symmetry_and_monotonicity
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_sym : symmetric_about_one f)
  (h_inc : increasing_on_zero_one f) :
  (∀ x, f (4 - x) + f x = 0) ∧
  (∀ x, 2 < x ∧ x < 3 → ∀ y, 2 < y ∧ y < 3 ∧ x < y → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_symmetry_and_monotonicity_l2152_215235


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l2152_215252

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the tangent line l
def tangent_line_l (x y : ℝ) : Prop := x = 2 ∨ 4*x - 3*y + 1 = 0

-- Theorem statement
theorem circle_and_tangent_line :
  -- Circle E passes through (0,0) and (1,1)
  circle_E 0 0 ∧ circle_E 1 1 ∧
  -- One of the three conditions is satisfied
  (circle_E 2 0 ∨ 
   (∀ m : ℝ, ∃ x y : ℝ, circle_E x y ∧ m*x - y - m = 0) ∨
   (∃ x : ℝ, circle_E x 0 ∧ x = 0)) →
  -- The tangent line passes through (2,3)
  (∃ x y : ℝ, circle_E x y ∧ tangent_line_l x y ∧
   ((x - 2)^2 + (y - 3)^2).sqrt = 1) :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l2152_215252


namespace NUMINAMATH_CALUDE_article_cost_l2152_215202

/-- Proves that the cost of an article is 80, given the specified conditions -/
theorem article_cost (original_profit_percent : Real) (reduced_cost_percent : Real)
  (price_reduction : Real) (new_profit_percent : Real)
  (h1 : original_profit_percent = 25)
  (h2 : reduced_cost_percent = 20)
  (h3 : price_reduction = 16.80)
  (h4 : new_profit_percent = 30) :
  ∃ (cost : Real), cost = 80 ∧
    (cost * (1 + original_profit_percent / 100) - price_reduction =
     (cost * (1 - reduced_cost_percent / 100)) * (1 + new_profit_percent / 100)) :=
by sorry

end NUMINAMATH_CALUDE_article_cost_l2152_215202


namespace NUMINAMATH_CALUDE_no_solution_for_x_equals_one_l2152_215275

theorem no_solution_for_x_equals_one (a : ℝ) (h : a ≠ 0) :
  ¬∃ x : ℝ, x = 1 ∧ a^2 * x^2 + (a + 1) * x + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_x_equals_one_l2152_215275


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l2152_215279

/-- A linear function passing through a quadrant if there exists a point (x, y) in that quadrant satisfying the function equation -/
def passes_through_quadrant (a b : ℝ) (quad : ℕ) : Prop :=
  ∃ x y : ℝ, 
    (quad = 1 → x > 0 ∧ y > 0) ∧
    (quad = 2 → x < 0 ∧ y > 0) ∧
    (quad = 3 → x < 0 ∧ y < 0) ∧
    (quad = 4 → x > 0 ∧ y < 0) ∧
    y = a * x + b

theorem linear_function_not_in_third_quadrant :
  ¬ passes_through_quadrant (-1/2) 1 3 :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l2152_215279


namespace NUMINAMATH_CALUDE_cricket_innings_problem_l2152_215242

theorem cricket_innings_problem (current_average : ℚ) (next_innings_runs : ℚ) (average_increase : ℚ) :
  current_average = 32 →
  next_innings_runs = 158 →
  average_increase = 6 →
  ∃ n : ℕ,
    n * current_average + next_innings_runs = (n + 1) * (current_average + average_increase) ∧
    n = 20 := by
  sorry

end NUMINAMATH_CALUDE_cricket_innings_problem_l2152_215242


namespace NUMINAMATH_CALUDE_remainder_seven_times_quotient_l2152_215232

theorem remainder_seven_times_quotient :
  {n : ℕ+ | ∃ q : ℕ, n = 23 * q + 7 * q ∧ 7 * q < 23} = {30, 60, 90} := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_times_quotient_l2152_215232


namespace NUMINAMATH_CALUDE_not_necessarily_equal_proportion_l2152_215221

theorem not_necessarily_equal_proportion (a b c d : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a * d = b * c) : 
  ¬(∀ a b c d, (a + 1) / b = (c + 1) / d) :=
by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_equal_proportion_l2152_215221


namespace NUMINAMATH_CALUDE_range_of_a_l2152_215224

def A (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ a}

def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 2 * x + 3}

def C (a : ℝ) : Set ℝ := {t | ∃ x ∈ A a, t = x^2}

theorem range_of_a (a : ℝ) (h1 : a ≥ -2) (h2 : C a ⊆ B a) : 
  a ∈ Set.Icc (1/2 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2152_215224


namespace NUMINAMATH_CALUDE_machine_production_difference_l2152_215208

/-- Proves that Machine B makes 20 more products than Machine A under given conditions -/
theorem machine_production_difference :
  ∀ (rate_A rate_B total_B : ℕ) (time : ℚ),
    rate_A = 8 →
    rate_B = 10 →
    total_B = 100 →
    time = total_B / rate_B →
    total_B - (rate_A * time) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_machine_production_difference_l2152_215208


namespace NUMINAMATH_CALUDE_range_of_H_l2152_215246

def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem range_of_H :
  Set.range H = {-4, 4} := by sorry

end NUMINAMATH_CALUDE_range_of_H_l2152_215246


namespace NUMINAMATH_CALUDE_students_liking_both_sports_l2152_215228

theorem students_liking_both_sports (basketball : ℕ) (cricket : ℕ) (total : ℕ) : 
  basketball = 12 → cricket = 8 → total = 17 → 
  basketball + cricket - total = 3 := by
sorry

end NUMINAMATH_CALUDE_students_liking_both_sports_l2152_215228


namespace NUMINAMATH_CALUDE_range_of_a_l2152_215237

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 0}
def B (a : ℝ) : Set ℝ := {1, 3, a}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a ≠ ∅ → a ∈ A := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2152_215237


namespace NUMINAMATH_CALUDE_unique_solution_equation_l2152_215212

theorem unique_solution_equation (x : ℝ) :
  x ≥ 0 →
  (2021 * x = 2022 * (x ^ (2021 / 2022)) - 1) ↔
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l2152_215212


namespace NUMINAMATH_CALUDE_focal_length_of_hyperbola_l2152_215226

-- Define the hyperbola C
def hyperbola (m : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 = 1

-- Define the asymptote
def asymptote (m : ℝ) (x y : ℝ) : Prop := Real.sqrt 3 * x + m * y = 0

-- Theorem statement
theorem focal_length_of_hyperbola (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x y, hyperbola m x y ↔ asymptote m x y) : 
  ∃ a b c : ℝ, a^2 = m ∧ b^2 = 1 ∧ c^2 = a^2 + b^2 ∧ 2 * c = 4 := by
  sorry

end NUMINAMATH_CALUDE_focal_length_of_hyperbola_l2152_215226


namespace NUMINAMATH_CALUDE_johns_numbers_l2152_215260

/-- Given a natural number, returns the number with its digits reversed -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is between 96 and 98 inclusive -/
def isBetween96And98 (n : ℕ) : Prop :=
  96 ≤ n ∧ n ≤ 98

/-- Represents the operation John performed on his number -/
def johnOperation (x : ℕ) : ℕ :=
  reverseDigits (4 * x + 17)

/-- A two-digit number satisfies John's conditions -/
def satisfiesConditions (x : ℕ) : Prop :=
  10 ≤ x ∧ x ≤ 99 ∧ isBetween96And98 (johnOperation x)

theorem johns_numbers :
  ∃ x y : ℕ, x ≠ y ∧ satisfiesConditions x ∧ satisfiesConditions y ∧
  (∀ z : ℕ, satisfiesConditions z → z = x ∨ z = y) ∧
  x = 13 ∧ y = 18 := by sorry

end NUMINAMATH_CALUDE_johns_numbers_l2152_215260


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l2152_215298

theorem circle_y_axis_intersection_sum :
  ∀ (y₁ y₂ : ℝ),
  (0 + 3)^2 + (y₁ - 5)^2 = 8^2 →
  (0 + 3)^2 + (y₂ - 5)^2 = 8^2 →
  y₁ ≠ y₂ →
  y₁ + y₂ = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l2152_215298


namespace NUMINAMATH_CALUDE_remainder_problem_l2152_215276

theorem remainder_problem (a b : ℕ) (h1 : a - b = 1311) (h2 : a / b = 11) (h3 : a = 1430) :
  a % b = 121 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2152_215276


namespace NUMINAMATH_CALUDE_age_difference_l2152_215294

/-- Proves that given a man and his son, where the son's present age is 24 and in two years
    the man's age will be twice his son's age, the difference between their present ages is 26 years. -/
theorem age_difference (man_age son_age : ℕ) : 
  son_age = 24 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2152_215294


namespace NUMINAMATH_CALUDE_sara_book_cost_l2152_215286

/-- The cost of Sara's first book -/
def first_book_cost : ℝ := sorry

/-- The cost of Sara's second book -/
def second_book_cost : ℝ := 6.5

/-- The amount Sara paid -/
def amount_paid : ℝ := 20

/-- The change Sara received -/
def change_received : ℝ := 8

theorem sara_book_cost : first_book_cost = 5.5 := by sorry

end NUMINAMATH_CALUDE_sara_book_cost_l2152_215286


namespace NUMINAMATH_CALUDE_pencil_cost_with_discount_cost_of_3000_pencils_l2152_215251

/-- The cost of pencils with a bulk discount -/
theorem pencil_cost_with_discount (base_quantity : ℕ) (base_cost : ℚ) 
  (order_quantity : ℕ) (discount_threshold : ℕ) (discount_rate : ℚ) : ℚ :=
  let base_price_per_pencil := base_cost / base_quantity
  let discounted_price_per_pencil := base_price_per_pencil * (1 - discount_rate)
  let total_cost := if order_quantity > discount_threshold
                    then order_quantity * discounted_price_per_pencil
                    else order_quantity * base_price_per_pencil
  total_cost

/-- The cost of 3000 pencils with the given conditions -/
theorem cost_of_3000_pencils : 
  pencil_cost_with_discount 150 40 3000 1000 (5/100) = 760 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_with_discount_cost_of_3000_pencils_l2152_215251


namespace NUMINAMATH_CALUDE_amara_clothing_proof_l2152_215256

def initial_clothing (donated_first : ℕ) (donated_second : ℕ) (thrown_away : ℕ) (remaining : ℕ) : ℕ :=
  remaining + donated_first + donated_second + thrown_away

theorem amara_clothing_proof :
  let donated_first := 5
  let donated_second := 3 * donated_first
  let thrown_away := 15
  let remaining := 65
  initial_clothing donated_first donated_second thrown_away remaining = 100 := by
  sorry

end NUMINAMATH_CALUDE_amara_clothing_proof_l2152_215256


namespace NUMINAMATH_CALUDE_riding_ratio_is_half_l2152_215203

/-- Represents the number of horses and men -/
def total_count : ℕ := 14

/-- Represents the number of legs walking on the ground -/
def legs_on_ground : ℕ := 70

/-- Represents the number of legs a horse has -/
def horse_legs : ℕ := 4

/-- Represents the number of legs a man has -/
def man_legs : ℕ := 2

/-- Represents the number of owners riding their horses -/
def riding_owners : ℕ := (total_count * horse_legs - legs_on_ground) / (horse_legs - man_legs)

/-- Represents the ratio of riding owners to total owners -/
def riding_ratio : ℚ := riding_owners / total_count

theorem riding_ratio_is_half : riding_ratio = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_riding_ratio_is_half_l2152_215203


namespace NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l2152_215213

/-- Pascal's triangle function -/
def pascal (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Predicate for a number being in Pascal's triangle -/
def inPascalTriangle (x : ℕ) : Prop := ∃ n k, pascal n k = x

/-- The set of four-digit numbers in Pascal's triangle -/
def fourDigitPascalNumbers : Set ℕ := {x | 1000 ≤ x ∧ x ≤ 9999 ∧ inPascalTriangle x}

/-- The third smallest element in a set of natural numbers -/
noncomputable def thirdSmallest (S : Set ℕ) : ℕ := sorry

theorem third_smallest_four_digit_pascal :
  thirdSmallest fourDigitPascalNumbers = 1002 := by sorry

end NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l2152_215213


namespace NUMINAMATH_CALUDE_wendy_facebook_pictures_l2152_215206

def total_pictures (one_album_pictures : ℕ) (num_other_albums : ℕ) (pictures_per_other_album : ℕ) : ℕ :=
  one_album_pictures + num_other_albums * pictures_per_other_album

theorem wendy_facebook_pictures :
  total_pictures 27 9 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_wendy_facebook_pictures_l2152_215206


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_cubes_l2152_215293

theorem sum_of_reciprocal_cubes (a b c : ℝ) 
  (sum_condition : a + b + c = 6)
  (prod_sum_condition : a * b + b * c + c * a = 5)
  (prod_condition : a * b * c = 1) :
  1 / a^3 + 1 / b^3 + 1 / c^3 = 128 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_cubes_l2152_215293


namespace NUMINAMATH_CALUDE_school_meeting_attendance_l2152_215207

/-- The number of parents at a school meeting -/
def num_parents : ℕ := 23

/-- The number of teachers at a school meeting -/
def num_teachers : ℕ := 8

/-- The total number of people at the school meeting -/
def total_people : ℕ := 31

/-- The number of parents who asked questions to the Latin teacher -/
def latin_teacher_parents : ℕ := 16

theorem school_meeting_attendance :
  (num_parents + num_teachers = total_people) ∧
  (num_parents = latin_teacher_parents + num_teachers - 1) ∧
  (∀ i : ℕ, i < num_teachers → latin_teacher_parents + i ≤ num_parents) ∧
  (latin_teacher_parents + num_teachers - 1 = num_parents) := by
  sorry

end NUMINAMATH_CALUDE_school_meeting_attendance_l2152_215207


namespace NUMINAMATH_CALUDE_dannys_collection_l2152_215264

theorem dannys_collection (initial_wrappers initial_caps found_wrappers found_caps : ℕ) 
  (h1 : initial_wrappers = 67)
  (h2 : initial_caps = 35)
  (h3 : found_wrappers = 18)
  (h4 : found_caps = 15) :
  (initial_wrappers + found_wrappers) - (initial_caps + found_caps) = 35 := by
  sorry

end NUMINAMATH_CALUDE_dannys_collection_l2152_215264


namespace NUMINAMATH_CALUDE_bananas_per_box_l2152_215291

theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) 
  (h1 : total_bananas = 40) (h2 : num_boxes = 8) : 
  total_bananas / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_bananas_per_box_l2152_215291


namespace NUMINAMATH_CALUDE_remainder_problem_l2152_215289

theorem remainder_problem (L S R : ℕ) (h1 : L - S = 1365) (h2 : S = 270) (h3 : L = 6 * S + R) : R = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2152_215289


namespace NUMINAMATH_CALUDE_incorrect_proposition_l2152_215205

theorem incorrect_proposition :
  ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_proposition_l2152_215205


namespace NUMINAMATH_CALUDE_amusement_park_average_cost_l2152_215222

/-- Represents the cost and trips data for a child's season pass -/
structure ChildData where
  pass_cost : ℕ
  trips : ℕ

/-- Calculates the average cost per trip given a list of ChildData -/
def average_cost_per_trip (children : List ChildData) : ℚ :=
  let total_cost := children.map (λ c => c.pass_cost) |>.sum
  let total_trips := children.map (λ c => c.trips) |>.sum
  (total_cost : ℚ) / total_trips

/-- The main theorem stating the average cost per trip for the given scenario -/
theorem amusement_park_average_cost :
  let children : List ChildData := [
    { pass_cost := 100, trips := 35 },
    { pass_cost := 90, trips := 25 },
    { pass_cost := 80, trips := 20 },
    { pass_cost := 70, trips := 15 }
  ]
  abs (average_cost_per_trip children - 3.58) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_amusement_park_average_cost_l2152_215222


namespace NUMINAMATH_CALUDE_table_sum_theorem_l2152_215201

/-- A 3x3 table filled with numbers from 1 to 9 -/
def Table := Fin 3 → Fin 3 → Fin 9

/-- The sum of elements along a diagonal -/
def diagonalSum (t : Table) (main : Bool) : Nat :=
  if main then t 0 0 + t 1 1 + t 2 2 else t 0 2 + t 1 1 + t 2 0

/-- The sum of elements in the specified cells -/
def specifiedSum (t : Table) : Nat :=
  t 1 0 + t 1 1 + t 1 2 + t 2 1 + t 2 2

/-- All numbers from 1 to 9 appear exactly once in the table -/
def isValid (t : Table) : Prop :=
  ∀ n : Fin 9, ∃! (i j : Fin 3), t i j = n

theorem table_sum_theorem (t : Table) (h_valid : isValid t) 
  (h_diag1 : diagonalSum t true = 7) (h_diag2 : diagonalSum t false = 21) :
  specifiedSum t = 25 := by
  sorry

end NUMINAMATH_CALUDE_table_sum_theorem_l2152_215201


namespace NUMINAMATH_CALUDE_max_intersections_8_6_l2152_215268

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating the maximum number of intersections for 8 x-axis points and 6 y-axis points -/
theorem max_intersections_8_6 :
  max_intersections 8 6 = 420 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_8_6_l2152_215268


namespace NUMINAMATH_CALUDE_no_functions_satisfy_condition_l2152_215244

theorem no_functions_satisfy_condition :
  ¬∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), f x * g y = x + y - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_functions_satisfy_condition_l2152_215244


namespace NUMINAMATH_CALUDE_mean_of_pencil_sharpening_counts_l2152_215273

def pencil_sharpening_counts : List ℕ := [13, 8, 13, 21, 7, 23]

theorem mean_of_pencil_sharpening_counts :
  (pencil_sharpening_counts.sum : ℚ) / pencil_sharpening_counts.length = 85/6 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_pencil_sharpening_counts_l2152_215273


namespace NUMINAMATH_CALUDE_total_paths_is_nine_l2152_215282

/-- A graph representing the paths between points A, B, C, and D -/
structure PathGraph where
  paths_AB : ℕ
  paths_BD : ℕ
  paths_DC : ℕ
  direct_AC : ℕ

/-- The total number of paths from A to C in the given graph -/
def total_paths (g : PathGraph) : ℕ :=
  g.paths_AB * g.paths_BD * g.paths_DC + g.direct_AC

/-- Theorem stating that the total number of paths from A to C is 9 -/
theorem total_paths_is_nine (g : PathGraph) 
  (h1 : g.paths_AB = 2)
  (h2 : g.paths_BD = 2)
  (h3 : g.paths_DC = 2)
  (h4 : g.direct_AC = 1) :
  total_paths g = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_paths_is_nine_l2152_215282


namespace NUMINAMATH_CALUDE_expand_product_l2152_215249

theorem expand_product (x : ℝ) (h : x ≠ 0) :
  (3 / 7) * ((7 / x^2) + 6 * x^3 - 2) = 3 / x^2 + (18 * x^3) / 7 - 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2152_215249


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l2152_215245

theorem modulus_of_complex_number (i : ℂ) (h : i * i = -1) :
  Complex.abs (2 + 1 / i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l2152_215245


namespace NUMINAMATH_CALUDE_minimum_questionnaires_to_mail_l2152_215297

theorem minimum_questionnaires_to_mail (response_rate : ℝ) (required_responses : ℕ) :
  response_rate = 0.62 →
  required_responses = 300 →
  ∃ n : ℕ, n ≥ (required_responses : ℝ) / response_rate ∧
    ∀ m : ℕ, m < n → (m : ℝ) * response_rate < required_responses := by
  sorry

#check minimum_questionnaires_to_mail

end NUMINAMATH_CALUDE_minimum_questionnaires_to_mail_l2152_215297


namespace NUMINAMATH_CALUDE_student_score_l2152_215225

-- Define the number of questions
def num_questions : ℕ := 5

-- Define the points per question
def points_per_question : ℕ := 20

-- Define the number of correct answers
def num_correct_answers : ℕ := 4

-- Theorem statement
theorem student_score (total_score : ℕ) :
  total_score = num_correct_answers * points_per_question :=
by sorry

end NUMINAMATH_CALUDE_student_score_l2152_215225


namespace NUMINAMATH_CALUDE_cone_volume_from_sphere_properties_l2152_215239

/-- Given a sphere and a cone with specific properties, prove that the volume of the cone is 12288π cm³ -/
theorem cone_volume_from_sphere_properties (r : ℝ) (h : ℝ) (S_sphere : ℝ) (S_cone : ℝ) (V_cone : ℝ) :
  r = 24 →
  h = 2 * r →
  S_sphere = 4 * π * r^2 →
  S_cone = S_sphere →
  V_cone = (1/3) * π * (S_cone / (2 * π * h))^2 * h →
  V_cone = 12288 * π := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_sphere_properties_l2152_215239


namespace NUMINAMATH_CALUDE_smallest_block_volume_l2152_215287

theorem smallest_block_volume (a b c : ℕ) : 
  (a - 1) * (b - 1) * (c - 1) = 252 → 
  a * b * c ≥ 392 ∧ 
  ∃ (a' b' c' : ℕ), (a' - 1) * (b' - 1) * (c' - 1) = 252 ∧ a' * b' * c' = 392 := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l2152_215287


namespace NUMINAMATH_CALUDE_smallest_product_of_factors_l2152_215241

def is_factor (n m : ℕ) : Prop := m % n = 0

theorem smallest_product_of_factors : 
  ∃ (a b : ℕ), 
    a ≠ b ∧ 
    a > 0 ∧ 
    b > 0 ∧ 
    is_factor a 60 ∧ 
    is_factor b 60 ∧ 
    ¬(is_factor (a * b) 60) ∧
    a * b = 8 ∧
    (∀ (c d : ℕ), 
      c ≠ d → 
      c > 0 → 
      d > 0 → 
      is_factor c 60 → 
      is_factor d 60 → 
      ¬(is_factor (c * d) 60) → 
      c * d ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_of_factors_l2152_215241


namespace NUMINAMATH_CALUDE_expression_evaluation_l2152_215270

theorem expression_evaluation : 
  let f (x : ℚ) := (2*x - 2) / (x + 2)
  let g (x : ℚ) := (2 * f x - 2) / (f x + 2)
  g 2 = -2/5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2152_215270


namespace NUMINAMATH_CALUDE_three_digit_base15_double_l2152_215250

/-- A function that converts a number from base 10 to base 15 --/
def toBase15 (n : ℕ) : ℕ :=
  (n / 100) * 15^2 + ((n / 10) % 10) * 15 + (n % 10)

/-- The set of three-digit numbers that satisfy the condition --/
def validNumbers : Finset ℕ := {150, 145, 290}

/-- The property that a number, when converted to base 15, is twice its original value --/
def satisfiesCondition (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ toBase15 n = 2 * n

theorem three_digit_base15_double :
  ∀ n : ℕ, satisfiesCondition n ↔ n ∈ validNumbers :=
sorry


end NUMINAMATH_CALUDE_three_digit_base15_double_l2152_215250


namespace NUMINAMATH_CALUDE_binomial_sum_l2152_215261

theorem binomial_sum (n : ℕ) (h : n > 0) : 
  Nat.choose n 1 + Nat.choose n (n - 2) = (n^2 + n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l2152_215261


namespace NUMINAMATH_CALUDE_cistern_empty_in_eight_minutes_l2152_215217

/-- Given a pipe that can empty 2/3 of a cistern in 10 minutes,
    this function calculates the part of the cistern that will be empty in a given number of minutes. -/
def cisternEmptyPart (emptyRate : Rat) (totalTime : Nat) (elapsedTime : Nat) : Rat :=
  (emptyRate / totalTime) * elapsedTime

/-- Theorem stating that given a pipe that can empty 2/3 of a cistern in 10 minutes,
    the part of the cistern that will be empty in 8 minutes is 8/15. -/
theorem cistern_empty_in_eight_minutes :
  cisternEmptyPart (2/3) 10 8 = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_cistern_empty_in_eight_minutes_l2152_215217


namespace NUMINAMATH_CALUDE_two_by_two_table_sum_l2152_215299

theorem two_by_two_table_sum (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a + b = c + d →
  a * c = b * d →
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_two_by_two_table_sum_l2152_215299


namespace NUMINAMATH_CALUDE_quadratic_root_to_coefficient_l2152_215243

theorem quadratic_root_to_coefficient (m : ℚ) : 
  (∀ x : ℂ, 6 * x^2 + 5 * x + m = 0 ↔ x = (-5 + Complex.I * Real.sqrt 231) / 12 ∨ x = (-5 - Complex.I * Real.sqrt 231) / 12) →
  m = 32 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_to_coefficient_l2152_215243


namespace NUMINAMATH_CALUDE_first_grade_muffins_l2152_215215

/-- The number of muffins baked by Mrs. Brier's class -/
def muffins_brier : ℕ := 18

/-- The number of muffins baked by Mrs. MacAdams's class -/
def muffins_macadams : ℕ := 20

/-- The number of muffins baked by Mrs. Flannery's class -/
def muffins_flannery : ℕ := 17

/-- The total number of muffins baked by first grade -/
def total_muffins : ℕ := muffins_brier + muffins_macadams + muffins_flannery

theorem first_grade_muffins : total_muffins = 55 := by
  sorry

end NUMINAMATH_CALUDE_first_grade_muffins_l2152_215215


namespace NUMINAMATH_CALUDE_f_difference_l2152_215216

/-- The function f(x) = 3x^2 + 5x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 4

/-- Theorem stating that f(x+h) - f(x) = h(6x + 3h + 5) for all real x and h -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h + 5) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l2152_215216


namespace NUMINAMATH_CALUDE_triangle_b_range_l2152_215229

open Real Set

-- Define the triangle and its properties
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions for the triangle
def TriangleConditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 3 ∧ t.A = π / 3

-- Define the condition for exactly one solution
def ExactlyOneSolution (t : Triangle) : Prop :=
  (t.a = t.b * Real.sin t.A) ∨ (t.a ≥ t.b ∧ t.a > t.b * Real.sin t.A)

-- Define the range of values for b
def BRange : Set ℝ := Ioc 0 (Real.sqrt 3) ∪ {2}

-- State the theorem
theorem triangle_b_range (t : Triangle) :
  TriangleConditions t → ExactlyOneSolution t → t.b ∈ BRange :=
by sorry

end NUMINAMATH_CALUDE_triangle_b_range_l2152_215229


namespace NUMINAMATH_CALUDE_tangent_to_parabola_l2152_215223

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the given line
def given_line (x y : ℝ) : Prop := 4 * x - y + 3 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 4 * x - y - 2 = 0

theorem tangent_to_parabola :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the parabola
    y₀ = parabola x₀ ∧
    -- The tangent line passes through (x₀, y₀)
    tangent_line x₀ y₀ ∧
    -- The tangent line is parallel to the given line
    (∀ (x y : ℝ), tangent_line x y ↔ ∃ (k : ℝ), y = 4 * x + k) ∧
    -- The tangent line touches the parabola at exactly one point
    (∀ (x y : ℝ), x ≠ x₀ → y = parabola x → ¬ tangent_line x y) :=
sorry

end NUMINAMATH_CALUDE_tangent_to_parabola_l2152_215223


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_l2152_215233

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.log (x + 1)

theorem tangent_line_perpendicular (n : ℝ) : 
  (∃ m : ℝ, (∀ x : ℝ, f x = m * x + f 0) ∧ 
   (m * (1 / n) = -1)) → n = -2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_l2152_215233


namespace NUMINAMATH_CALUDE_negation_equivalence_l2152_215292

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2152_215292

import Mathlib

namespace NUMINAMATH_CALUDE_find_largest_base_l1972_197288

theorem find_largest_base (x : ℤ) (base : ℕ) :
  (x ≤ 3) →
  (2.134 * (base : ℝ) ^ (x : ℝ) < 21000) →
  (∀ y : ℤ, y ≤ 3 → 2.134 * (base : ℝ) ^ (y : ℝ) < 21000) →
  base ≤ 21 :=
sorry

end NUMINAMATH_CALUDE_find_largest_base_l1972_197288


namespace NUMINAMATH_CALUDE_student_polynomial_correction_l1972_197281

/-- Given a polynomial P(x) that satisfies P(x) - 3x^2 = x^2 - 4x + 1,
    prove that P(x) * (-3x^2) = -12x^4 + 12x^3 - 3x^2 -/
theorem student_polynomial_correction (P : ℝ → ℝ) :
  (∀ x, P x - 3 * x^2 = x^2 - 4 * x + 1) →
  (∀ x, P x * (-3 * x^2) = -12 * x^4 + 12 * x^3 - 3 * x^2) :=
by sorry

end NUMINAMATH_CALUDE_student_polynomial_correction_l1972_197281


namespace NUMINAMATH_CALUDE_value_of_E_l1972_197257

theorem value_of_E (a b c : ℝ) (h1 : a ≠ b) (h2 : a^2 * (b + c) = 2023) (h3 : b^2 * (c + a) = 2023) :
  c^2 * (a + b) = 2023 := by
sorry

end NUMINAMATH_CALUDE_value_of_E_l1972_197257


namespace NUMINAMATH_CALUDE_conference_handshakes_l1972_197228

/-- The number of handshakes in a conference with multiple companies --/
def number_of_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_participants := num_companies * reps_per_company
  let handshakes_per_person := total_participants - reps_per_company
  (total_participants * handshakes_per_person) / 2

/-- Theorem: In a conference with 3 companies, each having 5 representatives,
    where every person shakes hands once with every person except those from their own company,
    the total number of handshakes is 75. --/
theorem conference_handshakes :
  number_of_handshakes 3 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1972_197228


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_5_and_6_l1972_197200

theorem smallest_common_multiple_of_5_and_6 : 
  ∃ n : ℕ, n > 0 ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, m > 0 → 5 ∣ m → 6 ∣ m → n ≤ m :=
by sorry

#check smallest_common_multiple_of_5_and_6

end NUMINAMATH_CALUDE_smallest_common_multiple_of_5_and_6_l1972_197200


namespace NUMINAMATH_CALUDE_fractional_equation_root_l1972_197278

theorem fractional_equation_root (x m : ℝ) : 
  (∃ x, x / (x - 3) - 2 = (m - 1) / (x - 3) ∧ x ≠ 3) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l1972_197278


namespace NUMINAMATH_CALUDE_john_mean_score_l1972_197247

def john_scores : List ℝ := [88, 92, 94, 86, 90, 85]

theorem john_mean_score :
  (john_scores.sum / john_scores.length : ℝ) = 535 / 6 := by
  sorry

end NUMINAMATH_CALUDE_john_mean_score_l1972_197247


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l1972_197227

theorem magnitude_of_complex_number : 
  Complex.abs ((1 + Complex.I)^2 / (1 - 2 * Complex.I)) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l1972_197227


namespace NUMINAMATH_CALUDE_function_properties_l1972_197225

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x

theorem function_properties :
  (∀ x ≠ 0, f x = x^2 + 1/x) →
  f 1 = 2 →
  (¬ (∀ x ≠ 0, f (-x) = f x) ∧ ¬ (∀ x ≠ 0, f (-x) = -f x)) ∧
  (∀ x y, 2 ≤ x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1972_197225


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l1972_197237

theorem initial_alcohol_percentage 
  (initial_volume : ℝ) 
  (added_alcohol : ℝ) 
  (added_water : ℝ) 
  (final_percentage : ℝ) :
  initial_volume = 40 →
  added_alcohol = 2.5 →
  added_water = 7.5 →
  final_percentage = 9 →
  ∃ (initial_percentage : ℝ),
    initial_percentage * initial_volume / 100 + added_alcohol = 
    final_percentage * (initial_volume + added_alcohol + added_water) / 100 ∧
    initial_percentage = 5 :=
by sorry

end NUMINAMATH_CALUDE_initial_alcohol_percentage_l1972_197237


namespace NUMINAMATH_CALUDE_swimming_speed_is_10_l1972_197248

/-- The swimming speed of a person in still water. -/
def swimming_speed : ℝ := 10

/-- The speed of the water current. -/
def water_speed : ℝ := 8

/-- The time taken to swim against the current. -/
def swim_time : ℝ := 8

/-- The distance swam against the current. -/
def swim_distance : ℝ := 16

/-- Theorem stating that the swimming speed in still water is 10 km/h given the conditions. -/
theorem swimming_speed_is_10 :
  swimming_speed = 10 ∧
  water_speed = 8 ∧
  swim_time = 8 ∧
  swim_distance = 16 ∧
  swim_distance = (swimming_speed - water_speed) * swim_time :=
by sorry

end NUMINAMATH_CALUDE_swimming_speed_is_10_l1972_197248


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_to_2011_l1972_197202

theorem last_four_digits_of_5_to_2011 : ∃ n : ℕ, 5^2011 ≡ 8125 [MOD 10000] :=
by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_to_2011_l1972_197202


namespace NUMINAMATH_CALUDE_pizza_slices_per_person_l1972_197284

theorem pizza_slices_per_person 
  (coworkers : ℕ) 
  (pizzas : ℕ) 
  (slices_per_pizza : ℕ) 
  (h1 : coworkers = 12)
  (h2 : pizzas = 3)
  (h3 : slices_per_pizza = 8)
  : (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_person_l1972_197284


namespace NUMINAMATH_CALUDE_largest_area_chord_construction_l1972_197291

/-- Represents a direction in 2D space -/
structure Direction where
  angle : Real

/-- Represents an ellipse -/
structure Ellipse where
  semi_major_axis : Real
  semi_minor_axis : Real

/-- Represents a chord of an ellipse -/
structure Chord where
  direction : Direction
  length : Real

/-- Represents a triangle -/
structure Triangle where
  base : Real
  height : Real

/-- Calculates the area of a triangle -/
def triangle_area (t : Triangle) : Real :=
  0.5 * t.base * t.height

/-- Finds the chord that forms the triangle with the largest area -/
def largest_area_chord (e : Ellipse) (i : Direction) : Chord :=
  sorry

/-- Theorem: The chord that forms the largest area triangle is constructed by 
    finding an intersection point and creating two mirrored right triangles -/
theorem largest_area_chord_construction (e : Ellipse) (i : Direction) :
  ∃ (c : Chord), c = largest_area_chord e i ∧
  ∃ (t1 t2 : Triangle), 
    triangle_area t1 = triangle_area t2 ∧
    t1.base = t2.base ∧
    t1.height = t2.height ∧
    t1.base * t1.base + t1.height * t1.height = c.length * c.length :=
  sorry

end NUMINAMATH_CALUDE_largest_area_chord_construction_l1972_197291


namespace NUMINAMATH_CALUDE_find_first_group_men_l1972_197216

/-- Represents the work rate of one person -/
structure WorkRate where
  rate : ℝ

/-- Represents a group of workers -/
structure WorkerGroup where
  men : ℕ
  women : ℕ

/-- Calculates the total work done by a group -/
def totalWork (m w : WorkRate) (g : WorkerGroup) : ℝ :=
  (g.men : ℝ) * m.rate + (g.women : ℝ) * w.rate

theorem find_first_group_men (m w : WorkRate) : ∃ x : ℕ, 
  totalWork m w ⟨x, 8⟩ = totalWork m w ⟨6, 2⟩ ∧
  2 * totalWork m w ⟨2, 3⟩ = totalWork m w ⟨x, 8⟩ ∧
  x = 3 := by
  sorry

#check find_first_group_men

end NUMINAMATH_CALUDE_find_first_group_men_l1972_197216


namespace NUMINAMATH_CALUDE_car_tractor_distance_theorem_l1972_197286

theorem car_tractor_distance_theorem (total_distance : ℝ) 
  (first_meeting_time : ℝ) (car_wait_time : ℝ) (car_catch_up_time : ℝ) :
  total_distance = 160 ∧ 
  first_meeting_time = 4/3 ∧ 
  car_wait_time = 1 ∧ 
  car_catch_up_time = 1/2 →
  ∃ (car_speed tractor_speed : ℝ),
    car_speed > 0 ∧ tractor_speed > 0 ∧
    car_speed + tractor_speed = total_distance / first_meeting_time ∧
    car_speed * (first_meeting_time + car_catch_up_time) = 165 ∧
    tractor_speed * (first_meeting_time + car_wait_time + car_catch_up_time) = 85 :=
by sorry

end NUMINAMATH_CALUDE_car_tractor_distance_theorem_l1972_197286


namespace NUMINAMATH_CALUDE_time_interval_for_population_change_l1972_197226

/-- Proves that given the specified birth and death rates and net population increase,
    the time interval is 2 seconds. -/
theorem time_interval_for_population_change (t : ℝ) : 
  (5 : ℝ) / t - (3 : ℝ) / t > 0 →  -- Ensure positive net change
  (5 - 3) * (86400 / t) = 86400 →
  t = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_interval_for_population_change_l1972_197226


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1972_197271

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1972_197271


namespace NUMINAMATH_CALUDE_mouse_jump_distance_l1972_197258

theorem mouse_jump_distance (grasshopper_jump frog_jump mouse_jump : ℕ) :
  grasshopper_jump = 25 →
  frog_jump = grasshopper_jump + 32 →
  mouse_jump = frog_jump - 26 →
  mouse_jump = 31 := by sorry

end NUMINAMATH_CALUDE_mouse_jump_distance_l1972_197258


namespace NUMINAMATH_CALUDE_warehouse_construction_l1972_197266

/-- Warehouse construction problem -/
theorem warehouse_construction (investment : ℝ) (front_cost side_cost top_cost : ℝ) 
  (h_investment : investment = 3200)
  (h_front_cost : front_cost = 40)
  (h_side_cost : side_cost = 45)
  (h_top_cost : top_cost = 20) :
  ∃ (x y : ℝ),
    0 < x ∧ x < 80 ∧
    y = (320 - 4*x) / (9 + 2*x) ∧
    x * y ≤ 100 ∧
    (∀ x' y' : ℝ, 0 < x' ∧ x' < 80 ∧ y' = (320 - 4*x') / (9 + 2*x') → x' * y' ≤ x * y) ∧
    x = 15 ∧ y = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_construction_l1972_197266


namespace NUMINAMATH_CALUDE_current_rate_l1972_197212

/-- The rate of the current given a man's rowing speed and time ratio -/
theorem current_rate (man_speed : ℝ) (time_ratio : ℝ) : 
  man_speed = 3.6 ∧ time_ratio = 2 → 
  ∃ c : ℝ, c = 1.2 ∧ (man_speed - c) / (man_speed + c) = 1 / time_ratio :=
by sorry

end NUMINAMATH_CALUDE_current_rate_l1972_197212


namespace NUMINAMATH_CALUDE_fourteenth_root_unity_l1972_197233

theorem fourteenth_root_unity (n : ℕ) : 
  0 ≤ n ∧ n ≤ 13 → 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * n * π / 14)) → 
  n = 4 := by sorry

end NUMINAMATH_CALUDE_fourteenth_root_unity_l1972_197233


namespace NUMINAMATH_CALUDE_simplify_expression_l1972_197218

theorem simplify_expression (a : ℝ) : a^4 * (-a)^3 = -a^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1972_197218


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l1972_197229

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 6 < 3 * ((7 * n + 21) / 7)) → 
  (∀ k : ℤ, k < n → k + 6 < 3 * ((7 * k + 21) / 7)) →
  n = -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l1972_197229


namespace NUMINAMATH_CALUDE_original_number_of_girls_l1972_197232

theorem original_number_of_girls (b g : ℚ) : 
  b > 0 ∧ g > 0 →  -- Initial numbers are positive
  3 * (g - 20) = b →  -- After 20 girls leave, ratio is 3 boys to 1 girl
  4 * (b - 60) = g - 20 →  -- After 60 boys leave, ratio is 1 boy to 4 girls
  g = 460 / 11 := by
sorry

end NUMINAMATH_CALUDE_original_number_of_girls_l1972_197232


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_five_l1972_197253

theorem sum_of_cubes_of_five : 5^3 + 5^3 + 5^3 + 5^3 = 625 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_five_l1972_197253


namespace NUMINAMATH_CALUDE_sam_seashells_l1972_197209

/-- Given that Sam found 35 seashells and gave 18 to Joan, prove that he now has 17 seashells. -/
theorem sam_seashells (initial : ℕ) (given_away : ℕ) (h1 : initial = 35) (h2 : given_away = 18) :
  initial - given_away = 17 := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l1972_197209


namespace NUMINAMATH_CALUDE_base_sum_theorem_l1972_197207

/-- Given two integer bases R₁ and R₂, if certain fractions have specific representations
    in these bases, then the sum of R₁ and R₂ is 21. -/
theorem base_sum_theorem (R₁ R₂ : ℕ) : 
  (R₁ > 1) → 
  (R₂ > 1) →
  ((4 * R₁ + 8) / (R₁^2 - 1) = (3 * R₂ + 6) / (R₂^2 - 1)) →
  ((8 * R₁ + 4) / (R₁^2 - 1) = (6 * R₂ + 3) / (R₂^2 - 1)) →
  R₁ + R₂ = 21 :=
by sorry

#check base_sum_theorem

end NUMINAMATH_CALUDE_base_sum_theorem_l1972_197207


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1972_197261

theorem sqrt_x_minus_one_real (x : ℝ) (h : x = 2) : ∃ y : ℝ, y ^ 2 = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1972_197261


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_two_l1972_197250

theorem sqrt_sum_equals_two (a b : ℝ) (h : a^2 + b^2 = 4) :
  (a * (b - 4))^(1/3) + ((a * b - 3 * a + 2 * b - 6) : ℝ)^(1/2) = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_two_l1972_197250


namespace NUMINAMATH_CALUDE_graph_vertical_shift_l1972_197292

-- Define a continuous function f on the real line
variable (f : ℝ → ℝ)
variable (h : Continuous f)

-- Define the vertical shift operation
def verticalShift (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f x + k

-- Theorem statement
theorem graph_vertical_shift :
  ∀ (x y : ℝ), y = f x + 2 ↔ y = (verticalShift f 2) x :=
by sorry

end NUMINAMATH_CALUDE_graph_vertical_shift_l1972_197292


namespace NUMINAMATH_CALUDE_square_product_existence_l1972_197295

theorem square_product_existence : ∃ n : ℕ+, (3150 : ℕ) * 14 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_product_existence_l1972_197295


namespace NUMINAMATH_CALUDE_root_equation_k_value_l1972_197285

theorem root_equation_k_value :
  ∀ k : ℝ, ((-2)^2 - k*(-2) + 2 = 0) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_k_value_l1972_197285


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l1972_197294

theorem intersection_in_fourth_quadrant (k : ℝ) :
  let line1 : ℝ → ℝ := λ x => -2 * x + 3 * k + 14
  let line2 : ℝ → ℝ := λ y => (3 * k + 2 + 4 * y) / 1
  let x := k + 6
  let y := k + 2
  (∀ x', line1 x' = line2 x') →
  (x > 0 ∧ y < 0) →
  -6 < k ∧ k < -2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l1972_197294


namespace NUMINAMATH_CALUDE_smallest_divisor_k_l1972_197276

def f (z : ℂ) : ℂ := z^10 + z^9 + z^8 + z^6 + z^5 + z^4 + z + 1

theorem smallest_divisor_k : 
  (∀ z : ℂ, f z = 0 → z^84 = 1) ∧ 
  (∀ k : ℕ, k < 84 → ∃ z : ℂ, f z = 0 ∧ z^k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisor_k_l1972_197276


namespace NUMINAMATH_CALUDE_room_width_calculation_l1972_197238

theorem room_width_calculation (length area : ℝ) (h1 : length = 12) (h2 : area = 96) :
  area / length = 8 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l1972_197238


namespace NUMINAMATH_CALUDE_circle_center_correct_l1972_197255

/-- The equation of a circle in the form x^2 - 2ax + y^2 - 2by + c = 0 -/
def CircleEquation (a b c : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 - 2*a*x + y^2 - 2*b*y + c = 0

/-- The center of a circle given by its equation -/
def CircleCenter (a b c : ℝ) : ℝ × ℝ := (a, b)

theorem circle_center_correct (x y : ℝ) :
  CircleEquation 1 2 (-28) x y → CircleCenter 1 2 (-28) = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_correct_l1972_197255


namespace NUMINAMATH_CALUDE_expression_simplification_l1972_197219

theorem expression_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ 4) :
  (x^2 - 4) * ((x + 2) / (x^2 - 2*x) - (x - 1) / (x^2 - 4*x + 4)) / ((x - 4) / x) = (x + 2) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1972_197219


namespace NUMINAMATH_CALUDE_selection_theorem_l1972_197279

/-- The number of ways to select k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of male students. -/
def num_males : ℕ := 4

/-- The number of female students. -/
def num_females : ℕ := 3

/-- The total number of students to be selected. -/
def num_selected : ℕ := 3

/-- The number of ways to select students with both genders represented. -/
def num_ways : ℕ := 
  choose num_males 2 * choose num_females 1 + 
  choose num_males 1 * choose num_females 2

theorem selection_theorem : num_ways = 30 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l1972_197279


namespace NUMINAMATH_CALUDE_remainder_8673_mod_7_l1972_197246

theorem remainder_8673_mod_7 : 8673 % 7 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_8673_mod_7_l1972_197246


namespace NUMINAMATH_CALUDE_x_minus_y_equals_eight_l1972_197215

theorem x_minus_y_equals_eight (x y : ℝ) (h1 : 4 = 0.25 * x) (h2 : 4 = 0.50 * y) : x - y = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_eight_l1972_197215


namespace NUMINAMATH_CALUDE_sailboat_sails_height_l1972_197206

theorem sailboat_sails_height (rectangular_length rectangular_width first_triangular_base second_triangular_base second_triangular_height total_canvas : ℝ) 
  (h1 : rectangular_length = 8)
  (h2 : rectangular_width = 5)
  (h3 : first_triangular_base = 3)
  (h4 : second_triangular_base = 4)
  (h5 : second_triangular_height = 6)
  (h6 : total_canvas = 58) :
  let rectangular_area := rectangular_length * rectangular_width
  let second_triangular_area := (second_triangular_base * second_triangular_height) / 2
  let first_triangular_area := total_canvas - rectangular_area - second_triangular_area
  first_triangular_area = (first_triangular_base * 4) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sailboat_sails_height_l1972_197206


namespace NUMINAMATH_CALUDE_distinguishable_triangles_count_l1972_197208

/-- Represents the number of available colors for the triangles -/
def num_colors : ℕ := 8

/-- Represents the number of triangles in the large equilateral triangle -/
def num_triangles : ℕ := 6

/-- Represents the number of corner triangles -/
def num_corners : ℕ := 3

/-- Represents the number of edge triangles -/
def num_edges : ℕ := 2

/-- Represents the number of center triangles -/
def num_center : ℕ := 1

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of distinguishable large equilateral triangles -/
def num_distinguishable_triangles : ℕ :=
  -- Corner configurations
  (num_colors + 
   num_colors * (num_colors - 1) + 
   choose num_colors num_corners) *
  -- Edge configurations
  (num_colors ^ num_edges) *
  -- Center configurations
  num_colors

theorem distinguishable_triangles_count :
  num_distinguishable_triangles = 61440 :=
sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_count_l1972_197208


namespace NUMINAMATH_CALUDE_distance_between_points_l1972_197254

theorem distance_between_points :
  let x₁ : ℝ := 2
  let y₁ : ℝ := 3
  let x₂ : ℝ := 5
  let y₂ : ℝ := 9
  let distance := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)
  distance = 3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l1972_197254


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1972_197289

theorem pure_imaginary_complex_number (x : ℝ) :
  (Complex.I * (x + 1) = (x^2 - 1) + Complex.I * (x + 1)) → (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1972_197289


namespace NUMINAMATH_CALUDE_equation_solution_l1972_197290

theorem equation_solution : ∃! (x : ℝ), x ≠ 0 ∧ (5*x)^20 = (20*x)^10 ∧ x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1972_197290


namespace NUMINAMATH_CALUDE_product_inequality_l1972_197241

theorem product_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  (1 + x) * (1 + y) * (1 + z) ≥ 2 * (1 + (y / x)^(1/3) + (z / y)^(1/3) + (x / z)^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1972_197241


namespace NUMINAMATH_CALUDE_magnitude_BC_l1972_197263

/-- Given two vectors BA and AC in R², prove that the magnitude of BC is 5. -/
theorem magnitude_BC (BA AC : ℝ × ℝ) (h1 : BA = (3, -2)) (h2 : AC = (0, 6)) : 
  ‖BA + AC‖ = 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_BC_l1972_197263


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1972_197280

theorem complex_equation_solution (a : ℝ) : 
  (Complex.mk 2 a) * (Complex.mk a (-2)) = Complex.I * (-4) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1972_197280


namespace NUMINAMATH_CALUDE_english_score_is_67_l1972_197217

def mathematics_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def biology_score : ℕ := 85
def average_marks : ℕ := 75
def total_subjects : ℕ := 5

def english_score : ℕ := average_marks * total_subjects - (mathematics_score + science_score + social_studies_score + biology_score)

theorem english_score_is_67 : english_score = 67 := by
  sorry

end NUMINAMATH_CALUDE_english_score_is_67_l1972_197217


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1972_197223

theorem quadratic_equation_roots (m : ℕ+) : 
  (∃ x : ℝ, x^2 - 2*x + 2*(m : ℝ) - 1 = 0) → 
  (m = 1 ∧ ∀ x : ℝ, x^2 - 2*x + 2*(m : ℝ) - 1 = 0 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1972_197223


namespace NUMINAMATH_CALUDE_total_interest_compound_linh_investment_interest_l1972_197265

/-- Calculate the total interest earned on an investment with compound interest -/
theorem total_interest_compound (P : ℝ) (r : ℝ) (n : ℕ) :
  let A := P * (1 + r) ^ n
  A - P = P * ((1 + r) ^ n - 1) := by
  sorry

/-- Prove the total interest earned for Linh's investment -/
theorem linh_investment_interest :
  let P : ℝ := 1200  -- Initial investment
  let r : ℝ := 0.08  -- Annual interest rate
  let n : ℕ := 4     -- Number of years
  let A := P * (1 + r) ^ n
  A - P = 1200 * ((1 + 0.08) ^ 4 - 1) := by
  sorry

end NUMINAMATH_CALUDE_total_interest_compound_linh_investment_interest_l1972_197265


namespace NUMINAMATH_CALUDE_downstream_distance_proof_l1972_197245

/-- Calculates the distance traveled downstream given boat speed, stream speed, and time -/
def distance_downstream (boat_speed stream_speed time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat traveling downstream for 7 hours, with a speed of 24 km/hr in still water
    and a stream speed of 4 km/hr, travels 196 km -/
theorem downstream_distance_proof :
  distance_downstream 24 4 7 = 196 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_proof_l1972_197245


namespace NUMINAMATH_CALUDE_gold_coins_distribution_l1972_197230

theorem gold_coins_distribution (x y : ℕ) (h : x * x - y * y = 81 * (x - y)) : x + y = 81 := by
  sorry

end NUMINAMATH_CALUDE_gold_coins_distribution_l1972_197230


namespace NUMINAMATH_CALUDE_hapok_guarantee_l1972_197234

/-- Represents the coin division game between Hapok and Glazok -/
structure CoinGame where
  total_coins : ℕ
  max_handfuls : ℕ

/-- Represents a strategy for Hapok -/
def HapokStrategy := ℕ → ℕ

/-- Represents a strategy for Glazok -/
def GlazokStrategy := ℕ → Bool

/-- The outcome of the game given strategies for both players -/
def gameOutcome (game : CoinGame) (hapok_strat : HapokStrategy) (glazok_strat : GlazokStrategy) : ℕ := sorry

/-- Hapok's guaranteed minimum coins -/
def hapokGuaranteedCoins (game : CoinGame) : ℕ := sorry

/-- The main theorem stating Hapok can guarantee at least 46 coins -/
theorem hapok_guarantee (game : CoinGame) (h1 : game.total_coins = 100) (h2 : game.max_handfuls = 9) :
  hapokGuaranteedCoins game ≥ 46 := sorry

end NUMINAMATH_CALUDE_hapok_guarantee_l1972_197234


namespace NUMINAMATH_CALUDE_no_solution_for_sock_problem_l1972_197264

theorem no_solution_for_sock_problem : ¬∃ (n m : ℕ), 
  n + m = 2009 ∧ 
  (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_sock_problem_l1972_197264


namespace NUMINAMATH_CALUDE_fourth_root_81_times_cube_root_27_times_sqrt_9_l1972_197220

theorem fourth_root_81_times_cube_root_27_times_sqrt_9 : 
  (81 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_81_times_cube_root_27_times_sqrt_9_l1972_197220


namespace NUMINAMATH_CALUDE_car_sale_profit_l1972_197204

def original_price : ℕ := 50000
def loss_percentage : ℚ := 10 / 100
def gain_percentage : ℚ := 20 / 100

def friend_selling_price : ℕ := 54000

theorem car_sale_profit (original_price : ℕ) (loss_percentage gain_percentage : ℚ) 
  (friend_selling_price : ℕ) : 
  let man_selling_price : ℚ := (1 - loss_percentage) * original_price
  let friend_buying_price : ℚ := man_selling_price
  (1 + gain_percentage) * friend_buying_price = friend_selling_price := by
  sorry

end NUMINAMATH_CALUDE_car_sale_profit_l1972_197204


namespace NUMINAMATH_CALUDE_square_side_increase_l1972_197222

theorem square_side_increase (p : ℝ) : 
  (1 + p / 100)^2 = 1.5625 → p = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_side_increase_l1972_197222


namespace NUMINAMATH_CALUDE_cases_in_1990_l1972_197213

/-- Calculates the number of disease cases in a given year, assuming a linear decrease from 1960 to 2000 --/
def diseaseCases (year : ℕ) : ℕ :=
  let initialCases : ℕ := 600000
  let finalCases : ℕ := 600
  let initialYear : ℕ := 1960
  let finalYear : ℕ := 2000
  let totalYears : ℕ := finalYear - initialYear
  let yearlyDecrease : ℕ := (initialCases - finalCases) / totalYears
  initialCases - yearlyDecrease * (year - initialYear)

theorem cases_in_1990 :
  diseaseCases 1990 = 150450 := by
  sorry

end NUMINAMATH_CALUDE_cases_in_1990_l1972_197213


namespace NUMINAMATH_CALUDE_simplify_fraction_l1972_197299

theorem simplify_fraction : (125 : ℚ) / 10000 * 40 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1972_197299


namespace NUMINAMATH_CALUDE_divisibility_implies_lower_bound_l1972_197249

theorem divisibility_implies_lower_bound (n a : ℕ) 
  (h1 : n > 1) 
  (h2 : a > n^2) 
  (h3 : ∀ i ∈ Finset.range n, ∃ x ∈ Finset.range n, (n^2 + i + 1) ∣ (a + x + 1)) : 
  a > n^4 - n^3 := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_lower_bound_l1972_197249


namespace NUMINAMATH_CALUDE_raj_earns_more_by_200_l1972_197287

/-- Represents the dimensions of a rectangular plot of land -/
structure Plot where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular plot -/
def area (p : Plot) : ℕ := p.length * p.width

/-- Calculates the earnings from selling a plot given a price per square foot -/
def earnings (p : Plot) (price_per_sqft : ℕ) : ℕ := area p * price_per_sqft

/-- The difference in earnings between two plots -/
def earnings_difference (p1 p2 : Plot) (price_per_sqft : ℕ) : ℤ :=
  (earnings p1 price_per_sqft : ℤ) - (earnings p2 price_per_sqft : ℤ)

theorem raj_earns_more_by_200 :
  let raj_plot : Plot := ⟨30, 50⟩
  let lena_plot : Plot := ⟨40, 35⟩
  let price_per_sqft : ℕ := 2
  earnings_difference raj_plot lena_plot price_per_sqft = 200 :=
sorry

end NUMINAMATH_CALUDE_raj_earns_more_by_200_l1972_197287


namespace NUMINAMATH_CALUDE_mixed_oil_rate_l1972_197211

/-- Calculates the rate of mixed oil per litre given the volumes and rates of three different oils. -/
theorem mixed_oil_rate (v1 v2 v3 r1 r2 r3 : ℚ) : 
  v1 > 0 ∧ v2 > 0 ∧ v3 > 0 ∧ r1 > 0 ∧ r2 > 0 ∧ r3 > 0 →
  (v1 * r1 + v2 * r2 + v3 * r3) / (v1 + v2 + v3) = 
    (15 * 50 + 8 * 75 + 10 * 65) / (15 + 8 + 10) :=
by
  sorry

#eval (15 * 50 + 8 * 75 + 10 * 65) / (15 + 8 + 10)

end NUMINAMATH_CALUDE_mixed_oil_rate_l1972_197211


namespace NUMINAMATH_CALUDE_cosA_sinB_value_l1972_197240

theorem cosA_sinB_value (A B : Real) (h1 : 0 < A) (h2 : A < π/2) (h3 : 0 < B) (h4 : B < π/2)
  (h5 : (4 + Real.tan A ^ 2) * (5 + Real.tan B ^ 2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
  Real.cos A * Real.sin B = 1 / Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_cosA_sinB_value_l1972_197240


namespace NUMINAMATH_CALUDE_polynomial_has_three_real_roots_l1972_197221

def P (x : ℝ) : ℝ := x^5 + x^4 - x^3 - x^2 - 2*x - 2

theorem polynomial_has_three_real_roots :
  ∃ (a b c : ℝ), (∀ x : ℝ, P x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
sorry

end NUMINAMATH_CALUDE_polynomial_has_three_real_roots_l1972_197221


namespace NUMINAMATH_CALUDE_probability_between_lines_l1972_197243

-- Define the lines
def line_l (x : ℝ) : ℝ := -2 * x + 8
def line_m (x : ℝ) : ℝ := -3 * x + 9

-- Define the region of interest
def region_of_interest (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ y ≤ line_l x ∧ y ≥ line_m x

-- Define the area calculation function
def area_between_lines : ℝ := 2.5

-- Define the total area under line l in the first quadrant
def total_area : ℝ := 16

-- Theorem statement
theorem probability_between_lines :
  (area_between_lines / total_area) = 0.15625 :=
sorry

end NUMINAMATH_CALUDE_probability_between_lines_l1972_197243


namespace NUMINAMATH_CALUDE_new_distance_between_cars_l1972_197277

/-- Calculates the new distance between cars in a convoy after speed reduction -/
theorem new_distance_between_cars 
  (initial_speed : ℝ) 
  (initial_distance : ℝ) 
  (reduced_speed : ℝ) 
  (h1 : initial_speed = 80) 
  (h2 : initial_distance = 10) 
  (h3 : reduced_speed = 60) : 
  (reduced_speed * (initial_distance / 1000) / initial_speed) * 1000 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_new_distance_between_cars_l1972_197277


namespace NUMINAMATH_CALUDE_r_amount_l1972_197272

def total_amount : ℝ := 9000

theorem r_amount (p q r : ℝ) 
  (h1 : p + q + r = total_amount)
  (h2 : r = (2/3) * (p + q)) :
  r = 3600 := by
  sorry

end NUMINAMATH_CALUDE_r_amount_l1972_197272


namespace NUMINAMATH_CALUDE_lecture_duration_in_minutes_l1972_197224

-- Define the duration of the lecture
def lecture_hours : ℕ := 8
def lecture_minutes : ℕ := 45

-- Define the conversion factor
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem lecture_duration_in_minutes :
  lecture_hours * minutes_per_hour + lecture_minutes = 525 := by
  sorry

end NUMINAMATH_CALUDE_lecture_duration_in_minutes_l1972_197224


namespace NUMINAMATH_CALUDE_log_product_equality_l1972_197298

theorem log_product_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x^2 / Real.log y^8) * (Real.log y^3 / Real.log x^7) *
  (Real.log x^4 / Real.log y^5) * (Real.log y^5 / Real.log x^4) *
  (Real.log x^7 / Real.log y^3) * (Real.log y^8 / Real.log x^2) =
  28/3 * (Real.log x / Real.log y) := by
  sorry

end NUMINAMATH_CALUDE_log_product_equality_l1972_197298


namespace NUMINAMATH_CALUDE_calculate_X_l1972_197296

theorem calculate_X (M N X : ℚ) : 
  M = 1764 / 4 →
  N = M / 4 →
  X = M - N →
  X = 330.75 := by
sorry

end NUMINAMATH_CALUDE_calculate_X_l1972_197296


namespace NUMINAMATH_CALUDE_inequality_solution_set_range_of_a_l1972_197239

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2|

-- Theorem for the first part
theorem inequality_solution_set :
  {x : ℝ | 2 * f x < 4 - |x - 1|} = {x : ℝ | -7/3 < x ∧ x < -1} := by sorry

-- Theorem for the second part
theorem range_of_a (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) ↔ -6 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_range_of_a_l1972_197239


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1972_197282

theorem complex_number_quadrant (z : ℂ) (h : z * Complex.I = -2 + Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1972_197282


namespace NUMINAMATH_CALUDE_unique_solution_3n_plus_1_equals_a_squared_l1972_197231

theorem unique_solution_3n_plus_1_equals_a_squared :
  ∀ a n : ℕ+, 3^(n : ℕ) + 1 = (a : ℕ)^2 → a = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_3n_plus_1_equals_a_squared_l1972_197231


namespace NUMINAMATH_CALUDE_inequality_proof_l1972_197242

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a + b + c ≥ (a * (b * c + c + 1)) / (c * a + a + 1) +
              (b * (c * a + a + 1)) / (a * b + b + 1) +
              (c * (a * b + b + 1)) / (b * c + c + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1972_197242


namespace NUMINAMATH_CALUDE_parallelogram_base_l1972_197297

/-- 
Given a parallelogram with area 612 square centimeters and height 18 cm, 
prove that its base is 34 cm.
-/
theorem parallelogram_base (area height : ℝ) (h1 : area = 612) (h2 : height = 18) :
  area / height = 34 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l1972_197297


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_16_l1972_197270

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter_16 (outer : Rectangle) (inner : Rectangle) (shaded_area : ℝ) :
  outer.width = 12 ∧ 
  outer.height = 10 ∧ 
  inner.width = 5 ∧ 
  inner.height = 3 ∧
  shaded_area = 120 →
  perimeter inner = 16 := by
sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_16_l1972_197270


namespace NUMINAMATH_CALUDE_limit_of_f_at_one_l1972_197274

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 - x - 2) / (4 * x^2 - 5 * x + 1)

theorem limit_of_f_at_one :
  ∃ (L : ℝ), L = 5/3 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |f x - L| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_f_at_one_l1972_197274


namespace NUMINAMATH_CALUDE_remainder_98_pow_50_mod_50_l1972_197214

theorem remainder_98_pow_50_mod_50 : 98^50 % 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_pow_50_mod_50_l1972_197214


namespace NUMINAMATH_CALUDE_leopard_arrangement_count_l1972_197273

/-- The number of snow leopards -/
def total_leopards : ℕ := 9

/-- The number of leopards with special placement requirements -/
def special_leopards : ℕ := 3

/-- The number of ways to arrange the shortest two leopards at the ends -/
def shortest_arrangements : ℕ := 2

/-- The number of ways to place the tallest leopard in the middle -/
def tallest_arrangement : ℕ := 1

/-- The number of remaining leopards to be arranged -/
def remaining_leopards : ℕ := total_leopards - special_leopards

/-- Theorem: The number of ways to arrange the leopards is 1440 -/
theorem leopard_arrangement_count : 
  shortest_arrangements * tallest_arrangement * Nat.factorial remaining_leopards = 1440 := by
  sorry

end NUMINAMATH_CALUDE_leopard_arrangement_count_l1972_197273


namespace NUMINAMATH_CALUDE_factorial_20_divisibility_l1972_197203

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def highest_power_dividing (base k : ℕ) (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1) / base) 0

theorem factorial_20_divisibility : 
  (highest_power_dividing 12 6 20 = 6) ∧ 
  (highest_power_dividing 10 4 20 = 4) := by
  sorry

end NUMINAMATH_CALUDE_factorial_20_divisibility_l1972_197203


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l1972_197275

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l1972_197275


namespace NUMINAMATH_CALUDE_cube_sum_minus_triple_product_l1972_197251

theorem cube_sum_minus_triple_product (x y z : ℝ) 
  (sum_eq : x + y + z = 13)
  (sum_products_eq : x*y + x*z + y*z = 32) :
  x^3 + y^3 + z^3 - 3*x*y*z = 949 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_minus_triple_product_l1972_197251


namespace NUMINAMATH_CALUDE_unpainted_cubes_6x6x6_l1972_197256

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  painted_area : Nat
  total_cubes : Nat

/-- Calculate the number of unpainted cubes in a PaintedCube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  cube.total_cubes - (6 * cube.painted_area - 4 * cube.painted_area + 8)

/-- Theorem: In a 6x6x6 cube with central 4x4 areas painted, there are 160 unpainted cubes -/
theorem unpainted_cubes_6x6x6 :
  let cube : PaintedCube := { size := 6, painted_area := 16, total_cubes := 216 }
  unpainted_cubes cube = 160 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_6x6x6_l1972_197256


namespace NUMINAMATH_CALUDE_square_2601_difference_of_squares_l1972_197201

theorem square_2601_difference_of_squares (x : ℤ) (h : x^2 = 2601) :
  (x + 2) * (x - 2) = 2597 := by
sorry

end NUMINAMATH_CALUDE_square_2601_difference_of_squares_l1972_197201


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l1972_197236

/-- The polynomial Q(x) -/
def Q (d : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 8

/-- Theorem: If x + 2 is a factor of Q(x), then d = -14 -/
theorem factor_implies_d_value (d : ℝ) :
  (∀ x, Q d x = 0 ↔ x = -2 ∨ (x + 2) * (x^2 - 5*x + 4 - d/2) = 0) →
  d = -14 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l1972_197236


namespace NUMINAMATH_CALUDE_safe_combinations_l1972_197260

def digits : Finset Nat := {1, 3, 5}

theorem safe_combinations : Fintype.card (Equiv.Perm digits) = 6 := by
  sorry

end NUMINAMATH_CALUDE_safe_combinations_l1972_197260


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1972_197293

/-- Given a rectangle with area 300 square meters, if its length is doubled and
    its width is tripled, the area of the new rectangle will be 1800 square meters. -/
theorem rectangle_area_change (length width : ℝ) 
    (h_area : length * width = 300) : 
    (2 * length) * (3 * width) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1972_197293


namespace NUMINAMATH_CALUDE_four_digit_sum_2008_l1972_197205

theorem four_digit_sum_2008 : ∃ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (n + (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) = 2008) ∧
  (∃ m : ℕ, m ≠ n ∧ 
    (1000 ≤ m ∧ m < 10000) ∧ 
    (m + (m / 1000 + (m / 100 % 10) + (m / 10 % 10) + (m % 10)) = 2008)) :=
by sorry

#check four_digit_sum_2008

end NUMINAMATH_CALUDE_four_digit_sum_2008_l1972_197205


namespace NUMINAMATH_CALUDE_intersected_cubes_count_l1972_197235

/-- Represents a cube composed of unit cubes -/
structure LargeCube where
  size : ℕ
  total_units : ℕ

/-- Represents a plane intersecting the large cube -/
structure IntersectingPlane where
  perpendicular_to_diagonal : Bool
  bisects_diagonal : Bool

/-- Counts the number of unit cubes intersected by the plane -/
def count_intersected_cubes (cube : LargeCube) (plane : IntersectingPlane) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem intersected_cubes_count 
  (cube : LargeCube) 
  (plane : IntersectingPlane) 
  (h1 : cube.size = 4) 
  (h2 : cube.total_units = 64) 
  (h3 : plane.perpendicular_to_diagonal) 
  (h4 : plane.bisects_diagonal) : 
  count_intersected_cubes cube plane = 56 :=
sorry

end NUMINAMATH_CALUDE_intersected_cubes_count_l1972_197235


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1972_197210

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1972_197210


namespace NUMINAMATH_CALUDE_ellipse_properties_l1972_197252

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_minor_axis : b = Real.sqrt 3
  h_eccentricity : Real.sqrt (a^2 - b^2) / a = 1/2

/-- The standard form of the ellipse -/
def standard_form (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2/4 + y^2/3 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1

/-- The maximum area of triangle F₁AB -/
def max_triangle_area (e : Ellipse) : Prop :=
  ∃ (max_area : ℝ),
    max_area = 3 ∧
    ∀ (A B : ℝ × ℝ),
      A ≠ B →
      (∃ (m : ℝ), (A.1 = m * A.2 + 1 ∧ A.1^2/e.a^2 + A.2^2/e.b^2 = 1) ∧
                  (B.1 = m * B.2 + 1 ∧ B.1^2/e.a^2 + B.2^2/e.b^2 = 1)) →
      abs (A.2 - B.2) ≤ max_area

theorem ellipse_properties (e : Ellipse) :
  standard_form e ∧ max_triangle_area e := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1972_197252


namespace NUMINAMATH_CALUDE_jerry_lawsuit_compensation_l1972_197269

def annual_salary : ℕ := 50000
def years_lost : ℕ := 30
def medical_bills : ℕ := 200000
def punitive_multiplier : ℕ := 3
def award_percentage : ℚ := 80 / 100

theorem jerry_lawsuit_compensation :
  let total_salary := annual_salary * years_lost
  let direct_damages := total_salary + medical_bills
  let punitive_damages := direct_damages * punitive_multiplier
  let total_asked := direct_damages + punitive_damages
  let awarded_amount := (total_asked : ℚ) * award_percentage
  awarded_amount = 5440000 := by sorry

end NUMINAMATH_CALUDE_jerry_lawsuit_compensation_l1972_197269


namespace NUMINAMATH_CALUDE_jacket_dimes_count_l1972_197267

/-- The value of a single dime in dollars -/
def dime_value : ℚ := 1 / 10

/-- The total amount of money found in dollars -/
def total_money : ℚ := 19 / 10

/-- The number of dimes found in the shorts -/
def dimes_in_shorts : ℕ := 4

/-- The number of dimes found in the jacket -/
def dimes_in_jacket : ℕ := 15

theorem jacket_dimes_count :
  dimes_in_jacket * dime_value + dimes_in_shorts * dime_value = total_money :=
by sorry

end NUMINAMATH_CALUDE_jacket_dimes_count_l1972_197267


namespace NUMINAMATH_CALUDE_examination_attendance_l1972_197262

theorem examination_attendance :
  ∀ (total_students : ℕ) (passed_percentage : ℚ) (failed_count : ℕ),
    passed_percentage = 35 / 100 →
    failed_count = 520 →
    (1 - passed_percentage) * total_students = failed_count →
    total_students = 800 := by
  sorry

end NUMINAMATH_CALUDE_examination_attendance_l1972_197262


namespace NUMINAMATH_CALUDE_luke_score_l1972_197244

/-- A trivia game where a player gains points each round -/
structure TriviaGame where
  points_per_round : ℕ
  num_rounds : ℕ

/-- Calculate the total points scored in a trivia game -/
def total_points (game : TriviaGame) : ℕ :=
  game.points_per_round * game.num_rounds

/-- Luke's trivia game -/
def luke_game : TriviaGame :=
  { points_per_round := 3
    num_rounds := 26 }

/-- Theorem: Luke scored 78 points in the trivia game -/
theorem luke_score : total_points luke_game = 78 := by
  sorry

end NUMINAMATH_CALUDE_luke_score_l1972_197244


namespace NUMINAMATH_CALUDE_total_paths_XZ_l1972_197283

-- Define the number of paths between points
def paths_XY : ℕ := 2
def paths_YZ : ℕ := 2
def direct_paths_XZ : ℕ := 2

-- Theorem statement
theorem total_paths_XZ : paths_XY * paths_YZ + direct_paths_XZ = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_paths_XZ_l1972_197283


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l1972_197259

/-- The amount of flour Mary put in -/
def flour_added : ℝ := 7.5

/-- The amount of excess flour added -/
def excess_flour : ℝ := 0.8

/-- The amount of flour the recipe wants -/
def recipe_flour : ℝ := flour_added - excess_flour

theorem recipe_flour_amount : recipe_flour = 6.7 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l1972_197259


namespace NUMINAMATH_CALUDE_exam_time_allocation_l1972_197268

/-- Represents the time allocation for an examination with two types of problems. -/
structure ExamTime where
  totalTime : ℕ  -- Total examination time in minutes
  totalQuestions : ℕ  -- Total number of questions
  typeAQuestions : ℕ  -- Number of Type A questions
  typeATimeFactor : ℕ  -- Time factor for Type A questions compared to Type B

/-- Calculates the time spent on Type A problems in an examination. -/
def timeSpentOnTypeA (exam : ExamTime) : ℚ :=
  let totalTypeATime := exam.typeAQuestions * exam.typeATimeFactor
  let totalTypeBTime := exam.totalQuestions - exam.typeAQuestions
  let totalWeightedTime := totalTypeATime + totalTypeBTime
  (exam.totalTime : ℚ) * totalTypeATime / totalWeightedTime

/-- Theorem stating that for the given exam conditions, the time spent on Type A problems is approximately 17 minutes. -/
theorem exam_time_allocation :
  let exam : ExamTime := {
    totalTime := 180,  -- 3 hours * 60 minutes
    totalQuestions := 200,
    typeAQuestions := 10,
    typeATimeFactor := 2
  }
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |timeSpentOnTypeA exam - 17| < ε :=
sorry

end NUMINAMATH_CALUDE_exam_time_allocation_l1972_197268

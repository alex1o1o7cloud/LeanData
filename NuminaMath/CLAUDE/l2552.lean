import Mathlib

namespace NUMINAMATH_CALUDE_oil_quantity_function_correct_l2552_255256

/-- Represents the remaining oil quantity in liters at time t in minutes -/
def Q (t : ℝ) : ℝ := 40 - 0.2 * t

/-- The initial oil quantity in liters -/
def initial_quantity : ℝ := 40

/-- The outflow rate in liters per minute -/
def outflow_rate : ℝ := 0.2

theorem oil_quantity_function_correct :
  ∀ t : ℝ, t ≥ 0 →
    Q t = initial_quantity - outflow_rate * t ∧
    Q 0 = initial_quantity ∧
    ∀ t₁ t₂ : ℝ, t₁ < t₂ → Q t₂ < Q t₁ := by
  sorry

end NUMINAMATH_CALUDE_oil_quantity_function_correct_l2552_255256


namespace NUMINAMATH_CALUDE_intersection_M_N_l2552_255274

def M : Set ℝ := {x | Real.log (x + 1) > 0}
def N : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 ∪ {2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2552_255274


namespace NUMINAMATH_CALUDE_forest_to_verdant_green_conversion_l2552_255235

/-- Represents the ratio of blue to yellow paint in forest green -/
def forest_green_ratio : ℚ := 4 / 3

/-- Represents the ratio of yellow to blue paint in verdant green -/
def verdant_green_ratio : ℚ := 4 / 3

/-- The amount of yellow paint added to change forest green to verdant green -/
def yellow_paint_added : ℝ := 2.333333333333333

/-- The original amount of yellow paint in the forest green mixture -/
def original_yellow_paint : ℝ := 3

theorem forest_to_verdant_green_conversion :
  let b := forest_green_ratio * original_yellow_paint
  (original_yellow_paint + yellow_paint_added) / b = verdant_green_ratio :=
by sorry

end NUMINAMATH_CALUDE_forest_to_verdant_green_conversion_l2552_255235


namespace NUMINAMATH_CALUDE_inequality_solution_l2552_255208

theorem inequality_solution (y : ℝ) : 
  (1 / (y * (y + 2)) - 1 / ((y + 2) * (y + 4)) < 1 / 4) ↔ 
  (y < -4 ∨ (-2 < y ∧ y < 0) ∨ 2 < y) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2552_255208


namespace NUMINAMATH_CALUDE_chocolate_cuts_l2552_255286

/-- The minimum number of cuts required to divide a single piece into n pieces -/
def min_cuts (n : ℕ) : ℕ := n - 1

/-- Theorem stating that the minimum number of cuts to get 24 pieces is 23 -/
theorem chocolate_cuts : min_cuts 24 = 23 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cuts_l2552_255286


namespace NUMINAMATH_CALUDE_man_walking_speed_l2552_255227

/-- Calculates the speed of a man walking in the same direction as a train,
    given the train's length, speed, and time to cross the man. -/
theorem man_walking_speed (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 600 →
  train_speed_kmh = 64 →
  crossing_time = 35.99712023038157 →
  ∃ (man_speed : ℝ), abs (man_speed - 1.10977777777778) < 0.00000000000001 :=
by sorry

end NUMINAMATH_CALUDE_man_walking_speed_l2552_255227


namespace NUMINAMATH_CALUDE_max_sum_abc_l2552_255257

theorem max_sum_abc (a b c : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1)
  (h : a * b * c + 2 * a^2 + 2 * b^2 + 2 * c^2 + c * a - c * b - 4 * a + 4 * b - c = 28) :
  a + b + c ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_sum_abc_l2552_255257


namespace NUMINAMATH_CALUDE_set_relationship_l2552_255268

-- Define the sets P, Q, and S
def P : Set (ℝ × ℝ) := {xy | ∃ (x y : ℝ), xy = (x, y) ∧ Real.log (x * y) = Real.log x + Real.log y}
def Q : Set (ℝ × ℝ) := {xy | ∃ (x y : ℝ), xy = (x, y) ∧ (2 : ℝ)^x * (2 : ℝ)^y = (2 : ℝ)^(x + y)}
def S : Set (ℝ × ℝ) := {xy | ∃ (x y : ℝ), xy = (x, y) ∧ Real.sqrt x * Real.sqrt y = Real.sqrt (x * y)}

-- State the theorem
theorem set_relationship : P ⊆ S ∧ S ⊆ Q := by sorry

end NUMINAMATH_CALUDE_set_relationship_l2552_255268


namespace NUMINAMATH_CALUDE_apples_used_l2552_255210

def initial_apples : ℕ := 40
def remaining_apples : ℕ := 39

theorem apples_used : initial_apples - remaining_apples = 1 := by
  sorry

end NUMINAMATH_CALUDE_apples_used_l2552_255210


namespace NUMINAMATH_CALUDE_cube_root_difference_theorem_l2552_255258

theorem cube_root_difference_theorem (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (1 - x^3)^(1/3) - (1 + x^3)^(1/3) = 1) : 
  x^3 = (x^2 * (28^(1/9))) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_difference_theorem_l2552_255258


namespace NUMINAMATH_CALUDE_solution_for_a_l2552_255220

theorem solution_for_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (eq1 : a + 1/b = 5) (eq2 : b + 1/a = 10) : 
  a = (5 + Real.sqrt 23) / 2 ∨ a = (5 - Real.sqrt 23) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_for_a_l2552_255220


namespace NUMINAMATH_CALUDE_common_remainder_l2552_255224

theorem common_remainder : ∃ r : ℕ, 
  r < 9 ∧ r < 11 ∧ r < 17 ∧
  (3374 % 9 = r) ∧ (3374 % 11 = r) ∧ (3374 % 17 = r) ∧
  r = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_remainder_l2552_255224


namespace NUMINAMATH_CALUDE_common_chord_equation_l2552_255273

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + x - 2*y - 20 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25

/-- The equation of the common chord -/
def common_chord (x y : ℝ) : Prop := x - 2*y + 5 = 0

/-- Theorem stating that the common chord of the two circles is x - 2y + 5 = 0 -/
theorem common_chord_equation :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2552_255273


namespace NUMINAMATH_CALUDE_inequality_proof_l2552_255237

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : 1/a + 1/b + 1/c = 1) :
  Real.sqrt (a + b*c) + Real.sqrt (b + c*a) + Real.sqrt (c + a*b) ≥ 
  Real.sqrt (a*b*c) + Real.sqrt a + Real.sqrt b + Real.sqrt c :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2552_255237


namespace NUMINAMATH_CALUDE_runners_meet_at_start_l2552_255287

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Represents the state of the race -/
structure RaceState where
  runner_a : Runner
  runner_b : Runner
  time : ℝ

def track_length : ℝ := 300

/-- Function to update the race state after each meeting -/
def update_race_state (state : RaceState) : RaceState :=
  sorry

/-- Function to check if both runners are at the starting point -/
def at_start (state : RaceState) : Bool :=
  sorry

/-- Theorem stating that the runners meet at the starting point after 250 seconds -/
theorem runners_meet_at_start :
  let initial_state : RaceState := {
    runner_a := { speed := 2, direction := true },
    runner_b := { speed := 4, direction := false },
    time := 0
  }
  let final_state := update_race_state initial_state
  (at_start final_state ∧ final_state.time = 250) := by
  sorry

end NUMINAMATH_CALUDE_runners_meet_at_start_l2552_255287


namespace NUMINAMATH_CALUDE_pen_rubber_length_difference_l2552_255239

/-- Given a rubber, pen, and pencil with certain length relationships,
    prove that the pen is 3 cm longer than the rubber. -/
theorem pen_rubber_length_difference :
  ∀ (rubber_length pen_length pencil_length : ℝ),
    pencil_length = 12 →
    pen_length = pencil_length - 2 →
    rubber_length + pen_length + pencil_length = 29 →
    pen_length - rubber_length = 3 :=
by sorry

end NUMINAMATH_CALUDE_pen_rubber_length_difference_l2552_255239


namespace NUMINAMATH_CALUDE_product_of_primes_minus_one_l2552_255240

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 0 ∧ m < n → n % m = 0 → m = 1

axiom every_nat_is_product_of_primes :
  ∀ n : Nat, n > 1 → ∃ (factors : List Nat), n = factors.prod ∧ ∀ p ∈ factors, isPrime p

theorem product_of_primes_minus_one (h : isPrime 11 ∧ isPrime 19) :
  2 * 3 * 5 * 7 - 1 = 11 * 19 :=
by sorry

end NUMINAMATH_CALUDE_product_of_primes_minus_one_l2552_255240


namespace NUMINAMATH_CALUDE_large_planter_capacity_l2552_255252

/-- Proves that each large planter can hold 20 seeds given the problem conditions -/
theorem large_planter_capacity
  (total_seeds : ℕ)
  (num_large_planters : ℕ)
  (small_planter_capacity : ℕ)
  (num_small_planters : ℕ)
  (h1 : total_seeds = 200)
  (h2 : num_large_planters = 4)
  (h3 : small_planter_capacity = 4)
  (h4 : num_small_planters = 30)
  : (total_seeds - num_small_planters * small_planter_capacity) / num_large_planters = 20 := by
  sorry

end NUMINAMATH_CALUDE_large_planter_capacity_l2552_255252


namespace NUMINAMATH_CALUDE_garden_area_l2552_255221

theorem garden_area (total_posts : ℕ) (post_spacing : ℝ) (longer_side_ratio : ℕ) :
  total_posts = 24 →
  post_spacing = 3 →
  longer_side_ratio = 3 →
  ∃ (short_side_posts long_side_posts : ℕ),
    short_side_posts > 1 ∧
    long_side_posts > 1 ∧
    long_side_posts = longer_side_ratio * short_side_posts ∧
    total_posts = 2 * short_side_posts + 2 * long_side_posts - 4 ∧
    (short_side_posts - 1) * post_spacing * (long_side_posts - 1) * post_spacing = 297 :=
by sorry

end NUMINAMATH_CALUDE_garden_area_l2552_255221


namespace NUMINAMATH_CALUDE_cubic_tangent_ratio_l2552_255204

-- Define the cubic function
def cubic (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the points A, T, B on the x-axis
structure RootPoints where
  α : ℝ
  γ : ℝ
  β : ℝ

-- Define the theorem
theorem cubic_tangent_ratio 
  (a b c : ℝ) 
  (roots : RootPoints) 
  (h1 : cubic a b c roots.α = 0)
  (h2 : cubic a b c roots.γ = 0)
  (h3 : cubic a b c roots.β = 0)
  (h4 : roots.α < roots.γ)
  (h5 : roots.γ < roots.β) :
  (roots.β - roots.α) / ((roots.α + roots.γ)/2 - (roots.β + roots.γ)/2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_tangent_ratio_l2552_255204


namespace NUMINAMATH_CALUDE_expected_heads_is_56_l2552_255209

/-- The number of fair coins --/
def n : ℕ := 90

/-- The probability of getting heads on a single fair coin toss --/
def p_heads : ℚ := 1/2

/-- The probability of getting tails followed by two consecutive heads --/
def p_tails_then_heads : ℚ := 1/2 * 1/4

/-- The total probability of a coin showing heads under the given rules --/
def p_total : ℚ := p_heads + p_tails_then_heads

/-- The expected number of coins showing heads --/
def expected_heads : ℚ := n * p_total

theorem expected_heads_is_56 : expected_heads = 56 := by sorry

end NUMINAMATH_CALUDE_expected_heads_is_56_l2552_255209


namespace NUMINAMATH_CALUDE_five_students_four_lectures_l2552_255201

/-- The number of ways students can choose lectures -/
def lecture_choices (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: 5 students choosing from 4 lectures results in 4^5 choices -/
theorem five_students_four_lectures :
  lecture_choices 5 4 = 4^5 := by
  sorry

end NUMINAMATH_CALUDE_five_students_four_lectures_l2552_255201


namespace NUMINAMATH_CALUDE_counterexample_exists_l2552_255212

theorem counterexample_exists : ∃ n : ℕ, 
  Nat.Prime n ∧ Even n ∧ ¬(Nat.Prime (n + 2)) := by
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2552_255212


namespace NUMINAMATH_CALUDE_sum_exterior_angles_dodecagon_l2552_255216

/-- A regular dodecagon is a polygon with 12 sides. -/
def RegularDodecagon : Type := Unit

/-- The sum of exterior angles of a polygon. -/
def SumOfExteriorAngles (p : Type) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of a regular dodecagon is 360°. -/
theorem sum_exterior_angles_dodecagon :
  SumOfExteriorAngles RegularDodecagon = 360 := by sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_dodecagon_l2552_255216


namespace NUMINAMATH_CALUDE_school_arrival_time_l2552_255296

/-- Represents the problem of calculating how late a boy arrived at school. -/
theorem school_arrival_time (distance : ℝ) (speed_day1 speed_day2 : ℝ) (early_time : ℝ) : 
  distance = 2.5 ∧ 
  speed_day1 = 5 ∧ 
  speed_day2 = 10 ∧ 
  early_time = 10/60 →
  (distance / speed_day1) * 60 - ((distance / speed_day2) * 60 + early_time * 60) = 5 := by
  sorry

#check school_arrival_time

end NUMINAMATH_CALUDE_school_arrival_time_l2552_255296


namespace NUMINAMATH_CALUDE_course_selection_theorem_l2552_255291

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def total_courses : ℕ := physical_education_courses + art_courses

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def two_course_selections : ℕ := choose physical_education_courses 1 * choose art_courses 1

def three_course_selections : ℕ := 
  choose physical_education_courses 2 * choose art_courses 1 + 
  choose physical_education_courses 1 * choose art_courses 2

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_theorem : total_selections = 64 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l2552_255291


namespace NUMINAMATH_CALUDE_max_weight_is_6250_l2552_255281

/-- The maximum number of crates the trailer can carry on a single trip -/
def max_crates : ℕ := 5

/-- The minimum weight of each crate in kilograms -/
def min_crate_weight : ℕ := 1250

/-- The maximum weight of crates on a single trip -/
def max_total_weight : ℕ := max_crates * min_crate_weight

/-- Theorem stating that the maximum weight of crates on a single trip is 6250 kg -/
theorem max_weight_is_6250 : max_total_weight = 6250 := by
  sorry

end NUMINAMATH_CALUDE_max_weight_is_6250_l2552_255281


namespace NUMINAMATH_CALUDE_square_sum_of_integers_l2552_255267

theorem square_sum_of_integers (x y z : ℤ) 
  (eq1 : x^2*y + y^2*z + z^2*x = 2186)
  (eq2 : x*y^2 + y*z^2 + z*x^2 = 2188) :
  x^2 + y^2 + z^2 = 245 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_integers_l2552_255267


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2552_255222

theorem line_segment_endpoint (y : ℝ) : 
  y < 0 → 
  ((3 - 1)^2 + (-2 - y)^2)^(1/2) = 15 → 
  y = -2 - (221 : ℝ)^(1/2) := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2552_255222


namespace NUMINAMATH_CALUDE_coins_sum_theorem_l2552_255280

theorem coins_sum_theorem (stack1 stack2 stack3 stack4 : ℕ) 
  (h1 : stack1 = 12)
  (h2 : stack2 = 17)
  (h3 : stack3 = 23)
  (h4 : stack4 = 8) :
  stack1 + stack2 + stack3 + stack4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_coins_sum_theorem_l2552_255280


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_intervals_existence_of_positive_value_l2552_255297

-- Define the function f(x) = ln x + ax
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x

-- Theorem for the tangent line equation
theorem tangent_line_at_one (a : ℝ) :
  a = 1 → ∃ (m b : ℝ), ∀ x y, y = f 1 x → (2 : ℝ) * x - y - 1 = 0 := by sorry

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) :
  (a ≥ 0 → ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a < 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < -1/a → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂, -1/a < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂)) := by sorry

-- Theorem for the range of a where f(x₀) > 0 exists
theorem existence_of_positive_value (a : ℝ) :
  (∃ x₀, 0 < x₀ ∧ f a x₀ > 0) ↔ a > -1 / Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_intervals_existence_of_positive_value_l2552_255297


namespace NUMINAMATH_CALUDE_line_E_passes_through_points_l2552_255214

def point := ℝ × ℝ

-- Define the line equations
def line_A (p : point) : Prop := 3 * p.1 - 2 * p.2 + 1 = 0
def line_B (p : point) : Prop := 4 * p.1 - 5 * p.2 + 13 = 0
def line_C (p : point) : Prop := 5 * p.1 + 2 * p.2 - 17 = 0
def line_D (p : point) : Prop := p.1 + 7 * p.2 - 24 = 0
def line_E (p : point) : Prop := p.1 - 4 * p.2 + 10 = 0

-- Define the given point and the endpoints of the line segment
def given_point : point := (4, 3)
def segment_start : point := (2, 7)
def segment_end : point := (8, -2)

-- Define the trisection points
def trisection_point1 : point := (4, 4)
def trisection_point2 : point := (6, 1)

-- Theorem statement
theorem line_E_passes_through_points :
  (line_E given_point ∨ line_E trisection_point1 ∨ line_E trisection_point2) ∧
  ¬(line_A given_point ∨ line_A trisection_point1 ∨ line_A trisection_point2) ∧
  ¬(line_B given_point ∨ line_B trisection_point1 ∨ line_B trisection_point2) ∧
  ¬(line_C given_point ∨ line_C trisection_point1 ∨ line_C trisection_point2) ∧
  ¬(line_D given_point ∨ line_D trisection_point1 ∨ line_D trisection_point2) :=
by sorry

end NUMINAMATH_CALUDE_line_E_passes_through_points_l2552_255214


namespace NUMINAMATH_CALUDE_log_relation_l2552_255218

theorem log_relation (y : ℝ) (k : ℝ) : 
  (Real.log 4 / Real.log 8 = y) → 
  (Real.log 81 / Real.log 2 = k * y) → 
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_log_relation_l2552_255218


namespace NUMINAMATH_CALUDE_cordelia_hair_bleaching_l2552_255200

/-- The time it takes to bleach Cordelia's hair. -/
def bleaching_time : ℝ := 3

/-- The total time for the hair coloring process. -/
def total_time : ℝ := 9

/-- The relationship between dyeing time and bleaching time. -/
def dyeing_time (b : ℝ) : ℝ := 2 * b

theorem cordelia_hair_bleaching :
  bleaching_time + dyeing_time bleaching_time = total_time ∧
  bleaching_time = 3 := by
sorry

end NUMINAMATH_CALUDE_cordelia_hair_bleaching_l2552_255200


namespace NUMINAMATH_CALUDE_exists_triangle_in_circumscribed_polygon_l2552_255244

/-- A polygon circumscribed around a circle. -/
structure CircumscribedPolygon where
  n : ℕ
  sides : Fin n → ℝ
  n_ge_4 : n ≥ 4

/-- Three sides of a polygon can form a triangle if they satisfy the triangle inequality. -/
def CanFormTriangle (p : CircumscribedPolygon) (i j k : Fin p.n) : Prop :=
  p.sides i + p.sides j > p.sides k ∧
  p.sides j + p.sides k > p.sides i ∧
  p.sides k + p.sides i > p.sides j

/-- In any polygon circumscribed around a circle with at least 4 sides,
    there exist three sides that can form a triangle. -/
theorem exists_triangle_in_circumscribed_polygon (p : CircumscribedPolygon) :
  ∃ (i j k : Fin p.n), CanFormTriangle p i j k := by
  sorry

end NUMINAMATH_CALUDE_exists_triangle_in_circumscribed_polygon_l2552_255244


namespace NUMINAMATH_CALUDE_fourth_operation_result_l2552_255292

def pattern_result (a b : ℕ) : ℕ := a * b + a * (b - a)

theorem fourth_operation_result : pattern_result 5 8 = 55 := by
  sorry

end NUMINAMATH_CALUDE_fourth_operation_result_l2552_255292


namespace NUMINAMATH_CALUDE_max_whole_nine_one_number_l2552_255254

def is_nine_one_number (t : ℕ) : Prop :=
  t ≥ 1000 ∧ t ≤ 9999 ∧
  (t / 1000 + (t / 10) % 10 = 9) ∧
  ((t / 100) % 10 - t % 10 = 1)

def P (t : ℕ) : ℕ := 2 * (t / 1000) + (t % 10)

def Q (t : ℕ) : ℕ := 2 * ((t / 100) % 10) + ((t / 10) % 10)

def G (t : ℕ) : ℚ := 2 * (P t : ℚ) / (Q t : ℚ)

def is_whole_nine_one_number (t : ℕ) : Prop :=
  is_nine_one_number t ∧ (G t).isInt

theorem max_whole_nine_one_number :
  ∃ M : ℕ,
    is_whole_nine_one_number M ∧
    ∀ t : ℕ, is_whole_nine_one_number t → t ≤ M ∧
    M = 7524 :=
sorry

end NUMINAMATH_CALUDE_max_whole_nine_one_number_l2552_255254


namespace NUMINAMATH_CALUDE_new_person_weight_l2552_255205

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 35 →
  ∃ (new_weight : ℝ), new_weight = 55 ∧
    new_weight = replaced_weight + initial_count * weight_increase :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2552_255205


namespace NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l2552_255213

theorem sin_minus_cos_sqrt_two (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi →
  Real.sin x - Real.cos x = Real.sqrt 2 →
  x = 3 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l2552_255213


namespace NUMINAMATH_CALUDE_molly_total_swim_distance_l2552_255277

def saturday_distance : ℕ := 45
def sunday_distance : ℕ := 28

theorem molly_total_swim_distance :
  saturday_distance + sunday_distance = 73 :=
by sorry

end NUMINAMATH_CALUDE_molly_total_swim_distance_l2552_255277


namespace NUMINAMATH_CALUDE_greatest_common_length_l2552_255225

theorem greatest_common_length (a b c : Nat) (ha : a = 39) (hb : b = 52) (hc : c = 65) :
  Nat.gcd a (Nat.gcd b c) = 13 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_length_l2552_255225


namespace NUMINAMATH_CALUDE_friends_behind_yuna_l2552_255202

theorem friends_behind_yuna (total_friends : ℕ) (friends_in_front : ℕ) : 
  total_friends = 6 → friends_in_front = 2 → total_friends - friends_in_front = 4 := by
  sorry

end NUMINAMATH_CALUDE_friends_behind_yuna_l2552_255202


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2552_255276

theorem circle_area_ratio (R : ℝ) (R_pos : R > 0) : 
  (π * (R/3)^2) / (π * R^2) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2552_255276


namespace NUMINAMATH_CALUDE_carl_kevin_historical_difference_l2552_255241

/-- A stamp collector's collection --/
structure StampCollection where
  total : ℕ
  international : ℕ
  historical : ℕ
  animal : ℕ

/-- Carl's stamp collection --/
def carl : StampCollection :=
  { total := 125
  , international := 45
  , historical := 60
  , animal := 20 }

/-- Kevin's stamp collection --/
def kevin : StampCollection :=
  { total := 95
  , international := 30
  , historical := 50
  , animal := 15 }

/-- The difference in historical stamps between two collections --/
def historicalStampDifference (c1 c2 : StampCollection) : ℕ :=
  c1.historical - c2.historical

/-- Theorem stating the difference in historical stamps between Carl and Kevin --/
theorem carl_kevin_historical_difference :
  historicalStampDifference carl kevin = 10 := by
  sorry

end NUMINAMATH_CALUDE_carl_kevin_historical_difference_l2552_255241


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2552_255229

/-- Polar to Cartesian conversion theorem for ρ = 4cosθ -/
theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 4 * Real.cos θ) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) →
  (x - 2)^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2552_255229


namespace NUMINAMATH_CALUDE_smartphone_price_difference_l2552_255223

/-- Calculates the final price after discount and tax --/
def finalPrice (basePrice : ℝ) (quantity : ℕ) (discount : ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := basePrice * quantity * (1 - discount)
  discountedPrice * (1 + taxRate)

/-- Proves that the difference between Jane's and Tom's total costs is $112.68 --/
theorem smartphone_price_difference : 
  let storeAPrice := 125
  let storeBPrice := 130
  let storeADiscount := 0.12
  let storeBDiscount := 0.15
  let storeATaxRate := 0.07
  let storeBTaxRate := 0.05
  let tomQuantity := 2
  let janeQuantity := 3
  abs (finalPrice storeBPrice janeQuantity storeBDiscount storeBTaxRate - 
       finalPrice storeAPrice tomQuantity storeADiscount storeATaxRate - 112.68) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_price_difference_l2552_255223


namespace NUMINAMATH_CALUDE_no_solution_equation_l2552_255238

theorem no_solution_equation :
  ¬ ∃ x : ℝ, x - 9 / (x - 5) = 5 - 9 / (x - 5) :=
sorry

end NUMINAMATH_CALUDE_no_solution_equation_l2552_255238


namespace NUMINAMATH_CALUDE_ascent_speed_l2552_255219

/-- Given a journey with ascent and descent, calculate the average speed during ascent -/
theorem ascent_speed
  (total_time : ℝ)
  (overall_speed : ℝ)
  (ascent_time : ℝ)
  (h_total_time : total_time = 6)
  (h_overall_speed : overall_speed = 3.5)
  (h_ascent_time : ascent_time = 4)
  (h_equal_distance : ∀ d : ℝ, d = overall_speed * total_time / 2) :
  ∃ (ascent_speed : ℝ), ascent_speed = 2.625 ∧ ascent_speed = (overall_speed * total_time / 2) / ascent_time :=
by sorry

end NUMINAMATH_CALUDE_ascent_speed_l2552_255219


namespace NUMINAMATH_CALUDE_difference_of_decimal_and_fraction_l2552_255278

theorem difference_of_decimal_and_fraction : 0.650 - (1 / 8 : ℚ) = 0.525 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_decimal_and_fraction_l2552_255278


namespace NUMINAMATH_CALUDE_monkey_bird_problem_l2552_255298

theorem monkey_bird_problem (initial_birds : ℕ) (eaten_birds : ℕ) (monkey_percentage : ℚ) : 
  initial_birds = 6 →
  eaten_birds = 2 →
  monkey_percentage = 6/10 →
  ∃ (initial_monkeys : ℕ), 
    initial_monkeys = 6 ∧
    (initial_monkeys : ℚ) / ((initial_monkeys : ℚ) + (initial_birds - eaten_birds : ℚ)) = monkey_percentage :=
by sorry

end NUMINAMATH_CALUDE_monkey_bird_problem_l2552_255298


namespace NUMINAMATH_CALUDE_max_intersections_fifth_degree_polynomials_l2552_255283

/-- A polynomial of degree 5 with leading coefficient 1 -/
def Polynomial5 : Type := ℝ → ℝ

/-- The difference of two polynomials of degree 5 -/
def PolynomialDifference (p q : Polynomial5) : ℝ → ℝ := fun x => p x - q x

theorem max_intersections_fifth_degree_polynomials (p q : Polynomial5) 
  (h_diff : p ≠ q) : 
  (∃ (S : Finset ℝ), ∀ x : ℝ, p x = q x ↔ x ∈ S) ∧ 
  (∀ (S : Finset ℝ), (∀ x : ℝ, p x = q x ↔ x ∈ S) → S.card ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_intersections_fifth_degree_polynomials_l2552_255283


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l2552_255284

theorem quadratic_root_zero (a : ℝ) :
  (∃ x : ℝ, 2 * x^2 + x + a^2 - 1 = 0 ∧ x = 0) →
  (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l2552_255284


namespace NUMINAMATH_CALUDE_simplify_expression_l2552_255242

theorem simplify_expression (z : ℝ) : z - 2*z + 4*z - 6 + 3 + 7 - 2 = 3*z + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2552_255242


namespace NUMINAMATH_CALUDE_travel_probability_is_two_thirds_l2552_255247

/-- Represents the probability of a bridge being destroyed in an earthquake -/
def p : ℝ := 0.5

/-- Represents the probability of a bridge surviving an earthquake -/
def q : ℝ := 1 - p

/-- Represents the probability of traveling from the first island to the shore after an earthquake -/
noncomputable def travel_probability : ℝ := q / (1 - p * q)

/-- Theorem stating that the probability of traveling from the first island to the shore
    after an earthquake is 2/3 -/
theorem travel_probability_is_two_thirds :
  travel_probability = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_travel_probability_is_two_thirds_l2552_255247


namespace NUMINAMATH_CALUDE_fraction_contradiction_l2552_255290

theorem fraction_contradiction : ¬∃ (x : ℚ), (8 * x = 4) ∧ ((1/4) * 16 = 10 * x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_contradiction_l2552_255290


namespace NUMINAMATH_CALUDE_congruence_problem_l2552_255228

theorem congruence_problem (x : ℤ) : 
  (5 * x + 3) % 18 = 1 → (3 * x + 8) % 18 = 14 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2552_255228


namespace NUMINAMATH_CALUDE_solve_for_y_l2552_255260

theorem solve_for_y (t : ℚ) (x y : ℚ) 
  (hx : x = 3 - 2 * t) 
  (hy : y = 3 * t + 10) 
  (hx_val : x = -4) : 
  y = 41 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2552_255260


namespace NUMINAMATH_CALUDE_rafael_hourly_rate_l2552_255243

theorem rafael_hourly_rate (monday_hours : ℕ) (tuesday_hours : ℕ) (remaining_hours : ℕ) (total_earnings : ℕ) :
  monday_hours = 10 →
  tuesday_hours = 8 →
  remaining_hours = 20 →
  total_earnings = 760 →
  (total_earnings : ℚ) / (monday_hours + tuesday_hours + remaining_hours : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_rafael_hourly_rate_l2552_255243


namespace NUMINAMATH_CALUDE_exists_number_of_1_and_2_divisible_by_2_pow_l2552_255236

/-- A function that checks if a natural number is composed of only digits 1 and 2 -/
def isComposedOf1And2 (x : ℕ) : Prop :=
  ∀ d, d ∈ x.digits 10 → d = 1 ∨ d = 2

/-- Theorem stating that for all natural numbers n, there exists a number x
    composed of only digits 1 and 2 such that x is divisible by 2^n -/
theorem exists_number_of_1_and_2_divisible_by_2_pow (n : ℕ) :
  ∃ x : ℕ, isComposedOf1And2 x ∧ (2^n ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_exists_number_of_1_and_2_divisible_by_2_pow_l2552_255236


namespace NUMINAMATH_CALUDE_gcd_fifteen_n_plus_five_nine_n_plus_four_l2552_255270

theorem gcd_fifteen_n_plus_five_nine_n_plus_four (n : ℕ) 
  (h_pos : n > 0) (h_mod : n % 3 = 1) : 
  Nat.gcd (15 * n + 5) (9 * n + 4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_fifteen_n_plus_five_nine_n_plus_four_l2552_255270


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2552_255215

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (x + 15) - 4 / Real.sqrt (x + 15) = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2552_255215


namespace NUMINAMATH_CALUDE_factorization_equality_l2552_255207

theorem factorization_equality (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2552_255207


namespace NUMINAMATH_CALUDE_perfect_square_5ab4_l2552_255231

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def ends_with_four (n : ℕ) : Prop := n % 10 = 4

def starts_with_five (n : ℕ) : Prop := 5000 ≤ n ∧ n < 6000

def is_5ab4_form (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 5000 + 100 * a + 10 * b + 4

theorem perfect_square_5ab4 (n : ℕ) :
  is_four_digit n →
  ends_with_four n →
  starts_with_five n →
  is_5ab4_form n →
  ∃ (m : ℕ), n = m^2 →
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 5000 + 100 * a + 10 * b + 4 ∧ a + b = 9 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_5ab4_l2552_255231


namespace NUMINAMATH_CALUDE_saturday_sales_proof_l2552_255230

/-- The number of caricatures sold on Saturday -/
def saturday_sales : ℕ := 24

/-- The price of each caricature in dollars -/
def price_per_caricature : ℚ := 20

/-- The number of caricatures sold on Sunday -/
def sunday_sales : ℕ := 16

/-- The total revenue for the weekend in dollars -/
def total_revenue : ℚ := 800

theorem saturday_sales_proof : 
  saturday_sales = 24 ∧ 
  price_per_caricature * (saturday_sales + sunday_sales : ℚ) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_saturday_sales_proof_l2552_255230


namespace NUMINAMATH_CALUDE_prime_polynomial_R_value_l2552_255226

theorem prime_polynomial_R_value :
  ∀ (R Q : ℤ),
    R > 0 →
    (∃ p : ℕ+, Nat.Prime p ∧ (R^3 + 4*R^2 + (Q - 93)*R + 14*Q + 10 : ℤ) = p) →
    R = 5 := by
  sorry

end NUMINAMATH_CALUDE_prime_polynomial_R_value_l2552_255226


namespace NUMINAMATH_CALUDE_carly_payment_l2552_255255

/-- The final amount Carly needs to pay after discount -/
def final_amount (wallet_cost purse_cost shoes_cost discount_rate : ℝ) : ℝ :=
  let total_cost := wallet_cost + purse_cost + shoes_cost
  total_cost * (1 - discount_rate)

/-- Theorem: Given the conditions, Carly needs to pay $198.90 after discount -/
theorem carly_payment : 
  ∀ (wallet_cost purse_cost shoes_cost : ℝ),
    wallet_cost = 22 →
    purse_cost = 4 * wallet_cost - 3 →
    shoes_cost = wallet_cost + purse_cost + 7 →
    final_amount wallet_cost purse_cost shoes_cost 0.1 = 198.90 :=
by
  sorry

#eval final_amount 22 85 114 0.1

end NUMINAMATH_CALUDE_carly_payment_l2552_255255


namespace NUMINAMATH_CALUDE_complement_of_union_l2552_255233

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 1, 3}
def B : Set Nat := {2, 3}

theorem complement_of_union (U A B : Set Nat) 
  (hU : U = {0, 1, 2, 3, 4})
  (hA : A = {0, 1, 3})
  (hB : B = {2, 3}) :
  (U \ (A ∪ B)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l2552_255233


namespace NUMINAMATH_CALUDE_sum_of_integers_l2552_255271

theorem sum_of_integers (x y z : ℕ+) (h : 27 * x.val + 28 * y.val + 29 * z.val = 363) :
  10 * (x.val + y.val + z.val) = 130 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2552_255271


namespace NUMINAMATH_CALUDE_intersection_point_l2552_255246

def L₁ (x : ℝ) : ℝ := 3 * x + 9
def L₂ (x : ℝ) : ℝ := -x + 6

def parameterization_L₁ (t : ℝ) : ℝ × ℝ := (t, 3 * t + 9)
def parameterization_L₂ (s : ℝ) : ℝ × ℝ := (s, -s + 6)

theorem intersection_point :
  ∃ (x y : ℝ), L₁ x = y ∧ L₂ x = y ∧ x = -3/4 ∧ y = 15/4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l2552_255246


namespace NUMINAMATH_CALUDE_w_in_terms_of_abc_l2552_255234

theorem w_in_terms_of_abc (w a b c x y z : ℝ) 
  (hdistinct : w ≠ a ∧ w ≠ b ∧ w ≠ c ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (heq1 : x + y + z = 1)
  (heq2 : x*a^2 + y*b^2 + z*c^2 = w^2)
  (heq3 : x*a^3 + y*b^3 + z*c^3 = w^3)
  (heq4 : x*a^4 + y*b^4 + z*c^4 = w^4) :
  w = -a*b*c / (a*b + b*c + c*a) := by
sorry

end NUMINAMATH_CALUDE_w_in_terms_of_abc_l2552_255234


namespace NUMINAMATH_CALUDE_blue_eyed_percentage_is_correct_l2552_255263

def cat_kittens : List (ℕ × ℕ) := [(5, 7), (6, 8), (4, 6), (7, 9), (3, 5)]

def total_blue_eyed : ℕ := (cat_kittens.map Prod.fst).sum

def total_kittens : ℕ := (cat_kittens.map (λ p => p.fst + p.snd)).sum

def blue_eyed_percentage : ℚ := (total_blue_eyed : ℚ) / (total_kittens : ℚ) * 100

theorem blue_eyed_percentage_is_correct : 
  blue_eyed_percentage = 125/3 := by sorry

end NUMINAMATH_CALUDE_blue_eyed_percentage_is_correct_l2552_255263


namespace NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l2552_255285

theorem diagonal_length_of_quadrilateral (offset1 offset2 area : ℝ) 
  (h1 : offset1 = 11)
  (h2 : offset2 = 9)
  (h3 : area = 400) :
  (2 * area) / (offset1 + offset2) = 40 :=
sorry

end NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l2552_255285


namespace NUMINAMATH_CALUDE_triangle_properties_l2552_255251

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given triangle satisfies the condition b² + c² - a² = 2bc sin(B+C) -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.b^2 + t.c^2 - t.a^2 = 2 * t.b * t.c * Real.sin (t.B + t.C)

/-- Theorem about the angle A and area of the triangle -/
theorem triangle_properties (t : Triangle) 
    (h1 : satisfiesCondition t) 
    (h2 : t.a = 2) 
    (h3 : t.B = π/3) : 
    t.A = π/4 ∧ 
    (1/2 * t.a * t.b * Real.sin t.C = (3 + Real.sqrt 3) / 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2552_255251


namespace NUMINAMATH_CALUDE_doubled_number_excess_l2552_255289

theorem doubled_number_excess (x : ℝ) : x^2 = 25 → 2*x - x/5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_doubled_number_excess_l2552_255289


namespace NUMINAMATH_CALUDE_gathering_handshakes_l2552_255293

/-- Represents a group of people at a gathering --/
structure Gathering where
  total : ℕ
  groupA : ℕ
  groupB : ℕ
  knownInA : ℕ
  hTotal : total = groupA + groupB
  hKnownInA : knownInA ≤ groupA

/-- Calculates the number of handshakes in the gathering --/
def handshakes (g : Gathering) : ℕ :=
  let handshakesBetweenGroups := g.groupB * (g.groupA - g.knownInA)
  let handshakesWithinB := g.groupB * (g.groupB - 1) / 2 - g.groupB * g.knownInA
  handshakesBetweenGroups + handshakesWithinB

/-- Theorem stating the number of handshakes in the specific gathering --/
theorem gathering_handshakes :
  ∃ (g : Gathering),
    g.total = 40 ∧
    g.groupA = 25 ∧
    g.groupB = 15 ∧
    g.knownInA = 5 ∧
    handshakes g = 330 := by
  sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l2552_255293


namespace NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l2552_255269

/-- Given a quadratic function f(x) = ax² + bx + c passing through
    (-2,5), (4,5), and (2,2), prove that the x-coordinate of its vertex is 1. -/
theorem parabola_vertex_x_coordinate 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f (-2) = 5) 
  (h3 : f 4 = 5) 
  (h4 : f 2 = 2) : 
  (- b) / (2 * a) = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l2552_255269


namespace NUMINAMATH_CALUDE_inequalities_from_sqrt_l2552_255279

theorem inequalities_from_sqrt (a b : ℝ) (h : Real.sqrt a > Real.sqrt b) :
  (a^2 > b^2) ∧ ((b + 1) / (a + 1) > b / a) ∧ (b + 1 / (b + 1) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_from_sqrt_l2552_255279


namespace NUMINAMATH_CALUDE_last_four_digits_of_7_to_5000_l2552_255248

theorem last_four_digits_of_7_to_5000 (h : 7^250 ≡ 1 [ZMOD 1250]) : 
  7^5000 ≡ 1 [ZMOD 1250] := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_7_to_5000_l2552_255248


namespace NUMINAMATH_CALUDE_probability_13_11_l2552_255295

/-- Represents a table tennis player -/
inductive Player : Type
| MaLong : Player
| FanZhendong : Player

/-- The probability of a player scoring when serving -/
def scoreProbability (server : Player) : ℚ :=
  match server with
  | Player.MaLong => 2/3
  | Player.FanZhendong => 1/2

/-- The probability of a player scoring when receiving -/
def receiveProbability (receiver : Player) : ℚ :=
  match receiver with
  | Player.MaLong => 1/2
  | Player.FanZhendong => 1/3

/-- Theorem stating the probability of reaching 13:11 score -/
theorem probability_13_11 :
  let initialServer := Player.MaLong
  let prob13_11 := (scoreProbability initialServer * receiveProbability Player.FanZhendong * scoreProbability initialServer * receiveProbability Player.FanZhendong) +
                   (receiveProbability Player.FanZhendong * scoreProbability Player.FanZhendong * scoreProbability initialServer * receiveProbability Player.FanZhendong) +
                   (receiveProbability Player.FanZhendong * scoreProbability Player.FanZhendong * receiveProbability initialServer * scoreProbability Player.FanZhendong) +
                   (scoreProbability initialServer * receiveProbability Player.FanZhendong * receiveProbability initialServer * scoreProbability Player.FanZhendong)
  prob13_11 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_13_11_l2552_255295


namespace NUMINAMATH_CALUDE_circle_center_is_two_two_l2552_255294

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y - 10 = 0

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 18

theorem circle_center_is_two_two :
  CircleCenter 2 2 := by sorry

end NUMINAMATH_CALUDE_circle_center_is_two_two_l2552_255294


namespace NUMINAMATH_CALUDE_angle_sum_when_product_is_four_l2552_255245

theorem angle_sum_when_product_is_four (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : (1 + Real.tan α) * (1 + Real.tan β) = 4) : α + β = π * 3/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_when_product_is_four_l2552_255245


namespace NUMINAMATH_CALUDE_one_zero_point_condition_l2552_255265

/-- A quadratic function with only one zero point -/
def has_one_zero_point (a : ℝ) : Prop :=
  ∃! x : ℝ, a * x^2 - x - 1 = 0

/-- The theorem stating the condition for a quadratic function to have only one zero point -/
theorem one_zero_point_condition (a : ℝ) :
  has_one_zero_point a ↔ a = 0 ∨ a = -1/4 :=
sorry

end NUMINAMATH_CALUDE_one_zero_point_condition_l2552_255265


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l2552_255232

def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let total_needed := (num_friends * (num_friends + 1)) / 2
  if total_needed > initial_coins then
    total_needed - initial_coins
  else
    0

theorem alex_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 90) :
  min_additional_coins num_friends initial_coins = 30 := by
  sorry

#eval min_additional_coins 15 90

end NUMINAMATH_CALUDE_alex_coin_distribution_l2552_255232


namespace NUMINAMATH_CALUDE_tan_value_from_ratio_l2552_255250

theorem tan_value_from_ratio (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 2) : 
  Real.tan α = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_ratio_l2552_255250


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2552_255259

theorem quadratic_equation_equivalence :
  ∀ (x : ℝ), 3 * x^2 + 1 = 6 * x ↔ 3 * x^2 - 6 * x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2552_255259


namespace NUMINAMATH_CALUDE_cube_roots_less_than_12_l2552_255262

theorem cube_roots_less_than_12 : 
  (Finset.range 1728).card = 1727 :=
by sorry

end NUMINAMATH_CALUDE_cube_roots_less_than_12_l2552_255262


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2552_255299

theorem least_subtraction_for_divisibility (n : ℕ) (h : n = 9671) :
  ∃ (k : ℕ), k = 1 ∧ 
  (∀ (m : ℕ), m < k → ¬(∃ (q : ℕ), n - m = 5 * q)) ∧
  (∃ (q : ℕ), n - k = 5 * q) :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2552_255299


namespace NUMINAMATH_CALUDE_ac_plus_bd_equals_negative_ten_l2552_255217

theorem ac_plus_bd_equals_negative_ten
  (a b c d : ℝ)
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 3)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = 6) :
  a * c + b * d = -10 := by
  sorry

end NUMINAMATH_CALUDE_ac_plus_bd_equals_negative_ten_l2552_255217


namespace NUMINAMATH_CALUDE_students_taking_statistics_l2552_255266

/-- Given a group of students with the following properties:
  * There are 89 students in total
  * 36 students are taking history
  * 59 students are taking history or statistics or both
  * 27 students are taking history but not statistics
  Prove that 32 students are taking statistics -/
theorem students_taking_statistics
  (total : ℕ)
  (history : ℕ)
  (history_or_statistics : ℕ)
  (history_not_statistics : ℕ)
  (h_total : total = 89)
  (h_history : history = 36)
  (h_history_or_statistics : history_or_statistics = 59)
  (h_history_not_statistics : history_not_statistics = 27) :
  history_or_statistics - (history - history_not_statistics) = 32 := by
  sorry

#check students_taking_statistics

end NUMINAMATH_CALUDE_students_taking_statistics_l2552_255266


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2552_255206

theorem other_root_of_quadratic (m : ℝ) : 
  (2^2 - 2 + m = 0) → ((-1)^2 - (-1) + m = 0) := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2552_255206


namespace NUMINAMATH_CALUDE_max_q_minus_r_l2552_255275

theorem max_q_minus_r (q r : ℕ+) (h : 1057 = 23 * q + r) : 
  ∀ (q' r' : ℕ+), 1057 = 23 * q' + r' → q - r ≥ q' - r' :=
by sorry

end NUMINAMATH_CALUDE_max_q_minus_r_l2552_255275


namespace NUMINAMATH_CALUDE_men_finished_race_l2552_255272

/-- The number of men who finished the race given the specified conditions -/
def men_who_finished (total_men : ℕ) : ℕ :=
  let tripped := total_men / 4
  let tripped_finished := tripped * 3 / 8
  let remaining_after_trip := total_men - tripped
  let dehydrated := remaining_after_trip * 2 / 9
  let dehydrated_not_finished := dehydrated * 11 / 14
  let remaining_after_dehydration := remaining_after_trip - dehydrated
  let lost := remaining_after_dehydration * 17 / 100
  let lost_finished := lost * 5 / 11
  let remaining_after_lost := remaining_after_dehydration - lost
  let obstacle := remaining_after_lost * 5 / 12
  let obstacle_finished := obstacle * 7 / 15
  let remaining_after_obstacle := remaining_after_lost - obstacle
  let cramps := remaining_after_obstacle * 3 / 7
  let cramps_finished := cramps * 4 / 5
  tripped_finished + lost_finished + obstacle_finished + cramps_finished

/-- Theorem stating that 25 men finished the race given the specified conditions -/
theorem men_finished_race : men_who_finished 80 = 25 := by
  sorry

end NUMINAMATH_CALUDE_men_finished_race_l2552_255272


namespace NUMINAMATH_CALUDE_meal_cost_calculation_l2552_255282

/-- Proves that given a meal with a 12% sales tax, an 18% tip on the original price,
    and a total cost of $33.00, the original cost of the meal before tax and tip is $25.5. -/
theorem meal_cost_calculation (original_cost : ℝ) : 
  let tax_rate : ℝ := 0.12
  let tip_rate : ℝ := 0.18
  let total_cost : ℝ := 33.00
  (1 + tax_rate + tip_rate) * original_cost = total_cost → original_cost = 25.5 := by
sorry

end NUMINAMATH_CALUDE_meal_cost_calculation_l2552_255282


namespace NUMINAMATH_CALUDE_rooms_already_painted_l2552_255261

/-- Given a painting job with the following parameters:
  * total_rooms: The total number of rooms to be painted
  * hours_per_room: The number of hours it takes to paint one room
  * remaining_hours: The number of hours left to complete the job
  This theorem proves that the number of rooms already painted is equal to
  the total number of rooms minus the number of rooms that can be painted
  in the remaining time. -/
theorem rooms_already_painted
  (total_rooms : ℕ)
  (hours_per_room : ℕ)
  (remaining_hours : ℕ)
  (h1 : total_rooms = 10)
  (h2 : hours_per_room = 8)
  (h3 : remaining_hours = 16) :
  total_rooms - (remaining_hours / hours_per_room) = 8 := by
  sorry

end NUMINAMATH_CALUDE_rooms_already_painted_l2552_255261


namespace NUMINAMATH_CALUDE_midpoint_one_sixth_one_ninth_l2552_255203

theorem midpoint_one_sixth_one_ninth :
  (1 / 6 + 1 / 9) / 2 = 5 / 36 := by sorry

end NUMINAMATH_CALUDE_midpoint_one_sixth_one_ninth_l2552_255203


namespace NUMINAMATH_CALUDE_election_result_l2552_255264

theorem election_result (total_votes : ℕ) (majority : ℕ) (winner_percentage : ℝ) :
  total_votes = 440 →
  majority = 176 →
  winner_percentage * (total_votes : ℝ) / 100 - (100 - winner_percentage) * (total_votes : ℝ) / 100 = majority →
  winner_percentage = 70 := by
sorry

end NUMINAMATH_CALUDE_election_result_l2552_255264


namespace NUMINAMATH_CALUDE_adams_shelves_l2552_255211

/-- The number of action figures that can fit on each shelf -/
def action_figures_per_shelf : ℕ := 11

/-- The total number of action figures that can be held by all shelves -/
def total_action_figures : ℕ := 44

/-- The number of shelves in Adam's room -/
def number_of_shelves : ℕ := total_action_figures / action_figures_per_shelf

/-- Theorem stating that the number of shelves in Adam's room is 4 -/
theorem adams_shelves : number_of_shelves = 4 := by
  sorry

end NUMINAMATH_CALUDE_adams_shelves_l2552_255211


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2552_255288

theorem inequality_equivalence (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z > 0) :
  (x * z^2 / z > y * z^2 / z) ↔ (x > y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2552_255288


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l2552_255253

theorem modulus_of_complex_number (z : ℂ) (h : z = 3 + 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l2552_255253


namespace NUMINAMATH_CALUDE_scavenger_hunt_items_l2552_255249

theorem scavenger_hunt_items (tanya samantha lewis : ℕ) : 
  tanya = 4 ∧ 
  samantha = 4 * tanya ∧ 
  lewis = samantha + 4 → 
  lewis = 20 := by
sorry

end NUMINAMATH_CALUDE_scavenger_hunt_items_l2552_255249

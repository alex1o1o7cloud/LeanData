import Mathlib

namespace ice_cream_sundaes_l2913_291364

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) : Nat.choose n 2 = 28 := by
  sorry

end ice_cream_sundaes_l2913_291364


namespace min_value_fraction_equality_condition_l2913_291392

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / (a + 2*b) + b / (a + b) ≥ (2*Real.sqrt 2 - 1) / (2*Real.sqrt 2) :=
by sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / (a + 2*b) + b / (a + b) = (2*Real.sqrt 2 - 1) / (2*Real.sqrt 2) ↔ a = Real.sqrt 2 * b :=
by sorry

end min_value_fraction_equality_condition_l2913_291392


namespace nine_point_four_minutes_in_seconds_l2913_291348

/-- Converts minutes to seconds -/
def minutes_to_seconds (minutes : ℝ) : ℝ :=
  minutes * 60

/-- Theorem stating that 9.4 minutes is equal to 564 seconds -/
theorem nine_point_four_minutes_in_seconds : 
  minutes_to_seconds 9.4 = 564 := by
  sorry

end nine_point_four_minutes_in_seconds_l2913_291348


namespace abcd_efgh_ratio_l2913_291328

theorem abcd_efgh_ratio 
  (a b c d e f g h : ℝ) 
  (hab : a / b = 1 / 3)
  (hbc : b / c = 2)
  (hcd : c / d = 1 / 2)
  (hde : d / e = 3)
  (hef : e / f = 1 / 2)
  (hfg : f / g = 5 / 3)
  (hgh : g / h = 4 / 9)
  (h_nonzero : b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0) :
  a * b * c * d / (e * f * g * h) = 1 / 97 := by
  sorry

end abcd_efgh_ratio_l2913_291328


namespace equal_roots_quadratic_l2913_291393

/-- A quadratic equation x^2 + kx + 1 = 0 has two equal real roots if and only if k = ±2 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 1 = 0 ∧ (∀ y : ℝ, y^2 + k*y + 1 = 0 → y = x)) ↔ 
  k = 2 ∨ k = -2 := by
sorry

end equal_roots_quadratic_l2913_291393


namespace eight_div_repeating_third_l2913_291375

/-- The repeating decimal 0.3333... --/
def repeating_third : ℚ := 1 / 3

/-- The result of 8 divided by 0.3333... --/
theorem eight_div_repeating_third : 8 / repeating_third = 24 := by
  sorry

end eight_div_repeating_third_l2913_291375


namespace problem_solution_l2913_291314

theorem problem_solution : (120 / (6 / 3)) - 15 = 45 := by
  sorry

end problem_solution_l2913_291314


namespace blue_marble_probability_l2913_291313

theorem blue_marble_probability (total : ℕ) (yellow : ℕ) :
  total = 60 →
  yellow = 20 →
  let green := yellow / 2
  let remaining := total - yellow - green
  let blue := remaining / 2
  (blue : ℚ) / total * 100 = 25 := by
sorry

end blue_marble_probability_l2913_291313


namespace arrival_time_difference_l2913_291300

-- Define the distance to the park
def distance_to_park : ℝ := 3

-- Define Jack's speed
def jack_speed : ℝ := 3

-- Define Jill's speed
def jill_speed : ℝ := 12

-- Define the conversion factor from hours to minutes
def hours_to_minutes : ℝ := 60

-- Theorem statement
theorem arrival_time_difference : 
  (distance_to_park / jack_speed - distance_to_park / jill_speed) * hours_to_minutes = 45 := by
  sorry

end arrival_time_difference_l2913_291300


namespace train_speed_l2913_291350

theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 300) (h2 : time = 30) :
  length / time = 10 := by
  sorry

end train_speed_l2913_291350


namespace paint_remaining_l2913_291390

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint > 0 → 
  (initial_paint - (initial_paint / 2) - ((initial_paint - (initial_paint / 2)) / 2)) / initial_paint = 1 / 2 := by
sorry

end paint_remaining_l2913_291390


namespace hiker_journey_distance_l2913_291367

/-- Represents the hiker's journey --/
structure HikerJourney where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The conditions of the hiker's journey --/
def journey_conditions (j : HikerJourney) : Prop :=
  j.distance = j.speed * j.time ∧
  j.distance = (j.speed + 1) * (3/4 * j.time) ∧
  j.distance = (j.speed - 1) * (j.time + 3)

/-- The theorem statement --/
theorem hiker_journey_distance :
  ∀ j : HikerJourney, journey_conditions j → j.distance = 90 := by
  sorry

end hiker_journey_distance_l2913_291367


namespace exam_score_l2913_291332

theorem exam_score (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : 
  total_questions = 60 →
  correct_answers = 38 →
  marks_per_correct = 4 →
  marks_lost_per_wrong = 1 →
  (correct_answers * marks_per_correct) - 
  ((total_questions - correct_answers) * marks_lost_per_wrong) = 130 :=
by
  sorry

end exam_score_l2913_291332


namespace max_non_managers_l2913_291371

/-- The maximum number of non-managers in a department with 9 managers,
    given that the ratio of managers to non-managers must be greater than 7:32 -/
theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 9 →
  (managers : ℚ) / non_managers > 7 / 32 →
  non_managers ≤ 41 :=
by sorry

end max_non_managers_l2913_291371


namespace tired_painting_time_l2913_291368

/-- Represents the time needed to paint houses -/
def paint_time (people : ℕ) (houses : ℕ) (hours : ℝ) (efficiency : ℝ) : Prop :=
  people * hours * efficiency = houses * 32

theorem tired_painting_time :
  paint_time 8 2 4 1 →
  paint_time 5 2 8 0.8 :=
by
  sorry

end tired_painting_time_l2913_291368


namespace exists_polyhedron_with_properties_l2913_291387

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  volume : ℝ
  surfaceArea : ℝ
  volumeBelowWater : ℝ
  surfaceAreaAboveWater : ℝ

/-- Theorem stating the existence of a convex polyhedron with the given properties -/
theorem exists_polyhedron_with_properties :
  ∃ (p : ConvexPolyhedron),
    p.volumeBelowWater = 0.9 * p.volume ∧
    p.surfaceAreaAboveWater > 0.5 * p.surfaceArea :=
sorry

end exists_polyhedron_with_properties_l2913_291387


namespace x_minus_y_values_l2913_291382

theorem x_minus_y_values (x y : ℝ) 
  (h1 : |x| = 5)
  (h2 : y^2 = 16)
  (h3 : x + y > 0) :
  x - y = 1 ∨ x - y = 9 := by
sorry

end x_minus_y_values_l2913_291382


namespace tensor_A_equals_result_l2913_291312

def A : Set ℕ := {0, 2, 3}

def tensor_op (S : Set ℕ) : Set ℕ := {x | ∃ a b, a ∈ S ∧ b ∈ S ∧ x = a + b}

theorem tensor_A_equals_result : tensor_op A = {0, 2, 3, 4, 5, 6} := by
  sorry

end tensor_A_equals_result_l2913_291312


namespace P_outside_triangle_l2913_291351

/-- The point P with coordinates (15.2, 12.4) -/
def P : ℝ × ℝ := (15.2, 12.4)

/-- The first line bounding the triangle: 8x - 15y - 35 = 0 -/
def line1 (x y : ℝ) : Prop := 8 * x - 15 * y - 35 = 0

/-- The second line bounding the triangle: x - 2y - 2 = 0 -/
def line2 (x y : ℝ) : Prop := x - 2 * y - 2 = 0

/-- The third line bounding the triangle: y = 0 -/
def line3 (y : ℝ) : Prop := y = 0

/-- The triangle bounded by the three lines -/
def triangle (x y : ℝ) : Prop := 
  (line1 x y ∨ line2 x y ∨ line3 y) ∧ 
  x ≥ 0 ∧ y ≥ 0 ∧ 8 * x - 15 * y - 35 ≤ 0 ∧ x - 2 * y - 2 ≤ 0

/-- Theorem stating that P is outside the triangle -/
theorem P_outside_triangle : ¬ triangle P.1 P.2 := by
  sorry

end P_outside_triangle_l2913_291351


namespace tan_two_alpha_l2913_291358

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem tan_two_alpha (α : ℝ) (h : (deriv f) α = 3 * f α) : Real.tan (2 * α) = -4/3 := by
  sorry

end tan_two_alpha_l2913_291358


namespace max_abs_z_l2913_291381

theorem max_abs_z (z : ℂ) (h : Complex.abs (z + 5 - 12*I) = 3) : 
  ∃ (max_abs : ℝ), max_abs = 16 ∧ Complex.abs z ≤ max_abs ∧ 
  ∀ (w : ℂ), Complex.abs (w + 5 - 12*I) = 3 → Complex.abs w ≤ max_abs :=
sorry

end max_abs_z_l2913_291381


namespace number_division_problem_l2913_291310

theorem number_division_problem :
  ∃ x : ℚ, (x / 5 = 80 + x / 6) → x = 2400 := by
sorry

end number_division_problem_l2913_291310


namespace largest_integer_inequality_l2913_291376

theorem largest_integer_inequality : ∀ y : ℤ, y ≤ 7 ↔ (y : ℚ) / 4 + 3 / 7 < 9 / 4 := by
  sorry

end largest_integer_inequality_l2913_291376


namespace probability_four_of_each_color_l2913_291305

-- Define the number of balls
def n : ℕ := 8

-- Define the probability of painting a ball black or white
def p : ℚ := 1/2

-- Define the number of ways to choose 4 balls out of 8
def ways_to_choose : ℕ := Nat.choose n (n/2)

-- Define the probability of one specific arrangement
def prob_one_arrangement : ℚ := p^n

-- Statement to prove
theorem probability_four_of_each_color :
  ways_to_choose * prob_one_arrangement = 35/128 := by
  sorry

end probability_four_of_each_color_l2913_291305


namespace subset_range_l2913_291318

theorem subset_range (a : ℝ) : 
  let A := {x : ℝ | 1 ≤ x ∧ x ≤ a}
  let B := {x : ℝ | 0 < x ∧ x < 5}
  A ⊆ B → (1 ≤ a ∧ a < 5) :=
by
  sorry

end subset_range_l2913_291318


namespace symmetry_implies_values_l2913_291301

-- Define the two linear functions
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2
def g (b : ℝ) (x : ℝ) : ℝ := 3 * x - b

-- Define the symmetry condition about y = x
def symmetric (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem symmetry_implies_values :
  ∃ a b : ℝ, symmetric (f a) (g b) → a = 1/3 ∧ b = 6 :=
sorry

end symmetry_implies_values_l2913_291301


namespace shirt_cost_problem_l2913_291377

theorem shirt_cost_problem (total_cost : ℕ) (num_shirts : ℕ) (known_shirt_cost : ℕ) (num_known_shirts : ℕ) :
  total_cost = 85 →
  num_shirts = 5 →
  known_shirt_cost = 15 →
  num_known_shirts = 3 →
  ∃ (remaining_shirt_cost : ℕ),
    remaining_shirt_cost * (num_shirts - num_known_shirts) + known_shirt_cost * num_known_shirts = total_cost ∧
    remaining_shirt_cost = 20 :=
by
  sorry

end shirt_cost_problem_l2913_291377


namespace strawberry_jelly_amount_l2913_291346

/-- The total amount of jelly in grams -/
def total_jelly : ℕ := 6310

/-- The amount of blueberry jelly in grams -/
def blueberry_jelly : ℕ := 4518

/-- The amount of strawberry jelly in grams -/
def strawberry_jelly : ℕ := total_jelly - blueberry_jelly

theorem strawberry_jelly_amount : strawberry_jelly = 1792 := by
  sorry

end strawberry_jelly_amount_l2913_291346


namespace only_cone_no_rectangular_cross_section_l2913_291315

-- Define the geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | RectangularPrism
  | Cube

-- Define a function that checks if a solid can have a rectangular cross-section
def canHaveRectangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => true
  | GeometricSolid.Cone => false
  | GeometricSolid.RectangularPrism => true
  | GeometricSolid.Cube => true

-- Theorem stating that only a cone cannot have a rectangular cross-section
theorem only_cone_no_rectangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveRectangularCrossSection solid) ↔ solid = GeometricSolid.Cone :=
by sorry

end only_cone_no_rectangular_cross_section_l2913_291315


namespace line_through_three_points_l2913_291388

/-- Given a line passing through points (0, 4), (5, k), and (15, 1), prove that k = 3 -/
theorem line_through_three_points (k : ℝ) : 
  (∃ (m b : ℝ), 4 = b ∧ k = 5*m + b ∧ 1 = 15*m + b) → k = 3 := by
  sorry

end line_through_three_points_l2913_291388


namespace donation_to_second_home_l2913_291338

theorem donation_to_second_home 
  (total_donation : ℝ)
  (first_home : ℝ)
  (third_home : ℝ)
  (h1 : total_donation = 700)
  (h2 : first_home = 245)
  (h3 : third_home = 230) :
  total_donation - first_home - third_home = 225 :=
by sorry

end donation_to_second_home_l2913_291338


namespace greatest_t_value_l2913_291345

theorem greatest_t_value : 
  let f : ℝ → ℝ := λ t => (t^2 - t - 90) / (t - 8)
  let g : ℝ → ℝ := λ t => 6 / (t + 7)
  ∃ t_max : ℝ, t_max = -1 ∧ 
    (∀ t : ℝ, t ≠ 8 ∧ t ≠ -7 → f t = g t → t ≤ t_max) :=
by sorry

end greatest_t_value_l2913_291345


namespace factorial_difference_l2913_291323

theorem factorial_difference : (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) - 
                               (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) - 
                               (7 * 6 * 5 * 4 * 3 * 2 * 1) + 
                               (6 * 5 * 4 * 3 * 2 * 1) = 318240 := by
  sorry

end factorial_difference_l2913_291323


namespace factorization_of_difference_of_squares_l2913_291362

theorem factorization_of_difference_of_squares (a b : ℝ) : 
  -a^2 + 4*b^2 = (2*b + a) * (2*b - a) := by sorry

end factorization_of_difference_of_squares_l2913_291362


namespace equation_solution_l2913_291386

theorem equation_solution : 
  ∃ x : ℚ, (x - 50) / 3 = (5 - 3 * x) / 4 + 2 ∧ x = 287 / 13 := by
  sorry

end equation_solution_l2913_291386


namespace value_of_x_l2913_291308

theorem value_of_x (x : ℚ) : (1/4 : ℚ) - (1/6 : ℚ) = 4/x → x = 48 := by
  sorry

end value_of_x_l2913_291308


namespace factorization_of_5_power_1985_minus_1_l2913_291317

theorem factorization_of_5_power_1985_minus_1 :
  ∃ (a b c : ℤ),
    (5^1985 - 1 : ℤ) = a * b * c ∧
    a > 5^100 ∧
    b > 5^100 ∧
    c > 5^100 ∧
    a = 5^397 - 1 ∧
    b = 5^794 - 5^596 + 3*5^397 - 5^199 + 1 ∧
    c = 5^794 + 5^596 + 3*5^397 + 5^199 + 1 :=
by sorry

end factorization_of_5_power_1985_minus_1_l2913_291317


namespace smallest_reducible_fraction_l2913_291333

theorem smallest_reducible_fraction : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (m - 17) ∧ k ∣ (7 * m + 8))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 17) ∧ k ∣ (7 * n + 8)) ∧
  n = 144 := by
  sorry

end smallest_reducible_fraction_l2913_291333


namespace economics_problem_l2913_291342

def R (x : ℕ) : ℚ := x^2 + 16/x^2 + 40

def C (x : ℕ) : ℚ := 10*x + 40/x

def MC (x : ℕ) : ℚ := C (x+1) - C x

def z (x : ℕ) : ℚ := R x - C x

theorem economics_problem (x : ℕ) (h : 1 ≤ x ∧ x ≤ 10) :
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 10 → R y ≥ 72) ∧
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 9 → MC y ≤ 86/9) ∧
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 10 → z y ≥ 7) :=
by sorry

end economics_problem_l2913_291342


namespace x_intercepts_count_l2913_291363

theorem x_intercepts_count :
  let f : ℝ → ℝ := λ x => (x - 5) * (x^2 + 5*x + 6) * (x - 1)
  ∃! (s : Finset ℝ), (∀ x ∈ s, f x = 0) ∧ s.card = 4 :=
sorry

end x_intercepts_count_l2913_291363


namespace line_circle_intersection_l2913_291337

theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), y = k * x + 1 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_intersection_l2913_291337


namespace polynomial_simplification_l2913_291322

/-- Proves the equality of two polynomial expressions -/
theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 8) - (x^6 + 2 * x^5 - 2 * x^4 + x^2 + 5) =
  x^6 - x^5 + 5 * x^4 + x^3 - x^2 + 3 := by
sorry

end polynomial_simplification_l2913_291322


namespace infinite_symmetric_subset_exists_l2913_291372

/-- A color type representing black and white --/
inductive Color
| Black
| White

/-- A point in the plane with integer coordinates --/
structure Point where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point with integer coordinates --/
def Coloring := Point → Color

/-- A set of points is symmetric about a center point if for every point in the set,
    its reflection about the center is also in the set --/
def IsSymmetric (S : Set Point) (center : Point) : Prop :=
  ∀ p ∈ S, Point.mk (2 * center.x - p.x) (2 * center.y - p.y) ∈ S

/-- The main theorem stating the existence of an infinite symmetric subset --/
theorem infinite_symmetric_subset_exists (coloring : Coloring) :
  ∃ (S : Set Point) (c : Color) (center : Point),
    Set.Infinite S ∧ (∀ p ∈ S, coloring p = c) ∧ IsSymmetric S center :=
  sorry

end infinite_symmetric_subset_exists_l2913_291372


namespace negation_absolute_value_inequality_l2913_291365

theorem negation_absolute_value_inequality :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| < 0) :=
by sorry

end negation_absolute_value_inequality_l2913_291365


namespace solution_set_l2913_291336

theorem solution_set (x : ℝ) :
  (|x^2 - x - 2| + |1/x| = |x^2 - x - 2 + 1/x|) →
  ((-1 ≤ x ∧ x < 0) ∨ x ≥ 2) :=
by sorry

end solution_set_l2913_291336


namespace perpendicular_implies_parallel_perpendicular_parallel_implies_perpendicular_planes_l2913_291384

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Theorem 1
theorem perpendicular_implies_parallel 
  (m : Line) (α β : Plane) :
  perpendicular m α → perpendicular m β → parallel_planes α β :=
sorry

-- Theorem 2
theorem perpendicular_parallel_implies_perpendicular_planes 
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → perpendicular_planes α β :=
sorry

end perpendicular_implies_parallel_perpendicular_parallel_implies_perpendicular_planes_l2913_291384


namespace opposite_pairs_l2913_291307

theorem opposite_pairs : 
  (3^2 = -(-(3^2))) ∧ 
  (-4 ≠ -(-4)) ∧ 
  (-3 ≠ -(-|-3|)) ∧ 
  (-2^3 ≠ -((-2)^3)) := by
  sorry

end opposite_pairs_l2913_291307


namespace lansing_elementary_students_l2913_291373

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := 247

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := num_schools * students_per_school

theorem lansing_elementary_students :
  total_students = 6175 :=
by sorry

end lansing_elementary_students_l2913_291373


namespace feet_to_inches_conversion_l2913_291321

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

/-- Initial height of the tree in feet -/
def initial_height : ℕ := 52

/-- Annual growth of the tree in feet -/
def annual_growth : ℕ := 5

/-- Time period in years -/
def time_period : ℕ := 8

/-- Theorem stating that the conversion factor from feet to inches is 12 -/
theorem feet_to_inches_conversion :
  feet_to_inches = 12 :=
sorry

end feet_to_inches_conversion_l2913_291321


namespace base6_addition_proof_l2913_291316

/-- Converts a base 6 number represented as a list of digits to a natural number. -/
def fromBase6 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Checks if a number is a single digit in base 6. -/
def isSingleDigitBase6 (n : Nat) : Prop := n < 6

theorem base6_addition_proof (C D : Nat) 
  (hC : isSingleDigitBase6 C) 
  (hD : isSingleDigitBase6 D) : 
  fromBase6 [1, 1, C] + fromBase6 [5, 2, D] + fromBase6 [C, 2, 4] = fromBase6 [4, 4, 3] → 
  (if C ≥ D then C - D else D - C) = 3 := by
  sorry

end base6_addition_proof_l2913_291316


namespace difference_of_squares_l2913_291355

theorem difference_of_squares (a b : ℝ) : (-a + b) * (-a - b) = b^2 - a^2 := by
  sorry

end difference_of_squares_l2913_291355


namespace value_of_x_l2913_291354

theorem value_of_x : ∃ x : ℝ, 3 * x + 15 = (1/3) * (7 * x + 45) ∧ x = 0 := by
  sorry

end value_of_x_l2913_291354


namespace hockey_season_games_l2913_291326

/-- Calculate the number of games in a hockey season -/
theorem hockey_season_games (n : ℕ) (m : ℕ) (h1 : n = 16) (h2 : m = 10) :
  (n * (n - 1) / 2) * m = 2400 := by
  sorry

end hockey_season_games_l2913_291326


namespace true_discount_calculation_l2913_291325

/-- Given a present worth and banker's discount, calculate the true discount -/
theorem true_discount_calculation (present_worth banker_discount : ℚ) :
  present_worth = 400 →
  banker_discount = 21 →
  ∃ true_discount : ℚ,
    banker_discount = true_discount + (true_discount * banker_discount / present_worth) ∧
    true_discount = 8400 / 421 :=
by sorry

end true_discount_calculation_l2913_291325


namespace x_axis_reflection_l2913_291344

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem x_axis_reflection :
  let p : ℝ × ℝ := (-3, 5)
  reflect_x p = (-3, -5) := by sorry

end x_axis_reflection_l2913_291344


namespace intersection_point_satisfies_equations_unique_intersection_point_l2913_291398

/-- The point of intersection for two lines defined by linear equations -/
def intersection_point : ℚ × ℚ := (24/25, 34/25)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: 2y = 7x - 4 -/
def line2 (x y : ℚ) : Prop := 2 * y = 7 * x - 4

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end intersection_point_satisfies_equations_unique_intersection_point_l2913_291398


namespace coffee_consumption_l2913_291357

theorem coffee_consumption (people : ℕ) (coffee_per_cup : ℚ) (coffee_cost : ℚ) (weekly_spend : ℚ) :
  people = 4 →
  coffee_per_cup = 1/2 →
  coffee_cost = 5/4 →
  weekly_spend = 35 →
  (weekly_spend / coffee_cost / coffee_per_cup / people / 7 : ℚ) = 2 := by
  sorry

end coffee_consumption_l2913_291357


namespace max_x_plus_z_l2913_291397

theorem max_x_plus_z (x y z t : ℝ) 
  (h1 : x^2 + y^2 = 4)
  (h2 : z^2 + t^2 = 9)
  (h3 : x*t + y*z = 6) :
  x + z ≤ Real.sqrt 13 :=
sorry

end max_x_plus_z_l2913_291397


namespace small_gardens_and_pepper_seeds_l2913_291385

/-- Represents the number of small gardens for each vegetable type -/
structure SmallGardens where
  tomatoes : ℕ
  lettuce : ℕ
  peppers : ℕ

/-- Represents the seed requirements for each vegetable type -/
structure SeedRequirements where
  tomatoes : ℕ
  lettuce : ℕ
  peppers : ℕ

def total_seeds : ℕ := 42
def big_garden_seeds : ℕ := 36

def small_gardens : SmallGardens :=
  { tomatoes := 3
  , lettuce := 2
  , peppers := 0 }

def seed_requirements : SeedRequirements :=
  { tomatoes := 4
  , lettuce := 3
  , peppers := 2 }

def remaining_seeds : ℕ := total_seeds - big_garden_seeds

theorem small_gardens_and_pepper_seeds :
  (small_gardens.tomatoes + small_gardens.lettuce + small_gardens.peppers = 5) ∧
  (small_gardens.peppers * seed_requirements.peppers = 0) :=
by sorry

end small_gardens_and_pepper_seeds_l2913_291385


namespace uniform_motion_final_position_l2913_291329

/-- A point moving with uniform velocity in a 2D plane. -/
structure MovingPoint where
  initial_position : ℝ × ℝ
  velocity : ℝ × ℝ

/-- Calculate the final position of a moving point after a given time. -/
def final_position (p : MovingPoint) (t : ℝ) : ℝ × ℝ :=
  (p.initial_position.1 + t * p.velocity.1, p.initial_position.2 + t * p.velocity.2)

theorem uniform_motion_final_position :
  let p : MovingPoint := { initial_position := (-10, 10), velocity := (4, -3) }
  final_position p 5 = (10, -5) := by
  sorry

end uniform_motion_final_position_l2913_291329


namespace cube_root_equation_sum_l2913_291394

theorem cube_root_equation_sum (x y z : ℕ+) :
  (4 * Real.sqrt (Real.rpow 7 (1/3) - Real.rpow 6 (1/3)) = Real.rpow x.val (1/3) + Real.rpow y.val (1/3) - Real.rpow z.val (1/3)) →
  x.val + y.val + z.val = 79 := by
sorry

end cube_root_equation_sum_l2913_291394


namespace solve_equation_l2913_291378

theorem solve_equation (x : ℝ) : 2*x + 5 - 3*x + 7 = 8 → x = 4 := by
  sorry

end solve_equation_l2913_291378


namespace twentynine_is_perfect_number_pairing_x_squared_minus_6x_plus_13_perfect_number_condition_for_S_l2913_291319

-- Definition of a perfect number
def isPerfectNumber (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2 ∧ a > 0 ∧ b > 0

-- Theorem 1
theorem twentynine_is_perfect_number : isPerfectNumber 29 :=
sorry

-- Theorem 2
theorem pairing_x_squared_minus_6x_plus_13 :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧
  (∀ x : ℝ, x^2 - 6*x + 13 = (x - m)^2 + n^2) ∧
  m * n = 6 :=
sorry

-- Theorem 3
theorem perfect_number_condition_for_S (k : ℝ) :
  (∀ x y : ℤ, ∃ a b : ℤ, x^2 + 4*y^2 + 4*x - 12*y + k = a^2 + b^2) ↔ k = 13 :=
sorry

end twentynine_is_perfect_number_pairing_x_squared_minus_6x_plus_13_perfect_number_condition_for_S_l2913_291319


namespace inequality_condition_l2913_291341

theorem inequality_condition (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x - 1 < 0) → 
  (-3 < m ∧ m < 1) ∧ 
  ¬(∀ m : ℝ, (∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x - 1 < 0) → (-3 < m ∧ m < 1)) :=
by sorry

end inequality_condition_l2913_291341


namespace special_triangle_perimeter_l2913_291334

/-- A triangle with sides satisfying x^2 - 6x + 8 = 0 --/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a^2 - 6*a + 8 = 0
  hb : b^2 - 6*b + 8 = 0
  hc : c^2 - 6*c + 8 = 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of possible perimeters for the special triangle --/
def possible_perimeters : Set ℝ := {6, 10, 12}

/-- Theorem stating that the perimeter of a SpecialTriangle is in the set of possible perimeters --/
theorem special_triangle_perimeter (t : SpecialTriangle) : 
  (t.a + t.b + t.c) ∈ possible_perimeters := by
  sorry

#check special_triangle_perimeter

end special_triangle_perimeter_l2913_291334


namespace car_speed_equality_l2913_291359

/-- Proves that given the conditions of the car problem, the average speed of Car Y equals that of Car X -/
theorem car_speed_equality (speed_x : ℝ) (start_delay : ℝ) (distance_after_y_start : ℝ) : 
  speed_x = 35 →
  start_delay = 72 / 60 →
  distance_after_y_start = 210 →
  ∃ (time_after_y_start : ℝ), 
    time_after_y_start > 0 ∧ 
    speed_x * time_after_y_start = distance_after_y_start ∧
    distance_after_y_start / time_after_y_start = speed_x := by
  sorry

#check car_speed_equality

end car_speed_equality_l2913_291359


namespace star_calculation_l2913_291324

-- Define the binary operation *
def star (a b : ℝ) : ℝ := (a - b)^2

-- State the theorem
theorem star_calculation (x y z : ℝ) : 
  star ((x - z)^2) ((z - y)^2) = (x^2 - 2*x*z + 2*z*y - y^2)^2 := by
  sorry

end star_calculation_l2913_291324


namespace smallest_m_satisfying_conditions_l2913_291347

theorem smallest_m_satisfying_conditions : ∃ m : ℕ,
  (100 ≤ m ∧ m ≤ 999) ∧
  (m + 6) % 9 = 0 ∧
  (m - 9) % 6 = 0 ∧
  (∀ n : ℕ, (100 ≤ n ∧ n < m ∧ (n + 6) % 9 = 0 ∧ (n - 9) % 6 = 0) → False) ∧
  m = 111 :=
by sorry

end smallest_m_satisfying_conditions_l2913_291347


namespace admission_probability_l2913_291391

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of universities -/
def num_universities : ℕ := 3

/-- The total number of possible admission arrangements -/
def total_arrangements : ℕ := num_universities ^ num_students

/-- The number of arrangements where each university admits at least one student -/
def favorable_arrangements : ℕ := 36

/-- The probability that each university admits at least one student -/
def probability : ℚ := favorable_arrangements / total_arrangements

theorem admission_probability : probability = 4 / 9 := by
  sorry

end admission_probability_l2913_291391


namespace inequality_system_no_solution_l2913_291311

theorem inequality_system_no_solution : 
  ∀ x : ℝ, ¬(2 * x^2 - 5 * x + 3 < 0 ∧ (x - 1) / (2 - x) ≤ 1) :=
by sorry

end inequality_system_no_solution_l2913_291311


namespace james_total_cost_l2913_291343

/-- The total cost of buying dirt bikes, off-road vehicles, and registering them all -/
def total_cost (dirt_bike_count : ℕ) (dirt_bike_price : ℕ) 
                (offroad_count : ℕ) (offroad_price : ℕ) 
                (registration_fee : ℕ) : ℕ :=
  dirt_bike_count * dirt_bike_price + 
  offroad_count * offroad_price + 
  (dirt_bike_count + offroad_count) * registration_fee

/-- Theorem stating the total cost for James' purchase -/
theorem james_total_cost : 
  total_cost 3 150 4 300 25 = 1825 := by
  sorry

end james_total_cost_l2913_291343


namespace heather_biking_days_l2913_291304

/-- Given that Heather bicycled 40.0 kilometers per day for some days and 320 kilometers in total,
    prove that the number of days she biked is 8. -/
theorem heather_biking_days (daily_distance : ℝ) (total_distance : ℝ) 
    (h1 : daily_distance = 40.0)
    (h2 : total_distance = 320) :
    total_distance / daily_distance = 8 := by
  sorry

end heather_biking_days_l2913_291304


namespace velvet_for_cloak_l2913_291360

/-- The number of hats that can be made from one yard of velvet -/
def hats_per_yard : ℕ := 4

/-- The total number of yards of velvet needed for 6 cloaks and 12 hats -/
def total_yards : ℕ := 21

/-- The number of cloaks made with the total yards -/
def num_cloaks : ℕ := 6

/-- The number of hats made with the total yards -/
def num_hats : ℕ := 12

/-- The number of yards needed to make one cloak -/
def yards_per_cloak : ℚ := 3

theorem velvet_for_cloak :
  yards_per_cloak = (total_yards - (num_hats / hats_per_yard : ℚ)) / num_cloaks := by
  sorry

end velvet_for_cloak_l2913_291360


namespace paint_usage_l2913_291374

theorem paint_usage (initial_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ) :
  initial_paint = 360 ∧
  first_week_fraction = 1/4 ∧
  second_week_fraction = 1/2 →
  let first_week_usage := first_week_fraction * initial_paint
  let remaining_paint := initial_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 225 :=
by sorry

end paint_usage_l2913_291374


namespace coconut_grove_problem_l2913_291302

/-- Coconut grove problem -/
theorem coconut_grove_problem (x : ℝ) : 
  (60 * (x + 4) + 120 * x + 180 * (x - 4)) / (3 * x) = 100 → x = 8 := by
  sorry

end coconut_grove_problem_l2913_291302


namespace water_depth_calculation_l2913_291331

def water_depth (ron_height dean_height_difference : ℝ) : ℝ :=
  let dean_height := ron_height - dean_height_difference
  2.5 * dean_height + 3

theorem water_depth_calculation (ron_height dean_height_difference : ℝ) 
  (h1 : ron_height = 14.2)
  (h2 : dean_height_difference = 8.3) :
  water_depth ron_height dean_height_difference = 17.75 := by
  sorry

end water_depth_calculation_l2913_291331


namespace expected_winnings_l2913_291327

-- Define the spinner outcomes
inductive Outcome
| Green
| Red
| Blue

-- Define the probability function
def probability (o : Outcome) : ℚ :=
  match o with
  | Outcome.Green => 1/4
  | Outcome.Red => 1/2
  | Outcome.Blue => 1/4

-- Define the winnings function
def winnings (o : Outcome) : ℤ :=
  match o with
  | Outcome.Green => 2
  | Outcome.Red => 4
  | Outcome.Blue => -6

-- Define the expected value function
def expectedValue : ℚ :=
  (probability Outcome.Green * winnings Outcome.Green) +
  (probability Outcome.Red * winnings Outcome.Red) +
  (probability Outcome.Blue * winnings Outcome.Blue)

-- Theorem stating the expected winnings
theorem expected_winnings : expectedValue = 1 := by
  sorry

end expected_winnings_l2913_291327


namespace fraction_evaluation_l2913_291340

theorem fraction_evaluation :
  let x : ℚ := 4/3
  let y : ℚ := 5/7
  (3*x + 7*y) / (21*x*y) = 9/140 := by
sorry

end fraction_evaluation_l2913_291340


namespace original_decimal_value_l2913_291335

theorem original_decimal_value : 
  ∃ x : ℝ, (x / 100 = x - 1.782) ∧ (x = 1.8) := by
  sorry

end original_decimal_value_l2913_291335


namespace quadratic_equation_solutions_l2913_291339

theorem quadratic_equation_solutions :
  {x : ℝ | x^2 - Real.sqrt 2 * x = 0} = {0, Real.sqrt 2} := by sorry

end quadratic_equation_solutions_l2913_291339


namespace arithmetic_operations_l2913_291320

theorem arithmetic_operations : 
  (-3 + 5 - (-2) = 4) ∧ 
  (-6 / (1/4) * (-4) = 96) ∧ 
  ((5/6 - 3/4 + 1/3) * (-24) = -10) ∧ 
  ((-1)^2023 - (4 - (-3)^2) / (2/7 - 1) = -8) := by
  sorry

#check arithmetic_operations

end arithmetic_operations_l2913_291320


namespace edmund_earnings_is_64_l2913_291352

/-- Calculates Edmund's earnings for extra chores over two weeks -/
def edmund_earnings (normal_chores_per_week : ℕ) (chores_per_day : ℕ) (days : ℕ) (pay_per_extra_chore : ℕ) : ℕ :=
  let total_chores := chores_per_day * days
  let normal_chores := normal_chores_per_week * 2
  let extra_chores := total_chores - normal_chores
  extra_chores * pay_per_extra_chore

/-- Proves that Edmund's earnings for extra chores over two weeks is $64 -/
theorem edmund_earnings_is_64 :
  edmund_earnings 12 4 14 2 = 64 := by
  sorry

end edmund_earnings_is_64_l2913_291352


namespace books_added_to_shelf_l2913_291361

theorem books_added_to_shelf (initial_action_figures initial_books added_books : ℕ) :
  initial_action_figures = 7 →
  initial_books = 2 →
  initial_action_figures = (initial_books + added_books) + 1 →
  added_books = 4 :=
by sorry

end books_added_to_shelf_l2913_291361


namespace age_of_b_l2913_291389

theorem age_of_b (a b c : ℕ) : 
  (a + b + c) / 3 = 29 → 
  (a + c) / 2 = 32 → 
  b = 23 := by
sorry

end age_of_b_l2913_291389


namespace sum_of_max_min_a_l2913_291303

theorem sum_of_max_min_a (a : ℝ) : 
  (∀ x y : ℝ, x^2 - a*x - 20*a^2 < 0 ∧ y^2 - a*y - 20*a^2 < 0 → |x - y| ≤ 9) →
  ∃ a_min a_max : ℝ, 
    (∀ a' : ℝ, (∃ x : ℝ, x^2 - a'*x - 20*a'^2 < 0) → a_min ≤ a' ∧ a' ≤ a_max) ∧
    a_min + a_max = 0 :=
sorry

end sum_of_max_min_a_l2913_291303


namespace solve_equation_l2913_291396

theorem solve_equation :
  ∃ x : ℚ, 5 * x + 9 * x = 350 - 10 * (x - 5) ∧ x = 50 / 3 :=
by
  sorry

end solve_equation_l2913_291396


namespace acid_dilution_l2913_291306

theorem acid_dilution (m : ℝ) (x : ℝ) (h : m > 25) :
  (m * m / 100 = (m - 5) / 100 * (m + x)) → x = 5 * m / (m - 5) := by
  sorry

end acid_dilution_l2913_291306


namespace shaded_area_percentage_l2913_291379

theorem shaded_area_percentage (total_squares : ℕ) (shaded_squares : ℕ) 
  (h1 : total_squares = 5) 
  (h2 : shaded_squares = 2) : 
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100 = 40 := by
  sorry

end shaded_area_percentage_l2913_291379


namespace no_solution_exists_l2913_291369

theorem no_solution_exists : ¬∃ (x y z : ℝ), 
  (x^2 ≠ y^2) ∧ (y^2 ≠ z^2) ∧ (z^2 ≠ x^2) ∧
  (1 / (x^2 - y^2) + 1 / (y^2 - z^2) + 1 / (z^2 - x^2) = 0) := by
  sorry

end no_solution_exists_l2913_291369


namespace simplify_radical_product_l2913_291395

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x) = 120 * x * Real.sqrt (2 * x) :=
by sorry

end simplify_radical_product_l2913_291395


namespace harmonic_sum_increase_l2913_291330

theorem harmonic_sum_increase (k : ℕ) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k := by
  sorry

end harmonic_sum_increase_l2913_291330


namespace probability_of_X_selection_l2913_291356

theorem probability_of_X_selection (p_Y p_both : ℝ) 
  (h1 : p_Y = 2/7)
  (h2 : p_both = 0.09523809523809523)
  (h3 : p_both = p_Y * p_X)
  : p_X = 1/3 :=
by sorry

end probability_of_X_selection_l2913_291356


namespace arrangements_equal_72_l2913_291309

-- Define the number of men and women
def num_men : ℕ := 4
def num_women : ℕ := 3

-- Define the number of groups and their sizes
def num_groups : ℕ := 3
def group_sizes : List ℕ := [3, 3, 2]

-- Define the minimum number of men and women in each group
def min_men_per_group : ℕ := 1
def min_women_per_group : ℕ := 1

-- Define a function to calculate the number of arrangements
def num_arrangements (m : ℕ) (w : ℕ) (gs : List ℕ) (min_m : ℕ) (min_w : ℕ) : ℕ := sorry

-- Theorem statement
theorem arrangements_equal_72 :
  num_arrangements num_men num_women group_sizes min_men_per_group min_women_per_group = 72 := by
  sorry

end arrangements_equal_72_l2913_291309


namespace hospital_staff_remaining_l2913_291380

/-- Given a hospital with an initial count of doctors and nurses,
    calculate the remaining staff after some quit. -/
def remaining_staff (initial_doctors : ℕ) (initial_nurses : ℕ)
                    (doctors_quit : ℕ) (nurses_quit : ℕ) : ℕ :=
  (initial_doctors - doctors_quit) + (initial_nurses - nurses_quit)

/-- Theorem stating that with 11 doctors and 18 nurses initially,
    if 5 doctors and 2 nurses quit, 22 staff members remain. -/
theorem hospital_staff_remaining :
  remaining_staff 11 18 5 2 = 22 := by
  sorry

end hospital_staff_remaining_l2913_291380


namespace x_equals_y_l2913_291399

theorem x_equals_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x = 1 + 1 / y) (h2 : y = 1 + 1 / x) : y = x := by
  sorry

end x_equals_y_l2913_291399


namespace x_abs_x_is_k_function_l2913_291366

/-- Definition of a K function -/
def is_k_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) + f x = 0) ∧
  (∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0)

/-- The function f(x) = x|x| -/
def f (x : ℝ) : ℝ := x * |x|

/-- Theorem: f(x) = x|x| is a K function -/
theorem x_abs_x_is_k_function : is_k_function f := by sorry

end x_abs_x_is_k_function_l2913_291366


namespace sqrt_x_minus_2_real_l2913_291353

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end sqrt_x_minus_2_real_l2913_291353


namespace jenna_earnings_l2913_291383

def calculate_earnings (distance : ℕ) : ℚ :=
  let first_100 := min distance 100
  let next_200 := min (distance - 100) 200
  let beyond_300 := max (distance - 300) 0
  0.4 * first_100 + 0.5 * next_200 + 0.6 * beyond_300

def round_trip_distance : ℕ := 800

def base_earnings : ℚ := 2 * calculate_earnings (round_trip_distance / 2)

def bonus : ℚ := 100 * (round_trip_distance / 500)

def weather_reduction : ℚ := 0.1

def rest_stop_reduction : ℚ := 0.05

def performance_incentive : ℚ := 0.05

def maintenance_cost : ℚ := 50 * (round_trip_distance / 500)

def fuel_cost_rate : ℚ := 0.15

theorem jenna_earnings :
  let reduced_bonus := bonus * (1 - weather_reduction)
  let earnings_with_bonus := base_earnings + reduced_bonus
  let earnings_with_incentive := earnings_with_bonus * (1 + performance_incentive)
  let earnings_after_rest_stop := earnings_with_incentive * (1 - rest_stop_reduction)
  let fuel_cost := earnings_after_rest_stop * fuel_cost_rate
  let net_earnings := earnings_after_rest_stop - maintenance_cost - fuel_cost
  net_earnings = 380 := by sorry

end jenna_earnings_l2913_291383


namespace lcm_of_21_and_12_l2913_291349

theorem lcm_of_21_and_12 (h : Nat.gcd 21 12 = 6) : Nat.lcm 21 12 = 42 := by
  sorry

end lcm_of_21_and_12_l2913_291349


namespace store_sales_total_l2913_291370

/-- The total money made from selling DVD players and a washing machine -/
def total_money (dvd_price : ℕ) (dvd_quantity : ℕ) (washing_machine_price : ℕ) : ℕ :=
  dvd_price * dvd_quantity + washing_machine_price

/-- Theorem: The total money made from selling 8 DVD players at 240 yuan each
    and one washing machine at 898 yuan is equal to 240 * 8 + 898 yuan -/
theorem store_sales_total :
  total_money 240 8 898 = 240 * 8 + 898 := by
  sorry

end store_sales_total_l2913_291370

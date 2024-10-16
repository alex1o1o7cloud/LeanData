import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2919_291968

theorem equation_solution : ∃ x : ℝ, 300 * x + (12 + 4) * (1 / 8) = 602 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2919_291968


namespace NUMINAMATH_CALUDE_milk_for_flour_batch_l2919_291998

/-- Given that 60 mL of milk is used for every 300 mL of flour,
    prove that 300 mL of milk is needed for 1500 mL of flour. -/
theorem milk_for_flour_batch (milk_per_portion : ℝ) (flour_per_portion : ℝ) 
    (total_flour : ℝ) (h1 : milk_per_portion = 60) 
    (h2 : flour_per_portion = 300) (h3 : total_flour = 1500) : 
    (total_flour / flour_per_portion) * milk_per_portion = 300 :=
by sorry

end NUMINAMATH_CALUDE_milk_for_flour_batch_l2919_291998


namespace NUMINAMATH_CALUDE_percentage_of_120_to_80_l2919_291944

theorem percentage_of_120_to_80 : 
  (120 : ℝ) / 80 * 100 = 150 := by sorry

end NUMINAMATH_CALUDE_percentage_of_120_to_80_l2919_291944


namespace NUMINAMATH_CALUDE_common_factor_of_polynomial_l2919_291910

/-- The common factor of the polynomial 3ma^2 - 6mab is 3ma -/
theorem common_factor_of_polynomial (m a b : ℤ) :
  ∃ (k₁ k₂ : ℤ), 3 * m * a^2 - 6 * m * a * b = 3 * m * a * (k₁ * a + k₂ * b) :=
sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomial_l2919_291910


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2919_291995

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | -1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2919_291995


namespace NUMINAMATH_CALUDE_euler_most_prolific_l2919_291925

/-- Represents a mathematician -/
structure Mathematician where
  name : String
  country : String
  published_volumes : ℕ

/-- The Swiss Society of Natural Sciences -/
def SwissSocietyOfNaturalSciences : Set Mathematician := sorry

/-- Leonhard Euler -/
def euler : Mathematician := {
  name := "Leonhard Euler",
  country := "Switzerland",
  published_volumes := 76  -- More than 75 volumes
}

/-- Predicate for being the most prolific mathematician -/
def most_prolific (m : Mathematician) : Prop :=
  ∀ n : Mathematician, n.published_volumes ≤ m.published_volumes

theorem euler_most_prolific :
  euler ∈ SwissSocietyOfNaturalSciences →
  euler.country = "Switzerland" →
  euler.published_volumes > 75 →
  most_prolific euler :=
sorry

end NUMINAMATH_CALUDE_euler_most_prolific_l2919_291925


namespace NUMINAMATH_CALUDE_russom_subway_tickets_l2919_291971

theorem russom_subway_tickets (bus_tickets : ℕ) (max_envelopes : ℕ) (subway_tickets : ℕ) : 
  bus_tickets = 18 →
  max_envelopes = 6 →
  bus_tickets % max_envelopes = 0 →
  subway_tickets % max_envelopes = 0 →
  subway_tickets > 0 →
  ∀ n : ℕ, n < subway_tickets → n % max_envelopes ≠ 0 ∨ n = 0 →
  subway_tickets = 6 :=
by sorry

end NUMINAMATH_CALUDE_russom_subway_tickets_l2919_291971


namespace NUMINAMATH_CALUDE_sons_age_l2919_291952

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2919_291952


namespace NUMINAMATH_CALUDE_no_solution_implies_a_equals_negative_two_l2919_291913

theorem no_solution_implies_a_equals_negative_two (a : ℝ) : 
  (∀ x y : ℝ, ¬(a * x + 2 * y = a + 2 ∧ 2 * x + a * y = 2 * a)) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_equals_negative_two_l2919_291913


namespace NUMINAMATH_CALUDE_equation_describes_ellipse_l2919_291979

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 12

-- Define the two fixed points
def point1 : ℝ × ℝ := (0, 2)
def point2 : ℝ × ℝ := (6, -4)

-- Theorem stating that the equation describes an ellipse
theorem equation_describes_ellipse :
  ∀ x y : ℝ, equation x y ↔ 
    (∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧
      Real.sqrt ((p.1 - point1.1)^2 + (p.2 - point1.2)^2) +
      Real.sqrt ((p.1 - point2.1)^2 + (p.2 - point2.2)^2) = 12) :=
by sorry

end NUMINAMATH_CALUDE_equation_describes_ellipse_l2919_291979


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2919_291965

-- Define the given line
def given_line (x y : ℝ) : Prop := 2*x + y - 5 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, 3)

-- Define the new line
def new_line (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  (∀ x y : ℝ, given_line x y → (new_line x y → ¬given_line x y)) ∧
  new_line point_A.1 point_A.2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2919_291965


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2919_291934

/-- The speed of the first train -/
def speed_first_train : ℝ := 20

/-- The distance between stations P and Q -/
def distance_PQ : ℝ := 110

/-- The speed of the second train -/
def speed_second_train : ℝ := 25

/-- The time the first train travels before meeting -/
def time_first_train : ℝ := 3

/-- The time the second train travels before meeting -/
def time_second_train : ℝ := 2

theorem train_speed_calculation :
  speed_first_train * time_first_train + speed_second_train * time_second_train = distance_PQ :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2919_291934


namespace NUMINAMATH_CALUDE_inequalities_hold_l2919_291955

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a^2 + b^2 ≥ 2 ∧ 1/a + 1/b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l2919_291955


namespace NUMINAMATH_CALUDE_codes_lost_calculation_l2919_291960

/-- The number of digits in each code -/
def code_length : ℕ := 4

/-- The base of the number system (decimal) -/
def base : ℕ := 10

/-- The total number of possible codes with leading zeros -/
def total_codes : ℕ := base ^ code_length

/-- The number of possible codes without leading zeros -/
def codes_without_leading_zeros : ℕ := (base - 1) * (base ^ (code_length - 1))

/-- The number of codes lost when disallowing leading zeros -/
def codes_lost : ℕ := total_codes - codes_without_leading_zeros

theorem codes_lost_calculation :
  codes_lost = 1000 :=
sorry

end NUMINAMATH_CALUDE_codes_lost_calculation_l2919_291960


namespace NUMINAMATH_CALUDE_polyhedron_sum_l2919_291947

structure Polyhedron where
  faces : ℕ
  triangles : ℕ
  pentagons : ℕ
  T : ℕ
  P : ℕ
  V : ℕ
  faces_sum : faces = triangles + pentagons
  faces_20 : faces = 20
  triangles_twice_pentagons : triangles = 2 * pentagons
  euler : V - ((3 * triangles + 5 * pentagons) / 2) + faces = 2

def vertex_sum (poly : Polyhedron) : ℕ := 100 * poly.P + 10 * poly.T + poly.V

theorem polyhedron_sum (poly : Polyhedron) (h1 : poly.T = 2) (h2 : poly.P = 2) : 
  vertex_sum poly = 238 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_sum_l2919_291947


namespace NUMINAMATH_CALUDE_tangent_curve_relation_l2919_291901

noncomputable def tangent_line (a : ℝ) (x : ℝ) : ℝ := x + a

noncomputable def curve (b : ℝ) (x : ℝ) : ℝ := Real.exp (x - 1) - b + 1

theorem tangent_curve_relation (a b : ℝ) :
  (∃ x₀ : ℝ, tangent_line a x₀ = curve b x₀ ∧ 
    (deriv (tangent_line a)) x₀ = (deriv (curve b)) x₀) →
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_curve_relation_l2919_291901


namespace NUMINAMATH_CALUDE_point_transformation_l2919_291990

/-- Rotate a point (x,y) by 180° around (h,k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2*h - x, 2*k - y)

/-- Reflect a point (x,y) about the line y = -x -/
def reflectAboutNegativeX (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

theorem point_transformation (a b : ℝ) :
  let p₁ := rotate180 a b 2 4
  let p₂ := reflectAboutNegativeX p₁.1 p₁.2
  p₂ = (-1, 4) → a - b = -9 := by
sorry

end NUMINAMATH_CALUDE_point_transformation_l2919_291990


namespace NUMINAMATH_CALUDE_max_intersection_points_circle_triangle_l2919_291939

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A triangle in a plane --/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The number of intersection points between a circle and a line segment --/
def intersectionPointsCircleLine (c : Circle) (p1 p2 : ℝ × ℝ) : ℕ := sorry

/-- The number of intersection points between a circle and a triangle --/
def intersectionPointsCircleTriangle (c : Circle) (t : Triangle) : ℕ :=
  (intersectionPointsCircleLine c (t.vertices 0) (t.vertices 1)) +
  (intersectionPointsCircleLine c (t.vertices 1) (t.vertices 2)) +
  (intersectionPointsCircleLine c (t.vertices 2) (t.vertices 0))

/-- The maximum number of intersection points between a circle and a triangle is 6 --/
theorem max_intersection_points_circle_triangle :
  ∃ (c : Circle) (t : Triangle), 
    (∀ (c' : Circle) (t' : Triangle), intersectionPointsCircleTriangle c' t' ≤ 6) ∧
    intersectionPointsCircleTriangle c t = 6 :=
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_circle_triangle_l2919_291939


namespace NUMINAMATH_CALUDE_midpoint_distance_theorem_l2919_291978

theorem midpoint_distance_theorem (t : ℝ) : 
  let P : ℝ × ℝ := (2*t + 3, t - 5)
  let Q : ℝ × ℝ := (t - 1, 2*t + 4)
  let M : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  ((M.1 - P.1)^2 + (M.2 - P.2)^2) = 3*t^2/4 →
  t = 29 ∨ t = -3 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_distance_theorem_l2919_291978


namespace NUMINAMATH_CALUDE_positive_integer_square_minus_five_times_zero_l2919_291951

theorem positive_integer_square_minus_five_times_zero (w : ℕ+) 
  (h : w.val ^ 2 - 5 * w.val = 0) : w.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_square_minus_five_times_zero_l2919_291951


namespace NUMINAMATH_CALUDE_main_divisors_equal_implies_equal_l2919_291997

/-- The two largest proper divisors of a composite natural number -/
def main_divisors (n : ℕ) : Set ℕ :=
  {d ∈ Nat.divisors n | d ≠ n ∧ d ≠ 1 ∧ ∀ k ∈ Nat.divisors n, k ≠ n → k ≠ 1 → d ≥ k}

/-- A natural number is composite if it has at least one proper divisor -/
def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem main_divisors_equal_implies_equal (a b : ℕ) 
  (ha : is_composite a) (hb : is_composite b) 
  (h : main_divisors a = main_divisors b) : 
  a = b :=
sorry

end NUMINAMATH_CALUDE_main_divisors_equal_implies_equal_l2919_291997


namespace NUMINAMATH_CALUDE_prize_probability_l2919_291950

theorem prize_probability (odds_favorable : ℕ) (odds_unfavorable : ℕ) 
  (h_odds : odds_favorable = 5 ∧ odds_unfavorable = 6) :
  let total_outcomes := odds_favorable + odds_unfavorable
  let prob_not_prize := odds_unfavorable / total_outcomes
  (prob_not_prize ^ 2 : ℚ) = 36 / 121 :=
by sorry

end NUMINAMATH_CALUDE_prize_probability_l2919_291950


namespace NUMINAMATH_CALUDE_box_of_balls_l2919_291935

theorem box_of_balls (blue : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) : 
  blue = 6 →
  red = 4 →
  green = 3 * blue →
  yellow = 2 * red →
  blue + red + green + yellow = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_box_of_balls_l2919_291935


namespace NUMINAMATH_CALUDE_seats_per_bus_is_60_field_trip_problem_l2919_291948

/-- Represents the field trip scenario -/
structure FieldTrip where
  total_students : ℕ
  num_buses : ℕ
  all_accommodated : Bool

/-- Calculates the number of seats per bus -/
def seats_per_bus (trip : FieldTrip) : ℕ :=
  trip.total_students / trip.num_buses

/-- Theorem stating that the number of seats per bus is 60 -/
theorem seats_per_bus_is_60 (trip : FieldTrip) 
  (h1 : trip.total_students = 180)
  (h2 : trip.num_buses = 3)
  (h3 : trip.all_accommodated = true) : 
  seats_per_bus trip = 60 := by
  sorry

/-- Main theorem proving the field trip problem -/
theorem field_trip_problem : 
  ∃ (trip : FieldTrip), seats_per_bus trip = 60 ∧ 
    trip.total_students = 180 ∧ 
    trip.num_buses = 3 ∧ 
    trip.all_accommodated = true := by
  sorry

end NUMINAMATH_CALUDE_seats_per_bus_is_60_field_trip_problem_l2919_291948


namespace NUMINAMATH_CALUDE_triangle_shape_l2919_291988

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = π) →
  (a^2 + b^2 + c^2 = 2 * Real.sqrt 3 * a * b * Real.sin C) →
  (a = b ∧ b = c ∧ A = B ∧ B = C ∧ C = π/3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_shape_l2919_291988


namespace NUMINAMATH_CALUDE_investment_period_l2919_291983

theorem investment_period (emma_investment briana_investment : ℝ)
  (emma_yield briana_yield : ℝ) (difference : ℝ) :
  emma_investment = 300 →
  briana_investment = 500 →
  emma_yield = 0.15 →
  briana_yield = 0.10 →
  difference = 10 →
  ∃ t : ℝ, t = 2 ∧ 
    t * (briana_investment * briana_yield - emma_investment * emma_yield) = difference :=
by sorry

end NUMINAMATH_CALUDE_investment_period_l2919_291983


namespace NUMINAMATH_CALUDE_owls_on_fence_l2919_291949

/-- The number of owls that joined the fence -/
def owls_joined (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem owls_on_fence (initial : ℕ) (final : ℕ) 
  (h_initial : initial = 3) 
  (h_final : final = 5) : 
  owls_joined initial final = 2 := by
  sorry

end NUMINAMATH_CALUDE_owls_on_fence_l2919_291949


namespace NUMINAMATH_CALUDE_parallelepiped_with_extensions_volume_l2919_291996

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a parallelepiped -/
def volume (p : Parallelepiped) : ℝ :=
  p.length * p.width * p.height

/-- Calculates the volume of extensions from all faces -/
def extension_volume (p : Parallelepiped) : ℝ :=
  2 * (p.length * p.width + p.width * p.height + p.length * p.height)

/-- The main theorem to prove -/
theorem parallelepiped_with_extensions_volume 
  (p : Parallelepiped) 
  (h1 : p.length = 2) 
  (h2 : p.width = 3) 
  (h3 : p.height = 4) :
  volume p + extension_volume p = 76 := by
  sorry

#check parallelepiped_with_extensions_volume

end NUMINAMATH_CALUDE_parallelepiped_with_extensions_volume_l2919_291996


namespace NUMINAMATH_CALUDE_problem_solution_l2919_291938

noncomputable def g (θ : Real) (x : Real) : Real := x * Real.sin θ - Real.log x - Real.sin θ

noncomputable def f (θ : Real) (x : Real) : Real := g θ x + (2*x - 1) / (2*x^2)

theorem problem_solution (θ : Real) (h1 : θ ∈ Set.Ioo 0 Real.pi) 
  (h2 : ∀ x ≥ 1, Monotone (g θ)) : 
  (θ = Real.pi/2) ∧ 
  (∀ x ∈ Set.Icc 1 2, f θ x > (deriv (f θ)) x + 1/2) ∧
  (∀ k > 1, ∃ x > 0, Real.exp x - x - 1 < k * g θ (x+1)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2919_291938


namespace NUMINAMATH_CALUDE_specific_parallelogram_area_and_height_l2919_291932

/-- Represents a parallelogram with given properties -/
structure Parallelogram where
  angle : ℝ  -- One angle of the parallelogram in degrees
  side1 : ℝ  -- Length of one side
  side2 : ℝ  -- Length of the adjacent side
  extension : ℝ  -- Length of extension beyond the vertex

/-- Calculates the area and height of a parallelogram with specific properties -/
def parallelogram_area_and_height (p : Parallelogram) : ℝ × ℝ :=
  sorry

/-- Theorem stating the area and height of a specific parallelogram -/
theorem specific_parallelogram_area_and_height :
  let p : Parallelogram := ⟨150, 10, 18, 2⟩
  let (area, height) := parallelogram_area_and_height p
  area = 36 * Real.sqrt 3 ∧ height = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_specific_parallelogram_area_and_height_l2919_291932


namespace NUMINAMATH_CALUDE_total_spent_is_520_l2919_291984

/-- Shopping expenses for Lisa and Carly -/
def shopping_expenses (T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ) : Prop :=
  T_L = 40 ∧
  J_L = T_L / 2 ∧
  C_L = 2 * T_L ∧
  S_L = 3 * J_L ∧
  T_C = T_L / 4 ∧
  J_C = 3 * J_L ∧
  C_C = C_L / 2 ∧
  S_C = S_L ∧
  D_C = 2 * S_C ∧
  A_C = J_C / 2

/-- The total amount spent by Lisa and Carly -/
def total_spent (T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ) : ℝ :=
  T_L + J_L + C_L + S_L + T_C + J_C + C_C + S_C + D_C + A_C

/-- Theorem stating that the total amount spent is $520 -/
theorem total_spent_is_520 :
  ∀ T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ,
  shopping_expenses T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C →
  total_spent T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C = 520 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_is_520_l2919_291984


namespace NUMINAMATH_CALUDE_product_sum_relation_l2919_291993

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 11 → b = 7 → b - a = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l2919_291993


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2919_291958

/-- The line x + y - 1 = 0 does not pass through the third quadrant -/
theorem line_not_in_third_quadrant :
  ∀ x y : ℝ, x + y - 1 = 0 → ¬(x < 0 ∧ y < 0) := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2919_291958


namespace NUMINAMATH_CALUDE_winning_candidate_vote_percentage_l2919_291986

theorem winning_candidate_vote_percentage 
  (total_members : ℕ) 
  (votes_cast : ℕ) 
  (winning_percentage : ℚ) 
  (h1 : total_members = 1600)
  (h2 : votes_cast = 525)
  (h3 : winning_percentage = 60 / 100) : 
  (((votes_cast : ℚ) * winning_percentage) / (total_members : ℚ)) * 100 = 19.6875 := by
  sorry

end NUMINAMATH_CALUDE_winning_candidate_vote_percentage_l2919_291986


namespace NUMINAMATH_CALUDE_campground_distance_l2919_291920

/-- The distance traveled by Sue's family to the campground -/
def distance_to_campground (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: The distance to the campground is 300 miles -/
theorem campground_distance :
  distance_to_campground 60 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_campground_distance_l2919_291920


namespace NUMINAMATH_CALUDE_library_repacking_l2919_291918

/-- Given a number of boxes and books per box, calculate the number of books left over when repacking into new boxes with a different number of books per box. -/
def books_left_over (initial_boxes : ℕ) (initial_books_per_box : ℕ) (new_books_per_box : ℕ) : ℕ :=
  let total_books := initial_boxes * initial_books_per_box
  total_books % new_books_per_box

/-- Prove that given 1575 boxes with 45 books each, when repacking into boxes of 50 books each, the number of books left over is 25. -/
theorem library_repacking : books_left_over 1575 45 50 = 25 := by
  sorry

end NUMINAMATH_CALUDE_library_repacking_l2919_291918


namespace NUMINAMATH_CALUDE_ap_special_condition_l2919_291930

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  first : ℝ
  diff : ℝ

/-- The nth term of an arithmetic progression -/
def nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.first + (n - 1 : ℝ) * ap.diff

theorem ap_special_condition (ap : ArithmeticProgression) :
  nthTerm ap 4 + nthTerm ap 20 = nthTerm ap 8 + nthTerm ap 15 + nthTerm ap 12 →
  ap.first = 10 * ap.diff := by
  sorry

end NUMINAMATH_CALUDE_ap_special_condition_l2919_291930


namespace NUMINAMATH_CALUDE_total_savings_is_40_l2919_291906

-- Define the number of coins each child has
def teagan_pennies : ℕ := 200
def rex_nickels : ℕ := 100
def toni_dimes : ℕ := 330

-- Define the conversion rates
def pennies_per_dollar : ℕ := 100
def nickels_per_dollar : ℕ := 20
def dimes_per_dollar : ℕ := 10

-- Define the total savings
def total_savings : ℚ :=
  (teagan_pennies : ℚ) / pennies_per_dollar +
  (rex_nickels : ℚ) / nickels_per_dollar +
  (toni_dimes : ℚ) / dimes_per_dollar

-- Theorem statement
theorem total_savings_is_40 : total_savings = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_is_40_l2919_291906


namespace NUMINAMATH_CALUDE_algebraic_expression_transformation_l2919_291942

theorem algebraic_expression_transformation (a b : ℝ) :
  (∀ x, x^2 - 6*x + b = (x - a)^2 - 1) → b - a = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_transformation_l2919_291942


namespace NUMINAMATH_CALUDE_inequality_range_l2919_291957

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * a * x + a - 2 < 0) ↔ a ∈ Set.Ioc (-8/5) 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l2919_291957


namespace NUMINAMATH_CALUDE_derivatives_verification_l2919_291956

theorem derivatives_verification :
  (∀ x : ℝ, deriv (λ x => x^2) x = 2 * x) ∧
  (∀ x : ℝ, deriv Real.sin x = Real.cos x) ∧
  (∀ x : ℝ, deriv (λ x => Real.exp (-x)) x = -Real.exp (-x)) ∧
  (∀ x : ℝ, x ≠ -1 → deriv (λ x => Real.log (x + 1)) x = 1 / (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_derivatives_verification_l2919_291956


namespace NUMINAMATH_CALUDE_fraction_transformation_l2919_291914

theorem fraction_transformation (x : ℤ) : 
  x = 437 → (537 - x : ℚ) / (463 + x) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l2919_291914


namespace NUMINAMATH_CALUDE_heart_ten_spade_probability_l2919_291987

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of tens in a standard deck -/
def NumTens : ℕ := 4

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Probability of drawing a specific sequence of cards -/
def SequenceProbability (firstCardProb : ℚ) (secondCardProb : ℚ) (thirdCardProb : ℚ) : ℚ :=
  firstCardProb * secondCardProb * thirdCardProb

theorem heart_ten_spade_probability :
  let probHeartNotTen := (NumHearts - 1) / StandardDeck
  let probTenAfterHeart := NumTens / (StandardDeck - 1)
  let probSpadeAfterHeartTen := NumSpades / (StandardDeck - 2)
  let probHeartTen := 1 / StandardDeck
  let probOtherTenAfterHeartTen := (NumTens - 1) / (StandardDeck - 1)
  
  SequenceProbability probHeartNotTen probTenAfterHeart probSpadeAfterHeartTen +
  SequenceProbability probHeartTen probOtherTenAfterHeartTen probSpadeAfterHeartTen = 63 / 107800 :=
by
  sorry

end NUMINAMATH_CALUDE_heart_ten_spade_probability_l2919_291987


namespace NUMINAMATH_CALUDE_power_function_value_l2919_291912

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = Real.sqrt 2 / 2 → f 9 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l2919_291912


namespace NUMINAMATH_CALUDE_integral_reciprocal_e_l2919_291909

open Real MeasureTheory

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem integral_reciprocal_e : ∫ x in Set.Icc (1/Real.exp 1) (Real.exp 1), f x = 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_e_l2919_291909


namespace NUMINAMATH_CALUDE_difference_of_squares_l2919_291929

theorem difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2919_291929


namespace NUMINAMATH_CALUDE_median_mean_difference_l2919_291917

structure ArticleData where
  frequencies : List (Nat × Nat)
  total_students : Nat
  sum_articles : Nat

def median (data : ArticleData) : Rat := 2

def mean (data : ArticleData) : Rat := data.sum_articles / data.total_students

theorem median_mean_difference (data : ArticleData) 
  (h1 : data.frequencies = [(0, 4), (1, 3), (2, 2), (3, 2), (4, 3), (5, 4)])
  (h2 : data.total_students = 18)
  (h3 : data.sum_articles = 45) :
  mean data - median data = 1/2 := by sorry

end NUMINAMATH_CALUDE_median_mean_difference_l2919_291917


namespace NUMINAMATH_CALUDE_divisibility_probability_l2919_291902

/-- The number of positive divisors of 10^99 -/
def total_divisors : ℕ := 10000

/-- The number of positive divisors of 10^99 that are multiples of 10^88 -/
def favorable_divisors : ℕ := 144

/-- The probability of a randomly chosen positive divisor of 10^99 being an integer multiple of 10^88 -/
def probability : ℚ := favorable_divisors / total_divisors

theorem divisibility_probability :
  probability = 9 / 625 :=
sorry

end NUMINAMATH_CALUDE_divisibility_probability_l2919_291902


namespace NUMINAMATH_CALUDE_function_value_at_ten_l2919_291903

theorem function_value_at_ten (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f (2*x + y) + 7*x*y + 3*y^2 = f (3*x - y) + 3*x^2 + 2) : 
  f 10 = -123 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_ten_l2919_291903


namespace NUMINAMATH_CALUDE_apple_consumption_duration_l2919_291931

theorem apple_consumption_duration (apples_per_box : ℕ) (num_boxes : ℕ) (num_people : ℕ) (apples_per_person_per_day : ℕ) :
  apples_per_box = 14 →
  num_boxes = 3 →
  num_people = 2 →
  apples_per_person_per_day = 1 →
  (apples_per_box * num_boxes) / (num_people * apples_per_person_per_day * 7) = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_consumption_duration_l2919_291931


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2919_291977

theorem simplify_polynomial (x : ℝ) : 
  3*x^3 + 4*x^2 + 5*x + 10 - (-6 + 3*x^3 - 2*x^2 + x) = 6*x^2 + 4*x + 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2919_291977


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l2919_291974

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem eighth_term_of_sequence (a₁ d : ℝ) :
  arithmeticSequence a₁ d 4 = 22 →
  arithmeticSequence a₁ d 6 = 46 →
  arithmeticSequence a₁ d 8 = 70 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l2919_291974


namespace NUMINAMATH_CALUDE_bill_split_correct_l2919_291937

/-- The number of people splitting the bill -/
def num_people : ℕ := 9

/-- The total bill amount in cents -/
def total_bill : ℕ := 51416

/-- The amount each person should pay in cents, rounded to the nearest cent -/
def amount_per_person : ℕ := 5713

/-- Theorem stating that the calculated amount per person is correct -/
theorem bill_split_correct : 
  (total_bill + num_people - 1) / num_people = amount_per_person :=
sorry

end NUMINAMATH_CALUDE_bill_split_correct_l2919_291937


namespace NUMINAMATH_CALUDE_value_range_of_f_l2919_291976

def f (x : ℝ) := 3 * x - 1

theorem value_range_of_f :
  Set.Icc (-16 : ℝ) 5 = Set.image f (Set.Ico (-5 : ℝ) 2) := by sorry

end NUMINAMATH_CALUDE_value_range_of_f_l2919_291976


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l2919_291927

theorem real_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (1 + i) / i
  (z.re : ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l2919_291927


namespace NUMINAMATH_CALUDE_triangle_and_division_counts_l2919_291972

/-- The number of non-congruent triangles formed by m equally spaced points on a circle -/
def num_triangles (m : ℕ) : ℕ :=
  let k := m / 6
  match m % 6 with
  | 0 => 3*k^2 - 3*k + 1
  | 1 => 3*k^2 - 2*k
  | 2 => 3*k^2 - k
  | 3 => 3*k^2
  | 4 => 3*k^2 + k
  | 5 => 3*k^2 + 2*k
  | _ => 0  -- This case should never occur

/-- The number of ways to divide m identical items into 3 groups -/
def num_divisions (m : ℕ) : ℕ :=
  let k := m / 6
  match m % 6 with
  | 0 => 3*k^2
  | 1 => 3*k^2 + k
  | 2 => 3*k^2 + 2*k
  | 3 => 3*k^2 + 3*k + 1
  | 4 => 3*k^2 + 4*k + 1
  | 5 => 3*k^2 + 5*k + 2
  | _ => 0  -- This case should never occur

theorem triangle_and_division_counts (m : ℕ) (h : m ≥ 3) :
  (num_triangles m = num_triangles m) ∧ (num_divisions m = num_divisions m) :=
sorry

end NUMINAMATH_CALUDE_triangle_and_division_counts_l2919_291972


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_n_199999_satisfies_condition_n_199999_is_largest_l2919_291973

theorem largest_n_divisible_by_seven (n : ℕ) : 
  (n < 200000 ∧ 
   (8 * (n - 3)^5 - 2 * n^2 + 18 * n - 36) % 7 = 0) →
  n ≤ 199999 :=
by sorry

theorem n_199999_satisfies_condition : 
  (8 * (199999 - 3)^5 - 2 * 199999^2 + 18 * 199999 - 36) % 7 = 0 :=
by sorry

theorem n_199999_is_largest : 
  ∀ m : ℕ, m < 200000 ∧ 
  (8 * (m - 3)^5 - 2 * m^2 + 18 * m - 36) % 7 = 0 →
  m ≤ 199999 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_n_199999_satisfies_condition_n_199999_is_largest_l2919_291973


namespace NUMINAMATH_CALUDE_exact_calculation_equals_rounded_l2919_291964

def round_to_nearest_hundred (x : ℤ) : ℤ :=
  if x % 100 < 50 then x - (x % 100) else x + (100 - (x % 100))

theorem exact_calculation_equals_rounded : round_to_nearest_hundred (63 + 48 - 21) = 100 := by
  sorry

end NUMINAMATH_CALUDE_exact_calculation_equals_rounded_l2919_291964


namespace NUMINAMATH_CALUDE_number_of_tablets_A_l2919_291967

/-- Given a box with tablets of medicine A and B, this theorem proves
    the number of tablets of medicine A, given certain conditions. -/
theorem number_of_tablets_A (num_B : ℕ) (min_extract : ℕ) : 
  num_B = 16 → min_extract = 18 → ∃ num_A : ℕ, num_A = 3 ∧ 
  (∀ k : ℕ, k < min_extract → 
    (k ≤ num_A + num_B ∧ (k < num_A + num_B - 1 ∨ k < num_A + num_B - num_A + 1 ∨ k < num_B + 2))) :=
by sorry

end NUMINAMATH_CALUDE_number_of_tablets_A_l2919_291967


namespace NUMINAMATH_CALUDE_common_factor_of_polynomial_l2919_291915

theorem common_factor_of_polynomial (x : ℝ) :
  ∃ (k : ℝ), 2*x^2 - 8*x = 2*x*k :=
by
  sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomial_l2919_291915


namespace NUMINAMATH_CALUDE_max_large_sculptures_l2919_291900

theorem max_large_sculptures (total_blocks : ℕ) (small_sculptures large_sculptures : ℕ) : 
  total_blocks = 30 →
  small_sculptures > large_sculptures →
  small_sculptures + 3 * large_sculptures + (small_sculptures + large_sculptures) / 2 ≤ total_blocks →
  large_sculptures ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_large_sculptures_l2919_291900


namespace NUMINAMATH_CALUDE_copenhagen_aarhus_distance_l2919_291970

/-- The distance between two city centers with a detour -/
def distance_with_detour (map_distance : ℝ) (scale : ℝ) (detour_increase : ℝ) : ℝ :=
  map_distance * scale * (1 + detour_increase)

/-- Theorem: The distance between Copenhagen and Aarhus is 420 km -/
theorem copenhagen_aarhus_distance :
  distance_with_detour 35 10 0.2 = 420 := by
  sorry

end NUMINAMATH_CALUDE_copenhagen_aarhus_distance_l2919_291970


namespace NUMINAMATH_CALUDE_ratio_fraction_equality_l2919_291945

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_equality_l2919_291945


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identities_l2919_291975

theorem triangle_trigonometric_identities (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →
  (a = 2 * Real.sin A) →
  (b = 2 * Real.sin B) →
  (c = 2 * Real.sin C) →
  (((a^2 * Real.sin (B - C)) / (Real.sin B * Real.sin C) +
    (b^2 * Real.sin (C - A)) / (Real.sin C * Real.sin A) +
    (c^2 * Real.sin (A - B)) / (Real.sin A * Real.sin B) = 0) ∧
   ((a^2 * Real.sin (B - C)) / (Real.sin B + Real.sin C) +
    (b^2 * Real.sin (C - A)) / (Real.sin C + Real.sin A) +
    (c^2 * Real.sin (A - B)) / (Real.sin A + Real.sin B) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identities_l2919_291975


namespace NUMINAMATH_CALUDE_class_average_l2919_291922

theorem class_average (total_students : ℕ) (top_scorers : ℕ) (zero_scorers : ℕ) (top_score : ℕ) (rest_average : ℕ) 
  (h1 : total_students = 20)
  (h2 : top_scorers = 2)
  (h3 : zero_scorers = 3)
  (h4 : top_score = 100)
  (h5 : rest_average = 40) :
  (top_scorers * top_score + zero_scorers * 0 + (total_students - top_scorers - zero_scorers) * rest_average) / total_students = 40 := by
  sorry

#check class_average

end NUMINAMATH_CALUDE_class_average_l2919_291922


namespace NUMINAMATH_CALUDE_retailer_markup_percentage_l2919_291911

/-- Proves that a retailer who marks up goods by x%, offers a 15% discount, 
    and makes 27.5% profit, must have marked up the goods by 50% --/
theorem retailer_markup_percentage 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (actual_profit_percentage : ℝ)
  (h1 : discount_percentage = 15)
  (h2 : actual_profit_percentage = 27.5)
  (h3 : cost_price > 0)
  (h4 : markup_percentage > 0)
  : 
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  selling_price = cost_price * (1 + actual_profit_percentage / 100) →
  markup_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_retailer_markup_percentage_l2919_291911


namespace NUMINAMATH_CALUDE_ratio_problem_l2919_291946

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 4) : 
  (a + b) / (b + c) = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2919_291946


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2919_291962

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℚ)
  (h_arith : is_arithmetic_sequence a)
  (h_2 : a 2 = 1)
  (h_8 : a 8 = 2 * a 6 + a 4) :
  a 5 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2919_291962


namespace NUMINAMATH_CALUDE_hiking_time_theorem_l2919_291928

/-- Calculates the total time for a hiker to return to the starting point given their hiking rate and distances. -/
def total_hiking_time (rate : ℝ) (initial_distance : ℝ) (total_distance : ℝ) : ℝ :=
  let additional_distance := total_distance - initial_distance
  let time_additional := additional_distance * rate
  let time_return := total_distance * rate
  time_additional + time_return

/-- Theorem stating that under given conditions, the total hiking time is 40 minutes. -/
theorem hiking_time_theorem (rate : ℝ) (initial_distance : ℝ) (total_distance : ℝ) :
  rate = 12 →
  initial_distance = 2.75 →
  total_distance = 3.041666666666667 →
  total_hiking_time rate initial_distance total_distance = 40 := by
  sorry

#eval total_hiking_time 12 2.75 3.041666666666667

end NUMINAMATH_CALUDE_hiking_time_theorem_l2919_291928


namespace NUMINAMATH_CALUDE_water_added_to_alcohol_solution_l2919_291981

/-- Proves that adding 5 liters of water to a 15-liter solution with 26% alcohol 
    results in a new solution with 19.5% alcohol -/
theorem water_added_to_alcohol_solution :
  let initial_volume : ℝ := 15
  let initial_alcohol_percentage : ℝ := 0.26
  let water_added : ℝ := 5
  let final_alcohol_percentage : ℝ := 0.195
  let initial_alcohol_volume := initial_volume * initial_alcohol_percentage
  let final_volume := initial_volume + water_added
  initial_alcohol_volume / final_volume = final_alcohol_percentage := by
  sorry


end NUMINAMATH_CALUDE_water_added_to_alcohol_solution_l2919_291981


namespace NUMINAMATH_CALUDE_inverse_variation_result_l2919_291907

/-- A function representing the inverse variation of 7y with the cube of x -/
def inverse_variation (x y : ℝ) : Prop :=
  ∃ k : ℝ, 7 * y = k / (x ^ 3)

/-- The theorem stating that given the inverse variation and initial condition,
    when x = 4, y = 1 -/
theorem inverse_variation_result :
  (∃ y₀ : ℝ, inverse_variation 2 y₀ ∧ y₀ = 8) →
  (∃ y : ℝ, inverse_variation 4 y ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_result_l2919_291907


namespace NUMINAMATH_CALUDE_abc_inequality_l2919_291908

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a / (a^3 - a^2 + 3)) + (b / (b^3 - b^2 + 3)) + (c / (c^3 - c^2 + 3)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2919_291908


namespace NUMINAMATH_CALUDE_periodic_function_l2919_291916

def isPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x

theorem periodic_function (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f (x + 1) + f (x - 1) = Real.sqrt 2 * f x) : 
  isPeriodic f := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_l2919_291916


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2919_291926

theorem geometric_sequence_problem (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 280) (h₂ : a₂ > 0) (h₃ : a₃ = 90 / 56) 
  (h₄ : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) : a₂ = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2919_291926


namespace NUMINAMATH_CALUDE_prob_same_foot_is_three_sevenths_l2919_291985

/-- The number of pairs of shoes in the cabinet -/
def num_pairs : ℕ := 4

/-- The total number of shoes in the cabinet -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes selected -/
def selected_shoes : ℕ := 2

/-- The number of ways to select 2 shoes out of the total shoes -/
def total_selections : ℕ := Nat.choose total_shoes selected_shoes

/-- The number of ways to select 2 shoes from the same foot -/
def same_foot_selections : ℕ := 2 * Nat.choose num_pairs selected_shoes

/-- The probability of selecting two shoes from the same foot -/
def prob_same_foot : ℚ := same_foot_selections / total_selections

theorem prob_same_foot_is_three_sevenths :
  prob_same_foot = 3 / 7 := by sorry

end NUMINAMATH_CALUDE_prob_same_foot_is_three_sevenths_l2919_291985


namespace NUMINAMATH_CALUDE_milly_study_time_l2919_291961

/-- Calculates the total study time for Milly given her homework durations. -/
theorem milly_study_time (math_time : ℕ) (math_time_eq : math_time = 60) :
  let geography_time := math_time / 2
  let science_time := (math_time + geography_time) / 2
  math_time + geography_time + science_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_milly_study_time_l2919_291961


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_attained_l2919_291989

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 5 * y < 75) :
  x * y * (75 - 2 * x - 5 * y) ≤ 1562.5 := by
  sorry

theorem max_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + 5 * y < 75 ∧
  x * y * (75 - 2 * x - 5 * y) > 1562.5 - ε := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_attained_l2919_291989


namespace NUMINAMATH_CALUDE_fruit_packing_lcm_l2919_291943

theorem fruit_packing_lcm : Nat.lcm 18 (Nat.lcm 9 (Nat.lcm 12 6)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_fruit_packing_lcm_l2919_291943


namespace NUMINAMATH_CALUDE_sharp_value_theorem_l2919_291923

/-- Define the function # -/
def sharp (k : ℚ) (p : ℚ) : ℚ := k * p + 20

/-- Main theorem -/
theorem sharp_value_theorem :
  ∀ k : ℚ, 
  (sharp k (sharp k (sharp k 18)) = -4) → 
  k = -4/3 := by
sorry

end NUMINAMATH_CALUDE_sharp_value_theorem_l2919_291923


namespace NUMINAMATH_CALUDE_largest_band_size_l2919_291969

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- The total number of band members --/
def totalMembers (f : BandFormation) : ℕ := f.rows * f.membersPerRow

/-- Conditions for the band formations --/
def validFormations (original new : BandFormation) (total : ℕ) : Prop :=
  total < 100 ∧
  totalMembers original + 4 = total ∧
  totalMembers new = total ∧
  new.membersPerRow = original.membersPerRow + 2 ∧
  new.rows + 3 = original.rows

/-- The theorem stating that the largest possible number of band members is 88 --/
theorem largest_band_size :
  ∀ original new : BandFormation,
  ∀ total : ℕ,
  validFormations original new total →
  total ≤ 88 :=
sorry

end NUMINAMATH_CALUDE_largest_band_size_l2919_291969


namespace NUMINAMATH_CALUDE_rod_cutting_l2919_291953

theorem rod_cutting (rod_length_m : ℕ) (piece_length_cm : ℕ) : 
  rod_length_m = 17 → piece_length_cm = 85 → (rod_length_m * 100) / piece_length_cm = 20 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l2919_291953


namespace NUMINAMATH_CALUDE_triangle_properties_l2919_291904

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  (t.b - t.c)^2 = t.a^2 - t.b * t.c ∧
  t.a = 3 ∧
  Real.sin t.C = 2 * Real.sin t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.A = π / 2 ∧ (1/2 * t.b * t.c * Real.sin t.A) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2919_291904


namespace NUMINAMATH_CALUDE_pen_average_price_l2919_291982

/-- Given the purchase of pens and pencils with specific quantities and prices,
    prove that the average price of a pen is $12. -/
theorem pen_average_price
  (num_pens : ℕ)
  (num_pencils : ℕ)
  (total_cost : ℚ)
  (pencil_avg_price : ℚ)
  (h1 : num_pens = 30)
  (h2 : num_pencils = 75)
  (h3 : total_cost = 510)
  (h4 : pencil_avg_price = 2) :
  (total_cost - num_pencils * pencil_avg_price) / num_pens = 12 :=
by sorry

end NUMINAMATH_CALUDE_pen_average_price_l2919_291982


namespace NUMINAMATH_CALUDE_third_job_hourly_rate_l2919_291905

-- Define the problem parameters
def total_earnings : ℝ := 430
def first_job_hours : ℝ := 15
def first_job_rate : ℝ := 8
def second_job_sales : ℝ := 1000
def second_job_commission_rate : ℝ := 0.1
def third_job_hours : ℝ := 12
def tax_deduction : ℝ := 50

-- Define the theorem
theorem third_job_hourly_rate :
  let first_job_earnings := first_job_hours * first_job_rate
  let second_job_earnings := second_job_sales * second_job_commission_rate
  let combined_wages := first_job_earnings + second_job_earnings
  let combined_wages_after_tax := combined_wages - tax_deduction
  let third_job_earnings := total_earnings - combined_wages_after_tax
  third_job_earnings / third_job_hours = 21.67 := by
  sorry

end NUMINAMATH_CALUDE_third_job_hourly_rate_l2919_291905


namespace NUMINAMATH_CALUDE_solution_range_l2919_291966

theorem solution_range (x : ℝ) :
  (5 * x - 8 > 12 - 2 * x) ∧ (|x - 1| ≤ 3) → (20 / 7 < x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l2919_291966


namespace NUMINAMATH_CALUDE_tan_P_is_two_one_l2919_291994

/-- Represents a right triangle PQR with altitude QS --/
structure RightTrianglePQR where
  -- Side lengths
  PQ : ℕ
  QR : ℕ
  PR : ℕ
  PS : ℕ
  -- PR = 3^5
  h_PR : PR = 3^5
  -- PS = 3^3
  h_PS : PS = 3^3
  -- Right angle at Q
  h_right_angle : PQ^2 + QR^2 = PR^2
  -- Altitude property
  h_altitude : PQ * PS = PR * QS

/-- The main theorem --/
theorem tan_P_is_two_one (t : RightTrianglePQR) : 
  (t.QR : ℚ) / t.PQ = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_P_is_two_one_l2919_291994


namespace NUMINAMATH_CALUDE_subtraction_correction_l2919_291919

theorem subtraction_correction (x : ℤ) : x - 63 = 24 → x - 36 = 51 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_correction_l2919_291919


namespace NUMINAMATH_CALUDE_average_of_five_quantities_l2919_291924

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 14) :
  (q1 + q2 + q3 + q4 + q5) / 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_quantities_l2919_291924


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2919_291963

theorem expand_and_simplify (x y : ℝ) : (x + 2*y) * (x - 2*y) - y * (3 - 4*y) = x^2 - 3*y := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2919_291963


namespace NUMINAMATH_CALUDE_min_value_expression_l2919_291980

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c) * (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2919_291980


namespace NUMINAMATH_CALUDE_prop_p_or_q_l2919_291921

theorem prop_p_or_q : 
  (∀ x : ℝ, x^2 + a*x + a^2 ≥ 0) ∨ (∃ x : ℝ, Real.sin x + Real.cos x = 2) :=
sorry

end NUMINAMATH_CALUDE_prop_p_or_q_l2919_291921


namespace NUMINAMATH_CALUDE_range_of_a_l2919_291936

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2919_291936


namespace NUMINAMATH_CALUDE_black_raisins_amount_l2919_291991

/-- The amount of yellow raisins added (in cups) -/
def yellow_raisins : ℝ := 0.3

/-- The total amount of raisins added (in cups) -/
def total_raisins : ℝ := 0.7

/-- The amount of black raisins added (in cups) -/
def black_raisins : ℝ := total_raisins - yellow_raisins

theorem black_raisins_amount : black_raisins = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_black_raisins_amount_l2919_291991


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_13_l2919_291954

theorem binomial_coefficient_19_13 
  (h1 : (20 : ℕ).choose 13 = 77520)
  (h2 : (20 : ℕ).choose 14 = 38760)
  (h3 : (18 : ℕ).choose 13 = 18564) :
  (19 : ℕ).choose 13 = 37128 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_13_l2919_291954


namespace NUMINAMATH_CALUDE_vector_magnitude_l2919_291992

/-- Given two vectors a and b in ℝ², where a is parallel to (a - b),
    prove that the magnitude of (a + b) is 3√5/2. -/
theorem vector_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), a = k • (a - b)) →
  ‖a + b‖ = 3 * Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2919_291992


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2919_291933

/-- A quadratic function satisfying specific conditions -/
def f (x : ℝ) : ℝ := 6 * x^2 - 4

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties :
  (f (-1) = 2) ∧
  (deriv f 0 = 0) ∧
  (∫ x in (0)..(1), f x = -2) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≥ -4) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≤ 2) ∧
  (f 0 = -4) ∧
  (f 1 = 2) ∧
  (f (-1) = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2919_291933


namespace NUMINAMATH_CALUDE_max_PXQ_value_l2919_291940

def is_two_digit_with_equal_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ (n / 10 = n % 10)

def is_one_digit (n : ℕ) : Prop :=
  0 < n ∧ n ≤ 9

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem max_PXQ_value :
  ∀ X XX PXQ : ℕ,
  is_two_digit_with_equal_digits XX →
  is_one_digit X →
  is_three_digit PXQ →
  XX * X = PXQ →
  PXQ ≤ 396 :=
sorry

end NUMINAMATH_CALUDE_max_PXQ_value_l2919_291940


namespace NUMINAMATH_CALUDE_money_difference_l2919_291941

/-- Given that Bob has $60, Phil has 1/3 of Bob's amount, and Jenna has twice Phil's amount,
    prove that the difference between Bob's and Jenna's amounts is $20. -/
theorem money_difference (bob_amount : ℕ) (phil_amount : ℕ) (jenna_amount : ℕ)
    (h1 : bob_amount = 60)
    (h2 : phil_amount = bob_amount / 3)
    (h3 : jenna_amount = 2 * phil_amount) :
    bob_amount - jenna_amount = 20 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l2919_291941


namespace NUMINAMATH_CALUDE_gcd_lcm_identity_l2919_291959

theorem gcd_lcm_identity (a b c : ℕ+) :
  (Nat.lcm (Nat.lcm a b) c)^2 / (Nat.lcm a b * Nat.lcm b c * Nat.lcm c a) =
  (Nat.gcd (Nat.gcd a b) c)^2 / (Nat.gcd a b * Nat.gcd b c * Nat.gcd c a) :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_identity_l2919_291959


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2919_291999

theorem complex_fraction_evaluation :
  2 - (1 / (2 + (1 / (2 - (1 / 3))))) = 21 / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2919_291999

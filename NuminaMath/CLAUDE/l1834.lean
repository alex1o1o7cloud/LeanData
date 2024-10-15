import Mathlib

namespace NUMINAMATH_CALUDE_lines_properties_l1834_183482

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x - y + 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 1 = 0

-- Theorem statement
theorem lines_properties (a : ℝ) :
  -- 1. The lines are always perpendicular
  (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ → l₂ a x₂ y₂ → (x₁ - x₂) * (y₁ - y₂) = 0) ∧
  -- 2. l₁ passes through (0,1) and l₂ passes through (-1,0)
  l₁ a 0 1 ∧ l₂ a (-1) 0 ∧
  -- 3. The maximum distance from the intersection point to the origin is √2
  (∃ x y : ℝ, l₁ a x y ∧ l₂ a x y ∧
    ∀ x' y' : ℝ, l₁ a x' y' → l₂ a x' y' → x'^2 + y'^2 ≤ 2) ∧
  (∃ a₀ x₀ y₀ : ℝ, l₁ a₀ x₀ y₀ ∧ l₂ a₀ x₀ y₀ ∧ x₀^2 + y₀^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_lines_properties_l1834_183482


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l1834_183419

-- Define the function f(x) = x³ - x² - x
def f (x : ℝ) := x^3 - x^2 - x

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1/3 : ℝ) 1, 
    ∀ y ∈ Set.Ioo (-1/3 : ℝ) 1, 
      x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l1834_183419


namespace NUMINAMATH_CALUDE_polynomial_equality_l1834_183476

theorem polynomial_equality (x : ℝ) :
  let k : ℝ → ℝ := λ x => -5*x^5 + 7*x^4 - 7*x^3 - x + 2
  5*x^5 + 3*x^3 + x + k x = 7*x^4 - 4*x^3 + 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1834_183476


namespace NUMINAMATH_CALUDE_min_distance_sum_l1834_183444

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 7 = 1

-- Define the left focus F
def left_focus : ℝ × ℝ := (-4, 0)

-- Define the fixed point A
def point_A : ℝ × ℝ := (1, 4)

-- Define a point on the right branch of the hyperbola
def is_on_right_branch (P : ℝ × ℝ) : Prop :=
  is_on_hyperbola P.1 P.2 ∧ P.1 > 0

-- Theorem statement
theorem min_distance_sum (P : ℝ × ℝ) (h : is_on_right_branch P) :
  dist P left_focus + dist P point_A ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1834_183444


namespace NUMINAMATH_CALUDE_polygon_sides_l1834_183437

theorem polygon_sides (n : ℕ) (h1 : n > 2) : 
  (140 + 145 * (n - 1) = 180 * (n - 2)) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1834_183437


namespace NUMINAMATH_CALUDE_at_least_four_boxes_same_items_l1834_183471

theorem at_least_four_boxes_same_items (boxes : Finset Nat) (items : Nat → Nat) : 
  boxes.card = 376 → 
  (∀ b ∈ boxes, items b ≤ 125) → 
  ∃ n : Nat, ∃ same_boxes : Finset Nat, same_boxes ⊆ boxes ∧ same_boxes.card ≥ 4 ∧ 
    ∀ b ∈ same_boxes, items b = n :=
by sorry

end NUMINAMATH_CALUDE_at_least_four_boxes_same_items_l1834_183471


namespace NUMINAMATH_CALUDE_car_speed_problem_l1834_183407

theorem car_speed_problem (v : ℝ) (h1 : v > 0) : 
  (60 / v - 60 / (v + 20) = 0.5) → v = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1834_183407


namespace NUMINAMATH_CALUDE_total_decrease_percentage_l1834_183412

-- Define the percentage decreases
def first_year_decrease : ℝ := 0.4
def second_year_decrease : ℝ := 0.1

-- Define the theorem
theorem total_decrease_percentage :
  ∀ (initial_value : ℝ), initial_value > 0 →
  let value_after_first_year := initial_value * (1 - first_year_decrease)
  let final_value := value_after_first_year * (1 - second_year_decrease)
  let total_decrease := (initial_value - final_value) / initial_value
  total_decrease = 0.46 := by
  sorry

end NUMINAMATH_CALUDE_total_decrease_percentage_l1834_183412


namespace NUMINAMATH_CALUDE_bus_seat_capacity_l1834_183443

theorem bus_seat_capacity :
  let left_seats : ℕ := 15
  let right_seats : ℕ := left_seats - 3
  let back_seat_capacity : ℕ := 12
  let total_capacity : ℕ := 93
  let seat_capacity : ℕ := (total_capacity - back_seat_capacity) / (left_seats + right_seats)
  seat_capacity = 3 := by sorry

end NUMINAMATH_CALUDE_bus_seat_capacity_l1834_183443


namespace NUMINAMATH_CALUDE_bug_crawl_distance_l1834_183469

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ

/-- Calculates the shortest distance between two points on the surface of a cone -/
noncomputable def shortestDistance (c : Cone) (p1 p2 : ConePoint) : ℝ :=
  sorry

theorem bug_crawl_distance (c : Cone) (p1 p2 : ConePoint) :
  c.baseRadius = 500 →
  c.height = 250 * Real.sqrt 3 →
  p1.distanceFromVertex = 100 →
  p2.distanceFromVertex = 300 * Real.sqrt 3 →
  shortestDistance c p1 p2 = 100 * Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_bug_crawl_distance_l1834_183469


namespace NUMINAMATH_CALUDE_eight_cubic_polynomials_l1834_183420

/-- A polynomial function of degree at most 3 -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + b * x^2 + c * x + d

/-- The condition that f(x) f(-x) = f(x^3) for all x -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, f x * f (-x) = f (x^3)

/-- The main theorem stating that there are exactly 8 cubic polynomials satisfying the condition -/
theorem eight_cubic_polynomials :
  ∃! (s : Finset (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (a b c d : ℝ), (a, b, c, d) ∈ s ↔ SatisfiesCondition (CubicPolynomial a b c d)) ∧
    Finset.card s = 8 := by
  sorry


end NUMINAMATH_CALUDE_eight_cubic_polynomials_l1834_183420


namespace NUMINAMATH_CALUDE_min_distinct_values_with_unique_mode_l1834_183490

theorem min_distinct_values_with_unique_mode (list_size : ℕ) (mode_frequency : ℕ) 
  (h1 : list_size = 3000)
  (h2 : mode_frequency = 15) :
  (∃ (distinct_values : ℕ), 
    distinct_values ≥ 215 ∧ 
    distinct_values * (mode_frequency - 1) + mode_frequency ≥ list_size ∧
    ∀ (n : ℕ), n < 215 → n * (mode_frequency - 1) + mode_frequency < list_size) :=
by sorry

end NUMINAMATH_CALUDE_min_distinct_values_with_unique_mode_l1834_183490


namespace NUMINAMATH_CALUDE_total_nuts_weight_l1834_183487

def almonds : Real := 0.14
def pecans : Real := 0.38

theorem total_nuts_weight : almonds + pecans = 0.52 := by sorry

end NUMINAMATH_CALUDE_total_nuts_weight_l1834_183487


namespace NUMINAMATH_CALUDE_line_equation_from_slope_and_point_l1834_183405

/-- A line in the 2D plane -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a line with slope 3 passing through (1, -2), its equation is 3x - y - 5 = 0 -/
theorem line_equation_from_slope_and_point :
  ∀ (l : Line),
  l.slope = 3 ∧ l.point = (1, -2) →
  ∃ (eq : LineEquation),
  eq.a = 3 ∧ eq.b = -1 ∧ eq.c = -5 ∧
  ∀ (x y : ℝ), eq.a * x + eq.b * y + eq.c = 0 ↔ y = l.slope * (x - l.point.1) + l.point.2 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_slope_and_point_l1834_183405


namespace NUMINAMATH_CALUDE_banana_production_ratio_l1834_183402

/-- The ratio of banana production between Jakies Island and a nearby island -/
theorem banana_production_ratio :
  ∀ (jakies_multiple : ℕ) (nearby_production : ℕ) (total_production : ℕ),
  nearby_production = 9000 →
  total_production = 99000 →
  total_production = nearby_production + jakies_multiple * nearby_production →
  (jakies_multiple * nearby_production) / nearby_production = 10 :=
by
  sorry

#check banana_production_ratio

end NUMINAMATH_CALUDE_banana_production_ratio_l1834_183402


namespace NUMINAMATH_CALUDE_principal_mistake_l1834_183408

theorem principal_mistake : ¬∃ (x y : ℕ), 2 * x = 2 * y + 11 := by
  sorry

end NUMINAMATH_CALUDE_principal_mistake_l1834_183408


namespace NUMINAMATH_CALUDE_inequality_proof_l1834_183495

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1834_183495


namespace NUMINAMATH_CALUDE_intersection_chord_length_l1834_183450

/-- The line C in the Cartesian plane -/
def line_C (x y : ℝ) : Prop := x - y - 1 = 0

/-- The circle P in the Cartesian plane -/
def circle_P (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

/-- The theorem stating that the length of the chord formed by the intersection
    of line C and circle P is √2 -/
theorem intersection_chord_length :
  ∃ (A B : ℝ × ℝ),
    line_C A.1 A.2 ∧ line_C B.1 B.2 ∧
    circle_P A.1 A.2 ∧ circle_P B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l1834_183450


namespace NUMINAMATH_CALUDE_ellipse_equation_l1834_183414

/-- Given an ellipse with the following properties:
  1. The axes of symmetry lie on the coordinate axes
  2. One endpoint of the minor axis and the two foci form an equilateral triangle
  3. The distance from the foci to the same vertex is √3
  Then the standard equation of the ellipse is x²/12 + y²/9 = 1 or y²/12 + x²/9 = 1 -/
theorem ellipse_equation (a c : ℝ) (h1 : a = 2 * c) (h2 : a - c = Real.sqrt 3) :
  ∃ (x y : ℝ), (x^2 / 12 + y^2 / 9 = 1) ∨ (y^2 / 12 + x^2 / 9 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1834_183414


namespace NUMINAMATH_CALUDE_third_card_value_l1834_183453

def sum_of_permutations (a b x : ℕ) : ℕ :=
  100000 * a + 10000 * b + 100 * x +
  100000 * a + 10000 * x + b +
  100000 * b + 10000 * a + x +
  100000 * b + 10000 * x + a +
  100000 * x + 10000 * b + a +
  100000 * x + 10000 * a + b

theorem third_card_value (x : ℕ) :
  x < 100 →
  sum_of_permutations 18 75 x = 2606058 →
  x = 36 := by
sorry

end NUMINAMATH_CALUDE_third_card_value_l1834_183453


namespace NUMINAMATH_CALUDE_gideon_age_proof_l1834_183404

/-- The number of years in a century -/
def years_in_century : ℕ := 100

/-- Gideon's current age -/
def gideon_age : ℕ := 45

/-- The number of marbles Gideon has -/
def gideon_marbles : ℕ := years_in_century

/-- Gideon's age five years from now -/
def gideon_future_age : ℕ := gideon_age + 5

theorem gideon_age_proof :
  gideon_age = 45 ∧
  gideon_marbles = years_in_century ∧
  gideon_future_age = 2 * (gideon_marbles / 4) :=
by sorry

end NUMINAMATH_CALUDE_gideon_age_proof_l1834_183404


namespace NUMINAMATH_CALUDE_second_player_wins_l1834_183485

/-- A game on a circle with 2n + 1 equally spaced points -/
structure CircleGame where
  n : ℕ
  h : n ≥ 2

/-- A strategy for the second player -/
def SecondPlayerStrategy (game : CircleGame) : Type :=
  ℕ → ℕ

/-- Predicate to check if a triangle is obtuse -/
def IsObtuse (p1 p2 p3 : ℕ) : Prop :=
  sorry

/-- Predicate to check if all remaining triangles are obtuse -/
def AllTrianglesObtuse (remaining_points : List ℕ) : Prop :=
  sorry

/-- Predicate to check if a strategy is winning for the second player -/
def IsWinningStrategy (game : CircleGame) (strategy : SecondPlayerStrategy game) : Prop :=
  ∀ (first_player_moves : List ℕ),
    AllTrianglesObtuse (sorry) -- remaining points after applying the strategy

theorem second_player_wins (game : CircleGame) :
  ∃ (strategy : SecondPlayerStrategy game), IsWinningStrategy game strategy :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l1834_183485


namespace NUMINAMATH_CALUDE_base8_addition_subtraction_l1834_183492

/-- Converts a base 8 number to base 10 --/
def base8ToBase10 (x : ℕ) : ℕ :=
  let ones := x % 10
  let eights := x / 10
  8 * eights + ones

/-- Converts a base 10 number to base 8 --/
def base10ToBase8 (x : ℕ) : ℕ :=
  let quotient := x / 8
  let remainder := x % 8
  10 * quotient + remainder

theorem base8_addition_subtraction :
  base10ToBase8 ((base8ToBase10 10 + base8ToBase10 26) - base8ToBase10 13) = 23 := by
  sorry

end NUMINAMATH_CALUDE_base8_addition_subtraction_l1834_183492


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l1834_183486

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_power_dividing_factorial :
  let n := 2520
  ∃ k : ℕ, k = 418 ∧
    (∀ m : ℕ, n^m ∣ factorial n → m ≤ k) ∧
    n^k ∣ factorial n :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l1834_183486


namespace NUMINAMATH_CALUDE_annie_completion_time_correct_l1834_183438

/-- Dan's time to complete the job alone -/
def dan_time : ℝ := 15

/-- Annie's time to complete the job alone -/
def annie_time : ℝ := 3.6

/-- Time Dan works before stopping -/
def dan_work_time : ℝ := 6

/-- Time Annie takes to finish the job after Dan stops -/
def annie_finish_time : ℝ := 6

/-- The theorem stating that Annie's time to complete the job alone is correct -/
theorem annie_completion_time_correct :
  (dan_work_time / dan_time) + (annie_finish_time / annie_time) = 1 := by
  sorry

end NUMINAMATH_CALUDE_annie_completion_time_correct_l1834_183438


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1834_183479

/-- The distance between the foci of an ellipse with equation x^2 + 9y^2 = 576 is 32√2 -/
theorem ellipse_foci_distance : 
  let a : ℝ := Real.sqrt (576 / 1)
  let b : ℝ := Real.sqrt (576 / 9)
  2 * Real.sqrt (a^2 - b^2) = 32 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1834_183479


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1834_183477

theorem solve_exponential_equation :
  ∃ x : ℝ, (16 : ℝ) ^ x * (16 : ℝ) ^ x * (16 : ℝ) ^ x * (16 : ℝ) ^ x = (256 : ℝ) ^ 4 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1834_183477


namespace NUMINAMATH_CALUDE_red_balls_count_l1834_183417

/-- Given a bag of balls with some red and some white balls, prove the number of red balls. -/
theorem red_balls_count (total_balls : ℕ) (red_prob : ℝ) (h_total : total_balls = 50) (h_prob : red_prob = 0.7) :
  ⌊(total_balls : ℝ) * red_prob⌋ = 35 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l1834_183417


namespace NUMINAMATH_CALUDE_symmetric_points_mn_l1834_183491

/-- Given two points P and Q that are symmetric about the origin, prove that mn = -2 --/
theorem symmetric_points_mn (m n : ℝ) : 
  (m - n = -3 ∧ 1 = -(m + n)) → m * n = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_mn_l1834_183491


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1834_183418

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ x - 4 = 21 * (1/x) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1834_183418


namespace NUMINAMATH_CALUDE_EF_length_l1834_183452

-- Define the segment AB and points C, D, E, F
def AB : ℝ := 26
def AC : ℝ := 1
def AD : ℝ := 8

-- Define the semicircle with diameter AB
def semicircle (x y : ℝ) : Prop :=
  x ≥ 0 ∧ x ≤ AB ∧ y ≥ 0 ∧ x * (AB - x) = y^2

-- Define the perpendicularity condition
def perpendicular (x y : ℝ) : Prop :=
  semicircle x y ∧ (x = AC ∨ x = AD)

-- Theorem statement
theorem EF_length :
  ∃ (xE yE xF yF : ℝ),
    perpendicular xE yE ∧
    perpendicular xF yF ∧
    xE = AC ∧
    xF = AD ∧
    (yF - yE)^2 + (xF - xE)^2 = (7 * Real.sqrt 2)^2 :=
sorry

end NUMINAMATH_CALUDE_EF_length_l1834_183452


namespace NUMINAMATH_CALUDE_min_candies_pile_l1834_183499

theorem min_candies_pile : ∃ N : ℕ, N > 0 ∧ 
  (∃ k₁ : ℕ, N - 5 = 2 * k₁) ∧ 
  (∃ k₂ : ℕ, N - 2 = 3 * k₂) ∧ 
  (∃ k₃ : ℕ, N - 3 = 5 * k₃) ∧ 
  (∀ M : ℕ, M > 0 → 
    ((∃ m₁ : ℕ, M - 5 = 2 * m₁) ∧ 
     (∃ m₂ : ℕ, M - 2 = 3 * m₂) ∧ 
     (∃ m₃ : ℕ, M - 3 = 5 * m₃)) → M ≥ N) ∧
  N = 53 := by
sorry

end NUMINAMATH_CALUDE_min_candies_pile_l1834_183499


namespace NUMINAMATH_CALUDE_number_divided_by_seven_l1834_183406

theorem number_divided_by_seven (x : ℝ) : x / 7 = 5 / 14 → x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_seven_l1834_183406


namespace NUMINAMATH_CALUDE_monica_savings_l1834_183489

def savings_pattern (week : ℕ) : ℕ :=
  let cycle := week % 20
  if cycle < 6 then 15 + 5 * cycle
  else if cycle < 12 then 40 - 5 * (cycle - 6)
  else if cycle < 18 then 15 + 5 * (cycle - 12)
  else 40 - 5 * (cycle - 18)

def total_savings : ℕ := (List.range 100).map savings_pattern |> List.sum

theorem monica_savings :
  total_savings = 1450 := by sorry

end NUMINAMATH_CALUDE_monica_savings_l1834_183489


namespace NUMINAMATH_CALUDE_license_plate_count_l1834_183484

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_count : ℕ := 10

/-- The number of even digits -/
def even_digit_count : ℕ := 5

/-- The number of odd digits -/
def odd_digit_count : ℕ := 5

/-- The number of letters in the license plate -/
def letter_count : ℕ := 3

/-- The number of digits in the license plate -/
def plate_digit_count : ℕ := 3

/-- The number of ways to arrange the odd, even, and any digit -/
def digit_arrangements : ℕ := 3

theorem license_plate_count :
  (alphabet_size ^ letter_count) *
  (even_digit_count * odd_digit_count * digit_count) *
  digit_arrangements = 13182000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1834_183484


namespace NUMINAMATH_CALUDE_express_w_in_terms_of_abc_l1834_183425

/-- Given distinct real numbers and a system of equations, prove the expression for w -/
theorem express_w_in_terms_of_abc (a b c w : ℝ) (x y z : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ w ≠ a ∧ w ≠ b ∧ w ≠ c ∧ 157 ≠ w ∧ 157 ≠ a ∧ 157 ≠ b ∧ 157 ≠ c)
  (h1 : x + y + z = 1)
  (h2 : x * a^2 + y * b^2 + z * c^2 = w^2)
  (h3 : x * a^3 + y * b^3 + z * c^3 = w^3)
  (h4 : x * a^4 + y * b^4 + z * c^4 = w^4) :
  (a*b + a*c + b*c = 0 → w = -a*b/(a+b)) ∧ 
  (a*b + a*c + b*c ≠ 0 → w = -a*b*c/(a*b + a*c + b*c)) :=
sorry

end NUMINAMATH_CALUDE_express_w_in_terms_of_abc_l1834_183425


namespace NUMINAMATH_CALUDE_max_value_problem_l1834_183400

theorem max_value_problem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 3) :
  (x * y) / (x + y) + (x * z) / (x + z) + (y * z) / (y + z) ≤ 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_l1834_183400


namespace NUMINAMATH_CALUDE_proportion_problem_l1834_183468

theorem proportion_problem (x y z v : ℤ) : 
  (x * v = y * z) →
  (x + v = y + z + 7) →
  (x^2 + v^2 = y^2 + z^2 + 21) →
  (x^4 + v^4 = y^4 + z^4 + 2625) →
  ((x = -3 ∧ v = 8 ∧ y = -6 ∧ z = 4) ∨ 
   (x = 8 ∧ v = -3 ∧ y = 4 ∧ z = -6)) := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l1834_183468


namespace NUMINAMATH_CALUDE_polynomial_properties_l1834_183403

theorem polynomial_properties :
  (∀ x : ℝ, x^2 + 2*x - 3 = (x-1)*(x+3)) ∧
  (∀ x : ℝ, x^2 + 4*x + 5 ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_polynomial_properties_l1834_183403


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l1834_183449

def polynomial (x : ℝ) : ℝ := 3 * (x^5 + 5*x^3 + 2)

theorem sum_of_squares_of_coefficients :
  (3^2 : ℝ) + 0^2 + 15^2 + 0^2 + 0^2 + 6^2 = 270 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l1834_183449


namespace NUMINAMATH_CALUDE_cyrus_day4_pages_l1834_183422

/-- Represents the number of pages written on each day --/
structure DailyPages where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Represents the book writing problem --/
structure BookWritingProblem where
  totalPages : ℕ
  pagesWritten : DailyPages
  remainingPages : ℕ

/-- The specific instance of the book writing problem --/
def cyrusProblem : BookWritingProblem where
  totalPages := 500
  pagesWritten := {
    day1 := 25,
    day2 := 50,
    day3 := 100,
    day4 := 10  -- This is what we want to prove
  }
  remainingPages := 315

/-- Theorem stating that Cyrus wrote 10 pages on day 4 --/
theorem cyrus_day4_pages : 
  cyrusProblem.pagesWritten.day4 = 10 ∧
  cyrusProblem.pagesWritten.day2 = 2 * cyrusProblem.pagesWritten.day1 ∧
  cyrusProblem.pagesWritten.day3 = 2 * cyrusProblem.pagesWritten.day2 ∧
  cyrusProblem.totalPages = 
    cyrusProblem.pagesWritten.day1 + 
    cyrusProblem.pagesWritten.day2 + 
    cyrusProblem.pagesWritten.day3 + 
    cyrusProblem.pagesWritten.day4 + 
    cyrusProblem.remainingPages := by
  sorry

end NUMINAMATH_CALUDE_cyrus_day4_pages_l1834_183422


namespace NUMINAMATH_CALUDE_sqrt_29_minus_1_between_4_and_5_l1834_183427

theorem sqrt_29_minus_1_between_4_and_5 :
  let a : ℝ := Real.sqrt 29 - 1
  4 < a ∧ a < 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_29_minus_1_between_4_and_5_l1834_183427


namespace NUMINAMATH_CALUDE_min_editors_l1834_183473

theorem min_editors (total : ℕ) (writers : ℕ) (x : ℕ) (both_max : ℕ) :
  total = 100 →
  writers = 40 →
  x ≤ both_max →
  both_max = 21 →
  total = writers + x + 2 * x →
  ∃ (editors : ℕ), editors ≥ 39 ∧ total = writers + editors + x :=
by sorry

end NUMINAMATH_CALUDE_min_editors_l1834_183473


namespace NUMINAMATH_CALUDE_sin_360_degrees_equals_zero_l1834_183409

theorem sin_360_degrees_equals_zero : Real.sin (2 * Real.pi) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_360_degrees_equals_zero_l1834_183409


namespace NUMINAMATH_CALUDE_matrix_addition_and_scalar_multiplication_l1834_183429

theorem matrix_addition_and_scalar_multiplication :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 2, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-2, 1; 0, 3]
  A + 3 • B = !![-2, 0; 2, 14] := by sorry

end NUMINAMATH_CALUDE_matrix_addition_and_scalar_multiplication_l1834_183429


namespace NUMINAMATH_CALUDE_function_inequality_implies_k_range_l1834_183433

open Real

theorem function_inequality_implies_k_range (k : ℝ) : k > 0 → 
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → 
    (exp 2 * x₁ / exp x₁) / k ≤ (exp 2 * x₂^2 + 1) / (x₂ * (k + 1))) → 
  k ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_k_range_l1834_183433


namespace NUMINAMATH_CALUDE_sqrt_two_power_2000_identity_l1834_183498

theorem sqrt_two_power_2000_identity : 
  (Real.sqrt 2 + 1)^2000 * (Real.sqrt 2 - 1)^2000 = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_two_power_2000_identity_l1834_183498


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l1834_183483

theorem unique_solution_sqrt_equation (m n : ℤ) :
  (5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n ↔ m = 0 ∧ n = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l1834_183483


namespace NUMINAMATH_CALUDE_quartic_root_product_l1834_183456

theorem quartic_root_product (k : ℝ) : 
  (∃ a b c d : ℝ, 
    (a^4 - 18*a^3 + k*a^2 + 200*a - 1984 = 0) ∧
    (b^4 - 18*b^3 + k*b^2 + 200*b - 1984 = 0) ∧
    (c^4 - 18*c^3 + k*c^2 + 200*c - 1984 = 0) ∧
    (d^4 - 18*d^3 + k*d^2 + 200*d - 1984 = 0) ∧
    (a * b = -32 ∨ a * c = -32 ∨ a * d = -32 ∨ b * c = -32 ∨ b * d = -32 ∨ c * d = -32)) →
  k = 86 := by
sorry

end NUMINAMATH_CALUDE_quartic_root_product_l1834_183456


namespace NUMINAMATH_CALUDE_plates_used_l1834_183474

theorem plates_used (guests : ℕ) (meals_per_day : ℕ) (plates_per_meal : ℕ) (days : ℕ) : 
  guests = 5 →
  meals_per_day = 3 →
  plates_per_meal = 2 →
  days = 4 →
  (guests + 1) * meals_per_day * plates_per_meal * days = 144 :=
by sorry

end NUMINAMATH_CALUDE_plates_used_l1834_183474


namespace NUMINAMATH_CALUDE_hall_dimensions_l1834_183445

/-- Given a rectangular hall with width half of its length and area 800 sq. m,
    prove that the difference between length and width is 20 meters. -/
theorem hall_dimensions (length width : ℝ) : 
  width = length / 2 →
  length * width = 800 →
  length - width = 20 :=
by sorry

end NUMINAMATH_CALUDE_hall_dimensions_l1834_183445


namespace NUMINAMATH_CALUDE_circle_properties_l1834_183410

-- Define the circle C: (x-2)²+y²=1
def C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point P(m,n) on the circle C
def P (m n : ℝ) : Prop := C m n

-- Theorem statement
theorem circle_properties :
  (∃ (m₀ n₀ : ℝ), P m₀ n₀ ∧ ∀ (m n : ℝ), P m n → |n / m| ≤ |n₀ / m₀| ∧ |n₀ / m₀| = Real.sqrt 3 / 3) ∧
  (∃ (m₁ n₁ : ℝ), P m₁ n₁ ∧ ∀ (m n : ℝ), P m n → m^2 + n^2 ≤ m₁^2 + n₁^2 ∧ m₁^2 + n₁^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1834_183410


namespace NUMINAMATH_CALUDE_integral_of_exponential_l1834_183423

theorem integral_of_exponential (x : ℝ) :
  let f : ℝ → ℝ := λ x => (3^(7*x - 1/9)) / (7 * Real.log 3)
  (deriv f) x = 3^(7*x - 1/9) := by
  sorry

end NUMINAMATH_CALUDE_integral_of_exponential_l1834_183423


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1834_183462

theorem smallest_part_of_proportional_division (total : ℝ) (a b c : ℝ) 
  (h_total : total = 120)
  (h_prop : a + b + c = 15)
  (h_a : a = 3)
  (h_b : b = 5)
  (h_c : c = 7) :
  min (total * a / (a + b + c)) (min (total * b / (a + b + c)) (total * c / (a + b + c))) = 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1834_183462


namespace NUMINAMATH_CALUDE_convex_regular_polygon_integer_angles_l1834_183447

/-- The number of positive integers n ≥ 3 such that 360 is divisible by n -/
def count_divisors : Nat :=
  (Finset.filter (fun n => n ≥ 3 ∧ 360 % n = 0) (Finset.range 361)).card

/-- Theorem stating that there are exactly 22 positive integers n ≥ 3 
    such that 360 is divisible by n -/
theorem convex_regular_polygon_integer_angles : count_divisors = 22 := by
  sorry

end NUMINAMATH_CALUDE_convex_regular_polygon_integer_angles_l1834_183447


namespace NUMINAMATH_CALUDE_circle_area_outside_triangle_l1834_183416

/-- Given a right triangle ABC with ∠BAC = 90° and AB = 6, and a circle tangent to AB at X and AC at Y 
    with points diametrically opposite X and Y lying on BC, the area of the portion of the circle 
    that lies outside the triangle is 18π - 18. -/
theorem circle_area_outside_triangle (A B C X Y : ℝ × ℝ) (r : ℝ) : 
  -- Triangle ABC is a right triangle with ∠BAC = 90°
  (A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 6 ∧ B.2 = 0 ∧ C.1 = 0 ∧ C.2 = 6) →
  -- Circle is tangent to AB at X and AC at Y
  (X.1 = r ∧ X.2 = 0 ∧ Y.1 = 0 ∧ Y.2 = r) →
  -- Points diametrically opposite X and Y lie on BC
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (2*r)^2 →
  -- The area of the portion of the circle outside the triangle
  π * r^2 - (B.1 * C.2 / 2) = 18 * π - 18 := by
sorry

end NUMINAMATH_CALUDE_circle_area_outside_triangle_l1834_183416


namespace NUMINAMATH_CALUDE_amount_difference_l1834_183494

def distribute_amount (total : ℝ) (p q r s t : ℝ) : Prop :=
  total = 25000 ∧
  p = 2 * q ∧
  s = 4 * r ∧
  q = r ∧
  p + q + r = (5/9) * total ∧
  s / (s + t) = 2/3 ∧
  s - p = 6944.4444

theorem amount_difference :
  ∀ (total p q r s t : ℝ),
  distribute_amount total p q r s t →
  s - p = 6944.4444 :=
by sorry

end NUMINAMATH_CALUDE_amount_difference_l1834_183494


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l1834_183434

theorem rectangular_prism_width (l w h : ℕ) : 
  l * w * h = 128 → 
  w = 2 * l → 
  w = 2 * h → 
  w + 2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l1834_183434


namespace NUMINAMATH_CALUDE_range_of_a_l1834_183463

/-- The line y = x + 2 intersects the x-axis at point M and the y-axis at point N. -/
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (0, 2)

/-- Point P moves on the circle (x-a)^2 + y^2 = 2, where a > 0 -/
def circle_equation (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = 2

/-- Angle MPN is always acute -/
def angle_MPN_acute (P : ℝ × ℝ) : Prop := sorry

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ P : ℝ × ℝ, circle_equation a P.1 P.2 → angle_MPN_acute P) →
  a > Real.sqrt 7 - 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1834_183463


namespace NUMINAMATH_CALUDE_A_subset_B_l1834_183428

/-- Set A is defined as {x | x(x-1) < 0} -/
def A : Set ℝ := {x | x * (x - 1) < 0}

/-- Set B is defined as {y | y = x^2 for some real x} -/
def B : Set ℝ := {y | ∃ x : ℝ, y = x^2}

/-- Theorem: A is a subset of B -/
theorem A_subset_B : A ⊆ B := by sorry

end NUMINAMATH_CALUDE_A_subset_B_l1834_183428


namespace NUMINAMATH_CALUDE_max_sum_of_vertex_products_l1834_183446

/-- Represents the set of numbers that can be assigned to cube faces -/
def CubeNumbers : Finset ℕ := {0, 1, 2, 3, 8, 9}

/-- A function that assigns numbers to cube faces -/
def FaceAssignment := Fin 6 → ℕ

/-- Predicate to check if a face assignment is valid -/
def ValidAssignment (f : FaceAssignment) : Prop :=
  (∀ i : Fin 6, f i ∈ CubeNumbers) ∧ (∀ i j : Fin 6, i ≠ j → f i ≠ f j)

/-- Calculate the product at a vertex given three face numbers -/
def VertexProduct (a b c : ℕ) : ℕ := a * b * c

/-- Calculate the sum of all vertex products for a given face assignment -/
def SumOfVertexProducts (f : FaceAssignment) : ℕ :=
  VertexProduct (f 0) (f 1) (f 2) +
  VertexProduct (f 0) (f 1) (f 3) +
  VertexProduct (f 0) (f 2) (f 4) +
  VertexProduct (f 0) (f 3) (f 4) +
  VertexProduct (f 1) (f 2) (f 5) +
  VertexProduct (f 1) (f 3) (f 5) +
  VertexProduct (f 2) (f 4) (f 5) +
  VertexProduct (f 3) (f 4) (f 5)

/-- The main theorem stating that the maximum sum of vertex products is 405 -/
theorem max_sum_of_vertex_products :
  ∃ (f : FaceAssignment), ValidAssignment f ∧
  SumOfVertexProducts f = 405 ∧
  ∀ (g : FaceAssignment), ValidAssignment g → SumOfVertexProducts g ≤ 405 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_vertex_products_l1834_183446


namespace NUMINAMATH_CALUDE_min_sum_tangents_l1834_183461

theorem min_sum_tangents (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  a = 2 * b * Real.sin C →  -- Given condition
  8 ≤ Real.tan A + Real.tan B + Real.tan C ∧
  (∃ (A' B' C' : Real), 0 < A' ∧ 0 < B' ∧ 0 < C' ∧ A' + B' + C' = π ∧
    Real.tan A' + Real.tan B' + Real.tan C' = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_tangents_l1834_183461


namespace NUMINAMATH_CALUDE_sum_of_squares_equals_18_l1834_183459

/-- Right triangle ABC with hypotenuse AB -/
structure RightTriangle where
  AC : ℝ
  BC : ℝ
  AB : ℝ
  right_angle : AC^2 + BC^2 = AB^2

theorem sum_of_squares_equals_18 (triangle : RightTriangle) (h : triangle.AB = 3) :
  triangle.AB^2 + triangle.BC^2 + triangle.AC^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equals_18_l1834_183459


namespace NUMINAMATH_CALUDE_nadia_hannah_distance_ratio_l1834_183460

/-- Proves the ratio of Nadia's distance to Hannah's distance -/
theorem nadia_hannah_distance_ratio :
  ∀ (nadia_distance hannah_distance : ℕ) (k : ℕ),
    nadia_distance = 18 →
    nadia_distance + hannah_distance = 27 →
    nadia_distance = k * hannah_distance →
    nadia_distance / hannah_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_nadia_hannah_distance_ratio_l1834_183460


namespace NUMINAMATH_CALUDE_inequality_solution_l1834_183424

/-- Given constants a, b, and c satisfying the specified conditions, prove that a + 2b + 3c = 48 -/
theorem inequality_solution (a b c : ℝ) : 
  (∀ x, ((x - a) * (x - b)) / (x - c) ≥ 0 ↔ (x < -6 ∨ (20 ≤ x ∧ x ≤ 23))) →
  a < b →
  a + 2*b + 3*c = 48 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1834_183424


namespace NUMINAMATH_CALUDE_parabola_equation_l1834_183464

/-- A parabola with vertex at the origin and directrix x = 4 has the standard equation y^2 = -16x -/
theorem parabola_equation (y x : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ y^2 = -2*p*x) → -- Standard form of parabola equation
  (4 = p/2) →                        -- Condition for directrix at x = 4
  y^2 = -16*x :=                     -- Resulting equation
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1834_183464


namespace NUMINAMATH_CALUDE_arithmetic_progression_squares_l1834_183466

theorem arithmetic_progression_squares (x : ℝ) : 
  ((x^2 - 2*x - 1)^2 + (x^2 + 2*x - 1)^2) / 2 = (x^2 + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_squares_l1834_183466


namespace NUMINAMATH_CALUDE_angle_complement_quadrant_l1834_183448

/-- An angle is in the fourth quadrant if it's between 270° and 360° (exclusive) -/
def is_fourth_quadrant (α : Real) : Prop :=
  270 < α ∧ α < 360

/-- An angle is in the third quadrant if it's between 180° and 270° (exclusive) -/
def is_third_quadrant (α : Real) : Prop :=
  180 < α ∧ α < 270

theorem angle_complement_quadrant (α : Real) :
  is_fourth_quadrant α → is_third_quadrant (180 - α) := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_quadrant_l1834_183448


namespace NUMINAMATH_CALUDE_polynomial_divisibility_implies_root_l1834_183432

theorem polynomial_divisibility_implies_root (r : ℝ) : 
  (∃ (p : ℝ → ℝ), (∀ x, 9 * x^3 - 6 * x^2 - 48 * x + 54 = (x - r)^2 * p x)) → 
  r = 4/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_implies_root_l1834_183432


namespace NUMINAMATH_CALUDE_system_solution_l1834_183465

theorem system_solution :
  ∀ x y : ℝ,
  x^2 - 3*y - 88 ≥ 0 →
  x + 6*y ≥ 0 →
  (5 * Real.sqrt (x^2 - 3*y - 88) + Real.sqrt (x + 6*y) = 19 ∧
   3 * Real.sqrt (x^2 - 3*y - 88) = 1 + 2 * Real.sqrt (x + 6*y)) →
  ((x = 10 ∧ y = 1) ∨ (x = -21/2 ∧ y = 53/12)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1834_183465


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_values_l1834_183457

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = 3 ∧ p.1 ≠ 2}
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + 2 * p.2 + a = 0}

-- State the theorem
theorem intersection_empty_implies_a_values :
  ∀ a : ℝ, (M ∩ N a = ∅) → (a = -6 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_values_l1834_183457


namespace NUMINAMATH_CALUDE_largest_710_double_correct_l1834_183481

/-- Converts a base-10 number to its base-7 representation as a list of digits --/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Interprets a list of digits as a base-10 number --/
def fromDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a number is a 7-10 double --/
def is710Double (n : ℕ) : Prop :=
  fromDigits (toBase7 n) = 2 * n

/-- The largest 7-10 double --/
def largest710Double : ℕ := 315

theorem largest_710_double_correct :
  is710Double largest710Double ∧
  ∀ n : ℕ, n > largest710Double → ¬is710Double n :=
sorry

end NUMINAMATH_CALUDE_largest_710_double_correct_l1834_183481


namespace NUMINAMATH_CALUDE_bicycle_discount_proof_l1834_183421

theorem bicycle_discount_proof (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 200 ∧ discount1 = 0.4 ∧ discount2 = 0.25 →
  original_price * (1 - discount1) * (1 - discount2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_discount_proof_l1834_183421


namespace NUMINAMATH_CALUDE_exists_triangle_area_not_greater_than_two_l1834_183454

/-- A lattice point in a 2D coordinate system -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Checks if a lattice point is within the 5x5 grid centered at the origin -/
def isWithinGrid (p : LatticePoint) : Prop :=
  |p.x| ≤ 2 ∧ |p.y| ≤ 2

/-- Checks if three points are collinear -/
def areCollinear (p1 p2 p3 : LatticePoint) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Calculates the area of a triangle formed by three lattice points -/
def triangleArea (p1 p2 p3 : LatticePoint) : ℚ :=
  |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)| / 2

/-- Main theorem statement -/
theorem exists_triangle_area_not_greater_than_two 
  (points : Fin 6 → LatticePoint)
  (h_within_grid : ∀ i, isWithinGrid (points i))
  (h_not_collinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬areCollinear (points i) (points j) (points k)) :
  ∃ i j k, i ≠ j → j ≠ k → i ≠ k → triangleArea (points i) (points j) (points k) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_triangle_area_not_greater_than_two_l1834_183454


namespace NUMINAMATH_CALUDE_lines_intersect_at_point_l1834_183441

/-- Represents a 2D point -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line in parametric form -/
structure ParametricLine where
  p : Point  -- Starting point
  v : Point  -- Direction vector

/-- The first line -/
def line1 : ParametricLine :=
  { p := { x := 1, y := 4 },
    v := { x := -2, y := 3 } }

/-- The second line -/
def line2 : ParametricLine :=
  { p := { x := 5, y := 2 },
    v := { x := 1, y := 6 } }

/-- A point on a parametric line -/
def pointOnLine (l : ParametricLine) (t : ℚ) : Point :=
  { x := l.p.x + t * l.v.x,
    y := l.p.y + t * l.v.y }

/-- The proposed intersection point -/
def intersectionPoint : Point :=
  { x := 21 / 5,
    y := -4 / 5 }

theorem lines_intersect_at_point :
  ∃ (t u : ℚ), pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint :=
sorry

end NUMINAMATH_CALUDE_lines_intersect_at_point_l1834_183441


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l1834_183475

/-- Given points A and B with line AB parallel to y-axis, prove (-a, a+3) is in first quadrant --/
theorem point_in_first_quadrant (a : ℝ) : 
  (a - 1 = -2) →  -- Line AB parallel to y-axis implies x-coordinates are equal
  ((-a > 0) ∧ (a + 3 > 0)) := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l1834_183475


namespace NUMINAMATH_CALUDE_total_chairs_count_l1834_183415

theorem total_chairs_count : ℕ := by
  -- Define the number of rows and chairs per row for each section
  let first_section_rows : ℕ := 5
  let first_section_chairs_per_row : ℕ := 10
  let second_section_rows : ℕ := 8
  let second_section_chairs_per_row : ℕ := 12

  -- Define the number of late arrivals and extra chairs per late arrival
  let late_arrivals : ℕ := 20
  let extra_chairs_per_late_arrival : ℕ := 3

  -- Calculate the total number of chairs
  let total_chairs := 
    (first_section_rows * first_section_chairs_per_row) +
    (second_section_rows * second_section_chairs_per_row) +
    (late_arrivals * extra_chairs_per_late_arrival)

  -- Prove that the total number of chairs is 206
  have h : total_chairs = 206 := by sorry

  exact 206


end NUMINAMATH_CALUDE_total_chairs_count_l1834_183415


namespace NUMINAMATH_CALUDE_disney_banquet_attendees_l1834_183472

/-- The number of people who attended a Disney banquet -/
theorem disney_banquet_attendees :
  ∀ (resident_price non_resident_price total_revenue : ℚ) 
    (num_residents : ℕ) (total_attendees : ℕ),
  resident_price = 1295/100 →
  non_resident_price = 1795/100 →
  total_revenue = 942370/100 →
  num_residents = 219 →
  total_revenue = (num_residents : ℚ) * resident_price + 
    ((total_attendees - num_residents) : ℚ) * non_resident_price →
  total_attendees = 586 := by
sorry

end NUMINAMATH_CALUDE_disney_banquet_attendees_l1834_183472


namespace NUMINAMATH_CALUDE_combined_work_time_l1834_183401

def team_A_time : ℝ := 15
def team_B_time : ℝ := 30

theorem combined_work_time :
  1 / (1 / team_A_time + 1 / team_B_time) = 10 := by
  sorry

end NUMINAMATH_CALUDE_combined_work_time_l1834_183401


namespace NUMINAMATH_CALUDE_max_angle_at_C_l1834_183493

/-- The line c given by the equation y = x + 1 -/
def line_c : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}

/-- Point A with coordinates (1, 0) -/
def point_A : ℝ × ℝ := (1, 0)

/-- Point B with coordinates (3, 0) -/
def point_B : ℝ × ℝ := (3, 0)

/-- Point C with coordinates (1, 2) -/
def point_C : ℝ × ℝ := (1, 2)

/-- The angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating that C maximizes the angle ACB -/
theorem max_angle_at_C :
  point_C ∈ line_c ∧
  ∀ p ∈ line_c, angle point_A p point_B ≤ angle point_A point_C point_B :=
by sorry

end NUMINAMATH_CALUDE_max_angle_at_C_l1834_183493


namespace NUMINAMATH_CALUDE_power_fraction_equality_l1834_183451

theorem power_fraction_equality : (2^2020 + 2^2016) / (2^2020 - 2^2016) = 17/15 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l1834_183451


namespace NUMINAMATH_CALUDE_smallest_interesting_number_l1834_183458

theorem smallest_interesting_number : 
  ∃ (n : ℕ), n = 1800 ∧ 
  (∀ (m : ℕ), m < n → ¬(∃ (k : ℕ), 2 * m = k ^ 2) ∨ ¬(∃ (l : ℕ), 15 * m = l ^ 3)) ∧
  (∃ (k : ℕ), 2 * n = k ^ 2) ∧
  (∃ (l : ℕ), 15 * n = l ^ 3) := by
sorry

end NUMINAMATH_CALUDE_smallest_interesting_number_l1834_183458


namespace NUMINAMATH_CALUDE_expression_simplification_l1834_183478

theorem expression_simplification (x : ℝ) (h : x = (1/2)⁻¹ + (-3)^0) :
  ((x^2 - 1) / (x^2 - 2*x + 1) - 1 / (x - 1)) / (3 / (x - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1834_183478


namespace NUMINAMATH_CALUDE_chess_team_arrangement_l1834_183497

/-- Represents the number of boys on the chess team -/
def num_boys : ℕ := 3

/-- Represents the number of girls on the chess team -/
def num_girls : ℕ := 2

/-- Represents the total number of team members -/
def total_members : ℕ := num_boys + num_girls

/-- Represents the number of ways to arrange the team members according to the specified conditions -/
def arrangements : ℕ := num_girls.factorial * num_boys.factorial

theorem chess_team_arrangement : arrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_arrangement_l1834_183497


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1834_183455

theorem right_triangle_perimeter (area : ℝ) (leg : ℝ) (h_area : area = 180) (h_leg : leg = 30) :
  ∃ (perimeter : ℝ), perimeter = 42 + 2 * Real.sqrt 261 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1834_183455


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l1834_183413

-- Define the quadratic function
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Theorem statement
theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, p a b c x = p a b c (21 - x)) →  -- Axis of symmetry at x = 10.5
  p a b c 0 = -4 →                           -- p(0) = -4
  p a b c 21 = -4 :=                         -- Conclusion: p(21) = -4
by
  sorry


end NUMINAMATH_CALUDE_quadratic_symmetry_l1834_183413


namespace NUMINAMATH_CALUDE_candy_ratio_l1834_183411

/-- Given:
  - There were 22 sweets on the table initially.
  - Jack took some portion of all the candies and 4 more candies.
  - Paul took the remaining 7 sweets.
Prove that the ratio of candies Jack took (excluding the 4 additional candies) 
to the total number of candies is 1/2. -/
theorem candy_ratio : 
  ∀ (jack_portion : ℕ),
  jack_portion + 4 + 7 = 22 →
  (jack_portion : ℚ) / 22 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_candy_ratio_l1834_183411


namespace NUMINAMATH_CALUDE_class_president_election_l1834_183435

theorem class_president_election (total_votes : ℕ) 
  (emily_votes : ℕ) (fiona_votes : ℕ) : 
  emily_votes = total_votes / 4 →
  fiona_votes = total_votes / 3 →
  emily_votes + fiona_votes = 77 →
  total_votes = 132 := by
sorry

end NUMINAMATH_CALUDE_class_president_election_l1834_183435


namespace NUMINAMATH_CALUDE_barley_percentage_is_80_percent_l1834_183426

/-- Represents the percentage of land that is cleared -/
def cleared_percentage : ℝ := 0.9

/-- Represents the percentage of cleared land planted with potato -/
def potato_percentage : ℝ := 0.1

/-- Represents the area of cleared land planted with tomato in acres -/
def tomato_area : ℝ := 90

/-- Represents the approximate total land area in acres -/
def total_land : ℝ := 1000

/-- Theorem stating that the percentage of cleared land planted with barley is 80% -/
theorem barley_percentage_is_80_percent :
  let cleared_land := cleared_percentage * total_land
  let barley_percentage := 1 - potato_percentage - (tomato_area / cleared_land)
  barley_percentage = 0.8 := by sorry

end NUMINAMATH_CALUDE_barley_percentage_is_80_percent_l1834_183426


namespace NUMINAMATH_CALUDE_triangle_properties_l1834_183431

open Real

theorem triangle_properties (A B C a b c : Real) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π) (h5 : cos (2 * A) - 3 * cos (B + C) = 1) (h6 : a > 0 ∧ b > 0 ∧ c > 0) :
  -- Part 1
  A = π / 3 ∧
  -- Part 2
  (∃ S : Real, S = 5 * Real.sqrt 3 ∧ b = 5 → sin B * sin C = 5 / 7) ∧
  -- Part 3
  (a = 1 → ∃ l : Real, l = a + b + c ∧ 2 < l ∧ l ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1834_183431


namespace NUMINAMATH_CALUDE_train_crossing_time_l1834_183470

/-- Given a train crossing a platform, calculate the time it takes to cross a signal pole --/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 675)
  (h3 : platform_crossing_time = 39)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1834_183470


namespace NUMINAMATH_CALUDE_equation_solution_l1834_183488

theorem equation_solution : 
  ∀ x : ℝ, (x + 1)^2 - 144 = 0 ↔ x = 11 ∨ x = -13 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1834_183488


namespace NUMINAMATH_CALUDE_min_expression_le_one_l1834_183442

theorem min_expression_le_one (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq_three : x + y + z = 3) :
  min (x * (x + y - z)) (min (y * (y + z - x)) (z * (z + x - y))) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_expression_le_one_l1834_183442


namespace NUMINAMATH_CALUDE_product_of_roots_implies_k_l1834_183480

-- Define the polynomial P(X)
def P (k X : ℝ) : ℝ := X^4 - 18*X^3 + k*X^2 + 200*X - 1984

-- Define the theorem
theorem product_of_roots_implies_k (k : ℝ) :
  (∃ a b c d : ℝ, 
    P k a = 0 ∧ P k b = 0 ∧ P k c = 0 ∧ P k d = 0 ∧
    ((a * b = -32) ∨ (a * c = -32) ∨ (a * d = -32) ∨ 
     (b * c = -32) ∨ (b * d = -32) ∨ (c * d = -32))) →
  k = 86 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_implies_k_l1834_183480


namespace NUMINAMATH_CALUDE_no_solution_exists_l1834_183467

theorem no_solution_exists : ¬∃ (x y : ℝ), 9^(y+1) / (1 + 4 / x^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1834_183467


namespace NUMINAMATH_CALUDE_alpha_value_l1834_183436

-- Define the triangle and point S
variable (P Q R S : Point)

-- Define the angles
variable (α β γ δ : ℝ)

-- Define the conditions
variable (triangle_PQR : Triangle P Q R)
variable (S_interior : InteriorPoint S triangle_PQR)
variable (QSP_bisected : AngleBisector S Q (Angle P S Q))
variable (delta_exterior : ExteriorAngle Q triangle_PQR δ)

-- Given angle values
variable (beta_value : β = 100)
variable (gamma_value : γ = 30)
variable (delta_value : δ = 150)

-- Theorem statement
theorem alpha_value : α = 215 := by sorry

end NUMINAMATH_CALUDE_alpha_value_l1834_183436


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1834_183440

def arithmeticSum (a1 : ℚ) (d : ℚ) (an : ℚ) : ℚ :=
  let n := (an - a1) / d + 1
  n * (a1 + an) / 2

theorem arithmetic_sequence_ratio : 
  let numerator := arithmeticSum 3 3 39
  let denominator := arithmeticSum 4 4 64
  numerator / denominator = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1834_183440


namespace NUMINAMATH_CALUDE_rachel_father_age_at_25_l1834_183496

/-- Rachel's current age -/
def rachel_age : ℕ := 12

/-- Rachel's grandfather's age in terms of Rachel's age -/
def grandfather_age_factor : ℕ := 7

/-- Rachel's mother's age in terms of grandfather's age -/
def mother_age_factor : ℚ := 1/2

/-- Age difference between Rachel's father and mother -/
def father_mother_age_diff : ℕ := 5

/-- Rachel's target age -/
def rachel_target_age : ℕ := 25

/-- Theorem stating Rachel's father's age when Rachel is 25 -/
theorem rachel_father_age_at_25 : 
  rachel_age * grandfather_age_factor * mother_age_factor + father_mother_age_diff + 
  (rachel_target_age - rachel_age) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rachel_father_age_at_25_l1834_183496


namespace NUMINAMATH_CALUDE_fourth_person_height_l1834_183430

/-- Heights of four people in increasing order -/
def Heights := Fin 4 → ℕ

/-- The condition that heights are in increasing order -/
def increasing_heights (h : Heights) : Prop :=
  ∀ i j, i < j → h i < h j

/-- The condition for the differences between heights -/
def height_differences (h : Heights) : Prop :=
  h 1 - h 0 = 2 ∧ h 2 - h 1 = 2 ∧ h 3 - h 2 = 6

/-- The condition for the average height -/
def average_height (h : Heights) : Prop :=
  (h 0 + h 1 + h 2 + h 3) / 4 = 79

theorem fourth_person_height (h : Heights) 
  (inc : increasing_heights h) 
  (diff : height_differences h) 
  (avg : average_height h) : 
  h 3 = 85 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l1834_183430


namespace NUMINAMATH_CALUDE_page_lines_increase_percentage_correct_increase_percentage_l1834_183439

theorem page_lines_increase_percentage : ℕ → ℝ → Prop :=
  fun original_lines increase_percentage =>
    let new_lines : ℕ := original_lines + 80
    new_lines = 240 →
    (increase_percentage * original_lines : ℝ) = 80 * 100

theorem correct_increase_percentage : 
  ∃ (original_lines : ℕ), page_lines_increase_percentage original_lines 50 := by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_percentage_correct_increase_percentage_l1834_183439

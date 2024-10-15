import Mathlib

namespace NUMINAMATH_CALUDE_real_roots_iff_a_leq_two_l2915_291563

theorem real_roots_iff_a_leq_two (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_real_roots_iff_a_leq_two_l2915_291563


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2915_291512

theorem complex_equation_solution (z : ℂ) : 
  (3 + Complex.I) * z = 2 - Complex.I → 
  z = (1 / 2 : ℂ) - (1 / 2 : ℂ) * Complex.I ∧ Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2915_291512


namespace NUMINAMATH_CALUDE_power_eight_mod_five_l2915_291528

theorem power_eight_mod_five : 8^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_eight_mod_five_l2915_291528


namespace NUMINAMATH_CALUDE_charles_total_earnings_l2915_291535

/-- Calculates Charles's total earnings from housesitting and dog walking -/
def charles_earnings (housesit_rate : ℕ) (dog_walk_rate : ℕ) (housesit_hours : ℕ) (dogs_walked : ℕ) : ℕ :=
  housesit_rate * housesit_hours + dog_walk_rate * dogs_walked

/-- Theorem stating that Charles's earnings are $216 given the specified rates and hours -/
theorem charles_total_earnings :
  charles_earnings 15 22 10 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_charles_total_earnings_l2915_291535


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l2915_291543

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: A regular dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l2915_291543


namespace NUMINAMATH_CALUDE_workday_end_time_l2915_291598

-- Define a custom time type
structure Time where
  hours : ℕ
  minutes : ℕ
  deriving Repr

def Time.toMinutes (t : Time) : ℕ :=
  t.hours * 60 + t.minutes

def Time.addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.toMinutes + m
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60 }

def Time.subtractMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.toMinutes - m
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60 }

def Time.differenceInMinutes (t1 t2 : Time) : ℕ :=
  if t1.toMinutes ≥ t2.toMinutes then
    t1.toMinutes - t2.toMinutes
  else
    t2.toMinutes - t1.toMinutes

theorem workday_end_time 
  (total_work_time : ℕ)
  (lunch_break : ℕ)
  (start_time : Time)
  (lunch_time : Time)
  (h1 : total_work_time = 8 * 60)  -- 8 hours in minutes
  (h2 : lunch_break = 30)  -- 30 minutes
  (h3 : start_time = { hours := 7, minutes := 0 })  -- 7:00 AM
  (h4 : lunch_time = { hours := 11, minutes := 30 })  -- 11:30 AM
  : Time.addMinutes lunch_time (total_work_time - Time.differenceInMinutes lunch_time start_time + lunch_break) = { hours := 15, minutes := 30 } :=
by sorry

end NUMINAMATH_CALUDE_workday_end_time_l2915_291598


namespace NUMINAMATH_CALUDE_smallest_digit_sum_of_difference_l2915_291541

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- A predicate that checks if all digits in a number are different -/
def allDigitsDifferent (n : ℕ) : Prop :=
  ∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10

theorem smallest_digit_sum_of_difference :
  ∀ a b : ℕ,
    100 ≤ a ∧ a < 1000 →
    100 ≤ b ∧ b < 1000 →
    a > b →
    allDigitsDifferent (1000000 * a + b) →
    100 ≤ a - b ∧ a - b < 1000 →
    (∀ D : ℕ, 100 ≤ D ∧ D < 1000 → D = a - b → sumOfDigits D ≥ 9) ∧
    (∃ D : ℕ, 100 ≤ D ∧ D < 1000 ∧ D = a - b ∧ sumOfDigits D = 9) :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_sum_of_difference_l2915_291541


namespace NUMINAMATH_CALUDE_number_of_roots_l2915_291571

/-- The number of real roots of a quadratic equation (m-5)x^2 - 2(m+2)x + m = 0,
    given that mx^2 - 2(m+2)x + m + 5 = 0 has no real roots -/
theorem number_of_roots (m : ℝ) 
  (h : ∀ x : ℝ, m * x^2 - 2*(m+2)*x + m + 5 ≠ 0) :
  (∃! x : ℝ, (m-5) * x^2 - 2*(m+2)*x + m = 0) ∨ 
  (∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2*(m+2)*x + m = 0 ∧ (m-5) * y^2 - 2*(m+2)*y + m = 0) :=
sorry

end NUMINAMATH_CALUDE_number_of_roots_l2915_291571


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2915_291565

/-- The quadratic function f(x) = x^2 - 12x + 36 -/
def f (x : ℝ) : ℝ := x^2 - 12*x + 36

theorem quadratic_minimum :
  ∃ (x_min : ℝ), f x_min = 0 ∧ ∀ (x : ℝ), f x ≥ 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2915_291565


namespace NUMINAMATH_CALUDE_polynomial_bound_l2915_291583

open Complex

theorem polynomial_bound (a b c : ℂ) :
  (∀ z : ℂ, Complex.abs z ≤ 1 → Complex.abs (a * z^2 + b * z + c) ≤ 1) →
  (∀ z : ℂ, Complex.abs z ≤ 1 → 0 ≤ Complex.abs (a * z + b) ∧ Complex.abs (a * z + b) ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_bound_l2915_291583


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2915_291527

/-- Proves that a boat traveling 45 miles upstream in 5 hours and 45 miles downstream in 3 hours has a speed of 12 mph in still water -/
theorem boat_speed_in_still_water : 
  ∀ (upstream_speed downstream_speed : ℝ),
  upstream_speed = 45 / 5 →
  downstream_speed = 45 / 3 →
  ∃ (boat_speed current_speed : ℝ),
  boat_speed - current_speed = upstream_speed ∧
  boat_speed + current_speed = downstream_speed ∧
  boat_speed = 12 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2915_291527


namespace NUMINAMATH_CALUDE_zachary_crunches_l2915_291530

/-- Proves that Zachary did 14 crunches given the conditions -/
theorem zachary_crunches : 
  ∀ (zachary_pushups zachary_total : ℕ),
  zachary_pushups = 53 →
  zachary_total = 67 →
  zachary_total - zachary_pushups = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_zachary_crunches_l2915_291530


namespace NUMINAMATH_CALUDE_vector_relation_l2915_291510

/-- Given points A, B, C, and D in a plane, where BC = 3CD, prove that AD = -1/3 AB + 4/3 AC -/
theorem vector_relation (A B C D : ℝ × ℝ) 
  (h : B - C = 3 * (C - D)) : 
  A - D = -1/3 * (A - B) + 4/3 * (A - C) := by
  sorry

end NUMINAMATH_CALUDE_vector_relation_l2915_291510


namespace NUMINAMATH_CALUDE_walk_distance_proof_l2915_291503

/-- Given a constant walking speed and time, calculates the distance walked. -/
def distance_walked (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that walking at 4 miles per hour for 2 hours results in a distance of 8 miles. -/
theorem walk_distance_proof :
  let speed : ℝ := 4
  let time : ℝ := 2
  distance_walked speed time = 8 := by
sorry

end NUMINAMATH_CALUDE_walk_distance_proof_l2915_291503


namespace NUMINAMATH_CALUDE_max_area_inscribed_triangle_l2915_291514

/-- The ellipse in which the triangle is inscribed -/
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

/-- A point on the ellipse -/
structure EllipsePoint where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

/-- The triangle inscribed in the ellipse -/
structure InscribedTriangle where
  A : EllipsePoint
  B : EllipsePoint
  C : EllipsePoint

/-- The condition that line segment AB passes through point P(1,0) -/
def passes_through_P (t : InscribedTriangle) : Prop :=
  ∃ k : ℝ, t.A.x + k * (t.B.x - t.A.x) = 1 ∧ t.A.y + k * (t.B.y - t.A.y) = 0

/-- The area of the triangle -/
noncomputable def triangle_area (t : InscribedTriangle) : ℝ :=
  abs ((t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y)) / 2

/-- The theorem to be proved -/
theorem max_area_inscribed_triangle :
  ∃ (t : InscribedTriangle), passes_through_P t ∧
    (∀ (t' : InscribedTriangle), passes_through_P t' → triangle_area t' ≤ triangle_area t) ∧
    triangle_area t = 16 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_area_inscribed_triangle_l2915_291514


namespace NUMINAMATH_CALUDE_salary_change_l2915_291533

theorem salary_change (initial_salary : ℝ) (h : initial_salary > 0) :
  let increased_salary := initial_salary * (1 + 0.2)
  let final_salary := increased_salary * (1 - 0.2)
  (final_salary - initial_salary) / initial_salary = -0.04 :=
by
  sorry

end NUMINAMATH_CALUDE_salary_change_l2915_291533


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expressions_l2915_291555

theorem simplify_trigonometric_expressions :
  (∀ α : ℝ, (1 + Real.tan α ^ 2) * Real.cos α ^ 2 = 1) ∧
  (Real.sin (7 * π / 6) + Real.tan (5 * π / 4) = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expressions_l2915_291555


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l2915_291521

theorem quadratic_function_inequality (a b c : ℝ) (h1 : b > 0) 
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  (a + b + c) / b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l2915_291521


namespace NUMINAMATH_CALUDE_mikes_shortfall_l2915_291502

theorem mikes_shortfall (max_marks : ℕ) (mikes_score : ℕ) (passing_percentage : ℚ) : 
  max_marks = 750 → 
  mikes_score = 212 → 
  passing_percentage = 30 / 100 → 
  (↑max_marks * passing_percentage).floor - mikes_score = 13 := by
  sorry

end NUMINAMATH_CALUDE_mikes_shortfall_l2915_291502


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2915_291549

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ (∀ x, f x ≠ 0) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  apply negation_of_existence

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2915_291549


namespace NUMINAMATH_CALUDE_point_symmetry_product_l2915_291580

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given that:
    - Point A lies on the y-axis with coordinates (3a-8, -3)
    - Points A and B(0, b) are symmetric with respect to the x-axis
    Prove that ab = 8 -/
theorem point_symmetry_product (a b : ℝ) : 
  let A : Point := ⟨3*a - 8, -3⟩
  let B : Point := ⟨0, b⟩
  (A.x = 0) →  -- A lies on the y-axis
  (A.y = -B.y) →  -- A and B are symmetric with respect to the x-axis
  a * b = 8 := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_product_l2915_291580


namespace NUMINAMATH_CALUDE_ticket_price_is_25_l2915_291590

-- Define the number of attendees for the first show
def first_show_attendees : ℕ := 200

-- Define the number of attendees for the second show
def second_show_attendees : ℕ := 3 * first_show_attendees

-- Define the total revenue
def total_revenue : ℕ := 20000

-- Define the ticket price
def ticket_price : ℚ := total_revenue / (first_show_attendees + second_show_attendees)

-- Theorem statement
theorem ticket_price_is_25 : ticket_price = 25 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_is_25_l2915_291590


namespace NUMINAMATH_CALUDE_paris_visits_l2915_291570

/-- Represents the attractions in Paris --/
inductive Attraction
  | EiffelTower
  | ArcDeTriomphe
  | Montparnasse
  | Playground

/-- Represents a nephew's statement about visiting an attraction --/
structure Statement where
  attraction : Attraction
  visited : Bool

/-- Represents a nephew's set of statements --/
structure NephewStatements where
  statements : List Statement

/-- The statements made by the three nephews --/
def nephewsStatements : List NephewStatements := [
  { statements := [
    { attraction := Attraction.EiffelTower, visited := true },
    { attraction := Attraction.ArcDeTriomphe, visited := true },
    { attraction := Attraction.Montparnasse, visited := false }
  ] },
  { statements := [
    { attraction := Attraction.EiffelTower, visited := true },
    { attraction := Attraction.Montparnasse, visited := true },
    { attraction := Attraction.ArcDeTriomphe, visited := false },
    { attraction := Attraction.Playground, visited := false }
  ] },
  { statements := [
    { attraction := Attraction.EiffelTower, visited := false },
    { attraction := Attraction.ArcDeTriomphe, visited := true }
  ] }
]

/-- The theorem to prove --/
theorem paris_visits (statements : List NephewStatements) 
  (h : statements = nephewsStatements) : 
  ∃ (visits : List Attraction),
    visits = [Attraction.EiffelTower, Attraction.ArcDeTriomphe, Attraction.Montparnasse] ∧
    Attraction.Playground ∉ visits :=
sorry

end NUMINAMATH_CALUDE_paris_visits_l2915_291570


namespace NUMINAMATH_CALUDE_line_slope_proof_l2915_291550

/-- Given two points (a, -1) and (2, 3) on a line with slope 2, prove that a = 0 -/
theorem line_slope_proof (a : ℝ) : 
  (3 - (-1)) / (2 - a) = 2 → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_proof_l2915_291550


namespace NUMINAMATH_CALUDE_marks_weekly_reading_time_l2915_291538

/-- Given Mark's daily reading time and weekly increase, prove his total weekly reading time -/
theorem marks_weekly_reading_time 
  (daily_reading_time : ℕ) 
  (weekly_increase : ℕ) 
  (h1 : daily_reading_time = 2)
  (h2 : weekly_increase = 4) :
  daily_reading_time * 7 + weekly_increase = 18 := by
  sorry

#check marks_weekly_reading_time

end NUMINAMATH_CALUDE_marks_weekly_reading_time_l2915_291538


namespace NUMINAMATH_CALUDE_birdhouse_cost_theorem_l2915_291591

/-- Calculates the total cost of building birdhouses -/
def total_cost_birdhouses (small_count large_count : ℕ) 
  (small_plank_req large_plank_req : ℕ) 
  (small_nail_req large_nail_req : ℕ) 
  (small_plank_cost large_plank_cost nail_cost : ℚ) 
  (discount_threshold : ℕ) (discount_rate : ℚ) : ℚ :=
  let total_small_planks := small_count * small_plank_req
  let total_large_planks := large_count * large_plank_req
  let total_nails := small_count * small_nail_req + large_count * large_nail_req
  let plank_cost := total_small_planks * small_plank_cost + total_large_planks * large_plank_cost
  let nail_cost_before_discount := total_nails * nail_cost
  let nail_cost_after_discount := 
    if total_nails > discount_threshold
    then nail_cost_before_discount * (1 - discount_rate)
    else nail_cost_before_discount
  plank_cost + nail_cost_after_discount

theorem birdhouse_cost_theorem :
  total_cost_birdhouses 3 2 7 10 20 36 3 5 (5/100) 100 (1/10) = 16894/100 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_cost_theorem_l2915_291591


namespace NUMINAMATH_CALUDE_f_difference_l2915_291594

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 7*x

-- State the theorem
theorem f_difference : f 3 - f (-3) = 42 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l2915_291594


namespace NUMINAMATH_CALUDE_cars_sold_per_day_second_period_l2915_291552

def total_quota : ℕ := 50
def total_days : ℕ := 30
def first_period : ℕ := 3
def second_period : ℕ := 4
def cars_per_day_first_period : ℕ := 5
def remaining_cars : ℕ := 23

theorem cars_sold_per_day_second_period :
  let cars_sold_first_period := first_period * cars_per_day_first_period
  let remaining_after_first_period := total_quota - cars_sold_first_period
  let cars_to_sell_second_period := remaining_after_first_period - remaining_cars
  cars_to_sell_second_period / second_period = 3 := by sorry

end NUMINAMATH_CALUDE_cars_sold_per_day_second_period_l2915_291552


namespace NUMINAMATH_CALUDE_swimming_speed_calculation_l2915_291506

/-- Represents the swimming scenario with a stream -/
structure SwimmingScenario where
  stream_speed : ℝ
  upstream_time : ℝ
  downstream_time : ℝ
  swimming_speed : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : SwimmingScenario) : Prop :=
  s.stream_speed = 3 ∧ s.upstream_time = 2 * s.downstream_time

/-- The theorem to be proved -/
theorem swimming_speed_calculation (s : SwimmingScenario) :
  problem_conditions s → s.swimming_speed = 9 := by
  sorry


end NUMINAMATH_CALUDE_swimming_speed_calculation_l2915_291506


namespace NUMINAMATH_CALUDE_triangle_equality_l2915_291529

/-- Given a triangle ABC with sides a, b, c opposite to angles α, β, γ respectively,
    and circumradius R, prove that if the given equation holds, then the triangle is equilateral. -/
theorem triangle_equality (a b c R : ℝ) (α β γ : ℝ) : 
  a > 0 → b > 0 → c > 0 → R > 0 →
  0 < α ∧ α < π → 0 < β ∧ β < π → 0 < γ ∧ γ < π →
  α + β + γ = π →
  (a * Real.cos α + b * Real.cos β + c * Real.cos γ) / (a * Real.sin β + b * Real.sin γ + c * Real.sin α) = (a + b + c) / (9 * R) →
  α = β ∧ β = γ ∧ γ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_l2915_291529


namespace NUMINAMATH_CALUDE_olympic_torch_relay_l2915_291574

/-- The total number of cities -/
def total_cities : ℕ := 8

/-- The number of cities to be selected for the relay route -/
def selected_cities : ℕ := 6

/-- The number of ways to select exactly one city from two cities -/
def select_one_from_two : ℕ := 2

/-- The number of ways to select 5 cities from 6 cities -/
def select_five_from_six : ℕ := 6

/-- The number of ways to select 4 cities from 6 cities -/
def select_four_from_six : ℕ := 15

/-- The number of permutations of 6 cities -/
def permutations_of_six : ℕ := 720

theorem olympic_torch_relay :
  (
    /- Condition 1 -/
    (select_one_from_two * select_five_from_six = 12) ∧
    (12 * permutations_of_six = 8640)
  ) ∧
  (
    /- Condition 2 -/
    (select_one_from_two * select_five_from_six + select_four_from_six = 27) ∧
    (27 * permutations_of_six = 19440)
  ) := by sorry

end NUMINAMATH_CALUDE_olympic_torch_relay_l2915_291574


namespace NUMINAMATH_CALUDE_max_value_inequality_l2915_291518

theorem max_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ k : ℝ, (a + b + c) * (1 / a + 1 / (b + c)) ≥ k) ↔ k ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2915_291518


namespace NUMINAMATH_CALUDE_third_equals_sixth_implies_seven_odd_terms_sum_128_implies_eight_and_max_term_l2915_291599

-- For the first part of the problem
theorem third_equals_sixth_implies_seven (n : ℕ) :
  (Nat.choose n 2 = Nat.choose n 5) → n = 7 := by sorry

-- For the second part of the problem
theorem odd_terms_sum_128_implies_eight_and_max_term (n : ℕ) (x : ℝ) :
  (2^(n-1) = 128) →
  n = 8 ∧
  (Nat.choose 8 4 * x^4 * x^(2/3) = 70 * x^4 * x^(2/3)) := by sorry

end NUMINAMATH_CALUDE_third_equals_sixth_implies_seven_odd_terms_sum_128_implies_eight_and_max_term_l2915_291599


namespace NUMINAMATH_CALUDE_not_parabola_l2915_291522

/-- A conic section represented by the equation x^2 + ky^2 = 1 -/
structure ConicSection (k : ℝ) where
  x : ℝ
  y : ℝ
  eq : x^2 + k * y^2 = 1

/-- Definition of a parabola -/
def IsParabola (c : ConicSection k) : Prop :=
  ∃ (a b h : ℝ), h ≠ 0 ∧ (c.x - a)^2 = 4 * h * (c.y - b)

/-- Theorem: For any real k, the equation x^2 + ky^2 = 1 cannot represent a parabola -/
theorem not_parabola (k : ℝ) : ¬∃ (c : ConicSection k), IsParabola c := by
  sorry

end NUMINAMATH_CALUDE_not_parabola_l2915_291522


namespace NUMINAMATH_CALUDE_unpainted_area_crossed_boards_l2915_291517

/-- The area of the unpainted region when two boards cross at 45 degrees -/
theorem unpainted_area_crossed_boards (width1 width2 : ℝ) (angle : ℝ) :
  width1 = 5 →
  width2 = 7 →
  angle = π / 4 →
  let projected_length := width2
  let overlap_height := width2 * Real.cos angle
  let unpainted_area := projected_length * overlap_height
  unpainted_area = 49 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_area_crossed_boards_l2915_291517


namespace NUMINAMATH_CALUDE_possible_a_values_l2915_291524

theorem possible_a_values :
  ∀ (a : ℤ), 
    (∃ (b c : ℤ), ∀ (x : ℤ), (x - a) * (x - 15) + 4 = (x + b) * (x + c)) ↔ 
    (a = 16 ∨ a = 21) := by
sorry

end NUMINAMATH_CALUDE_possible_a_values_l2915_291524


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2915_291578

/-- The number of coins being flipped -/
def num_coins : ℕ := 6

/-- The number of specific coins we're interested in -/
def num_specific_coins : ℕ := 3

/-- The number of possible outcomes for each coin (heads or tails) -/
def outcomes_per_coin : ℕ := 2

/-- The probability of three specific coins out of six showing the same face -/
def probability_same_face : ℚ := 1 / 4

theorem coin_flip_probability :
  (outcomes_per_coin ^ num_specific_coins * outcomes_per_coin ^ (num_coins - num_specific_coins)) /
  (outcomes_per_coin ^ num_coins) = probability_same_face :=
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2915_291578


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l2915_291501

/-- The distance traveled by a boat downstream -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Theorem: A boat with speed 13 km/hr in still water, traveling downstream
    in a stream with speed 4 km/hr for 4 hours, covers a distance of 68 km -/
theorem boat_downstream_distance :
  distance_downstream 13 4 4 = 68 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l2915_291501


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2915_291595

theorem complex_modulus_problem (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : 
  Complex.abs z = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2915_291595


namespace NUMINAMATH_CALUDE_product_mod_30_l2915_291567

theorem product_mod_30 : ∃ m : ℕ, 0 ≤ m ∧ m < 30 ∧ (33 * 77 * 99) % 30 = m ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_30_l2915_291567


namespace NUMINAMATH_CALUDE_distance_between_points_l2915_291586

theorem distance_between_points : 
  ∀ (A B : ℝ), A = -4 ∧ B = 2 → |B - A| = |2 - (-4)| := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l2915_291586


namespace NUMINAMATH_CALUDE_cubes_passed_in_specific_solid_l2915_291585

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem stating the number of cubes an internal diagonal passes through
    in a 105 × 140 × 195 rectangular solid -/
theorem cubes_passed_in_specific_solid :
  cubes_passed_by_diagonal 105 140 195 = 395 := by
  sorry

end NUMINAMATH_CALUDE_cubes_passed_in_specific_solid_l2915_291585


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2915_291505

/-- The number of different arrangements for selecting 4 students (1 girl and 3 boys) 
    from 8 students (6 boys and 2 girls) by stratified sampling based on gender, 
    with a girl as the first runner. -/
def stratifiedSamplingArrangements : ℕ := sorry

/-- The total number of students -/
def totalStudents : ℕ := 8

/-- The number of boys -/
def numBoys : ℕ := 6

/-- The number of girls -/
def numGirls : ℕ := 2

/-- The number of students to be selected -/
def selectedStudents : ℕ := 4

/-- The number of boys to be selected -/
def selectedBoys : ℕ := 3

/-- The number of girls to be selected -/
def selectedGirls : ℕ := 1

theorem stratified_sampling_theorem : 
  stratifiedSamplingArrangements = 240 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2915_291505


namespace NUMINAMATH_CALUDE_enclosing_polygons_sides_l2915_291596

/-- The number of sides of the central polygon -/
def central_sides : ℕ := 12

/-- The number of polygons enclosing the central polygon -/
def enclosing_polygons : ℕ := 12

/-- The number of enclosing polygons meeting at each vertex of the central polygon -/
def polygons_at_vertex : ℕ := 4

/-- The number of sides of each enclosing polygon -/
def n : ℕ := 12

/-- Theorem stating that n must be 12 for the given configuration -/
theorem enclosing_polygons_sides (h1 : central_sides = 12)
                                 (h2 : enclosing_polygons = 12)
                                 (h3 : polygons_at_vertex = 4) :
  n = 12 := by sorry

end NUMINAMATH_CALUDE_enclosing_polygons_sides_l2915_291596


namespace NUMINAMATH_CALUDE_no_nonzero_ending_product_zero_l2915_291504

theorem no_nonzero_ending_product_zero (x y : ℤ) : 
  (x % 10 ≠ 0) → (y % 10 ≠ 0) → (x * y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_nonzero_ending_product_zero_l2915_291504


namespace NUMINAMATH_CALUDE_symmetric_point_sum_l2915_291525

/-- A point is symmetric to the line x+y+1=0 if its symmetric point is also on this line -/
def is_symmetric_point (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), x + y + 1 = 0 ∧ (a + x) / 2 + (b + y) / 2 + 1 = 0

/-- Theorem: If a point (a,b) is symmetric to the line x+y+1=0 and its symmetric point
    is also on this line, then a+b=-1 -/
theorem symmetric_point_sum (a b : ℝ) (h : is_symmetric_point a b) : a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_sum_l2915_291525


namespace NUMINAMATH_CALUDE_octal_subtraction_l2915_291511

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to octal --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- Theorem: 53₈ - 27₈ = 24₈ in base 8 --/
theorem octal_subtraction :
  decimal_to_octal (octal_to_decimal 53 - octal_to_decimal 27) = 24 := by sorry

end NUMINAMATH_CALUDE_octal_subtraction_l2915_291511


namespace NUMINAMATH_CALUDE_ac_length_l2915_291536

/-- An isosceles trapezoid with specific measurements -/
structure IsoscelesTrapezoid where
  /-- Length of base AB -/
  ab : ℝ
  /-- Length of top side CD -/
  cd : ℝ
  /-- Length of leg AD (equal to BC) -/
  ad : ℝ
  /-- Constraint that AB > CD -/
  h_ab_gt_cd : ab > cd

/-- Theorem: In the given isosceles trapezoid, AC = 17 -/
theorem ac_length (t : IsoscelesTrapezoid) 
  (h_ab : t.ab = 21)
  (h_cd : t.cd = 9)
  (h_ad : t.ad = 10) : 
  Real.sqrt ((21 - 9) ^ 2 / 4 + 8 ^ 2) = 17 := by
  sorry


end NUMINAMATH_CALUDE_ac_length_l2915_291536


namespace NUMINAMATH_CALUDE_history_not_statistics_l2915_291560

theorem history_not_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ)
  (h_total : total = 90)
  (h_history : history = 36)
  (h_statistics : statistics = 32)
  (h_history_or_statistics : history_or_statistics = 57) :
  history - (history + statistics - history_or_statistics) = 25 := by
  sorry

end NUMINAMATH_CALUDE_history_not_statistics_l2915_291560


namespace NUMINAMATH_CALUDE_probability_of_two_packages_l2915_291587

/-- The number of tablets in a new package -/
def n : ℕ := 10

/-- The probability of having exactly two packages of tablets -/
def probability_two_packages : ℚ := (2^n - 1) / (2^(n-1) * n)

/-- Theorem stating the probability of having exactly two packages of tablets -/
theorem probability_of_two_packages :
  probability_two_packages = (2^n - 1) / (2^(n-1) * n) := by sorry

end NUMINAMATH_CALUDE_probability_of_two_packages_l2915_291587


namespace NUMINAMATH_CALUDE_remainder_theorem_l2915_291532

theorem remainder_theorem (x y u v : ℤ) (hx : 0 < x) (hy : 0 < y) (h_div : x = u * y + v) (h_rem : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2915_291532


namespace NUMINAMATH_CALUDE_sum_xy_is_zero_l2915_291534

theorem sum_xy_is_zero (x y : ℝ) 
  (h : (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = 1) : 
  x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_xy_is_zero_l2915_291534


namespace NUMINAMATH_CALUDE_binomial_coefficient_formula_l2915_291508

theorem binomial_coefficient_formula (n k : ℕ) (h : k ≤ n) :
  Nat.choose n k = n.factorial / ((n - k).factorial * k.factorial) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_formula_l2915_291508


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2915_291551

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0)
  (h4 : s + 2*y = 3*s) -- Outer square side length
  (h5 : x + y = 3*s) -- Outer square side length
  (h6 : (3*s)^2 = 9*s^2) -- Area of outer square is 9 times inner square
  : x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2915_291551


namespace NUMINAMATH_CALUDE_max_min_sum_on_interval_l2915_291540

def f (x : ℝ) := 2 * x^2 - 6 * x + 1

theorem max_min_sum_on_interval :
  ∃ (m M : ℝ),
    (∀ x ∈ Set.Icc (-1) 1, m ≤ f x ∧ f x ≤ M) ∧
    (∃ x₁ ∈ Set.Icc (-1) 1, f x₁ = m) ∧
    (∃ x₂ ∈ Set.Icc (-1) 1, f x₂ = M) ∧
    M + m = 6 :=
sorry

end NUMINAMATH_CALUDE_max_min_sum_on_interval_l2915_291540


namespace NUMINAMATH_CALUDE_youngest_member_age_l2915_291526

theorem youngest_member_age (n : ℕ) (current_avg : ℚ) (birth_avg : ℚ) 
  (h1 : n = 5)
  (h2 : current_avg = 20)
  (h3 : birth_avg = 25/2) :
  (n : ℚ) * current_avg - (n - 1 : ℚ) * birth_avg = 10 := by
  sorry

end NUMINAMATH_CALUDE_youngest_member_age_l2915_291526


namespace NUMINAMATH_CALUDE_smallest_n_square_fourth_power_l2915_291500

/-- The smallest positive integer n such that 5n is a perfect square and 3n is a perfect fourth power is 75. -/
theorem smallest_n_square_fourth_power : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 5 * n = k^2) ∧ 
    (∃ (m : ℕ), 3 * n = m^4)) ∧
  (∀ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 5 * n = k^2) ∧ 
    (∃ (m : ℕ), 3 * n = m^4) → 
    n ≥ 75) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_square_fourth_power_l2915_291500


namespace NUMINAMATH_CALUDE_prob_three_l_is_one_fiftyfifth_l2915_291582

/-- The number of cards in the deck -/
def total_cards : ℕ := 12

/-- The number of L cards in the deck -/
def l_cards : ℕ := 4

/-- The number of cards drawn -/
def cards_drawn : ℕ := 3

/-- The probability of drawing 3 L cards without replacement -/
def prob_three_l : ℚ := (l_cards.choose cards_drawn : ℚ) / (total_cards.choose cards_drawn : ℚ)

theorem prob_three_l_is_one_fiftyfifth : prob_three_l = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_l_is_one_fiftyfifth_l2915_291582


namespace NUMINAMATH_CALUDE_cubic_function_min_value_l2915_291576

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + c

-- State the theorem
theorem cubic_function_min_value 
  (c : ℝ) 
  (h_max : ∃ x, f c x ≤ 28 ∧ ∀ y, f c y ≤ f c x) : 
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f c x ≤ f c y) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f c x = -4) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_min_value_l2915_291576


namespace NUMINAMATH_CALUDE_sum_binary_digits_310_l2915_291558

def sum_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

theorem sum_binary_digits_310 : sum_binary_digits 310 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_binary_digits_310_l2915_291558


namespace NUMINAMATH_CALUDE_remainder_problem_l2915_291557

theorem remainder_problem (L S R : ℕ) : 
  L - S = 2395 → 
  S = 476 → 
  L = 6 * S + R → 
  R < S → 
  R = 15 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2915_291557


namespace NUMINAMATH_CALUDE_sum_equality_l2915_291589

def sum_ascending (n : ℕ) : ℕ := (n * (n + 1)) / 2

def sum_descending (n : ℕ) : ℕ := 
  if n = 0 then 0 else n + sum_descending (n - 1)

theorem sum_equality : 
  sum_ascending 1000 = sum_descending 1000 :=
by sorry

end NUMINAMATH_CALUDE_sum_equality_l2915_291589


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2915_291509

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 1 ∧ x₁^2 + 4*x₁ - 5 = 0 ∧ x₂^2 + 4*x₂ - 5 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 1/3 ∧ y₂ = -1 ∧ 3*y₁^2 + 2*y₁ = 1 ∧ 3*y₂^2 + 2*y₂ = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2915_291509


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_l2915_291592

def A (m : ℝ) : Set ℝ := {m + 1, -3}
def B (m : ℝ) : Set ℝ := {2*m + 1, m - 3}

theorem intersection_implies_m_value :
  ∀ m : ℝ, (A m ∩ B m = {-3}) → m = -2 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_l2915_291592


namespace NUMINAMATH_CALUDE_special_function_inequality_l2915_291577

/-- A function that is increasing on (1,+∞) and has F(x) = f(x+1) symmetrical about the y-axis -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (∀ x, f (-x + 1) = f (x + 1))

/-- Theorem: For a special function f, f(-1) > f(2) -/
theorem special_function_inequality (f : ℝ → ℝ) (h : SpecialFunction f) : f (-1) > f 2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l2915_291577


namespace NUMINAMATH_CALUDE_outfit_choices_l2915_291531

theorem outfit_choices (shirts : ℕ) (skirts : ℕ) (dresses : ℕ) : 
  shirts = 4 → skirts = 3 → dresses = 2 → shirts * skirts + dresses = 14 := by
  sorry

end NUMINAMATH_CALUDE_outfit_choices_l2915_291531


namespace NUMINAMATH_CALUDE_gingers_children_l2915_291581

/-- The number of cakes Ginger bakes for each child per year -/
def cakes_per_child : ℕ := 4

/-- The number of cakes Ginger bakes for her husband per year -/
def cakes_for_husband : ℕ := 6

/-- The number of cakes Ginger bakes for her parents per year -/
def cakes_for_parents : ℕ := 2

/-- The total number of cakes Ginger bakes in 10 years -/
def total_cakes : ℕ := 160

/-- The number of years over which the total cakes are counted -/
def years : ℕ := 10

/-- Ginger's number of children -/
def num_children : ℕ := 2

theorem gingers_children :
  num_children * cakes_per_child * years + cakes_for_husband * years + cakes_for_parents * years = total_cakes :=
by sorry

end NUMINAMATH_CALUDE_gingers_children_l2915_291581


namespace NUMINAMATH_CALUDE_polar_coords_of_negative_one_negative_one_l2915_291588

/-- Prove that the polar coordinates of the point P(-1, -1) are (√2, 5π/4) -/
theorem polar_coords_of_negative_one_negative_one :
  let x : ℝ := -1
  let y : ℝ := -1
  let ρ : ℝ := Real.sqrt 2
  let θ : ℝ := 5 * Real.pi / 4
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_polar_coords_of_negative_one_negative_one_l2915_291588


namespace NUMINAMATH_CALUDE_inverse_exponential_is_logarithm_l2915_291544

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_exponential_is_logarithm (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : f a 2 = 1) : 
  ∀ x, f a x = Real.log x / Real.log 2 := by
sorry

end NUMINAMATH_CALUDE_inverse_exponential_is_logarithm_l2915_291544


namespace NUMINAMATH_CALUDE_problem_statement_l2915_291593

noncomputable section

def f (a b x : ℝ) : ℝ := Real.exp x - a * x^2 - b * x - 1

def g (a b : ℝ) : ℝ → ℝ := λ x ↦ Real.exp x - 2 * a * x - b

theorem problem_statement (a b : ℝ) :
  (∀ x, |x - a| ≥ f a b x) →
  (∀ x, (Real.exp 1 - 1) * x - 1 = (f a b x - f a b 1) / (x - 1) + f a b 1) →
  (a ≤ 1/2) ∧
  (a = 0 ∧ b = 1) ∧
  (∀ x ∈ Set.Icc 0 1,
    g a b x ≥ min (1 - b)
      (min (2*a - 2*a * Real.log (2*a) - b)
        (1 - 2*a - b))) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2915_291593


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2915_291561

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo 1 2 = {x | a * x^2 + b * x + c < 0}) : 
  Set.Ioo (1/2) 1 = {x | c * x^2 + b * x + a < 0} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2915_291561


namespace NUMINAMATH_CALUDE_consecutive_sum_at_least_17_l2915_291566

theorem consecutive_sum_at_least_17 (a : Fin 10 → ℕ) 
  (h_perm : Function.Bijective a) 
  (h_range : ∀ i, a i ∈ Finset.range 11 \ {0}) : 
  ∃ i : Fin 10, a i + a (i + 1) + a (i + 2) ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_consecutive_sum_at_least_17_l2915_291566


namespace NUMINAMATH_CALUDE_otimes_nested_l2915_291573

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^2 + y

/-- Theorem: a ⊗ (a ⊗ a) = 2a^2 + a -/
theorem otimes_nested (a : ℝ) : otimes a (otimes a a) = 2 * a^2 + a := by
  sorry

end NUMINAMATH_CALUDE_otimes_nested_l2915_291573


namespace NUMINAMATH_CALUDE_binomial_18_10_l2915_291584

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 47190 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l2915_291584


namespace NUMINAMATH_CALUDE_ellipse_left_right_vertices_l2915_291569

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2/16 + y^2/7 = 1

-- Define the left and right vertices
def left_right_vertices : Set (ℝ × ℝ) := {(-4, 0), (4, 0)}

-- Theorem statement
theorem ellipse_left_right_vertices :
  ∀ (p : ℝ × ℝ), p ∈ left_right_vertices ↔ 
    (ellipse_equation p.1 p.2 ∧ 
     ∀ (q : ℝ × ℝ), ellipse_equation q.1 q.2 → abs q.1 ≤ abs p.1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_left_right_vertices_l2915_291569


namespace NUMINAMATH_CALUDE_work_completion_time_l2915_291546

theorem work_completion_time (aarti_rate ramesh_rate : ℚ) 
  (h1 : aarti_rate = 1 / 6)
  (h2 : ramesh_rate = 1 / 8)
  (h3 : (aarti_rate + ramesh_rate) * 3 = 1) :
  3 = 3 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2915_291546


namespace NUMINAMATH_CALUDE_city_population_theorem_l2915_291548

def city_population (initial_population immigration emigration pregnancy_rate twin_rate : ℕ) : ℕ :=
  let population_after_migration := initial_population + immigration - emigration
  let pregnancies := population_after_migration / 8
  let twin_pregnancies := pregnancies / 4
  let single_pregnancies := pregnancies - twin_pregnancies
  let births := single_pregnancies + 2 * twin_pregnancies
  population_after_migration + births

theorem city_population_theorem :
  city_population 300000 50000 30000 8 4 = 370000 := by
  sorry

end NUMINAMATH_CALUDE_city_population_theorem_l2915_291548


namespace NUMINAMATH_CALUDE_pet_insurance_cost_l2915_291597

/-- Calculates the monthly cost of pet insurance given the surgery cost, insurance duration,
    coverage percentage, and total savings. -/
def monthly_insurance_cost (surgery_cost : ℚ) (insurance_duration : ℕ) 
    (coverage_percent : ℚ) (total_savings : ℚ) : ℚ :=
  let insurance_payment := surgery_cost * coverage_percent
  let total_insurance_cost := insurance_payment - total_savings
  total_insurance_cost / insurance_duration

/-- Theorem stating that the monthly insurance cost is $20 given the specified conditions. -/
theorem pet_insurance_cost :
  monthly_insurance_cost 5000 24 (4/5) 3520 = 20 := by
  sorry

end NUMINAMATH_CALUDE_pet_insurance_cost_l2915_291597


namespace NUMINAMATH_CALUDE_smallest_binary_palindrome_l2915_291520

/-- Checks if a natural number is a palindrome in the given base. -/
def is_palindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def to_base (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- The number 33 in decimal. -/
def target_number : ℕ := 33

theorem smallest_binary_palindrome :
  (is_palindrome target_number 2) ∧
  (∃ (b : ℕ), b > 2 ∧ is_palindrome target_number b) ∧
  (∀ (m : ℕ), m < target_number →
    ¬(is_palindrome m 2 ∧ (∃ (b : ℕ), b > 2 ∧ is_palindrome m b))) ∧
  (to_base target_number 2 = [1, 0, 0, 0, 0, 1]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_binary_palindrome_l2915_291520


namespace NUMINAMATH_CALUDE_cave_door_weight_l2915_291553

/-- The weight already on the switch in pounds -/
def initial_weight : ℕ := 234

/-- The additional weight needed in pounds -/
def additional_weight : ℕ := 478

/-- The total weight needed to open the cave doors in pounds -/
def total_weight : ℕ := initial_weight + additional_weight

/-- Theorem stating that the total weight needed to open the cave doors is 712 pounds -/
theorem cave_door_weight : total_weight = 712 := by
  sorry

end NUMINAMATH_CALUDE_cave_door_weight_l2915_291553


namespace NUMINAMATH_CALUDE_min_value_expression_l2915_291523

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x^2 + 4*x*y + 9*y^2 + 8*y*z + 3*z^2 ≥ 9^(10/9) ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    x₀^2 + 4*x₀*y₀ + 9*y₀^2 + 8*y₀*z₀ + 3*z₀^2 = 9^(10/9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2915_291523


namespace NUMINAMATH_CALUDE_debby_water_bottles_l2915_291547

/-- The number of bottles Debby drinks per day -/
def bottles_per_day : ℕ := 5

/-- The number of days the water would last -/
def days_lasting : ℕ := 71

/-- The total number of bottles Debby bought -/
def total_bottles : ℕ := bottles_per_day * days_lasting

theorem debby_water_bottles : total_bottles = 355 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l2915_291547


namespace NUMINAMATH_CALUDE_sports_parade_children_count_l2915_291559

theorem sports_parade_children_count :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 150 ∧ n % 8 = 5 ∧ n % 10 = 7 ∧ n = 125 := by
  sorry

end NUMINAMATH_CALUDE_sports_parade_children_count_l2915_291559


namespace NUMINAMATH_CALUDE_bales_in_barn_l2915_291568

/-- The number of bales originally in the barn -/
def original_bales : ℕ := sorry

/-- The number of bales Keith stacked today -/
def keith_bales : ℕ := 67

/-- The total number of bales in the barn now -/
def total_bales : ℕ := 89

theorem bales_in_barn :
  original_bales + keith_bales = total_bales ∧ original_bales = 22 :=
by sorry

end NUMINAMATH_CALUDE_bales_in_barn_l2915_291568


namespace NUMINAMATH_CALUDE_quadratic_root_coefficients_l2915_291562

theorem quadratic_root_coefficients (b c : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1 - Complex.I * Real.sqrt 2) ^ 2 + b * (1 - Complex.I * Real.sqrt 2) + c = 0 →
  b = -2 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_coefficients_l2915_291562


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l2915_291537

/-- The number of ways to select a non-empty subset of fruits from a given number of identical apples and oranges, such that at least 2 oranges are selected. -/
def fruitBasketCombinations (apples oranges : ℕ) : ℕ :=
  (apples + 1) * (oranges - 1)

/-- Theorem stating that the number of fruit basket combinations with 4 apples and 12 oranges is 55. -/
theorem fruit_basket_problem :
  fruitBasketCombinations 4 12 = 55 := by
  sorry

#eval fruitBasketCombinations 4 12

end NUMINAMATH_CALUDE_fruit_basket_problem_l2915_291537


namespace NUMINAMATH_CALUDE_fraction_inequality_l2915_291507

theorem fraction_inequality (a b : ℝ) :
  ((b > 0 ∧ 0 > a) ∨ (0 > a ∧ a > b) ∨ (a > b ∧ b > 0)) → (1 / a < 1 / b) ∧
  (a > 0 ∧ 0 > b) → ¬(1 / a < 1 / b) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2915_291507


namespace NUMINAMATH_CALUDE_right_triangle_area_l2915_291513

theorem right_triangle_area (hypotenuse base : ℝ) (h1 : hypotenuse = 15) (h2 : base = 9) :
  let height : ℝ := Real.sqrt (hypotenuse^2 - base^2)
  let area : ℝ := (base * height) / 2
  area = 54 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2915_291513


namespace NUMINAMATH_CALUDE_final_student_count_l2915_291564

theorem final_student_count (initial_students leaving_students new_students : ℕ) :
  initial_students = 11 →
  leaving_students = 6 →
  new_students = 42 →
  initial_students - leaving_students + new_students = 47 :=
by
  sorry

end NUMINAMATH_CALUDE_final_student_count_l2915_291564


namespace NUMINAMATH_CALUDE_sphere_centers_distance_l2915_291542

/-- The distance between the centers of two spheres with masses M and m, 
    where a point B exists such that both spheres exert equal gravitational force on it,
    and A is a point between the centers with distance d from B. -/
theorem sphere_centers_distance (M m d : ℝ) (hM : M > 0) (hm : m > 0) (hd : d > 0) : 
  ∃ (distance : ℝ), distance = d / 2 * (M - m) / Real.sqrt (M * m) :=
sorry

end NUMINAMATH_CALUDE_sphere_centers_distance_l2915_291542


namespace NUMINAMATH_CALUDE_count_white_rhinos_l2915_291515

/-- Given information about rhinos and their weights, prove the number of white rhinos --/
theorem count_white_rhinos (white_rhino_weight : ℕ) (black_rhino_count : ℕ) (black_rhino_weight : ℕ) (total_weight : ℕ) : 
  white_rhino_weight = 5100 →
  black_rhino_count = 8 →
  black_rhino_weight = 2000 →
  total_weight = 51700 →
  (total_weight - black_rhino_count * black_rhino_weight) / white_rhino_weight = 7 := by
sorry

end NUMINAMATH_CALUDE_count_white_rhinos_l2915_291515


namespace NUMINAMATH_CALUDE_f_integer_iff_l2915_291579

def f (x : ℝ) : ℝ := (1 + x) ^ (1/3) + (3 - x) ^ (1/3)

theorem f_integer_iff (x : ℝ) : 
  ∃ (n : ℤ), f x = n ↔ 
  (x = 1 + Real.sqrt 5 ∨ 
   x = 1 - Real.sqrt 5 ∨ 
   x = 1 + (10/9) * Real.sqrt 3 ∨ 
   x = 1 - (10/9) * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_f_integer_iff_l2915_291579


namespace NUMINAMATH_CALUDE_three_digit_numbers_sum_divisibility_l2915_291516

theorem three_digit_numbers_sum_divisibility :
  ∃ (a b c d : ℕ),
    100 ≤ a ∧ a < 1000 ∧
    100 ≤ b ∧ b < 1000 ∧
    100 ≤ c ∧ c < 1000 ∧
    100 ≤ d ∧ d < 1000 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∃ (j : ℕ), a / 100 = j ∧ b / 100 = j ∧ c / 100 = j ∧ d / 100 = j) ∧
    (∃ (s : ℕ), s = a + b + c + d ∧ s % a = 0 ∧ s % b = 0 ∧ s % c = 0) ∧
    a = 108 ∧ b = 135 ∧ c = 180 ∧ d = 117 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_sum_divisibility_l2915_291516


namespace NUMINAMATH_CALUDE_factorization_problems_l2915_291545

theorem factorization_problems (x : ℝ) : 
  (2 * x^2 - 8 = 2 * (x + 2) * (x - 2)) ∧ 
  (2 * x^2 + 2 * x + (1/2) = 2 * (x + 1/2)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l2915_291545


namespace NUMINAMATH_CALUDE_net_percentage_gain_calculation_l2915_291539

/-- Calculate the net percentage gain from buying and selling glass bowls and ceramic plates --/
theorem net_percentage_gain_calculation 
  (glass_bowls_bought : ℕ) 
  (glass_bowls_price : ℚ) 
  (ceramic_plates_bought : ℕ) 
  (ceramic_plates_price : ℚ) 
  (discount_rate : ℚ) 
  (glass_bowls_sold : ℕ) 
  (glass_bowls_sell_price : ℚ) 
  (ceramic_plates_sold : ℕ) 
  (ceramic_plates_sell_price : ℚ) 
  (glass_bowls_broken : ℕ) 
  (ceramic_plates_broken : ℕ) :
  glass_bowls_bought = 250 →
  glass_bowls_price = 18 →
  ceramic_plates_bought = 150 →
  ceramic_plates_price = 25 →
  discount_rate = 5 / 100 →
  glass_bowls_sold = 200 →
  glass_bowls_sell_price = 25 →
  ceramic_plates_sold = 120 →
  ceramic_plates_sell_price = 32 →
  glass_bowls_broken = 30 →
  ceramic_plates_broken = 10 →
  ∃ (net_percentage_gain : ℚ), 
    abs (net_percentage_gain - (271 / 10000 : ℚ)) < (1 / 1000 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_net_percentage_gain_calculation_l2915_291539


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l2915_291575

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x > 1}

-- State the theorem
theorem domain_of_composite_function :
  {x : ℝ | ∃ y ∈ domain_f, y = 2*x + 1} = {x : ℝ | x > 0} := by sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l2915_291575


namespace NUMINAMATH_CALUDE_power_functions_inequality_l2915_291519

theorem power_functions_inequality (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) :
  (((x₁ + x₂) / 2) ^ 2 < (x₁^2 + x₂^2) / 2) ∧
  (2 / (x₁ + x₂) < (1 / x₁ + 1 / x₂) / 2) := by
  sorry

end NUMINAMATH_CALUDE_power_functions_inequality_l2915_291519


namespace NUMINAMATH_CALUDE_chord_equation_l2915_291572

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 5 = 1

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define a chord that is bisected by P
def bisected_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
  (x₁ + x₂) / 2 = P.1 ∧ (y₁ + y₂) / 2 = P.2

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 5 * x - 3 * y - 13 = 0

theorem chord_equation :
  ∀ x₁ y₁ x₂ y₂ : ℝ, bisected_chord x₁ y₁ x₂ y₂ →
  ∀ x y : ℝ, line_equation x y ↔ (y - P.2) = (5/3) * (x - P.1) :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l2915_291572


namespace NUMINAMATH_CALUDE_three_men_three_women_arrangements_l2915_291556

/-- The number of ways to arrange n men and n women in a row, such that no two men or two women are adjacent -/
def alternating_arrangements (n : ℕ) : ℕ :=
  2 * (n.factorial * n.factorial)

theorem three_men_three_women_arrangements :
  alternating_arrangements 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_three_men_three_women_arrangements_l2915_291556


namespace NUMINAMATH_CALUDE_complement_of_event_A_l2915_291554

/-- The total number of products in the batch -/
def total_products : ℕ := 10

/-- Event A: There are at least 2 defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- The complement of event A -/
def complement_A (defective : ℕ) : Prop := defective ≤ 1

/-- Theorem stating that the complement of event A is correctly defined -/
theorem complement_of_event_A :
  ∀ defective : ℕ, defective ≤ total_products →
    (¬ event_A defective ↔ complement_A defective) :=
by sorry

end NUMINAMATH_CALUDE_complement_of_event_A_l2915_291554

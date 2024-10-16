import Mathlib

namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3144_314485

theorem complex_fraction_simplification :
  (7 + 16 * Complex.I) / (4 - 5 * Complex.I) = -52/41 + (99/41) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3144_314485


namespace NUMINAMATH_CALUDE_log_equation_equivalence_l3144_314424

theorem log_equation_equivalence (x : ℝ) :
  x > 0 → ((Real.log x / Real.log 4) * (Real.log 5 / Real.log x) = Real.log 5 / Real.log 4 ↔ x ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_equivalence_l3144_314424


namespace NUMINAMATH_CALUDE_ocean_area_scientific_notation_l3144_314471

theorem ocean_area_scientific_notation : 
  361000000 = 3.61 * (10 ^ 8) := by sorry

end NUMINAMATH_CALUDE_ocean_area_scientific_notation_l3144_314471


namespace NUMINAMATH_CALUDE_diameter_circumference_relation_l3144_314481

theorem diameter_circumference_relation (c : ℝ) (d : ℝ) (π : ℝ) : c > 0 → d > 0 → π > 0 → c = π * d → d = (1 / π) * c := by
  sorry

end NUMINAMATH_CALUDE_diameter_circumference_relation_l3144_314481


namespace NUMINAMATH_CALUDE_polar_sin_is_circle_l3144_314401

-- Define the polar coordinate equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = Real.sin θ

-- Define the transformation from polar to Cartesian coordinates
def to_cartesian (x y ρ θ : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Define a circle in Cartesian coordinates
def is_circle (x y : ℝ) (h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem polar_sin_is_circle :
  ∃ (h k r : ℝ), ∀ (x y ρ θ : ℝ),
    polar_equation ρ θ → to_cartesian x y ρ θ →
    is_circle x y h k r :=
sorry

end NUMINAMATH_CALUDE_polar_sin_is_circle_l3144_314401


namespace NUMINAMATH_CALUDE_min_time_circular_chain_no_faster_solution_l3144_314426

/-- Represents a chain piece with a certain number of links -/
structure ChainPiece where
  links : ℕ

/-- Represents the time required for chain operations -/
structure ChainOperations where
  cutTime : ℕ
  joinTime : ℕ

/-- Calculates the minimum time required to form a circular chain -/
def minTimeToCircularChain (pieces : List ChainPiece) (ops : ChainOperations) : ℕ :=
  sorry

/-- Theorem stating the minimum time to form a circular chain from given pieces -/
theorem min_time_circular_chain :
  let pieces := [
    ChainPiece.mk 10,
    ChainPiece.mk 10,
    ChainPiece.mk 8,
    ChainPiece.mk 8,
    ChainPiece.mk 5,
    ChainPiece.mk 2
  ]
  let ops := ChainOperations.mk 1 2
  minTimeToCircularChain pieces ops = 15 := by
  sorry

/-- Theorem stating that it's impossible to form the circular chain in less than 15 minutes -/
theorem no_faster_solution (t : ℕ) :
  let pieces := [
    ChainPiece.mk 10,
    ChainPiece.mk 10,
    ChainPiece.mk 8,
    ChainPiece.mk 8,
    ChainPiece.mk 5,
    ChainPiece.mk 2
  ]
  let ops := ChainOperations.mk 1 2
  t < 15 → minTimeToCircularChain pieces ops ≠ t := by
  sorry

end NUMINAMATH_CALUDE_min_time_circular_chain_no_faster_solution_l3144_314426


namespace NUMINAMATH_CALUDE_factorization_equality_l3144_314466

theorem factorization_equality (a b c : ℝ) : a^2 - 2*a*b + b^2 - c^2 = (a - b + c) * (a - b - c) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3144_314466


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3144_314417

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 1 → x^2 + x - 2 > 0) ∧ 
  (∃ x : ℝ, x^2 + x - 2 > 0 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3144_314417


namespace NUMINAMATH_CALUDE_divide_by_repeating_decimal_l3144_314438

theorem divide_by_repeating_decimal : 
  let x : ℚ := 142857 / 999999
  7 / x = 49 := by sorry

end NUMINAMATH_CALUDE_divide_by_repeating_decimal_l3144_314438


namespace NUMINAMATH_CALUDE_seconds_in_misfortune_day_l3144_314494

/-- The number of minutes in a day on the island of Misfortune -/
def minutes_per_day : ℕ := 77

/-- The number of seconds in a minute on the island of Misfortune -/
def seconds_per_minute : ℕ := 91

/-- Theorem: The number of seconds in a day on the island of Misfortune is 1001 -/
theorem seconds_in_misfortune_day : 
  minutes_per_day * seconds_per_minute = 1001 := by
  sorry

end NUMINAMATH_CALUDE_seconds_in_misfortune_day_l3144_314494


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3144_314413

/-- A line passing through a point and perpendicular to another line --/
structure PerpendicularLine where
  -- The point that the line passes through
  point : ℝ × ℝ
  -- The line that our line is perpendicular to, represented by its coefficients (a, b, c) in ax + by + c = 0
  perp_line : ℝ × ℝ × ℝ

/-- The equation of a line, represented by its coefficients (a, b, c) in ax + by + c = 0 --/
def LineEquation := ℝ × ℝ × ℝ

/-- Check if a point lies on a line given by its equation --/
def point_on_line (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  let (x, y) := p
  let (a, b, c) := l
  a * x + b * y + c = 0

/-- Check if two lines are perpendicular --/
def perpendicular (l1 l2 : LineEquation) : Prop :=
  let (a1, b1, _) := l1
  let (a2, b2, _) := l2
  a1 * a2 + b1 * b2 = 0

/-- The main theorem --/
theorem perpendicular_line_equation (l : PerpendicularLine) :
  let given_line : LineEquation := (1, -2, -3)
  let result_line : LineEquation := (2, 1, -1)
  l.point = (-1, 3) ∧ perpendicular given_line (result_line) →
  point_on_line l.point result_line ∧ perpendicular given_line result_line :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3144_314413


namespace NUMINAMATH_CALUDE_time_is_one_point_two_hours_l3144_314415

/-- The number of letters in the name -/
def name_length : ℕ := 6

/-- The number of rearrangements that can be written per minute -/
def rearrangements_per_minute : ℕ := 10

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the time in hours to write all rearrangements of a name -/
def time_to_write_all_rearrangements : ℚ :=
  (name_length.factorial / rearrangements_per_minute : ℚ) / minutes_per_hour

/-- Theorem stating that the time to write all rearrangements is 1.2 hours -/
theorem time_is_one_point_two_hours :
  time_to_write_all_rearrangements = 6/5 := by sorry

end NUMINAMATH_CALUDE_time_is_one_point_two_hours_l3144_314415


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l3144_314418

theorem point_in_third_quadrant : ∃ (x y : ℝ), 
  x = Real.sin (2014 * π / 180) ∧ 
  y = Real.cos (2014 * π / 180) ∧ 
  x < 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l3144_314418


namespace NUMINAMATH_CALUDE_unique_base_system_solution_l3144_314488

/-- Represents a base-b numeral system where 1987 is written as xyz --/
structure BaseSystem where
  b : ℕ
  x : ℕ
  y : ℕ
  z : ℕ
  h1 : b > 1
  h2 : x < b ∧ y < b ∧ z < b
  h3 : x + y + z = 25
  h4 : x * b^2 + y * b + z = 1987

/-- The unique solution to the base system problem --/
theorem unique_base_system_solution :
  ∃! (s : BaseSystem), s.b = 19 ∧ s.x = 5 ∧ s.y = 9 ∧ s.z = 11 :=
sorry

end NUMINAMATH_CALUDE_unique_base_system_solution_l3144_314488


namespace NUMINAMATH_CALUDE_age_problem_solution_l3144_314457

/-- Represents the problem of finding when Anand's age was one-third of Bala's age -/
def age_problem (x : ℕ) : Prop :=
  let anand_current_age : ℕ := 15
  let bala_current_age : ℕ := anand_current_age + 10
  let anand_past_age : ℕ := anand_current_age - x
  let bala_past_age : ℕ := bala_current_age - x
  anand_past_age = bala_past_age / 3

/-- Theorem stating that 10 years ago, Anand's age was one-third of Bala's age -/
theorem age_problem_solution : age_problem 10 := by
  sorry

#check age_problem_solution

end NUMINAMATH_CALUDE_age_problem_solution_l3144_314457


namespace NUMINAMATH_CALUDE_youngest_child_age_proof_l3144_314472

def youngest_child_age (n : ℕ) (interval : ℕ) (total_age : ℕ) : ℕ :=
  (total_age - (n - 1) * n * interval / 2) / n

theorem youngest_child_age_proof (n : ℕ) (interval : ℕ) (total_age : ℕ) 
  (h1 : n = 5)
  (h2 : interval = 3)
  (h3 : total_age = 50)
  (h4 : youngest_child_age n interval total_age * 2 = youngest_child_age n interval total_age + (n - 1) * interval) :
  youngest_child_age n interval total_age = 4 := by
sorry

#eval youngest_child_age 5 3 50

end NUMINAMATH_CALUDE_youngest_child_age_proof_l3144_314472


namespace NUMINAMATH_CALUDE_translation_proof_l3144_314459

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector between two points -/
def vector (p q : Point) : Point :=
  ⟨q.x - p.x, q.y - p.y⟩

/-- Translate a point by a vector -/
def translate (p : Point) (v : Point) : Point :=
  ⟨p.x + v.x, p.y + v.y⟩

theorem translation_proof (A C D : Point)
    (h1 : A = ⟨-1, 4⟩)
    (h2 : C = ⟨4, 7⟩)
    (h3 : D = ⟨-4, 1⟩)
    (h4 : ∃ B : Point, vector A B = vector C D) :
    ∃ B : Point, B = ⟨-9, -2⟩ ∧ vector A B = vector C D := by
  sorry

#check translation_proof

end NUMINAMATH_CALUDE_translation_proof_l3144_314459


namespace NUMINAMATH_CALUDE_emily_has_ten_employees_l3144_314468

/-- Calculates the number of employees Emily has based on salary information. -/
def calculate_employees (emily_original_salary : ℕ) (emily_new_salary : ℕ) 
                        (employee_original_salary : ℕ) (employee_new_salary : ℕ) : ℕ :=
  (emily_original_salary - emily_new_salary) / (employee_new_salary - employee_original_salary)

/-- Proves that Emily has 10 employees given the salary information. -/
theorem emily_has_ten_employees :
  calculate_employees 1000000 850000 20000 35000 = 10 := by
  sorry

end NUMINAMATH_CALUDE_emily_has_ten_employees_l3144_314468


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3144_314434

theorem line_circle_intersection (k : ℝ) :
  ∃ (x y : ℝ), y = k * (x - 1) ∧ x^2 + y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3144_314434


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l3144_314478

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (loss_margin : ℕ)
  (h_total : total_votes = 10000)
  (h_margin : loss_margin = 4000) :
  (total_votes - loss_margin) * 2 * 100 / total_votes = 30 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l3144_314478


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l3144_314447

theorem complex_modulus_sqrt_two (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) :
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l3144_314447


namespace NUMINAMATH_CALUDE_max_planes_15_points_l3144_314431

/-- The number of points in the space -/
def n : ℕ := 15

/-- A function to calculate the number of combinations of n things taken k at a time -/
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The maximum number of unique planes determined by n points in general position -/
def max_planes (n : ℕ) : ℕ := combination n 3

theorem max_planes_15_points :
  max_planes n = 455 :=
sorry

end NUMINAMATH_CALUDE_max_planes_15_points_l3144_314431


namespace NUMINAMATH_CALUDE_area_change_not_triple_l3144_314443

theorem area_change_not_triple :
  ∀ (s r : ℝ), s > 0 → r > 0 →
  (3 * s)^2 ≠ 3 * s^2 ∧ π * (3 * r)^2 ≠ 3 * (π * r^2) :=
by sorry

end NUMINAMATH_CALUDE_area_change_not_triple_l3144_314443


namespace NUMINAMATH_CALUDE_share_difference_l3144_314430

/-- Represents the share ratio for each person --/
structure ShareRatio :=
  (faruk : ℕ)
  (vasim : ℕ)
  (ranjith : ℕ)
  (kavita : ℕ)
  (neel : ℕ)

/-- Represents the distribution problem --/
structure DistributionProblem :=
  (ratio : ShareRatio)
  (vasim_share : ℕ)
  (x : ℕ+)
  (y : ℕ+)

def total_ratio (r : ShareRatio) : ℕ :=
  r.faruk + r.vasim + r.ranjith + r.kavita + r.neel

def total_amount (p : DistributionProblem) : ℕ :=
  p.vasim_share * (p.x + p.y)

theorem share_difference (p : DistributionProblem) 
  (h1 : p.ratio = ⟨3, 5, 7, 9, 11⟩)
  (h2 : p.vasim_share = 1500)
  (h3 : total_amount p = total_ratio p.ratio * (p.vasim_share / p.ratio.vasim)) :
  p.ratio.ranjith * (p.vasim_share / p.ratio.vasim) - 
  p.ratio.faruk * (p.vasim_share / p.ratio.vasim) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_l3144_314430


namespace NUMINAMATH_CALUDE_translation_of_line_segment_l3144_314454

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (t : Translation) (p : Point) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_of_line_segment :
  let A : Point := { x := 1, y := 1 }
  let B : Point := { x := -2, y := 0 }
  let A' : Point := { x := 4, y := 0 }
  let t : Translation := { dx := A'.x - A.x, dy := A'.y - A.y }
  let B' : Point := applyTranslation t B
  B'.x = 1 ∧ B'.y = -1 := by sorry

end NUMINAMATH_CALUDE_translation_of_line_segment_l3144_314454


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3144_314405

/-- Proves that the rate of interest is 8% given the specified loan conditions -/
theorem interest_rate_calculation (principal : ℝ) (interest : ℝ) 
  (h1 : principal = 1100)
  (h2 : interest = 704)
  (h3 : ∀ r t, interest = principal * r * t / 100 → r = t) :
  ∃ r : ℝ, r = 8 ∧ interest = principal * r * r / 100 := by
  sorry


end NUMINAMATH_CALUDE_interest_rate_calculation_l3144_314405


namespace NUMINAMATH_CALUDE_tree_planting_cost_l3144_314474

/-- The cost of planting trees to achieve a specific temperature drop -/
theorem tree_planting_cost (initial_temp final_temp temp_drop_per_tree cost_per_tree : ℝ) : 
  initial_temp - final_temp = 1.8 →
  temp_drop_per_tree = 0.1 →
  cost_per_tree = 6 →
  ((initial_temp - final_temp) / temp_drop_per_tree) * cost_per_tree = 108 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_cost_l3144_314474


namespace NUMINAMATH_CALUDE_a_range_theorem_l3144_314411

-- Define the line equation
def line_equation (x y a : ℝ) : ℝ := x + y - a

-- Define the condition for points being on opposite sides of the line
def opposite_sides (a : ℝ) : Prop :=
  (line_equation 1 1 a) * (line_equation 2 (-1) a) < 0

-- Theorem statement
theorem a_range_theorem :
  ∀ a : ℝ, opposite_sides a ↔ a ∈ Set.Ioo 1 2 :=
by sorry

end NUMINAMATH_CALUDE_a_range_theorem_l3144_314411


namespace NUMINAMATH_CALUDE_newspaper_spending_difference_l3144_314407

/-- Calculates the difference in yearly newspaper spending between Juanita and Grant -/
theorem newspaper_spending_difference : 
  let grant_yearly_spending : ℚ := 200
  let juanita_daily_spending : ℚ := 0.5
  let juanita_sunday_spending : ℚ := 2
  let days_per_week : ℕ := 7
  let weekdays : ℕ := 6
  let weeks_per_year : ℕ := 52
  
  let juanita_weekly_spending : ℚ := juanita_daily_spending * weekdays + juanita_sunday_spending
  let juanita_yearly_spending : ℚ := juanita_weekly_spending * weeks_per_year
  
  juanita_yearly_spending - grant_yearly_spending = 60
  := by sorry

end NUMINAMATH_CALUDE_newspaper_spending_difference_l3144_314407


namespace NUMINAMATH_CALUDE_round_trip_ticket_percentage_l3144_314436

theorem round_trip_ticket_percentage (total_passengers : ℝ) :
  let round_trip_with_car := 0.20 * total_passengers
  let round_trip_without_car_ratio := 0.40
  let round_trip_passengers := round_trip_with_car / (1 - round_trip_without_car_ratio)
  round_trip_passengers / total_passengers = 1/3 := by
sorry

end NUMINAMATH_CALUDE_round_trip_ticket_percentage_l3144_314436


namespace NUMINAMATH_CALUDE_tv_watching_days_l3144_314408

/-- The number of days per week children are allowed to watch TV -/
def days_per_week : ℕ := sorry

/-- The number of minutes children watch TV each day they are allowed -/
def minutes_per_day : ℕ := 45

/-- The total number of hours children watch TV in 2 weeks -/
def total_hours_in_two_weeks : ℕ := 6

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem tv_watching_days : 
  days_per_week * minutes_per_day * 2 = total_hours_in_two_weeks * minutes_per_hour :=
sorry

end NUMINAMATH_CALUDE_tv_watching_days_l3144_314408


namespace NUMINAMATH_CALUDE_factorization_proof_l3144_314463

theorem factorization_proof (m n : ℝ) : 12 * m^2 * n - 12 * m * n + 3 * n = 3 * n * (2 * m - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3144_314463


namespace NUMINAMATH_CALUDE_same_solution_equations_l3144_314422

theorem same_solution_equations (c : ℝ) : 
  (∃ x : ℝ, 3 * x + 6 = 0 ∧ c * x - 15 = -3) → c = -6 := by
sorry

end NUMINAMATH_CALUDE_same_solution_equations_l3144_314422


namespace NUMINAMATH_CALUDE_triangle_existence_l3144_314483

/-- A triangle with sides x, 10 + x, and 24 can exist if and only if x is a positive integer and x ≥ 34. -/
theorem triangle_existence (x : ℕ) : 
  (∃ (a b c : ℝ), a = x ∧ b = x + 10 ∧ c = 24 ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a) ↔ 
  x ≥ 34 := by
  sorry

#check triangle_existence

end NUMINAMATH_CALUDE_triangle_existence_l3144_314483


namespace NUMINAMATH_CALUDE_f_properties_f_50_l3144_314419

/-- A cubic polynomial function satisfying specific conditions -/
def f (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the properties of the function f -/
theorem f_properties :
  f 0 = 1 ∧
  f 1 = 5 ∧
  f 2 = 13 ∧
  f 3 = 25 :=
sorry

/-- Theorem proving the value of f(50) -/
theorem f_50 : f 50 = 62676 :=
sorry

end NUMINAMATH_CALUDE_f_properties_f_50_l3144_314419


namespace NUMINAMATH_CALUDE_parabola_directrix_l3144_314496

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := y = (x^2 - 4*x + 4) / 8

/-- The directrix equation -/
def directrix_eq (y : ℝ) : Prop := y = -1/4

/-- Theorem: The directrix of the given parabola is y = -1/4 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_eq x y → ∃ y_d : ℝ, directrix_eq y_d ∧ 
  (∀ x' y' : ℝ, parabola_eq x' y' → 
    (x' - x)^2 + (y' - y)^2 = (y' - y_d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3144_314496


namespace NUMINAMATH_CALUDE_circle_tape_length_16_strips_l3144_314461

/-- The total length of a circle-shaped tape made from overlapping strips -/
def circle_tape_length (num_strips : ℕ) (strip_length : ℝ) (overlap_length : ℝ) : ℝ :=
  num_strips * strip_length - num_strips * overlap_length

/-- Theorem: The length of a circle-shaped tape made from 16 strips of 10.4 cm
    with 3.5 cm overlaps is 110.4 cm -/
theorem circle_tape_length_16_strips :
  circle_tape_length 16 10.4 3.5 = 110.4 := by
  sorry

#eval circle_tape_length 16 10.4 3.5

end NUMINAMATH_CALUDE_circle_tape_length_16_strips_l3144_314461


namespace NUMINAMATH_CALUDE_base_8_of_2023_l3144_314445

/-- Converts a base-10 number to its base-8 representation -/
def toBase8 (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The base-8 representation of 2023 (base 10) is 3747 -/
theorem base_8_of_2023 : toBase8 2023 = 3747 := by
  sorry

end NUMINAMATH_CALUDE_base_8_of_2023_l3144_314445


namespace NUMINAMATH_CALUDE_smallest_harmonic_sum_exceeding_10_l3144_314467

def harmonic_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => 1 / (i + 1 : ℚ))

theorem smallest_harmonic_sum_exceeding_10 :
  (∀ k < 12367, harmonic_sum k ≤ 10) ∧ harmonic_sum 12367 > 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_harmonic_sum_exceeding_10_l3144_314467


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3144_314475

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ + 2*m - 4 = 0 ∧ x₂^2 - x₂ + 2*m - 4 = 0) →
  (m ≤ 17/8 ∧
   (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ + 2*m - 4 = 0 ∧ x₂^2 - x₂ + 2*m - 4 = 0 →
    (x₁ - 3) * (x₂ - 3) = m^2 - 1 → m = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3144_314475


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l3144_314432

theorem quadratic_always_positive_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l3144_314432


namespace NUMINAMATH_CALUDE_video_game_shelves_l3144_314446

/-- Calculates the minimum number of shelves needed to display video games -/
def minimum_shelves_needed (total_games : ℕ) (action_games : ℕ) (adventure_games : ℕ) (simulation_games : ℕ) (shelf_capacity : ℕ) (special_display_per_genre : ℕ) : ℕ :=
  let remaining_action := action_games - special_display_per_genre
  let remaining_adventure := adventure_games - special_display_per_genre
  let remaining_simulation := simulation_games - special_display_per_genre
  let action_shelves := (remaining_action + shelf_capacity - 1) / shelf_capacity
  let adventure_shelves := (remaining_adventure + shelf_capacity - 1) / shelf_capacity
  let simulation_shelves := (remaining_simulation + shelf_capacity - 1) / shelf_capacity
  action_shelves + adventure_shelves + simulation_shelves + 1

theorem video_game_shelves :
  minimum_shelves_needed 163 73 51 39 84 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_video_game_shelves_l3144_314446


namespace NUMINAMATH_CALUDE_debate_team_arrangements_l3144_314435

/-- Represents the debate team composition -/
structure DebateTeam :=
  (male_count : Nat)
  (female_count : Nat)

/-- The number of arrangements where no two male members are adjacent -/
def non_adjacent_male_arrangements (team : DebateTeam) : Nat :=
  sorry

/-- The number of ways to divide the team into four groups of two and assign them to four classes -/
def seminar_groupings (team : DebateTeam) : Nat :=
  sorry

/-- The number of ways to select 4 members (with at least one male) and assign them to four speaker roles -/
def speaker_selections (team : DebateTeam) : Nat :=
  sorry

theorem debate_team_arrangements (team : DebateTeam) 
  (h1 : team.male_count = 3) 
  (h2 : team.female_count = 5) : 
  non_adjacent_male_arrangements team = 14400 ∧ 
  seminar_groupings team = 2520 ∧ 
  speaker_selections team = 1560 :=
sorry

end NUMINAMATH_CALUDE_debate_team_arrangements_l3144_314435


namespace NUMINAMATH_CALUDE_todd_ate_five_cupcakes_l3144_314497

def cupcake_problem (initial_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) : ℕ :=
  initial_cupcakes - packages * cupcakes_per_package

theorem todd_ate_five_cupcakes :
  cupcake_problem 50 9 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_todd_ate_five_cupcakes_l3144_314497


namespace NUMINAMATH_CALUDE_janine_read_five_books_last_month_l3144_314451

/-- The number of books Janine read last month -/
def last_month_books : ℕ := sorry

/-- The number of books Janine read this month -/
def this_month_books : ℕ := 2 * last_month_books

/-- The number of pages in each book -/
def pages_per_book : ℕ := 10

/-- The total number of pages Janine read in two months -/
def total_pages : ℕ := 150

theorem janine_read_five_books_last_month :
  last_month_books = 5 :=
by sorry

end NUMINAMATH_CALUDE_janine_read_five_books_last_month_l3144_314451


namespace NUMINAMATH_CALUDE_job_completion_proof_l3144_314455

/-- Given workers P, Q, and R who can complete a job in 3, 9, and 6 hours respectively,
    prove that the combined work of P (1 hour), Q (2 hours), and R (3 hours) completes the job. -/
theorem job_completion_proof (p q r : ℝ) (hp : p = 1/3) (hq : q = 1/9) (hr : r = 1/6) :
  p * 1 + q * 2 + r * 3 ≥ 1 := by
  sorry

#check job_completion_proof

end NUMINAMATH_CALUDE_job_completion_proof_l3144_314455


namespace NUMINAMATH_CALUDE_soda_problem_l3144_314442

theorem soda_problem (S : ℝ) : 
  (S / 2 + 2000 = S - (S / 2 - 2000)) → 
  ((S / 2 - 2000) / 2 + 2000 = S / 2 - 2000) → 
  S = 12000 := by
  sorry

end NUMINAMATH_CALUDE_soda_problem_l3144_314442


namespace NUMINAMATH_CALUDE_small_birdhouse_price_is_seven_l3144_314428

/-- Represents the price of birdhouses and sales information. -/
structure BirdhouseSales where
  large_price : ℕ
  medium_price : ℕ
  large_sold : ℕ
  medium_sold : ℕ
  small_sold : ℕ
  total_sales : ℕ

/-- Calculates the price of small birdhouses given the sales information. -/
def small_birdhouse_price (sales : BirdhouseSales) : ℕ :=
  (sales.total_sales - (sales.large_price * sales.large_sold + sales.medium_price * sales.medium_sold)) / sales.small_sold

/-- Theorem stating that the price of small birdhouses is $7 given the specific sales information. -/
theorem small_birdhouse_price_is_seven :
  let sales := BirdhouseSales.mk 22 16 2 2 3 97
  small_birdhouse_price sales = 7 := by
  sorry

#eval small_birdhouse_price (BirdhouseSales.mk 22 16 2 2 3 97)

end NUMINAMATH_CALUDE_small_birdhouse_price_is_seven_l3144_314428


namespace NUMINAMATH_CALUDE_fraction_value_l3144_314444

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l3144_314444


namespace NUMINAMATH_CALUDE_polygon_sides_diagonals_l3144_314498

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon has 11 sides if the number of its diagonals is 33 more than the number of its sides -/
theorem polygon_sides_diagonals : 
  ∃ (n : ℕ), n > 3 ∧ num_diagonals n = n + 33 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_diagonals_l3144_314498


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l3144_314470

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) (h1 : speed1 = 140) (h2 : speed2 = 40) :
  (speed1 + speed2) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l3144_314470


namespace NUMINAMATH_CALUDE_decimal_expansion_of_three_sevenths_l3144_314469

/-- The length of the smallest repeating block in the decimal expansion of 3/7 -/
def repeatingBlockLength : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 3/7

theorem decimal_expansion_of_three_sevenths :
  ∃ (d : ℕ → ℕ) (n : ℕ),
    (∀ k, d k < 10) ∧
    (∀ k, d (k + n) = d k) ∧
    (∀ m, m < n → ∃ k, d (k + m) ≠ d k) ∧
    fraction = ∑' k, (d k : ℚ) / 10^(k + 1) ∧
    n = repeatingBlockLength :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_of_three_sevenths_l3144_314469


namespace NUMINAMATH_CALUDE_triangle_area_l3144_314423

/-- Given a triangle ABC with angle A = 60°, side b = 4, and side a = 2√3, 
    prove that its area is 2√3 square units. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  A = π / 3 →  -- 60° in radians
  b = 4 → 
  a = 2 * Real.sqrt 3 → 
  (1 / 2) * a * b * Real.sin C = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3144_314423


namespace NUMINAMATH_CALUDE_cosine_values_l3144_314400

def terminalPoint : ℝ × ℝ := (-3, 4)

theorem cosine_values (α : ℝ) (h : terminalPoint ∈ {p : ℝ × ℝ | ∃ t, p.1 = t * Real.cos α ∧ p.2 = t * Real.sin α}) :
  Real.cos α = -3/5 ∧ Real.cos (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_cosine_values_l3144_314400


namespace NUMINAMATH_CALUDE_potato_problem_result_l3144_314416

/-- Represents the potato problem --/
structure PotatoProblem where
  totalPotatoes : Nat
  potatoesForWedges : Nat
  wedgesPerPotato : Nat
  chipsPerPotato : Nat

/-- Calculates the difference between potato chips and wedges --/
def chipWedgeDifference (p : PotatoProblem) : Nat :=
  let remainingPotatoes := p.totalPotatoes - p.potatoesForWedges
  let potatoesForChips := remainingPotatoes / 2
  let totalChips := potatoesForChips * p.chipsPerPotato
  let totalWedges := p.potatoesForWedges * p.wedgesPerPotato
  totalChips - totalWedges

/-- Theorem stating the result of the potato problem --/
theorem potato_problem_result :
  let p : PotatoProblem := {
    totalPotatoes := 67,
    potatoesForWedges := 13,
    wedgesPerPotato := 8,
    chipsPerPotato := 20
  }
  chipWedgeDifference p = 436 := by
  sorry

end NUMINAMATH_CALUDE_potato_problem_result_l3144_314416


namespace NUMINAMATH_CALUDE_platform_length_l3144_314402

/-- Given a train and platform with specific properties, prove the platform length --/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 30)
  (h3 : pole_crossing_time = 18) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 200 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3144_314402


namespace NUMINAMATH_CALUDE_train_speed_problem_l3144_314499

theorem train_speed_problem (V₁ V₂ : ℝ) (h₁ : V₁ > 0) (h₂ : V₂ > 0) (h₃ : V₂ > V₁) : 
  (∃ t : ℝ, t > 0 ∧ t * (V₁ + V₂) = 2400) ∧
  (∃ t : ℝ, t > 0 ∧ 2 * V₂ * (t - 3) = 2400) ∧
  (∃ t : ℝ, t > 0 ∧ 2 * V₁ * (t + 5) = 2400) →
  V₁ = 60 ∧ V₂ = 100 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3144_314499


namespace NUMINAMATH_CALUDE_arithmetic_sequence_scalar_multiple_l3144_314492

theorem arithmetic_sequence_scalar_multiple
  (a : ℕ → ℝ) (d c : ℝ) (h_arith : ∀ n, a (n + 1) - a n = d) (h_c : c ≠ 0) :
  ∃ (b : ℕ → ℝ), (∀ n, b n = c * a n) ∧ (∀ n, b (n + 1) - b n = c * d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_scalar_multiple_l3144_314492


namespace NUMINAMATH_CALUDE_two_numbers_with_specific_means_l3144_314487

theorem two_numbers_with_specific_means (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x * y = 600^2) → 
  ((x + y) / 2 = (2 * x * y) / (x + y) + 49) →
  ({x, y} : Set ℝ) = {800, 450} := by
sorry

end NUMINAMATH_CALUDE_two_numbers_with_specific_means_l3144_314487


namespace NUMINAMATH_CALUDE_car_production_increase_l3144_314489

/-- Proves that adding 50 cars to the monthly production of 100 cars
    will result in an annual production of 1800 cars. -/
theorem car_production_increase (current_monthly : ℕ) (target_yearly : ℕ) (increase : ℕ) :
  current_monthly = 100 →
  target_yearly = 1800 →
  increase = 50 →
  (current_monthly + increase) * 12 = target_yearly := by
  sorry

#check car_production_increase

end NUMINAMATH_CALUDE_car_production_increase_l3144_314489


namespace NUMINAMATH_CALUDE_bookshelf_selection_l3144_314462

theorem bookshelf_selection (math_books : ℕ) (chinese_books : ℕ) (english_books : ℕ) 
  (h1 : math_books = 3) (h2 : chinese_books = 5) (h3 : english_books = 8) :
  math_books + chinese_books + english_books = 16 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_selection_l3144_314462


namespace NUMINAMATH_CALUDE_gcd_9157_2695_l3144_314486

theorem gcd_9157_2695 : Nat.gcd 9157 2695 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9157_2695_l3144_314486


namespace NUMINAMATH_CALUDE_original_ribbon_length_is_correct_l3144_314450

/-- The length of ribbon tape used for one gift in meters -/
def ribbon_per_gift : ℝ := 0.84

/-- The number of gifts prepared -/
def num_gifts : ℕ := 10

/-- The length of leftover ribbon tape in meters -/
def leftover_ribbon : ℝ := 0.5

/-- The total length of the original ribbon tape in meters -/
def original_ribbon_length : ℝ := ribbon_per_gift * num_gifts + leftover_ribbon

theorem original_ribbon_length_is_correct :
  original_ribbon_length = 8.9 := by sorry

end NUMINAMATH_CALUDE_original_ribbon_length_is_correct_l3144_314450


namespace NUMINAMATH_CALUDE_x0_value_l3144_314456

noncomputable def f (x : ℝ) : ℝ := x * (2014 + Real.log x)

theorem x0_value (x₀ : ℝ) (h : (deriv f) x₀ = 2015) : x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l3144_314456


namespace NUMINAMATH_CALUDE_magic_forest_coin_difference_l3144_314464

theorem magic_forest_coin_difference :
  ∀ (x y : ℕ),
  let trees_with_no_coins := 2 * x
  let trees_with_one_coin := y
  let trees_with_two_coins := 3
  let trees_with_three_coins := x
  let trees_with_four_coins := 4
  let total_coins := y + 3 * x + 22
  let total_trees := 3 * x + y + 7
  total_coins - total_trees = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_magic_forest_coin_difference_l3144_314464


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3144_314453

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {1, 3, 6}

theorem intersection_with_complement : A ∩ (U \ B) = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3144_314453


namespace NUMINAMATH_CALUDE_dubblefud_red_ball_value_l3144_314404

/-- The value of a red ball in the game of Dubblefud -/
def red_ball_value : ℕ := sorry

/-- The value of a blue ball in the game of Dubblefud -/
def blue_ball_value : ℕ := 4

/-- The value of a green ball in the game of Dubblefud -/
def green_ball_value : ℕ := 5

/-- The number of red balls in the selection -/
def num_red_balls : ℕ := 4

/-- The number of blue balls in the selection -/
def num_blue_balls : ℕ := sorry

/-- The number of green balls in the selection -/
def num_green_balls : ℕ := sorry

theorem dubblefud_red_ball_value :
  (red_ball_value ^ num_red_balls) * 
  (blue_ball_value ^ num_blue_balls) * 
  (green_ball_value ^ num_green_balls) = 16000 ∧
  num_blue_balls = num_green_balls →
  red_ball_value = 1 :=
sorry

end NUMINAMATH_CALUDE_dubblefud_red_ball_value_l3144_314404


namespace NUMINAMATH_CALUDE_min_value_problem_l3144_314458

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (8^a * 2^b)) :
  1/a + 2/b ≥ 5 + 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l3144_314458


namespace NUMINAMATH_CALUDE_english_article_usage_l3144_314406

-- Define a type for articles
inductive Article
  | None
  | A
  | The

-- Define a function to represent the correctness of article usage
def correct_article_usage (first second : Article) : Prop :=
  first = Article.None ∧ second = Article.A

-- Define the theorem
theorem english_article_usage :
  correct_article_usage Article.None Article.A :=
sorry

end NUMINAMATH_CALUDE_english_article_usage_l3144_314406


namespace NUMINAMATH_CALUDE_relay_team_arrangements_l3144_314437

theorem relay_team_arrangements (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_arrangements_l3144_314437


namespace NUMINAMATH_CALUDE_max_value_x_plus_y_squared_l3144_314484

/-- Given real numbers x and y satisfying 3(x^3 + y^3) = x + y^2,
    the maximum value of x + y^2 is 1/3. -/
theorem max_value_x_plus_y_squared (x y : ℝ) 
  (h : 3 * (x^3 + y^3) = x + y^2) : 
  ∃ (M : ℝ), M = 1/3 ∧ ∀ (a b : ℝ), 3 * (a^3 + b^3) = a + b^2 → a + b^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_y_squared_l3144_314484


namespace NUMINAMATH_CALUDE_real_part_of_z_l3144_314449

def i : ℂ := Complex.I

theorem real_part_of_z (z : ℂ) (h : (1 + 2*i)*z = 3 + 4*i) : 
  Complex.re z = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3144_314449


namespace NUMINAMATH_CALUDE_potato_peeling_theorem_l3144_314412

def potato_peeling_problem (julie_rate ted_rate combined_time : ℝ) : Prop :=
  let julie_part := combined_time * julie_rate
  let ted_part := combined_time * ted_rate
  let remaining_part := 1 - (julie_part + ted_part)
  remaining_part / julie_rate = 1

theorem potato_peeling_theorem :
  potato_peeling_problem (1/10) (1/8) 4 := by
  sorry

end NUMINAMATH_CALUDE_potato_peeling_theorem_l3144_314412


namespace NUMINAMATH_CALUDE_daria_pizza_multiple_l3144_314490

/-- The multiple of pizza Daria consumes compared to Don -/
def daria_multiple (don_pizzas : ℕ) (total_pizzas : ℕ) : ℚ :=
  (total_pizzas : ℚ) / (don_pizzas : ℚ) - 1

theorem daria_pizza_multiple :
  daria_multiple 80 280 = 2.5 := by sorry

end NUMINAMATH_CALUDE_daria_pizza_multiple_l3144_314490


namespace NUMINAMATH_CALUDE_orange_removal_problem_l3144_314410

/-- Represents the number of oranges Mary must put back to achieve the desired average price -/
def oranges_to_remove : ℕ := sorry

/-- The price of an apple in cents -/
def apple_price : ℕ := 50

/-- The price of an orange in cents -/
def orange_price : ℕ := 60

/-- The total number of fruits initially selected -/
def total_fruits : ℕ := 10

/-- The initial average price of the fruits in cents -/
def initial_avg_price : ℕ := 56

/-- The desired average price after removing oranges in cents -/
def desired_avg_price : ℕ := 52

theorem orange_removal_problem :
  ∃ (apples oranges : ℕ),
    apples + oranges = total_fruits ∧
    (apple_price * apples + orange_price * oranges) / total_fruits = initial_avg_price ∧
    (apple_price * apples + orange_price * (oranges - oranges_to_remove)) / (total_fruits - oranges_to_remove) = desired_avg_price ∧
    oranges_to_remove = 5 := by sorry

end NUMINAMATH_CALUDE_orange_removal_problem_l3144_314410


namespace NUMINAMATH_CALUDE_cookies_in_box_l3144_314439

theorem cookies_in_box (cookies_per_bag : ℕ) (calories_per_cookie : ℕ) (total_calories : ℕ) :
  cookies_per_bag = 20 →
  calories_per_cookie = 20 →
  total_calories = 1600 →
  total_calories / (cookies_per_bag * calories_per_cookie) = 4 :=
by sorry

end NUMINAMATH_CALUDE_cookies_in_box_l3144_314439


namespace NUMINAMATH_CALUDE_find_number_l3144_314427

theorem find_number : ∃ x : ℝ, (38 + 2 * x = 124) ∧ (x = 43) := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3144_314427


namespace NUMINAMATH_CALUDE_blueberry_count_l3144_314480

theorem blueberry_count (total : ℕ) (raspberries : ℕ) (blackberries : ℕ) 
  (h1 : total = 42)
  (h2 : raspberries = total / 2)
  (h3 : blackberries = total / 3) :
  total - raspberries - blackberries = 7 := by
sorry

end NUMINAMATH_CALUDE_blueberry_count_l3144_314480


namespace NUMINAMATH_CALUDE_logistics_problem_l3144_314465

/-- Represents the problem of transporting goods using two types of trucks -/
theorem logistics_problem (total_goods : ℕ) (type_a_capacity : ℕ) (type_b_capacity : ℕ) (num_type_a : ℕ) :
  total_goods = 300 →
  type_a_capacity = 20 →
  type_b_capacity = 15 →
  num_type_a = 7 →
  ∃ (num_type_b : ℕ),
    num_type_b ≥ 11 ∧
    num_type_a * type_a_capacity + num_type_b * type_b_capacity ≥ total_goods ∧
    ∀ (m : ℕ), m < num_type_b →
      num_type_a * type_a_capacity + m * type_b_capacity < total_goods :=
by
  sorry


end NUMINAMATH_CALUDE_logistics_problem_l3144_314465


namespace NUMINAMATH_CALUDE_markus_family_ages_l3144_314421

theorem markus_family_ages (grandson_age : ℕ) : 
  grandson_age > 0 →
  let son_age := 2 * grandson_age
  let markus_age := 2 * son_age
  grandson_age + son_age + markus_age = 140 →
  grandson_age = 20 := by
sorry

end NUMINAMATH_CALUDE_markus_family_ages_l3144_314421


namespace NUMINAMATH_CALUDE_some_T_divisible_by_3_l3144_314448

def T : Set ℤ := {x | ∃ n : ℤ, x = (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2}

theorem some_T_divisible_by_3 : ∃ x ∈ T, 3 ∣ x := by
  sorry

end NUMINAMATH_CALUDE_some_T_divisible_by_3_l3144_314448


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l3144_314479

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 43 :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l3144_314479


namespace NUMINAMATH_CALUDE_two_true_propositions_l3144_314477

theorem two_true_propositions :
  let P1 := ∀ a b c : ℝ, a > b → a*c^2 > b*c^2
  let P2 := ∀ a b c : ℝ, a*c^2 > b*c^2 → a > b
  let P3 := ∀ a b c : ℝ, a ≤ b → a*c^2 ≤ b*c^2
  let P4 := ∀ a b c : ℝ, a*c^2 ≤ b*c^2 → a ≤ b
  (¬P1 ∧ P2 ∧ P3 ∧ ¬P4) ∨
  (¬P1 ∧ P2 ∧ ¬P3 ∧ P4) ∨
  (P1 ∧ ¬P2 ∧ P3 ∧ ¬P4) ∨
  (P1 ∧ ¬P2 ∧ ¬P3 ∧ P4) :=
by
  sorry

end NUMINAMATH_CALUDE_two_true_propositions_l3144_314477


namespace NUMINAMATH_CALUDE_function_inequality_implies_range_l3144_314491

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem function_inequality_implies_range (f : ℝ → ℝ) (a : ℝ) :
  decreasing_function f →
  (∀ x, x > 0 → f x ≠ 0) →
  f (2 * a^2 + a + 1) < f (3 * a^2 - 4 * a + 1) →
  (0 < a ∧ a < 1/3) ∨ (1 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_range_l3144_314491


namespace NUMINAMATH_CALUDE_factorial_30_prime_factors_l3144_314495

theorem factorial_30_prime_factors : 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_30_prime_factors_l3144_314495


namespace NUMINAMATH_CALUDE_angle_300_shares_terminal_side_with_neg_60_l3144_314482

-- Define the concept of angles sharing the same terminal side
def shares_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = k * 360 - α

-- Theorem statement
theorem angle_300_shares_terminal_side_with_neg_60 :
  shares_terminal_side (-60) 300 := by
  sorry

end NUMINAMATH_CALUDE_angle_300_shares_terminal_side_with_neg_60_l3144_314482


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l3144_314425

theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45) 
  (h2 : (b + c) / 2 = 50) : 
  c - a = 10 := by
sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l3144_314425


namespace NUMINAMATH_CALUDE_victors_friend_bought_two_decks_l3144_314452

/-- The number of decks Victor's friend bought given the conditions of the problem -/
def victors_friend_decks (deck_cost : ℕ) (victors_decks : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent - deck_cost * victors_decks) / deck_cost

/-- Theorem stating that Victor's friend bought 2 decks under the given conditions -/
theorem victors_friend_bought_two_decks :
  victors_friend_decks 8 6 64 = 2 := by
  sorry

end NUMINAMATH_CALUDE_victors_friend_bought_two_decks_l3144_314452


namespace NUMINAMATH_CALUDE_boat_round_trip_time_l3144_314440

theorem boat_round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 9)
  (h2 : stream_speed = 6)
  (h3 : distance = 170)
  : (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = 68 := by
  sorry

end NUMINAMATH_CALUDE_boat_round_trip_time_l3144_314440


namespace NUMINAMATH_CALUDE_f_zeros_l3144_314476

noncomputable def f (x : ℝ) : ℝ := (1/3) * x - Real.log x

theorem f_zeros (h : ∀ x, x > 0 → f x = (1/3) * x - Real.log x) :
  (∀ x, 1/Real.exp 1 < x ∧ x < 1 → f x ≠ 0) ∧
  (∃ x, 1 < x ∧ x < Real.exp 1 ∧ f x = 0) :=
sorry

end NUMINAMATH_CALUDE_f_zeros_l3144_314476


namespace NUMINAMATH_CALUDE_sixth_root_of_unity_product_l3144_314460

theorem sixth_root_of_unity_product (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_unity_product_l3144_314460


namespace NUMINAMATH_CALUDE_kannon_fruit_consumption_l3144_314409

/-- Represents Kannon's fruit consumption over two days -/
structure FruitConsumption where
  apples_last_night : ℕ
  bananas_last_night : ℕ
  oranges_last_night : ℕ
  apples_increase : ℕ
  bananas_multiplier : ℕ

/-- Calculates the total number of fruits eaten over two days -/
def total_fruits (fc : FruitConsumption) : ℕ :=
  let apples_today := fc.apples_last_night + fc.apples_increase
  let bananas_today := fc.bananas_last_night * fc.bananas_multiplier
  let oranges_today := 2 * apples_today
  (fc.apples_last_night + apples_today) +
  (fc.bananas_last_night + bananas_today) +
  (fc.oranges_last_night + oranges_today)

/-- Theorem stating that Kannon's total fruit consumption is 39 -/
theorem kannon_fruit_consumption :
  ∃ (fc : FruitConsumption),
    fc.apples_last_night = 3 ∧
    fc.bananas_last_night = 1 ∧
    fc.oranges_last_night = 4 ∧
    fc.apples_increase = 4 ∧
    fc.bananas_multiplier = 10 ∧
    total_fruits fc = 39 := by
  sorry

end NUMINAMATH_CALUDE_kannon_fruit_consumption_l3144_314409


namespace NUMINAMATH_CALUDE_not_coplanar_implies_no_intersection_l3144_314493

-- Define a point in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define a line in 3D space as two points
def Line3D := Point3D × Point3D

-- Define a function to check if four points are coplanar
def are_coplanar (E F G H : Point3D) : Prop := sorry

-- Define a function to check if two lines intersect
def lines_intersect (l1 l2 : Line3D) : Prop := sorry

theorem not_coplanar_implies_no_intersection 
  (E F G H : Point3D) : 
  ¬(are_coplanar E F G H) → ¬(lines_intersect (E, F) (G, H)) := by
  sorry

end NUMINAMATH_CALUDE_not_coplanar_implies_no_intersection_l3144_314493


namespace NUMINAMATH_CALUDE_coefficient_of_x_l3144_314433

/-- The coefficient of x in the expression 3(x - 4) + 4(7 - 2x^2 + 5x) - 8(2x - 1) is 7 -/
theorem coefficient_of_x (x : ℝ) : 
  let expr := 3*(x - 4) + 4*(7 - 2*x^2 + 5*x) - 8*(2*x - 1)
  ∃ (a b c : ℝ), expr = a*x^2 + 7*x + c :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l3144_314433


namespace NUMINAMATH_CALUDE_point_symmetry_l3144_314420

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry about x-axis
def symmetricAboutXAxis (p1 p2 : Point2D) : Prop :=
  p1.x = p2.x ∧ p1.y = -p2.y

-- Define symmetry about y-axis
def symmetricAboutYAxis (p1 p2 : Point2D) : Prop :=
  p1.x = -p2.x ∧ p1.y = p2.y

theorem point_symmetry (M P N : Point2D) :
  symmetricAboutXAxis M P →
  symmetricAboutYAxis N M →
  N = Point2D.mk 1 2 →
  P = Point2D.mk (-1) (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_l3144_314420


namespace NUMINAMATH_CALUDE_small_circle_radius_l3144_314403

/-- Given a configuration of circles where:
    - There is a large circle with radius 10 meters
    - Six congruent smaller circles are arranged around it
    - Each smaller circle touches two others and the larger circle
    This theorem proves that the radius of each smaller circle is 5√3 meters -/
theorem small_circle_radius (R : ℝ) (r : ℝ) : 
  R = 10 → -- The radius of the larger circle is 10 meters
  R = (2 * r) / Real.sqrt 3 → -- Relationship between radii based on hexagon geometry
  r = 5 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l3144_314403


namespace NUMINAMATH_CALUDE_weight_gain_difference_l3144_314473

/-- The weight gain problem at the family reunion -/
theorem weight_gain_difference (orlando_gain jose_gain fernando_gain : ℝ) : 
  orlando_gain = 5 →
  jose_gain > 2 * orlando_gain →
  fernando_gain = jose_gain / 2 - 3 →
  orlando_gain + jose_gain + fernando_gain = 20 →
  ∃ ε > 0, |jose_gain - 2 * orlando_gain - 3.67| < ε :=
by sorry

end NUMINAMATH_CALUDE_weight_gain_difference_l3144_314473


namespace NUMINAMATH_CALUDE_bellas_score_l3144_314429

theorem bellas_score (total_students : ℕ) (avg_without : ℚ) (avg_with : ℚ) (bella_score : ℚ) :
  total_students = 18 →
  avg_without = 75 →
  avg_with = 76 →
  bella_score = ((total_students : ℚ) * avg_with - (total_students - 1 : ℚ) * avg_without) →
  bella_score = 93 :=
by sorry

end NUMINAMATH_CALUDE_bellas_score_l3144_314429


namespace NUMINAMATH_CALUDE_exists_circle_with_n_points_l3144_314441

/-- A function that counts the number of lattice points strictly inside a circle -/
def count_lattice_points (center : ℝ × ℝ) (radius : ℝ) : ℕ :=
  sorry

/-- Theorem stating that for any non-negative integer, there exists a circle containing exactly that many lattice points -/
theorem exists_circle_with_n_points (n : ℕ) :
  ∃ (center : ℝ × ℝ) (radius : ℝ), count_lattice_points center radius = n :=
sorry

end NUMINAMATH_CALUDE_exists_circle_with_n_points_l3144_314441


namespace NUMINAMATH_CALUDE_seven_fifth_sum_minus_two_fifth_l3144_314414

theorem seven_fifth_sum_minus_two_fifth (n : ℕ) : 
  (7^5 : ℕ) + (7^5 : ℕ) + (7^5 : ℕ) + (7^5 : ℕ) + (7^5 : ℕ) + (7^5 : ℕ) - (2^5 : ℕ) = 6 * (7^5 : ℕ) - 32 := by
sorry

end NUMINAMATH_CALUDE_seven_fifth_sum_minus_two_fifth_l3144_314414

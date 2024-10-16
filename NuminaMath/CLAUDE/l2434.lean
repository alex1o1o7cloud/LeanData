import Mathlib

namespace NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l2434_243416

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * Real.log x + 11

theorem f_monotonicity_and_inequality :
  (∀ x ∈ Set.Ioo (0 : ℝ) 1, StrictMonoOn f (Set.Ioo 0 1)) ∧
  (∀ x ∈ Set.Ioi 1, StrictMonoOn f (Set.Ioi 1)) ∧
  (∀ x > 0, f x > -x^3 + 3*x^2 + (3 - x) * Real.exp x) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l2434_243416


namespace NUMINAMATH_CALUDE_completed_square_form_l2434_243456

theorem completed_square_form (x : ℝ) :
  x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_completed_square_form_l2434_243456


namespace NUMINAMATH_CALUDE_dropped_players_not_necessarily_played_each_other_l2434_243448

/-- Represents a round-robin chess tournament --/
structure ChessTournament where
  n : ℕ  -- Total number of participants
  games_played : ℕ  -- Total number of games played
  dropped_players : ℕ  -- Number of players who dropped out

/-- Calculates the total number of games in a round-robin tournament --/
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that in a specific tournament scenario, dropped players didn't necessarily play each other --/
theorem dropped_players_not_necessarily_played_each_other 
  (t : ChessTournament) 
  (h1 : t.games_played = 23) 
  (h2 : t.dropped_players = 2) 
  (h3 : ∃ k : ℕ, t.n = k + t.dropped_players) 
  (h4 : ∃ m : ℕ, m * t.dropped_players = t.games_played - total_games (t.n - t.dropped_players)) :
  ¬ (∀ dropped_player_games : ℕ, dropped_player_games * t.dropped_players = t.games_played - total_games (t.n - t.dropped_players) → dropped_player_games = t.n - t.dropped_players - 1) :=
sorry

end NUMINAMATH_CALUDE_dropped_players_not_necessarily_played_each_other_l2434_243448


namespace NUMINAMATH_CALUDE_five_people_lineup_l2434_243449

/-- The number of ways to arrange n people in a line where k people cannot be first -/
def arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n - k) * n.factorial

/-- The problem statement -/
theorem five_people_lineup : arrangements 5 2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_five_people_lineup_l2434_243449


namespace NUMINAMATH_CALUDE_complement_of_A_l2434_243445

def U : Set Int := Set.univ

def A : Set Int := {x : Int | x^2 - x - 2 ≥ 0}

theorem complement_of_A : Set.compl A = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l2434_243445


namespace NUMINAMATH_CALUDE_crease_length_l2434_243462

/-- The length of a crease in a folded rectangular paper -/
theorem crease_length (θ : Real) : 
  let paper_width : Real := 8
  let touch_point_distance : Real := 2
  let crease_length := Real.sqrt (40 + 24 * Real.cos θ)
  ∀ (actual_length : Real), 
    (actual_length = crease_length) ∧ 
    (actual_length^2 = paper_width^2 + touch_point_distance^2 - 
      2 * paper_width * touch_point_distance * Real.cos θ) :=
by sorry

end NUMINAMATH_CALUDE_crease_length_l2434_243462


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2434_243492

theorem quadratic_factorization (a b c d : ℤ) : 
  (∀ x : ℝ, 6 * x^2 + x - 12 = (a * x + b) * (c * x + d)) →
  |a| + |b| + |c| + |d| = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2434_243492


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l2434_243406

/-- Given n, s, and r where s = 3^n + 2 and r = 4^s - 2s, 
    prove that when n = 3, r = 4^29 - 58 -/
theorem r_value_when_n_is_3 (n : ℕ) (s : ℕ) (r : ℕ) 
    (h1 : s = 3^n + 2) 
    (h2 : r = 4^s - 2*s) : 
  n = 3 → r = 4^29 - 58 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l2434_243406


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2434_243463

-- Define the given line
def given_line (x y : ℝ) : Prop := 3*x + y + 5 = 0

-- Define the point P
def point_P : ℝ × ℝ := (2, 1)

-- Define the perpendicular line l
def line_l (x y : ℝ) : Prop := x - 3*y + 1 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  (∀ x y : ℝ, given_line x y → (line_l x y → ¬given_line x y)) ∧
  line_l point_P.1 point_P.2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2434_243463


namespace NUMINAMATH_CALUDE_larger_integer_value_l2434_243447

theorem larger_integer_value (a b : ℕ+) 
  (h1 : (a : ℝ) / (b : ℝ) = 7 / 3) 
  (h2 : (a : ℕ) * b = 294) : 
  max a b = ⌈7 * Real.sqrt 14⌉ := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l2434_243447


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2434_243493

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 4) :
  (∀ x : ℝ, g x = x + 5) ∨ (∀ x : ℝ, g x = -x - 3) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2434_243493


namespace NUMINAMATH_CALUDE_apples_in_basket_l2434_243499

theorem apples_in_basket (initial_apples : ℕ) : 
  let ricki_removed : ℕ := 14
  let samson_removed : ℕ := 2 * ricki_removed
  let apples_left : ℕ := 32
  initial_apples = apples_left + ricki_removed + samson_removed :=
by sorry

end NUMINAMATH_CALUDE_apples_in_basket_l2434_243499


namespace NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l2434_243488

theorem gcd_of_quadratic_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 2 * 8753 * k) :
  Int.gcd (4 * b^2 + 27 * b + 100) (3 * b + 7) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l2434_243488


namespace NUMINAMATH_CALUDE_max_ab_given_extremum_l2434_243453

/-- Given positive real numbers a and b, and a function f with an extremum at x = 1,
    the maximum value of ab is 9. -/
theorem max_ab_given_extremum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := fun x => 4 * x^3 - a * x^2 - 2 * b * x + 2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  (a * b ≤ 9) ∧ (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = 9 ∧
    let f₀ := fun x => 4 * x^3 - a₀ * x^2 - 2 * b₀ * x + 2
    ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f₀ x ≤ f₀ 1 ∨ f₀ x ≥ f₀ 1) :=
by sorry


end NUMINAMATH_CALUDE_max_ab_given_extremum_l2434_243453


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2434_243485

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate indicating if a quadratic polynomial has roots -/
def has_roots (p : QuadraticPolynomial) : Prop :=
  p.b ^ 2 - 4 * p.a * p.c ≥ 0

/-- Given polynomial with coefficients squared -/
def squared_poly (p : QuadraticPolynomial) : QuadraticPolynomial :=
  ⟨p.a ^ 2, p.b ^ 2, p.c ^ 2⟩

/-- Given polynomial with coefficients cubed -/
def cubed_poly (p : QuadraticPolynomial) : QuadraticPolynomial :=
  ⟨p.a ^ 3, p.b ^ 3, p.c ^ 3⟩

theorem quadratic_roots_theorem (p : QuadraticPolynomial) 
  (h : has_roots p) : 
  (¬ ∀ p, has_roots p → has_roots (squared_poly p)) ∧ 
  (∀ p, has_roots p → has_roots (cubed_poly p)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2434_243485


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_l2434_243495

/-- Represents a rectangle with integer side lengths where one side is 5 feet longer than the other. -/
structure Rectangle where
  short_side : ℕ
  long_side : ℕ
  constraint : long_side = short_side + 5

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℕ := r.short_side * r.long_side

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.short_side + r.long_side)

/-- Theorem: The rectangle with minimum perimeter satisfying the given conditions has dimensions 23 and 28 feet. -/
theorem min_perimeter_rectangle :
  ∀ r : Rectangle,
    area r ≥ 600 →
    perimeter r ≥ 102 ∧
    (perimeter r = 102 → r.short_side = 23 ∧ r.long_side = 28) :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_l2434_243495


namespace NUMINAMATH_CALUDE_enrollment_difference_l2434_243443

def highest_enrollment : ℕ := 2150
def lowest_enrollment : ℕ := 980

theorem enrollment_difference : highest_enrollment - lowest_enrollment = 1170 := by
  sorry

end NUMINAMATH_CALUDE_enrollment_difference_l2434_243443


namespace NUMINAMATH_CALUDE_inscribed_angle_theorem_l2434_243487

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an arc on a circle -/
structure Arc (c : Circle) where
  start_point : ℝ × ℝ
  end_point : ℝ × ℝ

/-- The angle subtended by an arc at the center of the circle -/
def central_angle (c : Circle) (a : Arc c) : ℝ :=
  sorry

/-- The angle subtended by an arc at a point on the circumference of the circle -/
def inscribed_angle (c : Circle) (a : Arc c) : ℝ :=
  sorry

/-- The Inscribed Angle Theorem -/
theorem inscribed_angle_theorem (c : Circle) (a : Arc c) :
  inscribed_angle c a = (1 / 2) * central_angle c a :=
sorry

end NUMINAMATH_CALUDE_inscribed_angle_theorem_l2434_243487


namespace NUMINAMATH_CALUDE_thomas_needs_2000_more_l2434_243469

/-- Thomas's savings scenario over two years -/
structure SavingsScenario where
  allowance_per_week : ℕ
  weeks_in_year : ℕ
  hourly_wage : ℕ
  hours_per_week : ℕ
  car_cost : ℕ
  weekly_expenses : ℕ

/-- Calculate the amount Thomas needs to save more -/
def amount_needed_more (s : SavingsScenario) : ℕ :=
  let first_year_savings := s.allowance_per_week * s.weeks_in_year
  let second_year_earnings := s.hourly_wage * s.hours_per_week * s.weeks_in_year
  let total_earnings := first_year_savings + second_year_earnings
  let total_expenses := s.weekly_expenses * (2 * s.weeks_in_year)
  let net_savings := total_earnings - total_expenses
  s.car_cost - net_savings

/-- Thomas's specific savings scenario -/
def thomas_scenario : SavingsScenario :=
  { allowance_per_week := 50
  , weeks_in_year := 52
  , hourly_wage := 9
  , hours_per_week := 30
  , car_cost := 15000
  , weekly_expenses := 35 }

/-- Theorem stating that Thomas needs $2000 more to buy the car -/
theorem thomas_needs_2000_more :
  amount_needed_more thomas_scenario = 2000 := by sorry

end NUMINAMATH_CALUDE_thomas_needs_2000_more_l2434_243469


namespace NUMINAMATH_CALUDE_denarii_problem_l2434_243494

theorem denarii_problem (x y : ℚ) : 
  x + 7 = 5 * (y - 7) ∧ 
  y + 5 = 7 * (x - 5) → 
  x = 121 / 17 ∧ y = 167 / 17 := by
sorry

end NUMINAMATH_CALUDE_denarii_problem_l2434_243494


namespace NUMINAMATH_CALUDE_inequality_proof_l2434_243437

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2434_243437


namespace NUMINAMATH_CALUDE_dice_probability_l2434_243442

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice re-rolled -/
def numRerolled : ℕ := 3

/-- The total number of possible outcomes when re-rolling -/
def totalOutcomes : ℕ := numFaces ^ numRerolled

/-- The number of ways the re-rolled dice can not match the pair -/
def waysNotMatchingPair : ℕ := (numFaces - 1) ^ numRerolled

/-- The number of ways at least one re-rolled die matches the pair -/
def waysMatchingPair : ℕ := totalOutcomes - waysNotMatchingPair

/-- The number of ways all re-rolled dice match each other -/
def waysAllMatch : ℕ := numFaces

/-- The number of successful outcomes -/
def successfulOutcomes : ℕ := waysMatchingPair + waysAllMatch - 1

/-- The probability of at least three dice showing the same value after re-rolling -/
def probability : ℚ := successfulOutcomes / totalOutcomes

theorem dice_probability : probability = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l2434_243442


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l2434_243455

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  area_ratio : ℝ
  total_length : ℝ
  h : AB > 0
  i : CD > 0
  j : area_ratio = 5 / 2
  k : AB + CD = total_length

/-- Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC to ADC is 5:2,
    and AB + CD = 280, then AB = 200 -/
theorem trapezoid_segment_length (t : Trapezoid) (h : t.total_length = 280) : t.AB = 200 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l2434_243455


namespace NUMINAMATH_CALUDE_marathon_distance_l2434_243407

/-- Calculates the distance Tomas can run after a given number of months of training -/
def distance_after_months (initial_distance : ℕ) (months : ℕ) : ℕ :=
  initial_distance * 2^months

/-- The marathon problem -/
theorem marathon_distance (initial_distance : ℕ) (training_months : ℕ) 
  (h1 : initial_distance = 3) 
  (h2 : training_months = 5) : 
  distance_after_months initial_distance training_months = 48 := by
  sorry

#eval distance_after_months 3 5

end NUMINAMATH_CALUDE_marathon_distance_l2434_243407


namespace NUMINAMATH_CALUDE_main_tire_mileage_approx_l2434_243446

/-- Represents the mileage distribution of a car's tires -/
structure CarTires where
  total_miles : ℕ
  num_main_tires : ℕ
  num_spare_tires : ℕ
  spare_multiplier : ℕ

/-- Calculates the mileage for each main tire -/
def main_tire_mileage (c : CarTires) : ℚ :=
  c.total_miles / (c.num_main_tires + c.spare_multiplier * c.num_spare_tires)

/-- Theorem stating the main tire mileage for the given conditions -/
theorem main_tire_mileage_approx :
  let c : CarTires := {
    total_miles := 40000,
    num_main_tires := 4,
    num_spare_tires := 1,
    spare_multiplier := 2
  }
  ∃ ε > 0, |main_tire_mileage c - 6667| < ε :=
sorry

end NUMINAMATH_CALUDE_main_tire_mileage_approx_l2434_243446


namespace NUMINAMATH_CALUDE_midpoint_property_l2434_243421

/-- Given two points D and E in the plane, and F as their midpoint, 
    prove that 2x - 4y = 14 where F = (x, y) -/
theorem midpoint_property (D E F : ℝ × ℝ) : 
  D = (30, 10) →
  E = (6, 1) →
  F = ((D.1 + E.1) / 2, (D.2 + E.2) / 2) →
  2 * F.1 - 4 * F.2 = 14 := by
sorry

end NUMINAMATH_CALUDE_midpoint_property_l2434_243421


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2434_243477

theorem simplify_trig_expression (α : ℝ) :
  (2 * Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 2 * Real.tan α :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2434_243477


namespace NUMINAMATH_CALUDE_integer_pair_sum_l2434_243465

theorem integer_pair_sum (m n : ℤ) (h : (m^2 + m*n + n^2) / (m + 2*n) = 13/3) : 
  m + 2*n = 9 := by
sorry

end NUMINAMATH_CALUDE_integer_pair_sum_l2434_243465


namespace NUMINAMATH_CALUDE_greatest_common_divisor_and_digit_sum_l2434_243433

def number1 : ℕ := 1305
def number2 : ℕ := 4665
def number3 : ℕ := 6905

def difference1 : ℕ := number2 - number1
def difference2 : ℕ := number3 - number2
def difference3 : ℕ := number3 - number1

def n : ℕ := Nat.gcd difference1 (Nat.gcd difference2 difference3)

def sum_of_digits (num : ℕ) : ℕ :=
  if num < 10 then num
  else (num % 10) + sum_of_digits (num / 10)

theorem greatest_common_divisor_and_digit_sum :
  n = 1120 ∧ sum_of_digits n = 4 := by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_and_digit_sum_l2434_243433


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2434_243405

/-- A parabola intersects a line at exactly one point if and only if b = 49/12 -/
theorem parabola_line_intersection (b : ℝ) : 
  (∃! x : ℝ, bx^2 + 5*x + 4 = -2*x + 1) ↔ b = 49/12 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2434_243405


namespace NUMINAMATH_CALUDE_variance_most_appropriate_for_stability_l2434_243434

/-- Represents a set of exam scores -/
def ExamScores := List Float

/-- Calculates the variance of a list of numbers -/
def variance (scores : ExamScores) : Float := sorry

/-- Represents different statistical measures -/
inductive StatisticalMeasure
| Mean
| Variance
| Median
| Mode

/-- Determines if a statistical measure is appropriate for measuring stability -/
def isAppropriateForStability (measure : StatisticalMeasure) : Prop := sorry

/-- Theorem: Variance is the most appropriate measure for understanding stability of exam scores -/
theorem variance_most_appropriate_for_stability (scores : ExamScores) :
  isAppropriateForStability StatisticalMeasure.Variance ∧
  (∀ m : StatisticalMeasure, m ≠ StatisticalMeasure.Variance →
    ¬(isAppropriateForStability m)) := by sorry

end NUMINAMATH_CALUDE_variance_most_appropriate_for_stability_l2434_243434


namespace NUMINAMATH_CALUDE_sqrt_fifteen_div_sqrt_three_eq_sqrt_five_l2434_243497

theorem sqrt_fifteen_div_sqrt_three_eq_sqrt_five : 
  Real.sqrt 15 / Real.sqrt 3 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifteen_div_sqrt_three_eq_sqrt_five_l2434_243497


namespace NUMINAMATH_CALUDE_sum_first_49_primes_l2434_243481

def first_n_primes (n : ℕ) : List ℕ := sorry

theorem sum_first_49_primes :
  (first_n_primes 49).sum = 10787 := by sorry

end NUMINAMATH_CALUDE_sum_first_49_primes_l2434_243481


namespace NUMINAMATH_CALUDE_xy_values_l2434_243419

theorem xy_values (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 108) : 
  xy = 0 ∨ xy = 72 := by
sorry

end NUMINAMATH_CALUDE_xy_values_l2434_243419


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2434_243457

def isArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_sum (a : ℕ → ℚ) 
  (h_arith : isArithmeticSequence a) 
  (h_sum : a 3 + a 4 + a 6 + a 7 = 25) : 
  a 2 + a 8 = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2434_243457


namespace NUMINAMATH_CALUDE_max_planes_is_six_l2434_243483

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A configuration of 6 points in 3D space -/
def Configuration := Fin 6 → Point3D

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Check if a plane contains at least 4 points from the configuration -/
def planeContainsAtLeast4Points (plane : Plane3D) (config : Configuration) : Prop :=
  ∃ (p1 p2 p3 p4 : Fin 6), p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    pointOnPlane (config p1) plane ∧ pointOnPlane (config p2) plane ∧
    pointOnPlane (config p3) plane ∧ pointOnPlane (config p4) plane

/-- Check if no line passes through 4 points in the configuration -/
def noLinePasses4Points (config : Configuration) : Prop :=
  ∀ (p1 p2 p3 p4 : Fin 6), p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 →
    ¬∃ (a b c : ℝ), ∀ (p : Fin 6), p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 →
      a * (config p).x + b * (config p).y + c = (config p).z

/-- The main theorem: The maximum number of planes satisfying the conditions is 6 -/
theorem max_planes_is_six (config : Configuration) 
    (h_no_line : noLinePasses4Points config) : 
    (∃ (planes : Fin 6 → Plane3D), ∀ (i : Fin 6), planeContainsAtLeast4Points (planes i) config) ∧
    (∀ (n : ℕ) (planes : Fin (n + 1) → Plane3D), 
      (∀ (i : Fin (n + 1)), planeContainsAtLeast4Points (planes i) config) → n ≤ 5) :=
  sorry


end NUMINAMATH_CALUDE_max_planes_is_six_l2434_243483


namespace NUMINAMATH_CALUDE_prob_is_one_fourth_l2434_243400

/-- The number of cards -/
def n : ℕ := 72

/-- The set of card numbers -/
def S : Finset ℕ := Finset.range n

/-- The set of multiples of 6 in S -/
def A : Finset ℕ := S.filter (fun x => x % 6 = 0)

/-- The set of multiples of 8 in S -/
def B : Finset ℕ := S.filter (fun x => x % 8 = 0)

/-- The probability of selecting a card that is a multiple of 6 or 8 or both -/
def prob : ℚ := (A ∪ B).card / S.card

theorem prob_is_one_fourth : prob = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_is_one_fourth_l2434_243400


namespace NUMINAMATH_CALUDE_congruent_triangles_side_lengths_isosceles_triangle_side_lengths_l2434_243430

-- Part 1
theorem congruent_triangles_side_lengths (m n : ℝ) :
  (6 :: 8 :: 10 :: []).toFinset = (6 :: (2*m-2) :: (n+1) :: []).toFinset →
  ((m = 5 ∧ n = 9) ∨ (m = 6 ∧ n = 7)) := by sorry

-- Part 2
theorem isosceles_triangle_side_lengths (a b : ℝ) :
  (a = b ∧ a + a + 5 = 16) →
  ((a = 5 ∧ b = 6) ∨ (a = 5.5 ∧ b = 5)) := by sorry

end NUMINAMATH_CALUDE_congruent_triangles_side_lengths_isosceles_triangle_side_lengths_l2434_243430


namespace NUMINAMATH_CALUDE_crucian_carp_cultivation_optimal_l2434_243427

/-- Represents the seafood wholesaler's crucian carp cultivation problem -/
structure CrucianCarpProblem where
  initialWeight : ℝ  -- Initial weight of crucian carp in kg
  initialPrice : ℝ   -- Initial price per kg in yuan
  priceIncrease : ℝ  -- Daily price increase per kg in yuan
  maxDays : ℕ        -- Maximum culture period in days
  dailyLoss : ℝ      -- Daily weight loss due to oxygen deficiency in kg
  lossPrice : ℝ      -- Price of oxygen-deficient carp per kg in yuan
  dailyExpense : ℝ   -- Daily expenses during culture in yuan

/-- Calculates the profit for a given number of culture days -/
def profit (p : CrucianCarpProblem) (days : ℝ) : ℝ :=
  p.dailyLoss * days * (p.lossPrice - p.initialPrice) +
  (p.initialWeight - p.dailyLoss * days) * (p.initialPrice + p.priceIncrease * days) -
  p.initialWeight * p.initialPrice -
  p.dailyExpense * days

/-- The main theorem to be proved -/
theorem crucian_carp_cultivation_optimal (p : CrucianCarpProblem)
  (h1 : p.initialWeight = 1000)
  (h2 : p.initialPrice = 10)
  (h3 : p.priceIncrease = 1)
  (h4 : p.maxDays = 20)
  (h5 : p.dailyLoss = 10)
  (h6 : p.lossPrice = 5)
  (h7 : p.dailyExpense = 450) :
  (∃ x : ℝ, x ≤ p.maxDays ∧ profit p x = 8500 ∧ x = 10) ∧
  (∀ x : ℝ, x ≤ p.maxDays → profit p x ≤ 6000) ∧
  (∃ x : ℝ, x ≤ p.maxDays ∧ profit p x = 6000) := by
  sorry


end NUMINAMATH_CALUDE_crucian_carp_cultivation_optimal_l2434_243427


namespace NUMINAMATH_CALUDE_polynomial_value_l2434_243496

theorem polynomial_value (x y : ℝ) (h : x + 2*y = 6) : 2*x + 4*y - 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l2434_243496


namespace NUMINAMATH_CALUDE_teaching_ratio_l2434_243476

def total_years : ℕ := 52
def calculus_years : ℕ := 4

def algebra_years (c : ℕ) : ℕ := 2 * c

def statistics_years (t a c : ℕ) : ℕ := t - a - c

theorem teaching_ratio :
  let c := calculus_years
  let a := algebra_years c
  let s := statistics_years total_years a c
  (s : ℚ) / a = 5 / 1 :=
sorry

end NUMINAMATH_CALUDE_teaching_ratio_l2434_243476


namespace NUMINAMATH_CALUDE_absolute_value_six_point_five_l2434_243459

theorem absolute_value_six_point_five (x : ℝ) : |x| = 6.5 ↔ x = 6.5 ∨ x = -6.5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_six_point_five_l2434_243459


namespace NUMINAMATH_CALUDE_average_monthly_bill_l2434_243425

theorem average_monthly_bill (first_four_months_avg : ℝ) (last_two_months_avg : ℝ) :
  first_four_months_avg = 30 →
  last_two_months_avg = 24 →
  (4 * first_four_months_avg + 2 * last_two_months_avg) / 6 = 28 :=
by sorry

end NUMINAMATH_CALUDE_average_monthly_bill_l2434_243425


namespace NUMINAMATH_CALUDE_apples_sold_per_day_l2434_243484

/-- Calculates the average number of apples sold per day given the total number of boxes,
    days, and apples per box. -/
def average_apples_per_day (boxes : ℕ) (days : ℕ) (apples_per_box : ℕ) : ℚ :=
  (boxes * apples_per_box : ℚ) / days

/-- Theorem stating that given 12 boxes of apples sold in 4 days,
    with 25 apples per box, the average number of apples sold per day is 75. -/
theorem apples_sold_per_day :
  average_apples_per_day 12 4 25 = 75 := by
  sorry

end NUMINAMATH_CALUDE_apples_sold_per_day_l2434_243484


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l2434_243473

/-- Calculates the sampling interval for systematic sampling -/
def sampling_interval (N : ℕ) (n : ℕ) : ℕ := N / n

/-- Theorem: The sampling interval for a population of 1000 and sample size of 20 is 50 -/
theorem systematic_sampling_interval :
  sampling_interval 1000 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l2434_243473


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2434_243441

/-- Given a principal sum and a time period of 8 years, if the simple interest
    is one-fifth of the principal, then the rate of interest per annum is 2.5%. -/
theorem interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  P / 5 = P * 8 * (2.5 / 100) := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2434_243441


namespace NUMINAMATH_CALUDE_remainder_theorem_l2434_243432

theorem remainder_theorem (P D D' Q Q' R R' : ℤ) 
  (h1 : P = Q * D + R) 
  (h2 : 0 ≤ R ∧ R < D) 
  (h3 : Q = Q' * D' + R') 
  (h4 : 0 ≤ R' ∧ R' < D') : 
  P % (D * D') = R + R' * D :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2434_243432


namespace NUMINAMATH_CALUDE_purple_chips_count_l2434_243479

/-- Represents the number of chips of each color selected -/
structure ChipSelection where
  blue : ℕ
  green : ℕ
  purple : ℕ
  red : ℕ

/-- The theorem stating the number of purple chips selected -/
theorem purple_chips_count 
  (x : ℕ) 
  (h1 : 5 < x) 
  (h2 : x < 11) 
  (selection : ChipSelection) 
  (h3 : 1^selection.blue * 5^selection.green * x^selection.purple * 11^selection.red = 28160) :
  selection.purple = 2 :=
sorry

end NUMINAMATH_CALUDE_purple_chips_count_l2434_243479


namespace NUMINAMATH_CALUDE_number_of_small_boxes_correct_number_of_small_boxes_l2434_243440

theorem number_of_small_boxes 
  (dolls_per_big_box : ℕ) 
  (dolls_per_small_box : ℕ) 
  (num_big_boxes : ℕ) 
  (total_dolls : ℕ) : ℕ :=
  let remaining_dolls := total_dolls - dolls_per_big_box * num_big_boxes
  remaining_dolls / dolls_per_small_box

theorem correct_number_of_small_boxes :
  number_of_small_boxes 7 4 5 71 = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_small_boxes_correct_number_of_small_boxes_l2434_243440


namespace NUMINAMATH_CALUDE_fraction_simplification_l2434_243491

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hxy : x^2 - 1/y ≠ 0) 
  (hyx : y^2 - 1/x ≠ 0) : 
  (x^2 - 1/y) / (y^2 - 1/x) = x * (x^2*y - 1) / (y * (y^2*x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2434_243491


namespace NUMINAMATH_CALUDE_g_negative_three_l2434_243451

def g (x : ℝ) : ℝ := 3*x^5 - 5*x^4 + 9*x^3 - 6*x^2 + 15*x - 210

theorem g_negative_three : g (-3) = -1686 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_three_l2434_243451


namespace NUMINAMATH_CALUDE_number_less_than_opposite_l2434_243461

theorem number_less_than_opposite (x : ℝ) : x = -x + (-4) ↔ x + 4 = -x := by sorry

end NUMINAMATH_CALUDE_number_less_than_opposite_l2434_243461


namespace NUMINAMATH_CALUDE_line_through_points_2m_minus_b_l2434_243438

/-- Given a line passing through points (1,3) and (4,15), prove that 2m - b = 9 where y = mx + b is the equation of the line. -/
theorem line_through_points_2m_minus_b (m b : ℝ) : 
  (3 : ℝ) = m * 1 + b → 
  (15 : ℝ) = m * 4 + b → 
  2 * m - b = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_2m_minus_b_l2434_243438


namespace NUMINAMATH_CALUDE_tim_income_percentage_forty_percent_less_l2434_243413

/-- Proves that Tim's income is 40% less than Juan's income given the conditions -/
theorem tim_income_percentage (tim mary juan : ℝ) 
  (h1 : mary = 1.4 * tim) 
  (h2 : mary = 0.84 * juan) : 
  tim = 0.6 * juan := by
  sorry

/-- Proves that 40% less is equivalent to 60% of the original amount -/
theorem forty_percent_less (x y : ℝ) (h : x = 0.6 * y) : 
  x = y - 0.4 * y := by
  sorry

end NUMINAMATH_CALUDE_tim_income_percentage_forty_percent_less_l2434_243413


namespace NUMINAMATH_CALUDE_cricket_average_l2434_243415

theorem cricket_average (initial_average : ℝ) (innings : ℕ) (new_score : ℝ) (average_increase : ℝ) :
  innings = 16 →
  new_score = 92 →
  average_increase = 4 →
  (innings * initial_average + new_score) / (innings + 1) = initial_average + average_increase →
  initial_average + average_increase = 28 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_l2434_243415


namespace NUMINAMATH_CALUDE_max_value_abcd_l2434_243402

theorem max_value_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a * b * c * d * (a + b + c + d)) / ((a + b)^2 * (c + d)^2) ≤ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_abcd_l2434_243402


namespace NUMINAMATH_CALUDE_set_operations_theorem_l2434_243458

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem set_operations_theorem :
  (Set.compl A ∪ B = {x | x < 5}) ∧
  (A ∩ Set.compl B = {x | x ≥ 5}) := by sorry

end NUMINAMATH_CALUDE_set_operations_theorem_l2434_243458


namespace NUMINAMATH_CALUDE_correct_card_to_disprove_jane_l2434_243478

-- Define the type for card sides
inductive CardSide
| Letter (c : Char)
| Number (n : Nat)

-- Define the structure for a card
structure Card where
  side1 : CardSide
  side2 : CardSide

-- Define the function to check if a number is odd
def isOdd (n : Nat) : Bool :=
  n % 2 = 1

-- Define the function to check if a character is a vowel
def isVowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

-- Define Jane's statement as a function
def janesStatement (card : Card) : Bool :=
  match card.side1, card.side2 with
  | CardSide.Number n, CardSide.Letter c => 
      ¬(isOdd n ∧ isVowel c)
  | CardSide.Letter c, CardSide.Number n => 
      ¬(isOdd n ∧ isVowel c)
  | _, _ => true

-- Define the theorem
theorem correct_card_to_disprove_jane : 
  ∀ (cards : List Card),
  cards = [
    Card.mk (CardSide.Letter 'A') (CardSide.Number 0),
    Card.mk (CardSide.Letter 'S') (CardSide.Number 0),
    Card.mk (CardSide.Number 5) (CardSide.Letter ' '),
    Card.mk (CardSide.Number 8) (CardSide.Letter ' '),
    Card.mk (CardSide.Number 7) (CardSide.Letter ' ')
  ] →
  ∃ (card : Card),
  card ∈ cards ∧ 
  card.side1 = CardSide.Letter 'A' ∧
  (∃ (n : Nat), card.side2 = CardSide.Number n ∧ isOdd n) ∧
  (∀ (c : Card), c ∈ cards ∧ c ≠ card → janesStatement c) :=
by sorry


end NUMINAMATH_CALUDE_correct_card_to_disprove_jane_l2434_243478


namespace NUMINAMATH_CALUDE_p_plus_q_equals_31_l2434_243475

theorem p_plus_q_equals_31 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-5 * x^2 + 18 * x + 27) / (x - 3)) →
  P + Q = 31 := by
sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_31_l2434_243475


namespace NUMINAMATH_CALUDE_derivative_sin_cos_l2434_243414

theorem derivative_sin_cos (x : ℝ) :
  deriv (λ x => 3 * Real.sin x - 4 * Real.cos x) x = 3 * Real.cos x + 4 * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_cos_l2434_243414


namespace NUMINAMATH_CALUDE_milk_remaining_l2434_243450

theorem milk_remaining (initial : ℚ) (given_away : ℚ) (remaining : ℚ) : 
  initial = 4 → given_away = 16/3 → remaining = initial - given_away → remaining = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_milk_remaining_l2434_243450


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2434_243454

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The fifth term of an arithmetic sequence is 5, given the sum of the first and ninth terms is 10 -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 1 + a 9 = 10) :
  a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2434_243454


namespace NUMINAMATH_CALUDE_equation_solution_l2434_243428

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (2*x + 1)*(3*x + 1)*(5*x + 1)*(30*x + 1)
  ∀ x : ℝ, f x = 10 ↔ x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2434_243428


namespace NUMINAMATH_CALUDE_existence_of_special_number_l2434_243424

/-- A function that returns the decimal representation of a natural number as a list of digits -/
def decimal_representation (n : ℕ) : List ℕ := sorry

/-- A function that counts the occurrences of a digit in a list of digits -/
def count_occurrences (digit : ℕ) (digits : List ℕ) : ℕ := sorry

/-- A function that interchanges two digits at given positions in a list of digits -/
def interchange_digits (digits : List ℕ) (pos1 pos2 : ℕ) : List ℕ := sorry

/-- A function that converts a list of digits back to a natural number -/
def from_digits (digits : List ℕ) : ℕ := sorry

/-- The set of prime divisors of a natural number -/
def prime_divisors (n : ℕ) : Set ℕ := sorry

theorem existence_of_special_number :
  ∃ n : ℕ,
    (∀ d : ℕ, d < 10 → count_occurrences d (decimal_representation n) ≥ 2006) ∧
    (∃ pos1 pos2 : ℕ,
      pos1 ≠ pos2 ∧
      let digits := decimal_representation n
      let m := from_digits (interchange_digits digits pos1 pos2)
      n ≠ m ∧
      prime_divisors n = prime_divisors m) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l2434_243424


namespace NUMINAMATH_CALUDE_profit_growth_rate_l2434_243482

/-- The average monthly growth rate that achieves the target profit -/
def average_growth_rate : ℝ := 0.2

/-- The initial profit in June -/
def initial_profit : ℝ := 2500

/-- The target profit in August -/
def target_profit : ℝ := 3600

/-- The number of months between June and August -/
def months : ℕ := 2

theorem profit_growth_rate :
  initial_profit * (1 + average_growth_rate) ^ months = target_profit :=
sorry

end NUMINAMATH_CALUDE_profit_growth_rate_l2434_243482


namespace NUMINAMATH_CALUDE_quadratic_discriminant_zero_implies_perfect_square_l2434_243404

/-- 
If the discriminant of a quadratic equation ax^2 + bx + c = 0 is zero,
then the left-hand side is a perfect square.
-/
theorem quadratic_discriminant_zero_implies_perfect_square
  (a b c : ℝ) (h : b^2 - 4*a*c = 0) :
  ∃ k : ℝ, ∀ x : ℝ, a*x^2 + b*x + c = k*(2*a*x + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_zero_implies_perfect_square_l2434_243404


namespace NUMINAMATH_CALUDE_inequality_proof_l2434_243426

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2434_243426


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_b_value_l2434_243436

theorem infinite_solutions_imply_b_value :
  ∀ b : ℚ, (∀ x : ℚ, 4 * (3 * x - b) = 3 * (4 * x + 7)) → b = -21/4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_b_value_l2434_243436


namespace NUMINAMATH_CALUDE_square_side_length_l2434_243435

theorem square_side_length (rectangle_width : ℝ) (rectangle_length : ℝ) 
  (h1 : rectangle_width = 3)
  (h2 : rectangle_length = 3)
  (h3 : square_area = rectangle_width * rectangle_length) : 
  ∃ (square_side : ℝ), square_side^2 = square_area ∧ square_side = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2434_243435


namespace NUMINAMATH_CALUDE_book_chapters_l2434_243422

theorem book_chapters (total_pages : ℕ) (pages_per_chapter : ℕ) (h1 : total_pages = 555) (h2 : pages_per_chapter = 111) :
  total_pages / pages_per_chapter = 5 := by
sorry

end NUMINAMATH_CALUDE_book_chapters_l2434_243422


namespace NUMINAMATH_CALUDE_square_plus_self_divisible_by_two_l2434_243470

theorem square_plus_self_divisible_by_two (a : ℤ) : 
  ∃ k : ℤ, a^2 + a = 2 * k :=
by
  sorry

end NUMINAMATH_CALUDE_square_plus_self_divisible_by_two_l2434_243470


namespace NUMINAMATH_CALUDE_min_value_of_complex_expression_l2434_243452

theorem min_value_of_complex_expression :
  ∃ (min_u : ℝ), min_u = (3/2) * Real.sqrt 3 ∧
  ∀ (z : ℂ), Complex.abs z = 2 →
  Complex.abs (z^2 - z + 1) ≥ min_u :=
sorry

end NUMINAMATH_CALUDE_min_value_of_complex_expression_l2434_243452


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2434_243410

/-- A triangle with an inscribed circle where the area is numerically twice the perimeter -/
structure SpecialTriangle where
  -- The semiperimeter of the triangle
  s : ℝ
  -- The area of the triangle
  A : ℝ
  -- The perimeter of the triangle
  p : ℝ
  -- The radius of the inscribed circle
  r : ℝ
  -- The semiperimeter is positive
  s_pos : 0 < s
  -- The perimeter is twice the semiperimeter
  perim_eq : p = 2 * s
  -- The area is twice the perimeter
  area_eq : A = 2 * p
  -- The area formula using inradius
  area_formula : A = r * s

/-- The radius of the inscribed circle in a SpecialTriangle is 4 -/
theorem inscribed_circle_radius (t : SpecialTriangle) : t.r = 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2434_243410


namespace NUMINAMATH_CALUDE_cara_arrangements_l2434_243429

def num_friends : ℕ := 7

def arrangements (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

theorem cara_arrangements :
  arrangements num_friends = 6 :=
sorry

end NUMINAMATH_CALUDE_cara_arrangements_l2434_243429


namespace NUMINAMATH_CALUDE_stratified_sampling_population_size_l2434_243403

theorem stratified_sampling_population_size
  (x : ℕ) -- number of individuals in stratum A
  (y : ℕ) -- number of individuals in stratum B
  (h1 : (20 : ℚ) * y / (x + y) = (1 : ℚ) / 12 * y) -- equation from stratified sampling
  : x + y = 240 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_population_size_l2434_243403


namespace NUMINAMATH_CALUDE_unique_solution_l2434_243489

/-- A function satisfying the given conditions -/
def satisfies_conditions (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a : ℝ, 
    (∀ x y z : ℝ, f x y + f y z + f z x = max x (max y z) - min x (min y z)) ∧
    (∀ x : ℝ, f x a = f a x)

/-- The theorem stating the unique solution -/
theorem unique_solution : 
  ∃! f : ℝ → ℝ → ℝ, satisfies_conditions f ∧ ∀ x y : ℝ, f x y = |x - y| / 2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2434_243489


namespace NUMINAMATH_CALUDE_debate_pairs_l2434_243418

theorem debate_pairs (n : ℕ) (h : n = 12) : Nat.choose n 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_debate_pairs_l2434_243418


namespace NUMINAMATH_CALUDE_only_statement_one_true_l2434_243439

variable (b x y : ℝ)

theorem only_statement_one_true :
  (∀ x y, b * (x + y) = b * x + b * y) ∧
  (∃ x y, b^(x + y) ≠ b^x + b^y) ∧
  (∃ x y, Real.log (x + y) ≠ Real.log x + Real.log y) ∧
  (∃ x y, Real.log x / Real.log y ≠ Real.log (x * y)) ∧
  (∃ x y, b * (x / y) ≠ (b * x) / (b * y)) :=
by sorry

end NUMINAMATH_CALUDE_only_statement_one_true_l2434_243439


namespace NUMINAMATH_CALUDE_abs_four_implies_plus_minus_four_l2434_243431

theorem abs_four_implies_plus_minus_four (x : ℝ) : |x| = 4 → x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_abs_four_implies_plus_minus_four_l2434_243431


namespace NUMINAMATH_CALUDE_det_scale_by_three_l2434_243468

theorem det_scale_by_three (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 7 →
  Matrix.det !![3*x, 3*y; 3*z, 3*w] = 63 := by
  sorry

end NUMINAMATH_CALUDE_det_scale_by_three_l2434_243468


namespace NUMINAMATH_CALUDE_total_fruits_is_236_l2434_243464

/-- The total number of fruits picked by Sara and Sally -/
def total_fruits (sara_pears sara_apples sara_plums sally_pears sally_apples sally_plums : ℕ) : ℕ :=
  (sara_pears + sally_pears) + (sara_apples + sally_apples) + (sara_plums + sally_plums)

/-- Theorem: The total number of fruits picked by Sara and Sally is 236 -/
theorem total_fruits_is_236 :
  total_fruits 45 22 64 11 38 56 = 236 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_is_236_l2434_243464


namespace NUMINAMATH_CALUDE_vector_relations_l2434_243408

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![3, -4]

def is_collinear (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

def is_perpendicular (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 0 + v 1 * w 1 = 0

theorem vector_relations :
  (∃ k : ℝ, is_collinear (fun i => k * a i - b i) (fun i => a i + b i) ∧ k = -1) ∧
  (∃ k : ℝ, is_perpendicular (fun i => k * a i - b i) (fun i => a i + b i) ∧ k = 16) := by
  sorry


end NUMINAMATH_CALUDE_vector_relations_l2434_243408


namespace NUMINAMATH_CALUDE_function_characterization_l2434_243411

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the property that the function must satisfy
def SatisfiesEquation (f : RealFunction) : Prop :=
  ∀ x y : ℝ, f ((x - y)^2) = x^2 - 2*y*f x + (f y)^2

-- State the theorem
theorem function_characterization :
  ∀ f : RealFunction, SatisfiesEquation f ↔ (∀ x : ℝ, f x = x ∨ f x = x + 1) :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l2434_243411


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l2434_243498

/-- There exist positive integers a and b with a > b > 1 such that
    for all positive integers k, there exists a positive integer n
    where an + b is a k-th power of a positive integer. -/
theorem existence_of_special_integers : ∃ (a b : ℕ), 
  a > b ∧ b > 1 ∧ 
  ∀ (k : ℕ), k > 0 → 
    ∃ (n m : ℕ), n > 0 ∧ m > 0 ∧ a * n + b = m ^ k :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l2434_243498


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2434_243474

/-- Given a train crossing a bridge and a lamp post, calculate the bridge length -/
theorem bridge_length_calculation (train_length : ℝ) (bridge_crossing_time : ℝ) (lamp_post_crossing_time : ℝ) :
  train_length = 833.33 →
  bridge_crossing_time = 120 →
  lamp_post_crossing_time = 30 →
  ∃ bridge_length : ℝ, bridge_length = 2500 := by
  sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l2434_243474


namespace NUMINAMATH_CALUDE_halloween_trick_or_treat_l2434_243460

theorem halloween_trick_or_treat (duration : ℕ) (houses_per_hour : ℕ) (treats_per_house : ℕ) (total_treats : ℕ) :
  duration = 4 →
  houses_per_hour = 5 →
  treats_per_house = 3 →
  total_treats = 180 →
  total_treats / (duration * houses_per_hour * treats_per_house) = 3 := by
sorry


end NUMINAMATH_CALUDE_halloween_trick_or_treat_l2434_243460


namespace NUMINAMATH_CALUDE_michaels_pets_l2434_243412

theorem michaels_pets (total_pets : ℕ) 
  (h1 : (total_pets : ℝ) * 0.25 = total_pets * 0.5 + 9) 
  (h2 : (total_pets : ℝ) * 0.25 + total_pets * 0.5 + 9 = total_pets) : 
  total_pets = 36 := by
  sorry

end NUMINAMATH_CALUDE_michaels_pets_l2434_243412


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2434_243472

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x - 7 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 + m * y - 7 = 0 ∧ y = -7/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2434_243472


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2434_243401

theorem sin_2alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.tan (π / 4 - α) = 1 / 3) : 
  Real.sin (2 * α) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2434_243401


namespace NUMINAMATH_CALUDE_ratio_of_arithmetic_sequences_l2434_243423

def arithmetic_sequence_sum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem ratio_of_arithmetic_sequences : 
  let seq1_sum := arithmetic_sequence_sum 4 4 48
  let seq2_sum := arithmetic_sequence_sum 2 3 35
  seq1_sum / seq2_sum = 52 / 37 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_arithmetic_sequences_l2434_243423


namespace NUMINAMATH_CALUDE_max_area_triangle_area_equals_perimeter_l2434_243444

theorem max_area_triangle_area_equals_perimeter : ∃ (a b c : ℕ+),
  (∃ (s : ℝ), s = (a + b + c : ℝ) / 2 ∧ 
   (s * (s - a) * (s - b) * (s - c) : ℝ) = ((a + b + c) ^ 2 : ℝ) / 4) ∧
  (∀ (x y z : ℕ+), 
    (∃ (t : ℝ), t = (x + y + z : ℝ) / 2 ∧ 
     (t * (t - x) * (t - y) * (t - z) : ℝ) = ((x + y + z) ^ 2 : ℝ) / 4) →
    (x + y + z : ℝ) ≤ (a + b + c : ℝ)) :=
by sorry

#check max_area_triangle_area_equals_perimeter

end NUMINAMATH_CALUDE_max_area_triangle_area_equals_perimeter_l2434_243444


namespace NUMINAMATH_CALUDE_order_of_powers_l2434_243409

theorem order_of_powers : 3^15 < 2^30 ∧ 2^30 < 10^10 := by
  sorry

end NUMINAMATH_CALUDE_order_of_powers_l2434_243409


namespace NUMINAMATH_CALUDE_peter_erasers_l2434_243467

theorem peter_erasers (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 35 → received = 17 → total = initial + received → total = 52 := by
  sorry

end NUMINAMATH_CALUDE_peter_erasers_l2434_243467


namespace NUMINAMATH_CALUDE_sequence_sum_l2434_243486

theorem sequence_sum (n : ℕ) (x : ℕ → ℝ) (h1 : x 1 = 3) 
  (h2 : ∀ k ∈ Finset.range (n - 1), x (k + 1) = x k + k) : 
  Finset.sum (Finset.range n) (λ k => x (k + 1)) = 3*n + (n*(n+1)*(2*n-1))/12 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l2434_243486


namespace NUMINAMATH_CALUDE_prob_green_or_yellow_l2434_243490

/-- A cube with colored faces -/
structure ColoredCube where
  green_faces : ℕ
  yellow_faces : ℕ
  blue_faces : ℕ

/-- The probability of an event -/
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

/-- Theorem: Probability of rolling a green or yellow face -/
theorem prob_green_or_yellow (cube : ColoredCube) 
  (h1 : cube.green_faces = 3)
  (h2 : cube.yellow_faces = 2)
  (h3 : cube.blue_faces = 1) :
  probability (cube.green_faces + cube.yellow_faces) 
    (cube.green_faces + cube.yellow_faces + cube.blue_faces) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_or_yellow_l2434_243490


namespace NUMINAMATH_CALUDE_distance_difference_l2434_243417

/-- The difference in distances from Q to the intersection points of a line and a parabola -/
theorem distance_difference (Q : ℝ × ℝ) (C D : ℝ × ℝ) : 
  Q.1 = 2 ∧ Q.2 = 0 →
  C.2 - 2 * C.1 + 4 = 0 →
  D.2 - 2 * D.1 + 4 = 0 →
  C.2^2 = 3 * C.1 + 4 →
  D.2^2 = 3 * D.1 + 4 →
  |((C.1 - Q.1)^2 + (C.2 - Q.2)^2).sqrt - ((D.1 - Q.1)^2 + (D.2 - Q.2)^2).sqrt| = 
  |2 * (5 : ℝ).sqrt - (8.90625 : ℝ).sqrt| :=
by sorry

end NUMINAMATH_CALUDE_distance_difference_l2434_243417


namespace NUMINAMATH_CALUDE_point_positions_l2434_243480

/-- Circle C is defined by the equation x^2 + y^2 - 2x + 4y - 4 = 0 --/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

/-- Point M has coordinates (2, -4) --/
def point_M : ℝ × ℝ := (2, -4)

/-- Point N has coordinates (-2, 1) --/
def point_N : ℝ × ℝ := (-2, 1)

/-- A point (x, y) is inside the circle if x^2 + y^2 - 2x + 4y - 4 < 0 --/
def inside_circle (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + y^2 - 2*x + 4*y - 4 < 0

/-- A point (x, y) is outside the circle if x^2 + y^2 - 2x + 4y - 4 > 0 --/
def outside_circle (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + y^2 - 2*x + 4*y - 4 > 0

theorem point_positions :
  inside_circle point_M ∧ outside_circle point_N :=
sorry

end NUMINAMATH_CALUDE_point_positions_l2434_243480


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2434_243420

theorem quadratic_equation_unique_solution (b : ℝ) (hb : b ≠ 0) :
  (∃! x, b * x^2 + 15 * x + 4 = 0) →
  (∃ x, b * x^2 + 15 * x + 4 = 0 ∧ x = -8/15) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2434_243420


namespace NUMINAMATH_CALUDE_honeydews_left_l2434_243466

/-- Represents the problem of Darryl's melon sales --/
structure MelonSales where
  cantaloupe_price : ℕ
  honeydew_price : ℕ
  initial_cantaloupes : ℕ
  initial_honeydews : ℕ
  dropped_cantaloupes : ℕ
  rotten_honeydews : ℕ
  final_cantaloupes : ℕ
  total_revenue : ℕ

/-- Theorem stating the number of honeydews left at the end of the day --/
theorem honeydews_left (sale : MelonSales)
  (h1 : sale.cantaloupe_price = 2)
  (h2 : sale.honeydew_price = 3)
  (h3 : sale.initial_cantaloupes = 30)
  (h4 : sale.initial_honeydews = 27)
  (h5 : sale.dropped_cantaloupes = 2)
  (h6 : sale.rotten_honeydews = 3)
  (h7 : sale.final_cantaloupes = 8)
  (h8 : sale.total_revenue = 85) :
  sale.initial_honeydews - sale.rotten_honeydews -
  ((sale.total_revenue - (sale.initial_cantaloupes - sale.dropped_cantaloupes - sale.final_cantaloupes) * sale.cantaloupe_price) / sale.honeydew_price) = 9 :=
sorry

end NUMINAMATH_CALUDE_honeydews_left_l2434_243466


namespace NUMINAMATH_CALUDE_log7_10_approximation_l2434_243471

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.301
def log10_3_approx : ℝ := 0.499

-- Define the target approximation
def log7_10_approx : ℝ := 2

-- Theorem statement
theorem log7_10_approximation :
  abs (Real.log 10 / Real.log 7 - log7_10_approx) < 0.1 :=
sorry


end NUMINAMATH_CALUDE_log7_10_approximation_l2434_243471

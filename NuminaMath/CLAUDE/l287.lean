import Mathlib

namespace NUMINAMATH_CALUDE_division_of_power_sixteen_l287_28794

theorem division_of_power_sixteen (m : ℕ) : m = 16^2024 → m/8 = 8 * 16^2020 := by
  sorry

end NUMINAMATH_CALUDE_division_of_power_sixteen_l287_28794


namespace NUMINAMATH_CALUDE_second_question_correct_percentage_l287_28799

-- Define the percentages as real numbers between 0 and 100
def first_correct : ℝ := 80
def neither_correct : ℝ := 5
def both_correct : ℝ := 60

-- Define the function to calculate the percentage who answered the second question correctly
def second_correct : ℝ := 100 - neither_correct - first_correct + both_correct

-- Theorem statement
theorem second_question_correct_percentage :
  second_correct = 75 :=
sorry

end NUMINAMATH_CALUDE_second_question_correct_percentage_l287_28799


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l287_28706

theorem purely_imaginary_condition (x : ℝ) : 
  (((x^2 - 1) : ℂ) + (x + 1)*I).re = 0 ∧ (((x^2 - 1) : ℂ) + (x + 1)*I).im ≠ 0 → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l287_28706


namespace NUMINAMATH_CALUDE_equation_is_parabola_and_ellipse_l287_28728

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents the equation y^4 - 6x^4 = 3y^2 + 1 -/
def equation (p : Point2D) : Prop :=
  p.y^4 - 6*p.x^4 = 3*p.y^2 + 1

/-- Represents a parabola in 2D space -/
def isParabola (S : Set Point2D) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ p ∈ S, p.y = a*p.x^2 + b*p.x + c

/-- Represents an ellipse in 2D space -/
def isEllipse (S : Set Point2D) : Prop :=
  ∃ h k a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ p ∈ S, 
    ((p.x - h)^2 / a^2) + ((p.y - k)^2 / b^2) = 1

/-- The set of all points satisfying the equation -/
def S : Set Point2D :=
  {p : Point2D | equation p}

/-- Theorem stating that the equation represents the union of a parabola and an ellipse -/
theorem equation_is_parabola_and_ellipse :
  ∃ P E : Set Point2D, isParabola P ∧ isEllipse E ∧ S = P ∪ E :=
sorry

end NUMINAMATH_CALUDE_equation_is_parabola_and_ellipse_l287_28728


namespace NUMINAMATH_CALUDE_tom_and_mary_ages_l287_28780

theorem tom_and_mary_ages :
  ∃ (tom_age mary_age : ℕ),
    tom_age^2 + mary_age = 62 ∧
    mary_age^2 + tom_age = 176 ∧
    tom_age = 7 ∧
    mary_age = 13 := by
  sorry

end NUMINAMATH_CALUDE_tom_and_mary_ages_l287_28780


namespace NUMINAMATH_CALUDE_smallest_y_congruence_l287_28702

theorem smallest_y_congruence : 
  ∃ y : ℕ, y > 0 ∧ (26 * y + 8) % 16 = 4 ∧ ∀ z : ℕ, z > 0 → (26 * z + 8) % 16 = 4 → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_congruence_l287_28702


namespace NUMINAMATH_CALUDE_vector_simplification_l287_28752

-- Define the Euclidean space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

-- Define points in the Euclidean space
variable (A B C D : E)

-- Define vectors as differences between points
def vector (P Q : E) : E := Q - P

-- State the theorem
theorem vector_simplification (A B C D : E) :
  vector A B + vector B C - vector A D = vector D C := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l287_28752


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l287_28725

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x ≥ -2) ∧
  (∃ x : ℝ, x ≥ -2 ∧ ¬(-1 ≤ x ∧ x ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l287_28725


namespace NUMINAMATH_CALUDE_black_squares_count_l287_28787

/-- Represents a checkerboard with alternating colors -/
structure Checkerboard :=
  (size : Nat)
  (has_black_corners : Bool)
  (alternating : Bool)

/-- Counts the number of black squares on a checkerboard -/
def count_black_squares (board : Checkerboard) : Nat :=
  sorry

/-- Theorem: A 32x32 checkerboard with black corners and alternating colors has 512 black squares -/
theorem black_squares_count (board : Checkerboard) :
  board.size = 32 ∧ board.has_black_corners ∧ board.alternating →
  count_black_squares board = 512 :=
sorry

end NUMINAMATH_CALUDE_black_squares_count_l287_28787


namespace NUMINAMATH_CALUDE_blue_marbles_percentage_l287_28779

/-- Represents the number of marbles Pete has initially -/
def initial_marbles : ℕ := 10

/-- Represents the number of marbles Pete has after trading -/
def final_marbles : ℕ := 15

/-- Represents the number of red marbles Pete keeps after trading -/
def kept_red_marbles : ℕ := 1

/-- Calculates the percentage of blue marbles initially -/
def blue_percentage (blue : ℕ) : ℚ :=
  (blue : ℚ) / (initial_marbles : ℚ) * 100

/-- The main theorem stating the percentage of blue marbles -/
theorem blue_marbles_percentage :
  ∃ (blue red : ℕ),
    blue + red = initial_marbles ∧
    blue + 2 * (red - kept_red_marbles) + kept_red_marbles = final_marbles ∧
    blue_percentage blue = 40 := by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_percentage_l287_28779


namespace NUMINAMATH_CALUDE_eight_pointed_star_tip_sum_l287_28722

/-- An 8-pointed star formed by connecting 8 evenly spaced points on a circle -/
structure EightPointedStar where
  /-- The number of points on the circle -/
  num_points : ℕ
  /-- The points are evenly spaced -/
  evenly_spaced : num_points = 8
  /-- The measure of each small arc between adjacent points -/
  small_arc_measure : ℝ
  /-- Each small arc is 1/8 of the full circle -/
  small_arc_def : small_arc_measure = 360 / 8

/-- The sum of angle measurements of the eight tips of the star -/
def sum_of_tip_angles (star : EightPointedStar) : ℝ :=
  8 * (360 - 4 * star.small_arc_measure)

theorem eight_pointed_star_tip_sum :
  ∀ (star : EightPointedStar), sum_of_tip_angles star = 1440 := by
  sorry

end NUMINAMATH_CALUDE_eight_pointed_star_tip_sum_l287_28722


namespace NUMINAMATH_CALUDE_largest_quantity_l287_28782

theorem largest_quantity : 
  (2008 / 2007 + 2008 / 2009 : ℚ) > (2009 / 2008 + 2009 / 2010 : ℚ) ∧ 
  (2009 / 2008 + 2009 / 2010 : ℚ) > (2008 / 2009 + 2010 / 2009 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l287_28782


namespace NUMINAMATH_CALUDE_intersection_distance_l287_28717

/-- Given a linear function f(x) = ax + b, if the distance between the intersection points
    of y=x^2+2 and y=f(x) is √10, and the distance between the intersection points of
    y=x^2-1 and y=f(x)+1 is √42, then the distance between the intersection points of
    y=x^2 and y=f(x)+1 is √34. -/
theorem intersection_distance (a b : ℝ) : 
  let f := (fun x : ℝ => a * x + b)
  let d1 := Real.sqrt ((a^2 + 1) * (a^2 + 4*b - 8))
  let d2 := Real.sqrt ((a^2 + 1) * (a^2 + 4*b + 8))
  d1 = Real.sqrt 10 ∧ d2 = Real.sqrt 42 →
  Real.sqrt ((a^2 + 1) * (a^2 + 4*b + 4)) = Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l287_28717


namespace NUMINAMATH_CALUDE_max_distance_from_origin_l287_28748

theorem max_distance_from_origin (x y : ℝ) :
  x^2 + y^2 - 4*x - 4*y + 6 = 0 →
  ∃ (max_val : ℝ), max_val = 3 * Real.sqrt 2 ∧
    ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 →
      Real.sqrt (x'^2 + y'^2) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_distance_from_origin_l287_28748


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l287_28795

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l287_28795


namespace NUMINAMATH_CALUDE_smallest_block_size_l287_28730

/-- Given a rectangular block with dimensions l, m, and n,
    where (l-1)(m-1)(n-1) = 210, the smallest possible value of l*m*n is 336. -/
theorem smallest_block_size (l m n : ℕ) (h : (l-1)*(m-1)*(n-1) = 210) :
  l*m*n ≥ 336 ∧ ∃ (l' m' n' : ℕ), (l'-1)*(m'-1)*(n'-1) = 210 ∧ l'*m'*n' = 336 := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_size_l287_28730


namespace NUMINAMATH_CALUDE_equation_solutions_l287_28767

theorem equation_solutions :
  (∃ x : ℚ, x + 2 * (x - 3) = 3 * (1 - x) ∧ x = 3/2) ∧
  (∃ x : ℚ, 1 - (2*x - 1)/3 = (3 + x)/6 ∧ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l287_28767


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l287_28734

/-- Given a sequence {aₙ} where Sₙ (the sum of the first n terms) is defined as Sₙ = n² + 1,
    prove that the 5th term of the sequence (a₅) is equal to 9. -/
theorem fifth_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = n^2 + 1) : a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l287_28734


namespace NUMINAMATH_CALUDE_company_age_distribution_l287_28726

def Department : Type := List Nat

def mode (dept : Department) : Nat :=
  sorry

def median (dept : Department) : Nat :=
  sorry

def average (dept : Department) : Real :=
  sorry

theorem company_age_distribution 
  (dept_A dept_B : Department) 
  (h_A : dept_A = [17, 22, 23, 24, 24, 25, 26, 32, 32, 32])
  (h_B : dept_B = [18, 20, 21, 24, 24, 28, 28, 30, 32, 50]) :
  (mode dept_A = 32) ∧ 
  (median dept_B = 26) ∧ 
  (average dept_A < average dept_B) :=
sorry

end NUMINAMATH_CALUDE_company_age_distribution_l287_28726


namespace NUMINAMATH_CALUDE_correct_number_of_groups_l287_28775

/-- The number of different groups of 3 marbles Tom can choose -/
def number_of_groups : ℕ := 16

/-- The number of red marbles Tom has -/
def red_marbles : ℕ := 1

/-- The number of blue marbles Tom has -/
def blue_marbles : ℕ := 1

/-- The number of black marbles Tom has -/
def black_marbles : ℕ := 1

/-- The number of white marbles Tom has -/
def white_marbles : ℕ := 4

/-- The total number of marbles Tom has -/
def total_marbles : ℕ := red_marbles + blue_marbles + black_marbles + white_marbles

/-- Theorem stating that the number of different groups of 3 marbles Tom can choose is correct -/
theorem correct_number_of_groups :
  number_of_groups = (Nat.choose white_marbles 3) + (Nat.choose 3 2 * Nat.choose white_marbles 1) :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_groups_l287_28775


namespace NUMINAMATH_CALUDE_count_integers_with_same_remainder_l287_28793

theorem count_integers_with_same_remainder : ∃! (S : Finset ℕ),
  (∀ n ∈ S, 150 < n ∧ n < 250 ∧ ∃ r : ℕ, r ≤ 6 ∧ n % 7 = r ∧ n % 9 = r) ∧
  S.card = 7 := by sorry

end NUMINAMATH_CALUDE_count_integers_with_same_remainder_l287_28793


namespace NUMINAMATH_CALUDE_spectators_with_type_A_l287_28744

/-- Represents the number of spectators who received type A wristbands for both hands -/
def x : ℕ := sorry

/-- Represents the number of spectators who received type B wristbands for both hands -/
def y : ℕ := sorry

/-- The ratio of spectators with type A to type B wristbands is 3:2 -/
axiom ratio_constraint : 2 * x = 3 * y

/-- The total number of wristbands distributed is 460 -/
axiom total_wristbands : 2 * x + 2 * y = 460

theorem spectators_with_type_A : x = 138 := by sorry

end NUMINAMATH_CALUDE_spectators_with_type_A_l287_28744


namespace NUMINAMATH_CALUDE_largest_prime_square_root_l287_28798

theorem largest_prime_square_root (p : ℕ) (a b : ℕ+) (h_prime : Nat.Prime p) 
  (h_eq : (p : ℝ) = (b.val : ℝ) / 2 * Real.sqrt ((a.val - b.val : ℝ) / (a.val + b.val))) :
  p ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_square_root_l287_28798


namespace NUMINAMATH_CALUDE_guido_cost_calculation_l287_28766

def lightning_cost : ℝ := 140000

def mater_cost : ℝ := 0.1 * lightning_cost

def sally_cost_before_mod : ℝ := 3 * mater_cost

def sally_cost_after_mod : ℝ := sally_cost_before_mod * 1.2

def guido_cost : ℝ := sally_cost_after_mod * 0.85

theorem guido_cost_calculation : guido_cost = 42840 := by
  sorry

end NUMINAMATH_CALUDE_guido_cost_calculation_l287_28766


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l287_28781

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ+ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ+, a n > 0) →
  a 3 * a 5 + 2 * a 4 * a 6 + a 5 * a 7 = 81 →
  a 4 + a 6 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l287_28781


namespace NUMINAMATH_CALUDE_change_for_fifty_cents_l287_28776

/-- Represents the number of ways to make change for a given amount in cents -/
def makeChange (amount : ℕ) (maxQuarters : ℕ) : ℕ := sorry

/-- The value of a quarter in cents -/
def quarterValue : ℕ := 25

/-- The value of a nickel in cents -/
def nickelValue : ℕ := 5

/-- The value of a penny in cents -/
def pennyValue : ℕ := 1

theorem change_for_fifty_cents :
  makeChange 50 2 = 18 := by sorry

end NUMINAMATH_CALUDE_change_for_fifty_cents_l287_28776


namespace NUMINAMATH_CALUDE_double_radius_ellipse_iff_l287_28731

/-- An ellipse is a "double-radius ellipse" if there exists a point P on the ellipse
    such that the ratio of the distances from P to the two foci is 2:1 -/
def is_double_radius_ellipse (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≥ b ∧ a ≤ 3 * Real.sqrt (a^2 - b^2)

/-- Theorem: Characterization of a double-radius ellipse -/
theorem double_radius_ellipse_iff (a b : ℝ) :
  is_double_radius_ellipse a b ↔ 
  (a > 0 ∧ b > 0 ∧ a ≥ b) ∧ (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧ 
    ∃ (d1 d2 : ℝ), d1 = 2*d2 ∧ 
      d1 + d2 = 2*a ∧
      d1^2 = (x + Real.sqrt (a^2 - b^2))^2 + y^2 ∧
      d2^2 = (x - Real.sqrt (a^2 - b^2))^2 + y^2) :=
sorry

end NUMINAMATH_CALUDE_double_radius_ellipse_iff_l287_28731


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l287_28732

-- Define the propositions p and q
def p (x : ℝ) : Prop := x / (x - 2) < 0
def q (x m : ℝ) : Prop := 0 < x ∧ x < m

-- Define the set of x that satisfy p
def set_p : Set ℝ := {x | p x}

-- Define the set of x that satisfy q
def set_q (m : ℝ) : Set ℝ := {x | q x m}

-- State the theorem
theorem necessary_not_sufficient_condition (m : ℝ) :
  (∀ x, q x m → p x) ∧ (∃ x, p x ∧ ¬q x m) → m > 2 :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l287_28732


namespace NUMINAMATH_CALUDE_analytical_method_is_effect_to_cause_l287_28763

/-- Represents the possible descriptions of the analytical method -/
inductive AnalyticalMethodDescription
  | causeToEffect
  | effectToCause
  | mutualInference
  | converseProof

/-- Definition of the analytical method -/
structure AnalyticalMethod :=
  (description : AnalyticalMethodDescription)
  (isReasoningMethod : Bool)

/-- Theorem stating that the analytical method is correctly described as reasoning from effect to cause -/
theorem analytical_method_is_effect_to_cause :
  ∀ (am : AnalyticalMethod), 
    am.isReasoningMethod = true → 
    am.description = AnalyticalMethodDescription.effectToCause :=
by
  sorry

end NUMINAMATH_CALUDE_analytical_method_is_effect_to_cause_l287_28763


namespace NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l287_28735

/-- Given two vectors a and b in ℝ², where a is perpendicular to b,
    prove that the magnitude of their difference is √10. -/
theorem perpendicular_vectors_difference_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : a.1 = x ∧ a.2 = 1)
  (h2 : b = (1, -2))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) : 
  ‖(a.1 - b.1, a.2 - b.2)‖ = Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l287_28735


namespace NUMINAMATH_CALUDE_mark_survival_days_l287_28771

/- Define the problem parameters -/
def num_astronauts : ℕ := 6
def food_days_per_astronaut : ℕ := 5
def water_per_astronaut : ℝ := 50
def potato_yield_per_sqm : ℝ := 2.5
def water_required_per_sqm : ℝ := 4
def potato_needed_per_day : ℝ := 1.875

/- Define the theorem -/
theorem mark_survival_days :
  let initial_food_days := num_astronauts * food_days_per_astronaut
  let total_water := num_astronauts * water_per_astronaut
  let irrigated_area := total_water / water_required_per_sqm
  let total_potatoes := irrigated_area * potato_yield_per_sqm
  let potato_days := total_potatoes / potato_needed_per_day
  initial_food_days + potato_days = 130 := by
  sorry


end NUMINAMATH_CALUDE_mark_survival_days_l287_28771


namespace NUMINAMATH_CALUDE_water_bottle_volume_l287_28713

theorem water_bottle_volume (total_cost : ℝ) (num_bottles : ℕ) (price_per_liter : ℝ) 
  (h1 : total_cost = 12)
  (h2 : num_bottles = 6)
  (h3 : price_per_liter = 1) :
  (total_cost / (num_bottles : ℝ)) / price_per_liter = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_volume_l287_28713


namespace NUMINAMATH_CALUDE_university_population_l287_28797

/-- Represents the total number of students at the university -/
def total_students : ℕ := 5000

/-- Represents the sample size -/
def sample_size : ℕ := 500

/-- Represents the number of freshmen in the sample -/
def freshmen_sample : ℕ := 200

/-- Represents the number of sophomores in the sample -/
def sophomore_sample : ℕ := 100

/-- Represents the number of students in other grades -/
def other_grades : ℕ := 2000

/-- Theorem stating that given the sample size, freshmen sample, sophomore sample, 
    and number of students in other grades, the total number of students at the 
    university is 5000 -/
theorem university_population : 
  sample_size = freshmen_sample + sophomore_sample + (other_grades / 10) ∧
  total_students = freshmen_sample * 10 + sophomore_sample * 10 + other_grades :=
sorry

end NUMINAMATH_CALUDE_university_population_l287_28797


namespace NUMINAMATH_CALUDE_quadratic_difference_theorem_l287_28786

theorem quadratic_difference_theorem (a b : ℝ) :
  (∀ x y : ℝ, (a*x^2 + 2*x*y - x) - (3*x^2 - 2*b*x*y + 3*y) = (-x + 3*y)) →
  a^2 - 4*b = 13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_difference_theorem_l287_28786


namespace NUMINAMATH_CALUDE_root_product_of_quartic_l287_28719

theorem root_product_of_quartic (p q r s : ℂ) : 
  (3 * p^4 - 8 * p^3 - 15 * p^2 + 10 * p - 2 = 0) →
  (3 * q^4 - 8 * q^3 - 15 * q^2 + 10 * q - 2 = 0) →
  (3 * r^4 - 8 * r^3 - 15 * r^2 + 10 * r - 2 = 0) →
  (3 * s^4 - 8 * s^3 - 15 * s^2 + 10 * s - 2 = 0) →
  p * q * r * s = 2/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_of_quartic_l287_28719


namespace NUMINAMATH_CALUDE_tony_remaining_money_l287_28758

def initial_money : ℕ := 20
def ticket_cost : ℕ := 8
def hotdog_cost : ℕ := 3

theorem tony_remaining_money :
  initial_money - ticket_cost - hotdog_cost = 9 := by
  sorry

end NUMINAMATH_CALUDE_tony_remaining_money_l287_28758


namespace NUMINAMATH_CALUDE_min_k_for_reciprocal_like_l287_28721

/-- A directed graph representing people liking each other in a group -/
structure LikeGraph where
  n : ℕ  -- number of people
  k : ℕ  -- number of people each person likes
  edges : Fin n → Finset (Fin n)
  outDegree : ∀ v, (edges v).card = k

/-- There exists a pair of people who like each other reciprocally -/
def hasReciprocalLike (g : LikeGraph) : Prop :=
  ∃ i j : Fin g.n, i ≠ j ∧ i ∈ g.edges j ∧ j ∈ g.edges i

/-- The minimum k that guarantees a reciprocal like in a group of 30 people -/
theorem min_k_for_reciprocal_like :
  ∀ k : ℕ, (∀ g : LikeGraph, g.n = 30 ∧ g.k = k → hasReciprocalLike g) ↔ k ≥ 15 :=
sorry

end NUMINAMATH_CALUDE_min_k_for_reciprocal_like_l287_28721


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l287_28760

theorem existence_of_special_sequence :
  ∃ (a : ℕ → ℝ) (x y : ℝ),
    (∀ n, a n ≠ 0) ∧
    (∀ n, a (n + 2) = x * a (n + 1) + y * a n) ∧
    (∀ r > 0, ∃ i j : ℕ, |a i| < r ∧ r < |a j|) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l287_28760


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l287_28746

theorem girls_to_boys_ratio (girls boys : ℕ) : 
  girls = boys + 5 →
  girls + boys = 35 →
  girls * 3 = boys * 4 := by
sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l287_28746


namespace NUMINAMATH_CALUDE_unique_number_existence_l287_28742

theorem unique_number_existence : ∃! N : ℕ, N / 1000 = 220 ∧ N % 1000 = 40 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_existence_l287_28742


namespace NUMINAMATH_CALUDE_three_digit_divisibility_l287_28718

/-- Given a three-digit number ABC divisible by 37, prove that BCA + CAB is also divisible by 37 -/
theorem three_digit_divisibility (A B C : ℕ) (h : 37 ∣ (100 * A + 10 * B + C)) :
  37 ∣ ((100 * B + 10 * C + A) + (100 * C + 10 * A + B)) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_l287_28718


namespace NUMINAMATH_CALUDE_smallest_valid_fourth_number_l287_28784

def is_valid_fourth_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  (6 + 8 + 2 + 4 + 8 + 5 + (n / 10) + (n % 10)) * 4 = 68 + 24 + 85 + n

theorem smallest_valid_fourth_number :
  ∀ m : ℕ, m ≥ 10 ∧ m < 57 → ¬(is_valid_fourth_number m) ∧ is_valid_fourth_number 57 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_fourth_number_l287_28784


namespace NUMINAMATH_CALUDE_milburg_population_l287_28765

/-- The total population of Milburg -/
def total_population (adults children teenagers seniors : ℕ) : ℕ :=
  adults + children + teenagers + seniors

/-- Theorem: The total population of Milburg is 12,292 -/
theorem milburg_population : total_population 5256 2987 1709 2340 = 12292 := by
  sorry

end NUMINAMATH_CALUDE_milburg_population_l287_28765


namespace NUMINAMATH_CALUDE_sum_of_integers_l287_28743

theorem sum_of_integers (a b c d : ℕ+) 
  (eq1 : a * b + c * d = 38)
  (eq2 : a * c + b * d = 34)
  (eq3 : a * d + b * c = 43) :
  a + b + c + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l287_28743


namespace NUMINAMATH_CALUDE_binary_1101_is_13_l287_28740

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101_is_13 : 
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_is_13_l287_28740


namespace NUMINAMATH_CALUDE_evelyn_initial_skittles_l287_28757

/-- The number of Skittles Evelyn shared with Christine -/
def shared_skittles : ℕ := 72

/-- The number of Skittles Evelyn had left after sharing -/
def remaining_skittles : ℕ := 4

/-- The initial number of Skittles Evelyn had -/
def initial_skittles : ℕ := shared_skittles + remaining_skittles

theorem evelyn_initial_skittles : initial_skittles = 76 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_initial_skittles_l287_28757


namespace NUMINAMATH_CALUDE_x_coordinate_is_nineteen_thirds_l287_28712

/-- The x-coordinate of a point on a line --/
def x_coordinate_on_line (x1 y1 x2 y2 y : ℚ) : ℚ :=
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  (y - b) / m

/-- Theorem: The x-coordinate of the point (x, 8) on the line passing through (3, 3) and (1, 0) is 19/3 --/
theorem x_coordinate_is_nineteen_thirds :
  x_coordinate_on_line 3 3 1 0 8 = 19/3 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_is_nineteen_thirds_l287_28712


namespace NUMINAMATH_CALUDE_brenda_erasers_count_l287_28747

/-- The number of groups Brenda creates -/
def num_groups : ℕ := 3

/-- The number of erasers in each group -/
def erasers_per_group : ℕ := 90

/-- The total number of erasers Brenda has -/
def total_erasers : ℕ := num_groups * erasers_per_group

theorem brenda_erasers_count : total_erasers = 270 := by
  sorry

end NUMINAMATH_CALUDE_brenda_erasers_count_l287_28747


namespace NUMINAMATH_CALUDE_claire_cleaning_hours_l287_28708

/-- Calculates the hours spent cleaning given Claire's daily schedule. -/
def hours_cleaning (total_day_hours sleep_hours cooking_hours crafting_hours : ℕ) : ℕ :=
  let working_hours := total_day_hours - sleep_hours
  let cleaning_hours := working_hours - cooking_hours - 2 * crafting_hours
  cleaning_hours

/-- Theorem stating that Claire spends 4 hours cleaning given her schedule. -/
theorem claire_cleaning_hours :
  hours_cleaning 24 8 2 5 = 4 := by
  sorry

#eval hours_cleaning 24 8 2 5

end NUMINAMATH_CALUDE_claire_cleaning_hours_l287_28708


namespace NUMINAMATH_CALUDE_solution_set_of_composite_function_l287_28720

/-- Given a function f(x) = 2x - 1, the solution set of f[f(x)] ≥ 1 is {x | x ≥ 1} -/
theorem solution_set_of_composite_function (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x - 1) :
  {x : ℝ | f (f x) ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_composite_function_l287_28720


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l287_28739

/-- Given a survey of pasta preferences among students, this theorem proves
    that the ratio of students preferring spaghetti to those preferring manicotti is 2. -/
theorem pasta_preference_ratio :
  let total_students : ℕ := 800
  let spaghetti_preference : ℕ := 320
  let manicotti_preference : ℕ := 160
  (spaghetti_preference : ℚ) / manicotti_preference = 2 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l287_28739


namespace NUMINAMATH_CALUDE_fraction_simplification_l287_28703

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) : (a - b) / (b - a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l287_28703


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l287_28770

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  sum_property : a 1 + a 3 = 8
  geometric_mean : (a 4) ^ 2 = (a 2) * (a 9)
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term 
  (seq : ArithmeticSequence) : seq.a 5 = 13 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l287_28770


namespace NUMINAMATH_CALUDE_inequality_theorem_l287_28790

theorem inequality_theorem (x y a : ℝ) (h1 : x < y) (h2 : a < 1) : x + a < y + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l287_28790


namespace NUMINAMATH_CALUDE_unique_phone_number_l287_28701

/-- A six-digit number -/
def SixDigitNumber := { n : ℕ // 100000 ≤ n ∧ n < 1000000 }

/-- The set of divisors we're interested in -/
def Divisors : Finset ℕ := {3, 4, 7, 9, 11, 13}

/-- The property that a number gives the same remainder when divided by all numbers in Divisors -/
def SameRemainder (n : ℕ) : Prop :=
  ∃ r, ∀ d ∈ Divisors, n % d = r

/-- The main theorem -/
theorem unique_phone_number :
  ∃! (T : SixDigitNumber),
    Odd T.val ∧
    (T.val / 100000 = 7) ∧
    ((T.val / 100) % 10 = 2) ∧
    SameRemainder T.val ∧
    T.val = 720721 := by
  sorry

#check unique_phone_number

end NUMINAMATH_CALUDE_unique_phone_number_l287_28701


namespace NUMINAMATH_CALUDE_smallest_packages_for_more_envelopes_l287_28741

theorem smallest_packages_for_more_envelopes (n : ℕ) : 
  (∀ k : ℕ, k < n → 10 * k ≤ 8 * k + 7) ∧ 
  (10 * n > 8 * n + 7) → 
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_packages_for_more_envelopes_l287_28741


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l287_28736

/-- Given a line segment with one endpoint (6, -1) and midpoint (3, 7),
    the sum of the coordinates of the other endpoint is 15. -/
theorem endpoint_coordinate_sum : ∀ (x y : ℝ),
  (6 + x) / 2 = 3 →
  (-1 + y) / 2 = 7 →
  x + y = 15 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l287_28736


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_parallel_l287_28733

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_parallel
  (α β : Plane) (m n : Line)
  (h_distinct_planes : α ≠ β)
  (h_distinct_lines : m ≠ n)
  (h_m_perp_α : perpendicular m α)
  (h_n_perp_β : perpendicular n β)
  (h_α_parallel_β : parallel α β) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_parallel_l287_28733


namespace NUMINAMATH_CALUDE_handshake_theorem_l287_28754

/-- Represents a gathering of people where each person shakes hands with a fixed number of others. -/
structure Gathering where
  num_people : ℕ
  handshakes_per_person : ℕ

/-- Calculates the total number of handshakes in a gathering. -/
def total_handshakes (g : Gathering) : ℕ :=
  g.num_people * g.handshakes_per_person / 2

/-- Theorem stating that in a gathering of 30 people where each person shakes hands with 3 others,
    the total number of handshakes is 45. -/
theorem handshake_theorem (g : Gathering) (h1 : g.num_people = 30) (h2 : g.handshakes_per_person = 3) :
  total_handshakes g = 45 := by
  sorry

#eval total_handshakes ⟨30, 3⟩

end NUMINAMATH_CALUDE_handshake_theorem_l287_28754


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l287_28714

theorem trigonometric_equation_solution (x : ℝ) : 
  (∃ k : ℤ, x = 2 * π / 9 + 2 * π / 3 * k ∨ x = -2 * π / 9 + 2 * π / 3 * k) ↔ 
  Real.cos (3 * x - π / 6) - Real.sin (3 * x - π / 6) * Real.tan (π / 6) = 1 / (2 * Real.cos (7 * π / 6)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l287_28714


namespace NUMINAMATH_CALUDE_printing_press_completion_time_l287_28773

-- Define the start time (9:00 AM)
def start_time : ℕ := 9

-- Define the time when half the order is completed (12:00 PM)
def half_time : ℕ := 12

-- Define the time to complete half the order
def half_duration : ℕ := half_time - start_time

-- Theorem: The printing press will finish the entire order at 3:00 PM
theorem printing_press_completion_time :
  start_time + 2 * half_duration = 15 := by
  sorry

end NUMINAMATH_CALUDE_printing_press_completion_time_l287_28773


namespace NUMINAMATH_CALUDE_constant_function_proof_l287_28769

theorem constant_function_proof (f : ℚ → ℚ) 
  (h1 : ∀ x y : ℚ, f (x + y) = f x + f y + 2547)
  (h2 : f 2004 = 2547) : 
  f 2547 = 2547 := by sorry

end NUMINAMATH_CALUDE_constant_function_proof_l287_28769


namespace NUMINAMATH_CALUDE_inequality_proof_l287_28762

theorem inequality_proof (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1/3) * (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l287_28762


namespace NUMINAMATH_CALUDE_total_tickets_is_340_l287_28774

/-- Represents the number of tickets sold for a theater performance. -/
structure TicketSales where
  orchestra : ℕ
  balcony : ℕ

/-- The total revenue from ticket sales. -/
def totalRevenue (sales : TicketSales) : ℕ :=
  12 * sales.orchestra + 8 * sales.balcony

/-- The difference between balcony and orchestra ticket sales. -/
def balconyOrchestraDiff (sales : TicketSales) : ℤ :=
  sales.balcony - sales.orchestra

/-- The total number of tickets sold. -/
def totalTickets (sales : TicketSales) : ℕ :=
  sales.orchestra + sales.balcony

/-- Theorem stating that given the conditions, the total number of tickets sold is 340. -/
theorem total_tickets_is_340 :
  ∃ (sales : TicketSales),
    totalRevenue sales = 3320 ∧
    balconyOrchestraDiff sales = 40 ∧
    totalTickets sales = 340 := by
  sorry


end NUMINAMATH_CALUDE_total_tickets_is_340_l287_28774


namespace NUMINAMATH_CALUDE_expression_value_l287_28785

theorem expression_value (a b : ℝ) (h : a + b = 1) :
  a^3 + b^3 + 3*(a^3*b + a*b^3) + 6*(a^3*b^2 + a^2*b^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l287_28785


namespace NUMINAMATH_CALUDE_f_sum_equals_four_l287_28745

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_sum_equals_four (f : ℝ → ℝ) 
  (h_even : isEven f)
  (h_period : ∀ x, f (x + 4) = f x + f 2)
  (h_f_one : f 1 = 4) :
  f 3 + f 10 = 4 := by sorry

end NUMINAMATH_CALUDE_f_sum_equals_four_l287_28745


namespace NUMINAMATH_CALUDE_skew_lines_distance_in_isosceles_triangle_sphere_setup_l287_28761

/-- Given an isosceles triangle ABC on plane P with two skew lines passing through A and C,
    tangent to a sphere touching P at B, prove the distance between the lines. -/
theorem skew_lines_distance_in_isosceles_triangle_sphere_setup
  (l a r α : ℝ)
  (hl : l > 0)
  (ha : a > 0)
  (hr : r > 0)
  (hα : 0 < α ∧ α < π / 2)
  (h_isosceles : 2 * a ≤ l) :
  ∃ x : ℝ, x = (2 * a * Real.tan α * Real.sqrt (2 * r * l * Real.sin α - (l^2 + r^2) * Real.sin α^2)) /
              Real.sqrt (l^2 - a^2 * Real.cos α^2) :=
by sorry

end NUMINAMATH_CALUDE_skew_lines_distance_in_isosceles_triangle_sphere_setup_l287_28761


namespace NUMINAMATH_CALUDE_tv_count_indeterminate_l287_28755

structure GroupInfo where
  total : ℕ
  married : ℕ
  radio : ℕ
  ac : ℕ
  tv_radio_ac_married : ℕ

def has_tv (info : GroupInfo) : Set ℕ :=
  { n | n ≥ info.tv_radio_ac_married ∧ n ≤ info.total }

theorem tv_count_indeterminate (info : GroupInfo) 
  (h_total : info.total = 100)
  (h_married : info.married = 81)
  (h_radio : info.radio = 85)
  (h_ac : info.ac = 70)
  (h_tram : info.tv_radio_ac_married = 11) :
  ∃ (n : ℕ), n ∈ has_tv info ∧ 
  ∀ (m : ℕ), m ≠ n → (m ∈ has_tv info ↔ n ∈ has_tv info) :=
sorry

end NUMINAMATH_CALUDE_tv_count_indeterminate_l287_28755


namespace NUMINAMATH_CALUDE_average_of_xyz_l287_28707

theorem average_of_xyz (x y z : ℝ) (h : (5 / 2) * (x + y + z) = 20) :
  (x + y + z) / 3 = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_xyz_l287_28707


namespace NUMINAMATH_CALUDE_arithmetic_sequence_probability_l287_28711

/-- The set of numbers from which we select -/
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 20}

/-- Predicate to check if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop := 2 * b = a + c

/-- The number of ways to select 3 different numbers from S -/
def totalSelections : ℕ := Nat.choose 20 3

/-- The number of ways to select 3 different numbers from S that form an arithmetic sequence -/
def validSelections : ℕ := 90

/-- The probability of selecting 3 different numbers from S that form an arithmetic sequence -/
def probability : ℚ := validSelections / totalSelections

theorem arithmetic_sequence_probability : probability = 1 / 38 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_probability_l287_28711


namespace NUMINAMATH_CALUDE_equation_solution_l287_28788

theorem equation_solution : 
  ∃! x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ∧ x = -48/23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l287_28788


namespace NUMINAMATH_CALUDE_shopping_mall_investment_strategy_l287_28724

/-- Profit when selling at the beginning of the month -/
def profit_beginning (x : ℝ) : ℝ := x * (1 + 0.15) * (1 + 0.10) - x

/-- Profit when selling at the end of the month -/
def profit_end (x : ℝ) : ℝ := x * (1 + 0.30) - x - 700

theorem shopping_mall_investment_strategy :
  (profit_beginning 15000 > profit_end 15000) ∧
  (profit_end 30000 > profit_beginning 30000) ∧
  (∀ x y : ℝ, profit_beginning x = 6000 ∧ profit_end y = 6000 → y < x) ∧
  (∀ x y : ℝ, profit_beginning x = 5300 ∧ profit_end y = 5300 → y < x) :=
sorry

end NUMINAMATH_CALUDE_shopping_mall_investment_strategy_l287_28724


namespace NUMINAMATH_CALUDE_C₂_function_l287_28705

-- Define the original function f
variable (f : ℝ → ℝ)

-- Define C as the graph of y = f(x)
def C (f : ℝ → ℝ) : Set (ℝ × ℝ) := {(x, y) | y = f x}

-- Define C₁ as symmetric to C with respect to x = 1
def C₁ (f : ℝ → ℝ) : Set (ℝ × ℝ) := {(x, y) | y = f (2 - x)}

-- Define C₂ as C₁ shifted one unit to the left
def C₂ (f : ℝ → ℝ) : Set (ℝ × ℝ) := {(x, y) | ∃ x', x = x' - 1 ∧ (x', y) ∈ C₁ f}

-- Theorem: The function corresponding to C₂ is y = f(1 - x)
theorem C₂_function (f : ℝ → ℝ) : C₂ f = {(x, y) | y = f (1 - x)} := by sorry

end NUMINAMATH_CALUDE_C₂_function_l287_28705


namespace NUMINAMATH_CALUDE_rubber_band_area_l287_28753

/-- Represents a nail on the board -/
structure Nail where
  x : ℝ
  y : ℝ

/-- Represents the quadrilateral formed by the rubber band -/
structure Quadrilateral where
  nails : Fin 4 → Nail

/-- The area of a quadrilateral formed by a rubber band looped around four nails arranged in a 2x2 grid with 1 unit spacing -/
def quadrilateralArea (q : Quadrilateral) : ℝ :=
  sorry

/-- The theorem stating that the area of the quadrilateral is 6 square units -/
theorem rubber_band_area (q : Quadrilateral) 
  (h1 : q.nails 0 = ⟨0, 0⟩)
  (h2 : q.nails 1 = ⟨1, 0⟩)
  (h3 : q.nails 2 = ⟨0, 1⟩)
  (h4 : q.nails 3 = ⟨1, 1⟩) :
  quadrilateralArea q = 6 :=
sorry

end NUMINAMATH_CALUDE_rubber_band_area_l287_28753


namespace NUMINAMATH_CALUDE_new_average_age_after_move_l287_28716

theorem new_average_age_after_move (room_a_initial_count : ℕ)
                                   (room_a_initial_avg : ℚ)
                                   (room_b_initial_count : ℕ)
                                   (room_b_initial_avg : ℚ)
                                   (moving_person_age : ℕ) :
  room_a_initial_count = 8 →
  room_a_initial_avg = 35 →
  room_b_initial_count = 5 →
  room_b_initial_avg = 30 →
  moving_person_age = 40 →
  let total_initial_a := room_a_initial_count * room_a_initial_avg
  let total_initial_b := room_b_initial_count * room_b_initial_avg
  let new_total_a := total_initial_a - moving_person_age
  let new_total_b := total_initial_b + moving_person_age
  let new_count_a := room_a_initial_count - 1
  let new_count_b := room_b_initial_count + 1
  let total_new_age := new_total_a + new_total_b
  let total_new_count := new_count_a + new_count_b
  (total_new_age / total_new_count : ℚ) = 33.08 := by
sorry

end NUMINAMATH_CALUDE_new_average_age_after_move_l287_28716


namespace NUMINAMATH_CALUDE_no_triangle_pairs_l287_28715

/-- Given a set of n different elements, prove that if 4m ≤ n², 
    then there exists a set of m non-ordered pairs that do not form any triangles. -/
theorem no_triangle_pairs (n m : ℕ) (h : 4 * m ≤ n ^ 2) :
  ∃ (S : Finset (Fin n)) (P : Finset (Fin n × Fin n)),
    S.card = n ∧ 
    P.card = m ∧
    (∀ (p : Fin n × Fin n), p ∈ P → p.1 ≠ p.2) ∧
    (∀ (a b c : Fin n × Fin n), a ∈ P → b ∈ P → c ∈ P → 
      ¬(a.1 = b.1 ∧ b.2 = c.1 ∧ c.2 = a.2)) :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_pairs_l287_28715


namespace NUMINAMATH_CALUDE_kelly_apples_l287_28704

/-- The number of apples Kelly has altogether, given her initial apples and the apples she picked. -/
def total_apples (initial : Float) (picked : Float) : Float :=
  initial + picked

/-- Theorem stating that Kelly has 161.0 apples altogether. -/
theorem kelly_apples : total_apples 56.0 105.0 = 161.0 := by
  sorry

end NUMINAMATH_CALUDE_kelly_apples_l287_28704


namespace NUMINAMATH_CALUDE_hyperbola_points_m_range_l287_28756

/-- Given points A(-1, y₁) and B(2, y₂) on the hyperbola y = (3+m)/x with y₁ > y₂, 
    the range of values for m is m < -3 -/
theorem hyperbola_points_m_range (m : ℝ) (y₁ y₂ : ℝ) : 
  y₁ = (3 + m) / (-1) → 
  y₂ = (3 + m) / 2 → 
  y₁ > y₂ → 
  m < -3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_points_m_range_l287_28756


namespace NUMINAMATH_CALUDE_even_and_mono_decreasing_implies_ordering_l287_28723

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of being monotonically decreasing on an interval
def IsMonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- State the theorem
theorem even_and_mono_decreasing_implies_ordering (heven : IsEven f) 
    (hmono : IsMonoDecreasing (fun x ↦ f (x - 2)) 0 2) :
  f 2 < f (-1) ∧ f (-1) < f 0 :=
by sorry

end NUMINAMATH_CALUDE_even_and_mono_decreasing_implies_ordering_l287_28723


namespace NUMINAMATH_CALUDE_seedling_probability_value_l287_28750

/-- The germination rate of seeds in a batch -/
def germination_rate : ℝ := 0.9

/-- The survival rate of seedlings after germination -/
def survival_rate : ℝ := 0.8

/-- The probability that a randomly selected seed will grow into a seedling -/
def seedling_probability : ℝ := germination_rate * survival_rate

/-- Theorem stating that the probability of a randomly selected seed growing into a seedling is 0.72 -/
theorem seedling_probability_value : seedling_probability = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_seedling_probability_value_l287_28750


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l287_28738

theorem quadratic_roots_condition (c : ℚ) : 
  (∀ x : ℚ, x^2 - 7*x + c = 0 ↔ ∃ s : ℤ, s^2 = 9*c ∧ x = (7 + s) / 2 ∨ x = (7 - s) / 2) →
  c = 49 / 13 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l287_28738


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_range_l287_28772

theorem quadratic_roots_difference_range (a b c x₁ x₂ : ℝ) :
  a > b →
  b > c →
  a + b + c = 0 →
  a * x₁^2 + 2 * b * x₁ + c = 0 →
  a * x₂^2 + 2 * b * x₂ + c = 0 →
  Real.sqrt 3 < |x₁ - x₂| ∧ |x₁ - x₂| < 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_range_l287_28772


namespace NUMINAMATH_CALUDE_right_triangle_leg_divisible_by_three_l287_28792

theorem right_triangle_leg_divisible_by_three (a b c : ℕ) (h : a * a + b * b = c * c) :
  3 ∣ a ∨ 3 ∣ b :=
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_divisible_by_three_l287_28792


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_sum_l287_28796

/-- A quadrilateral with vertices at (1,2), (4,6), (5,4), and (2,0) has a perimeter that can be
    expressed as a√2 + b√5 + c√10 where a, b, and c are integers, and their sum is 2. -/
theorem quadrilateral_perimeter_sum (a b c : ℤ) : 
  let v1 : ℝ × ℝ := (1, 2)
  let v2 : ℝ × ℝ := (4, 6)
  let v3 : ℝ × ℝ := (5, 4)
  let v4 : ℝ × ℝ := (2, 0)
  let perimeter := dist v1 v2 + dist v2 v3 + dist v3 v4 + dist v4 v1
  perimeter = a * Real.sqrt 2 + b * Real.sqrt 5 + c * Real.sqrt 10 →
  a + b + c = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_sum_l287_28796


namespace NUMINAMATH_CALUDE_small_circle_radius_l287_28768

/-- Given a large circle of radius 10 meters containing seven congruent smaller circles
    arranged with six forming a hexagon around one central circle, prove that the radius
    of each smaller circle is 5 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) :
  R = 10 ∧  -- Radius of the large circle
  (2 * r + 2 * r = 2 * R) →  -- Diameter of large circle equals two radii plus one diameter of small circles
  r = 5 :=
by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l287_28768


namespace NUMINAMATH_CALUDE_equation_exists_l287_28749

theorem equation_exists : ∃ (a b c d e f g h i : ℕ),
  (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ (d < 10) ∧ 
  (e < 10) ∧ (f < 10) ∧ (g < 10) ∧ (h < 10) ∧ (i < 10) ∧
  (a + 100 * b + 10 * c + d = 10 * e + f + 100 * g + 10 * h + i) ∧
  (b = d) ∧ (g = h) ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧ (a ≠ i) ∧
  (b ≠ c) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧ (b ≠ i) ∧
  (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧ (c ≠ i) ∧
  (e ≠ f) ∧ (e ≠ g) ∧ (e ≠ i) ∧
  (f ≠ g) ∧ (f ≠ i) ∧
  (g ≠ i) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_exists_l287_28749


namespace NUMINAMATH_CALUDE_parabola_directrix_l287_28783

/-- Given a parabola with equation y = 2x^2, its directrix equation is y = -1/8 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = 2 * x^2) → (∃ (k : ℝ), k = -1/8 ∧ k = y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l287_28783


namespace NUMINAMATH_CALUDE_tan_pi_36_is_root_l287_28791

theorem tan_pi_36_is_root : 
  let f (x : ℝ) := x^3 - 3 * Real.tan (π/12) * x^2 - 3 * x + Real.tan (π/12)
  f (Real.tan (π/36)) = 0 := by sorry

end NUMINAMATH_CALUDE_tan_pi_36_is_root_l287_28791


namespace NUMINAMATH_CALUDE_skyscraper_anniversary_l287_28777

theorem skyscraper_anniversary (current_year : ℕ) : 
  let years_since_built := 100
  let years_to_event := 95
  let event_year := current_year + years_to_event
  let years_at_event := years_since_built + years_to_event
  ∃ (anniversary : ℕ), anniversary > years_at_event ∧ anniversary - years_at_event = 5 :=
by sorry

end NUMINAMATH_CALUDE_skyscraper_anniversary_l287_28777


namespace NUMINAMATH_CALUDE_net_calorie_intake_l287_28764

/-- Calculate net calorie intake after jogging -/
theorem net_calorie_intake
  (breakfast_calories : ℕ)
  (jogging_time : ℕ)
  (calorie_burn_rate : ℕ)
  (h1 : breakfast_calories = 900)
  (h2 : jogging_time = 30)
  (h3 : calorie_burn_rate = 10) :
  breakfast_calories - jogging_time * calorie_burn_rate = 600 :=
by sorry

end NUMINAMATH_CALUDE_net_calorie_intake_l287_28764


namespace NUMINAMATH_CALUDE_reaction_gibbs_free_energy_change_l287_28759

/-- The standard Gibbs free energy of formation of NaOH in kJ/mol -/
def ΔG_f_NaOH : ℝ := -381.1

/-- The standard Gibbs free energy of formation of Na₂O in kJ/mol -/
def ΔG_f_Na2O : ℝ := -378

/-- The standard Gibbs free energy of formation of H₂O (liquid) in kJ/mol -/
def ΔG_f_H2O : ℝ := -237

/-- The temperature in Kelvin -/
def T : ℝ := 298

/-- 
The standard Gibbs free energy change (ΔG°₂₉₈) for the reaction Na₂O + H₂O → 2NaOH at 298 K
is equal to -147.2 kJ/mol, given the standard Gibbs free energies of formation for NaOH, Na₂O, and H₂O.
-/
theorem reaction_gibbs_free_energy_change : 
  2 * ΔG_f_NaOH - (ΔG_f_Na2O + ΔG_f_H2O) = -147.2 := by sorry

end NUMINAMATH_CALUDE_reaction_gibbs_free_energy_change_l287_28759


namespace NUMINAMATH_CALUDE_emily_seventy_percent_at_S_l287_28778

/-- Represents a point on the path -/
inductive Point
  | P | Q | R | S | T | U

/-- The distance between any two adjacent points -/
def unit_distance : ℝ := 1

/-- The total distance of the round trip -/
def total_distance : ℝ := 10 * unit_distance

/-- The distance from P to any given point -/
def distance_from_P (p : Point) : ℝ :=
  match p with
  | Point.P => 0
  | Point.Q => unit_distance
  | Point.R => 2 * unit_distance
  | Point.S => 3 * unit_distance
  | Point.T => 4 * unit_distance
  | Point.U => 5 * unit_distance

/-- The distance Emily has walked when she reaches a point on her return trip -/
def distance_walked (p : Point) : ℝ :=
  total_distance - distance_from_P p

theorem emily_seventy_percent_at_S :
  distance_walked Point.S = 0.7 * total_distance :=
sorry


end NUMINAMATH_CALUDE_emily_seventy_percent_at_S_l287_28778


namespace NUMINAMATH_CALUDE_intersection_condition_l287_28710

-- Define the parabola C: y^2 = x
def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the line l: y = kx + 1
def line (k x y : ℝ) : Prop := y = k * x + 1

-- Define the condition for two distinct intersection points
def has_two_distinct_intersections (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    parabola x₁ y₁ ∧ line k x₁ y₁ ∧
    parabola x₂ y₂ ∧ line k x₂ y₂

-- Theorem statement
theorem intersection_condition :
  (∀ k : ℝ, has_two_distinct_intersections k → k ≠ 0) ∧
  (∃ k : ℝ, k ≠ 0 ∧ ¬has_two_distinct_intersections k) :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l287_28710


namespace NUMINAMATH_CALUDE_geometric_sequence_S5_l287_28700

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_S5 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  (a 3 + a 4) / (a 1 + a 2) = 4 →
  ∃ S5 : ℝ, (S5 = 31 ∨ S5 = 11) ∧ S5 = (a 1 + a 2 + a 3 + a 4 + a 5) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_S5_l287_28700


namespace NUMINAMATH_CALUDE_trapezoid_circumradii_relation_l287_28729

-- Define a trapezoid
structure Trapezoid :=
  (A₁ A₂ A₃ A₄ : ℝ × ℝ)

-- Define the diagonal lengths
def diagonal₁ (t : Trapezoid) : ℝ := sorry
def diagonal₂ (t : Trapezoid) : ℝ := sorry

-- Define the circumradius of a triangle formed by three points of the trapezoid
def circumradius (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem trapezoid_circumradii_relation (t : Trapezoid) :
  let e := diagonal₁ t
  let f := diagonal₂ t
  let r₁ := circumradius t.A₂ t.A₃ t.A₄
  let r₂ := circumradius t.A₁ t.A₃ t.A₄
  let r₃ := circumradius t.A₁ t.A₂ t.A₄
  let r₄ := circumradius t.A₁ t.A₂ t.A₃
  (r₂ + r₄) / e = (r₁ + r₃) / f := by sorry

end NUMINAMATH_CALUDE_trapezoid_circumradii_relation_l287_28729


namespace NUMINAMATH_CALUDE_min_fraction_sum_l287_28709

def is_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

def are_distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem min_fraction_sum (A B C D : ℕ) 
  (h1 : is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D) 
  (h2 : are_distinct A B C D) :
  (A : ℚ) / B + (C : ℚ) / D ≥ 31 / 56 := by
  sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l287_28709


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l287_28737

/-- An isosceles trapezoid with given side length and base lengths has area 52 -/
theorem isosceles_trapezoid_area : 
  ∀ (side_length base1 base2 : ℝ),
  side_length = 5 →
  base1 = 10 →
  base2 = 16 →
  (1/2 : ℝ) * (base1 + base2) * Real.sqrt (side_length^2 - ((base2 - base1)/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l287_28737


namespace NUMINAMATH_CALUDE_sum_of_integers_l287_28789

theorem sum_of_integers (m n p q : ℕ+) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (6 - m) * (6 - n) * (6 - p) * (6 - q) = 4 →
  m + n + p + q = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l287_28789


namespace NUMINAMATH_CALUDE_frequency_of_score_range_l287_28727

/-- The frequency of students scoring in a certain range -/
def frequency (total : ℕ) (in_range : ℕ) : ℚ :=
  in_range / total

/-- Theorem: The frequency of students scoring between 80 and 100 points is 0.36 -/
theorem frequency_of_score_range : 
  frequency 500 180 = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_frequency_of_score_range_l287_28727


namespace NUMINAMATH_CALUDE_marble_difference_l287_28751

/-- The number of marbles each person has -/
structure Marbles where
  laurie : ℕ
  kurt : ℕ
  dennis : ℕ

/-- Given conditions about the marbles -/
def marble_conditions (m : Marbles) : Prop :=
  m.laurie = 37 ∧ m.laurie = m.kurt + 12 ∧ m.dennis = 70

/-- Theorem stating the difference between Dennis's and Kurt's marbles -/
theorem marble_difference (m : Marbles) (h : marble_conditions m) :
  m.dennis - m.kurt = 45 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l287_28751

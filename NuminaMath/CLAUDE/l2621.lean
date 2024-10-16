import Mathlib

namespace NUMINAMATH_CALUDE_algebraic_expression_evaluation_l2621_262199

theorem algebraic_expression_evaluation :
  let x : ℝ := 2 - Real.sqrt 3
  (7 + 4 * Real.sqrt 3) * x^2 - (2 + Real.sqrt 3) * x + Real.sqrt 3 = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_evaluation_l2621_262199


namespace NUMINAMATH_CALUDE_problem_solution_l2621_262139

theorem problem_solution (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 384) 
  (h2 : 3*m*n + 2*n^2 = 560) : 
  2*m^2 + 13*m*n + 6*n^2 - 444 = 2004 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2621_262139


namespace NUMINAMATH_CALUDE_total_trees_is_433_l2621_262182

/-- The total number of trees in the park after planting -/
def total_trees_after_planting (current_walnut current_oak current_maple new_walnut new_oak new_maple : ℕ) : ℕ :=
  current_walnut + current_oak + current_maple + new_walnut + new_oak + new_maple

/-- Theorem: The total number of trees after planting is 433 -/
theorem total_trees_is_433 :
  total_trees_after_planting 107 65 32 104 79 46 = 433 := by
  sorry


end NUMINAMATH_CALUDE_total_trees_is_433_l2621_262182


namespace NUMINAMATH_CALUDE_bead_removal_proof_l2621_262190

theorem bead_removal_proof (total_beads : ℕ) (parts : ℕ) (final_beads : ℕ) (x : ℕ) : 
  total_beads = 39 →
  parts = 3 →
  final_beads = 6 →
  2 * ((total_beads / parts) - x) = final_beads →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_bead_removal_proof_l2621_262190


namespace NUMINAMATH_CALUDE_find_p_value_l2621_262168

theorem find_p_value (p q r : ℂ) (h_p_real : p.im = 0) 
  (h_sum : p + q + r = 5)
  (h_sum_prod : p * q + q * r + r * p = 5)
  (h_prod : p * q * r = 5) : 
  p = 4 := by sorry

end NUMINAMATH_CALUDE_find_p_value_l2621_262168


namespace NUMINAMATH_CALUDE_wall_bricks_l2621_262107

/-- The number of bricks in the wall -/
def num_bricks : ℕ := 720

/-- The time it takes Brenda to build the wall alone (in hours) -/
def brenda_time : ℕ := 12

/-- The time it takes Brandon to build the wall alone (in hours) -/
def brandon_time : ℕ := 15

/-- The decrease in combined output when working together (in bricks per hour) -/
def output_decrease : ℕ := 12

/-- The time it takes Brenda and Brandon to build the wall together (in hours) -/
def combined_time : ℕ := 6

/-- Theorem stating that the number of bricks in the wall is 720 -/
theorem wall_bricks : 
  (combined_time : ℚ) * ((num_bricks / brenda_time : ℚ) + (num_bricks / brandon_time : ℚ) - output_decrease) = num_bricks := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_l2621_262107


namespace NUMINAMATH_CALUDE_trapezoid_intersection_l2621_262183

/-- Given a parabola y = x^2 and k > 0, prove that for any trapezoid inscribed in the parabola
    with bases parallel to the x-axis and the product of base lengths equal to k,
    the lateral sides of the trapezoid intersect at the point (0, -k/4). -/
theorem trapezoid_intersection (k : ℝ) (h : k > 0) :
  ∀ (a b : ℝ), 4 * a * b = k →
  ∃ (x y : ℝ), x = 0 ∧ y = -k / 4 ∧
  (∀ (t : ℝ), (t - a) * (b^2 - a^2) = (b - a) * (t^2 - a^2) ↔
               (t + a) * (b^2 - a^2) = (b - a) * (t^2 - a^2)) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_intersection_l2621_262183


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2621_262173

theorem complex_expression_simplification :
  (-5 + 3 * Complex.I) - (2 - 7 * Complex.I) * 3 = -11 + 24 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2621_262173


namespace NUMINAMATH_CALUDE_pet_store_puppies_l2621_262169

theorem pet_store_puppies (sold : ℕ) (num_cages : ℕ) (puppies_per_cage : ℕ) : 
  sold = 21 → num_cages = 9 → puppies_per_cage = 9 → 
  sold + (num_cages * puppies_per_cage) = 102 := by
sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l2621_262169


namespace NUMINAMATH_CALUDE_dividend_calculation_l2621_262192

/-- Proves that given a divisor of -4 2/3, a quotient of -57 1/5, and a remainder of 2 1/9, the dividend is equal to 269 2/45. -/
theorem dividend_calculation (divisor quotient remainder dividend : ℚ) : 
  divisor = -14/3 →
  quotient = -286/5 →
  remainder = 19/9 →
  dividend = divisor * quotient + remainder →
  dividend = 12107/45 :=
by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2621_262192


namespace NUMINAMATH_CALUDE_cone_volume_over_pi_l2621_262178

/-- Given a cone formed from a 240-degree sector of a circle with radius 24,
    the volume of the cone divided by π is equal to 2048√5/3 -/
theorem cone_volume_over_pi (r : ℝ) (θ : ℝ) :
  r = 24 →
  θ = 240 * π / 180 →
  let base_radius := r * θ / (2 * π)
  let height := Real.sqrt (r^2 - base_radius^2)
  let volume := (1/3) * π * base_radius^2 * height
  volume / π = 2048 * Real.sqrt 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_volume_over_pi_l2621_262178


namespace NUMINAMATH_CALUDE_sin_cos_function_at_pi_12_l2621_262124

theorem sin_cos_function_at_pi_12 :
  let f : ℝ → ℝ := λ x ↦ Real.sin x ^ 4 - Real.cos x ^ 4
  f (π / 12) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_function_at_pi_12_l2621_262124


namespace NUMINAMATH_CALUDE_magnitude_of_z_to_fourth_l2621_262191

-- Define the complex number
def z : ℂ := 4 - 3 * Complex.I

-- State the theorem
theorem magnitude_of_z_to_fourth : Complex.abs (z^4) = 625 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_to_fourth_l2621_262191


namespace NUMINAMATH_CALUDE_negation_equivalence_l2621_262177

theorem negation_equivalence (p q : Prop) : 
  let m := p ∧ q
  (¬p ∨ ¬q) ↔ ¬m := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2621_262177


namespace NUMINAMATH_CALUDE_camel_division_theorem_l2621_262140

/-- A representation of the "camel" figure --/
structure CamelFigure where
  area : ℕ
  has_spaced_cells : Bool

/-- Represents a division of the figure --/
inductive Division
  | GridLines
  | Arbitrary

/-- Represents the result of attempting to form a square --/
inductive SquareFormation
  | Possible
  | Impossible

/-- Function to determine if a square can be formed from the division --/
def can_form_square (figure : CamelFigure) (division : Division) : SquareFormation :=
  match division with
  | Division.GridLines => 
      if figure.has_spaced_cells then SquareFormation.Impossible else SquareFormation.Possible
  | Division.Arbitrary => 
      if figure.area == 25 then SquareFormation.Possible else SquareFormation.Impossible

/-- The main theorem about the camel figure --/
theorem camel_division_theorem (camel : CamelFigure) 
    (h1 : camel.area = 25) 
    (h2 : camel.has_spaced_cells = true) : 
    (can_form_square camel Division.GridLines = SquareFormation.Impossible) ∧ 
    (can_form_square camel Division.Arbitrary = SquareFormation.Possible) := by
  sorry

end NUMINAMATH_CALUDE_camel_division_theorem_l2621_262140


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2621_262130

theorem complex_fraction_equality (a : ℂ) :
  (1 + a * Complex.I) / (2 + Complex.I) = 1 + 2 * Complex.I →
  a = 5 + Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2621_262130


namespace NUMINAMATH_CALUDE_compute_expression_l2621_262150

theorem compute_expression : 7^2 + 4*5 - 2^3 = 61 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2621_262150


namespace NUMINAMATH_CALUDE_root_equality_implies_c_equals_two_l2621_262157

theorem root_equality_implies_c_equals_two :
  ∀ (a b c d : ℕ),
    a > 1 → b > 1 → c > 1 → d > 1 →
    (∀ (M : ℝ), M ≠ 1 →
      (M^(1/a + 1/(a*b) + 1/(a*b*c) + 1/(a*b*c*d)) = M^(37/48))) →
    c = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_equality_implies_c_equals_two_l2621_262157


namespace NUMINAMATH_CALUDE_xiao_hong_math_probability_expected_value_X_xiao_hong_more_likely_math_noon_l2621_262128

-- Define the students
inductive Student : Type
| XiaoHong : Student
| XiaoMing : Student

-- Define the subjects
inductive Subject : Type
| Math : Subject
| Physics : Subject

-- Define the time of day
inductive TimeOfDay : Type
| Noon : TimeOfDay
| Evening : TimeOfDay

-- Define the choice of subjects for a day
structure DailyChoice :=
  (noon : Subject)
  (evening : Subject)

-- Define the probabilities for each student's choices
def choice_probability (s : Student) (dc : DailyChoice) : ℚ :=
  match s, dc with
  | Student.XiaoHong, ⟨Subject.Math, Subject.Math⟩ => 1/4
  | Student.XiaoHong, ⟨Subject.Math, Subject.Physics⟩ => 1/5
  | Student.XiaoHong, ⟨Subject.Physics, Subject.Math⟩ => 7/20
  | Student.XiaoHong, ⟨Subject.Physics, Subject.Physics⟩ => 1/10
  | Student.XiaoMing, ⟨Subject.Math, Subject.Math⟩ => 1/5
  | Student.XiaoMing, ⟨Subject.Math, Subject.Physics⟩ => 1/4
  | Student.XiaoMing, ⟨Subject.Physics, Subject.Math⟩ => 3/20
  | Student.XiaoMing, ⟨Subject.Physics, Subject.Physics⟩ => 3/10

-- Define the number of subjects chosen in a day
def subjects_chosen (s : Student) (dc : DailyChoice) : ℕ :=
  match dc with
  | ⟨Subject.Math, Subject.Math⟩ => 2
  | ⟨Subject.Math, Subject.Physics⟩ => 2
  | ⟨Subject.Physics, Subject.Math⟩ => 2
  | ⟨Subject.Physics, Subject.Physics⟩ => 2

-- Theorem 1: Probability of Xiao Hong choosing math for both noon and evening for exactly 3 out of 5 days
theorem xiao_hong_math_probability : 
  (Finset.sum (Finset.range 6) (λ k => if k = 3 then Nat.choose 5 k * (1/4)^k * (3/4)^(5-k) else 0)) = 45/512 :=
sorry

-- Theorem 2: Expected value of X
theorem expected_value_X :
  (1/100 * 0 + 33/200 * 1 + 33/40 * 2) = 363/200 :=
sorry

-- Theorem 3: Xiao Hong is more likely to choose math at noon when doing physics in the evening
theorem xiao_hong_more_likely_math_noon :
  (choice_probability Student.XiaoHong ⟨Subject.Math, Subject.Physics⟩) / 
  (choice_probability Student.XiaoHong ⟨Subject.Physics, Subject.Physics⟩ + 
   choice_probability Student.XiaoHong ⟨Subject.Math, Subject.Physics⟩) >
  (choice_probability Student.XiaoMing ⟨Subject.Math, Subject.Physics⟩) / 
  (choice_probability Student.XiaoMing ⟨Subject.Physics, Subject.Physics⟩ + 
   choice_probability Student.XiaoMing ⟨Subject.Math, Subject.Physics⟩) :=
sorry

end NUMINAMATH_CALUDE_xiao_hong_math_probability_expected_value_X_xiao_hong_more_likely_math_noon_l2621_262128


namespace NUMINAMATH_CALUDE_computer_price_increase_l2621_262189

theorem computer_price_increase (d : ℝ) (h1 : 2 * d = 520) : 
  d * 1.3 = 338 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l2621_262189


namespace NUMINAMATH_CALUDE_smallest_degree_for_horizontal_asymptote_l2621_262103

/-- 
Given a rational function f(x) = (5x^7 + 4x^4 - 3x + 2) / q(x),
prove that the smallest degree of q(x) for f(x) to have a horizontal asymptote is 7.
-/
theorem smallest_degree_for_horizontal_asymptote 
  (q : ℝ → ℝ) -- q is a real-valued function of a real variable
  (f : ℝ → ℝ) -- f is the rational function
  (hf : ∀ x, f x = (5*x^7 + 4*x^4 - 3*x + 2) / q x) -- definition of f
  : (∃ L : ℝ, ∀ ε > 0, ∃ M : ℝ, ∀ x, abs x > M → abs (f x - L) < ε) ↔ 
    (∃ n : ℕ, n ≥ 7 ∧ ∀ x, abs (q x) ≤ abs x^n + 1 ∧ abs x^n ≤ abs (q x) + 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_degree_for_horizontal_asymptote_l2621_262103


namespace NUMINAMATH_CALUDE_tangent_line_of_cubic_with_even_derivative_l2621_262141

/-- The tangent line equation for a cubic function with specific properties -/
theorem tangent_line_of_cubic_with_even_derivative (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a - 3)*x
  let f' : ℝ → ℝ := λ x ↦ (3*x^2 + 2*a*x + (a - 3))
  (∀ x, f' x = f' (-x)) →
  (λ x ↦ -3*x) = (λ x ↦ f' 0 * x) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_of_cubic_with_even_derivative_l2621_262141


namespace NUMINAMATH_CALUDE_inequality_solution_set_existence_condition_l2621_262121

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 4*x
def g (a : ℝ) : ℝ := |a - 2| + |a + 1|

-- Theorem for the first part of the problem
theorem inequality_solution_set :
  {x : ℝ | f x ≥ g 3} = {x : ℝ | x ≤ -5 ∨ x ≥ 1} := by sorry

-- Theorem for the second part of the problem
theorem existence_condition (a : ℝ) :
  (∃ x : ℝ, f x + g a = 0) → -3/2 ≤ a ∧ a ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_existence_condition_l2621_262121


namespace NUMINAMATH_CALUDE_bus_stop_problem_l2621_262104

/-- The number of children who got on the bus at a stop -/
def children_at_stop (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem bus_stop_problem (initial : ℕ) (final : ℕ) 
  (h1 : initial = 18) 
  (h2 : final = 25) :
  children_at_stop initial final = 7 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_problem_l2621_262104


namespace NUMINAMATH_CALUDE_min_S_19_l2621_262187

/-- An arithmetic sequence with its sum properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2
  arithmetic : ∀ n m, a (n + m) - a n = m * (a 2 - a 1)

/-- The minimum value of S_19 given the conditions -/
theorem min_S_19 (seq : ArithmeticSequence) 
  (h1 : seq.S 8 ≤ 6) (h2 : seq.S 11 ≥ 27) : 
  seq.S 19 ≥ 133 := by
  sorry

#check min_S_19

end NUMINAMATH_CALUDE_min_S_19_l2621_262187


namespace NUMINAMATH_CALUDE_tromino_tileability_l2621_262159

/-- Definition of a size-n tromino -/
def tromino (n : ℕ) := (4 * n * n) - 1

/-- Predicate for whether a tromino can be tiled by size-1 trominos -/
def can_be_tiled (n : ℕ) : Prop := ∃ k : ℕ, tromino n = 3 * k

/-- Theorem stating the condition for a size-n tromino to be tileable -/
theorem tromino_tileability (n : ℕ) (h : n > 0) : 
  can_be_tiled n ↔ n % 2 ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_tromino_tileability_l2621_262159


namespace NUMINAMATH_CALUDE_green_equals_purple_l2621_262170

/-- Proves that the number of green shoe pairs is equal to the number of purple shoe pairs -/
theorem green_equals_purple (total : ℕ) (blue : ℕ) (purple : ℕ)
  (h_total : total = 1250)
  (h_blue : blue = 540)
  (h_purple : purple = 355)
  (h_sum : total = blue + purple + (total - blue - purple)) :
  total - blue - purple = purple := by
  sorry

end NUMINAMATH_CALUDE_green_equals_purple_l2621_262170


namespace NUMINAMATH_CALUDE_assignments_count_l2621_262180

/-- The number of interest groups available --/
def num_groups : ℕ := 3

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of ways to assign students to interest groups --/
def num_assignments : ℕ := num_groups ^ num_students

/-- Theorem stating that the number of assignments is 81 --/
theorem assignments_count : num_assignments = 81 := by
  sorry

end NUMINAMATH_CALUDE_assignments_count_l2621_262180


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2621_262185

theorem fraction_inequality_solution_set :
  {x : ℝ | (2*x + 1) / (x - 3) ≤ 0} = {x : ℝ | -1/2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2621_262185


namespace NUMINAMATH_CALUDE_certain_number_solution_l2621_262175

theorem certain_number_solution : ∃ x : ℚ, (40 * 30 + (12 + 8) * x) / 5 = 1212 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_solution_l2621_262175


namespace NUMINAMATH_CALUDE_hall_volume_theorem_l2621_262144

/-- Represents the dimensions of a rectangular hall. -/
structure HallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular hall given its dimensions. -/
def hallVolume (d : HallDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the sum of the areas of the floor and ceiling of a rectangular hall. -/
def floorCeilingArea (d : HallDimensions) : ℝ :=
  2 * d.length * d.width

/-- Calculates the sum of the areas of the four walls of a rectangular hall. -/
def wallsArea (d : HallDimensions) : ℝ :=
  2 * d.height * (d.length + d.width)

/-- Theorem stating the volume of a specific rectangular hall with given conditions. -/
theorem hall_volume_theorem (d : HallDimensions) 
    (h_length : d.length = 15)
    (h_width : d.width = 12)
    (h_area_equality : floorCeilingArea d = wallsArea d) :
    ∃ ε > 0, |hallVolume d - 1201.8| < ε := by
  sorry

end NUMINAMATH_CALUDE_hall_volume_theorem_l2621_262144


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2621_262111

theorem complex_fraction_simplification :
  (((1 : ℂ) + 2 * Complex.I) ^ 2) / ((3 : ℂ) - 4 * Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2621_262111


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2621_262131

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 75 → 
  E = 4 * F - 37 → 
  D + E + F = 180 → 
  F = 28.4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2621_262131


namespace NUMINAMATH_CALUDE_digits_sum_after_erasures_l2621_262197

/-- Represents the initial sequence of digits -/
def initial_sequence : List Nat := [1, 2, 3, 4, 5, 6]

/-- Applies the erasure steps to a given sequence -/
def apply_erasures (seq : List Nat) : List Nat :=
  sorry

/-- Gets the digit at a specific position in the final sequence -/
def get_digit_at_position (pos : Nat) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem digits_sum_after_erasures :
  get_digit_at_position 3000 + get_digit_at_position 3001 + get_digit_at_position 3002 = 8 :=
sorry

end NUMINAMATH_CALUDE_digits_sum_after_erasures_l2621_262197


namespace NUMINAMATH_CALUDE_constant_d_value_l2621_262105

-- Define the problem statement
theorem constant_d_value (a d : ℝ) :
  (∀ x : ℝ, (x + 3) * (x + a) = x^2 + d*x + 12) →
  d = 7 :=
by sorry

end NUMINAMATH_CALUDE_constant_d_value_l2621_262105


namespace NUMINAMATH_CALUDE_next_simultaneous_event_is_180_lcm_9_60_is_180_l2621_262164

/-- Represents the interval in minutes between lighting up events -/
def light_interval : ℕ := 9

/-- Represents the interval in minutes between chiming events -/
def chime_interval : ℕ := 60

/-- Calculates the next time both events occur simultaneously -/
def next_simultaneous_event : ℕ := Nat.lcm light_interval chime_interval

/-- Theorem stating that the next simultaneous event occurs after 180 minutes -/
theorem next_simultaneous_event_is_180 : next_simultaneous_event = 180 := by
  sorry

/-- Theorem stating that 180 minutes is the least common multiple of 9 and 60 -/
theorem lcm_9_60_is_180 : Nat.lcm 9 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_next_simultaneous_event_is_180_lcm_9_60_is_180_l2621_262164


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2621_262162

theorem quadratic_root_difference (c : ℝ) : 
  (∃ x y : ℝ, x^2 + 7*x + c = 0 ∧ y^2 + 7*y + c = 0 ∧ |x - y| = Real.sqrt 85) → 
  c = -9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2621_262162


namespace NUMINAMATH_CALUDE_line_through_points_and_midpoint_l2621_262123

/-- Given a line y = ax + b passing through (2, 3) and (10, 19) with their midpoint on the line, a - b = 3 -/
theorem line_through_points_and_midpoint (a b : ℝ) : 
  (3 = a * 2 + b) → 
  (19 = a * 10 + b) → 
  (11 = a * 6 + b) → 
  a - b = 3 := by sorry

end NUMINAMATH_CALUDE_line_through_points_and_midpoint_l2621_262123


namespace NUMINAMATH_CALUDE_sum_product_theorem_l2621_262129

theorem sum_product_theorem (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 1)
  (eq3 : a + c + d = 12)
  (eq4 : b + c + d = 7) :
  a * b + c * d = 176 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_product_theorem_l2621_262129


namespace NUMINAMATH_CALUDE_playing_hours_calculation_l2621_262158

def total_hours : ℝ := 24
def total_angle : ℝ := 360
def sleeping_angle : ℝ := 130
def eating_angle : ℝ := 110

theorem playing_hours_calculation :
  let playing_angle : ℝ := total_angle - sleeping_angle - eating_angle
  let playing_fraction : ℝ := playing_angle / total_angle
  playing_fraction * total_hours = 8 := by sorry

end NUMINAMATH_CALUDE_playing_hours_calculation_l2621_262158


namespace NUMINAMATH_CALUDE_turquoise_score_difference_is_correct_l2621_262156

/-- Calculates 5/8 of the difference between white and black scores in a turquoise mixture --/
def turquoise_score_difference (total : ℚ) : ℚ :=
  let white_ratio : ℚ := 5
  let black_ratio : ℚ := 3
  let total_ratio : ℚ := white_ratio + black_ratio
  let part_value : ℚ := total / total_ratio
  let white_scores : ℚ := white_ratio * part_value
  let black_scores : ℚ := black_ratio * part_value
  let difference : ℚ := white_scores - black_scores
  (5 : ℚ) / 8 * difference

/-- Theorem stating that 5/8 of the difference between white and black scores is 58.125 --/
theorem turquoise_score_difference_is_correct :
  turquoise_score_difference 372 = 58125 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_turquoise_score_difference_is_correct_l2621_262156


namespace NUMINAMATH_CALUDE_mod_eight_difference_l2621_262133

theorem mod_eight_difference (n : ℕ) : (47^1824 - 25^1824) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_difference_l2621_262133


namespace NUMINAMATH_CALUDE_sales_volume_estimate_l2621_262195

/-- Represents the linear regression equation for sales volume and price -/
def regression_equation (x : ℝ) : ℝ := -10 * x + 200

/-- The selling price in yuan -/
def selling_price : ℝ := 10

/-- Theorem stating that the estimated sales volume is approximately 100 pieces when the selling price is 10 yuan -/
theorem sales_volume_estimate :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |regression_equation selling_price - 100| < ε :=
sorry

end NUMINAMATH_CALUDE_sales_volume_estimate_l2621_262195


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2621_262160

theorem opposite_of_2023 : 
  ∀ y : ℤ, (2023 + y = 0) ↔ (y = -2023) := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2621_262160


namespace NUMINAMATH_CALUDE_supermarket_spending_l2621_262132

theorem supermarket_spending (total : ℚ) : 
  (1/4 : ℚ) * total + (1/3 : ℚ) * total + (1/6 : ℚ) * total + 6 = total →
  total = 24 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_l2621_262132


namespace NUMINAMATH_CALUDE_time_to_cross_signal_pole_l2621_262152

-- Define the train and platform parameters
def train_length : ℝ := 300
def platform_length : ℝ := 400
def time_cross_platform : ℝ := 42

-- Define the theorem
theorem time_to_cross_signal_pole :
  let total_distance := train_length + platform_length
  let train_speed := total_distance / time_cross_platform
  let time_cross_pole := train_length / train_speed
  time_cross_pole = 18 := by
  sorry


end NUMINAMATH_CALUDE_time_to_cross_signal_pole_l2621_262152


namespace NUMINAMATH_CALUDE_max_profit_zongzi_l2621_262138

/-- Represents the cost and selling prices of zongzi types A and B -/
structure ZongziPrices where
  cost_a : ℚ
  cost_b : ℚ
  sell_a : ℚ
  sell_b : ℚ

/-- Represents the purchase quantities of zongzi types A and B -/
structure ZongziQuantities where
  qty_a : ℕ
  qty_b : ℕ

/-- Calculates the profit given prices and quantities -/
def profit (p : ZongziPrices) (q : ZongziQuantities) : ℚ :=
  (p.sell_a - p.cost_a) * q.qty_a + (p.sell_b - p.cost_b) * q.qty_b

/-- Theorem stating the maximum profit achievable under given conditions -/
theorem max_profit_zongzi (p : ZongziPrices) (q : ZongziQuantities) :
  p.cost_b = p.cost_a + 2 →
  1000 / p.cost_a = 1200 / p.cost_b →
  p.sell_a = 12 →
  p.sell_b = 15 →
  q.qty_a + q.qty_b = 200 →
  q.qty_a ≥ 2 * q.qty_b →
  ∃ (max_q : ZongziQuantities),
    max_q.qty_a = 134 ∧
    max_q.qty_b = 66 ∧
    ∀ (other_q : ZongziQuantities),
      other_q.qty_a + other_q.qty_b = 200 →
      other_q.qty_a ≥ 2 * other_q.qty_b →
      profit p max_q ≥ profit p other_q :=
sorry

end NUMINAMATH_CALUDE_max_profit_zongzi_l2621_262138


namespace NUMINAMATH_CALUDE_sixth_score_for_mean_90_l2621_262135

def quiz_scores : List ℕ := [85, 90, 88, 92, 95]

def arithmetic_mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem sixth_score_for_mean_90 (x : ℕ) :
  arithmetic_mean (quiz_scores ++ [x]) = 90 → x = 90 := by
  sorry

end NUMINAMATH_CALUDE_sixth_score_for_mean_90_l2621_262135


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2621_262186

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence : arithmetic_sequence 3 4 30 = 119 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2621_262186


namespace NUMINAMATH_CALUDE_car_ownership_l2621_262113

theorem car_ownership (total : ℕ) (neither : ℕ) (both : ℕ) (bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 20)
  (h4 : bike_only = 35) :
  total - neither - bike_only = 44 :=
by sorry

end NUMINAMATH_CALUDE_car_ownership_l2621_262113


namespace NUMINAMATH_CALUDE_total_pets_is_415_l2621_262174

/-- The number of dogs at the farm -/
def num_dogs : ℕ := 43

/-- The number of fish at the farm -/
def num_fish : ℕ := 72

/-- The number of cats at the farm -/
def num_cats : ℕ := 34

/-- The number of chickens at the farm -/
def num_chickens : ℕ := 120

/-- The number of rabbits at the farm -/
def num_rabbits : ℕ := 57

/-- The number of parrots at the farm -/
def num_parrots : ℕ := 89

/-- The total number of pets at the farm -/
def total_pets : ℕ := num_dogs + num_fish + num_cats + num_chickens + num_rabbits + num_parrots

theorem total_pets_is_415 : total_pets = 415 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_is_415_l2621_262174


namespace NUMINAMATH_CALUDE_supplementary_angles_theorem_l2621_262127

theorem supplementary_angles_theorem (A B : ℝ) : 
  A + B = 180 →  -- angles A and B are supplementary
  A = 4 * B →    -- measure of angle A is 4 times angle B
  A = 144 :=     -- measure of angle A is 144 degrees
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_theorem_l2621_262127


namespace NUMINAMATH_CALUDE_max_value_theorem_l2621_262196

theorem max_value_theorem (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) 
  (h_sum : x₁ + x₂ + x₃ = 1) : 
  x₁ * x₂^2 * x₃ + x₁ * x₂ * x₃^2 ≤ 27/1024 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2621_262196


namespace NUMINAMATH_CALUDE_max_value_theorem_l2621_262143

theorem max_value_theorem (p q r s : ℝ) (h : p^2 + q^2 + r^2 - s^2 + 4 = 0) :
  ∃ (M : ℝ), M = -2 * Real.sqrt 2 ∧ ∀ (p' q' r' s' : ℝ), 
    p'^2 + q'^2 + r'^2 - s'^2 + 4 = 0 → 
    3*p' + 2*q' + r' - 4*abs s' ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2621_262143


namespace NUMINAMATH_CALUDE_bacon_percentage_of_total_l2621_262154

def total_sandwich_calories : ℕ := 1250
def bacon_strips : ℕ := 2
def calories_per_bacon_strip : ℕ := 125

def bacon_calories : ℕ := bacon_strips * calories_per_bacon_strip

theorem bacon_percentage_of_total (h : bacon_calories = bacon_strips * calories_per_bacon_strip) :
  (bacon_calories : ℚ) / total_sandwich_calories * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bacon_percentage_of_total_l2621_262154


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2621_262163

/-- Given three mutually externally tangent circles with radii a, b, and c,
    the radius r of the inscribed circle satisfies the given equation. -/
theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 3) (hb : b = 6) (hc : c = 18) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 9 / 8 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2621_262163


namespace NUMINAMATH_CALUDE_inequality_proof_l2621_262109

theorem inequality_proof (x y z t : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : t ≥ 0)
  (h5 : x * y * z = 1) (h6 : y + z + t = 2) :
  x^2 + y^2 + z^2 + t^2 ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2621_262109


namespace NUMINAMATH_CALUDE_factors_of_72_l2621_262108

def number_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factors_of_72 : number_of_factors 72 = 12 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_72_l2621_262108


namespace NUMINAMATH_CALUDE_exact_three_ones_between_zeros_l2621_262145

/-- A sequence of 10 elements consisting of 8 ones and 2 zeros -/
def Sequence := Fin 10 → Fin 2

/-- The number of sequences with exactly three ones between two zeros -/
def favorable_sequences : ℕ := 12

/-- The total number of possible sequences -/
def total_sequences : ℕ := Nat.choose 10 2

/-- The probability of having exactly three ones between two zeros -/
def probability : ℚ := favorable_sequences / total_sequences

theorem exact_three_ones_between_zeros :
  probability = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_exact_three_ones_between_zeros_l2621_262145


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l2621_262126

/-- Given a total purchase amount, sales tax paid, and tax rate,
    calculate the cost of tax-free items. -/
def cost_of_tax_free_items (total_purchase : ℚ) (sales_tax : ℚ) (tax_rate : ℚ) : ℚ :=
  total_purchase - sales_tax / tax_rate

/-- Theorem stating that given the specific conditions in the problem,
    the cost of tax-free items is 22. -/
theorem tax_free_items_cost :
  let total_purchase : ℚ := 25
  let sales_tax : ℚ := 30 / 100  -- 30 paise = 0.30 rupees
  let tax_rate : ℚ := 10 / 100   -- 10% = 0.10
  cost_of_tax_free_items total_purchase sales_tax tax_rate = 22 := by
  sorry


end NUMINAMATH_CALUDE_tax_free_items_cost_l2621_262126


namespace NUMINAMATH_CALUDE_table_price_is_300_l2621_262119

def table_selling_price (num_trees : ℕ) (planks_per_tree : ℕ) (planks_per_table : ℕ) 
                        (labor_cost : ℕ) (profit : ℕ) : ℕ :=
  let total_planks := num_trees * planks_per_tree
  let num_tables := total_planks / planks_per_table
  let total_revenue := labor_cost + profit
  total_revenue / num_tables

theorem table_price_is_300 :
  table_selling_price 30 25 15 3000 12000 = 300 := by
  sorry

end NUMINAMATH_CALUDE_table_price_is_300_l2621_262119


namespace NUMINAMATH_CALUDE_unique_integer_solution_l2621_262102

theorem unique_integer_solution : ∃! (x : ℤ), (45 + x / 89) * 89 = 4028 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l2621_262102


namespace NUMINAMATH_CALUDE_circle_point_x_coordinate_l2621_262115

theorem circle_point_x_coordinate 
  (x : ℝ) 
  (h1 : (x - 6)^2 + 10^2 = 12^2) : 
  x = 6 + 2 * Real.sqrt 11 ∨ x = 6 - 2 * Real.sqrt 11 := by
  sorry


end NUMINAMATH_CALUDE_circle_point_x_coordinate_l2621_262115


namespace NUMINAMATH_CALUDE_sphere_unique_orientation_independent_projections_l2621_262179

-- Define the type for 3D objects
inductive Object3D
  | Cube
  | RegularTetrahedron
  | RightTriangularPyramid
  | Sphere

-- Define a function to check if an object's projections are orientation-independent
def hasOrientationIndependentProjections (obj : Object3D) : Prop :=
  match obj with
  | Object3D.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_unique_orientation_independent_projections :
  ∀ (obj : Object3D), hasOrientationIndependentProjections obj ↔ obj = Object3D.Sphere :=
sorry

end NUMINAMATH_CALUDE_sphere_unique_orientation_independent_projections_l2621_262179


namespace NUMINAMATH_CALUDE_f_f_7_equals_0_l2621_262167

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_4 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x

theorem f_f_7_equals_0 (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_periodic : is_periodic_4 f)
  (h_f_1 : f 1 = 4) :
  f (f 7) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_f_7_equals_0_l2621_262167


namespace NUMINAMATH_CALUDE_cannoneer_count_l2621_262110

theorem cannoneer_count (total : ℕ) (cannoneers : ℕ) (women : ℕ) (men : ℕ)
  (h1 : women = 2 * cannoneers)
  (h2 : men = 2 * women)
  (h3 : total = men + women)
  (h4 : total = 378) :
  cannoneers = 63 := by
sorry

end NUMINAMATH_CALUDE_cannoneer_count_l2621_262110


namespace NUMINAMATH_CALUDE_triangle_max_area_l2621_262106

/-- Given two positive real numbers a and b representing the lengths of two sides of a triangle,
    the area of the triangle is maximized when these sides are perpendicular. -/
theorem triangle_max_area (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ θ : ℝ, 0 < θ ∧ θ < π → (1/2) * a * b * Real.sin θ ≤ (1/2) * a * b := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2621_262106


namespace NUMINAMATH_CALUDE_fraction_of_108_l2621_262166

theorem fraction_of_108 : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 108 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_108_l2621_262166


namespace NUMINAMATH_CALUDE_inequality_abc_l2621_262171

theorem inequality_abc (a b c : ℝ) (ha : a = Real.log 2.1) (hb : b = Real.exp 0.1) (hc : c = 1.1) :
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_abc_l2621_262171


namespace NUMINAMATH_CALUDE_possible_sets_for_B_l2621_262151

def set_A : Set ℕ := {1, 2}
def set_B : Set ℕ := {1, 2, 3, 4}

theorem possible_sets_for_B (B : Set ℕ) 
  (h1 : set_A ⊆ B) (h2 : B ⊆ set_B) :
  B = set_A ∨ B = {1, 2, 3} ∨ B = {1, 2, 4} :=
sorry

end NUMINAMATH_CALUDE_possible_sets_for_B_l2621_262151


namespace NUMINAMATH_CALUDE_sqrt_nine_minus_half_inverse_equals_one_l2621_262188

theorem sqrt_nine_minus_half_inverse_equals_one :
  Real.sqrt 9 - (1/2)⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_minus_half_inverse_equals_one_l2621_262188


namespace NUMINAMATH_CALUDE_evaluate_expression_l2621_262149

/-- Given x = -1 and y = 2, prove that -2x²y-3(2xy-x²y)+4xy evaluates to 6 -/
theorem evaluate_expression (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  -2 * x^2 * y - 3 * (2 * x * y - x^2 * y) + 4 * x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2621_262149


namespace NUMINAMATH_CALUDE_fraction_relation_l2621_262120

theorem fraction_relation (a b c : ℚ) 
  (h1 : a / b = 3 / 5) 
  (h2 : b / c = 2 / 7) : 
  c / a = 35 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relation_l2621_262120


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2621_262100

theorem inequality_equivalence (x : ℝ) (h : x ≠ 1) :
  1 / (x - 1) > 1 ↔ 1 < x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2621_262100


namespace NUMINAMATH_CALUDE_james_total_score_l2621_262114

/-- Calculates the total points scored by James in a basketball game -/
def total_points (field_goals three_pointers two_pointers free_throws : ℕ) : ℕ :=
  field_goals * 3 + three_pointers * 2 + two_pointers * 2 + free_throws * 1

theorem james_total_score :
  total_points 13 0 20 5 = 84 := by
  sorry

#eval total_points 13 0 20 5

end NUMINAMATH_CALUDE_james_total_score_l2621_262114


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l2621_262155

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that the loss percentage for a radio with cost price 1500 and selling price 1305 is 13% -/
theorem radio_loss_percentage :
  let cost_price : ℚ := 1500
  let selling_price : ℚ := 1305
  loss_percentage cost_price selling_price = 13 := by
  sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l2621_262155


namespace NUMINAMATH_CALUDE_clock_angle_at_3_25_l2621_262116

/-- The angle of the minute hand on a clock face at a given number of minutes past the hour -/
def minute_hand_angle (minutes : ℕ) : ℝ :=
  minutes * 6

/-- The angle of the hour hand on a clock face at a given hour and minute -/
def hour_hand_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  hour * 30 + minute * 0.5

/-- The angle between the hour hand and minute hand on a clock face -/
def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  |hour_hand_angle hour minute - minute_hand_angle minute|

theorem clock_angle_at_3_25 :
  clock_angle 3 25 = 47.5 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_25_l2621_262116


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2621_262181

theorem constant_term_expansion : 
  let p₁ : Polynomial ℤ := X^4 + 2*X^2 + 7
  let p₂ : Polynomial ℤ := 2*X^5 + 3*X^3 + 25
  (p₁ * p₂).coeff 0 = 175 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2621_262181


namespace NUMINAMATH_CALUDE_marble_selection_problem_l2621_262184

theorem marble_selection_problem :
  let total_marbles : ℕ := 18
  let specific_marbles : ℕ := 4
  let choose_marbles : ℕ := 6

  (specific_marbles.choose 1) * ((total_marbles - specific_marbles).choose (choose_marbles - 1)) = 8008 :=
by sorry

end NUMINAMATH_CALUDE_marble_selection_problem_l2621_262184


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2621_262142

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2621_262142


namespace NUMINAMATH_CALUDE_inequality_proof_l2621_262176

theorem inequality_proof (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2621_262176


namespace NUMINAMATH_CALUDE_equation_transformation_l2621_262146

theorem equation_transformation (x y : ℝ) :
  (2 * x - 3 * y = 6) ↔ (y = (2 * x - 6) / 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l2621_262146


namespace NUMINAMATH_CALUDE_certain_number_problem_l2621_262198

theorem certain_number_problem (x : ℝ) : 
  (0.55 * x = (4 / 5 : ℝ) * 25 + 2) → x = 40 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2621_262198


namespace NUMINAMATH_CALUDE_expression_value_l2621_262137

theorem expression_value (a b c d x : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1)
  (h3 : |x| = Real.sqrt 7) :
  x^2 + (a + b) * c * d * x + Real.sqrt (a + b) + (c * d)^(1/3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2621_262137


namespace NUMINAMATH_CALUDE_shopping_lottery_results_l2621_262147

/-- Represents the lottery event with 10 coupons -/
structure LotteryEvent where
  total_coupons : Nat
  first_prize_coupons : Nat
  second_prize_coupons : Nat
  non_prize_coupons : Nat
  first_prize_value : Nat
  second_prize_value : Nat
  drawn_coupons : Nat

/-- The specific lottery event described in the problem -/
def shopping_lottery : LotteryEvent :=
  { total_coupons := 10
  , first_prize_coupons := 1
  , second_prize_coupons := 3
  , non_prize_coupons := 6
  , first_prize_value := 50
  , second_prize_value := 10
  , drawn_coupons := 2
  }

/-- The probability of winning a prize in the shopping lottery -/
def win_probability (l : LotteryEvent) : Rat :=
  1 - (Nat.choose l.non_prize_coupons l.drawn_coupons) / (Nat.choose l.total_coupons l.drawn_coupons)

/-- The mathematical expectation of the total prize value in the shopping lottery -/
def prize_expectation (l : LotteryEvent) : Rat :=
  let p0 := (Nat.choose l.non_prize_coupons l.drawn_coupons) / (Nat.choose l.total_coupons l.drawn_coupons)
  let p10 := (Nat.choose l.second_prize_coupons 1 * Nat.choose l.non_prize_coupons 1) / (Nat.choose l.total_coupons l.drawn_coupons)
  let p20 := (Nat.choose l.second_prize_coupons 2) / (Nat.choose l.total_coupons l.drawn_coupons)
  let p50 := (Nat.choose l.first_prize_coupons 1 * Nat.choose l.non_prize_coupons 1) / (Nat.choose l.total_coupons l.drawn_coupons)
  let p60 := (Nat.choose l.first_prize_coupons 1 * Nat.choose l.second_prize_coupons 1) / (Nat.choose l.total_coupons l.drawn_coupons)
  0 * p0 + 10 * p10 + 20 * p20 + 50 * p50 + 60 * p60

theorem shopping_lottery_results :
  win_probability shopping_lottery = 2/3 ∧
  prize_expectation shopping_lottery = 16 := by
  sorry

end NUMINAMATH_CALUDE_shopping_lottery_results_l2621_262147


namespace NUMINAMATH_CALUDE_positive_difference_theorem_l2621_262125

theorem positive_difference_theorem : ∃ (x : ℝ), x > 0 ∧ x = |((8^2 * 8^2) / 8) - ((8^2 + 8^2) / 8)| ∧ x = 496 := by
  sorry

end NUMINAMATH_CALUDE_positive_difference_theorem_l2621_262125


namespace NUMINAMATH_CALUDE_coefficient_a2_l2621_262193

/-- Given z = 1 + i and (z+x)^4 = a_4x^4 + a_3x^3 + a_2x^2 + a_1x + a_0, prove that a_2 = 12i -/
theorem coefficient_a2 (z : ℂ) (a_4 a_3 a_2 a_1 a_0 : ℂ) :
  z = 1 + Complex.I →
  (∀ x : ℂ, (z + x)^4 = a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
  a_2 = 12 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_coefficient_a2_l2621_262193


namespace NUMINAMATH_CALUDE_average_of_remaining_quantities_l2621_262134

theorem average_of_remaining_quantities
  (total_count : ℕ)
  (subset_count : ℕ)
  (total_average : ℚ)
  (subset_average : ℚ)
  (h1 : total_count = 6)
  (h2 : subset_count = 4)
  (h3 : total_average = 8)
  (h4 : subset_average = 5) :
  let remaining_count := total_count - subset_count
  let remaining_sum := total_count * total_average - subset_count * subset_average
  remaining_sum / remaining_count = 14 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_quantities_l2621_262134


namespace NUMINAMATH_CALUDE_min_value_of_f_l2621_262148

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 4 * y^2 - 8 * x - 6 * y

/-- The theorem stating that the minimum value of f is -14 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = -14 ∧ ∀ (x y : ℝ), f x y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2621_262148


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2621_262172

theorem right_triangle_hypotenuse (x y : ℝ) :
  x > 0 ∧ y > 0 →
  (1/3) * π * x * y^2 = 800 * π →
  (1/3) * π * y * x^2 = 1920 * π →
  Real.sqrt (x^2 + y^2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2621_262172


namespace NUMINAMATH_CALUDE_steves_pencils_l2621_262118

/-- Steve's pencil distribution problem -/
theorem steves_pencils (boxes : ℕ) (pencils_per_box : ℕ) (lauren_pencils : ℕ) (matt_extra : ℕ) :
  boxes = 2 →
  pencils_per_box = 12 →
  lauren_pencils = 6 →
  matt_extra = 3 →
  boxes * pencils_per_box - lauren_pencils - (lauren_pencils + matt_extra) = 9 :=
by sorry

end NUMINAMATH_CALUDE_steves_pencils_l2621_262118


namespace NUMINAMATH_CALUDE_first_player_wins_l2621_262165

/-- Represents a chessboard with knights on opposite corners -/
structure Chessboard :=
  (squares : Finset (ℕ × ℕ))
  (knight1 : ℕ × ℕ)
  (knight2 : ℕ × ℕ)

/-- Represents a move in the game -/
def Move := ℕ × ℕ

/-- Checks if a knight can reach another position on the board -/
def can_reach (board : Chessboard) (start finish : ℕ × ℕ) : Prop :=
  sorry

/-- Represents the game state -/
structure GameState :=
  (board : Chessboard)
  (current_player : ℕ)

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a player has a winning strategy -/
def has_winning_strategy (player : ℕ) (state : GameState) : Prop :=
  sorry

/-- The main theorem stating that the first player has a winning strategy -/
theorem first_player_wins (initial_board : Chessboard) :
  has_winning_strategy 1 { board := initial_board, current_player := 1 } :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l2621_262165


namespace NUMINAMATH_CALUDE_lawn_mowing_time_l2621_262122

theorem lawn_mowing_time (mary_rate tom_rate : ℚ) (mary_time : ℚ) : 
  mary_rate = 1 / 3 →
  tom_rate = 1 / 6 →
  mary_time = 2 →
  (1 - mary_rate * mary_time) / tom_rate = 2 :=
by sorry

end NUMINAMATH_CALUDE_lawn_mowing_time_l2621_262122


namespace NUMINAMATH_CALUDE_simplify_expression_l2621_262117

theorem simplify_expression (a : ℝ) (h : 2 < a ∧ a < 3) :
  |a - 2| - Real.sqrt ((a - 3)^2) = 2*a - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2621_262117


namespace NUMINAMATH_CALUDE_baseball_gear_sale_l2621_262112

theorem baseball_gear_sale (bat_price glove_original_price glove_discount cleats_price total_amount : ℝ)
  (h1 : bat_price = 10)
  (h2 : glove_original_price = 30)
  (h3 : glove_discount = 0.2)
  (h4 : cleats_price = 10)
  (h5 : total_amount = 79) :
  let glove_sale_price := glove_original_price * (1 - glove_discount)
  let other_gear_total := bat_price + glove_sale_price + 2 * cleats_price
  total_amount - other_gear_total = 25 := by
sorry

end NUMINAMATH_CALUDE_baseball_gear_sale_l2621_262112


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_zero_l2621_262153

theorem logarithm_expression_equals_zero :
  Real.log 14 - 2 * Real.log (7/3) + Real.log 7 - Real.log 18 = 0 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_zero_l2621_262153


namespace NUMINAMATH_CALUDE_ant_movement_theorem_l2621_262194

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the movement of an ant --/
structure AntMovement where
  seconds : ℕ
  unitPerSecond : ℝ

/-- Calculates the expected area of the convex quadrilateral formed by ants --/
def expectedArea (rect : Rectangle) (movement : AntMovement) : ℝ :=
  (rect.length - 2 * movement.seconds * movement.unitPerSecond) *
  (rect.width - 2 * movement.seconds * movement.unitPerSecond)

/-- Theorem statement for the ant movement problem --/
theorem ant_movement_theorem (rect : Rectangle) (movement : AntMovement) :
  rect.length = 20 ∧ rect.width = 23 ∧ movement.seconds = 10 ∧ movement.unitPerSecond = 0.5 →
  expectedArea rect movement = 130 := by
  sorry


end NUMINAMATH_CALUDE_ant_movement_theorem_l2621_262194


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_nine_sqrt_three_l2621_262101

theorem sqrt_sum_equals_nine_sqrt_three : 
  Real.sqrt 12 + Real.sqrt 27 + Real.sqrt 48 = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_nine_sqrt_three_l2621_262101


namespace NUMINAMATH_CALUDE_count_negative_expressions_l2621_262136

theorem count_negative_expressions : 
  let expressions := [-3^2, (-3)^2, -(-3), -|-3|]
  (expressions.filter (· < 0)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_negative_expressions_l2621_262136


namespace NUMINAMATH_CALUDE_union_of_sets_l2621_262161

theorem union_of_sets : 
  let A : Set ℕ := {2, 3}
  let B : Set ℕ := {1, 2}
  A ∪ B = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2621_262161

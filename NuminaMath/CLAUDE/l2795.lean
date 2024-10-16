import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_to_square_l2795_279591

/-- Given a rectangle with area 54 m², if one side is tripled and the other is halved to form a square, 
    the side length of the resulting square is 9 m. -/
theorem rectangle_to_square (a b : ℝ) (h1 : a * b = 54) (h2 : 3 * a = b / 2) : 
  3 * a = 9 ∧ b / 2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l2795_279591


namespace NUMINAMATH_CALUDE_shirt_cost_is_15_l2795_279536

/-- The cost of one pair of jeans -/
def jeans_cost : ℝ := sorry

/-- The cost of one shirt -/
def shirt_cost : ℝ := sorry

/-- The first condition: 3 pairs of jeans and 2 shirts cost $69 -/
axiom condition1 : 3 * jeans_cost + 2 * shirt_cost = 69

/-- The second condition: 2 pairs of jeans and 3 shirts cost $71 -/
axiom condition2 : 2 * jeans_cost + 3 * shirt_cost = 71

/-- Theorem: The cost of one shirt is $15 -/
theorem shirt_cost_is_15 : shirt_cost = 15 := by sorry

end NUMINAMATH_CALUDE_shirt_cost_is_15_l2795_279536


namespace NUMINAMATH_CALUDE_fraction_leading_zeros_l2795_279598

/-- The number of leading zeros in the decimal representation of a rational number -/
def leadingZeros (q : ℚ) : ℕ := sorry

/-- The fraction we're analyzing -/
def fraction : ℚ := 1 / (2^7 * 5^9)

theorem fraction_leading_zeros : leadingZeros fraction = 8 := by sorry

end NUMINAMATH_CALUDE_fraction_leading_zeros_l2795_279598


namespace NUMINAMATH_CALUDE_square_area_difference_l2795_279592

/-- Given two squares ABCD and EGFO with the specified conditions, 
    prove that the difference between their areas is 11.5 -/
theorem square_area_difference (a b : ℕ+) 
  (h1 : (a.val : ℝ)^2 / 2 - (b.val : ℝ)^2 / 2 = 3.25) 
  (h2 : (b.val : ℝ) > (a.val : ℝ)) : 
  (a.val : ℝ)^2 - (b.val : ℝ)^2 = -11.5 := by
  sorry

end NUMINAMATH_CALUDE_square_area_difference_l2795_279592


namespace NUMINAMATH_CALUDE_system_of_equations_l2795_279571

theorem system_of_equations (p t j x y : ℝ) : 
  j = 0.75 * p →
  j = 0.8 * t →
  t = p * (1 - t / 100) →
  x = 0.1 * t →
  y = 0.5 * j →
  x + y = 12 →
  t = 24 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l2795_279571


namespace NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l2795_279599

theorem triangle_sine_sum_inequality (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) : 
  Real.sin α + Real.sin β + Real.sin γ ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l2795_279599


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt2_over_2_l2795_279560

theorem cos_sin_sum_equals_sqrt2_over_2 :
  Real.cos (16 * π / 180) * Real.cos (61 * π / 180) + 
  Real.sin (16 * π / 180) * Real.sin (61 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt2_over_2_l2795_279560


namespace NUMINAMATH_CALUDE_sin_cos_extrema_l2795_279557

theorem sin_cos_extrema (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  (∀ a b : ℝ, Real.sin a + Real.sin b = 1/3 → 
    Real.sin a - Real.cos b ^ 2 ≤ 4/9 ∧ 
    Real.sin a - Real.cos b ^ 2 ≥ -11/12) ∧
  (∃ c d : ℝ, Real.sin c + Real.sin d = 1/3 ∧ 
    Real.sin c - Real.cos d ^ 2 = 4/9) ∧
  (∃ e f : ℝ, Real.sin e + Real.sin f = 1/3 ∧ 
    Real.sin e - Real.cos f ^ 2 = -11/12) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_extrema_l2795_279557


namespace NUMINAMATH_CALUDE_problem_l_shape_surface_area_l2795_279588

/-- Represents a 3D L-shaped structure made of unit cubes -/
structure LShape where
  verticalHeight : ℕ
  verticalWidth : ℕ
  horizontalLength : ℕ
  totalCubes : ℕ

/-- Calculates the surface area of an L-shaped structure -/
def surfaceArea (l : LShape) : ℕ :=
  sorry

/-- The specific L-shape described in the problem -/
def problemLShape : LShape :=
  { verticalHeight := 3
  , verticalWidth := 2
  , horizontalLength := 3
  , totalCubes := 15 }

/-- Theorem stating that the surface area of the problem's L-shape is 34 square units -/
theorem problem_l_shape_surface_area :
  surfaceArea problemLShape = 34 :=
sorry

end NUMINAMATH_CALUDE_problem_l_shape_surface_area_l2795_279588


namespace NUMINAMATH_CALUDE_sixteen_is_sixtyfour_percent_of_twentyfive_l2795_279553

theorem sixteen_is_sixtyfour_percent_of_twentyfive :
  ∀ x : ℚ, (16 : ℚ) = 64 / 100 * x → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_is_sixtyfour_percent_of_twentyfive_l2795_279553


namespace NUMINAMATH_CALUDE_smallest_four_digit_palindrome_div_by_3_odd_first_l2795_279542

/-- A function that checks if a number is a four-digit palindrome -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- A function that checks if a number has an odd first digit -/
def has_odd_first_digit (n : ℕ) : Prop :=
  n ≥ 1000 ∧ Odd (n / 1000)

/-- The theorem stating that 1221 is the smallest four-digit palindrome 
    divisible by 3 with an odd first digit -/
theorem smallest_four_digit_palindrome_div_by_3_odd_first : 
  (∀ n : ℕ, is_four_digit_palindrome n ∧ n % 3 = 0 ∧ has_odd_first_digit n → n ≥ 1221) ∧
  is_four_digit_palindrome 1221 ∧ 1221 % 3 = 0 ∧ has_odd_first_digit 1221 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_palindrome_div_by_3_odd_first_l2795_279542


namespace NUMINAMATH_CALUDE_binomial_fraction_zero_l2795_279537

theorem binomial_fraction_zero : (Nat.choose 2 5 * 3^5) / Nat.choose 10 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_binomial_fraction_zero_l2795_279537


namespace NUMINAMATH_CALUDE_sphere_only_all_circular_views_l2795_279585

-- Define the geometric shapes
inductive GeometricShape
  | Cuboid
  | Cylinder
  | Cone
  | Sphere

-- Define the view types
inductive ViewType
  | Front
  | Left
  | Top

-- Define a function to check if a view is circular
def isCircularView (shape : GeometricShape) (view : ViewType) : Prop :=
  match shape, view with
  | GeometricShape.Sphere, _ => true
  | GeometricShape.Cylinder, ViewType.Top => true
  | GeometricShape.Cone, ViewType.Top => true
  | _, _ => false

-- Define a function to check if all three views are circular
def hasAllCircularViews (shape : GeometricShape) : Prop :=
  isCircularView shape ViewType.Front ∧
  isCircularView shape ViewType.Left ∧
  isCircularView shape ViewType.Top

-- Theorem: Only the sphere has circular views in all three perspectives
theorem sphere_only_all_circular_views :
  ∀ (shape : GeometricShape),
    hasAllCircularViews shape ↔ shape = GeometricShape.Sphere :=
by sorry

end NUMINAMATH_CALUDE_sphere_only_all_circular_views_l2795_279585


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l2795_279562

theorem smallest_number_of_eggs : ∀ n : ℕ,
  (n > 150) →
  (∃ k : ℕ, n = 15 * k - 6) →
  (∀ m : ℕ, (m > 150 ∧ ∃ j : ℕ, m = 15 * j - 6) → m ≥ n) →
  n = 159 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l2795_279562


namespace NUMINAMATH_CALUDE_molecular_weight_4_moles_BaI2_value_l2795_279503

/-- The molecular weight of 4 moles of Barium iodide (BaI2) -/
def molecular_weight_4_moles_BaI2 : ℝ :=
  let atomic_weight_Ba : ℝ := 137.33
  let atomic_weight_I : ℝ := 126.90
  let molecular_weight_BaI2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_I
  4 * molecular_weight_BaI2

/-- Theorem stating that the molecular weight of 4 moles of Barium iodide is 1564.52 grams -/
theorem molecular_weight_4_moles_BaI2_value : 
  molecular_weight_4_moles_BaI2 = 1564.52 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_4_moles_BaI2_value_l2795_279503


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l2795_279501

theorem cos_double_angle_special_case (α : Real) 
  (h : Real.sin (α + Real.pi / 2) = 1 / 2) : 
  Real.cos (2 * α) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l2795_279501


namespace NUMINAMATH_CALUDE_pyramid_theorem_l2795_279543

/-- Regular quadrangular pyramid with a plane through diagonal of base and height -/
structure RegularQuadPyramid where
  /-- Side length of the base -/
  a : ℝ
  /-- Angle between opposite slant heights -/
  α : ℝ
  /-- Ratio of section area to lateral surface area -/
  k : ℝ
  /-- Base side length is positive -/
  a_pos : 0 < a
  /-- Angle is between 0 and π -/
  α_range : 0 < α ∧ α < π
  /-- k is positive -/
  k_pos : 0 < k

/-- Theorem about the cosine of the angle between slant heights and permissible k values -/
theorem pyramid_theorem (p : RegularQuadPyramid) :
  (Real.cos p.α = 64 * p.k^2 - 1) ∧ 
  (p.k ≤ Real.sqrt 2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_theorem_l2795_279543


namespace NUMINAMATH_CALUDE_stadium_entrance_exit_ways_l2795_279582

/-- The number of gates on the south side of the stadium -/
def south_gates : ℕ := 4

/-- The number of gates on the north side of the stadium -/
def north_gates : ℕ := 3

/-- The total number of gates in the stadium -/
def total_gates : ℕ := south_gates + north_gates

/-- The number of different ways to enter and exit the stadium -/
def entrance_exit_ways : ℕ := total_gates * total_gates

theorem stadium_entrance_exit_ways :
  entrance_exit_ways = 49 := by sorry

end NUMINAMATH_CALUDE_stadium_entrance_exit_ways_l2795_279582


namespace NUMINAMATH_CALUDE_root_sum_squares_l2795_279584

theorem root_sum_squares (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) →
  (q^3 - 15*q^2 + 25*q - 10 = 0) →
  (r^3 - 15*r^2 + 25*r - 10 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l2795_279584


namespace NUMINAMATH_CALUDE_new_shoe_cost_calculation_l2795_279597

/-- The cost of repairing used shoes -/
def repair_cost : ℝ := 11.50

/-- The duration that repaired shoes last (in years) -/
def repaired_duration : ℝ := 1

/-- The duration that new shoes last (in years) -/
def new_duration : ℝ := 2

/-- The percentage increase in average yearly cost of new shoes compared to repaired shoes -/
def cost_increase_percentage : ℝ := 0.2173913043478261

/-- The cost of purchasing new shoes -/
def new_shoe_cost : ℝ := 2 * (repair_cost + cost_increase_percentage * repair_cost)

theorem new_shoe_cost_calculation :
  new_shoe_cost = 28 :=
sorry

end NUMINAMATH_CALUDE_new_shoe_cost_calculation_l2795_279597


namespace NUMINAMATH_CALUDE_total_amount_proof_l2795_279531

/-- Given that r has two-thirds of the total amount and r has Rs. 2800,
    prove that the total amount p, q, and r have among themselves is Rs. 4200. -/
theorem total_amount_proof (r : ℝ) (total : ℝ) 
    (h1 : r = (2/3) * total) 
    (h2 : r = 2800) : 
  total = 4200 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_proof_l2795_279531


namespace NUMINAMATH_CALUDE_custom_product_of_A_and_B_l2795_279556

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x ≥ 0}

-- Define the custom cartesian product operation
def custom_product (X Y : Set ℝ) : Set ℝ := {x : ℝ | x ∈ X ∪ Y ∧ x ∉ X ∩ Y}

-- Theorem statement
theorem custom_product_of_A_and_B :
  custom_product A B = {x : ℝ | x > 2} := by
  sorry

end NUMINAMATH_CALUDE_custom_product_of_A_and_B_l2795_279556


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_three_numbers_l2795_279522

theorem arithmetic_mean_of_three_numbers (a b c : ℕ) (h : a = 18 ∧ b = 27 ∧ c = 45) : 
  (a + b + c) / 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_three_numbers_l2795_279522


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l2795_279528

-- Define an increasing function on ℝ
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- State the theorem
theorem increasing_function_inequality (f : ℝ → ℝ) (a b : ℝ)
  (h_increasing : IncreasingFunction f) (h_sum_positive : a + b > 0) :
  f a + f b > f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l2795_279528


namespace NUMINAMATH_CALUDE_balls_to_one_pile_l2795_279518

/-- Represents a configuration of piles of balls -/
structure BallConfiguration (n : ℕ) where
  piles : List ℕ
  sum_balls : List.sum piles = 2^n

/-- Represents a move between two piles -/
inductive Move (n : ℕ)
| move : (a b : ℕ) → a ≥ b → a + b ≤ 2^n → Move n

/-- Represents a sequence of moves -/
def MoveSequence (n : ℕ) := List (Move n)

/-- Applies a move to a configuration -/
def applyMove (config : BallConfiguration n) (m : Move n) : BallConfiguration n :=
  sorry

/-- Applies a sequence of moves to a configuration -/
def applyMoveSequence (config : BallConfiguration n) (seq : MoveSequence n) : BallConfiguration n :=
  sorry

/-- Checks if all balls are in one pile -/
def isOnePile (config : BallConfiguration n) : Prop :=
  ∃ p, config.piles = [p]

/-- The main theorem to prove -/
theorem balls_to_one_pile (n : ℕ) (initial : BallConfiguration n) :
  ∃ (seq : MoveSequence n), isOnePile (applyMoveSequence initial seq) :=
sorry

end NUMINAMATH_CALUDE_balls_to_one_pile_l2795_279518


namespace NUMINAMATH_CALUDE_liam_juice_consumption_l2795_279589

/-- Proves that Liam drinks 17 glasses of juice in 5 hours and 40 minutes -/
theorem liam_juice_consumption :
  let minutes_per_glass : ℕ := 20
  let total_minutes : ℕ := 5 * 60 + 40
  let glasses : ℕ := total_minutes / minutes_per_glass
  glasses = 17 := by
  sorry

#check liam_juice_consumption

end NUMINAMATH_CALUDE_liam_juice_consumption_l2795_279589


namespace NUMINAMATH_CALUDE_digit_sequence_sum_value_l2795_279547

def is_increasing (n : ℕ) : Prop := sorry

def is_decreasing (n : ℕ) : Prop := sorry

def digit_sequence_sum : ℕ := sorry

theorem digit_sequence_sum_value : 
  digit_sequence_sum = (80 * 11^10 - 35 * 2^10) / 81 - 45 := by sorry

end NUMINAMATH_CALUDE_digit_sequence_sum_value_l2795_279547


namespace NUMINAMATH_CALUDE_email_difference_l2795_279540

def morning_emails : ℕ := 10
def afternoon_emails : ℕ := 7

theorem email_difference : morning_emails - afternoon_emails = 3 := by
  sorry

end NUMINAMATH_CALUDE_email_difference_l2795_279540


namespace NUMINAMATH_CALUDE_total_dolls_l2795_279578

/-- The number of dolls each person has -/
structure DollCounts where
  lisa : ℕ
  vera : ℕ
  sophie : ℕ
  aida : ℕ

/-- The conditions of the doll ownership -/
def validDollCounts (d : DollCounts) : Prop :=
  d.lisa = 20 ∧
  d.vera = 2 * d.lisa ∧
  d.sophie = 2 * d.vera ∧
  d.aida = 2 * d.sophie

/-- The theorem stating the total number of dolls -/
theorem total_dolls (d : DollCounts) (h : validDollCounts d) :
  d.lisa + d.vera + d.sophie + d.aida = 300 :=
by sorry

end NUMINAMATH_CALUDE_total_dolls_l2795_279578


namespace NUMINAMATH_CALUDE_bridge_length_is_80_l2795_279507

/-- The length of a bridge given train parameters and crossing time -/
def bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  train_speed * crossing_time - train_length

/-- Theorem: The bridge length is 80 meters given the specified conditions -/
theorem bridge_length_is_80 :
  bridge_length 280 18 20 = 80 := by
  sorry

#eval bridge_length 280 18 20

end NUMINAMATH_CALUDE_bridge_length_is_80_l2795_279507


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2795_279576

theorem absolute_value_inequality (x : ℝ) :
  |x - 1| + |x + 2| ≥ 5 ↔ x ≤ -3 ∨ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2795_279576


namespace NUMINAMATH_CALUDE_triangle_properties_l2795_279586

theorem triangle_properties (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  a * Real.sin B = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin C →
  Real.sin A = 2 * Real.sin B →
  a * c = 3/2 * (b^2 + c^2 - a^2) →
  Real.cos A = 1/3 ∧ 
  Real.sin (2*B - A) = (2 * Real.sqrt 14 - 10 * Real.sqrt 2) / 27 := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l2795_279586


namespace NUMINAMATH_CALUDE_problem_statement_l2795_279546

theorem problem_statement (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2795_279546


namespace NUMINAMATH_CALUDE_suraj_average_after_17th_innings_l2795_279545

def average_after_17th_innings (initial_average : ℝ) (score_17th : ℝ) (average_increase : ℝ) : Prop :=
  let total_runs_16 := 16 * initial_average
  let total_runs_17 := total_runs_16 + score_17th
  let new_average := total_runs_17 / 17
  new_average = initial_average + average_increase

theorem suraj_average_after_17th_innings :
  ∃ (initial_average : ℝ),
    average_after_17th_innings initial_average 112 6 ∧
    initial_average + 6 = 16 :=
by
  sorry

#check suraj_average_after_17th_innings

end NUMINAMATH_CALUDE_suraj_average_after_17th_innings_l2795_279545


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2795_279561

/-- Given an arithmetic sequence a with S₃ = 6, prove that 5a₁ + a₇ = 12 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) : 
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  (3 * a 1 + 3 * d = 6) →       -- S₃ = 6 condition
  5 * a 1 + a 7 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2795_279561


namespace NUMINAMATH_CALUDE_hyperbola_center_trajectory_l2795_279552

/-- The hyperbola equation with parameter m -/
def hyperbola (x y m : ℝ) : Prop :=
  x^2 - y^2 - 6*m*x - 4*m*y + 5*m^2 - 1 = 0

/-- The trajectory equation of the center -/
def trajectory_equation (x y : ℝ) : Prop :=
  2*x + 3*y = 0

/-- Theorem stating that the trajectory equation of the center of the hyperbola
    is 2x + 3y = 0 for all real m -/
theorem hyperbola_center_trajectory :
  ∀ m : ℝ, ∃ x y : ℝ, hyperbola x y m ∧ trajectory_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_center_trajectory_l2795_279552


namespace NUMINAMATH_CALUDE_proof_by_contradiction_principle_l2795_279508

theorem proof_by_contradiction_principle :
  ∀ (P : Prop), (¬P → False) → P :=
by
  sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_principle_l2795_279508


namespace NUMINAMATH_CALUDE_complex_simplification_l2795_279523

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The given complex number -/
noncomputable def z : ℂ := (9 + 2 * i) / (2 + i)

/-- The theorem stating that the given complex number equals 4 - i -/
theorem complex_simplification : z = 4 - i := by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2795_279523


namespace NUMINAMATH_CALUDE_children_on_airplane_l2795_279534

/-- Proves that the number of children on an airplane is 20 given specific conditions --/
theorem children_on_airplane (total_passengers : ℕ) (num_men : ℕ) :
  total_passengers = 80 →
  num_men = 30 →
  ∃ (num_women num_children : ℕ),
    num_women = num_men ∧
    num_children = total_passengers - (num_men + num_women) ∧
    num_children = 20 := by
  sorry

end NUMINAMATH_CALUDE_children_on_airplane_l2795_279534


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2795_279548

theorem fraction_multiplication (a b : ℝ) (h : a ≠ b) : 
  (3*a * 3*b) / (3*a - 3*b) = 3 * (a*b / (a - b)) := by
sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2795_279548


namespace NUMINAMATH_CALUDE_incenter_characterization_l2795_279506

/-- Triangle ABC with point P inside -/
structure Triangle :=
  (A B C P : ℝ × ℝ)

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Perpendicular distance from a point to a line segment -/
def perpDistance (point : ℝ × ℝ) (segment : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- Length of a line segment -/
def segmentLength (segment : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- The incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

theorem incenter_characterization (t : Triangle) :
  let l := perimeter t
  let s := area t
  let PD := perpDistance t.P (t.A, t.B)
  let PE := perpDistance t.P (t.B, t.C)
  let PF := perpDistance t.P (t.C, t.A)
  let AB := segmentLength (t.A, t.B)
  let BC := segmentLength (t.B, t.C)
  let CA := segmentLength (t.C, t.A)
  AB / PD + BC / PE + CA / PF ≤ l^2 / (2 * s) →
  t.P = incenter t :=
by sorry

end NUMINAMATH_CALUDE_incenter_characterization_l2795_279506


namespace NUMINAMATH_CALUDE_erased_value_determinable_l2795_279554

-- Define the type for our circle system
structure CircleSystem where
  -- The values in each circle (we'll use Option to represent the erased circle)
  circle_values : Fin 6 → Option ℝ
  -- The values on each segment
  segment_values : Fin 6 → ℝ

-- Define the property that circle values are sums of incoming segment values
def valid_circle_system (cs : CircleSystem) : Prop :=
  ∀ i : Fin 6, 
    cs.circle_values i = some (cs.segment_values i + cs.segment_values ((i + 5) % 6))

-- Define the property that exactly one circle value is erased (None)
def one_erased (cs : CircleSystem) : Prop :=
  ∃! i : Fin 6, cs.circle_values i = none

-- Theorem stating that the erased value can be determined
theorem erased_value_determinable (cs : CircleSystem) 
  (h1 : valid_circle_system cs) (h2 : one_erased cs) : 
  ∃ (x : ℝ), ∀ (cs' : CircleSystem), 
    valid_circle_system cs' → 
    (∀ i : Fin 6, cs.circle_values i ≠ none → cs'.circle_values i = cs.circle_values i) →
    (∀ i : Fin 6, cs'.segment_values i = cs.segment_values i) →
    (∃ i : Fin 6, cs'.circle_values i = some x ∧ cs.circle_values i = none) :=
sorry

end NUMINAMATH_CALUDE_erased_value_determinable_l2795_279554


namespace NUMINAMATH_CALUDE_problem_statement_l2795_279564

theorem problem_statement :
  (¬ (∃ x : ℝ, x^2 - x + 1 < 0)) ∧
  (¬ (∀ x : ℝ, x^2 - 4 ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2795_279564


namespace NUMINAMATH_CALUDE_line_problem_l2795_279587

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3 * x + 2 * y - 1 = 0
def l₂ (x y : ℝ) : Prop := 5 * x + 2 * y + 1 = 0
def l₃ (a x y : ℝ) : Prop := (a^2 - 1) * x + a * y - 1 = 0

-- Define the intersection point A
def A : ℝ × ℝ := (-1, 2)

-- Define parallelism
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, f x y ↔ g (k * x) (k * y)

-- Define a line with equal intercepts
def equal_intercepts (m b : ℝ) : Prop :=
  b / m + b = 0

theorem line_problem :
  (∃ a : ℝ, parallel (l₃ a) l₁ ∧ a = -1/2) ∧
  (∃ m b : ℝ, (m = -1 ∧ b = 1) ∨ (m = -2 ∧ b = 0) ∧
    l₁ A.1 A.2 ∧ l₂ A.1 A.2 ∧ equal_intercepts m b) :=
sorry

end NUMINAMATH_CALUDE_line_problem_l2795_279587


namespace NUMINAMATH_CALUDE_perimeter_circumference_ratio_l2795_279526

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The distance from the center of the circle to the intersection of diagonals -/
  d : ℝ
  /-- Condition that d is 3/5 of r -/
  h_d_ratio : d = 3/5 * r

/-- The perimeter of the trapezoid -/
def perimeter (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ := sorry

/-- The circumference of the inscribed circle -/
def circumference (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ := sorry

theorem perimeter_circumference_ratio 
  (t : IsoscelesTrapezoidWithInscribedCircle) : 
  perimeter t / circumference t = 5 / Real.pi := by sorry

end NUMINAMATH_CALUDE_perimeter_circumference_ratio_l2795_279526


namespace NUMINAMATH_CALUDE_average_weight_after_student_left_l2795_279527

theorem average_weight_after_student_left (initial_count : ℕ) (left_weight : ℝ) 
  (remaining_count : ℕ) (weight_increase : ℝ) (final_average : ℝ) : 
  initial_count = 60 →
  left_weight = 45 →
  remaining_count = 59 →
  weight_increase = 0.2 →
  final_average = 57 →
  (initial_count : ℝ) * (final_average - weight_increase) = 
    (remaining_count : ℝ) * final_average + left_weight := by
  sorry

end NUMINAMATH_CALUDE_average_weight_after_student_left_l2795_279527


namespace NUMINAMATH_CALUDE_least_k_value_l2795_279567

theorem least_k_value (a b c d : ℝ) : 
  ∃ k : ℝ, k = 4 ∧ 
  (∀ a b c d : ℝ, 
    Real.sqrt ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)) + 
    Real.sqrt ((b^2 + 1) * (c^2 + 1) * (d^2 + 1)) + 
    Real.sqrt ((c^2 + 1) * (d^2 + 1) * (a^2 + 1)) + 
    Real.sqrt ((d^2 + 1) * (a^2 + 1) * (b^2 + 1)) ≥ 
    2 * (a*b + b*c + c*d + d*a + a*c + b*d) - k) ∧
  (∀ k' : ℝ, k' < k → 
    ∃ a b c d : ℝ, 
      Real.sqrt ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)) + 
      Real.sqrt ((b^2 + 1) * (c^2 + 1) * (d^2 + 1)) + 
      Real.sqrt ((c^2 + 1) * (d^2 + 1) * (a^2 + 1)) + 
      Real.sqrt ((d^2 + 1) * (a^2 + 1) * (b^2 + 1)) < 
      2 * (a*b + b*c + c*d + d*a + a*c + b*d) - k') :=
by sorry

end NUMINAMATH_CALUDE_least_k_value_l2795_279567


namespace NUMINAMATH_CALUDE_sin_to_cos_shift_l2795_279500

theorem sin_to_cos_shift (x : ℝ) :
  let f : ℝ → ℝ := λ t ↦ Real.sin (t - π/3)
  let g : ℝ → ℝ := λ t ↦ Real.cos t
  f (x + 5*π/6) = g x := by
sorry

end NUMINAMATH_CALUDE_sin_to_cos_shift_l2795_279500


namespace NUMINAMATH_CALUDE_tenth_root_unity_l2795_279579

theorem tenth_root_unity : 
  ∃ (n : ℕ) (h : n < 10), 
    (Complex.tan (Real.pi / 5) + Complex.I) / (Complex.tan (Real.pi / 5) - Complex.I) = 
    Complex.exp (Complex.I * (2 * ↑n * Real.pi / 10)) :=
by sorry

end NUMINAMATH_CALUDE_tenth_root_unity_l2795_279579


namespace NUMINAMATH_CALUDE_logo_enlargement_l2795_279572

/-- Calculates the height of a proportionally enlarged logo -/
def enlargedLogoHeight (originalWidth originalHeight newWidth : ℚ) : ℚ :=
  (newWidth / originalWidth) * originalHeight

/-- Theorem: The enlarged logo height is 8 inches -/
theorem logo_enlargement (originalWidth originalHeight newWidth : ℚ) 
  (h1 : originalWidth = 3)
  (h2 : originalHeight = 2)
  (h3 : newWidth = 12) :
  enlargedLogoHeight originalWidth originalHeight newWidth = 8 := by
  sorry

end NUMINAMATH_CALUDE_logo_enlargement_l2795_279572


namespace NUMINAMATH_CALUDE_pauls_caramel_candy_boxes_l2795_279541

/-- Given that Paul bought 6 boxes of chocolate candy, each box has 9 pieces,
    and he had 90 candies in total, prove that he bought 4 boxes of caramel candy. -/
theorem pauls_caramel_candy_boxes (chocolate_boxes : ℕ) (pieces_per_box : ℕ) (total_candies : ℕ) :
  chocolate_boxes = 6 →
  pieces_per_box = 9 →
  total_candies = 90 →
  (total_candies - chocolate_boxes * pieces_per_box) / pieces_per_box = 4 := by
  sorry

end NUMINAMATH_CALUDE_pauls_caramel_candy_boxes_l2795_279541


namespace NUMINAMATH_CALUDE_cardinality_difference_constant_l2795_279563

/-- Given a finite set of positive integers, S_n is the set of all sums of exactly n elements from the set -/
def S_n (A : Finset Nat) (n : Nat) : Finset Nat :=
  sorry

/-- The main theorem stating the existence of N and k -/
theorem cardinality_difference_constant (A : Finset Nat) :
  ∃ (N k : Nat), ∀ n ≥ N, (S_n A (n + 1)).card = (S_n A n).card + k :=
sorry

end NUMINAMATH_CALUDE_cardinality_difference_constant_l2795_279563


namespace NUMINAMATH_CALUDE_max_intersection_points_l2795_279535

/-- Represents a line in the plane -/
structure Line :=
  (id : ℕ)

/-- The set of all lines -/
def all_lines : Finset Line := sorry

/-- The set of lines that are parallel to each other -/
def parallel_lines : Finset Line := sorry

/-- The set of lines that pass through point B -/
def point_b_lines : Finset Line := sorry

/-- A point of intersection between two lines -/
structure IntersectionPoint :=
  (l1 : Line)
  (l2 : Line)

/-- The set of all intersection points -/
def intersection_points : Finset IntersectionPoint := sorry

theorem max_intersection_points :
  (∀ l ∈ all_lines, l.id ≤ 150) →
  (∀ l ∈ all_lines, ∀ m ∈ all_lines, l ≠ m → l.id ≠ m.id) →
  (Finset.card all_lines = 150) →
  (∀ n : ℕ, n > 0 → parallel_lines.card = 100) →
  (∀ n : ℕ, n > 0 → point_b_lines.card = 50) →
  (∀ l ∈ parallel_lines, ∀ m ∈ parallel_lines, l ≠ m → ¬∃ p : IntersectionPoint, p.l1 = l ∧ p.l2 = m) →
  (∀ l ∈ point_b_lines, ∀ m ∈ point_b_lines, l ≠ m → ∃! p : IntersectionPoint, p.l1 = l ∧ p.l2 = m) →
  (∀ l ∈ parallel_lines, ∀ m ∈ point_b_lines, ∃! p : IntersectionPoint, p.l1 = l ∧ p.l2 = m) →
  Finset.card intersection_points = 5001 :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_points_l2795_279535


namespace NUMINAMATH_CALUDE_mothers_salary_l2795_279529

theorem mothers_salary (mother_salary : ℝ) : 
  let father_salary := 1.3 * mother_salary
  let combined_salary := mother_salary + father_salary
  let method1_savings := (combined_salary / 10) * 6
  let method2_savings := (combined_salary / 2) * (1 + 0.03 * 10)
  method1_savings = method2_savings - 2875 →
  mother_salary = 25000 := by
sorry

end NUMINAMATH_CALUDE_mothers_salary_l2795_279529


namespace NUMINAMATH_CALUDE_josie_cart_wait_time_l2795_279570

/-- Represents the shopping trip details -/
structure ShoppingTrip where
  total_time : ℕ
  shopping_time : ℕ
  wait_cabinet : ℕ
  wait_restock : ℕ
  wait_checkout : ℕ

/-- Calculates the time waited for a cart given a shopping trip -/
def time_waited_for_cart (trip : ShoppingTrip) : ℕ :=
  trip.total_time - trip.shopping_time - (trip.wait_cabinet + trip.wait_restock + trip.wait_checkout)

/-- Theorem stating that Josie waited 3 minutes for a cart -/
theorem josie_cart_wait_time :
  ∃ (trip : ShoppingTrip),
    trip.total_time = 90 ∧
    trip.shopping_time = 42 ∧
    trip.wait_cabinet = 13 ∧
    trip.wait_restock = 14 ∧
    trip.wait_checkout = 18 ∧
    time_waited_for_cart trip = 3 := by
  sorry

end NUMINAMATH_CALUDE_josie_cart_wait_time_l2795_279570


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l2795_279524

theorem loan_principal_calculation (interest_rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ) :
  interest_rate = 12 →
  time = 3 →
  interest = 6480 →
  interest = principal * interest_rate * time / 100 →
  principal = 18000 := by
sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l2795_279524


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2795_279513

/-- Proves that for a quadratic function y = ax^2 + bx + c with integer coefficients,
    if the vertex is at (2, 5) and the point (3, 8) lies on the parabola, then a = 3. -/
theorem quadratic_coefficient (a b c : ℤ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = a * x^2 + b * x + c) →
  (∃ y : ℝ, 5 = a * 2^2 + b * 2 + c ∧ 5 ≥ y) →
  (8 = a * 3^2 + b * 3 + c) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2795_279513


namespace NUMINAMATH_CALUDE_tan_theta_value_l2795_279594

theorem tan_theta_value (θ : ℝ) (h : Real.tan (π / 4 + θ) = 1 / 2) : Real.tan θ = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l2795_279594


namespace NUMINAMATH_CALUDE_kitchen_clock_correct_time_bedroom_clock_correct_time_clocks_same_time_l2795_279512

-- Constants
def minutes_per_hour : ℚ := 60
def hours_per_day : ℚ := 24
def clock_cycle_minutes : ℚ := 720

-- Clock rates
def kitchen_clock_advance_rate : ℚ := 1.5
def bedroom_clock_slow_rate : ℚ := 0.5

-- Theorem for kitchen clock
theorem kitchen_clock_correct_time (t : ℚ) :
  t * kitchen_clock_advance_rate = clock_cycle_minutes →
  t / (hours_per_day * minutes_per_hour) = 20 := by sorry

-- Theorem for bedroom clock
theorem bedroom_clock_correct_time (t : ℚ) :
  t * bedroom_clock_slow_rate = clock_cycle_minutes →
  t / (hours_per_day * minutes_per_hour) = 60 := by sorry

-- Theorem for both clocks showing the same time
theorem clocks_same_time (t : ℚ) :
  t * (kitchen_clock_advance_rate + bedroom_clock_slow_rate) = clock_cycle_minutes →
  t / (hours_per_day * minutes_per_hour) = 15 := by sorry

end NUMINAMATH_CALUDE_kitchen_clock_correct_time_bedroom_clock_correct_time_clocks_same_time_l2795_279512


namespace NUMINAMATH_CALUDE_average_age_combined_l2795_279574

theorem average_age_combined (num_students : ℕ) (num_teachers : ℕ) 
  (avg_age_students : ℚ) (avg_age_teachers : ℚ) :
  num_students = 40 →
  num_teachers = 60 →
  avg_age_students = 13 →
  avg_age_teachers = 42 →
  ((num_students : ℚ) * avg_age_students + (num_teachers : ℚ) * avg_age_teachers) / 
   ((num_students : ℚ) + (num_teachers : ℚ)) = 30.4 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l2795_279574


namespace NUMINAMATH_CALUDE_product_97_103_l2795_279590

theorem product_97_103 : 97 * 103 = 9991 := by
  sorry

end NUMINAMATH_CALUDE_product_97_103_l2795_279590


namespace NUMINAMATH_CALUDE_second_number_existence_and_uniqueness_l2795_279549

theorem second_number_existence_and_uniqueness :
  ∃! x : ℕ, x > 0 ∧ 220070 = (555 + x) * (2 * (x - 555)) + 70 :=
by sorry

end NUMINAMATH_CALUDE_second_number_existence_and_uniqueness_l2795_279549


namespace NUMINAMATH_CALUDE_picnic_watermelon_slices_l2795_279569

/-- The number of watermelons Danny brings -/
def danny_watermelons : ℕ := 3

/-- The number of slices Danny cuts each watermelon into -/
def danny_slices_per_watermelon : ℕ := 10

/-- The number of watermelons Danny's sister brings -/
def sister_watermelons : ℕ := 1

/-- The number of slices Danny's sister cuts her watermelon into -/
def sister_slices_per_watermelon : ℕ := 15

/-- The total number of watermelon slices at the picnic -/
def total_slices : ℕ := danny_watermelons * danny_slices_per_watermelon + sister_watermelons * sister_slices_per_watermelon

theorem picnic_watermelon_slices : total_slices = 45 := by
  sorry

end NUMINAMATH_CALUDE_picnic_watermelon_slices_l2795_279569


namespace NUMINAMATH_CALUDE_max_female_students_theorem_min_group_size_theorem_l2795_279593

/-- Represents the composition of a study group --/
structure StudyGroup where
  male_students : ℕ
  female_students : ℕ
  teachers : ℕ

/-- Checks if a study group satisfies the given conditions --/
def is_valid_group (g : StudyGroup) : Prop :=
  g.male_students > g.female_students ∧
  g.female_students > g.teachers ∧
  2 * g.teachers > g.male_students

/-- The maximum number of female students when there are 4 teachers --/
def max_female_students_with_4_teachers : ℕ := 6

/-- The minimum number of people in a valid study group --/
def min_group_size : ℕ := 12

/-- Theorem: The maximum number of female students is 6 when there are 4 teachers --/
theorem max_female_students_theorem :
  ∀ g : StudyGroup, is_valid_group g → g.teachers = 4 → g.female_students ≤ max_female_students_with_4_teachers :=
sorry

/-- Theorem: The minimum number of people in a valid study group is 12 --/
theorem min_group_size_theorem :
  ∀ g : StudyGroup, is_valid_group g → g.male_students + g.female_students + g.teachers ≥ min_group_size :=
sorry

end NUMINAMATH_CALUDE_max_female_students_theorem_min_group_size_theorem_l2795_279593


namespace NUMINAMATH_CALUDE_xor_inequality_iff_even_l2795_279573

-- Define bitwise XOR operation
def bitwise_xor (a b : ℕ) : ℕ := sorry

-- Define the property that needs to be proven
def xor_inequality_property (a : ℕ) : Prop :=
  ∀ x y : ℕ, x > y → y ≥ 0 → bitwise_xor x (a * x) ≠ bitwise_xor y (a * y)

-- Theorem statement
theorem xor_inequality_iff_even (a : ℕ) :
  a > 0 → (xor_inequality_property a ↔ Even a) :=
sorry

end NUMINAMATH_CALUDE_xor_inequality_iff_even_l2795_279573


namespace NUMINAMATH_CALUDE_translation_theorem_l2795_279532

/-- A translation in 2D space -/
structure Translation (α : Type*) [AddGroup α] where
  dx : α
  dy : α

/-- Apply a translation to a point -/
def applyTranslation (t : Translation ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + t.dx, p.2 + t.dy)

theorem translation_theorem (A A₁ D : ℝ × ℝ) (h : A = (1, -2) ∧ A₁ = (-1, 3)) :
  ∃ (t : Translation ℝ), applyTranslation t A = A₁ ∧ 
    applyTranslation t D = (D.1 - 2, D.2 + 5) :=
sorry

end NUMINAMATH_CALUDE_translation_theorem_l2795_279532


namespace NUMINAMATH_CALUDE_robbie_rice_solution_l2795_279505

/-- Robbie's daily rice consumption and fat intake --/
def robbie_rice_problem (x : ℝ) : Prop :=
  let morning_rice := x
  let afternoon_rice := 2
  let evening_rice := 5
  let fat_per_cup := 10
  let weekly_fat := 700
  let daily_rice := morning_rice + afternoon_rice + evening_rice
  let daily_fat := daily_rice * fat_per_cup
  daily_fat * 7 = weekly_fat

/-- The solution to Robbie's rice consumption problem --/
theorem robbie_rice_solution :
  ∃ x : ℝ, robbie_rice_problem x ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_robbie_rice_solution_l2795_279505


namespace NUMINAMATH_CALUDE_dot_product_when_x_negative_one_parallel_vectors_when_x_eight_l2795_279530

def a : ℝ × ℝ := (4, 7)
def b (x : ℝ) : ℝ × ℝ := (x, x + 6)

theorem dot_product_when_x_negative_one :
  (a.1 * (b (-1)).1 + a.2 * (b (-1)).2) = 31 := by sorry

theorem parallel_vectors_when_x_eight :
  (a.1 / (b 8).1 = a.2 / (b 8).2) := by sorry

end NUMINAMATH_CALUDE_dot_product_when_x_negative_one_parallel_vectors_when_x_eight_l2795_279530


namespace NUMINAMATH_CALUDE_ricas_prize_fraction_l2795_279595

theorem ricas_prize_fraction (total_prize : ℚ) (rica_remaining : ℚ) :
  total_prize = 1000 →
  rica_remaining = 300 →
  ∃ (rica_fraction : ℚ),
    rica_fraction * total_prize * (4/5) = rica_remaining ∧
    rica_fraction = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_ricas_prize_fraction_l2795_279595


namespace NUMINAMATH_CALUDE_two_different_color_chips_probability_l2795_279509

/-- The probability of selecting two chips of different colors from a bag with replacement -/
theorem two_different_color_chips_probability
  (blue : ℕ) (red : ℕ) (yellow : ℕ)
  (h_blue : blue = 6)
  (h_red : red = 5)
  (h_yellow : yellow = 4) :
  let total := blue + red + yellow
  let prob_blue := blue / total
  let prob_red := red / total
  let prob_yellow := yellow / total
  let prob_not_blue := (red + yellow) / total
  let prob_not_red := (blue + yellow) / total
  let prob_not_yellow := (blue + red) / total
  prob_blue * prob_not_blue + prob_red * prob_not_red + prob_yellow * prob_not_yellow = 148 / 225 := by
  sorry

end NUMINAMATH_CALUDE_two_different_color_chips_probability_l2795_279509


namespace NUMINAMATH_CALUDE_horner_method_operations_l2795_279565

/-- The number of arithmetic operations required to evaluate a polynomial using Horner's method -/
def horner_operations (n : ℕ) : ℕ := 2 * n

/-- Theorem: For a polynomial of degree n, Horner's method requires 2n arithmetic operations -/
theorem horner_method_operations (n : ℕ) :
  horner_operations n = 2 * n :=
by sorry

end NUMINAMATH_CALUDE_horner_method_operations_l2795_279565


namespace NUMINAMATH_CALUDE_bags_sold_thursday_l2795_279580

/-- Calculates the number of bags sold on Thursday given the total stock and sales on other days --/
theorem bags_sold_thursday (total_stock : ℕ) (monday_sales tuesday_sales wednesday_sales friday_sales : ℕ)
  (h1 : total_stock = 600)
  (h2 : monday_sales = 25)
  (h3 : tuesday_sales = 70)
  (h4 : wednesday_sales = 100)
  (h5 : friday_sales = 145)
  (h6 : (total_stock : ℚ) * (25 : ℚ) / 100 = total_stock - (monday_sales + tuesday_sales + wednesday_sales + friday_sales + 110)) :
  110 = total_stock - (monday_sales + tuesday_sales + wednesday_sales + friday_sales + (total_stock : ℚ) * (25 : ℚ) / 100) :=
by sorry

end NUMINAMATH_CALUDE_bags_sold_thursday_l2795_279580


namespace NUMINAMATH_CALUDE_shells_found_l2795_279519

def initial_shells : ℕ := 68
def final_shells : ℕ := 89

theorem shells_found (initial : ℕ) (final : ℕ) (h1 : initial = initial_shells) (h2 : final = final_shells) :
  final - initial = 21 := by
  sorry

end NUMINAMATH_CALUDE_shells_found_l2795_279519


namespace NUMINAMATH_CALUDE_simplify_exponents_l2795_279533

theorem simplify_exponents (t : ℝ) : (t^4 * t^5) * (t^2)^2 = t^13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponents_l2795_279533


namespace NUMINAMATH_CALUDE_only_traffic_light_is_random_l2795_279568

/-- Represents a phenomenon that can be observed --/
inductive Phenomenon
  | WaterBoiling : Phenomenon
  | TrafficLight : Phenomenon
  | RectangleArea : Phenomenon
  | LinearEquation : Phenomenon

/-- Determines if a phenomenon is random --/
def isRandom (p : Phenomenon) : Prop :=
  match p with
  | Phenomenon.TrafficLight => True
  | _ => False

/-- Theorem stating that only the traffic light phenomenon is random --/
theorem only_traffic_light_is_random :
  ∀ (p : Phenomenon), isRandom p ↔ p = Phenomenon.TrafficLight := by
  sorry


end NUMINAMATH_CALUDE_only_traffic_light_is_random_l2795_279568


namespace NUMINAMATH_CALUDE_S_when_m_is_one_l_range_when_m_is_neg_half_m_range_when_l_is_half_l2795_279510

-- Define the set S
def S (m l : ℝ) : Set ℝ := {x : ℝ | m ≤ x ∧ x ≤ l}

-- State the condition that if x ∈ S, then x^2 ∈ S
axiom S_closed_square (m l : ℝ) : ∀ x ∈ S m l, x^2 ∈ S m l

-- Theorem 1
theorem S_when_m_is_one (l : ℝ) : 
  S 1 l = {1} := by sorry

-- Theorem 2
theorem l_range_when_m_is_neg_half : 
  ∀ l, S (-1/2) l ≠ ∅ ↔ 1/4 ≤ l ∧ l ≤ 1 := by sorry

-- Theorem 3
theorem m_range_when_l_is_half : 
  ∀ m, S m (1/2) ≠ ∅ ↔ -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_S_when_m_is_one_l_range_when_m_is_neg_half_m_range_when_l_is_half_l2795_279510


namespace NUMINAMATH_CALUDE_unique_solution_l2795_279517

theorem unique_solution : ∃! (x y z : ℤ),
  (y^4 + 2*z^2) % 3 = 2 ∧
  (3*x^4 + z^2) % 5 = 1 ∧
  y^4 + 2*z^2 = 3*x^4 + z^2 - 6 ∧
  x = 5 ∧ y = 3 ∧ z = 19 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l2795_279517


namespace NUMINAMATH_CALUDE_brittany_test_average_l2795_279550

def test_average (score1 : ℚ) (score2 : ℚ) : ℚ :=
  (score1 + score2) / 2

theorem brittany_test_average :
  test_average 78 84 = 81 := by
  sorry

end NUMINAMATH_CALUDE_brittany_test_average_l2795_279550


namespace NUMINAMATH_CALUDE_cookies_per_box_l2795_279521

/-- Proof of the number of cookies per box in Brenda's banana pudding problem -/
theorem cookies_per_box 
  (num_trays : ℕ) 
  (cookies_per_tray : ℕ) 
  (cost_per_box : ℚ) 
  (total_cost : ℚ) 
  (h1 : num_trays = 3)
  (h2 : cookies_per_tray = 80)
  (h3 : cost_per_box = 7/2)
  (h4 : total_cost = 14) :
  (num_trays * cookies_per_tray) / (total_cost / cost_per_box) = 60 := by
  sorry

#eval (3 * 80) / (14 / (7/2)) -- Should evaluate to 60

end NUMINAMATH_CALUDE_cookies_per_box_l2795_279521


namespace NUMINAMATH_CALUDE_mini_football_betting_strategy_l2795_279515

theorem mini_football_betting_strategy :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧
    x₁ + x₂ + x₃ + x₄ = 1 ∧
    3 * x₁ ≥ 1 ∧
    4 * x₂ ≥ 1 ∧
    5 * x₃ ≥ 1 ∧
    8 * x₄ ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_mini_football_betting_strategy_l2795_279515


namespace NUMINAMATH_CALUDE_a5_is_zero_in_825_factorial_base_l2795_279558

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def factorialBaseCoeff (n k : ℕ) : ℕ :=
  (n / factorial k) % (k + 1)

theorem a5_is_zero_in_825_factorial_base : 
  factorialBaseCoeff 825 5 = 0 := by sorry

end NUMINAMATH_CALUDE_a5_is_zero_in_825_factorial_base_l2795_279558


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2795_279520

def M : ℕ := 18 * 18 * 125 * 210

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 14 = sum_even_divisors M := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2795_279520


namespace NUMINAMATH_CALUDE_school_supplies_cost_l2795_279525

/-- Calculates the total cost of school supplies for a class after applying a discount -/
def total_cost_after_discount (num_students : ℕ) 
                               (num_pens num_notebooks num_binders num_highlighters : ℕ)
                               (cost_pen cost_notebook cost_binder cost_highlighter : ℚ)
                               (discount : ℚ) : ℚ :=
  let cost_per_student := num_pens * cost_pen + 
                          num_notebooks * cost_notebook + 
                          num_binders * cost_binder + 
                          num_highlighters * cost_highlighter
  let total_cost := num_students * cost_per_student
  total_cost - discount

/-- Theorem stating the total cost of school supplies after discount -/
theorem school_supplies_cost :
  total_cost_after_discount 30 5 3 1 2 0.5 1.25 4.25 0.75 100 = 260 :=
by sorry

end NUMINAMATH_CALUDE_school_supplies_cost_l2795_279525


namespace NUMINAMATH_CALUDE_final_value_of_A_l2795_279596

/-- Given an initial value of A and an operation, prove the final value of A -/
theorem final_value_of_A (initial_A : Int) : 
  let A₁ := initial_A
  let A₂ := -A₁ + 10
  A₂ = -10 :=
by sorry

end NUMINAMATH_CALUDE_final_value_of_A_l2795_279596


namespace NUMINAMATH_CALUDE_point_inside_circle_a_range_l2795_279577

theorem point_inside_circle_a_range :
  ∀ a : ℝ,
  (((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1)) :=
by sorry

end NUMINAMATH_CALUDE_point_inside_circle_a_range_l2795_279577


namespace NUMINAMATH_CALUDE_survey_students_l2795_279551

theorem survey_students (total_allowance : ℚ) 
  (h1 : total_allowance = 320)
  (h2 : (2 : ℚ) / 3 * 6 + (1 : ℚ) / 3 * 4 = 16 / 3) : 
  ∃ (num_students : ℕ), num_students * (16 : ℚ) / 3 = total_allowance ∧ num_students = 60 := by
  sorry

end NUMINAMATH_CALUDE_survey_students_l2795_279551


namespace NUMINAMATH_CALUDE_sequence_problem_l2795_279566

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem sequence_problem (a b : ℕ → ℝ) 
    (h_geo : geometric_sequence a)
    (h_arith : arithmetic_sequence b)
    (h_a : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
    (h_b : b 1 + b 6 + b 11 = 7 * Real.pi) :
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l2795_279566


namespace NUMINAMATH_CALUDE_simplify_expression_l2795_279581

theorem simplify_expression (y : ℝ) : 3*y + 4*y^2 + 2 - (5 - (3*y + 4*y^2) - 8) = 8*y^2 + 6*y + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2795_279581


namespace NUMINAMATH_CALUDE_angle_FDE_l2795_279544

theorem angle_FDE (BAC : Real) (h : BAC = 70) : ∃ FDE : Real, FDE = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_FDE_l2795_279544


namespace NUMINAMATH_CALUDE_sum_of_base9_series_l2795_279511

/-- Converts a base 9 number to base 10 -/
def base9ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 9 -/
def base10ToBase9 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of an arithmetic series in base 10 -/
def arithmeticSeriesSum (n : ℕ) (a1 : ℕ) (an : ℕ) : ℕ := sorry

theorem sum_of_base9_series :
  let n : ℕ := 36
  let a1 : ℕ := base9ToBase10 1
  let an : ℕ := base9ToBase10 36
  let sum : ℕ := arithmeticSeriesSum n a1 an
  base10ToBase9 sum = 750 := by sorry

end NUMINAMATH_CALUDE_sum_of_base9_series_l2795_279511


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2795_279575

theorem system_solution_ratio (x y z : ℝ) (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  let k : ℝ := 95 / 12
  (x + k * y + 4 * z = 0) →
  (4 * x + k * y - 3 * z = 0) →
  (3 * x + 5 * y - 4 * z = 0) →
  x^2 * z / y^3 = -60 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2795_279575


namespace NUMINAMATH_CALUDE_oula_deliveries_l2795_279502

/-- Proves that Oula made 96 deliveries given the problem conditions -/
theorem oula_deliveries :
  ∀ (oula_deliveries tona_deliveries : ℕ) 
    (pay_per_delivery : ℕ) 
    (pay_difference : ℕ),
  pay_per_delivery = 100 →
  tona_deliveries = 3 * oula_deliveries / 4 →
  pay_difference = 2400 →
  pay_per_delivery * oula_deliveries - pay_per_delivery * tona_deliveries = pay_difference →
  oula_deliveries = 96 := by
sorry

end NUMINAMATH_CALUDE_oula_deliveries_l2795_279502


namespace NUMINAMATH_CALUDE_volleyball_tournament_winner_l2795_279504

/-- Represents a volleyball tournament -/
structure VolleyballTournament where
  /-- The number of teams in the tournament -/
  num_teams : ℕ
  /-- The number of games each team plays -/
  games_per_team : ℕ
  /-- The total number of games played in the tournament -/
  total_games : ℕ
  /-- There are no draws in the tournament -/
  no_draws : Bool

/-- Theorem stating that in a volleyball tournament with 6 teams, 
    where each team plays against every other team once and there are no draws, 
    at least one team must win 3 or more games -/
theorem volleyball_tournament_winner (t : VolleyballTournament) 
  (h1 : t.num_teams = 6)
  (h2 : t.games_per_team = 5)
  (h3 : t.total_games = t.num_teams * t.games_per_team / 2)
  (h4 : t.no_draws = true) :
  ∃ (team : ℕ), team ≤ t.num_teams ∧ (∃ (wins : ℕ), wins ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_volleyball_tournament_winner_l2795_279504


namespace NUMINAMATH_CALUDE_min_variance_sum_l2795_279514

theorem min_variance_sum (a b c : ℕ) : 
  70 ≤ a ∧ a < 80 →
  80 ≤ b ∧ b < 90 →
  90 ≤ c ∧ c ≤ 100 →
  let variance := (a - (a + b + c) / 3)^2 + (b - (a + b + c) / 3)^2 + (c - (a + b + c) / 3)^2
  (∀ a' b' c' : ℕ, 
    70 ≤ a' ∧ a' < 80 →
    80 ≤ b' ∧ b' < 90 →
    90 ≤ c' ∧ c' ≤ 100 →
    variance ≤ (a' - (a' + b' + c') / 3)^2 + (b' - (a' + b' + c') / 3)^2 + (c' - (a' + b' + c') / 3)^2) →
  a + b + c = 253 ∨ a + b + c = 254 :=
sorry

end NUMINAMATH_CALUDE_min_variance_sum_l2795_279514


namespace NUMINAMATH_CALUDE_maximize_gcd_sum_1998_l2795_279559

theorem maximize_gcd_sum_1998 : ∃ (a b c : ℕ+),
  (a + b + c : ℕ) = 1998 ∧
  (∀ (x y z : ℕ+), (x + y + z : ℕ) = 1998 → Nat.gcd (Nat.gcd a.val b.val) c.val ≥ Nat.gcd (Nat.gcd x.val y.val) z.val) ∧
  (0 < a.val ∧ a.val < b.val ∧ b.val ≤ c.val ∧ c.val < 2 * a.val) ∧
  ((a.val, b.val, c.val) = (518, 592, 888) ∨
   (a.val, b.val, c.val) = (518, 666, 814) ∨
   (a.val, b.val, c.val) = (518, 740, 740) ∨
   (a.val, b.val, c.val) = (592, 666, 740)) :=
by sorry

end NUMINAMATH_CALUDE_maximize_gcd_sum_1998_l2795_279559


namespace NUMINAMATH_CALUDE_last_item_to_second_recipient_l2795_279516

/-- Represents the cyclic distribution of items among recipients. -/
def cyclicDistribution (items : ℕ) (recipients : ℕ) : ℕ :=
  (items - 1) % recipients + 1

/-- Theorem stating that in a cyclic distribution of 278 items among 6 recipients,
    the 2nd recipient in the initial order receives the last item. -/
theorem last_item_to_second_recipient :
  cyclicDistribution 278 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_last_item_to_second_recipient_l2795_279516


namespace NUMINAMATH_CALUDE_greatest_even_perfect_square_under_200_l2795_279539

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

theorem greatest_even_perfect_square_under_200 :
  ∀ n : ℕ, is_perfect_square n → is_even n → n < 200 → n ≤ 196 :=
sorry

end NUMINAMATH_CALUDE_greatest_even_perfect_square_under_200_l2795_279539


namespace NUMINAMATH_CALUDE_decimal_to_base5_250_l2795_279538

/-- Converts a base-10 number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ := sorry

theorem decimal_to_base5_250 :
  toBase5 250 = [2, 0, 0, 0] := by sorry

end NUMINAMATH_CALUDE_decimal_to_base5_250_l2795_279538


namespace NUMINAMATH_CALUDE_range_of_m_l2795_279555

-- Define the sets A and B
def A : Set ℝ := {x | x < 1}
def B (m : ℝ) : Set ℝ := {x | x > m}

-- State the theorem
theorem range_of_m (m : ℝ) : (Set.compl A) ⊆ B m → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2795_279555


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2795_279583

theorem quadratic_inequality_solution (x : ℝ) : 
  3 * x^2 + 4 * x - 9 < 0 ∧ x ≥ -2 → -2 ≤ x ∧ x < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2795_279583

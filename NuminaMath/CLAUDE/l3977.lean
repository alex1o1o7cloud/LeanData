import Mathlib

namespace NUMINAMATH_CALUDE_boat_downstream_distance_l3977_397740

/-- Proof of downstream distance traveled by a boat given upstream travel time and distance, and stream speed. -/
theorem boat_downstream_distance
  (upstream_distance : ℝ)
  (upstream_time : ℝ)
  (stream_speed : ℝ)
  (downstream_time : ℝ)
  (h1 : upstream_distance = 75)
  (h2 : upstream_time = 15)
  (h3 : stream_speed = 3.75)
  (h4 : downstream_time = 8) :
  let upstream_speed := upstream_distance / upstream_time
  let boat_speed := upstream_speed + stream_speed
  let downstream_speed := boat_speed + stream_speed
  downstream_speed * downstream_time = 100 := by
  sorry


end NUMINAMATH_CALUDE_boat_downstream_distance_l3977_397740


namespace NUMINAMATH_CALUDE_max_leftover_candy_exists_max_leftover_candy_l3977_397704

theorem max_leftover_candy (x : ℕ) : x % 11 ≤ 10 := by sorry

theorem exists_max_leftover_candy : ∃ x : ℕ, x % 11 = 10 := by sorry

end NUMINAMATH_CALUDE_max_leftover_candy_exists_max_leftover_candy_l3977_397704


namespace NUMINAMATH_CALUDE_nested_radical_value_l3977_397739

/-- The value of the infinite nested radical sqrt(3 - sqrt(3 - sqrt(3 - ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt 3))))))

/-- Theorem stating that the nested radical equals (-1 + sqrt(13)) / 2 -/
theorem nested_radical_value : nestedRadical = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l3977_397739


namespace NUMINAMATH_CALUDE_james_weekly_income_l3977_397735

/-- Calculates the weekly income from car rental given hourly rate, hours per day, and days per week. -/
def weekly_income (hourly_rate : ℝ) (hours_per_day : ℝ) (days_per_week : ℝ) : ℝ :=
  hourly_rate * hours_per_day * days_per_week

/-- Proves that James' weekly income from car rental is $640 given the specified conditions. -/
theorem james_weekly_income :
  let hourly_rate : ℝ := 20
  let hours_per_day : ℝ := 8
  let days_per_week : ℝ := 4
  weekly_income hourly_rate hours_per_day days_per_week = 640 := by
  sorry

#eval weekly_income 20 8 4

end NUMINAMATH_CALUDE_james_weekly_income_l3977_397735


namespace NUMINAMATH_CALUDE_system_solutions_l3977_397761

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  x + y + z = 17 ∧ x*y + y*z + z*x = 94 ∧ x*y*z = 168

-- Define the set of solutions
def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1, 4, -12), (1, -12, 4), (4, 1, -12), (4, -12, 1), (-12, 1, 4), (-12, 4, 1)}

-- Theorem statement
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x, y, z) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l3977_397761


namespace NUMINAMATH_CALUDE_parallel_resistors_combined_resistance_l3977_397750

/-- The combined resistance of two resistors connected in parallel -/
def combined_resistance (r1 r2 : ℚ) : ℚ :=
  1 / (1 / r1 + 1 / r2)

/-- Theorem: The combined resistance of two resistors with 8 ohms and 9 ohms connected in parallel is 72/17 ohms -/
theorem parallel_resistors_combined_resistance :
  combined_resistance 8 9 = 72 / 17 := by
  sorry

end NUMINAMATH_CALUDE_parallel_resistors_combined_resistance_l3977_397750


namespace NUMINAMATH_CALUDE_distance_O_to_J_l3977_397711

/-- A right triangle with its circumcircle and incircle -/
structure RightTriangleWithCircles where
  /-- The center of the circumcircle -/
  O : ℝ × ℝ
  /-- The center of the incircle -/
  I : ℝ × ℝ
  /-- The radius of the circumcircle -/
  R : ℝ
  /-- The radius of the incircle -/
  r : ℝ
  /-- The vertex of the right angle -/
  C : ℝ × ℝ
  /-- The point symmetric to C with respect to I -/
  J : ℝ × ℝ
  /-- Ensure that C is the right angle vertex -/
  right_angle : (C.1 - O.1)^2 + (C.2 - O.2)^2 = R^2
  /-- Ensure that J is symmetric to C with respect to I -/
  symmetry : J.1 - I.1 = I.1 - C.1 ∧ J.2 - I.2 = I.2 - C.2

/-- The theorem to be proved -/
theorem distance_O_to_J (t : RightTriangleWithCircles) : 
  ((t.O.1 - t.J.1)^2 + (t.O.2 - t.J.2)^2)^(1/2) = t.R - 2 * t.r := by
  sorry

end NUMINAMATH_CALUDE_distance_O_to_J_l3977_397711


namespace NUMINAMATH_CALUDE_units_digit_problem_l3977_397727

theorem units_digit_problem : (8 * 25 * 983 - 8^3) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l3977_397727


namespace NUMINAMATH_CALUDE_coefficient_of_x_fourth_l3977_397795

theorem coefficient_of_x_fourth (x : ℝ) : 
  let expr := 5*(x^4 - 2*x^5) + 3*(x^2 - 3*x^4 + 2*x^6) - (2*x^5 - 3*x^4)
  ∃ (a b c d e f : ℝ), expr = a*x^6 + b*x^5 + (-1)*x^4 + d*x^3 + e*x^2 + f*x + c :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_fourth_l3977_397795


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3977_397714

/-- Given an arithmetic sequence {a_n} where a_4 = 4, prove that S_7 = 28 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n : ℝ) / 2 * (a 1 + a n)) →  -- Definition of S_n
  (∀ k m, a (k + m) - a k = m * (a 2 - a 1)) →  -- Definition of arithmetic sequence
  a 4 = 4 →  -- Given condition
  S 7 = 28 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3977_397714


namespace NUMINAMATH_CALUDE_function_properties_l3977_397733

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 3^x else m - x^2

theorem function_properties :
  (∀ m < 0, ¬ ∃ x, f m x = 0) ∧
  f (1/9) (f (1/9) (-1)) = 0 := by sorry

end NUMINAMATH_CALUDE_function_properties_l3977_397733


namespace NUMINAMATH_CALUDE_condition_for_inequality_l3977_397741

theorem condition_for_inequality (a b c : ℝ) :
  (¬ (∀ c, a > b → a * c^2 > b * c^2)) ∧
  ((a * c^2 > b * c^2) → a > b) :=
by sorry

end NUMINAMATH_CALUDE_condition_for_inequality_l3977_397741


namespace NUMINAMATH_CALUDE_roots_positive_implies_b_in_range_l3977_397792

/-- A quadratic function f(x) = x² - 2x + b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + b

/-- The discriminant of f(x) -/
def discriminant (b : ℝ) : ℝ := 4 - 4*b

theorem roots_positive_implies_b_in_range (b : ℝ) :
  (∀ x : ℝ, f b x = 0 → x > 0) →
  0 < b ∧ b ≤ 1 := by sorry

end NUMINAMATH_CALUDE_roots_positive_implies_b_in_range_l3977_397792


namespace NUMINAMATH_CALUDE_sum_of_factors_of_125_l3977_397715

theorem sum_of_factors_of_125 :
  ∃ (a b c : ℕ+),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a.val * b.val * c.val = 125) ∧
    (a.val + b.val + c.val = 31) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_of_125_l3977_397715


namespace NUMINAMATH_CALUDE_outfit_combinations_l3977_397748

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) : shirts = 5 → pants = 3 → shirts * pants = 15 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l3977_397748


namespace NUMINAMATH_CALUDE_bird_cost_problem_l3977_397757

/-- Calculates the cost per bird given the total money and number of birds -/
def cost_per_bird (total_money : ℚ) (num_birds : ℕ) : ℚ :=
  total_money / num_birds

/-- The problem statement -/
theorem bird_cost_problem :
  let total_money : ℚ := 4 * 50
  let total_wings : ℕ := 20
  let wings_per_bird : ℕ := 2
  let num_birds : ℕ := total_wings / wings_per_bird
  cost_per_bird total_money num_birds = 20 := by
  sorry

end NUMINAMATH_CALUDE_bird_cost_problem_l3977_397757


namespace NUMINAMATH_CALUDE_shooting_range_problem_l3977_397756

theorem shooting_range_problem :
  ∀ (total_targets : ℕ) 
    (red_targets green_targets : ℕ) 
    (red_score green_score : ℚ)
    (hit_red_targets : ℕ),
  total_targets = 100 →
  total_targets = red_targets + green_targets →
  red_targets < green_targets / 3 →
  red_score = 10 →
  green_score = 8.5 →
  (green_score * green_targets + red_score * hit_red_targets : ℚ) = 
    (green_score * green_targets + red_score * red_targets : ℚ) →
  red_targets = 20 := by
sorry

end NUMINAMATH_CALUDE_shooting_range_problem_l3977_397756


namespace NUMINAMATH_CALUDE_hcf_problem_l3977_397791

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 1991) (h2 : Nat.lcm a b = 181) :
  Nat.gcd a b = 11 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l3977_397791


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3977_397754

-- Define the space we're working in
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define lines and planes
def Line (V : Type*) [NormedAddCommGroup V] := V → Prop
def Plane (V : Type*) [NormedAddCommGroup V] := V → Prop

-- Define perpendicular relation between a line and a plane
def Perpendicular (l : Line V) (p : Plane V) : Prop := sorry

-- Define parallel relation between two lines
def Parallel (l1 l2 : Line V) : Prop := sorry

-- Theorem statement
theorem perpendicular_lines_parallel 
  (m n : Line V) (α : Plane V) 
  (hm : m ≠ n) 
  (h1 : Perpendicular m α) 
  (h2 : Perpendicular n α) : 
  Parallel m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3977_397754


namespace NUMINAMATH_CALUDE_max_guests_left_l3977_397708

/-- Represents a guest with their galoshes size -/
structure Guest where
  size : ℕ

/-- Represents the state of galoshes in the hallway -/
structure GaloshesState where
  sizes : Finset ℕ

/-- Defines when a guest can wear a pair of galoshes -/
def canWear (g : Guest) (s : ℕ) : Prop := g.size ≤ s

/-- Defines the initial state with 10 guests and their galoshes -/
def initialState : Finset Guest × GaloshesState :=
  sorry

/-- Simulates guests leaving and wearing galoshes -/
def guestsLeave (state : Finset Guest × GaloshesState) : Finset Guest × GaloshesState :=
  sorry

/-- Checks if any remaining guest can wear any remaining galoshes -/
def canAnyGuestLeave (state : Finset Guest × GaloshesState) : Prop :=
  sorry

theorem max_guests_left (final_state : Finset Guest × GaloshesState) :
  final_state = guestsLeave initialState →
  ¬canAnyGuestLeave final_state →
  Finset.card final_state.1 ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_guests_left_l3977_397708


namespace NUMINAMATH_CALUDE_largest_x_abs_equation_l3977_397775

theorem largest_x_abs_equation : 
  ∀ x : ℝ, |x - 3| = 14.5 → x ≤ 17.5 ∧ ∃ y : ℝ, |y - 3| = 14.5 ∧ y = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_abs_equation_l3977_397775


namespace NUMINAMATH_CALUDE_quadratic_not_through_point_l3977_397762

def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

theorem quadratic_not_through_point (p q : ℝ) :
  f p q 1 = 1 → f p q 3 = 1 → f p q 4 ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_not_through_point_l3977_397762


namespace NUMINAMATH_CALUDE_infinitely_many_losing_positions_l3977_397779

/-- The set of numbers from which the first player loses -/
def losingSet : Set ℕ := sorry

/-- A number is a winning position if it's not in the losing set -/
def winningPosition (n : ℕ) : Prop := n ∉ losingSet

/-- A perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- The property that defines a losing position -/
def isLosingPosition (n : ℕ) : Prop :=
  ∀ k : ℕ, isPerfectSquare k → k ≤ n → winningPosition (n - k)

/-- The main theorem: there are infinitely many losing positions -/
theorem infinitely_many_losing_positions :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ isLosingPosition n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_losing_positions_l3977_397779


namespace NUMINAMATH_CALUDE_parallelogram_side_lengths_l3977_397728

/-- A parallelogram with the given properties has sides of length 4 and 12 -/
theorem parallelogram_side_lengths 
  (perimeter : ℝ) 
  (triangle_perimeter_diff : ℝ) 
  (h_perimeter : perimeter = 32) 
  (h_diff : triangle_perimeter_diff = 8) :
  ∃ (a b : ℝ), a + b = perimeter / 2 ∧ b - a = triangle_perimeter_diff ∧ a = 4 ∧ b = 12 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_side_lengths_l3977_397728


namespace NUMINAMATH_CALUDE_simplify_expression_l3977_397771

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (15 * x^2) * (6 * x) * (1 / (3 * x)^2) = 10 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3977_397771


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3977_397700

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (5 + 11 - 7) = Real.sqrt (5 + 11) - Real.sqrt x → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3977_397700


namespace NUMINAMATH_CALUDE_orange_purchase_calculation_l3977_397718

/-- The amount of oranges initially planned to be purchased -/
def initial_purchase : ℝ := sorry

/-- The total amount of oranges purchased over three weeks -/
def total_purchase : ℝ := 75

theorem orange_purchase_calculation :
  initial_purchase = 14 :=
by
  have week1 : ℝ := initial_purchase + 5
  have week2 : ℝ := 2 * initial_purchase
  have week3 : ℝ := 2 * initial_purchase
  have total_equation : week1 + week2 + week3 = total_purchase := by sorry
  sorry

end NUMINAMATH_CALUDE_orange_purchase_calculation_l3977_397718


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_three_l3977_397726

theorem greatest_two_digit_multiple_of_three : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 3 = 0 → n ≤ 99 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_three_l3977_397726


namespace NUMINAMATH_CALUDE_circle_area_after_radius_multiplication_area_of_new_circle_l3977_397721

/-- Theorem: Area of a circle after radius multiplication -/
theorem circle_area_after_radius_multiplication (A : ℝ) (k : ℝ) :
  A > 0 → k > 0 → (k * (A / Real.pi).sqrt)^2 * Real.pi = k^2 * A := by
  sorry

/-- The area of a circle with radius multiplied by 5 -/
theorem area_of_new_circle (original_area : ℝ) (new_area : ℝ) :
  original_area = 30 →
  new_area = (5 * (original_area / Real.pi).sqrt)^2 * Real.pi →
  new_area = 750 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_after_radius_multiplication_area_of_new_circle_l3977_397721


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l3977_397766

theorem quadratic_equation_proof :
  ∃ (x y : ℝ), x + y = 10 ∧ |x - y| = 6 ∧ x^2 - 10*x + 16 = 0 ∧ y^2 - 10*y + 16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l3977_397766


namespace NUMINAMATH_CALUDE_min_xy_value_l3977_397749

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) :
  ∃ (x₀ y₀ : ℕ+), (1 : ℚ) / x₀ + (1 : ℚ) / (3 * y₀) = (1 : ℚ) / 6 ∧
    x₀.val * y₀.val = 48 ∧
    ∀ (a b : ℕ+), (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6 →
      x₀.val * y₀.val ≤ a.val * b.val :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l3977_397749


namespace NUMINAMATH_CALUDE_a_2012_value_l3977_397770

-- Define the sequence
def a : ℕ → ℚ
  | 0 => 2
  | (n + 1) => a n / (1 + a n)

-- State the theorem
theorem a_2012_value : a 2012 = 2 / 4025 := by
  sorry

end NUMINAMATH_CALUDE_a_2012_value_l3977_397770


namespace NUMINAMATH_CALUDE_real_number_line_bijection_l3977_397706

-- Define the set of points on a number line
def NumberLine : Type := ℝ

-- State the theorem
theorem real_number_line_bijection : 
  ∃ f : ℝ → NumberLine, Function.Bijective f := by sorry

end NUMINAMATH_CALUDE_real_number_line_bijection_l3977_397706


namespace NUMINAMATH_CALUDE_classical_mechanics_not_incorrect_l3977_397737

/-- Represents a scientific theory -/
structure ScientificTheory where
  name : String
  hasLimitations : Bool
  isIncorrect : Bool

/-- Classical mechanics as a scientific theory -/
def classicalMechanics : ScientificTheory := {
  name := "Classical Mechanics"
  hasLimitations := true
  isIncorrect := false
}

/-- Truth has relativity -/
axiom truth_relativity : Prop

/-- Scientific exploration is endless -/
axiom endless_exploration : Prop

/-- Theorem stating that classical mechanics is not an incorrect scientific theory -/
theorem classical_mechanics_not_incorrect :
  classicalMechanics.hasLimitations ∧ truth_relativity ∧ endless_exploration →
  ¬classicalMechanics.isIncorrect := by
  sorry


end NUMINAMATH_CALUDE_classical_mechanics_not_incorrect_l3977_397737


namespace NUMINAMATH_CALUDE_number_of_hens_l3977_397716

theorem number_of_hens (total_heads total_feet num_hens num_cows : ℕ) 
  (total_heads_eq : total_heads = 48)
  (total_feet_eq : total_feet = 144)
  (min_hens : num_hens ≥ 10)
  (min_cows : num_cows ≥ 5)
  (total_animals_eq : num_hens + num_cows = total_heads)
  (total_feet_calc : 2 * num_hens + 4 * num_cows = total_feet) :
  num_hens = 24 := by
sorry

end NUMINAMATH_CALUDE_number_of_hens_l3977_397716


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3977_397753

/-- The distance between foci of an ellipse with given semi-major and semi-minor axes -/
theorem ellipse_foci_distance (a b : ℝ) (ha : a = 7) (hb : b = 3) :
  2 * Real.sqrt (a^2 - b^2) = 4 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3977_397753


namespace NUMINAMATH_CALUDE_max_value_on_edge_l3977_397797

/-- A 2D grid represented as a function from pairs of integers to real numbers. -/
def Grid := ℤ × ℤ → ℝ

/-- Predicate to check if a cell is on the edge of the grid. -/
def isOnEdge (m n : ℕ) (i j : ℤ) : Prop :=
  i = 0 ∨ i = m - 1 ∨ j = 0 ∨ j = n - 1

/-- The set of valid coordinates in an m × n grid. -/
def validCoords (m n : ℕ) : Set (ℤ × ℤ) :=
  {(i, j) | 0 ≤ i ∧ i < m ∧ 0 ≤ j ∧ j < n}

/-- Predicate to check if a grid satisfies the arithmetic mean property. -/
def satisfiesArithmeticMeanProperty (g : Grid) (m n : ℕ) : Prop :=
  ∀ (i j : ℤ), (i, j) ∈ validCoords m n → ¬isOnEdge m n i j →
    g (i, j) = (g (i-1, j) + g (i+1, j) + g (i, j-1) + g (i, j+1)) / 4

/-- Theorem: The maximum value in a grid satisfying the arithmetic mean property
    must be on the edge. -/
theorem max_value_on_edge (g : Grid) (m n : ℕ) 
    (h_mean : satisfiesArithmeticMeanProperty g m n)
    (h_distinct : ∀ (i j k l : ℤ), (i, j) ≠ (k, l) → 
      (i, j) ∈ validCoords m n → (k, l) ∈ validCoords m n → g (i, j) ≠ g (k, l))
    (h_finite : m > 0 ∧ n > 0) :
    ∃ (i j : ℤ), (i, j) ∈ validCoords m n ∧ isOnEdge m n i j ∧
      ∀ (k l : ℤ), (k, l) ∈ validCoords m n → g (i, j) ≥ g (k, l) :=
  sorry

end NUMINAMATH_CALUDE_max_value_on_edge_l3977_397797


namespace NUMINAMATH_CALUDE_airplane_seats_l3977_397742

theorem airplane_seats (total_seats : ℕ) (first_class : ℕ) (coach_class : ℕ) : 
  total_seats = 387 →
  coach_class = 4 * first_class + 2 →
  first_class + coach_class = total_seats →
  coach_class = 310 := by
sorry

end NUMINAMATH_CALUDE_airplane_seats_l3977_397742


namespace NUMINAMATH_CALUDE_camryn_trumpet_practice_l3977_397786

/-- Represents the number of days between Camryn's practices for each instrument -/
structure PracticeSchedule where
  trumpet : ℕ
  flute : ℕ

/-- Checks if the practice schedule satisfies the given conditions -/
def is_valid_schedule (schedule : PracticeSchedule) : Prop :=
  schedule.flute = 3 ∧
  schedule.trumpet > 1 ∧
  schedule.trumpet < 33 ∧
  Nat.lcm schedule.trumpet schedule.flute = 33

theorem camryn_trumpet_practice (schedule : PracticeSchedule) :
  is_valid_schedule schedule → schedule.trumpet = 11 := by
  sorry

#check camryn_trumpet_practice

end NUMINAMATH_CALUDE_camryn_trumpet_practice_l3977_397786


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l3977_397725

theorem angle_sum_theorem (x₁ x₂ : Real) (h₁ : 0 ≤ x₁ ∧ x₁ ≤ 2 * Real.pi) 
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ 2 * Real.pi) 
  (eq₁ : Real.sin x₁ ^ 3 - Real.cos x₁ ^ 3 = (1 / Real.cos x₁) - (1 / Real.sin x₁))
  (eq₂ : Real.sin x₂ ^ 3 - Real.cos x₂ ^ 3 = (1 / Real.cos x₂) - (1 / Real.sin x₂)) :
  x₁ + x₂ = 3 * Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l3977_397725


namespace NUMINAMATH_CALUDE_width_to_perimeter_ratio_l3977_397776

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular room -/
def perimeter (room : RoomDimensions) : ℝ :=
  2 * (room.length + room.width)

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplifyRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem width_to_perimeter_ratio (room : RoomDimensions)
    (h1 : room.length = 25)
    (h2 : room.width = 15) :
    simplifyRatio (Nat.floor room.width) (Nat.floor (perimeter room)) = (3, 16) := by
  sorry

end NUMINAMATH_CALUDE_width_to_perimeter_ratio_l3977_397776


namespace NUMINAMATH_CALUDE_range_of_a_l3977_397719

def P (a : ℝ) : Set ℝ := {x | a - 4 < x ∧ x < a + 4}

def Q : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Q → x ∈ P a) ↔ -1 ≤ a ∧ a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3977_397719


namespace NUMINAMATH_CALUDE_complex_product_real_imag_parts_l3977_397798

theorem complex_product_real_imag_parts : ∃ (a b : ℝ), 
  (Complex.mk a b = (2 * Complex.I - 1) / Complex.I) ∧ (a * b = 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_imag_parts_l3977_397798


namespace NUMINAMATH_CALUDE_profit_180_greater_than_170_l3977_397783

/-- Sales data for 20 days -/
def sales_data : List (ℕ × ℕ) := [(150, 3), (160, 4), (170, 6), (180, 5), (190, 1), (200, 1)]

/-- Total number of days -/
def total_days : ℕ := 20

/-- Purchase price in yuan per kg -/
def purchase_price : ℚ := 6

/-- Selling price in yuan per kg -/
def selling_price : ℚ := 10

/-- Return price in yuan per kg -/
def return_price : ℚ := 4

/-- Calculate expected profit for a given purchase amount -/
def expected_profit (purchase_amount : ℕ) : ℚ :=
  sorry

/-- Theorem: Expected profit from 180 kg purchase is greater than 170 kg purchase -/
theorem profit_180_greater_than_170 :
  expected_profit 180 > expected_profit 170 :=
sorry

end NUMINAMATH_CALUDE_profit_180_greater_than_170_l3977_397783


namespace NUMINAMATH_CALUDE_z_value_l3977_397787

theorem z_value (x y z : ℝ) 
  (h1 : (x + y) / 2 = 4) 
  (h2 : x + y + z = 0) : 
  z = -8 := by
sorry

end NUMINAMATH_CALUDE_z_value_l3977_397787


namespace NUMINAMATH_CALUDE_triangular_coin_array_l3977_397793

theorem triangular_coin_array (N : ℕ) : (N * (N + 1)) / 2 = 3003 → N = 77 := by
  sorry

end NUMINAMATH_CALUDE_triangular_coin_array_l3977_397793


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3977_397734

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 4 = 0) → 
  (x₂^2 + 2*x₂ - 4 = 0) → 
  (x₁ + x₂ = -2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3977_397734


namespace NUMINAMATH_CALUDE_some_number_value_l3977_397782

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 25 * 45 * 49) : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3977_397782


namespace NUMINAMATH_CALUDE_oprah_band_total_weight_l3977_397717

/-- Represents the Oprah Winfrey High School marching band -/
structure MarchingBand where
  trumpet_count : ℕ
  clarinet_count : ℕ
  trombone_count : ℕ
  tuba_count : ℕ
  drum_count : ℕ
  trumpet_weight : ℕ
  clarinet_weight : ℕ
  trombone_weight : ℕ
  tuba_weight : ℕ
  drum_weight : ℕ

/-- Calculates the total weight carried by the marching band -/
def total_weight (band : MarchingBand) : ℕ :=
  band.trumpet_count * band.trumpet_weight +
  band.clarinet_count * band.clarinet_weight +
  band.trombone_count * band.trombone_weight +
  band.tuba_count * band.tuba_weight +
  band.drum_count * band.drum_weight

/-- The Oprah Winfrey High School marching band configuration -/
def oprah_band : MarchingBand := {
  trumpet_count := 6
  clarinet_count := 9
  trombone_count := 8
  tuba_count := 3
  drum_count := 2
  trumpet_weight := 5
  clarinet_weight := 5
  trombone_weight := 10
  tuba_weight := 20
  drum_weight := 15
}

/-- Theorem stating that the total weight carried by the Oprah Winfrey High School marching band is 245 pounds -/
theorem oprah_band_total_weight :
  total_weight oprah_band = 245 := by
  sorry

end NUMINAMATH_CALUDE_oprah_band_total_weight_l3977_397717


namespace NUMINAMATH_CALUDE_arthur_muffins_l3977_397765

theorem arthur_muffins (initial_muffins additional_muffins : ℕ) 
  (h1 : initial_muffins = 35)
  (h2 : additional_muffins = 48) :
  initial_muffins + additional_muffins = 83 :=
by sorry

end NUMINAMATH_CALUDE_arthur_muffins_l3977_397765


namespace NUMINAMATH_CALUDE_sprint_tournament_races_l3977_397701

/-- Calculates the minimum number of races needed to determine a champion -/
def minimumRaces (totalSprinters : Nat) (lanesPerRace : Nat) : Nat :=
  let eliminationsNeeded := totalSprinters - 1
  let eliminationsPerRace := lanesPerRace - 1
  (eliminationsNeeded + eliminationsPerRace - 1) / eliminationsPerRace

theorem sprint_tournament_races : 
  minimumRaces 256 8 = 37 := by
  sorry

#eval minimumRaces 256 8

end NUMINAMATH_CALUDE_sprint_tournament_races_l3977_397701


namespace NUMINAMATH_CALUDE_malt_shop_shakes_l3977_397723

/-- Given a malt shop scenario where:
  * Each shake uses 4 ounces of chocolate syrup
  * Each cone uses 6 ounces of chocolate syrup
  * 1 cone was sold
  * A total of 14 ounces of chocolate syrup was used
  Prove that 2 shakes were sold. -/
theorem malt_shop_shakes : 
  ∀ (shakes : ℕ), 
    (4 * shakes + 6 * 1 = 14) → shakes = 2 := by
  sorry

end NUMINAMATH_CALUDE_malt_shop_shakes_l3977_397723


namespace NUMINAMATH_CALUDE_min_value_abc_l3977_397731

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 9) :
  a^4 * b^3 * c^2 ≥ 1/10368 ∧ ∃ (a₀ b₀ c₀ : ℝ), 
    a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 1/a₀ + 1/b₀ + 1/c₀ = 9 ∧ a₀^4 * b₀^3 * c₀^2 = 1/10368 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l3977_397731


namespace NUMINAMATH_CALUDE_work_left_after_nine_days_l3977_397710

/-- The fraction of work left after 9 days given the work rates of A, B, and C -/
theorem work_left_after_nine_days (a_rate b_rate c_rate : ℚ) : 
  a_rate = 1 / 15 →
  b_rate = 1 / 20 →
  c_rate = 1 / 25 →
  let combined_rate := a_rate + b_rate + c_rate
  let work_done_first_four_days := 4 * combined_rate
  let ac_rate := a_rate + c_rate
  let work_done_next_five_days := 5 * ac_rate
  let total_work_done := work_done_first_four_days + work_done_next_five_days
  total_work_done ≥ 1 := by sorry

#check work_left_after_nine_days

end NUMINAMATH_CALUDE_work_left_after_nine_days_l3977_397710


namespace NUMINAMATH_CALUDE_regression_consistency_l3977_397743

/-- A structure representing a linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- A structure representing sample statistics -/
structure SampleStatistics where
  x_mean : ℝ
  y_mean : ℝ
  correlation : ℝ

/-- Checks if the given linear regression model is consistent with the sample statistics -/
def is_consistent_regression (stats : SampleStatistics) (model : LinearRegression) : Prop :=
  stats.correlation > 0 ∧ 
  stats.y_mean = model.slope * stats.x_mean + model.intercept

/-- The theorem stating that the given linear regression model is consistent with the sample statistics -/
theorem regression_consistency : 
  let stats : SampleStatistics := { x_mean := 3, y_mean := 3.5, correlation := 1 }
  let model : LinearRegression := { slope := 0.4, intercept := 2.3 }
  is_consistent_regression stats model := by
  sorry


end NUMINAMATH_CALUDE_regression_consistency_l3977_397743


namespace NUMINAMATH_CALUDE_chocolate_profit_l3977_397790

theorem chocolate_profit (num_bars : ℕ) (cost_per_bar : ℝ) (total_selling_price : ℝ) (packaging_cost_per_bar : ℝ) :
  num_bars = 5 →
  cost_per_bar = 5 →
  total_selling_price = 90 →
  packaging_cost_per_bar = 2 →
  total_selling_price - (num_bars * cost_per_bar + num_bars * packaging_cost_per_bar) = 55 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_profit_l3977_397790


namespace NUMINAMATH_CALUDE_third_term_of_geometric_series_l3977_397729

/-- Given an infinite geometric series with common ratio 1/4 and sum 16, 
    the third term of the sequence is 3/4. -/
theorem third_term_of_geometric_series (a : ℝ) : 
  (∃ (S : ℝ), S = 16 ∧ S = a / (1 - (1/4))) →
  a * (1/4)^2 = 3/4 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_series_l3977_397729


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3977_397712

theorem regular_polygon_sides (D : ℕ) : D = 20 → ∃ (n : ℕ), n > 2 ∧ D = n * (n - 3) / 2 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3977_397712


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3977_397773

/-- Given a line 2ax - by + 2 = 0 passing through the center of the circle (x + 1)^2 + (y - 2)^2 = 4,
    where a > 0 and b > 0, the minimum value of 1/a + 1/b is 4 -/
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : 2 * a * (-1) - b * 2 + 2 = 0) : 
  (∀ x y, (x + 1)^2 + (y - 2)^2 = 4 → 2 * a * x - b * y + 2 = 0) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 2 * a' * (-1) - b' * 2 + 2 = 0 → 1 / a' + 1 / b' ≥ 1 / a + 1 / b) →
  1 / a + 1 / b = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3977_397773


namespace NUMINAMATH_CALUDE_mary_income_proof_l3977_397799

/-- Calculates Mary's total income for a week --/
def maryIncome (maxHours regularhourlyRate overtimeRate1 overtimeRate2 bonus dues : ℝ) : ℝ :=
  let regularPay := regularhourlyRate * 20
  let overtimePay1 := overtimeRate1 * 20
  let overtimePay2 := overtimeRate2 * 20
  regularPay + overtimePay1 + overtimePay2 + bonus - dues

/-- Proves that Mary's total income is $650 given the specified conditions --/
theorem mary_income_proof :
  let maxHours : ℝ := 60
  let regularRate : ℝ := 8
  let overtimeRate1 : ℝ := regularRate * 1.25
  let overtimeRate2 : ℝ := regularRate * 1.5
  let bonus : ℝ := 100
  let dues : ℝ := 50
  maryIncome maxHours regularRate overtimeRate1 overtimeRate2 bonus dues = 650 := by
  sorry

#eval maryIncome 60 8 10 12 100 50

end NUMINAMATH_CALUDE_mary_income_proof_l3977_397799


namespace NUMINAMATH_CALUDE_car_speed_problem_l3977_397794

/-- The speed of Car A in km/h -/
def speed_A : ℝ := 80

/-- The time taken by Car A in hours -/
def time_A : ℝ := 5

/-- The speed of Car B in km/h -/
def speed_B : ℝ := 100

/-- The time taken by Car B in hours -/
def time_B : ℝ := 2

/-- The ratio of distances covered by Car A and Car B -/
def distance_ratio : ℝ := 2

theorem car_speed_problem :
  speed_A * time_A = distance_ratio * speed_B * time_B :=
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3977_397794


namespace NUMINAMATH_CALUDE_f_properties_l3977_397788

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

theorem f_properties :
  (∀ x > 0, f x ≥ 1) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ = f x₂ → x₁ + x₂ > 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3977_397788


namespace NUMINAMATH_CALUDE_max_path_length_l3977_397703

/-- A rectangular prism with dimensions 1, 2, and 3 -/
structure RectangularPrism where
  length : ℝ := 1
  width : ℝ := 2
  height : ℝ := 3

/-- A path in the rectangular prism -/
structure PrismPath (p : RectangularPrism) where
  -- The path starts and ends at the same corner
  start_end_same : Bool
  -- The path visits each corner exactly once
  visits_all_corners_once : Bool
  -- The path consists of straight lines between corners
  straight_lines : Bool
  -- The length of the path
  length : ℝ

/-- The theorem stating the maximum path length in the rectangular prism -/
theorem max_path_length (p : RectangularPrism) :
  ∃ (path : PrismPath p), 
    path.start_end_same ∧ 
    path.visits_all_corners_once ∧ 
    path.straight_lines ∧
    path.length = 2 * Real.sqrt 14 + 4 * Real.sqrt 13 ∧
    ∀ (other_path : PrismPath p), 
      other_path.start_end_same ∧ 
      other_path.visits_all_corners_once ∧ 
      other_path.straight_lines → 
      other_path.length ≤ path.length :=
sorry

end NUMINAMATH_CALUDE_max_path_length_l3977_397703


namespace NUMINAMATH_CALUDE_probability_second_green_given_first_green_l3977_397730

def total_balls : ℕ := 14
def green_balls : ℕ := 8
def red_balls : ℕ := 6

theorem probability_second_green_given_first_green :
  (green_balls : ℚ) / total_balls = 
  (green_balls : ℚ) / (green_balls + red_balls) :=
by sorry

end NUMINAMATH_CALUDE_probability_second_green_given_first_green_l3977_397730


namespace NUMINAMATH_CALUDE_card_sequence_return_l3977_397724

theorem card_sequence_return (n : ℕ) (hn : n > 0) : 
  Nat.totient (2 * n - 1) ≤ 2 * n - 2 := by
  sorry

end NUMINAMATH_CALUDE_card_sequence_return_l3977_397724


namespace NUMINAMATH_CALUDE_wall_width_is_eight_l3977_397764

/-- Proves that the width of a wall with given proportions and volume is 8 meters -/
theorem wall_width_is_eight (w h l : ℝ) (h_height : h = 6 * w) (h_length : l = 7 * h) (h_volume : w * h * l = 129024) :
  w = 8 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_is_eight_l3977_397764


namespace NUMINAMATH_CALUDE_carrie_harvest_money_l3977_397722

/-- Calculates the total money earned from selling tomatoes and carrots -/
def totalMoney (numTomatoes : ℕ) (numCarrots : ℕ) (priceTomato : ℚ) (priceCarrot : ℚ) : ℚ :=
  numTomatoes * priceTomato + numCarrots * priceCarrot

/-- Proves that the total money earned is correct for Carrie's harvest -/
theorem carrie_harvest_money :
  totalMoney 200 350 1 (3/2) = 725 := by
  sorry

end NUMINAMATH_CALUDE_carrie_harvest_money_l3977_397722


namespace NUMINAMATH_CALUDE_reeyas_average_is_73_l3977_397785

def reeyas_scores : List ℝ := [55, 67, 76, 82, 85]

theorem reeyas_average_is_73 : 
  (reeyas_scores.sum / reeyas_scores.length : ℝ) = 73 := by
  sorry

end NUMINAMATH_CALUDE_reeyas_average_is_73_l3977_397785


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l3977_397709

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := n.choose k

/-- The number of triangles with one side being a side of the decagon -/
def one_side_triangles : ℕ := n * (n - 4)

/-- The number of triangles with two sides being sides of the decagon -/
def two_side_triangles : ℕ := n

/-- The total number of favorable outcomes (triangles with at least one side being a side of the decagon) -/
def favorable_outcomes : ℕ := one_side_triangles + two_side_triangles

/-- The probability of a triangle having at least one side that is also a side of the decagon -/
def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l3977_397709


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3977_397702

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^3 + x < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3977_397702


namespace NUMINAMATH_CALUDE_total_items_in_jar_l3977_397774

-- Define the number of candies
def total_candies : ℕ := 3409
def chocolate_candies : ℕ := 1462
def gummy_candies : ℕ := 1947

-- Define the number of secret eggs
def total_eggs : ℕ := 145
def eggs_with_one_prize : ℕ := 98
def eggs_with_two_prizes : ℕ := 38
def eggs_with_three_prizes : ℕ := 9

-- Theorem to prove
theorem total_items_in_jar : 
  total_candies + 
  (eggs_with_one_prize * 1 + eggs_with_two_prizes * 2 + eggs_with_three_prizes * 3) = 3610 := by
  sorry

end NUMINAMATH_CALUDE_total_items_in_jar_l3977_397774


namespace NUMINAMATH_CALUDE_line_mb_product_l3977_397769

/-- Given a line y = mx + b passing through points (0, -3) and (2, 3), prove that mb = -9 -/
theorem line_mb_product (m b : ℝ) : 
  (0 : ℝ) = m * 0 + b → -- The line passes through (0, -3)
  (-3 : ℝ) = m * 0 + b → -- The line passes through (0, -3)
  (3 : ℝ) = m * 2 + b → -- The line passes through (2, 3)
  m * b = -9 := by
sorry

end NUMINAMATH_CALUDE_line_mb_product_l3977_397769


namespace NUMINAMATH_CALUDE_cos_sin_equation_solution_l3977_397760

theorem cos_sin_equation_solution (n : ℕ) :
  ∀ x : ℝ, (Real.cos x)^n - (Real.sin x)^n = 1 ↔
    (n % 2 = 0 ∧ ∃ k : ℤ, x = k * Real.pi) ∨
    (n % 2 = 1 ∧ (∃ k : ℤ, x = 2 * k * Real.pi ∨ x = (3 / 2 + 2 * k) * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_equation_solution_l3977_397760


namespace NUMINAMATH_CALUDE_triangle_properties_l3977_397796

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  Real.sqrt 3 * Real.cos t.B = t.b * Real.sin t.C

def condition2 (t : Triangle) : Prop :=
  2 * t.a - t.c = 2 * t.b * Real.cos t.C

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h_b : t.b = 2 * Real.sqrt 3)
  (h_cond1 : condition1 t)
  (h_cond2 : condition2 t) :
  (∃ (area : ℝ), t.a = 2 → area = 2 * Real.sqrt 3) ∧
  (2 * Real.sqrt 3 < t.a + t.c ∧ t.a + t.c ≤ 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3977_397796


namespace NUMINAMATH_CALUDE_inverse_of_A_cubed_l3977_397747

-- Define the matrix A⁻¹
def A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 0, 3]

-- State the theorem
theorem inverse_of_A_cubed :
  let A : Matrix (Fin 2) (Fin 2) ℝ := A_inv⁻¹
  (A^3)⁻¹ = !![8, -19; 0, 27] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_cubed_l3977_397747


namespace NUMINAMATH_CALUDE_square_product_of_b_values_l3977_397772

theorem square_product_of_b_values : ∃ (b₁ b₂ : ℝ),
  (∀ (x y : ℝ), (y = 3 ∨ y = 8 ∨ x = 2 ∨ x = b₁ ∨ x = b₂) →
    ((x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 8) ∨ (x = b₁ ∧ y = 3) ∨ (x = b₁ ∧ y = 8) ∨
     (x = b₂ ∧ y = 3) ∨ (x = b₂ ∧ y = 8) ∨ (x = 2 ∧ 3 ≤ y ∧ y ≤ 8) ∨
     (x = b₁ ∧ 3 ≤ y ∧ y ≤ 8) ∨ (x = b₂ ∧ 3 ≤ y ∧ y ≤ 8) ∨
     (3 ≤ x ∧ x ≤ 8 ∧ y = 3) ∨ (3 ≤ x ∧ x ≤ 8 ∧ y = 8))) ∧
  b₁ * b₂ = -21 :=
by sorry

end NUMINAMATH_CALUDE_square_product_of_b_values_l3977_397772


namespace NUMINAMATH_CALUDE_largest_integer_proof_l3977_397751

theorem largest_integer_proof (x : ℝ) (h : 20 * Real.sin x = 22 * Real.cos x) :
  ⌊(1 / (Real.sin x * Real.cos x) - 1)^7⌋ = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_proof_l3977_397751


namespace NUMINAMATH_CALUDE_A_inter_B_eq_a_geq_2_l3977_397759

-- Define sets A, B, and C
def A : Set ℝ := {x | (x + 4) * (x - 2) > 0}
def B : Set ℝ := {y | ∃ x, y = x^2 - 2*x + 2}
def C (a : ℝ) : Set ℝ := {x | -4 ≤ x ∧ x ≤ a}

-- Define the complement of A relative to C
def C_R_A (a : ℝ) : Set ℝ := {x | x ∈ C a ∧ x ∉ A}

-- Theorem for part (I)
theorem A_inter_B_eq : A ∩ B = {x | x > 2} := by sorry

-- Theorem for part (II)
theorem a_geq_2 (a : ℝ) : C_R_A a ⊆ C a → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_A_inter_B_eq_a_geq_2_l3977_397759


namespace NUMINAMATH_CALUDE_ellipse1_properties_ellipse2_properties_l3977_397720

-- Part 1
def ellipse1 (x y : ℝ) : Prop := x^2/15 + y^2/10 = 1

def given_ellipse (x y : ℝ) : Prop := 4*x^2 + 9*y^2 = 36

theorem ellipse1_properties :
  (∀ x y, ellipse1 x y → (x = 3 ∧ y = -2)) ∧
  (∀ x y, ellipse1 x y → ∃ c, c^2 = 5 ∧ 
    ∃ a b, a^2 = 15 ∧ b^2 = 10 ∧ c^2 = a^2 - b^2) :=
sorry

-- Part 2
def ellipse2 (x y : ℝ) : Prop := x^2/16 + y^2/8 = 1

def origin : ℝ × ℝ := (0, 0)
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0
def eccentricity (e : ℝ) : Prop := e = Real.sqrt 2 / 2
def triangle_perimeter (p : ℝ) : Prop := p = 16

theorem ellipse2_properties :
  (∀ x y, ellipse2 x y → ∃ c, c > 0 ∧ 
    (∃ f1 f2 : ℝ × ℝ, f1 = (-c, 0) ∧ f2 = (c, 0) ∧
      on_x_axis f1 ∧ on_x_axis f2)) ∧
  eccentricity (Real.sqrt 2 / 2) ∧
  (∃ p, triangle_perimeter p) :=
sorry

end NUMINAMATH_CALUDE_ellipse1_properties_ellipse2_properties_l3977_397720


namespace NUMINAMATH_CALUDE_function_property_l3977_397745

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) = f x * f y)
  (h2 : ∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y))
  (h3 : f 2 = 4) : 
  f 3 ≤ 9 := by sorry

end NUMINAMATH_CALUDE_function_property_l3977_397745


namespace NUMINAMATH_CALUDE_fourth_term_is_8000_l3977_397781

/-- Geometric sequence with first term 1 and common ratio 20 -/
def geometric_sequence (n : ℕ) : ℕ :=
  1 * 20^(n - 1)

/-- The fourth term of the geometric sequence is 8000 -/
theorem fourth_term_is_8000 : geometric_sequence 4 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_8000_l3977_397781


namespace NUMINAMATH_CALUDE_distance_between_points_l3977_397746

/-- The distance between points (1, 3) and (-5, 7) is 2√13 units. -/
theorem distance_between_points : 
  let pointA : ℝ × ℝ := (1, 3)
  let pointB : ℝ × ℝ := (-5, 7)
  Real.sqrt ((pointB.1 - pointA.1)^2 + (pointB.2 - pointA.2)^2) = 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l3977_397746


namespace NUMINAMATH_CALUDE_exponent_multiplication_and_zero_power_l3977_397763

theorem exponent_multiplication_and_zero_power :
  (∀ x : ℝ, x^2 * x^4 = x^6) ∧ ((-5^2)^0 = 1) := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_and_zero_power_l3977_397763


namespace NUMINAMATH_CALUDE_cube_cuboid_volume_ratio_l3977_397784

theorem cube_cuboid_volume_ratio :
  let cube_side : ℝ := 1
  let cuboid_width : ℝ := 50 / 100
  let cuboid_length : ℝ := 50 / 100
  let cuboid_height : ℝ := 20 / 100
  let cube_volume := cube_side ^ 3
  let cuboid_volume := cuboid_width * cuboid_length * cuboid_height
  cube_volume / cuboid_volume = 20 := by
    sorry

end NUMINAMATH_CALUDE_cube_cuboid_volume_ratio_l3977_397784


namespace NUMINAMATH_CALUDE_star_six_three_l3977_397744

-- Define the ⭐ operation
def star (x y : ℝ) : ℝ := 4 * x - 2 * y

-- State the theorem
theorem star_six_three : star 6 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_star_six_three_l3977_397744


namespace NUMINAMATH_CALUDE_angle_equivalence_l3977_397767

def angle_with_same_terminal_side (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 - 120

theorem angle_equivalence :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 360 ∧ angle_with_same_terminal_side θ ∧ θ = 240 := by
sorry

end NUMINAMATH_CALUDE_angle_equivalence_l3977_397767


namespace NUMINAMATH_CALUDE_remainder_equality_l3977_397713

theorem remainder_equality (A B D S S' s s' : ℕ) : 
  A > B →
  (A + 3) % D = S →
  (B - 2) % D = S' →
  ((A + 3) * (B - 2)) % D = s →
  (S * S') % D = s' →
  s = s' := by sorry

end NUMINAMATH_CALUDE_remainder_equality_l3977_397713


namespace NUMINAMATH_CALUDE_inequality_proof_l3977_397789

theorem inequality_proof (x y z : ℝ) (h : x^2 + y^2 + z^2 = 3) :
  x^3 - (y^2 + y*z + z^2)*x + y*z*(y + z) ≤ 3 * Real.sqrt 3 ∧
  (x^3 - (y^2 + y*z + z^2)*x + y*z*(y + z) = 3 * Real.sqrt 3 ↔ 
   x = Real.sqrt 3 ∧ y = 0 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3977_397789


namespace NUMINAMATH_CALUDE_angle_bisector_length_right_triangle_l3977_397705

theorem angle_bisector_length_right_triangle (a b c : ℝ) (h1 : a = 15) (h2 : b = 20) (h3 : c = 25) 
  (h4 : a^2 + b^2 = c^2) : ∃ (AA₁ : ℝ), AA₁ = (20 * Real.sqrt 10) / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_length_right_triangle_l3977_397705


namespace NUMINAMATH_CALUDE_box_height_is_twelve_l3977_397736

-- Define the box dimensions and costs
def box_base_length : ℝ := 20
def box_base_width : ℝ := 20
def cost_per_box : ℝ := 0.50
def total_volume_needed : ℝ := 2160000
def min_spending : ℝ := 225

-- Theorem to prove
theorem box_height_is_twelve :
  ∃ (h : ℝ), h > 0 ∧ 
    (total_volume_needed / (box_base_length * box_base_width * h)) * cost_per_box ≥ min_spending ∧
    ∀ (h' : ℝ), h' > h → 
      (total_volume_needed / (box_base_length * box_base_width * h')) * cost_per_box < min_spending ∧
    h = 12 :=
by sorry

end NUMINAMATH_CALUDE_box_height_is_twelve_l3977_397736


namespace NUMINAMATH_CALUDE_tetrahedron_sum_l3977_397732

/-- A regular tetrahedron is a three-dimensional shape with four congruent equilateral triangular faces. -/
structure RegularTetrahedron where
  -- We don't need to define any fields here, as we're only interested in its properties

/-- The number of edges in a regular tetrahedron -/
def num_edges (t : RegularTetrahedron) : ℕ := 6

/-- The number of vertices in a regular tetrahedron -/
def num_vertices (t : RegularTetrahedron) : ℕ := 4

/-- The number of faces in a regular tetrahedron -/
def num_faces (t : RegularTetrahedron) : ℕ := 4

/-- The theorem stating that the sum of edges, vertices, and faces of a regular tetrahedron is 14 -/
theorem tetrahedron_sum (t : RegularTetrahedron) : 
  num_edges t + num_vertices t + num_faces t = 14 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sum_l3977_397732


namespace NUMINAMATH_CALUDE_stratified_sampling_correctness_problem_case_proof_l3977_397707

/-- Represents the number of students in each year and the total sample size. -/
structure SchoolData where
  totalStudents : ℕ
  freshmanStudents : ℕ
  sophomoreStudents : ℕ
  juniorStudents : ℕ
  sampleSize : ℕ

/-- Calculates the number of students to be sampled from a specific year. -/
def sampledStudents (data : SchoolData) (yearStudents : ℕ) : ℕ :=
  (yearStudents * data.sampleSize) / data.totalStudents

/-- Theorem stating that the sum of sampled students from each year equals the total sample size. -/
theorem stratified_sampling_correctness (data : SchoolData) 
    (h1 : data.totalStudents = data.freshmanStudents + data.sophomoreStudents + data.juniorStudents)
    (h2 : data.sampleSize ≤ data.totalStudents) :
  sampledStudents data data.freshmanStudents +
  sampledStudents data data.sophomoreStudents +
  sampledStudents data data.juniorStudents = data.sampleSize := by
  sorry

/-- Verifies the specific case given in the problem. -/
def verifyProblemCase : Prop :=
  let data : SchoolData := {
    totalStudents := 1200,
    freshmanStudents := 300,
    sophomoreStudents := 400,
    juniorStudents := 500,
    sampleSize := 60
  }
  sampledStudents data data.freshmanStudents = 15 ∧
  sampledStudents data data.sophomoreStudents = 20 ∧
  sampledStudents data data.juniorStudents = 25

/-- Proves the specific case given in the problem. -/
theorem problem_case_proof : verifyProblemCase := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_correctness_problem_case_proof_l3977_397707


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3977_397780

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define set A
def A : Set Nat := {2, 3, 5, 6}

-- Define set B
def B : Set Nat := {1, 3, 4, 6, 7}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3977_397780


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l3977_397752

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (25*x)/(73*y)) :
  Real.sqrt x / Real.sqrt y = 5/2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l3977_397752


namespace NUMINAMATH_CALUDE_cube_root_square_l3977_397778

theorem cube_root_square (x : ℝ) : (x + 5) ^ (1/3 : ℝ) = 3 → (x + 5)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_square_l3977_397778


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l3977_397768

/-- A regular tetrahedron with specific properties -/
structure RegularTetrahedron where
  -- The distance from the midpoint of the height to a lateral face
  midpoint_to_face : ℝ
  -- The distance from the midpoint of the height to a lateral edge
  midpoint_to_edge : ℝ

/-- The volume of a regular tetrahedron -/
def volume (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the volume of a specific regular tetrahedron -/
theorem volume_of_specific_tetrahedron :
  ∃ (t : RegularTetrahedron),
    t.midpoint_to_face = 2 ∧
    t.midpoint_to_edge = Real.sqrt 10 ∧
    volume t = 80 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l3977_397768


namespace NUMINAMATH_CALUDE_students_in_front_of_yuna_l3977_397755

theorem students_in_front_of_yuna (total_students : ℕ) (students_behind_yuna : ℕ) : 
  total_students = 25 → students_behind_yuna = 9 → total_students - (students_behind_yuna + 1) = 15 := by
  sorry


end NUMINAMATH_CALUDE_students_in_front_of_yuna_l3977_397755


namespace NUMINAMATH_CALUDE_village_apple_trees_l3977_397758

theorem village_apple_trees (total_sample_percentage : ℚ) 
  (apple_trees_in_sample : ℕ) (total_apple_trees : ℕ) : 
  total_sample_percentage = 1/10 →
  apple_trees_in_sample = 80 →
  total_apple_trees = apple_trees_in_sample / total_sample_percentage →
  total_apple_trees = 800 := by
  sorry

end NUMINAMATH_CALUDE_village_apple_trees_l3977_397758


namespace NUMINAMATH_CALUDE_f_has_zero_at_two_two_is_zero_point_of_f_l3977_397738

/-- A function that has a zero point at 2 -/
def f (x : ℝ) : ℝ := x - 2

/-- Theorem stating that f has a zero point at 2 -/
theorem f_has_zero_at_two : f 2 = 0 := by
  sorry

/-- Definition of a zero point -/
def is_zero_point (g : ℝ → ℝ) (x : ℝ) : Prop := g x = 0

/-- Theorem stating that 2 is a zero point of f -/
theorem two_is_zero_point_of_f : is_zero_point f 2 := by
  sorry

end NUMINAMATH_CALUDE_f_has_zero_at_two_two_is_zero_point_of_f_l3977_397738


namespace NUMINAMATH_CALUDE_probability_divisible_by_15_l3977_397777

/-- The set of digits used to form the six-digit number -/
def digits : Finset Nat := {1, 2, 3, 4, 5, 9}

/-- The number of digits -/
def n : Nat := 6

/-- A permutation of the digits -/
def Permutation := Fin n → Fin n

/-- The set of all permutations -/
def allPermutations : Finset Permutation := sorry

/-- Predicate to check if a permutation results in a number divisible by 15 -/
def isDivisibleBy15 (p : Permutation) : Prop := sorry

/-- The number of permutations that result in a number divisible by 15 -/
def divisibleBy15Count : Nat := sorry

/-- The total number of permutations -/
def totalPermutations : Nat := Finset.card allPermutations

theorem probability_divisible_by_15 :
  (divisibleBy15Count : ℚ) / totalPermutations = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_15_l3977_397777

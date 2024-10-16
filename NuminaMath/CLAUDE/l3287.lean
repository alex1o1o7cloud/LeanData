import Mathlib

namespace NUMINAMATH_CALUDE_right_handed_players_count_l3287_328769

theorem right_handed_players_count (total_players throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 46)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) -- Ensure non-throwers are divisible by 3
  : (throwers + (2 * (total_players - throwers) / 3)) = 62 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l3287_328769


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3287_328786

/-- The equation of a line passing through (3, 2) and parallel to 4x + y - 2 = 0 -/
theorem parallel_line_equation : 
  ∀ (x y : ℝ), 
  (∃ (m b : ℝ), 
    -- The line passes through (3, 2)
    2 = m * 3 + b ∧ 
    -- The line is parallel to 4x + y - 2 = 0
    m = -4 ∧ 
    -- The equation of the line
    y = m * x + b) 
  ↔ 
  -- The resulting equation
  4 * x + y - 14 = 0 := by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l3287_328786


namespace NUMINAMATH_CALUDE_integer_pair_divisibility_l3287_328782

theorem integer_pair_divisibility (a b : ℤ) (ha : a > 1) (hb : b > 1)
  (hab : a ∣ (b + 1)) (hba : b ∣ (a^3 - 1)) :
  (∃ s : ℤ, s ≥ 2 ∧ a = s ∧ b = s^3 - 1) ∨
  (∃ s : ℤ, s ≥ 3 ∧ a = s ∧ b = s - 1) :=
by sorry

end NUMINAMATH_CALUDE_integer_pair_divisibility_l3287_328782


namespace NUMINAMATH_CALUDE_elliptical_path_derivative_l3287_328749

/-- The derivative of a vector function representing an elliptical path. -/
theorem elliptical_path_derivative (a b t : ℝ) :
  let r : ℝ → ℝ × ℝ := fun t => (a * Real.cos t, b * Real.sin t)
  let dr : ℝ → ℝ × ℝ := fun t => (-a * Real.sin t, b * Real.cos t)
  (deriv r) t = dr t := by
  sorry

end NUMINAMATH_CALUDE_elliptical_path_derivative_l3287_328749


namespace NUMINAMATH_CALUDE_mouse_jump_distance_l3287_328743

/-- Represents the jumping distances of animals in a contest -/
structure JumpContest where
  grasshopper : ℕ
  frog : ℕ
  mouse : ℕ

/-- Defines the conditions of the jumping contest -/
def validContest (contest : JumpContest) : Prop :=
  contest.grasshopper = 39 ∧
  contest.grasshopper = contest.frog + 19 ∧
  contest.mouse + 12 = contest.frog

/-- Theorem stating that in a valid contest, the mouse jumps 8 inches -/
theorem mouse_jump_distance (contest : JumpContest) 
  (h : validContest contest) : contest.mouse = 8 := by
  sorry

#check mouse_jump_distance

end NUMINAMATH_CALUDE_mouse_jump_distance_l3287_328743


namespace NUMINAMATH_CALUDE_additional_chicken_wings_l3287_328706

theorem additional_chicken_wings (num_friends : ℕ) (initial_wings : ℕ) (wings_per_person : ℕ) : 
  num_friends = 4 → initial_wings = 9 → wings_per_person = 4 →
  num_friends * wings_per_person - initial_wings = 7 := by
  sorry

end NUMINAMATH_CALUDE_additional_chicken_wings_l3287_328706


namespace NUMINAMATH_CALUDE_prime_equation_solutions_l3287_328740

/-- A prime number (not necessarily positive) -/
def IsPrime (n : ℤ) : Prop := n ≠ 0 ∧ n ≠ 1 ∧ n ≠ -1 ∧ ∀ m : ℤ, m ∣ n → (m = 1 ∨ m = -1 ∨ m = n ∨ m = -n)

/-- The set of solutions -/
def SolutionSet : Set (ℤ × ℤ × ℤ) :=
  {(5, 2, 2), (-5, -2, -2), (-5, 3, -2), (-5, -2, 3), (5, 2, -3), (5, -3, 2)}

theorem prime_equation_solutions :
  ∀ p q r : ℤ,
    IsPrime p ∧ IsPrime q ∧ IsPrime r →
    (1 / (p - q - r : ℚ) = 1 / (q : ℚ) + 1 / (r : ℚ)) ↔ (p, q, r) ∈ SolutionSet :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solutions_l3287_328740


namespace NUMINAMATH_CALUDE_min_correct_answers_quiz_problem_l3287_328761

/-- The minimum number of correctly answered questions to exceed 81 points in a quiz -/
theorem min_correct_answers (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (target_score : ℕ) : ℕ :=
  let min_correct := ((target_score + 1 + incorrect_points * total_questions) + (correct_points + incorrect_points) - 1) / (correct_points + incorrect_points)
  min_correct

/-- The specific quiz problem -/
theorem quiz_problem : min_correct_answers 22 4 2 81 = 21 := by
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_quiz_problem_l3287_328761


namespace NUMINAMATH_CALUDE_income_comparison_l3287_328738

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = 0.9 * juan) 
  (h2 : mary = 1.44 * juan) : 
  (mary - tim) / tim = 0.6 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l3287_328738


namespace NUMINAMATH_CALUDE_president_and_committee_selection_l3287_328747

/-- The number of ways to choose a president and a 3-person committee from a group of 10 people,
    where the order of committee selection doesn't matter and the president cannot be on the committee. -/
def select_president_and_committee (total_people : ℕ) (committee_size : ℕ) : ℕ :=
  total_people * (Nat.choose (total_people - 1) committee_size)

/-- Theorem stating that the number of ways to choose a president and a 3-person committee
    from a group of 10 people, where the order of committee selection doesn't matter and
    the president cannot be on the committee, is equal to 840. -/
theorem president_and_committee_selection :
  select_president_and_committee 10 3 = 840 := by
  sorry


end NUMINAMATH_CALUDE_president_and_committee_selection_l3287_328747


namespace NUMINAMATH_CALUDE_road_graveling_cost_l3287_328713

theorem road_graveling_cost (lawn_length lawn_width road_width gravel_cost : ℝ) :
  lawn_length = 80 ∧
  lawn_width = 60 ∧
  road_width = 10 ∧
  gravel_cost = 5 →
  (lawn_length * road_width + (lawn_width - road_width) * road_width) * gravel_cost = 6500 :=
by sorry

end NUMINAMATH_CALUDE_road_graveling_cost_l3287_328713


namespace NUMINAMATH_CALUDE_multiplication_result_l3287_328711

theorem multiplication_result : 163861 * 454733 = 74505853393 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_result_l3287_328711


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l3287_328704

-- Define the ellipse
structure Ellipse where
  center : ℝ × ℝ
  vertex : ℝ × ℝ
  focus : ℝ × ℝ
  b_point : ℝ × ℝ

-- Define the line
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define the problem conditions
def ellipse_conditions (e : Ellipse) : Prop :=
  e.center = (0, 0) ∧
  e.vertex = (0, 2) ∧
  e.b_point = (Real.sqrt 2, Real.sqrt 2) ∧
  Real.sqrt ((e.focus.1 - Real.sqrt 2)^2 + (e.focus.2 - Real.sqrt 2)^2) = 2

-- Define the theorem
theorem ellipse_and_line_properties (e : Ellipse) (l : Line) :
  ellipse_conditions e →
  (∀ x y, y = l.slope * x + l.y_intercept → x^2 / 12 + y^2 / 4 = 1) →
  (0, -3) ∈ {(x, y) | y = l.slope * x + l.y_intercept} →
  (∃ m n : ℝ × ℝ, m ≠ n ∧
    m ∈ {(x, y) | x^2 / 12 + y^2 / 4 = 1} ∧
    n ∈ {(x, y) | x^2 / 12 + y^2 / 4 = 1} ∧
    m ∈ {(x, y) | y = l.slope * x + l.y_intercept} ∧
    n ∈ {(x, y) | y = l.slope * x + l.y_intercept} ∧
    (m.1 - 0)^2 + (m.2 - 2)^2 = (n.1 - 0)^2 + (n.2 - 2)^2) →
  (x^2 / 12 + y^2 / 4 = 1 ∧ (l.slope = Real.sqrt 6 / 3 ∨ l.slope = -Real.sqrt 6 / 3) ∧ l.y_intercept = -3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l3287_328704


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_l3287_328723

/-- The cost of a Ferris wheel ride given the conditions of Zach's amusement park visit -/
theorem ferris_wheel_cost 
  (total_rides : Nat) 
  (roller_coaster_cost log_ride_cost : Nat) 
  (zach_initial_tickets zach_additional_tickets : Nat) :
  total_rides = 3 →
  roller_coaster_cost = 7 →
  log_ride_cost = 1 →
  zach_initial_tickets = 1 →
  zach_additional_tickets = 9 →
  roller_coaster_cost + log_ride_cost + 2 = zach_initial_tickets + zach_additional_tickets :=
by sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_l3287_328723


namespace NUMINAMATH_CALUDE_common_divisor_problem_l3287_328746

theorem common_divisor_problem (n : ℕ) (hn : n < 50) :
  (∃ d : ℕ, d > 1 ∧ d ∣ (3 * n + 5) ∧ d ∣ (5 * n + 4)) ↔ n ∈ ({7, 20, 33, 46} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_common_divisor_problem_l3287_328746


namespace NUMINAMATH_CALUDE_green_ball_probability_l3287_328709

/-- Represents a container with balls of different colors -/
structure Container where
  red : ℕ
  green : ℕ
  blue : ℕ

/-- Calculates the total number of balls in a container -/
def Container.total (c : Container) : ℕ := c.red + c.green + c.blue

/-- Represents the problem setup with three containers -/
structure BallProblem where
  container1 : Container
  container2 : Container
  container3 : Container

/-- The specific problem instance as described -/
def problem : BallProblem :=
  { container1 := { red := 10, green := 2, blue := 3 }
  , container2 := { red := 5, green := 4, blue := 2 }
  , container3 := { red := 3, green := 5, blue := 3 }
  }

/-- Calculates the probability of selecting a green ball given the problem setup -/
def probabilityGreenBall (p : BallProblem) : ℚ :=
  let p1 := (p.container1.green : ℚ) / p.container1.total
  let p2 := (p.container2.green : ℚ) / p.container2.total
  let p3 := (p.container3.green : ℚ) / p.container3.total
  (p1 + p2 + p3) / 3

theorem green_ball_probability :
  probabilityGreenBall problem = 157 / 495 := by sorry

end NUMINAMATH_CALUDE_green_ball_probability_l3287_328709


namespace NUMINAMATH_CALUDE_litter_patrol_collection_l3287_328792

theorem litter_patrol_collection (glass_bottles : ℕ) (aluminum_cans : ℕ) : 
  glass_bottles = 10 → aluminum_cans = 8 → glass_bottles + aluminum_cans = 18 := by
  sorry

end NUMINAMATH_CALUDE_litter_patrol_collection_l3287_328792


namespace NUMINAMATH_CALUDE_smallest_fraction_l3287_328785

theorem smallest_fraction (S : Set ℚ) (h : S = {1/2, 2/3, 1/4, 5/6, 7/12}) :
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = 1/4 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l3287_328785


namespace NUMINAMATH_CALUDE_unattainable_y_value_l3287_328722

theorem unattainable_y_value (x : ℝ) (h : x ≠ -4/3) :
  ¬∃ x, (2 - x) / (3 * x + 4) = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_unattainable_y_value_l3287_328722


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3287_328784

theorem simplify_sqrt_expression : Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3287_328784


namespace NUMINAMATH_CALUDE_z_values_l3287_328757

theorem z_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 72) :
  let z := ((x - 3)^2 * (x + 4)) / (2*x - 4)
  z = 64.8 ∨ z = -10.125 := by
sorry

end NUMINAMATH_CALUDE_z_values_l3287_328757


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3287_328750

theorem polynomial_evaluation (a : ℝ) (h : a = 2) : (7*a^2 - 20*a + 5) * (3*a - 4) = -14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3287_328750


namespace NUMINAMATH_CALUDE_pairing_probability_l3287_328751

theorem pairing_probability (n : ℕ) (h : n = 28) :
  (1 : ℚ) / (n - 1) = 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_pairing_probability_l3287_328751


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_three_l3287_328717

theorem gcd_of_powers_of_three : Nat.gcd (3^1200 - 1) (3^1210 - 1) = 3^10 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_three_l3287_328717


namespace NUMINAMATH_CALUDE_jia_age_is_24_l3287_328739

/-- Represents the ages of four individuals -/
structure Ages where
  jia : ℕ
  yi : ℕ
  bing : ℕ
  ding : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- Jia is 4 years older than Yi
  ages.jia = ages.yi + 4 ∧
  -- Ding is 17 years old
  ages.ding = 17 ∧
  -- The average age of Jia, Yi, and Bing is 1 year more than the average age of all four people
  (ages.jia + ages.yi + ages.bing) / 3 = (ages.jia + ages.yi + ages.bing + ages.ding) / 4 + 1 ∧
  -- The average age of Jia and Yi is 1 year more than the average age of Jia, Yi, and Bing
  (ages.jia + ages.yi) / 2 = (ages.jia + ages.yi + ages.bing) / 3 + 1

/-- The theorem stating that if the conditions are satisfied, Jia's age is 24 -/
theorem jia_age_is_24 (ages : Ages) (h : satisfies_conditions ages) : ages.jia = 24 := by
  sorry

end NUMINAMATH_CALUDE_jia_age_is_24_l3287_328739


namespace NUMINAMATH_CALUDE_altered_coin_probability_l3287_328773

theorem altered_coin_probability :
  ∃! p : ℝ, 0 < p ∧ p < 1/2 ∧ (20 : ℝ) * p^3 * (1-p)^3 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_altered_coin_probability_l3287_328773


namespace NUMINAMATH_CALUDE_jewelry_store_problem_l3287_328760

theorem jewelry_store_problem (S P : ℝ) 
  (h1 : S = P + 0.25 * S)
  (h2 : 16 = 0.8 * S - P) :
  P = 240 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_store_problem_l3287_328760


namespace NUMINAMATH_CALUDE_triangle_problem_l3287_328705

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a * Real.sin B + b * Real.cos A = c) →
  (a = Real.sqrt 2 * c) →
  (b = 2) →
  -- Conclusions
  (B = π / 4 ∧ c = 2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_problem_l3287_328705


namespace NUMINAMATH_CALUDE_participating_countries_form_set_l3287_328752

/-- A type representing countries --/
structure Country where
  name : String

/-- A type representing a specific event --/
structure Event where
  name : String
  year : Nat

/-- A predicate that determines if a country participated in an event --/
def participated (country : Country) (event : Event) : Prop := sorry

/-- Definition of a set with definite elements --/
def isDefiniteSet (S : Set α) : Prop :=
  ∀ x, (x ∈ S) ∨ (x ∉ S)

/-- Theorem stating that countries participating in a specific event form a definite set --/
theorem participating_countries_form_set (event : Event) :
  isDefiniteSet {country : Country | participated country event} := by
  sorry

end NUMINAMATH_CALUDE_participating_countries_form_set_l3287_328752


namespace NUMINAMATH_CALUDE_f_not_tangent_to_x_axis_max_a_for_monotone_g_l3287_328728

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x - (a / 2) * x^2

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - a * x

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := f a x + 2 * x

-- Define the derivative of g(x)
def g_deriv (a : ℝ) (x : ℝ) : ℝ := f_deriv a x + 2

-- Theorem 1: f(x) cannot be tangent to the x-axis for any a
theorem f_not_tangent_to_x_axis (a : ℝ) : ¬∃ x : ℝ, f a x = 0 ∧ f_deriv a x = 0 := by
  sorry

-- Theorem 2: The maximum integer value of a for which g(x) is monotonically increasing is 1
theorem max_a_for_monotone_g : 
  ∀ a : ℤ, (∀ x : ℝ, g_deriv a x ≥ 0) → a ≤ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_f_not_tangent_to_x_axis_max_a_for_monotone_g_l3287_328728


namespace NUMINAMATH_CALUDE_fourth_number_proof_l3287_328798

theorem fourth_number_proof (sum : ℝ) (a b c : ℝ) (h1 : sum = 221.2357) 
  (h2 : a = 217) (h3 : b = 2.017) (h4 : c = 0.217) : 
  sum - (a + b + c) = 2.0017 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l3287_328798


namespace NUMINAMATH_CALUDE_box_volume_percentage_l3287_328712

/-- The percentage of volume occupied by 4-inch cubes in a rectangular box -/
theorem box_volume_percentage :
  let box_length : ℕ := 8
  let box_width : ℕ := 6
  let box_height : ℕ := 12
  let cube_size : ℕ := 4
  let cubes_length : ℕ := box_length / cube_size
  let cubes_width : ℕ := box_width / cube_size
  let cubes_height : ℕ := box_height / cube_size
  let total_cubes : ℕ := cubes_length * cubes_width * cubes_height
  let cubes_volume : ℕ := total_cubes * (cube_size ^ 3)
  let box_volume : ℕ := box_length * box_width * box_height
  (cubes_volume : ℚ) / (box_volume : ℚ) = 2 / 3 :=
by
  sorry

#check box_volume_percentage

end NUMINAMATH_CALUDE_box_volume_percentage_l3287_328712


namespace NUMINAMATH_CALUDE_kddk_divisible_by_7_l3287_328793

/-- Represents a base-6 digit -/
def Base6Digit : Type := { n : ℕ // n < 6 }

/-- Converts a base-6 number of the form kddk to base 10 -/
def toBase10 (k d : Base6Digit) : ℕ :=
  217 * k.val + 42 * d.val

theorem kddk_divisible_by_7 (k d : Base6Digit) :
  7 ∣ toBase10 k d ↔ k = d :=
sorry

end NUMINAMATH_CALUDE_kddk_divisible_by_7_l3287_328793


namespace NUMINAMATH_CALUDE_prime_product_l3287_328758

theorem prime_product (p₁ p₂ p₃ p₄ : ℕ) : 
  p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧ p₄.Prime ∧ 
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
  2*p₁ + 3*p₂ + 5*p₃ + 7*p₄ = 162 ∧
  11*p₁ + 7*p₂ + 5*p₃ + 4*p₄ = 162 →
  p₁ * p₂ * p₃ * p₄ = 570 := by
sorry

end NUMINAMATH_CALUDE_prime_product_l3287_328758


namespace NUMINAMATH_CALUDE_f_of_5_eq_92_l3287_328788

/-- Given a function f(x) = 2x^2 + y where f(2) = 50, prove that f(5) = 92 -/
theorem f_of_5_eq_92 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 50) :
  f 5 = 92 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_eq_92_l3287_328788


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3287_328776

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 - 4*x - 5 = 0} = {-1, 5} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3287_328776


namespace NUMINAMATH_CALUDE_total_amount_paid_l3287_328797

def apple_quantity : ℕ := 8
def apple_rate : ℕ := 70
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 45

theorem total_amount_paid : 
  apple_quantity * apple_rate + mango_quantity * mango_rate = 965 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l3287_328797


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3287_328770

theorem polynomial_simplification (x : ℝ) :
  (3*x^2 + 5*x + 8)*(x - 2) - (x - 2)*(x^2 + 6*x - 72) + (2*x - 15)*(x - 2)*(x + 4) =
  4*x^3 - 17*x^2 + 38*x - 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3287_328770


namespace NUMINAMATH_CALUDE_base7_5304_equals_1866_l3287_328748

def base7_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem base7_5304_equals_1866 :
  base7_to_decimal [5, 3, 0, 4] = 1866 := by
  sorry

end NUMINAMATH_CALUDE_base7_5304_equals_1866_l3287_328748


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3287_328790

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perp : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (l : Line) (α β : Plane) :
  perpendicular l α → parallel l β → perp α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3287_328790


namespace NUMINAMATH_CALUDE_amount_calculation_l3287_328714

theorem amount_calculation (x : ℝ) (amount : ℝ) (h1 : x = 25.0) (h2 : 2 * x = 3 * x - amount) : amount = 25.0 := by
  sorry

end NUMINAMATH_CALUDE_amount_calculation_l3287_328714


namespace NUMINAMATH_CALUDE_smallest_k_correct_l3287_328731

/-- The smallest integer k for which kx^2 - 4x - 4 = 0 has two distinct real roots -/
def smallest_k : ℤ := 1

/-- Quadratic equation ax^2 + bx + c = 0 has two distinct real roots iff b^2 - 4ac > 0 -/
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c > 0

theorem smallest_k_correct :
  (∀ k : ℤ, k < smallest_k → ¬(has_two_distinct_real_roots k (-4) (-4))) ∧
  has_two_distinct_real_roots smallest_k (-4) (-4) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_correct_l3287_328731


namespace NUMINAMATH_CALUDE_carpet_rearrangement_l3287_328720

/-- Represents a piece of carpet with a given length -/
structure CarpetPiece where
  length : ℝ
  length_pos : length > 0

/-- Represents a corridor covered by carpet pieces -/
structure CarpetedCorridor where
  length : ℝ
  length_pos : length > 0
  pieces : List CarpetPiece
  covers_corridor : (pieces.map CarpetPiece.length).sum ≥ length

theorem carpet_rearrangement (corridor : CarpetedCorridor) :
  ∃ (subset : List CarpetPiece), subset ⊆ corridor.pieces ∧
    (subset.map CarpetPiece.length).sum ≥ corridor.length ∧
    (subset.map CarpetPiece.length).sum < 2 * corridor.length :=
by sorry

end NUMINAMATH_CALUDE_carpet_rearrangement_l3287_328720


namespace NUMINAMATH_CALUDE_at_most_one_integer_point_on_circle_l3287_328700

theorem at_most_one_integer_point_on_circle :
  ∀ (x y u v : ℤ),
  (x - Real.sqrt 2)^2 + (y - Real.sqrt 3)^2 = (u - Real.sqrt 2)^2 + (v - Real.sqrt 3)^2 →
  x = u ∧ y = v :=
by sorry

end NUMINAMATH_CALUDE_at_most_one_integer_point_on_circle_l3287_328700


namespace NUMINAMATH_CALUDE_cos_theta_plus_pi_fourth_l3287_328777

theorem cos_theta_plus_pi_fourth (θ : ℝ) (h : Real.sin (θ - π/4) = 1/5) :
  Real.cos (θ + π/4) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_theta_plus_pi_fourth_l3287_328777


namespace NUMINAMATH_CALUDE_range_of_a_l3287_328708

theorem range_of_a (a : ℝ) : 
  (∀ x, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ x, x ≥ a ∧ |x - 1| ≥ 1) → 
  a ≤ 0 := by
sorry


end NUMINAMATH_CALUDE_range_of_a_l3287_328708


namespace NUMINAMATH_CALUDE_xenia_earnings_l3287_328766

/-- Xenia's earnings over two weeks given her work hours and wage difference --/
theorem xenia_earnings (hours_week2 hours_week3 : ℕ) (wage_difference : ℚ) : 
  hours_week2 = 18 →
  hours_week3 = 24 →
  wage_difference = 38.4 →
  ∃ (hourly_wage : ℚ), 
    hourly_wage * (hours_week3 - hours_week2) = wage_difference ∧
    hourly_wage * (hours_week2 + hours_week3) = 268.8 := by
  sorry

end NUMINAMATH_CALUDE_xenia_earnings_l3287_328766


namespace NUMINAMATH_CALUDE_total_monthly_payment_l3287_328715

def basic_service : ℕ := 15
def movie_channels : ℕ := 12
def sports_channels : ℕ := movie_channels - 3

theorem total_monthly_payment :
  basic_service + movie_channels + sports_channels = 36 :=
by sorry

end NUMINAMATH_CALUDE_total_monthly_payment_l3287_328715


namespace NUMINAMATH_CALUDE_period_2_gym_class_size_l3287_328787

theorem period_2_gym_class_size : ℕ → Prop :=
  fun x => (2 * x - 5 = 11) → x = 8

#check period_2_gym_class_size

end NUMINAMATH_CALUDE_period_2_gym_class_size_l3287_328787


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l3287_328716

theorem cube_sum_inequality (a b c : ℝ) 
  (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1) 
  (h4 : a^3 + b^3 + c^3 = 1) : 
  a + b + c + a^2 + b^2 + c^2 ≤ 4 ∧ 
  (a + b + c + a^2 + b^2 + c^2 = 4 ↔ 
    (a = -1 ∧ b = 1 ∧ c = 1) ∨ 
    (a = -1 ∧ b = 1 ∧ c = 1) ∨ 
    (a = 1 ∧ b = -1 ∧ c = 1) ∨ 
    (a = 1 ∧ b = 1 ∧ c = -1) ∨ 
    (a = -1 ∧ b = 1 ∧ c = 1) ∨ 
    (a = 1 ∧ b = -1 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l3287_328716


namespace NUMINAMATH_CALUDE_polynomial_divisibility_and_divisor_l3287_328796

theorem polynomial_divisibility_and_divisor (m : ℤ) : 
  (∀ x : ℝ, (5 * x^2 - 9 * x + m) % (x - 2) = 0) →
  (m = -2 ∧ 2 % |m| = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_and_divisor_l3287_328796


namespace NUMINAMATH_CALUDE_temperature_difference_l3287_328702

def highest_temp : Int := 9
def lowest_temp : Int := -1

theorem temperature_difference : highest_temp - lowest_temp = 10 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l3287_328702


namespace NUMINAMATH_CALUDE_integer_solution_exists_l3287_328754

theorem integer_solution_exists : ∃ (x₁ x₂ y₁ y₂ y₃ y₄ : ℤ),
  (x₁ + x₂ = y₁ + y₂ + y₃ + y₄) ∧
  (x₁^2 + x₂^2 = y₁^2 + y₂^2 + y₃^2 + y₄^2) ∧
  (x₁^3 + x₂^3 = y₁^3 + y₂^3 + y₃^3 + y₄^3) ∧
  (abs x₁ > 2020) ∧ (abs x₂ > 2020) ∧
  (abs y₁ > 2020) ∧ (abs y₂ > 2020) ∧
  (abs y₃ > 2020) ∧ (abs y₄ > 2020) := by
  sorry

#print integer_solution_exists

end NUMINAMATH_CALUDE_integer_solution_exists_l3287_328754


namespace NUMINAMATH_CALUDE_twigs_to_find_l3287_328744

/-- The number of twigs already in the nest circle -/
def twigs_in_circle : ℕ := 12

/-- The number of additional twigs needed for each twig in the circle -/
def twigs_per_existing : ℕ := 6

/-- The fraction of needed twigs dropped by the tree -/
def tree_dropped_fraction : ℚ := 1/3

/-- Theorem stating how many twigs the bird still needs to find -/
theorem twigs_to_find : 
  (twigs_in_circle * twigs_per_existing : ℕ) - 
  (twigs_in_circle * twigs_per_existing : ℕ) * tree_dropped_fraction = 48 := by
  sorry

end NUMINAMATH_CALUDE_twigs_to_find_l3287_328744


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3287_328789

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3287_328789


namespace NUMINAMATH_CALUDE_point_C_coordinates_l3287_328721

-- Define the translation function
def translate (x y dx : ℝ) : ℝ × ℝ := (x + dx, y)

-- Define the symmetric point with respect to x-axis
def symmetricX (x y : ℝ) : ℝ × ℝ := (x, -y)

theorem point_C_coordinates :
  let A : ℝ × ℝ := (-1, 2)
  let B := translate A.1 A.2 2
  let C := symmetricX B.1 B.2
  C = (1, -2) := by sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l3287_328721


namespace NUMINAMATH_CALUDE_video_recorder_markup_percentage_l3287_328725

/-- Proves that the percentage markup over wholesale cost is 20% for a video recorder --/
theorem video_recorder_markup_percentage
  (wholesale_cost : ℝ)
  (employee_discount : ℝ)
  (employee_paid : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : employee_discount = 0.05)
  (h3 : employee_paid = 228)
  : ∃ (markup_percentage : ℝ),
    markup_percentage = 20 ∧
    employee_paid = (1 - employee_discount) * (wholesale_cost * (1 + markup_percentage / 100)) :=
by sorry

end NUMINAMATH_CALUDE_video_recorder_markup_percentage_l3287_328725


namespace NUMINAMATH_CALUDE_floor_of_negative_three_point_seven_l3287_328753

theorem floor_of_negative_three_point_seven :
  ⌊(-3.7 : ℝ)⌋ = -4 := by sorry

end NUMINAMATH_CALUDE_floor_of_negative_three_point_seven_l3287_328753


namespace NUMINAMATH_CALUDE_common_remainder_problem_l3287_328727

theorem common_remainder_problem (n : ℕ) : 
  n > 1 ∧ 
  n % 25 = n % 7 ∧ 
  n = 175 ∧ 
  ∀ m : ℕ, (m > 1 ∧ m % 25 = m % 7) → m ≥ n → 
  n % 25 = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_remainder_problem_l3287_328727


namespace NUMINAMATH_CALUDE_probability_white_and_black_l3287_328791

def total_balls : ℕ := 6
def red_balls : ℕ := 1
def white_balls : ℕ := 2
def black_balls : ℕ := 3
def drawn_balls : ℕ := 2

def favorable_outcomes : ℕ := white_balls * black_balls
def total_outcomes : ℕ := total_balls.choose drawn_balls

theorem probability_white_and_black :
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_white_and_black_l3287_328791


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3287_328765

theorem sin_cos_identity : 
  Real.sin (10 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) - 
  Real.cos (50 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3287_328765


namespace NUMINAMATH_CALUDE_zero_sponsorship_prob_high_sponsorship_prob_l3287_328774

-- Define the number of students and experts
def num_students : ℕ := 3
def num_experts : ℕ := 2

-- Define the probability of a "support" review
def support_prob : ℚ := 1/2

-- Define the function to calculate the probability of k successes in n trials
def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- Theorem for the probability of zero total sponsorship
theorem zero_sponsorship_prob :
  binomial_prob (num_students * num_experts) 0 support_prob = 1/64 := by sorry

-- Theorem for the probability of sponsorship exceeding 150,000 yuan
theorem high_sponsorship_prob :
  (binomial_prob (num_students * num_experts) 4 support_prob +
   binomial_prob (num_students * num_experts) 5 support_prob +
   binomial_prob (num_students * num_experts) 6 support_prob) = 11/32 := by sorry

end NUMINAMATH_CALUDE_zero_sponsorship_prob_high_sponsorship_prob_l3287_328774


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3287_328707

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3287_328707


namespace NUMINAMATH_CALUDE_subset_increase_l3287_328775

theorem subset_increase (m k : ℕ) (hm : m > 0) (hk : k ≥ 2) :
  let original_subsets := 2^m
  let new_subsets_one := 2^(m+1)
  let new_subsets_k := 2^(m+k)
  (new_subsets_one - original_subsets = 2^m) ∧
  (new_subsets_k - original_subsets = (2^k - 1) * 2^m) := by
  sorry

end NUMINAMATH_CALUDE_subset_increase_l3287_328775


namespace NUMINAMATH_CALUDE_inverse_function_property_l3287_328732

-- Define the function g
def g (x : ℝ) : ℝ := 1 + 2 * x

-- State the theorem
theorem inverse_function_property (f : ℝ → ℝ) :
  (∀ x, g (f x) = x) ∧ (∀ x, f (g x) = x) →
  f 1 = 0 := by sorry

end NUMINAMATH_CALUDE_inverse_function_property_l3287_328732


namespace NUMINAMATH_CALUDE_inequality_solution_l3287_328735

theorem inequality_solution (x : ℝ) : 2 ≤ (3*x)/(3*x-7) ∧ (3*x)/(3*x-7) < 6 ↔ 7/3 < x ∧ x < 42/15 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3287_328735


namespace NUMINAMATH_CALUDE_ad_arrangement_count_l3287_328762

def num_commercial_ads : ℕ := 4
def num_public_service_ads : ℕ := 2
def total_ads : ℕ := 6

theorem ad_arrangement_count :
  (num_commercial_ads.factorial) * (num_public_service_ads.factorial) = 48 :=
sorry

end NUMINAMATH_CALUDE_ad_arrangement_count_l3287_328762


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3287_328745

/-- A positive geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_2016 : a 2016 = a 2015 + 2 * a 2014)
  (h_mn : ∃ m n : ℕ, a m * a n = 16 * (a 1)^2) :
  ∃ m n : ℕ, (4 / m + 1 / n : ℝ) ≥ 3/2 ∧
    ∀ k l : ℕ, (4 / k + 1 / l : ℝ) ≥ 3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3287_328745


namespace NUMINAMATH_CALUDE_vector_problem_l3287_328741

def a : ℝ × ℝ := (-3, 1)
def b : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, 1)

theorem vector_problem :
  (∃ θ : ℝ, θ = Real.arccos ((-3 * 1 + 1 * (-2)) / (Real.sqrt 10 * Real.sqrt 5)) ∧ θ = 3 * π / 4) ∧
  (∃ k : ℝ, (∃ t : ℝ, t ≠ 0 ∧ c = t • (a + k • b)) ∧ k = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l3287_328741


namespace NUMINAMATH_CALUDE_tutors_next_meeting_l3287_328772

theorem tutors_next_meeting (elena fiona george harry : ℕ) 
  (h_elena : elena = 5)
  (h_fiona : fiona = 6)
  (h_george : george = 8)
  (h_harry : harry = 9) :
  Nat.lcm (Nat.lcm (Nat.lcm elena fiona) george) harry = 360 := by
  sorry

end NUMINAMATH_CALUDE_tutors_next_meeting_l3287_328772


namespace NUMINAMATH_CALUDE_divisor_sum_theorem_l3287_328737

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The property that the sum of divisors of three consecutive numbers is at most 8 -/
def divisor_sum_property (n : ℕ) : Prop :=
  num_divisors (n - 1) + num_divisors n + num_divisors (n + 1) ≤ 8

/-- Theorem stating that the divisor sum property holds if and only if n is 3, 4, or 6 -/
theorem divisor_sum_theorem (n : ℕ) (h : n ≥ 3) :
  divisor_sum_property n ↔ n = 3 ∨ n = 4 ∨ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_theorem_l3287_328737


namespace NUMINAMATH_CALUDE_train_distance_l3287_328779

/-- Proves that a train traveling at a speed derived from covering 2 miles in 2 minutes will travel 180 miles in 3 hours. -/
theorem train_distance (distance : ℝ) (time : ℝ) (hours : ℝ) : 
  distance = 2 → time = 2 → hours = 3 → (distance / time) * (hours * 60) = 180 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l3287_328779


namespace NUMINAMATH_CALUDE_painter_problem_l3287_328795

/-- Given two painters with a work ratio of 2:7 painting a total area of 270 square feet,
    the painter with the larger share paints 210 square feet. -/
theorem painter_problem (total_area : ℕ) (ratio_small : ℕ) (ratio_large : ℕ) :
  total_area = 270 →
  ratio_small = 2 →
  ratio_large = 7 →
  (ratio_large * total_area) / (ratio_small + ratio_large) = 210 :=
by sorry

end NUMINAMATH_CALUDE_painter_problem_l3287_328795


namespace NUMINAMATH_CALUDE_absolute_value_theorem_l3287_328763

theorem absolute_value_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 6*a*b) :
  |((a + 2*b) / (a - b))| = Real.sqrt 14 / 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_theorem_l3287_328763


namespace NUMINAMATH_CALUDE_circle_equation_l3287_328703

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line_l1 (x y : ℝ) : Prop := x - 3 * y = 0
def line_l2 (x y : ℝ) : Prop := x - y = 0

-- Define the conditions
def tangent_to_y_axis (c : Circle) : Prop :=
  c.center.1 = c.radius

def center_on_l1 (c : Circle) : Prop :=
  line_l1 c.center.1 c.center.2

def intersects_l2_with_chord (c : Circle) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_l2 x₁ y₁ ∧ line_l2 x₂ y₂ ∧
    (x₁ - c.center.1)^2 + (y₁ - c.center.2)^2 = c.radius^2 ∧
    (x₂ - c.center.1)^2 + (y₂ - c.center.2)^2 = c.radius^2 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8

-- Define the theorem
theorem circle_equation (c : Circle) :
  tangent_to_y_axis c →
  center_on_l1 c →
  intersects_l2_with_chord c →
  ((∀ x y, (x - 6 * Real.sqrt 2)^2 + (y - 2 * Real.sqrt 2)^2 = 72) ∨
   (∀ x y, (x + 6 * Real.sqrt 2)^2 + (y + 2 * Real.sqrt 2)^2 = 72)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3287_328703


namespace NUMINAMATH_CALUDE_clock_angle_at_2_20_clock_angle_at_2_20_is_50_l3287_328729

/-- The angle between clock hands at 2:20 -/
theorem clock_angle_at_2_20 : ℝ → Prop :=
  λ angle =>
    let total_degrees : ℝ := 360
    let hours_on_clock : ℕ := 12
    let minutes_in_hour : ℕ := 60
    let degrees_per_hour : ℝ := total_degrees / hours_on_clock
    let degrees_per_minute : ℝ := degrees_per_hour / minutes_in_hour
    let hour : ℕ := 2
    let minute : ℕ := 20
    let hour_hand_angle : ℝ := hour * degrees_per_hour + minute * degrees_per_minute
    let minute_hand_angle : ℝ := minute * (total_degrees / minutes_in_hour)
    let angle_diff : ℝ := |minute_hand_angle - hour_hand_angle|
    angle = min angle_diff (total_degrees - angle_diff)

/-- The smaller angle between the hour-hand and minute-hand of a clock at 2:20 is 50° -/
theorem clock_angle_at_2_20_is_50 : clock_angle_at_2_20 50 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_2_20_clock_angle_at_2_20_is_50_l3287_328729


namespace NUMINAMATH_CALUDE_rescue_center_dog_count_l3287_328726

/-- Calculates the final number of dogs in an animal rescue center after a series of events. -/
def final_dog_count (initial : ℕ) (moved_in : ℕ) (first_adoption : ℕ) (second_adoption : ℕ) : ℕ :=
  initial + moved_in - first_adoption - second_adoption

/-- Theorem stating that given specific values for initial count, dogs moved in, and adoptions,
    the final count of dogs is 200. -/
theorem rescue_center_dog_count :
  final_dog_count 200 100 40 60 = 200 := by
  sorry

#eval final_dog_count 200 100 40 60

end NUMINAMATH_CALUDE_rescue_center_dog_count_l3287_328726


namespace NUMINAMATH_CALUDE_impossible_transformation_l3287_328764

def sum_of_range (n : ℕ) : ℕ := n * (n + 1) / 2

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem impossible_transformation :
  is_odd (sum_of_range 2021) →
  ¬ ∃ (operations : ℕ), 
    ∃ (final_state : List ℕ),
      (final_state.length = 1 ∧ 
       final_state.head? = some 2048 ∧
       ∀ (intermediate_state : List ℕ),
         intermediate_state.sum = sum_of_range 2021) :=
by sorry

end NUMINAMATH_CALUDE_impossible_transformation_l3287_328764


namespace NUMINAMATH_CALUDE_subset_condition_l3287_328755

def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}

def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem subset_condition (m : ℝ) : S m ⊆ P → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l3287_328755


namespace NUMINAMATH_CALUDE_orange_sales_theorem_l3287_328756

def planned_daily_sales : ℕ := 10
def deviations : List ℤ := [4, -3, -5, 7, -8, 21, -6]
def selling_price : ℕ := 80
def shipping_fee : ℕ := 7

theorem orange_sales_theorem :
  let first_five_days_sales := planned_daily_sales * 5 + (deviations.take 5).sum
  let total_deviation := deviations.sum
  let total_sales := planned_daily_sales * 7 + total_deviation
  let total_earnings := total_sales * selling_price - total_sales * shipping_fee
  (first_five_days_sales = 45) ∧
  (total_deviation > 0) ∧
  (total_earnings = 5840) := by
  sorry

end NUMINAMATH_CALUDE_orange_sales_theorem_l3287_328756


namespace NUMINAMATH_CALUDE_cricket_run_rate_l3287_328736

/-- Calculates the required run rate for the remaining overs in a cricket game. -/
def required_run_rate (total_overs : ℕ) (first_overs : ℕ) (first_run_rate : ℚ) (target : ℕ) : ℚ :=
  let remaining_overs := total_overs - first_overs
  let runs_scored := first_run_rate * first_overs
  let runs_needed := target - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario. -/
theorem cricket_run_rate : required_run_rate 50 10 (34/10) 282 = 62/10 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_l3287_328736


namespace NUMINAMATH_CALUDE_round_windmill_iff_on_diagonal_l3287_328794

/-- A square in a 2D plane. -/
structure Square :=
  (A B C D : ℝ × ℝ)

/-- A point in a 2D plane. -/
def Point := ℝ × ℝ

/-- A line in a 2D plane. -/
structure Line :=
  (p1 p2 : Point)

/-- A windmill configuration. -/
structure Windmill :=
  (center : Point)
  (l1 l2 : Line)

/-- Checks if a point is inside a square. -/
def isInside (s : Square) (p : Point) : Prop := sorry

/-- Checks if two lines are perpendicular. -/
def arePerpendicular (l1 l2 : Line) : Prop := sorry

/-- Checks if a quadrilateral is cyclic. -/
def isCyclic (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if a point lies on the diagonal of a square. -/
def isOnDiagonal (s : Square) (p : Point) : Prop := sorry

/-- Theorem: A point P inside a square ABCD produces a round windmill for all
    possible configurations if and only if P lies on the diagonals of the square. -/
theorem round_windmill_iff_on_diagonal (s : Square) (p : Point) :
  isInside s p →
  (∀ (w : Windmill), w.center = p →
    arePerpendicular w.l1 w.l2 →
    (∃ W X Y Z, isCyclic W X Y Z)) ↔
  isOnDiagonal s p :=
sorry

end NUMINAMATH_CALUDE_round_windmill_iff_on_diagonal_l3287_328794


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_l3287_328719

theorem complex_arithmetic_expression : 
  (2*(3*(2*(3*(2*(3 * (2+1) * 2)+2)*2)+2)*2)+2) = 5498 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_l3287_328719


namespace NUMINAMATH_CALUDE_fraction_product_l3287_328710

theorem fraction_product : (2 : ℚ) / 9 * (5 : ℚ) / 8 = (5 : ℚ) / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l3287_328710


namespace NUMINAMATH_CALUDE_arithmetic_mean_increase_l3287_328724

theorem arithmetic_mean_increase (b₁ b₂ b₃ b₄ b₅ : ℝ) :
  let original_mean := (b₁ + b₂ + b₃ + b₄ + b₅) / 5
  let new_mean := ((b₁ + 30) + (b₂ + 30) + (b₃ + 30) + (b₄ + 30) + (b₅ + 30)) / 5
  new_mean = original_mean + 30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_increase_l3287_328724


namespace NUMINAMATH_CALUDE_cookie_price_calculation_l3287_328718

def trip_cost : ℝ := 5000
def hourly_wage : ℝ := 20
def hours_worked : ℝ := 10
def cookies_sold : ℝ := 24
def lottery_ticket_cost : ℝ := 10
def lottery_winnings : ℝ := 500
def gift_per_sister : ℝ := 500
def num_sisters : ℝ := 2
def additional_money_needed : ℝ := 3214

theorem cookie_price_calculation (trip_cost hourly_wage hours_worked 
  cookies_sold lottery_ticket_cost lottery_winnings gift_per_sister 
  num_sisters additional_money_needed : ℝ) :
  let total_earnings := hourly_wage * hours_worked + 
    lottery_winnings + gift_per_sister * num_sisters - lottery_ticket_cost
  let cookie_revenue := trip_cost - total_earnings
  cookie_revenue / cookies_sold = 204.33 := by
  sorry

end NUMINAMATH_CALUDE_cookie_price_calculation_l3287_328718


namespace NUMINAMATH_CALUDE_sand_pile_volume_l3287_328778

/-- Theorem: Volume of a conical sand pile --/
theorem sand_pile_volume (diameter : Real) (height_ratio : Real) :
  diameter = 8 →
  height_ratio = 0.75 →
  let height := height_ratio * diameter
  let radius := diameter / 2
  let volume := (1 / 3) * Real.pi * radius^2 * height
  volume = 32 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sand_pile_volume_l3287_328778


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l3287_328771

theorem quadratic_inequality_roots (b : ℝ) : 
  (∀ x : ℝ, -x^2 + b*x - 5 < 0 ↔ x < 1 ∨ x > 5) → b = 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l3287_328771


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3287_328780

theorem divisibility_equivalence (n : ℕ) :
  5 ∣ (1^n + 2^n + 3^n + 4^n) ↔ n % 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3287_328780


namespace NUMINAMATH_CALUDE_one_sheet_removal_median_l3287_328734

/-- Represents a collection of notes with pages and sheets -/
structure Notes where
  total_pages : ℕ
  total_sheets : ℕ
  last_sheet_pages : ℕ
  mk_notes_valid : total_pages = 2 * (total_sheets - 1) + last_sheet_pages

/-- Calculates the median page number after removing sheets -/
def median_after_removal (notes : Notes) (sheets_removed : ℕ) : ℕ :=
  (notes.total_pages - 2 * sheets_removed + 1) / 2

/-- Theorem stating that removing one sheet results in a median of 36 -/
theorem one_sheet_removal_median (notes : Notes)
  (h1 : notes.total_pages = 65)
  (h2 : notes.total_sheets = 33)
  (h3 : notes.last_sheet_pages = 1) :
  median_after_removal notes 1 = 36 := by
  sorry

#check one_sheet_removal_median

end NUMINAMATH_CALUDE_one_sheet_removal_median_l3287_328734


namespace NUMINAMATH_CALUDE_product_digit_sum_l3287_328768

/-- The number of digits in each factor -/
def n : ℕ := 2012

/-- The first factor in the multiplication -/
def first_factor : ℕ := (10^n - 1) * 4 / 9

/-- The second factor in the multiplication -/
def second_factor : ℕ := 10^n - 1

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (k : ℕ) : ℕ :=
  if k < 10 then k else k % 10 + sum_of_digits (k / 10)

/-- The main theorem to be proved -/
theorem product_digit_sum :
  sum_of_digits (first_factor * second_factor) = 18108 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l3287_328768


namespace NUMINAMATH_CALUDE_johns_age_l3287_328783

theorem johns_age (john dad : ℕ) 
  (h1 : john + 34 = dad) 
  (h2 : john + dad = 84) : 
  john = 25 := by
sorry

end NUMINAMATH_CALUDE_johns_age_l3287_328783


namespace NUMINAMATH_CALUDE_inequality_problem_l3287_328701

theorem inequality_problem (x y z : ℝ) (a : ℝ) : 
  (x^2 + y^2 + z^2 = 1) → 
  ((-3 : ℝ) ≤ x + 2*y + 2*z ∧ x + 2*y + 2*z ≤ 3) ∧ 
  ((∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 3| + a / 2 ≥ x + 2*y + 2*z) ↔ 
   (a ≥ 4 ∨ a ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l3287_328701


namespace NUMINAMATH_CALUDE_childrenWithSameCardsTheorem_l3287_328730

/-- Represents the number of children who can form a specific word -/
structure ChildrenCount where
  mama : Nat
  nyanya : Nat
  manya : Nat

/-- Calculates the number of children with all three cards the same -/
def childrenWithSameCards (c : ChildrenCount) : Nat :=
  c.mama + c.nyanya - c.manya

/-- Theorem stating that the number of children with all three cards the same
    is equal to the total number of children who can form either "MAMA" or "NYANYA"
    minus the number of children who can form "MANYA" -/
theorem childrenWithSameCardsTheorem (c : ChildrenCount) :
  childrenWithSameCards c = c.mama + c.nyanya - c.manya := by
  sorry

#eval childrenWithSameCards { mama := 20, nyanya := 30, manya := 40 }

end NUMINAMATH_CALUDE_childrenWithSameCardsTheorem_l3287_328730


namespace NUMINAMATH_CALUDE_current_visitors_count_l3287_328733

/-- The number of visitors to the Buckingham palace on the previous day -/
def previous_visitors : ℕ := 600

/-- The additional number of visitors compared to the previous day -/
def additional_visitors : ℕ := 61

/-- Theorem: The number of visitors to the Buckingham palace on the current day is 661 -/
theorem current_visitors_count : previous_visitors + additional_visitors = 661 := by
  sorry

end NUMINAMATH_CALUDE_current_visitors_count_l3287_328733


namespace NUMINAMATH_CALUDE_horner_v3_value_l3287_328799

def f (x : ℝ) : ℝ := 3*x^6 + 5*x^5 + 6*x^4 + 79*x^3 - 8*x^2 + 35*x + 12

def horner_v3 (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  v2 * x + 79

theorem horner_v3_value :
  horner_v3 f (-4) = -57 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_l3287_328799


namespace NUMINAMATH_CALUDE_disjoint_iff_valid_range_l3287_328767

/-- Set M represents a unit circle centered at the origin -/
def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Set N represents a diamond centered at (1, 1) with side length a/√2 -/
def N (a : ℝ) : Set (ℝ × ℝ) := {p | |p.1 - 1| + |p.2 - 1| = a}

/-- The range of a for which M and N are disjoint -/
def valid_range : Set ℝ := {a | a < 2 - Real.sqrt 2 ∨ a > 2 + Real.sqrt 2}

theorem disjoint_iff_valid_range (a : ℝ) : 
  Disjoint (M : Set (ℝ × ℝ)) (N a) ↔ a ∈ valid_range := by sorry

end NUMINAMATH_CALUDE_disjoint_iff_valid_range_l3287_328767


namespace NUMINAMATH_CALUDE_power_equality_l3287_328742

theorem power_equality (y x : ℕ) (h1 : 9^y = 3^x) (h2 : y = 6) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3287_328742


namespace NUMINAMATH_CALUDE_alex_total_fish_is_4000_l3287_328759

/-- The number of fish Brian catches per trip -/
def brian_fish_per_trip : ℕ := 400

/-- The number of times Chris goes fishing -/
def chris_trips : ℕ := 10

/-- The number of times Alex goes fishing -/
def alex_trips : ℕ := chris_trips / 2

/-- The number of fish Alex catches per trip -/
def alex_fish_per_trip : ℕ := brian_fish_per_trip * 2

/-- The total number of fish Alex caught -/
def alex_total_fish : ℕ := alex_trips * alex_fish_per_trip

theorem alex_total_fish_is_4000 : alex_total_fish = 4000 := by
  sorry

end NUMINAMATH_CALUDE_alex_total_fish_is_4000_l3287_328759


namespace NUMINAMATH_CALUDE_bella_steps_l3287_328781

/-- The distance between houses in miles -/
def distance : ℝ := 3

/-- The waiting time for Ella in minutes -/
def wait_time : ℝ := 10

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℝ := 4

/-- The length of Bella's step in feet -/
def step_length : ℝ := 3

/-- The number of feet in a mile -/
def feet_per_mile : ℝ := 5280

/-- The number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := 1328

theorem bella_steps :
  ∃ (bella_speed : ℝ),
    bella_speed > 0 ∧
    (wait_time * bella_speed + 
     (distance * feet_per_mile - wait_time * bella_speed) / (bella_speed * (1 + speed_ratio))) * 
    bella_speed / step_length = steps_taken := by
  sorry

end NUMINAMATH_CALUDE_bella_steps_l3287_328781

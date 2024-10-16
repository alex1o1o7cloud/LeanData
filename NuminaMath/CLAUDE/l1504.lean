import Mathlib

namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_area_l1504_150420

/-- Given four line segments with lengths 13, 14, 15, and 24,
    they form a cyclic quadrilateral with an area of √61560. -/
theorem cyclic_quadrilateral_area : 
  let a : ℝ := 13
  let b : ℝ := 14
  let c : ℝ := 15
  let d : ℝ := 24
  let s : ℝ := (a + b + c + d) / 2
  let area : ℝ := Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d))
  area = Real.sqrt 61560 := by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_area_l1504_150420


namespace NUMINAMATH_CALUDE_odd_square_difference_plus_one_is_perfect_square_l1504_150464

theorem odd_square_difference_plus_one_is_perfect_square 
  (m n : ℤ) 
  (h_m_odd : Odd m) 
  (h_n_odd : Odd n) 
  (h_divides : (m^2 - n^2 + 1) ∣ (n^2 - 1)) : 
  ∃ k : ℤ, m^2 - n^2 + 1 = k^2 :=
sorry

end NUMINAMATH_CALUDE_odd_square_difference_plus_one_is_perfect_square_l1504_150464


namespace NUMINAMATH_CALUDE_multiply_and_add_l1504_150406

theorem multiply_and_add : (23 * 37) + 16 = 867 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_l1504_150406


namespace NUMINAMATH_CALUDE_D_sqrt_rationality_l1504_150481

def D (a : ℤ) : ℚ :=
  let b := a + 2
  let c := a * b
  (a^2 + b^2 + c^2 : ℚ)

theorem D_sqrt_rationality :
  ∃ a₁ a₂ : ℤ, (∃ q : ℚ, q^2 = D a₁) ∧ (∀ q : ℚ, q^2 ≠ D a₂) :=
sorry

end NUMINAMATH_CALUDE_D_sqrt_rationality_l1504_150481


namespace NUMINAMATH_CALUDE_a_most_stable_l1504_150459

/-- Represents a participant in the shooting test -/
inductive Participant
  | A
  | B
  | C
  | D

/-- The variance of a participant's scores -/
def variance : Participant → ℝ
  | Participant.A => 0.12
  | Participant.B => 0.25
  | Participant.C => 0.35
  | Participant.D => 0.46

/-- A participant has the most stable performance if their variance is the lowest -/
def hasMostStablePerformance (p : Participant) : Prop :=
  ∀ q : Participant, variance p ≤ variance q

/-- Theorem: Participant A has the most stable performance -/
theorem a_most_stable : hasMostStablePerformance Participant.A := by
  sorry

end NUMINAMATH_CALUDE_a_most_stable_l1504_150459


namespace NUMINAMATH_CALUDE_min_value_theorem_l1504_150476

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 6*x + 36/x^2 ≥ 12 * (4^(1/4)) ∧
  (x^2 + 6*x + 36/x^2 = 12 * (4^(1/4)) ↔ x = 36^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1504_150476


namespace NUMINAMATH_CALUDE_system_solutions_l1504_150469

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x * (x^2 - 3*y^2) = 16
def equation2 (x y : ℝ) : Prop := y * (3*x^2 - y^2) = 88

-- Define the approximate equality for real numbers
def approx_equal (a b : ℝ) (ε : ℝ) : Prop := abs (a - b) < ε

-- Theorem statement
theorem system_solutions :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    -- Exact solution
    equation1 x₁ y₁ ∧ equation2 x₁ y₁ ∧ x₁ = 4 ∧ y₁ = 2 ∧
    -- Approximate solutions
    equation1 x₂ y₂ ∧ equation2 x₂ y₂ ∧ 
    approx_equal x₂ (-3.7) 0.1 ∧ approx_equal y₂ 2.5 0.1 ∧
    equation1 x₃ y₃ ∧ equation2 x₃ y₃ ∧ 
    approx_equal x₃ (-0.3) 0.1 ∧ approx_equal y₃ (-4.5) 0.1 :=
by sorry


end NUMINAMATH_CALUDE_system_solutions_l1504_150469


namespace NUMINAMATH_CALUDE_problem_statement_l1504_150448

theorem problem_statement (x : ℝ) (h : x = Real.sqrt 2) : 
  (x + 2)^2 - 4*x*(x + 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1504_150448


namespace NUMINAMATH_CALUDE_probability_is_34_39_l1504_150422

-- Define the total number of students and enrollments
def total_students : ℕ := 40
def french_enrollment : ℕ := 28
def spanish_enrollment : ℕ := 26
def german_enrollment : ℕ := 15
def french_spanish : ℕ := 10
def french_german : ℕ := 6
def spanish_german : ℕ := 8
def all_three : ℕ := 3

-- Define the function to calculate the probability
def probability_different_classes : ℚ := by sorry

-- Theorem statement
theorem probability_is_34_39 : 
  probability_different_classes = 34 / 39 := by sorry

end NUMINAMATH_CALUDE_probability_is_34_39_l1504_150422


namespace NUMINAMATH_CALUDE_negation_of_forall_greater_than_one_l1504_150483

theorem negation_of_forall_greater_than_one (x : ℝ) : 
  ¬(∀ x > 1, x - 1 > Real.log x) ↔ ∃ x > 1, x - 1 ≤ Real.log x :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_greater_than_one_l1504_150483


namespace NUMINAMATH_CALUDE_daves_trays_l1504_150412

/-- Given that Dave can carry 9 trays at a time, picked up 17 trays from one table,
    and made 8 trips in total, prove that he picked up 55 trays from the second table. -/
theorem daves_trays (trays_per_trip : ℕ) (trips : ℕ) (trays_first_table : ℕ)
    (h1 : trays_per_trip = 9)
    (h2 : trips = 8)
    (h3 : trays_first_table = 17) :
    trips * trays_per_trip - trays_first_table = 55 := by
  sorry

end NUMINAMATH_CALUDE_daves_trays_l1504_150412


namespace NUMINAMATH_CALUDE_x_value_l1504_150438

theorem x_value : ∃ x : ℝ, (0.25 * x = 0.15 * 1600 - 15) ∧ (x = 900) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1504_150438


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l1504_150403

theorem ratio_x_to_y (x y : ℝ) (h : (12*x - 5*y) / (15*x - 3*y) = 4/7) : 
  x / y = 23/24 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l1504_150403


namespace NUMINAMATH_CALUDE_sector_area_l1504_150445

theorem sector_area (θ : Real) (chord_length : Real) (area : Real) : 
  θ = 2 ∧ 
  chord_length = 2 * Real.sin 1 ∧ 
  area = (1 / 2) * 1 * θ →
  area = 1 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l1504_150445


namespace NUMINAMATH_CALUDE_steak_weight_problem_l1504_150449

theorem steak_weight_problem (original_weight : ℝ) : 
  (0.8 * (0.5 * original_weight) = 12) → original_weight = 30 := by
  sorry

end NUMINAMATH_CALUDE_steak_weight_problem_l1504_150449


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_parametric_quadratic_inequality_solution_l1504_150473

-- Define the quadratic function
def f (x : ℝ) := -x^2 - 2*x + 3

-- Define the parametric quadratic function
def g (a : ℝ) (x : ℝ) := -x^2 - 2*x + a

theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = {x : ℝ | x < -3 ∨ x > 1} := by sorry

theorem parametric_quadratic_inequality_solution (a : ℝ) :
  ({x : ℝ | g a x < 0} = Set.univ) ↔ a < -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_parametric_quadratic_inequality_solution_l1504_150473


namespace NUMINAMATH_CALUDE_range_of_t_largest_circle_range_of_t_with_P_inside_l1504_150436

-- Define the circle equation
def circle_equation (x y t : ℝ) : Prop :=
  x^2 + y^2 - 2*(t+3)*x + 2*(1-4*t^2)*y + 16*t^4 + 9 = 0

-- Define the point P
def point_P (t : ℝ) : ℝ × ℝ := (3, 4*t^2)

-- Theorem for the range of t
theorem range_of_t : ∀ t : ℝ, circle_equation x y t → -1/7 < t ∧ t < 1 :=
sorry

-- Theorem for the circle with the largest area
theorem largest_circle : ∃ t : ℝ, 
  ∀ x y : ℝ, circle_equation x y t → (x - 24/7)^2 + (y + 13/49)^2 = 16/7 :=
sorry

-- Theorem for the range of t when P is inside the circle
theorem range_of_t_with_P_inside : 
  ∀ t : ℝ, (∀ x y : ℝ, circle_equation x y t → 
    (point_P t).1^2 + (point_P t).2^2 < (x - (t+3))^2 + (y - (4*t^2-1))^2) →
  0 < t ∧ t < 3/4 :=
sorry

end NUMINAMATH_CALUDE_range_of_t_largest_circle_range_of_t_with_P_inside_l1504_150436


namespace NUMINAMATH_CALUDE_box_2_3_neg1_l1504_150455

def box (a b c : ℤ) : ℚ := (a ^ b : ℚ) - (b ^ c : ℚ) + (c ^ a : ℚ)

theorem box_2_3_neg1 : box 2 3 (-1) = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_box_2_3_neg1_l1504_150455


namespace NUMINAMATH_CALUDE_impossible_arrangement_l1504_150485

/-- Represents a domino tile with two numbers -/
structure Domino where
  first : Nat
  second : Nat
  first_range : first ≤ 6
  second_range : second ≤ 6

/-- The set of all 28 standard domino tiles -/
def StandardDominoes : Finset Domino := sorry

/-- Counts the number of even numbers on all tiles -/
def CountEvenNumbers (tiles : Finset Domino) : Nat := sorry

/-- Counts the number of odd numbers on all tiles -/
def CountOddNumbers (tiles : Finset Domino) : Nat := sorry

/-- Defines a valid arrangement of domino tiles -/
def ValidArrangement (arrangement : List Domino) : Prop := sorry

theorem impossible_arrangement :
  CountEvenNumbers StandardDominoes = 32 →
  CountOddNumbers StandardDominoes = 24 →
  ¬∃ (arrangement : List Domino), 
    (arrangement.length = StandardDominoes.card) ∧ 
    (ValidArrangement arrangement) := by
  sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l1504_150485


namespace NUMINAMATH_CALUDE_solution_set_l1504_150488

-- Define the variables
variable (a b : ℝ)

-- Define the conditions
def condition1 : Prop := ∀ x : ℝ, (a - b) * x + a + 2 * b > 0 ↔ x > 1 / 2
def condition2 : Prop := a > 0

-- Define the theorem
theorem solution_set (h1 : condition1 a b) (h2 : condition2 a) :
  ∀ x : ℝ, a * x < b ↔ x < -1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_l1504_150488


namespace NUMINAMATH_CALUDE_cube_preserves_order_l1504_150468

theorem cube_preserves_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l1504_150468


namespace NUMINAMATH_CALUDE_interest_calculation_l1504_150489

/-- Problem Statement: A sum is divided into two parts with specific interest conditions. -/
theorem interest_calculation (total_sum second_sum : ℕ) 
  (first_rate second_rate : ℚ) (first_years : ℕ) : 
  total_sum = 2795 →
  second_sum = 1720 →
  first_rate = 3/100 →
  second_rate = 5/100 →
  first_years = 8 →
  ∃ (second_years : ℕ),
    (total_sum - second_sum) * first_rate * first_years = 
    second_sum * second_rate * second_years ∧
    second_years = 3 := by
  sorry


end NUMINAMATH_CALUDE_interest_calculation_l1504_150489


namespace NUMINAMATH_CALUDE_ball_picking_problem_l1504_150496

/-- Represents a bag with black and white balls -/
structure BallBag where
  total_balls : ℕ
  white_balls : ℕ
  black_balls : ℕ
  h_total : total_balls = white_balls + black_balls

/-- The probability of picking a white ball -/
def prob_white (bag : BallBag) : ℚ :=
  bag.white_balls / bag.total_balls

theorem ball_picking_problem (bag : BallBag) 
  (h_total : bag.total_balls = 4)
  (h_prob : prob_white bag = 1/2) :
  (bag.white_balls = 2) ∧ 
  (1/3 : ℚ) = (bag.white_balls * (bag.white_balls - 1) + bag.black_balls * (bag.black_balls - 1)) / 
               (bag.total_balls * (bag.total_balls - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ball_picking_problem_l1504_150496


namespace NUMINAMATH_CALUDE_sequence_general_term_l1504_150428

/-- Given a sequence {a_n} where S_n represents the sum of the first n terms,
    prove that the general term a_n can be expressed as a_1 + (n-1)d,
    where d is the common difference (a_2 - a_1). -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ k, S k = k / 2 * (a 1 + a k)) →
  ∃ d : ℝ, d = a 2 - a 1 ∧ ∀ m, a m = a 1 + (m - 1) * d :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1504_150428


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1504_150447

theorem sufficient_not_necessary (a : ℝ) : 
  (a < -1 → |a| > 1) ∧ ¬(|a| > 1 → a < -1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1504_150447


namespace NUMINAMATH_CALUDE_divisor_is_one_l1504_150423

theorem divisor_is_one (x d : ℕ) (k n : ℤ) : 
  x % d = 5 →
  (x + 17) % 41 = 22 →
  x = k * d + 5 →
  x = 41 * n + 5 →
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_divisor_is_one_l1504_150423


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1504_150418

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1504_150418


namespace NUMINAMATH_CALUDE_place_four_men_five_women_l1504_150487

/-- The number of ways to place men and women into groups -/
def placeInGroups (numMen numWomen : ℕ) : ℕ :=
  let twoGroup := numMen * numWomen
  let threeGroup := (numMen - 1) * (numWomen.choose 2)
  let fourGroup := 1  -- As all remaining people form this group
  twoGroup * threeGroup * fourGroup

/-- Theorem stating the number of ways to place 4 men and 5 women into specific groups -/
theorem place_four_men_five_women :
  placeInGroups 4 5 = 360 := by
  sorry

#eval placeInGroups 4 5

end NUMINAMATH_CALUDE_place_four_men_five_women_l1504_150487


namespace NUMINAMATH_CALUDE_total_theme_parks_eq_395_l1504_150480

/-- The number of theme parks in four towns -/
def total_theme_parks (jamestown venice marina_del_ray newport_beach : ℕ) : ℕ :=
  jamestown + venice + marina_del_ray + newport_beach

/-- Theorem: The total number of theme parks in four towns is 395 -/
theorem total_theme_parks_eq_395 :
  ∃ (jamestown venice marina_del_ray newport_beach : ℕ),
    jamestown = 35 ∧
    venice = jamestown + 40 ∧
    marina_del_ray = jamestown + 60 ∧
    newport_beach = 2 * marina_del_ray ∧
    total_theme_parks jamestown venice marina_del_ray newport_beach = 395 :=
by sorry

end NUMINAMATH_CALUDE_total_theme_parks_eq_395_l1504_150480


namespace NUMINAMATH_CALUDE_sector_radius_l1504_150472

/-- Given a circular sector with area 240π and arc length 20π, prove that its radius is 24. -/
theorem sector_radius (A : ℝ) (L : ℝ) (r : ℝ) : 
  A = 240 * Real.pi → L = 20 * Real.pi → A = (1/2) * r^2 * (L/r) → r = 24 := by
sorry

end NUMINAMATH_CALUDE_sector_radius_l1504_150472


namespace NUMINAMATH_CALUDE_remainder_1897_2048_mod_600_l1504_150404

theorem remainder_1897_2048_mod_600 : (1897 * 2048) % 600 = 256 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1897_2048_mod_600_l1504_150404


namespace NUMINAMATH_CALUDE_total_blocks_is_55_l1504_150474

/-- Calculates the total number of blocks in Thomas's stacks --/
def total_blocks : ℕ :=
  let first_stack := 7
  let second_stack := first_stack + 3
  let third_stack := second_stack - 6
  let fourth_stack := third_stack + 10
  let fifth_stack := 2 * second_stack
  first_stack + second_stack + third_stack + fourth_stack + fifth_stack

/-- Theorem stating that the total number of blocks is 55 --/
theorem total_blocks_is_55 : total_blocks = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_blocks_is_55_l1504_150474


namespace NUMINAMATH_CALUDE_stating_rabbit_distribution_theorem_l1504_150490

/-- Represents the number of pet stores --/
def num_stores : ℕ := 5

/-- Represents the number of parent rabbits --/
def num_parents : ℕ := 3

/-- Represents the number of offspring rabbits --/
def num_offspring : ℕ := 3

/-- Represents the total number of rabbits --/
def total_rabbits : ℕ := num_parents + num_offspring

/-- 
  Represents the number of ways to distribute rabbits to pet stores
  such that no store gets both a parent and a child
--/
def distribution_count : ℕ := 398

/-- 
  Theorem stating that the number of ways to distribute the rabbits
  under the given conditions is equal to distribution_count
--/
theorem rabbit_distribution_theorem : 
  (∃ (f : Fin total_rabbits → Fin num_stores), 
    ∀ (i j : Fin total_rabbits), 
      i.val < num_parents ∧ j.val ≥ num_parents → f i ≠ f j) →
  distribution_count = 398 := by
  sorry

end NUMINAMATH_CALUDE_stating_rabbit_distribution_theorem_l1504_150490


namespace NUMINAMATH_CALUDE_digit_problem_l1504_150450

def Digit := Fin 9

def isConsecutive (a b : Digit) : Prop :=
  (a.val + 1 = b.val) ∨ (b.val + 1 = a.val)

def divides6DigitNumber (x : Nat) (a b c d e f : Digit) : Prop :=
  ∀ (perm : Fin 6 → Fin 6), x ∣ (100000 * (perm 0).val + 10000 * (perm 1).val + 1000 * (perm 2).val + 100 * (perm 3).val + 10 * (perm 4).val + (perm 5).val)

theorem digit_problem (a b c d e f : Digit) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f) :
  (∃ (x y : Digit), x ≠ y ∧ isConsecutive x y) ∧
  (∀ x : Nat, divides6DigitNumber x a b c d e f ↔ x = 1 ∨ x = 3 ∨ x = 9) :=
by sorry

end NUMINAMATH_CALUDE_digit_problem_l1504_150450


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l1504_150435

theorem geometric_sequence_common_ratio_sum 
  (k : ℝ) (p r : ℝ) (h_distinct : p ≠ r) (h_nonzero : k ≠ 0) 
  (h_equation : k * p^2 - k * r^2 = 3 * (k * p - k * r)) : 
  p + r = 3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l1504_150435


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribed_cube_l1504_150446

theorem sphere_surface_area_circumscribed_cube (edge_length : ℝ) 
  (h : edge_length = 2) : 
  4 * Real.pi * (edge_length * Real.sqrt 3 / 2) ^ 2 = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_circumscribed_cube_l1504_150446


namespace NUMINAMATH_CALUDE_wall_width_theorem_l1504_150442

theorem wall_width_theorem (width height length : ℝ) (volume : ℝ) :
  height = 6 * width →
  length = 7 * height →
  volume = width * height * length →
  volume = 16128 →
  width = 4 := by
sorry

end NUMINAMATH_CALUDE_wall_width_theorem_l1504_150442


namespace NUMINAMATH_CALUDE_lawn_length_is_80_l1504_150466

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  width : ℝ
  length : ℝ
  roadWidth : ℝ
  travelCost : ℝ
  totalCost : ℝ

/-- Calculates the area of the roads on the lawn -/
def roadArea (l : LawnWithRoads) : ℝ :=
  l.roadWidth * l.length + l.roadWidth * l.width - l.roadWidth * l.roadWidth

/-- Theorem: Given the specified conditions, the length of the lawn is 80 meters -/
theorem lawn_length_is_80 (l : LawnWithRoads) 
    (h1 : l.width = 60)
    (h2 : l.roadWidth = 10)
    (h3 : l.travelCost = 2)
    (h4 : l.totalCost = 2600)
    (h5 : l.totalCost = l.travelCost * roadArea l) :
  l.length = 80 := by
  sorry

end NUMINAMATH_CALUDE_lawn_length_is_80_l1504_150466


namespace NUMINAMATH_CALUDE_range_of_m_l1504_150417

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, ¬(Real.sin x + Real.cos x > m)) ∧ 
  (∀ x : ℝ, x^2 + m*x + 1 > 0) ↔ 
  -Real.sqrt 2 ≤ m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1504_150417


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1504_150499

theorem max_sum_of_factors (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  a * b * c * d = 360 →
  a + b + c + d ≤ 66 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1504_150499


namespace NUMINAMATH_CALUDE_green_marbles_count_l1504_150444

theorem green_marbles_count (total : ℕ) (white : ℕ) 
  (h1 : white = 40)
  (h2 : (1 : ℚ) / 4 + (1 : ℚ) / 3 + (1 : ℚ) / 6 + (white : ℚ) / total = 1) :
  ⌊(1 : ℚ) / 6 * total⌋ = 27 := by
  sorry

end NUMINAMATH_CALUDE_green_marbles_count_l1504_150444


namespace NUMINAMATH_CALUDE_function_relation_l1504_150402

/-- Given functions h and k, prove that C = 3D/4 -/
theorem function_relation (C D : ℝ) (h k : ℝ → ℝ) : 
  D ≠ 0 →
  (∀ x, h x = 2 * C * x - 3 * D^2) →
  (∀ x, k x = D * x) →
  h (k 2) = 0 →
  C = 3 * D / 4 := by
sorry

end NUMINAMATH_CALUDE_function_relation_l1504_150402


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1504_150424

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 3 + a 10 + a 11 = 40 →
  a 6 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1504_150424


namespace NUMINAMATH_CALUDE_rotated_region_volume_is_19pi_l1504_150414

/-- The volume of a solid formed by rotating a region about the y-axis. The region consists of:
    1. A vertical strip of 7 unit squares high and 1 unit wide along the y-axis.
    2. A horizontal strip of 3 unit squares wide and 2 units high along the x-axis, 
       starting from the top of the vertical strip. -/
def rotated_region_volume : ℝ := sorry

/-- The theorem states that the volume of the rotated region is equal to 19π cubic units. -/
theorem rotated_region_volume_is_19pi : rotated_region_volume = 19 * Real.pi := by sorry

end NUMINAMATH_CALUDE_rotated_region_volume_is_19pi_l1504_150414


namespace NUMINAMATH_CALUDE_probability_neither_correct_l1504_150494

theorem probability_neither_correct (P_A P_B P_AB : ℝ) 
  (h1 : P_A = 0.75)
  (h2 : P_B = 0.70)
  (h3 : P_AB = 0.65)
  (h4 : 0 ≤ P_A ∧ P_A ≤ 1)
  (h5 : 0 ≤ P_B ∧ P_B ≤ 1)
  (h6 : 0 ≤ P_AB ∧ P_AB ≤ 1) :
  1 - (P_A + P_B - P_AB) = 0.20 := by
  sorry

#check probability_neither_correct

end NUMINAMATH_CALUDE_probability_neither_correct_l1504_150494


namespace NUMINAMATH_CALUDE_find_divisor_l1504_150456

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = quotient * divisor + remainder →
  dividend = 265 →
  quotient = 12 →
  remainder = 1 →
  divisor = 22 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1504_150456


namespace NUMINAMATH_CALUDE_no_real_roots_iff_k_gt_9_l1504_150432

theorem no_real_roots_iff_k_gt_9 (k : ℝ) : 
  (∀ x : ℝ, x^2 + k ≠ 6*x) ↔ k > 9 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_k_gt_9_l1504_150432


namespace NUMINAMATH_CALUDE_nine_digit_repeat_gcd_l1504_150429

theorem nine_digit_repeat_gcd : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), 100 ≤ m ∧ m < 1000 → 
    (∃ (k : ℕ), k = m * 1001001 ∧ 
      Nat.gcd n k = n)) ∧ 
  (∀ (d : ℕ), d > n → 
    ∃ (m₁ m₂ : ℕ), 100 ≤ m₁ ∧ m₁ < 1000 ∧ 100 ≤ m₂ ∧ m₂ < 1000 ∧ 
      Nat.gcd (m₁ * 1001001) (m₂ * 1001001) < d) :=
by sorry

end NUMINAMATH_CALUDE_nine_digit_repeat_gcd_l1504_150429


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1504_150416

/-- A line that is tangent to a circle and intersects a parabola -/
structure TangentLine where
  -- The line equation: ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The circle equation: x^4 + y^2 = 8
  circle : (x y : ℝ) → x^4 + y^2 = 8
  -- The parabola equation: y^2 = 4x
  parabola : (x y : ℝ) → y^2 = 4*x
  -- The line is tangent to the circle
  is_tangent : ∃ (x y : ℝ), a*x + b*y + c = 0 ∧ x^4 + y^2 = 8
  -- The line intersects the parabola at two points
  intersects_parabola : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    a*x₁ + b*y₁ + c = 0 ∧ y₁^2 = 4*x₁ ∧
    a*x₂ + b*y₂ + c = 0 ∧ y₂^2 = 4*x₂
  -- The circle passes through the origin
  origin_on_circle : 0^4 + 0^2 = 8

/-- The theorem stating the equation of the tangent line -/
theorem tangent_line_equation (l : TangentLine) : 
  (l.a = 1 ∧ l.b = -1 ∧ l.c = -4) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -4) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1504_150416


namespace NUMINAMATH_CALUDE_ceiling_times_self_210_l1504_150498

theorem ceiling_times_self_210 : ∃ x : ℝ, ⌈x⌉ * x = 210 ∧ x = 14 := by sorry

end NUMINAMATH_CALUDE_ceiling_times_self_210_l1504_150498


namespace NUMINAMATH_CALUDE_intersection_equals_Q_l1504_150425

-- Define the sets P and Q
def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {x | x^2 ≤ 1}

-- Theorem statement
theorem intersection_equals_Q : P ∩ Q = Q := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_Q_l1504_150425


namespace NUMINAMATH_CALUDE_tan_five_pi_fourth_l1504_150467

theorem tan_five_pi_fourth : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_fourth_l1504_150467


namespace NUMINAMATH_CALUDE_special_triangle_line_BC_l1504_150477

/-- A triangle ABC with vertex A at (-4, 2) and two medians on specific lines -/
structure SpecialTriangle where
  /-- Vertex A of the triangle -/
  A : ℝ × ℝ
  /-- The line containing one median -/
  median1 : ℝ → ℝ → ℝ
  /-- The line containing another median -/
  median2 : ℝ → ℝ → ℝ
  /-- Condition: A is at (-4, 2) -/
  h_A : A = (-4, 2)
  /-- Condition: One median lies on 3x - 2y + 2 = 0 -/
  h_median1 : median1 x y = 3*x - 2*y + 2
  /-- Condition: Another median lies on 3x + 5y - 12 = 0 -/
  h_median2 : median2 x y = 3*x + 5*y - 12

/-- The equation of line BC in the special triangle -/
def lineBCEq (t : SpecialTriangle) (x y : ℝ) : ℝ := 2*x + y - 8

/-- Theorem: The equation of line BC in the special triangle is 2x + y - 8 = 0 -/
theorem special_triangle_line_BC (t : SpecialTriangle) :
  ∀ x y, lineBCEq t x y = 0 ↔ y = -2*x + 8 :=
sorry

end NUMINAMATH_CALUDE_special_triangle_line_BC_l1504_150477


namespace NUMINAMATH_CALUDE_largest_prime_divisor_13_plus_14_factorial_l1504_150470

theorem largest_prime_divisor_13_plus_14_factorial (p : ℕ) :
  (p.Prime ∧ p ∣ (Nat.factorial 13 + Nat.factorial 14)) →
  p ≤ 13 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_13_plus_14_factorial_l1504_150470


namespace NUMINAMATH_CALUDE_lowest_degree_is_four_l1504_150452

/-- A polynomial with coefficients in ℤ -/
def IntPolynomial := ℕ → ℤ

/-- The degree of an IntPolynomial -/
def degree (p : IntPolynomial) : ℕ := sorry

/-- The set of coefficients of an IntPolynomial -/
def coeffSet (p : IntPolynomial) : Set ℤ := sorry

/-- Predicate for a polynomial satisfying the given conditions -/
def satisfiesCondition (p : IntPolynomial) : Prop :=
  ∃ b : ℤ, (∃ x ∈ coeffSet p, x < b) ∧ 
           (∃ y ∈ coeffSet p, y > b) ∧ 
           b ∉ coeffSet p

/-- The main theorem statement -/
theorem lowest_degree_is_four :
  ∃ p : IntPolynomial, satisfiesCondition p ∧ degree p = 4 ∧
  ∀ q : IntPolynomial, satisfiesCondition q → degree q ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_lowest_degree_is_four_l1504_150452


namespace NUMINAMATH_CALUDE_b_share_is_360_l1504_150471

/-- Represents the rental information for a person --/
structure RentalInfo where
  horses : ℕ
  months : ℕ

/-- Calculates the total horse-months for a rental --/
def horsemonths (r : RentalInfo) : ℕ := r.horses * r.months

theorem b_share_is_360 (total_rent : ℕ) (a b c : RentalInfo) 
  (h1 : total_rent = 870)
  (h2 : a = ⟨12, 8⟩)
  (h3 : b = ⟨16, 9⟩)
  (h4 : c = ⟨18, 6⟩) :
  (horsemonths b * total_rent) / (horsemonths a + horsemonths b + horsemonths c) = 360 := by
  sorry

#eval (16 * 9 * 870) / (12 * 8 + 16 * 9 + 18 * 6)

end NUMINAMATH_CALUDE_b_share_is_360_l1504_150471


namespace NUMINAMATH_CALUDE_E_80_l1504_150439

/-- E(n) represents the number of ways to express n as a product of integers greater than 1, where order matters -/
def E (n : ℕ) : ℕ := sorry

/-- The prime factorization of 80 is 2^4 * 5 -/
axiom prime_factorization_80 : 80 = 2^4 * 5

/-- Theorem: The number of ways to express 80 as a product of integers greater than 1, where order matters, is 42 -/
theorem E_80 : E 80 = 42 := by sorry

end NUMINAMATH_CALUDE_E_80_l1504_150439


namespace NUMINAMATH_CALUDE_prism_volume_l1504_150415

/-- A right rectangular prism with face areas 45, 49, and 56 square units has a volume of 1470 cubic units. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 45) (h2 : b * c = 49) (h3 : a * c = 56) :
  a * b * c = 1470 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1504_150415


namespace NUMINAMATH_CALUDE_a_8_equals_8_l1504_150443

def sequence_property (a : ℕ+ → ℕ) : Prop :=
  ∀ (s t : ℕ+), a (s * t) = a s * a t

theorem a_8_equals_8 (a : ℕ+ → ℕ) (h1 : sequence_property a) (h2 : a 2 = 2) : 
  a 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_a_8_equals_8_l1504_150443


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l1504_150421

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 5, 7}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {2, 4} :=
sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l1504_150421


namespace NUMINAMATH_CALUDE_patricia_hair_donation_l1504_150441

/-- Calculates the amount of hair to donate given the current length, additional growth, and desired final length -/
def hair_to_donate (current_length additional_growth final_length : ℕ) : ℕ :=
  (current_length + additional_growth) - final_length

/-- Proves that Patricia needs to donate 23 inches of hair -/
theorem patricia_hair_donation :
  let current_length : ℕ := 14
  let additional_growth : ℕ := 21
  let final_length : ℕ := 12
  hair_to_donate current_length additional_growth final_length = 23 := by
  sorry

end NUMINAMATH_CALUDE_patricia_hair_donation_l1504_150441


namespace NUMINAMATH_CALUDE_cannot_form_square_l1504_150409

/-- Represents the collection of sticks --/
structure StickCollection where
  twoLengthCount : Nat
  threeLengthCount : Nat
  sevenLengthCount : Nat

/-- Checks if it's possible to form a square with given sticks --/
def canFormSquare (sticks : StickCollection) : Prop :=
  ∃ (side : ℕ), 
    4 * side = 2 * sticks.twoLengthCount + 
               3 * sticks.threeLengthCount + 
               7 * sticks.sevenLengthCount ∧
    ∃ (a b c : ℕ), 
      a + b + c = 4 ∧
      a * 2 + b * 3 + c * 7 = 4 * side ∧
      a ≤ sticks.twoLengthCount ∧
      b ≤ sticks.threeLengthCount ∧
      c ≤ sticks.sevenLengthCount

/-- The given collection of sticks --/
def givenSticks : StickCollection :=
  { twoLengthCount := 5
    threeLengthCount := 5
    sevenLengthCount := 1 }

theorem cannot_form_square : ¬(canFormSquare givenSticks) := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_square_l1504_150409


namespace NUMINAMATH_CALUDE_master_bedroom_size_l1504_150434

theorem master_bedroom_size (total_area guest_area master_area combined_area : ℝ) 
  (h1 : total_area = 2300)
  (h2 : combined_area = 1000)
  (h3 : guest_area = (1/4) * master_area)
  (h4 : total_area = combined_area + guest_area + master_area) :
  master_area = 1040 := by
sorry

end NUMINAMATH_CALUDE_master_bedroom_size_l1504_150434


namespace NUMINAMATH_CALUDE_prime_pythagorean_inequality_l1504_150400

theorem prime_pythagorean_inequality (p m n : ℕ) 
  (hp : Nat.Prime p) 
  (hm : m > 0) 
  (hn : n > 0) 
  (heq : p^2 + m^2 = n^2) : 
  m > p := by
  sorry

end NUMINAMATH_CALUDE_prime_pythagorean_inequality_l1504_150400


namespace NUMINAMATH_CALUDE_complex_magnitude_l1504_150410

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem complex_magnitude (h : z * i^2023 = 1 + i) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1504_150410


namespace NUMINAMATH_CALUDE_max_regions_three_triangles_is_20_l1504_150460

/-- The maximum number of regions formed by three triangles on a plane -/
def max_regions_three_triangles : ℕ := 20

/-- The number of triangles drawn on the plane -/
def num_triangles : ℕ := 3

/-- Theorem stating that the maximum number of regions formed by three triangles is 20 -/
theorem max_regions_three_triangles_is_20 :
  max_regions_three_triangles = 20 ∧ num_triangles = 3 := by sorry

end NUMINAMATH_CALUDE_max_regions_three_triangles_is_20_l1504_150460


namespace NUMINAMATH_CALUDE_S_congruence_l1504_150411

def is_valid_N (N : ℕ) : Prop :=
  300 ≤ N ∧ N ≤ 600

def base_4_repr (N : ℕ) : ℕ × ℕ × ℕ :=
  (N / 16, (N / 4) % 4, N % 4)

def base_7_repr (N : ℕ) : ℕ × ℕ × ℕ :=
  (N / 49, (N / 7) % 7, N % 7)

def S (N : ℕ) : ℕ :=
  let (a₁, a₂, a₃) := base_4_repr N
  let (b₁, b₂, b₃) := base_7_repr N
  16 * a₁ + 4 * a₂ + a₃ + 49 * b₁ + 7 * b₂ + b₃

theorem S_congruence (N : ℕ) (h : is_valid_N N) :
  S N % 100 = (3 * N) % 100 ↔ (base_4_repr N).2.2 + (base_7_repr N).2.2 ≡ 3 * N [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_S_congruence_l1504_150411


namespace NUMINAMATH_CALUDE_total_worth_of_gold_bars_l1504_150491

/-- Calculates the total worth of gold bars in a safe. -/
theorem total_worth_of_gold_bars
  (num_rows : ℕ)
  (bars_per_row : ℕ)
  (worth_per_bar : ℕ)
  (h1 : num_rows = 4)
  (h2 : bars_per_row = 20)
  (h3 : worth_per_bar = 20000) :
  num_rows * bars_per_row * worth_per_bar = 1600000 := by
  sorry

#check total_worth_of_gold_bars

end NUMINAMATH_CALUDE_total_worth_of_gold_bars_l1504_150491


namespace NUMINAMATH_CALUDE_existence_of_solution_l1504_150486

theorem existence_of_solution (p : Nat) (hp : Nat.Prime p) (hodd : Odd p) :
  ∃ (x y z t : Nat), x^2 + y^2 + z^2 = t * p ∧ t < p ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ t ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l1504_150486


namespace NUMINAMATH_CALUDE_log_equation_solution_l1504_150426

/-- Proves that 56 is the solution to the logarithmic equation log_7(x) - 3log_7(2) = 1 -/
theorem log_equation_solution :
  ∃ x : ℝ, (Real.log x / Real.log 7) - 3 * (Real.log 2 / Real.log 7) = 1 ∧ x = 56 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1504_150426


namespace NUMINAMATH_CALUDE_number_difference_l1504_150457

theorem number_difference (x y : ℚ) 
  (sum_eq : x + y = 40)
  (triple_minus_quad : 3 * y - 4 * x = 10) :
  abs (y - x) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1504_150457


namespace NUMINAMATH_CALUDE_number_of_observations_l1504_150495

theorem number_of_observations (original_mean new_mean : ℝ) (correction : ℝ) :
  original_mean = 36 →
  correction = 1 →
  new_mean = 36.02 →
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * new_mean = (n : ℝ) * original_mean + correction :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_observations_l1504_150495


namespace NUMINAMATH_CALUDE_tan_order_l1504_150493

open Real

noncomputable def f (x : ℝ) := tan (x + π/4)

theorem tan_order : f 0 > f (-1) ∧ f (-1) > f 1 := by sorry

end NUMINAMATH_CALUDE_tan_order_l1504_150493


namespace NUMINAMATH_CALUDE_shells_weight_calculation_l1504_150463

/-- Given an initial weight of shells and an additional weight of shells,
    calculate the total weight of shells. -/
def total_weight (initial_weight additional_weight : ℕ) : ℕ :=
  initial_weight + additional_weight

/-- Theorem: The total weight of shells is 17 pounds when
    the initial weight is 5 pounds and the additional weight is 12 pounds. -/
theorem shells_weight_calculation :
  total_weight 5 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_shells_weight_calculation_l1504_150463


namespace NUMINAMATH_CALUDE_roots_of_equation_l1504_150462

theorem roots_of_equation (x : ℝ) : 
  (2 * x^2 - x = 0) ↔ (x = 0 ∨ x = 1/2) := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1504_150462


namespace NUMINAMATH_CALUDE_parents_without_jobs_l1504_150475

/-- The percentage of parents without full-time jobs -/
def percentage_without_jobs (mother_job_rate : ℝ) (father_job_rate : ℝ) (mother_percentage : ℝ) : ℝ :=
  100 - (mother_job_rate * mother_percentage + father_job_rate * (100 - mother_percentage))

theorem parents_without_jobs :
  percentage_without_jobs 90 75 40 = 19 := by
  sorry

end NUMINAMATH_CALUDE_parents_without_jobs_l1504_150475


namespace NUMINAMATH_CALUDE_rancher_cows_count_l1504_150401

theorem rancher_cows_count (horses : ℕ) (cows : ℕ) : 
  cows = 5 * horses →
  cows + horses = 168 →
  cows = 140 := by
sorry

end NUMINAMATH_CALUDE_rancher_cows_count_l1504_150401


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l1504_150407

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem smallest_two_digit_prime_with_composite_reverse :
  ∃ (n : ℕ), is_two_digit n ∧ is_prime n ∧ ¬(is_prime (reverse_digits n)) ∧
  (∀ m : ℕ, is_two_digit m → is_prime m → m < n → is_prime (reverse_digits m)) ∧
  n = 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l1504_150407


namespace NUMINAMATH_CALUDE_pentagon_from_reflections_l1504_150451

/-- Given a set of reflection points, there exists a unique pentagon satisfying the reflection properties. -/
theorem pentagon_from_reflections (B : Fin 5 → ℝ × ℝ) :
  ∃! (A : Fin 5 → ℝ × ℝ), ∀ i : Fin 5, B i = 2 * A (i.succ) - A i :=
by sorry

end NUMINAMATH_CALUDE_pentagon_from_reflections_l1504_150451


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l1504_150454

theorem arctan_sum_special_case : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l1504_150454


namespace NUMINAMATH_CALUDE_positive_integer_solution_exists_l1504_150419

theorem positive_integer_solution_exists : 
  ∃ (x y z t : ℕ+), x + y + z + t = 10 :=
by sorry

#check positive_integer_solution_exists

end NUMINAMATH_CALUDE_positive_integer_solution_exists_l1504_150419


namespace NUMINAMATH_CALUDE_playground_students_l1504_150413

/-- The number of students initially on the playground -/
def initial_students : ℕ := 32

/-- The number of students who left the playground -/
def students_left : ℕ := 16

/-- The number of new students who came to the playground -/
def new_students : ℕ := 9

/-- The final number of students on the playground -/
def final_students : ℕ := 25

theorem playground_students :
  initial_students - students_left + new_students = final_students :=
by sorry

end NUMINAMATH_CALUDE_playground_students_l1504_150413


namespace NUMINAMATH_CALUDE_company_survey_problem_l1504_150430

/-- The number of employees who do not use social networks -/
def non_social_users : ℕ := 40

/-- The proportion of social network users who use VKontakte -/
def vk_users_ratio : ℚ := 3/4

/-- The proportion of social network users who use both VKontakte and Odnoklassniki -/
def both_users_ratio : ℚ := 13/20

/-- The proportion of total employees who use Odnoklassniki -/
def ok_users_ratio : ℚ := 5/6

/-- The total number of employees in the company -/
def total_employees : ℕ := 540

theorem company_survey_problem :
  ∃ (N : ℕ),
    N = total_employees ∧
    (N - non_social_users : ℚ) * (vk_users_ratio + (1 - vk_users_ratio)) = N * ok_users_ratio :=
sorry

end NUMINAMATH_CALUDE_company_survey_problem_l1504_150430


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1764_l1504_150492

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

theorem largest_perfect_square_factor_of_1764 :
  ∀ k : ℕ, is_perfect_square k ∧ k ∣ 1764 → k ≤ 1764 :=
by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1764_l1504_150492


namespace NUMINAMATH_CALUDE_equation_solutions_l1504_150461

theorem equation_solutions :
  (∃ x : ℚ, (1/2) * x - 3 = 2 * x + 1/2 ∧ x = -7/3) ∧
  (∃ x : ℚ, (x-3)/2 - (2*x+1)/3 = 1 ∧ x = -17) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1504_150461


namespace NUMINAMATH_CALUDE_walters_age_2001_l1504_150453

theorem walters_age_2001 (walter_age_1996 : ℕ) (grandmother_age_1996 : ℕ) :
  (grandmother_age_1996 = 3 * walter_age_1996) →
  (1996 - walter_age_1996 + 1996 - grandmother_age_1996 = 3864) →
  (walter_age_1996 + (2001 - 1996) = 37) :=
by sorry

end NUMINAMATH_CALUDE_walters_age_2001_l1504_150453


namespace NUMINAMATH_CALUDE_pizza_burger_overlap_l1504_150437

theorem pizza_burger_overlap (total : ℕ) (pizza : ℕ) (burger : ℕ) 
  (h_total : total = 200)
  (h_pizza : pizza = 125)
  (h_burger : burger = 115) :
  pizza + burger - total = 40 := by
  sorry

end NUMINAMATH_CALUDE_pizza_burger_overlap_l1504_150437


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1504_150433

theorem imaginary_part_of_z (z : ℂ) (h : (1 + z) / Complex.I = 1 - z) : 
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1504_150433


namespace NUMINAMATH_CALUDE_min_sum_of_square_areas_l1504_150427

theorem min_sum_of_square_areas (wire_length : ℝ) (h : wire_length = 16) :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ wire_length ∧
  (x^2 + (wire_length - x)^2 ≥ 8 ∧
   ∀ (y : ℝ), 0 ≤ y ∧ y ≤ wire_length →
     y^2 + (wire_length - y)^2 ≥ x^2 + (wire_length - x)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_square_areas_l1504_150427


namespace NUMINAMATH_CALUDE_beads_per_necklace_is_eight_l1504_150465

/-- The number of beads Emily has -/
def total_beads : ℕ := 16

/-- The number of necklaces Emily can make -/
def num_necklaces : ℕ := 2

/-- The number of beads per necklace -/
def beads_per_necklace : ℕ := total_beads / num_necklaces

theorem beads_per_necklace_is_eight : beads_per_necklace = 8 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_necklace_is_eight_l1504_150465


namespace NUMINAMATH_CALUDE_integer_as_sum_diff_squares_l1504_150497

theorem integer_as_sum_diff_squares (n : ℤ) :
  ∃ (a b c : ℤ), n = a^2 + b^2 - c^2 := by
  sorry

end NUMINAMATH_CALUDE_integer_as_sum_diff_squares_l1504_150497


namespace NUMINAMATH_CALUDE_group_size_l1504_150440

theorem group_size (iceland : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : iceland = 25)
  (h2 : norway = 23)
  (h3 : both = 21)
  (h4 : neither = 23) :
  iceland + norway - both + neither = 50 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l1504_150440


namespace NUMINAMATH_CALUDE_range_of_m_l1504_150405

-- Define the conditions
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, p x → q x m) ∧  -- q is necessary for p
  (∃ x, p x ∧ ¬(q x m)) ∧  -- q is not sufficient for p
  (m > 0) →
  m ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1504_150405


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_a_geq_2_sufficient_not_necessary_l1504_150479

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | -1-a ≤ x ∧ x ≤ 1-a}

-- Theorem 1: If A ∩ B = {x | 1/2 ≤ x < 1}, then a = -3/2
theorem intersection_implies_a_value (a : ℝ) : 
  A ∩ B a = {x | 1/2 ≤ x ∧ x < 1} → a = -3/2 := by sorry

-- Theorem 2: a ≥ 2 is a sufficient but not necessary condition for A ∩ B = ∅
theorem a_geq_2_sufficient_not_necessary (a : ℝ) :
  (a ≥ 2 → A ∩ B a = ∅) ∧ ¬(A ∩ B a = ∅ → a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_a_geq_2_sufficient_not_necessary_l1504_150479


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1504_150458

/-- Proves that given a sum P at simple interest for 10 years, 
    if increasing the interest rate by 5% results in Rs. 400 more interest, 
    then P = 800. -/
theorem simple_interest_problem (P R : ℝ) 
  (h1 : P > 0) 
  (h2 : R > 0) 
  (h3 : (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 400) : 
  P = 800 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1504_150458


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l1504_150478

/-- Given two lines l₁ and l₂ defined by linear equations with parameter m,
    prove that m = -2 is a sufficient but not necessary condition for l₁ // l₂ -/
theorem parallel_lines_condition (m : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | (m - 4) * x - (2 * m + 4) * y + 2 * m - 4 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | (m - 1) * x + (m + 2) * y + 1 = 0}
  (m = -2 → l₁ = l₂) ∧ ¬(l₁ = l₂ → m = -2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l1504_150478


namespace NUMINAMATH_CALUDE_gcd_105_88_l1504_150431

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_88_l1504_150431


namespace NUMINAMATH_CALUDE_factorization_theorem_1_factorization_theorem_2_l1504_150484

-- Define variables
variable (a b x y p : ℝ)

-- Theorem for the first expression
theorem factorization_theorem_1 : 
  8*a*x - b*x + 8*a*y - b*y = (x + y)*(8*a - b) := by sorry

-- Theorem for the second expression
theorem factorization_theorem_2 : 
  a*p + a*x - 2*b*x - 2*b*p = (p + x)*(a - 2*b) := by sorry

end NUMINAMATH_CALUDE_factorization_theorem_1_factorization_theorem_2_l1504_150484


namespace NUMINAMATH_CALUDE_equation_solutions_l1504_150408

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, (x₁ + 5)^2 = 16 ∧ (x₂ + 5)^2 = 16 ∧ x₁ = -9 ∧ x₂ = -1) ∧
  (∃ y₁ y₂ : ℝ, y₁^2 - 4*y₁ - 12 = 0 ∧ y₂^2 - 4*y₂ - 12 = 0 ∧ y₁ = 6 ∧ y₂ = -2) :=
by
  sorry

#check equation_solutions

end NUMINAMATH_CALUDE_equation_solutions_l1504_150408


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1504_150482

-- Define the concept of a line in a plane
def Line : Type := ℝ × ℝ → Prop

-- Define perpendicularity relation between lines
def Perpendicular (l1 l2 : Line) : Prop := sorry

-- Define parallel relation between lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- The main theorem
theorem perpendicular_lines_parallel (a b c : Line) :
  Perpendicular a b → Perpendicular c b → Parallel a c := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1504_150482

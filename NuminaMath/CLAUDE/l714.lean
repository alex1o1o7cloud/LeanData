import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_set_l714_71491

theorem inequality_solution_set (x : ℝ) : 
  (1 + 2 * (x - 1) ≤ 3) ↔ (x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l714_71491


namespace NUMINAMATH_CALUDE_lcm_36_105_l714_71439

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_105_l714_71439


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l714_71436

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  ∃ (min_val : Real), min_val = 3*Real.sqrt 2 + Real.sqrt 3 ∧
  ∀ θ', 0 < θ' ∧ θ' < π/2 →
    3 * Real.sin θ' + 2 / Real.cos θ' + Real.sqrt 3 * (Real.cos θ' / Real.sin θ') ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l714_71436


namespace NUMINAMATH_CALUDE_toddler_count_l714_71403

theorem toddler_count (bill_count : ℕ) (double_counted : ℕ) (missed : ℕ) : 
  bill_count = 26 → double_counted = 8 → missed = 3 → 
  bill_count - double_counted + missed = 21 := by
sorry

end NUMINAMATH_CALUDE_toddler_count_l714_71403


namespace NUMINAMATH_CALUDE_triangle_area_is_twelve_l714_71441

/-- The area of a triangular region bounded by the coordinate axes and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the line bounding the triangular region -/
def lineEquation (x y : ℝ) : Prop := 3 * x + 2 * y = 12

theorem triangle_area_is_twelve :
  ∃ (x_intercept y_intercept : ℝ),
    lineEquation x_intercept 0 ∧
    lineEquation 0 y_intercept ∧
    triangleArea = (1 / 2) * x_intercept * y_intercept :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_twelve_l714_71441


namespace NUMINAMATH_CALUDE_base7_to_base10_76543_l714_71408

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 number 76543 --/
def base7Number : List Nat := [3, 4, 5, 6, 7]

/-- Theorem: The base 10 equivalent of 76543 in base 7 is 19141 --/
theorem base7_to_base10_76543 :
  base7ToBase10 base7Number = 19141 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_76543_l714_71408


namespace NUMINAMATH_CALUDE_circle_reduction_l714_71446

/-- Represents a letter in the circle -/
inductive Letter
| A
| B

/-- Represents the circle of letters -/
def Circle := List Letter

/-- Represents a transformation rule -/
inductive Transform
| ABA_to_B
| B_to_ABA
| VAV_to_A
| A_to_VAV

/-- Applies a single transformation to the circle -/
def applyTransform (c : Circle) (t : Transform) : Circle :=
  sorry

/-- Checks if the circle contains exactly one letter -/
def isSingleLetter (c : Circle) : Bool :=
  sorry

/-- The main theorem to prove -/
theorem circle_reduction (initial : Circle) :
  initial.length = 41 →
  ∃ (final : Circle), (∃ (transforms : List Transform),
    (List.foldl applyTransform initial transforms = final) ∧
    isSingleLetter final) :=
  sorry

end NUMINAMATH_CALUDE_circle_reduction_l714_71446


namespace NUMINAMATH_CALUDE_extended_triangle_area_l714_71402

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the theorem
theorem extended_triangle_area (t : Triangle) :
  ∃ (new_area : ℝ), new_area = 7 * t.area :=
sorry

end NUMINAMATH_CALUDE_extended_triangle_area_l714_71402


namespace NUMINAMATH_CALUDE_percentage_problem_l714_71438

theorem percentage_problem : (45 * 7) / 900 * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l714_71438


namespace NUMINAMATH_CALUDE_two_digit_number_transformation_l714_71474

/-- Given a two-digit integer n = 10a + b, where n = (k+1)(a + b),
    prove that 10(a+1) + (b+1) = ((k+1)(a + b) + 11) / (a + b + 2) * (a + b + 2) -/
theorem two_digit_number_transformation (a b k : ℕ) (h1 : 10*a + b = (k+1)*(a + b)) :
  10*(a+1) + (b+1) = ((k+1)*(a + b) + 11) / (a + b + 2) * (a + b + 2) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_transformation_l714_71474


namespace NUMINAMATH_CALUDE_fifteen_subcommittees_l714_71435

/-- The number of ways to form a two-person sub-committee from a larger committee,
    where one member must be from a designated group. -/
def subcommittee_count (total : ℕ) (designated : ℕ) : ℕ :=
  designated * (total - designated)

/-- Theorem stating that for a committee of 8 people with a designated group of 3,
    there are 15 possible two-person sub-committees. -/
theorem fifteen_subcommittees :
  subcommittee_count 8 3 = 15 := by
  sorry

#eval subcommittee_count 8 3

end NUMINAMATH_CALUDE_fifteen_subcommittees_l714_71435


namespace NUMINAMATH_CALUDE_age_problem_l714_71462

theorem age_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 28 →
  (a + c) / 2 = 29 →
  b = 26 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l714_71462


namespace NUMINAMATH_CALUDE_cone_surface_area_ratio_l714_71440

/-- Given a cone whose lateral surface development forms a sector with a central angle of 120° and radius l, 
    prove that the ratio of its total surface area to its lateral surface area is 4:3. -/
theorem cone_surface_area_ratio (l : ℝ) (h : l > 0) : 
  let r := l / 3
  let lateral_area := π * l * r
  let base_area := π * r^2
  let total_area := lateral_area + base_area
  (total_area / lateral_area : ℝ) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_surface_area_ratio_l714_71440


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_ten_thousand_l714_71471

def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

theorem smallest_n_divisible_by_ten_thousand :
  ∀ n : ℕ, n > 0 → n < 9375 → ¬(10000 ∣ sum_of_naturals n) ∧ (10000 ∣ sum_of_naturals 9375) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_ten_thousand_l714_71471


namespace NUMINAMATH_CALUDE_positions_after_179_moves_l714_71456

/-- Represents the positions of the cat -/
inductive CatPosition
| TopLeft
| TopRight
| BottomRight
| BottomLeft

/-- Represents the positions of the mouse -/
inductive MousePosition
| TopMiddle
| TopRight
| RightMiddle
| BottomRight
| BottomMiddle
| BottomLeft
| LeftMiddle
| TopLeft

/-- Calculates the position of the cat after a given number of moves -/
def catPositionAfterMoves (moves : Nat) : CatPosition :=
  match moves % 4 with
  | 0 => CatPosition.BottomLeft
  | 1 => CatPosition.TopLeft
  | 2 => CatPosition.TopRight
  | _ => CatPosition.BottomRight

/-- Calculates the position of the mouse after a given number of moves -/
def mousePositionAfterMoves (moves : Nat) : MousePosition :=
  match moves % 8 with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.TopMiddle
  | 2 => MousePosition.TopRight
  | 3 => MousePosition.RightMiddle
  | 4 => MousePosition.BottomRight
  | 5 => MousePosition.BottomMiddle
  | 6 => MousePosition.BottomLeft
  | _ => MousePosition.LeftMiddle

theorem positions_after_179_moves :
  (catPositionAfterMoves 179 = CatPosition.BottomRight) ∧
  (mousePositionAfterMoves 179 = MousePosition.RightMiddle) := by
  sorry

end NUMINAMATH_CALUDE_positions_after_179_moves_l714_71456


namespace NUMINAMATH_CALUDE_fraction_equals_875_l714_71481

theorem fraction_equals_875 (a : ℕ+) (h : (a : ℚ) / ((a : ℚ) + 35) = 875 / 1000) : 
  a = 245 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_875_l714_71481


namespace NUMINAMATH_CALUDE_election_winner_votes_l714_71413

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) 
  (h1 : winner_percentage = 62 / 100) 
  (h2 : winner_percentage * total_votes - (1 - winner_percentage) * total_votes = vote_difference) 
  (h3 : vote_difference = 348) : 
  ⌊winner_percentage * total_votes⌋ = 899 :=
sorry

end NUMINAMATH_CALUDE_election_winner_votes_l714_71413


namespace NUMINAMATH_CALUDE_megan_candy_count_l714_71472

theorem megan_candy_count (mary_initial : ℕ) (megan : ℕ) : 
  mary_initial = 3 * megan →
  mary_initial + 10 = 25 →
  megan = 5 := by
sorry

end NUMINAMATH_CALUDE_megan_candy_count_l714_71472


namespace NUMINAMATH_CALUDE_sqrt_equation_condition_l714_71460

theorem sqrt_equation_condition (a b : ℝ) (k : ℕ+) :
  (Real.sqrt (a^2 + (k.val * b)^2) = a + k.val * b) ↔ (a * k.val * b = 0 ∧ a + k.val * b ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_sqrt_equation_condition_l714_71460


namespace NUMINAMATH_CALUDE_divisible_by_55_l714_71425

theorem divisible_by_55 (n : ℤ) : 
  55 ∣ (n^2 + 3*n + 1) ↔ n % 55 = 6 ∨ n % 55 = 46 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_55_l714_71425


namespace NUMINAMATH_CALUDE_smallest_n_with_shared_digit_arrangement_l714_71459

/-- A function that checks if two natural numbers share a digit in their decimal representation -/
def share_digit (a b : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ (d ∈ a.digits 10) ∧ (d ∈ b.digits 10)

/-- A function that checks if a list of natural numbers satisfies the neighboring digit condition -/
def valid_arrangement (lst : List ℕ) : Prop :=
  ∀ i : ℕ, i < lst.length → share_digit (lst.get! i) (lst.get! ((i + 1) % lst.length))

/-- The main theorem stating that 29 is the smallest N satisfying the conditions -/
theorem smallest_n_with_shared_digit_arrangement :
  ∀ N : ℕ, N ≥ 2 →
  (∃ lst : List ℕ, lst.length = N ∧ lst.toFinset = Finset.range N.succ ∧ valid_arrangement lst) →
  N ≥ 29 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_shared_digit_arrangement_l714_71459


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l714_71419

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_sum : a 6 + a 8 = 4) : 
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l714_71419


namespace NUMINAMATH_CALUDE_speed_calculation_l714_71450

/-- Given a speed v and time t, if increasing the speed by 12 miles per hour
    reduces the time by 1/4, then v = 36 miles per hour. -/
theorem speed_calculation (v t : ℝ) (h : v * t = (v + 12) * (3/4 * t)) : v = 36 :=
sorry

end NUMINAMATH_CALUDE_speed_calculation_l714_71450


namespace NUMINAMATH_CALUDE_cubic_and_quadratic_sum_l714_71431

theorem cubic_and_quadratic_sum (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (prod_eq : x * y = 12) : 
  x^3 + y^3 = 224 ∧ x^2 + y^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_cubic_and_quadratic_sum_l714_71431


namespace NUMINAMATH_CALUDE_trajectory_equation_l714_71489

/-- The trajectory of a moving point equidistant from F(2, 0) and the y-axis -/
theorem trajectory_equation (x y : ℝ) :
  (|x| = Real.sqrt ((x - 2)^2 + y^2)) ↔ (y^2 = 4 * (x - 1)) :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l714_71489


namespace NUMINAMATH_CALUDE_rectangle_length_l714_71430

/-- Given a rectangle with perimeter 42 and width 4, its length is 17. -/
theorem rectangle_length (P w l : ℝ) (h1 : P = 42) (h2 : w = 4) (h3 : P = 2 * (l + w)) : l = 17 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l714_71430


namespace NUMINAMATH_CALUDE_asymptotes_necessary_not_sufficient_l714_71490

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form (x²/a² - y²/b² = 1) -/
  equation : ℝ → ℝ → Prop

/-- Represents the asymptotes of a hyperbola -/
structure Asymptotes where
  /-- The equation of the asymptotes in the form y = ±mx -/
  equation : ℝ → ℝ → Prop

/-- The specific hyperbola C with equation x²/9 - y²/16 = 1 -/
def hyperbola_C : Hyperbola :=
  { equation := fun x y => x^2 / 9 - y^2 / 16 = 1 }

/-- The asymptotes with equation y = ±(4/3)x -/
def asymptotes_C : Asymptotes :=
  { equation := fun x y => y = 4/3 * x ∨ y = -4/3 * x }

/-- Theorem stating that the given asymptote equation is a necessary but not sufficient condition for the hyperbola equation -/
theorem asymptotes_necessary_not_sufficient :
  (∀ x y, hyperbola_C.equation x y → asymptotes_C.equation x y) ∧
  ¬(∀ x y, asymptotes_C.equation x y → hyperbola_C.equation x y) := by
  sorry

end NUMINAMATH_CALUDE_asymptotes_necessary_not_sufficient_l714_71490


namespace NUMINAMATH_CALUDE_find_d_l714_71455

theorem find_d : ∃ d : ℚ, 
  (∃ n : ℤ, n = ⌊d⌋ ∧ 3 * n^2 + 19 * n - 28 = 0) ∧
  (4 * (d - ↑⌊d⌋)^2 - 11 * (d - ↑⌊d⌋) + 3 = 0) ∧
  (0 ≤ d - ↑⌊d⌋ ∧ d - ↑⌊d⌋ < 1) ∧
  d = -29/4 := by
  sorry

end NUMINAMATH_CALUDE_find_d_l714_71455


namespace NUMINAMATH_CALUDE_count_paths_l714_71485

/-- The number of paths on a 6x5 grid from A to B with specific conditions -/
def num_paths : ℕ := 252

/-- The width of the grid -/
def grid_width : ℕ := 6

/-- The height of the grid -/
def grid_height : ℕ := 5

/-- The total number of moves required -/
def total_moves : ℕ := 11

/-- Theorem stating the number of paths under given conditions -/
theorem count_paths :
  num_paths = Nat.choose (total_moves - 1) grid_height :=
sorry

end NUMINAMATH_CALUDE_count_paths_l714_71485


namespace NUMINAMATH_CALUDE_no_integer_tangent_length_l714_71412

theorem no_integer_tangent_length (t : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 2 * π * r = 8 * π) →  -- Circle with circumference 8π
  (t^2 = (8*π/3) * π) →                    -- Tangent-secant relationship
  ¬(∃ (n : ℤ), t = n) :=                   -- No integer solution for t
by sorry

end NUMINAMATH_CALUDE_no_integer_tangent_length_l714_71412


namespace NUMINAMATH_CALUDE_part_dimensions_l714_71477

/-- Given a base dimension with upper and lower tolerances, 
    prove the maximum and minimum allowable dimensions. -/
theorem part_dimensions 
  (base : ℝ) 
  (upper_tolerance : ℝ) 
  (lower_tolerance : ℝ) 
  (h_base : base = 7) 
  (h_upper : upper_tolerance = 0.05) 
  (h_lower : lower_tolerance = 0.02) : 
  (base + upper_tolerance = 7.05) ∧ (base - lower_tolerance = 6.98) := by
  sorry

end NUMINAMATH_CALUDE_part_dimensions_l714_71477


namespace NUMINAMATH_CALUDE_next_roll_for_average_three_l714_71479

def rolls : List Nat := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

theorem next_roll_for_average_three :
  let n : Nat := rolls.length
  let sum : Nat := rolls.sum
  let target_average : Rat := 3
  let next_roll : Nat := 2
  (sum + next_roll : Rat) / (n + 1) = target_average := by sorry

end NUMINAMATH_CALUDE_next_roll_for_average_three_l714_71479


namespace NUMINAMATH_CALUDE_range_of_a_l714_71427

-- Define the statements p and q
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → a^x > a^y
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + x + a > 0

-- Define the set of valid a values
def valid_a : Set ℝ := {a | (0 < a ∧ a ≤ 1/4) ∨ (a ≥ 1)}

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ valid_a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l714_71427


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l714_71401

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l714_71401


namespace NUMINAMATH_CALUDE_quadratic_coefficient_for_specific_parabola_l714_71443

/-- A quadratic function with vertex (h, k) passing through point (x₀, y₀) has coefficient a -/
def quadratic_coefficient (h k x₀ y₀ : ℚ) : ℚ :=
  (y₀ - k) / ((x₀ - h)^2)

theorem quadratic_coefficient_for_specific_parabola :
  quadratic_coefficient 2 (-3) 6 (-63) = -15/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_for_specific_parabola_l714_71443


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l714_71453

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l714_71453


namespace NUMINAMATH_CALUDE_no_five_digit_perfect_square_with_all_even_or_odd_digits_l714_71452

theorem no_five_digit_perfect_square_with_all_even_or_odd_digits : 
  ¬ ∃ (n : ℕ), 
    (∃ (k : ℕ), n = k^2) ∧ 
    (10000 ≤ n ∧ n < 100000) ∧
    (∀ (d₁ d₂ : ℕ), d₁ < 5 → d₂ < 5 → d₁ ≠ d₂ → 
      (n / 10^d₁ % 10) ≠ (n / 10^d₂ % 10)) ∧
    ((∀ (d : ℕ), d < 5 → Even (n / 10^d % 10)) ∨ 
     (∀ (d : ℕ), d < 5 → Odd (n / 10^d % 10))) :=
by sorry

end NUMINAMATH_CALUDE_no_five_digit_perfect_square_with_all_even_or_odd_digits_l714_71452


namespace NUMINAMATH_CALUDE_ratio_q_p_l714_71457

/-- The number of cards in the box -/
def total_cards : ℕ := 60

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 15

/-- The number of cards for each number -/
def cards_per_number : ℕ := 4

/-- The number of cards drawn -/
def cards_drawn : ℕ := 4

/-- The probability of drawing four cards with the same number -/
def p' : ℚ := (distinct_numbers * 1) / Nat.choose total_cards cards_drawn

/-- The probability of drawing three cards with one number and one card with a different number -/
def q' : ℚ := (distinct_numbers * (distinct_numbers - 1) * Nat.choose cards_per_number 3 * Nat.choose cards_per_number 1) / Nat.choose total_cards cards_drawn

/-- The main theorem stating the ratio of q' to p' -/
theorem ratio_q_p : q' / p' = 224 := by sorry

end NUMINAMATH_CALUDE_ratio_q_p_l714_71457


namespace NUMINAMATH_CALUDE_x_y_relation_l714_71444

theorem x_y_relation (Q : ℝ) (x y : ℝ) (hx : x = Real.sqrt (Q/2 + Real.sqrt (Q/2)))
  (hy : y = Real.sqrt (Q/2 - Real.sqrt (Q/2))) :
  (x^6 + y^6) / 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_y_relation_l714_71444


namespace NUMINAMATH_CALUDE_gcd_60_75_l714_71493

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_60_75_l714_71493


namespace NUMINAMATH_CALUDE_g_of_negative_three_l714_71426

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x - 2

-- Theorem statement
theorem g_of_negative_three : g (-3) = -17 := by
  sorry

end NUMINAMATH_CALUDE_g_of_negative_three_l714_71426


namespace NUMINAMATH_CALUDE_triangle_problem_l714_71417

open Real

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  cos (A - π/3) = 2 * cos A →
  b = 2 →
  (1/2) * b * c * sin A = 3 * sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c * cos A →
  cos (2*C) = 1 - a^2 / (6 * b^2) →
  (a = 2 * sqrt 7 ∧ (B = π/12 ∨ B = 7*π/12)) := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l714_71417


namespace NUMINAMATH_CALUDE_no_intersection_and_in_circle_l714_71424

theorem no_intersection_and_in_circle : ¬∃ (a b : ℝ), 
  (∃ (n : ℤ), ∃ (m : ℤ), n = m ∧ a * n + b = 3 * m^2 + 15) ∧ 
  (a^2 + b^2 ≤ 144) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_and_in_circle_l714_71424


namespace NUMINAMATH_CALUDE_new_student_weight_l714_71410

theorem new_student_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_decrease : ℝ) :
  initial_count = 4 →
  replaced_weight = 96 →
  avg_decrease = 8 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * avg_decrease + replaced_weight ∧
    new_weight = 160 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l714_71410


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l714_71475

theorem consecutive_even_integers_sum (x : ℕ) (h1 : x > 4) : 
  (x - 4) * (x - 2) * x * (x + 2) = 48 * (4 * x) → 
  (x - 4) + (x - 2) + x + (x + 2) = 28 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l714_71475


namespace NUMINAMATH_CALUDE_euler_line_equation_l714_71467

/-- The Euler line of a triangle ABC with vertices A(2,0), B(0,4), and AC = BC -/
def euler_line (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 2 * p.2 + 3 = 0}

/-- Triangle ABC with given properties -/
structure TriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  A_coord : A = (2, 0)
  B_coord : B = (0, 4)
  isosceles : (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

theorem euler_line_equation (t : TriangleABC) :
  euler_line t.A t.B t.C = {p : ℝ × ℝ | p.1 - 2 * p.2 + 3 = 0} :=
by sorry

end NUMINAMATH_CALUDE_euler_line_equation_l714_71467


namespace NUMINAMATH_CALUDE_purchase_cost_l714_71416

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℕ := 3

/-- The cost of a single soda in dollars -/
def soda_cost : ℕ := 2

/-- The number of sandwiches to be purchased -/
def num_sandwiches : ℕ := 5

/-- The number of sodas to be purchased -/
def num_sodas : ℕ := 8

/-- The total cost of the purchase in dollars -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem purchase_cost : total_cost = 31 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l714_71416


namespace NUMINAMATH_CALUDE_unknown_number_proof_l714_71469

theorem unknown_number_proof (x : ℝ) : x + 5 * 12 / (180 / 3) = 41 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l714_71469


namespace NUMINAMATH_CALUDE_popped_kernels_in_final_bag_l714_71447

theorem popped_kernels_in_final_bag 
  (bag1_popped bag1_total bag2_popped bag2_total bag3_total : ℕ)
  (average_percent : ℚ)
  (h1 : bag1_popped = 60)
  (h2 : bag1_total = 75)
  (h3 : bag2_popped = 42)
  (h4 : bag2_total = 50)
  (h5 : bag3_total = 100)
  (h6 : average_percent = 82/100)
  (h7 : (bag1_popped : ℚ) / bag1_total + (bag2_popped : ℚ) / bag2_total + 
        (bag3_popped : ℚ) / bag3_total = 3 * average_percent) :
  bag3_popped = 82 := by
  sorry

#check popped_kernels_in_final_bag

end NUMINAMATH_CALUDE_popped_kernels_in_final_bag_l714_71447


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l714_71492

theorem green_shirt_pairs (total_students : ℕ) (blue_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (blue_pairs : ℕ) :
  total_students = 132 →
  blue_students = 65 →
  green_students = 67 →
  total_pairs = 66 →
  blue_pairs = 29 →
  blue_students + green_students = total_students →
  ∃ (green_pairs : ℕ), green_pairs = 30 ∧ 
    blue_pairs + green_pairs + (total_students - 2 * (blue_pairs + green_pairs)) / 2 = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l714_71492


namespace NUMINAMATH_CALUDE_beetle_average_speed_l714_71411

/-- Represents the terrain types --/
inductive Terrain
  | Flat
  | Sandy
  | Gravel

/-- Represents an insect (ant or beetle) --/
structure Insect where
  flatSpeed : ℝ  -- Speed on flat terrain in meters per minute
  sandySpeedFactor : ℝ  -- Factor to multiply flat speed for sandy terrain
  gravelSpeedFactor : ℝ  -- Factor to multiply flat speed for gravel terrain

/-- Calculates the distance traveled by an insect on a given terrain for a given time --/
def distanceTraveled (insect : Insect) (terrain : Terrain) (time : ℝ) : ℝ :=
  match terrain with
  | Terrain.Flat => insect.flatSpeed * time
  | Terrain.Sandy => insect.flatSpeed * insect.sandySpeedFactor * time
  | Terrain.Gravel => insect.flatSpeed * insect.gravelSpeedFactor * time

/-- The main theorem to prove --/
theorem beetle_average_speed :
  let ant : Insect := {
    flatSpeed := 50,  -- 600 meters / 12 minutes
    sandySpeedFactor := 0.9,  -- 10% decrease
    gravelSpeedFactor := 0.8  -- 20% decrease
  }
  let beetle : Insect := {
    flatSpeed := ant.flatSpeed * 0.85,  -- 15% less than ant
    sandySpeedFactor := 0.95,  -- 5% decrease
    gravelSpeedFactor := 0.75  -- 25% decrease
  }
  let totalDistance := 
    distanceTraveled beetle Terrain.Flat 4 +
    distanceTraveled beetle Terrain.Sandy 3 +
    distanceTraveled beetle Terrain.Gravel 5
  let totalTime := 12
  let averageSpeed := totalDistance / totalTime
  averageSpeed * (60 / 1000) = 2.2525 := by
  sorry

end NUMINAMATH_CALUDE_beetle_average_speed_l714_71411


namespace NUMINAMATH_CALUDE_doubled_to_original_ratio_l714_71458

theorem doubled_to_original_ratio (x : ℝ) (h : 3 * (2 * x + 9) = 51) : 
  (2 * x) / x = 2 :=
by sorry

end NUMINAMATH_CALUDE_doubled_to_original_ratio_l714_71458


namespace NUMINAMATH_CALUDE_triangle_inequality_l714_71461

theorem triangle_inequality (a b c : ℝ) (S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0) 
  (h_S : S = Real.sqrt (((a + b + c) / 2) * (((a + b + c) / 2) - a) * 
    (((a + b + c) / 2) - b) * (((a + b + c) / 2) - c))) :
  a^2 + b^2 + c^2 - (a-b)^2 - (b-c)^2 - (c-a)^2 ≥ 4 * Real.sqrt 3 * S := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l714_71461


namespace NUMINAMATH_CALUDE_lagrange_interpolation_uniqueness_existence_l714_71448

theorem lagrange_interpolation_uniqueness_existence
  (n : ℕ) 
  (x : Fin (n + 1) → ℝ) 
  (a : Fin (n + 1) → ℝ) 
  (h_distinct : ∀ (i j : Fin (n + 1)), i ≠ j → x i ≠ x j) :
  ∃! P : Polynomial ℝ, 
    (Polynomial.degree P ≤ n) ∧ 
    (∀ i : Fin (n + 1), P.eval (x i) = a i) :=
sorry

end NUMINAMATH_CALUDE_lagrange_interpolation_uniqueness_existence_l714_71448


namespace NUMINAMATH_CALUDE_maintain_ratio_theorem_l714_71406

/-- Represents the ingredients in a cake recipe -/
structure Recipe where
  flour : Float
  sugar : Float
  oil : Float

/-- Calculates the new amounts of ingredients while maintaining the ratio -/
def calculate_new_amounts (original : Recipe) (new_flour : Float) : Recipe :=
  let scale_factor := new_flour / original.flour
  { flour := new_flour,
    sugar := original.sugar * scale_factor,
    oil := original.oil * scale_factor }

/-- Rounds a float to two decimal places -/
def round_to_two_decimals (x : Float) : Float :=
  (x * 100).round / 100

theorem maintain_ratio_theorem (original : Recipe) (extra_flour : Float) :
  let new_recipe := calculate_new_amounts original (original.flour + extra_flour)
  round_to_two_decimals new_recipe.sugar = 3.86 ∧
  round_to_two_decimals new_recipe.oil = 2.57 :=
by sorry

end NUMINAMATH_CALUDE_maintain_ratio_theorem_l714_71406


namespace NUMINAMATH_CALUDE_seashell_count_l714_71433

theorem seashell_count (sam_shells mary_shells : ℕ) 
  (h1 : sam_shells = 18) (h2 : mary_shells = 47) : 
  sam_shells + mary_shells = 65 := by
sorry

end NUMINAMATH_CALUDE_seashell_count_l714_71433


namespace NUMINAMATH_CALUDE_stationery_store_problem_l714_71421

theorem stationery_store_problem (total_cost selling_price_A selling_price_B cost_price_A cost_price_B total_profit : ℚ)
  (profit_second_purchase : ℚ) :
  total_cost = 1200 ∧
  selling_price_A = 15 ∧
  selling_price_B = 12 ∧
  cost_price_A = 12 ∧
  cost_price_B = 10 ∧
  total_profit = 270 ∧
  profit_second_purchase = 340 →
  ∃ (num_A num_B : ℕ) (min_price_A : ℚ),
    num_A = 50 ∧
    num_B = 60 ∧
    min_price_A = 14 ∧
    num_A * cost_price_A + num_B * cost_price_B = total_cost ∧
    num_A * (selling_price_A - cost_price_A) + num_B * (selling_price_B - cost_price_B) = total_profit ∧
    num_A * (min_price_A - cost_price_A) + 2 * num_B * (selling_price_B - cost_price_B) ≥ profit_second_purchase :=
by sorry

end NUMINAMATH_CALUDE_stationery_store_problem_l714_71421


namespace NUMINAMATH_CALUDE_integer_roots_iff_m_in_M_l714_71404

/-- The set of values for m where the equation has only integer roots -/
def M : Set ℝ := {3, 7, 15, 6, 9}

/-- The quadratic equation in x parameterized by m -/
def equation (m : ℝ) (x : ℝ) : ℝ :=
  (m - 6) * (m - 9) * x^2 + (15 * m - 117) * x + 54

/-- A predicate to check if a real number is an integer -/
def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- The main theorem stating that the equation has only integer roots iff m ∈ M -/
theorem integer_roots_iff_m_in_M (m : ℝ) : 
  (∀ x : ℝ, equation m x = 0 → is_integer x) ↔ m ∈ M := by sorry

end NUMINAMATH_CALUDE_integer_roots_iff_m_in_M_l714_71404


namespace NUMINAMATH_CALUDE_plane_angle_in_right_triangle_l714_71432

/-- Given a right triangle and a plane through its hypotenuse, 
    this theorem relates the angles the plane makes with the triangle and its legs. -/
theorem plane_angle_in_right_triangle 
  (α β : Real) 
  (h_α : 0 < α ∧ α < π / 2) 
  (h_β : 0 < β ∧ β < π / 2) : 
  ∃ γ, γ = Real.arcsin (Real.sqrt (Real.sin (α + β) * Real.sin (α - β))) ∧ 
           0 ≤ γ ∧ γ ≤ π / 2 := by
  sorry


end NUMINAMATH_CALUDE_plane_angle_in_right_triangle_l714_71432


namespace NUMINAMATH_CALUDE_andy_remaining_demerits_l714_71463

/-- The number of additional demerits Andy can get before being fired -/
def additional_demerits (max_demerits : ℕ) (lateness_instances : ℕ) (demerits_per_lateness : ℕ) (joke_demerits : ℕ) : ℕ :=
  max_demerits - (lateness_instances * demerits_per_lateness + joke_demerits)

/-- Theorem stating that Andy can get 23 more demerits before being fired -/
theorem andy_remaining_demerits :
  additional_demerits 50 6 2 15 = 23 := by
  sorry

end NUMINAMATH_CALUDE_andy_remaining_demerits_l714_71463


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l714_71484

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l714_71484


namespace NUMINAMATH_CALUDE_sum_of_triangles_l714_71488

/-- The triangle operation defined as a × b - c -/
def triangle (a b c : ℝ) : ℝ := a * b - c

/-- Theorem stating that the sum of two specific triangle operations equals -2 -/
theorem sum_of_triangles : triangle 2 3 5 + triangle 1 4 7 = -2 := by sorry

end NUMINAMATH_CALUDE_sum_of_triangles_l714_71488


namespace NUMINAMATH_CALUDE_mod_difference_equals_negative_four_l714_71465

-- Define the % operation
def mod (x y : ℤ) : ℤ := x * y - 3 * x - y

-- State the theorem
theorem mod_difference_equals_negative_four : 
  (mod 6 4) - (mod 4 6) = -4 := by
  sorry

end NUMINAMATH_CALUDE_mod_difference_equals_negative_four_l714_71465


namespace NUMINAMATH_CALUDE_school_club_members_l714_71451

theorem school_club_members :
  ∃! n : ℕ, 0 < n ∧ n < 50 ∧ n % 6 = 4 ∧ n % 5 = 2 ∧ n = 28 := by
  sorry

end NUMINAMATH_CALUDE_school_club_members_l714_71451


namespace NUMINAMATH_CALUDE_same_color_probability_l714_71423

/-- The probability of drawing two balls of the same color from a bag containing
    8 blue balls and 7 yellow balls, with replacement. -/
theorem same_color_probability (blue_balls yellow_balls : ℕ) 
    (h_blue : blue_balls = 8) (h_yellow : yellow_balls = 7) :
    let total_balls := blue_balls + yellow_balls
    let p_blue := blue_balls / total_balls
    let p_yellow := yellow_balls / total_balls
    p_blue ^ 2 + p_yellow ^ 2 = 113 / 225 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l714_71423


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l714_71400

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 3 * n - 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℚ := 2 * (4 ^ (n - 1))

-- Define S_n (sum of first n terms of a_n)
def S (n : ℕ) : ℚ := (3 / 2) * n^2 + (1 / 2) * n

-- Define T_n (sum of first n terms of b_n)
def T (n : ℕ) : ℚ := (2 / 3) * (4^n - 1)

theorem arithmetic_geometric_sequence_properties :
  (∀ n : ℕ, n ≥ 1 → S n = (3 / 2) * n^2 + (1 / 2) * n) ∧
  (b 1 = a 1) ∧
  (b 2 = a 3) →
  (∀ n : ℕ, n ≥ 1 → a n = 3 * n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → T n = (2 / 3) * (4^n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l714_71400


namespace NUMINAMATH_CALUDE_compare_star_operations_l714_71418

-- Define the new operation
def star (a b : ℤ) : ℚ := (a * b : ℚ) - (a : ℚ) / (b : ℚ)

-- Theorem statement
theorem compare_star_operations : star 6 (-3) < star 4 (-4) := by
  sorry

end NUMINAMATH_CALUDE_compare_star_operations_l714_71418


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l714_71449

theorem sandy_correct_sums (total_sums : ℕ) (correct_marks : ℕ) (incorrect_marks : ℕ) (total_marks : ℤ) :
  total_sums = 30 →
  correct_marks = 3 →
  incorrect_marks = 2 →
  total_marks = 45 →
  ∃ (correct_sums : ℕ),
    correct_sums * correct_marks - (total_sums - correct_sums) * incorrect_marks = total_marks ∧
    correct_sums = 21 :=
by sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l714_71449


namespace NUMINAMATH_CALUDE_complement_union_equals_set_l714_71409

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {3, 4, 5}
def B : Set Nat := {1, 3, 6}

theorem complement_union_equals_set : 
  (U \ (A ∪ B)) = {2, 7} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_set_l714_71409


namespace NUMINAMATH_CALUDE_bet_winnings_l714_71442

theorem bet_winnings (initial_amount : ℚ) : 
  initial_amount > 0 →
  initial_amount + 2 * initial_amount = 1200 →
  initial_amount = 400 := by
sorry

end NUMINAMATH_CALUDE_bet_winnings_l714_71442


namespace NUMINAMATH_CALUDE_fraction_equality_existence_l714_71405

theorem fraction_equality_existence :
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / c + (1 : ℚ) / d) ∧
  (∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / c + (1 : ℚ) / d + (1 : ℚ) / e) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_existence_l714_71405


namespace NUMINAMATH_CALUDE_shopping_mall_entrances_exits_l714_71480

theorem shopping_mall_entrances_exits (n : ℕ) (h : n = 4) :
  (n * (n - 1) : ℕ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_shopping_mall_entrances_exits_l714_71480


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l714_71445

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l714_71445


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l714_71428

/-- Given two similar triangles ABC and DEF, prove that DF = 6 -/
theorem similar_triangles_side_length 
  (A B C D E F : ℝ × ℝ) -- Points in 2D space
  (AB BC AC : ℝ) -- Sides of triangle ABC
  (DE EF : ℝ) -- Known sides of triangle DEF
  (angle_BAC angle_EDF : ℝ) -- Angles in radians
  (h_AB : dist A B = 8)
  (h_BC : dist B C = 18)
  (h_AC : dist A C = 12)
  (h_DE : dist D E = 4)
  (h_EF : dist E F = 9)
  (h_angle_BAC : angle_BAC = 2 * π / 3) -- 120° in radians
  (h_angle_EDF : angle_EDF = 2 * π / 3) -- 120° in radians
  : dist D F = 6 := by
  sorry


end NUMINAMATH_CALUDE_similar_triangles_side_length_l714_71428


namespace NUMINAMATH_CALUDE_work_completion_theorem_l714_71487

/-- Calculates the number of men needed to complete a job in a given number of days,
    given the initial number of men and days required. -/
def men_needed (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) : ℕ :=
  (initial_men * initial_days) / new_days

theorem work_completion_theorem (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) :
  initial_men = 25 → initial_days = 96 → new_days = 60 →
  men_needed initial_men initial_days new_days = 40 := by
  sorry

#eval men_needed 25 96 60

end NUMINAMATH_CALUDE_work_completion_theorem_l714_71487


namespace NUMINAMATH_CALUDE_points_earned_is_75_l714_71486

/-- Represents the point system and enemy counts in the video game level -/
structure GameLevel where
  goblin_points : ℕ
  orc_points : ℕ
  dragon_points : ℕ
  goblins_defeated : ℕ
  orcs_defeated : ℕ
  dragons_defeated : ℕ

/-- Calculates the total points earned in a game level -/
def total_points (level : GameLevel) : ℕ :=
  level.goblin_points * level.goblins_defeated +
  level.orc_points * level.orcs_defeated +
  level.dragon_points * level.dragons_defeated

/-- Theorem stating that the total points earned in the given scenario is 75 -/
theorem points_earned_is_75 (level : GameLevel) 
  (h1 : level.goblin_points = 3)
  (h2 : level.orc_points = 5)
  (h3 : level.dragon_points = 10)
  (h4 : level.goblins_defeated = 10)
  (h5 : level.orcs_defeated = 7)
  (h6 : level.dragons_defeated = 1) :
  total_points level = 75 := by
  sorry


end NUMINAMATH_CALUDE_points_earned_is_75_l714_71486


namespace NUMINAMATH_CALUDE_elizabeth_position_l714_71466

theorem elizabeth_position (total_distance : ℝ) (total_steps : ℕ) (steps_taken : ℕ) : 
  total_distance = 24 → 
  total_steps = 6 → 
  steps_taken = 4 → 
  (total_distance / total_steps) * steps_taken = 16 := by
sorry

end NUMINAMATH_CALUDE_elizabeth_position_l714_71466


namespace NUMINAMATH_CALUDE_overlapping_semicircles_area_l714_71468

/-- Given a pattern of overlapping semicircles, this theorem calculates the shaded area. -/
theorem overlapping_semicircles_area (diameter : ℝ) (overlap : ℝ) (total_length : ℝ) : 
  diameter = 3 ∧ overlap = 0.5 ∧ total_length = 12 →
  (∃ (shaded_area : ℝ), shaded_area = 5.625 * Real.pi) := by
  sorry

#check overlapping_semicircles_area

end NUMINAMATH_CALUDE_overlapping_semicircles_area_l714_71468


namespace NUMINAMATH_CALUDE_intersection_set_equality_l714_71437

theorem intersection_set_equality : 
  let S := {α : ℝ | ∃ k : ℤ, α = k * π / 2 - π / 5} ∩ {α : ℝ | -π < α ∧ α < π}
  S = {-π/5, -7*π/10, 3*π/10, 4*π/5} := by sorry

end NUMINAMATH_CALUDE_intersection_set_equality_l714_71437


namespace NUMINAMATH_CALUDE_variance_of_defective_parts_l714_71495

def defective_parts : List ℕ := [3, 3, 0, 2, 3, 0, 3]

def mean (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

def variance (list : List ℕ) : ℚ :=
  let μ := mean list
  (list.map (fun x => ((x : ℚ) - μ) ^ 2)).sum / list.length

theorem variance_of_defective_parts :
  variance defective_parts = 12 / 7 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_defective_parts_l714_71495


namespace NUMINAMATH_CALUDE_parabola_vertex_l714_71464

/-- The vertex of a parabola given by y^2 - 4y + 2x + 7 = 0 is (-3/2, 2) -/
theorem parabola_vertex :
  let f : ℝ → ℝ → ℝ := λ x y => y^2 - 4*y + 2*x + 7
  ∃! (vx vy : ℝ), (∀ x y, f x y = 0 → (x - vx)^2 ≤ (x + 3/2)^2 ∧ y = vy) ∧ vx = -3/2 ∧ vy = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l714_71464


namespace NUMINAMATH_CALUDE_sum_of_integers_l714_71494

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + r = 7)
  (eq2 : q - r + s = 8)
  (eq3 : r - s + p = 4)
  (eq4 : s - p + q = 3) :
  p + q + r + s = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l714_71494


namespace NUMINAMATH_CALUDE_traffic_sampling_is_systematic_l714_71407

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Quota

/-- Represents the characteristics of the sampling process --/
structure SamplingProcess where
  interval : ℕ  -- Time interval between samples
  continuous_stream : Bool  -- Whether there's a continuous stream of units to sample

/-- Determines if a sampling process is systematic --/
def is_systematic (process : SamplingProcess) : Prop :=
  process.interval > 0 ∧ process.continuous_stream

/-- The traffic police sampling process --/
def traffic_sampling : SamplingProcess :=
  { interval := 3,  -- 3 minutes interval
    continuous_stream := true }  -- Continuous stream of passing cars

/-- Theorem stating that the traffic sampling method is systematic --/
theorem traffic_sampling_is_systematic :
  is_systematic traffic_sampling ↔ SamplingMethod.Systematic = 
    (match traffic_sampling with
     | { interval := 3, continuous_stream := true } => SamplingMethod.Systematic
     | _ => SamplingMethod.SimpleRandom) :=
sorry

end NUMINAMATH_CALUDE_traffic_sampling_is_systematic_l714_71407


namespace NUMINAMATH_CALUDE_distance_after_three_minutes_l714_71414

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

/-- Theorem: The distance between two vehicles moving at 65 km/h and 85 km/h after 3 minutes is 1 km -/
theorem distance_after_three_minutes :
  let v1 : ℝ := 65  -- Speed of the truck in km/h
  let v2 : ℝ := 85  -- Speed of the car in km/h
  let t : ℝ := 3 / 60  -- 3 minutes converted to hours
  distance_between_vehicles v1 v2 t = 1 := by
  sorry


end NUMINAMATH_CALUDE_distance_after_three_minutes_l714_71414


namespace NUMINAMATH_CALUDE_rectangle_rotation_volume_l714_71473

/-- The volume of a solid formed by rotating a rectangle around one of its sides -/
theorem rectangle_rotation_volume (length width : ℝ) (h_length : length = 6) (h_width : width = 4) :
  ∃ (volume : ℝ), (volume = 96 * Real.pi ∨ volume = 144 * Real.pi) ∧
  (∃ (axis : ℝ), (axis = length ∨ axis = width) ∧
    volume = Real.pi * (axis / 2) ^ 2 * (if axis = length then width else length)) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_rotation_volume_l714_71473


namespace NUMINAMATH_CALUDE_more_freshmen_than_sophomores_l714_71429

theorem more_freshmen_than_sophomores 
  (total : ℕ) 
  (junior_percent : ℚ) 
  (not_sophomore_percent : ℚ) 
  (seniors : ℕ) 
  (h1 : total = 800)
  (h2 : junior_percent = 22/100)
  (h3 : not_sophomore_percent = 74/100)
  (h4 : seniors = 160)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_more_freshmen_than_sophomores_l714_71429


namespace NUMINAMATH_CALUDE_ab_is_zero_l714_71482

theorem ab_is_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_is_zero_l714_71482


namespace NUMINAMATH_CALUDE_product_of_differences_of_squares_l714_71422

theorem product_of_differences_of_squares : 
  let P := Real.sqrt 2023 + Real.sqrt 2022
  let Q := Real.sqrt 2023 - Real.sqrt 2022
  let R := Real.sqrt 2023 + Real.sqrt 2024
  let S := Real.sqrt 2023 - Real.sqrt 2024
  (P * Q) * (R * S) = -1 := by sorry

end NUMINAMATH_CALUDE_product_of_differences_of_squares_l714_71422


namespace NUMINAMATH_CALUDE_smallest_angle_satisfying_equation_l714_71478

theorem smallest_angle_satisfying_equation :
  let f : ℝ → ℝ := λ x => 9 * Real.sin x * Real.cos x ^ 4 - 9 * Real.sin x ^ 4 * Real.cos x
  ∃ x : ℝ, x > 0 ∧ f x = 1/2 ∧ ∀ y : ℝ, y > 0 → f y = 1/2 → x ≤ y ∧ x = π/6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_satisfying_equation_l714_71478


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l714_71483

theorem sum_of_reciprocal_equations (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 4)
  (h2 : x⁻¹ - y⁻¹ = -6) : 
  x + y = -4/5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l714_71483


namespace NUMINAMATH_CALUDE_possible_values_of_a_l714_71415

theorem possible_values_of_a (A B : Set ℝ) (a : ℝ) :
  A = {1/2, 3} →
  B = {x | 2 * x = a} →
  B ⊆ A →
  {a | ∃ x ∈ B, 2 * x = a} = {1, 6} := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l714_71415


namespace NUMINAMATH_CALUDE_exists_counterexample_to_inequality_l714_71434

theorem exists_counterexample_to_inequality :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^3 + b^3 < 2*a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_counterexample_to_inequality_l714_71434


namespace NUMINAMATH_CALUDE_min_value_of_f_plus_f_prime_l714_71420

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

-- Theorem statement
theorem min_value_of_f_plus_f_prime (a : ℝ) :
  (∃ x, f_derivative a x = 0 ∧ x = 2) →
  (∀ m n, m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 →
    f a m + f_derivative a n ≥ -13) ∧
  (∃ m n, m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a m + f_derivative a n = -13) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_plus_f_prime_l714_71420


namespace NUMINAMATH_CALUDE_inner_square_probability_10x10_l714_71454

/-- Represents a square checkerboard -/
structure Checkerboard where
  size : ℕ
  total_squares : ℕ
  perimeter_squares : ℕ
  inner_squares : ℕ

/-- Creates a 10x10 checkerboard -/
def create_10x10_board : Checkerboard :=
  { size := 10,
    total_squares := 100,
    perimeter_squares := 36,
    inner_squares := 64 }

/-- Calculates the probability of choosing an inner square -/
def inner_square_probability (board : Checkerboard) : ℚ :=
  board.inner_squares / board.total_squares

theorem inner_square_probability_10x10 :
  inner_square_probability create_10x10_board = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inner_square_probability_10x10_l714_71454


namespace NUMINAMATH_CALUDE_parallel_line_slope_l714_71498

/-- Given a line with equation 3x - 6y = 12, prove that the slope of any parallel line is 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 12) → 
  ∃ m : ℝ, m = (1 : ℝ) / 2 ∧ ∀ (x₁ y₁ : ℝ), (3 * x₁ - 6 * y₁ = 12) → 
    ∃ b : ℝ, y₁ = m * x₁ + b :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l714_71498


namespace NUMINAMATH_CALUDE_point_transformation_l714_71497

/-- Rotate a point (x, y) counterclockwise by 90° around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

/-- Reflect a point (x, y) about the line y = x -/
def reflectYeqX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate90 a b 2 3
  let reflected := reflectYeqX rotated.1 rotated.2
  reflected = (-4, 2) → b - a = -6 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l714_71497


namespace NUMINAMATH_CALUDE_intersection_centroids_exist_l714_71496

/-- Represents a point on the grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a line on the grid -/
inductive GridLine
  | Horizontal (y : Int)
  | Vertical (x : Int)

/-- The grid size -/
def gridSize : Nat := 4030

/-- The number of selected lines in each direction -/
def selectedLines : Nat := 2017

/-- Checks if a point is within the grid bounds -/
def isWithinGrid (p : GridPoint) : Prop :=
  -gridSize / 2 ≤ p.x ∧ p.x ≤ gridSize / 2 ∧
  -gridSize / 2 ≤ p.y ∧ p.y ≤ gridSize / 2

/-- Checks if a point is an intersection of selected lines -/
def isIntersection (p : GridPoint) (horizontalLines : List Int) (verticalLines : List Int) : Prop :=
  p.y ∈ horizontalLines ∧ p.x ∈ verticalLines

/-- Calculates the centroid of a triangle -/
def centroid (a b c : GridPoint) : GridPoint :=
  { x := (a.x + b.x + c.x) / 3
  , y := (a.y + b.y + c.y) / 3 }

/-- The main theorem -/
theorem intersection_centroids_exist 
  (horizontalLines : List Int) 
  (verticalLines : List Int) 
  (h1 : horizontalLines.length = selectedLines)
  (h2 : verticalLines.length = selectedLines)
  (h3 : ∀ y ∈ horizontalLines, -gridSize / 2 ≤ y ∧ y ≤ gridSize / 2)
  (h4 : ∀ x ∈ verticalLines, -gridSize / 2 ≤ x ∧ x ≤ gridSize / 2) :
  ∃ (a b c d e f : GridPoint),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    isWithinGrid a ∧ isWithinGrid b ∧ isWithinGrid c ∧
    isWithinGrid d ∧ isWithinGrid e ∧ isWithinGrid f ∧
    isIntersection a horizontalLines verticalLines ∧
    isIntersection b horizontalLines verticalLines ∧
    isIntersection c horizontalLines verticalLines ∧
    isIntersection d horizontalLines verticalLines ∧
    isIntersection e horizontalLines verticalLines ∧
    isIntersection f horizontalLines verticalLines ∧
    centroid a b c = { x := 0, y := 0 } ∧
    centroid d e f = { x := 0, y := 0 } :=
  by sorry


end NUMINAMATH_CALUDE_intersection_centroids_exist_l714_71496


namespace NUMINAMATH_CALUDE_fraction_equality_implies_value_l714_71476

theorem fraction_equality_implies_value (a : ℝ) (x : ℝ) :
  (a - 2) / x = 1 / (2 * a + 7) → x = 2 * a^2 + 3 * a - 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_value_l714_71476


namespace NUMINAMATH_CALUDE_h_over_g_equals_64_l714_71470

theorem h_over_g_equals_64 (G H : ℤ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 5 →
    G / (x + 3) + H / (x^2 - 5*x) = (x^2 - 3*x + 8) / (x^3 + x^2 - 15*x)) →
  (H : ℚ) / (G : ℚ) = 64 := by
sorry

end NUMINAMATH_CALUDE_h_over_g_equals_64_l714_71470


namespace NUMINAMATH_CALUDE_tangent_point_value_l714_71499

/-- The value of 'a' for which the line y = x + 1 is tangent to the curve y = ln(x + a) --/
def tangent_point (a : ℝ) : Prop :=
  ∃ x : ℝ, 
    -- The y-coordinate of the line and curve are equal at the point of tangency
    x + 1 = Real.log (x + a) ∧ 
    -- The slope of the line (which is 1) equals the derivative of ln(x + a) at the point of tangency
    1 = 1 / (x + a)

/-- Theorem stating that 'a' must equal 2 for the tangency condition to be satisfied --/
theorem tangent_point_value : 
  ∃ a : ℝ, tangent_point a ∧ a = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_point_value_l714_71499

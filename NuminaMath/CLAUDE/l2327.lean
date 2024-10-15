import Mathlib

namespace NUMINAMATH_CALUDE_cost_dozen_pens_l2327_232725

-- Define the cost of one pen and one pencil
def cost_pen : ℝ := sorry
def cost_pencil : ℝ := sorry

-- Define the conditions
axiom total_cost : 3 * cost_pen + 5 * cost_pencil = 100
axiom cost_ratio : cost_pen = 5 * cost_pencil

-- Theorem to prove
theorem cost_dozen_pens : 12 * cost_pen = 300 := by
  sorry

end NUMINAMATH_CALUDE_cost_dozen_pens_l2327_232725


namespace NUMINAMATH_CALUDE_two_students_choose_A_l2327_232732

/-- The number of ways to choose exactly two students from four to take course A -/
def waysToChooseTwoForA : ℕ := 24

/-- The number of students -/
def numStudents : ℕ := 4

/-- The number of courses -/
def numCourses : ℕ := 3

theorem two_students_choose_A :
  waysToChooseTwoForA = (numStudents.choose 2) * (2^(numStudents - 2)) :=
sorry

end NUMINAMATH_CALUDE_two_students_choose_A_l2327_232732


namespace NUMINAMATH_CALUDE_average_equality_l2327_232771

theorem average_equality (n : ℕ) (scores : Fin n → ℝ) :
  let original_avg : ℝ := (Finset.sum Finset.univ (λ i => scores i)) / n
  let new_sum : ℝ := (Finset.sum Finset.univ (λ i => scores i)) + 2 * original_avg
  new_sum / (n + 2) = original_avg := by
  sorry

end NUMINAMATH_CALUDE_average_equality_l2327_232771


namespace NUMINAMATH_CALUDE_tangent_point_for_equal_volume_l2327_232712

theorem tangent_point_for_equal_volume (ξ η : ℝ) : 
  ξ^2 + η^2 = 1 →  -- Point (ξ, η) is on the unit circle
  0 < ξ →          -- ξ is positive (first quadrant)
  ξ < 1 →          -- ξ is less than 1 (valid tangent)
  (((1 - ξ^2)^2 / (3 * ξ)) - ((1 - ξ)^2 * (2 + ξ) / 3)) * π = 4 * π / 3 →  -- Volume equation
  ξ = 3 - 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_for_equal_volume_l2327_232712


namespace NUMINAMATH_CALUDE_prob_odd_divisor_21_factorial_l2327_232727

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def primeFactorization (n : ℕ) : List (ℕ × ℕ) := sorry

def numDivisors (n : ℕ) : ℕ := sorry

def numOddDivisors (n : ℕ) : ℕ := sorry

theorem prob_odd_divisor_21_factorial :
  let n := factorial 21
  let totalDivisors := numDivisors n
  let oddDivisors := numOddDivisors n
  (oddDivisors : ℚ) / totalDivisors = 1 / 19 := by sorry

end NUMINAMATH_CALUDE_prob_odd_divisor_21_factorial_l2327_232727


namespace NUMINAMATH_CALUDE_inequality_of_powers_l2327_232760

theorem inequality_of_powers (α : Real) (h : α ∈ Set.Ioo (π/4) (π/2)) :
  (Real.cos α) ^ (Real.sin α) < (Real.cos α) ^ (Real.cos α) ∧
  (Real.cos α) ^ (Real.cos α) < (Real.sin α) ^ (Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l2327_232760


namespace NUMINAMATH_CALUDE_tiffany_bags_next_day_l2327_232778

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 7

/-- The additional number of bags Tiffany found on the next day compared to Monday -/
def additional_bags : ℕ := 5

/-- The total number of bags Tiffany found on the next day -/
def next_day_bags : ℕ := monday_bags + additional_bags

theorem tiffany_bags_next_day : next_day_bags = 12 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_next_day_l2327_232778


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2327_232722

theorem complex_magnitude_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2327_232722


namespace NUMINAMATH_CALUDE_pentagonal_pyramid_faces_pentagonal_pyramid_faces_proof_l2327_232707

/-- A pentagonal pyramid is a three-dimensional shape with a pentagonal base and triangular faces connecting the base to an apex. -/
structure PentagonalPyramid where
  base : Pentagon
  triangular_faces : Fin 5 → Triangle

/-- A pentagon is a polygon with 5 sides. -/
structure Pentagon where
  sides : Fin 5 → Segment

/-- Theorem: The number of faces of a pentagonal pyramid is 6. -/
theorem pentagonal_pyramid_faces (p : PentagonalPyramid) : Nat :=
  6

#check pentagonal_pyramid_faces

/-- Proof of the theorem -/
theorem pentagonal_pyramid_faces_proof (p : PentagonalPyramid) : 
  pentagonal_pyramid_faces p = 6 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_pyramid_faces_pentagonal_pyramid_faces_proof_l2327_232707


namespace NUMINAMATH_CALUDE_system_solution_l2327_232749

theorem system_solution (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + x + y = 80) : 
  x^2 + y^2 = 3980 / 121 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2327_232749


namespace NUMINAMATH_CALUDE_hillarys_deposit_l2327_232795

/-- Hillary's flea market earnings and deposit problem -/
theorem hillarys_deposit (crafts_sold : ℕ) (price_per_craft extra_tip remaining_cash : ℝ) 
  (h1 : crafts_sold = 3)
  (h2 : price_per_craft = 12)
  (h3 : extra_tip = 7)
  (h4 : remaining_cash = 25) :
  let total_earnings := crafts_sold * price_per_craft + extra_tip
  total_earnings - remaining_cash = 18 := by sorry

end NUMINAMATH_CALUDE_hillarys_deposit_l2327_232795


namespace NUMINAMATH_CALUDE_haley_money_difference_l2327_232762

/-- Calculates the difference between the final and initial amount of money Haley has after various transactions. -/
theorem haley_money_difference :
  let initial_amount : ℚ := 2
  let chores_earnings : ℚ := 5.25
  let birthday_gift : ℚ := 10
  let neighbor_help : ℚ := 7.5
  let found_money : ℚ := 0.5
  let aunt_gift_pounds : ℚ := 3
  let pound_to_dollar : ℚ := 1.3
  let candy_spent : ℚ := 3.75
  let money_lost : ℚ := 1.5
  
  let total_received : ℚ := chores_earnings + birthday_gift + neighbor_help + found_money + aunt_gift_pounds * pound_to_dollar
  let total_spent : ℚ := candy_spent + money_lost
  let final_amount : ℚ := initial_amount + total_received - total_spent
  
  final_amount - initial_amount = 19.9 := by sorry

end NUMINAMATH_CALUDE_haley_money_difference_l2327_232762


namespace NUMINAMATH_CALUDE_lcm_problem_l2327_232774

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 30 m = 90) (h2 : Nat.lcm m 45 = 180) : m = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2327_232774


namespace NUMINAMATH_CALUDE_at_op_zero_at_op_distributive_at_op_max_for_rectangle_l2327_232780

/-- Operation @ for real numbers -/
def at_op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

/-- Theorem 1: If a @ b = 0, then a = 0 or b = 0 -/
theorem at_op_zero (a b : ℝ) : at_op a b = 0 → a = 0 ∨ b = 0 := by sorry

/-- Theorem 2: a @ (b + c) = a @ b + a @ c -/
theorem at_op_distributive (a b c : ℝ) : at_op a (b + c) = at_op a b + at_op a c := by sorry

/-- Theorem 3: For a rectangle with fixed perimeter, a @ b is maximized when a = b -/
theorem at_op_max_for_rectangle (a b : ℝ) (h : a > 0 ∧ b > 0) (perimeter : ℝ) 
  (h_perimeter : 2 * (a + b) = perimeter) :
  ∀ x y, x > 0 → y > 0 → 2 * (x + y) = perimeter → at_op a b ≥ at_op x y := by sorry

end NUMINAMATH_CALUDE_at_op_zero_at_op_distributive_at_op_max_for_rectangle_l2327_232780


namespace NUMINAMATH_CALUDE_main_theorem_l2327_232703

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the main theorem
theorem main_theorem (x : ℝ) (h : (lg x)^2 * lg (10 * x) < 0) :
  (1 / lg (10 * x)) * Real.sqrt ((lg x)^2 + (lg (10 * x))^2) = -1 :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l2327_232703


namespace NUMINAMATH_CALUDE_triangulations_equal_catalan_l2327_232724

/-- Number of triangulations of an n-sided polygon -/
def T (n : ℕ) : ℕ := sorry

/-- Catalan numbers -/
def C (n : ℕ) : ℕ := sorry

/-- Theorem: The number of triangulations of an n-sided polygon
    is equal to the (n-2)th Catalan number -/
theorem triangulations_equal_catalan (n : ℕ) : T n = C (n - 2) := by sorry

end NUMINAMATH_CALUDE_triangulations_equal_catalan_l2327_232724


namespace NUMINAMATH_CALUDE_quadruple_solution_l2327_232752

theorem quadruple_solution :
  ∀ a b c d : ℕ+,
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    a + b = c * d ∧ a * b = c + d →
    (a = 1 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
    (a = 1 ∧ b = 5 ∧ c = 3 ∧ d = 2) ∨
    (a = 5 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
    (a = 5 ∧ b = 1 ∧ c = 3 ∧ d = 2) ∨
    (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5) ∨
    (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 1) ∨
    (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∨
    (a = 3 ∧ b = 2 ∧ c = 5 ∧ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadruple_solution_l2327_232752


namespace NUMINAMATH_CALUDE_min_socks_for_ten_pairs_five_colors_l2327_232775

/-- The minimum number of socks needed to guarantee a certain number of pairs, given a number of colors -/
def min_socks (colors : ℕ) (pairs : ℕ) : ℕ := 2 * pairs + colors - 1

/-- Theorem stating that 24 socks are needed to guarantee 10 pairs with 5 colors -/
theorem min_socks_for_ten_pairs_five_colors :
  min_socks 5 10 = 24 := by sorry

end NUMINAMATH_CALUDE_min_socks_for_ten_pairs_five_colors_l2327_232775


namespace NUMINAMATH_CALUDE_simplify_expression_l2327_232746

theorem simplify_expression (x y : ℝ) : (5*x - y) - 3*(2*x - 3*y) + x = 8*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2327_232746


namespace NUMINAMATH_CALUDE_system_of_equations_l2327_232755

theorem system_of_equations (a b : ℝ) 
  (eq1 : 2020*a + 2024*b = 2040)
  (eq2 : 2022*a + 2026*b = 2050)
  (eq3 : 2025*a + 2028*b = 2065) :
  a + 2*b = 5 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l2327_232755


namespace NUMINAMATH_CALUDE_participants_meet_on_DA_l2327_232785

/-- Represents a participant in the square walking problem -/
structure Participant where
  speed : ℝ
  startPoint : ℕ

/-- Represents the square and the walking problem -/
structure SquareWalk where
  sideLength : ℝ
  participantA : Participant
  participantB : Participant

/-- The point where the participants meet -/
def meetingPoint (sw : SquareWalk) : ℕ :=
  sorry

theorem participants_meet_on_DA (sw : SquareWalk) 
  (h1 : sw.sideLength = 90)
  (h2 : sw.participantA.speed = 65)
  (h3 : sw.participantB.speed = 72)
  (h4 : sw.participantA.startPoint = 0)
  (h5 : sw.participantB.startPoint = 1) :
  meetingPoint sw = 3 :=
sorry

end NUMINAMATH_CALUDE_participants_meet_on_DA_l2327_232785


namespace NUMINAMATH_CALUDE_some_number_value_l2327_232714

theorem some_number_value : ∃ n : ℤ, (481 + 426) * n - 4 * 481 * 426 = 3025 ∧ n = 906 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2327_232714


namespace NUMINAMATH_CALUDE_fill_box_with_cubes_l2327_232779

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.depth

/-- Finds the greatest common divisor of three natural numbers -/
def gcd3 (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

/-- Calculates the side length of the largest cube that can fit evenly into the box -/
def largestCubeSideLength (d : BoxDimensions) : ℕ :=
  gcd3 d.length d.width d.depth

/-- Calculates the number of cubes needed to fill the box completely -/
def numberOfCubes (d : BoxDimensions) : ℕ :=
  boxVolume d / (largestCubeSideLength d)^3

/-- The main theorem stating that 80 cubes are needed to fill the given box -/
theorem fill_box_with_cubes (d : BoxDimensions) 
  (h1 : d.length = 30) (h2 : d.width = 48) (h3 : d.depth = 12) : 
  numberOfCubes d = 80 := by
  sorry

end NUMINAMATH_CALUDE_fill_box_with_cubes_l2327_232779


namespace NUMINAMATH_CALUDE_cost_to_fill_can_n_l2327_232742

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The cost in dollars to fill a given volume of gasoline -/
def fillCost (volume : ℝ) : ℝ := sorry

theorem cost_to_fill_can_n (can_b can_n : Cylinder) (half_b_cost : ℝ) : 
  can_n.radius = 2 * can_b.radius →
  can_n.height = can_b.height / 2 →
  fillCost (π * can_b.radius^2 * can_b.height / 2) = 4 →
  fillCost (π * can_n.radius^2 * can_n.height) = 16 := by sorry

end NUMINAMATH_CALUDE_cost_to_fill_can_n_l2327_232742


namespace NUMINAMATH_CALUDE_borrowed_sheets_theorem_l2327_232759

/-- Represents a set of notes with sheets and pages -/
structure Notes where
  total_sheets : ℕ
  pages_per_sheet : ℕ
  total_pages : ℕ
  h_pages : total_pages = total_sheets * pages_per_sheet

/-- Represents the state of notes after some sheets are borrowed -/
structure BorrowedNotes where
  original : Notes
  borrowed_sheets : ℕ
  sheets_before : ℕ
  h_valid : sheets_before + borrowed_sheets < original.total_sheets

/-- Calculates the average page number of remaining sheets -/
def average_page_number (bn : BorrowedNotes) : ℚ :=
  let remaining_pages := bn.original.total_pages - bn.borrowed_sheets * bn.original.pages_per_sheet
  let sum_before := bn.sheets_before * (bn.sheets_before * bn.original.pages_per_sheet + 1)
  let first_after := (bn.sheets_before + bn.borrowed_sheets) * bn.original.pages_per_sheet + 1
  let last_after := bn.original.total_pages
  let sum_after := (first_after + last_after) * (last_after - first_after + 1) / 2
  (sum_before + sum_after) / remaining_pages

/-- Theorem stating that if 17 sheets are borrowed from a 35-sheet set of notes,
    the average page number of remaining sheets is 28 -/
theorem borrowed_sheets_theorem (bn : BorrowedNotes)
  (h_total_sheets : bn.original.total_sheets = 35)
  (h_pages_per_sheet : bn.original.pages_per_sheet = 2)
  (h_total_pages : bn.original.total_pages = 70)
  (h_avg : average_page_number bn = 28) :
  bn.borrowed_sheets = 17 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sheets_theorem_l2327_232759


namespace NUMINAMATH_CALUDE_area_after_reflection_l2327_232782

/-- Right triangle ABC with given side lengths -/
structure RightTriangle where
  AB : ℝ
  BC : ℝ
  right_angle : AB > 0 ∧ BC > 0

/-- Points after reflection -/
structure ReflectedPoints where
  A' : ℝ × ℝ
  B' : ℝ × ℝ
  C' : ℝ × ℝ

/-- Function to perform reflections -/
def reflect (t : RightTriangle) : ReflectedPoints := sorry

/-- Calculate area of triangle A'B'C' -/
def area_A'B'C' (p : ReflectedPoints) : ℝ := sorry

/-- Main theorem -/
theorem area_after_reflection (t : RightTriangle) 
  (h1 : t.AB = 5)
  (h2 : t.BC = 12) : 
  area_A'B'C' (reflect t) = 17.5 := by sorry

end NUMINAMATH_CALUDE_area_after_reflection_l2327_232782


namespace NUMINAMATH_CALUDE_incorrect_vs_correct_calculation_l2327_232757

theorem incorrect_vs_correct_calculation (x : ℝ) (h : x - 3 + 49 = 66) : 
  (3 * x + 49) - 66 = 43 := by
sorry

end NUMINAMATH_CALUDE_incorrect_vs_correct_calculation_l2327_232757


namespace NUMINAMATH_CALUDE_f_f_eq_f_solution_l2327_232710

def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem f_f_eq_f_solution :
  ∀ x : ℝ, f (f x) = f x ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_f_eq_f_solution_l2327_232710


namespace NUMINAMATH_CALUDE_seventh_root_of_negative_two_plus_fourth_root_of_negative_three_l2327_232711

theorem seventh_root_of_negative_two_plus_fourth_root_of_negative_three : 
  ((-2 : ℝ) ^ 7) ^ (1/7) + ((-3 : ℝ) ^ 4) ^ (1/4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_negative_two_plus_fourth_root_of_negative_three_l2327_232711


namespace NUMINAMATH_CALUDE_train_speed_second_part_l2327_232700

/-- Proves that the speed of a train during the second part of a journey is 20 kmph,
    given specific conditions about the journey. -/
theorem train_speed_second_part 
  (x : ℝ) 
  (h_positive : x > 0) 
  (speed_first : ℝ) 
  (h_speed_first : speed_first = 40) 
  (distance_first : ℝ) 
  (h_distance_first : distance_first = x) 
  (distance_second : ℝ) 
  (h_distance_second : distance_second = 2 * x) 
  (distance_total : ℝ) 
  (h_distance_total : distance_total = 6 * x) 
  (speed_average : ℝ) 
  (h_speed_average : speed_average = 48) : 
  ∃ (speed_second : ℝ), speed_second = 20 := by
sorry


end NUMINAMATH_CALUDE_train_speed_second_part_l2327_232700


namespace NUMINAMATH_CALUDE_point_inside_given_circle_l2327_232701

def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 18

def point_inside_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 3)^2 < 18

theorem point_inside_given_circle :
  point_inside_circle 1 1 := by sorry

end NUMINAMATH_CALUDE_point_inside_given_circle_l2327_232701


namespace NUMINAMATH_CALUDE_right_triangle_sides_l2327_232734

theorem right_triangle_sides : ∀ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b + c = 40) →
  (a^2 + b^2 = c^2) →
  ((a + 4)^2 + (b + 1)^2 = (c + 3)^2) →
  (a < b) →
  (a = 8 ∧ b = 15 ∧ c = 17) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l2327_232734


namespace NUMINAMATH_CALUDE_distribution_methods_count_l2327_232740

/-- The number of ways to distribute tickets to tourists -/
def distribute_tickets : ℕ :=
  Nat.choose 6 2 * Nat.choose 4 2 * (Nat.factorial 2)

/-- Theorem stating that the number of distribution methods is 180 -/
theorem distribution_methods_count : distribute_tickets = 180 := by
  sorry

end NUMINAMATH_CALUDE_distribution_methods_count_l2327_232740


namespace NUMINAMATH_CALUDE_chess_tournament_players_l2327_232705

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- Number of players not among the 12 lowest-scoring players
  total_players : ℕ := n + 12
  total_points : ℕ := n * (n - 1) + 132
  games_played : ℕ := (total_players * (total_players - 1)) / 2

/-- The theorem stating that the total number of players is 24 -/
theorem chess_tournament_players (t : ChessTournament) : t.total_players = 24 := by
  sorry

#check chess_tournament_players

end NUMINAMATH_CALUDE_chess_tournament_players_l2327_232705


namespace NUMINAMATH_CALUDE_product_of_primes_l2327_232733

def smallest_one_digit_primes : List Nat := [2, 3]
def largest_three_digit_prime : Nat := 997

theorem product_of_primes :
  (smallest_one_digit_primes.prod * largest_three_digit_prime) = 5982 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_l2327_232733


namespace NUMINAMATH_CALUDE_gcd_problem_l2327_232715

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 1177) :
  Int.gcd (3 * b^2 + 34 * b + 76) (b + 14) = 2 := by sorry

end NUMINAMATH_CALUDE_gcd_problem_l2327_232715


namespace NUMINAMATH_CALUDE_correct_conclusions_l2327_232751

theorem correct_conclusions :
  (∀ a b : ℝ, a + b > 0 ∧ a * b > 0 → a > 0 ∧ b > 0) ∧
  (∀ a b : ℝ, b ≠ 0 → a / b = -1 → a + b = 0) ∧
  (∀ a b c : ℝ, a < b ∧ b < c → |a - b| + |b - c| = |a - c|) :=
by sorry

end NUMINAMATH_CALUDE_correct_conclusions_l2327_232751


namespace NUMINAMATH_CALUDE_tiles_needed_l2327_232781

-- Define the dimensions
def tile_size : ℕ := 6
def kitchen_width : ℕ := 48
def kitchen_height : ℕ := 72

-- Define the theorem
theorem tiles_needed : 
  (kitchen_width / tile_size) * (kitchen_height / tile_size) = 96 := by
  sorry

end NUMINAMATH_CALUDE_tiles_needed_l2327_232781


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2327_232787

/-- Given a hyperbola with equation x² - y²/b² = 1 where b > 0,
    if one of its asymptotes has the equation y = 3x, then b = 3 -/
theorem hyperbola_asymptote (b : ℝ) (h1 : b > 0) :
  (∃ x y : ℝ, x^2 - y^2/b^2 = 1 ∧ y = 3*x) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2327_232787


namespace NUMINAMATH_CALUDE_pascal_row_15_sum_l2327_232767

/-- Definition of Pascal's Triangle sum for a given row -/
def pascal_sum (n : ℕ) : ℕ := 2^n

/-- Theorem: The sum of numbers in row 15 of Pascal's Triangle is 32768 -/
theorem pascal_row_15_sum : pascal_sum 15 = 32768 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row_15_sum_l2327_232767


namespace NUMINAMATH_CALUDE_proposition_truth_l2327_232716

theorem proposition_truth (p q : Prop) 
  (h1 : ¬p) 
  (h2 : p ∨ q) : 
  ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_proposition_truth_l2327_232716


namespace NUMINAMATH_CALUDE_nancy_homework_pages_l2327_232723

theorem nancy_homework_pages (total_problems : ℕ) (finished_problems : ℕ) (problems_per_page : ℕ) : 
  total_problems = 101 → 
  finished_problems = 47 → 
  problems_per_page = 9 → 
  (total_problems - finished_problems) / problems_per_page = 6 := by
sorry

end NUMINAMATH_CALUDE_nancy_homework_pages_l2327_232723


namespace NUMINAMATH_CALUDE_negation_of_all_ge_two_l2327_232748

theorem negation_of_all_ge_two :
  (¬ (∀ x : ℝ, x ≥ 2)) ↔ (∃ x₀ : ℝ, x₀ < 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_ge_two_l2327_232748


namespace NUMINAMATH_CALUDE_blue_section_damage_probability_l2327_232726

/-- The number of trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The number of successes we're interested in -/
def k : ℕ := 7

/-- The probability of exactly k successes in n Bernoulli trials with probability p -/
def bernoulli_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem blue_section_damage_probability :
  bernoulli_probability n k p = 128/823543 := by
  sorry

end NUMINAMATH_CALUDE_blue_section_damage_probability_l2327_232726


namespace NUMINAMATH_CALUDE_parabola_equation_l2327_232790

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a line passing through two points -/
structure Line where
  a : Point
  b : Point

theorem parabola_equation (c : Parabola) (l : Line) :
  let f := Point.mk (c.p / 2) 0  -- Focus of the parabola
  let m := Point.mk 3 2  -- Midpoint of AB
  (l.a.y ^ 2 = 2 * c.p * l.a.x) ∧  -- A is on the parabola
  (l.b.y ^ 2 = 2 * c.p * l.b.x) ∧  -- B is on the parabola
  ((l.a.x + l.b.x) / 2 = m.x) ∧  -- M is the midpoint of AB (x-coordinate)
  ((l.a.y + l.b.y) / 2 = m.y) ∧  -- M is the midpoint of AB (y-coordinate)
  (f.x - l.a.x) * (l.b.y - l.a.y) = (f.y - l.a.y) * (l.b.x - l.a.x)  -- L passes through F
  →
  c.p = 2 ∨ c.p = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2327_232790


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l2327_232739

/-- The probability of picking two red balls from a bag containing 4 red, 4 blue, and 2 green balls -/
theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ) :
  total_balls = red_balls + blue_balls + green_balls →
  red_balls = 4 →
  blue_balls = 4 →
  green_balls = 2 →
  (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l2327_232739


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l2327_232731

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def starters : ℕ := 7

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem volleyball_team_selection :
  choose total_players starters - choose (total_players - quadruplets) starters = 28392 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l2327_232731


namespace NUMINAMATH_CALUDE_factorial_p_adic_valuation_binomial_p_adic_valuation_binomial_p_adic_valuation_carries_binomial_p_adic_valuation_zero_l2327_232737

-- Define p-adic valuation
noncomputable def v_p (p : ℕ) (n : ℕ) : ℚ := sorry

-- Define sum of digits in base p
def τ_p (p : ℕ) (n : ℕ) : ℕ := sorry

-- Define number of carries when adding in base p
def carries_base_p (p : ℕ) (a b : ℕ) : ℕ := sorry

-- Lemma
theorem factorial_p_adic_valuation (p : ℕ) (n : ℕ) : 
  v_p p (n.factorial) = (n - τ_p p n) / (p - 1) := sorry

-- Theorem 1
theorem binomial_p_adic_valuation (p : ℕ) (n k : ℕ) (h : k ≤ n) :
  v_p p (n.choose k) = (τ_p p k + τ_p p (n - k) - τ_p p n) / (p - 1) := sorry

-- Theorem 2
theorem binomial_p_adic_valuation_carries (p : ℕ) (n k : ℕ) (h : k ≤ n) :
  v_p p (n.choose k) = carries_base_p p k (n - k) := sorry

-- Theorem 3
theorem binomial_p_adic_valuation_zero (p : ℕ) (n k : ℕ) (h : k ≤ n) :
  v_p p (n.choose k) = 0 ↔ carries_base_p p k (n - k) = 0 := sorry

end NUMINAMATH_CALUDE_factorial_p_adic_valuation_binomial_p_adic_valuation_binomial_p_adic_valuation_carries_binomial_p_adic_valuation_zero_l2327_232737


namespace NUMINAMATH_CALUDE_optimal_plan_maximizes_profit_l2327_232798

/-- Represents the production plan for transformers --/
structure ProductionPlan where
  typeA : ℕ
  typeB : ℕ

/-- Calculates the profit for a given production plan --/
def profit (plan : ProductionPlan) : ℕ :=
  12 * plan.typeA + 10 * plan.typeB

/-- Checks if a production plan is feasible given the resource constraints --/
def isFeasible (plan : ProductionPlan) : Prop :=
  5 * plan.typeA + 3 * plan.typeB ≤ 481 ∧
  3 * plan.typeA + 2 * plan.typeB ≤ 301

/-- The optimal production plan --/
def optimalPlan : ProductionPlan :=
  { typeA := 1, typeB := 149 }

/-- Theorem stating that the optimal plan achieves the maximum profit --/
theorem optimal_plan_maximizes_profit :
  isFeasible optimalPlan ∧
  ∀ plan, isFeasible plan → profit plan ≤ profit optimalPlan :=
by sorry

#eval profit optimalPlan  -- Should output 1502

end NUMINAMATH_CALUDE_optimal_plan_maximizes_profit_l2327_232798


namespace NUMINAMATH_CALUDE_probability_outside_circle_l2327_232789

/-- A die roll outcome is a natural number between 1 and 6 -/
def DieRoll : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

/-- A point P is defined by two die roll outcomes -/
structure Point where
  m : DieRoll
  n : DieRoll

/-- A point P(m,n) is outside the circle if m^2 + n^2 > 25 -/
def isOutsideCircle (p : Point) : Prop :=
  (p.m.val ^ 2 + p.n.val ^ 2 : ℚ) > 25

/-- The total number of possible outcomes when rolling a die twice -/
def totalOutcomes : ℕ := 36

/-- The number of outcomes resulting in a point outside the circle -/
def favorableOutcomes : ℕ := 11

/-- The main theorem: probability of a point being outside the circle -/
theorem probability_outside_circle :
  (favorableOutcomes : ℚ) / totalOutcomes = 11 / 36 := by sorry

end NUMINAMATH_CALUDE_probability_outside_circle_l2327_232789


namespace NUMINAMATH_CALUDE_system_solution_l2327_232702

theorem system_solution (x y z : ℚ) 
  (eq1 : y + z = 15 - 2*x)
  (eq2 : x + z = -10 - 2*y)
  (eq3 : x + y = 4 - 2*z) :
  2*x + 2*y + 2*z = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2327_232702


namespace NUMINAMATH_CALUDE_stating_prob_reach_heaven_l2327_232794

/-- A point in the 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The starting point of the walk -/
def start : LatticePoint := ⟨1, 1⟩

/-- Predicate for heaven points -/
def is_heaven (p : LatticePoint) : Prop :=
  ∃ m n : ℤ, p.x = 6 * m ∧ p.y = 6 * n

/-- Predicate for hell points -/
def is_hell (p : LatticePoint) : Prop :=
  ∃ m n : ℤ, p.x = 6 * m + 3 ∧ p.y = 6 * n + 3

/-- The probability of reaching heaven -/
def prob_heaven : ℚ := 13 / 22

/-- 
Theorem stating that the probability of reaching heaven 
before hell in a random lattice walk starting from (1,1) is 13/22 
-/
theorem prob_reach_heaven : 
  prob_heaven = 13 / 22 :=
sorry

end NUMINAMATH_CALUDE_stating_prob_reach_heaven_l2327_232794


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l2327_232763

theorem youngest_sibling_age (y : ℝ) : 
  (y + (y + 3) + (y + 6) + (y + 7)) / 4 = 30 → y = 26 := by
  sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l2327_232763


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2327_232743

theorem arithmetic_calculation : 15 * 20 - 25 * 15 + 10 * 25 = 175 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2327_232743


namespace NUMINAMATH_CALUDE_meet_once_l2327_232766

/-- Represents the movement of Hannah and the van --/
structure Movement where
  hannah_speed : ℝ
  van_speed : ℝ
  pail_distance : ℝ
  van_stop_time : ℝ

/-- Calculates the number of meetings between Hannah and the van --/
def number_of_meetings (m : Movement) : ℕ :=
  sorry

/-- Theorem stating that Hannah and the van meet exactly once --/
theorem meet_once (m : Movement) 
  (h1 : m.hannah_speed = 6)
  (h2 : m.van_speed = 12)
  (h3 : m.pail_distance = 150)
  (h4 : m.van_stop_time = 45)
  : number_of_meetings m = 1 := by
  sorry

end NUMINAMATH_CALUDE_meet_once_l2327_232766


namespace NUMINAMATH_CALUDE_scientific_notation_of_13000_l2327_232720

theorem scientific_notation_of_13000 :
  ∃ (a : ℝ) (n : ℤ), 13000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.3 ∧ n = 4 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_13000_l2327_232720


namespace NUMINAMATH_CALUDE_equation_substitution_l2327_232793

theorem equation_substitution :
  ∀ x y : ℝ,
  (y = x + 1) →
  (3 * x - y = 18) →
  (3 * x - x - 1 = 18) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_substitution_l2327_232793


namespace NUMINAMATH_CALUDE_unique_solution_for_diophantine_equation_l2327_232784

theorem unique_solution_for_diophantine_equation :
  ∀ m a b : ℤ,
    m > 1 ∧ a > 1 ∧ b > 1 →
    (m + 1) * a = m * b + 1 →
    m = 2 ∧ a = 3 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_diophantine_equation_l2327_232784


namespace NUMINAMATH_CALUDE_delta_composition_l2327_232747

-- Define the Delta operations
def rightDelta (x : ℤ) : ℤ := 9 - x
def leftDelta (x : ℤ) : ℤ := x - 9

-- State the theorem
theorem delta_composition : leftDelta (rightDelta 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_delta_composition_l2327_232747


namespace NUMINAMATH_CALUDE_infinitely_many_primes_with_property_l2327_232770

-- Define the property for a prime p
def hasDivisibilityProperty (p : Nat) : Prop :=
  ∃ k : Nat, k > 0 ∧ p ∣ (2^k - 3)

-- State the theorem
theorem infinitely_many_primes_with_property :
  ∀ n : Nat, ∃ p : Nat, p > n ∧ Prime p ∧ hasDivisibilityProperty p := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_with_property_l2327_232770


namespace NUMINAMATH_CALUDE_least_multiple_with_digit_product_multiple_l2327_232799

/-- Returns the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Returns true if n is a multiple of m -/
def isMultipleOf (n m : ℕ) : Prop := sorry

/-- Returns true if n is the least number satisfying the given property -/
def isLeast (n : ℕ) (property : ℕ → Prop) : Prop := sorry

theorem least_multiple_with_digit_product_multiple :
  isLeast 315 (λ n : ℕ => isMultipleOf n 15 ∧ 
                          n > 0 ∧ 
                          isMultipleOf (digitProduct n) 15 ∧ 
                          digitProduct n > 0) := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_with_digit_product_multiple_l2327_232799


namespace NUMINAMATH_CALUDE_school_test_questions_l2327_232728

theorem school_test_questions (sections : ℕ) (correct_answers : ℕ) 
  (h_sections : sections = 5)
  (h_correct : correct_answers = 20)
  (h_percentage : ∀ x : ℕ, x > 0 → (60 : ℚ) / 100 < (correct_answers : ℚ) / x ∧ (correct_answers : ℚ) / x < 70 / 100 → x = 30) :
  ∃! total_questions : ℕ, 
    total_questions > 0 ∧
    total_questions % sections = 0 ∧
    (60 : ℚ) / 100 < (correct_answers : ℚ) / total_questions ∧
    (correct_answers : ℚ) / total_questions < 70 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_school_test_questions_l2327_232728


namespace NUMINAMATH_CALUDE_solution_count_l2327_232791

theorem solution_count : ∃ (S : Finset ℕ), 
  (∀ x ∈ S, 1 ≤ x ∧ x ≤ 200) ∧
  (∀ x ∈ S, ∃ k ∈ Finset.range 200, x = k + 1) ∧
  (∀ x ∈ S, ∀ k ∈ Finset.range 10, x ≠ (k + 1)^2) ∧
  Finset.card S = 190 := by
sorry

end NUMINAMATH_CALUDE_solution_count_l2327_232791


namespace NUMINAMATH_CALUDE_plot_length_is_sixty_l2327_232704

/-- Given a rectangular plot with the following properties:
    1. The length is 20 meters more than the breadth.
    2. The cost of fencing the plot at 26.50 per meter is Rs. 5300.
    This theorem proves that the length of the plot is 60 meters. -/
theorem plot_length_is_sixty (breadth : ℝ) (length : ℝ) (perimeter : ℝ) :
  length = breadth + 20 →
  perimeter = 2 * (length + breadth) →
  26.50 * perimeter = 5300 →
  length = 60 := by
sorry

end NUMINAMATH_CALUDE_plot_length_is_sixty_l2327_232704


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt13_l2327_232764

theorem closest_integer_to_sqrt13 : 
  ∀ n : ℤ, n ∈ ({2, 3, 4, 5} : Set ℤ) → |n - Real.sqrt 13| ≥ |4 - Real.sqrt 13| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt13_l2327_232764


namespace NUMINAMATH_CALUDE_house_resale_price_l2327_232753

theorem house_resale_price (initial_value : ℝ) (loss_percent : ℝ) (interest_rate : ℝ) (gain_percent : ℝ) : 
  initial_value = 12000 ∧ 
  loss_percent = 0.15 ∧ 
  interest_rate = 0.05 ∧ 
  gain_percent = 0.2 → 
  initial_value * (1 - loss_percent) * (1 + interest_rate) * (1 + gain_percent) = 12852 :=
by sorry

end NUMINAMATH_CALUDE_house_resale_price_l2327_232753


namespace NUMINAMATH_CALUDE_second_polygon_sides_l2327_232741

/-- Given two regular polygons with the same perimeter, where the first has 24 sides
    and its side length is three times that of the second, prove the second has 72 sides. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 →  -- Ensure side length is positive
  24 * (3 * s) = n * s →  -- Same perimeter condition
  n = 72 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l2327_232741


namespace NUMINAMATH_CALUDE_august_mail_l2327_232735

def mail_sequence (n : ℕ) : ℕ := 5 * 2^n

theorem august_mail :
  mail_sequence 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_august_mail_l2327_232735


namespace NUMINAMATH_CALUDE_smallest_circle_passing_through_intersection_l2327_232719

-- Define the line
def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the smallest circle
def smallest_circle (x y : ℝ) : Prop := 5*x^2 + 5*y^2 + 6*x - 18*y - 1 = 0

-- Theorem statement
theorem smallest_circle_passing_through_intersection :
  ∃ (x1 y1 x2 y2 : ℝ),
    line x1 y1 ∧ line x2 y2 ∧
    original_circle x1 y1 ∧ original_circle x2 y2 ∧
    (∀ (x y : ℝ), smallest_circle x y ↔ 
      ((x - x1)^2 + (y - y1)^2 = (x - x2)^2 + (y - y2)^2 ∧
       ∀ (c : ℝ → ℝ → Prop), (c x1 y1 ∧ c x2 y2) → 
         (∃ (xc yc r : ℝ), ∀ (x y : ℝ), c x y ↔ (x - xc)^2 + (y - yc)^2 = r^2) →
         (∃ (xs ys rs : ℝ), ∀ (x y : ℝ), smallest_circle x y ↔ (x - xs)^2 + (y - ys)^2 = rs^2 ∧ rs ≤ r))) :=
by sorry


end NUMINAMATH_CALUDE_smallest_circle_passing_through_intersection_l2327_232719


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l2327_232706

theorem greatest_integer_b_for_all_real_domain : ∃ (b : ℤ),
  (∀ (x : ℝ), x^2 + (b : ℝ) * x + 12 ≠ 0) ∧
  (∀ (c : ℤ), c > b → ∃ (x : ℝ), x^2 + (c : ℝ) * x + 12 = 0) ∧
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l2327_232706


namespace NUMINAMATH_CALUDE_triangle_angle_difference_l2327_232738

/-- In a triangle with angles a, b, and c, where b = 2a and c = a - 15, 
    prove that a - c = 15 --/
theorem triangle_angle_difference (a b c : ℝ) : 
  a + b + c = 180 → b = 2 * a → c = a - 15 → a - c = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_difference_l2327_232738


namespace NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l2327_232750

theorem no_solutions_absolute_value_equation :
  ¬ ∃ x : ℝ, x > 0 ∧ |x + 4| = 3 - x := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l2327_232750


namespace NUMINAMATH_CALUDE_two_digit_number_solution_l2327_232730

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n ≤ 99 }

/-- Converts a two-digit number to its decimal representation -/
def toDecimal (n : TwoDigitNumber) : ℚ :=
  n.val / 100

/-- Converts a two-digit number to its repeating decimal representation -/
def toRepeatingDecimal (n : TwoDigitNumber) : ℚ :=
  n.val / 99

theorem two_digit_number_solution (cd : TwoDigitNumber) :
  54 * (toRepeatingDecimal cd - toDecimal cd) = (36 : ℚ) / 100 →
  cd.val = 65 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_solution_l2327_232730


namespace NUMINAMATH_CALUDE_f_inequality_iff_a_bound_l2327_232768

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a / x - (a + 1) * Real.log x

theorem f_inequality_iff_a_bound (a : ℝ) :
  (∀ x > 0, f a x ≤ x) ↔ a ≥ 1 / (Real.exp 1 - 1) := by sorry

end NUMINAMATH_CALUDE_f_inequality_iff_a_bound_l2327_232768


namespace NUMINAMATH_CALUDE_computer_price_problem_l2327_232777

theorem computer_price_problem (P : ℝ) : 
  1.30 * P = 351 ∧ 2 * P = 540 → P = 270 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_problem_l2327_232777


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_function_order_l2327_232758

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution_implies_function_order
  (a b c : ℝ)
  (h1 : ∀ x, (x < -2 ∨ x > 4) ↔ a * x^2 + b * x + c < 0) :
  f a b c 5 < f a b c 2 ∧ f a b c 2 < f a b c 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_function_order_l2327_232758


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l2327_232756

/-- An ellipse with center at the origin, one focus at (0,2), and a chord formed by
    the intersection with the line y=3x+7 whose midpoint has a y-coordinate of 1 --/
structure SpecialEllipse where
  /-- One focus of the ellipse --/
  focus : ℝ × ℝ
  /-- Slope of the intersecting line --/
  m : ℝ
  /-- y-intercept of the intersecting line --/
  b : ℝ
  /-- y-coordinate of the chord's midpoint --/
  midpoint_y : ℝ
  /-- Conditions for the special ellipse --/
  h1 : focus = (0, 2)
  h2 : m = 3
  h3 : b = 7
  h4 : midpoint_y = 1

/-- The equation of the ellipse --/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 8 + y^2 / 12 = 1

/-- Theorem stating that the given special ellipse has the specified equation --/
theorem special_ellipse_equation (e : SpecialEllipse) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | ellipse_equation p.1 p.2} ↔
    (x, y) ∈ {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 12 = 1} :=
by sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l2327_232756


namespace NUMINAMATH_CALUDE_max_cars_in_parking_lot_l2327_232744

/-- Represents a parking lot configuration -/
structure ParkingLot :=
  (grid : Fin 7 → Fin 7 → Bool)
  (gate : Fin 7 × Fin 7)

/-- Checks if a car can exit from its position -/
def canExit (lot : ParkingLot) (pos : Fin 7 × Fin 7) : Prop :=
  sorry

/-- Counts the number of cars in the parking lot -/
def carCount (lot : ParkingLot) : Nat :=
  sorry

/-- Checks if the parking lot configuration is valid -/
def isValidConfig (lot : ParkingLot) : Prop :=
  ∀ pos, lot.grid pos.1 pos.2 → canExit lot pos

/-- The main theorem stating the maximum number of cars that can be parked -/
theorem max_cars_in_parking_lot :
  ∃ (lot : ParkingLot), isValidConfig lot ∧ carCount lot = 28 ∧
  ∀ (other : ParkingLot), isValidConfig other → carCount other ≤ 28 :=
sorry

end NUMINAMATH_CALUDE_max_cars_in_parking_lot_l2327_232744


namespace NUMINAMATH_CALUDE_expression_simplification_l2327_232761

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b > 0) (hab : a^(1/3) * b^(1/4) ≠ 2) :
  ((a^2 * b * Real.sqrt b - 6 * a^(5/3) * b^(5/4) + 12 * a * b * a^(1/3) - 8 * a * b^(3/4))^(2/3)) /
  (a * b * a^(1/3) - 4 * a * b^(3/4) + 4 * a^(2/3) * Real.sqrt b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2327_232761


namespace NUMINAMATH_CALUDE_line_plane_intersection_equivalence_l2327_232783

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (intersect : Line → Line → Prop)
variable (within : Line → Plane → Prop)
variable (intersects_plane : Line → Plane → Prop)
variable (planes_intersect : Plane → Plane → Prop)

-- Define the specific lines and planes
variable (l m : Line)
variable (α β : Plane)

-- State the theorem
theorem line_plane_intersection_equivalence 
  (h1 : intersect l m)
  (h2 : within l α)
  (h3 : within m α)
  (h4 : ¬ within l β)
  (h5 : ¬ within m β) :
  (intersects_plane l β ∨ intersects_plane m β) ↔ planes_intersect α β := by
  sorry

end NUMINAMATH_CALUDE_line_plane_intersection_equivalence_l2327_232783


namespace NUMINAMATH_CALUDE_hedgehog_strawberry_baskets_l2327_232713

theorem hedgehog_strawberry_baskets :
  ∀ (baskets : ℕ) (strawberries_per_basket : ℕ) (hedgehogs : ℕ) (strawberries_eaten_per_hedgehog : ℕ),
    strawberries_per_basket = 900 →
    hedgehogs = 2 →
    strawberries_eaten_per_hedgehog = 1050 →
    (baskets * strawberries_per_basket : ℚ) * (2 : ℚ) / 9 = 
      baskets * strawberries_per_basket - hedgehogs * strawberries_eaten_per_hedgehog →
    baskets = 3 := by
  sorry

end NUMINAMATH_CALUDE_hedgehog_strawberry_baskets_l2327_232713


namespace NUMINAMATH_CALUDE_power_sum_equality_l2327_232769

theorem power_sum_equality : (-2)^48 + 3^(4^3 + 5^2 - 7^2) = 2^48 + 3^40 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2327_232769


namespace NUMINAMATH_CALUDE_isolation_process_complete_l2327_232718

/-- Represents a step in the process of isolating and counting bacteria --/
inductive ProcessStep
  | SoilSampling
  | SampleDilution
  | SpreadingDilution
  | SelectingColonies
  | Identification

/-- Represents the process of isolating and counting bacteria that decompose urea in soil --/
def IsolationProcess : List ProcessStep := 
  [ProcessStep.SoilSampling, 
   ProcessStep.SampleDilution, 
   ProcessStep.SpreadingDilution, 
   ProcessStep.SelectingColonies, 
   ProcessStep.Identification]

/-- The theorem states that the IsolationProcess contains all necessary steps in the correct order --/
theorem isolation_process_complete : 
  IsolationProcess = 
    [ProcessStep.SoilSampling, 
     ProcessStep.SampleDilution, 
     ProcessStep.SpreadingDilution, 
     ProcessStep.SelectingColonies, 
     ProcessStep.Identification] := by
  sorry


end NUMINAMATH_CALUDE_isolation_process_complete_l2327_232718


namespace NUMINAMATH_CALUDE_combined_average_mark_l2327_232729

/-- Given two classes with specified number of students and average marks,
    calculate the combined average mark of all students. -/
theorem combined_average_mark (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 → n2 > 0 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 / ((n1 : ℚ) + (n2 : ℚ)) =
  ((55 : ℚ) * 60 + (48 : ℚ) * 58) / ((55 : ℚ) + (48 : ℚ)) := by
  sorry

#eval ((55 : ℚ) * 60 + (48 : ℚ) * 58) / ((55 : ℚ) + (48 : ℚ))

end NUMINAMATH_CALUDE_combined_average_mark_l2327_232729


namespace NUMINAMATH_CALUDE_card_value_decrease_l2327_232765

theorem card_value_decrease (x : ℝ) : 
  (1 - x/100) * (1 - x/100) = 0.81 → x = 10 := by
sorry

end NUMINAMATH_CALUDE_card_value_decrease_l2327_232765


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2327_232721

theorem division_remainder_problem : ∃ (x : ℕ), 
  (1782 - x = 1500) ∧ 
  (∃ (r : ℕ), 1782 = 6 * x + r) ∧
  (1782 % x = 90) := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2327_232721


namespace NUMINAMATH_CALUDE_greatest_integer_solution_seven_satisfies_inequality_no_greater_integer_l2327_232717

theorem greatest_integer_solution (x : ℤ) : (7 : ℤ) - 5*x + x^2 > 24 → x ≤ 7 :=
by sorry

theorem seven_satisfies_inequality : (7 : ℤ) - 5*7 + 7^2 > 24 :=
by sorry

theorem no_greater_integer :
  ∀ y : ℤ, y > 7 → ¬((7 : ℤ) - 5*y + y^2 > 24) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_seven_satisfies_inequality_no_greater_integer_l2327_232717


namespace NUMINAMATH_CALUDE_power_product_equal_thousand_l2327_232754

theorem power_product_equal_thousand : 2^3 * 5^3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equal_thousand_l2327_232754


namespace NUMINAMATH_CALUDE_property_price_calculation_l2327_232792

/-- The price of the property in dollars given the price per square foot, house size, and barn size. -/
def property_price (price_per_sqft : ℚ) (house_size : ℚ) (barn_size : ℚ) : ℚ :=
  price_per_sqft * (house_size + barn_size)

/-- Theorem stating that the property price is $333,200 given the specified conditions. -/
theorem property_price_calculation :
  property_price 98 2400 1000 = 333200 := by
  sorry

end NUMINAMATH_CALUDE_property_price_calculation_l2327_232792


namespace NUMINAMATH_CALUDE_factorize_quadratic_xy_value_l2327_232788

-- Problem 1
theorem factorize_quadratic (x : ℝ) : 
  x^2 - 120*x + 3456 = (x - 48) * (x - 72) := by sorry

-- Problem 2
theorem xy_value (x y : ℝ) : 
  x^2 + y^2 + 8*x - 12*y + 52 = 0 → x*y = -24 := by sorry

end NUMINAMATH_CALUDE_factorize_quadratic_xy_value_l2327_232788


namespace NUMINAMATH_CALUDE_sector_max_area_l2327_232797

theorem sector_max_area (R c : ℝ) (h : c > 0) :
  let perimeter := 2 * R + R * (c / R - 2)
  let area := (1 / 2) * R * (c / R - 2) * R
  ∀ R > 0, perimeter = c → area ≤ c^2 / 16 :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_l2327_232797


namespace NUMINAMATH_CALUDE_problem_solution_l2327_232773

theorem problem_solution (x y z : ℝ) (h1 : x + y + z = 25) (h2 : y + z = 14) : x = 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2327_232773


namespace NUMINAMATH_CALUDE_system_solution_conditions_l2327_232786

/-- Given a system of equations, prove the existence of conditions for distinct positive solutions -/
theorem system_solution_conditions (a b : ℝ) :
  ∃ (x y z : ℝ), 
    (x + y + z = a) ∧ 
    (x^2 + y^2 + z^2 = b^2) ∧ 
    (x * y = z^2) ∧ 
    (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ 
    (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z) ∧
    (∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ a = c ∧ b = d) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_conditions_l2327_232786


namespace NUMINAMATH_CALUDE_market_fruit_count_l2327_232708

/-- The number of apples in the market -/
def num_apples : ℕ := 164

/-- The difference between the number of apples and oranges -/
def apple_orange_diff : ℕ := 27

/-- The number of oranges in the market -/
def num_oranges : ℕ := num_apples - apple_orange_diff

/-- The total number of fruits (apples and oranges) in the market -/
def total_fruits : ℕ := num_apples + num_oranges

theorem market_fruit_count : total_fruits = 301 := by
  sorry

end NUMINAMATH_CALUDE_market_fruit_count_l2327_232708


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2327_232745

theorem triangle_perimeter (a : ℕ) (h1 : Odd a) (h2 : 3 < a) (h3 : a < 9) :
  (3 + 6 + a = 14) ∨ (3 + 6 + a = 16) := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2327_232745


namespace NUMINAMATH_CALUDE_job_completion_relationship_l2327_232709

/-- Represents the relationship between number of machines and time to finish a job -/
theorem job_completion_relationship (D : ℝ) : 
  D > 0 → -- D is positive (time can't be negative or zero)
  (15 : ℝ) / 20 = (3 / 4 * D) / D := by
  sorry

#check job_completion_relationship

end NUMINAMATH_CALUDE_job_completion_relationship_l2327_232709


namespace NUMINAMATH_CALUDE_impossible_to_measure_one_liter_l2327_232736

/-- Represents the state of water in the containers -/
structure WaterState where
  jug : ℕ  -- Amount of water in the 4-liter jug
  pot : ℕ  -- Amount of water in the 6-liter pot

/-- Possible operations on the containers -/
inductive Operation
  | FillJug
  | FillPot
  | EmptyJug
  | EmptyPot
  | PourJugToPot
  | PourPotToJug

/-- Applies an operation to a water state -/
def applyOperation (state : WaterState) (op : Operation) : WaterState :=
  match op with
  | Operation.FillJug => { jug := 4, pot := state.pot }
  | Operation.FillPot => { jug := state.jug, pot := 6 }
  | Operation.EmptyJug => { jug := 0, pot := state.pot }
  | Operation.EmptyPot => { jug := state.jug, pot := 0 }
  | Operation.PourJugToPot =>
      let amount := min state.jug (6 - state.pot)
      { jug := state.jug - amount, pot := state.pot + amount }
  | Operation.PourPotToJug =>
      let amount := min state.pot (4 - state.jug)
      { jug := state.jug + amount, pot := state.pot - amount }

/-- Theorem: It's impossible to measure exactly one liter of water -/
theorem impossible_to_measure_one_liter :
  ∀ (initial : WaterState) (ops : List Operation),
    (initial.jug = 0 ∧ initial.pot = 0) →
    let final := ops.foldl applyOperation initial
    (final.jug ≠ 1 ∧ final.pot ≠ 1) :=
  sorry


end NUMINAMATH_CALUDE_impossible_to_measure_one_liter_l2327_232736


namespace NUMINAMATH_CALUDE_no_100_equilateral_division_l2327_232776

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields and conditions for a convex polygon
  -- This is a simplified representation
  is_convex : Bool

/-- An equilateral triangle -/
structure EquilateralTriangle where
  -- Add necessary fields and conditions for an equilateral triangle
  -- This is a simplified representation
  is_equilateral : Bool

/-- A division of a convex polygon into equilateral triangles -/
structure PolygonDivision (P : ConvexPolygon) where
  triangles : List EquilateralTriangle
  is_valid_division : Bool  -- This would ensure the division is valid

/-- Theorem stating that no convex polygon can be divided into 100 different equilateral triangles -/
theorem no_100_equilateral_division (P : ConvexPolygon) :
  ¬∃ (d : PolygonDivision P), d.is_valid_division ∧ d.triangles.length = 100 := by
  sorry

end NUMINAMATH_CALUDE_no_100_equilateral_division_l2327_232776


namespace NUMINAMATH_CALUDE_min_value_of_a_l2327_232796

theorem min_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x > a → x^2 - x - 6 > 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 > 0 ∧ x ≤ a) → 
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2327_232796


namespace NUMINAMATH_CALUDE_car_rental_cost_equality_l2327_232772

/-- The fixed amount Samuel paid for car rental -/
def samuel_fixed_amount : ℝ := 24

/-- The per-kilometer rate for Samuel's rental -/
def samuel_rate : ℝ := 0.16

/-- The fixed amount Carrey paid for car rental -/
def carrey_fixed_amount : ℝ := 20

/-- The per-kilometer rate for Carrey's rental -/
def carrey_rate : ℝ := 0.25

/-- The distance driven by both Samuel and Carrey -/
def distance_driven : ℝ := 44.44444444444444

theorem car_rental_cost_equality :
  samuel_fixed_amount + samuel_rate * distance_driven =
  carrey_fixed_amount + carrey_rate * distance_driven :=
by sorry


end NUMINAMATH_CALUDE_car_rental_cost_equality_l2327_232772

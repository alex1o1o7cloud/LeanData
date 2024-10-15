import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l4138_413853

theorem expression_evaluation (a b : ℤ) (h1 : a = 1) (h2 : b = -2) : 
  -2*a - 2*b^2 + 3*a*b - b^3 = -8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4138_413853


namespace NUMINAMATH_CALUDE_select_representatives_count_l4138_413893

/-- The number of ways to select subject representatives -/
def num_ways_to_select_representatives (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2).choose 2

/-- Theorem stating that selecting 4 students from 5 for specific subject representations results in 60 different ways -/
theorem select_representatives_count :
  num_ways_to_select_representatives 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_select_representatives_count_l4138_413893


namespace NUMINAMATH_CALUDE_coin_toss_outcomes_l4138_413864

/-- The number of possible outcomes when throwing three coins -/
def coin_outcomes : ℕ := 8

/-- The number of coins being thrown -/
def num_coins : ℕ := 3

/-- The number of possible states for each coin (heads or tails) -/
def states_per_coin : ℕ := 2

/-- Theorem stating that the number of possible outcomes when throwing three coins,
    each with two possible states, is equal to 8 -/
theorem coin_toss_outcomes :
  coin_outcomes = states_per_coin ^ num_coins :=
by sorry

end NUMINAMATH_CALUDE_coin_toss_outcomes_l4138_413864


namespace NUMINAMATH_CALUDE_train_speed_l4138_413896

/-- The speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) (h1 : train_length = 357) 
  (h2 : bridge_length = 137) (h3 : time = 42.34285714285714) : 
  (train_length + bridge_length) / time = 11.66666666666667 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l4138_413896


namespace NUMINAMATH_CALUDE_sum_of_two_smallest_numbers_l4138_413890

theorem sum_of_two_smallest_numbers : ∀ (a b c : ℕ), 
  a = 10 ∧ b = 11 ∧ c = 12 → 
  min a (min b c) + min (max a b) (min b c) = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_smallest_numbers_l4138_413890


namespace NUMINAMATH_CALUDE_comic_book_frames_per_page_l4138_413834

/-- Given a comic book with a total number of frames and pages, 
    calculate the number of frames per page. -/
def frames_per_page (total_frames : ℕ) (total_pages : ℕ) : ℕ :=
  total_frames / total_pages

/-- Theorem stating that for a comic book with 143 frames and 13 pages, 
    the number of frames per page is 11. -/
theorem comic_book_frames_per_page :
  frames_per_page 143 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_frames_per_page_l4138_413834


namespace NUMINAMATH_CALUDE_polynomial_fits_data_l4138_413830

def f (x : ℝ) : ℝ := x^3 + 2*x^2 + x + 1

theorem polynomial_fits_data : 
  f 1 = 5 ∧ f 2 = 15 ∧ f 3 = 35 ∧ f 4 = 69 ∧ f 5 = 119 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_fits_data_l4138_413830


namespace NUMINAMATH_CALUDE_ship_ratio_proof_l4138_413870

theorem ship_ratio_proof (total_people : ℕ) (first_ship : ℕ) (ratio : ℚ) : 
  total_people = 847 →
  first_ship = 121 →
  first_ship + first_ship * ratio + first_ship * ratio^2 = total_people →
  ratio = 2 := by
sorry

end NUMINAMATH_CALUDE_ship_ratio_proof_l4138_413870


namespace NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l4138_413827

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcf_seven_eight_factorial :
  Nat.gcd (factorial 7) (factorial 8) = factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l4138_413827


namespace NUMINAMATH_CALUDE_triangle_max_third_side_l4138_413818

theorem triangle_max_third_side (a b : ℝ) (ha : a = 5) (hb : b = 10) :
  ∃ (c : ℕ), c = 14 ∧ 
  (∀ (x : ℕ), x > c → ¬(a + b > x ∧ b + x > a ∧ x + a > b)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_third_side_l4138_413818


namespace NUMINAMATH_CALUDE_tournament_outcomes_l4138_413854

/-- Represents a tournament with n players --/
def Tournament (n : ℕ) := ℕ

/-- The number of possible outcomes in a tournament --/
def possibleOutcomes (t : Tournament n) : ℕ := 2^(n-1)

theorem tournament_outcomes (t : Tournament 6) : 
  possibleOutcomes t = 32 := by
  sorry

#check tournament_outcomes

end NUMINAMATH_CALUDE_tournament_outcomes_l4138_413854


namespace NUMINAMATH_CALUDE_rearranged_triple_divisible_by_27_l4138_413871

/-- Given a natural number, rearranging its digits to get a number
    that is three times the original results in a number divisible by 27. -/
theorem rearranged_triple_divisible_by_27 (n m : ℕ) :
  (∃ (f : ℕ → ℕ), f n = m) →  -- n and m have the same digits (rearranged)
  m = 3 * n →                 -- m is three times n
  27 ∣ m :=                   -- m is divisible by 27
by sorry

end NUMINAMATH_CALUDE_rearranged_triple_divisible_by_27_l4138_413871


namespace NUMINAMATH_CALUDE_parallel_through_common_parallel_l4138_413873

-- Define the types for lines and the parallel relation
variable {Line : Type}
variable (parallel : Line → Line → Prop)

-- State the axiom of parallels
axiom parallel_transitive {x y z : Line} : parallel x z → parallel y z → parallel x y

-- Theorem statement
theorem parallel_through_common_parallel (a b c : Line) :
  parallel a c → parallel b c → parallel a b :=
by sorry

end NUMINAMATH_CALUDE_parallel_through_common_parallel_l4138_413873


namespace NUMINAMATH_CALUDE_expression_simplification_l4138_413820

theorem expression_simplification (a b c : ℚ) 
  (ha : a = 1/3) (hb : b = 1/2) (hc : c = 1) : 
  (2*a^2 - b) - (a^2 - 4*b) - (b + c) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4138_413820


namespace NUMINAMATH_CALUDE_smallest_class_size_l4138_413888

theorem smallest_class_size :
  ∀ n : ℕ,
  (∃ x : ℕ, n = 5 * x + 2 ∧ x > 0) →
  n > 30 →
  n ≥ 32 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_l4138_413888


namespace NUMINAMATH_CALUDE_y_intercept_approx_20_l4138_413849

/-- A straight line in the xy-plane with given slope and point -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Theorem: The y-intercept of the given line is approximately 20 -/
theorem y_intercept_approx_20 (l : Line) 
  (h1 : l.slope = 3.8666666666666667)
  (h2 : l.point = (150, 600)) :
  ∃ ε > 0, |y_intercept l - 20| < ε :=
sorry

end NUMINAMATH_CALUDE_y_intercept_approx_20_l4138_413849


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l4138_413813

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {a b c d : ℝ} : 
  (∀ x y, a*x + b*y + c = 0 ↔ d*x - y = 0) → a/b = -d

/-- The value of m for parallel lines -/
theorem parallel_lines_m_value : 
  (∀ x y, x + 2*y - 1 = 0 ↔ m*x - y = 0) → m = -1/2 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l4138_413813


namespace NUMINAMATH_CALUDE_burger_problem_l4138_413866

theorem burger_problem (total_time : ℕ) (cook_time_per_side : ℕ) (grill_capacity : ℕ) (total_guests : ℕ) :
  total_time = 72 →
  cook_time_per_side = 4 →
  grill_capacity = 5 →
  total_guests = 30 →
  ∃ (burgers_per_half : ℕ),
    burgers_per_half * (total_guests / 2) + (total_guests / 2) = 
      (total_time / (2 * cook_time_per_side)) * grill_capacity ∧
    burgers_per_half = 2 :=
by sorry

end NUMINAMATH_CALUDE_burger_problem_l4138_413866


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l4138_413835

theorem no_positive_integer_solutions :
  ¬ ∃ (x y : ℕ+), x^2 - 2*y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l4138_413835


namespace NUMINAMATH_CALUDE_total_machine_time_for_dolls_and_accessories_l4138_413867

/-- Calculates the total combined machine operation time for manufacturing dolls and accessories -/
def totalMachineTime (numDolls : ℕ) (numAccessoriesPerDoll : ℕ) (dollTime : ℕ) (accessoryTime : ℕ) : ℕ :=
  numDolls * dollTime + numDolls * numAccessoriesPerDoll * accessoryTime

/-- The number of dolls manufactured -/
def dollCount : ℕ := 12000

/-- The number of accessories per doll -/
def accessoriesPerDoll : ℕ := 2 + 3 + 1 + 5

/-- Time taken to manufacture one doll (in seconds) -/
def dollManufactureTime : ℕ := 45

/-- Time taken to manufacture one accessory (in seconds) -/
def accessoryManufactureTime : ℕ := 10

theorem total_machine_time_for_dolls_and_accessories :
  totalMachineTime dollCount accessoriesPerDoll dollManufactureTime accessoryManufactureTime = 1860000 := by
  sorry

end NUMINAMATH_CALUDE_total_machine_time_for_dolls_and_accessories_l4138_413867


namespace NUMINAMATH_CALUDE_sector_max_area_l4138_413881

/-- Given a sector with circumference 20cm, its maximum area is 25cm² -/
theorem sector_max_area :
  ∀ r l : ℝ,
  r > 0 →
  l > 0 →
  l + 2 * r = 20 →
  ∀ A : ℝ,
  A = 1/2 * l * r →
  A ≤ 25 :=
by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l4138_413881


namespace NUMINAMATH_CALUDE_arrangement_plans_count_l4138_413833

/-- The number of ways to arrange teachers into classes -/
def arrangement_count (n m : ℕ) : ℕ :=
  -- n: total number of teachers
  -- m: number of classes
  sorry

/-- Xiao Li must be in class one -/
def xiao_li_in_class_one : Prop :=
  sorry

/-- Each class must have at least one teacher -/
def at_least_one_teacher_per_class : Prop :=
  sorry

/-- The main theorem stating the number of arrangement plans -/
theorem arrangement_plans_count :
  arrangement_count 5 3 = 50 ∧ xiao_li_in_class_one ∧ at_least_one_teacher_per_class :=
sorry

end NUMINAMATH_CALUDE_arrangement_plans_count_l4138_413833


namespace NUMINAMATH_CALUDE_colors_drying_time_l4138_413816

/-- Represents the time in minutes for a laundry load -/
structure LaundryTime where
  washing : ℕ
  drying : ℕ

/-- The total time for all three loads of laundry -/
def total_time : ℕ := 344

/-- The laundry time for the whites -/
def whites : LaundryTime := { washing := 72, drying := 50 }

/-- The laundry time for the darks -/
def darks : LaundryTime := { washing := 58, drying := 65 }

/-- The washing time for the colors -/
def colors_washing : ℕ := 45

/-- The theorem stating that the drying time for colors is 54 minutes -/
theorem colors_drying_time : 
  total_time - (whites.washing + whites.drying + darks.washing + darks.drying + colors_washing) = 54 := by
  sorry

end NUMINAMATH_CALUDE_colors_drying_time_l4138_413816


namespace NUMINAMATH_CALUDE_truck_toll_theorem_l4138_413875

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on each other axle -/
def calculateAxles (totalWheels : ℕ) (frontAxleWheels : ℕ) (otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

/-- Calculates the toll for a truck given the number of axles -/
def calculateToll (axles : ℕ) : ℚ :=
  3.5 + 0.5 * (axles - 2)

theorem truck_toll_theorem :
  let totalWheels : ℕ := 18
  let frontAxleWheels : ℕ := 2
  let otherAxleWheels : ℕ := 4
  let axles : ℕ := calculateAxles totalWheels frontAxleWheels otherAxleWheels
  calculateToll axles = 5 := by
  sorry

end NUMINAMATH_CALUDE_truck_toll_theorem_l4138_413875


namespace NUMINAMATH_CALUDE_correct_scientific_statement_only_mathematical_models_correct_l4138_413848

-- Define the type for scientific statements
inductive ScientificStatement
  | PopulationDensityEstimation
  | PreliminaryExperiment
  | MathematicalModels
  | SpeciesRichness

-- Define a function to check if a statement is correct
def isCorrectStatement (s : ScientificStatement) : Prop :=
  match s with
  | .MathematicalModels => True
  | _ => False

-- Theorem to prove
theorem correct_scientific_statement :
  ∃ (s : ScientificStatement), isCorrectStatement s :=
  sorry

-- Additional theorem to show that only MathematicalModels is correct
theorem only_mathematical_models_correct (s : ScientificStatement) :
  isCorrectStatement s ↔ s = ScientificStatement.MathematicalModels :=
  sorry

end NUMINAMATH_CALUDE_correct_scientific_statement_only_mathematical_models_correct_l4138_413848


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l4138_413885

/-- Represents the tile configuration of a square pattern -/
structure TilePattern where
  black : ℕ
  white : ℕ

/-- Extends a tile pattern by adding two borders of black tiles -/
def extendPattern (p : TilePattern) : TilePattern :=
  let side := Nat.sqrt (p.black + p.white)
  let newBlack := p.black + 4 * side + 4 * (side - 1) + 4
  { black := newBlack, white := p.white }

/-- The ratio of black to white tiles in a pattern -/
def blackWhiteRatio (p : TilePattern) : ℚ :=
  p.black / p.white

theorem extended_pattern_ratio :
  let original := TilePattern.mk 10 26
  let extended := extendPattern original
  blackWhiteRatio extended = 37 / 13 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l4138_413885


namespace NUMINAMATH_CALUDE_product_of_square_roots_l4138_413840

theorem product_of_square_roots : 
  let P : ℝ := Real.sqrt 2025 + Real.sqrt 2024
  let Q : ℝ := -Real.sqrt 2025 - Real.sqrt 2024
  let R : ℝ := Real.sqrt 2025 - Real.sqrt 2024
  let S : ℝ := Real.sqrt 2024 - Real.sqrt 2025
  P * Q * R * S = -1 := by
sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l4138_413840


namespace NUMINAMATH_CALUDE_cubic_expression_value_l4138_413810

theorem cubic_expression_value (α : ℝ) (h1 : α > 0) (h2 : α^2 - 8*α - 5 = 0) :
  α^3 - 7*α^2 - 13*α + 6 = 11 := by
sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l4138_413810


namespace NUMINAMATH_CALUDE_chess_match_probability_l4138_413841

/-- Given a chess match between players A and B, this theorem proves
    the probability of player B not losing, given the probabilities
    of a draw and player B winning. -/
theorem chess_match_probability (draw_prob win_prob : ℝ) :
  draw_prob = (1 : ℝ) / 2 →
  win_prob = (1 : ℝ) / 3 →
  draw_prob + win_prob = (5 : ℝ) / 6 :=
by sorry

end NUMINAMATH_CALUDE_chess_match_probability_l4138_413841


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l4138_413801

theorem smallest_sum_reciprocals (x y : ℕ+) (hxy : x ≠ y) (hsum : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 ∧ (↑a + ↑b : ℕ) = 40 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 10 → (↑c + ↑d : ℕ) ≥ 40 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l4138_413801


namespace NUMINAMATH_CALUDE_message_sending_methods_l4138_413869

/-- The number of friends the student has -/
def num_friends : ℕ := 4

/-- The number of suitable messages in the draft box -/
def num_messages : ℕ := 3

/-- The number of different methods to send messages -/
def num_methods : ℕ := num_messages ^ num_friends

/-- Theorem stating that the number of different methods to send messages is 81 -/
theorem message_sending_methods : num_methods = 81 := by
  sorry

end NUMINAMATH_CALUDE_message_sending_methods_l4138_413869


namespace NUMINAMATH_CALUDE_f_at_5_eq_neg_13_l4138_413898

/-- A polynomial function of degree 7 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

/-- Theorem stating that f(5) = -13 given f(-5) = 17 -/
theorem f_at_5_eq_neg_13 {a b c : ℝ} (h : f a b c (-5) = 17) : f a b c 5 = -13 := by
  sorry

end NUMINAMATH_CALUDE_f_at_5_eq_neg_13_l4138_413898


namespace NUMINAMATH_CALUDE_smallest_invertible_domain_l4138_413880

/-- The function g(x) = (x-3)^2 + 1 -/
def g (x : ℝ) : ℝ := (x - 3)^2 + 1

/-- g is invertible on [c,∞) -/
def invertible_on (c : ℝ) : Prop :=
  ∀ x y, x ≥ c → y ≥ c → g x = g y → x = y

/-- The smallest value of c for which g is invertible on [c,∞) -/
theorem smallest_invertible_domain : 
  (∃ c, invertible_on c ∧ ∀ c', c' < c → ¬invertible_on c') ∧ 
  (∀ c, invertible_on c → c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_invertible_domain_l4138_413880


namespace NUMINAMATH_CALUDE_smallest_c_value_l4138_413886

theorem smallest_c_value (a b c : ℤ) : 
  (b - a = c - b) →  -- arithmetic progression
  (c * c = a * b) →  -- geometric progression
  (∃ (a' b' c' : ℤ), b' - a' = c' - b' ∧ c' * c' = a' * b' ∧ c' < c) →
  c ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_value_l4138_413886


namespace NUMINAMATH_CALUDE_plane_equation_correct_l4138_413831

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- The plane equation we want to prove -/
def targetEquation : PlaneEquation :=
  { A := 15, B := 7, C := 17, D := -26 }

/-- The three given points -/
def p1 : Point3D := { x := 2, y := -3, z := 1 }
def p2 : Point3D := { x := -1, y := 1, z := 2 }
def p3 : Point3D := { x := 4, y := 0, z := -2 }

theorem plane_equation_correct :
  (pointOnPlane p1 targetEquation) ∧
  (pointOnPlane p2 targetEquation) ∧
  (pointOnPlane p3 targetEquation) ∧
  (targetEquation.A > 0) ∧
  (Nat.gcd (Nat.gcd (Int.natAbs targetEquation.A) (Int.natAbs targetEquation.B))
           (Nat.gcd (Int.natAbs targetEquation.C) (Int.natAbs targetEquation.D)) = 1) :=
by sorry


end NUMINAMATH_CALUDE_plane_equation_correct_l4138_413831


namespace NUMINAMATH_CALUDE_apples_left_l4138_413847

/-- The number of bags with 20 apples each -/
def bags_20 : ℕ := 4

/-- The number of apples in each of the first type of bags -/
def apples_per_bag_20 : ℕ := 20

/-- The number of bags with 25 apples each -/
def bags_25 : ℕ := 6

/-- The number of apples in each of the second type of bags -/
def apples_per_bag_25 : ℕ := 25

/-- The number of apples Ella sells -/
def apples_sold : ℕ := 200

/-- The theorem stating that Ella has 30 apples left -/
theorem apples_left : 
  bags_20 * apples_per_bag_20 + bags_25 * apples_per_bag_25 - apples_sold = 30 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_l4138_413847


namespace NUMINAMATH_CALUDE_sqrt_10_irrational_l4138_413852

theorem sqrt_10_irrational : Irrational (Real.sqrt 10) := by sorry

end NUMINAMATH_CALUDE_sqrt_10_irrational_l4138_413852


namespace NUMINAMATH_CALUDE_painting_time_with_break_l4138_413857

/-- The time it takes to paint a room together, including a break -/
theorem painting_time_with_break (doug_rate dave_rate ella_rate : ℝ) 
  (break_time : ℝ) (h1 : doug_rate = 1 / 5) (h2 : dave_rate = 1 / 7) 
  (h3 : ella_rate = 1 / 10) (h4 : break_time = 2) : 
  ∃ t : ℝ, (doug_rate + dave_rate + ella_rate) * (t - break_time) = 1 ∧ t = 132 / 31 := by
  sorry

end NUMINAMATH_CALUDE_painting_time_with_break_l4138_413857


namespace NUMINAMATH_CALUDE_new_cost_is_fifty_l4138_413837

/-- Represents the manufacturing cost and profit scenario for Crazy Eddie's key chains --/
structure KeyChainScenario where
  initialCost : ℝ
  initialProfitRate : ℝ
  newProfitRate : ℝ
  sellingPrice : ℝ

/-- Calculates the new manufacturing cost given a KeyChainScenario --/
def newManufacturingCost (scenario : KeyChainScenario) : ℝ :=
  scenario.sellingPrice * (1 - scenario.newProfitRate)

/-- Theorem stating that under the given conditions, the new manufacturing cost is $50 --/
theorem new_cost_is_fifty :
  ∀ (scenario : KeyChainScenario),
    scenario.initialCost = 70 ∧
    scenario.initialProfitRate = 0.3 ∧
    scenario.newProfitRate = 0.5 ∧
    scenario.sellingPrice = scenario.initialCost / (1 - scenario.initialProfitRate) →
    newManufacturingCost scenario = 50 := by
  sorry


end NUMINAMATH_CALUDE_new_cost_is_fifty_l4138_413837


namespace NUMINAMATH_CALUDE_min_keystrokes_to_243_l4138_413822

-- Define the allowed operations
def add_one (n : ℕ) : ℕ := n + 1
def multiply_two (n : ℕ) : ℕ := n * 2
def multiply_three (n : ℕ) : ℕ := if n % 3 = 0 then n * 3 else n

-- Define a function to represent a sequence of operations
def apply_operations (ops : List (ℕ → ℕ)) (start : ℕ) : ℕ :=
  ops.foldl (λ acc op => op acc) start

-- Define the theorem
theorem min_keystrokes_to_243 :
  ∃ (ops : List (ℕ → ℕ)), 
    (∀ op ∈ ops, op ∈ [add_one, multiply_two, multiply_three]) ∧
    apply_operations ops 1 = 243 ∧
    ops.length = 5 ∧
    (∀ (other_ops : List (ℕ → ℕ)), 
      (∀ op ∈ other_ops, op ∈ [add_one, multiply_two, multiply_three]) →
      apply_operations other_ops 1 = 243 →
      other_ops.length ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_min_keystrokes_to_243_l4138_413822


namespace NUMINAMATH_CALUDE_ian_hourly_rate_l4138_413815

/-- Represents Ian's survey work and earnings -/
structure SurveyWork where
  hours_worked : ℕ
  money_left : ℕ
  spend_ratio : ℚ

/-- Calculates Ian's hourly rate given his survey work details -/
def hourly_rate (work : SurveyWork) : ℚ :=
  (work.money_left / (1 - work.spend_ratio)) / work.hours_worked

/-- Theorem stating that Ian's hourly rate is $18 -/
theorem ian_hourly_rate :
  let work : SurveyWork := {
    hours_worked := 8,
    money_left := 72,
    spend_ratio := 1/2
  }
  hourly_rate work = 18 := by sorry

end NUMINAMATH_CALUDE_ian_hourly_rate_l4138_413815


namespace NUMINAMATH_CALUDE_photo_comparison_l4138_413838

theorem photo_comparison (claire lisa robert : ℕ) 
  (h1 : lisa = 3 * claire)
  (h2 : robert = lisa)
  : robert = 2 * claire + claire := by
  sorry

end NUMINAMATH_CALUDE_photo_comparison_l4138_413838


namespace NUMINAMATH_CALUDE_function_root_implies_a_range_l4138_413877

theorem function_root_implies_a_range (a : ℝ) : 
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ a * x + 1 = 0) → (a < -1 ∨ a > 1) := by
  sorry

end NUMINAMATH_CALUDE_function_root_implies_a_range_l4138_413877


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l4138_413843

/-- The decimal representation of the repeating decimal 0.565656... -/
def repeating_decimal : ℚ := 56 / 99

/-- The reciprocal of the repeating decimal 0.565656... -/
def reciprocal : ℚ := 99 / 56

/-- Theorem: The reciprocal of the common fraction form of 0.565656... is 99/56 -/
theorem reciprocal_of_repeating_decimal : (repeating_decimal)⁻¹ = reciprocal := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l4138_413843


namespace NUMINAMATH_CALUDE_divisibility_condition_l4138_413800

theorem divisibility_condition (n p : ℕ+) (h_prime : Nat.Prime p) (h_bound : n ≤ 2 * p) :
  (((p : ℤ) - 1) ^ (n : ℕ) + 1) % (n ^ (p - 1 : ℕ)) = 0 ↔ 
  ((n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l4138_413800


namespace NUMINAMATH_CALUDE_part1_part2_l4138_413832

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x, f a x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1 := by sorry

-- Part 2
theorem part2 (m : ℝ) :
  (∃ n : ℝ, |2 * n - 1| + 1 ≤ m - (|2 * (-n) - 1| + 1)) → m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l4138_413832


namespace NUMINAMATH_CALUDE_exp_25pi_i_div_2_equals_i_l4138_413803

theorem exp_25pi_i_div_2_equals_i :
  Complex.exp (25 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_exp_25pi_i_div_2_equals_i_l4138_413803


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l4138_413879

-- Define the new operation ※
def star_op (a b : ℝ) : ℝ := a * b - a + b - 2

-- Theorem statement
theorem unique_positive_integer_solution :
  ∃! (x : ℕ), x > 0 ∧ star_op 3 (x : ℝ) < 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l4138_413879


namespace NUMINAMATH_CALUDE_tax_savings_proof_l4138_413824

def original_tax_rate : ℚ := 40 / 100
def new_tax_rate : ℚ := 33 / 100
def annual_income : ℚ := 45000

def differential_savings : ℚ := original_tax_rate * annual_income - new_tax_rate * annual_income

theorem tax_savings_proof : differential_savings = 3150 := by
  sorry

end NUMINAMATH_CALUDE_tax_savings_proof_l4138_413824


namespace NUMINAMATH_CALUDE_smallest_integer_y_five_satisfies_inequality_smallest_integer_is_five_l4138_413874

theorem smallest_integer_y (y : ℤ) : (7 + 3 * y < 25) ↔ y ≤ 5 := by sorry

theorem five_satisfies_inequality : 7 + 3 * 5 < 25 := by sorry

theorem smallest_integer_is_five : ∃ (y : ℤ), y = 5 ∧ (7 + 3 * y < 25) ∧ ∀ (z : ℤ), (7 + 3 * z < 25) → z ≥ y := by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_five_satisfies_inequality_smallest_integer_is_five_l4138_413874


namespace NUMINAMATH_CALUDE_football_players_count_l4138_413805

theorem football_players_count (cricket_players hockey_players softball_players total_players : ℕ) 
  (h1 : cricket_players = 22)
  (h2 : hockey_players = 15)
  (h3 : softball_players = 19)
  (h4 : total_players = 77) :
  total_players - (cricket_players + hockey_players + softball_players) = 21 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l4138_413805


namespace NUMINAMATH_CALUDE_rug_coverage_area_l4138_413850

/-- Given three rugs with specified overlap areas, calculate the total floor area covered. -/
theorem rug_coverage_area (total_rug_area : ℝ) (double_layer_area : ℝ) (triple_layer_area : ℝ)
  (h1 : total_rug_area = 200)
  (h2 : double_layer_area = 24)
  (h3 : triple_layer_area = 19) :
  total_rug_area - double_layer_area - 2 * triple_layer_area = 138 := by
  sorry

#check rug_coverage_area

end NUMINAMATH_CALUDE_rug_coverage_area_l4138_413850


namespace NUMINAMATH_CALUDE_calories_left_for_dinner_l4138_413894

def daily_calorie_limit : ℕ := 2200
def breakfast_calories : ℕ := 353
def lunch_calories : ℕ := 885
def snack_calories : ℕ := 130

theorem calories_left_for_dinner :
  daily_calorie_limit - (breakfast_calories + lunch_calories + snack_calories) = 832 := by
  sorry

end NUMINAMATH_CALUDE_calories_left_for_dinner_l4138_413894


namespace NUMINAMATH_CALUDE_calculate_expression_l4138_413861

theorem calculate_expression : -2^4 + 3 * (-1)^2010 - (-2)^2 = -17 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4138_413861


namespace NUMINAMATH_CALUDE_color_assignment_l4138_413817

-- Define the colors
inductive Color
| White
| Red
| Blue

-- Define the friends
inductive Friend
| Tamara
| Valya
| Lida

-- Define a function to assign colors to dresses
def dress : Friend → Color := sorry

-- Define a function to assign colors to shoes
def shoes : Friend → Color := sorry

-- Define the theorem
theorem color_assignment :
  -- Tamara's dress and shoes match
  (dress Friend.Tamara = shoes Friend.Tamara) ∧
  -- Valya wore white shoes
  (shoes Friend.Valya = Color.White) ∧
  -- Neither Lida's dress nor her shoes were red
  (dress Friend.Lida ≠ Color.Red ∧ shoes Friend.Lida ≠ Color.Red) ∧
  -- All friends have different dress colors
  (dress Friend.Tamara ≠ dress Friend.Valya ∧
   dress Friend.Tamara ≠ dress Friend.Lida ∧
   dress Friend.Valya ≠ dress Friend.Lida) ∧
  -- All friends have different shoe colors
  (shoes Friend.Tamara ≠ shoes Friend.Valya ∧
   shoes Friend.Tamara ≠ shoes Friend.Lida ∧
   shoes Friend.Valya ≠ shoes Friend.Lida) →
  -- The only valid assignment is:
  (dress Friend.Tamara = Color.Red ∧ shoes Friend.Tamara = Color.Red) ∧
  (dress Friend.Valya = Color.Blue ∧ shoes Friend.Valya = Color.White) ∧
  (dress Friend.Lida = Color.White ∧ shoes Friend.Lida = Color.Blue) :=
by
  sorry

end NUMINAMATH_CALUDE_color_assignment_l4138_413817


namespace NUMINAMATH_CALUDE_julia_played_with_16_kids_l4138_413811

def kids_on_tuesday : ℕ := 4

def kids_difference : ℕ := 12

def kids_on_monday : ℕ := kids_on_tuesday + kids_difference

theorem julia_played_with_16_kids : kids_on_monday = 16 := by
  sorry

end NUMINAMATH_CALUDE_julia_played_with_16_kids_l4138_413811


namespace NUMINAMATH_CALUDE_square_area_error_l4138_413872

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l4138_413872


namespace NUMINAMATH_CALUDE_hyperbola_condition_l4138_413846

/-- A hyperbola is represented by an equation of the form ax²/p + by²/q = 1, 
    where a and b are non-zero real numbers with opposite signs, 
    and p and q are non-zero real numbers. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  p : ℝ
  q : ℝ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  p_nonzero : p ≠ 0
  q_nonzero : q ≠ 0
  opposite_signs : a * b < 0

/-- The equation x²/(k-1) + y²/(k+1) = 1 -/
def equation (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 1) + y^2 / (k + 1) = 1

/-- The condition -1 < k < 1 -/
def condition (k : ℝ) : Prop :=
  -1 < k ∧ k < 1

/-- Theorem stating that the condition is necessary and sufficient 
    for the equation to represent a hyperbola -/
theorem hyperbola_condition (k : ℝ) : 
  (∃ h : Hyperbola, equation k ↔ h.a * x^2 / h.p + h.b * y^2 / h.q = 1) ↔ condition k :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l4138_413846


namespace NUMINAMATH_CALUDE_base_conversion_equality_l4138_413851

theorem base_conversion_equality (b : ℕ) : b > 0 → (
  4 * 5 + 3 = 1 * b^2 + 2 * b + 1 ↔ b = 4
) := by sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l4138_413851


namespace NUMINAMATH_CALUDE_anne_distance_l4138_413812

/-- Given a speed and time, calculates the distance traveled -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Anne's distance traveled is 6 miles -/
theorem anne_distance :
  let speed : ℝ := 2  -- miles per hour
  let time : ℝ := 3   -- hours
  distance speed time = 6 := by sorry

end NUMINAMATH_CALUDE_anne_distance_l4138_413812


namespace NUMINAMATH_CALUDE_james_ownership_l4138_413884

theorem james_ownership (total : ℕ) (difference : ℕ) (james : ℕ) (ali : ℕ) :
  total = 250 →
  difference = 40 →
  james = ali + difference →
  total = james + ali →
  james = 145 := by
sorry

end NUMINAMATH_CALUDE_james_ownership_l4138_413884


namespace NUMINAMATH_CALUDE_floor_neg_seven_fourths_l4138_413889

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_neg_seven_fourths_l4138_413889


namespace NUMINAMATH_CALUDE_population_change_l4138_413828

/-- Theorem: Given an initial population that increases by 30% in the first year
    and then decreases by x% in the second year, if the initial population is 15000
    and the final population is 13650, then x = 30. -/
theorem population_change (x : ℝ) : 
  let initial_population : ℝ := 15000
  let first_year_increase : ℝ := 0.3
  let final_population : ℝ := 13650
  let population_after_first_year : ℝ := initial_population * (1 + first_year_increase)
  let population_after_second_year : ℝ := population_after_first_year * (1 - x / 100)
  population_after_second_year = final_population → x = 30 := by
sorry

end NUMINAMATH_CALUDE_population_change_l4138_413828


namespace NUMINAMATH_CALUDE_intersecting_lines_theorem_l4138_413842

/-- Given two lines l₁ and l₂ that intersect at point P, and a third line l₃ -/
structure IntersectingLines where
  /-- The equation of line l₁ is 3x + 4y - 2 = 0 -/
  l₁ : ℝ → ℝ → Prop
  l₁_eq : ∀ x y, l₁ x y ↔ 3 * x + 4 * y - 2 = 0

  /-- The equation of line l₂ is 2x + y + 2 = 0 -/
  l₂ : ℝ → ℝ → Prop
  l₂_eq : ∀ x y, l₂ x y ↔ 2 * x + y + 2 = 0

  /-- P is the intersection point of l₁ and l₂ -/
  P : ℝ × ℝ
  P_on_l₁ : l₁ P.1 P.2
  P_on_l₂ : l₂ P.1 P.2

  /-- The equation of line l₃ is x - 2y - 1 = 0 -/
  l₃ : ℝ → ℝ → Prop
  l₃_eq : ∀ x y, l₃ x y ↔ x - 2 * y - 1 = 0

/-- The main theorem stating the equations of the two required lines -/
theorem intersecting_lines_theorem (g : IntersectingLines) :
  (∀ x y, x + y = 0 ↔ (∃ t : ℝ, x = t * g.P.1 ∧ y = t * g.P.2)) ∧
  (∀ x y, 2 * x + y + 2 = 0 ↔ (g.l₃ x y → (x - g.P.1) * 1 + (y - g.P.2) * 2 = 0)) :=
sorry

end NUMINAMATH_CALUDE_intersecting_lines_theorem_l4138_413842


namespace NUMINAMATH_CALUDE_exists_points_with_midpoint_l4138_413829

/-- Definition of the hyperbola -/
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2/9 = 1

/-- Definition of midpoint -/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2

/-- Theorem statement -/
theorem exists_points_with_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_on_hyperbola x₁ y₁ ∧
    is_on_hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ :=
by sorry

end NUMINAMATH_CALUDE_exists_points_with_midpoint_l4138_413829


namespace NUMINAMATH_CALUDE_sin_150_cos_30_plus_cos_150_sin_30_eq_zero_l4138_413883

theorem sin_150_cos_30_plus_cos_150_sin_30_eq_zero : 
  Real.sin (150 * π / 180) * Real.cos (30 * π / 180) + 
  Real.cos (150 * π / 180) * Real.sin (30 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_cos_30_plus_cos_150_sin_30_eq_zero_l4138_413883


namespace NUMINAMATH_CALUDE_subscription_difference_is_4000_l4138_413856

/-- Represents the subscription amounts and profit distribution in a business venture. -/
structure BusinessVenture where
  total_subscription : ℕ
  total_profit : ℕ
  c_profit : ℕ
  b_extra : ℕ

/-- Calculates the difference between A's and B's subscriptions. -/
def subscription_difference (bv : BusinessVenture) : ℕ :=
  let c_subscription := bv.c_profit * bv.total_subscription / bv.total_profit
  let b_subscription := c_subscription + bv.b_extra
  let a_subscription := bv.total_subscription - b_subscription - c_subscription
  a_subscription - b_subscription

/-- Theorem stating that the difference between A's and B's subscriptions is 4000. -/
theorem subscription_difference_is_4000 :
  subscription_difference ⟨50000, 35000, 8400, 5000⟩ = 4000 := by
  sorry


end NUMINAMATH_CALUDE_subscription_difference_is_4000_l4138_413856


namespace NUMINAMATH_CALUDE_football_lineup_count_l4138_413899

/-- The number of different lineups that can be created from a football team --/
def number_of_lineups (total_players : ℕ) (skilled_players : ℕ) : ℕ :=
  skilled_players * (total_players - 1) * (total_players - 2) * (total_players - 3) * (total_players - 4)

/-- Theorem stating that the number of lineups for a team of 15 players with 5 skilled players is 109200 --/
theorem football_lineup_count :
  number_of_lineups 15 5 = 109200 := by
  sorry

end NUMINAMATH_CALUDE_football_lineup_count_l4138_413899


namespace NUMINAMATH_CALUDE_ammonia_formation_l4138_413855

/-- Represents the chemical reaction between Potassium hydroxide and Ammonium iodide -/
structure ChemicalReaction where
  koh : ℝ  -- moles of Potassium hydroxide
  nh4i : ℝ  -- moles of Ammonium iodide
  nh3 : ℝ  -- moles of Ammonia formed

/-- Theorem stating that the moles of Ammonia formed equals the moles of Ammonium iodide used -/
theorem ammonia_formation (reaction : ChemicalReaction) 
  (h1 : reaction.nh4i = 3)  -- 3 moles of Ammonium iodide are used
  (h2 : reaction.nh3 = 3)   -- The total moles of Ammonia formed is 3
  : reaction.nh3 = reaction.nh4i := by
  sorry


end NUMINAMATH_CALUDE_ammonia_formation_l4138_413855


namespace NUMINAMATH_CALUDE_roller_derby_team_size_l4138_413859

theorem roller_derby_team_size :
  ∀ (num_teams : ℕ) (skates_per_member : ℕ) (laces_per_skate : ℕ) (total_laces : ℕ),
    num_teams = 4 →
    skates_per_member = 2 →
    laces_per_skate = 3 →
    total_laces = 240 →
    ∃ (members_per_team : ℕ),
      members_per_team * num_teams * skates_per_member * laces_per_skate = total_laces ∧
      members_per_team = 10 :=
by sorry

end NUMINAMATH_CALUDE_roller_derby_team_size_l4138_413859


namespace NUMINAMATH_CALUDE_gcd_7_factorial_8_factorial_l4138_413897

theorem gcd_7_factorial_8_factorial : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7_factorial_8_factorial_l4138_413897


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4138_413868

theorem inequality_system_solution :
  ∀ x : ℝ, (3 * x + 1 ≥ 7 ∧ 4 * x - 3 < 9) ↔ (2 ≤ x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4138_413868


namespace NUMINAMATH_CALUDE_ratio_determination_l4138_413806

/-- Given constants a and b, and unknowns x and y, the equation
    ax³ + bx²y + bxy² + ay³ = 0 can be transformed into a polynomial
    equation in terms of t, where t = x/y. -/
theorem ratio_determination (a b x y : ℝ) :
  ∃ t, t = x / y ∧ a * t^3 + b * t^2 + b * t + a = 0 :=
by sorry

end NUMINAMATH_CALUDE_ratio_determination_l4138_413806


namespace NUMINAMATH_CALUDE_circle_to_square_impossible_l4138_413882

/-- Represents a piece of paper with a boundary --/
structure PaperPiece where
  boundary : Set ℝ × ℝ

/-- Represents a cut on a paper piece --/
inductive Cut
  | StraightLine : (ℝ × ℝ) → (ℝ × ℝ) → Cut
  | CircularArc : (ℝ × ℝ) → ℝ → ℝ → ℝ → Cut

/-- Represents a transformation of paper pieces --/
def Transform := List PaperPiece → List PaperPiece

/-- Checks if a shape is a circle --/
def is_circle (p : PaperPiece) : Prop := sorry

/-- Checks if a shape is a square --/
def is_square (p : PaperPiece) : Prop := sorry

/-- Calculates the area of a paper piece --/
def area (p : PaperPiece) : ℝ := sorry

/-- Theorem stating the impossibility of transforming a circle to a square of equal area --/
theorem circle_to_square_impossible 
  (initial : PaperPiece) 
  (cuts : List Cut) 
  (transform : Transform) :
  is_circle initial →
  (∃ final, is_square final ∧ area final = area initial ∧ 
    transform [initial] = final :: (transform [initial]).tail) →
  False := by
  sorry

#check circle_to_square_impossible

end NUMINAMATH_CALUDE_circle_to_square_impossible_l4138_413882


namespace NUMINAMATH_CALUDE_rectangle_area_l4138_413876

/-- Given a rectangle with perimeter 80 meters and length three times the width, 
    prove that its area is 300 square meters. -/
theorem rectangle_area (l w : ℝ) 
  (perimeter_eq : 2 * l + 2 * w = 80)
  (length_width_relation : l = 3 * w) : 
  l * w = 300 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l4138_413876


namespace NUMINAMATH_CALUDE_additional_cartons_needed_l4138_413878

/-- Given the total required cartons, the number of strawberry cartons, and the number of blueberry cartons,
    prove that the additional cartons needed is equal to the total required minus the sum of strawberry and blueberry cartons. -/
theorem additional_cartons_needed
  (total_required : ℕ)
  (strawberry_cartons : ℕ)
  (blueberry_cartons : ℕ)
  (h : total_required = 42 ∧ strawberry_cartons = 2 ∧ blueberry_cartons = 7) :
  total_required - (strawberry_cartons + blueberry_cartons) = 33 :=
by sorry

end NUMINAMATH_CALUDE_additional_cartons_needed_l4138_413878


namespace NUMINAMATH_CALUDE_investor_profit_l4138_413809

def total_investment : ℝ := 1900
def investment_fund1 : ℝ := 1700
def profit_rate_fund1 : ℝ := 0.09
def profit_rate_fund2 : ℝ := 0.02

def investment_fund2 : ℝ := total_investment - investment_fund1

def profit_fund1 : ℝ := investment_fund1 * profit_rate_fund1
def profit_fund2 : ℝ := investment_fund2 * profit_rate_fund2

def total_profit : ℝ := profit_fund1 + profit_fund2

theorem investor_profit : total_profit = 157 := by
  sorry

end NUMINAMATH_CALUDE_investor_profit_l4138_413809


namespace NUMINAMATH_CALUDE_cubic_function_properties_l4138_413844

/-- A cubic function with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + b

/-- The derivative of f -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem cubic_function_properties (a b : ℝ) (h_a : a ≠ 0) :
  (f' a 2 = 0 ∧ f a b 2 = 8) →
  (a = 4 ∧ b = 24) ∧
  (∀ x, x < -2 → (f' a x > 0)) ∧
  (∀ x, x > 2 → (f' a x > 0)) ∧
  (∀ x, -2 < x ∧ x < 2 → (f' a x < 0)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - (-2)| ∧ |x - (-2)| < δ → f a b x < f a b (-2)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → f a b x > f a b 2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l4138_413844


namespace NUMINAMATH_CALUDE_log_product_range_l4138_413825

theorem log_product_range : ∃ y : ℝ,
  y = Real.log 6 / Real.log 5 * Real.log 7 / Real.log 6 * Real.log 8 / Real.log 7 * Real.log 9 / Real.log 8 * Real.log 10 / Real.log 9 ∧
  1 < y ∧ y < 2 := by
  sorry

end NUMINAMATH_CALUDE_log_product_range_l4138_413825


namespace NUMINAMATH_CALUDE_max_value_complex_l4138_413819

theorem max_value_complex (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 + 3*z + Complex.I*2) ≤ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_complex_l4138_413819


namespace NUMINAMATH_CALUDE_expression_value_l4138_413808

theorem expression_value (a b : ℝ) (h : a * 1 + b * 2 = 3) : 2 * a + 4 * b - 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4138_413808


namespace NUMINAMATH_CALUDE_base_10_satisfies_equation_l4138_413804

def base_x_addition (x : ℕ) (a b c : ℕ) : Prop :=
  a + b = c

def to_base_10 (x : ℕ) (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 * x^3 + d2 * x^2 + d3 * x + d4

theorem base_10_satisfies_equation : 
  ∃ x : ℕ, x > 1 ∧ base_x_addition x 
    (to_base_10 x 8374) 
    (to_base_10 x 6250) 
    (to_base_10 x 15024) :=
by
  sorry

end NUMINAMATH_CALUDE_base_10_satisfies_equation_l4138_413804


namespace NUMINAMATH_CALUDE_unique_brigade_solution_l4138_413839

/-- Represents a brigade with newspapers and members -/
structure Brigade where
  newspapers : ℕ
  members : ℕ

/-- Properties of a valid brigade -/
def is_valid_brigade (b : Brigade) : Prop :=
  ∀ (m : ℕ) (n : ℕ), m ≤ b.members → n ≤ b.newspapers →
    (∃! (c : ℕ), c = 2) ∧  -- Each member reads exactly 2 newspapers
    (∃! (d : ℕ), d = 5) ∧  -- Each newspaper is read by exactly 5 members
    (∃! (e : ℕ), e = 1)    -- Each combination of 2 newspapers is read by exactly 1 member

/-- Theorem stating the unique solution for a valid brigade -/
theorem unique_brigade_solution (b : Brigade) (h : is_valid_brigade b) :
  b.newspapers = 6 ∧ b.members = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_brigade_solution_l4138_413839


namespace NUMINAMATH_CALUDE_z_coordinate_for_x_7_l4138_413891

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- Given a line and an x-coordinate, find the corresponding z-coordinate -/
def find_z_coordinate (line : Line3D) (x : ℝ) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem z_coordinate_for_x_7 :
  let line := Line3D.mk (1, 3, 2) (4, 4, -1)
  find_z_coordinate line 7 = -4 := by sorry

end NUMINAMATH_CALUDE_z_coordinate_for_x_7_l4138_413891


namespace NUMINAMATH_CALUDE_orange_apple_cost_difference_l4138_413865

/-- The cost difference between an orange and an apple -/
def cost_difference (apple_cost orange_cost : ℚ) : ℚ := orange_cost - apple_cost

theorem orange_apple_cost_difference 
  (apple_cost orange_cost : ℚ) 
  (total_cost : ℚ)
  (h1 : apple_cost > 0)
  (h2 : orange_cost > apple_cost)
  (h3 : 3 * apple_cost + 7 * orange_cost = total_cost)
  (h4 : total_cost = 456/100) : 
  ∃ (diff : ℚ), cost_difference apple_cost orange_cost = diff ∧ diff > 0 := by
  sorry

#eval cost_difference (26/100) (36/100)

end NUMINAMATH_CALUDE_orange_apple_cost_difference_l4138_413865


namespace NUMINAMATH_CALUDE_triangle_ad_length_l4138_413887

/-- Triangle ABC with perpendicular from A to BC at point D -/
structure Triangle :=
  (A B C D : ℝ × ℝ)
  (AB : ℝ)
  (AC : ℝ)
  (BD : ℝ)
  (CD : ℝ)
  (AD : ℝ)
  (is_right_angle : (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0)
  (AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = AB)
  (AC_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = AC)
  (BD_CD_ratio : BD / CD = 2 / 5)

/-- Theorem: In triangle ABC, if AB = 10, AC = 17, D is the foot of the perpendicular from A to BC,
    and BD:CD = 2:5, then AD = 8 -/
theorem triangle_ad_length (t : Triangle) (h1 : t.AB = 10) (h2 : t.AC = 17) : t.AD = 8 := by
  sorry


end NUMINAMATH_CALUDE_triangle_ad_length_l4138_413887


namespace NUMINAMATH_CALUDE_swimming_pool_width_l4138_413802

/-- Represents the dimensions and area of a rectangular swimming pool -/
structure SwimmingPool where
  length : ℝ
  width : ℝ
  area : ℝ

/-- Theorem: Given a rectangular swimming pool with area 143.2 m² and length 4 m, its width is 35.8 m -/
theorem swimming_pool_width (pool : SwimmingPool) 
  (h_area : pool.area = 143.2)
  (h_length : pool.length = 4)
  (h_rectangle : pool.area = pool.length * pool.width) : 
  pool.width = 35.8 := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_width_l4138_413802


namespace NUMINAMATH_CALUDE_distance_origin_to_line_through_focus_l4138_413895

/-- Parabola type representing y^2 = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line type -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Distance from a point to a line -/
def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Theorem: Distance from origin to line through focus of parabola -/
theorem distance_origin_to_line_through_focus 
  (C : Parabola) 
  (l : Line) 
  (A B : ℝ × ℝ) :
  C.equation = (fun x y => y^2 = 8*x) →
  C.focus = (2, 0) →
  (∃ (x y : ℝ), l.equation x y ∧ C.equation x y) →
  (∃ (x y : ℝ), l.equation 2 0) →
  C.equation A.1 A.2 →
  C.equation B.1 B.2 →
  l.equation A.1 A.2 →
  l.equation B.1 B.2 →
  distance A B = 10 →
  distancePointToLine (0, 0) l = 4 * Real.sqrt 5 / 5 := by
    sorry

end NUMINAMATH_CALUDE_distance_origin_to_line_through_focus_l4138_413895


namespace NUMINAMATH_CALUDE_snooker_tournament_revenue_l4138_413858

theorem snooker_tournament_revenue
  (total_tickets : ℕ)
  (vip_price general_price : ℚ)
  (fewer_vip : ℕ) :
  total_tickets = 320 →
  vip_price = 40 →
  general_price = 15 →
  fewer_vip = 212 →
  ∃ (vip_tickets general_tickets : ℕ),
    vip_tickets + general_tickets = total_tickets ∧
    vip_tickets = general_tickets - fewer_vip ∧
    vip_price * vip_tickets + general_price * general_tickets = 6150 :=
by sorry

end NUMINAMATH_CALUDE_snooker_tournament_revenue_l4138_413858


namespace NUMINAMATH_CALUDE_B_power_100_l4138_413845

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_100 : B^100 = B := by sorry

end NUMINAMATH_CALUDE_B_power_100_l4138_413845


namespace NUMINAMATH_CALUDE_conditional_probability_l4138_413826

/-- Represents the probability space for the household appliance problem -/
structure ApplianceProbability where
  /-- Probability that the appliance lasts for 3 years -/
  three_years : ℝ
  /-- Probability that the appliance lasts for 4 years -/
  four_years : ℝ
  /-- Assumption that the probability of lasting 3 years is 0.8 -/
  three_years_prob : three_years = 0.8
  /-- Assumption that the probability of lasting 4 years is 0.4 -/
  four_years_prob : four_years = 0.4
  /-- Assumption that probabilities are between 0 and 1 -/
  prob_bounds : 0 ≤ three_years ∧ three_years ≤ 1 ∧ 0 ≤ four_years ∧ four_years ≤ 1

/-- The main theorem stating the conditional probability -/
theorem conditional_probability (ap : ApplianceProbability) :
  (ap.four_years / ap.three_years) = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_conditional_probability_l4138_413826


namespace NUMINAMATH_CALUDE_B_power_150_l4138_413836

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150 : B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by sorry

end NUMINAMATH_CALUDE_B_power_150_l4138_413836


namespace NUMINAMATH_CALUDE_breakfast_cost_is_correct_l4138_413807

/-- Calculates the total cost of breakfast for Francis and Kiera -/
def breakfast_cost : ℝ :=
  let muffin_price : ℝ := 2
  let fruit_cup_price : ℝ := 3
  let coffee_price : ℝ := 1.5
  let discount_rate : ℝ := 0.1
  
  let francis_muffins : ℕ := 2
  let francis_fruit_cups : ℕ := 2
  let francis_coffee : ℕ := 1
  
  let kiera_muffins : ℕ := 2
  let kiera_fruit_cups : ℕ := 1
  let kiera_coffee : ℕ := 2
  
  let francis_cost : ℝ := 
    muffin_price * francis_muffins + 
    fruit_cup_price * francis_fruit_cups + 
    coffee_price * francis_coffee
  
  let kiera_cost_before_discount : ℝ := 
    muffin_price * kiera_muffins + 
    fruit_cup_price * kiera_fruit_cups + 
    coffee_price * kiera_coffee
  
  let discount_amount : ℝ := 
    discount_rate * (muffin_price * 2 + fruit_cup_price)
  
  let kiera_cost : ℝ := kiera_cost_before_discount - discount_amount
  
  francis_cost + kiera_cost

theorem breakfast_cost_is_correct : breakfast_cost = 20.8 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_cost_is_correct_l4138_413807


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l4138_413814

/-- In a right triangle LMN, given cos M and the length of LM, we can determine the length of LN. -/
theorem right_triangle_side_length 
  (L M N : ℝ × ℝ) 
  (right_angle_M : (N.1 - M.1) * (L.2 - M.2) = (L.1 - M.1) * (N.2 - M.2)) 
  (cos_M : Real.cos (Real.arctan ((L.2 - M.2) / (L.1 - M.1))) = 3/5) 
  (LM_length : Real.sqrt ((L.1 - M.1)^2 + (L.2 - M.2)^2) = 15) :
  Real.sqrt ((L.1 - N.1)^2 + (L.2 - N.2)^2) = 9 := by
    sorry


end NUMINAMATH_CALUDE_right_triangle_side_length_l4138_413814


namespace NUMINAMATH_CALUDE_fibonacci_seventh_term_l4138_413892

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_seventh_term : fibonacci 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_seventh_term_l4138_413892


namespace NUMINAMATH_CALUDE_cindy_marbles_l4138_413860

theorem cindy_marbles (initial_marbles : ℕ) (friends : ℕ) (remaining_multiplier : ℕ) (remaining_total : ℕ) :
  initial_marbles = 500 →
  friends = 4 →
  remaining_multiplier = 4 →
  remaining_total = 720 →
  remaining_multiplier * (initial_marbles - friends * (initial_marbles - (remaining_total / remaining_multiplier))) = remaining_total →
  initial_marbles - (remaining_total / remaining_multiplier) = friends * 80 :=
by sorry

end NUMINAMATH_CALUDE_cindy_marbles_l4138_413860


namespace NUMINAMATH_CALUDE_probability_perfect_square_sum_l4138_413823

def roll_outcomes : ℕ := 64

def favorable_outcomes : ℕ := 12

theorem probability_perfect_square_sum (roll_outcomes : ℕ) (favorable_outcomes : ℕ) :
  (favorable_outcomes : ℚ) / (roll_outcomes : ℚ) = 3 / 16 :=
by sorry

end NUMINAMATH_CALUDE_probability_perfect_square_sum_l4138_413823


namespace NUMINAMATH_CALUDE_koolaid_percentage_is_four_percent_l4138_413862

/-- Calculates the percentage of koolaid powder in a mixture --/
def koolaid_percentage (initial_powder : ℚ) (initial_water : ℚ) (evaporated_water : ℚ) : ℚ :=
  let remaining_water := initial_water - evaporated_water
  let final_water := 4 * remaining_water
  let total_volume := final_water + initial_powder
  (initial_powder / total_volume) * 100

/-- Theorem stating that the percentage of koolaid powder is 4% given the initial conditions --/
theorem koolaid_percentage_is_four_percent :
  koolaid_percentage 2 16 4 = 4 := by sorry

end NUMINAMATH_CALUDE_koolaid_percentage_is_four_percent_l4138_413862


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_3_intersection_empty_iff_a_less_than_1_l4138_413821

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

-- Define the universal set U (assuming it's the real numbers)
def U : Set ℝ := Set.univ

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_3 :
  (A 3 ∩ B = {x | (-1 ≤ x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x ≤ 5)}) ∧
  (A 3 ∪ (U \ B) = {x | -1 ≤ x ∧ x ≤ 5}) := by sorry

-- Theorem for part (2)
theorem intersection_empty_iff_a_less_than_1 :
  ∀ a : ℝ, (A a ∩ B = ∅) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_3_intersection_empty_iff_a_less_than_1_l4138_413821


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l4138_413863

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (7 : ℝ)^3 + (9 : ℝ)^3 - 100 → 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |x^(1/3) - n| ≤ |x^(1/3) - m| :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l4138_413863

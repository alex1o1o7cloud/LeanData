import Mathlib

namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l4124_412452

/-- A point P(x, y) is in the second quadrant if and only if x < 0 and y > 0 -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point P is a - 2 -/
def x_coordinate (a : ℝ) : ℝ := a - 2

/-- The y-coordinate of point P is 2 -/
def y_coordinate : ℝ := 2

/-- Theorem: For a point P(a-2, 2) to be in the second quadrant, a must be less than 2 -/
theorem point_in_second_quadrant (a : ℝ) : 
  second_quadrant (x_coordinate a) y_coordinate ↔ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l4124_412452


namespace NUMINAMATH_CALUDE_paperware_cost_relationship_l4124_412430

/-- Represents the cost of paper plates and cups -/
structure PaperwareCost where
  plate : ℝ
  cup : ℝ

/-- The total cost of a given number of plates and cups -/
def total_cost (c : PaperwareCost) (plates : ℝ) (cups : ℝ) : ℝ :=
  c.plate * plates + c.cup * cups

/-- Theorem stating the relationship between the costs of different quantities of plates and cups -/
theorem paperware_cost_relationship (c : PaperwareCost) :
  total_cost c 20 40 = 1.20 → total_cost c 100 200 = 6.00 := by
  sorry

end NUMINAMATH_CALUDE_paperware_cost_relationship_l4124_412430


namespace NUMINAMATH_CALUDE_edward_spent_sixteen_l4124_412477

/-- The amount of money Edward spent -/
def amount_spent (initial_amount remaining_amount : ℕ) : ℕ :=
  initial_amount - remaining_amount

/-- Theorem: Edward spent $16 -/
theorem edward_spent_sixteen :
  amount_spent 18 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_edward_spent_sixteen_l4124_412477


namespace NUMINAMATH_CALUDE_inverse_proportion_points_l4124_412407

/-- Given that (2,3) lies on the graph of y = k/x (k ≠ 0), prove that (1,6) also lies on the same graph. -/
theorem inverse_proportion_points : ∀ k : ℝ, k ≠ 0 → (3 = k / 2) → (6 = k / 1) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_l4124_412407


namespace NUMINAMATH_CALUDE_vampire_survival_l4124_412475

/-- The number of people a vampire needs to suck blood from each day to survive -/
def vampire_daily_victims : ℕ :=
  let gallons_per_week : ℕ := 7
  let pints_per_gallon : ℕ := 8
  let pints_per_person : ℕ := 2
  let days_per_week : ℕ := 7
  (gallons_per_week * pints_per_gallon) / (pints_per_person * days_per_week)

theorem vampire_survival : vampire_daily_victims = 4 := by
  sorry

end NUMINAMATH_CALUDE_vampire_survival_l4124_412475


namespace NUMINAMATH_CALUDE_equation_solution_l4124_412427

theorem equation_solution :
  ∃ y : ℝ, y - 9 / (y - 4) = 2 - 9 / (y - 4) ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4124_412427


namespace NUMINAMATH_CALUDE_sqrt_72_plus_sqrt_32_l4124_412431

theorem sqrt_72_plus_sqrt_32 : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_72_plus_sqrt_32_l4124_412431


namespace NUMINAMATH_CALUDE_probability_same_color_is_one_third_l4124_412439

/-- The set of available colors for sportswear -/
inductive Color
  | Red
  | White
  | Blue

/-- The probability of two athletes choosing the same color from three options -/
def probability_same_color : ℚ :=
  1 / 3

/-- Theorem stating that the probability of two athletes choosing the same color is 1/3 -/
theorem probability_same_color_is_one_third :
  probability_same_color = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_is_one_third_l4124_412439


namespace NUMINAMATH_CALUDE_tennis_tournament_result_l4124_412405

/-- Represents the number of participants with k points after m rounds in a tournament with 2^n participants. -/
def f (n m k : ℕ) : ℕ := 2^(n - m) * Nat.choose m k

/-- The rules of the tennis tournament. -/
structure TournamentRules where
  participants : ℕ
  victoryPoints : ℕ
  lossPoints : ℕ
  additionalPointRule : Bool
  pairingRule : Bool

/-- The specific tournament in question. -/
def tennisTournament : TournamentRules where
  participants := 256  -- Including two fictitious participants
  victoryPoints := 1
  lossPoints := 0
  additionalPointRule := true
  pairingRule := true

/-- The theorem to be proved. -/
theorem tennis_tournament_result (t : TournamentRules) (h : t = tennisTournament) :
  f 8 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_tennis_tournament_result_l4124_412405


namespace NUMINAMATH_CALUDE_necessary_to_sufficient_negation_l4124_412499

theorem necessary_to_sufficient_negation (A B : Prop) :
  (B → A) → (¬A → ¬B) := by sorry

end NUMINAMATH_CALUDE_necessary_to_sufficient_negation_l4124_412499


namespace NUMINAMATH_CALUDE_trapezoid_balance_l4124_412429

-- Define the shapes and their weights
variable (C P T : ℝ)

-- Define the balance conditions
axiom balance1 : C = 2 * P
axiom balance2 : T = C + P

-- Theorem to prove
theorem trapezoid_balance : T = 3 * P := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_balance_l4124_412429


namespace NUMINAMATH_CALUDE_twelve_digit_159_div37_not_sum76_l4124_412497

-- Define a function to check if a number consists only of digits 1, 5, and 9
def only_159_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1 ∨ d = 5 ∨ d = 9

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Theorem statement
theorem twelve_digit_159_div37_not_sum76 (n : ℕ) :
  n ≥ 10^11 ∧ n < 10^12 ∧ only_159_digits n ∧ n % 37 = 0 →
  sum_of_digits n ≠ 76 := by
  sorry

end NUMINAMATH_CALUDE_twelve_digit_159_div37_not_sum76_l4124_412497


namespace NUMINAMATH_CALUDE_time_against_walkway_l4124_412468

/-- The time it takes to walk against a moving walkway given specific conditions -/
theorem time_against_walkway 
  (walkway_length : ℝ) 
  (time_with_walkway : ℝ) 
  (time_without_movement : ℝ) 
  (h1 : walkway_length = 100) 
  (h2 : time_with_walkway = 25) 
  (h3 : time_without_movement = 42.857142857142854) :
  let person_speed := walkway_length / time_without_movement
  let walkway_speed := walkway_length / time_with_walkway - person_speed
  walkway_length / (person_speed - walkway_speed) = 150 := by
  sorry

end NUMINAMATH_CALUDE_time_against_walkway_l4124_412468


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l4124_412428

def complex_equation (z : ℂ) : Prop :=
  (1 - Complex.I) * z = (1 + Complex.I)^2

theorem complex_magnitude_theorem (z : ℂ) :
  complex_equation z → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l4124_412428


namespace NUMINAMATH_CALUDE_gcd_and_binary_conversion_l4124_412435

theorem gcd_and_binary_conversion :
  (Nat.gcd 153 119 = 17) ∧
  (ToString.toString (Nat.toDigits 2 89) = "1011001") := by
  sorry

end NUMINAMATH_CALUDE_gcd_and_binary_conversion_l4124_412435


namespace NUMINAMATH_CALUDE_no_integral_solution_l4124_412420

theorem no_integral_solution : ¬∃ (n m : ℤ), n^2 + (n+1)^2 + (n+2)^2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integral_solution_l4124_412420


namespace NUMINAMATH_CALUDE_friendship_theorem_l4124_412478

/-- A graph representing friendships in a class -/
structure FriendshipGraph where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  symmetric : ∀ {i j}, (i, j) ∈ edges → (j, i) ∈ edges
  no_self_loops : ∀ i, (i, i) ∉ edges

/-- The degree of a vertex in the graph -/
def degree (G : FriendshipGraph) (v : ℕ) : ℕ :=
  (G.edges.filter (λ e => e.1 = v ∨ e.2 = v)).card

/-- A clique in the graph -/
def is_clique (G : FriendshipGraph) (S : Finset ℕ) : Prop :=
  ∀ i j, i ∈ S → j ∈ S → i ≠ j → (i, j) ∈ G.edges

theorem friendship_theorem (G : FriendshipGraph) 
  (h1 : G.vertices.card = 20)
  (h2 : ∀ v ∈ G.vertices, degree G v ≥ 14) :
  ∃ S : Finset ℕ, S.card = 4 ∧ is_clique G S :=
sorry

end NUMINAMATH_CALUDE_friendship_theorem_l4124_412478


namespace NUMINAMATH_CALUDE_geography_book_count_l4124_412443

/-- Given a shelf of books with specific counts, calculate the number of geography books. -/
theorem geography_book_count (total : ℕ) (history : ℕ) (math : ℕ) 
  (h_total : total = 100)
  (h_history : history = 32)
  (h_math : math = 43) :
  total - history - math = 25 := by
  sorry

end NUMINAMATH_CALUDE_geography_book_count_l4124_412443


namespace NUMINAMATH_CALUDE_general_form_equation_l4124_412462

theorem general_form_equation (x : ℝ) : 
  (x - 1) * (x - 2) = 4 ↔ x^2 - 3*x - 2 = 0 := by sorry

end NUMINAMATH_CALUDE_general_form_equation_l4124_412462


namespace NUMINAMATH_CALUDE_duck_pond_problem_l4124_412490

theorem duck_pond_problem (initial_ducks : ℕ) (final_ducks : ℕ) 
  (h1 : initial_ducks = 320)
  (h2 : final_ducks = 140) : 
  ∃ (F : ℚ),
    F = 1/6 ∧
    final_ducks = (initial_ducks * 3/4 * (1 - F) * 0.7).floor := by
  sorry

end NUMINAMATH_CALUDE_duck_pond_problem_l4124_412490


namespace NUMINAMATH_CALUDE_total_squares_count_l4124_412417

/-- Represents a point in the grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a square in the grid --/
structure GridSquare where
  topLeft : GridPoint
  size : ℕ

/-- The set of all points in the grid, including additional points --/
def gridPoints : Set GridPoint := sorry

/-- Checks if a square is valid within the given grid --/
def isValidSquare (square : GridSquare) : Prop := sorry

/-- Counts the number of valid squares of a given size --/
def countValidSquares (size : ℕ) : ℕ := sorry

/-- The main theorem to prove --/
theorem total_squares_count :
  (countValidSquares 1) + (countValidSquares 2) = 59 := by sorry

end NUMINAMATH_CALUDE_total_squares_count_l4124_412417


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l4124_412409

/-- Given a quadratic equation x^2 + (2k-1)x + k^2 - k = 0 where x = 2 is one of the roots,
    prove that it has two distinct real roots and the value of -2k^2 - 6k - 5 is -1 -/
theorem quadratic_equation_properties (k : ℝ) :
  (∃ x : ℝ, x^2 + (2*k - 1)*x + k^2 - k = 0) →
  (2^2 + (2*k - 1)*2 + k^2 - k = 0) →
  (∃ x y : ℝ, x ≠ y ∧ x^2 + (2*k - 1)*x + k^2 - k = 0 ∧ y^2 + (2*k - 1)*y + k^2 - k = 0) ∧
  (-2*k^2 - 6*k - 5 = -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l4124_412409


namespace NUMINAMATH_CALUDE_milk_quality_theorem_l4124_412414

/-- The probability of a single bottle of milk being qualified -/
def p_qualified : ℝ := 0.8

/-- The number of bottles bought -/
def n_bottles : ℕ := 2

/-- The number of days considered -/
def n_days : ℕ := 3

/-- The probability that all bought bottles are qualified -/
def prob_all_qualified : ℝ := p_qualified ^ n_bottles

/-- The probability of drinking unqualified milk in a day -/
def p_unqualified_day : ℝ := 1 - p_qualified ^ n_bottles

/-- The expected number of days drinking unqualified milk -/
def expected_unqualified_days : ℝ := n_days * p_unqualified_day

theorem milk_quality_theorem :
  prob_all_qualified = 0.64 ∧ expected_unqualified_days = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_milk_quality_theorem_l4124_412414


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l4124_412445

/-- Represents the atomic weights of elements in g/mol -/
def atomic_weight : String → ℝ
  | "C" => 12.01
  | "H" => 1.008
  | "O" => 16.00
  | _ => 0

/-- Calculates the total mass of a given number of atoms of an element -/
def total_mass (element : String) (num_atoms : ℕ) : ℝ :=
  (atomic_weight element) * (num_atoms : ℝ)

/-- Represents the molecular composition of the compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ
  molecular_weight : ℝ

/-- Calculates the total mass of the compound based on its composition -/
def compound_mass (c : Compound) : ℝ :=
  total_mass "C" c.carbon + total_mass "H" c.hydrogen + total_mass "O" c.oxygen

/-- The theorem to be proved -/
theorem compound_oxygen_atoms (c : Compound) 
  (h1 : c.carbon = 3)
  (h2 : c.hydrogen = 6)
  (h3 : c.molecular_weight = 58) :
  c.oxygen = 1 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_atoms_l4124_412445


namespace NUMINAMATH_CALUDE_luke_coin_count_l4124_412412

theorem luke_coin_count (quarter_piles dime_piles coins_per_pile : ℕ) 
  (h1 : quarter_piles = 5)
  (h2 : dime_piles = 5)
  (h3 : coins_per_pile = 3) : 
  quarter_piles * coins_per_pile + dime_piles * coins_per_pile = 30 := by
  sorry

end NUMINAMATH_CALUDE_luke_coin_count_l4124_412412


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l4124_412424

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_general_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : |a 1| = 1)
  (h_a5_a2 : a 5 = -8 * a 2)
  (h_a5_gt_a2 : a 5 > a 2) :
  ∃ r : ℝ, r = -2 ∧ ∀ n : ℕ, a n = r^(n-1) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l4124_412424


namespace NUMINAMATH_CALUDE_triangle_base_length_l4124_412422

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 9 →
  height = 6 →
  area = (base * height) / 2 →
  base = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l4124_412422


namespace NUMINAMATH_CALUDE_children_with_cats_l4124_412459

/-- Represents the number of children in each category -/
structure KindergartenPets where
  total : ℕ
  onlyDogs : ℕ
  bothPets : ℕ
  onlyCats : ℕ

/-- The conditions of the kindergarten pet situation -/
def kindergartenConditions : KindergartenPets where
  total := 30
  onlyDogs := 18
  bothPets := 6
  onlyCats := 30 - 18 - 6

theorem children_with_cats (k : KindergartenPets) 
  (h1 : k.total = 30)
  (h2 : k.onlyDogs = 18)
  (h3 : k.bothPets = 6)
  (h4 : k.total = k.onlyDogs + k.onlyCats + k.bothPets) :
  k.onlyCats + k.bothPets = 12 := by
  sorry

#eval kindergartenConditions.onlyCats + kindergartenConditions.bothPets

end NUMINAMATH_CALUDE_children_with_cats_l4124_412459


namespace NUMINAMATH_CALUDE_green_apples_count_l4124_412479

theorem green_apples_count (total : ℕ) (red_to_green_ratio : ℕ) 
  (h1 : total = 496) 
  (h2 : red_to_green_ratio = 3) : 
  ∃ (green : ℕ), green = 124 ∧ total = green * (red_to_green_ratio + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_green_apples_count_l4124_412479


namespace NUMINAMATH_CALUDE_right_triangle_trig_inequality_l4124_412471

theorem right_triangle_trig_inequality (A B C : Real) (h1 : 0 < A) (h2 : A < π/4) 
  (h3 : A + B + C = π) (h4 : C = π/2) : Real.sin B > Real.cos B := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_inequality_l4124_412471


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_l4124_412442

/-- Calculates the net rate of pay for a driver given specific conditions. -/
theorem driver_net_pay_rate 
  (travel_time : ℝ) 
  (speed : ℝ) 
  (fuel_efficiency : ℝ) 
  (pay_per_mile : ℝ) 
  (gas_price : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_per_mile = 0.60)
  (h5 : gas_price = 2.50) :
  (pay_per_mile * speed * travel_time - (speed * travel_time / fuel_efficiency) * gas_price) / travel_time = 25 :=
by sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_l4124_412442


namespace NUMINAMATH_CALUDE_total_problems_is_550_l4124_412451

/-- The total number of math problems practiced by Marvin, Arvin, and Kevin over two days -/
def totalProblems (marvinYesterday : ℕ) : ℕ :=
  let marvinToday := 3 * marvinYesterday
  let arvinYesterday := 2 * marvinYesterday
  let arvinToday := 2 * marvinToday
  let kevinYesterday := 30
  let kevinToday := kevinYesterday + 10
  (marvinYesterday + marvinToday) + (arvinYesterday + arvinToday) + (kevinYesterday + kevinToday)

/-- Theorem stating that the total number of problems practiced is 550 -/
theorem total_problems_is_550 : totalProblems 40 = 550 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_is_550_l4124_412451


namespace NUMINAMATH_CALUDE_one_element_condition_at_most_one_element_condition_l4124_412470

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

-- Theorem 1
theorem one_element_condition (a : ℝ) :
  (∃! x, x ∈ A a) ↔ (a = 0 ∨ a = 1) := by sorry

-- Theorem 2
theorem at_most_one_element_condition (a : ℝ) :
  (∃ x, x ∈ A a → ∀ y, y ∈ A a → x = y) ↔ (a ≥ 1 ∨ a = 0) := by sorry

end NUMINAMATH_CALUDE_one_element_condition_at_most_one_element_condition_l4124_412470


namespace NUMINAMATH_CALUDE_second_train_length_l4124_412485

/-- Calculates the length of the second train given the parameters of two trains passing each other. -/
theorem second_train_length
  (first_train_length : ℝ)
  (first_train_speed : ℝ)
  (second_train_speed : ℝ)
  (initial_distance : ℝ)
  (crossing_time : ℝ)
  (h1 : first_train_length = 100)
  (h2 : first_train_speed = 10)
  (h3 : second_train_speed = 15)
  (h4 : initial_distance = 50)
  (h5 : crossing_time = 60)
  : ∃ (second_train_length : ℝ),
    second_train_length = 150 ∧
    second_train_length + first_train_length + initial_distance =
      (second_train_speed - first_train_speed) * crossing_time :=
by
  sorry


end NUMINAMATH_CALUDE_second_train_length_l4124_412485


namespace NUMINAMATH_CALUDE_g_difference_l4124_412489

def g (n : ℕ) : ℚ := (1/4) * n * (n+1) * (n+2) * (n+3)

theorem g_difference (s : ℕ) : g s - g (s-1) = s * (s+1) * (s+2) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l4124_412489


namespace NUMINAMATH_CALUDE_range_of_z_l4124_412483

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  4 ≤ x^2 + 4*y^2 ∧ x^2 + 4*y^2 ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_range_of_z_l4124_412483


namespace NUMINAMATH_CALUDE_a_range_l4124_412401

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := 2 * x - 3 * y + a = 0

-- Define the condition for points being on opposite sides of the line
def opposite_sides (a : ℝ) : Prop :=
  (2 * 2 - 3 * 1 + a) * (2 * 4 - 3 * 3 + a) < 0

-- Theorem statement
theorem a_range (a : ℝ) :
  (∀ x y, line_equation x y a) →
  opposite_sides a →
  -1 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_a_range_l4124_412401


namespace NUMINAMATH_CALUDE_not_prime_cubic_polynomial_l4124_412487

theorem not_prime_cubic_polynomial (n : ℕ+) : ¬ Prime (n.val^3 - 9*n.val^2 + 19*n.val - 13) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_cubic_polynomial_l4124_412487


namespace NUMINAMATH_CALUDE_rectangle_max_area_l4124_412494

/-- Given a fixed perimeter, the area of a rectangle is maximized when it is a square -/
theorem rectangle_max_area (P : ℝ) (h : P > 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = P / 2 →
  x * y ≤ (P / 4) ^ 2 ∧ 
  (x * y = (P / 4) ^ 2 ↔ x = y) := by
sorry


end NUMINAMATH_CALUDE_rectangle_max_area_l4124_412494


namespace NUMINAMATH_CALUDE_no_eight_face_polyhedron_from_cube_cut_l4124_412484

/-- Represents a polyhedron --/
structure Polyhedron where
  faces : ℕ

/-- Represents a cube --/
structure Cube where
  faces : ℕ
  faces_eq_six : faces = 6

/-- Represents the result of cutting a cube with a single plane --/
structure CubeCut where
  original : Cube
  piece1 : Polyhedron
  piece2 : Polyhedron
  single_cut : piece1.faces + piece2.faces = original.faces + 2

/-- Theorem stating that a polyhedron with 8 faces cannot be obtained from cutting a cube with a single plane --/
theorem no_eight_face_polyhedron_from_cube_cut (cut : CubeCut) :
  cut.piece1.faces ≠ 8 ∧ cut.piece2.faces ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_eight_face_polyhedron_from_cube_cut_l4124_412484


namespace NUMINAMATH_CALUDE_b_investment_is_13650_l4124_412447

/-- Represents the investment and profit distribution in a partnership business. -/
structure Partnership where
  a_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- Calculates B's investment given the partnership details. -/
def calculate_b_investment (p : Partnership) : ℕ :=
  p.total_profit * p.a_investment / p.a_profit_share - p.a_investment - p.c_investment

/-- Theorem stating that B's investment is 13650 given the specific partnership details. -/
theorem b_investment_is_13650 (p : Partnership) 
  (h1 : p.a_investment = 6300)
  (h2 : p.c_investment = 10500)
  (h3 : p.total_profit = 12100)
  (h4 : p.a_profit_share = 3630) :
  calculate_b_investment p = 13650 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_is_13650_l4124_412447


namespace NUMINAMATH_CALUDE_no_real_roots_l4124_412476

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the quadratic equation
def quadratic_equation (t : Triangle) (x : ℝ) : Prop :=
  t.a^2 * x^2 + (t.b^2 - t.a^2 - t.c^2) * x + t.c^2 = 0

-- Theorem statement
theorem no_real_roots (t : Triangle) : ¬∃ x : ℝ, quadratic_equation t x := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l4124_412476


namespace NUMINAMATH_CALUDE_largest_common_term_correct_l4124_412423

/-- First arithmetic progression -/
def seq1 (n : ℕ) : ℕ := 7 * (n + 1)

/-- Second arithmetic progression -/
def seq2 (m : ℕ) : ℕ := 8 + 12 * m

/-- Predicate for common terms -/
def isCommonTerm (a : ℕ) : Prop :=
  ∃ n m : ℕ, seq1 n = a ∧ seq2 m = a

/-- The largest common term less than 500 -/
def largestCommonTerm : ℕ := 476

theorem largest_common_term_correct :
  isCommonTerm largestCommonTerm ∧
  largestCommonTerm < 500 ∧
  ∀ x : ℕ, isCommonTerm x → x < 500 → x ≤ largestCommonTerm :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_correct_l4124_412423


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l4124_412400

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 1)) / (x - 2)) ↔ (x ≥ -1 ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l4124_412400


namespace NUMINAMATH_CALUDE_lowest_price_breaks_even_l4124_412436

/-- Calculates the lowest price per component to break even --/
def lowest_price_per_component (production_cost shipping_cost : ℚ) 
  (fixed_costs : ℚ) (num_components : ℕ) : ℚ :=
  (production_cost + shipping_cost + fixed_costs / num_components)

theorem lowest_price_breaks_even 
  (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (num_components : ℕ) :
  let price := lowest_price_per_component production_cost shipping_cost fixed_costs num_components
  (price * num_components : ℚ) = (production_cost + shipping_cost) * num_components + fixed_costs :=
by sorry

#eval lowest_price_per_component 80 5 16500 150

end NUMINAMATH_CALUDE_lowest_price_breaks_even_l4124_412436


namespace NUMINAMATH_CALUDE_sales_volume_estimate_l4124_412472

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := -5 * x + 150

-- Define the theorem
theorem sales_volume_estimate :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |regression_equation 10 - 100| < ε :=
sorry

end NUMINAMATH_CALUDE_sales_volume_estimate_l4124_412472


namespace NUMINAMATH_CALUDE_total_fruit_weight_l4124_412403

def apple_weight : ℕ := 240

theorem total_fruit_weight :
  let pear_weight := 3 * apple_weight
  apple_weight + pear_weight = 960 := by
  sorry

end NUMINAMATH_CALUDE_total_fruit_weight_l4124_412403


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l4124_412469

theorem consecutive_pages_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20736 → n + (n + 1) = 287 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l4124_412469


namespace NUMINAMATH_CALUDE_statements_B_and_C_are_correct_l4124_412465

theorem statements_B_and_C_are_correct (a b c d : ℝ) :
  (((a * b > 0 ∧ b * c - a * d > 0) → (c / a - d / b > 0)) ∧
   ((a > b ∧ c > d) → (a - d > b - c))) := by
  sorry

end NUMINAMATH_CALUDE_statements_B_and_C_are_correct_l4124_412465


namespace NUMINAMATH_CALUDE_parabola_directrix_l4124_412481

/-- The directrix of a parabola y = -x^2 --/
theorem parabola_directrix : ∃ (d : ℝ), ∀ (x y : ℝ),
  y = -x^2 → (∃ (p : ℝ × ℝ), (x - p.1)^2 + (y - p.2)^2 = (y - d)^2 ∧ y ≤ d) → d = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l4124_412481


namespace NUMINAMATH_CALUDE_roots_properties_l4124_412464

def equation_roots (b : ℝ) (θ : ℝ) : Prop :=
  169 * (Real.sin θ)^2 - b * (Real.sin θ) + 60 = 0 ∧
  169 * (Real.cos θ)^2 - b * (Real.cos θ) + 60 = 0

theorem roots_properties (θ : ℝ) (h : π/4 < θ ∧ θ < 3*π/4) :
  ∃ b : ℝ, equation_roots b θ ∧
    b = 221 ∧
    (Real.sin θ / (1 - Real.cos θ)) + ((1 + Real.cos θ) / Real.sin θ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_roots_properties_l4124_412464


namespace NUMINAMATH_CALUDE_ounces_in_pound_l4124_412421

/-- Represents the number of ounces in one pound -/
def ounces_per_pound : ℕ := sorry

theorem ounces_in_pound : 
  (2100 : ℕ) * 13 = 1680 * (16 + 4 / ounces_per_pound) → ounces_per_pound = 16 := by
  sorry

end NUMINAMATH_CALUDE_ounces_in_pound_l4124_412421


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l4124_412444

theorem smaller_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => x^2 - 13*x + 36
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂) ∧
  x₁ = 4 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l4124_412444


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l4124_412418

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l4124_412418


namespace NUMINAMATH_CALUDE_deer_weight_l4124_412411

/-- Calculates the weight of each deer given hunting frequency, season length, and total kept weight --/
theorem deer_weight 
  (hunts_per_month : ℕ)
  (season_fraction : ℚ)
  (deer_per_hunt : ℕ)
  (kept_fraction : ℚ)
  (total_kept_weight : ℕ)
  (h1 : hunts_per_month = 6)
  (h2 : season_fraction = 1/4)
  (h3 : deer_per_hunt = 2)
  (h4 : kept_fraction = 1/2)
  (h5 : total_kept_weight = 10800) :
  (total_kept_weight / kept_fraction) / (hunts_per_month * (season_fraction * 12) * deer_per_hunt) = 600 := by
  sorry

end NUMINAMATH_CALUDE_deer_weight_l4124_412411


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4124_412426

theorem inequality_system_solution :
  ∀ x : ℝ, (2 * x + 1 < 5 ∧ 3 - x > 2) ↔ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4124_412426


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l4124_412433

theorem gain_percent_calculation (MP : ℝ) (MP_pos : MP > 0) : 
  let CP := 0.64 * MP
  let SP := 0.82 * MP
  let gain_percent := ((SP - CP) / CP) * 100
  gain_percent = 28.125 := by sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l4124_412433


namespace NUMINAMATH_CALUDE_complex_circle_theorem_l4124_412408

def complex_circle_problem (a₁ a₂ a₃ a₄ a₅ : ℂ) (s : ℝ) : Prop :=
  (a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0 ∧ a₅ ≠ 0) ∧
  (a₂ / a₁ = a₃ / a₂) ∧ (a₃ / a₂ = a₄ / a₃) ∧ (a₄ / a₃ = a₅ / a₄) ∧
  (a₁ + a₂ + a₃ + a₄ + a₅ = 4 * (1 / a₁ + 1 / a₂ + 1 / a₃ + 1 / a₄ + 1 / a₅)) ∧
  (a₁ + a₂ + a₃ + a₄ + a₅ = s) ∧
  (Complex.abs s ≤ 2) →
  Complex.abs a₁ = 2 ∧ Complex.abs a₂ = 2 ∧ Complex.abs a₃ = 2 ∧ Complex.abs a₄ = 2 ∧ Complex.abs a₅ = 2

theorem complex_circle_theorem (a₁ a₂ a₃ a₄ a₅ : ℂ) (s : ℝ) :
  complex_circle_problem a₁ a₂ a₃ a₄ a₅ s := by
  sorry

end NUMINAMATH_CALUDE_complex_circle_theorem_l4124_412408


namespace NUMINAMATH_CALUDE_no_prime_generating_pair_l4124_412437

theorem no_prime_generating_pair : 
  ¬ ∃ (a b : ℕ+), ∀ (p q : ℕ), 
    1000 < p ∧ 1000 < q ∧ 
    Nat.Prime p ∧ Nat.Prime q ∧ 
    p ≠ q → 
    Nat.Prime (a * p + b * q) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_generating_pair_l4124_412437


namespace NUMINAMATH_CALUDE_vasya_numbers_l4124_412453

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vasya_numbers_l4124_412453


namespace NUMINAMATH_CALUDE_largest_number_digit_sum_l4124_412455

def digits : Finset ℕ := {5, 6, 4, 7}

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  ∃ (a b c : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  n = 100 * a + 10 * b + c

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_number_digit_sum :
  ∃ (max_n : ℕ), is_valid_number max_n ∧
  ∀ (n : ℕ), is_valid_number n → n ≤ max_n ∧
  digit_sum max_n = 18 :=
sorry

end NUMINAMATH_CALUDE_largest_number_digit_sum_l4124_412455


namespace NUMINAMATH_CALUDE_game_result_l4124_412482

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 9
  else if n % 2 = 0 then 3
  else 1

def allie_rolls : List ℕ := [6, 3, 4]
def betty_rolls : List ℕ := [1, 2, 5, 6]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result : 
  total_points allie_rolls * total_points betty_rolls = 294 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l4124_412482


namespace NUMINAMATH_CALUDE_equation_satisfied_at_eight_l4124_412461

def f (x : ℝ) : ℝ := 2 * x - 3

theorem equation_satisfied_at_eight :
  ∃ x : ℝ, 2 * (f x) - 21 = f (x - 4) ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_at_eight_l4124_412461


namespace NUMINAMATH_CALUDE_emu_egg_production_l4124_412441

/-- The number of eggs laid by each female emu per day -/
def eggs_per_female_emu_per_day (num_pens : ℕ) (emus_per_pen : ℕ) (total_eggs_per_week : ℕ) : ℚ :=
  let total_emus := num_pens * emus_per_pen
  let female_emus := total_emus / 2
  (total_eggs_per_week : ℚ) / (female_emus : ℚ) / 7

theorem emu_egg_production :
  eggs_per_female_emu_per_day 4 6 84 = 1 := by
  sorry

#eval eggs_per_female_emu_per_day 4 6 84

end NUMINAMATH_CALUDE_emu_egg_production_l4124_412441


namespace NUMINAMATH_CALUDE_infinite_divisibility_l4124_412454

theorem infinite_divisibility (p : Nat) (h_prime : Nat.Prime p) (h_mod : p % 4 = 1) (h_not_17 : p ≠ 17) :
  let n := p
  ∃ k : Nat, 3^((n - 2)^(n - 1) - 1) - 1 = 17 * n^2 * k := by
  sorry

end NUMINAMATH_CALUDE_infinite_divisibility_l4124_412454


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l4124_412416

theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    b = (3/2) * a →
    c = 2 * a →
    a + b + c = 180 →
    max a (max b c) = 80 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l4124_412416


namespace NUMINAMATH_CALUDE_amandas_family_size_l4124_412493

theorem amandas_family_size :
  let total_rooms : ℕ := 9
  let rooms_with_four_walls : ℕ := 5
  let rooms_with_five_walls : ℕ := 4
  let walls_per_person : ℕ := 8
  let total_walls : ℕ := rooms_with_four_walls * 4 + rooms_with_five_walls * 5
  total_rooms = rooms_with_four_walls + rooms_with_five_walls →
  total_walls % walls_per_person = 0 →
  total_walls / walls_per_person = 5 :=
by sorry

end NUMINAMATH_CALUDE_amandas_family_size_l4124_412493


namespace NUMINAMATH_CALUDE_max_sum_of_product_60_l4124_412415

theorem max_sum_of_product_60 (a b c : ℕ) (h : a * b * c = 60) :
  a + b + c ≤ 62 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_product_60_l4124_412415


namespace NUMINAMATH_CALUDE_area_of_problem_shape_l4124_412425

/-- A composite shape with right-angled corners -/
structure CompositeShape :=
  (height1 : ℕ)
  (width1 : ℕ)
  (height2 : ℕ)
  (width2 : ℕ)
  (height3 : ℕ)
  (width3 : ℕ)

/-- Calculate the area of the composite shape -/
def area (shape : CompositeShape) : ℕ :=
  shape.height1 * shape.width1 +
  shape.height2 * shape.width2 +
  shape.height3 * shape.width3

/-- The specific shape from the problem -/
def problem_shape : CompositeShape :=
  { height1 := 8
  , width1 := 4
  , height2 := 6
  , width2 := 4
  , height3 := 5
  , width3 := 3 }

theorem area_of_problem_shape :
  area problem_shape = 71 :=
by sorry

end NUMINAMATH_CALUDE_area_of_problem_shape_l4124_412425


namespace NUMINAMATH_CALUDE_smallest_staircase_length_l4124_412492

theorem smallest_staircase_length (n : ℕ) : 
  n > 30 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 ∧
  (∀ m : ℕ, m > 30 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_staircase_length_l4124_412492


namespace NUMINAMATH_CALUDE_triangle_problem_l4124_412450

theorem triangle_problem (a b c A B C : ℝ) (h1 : a = 3) 
  (h2 : (a + b) * Real.sin B = (Real.sin A + Real.sin C) * (a + b - c))
  (h3 : a * Real.cos B + b * Real.cos A = Real.sqrt 3) :
  A = π / 3 ∧ (1 / 2 : ℝ) * a * c = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l4124_412450


namespace NUMINAMATH_CALUDE_sphere_radius_l4124_412413

/-- The radius of a sphere that forms a quarter-sphere with radius 4∛4 cm is 4 cm. -/
theorem sphere_radius (r : ℝ) : r = 4 * Real.rpow 4 (1/3) → 4 = (1/4)^(1/3) * r := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_l4124_412413


namespace NUMINAMATH_CALUDE_sequence_decreasing_l4124_412466

def x (a : ℝ) (n : ℕ) : ℝ := 2^n * (a^(1/(2*n)) - 1)

theorem sequence_decreasing (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  ∀ n : ℕ, x a n > x a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_decreasing_l4124_412466


namespace NUMINAMATH_CALUDE_min_blue_beads_l4124_412402

/-- Represents a necklace with red and blue beads. -/
structure Necklace :=
  (red_beads : ℕ)
  (blue_beads : ℕ)

/-- Checks if a necklace satisfies the condition that any segment
    containing 10 red beads also contains at least 7 blue beads. -/
def satisfies_condition (n : Necklace) : Prop :=
  ∀ (segment : List (Bool)), 
    segment.length ≤ n.red_beads + n.blue_beads →
    (segment.filter id).length = 10 →
    (segment.filter not).length ≥ 7

/-- The main theorem: The minimum number of blue beads in a necklace
    with 100 red beads that satisfies the given condition is 78. -/
theorem min_blue_beads :
  ∃ (n : Necklace), 
    n.red_beads = 100 ∧ 
    satisfies_condition n ∧
    n.blue_beads = 78 ∧
    (∀ (m : Necklace), m.red_beads = 100 → satisfies_condition m → m.blue_beads ≥ 78) :=
by sorry

end NUMINAMATH_CALUDE_min_blue_beads_l4124_412402


namespace NUMINAMATH_CALUDE_oil_production_theorem_l4124_412491

/-- Oil production per person for different regions --/
def oil_production_problem : Prop :=
  let west_production := 55.084
  let non_west_production := 214.59
  let russia_production := 1038.33
  let total_production := 13737.1
  let russia_percentage := 0.09
  let russia_population := 147000000

  let russia_total_production := total_production * russia_percentage
  let russia_per_person := russia_total_production / russia_population

  (west_production = 55.084) ∧
  (non_west_production = 214.59) ∧
  (russia_per_person = 1038.33)

theorem oil_production_theorem : oil_production_problem := by
  sorry

end NUMINAMATH_CALUDE_oil_production_theorem_l4124_412491


namespace NUMINAMATH_CALUDE_scalene_triangle_ratio_bounds_l4124_412498

theorem scalene_triangle_ratio_bounds (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_scalene : a > b ∧ b > c) (h_avg : a + c = 2 * b) : 1/3 < c/a ∧ c/a < 1 := by
  sorry

end NUMINAMATH_CALUDE_scalene_triangle_ratio_bounds_l4124_412498


namespace NUMINAMATH_CALUDE_circle_equation_l4124_412419

/-- The standard equation of a circle with center on y = 2x and tangent to x-axis at (-1, 0) -/
theorem circle_equation : ∃ (h k : ℝ), 
  (h = -1 ∧ k = -2) ∧  -- Center on y = 2x
  ((x : ℝ) + 1)^2 + ((y : ℝ) + 2)^2 = 4 ∧  -- Standard equation
  (∀ (x y : ℝ), y = 2*x → (x - h)^2 + (y - k)^2 = 4) ∧  -- Center on y = 2x
  ((-1 : ℝ) - h)^2 + (0 - k)^2 = 4  -- Tangent to x-axis at (-1, 0)
  := by sorry

end NUMINAMATH_CALUDE_circle_equation_l4124_412419


namespace NUMINAMATH_CALUDE_no_mems_are_veens_l4124_412434

universe u

def Mem : Type u := sorry
def En : Type u := sorry
def Veen : Type u := sorry

variable (is_mem : Mem → Prop)
variable (is_en : En → Prop)
variable (is_veen : Veen → Prop)

axiom all_mems_are_ens : ∀ (m : Mem), ∃ (e : En), is_mem m → is_en e
axiom no_ens_are_veens : ¬∃ (e : En) (v : Veen), is_en e ∧ is_veen v

theorem no_mems_are_veens : ¬∃ (m : Mem) (v : Veen), is_mem m ∧ is_veen v := by
  sorry

end NUMINAMATH_CALUDE_no_mems_are_veens_l4124_412434


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l4124_412440

theorem ellipse_foci_distance (x y : ℝ) :
  (9 * x^2 + y^2 = 36) →
  (∃ (c : ℝ), c > 0 ∧ c^2 = 32 ∧ 2 * c = 8 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l4124_412440


namespace NUMINAMATH_CALUDE_range_of_a_l4124_412458

theorem range_of_a (x a : ℝ) :
  (∀ x, (x - 2) * (x - 3) < 0 → -4 < x - a ∧ x - a < 4) →
  -1 ≤ a ∧ a ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4124_412458


namespace NUMINAMATH_CALUDE_apple_picking_contest_l4124_412473

/-- The number of apples picked by Marin -/
def marin_apples : ℕ := 9

/-- The number of apples picked by Donald -/
def donald_apples : ℕ := 11

/-- The number of apples picked by Ana -/
def ana_apples : ℕ := 2 * (marin_apples + donald_apples)

/-- The total number of apples picked by all three participants -/
def total_apples : ℕ := marin_apples + donald_apples + ana_apples

theorem apple_picking_contest :
  total_apples = 60 := by sorry

end NUMINAMATH_CALUDE_apple_picking_contest_l4124_412473


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_l4124_412486

/-- An isosceles triangle with two sides of length 4 and a base of length 3 -/
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ
  is_isosceles : side = 4 ∧ base = 3

/-- A circle passing through the vertices of a triangle -/
structure CircumscribedCircle (t : IsoscelesTriangle) where
  radius : ℝ
  passes_through_vertices : True  -- This is a simplification, as we can't easily express this condition in Lean

/-- The theorem stating that the area of the circumscribed circle is 16π -/
theorem circumscribed_circle_area (t : IsoscelesTriangle) 
  (c : CircumscribedCircle t) : Real.pi * c.radius ^ 2 = 16 * Real.pi := by
  sorry

#check circumscribed_circle_area

end NUMINAMATH_CALUDE_circumscribed_circle_area_l4124_412486


namespace NUMINAMATH_CALUDE_lexie_paintings_count_l4124_412446

/-- The number of rooms where paintings are placed -/
def num_rooms : ℕ := 4

/-- The number of paintings placed in each room -/
def paintings_per_room : ℕ := 8

/-- The total number of Lexie's watercolor paintings -/
def total_paintings : ℕ := num_rooms * paintings_per_room

theorem lexie_paintings_count : total_paintings = 32 := by
  sorry

end NUMINAMATH_CALUDE_lexie_paintings_count_l4124_412446


namespace NUMINAMATH_CALUDE_quadratic_root_sum_equality_l4124_412410

theorem quadratic_root_sum_equality (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h₁ : b₁^2 - 4*c₁ = 1)
  (h₂ : b₂^2 - 4*c₂ = 4)
  (h₃ : b₃^2 - 4*c₃ = 9) :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁^2 + b₁*x₁ + c₁ = 0) ∧
    (y₁^2 + b₁*y₁ + c₁ = 0) ∧
    (x₂^2 + b₂*x₂ + c₂ = 0) ∧
    (y₂^2 + b₂*y₂ + c₂ = 0) ∧
    (x₃^2 + b₃*x₃ + c₃ = 0) ∧
    (y₃^2 + b₃*y₃ + c₃ = 0) ∧
    (x₁ + x₂ + y₃ = y₁ + y₂ + x₃) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_equality_l4124_412410


namespace NUMINAMATH_CALUDE_square_sum_value_l4124_412438

theorem square_sum_value (x y : ℝ) (h1 : x - y = 12) (h2 : x * y = 9) : x^2 + y^2 = 162 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l4124_412438


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l4124_412467

/-- The shaded area of a square with six inscribed circles --/
theorem shaded_area_square_with_circles (square_side : ℝ) (circle_diameter : ℝ) :
  square_side = 24 →
  circle_diameter = 8 →
  let square_area := square_side ^ 2
  let circle_area := π * (circle_diameter / 2) ^ 2
  let total_circles_area := 6 * circle_area
  let shaded_area := square_area - total_circles_area
  shaded_area = 576 - 96 * π := by
  sorry

#check shaded_area_square_with_circles

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l4124_412467


namespace NUMINAMATH_CALUDE_linear_system_solution_l4124_412456

theorem linear_system_solution (x y a : ℝ) : 
  x + 2*y = 2 → 
  2*x + y = a → 
  x + y = 5 → 
  a = 13 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l4124_412456


namespace NUMINAMATH_CALUDE_total_tires_in_parking_lot_l4124_412432

def num_cars : ℕ := 30
def regular_tires_per_car : ℕ := 4
def spare_tires_per_car : ℕ := 1

theorem total_tires_in_parking_lot :
  (num_cars * (regular_tires_per_car + spare_tires_per_car)) = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_tires_in_parking_lot_l4124_412432


namespace NUMINAMATH_CALUDE_jeff_vehicle_collection_l4124_412448

theorem jeff_vehicle_collection (trucks : ℕ) : 
  let cars := 2 * trucks
  trucks + cars = 3 * trucks := by
sorry

end NUMINAMATH_CALUDE_jeff_vehicle_collection_l4124_412448


namespace NUMINAMATH_CALUDE_product_of_largest_primes_eq_679679_l4124_412474

/-- The largest one-digit prime number -/
def largest_one_digit_prime : ℕ := 7

/-- The largest two-digit prime number -/
def largest_two_digit_prime : ℕ := 97

/-- The largest three-digit prime number -/
def largest_three_digit_prime : ℕ := 997

/-- The product of the largest one-digit, two-digit, and three-digit primes -/
def product_of_largest_primes : ℕ := 
  largest_one_digit_prime * largest_two_digit_prime * largest_three_digit_prime

theorem product_of_largest_primes_eq_679679 : 
  product_of_largest_primes = 679679 := by
  sorry

end NUMINAMATH_CALUDE_product_of_largest_primes_eq_679679_l4124_412474


namespace NUMINAMATH_CALUDE_bulk_warehouse_case_price_l4124_412495

/-- The price of a case at the bulk warehouse -/
def bulk_case_price (cans_per_case : ℕ) (grocery_cans : ℕ) (grocery_price : ℚ) (price_difference : ℚ) : ℚ :=
  let grocery_price_per_can : ℚ := grocery_price / grocery_cans
  let bulk_price_per_can : ℚ := grocery_price_per_can - price_difference
  cans_per_case * bulk_price_per_can

/-- Theorem stating that the price of a case at the bulk warehouse is $12.00 -/
theorem bulk_warehouse_case_price :
  bulk_case_price 48 12 6 (25/100) = 12 := by
  sorry

end NUMINAMATH_CALUDE_bulk_warehouse_case_price_l4124_412495


namespace NUMINAMATH_CALUDE_unique_prime_satisfying_conditions_l4124_412457

theorem unique_prime_satisfying_conditions : ∃! p : ℕ, 
  Prime p ∧ 
  100 < p ∧ p < 500 ∧
  (2^2016 : ℤ) ≡ (-2^21 : ℤ) [ZMOD p] ∧
  ∃ e : ℕ, e > 100 ∧ 
    (e : ℤ) ≡ 2016 [ZMOD (p-1)] ∧ 
    e - (p-1)/2 = 21 ∧
  p = 211 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_satisfying_conditions_l4124_412457


namespace NUMINAMATH_CALUDE_line_intersects_both_axes_l4124_412449

/-- A line in the form Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ
  not_both_zero : A ≠ 0 ∨ B ≠ 0

/-- Predicate for a line intersecting both coordinate axes -/
def intersects_both_axes (l : Line) : Prop :=
  ∃ x y : ℝ, (l.A * x + l.C = 0) ∧ (l.B * y + l.C = 0)

/-- Theorem stating the condition for a line to intersect both coordinate axes -/
theorem line_intersects_both_axes (l : Line) : 
  intersects_both_axes l ↔ l.A ≠ 0 ∧ l.B ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_both_axes_l4124_412449


namespace NUMINAMATH_CALUDE_unique_root_implies_a_equals_three_a_equals_three_implies_unique_root_unique_root_iff_a_equals_three_l4124_412496

/-- The function f(x) defined by the equation --/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * abs x + a^2 - 9

/-- Theorem stating that a = 3 is the only value for which f has a unique root --/
theorem unique_root_implies_a_equals_three :
  ∀ a : ℝ, (∃! x : ℝ, f a x = 0) → a = 3 :=
by sorry

/-- Theorem stating that when a = 3, f has a unique root --/
theorem a_equals_three_implies_unique_root :
  ∃! x : ℝ, f 3 x = 0 :=
by sorry

/-- The main theorem combining the above results --/
theorem unique_root_iff_a_equals_three :
  ∀ a : ℝ, (∃! x : ℝ, f a x = 0) ↔ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_implies_a_equals_three_a_equals_three_implies_unique_root_unique_root_iff_a_equals_three_l4124_412496


namespace NUMINAMATH_CALUDE_number_transformation_l4124_412488

theorem number_transformation (x : ℝ) : 
  x + 0.40 * x = 1680 → x * 0.80 * 1.15 = 1104 := by
  sorry

end NUMINAMATH_CALUDE_number_transformation_l4124_412488


namespace NUMINAMATH_CALUDE_range_of_x_l4124_412463

theorem range_of_x (x : ℝ) : 
  (6 - 3 * x ≥ 0) ∧ (1 / (x + 1) ≥ 0) → x ∈ Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l4124_412463


namespace NUMINAMATH_CALUDE_max_angle_cone_from_semicircle_l4124_412460

/-- The maximum angle between generatrices of a cone formed by a semicircle -/
theorem max_angle_cone_from_semicircle :
  ∀ (r : ℝ),
  r > 0 →
  let semicircle_arc_length := r * Real.pi
  let base_circumference := 2 * r * Real.pi / 2
  semicircle_arc_length = base_circumference →
  ∃ (θ : ℝ),
  θ = 60 * (Real.pi / 180) ∧
  ∀ (α : ℝ),
  (α ≥ 0 ∧ α ≤ θ) →
  ∃ (g₁ g₂ : ℝ × ℝ),
  (g₁.1 - g₂.1)^2 + (g₁.2 - g₂.2)^2 ≤ r^2 ∧
  Real.arccos ((g₁.1 * g₂.1 + g₁.2 * g₂.2) / r^2) = α :=
by sorry

end NUMINAMATH_CALUDE_max_angle_cone_from_semicircle_l4124_412460


namespace NUMINAMATH_CALUDE_inverse_function_range_l4124_412480

/-- Given a function f and its inverse, prove the range of a -/
theorem inverse_function_range (a : ℝ) (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : 
  (∀ x, f x = a^(x+1) - 2) →
  (a > 1) →
  (∀ x, f_inv (f x) = x) →
  (∀ x, x ≤ 0 → f_inv x ≤ 0) →
  a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_range_l4124_412480


namespace NUMINAMATH_CALUDE_commute_days_l4124_412404

theorem commute_days (bus_to_work bus_to_home train_days train_both : ℕ) : 
  bus_to_work = 12 → 
  bus_to_home = 20 → 
  train_days = 14 → 
  train_both = 2 → 
  ∃ x : ℕ, x = 23 ∧ 
    x = (bus_to_home - bus_to_work + train_both) + 
        (bus_to_work - train_both) + 
        (train_days - (bus_to_home - bus_to_work)) + 
        train_both :=
by sorry

end NUMINAMATH_CALUDE_commute_days_l4124_412404


namespace NUMINAMATH_CALUDE_hyperbola_tangent_line_l4124_412406

/-- The equation of a tangent line to a hyperbola -/
theorem hyperbola_tangent_line (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : x₀^2 / a^2 - y₀^2 / b^2 = 1) :
  ∃ (x y : ℝ → ℝ), ∀ t, 
    (x t)^2 / a^2 - (y t)^2 / b^2 = 1 ∧ 
    x 0 = x₀ ∧ 
    y 0 = y₀ ∧
    (∀ s, x₀ * (x s) / a^2 - y₀ * (y s) / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_line_l4124_412406

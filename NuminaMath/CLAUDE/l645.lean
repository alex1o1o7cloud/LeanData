import Mathlib

namespace NUMINAMATH_CALUDE_min_value_f_l645_64520

def f (c d x : ℝ) : ℝ := x^3 + c*x + d

theorem min_value_f (c d : ℝ) (h : c = 0) : 
  ∀ x, f c d x ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_value_f_l645_64520


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l645_64539

-- Define the triangle PQR
structure Triangle :=
  (P Q R : Point)
  (altitude : ℝ)
  (base : ℝ)

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : Point)
  (width : ℝ)
  (height : ℝ)

-- Define the problem
def inscribed_rectangle_problem (triangle : Triangle) (rect : Rectangle) : Prop :=
  -- Rectangle ABCD is inscribed in triangle PQR
  -- Side AD of the rectangle is on side PR of the triangle
  -- Triangle's altitude from vertex Q to side PR is 8 inches
  triangle.altitude = 8 ∧
  -- PR = 12 inches
  triangle.base = 12 ∧
  -- Length of AB is equal to a third the length of AD
  rect.width = rect.height / 3 ∧
  -- The area of the rectangle is 64/3 square inches
  rect.width * rect.height = 64 / 3

-- Theorem statement
theorem inscribed_rectangle_area 
  (triangle : Triangle) (rect : Rectangle) :
  inscribed_rectangle_problem triangle rect → 
  rect.width * rect.height = 64 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l645_64539


namespace NUMINAMATH_CALUDE_probability_of_humanities_is_two_thirds_l645_64558

/-- Represents a school subject -/
inductive Subject
| Mathematics
| Chinese
| Politics
| Geography
| English
| History
| PhysicalEducation

/-- Represents the time of day for a class -/
inductive TimeOfDay
| Morning
| Afternoon

/-- Defines whether a subject is considered a humanities subject -/
def isHumanities (s : Subject) : Bool :=
  match s with
  | Subject.Politics | Subject.History | Subject.Geography => true
  | _ => false

/-- Returns the list of subjects for a given time of day -/
def subjectsForTime (t : TimeOfDay) : List Subject :=
  match t with
  | TimeOfDay.Morning => [Subject.Mathematics, Subject.Chinese, Subject.Politics, Subject.Geography]
  | TimeOfDay.Afternoon => [Subject.English, Subject.History, Subject.PhysicalEducation]

/-- Calculates the probability of selecting at least one humanities class -/
def probabilityOfHumanities : ℚ :=
  let morningSubjects := subjectsForTime TimeOfDay.Morning
  let afternoonSubjects := subjectsForTime TimeOfDay.Afternoon
  let totalCombinations := morningSubjects.length * afternoonSubjects.length
  let humanitiesCombinations := 
    (morningSubjects.filter isHumanities).length * afternoonSubjects.length +
    (morningSubjects.filter (not ∘ isHumanities)).length * (afternoonSubjects.filter isHumanities).length
  humanitiesCombinations / totalCombinations

theorem probability_of_humanities_is_two_thirds :
  probabilityOfHumanities = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_humanities_is_two_thirds_l645_64558


namespace NUMINAMATH_CALUDE_employee_transportation_difference_l645_64519

/-- Proves the difference between employees who drive and those who take public transportation -/
theorem employee_transportation_difference
  (total_employees : ℕ)
  (drive_percentage : ℚ)
  (public_transport_fraction : ℚ)
  (h_total : total_employees = 200)
  (h_drive : drive_percentage = 3/5)
  (h_public : public_transport_fraction = 1/2) :
  (drive_percentage * total_employees : ℚ) -
  (public_transport_fraction * (total_employees - drive_percentage * total_employees) : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_employee_transportation_difference_l645_64519


namespace NUMINAMATH_CALUDE_work_completion_time_l645_64586

theorem work_completion_time (work_rate_individual : ℝ) (total_work : ℝ) : 
  work_rate_individual > 0 → total_work > 0 →
  (total_work / work_rate_individual = 50) →
  (total_work / (2 * work_rate_individual) = 25) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l645_64586


namespace NUMINAMATH_CALUDE_probability_less_than_4_l645_64509

/-- A square in the 2D plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point in the square satisfies a condition -/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The specific square with vertices (0,0), (0,3), (3,3), and (3,0) -/
def specificSquare : Square :=
  { bottomLeft := (0, 0), sideLength := 3 }

/-- The condition x + y < 4 -/
def conditionLessThan4 (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 4

theorem probability_less_than_4 :
  probability specificSquare conditionLessThan4 = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_4_l645_64509


namespace NUMINAMATH_CALUDE_a_greater_than_b_squared_l645_64516

theorem a_greater_than_b_squared {a b : ℝ} (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_squared_l645_64516


namespace NUMINAMATH_CALUDE_polynomial_existence_l645_64524

theorem polynomial_existence (n : ℕ) : ∃ P : Polynomial ℤ,
  (∀ (i : ℕ), (P.coeff i) ∈ ({0, -1, 1} : Set ℤ)) ∧
  (P.degree ≤ 2^n) ∧
  ((X - 1)^n ∣ P) ∧
  (P ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_existence_l645_64524


namespace NUMINAMATH_CALUDE_max_teams_in_tournament_l645_64587

/-- The number of players in each team -/
def players_per_team : ℕ := 3

/-- The maximum number of games that can be played in the tournament -/
def max_games : ℕ := 200

/-- The number of games played between two teams -/
def games_between_teams : ℕ := players_per_team * players_per_team

/-- The function to calculate the total number of games for a given number of teams -/
def total_games (n : ℕ) : ℕ := games_between_teams * (n * (n - 1) / 2)

/-- The theorem stating the maximum number of teams that can participate -/
theorem max_teams_in_tournament : 
  ∃ (n : ℕ), n > 0 ∧ total_games n ≤ max_games ∧ ∀ m : ℕ, m > n → total_games m > max_games :=
by sorry

end NUMINAMATH_CALUDE_max_teams_in_tournament_l645_64587


namespace NUMINAMATH_CALUDE_path_area_and_cost_l645_64518

/-- Calculates the area of a rectangular path surrounding a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem path_area_and_cost (field_length field_width path_width cost_per_unit : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 40)
  (h3 : path_width = 2.5)
  (h4 : cost_per_unit = 2) :
  path_area field_length field_width path_width = 600 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_unit = 1200 := by
  sorry

#eval path_area 75 40 2.5
#eval construction_cost (path_area 75 40 2.5) 2

end NUMINAMATH_CALUDE_path_area_and_cost_l645_64518


namespace NUMINAMATH_CALUDE_geometric_sum_abs_l645_64577

def geometric_sequence (n : ℕ) (a₁ : ℝ) (r : ℝ) : ℝ := a₁ * r^(n-1)

theorem geometric_sum_abs (a₁ r : ℝ) (h : a₁ = 1 ∧ r = -2) :
  let a := geometric_sequence
  a 1 a₁ r + |a 2 a₁ r| + |a 3 a₁ r| + a 4 a₁ r = 15 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_abs_l645_64577


namespace NUMINAMATH_CALUDE_final_sequence_values_l645_64591

/-- The number of elements in the initial sequence -/
def n : ℕ := 2022

/-- Function to calculate the new value for a given position after one iteration -/
def newValue (i : ℕ) : ℕ := i^2 + 1

/-- The number of iterations required to reduce the sequence to two numbers -/
def iterations : ℕ := (n - 2) / 2

/-- The final two numbers in the sequence after all iterations -/
def finalPair : (ℕ × ℕ) := (newValue (n/2) + iterations, newValue (n/2 + 1) + iterations)

/-- Theorem stating the final two numbers in the sequence -/
theorem final_sequence_values :
  finalPair = (1023131, 1025154) := by sorry

end NUMINAMATH_CALUDE_final_sequence_values_l645_64591


namespace NUMINAMATH_CALUDE_game_ends_after_28_rounds_l645_64595

/-- Represents the state of the game at any given round -/
structure GameState where
  x : Nat
  y : Nat
  z : Nat

/-- Represents the rules of the token redistribution game -/
def redistributeTokens (state : GameState) : GameState :=
  sorry

/-- Determines if the game has ended (i.e., if any player has run out of tokens) -/
def gameEnded (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends -/
def countRounds (state : GameState) : Nat :=
  sorry

/-- Theorem stating that the game ends after 28 rounds -/
theorem game_ends_after_28_rounds :
  countRounds (GameState.mk 18 15 12) = 28 := by
  sorry

end NUMINAMATH_CALUDE_game_ends_after_28_rounds_l645_64595


namespace NUMINAMATH_CALUDE_linear_increase_l645_64576

/-- A linear function f(x) = 5x - 3 -/
def f (x : ℝ) : ℝ := 5 * x - 3

/-- Theorem: For a linear function f(x) = 5x - 3, 
    if x₁ < x₂, then f(x₁) < f(x₂) -/
theorem linear_increase (x₁ x₂ : ℝ) (h : x₁ < x₂) : f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_linear_increase_l645_64576


namespace NUMINAMATH_CALUDE_betty_boxes_l645_64548

theorem betty_boxes (total_oranges : ℕ) (oranges_per_box : ℕ) (boxes : ℕ) : 
  total_oranges = 24 → 
  oranges_per_box = 8 → 
  total_oranges = boxes * oranges_per_box → 
  boxes = 3 := by
sorry

end NUMINAMATH_CALUDE_betty_boxes_l645_64548


namespace NUMINAMATH_CALUDE_data_plan_total_cost_l645_64581

/-- Calculates the total cost of a data plan over 6 months with special conditions -/
def data_plan_cost (regular_charge : ℚ) (promo_rate : ℚ) (extra_fee : ℚ) : ℚ :=
  let first_month := regular_charge * promo_rate
  let fourth_month := regular_charge + extra_fee
  let regular_months := 4 * regular_charge
  first_month + fourth_month + regular_months

/-- Proves that the total cost for the given conditions is $175 -/
theorem data_plan_total_cost :
  data_plan_cost 30 (1/3) 15 = 175 := by
  sorry

#eval data_plan_cost 30 (1/3) 15

end NUMINAMATH_CALUDE_data_plan_total_cost_l645_64581


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l645_64579

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_side_ratio (t : Triangle) : 
  (t.A : Real) / (t.B : Real) = 1 / 2 ∧ 
  (t.B : Real) / (t.C : Real) = 2 / 3 → 
  t.a / t.b = 1 / Real.sqrt 3 ∧ 
  t.b / t.c = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l645_64579


namespace NUMINAMATH_CALUDE_aku_birthday_friends_l645_64565

/-- Given the conditions of Aku's birthday party, prove the number of friends invited. -/
theorem aku_birthday_friends (packages : Nat) (cookies_per_package : Nat) (cookies_per_child : Nat) :
  packages = 3 →
  cookies_per_package = 25 →
  cookies_per_child = 15 →
  (packages * cookies_per_package) / cookies_per_child - 1 = 4 := by
  sorry

#eval (3 * 25) / 15 - 1  -- Expected output: 4

end NUMINAMATH_CALUDE_aku_birthday_friends_l645_64565


namespace NUMINAMATH_CALUDE_sum_of_first_six_primes_mod_seventh_prime_l645_64546

theorem sum_of_first_six_primes_mod_seventh_prime : 
  let sum_first_six_primes := 41
  let seventh_prime := 17
  sum_first_six_primes % seventh_prime = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_six_primes_mod_seventh_prime_l645_64546


namespace NUMINAMATH_CALUDE_eliza_numbers_l645_64563

theorem eliza_numbers (a b : ℤ) (h1 : 2 * a + 3 * b = 110) (h2 : a = 32 ∨ b = 32) : 
  (a = 7 ∧ b = 32) ∨ (a = 32 ∧ b = 7) := by
sorry

end NUMINAMATH_CALUDE_eliza_numbers_l645_64563


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l645_64574

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (1 + Real.sqrt (2 * y - 3)) = Real.sqrt 6 → y = 14 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l645_64574


namespace NUMINAMATH_CALUDE_jose_share_correct_l645_64536

/-- Calculates an investor's share of the profit based on their investment amount, duration, and the total profit, given the investments and durations of all participants. -/
def calculate_share (tom_investment : ℕ) (tom_duration : ℕ) (jose_investment : ℕ) (jose_duration : ℕ) (maria_investment : ℕ) (maria_duration : ℕ) (total_profit : ℕ) : ℚ :=
  let total_capital_months : ℕ := tom_investment * tom_duration + jose_investment * jose_duration + maria_investment * maria_duration
  (jose_investment * jose_duration : ℚ) / total_capital_months * total_profit

/-- Proves that Jose's share of the profit is correct given the specific investments and durations. -/
theorem jose_share_correct (total_profit : ℕ) : 
  calculate_share 30000 12 45000 10 60000 8 total_profit = 
  (45000 * 10 : ℚ) / (30000 * 12 + 45000 * 10 + 60000 * 8) * total_profit :=
by sorry

end NUMINAMATH_CALUDE_jose_share_correct_l645_64536


namespace NUMINAMATH_CALUDE_product_increase_thirteen_times_l645_64532

theorem product_increase_thirteen_times :
  ∃ (a b c d e f g : ℕ),
    (a - 3) * (b - 3) * (c - 3) * (d - 3) * (e - 3) * (f - 3) * (g - 3) = 13 * (a * b * c * d * e * f * g) :=
by sorry

end NUMINAMATH_CALUDE_product_increase_thirteen_times_l645_64532


namespace NUMINAMATH_CALUDE_set_B_representation_l645_64594

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | f a b x - x = 0}

-- Define set B
def B (a b : ℝ) : Set ℝ := {x | f a b x - a*x = 0}

-- State the theorem
theorem set_B_representation (a b : ℝ) : 
  A a b = {1, -3} → B a b = {-2 - Real.sqrt 7, -2 + Real.sqrt 7} := by
  sorry

end NUMINAMATH_CALUDE_set_B_representation_l645_64594


namespace NUMINAMATH_CALUDE_congruence_problem_l645_64570

theorem congruence_problem (x : ℤ) : 
  (3 * x + 8) % 17 = 3 → (2 * x + 14) % 17 = 5 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l645_64570


namespace NUMINAMATH_CALUDE_equal_money_after_transfer_l645_64575

/-- Represents the amount of gold coins each merchant has -/
structure Merchants where
  foma : ℤ
  ierema : ℤ
  yuliy : ℤ

/-- The conditions of the problem -/
def satisfies_conditions (m : Merchants) : Prop :=
  (m.ierema + 70 = m.yuliy) ∧ (m.foma - 40 = m.yuliy)

/-- The theorem to prove -/
theorem equal_money_after_transfer (m : Merchants) 
  (h : satisfies_conditions m) : 
  m.foma - 55 = m.ierema + 55 := by
  sorry

#check equal_money_after_transfer

end NUMINAMATH_CALUDE_equal_money_after_transfer_l645_64575


namespace NUMINAMATH_CALUDE_crate_middle_dimension_l645_64552

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder fits upright in a crate -/
def cylinderFitsUpright (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.height ≤ crate.width) ∨
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.height ≤ crate.height) ∨
  (cylinder.radius * 2 ≤ crate.width ∧ cylinder.height ≤ crate.length) ∨
  (cylinder.radius * 2 ≤ crate.width ∧ cylinder.height ≤ crate.height) ∨
  (cylinder.radius * 2 ≤ crate.height ∧ cylinder.height ≤ crate.length) ∨
  (cylinder.radius * 2 ≤ crate.height ∧ cylinder.height ≤ crate.width)

theorem crate_middle_dimension (x : ℝ) :
  let crate := CrateDimensions.mk 5 x 12
  let cylinder := Cylinder.mk 5 12
  cylinderFitsUpright crate cylinder → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_crate_middle_dimension_l645_64552


namespace NUMINAMATH_CALUDE_cos_five_pi_thirds_plus_two_alpha_l645_64557

theorem cos_five_pi_thirds_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π/6 - α) = 2/3) : 
  Real.cos (5*π/3 + 2*α) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_five_pi_thirds_plus_two_alpha_l645_64557


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_complements_A_B_l645_64549

-- Define the universal set U
def U : Set Nat := {x | 1 ≤ x ∧ x ≤ 10}

-- Define sets A and B
def A : Set Nat := {1, 2, 3, 5, 8}
def B : Set Nat := {1, 3, 5, 7, 9}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 3, 5} := by sorry

-- Theorem for the intersection of complements of A and B
theorem intersection_complements_A_B : (U \ A) ∩ (U \ B) = {4, 6, 10} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_complements_A_B_l645_64549


namespace NUMINAMATH_CALUDE_square_of_five_equals_twentyfive_l645_64596

theorem square_of_five_equals_twentyfive : (5 : ℕ)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_five_equals_twentyfive_l645_64596


namespace NUMINAMATH_CALUDE_five_apples_ten_oranges_baskets_l645_64568

/-- Represents the number of different fruit baskets that can be made -/
def fruitBaskets (apples oranges : ℕ) : ℕ :=
  (apples + 1) * (oranges + 1) - 1

/-- Theorem stating that the number of different fruit baskets
    with 5 apples and 10 oranges is 65 -/
theorem five_apples_ten_oranges_baskets :
  fruitBaskets 5 10 = 65 := by
  sorry

end NUMINAMATH_CALUDE_five_apples_ten_oranges_baskets_l645_64568


namespace NUMINAMATH_CALUDE_special_arrangements_count_l645_64562

/-- The number of ways to arrange guests in a special circular formation. -/
def specialArrangements (n : ℕ) : ℕ :=
  (3 * n).factorial

/-- The main theorem stating that the number of special arrangements is (3n)! -/
theorem special_arrangements_count (n : ℕ) :
  specialArrangements n = (3 * n).factorial := by
  sorry

end NUMINAMATH_CALUDE_special_arrangements_count_l645_64562


namespace NUMINAMATH_CALUDE_root_in_interval_l645_64599

theorem root_in_interval :
  ∃ x₀ ∈ Set.Ioo (1/2 : ℝ) 1, Real.exp x₀ = 3 - 2 * x₀ := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l645_64599


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l645_64535

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  Complex.im ((2 - i) ^ 2) = -4 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l645_64535


namespace NUMINAMATH_CALUDE_average_increase_is_three_l645_64560

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  averageRuns : Rat

/-- Calculates the increase in average runs after a new inning -/
def averageIncrease (prev : BatsmanPerformance) (newRuns : Nat) (newAverage : Rat) : Rat :=
  newAverage - prev.averageRuns

/-- Theorem: The increase in the batsman's average is 3 runs -/
theorem average_increase_is_three 
  (prev : BatsmanPerformance) 
  (h1 : prev.innings = 16) 
  (h2 : (prev.totalRuns + 87 : Rat) / 17 = 39) : 
  averageIncrease prev 87 39 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_increase_is_three_l645_64560


namespace NUMINAMATH_CALUDE_fraction_multiplication_l645_64547

theorem fraction_multiplication (a b : ℝ) (h : a ≠ b) : 
  (3*a * 3*b) / (3*a - 3*b) = 3 * (a*b / (a - b)) := by
sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l645_64547


namespace NUMINAMATH_CALUDE_arman_age_to_40_l645_64513

/-- Given that Arman is six times older than his sister and his sister was 2 years old four years ago,
    prove that Arman will be 40 years old in 4 years. -/
theorem arman_age_to_40 (arman_age sister_age : ℕ) : 
  sister_age = 2 + 4 →  -- Sister's current age
  arman_age = 6 * sister_age →  -- Arman's current age
  40 - arman_age = 4 :=
by sorry

end NUMINAMATH_CALUDE_arman_age_to_40_l645_64513


namespace NUMINAMATH_CALUDE_larger_number_problem_l645_64571

theorem larger_number_problem (smaller larger : ℕ) : 
  larger - smaller = 1365 →
  larger = 6 * smaller + 35 →
  larger = 1631 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l645_64571


namespace NUMINAMATH_CALUDE_triangular_prism_no_body_diagonal_l645_64543

-- Define what a prism is
structure Prism where
  base : Type
  has_base_diagonal : Bool

-- Define the property of having a body diagonal
def has_body_diagonal (p : Prism) : Bool := p.has_base_diagonal

-- Define specific types of prisms
def triangular_prism : Prism := { base := Unit, has_base_diagonal := false }

-- Theorem statement
theorem triangular_prism_no_body_diagonal : 
  ¬(has_body_diagonal triangular_prism) := by sorry

end NUMINAMATH_CALUDE_triangular_prism_no_body_diagonal_l645_64543


namespace NUMINAMATH_CALUDE_find_x_l645_64582

theorem find_x : ∃ x : ℝ, 121 * x = 75625 ∧ x = 625 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l645_64582


namespace NUMINAMATH_CALUDE_initial_puppies_count_l645_64589

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has now -/
def puppies_remaining : ℕ := 5

/-- The initial number of puppies Alyssa had -/
def initial_puppies : ℕ := puppies_given_away + puppies_remaining

theorem initial_puppies_count : initial_puppies = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l645_64589


namespace NUMINAMATH_CALUDE_line_equation_proof_l645_64561

/-- A line that passes through a point and intersects both axes -/
structure IntersectingLine where
  -- The point through which the line passes
  P : ℝ × ℝ
  -- The point where the line intersects the x-axis
  A : ℝ × ℝ
  -- The point where the line intersects the y-axis
  B : ℝ × ℝ
  -- Ensure A is on the x-axis
  hA : A.2 = 0
  -- Ensure B is on the y-axis
  hB : B.1 = 0
  -- Ensure P is the midpoint of AB
  hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- The equation of the line is 3x - 2y + 24 = 0 -/
def lineEquation (l : IntersectingLine) : Prop :=
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | 3 * p.1 - 2 * p.2 + 24 = 0} ↔ 
    ∃ t : ℝ, (x, y) = (1 - t) • l.A + t • l.B

/-- The main theorem -/
theorem line_equation_proof (l : IntersectingLine) (h : l.P = (-4, 6)) : 
  lineEquation l := by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l645_64561


namespace NUMINAMATH_CALUDE_trolley_passengers_third_stop_l645_64501

/-- Proves the number of people who got off on the third stop of a trolley ride --/
theorem trolley_passengers_third_stop 
  (initial_passengers : ℕ) 
  (second_stop_off : ℕ) 
  (second_stop_on_multiplier : ℕ) 
  (third_stop_on : ℕ) 
  (final_passengers : ℕ) 
  (h1 : initial_passengers = 10)
  (h2 : second_stop_off = 3)
  (h3 : second_stop_on_multiplier = 2)
  (h4 : third_stop_on = 2)
  (h5 : final_passengers = 12) :
  initial_passengers - second_stop_off + second_stop_on_multiplier * initial_passengers - 
  (initial_passengers - second_stop_off + second_stop_on_multiplier * initial_passengers + third_stop_on - final_passengers) = 17 := by
  sorry


end NUMINAMATH_CALUDE_trolley_passengers_third_stop_l645_64501


namespace NUMINAMATH_CALUDE_max_distance_between_functions_l645_64505

theorem max_distance_between_functions : ∃ (C : ℝ),
  C = Real.sqrt 5 ∧ 
  ∀ x : ℝ, |2 * Real.sin x - Real.sin (π / 2 - x)| ≤ C ∧
  ∃ x₀ : ℝ, |2 * Real.sin x₀ - Real.sin (π / 2 - x₀)| = C :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_functions_l645_64505


namespace NUMINAMATH_CALUDE_circle_center_correct_l645_64523

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - 2 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (1, -1)

/-- Theorem: The center of the circle defined by CircleEquation is CircleCenter -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l645_64523


namespace NUMINAMATH_CALUDE_cubic_function_equality_l645_64585

/-- Given two cubic functions f and g, prove that f(g(x)) = g(f(x)) for all x if and only if d = ±a -/
theorem cubic_function_equality (a b c d e f : ℝ) :
  (∀ x : ℝ, (a * (d * x^3 + e * x + f)^3 + b * (d * x^3 + e * x + f) + c) = 
            (d * (a * x^3 + b * x + c)^3 + e * (a * x^3 + b * x + c) + f)) ↔ 
  (d = a ∨ d = -a) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_equality_l645_64585


namespace NUMINAMATH_CALUDE_eleven_distinct_points_l645_64593

/-- Represents a circular track with a cyclist and pedestrian -/
structure Track where
  length : ℝ
  pedestrian_speed : ℝ
  cyclist_speed : ℝ
  (cyclist_faster : cyclist_speed = pedestrian_speed * 1.55)
  (positive_speed : pedestrian_speed > 0)

/-- Calculates the number of distinct overtaking points on the track -/
def distinct_overtaking_points (track : Track) : ℕ :=
  sorry

/-- Theorem stating that there are 11 distinct overtaking points -/
theorem eleven_distinct_points (track : Track) :
  distinct_overtaking_points track = 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_distinct_points_l645_64593


namespace NUMINAMATH_CALUDE_circles_intersect_iff_l645_64598

/-- Two circles C1 and C2 in the plane -/
structure TwoCircles where
  /-- The parameter a for the second circle -/
  a : ℝ
  /-- a is positive -/
  a_pos : a > 0

/-- The condition for the two circles to intersect -/
def intersect (c : TwoCircles) : Prop :=
  3 < c.a ∧ c.a < 5

/-- Theorem stating the necessary and sufficient condition for the circles to intersect -/
theorem circles_intersect_iff (c : TwoCircles) :
  (∃ (x y : ℝ), x^2 + (y-1)^2 = 1 ∧ (x-c.a)^2 + (y-1)^2 = 16) ↔ intersect c := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_iff_l645_64598


namespace NUMINAMATH_CALUDE_both_activities_count_l645_64538

/-- Represents a group of people with preferences for reading books and listening to songs -/
structure GroupPreferences where
  total : ℕ
  book_lovers : ℕ
  song_lovers : ℕ
  both_lovers : ℕ

/-- The principle of inclusion-exclusion for two sets -/
def inclusion_exclusion (g : GroupPreferences) : Prop :=
  g.total = g.book_lovers + g.song_lovers - g.both_lovers

/-- Theorem stating the number of people who like both activities -/
theorem both_activities_count (g : GroupPreferences) 
  (h1 : g.total = 100)
  (h2 : g.book_lovers = 50)
  (h3 : g.song_lovers = 70)
  (h4 : inclusion_exclusion g) : 
  g.both_lovers = 20 := by
  sorry


end NUMINAMATH_CALUDE_both_activities_count_l645_64538


namespace NUMINAMATH_CALUDE_vectors_form_basis_l645_64502

theorem vectors_form_basis (a b : ℝ × ℝ) : 
  a = (1, -2) ∧ b = (3, 5) → 
  (∃ (x y : ℝ), ∀ v : ℝ × ℝ, v = x • a + y • b) ∧ 
  (¬ ∃ (k : ℝ), a = k • b) :=
sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l645_64502


namespace NUMINAMATH_CALUDE_max_pencils_buyable_l645_64580

def total_money : ℚ := 36
def pencil_cost : ℚ := 1.80
def pen_cost : ℚ := 2.60
def num_pens : ℕ := 9

theorem max_pencils_buyable :
  ∃ (num_pencils : ℕ),
    (num_pencils * pencil_cost + num_pens * pen_cost ≤ total_money) ∧
    ((num_pencils + num_pens) % 3 = 0) ∧
    (∀ (n : ℕ), n > num_pencils →
      (n * pencil_cost + num_pens * pen_cost > total_money ∨
       (n + num_pens) % 3 ≠ 0)) ∧
    num_pencils = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_pencils_buyable_l645_64580


namespace NUMINAMATH_CALUDE_g_minus_one_eq_zero_l645_64511

/-- The function g(x) as defined in the problem -/
def g (x s : ℝ) : ℝ := 3 * x^5 - 2 * x^3 + x^2 - 4 * x + s

/-- Theorem stating that g(-1) = 0 when s = -4 -/
theorem g_minus_one_eq_zero :
  g (-1) (-4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_minus_one_eq_zero_l645_64511


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l645_64573

theorem quadratic_inequality_solution_sets 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 + a*x + b < 0}) :
  {x : ℝ | b*x^2 + a*x + 1 > 0} = Set.Iic (1/3) ∪ Set.Ioi (1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l645_64573


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l645_64540

theorem power_fraction_simplification :
  (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l645_64540


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l645_64590

theorem perfect_square_trinomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + a*x + 81 = (x + b)^2) → (a = 18 ∨ a = -18) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l645_64590


namespace NUMINAMATH_CALUDE_chess_master_exhibition_l645_64526

theorem chess_master_exhibition (x : ℝ) 
  (h1 : 0.1 * x + 8 + 0.1 * (0.9 * x - 8) + 2 + 7 = x) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_chess_master_exhibition_l645_64526


namespace NUMINAMATH_CALUDE_inequality_and_function_property_l645_64512

def f (x : ℝ) := |x - 1|

theorem inequality_and_function_property :
  (∀ x : ℝ, f x + f (x + 4) ≥ 8 ↔ x ≤ -5 ∨ x ≥ 3) ∧
  (∀ a b : ℝ, |a| < 1 → |b| < 1 → a ≠ 0 → f (a * b) > |a| * f (b / a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_function_property_l645_64512


namespace NUMINAMATH_CALUDE_positive_numbers_inequalities_l645_64550

theorem positive_numbers_inequalities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c = 1 → (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c) ∧
  (∃ r : ℝ, r > 0 ∧ b = a * r ∧ c = b * r → a^2 + b^2 + c^2 > (a - b + c)^2) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequalities_l645_64550


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l645_64542

theorem trigonometric_inequality (a b A G : ℝ) : 
  a = Real.sin (π / 3) →
  b = Real.cos (π / 3) →
  A = (a + b) / 2 →
  G = Real.sqrt (a * b) →
  b < G ∧ G < A ∧ A < a :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l645_64542


namespace NUMINAMATH_CALUDE_functional_equation_solution_l645_64525

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x - (1/2) * f (x/2) = x^2

/-- The theorem stating that the function satisfying the equation is f(x) = (8/7) * x^2 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f = fun x ↦ (8/7) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l645_64525


namespace NUMINAMATH_CALUDE_tournament_has_24_players_l645_64537

/-- Represents a tournament with the given conditions --/
structure Tournament where
  n : ℕ  -- Total number of players
  pointsAgainstLowest12 : ℕ → ℚ  -- Points each player earned against the lowest 12
  totalPoints : ℕ → ℚ  -- Total points of each player

/-- The conditions of the tournament --/
def tournamentConditions (t : Tournament) : Prop :=
  -- Each player plays against every other player
  ∀ i, t.totalPoints i ≤ (t.n - 1 : ℚ)
  -- Half of each player's points are from the lowest 12
  ∧ ∀ i, 2 * t.pointsAgainstLowest12 i = t.totalPoints i
  -- There are exactly 12 lowest-scoring players
  ∧ ∃ lowest12 : Finset ℕ, lowest12.card = 12 
    ∧ ∀ i ∈ lowest12, ∀ j ∉ lowest12, t.totalPoints i ≤ t.totalPoints j

/-- The theorem stating that the tournament has 24 players --/
theorem tournament_has_24_players (t : Tournament) 
  (h : tournamentConditions t) : t.n = 24 :=
sorry

end NUMINAMATH_CALUDE_tournament_has_24_players_l645_64537


namespace NUMINAMATH_CALUDE_boys_camp_total_l645_64517

theorem boys_camp_total (total : ℝ) 
  (h1 : 0.2 * total = total_school_A)
  (h2 : 0.3 * total_school_A = science_school_A)
  (h3 : total_school_A - science_school_A = 77) : 
  total = 550 := by
sorry

end NUMINAMATH_CALUDE_boys_camp_total_l645_64517


namespace NUMINAMATH_CALUDE_remainder_problem_l645_64597

theorem remainder_problem (n : ℤ) (h : n % 7 = 3) : (4 * n - 9) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l645_64597


namespace NUMINAMATH_CALUDE_alphabet_theorem_l645_64556

theorem alphabet_theorem (total : ℕ) (both : ℕ) (line_only : ℕ) 
  (h1 : total = 76) 
  (h2 : both = 20) 
  (h3 : line_only = 46) 
  (h4 : total = both + line_only + (total - (both + line_only))) :
  total - (both + line_only) = 30 := by
  sorry

end NUMINAMATH_CALUDE_alphabet_theorem_l645_64556


namespace NUMINAMATH_CALUDE_parallelogram_area_l645_64578

/-- The area of a parallelogram, given the area of a triangle formed by its diagonal -/
theorem parallelogram_area (triangle_area : ℝ) (h : triangle_area = 64) : 
  2 * triangle_area = 128 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l645_64578


namespace NUMINAMATH_CALUDE_flag_paint_cost_l645_64572

-- Define constants
def flag_width : Real := 3.5
def flag_height : Real := 2.5
def paint_cost_per_quart : Real := 4
def paint_coverage_per_quart : Real := 4
def sq_ft_per_sq_m : Real := 10.7639

-- Define the theorem
theorem flag_paint_cost : 
  let flag_area := flag_width * flag_height
  let total_area := 2 * flag_area
  let total_area_sq_ft := total_area * sq_ft_per_sq_m
  let quarts_needed := ⌈total_area_sq_ft / paint_coverage_per_quart⌉
  let total_cost := quarts_needed * paint_cost_per_quart
  total_cost = 192 := by
  sorry


end NUMINAMATH_CALUDE_flag_paint_cost_l645_64572


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l645_64545

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for the range of a when B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l645_64545


namespace NUMINAMATH_CALUDE_knight_same_color_probability_l645_64554

/-- Represents the colors of the chessboard squares -/
inductive ChessColor
| Red
| Green
| Blue

/-- Represents a square on the chessboard -/
structure ChessSquare where
  row : Fin 8
  col : Fin 8
  color : ChessColor

/-- The chessboard with its colored squares -/
def chessboard : Array ChessSquare := sorry

/-- Determines if a knight's move is legal -/
def isLegalKnightMove (start finish : ChessSquare) : Bool := sorry

/-- Calculates the probability of a knight landing on the same color after one move -/
def knightSameColorProbability (board : Array ChessSquare) : ℚ := sorry

/-- The main theorem to prove -/
theorem knight_same_color_probability :
  knightSameColorProbability chessboard = 1/2 := by sorry

end NUMINAMATH_CALUDE_knight_same_color_probability_l645_64554


namespace NUMINAMATH_CALUDE_prob_one_two_given_different_l645_64534

/-- The probability of at least one die showing 2, given that two fair dice show different numbers -/
theorem prob_one_two_given_different : ℝ := by
  -- Define the sample space of outcomes where the two dice show different numbers
  let different_outcomes : Finset (ℕ × ℕ) := sorry

  -- Define the event of at least one die showing 2, given different numbers
  let event_one_two : Finset (ℕ × ℕ) := sorry

  -- Define the probability measure
  let prob : Finset (ℕ × ℕ) → ℝ := sorry

  -- The probability is the measure of the event divided by the measure of the sample space
  have h : prob event_one_two / prob different_outcomes = 1 / 3 := by sorry

  exact 1 / 3

end NUMINAMATH_CALUDE_prob_one_two_given_different_l645_64534


namespace NUMINAMATH_CALUDE_x_equals_seven_l645_64583

theorem x_equals_seven (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 7 * x^3 + 14 * x^2 * y = x^4 + 2 * x^3 * y) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_seven_l645_64583


namespace NUMINAMATH_CALUDE_quadratic_roots_value_l645_64551

theorem quadratic_roots_value (x₁ x₂ m : ℝ) : 
  (∀ x, x^2 - 8*x + m = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ = 3*x₂ →
  m = 12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_value_l645_64551


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l645_64553

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l645_64553


namespace NUMINAMATH_CALUDE_nancy_tortilla_chips_nancy_final_chips_l645_64506

/-- Calculates the number of tortilla chips Nancy has left after sharing with her family members -/
theorem nancy_tortilla_chips (initial_chips : ℝ) (brother_chips : ℝ) 
  (sister_fraction : ℝ) (cousin_percent : ℝ) : ℝ :=
  let remaining_after_brother := initial_chips - brother_chips
  let sister_chips := sister_fraction * remaining_after_brother
  let remaining_after_sister := remaining_after_brother - sister_chips
  let cousin_chips := (cousin_percent / 100) * remaining_after_sister
  let final_chips := remaining_after_sister - cousin_chips
  final_chips

/-- Proves that Nancy has 18.75 tortilla chips left for herself -/
theorem nancy_final_chips : 
  nancy_tortilla_chips 50 12.5 (1/3) 25 = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_nancy_tortilla_chips_nancy_final_chips_l645_64506


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l645_64527

theorem sum_of_roots_quadratic : ∃ (x₁ x₂ : ℝ), 
  x₁^2 - 7*x₁ + 10 = 0 ∧ 
  x₂^2 - 7*x₂ + 10 = 0 ∧ 
  x₁ + x₂ = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l645_64527


namespace NUMINAMATH_CALUDE_total_weight_is_130_l645_64510

-- Define the weights as real numbers
variable (M D C : ℝ)

-- State the conditions
variable (h1 : D + C = 60)
variable (h2 : C = (1/5) * M)
variable (h3 : D = 46)

-- Theorem to prove
theorem total_weight_is_130 : M + D + C = 130 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_130_l645_64510


namespace NUMINAMATH_CALUDE_complex_distance_range_l645_64541

theorem complex_distance_range (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ Complex.abs (1 + Complex.I * Real.sqrt 3 + z) = x :=
sorry

end NUMINAMATH_CALUDE_complex_distance_range_l645_64541


namespace NUMINAMATH_CALUDE_quadratic_sum_l645_64515

/-- Given a quadratic function g(x) = dx^2 + ex + f, 
    if g(0) = 8 and g(1) = 5, then d + e + 2f = 13 -/
theorem quadratic_sum (d e f : ℝ) : 
  let g : ℝ → ℝ := λ x ↦ d * x^2 + e * x + f
  (g 0 = 8) → (g 1 = 5) → d + e + 2 * f = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l645_64515


namespace NUMINAMATH_CALUDE_unique_birth_date_l645_64592

/-- Represents a date in the 20th century -/
structure Date where
  day : Nat
  month : Nat
  year : Nat
  h1 : 1 ≤ day ∧ day ≤ 31
  h2 : 1 ≤ month ∧ month ≤ 12
  h3 : 1900 ≤ year ∧ year ≤ 1999

def date_to_number (d : Date) : Nat :=
  d.day * 10000 + d.month * 100 + (d.year - 1900)

/-- The birth dates of two friends satisfy the given conditions -/
def valid_birth_dates (d1 d2 : Date) : Prop :=
  d1.month = d2.month ∧
  d1.year = d2.year ∧
  d2.day = d1.day + 7 ∧
  date_to_number d2 = 6 * date_to_number d1

theorem unique_birth_date :
  ∃! d : Date, ∃ d2 : Date, valid_birth_dates d d2 ∧ d.day = 1 ∧ d.month = 4 ∧ d.year = 1900 :=
by sorry

end NUMINAMATH_CALUDE_unique_birth_date_l645_64592


namespace NUMINAMATH_CALUDE_range_of_a_l645_64504

-- Define set A
def A : Set ℝ := {x | x * (4 - x) ≥ 3}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = A → a < 1 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_range_of_a_l645_64504


namespace NUMINAMATH_CALUDE_polynomial_expansion_l645_64529

theorem polynomial_expansion (x : ℝ) : 
  (2*x^2 + 5*x + 8)*(x+1) - (x+1)*(x^2 - 2*x + 50) + (3*x - 7)*(x+1)*(x - 2) = 
  4*x^3 - 2*x^2 - 34*x - 28 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l645_64529


namespace NUMINAMATH_CALUDE_semicircle_arc_length_l645_64533

-- Define the right triangle with inscribed semicircle
structure RightTriangleWithSemicircle where
  -- Hypotenuse segments
  a : ℝ
  b : ℝ
  -- Assumption: a and b are positive
  ha : a > 0
  hb : b > 0
  -- Assumption: The semicircle is inscribed in the right triangle
  -- with its diameter on the hypotenuse

-- Define the theorem
theorem semicircle_arc_length 
  (triangle : RightTriangleWithSemicircle) 
  (h_a : triangle.a = 30) 
  (h_b : triangle.b = 40) : 
  ∃ (arc_length : ℝ), arc_length = 12 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_semicircle_arc_length_l645_64533


namespace NUMINAMATH_CALUDE_integer_subset_condition_l645_64522

theorem integer_subset_condition (a b : ℤ) : 
  (a * b * (a - b) ≠ 0) →
  (∃ (Z₀ : Set ℤ), ∀ (n : ℤ), (n ∈ Z₀ ∨ (n + a) ∈ Z₀ ∨ (n + b) ∈ Z₀) ∧ 
    ¬(n ∈ Z₀ ∧ (n + a) ∈ Z₀) ∧ ¬(n ∈ Z₀ ∧ (n + b) ∈ Z₀) ∧ ¬((n + a) ∈ Z₀ ∧ (n + b) ∈ Z₀)) ↔
  (∃ (k y z : ℤ), a = k * y ∧ b = k * z ∧ y % 3 ≠ 0 ∧ z % 3 ≠ 0 ∧ (y - z) % 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_integer_subset_condition_l645_64522


namespace NUMINAMATH_CALUDE_carpet_dimensions_l645_64528

/-- Represents a rectangular carpet -/
structure Carpet where
  length : ℕ
  width : ℕ

/-- Represents a rectangular room -/
structure Room where
  length : ℕ
  width : ℕ

/-- Check if a carpet fits perfectly in a room -/
def fits_perfectly (c : Carpet) (r : Room) : Prop :=
  c.length * c.length + c.width * c.width = r.length * r.length + r.width * r.width

/-- The main theorem -/
theorem carpet_dimensions (c : Carpet) (r1 r2 : Room) (h1 : r1.length = r2.length)
  (h2 : r1.width = 38) (h3 : r2.width = 50) (h4 : fits_perfectly c r1) (h5 : fits_perfectly c r2) :
  c.length = 25 ∧ c.width = 50 := by
  sorry

#check carpet_dimensions

end NUMINAMATH_CALUDE_carpet_dimensions_l645_64528


namespace NUMINAMATH_CALUDE_cubic_root_equation_l645_64555

theorem cubic_root_equation (y : ℝ) : 
  (1 + Real.sqrt (2 * y - 3)) ^ (1/3 : ℝ) = 3 → y = 339.5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_l645_64555


namespace NUMINAMATH_CALUDE_expression_evaluation_l645_64588

theorem expression_evaluation (a b : ℤ) (h1 : a = 3) (h2 : b = 2) : 
  (a^2 + a*b + b^2)^2 - (a^2 - a*b + b^2)^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l645_64588


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l645_64584

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 + 4*x + m + 5 = 0) ↔ m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l645_64584


namespace NUMINAMATH_CALUDE_fibonacci_unique_triple_l645_64500

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def satisfies_conditions (a b c : ℕ) : Prop :=
  b < a ∧ c < a ∧ ∀ n, (fibonacci n - n * b * c^n) % a = 0

theorem fibonacci_unique_triple :
  ∃! (triple : ℕ × ℕ × ℕ), 
    let (a, b, c) := triple
    satisfies_conditions a b c ∧ a = 5 ∧ b = 2 ∧ c = 3 :=
sorry

end NUMINAMATH_CALUDE_fibonacci_unique_triple_l645_64500


namespace NUMINAMATH_CALUDE_nelly_outbid_joe_l645_64544

def joes_bid : ℕ := 160000
def nellys_bid : ℕ := 482000

theorem nelly_outbid_joe : nellys_bid - 3 * joes_bid = 2000 := by
  sorry

end NUMINAMATH_CALUDE_nelly_outbid_joe_l645_64544


namespace NUMINAMATH_CALUDE_restaurant_additional_hamburgers_l645_64566

/-- The number of additional hamburgers made by a restaurant -/
def additional_hamburgers (initial : ℝ) (final : ℝ) : ℝ :=
  final - initial

/-- Proof that the restaurant made 3 additional hamburgers -/
theorem restaurant_additional_hamburgers : 
  let initial_hamburgers : ℝ := 9.0
  let final_hamburgers : ℝ := 12.0
  additional_hamburgers initial_hamburgers final_hamburgers = 3 := by
sorry

end NUMINAMATH_CALUDE_restaurant_additional_hamburgers_l645_64566


namespace NUMINAMATH_CALUDE_davonte_mercedes_difference_l645_64507

/-- Proves that Davonte ran 2 kilometers farther than Mercedes -/
theorem davonte_mercedes_difference (jonathan_distance : ℝ) 
  (h1 : jonathan_distance = 7.5)
  (mercedes_distance : ℝ) 
  (h2 : mercedes_distance = 2 * jonathan_distance)
  (davonte_distance : ℝ)
  (h3 : mercedes_distance + davonte_distance = 32) :
  davonte_distance - mercedes_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_davonte_mercedes_difference_l645_64507


namespace NUMINAMATH_CALUDE_final_value_calculation_l645_64531

theorem final_value_calculation (initial_number : ℕ) : 
  initial_number = 10 → 3 * (2 * initial_number + 8) = 84 := by
  sorry

end NUMINAMATH_CALUDE_final_value_calculation_l645_64531


namespace NUMINAMATH_CALUDE_stratified_sample_school_b_l645_64559

/-- Represents the number of students in each school -/
structure SchoolPopulation where
  a : ℕ  -- Number of students in school A
  b : ℕ  -- Number of students in school B
  c : ℕ  -- Number of students in school C

/-- The total number of students across all three schools -/
def total_students : ℕ := 1500

/-- The size of the sample to be drawn -/
def sample_size : ℕ := 120

/-- Checks if the given school population forms an arithmetic sequence -/
def is_arithmetic_sequence (pop : SchoolPopulation) : Prop :=
  pop.b - pop.a = pop.c - pop.b

/-- Checks if the given school population sums to the total number of students -/
def is_valid_population (pop : SchoolPopulation) : Prop :=
  pop.a + pop.b + pop.c = total_students

/-- Calculates the number of students to be sampled from a given school -/
def stratified_sample_size (school_size : ℕ) : ℕ :=
  school_size * sample_size / total_students

/-- The main theorem: proves that the number of students to be sampled from school B is 40 -/
theorem stratified_sample_school_b :
  ∀ pop : SchoolPopulation,
  is_arithmetic_sequence pop →
  is_valid_population pop →
  stratified_sample_size pop.b = 40 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_school_b_l645_64559


namespace NUMINAMATH_CALUDE_fraction_equality_l645_64564

theorem fraction_equality : (81081 : ℝ)^4 / (27027 : ℝ)^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l645_64564


namespace NUMINAMATH_CALUDE_second_equation_value_l645_64567

theorem second_equation_value (x y : ℝ) 
  (eq1 : 2 * x + y = 26) 
  (eq2 : (x + y) / 3 = 4) : 
  x + 2 * y = 10 := by
sorry

end NUMINAMATH_CALUDE_second_equation_value_l645_64567


namespace NUMINAMATH_CALUDE_solid_is_frustum_l645_64503

/-- A solid with specified view characteristics -/
structure Solid where
  top_view : Bool
  bottom_view : Bool
  front_view : Bool
  side_view : Bool

/-- Definition of a frustum based on its views -/
def is_frustum (s : Solid) : Prop :=
  s.top_view = true ∧ 
  s.bottom_view = true ∧ 
  s.front_view = true ∧ 
  s.side_view = true

/-- Theorem: A solid with circular top and bottom views, and trapezoidal front and side views, is a frustum -/
theorem solid_is_frustum (s : Solid) 
  (h_top : s.top_view = true)
  (h_bottom : s.bottom_view = true)
  (h_front : s.front_view = true)
  (h_side : s.side_view = true) : 
  is_frustum s := by
  sorry

end NUMINAMATH_CALUDE_solid_is_frustum_l645_64503


namespace NUMINAMATH_CALUDE_intersection_M_N_l645_64569

def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}
def N : Set ℝ := {x : ℝ | 1/3 ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l645_64569


namespace NUMINAMATH_CALUDE_rectangular_field_width_l645_64530

theorem rectangular_field_width :
  ∀ (width length perimeter : ℝ),
    length = (7 / 5) * width →
    perimeter = 2 * length + 2 * width →
    perimeter = 360 →
    width = 75 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l645_64530


namespace NUMINAMATH_CALUDE_range_of_a_l645_64521

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 2 < 0) ↔ -8 < a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l645_64521


namespace NUMINAMATH_CALUDE_danny_wrappers_found_l645_64508

theorem danny_wrappers_found (initial_caps : ℕ) (found_caps : ℕ) (total_caps : ℕ) (total_wrappers : ℕ) 
  (h1 : initial_caps = 6)
  (h2 : found_caps = 22)
  (h3 : total_caps = 28)
  (h4 : total_wrappers = 63)
  (h5 : found_caps = total_caps - initial_caps)
  : ∃ (found_wrappers : ℕ), found_wrappers = 22 ∧ total_wrappers ≥ found_wrappers :=
by
  sorry

#check danny_wrappers_found

end NUMINAMATH_CALUDE_danny_wrappers_found_l645_64508


namespace NUMINAMATH_CALUDE_linear_system_sum_a_d_l645_64514

theorem linear_system_sum_a_d :
  ∀ (a b c d e : ℝ),
    a + b = 14 →
    b + c = 9 →
    c + d = 3 →
    d + e = 6 →
    a - 2 * e = 1 →
    a + d = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_linear_system_sum_a_d_l645_64514

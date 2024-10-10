import Mathlib

namespace smallest_number_proof_l3060_306099

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_single : a < 10
  b_single : b < 10
  c_single : c < 10
  d_single : d < 10

/-- Checks if the given four-digit number satisfies the product conditions -/
def satisfies_conditions (n : FourDigitNumber) : Prop :=
  (n.a * n.b = 21 ∧ n.b * n.c = 20) ∨
  (n.a * n.b = 21 ∧ n.c * n.d = 20) ∨
  (n.b * n.c = 21 ∧ n.c * n.d = 20)

/-- The smallest four-digit number satisfying the conditions -/
def smallest_satisfying_number : FourDigitNumber :=
  { a := 3, b := 7, c := 4, d := 5,
    a_single := by norm_num,
    b_single := by norm_num,
    c_single := by norm_num,
    d_single := by norm_num }

theorem smallest_number_proof :
  satisfies_conditions smallest_satisfying_number ∧
  ∀ n : FourDigitNumber, satisfies_conditions n →
    n.a * 1000 + n.b * 100 + n.c * 10 + n.d ≥
    smallest_satisfying_number.a * 1000 +
    smallest_satisfying_number.b * 100 +
    smallest_satisfying_number.c * 10 +
    smallest_satisfying_number.d :=
by sorry

end smallest_number_proof_l3060_306099


namespace reservoir_percentage_before_storm_l3060_306020

-- Define the reservoir capacity in billion gallons
def reservoir_capacity : ℝ := 550

-- Define the original contents in billion gallons
def original_contents : ℝ := 220

-- Define the amount of water added by the storm in billion gallons
def storm_water : ℝ := 110

-- Define the percentage full after the storm
def post_storm_percentage : ℝ := 0.60

-- Theorem to prove
theorem reservoir_percentage_before_storm :
  (original_contents / reservoir_capacity) * 100 = 40 :=
by
  sorry

end reservoir_percentage_before_storm_l3060_306020


namespace breakfast_egg_scramble_time_l3060_306089

/-- Calculates the time to scramble each egg given the breakfast preparation parameters. -/
def time_to_scramble_egg (num_sausages : ℕ) (num_eggs : ℕ) (time_per_sausage : ℕ) (total_time : ℕ) : ℕ :=
  let time_for_sausages := num_sausages * time_per_sausage
  let time_for_eggs := total_time - time_for_sausages
  time_for_eggs / num_eggs

/-- Proves that the time to scramble each egg is 4 minutes given the specific breakfast parameters. -/
theorem breakfast_egg_scramble_time :
  time_to_scramble_egg 3 6 5 39 = 4 := by
  sorry

end breakfast_egg_scramble_time_l3060_306089


namespace john_change_proof_l3060_306093

/-- Calculates the change received when buying oranges -/
def calculate_change (num_oranges : ℕ) (cost_per_orange_cents : ℕ) (paid_dollars : ℕ) : ℚ :=
  paid_dollars - (num_oranges * cost_per_orange_cents) / 100

theorem john_change_proof :
  calculate_change 4 75 10 = 7 := by
  sorry

#eval calculate_change 4 75 10

end john_change_proof_l3060_306093


namespace star_polygon_angle_sum_l3060_306044

/-- Represents a star polygon created from an n-sided convex polygon. -/
structure StarPolygon where
  n : ℕ
  h_n : n ≥ 6

/-- Calculates the sum of internal angles at the intersections of a star polygon. -/
def sum_of_internal_angles (sp : StarPolygon) : ℝ :=
  180 * (sp.n - 4)

/-- Theorem stating that the sum of internal angles at the intersections
    of a star polygon is 180(n-4) degrees. -/
theorem star_polygon_angle_sum (sp : StarPolygon) :
  sum_of_internal_angles sp = 180 * (sp.n - 4) := by
  sorry

end star_polygon_angle_sum_l3060_306044


namespace remaining_cookies_l3060_306087

theorem remaining_cookies (white_initial : ℕ) (black_initial : ℕ) : 
  white_initial = 80 →
  black_initial = white_initial + 50 →
  (white_initial - (3 * white_initial / 4)) + (black_initial / 2) = 85 := by
  sorry

end remaining_cookies_l3060_306087


namespace smallest_value_complex_sum_l3060_306030

theorem smallest_value_complex_sum (a b c : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_omega_cube : ω^3 = 1)
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ 
  (∀ (x y z : ℤ) (h_xyz_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z),
    Complex.abs (↑x + ↑y * ω + ↑z * ω^2) ≥ m) ∧
  (∃ (p q r : ℤ) (h_pqr_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r),
    Complex.abs (↑p + ↑q * ω + ↑r * ω^2) = m) :=
by sorry

end smallest_value_complex_sum_l3060_306030


namespace simplify_expression_l3060_306097

theorem simplify_expression (z : ℝ) : (7 - Real.sqrt (z^2 - 49))^2 = z^2 - 14 * Real.sqrt (z^2 - 49) := by
  sorry

end simplify_expression_l3060_306097


namespace total_sheep_l3060_306066

theorem total_sheep (aaron_sheep beth_sheep : ℕ) 
  (h1 : aaron_sheep = 532)
  (h2 : beth_sheep = 76)
  (h3 : aaron_sheep = 7 * beth_sheep) : 
  aaron_sheep + beth_sheep = 608 := by
sorry

end total_sheep_l3060_306066


namespace only_newborn_babies_is_set_l3060_306007

-- Define a type for statements
inductive Statement
| NewbornBabies
| VerySmallNumbers
| HealthyStudents
| CutePandas

-- Define a function to check if a statement satisfies definiteness
def satisfiesDefiniteness (s : Statement) : Prop :=
  match s with
  | Statement.NewbornBabies => true
  | _ => false

-- Theorem: Only NewbornBabies satisfies definiteness
theorem only_newborn_babies_is_set :
  ∀ s : Statement, satisfiesDefiniteness s ↔ s = Statement.NewbornBabies :=
by
  sorry


end only_newborn_babies_is_set_l3060_306007


namespace yard_sale_books_bought_l3060_306054

/-- The number of books Mike bought at a yard sale -/
def books_bought (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem: The number of books Mike bought at the yard sale is the difference between his final and initial number of books -/
theorem yard_sale_books_bought (initial_books final_books : ℕ) 
  (h : final_books ≥ initial_books) : 
  books_bought initial_books final_books = final_books - initial_books :=
by
  sorry

/-- Given Mike's initial and final number of books, calculate how many he bought -/
def mikes_books : ℕ := 
  books_bought 35 56

#eval mikes_books

end yard_sale_books_bought_l3060_306054


namespace inequality_proof_l3060_306003

theorem inequality_proof (x y : ℤ) : x * (x + 1) ≠ 2 * (5 * y + 2) := by
  sorry

end inequality_proof_l3060_306003


namespace bella_position_at_102_l3060_306078

/-- Represents a point on a 2D coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a direction on the coordinate plane -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents Bella's state at any given point -/
structure BellaState where
  position : Point
  facing : Direction
  lastMove : ℕ

/-- Defines the movement rules for Bella -/
def moveRules (n : ℕ) (state : BellaState) : BellaState :=
  sorry

/-- The main theorem to prove -/
theorem bella_position_at_102 :
  let initialState : BellaState := {
    position := { x := 0, y := 0 },
    facing := Direction.North,
    lastMove := 0
  }
  let finalState := (moveRules 102 initialState)
  finalState.position = { x := -23, y := 29 } :=
sorry

end bella_position_at_102_l3060_306078


namespace solution_equality_l3060_306052

-- Define the function F
def F (a b c : ℝ) : ℝ := a * b^3 + c

-- Theorem statement
theorem solution_equality :
  ∃ a : ℝ, F a 2 3 = F a 3 4 ∧ a = -1/19 := by
  sorry

end solution_equality_l3060_306052


namespace largest_among_a_ab_aplusb_l3060_306011

theorem largest_among_a_ab_aplusb (a b : ℚ) (h : b < 0) :
  (a - b) = max a (max (a - b) (a + b)) := by
  sorry

end largest_among_a_ab_aplusb_l3060_306011


namespace ages_solution_l3060_306086

/-- Represents the ages of Ann, Kristine, and Brad -/
structure Ages where
  ann : ℕ
  kristine : ℕ
  brad : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.ann = ages.kristine + 5 ∧
  ages.brad = ages.ann - 3 ∧
  ages.brad = 2 * ages.kristine

/-- The theorem to be proved -/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧
    ages.kristine = 2 ∧ ages.ann = 7 ∧ ages.brad = 4 := by
  sorry

end ages_solution_l3060_306086


namespace exists_quadratic_function_l3060_306070

/-- A quadratic function that fits the given points -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem exists_quadratic_function : 
  ∃ (a b c : ℝ), 
    quadratic_function a b c 1 = 1 ∧
    quadratic_function a b c 2 = 4 ∧
    quadratic_function a b c 4 = 16 ∧
    quadratic_function a b c 5 = 25 ∧
    quadratic_function a b c 7 = 49 ∧
    quadratic_function a b c 8 = 64 ∧
    quadratic_function a b c 10 = 100 :=
by
  sorry

end exists_quadratic_function_l3060_306070


namespace two_machines_total_copies_l3060_306004

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  rate : ℕ  -- copies per minute

/-- Calculates the number of copies made by a machine in a given time -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.rate * minutes

/-- Represents the problem setup with two copy machines -/
structure TwoMachinesProblem where
  machine1 : CopyMachine
  machine2 : CopyMachine
  time : ℕ  -- in minutes

/-- The main theorem to be proved -/
theorem two_machines_total_copies 
  (problem : TwoMachinesProblem) 
  (h1 : problem.machine1.rate = 25)
  (h2 : problem.machine2.rate = 55)
  (h3 : problem.time = 30) : 
  copies_made problem.machine1 problem.time + copies_made problem.machine2 problem.time = 2400 :=
by sorry

end two_machines_total_copies_l3060_306004


namespace real_part_of_z_l3060_306023

theorem real_part_of_z (z : ℂ) (h : Complex.I * z = 1 + 2 * Complex.I) : 
  Complex.re z = 2 := by
  sorry

end real_part_of_z_l3060_306023


namespace rational_equality_l3060_306021

theorem rational_equality (n : ℕ) (x y : ℚ) 
  (h_odd : Odd n) 
  (h_pos : 0 < n) 
  (h_eq : x^n + 2*y = y^n + 2*x) : 
  x = y := by sorry

end rational_equality_l3060_306021


namespace sequence_a_general_term_sequence_b_general_term_l3060_306084

-- Define the sequences
def sequence_a : ℕ → ℕ
  | 1 => 0
  | 2 => 3
  | 3 => 26
  | 4 => 255
  | 5 => 3124
  | _ => 0  -- Default case, not used in the proof

def sequence_b : ℕ → ℕ
  | 1 => 1
  | 2 => 2
  | 3 => 12
  | 4 => 288
  | 5 => 34560
  | _ => 0  -- Default case, not used in the proof

-- Define the general term for sequence a
def general_term_a (n : ℕ) : ℕ := n^n - 1

-- Define the general term for sequence b
def general_term_b (n : ℕ) : ℕ := (List.range n).foldl (λ acc i => acc * Nat.factorial (i + 1)) 1

-- Theorem for sequence a
theorem sequence_a_general_term (n : ℕ) (h : n > 0 ∧ n ≤ 5) :
  sequence_a n = general_term_a n := by
  sorry

-- Theorem for sequence b
theorem sequence_b_general_term (n : ℕ) (h : n > 0 ∧ n ≤ 5) :
  sequence_b n = general_term_b n := by
  sorry

end sequence_a_general_term_sequence_b_general_term_l3060_306084


namespace calculation_proof_l3060_306009

theorem calculation_proof :
  ((-1/3 : ℚ) - 15 + (-2/3) + 1 = -15) ∧
  (16 / (-2)^3 - (-1/8) * (-4 : ℚ) = -5/2) := by
  sorry

end calculation_proof_l3060_306009


namespace urn_probability_l3060_306065

/-- Represents the total number of chips in the urn -/
def total_chips : ℕ := 15

/-- Represents the number of chips of each color -/
def chips_per_color : ℕ := 5

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the number of chips with each number -/
def chips_per_number : ℕ := 3

/-- Represents the number of different numbers on the chips -/
def num_numbers : ℕ := 5

/-- The probability of drawing two chips with either the same color or the same number -/
theorem urn_probability : 
  (num_colors * (chips_per_color.choose 2) + num_numbers * (chips_per_number.choose 2)) / (total_chips.choose 2) = 3 / 7 :=
by sorry

end urn_probability_l3060_306065


namespace sin_cos_problem_l3060_306010

theorem sin_cos_problem (x : ℝ) (h : Real.sin x = 3 * Real.cos x) :
  Real.sin x * Real.cos x = 3 / 10 := by
  sorry

end sin_cos_problem_l3060_306010


namespace trips_per_month_l3060_306016

/-- Given a person who spends 72 hours driving in a year, with each round trip
    taking 3 hours, prove that the number of trips per month is 2. -/
theorem trips_per_month (hours_per_year : ℕ) (hours_per_trip : ℕ) 
    (months_per_year : ℕ) : ℕ :=
  by
  have h1 : hours_per_year = 72 := by sorry
  have h2 : hours_per_trip = 3 := by sorry
  have h3 : months_per_year = 12 := by sorry
  
  let trips_per_year : ℕ := hours_per_year / hours_per_trip
  
  have h4 : trips_per_year = 24 := by sorry
  
  exact trips_per_year / months_per_year

end trips_per_month_l3060_306016


namespace vote_intersection_l3060_306056

theorem vote_intersection (U A B : Finset Int) (h1 : U.card = 300) 
  (h2 : A.card = 230) (h3 : B.card = 190) (h4 : (U \ A).card + (U \ B).card - U.card = 40) :
  (A ∩ B).card = 160 := by
  sorry

end vote_intersection_l3060_306056


namespace choir_members_count_l3060_306041

theorem choir_members_count :
  ∃! n : ℕ, 150 ≤ n ∧ n ≤ 300 ∧ n % 10 = 6 ∧ n % 11 = 6 ∧ n = 226 := by
  sorry

end choir_members_count_l3060_306041


namespace tangent_and_trigonometric_identity_l3060_306095

theorem tangent_and_trigonometric_identity (α β : Real) 
  (h1 : Real.tan (α + β) = 2)
  (h2 : Real.tan (Real.pi - β) = 3/2) :
  (Real.tan α = -7/4) ∧ 
  ((Real.sin (Real.pi/2 + α) - Real.sin (Real.pi + α)) / (Real.cos α + 2 * Real.sin α) = 3/10) := by
  sorry

end tangent_and_trigonometric_identity_l3060_306095


namespace jo_equals_alex_sum_l3060_306017

def roundToNearestMultipleOf5 (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def joSum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def alexSum (n : ℕ) : ℕ :=
  (Finset.range n).sum (roundToNearestMultipleOf5 ∘ (· + 1))

theorem jo_equals_alex_sum :
  joSum 100 = alexSum 100 := by
  sorry

end jo_equals_alex_sum_l3060_306017


namespace no_solution_exists_l3060_306077

theorem no_solution_exists : ¬ ∃ (a b : ℤ), (2006 * 2006) ∣ (a^2006 + b^2006 + 1) := by
  sorry

end no_solution_exists_l3060_306077


namespace income_education_relationship_l3060_306083

/-- Represents the linear regression model for annual income and educational expenditure -/
structure IncomeEducationModel where
  -- x: annual income in ten thousand yuan
  -- y: annual educational expenditure in ten thousand yuan
  slope : Real
  intercept : Real
  equation : Real → Real := λ x => slope * x + intercept

/-- Theorem: In the given linear regression model, an increase of 1 in income
    results in an increase of 0.15 in educational expenditure -/
theorem income_education_relationship (model : IncomeEducationModel)
    (h_slope : model.slope = 0.15)
    (h_intercept : model.intercept = 0.2) :
    ∀ x : Real, model.equation (x + 1) - model.equation x = 0.15 := by
  sorry

#check income_education_relationship

end income_education_relationship_l3060_306083


namespace complex_root_coefficients_l3060_306072

theorem complex_root_coefficients :
  ∀ (b c : ℝ),
  (Complex.I * Real.sqrt 2 + 1) ^ 2 + b * (Complex.I * Real.sqrt 2 + 1) + c = 0 →
  b = -2 ∧ c = 3 := by
  sorry

end complex_root_coefficients_l3060_306072


namespace arithmetic_geometric_sequence_ratio_l3060_306074

theorem arithmetic_geometric_sequence_ratio (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ -- positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ -- distinct
  2 * a = b + c ∧ -- arithmetic sequence
  a * a = b * c -- geometric sequence
  → ∃ (k : ℝ), k ≠ 0 ∧ a = 2 * k ∧ b = 4 * k ∧ c = k := by
sorry

end arithmetic_geometric_sequence_ratio_l3060_306074


namespace min_value_theorem_l3060_306012

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 2) :
  (2 / a) + (1 / b) ≥ 9 / 2 := by
  sorry

end min_value_theorem_l3060_306012


namespace angle_inequality_equivalence_l3060_306096

theorem angle_inequality_equivalence (θ : Real) (h1 : 0 ≤ θ) (h2 : θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 2 → x^2 * Real.cos θ - x * (2 - x) + (2 - x)^2 * Real.sin θ > 0) ↔
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) := by
  sorry

end angle_inequality_equivalence_l3060_306096


namespace increase_by_percentage_l3060_306075

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 75 → percentage = 150 → result = initial * (1 + percentage / 100) → result = 187.5 := by
  sorry

end increase_by_percentage_l3060_306075


namespace mr_gates_classes_l3060_306088

/-- Proves that given the conditions in the problem, Mr. Gates has 4 classes --/
theorem mr_gates_classes : 
  ∀ (buns_per_package : ℕ) 
    (packages_bought : ℕ) 
    (students_per_class : ℕ) 
    (buns_per_student : ℕ),
  buns_per_package = 8 →
  packages_bought = 30 →
  students_per_class = 30 →
  buns_per_student = 2 →
  (packages_bought * buns_per_package) / (students_per_class * buns_per_student) = 4 :=
by sorry

end mr_gates_classes_l3060_306088


namespace existence_of_non_divisible_k_l3060_306013

theorem existence_of_non_divisible_k (a b c n : ℤ) (h : n ≥ 3) :
  ∃ k : ℤ, ¬(n ∣ (k + a)) ∧ ¬(n ∣ (k + b)) ∧ ¬(n ∣ (k + c)) := by
  sorry

end existence_of_non_divisible_k_l3060_306013


namespace equation_solutions_l3060_306001

theorem equation_solutions :
  (∃ x : ℝ, 3 * x * (x - 1) = 2 - 2 * x) ∧
  (∃ x : ℝ, 3 * x^2 - 6 * x + 2 = 0) ∧
  (∀ x : ℝ, 3 * x * (x - 1) = 2 - 2 * x ↔ (x = 1 ∨ x = -2/3)) ∧
  (∀ x : ℝ, 3 * x^2 - 6 * x + 2 = 0 ↔ (x = 1 + Real.sqrt 3 / 3 ∨ x = 1 - Real.sqrt 3 / 3)) :=
by sorry

end equation_solutions_l3060_306001


namespace solve_x_l3060_306026

def symbol_value (a b c d : ℤ) : ℤ := a * d - b * c

theorem solve_x : ∃ x : ℤ, symbol_value (x - 1) 2 3 (-5) = 9 ∧ x = -2 := by sorry

end solve_x_l3060_306026


namespace total_books_calculation_l3060_306080

theorem total_books_calculation (darryl_books : ℕ) (lamont_books : ℕ) (loris_books : ℕ) (danielle_books : ℕ) : 
  darryl_books = 20 →
  lamont_books = 2 * darryl_books →
  loris_books + 3 = lamont_books →
  danielle_books = lamont_books + darryl_books + 10 →
  darryl_books + lamont_books + loris_books + danielle_books = 167 := by
sorry

end total_books_calculation_l3060_306080


namespace range_of_m_l3060_306027

def p (a : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 2 * a * x + 2 * a + 5 = 0 → x ∈ ({x | 4 * x^2 - 2 * a * x + 2 * a + 5 = 0} : Set ℝ)

def q (m : ℝ) : Prop := ∀ x : ℝ, 1 - m ≤ x ∧ x ≤ 1 + m

theorem range_of_m :
  (∀ a : ℝ, (¬(p a) → ∃ m : ℝ, m > 0 ∧ ¬(q m)) ∧
   (∃ m : ℝ, m > 0 ∧ ¬(q m) ∧ p a)) →
  {m : ℝ | m ≥ 9} = {m : ℝ | m > 0 ∧ (∀ a : ℝ, p a → q m)} :=
by sorry

end range_of_m_l3060_306027


namespace tank_full_after_45_minutes_l3060_306046

/-- Represents the state of a water tank system with three pipes. -/
structure TankSystem where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℕ

/-- Calculates the net water gain in one cycle. -/
def net_gain_per_cycle (system : TankSystem) : ℕ :=
  system.pipeA_rate + system.pipeB_rate - system.pipeC_rate

/-- Calculates the number of cycles needed to fill the tank. -/
def cycles_to_fill (system : TankSystem) : ℕ :=
  system.capacity / net_gain_per_cycle system

/-- Calculates the time in minutes to fill the tank. -/
def time_to_fill (system : TankSystem) : ℕ :=
  cycles_to_fill system * 3

/-- Theorem stating that the given tank system will be full after 45 minutes. -/
theorem tank_full_after_45_minutes (system : TankSystem)
  (h_capacity : system.capacity = 750)
  (h_pipeA : system.pipeA_rate = 40)
  (h_pipeB : system.pipeB_rate = 30)
  (h_pipeC : system.pipeC_rate = 20) :
  time_to_fill system = 45 := by
  sorry

end tank_full_after_45_minutes_l3060_306046


namespace function_properties_l3060_306006

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x - a

theorem function_properties (a : ℝ) (h : a ≠ 0) :
  (f a 0 = 2 → a = -1) ∧
  (a = -1 → 
    (∀ x y : ℝ, x < y → x < 0 → f a x > f a y) ∧
    (∀ x y : ℝ, x < y → 0 < x → f a x < f a y) ∧
    (∀ x : ℝ, f a x ≥ 2) ∧
    (f a 0 = 2)) ∧
  ((∀ x : ℝ, f a x ≠ 0) → -Real.exp 2 < a ∧ a < 0) :=
by sorry

#check function_properties

end function_properties_l3060_306006


namespace star_operation_proof_l3060_306022

-- Define the ※ operation
def star (a b : ℕ) : ℚ :=
  (b : ℚ) / 2 * (2 * (a : ℚ) / 10 + ((b : ℚ) - 1) / 10)

-- State the theorem
theorem star_operation_proof (a : ℕ) :
  star 1 2 = (3 : ℚ) / 10 ∧
  star 2 3 = (9 : ℚ) / 10 ∧
  star 5 4 = (26 : ℚ) / 10 ∧
  star a 15 = (165 : ℚ) / 10 →
  a = 4 := by
  sorry

end star_operation_proof_l3060_306022


namespace glue_per_clipping_l3060_306033

theorem glue_per_clipping 
  (num_friends : ℕ) 
  (clippings_per_friend : ℕ) 
  (total_glue_drops : ℕ) : 
  num_friends = 7 → 
  clippings_per_friend = 3 → 
  total_glue_drops = 126 → 
  total_glue_drops / (num_friends * clippings_per_friend) = 6 := by
  sorry

end glue_per_clipping_l3060_306033


namespace min_questions_to_find_z_l3060_306051

/-- Represents a person in the company -/
structure Person where
  id : Nat

/-- Represents the company with n people -/
structure Company where
  n : Nat
  people : Finset Person
  z : Person
  knows : Person → Person → Prop

/-- Axioms for the company structure -/
axiom company_size (c : Company) : c.people.card = c.n

axiom z_knows_all (c : Company) (p : Person) : 
  p ∈ c.people → p ≠ c.z → c.knows c.z p

axiom z_known_by_none (c : Company) (p : Person) : 
  p ∈ c.people → p ≠ c.z → ¬(c.knows p c.z)

/-- The main theorem to prove -/
theorem min_questions_to_find_z (c : Company) :
  ∃ (strategy : Nat → Person × Person),
    (∀ k, k < c.n - 1 → 
      (strategy k).1 ∈ c.people ∧ (strategy k).2 ∈ c.people) ∧
    (∀ result : Nat → Bool,
      ∃! p, p ∈ c.people ∧ 
        ∀ k, k < c.n - 1 → 
          result k = c.knows (strategy k).1 (strategy k).2 →
          p ≠ (strategy k).1 ∧ p ≠ (strategy k).2) ∧
  ¬∃ (strategy : Nat → Person × Person),
    (∀ k, k < c.n - 2 → 
      (strategy k).1 ∈ c.people ∧ (strategy k).2 ∈ c.people) ∧
    (∀ result : Nat → Bool,
      ∃! p, p ∈ c.people ∧ 
        ∀ k, k < c.n - 2 → 
          result k = c.knows (strategy k).1 (strategy k).2 →
          p ≠ (strategy k).1 ∧ p ≠ (strategy k).2) :=
by
  sorry

end min_questions_to_find_z_l3060_306051


namespace halloween_candy_distribution_l3060_306037

theorem halloween_candy_distribution (initial_candy : ℕ) (eaten_candy : ℕ) (num_piles : ℕ) 
  (h1 : initial_candy = 78)
  (h2 : eaten_candy = 30)
  (h3 : num_piles = 6)
  : (initial_candy - eaten_candy) / num_piles = 8 := by
  sorry

end halloween_candy_distribution_l3060_306037


namespace total_cost_calculation_l3060_306049

def silverware_cost : ℝ := 20
def plate_cost_ratio : ℝ := 0.5

theorem total_cost_calculation :
  let plate_cost := plate_cost_ratio * silverware_cost
  silverware_cost + plate_cost = 30 :=
by sorry

end total_cost_calculation_l3060_306049


namespace cone_height_l3060_306081

/-- The height of a cone with base area π and slant height 2 is √3 -/
theorem cone_height (base_area : Real) (slant_height : Real) :
  base_area = Real.pi → slant_height = 2 → ∃ (height : Real), height = Real.sqrt 3 := by
  sorry

end cone_height_l3060_306081


namespace typists_pages_time_relation_l3060_306036

/-- Given that 10 typists can type 25 pages in 5 minutes, 
    prove that 2 typists can type 2 pages in 2 minutes. -/
theorem typists_pages_time_relation : 
  ∀ (n : ℕ), 
    (10 : ℝ) * (25 : ℝ) / (5 : ℝ) = n * (2 : ℝ) / (2 : ℝ) → 
    n = 2 :=
by sorry

end typists_pages_time_relation_l3060_306036


namespace connie_grandmother_birth_year_l3060_306060

/-- Calculates the birth year of Connie's grandmother given the birth years of her siblings and the gap condition. -/
def grandmotherBirthYear (brotherBirthYear sisterBirthYear : ℕ) : ℕ :=
  let siblingGap := sisterBirthYear - brotherBirthYear
  sisterBirthYear - 2 * siblingGap

/-- Proves that Connie's grandmother was born in 1928 given the known conditions. -/
theorem connie_grandmother_birth_year :
  grandmotherBirthYear 1932 1936 = 1928 := by
  sorry

#eval grandmotherBirthYear 1932 1936

end connie_grandmother_birth_year_l3060_306060


namespace diagonal_cubes_150_324_375_l3060_306063

/-- The number of unit cubes that a diagonal passes through in a rectangular prism -/
def diagonal_cubes (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd a c - Nat.gcd b c + Nat.gcd (Nat.gcd a b) c

/-- Theorem: In a 150 × 324 × 375 rectangular prism, the diagonal passes through 768 unit cubes -/
theorem diagonal_cubes_150_324_375 :
  diagonal_cubes 150 324 375 = 768 := by
  sorry

end diagonal_cubes_150_324_375_l3060_306063


namespace f_properties_l3060_306002

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

theorem f_properties :
  ∃ (p : ℝ),
    (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
    (p = 2 * Real.pi) ∧
    (∀ (x : ℝ), f x ≤ 2) ∧
    (∃ (x : ℝ), f x = 2) ∧
    (∀ (k : ℤ),
      ∀ (x : ℝ),
        (5 * Real.pi / 4 + 2 * ↑k * Real.pi ≤ x ∧ x ≤ 9 * Real.pi / 4 + 2 * ↑k * Real.pi) →
        (∀ (y : ℝ), x < y → f (-y) < f (-x))) :=
by sorry

end f_properties_l3060_306002


namespace perfect_square_trinomial_l3060_306025

theorem perfect_square_trinomial (b : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, 16 * x^2 - b * x + 9 = (a * x + 3)^2) ↔ b = 24 ∨ b = -24 := by
  sorry

end perfect_square_trinomial_l3060_306025


namespace nonzero_digits_after_decimal_l3060_306005

theorem nonzero_digits_after_decimal (n : ℕ) (d : ℕ) (h : n = 60 ∧ d = 2^3 * 5^8) :
  (Nat.digits 10 (n * 10^7 / d)).length - 7 = 3 :=
sorry

end nonzero_digits_after_decimal_l3060_306005


namespace unique_solution_quadratic_equation_l3060_306032

theorem unique_solution_quadratic_equation :
  ∃! x : ℝ, (2016 + 3*x)^2 = (3*x)^2 ∧ x = -336 := by
sorry

end unique_solution_quadratic_equation_l3060_306032


namespace perpendicular_lines_m_values_l3060_306062

/-- Two lines are perpendicular if the sum of the products of their coefficients of x and y is zero -/
def are_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

/-- The first line: mx - (m+2)y + 2 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x - (m + 2) * y + 2 = 0

/-- The second line: 3x - my - 1 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop := 3 * x - m * y - 1 = 0

theorem perpendicular_lines_m_values :
  ∀ m : ℝ, are_perpendicular m (-(m+2)) 3 (-m) → m = 0 ∨ m = -5 := by sorry

end perpendicular_lines_m_values_l3060_306062


namespace arithmetic_sequence_problem_l3060_306069

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 + 3 * a 8 + a 15 = 120) :
  3 * a 9 - a 11 = 48 := by
sorry

end arithmetic_sequence_problem_l3060_306069


namespace f_min_value_l3060_306082

/-- The polynomial f(x) defined for a positive integer n and real x -/
def f (n : ℕ+) (x : ℝ) : ℝ :=
  (Finset.range (2*n+1)).sum (fun k => (2*n+1-k) * x^k)

/-- Theorem stating that the minimum value of f(x) is n+1 and occurs at x = -1 -/
theorem f_min_value (n : ℕ+) :
  (∀ x : ℝ, f n x ≥ f n (-1)) ∧ f n (-1) = n + 1 := by
  sorry

end f_min_value_l3060_306082


namespace equation_solution_l3060_306092

theorem equation_solution (x y : ℝ) 
  (h1 : x + 2 ≠ 0) 
  (h2 : x - y + 1 ≠ 0) 
  (h3 : (x - y) / (x + 2) = y / (x - y + 1)) : 
  x = (y - 1 + Real.sqrt (-3 * y^2 + 10 * y + 1)) / 2 ∨ 
  x = (y - 1 - Real.sqrt (-3 * y^2 + 10 * y + 1)) / 2 :=
sorry

end equation_solution_l3060_306092


namespace solve_plane_problem_l3060_306038

def plane_problem (distance : ℝ) (time_with_wind : ℝ) (time_against_wind : ℝ) : Prop :=
  ∃ (plane_speed : ℝ) (wind_speed : ℝ),
    (plane_speed + wind_speed) * time_with_wind = distance ∧
    (plane_speed - wind_speed) * time_against_wind = distance ∧
    plane_speed = 262.5

theorem solve_plane_problem :
  plane_problem 900 3 4 := by
  sorry

end solve_plane_problem_l3060_306038


namespace remainder_of_binary_number_div_8_l3060_306014

def binary_number : ℕ := 0b100101110011

theorem remainder_of_binary_number_div_8 :
  binary_number % 8 = 3 := by
sorry

end remainder_of_binary_number_div_8_l3060_306014


namespace teacher_volunteers_count_l3060_306059

/-- Calculates the number of teacher volunteers for a school Christmas play. -/
def teacher_volunteers (total_needed : ℕ) (math_classes : ℕ) (students_per_class : ℕ) (more_needed : ℕ) : ℕ :=
  total_needed - (math_classes * students_per_class) - more_needed

/-- Theorem stating that the number of teacher volunteers is 13. -/
theorem teacher_volunteers_count : teacher_volunteers 50 6 5 7 = 13 := by
  sorry

end teacher_volunteers_count_l3060_306059


namespace parking_space_difference_l3060_306029

/-- Represents a parking garage with four levels -/
structure ParkingGarage where
  level1 : Nat
  level2 : Nat
  level3 : Nat
  level4 : Nat

/-- Theorem stating the difference in parking spaces between the third and fourth levels -/
theorem parking_space_difference (garage : ParkingGarage) : 
  garage.level1 = 90 →
  garage.level2 = garage.level1 + 8 →
  garage.level3 = garage.level2 + 12 →
  garage.level1 + garage.level2 + garage.level3 + garage.level4 = 299 →
  garage.level3 - garage.level4 = 109 := by
  sorry

end parking_space_difference_l3060_306029


namespace exp_equals_derivative_l3060_306008

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem exp_equals_derivative :
  ∀ x : ℝ, f x = deriv f x :=
by sorry

end exp_equals_derivative_l3060_306008


namespace exists_number_with_large_square_digit_sum_l3060_306000

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number whose square's digit sum exceeds 1000 times its own digit sum -/
theorem exists_number_with_large_square_digit_sum :
  ∃ n : ℕ, sumOfDigits (n^2) > 1000 * sumOfDigits n := by
  sorry

end exists_number_with_large_square_digit_sum_l3060_306000


namespace constant_function_shifted_l3060_306042

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 5

-- State the theorem
theorem constant_function_shifted (x : ℝ) : f (x + 3) = 5 := by
  sorry

end constant_function_shifted_l3060_306042


namespace anthony_ate_two_bananas_l3060_306057

/-- The number of bananas Anthony bought -/
def initial_bananas : ℕ := 12

/-- The number of bananas Anthony has left -/
def remaining_bananas : ℕ := 10

/-- The number of bananas Anthony ate -/
def eaten_bananas : ℕ := initial_bananas - remaining_bananas

theorem anthony_ate_two_bananas : eaten_bananas = 2 := by
  sorry

end anthony_ate_two_bananas_l3060_306057


namespace gold_coins_percentage_l3060_306085

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  total : ℝ
  coins_and_beads : ℝ
  beads : ℝ
  gold_coins : ℝ

/-- The percentage of gold coins in the urn is 36% -/
theorem gold_coins_percentage (urn : UrnComposition) : 
  urn.coins_and_beads / urn.total = 0.75 →
  urn.beads / urn.total = 0.15 →
  urn.gold_coins / (urn.coins_and_beads - urn.beads) = 0.6 →
  urn.gold_coins / urn.total = 0.36 := by
  sorry

#check gold_coins_percentage

end gold_coins_percentage_l3060_306085


namespace georges_required_speed_l3060_306053

/-- George's usual walking distance to school in miles -/
def usual_distance : ℝ := 1.5

/-- George's usual walking speed in miles per hour -/
def usual_speed : ℝ := 4

/-- Distance George walks at a slower pace today in miles -/
def slow_distance : ℝ := 1

/-- George's slower walking speed today in miles per hour -/
def slow_speed : ℝ := 3

/-- Remaining distance George needs to run in miles -/
def remaining_distance : ℝ := 0.5

/-- Theorem stating the speed George needs to run to arrive on time -/
theorem georges_required_speed : 
  ∃ (required_speed : ℝ),
    (usual_distance / usual_speed = slow_distance / slow_speed + remaining_distance / required_speed) ∧
    required_speed = 12 := by
  sorry

end georges_required_speed_l3060_306053


namespace class_size_is_fifteen_l3060_306068

/-- Given a class of students with the following properties:
  1. The average age of all students is 15 years
  2. The average age of 6 students is 14 years
  3. The average age of 8 students is 16 years
  4. The age of the 15th student is 13 years
  Prove that the total number of students in the class is 15 -/
theorem class_size_is_fifteen (N : ℕ) 
  (h1 : (N : ℚ) * 15 = (6 : ℚ) * 14 + (8 : ℚ) * 16 + 13)
  (h2 : N ≥ 15) : N = 15 := by
  sorry


end class_size_is_fifteen_l3060_306068


namespace break_even_price_per_lot_l3060_306090

/-- Given a land purchase scenario, calculate the break-even price per lot -/
theorem break_even_price_per_lot (acres : ℕ) (price_per_acre : ℕ) (num_lots : ℕ) :
  acres = 4 →
  price_per_acre = 1863 →
  num_lots = 9 →
  (acres * price_per_acre) / num_lots = 828 := by
  sorry

end break_even_price_per_lot_l3060_306090


namespace quadratic_inequality_proof_l3060_306048

theorem quadratic_inequality_proof (a : ℝ) 
  (h : ∀ x : ℝ, x^2 - 2*a*x + a > 0) : 
  (0 < a ∧ a < 1) ∧ 
  (∀ x : ℝ, (a^(x^2 - 3) < a^(2*x) ∧ a^(2*x) < 1) ↔ x > 3) :=
sorry

end quadratic_inequality_proof_l3060_306048


namespace line_sum_m_b_l3060_306079

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) with slope m and y-intercept b -/
structure Line where
  x₁ : ℚ
  y₁ : ℚ
  x₂ : ℚ
  y₂ : ℚ
  m : ℚ
  b : ℚ
  eq₁ : y₁ = m * x₁ + b
  eq₂ : y₂ = m * x₂ + b

/-- Theorem: For a line passing through (2, -1) and (5, 3), m + b = -7/3 -/
theorem line_sum_m_b :
  ∀ l : Line,
    l.x₁ = 2 ∧ l.y₁ = -1 ∧ l.x₂ = 5 ∧ l.y₂ = 3 →
    l.m + l.b = -7/3 := by
  sorry

end line_sum_m_b_l3060_306079


namespace negation_of_existence_negation_of_logarithmic_inequality_l3060_306055

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ P x) ↔ (∀ x : ℝ, x > 0 → ¬ P x) :=
by sorry

theorem negation_of_logarithmic_inequality :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x + x - 1 ≤ 0) ↔
  (∀ x : ℝ, x > 0 → Real.log x + x - 1 > 0) :=
by sorry

end negation_of_existence_negation_of_logarithmic_inequality_l3060_306055


namespace mason_tables_theorem_l3060_306018

/-- The number of tables Mason needs settings for -/
def num_tables : ℕ :=
  let silverware_weight : ℕ := 4  -- weight of one piece of silverware in ounces
  let silverware_per_setting : ℕ := 3  -- number of silverware pieces per setting
  let plate_weight : ℕ := 12  -- weight of one plate in ounces
  let plates_per_setting : ℕ := 2  -- number of plates per setting
  let settings_per_table : ℕ := 8  -- number of settings per table
  let backup_settings : ℕ := 20  -- number of backup settings
  let total_weight : ℕ := 5040  -- total weight of all settings in ounces

  -- Calculate the result
  (total_weight / (silverware_weight * silverware_per_setting + plate_weight * plates_per_setting) - backup_settings) / settings_per_table

theorem mason_tables_theorem : num_tables = 15 := by
  sorry

end mason_tables_theorem_l3060_306018


namespace range_theorem_l3060_306076

/-- A monotonically decreasing odd function on ℝ with f(1) = -1 -/
def f : ℝ → ℝ :=
  sorry

/-- f is monotonically decreasing -/
axiom f_monotone : ∀ x y, x ≤ y → f y ≤ f x

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f(1) = -1 -/
axiom f_one : f 1 = -1

/-- The range of x satisfying -1 ≤ f(x-2) ≤ 1 is [1, 3] -/
theorem range_theorem : Set.Icc 1 3 = {x | -1 ≤ f (x - 2) ∧ f (x - 2) ≤ 1} :=
  sorry

end range_theorem_l3060_306076


namespace complex_modulus_squared_l3060_306047

theorem complex_modulus_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 6 - 9*I) : 
  Complex.abs z^2 = 39/4 := by
sorry

end complex_modulus_squared_l3060_306047


namespace p_properties_l3060_306043

/-- The product of digits function -/
def p (n : ℕ+) : ℕ := sorry

/-- Theorem stating the properties of p(n) -/
theorem p_properties (n : ℕ+) : 
  (p n ≤ n) ∧ (10 * p n = n^2 + 4*n - 2005 ↔ n = 45) := by sorry

end p_properties_l3060_306043


namespace quadratic_equation_value_l3060_306073

theorem quadratic_equation_value (x : ℝ) (h : x = 2) : x^2 + 5*x - 14 = 0 := by
  sorry

end quadratic_equation_value_l3060_306073


namespace inverse_variation_problem_l3060_306058

/-- Given that a² and √b vary inversely, prove that b = 16 when a + b = 20 -/
theorem inverse_variation_problem (a b : ℝ) (k : ℝ) : 
  (∀ (a b : ℝ), a^2 * (b^(1/2)) = k) →  -- a² and √b vary inversely
  (4^2 * 16^(1/2) = k) →                -- a = 4 when b = 16
  (a + b = 20) →                        -- condition for the question
  (b = 16) :=                           -- conclusion to prove
by sorry

end inverse_variation_problem_l3060_306058


namespace park_trees_theorem_l3060_306031

/-- The number of dogwood trees remaining in the park after a day's work -/
def remaining_trees (first_part : ℝ) (second_part : ℝ) (third_part : ℝ) 
  (trees_cut : ℝ) (trees_planted : ℝ) : ℝ :=
  first_part + second_part + third_part - trees_cut + trees_planted

/-- Theorem stating the number of remaining trees after the day's work -/
theorem park_trees_theorem (first_part : ℝ) (second_part : ℝ) (third_part : ℝ) 
  (trees_cut : ℝ) (trees_planted : ℝ) :
  first_part = 5.0 →
  second_part = 4.0 →
  third_part = 6.0 →
  trees_cut = 7.0 →
  trees_planted = 3.0 →
  remaining_trees first_part second_part third_part trees_cut trees_planted = 11.0 :=
by
  sorry

#eval remaining_trees 5.0 4.0 6.0 7.0 3.0

end park_trees_theorem_l3060_306031


namespace expression_value_l3060_306098

theorem expression_value : 
  (2024^3 - 2 * 2024^2 * 2025 + 3 * 2024 * 2025^2 - 2025^3 + 4) / (2024 * 2025) = 2022 := by
  sorry

end expression_value_l3060_306098


namespace v_closed_under_multiplication_l3060_306028

/-- The set of cubes of positive integers -/
def v : Set ℕ := {n | ∃ m : ℕ+, n = m^3}

/-- Proof that v is closed under multiplication -/
theorem v_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ v → b ∈ v → (a * b) ∈ v := by
  sorry

end v_closed_under_multiplication_l3060_306028


namespace min_coins_for_distribution_l3060_306039

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_coins_for_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 63) :
  min_additional_coins num_friends initial_coins = 57 := by
  sorry

#eval min_additional_coins 15 63

end min_coins_for_distribution_l3060_306039


namespace purely_imaginary_implies_a_equals_one_l3060_306015

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def PurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real number a. -/
def z (a : ℝ) : ℂ :=
  ⟨a^2 + 2*a - 3, a + 3⟩

theorem purely_imaginary_implies_a_equals_one :
  ∀ a : ℝ, PurelyImaginary (z a) → a = 1 :=
by sorry

end purely_imaginary_implies_a_equals_one_l3060_306015


namespace x_squared_mod_25_l3060_306045

theorem x_squared_mod_25 (x : ℤ) 
  (h1 : 5 * x ≡ 15 [ZMOD 25])
  (h2 : 2 * x ≡ 10 [ZMOD 25]) : 
  x^2 ≡ 0 [ZMOD 25] := by
sorry

end x_squared_mod_25_l3060_306045


namespace function_behavior_l3060_306071

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : is_symmetric_about_one f)
  (h3 : is_decreasing_on f 1 2) :
  is_increasing_on f (-2) (-1) ∧ is_decreasing_on f 3 4 := by
  sorry

end function_behavior_l3060_306071


namespace line_intersects_circle_l3060_306064

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the point P
def point_P : ℝ × ℝ := (3, 0)

-- Define a line passing through P
def line_through_P (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x - point_P.1) + point_P.2

-- Theorem statement
theorem line_intersects_circle :
  ∀ m : ℝ, ∃ x y : ℝ, circle_C x y ∧ line_through_P m x y :=
sorry

end line_intersects_circle_l3060_306064


namespace max_value_on_curve_l3060_306067

theorem max_value_on_curve (b : ℝ) (h : b > 0) :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ x^2 + 2*y
  let S : Set (ℝ × ℝ) := {(x, y) | x^2/4 + y^2/b^2 = 1}
  (∃ (M : ℝ), ∀ (p : ℝ × ℝ), p ∈ S → f p ≤ M) ∧
  (0 < b ∧ b ≤ 4 → ∀ (M : ℝ), (∀ (p : ℝ × ℝ), p ∈ S → f p ≤ M) → b^2/4 + 4 ≤ M) ∧
  (b > 4 → ∀ (M : ℝ), (∀ (p : ℝ × ℝ), p ∈ S → f p ≤ M) → 2*b ≤ M) :=
by sorry

end max_value_on_curve_l3060_306067


namespace remainder_451951_div_5_l3060_306019

theorem remainder_451951_div_5 : 451951 % 5 = 1 := by
  sorry

end remainder_451951_div_5_l3060_306019


namespace circle_equation_from_diameter_l3060_306061

theorem circle_equation_from_diameter (P Q : ℝ × ℝ) : 
  P = (4, 0) → Q = (0, 2) → 
  ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 5 ↔ 
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
      x = 4 * (1 - t) + 0 * t ∧ 
      y = 0 * (1 - t) + 2 * t ∧
      (x - 4)^2 + (y - 0)^2 = (0 - 4)^2 + (2 - 0)^2 / 4) :=
by sorry

end circle_equation_from_diameter_l3060_306061


namespace grid_arrangement_impossibility_l3060_306035

theorem grid_arrangement_impossibility :
  ¬ ∃ (grid : Fin 25 → Fin 41 → ℤ),
    (∀ i j i' j', grid i j = grid i' j' → (i = i' ∧ j = j')) ∧
    (∀ i j,
      (i.val + 1 < 25 → |grid i j - grid ⟨i.val + 1, sorry⟩ j| ≤ 16) ∧
      (j.val + 1 < 41 → |grid i j - grid i ⟨j.val + 1, sorry⟩| ≤ 16)) :=
sorry

end grid_arrangement_impossibility_l3060_306035


namespace circle_intersection_area_l3060_306024

noncomputable def circleIntersection (r : ℝ) (bd ed : ℝ) : ℝ :=
  let ad := 2 * r + bd
  let ea := Real.sqrt (ad^2 + ed^2)
  let ec := ed^2 / ea
  let ac := ea - ec
  let bc := Real.sqrt ((2*r)^2 - ac^2)
  1/2 * bc * ac

theorem circle_intersection_area (r bd ed : ℝ) (hr : r = 4) (hbd : bd = 6) (hed : ed = 5) :
  circleIntersection r bd ed = 11627.6 / 221 :=
sorry

end circle_intersection_area_l3060_306024


namespace work_completion_theorem_l3060_306034

theorem work_completion_theorem (total_work : ℝ) :
  (34 : ℝ) * 18 * total_work = 17 * 36 * total_work := by
  sorry

#check work_completion_theorem

end work_completion_theorem_l3060_306034


namespace factorization_equality_l3060_306050

theorem factorization_equality (a : ℝ) : a * (a - 2) + 1 = (a - 1)^2 := by
  sorry

end factorization_equality_l3060_306050


namespace cube_of_product_l3060_306040

theorem cube_of_product (x y : ℝ) : (-3 * x^2 * y)^3 = -27 * x^6 * y^3 := by
  sorry

end cube_of_product_l3060_306040


namespace triangle_side_length_l3060_306091

theorem triangle_side_length (a b c : ℝ) (area : ℝ) : 
  a = 1 → b = Real.sqrt 7 → area = Real.sqrt 3 / 2 → 
  (c = 2 ∨ c = 2 * Real.sqrt 3) := by sorry

end triangle_side_length_l3060_306091


namespace no_extremum_implies_a_nonnegative_l3060_306094

/-- A function that has no extremum on ℝ -/
def NoExtremum (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ ε > 0, ∀ y : ℝ, |y - x| < ε → f y ≠ f x ∨ (f y < f x ∧ f y > f x)

/-- The main theorem -/
theorem no_extremum_implies_a_nonnegative (a : ℝ) :
  NoExtremum (fun x => Real.exp x + a * x) → a ≥ 0 := by
  sorry


end no_extremum_implies_a_nonnegative_l3060_306094

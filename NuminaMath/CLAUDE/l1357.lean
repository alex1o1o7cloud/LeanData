import Mathlib

namespace NUMINAMATH_CALUDE_a_less_than_b_plus_one_l1357_135771

theorem a_less_than_b_plus_one (a b : ℝ) (h : a < b) : a < b + 1 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_b_plus_one_l1357_135771


namespace NUMINAMATH_CALUDE_jack_collection_books_per_author_l1357_135716

/-- Represents Jack's classic book collection -/
structure ClassicCollection where
  authors : Nat
  total_books : Nat

/-- Calculates the number of books per author in a classic collection -/
def books_per_author (c : ClassicCollection) : Nat :=
  c.total_books / c.authors

/-- Theorem: In Jack's collection of 6 authors and 198 books, each author has 33 books -/
theorem jack_collection_books_per_author :
  let jack_collection : ClassicCollection := { authors := 6, total_books := 198 }
  books_per_author jack_collection = 33 := by
  sorry

end NUMINAMATH_CALUDE_jack_collection_books_per_author_l1357_135716


namespace NUMINAMATH_CALUDE_factorization_mx_plus_my_l1357_135755

theorem factorization_mx_plus_my (m x y : ℝ) : m * x + m * y = m * (x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_mx_plus_my_l1357_135755


namespace NUMINAMATH_CALUDE_carson_seed_amount_l1357_135790

-- Define variables
variable (seed : ℝ)
variable (fertilizer : ℝ)

-- Define the conditions
def seed_fertilizer_ratio : Prop := seed = 3 * fertilizer
def total_amount : Prop := seed + fertilizer = 60

-- Theorem statement
theorem carson_seed_amount 
  (h1 : seed_fertilizer_ratio seed fertilizer)
  (h2 : total_amount seed fertilizer) :
  seed = 45 := by
  sorry

end NUMINAMATH_CALUDE_carson_seed_amount_l1357_135790


namespace NUMINAMATH_CALUDE_elliptical_machine_payment_l1357_135753

/-- Proves that the daily minimum payment for an elliptical machine is $6 given the specified conditions --/
theorem elliptical_machine_payment 
  (total_cost : ℝ) 
  (down_payment_ratio : ℝ) 
  (payment_days : ℕ) 
  (h1 : total_cost = 120) 
  (h2 : down_payment_ratio = 1/2) 
  (h3 : payment_days = 10) : 
  (total_cost * (1 - down_payment_ratio)) / payment_days = 6 := by
sorry

end NUMINAMATH_CALUDE_elliptical_machine_payment_l1357_135753


namespace NUMINAMATH_CALUDE_pluto_orbit_scientific_notation_l1357_135740

/-- The radius of Pluto's orbit in kilometers -/
def pluto_orbit_radius : ℝ := 5900000000

/-- The scientific notation representation of Pluto's orbit radius -/
def pluto_orbit_scientific : ℝ := 5.9 * (10 ^ 9)

/-- Theorem stating that the radius of Pluto's orbit is equal to its scientific notation representation -/
theorem pluto_orbit_scientific_notation : pluto_orbit_radius = pluto_orbit_scientific := by
  sorry

end NUMINAMATH_CALUDE_pluto_orbit_scientific_notation_l1357_135740


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1357_135702

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3}

-- Define set A
def A : Finset Nat := {1, 2}

-- Define set B
def B : Finset Nat := {2, 3}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1357_135702


namespace NUMINAMATH_CALUDE_negation_equivalence_l1357_135713

theorem negation_equivalence :
  (¬ ∃ a ∈ Set.Icc (0 : ℝ) 1, a^4 + a^2 > 1) ↔
  (∀ a ∈ Set.Icc (0 : ℝ) 1, a^4 + a^2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1357_135713


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l1357_135735

theorem point_movement_on_number_line (A : ℝ) (movement : ℝ) : 
  A = -2 → movement = 4 → (A - movement = -6 ∨ A + movement = 2) := by
  sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l1357_135735


namespace NUMINAMATH_CALUDE_sequence_length_bound_l1357_135767

theorem sequence_length_bound (N : ℕ) (m : ℕ) (a : ℕ → ℕ) :
  (∀ i j, 1 ≤ i → i < j → j ≤ m → a i < a j) →
  (∀ i, 1 ≤ i → i ≤ m → a i ≤ N) →
  (∀ i j, 1 ≤ i → i < j → j ≤ m → Nat.lcm (a i) (a j) ≤ N) →
  m ≤ 2 * Int.floor (Real.sqrt N) :=
by sorry

end NUMINAMATH_CALUDE_sequence_length_bound_l1357_135767


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l1357_135710

theorem congruence_solutions_count :
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, x > 0 ∧ x < 120 ∧ (x + 17) % 38 = 75 % 38) ∧
    (∀ x : ℕ, x > 0 ∧ x < 120 ∧ (x + 17) % 38 = 75 % 38 → x ∈ S) ∧
    Finset.card S = 3 :=
by sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l1357_135710


namespace NUMINAMATH_CALUDE_sin_value_for_given_condition_l1357_135707

theorem sin_value_for_given_condition (θ : Real) 
  (h1 : 5 * Real.tan θ = 2 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < π) : 
  Real.sin θ = (Real.sqrt 41 - 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_value_for_given_condition_l1357_135707


namespace NUMINAMATH_CALUDE_geometric_sequence_nth_term_l1357_135776

theorem geometric_sequence_nth_term (a₁ q : ℚ) (n : ℕ) (h1 : a₁ = 1/2) (h2 : q = 1/2) :
  a₁ * q^(n - 1) = 1/32 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_nth_term_l1357_135776


namespace NUMINAMATH_CALUDE_power_of_negative_square_l1357_135700

theorem power_of_negative_square (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_square_l1357_135700


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l1357_135787

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_eight_ten :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l1357_135787


namespace NUMINAMATH_CALUDE_yoongis_number_l1357_135750

theorem yoongis_number (x : ℤ) (h : x - 10 = 15) : x + 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_yoongis_number_l1357_135750


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1357_135720

theorem arithmetic_calculation : 
  (1 + 0.23 + 0.34) * (0.23 + 0.34 + 0.45) - (1 + 0.23 + 0.34 + 0.45) * (0.23 + 0.34) = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1357_135720


namespace NUMINAMATH_CALUDE_red_spot_percentage_is_40_l1357_135708

/-- Represents the farm with cows and their spot characteristics -/
structure Farm where
  total_cows : ℕ
  no_spot_cows : ℕ
  blue_spot_ratio : ℚ

/-- Calculates the percentage of cows with a red spot -/
def red_spot_percentage (farm : Farm) : ℚ :=
  let no_red_spot := farm.no_spot_cows / farm.blue_spot_ratio
  let red_spot := farm.total_cows - no_red_spot
  (red_spot / farm.total_cows) * 100

/-- Theorem stating that for the given farm conditions, 
    the percentage of cows with a red spot is 40% -/
theorem red_spot_percentage_is_40 (farm : Farm) 
  (h1 : farm.total_cows = 140)
  (h2 : farm.no_spot_cows = 63)
  (h3 : farm.blue_spot_ratio = 3/4) :
  red_spot_percentage farm = 40 := by
  sorry

#eval red_spot_percentage ⟨140, 63, 3/4⟩

end NUMINAMATH_CALUDE_red_spot_percentage_is_40_l1357_135708


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1357_135704

/-- If a point M(a+3, a+1) lies on the x-axis, then its coordinates are (2,0) -/
theorem point_on_x_axis (a : ℝ) :
  (a + 1 = 0) →  -- Condition for M to be on x-axis
  ((a + 3, a + 1) : ℝ × ℝ) = (2, 0) := by
sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1357_135704


namespace NUMINAMATH_CALUDE_wheel_speed_proof_l1357_135785

/-- Proves that the original speed of a wheel is 20 mph given specific conditions -/
theorem wheel_speed_proof (circumference : Real) (speed_increase : Real) (time_decrease : Real) :
  circumference = 50 / 5280 → -- circumference in miles
  speed_increase = 10 → -- speed increase in mph
  time_decrease = 1 / (2 * 3600) → -- time decrease in hours
  ∃ (r : Real),
    r > 0 ∧
    r * (50 * 3600 / (5280 * r)) = 50 / 5280 * 3600 ∧
    (r + speed_increase) * (50 * 3600 / (5280 * r) - time_decrease) = 50 / 5280 * 3600 ∧
    r = 20 :=
by sorry


end NUMINAMATH_CALUDE_wheel_speed_proof_l1357_135785


namespace NUMINAMATH_CALUDE_fourth_draw_probability_problem_solution_l1357_135730

/-- A box containing red and black balls -/
structure Box where
  red_balls : ℕ
  black_balls : ℕ

/-- The probability of selecting a black ball from a box -/
def prob_black (b : Box) : ℚ :=
  b.black_balls / (b.red_balls + b.black_balls)

/-- The box described in the problem -/
def problem_box : Box :=
  { red_balls := 4, black_balls := 4 }

theorem fourth_draw_probability (b : Box) :
  prob_black b = 1 / 2 →
  (∀ n : ℕ, n > 0 → prob_black { red_balls := b.red_balls - min n b.red_balls,
                                 black_balls := b.black_balls - min n b.black_balls } = 1 / 2) →
  prob_black { red_balls := b.red_balls - min 3 b.red_balls,
               black_balls := b.black_balls - min 3 b.black_balls } = 1 / 2 :=
by sorry

theorem problem_solution :
  prob_black problem_box = 1 / 2 ∧
  (∀ n : ℕ, n > 0 → prob_black { red_balls := problem_box.red_balls - min n problem_box.red_balls,
                                 black_balls := problem_box.black_balls - min n problem_box.black_balls } = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_fourth_draw_probability_problem_solution_l1357_135730


namespace NUMINAMATH_CALUDE_system_solvability_l1357_135744

/-- The set of values for parameter a such that the system has at least one solution -/
def ValidAValues : Set ℝ := {a | a < 0 ∨ a ≥ 2/3}

/-- The system of equations -/
def System (a b x y : ℝ) : Prop :=
  x = |y + a| + 4/a ∧ x^2 + y^2 + 24 + b*(2*y + b) = 10*x

/-- Theorem stating the condition for the existence of a solution -/
theorem system_solvability (a : ℝ) :
  (∃ b x y, System a b x y) ↔ a ∈ ValidAValues :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l1357_135744


namespace NUMINAMATH_CALUDE_sigma_odd_implies_perfect_square_l1357_135779

/-- The number of positive divisors of a natural number -/
def sigma (n : ℕ) : ℕ := sorry

/-- Theorem: If the number of positive divisors of a natural number is odd, then the number is a perfect square -/
theorem sigma_odd_implies_perfect_square (N : ℕ) : 
  Odd (sigma N) → ∃ m : ℕ, N = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sigma_odd_implies_perfect_square_l1357_135779


namespace NUMINAMATH_CALUDE_jack_and_jill_speed_l1357_135727

theorem jack_and_jill_speed : 
  ∀ x : ℝ, 
    x ≠ -2 →
    (x^2 - 7*x - 12 = (x^2 - 3*x - 10) / (x + 2)) →
    (x^2 - 7*x - 12 = 2) := by
  sorry

end NUMINAMATH_CALUDE_jack_and_jill_speed_l1357_135727


namespace NUMINAMATH_CALUDE_standard_deviation_double_data_l1357_135766

def data1 : List ℝ := [2, 3, 4, 5]
def data2 : List ℝ := [4, 6, 8, 10]

def standard_deviation (data : List ℝ) : ℝ := sorry

theorem standard_deviation_double_data :
  standard_deviation data1 = (1 / 2) * standard_deviation data2 := by sorry

end NUMINAMATH_CALUDE_standard_deviation_double_data_l1357_135766


namespace NUMINAMATH_CALUDE_cylinder_generatrix_length_l1357_135748

/-- The length of the generatrix of a cylinder with base radius 1 and lateral surface area 6π is 2 -/
theorem cylinder_generatrix_length :
  ∀ (generatrix : ℝ),
  (generatrix > 0) →
  (2 * π * 1 + 2 * π * generatrix = 6 * π) →
  generatrix = 2 := by
sorry

end NUMINAMATH_CALUDE_cylinder_generatrix_length_l1357_135748


namespace NUMINAMATH_CALUDE_income_minus_expenses_tax_lower_l1357_135736

/-- Represents the tax options available --/
inductive TaxOption
  | IncomeTax
  | IncomeMinusExpensesTax

/-- Calculates the tax payable for a given option --/
def calculateTax (option : TaxOption) (totalIncome expenses insuranceContributions : ℕ) : ℕ :=
  match option with
  | TaxOption.IncomeTax =>
      let incomeTax := totalIncome * 6 / 100
      let maxDeduction := min (incomeTax / 2) insuranceContributions
      incomeTax - maxDeduction
  | TaxOption.IncomeMinusExpensesTax =>
      let taxBase := totalIncome - expenses
      let regularTax := taxBase * 15 / 100
      let minimumTax := totalIncome * 1 / 100
      max regularTax minimumTax

/-- Theorem stating that the Income minus expenses tax option results in lower tax --/
theorem income_minus_expenses_tax_lower
  (totalIncome expenses insuranceContributions : ℕ)
  (h1 : totalIncome = 150000000)
  (h2 : expenses = 141480000)
  (h3 : insuranceContributions = 16560000) :
  calculateTax TaxOption.IncomeMinusExpensesTax totalIncome expenses insuranceContributions <
  calculateTax TaxOption.IncomeTax totalIncome expenses insuranceContributions :=
by
  sorry


end NUMINAMATH_CALUDE_income_minus_expenses_tax_lower_l1357_135736


namespace NUMINAMATH_CALUDE_total_books_l1357_135738

theorem total_books (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 6)
  (h2 : mystery_shelves = 5)
  (h3 : picture_shelves = 4) :
  books_per_shelf * mystery_shelves + books_per_shelf * picture_shelves = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l1357_135738


namespace NUMINAMATH_CALUDE_factorization_count_l1357_135728

theorem factorization_count : 
  ∃! (S : Finset ℤ), 
    (∀ m : ℤ, m ∈ S ↔ 
      ∃ a b : ℤ, ∀ x : ℝ, x^2 + m*x - 16 = (x + a)*(x + b)) ∧ 
    S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_factorization_count_l1357_135728


namespace NUMINAMATH_CALUDE_x_plus_y_equals_48_l1357_135717

-- Define the arithmetic sequence
def arithmetic_sequence : List ℝ := [3, 9, 15, 33]

-- Define x and y as the last two terms before 33
def x : ℝ := arithmetic_sequence[arithmetic_sequence.length - 3]
def y : ℝ := arithmetic_sequence[arithmetic_sequence.length - 2]

-- Theorem to prove
theorem x_plus_y_equals_48 : x + y = 48 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_48_l1357_135717


namespace NUMINAMATH_CALUDE_parabola_point_movement_l1357_135709

/-- Represents a parabola of the form y = x^2 - 2mx - 3 -/
structure Parabola where
  m : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the parabola -/
def on_parabola (p : Parabola) (pt : Point) : Prop :=
  pt.y = pt.x^2 - 2*p.m*pt.x - 3

/-- Calculates the vertex of the parabola -/
def vertex (p : Parabola) : Point :=
  { x := p.m, y := -(p.m^2) - 3 }

theorem parabola_point_movement (p : Parabola) (A : Point) (n b : ℝ) :
  on_parabola p { x := -2, y := n } →
  { x := 1, y := n - b } = vertex p →
  b = 9 := by sorry

end NUMINAMATH_CALUDE_parabola_point_movement_l1357_135709


namespace NUMINAMATH_CALUDE_tree_watering_l1357_135745

theorem tree_watering (num_boys : ℕ) (trees_per_boy : ℕ) :
  num_boys = 9 →
  trees_per_boy = 3 →
  num_boys * trees_per_boy = 27 :=
by sorry

end NUMINAMATH_CALUDE_tree_watering_l1357_135745


namespace NUMINAMATH_CALUDE_second_race_lead_l1357_135798

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race setup -/
structure Race where
  distance : ℝ
  sunny : Runner
  windy : Runner

theorem second_race_lead (h d : ℝ) (first_race second_race : Race) 
  (h_positive : h > 0)
  (d_positive : d > 0)
  (first_race_distance : first_race.distance = 2 * h)
  (second_race_distance : second_race.distance = 2 * h)
  (same_speeds : first_race.sunny.speed = second_race.sunny.speed ∧ 
                 first_race.windy.speed = second_race.windy.speed)
  (first_race_lead : first_race.sunny.speed * first_race.distance = 
                     first_race.windy.speed * (first_race.distance - 2 * d))
  (second_race_start : second_race.sunny.speed * (second_race.distance + 2 * d) = 
                       second_race.windy.speed * second_race.distance) :
  second_race.sunny.speed * second_race.distance - 
  second_race.windy.speed * second_race.distance = 2 * d^2 / h := by
  sorry

end NUMINAMATH_CALUDE_second_race_lead_l1357_135798


namespace NUMINAMATH_CALUDE_weight_sum_l1357_135733

/-- Given the weights of four people satisfying certain conditions, 
    prove that the sum of the first and fourth person's weights is 372 pounds. -/
theorem weight_sum (e f g h : ℝ) 
  (ef_sum : e + f = 320)
  (fg_sum : f + g = 298)
  (gh_sum : g + h = 350) :
  e + h = 372 := by
  sorry

end NUMINAMATH_CALUDE_weight_sum_l1357_135733


namespace NUMINAMATH_CALUDE_double_mean_value_function_range_l1357_135721

/-- Definition of a double mean value function -/
def is_double_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    (deriv^[2] f x₁ = (f b - f a) / (b - a)) ∧
    (deriv^[2] f x₂ = (f b - f a) / (b - a))

/-- The main theorem -/
theorem double_mean_value_function_range (a : ℝ) (m : ℝ) :
  is_double_mean_value_function (fun x => 2 * x^3 - x^2 + m) 0 (2 * a) →
  1/8 < a ∧ a < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_double_mean_value_function_range_l1357_135721


namespace NUMINAMATH_CALUDE_shooting_competition_problem_prove_shooting_competition_l1357_135774

/-- Represents the penalty points for misses in a shooting competition -/
def penalty_points (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (n : ℚ) / 2 * (2 + (n - 1))

/-- The shooting competition problem -/
theorem shooting_competition_problem 
  (total_shots : ℕ) 
  (total_penalty : ℚ) 
  (hits : ℕ) : Prop :=
  total_shots = 25 ∧ 
  total_penalty = 7 ∧ 
  penalty_points (total_shots - hits) = total_penalty ∧
  hits = 21

/-- Proof of the shooting competition problem -/
theorem prove_shooting_competition : 
  ∃ (hits : ℕ), shooting_competition_problem 25 7 hits :=
sorry

end NUMINAMATH_CALUDE_shooting_competition_problem_prove_shooting_competition_l1357_135774


namespace NUMINAMATH_CALUDE_no_charming_numbers_l1357_135719

def is_charming (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = a + b^3

theorem no_charming_numbers : ¬∃ (n : ℕ), is_charming n :=
sorry

end NUMINAMATH_CALUDE_no_charming_numbers_l1357_135719


namespace NUMINAMATH_CALUDE_even_painted_faces_count_l1357_135711

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Represents a cube cut from the block -/
structure Cube where
  x : Nat
  y : Nat
  z : Nat

/-- Returns the number of painted faces for a cube in the given position -/
def numPaintedFaces (b : Block) (c : Cube) : Nat :=
  sorry

/-- Returns true if the number is even -/
def isEven (n : Nat) : Bool :=
  sorry

/-- Counts the number of cubes with an even number of painted faces -/
def countEvenPaintedFaces (b : Block) : Nat :=
  sorry

/-- Theorem: In a 6x3x2 inch block painted on all sides and cut into 1 inch cubes,
    the number of cubes with an even number of painted faces is 20 -/
theorem even_painted_faces_count (b : Block) :
  b.length = 6 → b.width = 3 → b.height = 2 →
  countEvenPaintedFaces b = 20 :=
by sorry

end NUMINAMATH_CALUDE_even_painted_faces_count_l1357_135711


namespace NUMINAMATH_CALUDE_platform_length_l1357_135703

/-- Given a train of length 600 meters that crosses a platform in 39 seconds
    and a signal pole in 18 seconds, prove that the platform length is 700 meters. -/
theorem platform_length (train_length : ℝ) (platform_cross_time : ℝ) (pole_cross_time : ℝ) :
  train_length = 600 ∧ 
  platform_cross_time = 39 ∧ 
  pole_cross_time = 18 →
  (train_length + (train_length / pole_cross_time * platform_cross_time - train_length)) = 700 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l1357_135703


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l1357_135714

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone already seated. -/
def valid_seating (table : CircularTable) : Prop :=
  table.seated_people > 0 ∧
  table.seated_people ≤ table.total_chairs ∧
  ∀ k : ℕ, k < table.total_chairs → 
    ∃ i j : ℕ, i < table.seated_people ∧ j < table.seated_people ∧
      (i * (table.total_chairs / table.seated_people) % table.total_chairs = k ∨
       j * (table.total_chairs / table.seated_people) % table.total_chairs = (k + 1) % table.total_chairs)

/-- The main theorem stating that 18 is the smallest number of people that can be validly seated. -/
theorem smallest_valid_seating :
  ∀ n : ℕ, n < 18 → ¬(valid_seating ⟨72, n⟩) ∧ 
  valid_seating ⟨72, 18⟩ := by
  sorry

#check smallest_valid_seating

end NUMINAMATH_CALUDE_smallest_valid_seating_l1357_135714


namespace NUMINAMATH_CALUDE_sum_equality_l1357_135752

theorem sum_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d)
  (sum_nonzero : a + b ≠ 0)
  (product_eq : a * c = b * d) : 
  a + c = b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_equality_l1357_135752


namespace NUMINAMATH_CALUDE_distance_to_destination_l1357_135763

/-- Proves that the distance to the destination is 2.25 kilometers given the specified conditions. -/
theorem distance_to_destination
  (rowing_speed : ℝ)
  (river_speed : ℝ)
  (round_trip_time : ℝ)
  (h1 : rowing_speed = 4)
  (h2 : river_speed = 2)
  (h3 : round_trip_time = 1.5)
  : ∃ (distance : ℝ),
    distance = 2.25 ∧
    round_trip_time = distance / (rowing_speed + river_speed) + distance / (rowing_speed - river_speed) :=
by
  sorry

end NUMINAMATH_CALUDE_distance_to_destination_l1357_135763


namespace NUMINAMATH_CALUDE_denominator_value_l1357_135789

theorem denominator_value (p q x : ℚ) : 
  p / q = 4 / 5 → 
  4 / 7 + (2 * q - p) / x = 1 → 
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_denominator_value_l1357_135789


namespace NUMINAMATH_CALUDE_inequality_proof_l1357_135772

theorem inequality_proof (b : ℝ) (n : ℕ) (h1 : b > 0) (h2 : n > 2) :
  let floor_b := ⌊b⌋
  let d := ((floor_b + 1 - b) * floor_b) / (floor_b + 1)
  (d + n - 2) / (floor_b + n - 2) > (floor_b + n - 1 - b) / (floor_b + n - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1357_135772


namespace NUMINAMATH_CALUDE_employee_hire_year_l1357_135780

/-- Rule of 70 retirement provision -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year the employee was hired -/
def hire_year : ℕ := sorry

/-- The employee's age when hired -/
def hire_age : ℕ := 32

/-- The year the employee became eligible for retirement -/
def retirement_year : ℕ := 2007

theorem employee_hire_year :
  rule_of_70 (hire_age + (retirement_year - hire_year)) (retirement_year - hire_year) ∧
  hire_year = 1969 := by
  sorry

end NUMINAMATH_CALUDE_employee_hire_year_l1357_135780


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sums_l1357_135765

theorem polynomial_coefficient_sums :
  ∀ (a a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (3 - 2 * x)^5 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) →
  (a₁ + a₂ + a₃ + a₄ + a₅ = -242) ∧
  (|a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 2882) := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sums_l1357_135765


namespace NUMINAMATH_CALUDE_complex_equation_proof_l1357_135757

/-- Given the complex equation (2+i)/(i+1) - 2i = a + bi, prove that b - ai = -5/2 - 3/2i --/
theorem complex_equation_proof (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (2 + i) / (i + 1) - 2 * i = a + b * i) : 
  b - a * i = -5/2 - 3/2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l1357_135757


namespace NUMINAMATH_CALUDE_pizza_equivalents_theorem_l1357_135712

/-- Calculates the total quantity of pizza equivalents served -/
def total_pizza_equivalents (lunch_pizzas : ℕ) (dinner_pizzas : ℕ) (lunch_calzones : ℕ) : ℕ :=
  lunch_pizzas + dinner_pizzas + (lunch_calzones / 2)

/-- Proves that the total quantity of pizza equivalents served is 17 -/
theorem pizza_equivalents_theorem :
  total_pizza_equivalents 9 6 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_pizza_equivalents_theorem_l1357_135712


namespace NUMINAMATH_CALUDE_total_pencils_l1357_135722

theorem total_pencils (jessica_pencils sandy_pencils jason_pencils : ℕ) 
  (h1 : jessica_pencils = 8)
  (h2 : sandy_pencils = 8)
  (h3 : jason_pencils = 8) :
  jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l1357_135722


namespace NUMINAMATH_CALUDE_equilateral_roots_ratio_l1357_135732

/-- Given complex numbers z₁ and z₂ that are roots of z² + pz + q = 0,
    where p and q are complex numbers, and 0, z₁, and z₂ form an
    equilateral triangle in the complex plane, then p²/q = 1. -/
theorem equilateral_roots_ratio (p q z₁ z₂ : ℂ) :
  z₁^2 + p*z₁ + q = 0 →
  z₂^2 + p*z₂ + q = 0 →
  ∃ (ω : ℂ), ω^3 = 1 ∧ ω ≠ 1 ∧ z₂ = ω * z₁ →
  p^2 / q = 1 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_roots_ratio_l1357_135732


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l1357_135797

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (2 : ℚ) / 3 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (2 : ℚ) / 3 → q ≤ q') →
  q - p = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l1357_135797


namespace NUMINAMATH_CALUDE_quadratic_roots_squared_l1357_135775

theorem quadratic_roots_squared (α β : ℝ) : 
  (α^2 - 3*α - 1 = 0) → 
  (β^2 - 3*β - 1 = 0) → 
  (α + β = 3) →
  (α * β = -1) →
  ((α^2)^2 - 11*(α^2) + 1 = 0) ∧ ((β^2)^2 - 11*(β^2) + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_squared_l1357_135775


namespace NUMINAMATH_CALUDE_min_sum_squares_l1357_135756

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_sum_squares (a b c d e f g h : Int)
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S)
  (he : e ∈ S) (hf : f ∈ S) (hg : g ∈ S) (hh : h ∈ S)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
               f ≠ g ∧ f ≠ h ∧
               g ≠ h) :
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1357_135756


namespace NUMINAMATH_CALUDE_analysis_time_proof_l1357_135795

/-- The number of bones in a human body -/
def num_bones : ℕ := 206

/-- The time spent analyzing each bone (in hours) -/
def time_per_bone : ℕ := 1

/-- The total time needed to analyze all bones in a human body -/
def total_analysis_time : ℕ := num_bones * time_per_bone

theorem analysis_time_proof : total_analysis_time = 206 := by
  sorry

end NUMINAMATH_CALUDE_analysis_time_proof_l1357_135795


namespace NUMINAMATH_CALUDE_nail_count_l1357_135762

/-- Given that Violet has 3 more than twice as many nails as Tickletoe and Violet has 27 nails, 
    prove that the total number of nails they have together is 39. -/
theorem nail_count (tickletoe_nails : ℕ) : 
  (2 * tickletoe_nails + 3 = 27) → (tickletoe_nails + 27 = 39) := by
  sorry

end NUMINAMATH_CALUDE_nail_count_l1357_135762


namespace NUMINAMATH_CALUDE_class_size_ratio_l1357_135768

/-- Given three classes A, B, and C, prove that the ratio of the size of Class A to Class C is 1/3 -/
theorem class_size_ratio (size_A size_B size_C : ℕ) : 
  size_A = 2 * size_B → 
  size_B = 20 → 
  size_C = 120 → 
  (size_A : ℚ) / size_C = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_class_size_ratio_l1357_135768


namespace NUMINAMATH_CALUDE_min_top_supervisors_bound_l1357_135760

/-- Represents the structure of a company --/
structure Company where
  total_employees : ℕ
  supervisor_subordinate_sum : ℕ
  propagation_days : ℕ

/-- Calculates the minimum number of top-level supervisors --/
def min_top_supervisors (c : Company) : ℕ :=
  ((c.total_employees - 1) / (1 + c.supervisor_subordinate_sum + c.supervisor_subordinate_sum ^ 2 + c.supervisor_subordinate_sum ^ 3 + c.supervisor_subordinate_sum ^ 4)) + 1

/-- The theorem to be proved --/
theorem min_top_supervisors_bound (c : Company) 
  (h1 : c.total_employees = 50000)
  (h2 : c.supervisor_subordinate_sum = 7)
  (h3 : c.propagation_days = 4) :
  min_top_supervisors c ≥ 97 := by
  sorry

#eval min_top_supervisors ⟨50000, 7, 4⟩

end NUMINAMATH_CALUDE_min_top_supervisors_bound_l1357_135760


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l1357_135792

theorem monotonic_increase_interval
  (f : ℝ → ℝ)
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < π / 2)
  (h_f : ∀ x, f x = Real.sin (ω * x + φ))
  (x₁ x₂ : ℝ)
  (h_x₁ : f x₁ = 1)
  (h_x₂ : f x₂ = 0)
  (h_x_diff : |x₁ - x₂| = 1 / 2)
  (h_f_half : f (1 / 2) = 1 / 2) :
  ∃ k : ℤ, StrictMonoOn f (Set.Icc (- 5 / 6 + 2 * k) (1 / 6 + 2 * k)) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l1357_135792


namespace NUMINAMATH_CALUDE_initial_eggs_count_l1357_135754

/-- The number of eggs initially in the box -/
def initial_eggs : ℕ := sorry

/-- The number of eggs Daniel adds to the box -/
def added_eggs : ℕ := 4

/-- The total number of eggs after Daniel adds eggs -/
def total_eggs : ℕ := 11

/-- Theorem stating that the initial number of eggs is 7 -/
theorem initial_eggs_count : initial_eggs = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_eggs_count_l1357_135754


namespace NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l1357_135769

/-- The volume of a cone formed by rolling up a three-quarter sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 4) :
  let sector_angle : ℝ := 3 * π / 2
  let base_radius : ℝ := sector_angle * r / (2 * π)
  let cone_height : ℝ := Real.sqrt (r^2 - base_radius^2)
  (1/3) * π * base_radius^2 * cone_height = 3 * π * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l1357_135769


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1357_135794

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -14/27
  let a₃ : ℚ := 56/243
  let r : ℚ := -16/27
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1357_135794


namespace NUMINAMATH_CALUDE_inscribed_squares_segment_product_l1357_135724

theorem inscribed_squares_segment_product :
  ∀ (small_area large_area : ℝ) (x : ℝ),
    small_area = 16 →
    large_area = 25 →
    x + 3*x = Real.sqrt large_area →
    x * (3*x) = 75/16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_segment_product_l1357_135724


namespace NUMINAMATH_CALUDE_johns_base_salary_l1357_135715

/-- John's monthly savings rate as a decimal -/
def savings_rate : ℝ := 0.10

/-- John's monthly savings amount in dollars -/
def savings_amount : ℝ := 400

/-- Theorem stating John's monthly base salary -/
theorem johns_base_salary :
  ∀ (base_salary : ℝ),
  base_salary * savings_rate = savings_amount →
  base_salary = 4000 := by
  sorry

end NUMINAMATH_CALUDE_johns_base_salary_l1357_135715


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l1357_135737

theorem absolute_value_equation_product (x₁ x₂ : ℝ) : 
  (|4 * x₁| + 3 = 35) ∧ (|4 * x₂| + 3 = 35) ∧ (x₁ ≠ x₂) → x₁ * x₂ = -64 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l1357_135737


namespace NUMINAMATH_CALUDE_four_inch_cube_painted_faces_l1357_135749

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Represents a painted cube -/
structure PaintedCube extends Cube where
  paintedFaces : ℕ

/-- Calculates the number of smaller cubes with at least two painted faces
    when a painted cube is cut into unit cubes -/
def numCubesWithTwoPaintedFaces (c : PaintedCube) : ℕ :=
  sorry

theorem four_inch_cube_painted_faces :
  let bigCube : PaintedCube := ⟨⟨4⟩, 6⟩
  numCubesWithTwoPaintedFaces bigCube = 32 := by sorry

end NUMINAMATH_CALUDE_four_inch_cube_painted_faces_l1357_135749


namespace NUMINAMATH_CALUDE_people_behind_yuna_l1357_135705

theorem people_behind_yuna (total : Nat) (in_front : Nat) (behind : Nat) : 
  total = 7 → in_front = 2 → behind = total - in_front - 1 → behind = 4 := by
  sorry

end NUMINAMATH_CALUDE_people_behind_yuna_l1357_135705


namespace NUMINAMATH_CALUDE_rectangle_area_l1357_135726

theorem rectangle_area (w : ℝ) (l : ℝ) (A : ℝ) (P : ℝ) : 
  l = w + 6 →
  A = w * l →
  P = 2 * (w + l) →
  A = 2 * P →
  w = 3 →
  A = 27 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1357_135726


namespace NUMINAMATH_CALUDE_sum_of_sequence_l1357_135773

/-- Calculates the sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

/-- The number of terms in the sequence -/
def num_terms (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

theorem sum_of_sequence : 
  let a₁ := 71  -- First term
  let aₙ := 361 -- Last term
  let d := 10   -- Common difference
  let n := num_terms a₁ aₙ d
  arithmetic_sum a₁ aₙ n = 6480 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_sequence_l1357_135773


namespace NUMINAMATH_CALUDE_slope_one_fourth_implies_y_six_l1357_135796

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q is 1/4, then the y-coordinate of Q is 6. -/
theorem slope_one_fourth_implies_y_six (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = -3 →
  y₁ = 4 →
  x₂ = 5 →
  (y₂ - y₁) / (x₂ - x₁) = 1/4 →
  y₂ = 6 :=
by sorry

end NUMINAMATH_CALUDE_slope_one_fourth_implies_y_six_l1357_135796


namespace NUMINAMATH_CALUDE_correct_units_l1357_135729

/-- Represents the number of units in a building project -/
structure BuildingProject where
  first_building : ℕ
  second_building : ℕ
  third_building : ℕ
  apartments : ℕ
  condos : ℕ
  townhouses : ℕ
  bungalows : ℕ

/-- Calculates the correct number of units for each type in the building project -/
def calculate_units (project : BuildingProject) : Prop :=
  -- First building conditions
  project.first_building = 4000 ∧
  project.apartments ≥ 2000 ∧
  project.condos ≥ 2000 ∧
  -- Second building conditions
  project.second_building = (2 : ℕ) * project.first_building / 5 ∧
  -- Third building conditions
  project.third_building = (6 : ℕ) * project.second_building / 5 ∧
  project.townhouses = (3 : ℕ) * project.third_building / 5 ∧
  project.bungalows = (2 : ℕ) * project.third_building / 5 ∧
  -- Total units calculation
  project.apartments = 3200 ∧
  project.condos = 2400 ∧
  project.townhouses = 1152 ∧
  project.bungalows = 768

/-- Theorem stating that the calculated units are correct -/
theorem correct_units (project : BuildingProject) : 
  calculate_units project → 
  project.apartments = 3200 ∧ 
  project.condos = 2400 ∧ 
  project.townhouses = 1152 ∧ 
  project.bungalows = 768 := by
  sorry

end NUMINAMATH_CALUDE_correct_units_l1357_135729


namespace NUMINAMATH_CALUDE_three_can_volume_l1357_135758

theorem three_can_volume : 
  ∀ (v1 v2 v3 : ℕ),
  v2 = (3 * v1) / 2 →
  v3 = 64 * v1 / 3 →
  v1 + v2 + v3 < 30 →
  v1 + v2 + v3 = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_three_can_volume_l1357_135758


namespace NUMINAMATH_CALUDE_clever_calculation_l1357_135725

theorem clever_calculation :
  (46.3 * 0.56 + 5.37 * 5.6 + 1 * 0.056 = 56.056) ∧
  (101 * 92 - 92 = 9200) ∧
  (36000 / 125 / 8 = 36) := by
  sorry

end NUMINAMATH_CALUDE_clever_calculation_l1357_135725


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1357_135777

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def common_difference (a : ℕ → ℚ) : ℚ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_condition1 : a 7 - 2 * a 4 = -1)
  (h_condition2 : a 3 = 0) :
  common_difference a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1357_135777


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1357_135743

theorem mod_equivalence_unique_solution :
  ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1357_135743


namespace NUMINAMATH_CALUDE_solve_for_y_l1357_135786

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1357_135786


namespace NUMINAMATH_CALUDE_sequence_is_increasing_l1357_135764

def isIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

theorem sequence_is_increasing (a : ℕ → ℝ) 
    (h : ∀ n, a (n + 1) - a n - 3 = 0) : 
    isIncreasing a := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_increasing_l1357_135764


namespace NUMINAMATH_CALUDE_food_festival_total_cost_l1357_135751

def food_festival_cost (hot_dog_price1 hot_dog_price2 hot_dog_price3 : ℚ)
                       (ice_cream_price1 ice_cream_price2 : ℚ)
                       (lemonade_price1 lemonade_price2 lemonade_price3 : ℚ) : ℚ :=
  3 * hot_dog_price1 + 3 * hot_dog_price2 + 2 * hot_dog_price3 +
  2 * ice_cream_price1 + 3 * ice_cream_price2 +
  lemonade_price1 + lemonade_price2 + lemonade_price3

theorem food_festival_total_cost :
  food_festival_cost 0.60 0.75 0.90 1.50 2.00 2.50 3.00 3.50 = 23.85 := by
  sorry

end NUMINAMATH_CALUDE_food_festival_total_cost_l1357_135751


namespace NUMINAMATH_CALUDE_anthony_total_pencils_l1357_135799

def initial_pencils : ℕ := 9
def gifted_pencils : ℕ := 56

theorem anthony_total_pencils : 
  initial_pencils + gifted_pencils = 65 := by
  sorry

end NUMINAMATH_CALUDE_anthony_total_pencils_l1357_135799


namespace NUMINAMATH_CALUDE_product_scaling_l1357_135783

theorem product_scaling (a b c : ℝ) (h : 14.97 * 46 = 688.62) : 
  1.497 * 4.6 = 6.8862 := by sorry

end NUMINAMATH_CALUDE_product_scaling_l1357_135783


namespace NUMINAMATH_CALUDE_min_positive_period_of_f_l1357_135782

open Real

noncomputable def f (x : ℝ) : ℝ := (sin x - Real.sqrt 3 * cos x) * (cos x - Real.sqrt 3 * sin x)

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = π :=
sorry

end NUMINAMATH_CALUDE_min_positive_period_of_f_l1357_135782


namespace NUMINAMATH_CALUDE_misha_max_cities_l1357_135770

/-- The maximum number of cities Misha can visit -/
def max_cities_visited (n k : ℕ) : ℕ :=
  if k ≥ n - 3 then min (n - k) 2 else n - k

/-- Theorem stating the maximum number of cities Misha can visit -/
theorem misha_max_cities (n k : ℕ) (h1 : n ≥ 2) (h2 : k ≥ 1) :
  max_cities_visited n k = 
    if k ≥ n - 3 then min (n - k) 2 else n - k :=
by sorry

end NUMINAMATH_CALUDE_misha_max_cities_l1357_135770


namespace NUMINAMATH_CALUDE_sqrt_of_negative_nine_squared_l1357_135734

theorem sqrt_of_negative_nine_squared : Real.sqrt ((-9)^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_negative_nine_squared_l1357_135734


namespace NUMINAMATH_CALUDE_birthday_money_possibility_l1357_135788

theorem birthday_money_possibility (x y : ℕ) : ∃ (a : ℕ), 
  a < 10 ∧ 
  (x * y) % 10 = a ∧ 
  ((x + 1) * (y + 1)) % 10 = a ∧ 
  ((x + 2) * (y + 2)) % 10 = 0 := by
sorry

end NUMINAMATH_CALUDE_birthday_money_possibility_l1357_135788


namespace NUMINAMATH_CALUDE_lower_selling_price_l1357_135791

theorem lower_selling_price 
  (cost : ℕ) 
  (higher_price lower_price : ℕ) 
  (h1 : cost = 400)
  (h2 : higher_price = 600)
  (h3 : (higher_price - cost) = (lower_price - cost) + (cost * 5 / 100)) :
  lower_price = 580 := by
sorry

end NUMINAMATH_CALUDE_lower_selling_price_l1357_135791


namespace NUMINAMATH_CALUDE_distance_sum_squares_constant_l1357_135742

/-- Two concentric circles with radii R₁ and R₂ -/
structure ConcentricCircles (R₁ R₂ : ℝ) where
  center : ℝ × ℝ

/-- A point on a circle -/
structure PointOnCircle (c : ℝ × ℝ) (R : ℝ) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.1)^2 + (point.2 - c.2)^2 = R^2

/-- A diameter of a circle -/
structure Diameter (c : ℝ × ℝ) (R : ℝ) where
  endpointA : ℝ × ℝ
  endpointB : ℝ × ℝ
  is_diameter : (endpointA.1 - c.1)^2 + (endpointA.2 - c.2)^2 = R^2 ∧
                (endpointB.1 - c.1)^2 + (endpointB.2 - c.2)^2 = R^2 ∧
                (endpointA.1 - endpointB.1)^2 + (endpointA.2 - endpointB.2)^2 = 4 * R^2

/-- The theorem statement -/
theorem distance_sum_squares_constant
  (R₁ R₂ : ℝ) (circles : ConcentricCircles R₁ R₂)
  (C : PointOnCircle circles.center R₂)
  (AB : Diameter circles.center R₁) :
  let distAC := ((AB.endpointA.1 - C.point.1)^2 + (AB.endpointA.2 - C.point.2)^2)
  let distBC := ((AB.endpointB.1 - C.point.1)^2 + (AB.endpointB.2 - C.point.2)^2)
  distAC + distBC = 2 * R₁^2 + 2 * R₂^2 := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_squares_constant_l1357_135742


namespace NUMINAMATH_CALUDE_largest_n_with_triangle_property_l1357_135723

/-- A set of consecutive positive integers has the triangle property for all 9-element subsets -/
def has_triangle_property (s : Set ℕ) : Prop :=
  ∀ (x y z : ℕ), x ∈ s → y ∈ s → z ∈ s → x < y → y < z → z < x + y

/-- The set of consecutive positive integers from 6 to n -/
def consecutive_set (n : ℕ) : Set ℕ :=
  {x : ℕ | 6 ≤ x ∧ x ≤ n}

/-- The theorem stating that 224 is the largest possible value of n -/
theorem largest_n_with_triangle_property :
  ∀ n : ℕ, (has_triangle_property (consecutive_set n)) → n ≤ 224 :=
sorry

end NUMINAMATH_CALUDE_largest_n_with_triangle_property_l1357_135723


namespace NUMINAMATH_CALUDE_total_marbles_count_l1357_135778

theorem total_marbles_count (num_jars : ℕ) (marbles_per_jar : ℕ) : 
  num_jars = 16 →
  marbles_per_jar = 5 →
  (∃ (num_clay_pots : ℕ), num_jars = 2 * num_clay_pots) →
  (∃ (total_marbles : ℕ), 
    total_marbles = num_jars * marbles_per_jar + 
                    (num_jars / 2) * (3 * marbles_per_jar) ∧
    total_marbles = 200) :=
by
  sorry

#check total_marbles_count

end NUMINAMATH_CALUDE_total_marbles_count_l1357_135778


namespace NUMINAMATH_CALUDE_magazine_cost_l1357_135761

theorem magazine_cost (total_books : ℕ) (num_magazines : ℕ) (book_cost : ℕ) (total_spent : ℕ) : 
  total_books = 16 → 
  num_magazines = 3 → 
  book_cost = 11 → 
  total_spent = 179 → 
  (total_spent - total_books * book_cost) / num_magazines = 1 := by
sorry

end NUMINAMATH_CALUDE_magazine_cost_l1357_135761


namespace NUMINAMATH_CALUDE_ones_12_div_13_ones_16_div_17_l1357_135746

/-- The number formed by n consecutive ones -/
def ones (n : ℕ) : ℕ := (10^n - 1) / 9

/-- Theorem: The number formed by 12 consecutive ones is divisible by 13 -/
theorem ones_12_div_13 : 13 ∣ ones 12 := by sorry

/-- Theorem: The number formed by 16 consecutive ones is divisible by 17 -/
theorem ones_16_div_17 : 17 ∣ ones 16 := by sorry

end NUMINAMATH_CALUDE_ones_12_div_13_ones_16_div_17_l1357_135746


namespace NUMINAMATH_CALUDE_length_of_AB_l1357_135739

/-- Given two points P and Q on a line segment AB, prove that AB has length 35 -/
theorem length_of_AB (A B P Q : ℝ) : 
  (0 < A ∧ A < P ∧ P < Q ∧ Q < B) →  -- P and Q are on AB and on the same side of midpoint
  (P - A) / (B - P) = 1 / 4 →        -- P divides AB in ratio 1:4
  (Q - A) / (B - Q) = 2 / 5 →        -- Q divides AB in ratio 2:5
  Q - P = 3 →                        -- Distance PQ = 3
  B - A = 35 := by                   -- Length of AB is 35
sorry


end NUMINAMATH_CALUDE_length_of_AB_l1357_135739


namespace NUMINAMATH_CALUDE_min_value_of_m_minus_n_l1357_135741

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1

noncomputable def g (x : ℝ) : ℝ := 2 * Real.exp (x - 1/2)

theorem min_value_of_m_minus_n (m n : ℝ) (h : f m = g n) :
  ∃ (k : ℝ), k = 1/2 + Real.log 2 ∧ m - n ≥ k := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_m_minus_n_l1357_135741


namespace NUMINAMATH_CALUDE_songs_added_l1357_135784

theorem songs_added (initial : ℕ) (deleted : ℕ) (final : ℕ) 
  (h1 : initial = 30) 
  (h2 : deleted = 8) 
  (h3 : final = 32) : 
  final - (initial - deleted) = 10 := by
  sorry

end NUMINAMATH_CALUDE_songs_added_l1357_135784


namespace NUMINAMATH_CALUDE_initial_oranges_count_l1357_135759

/-- The number of oranges Susan took from the box -/
def oranges_taken : ℕ := 35

/-- The number of oranges left in the box -/
def oranges_left : ℕ := 20

/-- The initial number of oranges in the box -/
def initial_oranges : ℕ := oranges_taken + oranges_left

theorem initial_oranges_count : initial_oranges = 55 := by
  sorry

end NUMINAMATH_CALUDE_initial_oranges_count_l1357_135759


namespace NUMINAMATH_CALUDE_probability_of_drawing_red_ball_l1357_135747

theorem probability_of_drawing_red_ball (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) :
  total_balls = red_balls + black_balls →
  red_balls = 3 →
  black_balls = 3 →
  (red_balls : ℚ) / (total_balls : ℚ) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_drawing_red_ball_l1357_135747


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l1357_135793

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

theorem parallel_vectors_sum (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → a.1 + b.1 = -2 ∧ a.2 + b.2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l1357_135793


namespace NUMINAMATH_CALUDE_custom_operation_equality_l1357_135731

/-- Custom operation ⊕ for real numbers -/
def circle_plus (a b : ℝ) : ℝ := (a + b) ^ 2

/-- Theorem stating the equality for the given expression -/
theorem custom_operation_equality (x y : ℝ) : 
  circle_plus (circle_plus ((x + y) ^ 2) ((y + x) ^ 2)) 2 = 4 * ((x + y) ^ 2 + 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equality_l1357_135731


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l1357_135701

theorem smaller_number_in_ratio (a b : ℝ) : 
  a / b = 3 / 4 → a + b = 420 → a = 180 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l1357_135701


namespace NUMINAMATH_CALUDE_triangle_count_l1357_135781

/-- The total number of triangles in a specially divided rectangle -/
def total_triangles (small_right : ℕ) (isosceles_quarter_width : ℕ) (isosceles_third_length : ℕ) (larger_right : ℕ) (large_isosceles : ℕ) : ℕ :=
  small_right + isosceles_quarter_width + isosceles_third_length + larger_right + large_isosceles

/-- Theorem stating the total number of triangles in the specially divided rectangle -/
theorem triangle_count :
  total_triangles 24 8 12 16 4 = 64 := by sorry

end NUMINAMATH_CALUDE_triangle_count_l1357_135781


namespace NUMINAMATH_CALUDE_grape_jelly_beans_problem_l1357_135706

theorem grape_jelly_beans_problem (g c : ℕ) : 
  g = 3 * c →                   -- Initial ratio
  g - 15 = 5 * (c - 5) →        -- Final ratio after eating
  g = 15                        -- Conclusion: original number of grape jelly beans
  := by sorry

end NUMINAMATH_CALUDE_grape_jelly_beans_problem_l1357_135706


namespace NUMINAMATH_CALUDE_three_positions_from_six_people_l1357_135718

/-- The number of ways to select 3 distinct positions from a group of 6 people. -/
def select_positions (n : ℕ) (k : ℕ) : ℕ :=
  n * (n - 1) * (n - 2)

/-- Theorem stating that selecting 3 distinct positions from 6 people results in 120 ways. -/
theorem three_positions_from_six_people :
  select_positions 6 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_three_positions_from_six_people_l1357_135718

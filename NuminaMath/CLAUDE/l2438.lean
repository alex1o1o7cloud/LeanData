import Mathlib

namespace NUMINAMATH_CALUDE_marbles_exceed_200_on_friday_l2438_243812

def marbles (k : ℕ) : ℕ := 4 * 3^k

theorem marbles_exceed_200_on_friday :
  (∀ j : ℕ, j < 4 → marbles j ≤ 200) ∧ marbles 4 > 200 :=
sorry

end NUMINAMATH_CALUDE_marbles_exceed_200_on_friday_l2438_243812


namespace NUMINAMATH_CALUDE_smallest_valid_configuration_l2438_243819

/-- Represents a bench configuration at a concert --/
structure BenchConfiguration where
  M : ℕ  -- Number of bench sections
  adultsPerBench : ℕ  -- Number of adults per bench
  childrenPerBench : ℕ  -- Number of children per bench

/-- Checks if a given bench configuration is valid --/
def isValidConfiguration (config : BenchConfiguration) : Prop :=
  ∃ (adults children : ℕ),
    adults + children = config.M * config.adultsPerBench ∧
    children = 2 * adults ∧
    children ≤ config.M * config.childrenPerBench

/-- The theorem to be proved --/
theorem smallest_valid_configuration :
  ∃ (config : BenchConfiguration),
    config.M = 6 ∧
    config.adultsPerBench = 8 ∧
    config.childrenPerBench = 12 ∧
    isValidConfiguration config ∧
    (∀ (otherConfig : BenchConfiguration),
      otherConfig.adultsPerBench = 8 →
      otherConfig.childrenPerBench = 12 →
      isValidConfiguration otherConfig →
      otherConfig.M ≥ config.M) :=
  sorry

end NUMINAMATH_CALUDE_smallest_valid_configuration_l2438_243819


namespace NUMINAMATH_CALUDE_grid_sign_flip_l2438_243863

-- Define the grid
def Grid (m n : ℕ) := Fin m → Fin n → Int

-- Define the sign-flipping operation
def flip_signs (g : Grid m n) (i j : ℕ) : Grid m n :=
  sorry

-- Define the property of all signs being flipped
def all_signs_flipped (g g' : Grid m n) : Prop :=
  ∀ i j, g' i j = -g i j

-- Theorem statement
theorem grid_sign_flip (m n : ℕ) :
  (∃ (ops : List (ℕ × ℕ)) (g : Grid m n),
    all_signs_flipped g (ops.foldl (λ acc (i, j) => flip_signs acc i j) g)) ↔
  (∃ k l : ℕ, m = 4 * k ∧ n = 4 * l) :=
sorry

end NUMINAMATH_CALUDE_grid_sign_flip_l2438_243863


namespace NUMINAMATH_CALUDE_ducks_remaining_theorem_l2438_243875

def ducks_remaining (initial : ℕ) : ℕ :=
  let after_first := initial - (initial / 4)
  let after_second := after_first - (after_first / 6)
  after_second - (after_second * 3 / 10)

theorem ducks_remaining_theorem :
  ducks_remaining 320 = 140 := by
  sorry

end NUMINAMATH_CALUDE_ducks_remaining_theorem_l2438_243875


namespace NUMINAMATH_CALUDE_security_system_probability_l2438_243807

theorem security_system_probability (p : ℝ) : 
  (1/8 : ℝ) * (1 - p) + (7/8 : ℝ) * p = 9/40 → p = 2/15 := by
sorry

end NUMINAMATH_CALUDE_security_system_probability_l2438_243807


namespace NUMINAMATH_CALUDE_star_equation_solution_l2438_243811

/-- The "※" operation for positive real numbers -/
def star (a b : ℝ) : ℝ := a * b + a + b^2

/-- Theorem: If 1※k = 3, then k = 1 -/
theorem star_equation_solution (k : ℝ) (hk : k > 0) (h : star 1 k = 3) : k = 1 := by
  sorry

#check star_equation_solution

end NUMINAMATH_CALUDE_star_equation_solution_l2438_243811


namespace NUMINAMATH_CALUDE_train_seats_problem_l2438_243840

theorem train_seats_problem (total_cars : ℕ) 
  (half_free : ℕ) (third_free : ℕ) (all_occupied : ℕ)
  (h1 : total_cars = 18)
  (h2 : half_free + third_free + all_occupied = total_cars)
  (h3 : (half_free * 6 + third_free * 4) * 2 = total_cars * 4) :
  all_occupied = 13 := by
  sorry

end NUMINAMATH_CALUDE_train_seats_problem_l2438_243840


namespace NUMINAMATH_CALUDE_infinite_sum_equals_floor_l2438_243847

noncomputable def infiniteSum (x : ℝ) : ℕ → ℝ
  | 0 => ⌊(x + 1) / 2⌋
  | n + 1 => infiniteSum x n + ⌊(x + 2^(n+1)) / 2^(n+2)⌋

theorem infinite_sum_equals_floor (x : ℝ) :
  (∀ y : ℝ, ⌊2 * y⌋ = ⌊y⌋ + ⌊y + 1/2⌋) →
  (∃ N : ℕ, ∀ n ≥ N, ⌊(x + 2^n) / 2^(n+1)⌋ = 0) →
  (∃ M : ℕ, ∀ m ≥ M, infiniteSum x m = ⌊x⌋) :=
by sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_floor_l2438_243847


namespace NUMINAMATH_CALUDE_unique_distance_l2438_243817

/-- A two-digit number is represented as 10a + b where a and b are single digits -/
def two_digit_number (a b : ℕ) : Prop := 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

/-- Inserting a zero between digits of a two-digit number -/
def insert_zero (a b : ℕ) : ℕ := 100 * a + b

/-- The property that inserting a zero results in 9 times the original number -/
def nine_times_property (a b : ℕ) : Prop :=
  insert_zero a b = 9 * (10 * a + b)

theorem unique_distance : 
  ∀ a b : ℕ, two_digit_number a b → nine_times_property a b → a = 4 ∧ b = 5 := by
  sorry

#check unique_distance

end NUMINAMATH_CALUDE_unique_distance_l2438_243817


namespace NUMINAMATH_CALUDE_fish_remaining_l2438_243865

theorem fish_remaining (initial : ℝ) (given_away : ℝ) (remaining : ℝ) : 
  initial = 47.0 → given_away = 22.0 → remaining = initial - given_away → remaining = 25.0 := by
  sorry

end NUMINAMATH_CALUDE_fish_remaining_l2438_243865


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2438_243879

theorem trigonometric_identity (α : ℝ) :
  (Real.cos (4 * α - 3 * Real.pi) ^ 2 - 4 * Real.cos (2 * α - Real.pi) ^ 2 + 3) /
  (Real.cos (4 * α + 3 * Real.pi) ^ 2 + 4 * Real.cos (2 * α + Real.pi) ^ 2 - 1) =
  Real.tan (2 * α) ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2438_243879


namespace NUMINAMATH_CALUDE_circle_diameter_l2438_243859

theorem circle_diameter (A : Real) (r : Real) (d : Real) : 
  A = Real.pi * r^2 → A = 64 * Real.pi → d = 2 * r → d = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l2438_243859


namespace NUMINAMATH_CALUDE_taxi_fare_equality_l2438_243890

/-- Taxi fare calculation problem -/
theorem taxi_fare_equality (mike_start_fee annie_start_fee annie_toll_fee : ℚ)
  (per_mile_rate : ℚ) (annie_miles : ℚ) :
  mike_start_fee = 2.5 ∧
  annie_start_fee = 2.5 ∧
  annie_toll_fee = 5 ∧
  per_mile_rate = 0.25 ∧
  annie_miles = 22 →
  ∃ (mike_miles : ℚ),
    mike_start_fee + per_mile_rate * mike_miles =
    annie_start_fee + annie_toll_fee + per_mile_rate * annie_miles ∧
    mike_miles = 42 :=
by sorry

end NUMINAMATH_CALUDE_taxi_fare_equality_l2438_243890


namespace NUMINAMATH_CALUDE_mingi_initial_tomatoes_l2438_243882

/-- The number of cherry tomatoes Mingi gave to each classmate -/
def tomatoes_per_classmate : ℕ := 15

/-- The number of classmates Mingi gave cherry tomatoes to -/
def number_of_classmates : ℕ := 20

/-- The number of cherry tomatoes Mingi had left after giving them away -/
def remaining_tomatoes : ℕ := 6

/-- The total number of cherry tomatoes Mingi had initially -/
def initial_tomatoes : ℕ := tomatoes_per_classmate * number_of_classmates + remaining_tomatoes

theorem mingi_initial_tomatoes : initial_tomatoes = 306 := by
  sorry

end NUMINAMATH_CALUDE_mingi_initial_tomatoes_l2438_243882


namespace NUMINAMATH_CALUDE_silverware_probability_l2438_243877

def forks : ℕ := 8
def spoons : ℕ := 10
def knives : ℕ := 6

def total_silverware : ℕ := forks + spoons + knives

def favorable_outcomes : ℕ := forks * spoons * knives

def total_outcomes : ℕ := Nat.choose total_silverware 3

theorem silverware_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 120 / 506 := by
  sorry

end NUMINAMATH_CALUDE_silverware_probability_l2438_243877


namespace NUMINAMATH_CALUDE_lcm_of_given_numbers_l2438_243802

theorem lcm_of_given_numbers : 
  Nat.lcm 24 (Nat.lcm 30 (Nat.lcm 40 (Nat.lcm 50 60))) = 600 := by sorry

end NUMINAMATH_CALUDE_lcm_of_given_numbers_l2438_243802


namespace NUMINAMATH_CALUDE_second_number_difference_l2438_243852

theorem second_number_difference (first_number second_number : ℤ) : 
  first_number = 15 →
  second_number = 55 →
  first_number + second_number = 70 →
  second_number - 3 * first_number = 10 := by
sorry

end NUMINAMATH_CALUDE_second_number_difference_l2438_243852


namespace NUMINAMATH_CALUDE_circle_ratio_l2438_243843

theorem circle_ratio (r R : ℝ) (h1 : r > 0) (h2 : R > 0) (h3 : r ≤ R) : 
  π * R^2 = 3 * (π * R^2 - π * r^2) → R / r = Real.sqrt (3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l2438_243843


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l2438_243878

theorem ratio_of_sum_to_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l2438_243878


namespace NUMINAMATH_CALUDE_solve_for_y_l2438_243808

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2438_243808


namespace NUMINAMATH_CALUDE_simplify_expression_l2438_243855

theorem simplify_expression (x : ℝ) : (3 * x)^4 - (4 * x) * (x^3) = 77 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2438_243855


namespace NUMINAMATH_CALUDE_max_value_in_equation_max_value_achievable_l2438_243816

/-- Represents a three-digit number composed of different non-zero digits from 1 to 9 -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 1 ≤ z ∧ z ≤ 9 ∧ n = 100 * x + 10 * y + z) }

/-- The main theorem stating the maximum value of a in the given equation -/
theorem max_value_in_equation (a b c d : ThreeDigitNumber) 
  (h : 1984 - a.val = 2015 - b.val - c.val - d.val) : 
  a.val ≤ 214 := by
  sorry

/-- The theorem proving that 214 is achievable -/
theorem max_value_achievable : 
  ∃ (a b c d : ThreeDigitNumber), 1984 - a.val = 2015 - b.val - c.val - d.val ∧ a.val = 214 := by
  sorry

end NUMINAMATH_CALUDE_max_value_in_equation_max_value_achievable_l2438_243816


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2438_243854

/-- The eccentricity of a hyperbola with asymptotic lines y = ±(3/2)x is either √13/2 or √13/3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  (b / a = 3 / 2 ∨ a / b = 3 / 2) →
  c^2 = a^2 + b^2 →
  (c / a = Real.sqrt 13 / 2 ∨ c / a = Real.sqrt 13 / 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2438_243854


namespace NUMINAMATH_CALUDE_theater_seats_l2438_243834

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increment : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increment + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- Theorem stating that a theater with given conditions has 570 seats -/
theorem theater_seats :
  ∃ (t : Theater), t.first_row_seats = 12 ∧ t.seat_increment = 2 ∧ t.last_row_seats = 48 ∧ total_seats t = 570 :=
by
  sorry


end NUMINAMATH_CALUDE_theater_seats_l2438_243834


namespace NUMINAMATH_CALUDE_sum_ends_with_1379_l2438_243864

theorem sum_ends_with_1379 (S : Finset ℕ) (h1 : S.card = 10000) 
  (h2 : ∀ n ∈ S, Odd n ∧ ¬(5 ∣ n)) : 
  ∃ T ⊆ S, (T.sum id) % 10000 = 1379 := by
sorry

end NUMINAMATH_CALUDE_sum_ends_with_1379_l2438_243864


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l2438_243857

theorem smallest_positive_integer_with_given_remainders :
  ∃ b : ℕ, b > 0 ∧ 
    b % 5 = 4 ∧ 
    b % 7 = 6 ∧ 
    b % 11 = 10 ∧ 
    (∀ c : ℕ, c > 0 ∧ c % 5 = 4 ∧ c % 7 = 6 ∧ c % 11 = 10 → b ≤ c) ∧
    b = 384 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l2438_243857


namespace NUMINAMATH_CALUDE_square_9801_property_l2438_243803

theorem square_9801_property (y : ℤ) (h : y^2 = 9801) : (y + 2) * (y - 2) = 9797 := by
  sorry

end NUMINAMATH_CALUDE_square_9801_property_l2438_243803


namespace NUMINAMATH_CALUDE_jackies_tree_climbing_ratio_l2438_243860

/-- Given the following conditions about Jackie's tree climbing:
  - Jackie climbed 4 trees in total
  - The first tree is 1000 feet tall
  - Two trees are of equal height
  - The fourth tree is 200 feet taller than the first tree
  - The average height of all trees is 800 feet

  Prove that the ratio of the height of the two equal trees to the height of the first tree is 1:2.
-/
theorem jackies_tree_climbing_ratio :
  ∀ (h₁ h₂ h₄ : ℝ),
  h₁ = 1000 →
  h₄ = h₁ + 200 →
  (h₁ + 2 * h₂ + h₄) / 4 = 800 →
  h₂ / h₁ = 1 / 2 :=
by sorry


end NUMINAMATH_CALUDE_jackies_tree_climbing_ratio_l2438_243860


namespace NUMINAMATH_CALUDE_albert_took_five_candies_l2438_243871

/-- The number of candies Albert took away -/
def candies_taken (initial final : ℕ) : ℕ := initial - final

/-- Proof that Albert took 5 candies -/
theorem albert_took_five_candies :
  candies_taken 76 71 = 5 := by
  sorry

end NUMINAMATH_CALUDE_albert_took_five_candies_l2438_243871


namespace NUMINAMATH_CALUDE_special_function_value_l2438_243835

/-- A function satisfying the given property -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y

theorem special_function_value (f : ℝ → ℝ) 
  (h : SpecialFunction f) (h250 : f 250 = 4) : f 300 = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l2438_243835


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2438_243826

/-- Given a geometric sequence {a_n} with specific conditions, prove that a_1 = -1/2 -/
theorem geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) 
  (h_product : a 2 * a 5 * a 8 = -8) 
  (h_sum : a 1 + a 2 + a 3 = a 2 + 3 * a 1) : 
  a 1 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2438_243826


namespace NUMINAMATH_CALUDE_mildred_total_oranges_l2438_243806

/-- The number of oranges Mildred initially collected -/
def initial_oranges : ℕ := 77

/-- The number of oranges Mildred's father gave her -/
def additional_oranges : ℕ := 2

/-- Theorem: Mildred's total number of oranges is 79 -/
theorem mildred_total_oranges : 
  initial_oranges + additional_oranges = 79 := by
  sorry

end NUMINAMATH_CALUDE_mildred_total_oranges_l2438_243806


namespace NUMINAMATH_CALUDE_factorial_fraction_l2438_243895

theorem factorial_fraction (N : ℕ) : 
  (Nat.factorial (N + 1) * (N + 2)) / Nat.factorial (N + 3) = 1 / (N + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_l2438_243895


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l2438_243821

-- Problem 1
theorem inequality_one (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := by sorry

-- Problem 2
theorem inequality_two (a b : ℝ) (h1 : |a| < 1) (h2 : |b| < 1) :
  |1 - a * b| > |a - b| := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l2438_243821


namespace NUMINAMATH_CALUDE_puppies_ratio_l2438_243869

/-- Puppies problem -/
theorem puppies_ratio (total : ℕ) (kept : ℕ) (price : ℕ) (stud_fee : ℕ) (profit : ℕ) :
  total = 8 →
  kept = 1 →
  price = 600 →
  stud_fee = 300 →
  profit = 1500 →
  (total - kept - (profit + stud_fee) / price : ℚ) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_puppies_ratio_l2438_243869


namespace NUMINAMATH_CALUDE_girls_in_second_grade_proof_l2438_243861

/-- The number of girls in the second grade -/
def girls_in_second_grade : ℕ := 11

/-- The number of boys in the second grade -/
def boys_in_second_grade : ℕ := 20

/-- The total number of students in grades 2 and 3 -/
def total_students : ℕ := 93

theorem girls_in_second_grade_proof :
  girls_in_second_grade = 11 :=
by
  have h1 : boys_in_second_grade = 20 := rfl
  have h2 : total_students = 93 := rfl
  have h3 : (boys_in_second_grade + girls_in_second_grade) * 3 = total_students :=
    sorry
  sorry

#check girls_in_second_grade_proof

end NUMINAMATH_CALUDE_girls_in_second_grade_proof_l2438_243861


namespace NUMINAMATH_CALUDE_four_digit_sum_27_eq_3276_l2438_243822

/-- The number of four-digit whole numbers whose digits sum to 27 -/
def four_digit_sum_27 : ℕ :=
  (Finset.range 10).sum (fun a =>
    (Finset.range 10).sum (fun b =>
      (Finset.range 10).sum (fun c =>
        (Finset.range 10).sum (fun d =>
          if a ≥ 1 ∧ a + b + c + d = 27 then 1 else 0))))

theorem four_digit_sum_27_eq_3276 : four_digit_sum_27 = 3276 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_sum_27_eq_3276_l2438_243822


namespace NUMINAMATH_CALUDE_scientific_notation_of_32_9_billion_l2438_243888

def billion : ℝ := 1000000000

theorem scientific_notation_of_32_9_billion :
  32.9 * billion = 3.29 * (10 : ℝ)^9 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_32_9_billion_l2438_243888


namespace NUMINAMATH_CALUDE_family_ages_solution_l2438_243809

/-- Represents the ages of Priya and her parents -/
structure FamilyAges where
  priya : ℕ
  father : ℕ
  mother : ℕ

/-- Conditions for the family ages problem -/
def FamilyAgesProblem (ages : FamilyAges) : Prop :=
  ages.father - ages.priya = 31 ∧
  ages.father + 8 + ages.priya + 8 = 69 ∧
  ages.father - ages.mother = 4 ∧
  ages.priya + 5 + ages.mother + 5 = 65

/-- Theorem stating the solution to the family ages problem -/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), FamilyAgesProblem ages ∧ 
    ages.priya = 11 ∧ ages.father = 42 ∧ ages.mother = 38 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l2438_243809


namespace NUMINAMATH_CALUDE_student_number_factor_l2438_243899

theorem student_number_factor (x f : ℝ) : 
  x = 110 → x * f - 220 = 110 → f = 3 := by
  sorry

end NUMINAMATH_CALUDE_student_number_factor_l2438_243899


namespace NUMINAMATH_CALUDE_cube_root_cube_equality_l2438_243813

theorem cube_root_cube_equality (x : ℝ) : x = (x^3)^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_cube_equality_l2438_243813


namespace NUMINAMATH_CALUDE_circus_ticket_sales_l2438_243886

theorem circus_ticket_sales (total_tickets : ℕ) (adult_price children_price : ℚ) 
  (total_receipts : ℚ) (h1 : total_tickets = 522) 
  (h2 : adult_price = 15) (h3 : children_price = 8) (h4 : total_receipts = 5086) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * children_price = total_receipts ∧
    adult_tickets = 130 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_sales_l2438_243886


namespace NUMINAMATH_CALUDE_negation_equivalence_l2438_243889

theorem negation_equivalence : 
  (¬(∀ x : ℝ, (x = 0 ∨ x = 1) → x^2 - x = 0)) ↔ 
  (∀ x : ℝ, (x ≠ 0 ∧ x ≠ 1) → x^2 - x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2438_243889


namespace NUMINAMATH_CALUDE_oliver_seashells_l2438_243841

/-- The number of seashells Oliver collected. -/
def total_seashells : ℕ := 4

/-- The number of seashells Oliver collected on Tuesday. -/
def tuesday_seashells : ℕ := 2

/-- The number of seashells Oliver collected on Monday. -/
def monday_seashells : ℕ := total_seashells - tuesday_seashells

theorem oliver_seashells :
  monday_seashells = total_seashells - tuesday_seashells :=
by sorry

end NUMINAMATH_CALUDE_oliver_seashells_l2438_243841


namespace NUMINAMATH_CALUDE_mike_plants_cost_l2438_243848

def rose_bush_price : ℝ := 75
def tiger_tooth_aloe_price : ℝ := 100
def total_rose_bushes : ℕ := 6
def friend_rose_bushes : ℕ := 2
def mike_rose_bushes : ℕ := total_rose_bushes - friend_rose_bushes
def tiger_tooth_aloes : ℕ := 2
def rose_bush_tax_rate : ℝ := 0.05
def tiger_tooth_aloe_tax_rate : ℝ := 0.07

def mike_total_cost : ℝ :=
  (mike_rose_bushes : ℝ) * rose_bush_price * (1 + rose_bush_tax_rate) +
  (tiger_tooth_aloes : ℝ) * tiger_tooth_aloe_price * (1 + tiger_tooth_aloe_tax_rate)

theorem mike_plants_cost :
  mike_total_cost = 529 := by sorry

end NUMINAMATH_CALUDE_mike_plants_cost_l2438_243848


namespace NUMINAMATH_CALUDE_paving_rate_calculation_l2438_243868

-- Define the room dimensions and total cost
def roomLength : Real := 6.5
def roomWidth : Real := 2.75
def totalCost : Real := 10725

-- Define the theorem
theorem paving_rate_calculation :
  let area := roomLength * roomWidth
  let ratePerSqMetre := totalCost / area
  ratePerSqMetre = 600 := by
  sorry


end NUMINAMATH_CALUDE_paving_rate_calculation_l2438_243868


namespace NUMINAMATH_CALUDE_success_permutations_l2438_243850

/-- The number of letters in the word "SUCCESS" -/
def total_letters : ℕ := 7

/-- The number of times 'S' appears in "SUCCESS" -/
def s_count : ℕ := 3

/-- The number of times 'C' appears in "SUCCESS" -/
def c_count : ℕ := 2

/-- The number of times 'U' appears in "SUCCESS" -/
def u_count : ℕ := 1

/-- The number of times 'E' appears in "SUCCESS" -/
def e_count : ℕ := 1

/-- The number of unique arrangements of the letters in "SUCCESS" -/
def success_arrangements : ℕ := 420

theorem success_permutations :
  Nat.factorial total_letters / (Nat.factorial s_count * Nat.factorial c_count * Nat.factorial u_count * Nat.factorial e_count) = success_arrangements :=
by sorry

end NUMINAMATH_CALUDE_success_permutations_l2438_243850


namespace NUMINAMATH_CALUDE_find_x_l2438_243824

theorem find_x : ∃ x : ℝ, 3 * x = (26 - x) + 18 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2438_243824


namespace NUMINAMATH_CALUDE_fifth_term_value_l2438_243838

theorem fifth_term_value (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : ∀ n : ℕ, S n = 2 * n^2 + 3 * n - 1) :
  a 5 = 21 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_value_l2438_243838


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_is_11_l2438_243823

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℚ) : Prop :=
  ∃ m : ℚ, m * m = n

-- Define a function to check if a number is in its simplest quadratic radical form
def is_simplest_quadratic_radical (a : ℚ) : Prop :=
  a > 0 ∧ ¬(is_perfect_square a) ∧
  ∀ b c : ℚ, (b > 1 ∧ c > 0 ∧ a = b * c) → ¬(is_perfect_square b)

-- Theorem statement
theorem simplest_quadratic_radical_is_11 :
  is_simplest_quadratic_radical 11 ∧
  ¬(is_simplest_quadratic_radical (5/2)) ∧
  ¬(is_simplest_quadratic_radical 12) ∧
  ¬(is_simplest_quadratic_radical (1/3)) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_is_11_l2438_243823


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l2438_243862

/-- A line passing through (1,1) with equal horizontal and vertical intercepts -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (1,1) -/
  passes_through_one_one : slope + y_intercept = 1
  /-- The horizontal and vertical intercepts are equal -/
  equal_intercepts : y_intercept = slope * y_intercept

/-- The equation of a line with equal intercepts passing through (1,1) is either y = x or x + y - 2 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = 1 ∧ l.y_intercept = 0) ∨ (l.slope = -1 ∧ l.y_intercept = 2) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l2438_243862


namespace NUMINAMATH_CALUDE_train_speed_l2438_243849

/-- Given a train and a platform, calculate the speed of the train -/
theorem train_speed (train_length platform_length time : ℝ) 
  (h1 : train_length = 150)
  (h2 : platform_length = 250)
  (h3 : time = 8) :
  (train_length + platform_length) / time = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2438_243849


namespace NUMINAMATH_CALUDE_train_ride_nap_time_l2438_243897

theorem train_ride_nap_time (total_time reading_time eating_time movie_time : ℕ) 
  (h1 : total_time = 9)
  (h2 : reading_time = 2)
  (h3 : eating_time = 1)
  (h4 : movie_time = 3) :
  total_time - (reading_time + eating_time + movie_time) = 3 :=
by sorry

end NUMINAMATH_CALUDE_train_ride_nap_time_l2438_243897


namespace NUMINAMATH_CALUDE_number_added_to_23_l2438_243873

theorem number_added_to_23 : ∃! x : ℝ, 23 + x = 34 := by
  sorry

end NUMINAMATH_CALUDE_number_added_to_23_l2438_243873


namespace NUMINAMATH_CALUDE_lines_intersection_l2438_243853

/-- Represents a 2D point or vector -/
structure Point2D where
  x : ℚ
  y : ℚ

/-- Represents a parametric line in 2D -/
structure ParametricLine where
  origin : Point2D
  direction : Point2D

def line1 : ParametricLine := {
  origin := { x := 2, y := 3 },
  direction := { x := 3, y := -1 }
}

def line2 : ParametricLine := {
  origin := { x := 4, y := 1 },
  direction := { x := 1, y := 5 }
}

def intersection : Point2D := {
  x := 26 / 7,
  y := 17 / 7
}

/-- 
  Theorem: The point (26/7, 17/7) is the unique intersection point of the two given lines.
-/
theorem lines_intersection (t u : ℚ) : 
  (∃! p : Point2D, 
    p.x = line1.origin.x + t * line1.direction.x ∧ 
    p.y = line1.origin.y + t * line1.direction.y ∧
    p.x = line2.origin.x + u * line2.direction.x ∧ 
    p.y = line2.origin.y + u * line2.direction.y) ∧
  (intersection.x = line1.origin.x + t * line1.direction.x) ∧
  (intersection.y = line1.origin.y + t * line1.direction.y) ∧
  (intersection.x = line2.origin.x + u * line2.direction.x) ∧
  (intersection.y = line2.origin.y + u * line2.direction.y) :=
by sorry

end NUMINAMATH_CALUDE_lines_intersection_l2438_243853


namespace NUMINAMATH_CALUDE_total_fallen_blocks_l2438_243832

/-- Represents the heights of three stacks of blocks -/
structure BlockStacks where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of fallen blocks -/
def fallen_blocks (stacks : BlockStacks) (standing_second standing_third : ℕ) : ℕ :=
  stacks.first + (stacks.second - standing_second) + (stacks.third - standing_third)

theorem total_fallen_blocks : 
  let stacks : BlockStacks := { 
    first := 7, 
    second := 7 + 5, 
    third := 7 + 5 + 7 
  }
  fallen_blocks stacks 2 3 = 33 := by
  sorry

#eval fallen_blocks { first := 7, second := 7 + 5, third := 7 + 5 + 7 } 2 3

end NUMINAMATH_CALUDE_total_fallen_blocks_l2438_243832


namespace NUMINAMATH_CALUDE_bank_deposit_years_l2438_243839

/-- Proves that the number of years for the second bank deposit is 5 given the problem conditions. -/
theorem bank_deposit_years (principal : ℚ) (rate : ℚ) (years1 : ℚ) (interest_diff : ℚ) 
  (h1 : principal = 640)
  (h2 : rate = 15 / 100)
  (h3 : years1 = 7 / 2)
  (h4 : interest_diff = 144) :
  ∃ (years2 : ℚ), 
    principal * rate * years2 - principal * rate * years1 = interest_diff ∧ 
    years2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_bank_deposit_years_l2438_243839


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_side_equation_l2438_243887

/-- Given a triangle with sides a, b, and c satisfying a² + bc = b² + ac, prove it's isosceles --/
theorem isosceles_triangle_from_side_equation 
  (a b c : ℝ) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_equation : a^2 + b*c = b^2 + a*c) : 
  a = b :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_side_equation_l2438_243887


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_radius_from_inscribed_rectangle_l2438_243870

/-- The area of a semicircle with an inscribed 1×3 rectangle -/
theorem semicircle_area_with_inscribed_rectangle (r : ℝ) : 
  (r^2 = 5/4) → -- The radius squared equals 5/4
  (π * r^2 / 2 = 5*π/4) := -- The area of the semicircle equals 5π/4
by sorry

/-- The relationship between the radius and the inscribed rectangle -/
theorem radius_from_inscribed_rectangle (r : ℝ) :
  (r^2 = 5/4) ↔ -- The radius squared equals 5/4
  (∃ (w h : ℝ), w = 1 ∧ h = 3 ∧ w^2 + (h/2)^2 = r^2) := -- There exists a 1×3 rectangle inscribed
by sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_radius_from_inscribed_rectangle_l2438_243870


namespace NUMINAMATH_CALUDE_min_value_C_over_D_l2438_243814

theorem min_value_C_over_D (C D y : ℝ) (hC : C > 0) (hD : D > 0) (hy : y > 0)
  (hCy : y^3 + 1/y^3 = C) (hDy : y - 1/y = D) :
  C / D ≥ 6 ∧ ∃ y > 0, y^3 + 1/y^3 = C ∧ y - 1/y = D ∧ C / D = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_C_over_D_l2438_243814


namespace NUMINAMATH_CALUDE_bottle_cap_wrapper_difference_l2438_243881

-- Define the initial counts and newly found items
def initial_bottle_caps : ℕ := 12
def initial_wrappers : ℕ := 11
def found_bottle_caps : ℕ := 58
def found_wrappers : ℕ := 25

-- Define the total counts
def total_bottle_caps : ℕ := initial_bottle_caps + found_bottle_caps
def total_wrappers : ℕ := initial_wrappers + found_wrappers

-- State the theorem
theorem bottle_cap_wrapper_difference :
  total_bottle_caps - total_wrappers = 34 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_wrapper_difference_l2438_243881


namespace NUMINAMATH_CALUDE_red_cards_probability_l2438_243829

/-- The probability of drawing three red cards in succession from a deck of 60 cards,
    where 30 cards are red and 30 are black, is equal to 29/247. -/
theorem red_cards_probability (total_cards : ℕ) (red_cards : ℕ) 
  (h1 : total_cards = 60) 
  (h2 : red_cards = 30) :
  (red_cards * (red_cards - 1) * (red_cards - 2)) / 
  (total_cards * (total_cards - 1) * (total_cards - 2)) = 29 / 247 := by
  sorry

#eval (30 * 29 * 28) / (60 * 59 * 58)

end NUMINAMATH_CALUDE_red_cards_probability_l2438_243829


namespace NUMINAMATH_CALUDE_phi_values_l2438_243894

theorem phi_values (φ : Real) : 
  Real.sqrt 3 * Real.sin (20 * π / 180) = 2 * Real.cos φ - Real.sin φ → 
  φ = 140 * π / 180 ∨ φ = 40 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_phi_values_l2438_243894


namespace NUMINAMATH_CALUDE_sean_whistles_l2438_243872

/-- Given that Sean has 32 more whistles than Charles and Charles has 13 whistles,
    prove that Sean has 45 whistles. -/
theorem sean_whistles (charles_whistles : ℕ) (sean_extra_whistles : ℕ) 
  (h1 : charles_whistles = 13)
  (h2 : sean_extra_whistles = 32) :
  charles_whistles + sean_extra_whistles = 45 := by
  sorry

end NUMINAMATH_CALUDE_sean_whistles_l2438_243872


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l2438_243858

theorem perfect_square_polynomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 8*x + k = (x + a)^2) → k = 16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l2438_243858


namespace NUMINAMATH_CALUDE_x_squared_minus_y_equals_three_l2438_243876

theorem x_squared_minus_y_equals_three (x y : ℝ) :
  |x + 1| + (2 * x - y)^2 = 0 → x^2 - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_equals_three_l2438_243876


namespace NUMINAMATH_CALUDE_angle_ABH_measure_l2438_243820

/-- A regular octagon ABCDEFGH -/
structure RegularOctagon where
  -- Define the octagon (we don't need to specify all vertices, just declare it's regular)
  vertices : Fin 8 → ℝ × ℝ
  is_regular : True  -- We assume it's regular without specifying the conditions

/-- The measure of an angle in a regular octagon -/
def regular_octagon_angle : ℝ := 135

/-- Angle ABH in the regular octagon -/
def angle_ABH (octagon : RegularOctagon) : ℝ :=
  22.5

/-- Theorem: In a regular octagon ABCDEFGH, the measure of angle ABH is 22.5° -/
theorem angle_ABH_measure (octagon : RegularOctagon) :
  angle_ABH octagon = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_angle_ABH_measure_l2438_243820


namespace NUMINAMATH_CALUDE_smallest_scalene_triangle_perimeter_l2438_243818

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers form a valid triangle -/
def isTriangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧
  a < b ∧ b < c ∧
  b = a + 2 ∧ c = b + 2

theorem smallest_scalene_triangle_perimeter :
  ∀ p q r : ℕ,
    areConsecutiveOddPrimes p q r →
    isTriangle p q r →
    isPrime (p + q + r) →
    p + q + r ≥ 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_scalene_triangle_perimeter_l2438_243818


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2438_243893

theorem sum_of_solutions (x y : ℝ) (h1 : x + 6 * y = 12) (h2 : 3 * x - 2 * y = 8) : x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2438_243893


namespace NUMINAMATH_CALUDE_eric_white_marbles_l2438_243844

theorem eric_white_marbles (total : ℕ) (blue : ℕ) (green : ℕ) (white : ℕ) 
  (h1 : total = 20) 
  (h2 : blue = 6) 
  (h3 : green = 2) 
  (h4 : total = white + blue + green) : 
  white = 12 := by
  sorry

end NUMINAMATH_CALUDE_eric_white_marbles_l2438_243844


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l2438_243827

theorem number_exceeding_percentage : 
  ∃ (x : ℝ), x = 200 ∧ x = 0.25 * x + 150 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l2438_243827


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2438_243825

theorem quadratic_roots_range (k : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 + k*x₁ - k = 0 ∧ x₂^2 + k*x₂ - k = 0) →
  (1 < x₁ ∧ x₁ < 2 ∧ 2 < x₂ ∧ x₂ < 3) →
  -9/2 < k ∧ k < -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2438_243825


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l2438_243885

theorem price_reduction_percentage (last_year_price : ℝ) : 
  let this_year_price := last_year_price * (1 + 0.25)
  let next_year_target := last_year_price * (1 + 0.10)
  ∃ (reduction_percentage : ℝ), 
    this_year_price * (1 - reduction_percentage) = next_year_target ∧ 
    reduction_percentage = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l2438_243885


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_for_empty_intersection_l2438_243856

-- Define set A
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3*a) < 0}

-- Part 1: Intersection when a = 1
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x | 2 < x ∧ x < 3} := by sorry

-- Part 2: Range of a when intersection is empty
theorem range_of_a_for_empty_intersection :
  ∀ a, A ∩ B a = ∅ ↔ a ≤ 2/3 ∨ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_for_empty_intersection_l2438_243856


namespace NUMINAMATH_CALUDE_tree_structure_equation_l2438_243815

/-- Represents the structure of a tree with branches and small branches. -/
structure TreeStructure where
  branches : ℕ
  total_count : ℕ

/-- The equation for the tree structure is correct if it satisfies the given conditions. -/
def is_correct_equation (t : TreeStructure) : Prop :=
  1 + t.branches + t.branches^2 = t.total_count

/-- Theorem stating that the equation correctly represents the tree structure. -/
theorem tree_structure_equation (t : TreeStructure) 
  (h : t.total_count = 57) : is_correct_equation t := by
  sorry

end NUMINAMATH_CALUDE_tree_structure_equation_l2438_243815


namespace NUMINAMATH_CALUDE_fence_cost_calculation_l2438_243830

/-- Calculates the total cost of installing two types of fences around a rectangular field -/
theorem fence_cost_calculation (length width : ℝ) (barbed_wire_cost picket_fence_cost : ℝ) 
  (num_gates gate_width : ℝ) : 
  length = 500 ∧ 
  width = 150 ∧ 
  barbed_wire_cost = 1.2 ∧ 
  picket_fence_cost = 2.5 ∧ 
  num_gates = 4 ∧ 
  gate_width = 1.25 → 
  (2 * (length + width) - num_gates * gate_width) * barbed_wire_cost + 
  2 * (length + width) * picket_fence_cost = 4804 := by
  sorry


end NUMINAMATH_CALUDE_fence_cost_calculation_l2438_243830


namespace NUMINAMATH_CALUDE_large_square_perimeter_l2438_243874

-- Define the original square's perimeter
def original_perimeter : ℝ := 56

-- Define the number of parts the original square is divided into
def division_parts : ℕ := 4

-- Define the number of small squares used to form the large square
def small_squares : ℕ := 441

-- Theorem statement
theorem large_square_perimeter (original_perimeter : ℝ) (division_parts : ℕ) (small_squares : ℕ) :
  original_perimeter = 56 ∧ 
  division_parts = 4 ∧ 
  small_squares = 441 →
  (original_perimeter / (4 * Real.sqrt (small_squares : ℝ))) * 
  (4 * Real.sqrt (small_squares : ℝ)) = 588 := by
  sorry


end NUMINAMATH_CALUDE_large_square_perimeter_l2438_243874


namespace NUMINAMATH_CALUDE_max_sequence_length_l2438_243831

def is_valid_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (∀ i : ℕ, i + 4 < n → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4)) < 0) ∧
  (∀ i : ℕ, i + 8 < n → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) + a (i+7) + a (i+8)) > 0)

theorem max_sequence_length :
  (∃ (a : ℕ → ℝ), is_valid_sequence a 12) ∧
  (∀ (a : ℕ → ℝ) (n : ℕ), n > 12 → ¬ is_valid_sequence a n) :=
sorry

end NUMINAMATH_CALUDE_max_sequence_length_l2438_243831


namespace NUMINAMATH_CALUDE_cube_has_eight_vertices_l2438_243800

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- The number of vertices in a cube -/
def num_vertices (c : Cube) : ℕ := 8

/-- Theorem: A cube has 8 vertices -/
theorem cube_has_eight_vertices (c : Cube) : num_vertices c = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_has_eight_vertices_l2438_243800


namespace NUMINAMATH_CALUDE_old_man_gold_coins_l2438_243891

theorem old_man_gold_coins (x y : ℕ) (h : x^2 - y^2 = 81 * (x - y)) : x + y = 81 := by
  sorry

end NUMINAMATH_CALUDE_old_man_gold_coins_l2438_243891


namespace NUMINAMATH_CALUDE_jane_albert_same_committee_l2438_243880

/-- The number of MBAs --/
def n : ℕ := 6

/-- The number of members in each committee --/
def k : ℕ := 3

/-- The number of committees to be formed --/
def num_committees : ℕ := 2

/-- The total number of ways to form the committees --/
def total_ways : ℕ := Nat.choose n k

/-- The number of ways Jane and Albert can be on the same committee --/
def favorable_ways : ℕ := Nat.choose (n - 2) (k - 2)

/-- The probability that Jane and Albert are on the same committee --/
def prob_same_committee : ℚ := favorable_ways / total_ways

theorem jane_albert_same_committee :
  prob_same_committee = 1 / 5 :=
sorry

end NUMINAMATH_CALUDE_jane_albert_same_committee_l2438_243880


namespace NUMINAMATH_CALUDE_integer_root_values_l2438_243851

def polynomial (x b : ℤ) : ℤ := x^3 + 6*x^2 + b*x + 12

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

theorem integer_root_values :
  {b : ℤ | has_integer_root b} = {-217, -74, -43, -31, -22, -19, 19, 22, 31, 43, 74, 217} :=
by sorry

end NUMINAMATH_CALUDE_integer_root_values_l2438_243851


namespace NUMINAMATH_CALUDE_deanna_speed_l2438_243801

/-- Proves that given the conditions of Deanna's trip, her speed in the first 30 minutes was 90 km/h -/
theorem deanna_speed (v : ℝ) : 
  (v * (1/2) + (v + 20) * (1/2) = 100) → 
  v = 90 := by
  sorry

end NUMINAMATH_CALUDE_deanna_speed_l2438_243801


namespace NUMINAMATH_CALUDE_geometric_progression_sum_l2438_243833

theorem geometric_progression_sum (p q : ℝ) : 
  p ≠ q →                  -- Two distinct geometric progressions
  p + q = 3 →              -- Sum of common ratios is 3
  1 * p^5 + 1 * q^5 = 573 →  -- Sum of sixth terms is 573 (1 is the first term)
  1 * p^4 + 1 * q^4 = 161    -- Sum of fifth terms is 161
  := by sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_l2438_243833


namespace NUMINAMATH_CALUDE_sample_size_equals_selected_high_school_entrance_exam_sample_size_l2438_243896

/-- Represents a statistical sample --/
structure Sample where
  population : ℕ
  selected : ℕ

/-- Definition of sample size --/
def sampleSize (s : Sample) : ℕ := s.selected

/-- Theorem stating that the sample size is equal to the number of selected students --/
theorem sample_size_equals_selected (s : Sample) 
  (h₁ : s.population = 150000) 
  (h₂ : s.selected = 1000) : 
  sampleSize s = 1000 := by
  sorry

/-- Main theorem proving the sample size for the given problem --/
theorem high_school_entrance_exam_sample_size :
  ∃ s : Sample, s.population = 150000 ∧ s.selected = 1000 ∧ sampleSize s = 1000 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_equals_selected_high_school_entrance_exam_sample_size_l2438_243896


namespace NUMINAMATH_CALUDE_victoria_rice_packets_l2438_243837

def rice_packets (initial_balance : ℕ) (rice_cost : ℕ) (wheat_flour_packets : ℕ) (wheat_flour_cost : ℕ) (soda_cost : ℕ) (remaining_balance : ℕ) : ℕ :=
  (initial_balance - (wheat_flour_packets * wheat_flour_cost + soda_cost + remaining_balance)) / rice_cost

theorem victoria_rice_packets :
  rice_packets 500 20 3 25 150 235 = 2 := by
sorry

end NUMINAMATH_CALUDE_victoria_rice_packets_l2438_243837


namespace NUMINAMATH_CALUDE_frank_bakes_for_five_days_l2438_243810

-- Define the problem parameters
def cookies_per_tray : ℕ := 12
def trays_per_day : ℕ := 2
def frank_eats_per_day : ℕ := 1
def ted_eats_last_day : ℕ := 4
def cookies_left : ℕ := 134

-- Define the function to calculate the number of days
def days_baking (cookies_per_tray trays_per_day frank_eats_per_day ted_eats_last_day cookies_left : ℕ) : ℕ :=
  (cookies_left + ted_eats_last_day) / (cookies_per_tray * trays_per_day - frank_eats_per_day)

-- Theorem statement
theorem frank_bakes_for_five_days :
  days_baking cookies_per_tray trays_per_day frank_eats_per_day ted_eats_last_day cookies_left = 5 :=
sorry

end NUMINAMATH_CALUDE_frank_bakes_for_five_days_l2438_243810


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_triangle_l2438_243898

theorem semicircle_area_with_inscribed_triangle (a b r : ℝ) : 
  a > 0 → b > 0 → r > 0 →
  a^2 + b^2 = (2*r)^2 →
  a = 1 →
  b = Real.sqrt 3 →
  (π * r^2) / 2 = π / 2 := by
sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_triangle_l2438_243898


namespace NUMINAMATH_CALUDE_carla_wins_one_l2438_243866

/-- Represents a player in the chess tournament -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Carla : Player

/-- Represents the result of a game for a player -/
inductive GameResult : Type
| Win : GameResult
| Loss : GameResult

/-- The number of games each player plays against each other player -/
def gamesPerPair : Nat := 2

/-- The total number of games in the tournament -/
def totalGames : Nat := 12

/-- The number of wins for a given player -/
def wins (p : Player) : Nat :=
  match p with
  | Player.Alice => 5
  | Player.Bob => 6
  | Player.Carla => 1  -- This is what we want to prove

/-- The number of losses for a given player -/
def losses (p : Player) : Nat :=
  match p with
  | Player.Alice => 3
  | Player.Bob => 2
  | Player.Carla => 5

theorem carla_wins_one :
  (∀ p : Player, wins p + losses p = totalGames / 2) ∧
  (wins Player.Alice + wins Player.Bob + wins Player.Carla = totalGames) :=
by sorry

end NUMINAMATH_CALUDE_carla_wins_one_l2438_243866


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2438_243828

theorem algebraic_expression_value :
  let x : ℚ := 4
  let y : ℚ := -1/5
  ((x + 2*y)^2 - y*(x + 4*y) - x^2) / (-2*y) = -6 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2438_243828


namespace NUMINAMATH_CALUDE_tan_690_degrees_l2438_243884

theorem tan_690_degrees : Real.tan (690 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_690_degrees_l2438_243884


namespace NUMINAMATH_CALUDE_typing_time_l2438_243842

/-- Proves that given Tom's typing speed and page length, it takes 50 minutes to type 10 pages -/
theorem typing_time (typing_speed : ℕ) (words_per_page : ℕ) (pages : ℕ) : 
  typing_speed = 90 → words_per_page = 450 → pages = 10 → 
  (pages * words_per_page) / typing_speed = 50 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_l2438_243842


namespace NUMINAMATH_CALUDE_cosine_inequality_existence_l2438_243883

theorem cosine_inequality_existence (a b c : ℝ) :
  ∃ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) + c * Real.cos (9 * x) ≥ 1/2 * (|a| + |b| + |c|) := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_existence_l2438_243883


namespace NUMINAMATH_CALUDE_cone_base_radius_l2438_243836

/-- Given a cone whose lateral surface is a sector of a circle with a central angle of 216° 
    and a radius of 15 cm, the radius of the base of the cone is 9 cm. -/
theorem cone_base_radius (central_angle : ℝ) (sector_radius : ℝ) (base_radius : ℝ) : 
  central_angle = 216 * (π / 180) →  -- Convert 216° to radians
  sector_radius = 15 →
  base_radius = sector_radius * (central_angle / (2 * π)) →
  base_radius = 9 := by
sorry


end NUMINAMATH_CALUDE_cone_base_radius_l2438_243836


namespace NUMINAMATH_CALUDE_sqrt_floor_impossibility_l2438_243846

theorem sqrt_floor_impossibility (x : ℝ) (h1 : 100 ≤ x ∧ x ≤ 200) (h2 : ⌊Real.sqrt x⌋ = 14) : 
  ⌊Real.sqrt (50 * x)⌋ ≠ 140 := by
sorry

end NUMINAMATH_CALUDE_sqrt_floor_impossibility_l2438_243846


namespace NUMINAMATH_CALUDE_product_equals_zero_l2438_243867

theorem product_equals_zero (b : ℤ) (h : b = 3) :
  (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) * 
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l2438_243867


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2438_243845

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 9 = 0 ∧ x₂^2 + m*x₂ + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2438_243845


namespace NUMINAMATH_CALUDE_hyperbola_point_distance_to_x_axis_l2438_243892

/-- The distance from a point on a hyperbola to the x-axis, given specific conditions -/
theorem hyperbola_point_distance_to_x_axis 
  (x y : ℝ) -- Coordinates of point P
  (h1 : x^2 / 9 - y^2 / 16 = 1) -- Equation of the hyperbola
  (h2 : (y - 0) * (y - 0) = -(x + 5) * (x - 5)) -- Condition for PF₁ ⊥ PF₂
  : |y| = 16 / 5 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_point_distance_to_x_axis_l2438_243892


namespace NUMINAMATH_CALUDE_probability_green_in_specific_bag_l2438_243804

structure Bag where
  total_balls : ℕ
  green_balls : ℕ
  white_balls : ℕ

def probability_green (b : Bag) : ℚ :=
  b.green_balls / b.total_balls

theorem probability_green_in_specific_bag : 
  ∃ (b : Bag), b.total_balls = 9 ∧ b.green_balls = 7 ∧ b.white_balls = 2 ∧ 
    probability_green b = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_in_specific_bag_l2438_243804


namespace NUMINAMATH_CALUDE_equation_solution_l2438_243805

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = -3/2 ∧ x₂ = 7/2 ∧ 
  (∀ x : ℝ, 4 * (1 - x)^2 = 25 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2438_243805

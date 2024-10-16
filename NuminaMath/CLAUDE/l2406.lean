import Mathlib

namespace NUMINAMATH_CALUDE_solution_in_quadrant_I_l2406_240684

theorem solution_in_quadrant_I (k : ℝ) :
  (∃ x y : ℝ, 3 * x - 4 * y = 6 ∧ k * x + 2 * y = 8 ∧ x > 0 ∧ y > 0) ↔ -3/2 < k ∧ k < 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_I_l2406_240684


namespace NUMINAMATH_CALUDE_exists_positive_sum_G2_l2406_240638

-- Define the grid as a function from pairs of integers to real numbers
def Grid := ℤ × ℤ → ℝ

-- Define a shape as a finite set of integer pairs (representing cell positions)
def Shape := Finset (ℤ × ℤ)

-- Define the sum of numbers covered by a shape at a given position
def shapeSum (g : Grid) (s : Shape) (pos : ℤ × ℤ) : ℝ :=
  s.sum (λ (x, y) => g (x + pos.1, y + pos.2))

-- State the theorem
theorem exists_positive_sum_G2 (g : Grid) (G1 G2 : Shape) 
  (h : ∀ pos : ℤ × ℤ, shapeSum g G1 pos > 0) :
  ∃ pos : ℤ × ℤ, shapeSum g G2 pos > 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_sum_G2_l2406_240638


namespace NUMINAMATH_CALUDE_max_base_eight_digit_sum_l2406_240624

/-- Represents a positive integer in base 8 --/
def BaseEightRepresentation := List Nat

/-- Converts a natural number to its base-eight representation --/
def toBaseEight (n : Nat) : BaseEightRepresentation :=
  sorry

/-- Calculates the sum of digits in a base-eight representation --/
def digitSum (rep : BaseEightRepresentation) : Nat :=
  sorry

/-- Theorem stating the maximum digit sum for numbers less than 1729 in base 8 --/
theorem max_base_eight_digit_sum :
  (∃ (n : Nat), n < 1729 ∧ 
    digitSum (toBaseEight n) = 19 ∧ 
    ∀ (m : Nat), m < 1729 → digitSum (toBaseEight m) ≤ 19) :=
  sorry

end NUMINAMATH_CALUDE_max_base_eight_digit_sum_l2406_240624


namespace NUMINAMATH_CALUDE_largest_p_value_l2406_240649

theorem largest_p_value (m n p : ℕ) : 
  m ≥ 3 → n ≥ 3 → p ≥ 3 →
  (1 : ℚ) / m + (1 : ℚ) / n + (1 : ℚ) / p = (1 : ℚ) / 2 →
  p ≤ 42 :=
sorry

end NUMINAMATH_CALUDE_largest_p_value_l2406_240649


namespace NUMINAMATH_CALUDE_bottle_cap_count_l2406_240664

theorem bottle_cap_count : 
  ∀ (cost_per_cap total_cost num_caps : ℕ),
  cost_per_cap = 2 →
  total_cost = 12 →
  total_cost = cost_per_cap * num_caps →
  num_caps = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_count_l2406_240664


namespace NUMINAMATH_CALUDE_traffic_light_is_random_l2406_240601

-- Define the concept of a random event
def is_random_event (event : String) : Prop := sorry

-- Define the phenomena
def water_boiling : String := "Under standard atmospheric pressure, water will boil when heated to 100°C"
def traffic_light : String := "Encountering a red light when walking to a crossroads"
def rectangle_area : String := "The area of a rectangle with length and width a and b respectively is a × b"
def linear_equation : String := "A linear equation with real coefficients must have a real root"

-- Theorem to prove
theorem traffic_light_is_random : is_random_event traffic_light :=
by sorry

end NUMINAMATH_CALUDE_traffic_light_is_random_l2406_240601


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2406_240642

/-- Given sets A and B with the specified elements, if their intersection is {-3},
    then a = -1. -/
theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 2*a-1, a^2+1}
  (A ∩ B : Set ℝ) = {-3} → a = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2406_240642


namespace NUMINAMATH_CALUDE_ellipse_sum_range_l2406_240607

theorem ellipse_sum_range (x y : ℝ) (h : x^2/16 + y^2/9 = 1) :
  ∃ (z : ℝ), z = x + y ∧ -5 ≤ z ∧ z ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_sum_range_l2406_240607


namespace NUMINAMATH_CALUDE_parentheses_make_equations_true_l2406_240669

theorem parentheses_make_equations_true : 
  (5 * (4 + 3) = 35) ∧ (32 / (9 - 5) = 8) := by
  sorry

end NUMINAMATH_CALUDE_parentheses_make_equations_true_l2406_240669


namespace NUMINAMATH_CALUDE_smallest_number_with_prime_property_l2406_240666

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def remove_first_digit (n : ℕ) : ℕ := n % 1000

theorem smallest_number_with_prime_property : 
  ∃! n : ℕ, 
    (∀ m : ℕ, m < n → 
      ¬(∃ p q : ℕ, 
        is_prime p ∧ 
        is_prime q ∧ 
        remove_first_digit m = 4 * p ∧ 
        remove_first_digit m + 1 = 5 * q)) ∧
    (∃ p q : ℕ, 
      is_prime p ∧ 
      is_prime q ∧ 
      remove_first_digit n = 4 * p ∧ 
      remove_first_digit n + 1 = 5 * q) ∧
    n = 1964 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_prime_property_l2406_240666


namespace NUMINAMATH_CALUDE_stairs_fibonacci_equivalence_nine_steps_ways_l2406_240695

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

def climbStairs : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => climbStairs n + climbStairs (n + 1)

theorem stairs_fibonacci_equivalence (n : ℕ) : climbStairs n = fibonacci (n + 1) := by
  sorry

theorem nine_steps_ways : climbStairs 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_stairs_fibonacci_equivalence_nine_steps_ways_l2406_240695


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2406_240633

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (5*x - 4)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2406_240633


namespace NUMINAMATH_CALUDE_picture_placement_l2406_240660

theorem picture_placement (wall_width : ℝ) (picture_width : ℝ) (space_between : ℝ)
  (h1 : wall_width = 25)
  (h2 : picture_width = 2)
  (h3 : space_between = 1)
  (h4 : 2 * picture_width + space_between < wall_width) :
  let distance := (wall_width - (2 * picture_width + space_between)) / 2
  distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_picture_placement_l2406_240660


namespace NUMINAMATH_CALUDE_student_group_assignment_l2406_240631

/-- The number of ways to assign students to groups -/
def assignment_count (num_students : ℕ) (num_groups : ℕ) : ℕ :=
  num_groups ^ num_students

/-- Theorem: The number of ways to assign 4 students to 3 groups is 3^4 -/
theorem student_group_assignment :
  assignment_count 4 3 = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_student_group_assignment_l2406_240631


namespace NUMINAMATH_CALUDE_triangle_inequality_l2406_240616

theorem triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : a + b + c = 2) :
  a^2 + b^2 + c^2 < 2*(1 - a*b*c) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2406_240616


namespace NUMINAMATH_CALUDE_find_divisor_l2406_240615

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 217 →
  quotient = 54 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 4 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2406_240615


namespace NUMINAMATH_CALUDE_sin_cos_inequality_l2406_240690

theorem sin_cos_inequality (x : ℝ) : (Real.sin x + 2 * Real.cos (2 * x)) * (2 * Real.sin (2 * x) - Real.cos x) < 4.5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_inequality_l2406_240690


namespace NUMINAMATH_CALUDE_mikes_net_salary_calculation_l2406_240654

-- Define the initial conditions
def freds_initial_salary : ℝ := 1000
def freds_bonus : ℝ := 500
def freds_investment_return : ℝ := 0.20
def mikes_salary_multiplier : ℝ := 10
def mikes_bonus_percentage : ℝ := 0.10
def mikes_investment_return : ℝ := 0.25
def mikes_salary_increase : ℝ := 0.40
def mikes_tax_rate : ℝ := 0.15

-- Define the theorem
theorem mikes_net_salary_calculation :
  let mikes_initial_salary := freds_initial_salary * mikes_salary_multiplier
  let mikes_initial_total := mikes_initial_salary * (1 + mikes_bonus_percentage)
  let mikes_investment_result := mikes_initial_total * (1 + mikes_investment_return)
  let mikes_new_salary := mikes_initial_salary * (1 + mikes_salary_increase)
  let mikes_tax := mikes_new_salary * mikes_tax_rate
  mikes_new_salary - mikes_tax = 11900 :=
by sorry

end NUMINAMATH_CALUDE_mikes_net_salary_calculation_l2406_240654


namespace NUMINAMATH_CALUDE_fair_distribution_correctness_l2406_240686

/-- Represents the amount of bread each person has initially -/
structure BreadDistribution where
  personA : ℚ
  personB : ℚ

/-- Represents the fair distribution of currency -/
structure CurrencyDistribution where
  personA : ℚ
  personB : ℚ

/-- Calculates the fair distribution of currency based on initial bread distribution -/
def calculateFairDistribution (initial : BreadDistribution) (totalCurrency : ℚ) : CurrencyDistribution :=
  sorry

theorem fair_distribution_correctness 
  (initial : BreadDistribution)
  (h1 : initial.personA = 3)
  (h2 : initial.personB = 2)
  (totalCurrency : ℚ)
  (h3 : totalCurrency = 50) :
  let result := calculateFairDistribution initial totalCurrency
  result.personA = 40 ∧ result.personB = 10 := by
  sorry

end NUMINAMATH_CALUDE_fair_distribution_correctness_l2406_240686


namespace NUMINAMATH_CALUDE_cecilia_always_wins_l2406_240671

theorem cecilia_always_wins (a : ℕ+) : ∃ b : ℕ+, 
  (Nat.gcd a b = 1) ∧ 
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ 
    (p * q * r ∣ a^3 + b^3)) := by
  sorry

end NUMINAMATH_CALUDE_cecilia_always_wins_l2406_240671


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2406_240677

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2406_240677


namespace NUMINAMATH_CALUDE_markers_problem_l2406_240668

theorem markers_problem (initial_markers : ℕ) (markers_per_box : ℕ) (total_markers : ℕ) :
  initial_markers = 32 →
  markers_per_box = 9 →
  total_markers = 86 →
  (total_markers - initial_markers) / markers_per_box = 6 :=
by sorry

end NUMINAMATH_CALUDE_markers_problem_l2406_240668


namespace NUMINAMATH_CALUDE_multiplication_mistake_l2406_240675

theorem multiplication_mistake (x : ℤ) : 
  (43 * x - 34 * x = 1206) → x = 134 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_l2406_240675


namespace NUMINAMATH_CALUDE_tv_purchase_time_l2406_240617

/-- Calculates the number of months required to save for a television purchase. -/
def months_to_purchase_tv (monthly_income : ℕ) (food_expense : ℕ) (utilities_expense : ℕ) 
  (other_expenses : ℕ) (current_savings : ℕ) (tv_cost : ℕ) : ℕ :=
  let total_expenses := food_expense + utilities_expense + other_expenses
  let monthly_savings := monthly_income - total_expenses
  let additional_savings_needed := tv_cost - current_savings
  (additional_savings_needed + monthly_savings - 1) / monthly_savings

theorem tv_purchase_time :
  months_to_purchase_tv 30000 15000 5000 2500 10000 25000 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tv_purchase_time_l2406_240617


namespace NUMINAMATH_CALUDE_accounting_majors_l2406_240697

theorem accounting_majors (p q r s : ℕ+) 
  (h1 : p * q * r * s = 1365)
  (h2 : 1 < p)
  (h3 : p < q)
  (h4 : q < r)
  (h5 : r < s) :
  p = 3 := by sorry

end NUMINAMATH_CALUDE_accounting_majors_l2406_240697


namespace NUMINAMATH_CALUDE_common_tangents_of_specific_circles_l2406_240608

/-- The number of common tangents to two intersecting circles -/
def num_common_tangents (c1_center : ℝ × ℝ) (c1_radius : ℝ) (c2_center : ℝ × ℝ) (c2_radius : ℝ) : ℕ :=
  sorry

/-- The theorem stating that the number of common tangents to the given circles is 2 -/
theorem common_tangents_of_specific_circles : 
  num_common_tangents (2, 1) 2 (-1, 2) 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_tangents_of_specific_circles_l2406_240608


namespace NUMINAMATH_CALUDE_birds_count_l2406_240635

/-- The number of birds on the fence -/
def birds : ℕ := sorry

/-- Ten more than twice the number of birds on the fence is 50 -/
axiom birds_condition : 10 + 2 * birds = 50

/-- Prove that the number of birds on the fence is 20 -/
theorem birds_count : birds = 20 := by sorry

end NUMINAMATH_CALUDE_birds_count_l2406_240635


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l2406_240643

-- Define the polynomials
def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def j (x : ℝ) : ℝ := x^2 - x - 3

-- State the theorem
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x + j x = -3 * x^2 + 11 * x - 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l2406_240643


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l2406_240674

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem fifth_term_of_arithmetic_sequence 
  (a d : ℤ) 
  (h1 : arithmetic_sequence a d 10 = 15) 
  (h2 : arithmetic_sequence a d 11 = 18) : 
  arithmetic_sequence a d 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l2406_240674


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2406_240600

theorem rationalize_denominator : 
  (7 : ℝ) / (Real.sqrt 175 - Real.sqrt 75) = 7 * (Real.sqrt 7 + Real.sqrt 3) / 20 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2406_240600


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l2406_240647

/-- A coloring function for the Cartesian plane with integer coordinates. -/
def ColoringFunction := ℤ → ℤ → Fin 3

/-- Proposition that a color appears infinitely many times on infinitely many horizontal lines. -/
def InfiniteAppearance (f : ColoringFunction) (c : Fin 3) : Prop :=
  ∀ N : ℕ, ∃ y > N, ∀ M : ℕ, ∃ x > M, f x y = c

/-- Proposition that three points of different colors are not collinear. -/
def NotCollinear (f : ColoringFunction) : Prop :=
  ∀ x₁ y₁ x₂ y₂ x₃ y₃ : ℤ,
    f x₁ y₁ ≠ f x₂ y₂ ∧ f x₂ y₂ ≠ f x₃ y₃ ∧ f x₃ y₃ ≠ f x₁ y₁ →
    (x₁ - x₂) * (y₃ - y₂) ≠ (x₃ - x₂) * (y₁ - y₂)

/-- Theorem stating the existence of a valid coloring function. -/
theorem exists_valid_coloring : ∃ f : ColoringFunction,
  (∀ c : Fin 3, InfiniteAppearance f c) ∧ NotCollinear f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l2406_240647


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2406_240632

theorem quadratic_no_real_roots (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + 2 * x - 2 ≠ 0) ↔ m < (1 : ℝ) / 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2406_240632


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2406_240604

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| < 2}
def B : Set ℝ := {x : ℝ | x^2 - 3*x < 0}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2406_240604


namespace NUMINAMATH_CALUDE_inequality_proof_l2406_240627

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2406_240627


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2406_240689

theorem floor_equation_solution (n : ℤ) : 
  (⌊n^2 / 4⌋ - ⌊n / 2⌋^2 = 3) ↔ (n = 7) :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2406_240689


namespace NUMINAMATH_CALUDE_exponent_simplification_l2406_240641

theorem exponent_simplification :
  3000 * (3000 ^ 2500) * 2 = 2 * 3000 ^ 2501 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l2406_240641


namespace NUMINAMATH_CALUDE_complement_of_A_l2406_240698

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x ≥ 1} ∪ {x : ℝ | x ≤ 0}

theorem complement_of_A : Set.compl A = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2406_240698


namespace NUMINAMATH_CALUDE_range_of_expression_l2406_240639

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (z : ℝ), z = 4*(x - 1/2)^2 + (y - 1)^2 + 4*x*y ∧ 1 ≤ z ∧ z ≤ 22 + 4*Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l2406_240639


namespace NUMINAMATH_CALUDE_bowling_team_weight_specific_bowling_problem_l2406_240662

/-- Given a bowling team with initial players and weights, prove the weight of a new player --/
theorem bowling_team_weight (initial_players : ℕ) (initial_avg_weight : ℝ) 
  (new_player1_weight : ℝ) (new_avg_weight : ℝ) : ℝ :=
  let total_initial_weight := initial_players * initial_avg_weight
  let new_total_players := initial_players + 2
  let new_total_weight := new_total_players * new_avg_weight
  let new_players_total_weight := new_total_weight - total_initial_weight
  let new_player2_weight := new_players_total_weight - new_player1_weight
  new_player2_weight

/-- The specific bowling team problem --/
theorem specific_bowling_problem : 
  bowling_team_weight 7 76 110 78 = 60 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_weight_specific_bowling_problem_l2406_240662


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2406_240691

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a > 1 ∧ b > 1) → (a + b > 2 ∧ a * b > 1) ∧
  ¬((a + b > 2 ∧ a * b > 1) → (a > 1 ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2406_240691


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_5_l2406_240694

theorem smallest_common_multiple_of_6_and_5 : ∃ n : ℕ, 
  n > 0 ∧ 
  6 ∣ n ∧ 
  5 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → 6 ∣ m → 5 ∣ m → n ≤ m :=
by
  use 30
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_5_l2406_240694


namespace NUMINAMATH_CALUDE_no_77_cents_combination_l2406_240657

/-- Represents the set of available coin values in cents -/
def CoinValues : Set ℕ := {1, 5, 10, 50}

/-- Represents a selection of exactly three coins -/
def CoinSelection := Fin 3 → ℕ

/-- The sum of a coin selection -/
def sum_coins (selection : CoinSelection) : ℕ :=
  (selection 0) + (selection 1) + (selection 2)

/-- Predicate to check if a selection is valid (all coins are from CoinValues) -/
def valid_selection (selection : CoinSelection) : Prop :=
  ∀ i, selection i ∈ CoinValues

theorem no_77_cents_combination :
  ¬∃ (selection : CoinSelection), valid_selection selection ∧ sum_coins selection = 77 := by
  sorry

#check no_77_cents_combination

end NUMINAMATH_CALUDE_no_77_cents_combination_l2406_240657


namespace NUMINAMATH_CALUDE_probability_two_even_balls_l2406_240603

def total_balls : ℕ := 17
def even_balls : ℕ := 8

theorem probability_two_even_balls :
  (even_balls : ℚ) / total_balls * (even_balls - 1) / (total_balls - 1) = 7 / 34 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_even_balls_l2406_240603


namespace NUMINAMATH_CALUDE_april_roses_unsold_l2406_240678

/-- The number of roses left unsold after a sale --/
def roses_left_unsold (initial_roses : ℕ) (price_per_rose : ℕ) (total_earned : ℕ) : ℕ :=
  initial_roses - (total_earned / price_per_rose)

/-- Theorem: Given the conditions of April's rose sale, prove that 4 roses were left unsold --/
theorem april_roses_unsold : roses_left_unsold 13 4 36 = 4 := by
  sorry

end NUMINAMATH_CALUDE_april_roses_unsold_l2406_240678


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2406_240679

theorem arithmetic_sequence_problem (a : ℕ → ℕ) (n : ℕ) :
  (∀ k, a (k + 1) = a k + 5) →  -- arithmetic sequence with common difference 5
  a 1 = 1 →                    -- first term is 1
  a n = 2016 →                 -- n-th term is 2016
  n = 404 :=                   -- prove n is 404
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2406_240679


namespace NUMINAMATH_CALUDE_derivative_x_cos_x_l2406_240619

theorem derivative_x_cos_x (x : ℝ) :
  deriv (fun x => x * Real.cos x) x = Real.cos x - x * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_cos_x_l2406_240619


namespace NUMINAMATH_CALUDE_most_accurate_approximation_l2406_240665

def reading_lower_bound : ℝ := 10.65
def reading_upper_bound : ℝ := 10.85
def major_tick_interval : ℝ := 0.1

def options : List ℝ := [10.68, 10.72, 10.74, 10.75]

theorem most_accurate_approximation :
  ∃ (x : ℝ), 
    reading_lower_bound ≤ x ∧ 
    x ≤ reading_upper_bound ∧ 
    (∀ y ∈ options, |x - 10.75| ≤ |x - y|) :=
by sorry

end NUMINAMATH_CALUDE_most_accurate_approximation_l2406_240665


namespace NUMINAMATH_CALUDE_smallest_square_l2406_240614

theorem smallest_square (a b : ℕ+) 
  (h1 : ∃ r : ℕ, (15 : ℤ) * a + (16 : ℤ) * b = r^2)
  (h2 : ∃ s : ℕ, (16 : ℤ) * a - (15 : ℤ) * b = s^2) :
  (481 : ℕ)^2 ≤ min ((15 : ℤ) * a + (16 : ℤ) * b) ((16 : ℤ) * a - (15 : ℤ) * b) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_l2406_240614


namespace NUMINAMATH_CALUDE_line_points_Q_value_l2406_240658

/-- Given a line x = 8y + 5 passing through points (m, n) and (m + Q, n + p), where p = 0.25,
    prove that Q = 2. -/
theorem line_points_Q_value (m n Q p : ℝ) : 
  p = 0.25 →
  m = 8 * n + 5 →
  m + Q = 8 * (n + p) + 5 →
  Q = 2 := by
sorry

end NUMINAMATH_CALUDE_line_points_Q_value_l2406_240658


namespace NUMINAMATH_CALUDE_pick_shoes_five_pairs_l2406_240656

/-- The number of ways to pick 4 shoes from 5 pairs such that exactly one pair is among them -/
def pick_shoes (num_pairs : ℕ) : ℕ := 
  num_pairs * (Nat.choose (num_pairs - 1) 2) * 2 * 2

/-- Theorem stating that picking 4 shoes from 5 pairs with exactly one pair among them can be done in 120 ways -/
theorem pick_shoes_five_pairs : pick_shoes 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_pick_shoes_five_pairs_l2406_240656


namespace NUMINAMATH_CALUDE_decimal_addition_l2406_240613

theorem decimal_addition : 5.763 + 2.489 = 8.152 := by sorry

end NUMINAMATH_CALUDE_decimal_addition_l2406_240613


namespace NUMINAMATH_CALUDE_bell_pepper_slices_l2406_240693

theorem bell_pepper_slices (num_peppers : ℕ) (slices_per_pepper : ℕ) (smaller_pieces : ℕ) : 
  num_peppers = 5 →
  slices_per_pepper = 20 →
  smaller_pieces = 3 →
  let total_slices := num_peppers * slices_per_pepper
  let large_slices := total_slices / 2
  let small_pieces := large_slices * smaller_pieces
  total_slices - large_slices + small_pieces = 200 := by
  sorry

end NUMINAMATH_CALUDE_bell_pepper_slices_l2406_240693


namespace NUMINAMATH_CALUDE_halfway_fraction_l2406_240652

theorem halfway_fraction : ∃ (n d : ℕ), d ≠ 0 ∧ (n : ℚ) / d = (1 : ℚ) / 3 / 2 + (3 : ℚ) / 4 / 2 ∧ n = 13 ∧ d = 24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l2406_240652


namespace NUMINAMATH_CALUDE_simplified_expression_ratio_l2406_240688

theorem simplified_expression_ratio (m : ℝ) :
  let original := (6 * m + 18) / 6
  ∃ (c d : ℤ), (∃ (x : ℝ), original = c * x + d) ∧ (c : ℚ) / d = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_ratio_l2406_240688


namespace NUMINAMATH_CALUDE_he_has_21_apples_l2406_240636

/-- The number of apples Adam and Jackie have together -/
def total_adam_jackie : ℕ := 12

/-- The number of additional apples He has compared to Adam and Jackie together -/
def additional_apples : ℕ := 9

/-- The number of additional apples Adam has compared to Jackie -/
def adam_more_than_jackie : ℕ := 8

/-- The number of apples He has -/
def he_apples : ℕ := total_adam_jackie + additional_apples

theorem he_has_21_apples : he_apples = 21 := by
  sorry

end NUMINAMATH_CALUDE_he_has_21_apples_l2406_240636


namespace NUMINAMATH_CALUDE_books_before_adding_l2406_240676

theorem books_before_adding (total_after : ℕ) (added : ℕ) (h1 : total_after = 19) (h2 : added = 10) :
  total_after - added = 9 := by
  sorry

end NUMINAMATH_CALUDE_books_before_adding_l2406_240676


namespace NUMINAMATH_CALUDE_orthogonal_circles_product_l2406_240625

theorem orthogonal_circles_product (x y u v : ℝ) 
  (h1 : x^2 + y^2 = 1)
  (h2 : u^2 + v^2 = 1)
  (h3 : x*u + y*v = 0) :
  x*y + u*v = 0 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_circles_product_l2406_240625


namespace NUMINAMATH_CALUDE_translation_of_line_segment_l2406_240621

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def translatePoint (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_of_line_segment (A B A' : Point) :
  A.x = -4 ∧ A.y = -1 ∧
  B.x = 1 ∧ B.y = 1 ∧
  A'.x = -2 ∧ A'.y = 2 →
  ∃ (t : Translation), translatePoint A t = A' ∧ translatePoint B t = { x := 3, y := 4 } := by
  sorry

end NUMINAMATH_CALUDE_translation_of_line_segment_l2406_240621


namespace NUMINAMATH_CALUDE_remainder_of_55_power_55_plus_55_mod_56_l2406_240672

theorem remainder_of_55_power_55_plus_55_mod_56 :
  (55^55 + 55) % 56 = 54 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_55_power_55_plus_55_mod_56_l2406_240672


namespace NUMINAMATH_CALUDE_fraction_sum_minus_eight_l2406_240696

theorem fraction_sum_minus_eight : 
  (4/3 : ℚ) + (7/5 : ℚ) + (12/10 : ℚ) + (23/20 : ℚ) + (45/40 : ℚ) + (89/80 : ℚ) - 8 = -163/240 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_minus_eight_l2406_240696


namespace NUMINAMATH_CALUDE_daughters_to_sons_ratio_l2406_240681

theorem daughters_to_sons_ratio (total_children : ℕ) (sons : ℕ) (daughters : ℕ) : 
  total_children = 21 → sons = 3 → daughters = total_children - sons → 
  (daughters : ℚ) / (sons : ℚ) = 6 / 1 := by
  sorry

end NUMINAMATH_CALUDE_daughters_to_sons_ratio_l2406_240681


namespace NUMINAMATH_CALUDE_pages_read_per_year_l2406_240699

/-- The number of pages read in a year given monthly reading habits and book lengths -/
theorem pages_read_per_year
  (novels_per_month : ℕ)
  (pages_per_novel : ℕ)
  (months_per_year : ℕ)
  (h1 : novels_per_month = 4)
  (h2 : pages_per_novel = 200)
  (h3 : months_per_year = 12) :
  novels_per_month * pages_per_novel * months_per_year = 9600 :=
by sorry

end NUMINAMATH_CALUDE_pages_read_per_year_l2406_240699


namespace NUMINAMATH_CALUDE_equation_solution_l2406_240622

theorem equation_solution : ∃! x : ℚ, x + 5/8 = 1/4 - 2/5 + 7/10 ∧ x = -3/40 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2406_240622


namespace NUMINAMATH_CALUDE_point_outside_circle_l2406_240661

theorem point_outside_circle (m : ℝ) : 
  let P : ℝ × ℝ := (m^2, 5)
  let circle_equation (x y : ℝ) := x^2 + y^2 = 24
  ∀ x y, circle_equation x y → (P.1 - x)^2 + (P.2 - y)^2 > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l2406_240661


namespace NUMINAMATH_CALUDE_a_2007_mod_100_l2406_240646

/-- Sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 7
  | n + 1 => 7^(a n)

/-- Theorem stating that a_2007 ≡ 43 (mod 100) -/
theorem a_2007_mod_100 : a 2006 % 100 = 43 := by
  sorry

end NUMINAMATH_CALUDE_a_2007_mod_100_l2406_240646


namespace NUMINAMATH_CALUDE_y_equals_x_cubed_l2406_240626

/-- Represents a pair of x and y values from the table -/
structure XYPair where
  x : ℕ
  y : ℕ

/-- The set of (x, y) pairs from the given table -/
def xyTable : List XYPair := [
  ⟨1, 1⟩,
  ⟨2, 8⟩,
  ⟨3, 27⟩,
  ⟨4, 64⟩,
  ⟨5, 125⟩
]

/-- Theorem stating that y = x^3 holds for all pairs in the table -/
theorem y_equals_x_cubed (pair : XYPair) (h : pair ∈ xyTable) : pair.y = pair.x ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_y_equals_x_cubed_l2406_240626


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l2406_240653

theorem opposite_of_negative_three : -((-3) : ℤ) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l2406_240653


namespace NUMINAMATH_CALUDE_unique_solution_for_xy_equation_l2406_240651

theorem unique_solution_for_xy_equation :
  ∀ x y : ℤ, x > y ∧ y > 0 ∧ x + y + x * y = 99 → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_xy_equation_l2406_240651


namespace NUMINAMATH_CALUDE_three_minus_a_equals_four_l2406_240602

theorem three_minus_a_equals_four (a b : ℝ) 
  (eq1 : 3 + a = 4 - b) 
  (eq2 : 4 + b = 7 + a) : 
  3 - a = 4 := by
sorry

end NUMINAMATH_CALUDE_three_minus_a_equals_four_l2406_240602


namespace NUMINAMATH_CALUDE_tangent_line_minimum_value_l2406_240618

theorem tangent_line_minimum_value (k b : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧
    k = 1 / (2 * Real.sqrt x₀) ∧
    b = Real.sqrt x₀ / 2 + 1 ∧
    k * x₀ + b = Real.sqrt x₀ + 1) →
  k^2 + b^2 - 2*b ≥ -1/2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_value_l2406_240618


namespace NUMINAMATH_CALUDE_congruence_problem_l2406_240670

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 25 ∧ -150 ≡ n [ZMOD 25] → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2406_240670


namespace NUMINAMATH_CALUDE_min_value_of_f_l2406_240663

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 - Real.sqrt 3 * abs x + 1) + Real.sqrt (x^2 + Real.sqrt 3 * abs x + 3)

theorem min_value_of_f :
  (∀ x : ℝ, f x ≥ Real.sqrt 7) ∧
  f (Real.sqrt 3 / 4) = Real.sqrt 7 ∧
  f (-Real.sqrt 3 / 4) = Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2406_240663


namespace NUMINAMATH_CALUDE_second_car_speed_l2406_240683

/-- Proves that given the conditions of the problem, the speed of the second car is 100 km/hr -/
theorem second_car_speed (car_a_speed : ℝ) (car_a_time : ℝ) (second_car_time : ℝ) (distance_ratio : ℝ) :
  car_a_speed = 50 →
  car_a_time = 6 →
  second_car_time = 1 →
  distance_ratio = 3 →
  (car_a_speed * car_a_time) / (distance_ratio * second_car_time) = 100 := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l2406_240683


namespace NUMINAMATH_CALUDE_touching_spheres_radii_l2406_240644

/-- Given four spheres of radius r, where each sphere touches the other three,
    calculate the radii of spheres that touch all four spheres internally and externally. -/
theorem touching_spheres_radii (r : ℝ) (r_pos : r > 0) :
  ∃ (p R : ℝ),
    (p = r * (Real.sqrt 6 / 2 - 1)) ∧
    (R = r * (Real.sqrt 6 / 2 + 1)) ∧
    (p > 0) ∧ (R > 0) :=
by sorry

end NUMINAMATH_CALUDE_touching_spheres_radii_l2406_240644


namespace NUMINAMATH_CALUDE_square_circle_perimeter_l2406_240667

/-- Given a square with perimeter 28 cm and a circle with radius equal to the side of the square,
    the perimeter of the circle is 14π cm. -/
theorem square_circle_perimeter : 
  ∀ (square_side circle_radius : ℝ),
    square_side * 4 = 28 →
    circle_radius = square_side →
    2 * Real.pi * circle_radius = 14 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_square_circle_perimeter_l2406_240667


namespace NUMINAMATH_CALUDE_missing_number_proof_l2406_240628

theorem missing_number_proof (x : ℝ) : x * 240 = 173 * 240 → x = 173 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2406_240628


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2406_240682

theorem quadratic_inequality_condition (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * x + 1 < 0) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2406_240682


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2406_240623

/-- The positive slope of an asymptote of the hyperbola defined by 
    √((x-2)² + (y-3)²) - √((x-8)² + (y-3)²) = 4 -/
theorem hyperbola_asymptote_slope : ∃ (m : ℝ), m > 0 ∧ 
  (∀ (x y : ℝ), Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4 →
    m = Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2406_240623


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2406_240680

theorem cubic_equation_roots : ∃! (p n₁ n₂ : ℝ), 
  p > 0 ∧ n₁ < 0 ∧ n₂ < 0 ∧
  p^3 + 3*p^2 - 4*p + 12 = 0 ∧
  n₁^3 + 3*n₁^2 - 4*n₁ + 12 = 0 ∧
  n₂^3 + 3*n₂^2 - 4*n₂ + 12 = 0 ∧
  p ≠ n₁ ∧ p ≠ n₂ ∧ n₁ ≠ n₂ :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2406_240680


namespace NUMINAMATH_CALUDE_perpendicular_bisector_value_l2406_240620

/-- The perpendicular bisector of a line segment from (x₁, y₁) to (x₂, y₂) is defined as
    the line that passes through the midpoint of the segment and is perpendicular to it. --/
def is_perpendicular_bisector (a b c : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  -- The line ax + by + c = 0 passes through the midpoint
  a * midpoint_x + b * midpoint_y + c = 0 ∧
  -- The line is perpendicular to the segment
  a * (x₂ - x₁) + b * (y₂ - y₁) = 0

/-- Given that the line x + y = b is the perpendicular bisector of the line segment 
    from (0, 5) to (8, 10), prove that b = 11.5 --/
theorem perpendicular_bisector_value : 
  is_perpendicular_bisector 1 1 (-b) 0 5 8 10 → b = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_value_l2406_240620


namespace NUMINAMATH_CALUDE_combined_fuel_efficiency_l2406_240634

theorem combined_fuel_efficiency
  (m : ℝ) -- distance driven by each car
  (h_pos : m > 0) -- ensure distance is positive
  (efficiency_linda : ℝ := 30) -- Linda's car efficiency
  (efficiency_joe : ℝ := 15) -- Joe's car efficiency
  (efficiency_anne : ℝ := 20) -- Anne's car efficiency
  : (3 * m) / (m / efficiency_linda + m / efficiency_joe + m / efficiency_anne) = 20 :=
by sorry

end NUMINAMATH_CALUDE_combined_fuel_efficiency_l2406_240634


namespace NUMINAMATH_CALUDE_first_solution_concentration_l2406_240650

-- Define the variables
def total_volume : ℝ := 630
def final_concentration : ℝ := 50
def first_solution_volume : ℝ := 420
def second_solution_concentration : ℝ := 30

-- Define the theorem
theorem first_solution_concentration :
  ∃ (x : ℝ),
    x * first_solution_volume / 100 +
    second_solution_concentration * (total_volume - first_solution_volume) / 100 =
    final_concentration * total_volume / 100 ∧
    x = 60 := by
  sorry

end NUMINAMATH_CALUDE_first_solution_concentration_l2406_240650


namespace NUMINAMATH_CALUDE_cube_coverage_tape_pieces_correct_l2406_240687

/-- Represents the number of tape pieces needed to cover a cube --/
def tape_pieces (n : ℕ) : ℕ := 2 * n

/-- Theorem stating that the number of tape pieces needed to cover a cube with edge length n is 2n --/
theorem cube_coverage (n : ℕ) :
  tape_pieces n = 2 * n :=
by sorry

/-- Represents the properties of the tape coverage method --/
structure TapeCoverage where
  edge_length : ℕ
  tape_width : ℕ
  parallel_to_edge : Bool
  can_cross_edges : Bool
  no_overhang : Bool

/-- Theorem stating that the tape_pieces function gives the correct number of pieces
    for a cube coverage satisfying the given constraints --/
theorem tape_pieces_correct (coverage : TapeCoverage) 
  (h1 : coverage.tape_width = 1)
  (h2 : coverage.parallel_to_edge = true)
  (h3 : coverage.can_cross_edges = true)
  (h4 : coverage.no_overhang = true) :
  tape_pieces coverage.edge_length = 2 * coverage.edge_length :=
by sorry

end NUMINAMATH_CALUDE_cube_coverage_tape_pieces_correct_l2406_240687


namespace NUMINAMATH_CALUDE_coefficient_x_plus_one_squared_in_x_to_tenth_l2406_240606

theorem coefficient_x_plus_one_squared_in_x_to_tenth : ∃ (a₀ a₁ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ),
  ∀ x : ℝ, x^10 = a₀ + a₁*(x+1) + 45*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
            a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_plus_one_squared_in_x_to_tenth_l2406_240606


namespace NUMINAMATH_CALUDE_mildred_weight_l2406_240645

/-- Given that Carol weighs 9 pounds and Mildred is 50 pounds heavier than Carol,
    prove that Mildred weighs 59 pounds. -/
theorem mildred_weight (carol_weight : ℕ) (weight_difference : ℕ) 
  (h1 : carol_weight = 9)
  (h2 : weight_difference = 50) :
  carol_weight + weight_difference = 59 := by
  sorry

end NUMINAMATH_CALUDE_mildred_weight_l2406_240645


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l2406_240611

theorem sin_cos_fourth_power_sum (α : ℝ) (h : Real.sin α - Real.cos α = 1/2) :
  Real.sin α ^ 4 + Real.cos α ^ 4 = 23/32 := by sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l2406_240611


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l2406_240685

theorem lcm_factor_problem (A B : ℕ) (hcf lcm x : ℕ) : 
  A > 0 → B > 0 → 
  Nat.gcd A B = hcf →
  hcf = 20 →
  A = 280 →
  lcm = Nat.lcm A B →
  lcm = 20 * 13 * x →
  x = 14 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l2406_240685


namespace NUMINAMATH_CALUDE_sunglasses_and_caps_probability_l2406_240673

theorem sunglasses_and_caps_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (prob_sunglasses_also_cap : ℚ) :
  total_sunglasses = 80 →
  total_caps = 45 →
  prob_sunglasses_also_cap = 3/8 →
  (total_sunglasses * prob_sunglasses_also_cap : ℚ) / total_caps = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_sunglasses_and_caps_probability_l2406_240673


namespace NUMINAMATH_CALUDE_regular_triangle_rotation_l2406_240692

/-- The minimum angle of rotation (in degrees) for a regular triangle to coincide with itself. -/
def min_rotation_angle_regular_triangle : ℝ := 120

/-- Theorem stating that the minimum angle of rotation for a regular triangle to coincide with itself is 120 degrees. -/
theorem regular_triangle_rotation :
  min_rotation_angle_regular_triangle = 120 := by sorry

end NUMINAMATH_CALUDE_regular_triangle_rotation_l2406_240692


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_l2406_240605

theorem polynomial_equality_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = 
    a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + 
    a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + 
    a₉*(x+2)^9 + a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_l2406_240605


namespace NUMINAMATH_CALUDE_correct_quotient_calculation_l2406_240655

theorem correct_quotient_calculation (A B : ℕ) (dividend : ℕ) : 
  A > 0 → 
  A * 100 + B * 10 > 0 →
  dividend / (A * 10 + B) = 210 → 
  dividend / (A * 100 + B * 10) = 21 := by
sorry

end NUMINAMATH_CALUDE_correct_quotient_calculation_l2406_240655


namespace NUMINAMATH_CALUDE_no_natural_number_divisible_by_100_l2406_240610

theorem no_natural_number_divisible_by_100 : ∀ n : ℕ, ¬(100 ∣ (n^2 + 6*n + 2019)) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_number_divisible_by_100_l2406_240610


namespace NUMINAMATH_CALUDE_flash_overtakes_ace_l2406_240629

/-- The distance Flash needs to jog to overtake Ace -/
def overtake_distance (v y t : ℝ) : ℝ :=
  2 * (y + 60 * v * t)

/-- Theorem stating the distance Flash needs to jog to overtake Ace -/
theorem flash_overtakes_ace (v y t : ℝ) (hv : v > 0) (hy : y ≥ 0) (ht : t ≥ 0) :
  ∃ d : ℝ, d = overtake_distance v y t ∧ d > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_flash_overtakes_ace_l2406_240629


namespace NUMINAMATH_CALUDE_tree_distance_l2406_240630

theorem tree_distance (yard_length : ℝ) (num_trees : ℕ) 
  (h1 : yard_length = 320)
  (h2 : num_trees = 47)
  (h3 : num_trees ≥ 2) :
  let distance := yard_length / (num_trees - 1)
  distance = 320 / 46 := by
sorry

end NUMINAMATH_CALUDE_tree_distance_l2406_240630


namespace NUMINAMATH_CALUDE_arithmetic_sum_11_l2406_240612

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem: The sum of the first 11 terms of the arithmetic sequence
    with a₁ = -11 and d = 2 is equal to -11 -/
theorem arithmetic_sum_11 :
  arithmetic_sum (-11) 2 11 = -11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_11_l2406_240612


namespace NUMINAMATH_CALUDE_rectangle_width_l2406_240659

theorem rectangle_width (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * width → 
  area = length * width → 
  area = 48 → 
  width = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l2406_240659


namespace NUMINAMATH_CALUDE_value_of_expression_l2406_240637

theorem value_of_expression : (-0.125)^2009 * (-8)^2010 = -8 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2406_240637


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l2406_240648

theorem greatest_whole_number_satisfying_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 5 * x - 4 < 3 - 2 * x := by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l2406_240648


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l2406_240640

theorem inscribed_triangle_area (r : ℝ) (a b c : ℝ) (h_radius : r = 5) 
  (h_ratio : ∃ (k : ℝ), a = 4*k ∧ b = 5*k ∧ c = 6*k) 
  (h_inscribed : c = 2*r) : 
  (1/2 : ℝ) * a * b = 250/9 := by
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l2406_240640


namespace NUMINAMATH_CALUDE_notebook_distribution_l2406_240609

theorem notebook_distribution (class_a class_b notebooks_a notebooks_b : ℕ) 
  (h1 : notebooks_a = class_a / 8)
  (h2 : notebooks_b = 2 * class_a)
  (h3 : 16 = (class_a / 2) / 8)
  (h4 : class_a + class_b = (120 * class_a) / 100) :
  class_a * notebooks_a + class_b * notebooks_b = 2176 := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l2406_240609

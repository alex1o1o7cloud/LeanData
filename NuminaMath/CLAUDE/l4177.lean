import Mathlib

namespace NUMINAMATH_CALUDE_distinct_combinations_count_l4177_417791

def word : String := "BIOLOGY"

def num_vowels : Nat := 3
def num_consonants : Nat := 3

def is_vowel (c : Char) : Bool :=
  c = 'I' || c = 'O'

def is_consonant (c : Char) : Bool :=
  c = 'B' || c = 'L' || c = 'G'

def indistinguishable (c : Char) : Bool :=
  c = 'I' || c = 'G'

theorem distinct_combinations_count :
  (∃ (vowel_combs consonant_combs : Nat),
    vowel_combs * consonant_combs = 12 ∧
    vowel_combs = (word.toList.filter is_vowel).length.choose num_vowels ∧
    consonant_combs = (word.toList.filter is_consonant).length.choose num_consonants) :=
by sorry

end NUMINAMATH_CALUDE_distinct_combinations_count_l4177_417791


namespace NUMINAMATH_CALUDE_larger_number_problem_l4177_417708

theorem larger_number_problem (L S : ℕ) (hL : L > S) :
  L - S = 1390 → L = 6 * S + 15 → L = 1665 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l4177_417708


namespace NUMINAMATH_CALUDE_harmonic_mean_three_fourths_five_sixths_l4177_417756

def harmonic_mean (a b : ℚ) : ℚ := 2 / (1/a + 1/b)

theorem harmonic_mean_three_fourths_five_sixths :
  harmonic_mean (3/4) (5/6) = 15/19 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_three_fourths_five_sixths_l4177_417756


namespace NUMINAMATH_CALUDE_function_properties_l4177_417736

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * sin (ω * x) + cos (ω * x + π / 3) + cos (ω * x - π / 3) - 1

noncomputable def g (x : ℝ) : ℝ :=
  2 * sin (2 * x - π / 6) - 1

theorem function_properties (ω : ℝ) (h_ω : ω > 0) :
  (∀ x, f ω x = 2 * sin (2 * x + π / 6) - 1) ∧
  (∀ x, g x = 2 * sin (2 * x - π / 6) - 1) ∧
  (Set.Icc 0 (π / 2)).image g = Set.Icc (-2) 1 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4177_417736


namespace NUMINAMATH_CALUDE_smallest_multiple_of_9_l4177_417755

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

theorem smallest_multiple_of_9 :
  ∀ d : ℕ, d < 10 →
    (is_multiple_of_9 456786 ∧
     is_six_digit 456786 ∧
     456786 = 45678 * 10 + 6) ∧
    (∀ n : ℕ, is_six_digit n ∧ n < 456786 ∧ ∃ d' : ℕ, d' < 10 ∧ n = 45678 * 10 + d' →
      ¬is_multiple_of_9 n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_9_l4177_417755


namespace NUMINAMATH_CALUDE_ken_kept_twenty_pencils_l4177_417770

/-- The number of pencils Ken initially had -/
def initial_pencils : ℕ := 50

/-- The number of pencils Ken gave to Manny -/
def pencils_to_manny : ℕ := 10

/-- The number of additional pencils Ken gave to Nilo compared to Manny -/
def additional_pencils_to_nilo : ℕ := 10

/-- The number of pencils Ken kept -/
def pencils_kept : ℕ := initial_pencils - (pencils_to_manny + (pencils_to_manny + additional_pencils_to_nilo))

theorem ken_kept_twenty_pencils : pencils_kept = 20 := by sorry

end NUMINAMATH_CALUDE_ken_kept_twenty_pencils_l4177_417770


namespace NUMINAMATH_CALUDE_assignment_count_is_correct_l4177_417700

/-- The number of ways to assign 4 people to 3 offices with at least one person in each office -/
def assignmentCount : ℕ := 36

/-- The number of people to be assigned -/
def numPeople : ℕ := 4

/-- The number of offices -/
def numOffices : ℕ := 3

theorem assignment_count_is_correct :
  assignmentCount = (numPeople.choose 2) * numOffices * 2 :=
sorry

end NUMINAMATH_CALUDE_assignment_count_is_correct_l4177_417700


namespace NUMINAMATH_CALUDE_max_value_of_f_l4177_417713

open Real

theorem max_value_of_f (φ : ℝ) :
  (⨆ x, cos (x + 2*φ) + 2*sin φ * sin (x + φ)) = 1 := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l4177_417713


namespace NUMINAMATH_CALUDE_abc_relationship_l4177_417717

theorem abc_relationship : ∀ a b c : ℝ,
  a = 2^(1/5) →
  b = 1 - 2 * Real.log 2 / Real.log 10 →
  c = 2 - Real.log 10 / Real.log 3 →
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_abc_relationship_l4177_417717


namespace NUMINAMATH_CALUDE_min_value_absolute_sum_l4177_417799

theorem min_value_absolute_sum (x : ℝ) : 
  |x - 4| + |x + 6| + |x - 5| ≥ 1 ∧ ∃ y : ℝ, |y - 4| + |y + 6| + |y - 5| = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_absolute_sum_l4177_417799


namespace NUMINAMATH_CALUDE_negation_of_conditional_l4177_417720

theorem negation_of_conditional (a b : ℝ) :
  ¬(a > b → a - 1 > b - 1) ↔ (a ≤ b → a - 1 ≤ b - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_conditional_l4177_417720


namespace NUMINAMATH_CALUDE_function_symmetry_l4177_417757

/-- The function f(x) = 2sin(4x + π/4) is symmetric about the point (-π/16, 0) -/
theorem function_symmetry (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (4 * x + π / 4)
  ∀ y : ℝ, f ((-π/16) + y) = f ((-π/16) - y) :=
by sorry

end NUMINAMATH_CALUDE_function_symmetry_l4177_417757


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_45_l4177_417711

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem no_primes_divisible_by_45 : ¬∃ p : ℕ, is_prime p ∧ 45 ∣ p :=
sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_45_l4177_417711


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l4177_417789

/-- An arithmetic sequence with common difference 1 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 1

/-- The nth term of the sequence -/
def a (n : ℕ) : ℝ := sorry

theorem arithmetic_sequence_first_term :
  arithmetic_sequence a ∧ a 5 ^ 2 = a 3 * a 11 → a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l4177_417789


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4177_417744

theorem imaginary_part_of_z (z : ℂ) (h : z + z * Complex.I = 2) : 
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4177_417744


namespace NUMINAMATH_CALUDE_david_scott_age_difference_l4177_417769

/-- Represents the ages of the three sons -/
structure Ages where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.david > ages.scott ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 11 + 3

/-- The theorem stating that David is 8 years older than Scott -/
theorem david_scott_age_difference (ages : Ages) : 
  problem_conditions ages → ages.david - ages.scott = 8 := by
  sorry

end NUMINAMATH_CALUDE_david_scott_age_difference_l4177_417769


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l4177_417797

theorem complex_arithmetic_equality : (-1 : ℚ)^2023 + (6 - 5/4) * 4/3 + 4 / (-2/3) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l4177_417797


namespace NUMINAMATH_CALUDE_burgerCaloriesTheorem_l4177_417721

/-- Calculates the total calories consumed over a number of days, given the number of burgers eaten per day and calories per burger. -/
def totalCalories (burgersPerDay : ℕ) (caloriesPerBurger : ℕ) (days : ℕ) : ℕ :=
  burgersPerDay * caloriesPerBurger * days

/-- Theorem stating that eating 3 burgers per day, with 20 calories per burger, results in 120 calories consumed after two days. -/
theorem burgerCaloriesTheorem : totalCalories 3 20 2 = 120 := by
  sorry


end NUMINAMATH_CALUDE_burgerCaloriesTheorem_l4177_417721


namespace NUMINAMATH_CALUDE_rainbow_bead_arrangement_probability_l4177_417712

def num_beads : ℕ := 7

def num_permutations (n : ℕ) : ℕ := Nat.factorial n

def probability_specific_arrangement (n : ℕ) : ℚ :=
  1 / (num_permutations n)

theorem rainbow_bead_arrangement_probability :
  probability_specific_arrangement num_beads = 1 / 5040 := by
  sorry

end NUMINAMATH_CALUDE_rainbow_bead_arrangement_probability_l4177_417712


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4177_417773

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  s : ℕ → ℤ  -- The sum of the first n terms
  first_term : a 1 = 31
  sum_equality : s 10 = s 22

/-- The sum formula for the arithmetic sequence -/
def sum_formula (n : ℕ) : ℤ := 32 * n - n^2

/-- Theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.s n = sum_formula n) ∧
  (∃ n, ∀ m, seq.s m ≤ seq.s n) ∧
  (seq.s 16 = 256) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4177_417773


namespace NUMINAMATH_CALUDE_smallest_integer_solution_of_inequalities_l4177_417718

theorem smallest_integer_solution_of_inequalities :
  ∀ x : ℤ,
  (5 * x + 7 > 3 * (x + 1)) ∧
  (1 - (3/2) * x ≤ (1/2) * x - 1) →
  x ≥ 1 ∧
  ∀ y : ℤ, y < 1 →
    ¬((5 * y + 7 > 3 * (y + 1)) ∧
      (1 - (3/2) * y ≤ (1/2) * y - 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_of_inequalities_l4177_417718


namespace NUMINAMATH_CALUDE_computer_price_increase_l4177_417729

theorem computer_price_increase (d : ℝ) : 
  (d * 1.3 = 377) → (2 * d = 580) := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l4177_417729


namespace NUMINAMATH_CALUDE_bird_families_to_africa_l4177_417745

theorem bird_families_to_africa (total : ℕ) (to_asia : ℕ) (remaining : ℕ) : 
  total = 85 → to_asia = 37 → remaining = 25 → total - to_asia - remaining = 23 :=
by sorry

end NUMINAMATH_CALUDE_bird_families_to_africa_l4177_417745


namespace NUMINAMATH_CALUDE_f_quadrants_l4177_417754

/-- A linear function in the Cartesian coordinate system -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Quadrants in the Cartesian coordinate system -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- The set of quadrants a linear function passes through -/
def quadrants_passed (f : LinearFunction) : Set Quadrant :=
  sorry

/-- The specific linear function y = -x - 2 -/
def f : LinearFunction :=
  { slope := -1, intercept := -2 }

/-- Theorem stating which quadrants the function y = -x - 2 passes through -/
theorem f_quadrants :
  quadrants_passed f = {Quadrant.II, Quadrant.III, Quadrant.IV} :=
sorry

end NUMINAMATH_CALUDE_f_quadrants_l4177_417754


namespace NUMINAMATH_CALUDE_total_peppers_weight_l4177_417774

theorem total_peppers_weight :
  2.8333333333333335 + 3.254 + 1.375 + 0.567 = 8.029333333333333 := by
  sorry

end NUMINAMATH_CALUDE_total_peppers_weight_l4177_417774


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_l4177_417778

theorem polynomial_perfect_square (k : ℚ) : 
  (∃ a : ℚ, ∀ x : ℚ, x^2 + 2*(k-9)*x + (k^2 + 3*k + 4) = (x + a)^2) ↔ k = 11/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_l4177_417778


namespace NUMINAMATH_CALUDE_only_η_is_hypergeometric_l4177_417786

-- Define the types for balls and random variables
inductive BallColor
| Black
| White

structure Ball :=
  (color : BallColor)
  (number : Nat)

def TotalBalls : Nat := 10
def BlackBalls : Nat := 6
def WhiteBalls : Nat := 4
def DrawnBalls : Nat := 4

-- Define the random variables
def X (draw : Finset Ball) : Nat := sorry
def Y (draw : Finset Ball) : Nat := sorry
def ξ (draw : Finset Ball) : Nat := sorry
def η (draw : Finset Ball) : Nat := sorry

-- Define the hypergeometric distribution
def IsHypergeometric (f : (Finset Ball) → Nat) : Prop := sorry

-- State the theorem
theorem only_η_is_hypergeometric :
  IsHypergeometric η ∧
  ¬IsHypergeometric X ∧
  ¬IsHypergeometric Y ∧
  ¬IsHypergeometric ξ :=
sorry

end NUMINAMATH_CALUDE_only_η_is_hypergeometric_l4177_417786


namespace NUMINAMATH_CALUDE_terminal_side_second_quadrant_l4177_417750

-- Define the quadrants
inductive Quadrant
| I
| II
| III
| IV

-- Define a function to determine the quadrant of an angle
def angle_quadrant (θ : ℝ) : Quadrant := sorry

-- Define a function to determine the quadrant of the terminal side of an angle
def terminal_side_quadrant (θ : ℝ) : Quadrant := sorry

-- Theorem statement
theorem terminal_side_second_quadrant (α : ℝ) :
  angle_quadrant α = Quadrant.III →
  |Real.cos (α/2)| = -Real.cos (α/2) →
  terminal_side_quadrant (α/2) = Quadrant.II :=
by sorry

end NUMINAMATH_CALUDE_terminal_side_second_quadrant_l4177_417750


namespace NUMINAMATH_CALUDE_abc_sum_bounds_l4177_417788

theorem abc_sum_bounds (a b c d : ℝ) (h : a + b + c = -d) (h_d : d ≠ 0) :
  0 ≤ a * b + a * c + b * c ∧ a * b + a * c + b * c ≤ d^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_bounds_l4177_417788


namespace NUMINAMATH_CALUDE_vector_relations_l4177_417761

def vector_a : Fin 2 → ℝ := ![2, 3]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![x, -6]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, u i = k * v i)

def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem vector_relations :
  (∃ x : ℝ, parallel vector_a (vector_b x) ↔ x = -4) ∧
  (∃ x : ℝ, perpendicular vector_a (vector_b x) ↔ x = 9) := by
  sorry

end NUMINAMATH_CALUDE_vector_relations_l4177_417761


namespace NUMINAMATH_CALUDE_david_ride_bounds_l4177_417730

/-- Represents the number of rides for each person -/
structure RideCount where
  alena : Nat
  bara : Nat
  cenek : Nat
  david : Nat

/-- The constraint that each person rides with every other person at least once -/
def allPairsRide (rc : RideCount) : Prop :=
  rc.alena + rc.bara + rc.cenek + rc.david = 2 * (rc.alena + rc.bara + rc.cenek + rc.david) / 2

/-- The given ride counts for Alena, Bára, and Čeněk -/
def givenRideCounts : RideCount → Prop
  | rc => rc.alena = 11 ∧ rc.bara = 20 ∧ rc.cenek = 4

theorem david_ride_bounds (rc : RideCount) 
  (h1 : givenRideCounts rc) 
  (h2 : allPairsRide rc) : 
  11 ≤ rc.david ∧ rc.david ≤ 29 := by
  sorry

#check david_ride_bounds

end NUMINAMATH_CALUDE_david_ride_bounds_l4177_417730


namespace NUMINAMATH_CALUDE_sheridan_cats_l4177_417765

/-- The total number of cats Mrs. Sheridan has after buying more -/
def total_cats (initial : Float) (bought : Float) : Float :=
  initial + bought

/-- Theorem stating that Mrs. Sheridan's total number of cats is 54.0 -/
theorem sheridan_cats : total_cats 11.0 43.0 = 54.0 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_cats_l4177_417765


namespace NUMINAMATH_CALUDE_adult_office_visit_cost_l4177_417781

/-- Represents the cost of an adult's office visit -/
def adult_cost : ℝ := sorry

/-- Represents the number of adult patients seen per hour -/
def adults_per_hour : ℕ := 4

/-- Represents the number of child patients seen per hour -/
def children_per_hour : ℕ := 3

/-- Represents the cost of a child's office visit -/
def child_cost : ℝ := 25

/-- Represents the number of hours worked in a day -/
def hours_per_day : ℕ := 8

/-- Represents the total income for a day -/
def daily_income : ℝ := 2200

theorem adult_office_visit_cost :
  adult_cost * (adults_per_hour * hours_per_day : ℝ) +
  child_cost * (children_per_hour * hours_per_day : ℝ) =
  daily_income ∧ adult_cost = 50 := by sorry

end NUMINAMATH_CALUDE_adult_office_visit_cost_l4177_417781


namespace NUMINAMATH_CALUDE_fraction_sum_l4177_417751

theorem fraction_sum (a b : ℚ) (h : a / b = 3 / 4) : (a + b) / b = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l4177_417751


namespace NUMINAMATH_CALUDE_prob_even_sum_is_half_l4177_417777

/-- Represents a wheel with a given number of sections and even sections -/
structure Wheel where
  total_sections : ℕ
  even_sections : ℕ

/-- Calculates the probability of getting an even sum when spinning two wheels -/
def prob_even_sum (wheel1 wheel2 : Wheel) : ℚ :=
  let p1_even := wheel1.even_sections / wheel1.total_sections
  let p2_even := wheel2.even_sections / wheel2.total_sections
  p1_even * p2_even + (1 - p1_even) * (1 - p2_even)

/-- The main theorem stating that the probability of getting an even sum
    when spinning the two given wheels is 1/2 -/
theorem prob_even_sum_is_half :
  let wheel1 : Wheel := { total_sections := 5, even_sections := 2 }
  let wheel2 : Wheel := { total_sections := 4, even_sections := 2 }
  prob_even_sum wheel1 wheel2 = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_prob_even_sum_is_half_l4177_417777


namespace NUMINAMATH_CALUDE_no_ten_goals_possible_l4177_417746

/-- Represents a player in the hockey match -/
inductive Player
| Anton
| Ilya
| Sergey

/-- Represents the number of goals scored by each player -/
def GoalCount := Player → ℕ

/-- Represents the statements made by each player -/
def Statements := Player → Player → ℕ

/-- Checks if the statements are consistent with the goal count and the truth-lie condition -/
def ConsistentStatements (gc : GoalCount) (s : Statements) : Prop :=
  ∀ p : Player, (s p p = gc p ∧ s p (nextPlayer p) ≠ gc (nextPlayer p)) ∨
                (s p p ≠ gc p ∧ s p (nextPlayer p) = gc (nextPlayer p))
where
  nextPlayer : Player → Player
  | Player.Anton => Player.Ilya
  | Player.Ilya => Player.Sergey
  | Player.Sergey => Player.Anton

/-- The main theorem stating that it's impossible to have a total of 10 goals -/
theorem no_ten_goals_possible (gc : GoalCount) (s : Statements) :
  ConsistentStatements gc s → (gc Player.Anton + gc Player.Ilya + gc Player.Sergey ≠ 10) := by
  sorry

end NUMINAMATH_CALUDE_no_ten_goals_possible_l4177_417746


namespace NUMINAMATH_CALUDE_third_shiny_on_fifth_probability_l4177_417747

def total_pennies : ℕ := 10
def shiny_pennies : ℕ := 5
def dull_pennies : ℕ := 5
def draws : ℕ := 5

def probability_third_shiny_on_fifth : ℚ :=
  (Nat.choose 4 2 * Nat.choose 6 2) / Nat.choose total_pennies draws

theorem third_shiny_on_fifth_probability :
  probability_third_shiny_on_fifth = 5 / 14 := by sorry

end NUMINAMATH_CALUDE_third_shiny_on_fifth_probability_l4177_417747


namespace NUMINAMATH_CALUDE_remainder_eight_pow_215_mod_9_l4177_417793

theorem remainder_eight_pow_215_mod_9 : 8^215 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eight_pow_215_mod_9_l4177_417793


namespace NUMINAMATH_CALUDE_greatest_odd_factor_number_proof_l4177_417731

/-- A number has an odd number of factors if and only if it is a perfect square -/
def has_odd_factors (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- The greatest whole number less than 1000 that has an odd number of positive factors -/
def greatest_odd_factor_number : ℕ := 961

theorem greatest_odd_factor_number_proof :
  (greatest_odd_factor_number < 1000) ∧
  has_odd_factors greatest_odd_factor_number ∧
  ∀ m : ℕ, m < 1000 → has_odd_factors m → m ≤ greatest_odd_factor_number :=
sorry

end NUMINAMATH_CALUDE_greatest_odd_factor_number_proof_l4177_417731


namespace NUMINAMATH_CALUDE_constant_t_equation_l4177_417722

theorem constant_t_equation (t : ℝ) : 
  (∀ x : ℝ, (3*x^2 - 4*x + 5)*(2*x^2 + t*x + 8) = 6*x^4 - 26*x^3 + 58*x^2 - 76*x + 40) ↔ 
  t = -6 := by
sorry

end NUMINAMATH_CALUDE_constant_t_equation_l4177_417722


namespace NUMINAMATH_CALUDE_last_two_digits_product_l4177_417704

theorem last_two_digits_product (n : ℤ) : 
  (n % 100 ≥ 10) →  -- Ensure n has at least two digits
  (n % 4 = 0) →     -- n is divisible by 4
  ((n % 100) / 10 + n % 10 = 16) →  -- Sum of last two digits is 16
  ((n % 100) / 10) * (n % 10) = 64 := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l4177_417704


namespace NUMINAMATH_CALUDE_inverse_of_f_l4177_417735

noncomputable def f (x : ℝ) := Real.log x + 1

theorem inverse_of_f (x : ℝ) :
  x > 0 → f (Real.exp (x - 1)) = x ∧ Real.exp (f x - 1) = x := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_f_l4177_417735


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l4177_417784

/-- Represents a participant's score in a math challenge --/
structure Score where
  points_scored : ℚ
  points_attempted : ℚ

/-- Calculates the success ratio of a score --/
def successRatio (s : Score) : ℚ := s.points_scored / s.points_attempted

/-- Represents the scores of a participant over two days --/
structure TwoDayScore where
  day1 : Score
  day2 : Score

/-- Calculates the overall success ratio for a two-day score --/
def overallSuccessRatio (s : TwoDayScore) : ℚ :=
  (s.day1.points_scored + s.day2.points_scored) / (s.day1.points_attempted + s.day2.points_attempted)

/-- Gamma's score for each day --/
def gammaScore : Score := { points_scored := 180, points_attempted := 300 }

/-- Delta's maximum possible two-day score --/
def deltaMaxScore : TwoDayScore := {
  day1 := { points_scored := 179, points_attempted := 299 },
  day2 := { points_scored := 180, points_attempted := 301 }
}

theorem delta_max_success_ratio :
  (∀ s : TwoDayScore,
    s.day1.points_attempted + s.day2.points_attempted = 600 ∧
    successRatio s.day1 < successRatio gammaScore ∧
    successRatio s.day2 < successRatio gammaScore) →
  (∀ s : TwoDayScore,
    s.day1.points_attempted + s.day2.points_attempted = 600 ∧
    successRatio s.day1 < successRatio gammaScore ∧
    successRatio s.day2 < successRatio gammaScore →
    overallSuccessRatio s ≤ overallSuccessRatio deltaMaxScore) :=
by sorry

end NUMINAMATH_CALUDE_delta_max_success_ratio_l4177_417784


namespace NUMINAMATH_CALUDE_derivative_of_linear_function_l4177_417734

theorem derivative_of_linear_function (x : ℝ) :
  let y : ℝ → ℝ := λ x => 2 * x
  (deriv y) x = 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_linear_function_l4177_417734


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4177_417792

/-- The function f(x) = ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

/-- The condition a < -2 is sufficient but not necessary for f to have a zero in [-1, 2] -/
theorem sufficient_not_necessary (a : ℝ) :
  (a < -2 → ∃ x ∈ Set.Icc (-1) 2, f a x = 0) ∧
  ¬(∃ x ∈ Set.Icc (-1) 2, f a x = 0 → a < -2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4177_417792


namespace NUMINAMATH_CALUDE_add_point_three_to_twenty_nine_point_eight_l4177_417752

theorem add_point_three_to_twenty_nine_point_eight : 
  29.8 + 0.3 = 30.1 := by
  sorry

end NUMINAMATH_CALUDE_add_point_three_to_twenty_nine_point_eight_l4177_417752


namespace NUMINAMATH_CALUDE_binomial_square_polynomial_l4177_417782

theorem binomial_square_polynomial : ∃ (r s : ℝ), (r * X + s) ^ 2 = 4 * X ^ 2 + 12 * X + 9 :=
sorry

end NUMINAMATH_CALUDE_binomial_square_polynomial_l4177_417782


namespace NUMINAMATH_CALUDE_brett_marbles_l4177_417727

theorem brett_marbles (red : ℕ) (blue : ℕ) : 
  blue = red + 24 → 
  blue = 5 * red → 
  red = 6 := by
  sorry

end NUMINAMATH_CALUDE_brett_marbles_l4177_417727


namespace NUMINAMATH_CALUDE_star_three_five_l4177_417741

def star (a b : ℕ) : ℕ := (a + b) ^ 3

theorem star_three_five : star 3 5 = 512 := by
  sorry

end NUMINAMATH_CALUDE_star_three_five_l4177_417741


namespace NUMINAMATH_CALUDE_existence_of_solutions_l4177_417758

theorem existence_of_solutions (n : ℕ) : ∃ x y z : ℕ, x^2 + y^2 = z^n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solutions_l4177_417758


namespace NUMINAMATH_CALUDE_theater_ticket_contradiction_l4177_417771

theorem theater_ticket_contradiction :
  ∀ (adult_price child_price : ℚ) 
    (total_tickets adult_tickets : ℕ) 
    (total_receipts : ℚ),
  adult_price = 12 →
  total_tickets = 130 →
  adult_tickets = 90 →
  total_receipts = 840 →
  ¬(adult_price * adult_tickets + 
    child_price * (total_tickets - adult_tickets) = 
    total_receipts) :=
by
  sorry

#check theater_ticket_contradiction

end NUMINAMATH_CALUDE_theater_ticket_contradiction_l4177_417771


namespace NUMINAMATH_CALUDE_women_per_table_women_per_table_solution_l4177_417725

theorem women_per_table (num_tables : ℕ) (men_per_table : ℕ) (total_customers : ℕ) : ℕ :=
  let total_men := num_tables * men_per_table
  let total_women := total_customers - total_men
  total_women / num_tables

theorem women_per_table_solution :
  women_per_table 9 3 90 = 7 := by
  sorry

end NUMINAMATH_CALUDE_women_per_table_women_per_table_solution_l4177_417725


namespace NUMINAMATH_CALUDE_percentage_of_percentage_l4177_417767

theorem percentage_of_percentage (y : ℝ) (h : y ≠ 0) :
  (30 / 100) * (60 / 100) * y = (18 / 100) * y :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l4177_417767


namespace NUMINAMATH_CALUDE_log_6_6_log_2_8_log_equation_l4177_417714

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem statements
theorem log_6_6 : log 6 6 = 1 := by sorry

theorem log_2_8 : log 2 8 = 3 := by sorry

theorem log_equation (m : ℝ) : log 2 (m - 2) = 4 → m = 18 := by sorry

end NUMINAMATH_CALUDE_log_6_6_log_2_8_log_equation_l4177_417714


namespace NUMINAMATH_CALUDE_pacos_marble_purchase_l4177_417762

theorem pacos_marble_purchase : 
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_pacos_marble_purchase_l4177_417762


namespace NUMINAMATH_CALUDE_subtract_fractions_l4177_417768

theorem subtract_fractions : (3 : ℚ) / 4 - (1 : ℚ) / 6 = (7 : ℚ) / 12 := by sorry

end NUMINAMATH_CALUDE_subtract_fractions_l4177_417768


namespace NUMINAMATH_CALUDE_range_of_a_l4177_417766

/-- Custom operation ⊗ -/
def custom_op (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of a given the condition -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, custom_op (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4177_417766


namespace NUMINAMATH_CALUDE_log_y_equals_negative_two_l4177_417707

theorem log_y_equals_negative_two (y : ℝ) : 
  y = (Real.log 3 / Real.log 27) ^ (Real.log 81 / Real.log 3) → 
  Real.log y / Real.log 9 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_y_equals_negative_two_l4177_417707


namespace NUMINAMATH_CALUDE_algorithm_output_l4177_417701

def algorithm (n : ℕ) : ℤ :=
  let init := (0 : ℤ)
  init - 3 * n

theorem algorithm_output : algorithm 3 = -9 := by
  sorry

end NUMINAMATH_CALUDE_algorithm_output_l4177_417701


namespace NUMINAMATH_CALUDE_count_positive_area_triangles_l4177_417702

/-- A point in the 4x4 grid -/
structure GridPoint where
  x : Fin 4
  y : Fin 4

/-- A triangle formed by three points in the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Predicate to check if three points are collinear -/
def collinear (p1 p2 p3 : GridPoint) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Predicate to check if a triangle has positive area -/
def positiveArea (t : GridTriangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of all triangles with positive area in the 4x4 grid -/
def PositiveAreaTriangles : Finset GridTriangle :=
  sorry

theorem count_positive_area_triangles :
  Finset.card PositiveAreaTriangles = 520 :=
sorry

end NUMINAMATH_CALUDE_count_positive_area_triangles_l4177_417702


namespace NUMINAMATH_CALUDE_chemists_sons_ages_l4177_417759

theorem chemists_sons_ages (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive integers
  a * b * c = 36 →  -- product is 36
  a + b + c = 13 →  -- sum is 13
  (a ≥ b ∧ a ≥ c) ∨ (b ≥ a ∧ b ≥ c) ∨ (c ≥ a ∧ c ≥ b) →  -- unique oldest son
  (a = 2 ∧ b = 2 ∧ c = 9) ∨ (a = 2 ∧ b = 9 ∧ c = 2) ∨ (a = 9 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_chemists_sons_ages_l4177_417759


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l4177_417715

theorem bowling_ball_weight (b c k : ℝ) 
  (h1 : 9 * b = 6 * c)
  (h2 : c + k = 42)
  (h3 : 3 * k = 2 * c) :
  b = 16.8 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l4177_417715


namespace NUMINAMATH_CALUDE_rectangle_ratio_l4177_417742

theorem rectangle_ratio (w : ℚ) : 
  w > 0 ∧ 2 * w + 2 * 10 = 32 → w / 10 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l4177_417742


namespace NUMINAMATH_CALUDE_parallel_line_slope_l4177_417740

/-- The slope of a line parallel to 3x - 6y = 12 is 1/2 -/
theorem parallel_line_slope (a b c : ℝ) (h : b ≠ 0) :
  (∃ k : ℝ, k ≠ 0 ∧ a = 3 * k ∧ b = -6 * k ∧ c = 12 * k) →
  (a / b : ℝ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l4177_417740


namespace NUMINAMATH_CALUDE_quadratic_always_positive_triangle_angle_sum_product_zero_implies_factor_zero_factors_nonzero_implies_x_not_roots_l4177_417705

-- Statement 1
theorem quadratic_always_positive : ∀ x : ℝ, x^2 - x + 1 > 0 := by sorry

-- Statement 2
theorem triangle_angle_sum : ∀ a b c : ℝ, 
  0 < a ∧ 0 < b ∧ 0 < c → a + b + c = 180 := by sorry

-- Statement 3
theorem product_zero_implies_factor_zero : ∀ a b c : ℝ, 
  a * b * c = 0 → a = 0 ∨ b = 0 ∨ c = 0 := by sorry

-- Statement 4
theorem factors_nonzero_implies_x_not_roots : ∀ x : ℝ, 
  (x - 1) * (x - 2) ≠ 0 → x ≠ 1 ∧ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_triangle_angle_sum_product_zero_implies_factor_zero_factors_nonzero_implies_x_not_roots_l4177_417705


namespace NUMINAMATH_CALUDE_eggs_per_basket_l4177_417723

theorem eggs_per_basket (total_eggs : ℕ) (num_baskets : ℕ) 
  (h1 : total_eggs = 8484) (h2 : num_baskets = 303) :
  total_eggs / num_baskets = 28 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_basket_l4177_417723


namespace NUMINAMATH_CALUDE_correct_product_l4177_417772

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem correct_product (a b : ℕ) : 
  (a ≥ 10 ∧ a ≤ 99) →
  (reverse_digits a * b + 5 = 266) →
  (a * b = 828) :=
by sorry

end NUMINAMATH_CALUDE_correct_product_l4177_417772


namespace NUMINAMATH_CALUDE_min_score_given_average_l4177_417738

theorem min_score_given_average (x y : ℝ) : 
  x ≥ 0 ∧ x ≤ 100 ∧ 
  y ≥ 0 ∧ y ≤ 100 ∧ 
  (69 + 53 + 69 + 71 + 78 + x + y) / 7 = 66 →
  x ≥ 22 :=
by sorry

end NUMINAMATH_CALUDE_min_score_given_average_l4177_417738


namespace NUMINAMATH_CALUDE_yard_fencing_l4177_417743

theorem yard_fencing (length width : ℝ) : 
  length > 0 → 
  width > 0 → 
  length * width = 320 → 
  2 * width + length = 56 → 
  length = 40 := by
sorry

end NUMINAMATH_CALUDE_yard_fencing_l4177_417743


namespace NUMINAMATH_CALUDE_smallest_with_70_divisors_l4177_417739

/-- The number of natural divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- A natural number has exactly 70 divisors -/
def has_70_divisors (n : ℕ) : Prop := num_divisors n = 70

/-- 25920 is the smallest natural number with exactly 70 divisors -/
theorem smallest_with_70_divisors : 
  has_70_divisors 25920 ∧ ∀ m < 25920, ¬has_70_divisors m :=
sorry

end NUMINAMATH_CALUDE_smallest_with_70_divisors_l4177_417739


namespace NUMINAMATH_CALUDE_ellipse_sum_theorem_l4177_417779

/-- Theorem: For an ellipse with given parameters, the sum h + k + a + b + 2c equals 9 + 2√33 -/
theorem ellipse_sum_theorem (h k a b c : ℝ) : 
  h = 3 → 
  k = -5 → 
  a = 7 → 
  b = 4 → 
  c = Real.sqrt (a^2 - b^2) → 
  h + k + a + b + 2*c = 9 + 2 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_theorem_l4177_417779


namespace NUMINAMATH_CALUDE_min_value_a_l4177_417760

theorem min_value_a (a b c d : ℕ+) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 2004)
  (h3 : a^2 - b^2 + c^2 - d^2 = 2004) :
  a ≥ 503 ∧ ∃ (a₀ b₀ c₀ d₀ : ℕ+), 
    a₀ = 503 ∧ 
    a₀ > b₀ ∧ b₀ > c₀ ∧ c₀ > d₀ ∧
    a₀ + b₀ + c₀ + d₀ = 2004 ∧
    a₀^2 - b₀^2 + c₀^2 - d₀^2 = 2004 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l4177_417760


namespace NUMINAMATH_CALUDE_log_5_18_l4177_417716

-- Define the given conditions
variable (a b : ℝ)
variable (h1 : Real.log 2 / Real.log 10 = a)
variable (h2 : Real.log 3 / Real.log 10 = b)

-- State the theorem to be proved
theorem log_5_18 : Real.log 18 / Real.log 5 = (a + 2*b) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_log_5_18_l4177_417716


namespace NUMINAMATH_CALUDE_hare_wolf_distance_l4177_417764

def track_length : ℝ := 200
def hare_speed : ℝ := 5
def wolf_speed : ℝ := 3
def time_elapsed : ℝ := 40

def distance_traveled (speed : ℝ) : ℝ := speed * time_elapsed

def relative_speed : ℝ := hare_speed - wolf_speed

theorem hare_wolf_distance :
  ∀ d : ℝ, d > 0 ∧ d < track_length / 2 →
  (d = distance_traveled relative_speed ∨ d = track_length - distance_traveled relative_speed) →
  d = 40 ∨ d = 60 := by sorry

end NUMINAMATH_CALUDE_hare_wolf_distance_l4177_417764


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4177_417706

theorem quadratic_equation_solution (h : 108 * (3/4)^2 + 61 = 145 * (3/4) - 7) :
  ∃ x : ℚ, x ≠ 3/4 ∧ 108 * x^2 + 61 = 145 * x - 7 ∧ x = 68/81 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4177_417706


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l4177_417726

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 5 < x ∧ x < 21)
  (h2 : 7 < x)
  (h3 : 13 > x ∧ x > 2)
  (h4 : 12 > x ∧ x > 9)
  (h5 : x + 1 < 13) :
  ∀ y : ℤ, x < y → 12 ≤ y := by
  sorry

#check smallest_upper_bound

end NUMINAMATH_CALUDE_smallest_upper_bound_l4177_417726


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4177_417787

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 6 + 2 * a 4 * a 5 + a 5 ^ 2 = 25 →
  a 4 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4177_417787


namespace NUMINAMATH_CALUDE_solve_consecutive_integer_sets_l4177_417703

/-- A set of consecutive integers -/
structure ConsecutiveIntegerSet where
  start : ℤ
  size : ℕ

/-- The sum of elements in a ConsecutiveIntegerSet -/
def sum_of_set (s : ConsecutiveIntegerSet) : ℤ :=
  (s.size : ℤ) * (2 * s.start + s.size - 1) / 2

/-- The greatest element in a ConsecutiveIntegerSet -/
def greatest_element (s : ConsecutiveIntegerSet) : ℤ :=
  s.start + s.size - 1

theorem solve_consecutive_integer_sets :
  ∃ (m : ℕ) (a b : ConsecutiveIntegerSet),
    m > 0 ∧
    a.size = m ∧
    b.size = 2 * m ∧
    sum_of_set a = 2 * m ∧
    sum_of_set b = m ∧
    |greatest_element a - greatest_element b| = 99 →
    m = 201 := by
  sorry

end NUMINAMATH_CALUDE_solve_consecutive_integer_sets_l4177_417703


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_cone_lateral_surface_area_proof_l4177_417790

/-- The lateral surface area of a cone with base radius 3 and lateral surface that unfolds into a semicircle -/
theorem cone_lateral_surface_area : Real :=
  let base_radius : Real := 3
  let lateral_surface_is_semicircle : Prop := True  -- This is a placeholder for the condition
  18 * Real.pi

/-- Proof of the lateral surface area of the cone -/
theorem cone_lateral_surface_area_proof :
  cone_lateral_surface_area = 18 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_cone_lateral_surface_area_proof_l4177_417790


namespace NUMINAMATH_CALUDE_min_value_expression_l4177_417719

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b)^2 ≥ 7 ∧
  (2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b)^2 = 7 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l4177_417719


namespace NUMINAMATH_CALUDE_expression_simplification_l4177_417710

theorem expression_simplification (x : ℝ) (hx : x ≠ 0) :
  ((2 * x^2)^3 - 6 * x^3 * (x^3 - 2 * x^2)) / (2 * x^4) = x^2 + 6 * x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4177_417710


namespace NUMINAMATH_CALUDE_matrix_vector_computation_l4177_417749

variable {n : ℕ}
variable (N : Matrix (Fin 2) (Fin n) ℝ)
variable (a b : Fin n → ℝ)

theorem matrix_vector_computation 
  (ha : N.mulVec a = ![3, 4]) 
  (hb : N.mulVec b = ![1, -2]) :
  N.mulVec (2 • a - 4 • b) = ![2, 16] := by
  sorry

end NUMINAMATH_CALUDE_matrix_vector_computation_l4177_417749


namespace NUMINAMATH_CALUDE_regression_properties_l4177_417783

-- Define the data points
def data_points : List (ℝ × ℝ) := [(5, 17), (6, 20), (8, 25), (9, 28), (12, 35)]

-- Define the empirical regression equation
def regression_equation (x : ℝ) : ℝ := 2.6 * x + 4.2

-- Theorem to prove the three statements
theorem regression_properties :
  -- 1. The point (8, 25) lies on the regression line
  regression_equation 8 = 25 ∧
  -- 2. The y-intercept of the regression line is 4.2
  regression_equation 0 = 4.2 ∧
  -- 3. The residual for x = 5 is -0.2
  17 - regression_equation 5 = -0.2 := by
  sorry

end NUMINAMATH_CALUDE_regression_properties_l4177_417783


namespace NUMINAMATH_CALUDE_sqrt_two_cos_thirty_degrees_l4177_417763

theorem sqrt_two_cos_thirty_degrees : 
  Real.sqrt 2 * Real.cos (30 * π / 180) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_cos_thirty_degrees_l4177_417763


namespace NUMINAMATH_CALUDE_vector_BA_l4177_417728

def complex_vector (a b : ℂ) : ℂ := a - b

theorem vector_BA (OA OB : ℂ) :
  OA = 2 - 3*I ∧ OB = -3 + 2*I →
  complex_vector OA OB = 5 - 5*I :=
by sorry

end NUMINAMATH_CALUDE_vector_BA_l4177_417728


namespace NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l4177_417709

theorem largest_of_seven_consecutive_integers (a : ℕ) 
  (h1 : a > 0)
  (h2 : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) = 77) :
  a + 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l4177_417709


namespace NUMINAMATH_CALUDE_park_diameter_l4177_417724

/-- Given a circular park with a central pond, vegetable garden, and jogging path,
    this theorem proves that the diameter of the outer boundary is 64 feet. -/
theorem park_diameter (pond_diameter vegetable_width jogging_width : ℝ) 
  (h1 : pond_diameter = 20)
  (h2 : vegetable_width = 12)
  (h3 : jogging_width = 10) :
  2 * (pond_diameter / 2 + vegetable_width + jogging_width) = 64 := by
  sorry

end NUMINAMATH_CALUDE_park_diameter_l4177_417724


namespace NUMINAMATH_CALUDE_perimeter_equality_l4177_417776

/-- The perimeter of a rectangle -/
def rectangle_perimeter (width : ℕ) (height : ℕ) : ℕ :=
  2 * (width + height)

/-- The perimeter of a figure composed of two rectangles sharing one edge -/
def composite_perimeter (width1 : ℕ) (height1 : ℕ) (width2 : ℕ) (height2 : ℕ) (shared_edge : ℕ) : ℕ :=
  rectangle_perimeter width1 height1 + rectangle_perimeter width2 height2 - 2 * shared_edge

theorem perimeter_equality :
  rectangle_perimeter 4 3 = composite_perimeter 2 3 3 2 3 := by
  sorry

#eval rectangle_perimeter 4 3
#eval composite_perimeter 2 3 3 2 3

end NUMINAMATH_CALUDE_perimeter_equality_l4177_417776


namespace NUMINAMATH_CALUDE_goose_eggs_count_l4177_417785

theorem goose_eggs_count (
  total_eggs : ℕ
  ) (
  hatched_ratio : Rat
  ) (
  first_month_survival_ratio : Rat
  ) (
  first_year_death_ratio : Rat
  ) (
  first_year_survivors : ℕ
  ) : total_eggs = 2200 :=
  by
  have h1 : hatched_ratio = 2 / 3 := by sorry
  have h2 : first_month_survival_ratio = 3 / 4 := by sorry
  have h3 : first_year_death_ratio = 3 / 5 := by sorry
  have h4 : first_year_survivors = 110 := by sorry
  have h5 : ∀ e, e ≤ 1 := by sorry  -- No more than one goose hatched from each egg
  
  sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l4177_417785


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_one_l4177_417794

theorem sum_of_fractions_geq_one (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_one_l4177_417794


namespace NUMINAMATH_CALUDE_parallel_line_slope_l4177_417796

/-- Given a line with equation 3x - 6y = 12, prove that the slope of any parallel line is 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 12) → 
  ∃ (m b : ℝ), (y = m * x + b) ∧ (m = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l4177_417796


namespace NUMINAMATH_CALUDE_sine_inequality_l4177_417798

theorem sine_inequality : 
  let sin60 := Real.sqrt 3 / 2
  let sin62 := (Real.sqrt 2 / 2) * (Real.sin (17 * π / 180) + Real.cos (17 * π / 180))
  let sin64 := 2 * (Real.cos (13 * π / 180))^2 - 1
  sin60 < sin62 ∧ sin62 < sin64 := by sorry

end NUMINAMATH_CALUDE_sine_inequality_l4177_417798


namespace NUMINAMATH_CALUDE_system_solution_l4177_417748

theorem system_solution :
  ∀ (x y z : ℝ),
    (y + z = x * y * z ∧
     z + x = x * y * z ∧
     x + y = x * y * z) →
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨
     (x = Real.sqrt 2 ∧ y = Real.sqrt 2 ∧ z = Real.sqrt 2) ∨
     (x = -Real.sqrt 2 ∧ y = -Real.sqrt 2 ∧ z = -Real.sqrt 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4177_417748


namespace NUMINAMATH_CALUDE_box_triangle_area_theorem_l4177_417795

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  width : ℝ
  length : ℝ
  height : ℚ

/-- Calculates the area of the triangle formed by the center points of three faces meeting at a corner -/
def triangleArea (d : BoxDimensions) : ℝ :=
  sorry

/-- Checks if two integers are relatively prime -/
def relativelyPrime (m n : ℕ) : Prop :=
  sorry

theorem box_triangle_area_theorem 
  (d : BoxDimensions)
  (m n : ℕ)
  (h1 : d.width = 15)
  (h2 : d.length = 20)
  (h3 : d.height = m / n)
  (h4 : relativelyPrime m n)
  (h5 : triangleArea d = 40) :
  m + n = 69 :=
sorry

end NUMINAMATH_CALUDE_box_triangle_area_theorem_l4177_417795


namespace NUMINAMATH_CALUDE_fraction_addition_l4177_417775

theorem fraction_addition (y C D : ℚ) : 
  (6 * y - 15) / (3 * y^3 - 13 * y^2 + 4 * y + 12) = C / (y + 3) + D / (3 * y^2 - 10 * y + 4) →
  C = -3/17 ∧ D = 81/17 := by
sorry

end NUMINAMATH_CALUDE_fraction_addition_l4177_417775


namespace NUMINAMATH_CALUDE_sum_equals_five_l4177_417733

/-- Definition of the star operation -/
def star (a b : ℕ) : ℤ := a^b - a*b

/-- Theorem statement -/
theorem sum_equals_five (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (h : star a b = 3) : a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_five_l4177_417733


namespace NUMINAMATH_CALUDE_mystery_number_l4177_417753

theorem mystery_number (x : ℕ) (h : x + 45 = 92) : x = 47 := by
  sorry

end NUMINAMATH_CALUDE_mystery_number_l4177_417753


namespace NUMINAMATH_CALUDE_trapezoid_area_l4177_417780

/-- The area of a trapezoid with height x, one base 3x, and the other base 5x, is 4x² -/
theorem trapezoid_area (x : ℝ) (h : x > 0) : 
  let height := x
  let base1 := 3 * x
  let base2 := 5 * x
  let area := height * (base1 + base2) / 2
  area = 4 * x^2 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_area_l4177_417780


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l4177_417732

/-- The modulus of (2i / (1 + i)) is √2 -/
theorem modulus_of_complex_fraction : 
  Complex.abs (2 * Complex.I / (1 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l4177_417732


namespace NUMINAMATH_CALUDE_fraction_decimal_difference_l4177_417737

theorem fraction_decimal_difference : 
  2/3 - 0.66666667 = 1/(3 * 10^8) := by sorry

end NUMINAMATH_CALUDE_fraction_decimal_difference_l4177_417737

import Mathlib

namespace NUMINAMATH_CALUDE_fifteenth_student_age_l635_63522

/-- Given a class of 15 students with an average age of 15 years,
    where 8 students have an average age of 14 years and 6 students
    have an average age of 16 years, the age of the 15th student is 17 years. -/
theorem fifteenth_student_age
  (total_students : Nat)
  (total_average_age : ℚ)
  (group1_students : Nat)
  (group1_average_age : ℚ)
  (group2_students : Nat)
  (group2_average_age : ℚ)
  (h1 : total_students = 15)
  (h2 : total_average_age = 15)
  (h3 : group1_students = 8)
  (h4 : group1_average_age = 14)
  (h5 : group2_students = 6)
  (h6 : group2_average_age = 16) :
  (total_students * total_average_age) - (group1_students * group1_average_age) - (group2_students * group2_average_age) = 17 := by
  sorry


end NUMINAMATH_CALUDE_fifteenth_student_age_l635_63522


namespace NUMINAMATH_CALUDE_statement_1_statement_4_l635_63587

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contained : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Statement ①
theorem statement_1 (m n : Line) (α : Plane) :
  perpendicular m n → perpendicularLP m α → ¬contained n α → parallel n α :=
sorry

-- Statement ④
theorem statement_4 (m n : Line) (α β : Plane) :
  perpendicular m n → perpendicularLP m α → perpendicularLP n β → perpendicularPP α β :=
sorry

end NUMINAMATH_CALUDE_statement_1_statement_4_l635_63587


namespace NUMINAMATH_CALUDE_arrangements_count_l635_63549

/-- The number of applicants --/
def num_applicants : ℕ := 5

/-- The number of students to be selected --/
def num_selected : ℕ := 3

/-- The number of events --/
def num_events : ℕ := 3

/-- Function to calculate the number of arrangements --/
def num_arrangements (n_applicants : ℕ) (n_selected : ℕ) (n_events : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of arrangements --/
theorem arrangements_count :
  num_arrangements num_applicants num_selected num_events = 48 :=
sorry

end NUMINAMATH_CALUDE_arrangements_count_l635_63549


namespace NUMINAMATH_CALUDE_max_candy_remainder_l635_63542

theorem max_candy_remainder (n : ℕ) : 
  ∃ (k : ℕ), n^2 = 5 * k + 4 ∧ 
  ∀ (m : ℕ), n^2 = 5 * m + (n^2 % 5) → n^2 % 5 ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_candy_remainder_l635_63542


namespace NUMINAMATH_CALUDE_weeks_to_afford_bicycle_l635_63503

def bicycle_cost : ℕ := 600
def birthday_money : ℕ := 165
def weekly_earnings : ℕ := 20

theorem weeks_to_afford_bicycle :
  let total_money : ℕ → ℕ := λ weeks => birthday_money + weekly_earnings * weeks
  ∀ weeks : ℕ, total_money weeks ≥ bicycle_cost → weeks ≥ 22 :=
by
  sorry

end NUMINAMATH_CALUDE_weeks_to_afford_bicycle_l635_63503


namespace NUMINAMATH_CALUDE_change_percentage_closest_to_five_l635_63540

def item_prices : List ℚ := [12.99, 9.99, 7.99, 6.50, 4.99, 3.75, 1.27]
def payment : ℚ := 50

def total_price : ℚ := item_prices.sum
def change : ℚ := payment - total_price
def change_percentage : ℚ := (change / payment) * 100

theorem change_percentage_closest_to_five :
  ∀ x ∈ [3, 5, 7, 10, 12], |change_percentage - 5| ≤ |change_percentage - x| :=
by sorry

end NUMINAMATH_CALUDE_change_percentage_closest_to_five_l635_63540


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l635_63563

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 → 
  (∃ r : ℝ, 160 * r = b ∧ b * r = 108 / 64) → 
  b = 15 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l635_63563


namespace NUMINAMATH_CALUDE_square_roots_theorem_l635_63523

theorem square_roots_theorem (x a : ℝ) (hx : x > 0) :
  (∃ (r₁ r₂ : ℝ), r₁ = 2*a - 1 ∧ r₂ = -a + 2 ∧ r₁^2 = x ∧ r₂^2 = x) →
  a = -1 ∧ x = 9 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l635_63523


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l635_63512

/-- Given that a² varies inversely with b⁴, and a = 7 when b = 2, 
    prove that a² = 3.0625 when b = 4 -/
theorem inverse_variation_problem (a b : ℝ) (k : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → a^2 * x^4 = k) →  -- a² varies inversely with b⁴
  (7^2 * 2^4 = k) →                             -- a = 7 when b = 2
  (a^2 * 4^4 = k) →                             -- condition for b = 4
  a^2 = 3.0625                                  -- conclusion
:= by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l635_63512


namespace NUMINAMATH_CALUDE_unique_solution_cos_arctan_sin_arccos_l635_63562

theorem unique_solution_cos_arctan_sin_arccos (z : ℝ) :
  (∃! z : ℝ, 0 ≤ z ∧ z ≤ 1 ∧ Real.cos (Real.arctan (Real.sin (Real.arccos z))) = z) ∧
  (Real.cos (Real.arctan (Real.sin (Real.arccos (Real.sqrt 2 / 2)))) = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cos_arctan_sin_arccos_l635_63562


namespace NUMINAMATH_CALUDE_prob_red_then_green_l635_63586

/-- A bag containing one red ball and one green ball -/
structure Bag :=
  (red : Nat)
  (green : Nat)

/-- The initial state of the bag -/
def initial_bag : Bag :=
  { red := 1, green := 1 }

/-- A draw from the bag -/
inductive Draw
  | Red
  | Green

/-- The probability of drawing a specific sequence of two balls -/
def prob_draw (first second : Draw) : ℚ :=
  1 / 4

/-- Theorem: The probability of drawing a red ball first and a green ball second is 1/4 -/
theorem prob_red_then_green :
  prob_draw Draw.Red Draw.Green = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_green_l635_63586


namespace NUMINAMATH_CALUDE_arrangements_with_restrictions_total_arrangements_prove_total_arrangements_l635_63533

-- Define the number of people
def n : ℕ := 5

-- Define the function to calculate permutations
def permutations (k : ℕ) : ℕ := Nat.factorial k

-- Define the function to calculate arrangements
def arrangements (n k : ℕ) : ℕ := permutations n / permutations (n - k)

-- Theorem statement
theorem arrangements_with_restrictions :
  arrangements n n - 2 * arrangements (n - 1) (n - 1) + arrangements (n - 2) (n - 2) = 78 := by
  sorry

-- The result we want to prove
theorem total_arrangements : ℕ := 78

-- The main theorem
theorem prove_total_arrangements :
  arrangements n n - 2 * arrangements (n - 1) (n - 1) + arrangements (n - 2) (n - 2) = total_arrangements := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_restrictions_total_arrangements_prove_total_arrangements_l635_63533


namespace NUMINAMATH_CALUDE_pet_store_cages_l635_63583

theorem pet_store_cages (total_birds : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) 
  (h1 : total_birds = 54)
  (h2 : parrots_per_cage = 2)
  (h3 : parakeets_per_cage = 7) :
  total_birds / (parrots_per_cage + parakeets_per_cage) = 6 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l635_63583


namespace NUMINAMATH_CALUDE_circle_center_l635_63548

/-- The center of a circle defined by the equation (x+2)^2 + (y-1)^2 = 1 is at the point (-2, 1) -/
theorem circle_center (x y : ℝ) : 
  ((x + 2)^2 + (y - 1)^2 = 1) → ((-2, 1) : ℝ × ℝ) = (x, y) := by
sorry

end NUMINAMATH_CALUDE_circle_center_l635_63548


namespace NUMINAMATH_CALUDE_newer_car_distance_l635_63560

theorem newer_car_distance (older_distance : ℝ) (percentage_increase : ℝ) 
  (h1 : older_distance = 150)
  (h2 : percentage_increase = 0.30) : 
  older_distance * (1 + percentage_increase) = 195 :=
by sorry

end NUMINAMATH_CALUDE_newer_car_distance_l635_63560


namespace NUMINAMATH_CALUDE_exterior_angle_pentagon_octagon_exterior_angle_pentagon_octagon_is_117_l635_63535

/-- The measure of the exterior angle DEF in a configuration where a regular pentagon
    and a regular octagon share a side. -/
theorem exterior_angle_pentagon_octagon : ℝ :=
  let pentagon_interior_angle : ℝ := 180 * (5 - 2) / 5
  let octagon_interior_angle : ℝ := 180 * (8 - 2) / 8
  let sum_of_angles_at_E : ℝ := 360
  117

/-- Proof that the exterior angle DEF measures 117° when a regular pentagon ABCDE
    and a regular octagon AEFGHIJK share a side AE in a plane. -/
theorem exterior_angle_pentagon_octagon_is_117 :
  exterior_angle_pentagon_octagon = 117 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_pentagon_octagon_exterior_angle_pentagon_octagon_is_117_l635_63535


namespace NUMINAMATH_CALUDE_digit_puzzle_l635_63591

def is_not_zero (d : Nat) : Prop := d ≠ 0
def is_even (d : Nat) : Prop := d % 2 = 0
def is_five (d : Nat) : Prop := d = 5
def is_not_six (d : Nat) : Prop := d ≠ 6
def is_less_than_seven (d : Nat) : Prop := d < 7

theorem digit_puzzle (d : Nat) 
  (h_range : d ≤ 9)
  (h_statements : ∃! (s : Fin 5), ¬(
    match s with
    | 0 => is_not_zero d
    | 1 => is_even d
    | 2 => is_five d
    | 3 => is_not_six d
    | 4 => is_less_than_seven d
  )) :
  ¬(is_even d) :=
sorry

end NUMINAMATH_CALUDE_digit_puzzle_l635_63591


namespace NUMINAMATH_CALUDE_clay_pot_earnings_l635_63550

/-- Calculate the money earned from selling clay pots --/
theorem clay_pot_earnings (total_pots : ℕ) (cracked_fraction : ℚ) (price_per_pot : ℕ) : 
  total_pots = 80 →
  cracked_fraction = 2 / 5 →
  price_per_pot = 40 →
  (total_pots : ℚ) * (1 - cracked_fraction) * price_per_pot = 1920 := by
  sorry

end NUMINAMATH_CALUDE_clay_pot_earnings_l635_63550


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l635_63508

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a4 : a 4 = 9)
  (h_sum : a 3 + a 7 = 20) :
  ∃ d : ℝ, d = 1 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l635_63508


namespace NUMINAMATH_CALUDE_inverse_proportion_k_range_l635_63501

/-- Prove that for an inverse proportion function y = (4-k)/x with points A(x₁, y₁) and B(x₂, y₂) 
    on its graph, where x₁ < 0 < x₂ and y₁ < y₂, the range of values for k is k < 4. -/
theorem inverse_proportion_k_range (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁ < 0) (h2 : 0 < x₂) (h3 : y₁ < y₂)
  (h4 : y₁ = (4 - k) / x₁) (h5 : y₂ = (4 - k) / x₂) :
  k < 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_range_l635_63501


namespace NUMINAMATH_CALUDE_ten_person_round_robin_matches_l635_63575

/-- Calculates the number of matches in a round-robin tournament -/
def roundRobinMatches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: A 10-person round-robin tournament has 45 matches -/
theorem ten_person_round_robin_matches :
  roundRobinMatches 10 = 45 := by
  sorry

#eval roundRobinMatches 10  -- Should output 45

end NUMINAMATH_CALUDE_ten_person_round_robin_matches_l635_63575


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_condition_l635_63516

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_sum_condition
  (a₁ d : ℝ) (n : ℕ)
  (h1 : a₁ = 2)
  (h2 : d = 3)
  (h3 : arithmetic_sequence a₁ d n + arithmetic_sequence a₁ d (n + 2) = 28) :
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_condition_l635_63516


namespace NUMINAMATH_CALUDE_vector_expression_evaluation_l635_63539

/-- Given the vector expression, prove that it equals the result vector. -/
theorem vector_expression_evaluation :
  (⟨3, -2⟩ : ℝ × ℝ) - 5 • ⟨2, -6⟩ + ⟨0, 3⟩ = ⟨-7, 31⟩ := by
  sorry

end NUMINAMATH_CALUDE_vector_expression_evaluation_l635_63539


namespace NUMINAMATH_CALUDE_no_prime_pair_with_odd_difference_quotient_l635_63500

theorem no_prime_pair_with_odd_difference_quotient :
  ¬ ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p > q ∧ (∃ (k : ℕ), 2 * k + 1 = (p^2 - q^2) / 4) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_pair_with_odd_difference_quotient_l635_63500


namespace NUMINAMATH_CALUDE_unique_function_property_l635_63576

theorem unique_function_property (f : ℝ → ℝ) 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (x + 1) = f x + 1)
  (h3 : ∀ x ≠ 0, f (1 / x) = f x / x^2) :
  ∀ x, f x = x := by
sorry

end NUMINAMATH_CALUDE_unique_function_property_l635_63576


namespace NUMINAMATH_CALUDE_irrational_approximation_l635_63552

theorem irrational_approximation (k : ℝ) (ε : ℝ) 
  (h_irr : Irrational k) (h_pos : ε > 0) :
  ∃ (m n : ℤ), |m * k - n| < ε :=
sorry

end NUMINAMATH_CALUDE_irrational_approximation_l635_63552


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_360_l635_63514

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_360 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ sum_of_divisors 360 ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ sum_of_divisors 360 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_360_l635_63514


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l635_63519

theorem pure_imaginary_condition (x : ℝ) : 
  (((x^2 - 1) : ℂ) + (x - 1) * Complex.I).re = 0 ∧ 
  (((x^2 - 1) : ℂ) + (x - 1) * Complex.I).im ≠ 0 → 
  x = -1 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l635_63519


namespace NUMINAMATH_CALUDE_fifth_odd_with_odd_factors_is_81_l635_63558

/-- A function that returns true if a number is a perfect square, false otherwise -/
def is_perfect_square (n : ℕ) : Bool := sorry

/-- A function that returns true if a number has an odd number of factors, false otherwise -/
def has_odd_factors (n : ℕ) : Bool := is_perfect_square n

/-- A function that returns the nth odd integer with an odd number of factors -/
def nth_odd_with_odd_factors (n : ℕ) : ℕ := sorry

theorem fifth_odd_with_odd_factors_is_81 :
  nth_odd_with_odd_factors 5 = 81 := by sorry

end NUMINAMATH_CALUDE_fifth_odd_with_odd_factors_is_81_l635_63558


namespace NUMINAMATH_CALUDE_number_problem_l635_63553

theorem number_problem (X Y Z : ℝ) 
  (h1 : X - Y = 3500)
  (h2 : (3/5) * X = (2/3) * Y)
  (h3 : 0.097 * Y = Real.sqrt Z) :
  X = 35000 ∧ Y = 31500 ∧ Z = 9333580.25 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l635_63553


namespace NUMINAMATH_CALUDE_football_players_count_l635_63599

theorem football_players_count (cricket_players hockey_players softball_players total_players : ℕ) 
  (h1 : cricket_players = 10)
  (h2 : hockey_players = 12)
  (h3 : softball_players = 13)
  (h4 : total_players = 51) :
  total_players - (cricket_players + hockey_players + softball_players) = 16 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l635_63599


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l635_63529

/-- An increasing arithmetic sequence of integers -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_product (a : ℕ → ℤ) :
  ArithmeticSequence a → a 4 * a 5 = 24 → a 3 * a 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l635_63529


namespace NUMINAMATH_CALUDE_total_votes_is_1375_l635_63598

/-- Represents the election results with given conditions -/
structure ElectionResults where
  winners_votes : ℕ  -- Combined majority of winners
  spoiled_votes : ℕ  -- Number of spoiled votes
  final_percentages : List ℚ  -- Final round percentages for top three candidates

/-- Calculates the total number of votes cast in the election -/
def total_votes (results : ElectionResults) : ℕ :=
  results.winners_votes + results.spoiled_votes

/-- Theorem stating that the total number of votes is 1375 given the conditions -/
theorem total_votes_is_1375 (results : ElectionResults) 
  (h1 : results.winners_votes = 1050)
  (h2 : results.spoiled_votes = 325)
  (h3 : results.final_percentages = [41/100, 34/100, 25/100]) :
  total_votes results = 1375 := by
  sorry

#eval total_votes { winners_votes := 1050, spoiled_votes := 325, final_percentages := [41/100, 34/100, 25/100] }

end NUMINAMATH_CALUDE_total_votes_is_1375_l635_63598


namespace NUMINAMATH_CALUDE_davids_english_marks_l635_63582

def davidsMathMarks : ℕ := 65
def davidsPhysicsMarks : ℕ := 82
def davidsChemistryMarks : ℕ := 67
def davidsBiologyMarks : ℕ := 85
def davidsAverageMarks : ℕ := 76
def totalSubjects : ℕ := 5

theorem davids_english_marks :
  ∃ (englishMarks : ℕ), 
    (englishMarks + davidsMathMarks + davidsPhysicsMarks + davidsChemistryMarks + davidsBiologyMarks) / totalSubjects = davidsAverageMarks ∧
    englishMarks = 81 :=
by sorry

end NUMINAMATH_CALUDE_davids_english_marks_l635_63582


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l635_63527

/-- Given a parabola with equation x = 2py^2 where p > 0, its focus has coordinates (1/(8p), 0) -/
theorem parabola_focus_coordinates (p : ℝ) (hp : p > 0) :
  let parabola := {(x, y) : ℝ × ℝ | x = 2 * p * y^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (1 / (8 * p), 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l635_63527


namespace NUMINAMATH_CALUDE_greatest_n_is_5_l635_63517

/-- A coloring of a board is a function that assigns a color to each square. -/
def Coloring (m n : ℕ) := Fin m → Fin n → Fin 3

/-- A board has a valid coloring if no rectangle has all four corners of the same color. -/
def ValidColoring (m n : ℕ) (c : Coloring m n) : Prop :=
  ∀ (r1 r2 : Fin m) (c1 c2 : Fin n),
    r1 ≠ r2 → c1 ≠ c2 →
    (c r1 c1 = c r1 c2 ∧ c r1 c1 = c r2 c1 ∧ c r1 c1 = c r2 c2) → False

/-- The main theorem: The greatest possible value of n is 5. -/
theorem greatest_n_is_5 :
  (∃ (c : Coloring 6 4), ValidColoring 6 4 c) ∧
  (∀ n > 5, ¬∃ (c : Coloring (n+1) (n-1)), ValidColoring (n+1) (n-1) c) :=
sorry

end NUMINAMATH_CALUDE_greatest_n_is_5_l635_63517


namespace NUMINAMATH_CALUDE_wrappers_collection_proof_l635_63547

/-- The number of wrappers collected by Andy -/
def andy_wrappers : ℕ := 34

/-- The number of wrappers collected by Max -/
def max_wrappers : ℕ := 15

/-- The number of wrappers collected by Zoe -/
def zoe_wrappers : ℕ := 25

/-- The total number of wrappers collected by all three friends -/
def total_wrappers : ℕ := andy_wrappers + max_wrappers + zoe_wrappers

theorem wrappers_collection_proof : total_wrappers = 74 := by
  sorry

end NUMINAMATH_CALUDE_wrappers_collection_proof_l635_63547


namespace NUMINAMATH_CALUDE_system_solutions_l635_63554

/-- The system of equations has only two solutions -/
theorem system_solutions :
  ∀ x y z : ℝ,
  (x + y * z = 2 ∧ y + x * z = 2 ∧ z + x * y = 2) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -2 ∧ y = -2 ∧ z = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l635_63554


namespace NUMINAMATH_CALUDE_max_length_sum_l635_63577

/-- The length of an integer is the number of positive prime factors, not necessarily distinct, whose product is equal to the integer. -/
def length (k : ℕ) : ℕ := sorry

/-- A number is prime if it has exactly two factors -/
def isPrime (p : ℕ) : Prop := sorry

theorem max_length_sum :
  ∀ x y z : ℕ,
  x > 1 → y > 1 → z > 1 →
  (∃ p q : ℕ, isPrime p ∧ isPrime q ∧ p ≠ q ∧ x = p * q) →
  (∃ p q r : ℕ, isPrime p ∧ isPrime q ∧ isPrime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ y = p * q * r) →
  x + 3 * y + 5 * z < 5000 →
  length x + length y + length z ≤ 14 :=
sorry

end NUMINAMATH_CALUDE_max_length_sum_l635_63577


namespace NUMINAMATH_CALUDE_ab_value_l635_63564

theorem ab_value (a b : ℝ) (h1 : a - b = 10) (h2 : a^2 + b^2 = 150) : a * b = 25 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l635_63564


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l635_63579

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l635_63579


namespace NUMINAMATH_CALUDE_greatest_k_value_l635_63590

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 145) →
  k ≤ Real.sqrt 177 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l635_63590


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l635_63593

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  (a 1 = 1) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n) →
  (∀ n : ℕ, n ≥ 1 → a n = 2^(n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l635_63593


namespace NUMINAMATH_CALUDE_gp_solution_and_sum_l635_63505

/-- Given a real number x, returns true if 10+x, 30+x, and 90+x form a geometric progression -/
def isGeometricProgression (x : ℝ) : Prop :=
  (30 + x)^2 = (10 + x) * (90 + x)

/-- Computes the sum of the terms in the progression for a given x -/
def sumOfProgression (x : ℝ) : ℝ :=
  (10 + x) + (30 + x) + (90 + x)

theorem gp_solution_and_sum :
  ∃! x : ℝ, isGeometricProgression x ∧ sumOfProgression x = 130 :=
sorry

end NUMINAMATH_CALUDE_gp_solution_and_sum_l635_63505


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l635_63520

/-- Given a point P in the Cartesian coordinate system, 
    find its symmetric point with respect to the x-axis. -/
def symmetric_point_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

/-- Theorem: The coordinates of the point symmetric to P (-1, 2) 
    with respect to the x-axis are (-1, -2). -/
theorem symmetric_point_theorem : 
  symmetric_point_x_axis (-1, 2) = (-1, -2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l635_63520


namespace NUMINAMATH_CALUDE_unique_divisibility_by_99_l635_63555

-- Define the structure of the number N
def N (a b : ℕ) : ℕ := a * 10^9 + 2018 * 10^5 + b * 10^4 + 2019

-- Define the divisibility condition
def is_divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

-- State the theorem
theorem unique_divisibility_by_99 :
  ∃! (a b : ℕ), a < 10 ∧ b < 10 ∧ is_divisible_by_99 (N a b) :=
sorry

end NUMINAMATH_CALUDE_unique_divisibility_by_99_l635_63555


namespace NUMINAMATH_CALUDE_marble_probability_value_l635_63524

/-- The probability of having one white and one blue marble left when drawing
    marbles randomly from a bag containing 3 blue and 5 white marbles until 2 are left -/
def marble_probability : ℚ :=
  let total_marbles : ℕ := 8
  let blue_marbles : ℕ := 3
  let white_marbles : ℕ := 5
  let marbles_drawn : ℕ := 6
  let favorable_outcomes : ℕ := Nat.choose white_marbles white_marbles * Nat.choose blue_marbles (blue_marbles - 1)
  let total_outcomes : ℕ := Nat.choose total_marbles marbles_drawn
  (favorable_outcomes : ℚ) / total_outcomes

/-- Theorem stating that the probability of having one white and one blue marble left
    is equal to 3/28 -/
theorem marble_probability_value : marble_probability = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_value_l635_63524


namespace NUMINAMATH_CALUDE_mile_to_rod_l635_63578

-- Define the conversion factors
def mile_to_furlong : ℝ := 8
def furlong_to_pace : ℝ := 220
def pace_to_rod : ℝ := 0.2

-- Theorem statement
theorem mile_to_rod : 
  1 * mile_to_furlong * furlong_to_pace * pace_to_rod = 352 := by
  sorry

end NUMINAMATH_CALUDE_mile_to_rod_l635_63578


namespace NUMINAMATH_CALUDE_alice_bushes_l635_63506

/-- The number of bushes Alice needs to buy for her yard -/
def bushes_needed (sides : ℕ) (side_length : ℕ) (bush_length : ℕ) : ℕ :=
  (sides * side_length) / bush_length

/-- Theorem: Alice needs to buy 12 bushes -/
theorem alice_bushes :
  bushes_needed 3 16 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_alice_bushes_l635_63506


namespace NUMINAMATH_CALUDE_pizza_burger_overlap_l635_63573

theorem pizza_burger_overlap (total : ℕ) (pizza : ℕ) (burger : ℕ) 
  (h_total : total = 200)
  (h_pizza : pizza = 125)
  (h_burger : burger = 115) :
  pizza + burger - total = 40 := by
  sorry

end NUMINAMATH_CALUDE_pizza_burger_overlap_l635_63573


namespace NUMINAMATH_CALUDE_cds_per_rack_is_eight_l635_63596

/-- The number of racks a shelf can hold -/
def num_racks : ℕ := 4

/-- The total number of CDs a shelf can hold -/
def total_cds : ℕ := 32

/-- The number of CDs each rack can hold -/
def cds_per_rack : ℕ := total_cds / num_racks

theorem cds_per_rack_is_eight : cds_per_rack = 8 := by
  sorry

end NUMINAMATH_CALUDE_cds_per_rack_is_eight_l635_63596


namespace NUMINAMATH_CALUDE_square_partition_theorem_l635_63509

/-- A rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- Predicate indicating if one rectangle can be placed inside another (possibly with rotation) -/
def can_fit_inside (r1 r2 : Rectangle) : Prop :=
  (r1.a ≤ r2.a ∧ r1.b ≤ r2.b) ∨ (r1.a ≤ r2.b ∧ r1.b ≤ r2.a)

theorem square_partition_theorem (n : ℕ) (hn : n^2 ≥ 4) :
  ∃ (rectangles : Fin (n^2) → Rectangle),
    (∀ i j, i ≠ j → rectangles i ≠ rectangles j) →
    (∃ (chosen : Fin (2*n) → Fin (n^2)),
      ∀ i j, i < j → can_fit_inside (rectangles (chosen i)) (rectangles (chosen j))) :=
  sorry

end NUMINAMATH_CALUDE_square_partition_theorem_l635_63509


namespace NUMINAMATH_CALUDE_jerome_toy_car_ratio_l635_63536

/-- Proves that the ratio of toy cars Jerome bought this month to last month is 2:1 -/
theorem jerome_toy_car_ratio :
  let original_cars : ℕ := 25
  let cars_bought_last_month : ℕ := 5
  let total_cars_now : ℕ := 40
  let cars_bought_this_month : ℕ := total_cars_now - original_cars - cars_bought_last_month
  cars_bought_this_month / cars_bought_last_month = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerome_toy_car_ratio_l635_63536


namespace NUMINAMATH_CALUDE_maggie_fish_books_l635_63546

/-- The number of fish books Maggie bought -/
def fish_books : ℕ := sorry

/-- The total amount Maggie spent -/
def total_spent : ℕ := 170

/-- The number of plant books Maggie bought -/
def plant_books : ℕ := 9

/-- The number of science magazines Maggie bought -/
def science_magazines : ℕ := 10

/-- The cost of each book -/
def book_cost : ℕ := 15

/-- The cost of each magazine -/
def magazine_cost : ℕ := 2

theorem maggie_fish_books : 
  fish_books = 1 := by sorry

end NUMINAMATH_CALUDE_maggie_fish_books_l635_63546


namespace NUMINAMATH_CALUDE_vectors_not_coplanar_l635_63534

/-- Prove that the given vectors are not coplanar -/
theorem vectors_not_coplanar (a b c : ℝ × ℝ × ℝ) :
  a = (-7, 10, -5) →
  b = (0, -2, -1) →
  c = (-2, 4, -1) →
  ¬(∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_vectors_not_coplanar_l635_63534


namespace NUMINAMATH_CALUDE_vector_decomposition_l635_63513

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![3, 3, -1]
def p : Fin 3 → ℝ := ![3, 1, 0]
def q : Fin 3 → ℝ := ![-1, 2, 1]
def r : Fin 3 → ℝ := ![-1, 0, 2]

/-- Theorem: x can be decomposed as p + q - r -/
theorem vector_decomposition : x = p + q - r := by
  sorry

end NUMINAMATH_CALUDE_vector_decomposition_l635_63513


namespace NUMINAMATH_CALUDE_debby_water_bottles_l635_63543

theorem debby_water_bottles (initial_bottles : ℕ) (bottles_per_day : ℕ) (remaining_bottles : ℕ) 
  (h1 : initial_bottles = 301)
  (h2 : bottles_per_day = 144)
  (h3 : remaining_bottles = 157) :
  (initial_bottles - remaining_bottles) / bottles_per_day = 1 :=
sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l635_63543


namespace NUMINAMATH_CALUDE_tshirt_profit_calculation_l635_63568

-- Define the profit per jersey
def profit_per_jersey : ℕ := 5

-- Define the profit per t-shirt
def profit_per_tshirt : ℕ := 215

-- Define the number of t-shirts sold
def tshirts_sold : ℕ := 20

-- Define the number of jerseys sold
def jerseys_sold : ℕ := 64

-- Theorem to prove
theorem tshirt_profit_calculation :
  tshirts_sold * profit_per_tshirt = 4300 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_profit_calculation_l635_63568


namespace NUMINAMATH_CALUDE_building_residents_contradiction_l635_63532

theorem building_residents_contradiction (chess : ℕ) (arkhangelsk : ℕ) (airplane : ℕ)
  (chess_airplane : ℕ) (arkhangelsk_airplane : ℕ) (chess_arkhangelsk : ℕ)
  (chess_arkhangelsk_airplane : ℕ) :
  chess = 25 →
  arkhangelsk = 30 →
  airplane = 28 →
  chess_airplane = 18 →
  arkhangelsk_airplane = 17 →
  chess_arkhangelsk = 16 →
  chess_arkhangelsk_airplane = 15 →
  chess + arkhangelsk + airplane - chess_arkhangelsk - chess_airplane - arkhangelsk_airplane + chess_arkhangelsk_airplane > 45 :=
by sorry

end NUMINAMATH_CALUDE_building_residents_contradiction_l635_63532


namespace NUMINAMATH_CALUDE_danny_share_l635_63515

/-- Represents the share of money each person receives -/
structure Share :=
  (amount : ℝ)
  (removed : ℝ)

/-- The problem setup -/
def problem_setup :=
  (total : ℝ) →
  (alice : Share) →
  (bond : Share) →
  (charlie : Share) →
  (danny : Share) →
  Prop

/-- The conditions of the problem -/
def conditions (total : ℝ) (alice bond charlie danny : Share) : Prop :=
  total = 2210 ∧
  alice.removed = 30 ∧
  bond.removed = 50 ∧
  charlie.removed = 40 ∧
  danny.removed = 2 * charlie.removed ∧
  (alice.amount - alice.removed) / (bond.amount - bond.removed) = 11 / 18 ∧
  (alice.amount - alice.removed) / (charlie.amount - charlie.removed) = 11 / 24 ∧
  (alice.amount - alice.removed) / (danny.amount - danny.removed) = 11 / 32 ∧
  alice.amount + bond.amount + charlie.amount + danny.amount = total

/-- The theorem to prove -/
theorem danny_share (total : ℝ) (alice bond charlie danny : Share) :
  conditions total alice bond charlie danny →
  danny.amount = 916.80 :=
sorry

end NUMINAMATH_CALUDE_danny_share_l635_63515


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l635_63511

theorem real_part_of_complex_product : ∃ (z : ℂ), z = (1 + Complex.I) * (2 - Complex.I) ∧ z.re = 3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l635_63511


namespace NUMINAMATH_CALUDE_sequence_divisibility_implies_zero_l635_63561

theorem sequence_divisibility_implies_zero (x : ℕ → ℤ) :
  (∀ i j : ℕ, i ≠ j → (i * j : ℤ) ∣ (x i + x j)) →
  ∀ n : ℕ, x n = 0 :=
by sorry

end NUMINAMATH_CALUDE_sequence_divisibility_implies_zero_l635_63561


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l635_63594

/-- Given line equation -/
def given_line (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0

/-- Candidate line equation -/
def candidate_line (x y : ℝ) : Prop := 3 * x + 2 * y - 4 = 0

/-- Point that the candidate line should pass through -/
def point : ℝ × ℝ := (2, -1)

theorem perpendicular_line_through_point :
  (candidate_line point.1 point.2) ∧ 
  (∀ (x y : ℝ), given_line x y → 
    (3 * 2 + 2 * (-3) = 0)) := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l635_63594


namespace NUMINAMATH_CALUDE_function_symmetry_l635_63570

theorem function_symmetry (f : ℝ → ℝ) (x : ℝ) : f (x - 1) = f (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l635_63570


namespace NUMINAMATH_CALUDE_towers_count_l635_63507

/-- Represents the number of cubes of each color -/
structure CubeSet where
  yellow : Nat
  purple : Nat
  orange : Nat

/-- Calculates the number of different towers that can be built -/
def countTowers (cubes : CubeSet) (towerHeight : Nat) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem towers_count (cubes : CubeSet) (h : cubes = { yellow := 3, purple := 3, orange := 2 }) :
  countTowers cubes 6 = 350 := by
  sorry

end NUMINAMATH_CALUDE_towers_count_l635_63507


namespace NUMINAMATH_CALUDE_sqrt_n_plus_9_equals_25_l635_63567

theorem sqrt_n_plus_9_equals_25 (n : ℝ) : Real.sqrt (n + 9) = 25 → n = 616 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_n_plus_9_equals_25_l635_63567


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l635_63518

/-- The number of diagonals in a convex n-gon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: A convex dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l635_63518


namespace NUMINAMATH_CALUDE_train_length_problem_l635_63581

/-- Given a train traveling at constant speed through a tunnel and over a bridge,
    prove that the length of the train is 200m. -/
theorem train_length_problem (tunnel_length : ℝ) (tunnel_time : ℝ) (bridge_length : ℝ) (bridge_time : ℝ)
    (h1 : tunnel_length = 860)
    (h2 : tunnel_time = 22)
    (h3 : bridge_length = 790)
    (h4 : bridge_time = 33)
    (h5 : (bridge_length + x) / bridge_time = (tunnel_length - x) / tunnel_time) :
    x = 200 := by
  sorry

#check train_length_problem

end NUMINAMATH_CALUDE_train_length_problem_l635_63581


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l635_63585

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 18

/-- The number of chocolate bars in each small box -/
def bars_per_small_box : ℕ := 28

/-- The total number of chocolate bars in the large box -/
def total_chocolate_bars : ℕ := num_small_boxes * bars_per_small_box

theorem chocolate_bars_count : total_chocolate_bars = 504 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l635_63585


namespace NUMINAMATH_CALUDE_hotel_flat_fee_l635_63531

/-- Given a hotel's pricing structure and two customers' stays, calculate the flat fee for the first night. -/
theorem hotel_flat_fee (linda_total linda_nights bob_total bob_nights : ℕ) 
  (h1 : linda_total = 205)
  (h2 : linda_nights = 4)
  (h3 : bob_total = 350)
  (h4 : bob_nights = 7) :
  ∃ (flat_fee nightly_rate : ℕ),
    flat_fee + (linda_nights - 1) * nightly_rate = linda_total ∧
    flat_fee + (bob_nights - 1) * nightly_rate = bob_total ∧
    flat_fee = 60 := by
  sorry

#check hotel_flat_fee

end NUMINAMATH_CALUDE_hotel_flat_fee_l635_63531


namespace NUMINAMATH_CALUDE_connecting_point_on_line_connecting_point_on_line_x_plus_1_connecting_point_area_and_distance_l635_63530

-- Define the concept of a "connecting point"
def is_connecting_point (P Q : ℝ × ℝ) : Prop :=
  Q.1 = P.1 ∧ 
  ((P.1 ≥ 0 ∧ Q.2 = P.2) ∨ (P.1 < 0 ∧ Q.2 = -P.2))

-- Part 1
theorem connecting_point_on_line (k : ℝ) (A A' : ℝ × ℝ) :
  k ≠ 0 →
  A.2 = k * A.1 →
  is_connecting_point A A' →
  A' = (-2, -6) →
  k = -3 :=
sorry

-- Part 2
theorem connecting_point_on_line_x_plus_1 (m : ℝ) (B B' : ℝ × ℝ) :
  B.2 = B.1 + 1 →
  is_connecting_point B B' →
  B' = (m, 2) →
  (m ≥ 0 → B = (1, 2)) ∧
  (m < 0 → B = (-3, -2)) :=
sorry

-- Part 3
theorem connecting_point_area_and_distance (P C C' : ℝ × ℝ) :
  P = (1, 0) →
  C.2 = -2 * C.1 + 2 →
  is_connecting_point C C' →
  abs ((P.1 - C.1) * (C'.2 - C.2)) / 2 = 18 →
  Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) = 3 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_connecting_point_on_line_connecting_point_on_line_x_plus_1_connecting_point_area_and_distance_l635_63530


namespace NUMINAMATH_CALUDE_workshop_workers_l635_63545

theorem workshop_workers (total_average : ℝ) (tech_count : ℕ) (tech_average : ℝ) (nontech_average : ℝ) : 
  total_average = 9500 → 
  tech_count = 7 → 
  tech_average = 12000 → 
  nontech_average = 6000 → 
  ∃ (total_workers : ℕ), total_workers = 12 ∧ 
    (total_workers : ℝ) * total_average = 
      (tech_count : ℝ) * tech_average + 
      ((total_workers - tech_count) : ℝ) * nontech_average :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l635_63545


namespace NUMINAMATH_CALUDE_chocolate_factory_order_completion_l635_63569

/-- Represents the number of days required to complete an order of candies. -/
def days_to_complete_order (candies_per_hour : ℕ) (hours_per_day : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies / candies_per_hour + hours_per_day - 1) / hours_per_day

/-- Theorem stating that it takes 8 days to complete the order under given conditions. -/
theorem chocolate_factory_order_completion :
  days_to_complete_order 50 10 4000 = 8 := by
  sorry

#eval days_to_complete_order 50 10 4000

end NUMINAMATH_CALUDE_chocolate_factory_order_completion_l635_63569


namespace NUMINAMATH_CALUDE_complement_of_intersection_union_condition_l635_63504

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

-- Theorem 1: Complement of intersection
theorem complement_of_intersection :
  (Aᶜ ∪ Bᶜ : Set ℝ) = {x | x < 2 ∨ x ≥ 3} := by sorry

-- Theorem 2: Condition for B ∪ C = C
theorem union_condition (a : ℝ) :
  B ∪ C a = C a → a ≥ -4 := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_union_condition_l635_63504


namespace NUMINAMATH_CALUDE_sine_shift_to_cosine_l635_63525

open Real

theorem sine_shift_to_cosine (x : ℝ) :
  let f (t : ℝ) := sin (2 * t + π / 6)
  let g (t : ℝ) := f (t + π / 6)
  g x = cos (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_sine_shift_to_cosine_l635_63525


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l635_63538

/-- Represents a rectangular prism -/
structure RectangularPrism where
  -- We don't need to define any specific properties here

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (rp : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- Theorem: The sum of edges, vertices, and faces of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  num_edges rp + num_vertices rp + num_faces rp = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l635_63538


namespace NUMINAMATH_CALUDE_no_three_points_property_H_l635_63557

/-- Definition of the ellipse C -/
def C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Definition of property H for a line intersecting the ellipse C -/
def property_H (A B M : ℝ × ℝ) : Prop :=
  C A.1 A.2 ∧ C B.1 B.2 ∧ C M.1 M.2 ∧
  M.1 = 3/5 * A.1 + 4/5 * B.1 ∧
  M.2 = 3/5 * A.2 + 4/5 * B.2

/-- Main theorem: No three distinct points on C form lines all having property H -/
theorem no_three_points_property_H :
  ¬ ∃ (P Q R : ℝ × ℝ),
    C P.1 P.2 ∧ C Q.1 Q.2 ∧ C R.1 R.2 ∧
    P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧
    (∃ M₁, property_H P Q M₁) ∧
    (∃ M₂, property_H Q R M₂) ∧
    (∃ M₃, property_H R P M₃) :=
sorry

end NUMINAMATH_CALUDE_no_three_points_property_H_l635_63557


namespace NUMINAMATH_CALUDE_number_difference_l635_63526

theorem number_difference (x y : ℚ) 
  (sum_eq : x + y = 40)
  (triple_minus_quad : 3 * y - 4 * x = 10) :
  abs (y - x) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l635_63526


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l635_63584

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l635_63584


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l635_63574

/-- Given that x and y are inversely proportional, and when their sum is 50, x is three times y,
    prove that y = -39.0625 when x = -12 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) : 
  (∀ x y, x * y = k) →  -- x and y are inversely proportional
  (∃ x y, x + y = 50 ∧ x = 3 * y) →  -- when their sum is 50, x is three times y
  (x = -12 → y = -39.0625) :=  -- prove that y = -39.0625 when x = -12
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l635_63574


namespace NUMINAMATH_CALUDE_root_range_implies_a_range_l635_63588

theorem root_range_implies_a_range :
  ∀ a : ℝ,
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ x^2 - 4*x + 3*a^2 - 2 = 0) →
  a ∈ Set.Icc (-Real.sqrt (5/3)) (Real.sqrt (5/3)) :=
by sorry

end NUMINAMATH_CALUDE_root_range_implies_a_range_l635_63588


namespace NUMINAMATH_CALUDE_business_value_calculation_l635_63592

theorem business_value_calculation (man_share : ℚ) (sold_fraction : ℚ) (sale_price : ℚ) : 
  man_share = 2/3 → 
  sold_fraction = 3/4 → 
  sale_price = 6500 → 
  (sale_price / sold_fraction) / man_share = 39000 :=
by
  sorry

end NUMINAMATH_CALUDE_business_value_calculation_l635_63592


namespace NUMINAMATH_CALUDE_bridget_sarah_cents_difference_bridget_sarah_solution_l635_63541

theorem bridget_sarah_cents_difference : ℕ → ℕ → ℕ → Prop :=
  fun total sarah_cents difference =>
    total = 300 ∧
    sarah_cents = 125 ∧
    difference = total - 2 * sarah_cents

theorem bridget_sarah_solution :
  ∃ (difference : ℕ), bridget_sarah_cents_difference 300 125 difference ∧ difference = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_bridget_sarah_cents_difference_bridget_sarah_solution_l635_63541


namespace NUMINAMATH_CALUDE_remainder_zero_l635_63589

def nines : ℕ := 10^20089 - 1
def threes : ℕ := 3 * (10^20083 - 1) / 9

theorem remainder_zero :
  (nines^2007 - threes^2007) % 11 = 0 := by sorry

end NUMINAMATH_CALUDE_remainder_zero_l635_63589


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l635_63595

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧ 
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l635_63595


namespace NUMINAMATH_CALUDE_trapezoid_ratio_satisfies_equation_l635_63566

/-- Represents a trapezoid with a point inside dividing it into four triangles -/
structure TrapezoidWithPoint where
  AB : ℝ
  CD : ℝ
  area_PCD : ℝ
  area_PAD : ℝ
  area_PBC : ℝ
  area_PAB : ℝ
  h_AB_gt_CD : AB > CD
  h_areas : area_PCD = 3 ∧ area_PAD = 5 ∧ area_PBC = 6 ∧ area_PAB = 8

/-- The ratio of AB to CD satisfies a specific quadratic equation -/
theorem trapezoid_ratio_satisfies_equation (t : TrapezoidWithPoint) :
  let k := t.AB / t.CD
  k^2 + (22/6) * k + 16/6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_ratio_satisfies_equation_l635_63566


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l635_63559

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define the perimeter of an equilateral triangle
def perimeter (t : EquilateralTriangle) : ℝ := 3 * t.side_length

-- Theorem statement
theorem equilateral_triangle_side_length 
  (t : EquilateralTriangle) 
  (h : perimeter t = 18) : 
  t.side_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l635_63559


namespace NUMINAMATH_CALUDE_library_books_end_of_month_l635_63572

theorem library_books_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) 
  (h1 : initial_books = 300)
  (h2 : loaned_books = 160)
  (h3 : return_rate = 65 / 100) :
  initial_books - loaned_books + (return_rate * loaned_books).floor = 244 :=
by sorry

end NUMINAMATH_CALUDE_library_books_end_of_month_l635_63572


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l635_63597

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposite and their y-coordinates are equal -/
def symmetric_wrt_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

/-- Given two points M(a, 3) and N(5, b) that are symmetric with respect to the y-axis,
    prove that a - b = -8 -/
theorem symmetric_points_difference (a b : ℝ) 
    (h : symmetric_wrt_y_axis (a, 3) (5, b)) : a - b = -8 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l635_63597


namespace NUMINAMATH_CALUDE_power_function_through_point_l635_63537

/-- A power function that passes through the point (2, √2) -/
def f (x : ℝ) : ℝ := x^(1/2)

/-- Theorem: For the power function f(x) that passes through (2, √2), f(5) = √5 -/
theorem power_function_through_point (h : f 2 = Real.sqrt 2) : f 5 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l635_63537


namespace NUMINAMATH_CALUDE_cube_face_projections_l635_63510

/-- Given three faces of a unit cube sharing a common vertex, if their projections onto a fixed plane
have areas in the ratio 6:10:15, then the sum of these areas is 31/19. -/
theorem cube_face_projections (x y z : ℝ) : 
  x > 0 ∧ y > 0 ∧ z > 0 →  -- Ensure positive areas
  x^2 + y^2 + z^2 = 1 →  -- Sum of squares of projection areas equals 1
  x / 6 = y / 10 ∧ y / 10 = z / 15 →  -- Ratio condition
  x + y + z = 31 / 19 := by
sorry

end NUMINAMATH_CALUDE_cube_face_projections_l635_63510


namespace NUMINAMATH_CALUDE_class_project_total_l635_63580

/-- Calculates the total amount gathered for a class project with discounts and fees -/
theorem class_project_total (total_students : ℕ) (full_price : ℚ) 
  (full_paying : ℕ) (high_merit : ℕ) (financial_needs : ℕ) (special_discount : ℕ)
  (high_merit_discount : ℚ) (financial_needs_discount : ℚ) (special_discount_rate : ℚ)
  (admin_fee : ℚ) :
  total_students = 35 →
  full_price = 50 →
  full_paying = 20 →
  high_merit = 5 →
  financial_needs = 7 →
  special_discount = 3 →
  high_merit_discount = 25 / 100 →
  financial_needs_discount = 1 / 2 →
  special_discount_rate = 10 / 100 →
  admin_fee = 100 →
  (full_paying * full_price + 
   high_merit * (full_price * (1 - high_merit_discount)) +
   financial_needs * (full_price * financial_needs_discount) +
   special_discount * (full_price * (1 - special_discount_rate))) - admin_fee = 1397.5 := by
  sorry


end NUMINAMATH_CALUDE_class_project_total_l635_63580


namespace NUMINAMATH_CALUDE_apple_problem_l635_63551

/-- Proves that given the conditions of the apple problem, each child originally had 15 apples -/
theorem apple_problem (num_children : Nat) (apples_eaten : Nat) (apples_sold : Nat) (apples_left : Nat) :
  num_children = 5 →
  apples_eaten = 8 →
  apples_sold = 7 →
  apples_left = 60 →
  ∃ x : Nat, num_children * x - apples_eaten - apples_sold = apples_left ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_problem_l635_63551


namespace NUMINAMATH_CALUDE_intersection_sum_l635_63521

/-- Given two lines y = nx + 3 and y = 5x + c that intersect at (4, 11), prove that n + c = -7 -/
theorem intersection_sum (n c : ℝ) : 
  (4 * n + 3 = 11) → (5 * 4 + c = 11) → n + c = -7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l635_63521


namespace NUMINAMATH_CALUDE_machine_total_time_l635_63544

/-- The total time a machine worked, including downtime, given its production rates and downtime -/
theorem machine_total_time
  (time_A : ℕ) (shirts_A : ℕ) (time_B : ℕ) (shirts_B : ℕ) (downtime : ℕ)
  (h_A : time_A = 75 ∧ shirts_A = 13)
  (h_B : time_B = 5 ∧ shirts_B = 3)
  (h_downtime : downtime = 120) :
  time_A + time_B + downtime = 200 := by
  sorry


end NUMINAMATH_CALUDE_machine_total_time_l635_63544


namespace NUMINAMATH_CALUDE_coin_ratio_l635_63528

theorem coin_ratio (pennies nickels dimes quarters : ℕ) 
  (h1 : nickels = 5 * dimes)
  (h2 : pennies = 3 * nickels)
  (h3 : pennies = 120)
  (h4 : pennies + 5 * nickels + 10 * dimes + 25 * quarters = 800) :
  quarters = 2 * dimes := by
  sorry

end NUMINAMATH_CALUDE_coin_ratio_l635_63528


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l635_63556

def sequence_a (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => (sequence_a a₁ n) ^ 2

def monotonically_increasing (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, f (n + 1) > f n

theorem sufficient_but_not_necessary :
  (∀ a₁ : ℝ, a₁ = 2 → monotonically_increasing (sequence_a a₁)) ∧
  (∃ a₁ : ℝ, a₁ ≠ 2 ∧ monotonically_increasing (sequence_a a₁)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l635_63556


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l635_63565

theorem cow_chicken_problem (c h : ℕ) : 
  4 * c + 2 * h = 2 * (c + h) + 16 → c = 8 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l635_63565


namespace NUMINAMATH_CALUDE_evan_future_books_l635_63502

/-- Calculates the number of books Evan will have in 5 years -/
def books_in_five_years (books_two_years_ago : ℕ) : ℕ :=
  let current_books := books_two_years_ago - 40
  5 * current_books + 60

/-- Proves that Evan will have 860 books in 5 years -/
theorem evan_future_books :
  books_in_five_years 200 = 860 := by
  sorry

#eval books_in_five_years 200

end NUMINAMATH_CALUDE_evan_future_books_l635_63502


namespace NUMINAMATH_CALUDE_john_lift_weight_l635_63571

/-- Calculates the final weight John can lift after training and using a magical bracer -/
def final_lift_weight (initial_weight : ℕ) (weight_increase : ℕ) (bracer_multiplier : ℕ) : ℕ :=
  let after_training := initial_weight + weight_increase
  let bracer_increase := after_training * bracer_multiplier
  after_training + bracer_increase

/-- Proves that John can lift 2800 pounds after training and using the magical bracer -/
theorem john_lift_weight :
  final_lift_weight 135 265 6 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_john_lift_weight_l635_63571

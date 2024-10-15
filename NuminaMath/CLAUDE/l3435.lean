import Mathlib

namespace NUMINAMATH_CALUDE_square_area_with_circles_l3435_343533

/-- The area of a square containing a 3x3 grid of circles with radius 3 inches -/
theorem square_area_with_circles (r : ℝ) (h : r = 3) : 
  (3 * (2 * r))^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_circles_l3435_343533


namespace NUMINAMATH_CALUDE_q_plus_r_at_one_eq_neg_47_l3435_343510

/-- The polynomial f(x) = 3x^5 + 4x^4 - 5x^3 + 2x^2 + x + 6 -/
def f (x : ℝ) : ℝ := 3*x^5 + 4*x^4 - 5*x^3 + 2*x^2 + x + 6

/-- The polynomial d(x) = x^3 + 2x^2 - x - 3 -/
def d (x : ℝ) : ℝ := x^3 + 2*x^2 - x - 3

/-- The existence of polynomials q and r satisfying the division algorithm -/
axiom exists_q_r : ∃ (q r : ℝ → ℝ), ∀ x, f x = q x * d x + r x

/-- The degree of r is less than the degree of d -/
axiom deg_r_lt_deg_d : sorry -- We can't easily express polynomial degrees in this simple setup

theorem q_plus_r_at_one_eq_neg_47 : 
  ∃ (q r : ℝ → ℝ), (∀ x, f x = q x * d x + r x) ∧ q 1 + r 1 = -47 := by
  sorry

end NUMINAMATH_CALUDE_q_plus_r_at_one_eq_neg_47_l3435_343510


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l3435_343576

def anna_lap_time : ℕ := 4
def stephanie_lap_time : ℕ := 7
def james_lap_time : ℕ := 6

theorem earliest_meeting_time :
  let meeting_time := lcm (lcm anna_lap_time stephanie_lap_time) james_lap_time
  meeting_time = 84 := by
  sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l3435_343576


namespace NUMINAMATH_CALUDE_isosceles_trajectory_equation_l3435_343558

/-- An isosceles triangle ABC with vertices A(3,20) and B(3,5) -/
structure IsoscelesTriangle where
  C : ℝ × ℝ
  isIsosceles : (C.1 - 3)^2 + (C.2 - 20)^2 = (3 - 3)^2 + (5 - 20)^2
  notCollinear : C.1 ≠ 3

/-- The trajectory equation of point C in an isosceles triangle ABC -/
def trajectoryEquation (t : IsoscelesTriangle) : Prop :=
  (t.C.1 - 3)^2 + (t.C.2 - 20)^2 = 225

/-- Theorem: The trajectory equation holds for any isosceles triangle satisfying the given conditions -/
theorem isosceles_trajectory_equation (t : IsoscelesTriangle) : trajectoryEquation t := by
  sorry


end NUMINAMATH_CALUDE_isosceles_trajectory_equation_l3435_343558


namespace NUMINAMATH_CALUDE_mother_age_is_55_l3435_343565

/-- The mother's age in years -/
def mother_age : ℕ := 55

/-- The daughter's age in years -/
def daughter_age : ℕ := mother_age - 27

theorem mother_age_is_55 :
  (mother_age = daughter_age + 27) ∧
  (mother_age - 1 = 2 * (daughter_age - 1)) →
  mother_age = 55 := by
  sorry

#check mother_age_is_55

end NUMINAMATH_CALUDE_mother_age_is_55_l3435_343565


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l3435_343562

theorem largest_prime_divisor :
  ∃ (p : ℕ), Nat.Prime p ∧ 
    p ∣ (2^(p+1) + 3^(p+1) + 5^(p+1) + 7^(p+1)) ∧
    ∀ (q : ℕ), Nat.Prime q → q ∣ (2^(q+1) + 3^(q+1) + 5^(q+1) + 7^(q+1)) → q ≤ p :=
by
  use 29
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l3435_343562


namespace NUMINAMATH_CALUDE_sons_age_l3435_343516

theorem sons_age (son_age woman_age : ℕ) : 
  woman_age = 2 * son_age + 3 →
  woman_age + son_age = 84 →
  son_age = 27 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3435_343516


namespace NUMINAMATH_CALUDE_chord_line_equation_l3435_343504

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a circle and a point that is the midpoint of a chord, 
    return the line containing that chord -/
def chordLine (c : Circle) (p : ℝ × ℝ) : Line :=
  sorry

theorem chord_line_equation (c : Circle) (p : ℝ × ℝ) :
  let circle : Circle := { center := (3, 0), radius := 3 }
  let midpoint : ℝ × ℝ := (4, 2)
  let line := chordLine circle midpoint
  line.a = 1 ∧ line.b = 2 ∧ line.c = -8 := by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l3435_343504


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_third_l3435_343534

-- Define the repeating decimal 0.333...
def repeating_third : ℚ := 1/3

-- Theorem statement
theorem reciprocal_of_repeating_third :
  (repeating_third⁻¹ : ℚ) = 3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_third_l3435_343534


namespace NUMINAMATH_CALUDE_captainSelection_l3435_343556

/-- The number of ways to select a captain and a vice-captain from a team of 11 people -/
def selectCaptains : ℕ :=
  11 * 10

/-- Theorem stating that the number of ways to select a captain and a vice-captain
    from a team of 11 people is equal to 110 -/
theorem captainSelection : selectCaptains = 110 := by
  sorry

end NUMINAMATH_CALUDE_captainSelection_l3435_343556


namespace NUMINAMATH_CALUDE_binary_sequence_equiv_powerset_nat_l3435_343554

/-- The type of infinite binary sequences -/
def BinarySequence := ℕ → Bool

/-- The theorem stating the equinumerosity of binary sequences and subsets of naturals -/
theorem binary_sequence_equiv_powerset_nat :
  ∃ (f : BinarySequence → Set ℕ), Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_binary_sequence_equiv_powerset_nat_l3435_343554


namespace NUMINAMATH_CALUDE_add_518_276_base_12_l3435_343596

/-- Addition in base 12 --/
def add_base_12 (a b : ℕ) : ℕ :=
  sorry

/-- Conversion from base 12 to base 10 --/
def base_12_to_10 (n : ℕ) : ℕ :=
  sorry

/-- Conversion from base 10 to base 12 --/
def base_10_to_12 (n : ℕ) : ℕ :=
  sorry

theorem add_518_276_base_12 :
  add_base_12 (base_10_to_12 518) (base_10_to_12 276) = base_10_to_12 792 :=
sorry

end NUMINAMATH_CALUDE_add_518_276_base_12_l3435_343596


namespace NUMINAMATH_CALUDE_trig_functions_right_triangle_l3435_343521

/-- Define trigonometric functions for a right-angled triangle --/
theorem trig_functions_right_triangle 
  (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ (A : ℝ), 
    Real.sin A = a / c ∧ 
    Real.cos A = b / c ∧ 
    Real.tan A = a / b :=
sorry

end NUMINAMATH_CALUDE_trig_functions_right_triangle_l3435_343521


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l3435_343557

/-- A rectangular prism with given conditions -/
structure RectangularPrism where
  length : ℝ
  breadth : ℝ
  height : ℝ
  length_breadth_diff : length - breadth = 23
  perimeter : 2 * length + 2 * breadth = 166

/-- The volume of a rectangular prism is 1590h cubic meters -/
theorem rectangular_prism_volume (prism : RectangularPrism) : 
  prism.length * prism.breadth * prism.height = 1590 * prism.height := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l3435_343557


namespace NUMINAMATH_CALUDE_infinitely_many_good_numbers_good_not_divisible_by_seven_l3435_343500

/-- A natural number n is good if there exist natural numbers a and b
    such that a + b = n and ab | n^2 + n + 1 -/
def is_good (n : ℕ) : Prop :=
  ∃ a b : ℕ, a + b = n ∧ (n^2 + n + 1) % (a * b) = 0

theorem infinitely_many_good_numbers :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ is_good n :=
sorry

theorem good_not_divisible_by_seven :
  ∀ n : ℕ, is_good n → ¬(7 ∣ n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_good_numbers_good_not_divisible_by_seven_l3435_343500


namespace NUMINAMATH_CALUDE_average_letters_per_day_l3435_343501

def letters_monday : ℕ := 7
def letters_tuesday : ℕ := 10
def letters_wednesday : ℕ := 3
def letters_thursday : ℕ := 5
def letters_friday : ℕ := 12
def total_days : ℕ := 5

theorem average_letters_per_day :
  (letters_monday + letters_tuesday + letters_wednesday + letters_thursday + letters_friday : ℚ) / total_days = 37 / 5 := by
  sorry

end NUMINAMATH_CALUDE_average_letters_per_day_l3435_343501


namespace NUMINAMATH_CALUDE_linear_function_k_value_l3435_343553

/-- Proves that for the linear function y = kx + 3 passing through the point (2, 5), the value of k is 1. -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 3) → -- Condition 1: The function is y = kx + 3
  (5 : ℝ) = k * 2 + 3 →        -- Condition 2: The function passes through the point (2, 5)
  k = 1 :=                     -- Conclusion: The value of k is 1
by sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l3435_343553


namespace NUMINAMATH_CALUDE_circle_line_distance_l3435_343588

/-- Represents a circle in 2D space --/
structure Circle where
  center : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  equation : ℝ → ℝ → Prop

/-- Calculates the distance from a point to a line --/
def distancePointToLine (point : ℝ × ℝ) (line : Line) : ℝ := sorry

/-- The main theorem --/
theorem circle_line_distance (c : Circle) (l : Line) :
  c.equation = fun x y => x^2 + y^2 - 2*x - 8*y + 1 = 0 →
  l.equation = fun x y => l.a*x - y + 1 = 0 →
  c.center = (1, 4) →
  distancePointToLine c.center l = 1 →
  l.a = 4/3 := by sorry

end NUMINAMATH_CALUDE_circle_line_distance_l3435_343588


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l3435_343529

theorem polynomial_product_expansion (x : ℝ) :
  (x^3 - 3*x^2 + 3*x - 1) * (x^2 + 3*x + 3) = x^5 - 3*x^2 + 6*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l3435_343529


namespace NUMINAMATH_CALUDE_perpendicular_tangents_theorem_l3435_343594

noncomputable def f (x : ℝ) : ℝ := abs x / Real.exp x

def is_perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

theorem perpendicular_tangents_theorem (x₀ : ℝ) (m : ℤ) :
  x₀ > 0 ∧
  x₀ ∈ Set.Ioo (m / 4 : ℝ) ((m + 1) / 4 : ℝ) ∧
  is_perpendicular ((deriv f) (-1)) ((deriv f) x₀) →
  m = 2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_theorem_l3435_343594


namespace NUMINAMATH_CALUDE_rectangle_division_even_triangles_l3435_343578

theorem rectangle_division_even_triangles 
  (a b c d : ℕ) 
  (h_rect : a > 0 ∧ b > 0) 
  (h_tri : c > 0 ∧ d > 0) 
  (h_div : (a * b) % (c * d / 2) = 0) :
  ∃ k : ℕ, k % 2 = 0 ∧ k * (c * d / 2) = a * b :=
sorry

end NUMINAMATH_CALUDE_rectangle_division_even_triangles_l3435_343578


namespace NUMINAMATH_CALUDE_key_sequence_produces_desired_output_l3435_343525

/-- Represents the mapping of keys to displayed letters on the magical keyboard. -/
def keyboard_mapping : Char → Char
| 'Q' => 'A'
| 'S' => 'D'
| 'D' => 'S'
| 'J' => 'H'
| 'K' => 'O'
| 'L' => 'P'
| 'R' => 'E'
| 'N' => 'M'
| 'Y' => 'T'
| c => c  -- For all other characters, map to themselves

/-- The sequence of key presses -/
def key_sequence : List Char := ['J', 'K', 'L', 'R', 'N', 'Q', 'Y', 'J']

/-- The desired display output -/
def desired_output : List Char := ['H', 'O', 'P', 'E', 'M', 'A', 'T', 'H']

/-- Theorem stating that the key sequence produces the desired output -/
theorem key_sequence_produces_desired_output :
  key_sequence.map keyboard_mapping = desired_output := by
  sorry

#eval key_sequence.map keyboard_mapping

end NUMINAMATH_CALUDE_key_sequence_produces_desired_output_l3435_343525


namespace NUMINAMATH_CALUDE_perfect_square_triples_l3435_343571

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def satisfies_condition (a b c : ℕ) : Prop :=
  is_perfect_square (2^a + 2^b + 2^c + 3)

theorem perfect_square_triples :
  ∀ a b c : ℕ, satisfies_condition a b c ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 3 ∧ b = 2 ∧ c = 1) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_triples_l3435_343571


namespace NUMINAMATH_CALUDE_product_sum_digits_base7_l3435_343550

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Sums the digits of a number in base-7 --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem product_sum_digits_base7 :
  let a := 35
  let b := 52
  sumDigitsBase7 (toBase7 (toBase10 a * toBase10 b)) = 16 := by sorry

end NUMINAMATH_CALUDE_product_sum_digits_base7_l3435_343550


namespace NUMINAMATH_CALUDE_perception_arrangements_l3435_343542

def word_length : ℕ := 10

def repeating_letters : List (Char × ℕ) := [('E', 2), ('P', 2), ('I', 2)]

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem perception_arrangements : 
  (factorial word_length) / ((repeating_letters.map (λ (_, count) => factorial count)).prod) = 453600 := by
  sorry

end NUMINAMATH_CALUDE_perception_arrangements_l3435_343542


namespace NUMINAMATH_CALUDE_tan_difference_of_angles_l3435_343599

theorem tan_difference_of_angles (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α - Real.sin β = -1/2) (h4 : Real.cos α - Real.cos β = 1/2) :
  Real.tan (α - β) = -Real.sqrt 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_of_angles_l3435_343599


namespace NUMINAMATH_CALUDE_not_perfect_square_l3435_343537

theorem not_perfect_square (n : ℕ) (h : n > 1) : ¬∃ (m : ℕ), 9*n^2 - 9*n + 9 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l3435_343537


namespace NUMINAMATH_CALUDE_wilted_flowers_count_l3435_343561

def initial_flowers : ℕ := 88
def flowers_per_bouquet : ℕ := 5
def bouquets_made : ℕ := 8

theorem wilted_flowers_count : 
  initial_flowers - (flowers_per_bouquet * bouquets_made) = 48 := by
  sorry

end NUMINAMATH_CALUDE_wilted_flowers_count_l3435_343561


namespace NUMINAMATH_CALUDE_range_of_a_l3435_343573

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| - |x + 2| ≤ 3) → -5 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3435_343573


namespace NUMINAMATH_CALUDE_mcgregor_books_finished_l3435_343569

theorem mcgregor_books_finished (total_books : ℕ) (floyd_finished : ℕ) (books_left : ℕ) : 
  total_books = 89 → floyd_finished = 32 → books_left = 23 → 
  total_books - floyd_finished - books_left = 34 := by
sorry

end NUMINAMATH_CALUDE_mcgregor_books_finished_l3435_343569


namespace NUMINAMATH_CALUDE_pasta_sauce_free_percentage_l3435_343580

/-- Given a pasta dish weighing 200 grams with 50 grams of sauce,
    prove that 75% of the dish is sauce-free. -/
theorem pasta_sauce_free_percentage
  (total_weight : ℝ)
  (sauce_weight : ℝ)
  (h_total : total_weight = 200)
  (h_sauce : sauce_weight = 50) :
  (total_weight - sauce_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_pasta_sauce_free_percentage_l3435_343580


namespace NUMINAMATH_CALUDE_sum_reciprocal_squares_l3435_343535

theorem sum_reciprocal_squares (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squares_l3435_343535


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3435_343551

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3435_343551


namespace NUMINAMATH_CALUDE_total_credits_proof_l3435_343579

theorem total_credits_proof (emily_credits : ℕ) 
  (h1 : emily_credits = 20)
  (h2 : ∃ aria_credits : ℕ, aria_credits = 2 * emily_credits)
  (h3 : ∃ spencer_credits : ℕ, spencer_credits = emily_credits / 2)
  (h4 : ∃ hannah_credits : ℕ, hannah_credits = 3 * (emily_credits / 2)) :
  2 * (emily_credits + 2 * emily_credits + emily_credits / 2 + 3 * (emily_credits / 2)) = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_credits_proof_l3435_343579


namespace NUMINAMATH_CALUDE_thabo_hardcover_nonfiction_l3435_343517

/-- The number of books Thabo owns -/
def total_books : ℕ := 280

/-- The number of paperback nonfiction books -/
def paperback_nonfiction (hardcover_nonfiction : ℕ) : ℕ := hardcover_nonfiction + 20

/-- The number of paperback fiction books -/
def paperback_fiction (hardcover_nonfiction : ℕ) : ℕ := 2 * (paperback_nonfiction hardcover_nonfiction)

/-- Theorem stating the number of hardcover nonfiction books Thabo owns -/
theorem thabo_hardcover_nonfiction :
  ∃ (hardcover_nonfiction : ℕ),
    hardcover_nonfiction + paperback_nonfiction hardcover_nonfiction + paperback_fiction hardcover_nonfiction = total_books ∧
    hardcover_nonfiction = 55 := by
  sorry

end NUMINAMATH_CALUDE_thabo_hardcover_nonfiction_l3435_343517


namespace NUMINAMATH_CALUDE_theo_homework_assignments_l3435_343507

/-- Calculates the number of assignments for a given set number -/
def assignmentsPerSet (setNumber : Nat) : Nat :=
  2^(setNumber - 1)

/-- Calculates the total assignments for a given number of sets -/
def totalAssignments (sets : Nat) : Nat :=
  (List.range sets).map (fun i => 6 * assignmentsPerSet (i + 1)) |>.sum

theorem theo_homework_assignments :
  totalAssignments 5 = 186 := by
  sorry

#eval totalAssignments 5

end NUMINAMATH_CALUDE_theo_homework_assignments_l3435_343507


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l3435_343546

/-- Given a circle with polar equation ρ = 5cos(θ) - 5√3sin(θ), 
    its center coordinates in polar form are (5, 5π/3) -/
theorem circle_center_coordinates (θ : Real) (ρ : Real) :
  ρ = 5 * Real.cos θ - 5 * Real.sqrt 3 * Real.sin θ →
  ∃ (r : Real) (φ : Real),
    r = 5 ∧ φ = 5 * Real.pi / 3 ∧
    r * Real.cos φ = 5 / 2 ∧
    r * Real.sin φ = -5 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l3435_343546


namespace NUMINAMATH_CALUDE_inequality_proof_l3435_343586

theorem inequality_proof (a b : ℝ) (h : a < b) : 7 * a - 7 * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3435_343586


namespace NUMINAMATH_CALUDE_bicycle_journey_l3435_343593

theorem bicycle_journey (t₁ t₂ : ℝ) (h₁ : t₁ > 0) (h₂ : t₂ > 0) : 
  (5 * t₁ + 15 * t₂) / (t₁ + t₂) = 10 → t₂ / (t₁ + t₂) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_journey_l3435_343593


namespace NUMINAMATH_CALUDE_negation_of_all_nonnegative_l3435_343587

theorem negation_of_all_nonnegative (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x ≥ 0) ↔ (∃ x : ℝ, x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_nonnegative_l3435_343587


namespace NUMINAMATH_CALUDE_unique_ages_l3435_343589

/-- Represents the ages of Gala, Vova, and Katya -/
structure Ages where
  gala : ℕ
  vova : ℕ
  katya : ℕ

/-- Checks if the given ages satisfy all the conditions -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.gala < 6 ∧
  ages.vova + ages.katya = 112 ∧
  (ages.vova / ages.gala : ℚ) = (ages.katya / ages.vova : ℚ)

/-- Theorem stating that the only ages satisfying all conditions are 2, 14, and 98 -/
theorem unique_ages : ∃! ages : Ages, satisfies_conditions ages ∧ ages.gala = 2 ∧ ages.vova = 14 ∧ ages.katya = 98 := by
  sorry

end NUMINAMATH_CALUDE_unique_ages_l3435_343589


namespace NUMINAMATH_CALUDE_difference_closure_l3435_343540

def is_closed_set (A : Set Int) : Prop :=
  (∃ (a b : Int), a ∈ A ∧ a > 0 ∧ b ∈ A ∧ b < 0) ∧
  (∀ a b : Int, a ∈ A → b ∈ A → (2 * a) ∈ A ∧ (a + b) ∈ A)

theorem difference_closure (A : Set Int) (h : is_closed_set A) :
  ∀ x y : Int, x ∈ A → y ∈ A → (x - y) ∈ A :=
by sorry

end NUMINAMATH_CALUDE_difference_closure_l3435_343540


namespace NUMINAMATH_CALUDE_unique_solution_system_l3435_343509

theorem unique_solution_system : 
  ∃! (x y z : ℕ+), 
    (2 * x * z = y^2) ∧ 
    (x + z = 1987) ∧
    (x = 1458) ∧ 
    (y = 1242) ∧ 
    (z = 529) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3435_343509


namespace NUMINAMATH_CALUDE_max_value_trig_sum_l3435_343524

theorem max_value_trig_sum (a b φ : ℝ) :
  ∃ (max : ℝ), ∀ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) ≤ max ∧
  ∃ θ₀ : ℝ, a * Real.cos (θ₀ + φ) + b * Real.sin (θ₀ + φ) = max ∧
  max = Real.sqrt (a^2 + b^2) :=
sorry

end NUMINAMATH_CALUDE_max_value_trig_sum_l3435_343524


namespace NUMINAMATH_CALUDE_display_rows_l3435_343544

/-- Represents the number of cans in a row given its position from the top. -/
def cans_in_row (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- Represents the total number of cans in the first n rows. -/
def total_cans (n : ℕ) : ℕ := n * (cans_in_row 1 + cans_in_row n) / 2

/-- The number of rows in the display is 10, given the conditions. -/
theorem display_rows :
  ∃ (n : ℕ), n > 0 ∧ total_cans n = 145 ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_display_rows_l3435_343544


namespace NUMINAMATH_CALUDE_milk_packets_average_price_l3435_343520

theorem milk_packets_average_price 
  (total_packets : ℕ) 
  (kept_packets : ℕ) 
  (returned_packets : ℕ) 
  (kept_avg_price : ℚ) 
  (returned_avg_price : ℚ) :
  total_packets = kept_packets + returned_packets →
  kept_packets = 3 →
  returned_packets = 2 →
  kept_avg_price = 12 →
  returned_avg_price = 32 →
  (kept_packets * kept_avg_price + returned_packets * returned_avg_price) / total_packets = 20 :=
by sorry

end NUMINAMATH_CALUDE_milk_packets_average_price_l3435_343520


namespace NUMINAMATH_CALUDE_a_squared_b_plus_ab_squared_l3435_343539

theorem a_squared_b_plus_ab_squared (a b : ℝ) (h1 : a + b = 6) (h2 : a * b = 7) :
  a^2 * b + a * b^2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_a_squared_b_plus_ab_squared_l3435_343539


namespace NUMINAMATH_CALUDE_no_valid_balanced_coloring_l3435_343572

/-- A chessboard is a 2D grid of squares that can be colored black or white -/
def Chessboard := Fin 1900 → Fin 1900 → Bool

/-- A point on the chessboard -/
def Point := Fin 1900 × Fin 1900

/-- The center point of the chessboard -/
def center : Point := (949, 949)

/-- Two points are symmetric if they are equidistant from the center in opposite directions -/
def symmetric (p q : Point) : Prop :=
  p.1 + q.1 = 2 * center.1 ∧ p.2 + q.2 = 2 * center.2

/-- A valid coloring satisfies the symmetry condition -/
def valid_coloring (c : Chessboard) : Prop :=
  ∀ p q : Point, symmetric p q → c p.1 p.2 ≠ c q.1 q.2

/-- A balanced coloring has an equal number of black and white squares in each row and column -/
def balanced_coloring (c : Chessboard) : Prop :=
  (∀ i : Fin 1900, (Finset.filter (λ j => c i j) Finset.univ).card = 950) ∧
  (∀ j : Fin 1900, (Finset.filter (λ i => c i j) Finset.univ).card = 950)

/-- The main theorem: it's impossible to have a valid and balanced coloring -/
theorem no_valid_balanced_coloring :
  ¬∃ c : Chessboard, valid_coloring c ∧ balanced_coloring c := by
  sorry

end NUMINAMATH_CALUDE_no_valid_balanced_coloring_l3435_343572


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3435_343564

theorem cubic_equation_solution (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3435_343564


namespace NUMINAMATH_CALUDE_board_numbers_theorem_l3435_343515

theorem board_numbers_theorem (a b c : ℝ) : 
  ({a, b, c} : Set ℝ) = {a - 2, b + 2, c^2} → 
  a + b + c = 2005 → 
  a = 1003 ∨ a = 1002 := by
  sorry

end NUMINAMATH_CALUDE_board_numbers_theorem_l3435_343515


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3435_343511

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 4 = 0 ∧ 
   ∀ y : ℝ, y^2 + k*y + 4 = 0 → y = x) → 
  k = 4 ∨ k = -4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3435_343511


namespace NUMINAMATH_CALUDE_tully_age_proof_l3435_343563

def kate_current_age : ℕ := 29

theorem tully_age_proof (tully_age_year_ago : ℕ) : 
  (tully_age_year_ago + 4 = 2 * (kate_current_age + 3)) → tully_age_year_ago = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_tully_age_proof_l3435_343563


namespace NUMINAMATH_CALUDE_squirrel_nut_difference_l3435_343505

theorem squirrel_nut_difference :
  let num_squirrels : ℕ := 4
  let num_nuts : ℕ := 2
  num_squirrels - num_nuts = 2 :=
by sorry

end NUMINAMATH_CALUDE_squirrel_nut_difference_l3435_343505


namespace NUMINAMATH_CALUDE_circle_trajectory_intersection_l3435_343508

-- Define the point F
def F : ℝ × ℝ := (1, 0)

-- Define the trajectory curve E
def E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line that intersects E and C₁
def line (x y b : ℝ) : Prop := y = (1/2)*x + b

-- Define the condition for complementary angles
def complementary_angles (B D : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := B
  let (x₂, y₂) := D
  (y₁ / (x₁ - 1)) + (y₂ / (x₂ - 1)) = 0

theorem circle_trajectory_intersection :
  ∀ (A B C D : ℝ × ℝ) (b : ℝ),
  E B.1 B.2 → E D.1 D.2 →
  C₁ A.1 A.2 → C₁ C.1 C.2 →
  line A.1 A.2 b → line B.1 B.2 b → line C.1 C.2 b → line D.1 D.2 b →
  complementary_angles B D →
  ∃ (AB CD : ℝ), AB + CD = (36 * Real.sqrt 5) / 5 :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_intersection_l3435_343508


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l3435_343584

theorem intersection_of_three_lines (k : ℝ) : 
  (∃! p : ℝ × ℝ, 
    (p.1 + k * p.2 = 0) ∧ 
    (2 * p.1 + 3 * p.2 + 8 = 0) ∧ 
    (p.1 - p.2 - 1 = 0)) → 
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l3435_343584


namespace NUMINAMATH_CALUDE_actual_average_height_l3435_343549

/-- The number of boys in the class -/
def num_boys : ℕ := 35

/-- The initial average height in centimeters -/
def initial_avg : ℚ := 182

/-- The incorrectly recorded height in centimeters -/
def incorrect_height : ℚ := 166

/-- The correct height in centimeters -/
def correct_height : ℚ := 106

/-- The actual average height after correction -/
def actual_avg : ℚ := (num_boys * initial_avg - (incorrect_height - correct_height)) / num_boys

theorem actual_average_height :
  ∃ ε > 0, abs (actual_avg - 180.29) < ε :=
sorry

end NUMINAMATH_CALUDE_actual_average_height_l3435_343549


namespace NUMINAMATH_CALUDE_problem_solution_l3435_343560

theorem problem_solution : ∀ M N X : ℕ,
  M = 2022 / 3 →
  N = M / 3 →
  X = M + N →
  X = 898 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3435_343560


namespace NUMINAMATH_CALUDE_max_square_plots_l3435_343514

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available internal fencing -/
def available_fencing : ℕ := 1994

/-- Calculates the number of square plots given the side length -/
def num_plots (dims : FieldDimensions) (side_length : ℕ) : ℕ :=
  (dims.width / side_length) * (dims.length / side_length)

/-- Calculates the required internal fencing for a given configuration -/
def required_fencing (dims : FieldDimensions) (side_length : ℕ) : ℕ :=
  (dims.width / side_length - 1) * dims.length + (dims.length / side_length - 1) * dims.width

/-- Theorem stating that 78 is the maximum number of square plots -/
theorem max_square_plots (dims : FieldDimensions) 
    (h_width : dims.width = 24) 
    (h_length : dims.length = 52) : 
    ∀ side_length : ℕ, 
      side_length > 0 → 
      dims.width % side_length = 0 → 
      dims.length % side_length = 0 → 
      required_fencing dims side_length ≤ available_fencing → 
      num_plots dims side_length ≤ 78 :=
  sorry

#check max_square_plots

end NUMINAMATH_CALUDE_max_square_plots_l3435_343514


namespace NUMINAMATH_CALUDE_inverse_square_relation_l3435_343582

theorem inverse_square_relation (k : ℝ) (a b c : ℝ) :
  (∀ a b c, a^2 * b^2 / c = k) →
  (4^2 * 2^2 / 3 = k) →
  (a^2 * 4^2 / 6 = k) →
  a^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_relation_l3435_343582


namespace NUMINAMATH_CALUDE_unique_pair_sum_28_l3435_343547

theorem unique_pair_sum_28 (a b : ℕ) : 
  a ≠ b → a > 11 → b > 11 → a + b = 28 → (Even a ∨ Even b) → 
  ((a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12)) := by
sorry

end NUMINAMATH_CALUDE_unique_pair_sum_28_l3435_343547


namespace NUMINAMATH_CALUDE_ten_apples_left_l3435_343531

/-- The number of apples left after Frank's dog eats some -/
def apples_left (on_tree : ℕ) (on_ground : ℕ) (eaten : ℕ) : ℕ :=
  on_tree + (on_ground - eaten)

/-- Theorem: Given the initial conditions, there are 10 apples left -/
theorem ten_apples_left : apples_left 5 8 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_apples_left_l3435_343531


namespace NUMINAMATH_CALUDE_janes_mean_score_l3435_343532

def janes_scores : List ℝ := [85, 90, 95, 80, 100]

theorem janes_mean_score : 
  (janes_scores.sum / janes_scores.length : ℝ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_janes_mean_score_l3435_343532


namespace NUMINAMATH_CALUDE_tens_digit_of_23_to_2045_l3435_343502

theorem tens_digit_of_23_to_2045 : 23^2045 ≡ 43 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_23_to_2045_l3435_343502


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l3435_343597

theorem least_number_with_remainder (n : ℕ) : n = 266 ↔ 
  (n > 0 ∧ 
   n % 33 = 2 ∧ 
   n % 8 = 2 ∧ 
   ∀ m : ℕ, m > 0 → m % 33 = 2 → m % 8 = 2 → m ≥ n) := by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l3435_343597


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l3435_343512

-- Define the diamond operation
def diamond (A B : ℝ) : ℝ := 4 * A + B^2 + 7

-- Theorem statement
theorem diamond_equation_solution :
  ∃ A : ℝ, diamond A 3 = 85 ∧ A = 17.25 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l3435_343512


namespace NUMINAMATH_CALUDE_power_mod_29_l3435_343527

theorem power_mod_29 : 17^2003 % 29 = 26 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_29_l3435_343527


namespace NUMINAMATH_CALUDE_product_ratio_l3435_343506

def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def first_three_primes : List ℕ := [2, 3, 5]
def next_three_composites : List ℕ := [14, 15, 16]

theorem product_ratio :
  (first_six_composites.prod) / ((first_three_primes ++ next_three_composites).prod) = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_l3435_343506


namespace NUMINAMATH_CALUDE_possible_values_of_y_l3435_343585

theorem possible_values_of_y (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  let y := ((x - 3)^2 * (x + 4)) / (2 * x - 5)
  y = 0 ∨ y = 144 ∨ y = -24 := by sorry

end NUMINAMATH_CALUDE_possible_values_of_y_l3435_343585


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l3435_343592

/-- Given a rectangular metallic sheet with length 48 m, from which squares of side 8 m
    are cut from each corner to form an open box with volume 5120 m³,
    prove that the width of the original metallic sheet is 36 m. -/
theorem metallic_sheet_width :
  ∀ (w : ℝ),
  let length : ℝ := 48
  let cut_side : ℝ := 8
  let box_volume : ℝ := 5120
  let box_length : ℝ := length - 2 * cut_side
  let box_width : ℝ := w - 2 * cut_side
  let box_height : ℝ := cut_side
  box_volume = box_length * box_width * box_height →
  w = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_metallic_sheet_width_l3435_343592


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l3435_343559

theorem tan_sum_pi_twelfths : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l3435_343559


namespace NUMINAMATH_CALUDE_meaningful_range_l3435_343523

def is_meaningful (x : ℝ) : Prop :=
  x ≥ 3 ∧ x ≠ 4

theorem meaningful_range (x : ℝ) :
  (∃ y : ℝ, y = Real.sqrt (x - 3) / (x - 4)) ↔ is_meaningful x :=
sorry

end NUMINAMATH_CALUDE_meaningful_range_l3435_343523


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3435_343552

theorem quadratic_one_solution (q : ℝ) : 
  q ≠ 0 ∧ (∃! x : ℝ, q * x^2 - 18 * x + 8 = 0) ↔ q = 81/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3435_343552


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l3435_343528

/-- The focal distance of the hyperbola 2x^2 - y^2 = 6 is 6 -/
theorem hyperbola_focal_distance :
  let hyperbola := {(x, y) : ℝ × ℝ | 2 * x^2 - y^2 = 6}
  ∃ f : ℝ, f = 6 ∧ ∀ (x y : ℝ), (x, y) ∈ hyperbola →
    ∃ (F₁ F₂ : ℝ × ℝ), abs (x - F₁.1) + abs (x - F₂.1) = 2 * f :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l3435_343528


namespace NUMINAMATH_CALUDE_fishbowl_count_l3435_343541

theorem fishbowl_count (total_fish : ℕ) (fish_per_bowl : ℕ) (h1 : total_fish = 6003) (h2 : fish_per_bowl = 23) :
  total_fish / fish_per_bowl = 261 :=
by sorry

end NUMINAMATH_CALUDE_fishbowl_count_l3435_343541


namespace NUMINAMATH_CALUDE_inequalities_proof_l3435_343583

theorem inequalities_proof (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  (a + b < a * b) ∧ (b / a + a / b > 2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3435_343583


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3435_343536

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x ≠ 1 ∧ (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3435_343536


namespace NUMINAMATH_CALUDE_estate_division_l3435_343574

theorem estate_division (estate : ℝ) 
  (wife_share son_share daughter_share cook_share : ℝ) : 
  (daughter_share + son_share = estate / 2) →
  (daughter_share = 4 * son_share / 3) →
  (wife_share = 2 * son_share) →
  (cook_share = 500) →
  (estate = wife_share + son_share + daughter_share + cook_share) →
  estate = 7000 := by
  sorry

#check estate_division

end NUMINAMATH_CALUDE_estate_division_l3435_343574


namespace NUMINAMATH_CALUDE_football_lineup_combinations_l3435_343538

def total_members : ℕ := 12
def offensive_linemen : ℕ := 4
def positions : ℕ := 5

def lineup_combinations : ℕ :=
  offensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

theorem football_lineup_combinations :
  lineup_combinations = 31680 := by
  sorry

end NUMINAMATH_CALUDE_football_lineup_combinations_l3435_343538


namespace NUMINAMATH_CALUDE_rectangle_area_l3435_343577

theorem rectangle_area (width : ℝ) (length : ℝ) : 
  width = 7 → length = 4 * width → width * length = 196 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3435_343577


namespace NUMINAMATH_CALUDE_marbles_lost_l3435_343519

theorem marbles_lost (initial : ℕ) (final : ℕ) (h1 : initial = 38) (h2 : final = 23) :
  initial - final = 15 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_l3435_343519


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3435_343548

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (2, m) (m, 2) → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3435_343548


namespace NUMINAMATH_CALUDE_parallel_intersection_lines_l3435_343590

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the parallelism relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection operation for a plane and a line
variable (intersect : Plane → Plane → Line)

-- Define the parallelism relation for lines
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_intersection_lines
  (m n : Line)
  (α β γ : Plane)
  (h1 : α ≠ β)
  (h2 : α ≠ γ)
  (h3 : β ≠ γ)
  (h4 : parallel_planes α β)
  (h5 : intersect α γ = m)
  (h6 : intersect β γ = n) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_parallel_intersection_lines_l3435_343590


namespace NUMINAMATH_CALUDE_import_tax_calculation_l3435_343595

/-- Given an item with a total value V, subject to a 7% import tax on the portion
    exceeding $1,000, prove that if the tax paid is $109.90, then V = $2,567. -/
theorem import_tax_calculation (V : ℝ) : 
  (0.07 * (V - 1000) = 109.90) → V = 2567 := by
  sorry

end NUMINAMATH_CALUDE_import_tax_calculation_l3435_343595


namespace NUMINAMATH_CALUDE_population_reaches_max_capacity_l3435_343543

/-- The number of acres on the island of Nisos -/
def island_acres : ℕ := 36000

/-- The number of acres required per person -/
def acres_per_person : ℕ := 2

/-- The initial population in 2040 -/
def initial_population : ℕ := 300

/-- The number of years it takes for the population to quadruple -/
def quadruple_period : ℕ := 30

/-- The maximum capacity of the island -/
def max_capacity : ℕ := island_acres / acres_per_person

/-- The population after n periods -/
def population (n : ℕ) : ℕ := initial_population * 4^n

/-- The number of years from 2040 until the population reaches or exceeds the maximum capacity -/
theorem population_reaches_max_capacity : 
  ∃ n : ℕ, n * quadruple_period = 90 ∧ population n ≥ max_capacity ∧ population (n - 1) < max_capacity :=
sorry

end NUMINAMATH_CALUDE_population_reaches_max_capacity_l3435_343543


namespace NUMINAMATH_CALUDE_unique_abc_solution_l3435_343522

/-- Represents a base-7 number with two digits -/
def Base7TwoDigit (a b : Nat) : Nat := 7 * a + b

/-- Represents a base-7 number with one digit -/
def Base7OneDigit (c : Nat) : Nat := c

/-- Represents a base-7 number with two digits, where the first digit is 'c' and the second is 0 -/
def Base7TwoDigitWithZero (c : Nat) : Nat := 7 * c

theorem unique_abc_solution :
  ∀ (A B C : Nat),
    A ≠ 0 → B ≠ 0 → C ≠ 0 →
    A < 7 → B < 7 → C < 7 →
    A ≠ B → B ≠ C → A ≠ C →
    Base7TwoDigit A B + Base7OneDigit C = Base7TwoDigitWithZero C →
    Base7TwoDigit A B + Base7TwoDigit B A = Base7TwoDigit C C →
    A = 3 ∧ B = 2 ∧ C = 5 := by
  sorry

#check unique_abc_solution

end NUMINAMATH_CALUDE_unique_abc_solution_l3435_343522


namespace NUMINAMATH_CALUDE_sum_of_z_values_l3435_343545

-- Define the function f
def f (x : ℝ) : ℝ := (2*x)^2 - 3*(2*x) + 2

-- State the theorem
theorem sum_of_z_values (z : ℝ) : 
  (∃ z₁ z₂, f z₁ = 4 ∧ f z₂ = 4 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_z_values_l3435_343545


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3435_343526

theorem quadratic_equations_solutions :
  (∀ x, x * (x - 3) + x = 3 ↔ x = 3 ∨ x = -1) ∧
  (∀ x, 3 * x^2 - 1 = 4 * x ↔ x = (2 + Real.sqrt 7) / 3 ∨ x = (2 - Real.sqrt 7) / 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3435_343526


namespace NUMINAMATH_CALUDE_softball_team_composition_l3435_343568

theorem softball_team_composition (total : ℕ) (ratio : ℚ) : 
  total = 16 ∧ ratio = 5/11 → ∃ (men women : ℕ), 
    men + women = total ∧ 
    (men : ℚ) / (women : ℚ) = ratio ∧ 
    women - men = 6 :=
by sorry

end NUMINAMATH_CALUDE_softball_team_composition_l3435_343568


namespace NUMINAMATH_CALUDE_share_difference_l3435_343570

def money_distribution (total : ℕ) (faruk vasim ranjith : ℕ) : Prop :=
  faruk + vasim + ranjith = total ∧ 3 * ranjith = 7 * faruk ∧ faruk = vasim

theorem share_difference (total : ℕ) (faruk vasim ranjith : ℕ) :
  money_distribution total faruk vasim ranjith → vasim = 1500 → ranjith - faruk = 2000 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_l3435_343570


namespace NUMINAMATH_CALUDE_unique_base_solution_l3435_343591

/-- Converts a base-10 number to its representation in base b -/
def toBase (n : ℕ) (b : ℕ) : List ℕ := sorry

/-- Converts a number represented as a list of digits in base b to base 10 -/
def fromBase (digits : List ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if the equation 742_b - 305_b = 43C_b holds for a given base b -/
def equationHolds (b : ℕ) : Prop :=
  let lhs := fromBase (toBase 742 b) b - fromBase (toBase 305 b) b
  let rhs := fromBase (toBase 43 b) b * 12
  lhs = rhs

theorem unique_base_solution :
  ∃! b : ℕ, b > 1 ∧ equationHolds b :=
sorry

end NUMINAMATH_CALUDE_unique_base_solution_l3435_343591


namespace NUMINAMATH_CALUDE_seating_solution_l3435_343567

/-- A seating arrangement with rows of 6 or 7 people. -/
structure SeatingArrangement where
  rows_with_7 : ℕ
  rows_with_6 : ℕ
  total_people : ℕ
  h1 : total_people = 7 * rows_with_7 + 6 * rows_with_6
  h2 : total_people = 59

/-- The solution to the seating arrangement problem. -/
theorem seating_solution (s : SeatingArrangement) : s.rows_with_7 = 5 := by
  sorry

#check seating_solution

end NUMINAMATH_CALUDE_seating_solution_l3435_343567


namespace NUMINAMATH_CALUDE_letter_digit_impossibility_l3435_343581

theorem letter_digit_impossibility :
  ¬ ∃ (f : Fin 7 → Fin 10),
    Function.Injective f ∧
    (f 0 * f 1 * 0 : ℕ) = (f 2 * f 3 * f 4 * f 5 * f 6 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_letter_digit_impossibility_l3435_343581


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l3435_343530

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 14

-- Define the two fixed points
def point1 : ℝ × ℝ := (0, 2)
def point2 : ℝ × ℝ := (6, -4)

-- Theorem stating that the equation describes an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ) (h : 0 < a ∧ 0 < b),
    ∀ (x y : ℝ), conic_equation x y ↔
      (x - (point1.1 + point2.1)/2)^2/a^2 + (y - (point1.2 + point2.2)/2)^2/b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l3435_343530


namespace NUMINAMATH_CALUDE_complex_division_result_l3435_343575

theorem complex_division_result : (4 - 2*I) / (1 + I) = 1 - 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l3435_343575


namespace NUMINAMATH_CALUDE_sum_first_150_remainder_l3435_343513

theorem sum_first_150_remainder (n : ℕ) (h : n = 150) : 
  (n * (n + 1) / 2) % 12000 = 11325 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_150_remainder_l3435_343513


namespace NUMINAMATH_CALUDE_range_of_m_l3435_343566

/-- The equation |(x-1)(x-3)| = m*x has four distinct real roots -/
def has_four_distinct_roots (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (∀ (x : ℝ), |((x - 1) * (x - 3))| = m * x ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

/-- The theorem stating the range of m -/
theorem range_of_m : 
  ∀ (m : ℝ), has_four_distinct_roots m ↔ 0 < m ∧ m < 4 - 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3435_343566


namespace NUMINAMATH_CALUDE_range_of_expression_l3435_343555

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (z : ℝ), z = 4 * Real.arcsin x - Real.arccos y ∧ 
  -5 * π / 2 ≤ z ∧ z ≤ 3 * π / 2 ∧
  (∃ (x₁ y₁ : ℝ), x₁^2 + y₁^2 = 1 ∧ 4 * Real.arcsin x₁ - Real.arccos y₁ = -5 * π / 2) ∧
  (∃ (x₂ y₂ : ℝ), x₂^2 + y₂^2 = 1 ∧ 4 * Real.arcsin x₂ - Real.arccos y₂ = 3 * π / 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l3435_343555


namespace NUMINAMATH_CALUDE_baker_pastries_l3435_343598

/-- Given that Baker made 43 cakes, sold 154 pastries and 78 cakes,
    and sold 76 more pastries than cakes, prove that Baker made 154 pastries. -/
theorem baker_pastries :
  let cakes_made : ℕ := 43
  let pastries_sold : ℕ := 154
  let cakes_sold : ℕ := 78
  let difference : ℕ := 76
  pastries_sold = cakes_sold + difference →
  pastries_sold = 154
:= by sorry

end NUMINAMATH_CALUDE_baker_pastries_l3435_343598


namespace NUMINAMATH_CALUDE_ninety_ninth_digit_sum_l3435_343503

/-- The decimal expansion of 2/9 -/
def decimal_expansion_2_9 : ℚ := 2/9

/-- The decimal expansion of 3/11 -/
def decimal_expansion_3_11 : ℚ := 3/11

/-- The 99th digit after the decimal point in a rational number -/
def digit_99 (q : ℚ) : ℕ :=
  sorry

/-- Theorem: The 99th digit after the decimal point in the decimal expansion of 2/9 + 3/11 is 4 -/
theorem ninety_ninth_digit_sum :
  digit_99 (decimal_expansion_2_9 + decimal_expansion_3_11) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ninety_ninth_digit_sum_l3435_343503


namespace NUMINAMATH_CALUDE_center_value_theorem_l3435_343518

/-- Represents a 6x6 matrix with arithmetic sequences in rows and columns -/
def ArithmeticMatrix := Matrix (Fin 6) (Fin 6) ℝ

/-- Checks if a sequence is arithmetic -/
def is_arithmetic_sequence (seq : Fin 6 → ℝ) : Prop :=
  ∀ i j k : Fin 6, i < j ∧ j < k → seq j - seq i = seq k - seq j

/-- The matrix has arithmetic sequences in all rows and columns -/
def matrix_arithmetic (M : ArithmeticMatrix) : Prop :=
  (∀ i : Fin 6, is_arithmetic_sequence (λ j => M i j)) ∧
  (∀ j : Fin 6, is_arithmetic_sequence (λ i => M i j))

theorem center_value_theorem (M : ArithmeticMatrix) 
  (h_arithmetic : matrix_arithmetic M)
  (h_first_row : M 0 1 = 3 ∧ M 0 4 = 27)
  (h_last_row : M 5 1 = 25 ∧ M 5 4 = 85) :
  M 2 2 = 30 ∧ M 2 3 = 30 ∧ M 3 2 = 30 ∧ M 3 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_center_value_theorem_l3435_343518

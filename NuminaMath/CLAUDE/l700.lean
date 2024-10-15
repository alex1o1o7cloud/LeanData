import Mathlib

namespace NUMINAMATH_CALUDE_stability_comparison_l700_70092

/-- Represents a student's test performance -/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Determines if the first student's performance is more stable than the second -/
def more_stable (student1 student2 : StudentPerformance) : Prop :=
  student1.variance < student2.variance

theorem stability_comparison 
  (student_A student_B : StudentPerformance)
  (h_same_average : student_A.average_score = student_B.average_score)
  (h_A_variance : student_A.variance = 51)
  (h_B_variance : student_B.variance = 12) :
  more_stable student_B student_A :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l700_70092


namespace NUMINAMATH_CALUDE_inequality_proof_l700_70006

theorem inequality_proof (a : ℝ) : 2 * a^4 + 2 * a^2 - 1 ≥ (3/2) * (a^2 + a - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l700_70006


namespace NUMINAMATH_CALUDE_arithmetic_progression_possible_n_values_l700_70038

theorem arithmetic_progression_possible_n_values : 
  ∃! (S : Finset ℕ), 
    S.Nonempty ∧ 
    (∀ n ∈ S, n > 1) ∧
    (S.card = 4) ∧
    (∀ n ∈ S, ∃ a : ℤ, 120 = n * (a + (3 * n / 2 : ℚ) - (3 / 2 : ℚ))) ∧
    (∀ n : ℕ, n > 1 → (∃ a : ℤ, 120 = n * (a + (3 * n / 2 : ℚ) - (3 / 2 : ℚ))) → n ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_possible_n_values_l700_70038


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l700_70001

theorem complex_magnitude_problem (x y : ℝ) (h : (x + y * Complex.I) * Complex.I = 1 + Complex.I) :
  Complex.abs (x + 2 * y * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l700_70001


namespace NUMINAMATH_CALUDE_largest_floor_value_l700_70089

/-- A positive real number that rounds to 20 -/
def A : ℝ := sorry

/-- A positive real number that rounds to 23 -/
def B : ℝ := sorry

/-- A rounds to 20 -/
axiom hA : 19.5 ≤ A ∧ A < 20.5

/-- B rounds to 23 -/
axiom hB : 22.5 ≤ B ∧ B < 23.5

/-- A and B are positive -/
axiom pos_A : A > 0
axiom pos_B : B > 0

theorem largest_floor_value :
  ∃ (x : ℝ) (y : ℝ), 19.5 ≤ x ∧ x < 20.5 ∧ 22.5 ≤ y ∧ y < 23.5 ∧
  ∀ (a : ℝ) (b : ℝ), 19.5 ≤ a ∧ a < 20.5 ∧ 22.5 ≤ b ∧ b < 23.5 →
  ⌊100 * x / y⌋ ≥ ⌊100 * a / b⌋ ∧ ⌊100 * x / y⌋ = 91 :=
sorry

end NUMINAMATH_CALUDE_largest_floor_value_l700_70089


namespace NUMINAMATH_CALUDE_longest_side_is_72_l700_70000

/-- A rectangle with specific properties --/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 120
  area_eq : length * width = 2880

/-- The longest side of a SpecialRectangle is 72 --/
theorem longest_side_is_72 (rect : SpecialRectangle) : 
  max rect.length rect.width = 72 := by
  sorry

#check longest_side_is_72

end NUMINAMATH_CALUDE_longest_side_is_72_l700_70000


namespace NUMINAMATH_CALUDE_smaller_part_is_4000_l700_70072

/-- Represents an investment split into two parts -/
structure Investment where
  total : ℝ
  greater_part : ℝ
  smaller_part : ℝ
  greater_rate : ℝ
  smaller_rate : ℝ

/-- Conditions for the investment problem -/
def investment_conditions (i : Investment) : Prop :=
  i.total = 10000 ∧
  i.greater_part + i.smaller_part = i.total ∧
  i.greater_rate = 0.06 ∧
  i.smaller_rate = 0.05 ∧
  i.greater_rate * i.greater_part = i.smaller_rate * i.smaller_part + 160

/-- Theorem stating that under the given conditions, the smaller part of the investment is 4000 -/
theorem smaller_part_is_4000 (i : Investment) 
  (h : investment_conditions i) : i.smaller_part = 4000 := by
  sorry

end NUMINAMATH_CALUDE_smaller_part_is_4000_l700_70072


namespace NUMINAMATH_CALUDE_three_card_picks_count_l700_70098

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- The number of ways to pick three different cards from a standard deck where order matters -/
def threeCardPicks (d : Deck) : ℕ :=
  52 * 51 * 50

/-- Theorem stating that the number of ways to pick three different cards from a standard 
    52-card deck, where order matters, is equal to 132600 -/
theorem three_card_picks_count (d : Deck) : threeCardPicks d = 132600 := by
  sorry

end NUMINAMATH_CALUDE_three_card_picks_count_l700_70098


namespace NUMINAMATH_CALUDE_smallest_n_with_divisibility_n_98_satisfies_conditions_smallest_n_is_98_l700_70054

/-- Checks if at least one of three consecutive integers is divisible by a given number. -/
def oneOfThreeDivisibleBy (n : ℕ) (d : ℕ) : Prop :=
  d ∣ n ∨ d ∣ (n + 1) ∨ d ∣ (n + 2)

/-- The main theorem stating that 98 is the smallest positive integer satisfying the given conditions. -/
theorem smallest_n_with_divisibility : ∀ n : ℕ, n > 0 →
  (oneOfThreeDivisibleBy n (2^2) ∧
   oneOfThreeDivisibleBy n (3^2) ∧
   oneOfThreeDivisibleBy n (5^2) ∧
   oneOfThreeDivisibleBy n (7^2)) →
  n ≥ 98 :=
by sorry

/-- Proof that 98 satisfies all the divisibility conditions. -/
theorem n_98_satisfies_conditions :
  oneOfThreeDivisibleBy 98 (2^2) ∧
  oneOfThreeDivisibleBy 98 (3^2) ∧
  oneOfThreeDivisibleBy 98 (5^2) ∧
  oneOfThreeDivisibleBy 98 (7^2) :=
by sorry

/-- The final theorem combining the above results to prove 98 is the smallest such positive integer. -/
theorem smallest_n_is_98 :
  ∃ n : ℕ, n > 0 ∧
  oneOfThreeDivisibleBy n (2^2) ∧
  oneOfThreeDivisibleBy n (3^2) ∧
  oneOfThreeDivisibleBy n (5^2) ∧
  oneOfThreeDivisibleBy n (7^2) ∧
  ∀ m : ℕ, m > 0 →
    (oneOfThreeDivisibleBy m (2^2) ∧
     oneOfThreeDivisibleBy m (3^2) ∧
     oneOfThreeDivisibleBy m (5^2) ∧
     oneOfThreeDivisibleBy m (7^2)) →
    m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_divisibility_n_98_satisfies_conditions_smallest_n_is_98_l700_70054


namespace NUMINAMATH_CALUDE_draw_points_value_l700_70085

/-- Represents the points system in a football competition --/
structure PointSystem where
  victory_points : ℕ
  draw_points : ℕ
  defeat_points : ℕ

/-- Represents the state of a team in the competition --/
structure TeamState where
  total_matches : ℕ
  matches_played : ℕ
  current_points : ℕ
  target_points : ℕ
  min_victories : ℕ

/-- The theorem to prove --/
theorem draw_points_value (ps : PointSystem) (ts : TeamState) : 
  ps.victory_points = 3 ∧ 
  ps.defeat_points = 0 ∧
  ts.total_matches = 20 ∧ 
  ts.matches_played = 5 ∧ 
  ts.current_points = 14 ∧ 
  ts.target_points = 40 ∧
  ts.min_victories = 6 →
  ps.draw_points = 2 := by
  sorry


end NUMINAMATH_CALUDE_draw_points_value_l700_70085


namespace NUMINAMATH_CALUDE_sphere_in_cube_volume_ratio_l700_70015

theorem sphere_in_cube_volume_ratio (cube_side : ℝ) (h : cube_side = 8) :
  let sphere_volume := (4 / 3) * Real.pi * (cube_side / 2)^3
  let cube_volume := cube_side^3
  sphere_volume / cube_volume = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_sphere_in_cube_volume_ratio_l700_70015


namespace NUMINAMATH_CALUDE_ages_sum_l700_70051

theorem ages_sum (a b c : ℕ) 
  (h1 : a = 20 + b + c) 
  (h2 : a^2 = 2000 + (b + c)^2) : 
  a + b + c = 80 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l700_70051


namespace NUMINAMATH_CALUDE_fraction_simplification_l700_70004

theorem fraction_simplification (a b x : ℝ) 
  (h1 : x = b / a) 
  (h2 : a ≠ b) 
  (h3 : a ≠ 0) : 
  (2 * a + b) / (2 * a - b) = (2 + x) / (2 - x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l700_70004


namespace NUMINAMATH_CALUDE_quadratic_properties_l700_70033

/-- A quadratic function passing through given points -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) :
  quadratic_function a b c (-2) = 6 →
  quadratic_function a b c 0 = -4 →
  quadratic_function a b c 1 = -6 →
  quadratic_function a b c 3 = -4 →
  (a > 0 ∧ ∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > (3/2 : ℝ) → quadratic_function a b c x₁ > quadratic_function a b c x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l700_70033


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l700_70079

theorem binomial_expansion_coefficient (n : ℕ) : 
  (Nat.choose n 2) * 9 = 54 → n = 4 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l700_70079


namespace NUMINAMATH_CALUDE_sequence_classification_l700_70016

/-- Given a sequence {a_n} where the sum of the first n terms S_n = a^n - 2 (a is a constant, a ≠ 0),
    the sequence {a_n} forms either an arithmetic sequence or a geometric sequence from the second term onwards. -/
theorem sequence_classification (a : ℝ) (h_a : a ≠ 0) :
  let S : ℕ → ℝ := λ n => a ^ n - 2
  let a_seq : ℕ → ℝ := λ n => S n - S (n - 1)
  (∀ n : ℕ, n ≥ 2 → ∃ d : ℝ, a_seq (n + 1) - a_seq n = d) ∨
  (∀ n : ℕ, n ≥ 2 → ∃ r : ℝ, a_seq (n + 1) / a_seq n = r) :=
by sorry

end NUMINAMATH_CALUDE_sequence_classification_l700_70016


namespace NUMINAMATH_CALUDE_abcd_inequality_l700_70029

theorem abcd_inequality (a b c d : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (hd : 0 < d ∧ d < 1) 
  (h_prod : a * b * c * d = (1 - a) * (1 - b) * (1 - c) * (1 - d)) : 
  (a + b + c + d) - (a + c) * (b + d) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_abcd_inequality_l700_70029


namespace NUMINAMATH_CALUDE_inequality_solution_set_l700_70063

theorem inequality_solution_set (x : ℝ) : 
  x^6 - (x + 2) > (x + 2)^3 - x^2 ↔ x < -1 ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l700_70063


namespace NUMINAMATH_CALUDE_tank_filling_time_l700_70005

/-- Given a tap that can fill a tank in 16 hours, and 3 additional similar taps opened after half the tank is filled, prove that the total time taken to fill the tank completely is 10 hours. -/
theorem tank_filling_time (fill_time : ℝ) (additional_taps : ℕ) : 
  fill_time = 16 → additional_taps = 3 → 
  (fill_time / 2) + (fill_time / (2 * (additional_taps + 1))) = 10 :=
by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l700_70005


namespace NUMINAMATH_CALUDE_largest_calculation_l700_70022

theorem largest_calculation :
  let a := 2 + 0 + 1 + 8
  let b := 2 * 0 + 1 + 8
  let c := 2 + 0 * 1 + 8
  let d := 2 + 0 + 1 * 8
  let e := 2 * 0 + 1 * 8
  (a ≥ b) ∧ (a ≥ c) ∧ (a ≥ d) ∧ (a ≥ e) := by
  sorry

end NUMINAMATH_CALUDE_largest_calculation_l700_70022


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l700_70096

theorem circle_area_from_circumference (k : ℝ) : 
  (∃ (r : ℝ), 2 * π * r = 30 * π ∧ π * r^2 = k * π) → k = 225 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l700_70096


namespace NUMINAMATH_CALUDE_problem_1_proof_l700_70011

theorem problem_1_proof : (1 : ℝ) - 1^2 + (64 : ℝ)^(1/3) - (-2) * (9 : ℝ)^(1/2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_proof_l700_70011


namespace NUMINAMATH_CALUDE_smaller_number_l700_70076

theorem smaller_number (L S : ℕ) (hL : L > S) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : S = 270 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_l700_70076


namespace NUMINAMATH_CALUDE_integer_triple_divisibility_l700_70082

theorem integer_triple_divisibility :
  ∀ a b c : ℤ,
  (1 < a ∧ a < b ∧ b < c) →
  ((a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) →
  ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by sorry

end NUMINAMATH_CALUDE_integer_triple_divisibility_l700_70082


namespace NUMINAMATH_CALUDE_parity_and_squares_equivalence_l700_70034

theorem parity_and_squares_equivalence (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a % 2 = b % 2) ↔ (∃ (c d : ℕ), 0 < c ∧ 0 < d ∧ a^2 + b^2 + c^2 + 1 = d^2) := by
  sorry

end NUMINAMATH_CALUDE_parity_and_squares_equivalence_l700_70034


namespace NUMINAMATH_CALUDE_triangle_side_length_l700_70067

/-- Given a triangle DEF with side lengths and a median, prove the length of DF. -/
theorem triangle_side_length (DE EF DM : ℝ) (hDE : DE = 7) (hEF : EF = 10) (hDM : DM = 5) :
  ∃ (DF : ℝ), DF = Real.sqrt 149 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l700_70067


namespace NUMINAMATH_CALUDE_custom_op_nested_l700_70009

/-- Custom binary operation ⊗ -/
def custom_op (x y : ℝ) : ℝ := x^3 - y^2 + x

/-- Theorem stating the result of k ⊗ (k ⊗ k) -/
theorem custom_op_nested (k : ℝ) : custom_op k (custom_op k k) = -k^6 + 2*k^5 - 3*k^4 + 3*k^3 - k^2 + 2*k := by
  sorry

end NUMINAMATH_CALUDE_custom_op_nested_l700_70009


namespace NUMINAMATH_CALUDE_sum_of_perfect_squares_l700_70087

theorem sum_of_perfect_squares (x : ℕ) (h : ∃ k : ℕ, x = k ^ 2) :
  ∃ y : ℕ, y > x ∧ (∃ m : ℕ, y = m ^ 2) ∧ x + y = 2 * x + 2 * (x.sqrt) + 1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_perfect_squares_l700_70087


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l700_70099

theorem weight_of_replaced_person 
  (n : ℕ) 
  (original_total : ℝ) 
  (new_weight : ℝ) 
  (average_increase : ℝ) 
  (h1 : n = 10)
  (h2 : new_weight = 75)
  (h3 : average_increase = 3)
  : 
  (original_total + new_weight - (original_total / n + average_increase * n)) = 45 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l700_70099


namespace NUMINAMATH_CALUDE_find_number_l700_70017

theorem find_number : ∃ N : ℕ,
  (N = (555 + 445) * (2 * (555 - 445)) + 30) ∧ 
  (N = 220030) := by
  sorry

end NUMINAMATH_CALUDE_find_number_l700_70017


namespace NUMINAMATH_CALUDE_chairs_to_hall_l700_70070

theorem chairs_to_hall (num_students : ℕ) (chairs_per_trip : ℕ) (num_trips : ℕ) :
  num_students = 5 →
  chairs_per_trip = 5 →
  num_trips = 10 →
  num_students * chairs_per_trip * num_trips = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_chairs_to_hall_l700_70070


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l700_70018

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  (∀ n : ℕ, a n > 0) → a 4 * a 10 = 16 → a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l700_70018


namespace NUMINAMATH_CALUDE_race_outcomes_count_l700_70019

/-- The number of participants in the race -/
def total_participants : ℕ := 6

/-- The number of participants eligible for top three positions -/
def eligible_participants : ℕ := total_participants - 1

/-- The number of top positions to be filled -/
def top_positions : ℕ := 3

/-- Calculates the number of permutations for selecting k items from n items -/
def permutations (n k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

/-- The main theorem stating the number of possible race outcomes -/
theorem race_outcomes_count : 
  permutations eligible_participants top_positions = 60 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_count_l700_70019


namespace NUMINAMATH_CALUDE_race_time_difference_l700_70045

/-- The time difference between two runners in a race -/
def time_difference (distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  distance * speed2 - distance * speed1

/-- Proof of the time difference in the race -/
theorem race_time_difference :
  let malcolm_speed : ℝ := 7
  let joshua_speed : ℝ := 8
  let race_distance : ℝ := 15
  time_difference race_distance malcolm_speed joshua_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l700_70045


namespace NUMINAMATH_CALUDE_unique_magnitude_quadratic_l700_70086

/-- For the quadratic equation z^2 - 10z + 50 = 0, there is only one possible value for |z| -/
theorem unique_magnitude_quadratic : 
  ∃! m : ℝ, ∀ z : ℂ, z^2 - 10*z + 50 = 0 → Complex.abs z = m :=
by sorry

end NUMINAMATH_CALUDE_unique_magnitude_quadratic_l700_70086


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l700_70049

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^2 - 22 * X + 70 = (X - 7) * q + 63 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l700_70049


namespace NUMINAMATH_CALUDE_factor_expression_l700_70057

theorem factor_expression (x : ℝ) : 75 * x^13 + 450 * x^26 = 75 * x^13 * (1 + 6 * x^13) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l700_70057


namespace NUMINAMATH_CALUDE_people_in_virginia_l700_70025

/-- The number of people landing in Virginia given the initial passengers, layover changes, and crew members. -/
def peopleInVirginia (initialPassengers : ℕ) (texasOff texasOn ncOff ncOn crewMembers : ℕ) : ℕ :=
  initialPassengers - texasOff + texasOn - ncOff + ncOn + crewMembers

/-- Theorem stating that the number of people landing in Virginia is 67. -/
theorem people_in_virginia :
  peopleInVirginia 124 58 24 47 14 10 = 67 := by
  sorry

end NUMINAMATH_CALUDE_people_in_virginia_l700_70025


namespace NUMINAMATH_CALUDE_cot_thirty_degrees_l700_70032

theorem cot_thirty_degrees : 
  let θ : Real := 30 * π / 180 -- Convert 30 degrees to radians
  let cot (x : Real) := 1 / Real.tan x -- Definition of cotangent
  (Real.tan θ = 1 / Real.sqrt 3) → -- Given condition
  (cot θ = Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_cot_thirty_degrees_l700_70032


namespace NUMINAMATH_CALUDE_roller_coaster_cars_l700_70042

theorem roller_coaster_cars (n : ℕ) (h : n > 0) :
  (n - 1 : ℚ) / n = 1/2 ↔ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_cars_l700_70042


namespace NUMINAMATH_CALUDE_set_357_forms_triangle_l700_70046

/-- Triangle inequality theorem: the sum of any two sides must be greater than the third side --/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A set of three line segments can form a triangle if it satisfies the triangle inequality --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set (3, 5, 7) can form a triangle --/
theorem set_357_forms_triangle : can_form_triangle 3 5 7 := by
  sorry

end NUMINAMATH_CALUDE_set_357_forms_triangle_l700_70046


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l700_70095

/-- For a geometric sequence with first term a₁ and common ratio q -/
structure GeometricSequence where
  a₁ : ℝ
  q : ℝ
  h₁ : a₁ > 0

/-- The third term of a geometric sequence -/
def GeometricSequence.a₃ (g : GeometricSequence) : ℝ := g.a₁ * g.q^2

theorem geometric_sequence_condition (g : GeometricSequence) :
  (g.q > 1 → g.a₁ < g.a₃) ∧ 
  ¬(g.a₁ < g.a₃ → g.q > 1) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l700_70095


namespace NUMINAMATH_CALUDE_freshmen_in_liberal_arts_l700_70028

theorem freshmen_in_liberal_arts (total_students : ℝ) (freshmen_percent : ℝ) 
  (psych_majors_percent : ℝ) (freshmen_psych_liberal_arts_percent : ℝ) :
  freshmen_percent = 80 →
  psych_majors_percent = 50 →
  freshmen_psych_liberal_arts_percent = 24 →
  (freshmen_psych_liberal_arts_percent * total_students) / 
    (psych_majors_percent / 100 * freshmen_percent * total_students / 100) = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_freshmen_in_liberal_arts_l700_70028


namespace NUMINAMATH_CALUDE_jane_sum_minus_liam_sum_l700_70013

def jane_list : List Nat := List.range 50

def replace_3_with_2 (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def liam_list : List Nat := jane_list.map replace_3_with_2

theorem jane_sum_minus_liam_sum : 
  jane_list.sum - liam_list.sum = 105 := by sorry

end NUMINAMATH_CALUDE_jane_sum_minus_liam_sum_l700_70013


namespace NUMINAMATH_CALUDE_probability_proof_l700_70043

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def total_balls : ℕ := white_balls + black_balls

def probability_one_white_one_black : ℚ :=
  (white_balls * black_balls : ℚ) / (total_balls * (total_balls - 1) / 2)

theorem probability_proof :
  probability_one_white_one_black = 56 / 105 := by
  sorry

end NUMINAMATH_CALUDE_probability_proof_l700_70043


namespace NUMINAMATH_CALUDE_max_individual_award_l700_70061

theorem max_individual_award 
  (total_prize : ℕ) 
  (num_winners : ℕ) 
  (min_award : ℕ) 
  (h1 : total_prize = 2500)
  (h2 : num_winners = 25)
  (h3 : min_award = 50)
  (h4 : (3 : ℚ) / 5 * total_prize = (2 : ℚ) / 5 * num_winners * max_award)
  : ∃ max_award : ℕ, max_award = 1300 := by
  sorry

end NUMINAMATH_CALUDE_max_individual_award_l700_70061


namespace NUMINAMATH_CALUDE_ellipse_line_theorem_l700_70007

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the line l
def line_l (x : ℝ) : Prop := x = -2

-- Define a line passing through a point
def line_through_point (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the perpendicular bisector of a line segment
def perpendicular_bisector (k : ℝ) (x y : ℝ) : Prop := 
  y + k / (1 + 2*k^2) = -(1/k) * (x - 2*k^2 / (1 + 2*k^2))

-- Define the theorem
theorem ellipse_line_theorem (k : ℝ) (x₁ y₁ x₂ y₂ xp yp xc yc : ℝ) : 
  ellipse x₁ y₁ → 
  ellipse x₂ y₂ → 
  line_through_point k x₁ y₁ → 
  line_through_point k x₂ y₂ → 
  perpendicular_bisector k xp yp → 
  perpendicular_bisector k xc yc → 
  line_l xp → 
  (xc - 1)^2 + yc^2 = ((x₂ - x₁)^2 + (y₂ - y₁)^2) / 4 → 
  (xp - xc)^2 + (yp - yc)^2 = 4 * ((x₂ - x₁)^2 + (y₂ - y₁)^2) → 
  (k = 1 ∨ k = -1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_theorem_l700_70007


namespace NUMINAMATH_CALUDE_roulette_sectors_l700_70077

def roulette_wheel (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 10 ∧ 
  (1 - (5 / n)^2 : ℚ) = 3/4

theorem roulette_sectors : ∃ (n : ℕ), roulette_wheel n ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_roulette_sectors_l700_70077


namespace NUMINAMATH_CALUDE_proper_subsets_of_abc_l700_70053

def S : Set (Set Char) := {{'a', 'b', 'c'}}

theorem proper_subsets_of_abc :
  {s : Set Char | s ⊂ {'a', 'b', 'c'}} =
  {∅, {'a'}, {'b'}, {'c'}, {'a', 'b'}, {'a', 'c'}, {'b', 'c'}} := by
  sorry

end NUMINAMATH_CALUDE_proper_subsets_of_abc_l700_70053


namespace NUMINAMATH_CALUDE_eve_ran_distance_l700_70080

/-- The distance Eve walked in miles -/
def distance_walked : ℝ := 0.6

/-- The additional distance Eve ran compared to what she walked, in miles -/
def additional_distance : ℝ := 0.1

/-- The total distance Eve ran in miles -/
def distance_ran : ℝ := distance_walked + additional_distance

theorem eve_ran_distance : distance_ran = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_eve_ran_distance_l700_70080


namespace NUMINAMATH_CALUDE_blueberries_per_basket_l700_70039

theorem blueberries_per_basket (initial_basket : ℕ) (additional_baskets : ℕ) (total_blueberries : ℕ) : 
  initial_basket > 0 →
  additional_baskets = 9 →
  total_blueberries = 200 →
  total_blueberries = (initial_basket + additional_baskets) * initial_basket →
  initial_basket = 20 := by
  sorry

end NUMINAMATH_CALUDE_blueberries_per_basket_l700_70039


namespace NUMINAMATH_CALUDE_class_size_l700_70081

theorem class_size (S : ℕ) 
  (h1 : S / 3 + S * 2 / 5 + 12 = S) : S = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l700_70081


namespace NUMINAMATH_CALUDE_opposite_face_of_A_is_B_l700_70091

/-- Represents the letters on the cube faces -/
inductive CubeLetter
  | A | B | V | G | D | E

/-- Represents a face of the cube -/
structure CubeFace where
  letter : CubeLetter

/-- Represents the cube -/
structure Cube where
  faces : Finset CubeFace
  face_count : faces.card = 6

/-- Represents a perspective of the cube showing three visible faces -/
structure CubePerspective where
  visible_faces : Finset CubeFace
  visible_count : visible_faces.card = 3

/-- Defines the opposite face relation -/
def opposite_face (c : Cube) (f1 f2 : CubeFace) : Prop :=
  f1 ∈ c.faces ∧ f2 ∈ c.faces ∧ f1 ≠ f2 ∧ 
  ∀ (p : CubePerspective), ¬(f1 ∈ p.visible_faces ∧ f2 ∈ p.visible_faces)

theorem opposite_face_of_A_is_B 
  (c : Cube) 
  (p1 p2 p3 : CubePerspective) 
  (hA : ∃ (fA : CubeFace), fA ∈ c.faces ∧ fA.letter = CubeLetter.A)
  (hB : ∃ (fB : CubeFace), fB ∈ c.faces ∧ fB.letter = CubeLetter.B)
  (h_perspectives : 
    (∃ (f1 f2 : CubeFace), f1 ∈ p1.visible_faces ∧ f2 ∈ p1.visible_faces ∧ 
      f1.letter = CubeLetter.A ∧ f2.letter = CubeLetter.B) ∧
    (∃ (f1 f2 : CubeFace), f1 ∈ p2.visible_faces ∧ f2 ∈ p2.visible_faces ∧ 
      f1.letter = CubeLetter.B) ∧
    (∃ (f1 f2 : CubeFace), f1 ∈ p3.visible_faces ∧ f2 ∈ p3.visible_faces ∧ 
      f1.letter = CubeLetter.A)) :
  ∃ (fA fB : CubeFace), 
    fA.letter = CubeLetter.A ∧ 
    fB.letter = CubeLetter.B ∧ 
    opposite_face c fA fB :=
  sorry

end NUMINAMATH_CALUDE_opposite_face_of_A_is_B_l700_70091


namespace NUMINAMATH_CALUDE_final_elevation_proof_l700_70008

def calculate_final_elevation (initial_elevation : ℝ) 
                               (rate1 rate2 rate3 : ℝ) 
                               (time1 time2 time3 : ℝ) : ℝ :=
  initial_elevation - (rate1 * time1 + rate2 * time2 + rate3 * time3)

theorem final_elevation_proof (initial_elevation : ℝ) 
                              (rate1 rate2 rate3 : ℝ) 
                              (time1 time2 time3 : ℝ) :
  calculate_final_elevation initial_elevation rate1 rate2 rate3 time1 time2 time3 =
  initial_elevation - (rate1 * time1 + rate2 * time2 + rate3 * time3) :=
by
  sorry

#eval calculate_final_elevation 400 10 15 12 5 3 6

end NUMINAMATH_CALUDE_final_elevation_proof_l700_70008


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l700_70069

/-- Given three real numbers form a geometric progression, prove that the first term is 15 + 5√5 --/
theorem geometric_progression_solution (x : ℝ) : 
  (2*x + 10)^2 = x * (5*x + 10) → x = 15 + 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l700_70069


namespace NUMINAMATH_CALUDE_largest_fraction_l700_70052

theorem largest_fraction :
  let a := (8 + 5) / 3
  let b := 8 / (3 + 5)
  let c := (3 + 5) / 8
  let d := (8 + 3) / 5
  let e := 3 / (8 + 5)
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l700_70052


namespace NUMINAMATH_CALUDE_bakery_problem_l700_70023

/-- The number of ways to distribute additional items into bins, given a minimum per bin -/
def distribute_items (total_items : ℕ) (num_bins : ℕ) (min_per_bin : ℕ) : ℕ :=
  Nat.choose (total_items - num_bins * min_per_bin + num_bins - 1) (num_bins - 1)

theorem bakery_problem :
  distribute_items 10 4 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_bakery_problem_l700_70023


namespace NUMINAMATH_CALUDE_correct_employee_count_l700_70097

/-- The number of employees in Kim's office -/
def num_employees : ℕ := 9

/-- The total time Kim spends on her morning routine in minutes -/
def total_time : ℕ := 50

/-- The time Kim spends making coffee in minutes -/
def coffee_time : ℕ := 5

/-- The time Kim spends per employee for status update in minutes -/
def status_update_time : ℕ := 2

/-- The time Kim spends per employee for payroll update in minutes -/
def payroll_update_time : ℕ := 3

/-- Theorem stating that the number of employees is correct given the conditions -/
theorem correct_employee_count :
  num_employees * (status_update_time + payroll_update_time) + coffee_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_correct_employee_count_l700_70097


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l700_70056

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), (2^k : ℤ) ∣ (10^1004 - 4^502) ∧ 
  ∀ (m : ℕ), (2^m : ℤ) ∣ (10^1004 - 4^502) → m ≤ k :=
by
  use 1007
  sorry

#eval 1007  -- This will output the answer

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l700_70056


namespace NUMINAMATH_CALUDE_square_equality_l700_70047

theorem square_equality (a b : ℝ) : a = b → a^2 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_l700_70047


namespace NUMINAMATH_CALUDE_horner_method_f_3_f_3_equals_328_l700_70050

/-- Horner's method representation of a polynomial -/
def horner_rep (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

/-- The polynomial f(x) = x^5 + 2x^3 + 3x^2 + x + 1 -/
def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem horner_method_f_3 :
  f 3 = horner_rep [1, 0, 2, 3, 1, 1] 3 := by
  sorry

theorem f_3_equals_328 : f 3 = 328 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_3_f_3_equals_328_l700_70050


namespace NUMINAMATH_CALUDE_b_present_age_l700_70048

/-- Given two people A and B, prove that B's present age is 34 years -/
theorem b_present_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- In 10 years, A will be twice as old as B was 10 years ago
  (a = b + 4) →              -- A is now 4 years older than B
  b = 34 := by
sorry

end NUMINAMATH_CALUDE_b_present_age_l700_70048


namespace NUMINAMATH_CALUDE_skew_sufficient_not_necessary_for_non_intersecting_l700_70044

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l₁ l₂ : Line3D) : Prop :=
  sorry

/-- Two lines intersect if they share a common point -/
def intersect (l₁ l₂ : Line3D) : Prop :=
  sorry

/-- Main theorem: Skew lines are sufficient but not necessary for non-intersecting lines -/
theorem skew_sufficient_not_necessary_for_non_intersecting :
  (∀ l₁ l₂ : Line3D, are_skew l₁ l₂ → ¬(intersect l₁ l₂)) ∧
  (∃ l₁ l₂ : Line3D, ¬(intersect l₁ l₂) ∧ ¬(are_skew l₁ l₂)) :=
by sorry

end NUMINAMATH_CALUDE_skew_sufficient_not_necessary_for_non_intersecting_l700_70044


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l700_70065

/-- Given a point P(-3, 1), its symmetric point with respect to the x-axis has coordinates (-3, -1) -/
theorem symmetric_point_wrt_x_axis :
  let P : ℝ × ℝ := (-3, 1)
  let symmetric_point := (P.1, -P.2)
  symmetric_point = (-3, -1) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l700_70065


namespace NUMINAMATH_CALUDE_transformer_min_current_load_l700_70062

def number_of_units : ℕ := 3
def running_current_per_unit : ℕ := 40
def starting_current_multiplier : ℕ := 2

theorem transformer_min_current_load :
  let total_running_current := number_of_units * running_current_per_unit
  let min_starting_current := starting_current_multiplier * total_running_current
  min_starting_current = 240 := by
  sorry

end NUMINAMATH_CALUDE_transformer_min_current_load_l700_70062


namespace NUMINAMATH_CALUDE_stream_speed_l700_70084

theorem stream_speed (rowing_speed : ℝ) (total_time : ℝ) (distance : ℝ) (stream_speed : ℝ) : 
  rowing_speed = 10 →
  total_time = 5 →
  distance = 24 →
  (distance / (rowing_speed - stream_speed) + distance / (rowing_speed + stream_speed) = total_time) →
  stream_speed = 2 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l700_70084


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l700_70073

/-- Given a triangle with perimeter 720 cm and longest side 280 cm, 
    prove that the ratio of the sides can be expressed as k:l:1, where k + l = 1.5714 -/
theorem triangle_side_ratio (a b c : ℝ) (h_perimeter : a + b + c = 720) 
  (h_longest : c = 280) (h_c_longest : a ≤ c ∧ b ≤ c) :
  ∃ (k l : ℝ), k + l = 1.5714 ∧ (a / c = k ∧ b / c = l) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l700_70073


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l700_70078

/-- The perimeter of a rhombus given the lengths of its diagonals -/
theorem rhombus_perimeter (d1 d2 θ : ℝ) (h1 : d1 > 0) (h2 : d2 > 0) (h3 : 0 < θ ∧ θ < π) :
  ∃ (P : ℝ), P = 2 * Real.sqrt (d1^2 + d2^2) ∧ P > 0 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l700_70078


namespace NUMINAMATH_CALUDE_complex_distance_sum_l700_70093

/-- Given a complex number z satisfying |z - 3 - 2i| = 7, 
    prove that |z - 2 + i|^2 + |z - 11 - 5i|^2 = 554 -/
theorem complex_distance_sum (z : ℂ) (h : Complex.abs (z - (3 + 2*I)) = 7) : 
  (Complex.abs (z - (2 - I)))^2 + (Complex.abs (z - (11 + 5*I)))^2 = 554 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_sum_l700_70093


namespace NUMINAMATH_CALUDE_class_one_is_correct_l700_70031

/-- Represents the correct way to refer to a numbered class -/
inductive ClassReference
  | CardinalNumber (n : Nat)
  | OrdinalNumber (n : Nat)

/-- Checks if a class reference is correct -/
def is_correct_reference (ref : ClassReference) : Prop :=
  match ref with
  | ClassReference.CardinalNumber n => true
  | ClassReference.OrdinalNumber n => false

/-- The statement that "Class One" is the correct way to refer to the first class -/
theorem class_one_is_correct :
  is_correct_reference (ClassReference.CardinalNumber 1) = true :=
sorry


end NUMINAMATH_CALUDE_class_one_is_correct_l700_70031


namespace NUMINAMATH_CALUDE_johns_order_cost_l700_70002

/-- Calculates the discounted price of an order given the store's discount policy and purchase details. -/
def discountedPrice (itemPrice : ℕ) (itemCount : ℕ) (discountThreshold : ℕ) (discountRate : ℚ) : ℚ :=
  let totalPrice := itemPrice * itemCount
  let discountableAmount := max (totalPrice - discountThreshold) 0
  let discount := (discountableAmount : ℚ) * discountRate
  (totalPrice : ℚ) - discount

/-- Theorem stating that John's order costs $1360 after the discount. -/
theorem johns_order_cost :
  discountedPrice 200 7 1000 (1 / 10) = 1360 := by
  sorry

end NUMINAMATH_CALUDE_johns_order_cost_l700_70002


namespace NUMINAMATH_CALUDE_average_screen_time_l700_70090

/-- Calculates the average screen time per player in minutes given the screen times for 5 players in seconds -/
theorem average_screen_time (point_guard shooting_guard small_forward power_forward center : ℕ) 
  (h1 : point_guard = 130)
  (h2 : shooting_guard = 145)
  (h3 : small_forward = 85)
  (h4 : power_forward = 60)
  (h5 : center = 180) :
  (point_guard + shooting_guard + small_forward + power_forward + center) / (5 * 60) = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_screen_time_l700_70090


namespace NUMINAMATH_CALUDE_oliver_good_games_l700_70024

theorem oliver_good_games (total_games bad_games : ℕ) 
  (h1 : total_games = 11) 
  (h2 : bad_games = 5) : 
  total_games - bad_games = 6 := by
  sorry

end NUMINAMATH_CALUDE_oliver_good_games_l700_70024


namespace NUMINAMATH_CALUDE_find_M_l700_70075

theorem find_M : ∃ (M : ℕ), M > 0 ∧ 18^2 * 45^2 = 15^2 * M^2 ∧ M = 54 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l700_70075


namespace NUMINAMATH_CALUDE_student_age_problem_l700_70074

theorem student_age_problem (total_students : ℕ) (total_average_age : ℕ) 
  (group1_students : ℕ) (group1_average_age : ℕ) 
  (group2_students : ℕ) (group2_average_age : ℕ) :
  total_students = 20 →
  total_average_age = 20 →
  group1_students = 9 →
  group1_average_age = 11 →
  group2_students = 10 →
  group2_average_age = 24 →
  (total_students * total_average_age - 
   (group1_students * group1_average_age + group2_students * group2_average_age)) = 61 :=
by sorry

end NUMINAMATH_CALUDE_student_age_problem_l700_70074


namespace NUMINAMATH_CALUDE_january_salary_is_5300_l700_70066

/-- Represents monthly salaries -/
structure MonthlySalaries where
  J : ℕ  -- January
  F : ℕ  -- February
  M : ℕ  -- March
  A : ℕ  -- April
  Ma : ℕ -- May
  Ju : ℕ -- June

/-- Theorem stating the conditions and the result to be proved -/
theorem january_salary_is_5300 (s : MonthlySalaries) : 
  (s.J + s.F + s.M + s.A) / 4 = 8000 →
  (s.F + s.M + s.A + s.Ma) / 4 = 8300 →
  (s.M + s.A + s.Ma + s.Ju) / 4 = 8600 →
  s.Ma = 6500 →
  s.J = 5300 := by
  sorry

#check january_salary_is_5300

end NUMINAMATH_CALUDE_january_salary_is_5300_l700_70066


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l700_70041

theorem absolute_value_inequality (x : ℝ) :
  3 ≤ |x - 5| ∧ |x - 5| ≤ 10 ↔ (-5 ≤ x ∧ x ≤ 2) ∨ (8 ≤ x ∧ x ≤ 15) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l700_70041


namespace NUMINAMATH_CALUDE_ball_placement_count_ball_placement_proof_l700_70060

theorem ball_placement_count : ℕ :=
  let n_balls : ℕ := 5
  let n_boxes : ℕ := 4
  let ways_to_divide : ℕ := Nat.choose n_balls (n_balls - n_boxes + 1)
  let ways_to_arrange : ℕ := Nat.factorial n_boxes
  ways_to_divide * ways_to_arrange

theorem ball_placement_proof :
  ball_placement_count = 240 := by
  sorry

end NUMINAMATH_CALUDE_ball_placement_count_ball_placement_proof_l700_70060


namespace NUMINAMATH_CALUDE_harris_feeds_one_carrot_per_day_l700_70030

/-- Represents the number of carrots Harris feeds his dog per day -/
def carrots_per_day : ℚ :=
  let carrots_per_bag : ℕ := 5
  let cost_per_bag : ℚ := 2
  let annual_spend : ℚ := 146
  let days_per_year : ℕ := 365
  (annual_spend / days_per_year.cast) / cost_per_bag * carrots_per_bag

/-- Proves that Harris feeds his dog 1 carrot per day -/
theorem harris_feeds_one_carrot_per_day : 
  carrots_per_day = 1 := by sorry

end NUMINAMATH_CALUDE_harris_feeds_one_carrot_per_day_l700_70030


namespace NUMINAMATH_CALUDE_complex_number_problem_l700_70094

theorem complex_number_problem (z : ℂ) : 
  Complex.abs z = 1 ∧ 
  (∃ (y : ℝ), (3 + 4*I) * z = y * I) → 
  z = Complex.mk (-4/5) (-3/5) ∨ 
  z = Complex.mk (4/5) (3/5) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l700_70094


namespace NUMINAMATH_CALUDE_t_shirt_packages_l700_70037

theorem t_shirt_packages (total_shirts : ℕ) (shirts_per_package : ℕ) (h1 : total_shirts = 51) (h2 : shirts_per_package = 3) :
  total_shirts / shirts_per_package = 17 :=
by sorry

end NUMINAMATH_CALUDE_t_shirt_packages_l700_70037


namespace NUMINAMATH_CALUDE_a_squared_b_plus_ab_squared_equals_four_l700_70020

theorem a_squared_b_plus_ab_squared_equals_four :
  let a : ℝ := 2 + Real.sqrt 3
  let b : ℝ := 2 - Real.sqrt 3
  a^2 * b + a * b^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_a_squared_b_plus_ab_squared_equals_four_l700_70020


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l700_70036

def i : ℂ := Complex.I

theorem complex_expression_evaluation : i * (1 - 2*i) = 2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l700_70036


namespace NUMINAMATH_CALUDE_bisecting_line_sum_l700_70026

/-- Triangle DEF with vertices D(0, 10), E(4, 0), and F(10, 0) -/
structure Triangle :=
  (D : ℝ × ℝ)
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)

/-- A line defined by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

/-- Checks if a line bisects the area of a triangle -/
def bisects_area (t : Triangle) (l : Line) : Prop :=
  sorry

/-- The specific triangle DEF from the problem -/
def triangle_DEF : Triangle :=
  { D := (0, 10),
    E := (4, 0),
    F := (10, 0) }

/-- Main theorem: The line through E that bisects the area of triangle DEF
    has a slope and y-intercept whose sum is -15 -/
theorem bisecting_line_sum (l : Line) :
  bisects_area triangle_DEF l → l.slope + l.y_intercept = -15 :=
by sorry

end NUMINAMATH_CALUDE_bisecting_line_sum_l700_70026


namespace NUMINAMATH_CALUDE_square_difference_equality_l700_70064

theorem square_difference_equality : (45 + 15)^2 - (45^2 + 15^2) = 1350 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l700_70064


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l700_70035

theorem complex_equation_solutions :
  ∃ (s : Finset ℂ), (∀ z ∈ s, (z^4 + 1) / (z^2 - z - 2) = 0) ∧ s.card = 4 :=
by
  -- We define the numerator and denominator polynomials
  let num := fun (z : ℂ) ↦ z^4 + 1
  let den := fun (z : ℂ) ↦ z^2 - z - 2

  -- We assume the factorizations given in the problem
  have h_num : ∀ z, num z = (z^2 + Real.sqrt 2 * z + 1) * (z^2 - Real.sqrt 2 * z + 1) := by sorry
  have h_den : ∀ z, den z = (z - 2) * (z + 1) := by sorry

  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l700_70035


namespace NUMINAMATH_CALUDE_complex_power_eight_l700_70059

theorem complex_power_eight (z : ℂ) : z = (-Real.sqrt 3 + I) / 2 → z^8 = -1/2 - (Real.sqrt 3 / 2) * I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_eight_l700_70059


namespace NUMINAMATH_CALUDE_divisibility_implication_l700_70010

theorem divisibility_implication (u v : ℤ) : 
  (9 ∣ u^2 + u*v + v^2) → (3 ∣ u) ∧ (3 ∣ v) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l700_70010


namespace NUMINAMATH_CALUDE_zoo_ticket_sales_l700_70040

/-- Calculates the total money made from ticket sales at a zoo -/
theorem zoo_ticket_sales (total_people : ℕ) (adult_price kid_price : ℕ) (num_kids : ℕ) : 
  total_people = 254 → 
  adult_price = 28 → 
  kid_price = 12 → 
  num_kids = 203 → 
  (total_people - num_kids) * adult_price + num_kids * kid_price = 3864 := by
sorry

end NUMINAMATH_CALUDE_zoo_ticket_sales_l700_70040


namespace NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l700_70071

theorem sqrt_sum_greater_than_sqrt_of_sum : Real.sqrt 2 + Real.sqrt 3 > Real.sqrt (2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l700_70071


namespace NUMINAMATH_CALUDE_at_least_fifteen_equal_differences_l700_70088

theorem at_least_fifteen_equal_differences
  (a : Fin 100 → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_bounded : ∀ i, 1 ≤ a i ∧ a i ≤ 400)
  (h_increasing : ∀ i j, i < j → a i < a j) :
  ∃ (v : ℕ) (s : Finset (Fin 99)),
    s.card ≥ 15 ∧ ∀ i ∈ s, a (i + 1) - a i = v :=
sorry

end NUMINAMATH_CALUDE_at_least_fifteen_equal_differences_l700_70088


namespace NUMINAMATH_CALUDE_division_remainder_and_double_l700_70083

theorem division_remainder_and_double : 
  let dividend := 4509
  let divisor := 98
  let remainder := dividend % divisor
  let doubled_remainder := 2 * remainder
  remainder = 1 ∧ doubled_remainder = 2 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_and_double_l700_70083


namespace NUMINAMATH_CALUDE_train_crossing_time_l700_70021

/-- The time taken for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 150 ∧ 
  train_speed = 85 * (1000 / 3600) ∧ 
  man_speed = 5 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 6 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l700_70021


namespace NUMINAMATH_CALUDE_platform_length_l700_70068

/-- Calculates the length of a platform given train parameters -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 750)
  (h2 : time_platform = 97)
  (h3 : time_pole = 90) :
  ∃ (platform_length : ℝ), abs (platform_length - 58.33) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_l700_70068


namespace NUMINAMATH_CALUDE_pizza_pepperoni_ratio_l700_70003

/-- Represents a pizza with pepperoni slices -/
structure Pizza :=
  (total_pepperoni : ℕ)

/-- Represents a slice of pizza -/
structure PizzaSlice :=
  (pepperoni : ℕ)

def cut_pizza (p : Pizza) (slice1_pepperoni : ℕ) : PizzaSlice × PizzaSlice :=
  let slice1 := PizzaSlice.mk slice1_pepperoni
  let slice2 := PizzaSlice.mk (p.total_pepperoni - slice1_pepperoni)
  (slice1, slice2)

def pepperoni_ratio (slice1 : PizzaSlice) (slice2 : PizzaSlice) : ℚ :=
  slice1.pepperoni / slice2.pepperoni

theorem pizza_pepperoni_ratio :
  let original_pizza := Pizza.mk 40
  let (jellys_slice, other_slice) := cut_pizza original_pizza 10
  let jellys_slice_after_loss := PizzaSlice.mk (jellys_slice.pepperoni - 1)
  pepperoni_ratio jellys_slice_after_loss other_slice = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pepperoni_ratio_l700_70003


namespace NUMINAMATH_CALUDE_unit_cost_decrease_l700_70055

/-- Regression equation for unit product cost -/
def regression_equation (x : ℝ) : ℝ := 356 - 1.5 * x

/-- Theorem stating the relationship between output and unit product cost -/
theorem unit_cost_decrease (x : ℝ) :
  regression_equation (x + 1) = regression_equation x - 1.5 := by
  sorry

end NUMINAMATH_CALUDE_unit_cost_decrease_l700_70055


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l700_70058

theorem quadratic_inequality_solution_set :
  {x : ℝ | 3 * x^2 + 2 * x - 5 < 8} = {x : ℝ | -2 * Real.sqrt 10 / 6 - 1 / 3 < x ∧ x < 2 * Real.sqrt 10 / 6 - 1 / 3} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l700_70058


namespace NUMINAMATH_CALUDE_concatNaturalsDecimal_irrational_l700_70012

/-- The infinite decimal formed by concatenating all natural numbers in order after the decimal point -/
def concatNaturalsDecimal : ℝ :=
  sorry  -- Definition of the decimal (implementation details omitted)

/-- The infinite decimal formed by concatenating all natural numbers in order after the decimal point is irrational -/
theorem concatNaturalsDecimal_irrational : Irrational concatNaturalsDecimal := by
  sorry

end NUMINAMATH_CALUDE_concatNaturalsDecimal_irrational_l700_70012


namespace NUMINAMATH_CALUDE_weight_replacement_l700_70027

theorem weight_replacement (n : ℕ) (avg_increase weight_new : ℝ) :
  n = 8 →
  avg_increase = 2.5 →
  weight_new = 70 →
  weight_new - n * avg_increase = 50 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l700_70027


namespace NUMINAMATH_CALUDE_cubic_sum_of_roots_l700_70014

theorem cubic_sum_of_roots (r s : ℝ) : 
  r^2 - 5*r + 6 = 0 → 
  s^2 - 5*s + 6 = 0 → 
  r^3 + s^3 = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_of_roots_l700_70014

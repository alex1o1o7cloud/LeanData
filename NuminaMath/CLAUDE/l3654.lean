import Mathlib

namespace women_average_age_l3654_365464

/-- The average age of two women given specific conditions about a group of men --/
theorem women_average_age (n : ℕ) (A : ℝ) (age1 age2 : ℕ) (increase : ℝ) : 
  n = 10 ∧ age1 = 18 ∧ age2 = 22 ∧ increase = 6 →
  (n : ℝ) * (A + increase) - (n : ℝ) * A = 
    (((n : ℝ) * (A + increase) - (n : ℝ) * A + age1 + age2) / 2) * 2 - (age1 + age2) →
  ((n : ℝ) * (A + increase) - (n : ℝ) * A + age1 + age2) / 2 = 50 :=
by sorry

end women_average_age_l3654_365464


namespace second_number_is_984_l3654_365468

theorem second_number_is_984 (a b : ℕ) : 
  a < 10 ∧ b < 10 ∧ 
  a + b = 10 ∧
  (1000 + 300 + 10 * b + 7) % 11 = 0 →
  1000 + 300 + 10 * b + 7 - (400 + 10 * a + 3) = 984 := by
sorry

end second_number_is_984_l3654_365468


namespace counterexample_exists_l3654_365420

theorem counterexample_exists : ∃ n : ℕ, ¬ Nat.Prime n ∧ ¬ Nat.Prime (n - 3) ∧ n = 18 := by
  sorry

end counterexample_exists_l3654_365420


namespace exam_maximum_marks_l3654_365431

theorem exam_maximum_marks (percentage : ℝ) (scored_marks : ℝ) (max_marks : ℝ) : 
  percentage = 92 / 100 → 
  scored_marks = 460 → 
  percentage * max_marks = scored_marks → 
  max_marks = 500 := by
sorry

end exam_maximum_marks_l3654_365431


namespace amc10_paths_count_l3654_365480

/-- Represents the grid structure for spelling "AMC10" --/
structure AMC10Grid where
  a_to_m : Nat  -- Number of 'M's adjacent to central 'A'
  m_to_c : Nat  -- Number of 'C's adjacent to each 'M' (excluding path back to 'A')
  c_to_10 : Nat -- Number of '10' blocks adjacent to each 'C'

/-- Calculates the number of paths to spell "AMC10" in the given grid --/
def count_paths (grid : AMC10Grid) : Nat :=
  grid.a_to_m * grid.m_to_c * grid.c_to_10

/-- The specific grid configuration for the problem --/
def problem_grid : AMC10Grid :=
  { a_to_m := 4, m_to_c := 3, c_to_10 := 1 }

/-- Theorem stating that the number of paths to spell "AMC10" in the problem grid is 12 --/
theorem amc10_paths_count :
  count_paths problem_grid = 12 := by
  sorry

end amc10_paths_count_l3654_365480


namespace star_three_five_l3654_365479

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 + 2*x*y + y^2

-- Theorem statement
theorem star_three_five : star 3 5 = 64 := by
  sorry

end star_three_five_l3654_365479


namespace polynomial_factorization_1_l3654_365467

theorem polynomial_factorization_1 (a : ℝ) : 
  a^7 + a^5 + 1 = (a^2 + a + 1) * (a^5 - a^4 + a^3 - a + 1) := by sorry

end polynomial_factorization_1_l3654_365467


namespace find_N_l3654_365472

/-- Given three numbers a, b, and c, and a value N, satisfying certain conditions,
    prove that N = 41 is the integer solution that best satisfies all conditions. -/
theorem find_N : ∃ (a b c N : ℚ),
  a + b + c = 90 ∧
  a - 7 = N ∧
  b + 7 = N ∧
  5 * c = N ∧
  N.floor = 41 :=
by sorry

end find_N_l3654_365472


namespace min_coach_handshakes_zero_l3654_365476

/-- Represents the total number of handshakes in the gymnastics meet -/
def total_handshakes : ℕ := 325

/-- Calculates the number of handshakes between gymnasts given the total number of gymnasts -/
def gymnast_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the minimum number of coach handshakes is 0 -/
theorem min_coach_handshakes_zero :
  ∃ (n : ℕ), gymnast_handshakes n = total_handshakes ∧ n > 1 :=
sorry

end min_coach_handshakes_zero_l3654_365476


namespace point_movement_l3654_365432

/-- Given a point P in a Cartesian coordinate system, moving it upwards and to the left results in the expected new coordinates. -/
theorem point_movement (x y dx dy : ℤ) :
  let P : ℤ × ℤ := (x, y)
  let P' : ℤ × ℤ := (x - dx, y + dy)
  (P = (-2, 5) ∧ dx = 1 ∧ dy = 3) → P' = (-3, 8) := by
  sorry

#check point_movement

end point_movement_l3654_365432


namespace polynomial_remainder_theorem_l3654_365405

theorem polynomial_remainder_theorem (f : ℝ → ℝ) (a b c l m n p q r : ℝ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (∃ Q₁ : ℝ → ℝ, ∀ x, f x = (x - a) * (x - b) * Q₁ x + (p * x + l)) →
  (∃ Q₂ : ℝ → ℝ, ∀ x, f x = (x - b) * (x - c) * Q₂ x + (q * x + m)) →
  (∃ Q₃ : ℝ → ℝ, ∀ x, f x = (x - c) * (x - a) * Q₃ x + (r * x + n)) →
  l * (1 / a - 1 / b) + m * (1 / b - 1 / c) + n * (1 / c - 1 / a) = 0 := by
sorry

end polynomial_remainder_theorem_l3654_365405


namespace train_length_l3654_365458

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (length : ℝ) : 
  speed_kmh = 108 → time_sec = 50 → length = (speed_kmh * (5/18)) * time_sec → length = 1500 := by
  sorry

#check train_length

end train_length_l3654_365458


namespace absolute_value_inequality_l3654_365469

theorem absolute_value_inequality (x : ℝ) :
  |x - 2| + |x + 3| < 8 ↔ -4.5 < x ∧ x < 3.5 := by
  sorry

end absolute_value_inequality_l3654_365469


namespace constant_killing_time_l3654_365404

/-- The time it takes for lions to kill deers -/
def killing_time (n : ℕ) : ℝ :=
  14

/-- Given conditions -/
axiom condition_14 : killing_time 14 = 14
axiom condition_100 : killing_time 100 = 14

/-- Theorem: For any positive number of lions, it takes 14 minutes to kill the same number of deers -/
theorem constant_killing_time (n : ℕ) (h : n > 0) : killing_time n = 14 := by
  sorry

end constant_killing_time_l3654_365404


namespace password_count_l3654_365418

/-- The number of digits in the password -/
def password_length : ℕ := 4

/-- The number of available digits (0-9 excluding 7) -/
def available_digits : ℕ := 9

/-- The total number of possible passwords without restrictions -/
def total_passwords : ℕ := available_digits ^ password_length

/-- The number of ways to choose digits for a password with all different digits -/
def ways_to_choose_digits : ℕ := Nat.choose available_digits password_length

/-- The number of ways to arrange the chosen digits -/
def ways_to_arrange_digits : ℕ := Nat.factorial password_length

/-- The number of passwords with all different digits -/
def passwords_with_different_digits : ℕ := ways_to_choose_digits * ways_to_arrange_digits

/-- The number of passwords with at least two identical digits -/
def passwords_with_identical_digits : ℕ := total_passwords - passwords_with_different_digits

theorem password_count : passwords_with_identical_digits = 3537 := by
  sorry

end password_count_l3654_365418


namespace straw_hat_value_is_four_l3654_365461

/-- Represents the sheep problem scenario -/
structure SheepProblem where
  x : ℕ  -- number of sheep
  y : ℕ  -- number of times 10 yuan was taken
  z : ℕ  -- last amount taken by younger brother
  h1 : x^2 = x * x  -- price of each sheep equals number of sheep
  h2 : x^2 = 20 * y + 10 + z  -- total money distribution
  h3 : y ≥ 1  -- at least one round of 10 yuan taken
  h4 : z < 10  -- younger brother's last amount less than 10

/-- The value of the straw hat that equalizes the brothers' shares -/
def strawHatValue (p : SheepProblem) : ℕ := 10 - p.z

/-- Theorem stating the value of the straw hat is 4 yuan -/
theorem straw_hat_value_is_four (p : SheepProblem) : strawHatValue p = 4 := by
  sorry

#check straw_hat_value_is_four

end straw_hat_value_is_four_l3654_365461


namespace line_opposite_sides_range_l3654_365402

/-- Given that the points (1, 1) and (0, 1) are on opposite sides of the line 3x - 2y + a = 0,
    the range of values for a is (-1, 2). -/
theorem line_opposite_sides_range (a : ℝ) : 
  (∃ (x y : ℝ), (x = 1 ∧ y = 1) ∨ (x = 0 ∧ y = 1)) →
  ((3 * 1 - 2 * 1 + a) * (3 * 0 - 2 * 1 + a) < 0) →
  a ∈ Set.Ioo (-1 : ℝ) 2 :=
by sorry

end line_opposite_sides_range_l3654_365402


namespace card_relationship_l3654_365457

theorem card_relationship (c : ℝ) (h1 : c > 0) : 
  let b := 1.2 * c
  let d := 1.4 * b
  d = 1.68 * c := by sorry

end card_relationship_l3654_365457


namespace average_temperature_l3654_365444

def temperatures : List ℝ := [52, 64, 59, 60, 47]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 56.4 := by
  sorry

end average_temperature_l3654_365444


namespace arithmetic_mean_problem_l3654_365416

theorem arithmetic_mean_problem (a b c : ℝ) :
  let numbers := [a, b, c, 108]
  (numbers.sum / numbers.length = 92) →
  ((a + b + c) / 3 = 260 / 3) := by
sorry

end arithmetic_mean_problem_l3654_365416


namespace min_value_expression_min_value_attained_l3654_365411

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 / (y - 1)) + (y^2 / (x - 1)) + (x + y) ≥ 12 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 1 ∧ y > 1 ∧
  (x^2 / (y - 1)) + (y^2 / (x - 1)) + (x + y) < 12 + ε :=
by sorry

end min_value_expression_min_value_attained_l3654_365411


namespace speed_ratio_l3654_365438

/-- Represents the position and speed of an object moving in a straight line. -/
structure Mover where
  speed : ℝ
  initialPosition : ℝ

/-- The problem setup -/
def problem (a b : Mover) : Prop :=
  -- A and B move uniformly along two straight paths intersecting at right angles at point O
  -- When A is at O, B is 400 yards short of O
  a.initialPosition = 0 ∧ b.initialPosition = -400 ∧
  -- In 3 minutes, they are equidistant from O
  (3 * a.speed)^2 = (-400 + 3 * b.speed)^2 ∧
  -- In 10 minutes (3 + 7 minutes), they are again equidistant from O
  (10 * a.speed)^2 = (-400 + 10 * b.speed)^2

/-- The theorem to be proved -/
theorem speed_ratio (a b : Mover) :
  problem a b → a.speed / b.speed = 5 / 6 := by
  sorry

end speed_ratio_l3654_365438


namespace sequence_integer_count_l3654_365474

def sequence_term (n : ℕ) : ℚ :=
  9720 / 2^n

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n))) ∧
  (∀ (k : ℕ), k > 0 →
    ((∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
     (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n)))
    → k = 4) :=
by sorry

end sequence_integer_count_l3654_365474


namespace bee_swarm_size_l3654_365448

theorem bee_swarm_size :
  ∃ n : ℕ,
    n > 0 ∧
    (n : ℝ) = (Real.sqrt ((n : ℝ) / 2)) + (8 / 9 * n) + 1 ∧
    n = 72 := by
  sorry

end bee_swarm_size_l3654_365448


namespace point_symmetry_wrt_origin_l3654_365473

/-- Given a point M with coordinates (-2,3), its coordinates with respect to the origin are (2,-3). -/
theorem point_symmetry_wrt_origin : 
  let M : ℝ × ℝ := (-2, 3)
  (- M.1, - M.2) = (2, -3) := by sorry

end point_symmetry_wrt_origin_l3654_365473


namespace intersection_condition_l3654_365442

/-- Curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

/-- Curve C₂ in Cartesian coordinates -/
def C₂ (x y : ℝ) : Prop := y = x

/-- C₂ translated downward by m units -/
def C₂_translated (x y m : ℝ) : Prop := y = x - m

/-- Two points in common between C₁ and translated C₂ -/
def two_intersections (m : ℝ) : Prop :=
  ∃! (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ 
    C₁ p₁.1 p₁.2 ∧ C₂_translated p₁.1 p₁.2 m ∧
    C₁ p₂.1 p₂.2 ∧ C₂_translated p₂.1 p₂.2 m

/-- Main theorem -/
theorem intersection_condition (m : ℝ) :
  (m > 0 ∧ two_intersections m) ↔ (4 ≤ m ∧ m < 2 + 2 * Real.sqrt 2) :=
sorry

end intersection_condition_l3654_365442


namespace ellie_distance_after_six_steps_l3654_365412

/-- The distance Ellie walks after n steps, starting from 0 and aiming for a target 5 meters away,
    walking 1/4 of the remaining distance with each step. -/
def ellieDistance (n : ℕ) : ℚ :=
  5 * (1 - (3/4)^n)

/-- Theorem stating that after 6 steps, Ellie has walked 16835/4096 meters. -/
theorem ellie_distance_after_six_steps :
  ellieDistance 6 = 16835 / 4096 := by
  sorry


end ellie_distance_after_six_steps_l3654_365412


namespace least_positive_linear_combination_l3654_365498

theorem least_positive_linear_combination : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (∃ (x y : ℤ), 24 * x + 18 * y = m) → n ≤ m) ∧ 
  (∃ (x y : ℤ), 24 * x + 18 * y = n) :=
by sorry

end least_positive_linear_combination_l3654_365498


namespace man_speed_man_speed_specific_case_l3654_365440

/-- Calculates the speed of a man running opposite to a train, given the train's length, speed, and time to pass the man. -/
theorem man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / time_to_pass
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * 3600 / 1000
  man_speed_kmh

/-- Proves that the speed of the man is approximately 6 km/hr given the specific conditions. -/
theorem man_speed_specific_case : 
  ∃ ε > 0, |man_speed 110 84 4.399648028157747 - 6| < ε :=
sorry

end man_speed_man_speed_specific_case_l3654_365440


namespace quadratic_inequality_solution_set_l3654_365488

/-- Given that the solution set of ax^2 - 1999x + b > 0 is {x | -3 < x < -1},
    prove that the solution set of ax^2 + 1999x + b > 0 is {x | 1 < x < 3} -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x : ℝ, (a * x^2 - 1999 * x + b > 0) ↔ (-3 < x ∧ x < -1)) :
  ∀ x : ℝ, (a * x^2 + 1999 * x + b > 0) ↔ (1 < x ∧ x < 3) := by
  sorry


end quadratic_inequality_solution_set_l3654_365488


namespace canoe_rowing_probability_l3654_365441

theorem canoe_rowing_probability (p : ℝ) (h_p : p = 3/5) :
  let q := 1 - p
  p * p + p * q + q * p = 21/25 := by sorry

end canoe_rowing_probability_l3654_365441


namespace stratified_sampling_medium_stores_l3654_365429

/-- Calculates the number of medium stores to be drawn in stratified sampling -/
def medium_stores_drawn (total_stores : ℕ) (medium_stores : ℕ) (sample_size : ℕ) : ℕ :=
  (medium_stores * sample_size) / total_stores

theorem stratified_sampling_medium_stores 
  (total_stores : ℕ) (medium_stores : ℕ) (sample_size : ℕ)
  (h1 : total_stores = 300)
  (h2 : medium_stores = 75)
  (h3 : sample_size = 20) :
  medium_stores_drawn total_stores medium_stores sample_size = 5 := by
sorry

#eval medium_stores_drawn 300 75 20

end stratified_sampling_medium_stores_l3654_365429


namespace triangle_properties_l3654_365407

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle --/
def TriangleConditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B = 3 ∧
  t.b * Real.sin t.A = 4 ∧
  (1/2) * t.a * t.b * Real.sin t.C = 10

/-- Theorem stating the length of side a and the perimeter of the triangle --/
theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.a = 5 ∧ t.a + t.b + t.c = 10 + 2 * Real.sqrt 5 := by
  sorry

end triangle_properties_l3654_365407


namespace jacket_cost_calculation_l3654_365423

/-- The amount spent on shorts -/
def shorts_cost : ℚ := 1428 / 100

/-- The total amount spent on clothing -/
def total_cost : ℚ := 1902 / 100

/-- The amount spent on the jacket -/
def jacket_cost : ℚ := total_cost - shorts_cost

theorem jacket_cost_calculation : jacket_cost = 474 / 100 := by
  sorry

end jacket_cost_calculation_l3654_365423


namespace max_intersections_seven_segments_l3654_365414

/-- A closed polyline with a given number of segments. -/
structure ClosedPolyline :=
  (segments : ℕ)

/-- The maximum number of self-intersection points for a closed polyline. -/
def max_self_intersections (p : ClosedPolyline) : ℕ :=
  (p.segments * (p.segments - 3)) / 2

/-- Theorem: The maximum number of self-intersection points in a closed polyline with 7 segments is 14. -/
theorem max_intersections_seven_segments :
  ∃ (p : ClosedPolyline), p.segments = 7 ∧ max_self_intersections p = 14 :=
sorry

end max_intersections_seven_segments_l3654_365414


namespace valid_triples_eq_solution_set_l3654_365421

/-- Represents a triple of side lengths (a, b, c) of a triangle -/
structure TriangleSides where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The set of all valid triangle side triples satisfying the given conditions -/
def validTriples : Set TriangleSides := {t | 
  t.a ≤ t.b ∧ t.b ≤ t.c ∧  -- a ≤ b ≤ c
  t.b ^ 2 = t.a * t.c ∧    -- geometric progression
  (t.a = 100 ∨ t.c = 100)  -- at least one of a or c is 100
}

/-- The set of all solutions given in the problem -/
def solutionSet : Set TriangleSides := {
  ⟨49, 70, 100⟩, ⟨64, 80, 100⟩, ⟨81, 90, 100⟩, ⟨100, 100, 100⟩,
  ⟨100, 110, 121⟩, ⟨100, 120, 144⟩, ⟨100, 130, 169⟩, ⟨100, 140, 196⟩,
  ⟨100, 150, 225⟩, ⟨100, 160, 256⟩
}

/-- Theorem stating that the set of valid triples is exactly the solution set -/
theorem valid_triples_eq_solution_set : validTriples = solutionSet := by
  sorry

end valid_triples_eq_solution_set_l3654_365421


namespace transistor_growth_1992_to_2004_l3654_365481

/-- Moore's Law: Number of transistors doubles every 2 years -/
def moores_law (initial_transistors : ℕ) (years : ℕ) : ℕ :=
  initial_transistors * 2^(years / 2)

/-- Theorem: A CPU with 2,000,000 transistors in 1992 would have 128,000,000 transistors in 2004 -/
theorem transistor_growth_1992_to_2004 :
  moores_law 2000000 (2004 - 1992) = 128000000 := by
  sorry

#eval moores_law 2000000 (2004 - 1992)

end transistor_growth_1992_to_2004_l3654_365481


namespace expression_value_l3654_365443

theorem expression_value : (3 * 12 + 18) / (6 - 3) = 18 := by
  sorry

end expression_value_l3654_365443


namespace energy_change_in_triangle_l3654_365490

/-- The energy stored between two point charges -/
def energy_between_charges (distance : ℝ) : ℝ := sorry

/-- The total energy stored in a system of three point charges -/
def total_energy (d1 d2 d3 : ℝ) : ℝ := 
  energy_between_charges d1 + energy_between_charges d2 + energy_between_charges d3

theorem energy_change_in_triangle (initial_energy : ℝ) :
  initial_energy = 18 →
  ∃ (energy_func : ℝ → ℝ),
    (energy_func 1 + energy_func 1 + energy_func (Real.sqrt 2) = initial_energy) ∧
    (energy_func 1 + energy_func (Real.sqrt 2 / 2) + energy_func (Real.sqrt 2 / 2) = 6 + 12 * Real.sqrt 2) := by
  sorry

#check energy_change_in_triangle

end energy_change_in_triangle_l3654_365490


namespace no_real_solution_l3654_365419

theorem no_real_solution :
  ¬∃ (a b c d : ℝ), 
    a^3 + c^3 = 2 ∧
    a^2 * b + c^2 * d = 0 ∧
    b^3 + d^3 = 1 ∧
    a * b^2 + c * d^2 = -6 := by
  sorry

end no_real_solution_l3654_365419


namespace a_100_equals_116_l3654_365408

/-- Sequence of positive integers not divisible by 7 -/
def a : ℕ → ℕ :=
  λ n => (n + (n - 1) / 6) + 1

theorem a_100_equals_116 : a 100 = 116 := by
  sorry

end a_100_equals_116_l3654_365408


namespace cubic_equation_solution_l3654_365445

theorem cubic_equation_solution (a : ℝ) : a^3 = 21 * 25 * 35 * 63 → a = 105 := by
  sorry

end cubic_equation_solution_l3654_365445


namespace five_balls_three_boxes_l3654_365491

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 56 := by
  sorry

end five_balls_three_boxes_l3654_365491


namespace correct_stratified_sample_l3654_365463

/-- Represents the composition of a student body -/
structure StudentBody where
  total : ℕ
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  sum_eq_total : freshmen + sophomores + juniors = total

/-- Represents a stratified sample from a student body -/
structure StratifiedSample where
  body : StudentBody
  sample_size : ℕ
  sampled_freshmen : ℕ
  sampled_sophomores : ℕ
  sampled_juniors : ℕ
  sum_eq_sample_size : sampled_freshmen + sampled_sophomores + sampled_juniors = sample_size

/-- Checks if a stratified sample is proportionally correct -/
def is_proportional_sample (sample : StratifiedSample) : Prop :=
  sample.sampled_freshmen * sample.body.total = sample.body.freshmen * sample.sample_size ∧
  sample.sampled_sophomores * sample.body.total = sample.body.sophomores * sample.sample_size ∧
  sample.sampled_juniors * sample.body.total = sample.body.juniors * sample.sample_size

theorem correct_stratified_sample :
  let school : StudentBody := {
    total := 1000,
    freshmen := 400,
    sophomores := 340,
    juniors := 260,
    sum_eq_total := by sorry
  }
  let sample : StratifiedSample := {
    body := school,
    sample_size := 50,
    sampled_freshmen := 20,
    sampled_sophomores := 17,
    sampled_juniors := 13,
    sum_eq_sample_size := by sorry
  }
  is_proportional_sample sample := by sorry

end correct_stratified_sample_l3654_365463


namespace circle_radius_from_area_circumference_difference_l3654_365495

theorem circle_radius_from_area_circumference_difference 
  (x y : ℝ) (h : x - y = 72 * Real.pi) : ∃ r : ℝ, r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 12 := by
  sorry

end circle_radius_from_area_circumference_difference_l3654_365495


namespace sum_of_divisors_of_twelve_l3654_365433

theorem sum_of_divisors_of_twelve : (Finset.filter (λ x => 12 % x = 0) (Finset.range 13)).sum id = 28 := by
  sorry

end sum_of_divisors_of_twelve_l3654_365433


namespace orange_theft_ratio_l3654_365450

/-- Proves the ratio of stolen oranges to remaining oranges is 1:2 --/
theorem orange_theft_ratio :
  ∀ (initial_oranges eaten_oranges returned_oranges final_oranges : ℕ),
    initial_oranges = 60 →
    eaten_oranges = 10 →
    returned_oranges = 5 →
    final_oranges = 30 →
    ∃ (stolen_oranges : ℕ),
      stolen_oranges = initial_oranges - eaten_oranges - (final_oranges - returned_oranges) ∧
      2 * stolen_oranges = initial_oranges - eaten_oranges :=
by
  sorry

#check orange_theft_ratio

end orange_theft_ratio_l3654_365450


namespace second_turkey_weight_proof_l3654_365482

/-- The weight of the second turkey in kilograms -/
def second_turkey_weight : ℝ := 9

/-- The total cost of all turkeys in dollars -/
def total_cost : ℝ := 66

/-- The cost of turkey per kilogram in dollars -/
def cost_per_kg : ℝ := 2

/-- The weight of the first turkey in kilograms -/
def first_turkey_weight : ℝ := 6

theorem second_turkey_weight_proof :
  second_turkey_weight = 9 :=
by
  have h1 : total_cost = (first_turkey_weight + second_turkey_weight + 2 * second_turkey_weight) * cost_per_kg :=
    sorry
  have h2 : total_cost = (6 + 3 * second_turkey_weight) * 2 :=
    sorry
  have h3 : 66 = (6 + 3 * second_turkey_weight) * 2 :=
    sorry
  have h4 : 33 = 6 + 3 * second_turkey_weight :=
    sorry
  have h5 : 27 = 3 * second_turkey_weight :=
    sorry
  sorry

end second_turkey_weight_proof_l3654_365482


namespace hyperbola_properties_l3654_365434

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0
  h_asymptote : b / a = Real.sqrt 3
  h_vertex : a = 1

/-- Represents a point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The hyperbola equation -/
def hyperbola_eq (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- The line equation -/
def line_eq (m : ℝ) (p : Point) : Prop :=
  p.y = p.x + m

/-- Theorem stating the properties of the hyperbola and its intersection with a line -/
theorem hyperbola_properties (h : Hyperbola) (m : ℝ) (A B M : Point) 
    (h_distinct : A ≠ B)
    (h_intersect_A : hyperbola_eq h A ∧ line_eq m A)
    (h_intersect_B : hyperbola_eq h B ∧ line_eq m B)
    (h_midpoint : M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2)
    (h_nonzero : M.x ≠ 0) :
  (h.a = 1 ∧ h.b = Real.sqrt 3) ∧ M.y / M.x = 3 := by sorry

end hyperbola_properties_l3654_365434


namespace competition_outcomes_l3654_365428

/-- The number of possible outcomes for champions in a competition -/
def num_outcomes (num_students : ℕ) (num_events : ℕ) : ℕ :=
  num_students ^ num_events

/-- Theorem: Given 3 students competing in 2 events, where each event has one champion,
    the total number of possible outcomes for the champions is 9. -/
theorem competition_outcomes :
  num_outcomes 3 2 = 9 := by
  sorry

end competition_outcomes_l3654_365428


namespace probability_ten_people_no_adjacent_standing_l3654_365439

/-- Represents the number of valid arrangements for n people where no two adjacent people are standing --/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The probability of no two adjacent people standing in a circular arrangement of n people --/
def probabilityNoAdjacentStanding (n : ℕ) : ℚ :=
  validArrangements n / (2 ^ n : ℚ)

theorem probability_ten_people_no_adjacent_standing :
  probabilityNoAdjacentStanding 10 = 123 / 1024 := by
  sorry


end probability_ten_people_no_adjacent_standing_l3654_365439


namespace work_completion_time_l3654_365454

theorem work_completion_time 
  (total_work : ℝ) 
  (p_q_together_time : ℝ) 
  (p_alone_time : ℝ) 
  (h1 : p_q_together_time = 6)
  (h2 : p_alone_time = 15)
  : ∃ q_alone_time : ℝ, q_alone_time = 10 ∧ 
    (1 / p_q_together_time = 1 / p_alone_time + 1 / q_alone_time) :=
by sorry

end work_completion_time_l3654_365454


namespace area_triangle_PQR_l3654_365430

/-- Square pyramid with given dimensions and points --/
structure SquarePyramid where
  baseSide : ℝ
  altitude : ℝ
  P : ℝ  -- Distance from W to P along WO
  Q : ℝ  -- Distance from Y to Q along YO
  R : ℝ  -- Distance from X to R along XO

/-- Theorem stating the area of triangle PQR in the given square pyramid --/
theorem area_triangle_PQR (pyramid : SquarePyramid)
  (h1 : pyramid.baseSide = 4)
  (h2 : pyramid.altitude = 8)
  (h3 : pyramid.P = 1/4 * (pyramid.baseSide * Real.sqrt 2 / 2))
  (h4 : pyramid.Q = 1/2 * (pyramid.baseSide * Real.sqrt 2 / 2))
  (h5 : pyramid.R = 3/4 * (pyramid.baseSide * Real.sqrt 2 / 2)) :
  let WO := Real.sqrt ((pyramid.baseSide * Real.sqrt 2 / 2)^2 + pyramid.altitude^2)
  let PQ := pyramid.Q - pyramid.P
  let RQ := pyramid.R - pyramid.Q
  1/2 * PQ * RQ = 2.25 := by
  sorry

end area_triangle_PQR_l3654_365430


namespace larger_field_time_calculation_l3654_365497

-- Define the smaller field's dimensions
def small_width : ℝ := 1  -- We can use any positive real number as the base
def small_length : ℝ := 1.5 * small_width

-- Define the larger field's dimensions
def large_width : ℝ := 4 * small_width
def large_length : ℝ := 3 * small_length

-- Define the perimeters
def small_perimeter : ℝ := 2 * (small_length + small_width)
def large_perimeter : ℝ := 2 * (large_length + large_width)

-- Define the time to complete one round of the smaller field
def small_field_time : ℝ := 20

-- Theorem to prove
theorem larger_field_time_calculation :
  (large_perimeter / small_perimeter) * small_field_time = 68 := by
  sorry

end larger_field_time_calculation_l3654_365497


namespace complex_number_problem_l3654_365449

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_problem (z : ℂ) 
  (h1 : is_purely_imaginary z) 
  (h2 : is_purely_imaginary ((z + 2)^2 - 8*I)) : 
  z = -2*I := by sorry

end complex_number_problem_l3654_365449


namespace max_colored_cells_4x3000_exists_optimal_board_l3654_365426

/-- A tetromino is a geometric shape composed of four square cells connected orthogonally. -/
def Tetromino : Type := Unit

/-- A board is represented as a 2D array of boolean values, where true represents a colored cell. -/
def Board : Type := Array (Array Bool)

/-- Check if a given board contains a tetromino. -/
def containsTetromino (board : Board) : Bool :=
  sorry

/-- Count the number of colored cells in a board. -/
def countColoredCells (board : Board) : Nat :=
  sorry

/-- Create a 4 × 3000 board. -/
def create4x3000Board : Board :=
  sorry

/-- The main theorem stating the maximum number of cells that can be colored. -/
theorem max_colored_cells_4x3000 :
  ∀ (board : Board),
    board = create4x3000Board →
    ¬containsTetromino board →
    countColoredCells board ≤ 7000 :=
  sorry

/-- The existence of a board with exactly 7000 colored cells and no tetromino. -/
theorem exists_optimal_board :
  ∃ (board : Board),
    board = create4x3000Board ∧
    ¬containsTetromino board ∧
    countColoredCells board = 7000 :=
  sorry

end max_colored_cells_4x3000_exists_optimal_board_l3654_365426


namespace unique_six_digit_number_l3654_365413

/-- A function that returns the set of digits of a natural number -/
def digits (n : ℕ) : Finset ℕ :=
  sorry

/-- A function that checks if a natural number is a six-digit number -/
def isSixDigit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

/-- The theorem stating that 142857 is the unique six-digit number satisfying the given conditions -/
theorem unique_six_digit_number :
  ∃! p : ℕ, isSixDigit p ∧
    (∀ i : Fin 6, isSixDigit ((i.val + 1) * p)) ∧
    (∀ i : Fin 6, digits ((i.val + 1) * p) = digits p) ∧
    p = 142857 :=
  sorry

end unique_six_digit_number_l3654_365413


namespace min_value_of_m_l3654_365478

theorem min_value_of_m (x y : ℝ) (h1 : y = x^2 - 2) (h2 : x > Real.sqrt 3) :
  let m := (3*x + y - 4)/(x - 1) + (x + 3*y - 4)/(y - 1)
  m ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), y₀ = x₀^2 - 2 ∧ x₀ > Real.sqrt 3 ∧
    (3*x₀ + y₀ - 4)/(x₀ - 1) + (x₀ + 3*y₀ - 4)/(y₀ - 1) = 8 :=
by sorry

end min_value_of_m_l3654_365478


namespace initial_deficit_calculation_l3654_365453

/-- Represents the score difference at the start of the final quarter -/
def initial_deficit : ℤ := sorry

/-- Liz's free throw points -/
def free_throw_points : ℕ := 5

/-- Liz's three-pointer points -/
def three_pointer_points : ℕ := 9

/-- Liz's jump shot points -/
def jump_shot_points : ℕ := 8

/-- Other team's points in the final quarter -/
def other_team_points : ℕ := 10

/-- Final score difference (negative means Liz's team lost) -/
def final_score_difference : ℤ := -8

theorem initial_deficit_calculation :
  initial_deficit = 20 :=
by sorry

end initial_deficit_calculation_l3654_365453


namespace ellipse_hyperbola_foci_coincide_l3654_365427

/-- The squared semi-major axis of the ellipse -/
def a_squared_ellipse : ℝ := 25

/-- The squared semi-major axis of the hyperbola -/
def a_squared_hyperbola : ℝ := 196

/-- The squared semi-minor axis of the hyperbola -/
def b_squared_hyperbola : ℝ := 121

/-- The equation of the ellipse -/
def ellipse_equation (x y b : ℝ) : Prop :=
  x^2 / a_squared_ellipse + y^2 / b^2 = 1

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / a_squared_hyperbola - y^2 / b_squared_hyperbola = 1/49

/-- The theorem stating that if the foci of the ellipse and hyperbola coincide,
    then the squared semi-minor axis of the ellipse is 908/49 -/
theorem ellipse_hyperbola_foci_coincide :
  ∃ b : ℝ, (∀ x y : ℝ, ellipse_equation x y b ↔ hyperbola_equation x y) →
    b^2 = 908/49 := by sorry

end ellipse_hyperbola_foci_coincide_l3654_365427


namespace enrollment_difference_l3654_365496

def maple_ridge_enrollment : ℕ := 1500
def south_park_enrollment : ℕ := 2100
def lakeside_enrollment : ℕ := 2700
def riverdale_enrollment : ℕ := 1800
def brookwood_enrollment : ℕ := 900

def school_enrollments : List ℕ := [
  maple_ridge_enrollment,
  south_park_enrollment,
  lakeside_enrollment,
  riverdale_enrollment,
  brookwood_enrollment
]

theorem enrollment_difference : 
  (List.maximum school_enrollments).get! - (List.minimum school_enrollments).get! = 1800 := by
  sorry

end enrollment_difference_l3654_365496


namespace revenue_decrease_l3654_365477

theorem revenue_decrease (current_revenue : ℝ) (decrease_percentage : ℝ) (original_revenue : ℝ) : 
  current_revenue = 48.0 ∧ 
  decrease_percentage = 33.33333333333333 / 100 ∧
  current_revenue = original_revenue * (1 - decrease_percentage) →
  original_revenue = 72.0 := by
sorry

end revenue_decrease_l3654_365477


namespace flower_bed_fraction_is_correct_l3654_365486

/-- Represents the dimensions and areas of a yard with flower beds -/
structure YardWithFlowerBeds where
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ
  trapezoid_height : ℝ
  total_length : ℝ

/-- Calculates the fraction of the yard occupied by flower beds -/
def flower_bed_fraction (yard : YardWithFlowerBeds) : ℚ :=
  125 / 310

/-- Theorem stating that the fraction of the yard occupied by flower beds is 125/310 -/
theorem flower_bed_fraction_is_correct (yard : YardWithFlowerBeds) 
  (h1 : yard.trapezoid_short_side = 30)
  (h2 : yard.trapezoid_long_side = 40)
  (h3 : yard.trapezoid_height = 6)
  (h4 : yard.total_length = 60) : 
  flower_bed_fraction yard = 125 / 310 := by
  sorry

end flower_bed_fraction_is_correct_l3654_365486


namespace man_downstream_speed_l3654_365483

/-- The speed of a man rowing in a stream -/
structure RowingSpeed :=
  (still : ℝ)        -- Speed in still water
  (upstream : ℝ)     -- Speed upstream
  (downstream : ℝ)   -- Speed downstream

/-- Calculate the downstream speed given still water and upstream speeds -/
def calculate_downstream_speed (s : RowingSpeed) : Prop :=
  s.downstream = s.still + (s.still - s.upstream)

/-- Theorem: The man's downstream speed is 55 kmph -/
theorem man_downstream_speed :
  ∃ (s : RowingSpeed), s.still = 50 ∧ s.upstream = 45 ∧ s.downstream = 55 ∧ calculate_downstream_speed s :=
sorry

end man_downstream_speed_l3654_365483


namespace triangle_max_area_l3654_365424

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C,
    if (a-b+c)/c = b/(a+b-c) and a = 2, then the maximum area of triangle ABC is √3. -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  (a - b + c) / c = b / (a + b - c) →
  a = 2 →
  ∃ (S : ℝ), S ≤ Real.sqrt 3 ∧ 
    (∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ S) ∧
    (∃ (b' c' : ℝ), (1/2) * b' * c' * Real.sin A = Real.sqrt 3) :=
by sorry

end triangle_max_area_l3654_365424


namespace area_ratio_second_third_neighbor_octagons_l3654_365465

/-- A regular octagon -/
structure RegularOctagon where
  -- Add necessary fields

/-- The octagon formed by connecting second neighboring vertices -/
def secondNeighborOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The octagon formed by connecting third neighboring vertices -/
def thirdNeighborOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- The theorem stating the ratio of areas -/
theorem area_ratio_second_third_neighbor_octagons (o : RegularOctagon) :
  area (secondNeighborOctagon o) / area (thirdNeighborOctagon o) = 2 + Real.sqrt 2 := by
  sorry

end area_ratio_second_third_neighbor_octagons_l3654_365465


namespace concave_integral_inequality_l3654_365406

open Set Function MeasureTheory

variable {m : ℕ}
variable (P : Set (EuclideanSpace ℝ (Fin m)))
variable (f : EuclideanSpace ℝ (Fin m) → ℝ)
variable (ξ : EuclideanSpace ℝ (Fin m))

theorem concave_integral_inequality
  (h_nonempty : Set.Nonempty P)
  (h_compact : IsCompact P)
  (h_convex : Convex ℝ P)
  (h_concave : ConcaveOn ℝ P f)
  (h_nonneg : ∀ x ∈ P, 0 ≤ f x) :
  ∫ x in P, ⟪ξ, x⟫_ℝ * f x ≤ 
    ((m + 1 : ℝ) / (m + 2 : ℝ) * ⨆ (x : EuclideanSpace ℝ (Fin m)) (h : x ∈ P), ⟪ξ, x⟫_ℝ + 
     (1 : ℝ) / (m + 2 : ℝ) * ⨅ (x : EuclideanSpace ℝ (Fin m)) (h : x ∈ P), ⟪ξ, x⟫_ℝ) * 
    ∫ x in P, f x :=
sorry

end concave_integral_inequality_l3654_365406


namespace log_equation_solution_l3654_365446

theorem log_equation_solution :
  ∃ x : ℝ, (Real.log x + 3 * Real.log 2 - 4 * Real.log 5 = 1) ∧ (x = 781.25) := by
  sorry

end log_equation_solution_l3654_365446


namespace leyden_quadruple_theorem_l3654_365470

/-- Definition of a Leyden quadruple -/
structure LeydenQuadruple where
  p : ℕ
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ

/-- The main theorem about Leyden quadruples -/
theorem leyden_quadruple_theorem (q : LeydenQuadruple) :
  (q.a₁ + q.a₂ + q.a₃) / 3 = q.p + 2 ↔ q.p = 5 := by
  sorry

end leyden_quadruple_theorem_l3654_365470


namespace luke_spent_eleven_l3654_365484

/-- The amount of money Luke spent, given his initial amount, 
    the amount he received, and his current amount. -/
def money_spent (initial amount_received current : ℕ) : ℕ :=
  initial + amount_received - current

/-- Theorem stating that Luke spent $11 -/
theorem luke_spent_eleven : 
  money_spent 48 21 58 = 11 := by sorry

end luke_spent_eleven_l3654_365484


namespace rectangle_perimeter_l3654_365452

theorem rectangle_perimeter (a b : ℕ) : 
  a ≠ b →  -- non-square condition
  a * b - 3 * (a + b) = 3 * a * b - 9 →  -- given equation
  2 * (a + b) = 14 :=  -- perimeter = 14
by sorry

end rectangle_perimeter_l3654_365452


namespace senior_junior_ratio_l3654_365417

theorem senior_junior_ratio (j k : ℕ) (hj : j > 0) (hk : k > 0)
  (h_junior_contestants : (3 * j) / 5 = (j * 3) / 5)
  (h_senior_contestants : k / 5 = (k * 1) / 5)
  (h_equal_contestants : (3 * j) / 5 = k / 5) :
  k = 3 * j :=
sorry

end senior_junior_ratio_l3654_365417


namespace oliver_money_result_l3654_365436

/-- Calculates the remaining money after Oliver's transactions -/
def oliver_money (initial : ℝ) (feb_spend_percent : ℝ) (march_add : ℝ) (final_spend_percent : ℝ) : ℝ :=
  let after_feb := initial * (1 - feb_spend_percent)
  let after_march := after_feb + march_add
  after_march * (1 - final_spend_percent)

/-- Theorem stating that Oliver's remaining money is $54.04 -/
theorem oliver_money_result :
  oliver_money 33 0.15 32 0.10 = 54.04 := by
  sorry

end oliver_money_result_l3654_365436


namespace bacteria_growth_l3654_365485

theorem bacteria_growth (initial_count : ℕ) : 
  (initial_count * (4 ^ 15) = 4194304) → initial_count = 1 := by
  sorry

end bacteria_growth_l3654_365485


namespace n_minus_m_equals_six_l3654_365475

-- Define the sets M and N
def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 3, 6}

-- Define the set difference operation
def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- Theorem statement
theorem n_minus_m_equals_six : set_difference N M = {6} := by
  sorry

end n_minus_m_equals_six_l3654_365475


namespace zoo_enclosure_claws_l3654_365460

theorem zoo_enclosure_claws (num_wombats : ℕ) (num_rheas : ℕ) 
  (wombat_claws : ℕ) (rhea_claws : ℕ) : 
  num_wombats = 9 → 
  num_rheas = 3 → 
  wombat_claws = 4 → 
  rhea_claws = 1 → 
  num_wombats * wombat_claws + num_rheas * rhea_claws = 39 := by
  sorry

end zoo_enclosure_claws_l3654_365460


namespace total_candies_l3654_365451

/-- The total number of candies for six people given specific relationships between their candy counts. -/
theorem total_candies (adam james rubert lisa chris emily : ℕ) : 
  adam = 6 ∧ 
  james = 3 * adam ∧ 
  rubert = 4 * james ∧ 
  lisa = 2 * rubert ∧ 
  chris = lisa + 5 ∧ 
  emily = 3 * chris - 7 → 
  adam + james + rubert + lisa + chris + emily = 829 := by
  sorry

#eval 6 + 3 * 6 + 4 * (3 * 6) + 2 * (4 * (3 * 6)) + (2 * (4 * (3 * 6)) + 5) + (3 * (2 * (4 * (3 * 6)) + 5) - 7)

end total_candies_l3654_365451


namespace normal_pumping_rate_l3654_365422

/-- Proves that given a pond with a capacity of 200 gallons, filled in 50 minutes at 2/3 of the normal pumping rate, the normal pumping rate is 6 gallons per minute. -/
theorem normal_pumping_rate (pond_capacity : ℝ) (filling_time : ℝ) (restriction_factor : ℝ) :
  pond_capacity = 200 →
  filling_time = 50 →
  restriction_factor = 2/3 →
  (restriction_factor * (pond_capacity / filling_time)) = 6 := by
  sorry

end normal_pumping_rate_l3654_365422


namespace highest_numbered_street_l3654_365459

/-- The length of Gretzky Street in meters -/
def street_length : ℕ := 5600

/-- The distance between intersecting streets in meters -/
def intersection_distance : ℕ := 350

/-- The number of non-numbered intersecting streets (Orr and Howe) -/
def non_numbered_streets : ℕ := 2

/-- Theorem stating the highest-numbered intersecting street -/
theorem highest_numbered_street :
  (street_length / intersection_distance) - non_numbered_streets = 14 := by
  sorry

end highest_numbered_street_l3654_365459


namespace repeating_six_equals_two_thirds_l3654_365489

/-- The decimal representation of a repeating decimal with a single digit. -/
def repeating_decimal (d : ℕ) : ℚ :=
  (d : ℚ) / 9

/-- Theorem stating that 0.666... (repeating) is equal to 2/3 -/
theorem repeating_six_equals_two_thirds : repeating_decimal 6 = 2 / 3 := by
  sorry

end repeating_six_equals_two_thirds_l3654_365489


namespace soap_cost_for_year_l3654_365415

/-- The cost of soap for a year given the duration and price of a single bar -/
theorem soap_cost_for_year (months_per_bar : ℕ) (price_per_bar : ℕ) : 
  months_per_bar = 2 → price_per_bar = 8 → (12 / months_per_bar) * price_per_bar = 48 := by
  sorry

#check soap_cost_for_year

end soap_cost_for_year_l3654_365415


namespace line_through_points_l3654_365493

-- Define the line
def line (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem line_through_points :
  ∀ (a b : ℝ),
  (line a b 3 = 10) →
  (line a b 7 = 22) →
  a - b = 2 := by
  sorry

end line_through_points_l3654_365493


namespace polynomial_factor_l3654_365471

theorem polynomial_factor (a : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + a*x - 5 = (x - 2) * (x + k)) → a = 1/2 := by
  sorry

end polynomial_factor_l3654_365471


namespace inequality_proof_l3654_365437

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) + Real.sqrt (c^2 - c*a + a^2) + 9 * (a*b*c)^(1/3) ≤ 4*(a + b + c) := by
  sorry

end inequality_proof_l3654_365437


namespace lake_crossing_time_difference_l3654_365466

theorem lake_crossing_time_difference 
  (lake_width : ℝ) 
  (janet_speed : ℝ) 
  (sister_speed : ℝ) 
  (h1 : lake_width = 60) 
  (h2 : janet_speed = 30) 
  (h3 : sister_speed = 12) : 
  (lake_width / sister_speed) - (lake_width / janet_speed) = 3 := by
sorry

end lake_crossing_time_difference_l3654_365466


namespace exists_same_color_rectangle_l3654_365410

/-- A color representation --/
inductive Color
| Black
| White

/-- A 3 × 7 grid where each cell is colored either black or white --/
def Grid := Fin 3 → Fin 7 → Color

/-- A rectangle in the grid, represented by its top-left and bottom-right corners --/
structure Rectangle where
  top_left : Fin 3 × Fin 7
  bottom_right : Fin 3 × Fin 7

/-- Check if a rectangle has all corners of the same color --/
def has_same_color_corners (g : Grid) (r : Rectangle) : Prop :=
  let (t, l) := r.top_left
  let (b, r) := r.bottom_right
  g t l = g t r ∧ g t l = g b l ∧ g t l = g b r

/-- Main theorem: There exists a rectangle with all corners of the same color --/
theorem exists_same_color_rectangle (g : Grid) : 
  ∃ r : Rectangle, has_same_color_corners g r := by sorry

end exists_same_color_rectangle_l3654_365410


namespace new_average_after_exclusion_l3654_365487

theorem new_average_after_exclusion (total_students : ℕ) (initial_average : ℚ) 
  (excluded_students : ℕ) (excluded_average : ℚ) (new_average : ℚ) : 
  total_students = 20 →
  initial_average = 90 →
  excluded_students = 2 →
  excluded_average = 45 →
  new_average = (total_students * initial_average - excluded_students * excluded_average) / 
    (total_students - excluded_students) →
  new_average = 95 := by
  sorry

end new_average_after_exclusion_l3654_365487


namespace union_of_A_and_B_l3654_365435

def A : Set ℤ := {-1, 1}

def B : Set ℤ := {x | |x + 1/2| < 3/2}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by sorry

end union_of_A_and_B_l3654_365435


namespace polynomial_division_degree_l3654_365494

theorem polynomial_division_degree (f d q r : Polynomial ℝ) :
  (Polynomial.degree f = 15) →
  (Polynomial.degree q = 7) →
  (r = 5 * X^2 + 3 * X - 8) →
  (f = d * q + r) →
  Polynomial.degree d = 8 := by
sorry

end polynomial_division_degree_l3654_365494


namespace max_flowers_grown_l3654_365492

theorem max_flowers_grown (total_seeds : ℕ) (seeds_per_bed : ℕ) : 
  total_seeds = 55 → seeds_per_bed = 15 → ∃ (max_flowers : ℕ), max_flowers ≤ 55 ∧ 
  ∀ (actual_flowers : ℕ), actual_flowers ≤ max_flowers := by
  sorry

end max_flowers_grown_l3654_365492


namespace consecutive_integers_square_sum_product_difference_l3654_365462

theorem consecutive_integers_square_sum_product_difference : 
  let a : ℕ := 9
  let b : ℕ := 10
  (a^2 + b^2) - (a * b) = 91 :=
by sorry

end consecutive_integers_square_sum_product_difference_l3654_365462


namespace dodecagon_interior_angles_sum_l3654_365400

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180° --/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A dodecagon is a polygon with 12 sides --/
def is_dodecagon (n : ℕ) : Prop := n = 12

theorem dodecagon_interior_angles_sum :
  ∀ n : ℕ, is_dodecagon n → sum_interior_angles n = 1800 :=
by sorry

end dodecagon_interior_angles_sum_l3654_365400


namespace probability_of_shaded_triangle_l3654_365499

/-- Given a diagram with 6 triangles, where 3 are shaded and all have equal selection probability, 
    the probability of selecting a shaded triangle is 1/2 -/
theorem probability_of_shaded_triangle (total_triangles : ℕ) (shaded_triangles : ℕ) :
  total_triangles = 6 →
  shaded_triangles = 3 →
  (shaded_triangles : ℚ) / (total_triangles : ℚ) = 1 / 2 :=
by sorry

end probability_of_shaded_triangle_l3654_365499


namespace expand_product_l3654_365456

theorem expand_product (x : ℝ) : (x + 4) * (x - 5 + 2) = x^2 + x - 12 := by
  sorry

end expand_product_l3654_365456


namespace negation_exists_not_eq_forall_eq_l3654_365401

theorem negation_exists_not_eq_forall_eq :
  (¬ ∃ x : ℝ, x^2 ≠ 1) ↔ (∀ x : ℝ, x^2 = 1) := by sorry

end negation_exists_not_eq_forall_eq_l3654_365401


namespace product_ab_equals_negative_one_l3654_365425

theorem product_ab_equals_negative_one (a b : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → 0 ≤ x^4 - x^3 + a*x + b ∧ x^4 - x^3 + a*x + b ≤ (x^2 - 1)^2) → 
  a * b = -1 := by
sorry

end product_ab_equals_negative_one_l3654_365425


namespace jessica_cut_nineteen_orchids_l3654_365455

/-- The number of orchids Jessica cut from her garden -/
def orchids_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) : ℕ :=
  final_orchids - initial_orchids

/-- Theorem stating that Jessica cut 19 orchids -/
theorem jessica_cut_nineteen_orchids :
  orchids_cut 12 2 10 21 = 19 := by
  sorry

end jessica_cut_nineteen_orchids_l3654_365455


namespace tangent_curves_n_value_l3654_365403

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 4

-- Define the hyperbola equation
def hyperbola (x y n : ℝ) : Prop := x^2 - n * (y - 1)^2 = 1

-- Define the tangency condition
def are_tangent (n : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse x y ∧ hyperbola x y n ∧
  ∀ x' y' : ℝ, ellipse x' y' ∧ hyperbola x' y' n → (x', y') = (x, y)

-- State the theorem
theorem tangent_curves_n_value :
  ∀ n : ℝ, are_tangent n → n = 3 :=
by sorry

end tangent_curves_n_value_l3654_365403


namespace min_value_expression_l3654_365409

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (3 * r) / (p + 2 * q) + (3 * p) / (2 * r + q) + (2 * q) / (p + r) ≥ 29 / 6 :=
sorry

end min_value_expression_l3654_365409


namespace paper_towel_savings_l3654_365447

/-- Calculates the percent of savings per roll when buying a package of paper towels
    compared to buying individual rolls. -/
def percent_savings (package_price : ℚ) (package_size : ℕ) (individual_price : ℚ) : ℚ :=
  let package_price_per_roll := package_price / package_size
  let savings_per_roll := individual_price - package_price_per_roll
  (savings_per_roll / individual_price) * 100

/-- Theorem stating that the percent of savings for a 12-roll package priced at $9
    compared to buying 12 rolls individually at $1 each is 25%. -/
theorem paper_towel_savings :
  percent_savings 9 12 1 = 25 := by
  sorry

end paper_towel_savings_l3654_365447

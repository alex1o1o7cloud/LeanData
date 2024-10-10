import Mathlib

namespace b_over_a_is_sqrt_2_angle_B_is_45_degrees_l3100_310070

noncomputable section

variable (A B C : ℝ)
variable (a b c : ℝ)

-- Define the triangle ABC
axiom triangle_abc : a > 0 ∧ b > 0 ∧ c > 0

-- Define the relationship between sides and angles
axiom sine_law : a / Real.sin A = b / Real.sin B

-- Given conditions
axiom condition1 : a * Real.sin A * Real.sin B + b * Real.cos A ^ 2 = Real.sqrt 2 * a
axiom condition2 : c ^ 2 = b ^ 2 + Real.sqrt 3 * a ^ 2

-- Theorems to prove
theorem b_over_a_is_sqrt_2 : b / a = Real.sqrt 2 := by sorry

theorem angle_B_is_45_degrees : B = Real.pi / 4 := by sorry

end b_over_a_is_sqrt_2_angle_B_is_45_degrees_l3100_310070


namespace cookie_selection_count_jamie_cookie_selections_l3100_310007

theorem cookie_selection_count : Nat → Nat → Nat
  | n, k => Nat.choose (n + k - 1) (k - 1)

theorem jamie_cookie_selections :
  cookie_selection_count 7 4 = 120 := by
  sorry

end cookie_selection_count_jamie_cookie_selections_l3100_310007


namespace lisa_heavier_than_sam_l3100_310042

/-- Proves that Lisa is 7.8 pounds heavier than Sam given the specified conditions -/
theorem lisa_heavier_than_sam (jack sam lisa : ℝ) 
  (total_weight : jack + sam + lisa = 210)
  (jack_weight : jack = 52)
  (sam_jack_relation : jack = sam * 0.8)
  (lisa_jack_relation : lisa = jack * 1.4) :
  lisa - sam = 7.8 := by
  sorry

end lisa_heavier_than_sam_l3100_310042


namespace abc_divisibility_problem_l3100_310052

theorem abc_divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (∃ n : ℕ, abc - 1 = n * ((a - 1) * (b - 1) * (c - 1))) →
    ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by
  sorry

#check abc_divisibility_problem

end abc_divisibility_problem_l3100_310052


namespace exam_scores_difference_l3100_310009

/-- Given five exam scores with specific properties, prove that the absolute difference between two of them is 18. -/
theorem exam_scores_difference (x y : ℝ) : 
  (x + y + 105 + 109 + 110) / 5 = 108 →
  ((x - 108)^2 + (y - 108)^2 + (105 - 108)^2 + (109 - 108)^2 + (110 - 108)^2) / 5 = 35.2 →
  |x - y| = 18 := by
sorry

end exam_scores_difference_l3100_310009


namespace colored_squares_count_l3100_310040

/-- The size of the square grid -/
def gridSize : ℕ := 101

/-- The number of L-shaped layers in the grid -/
def numLayers : ℕ := gridSize / 2

/-- The number of squares colored in the nth L-shaped layer -/
def squaresInLayer (n : ℕ) : ℕ := 8 * n

/-- The total number of colored squares in the grid -/
def totalColoredSquares : ℕ := 1 + (numLayers * (numLayers + 1) * 4)

/-- Theorem stating that the total number of colored squares is 10201 -/
theorem colored_squares_count :
  totalColoredSquares = 10201 := by sorry

end colored_squares_count_l3100_310040


namespace invalid_assignment_l3100_310028

-- Define what constitutes a valid assignment statement
def is_valid_assignment (lhs : String) (rhs : String) : Prop :=
  lhs.length = 1 ∧ lhs.all Char.isAlpha

-- Define the statement in question
def statement : String × String := ("x*y", "a")

-- Theorem to prove
theorem invalid_assignment :
  ¬(is_valid_assignment statement.1 statement.2) :=
sorry

end invalid_assignment_l3100_310028


namespace divisors_of_20_factorial_l3100_310056

theorem divisors_of_20_factorial : (Nat.divisors (Nat.factorial 20)).card = 41040 := by
  sorry

end divisors_of_20_factorial_l3100_310056


namespace complex_power_difference_l3100_310087

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^16 - (1 - i)^16 = 0 := by
  sorry

end complex_power_difference_l3100_310087


namespace unique_number_guess_l3100_310021

/-- Represents the color feedback for a digit guess -/
inductive Color
  | Green
  | Yellow
  | Gray

/-- Represents a single round of guessing -/
structure GuessRound where
  digits : Fin 5 → Nat
  colors : Fin 5 → Color

/-- The set of all possible digits (0-9) -/
def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The correct five-digit number we're trying to prove -/
def CorrectNumber : Fin 5 → Nat := ![7, 1, 2, 8, 4]

theorem unique_number_guess (round1 round2 round3 : GuessRound) : 
  (round1.digits = ![2, 6, 1, 3, 8] ∧ 
   round1.colors = ![Color.Yellow, Color.Gray, Color.Yellow, Color.Gray, Color.Yellow]) →
  (round2.digits = ![4, 1, 9, 6, 2] ∧
   round2.colors = ![Color.Yellow, Color.Green, Color.Gray, Color.Gray, Color.Yellow]) →
  (round3.digits = ![8, 1, 0, 2, 5] ∧
   round3.colors = ![Color.Yellow, Color.Green, Color.Gray, Color.Yellow, Color.Gray]) →
  (∀ n : Fin 5, CorrectNumber n ∈ Digits) →
  (∀ i j : Fin 5, i ≠ j → CorrectNumber i ≠ CorrectNumber j) →
  CorrectNumber = ![7, 1, 2, 8, 4] := by
  sorry


end unique_number_guess_l3100_310021


namespace quadratic_equation_solution_l3100_310043

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 3 ∧ x₂ = -5 ∧ 
  x₁^2 + 2*x₁ - 15 = 0 ∧ 
  x₂^2 + 2*x₂ - 15 = 0 :=
by sorry

end quadratic_equation_solution_l3100_310043


namespace min_segments_on_cube_edges_l3100_310069

/-- A cube representation -/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)

/-- A broken line on the surface of a cube -/
structure BrokenLine where
  segments : Finset (Fin 8 × Fin 8)
  num_segments : Nat
  is_closed : Bool
  vertices_on_cube : Bool

/-- Theorem statement -/
theorem min_segments_on_cube_edges (c : Cube) (bl : BrokenLine) :
  bl.num_segments = 8 ∧ bl.is_closed ∧ bl.vertices_on_cube →
  ∃ (coinciding_segments : Finset (Fin 8 × Fin 8)),
    coinciding_segments ⊆ c.edges ∧
    coinciding_segments ⊆ bl.segments ∧
    coinciding_segments.card = 2 ∧
    ∀ (cs : Finset (Fin 8 × Fin 8)),
      cs ⊆ c.edges ∧ cs ⊆ bl.segments →
      cs.card ≥ 2 := by
  sorry

end min_segments_on_cube_edges_l3100_310069


namespace a_power_sum_l3100_310010

theorem a_power_sum (a : ℂ) (h : a^2 - a + 1 = 0) : a^10 + a^20 + a^30 = 3 := by
  sorry

end a_power_sum_l3100_310010


namespace clothing_store_problem_l3100_310035

theorem clothing_store_problem (cost_A B : ℕ) (profit_A B : ℕ) :
  3 * cost_A + 2 * cost_B = 450 →
  cost_A + cost_B = 175 →
  profit_A = 30 →
  profit_B = 20 →
  (∀ m : ℕ, m ≤ 100 → profit_A * m + profit_B * (100 - m) ≥ 2400 →
    ∃ n : ℕ, n ≥ m ∧ n ≥ 40) →
  ∃ m : ℕ, m ≥ 40 ∧ ∀ n : ℕ, n < m → profit_A * n + profit_B * (100 - n) < 2400 :=
by
  sorry

end clothing_store_problem_l3100_310035


namespace cross_to_square_l3100_310032

/-- Represents a cross made of unit squares -/
structure Cross :=
  (num_squares : ℕ)
  (side_length : ℝ)

/-- Represents a square -/
structure Square :=
  (side_length : ℝ)

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.side_length ^ 2

/-- The cross in the problem -/
def problem_cross : Cross := { num_squares := 5, side_length := 1 }

/-- The theorem to be proved -/
theorem cross_to_square (c : Cross) (s : Square) 
  (h1 : c = problem_cross) 
  (h2 : s.side_length = Real.sqrt 5) : 
  s.area = c.num_squares * c.side_length ^ 2 := by
  sorry

end cross_to_square_l3100_310032


namespace negative_of_negative_six_equals_six_l3100_310080

theorem negative_of_negative_six_equals_six : -(-6) = 6 := by
  sorry

end negative_of_negative_six_equals_six_l3100_310080


namespace qingming_rain_is_random_l3100_310072

/-- An event that occurs during a specific season --/
structure SeasonalEvent where
  season : String
  description : String

/-- A property indicating whether an event can be predicted with certainty --/
def isPredictable (e : SeasonalEvent) : Prop := sorry

/-- A property indicating whether an event's occurrence varies from year to year --/
def hasVariableOccurrence (e : SeasonalEvent) : Prop := sorry

/-- Definition of a random event --/
def isRandomEvent (e : SeasonalEvent) : Prop := 
  ¬(isPredictable e) ∧ hasVariableOccurrence e

/-- The main theorem --/
theorem qingming_rain_is_random (e : SeasonalEvent) 
  (h1 : e.season = "Qingming")
  (h2 : e.description = "drizzling rain")
  (h3 : ¬(isPredictable e))
  (h4 : hasVariableOccurrence e) : 
  isRandomEvent e := by
  sorry

end qingming_rain_is_random_l3100_310072


namespace quadratic_and_trig_problem_l3100_310001

theorem quadratic_and_trig_problem :
  -- Part 1: Quadratic equation
  (∃ x1 x2 : ℝ, x1 = 1 + Real.sqrt 2 ∧ x2 = 1 - Real.sqrt 2 ∧
    x1^2 - 2*x1 - 1 = 0 ∧ x2^2 - 2*x2 - 1 = 0) ∧
  -- Part 2: Trigonometric expression
  (4 * (Real.sin (60 * π / 180))^2 - Real.tan (45 * π / 180) +
   Real.sqrt 2 * Real.cos (45 * π / 180) - Real.sin (30 * π / 180) = 5/2) :=
by sorry

end quadratic_and_trig_problem_l3100_310001


namespace lcm_problem_l3100_310019

theorem lcm_problem (a b c : ℕ+) (h1 : b = 30) (h2 : c = 40) (h3 : Nat.lcm (Nat.lcm a.val b.val) c.val = 120) : a = 60 := by
  sorry

end lcm_problem_l3100_310019


namespace quadratic_roots_equal_integral_l3100_310086

/-- The roots of the quadratic equation 3x^2 - 6x + c = 0 are equal and integral when the discriminant is zero -/
theorem quadratic_roots_equal_integral (c : ℝ) :
  (∀ x : ℝ, 3 * x^2 - 6 * x + c = 0 ↔ x = 1) ∧ ((-6)^2 - 4 * 3 * c = 0) := by
  sorry


end quadratic_roots_equal_integral_l3100_310086


namespace three_person_subcommittee_from_eight_l3100_310026

theorem three_person_subcommittee_from_eight (n : ℕ) (k : ℕ) : n = 8 ∧ k = 3 → Nat.choose n k = 56 := by
  sorry

end three_person_subcommittee_from_eight_l3100_310026


namespace complex_magnitude_sum_l3100_310081

theorem complex_magnitude_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) = 2 * Real.sqrt 34 := by
  sorry

end complex_magnitude_sum_l3100_310081


namespace chair_count_sequence_l3100_310066

theorem chair_count_sequence (a : ℕ → ℕ) 
  (h1 : a 1 = 14)
  (h2 : a 2 = 23)
  (h3 : a 3 = 32)
  (h5 : a 5 = 50)
  (h6 : a 6 = 59)
  (h_arithmetic : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = a 2 - a 1) :
  a 4 = 41 := by
  sorry

end chair_count_sequence_l3100_310066


namespace nine_digit_divisibility_l3100_310083

theorem nine_digit_divisibility (a b c : Nat) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9) (h4 : a ≠ 0) :
  ∃ k : Nat, (100 * a + 10 * b + c) * 1001001 = k * (100000000 * a + 10000000 * b + 1000000 * c +
                                                     100000 * a + 10000 * b + 1000 * c +
                                                     100 * a + 10 * b + c) :=
sorry

end nine_digit_divisibility_l3100_310083


namespace necessary_not_sufficient_condition_a_plus_2b_necessary_not_sufficient_l3100_310015

/-- For all x in [0, 1], a+2b>0 is a necessary but not sufficient condition for ax+b>0 to always hold true -/
theorem necessary_not_sufficient_condition (a b : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → (a * x + b > 0)) ↔ (b > 0 ∧ a + b > 0) :=
by sorry

/-- a+2b>0 is necessary but not sufficient for the above condition -/
theorem a_plus_2b_necessary_not_sufficient (a b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → (a * x + b > 0)) → (a + 2*b > 0) ∧
  ¬(∀ a b : ℝ, (a + 2*b > 0) → (∀ x : ℝ, x ∈ Set.Icc 0 1 → (a * x + b > 0))) :=
by sorry

end necessary_not_sufficient_condition_a_plus_2b_necessary_not_sufficient_l3100_310015


namespace shorter_segment_length_l3100_310029

-- Define the triangle
def triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the altitude and segments
def altitude_segment (a b c x h : ℝ) : Prop :=
  triangle a b c ∧
  x > 0 ∧ h > 0 ∧
  x + (c - x) = c ∧
  a^2 = x^2 + h^2 ∧
  b^2 = (c - x)^2 + h^2

-- Theorem statement
theorem shorter_segment_length :
  ∀ (x h : ℝ),
  altitude_segment 40 50 90 x h →
  x = 40 :=
sorry

end shorter_segment_length_l3100_310029


namespace geometric_sequence_formula_l3100_310005

/-- A geometric sequence with common ratio 4 and sum of first three terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 4 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The general term formula for the geometric sequence -/
theorem geometric_sequence_formula (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∀ n : ℕ, a n = 4^(n - 1) := by
  sorry

end geometric_sequence_formula_l3100_310005


namespace semicircle_area_theorem_l3100_310017

theorem semicircle_area_theorem (x y z : ℝ) : 
  x^2 + y^2 = z^2 →
  (1/8) * π * x^2 = 50 * π →
  (1/8) * π * y^2 = 288 * π →
  (1/8) * π * z^2 = 338 * π :=
sorry

end semicircle_area_theorem_l3100_310017


namespace additional_cars_needed_l3100_310063

def current_cars : ℕ := 23
def cars_per_row : ℕ := 6

theorem additional_cars_needed :
  let next_multiple := (current_cars + cars_per_row - 1) / cars_per_row * cars_per_row
  next_multiple - current_cars = 1 := by sorry

end additional_cars_needed_l3100_310063


namespace inequality_proofs_l3100_310059

theorem inequality_proofs :
  (∀ x : ℝ, x * (1 - x) ≤ 1 / 4) ∧
  (∀ x a : ℝ, x * (a - x) ≤ a^2 / 4) := by
  sorry

end inequality_proofs_l3100_310059


namespace quadratic_inequality_range_l3100_310008

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x + 1 ≥ 0) ↔ m ∈ Set.Icc (-2 : ℝ) 2 := by
  sorry

end quadratic_inequality_range_l3100_310008


namespace domain_of_g_is_closed_unit_interval_l3100_310099

-- Define the function f with domain [0,1]
def f : Set ℝ := Set.Icc 0 1

-- Define the function g(x) = f(x^2)
def g (x : ℝ) : Prop := x^2 ∈ f

-- Theorem statement
theorem domain_of_g_is_closed_unit_interval :
  {x : ℝ | g x} = Set.Icc (-1) 1 := by sorry

end domain_of_g_is_closed_unit_interval_l3100_310099


namespace probability_of_prime_ball_l3100_310024

def ball_numbers : List Nat := [3, 4, 5, 6, 7, 8, 11, 13]

def is_prime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d ≤ 1 || n % (d + 2) ≠ 0)

def count_primes (numbers : List Nat) : Nat :=
  (numbers.filter is_prime).length

theorem probability_of_prime_ball :
  (count_primes ball_numbers : Rat) / (ball_numbers.length : Rat) = 5 / 8 := by
  sorry

end probability_of_prime_ball_l3100_310024


namespace michael_twice_jacob_age_l3100_310036

/-- Given that:
    - Jacob is 12 years younger than Michael
    - Jacob will be 13 years old in 4 years
    - At some point in the future, Michael will be twice as old as Jacob
    This theorem proves that Michael will be twice as old as Jacob in 3 years. -/
theorem michael_twice_jacob_age (jacob_age : ℕ) (michael_age : ℕ) (years_until_twice : ℕ) :
  michael_age = jacob_age + 12 →
  jacob_age + 4 = 13 →
  michael_age + years_until_twice = 2 * (jacob_age + years_until_twice) →
  years_until_twice = 3 := by
  sorry

end michael_twice_jacob_age_l3100_310036


namespace average_weight_increase_l3100_310093

/-- Proves that replacing a sailor weighing 56 kg with a sailor weighing 64 kg
    in a group of 8 sailors increases the average weight by 1 kg. -/
theorem average_weight_increase (initial_average : ℝ) : 
  let total_weight := 8 * initial_average
  let new_total_weight := total_weight - 56 + 64
  let new_average := new_total_weight / 8
  new_average - initial_average = 1 := by
  sorry

end average_weight_increase_l3100_310093


namespace division_remainder_problem_l3100_310057

theorem division_remainder_problem (L S : ℕ) : 
  L - S = 1345 → 
  L = 1596 → 
  L / S = 6 → 
  L % S = 90 := by
sorry

end division_remainder_problem_l3100_310057


namespace fiona_cleaning_time_proof_l3100_310046

/-- Calculates Fiona's cleaning time in minutes given the total cleaning time and Lilly's fraction of work -/
def fiona_cleaning_time (total_time : ℝ) (lilly_fraction : ℝ) : ℝ :=
  (total_time - lilly_fraction * total_time) * 60

/-- Theorem: Given a total cleaning time of 8 hours and Lilly spending 1/4 of the total time, 
    Fiona's cleaning time in minutes is equal to 360. -/
theorem fiona_cleaning_time_proof :
  fiona_cleaning_time 8 (1/4) = 360 := by
  sorry

end fiona_cleaning_time_proof_l3100_310046


namespace pizza_distribution_l3100_310030

theorem pizza_distribution (treShawn Michael LaMar : ℚ) : 
  treShawn = 1/2 →
  Michael = 1/3 →
  treShawn + Michael + LaMar = 1 →
  LaMar = 1/6 := by
sorry

end pizza_distribution_l3100_310030


namespace perpendicular_points_coplanar_l3100_310088

-- Define the types for points and spheres
variable (Point Sphere : Type)

-- Define the property of a point being on a sphere
variable (onSphere : Point → Sphere → Prop)

-- Define the property of points being distinct
variable (distinct : List Point → Prop)

-- Define the property of points being coplanar
variable (coplanar : List Point → Prop)

-- Define the property of a point being on a line
variable (onLine : Point → Point → Point → Prop)

-- Define the property of a line being perpendicular to another line
variable (perpendicular : Point → Point → Point → Point → Prop)

-- Define the quadrilateral pyramid inscribed in a sphere
variable (S A B C D : Point) (sphere1 : Sphere)
variable (inscribed : onSphere S sphere1 ∧ onSphere A sphere1 ∧ onSphere B sphere1 ∧ onSphere C sphere1 ∧ onSphere D sphere1)

-- Define the perpendicular points
variable (A1 B1 C1 D1 : Point)
variable (perp : perpendicular A A1 S C ∧ perpendicular B B1 S D ∧ perpendicular C C1 S A ∧ perpendicular D D1 S B)

-- Define the property of perpendicular points being on the respective lines
variable (onLines : onLine A1 S C ∧ onLine B1 S D ∧ onLine C1 S A ∧ onLine D1 S B)

-- Define the property of S, A1, B1, C1, D1 being distinct and on another sphere
variable (sphere2 : Sphere)
variable (distinctOnSphere : distinct [S, A1, B1, C1, D1] ∧ 
                             onSphere S sphere2 ∧ onSphere A1 sphere2 ∧ onSphere B1 sphere2 ∧ onSphere C1 sphere2 ∧ onSphere D1 sphere2)

-- Theorem statement
theorem perpendicular_points_coplanar : 
  coplanar [A1, B1, C1, D1] :=
sorry

end perpendicular_points_coplanar_l3100_310088


namespace function_not_linear_plus_integer_l3100_310020

theorem function_not_linear_plus_integer : 
  ∃ (f : ℚ → ℚ), 
    (∀ x y : ℚ, ∃ z : ℤ, f (x + y) - f x - f y = ↑z) ∧ 
    (¬ ∃ c : ℚ, ∀ x : ℚ, ∃ z : ℤ, f x - c * x = ↑z) := by
  sorry

end function_not_linear_plus_integer_l3100_310020


namespace car_distance_traveled_l3100_310054

theorem car_distance_traveled (time : ℝ) (speed : ℝ) (distance : ℝ) : 
  time = 11 → speed = 65 → distance = time * speed → distance = 715 := by
  sorry

end car_distance_traveled_l3100_310054


namespace expression_value_l3100_310078

theorem expression_value (a b : ℤ) (h : a = b + 1) : 3 + 2*a - 2*b = 5 := by
  sorry

end expression_value_l3100_310078


namespace coin_radius_l3100_310037

/-- Given a coin with diameter 14 millimeters, its radius is 7 millimeters. -/
theorem coin_radius (d : ℝ) (h : d = 14) : d / 2 = 7 := by
  sorry

end coin_radius_l3100_310037


namespace condition_necessary_not_sufficient_l3100_310049

-- Define a sequence type
def Sequence := ℕ → ℝ

-- Define the property of a sequence satisfying a_n = 2a_{n-1} for n ≥ 2
def SatisfiesCondition (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)

-- Define a geometric sequence with common ratio 2
def IsGeometricSequenceWithRatio2 (a : Sequence) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n

-- Theorem statement
theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometricSequenceWithRatio2 a → SatisfiesCondition a) ∧
  (∃ a : Sequence, SatisfiesCondition a ∧ ¬IsGeometricSequenceWithRatio2 a) := by
  sorry

end condition_necessary_not_sufficient_l3100_310049


namespace stating_isosceles_triangle_base_height_l3100_310062

/-- Represents an isosceles triangle with leg length a -/
structure IsoscelesTriangle (a : ℝ) where
  (a_pos : a > 0)

/-- The height from one leg to the other leg forms a 30° angle -/
def height_angle (t : IsoscelesTriangle a) : ℝ := 30

/-- The height from the base of the isosceles triangle -/
def base_height (t : IsoscelesTriangle a) : Set ℝ :=
  {h | h = (Real.sqrt 3 / 2) * a ∨ h = (1 / 2) * a}

/-- 
  Theorem stating that for an isosceles triangle with leg length a, 
  where the height from one leg to the other leg forms a 30° angle, 
  the height from the base is either (√3/2)a or (1/2)a.
-/
theorem isosceles_triangle_base_height (a : ℝ) (t : IsoscelesTriangle a) :
  ∀ h, h ∈ base_height t ↔ 
    (h = (Real.sqrt 3 / 2) * a ∨ h = (1 / 2) * a) ∧ 
    height_angle t = 30 := by
  sorry

end stating_isosceles_triangle_base_height_l3100_310062


namespace betty_garden_ratio_l3100_310092

/-- Represents a herb garden with oregano and basil plants -/
structure HerbGarden where
  total_plants : ℕ
  basil_plants : ℕ
  oregano_plants : ℕ
  total_eq : total_plants = oregano_plants + basil_plants

/-- The ratio of oregano to basil plants in Betty's garden is 12:5 -/
theorem betty_garden_ratio (garden : HerbGarden) 
    (h1 : garden.total_plants = 17)
    (h2 : garden.basil_plants = 5) :
    garden.oregano_plants / garden.basil_plants = 12 / 5 := by
  sorry

#check betty_garden_ratio

end betty_garden_ratio_l3100_310092


namespace farm_sheep_count_l3100_310034

/-- Given a farm with sheep and horses, prove that the number of sheep is 16 -/
theorem farm_sheep_count (sheep horses : ℕ) (horse_food_per_day total_horse_food : ℕ) : 
  (sheep : ℚ) / horses = 2 / 7 →
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  horses * horse_food_per_day = total_horse_food →
  sheep = 16 := by
sorry

end farm_sheep_count_l3100_310034


namespace highway_work_completion_fraction_l3100_310097

theorem highway_work_completion_fraction :
  let total_length : ℝ := 2 -- km
  let initial_workers : ℕ := 100
  let initial_duration : ℕ := 50 -- days
  let initial_daily_hours : ℕ := 8
  let actual_work_days : ℕ := 25
  let additional_workers : ℕ := 60
  let new_daily_hours : ℕ := 10

  let total_man_hours : ℝ := initial_workers * initial_duration * initial_daily_hours
  let remaining_man_hours : ℝ := (initial_workers + additional_workers) * (initial_duration - actual_work_days) * new_daily_hours

  total_man_hours = remaining_man_hours →
  (total_man_hours - remaining_man_hours) / total_man_hours = 1 / 2 :=
by
  sorry

end highway_work_completion_fraction_l3100_310097


namespace table_tennis_tournament_impossibility_l3100_310013

theorem table_tennis_tournament_impossibility (k : ℕ) (h : k > 0) :
  let participants := 2 * k
  let total_matches := k * (2 * k - 1)
  let total_judgements := 2 * total_matches
  ¬ ∃ (judgements_per_participant : ℕ),
    (judgements_per_participant * participants = total_judgements ∧
     judgements_per_participant * 2 = 2 * k - 1) :=
by sorry

end table_tennis_tournament_impossibility_l3100_310013


namespace total_distance_rowed_l3100_310014

/-- Calculates the total distance rowed by a man given specific conditions -/
theorem total_distance_rowed (still_water_speed wind_speed river_speed : ℝ)
  (total_time : ℝ) (h1 : still_water_speed = 8)
  (h2 : wind_speed = 1.5) (h3 : river_speed = 3.5) (h4 : total_time = 2) :
  let speed_to := still_water_speed - river_speed - wind_speed
  let speed_from := still_water_speed + river_speed + wind_speed
  let distance := (speed_to * speed_from * total_time) / (speed_to + speed_from)
  2 * distance = 9.75 :=
by sorry

end total_distance_rowed_l3100_310014


namespace factors_of_210_l3100_310089

theorem factors_of_210 : Nat.card (Nat.divisors 210) = 16 := by
  sorry

end factors_of_210_l3100_310089


namespace a_range_l3100_310084

/-- Proposition p: A real number x satisfies 2 < x < 3 -/
def p (x : ℝ) : Prop := 2 < x ∧ x < 3

/-- Proposition q: A real number x satisfies 2x^2 - 9x + a < 0 -/
def q (x a : ℝ) : Prop := 2 * x^2 - 9 * x + a < 0

/-- p is a sufficient condition for q -/
def p_implies_q (a : ℝ) : Prop := ∀ x, p x → q x a

theorem a_range (a : ℝ) : p_implies_q a ↔ 7 ≤ a ∧ a ≤ 8 := by sorry

end a_range_l3100_310084


namespace triangle_abc_isosceles_l3100_310082

/-- Given points A(3,5), B(-6,-2), and C(0,-6), prove that AB = AC -/
theorem triangle_abc_isosceles (A B C : ℝ × ℝ) : 
  A = (3, 5) → B = (-6, -2) → C = (0, -6) → 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 := by
  sorry

#check triangle_abc_isosceles

end triangle_abc_isosceles_l3100_310082


namespace power_difference_equals_one_ninth_l3100_310090

theorem power_difference_equals_one_ninth (x y : ℕ) : 
  (2^x : ℕ) ∣ 360 ∧ 
  ∀ k > x, ¬((2^k : ℕ) ∣ 360) ∧ 
  (5^y : ℕ) ∣ 360 ∧ 
  ∀ m > y, ¬((5^m : ℕ) ∣ 360) → 
  (1/3 : ℚ)^(x - y) = 1/9 := by
sorry

end power_difference_equals_one_ninth_l3100_310090


namespace price_per_cup_l3100_310016

/-- Represents the number of trees each sister has -/
def trees : ℕ := 110

/-- Represents the number of oranges Gabriela's trees produce per tree -/
def gabriela_oranges_per_tree : ℕ := 600

/-- Represents the number of oranges Alba's trees produce per tree -/
def alba_oranges_per_tree : ℕ := 400

/-- Represents the number of oranges Maricela's trees produce per tree -/
def maricela_oranges_per_tree : ℕ := 500

/-- Represents the number of oranges needed to make one cup of juice -/
def oranges_per_cup : ℕ := 3

/-- Represents the total earnings from selling the juice -/
def total_earnings : ℕ := 220000

/-- Calculates the total number of oranges harvested by all sisters -/
def total_oranges : ℕ := 
  trees * gabriela_oranges_per_tree + 
  trees * alba_oranges_per_tree + 
  trees * maricela_oranges_per_tree

/-- Calculates the total number of cups of juice that can be made -/
def total_cups : ℕ := total_oranges / oranges_per_cup

/-- Theorem stating that the price per cup of juice is $4 -/
theorem price_per_cup : total_earnings / total_cups = 4 := by
  sorry

end price_per_cup_l3100_310016


namespace pet_shop_dogs_l3100_310000

theorem pet_shop_dogs (ratio_dogs : ℕ) (ratio_cats : ℕ) (ratio_bunnies : ℕ) 
  (total_dogs_bunnies : ℕ) : 
  ratio_dogs = 3 → 
  ratio_cats = 5 → 
  ratio_bunnies = 9 → 
  total_dogs_bunnies = 204 → 
  ∃ (x : ℕ), x * (ratio_dogs + ratio_bunnies) = total_dogs_bunnies ∧ 
             x * ratio_dogs = 51 := by
  sorry

end pet_shop_dogs_l3100_310000


namespace baking_powder_difference_l3100_310095

/-- The amount of baking powder Kelly had yesterday, in boxes -/
def yesterday_supply : ℚ := 0.4

/-- The amount of baking powder Kelly has today, in boxes -/
def today_supply : ℚ := 0.3

/-- The difference in baking powder supply between yesterday and today -/
def supply_difference : ℚ := yesterday_supply - today_supply

theorem baking_powder_difference :
  supply_difference = 0.1 := by sorry

end baking_powder_difference_l3100_310095


namespace abs_over_a_plus_one_l3100_310077

theorem abs_over_a_plus_one (a : ℝ) (h : a ≠ 0) :
  (|a| / a + 1 = 0) ∨ (|a| / a + 1 = 2) := by
  sorry

end abs_over_a_plus_one_l3100_310077


namespace complex_cube_root_magnitude_l3100_310011

theorem complex_cube_root_magnitude (w : ℂ) (h : w^3 = 64 - 48*I) : 
  Complex.abs w = 2 * Real.rpow 10 (1/3) := by
sorry

end complex_cube_root_magnitude_l3100_310011


namespace distance_to_first_sign_l3100_310094

/-- Given a bike ride with two stop signs, calculate the distance to the first stop sign -/
theorem distance_to_first_sign
  (total_distance : ℕ)
  (distance_after_second_sign : ℕ)
  (distance_between_signs : ℕ)
  (h1 : total_distance = 1000)
  (h2 : distance_after_second_sign = 275)
  (h3 : distance_between_signs = 375) :
  total_distance - distance_after_second_sign - distance_between_signs = 350 :=
by sorry

end distance_to_first_sign_l3100_310094


namespace line_through_points_l3100_310051

theorem line_through_points (a b : ℚ) : 
  (7 : ℚ) = a * 3 + b ∧ (19 : ℚ) = a * 10 + b → a - b = -1/7 := by
  sorry

end line_through_points_l3100_310051


namespace EF_length_is_19_2_l3100_310061

/-- Two similar right triangles ABC and DEF with given side lengths -/
structure SimilarRightTriangles where
  -- Triangle ABC
  AB : ℝ
  BC : ℝ
  -- Triangle DEF
  DE : ℝ
  -- Similarity ratio
  k : ℝ
  -- Conditions
  AB_positive : AB > 0
  BC_positive : BC > 0
  DE_positive : DE > 0
  k_positive : k > 0
  similarity : k = DE / AB
  AB_value : AB = 10
  BC_value : BC = 8
  DE_value : DE = 24

/-- The length of EF in the similar right triangles -/
def EF_length (t : SimilarRightTriangles) : ℝ :=
  t.k * t.BC

/-- Theorem: The length of EF is 19.2 -/
theorem EF_length_is_19_2 (t : SimilarRightTriangles) : EF_length t = 19.2 := by
  sorry

#check EF_length_is_19_2

end EF_length_is_19_2_l3100_310061


namespace science_team_selection_ways_l3100_310073

theorem science_team_selection_ways (total_boys : ℕ) (total_girls : ℕ) 
  (team_size : ℕ) (required_boys : ℕ) (required_girls : ℕ) : 
  total_boys = 7 → total_girls = 10 → team_size = 8 → 
  required_boys = 4 → required_girls = 4 →
  (Nat.choose total_boys required_boys) * (Nat.choose total_girls required_girls) = 7350 :=
by sorry

end science_team_selection_ways_l3100_310073


namespace final_count_A_l3100_310096

/-- Represents a switch with its ID and position -/
structure Switch where
  id : Nat
  position : Fin 3

/-- Represents the state of all switches -/
def SwitchState := Fin 1000 → Switch

/-- Checks if one number divides another -/
def divides (a b : Nat) : Prop := ∃ k, b = a * k

/-- Represents a single step in the process -/
def step (s : SwitchState) (i : Fin 1000) : SwitchState := sorry

/-- Represents the entire process of 1000 steps -/
def process (initial : SwitchState) : SwitchState := sorry

/-- Counts the number of switches in position A -/
def countA (s : SwitchState) : Nat := sorry

/-- The main theorem to prove -/
theorem final_count_A (initial : SwitchState) : 
  (∀ i, (initial i).position = 0) →
  (∀ i, ∃ x y z, x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ (initial i).id = 2^x * 3^y * 7^z) →
  countA (process initial) = 660 := sorry

end final_count_A_l3100_310096


namespace smallest_four_digit_sum_20_l3100_310076

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is four digits -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- Theorem: 1999 is the smallest four-digit number whose digits sum to 20 -/
theorem smallest_four_digit_sum_20 : 
  (∀ n : ℕ, is_four_digit n → sum_of_digits n = 20 → 1999 ≤ n) ∧ 
  (is_four_digit 1999 ∧ sum_of_digits 1999 = 20) := by sorry

end smallest_four_digit_sum_20_l3100_310076


namespace zeta_sum_seventh_power_l3100_310012

theorem zeta_sum_seventh_power (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 1)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 3)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 7) :
  ζ₁^7 + ζ₂^7 + ζ₃^7 = 71 := by
  sorry

end zeta_sum_seventh_power_l3100_310012


namespace square_root_sum_equals_absolute_value_sum_l3100_310033

theorem square_root_sum_equals_absolute_value_sum (x : ℝ) :
  Real.sqrt (x^2 + 4*x + 4) + Real.sqrt (x^2 - 6*x + 9) = |x + 2| + |x - 3| := by
  sorry

end square_root_sum_equals_absolute_value_sum_l3100_310033


namespace factor_x6_minus_81_l3100_310053

theorem factor_x6_minus_81 (x : ℝ) : x^6 - 81 = (x^3 + 9) * (x - 3) * (x^2 + 3*x + 9) := by
  sorry

end factor_x6_minus_81_l3100_310053


namespace inequality_solution_sets_l3100_310065

theorem inequality_solution_sets (a b x : ℝ) :
  let f := fun x => b * x^2 - (3 * a * b - b) * x + 2 * a^2 * b - a * b
  (∀ x, b = 1 ∧ a > 1 → (f x < 0 ↔ a < x ∧ x < 2 * a - 1)) ∧
  (∀ x, b = a ∧ a ≤ 1 → 
    ((a = 0 ∨ a = 1) → ¬∃ x, f x < 0) ∧
    (0 < a ∧ a < 1 → (f x < 0 ↔ 2 * a - 1 < x ∧ x < a)) ∧
    (a < 0 → (f x < 0 ↔ x < 2 * a - 1 ∨ x > a))) := by
  sorry

end inequality_solution_sets_l3100_310065


namespace no_third_quadrant_implies_m_leq_1_l3100_310041

def linear_function (x m : ℝ) : ℝ := -2 * x + 1 - m

theorem no_third_quadrant_implies_m_leq_1 :
  ∀ m : ℝ, (∀ x y : ℝ, y = linear_function x m → (x < 0 → y ≥ 0)) → m ≤ 1 := by
  sorry

end no_third_quadrant_implies_m_leq_1_l3100_310041


namespace richmond_tigers_ticket_sales_l3100_310039

theorem richmond_tigers_ticket_sales (total_tickets : ℕ) (first_half_tickets : ℕ) (second_half_tickets : ℕ) :
  total_tickets = 9570 →
  first_half_tickets = 3867 →
  second_half_tickets = total_tickets - first_half_tickets →
  second_half_tickets = 5703 :=
by
  sorry

end richmond_tigers_ticket_sales_l3100_310039


namespace ripe_apples_weight_l3100_310075

/-- Given the total number of apples, the number of unripe apples, and the weight of each ripe apple,
    prove that the total weight of ripe apples is equal to the product of the number of ripe apples
    and the weight of each ripe apple. -/
theorem ripe_apples_weight
  (total_apples : ℕ)
  (unripe_apples : ℕ)
  (ripe_apple_weight : ℕ)
  (h1 : unripe_apples ≤ total_apples) :
  (total_apples - unripe_apples) * ripe_apple_weight =
    (total_apples - unripe_apples) * ripe_apple_weight :=
by sorry

end ripe_apples_weight_l3100_310075


namespace chromium_percentage_in_first_alloy_l3100_310055

/-- Given two alloys, proves that the percentage of chromium in the first alloy is 12% -/
theorem chromium_percentage_in_first_alloy :
  let weight_first_alloy : ℝ := 15
  let weight_second_alloy : ℝ := 35
  let chromium_percentage_second_alloy : ℝ := 10
  let chromium_percentage_new_alloy : ℝ := 10.6
  let total_weight : ℝ := weight_first_alloy + weight_second_alloy
  ∃ (chromium_percentage_first_alloy : ℝ),
    chromium_percentage_first_alloy * weight_first_alloy / 100 +
    chromium_percentage_second_alloy * weight_second_alloy / 100 =
    chromium_percentage_new_alloy * total_weight / 100 ∧
    chromium_percentage_first_alloy = 12 :=
by
  sorry


end chromium_percentage_in_first_alloy_l3100_310055


namespace sum_of_powers_l3100_310018

theorem sum_of_powers (x : ℝ) (h1 : x^2023 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^2022 + x^2021 + x^2020 + x^2019 + x^2018 + x^2017 + x^2016 + x^2015 + x^2014 + x^2013 +
  x^2012 + x^2011 + x^2010 + x^2009 + x^2008 + x^2007 + x^2006 + x^2005 + x^2004 + x^2003 +
  x^2002 + x^2001 + x^2000 + x^1999 + x^1998 + x^1997 + x^1996 + x^1995 + x^1994 + x^1993 +
  x^1992 + x^1991 + x^1990 + x^1989 + x^1988 + x^1987 + x^1986 + x^1985 + x^1984 + x^1983 +
  x^1982 + x^1981 + x^1980 + x^1979 + x^1978 + x^1977 + x^1976 + x^1975 + x^1974 + x^1973 +
  -- ... (omitting middle terms for brevity)
  x^50 + x^49 + x^48 + x^47 + x^46 + x^45 + x^44 + x^43 + x^42 + x^41 +
  x^40 + x^39 + x^38 + x^37 + x^36 + x^35 + x^34 + x^33 + x^32 + x^31 +
  x^30 + x^29 + x^28 + x^27 + x^26 + x^25 + x^24 + x^23 + x^22 + x^21 +
  x^20 + x^19 + x^18 + x^17 + x^16 + x^15 + x^14 + x^13 + x^12 + x^11 +
  x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := by
  sorry

end sum_of_powers_l3100_310018


namespace first_group_count_l3100_310068

theorem first_group_count (total_count : Nat) (total_avg : ℝ) (first_group_avg : ℝ) 
  (last_group_count : Nat) (last_group_avg : ℝ) (sixth_number : ℝ) : 
  total_count = 11 →
  total_avg = 10.7 →
  first_group_avg = 10.5 →
  last_group_count = 6 →
  last_group_avg = 11.4 →
  sixth_number = 13.700000000000017 →
  (total_count - last_group_count : ℝ) = 4 := by
sorry

end first_group_count_l3100_310068


namespace max_plus_min_of_f_l3100_310022

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + 1)^2 / (Real.sin x^2 + 1)

theorem max_plus_min_of_f : 
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
                (∀ x, m ≤ f x) ∧ (∃ x, f x = m) ∧ 
                (M + m = 2) :=
sorry

end max_plus_min_of_f_l3100_310022


namespace dividend_calculation_l3100_310050

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 20)
  (h2 : quotient = 8)
  (h3 : remainder = 6) :
  divisor * quotient + remainder = 166 := by
  sorry

end dividend_calculation_l3100_310050


namespace dougs_age_l3100_310047

/-- Given the ages of Qaddama, Jack, and Doug, prove Doug's age --/
theorem dougs_age (qaddama jack doug : ℕ) 
  (h1 : qaddama = jack + 6)
  (h2 : doug = jack + 3)
  (h3 : qaddama = 19) : 
  doug = 16 := by
  sorry

end dougs_age_l3100_310047


namespace expression_domain_l3100_310027

def expression_defined (x : ℝ) : Prop :=
  x + 2 > 0 ∧ 5 - x > 0

theorem expression_domain : ∀ x : ℝ, expression_defined x ↔ -2 < x ∧ x < 5 := by
  sorry

end expression_domain_l3100_310027


namespace min_value_inequality_l3100_310004

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2*x + y = 2) :
  1/x^2 + 4/y^2 ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + y₀ = 2 ∧ 1/x₀^2 + 4/y₀^2 = 8 :=
by sorry

end min_value_inequality_l3100_310004


namespace max_value_of_fraction_l3100_310085

theorem max_value_of_fraction (x y z u v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0) (hv : v > 0) :
  (x*y + y*z + z*u + u*v) / (2*x^2 + y^2 + 2*z^2 + u^2 + 2*v^2) ≤ 1/2 := by
  sorry

end max_value_of_fraction_l3100_310085


namespace partial_square_division_l3100_310048

/-- Represents a square with a side length and a removed portion. -/
structure PartialSquare where
  side_length : ℝ
  removed_fraction : ℝ

/-- Represents a division of the remaining area into parts. -/
structure AreaDivision where
  num_parts : ℕ
  area_per_part : ℝ

/-- Theorem stating that a square with side length 4 and one fourth removed
    can be divided into four equal parts with area 3 each. -/
theorem partial_square_division (s : PartialSquare)
  (h1 : s.side_length = 4)
  (h2 : s.removed_fraction = 1/4) :
  ∃ (d : AreaDivision), 
    d.num_parts = 4 ∧ 
    d.area_per_part = 3 ∧
    d.num_parts * d.area_per_part = s.side_length^2 - s.side_length^2 * s.removed_fraction :=
by sorry

end partial_square_division_l3100_310048


namespace five_cubic_yards_to_cubic_inches_l3100_310060

/-- Converts cubic yards to cubic inches -/
def cubic_yards_to_cubic_inches (yards : ℕ) : ℕ :=
  let feet_per_yard : ℕ := 3
  let inches_per_foot : ℕ := 12
  yards * (feet_per_yard ^ 3) * (inches_per_foot ^ 3)

/-- Theorem stating that 5 cubic yards equals 233280 cubic inches -/
theorem five_cubic_yards_to_cubic_inches :
  cubic_yards_to_cubic_inches 5 = 233280 := by
  sorry

end five_cubic_yards_to_cubic_inches_l3100_310060


namespace function_equality_l3100_310006

theorem function_equality (f : ℕ+ → ℕ+) :
  (∀ x y : ℕ+, f (x + y * f x) = x * f (y + 1)) →
  (∀ x : ℕ+, f x = x) :=
by sorry

end function_equality_l3100_310006


namespace no_non_zero_integer_solution_l3100_310003

theorem no_non_zero_integer_solution :
  ∀ (a b c n : ℤ), 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end no_non_zero_integer_solution_l3100_310003


namespace music_books_cost_l3100_310023

def total_budget : ℕ := 500
def maths_books : ℕ := 4
def maths_book_price : ℕ := 20
def science_book_price : ℕ := 10
def art_book_price : ℕ := 20

def science_books : ℕ := maths_books + 6
def art_books : ℕ := 2 * maths_books

def maths_cost : ℕ := maths_books * maths_book_price
def science_cost : ℕ := science_books * science_book_price
def art_cost : ℕ := art_books * art_book_price

def total_cost_except_music : ℕ := maths_cost + science_cost + art_cost

theorem music_books_cost (music_cost : ℕ) :
  music_cost = total_budget - total_cost_except_music →
  music_cost = 160 := by
  sorry

end music_books_cost_l3100_310023


namespace equal_color_squares_count_l3100_310045

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the 5x5 grid with a specific pattern of black cells -/
def Grid : Matrix (Fin 5) (Fin 5) Cell := sorry

/-- Checks if a sub-square has an equal number of black and white cells -/
def has_equal_colors (top_left : Fin 5 × Fin 5) (size : Nat) : Bool :=
  sorry

/-- Counts the number of sub-squares with equal black and white cells -/
def count_equal_color_squares (g : Matrix (Fin 5) (Fin 5) Cell) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem equal_color_squares_count :
  count_equal_color_squares Grid = 16 :=
sorry

end equal_color_squares_count_l3100_310045


namespace max_player_salary_l3100_310025

theorem max_player_salary (num_players : ℕ) (min_salary : ℕ) (max_total_salary : ℕ) :
  num_players = 18 →
  min_salary = 20000 →
  max_total_salary = 800000 →
  ∃ (max_single_salary : ℕ),
    max_single_salary = 460000 ∧
    max_single_salary + (num_players - 1) * min_salary ≤ max_total_salary ∧
    ∀ (salary : ℕ), salary + (num_players - 1) * min_salary ≤ max_total_salary → salary ≤ max_single_salary :=
by sorry

#check max_player_salary

end max_player_salary_l3100_310025


namespace solution_form_l3100_310064

/-- A continuous function satisfying the given integral equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  Continuous f ∧
  ∀ (x : ℝ) (n : ℕ), n ≠ 0 →
    (n : ℝ)^2 * ∫ t in x..(x + 1 / (n : ℝ)), f t = (n : ℝ) * f x + 1 / 2

/-- The theorem stating the form of functions satisfying the equation -/
theorem solution_form (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end solution_form_l3100_310064


namespace sally_peaches_theorem_l3100_310098

/-- The number of peaches Sally picked from the orchard -/
def peaches_picked (initial final : ℕ) : ℕ := final - initial

theorem sally_peaches_theorem (initial final picked : ℕ) 
  (h1 : initial = 13)
  (h2 : final = 55)
  (h3 : picked = peaches_picked initial final) :
  picked = 42 := by sorry

end sally_peaches_theorem_l3100_310098


namespace expression_evaluation_l3100_310002

theorem expression_evaluation : 
  (((3 : ℚ) + 6 + 9) / ((2 : ℚ) + 5 + 8) - ((2 : ℚ) + 5 + 8) / ((3 : ℚ) + 6 + 9)) = 11 / 30 := by
  sorry

end expression_evaluation_l3100_310002


namespace smallest_number_greater_than_digit_sum_by_1755_l3100_310067

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem smallest_number_greater_than_digit_sum_by_1755 :
  (∀ m : ℕ, m < 1770 → m ≠ sum_of_digits m + 1755) ∧
  1770 = sum_of_digits 1770 + 1755 := by
  sorry

end smallest_number_greater_than_digit_sum_by_1755_l3100_310067


namespace baseball_card_value_decrease_l3100_310071

theorem baseball_card_value_decrease (initial_value : ℝ) (h_initial_positive : initial_value > 0) : 
  let first_year_value := initial_value * (1 - 0.3)
  let total_decrease_percent := 0.37
  let second_year_decrease_percent := (initial_value * total_decrease_percent - (initial_value - first_year_value)) / first_year_value
  second_year_decrease_percent = 0.1 := by
sorry

end baseball_card_value_decrease_l3100_310071


namespace guanghua_community_households_l3100_310038

theorem guanghua_community_households (num_buildings : ℕ) (floors_per_building : ℕ) (households_per_floor : ℕ) 
  (h1 : num_buildings = 14)
  (h2 : floors_per_building = 7)
  (h3 : households_per_floor = 8) :
  num_buildings * floors_per_building * households_per_floor = 784 := by
  sorry

end guanghua_community_households_l3100_310038


namespace two_a_squared_eq_three_b_cubed_l3100_310074

theorem two_a_squared_eq_three_b_cubed (a b : ℕ+) :
  2 * a ^ 2 = 3 * b ^ 3 ↔ ∃ d : ℕ+, a = 18 * d ^ 3 ∧ b = 6 * d ^ 2 := by
  sorry

end two_a_squared_eq_three_b_cubed_l3100_310074


namespace inheritance_calculation_l3100_310058

-- Define the original inheritance amount
def original_inheritance : ℝ := 45500

-- Define the federal tax rate
def federal_tax_rate : ℝ := 0.25

-- Define the state tax rate
def state_tax_rate : ℝ := 0.15

-- Define the total tax paid
def total_tax_paid : ℝ := 16500

-- Theorem statement
theorem inheritance_calculation :
  let remaining_after_federal := original_inheritance * (1 - federal_tax_rate)
  let state_tax := remaining_after_federal * state_tax_rate
  let total_tax := original_inheritance * federal_tax_rate + state_tax
  total_tax = total_tax_paid :=
by sorry

end inheritance_calculation_l3100_310058


namespace sock_pair_selection_l3100_310091

def total_socks : ℕ := 20
def white_socks : ℕ := 6
def brown_socks : ℕ := 7
def blue_socks : ℕ := 3
def red_socks : ℕ := 4

theorem sock_pair_selection :
  (Nat.choose white_socks 2) +
  (Nat.choose brown_socks 2) +
  (Nat.choose blue_socks 2) +
  (red_socks * white_socks) +
  (red_socks * brown_socks) +
  (red_socks * blue_socks) +
  (Nat.choose red_socks 2) = 109 := by
  sorry

end sock_pair_selection_l3100_310091


namespace ticket_price_increase_l3100_310044

theorem ticket_price_increase (P V : ℝ) (h1 : P > 0) (h2 : V > 0) : 
  (P + 0.5 * P) * (0.8 * V) = 1.2 * (P * V) := by sorry

#check ticket_price_increase

end ticket_price_increase_l3100_310044


namespace max_sum_of_cubes_l3100_310079

/-- Given a system of equations, find the maximum value of x³ + y³ + z³ -/
theorem max_sum_of_cubes (x y z : ℝ) : 
  x^3 - x*y*z = 2 → 
  y^3 - x*y*z = 6 → 
  z^3 - x*y*z = 20 → 
  x^3 + y^3 + z^3 ≤ 151/7 := by
sorry

end max_sum_of_cubes_l3100_310079


namespace city_population_ratio_l3100_310031

theorem city_population_ratio (X Y Z : ℕ) (hY : Y = 2 * Z) (hX : X = 16 * Z) :
  X / Y = 8 := by
  sorry

end city_population_ratio_l3100_310031

import Mathlib

namespace NUMINAMATH_CALUDE_gcf_lcm_sum_l804_80461

theorem gcf_lcm_sum (A B : ℕ) : 
  (A = Nat.gcd 9 (Nat.gcd 15 27)) →
  (B = Nat.lcm 9 (Nat.lcm 15 27)) →
  A + B = 138 := by
sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_l804_80461


namespace NUMINAMATH_CALUDE_derivative_f_l804_80455

noncomputable def f (x : ℝ) : ℝ := 2 / x + x

theorem derivative_f :
  (∀ x : ℝ, x ≠ 0 → HasDerivAt f ((- 2 / x ^ 2) + 1) x) ∧
  HasDerivAt f (-1) 1 ∧
  HasDerivAt f (1/2) (-2) := by sorry

end NUMINAMATH_CALUDE_derivative_f_l804_80455


namespace NUMINAMATH_CALUDE_four_good_points_l804_80451

/-- A point (x, y) is a "good point" if x is an integer, y is a perfect square,
    and y = (x - 90)^2 - 4907 -/
def is_good_point (x y : ℤ) : Prop :=
  ∃ (m : ℤ), y = m^2 ∧ y = (x - 90)^2 - 4907

/-- The set of all "good points" -/
def good_points : Set (ℤ × ℤ) :=
  {p | is_good_point p.1 p.2}

/-- The theorem stating that there are exactly four "good points" -/
theorem four_good_points :
  good_points = {(444, 120409), (-264, 120409), (2544, 6017209), (-2364, 6017209)} := by
  sorry

#check four_good_points

end NUMINAMATH_CALUDE_four_good_points_l804_80451


namespace NUMINAMATH_CALUDE_stream_to_meadow_distance_l804_80448

/-- Given a hiking trip with known distances, prove the distance between two points -/
theorem stream_to_meadow_distance 
  (total_distance : ℝ)
  (car_to_stream : ℝ)
  (meadow_to_campsite : ℝ)
  (h1 : total_distance = 0.7)
  (h2 : car_to_stream = 0.2)
  (h3 : meadow_to_campsite = 0.1) :
  total_distance - car_to_stream - meadow_to_campsite = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_stream_to_meadow_distance_l804_80448


namespace NUMINAMATH_CALUDE_fraction_sum_equals_percentage_l804_80459

theorem fraction_sum_equals_percentage : (4/20 : ℚ) + (8/200 : ℚ) + (12/2000 : ℚ) = (246/1000 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_percentage_l804_80459


namespace NUMINAMATH_CALUDE_quadratic_real_roots_discriminant_nonnegative_l804_80499

theorem quadratic_real_roots_discriminant_nonnegative
  (a b c : ℝ) (ha : a ≠ 0)
  (h_real_roots : ∃ x : ℝ, a * x^2 + b * x + c = 0) :
  b^2 - 4*a*c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_discriminant_nonnegative_l804_80499


namespace NUMINAMATH_CALUDE_part_I_part_II_l804_80470

-- Define the statements p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

-- Part I
theorem part_I (m : ℝ) (h1 : m > 0) (h2 : ∀ x, p x → q m x) : m ≥ 4 := by
  sorry

-- Part II
theorem part_II (x : ℝ) (h1 : ∀ x, p x ∨ q 5 x) (h2 : ¬∀ x, p x ∧ q 5 x) :
  x ∈ Set.Icc (-3 : ℝ) (-2) ∪ Set.Ioc 6 7 := by
  sorry

end NUMINAMATH_CALUDE_part_I_part_II_l804_80470


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l804_80423

theorem smallest_number_with_conditions : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 4 = 2) ∧ 
  (n % 5 = 2) ∧ 
  (n % 6 = 2) ∧ 
  (n % 11 = 0) ∧ 
  (∀ m : ℕ, m > 1 → 
    (m % 3 = 2) → 
    (m % 4 = 2) → 
    (m % 5 = 2) → 
    (m % 6 = 2) → 
    (m % 11 = 0) → 
    m ≥ n) ∧
  n = 242 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l804_80423


namespace NUMINAMATH_CALUDE_ratio_equality_l804_80438

theorem ratio_equality (a b : ℝ) (h : 4 * a = 5 * b) : a / b = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l804_80438


namespace NUMINAMATH_CALUDE_no_simultaneous_negative_polynomials_l804_80422

theorem no_simultaneous_negative_polynomials :
  ∀ (m n : ℝ), ¬(3 * m^2 + 4 * m * n - 2 * n^2 < 0 ∧ -m^2 - 4 * m * n + 3 * n^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_negative_polynomials_l804_80422


namespace NUMINAMATH_CALUDE_train_length_calculation_l804_80453

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length_calculation (train_speed : Real) (crossing_time : Real) (bridge_length : Real) :
  let speed_ms : Real := train_speed * (1000 / 3600)
  let total_distance : Real := speed_ms * crossing_time
  let train_length : Real := total_distance - bridge_length
  train_speed = 45 ∧ crossing_time = 30 ∧ bridge_length = 230 →
  train_length = 145 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l804_80453


namespace NUMINAMATH_CALUDE_johns_remaining_money_l804_80443

theorem johns_remaining_money (initial_amount : ℚ) : 
  initial_amount = 200 → 
  initial_amount - (3/8 * initial_amount + 3/10 * initial_amount) = 65 := by
sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l804_80443


namespace NUMINAMATH_CALUDE_probability_both_presidents_selected_l804_80463

/-- Represents a math club with its total number of members -/
structure MathClub where
  members : Nat
  presidents : Nat
  mascots : Nat

/-- The list of math clubs in the district -/
def mathClubs : List MathClub := [
  { members := 6, presidents := 2, mascots := 1 },
  { members := 9, presidents := 2, mascots := 1 },
  { members := 10, presidents := 2, mascots := 1 },
  { members := 11, presidents := 2, mascots := 1 }
]

/-- The number of members to be selected from a club -/
def selectCount : Nat := 4

/-- Calculates the probability of selecting both presidents when choosing
    a specific number of members from a given club -/
def probBothPresidentsSelected (club : MathClub) (selectCount : Nat) : Rat :=
  sorry

/-- Calculates the overall probability of selecting both presidents when
    choosing from a randomly selected club -/
def overallProbability (clubs : List MathClub) (selectCount : Nat) : Rat :=
  sorry

/-- The main theorem stating the probability of selecting both presidents -/
theorem probability_both_presidents_selected :
  overallProbability mathClubs selectCount = 7/25 := by sorry

end NUMINAMATH_CALUDE_probability_both_presidents_selected_l804_80463


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l804_80442

/-- Two lines in R² defined by their parametric equations -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : Line2D) : Prop :=
  ∃ c : ℝ, l1.direction = (c * l2.direction.1, c * l2.direction.2)

/-- The problem statement -/
theorem parallel_lines_k_value :
  ∃! k : ℝ, are_parallel
    (Line2D.mk (2, 3) (6, -9))
    (Line2D.mk (-1, 0) (3, k))
  ∧ k = -4.5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l804_80442


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l804_80432

theorem floor_ceil_sum : ⌊(1.99 : ℝ)⌋ + ⌈(3.02 : ℝ)⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l804_80432


namespace NUMINAMATH_CALUDE_fish_pond_population_l804_80467

/-- Represents the total number of fish in a pond using the mark and recapture method. -/
def totalFishInPond (initialMarked : ℕ) (secondCatch : ℕ) (markedInSecondCatch : ℕ) : ℕ :=
  (initialMarked * secondCatch) / markedInSecondCatch

/-- Theorem stating that under the given conditions, the total number of fish in the pond is 2400. -/
theorem fish_pond_population :
  let initialMarked : ℕ := 80
  let secondCatch : ℕ := 150
  let markedInSecondCatch : ℕ := 5
  totalFishInPond initialMarked secondCatch markedInSecondCatch = 2400 := by
  sorry

#eval totalFishInPond 80 150 5

end NUMINAMATH_CALUDE_fish_pond_population_l804_80467


namespace NUMINAMATH_CALUDE_insect_leg_paradox_l804_80477

theorem insect_leg_paradox (total_legs : ℕ) (six_leg_insects : ℕ) (eight_leg_insects : ℕ) 
  (h1 : total_legs = 190)
  (h2 : 6 * six_leg_insects = 78)
  (h3 : 8 * eight_leg_insects = 24) :
  ¬∃ (ten_leg_insects : ℕ), 
    6 * six_leg_insects + 8 * eight_leg_insects + 10 * ten_leg_insects = total_legs :=
by sorry

end NUMINAMATH_CALUDE_insect_leg_paradox_l804_80477


namespace NUMINAMATH_CALUDE_house_of_cards_layers_l804_80425

/-- Calculates the maximum number of layers in a house of cards --/
def maxLayers (decks : ℕ) (cardsPerDeck : ℕ) (cardsPerLayer : ℕ) : ℕ :=
  (decks * cardsPerDeck) / cardsPerLayer

/-- Theorem: Given 16 decks of 52 cards each, using 26 cards per layer,
    the maximum number of layers in a house of cards is 32 --/
theorem house_of_cards_layers :
  maxLayers 16 52 26 = 32 := by
  sorry

end NUMINAMATH_CALUDE_house_of_cards_layers_l804_80425


namespace NUMINAMATH_CALUDE_fraction_equals_ratio_l804_80458

def numerator_terms : List Nat := [12, 28, 44, 60, 76]
def denominator_terms : List Nat := [8, 24, 40, 56, 72]

def fraction_term (n : Nat) : Rat :=
  (n^4 + 400 : Rat) / 1

theorem fraction_equals_ratio : 
  (List.prod (numerator_terms.map fraction_term)) / 
  (List.prod (denominator_terms.map fraction_term)) = 
  6712 / 148 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_ratio_l804_80458


namespace NUMINAMATH_CALUDE_mod_inverse_89_mod_90_l804_80417

theorem mod_inverse_89_mod_90 : ∃ x : ℕ, 0 ≤ x ∧ x < 90 ∧ (89 * x) % 90 = 1 :=
by
  use 89
  sorry

end NUMINAMATH_CALUDE_mod_inverse_89_mod_90_l804_80417


namespace NUMINAMATH_CALUDE_expression_bounds_l804_80478

theorem expression_bounds (a b c d e : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) : 
  2 * Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
    Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-e)^2) + 
    Real.sqrt (e^2 + (1-a)^2) ∧
  Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
  Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-e)^2) + 
  Real.sqrt (e^2 + (1-a)^2) ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l804_80478


namespace NUMINAMATH_CALUDE_classroom_population_classroom_population_is_8_l804_80489

theorem classroom_population : ℕ :=
  let student_count : ℕ := sorry
  let student_avg_age : ℚ := 8
  let total_avg_age : ℚ := 11
  let teacher_age : ℕ := 32

  have h1 : (student_count * student_avg_age + teacher_age) / (student_count + 1) = total_avg_age := by sorry

  student_count + 1

theorem classroom_population_is_8 : classroom_population = 8 := by sorry

end NUMINAMATH_CALUDE_classroom_population_classroom_population_is_8_l804_80489


namespace NUMINAMATH_CALUDE_quadratic_triangle_area_l804_80494

/-- Given a quadratic function y = ax^2 + bx + c where b^2 - 4ac > 0,
    the area of the triangle formed by its intersections with the x-axis and y-axis is |c|/(2|a|) -/
theorem quadratic_triangle_area (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let f : ℝ → ℝ := λ x => a*x^2 + b*x + c
  let triangle_area := (abs c) / (2 * abs a)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (1/2 * abs (x₁ - x₂) * abs c = triangle_area) :=
sorry

end NUMINAMATH_CALUDE_quadratic_triangle_area_l804_80494


namespace NUMINAMATH_CALUDE_class_average_theorem_l804_80485

theorem class_average_theorem (total_students : ℕ) (boys_percentage : ℚ) (girls_percentage : ℚ)
  (boys_score : ℚ) (girls_score : ℚ) :
  boys_percentage = 2/5 →
  girls_percentage = 3/5 →
  boys_score = 4/5 →
  girls_score = 9/10 →
  (boys_percentage * boys_score + girls_percentage * girls_score : ℚ) = 43/50 :=
by sorry

end NUMINAMATH_CALUDE_class_average_theorem_l804_80485


namespace NUMINAMATH_CALUDE_intersection_A_B_l804_80429

def A : Set ℝ := {x | x^2 - 4 > 0}
def B : Set ℝ := {x | x + 2 < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | x < -2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l804_80429


namespace NUMINAMATH_CALUDE_second_pump_rate_l804_80434

/-- Proves that the rate of the second pump is 70 gallons per hour given the conditions -/
theorem second_pump_rate (pump1_rate : ℝ) (total_time : ℝ) (total_volume : ℝ) (pump2_time : ℝ)
  (h1 : pump1_rate = 180)
  (h2 : total_time = 6)
  (h3 : total_volume = 1325)
  (h4 : pump2_time = 3.5) :
  (total_volume - pump1_rate * total_time) / pump2_time = 70 := by
  sorry

end NUMINAMATH_CALUDE_second_pump_rate_l804_80434


namespace NUMINAMATH_CALUDE_even_operations_l804_80436

theorem even_operations (n : ℤ) (h : Even n) :
  Even (5 * n) ∧ Even (n ^ 2) ∧ Even (n ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_even_operations_l804_80436


namespace NUMINAMATH_CALUDE_difference_of_squares_l804_80492

theorem difference_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l804_80492


namespace NUMINAMATH_CALUDE_function_expression_l804_80439

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (2 * x + 1) = x + 1) :
  ∀ x, f x = (1/2) * (x + 1) := by
sorry

end NUMINAMATH_CALUDE_function_expression_l804_80439


namespace NUMINAMATH_CALUDE_disneyland_arrangement_l804_80462

def number_of_arrangements (total : ℕ) (type_a : ℕ) (type_b : ℕ) : ℕ :=
  (Nat.factorial type_a) * (Nat.factorial type_b)

theorem disneyland_arrangement :
  let total := 6
  let type_a := 2
  let type_b := 4
  number_of_arrangements total type_a type_b = 48 := by
  sorry

end NUMINAMATH_CALUDE_disneyland_arrangement_l804_80462


namespace NUMINAMATH_CALUDE_polygon_sides_l804_80414

/-- 
A polygon has n sides. 
The sum of its interior angles is (n - 2) * 180°.
The sum of its exterior angles is 360°.
The sum of its interior angles is three times the sum of its exterior angles.
Prove that n = 8.
-/
theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l804_80414


namespace NUMINAMATH_CALUDE_propositions_p_and_q_l804_80491

theorem propositions_p_and_q : 
  (∃ a b c : ℝ, a < b ∧ a * c^2 ≥ b * c^2) ∧ 
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 - Real.log x₀ = 0) := by
  sorry

end NUMINAMATH_CALUDE_propositions_p_and_q_l804_80491


namespace NUMINAMATH_CALUDE_music_practice_time_calculation_l804_80433

/-- The total time Joan had for her music practice -/
def total_time : ℕ := 120

/-- Time spent on piano practice -/
def piano_time : ℕ := 30

/-- Time spent writing music -/
def writing_time : ℕ := 25

/-- Time spent reading about piano history -/
def reading_time : ℕ := 38

/-- Time left for finger exerciser -/
def exerciser_time : ℕ := 27

/-- Theorem stating that the total time is equal to the sum of individual activities -/
theorem music_practice_time_calculation :
  total_time = piano_time + writing_time + reading_time + exerciser_time := by
  sorry

end NUMINAMATH_CALUDE_music_practice_time_calculation_l804_80433


namespace NUMINAMATH_CALUDE_euro_problem_l804_80430

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- Theorem statement
theorem euro_problem (a : ℝ) :
  euro a (euro 4 5) = 640 → a = 8 := by
  sorry

end NUMINAMATH_CALUDE_euro_problem_l804_80430


namespace NUMINAMATH_CALUDE_characterize_satisfying_functions_l804_80431

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

/-- The theorem stating the form of functions satisfying the equation -/
theorem characterize_satisfying_functions
  (f : ℝ → ℝ)
  (h_smooth : ContDiff ℝ ⊤ f)
  (h_satisfies : SatisfiesEquation f) :
  ∃ a : ℝ, ∀ x : ℝ, f x = x^2 + a * x :=
sorry

end NUMINAMATH_CALUDE_characterize_satisfying_functions_l804_80431


namespace NUMINAMATH_CALUDE_ab_plus_cd_value_l804_80415

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 3)
  (eq2 : a + b + d = -1)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = 0) :
  a * b + c * d = -127 / 9 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_value_l804_80415


namespace NUMINAMATH_CALUDE_total_steps_rachel_l804_80420

theorem total_steps_rachel (steps_up steps_down : ℕ) 
  (h1 : steps_up = 567) 
  (h2 : steps_down = 325) : 
  steps_up + steps_down = 892 := by
sorry

end NUMINAMATH_CALUDE_total_steps_rachel_l804_80420


namespace NUMINAMATH_CALUDE_fraction_inequality_l804_80421

theorem fraction_inequality (a b : ℝ) : a < b → b < 0 → (1 : ℝ) / a > (1 : ℝ) / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l804_80421


namespace NUMINAMATH_CALUDE_flag_arrangement_remainder_l804_80473

/-- The number of distinguishable arrangements of flags on two poles -/
def N : ℕ :=
  let blue_flags := 10
  let green_flags := 9
  let total_flags := blue_flags + green_flags
  let poles := 2
  -- Definition of N based on the problem conditions
  -- (Actual calculation is omitted as it's part of the proof)
  2310

/-- Theorem stating that N mod 1000 = 310 -/
theorem flag_arrangement_remainder :
  N % 1000 = 310 := by
  sorry

end NUMINAMATH_CALUDE_flag_arrangement_remainder_l804_80473


namespace NUMINAMATH_CALUDE_french_not_english_speakers_l804_80469

/-- The number of students who speak French but not English in a survey -/
theorem french_not_english_speakers (total : ℕ) (french_speakers : ℕ) (both_speakers : ℕ) 
  (h1 : total = 200)
  (h2 : french_speakers = total / 4)
  (h3 : both_speakers = 10) :
  french_speakers - both_speakers = 40 := by
  sorry

end NUMINAMATH_CALUDE_french_not_english_speakers_l804_80469


namespace NUMINAMATH_CALUDE_solution_set_inequality_l804_80480

theorem solution_set_inequality (x : ℝ) : 
  (x ≠ -1 ∧ (2*x - 1)/(x + 1) ≤ 1) ↔ -1 < x ∧ x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l804_80480


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l804_80486

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 25 (2 * x) = Nat.choose 25 (x + 4)) ↔ (x = 4 ∨ x = 7) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l804_80486


namespace NUMINAMATH_CALUDE_flight_duration_sum_l804_80428

/-- Represents a time with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes, accounting for day change -/
def timeDiffMinutes (t1 t2 : Time) : ℕ :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  if totalMinutes2 < totalMinutes1 then
    (24 * 60 - totalMinutes1) + totalMinutes2
  else
    totalMinutes2 - totalMinutes1

theorem flight_duration_sum (departure : Time) (arrival : Time) (h m : ℕ) :
  departure.hours = 17 ∧ departure.minutes = 30 ∧
  arrival.hours = 2 ∧ arrival.minutes = 15 ∧
  0 < m ∧ m < 60 ∧
  timeDiffMinutes departure arrival + 3 * 60 = h * 60 + m →
  h + m = 56 := by
  sorry

end NUMINAMATH_CALUDE_flight_duration_sum_l804_80428


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_locus_l804_80437

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the square of the distance between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem: Locus of points for isosceles right triangle -/
theorem isosceles_right_triangle_locus (s : ℝ) (h : s > 0) :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨s, 0⟩
  let C : Point := ⟨0, s⟩
  let center : Point := ⟨s/3, s/3⟩
  let radius : ℝ := Real.sqrt (s^2/3)
  ∀ P : Point, 
    (distanceSquared P A + distanceSquared P B + distanceSquared P C = 4 * s^2) ↔ 
    (distanceSquared P center = radius^2) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_locus_l804_80437


namespace NUMINAMATH_CALUDE_james_payment_is_six_l804_80426

/-- Calculates James's share of the payment for stickers -/
def jamesPayment (packs : ℕ) (stickersPerPack : ℕ) (stickerCost : ℚ) (friendSharePercent : ℚ) : ℚ :=
  let totalStickers := packs * stickersPerPack
  let totalCost := totalStickers * stickerCost
  totalCost * (1 - friendSharePercent)

/-- Proves that James pays $6 for his share of the stickers -/
theorem james_payment_is_six :
  jamesPayment 4 30 (1/10) (1/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_james_payment_is_six_l804_80426


namespace NUMINAMATH_CALUDE_maria_friends_money_l804_80403

def problem (maria_total rene_amount : ℚ) : Prop :=
  let isha_amount := maria_total / 4
  let florence_amount := isha_amount / 2
  let john_amount := florence_amount / 3
  florence_amount = 4 * rene_amount ∧
  rene_amount = 450 ∧
  isha_amount + florence_amount + rene_amount + john_amount = 6450

theorem maria_friends_money :
  ∃ maria_total : ℚ, problem maria_total 450 :=
sorry

end NUMINAMATH_CALUDE_maria_friends_money_l804_80403


namespace NUMINAMATH_CALUDE_largest_divisor_of_sequence_l804_80464

theorem largest_divisor_of_sequence (n : ℕ) : ∃ (k : ℕ), k = 30 ∧ k ∣ (n^5 - n) ∧ ∀ m : ℕ, m > k → ¬(∀ n : ℕ, m ∣ (n^5 - n)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_sequence_l804_80464


namespace NUMINAMATH_CALUDE_beth_crayons_left_l804_80445

/-- The number of crayons Beth has left after giving some away -/
def crayons_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Proof that Beth has 52 crayons left -/
theorem beth_crayons_left :
  let initial_crayons : ℕ := 106
  let crayons_given_away : ℕ := 54
  crayons_left initial_crayons crayons_given_away = 52 := by
sorry

end NUMINAMATH_CALUDE_beth_crayons_left_l804_80445


namespace NUMINAMATH_CALUDE_distance_AD_MN_l804_80441

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents the pyramid structure described in the problem -/
structure Pyramid where
  a : ℝ
  b : ℝ
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  M : Point3D
  N : Point3D

/-- The distance between two skew lines in 3D space -/
def distanceBetweenSkewLines (l1 l2 : Line3D) : ℝ :=
  sorry

/-- The main theorem stating the distance between AD and MN -/
theorem distance_AD_MN (p : Pyramid) :
  let AD := Line3D.mk p.A (Point3D.mk p.a p.a 0)
  let MN := Line3D.mk p.M (Point3D.mk 0 (p.a / 2) p.b)
  distanceBetweenSkewLines AD MN = (p.b / (2 * p.a)) * Real.sqrt (4 * p.a^2 - p.b^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_AD_MN_l804_80441


namespace NUMINAMATH_CALUDE_division_problem_l804_80406

theorem division_problem (n : ℕ) : 
  n / 15 = 9 ∧ n % 15 = 1 → n = 136 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l804_80406


namespace NUMINAMATH_CALUDE_cos_alpha_minus_beta_l804_80424

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : 2 * Real.cos α - Real.cos β = 3/2)
  (h2 : 2 * Real.sin α - Real.sin β = 2) :
  Real.cos (α - β) = -5/16 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_beta_l804_80424


namespace NUMINAMATH_CALUDE_power_division_equals_729_l804_80409

theorem power_division_equals_729 : 3^12 / 27^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equals_729_l804_80409


namespace NUMINAMATH_CALUDE_f_three_minus_f_four_equals_negative_one_l804_80497

-- Define the properties of function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_two_negation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

-- Theorem statement
theorem f_three_minus_f_four_equals_negative_one
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : has_period_two_negation f)
  (h_f_one : f 1 = 1) :
  f 3 - f 4 = -1 :=
sorry

end NUMINAMATH_CALUDE_f_three_minus_f_four_equals_negative_one_l804_80497


namespace NUMINAMATH_CALUDE_root_ratio_quadratic_equation_l804_80488

theorem root_ratio_quadratic_equation (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x/y = 2/3 ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0) :
  6*b^2 = 25*a*c := by
  sorry

end NUMINAMATH_CALUDE_root_ratio_quadratic_equation_l804_80488


namespace NUMINAMATH_CALUDE_cnc_processing_time_l804_80401

/-- The time required for one CNC machine to process a given number of parts, 
    given the rate of multiple machines. -/
theorem cnc_processing_time 
  (machines : ℕ) 
  (parts : ℕ) 
  (hours : ℕ) 
  (target_parts : ℕ) : 
  machines > 0 → 
  parts > 0 → 
  hours > 0 → 
  target_parts > 0 → 
  (3 : ℕ) = machines → 
  (960 : ℕ) = parts → 
  (4 : ℕ) = hours → 
  (400 : ℕ) = target_parts → 
  (5 : ℕ) = (target_parts * machines * hours) / parts := by
  sorry


end NUMINAMATH_CALUDE_cnc_processing_time_l804_80401


namespace NUMINAMATH_CALUDE_peggy_record_profit_difference_l804_80457

/-- Represents the profit difference between two offers for a record collection. -/
def profit_difference (total_records : ℕ) (sammy_price : ℕ) (bryan_price_high : ℕ) (bryan_price_low : ℕ) : ℕ :=
  let sammy_offer := total_records * sammy_price
  let bryan_offer := (total_records / 2) * bryan_price_high + (total_records / 2) * bryan_price_low
  sammy_offer - bryan_offer

/-- Theorem stating the profit difference for Peggy's record collection. -/
theorem peggy_record_profit_difference :
  profit_difference 200 4 6 1 = 100 := by
  sorry

end NUMINAMATH_CALUDE_peggy_record_profit_difference_l804_80457


namespace NUMINAMATH_CALUDE_intersection_A_B_l804_80474

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {y | ∃ x ∈ A, y = 3 * x - 2}

theorem intersection_A_B : A ∩ B = {1, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l804_80474


namespace NUMINAMATH_CALUDE_cubic_factorization_l804_80452

theorem cubic_factorization (x : ℝ) : x^3 - 16*x = x*(x+4)*(x-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l804_80452


namespace NUMINAMATH_CALUDE_election_result_l804_80405

/-- The percentage of votes received by candidate A out of the total valid votes -/
def candidate_A_percentage : ℝ := 65

/-- The percentage of invalid votes out of the total votes -/
def invalid_vote_percentage : ℝ := 15

/-- The total number of votes cast in the election -/
def total_votes : ℕ := 560000

/-- The number of valid votes polled in favor of candidate A -/
def votes_for_candidate_A : ℕ := 309400

theorem election_result :
  (candidate_A_percentage / 100) * ((100 - invalid_vote_percentage) / 100) * total_votes = votes_for_candidate_A := by
  sorry

end NUMINAMATH_CALUDE_election_result_l804_80405


namespace NUMINAMATH_CALUDE_smallest_multiple_of_36_and_45_not_11_l804_80483

theorem smallest_multiple_of_36_and_45_not_11 : 
  ∃ (n : ℕ), n > 0 ∧ 36 ∣ n ∧ 45 ∣ n ∧ ¬(11 ∣ n) ∧
  ∀ (m : ℕ), m > 0 ∧ 36 ∣ m ∧ 45 ∣ m ∧ ¬(11 ∣ m) → n ≤ m :=
by
  -- The proof would go here
  sorry

#check smallest_multiple_of_36_and_45_not_11

end NUMINAMATH_CALUDE_smallest_multiple_of_36_and_45_not_11_l804_80483


namespace NUMINAMATH_CALUDE_expected_red_lights_l804_80446

-- Define the number of intersections
def num_intersections : ℕ := 3

-- Define the probability of encountering a red light at each intersection
def red_light_prob : ℝ := 0.3

-- State the theorem
theorem expected_red_lights :
  let num_intersections : ℕ := 3
  let red_light_prob : ℝ := 0.3
  (num_intersections : ℝ) * red_light_prob = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_lights_l804_80446


namespace NUMINAMATH_CALUDE_f_g_2_equals_22_l804_80416

-- Define the functions g and f
def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3*x - 2

-- State the theorem
theorem f_g_2_equals_22 : f (g 2) = 22 := by
  sorry

end NUMINAMATH_CALUDE_f_g_2_equals_22_l804_80416


namespace NUMINAMATH_CALUDE_equivalence_of_propositions_l804_80460

theorem equivalence_of_propositions (a b c : ℝ) :
  (a < b → a + c < b + c) ∧
  (a + c < b + c → a < b) ∧
  (a ≥ b → a + c ≥ b + c) ∧
  (a + c ≥ b + c → a ≥ b) := by
  sorry

end NUMINAMATH_CALUDE_equivalence_of_propositions_l804_80460


namespace NUMINAMATH_CALUDE_inequality_always_true_range_l804_80454

theorem inequality_always_true_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_always_true_range_l804_80454


namespace NUMINAMATH_CALUDE_dispersion_measure_properties_l804_80487

-- Define a type for datasets
structure Dataset where
  data : List ℝ

-- Define a type for dispersion measures
structure DispersionMeasure where
  measure : Dataset → ℝ

-- Statement 1: Multiple values can be used to describe the degree of dispersion
def multipleValuesUsed (d : DispersionMeasure) : Prop :=
  ∃ (d1 d2 : DispersionMeasure), d1 ≠ d2

-- Statement 2: One should make full use of the obtained data
def fullDataUsed (d : DispersionMeasure) : Prop :=
  ∀ (dataset : Dataset), d.measure dataset = d.measure dataset

-- Statement 3: For different datasets, when the degree of dispersion is large, this value should be smaller (incorrect statement)
def incorrectDispersionRelation (d : DispersionMeasure) : Prop :=
  ∃ (dataset1 dataset2 : Dataset),
    d.measure dataset1 > d.measure dataset2 →
    d.measure dataset1 < d.measure dataset2

theorem dispersion_measure_properties :
  ∃ (d : DispersionMeasure),
    multipleValuesUsed d ∧
    fullDataUsed d ∧
    ¬incorrectDispersionRelation d :=
  sorry

end NUMINAMATH_CALUDE_dispersion_measure_properties_l804_80487


namespace NUMINAMATH_CALUDE_min_value_sum_l804_80498

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  a + 2*b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l804_80498


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l804_80482

theorem consecutive_integers_product_sum (n : ℤ) : 
  (n - 1) * n * (n + 1) = 336 → (n - 1) + n + (n + 1) = 21 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l804_80482


namespace NUMINAMATH_CALUDE_factorization_x4_minus_4x2_l804_80400

theorem factorization_x4_minus_4x2 (x : ℝ) : x^4 - 4*x^2 = x^2 * (x - 2) * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_4x2_l804_80400


namespace NUMINAMATH_CALUDE_ellipse_problem_l804_80419

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    foci F₁ and F₂, and a point P on the ellipse such that PF₁ ⊥ PF₂,
    if the area of triangle PF₁F₂ is 9, then b = 3 -/
theorem ellipse_problem (a b : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧
  (F₁.1^2 / a^2 + F₁.2^2 / b^2 = 1) ∧
  (F₂.1^2 / a^2 + F₂.2^2 / b^2 = 1) ∧
  ((P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0) ∧
  (1/2 * abs ((P.1 - F₁.1) * (P.2 - F₂.2) - (P.2 - F₁.2) * (P.1 - F₂.1)) = 9) →
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_problem_l804_80419


namespace NUMINAMATH_CALUDE_age_determination_l804_80471

def binary_sum (n : ℕ) : Prop :=
  ∃ (a b c d : Bool),
    n = (if a then 1 else 0) + 
        (if b then 2 else 0) + 
        (if c then 4 else 0) + 
        (if d then 8 else 0)

theorem age_determination (n : ℕ) (h : n < 16) : binary_sum n := by
  sorry

end NUMINAMATH_CALUDE_age_determination_l804_80471


namespace NUMINAMATH_CALUDE_agricultural_experiment_l804_80481

theorem agricultural_experiment (seeds_second_plot : ℕ) : 
  (300 : ℝ) * 0.30 + seeds_second_plot * 0.35 = (300 + seeds_second_plot) * 0.32 →
  seeds_second_plot = 200 := by
sorry

end NUMINAMATH_CALUDE_agricultural_experiment_l804_80481


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l804_80418

theorem sum_reciprocals_bound (n : ℕ) (h : n > 1) :
  (Finset.range (2^n - 1)).sum (fun i => 1 / (i + 1 : ℝ)) < n := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l804_80418


namespace NUMINAMATH_CALUDE_savings_difference_l804_80490

/-- The original price of the office supplies -/
def original_price : ℝ := 15000

/-- The first discount rate in the successive discounts option -/
def discount1 : ℝ := 0.30

/-- The second discount rate in the successive discounts option -/
def discount2 : ℝ := 0.15

/-- The single discount rate in the alternative option -/
def single_discount : ℝ := 0.40

/-- The price after applying two successive discounts -/
def price_after_successive_discounts : ℝ :=
  original_price * (1 - discount1) * (1 - discount2)

/-- The price after applying a single discount -/
def price_after_single_discount : ℝ :=
  original_price * (1 - single_discount)

/-- Theorem stating the difference in savings between the two discount options -/
theorem savings_difference :
  price_after_single_discount - price_after_successive_discounts = 75 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_l804_80490


namespace NUMINAMATH_CALUDE_expression_evaluation_l804_80472

theorem expression_evaluation :
  let x : ℚ := 3/2
  (2 + x) * (2 - x) + (x - 1) * (x + 5) = 5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l804_80472


namespace NUMINAMATH_CALUDE_bench_press_difference_is_fifty_l804_80468

/-- Represents the bench press capabilities of three individuals -/
structure BenchPress where
  dave_weight : ℝ
  dave_multiplier : ℝ
  craig_percentage : ℝ
  mark_weight : ℝ

/-- Calculates the difference between Craig's and Mark's bench press weights -/
def bench_press_difference (bp : BenchPress) : ℝ :=
  bp.dave_weight * bp.dave_multiplier * bp.craig_percentage - bp.mark_weight

/-- Theorem stating the difference between Craig's and Mark's bench press weights -/
theorem bench_press_difference_is_fifty (bp : BenchPress) 
  (h1 : bp.dave_weight = 175)
  (h2 : bp.dave_multiplier = 3)
  (h3 : bp.craig_percentage = 0.2)
  (h4 : bp.mark_weight = 55) :
  bench_press_difference bp = 50 := by
  sorry

#eval bench_press_difference { dave_weight := 175, dave_multiplier := 3, craig_percentage := 0.2, mark_weight := 55 }

end NUMINAMATH_CALUDE_bench_press_difference_is_fifty_l804_80468


namespace NUMINAMATH_CALUDE_polynomial_simplification_l804_80408

theorem polynomial_simplification (x : ℝ) :
  (2 * x^3 - 5 * x^2 + 8 * x - 9) + (3 * x^4 - 2 * x^3 + x^2 - 8 * x + 6) = 3 * x^4 - 4 * x^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l804_80408


namespace NUMINAMATH_CALUDE_journey_length_l804_80404

theorem journey_length : 
  ∀ (x : ℚ), 
  (x / 4 : ℚ) + 25 + (x / 6 : ℚ) = x → 
  x = 300 / 7 := by
sorry

end NUMINAMATH_CALUDE_journey_length_l804_80404


namespace NUMINAMATH_CALUDE_stating_ancient_chinese_problem_correct_l804_80447

/-- Represents the system of equations for the ancient Chinese mathematical problem. -/
def ancient_chinese_problem (x y : ℝ) : Prop :=
  (y = 8 * x - 3) ∧ (y = 7 * x + 4)

/-- 
Theorem stating that the system of equations correctly represents the given problem,
where x is the number of people and y is the price of the items in coins.
-/
theorem ancient_chinese_problem_correct (x y : ℝ) :
  ancient_chinese_problem x y ↔
  (∃ (total_price : ℝ),
    (8 * x = total_price + 3) ∧
    (7 * x = total_price - 4) ∧
    (y = total_price)) :=
by sorry

end NUMINAMATH_CALUDE_stating_ancient_chinese_problem_correct_l804_80447


namespace NUMINAMATH_CALUDE_max_profit_and_volume_l804_80456

/-- Represents the annual profit function for a company producing a certain product. -/
noncomputable def annual_profit (x : ℝ) : ℝ :=
  if x < 80 then
    -(1/360) * x^3 + 30*x - 250
  else
    1200 - (x + 10000/x)

/-- Theorem stating the maximum annual profit and the production volume that achieves it. -/
theorem max_profit_and_volume :
  (∃ (max_profit : ℝ) (optimal_volume : ℝ),
    max_profit = 1000 ∧
    optimal_volume = 100 ∧
    ∀ x, x > 0 → annual_profit x ≤ max_profit ∧
    annual_profit optimal_volume = max_profit) :=
sorry


end NUMINAMATH_CALUDE_max_profit_and_volume_l804_80456


namespace NUMINAMATH_CALUDE_at_least_one_intersection_l804_80402

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of a line lying in a plane
variable (lies_in : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the property of a line intersecting another line
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem at_least_one_intersection 
  (a b c : Line) (α β : Plane)
  (h1 : skew a b)
  (h2 : lies_in a α)
  (h3 : lies_in b β)
  (h4 : c = intersect α β) :
  intersects c a ∨ intersects c b :=
sorry

end NUMINAMATH_CALUDE_at_least_one_intersection_l804_80402


namespace NUMINAMATH_CALUDE_work_comparison_l804_80407

/-- Represents the amount of work that can be done by a group of people in a given time -/
structure WorkCapacity where
  people : ℕ
  work : ℝ
  days : ℕ

/-- Given two work capacities, proves that the first group did twice the initially considered work -/
theorem work_comparison (w1 w2 : WorkCapacity) : 
  w1.people = 3 ∧ 
  w1.days = 3 ∧ 
  w2.people = 6 ∧ 
  w2.days = 3 ∧ 
  w2.work = 6 * w1.work → 
  w1.work = 2 * w1.work := by
sorry

end NUMINAMATH_CALUDE_work_comparison_l804_80407


namespace NUMINAMATH_CALUDE_parabola_reflection_l804_80449

/-- Given a parabola y = x^2 and a line y = x + 2, prove that the reflection of the parabola about the line is x = y^2 - 4y + 2 -/
theorem parabola_reflection (x y : ℝ) :
  (y = x^2) ∧ (∃ (x' y' : ℝ), y' = x' + 2 ∧ 
    ((x' = y - 2 ∧ y' = x + 2) ∨ (x' = x ∧ y' = y))) →
  x = y^2 - 4*y + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_reflection_l804_80449


namespace NUMINAMATH_CALUDE_f_properties_l804_80475

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 4^(-x) - a * 2^(-x) else 4^x - a * 2^x

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem f_properties (a : ℝ) (h : a > 0) :
  is_even_function (f a) ∧
  (∀ x > 0, f a x = 4^x - a * 2^x) ∧
  (∀ x > 0, f a x ≥ 
    if 0 < a ∧ a ≤ 2 then 1 - a
    else if a > 2 then -a^2 / 4
    else 0) ∧
  (∃ x > 0, f a x = 
    if 0 < a ∧ a ≤ 2 then 1 - a
    else if a > 2 then -a^2 / 4
    else 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l804_80475


namespace NUMINAMATH_CALUDE_a_value_l804_80427

theorem a_value (a : ℚ) (h : a + a/4 = 10/4) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l804_80427


namespace NUMINAMATH_CALUDE_max_terms_arithmetic_sequence_l804_80466

/-- Given an arithmetic sequence {a_n} with a_1 ∈ ℝ, common difference d = 2,
    and S as the sum of all terms, if a_1^2 + S ≤ 96,
    then the maximum number of terms in the sequence is 12. -/
theorem max_terms_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (S : ℝ) (n : ℕ) :
  d = 2 →
  S = n * a₁ + n * (n - 1) / 2 * d →
  a₁^2 + S ≤ 96 →
  n ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_terms_arithmetic_sequence_l804_80466


namespace NUMINAMATH_CALUDE_window_installation_time_l804_80413

theorem window_installation_time (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ) :
  total_windows = 10 →
  installed_windows = 6 →
  time_per_window = 5 →
  (total_windows - installed_windows) * time_per_window = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_window_installation_time_l804_80413


namespace NUMINAMATH_CALUDE_six_point_configuration_exists_l804_80440

/-- A configuration of six points in 3D space -/
def Configuration := Fin 6 → ℝ × ℝ × ℝ

/-- Predicate to check if two line segments intersect only at their endpoints -/
def valid_intersection (a b c d : ℝ × ℝ × ℝ) : Prop :=
  (a = c ∧ b ≠ d) ∨ (a = d ∧ b ≠ c) ∨ (b = c ∧ a ≠ d) ∨ (b = d ∧ a ≠ c) ∨ (a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d)

/-- Predicate to check if a configuration is valid -/
def valid_configuration (config : Configuration) : Prop :=
  ∀ i j k l : Fin 6, i ≠ j → k ≠ l → i ≠ k ∨ i ≠ l → j ≠ k ∨ j ≠ l →
    valid_intersection (config i) (config j) (config k) (config l)

theorem six_point_configuration_exists : ∃ config : Configuration, valid_configuration config :=
sorry

end NUMINAMATH_CALUDE_six_point_configuration_exists_l804_80440


namespace NUMINAMATH_CALUDE_linear_function_is_shifted_odd_exponential_function_is_not_shifted_odd_sine_shifted_odd_condition_cubic_function_not_shifted_odd_condition_l804_80493

/-- A function is a shifted odd function if there exists a real number m such that
    f(x+m) - f(m) is an odd function over ℝ. -/
def is_shifted_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, f (x + m) - f m = -(f (-x + m) - f m)

/-- The function f(x) = 2x + 1 is a shifted odd function. -/
theorem linear_function_is_shifted_odd :
  is_shifted_odd_function (fun x => 2 * x + 1) :=
sorry

/-- The function g(x) = 2^x is not a shifted odd function. -/
theorem exponential_function_is_not_shifted_odd :
  ¬ is_shifted_odd_function (fun x => 2^x) :=
sorry

/-- For f(x) = sin(x + φ) to be a shifted odd function with shift difference π/4,
    φ must equal kπ - π/4 for some integer k. -/
theorem sine_shifted_odd_condition (φ : ℝ) :
  is_shifted_odd_function (fun x => Real.sin (x + φ)) ∧ 
  (∃ m : ℝ, m = π/4 ∧ ∀ x : ℝ, Real.sin (x + m + φ) - Real.sin (m + φ) = -(Real.sin (-x + m + φ) - Real.sin (m + φ))) ↔
  ∃ k : ℤ, φ = k * π - π/4 :=
sorry

/-- For f(x) = x^3 + bx^2 + cx to not be a shifted odd function for any m in [-1/2, +∞),
    b must be greater than 3/2, and c can be any real number. -/
theorem cubic_function_not_shifted_odd_condition (b c : ℝ) :
  (∀ m : ℝ, m ≥ -1/2 → ¬ is_shifted_odd_function (fun x => x^3 + b*x^2 + c*x)) ↔
  b > 3/2 :=
sorry

end NUMINAMATH_CALUDE_linear_function_is_shifted_odd_exponential_function_is_not_shifted_odd_sine_shifted_odd_condition_cubic_function_not_shifted_odd_condition_l804_80493


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l804_80435

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x| ≤ 2
def q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, q x → p x) ∧ 
  (∃ x : ℝ, p x ∧ ¬(q x)) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l804_80435


namespace NUMINAMATH_CALUDE_square_side_length_l804_80496

/-- Given a rectangle formed by three squares and two other rectangles, 
    prove that the middle square has a side length of 651 -/
theorem square_side_length (s₁ s₂ s₃ : ℕ) : 
  s₁ + s₂ + s₃ = 3322 →
  s₁ - s₂ + s₃ = 2020 →
  s₂ = 651 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l804_80496


namespace NUMINAMATH_CALUDE_correct_division_formula_l804_80465

theorem correct_division_formula : 
  (240 : ℕ) / (13 + 11) = 240 / (13 + 11) := by sorry

end NUMINAMATH_CALUDE_correct_division_formula_l804_80465


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_products_l804_80444

def consecutive_product (n : ℕ) (count : ℕ) : ℕ :=
  (List.range count).foldl (λ acc _ => acc * n) 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_sum_of_products : 
  units_digit (consecutive_product 2017 2016 + consecutive_product 2016 2017) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_products_l804_80444


namespace NUMINAMATH_CALUDE_remainder_3_pow_2017_mod_17_l804_80476

theorem remainder_3_pow_2017_mod_17 : 3^2017 % 17 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2017_mod_17_l804_80476


namespace NUMINAMATH_CALUDE_worker_y_fraction_l804_80410

theorem worker_y_fraction (fx fy : ℝ) : 
  fx + fy = 1 →
  0.005 * fx + 0.008 * fy = 0.0074 →
  fy = 0.8 := by
sorry

end NUMINAMATH_CALUDE_worker_y_fraction_l804_80410


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l804_80412

theorem perfect_square_polynomial (n : ℤ) : 
  (∃ k : ℤ, n^4 + 6*n^3 + 11*n^2 + 3*n + 31 = k^2) ↔ n = 10 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l804_80412


namespace NUMINAMATH_CALUDE_pattern_holds_squares_in_figure_150_l804_80495

/-- The number of unit squares in figure n -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The sequence of unit squares follows the given pattern for the first four figures -/
theorem pattern_holds : f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 := by sorry

/-- The number of unit squares in figure 150 is 67951 -/
theorem squares_in_figure_150 : f 150 = 67951 := by sorry

end NUMINAMATH_CALUDE_pattern_holds_squares_in_figure_150_l804_80495


namespace NUMINAMATH_CALUDE_arithmetic_sequence_partial_sum_l804_80484

-- Define the arithmetic sequence and its partial sums
def arithmetic_sequence (n : ℕ) : ℝ := sorry
def S (n : ℕ) : ℝ := sorry

-- State the theorem
theorem arithmetic_sequence_partial_sum :
  S 3 = 6 ∧ S 9 = 27 → S 6 = 15 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_partial_sum_l804_80484


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l804_80479

theorem exactly_one_greater_than_one 
  (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (prod_eq_one : a * b * c = 1)
  (sum_gt_recip_sum : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l804_80479


namespace NUMINAMATH_CALUDE_reflection_of_S_l804_80411

-- Define the reflection across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the reflection across the line y = -x
def reflect_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

-- Define the composition of both reflections
def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_neg_x (reflect_x_axis p)

-- Theorem statement
theorem reflection_of_S :
  double_reflection (5, 0) = (0, -5) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_S_l804_80411


namespace NUMINAMATH_CALUDE_ordering_abc_l804_80450

theorem ordering_abc (a b c : ℝ) : 
  a = 6 - Real.log 2 - Real.log 3 →
  b = Real.exp 1 - Real.log 3 →
  c = Real.exp 2 - 2 →
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l804_80450

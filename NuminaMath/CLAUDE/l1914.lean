import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1914_191482

theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : w = 6) : 
  let triangle_area := (1/2) * a * b
  let rectangle_length := triangle_area / w
  2 * (rectangle_length + w) = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1914_191482


namespace NUMINAMATH_CALUDE_unique_number_l1914_191496

theorem unique_number : ∃! x : ℚ, x / 3 = x - 4 := by sorry

end NUMINAMATH_CALUDE_unique_number_l1914_191496


namespace NUMINAMATH_CALUDE_population_closest_to_target_in_2060_l1914_191441

def initial_population : ℕ := 500
def growth_rate : ℕ := 4
def years_per_growth : ℕ := 30
def target_population : ℕ := 10000
def initial_year : ℕ := 2000

def population_at_year (year : ℕ) : ℕ :=
  initial_population * growth_rate ^ ((year - initial_year) / years_per_growth)

theorem population_closest_to_target_in_2060 :
  ∀ year : ℕ, year ≤ 2060 → population_at_year year ≤ target_population ∧
  population_at_year 2060 > population_at_year (2060 - years_per_growth) ∧
  population_at_year (2060 + years_per_growth) > target_population :=
by sorry

end NUMINAMATH_CALUDE_population_closest_to_target_in_2060_l1914_191441


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1914_191440

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x < 23 ∧ (1054 + x) % 23 = 0 ∧ ∀ y : ℕ, y < x → (1054 + y) % 23 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1914_191440


namespace NUMINAMATH_CALUDE_find_divisor_l1914_191449

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 4 + remainder) :
  ∃ (divisor : ℕ), dividend = quotient * divisor + remainder ∧ divisor = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1914_191449


namespace NUMINAMATH_CALUDE_otimes_composition_l1914_191412

-- Define the binary operation ⊗
def otimes (x y : ℝ) : ℝ := x^3 + y^3 - x - y

-- Theorem statement
theorem otimes_composition (a b : ℝ) : 
  otimes a (otimes b a) = a^3 + (b^3 + a^3 - b - a)^3 - a - (b^3 + a^3 - b - a) := by
  sorry

end NUMINAMATH_CALUDE_otimes_composition_l1914_191412


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l1914_191422

theorem circle_area_from_circumference : 
  ∀ (r : ℝ), (2 * π * r = 36 * π) → (π * r^2 = 324 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l1914_191422


namespace NUMINAMATH_CALUDE_complex_coordinate_of_reciprocal_i_cubed_l1914_191473

theorem complex_coordinate_of_reciprocal_i_cubed :
  let z : ℂ := (Complex.I ^ 3)⁻¹
  z = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_coordinate_of_reciprocal_i_cubed_l1914_191473


namespace NUMINAMATH_CALUDE_trig_identity_proof_l1914_191480

theorem trig_identity_proof : 
  Real.sin (410 * π / 180) * Real.sin (550 * π / 180) - 
  Real.sin (680 * π / 180) * Real.cos (370 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l1914_191480


namespace NUMINAMATH_CALUDE_quiz_probability_l1914_191415

theorem quiz_probability (n : ℕ) : 
  (1 : ℚ) / 3 * (1 / 2) ^ n = 1 / 12 → n = 2 :=
by sorry

end NUMINAMATH_CALUDE_quiz_probability_l1914_191415


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_l1914_191490

/-- Given points A, B, and C in a 2D plane, prove that if AB is perpendicular to BC,
    then the x-coordinate of C is 8/3. -/
theorem perpendicular_vectors_imply_m (A B C : ℝ × ℝ) :
  A = (-1, 3) →
  B = (2, 1) →
  C.2 = 2 →
  (B.1 - A.1, B.2 - A.2) • (C.1 - B.1, C.2 - B.2) = 0 →
  C.1 = 8/3 := by
  sorry

#check perpendicular_vectors_imply_m

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_l1914_191490


namespace NUMINAMATH_CALUDE_train_meeting_time_l1914_191438

/-- Represents the problem of two trains meeting on a journey from Delhi to Bombay -/
theorem train_meeting_time 
  (speed_first : ℝ) 
  (speed_second : ℝ) 
  (departure_second : ℝ) 
  (meeting_distance : ℝ) 
  (h1 : speed_first = 60) 
  (h2 : speed_second = 80) 
  (h3 : departure_second = 16.5) 
  (h4 : meeting_distance = 480) : 
  ∃ (departure_first : ℝ), 
    speed_first * (departure_second - departure_first) = meeting_distance ∧ 
    departure_first = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_time_l1914_191438


namespace NUMINAMATH_CALUDE_correct_reassembly_probability_l1914_191460

/-- Represents the number of subcubes in each dimension of the larger cube -/
def cubeDimension : ℕ := 3

/-- Represents the total number of subcubes in the larger cube -/
def totalSubcubes : ℕ := cubeDimension ^ 3

/-- Represents the number of corner subcubes -/
def cornerCubes : ℕ := 8

/-- Represents the number of edge subcubes -/
def edgeCubes : ℕ := 12

/-- Represents the number of face subcubes -/
def faceCubes : ℕ := 6

/-- Represents the number of center subcubes -/
def centerCubes : ℕ := 1

/-- Represents the number of possible orientations for a corner subcube -/
def cornerOrientations : ℕ := 3

/-- Represents the number of possible orientations for an edge subcube -/
def edgeOrientations : ℕ := 2

/-- Represents the number of possible orientations for a face subcube -/
def faceOrientations : ℕ := 4

/-- Represents the total number of possible orientations for any subcube -/
def totalOrientations : ℕ := 24

/-- Calculates the number of correct reassemblings -/
def correctReassemblings : ℕ :=
  (cornerOrientations ^ cornerCubes) * (cornerCubes.factorial) *
  (edgeOrientations ^ edgeCubes) * (edgeCubes.factorial) *
  (faceOrientations ^ faceCubes) * (faceCubes.factorial) *
  (centerCubes.factorial)

/-- Calculates the total number of possible reassemblings -/
def totalReassemblings : ℕ :=
  (totalOrientations ^ totalSubcubes) * (totalSubcubes.factorial)

/-- Theorem: The probability of correctly reassembling the cube is equal to
    the ratio of correct reassemblings to total possible reassemblings -/
theorem correct_reassembly_probability :
  (correctReassemblings : ℚ) / totalReassemblings =
  (correctReassemblings : ℚ) / totalReassemblings :=
by
  sorry

end NUMINAMATH_CALUDE_correct_reassembly_probability_l1914_191460


namespace NUMINAMATH_CALUDE_card_drawing_probability_l1914_191431

/-- Represents a standard 52-card deck --/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in each suit --/
def CardsPerSuit : ℕ := 13

/-- Represents the number of suits in a standard deck --/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards drawn --/
def CardsDrawn : ℕ := 8

/-- The probability of the specified event occurring --/
def probability_of_event : ℚ := 3 / 16384

theorem card_drawing_probability :
  (1 : ℚ) / NumberOfSuits *     -- Probability of first card being any suit
  (3 : ℚ) / NumberOfSuits *     -- Probability of second card being a different suit
  (2 : ℚ) / NumberOfSuits *     -- Probability of third card being a different suit
  (1 : ℚ) / NumberOfSuits *     -- Probability of fourth card being the remaining suit
  ((1 : ℚ) / NumberOfSuits)^4   -- Probability of next four cards matching the suit sequence
  = probability_of_event := by sorry

#check card_drawing_probability

end NUMINAMATH_CALUDE_card_drawing_probability_l1914_191431


namespace NUMINAMATH_CALUDE_inequality_solution_l1914_191451

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4*x - 5) / (x^2 + 3*x + 2) < 0 ↔ x ∈ Set.Ioo (-2 : ℝ) (-1) ∪ Set.Ioo (-1 : ℝ) 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1914_191451


namespace NUMINAMATH_CALUDE_cleaner_used_is_80_l1914_191402

/-- Represents the flow rate of cleaner through a pipe at different time intervals -/
structure FlowRate :=
  (initial : ℝ)
  (after15min : ℝ)
  (after25min : ℝ)

/-- Calculates the total amount of cleaner used over a 30-minute period -/
def totalCleanerUsed (flow : FlowRate) : ℝ :=
  flow.initial * 15 + flow.after15min * 10 + flow.after25min * 5

/-- The flow rates given in the problem -/
def problemFlow : FlowRate :=
  { initial := 2
  , after15min := 3
  , after25min := 4 }

/-- Theorem stating that the total cleaner used is 80 ounces -/
theorem cleaner_used_is_80 : totalCleanerUsed problemFlow = 80 := by
  sorry

end NUMINAMATH_CALUDE_cleaner_used_is_80_l1914_191402


namespace NUMINAMATH_CALUDE_lemon_sequences_l1914_191499

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of times the class meets in a week -/
def meetings_per_week : ℕ := 5

/-- The number of possible sequences of lemon recipients in a week -/
def num_sequences : ℕ := num_students ^ meetings_per_week

/-- Theorem stating the number of possible sequences of lemon recipients -/
theorem lemon_sequences :
  num_sequences = 759375 :=
by sorry

end NUMINAMATH_CALUDE_lemon_sequences_l1914_191499


namespace NUMINAMATH_CALUDE_uncle_bob_parking_probability_l1914_191470

def total_spaces : ℕ := 20
def parked_cars : ℕ := 14
def required_spaces : ℕ := 3

theorem uncle_bob_parking_probability :
  let total_configurations := Nat.choose total_spaces parked_cars
  let unfavorable_configurations := Nat.choose (parked_cars - required_spaces + 2) (parked_cars - required_spaces + 2 - parked_cars)
  (total_configurations - unfavorable_configurations) / total_configurations = 19275 / 19380 := by
  sorry

end NUMINAMATH_CALUDE_uncle_bob_parking_probability_l1914_191470


namespace NUMINAMATH_CALUDE_sixty_percent_of_three_fifths_of_hundred_l1914_191437

theorem sixty_percent_of_three_fifths_of_hundred (n : ℝ) : n = 100 → (0.6 * (3/5 * n)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_sixty_percent_of_three_fifths_of_hundred_l1914_191437


namespace NUMINAMATH_CALUDE_angle_D_measure_l1914_191474

-- Define the hexagon and its properties
def ConvexHexagon (A B C D E F : ℝ) : Prop :=
  -- Angles A, B, and C are congruent
  A = B ∧ B = C
  -- Angles D, E, and F are congruent
  ∧ D = E ∧ E = F
  -- The measure of angle A is 50° less than the measure of angle D
  ∧ A + 50 = D
  -- Sum of interior angles of a hexagon is 720°
  ∧ A + B + C + D + E + F = 720

-- Theorem statement
theorem angle_D_measure (A B C D E F : ℝ) 
  (h : ConvexHexagon A B C D E F) : D = 145 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l1914_191474


namespace NUMINAMATH_CALUDE_die_toss_results_l1914_191484

/-- The number of faces on a fair die -/
def numFaces : ℕ := 6

/-- The number of tosses when the process stops -/
def numTosses : ℕ := 5

/-- The number of different numbers recorded when the process stops -/
def numDifferent : ℕ := 3

/-- The total number of different recording results -/
def totalResults : ℕ := 840

/-- Theorem stating the total number of different recording results -/
theorem die_toss_results :
  (numFaces = 6) →
  (numTosses = 5) →
  (numDifferent = 3) →
  (totalResults = 840) := by
  sorry

end NUMINAMATH_CALUDE_die_toss_results_l1914_191484


namespace NUMINAMATH_CALUDE_barbara_paper_count_l1914_191489

/-- The number of sheets in a bundle of colored paper -/
def sheets_per_bundle : ℕ := 2

/-- The number of sheets in a bunch of white paper -/
def sheets_per_bunch : ℕ := 4

/-- The number of sheets in a heap of scrap paper -/
def sheets_per_heap : ℕ := 20

/-- The number of bundles of colored paper -/
def colored_bundles : ℕ := 3

/-- The number of bunches of white paper -/
def white_bunches : ℕ := 2

/-- The number of heaps of scrap paper -/
def scrap_heaps : ℕ := 5

/-- The total number of sheets Barbara removed from the chest of drawers -/
def total_sheets : ℕ := colored_bundles * sheets_per_bundle + 
                         white_bunches * sheets_per_bunch + 
                         scrap_heaps * sheets_per_heap

theorem barbara_paper_count : total_sheets = 114 := by
  sorry

end NUMINAMATH_CALUDE_barbara_paper_count_l1914_191489


namespace NUMINAMATH_CALUDE_wafting_pie_egg_usage_l1914_191483

/-- The Wafting Pie Company's egg usage problem -/
theorem wafting_pie_egg_usage 
  (total_eggs : ℕ) 
  (morning_eggs : ℕ) 
  (h1 : total_eggs = 1339)
  (h2 : morning_eggs = 816) :
  total_eggs - morning_eggs = 523 := by
  sorry

end NUMINAMATH_CALUDE_wafting_pie_egg_usage_l1914_191483


namespace NUMINAMATH_CALUDE_n_has_9_digits_l1914_191423

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect cube -/
axiom n_sq_cube : ∃ k : ℕ, n^2 = k^3

/-- n^3 is a perfect square -/
axiom n_cube_square : ∃ k : ℕ, n^3 = k^2

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ m : ℕ, m < n → ¬(30 ∣ m ∧ (∃ k : ℕ, m^2 = k^3) ∧ (∃ k : ℕ, m^3 = k^2))

/-- Function to count the number of digits in a natural number -/
def count_digits (x : ℕ) : ℕ := sorry

theorem n_has_9_digits : count_digits n = 9 := by sorry

end NUMINAMATH_CALUDE_n_has_9_digits_l1914_191423


namespace NUMINAMATH_CALUDE_power_fraction_equals_two_l1914_191477

theorem power_fraction_equals_two : (2^4 - 2) / (2^3 - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equals_two_l1914_191477


namespace NUMINAMATH_CALUDE_sequence_properties_l1914_191444

theorem sequence_properties (a : Fin 4 → ℝ) 
  (h_decreasing : ∀ i j : Fin 4, i < j → a i > a j)
  (h_nonneg : a 3 ≥ 0)
  (h_closed : ∀ i j : Fin 4, i ≤ j → ∃ k : Fin 4, a i - a j = a k) :
  (∃ d : ℝ, ∀ i : Fin 4, i.val < 3 → a i.succ = a i - d) ∧ 
  (∃ i j : Fin 4, i < j ∧ (i.val + 1) * a i = (j.val + 1) * a j) ∧
  (∃ i : Fin 4, a i = 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1914_191444


namespace NUMINAMATH_CALUDE_toms_remaining_candy_l1914_191446

theorem toms_remaining_candy (initial_boxes : ℕ) (boxes_given_away : ℕ) (pieces_per_box : ℕ) : 
  initial_boxes = 14 → boxes_given_away = 8 → pieces_per_box = 3 →
  (initial_boxes - boxes_given_away) * pieces_per_box = 18 := by
  sorry

end NUMINAMATH_CALUDE_toms_remaining_candy_l1914_191446


namespace NUMINAMATH_CALUDE_ratio_difference_increases_dependence_l1914_191492

/-- Represents a 2x2 contingency table -/
structure ContingencyTable where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the chi-square statistic for a 2x2 contingency table -/
def chi_square (table : ContingencyTable) : ℝ :=
  sorry

/-- Represents the probability of dependence between two variables -/
def dependence_probability (chi_square_value : ℝ) : ℝ :=
  sorry

/-- Theorem: As the difference between ratios increases, the probability of dependence increases -/
theorem ratio_difference_increases_dependence (table : ContingencyTable) :
  let ratio1 := table.a / (table.a + table.b)
  let ratio2 := table.c / (table.c + table.d)
  let diff := |ratio1 - ratio2|
  ∀ ε > 0, ∃ δ > 0,
    ∀ table' : ContingencyTable,
      let ratio1' := table'.a / (table'.a + table'.b)
      let ratio2' := table'.c / (table'.c + table'.d)
      let diff' := |ratio1' - ratio2'|
      diff' > diff + δ →
        dependence_probability (chi_square table') > dependence_probability (chi_square table) + ε :=
by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_increases_dependence_l1914_191492


namespace NUMINAMATH_CALUDE_arccos_zero_equals_pi_half_l1914_191450

theorem arccos_zero_equals_pi_half : Real.arccos 0 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_zero_equals_pi_half_l1914_191450


namespace NUMINAMATH_CALUDE_tangent_point_bounds_l1914_191409

/-- A point (a,b) through which two distinct tangent lines can be drawn to the curve y = e^x -/
structure TangentPoint where
  a : ℝ
  b : ℝ
  two_tangents : ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    b = Real.exp t₁ * (a - t₁ + 1) ∧
    b = Real.exp t₂ * (a - t₂ + 1)

/-- If two distinct tangent lines to y = e^x can be drawn through (a,b), then 0 < b < e^a -/
theorem tangent_point_bounds (p : TangentPoint) : 0 < p.b ∧ p.b < Real.exp p.a := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_bounds_l1914_191409


namespace NUMINAMATH_CALUDE_remainder_of_m_l1914_191435

theorem remainder_of_m (m : ℕ) (h1 : m^2 % 7 = 1) (h2 : m^3 % 7 = 6) : m % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_m_l1914_191435


namespace NUMINAMATH_CALUDE_toothpick_pattern_200th_stage_l1914_191429

/-- 
Given an arithmetic sequence where:
- a is the first term
- d is the common difference
- n is the term number
This function calculates the nth term of the sequence.
-/
def arithmeticSequenceTerm (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

/--
Theorem: In an arithmetic sequence where the first term is 6 and the common difference is 5,
the 200th term is equal to 1001.
-/
theorem toothpick_pattern_200th_stage :
  arithmeticSequenceTerm 6 5 200 = 1001 := by
  sorry

#eval arithmeticSequenceTerm 6 5 200

end NUMINAMATH_CALUDE_toothpick_pattern_200th_stage_l1914_191429


namespace NUMINAMATH_CALUDE_log_identity_l1914_191413

theorem log_identity : (Real.log 2 / Real.log 10) ^ 2 + (Real.log 5 / Real.log 10) ^ 2 + 2 * (Real.log 2 / Real.log 10) * (Real.log 5 / Real.log 10) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l1914_191413


namespace NUMINAMATH_CALUDE_lcm_12_18_l1914_191424

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l1914_191424


namespace NUMINAMATH_CALUDE_complex_power_sum_l1914_191457

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (Real.pi / 4)) :
  z^12 + (1 / z)^12 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1914_191457


namespace NUMINAMATH_CALUDE_third_subtraction_difference_1230_411_l1914_191426

/-- The difference obtained from the third subtraction when using the method of successive subtraction to find the GCD of 1230 and 411 -/
def third_subtraction_difference (a b : ℕ) : ℕ :=
  let d₁ := a - b
  let d₂ := d₁ - b
  d₂ - b

theorem third_subtraction_difference_1230_411 :
  third_subtraction_difference 1230 411 = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_subtraction_difference_1230_411_l1914_191426


namespace NUMINAMATH_CALUDE_two_person_subcommittees_with_male_l1914_191485

theorem two_person_subcommittees_with_male (total : Nat) (men : Nat) (women : Nat) :
  total = 8 →
  men = 5 →
  women = 3 →
  Nat.choose total 2 - Nat.choose women 2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_two_person_subcommittees_with_male_l1914_191485


namespace NUMINAMATH_CALUDE_song_circle_l1914_191491

theorem song_circle (S : Finset Nat) (covers : Finset Nat → Finset Nat)
  (h_card : S.card = 12)
  (h_cover_10 : ∀ T ⊆ S, T.card = 10 → (covers T).card = 20)
  (h_cover_8 : ∀ T ⊆ S, T.card = 8 → (covers T).card = 16) :
  (covers S).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_song_circle_l1914_191491


namespace NUMINAMATH_CALUDE_apartment_occupancy_l1914_191476

theorem apartment_occupancy (total_floors : ℕ) (apartments_per_floor : ℕ) (total_people : ℕ) : 
  total_floors = 12 →
  apartments_per_floor = 10 →
  total_people = 360 →
  ∃ (people_per_apartment : ℕ), 
    people_per_apartment * (apartments_per_floor * total_floors / 2 + apartments_per_floor * total_floors / 4) = total_people ∧
    people_per_apartment = 4 :=
by sorry

end NUMINAMATH_CALUDE_apartment_occupancy_l1914_191476


namespace NUMINAMATH_CALUDE_final_savings_calculation_l1914_191454

/-- Calculate final savings after a period of time given initial savings, monthly income, and monthly expenses. -/
def calculate_final_savings (initial_savings : ℕ) (monthly_income : ℕ) (monthly_expenses : ℕ) (months : ℕ) : ℕ :=
  initial_savings + months * monthly_income - months * monthly_expenses

/-- Theorem: Given the specific financial conditions, the final savings will be 1106900 rubles. -/
theorem final_savings_calculation :
  let initial_savings : ℕ := 849400
  let monthly_income : ℕ := 45000 + 35000 + 7000 + 10000 + 13000
  let monthly_expenses : ℕ := 30000 + 10000 + 5000 + 4500 + 9000
  let months : ℕ := 5
  calculate_final_savings initial_savings monthly_income monthly_expenses months = 1106900 := by
  sorry

end NUMINAMATH_CALUDE_final_savings_calculation_l1914_191454


namespace NUMINAMATH_CALUDE_student_boat_problem_l1914_191455

theorem student_boat_problem (students boats : ℕ) : 
  (7 * boats + 5 = students) → 
  (8 * boats = students + 2) → 
  (students = 54 ∧ boats = 7) :=
by sorry

end NUMINAMATH_CALUDE_student_boat_problem_l1914_191455


namespace NUMINAMATH_CALUDE_value_of_a_l1914_191452

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 8
def g (x : ℝ) : ℝ := x^2 - 4

-- State the theorem
theorem value_of_a (a : ℝ) (ha : a > 0) (h : f (g a) = 8) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1914_191452


namespace NUMINAMATH_CALUDE_M_union_N_eq_M_l1914_191481

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | |p.1 * p.2| = 1 ∧ p.1 > 0}
def N : Set (ℝ × ℝ) := {p | Real.arctan p.1 + Real.arctan p.2 = Real.pi}

-- Theorem statement
theorem M_union_N_eq_M : M ∪ N = M := by
  sorry

end NUMINAMATH_CALUDE_M_union_N_eq_M_l1914_191481


namespace NUMINAMATH_CALUDE_flag_width_calculation_l1914_191488

theorem flag_width_calculation (height : ℝ) (paint_cost : ℝ) (paint_coverage : ℝ) 
  (total_spent : ℝ) (h1 : height = 4) (h2 : paint_cost = 2) (h3 : paint_coverage = 4) 
  (h4 : total_spent = 20) : ∃ (width : ℝ), width = 5 := by
  sorry

end NUMINAMATH_CALUDE_flag_width_calculation_l1914_191488


namespace NUMINAMATH_CALUDE_polynomial_sum_zero_l1914_191498

theorem polynomial_sum_zero (a b c d : ℝ) :
  (∀ x : ℝ, (1 + x)^2 * (1 - x) = a + b*x + c*x^2 + d*x^3) →
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_zero_l1914_191498


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1914_191495

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 = 10*x - 14) → (∃ y : ℝ, y^2 = 10*y - 14 ∧ x + y = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1914_191495


namespace NUMINAMATH_CALUDE_preimage_of_three_one_l1914_191436

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- Theorem stating that (2, 1) is the preimage of (3, 1) under f -/
theorem preimage_of_three_one :
  ∃! p : ℝ × ℝ, f p = (3, 1) ∧ p = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_three_one_l1914_191436


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1914_191414

-- Define the lines l₁ and l₂
def l₁ (x y a : ℝ) : Prop := 3 * x + 2 * a * y - 5 = 0
def l₂ (x y a : ℝ) : Prop := (3 * a - 1) * x - a * y - 2 = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop := ∀ x y, l₁ x y a ↔ l₂ x y a

-- Theorem statement
theorem parallel_lines_a_value (a : ℝ) :
  parallel a → (a = 0 ∨ a = -1/6) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1914_191414


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1914_191403

/-- 
Given that x·(4x + 3) < d if and only when x ∈ (-5/2, 1), prove that d = 10
-/
theorem quadratic_inequality_solution (d : ℝ) : 
  (∀ x : ℝ, x * (4 * x + 3) < d ↔ -5/2 < x ∧ x < 1) → d = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1914_191403


namespace NUMINAMATH_CALUDE_problem_solution_l1914_191439

def vector := ℝ × ℝ

noncomputable def problem (x : ℝ) : Prop :=
  let a : vector := (1, Real.sin x)
  let b : vector := (Real.sin x, -1)
  let c : vector := (1, Real.cos x)
  0 < x ∧ x < Real.pi ∧
  ¬ (∃ (k : ℝ), (1 + Real.sin x, Real.sin x - 1) = (k * c.1, k * c.2)) ∧
  x = Real.pi / 2 ∧
  ∃ (A B C : ℝ), 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
    A + B + C = Real.pi ∧
    B = Real.pi / 2 ∧
    2 * (Real.sin B)^2 + 2 * (Real.sin C)^2 - 2 * (Real.sin A)^2 = Real.sin B * Real.sin C

theorem problem_solution (x : ℝ) (h : problem x) :
  ∃ (A B C : ℝ), Real.sin (C - Real.pi / 3) = (1 - 3 * Real.sqrt 5) / 8 := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_problem_solution_l1914_191439


namespace NUMINAMATH_CALUDE_jack_closet_capacity_l1914_191465

/-- Represents the storage capacity of a closet -/
structure ClosetCapacity where
  cansPerRow : ℕ
  rowsPerShelf : ℕ
  shelvesPerCloset : ℕ

/-- Calculates the total number of cans that can be stored in a closet -/
def totalCansPerCloset (c : ClosetCapacity) : ℕ :=
  c.cansPerRow * c.rowsPerShelf * c.shelvesPerCloset

/-- Theorem: Given Jack's closet configuration, he can store 480 cans in each closet -/
theorem jack_closet_capacity :
  let jackCloset : ClosetCapacity := {
    cansPerRow := 12,
    rowsPerShelf := 4,
    shelvesPerCloset := 10
  }
  totalCansPerCloset jackCloset = 480 := by
  sorry


end NUMINAMATH_CALUDE_jack_closet_capacity_l1914_191465


namespace NUMINAMATH_CALUDE_expand_expression_l1914_191459

theorem expand_expression (x : ℝ) : (x - 1) * (4 * x + 5) = 4 * x^2 + x - 5 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1914_191459


namespace NUMINAMATH_CALUDE_november_savings_l1914_191445

def september_savings : ℕ := 50
def october_savings : ℕ := 37
def mom_gift : ℕ := 25
def video_game_cost : ℕ := 87
def money_left : ℕ := 36

theorem november_savings :
  ∃ (november_savings : ℕ),
    september_savings + october_savings + november_savings + mom_gift - video_game_cost = money_left ∧
    november_savings = 11 :=
sorry

end NUMINAMATH_CALUDE_november_savings_l1914_191445


namespace NUMINAMATH_CALUDE_f_value_at_2012_l1914_191462

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h1 : ∀ x : ℝ, f (x + 3) ≤ f x + 3)
variable (h2 : ∀ x : ℝ, f (x + 2) ≥ f x + 2)
variable (h3 : f 998 = 1002)

-- State the theorem
theorem f_value_at_2012 : f 2012 = 2016 := by sorry

end NUMINAMATH_CALUDE_f_value_at_2012_l1914_191462


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l1914_191427

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2)
  f (-2) = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l1914_191427


namespace NUMINAMATH_CALUDE_sarahs_age_l1914_191401

theorem sarahs_age (ana billy mark sarah : ℕ) : 
  ana + 3 = 15 →
  billy = ana / 2 →
  mark = billy + 4 →
  sarah = 3 * mark - 4 →
  sarah = 26 := by
sorry

end NUMINAMATH_CALUDE_sarahs_age_l1914_191401


namespace NUMINAMATH_CALUDE_range_of_m_l1914_191434

-- Define the sets A and B
def A : Set ℝ := {x | (2 - x) / (2 * x - 1) > 1}
def B (m : ℝ) : Set ℝ := {x | x^2 + 2*x + 1 - m ≤ 0}

-- State the theorem
theorem range_of_m (h : ∀ m > 0, A ⊆ B m ∧ ∃ x, x ∈ B m ∧ x ∉ A) :
  {m : ℝ | m ≥ 4} = {m : ℝ | m > 0 ∧ A ⊆ B m ∧ ∃ x, x ∈ B m ∧ x ∉ A} :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1914_191434


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1914_191479

/-- Proves that for any real number m, the line (2m+1)x + (m+1)y - 7m - 4 = 0 passes through the point (3, 1) -/
theorem fixed_point_on_line (m : ℝ) : (2 * m + 1) * 3 + (m + 1) * 1 - 7 * m - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1914_191479


namespace NUMINAMATH_CALUDE_cosine_sum_l1914_191416

theorem cosine_sum (α β : Real) : 
  0 < α ∧ α < Real.pi/2 ∧
  -Real.pi/2 < β ∧ β < 0 ∧
  Real.cos (Real.pi/4 + α) = 1/3 ∧
  Real.cos (Real.pi/4 - β/2) = Real.sqrt 3/3 →
  Real.cos (α + β/2) = 5 * Real.sqrt 3/9 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_l1914_191416


namespace NUMINAMATH_CALUDE_fly_path_shortest_distance_l1914_191467

/-- Represents a right circular cone. -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone. -/
structure SurfacePoint where
  distanceFromVertex : ℝ

/-- Calculates the shortest distance between two points on the surface of a cone. -/
def shortestSurfaceDistance (c : Cone) (p1 p2 : SurfacePoint) : ℝ :=
  sorry

theorem fly_path_shortest_distance :
  let c : Cone := { baseRadius := 600, height := 200 * Real.sqrt 7 }
  let p1 : SurfacePoint := { distanceFromVertex := 125 }
  let p2 : SurfacePoint := { distanceFromVertex := 375 * Real.sqrt 2 }
  shortestSurfaceDistance c p1 p2 = 625 := by sorry

end NUMINAMATH_CALUDE_fly_path_shortest_distance_l1914_191467


namespace NUMINAMATH_CALUDE_change_is_five_l1914_191419

/-- Given a meal cost, drink cost, tip percentage, and payment amount, 
    calculate the change received. -/
def calculate_change (meal_cost drink_cost tip_percentage payment : ℚ) : ℚ :=
  let total_before_tip := meal_cost + drink_cost
  let tip_amount := total_before_tip * (tip_percentage / 100)
  let total_with_tip := total_before_tip + tip_amount
  payment - total_with_tip

/-- Theorem stating that given the specified costs and payment, 
    the change received is $5. -/
theorem change_is_five :
  calculate_change 10 2.5 20 20 = 5 := by
  sorry

end NUMINAMATH_CALUDE_change_is_five_l1914_191419


namespace NUMINAMATH_CALUDE_grant_school_students_l1914_191497

theorem grant_school_students (S : ℕ) : 
  (S / 3 : ℚ) / 4 = 15 → S = 180 := by
  sorry

end NUMINAMATH_CALUDE_grant_school_students_l1914_191497


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1914_191407

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 2 = 5 → a 6 = 33 → a 3 + a 5 = 38 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1914_191407


namespace NUMINAMATH_CALUDE_spiders_can_catch_fly_l1914_191408

-- Define the cube
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 12)

-- Define the creatures
inductive Creature
| Spider
| Fly

-- Define the position of a creature on the cube
structure Position where
  creature : Creature
  vertex : Fin 8

-- Define the speed of creatures
def speed (c : Creature) : ℕ :=
  match c with
  | Creature.Spider => 1
  | Creature.Fly => 3

-- Define the initial state
def initial_state (cube : Cube) : Finset Position :=
  sorry

-- Define the catching condition
def can_catch (cube : Cube) (positions : Finset Position) : Prop :=
  sorry

-- The main theorem
theorem spiders_can_catch_fly (cube : Cube) :
  ∃ (final_positions : Finset Position),
    can_catch cube final_positions :=
  sorry

end NUMINAMATH_CALUDE_spiders_can_catch_fly_l1914_191408


namespace NUMINAMATH_CALUDE_complete_square_transform_l1914_191430

theorem complete_square_transform (a : ℝ) : a^2 + 4*a - 5 = (a + 2)^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_transform_l1914_191430


namespace NUMINAMATH_CALUDE_f_3_range_l1914_191428

/-- Given a quadratic function f(x) = ax^2 - c satisfying certain conditions,
    prove that f(3) is within a specific range. -/
theorem f_3_range (a c : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x^2 - c)
    (h_1 : -4 ≤ f 1 ∧ f 1 ≤ -1)
    (h_2 : -1 ≤ f 2 ∧ f 2 ≤ 5) :
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_f_3_range_l1914_191428


namespace NUMINAMATH_CALUDE_total_balloons_l1914_191447

/-- Represents the number of balloons of each color -/
structure BalloonCounts where
  gold : ℕ
  silver : ℕ
  black : ℕ
  blue : ℕ
  red : ℕ

/-- The conditions of the balloon problem -/
def balloon_problem (b : BalloonCounts) : Prop :=
  b.gold = 141 ∧
  b.silver = 2 * b.gold ∧
  b.black = 150 ∧
  b.blue = b.silver / 2 ∧
  b.red = b.blue / 3

/-- The theorem stating the total number of balloons -/
theorem total_balloons (b : BalloonCounts) 
  (h : balloon_problem b) : 
  b.gold + b.silver + b.black + b.blue + b.red = 761 := by
  sorry

#check total_balloons

end NUMINAMATH_CALUDE_total_balloons_l1914_191447


namespace NUMINAMATH_CALUDE_male_kittens_count_l1914_191475

/-- Given an initial number of cats, number of female kittens, and total number of cats after kittens are born, 
    calculate the number of male kittens. -/
def male_kittens (initial_cats female_kittens total_cats : ℕ) : ℕ :=
  total_cats - initial_cats - female_kittens

/-- Theorem stating that given the problem conditions, the number of male kittens is 2. -/
theorem male_kittens_count : male_kittens 2 3 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_male_kittens_count_l1914_191475


namespace NUMINAMATH_CALUDE_rope_length_comparison_l1914_191432

theorem rope_length_comparison (L : ℝ) (h : L > 0) : 
  ¬ (∀ L, L - (1/3) = L - (L/3)) :=
sorry

end NUMINAMATH_CALUDE_rope_length_comparison_l1914_191432


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l1914_191469

/-- Given a function f(x) = x^2 - ax - a with maximum value 1 on [0, 2], prove a = 1 -/
theorem max_value_implies_a_equals_one (a : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 - a*x - a) ∧
   (∀ x ∈ Set.Icc 0 2, f x ≤ 1) ∧
   (∃ x ∈ Set.Icc 0 2, f x = 1)) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l1914_191469


namespace NUMINAMATH_CALUDE_function_inequality_implies_lower_bound_on_a_l1914_191442

open Real

theorem function_inequality_implies_lower_bound_on_a :
  ∀ a : ℝ,
  (∀ x : ℝ, x > 0 → (log x - a ≤ x * exp x - x)) →
  a ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_lower_bound_on_a_l1914_191442


namespace NUMINAMATH_CALUDE_circle_condition_tangent_circles_intersecting_circle_line_l1914_191478

-- Define the equation C
def equation_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the given circle equation
def given_circle (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 12*y + 36 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem 1: For equation C to represent a circle, m < 5
theorem circle_condition (m : ℝ) : 
  (∃ x y, equation_C x y m) → m < 5 :=
sorry

-- Theorem 2: When circle C is tangent to the given circle, m = 4
theorem tangent_circles (m : ℝ) :
  (∃ x y, equation_C x y m ∧ given_circle x y) → m = 4 :=
sorry

-- Theorem 3: When circle C intersects line l at points M and N with |MN| = 4√5/5, m = 4
theorem intersecting_circle_line (m : ℝ) :
  (∃ x1 y1 x2 y2, 
    equation_C x1 y1 m ∧ equation_C x2 y2 m ∧
    line_l x1 y1 ∧ line_l x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = (4*Real.sqrt 5/5)^2) → 
  m = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_tangent_circles_intersecting_circle_line_l1914_191478


namespace NUMINAMATH_CALUDE_allocation_schemes_count_l1914_191406

/-- The number of ways to distribute n volunteers into k groups with size constraints -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to assign k groups to k different areas -/
def assign (k : ℕ) : ℕ := sorry

/-- The total number of allocation schemes -/
def total_schemes (n : ℕ) (k : ℕ) : ℕ :=
  distribute n k * assign k

theorem allocation_schemes_count :
  total_schemes 6 4 = 1080 := by sorry

end NUMINAMATH_CALUDE_allocation_schemes_count_l1914_191406


namespace NUMINAMATH_CALUDE_shooting_stars_count_difference_l1914_191405

theorem shooting_stars_count_difference (bridget_count reginald_count sam_count : ℕ) : 
  bridget_count = 14 →
  reginald_count = bridget_count - 2 →
  sam_count > reginald_count →
  sam_count = (bridget_count + reginald_count + sam_count) / 3 + 2 →
  sam_count - reginald_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_shooting_stars_count_difference_l1914_191405


namespace NUMINAMATH_CALUDE_puppies_given_away_l1914_191466

/-- Given that Sandy initially had some puppies and now has fewer,
    prove that the number of puppies given away is the difference
    between the initial and current number of puppies. -/
theorem puppies_given_away
  (initial_puppies : ℕ)
  (current_puppies : ℕ)
  (h : current_puppies ≤ initial_puppies) :
  initial_puppies - current_puppies =
  initial_puppies - current_puppies :=
by sorry

end NUMINAMATH_CALUDE_puppies_given_away_l1914_191466


namespace NUMINAMATH_CALUDE_power_of_two_in_factorial_eight_l1914_191421

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem power_of_two_in_factorial_eight :
  ∀ i k m p : ℕ,
  i > 0 → k > 0 → m > 0 → p > 0 →
  factorial 8 = 2^i * 3^k * 5^m * 7^p →
  i + k + m + p = 11 →
  i = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_power_of_two_in_factorial_eight_l1914_191421


namespace NUMINAMATH_CALUDE_exam_students_count_l1914_191472

theorem exam_students_count : 
  ∀ N : ℕ,
  (N : ℝ) * 80 = 160 + (N - 8 : ℝ) * 90 →
  N = 56 :=
by
  sorry

#check exam_students_count

end NUMINAMATH_CALUDE_exam_students_count_l1914_191472


namespace NUMINAMATH_CALUDE_only_prop2_and_prop4_true_l1914_191410

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations between lines and planes
def parallel (a b : Plane) : Prop := sorry
def perpendicular (a b : Plane) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def line_parallel (a b : Line) : Prop := sorry
def line_perpendicular (a b : Line) : Prop := sorry
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry
def line_perpendicular_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the propositions
def proposition1 (m n : Line) (α β : Plane) : Prop :=
  parallel α β ∧ contained_in m β ∧ contained_in n α → line_parallel m n

def proposition2 (m n : Line) (α β : Plane) : Prop :=
  parallel α β ∧ line_perpendicular_plane m β ∧ line_parallel_plane n α → line_perpendicular m n

def proposition3 (m n : Line) (α β : Plane) : Prop :=
  perpendicular α β ∧ line_perpendicular_plane m α ∧ line_parallel_plane n β → line_parallel m n

def proposition4 (m n : Line) (α β : Plane) : Prop :=
  perpendicular α β ∧ line_perpendicular_plane m α ∧ line_perpendicular_plane n β → line_perpendicular m n

-- Theorem stating that only propositions 2 and 4 are true
theorem only_prop2_and_prop4_true (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) (h_diff_planes : α ≠ β) : 
  (¬ proposition1 m n α β) ∧ 
  proposition2 m n α β ∧ 
  (¬ proposition3 m n α β) ∧ 
  proposition4 m n α β := by
  sorry

end NUMINAMATH_CALUDE_only_prop2_and_prop4_true_l1914_191410


namespace NUMINAMATH_CALUDE_school_pet_ownership_l1914_191456

theorem school_pet_ownership (total_students : ℕ) (cat_owners : ℕ) (rabbit_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 80)
  (h3 : rabbit_owners = 120) :
  (cat_owners : ℚ) / total_students * 100 = 16 ∧
  (rabbit_owners : ℚ) / total_students * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_school_pet_ownership_l1914_191456


namespace NUMINAMATH_CALUDE_multiple_in_difference_l1914_191417

theorem multiple_in_difference (n m : ℤ) (h1 : n = -7) (h2 : 3 * n = m * n - 7) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_in_difference_l1914_191417


namespace NUMINAMATH_CALUDE_dragons_games_count_l1914_191400

theorem dragons_games_count :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (0.4 : ℝ) * initial_games →
    (initial_wins + 5 : ℝ) / (initial_games + 8) = 0.55 →
    initial_games + 8 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_dragons_games_count_l1914_191400


namespace NUMINAMATH_CALUDE_simplify_expression_l1914_191404

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1914_191404


namespace NUMINAMATH_CALUDE_fraction_simplification_l1914_191464

theorem fraction_simplification :
  (1/2 - 1/3 + 1/5) / (1/3 - 1/2 + 1/7) = -77/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1914_191464


namespace NUMINAMATH_CALUDE_fair_lines_theorem_l1914_191420

/-- Represents the number of people in the bumper cars line -/
def bumper_cars_line (initial : ℕ) (left : ℕ) (joined : ℕ) : ℕ :=
  initial - left + joined

/-- Represents the total number of people in both lines -/
def total_people (bumper_cars : ℕ) (roller_coaster : ℕ) : ℕ :=
  bumper_cars + roller_coaster

theorem fair_lines_theorem (x y Z : ℕ) (h1 : Z = bumper_cars_line 25 x y) 
  (h2 : Z ≥ x) : total_people Z 15 = 40 - x + y := by
  sorry

#check fair_lines_theorem

end NUMINAMATH_CALUDE_fair_lines_theorem_l1914_191420


namespace NUMINAMATH_CALUDE_tank_fill_time_l1914_191493

/-- Represents the time (in minutes) it takes for a pipe to fill or empty the tank -/
structure PipeRate where
  rate : ℚ
  filling : Bool

/-- Represents the state of the tank -/
structure TankState where
  filled : ℚ  -- Fraction of the tank that is filled

/-- Represents the system of pipes and the tank -/
structure PipeSystem where
  pipes : Fin 4 → PipeRate
  cycle_time : ℚ
  cycle_effect : ℚ

def apply_pipe (p : PipeRate) (t : TankState) (duration : ℚ) : TankState :=
  if p.filling then
    { filled := t.filled + duration / p.rate }
  else
    { filled := t.filled - duration / p.rate }

def apply_cycle (s : PipeSystem) (t : TankState) : TankState :=
  { filled := t.filled + s.cycle_effect }

def time_to_fill (s : PipeSystem) : ℚ :=
  s.cycle_time * (1 / s.cycle_effect)

theorem tank_fill_time (s : PipeSystem) (h1 : s.pipes 0 = ⟨20, true⟩)
    (h2 : s.pipes 1 = ⟨30, true⟩) (h3 : s.pipes 2 = ⟨15, false⟩)
    (h4 : s.pipes 3 = ⟨40, true⟩) (h5 : s.cycle_time = 16)
    (h6 : s.cycle_effect = 1/10) : time_to_fill s = 160 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l1914_191493


namespace NUMINAMATH_CALUDE_total_throw_distance_l1914_191448

/-- Proves the total distance thrown over two days is 1600 yards. -/
theorem total_throw_distance (T : ℝ) : 
  let throw_distance_T := 20
  let throw_distance_80 := 2 * throw_distance_T
  let saturday_throws := 20
  let sunday_throws := 30
  let saturday_distance := saturday_throws * throw_distance_T
  let sunday_distance := sunday_throws * throw_distance_80
  saturday_distance + sunday_distance = 1600 := by sorry

end NUMINAMATH_CALUDE_total_throw_distance_l1914_191448


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1914_191494

theorem arithmetic_geometric_sequence_ratio (d : ℝ) (q : ℚ) (a b : ℕ → ℝ) :
  d ≠ 0 →
  0 < q →
  q < 1 →
  (∀ n, a (n + 1) = a n + d) →
  (∀ n, b (n + 1) = q * b n) →
  a 1 = d →
  b 1 = d^2 →
  ∃ m : ℕ+, (a 1^2 + a 2^2 + a 3^2) / (b 1 + b 2 + b 3) = m →
  q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1914_191494


namespace NUMINAMATH_CALUDE_problem_statement_l1914_191433

theorem problem_statement (x y : ℝ) : 
  16 * (4 : ℝ)^x = 3^(y + 2) → y = -2 → x = -2 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1914_191433


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1914_191487

/-- Given two points M and N that are symmetric with respect to the x-axis,
    prove that the sum of their x and y coordinates is -3. -/
theorem symmetric_points_sum (b a : ℝ) : 
  (∃ (M N : ℝ × ℝ), 
    M = (-2, b) ∧ 
    N = (a, 1) ∧ 
    (M.1 = N.1 ∧ M.2 = -N.2)) → 
  a + b = -3 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1914_191487


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1914_191458

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h1 : a 2 + a 4 = 4) 
    (h2 : a 3 + a 5 = 10) : 
  a 5 + a 7 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1914_191458


namespace NUMINAMATH_CALUDE_cloth_sale_problem_l1914_191486

/-- Prove that the number of meters of cloth sold is 85, given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_problem (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
  (h1 : total_selling_price = 8925)
  (h2 : profit_per_meter = 35)
  (h3 : cost_price_per_meter = 70) :
  (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 85 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_problem_l1914_191486


namespace NUMINAMATH_CALUDE_student_guinea_pig_difference_is_126_l1914_191463

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 24

/-- The number of guinea pigs in each classroom -/
def guinea_pigs_per_classroom : ℕ := 3

/-- The number of classrooms -/
def number_of_classrooms : ℕ := 6

/-- The difference between the total number of students and the total number of guinea pigs -/
def student_guinea_pig_difference : ℕ := 
  (students_per_classroom * number_of_classrooms) - (guinea_pigs_per_classroom * number_of_classrooms)

theorem student_guinea_pig_difference_is_126 : student_guinea_pig_difference = 126 := by
  sorry

end NUMINAMATH_CALUDE_student_guinea_pig_difference_is_126_l1914_191463


namespace NUMINAMATH_CALUDE_textbook_cost_calculation_l1914_191468

theorem textbook_cost_calculation : 
  let sale_price : ℝ := 15 * (1 - 0.2)
  let sale_books : ℝ := 5
  let friend_books_cost : ℝ := 12 + 2 * 15
  let online_books_cost : ℝ := 45 * (1 - 0.1)
  let bookstore_books_cost : ℝ := 3 * 45
  let tax_rate : ℝ := 0.08
  
  sale_price * sale_books + friend_books_cost + online_books_cost + bookstore_books_cost + 
  ((sale_price * sale_books + friend_books_cost + online_books_cost + bookstore_books_cost) * tax_rate) = 299.70 := by
sorry


end NUMINAMATH_CALUDE_textbook_cost_calculation_l1914_191468


namespace NUMINAMATH_CALUDE_factor_tree_value_l1914_191411

-- Define the structure of the factor tree
structure FactorTree :=
  (A B C D E : ℝ)

-- Define the conditions of the factor tree
def valid_factor_tree (t : FactorTree) : Prop :=
  t.A^2 = t.B * t.C ∧
  t.B = 2 * t.D ∧
  t.D = 2 * 4 ∧
  t.C = 7 * t.E ∧
  t.E = 7 * 2

-- Theorem statement
theorem factor_tree_value (t : FactorTree) (h : valid_factor_tree t) : 
  t.A = 28 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_value_l1914_191411


namespace NUMINAMATH_CALUDE_midpoint_sum_midpoint_sum_specific_l1914_191425

/-- Given a line segment with endpoints (3, 4) and (9, 18), 
    the sum of the coordinates of its midpoint is 17. -/
theorem midpoint_sum : ℝ → ℝ → ℝ → ℝ → ℝ := fun x₁ y₁ x₂ y₂ =>
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y

#check midpoint_sum 3 4 9 18 = 17

theorem midpoint_sum_specific : midpoint_sum 3 4 9 18 = 17 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_midpoint_sum_specific_l1914_191425


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_l1914_191418

/-- The number of faces on each cube -/
def faces_per_cube : ℕ := 6

/-- The number of digits (0 to 9) -/
def num_digits : ℕ := 10

/-- The length of the number we need to be able to form -/
def number_length : ℕ := 30

/-- The minimum number of each non-zero digit needed -/
def min_nonzero_digits : ℕ := number_length

/-- The minimum number of zero digits needed -/
def min_zero_digits : ℕ := number_length - 1

/-- The total minimum number of digit instances needed -/
def total_min_digits : ℕ := min_nonzero_digits * (num_digits - 1) + min_zero_digits

/-- The smallest number of cubes needed to form any 30-digit number -/
def min_cubes : ℕ := 50

theorem smallest_number_of_cubes : 
  faces_per_cube * min_cubes ≥ total_min_digits ∧ 
  ∀ n : ℕ, n < min_cubes → faces_per_cube * n < total_min_digits :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_cubes_l1914_191418


namespace NUMINAMATH_CALUDE_prime_square_sum_l1914_191453

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2,2,5), (2,5,2), (3,2,3), (3,3,2)} ∪ {(p,q,r) | p = 2 ∧ q = r ∧ q ≥ 3 ∧ Nat.Prime q}

theorem prime_square_sum (p q r : ℕ) :
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ is_perfect_square (p^q + p^r) ↔ (p,q,r) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_prime_square_sum_l1914_191453


namespace NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l1914_191443

/-- Represents the scoring system and conditions of the AMC 12 problem -/
structure AMC12Scoring where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Int
  incorrect_points : Int
  unanswered_points : Int
  target_score : Int

/-- Calculates the score based on the number of correct answers -/
def calculate_score (s : AMC12Scoring) (correct_answers : Nat) : Int :=
  let incorrect_answers := s.attempted_problems - correct_answers
  let unanswered := s.total_problems - s.attempted_problems
  correct_answers * s.correct_points + 
  incorrect_answers * s.incorrect_points + 
  unanswered * s.unanswered_points

/-- Theorem stating the minimum number of correct answers needed to reach the target score -/
theorem min_correct_answers_for_target_score (s : AMC12Scoring) 
  (h1 : s.total_problems = 30)
  (h2 : s.attempted_problems = 26)
  (h3 : s.correct_points = 7)
  (h4 : s.incorrect_points = -1)
  (h5 : s.unanswered_points = 1)
  (h6 : s.target_score = 150) :
  ∃ n : Nat, (∀ m : Nat, m < n → calculate_score s m < s.target_score) ∧ 
             calculate_score s n ≥ s.target_score ∧
             n = 22 := by
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l1914_191443


namespace NUMINAMATH_CALUDE_tree_growth_theorem_l1914_191461

-- Define growth rates and initial heights
def growth_rate_A : ℚ := 25  -- 50 cm / 2 weeks
def growth_rate_B : ℚ := 70 / 3
def growth_rate_C : ℚ := 90 / 4
def initial_height_A : ℚ := 200
def initial_height_B : ℚ := 150
def initial_height_C : ℚ := 250
def weeks : ℕ := 16

-- Calculate final heights
def final_height_A : ℚ := initial_height_A + growth_rate_A * weeks
def final_height_B : ℚ := initial_height_B + growth_rate_B * weeks
def final_height_C : ℚ := initial_height_C + growth_rate_C * weeks

-- Define the combined final height
def combined_final_height : ℚ := final_height_A + final_height_B + final_height_C

-- Theorem to prove
theorem tree_growth_theorem :
  (combined_final_height : ℚ) = 1733.33 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_theorem_l1914_191461


namespace NUMINAMATH_CALUDE_max_b_value_max_b_value_achieved_l1914_191471

theorem max_b_value (b : ℕ+) (x : ℤ) (h : x^2 + b * x = -20) : b ≤ 21 :=
sorry

theorem max_b_value_achieved : ∃ (b : ℕ+) (x : ℤ), x^2 + b * x = -20 ∧ b = 21 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_max_b_value_achieved_l1914_191471

import Mathlib

namespace NUMINAMATH_CALUDE_bunny_rate_is_three_l3444_344423

/-- The number of times a single bunny comes out of its burrow per minute. -/
def bunny_rate : ℕ := sorry

/-- The number of bunnies. -/
def num_bunnies : ℕ := 20

/-- The number of hours observed. -/
def observation_hours : ℕ := 10

/-- The total number of times bunnies come out in the observation period. -/
def total_exits : ℕ := 36000

/-- Proves that the bunny_rate is 3 given the conditions of the problem. -/
theorem bunny_rate_is_three : bunny_rate = 3 := by
  sorry

end NUMINAMATH_CALUDE_bunny_rate_is_three_l3444_344423


namespace NUMINAMATH_CALUDE_lizzy_scored_67_percent_l3444_344402

/-- Represents the exam scores of four students -/
structure ExamScores where
  max_score : ℕ
  gibi_percent : ℕ
  jigi_percent : ℕ
  mike_percent : ℕ
  average_mark : ℕ

/-- Calculates Lizzy's score as a percentage -/
def lizzy_percent (scores : ExamScores) : ℕ :=
  let total_marks := scores.average_mark * 4
  let others_total := (scores.gibi_percent + scores.jigi_percent + scores.mike_percent) * scores.max_score / 100
  let lizzy_score := total_marks - others_total
  lizzy_score * 100 / scores.max_score

/-- Theorem stating that Lizzy's score is 67% given the conditions -/
theorem lizzy_scored_67_percent (scores : ExamScores)
  (h_max : scores.max_score = 700)
  (h_gibi : scores.gibi_percent = 59)
  (h_jigi : scores.jigi_percent = 55)
  (h_mike : scores.mike_percent = 99)
  (h_avg : scores.average_mark = 490) :
  lizzy_percent scores = 67 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_scored_67_percent_l3444_344402


namespace NUMINAMATH_CALUDE_max_rectangles_is_k_times_l_l3444_344493

/-- A partition of a square into rectangles -/
structure SquarePartition where
  k : ℕ  -- number of rectangles intersected by a vertical line
  l : ℕ  -- number of rectangles intersected by a horizontal line
  no_interior_intersections : Bool  -- no two segments intersect at an interior point
  no_collinear_segments : Bool  -- no two segments lie on the same line

/-- The number of rectangles in a square partition -/
def num_rectangles (p : SquarePartition) : ℕ := sorry

/-- The maximum number of rectangles in any valid square partition -/
def max_rectangles (p : SquarePartition) : ℕ := p.k * p.l

/-- Theorem: The maximum number of rectangles in a valid square partition is k * l -/
theorem max_rectangles_is_k_times_l (p : SquarePartition) 
  (h1 : p.no_interior_intersections = true) 
  (h2 : p.no_collinear_segments = true) : 
  num_rectangles p ≤ max_rectangles p := by sorry

end NUMINAMATH_CALUDE_max_rectangles_is_k_times_l_l3444_344493


namespace NUMINAMATH_CALUDE_angle_complement_measure_l3444_344486

theorem angle_complement_measure : 
  ∀ x : ℝ, 
  (x + (3 * x + 10) = 90) →  -- Condition 1 and 2 combined
  (3 * x + 10 = 70) :=        -- The complement measure to prove
by
  sorry

end NUMINAMATH_CALUDE_angle_complement_measure_l3444_344486


namespace NUMINAMATH_CALUDE_unique_solution_l3444_344495

-- Define the system of linear equations
def equation1 (x y : ℝ) : Prop := x + y = 3
def equation2 (x y : ℝ) : Prop := x - y = 1

-- Theorem statement
theorem unique_solution :
  ∃! (x y : ℝ), equation1 x y ∧ equation2 x y ∧ x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3444_344495


namespace NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l3444_344434

-- Define the solution set for the first inequality
def solution_set_1 (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Define the solution set for the second inequality
def solution_set_2 (x : ℝ) : Prop := x - 1/2 < x ∧ x < 1/3

-- Theorem for the first part
theorem inequality_solution_1 : 
  ∀ x : ℝ, (1/x > 1) ↔ solution_set_1 x := by sorry

-- Theorem for the second part
theorem inequality_solution_2 (a b : ℝ) : 
  (∀ x : ℝ, solution_set_2 x ↔ a^2 + b + 2 > 0) → a + b = 10 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l3444_344434


namespace NUMINAMATH_CALUDE_fraction_transformation_l3444_344459

theorem fraction_transformation (p q r s x y : ℝ) 
  (h1 : p ≠ q) 
  (h2 : q ≠ 0) 
  (h3 : y ≠ 0) 
  (h4 : s ≠ y * r) 
  (h5 : (p + x) / (q + y * x) = r / s) : 
  x = (q * r - p * s) / (s - y * r) := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l3444_344459


namespace NUMINAMATH_CALUDE_right_triangle_third_side_length_l3444_344499

theorem right_triangle_third_side_length (a b c : ℝ) : 
  a = 8 → b = 15 → c ≥ 0 → a^2 + b^2 = c^2 → c ≥ 17 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_length_l3444_344499


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l3444_344432

/-- Given a point M in polar coordinates (r, θ), 
    prove that its Cartesian coordinates are (x, y) --/
theorem polar_to_cartesian 
  (r : ℝ) (θ : ℝ) 
  (x : ℝ) (y : ℝ) 
  (h1 : r = 2) 
  (h2 : θ = π/6) 
  (h3 : x = r * Real.cos θ) 
  (h4 : y = r * Real.sin θ) : 
  x = Real.sqrt 3 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l3444_344432


namespace NUMINAMATH_CALUDE_cookie_dough_thickness_l3444_344426

/-- The thickness of a cylindrical layer formed by doubling the volume of a sphere
    and spreading it over a circular area. -/
theorem cookie_dough_thickness 
  (initial_radius : ℝ) 
  (final_radius : ℝ) 
  (initial_radius_value : initial_radius = 3)
  (final_radius_value : final_radius = 9) :
  let initial_volume := (4/3) * Real.pi * initial_radius^3
  let doubled_volume := 2 * initial_volume
  let final_area := Real.pi * final_radius^2
  let thickness := doubled_volume / final_area
  thickness = 8/9 := by sorry

end NUMINAMATH_CALUDE_cookie_dough_thickness_l3444_344426


namespace NUMINAMATH_CALUDE_intersection_of_N_and_complement_of_M_l3444_344479

open Set

theorem intersection_of_N_and_complement_of_M : 
  let M : Set ℝ := {x | x > 2}
  let N : Set ℝ := {x | 1 < x ∧ x < 3}
  (N ∩ (univ \ M)) = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_N_and_complement_of_M_l3444_344479


namespace NUMINAMATH_CALUDE_triangle_longest_side_l3444_344487

theorem triangle_longest_side (x : ℚ) : 
  (x + 3 : ℚ) + (2 * x - 1 : ℚ) + (3 * x + 5 : ℚ) = 45 → 
  max (x + 3) (max (2 * x - 1) (3 * x + 5)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l3444_344487


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3444_344436

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3444_344436


namespace NUMINAMATH_CALUDE_pizza_dough_milk_calculation_l3444_344419

/-- Given a ratio of milk to flour for pizza dough, calculate the amount of milk needed for a specific amount of flour. -/
theorem pizza_dough_milk_calculation 
  (milk_base : ℚ)  -- Base amount of milk in mL
  (flour_base : ℚ) -- Base amount of flour in mL
  (flour_total : ℚ) -- Total amount of flour to be used in mL
  (h1 : milk_base = 50)  -- Condition 1: Base milk amount
  (h2 : flour_base = 250) -- Condition 1: Base flour amount
  (h3 : flour_total = 750) -- Condition 2: Total flour amount
  : (flour_total / flour_base) * milk_base = 150 := by
  sorry

end NUMINAMATH_CALUDE_pizza_dough_milk_calculation_l3444_344419


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l3444_344455

/-- Taxi fare calculation -/
theorem taxi_fare_calculation 
  (base_distance : ℝ) 
  (rate_multiplier : ℝ) 
  (total_distance_1 : ℝ) 
  (total_fare_1 : ℝ) 
  (total_distance_2 : ℝ) 
  (h1 : base_distance = 60) 
  (h2 : rate_multiplier = 1.25) 
  (h3 : total_distance_1 = 80) 
  (h4 : total_fare_1 = 180) 
  (h5 : total_distance_2 = 100) :
  let base_rate := total_fare_1 / (base_distance + rate_multiplier * (total_distance_1 - base_distance))
  let total_fare_2 := base_rate * (base_distance + rate_multiplier * (total_distance_2 - base_distance))
  total_fare_2 = 3960 / 17 := by
sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l3444_344455


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l3444_344445

-- Define the arithmetic sequence
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

-- State the theorem
theorem arithmetic_sequence_common_difference_range :
  ∀ d : ℝ,
  (∀ n : ℕ, n < 6 → arithmeticSequence (-15) d n ≤ 0) ∧
  (∀ n : ℕ, n ≥ 6 → arithmeticSequence (-15) d n > 0) →
  3 < d ∧ d ≤ 15/4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l3444_344445


namespace NUMINAMATH_CALUDE_additional_chicken_wings_l3444_344462

theorem additional_chicken_wings (num_friends : ℕ) (initial_wings : ℕ) (wings_per_person : ℕ) : 
  num_friends = 4 → initial_wings = 9 → wings_per_person = 4 →
  num_friends * wings_per_person - initial_wings = 7 := by
  sorry

end NUMINAMATH_CALUDE_additional_chicken_wings_l3444_344462


namespace NUMINAMATH_CALUDE_min_omega_value_l3444_344424

theorem min_omega_value (y : ℝ → ℝ) (ω : ℝ) :
  (∀ x, y x = 2 * Real.sin (ω * x + π / 3)) →
  ω > 0 →
  (∀ x, y x = y (x - π / 3)) →
  (∃ k : ℕ, ω = 6 * k) →
  6 ≤ ω :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l3444_344424


namespace NUMINAMATH_CALUDE_dance_parity_l3444_344472

theorem dance_parity (n : ℕ) (h_odd : Odd n) (dances : Fin n → ℕ) : 
  ∃ i : Fin n, Even (dances i) := by
  sorry

end NUMINAMATH_CALUDE_dance_parity_l3444_344472


namespace NUMINAMATH_CALUDE_union_equals_interval_l3444_344461

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {x : ℝ | x^2 - 4*x ≤ 0}

-- Define the interval (-3, 4]
def interval : Set ℝ := Set.Ioc (-3) 4

-- Theorem statement
theorem union_equals_interval : A ∪ B = interval := by sorry

end NUMINAMATH_CALUDE_union_equals_interval_l3444_344461


namespace NUMINAMATH_CALUDE_art_team_arrangement_l3444_344485

/-- Given a team of 1000 members arranged in rows where each row from the second onward
    has one more person than the previous row, prove that there are 25 rows with 28 members
    in the first row. -/
theorem art_team_arrangement (k m : ℕ) : k > 16 →
  (k * (2 * m + k - 1)) / 2 = 1000 → k = 25 ∧ m = 28 := by
  sorry

end NUMINAMATH_CALUDE_art_team_arrangement_l3444_344485


namespace NUMINAMATH_CALUDE_problem_solution_l3444_344431

theorem problem_solution (x y : ℝ) (hx : x = 3 + 2 * Real.sqrt 2) (hy : y = 3 - 2 * Real.sqrt 2) :
  (x + y = 6) ∧
  (x - y = 4 * Real.sqrt 2) ∧
  (x * y = 1) ∧
  (x^2 - 3*x*y + y^2 - x - y = 25) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3444_344431


namespace NUMINAMATH_CALUDE_height_difference_l3444_344420

/-- The height of the Eiffel Tower in meters -/
def eiffel_tower_height_m : ℝ := 324

/-- The height of the Eiffel Tower in feet -/
def eiffel_tower_height_ft : ℝ := 1063

/-- The height of the Burj Khalifa in meters -/
def burj_khalifa_height_m : ℝ := 830

/-- The height of the Burj Khalifa in feet -/
def burj_khalifa_height_ft : ℝ := 2722

/-- The difference in height between the Burj Khalifa and the Eiffel Tower in meters and feet -/
theorem height_difference :
  (burj_khalifa_height_m - eiffel_tower_height_m = 506) ∧
  (burj_khalifa_height_ft - eiffel_tower_height_ft = 1659) := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l3444_344420


namespace NUMINAMATH_CALUDE_product_104_96_l3444_344477

theorem product_104_96 : 104 * 96 = 9984 := by
  sorry

end NUMINAMATH_CALUDE_product_104_96_l3444_344477


namespace NUMINAMATH_CALUDE_max_quadratic_solution_l3444_344451

theorem max_quadratic_solution (k a b c : ℕ+) (r : ℝ) :
  (∃ m n l : ℕ, a = k ^ m ∧ b = k ^ n ∧ c = k ^ l) →
  (a * r ^ 2 - b * r + c = 0) →
  (∀ x : ℝ, x ≠ r → a * x ^ 2 - b * x + c ≠ 0) →
  r < 100 →
  r ≤ 64 := by
sorry

end NUMINAMATH_CALUDE_max_quadratic_solution_l3444_344451


namespace NUMINAMATH_CALUDE_tournament_committee_count_l3444_344447

/-- Number of teams in the frisbee association -/
def num_teams : ℕ := 6

/-- Number of members in each team -/
def team_size : ℕ := 8

/-- Number of members selected from the host team -/
def host_select : ℕ := 3

/-- Number of members selected from each regular non-host team -/
def nonhost_select : ℕ := 2

/-- Number of members selected from the special non-host team -/
def special_nonhost_select : ℕ := 3

/-- Total number of possible tournament committees -/
def total_committees : ℕ := 11568055296

theorem tournament_committee_count :
  (num_teams) *
  (team_size.choose host_select) *
  ((team_size.choose nonhost_select) ^ (num_teams - 2)) *
  (team_size.choose special_nonhost_select) =
  total_committees :=
sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l3444_344447


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3444_344433

/-- Given a quadratic function f(x) = ax² + bx + 1 where a ≠ 0,
    if the solution set of f(x) > 0 is {x | x ∈ ℝ, x ≠ -b/(2a)},
    then the minimum value of (b⁴ + 4)/(4a) is 4. -/
theorem quadratic_minimum (a b : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, (a * x^2 + b * x + 1 > 0 ↔ x ≠ -b / (2 * a))) →
  (∃ m : ℝ, m = 4 ∧ ∀ y : ℝ, y = (b^4 + 4) / (4 * a) → y ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3444_344433


namespace NUMINAMATH_CALUDE_trisection_point_intersection_l3444_344457

noncomputable section

def f (x : ℝ) := Real.log x / Real.log 2

theorem trisection_point_intersection
  (x₁ x₂ : ℝ)
  (h_order : 0 < x₁ ∧ x₁ < x₂)
  (h_x₁ : x₁ = 4)
  (h_x₂ : x₂ = 16) :
  ∃ x₄ : ℝ, f x₄ = (2 * f x₁ + f x₂) / 3 ∧ x₄ = 2^(8/3) :=
sorry


end NUMINAMATH_CALUDE_trisection_point_intersection_l3444_344457


namespace NUMINAMATH_CALUDE_school_teachers_count_l3444_344452

theorem school_teachers_count (total_people : ℕ) (sample_size : ℕ) (students_in_sample : ℕ) 
  (h1 : total_people = 3200)
  (h2 : sample_size = 160)
  (h3 : students_in_sample = 150) :
  total_people - (total_people * students_in_sample / sample_size) = 200 := by
  sorry

end NUMINAMATH_CALUDE_school_teachers_count_l3444_344452


namespace NUMINAMATH_CALUDE_hens_count_l3444_344475

/-- Represents the number of hens and cows in a farm -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of heads in the farm -/
def totalHeads (f : Farm) : ℕ := f.hens + f.cows

/-- The total number of feet in the farm -/
def totalFeet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- A farm satisfying the given conditions -/
def satisfiesConditions (f : Farm) : Prop :=
  totalHeads f = 50 ∧ totalFeet f = 144

theorem hens_count (f : Farm) (h : satisfiesConditions f) : f.hens = 28 := by
  sorry

end NUMINAMATH_CALUDE_hens_count_l3444_344475


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3444_344443

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3444_344443


namespace NUMINAMATH_CALUDE_square_side_equals_circle_circumference_divided_by_four_l3444_344440

theorem square_side_equals_circle_circumference_divided_by_four (π : ℝ) (h : π = Real.pi) :
  let r : ℝ := 3
  let c : ℝ := 2 * π * r
  let y : ℝ := c / 4
  y = 3 * π / 2 := by
sorry

end NUMINAMATH_CALUDE_square_side_equals_circle_circumference_divided_by_four_l3444_344440


namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l3444_344441

-- Part 1
theorem calculation_proof : (1 * (1/2)⁻¹) + 2 * Real.cos (π/4) - Real.sqrt 8 + |1 - Real.sqrt 2| = 1 := by sorry

-- Part 2
theorem inequality_system_solution :
  ∀ x : ℝ, (x/2 + 1 > 0 ∧ 2*(x-1) + 3 ≥ 3*x) ↔ (-2 < x ∧ x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l3444_344441


namespace NUMINAMATH_CALUDE_prob_at_least_one_japanese_events_independent_iff_l3444_344490

-- Define the Little Green Lotus structure
structure LittleGreenLotus where
  isBoy : Bool
  speaksJapanese : Bool
  speaksKorean : Bool

-- Define the total number of Little Green Lotus
def totalLotus : ℕ := 36

-- Define the number of boys and girls
def numBoys : ℕ := 12
def numGirls : ℕ := 24

-- Define the number of boys and girls who can speak Japanese
def numBoysJapanese : ℕ := 8
def numGirlsJapanese : ℕ := 12

-- Define the number of boys and girls who can speak Korean as variables
variable (m n : ℕ)

-- Define the constraints on m
axiom m_bounds : 6 ≤ m ∧ m ≤ 8

-- Define the events A and B
def eventA (lotus : LittleGreenLotus) : Prop := lotus.isBoy
def eventB (lotus : LittleGreenLotus) : Prop := lotus.speaksKorean

-- Theorem 1: Probability of at least one of two randomly selected Little Green Lotus can speak Japanese
theorem prob_at_least_one_japanese :
  (totalLotus.choose 2 - (totalLotus - numBoysJapanese - numGirlsJapanese).choose 2) / totalLotus.choose 2 = 17 / 21 := by
  sorry

-- Theorem 2: Events A and B are independent if and only if n = 2m
theorem events_independent_iff (m n : ℕ) (h : 6 ≤ m ∧ m ≤ 8) :
  (numBoys * (m + n) = m * totalLotus) ↔ n = 2 * m := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_japanese_events_independent_iff_l3444_344490


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3444_344422

/-- An arithmetic sequence with first three terms a-1, a+1, and 2a+3 has general term 2n-3 -/
theorem arithmetic_sequence_general_term (a : ℝ) (n : ℕ) :
  let a₁ := a - 1
  let a₂ := a + 1
  let a₃ := 2 * a + 3
  let d := a₂ - a₁
  let aₙ := a₁ + (n - 1) * d
  (a₁ + a₃) / 2 = a₂ → aₙ = 2 * n - 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3444_344422


namespace NUMINAMATH_CALUDE_tens_digit_of_2013_pow_2018_minus_2019_l3444_344418

theorem tens_digit_of_2013_pow_2018_minus_2019 :
  ∃ n : ℕ, 2013^2018 - 2019 = 100 * n + 50 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_2013_pow_2018_minus_2019_l3444_344418


namespace NUMINAMATH_CALUDE_total_owed_after_borrowing_l3444_344470

/-- The total amount owed when borrowing additional money -/
theorem total_owed_after_borrowing (initial_debt additional_borrowed : ℕ) :
  initial_debt = 20 →
  additional_borrowed = 8 →
  initial_debt + additional_borrowed = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_owed_after_borrowing_l3444_344470


namespace NUMINAMATH_CALUDE_initial_roses_count_l3444_344400

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of roses added to the vase -/
def added_roses : ℕ := 10

/-- The total number of roses after adding -/
def total_roses : ℕ := 16

/-- Theorem stating that the initial number of roses is 6 -/
theorem initial_roses_count : initial_roses = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_roses_count_l3444_344400


namespace NUMINAMATH_CALUDE_compound_oxygen_count_l3444_344439

/-- Represents the number of atoms of an element in a compound -/
@[ext] structure AtomCount where
  al : ℕ
  o : ℕ
  h : ℕ

/-- Calculates the molecular weight of a compound given its atom counts -/
def molecularWeight (atoms : AtomCount) : ℕ :=
  27 * atoms.al + 16 * atoms.o + atoms.h

/-- Theorem stating that a compound with 1 Al, 3 H, and molecular weight 78 has 3 O atoms -/
theorem compound_oxygen_count :
  ∃ (atoms : AtomCount),
    atoms.al = 1 ∧
    atoms.h = 3 ∧
    molecularWeight atoms = 78 ∧
    atoms.o = 3 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_count_l3444_344439


namespace NUMINAMATH_CALUDE_sally_buttons_count_l3444_344446

/-- The number of buttons needed for all clothing items Sally sews over three days -/
def total_buttons : ℕ :=
  let shirt_buttons := 5
  let pants_buttons := 3
  let jacket_buttons := 10
  let monday := 4 * shirt_buttons + 2 * pants_buttons + 1 * jacket_buttons
  let tuesday := 3 * shirt_buttons + 1 * pants_buttons + 2 * jacket_buttons
  let wednesday := 2 * shirt_buttons + 3 * pants_buttons + 1 * jacket_buttons
  monday + tuesday + wednesday

/-- Theorem stating that the total number of buttons Sally needs is 103 -/
theorem sally_buttons_count : total_buttons = 103 := by
  sorry

end NUMINAMATH_CALUDE_sally_buttons_count_l3444_344446


namespace NUMINAMATH_CALUDE_smallest_value_sum_of_fractions_lower_bound_achievable_l3444_344404

theorem smallest_value_sum_of_fractions (a b : ℤ) (h : a > b) :
  (((a + b : ℚ) / (a - b)) + ((a - b : ℚ) / (a + b))) ≥ 2 :=
sorry

theorem lower_bound_achievable :
  ∃ (a b : ℤ), a > b ∧ (((a + b : ℚ) / (a - b)) + ((a - b : ℚ) / (a + b))) = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_sum_of_fractions_lower_bound_achievable_l3444_344404


namespace NUMINAMATH_CALUDE_negative_b_from_cubic_inequality_l3444_344471

theorem negative_b_from_cubic_inequality (a b : ℝ) 
  (h1 : a * b ≠ 0)
  (h2 : ∀ x : ℝ, x ≥ 0 → (x - a) * (x - b) * (x - 2*a - b) ≥ 0) :
  b < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_b_from_cubic_inequality_l3444_344471


namespace NUMINAMATH_CALUDE_sum_zero_implies_product_sum_nonpositive_l3444_344488

theorem sum_zero_implies_product_sum_nonpositive
  (a b c : ℝ) (h : a + b + c = 0) :
  a * b + b * c + c * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_implies_product_sum_nonpositive_l3444_344488


namespace NUMINAMATH_CALUDE_group_size_proof_l3444_344444

def group_collection (n : ℕ) : ℕ := n * n

theorem group_size_proof (total_rupees : ℕ) (h : group_collection 90 = total_rupees * 100) : 
  ∃ (n : ℕ), group_collection n = total_rupees * 100 ∧ n = 90 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l3444_344444


namespace NUMINAMATH_CALUDE_mean_squares_sum_l3444_344481

theorem mean_squares_sum (a b c : ℝ) : 
  (a + b + c) / 3 = 12 →
  (a * b * c) ^ (1/3 : ℝ) = 5 →
  3 / (1/a + 1/b + 1/c) = 4 →
  a^2 + b^2 + c^2 = 1108.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_squares_sum_l3444_344481


namespace NUMINAMATH_CALUDE_quadratic_relationship_l3444_344437

/-- A quadratic function f(x) = 3x^2 + ax + b where f(x - 1) is an even function -/
def f (a b : ℝ) (x : ℝ) : ℝ := 3 * x^2 + a * x + b

theorem quadratic_relationship (a b : ℝ) 
  (h : ∀ x, f a b (x - 1) = f a b (1 - x)) : 
  f a b (-1) < f a b (-3/2) ∧ f a b (-3/2) = f a b (3/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_relationship_l3444_344437


namespace NUMINAMATH_CALUDE_playground_area_l3444_344476

/-- The area of a rectangular playground with perimeter 90 meters and length three times the width -/
theorem playground_area : 
  ∀ (length width : ℝ),
  length > 0 → width > 0 →
  2 * (length + width) = 90 →
  length = 3 * width →
  length * width = 379.6875 := by
sorry

end NUMINAMATH_CALUDE_playground_area_l3444_344476


namespace NUMINAMATH_CALUDE_apartments_on_more_floors_proof_l3444_344491

/-- Represents the number of apartments on a floor with more apartments -/
def apartments_on_more_floors : ℕ := 6

/-- Represents the total number of floors in the building -/
def total_floors : ℕ := 12

/-- Represents the number of apartments on floors with fewer apartments -/
def apartments_on_fewer_floors : ℕ := 5

/-- Represents the maximum number of residents per apartment -/
def max_residents_per_apartment : ℕ := 4

/-- Represents the maximum total number of residents in the building -/
def max_total_residents : ℕ := 264

theorem apartments_on_more_floors_proof :
  let floors_with_more := total_floors / 2
  let floors_with_fewer := total_floors / 2
  let total_apartments_fewer := floors_with_fewer * apartments_on_fewer_floors
  let total_apartments := max_total_residents / max_residents_per_apartment
  let apartments_on_more_total := total_apartments - total_apartments_fewer
  apartments_on_more_floors = apartments_on_more_total / floors_with_more :=
by
  sorry

#check apartments_on_more_floors_proof

end NUMINAMATH_CALUDE_apartments_on_more_floors_proof_l3444_344491


namespace NUMINAMATH_CALUDE_three_digit_sum_product_l3444_344489

theorem three_digit_sum_product (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) :
  let y : ℕ := 9
  let z : ℕ := 9
  100 * x + 10 * y + z = x + y + z + x * y + y * z + z * x + x * y * z :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_product_l3444_344489


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3444_344478

def A : Set ℕ := {0, 1, 2}

def B : Set ℕ := {x | ∃ a ∈ A, x = 2^a}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3444_344478


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l3444_344469

theorem quadratic_vertex_form (x : ℝ) : 
  ∃ (a h k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l3444_344469


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3444_344467

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  eq : (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1
  a_pos : a > 0
  b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point (x y : ℝ)

/-- The left focus of the hyperbola -/
def leftFocus (h : Hyperbola a b) : Point c 0 := sorry

/-- The right vertex of the hyperbola -/
def rightVertex (h : Hyperbola a b) : Point a 0 := sorry

/-- The upper endpoint of the imaginary axis -/
def upperImaginaryEndpoint (h : Hyperbola a b) : Point 0 b := sorry

/-- The point where AB intersects the asymptote -/
def intersectionPoint (h : Hyperbola a b) : Point (a/2) (b/2) := sorry

/-- FM bisects ∠BFA -/
def fmBisectsAngle (h : Hyperbola a b) : Prop := sorry

/-- Eccentricity of the hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Main theorem: The eccentricity of the hyperbola is 1 + √3 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (bisect : fmBisectsAngle h) : eccentricity h = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3444_344467


namespace NUMINAMATH_CALUDE_matrix_determinant_from_eigenvectors_l3444_344464

/-- Given a 2x2 matrix A with specific eigenvectors and eigenvalues, prove that its determinant is -4 -/
theorem matrix_determinant_from_eigenvectors (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A.mulVec ![1, -1] = (-1 : ℝ) • ![1, -1]) → 
  (A.mulVec ![3, 2] = 4 • ![3, 2]) → 
  a * d - b * c = -4 := by
  sorry


end NUMINAMATH_CALUDE_matrix_determinant_from_eigenvectors_l3444_344464


namespace NUMINAMATH_CALUDE_students_per_class_l3444_344412

/-- Given that John buys index cards for his students, this theorem proves
    the number of students in each class. -/
theorem students_per_class
  (total_packs : ℕ)
  (num_classes : ℕ)
  (packs_per_student : ℕ)
  (h1 : total_packs = 360)
  (h2 : num_classes = 6)
  (h3 : packs_per_student = 2) :
  total_packs / (num_classes * packs_per_student) = 30 :=
by sorry

end NUMINAMATH_CALUDE_students_per_class_l3444_344412


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l3444_344421

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

theorem parallel_vectors_sum (m : ℝ) :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![m, 2]
  are_parallel a b →
  (3 • a + 2 • b) = ![14, 7] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l3444_344421


namespace NUMINAMATH_CALUDE_school_bus_capacity_l3444_344407

theorem school_bus_capacity 
  (columns_per_bus : ℕ) 
  (rows_per_bus : ℕ) 
  (number_of_buses : ℕ) 
  (h1 : columns_per_bus = 4) 
  (h2 : rows_per_bus = 10) 
  (h3 : number_of_buses = 6) : 
  columns_per_bus * rows_per_bus * number_of_buses = 240 := by
  sorry

end NUMINAMATH_CALUDE_school_bus_capacity_l3444_344407


namespace NUMINAMATH_CALUDE_drawings_on_last_page_l3444_344449

-- Define the given conditions
def initial_notebooks : ℕ := 10
def pages_per_notebook : ℕ := 30
def initial_drawings_per_page : ℕ := 4
def new_drawings_per_page : ℕ := 8
def filled_notebooks : ℕ := 6
def filled_pages_in_seventh : ℕ := 25

-- Define the theorem
theorem drawings_on_last_page : 
  let total_drawings := initial_notebooks * pages_per_notebook * initial_drawings_per_page
  let full_pages := total_drawings / new_drawings_per_page
  let pages_in_complete_notebooks := filled_notebooks * pages_per_notebook
  let remaining_drawings := total_drawings - (full_pages * new_drawings_per_page)
  remaining_drawings = 0 := by
  sorry

end NUMINAMATH_CALUDE_drawings_on_last_page_l3444_344449


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3444_344456

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (Complex.I * b : ℂ) = (1 + a * Complex.I) / (1 - Complex.I)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3444_344456


namespace NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l3444_344466

theorem lcm_from_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 12 → a * b = 2460 → Nat.lcm a b = 205 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l3444_344466


namespace NUMINAMATH_CALUDE_product_digit_exclusion_l3444_344409

theorem product_digit_exclusion : ∃ d : ℕ, d < 10 ∧ 
  (32 % 10 ≠ d) ∧ ((1024 / 32) % 10 ≠ d) := by
  sorry

end NUMINAMATH_CALUDE_product_digit_exclusion_l3444_344409


namespace NUMINAMATH_CALUDE_falls_difference_l3444_344405

/-- The number of falls for each person --/
structure Falls where
  steven : ℕ
  stephanie : ℕ
  sonya : ℕ

/-- The conditions of the problem --/
def satisfies_conditions (f : Falls) : Prop :=
  f.steven = 3 ∧
  f.stephanie > f.steven ∧
  f.sonya = 6 ∧
  f.sonya = f.stephanie / 2 - 2

/-- The theorem to prove --/
theorem falls_difference (f : Falls) (h : satisfies_conditions f) :
  f.stephanie - f.steven = 13 := by
  sorry

end NUMINAMATH_CALUDE_falls_difference_l3444_344405


namespace NUMINAMATH_CALUDE_opposite_of_three_l3444_344403

theorem opposite_of_three : -(3 : ℝ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l3444_344403


namespace NUMINAMATH_CALUDE_new_average_age_l3444_344435

theorem new_average_age (n : ℕ) (initial_avg : ℝ) (new_person_age : ℝ) :
  n = 17 ∧ initial_avg = 14 ∧ new_person_age = 32 →
  (n * initial_avg + new_person_age) / (n + 1) = 15 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l3444_344435


namespace NUMINAMATH_CALUDE_doctor_team_formation_l3444_344453

theorem doctor_team_formation (male_doctors female_doctors team_size : ℕ) 
  (h1 : male_doctors = 5)
  (h2 : female_doctors = 4)
  (h3 : team_size = 3) : 
  (Nat.choose male_doctors 2 * Nat.choose female_doctors 1 + 
   Nat.choose male_doctors 1 * Nat.choose female_doctors 2) = 70 := by
  sorry

end NUMINAMATH_CALUDE_doctor_team_formation_l3444_344453


namespace NUMINAMATH_CALUDE_composite_property_l3444_344408

theorem composite_property (n : ℕ) 
  (h1 : ∃ a : ℕ, 3 * n + 1 = a ^ 2)
  (h2 : ∃ b : ℕ, 10 * n + 1 = b ^ 2) : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ 29 * n + 11 = x * y :=
sorry

end NUMINAMATH_CALUDE_composite_property_l3444_344408


namespace NUMINAMATH_CALUDE_oil_price_reduction_is_fifty_percent_l3444_344484

/-- Calculates the percentage reduction in oil price given the reduced price and additional quantity -/
def oil_price_reduction (reduced_price : ℚ) (additional_quantity : ℚ) : ℚ :=
  let original_price := (800 : ℚ) / (((800 : ℚ) / reduced_price) - additional_quantity)
  ((original_price - reduced_price) / original_price) * 100

/-- Theorem stating that under the given conditions, the oil price reduction is 50% -/
theorem oil_price_reduction_is_fifty_percent :
  oil_price_reduction 80 5 = 50 := by sorry

end NUMINAMATH_CALUDE_oil_price_reduction_is_fifty_percent_l3444_344484


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3444_344417

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 24) :
  1 / x + 1 / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3444_344417


namespace NUMINAMATH_CALUDE_complex_power_modulus_l3444_344427

theorem complex_power_modulus : Complex.abs ((2 + Complex.I) ^ 8) = 625 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l3444_344427


namespace NUMINAMATH_CALUDE_candy_distribution_l3444_344473

theorem candy_distribution (total_candies : ℕ) (candies_per_student : ℕ) 
  (h1 : total_candies = 901)
  (h2 : candies_per_student = 53)
  (h3 : total_candies % candies_per_student = 0) :
  total_candies / candies_per_student = 17 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3444_344473


namespace NUMINAMATH_CALUDE_f_upper_bound_l3444_344410

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧
  f 1 = 1 ∧
  ∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂

theorem f_upper_bound (f : ℝ → ℝ) (h : f_properties f) :
  ∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x :=
sorry

end NUMINAMATH_CALUDE_f_upper_bound_l3444_344410


namespace NUMINAMATH_CALUDE_triangle_properties_l3444_344430

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  sin (2 * C) = Real.sqrt 3 * sin C →
  b = 4 →
  (1 / 2) * a * b * sin C = 2 * Real.sqrt 3 →
  -- Conclusions
  C = π / 6 ∧
  a + b + c = 6 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3444_344430


namespace NUMINAMATH_CALUDE_wednesday_fraction_is_one_fourth_l3444_344496

/-- Represents the daily fabric delivery and earnings for a textile company. -/
structure TextileDelivery where
  monday_yards : ℕ
  tuesday_multiplier : ℕ
  fabric_cost : ℕ
  total_earnings : ℕ

/-- Calculates the fraction of fabric delivered on Wednesday compared to Tuesday. -/
def wednesday_fraction (d : TextileDelivery) : ℚ :=
  let monday_earnings := d.monday_yards * d.fabric_cost
  let tuesday_yards := d.monday_yards * d.tuesday_multiplier
  let tuesday_earnings := tuesday_yards * d.fabric_cost
  let wednesday_earnings := d.total_earnings - monday_earnings - tuesday_earnings
  let wednesday_yards := wednesday_earnings / d.fabric_cost
  wednesday_yards / tuesday_yards

/-- Theorem stating that the fraction of fabric delivered on Wednesday compared to Tuesday is 1/4. -/
theorem wednesday_fraction_is_one_fourth (d : TextileDelivery) 
    (h1 : d.monday_yards = 20)
    (h2 : d.tuesday_multiplier = 2)
    (h3 : d.fabric_cost = 2)
    (h4 : d.total_earnings = 140) : 
  wednesday_fraction d = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_fraction_is_one_fourth_l3444_344496


namespace NUMINAMATH_CALUDE_triangle_circles_QR_length_l3444_344401

-- Define the right triangle DEF
def Triangle (DE EF DF : ℝ) := DE = 5 ∧ EF = 12 ∧ DF = 13

-- Define the circle centered at Q
def CircleQ (Q E D : ℝ × ℝ) := 
  (Q.1 - E.1)^2 + (Q.2 - E.2)^2 = (Q.1 - D.1)^2 + (Q.2 - D.2)^2

-- Define the circle centered at R
def CircleR (R D F : ℝ × ℝ) := 
  (R.1 - D.1)^2 + (R.2 - D.2)^2 = (R.1 - F.1)^2 + (R.2 - F.2)^2

-- Define the tangency conditions
def TangentQ (Q E : ℝ × ℝ) := True  -- Placeholder for tangency condition
def TangentR (R D : ℝ × ℝ) := True  -- Placeholder for tangency condition

-- State the theorem
theorem triangle_circles_QR_length 
  (D E F Q R : ℝ × ℝ) 
  (h_triangle : Triangle (dist D E) (dist E F) (dist D F))
  (h_circleQ : CircleQ Q E D)
  (h_circleR : CircleR R D F)
  (h_tangentQ : TangentQ Q E)
  (h_tangentR : TangentR R D) :
  dist Q R = 5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_circles_QR_length_l3444_344401


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3444_344498

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^4) = 0.0094 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3444_344498


namespace NUMINAMATH_CALUDE_removed_triangles_area_l3444_344497

/-- Given a square with side length x and isosceles right triangles removed from each corner to form a rectangle with perimeter 32, the combined area of the four removed triangles is x²/2. -/
theorem removed_triangles_area (x : ℝ) (r s : ℝ) : 
  x > 0 → 
  2 * (r + s) + 2 * |r - s| = 32 → 
  (r + s)^2 + (r - s)^2 = x^2 → 
  2 * r * s = x^2 / 2 :=
by sorry

#check removed_triangles_area

end NUMINAMATH_CALUDE_removed_triangles_area_l3444_344497


namespace NUMINAMATH_CALUDE_target_line_is_correct_l3444_344474

/-- The line we want to prove is correct -/
def target_line (x y : ℝ) : Prop := y = -3 * x - 2

/-- The line perpendicular to our target line -/
def perpendicular_line (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

/-- The curve to which our target line is tangent -/
def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 - 1

/-- Theorem stating that our target line is perpendicular to the given line
    and tangent to the given curve -/
theorem target_line_is_correct :
  (∀ x y : ℝ, perpendicular_line x y → 
    ∃ k : ℝ, k ≠ 0 ∧ (∀ x' y' : ℝ, target_line x' y' → 
      y' - y = k * (x' - x))) ∧ 
  (∃ x y : ℝ, target_line x y ∧ y = curve x ∧ 
    ∀ h : ℝ, h ≠ 0 → (curve (x + h) - curve x) / h ≠ -3) :=
sorry

end NUMINAMATH_CALUDE_target_line_is_correct_l3444_344474


namespace NUMINAMATH_CALUDE_smallest_n_squared_l3444_344415

theorem smallest_n_squared (n : ℕ+) : 
  (∃ x y z : ℕ+, n.val^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10) ↔ 
  n.val ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_squared_l3444_344415


namespace NUMINAMATH_CALUDE_square_minus_twelve_plus_fiftyfour_l3444_344460

theorem square_minus_twelve_plus_fiftyfour (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : a^2 + b^2 = 74) (h4 : a * b = 35) : 
  a^2 - 12 * a + 54 = 19 := by
sorry

end NUMINAMATH_CALUDE_square_minus_twelve_plus_fiftyfour_l3444_344460


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l3444_344448

theorem min_sum_with_reciprocal_constraint (x y : ℝ) : 
  x > 0 → y > 0 → (1 / (x + 2) + 1 / (y + 2) = 1 / 6) → 
  x + y ≥ 20 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / (x + 2) + 1 / (y + 2) = 1 / 6 ∧ x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l3444_344448


namespace NUMINAMATH_CALUDE_at_least_one_triangle_inside_l3444_344492

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a pentagon -/
structure Pentagon :=
  (vertices : Fin 5 → Point)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (vertices : Fin 3 → Point)

/-- Checks if a pentagon is convex and equilateral -/
def isConvexEquilateralPentagon (p : Pentagon) : Prop :=
  sorry

/-- Constructs equilateral triangles on the sides of a pentagon -/
def constructTriangles (p : Pentagon) : Fin 5 → EquilateralTriangle :=
  sorry

/-- Checks if a triangle is entirely contained within a pentagon -/
def isTriangleContained (t : EquilateralTriangle) (p : Pentagon) : Prop :=
  sorry

/-- The main theorem -/
theorem at_least_one_triangle_inside (p : Pentagon) 
  (h : isConvexEquilateralPentagon p) :
  ∃ (i : Fin 5), isTriangleContained (constructTriangles p i) p :=
sorry

end NUMINAMATH_CALUDE_at_least_one_triangle_inside_l3444_344492


namespace NUMINAMATH_CALUDE_square_root_expressions_l3444_344406

theorem square_root_expressions :
  (∃ x : ℝ, x^2 = 12) ∧ 
  (∃ y : ℝ, y^2 = 8) ∧ 
  (∃ z : ℝ, z^2 = 6) ∧ 
  (∃ w : ℝ, w^2 = 3) ∧ 
  (∃ v : ℝ, v^2 = 2) →
  (∃ a b : ℝ, a^2 = 12 ∧ b^2 = 8 ∧ 
    a + b * Real.sqrt 6 = 6 * Real.sqrt 3) ∧
  (∃ c d e : ℝ, c^2 = 12 ∧ d^2 = 3 ∧ e^2 = 2 ∧
    c + 1 / (Real.sqrt 3 - Real.sqrt 2) - Real.sqrt 6 * d = 3 * Real.sqrt 3 - 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_square_root_expressions_l3444_344406


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3444_344480

/-- Given a function g: ℝ → ℝ satisfying the functional equation
    (g x * g y - g (x * y)) / 4 = x + y + 3 for all x, y ∈ ℝ,
    prove that g x = x + 4 for all x ∈ ℝ. -/
theorem functional_equation_solution (g : ℝ → ℝ)
    (h : ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 3) :
  ∀ x : ℝ, g x = x + 4 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3444_344480


namespace NUMINAMATH_CALUDE_peters_initial_money_l3444_344429

/-- The cost of Peter's glasses purchase --/
def glasses_purchase (small_cost large_cost : ℕ) (small_count large_count : ℕ) (change : ℕ) : Prop :=
  ∃ (initial_amount : ℕ),
    initial_amount = small_cost * small_count + large_cost * large_count + change

/-- Theorem stating Peter's initial amount of money --/
theorem peters_initial_money :
  glasses_purchase 3 5 8 5 1 → ∃ (initial_amount : ℕ), initial_amount = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_peters_initial_money_l3444_344429


namespace NUMINAMATH_CALUDE_max_digit_sum_diff_l3444_344428

/-- Sum of digits function -/
def S (x : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem max_digit_sum_diff :
  (∀ x : ℕ, x > 0 → S (x + 2019) - S x ≤ 12) ∧
  (∃ x : ℕ, x > 0 ∧ S (x + 2019) - S x = 12) :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_diff_l3444_344428


namespace NUMINAMATH_CALUDE_exists_alpha_for_sequence_l3444_344450

/-- A sequence of non-zero real numbers satisfying the given condition -/
def SequenceA (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n ≠ 0 ∧ a n ^ 2 - a (n - 1) * a (n + 1) = 1

/-- The theorem to be proved -/
theorem exists_alpha_for_sequence (a : ℕ → ℝ) (h : SequenceA a) :
  ∃ α : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = α * a n - a (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_alpha_for_sequence_l3444_344450


namespace NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l3444_344414

-- Define the function f
def f (x : ℝ) : ℝ := x^6 - 2*x^4 + 7*x

-- State the theorem
theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 42 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l3444_344414


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3444_344463

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3444_344463


namespace NUMINAMATH_CALUDE_square_field_side_length_l3444_344465

theorem square_field_side_length (area : ℝ) (side : ℝ) :
  area = 225 →
  side * side = area →
  side = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_field_side_length_l3444_344465


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_three_l3444_344416

theorem sum_of_roots_equals_three : ∃ (P Q : ℝ), 
  (∀ x : ℝ, 3 * x^2 - 9 * x + 6 = 0 ↔ (x = P ∨ x = Q)) ∧ 
  P + Q = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_three_l3444_344416


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3444_344425

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The maximum value function of |f(x)| on [0,t] -/
noncomputable def φ (t : ℝ) : ℝ :=
  if t ≤ 1 then 2*t - t^2
  else if t ≤ 1 + Real.sqrt 2 then 1
  else t^2 - 2*t

theorem quadratic_function_properties :
  (∀ x, f x ≥ -1) ∧  -- minimum value is -1
  (f 0 = 0) ∧        -- f(0) = 0
  (∀ x, f (1 + x) = f (1 - x)) ∧  -- symmetry property
  (∃ a b c, ∀ x, f x = a*x^2 + b*x + c ∧ a ≠ 0) →  -- f is quadratic
  (∀ x, f x = x^2 - 2*x) ∧  -- part 1
  (∀ m, (∀ x, -3 ≤ x ∧ x ≤ 3 → f x > 2*m*x - 4) ↔ -3 < m ∧ m < 1) ∧  -- part 2
  (∀ t, t > 0 → ∀ x, 0 ≤ x ∧ x ≤ t → |f x| ≤ φ t) ∧  -- part 3
  (∀ t, t > 0 → ∃ x, 0 ≤ x ∧ x ≤ t ∧ |f x| = φ t)  -- part 3 (maximum is achieved)
:= by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3444_344425


namespace NUMINAMATH_CALUDE_unique_score_170_l3444_344413

/-- Represents a test score --/
structure TestScore where
  total_questions : ℕ
  correct : ℕ
  wrong : ℕ
  score : ℤ

/-- Calculates the score based on the number of correct and wrong answers --/
def calculate_score (ts : TestScore) : ℤ :=
  30 + 4 * ts.correct - ts.wrong

/-- Checks if a TestScore is valid according to the rules --/
def is_valid_score (ts : TestScore) : Prop :=
  ts.correct + ts.wrong ≤ ts.total_questions ∧
  ts.score = calculate_score ts ∧
  ts.score > 90

/-- Theorem stating that 170 is the only score above 90 that uniquely determines the number of correct answers --/
theorem unique_score_170 :
  ∀ (ts : TestScore),
    ts.total_questions = 35 →
    is_valid_score ts →
    (∀ (ts' : TestScore),
      ts'.total_questions = 35 →
      is_valid_score ts' →
      ts'.score = ts.score →
      ts'.correct = ts.correct) →
    ts.score = 170 :=
sorry

end NUMINAMATH_CALUDE_unique_score_170_l3444_344413


namespace NUMINAMATH_CALUDE_male_salmon_count_l3444_344411

theorem male_salmon_count (female_salmon : ℕ) (total_salmon : ℕ) 
  (h1 : female_salmon = 259378) 
  (h2 : total_salmon = 971639) : 
  total_salmon - female_salmon = 712261 := by
  sorry

end NUMINAMATH_CALUDE_male_salmon_count_l3444_344411


namespace NUMINAMATH_CALUDE_perimeter_ratio_of_similar_triangles_l3444_344494

/-- Two triangles are similar with a given ratio -/
def similar_triangles (t1 t2 : Set (ℝ × ℝ)) (r : ℝ) : Prop := sorry

/-- The perimeter of a triangle -/
def perimeter (t : Set (ℝ × ℝ)) : ℝ := sorry

theorem perimeter_ratio_of_similar_triangles 
  (abc a1b1c1 : Set (ℝ × ℝ)) : 
  similar_triangles abc a1b1c1 (1/2) → 
  perimeter abc / perimeter a1b1c1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_of_similar_triangles_l3444_344494


namespace NUMINAMATH_CALUDE_hyperbola_equation_final_hyperbola_equation_l3444_344458

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (C₁ : ℝ → ℝ → Prop) (C₂ : ℝ → ℝ → Prop),
    (∀ x y, C₁ x y ↔ x^2 = 2*y) ∧ 
    (∀ x y, C₂ x y ↔ x^2/a^2 - y^2/b^2 = 1) ∧
    (∃ A : ℝ × ℝ, A.1 = a ∧ A.2 = 0 ∧ C₂ A.1 A.2) ∧
    (a^2 + b^2 = 5*a^2) ∧
    (∃ l : ℝ → ℝ, (∀ x, l x = b/a*(x - a)) ∧
      (∀ x, C₁ x (l x) → (∃! y, C₁ x y ∧ y = l x)))) →
  a = 1 ∧ b = 2 :=
by sorry

/-- The final form of the hyperbola equation -/
theorem final_hyperbola_equation :
  ∃ (C : ℝ → ℝ → Prop), ∀ x y, C x y ↔ x^2 - y^2/4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_final_hyperbola_equation_l3444_344458


namespace NUMINAMATH_CALUDE_roots_sum_product_squares_l3444_344438

theorem roots_sum_product_squares (x₁ x₂ : ℝ) : 
  ((x₁ - 2)^2 = 3*(x₁ + 5)) ∧ ((x₂ - 2)^2 = 3*(x₂ + 5)) →
  x₁*x₂ + x₁^2 + x₂^2 = 60 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_product_squares_l3444_344438


namespace NUMINAMATH_CALUDE_power_multiplication_l3444_344454

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3444_344454


namespace NUMINAMATH_CALUDE_total_slices_is_136_l3444_344483

/-- The number of slices in a small pizza -/
def small_slices : ℕ := 6

/-- The number of slices in a medium pizza -/
def medium_slices : ℕ := 8

/-- The number of slices in a large pizza -/
def large_slices : ℕ := 12

/-- The total number of pizzas bought -/
def total_pizzas : ℕ := 15

/-- The number of small pizzas ordered -/
def small_pizzas : ℕ := 4

/-- The number of medium pizzas ordered -/
def medium_pizzas : ℕ := 5

/-- Theorem stating that the total number of slices is 136 -/
theorem total_slices_is_136 : 
  small_pizzas * small_slices + 
  medium_pizzas * medium_slices + 
  (total_pizzas - small_pizzas - medium_pizzas) * large_slices = 136 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_is_136_l3444_344483


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_36_56_l3444_344442

theorem gcf_lcm_sum_36_56 : Nat.gcd 36 56 + Nat.lcm 36 56 = 508 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_36_56_l3444_344442


namespace NUMINAMATH_CALUDE_bridge_length_l3444_344482

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) :
  train_length = 145 →
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  (train_speed * crossing_time) - train_length = 230 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l3444_344482


namespace NUMINAMATH_CALUDE_largest_multiple_12_negation_gt_neg_150_l3444_344468

theorem largest_multiple_12_negation_gt_neg_150 :
  ∀ n : ℤ, n ≥ 0 → 12 ∣ n → -n > -150 → n ≤ 144 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_12_negation_gt_neg_150_l3444_344468

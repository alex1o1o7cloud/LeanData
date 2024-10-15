import Mathlib

namespace NUMINAMATH_CALUDE_pencils_left_l2064_206480

def initial_pencils : ℕ := 4527
def pencils_to_dorothy : ℕ := 1896
def pencils_to_samuel : ℕ := 754
def pencils_to_alina : ℕ := 307

theorem pencils_left : 
  initial_pencils - (pencils_to_dorothy + pencils_to_samuel + pencils_to_alina) = 1570 := by
  sorry

end NUMINAMATH_CALUDE_pencils_left_l2064_206480


namespace NUMINAMATH_CALUDE_hexagon_area_l2064_206452

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The hexagon defined by its vertices -/
def hexagon : List Point := [
  ⟨0, 3⟩, ⟨3, 3⟩, ⟨4, 0⟩, ⟨3, -3⟩, ⟨0, -3⟩, ⟨-1, 0⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point) : ℝ := sorry

/-- Theorem: The area of the specified hexagon is 18 square units -/
theorem hexagon_area : polygonArea hexagon = 18 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_l2064_206452


namespace NUMINAMATH_CALUDE_oak_willow_difference_l2064_206455

theorem oak_willow_difference (total_trees : ℕ) (willow_percent oak_percent : ℚ) : 
  total_trees = 712 →
  willow_percent = 34 / 100 →
  oak_percent = 45 / 100 →
  ⌊oak_percent * total_trees⌋ - ⌊willow_percent * total_trees⌋ = 78 := by
  sorry

end NUMINAMATH_CALUDE_oak_willow_difference_l2064_206455


namespace NUMINAMATH_CALUDE_mystery_number_multiple_of_four_l2064_206464

def mystery_number (k : ℕ) : ℕ := (2*k+2)^2 - (2*k)^2

theorem mystery_number_multiple_of_four (k : ℕ) :
  ∃ m : ℕ, mystery_number k = 4 * m :=
sorry

end NUMINAMATH_CALUDE_mystery_number_multiple_of_four_l2064_206464


namespace NUMINAMATH_CALUDE_fourth_sample_is_37_l2064_206400

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  first_sample : ℕ
  h_total : total_students > 0
  h_sample : sample_size > 0
  h_first : first_sample > 0
  h_first_le_total : first_sample ≤ total_students

/-- The sampling interval for a systematic sampling. -/
def sampling_interval (s : SystematicSampling) : ℕ :=
  s.total_students / s.sample_size

/-- The nth sample in a systematic sampling. -/
def nth_sample (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.first_sample + (n - 1) * sampling_interval s

/-- Theorem: In a systematic sampling of 64 students with a sample size of 4,
    if the first three samples are 5, 21, and 53, then the fourth sample must be 37. -/
theorem fourth_sample_is_37 :
  ∀ (s : SystematicSampling),
    s.total_students = 64 →
    s.sample_size = 4 →
    s.first_sample = 5 →
    nth_sample s 2 = 21 →
    nth_sample s 3 = 53 →
    nth_sample s 4 = 37 := by
  sorry


end NUMINAMATH_CALUDE_fourth_sample_is_37_l2064_206400


namespace NUMINAMATH_CALUDE_fresh_produce_to_soda_ratio_l2064_206473

/-- Proves that the ratio of fresh produce weight to soda weight is 2:1 --/
theorem fresh_produce_to_soda_ratio :
  let empty_truck_weight : ℕ := 12000
  let soda_crates : ℕ := 20
  let soda_crate_weight : ℕ := 50
  let dryers : ℕ := 3
  let dryer_weight : ℕ := 3000
  let loaded_truck_weight : ℕ := 24000
  let soda_weight := soda_crates * soda_crate_weight
  let dryers_weight := dryers * dryer_weight
  let fresh_produce_weight := loaded_truck_weight - (empty_truck_weight + soda_weight + dryers_weight)
  (fresh_produce_weight : ℚ) / soda_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_fresh_produce_to_soda_ratio_l2064_206473


namespace NUMINAMATH_CALUDE_river_speed_is_6_l2064_206481

/-- Proves that the speed of the river is 6 km/h given the conditions of the boat problem -/
theorem river_speed_is_6 (total_distance : ℝ) (downstream_distance : ℝ) (still_water_speed : ℝ)
  (h1 : total_distance = 150)
  (h2 : downstream_distance = 90)
  (h3 : still_water_speed = 30)
  (h4 : downstream_distance / (still_water_speed + 6) = (total_distance - downstream_distance) / (still_water_speed - 6)) :
  6 = 6 := by
sorry

end NUMINAMATH_CALUDE_river_speed_is_6_l2064_206481


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l2064_206478

theorem product_mod_seventeen : (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l2064_206478


namespace NUMINAMATH_CALUDE_barn_painted_area_l2064_206459

/-- Calculates the total area to be painted for a rectangular barn -/
def total_painted_area (width length height : ℝ) : ℝ :=
  2 * (width * height + length * height) + width * length

/-- Theorem stating the total area to be painted for the given barn dimensions -/
theorem barn_painted_area :
  total_painted_area 12 15 6 = 828 := by
  sorry

end NUMINAMATH_CALUDE_barn_painted_area_l2064_206459


namespace NUMINAMATH_CALUDE_stones_division_l2064_206412

/-- Definition of similar sizes -/
def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

/-- A step in the combining process -/
inductive CombineStep
  | combine (x y : ℕ) (h : similar_sizes x y) : CombineStep

/-- A sequence of combining steps -/
def CombineSequence := List CombineStep

/-- The result of applying a sequence of combining steps -/
def apply_sequence (initial : List ℕ) (seq : CombineSequence) : List ℕ := sorry

/-- The theorem stating that any pile can be divided into single stones -/
theorem stones_division (n : ℕ) : 
  ∃ (seq : CombineSequence), 
    apply_sequence (List.replicate n 1) seq = [n] := sorry

end NUMINAMATH_CALUDE_stones_division_l2064_206412


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l2064_206416

/-- The time (in minutes) it takes Betty to complete one lap -/
def betty_lap_time : ℕ := 5

/-- The time (in minutes) it takes Charles to complete one lap -/
def charles_lap_time : ℕ := 8

/-- The time (in minutes) it takes Lisa to complete one lap -/
def lisa_lap_time : ℕ := 9

/-- The time (in minutes) Lisa takes as a break after every two laps -/
def lisa_break_time : ℕ := 3

/-- The effective time (in minutes) it takes Lisa to complete one lap, considering her breaks -/
def lisa_effective_lap_time : ℚ := (2 * lisa_lap_time + lisa_break_time) / 2

/-- The start time of the jogging -/
def start_time : String := "6:00 AM"

/-- Proves that the earliest time when all three joggers meet back at the starting point is 1:00 PM -/
theorem earliest_meeting_time : 
  ∃ (t : ℕ), t * betty_lap_time = t * charles_lap_time ∧ 
             t * betty_lap_time = (t : ℚ) * lisa_effective_lap_time ∧
             t * betty_lap_time = 420 := by sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l2064_206416


namespace NUMINAMATH_CALUDE_division_and_addition_l2064_206437

theorem division_and_addition : (150 / (10 / 2)) + 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_l2064_206437


namespace NUMINAMATH_CALUDE_exists_a_divides_a_squared_minus_a_l2064_206414

theorem exists_a_divides_a_squared_minus_a (n k : ℕ) 
  (h1 : n > 1) 
  (h2 : k = (Nat.factors n).card) : 
  ∃ a : ℕ, 1 < a ∧ a < n / k + 1 ∧ n ∣ (a^2 - a) := by
  sorry

end NUMINAMATH_CALUDE_exists_a_divides_a_squared_minus_a_l2064_206414


namespace NUMINAMATH_CALUDE_stacy_berries_l2064_206405

theorem stacy_berries (total : ℕ) (stacy steve skylar : ℕ) : 
  total = 1100 →
  stacy = 4 * steve →
  steve = 2 * skylar →
  total = stacy + steve + skylar →
  stacy = 800 := by
sorry

end NUMINAMATH_CALUDE_stacy_berries_l2064_206405


namespace NUMINAMATH_CALUDE_probability_two_aces_standard_deck_l2064_206436

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ace_count : ℕ)

/-- The probability of drawing two Aces as the top two cards from a randomly arranged deck -/
def probability_two_aces (d : Deck) : ℚ :=
  (d.ace_count : ℚ) / d.total_cards * (d.ace_count - 1) / (d.total_cards - 1)

/-- Theorem: The probability of drawing two Aces as the top two cards from a standard deck is 1/221 -/
theorem probability_two_aces_standard_deck :
  probability_two_aces ⟨52, 4⟩ = 1 / 221 := by
  sorry

#eval probability_two_aces ⟨52, 4⟩

end NUMINAMATH_CALUDE_probability_two_aces_standard_deck_l2064_206436


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2064_206476

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 4 ∧ c^2 - 10*c + 16 = 0 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2064_206476


namespace NUMINAMATH_CALUDE_divisibility_by_six_l2064_206428

theorem divisibility_by_six (n : ℕ) : ∃ k : ℤ, (17 : ℤ)^n - (11 : ℤ)^n = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l2064_206428


namespace NUMINAMATH_CALUDE_root_problems_l2064_206407

theorem root_problems :
  (∃ x : ℝ, x^2 = 16 ∧ (x = 4 ∨ x = -4)) ∧
  (∃ y : ℝ, y^3 = -27 ∧ y = -3) ∧
  (Real.sqrt ((-4)^2) = 4) ∧
  (∃ z : ℝ, z^2 = 9 ∧ z = 3) := by
  sorry

end NUMINAMATH_CALUDE_root_problems_l2064_206407


namespace NUMINAMATH_CALUDE_simplify_expression_l2064_206495

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2064_206495


namespace NUMINAMATH_CALUDE_integer_pair_existence_l2064_206432

theorem integer_pair_existence : ∃ (x y : ℤ), 
  (x * y + (x + y) = 95) ∧ 
  (x * y - (x + y) = 59) ∧ 
  ((x = 11 ∧ y = 7) ∨ (x = 7 ∧ y = 11)) := by
  sorry

end NUMINAMATH_CALUDE_integer_pair_existence_l2064_206432


namespace NUMINAMATH_CALUDE_stamp_exhibition_problem_l2064_206445

theorem stamp_exhibition_problem (x : ℕ) : 
  (∃ (s : ℕ), s = 3 * (s / x) + 24 ∧ s = 4 * (s / x) - 26) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_stamp_exhibition_problem_l2064_206445


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2064_206462

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs (2 * z - w) = 29)
  (h2 : Complex.abs (z + 2 * w) = 7)
  (h3 : Complex.abs (z + w) = 3) :
  Complex.abs z = 11 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2064_206462


namespace NUMINAMATH_CALUDE_block_edge_sum_l2064_206415

/-- A rectangular block with a square base -/
structure Block where
  side : ℝ  -- side length of the square base
  height : ℝ  -- height of the block

/-- The volume of the block -/
def volume (b : Block) : ℝ := b.side^2 * b.height

/-- The surface area of the vertical sides of the block -/
def verticalSurfaceArea (b : Block) : ℝ := 4 * b.side * b.height

/-- The sum of the lengths of all edges of the block -/
def sumOfEdges (b : Block) : ℝ := 8 * b.side + 4 * b.height

theorem block_edge_sum (b : Block) 
  (h_volume : volume b = 576) 
  (h_area : verticalSurfaceArea b = 384) : 
  sumOfEdges b = 112 := by
  sorry


end NUMINAMATH_CALUDE_block_edge_sum_l2064_206415


namespace NUMINAMATH_CALUDE_ryan_reads_more_pages_l2064_206413

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pages Ryan read -/
def ryan_total_pages : ℕ := 2100

/-- The number of pages Ryan's brother read per day -/
def brother_pages_per_day : ℕ := 200

/-- The difference in average pages read per day between Ryan and his brother -/
def page_difference : ℕ := ryan_total_pages / days_in_week - brother_pages_per_day

theorem ryan_reads_more_pages :
  page_difference = 100 := by sorry

end NUMINAMATH_CALUDE_ryan_reads_more_pages_l2064_206413


namespace NUMINAMATH_CALUDE_ratio_problem_l2064_206494

theorem ratio_problem (second_part : ℝ) (percent : ℝ) (first_part : ℝ) :
  second_part = 5 →
  percent = 180 →
  first_part / second_part = percent / 100 →
  first_part = 9 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2064_206494


namespace NUMINAMATH_CALUDE_y_value_l2064_206458

theorem y_value : ∀ y : ℚ, (2 / 5 - 1 / 7 : ℚ) = 14 / y → y = 490 / 9 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2064_206458


namespace NUMINAMATH_CALUDE_correct_calculation_l2064_206420

theorem correct_calculation (x y : ℝ) : -x^2*y + 3*x^2*y = 2*x^2*y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2064_206420


namespace NUMINAMATH_CALUDE_greatest_integer_prime_quadratic_l2064_206403

theorem greatest_integer_prime_quadratic : 
  ∃ (x : ℤ), (∀ (y : ℤ), y > x → ¬(Nat.Prime (Int.natAbs (4*y^2 - 35*y + 21)))) ∧ 
  (Nat.Prime (Int.natAbs (4*x^2 - 35*x + 21))) ∧ 
  x = 8 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_prime_quadratic_l2064_206403


namespace NUMINAMATH_CALUDE_wrappers_found_at_park_l2064_206402

-- Define the variables
def bottle_caps_found : ℕ := 15
def total_wrappers : ℕ := 67
def total_bottle_caps : ℕ := 35
def wrapper_excess : ℕ := 32

-- Define the theorem
theorem wrappers_found_at_park :
  total_wrappers = total_bottle_caps + wrapper_excess →
  total_wrappers - (total_bottle_caps + wrapper_excess - bottle_caps_found) = 0 :=
by sorry

end NUMINAMATH_CALUDE_wrappers_found_at_park_l2064_206402


namespace NUMINAMATH_CALUDE_total_coins_is_32_l2064_206491

/-- The number of dimes -/
def num_dimes : ℕ := 22

/-- The number of quarters -/
def num_quarters : ℕ := 10

/-- The total number of coins -/
def total_coins : ℕ := num_dimes + num_quarters

/-- Theorem: The total number of coins is 32 -/
theorem total_coins_is_32 : total_coins = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_is_32_l2064_206491


namespace NUMINAMATH_CALUDE_sequence_2017_l2064_206487

/-- Property P: If aₚ = aₖ, then aₚ₊₁ = aₖ₊₁ for p, q ∈ ℕ* -/
def PropertyP (a : ℕ → ℕ) : Prop :=
  ∀ p q : ℕ, p ≠ 0 → q ≠ 0 → a p = a q → a (p + 1) = a (q + 1)

/-- The sequence satisfying the given conditions -/
def Sequence (a : ℕ → ℕ) : Prop :=
  PropertyP a ∧
  a 1 = 1 ∧
  a 2 = 2 ∧
  a 3 = 3 ∧
  a 5 = 2 ∧
  a 6 + a 7 + a 8 = 21

theorem sequence_2017 (a : ℕ → ℕ) (h : Sequence a) : a 2017 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2017_l2064_206487


namespace NUMINAMATH_CALUDE_total_shared_amount_l2064_206469

/-- Represents the money sharing problem with three people --/
structure MoneySharing where
  ratio1 : ℕ
  ratio2 : ℕ
  ratio3 : ℕ
  share1 : ℕ

/-- Theorem stating that given the conditions, the total shared amount is 195 --/
theorem total_shared_amount (ms : MoneySharing) 
  (h1 : ms.ratio1 = 2)
  (h2 : ms.ratio2 = 3)
  (h3 : ms.ratio3 = 8)
  (h4 : ms.share1 = 30) :
  ms.share1 + (ms.share1 / ms.ratio1 * ms.ratio2) + (ms.share1 / ms.ratio1 * ms.ratio3) = 195 := by
  sorry

#check total_shared_amount

end NUMINAMATH_CALUDE_total_shared_amount_l2064_206469


namespace NUMINAMATH_CALUDE_min_disks_for_given_files_l2064_206496

/-- Represents the minimum number of disks needed to store files --/
def min_disks (total_files : ℕ) (disk_space : ℚ) 
  (files_1_2MB : ℕ) (files_0_9MB : ℕ) (files_0_5MB : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the minimum number of disks needed --/
theorem min_disks_for_given_files : 
  min_disks 40 2 5 15 20 = 16 := by sorry

end NUMINAMATH_CALUDE_min_disks_for_given_files_l2064_206496


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2064_206424

/-- The x-coordinate of the end point of the line segment -/
def x : ℝ := 3.4213

/-- The y-coordinate of the end point of the line segment -/
def y : ℝ := 7.8426

/-- The start point of the line segment -/
def start_point : ℝ × ℝ := (2, 2)

/-- The length of the line segment -/
def segment_length : ℝ := 6

theorem line_segment_endpoint :
  x > 0 ∧
  y = 2 * x + 1 ∧
  Real.sqrt ((x - 2)^2 + (y - 2)^2) = segment_length :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2064_206424


namespace NUMINAMATH_CALUDE_staircase_perimeter_l2064_206499

/-- Represents a staircase-shaped region with specific properties -/
structure StaircaseRegion where
  tickMarkSides : ℕ
  tickMarkLength : ℝ
  bottomBaseLength : ℝ
  totalArea : ℝ

/-- Calculates the perimeter of a StaircaseRegion -/
def perimeter (s : StaircaseRegion) : ℝ :=
  s.bottomBaseLength + s.tickMarkSides * s.tickMarkLength

theorem staircase_perimeter (s : StaircaseRegion) 
  (h1 : s.tickMarkSides = 12)
  (h2 : s.tickMarkLength = 1)
  (h3 : s.bottomBaseLength = 12)
  (h4 : s.totalArea = 78) :
  perimeter s = 34.5 := by
  sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l2064_206499


namespace NUMINAMATH_CALUDE_circulus_vitiosus_characterization_l2064_206439

/-- Definition of a logical fallacy --/
def LogicalFallacy : Type := String

/-- Definition of a premise in an argument --/
def Premise : Type := String

/-- Definition of a conclusion in an argument --/
def Conclusion : Type := String

/-- Definition of an argument structure --/
structure Argument where
  premises : List Premise
  conclusion : Conclusion

/-- Definition of circular reasoning (circulus vitiosus) --/
def CirculusVitiosus (arg : Argument) : Prop :=
  arg.conclusion ∈ arg.premises

/-- Theorem stating the characteristic of circulus vitiosus --/
theorem circulus_vitiosus_characterization :
  ∀ (arg : Argument),
    CirculusVitiosus arg ↔
    ∃ (premise : Premise),
      premise ∈ arg.premises ∧ premise = arg.conclusion := by
  sorry

#check circulus_vitiosus_characterization

end NUMINAMATH_CALUDE_circulus_vitiosus_characterization_l2064_206439


namespace NUMINAMATH_CALUDE_range_of_a_for_second_quadrant_l2064_206472

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := (1 - Complex.I) * (a + Complex.I)

-- Define what it means for a complex number to be in the second quadrant
def in_second_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im > 0

-- State the theorem
theorem range_of_a_for_second_quadrant :
  ∀ a : ℝ, in_second_quadrant (z a) ↔ a < -1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_second_quadrant_l2064_206472


namespace NUMINAMATH_CALUDE_no_rational_roots_l2064_206435

-- Define the polynomial
def f (x : ℚ) : ℚ := 5 * x^3 - 4 * x^2 - 8 * x + 3

-- Theorem statement
theorem no_rational_roots : ∀ x : ℚ, f x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l2064_206435


namespace NUMINAMATH_CALUDE_sum_of_roots_l2064_206466

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x + 14*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 14*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2184 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2064_206466


namespace NUMINAMATH_CALUDE_rational_function_value_at_one_l2064_206465

/-- A rational function with specific properties --/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_quadratic : ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c
  q_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  asymptote_minus_three : q (-3) = 0
  asymptote_two : q 2 = 0
  passes_origin : p 0 = 0 ∧ q 0 ≠ 0
  passes_one_two : p 1 = 2 * q 1 ∧ q 1 ≠ 0

/-- The main theorem --/
theorem rational_function_value_at_one (f : RationalFunction) : f.p 1 / f.q 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_at_one_l2064_206465


namespace NUMINAMATH_CALUDE_a_value_is_negative_six_l2064_206454

/-- The coefficient of x^4 in the expansion of (2+ax)(1-x)^6 -/
def coefficient (a : ℝ) : ℝ := 30 - 20 * a

/-- The theorem stating that a = -6 given the coefficient of x^4 is 150 -/
theorem a_value_is_negative_six : 
  ∃ a : ℝ, coefficient a = 150 ∧ a = -6 :=
sorry

end NUMINAMATH_CALUDE_a_value_is_negative_six_l2064_206454


namespace NUMINAMATH_CALUDE_middle_book_price_l2064_206468

/-- A sequence of 49 numbers where each number differs by 5 from its adjacent numbers -/
def IncreasingSequence (a : ℕ → ℚ) : Prop :=
  (∀ n < 48, a (n + 1) = a n + 5) ∧ 
  (∀ n, n < 49)

theorem middle_book_price
  (a : ℕ → ℚ)
  (h1 : IncreasingSequence a)
  (h2 : a 48 = 2 * (a 23 + a 24 + a 25)) :
  a 24 = 24 := by
  sorry

end NUMINAMATH_CALUDE_middle_book_price_l2064_206468


namespace NUMINAMATH_CALUDE_triangle_base_height_proof_l2064_206444

theorem triangle_base_height_proof :
  ∀ (base height : ℝ),
    base = height - 4 →
    (1/2) * base * height = 96 →
    base = 12 ∧ height = 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_height_proof_l2064_206444


namespace NUMINAMATH_CALUDE_pencils_left_l2064_206451

def initial_pencils : ℕ := 142
def pencils_given_away : ℕ := 31

theorem pencils_left : initial_pencils - pencils_given_away = 111 := by
  sorry

end NUMINAMATH_CALUDE_pencils_left_l2064_206451


namespace NUMINAMATH_CALUDE_prime_power_triples_l2064_206417

theorem prime_power_triples (p : ℕ) (x y : ℕ+) :
  (Nat.Prime p ∧
   ∃ (a b : ℕ), x^(p-1) + y = p^a ∧ x + y^(p-1) = p^b) →
  ((p = 3 ∧ ((x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2))) ∨
   (p = 2 ∧ ∃ (k : ℕ), 0 < x.val ∧ x.val < 2^k ∧ y = ⟨2^k - x.val, sorry⟩)) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_triples_l2064_206417


namespace NUMINAMATH_CALUDE_string_length_problem_l2064_206404

theorem string_length_problem (total_strings : ℕ) (avg_length : ℝ) (other_strings : ℕ) (other_avg : ℝ) :
  total_strings = 6 →
  avg_length = 80 →
  other_strings = 4 →
  other_avg = 85 →
  let remaining_strings := total_strings - other_strings
  let total_length := avg_length * total_strings
  let other_length := other_avg * other_strings
  let remaining_length := total_length - other_length
  remaining_length / remaining_strings = 70 := by
sorry

end NUMINAMATH_CALUDE_string_length_problem_l2064_206404


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l2064_206431

theorem salary_increase_percentage 
  (original_salary : ℝ) 
  (current_salary : ℝ) 
  (decrease_percentage : ℝ) 
  (increase_percentage : ℝ) :
  original_salary = 2000 →
  current_salary = 2090 →
  decrease_percentage = 5 →
  current_salary = (1 - decrease_percentage / 100) * (original_salary * (1 + increase_percentage / 100)) →
  increase_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l2064_206431


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l2064_206463

theorem part_to_whole_ratio (N : ℝ) (part : ℝ) : 
  (1/4 : ℝ) * part * (2/5 : ℝ) * N = 20 →
  (40/100 : ℝ) * N = 240 →
  part / ((2/5 : ℝ) * N) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l2064_206463


namespace NUMINAMATH_CALUDE_launch_vehicle_ratio_l2064_206430

/-- Represents a three-stage cylindrical launch vehicle -/
structure LaunchVehicle where
  l₁ : ℝ  -- Length of the first stage
  l₂ : ℝ  -- Length of the second (middle) stage
  l₃ : ℝ  -- Length of the third stage

/-- The conditions for the launch vehicle -/
def LaunchVehicleConditions (v : LaunchVehicle) : Prop :=
  v.l₁ > 0 ∧ v.l₂ > 0 ∧ v.l₃ > 0 ∧
  v.l₂ = (v.l₁ + v.l₃) / 2 ∧
  v.l₂^3 = (6 / 13) * (v.l₁^3 + v.l₃^3)

/-- The theorem stating the ratio of lengths of first and third stages -/
theorem launch_vehicle_ratio (v : LaunchVehicle) 
  (h : LaunchVehicleConditions v) : v.l₁ / v.l₃ = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_launch_vehicle_ratio_l2064_206430


namespace NUMINAMATH_CALUDE_tangent_line_at_1_1_l2064_206475

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem tangent_line_at_1_1 :
  let point : ℝ × ℝ := (1, 1)
  let slope : ℝ := f' point.1
  let tangent_line (x : ℝ) : ℝ := slope * (x - point.1) + point.2
  ∀ x, tangent_line x = -3 * x + 4 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_at_1_1_l2064_206475


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2064_206419

-- Define the condition for a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  (k - 2) * (k - 6) < 0

-- Define the condition given in the problem
def condition (k : ℝ) : Prop :=
  1 < k ∧ k < 7

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ k, is_hyperbola k → condition k) ∧
  (∃ k, condition k ∧ ¬is_hyperbola k) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2064_206419


namespace NUMINAMATH_CALUDE_earthquake_aid_calculation_l2064_206442

/-- Calculates the total financial aid for a school with high school and junior high students -/
def total_financial_aid (total_students : ℕ) (hs_rate : ℕ) (jhs_rate : ℕ) (hs_exclusion_rate : ℚ) : ℕ :=
  651700

/-- The total financial aid for the given school conditions is 651,700 yuan -/
theorem earthquake_aid_calculation :
  let total_students : ℕ := 1862
  let hs_rate : ℕ := 500
  let jhs_rate : ℕ := 350
  let hs_exclusion_rate : ℚ := 30 / 100
  total_financial_aid total_students hs_rate jhs_rate hs_exclusion_rate = 651700 := by
  sorry

end NUMINAMATH_CALUDE_earthquake_aid_calculation_l2064_206442


namespace NUMINAMATH_CALUDE_inclination_angle_range_l2064_206484

/-- A line passing through a point -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- A line segment defined by two endpoints -/
structure LineSegment where
  pointA : ℝ × ℝ
  pointB : ℝ × ℝ

/-- Checks if a line intersects a line segment -/
def intersects (l : Line) (seg : LineSegment) : Prop := sorry

/-- The inclination angle of a line -/
def inclinationAngle (l : Line) : ℝ := sorry

/-- The theorem statement -/
theorem inclination_angle_range 
  (l : Line) 
  (seg : LineSegment) :
  l.point = (0, -2) →
  seg.pointA = (1, -1) →
  seg.pointB = (2, -4) →
  intersects l seg →
  let α := inclinationAngle l
  (0 ≤ α ∧ α ≤ Real.pi / 4) ∨ (3 * Real.pi / 4 ≤ α ∧ α < Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_inclination_angle_range_l2064_206484


namespace NUMINAMATH_CALUDE_divisibility_implies_inequality_l2064_206425

theorem divisibility_implies_inequality (a k : ℕ+) 
  (h : (a^2 + k) ∣ ((a - 1) * a * (a + 1))) : 
  k ≥ a := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_inequality_l2064_206425


namespace NUMINAMATH_CALUDE_floor_sqrt_63_l2064_206422

theorem floor_sqrt_63 : ⌊Real.sqrt 63⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_63_l2064_206422


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2064_206492

theorem parabola_line_intersection (α : Real) : 
  (∃! x, 3 * x^2 + 1 = 4 * Real.sin α * x) → α = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2064_206492


namespace NUMINAMATH_CALUDE_reciprocal_contraction_l2064_206448

open Real

theorem reciprocal_contraction {x₁ x₂ : ℝ} (h₁ : 1 < x₁) (h₂ : x₁ < 2) (h₃ : 1 < x₂) (h₄ : x₂ < 2) (h₅ : x₁ ≠ x₂) :
  |1 / x₁ - 1 / x₂| < |x₂ - x₁| := by
sorry

end NUMINAMATH_CALUDE_reciprocal_contraction_l2064_206448


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2064_206408

theorem sum_of_a_and_b (a b : ℝ) (h1 : a * b > 0) (h2 : |a| = 2) (h3 : |b| = 7) :
  a + b = 9 ∨ a + b = -9 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2064_206408


namespace NUMINAMATH_CALUDE_g_at_neg_three_l2064_206401

-- Define the property of g
def satisfies_equation (g : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = x^3

-- State the theorem
theorem g_at_neg_three (g : ℚ → ℚ) (h : satisfies_equation g) : g (-3) = -6565 / 189 := by
  sorry

end NUMINAMATH_CALUDE_g_at_neg_three_l2064_206401


namespace NUMINAMATH_CALUDE_yola_past_weight_l2064_206450

/-- Proves Yola's weight from 2 years ago given current weights and differences -/
theorem yola_past_weight 
  (yola_current : ℕ) 
  (wanda_yola_diff : ℕ) 
  (wanda_yola_past_diff : ℕ) 
  (h1 : yola_current = 220)
  (h2 : wanda_yola_diff = 30)
  (h3 : wanda_yola_past_diff = 80) : 
  yola_current - (wanda_yola_past_diff - wanda_yola_diff) = 170 := by
  sorry

#check yola_past_weight

end NUMINAMATH_CALUDE_yola_past_weight_l2064_206450


namespace NUMINAMATH_CALUDE_min_books_proof_l2064_206440

def scooter_cost : ℕ := 3000
def earning_per_book : ℕ := 15
def transport_cost_per_book : ℕ := 4

def min_books_to_earn_back : ℕ := 273

theorem min_books_proof :
  min_books_to_earn_back = (
    let profit_per_book := earning_per_book - transport_cost_per_book
    (scooter_cost + profit_per_book - 1) / profit_per_book
  ) :=
by sorry

end NUMINAMATH_CALUDE_min_books_proof_l2064_206440


namespace NUMINAMATH_CALUDE_batsman_average_l2064_206479

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℚ) :
  total_innings = 25 →
  last_innings_score = 95 →
  average_increase = 3.5 →
  (∃ (previous_average : ℚ),
    (previous_average * (total_innings - 1) + last_innings_score) / total_innings = 
    previous_average + average_increase) →
  (∃ (final_average : ℚ), final_average = 11) :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_l2064_206479


namespace NUMINAMATH_CALUDE_max_value_theorem_l2064_206467

theorem max_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 3 * x^2 - 2 * x * y + y^2 = 6) :
  ∃ (z : ℝ), z = 9 + 3 * Real.sqrt 3 ∧ 
  ∀ (w : ℝ), w = 3 * x^2 + 2 * x * y + y^2 → w ≤ z :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2064_206467


namespace NUMINAMATH_CALUDE_triangle_parallel_lines_l2064_206470

theorem triangle_parallel_lines (base : ℝ) (h1 : base = 20) : 
  ∀ (line1 line2 : ℝ),
    (line1 / base)^2 = 1/4 →
    (line2 / line1)^2 = 1/3 →
    line2 = 10 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_parallel_lines_l2064_206470


namespace NUMINAMATH_CALUDE_key_chain_manufacturing_cost_l2064_206429

theorem key_chain_manufacturing_cost 
  (P : ℝ) -- Selling price
  (old_profit_percentage : ℝ) -- Old profit percentage
  (new_profit_percentage : ℝ) -- New profit percentage
  (new_manufacturing_cost : ℝ) -- New manufacturing cost
  (h1 : old_profit_percentage = 0.4) -- Old profit was 40%
  (h2 : new_profit_percentage = 0.5) -- New profit is 50%
  (h3 : new_manufacturing_cost = 50) -- New manufacturing cost is $50
  (h4 : P = new_manufacturing_cost / (1 - new_profit_percentage)) -- Selling price calculation
  : (1 - old_profit_percentage) * P = 60 := by
  sorry


end NUMINAMATH_CALUDE_key_chain_manufacturing_cost_l2064_206429


namespace NUMINAMATH_CALUDE_wednesday_rainfall_l2064_206490

/-- Rainfall recorded over three days -/
def total_rainfall : ℝ := 0.67

/-- Rainfall recorded on Monday -/
def monday_rainfall : ℝ := 0.17

/-- Rainfall recorded on Tuesday -/
def tuesday_rainfall : ℝ := 0.42

/-- Theorem stating that the rainfall on Wednesday is 0.08 cm -/
theorem wednesday_rainfall : 
  total_rainfall - (monday_rainfall + tuesday_rainfall) = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_rainfall_l2064_206490


namespace NUMINAMATH_CALUDE_shaded_fraction_of_square_l2064_206457

theorem shaded_fraction_of_square (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 4 →
  triangle_base = 3 →
  triangle_height = 2 →
  (square_side^2 - 2 * (triangle_base * triangle_height / 2)) / square_side^2 = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_square_l2064_206457


namespace NUMINAMATH_CALUDE_sufficient_condition_problem_l2064_206493

theorem sufficient_condition_problem (p q r s : Prop) 
  (h1 : p → q)
  (h2 : s → q)
  (h3 : q → r)
  (h4 : r → s) :
  p → s := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_problem_l2064_206493


namespace NUMINAMATH_CALUDE_election_votes_l2064_206460

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (62 * total_votes) / 100 - (38 * total_votes) / 100 = 384) :
  (62 * total_votes) / 100 = 992 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l2064_206460


namespace NUMINAMATH_CALUDE_g20_asia_members_l2064_206433

/-- Represents the continents in the G20 --/
inductive Continent
  | Asia
  | Europe
  | Africa
  | Oceania
  | America

/-- Structure representing the G20 membership distribution --/
structure G20 where
  members : Continent → ℕ
  total_twenty : (members Continent.Asia + members Continent.Europe + members Continent.Africa + 
                  members Continent.Oceania + members Continent.America) = 20
  asia_highest : ∀ c : Continent, members Continent.Asia ≥ members c
  africa_oceania_least : members Continent.Africa = members Continent.Oceania ∧ 
                         ∀ c : Continent, members c ≥ members Continent.Africa
  consecutive : ∃ x : ℕ, members Continent.America = x ∧ 
                         members Continent.Europe = x + 1 ∧ 
                         members Continent.Asia = x + 2

theorem g20_asia_members (g : G20) : g.members Continent.Asia = 7 := by
  sorry

end NUMINAMATH_CALUDE_g20_asia_members_l2064_206433


namespace NUMINAMATH_CALUDE_blind_box_probabilities_l2064_206426

def total_boxes : ℕ := 7
def rabbit_boxes : ℕ := 4
def dog_boxes : ℕ := 3

theorem blind_box_probabilities :
  (∀ (n m : ℕ), n + m = total_boxes → n = rabbit_boxes → m = dog_boxes →
    (Nat.choose rabbit_boxes 1 * Nat.choose (total_boxes - 1) 1 ≠ 0 →
      (Nat.choose rabbit_boxes 1 * Nat.choose (rabbit_boxes - 1) 1 : ℚ) /
      (Nat.choose rabbit_boxes 1 * Nat.choose (total_boxes - 1) 1 : ℚ) = 1 / 2)) ∧
  (∀ (n m : ℕ), n + m = total_boxes → n = rabbit_boxes → m = dog_boxes →
    (Nat.choose total_boxes 1 ≠ 0 →
      (Nat.choose dog_boxes 1 : ℚ) / (Nat.choose total_boxes 1 : ℚ) = 3 / 7)) :=
sorry

end NUMINAMATH_CALUDE_blind_box_probabilities_l2064_206426


namespace NUMINAMATH_CALUDE_path_area_calculation_l2064_206411

/-- Calculates the area of a path surrounding a rectangular field -/
def pathArea (fieldLength fieldWidth pathWidth : ℝ) : ℝ :=
  let totalLength := fieldLength + 2 * pathWidth
  let totalWidth := fieldWidth + 2 * pathWidth
  totalLength * totalWidth - fieldLength * fieldWidth

theorem path_area_calculation :
  let fieldLength : ℝ := 75
  let fieldWidth : ℝ := 55
  let pathWidth : ℝ := 2.5
  pathArea fieldLength fieldWidth pathWidth = 675 := by
  sorry

end NUMINAMATH_CALUDE_path_area_calculation_l2064_206411


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2064_206449

theorem arithmetic_mean_of_special_set (n : ℕ) (hn : n > 2) : 
  let set := [1 - 1 / n, 1 + 1 / n] ++ List.replicate (n - 2) 1
  (List.sum set) / n = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2064_206449


namespace NUMINAMATH_CALUDE_f_continuous_at_1_l2064_206474

def f (x : ℝ) : ℝ := -4 * x^2 - 6

theorem f_continuous_at_1 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 1| < δ → |f x - f 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_f_continuous_at_1_l2064_206474


namespace NUMINAMATH_CALUDE_biff_kenneth_race_l2064_206446

/-- Biff and Kenneth's rowboat race problem -/
theorem biff_kenneth_race (race_distance : ℝ) (kenneth_speed : ℝ) (kenneth_extra_distance : ℝ) :
  race_distance = 500 →
  kenneth_speed = 51 →
  kenneth_extra_distance = 10 →
  ∃ (biff_speed : ℝ),
    biff_speed = 50 ∧
    biff_speed * (race_distance + kenneth_extra_distance) / kenneth_speed = race_distance :=
by sorry

end NUMINAMATH_CALUDE_biff_kenneth_race_l2064_206446


namespace NUMINAMATH_CALUDE_parabola_coefficients_l2064_206489

/-- A parabola with equation y = ax^2 + bx + c, vertex at (4, 5), and passing through (2, 3) has coefficients (a, b, c) = (-1/2, 4, -3) -/
theorem parabola_coefficients :
  ∀ (a b c : ℝ),
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (5 : ℝ) = a * 4^2 + b * 4 + c →
  (∀ x : ℝ, a * (x - 4)^2 + 5 = a * x^2 + b * x + c) →
  (3 : ℝ) = a * 2^2 + b * 2 + c →
  (a = -1/2 ∧ b = 4 ∧ c = -3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l2064_206489


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l2064_206438

theorem boys_neither_happy_nor_sad (total_children total_boys total_girls happy_children sad_children neither_children happy_boys sad_girls : ℕ) : 
  total_children = 60 →
  total_boys = 16 →
  total_girls = 44 →
  happy_children = 30 →
  sad_children = 10 →
  neither_children = 20 →
  happy_boys = 6 →
  sad_girls = 4 →
  total_children = total_boys + total_girls →
  happy_children + sad_children + neither_children = total_children →
  (total_boys - happy_boys - (sad_children - sad_girls) = 4) :=
by sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l2064_206438


namespace NUMINAMATH_CALUDE_prism_properties_l2064_206427

/-- Represents a prism with n sides in its base. -/
structure Prism (n : ℕ) where
  base_sides : n ≥ 3

/-- Properties of a prism. -/
def Prism.properties (p : Prism n) : Prop :=
  let lateral_faces := n
  let lateral_edges := n
  let total_edges := 3 * n
  let total_faces := n + 2
  let total_vertices := 2 * n
  lateral_faces = lateral_edges ∧
  total_edges % 3 = 0 ∧
  (n ≥ 4 → Even total_faces) ∧
  Even total_vertices

/-- Theorem stating the properties of a prism. -/
theorem prism_properties (n : ℕ) (p : Prism n) : p.properties := by
  sorry

end NUMINAMATH_CALUDE_prism_properties_l2064_206427


namespace NUMINAMATH_CALUDE_stratified_sampling_survey_l2064_206477

theorem stratified_sampling_survey (total_households : ℕ) 
                                   (middle_income : ℕ) 
                                   (low_income : ℕ) 
                                   (high_income_selected : ℕ) : 
  total_households = 480 →
  middle_income = 200 →
  low_income = 160 →
  high_income_selected = 6 →
  ∃ (total_selected : ℕ), 
    total_selected * (total_households - middle_income - low_income) = 
    high_income_selected * total_households ∧
    total_selected = 24 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_survey_l2064_206477


namespace NUMINAMATH_CALUDE_not_prime_two_pow_plus_one_l2064_206488

theorem not_prime_two_pow_plus_one (n m : ℕ) (h1 : m > 1) (h2 : Odd m) (h3 : m ∣ n) :
  ¬ Prime (2^n + 1) := by
sorry

end NUMINAMATH_CALUDE_not_prime_two_pow_plus_one_l2064_206488


namespace NUMINAMATH_CALUDE_expression_simplification_l2064_206421

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (x + 3)^2 + (x + 2)*(x - 2) - x*(x + 6) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2064_206421


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_even_integers_l2064_206441

/-- Sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of the first n positive even integers -/
def sum_first_n_even (n : ℕ) : ℕ := 2 * sum_first_n n

/-- Sum of five consecutive even integers -/
def sum_five_consecutive_even (n : ℕ) : ℕ := 5 * n - 20

theorem largest_of_five_consecutive_even_integers :
  ∃ (n : ℕ), sum_five_consecutive_even n = sum_first_n_even 30 ∧
             n = 190 ∧
             ∀ (m : ℕ), sum_five_consecutive_even m = sum_first_n_even 30 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_even_integers_l2064_206441


namespace NUMINAMATH_CALUDE_delta_nabla_equality_l2064_206434

/-- Definition of the Δ operation -/
def delta (a b : ℕ) : ℕ := 3 * a + 2 * b

/-- Definition of the ∇ operation -/
def nabla (a b : ℕ) : ℕ := 2 * a + 3 * b

/-- Theorem stating that 3 Δ (2 ∇ 1) = 23 -/
theorem delta_nabla_equality : delta 3 (nabla 2 1) = 23 := by
  sorry

end NUMINAMATH_CALUDE_delta_nabla_equality_l2064_206434


namespace NUMINAMATH_CALUDE_middle_card_is_six_l2064_206418

def is_valid_set (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b + c = 20 ∧ a % 2 = 0 ∧ c % 2 = 0

def possible_after_aria (a b c : ℕ) : Prop :=
  is_valid_set a b c ∧ a ≠ 6

def possible_after_cece (a b c : ℕ) : Prop :=
  possible_after_aria a b c ∧ c ≠ 13

def possible_after_bruce (a b c : ℕ) : Prop :=
  possible_after_cece a b c ∧ (b ≠ 5 ∨ a ≠ 4)

theorem middle_card_is_six :
  ∀ a b c : ℕ, possible_after_bruce a b c → b = 6 :=
sorry

end NUMINAMATH_CALUDE_middle_card_is_six_l2064_206418


namespace NUMINAMATH_CALUDE_koi_fish_count_l2064_206498

theorem koi_fish_count : ∃ k : ℕ, (2 * k - 14 = 64) ∧ (k = 39) := by
  sorry

end NUMINAMATH_CALUDE_koi_fish_count_l2064_206498


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2064_206461

theorem weight_of_replaced_person 
  (n : ℕ) 
  (avg_increase : ℝ) 
  (new_person_weight : ℝ) 
  (h1 : n = 8)
  (h2 : avg_increase = 2.5)
  (h3 : new_person_weight = 60) :
  ∃ (replaced_weight : ℝ), replaced_weight = new_person_weight - n * avg_increase :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2064_206461


namespace NUMINAMATH_CALUDE_fifth_month_sale_is_2560_l2064_206410

/-- Calculates the sale in the fifth month given the sales of the first four months,
    the average sale over six months, and the sale in the sixth month. -/
def fifth_month_sale (sale1 sale2 sale3 sale4 average_sale sixth_month_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale3 + sale4 + sixth_month_sale)

/-- Proves that the sale in the fifth month is 2560 given the specified conditions. -/
theorem fifth_month_sale_is_2560 :
  fifth_month_sale 2435 2920 2855 3230 2500 1000 = 2560 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_is_2560_l2064_206410


namespace NUMINAMATH_CALUDE_solution_set_when_m_3_range_of_m_for_nonnegative_f_l2064_206423

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m - 1

-- Statement for Question 1
theorem solution_set_when_m_3 :
  {x : ℝ | f 3 x ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Statement for Question 2
theorem range_of_m_for_nonnegative_f :
  ∀ m : ℝ, (∀ x ∈ Set.Icc 2 4, f m x ≥ -1) ↔ m ≤ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_3_range_of_m_for_nonnegative_f_l2064_206423


namespace NUMINAMATH_CALUDE_prob_three_even_out_of_six_l2064_206482

/-- A fair 20-sided die -/
def Die : Type := Fin 20

/-- The probability of a single die showing an even number -/
def prob_even : ℚ := 1/2

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of dice we want to show even numbers -/
def target_even : ℕ := 3

/-- The probability of exactly three out of six fair 20-sided dice showing an even number -/
theorem prob_three_even_out_of_six : 
  (Nat.choose num_dice target_even : ℚ) * prob_even^target_even * (1 - prob_even)^(num_dice - target_even) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_out_of_six_l2064_206482


namespace NUMINAMATH_CALUDE_equation_solutions_l2064_206406

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 = 4 ↔ x = -1 ∨ x = 3) ∧
  (∀ x : ℝ, x^2 + 3*x - 4 = 0 ↔ x = -4 ∨ x = 1) ∧
  (∀ x : ℝ, 4*x*(2*x + 1) = 3*(2*x + 1) ↔ x = -1/2 ∨ x = 3/4) ∧
  (∀ x : ℝ, 2*x^2 + 5*x - 3 = 0 ↔ x = 1/2 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2064_206406


namespace NUMINAMATH_CALUDE_problem_solution_l2064_206471

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | (x-1)*(x-a+1) = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

theorem problem_solution (a m : ℝ) 
  (h1 : A ∪ B a = A) 
  (h2 : A ∩ C m = C m) : 
  (a = 2 ∨ a = 3) ∧ 
  (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2064_206471


namespace NUMINAMATH_CALUDE_probability_qualified_product_l2064_206409

/-- The probability of buying a qualified product from a market with two factories -/
theorem probability_qualified_product
  (factory_a_share : ℝ)
  (factory_b_share : ℝ)
  (factory_a_qualification_rate : ℝ)
  (factory_b_qualification_rate : ℝ)
  (h1 : factory_a_share = 0.6)
  (h2 : factory_b_share = 0.4)
  (h3 : factory_a_qualification_rate = 0.95)
  (h4 : factory_b_qualification_rate = 0.9)
  (h5 : factory_a_share + factory_b_share = 1) :
  factory_a_share * factory_a_qualification_rate +
  factory_b_share * factory_b_qualification_rate = 0.93 :=
by sorry

end NUMINAMATH_CALUDE_probability_qualified_product_l2064_206409


namespace NUMINAMATH_CALUDE_slope_product_sufficient_not_necessary_l2064_206485

/-- A line in a 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Two lines are perpendicular -/
def perpendicular (l₁ l₂ : Line) : Prop :=
  sorry

/-- The product of slopes of two lines is -1 -/
def slope_product_negative_one (l₁ l₂ : Line) : Prop :=
  l₁.slope * l₂.slope = -1

/-- The product of slopes being -1 is sufficient but not necessary for perpendicularity -/
theorem slope_product_sufficient_not_necessary :
  (∀ l₁ l₂ : Line, slope_product_negative_one l₁ l₂ → perpendicular l₁ l₂) ∧
  ¬(∀ l₁ l₂ : Line, perpendicular l₁ l₂ → slope_product_negative_one l₁ l₂) :=
sorry

end NUMINAMATH_CALUDE_slope_product_sufficient_not_necessary_l2064_206485


namespace NUMINAMATH_CALUDE_part_one_part_two_l2064_206483

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem for part 1 of the problem -/
theorem part_one (t : Triangle) (h : 2 * t.a * Real.sin t.B = Real.sqrt 3 * t.b) :
  t.A = Real.pi / 3 ∨ t.A = 2 * Real.pi / 3 :=
sorry

/-- Theorem for part 2 of the problem -/
theorem part_two (t : Triangle) (h : t.a / 2 = t.b * Real.sin t.A) :
  (∀ x : Triangle, x.c / x.b + x.b / x.c ≤ 2 * Real.sqrt 2) ∧
  (∃ x : Triangle, x.c / x.b + x.b / x.c = 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2064_206483


namespace NUMINAMATH_CALUDE_probability_three_green_marbles_l2064_206497

theorem probability_three_green_marbles : 
  let total_marbles : ℕ := 15
  let green_marbles : ℕ := 8
  let purple_marbles : ℕ := 7
  let total_trials : ℕ := 7
  let green_trials : ℕ := 3
  
  let prob_green : ℚ := green_marbles / total_marbles
  let prob_purple : ℚ := purple_marbles / total_marbles
  
  let ways_to_choose_green : ℕ := Nat.choose total_trials green_trials
  let prob_specific_outcome : ℚ := prob_green ^ green_trials * prob_purple ^ (total_trials - green_trials)
  
  ways_to_choose_green * prob_specific_outcome = 43079680 / 170859375 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_three_green_marbles_l2064_206497


namespace NUMINAMATH_CALUDE_monotonicity_not_algorithmic_l2064_206453

-- Define the concept of an algorithm
def Algorithm : Type := Unit

-- Define the problems
def SumProblem : Type := Unit
def LinearSystemProblem : Type := Unit
def CircleAreaProblem : Type := Unit
def MonotonicityProblem : Type := Unit

-- Define solvability by algorithm
def SolvableByAlgorithm (p : Type) : Prop := ∃ (a : Algorithm), True

-- State the theorem
theorem monotonicity_not_algorithmic :
  SolvableByAlgorithm SumProblem ∧
  SolvableByAlgorithm LinearSystemProblem ∧
  SolvableByAlgorithm CircleAreaProblem ∧
  ¬SolvableByAlgorithm MonotonicityProblem :=
sorry

end NUMINAMATH_CALUDE_monotonicity_not_algorithmic_l2064_206453


namespace NUMINAMATH_CALUDE_inequality_solution_l2064_206456

theorem inequality_solution (x : ℝ) (h : x ≠ -1) :
  (x - 2) / (x + 1) ≤ 2 ↔ x ≤ -4 ∨ x > -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2064_206456


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2064_206486

/-- Given a circle with equation x^2 - 16x + y^2 + 6y = -75, 
    prove that the sum of its center coordinates and radius is 5 + √2 -/
theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), 
    (∀ x y : ℝ, x^2 - 16*x + y^2 + 6*y = -75 ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
    a + b + r = 5 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2064_206486


namespace NUMINAMATH_CALUDE_regular_pentagon_angle_l2064_206443

theorem regular_pentagon_angle (n : ℕ) (h : n = 5) :
  let central_angle := 360 / n
  2 * central_angle = 144 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_angle_l2064_206443


namespace NUMINAMATH_CALUDE_largest_value_l2064_206447

theorem largest_value : 
  (4^2 : ℝ) ≥ 4 * 2 ∧ 
  (4^2 : ℝ) ≥ 4 - 2 ∧ 
  (4^2 : ℝ) ≥ 4 / 2 ∧ 
  (4^2 : ℝ) ≥ 4 + 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l2064_206447

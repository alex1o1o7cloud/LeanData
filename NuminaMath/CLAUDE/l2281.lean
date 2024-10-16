import Mathlib

namespace NUMINAMATH_CALUDE_daniels_horses_l2281_228167

/-- The number of horses Daniel has -/
def num_horses : ℕ := 2

/-- The number of dogs Daniel has -/
def num_dogs : ℕ := 5

/-- The number of cats Daniel has -/
def num_cats : ℕ := 7

/-- The number of turtles Daniel has -/
def num_turtles : ℕ := 3

/-- The number of goats Daniel has -/
def num_goats : ℕ := 1

/-- The number of legs each animal has -/
def legs_per_animal : ℕ := 4

/-- The total number of legs of all animals -/
def total_legs : ℕ := 72

theorem daniels_horses :
  num_horses * legs_per_animal +
  num_dogs * legs_per_animal +
  num_cats * legs_per_animal +
  num_turtles * legs_per_animal +
  num_goats * legs_per_animal = total_legs :=
by sorry

end NUMINAMATH_CALUDE_daniels_horses_l2281_228167


namespace NUMINAMATH_CALUDE_first_term_of_specific_sequence_l2281_228172

/-- A geometric sequence is defined by its fifth and sixth terms -/
structure GeometricSequence where
  fifth_term : ℚ
  sixth_term : ℚ

/-- The first term of a geometric sequence -/
def first_term (seq : GeometricSequence) : ℚ :=
  256 / 27

/-- Theorem: Given a geometric sequence where the fifth term is 48 and the sixth term is 72, 
    the first term is 256/27 -/
theorem first_term_of_specific_sequence :
  ∀ (seq : GeometricSequence), 
    seq.fifth_term = 48 ∧ seq.sixth_term = 72 → first_term seq = 256 / 27 :=
by
  sorry

end NUMINAMATH_CALUDE_first_term_of_specific_sequence_l2281_228172


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2281_228114

theorem inscribed_cube_surface_area (V : ℝ) (h : V = 256 * Real.pi / 3) :
  let R := (3 * V / (4 * Real.pi)) ^ (1/3)
  let a := 2 * R / Real.sqrt 3
  6 * a^2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2281_228114


namespace NUMINAMATH_CALUDE_kenny_basketball_time_l2281_228126

def trumpet_practice : ℕ := 40

theorem kenny_basketball_time (run_time trumpet_time basketball_time : ℕ) 
  (h1 : trumpet_time = trumpet_practice)
  (h2 : trumpet_time = 2 * run_time)
  (h3 : run_time = 2 * basketball_time) : 
  basketball_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_kenny_basketball_time_l2281_228126


namespace NUMINAMATH_CALUDE_state_university_cost_l2281_228146

theorem state_university_cost (tuition room_and_board total_cost : ℕ) : 
  tuition = 1644 →
  tuition = room_and_board + 704 →
  total_cost = tuition + room_and_board →
  total_cost = 2584 := by
  sorry

end NUMINAMATH_CALUDE_state_university_cost_l2281_228146


namespace NUMINAMATH_CALUDE_largest_b_value_l2281_228177

theorem largest_b_value (b : ℚ) (h : (3*b+4)*(b-2) = 9*b) : 
  ∃ (max_b : ℚ), max_b = 4 ∧ ∀ (x : ℚ), (3*x+4)*(x-2) = 9*x → x ≤ max_b :=
by sorry

end NUMINAMATH_CALUDE_largest_b_value_l2281_228177


namespace NUMINAMATH_CALUDE_sum_of_roots_l2281_228181

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a = 1) 
  (hb : b^3 - 3*b^2 + 5*b = 5) : 
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2281_228181


namespace NUMINAMATH_CALUDE_mark_radiator_cost_l2281_228130

/-- The total cost Mark paid for replacing his car radiator -/
def total_cost (labor_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  labor_hours * hourly_rate + part_cost

/-- Theorem stating that Mark paid $300 for replacing his car radiator -/
theorem mark_radiator_cost :
  total_cost 2 75 150 = 300 := by
  sorry

end NUMINAMATH_CALUDE_mark_radiator_cost_l2281_228130


namespace NUMINAMATH_CALUDE_vasya_no_purchase_days_l2281_228102

theorem vasya_no_purchase_days :
  ∀ (x y z w : ℕ),
    x + y + z + w = 15 →  -- Total school days
    9 * x + 4 * z = 30 →  -- Total marshmallows bought
    2 * y + z = 9 →       -- Total meat pies bought
    w = 7 :=              -- Days with no purchase
by
  sorry

end NUMINAMATH_CALUDE_vasya_no_purchase_days_l2281_228102


namespace NUMINAMATH_CALUDE_four_line_theorem_l2281_228104

-- Define the type for lines in space
variable (Line : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem four_line_theorem (a b c d : Line) 
  (h1 : perp a b) (h2 : perp b c) (h3 : perp c d) (h4 : perp d a) :
  para b d ∨ para a c :=
sorry

end NUMINAMATH_CALUDE_four_line_theorem_l2281_228104


namespace NUMINAMATH_CALUDE_apple_bags_theorem_l2281_228108

theorem apple_bags_theorem (A B C : ℕ) 
  (h1 : A + B = 11) 
  (h2 : B + C = 18) 
  (h3 : A + C = 19) : 
  A + B + C = 24 := by
sorry

end NUMINAMATH_CALUDE_apple_bags_theorem_l2281_228108


namespace NUMINAMATH_CALUDE_second_boy_speed_l2281_228129

/-- Given two boys walking in the same direction for 16 hours, with one boy walking at 5.5 kmph
    and ending up 32 km apart, prove that the speed of the second boy is 7.5 kmph. -/
theorem second_boy_speed (first_speed : ℝ) (time : ℝ) (distance : ℝ) (second_speed : ℝ) :
  first_speed = 5.5 →
  time = 16 →
  distance = 32 →
  distance = (second_speed - first_speed) * time →
  second_speed = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_second_boy_speed_l2281_228129


namespace NUMINAMATH_CALUDE_man_speed_against_stream_l2281_228155

/-- Calculates the speed against the stream given the rate in still water and speed with the stream -/
def speed_against_stream (rate_still : ℝ) (speed_with_stream : ℝ) : ℝ :=
  |rate_still - (speed_with_stream - rate_still)|

/-- Theorem: Given a man's rate in still water of 2 km/h and speed with the stream of 6 km/h,
    his speed against the stream is 2 km/h -/
theorem man_speed_against_stream :
  speed_against_stream 2 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_against_stream_l2281_228155


namespace NUMINAMATH_CALUDE_parallel_vectors_characterization_l2281_228128

/-- Two vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  x₁ * y₂ - x₂ * y₁ = 0

/-- The proposed condition for parallel vectors -/
def proposed_condition (a b : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  x₁ * y₂ = x₂ * y₁

theorem parallel_vectors_characterization (a b : ℝ × ℝ) :
  (are_parallel a b ↔ proposed_condition a b) ∧
  (∃ a b : ℝ × ℝ, are_parallel a b ≠ proposed_condition a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_characterization_l2281_228128


namespace NUMINAMATH_CALUDE_phone_bill_increase_l2281_228194

theorem phone_bill_increase (original_bill : ℝ) (increase_percent : ℝ) (months : ℕ) : 
  original_bill = 50 ∧ 
  increase_percent = 10 ∧ 
  months = 12 → 
  original_bill * (1 + increase_percent / 100) * months = 660 := by
  sorry

end NUMINAMATH_CALUDE_phone_bill_increase_l2281_228194


namespace NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_fifth_root_32_l2281_228198

theorem cube_root_125_times_fourth_root_256_times_fifth_root_32 :
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (32 : ℝ) ^ (1/5) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_fifth_root_32_l2281_228198


namespace NUMINAMATH_CALUDE_banana_purchase_l2281_228196

theorem banana_purchase (banana_price apple_price total_weight total_cost : ℚ)
  (h1 : banana_price = 76 / 100)
  (h2 : apple_price = 59 / 100)
  (h3 : total_weight = 30)
  (h4 : total_cost = 1940 / 100) :
  ∃ (banana_weight : ℚ),
    banana_weight + (total_weight - banana_weight) = total_weight ∧
    banana_price * banana_weight + apple_price * (total_weight - banana_weight) = total_cost ∧
    banana_weight = 10 := by
sorry

end NUMINAMATH_CALUDE_banana_purchase_l2281_228196


namespace NUMINAMATH_CALUDE_simplify_expression_l2281_228127

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 8) - (x + 6)*(3*x - 2) = 4*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2281_228127


namespace NUMINAMATH_CALUDE_quadratic_inequality_max_value_l2281_228142

theorem quadratic_inequality_max_value (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) →
  (∃ M, M = -4 ∧ ∀ a' b' c', (∀ x, a'*x^2 + b'*x + c' > 0 ↔ -1 < x ∧ x < 2) → b' - c' + 4/a' ≤ M) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_max_value_l2281_228142


namespace NUMINAMATH_CALUDE_sum_segment_lengths_equals_78_l2281_228133

/-- Triangle with vertices A, B, C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Sum of lengths of segments cut by horizontal integer lines -/
def sumSegmentLengths (t : Triangle) : ℝ :=
  sorry

/-- The specific triangle in the problem -/
def problemTriangle : Triangle :=
  { A := (1, 3.5),
    B := (13.5, 3.5),
    C := (11, 16) }

theorem sum_segment_lengths_equals_78 :
  sumSegmentLengths problemTriangle = 78 :=
sorry

end NUMINAMATH_CALUDE_sum_segment_lengths_equals_78_l2281_228133


namespace NUMINAMATH_CALUDE_paper_envelope_problem_l2281_228175

/-- 
Given that each paper envelope can contain 10 white papers and 12 paper envelopes are needed,
prove that the total number of clean white papers is 120.
-/
theorem paper_envelope_problem (papers_per_envelope : ℕ) (num_envelopes : ℕ) 
  (h1 : papers_per_envelope = 10) 
  (h2 : num_envelopes = 12) : 
  papers_per_envelope * num_envelopes = 120 := by
  sorry

end NUMINAMATH_CALUDE_paper_envelope_problem_l2281_228175


namespace NUMINAMATH_CALUDE_merill_marbles_vivian_marbles_l2281_228169

-- Define the number of marbles for each person
def Selma : ℕ := 50
def Elliot : ℕ := 15
def Merill : ℕ := 2 * Elliot
def Vivian : ℕ := 21

-- Theorem to prove Merill's marbles
theorem merill_marbles : Merill = 30 := by sorry

-- Theorem to prove Vivian's marbles
theorem vivian_marbles : Vivian = 21 ∧ Vivian > Elliot + 5 ∧ Vivian ≥ (135 * Elliot) / 100 := by sorry

end NUMINAMATH_CALUDE_merill_marbles_vivian_marbles_l2281_228169


namespace NUMINAMATH_CALUDE_jacob_insects_compared_to_dean_l2281_228119

theorem jacob_insects_compared_to_dean :
  ∀ (angela_insects jacob_insects dean_insects : ℕ),
    angela_insects = 75 →
    dean_insects = 30 →
    angela_insects * 2 = jacob_insects →
    jacob_insects / dean_insects = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_jacob_insects_compared_to_dean_l2281_228119


namespace NUMINAMATH_CALUDE_eat_cereal_together_l2281_228153

/-- The time taken for two people to eat a certain amount of cereal together -/
def time_to_eat_together (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- Theorem: Mr. Fat and Mr. Thin will take 96 minutes to eat 4 pounds of cereal together -/
theorem eat_cereal_together : 
  let fat_rate : ℚ := 1 / 40
  let thin_rate : ℚ := 1 / 15
  let amount : ℚ := 4
  time_to_eat_together fat_rate thin_rate amount = 96 := by
  sorry

#eval time_to_eat_together (1 / 40) (1 / 15) 4

end NUMINAMATH_CALUDE_eat_cereal_together_l2281_228153


namespace NUMINAMATH_CALUDE_horner_method_v3_horner_method_correctness_l2281_228139

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v3 (x : ℝ) : ℝ :=
  let v1 := 3 * x + 5
  let v2 := v1 * x + 6
  v2 * x + 79

theorem horner_method_v3 :
  horner_v3 (-4) = -57 :=
by sorry

theorem horner_method_correctness :
  horner_v3 (-4) = horner_polynomial (-4) :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_horner_method_correctness_l2281_228139


namespace NUMINAMATH_CALUDE_paper_stack_height_l2281_228109

/-- Given a ream of paper with known thickness and sheet count, 
    calculate the number of sheets in a stack of a different height -/
theorem paper_stack_height (ream_sheets : ℕ) (ream_thickness : ℝ) (stack_height : ℝ) :
  ream_sheets > 0 →
  ream_thickness > 0 →
  stack_height > 0 →
  ream_sheets * (stack_height / ream_thickness) = 900 :=
by
  -- Assuming ream_sheets = 400, ream_thickness = 4, and stack_height = 9
  sorry

#check paper_stack_height 400 4 9

end NUMINAMATH_CALUDE_paper_stack_height_l2281_228109


namespace NUMINAMATH_CALUDE_henry_collection_cost_l2281_228182

/-- The amount of money Henry needs to finish his action figure collection -/
def money_needed (current : ℕ) (total_needed : ℕ) (cost_per_figure : ℕ) : ℕ :=
  (total_needed - current) * cost_per_figure

/-- Proof that Henry needs $30 to finish his collection -/
theorem henry_collection_cost : money_needed 3 8 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_henry_collection_cost_l2281_228182


namespace NUMINAMATH_CALUDE_tian_ji_winning_probability_l2281_228186

/-- Represents the horses of each competitor -/
inductive Horse : Type
  | top : Horse
  | middle : Horse
  | bottom : Horse

/-- Defines the ordering of horses based on their performance -/
def beats (h1 h2 : Horse) : Prop :=
  match h1, h2 with
  | Horse.top, Horse.middle => true
  | Horse.top, Horse.bottom => true
  | Horse.middle, Horse.bottom => true
  | _, _ => false

/-- King Qi's horses -/
def kingQi : Horse → Horse
  | Horse.top => Horse.top
  | Horse.middle => Horse.middle
  | Horse.bottom => Horse.bottom

/-- Tian Ji's horses -/
def tianJi : Horse → Horse
  | Horse.top => Horse.top
  | Horse.middle => Horse.middle
  | Horse.bottom => Horse.bottom

/-- The conditions of the horse performances -/
axiom horse_performance :
  (beats (tianJi Horse.top) (kingQi Horse.middle)) ∧
  (beats (kingQi Horse.top) (tianJi Horse.top)) ∧
  (beats (tianJi Horse.middle) (kingQi Horse.bottom)) ∧
  (beats (kingQi Horse.middle) (tianJi Horse.middle)) ∧
  (beats (kingQi Horse.bottom) (tianJi Horse.bottom))

/-- The probability of Tian Ji's horse winning -/
def winning_probability : ℚ := 1/3

/-- The main theorem to prove -/
theorem tian_ji_winning_probability :
  winning_probability = 1/3 := by sorry

end NUMINAMATH_CALUDE_tian_ji_winning_probability_l2281_228186


namespace NUMINAMATH_CALUDE_tan_product_equals_15_l2281_228135

theorem tan_product_equals_15 : 
  15 * Real.tan (44 * π / 180) * Real.tan (45 * π / 180) * Real.tan (46 * π / 180) = 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_equals_15_l2281_228135


namespace NUMINAMATH_CALUDE_common_external_tangent_y_intercept_value_l2281_228151

/-- The y-intercept of the common external tangent to two circles --/
def common_external_tangent_y_intercept : ℝ := sorry

/-- First circle center --/
def center1 : ℝ × ℝ := (1, 3)

/-- Second circle center --/
def center2 : ℝ × ℝ := (13, 6)

/-- First circle radius --/
def radius1 : ℝ := 3

/-- Second circle radius --/
def radius2 : ℝ := 6

theorem common_external_tangent_y_intercept_value :
  ∃ (m : ℝ), m > 0 ∧ 
  ∀ (x y : ℝ), y = m * x + common_external_tangent_y_intercept →
  ((x - center1.1)^2 + (y - center1.2)^2 = radius1^2 ∨
   (x - center2.1)^2 + (y - center2.2)^2 = radius2^2) →
  ∀ (x' y' : ℝ), (x' - center1.1)^2 + (y' - center1.2)^2 < radius1^2 →
                 (x' - center2.1)^2 + (y' - center2.2)^2 < radius2^2 →
                 y' ≠ m * x' + common_external_tangent_y_intercept := by
  sorry

#check common_external_tangent_y_intercept_value

end NUMINAMATH_CALUDE_common_external_tangent_y_intercept_value_l2281_228151


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2281_228143

theorem sqrt_fraction_simplification : 
  Real.sqrt ((25 : ℝ) / 36 - 4 / 9) = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2281_228143


namespace NUMINAMATH_CALUDE_rally_attendance_l2281_228197

/-- Represents the rally attendance problem --/
def RallyAttendance (total_receipts : ℚ) (before_rally_tickets : ℕ) 
  (before_rally_price : ℚ) (at_door_price : ℚ) : Prop :=
  ∃ (at_door_tickets : ℕ),
    total_receipts = before_rally_price * before_rally_tickets + at_door_price * at_door_tickets ∧
    before_rally_tickets + at_door_tickets = 750

/-- Theorem stating the total attendance at the rally --/
theorem rally_attendance :
  RallyAttendance (1706.25 : ℚ) 475 2 (2.75 : ℚ) :=
by
  sorry


end NUMINAMATH_CALUDE_rally_attendance_l2281_228197


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_169_l2281_228170

theorem greatest_prime_factor_of_169 : ∃ p : ℕ, p.Prime ∧ p ∣ 169 ∧ ∀ q : ℕ, q.Prime → q ∣ 169 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_169_l2281_228170


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l2281_228178

theorem largest_prime_divisor_of_sum_of_squares : ∃ p : ℕ, 
  Nat.Prime p ∧ 
  p ∣ (36^2 + 49^2) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (36^2 + 49^2) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l2281_228178


namespace NUMINAMATH_CALUDE_find_divisor_l2281_228107

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (h1 : dividend = 1254) (h2 : quotient = 209) 
  (h3 : dividend % (dividend / quotient) = 0) : dividend / quotient = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2281_228107


namespace NUMINAMATH_CALUDE_inequality_solution_l2281_228150

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 1) + 10 / (x + 4) ≥ 3 / (x + 2)) ↔ 
  (x ∈ Set.Ioc (-4) (-1) ∪ Set.Ioi (-4/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2281_228150


namespace NUMINAMATH_CALUDE_inscribed_prism_surface_area_l2281_228174

/-- The surface area of a right square prism inscribed in a sphere -/
theorem inscribed_prism_surface_area (r h : ℝ) (hr : r = Real.sqrt 6) (hh : h = 4) :
  let a := Real.sqrt ((r^2 - h^2/4) / 2)
  2 * a^2 + 4 * a * h = 40 :=
sorry

end NUMINAMATH_CALUDE_inscribed_prism_surface_area_l2281_228174


namespace NUMINAMATH_CALUDE_square_product_sequence_max_l2281_228154

/-- A sequence of natural numbers where each pair of consecutive numbers has a perfect square product -/
def SquareProductSequence (a : ℕ → ℕ) : Prop :=
  ∀ n, ∃ k, (a n) * (a (n + 1)) = k^2

theorem square_product_sequence_max (a : ℕ → ℕ) :
  (∀ i j, i ≠ j → a i ≠ a j) →  -- All numbers are different
  (SquareProductSequence a) →   -- Product of consecutive pairs is a perfect square
  (a 0 = 42) →                  -- First number is 42
  (∃ n, n < 20 ∧ a n ≥ 16800) :=  -- At least one of the first 20 numbers is ≥ 16800
by sorry

end NUMINAMATH_CALUDE_square_product_sequence_max_l2281_228154


namespace NUMINAMATH_CALUDE_m_value_in_set_union_l2281_228161

def A (m : ℝ) : Set ℝ := {2, m}
def B (m : ℝ) : Set ℝ := {1, m^2}

theorem m_value_in_set_union (m : ℝ) :
  A m ∪ B m = {1, 2, 3, 9} → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_m_value_in_set_union_l2281_228161


namespace NUMINAMATH_CALUDE_sun_radius_scientific_notation_l2281_228111

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The radius of the Sun in meters -/
def sun_radius : ℝ := 696000000

/-- Converts a real number to scientific notation -/
noncomputable def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem sun_radius_scientific_notation :
  to_scientific_notation sun_radius = ScientificNotation.mk 6.96 8 sorry := by
  sorry

end NUMINAMATH_CALUDE_sun_radius_scientific_notation_l2281_228111


namespace NUMINAMATH_CALUDE_average_tomatoes_proof_l2281_228189

/-- The number of tomatoes reaped on day 1 -/
def day1_tomatoes : ℕ := 120

/-- The number of tomatoes reaped on day 2 -/
def day2_tomatoes : ℕ := day1_tomatoes + 50

/-- The number of tomatoes reaped on day 3 -/
def day3_tomatoes : ℕ := 2 * day2_tomatoes

/-- The number of tomatoes reaped on day 4 -/
def day4_tomatoes : ℕ := day1_tomatoes / 2

/-- The total number of tomatoes reaped over 4 days -/
def total_tomatoes : ℕ := day1_tomatoes + day2_tomatoes + day3_tomatoes + day4_tomatoes

/-- The number of days -/
def num_days : ℕ := 4

/-- The average number of tomatoes reaped per day -/
def average_tomatoes : ℚ := total_tomatoes / num_days

theorem average_tomatoes_proof : average_tomatoes = 172.5 := by
  sorry

end NUMINAMATH_CALUDE_average_tomatoes_proof_l2281_228189


namespace NUMINAMATH_CALUDE_park_population_l2281_228137

/-- Calculates the total population of lions, leopards, and elephants in a park. -/
theorem park_population (num_lions : ℕ) (num_leopards : ℕ) (num_elephants : ℕ) : 
  num_lions = 200 →
  num_lions = 2 * num_leopards →
  num_elephants = (num_lions + num_leopards) / 2 →
  num_lions + num_leopards + num_elephants = 450 := by
  sorry

#check park_population

end NUMINAMATH_CALUDE_park_population_l2281_228137


namespace NUMINAMATH_CALUDE_distance_between_cities_l2281_228180

/-- The distance between City A and City B in miles -/
def distance : ℝ := 427.5

/-- Yesterday's travel time from A to B in hours -/
def yesterday_time : ℝ := 6

/-- Today's travel time from B to A in hours -/
def today_time : ℝ := 4.5

/-- Time saved on each trip in hours -/
def time_saved : ℝ := 0.5

/-- Average speed for the round trip if time were saved, in miles per hour -/
def average_speed : ℝ := 90

theorem distance_between_cities :
  distance = 427.5 ∧
  yesterday_time = 6 ∧
  today_time = 4.5 ∧
  (2 * distance) / (yesterday_time + today_time - 2 * time_saved) = average_speed :=
by sorry

end NUMINAMATH_CALUDE_distance_between_cities_l2281_228180


namespace NUMINAMATH_CALUDE_vanilla_to_cream_cheese_ratio_l2281_228184

-- Define the ratios and quantities
def sugar_to_cream_cheese_ratio : ℚ := 1 / 4
def vanilla_to_eggs_ratio : ℚ := 1 / 2
def sugar_used : ℚ := 2
def eggs_used : ℚ := 8
def teaspoons_per_cup : ℚ := 48

-- Theorem to prove
theorem vanilla_to_cream_cheese_ratio :
  let cream_cheese := sugar_used / sugar_to_cream_cheese_ratio
  let vanilla := eggs_used * vanilla_to_eggs_ratio
  let cream_cheese_teaspoons := cream_cheese * teaspoons_per_cup
  vanilla / cream_cheese_teaspoons = 1 / 96 :=
by sorry

end NUMINAMATH_CALUDE_vanilla_to_cream_cheese_ratio_l2281_228184


namespace NUMINAMATH_CALUDE_equation_solution_l2281_228131

theorem equation_solution : ∃ x : ℕ, (8000 * 6000 : ℕ) = x * (10^5 : ℕ) ∧ x = 480 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2281_228131


namespace NUMINAMATH_CALUDE_average_of_combined_results_l2281_228124

theorem average_of_combined_results (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ) 
  (h₁ : n₁ = 55) (h₂ : n₂ = 28) (h₃ : avg₁ = 28) (h₄ : avg₂ = 55) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℚ) = (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) :=
by sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l2281_228124


namespace NUMINAMATH_CALUDE_prime_power_square_sum_l2281_228145

theorem prime_power_square_sum (p n k : ℕ) : 
  p.Prime → p > 0 → n > 0 → k > 0 → 144 + p^n = k^2 →
  ((p = 5 ∧ n = 2 ∧ k = 13) ∨ (p = 2 ∧ n = 8 ∧ k = 20) ∨ (p = 3 ∧ n = 4 ∧ k = 15)) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_square_sum_l2281_228145


namespace NUMINAMATH_CALUDE_largest_divisor_of_three_consecutive_integers_l2281_228123

theorem largest_divisor_of_three_consecutive_integers :
  ∃ (d : ℕ), d > 0 ∧
  (∀ (n : ℤ), (n * (n + 1) * (n + 2)) % d = 0) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℤ), (m * (m + 1) * (m + 2)) % k ≠ 0) ∧
  d = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_three_consecutive_integers_l2281_228123


namespace NUMINAMATH_CALUDE_zhang_qiujian_problem_l2281_228141

theorem zhang_qiujian_problem (x y : ℤ) : 
  (x + 10 - (y - 10) = 5 * (y - 10) ∧ x - 10 = y + 10) ↔
  (x = y + 10 ∧ 
   x + 10 - (y - 10) = 5 * (y - 10) ∧ 
   x - 10 = y + 10) :=
by sorry

end NUMINAMATH_CALUDE_zhang_qiujian_problem_l2281_228141


namespace NUMINAMATH_CALUDE_custom_op_negative_four_six_l2281_228168

/-- Custom binary operation "*" -/
def custom_op (a b : ℤ) : ℤ := a + 2 * b^2

/-- Theorem stating that (-4) * 6 = 68 under the custom operation -/
theorem custom_op_negative_four_six : custom_op (-4) 6 = 68 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_negative_four_six_l2281_228168


namespace NUMINAMATH_CALUDE_fruit_orders_eq_six_l2281_228162

/-- Represents the types of fruit in the basket -/
inductive Fruit
  | Apple
  | Peach
  | Pear

/-- The number of fruits in the basket -/
def basket_size : Nat := 3

/-- The number of chances to draw -/
def draw_chances : Nat := 2

/-- Calculates the number of different orders of fruit that can be drawn -/
def fruit_orders : Nat :=
  basket_size * (basket_size - 1)

theorem fruit_orders_eq_six :
  fruit_orders = 6 :=
sorry

end NUMINAMATH_CALUDE_fruit_orders_eq_six_l2281_228162


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l2281_228147

theorem complex_number_coordinates : (Complex.I + 1)^2 * Complex.I = -2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l2281_228147


namespace NUMINAMATH_CALUDE_range_of_m_l2281_228120

theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a^2 + b^2 = 1) (h2 : a^3 + b^3 + 1 = m * (a + b + 1)^3) :
  (3 * Real.sqrt 2 - 4) / 2 ≤ m ∧ m < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2281_228120


namespace NUMINAMATH_CALUDE_product_of_sums_l2281_228171

theorem product_of_sums (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 35)
  (h_sum_prod : a * b + b * c + c * a = 320)
  (h_prod : a * b * c = 600) :
  (a + b) * (b + c) * (c + a) = 10600 := by
sorry

end NUMINAMATH_CALUDE_product_of_sums_l2281_228171


namespace NUMINAMATH_CALUDE_mountain_climb_time_l2281_228195

/-- Represents a climber with ascending and descending speeds -/
structure Climber where
  ascendSpeed : ℝ
  descendSpeed : ℝ

/-- The mountain climbing scenario -/
structure MountainClimb where
  a : Climber
  b : Climber
  mountainHeight : ℝ
  meetingDistance : ℝ
  meetingTime : ℝ

theorem mountain_climb_time (mc : MountainClimb) : 
  mc.a.descendSpeed = 1.5 * mc.a.ascendSpeed →
  mc.b.descendSpeed = 1.5 * mc.b.ascendSpeed →
  mc.a.ascendSpeed > mc.b.ascendSpeed →
  mc.meetingTime = 1 →
  mc.meetingDistance = 600 →
  (mc.mountainHeight / mc.a.ascendSpeed + mc.mountainHeight / mc.a.descendSpeed = 1.5) :=
by sorry

end NUMINAMATH_CALUDE_mountain_climb_time_l2281_228195


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_nine_over_x_l2281_228183

theorem min_value_of_x_plus_nine_over_x (x : ℝ) (hx : x > 0) :
  x + 9 / x ≥ 6 ∧ (x + 9 / x = 6 ↔ x = 3) := by sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_nine_over_x_l2281_228183


namespace NUMINAMATH_CALUDE_imaginary_part_of_fraction_l2281_228101

theorem imaginary_part_of_fraction (i : ℂ) : i * i = -1 → Complex.im (5 * i / (2 - i)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_fraction_l2281_228101


namespace NUMINAMATH_CALUDE_fifth_house_gnomes_l2281_228157

/-- The number of houses on the street -/
def num_houses : Nat := 5

/-- The number of gnomes in each of the first four houses -/
def gnomes_per_house : Nat := 3

/-- The total number of gnomes on the street -/
def total_gnomes : Nat := 20

/-- The number of gnomes in the fifth house -/
def gnomes_in_fifth_house : Nat := total_gnomes - (4 * gnomes_per_house)

theorem fifth_house_gnomes :
  gnomes_in_fifth_house = 8 := by sorry

end NUMINAMATH_CALUDE_fifth_house_gnomes_l2281_228157


namespace NUMINAMATH_CALUDE_leadership_diagram_is_organizational_structure_l2281_228122

/-- Represents types of diagrams --/
inductive Diagram
  | ProgramFlowchart
  | ProcessFlowchart
  | KnowledgeStructureDiagram
  | OrganizationalStructureDiagram

/-- Represents a leadership relationship diagram --/
structure LeadershipDiagram where
  represents_leadership : Bool
  represents_structure : Bool

/-- Definition of an organizational structure diagram --/
def is_organizational_structure_diagram (d : LeadershipDiagram) : Prop :=
  d.represents_leadership ∧ d.represents_structure

/-- Theorem stating that a leadership relationship diagram in a governance group 
    is an organizational structure diagram --/
theorem leadership_diagram_is_organizational_structure :
  ∀ (d : LeadershipDiagram),
  d.represents_leadership ∧ d.represents_structure →
  is_organizational_structure_diagram d :=
by
  sorry

#check leadership_diagram_is_organizational_structure

end NUMINAMATH_CALUDE_leadership_diagram_is_organizational_structure_l2281_228122


namespace NUMINAMATH_CALUDE_harvard_attendance_percentage_l2281_228134

theorem harvard_attendance_percentage 
  (total_applicants : ℕ) 
  (acceptance_rate : ℚ)
  (other_schools_rate : ℚ)
  (attending_students : ℕ) :
  total_applicants = 20000 →
  acceptance_rate = 5 / 100 →
  other_schools_rate = 1 / 10 →
  attending_students = 900 →
  (attending_students : ℚ) / (total_applicants * acceptance_rate) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_harvard_attendance_percentage_l2281_228134


namespace NUMINAMATH_CALUDE_standard_deviation_from_age_range_job_applicants_standard_deviation_l2281_228100

/-- Given an average age and a number of distinct integer ages within one standard deviation,
    calculate the standard deviation. -/
theorem standard_deviation_from_age_range (average_age : ℕ) (distinct_ages : ℕ) : ℕ :=
  let standard_deviation := (distinct_ages - 1) / 2
  standard_deviation

/-- Prove that for an average age of 20 and 17 distinct integer ages within one standard deviation,
    the standard deviation is 8. -/
theorem job_applicants_standard_deviation : 
  standard_deviation_from_age_range 20 17 = 8 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_from_age_range_job_applicants_standard_deviation_l2281_228100


namespace NUMINAMATH_CALUDE_number_of_sides_interior_angle_measure_l2281_228187

/-- A regular polygon where the sum of interior angles is 180° more than three times the sum of exterior angles. -/
structure RegularPolygon where
  n : ℕ  -- number of sides
  sum_interior_angles : ℝ
  sum_exterior_angles : ℝ
  h1 : sum_interior_angles = (n - 2) * 180
  h2 : sum_exterior_angles = 360
  h3 : sum_interior_angles = 3 * sum_exterior_angles + 180

/-- The number of sides of the regular polygon is 9. -/
theorem number_of_sides (p : RegularPolygon) : p.n = 9 := by sorry

/-- The measure of each interior angle of the regular polygon is 140°. -/
theorem interior_angle_measure (p : RegularPolygon) : 
  (p.sum_interior_angles / p.n : ℝ) = 140 := by sorry

end NUMINAMATH_CALUDE_number_of_sides_interior_angle_measure_l2281_228187


namespace NUMINAMATH_CALUDE_solution_of_inequality1_solution_of_inequality2_l2281_228125

-- Define the solution set for the first inequality
def solutionSet1 : Set ℝ := {x | x > -1 ∧ x < 1}

-- Define the solution set for the second inequality
def solutionSet2 (a : ℝ) : Set ℝ :=
  if a = -2 then Set.univ
  else if a > -2 then {x | x ≤ -2 ∨ x ≥ a}
  else {x | x ≤ a ∨ x ≥ -2}

-- Theorem for the first inequality
theorem solution_of_inequality1 :
  {x : ℝ | (2 * x) / (x + 1) < 1} = solutionSet1 := by sorry

-- Theorem for the second inequality
theorem solution_of_inequality2 (a : ℝ) :
  {x : ℝ | x^2 + (2 - a) * x - 2 * a ≥ 0} = solutionSet2 a := by sorry

end NUMINAMATH_CALUDE_solution_of_inequality1_solution_of_inequality2_l2281_228125


namespace NUMINAMATH_CALUDE_crabapple_sequences_l2281_228140

/-- The number of students in Mrs. Crabapple's British Literature class -/
def num_students : ℕ := 13

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of possible sequences of crabapple recipients in a week -/
def num_sequences : ℕ := num_students ^ meetings_per_week

theorem crabapple_sequences :
  num_sequences = 2197 :=
sorry

end NUMINAMATH_CALUDE_crabapple_sequences_l2281_228140


namespace NUMINAMATH_CALUDE_dhoni_leftover_earnings_l2281_228113

theorem dhoni_leftover_earnings (total_earnings rent_percentage dishwasher_discount : ℝ) :
  rent_percentage = 40 →
  dishwasher_discount = 20 →
  let dishwasher_percentage := rent_percentage - (dishwasher_discount / 100) * rent_percentage
  let total_spent_percentage := rent_percentage + dishwasher_percentage
  let leftover_percentage := 100 - total_spent_percentage
  leftover_percentage = 28 :=
by sorry

end NUMINAMATH_CALUDE_dhoni_leftover_earnings_l2281_228113


namespace NUMINAMATH_CALUDE_fruit_picking_combinations_l2281_228136

def fruit_types : ℕ := 3
def picks : ℕ := 2

theorem fruit_picking_combinations : (fruit_types.choose picks) = 6 := by
  sorry

end NUMINAMATH_CALUDE_fruit_picking_combinations_l2281_228136


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2281_228152

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 + a * Complex.I) / (1 - Complex.I) = -2 - Complex.I →
  a = -3 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2281_228152


namespace NUMINAMATH_CALUDE_jasons_leg_tattoos_l2281_228163

theorem jasons_leg_tattoos (jason_arm_tattoos : ℕ) (adam_tattoos : ℕ) :
  jason_arm_tattoos = 2 →
  adam_tattoos = 23 →
  ∃ (jason_leg_tattoos : ℕ),
    adam_tattoos = 2 * (2 * jason_arm_tattoos + 2 * jason_leg_tattoos) + 3 ∧
    jason_leg_tattoos = 3 :=
by sorry

end NUMINAMATH_CALUDE_jasons_leg_tattoos_l2281_228163


namespace NUMINAMATH_CALUDE_train_speed_l2281_228121

/-- Calculates the speed of a train passing over a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 320 →
  bridge_length = 140 →
  time = 36.8 →
  (((train_length + bridge_length) / time) * 3.6) = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2281_228121


namespace NUMINAMATH_CALUDE_small_semicircle_radius_l2281_228176

/-- Given a configuration of a large semicircle, a circle, and a small semicircle that are all
    pairwise tangent, this theorem proves that the radius of the smaller semicircle is 4 when
    the radius of the large semicircle is 12 and the radius of the circle is 6. -/
theorem small_semicircle_radius
  (R : ℝ) -- Radius of the large semicircle
  (r : ℝ) -- Radius of the circle
  (x : ℝ) -- Radius of the small semicircle
  (h1 : R = 12) -- Given radius of large semicircle
  (h2 : r = 6)  -- Given radius of circle
  (h3 : R > 0 ∧ r > 0 ∧ x > 0) -- All radii are positive
  (h4 : R > r ∧ R > x) -- Large semicircle is the largest
  (h5 : (R - x)^2 + r^2 = (r + x)^2) -- Pythagorean theorem for tangent circles
  : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_small_semicircle_radius_l2281_228176


namespace NUMINAMATH_CALUDE_mango_purchase_problem_l2281_228179

/-- The problem of calculating the amount of mangoes purchased --/
theorem mango_purchase_problem (grape_kg : ℕ) (grape_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) :
  grape_kg = 8 →
  grape_rate = 80 →
  mango_rate = 55 →
  total_paid = 1135 →
  ∃ (mango_kg : ℕ), mango_kg * mango_rate + grape_kg * grape_rate = total_paid ∧ mango_kg = 9 :=
by sorry

end NUMINAMATH_CALUDE_mango_purchase_problem_l2281_228179


namespace NUMINAMATH_CALUDE_cuboid_max_volume_l2281_228117

theorem cuboid_max_volume (d : ℝ) (p : ℝ) (h1 : d = 10) (h2 : p = 8) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  a * b * c ≤ 192 ∧
  a^2 + b^2 + c^2 = d^2 ∧
  a^2 + b^2 = p^2 ∧
  (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    x^2 + y^2 + z^2 = d^2 → x^2 + y^2 = p^2 → x * y * z ≤ 192) :=
by
  sorry

end NUMINAMATH_CALUDE_cuboid_max_volume_l2281_228117


namespace NUMINAMATH_CALUDE_logical_equivalences_l2281_228110

theorem logical_equivalences :
  (∀ A B C : Prop,
    (A ∨ (B ∧ C) ↔ (A ∨ B) ∧ (A ∨ C)) ∧
    (¬((A ∨ (¬B)) ∨ (C ∧ (A ∨ (¬B)))) ↔ (¬A) ∧ B)) := by
  sorry

end NUMINAMATH_CALUDE_logical_equivalences_l2281_228110


namespace NUMINAMATH_CALUDE_two_thirds_of_five_times_nine_l2281_228173

theorem two_thirds_of_five_times_nine : (2 / 3 : ℚ) * (5 * 9) = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_five_times_nine_l2281_228173


namespace NUMINAMATH_CALUDE_soccer_penalty_kicks_l2281_228105

theorem soccer_penalty_kicks (total_players : ℕ) (goalies : ℕ) (shots : ℕ) :
  total_players = 22 →
  goalies = 4 →
  shots = goalies * (total_players - 1) →
  shots = 84 :=
by sorry

end NUMINAMATH_CALUDE_soccer_penalty_kicks_l2281_228105


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l2281_228190

/-- Given a geometric sequence with common ratio q and sum of first n terms S_n -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q ∧ S n = a 1 * (1 - q^n) / (1 - q)

theorem common_ratio_of_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  geometric_sequence a q S →
  a 2013 = 2 * S 2014 + 6 →
  3 * a 2014 = 2 * S 2015 + 6 →
  q = 1/2 ∨ q = 1 := by
sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l2281_228190


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2281_228193

theorem nested_fraction_equality : 
  1 + 2 / (3 + 6 / (7 + 8 / 9)) = 409 / 267 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2281_228193


namespace NUMINAMATH_CALUDE_board_cutting_l2281_228115

theorem board_cutting (total_length : ℝ) (shorter_length : ℝ) : 
  total_length = 69 →
  shorter_length + 2 * shorter_length = total_length →
  shorter_length = 23 := by
sorry

end NUMINAMATH_CALUDE_board_cutting_l2281_228115


namespace NUMINAMATH_CALUDE_complex_number_relation_l2281_228132

theorem complex_number_relation (a b c p q r : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
  (eq1 : a = (b + c) / (p - 3))
  (eq2 : b = (a + c) / (q - 3))
  (eq3 : c = (a + b) / (r - 3))
  (eq4 : p * q + p * r + q * r = 10)
  (eq5 : p + q + r = 6) :
  p * q * r = 14 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_relation_l2281_228132


namespace NUMINAMATH_CALUDE_logical_implications_l2281_228192

theorem logical_implications (p q : Prop) : 
  (((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q))) ∧
  (((p ∧ q) → ¬(¬p)) ∧ ¬(¬(¬p) → (p ∧ q))) ∧
  ((¬p → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → ¬p)) := by
  sorry

end NUMINAMATH_CALUDE_logical_implications_l2281_228192


namespace NUMINAMATH_CALUDE_third_roots_unity_quadratic_roots_l2281_228118

theorem third_roots_unity_quadratic_roots :
  ∃ (S : Finset ℂ), 
    (∀ z ∈ S, z^3 = 1) ∧ 
    (∀ z ∈ S, ∃ a : ℤ, z^2 + a*z - 1 = 0) ∧
    S.card = 3 ∧
    (∀ z : ℂ, z^3 = 1 → (∃ a : ℤ, z^2 + a*z - 1 = 0) → z ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_third_roots_unity_quadratic_roots_l2281_228118


namespace NUMINAMATH_CALUDE_green_ball_theorem_l2281_228116

/-- Represents the price and quantity information for green balls --/
structure GreenBallInfo where
  saltyCost : ℚ
  saltyQuantity : ℕ
  duckCost : ℚ
  duckQuantity : ℕ

/-- Represents a purchase plan --/
structure PurchasePlan where
  saltyQuantity : ℕ
  duckQuantity : ℕ

/-- Represents an exchange method --/
structure ExchangeMethod where
  coupons : ℕ
  saltyCoupons : ℕ
  duckCoupons : ℕ

/-- Main theorem about green ball prices, purchase plans, and exchange methods --/
theorem green_ball_theorem (info : GreenBallInfo) 
  (h1 : info.duckCost = 2 * info.saltyCost)
  (h2 : info.duckCost * info.duckQuantity = 40)
  (h3 : info.saltyCost * info.saltyQuantity = 30)
  (h4 : info.saltyQuantity = info.duckQuantity + 4)
  (h5 : ∀ plan : PurchasePlan, 
    plan.saltyQuantity ≥ 20 ∧ 
    plan.duckQuantity ≥ 20 ∧ 
    plan.saltyQuantity % 10 = 0 ∧
    info.saltyCost * plan.saltyQuantity + info.duckCost * plan.duckQuantity = 200)
  (h6 : ∀ method : ExchangeMethod,
    1 < method.coupons ∧ 
    method.coupons < 10 ∧
    method.saltyCoupons + method.duckCoupons = method.coupons) :
  (info.saltyCost = 5/2 ∧ info.duckCost = 5) ∧
  (∃ (plans : List PurchasePlan), plans = 
    [(PurchasePlan.mk 20 30), (PurchasePlan.mk 30 25), (PurchasePlan.mk 40 20)]) ∧
  (∃ (methods : List ExchangeMethod), methods = 
    [(ExchangeMethod.mk 5 5 0), (ExchangeMethod.mk 5 0 5), 
     (ExchangeMethod.mk 8 6 2), (ExchangeMethod.mk 8 1 7)]) :=
by sorry

end NUMINAMATH_CALUDE_green_ball_theorem_l2281_228116


namespace NUMINAMATH_CALUDE_haley_magazine_boxes_l2281_228158

theorem haley_magazine_boxes (total_magazines : ℕ) (magazines_per_box : ℕ) (h1 : total_magazines = 63) (h2 : magazines_per_box = 9) :
  total_magazines / magazines_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_haley_magazine_boxes_l2281_228158


namespace NUMINAMATH_CALUDE_sets_equality_l2281_228188

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x + 1 = 0}
def N : Set ℝ := {1}

-- Theorem statement
theorem sets_equality : M = N := by
  sorry

end NUMINAMATH_CALUDE_sets_equality_l2281_228188


namespace NUMINAMATH_CALUDE_quadratic_no_roots_l2281_228106

/-- A quadratic polynomial function -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The derivative of a quadratic polynomial function -/
def quadratic_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

/-- Theorem: If the graph of a quadratic polynomial and its derivative
    divide the coordinate plane into four parts, then the polynomial has no real roots -/
theorem quadratic_no_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    quadratic a b c x₁ = quadratic_derivative a b x₁ ∧
    quadratic a b c x₂ = quadratic_derivative a b x₂) →
  (∀ x : ℝ, quadratic a b c x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_l2281_228106


namespace NUMINAMATH_CALUDE_all_positives_can_be_written_l2281_228191

/-- The predicate that determines if a number can be written on the board -/
def CanBeWritten (n : ℕ) : Prop :=
  ∃ (sequence : ℕ → ℕ), sequence 0 = 1 ∧
  (∀ k, ∃ b, (sequence k + b + 1) ∣ (sequence k^2 + b^2 + 1) ∧
            sequence (k + 1) = b)

/-- The main theorem stating that any positive integer can be written on the board -/
theorem all_positives_can_be_written :
  ∀ n : ℕ, n > 0 → CanBeWritten n :=
sorry

end NUMINAMATH_CALUDE_all_positives_can_be_written_l2281_228191


namespace NUMINAMATH_CALUDE_marks_ratio_l2281_228199

def total_marks : ℕ := 170
def science_marks : ℕ := 17

def english_math_ratio : ℚ := 1 / 4

theorem marks_ratio : 
  ∃ (english_marks math_marks : ℕ),
    english_marks + math_marks + science_marks = total_marks ∧
    english_marks / math_marks = english_math_ratio ∧
    english_marks / science_marks = 31 / 17 :=
by sorry

end NUMINAMATH_CALUDE_marks_ratio_l2281_228199


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l2281_228148

/-- Given points A, B, C on the graph of y = 3/x, prove y₂ < y₁ < y₃ -/
theorem inverse_proportion_order (y₁ y₂ y₃ : ℝ) : 
  y₁ = 3 / (-2) → y₂ = 3 / (-1) → y₃ = 3 / 1 → y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l2281_228148


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_age_l2281_228185

theorem mans_age_twice_sons_age (son_age : ℕ) (age_difference : ℕ) (years : ℕ) : 
  son_age = 16 →
  age_difference = 18 →
  (son_age + age_difference + years) = 2 * (son_age + years) →
  years = 2 := by
sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_age_l2281_228185


namespace NUMINAMATH_CALUDE_prob_all_cocaptains_l2281_228166

def team_sizes : List Nat := [4, 6, 7, 9]
def num_teams : Nat := 4
def num_cocaptains : Nat := 3

def prob_select_cocaptains (n : Nat) : Rat :=
  6 / (n * (n - 1) * (n - 2))

theorem prob_all_cocaptains : 
  (1 : Rat) / num_teams * (team_sizes.map prob_select_cocaptains).sum = 143 / 1680 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_cocaptains_l2281_228166


namespace NUMINAMATH_CALUDE_sarah_ate_one_apple_l2281_228149

/-- The number of apples Sarah ate while walking home -/
def apples_eaten (total : ℕ) (to_teachers : ℕ) (to_friends : ℕ) (left : ℕ) : ℕ :=
  total - (to_teachers + to_friends) - left

/-- Theorem stating that Sarah ate 1 apple while walking home -/
theorem sarah_ate_one_apple :
  apples_eaten 25 16 5 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sarah_ate_one_apple_l2281_228149


namespace NUMINAMATH_CALUDE_diagonals_to_sides_ratio_for_pentagon_l2281_228164

-- Define the number of diagonals function
def num_diagonals (n : ℕ) : ℚ := n * (n - 3) / 2

-- Theorem statement
theorem diagonals_to_sides_ratio_for_pentagon :
  let n : ℕ := 5
  (num_diagonals n) / n = 1 := by sorry

end NUMINAMATH_CALUDE_diagonals_to_sides_ratio_for_pentagon_l2281_228164


namespace NUMINAMATH_CALUDE_lcm_triple_count_l2281_228156

/-- The least common multiple of two positive integers -/
def lcm (a b : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a, b, c) satisfying the LCM conditions -/
def count_triples : ℕ := sorry

/-- Main theorem: There are exactly 70 ordered triples satisfying the LCM conditions -/
theorem lcm_triple_count :
  count_triples = 70 :=
by sorry

end NUMINAMATH_CALUDE_lcm_triple_count_l2281_228156


namespace NUMINAMATH_CALUDE_expression_evaluation_l2281_228138

/-- Given a = -2 and b = -1/2, prove that 2(3a^2 - 4ab) - [a^2 - 3(2a + 3ab)] evaluates to 9 -/
theorem expression_evaluation :
  let a : ℚ := -2
  let b : ℚ := -1/2
  2 * (3 * a^2 - 4 * a * b) - (a^2 - 3 * (2 * a + 3 * a * b)) = 9 := by
sorry


end NUMINAMATH_CALUDE_expression_evaluation_l2281_228138


namespace NUMINAMATH_CALUDE_loan_interest_equality_l2281_228159

theorem loan_interest_equality (total : ℝ) (second_part : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time1 : ℝ) (n : ℝ) :
  total = 2665 →
  second_part = 1332.5 →
  rate1 = 0.03 →
  rate2 = 0.05 →
  time1 = 5 →
  (total - second_part) * rate1 * time1 = second_part * rate2 * n →
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_loan_interest_equality_l2281_228159


namespace NUMINAMATH_CALUDE_smallest_n_perfect_square_and_fifth_power_l2281_228144

theorem smallest_n_perfect_square_and_fifth_power : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (a : ℕ), 4 * n = a^2) ∧
  (∃ (b : ℕ), 5 * n = b^5) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), 4 * m = x^2) → 
    (∃ (y : ℕ), 5 * m = y^5) → 
    m ≥ n) ∧
  n = 3125 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_square_and_fifth_power_l2281_228144


namespace NUMINAMATH_CALUDE_expected_score_is_one_l2281_228165

/-- The number of black balls in the bag -/
def num_black : ℕ := 3

/-- The number of red balls in the bag -/
def num_red : ℕ := 1

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_black + num_red

/-- The score for drawing a black ball -/
def black_score : ℝ := 0

/-- The score for drawing a red ball -/
def red_score : ℝ := 2

/-- The expected value of the score when drawing two balls -/
def expected_score : ℝ := 1

/-- Theorem stating that the expected score when drawing two balls is 1 -/
theorem expected_score_is_one :
  let prob_two_black : ℝ := (num_black / total_balls) * ((num_black - 1) / (total_balls - 1))
  let prob_one_each : ℝ := (num_black / total_balls) * (num_red / (total_balls - 1)) +
                           (num_red / total_balls) * (num_black / (total_balls - 1))
  prob_two_black * (2 * black_score) + prob_one_each * (black_score + red_score) = expected_score :=
by sorry

end NUMINAMATH_CALUDE_expected_score_is_one_l2281_228165


namespace NUMINAMATH_CALUDE_sin_cos_15_ratio_l2281_228160

theorem sin_cos_15_ratio : 
  (Real.sin (15 * π / 180) + Real.cos (15 * π / 180)) / 
  (Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_15_ratio_l2281_228160


namespace NUMINAMATH_CALUDE_quadratic_exponent_implies_m_eq_two_l2281_228103

/-- A function is quadratic if it can be expressed as ax² + bx + c, where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The main theorem: If y = (m+2)x^(m²-2) is quadratic, then m = 2 -/
theorem quadratic_exponent_implies_m_eq_two (m : ℝ) :
  IsQuadratic (fun x ↦ (m + 2) * x^(m^2 - 2)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_exponent_implies_m_eq_two_l2281_228103


namespace NUMINAMATH_CALUDE_dumplings_eaten_l2281_228112

theorem dumplings_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 14 → remaining = 7 → eaten = initial - remaining :=
by sorry

end NUMINAMATH_CALUDE_dumplings_eaten_l2281_228112

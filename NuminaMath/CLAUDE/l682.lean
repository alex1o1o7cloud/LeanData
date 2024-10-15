import Mathlib

namespace NUMINAMATH_CALUDE_f_of_f_one_equals_seven_l682_68287

def f (x : ℝ) : ℝ := 3 * x^2 - 5

theorem f_of_f_one_equals_seven : f (f 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_one_equals_seven_l682_68287


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l682_68274

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a line given by the equation (3+k)x + (1-2k)y + 1 + 5k = 0 -/
def lies_on_line (p : Point) (k : ℝ) : Prop :=
  (3 + k) * p.x + (1 - 2*k) * p.y + 1 + 5*k = 0

/-- The theorem stating that (-1, 2) is the unique fixed point for all lines -/
theorem fixed_point_theorem :
  ∃! p : Point, ∀ k : ℝ, lies_on_line p k :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l682_68274


namespace NUMINAMATH_CALUDE_graduation_photo_arrangements_l682_68202

/-- The number of students in the class -/
def num_students : ℕ := 6

/-- The total number of people (students + teacher) -/
def total_people : ℕ := num_students + 1

/-- The number of arrangements with the teacher in the middle -/
def total_arrangements : ℕ := (num_students.factorial)

/-- The number of arrangements with the teacher in the middle and students A and B adjacent -/
def adjacent_arrangements : ℕ := 4 * 2 * ((num_students - 2).factorial)

/-- The number of valid arrangements -/
def valid_arrangements : ℕ := total_arrangements - adjacent_arrangements

theorem graduation_photo_arrangements :
  valid_arrangements = 528 := by sorry

end NUMINAMATH_CALUDE_graduation_photo_arrangements_l682_68202


namespace NUMINAMATH_CALUDE_stratified_sampling_science_students_l682_68259

theorem stratified_sampling_science_students 
  (total_students : ℕ) 
  (science_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 720) 
  (h2 : science_students = 480) 
  (h3 : sample_size = 90) :
  (science_students : ℚ) / (total_students : ℚ) * (sample_size : ℚ) = 60 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_science_students_l682_68259


namespace NUMINAMATH_CALUDE_rope_folding_segments_l682_68240

/-- The number of segments produced by folding a rope n times and cutting in the middle of the last fold -/
def num_segments (n : ℕ) : ℕ := 2^n + 1

/-- Theorem stating that the number of segments follows the pattern for all natural numbers -/
theorem rope_folding_segments (n : ℕ) : num_segments n = 2^n + 1 := by
  sorry

/-- Verifying the given examples -/
example : num_segments 1 = 3 := by sorry
example : num_segments 2 = 5 := by sorry
example : num_segments 3 = 9 := by sorry

end NUMINAMATH_CALUDE_rope_folding_segments_l682_68240


namespace NUMINAMATH_CALUDE_john_rejection_percentage_l682_68214

theorem john_rejection_percentage
  (jane_rejection_rate : ℝ)
  (total_rejection_rate : ℝ)
  (jane_inspection_fraction : ℝ)
  (h1 : jane_rejection_rate = 0.009)
  (h2 : total_rejection_rate = 0.0075)
  (h3 : jane_inspection_fraction = 0.625)
  : ∃ (john_rejection_rate : ℝ),
    john_rejection_rate = 0.005 ∧
    jane_rejection_rate * jane_inspection_fraction +
    john_rejection_rate * (1 - jane_inspection_fraction) =
    total_rejection_rate :=
by sorry

end NUMINAMATH_CALUDE_john_rejection_percentage_l682_68214


namespace NUMINAMATH_CALUDE_box_depth_proof_l682_68221

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents a cube -/
structure Cube where
  edgeLength : ℕ

/-- Theorem: Given a box with specific dimensions filled with cubes, prove its depth -/
theorem box_depth_proof (box : BoxDimensions) (cube : Cube) (numCubes : ℕ) :
  box.length = 36 →
  box.width = 45 →
  numCubes = 40 →
  (box.length * box.width * box.depth = numCubes * cube.edgeLength ^ 3) →
  (box.length % cube.edgeLength = 0) →
  (box.width % cube.edgeLength = 0) →
  (box.depth % cube.edgeLength = 0) →
  box.depth = 18 := by
  sorry


end NUMINAMATH_CALUDE_box_depth_proof_l682_68221


namespace NUMINAMATH_CALUDE_amy_homework_time_l682_68278

theorem amy_homework_time (math_problems : ℕ) (spelling_problems : ℕ) (problems_per_hour : ℕ) : 
  math_problems = 18 → spelling_problems = 6 → problems_per_hour = 4 →
  (math_problems + spelling_problems) / problems_per_hour = 6 := by
  sorry

end NUMINAMATH_CALUDE_amy_homework_time_l682_68278


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1512_l682_68245

/-- The largest perfect square factor of 1512 is 36 -/
theorem largest_perfect_square_factor_of_1512 :
  ∃ (n : ℕ), n * n = 36 ∧ n * n ∣ 1512 ∧ ∀ (m : ℕ), m * m ∣ 1512 → m * m ≤ n * n :=
by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1512_l682_68245


namespace NUMINAMATH_CALUDE_price_reduction_effect_l682_68200

theorem price_reduction_effect (P Q : ℝ) (P_positive : P > 0) (Q_positive : Q > 0) :
  let new_price := P * (1 - 0.35)
  let new_quantity := Q * (1 + 0.8)
  let original_revenue := P * Q
  let new_revenue := new_price * new_quantity
  (new_revenue - original_revenue) / original_revenue = 0.17 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_effect_l682_68200


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l682_68217

theorem sin_cos_sixth_power_sum (θ : Real) (h : Real.sin (2 * θ) = 1/3) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l682_68217


namespace NUMINAMATH_CALUDE_inequality_equivalence_l682_68206

theorem inequality_equivalence (x : ℝ) : 
  (x ∈ Set.Icc (-1 : ℝ) 1) ↔ 
  (∀ (n : ℕ) (a : ℕ → ℝ), n ≥ 2 → (∀ i, i ∈ Finset.range n → a i ≥ 1) → 
    ((Finset.range n).prod (λ i => (a i + x) / 2) ≤ 
     ((Finset.range n).prod (λ i => a i) + x) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l682_68206


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l682_68232

theorem min_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12) :
  ∃ (a b : ℕ+), a ≠ b ∧ ((1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12) ∧ (a + b = 50) ∧ 
  (∀ (c d : ℕ+), c ≠ d → ((1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12) → (c + d ≥ 50)) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l682_68232


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l682_68271

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 15 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 289) ∧
  (has_no_small_prime_factors 289) ∧
  (∀ m : ℕ, m < 289 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l682_68271


namespace NUMINAMATH_CALUDE_age_ratio_proof_l682_68257

/-- Represents a person's age -/
structure Age where
  years : ℕ

/-- Represents the ratio between two ages -/
structure AgeRatio where
  numerator : ℕ
  denominator : ℕ

def Arun : Age := ⟨20⟩
def Deepak : Age := ⟨30⟩

def currentRatio : AgeRatio := ⟨2, 3⟩

theorem age_ratio_proof :
  (Arun.years + 5 = 25) ∧
  (Deepak.years = 30) →
  (currentRatio.numerator * Deepak.years = currentRatio.denominator * Arun.years) :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l682_68257


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l682_68226

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 3

-- Define the conditions
def min_at_2 (a b : ℝ) : Prop := ∀ x, f a b x ≥ f a b 2

def intercept_length_2 (a b : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ < x₂ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0 ∧ x₂ - x₁ = 2

-- Define g(x)
def g (a b m : ℝ) (x : ℝ) : ℝ := f a b x - m * x

-- Define the conditions for g(x)
def g_zeros_in_intervals (a b m : ℝ) : Prop :=
  ∃ x₁ x₂, 0 < x₁ ∧ x₁ < 2 ∧ 2 < x₂ ∧ x₂ < 3 ∧ g a b m x₁ = 0 ∧ g a b m x₂ = 0

-- Define the minimum value condition
def min_value_condition (a b : ℝ) (t : ℝ) : Prop :=
  ∀ x ∈ Set.Icc t (t + 1), f a b x ≥ -1/2 ∧ ∃ x₀ ∈ Set.Icc t (t + 1), f a b x₀ = -1/2

-- State the theorem
theorem quadratic_function_properties :
  ∀ a b : ℝ, min_at_2 a b → intercept_length_2 a b →
  (∃ m : ℝ, g_zeros_in_intervals a b m ∧ -1/2 < m ∧ m < 0) ∧
  (∃ t : ℝ, (min_value_condition a b t ∧ t = 1 - Real.sqrt 2 / 2) ∨
            (min_value_condition a b t ∧ t = 2 + Real.sqrt 2 / 2)) ∧
  a = 1 ∧ b = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l682_68226


namespace NUMINAMATH_CALUDE_runners_on_circular_track_l682_68246

/-- Represents a runner on a circular track -/
structure Runner where
  lap_time : ℝ
  speed : ℝ

/-- Theorem about two runners on a circular track -/
theorem runners_on_circular_track
  (track_length : ℝ)
  (troye daniella : Runner)
  (h1 : track_length > 0)
  (h2 : troye.lap_time = 56)
  (h3 : troye.speed = track_length / troye.lap_time)
  (h4 : daniella.speed = track_length / daniella.lap_time)
  (h5 : troye.speed + daniella.speed = track_length / 24) :
  daniella.lap_time = 42 := by
  sorry

end NUMINAMATH_CALUDE_runners_on_circular_track_l682_68246


namespace NUMINAMATH_CALUDE_sat_markings_count_l682_68254

/-- The number of ways to mark a single question on the SAT answer sheet -/
def markings_per_question : ℕ := 32

/-- The number of questions to be marked -/
def num_questions : ℕ := 10

/-- Function to calculate the number of valid sequences of length n with no consecutive 1s -/
def f : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n + 2) => f (n + 1) + f n

/-- The number of letters in the SAT answer sheet -/
def num_letters : ℕ := 5

/-- Theorem stating the total number of ways to mark the SAT answer sheet -/
theorem sat_markings_count :
  (f num_questions) ^ num_letters = 2^20 * 3^10 := by sorry

end NUMINAMATH_CALUDE_sat_markings_count_l682_68254


namespace NUMINAMATH_CALUDE_sinC_value_sine_law_extension_l682_68268

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  area : ℝ

/-- The area of the triangle satisfies the given condition -/
def areaCondition (t : Triangle) : Prop :=
  t.area = (t.a + t.b)^2 - t.c^2

/-- The sum of two sides equals 4 -/
def sideSum (t : Triangle) : Prop :=
  t.a + t.b = 4

/-- Theorem 1: If the area condition and side sum condition hold, then sin C = 8/17 -/
theorem sinC_value (t : Triangle) (h1 : areaCondition t) (h2 : sideSum t) :
  Real.sin t.C = 8 / 17 := by sorry

/-- Theorem 2: The ratio of squared difference of sides to the square of the third side
    equals the ratio of sine of difference of angles to the sine of the third angle -/
theorem sine_law_extension (t : Triangle) :
  (t.a^2 - t.b^2) / t.c^2 = Real.sin (t.A - t.B) / Real.sin t.C := by sorry

end NUMINAMATH_CALUDE_sinC_value_sine_law_extension_l682_68268


namespace NUMINAMATH_CALUDE_at_most_two_protocols_l682_68295

/-- Represents a skier in the race -/
structure Skier :=
  (number : Nat)
  (startPosition : Nat)
  (finishPosition : Nat)
  (overtakes : Nat)
  (overtakenBy : Nat)

/-- Represents the race conditions -/
structure RaceConditions :=
  (skiers : List Skier)
  (totalSkiers : Nat)
  (h_totalSkiers : totalSkiers = 7)
  (h_startSequence : ∀ s : Skier, s ∈ skiers → s.number = s.startPosition)
  (h_constantSpeed : ∀ s : Skier, s ∈ skiers → s.overtakes + s.overtakenBy = 2)
  (h_uniqueFinish : ∀ s1 s2 : Skier, s1 ∈ skiers → s2 ∈ skiers → s1.finishPosition = s2.finishPosition → s1 = s2)

/-- The theorem to be proved -/
theorem at_most_two_protocols (rc : RaceConditions) : 
  (∃ p1 p2 : List Nat, 
    (∀ p : List Nat, p.length = rc.totalSkiers ∧ (∀ s : Skier, s ∈ rc.skiers → s.finishPosition = p.indexOf s.number + 1) → p = p1 ∨ p = p2) ∧
    p1 ≠ p2) :=
sorry

end NUMINAMATH_CALUDE_at_most_two_protocols_l682_68295


namespace NUMINAMATH_CALUDE_organize_60_toys_in_15_minutes_l682_68255

/-- Represents the toy organizing scenario with Mia and her dad -/
structure ToyOrganizing where
  totalToys : ℕ
  cycleTime : ℕ
  dadPlaces : ℕ
  miaTakesOut : ℕ

/-- Calculates the time in minutes to organize all toys -/
def timeToOrganize (scenario : ToyOrganizing) : ℚ :=
  sorry

/-- Theorem stating that the time to organize 60 toys is 15 minutes -/
theorem organize_60_toys_in_15_minutes :
  let scenario : ToyOrganizing := {
    totalToys := 60,
    cycleTime := 30,  -- in seconds
    dadPlaces := 6,
    miaTakesOut := 4
  }
  timeToOrganize scenario = 15 := by sorry

end NUMINAMATH_CALUDE_organize_60_toys_in_15_minutes_l682_68255


namespace NUMINAMATH_CALUDE_blue_ball_weight_l682_68292

theorem blue_ball_weight (brown_weight total_weight : ℝ) 
  (h1 : brown_weight = 3.12)
  (h2 : total_weight = 9.12) :
  total_weight - brown_weight = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_ball_weight_l682_68292


namespace NUMINAMATH_CALUDE_smallest_circle_radius_l682_68294

/-- Given a circle of radius r that touches two identical circles and a smaller circle,
    all externally tangent to each other, the radius of the smallest circle is r/6. -/
theorem smallest_circle_radius (r : ℝ) (hr : r > 0) : ∃ (r_small : ℝ), r_small = r / 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_circle_radius_l682_68294


namespace NUMINAMATH_CALUDE_charity_fundraising_l682_68263

theorem charity_fundraising 
  (total_amount : ℕ) 
  (num_friends : ℕ) 
  (min_amount : ℕ) 
  (h1 : total_amount = 3000)
  (h2 : num_friends = 10)
  (h3 : min_amount = 300) :
  (total_amount / num_friends = min_amount) ∧ 
  (∀ (amount : ℕ), amount ≥ min_amount → amount * num_friends = total_amount → amount = min_amount) :=
by sorry

end NUMINAMATH_CALUDE_charity_fundraising_l682_68263


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l682_68205

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference between consecutive terms -/
  d : ℝ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a + (n - 1 : ℝ) * seq.d

theorem arithmetic_sequence_seventh_term
  (seq : ArithmeticSequence)
  (h3 : seq.nthTerm 3 = 17)
  (h5 : seq.nthTerm 5 = 39) :
  seq.nthTerm 7 = 61 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l682_68205


namespace NUMINAMATH_CALUDE_two_roots_iff_twenty_l682_68283

/-- The quadratic equation in x parameterized by a -/
def f (a : ℝ) (x : ℝ) : ℝ := a^2 * (x - 2) + a * (39 - 20*x) + 20

/-- The proposition that the equation has at least two distinct roots -/
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0

theorem two_roots_iff_twenty :
  ∀ a : ℝ, has_two_distinct_roots a ↔ a = 20 := by sorry

end NUMINAMATH_CALUDE_two_roots_iff_twenty_l682_68283


namespace NUMINAMATH_CALUDE_sam_initial_dimes_l682_68275

/-- Represents the number of cents in a coin -/
def cents_in_coin (coin : String) : ℕ :=
  match coin with
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Represents Sam's initial coin counts and purchases -/
structure SamsPurchase where
  initial_quarters : ℕ
  candy_bars : ℕ
  candy_bar_price : ℕ
  lollipops : ℕ
  lollipop_price : ℕ
  cents_left : ℕ

/-- Theorem stating that Sam had 19 dimes initially -/
theorem sam_initial_dimes (purchase : SamsPurchase)
  (h1 : purchase.initial_quarters = 6)
  (h2 : purchase.candy_bars = 4)
  (h3 : purchase.candy_bar_price = 3)
  (h4 : purchase.lollipops = 1)
  (h5 : purchase.lollipop_price = 1)
  (h6 : purchase.cents_left = 195) :
  (purchase.cents_left +
   purchase.candy_bars * purchase.candy_bar_price * cents_in_coin "dime" +
   purchase.lollipops * cents_in_coin "quarter" -
   purchase.initial_quarters * cents_in_coin "quarter") / cents_in_coin "dime" = 19 := by
  sorry

#eval cents_in_coin "dime"  -- Should output 10
#eval cents_in_coin "quarter"  -- Should output 25

end NUMINAMATH_CALUDE_sam_initial_dimes_l682_68275


namespace NUMINAMATH_CALUDE_polynomial_value_bound_l682_68288

/-- A polynomial with three distinct real roots -/
structure TripleRootPoly where
  a : ℝ
  b : ℝ
  c : ℝ
  has_three_distinct_roots : ∃ (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    ∀ t, t^3 + a*t^2 + b*t + c = 0 ↔ t = r₁ ∨ t = r₂ ∨ t = r₃

/-- The polynomial P(t) = t^3 + at^2 + bt + c -/
def P (poly : TripleRootPoly) (t : ℝ) : ℝ :=
  t^3 + poly.a*t^2 + poly.b*t + poly.c

/-- The equation (x^2 + x + 2013)^3 + a(x^2 + x + 2013)^2 + b(x^2 + x + 2013) + c = 0 has no real roots -/
def no_real_roots (poly : TripleRootPoly) : Prop :=
  ∀ x : ℝ, (x^2 + x + 2013)^3 + poly.a*(x^2 + x + 2013)^2 + poly.b*(x^2 + x + 2013) + poly.c ≠ 0

/-- The main theorem -/
theorem polynomial_value_bound (poly : TripleRootPoly) (h : no_real_roots poly) : 
  P poly 2013 > 1/64 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_bound_l682_68288


namespace NUMINAMATH_CALUDE_exists_sum_digits_div_11_l682_68234

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among any 39 consecutive natural numbers, there is always one whose sum of digits is divisible by 11 -/
theorem exists_sum_digits_div_11 (start : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (sum_of_digits (start + k)) % 11 = 0 := by sorry

end NUMINAMATH_CALUDE_exists_sum_digits_div_11_l682_68234


namespace NUMINAMATH_CALUDE_playstation_cost_proof_l682_68266

-- Define the given values
def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def game_price : ℝ := 7.5
def games_to_sell : ℕ := 20

-- Define the cost of the PlayStation
def playstation_cost : ℝ := 500

-- Theorem statement
theorem playstation_cost_proof :
  birthday_money + christmas_money + game_price * (games_to_sell : ℝ) = playstation_cost := by
  sorry

end NUMINAMATH_CALUDE_playstation_cost_proof_l682_68266


namespace NUMINAMATH_CALUDE_planting_methods_result_l682_68238

/-- The number of rows in the field -/
def total_rows : ℕ := 10

/-- The minimum required interval between crops A and B -/
def min_interval : ℕ := 6

/-- The number of crops to be planted -/
def num_crops : ℕ := 2

/-- Calculates the number of ways to plant two crops with the given constraints -/
def planting_methods (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  -- n: total rows
  -- k: number of crops
  -- m: minimum interval
  sorry

theorem planting_methods_result : planting_methods total_rows num_crops min_interval = 12 := by
  sorry

end NUMINAMATH_CALUDE_planting_methods_result_l682_68238


namespace NUMINAMATH_CALUDE_lines_perpendicular_l682_68256

-- Define the lines l₁ and l
def l₁ (a : ℝ) (x y : ℝ) : Prop := 2 * x - a * y - 1 = 0
def l (x y : ℝ) : Prop := x + 2 * y = 0

-- Define the theorem
theorem lines_perpendicular :
  ∃ a : ℝ, 
    (l₁ a 1 1) ∧ 
    (∀ x y : ℝ, l₁ a x y → l x y → (2 : ℝ) * (-1/2 : ℝ) = -1) :=
by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_l682_68256


namespace NUMINAMATH_CALUDE_counterexample_exists_l682_68265

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem counterexample_exists : ∃ n : ℕ, 
  (sum_of_digits n % 27 = 0) ∧ 
  (n % 27 ≠ 0) ∧ 
  (n = 9918) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l682_68265


namespace NUMINAMATH_CALUDE_remainder_theorem_l682_68289

theorem remainder_theorem (w : ℤ) (h : (w + 3) % 11 = 0) : w % 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l682_68289


namespace NUMINAMATH_CALUDE_floor_sum_equals_155_l682_68282

theorem floor_sum_equals_155 (p q r s : ℝ) : 
  p > 0 → q > 0 → r > 0 → s > 0 →
  p^2 + q^2 = 3024 →
  r^2 + s^2 = 3024 →
  p * r = 1500 →
  q * s = 1500 →
  ⌊p + q + r + s⌋ = 155 := by
sorry

end NUMINAMATH_CALUDE_floor_sum_equals_155_l682_68282


namespace NUMINAMATH_CALUDE_inverse_variation_proof_l682_68290

/-- Given that x² varies inversely with y⁴, prove that when x = 5 for y = 2, 
    then x² = 25/16 when y = 4 -/
theorem inverse_variation_proof (x y : ℝ) (h : ∃ k : ℝ, x^2 * y^4 = k) 
  (h_initial : (5 : ℝ)^2 * 2^4 = x^2 * y^4) : 
  (∃ x' : ℝ, x'^2 * 4^4 = x^2 * y^4 ∧ x'^2 = 25/16) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_proof_l682_68290


namespace NUMINAMATH_CALUDE_a7_equals_one_l682_68241

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem a7_equals_one (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 13 = 1 →
  a 1 + a 13 = 8 →
  a 7 = 1 :=
by sorry

end NUMINAMATH_CALUDE_a7_equals_one_l682_68241


namespace NUMINAMATH_CALUDE_horner_evaluation_exclude_l682_68242

def horner_polynomial (x : ℤ) : ℤ :=
  ((7 * x + 3) * x - 5) * x + 11

def horner_step1 (x : ℤ) : ℤ :=
  7 * x + 3

def horner_step2 (x : ℤ) : ℤ :=
  (7 * x + 3) * x - 5

theorem horner_evaluation_exclude (x : ℤ) :
  x = 23 →
  horner_polynomial x ≠ 85169 ∧
  horner_step1 x ≠ 85169 ∧
  horner_step2 x ≠ 85169 :=
by sorry

end NUMINAMATH_CALUDE_horner_evaluation_exclude_l682_68242


namespace NUMINAMATH_CALUDE_min_points_eleventh_game_l682_68210

/-- Represents the scores of a basketball player -/
structure BasketballScores where
  scores_7_to_10 : Fin 4 → ℕ
  total_after_6 : ℕ
  total_after_10 : ℕ
  total_after_11 : ℕ

/-- The minimum number of points required in the 11th game -/
def min_points_11th_game (bs : BasketballScores) : ℕ := bs.total_after_11 - bs.total_after_10

/-- Theorem stating the minimum points required in the 11th game -/
theorem min_points_eleventh_game 
  (bs : BasketballScores)
  (h1 : bs.scores_7_to_10 = ![21, 15, 12, 19])
  (h2 : (bs.total_after_10 : ℚ) / 10 > (bs.total_after_6 : ℚ) / 6)
  (h3 : (bs.total_after_11 : ℚ) / 11 > 20)
  (h4 : bs.total_after_10 = bs.total_after_6 + (bs.scores_7_to_10 0) + (bs.scores_7_to_10 1) + 
                            (bs.scores_7_to_10 2) + (bs.scores_7_to_10 3))
  : min_points_11th_game bs = 58 := by
  sorry


end NUMINAMATH_CALUDE_min_points_eleventh_game_l682_68210


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l682_68258

theorem solution_set_reciprocal_inequality (x : ℝ) : 
  (1 / x > 2) ↔ (0 < x ∧ x < 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l682_68258


namespace NUMINAMATH_CALUDE_painting_time_for_six_stools_l682_68279

/-- Represents the painting process for stools -/
structure StoolPainting where
  num_stools : Nat
  first_coat_time : Nat
  wait_time : Nat

/-- Calculates the minimum time required to paint all stools -/
def minimum_painting_time (sp : StoolPainting) : Nat :=
  sp.num_stools * sp.first_coat_time + sp.wait_time + sp.first_coat_time

/-- Theorem stating that the minimum time to paint 6 stools is 24 minutes -/
theorem painting_time_for_six_stools :
  let sp : StoolPainting := {
    num_stools := 6,
    first_coat_time := 2,
    wait_time := 10
  }
  minimum_painting_time sp = 24 := by
  sorry


end NUMINAMATH_CALUDE_painting_time_for_six_stools_l682_68279


namespace NUMINAMATH_CALUDE_value_of_m_l682_68219

theorem value_of_m (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (expansion : ∀ x, (1 + m * x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6)
  (alternating_sum : a₁ - a₂ + a₃ - a₄ + a₅ - a₆ = -63) :
  m = 3 ∨ m = -1 := by
sorry

end NUMINAMATH_CALUDE_value_of_m_l682_68219


namespace NUMINAMATH_CALUDE_parallel_resistors_l682_68297

theorem parallel_resistors (x y R : ℝ) (hx : x = 4) (hy : y = 5) 
  (hR : 1 / R = 1 / x + 1 / y) : R = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_resistors_l682_68297


namespace NUMINAMATH_CALUDE_sum_f_negative_l682_68218

/-- A monotonically decreasing odd function -/
def MonoDecreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x > f y) ∧ (∀ x, f (-x) = -f x)

/-- Theorem: Sum of function values is negative under given conditions -/
theorem sum_f_negative
  (f : ℝ → ℝ)
  (hf : MonoDecreasingOddFunction f)
  (x₁ x₂ x₃ : ℝ)
  (h₁₂ : x₁ + x₂ > 0)
  (h₂₃ : x₂ + x₃ > 0)
  (h₃₁ : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 :=
sorry

end NUMINAMATH_CALUDE_sum_f_negative_l682_68218


namespace NUMINAMATH_CALUDE_two_card_selections_65_l682_68237

/-- The number of ways to select two different cards from a deck of 65 cards, where the order of selection matters. -/
def two_card_selections (total_cards : ℕ) : ℕ :=
  total_cards * (total_cards - 1)

/-- Theorem stating that selecting two different cards from a deck of 65 cards, where the order matters, can be done in 4160 ways. -/
theorem two_card_selections_65 :
  two_card_selections 65 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_two_card_selections_65_l682_68237


namespace NUMINAMATH_CALUDE_sum_not_odd_l682_68267

theorem sum_not_odd (n m : ℤ) 
  (h1 : Even (n^3 + m^3))
  (h2 : (n^3 + m^3) % 4 = 0) : 
  ¬(Odd (n + m)) := by
sorry

end NUMINAMATH_CALUDE_sum_not_odd_l682_68267


namespace NUMINAMATH_CALUDE_train_speed_calculation_l682_68296

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 90)
  (h2 : bridge_length = 200)
  (h3 : crossing_time = 36) :
  ∃ (speed : ℝ), 
    (speed ≥ 28.9 ∧ speed ≤ 29.1) ∧ 
    speed = (train_length + bridge_length) / crossing_time * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l682_68296


namespace NUMINAMATH_CALUDE_dorothy_age_proof_l682_68285

/-- Given Dorothy's age relationships with her sister, prove Dorothy's current age --/
theorem dorothy_age_proof (dorothy_age sister_age : ℕ) : 
  sister_age = 5 →
  dorothy_age = 3 * sister_age →
  dorothy_age + 5 = 2 * (sister_age + 5) →
  dorothy_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_age_proof_l682_68285


namespace NUMINAMATH_CALUDE_even_function_increasing_interval_l682_68250

def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + (m - 1) * x + 2

theorem even_function_increasing_interval (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  (∃ a : ℝ, ∀ x y : ℝ, x < y ∧ y ≤ 0 → f m x < f m y) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y → f m x > f m y) :=
sorry

end NUMINAMATH_CALUDE_even_function_increasing_interval_l682_68250


namespace NUMINAMATH_CALUDE_num_ways_to_sum_eq_two_pow_n_minus_one_l682_68253

/-- The number of ways to express a natural number as a sum of one or more natural numbers, considering the order of the terms. -/
def numWaysToSum (n : ℕ) : ℕ := 2^(n-1)

/-- Theorem: For any natural number n, the number of ways to express n as a sum of one or more natural numbers, considering the order of the terms, is equal to 2^(n-1). -/
theorem num_ways_to_sum_eq_two_pow_n_minus_one (n : ℕ) : 
  numWaysToSum n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_num_ways_to_sum_eq_two_pow_n_minus_one_l682_68253


namespace NUMINAMATH_CALUDE_max_product_min_sum_l682_68215

theorem max_product_min_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y, x > 0 → y > 0 → x + y = 2 → x * y ≤ a * b → x * y ≤ 1) ∧
  (∀ x y, x > 0 → y > 0 → x + y = 2 → 2/x + 8/y ≥ 2/a + 8/b → 2/x + 8/y ≥ 9) := by
sorry

end NUMINAMATH_CALUDE_max_product_min_sum_l682_68215


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l682_68261

theorem min_value_expression (x : ℝ) : 
  (13 - x) * (8 - x) * (13 + x) * (8 + x) ≥ -2746.25 :=
by
  sorry

theorem min_value_attained : 
  ∃ x : ℝ, (13 - x) * (8 - x) * (13 + x) * (8 + x) = -2746.25 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l682_68261


namespace NUMINAMATH_CALUDE_right_angled_tetrahedron_volume_l682_68273

/-- A tetrahedron with all faces being right-angled triangles and three edges of length s -/
structure RightAngledTetrahedron (s : ℝ) where
  (s_pos : s > 0)
  (all_faces_right_angled : True)  -- This is a placeholder for the condition
  (three_edges_equal : True)  -- This is a placeholder for the condition

/-- The volume of a right-angled tetrahedron -/
noncomputable def volume (t : RightAngledTetrahedron s) : ℝ :=
  (s^3 * Real.sqrt 2) / 12

/-- Theorem stating the volume of a right-angled tetrahedron -/
theorem right_angled_tetrahedron_volume (s : ℝ) (t : RightAngledTetrahedron s) :
  volume t = (s^3 * Real.sqrt 2) / 12 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_tetrahedron_volume_l682_68273


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l682_68248

/-- Calculates the total charge for a taxi trip -/
def taxiCharge (initialFee : ℚ) (additionalChargePerIncrement : ℚ) (incrementDistance : ℚ) (tripDistance : ℚ) : ℚ :=
  initialFee + (tripDistance / incrementDistance) * additionalChargePerIncrement

theorem taxi_charge_calculation :
  let initialFee : ℚ := 235/100
  let additionalChargePerIncrement : ℚ := 35/100
  let incrementDistance : ℚ := 2/5
  let tripDistance : ℚ := 36/10
  taxiCharge initialFee additionalChargePerIncrement incrementDistance tripDistance = 550/100 := by
  sorry

#eval taxiCharge (235/100) (35/100) (2/5) (36/10)

end NUMINAMATH_CALUDE_taxi_charge_calculation_l682_68248


namespace NUMINAMATH_CALUDE_area_S_bounds_l682_68227

theorem area_S_bounds (t : ℝ) (k : ℤ) (h_t : t ≥ 0) (h_k : 2 ≤ k ∧ k ≤ 4) : 
  let T : ℝ := t - ⌊t⌋
  let S : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - T - 1)^2 + (p.2 - k)^2 ≤ (T + 1)^2}
  0 ≤ Real.pi * (T + 1)^2 ∧ Real.pi * (T + 1)^2 ≤ 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_area_S_bounds_l682_68227


namespace NUMINAMATH_CALUDE_largest_prime_factor_l682_68203

theorem largest_prime_factor : 
  let n := 20^3 + 15^4 - 10^5 + 2*25^3
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ n ∧ ∀ q, Nat.Prime q → q ∣ n → q ≤ p ∧ p = 11 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l682_68203


namespace NUMINAMATH_CALUDE_abc_product_l682_68286

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 24 * Real.rpow 3 (1/3))
  (hac : a * c = 40 * Real.rpow 3 (1/3))
  (hbc : b * c = 15 * Real.rpow 3 (1/3)) :
  a * b * c = 120 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l682_68286


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l682_68207

theorem smallest_n_satisfying_condition : ∃ n : ℕ, 
  (n > 1) ∧ 
  (∀ p : ℕ, 2 ≤ p ∧ p ≤ 10 → p ∣ (n^(p-1) - 1)) ∧
  (∀ m : ℕ, 1 < m ∧ m < n → ∃ q : ℕ, 2 ≤ q ∧ q ≤ 10 ∧ ¬(q ∣ (m^(q-1) - 1))) ∧
  n = 2521 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l682_68207


namespace NUMINAMATH_CALUDE_divisibility_implies_B_zero_l682_68280

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number (B : ℕ) : ℕ := 3538084 * 10 + B

theorem divisibility_implies_B_zero (B : ℕ) (h_digit : is_digit B) :
  (∃ k₂ k₄ k₅ k₆ k₈ : ℕ, 
    number B = 2 * k₂ ∧
    number B = 4 * k₄ ∧
    number B = 5 * k₅ ∧
    number B = 6 * k₆ ∧
    number B = 8 * k₈) →
  B = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implies_B_zero_l682_68280


namespace NUMINAMATH_CALUDE_sandy_comic_books_l682_68260

theorem sandy_comic_books (initial : ℕ) (final : ℕ) (bought : ℕ) : 
  initial = 14 →
  final = 13 →
  bought = final - (initial / 2) →
  bought = 6 := by
sorry

end NUMINAMATH_CALUDE_sandy_comic_books_l682_68260


namespace NUMINAMATH_CALUDE_geometric_series_sum_l682_68251

/-- The sum of a geometric series with n terms, first term a, and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the specific geometric series in the problem -/
def specificSum : ℚ :=
  geometricSum (1/4) (1/4) 8

theorem geometric_series_sum :
  specificSum = 65535 / 196608 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l682_68251


namespace NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l682_68212

/-- A regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_150_degrees_has_12_sides :
  ∀ n : ℕ, 
    n > 2 →
    (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l682_68212


namespace NUMINAMATH_CALUDE_polynomial_factorization_l682_68276

theorem polynomial_factorization (x : ℝ) : 
  x^4 - 4*x^3 + 6*x^2 - 4*x + 1 = (x - 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l682_68276


namespace NUMINAMATH_CALUDE_condition_relationship_l682_68235

theorem condition_relationship :
  (∀ x : ℝ, (2 ≤ x ∧ x ≤ 3) → (x < -3 ∨ x ≥ 1)) ∧
  (∃ x : ℝ, (x < -3 ∨ x ≥ 1) ∧ ¬(2 ≤ x ∧ x ≤ 3)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l682_68235


namespace NUMINAMATH_CALUDE_time_for_c_alone_l682_68269

/-- The time required for C to complete the job alone given the work rates of A, B, and C together -/
theorem time_for_c_alone (r_ab r_bc r_ca : ℚ) : 
  r_ab = 1/3 → r_bc = 1/6 → r_ca = 1/4 → (1 : ℚ) / (3/8 - 1/3) = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_time_for_c_alone_l682_68269


namespace NUMINAMATH_CALUDE_kims_sweater_difference_l682_68204

/-- The number of sweaters Kim knit on each day of the week --/
structure WeeklySweaters where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Kim's sweater knitting for the week --/
def kimsSweaterWeek (s : WeeklySweaters) : Prop :=
  s.monday = 8 ∧
  s.tuesday > s.monday ∧
  s.wednesday = s.tuesday - 4 ∧
  s.thursday = s.tuesday - 4 ∧
  s.friday = s.monday / 2 ∧
  s.monday + s.tuesday + s.wednesday + s.thursday + s.friday = 34

theorem kims_sweater_difference (s : WeeklySweaters) 
  (h : kimsSweaterWeek s) : s.tuesday - s.monday = 2 := by
  sorry

end NUMINAMATH_CALUDE_kims_sweater_difference_l682_68204


namespace NUMINAMATH_CALUDE_jeremy_songs_theorem_l682_68298

theorem jeremy_songs_theorem (x y : ℕ) : 
  x % 2 = 0 ∧ 
  9 = 2 * Int.sqrt x - 5 ∧ 
  y = (9 + x) / 2 → 
  9 + x + y = 110 := by
sorry

end NUMINAMATH_CALUDE_jeremy_songs_theorem_l682_68298


namespace NUMINAMATH_CALUDE_rectangular_field_area_l682_68220

theorem rectangular_field_area (length breadth : ℝ) : 
  breadth = 0.6 * length →
  2 * (length + breadth) = 800 →
  length * breadth = 37500 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l682_68220


namespace NUMINAMATH_CALUDE_unknown_percentage_of_250_l682_68231

/-- Given that 28% of 400 plus some percentage of 250 equals 224.5,
    prove that the unknown percentage of 250 is 45%. -/
theorem unknown_percentage_of_250 (p : ℝ) : 
  (0.28 * 400 + p / 100 * 250 = 224.5) → p = 45 :=
by sorry

end NUMINAMATH_CALUDE_unknown_percentage_of_250_l682_68231


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_arcseconds_to_degrees_conversion_negative_fraction_comparison_l682_68249

-- Define the conversion factor from arcseconds to degrees
def arcseconds_to_degrees (x : ℚ) : ℚ := x / 3600

-- Theorem statements
theorem reciprocal_of_negative_three : ((-3)⁻¹ : ℚ) = -1/3 := by sorry

theorem arcseconds_to_degrees_conversion : arcseconds_to_degrees 7200 = 2 := by sorry

theorem negative_fraction_comparison : (-3/4 : ℚ) > -4/5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_arcseconds_to_degrees_conversion_negative_fraction_comparison_l682_68249


namespace NUMINAMATH_CALUDE_triangle_side_length_l682_68244

theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  b = 3 → c = 3 → B = π / 6 → a = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l682_68244


namespace NUMINAMATH_CALUDE_expression_value_l682_68229

theorem expression_value (a b c d x : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c ≠ 0 ∧ d ≠ 0)
  (h3 : c * d = 1) 
  (h4 : |x| = Real.sqrt 7) : 
  (x^2 + (a + b + c * d) * x + Real.sqrt (a + b) + (c * d) ^ (1/3 : ℝ) = 8 + Real.sqrt 7) ∨
  (x^2 + (a + b + c * d) * x + Real.sqrt (a + b) + (c * d) ^ (1/3 : ℝ) = 8 - Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l682_68229


namespace NUMINAMATH_CALUDE_lineGraphMostSuitable_l682_68299

/-- Represents different types of graphs --/
inductive GraphType
  | LineGraph
  | PieChart
  | BarGraph
  | Histogram

/-- Represents the properties of data to be visualized --/
structure DataProperties where
  timeDependent : Bool
  continuous : Bool
  showsTrends : Bool

/-- Determines if a graph type is suitable for given data properties --/
def isSuitable (g : GraphType) (d : DataProperties) : Prop :=
  match g with
  | GraphType.LineGraph => d.timeDependent ∧ d.continuous ∧ d.showsTrends
  | GraphType.PieChart => ¬d.timeDependent
  | GraphType.BarGraph => d.timeDependent
  | GraphType.Histogram => ¬d.timeDependent

/-- The properties of temperature data over a week --/
def temperatureDataProperties : DataProperties :=
  { timeDependent := true
    continuous := true
    showsTrends := true }

/-- Theorem stating that a line graph is the most suitable for temperature data --/
theorem lineGraphMostSuitable :
    ∀ g : GraphType, isSuitable g temperatureDataProperties → g = GraphType.LineGraph :=
  sorry


end NUMINAMATH_CALUDE_lineGraphMostSuitable_l682_68299


namespace NUMINAMATH_CALUDE_wood_carving_shelves_l682_68281

theorem wood_carving_shelves (total_carvings : ℕ) (carvings_per_shelf : ℕ) (shelves_filled : ℕ) : 
  total_carvings = 56 → 
  carvings_per_shelf = 8 → 
  shelves_filled = total_carvings / carvings_per_shelf → 
  shelves_filled = 7 := by
sorry

end NUMINAMATH_CALUDE_wood_carving_shelves_l682_68281


namespace NUMINAMATH_CALUDE_function_identity_l682_68230

theorem function_identity (f : ℕ+ → ℤ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ+, f (m * n) = f m * f n)
  (h3 : ∀ m n : ℕ+, m > n → f m > f n) :
  ∀ n : ℕ+, f n = n := by
sorry

end NUMINAMATH_CALUDE_function_identity_l682_68230


namespace NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l682_68239

theorem least_positive_integer_with_given_remainders :
  ∃ (b : ℕ), b > 0 ∧
    b % 6 = 5 ∧
    b % 7 = 6 ∧
    b % 8 = 7 ∧
    b % 9 = 8 ∧
    (∀ (x : ℕ), x > 0 ∧
      x % 6 = 5 ∧
      x % 7 = 6 ∧
      x % 8 = 7 ∧
      x % 9 = 8 →
      x ≥ b) ∧
    b = 503 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l682_68239


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l682_68252

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l682_68252


namespace NUMINAMATH_CALUDE_power_sum_difference_equals_ten_l682_68236

theorem power_sum_difference_equals_ten : 2^5 + 5^2 / 5^1 - 3^3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_equals_ten_l682_68236


namespace NUMINAMATH_CALUDE_power_division_thirteen_l682_68272

theorem power_division_thirteen : 13^8 / 13^5 = 2197 := by sorry

end NUMINAMATH_CALUDE_power_division_thirteen_l682_68272


namespace NUMINAMATH_CALUDE_susie_house_rooms_l682_68228

/-- The number of rooms in Susie's house -/
def number_of_rooms : ℕ := 6

/-- The time it takes Susie to vacuum the whole house (in hours) -/
def total_vacuum_time : ℚ := 2

/-- The time it takes Susie to vacuum one room (in minutes) -/
def time_per_room : ℕ := 20

/-- Theorem stating that the number of rooms in Susie's house is 6 -/
theorem susie_house_rooms :
  number_of_rooms = (total_vacuum_time * 60) / time_per_room := by
  sorry

end NUMINAMATH_CALUDE_susie_house_rooms_l682_68228


namespace NUMINAMATH_CALUDE_binary_ones_condition_theorem_l682_68201

/-- The number of 1's in the binary representation of a natural number -/
def binary_ones (n : ℕ) : ℕ := sorry

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, binary_ones (f x + y) = binary_ones (f y + x)

/-- The main theorem -/
theorem binary_ones_condition_theorem (f : ℕ → ℕ) :
  satisfies_condition f → ∃ c : ℕ, ∀ x : ℕ, f x = x + c := by sorry

end NUMINAMATH_CALUDE_binary_ones_condition_theorem_l682_68201


namespace NUMINAMATH_CALUDE_cosine_is_even_l682_68224

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem cosine_is_even : IsEven Real.cos := by
  sorry

end NUMINAMATH_CALUDE_cosine_is_even_l682_68224


namespace NUMINAMATH_CALUDE_min_value_function_l682_68262

theorem min_value_function (x : ℝ) (h : x > 3) :
  1 / (x - 3) + x ≥ 5 ∧ (1 / (x - 3) + x = 5 ↔ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l682_68262


namespace NUMINAMATH_CALUDE_class_size_l682_68211

theorem class_size (initial_absent : ℚ) (final_absent : ℚ) (total : ℕ) : 
  initial_absent = 1 / 6 →
  final_absent = 1 / 5 →
  (initial_absent / (1 + initial_absent)) * total + 1 = (final_absent / (1 + final_absent)) * total →
  total = 42 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l682_68211


namespace NUMINAMATH_CALUDE_seven_minus_a_greater_than_b_l682_68264

theorem seven_minus_a_greater_than_b (a b : ℝ) (h : b < a ∧ a < 0) : 7 - a > b := by
  sorry

end NUMINAMATH_CALUDE_seven_minus_a_greater_than_b_l682_68264


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l682_68216

theorem arithmetic_calculation : 8 / 4 - 3 - 9 + 3 * 7 - 2^2 = 7 := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l682_68216


namespace NUMINAMATH_CALUDE_probability_at_least_two_special_items_l682_68223

theorem probability_at_least_two_special_items (total : Nat) (special : Nat) (select : Nat) 
  (h1 : total = 8) (h2 : special = 3) (h3 : select = 4) : 
  (Nat.choose special 2 * Nat.choose (total - special) (select - 2) + 
   Nat.choose special 3 * Nat.choose (total - special) (select - 3)) / 
  Nat.choose total select = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_special_items_l682_68223


namespace NUMINAMATH_CALUDE_union_of_subsets_l682_68291

def A : Set ℕ := {1, 3}
def B : Set ℕ := {1, 2, 3}

theorem union_of_subsets :
  A ⊆ B → A ∪ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_subsets_l682_68291


namespace NUMINAMATH_CALUDE_cos_alpha_value_l682_68225

theorem cos_alpha_value (α : ℝ) (h1 : α ∈ Set.Icc 0 (π / 2)) 
  (h2 : Real.cos (α + π / 6) = 1 / 3) : 
  Real.cos α = (2 * Real.sqrt 2 + Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l682_68225


namespace NUMINAMATH_CALUDE_income_mean_difference_l682_68270

/-- The number of families --/
def num_families : ℕ := 1200

/-- The correct highest income --/
def correct_highest_income : ℕ := 150000

/-- The incorrect highest income --/
def incorrect_highest_income : ℕ := 1500000

/-- The sum of all incomes except the highest --/
def S : ℕ := sorry

/-- The difference between the mean of incorrect data and actual data --/
def mean_difference : ℚ :=
  (S + incorrect_highest_income : ℚ) / num_families -
  (S + correct_highest_income : ℚ) / num_families

theorem income_mean_difference :
  mean_difference = 1125 := by sorry

end NUMINAMATH_CALUDE_income_mean_difference_l682_68270


namespace NUMINAMATH_CALUDE_jeans_discount_percentage_l682_68284

/-- Calculate the discount percentage on jeans --/
theorem jeans_discount_percentage
  (original_price : ℝ)
  (discounted_price_for_three : ℝ)
  (h1 : original_price = 40)
  (h2 : discounted_price_for_three = 112) :
  (original_price * 3 - discounted_price_for_three) / (original_price * 2) = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_jeans_discount_percentage_l682_68284


namespace NUMINAMATH_CALUDE_express_vector_as_linear_combination_l682_68213

/-- Given two vectors a and b in ℝ², express vector c as a linear combination of a and b -/
theorem express_vector_as_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) (hb : b = (1, -1)) (hc : c = (2, 3)) :
  ∃ x y : ℝ, c = x • a + y • b ∧ x = (5 : ℝ) / 2 ∧ y = -(1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_express_vector_as_linear_combination_l682_68213


namespace NUMINAMATH_CALUDE_triangle_angle_B_l682_68247

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_B (t : Triangle) : 
  t.a = 2 * Real.sqrt 3 → 
  t.b = 2 → 
  t.A = π / 3 → 
  t.B = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_B_l682_68247


namespace NUMINAMATH_CALUDE_set_equality_l682_68277

def S : Set ℤ := {x | -3 < x ∧ x < 3}

theorem set_equality : S = {-2, -1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_equality_l682_68277


namespace NUMINAMATH_CALUDE_total_time_is_8_days_l682_68293

-- Define the problem parameters
def plow_rate : ℝ := 10  -- acres per day
def mow_rate : ℝ := 12   -- acres per day
def farmland_area : ℝ := 55  -- acres
def grassland_area : ℝ := 30  -- acres

-- Theorem statement
theorem total_time_is_8_days : 
  (farmland_area / plow_rate) + (grassland_area / mow_rate) = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_8_days_l682_68293


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l682_68243

/-- Given a point P and a line L, this theorem proves that a specific equation
    represents a line passing through P and parallel to L. -/
theorem parallel_line_through_point (x y : ℝ) :
  let P : ℝ × ℝ := (2, Real.sqrt 3)
  let L : ℝ → ℝ → ℝ := fun x y => Real.sqrt 3 * x - y + 2
  let parallel_line : ℝ → ℝ → ℝ := fun x y => Real.sqrt 3 * x - y - Real.sqrt 3
  (parallel_line P.1 P.2 = 0) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ ∀ x y, parallel_line x y = k * L x y) :=
by sorry


end NUMINAMATH_CALUDE_parallel_line_through_point_l682_68243


namespace NUMINAMATH_CALUDE_systematic_sampling_eighth_group_l682_68208

/-- Systematic sampling function -/
def systematicSample (totalSize : ℕ) (sampleSize : ℕ) (thirdGroupNumber : ℕ) (groupNumber : ℕ) : ℕ :=
  let groupCount := totalSize / sampleSize
  let commonDifference := groupCount
  thirdGroupNumber + (groupNumber - 3) * commonDifference

/-- Theorem: In a systematic sampling of 840 employees with a sample size of 42,
    if the number drawn in the third group is 44, then the number drawn in the eighth group is 144. -/
theorem systematic_sampling_eighth_group :
  systematicSample 840 42 44 8 = 144 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_eighth_group_l682_68208


namespace NUMINAMATH_CALUDE_sqrt_product_quotient_l682_68209

theorem sqrt_product_quotient : (Real.sqrt 3 * Real.sqrt 15) / Real.sqrt 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_quotient_l682_68209


namespace NUMINAMATH_CALUDE_tv_price_increase_l682_68233

theorem tv_price_increase (P : ℝ) (x : ℝ) (h1 : P > 0) :
  (0.80 * P + x / 100 * (0.80 * P) = 1.20 * P) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_increase_l682_68233


namespace NUMINAMATH_CALUDE_flash_ace_chase_l682_68222

/-- The problem of Flash catching Ace -/
theorem flash_ace_chase (x y : ℝ) (hx : x > 1) : 
  let ace_speed := 1  -- We can set Ace's speed to 1 without loss of generality
  let flash_east_speed := x * ace_speed
  let flash_west_speed := (x + 1) * ace_speed
  let east_headstart := 2 * y
  let west_headstart := y
  let east_distance := (flash_east_speed * east_headstart) / (flash_east_speed - ace_speed)
  let west_distance := (flash_west_speed * west_headstart) / (flash_west_speed - ace_speed)
  east_distance + west_distance = (2 * x * y) / (x - 1) + ((x + 1) * y) / x :=
by sorry

end NUMINAMATH_CALUDE_flash_ace_chase_l682_68222

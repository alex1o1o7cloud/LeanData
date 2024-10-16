import Mathlib

namespace NUMINAMATH_CALUDE_balloon_count_l3912_391289

theorem balloon_count (friend_balloons : ℕ) (difference : ℕ) : 
  friend_balloons = 5 → difference = 2 → friend_balloons + difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l3912_391289


namespace NUMINAMATH_CALUDE_power_equality_solution_l3912_391223

theorem power_equality_solution : ∃ x : ℝ, x^5 = 5^10 ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_solution_l3912_391223


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l3912_391248

open Real

/-- The function f(x) = √3 sin x - cos x is strictly increasing in the intervals [-π/3 + 2kπ, 2π/3 + 2kπ], where k ∈ ℤ -/
theorem f_strictly_increasing (x : ℝ) :
  ∃ (k : ℤ), x ∈ Set.Icc (-π/3 + 2*π*k) (2*π/3 + 2*π*k) →
  StrictMonoOn (λ x => Real.sqrt 3 * sin x - cos x) (Set.Icc (-π/3 + 2*π*k) (2*π/3 + 2*π*k)) :=
by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l3912_391248


namespace NUMINAMATH_CALUDE_quadratic_properties_l3912_391232

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  f : ℝ → ℝ
  quad : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  axis_sym : ∀ x, f (1 + x) = f (1 - x)
  vertex : f 1 = -4
  table_values : f (-2) = 5 ∧ f (-1) = 0 ∧ f 0 = -3 ∧ f 1 = -4 ∧ f 2 = -3 ∧ f 3 = 0

theorem quadratic_properties (f : QuadraticFunction) :
  (∃ a > 0, ∀ x, f.f x = a * x^2 + f.f 1 - a) ∧
  f.f 4 = 5 ∧
  f.f (-3) > f.f 2 ∧
  {x : ℝ | f.f x < 0} = {x : ℝ | -1 < x ∧ x < 3} ∧
  {x : ℝ | f.f x = 5} = {-2, 4} := by
  sorry


end NUMINAMATH_CALUDE_quadratic_properties_l3912_391232


namespace NUMINAMATH_CALUDE_sum_smallest_largest_prime_l3912_391235

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def primes_in_range (a b : ℕ) : Set ℕ :=
  {n : ℕ | a ≤ n ∧ n ≤ b ∧ is_prime n}

theorem sum_smallest_largest_prime :
  let P := primes_in_range 1 50
  ∃ (p q : ℕ), p ∈ P ∧ q ∈ P ∧
    (∀ x ∈ P, p ≤ x) ∧
    (∀ x ∈ P, x ≤ q) ∧
    p + q = 49 :=
sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_prime_l3912_391235


namespace NUMINAMATH_CALUDE_marble_bag_problem_l3912_391214

theorem marble_bag_problem (T : ℕ) (h1 : T > 12) : 
  (((T - 12 : ℚ) / T) * ((T - 12 : ℚ) / T) = 36 / 49) → T = 84 := by
  sorry

end NUMINAMATH_CALUDE_marble_bag_problem_l3912_391214


namespace NUMINAMATH_CALUDE_continuity_at_nine_l3912_391271

def f (x : ℝ) : ℝ := 4 * x^2 + 4

theorem continuity_at_nine :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 9| < δ → |f x - f 9| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_nine_l3912_391271


namespace NUMINAMATH_CALUDE_sequences_equality_l3912_391204

-- Define A_n
def A (n : ℕ) : ℕ :=
  sorry

-- Define B_n
def B (n : ℕ) : ℕ :=
  sorry

-- Define C_n
def C (n : ℕ) : ℕ :=
  sorry

-- Theorem statement
theorem sequences_equality (n : ℕ) (h : n ≥ 1) : A n = B n ∧ B n = C n :=
  sorry

end NUMINAMATH_CALUDE_sequences_equality_l3912_391204


namespace NUMINAMATH_CALUDE_charity_event_probability_l3912_391265

/-- The probability of selecting a boy for Saturday and a girl for Sunday
    from a group of 2 boys and 2 girls for a two-day event. -/
theorem charity_event_probability :
  let total_people : ℕ := 2 + 2  -- 2 boys + 2 girls
  let total_combinations : ℕ := total_people * (total_people - 1)
  let favorable_outcomes : ℕ := 2 * 2  -- 2 boys for Saturday * 2 girls for Sunday
  (favorable_outcomes : ℚ) / total_combinations = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_charity_event_probability_l3912_391265


namespace NUMINAMATH_CALUDE_ratio_exists_l3912_391254

theorem ratio_exists : ∃ (m n : ℤ), 
  m > 100 ∧ 
  n > 100 ∧ 
  m + n = 300 ∧ 
  3 * n = 2 * m := by
sorry

end NUMINAMATH_CALUDE_ratio_exists_l3912_391254


namespace NUMINAMATH_CALUDE_george_blocks_count_l3912_391246

/-- Calculates the total number of blocks given the number of large boxes, small boxes per large box,
    blocks per small box, and individual blocks outside the boxes. -/
def totalBlocks (largBoxes smallBoxesPerLarge blocksPerSmall individualBlocks : ℕ) : ℕ :=
  largBoxes * smallBoxesPerLarge * blocksPerSmall + individualBlocks

/-- Proves that George has 366 blocks in total -/
theorem george_blocks_count :
  totalBlocks 5 8 9 6 = 366 := by
  sorry

end NUMINAMATH_CALUDE_george_blocks_count_l3912_391246


namespace NUMINAMATH_CALUDE_scout_troop_profit_l3912_391226

/-- Calculates the profit for a scout troop selling candy bars -/
theorem scout_troop_profit (num_bars : ℕ) (buy_rate : ℚ) (sell_rate : ℚ) : 
  num_bars = 1200 → 
  buy_rate = 1/3 → 
  sell_rate = 3/5 → 
  (sell_rate * num_bars : ℚ) - (buy_rate * num_bars : ℚ) = 320 := by
  sorry

#check scout_troop_profit

end NUMINAMATH_CALUDE_scout_troop_profit_l3912_391226


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l3912_391266

/-- The range of k for which the quadratic equation kx^2 - 4x - 2 = 0 has two distinct real roots -/
theorem quadratic_distinct_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 4 * x₁ - 2 = 0 ∧ k * x₂^2 - 4 * x₂ - 2 = 0) ↔ 
  (k > -2 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l3912_391266


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3912_391299

/-- An arithmetic sequence with its first term and sum function -/
structure ArithmeticSequence where
  a₁ : ℤ
  S : ℕ → ℤ

/-- The specific arithmetic sequence from the problem -/
def problemSequence : ArithmeticSequence where
  a₁ := -2012
  S := sorry  -- Definition of S is left as sorry as it's not explicitly given in the conditions

/-- The main theorem to prove -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h : seq.a₁ = -2012)
  (h_sum_diff : seq.S 2012 / 2012 - seq.S 10 / 10 = 2002) :
  seq.S 2017 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3912_391299


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l3912_391249

theorem quadratic_equation_transformation (a b c : ℝ) : 
  (∀ x, a * (x - 1)^2 + b * (x - 1) + c = 2 * x^2 - 3 * x - 1) →
  a = 2 ∧ b = 1 ∧ c = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l3912_391249


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3912_391292

theorem polynomial_factorization (x : ℤ) : 
  (x^3 - x^2 + 2*x - 1) * (x^3 - x - 1) = x^6 - x^5 + x^4 - x^3 - x^2 - x + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3912_391292


namespace NUMINAMATH_CALUDE_boat_distance_downstream_l3912_391295

/-- Calculates the distance traveled downstream by a boat -/
theorem boat_distance_downstream 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 22) 
  (h2 : stream_speed = 5) 
  (h3 : time = 5) : 
  boat_speed + stream_speed * time = 135 := by
  sorry

#check boat_distance_downstream

end NUMINAMATH_CALUDE_boat_distance_downstream_l3912_391295


namespace NUMINAMATH_CALUDE_divisibility_by_7_and_11_l3912_391290

theorem divisibility_by_7_and_11 (n : ℕ) (h : n > 0) :
  (∃ k : ℤ, 3^(2*n+1) + 2^(n+2) = 7*k) ∧
  (∃ m : ℤ, 3^(2*n+2) + 2^(6*n+1) = 11*m) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_7_and_11_l3912_391290


namespace NUMINAMATH_CALUDE_tiling_scheme_proof_l3912_391243

theorem tiling_scheme_proof : 
  let triangle_angle : ℝ := 60
  let hexagon_angle : ℝ := 120
  let num_triangles : ℕ := 4
  let num_hexagons : ℕ := 1
  (num_triangles : ℝ) * triangle_angle + (num_hexagons : ℝ) * hexagon_angle = 360 :=
by sorry

end NUMINAMATH_CALUDE_tiling_scheme_proof_l3912_391243


namespace NUMINAMATH_CALUDE_appliance_sales_prediction_l3912_391227

/-- Represents the sales and cost data for an appliance -/
structure ApplianceData where
  sales : ℕ
  cost : ℕ

/-- Checks if two ApplianceData are inversely proportional -/
def inversely_proportional (a b : ApplianceData) : Prop :=
  a.sales * a.cost = b.sales * b.cost

theorem appliance_sales_prediction
  (blender_initial blender_final microwave_initial microwave_final : ApplianceData)
  (h1 : inversely_proportional blender_initial blender_final)
  (h2 : inversely_proportional microwave_initial microwave_final)
  (h3 : blender_initial.sales = 15)
  (h4 : blender_initial.cost = 300)
  (h5 : blender_final.cost = 450)
  (h6 : microwave_initial.sales = 25)
  (h7 : microwave_initial.cost = 400)
  (h8 : microwave_final.cost = 500) :
  blender_final.sales = 10 ∧ microwave_final.sales = 20 := by
  sorry

end NUMINAMATH_CALUDE_appliance_sales_prediction_l3912_391227


namespace NUMINAMATH_CALUDE_mode_most_relevant_for_market_share_l3912_391233

/-- Represents a clothing model with its sales data -/
structure ClothingModel where
  id : ℕ
  sales : ℕ

/-- Represents a collection of clothing models -/
def ClothingModelData := List ClothingModel

/-- Calculates the mode of a list of natural numbers -/
def mode (l : List ℕ) : Option ℕ :=
  sorry

/-- Calculates the mean of a list of natural numbers -/
def mean (l : List ℕ) : ℚ :=
  sorry

/-- Calculates the median of a list of natural numbers -/
def median (l : List ℕ) : ℚ :=
  sorry

/-- Determines the most relevant statistical measure for market share survey -/
def mostRelevantMeasure (data : ClothingModelData) : String :=
  sorry

theorem mode_most_relevant_for_market_share (data : ClothingModelData) :
  mostRelevantMeasure data = "mode" :=
sorry

end NUMINAMATH_CALUDE_mode_most_relevant_for_market_share_l3912_391233


namespace NUMINAMATH_CALUDE_complete_square_result_l3912_391221

/-- Given a quadratic equation 16x^2 + 32x - 512 = 0, prove that when solved by completing the square
    to the form (x + r)^2 = s, the value of s is 33. -/
theorem complete_square_result (x r s : ℝ) : 
  (16 * x^2 + 32 * x - 512 = 0) →
  ((x + r)^2 = s) →
  (s = 33) := by
sorry

end NUMINAMATH_CALUDE_complete_square_result_l3912_391221


namespace NUMINAMATH_CALUDE_unique_lottery_number_l3912_391262

/-- A five-digit number -/
def FiveDigitNumber := ℕ

/-- Check if a number is a five-digit number -/
def isFiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sumOfDigits (n / 10)

/-- Neighbor's age -/
def neighborAge : ℕ := 45

/-- Theorem: The only five-digit number where the sum of its digits equals 45
    and can be easily solved is 99999 -/
theorem unique_lottery_number :
  ∃! (n : FiveDigitNumber), 
    isFiveDigitNumber n ∧ 
    sumOfDigits n = neighborAge ∧
    (∀ (m : FiveDigitNumber), isFiveDigitNumber m → sumOfDigits m = neighborAge → m = n) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_lottery_number_l3912_391262


namespace NUMINAMATH_CALUDE_max_value_yzx_l3912_391257

theorem max_value_yzx (x y z : ℝ) 
  (h1 : x^2 + z^2 = 1) 
  (h2 : y^2 + 2*y*(x + z) = 6) : 
  ∃ (M : ℝ), M = Real.sqrt 7 ∧ ∀ (x' y' z' : ℝ), 
    x'^2 + z'^2 = 1 → y'^2 + 2*y'*(x' + z') = 6 → 
    y'*(z' - x') ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_yzx_l3912_391257


namespace NUMINAMATH_CALUDE_circular_garden_fence_area_ratio_l3912_391252

theorem circular_garden_fence_area_ratio (r : ℝ) (h : r = 12) : 
  (2 * Real.pi * r) / (Real.pi * r^2) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_fence_area_ratio_l3912_391252


namespace NUMINAMATH_CALUDE_blue_shells_count_l3912_391230

theorem blue_shells_count (total purple pink yellow orange : ℕ) 
  (h_total : total = 65)
  (h_purple : purple = 13)
  (h_pink : pink = 8)
  (h_yellow : yellow = 18)
  (h_orange : orange = 14) :
  total - (purple + pink + yellow + orange) = 12 := by
  sorry

end NUMINAMATH_CALUDE_blue_shells_count_l3912_391230


namespace NUMINAMATH_CALUDE_radio_contest_winner_l3912_391237

theorem radio_contest_winner (n : ℕ) : 
  n > 1 ∧ 
  n < 35 ∧ 
  35 % n = 0 ∧ 
  35 % 7 = 0 ∧ 
  n ≠ 7 → 
  n = 5 := by sorry

end NUMINAMATH_CALUDE_radio_contest_winner_l3912_391237


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l3912_391200

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ+, n ∣ (2^n.val - 1) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l3912_391200


namespace NUMINAMATH_CALUDE_count_rectangular_subsets_5x5_l3912_391294

/-- The number of ways to select a rectangular subset in a 5x5 grid -/
def rectangular_subsets_5x5 : ℕ := 225

/-- A proof that there are 225 ways to select a rectangular subset in a 5x5 grid -/
theorem count_rectangular_subsets_5x5 : rectangular_subsets_5x5 = 225 := by
  sorry

end NUMINAMATH_CALUDE_count_rectangular_subsets_5x5_l3912_391294


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l3912_391231

theorem smallest_undefined_value (x : ℝ) :
  let f := fun x => (x - 3) / (6 * x^2 - 47 * x + 7)
  let smallest_x := (47 - Real.sqrt 2041) / 12
  (∀ y < smallest_x, f y ≠ 0⁻¹) ∧
  (f smallest_x = 0⁻¹) :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l3912_391231


namespace NUMINAMATH_CALUDE_subsets_containing_five_and_six_l3912_391213

def S : Finset Nat := {1, 2, 3, 4, 5, 6}

theorem subsets_containing_five_and_six :
  (Finset.filter (λ s : Finset Nat => 5 ∈ s ∧ 6 ∈ s) (Finset.powerset S)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_subsets_containing_five_and_six_l3912_391213


namespace NUMINAMATH_CALUDE_hyperbola_min_distance_hyperbola_min_distance_achieved_l3912_391276

theorem hyperbola_min_distance (x y : ℝ) : 
  (x^2 / 8) - (y^2 / 4) = 1 → |x - y| ≥ 2 :=
by sorry

theorem hyperbola_min_distance_achieved : 
  ∃ (x y : ℝ), (x^2 / 8) - (y^2 / 4) = 1 ∧ |x - y| = 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_min_distance_hyperbola_min_distance_achieved_l3912_391276


namespace NUMINAMATH_CALUDE_M_is_power_of_three_l3912_391234

/-- Arithmetic sequence with a_n = n -/
def arithmetic_seq (n : ℕ) : ℕ := n

/-- Sequence of t_n values -/
def t_seq : ℕ → ℕ
  | 0 => 0
  | n+1 => (3^(n+1) - 1) / 2

/-- M_n is the sum of terms from (t_{n-1}+1)th to t_n th term -/
def M (n : ℕ) : ℕ :=
  let a := t_seq (n-1)
  let b := t_seq n
  (b * (b + 1) - a * (a + 1)) / 2

/-- Main theorem: M_n = 3^(2n-2) for all n ∈ ℕ -/
theorem M_is_power_of_three (n : ℕ) : M n = 3^(2*n - 2) := by
  sorry


end NUMINAMATH_CALUDE_M_is_power_of_three_l3912_391234


namespace NUMINAMATH_CALUDE_trip_length_proof_l3912_391278

/-- Represents the total length of the trip in miles -/
def total_distance : ℝ := 95

/-- Represents the distance traveled on battery -/
def battery_distance : ℝ := 30

/-- Represents the distance traveled on first gasoline mode -/
def first_gas_distance : ℝ := 70

/-- Represents the rate of gasoline consumption in the first gasoline mode -/
def first_gas_rate : ℝ := 0.03

/-- Represents the rate of gasoline consumption in the second gasoline mode -/
def second_gas_rate : ℝ := 0.04

/-- Represents the overall average miles per gallon -/
def average_mpg : ℝ := 50

theorem trip_length_proof :
  total_distance = battery_distance + first_gas_distance +
    (total_distance - battery_distance - first_gas_distance) ∧
  (first_gas_rate * first_gas_distance +
   second_gas_rate * (total_distance - battery_distance - first_gas_distance)) *
    average_mpg = total_distance :=
by sorry

end NUMINAMATH_CALUDE_trip_length_proof_l3912_391278


namespace NUMINAMATH_CALUDE_angle_C_is_30_degrees_l3912_391247

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle measure function
def angle_measure (T : Triangle) (vertex : ℕ) : ℝ := sorry

-- Define the side length function
def side_length (T : Triangle) (side : ℕ) : ℝ := sorry

theorem angle_C_is_30_degrees (T : Triangle) :
  angle_measure T 1 = π / 4 →  -- ∠A = 45°
  side_length T 1 = Real.sqrt 2 →  -- AB = √2
  side_length T 2 = 2 →  -- BC = 2
  angle_measure T 3 = π / 6  -- ∠C = 30°
  := by sorry

end NUMINAMATH_CALUDE_angle_C_is_30_degrees_l3912_391247


namespace NUMINAMATH_CALUDE_total_students_at_competition_l3912_391291

/-- The number of students from each school at a science fair competition --/
structure SchoolAttendance where
  quantum : ℕ
  schrodinger : ℕ
  einstein : ℕ
  newton : ℕ
  galileo : ℕ
  pascal : ℕ
  faraday : ℕ

/-- The conditions of the science fair competition --/
def scienceFairConditions (s : SchoolAttendance) : Prop :=
  s.quantum = 90 ∧
  s.schrodinger = (2 * s.quantum) / 3 ∧
  s.einstein = (4 * s.schrodinger) / 9 ∧
  s.newton = (5 * s.einstein) / 12 ∧
  s.galileo = (11 * s.newton) / 20 ∧
  s.pascal = (13 * s.galileo) / 50 ∧
  s.faraday = 4 * (s.quantum + s.schrodinger + s.einstein + s.newton + s.galileo + s.pascal)

/-- The theorem stating the total number of students at the competition --/
theorem total_students_at_competition (s : SchoolAttendance) 
  (h : scienceFairConditions s) : 
  s.quantum + s.schrodinger + s.einstein + s.newton + s.galileo + s.pascal + s.faraday = 980 := by
  sorry

end NUMINAMATH_CALUDE_total_students_at_competition_l3912_391291


namespace NUMINAMATH_CALUDE_candidates_appeared_l3912_391263

theorem candidates_appeared (x : ℝ) 
  (h1 : 0.07 * x = 0.06 * x + 82) : x = 8200 := by
  sorry

end NUMINAMATH_CALUDE_candidates_appeared_l3912_391263


namespace NUMINAMATH_CALUDE_no_solution_for_divisibility_l3912_391201

theorem no_solution_for_divisibility (n : ℕ) (hn : n ≥ 1) : ¬(9 ∣ (7^n + n^3)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_divisibility_l3912_391201


namespace NUMINAMATH_CALUDE_power_calculation_l3912_391283

theorem power_calculation : 16^16 * 2^10 / 4^22 = 2^30 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l3912_391283


namespace NUMINAMATH_CALUDE_sum_of_squares_verify_sum_of_squares_l3912_391284

theorem sum_of_squares : ℕ → Prop
  | 1009 => 1009 = 15^2 + 28^2
  | 2018 => 2018 = 43^2 + 13^2
  | _ => True

theorem verify_sum_of_squares :
  sum_of_squares 1009 ∧ sum_of_squares 2018 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_verify_sum_of_squares_l3912_391284


namespace NUMINAMATH_CALUDE_min_bilingual_students_l3912_391293

theorem min_bilingual_students (total : ℕ) (hindi : ℕ) (english : ℕ) 
  (h_total : total = 40)
  (h_hindi : hindi = 30)
  (h_english : english = 20) :
  ∃ (both : ℕ), both ≥ hindi + english - total ∧ 
    ∀ (x : ℕ), x ≥ hindi + english - total → x ≥ both :=
by sorry

end NUMINAMATH_CALUDE_min_bilingual_students_l3912_391293


namespace NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l3912_391224

theorem rectangle_cylinder_volume_ratio :
  let rectangle_length : ℝ := 10
  let rectangle_width : ℝ := 6
  let cylinder_A_height : ℝ := rectangle_width
  let cylinder_A_circumference : ℝ := rectangle_length
  let cylinder_B_height : ℝ := rectangle_length
  let cylinder_B_circumference : ℝ := rectangle_width
  let cylinder_A_volume : ℝ := (cylinder_A_circumference^2 * cylinder_A_height) / (4 * π)
  let cylinder_B_volume : ℝ := (cylinder_B_circumference^2 * cylinder_B_height) / (4 * π)
  max cylinder_A_volume cylinder_B_volume / min cylinder_A_volume cylinder_B_volume = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l3912_391224


namespace NUMINAMATH_CALUDE_power_product_equals_negative_one_l3912_391286

theorem power_product_equals_negative_one : 
  (4 : ℝ)^7 * (-0.25 : ℝ)^7 = -1 := by sorry

end NUMINAMATH_CALUDE_power_product_equals_negative_one_l3912_391286


namespace NUMINAMATH_CALUDE_sum_of_squares_l3912_391206

theorem sum_of_squares (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 4) 
  (h2 : a/x + b/y + c/z = 3) : 
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3912_391206


namespace NUMINAMATH_CALUDE_fraction_power_six_l3912_391255

theorem fraction_power_six :
  (5 / 3 : ℚ) ^ 6 = 15625 / 729 := by sorry

end NUMINAMATH_CALUDE_fraction_power_six_l3912_391255


namespace NUMINAMATH_CALUDE_expression_equivalence_l3912_391274

theorem expression_equivalence (y : ℝ) (Q : ℝ) (h : 5 * (3 * y + 7 * Real.pi) = Q) :
  10 * (6 * y + 14 * Real.pi + 3) = 4 * Q + 30 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3912_391274


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3912_391203

def M : Set ℝ := {x | x^2 - 3*x = 0}
def N : Set ℝ := {x | x^2 - 5*x + 6 = 0}

theorem union_of_M_and_N : M ∪ N = {0, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3912_391203


namespace NUMINAMATH_CALUDE_faye_apps_left_l3912_391229

/-- The number of apps left after deletion -/
def apps_left (initial : ℕ) (deleted : ℕ) : ℕ :=
  initial - deleted

/-- Theorem stating that Faye has 4 apps left -/
theorem faye_apps_left : apps_left 12 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_faye_apps_left_l3912_391229


namespace NUMINAMATH_CALUDE_quadrilaterals_on_circle_l3912_391260

/-- The number of distinct convex quadrilaterals formed from points on a circle -/
theorem quadrilaterals_on_circle (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 4) :
  (Nat.choose n k) = 495 := by
  sorry

end NUMINAMATH_CALUDE_quadrilaterals_on_circle_l3912_391260


namespace NUMINAMATH_CALUDE_cookies_left_after_three_days_l3912_391205

/-- Calculates the number of cookies left after a specified number of days -/
def cookies_left (initial_cookies : ℕ) (first_day_consumption : ℕ) (julie_daily : ℕ) (matt_daily : ℕ) (days : ℕ) : ℕ :=
  initial_cookies - (first_day_consumption + (julie_daily + matt_daily) * days)

/-- Theorem stating the number of cookies left after 3 days -/
theorem cookies_left_after_three_days : 
  cookies_left 32 9 2 3 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_after_three_days_l3912_391205


namespace NUMINAMATH_CALUDE_inequality_proof_l3912_391239

theorem inequality_proof (n : ℕ+) : (2 * n.val ^ 2 + 3 * n.val + 1) ^ n.val ≥ 6 ^ n.val * (n.val.factorial) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3912_391239


namespace NUMINAMATH_CALUDE_bill_pot_stacking_l3912_391212

/-- Calculates the total number of pots that can be stacked given the vertical stack size, 
    number of stacks per shelf, and number of shelves. -/
def total_pots (vertical_stack : ℕ) (stacks_per_shelf : ℕ) (num_shelves : ℕ) : ℕ :=
  vertical_stack * stacks_per_shelf * num_shelves

/-- Proves that given the specific conditions of Bill's pot stacking problem, 
    the total number of pots is 60. -/
theorem bill_pot_stacking : total_pots 5 3 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_bill_pot_stacking_l3912_391212


namespace NUMINAMATH_CALUDE_rebus_solution_l3912_391268

theorem rebus_solution :
  ∃! (A B G D V : ℕ),
    A * B + 8 = 3 * B ∧
    G * D + B = V ∧
    G * B + 3 = A * D ∧
    A = 2 ∧ B = 7 ∧ G = 1 ∧ D = 0 ∧ V = 15 := by
  sorry

end NUMINAMATH_CALUDE_rebus_solution_l3912_391268


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l3912_391282

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

end NUMINAMATH_CALUDE_taxi_fare_calculation_l3912_391282


namespace NUMINAMATH_CALUDE_square_difference_ratio_l3912_391242

theorem square_difference_ratio : (1722^2 - 1715^2) / (1730^2 - 1705^2) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_ratio_l3912_391242


namespace NUMINAMATH_CALUDE_debby_water_bottles_l3912_391280

/-- The number of water bottles Debby drank in one day -/
def bottles_drank : ℕ := 144

/-- The number of water bottles Debby has left -/
def bottles_left : ℕ := 157

/-- The initial number of water bottles Debby bought -/
def initial_bottles : ℕ := bottles_drank + bottles_left

theorem debby_water_bottles : initial_bottles = 301 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l3912_391280


namespace NUMINAMATH_CALUDE_max_profit_at_initial_price_l3912_391270

/-- Represents the daily profit function for a clothing store -/
def daily_profit (x : ℝ) : ℝ :=
  (30 - x) * (30 + x)

/-- Theorem stating that the maximum daily profit occurs at the initial selling price -/
theorem max_profit_at_initial_price :
  ∀ x : ℝ, daily_profit 0 ≥ daily_profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_initial_price_l3912_391270


namespace NUMINAMATH_CALUDE_total_shells_l3912_391279

theorem total_shells (morning_shells afternoon_shells : ℕ) 
  (h1 : morning_shells = 292) 
  (h2 : afternoon_shells = 324) : 
  morning_shells + afternoon_shells = 616 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_l3912_391279


namespace NUMINAMATH_CALUDE_solve_salary_problem_l3912_391209

def salary_problem (S : ℝ) : Prop :=
  let rent := (2/5) * S
  let food := (3/10) * S
  let conveyance := (1/8) * S
  (food + conveyance = 3400) →
  (S - (rent + food + conveyance) = 1400)

theorem solve_salary_problem :
  ∃ S : ℝ, salary_problem S :=
sorry

end NUMINAMATH_CALUDE_solve_salary_problem_l3912_391209


namespace NUMINAMATH_CALUDE_cricketer_average_score_l3912_391273

theorem cricketer_average_score 
  (initial_innings : ℕ) 
  (last_inning_score : ℕ) 
  (average_increase : ℕ) 
  (h1 : initial_innings = 18) 
  (h2 : last_inning_score = 95) 
  (h3 : average_increase = 4) :
  (initial_innings * (average_increase + (last_inning_score / (initial_innings + 1))) + last_inning_score) / (initial_innings + 1) = 23 :=
by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l3912_391273


namespace NUMINAMATH_CALUDE_quadrilateral_is_trapezium_l3912_391258

/-- A quadrilateral with angles x°, 5x°, 2x°, and 4x° is a trapezium -/
theorem quadrilateral_is_trapezium (x : ℝ) 
  (angle_sum : x + 5*x + 2*x + 4*x = 360) : 
  ∃ (a b c d : ℝ), 
    a + b + c + d = 360 ∧ 
    a + c = 180 ∧
    (a = x ∨ a = 5*x ∨ a = 2*x ∨ a = 4*x) ∧
    (b = x ∨ b = 5*x ∨ b = 2*x ∨ b = 4*x) ∧
    (c = x ∨ c = 5*x ∨ c = 2*x ∨ c = 4*x) ∧
    (d = x ∨ d = 5*x ∨ d = 2*x ∨ d = 4*x) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_is_trapezium_l3912_391258


namespace NUMINAMATH_CALUDE_sqrt3_times_sqrt10_minus_sqrt3_bounds_l3912_391251

theorem sqrt3_times_sqrt10_minus_sqrt3_bounds :
  2 < Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3) ∧
  Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3) < 3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt3_times_sqrt10_minus_sqrt3_bounds_l3912_391251


namespace NUMINAMATH_CALUDE_no_real_roots_for_sqrt_equation_l3912_391297

theorem no_real_roots_for_sqrt_equation :
  ¬ ∃ x : ℝ, Real.sqrt (x + 4) - Real.sqrt (x - 3) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_for_sqrt_equation_l3912_391297


namespace NUMINAMATH_CALUDE_h_negative_a_equals_negative_two_l3912_391261

-- Define the functions
variable (f g h : ℝ → ℝ)
variable (a : ℝ)

-- Define the properties of the functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem h_negative_a_equals_negative_two 
  (hf_even : is_even f)
  (hg_odd : is_odd g)
  (hf_a : f a = 2)
  (hg_a : g a = 3)
  (hh : ∀ x, h x = f x + g x - 1) :
  h (-a) = -2 := by sorry

end NUMINAMATH_CALUDE_h_negative_a_equals_negative_two_l3912_391261


namespace NUMINAMATH_CALUDE_sequence_sum_99_100_l3912_391219

def sequence_term (n : ℕ) : ℚ :=
  let group := (n.sqrt : ℕ)
  let position := n - (group - 1) * group
  ↑(group + 1 - position) / position

theorem sequence_sum_99_100 : 
  sequence_term 99 + sequence_term 100 = 37 / 24 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_99_100_l3912_391219


namespace NUMINAMATH_CALUDE_least_four_digit_divisible_by_digits_l3912_391241

/-- A function that checks if a number is a four-digit positive integer with all different digits -/
def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ 
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10)

/-- A function that checks if a number is divisible by all of its digits -/
def divisible_by_digits (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 4 → (n % ((n / 10^i) % 10) = 0 ∨ (n / 10^i) % 10 = 0)

/-- The main theorem stating that 1236 is the least number satisfying the conditions -/
theorem least_four_digit_divisible_by_digits :
  is_valid_number 1236 ∧ 
  divisible_by_digits 1236 ∧
  (∀ m : ℕ, m < 1236 → ¬(is_valid_number m ∧ divisible_by_digits m)) :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_divisible_by_digits_l3912_391241


namespace NUMINAMATH_CALUDE_power_difference_lower_bound_l3912_391285

theorem power_difference_lower_bound 
  (m n : ℕ) 
  (h1 : m > 1) 
  (h2 : 2^(2*m + 1) - n^2 ≥ 0) : 
  2^(2*m + 1) - n^2 ≥ 7 := by
sorry

end NUMINAMATH_CALUDE_power_difference_lower_bound_l3912_391285


namespace NUMINAMATH_CALUDE_total_increase_in_two_centuries_l3912_391298

/-- Represents the increase in height per decade in meters -/
def increase_per_decade : ℝ := 90

/-- Represents the number of decades in 2 centuries -/
def decades_in_two_centuries : ℕ := 20

/-- Represents the total increase in height over 2 centuries in meters -/
def total_increase : ℝ := increase_per_decade * decades_in_two_centuries

/-- Theorem stating that the total increase in height over 2 centuries is 1800 meters -/
theorem total_increase_in_two_centuries : total_increase = 1800 := by
  sorry

end NUMINAMATH_CALUDE_total_increase_in_two_centuries_l3912_391298


namespace NUMINAMATH_CALUDE_equation_real_solution_l3912_391202

theorem equation_real_solution (x : ℝ) :
  (∀ y : ℝ, ∃ z : ℝ, x^2 + y^2 + z^2 + 2*x*y*z = 1) ↔ (x = 1 ∨ x = -1) :=
sorry

end NUMINAMATH_CALUDE_equation_real_solution_l3912_391202


namespace NUMINAMATH_CALUDE_refrigerator_is_right_prism_other_objects_not_right_prisms_l3912_391287

-- Define the properties of a right prism
structure RightPrism :=
  (has_congruent_polygonal_bases : Bool)
  (has_rectangular_lateral_faces : Bool)

-- Define the properties of a refrigerator
structure Refrigerator :=
  (shape : RightPrism)

-- Theorem stating that a refrigerator can be modeled as a right prism
theorem refrigerator_is_right_prism (r : Refrigerator) : 
  r.shape.has_congruent_polygonal_bases ∧ r.shape.has_rectangular_lateral_faces := by
  sorry

-- Define other objects for comparison
structure Basketball :=
  (is_spherical : Bool)

structure Shuttlecock :=
  (has_conical_shape : Bool)

structure Thermos :=
  (is_cylindrical : Bool)

-- Theorem stating that other objects are not right prisms
theorem other_objects_not_right_prisms : 
  ∀ (b : Basketball) (s : Shuttlecock) (t : Thermos),
  ¬(∃ (rp : RightPrism), rp.has_congruent_polygonal_bases ∧ rp.has_rectangular_lateral_faces) := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_is_right_prism_other_objects_not_right_prisms_l3912_391287


namespace NUMINAMATH_CALUDE_square_of_1009_l3912_391250

theorem square_of_1009 : 1009 * 1009 = 1018081 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1009_l3912_391250


namespace NUMINAMATH_CALUDE_min_inequality_solution_set_l3912_391253

open Set Real

theorem min_inequality_solution_set (x : ℝ) (hx : x ≠ 0) :
  min 4 (x + 4 / x) ≥ 8 * min x (1 / x) ↔ x ∈ Iic 0 ∪ Ioo 0 (1 / 2) ∪ Ici 2 :=
sorry

end NUMINAMATH_CALUDE_min_inequality_solution_set_l3912_391253


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_pi_3_l3912_391215

open Real

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := sin (ω * x + φ) + 1

theorem sin_2alpha_plus_pi_3 (ω φ α : ℝ) :
  ω > 0 →
  0 ≤ φ ∧ φ ≤ π/2 →
  (∀ x : ℝ, f ω φ (x + π/ω) = f ω φ x) →
  f ω φ (π/3) = 2 →
  f ω φ α = 8/5 →
  π/3 < α ∧ α < 5*π/6 →
  sin (2*α + π/3) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_pi_3_l3912_391215


namespace NUMINAMATH_CALUDE_satellite_sensor_ratio_l3912_391269

theorem satellite_sensor_ratio (total_sensors : ℝ) (non_upgraded_per_unit : ℝ) : 
  total_sensors > 0 →
  non_upgraded_per_unit ≥ 0 →
  (24 * non_upgraded_per_unit + 0.25 * total_sensors = total_sensors) →
  (non_upgraded_per_unit / (0.25 * total_sensors) = 1 / 8) :=
by
  sorry

end NUMINAMATH_CALUDE_satellite_sensor_ratio_l3912_391269


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3912_391245

/-- The equation of a hyperbola passing through a specific point with its asymptote tangent to a circle -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (16 / a^2 - 4 / b^2 = 1) →  -- Hyperbola passes through (4, 2)
  (|2 * Real.sqrt 2 * b| / Real.sqrt (b^2 + a^2) = Real.sqrt (8/3)) →  -- Asymptote tangent to circle
  (∀ x y : ℝ, x^2 / 8 - y^2 / 4 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3912_391245


namespace NUMINAMATH_CALUDE_x_power_ln_ln_minus_ln_x_power_ln_l3912_391264

theorem x_power_ln_ln_minus_ln_x_power_ln (x : ℝ) (h : x > 1) :
  x^(Real.log (Real.log x)) - (Real.log x)^(Real.log x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_power_ln_ln_minus_ln_x_power_ln_l3912_391264


namespace NUMINAMATH_CALUDE_inequality_implies_range_l3912_391296

theorem inequality_implies_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |2*x + 2| ≥ a^2 + (1/2)*a + 2) → 
  -1/2 ≤ a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l3912_391296


namespace NUMINAMATH_CALUDE_circle_tangent_trajectory_l3912_391275

-- Define the circle M
def CircleM (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 10

-- Define the line on which the center of M lies
def CenterLine (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define points A, B, C, and D
def PointA : ℝ × ℝ := (-5, 0)
def PointB : ℝ × ℝ := (1, 0)
def PointC : ℝ × ℝ := (1, 2)
def PointD : ℝ × ℝ := (-3, 4)

-- Define the tangent line through C
def TangentLineC (x y : ℝ) : Prop := 3*x + y - 5 = 0

-- Define the trajectory of Q
def TrajectoryQ (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 10

theorem circle_tangent_trajectory :
  -- The center of M is on the given line
  (∃ x y, CircleM x y ∧ CenterLine x y) ∧
  -- M passes through A and B
  (CircleM PointA.1 PointA.2 ∧ CircleM PointB.1 PointB.2) →
  -- 1. Equation of circle M is correct
  (∀ x y, CircleM x y ↔ (x + 2)^2 + (y - 1)^2 = 10) ∧
  -- 2. Equation of tangent line through C is correct
  (∀ x y, TangentLineC x y ↔ 3*x + y - 5 = 0) ∧
  -- 3. Trajectory equation of Q is correct
  (∀ x y, (TrajectoryQ x y ∧ ¬((x, y) = (-1, 8) ∨ (x, y) = (-3, 4))) ↔
    (∃ x₀ y₀, CircleM x₀ y₀ ∧ 
      x = (-5 + x₀ + 3)/2 ∧ 
      y = (y₀ + 4)/2 ∧
      ¬((x, y) = (-1, 8) ∨ (x, y) = (-3, 4)))) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_trajectory_l3912_391275


namespace NUMINAMATH_CALUDE_new_triangle_is_acute_l3912_391244

-- Define a right triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2
  positive : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the new triangle after increasing each side by x
def NewTriangle (t : RightTriangle) (x : ℝ) : Prop :=
  ∀ (h : 0 < x),
    let new_a := t.a + x
    let new_b := t.b + x
    let new_c := t.c + x
    (new_a^2 + new_b^2 - new_c^2) / (2 * new_a * new_b) > 0

-- Theorem statement
theorem new_triangle_is_acute (t : RightTriangle) :
  ∀ x, NewTriangle t x :=
sorry

end NUMINAMATH_CALUDE_new_triangle_is_acute_l3912_391244


namespace NUMINAMATH_CALUDE_cos_36_degrees_l3912_391272

theorem cos_36_degrees (x y : ℝ) : 
  x = Real.cos (36 * π / 180) →
  y = Real.cos (72 * π / 180) →
  y = 2 * x^2 - 1 →
  x = 2 * y^2 - 1 →
  x = (1 + Real.sqrt 5) / 4 := by
sorry

end NUMINAMATH_CALUDE_cos_36_degrees_l3912_391272


namespace NUMINAMATH_CALUDE_sin_2theta_value_l3912_391288

theorem sin_2theta_value (θ : ℝ) 
  (h : ∑' n, (Real.sin θ) ^ (2 * n) = 3) : 
  Real.sin (2 * θ) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l3912_391288


namespace NUMINAMATH_CALUDE_multiplier_is_three_l3912_391240

theorem multiplier_is_three (n : ℝ) (h1 : 3 * n = (26 - n) + 14) (h2 : n = 10) : 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_is_three_l3912_391240


namespace NUMINAMATH_CALUDE_tree_subgraph_existence_l3912_391238

-- Define a tree
def is_tree (T : SimpleGraph V) : Prop := sorry

-- Define the order of a graph
def graph_order (G : SimpleGraph V) : ℕ := sorry

-- Define the minimum degree of a graph
def min_degree (G : SimpleGraph V) : ℕ := sorry

-- Define graph isomorphism
def is_isomorphic_subgraph (T G : SimpleGraph V) : Prop := sorry

theorem tree_subgraph_existence 
  {V : Type*} (T G : SimpleGraph V) :
  is_tree T →
  min_degree G ≥ graph_order T - 1 →
  is_isomorphic_subgraph T G :=
by sorry

end NUMINAMATH_CALUDE_tree_subgraph_existence_l3912_391238


namespace NUMINAMATH_CALUDE_other_communities_count_l3912_391222

theorem other_communities_count (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ)
  (h_total : total = 650)
  (h_muslim : muslim_percent = 44/100)
  (h_hindu : hindu_percent = 28/100)
  (h_sikh : sikh_percent = 10/100) :
  ⌊(1 - (muslim_percent + hindu_percent + sikh_percent)) * total⌋ = 117 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l3912_391222


namespace NUMINAMATH_CALUDE_john_caffeine_consumption_l3912_391210

/-- The amount of caffeine John consumed from two energy drinks and a caffeine pill -/
theorem john_caffeine_consumption (first_drink_oz : ℝ) (first_drink_caffeine : ℝ) 
  (second_drink_oz : ℝ) (second_drink_caffeine_multiplier : ℝ) :
  first_drink_oz = 12 ∧ 
  first_drink_caffeine = 250 ∧ 
  second_drink_oz = 2 ∧ 
  second_drink_caffeine_multiplier = 3 →
  (let first_drink_caffeine_per_oz := first_drink_caffeine / first_drink_oz
   let second_drink_caffeine_per_oz := first_drink_caffeine_per_oz * second_drink_caffeine_multiplier
   let second_drink_caffeine := second_drink_caffeine_per_oz * second_drink_oz
   let total_drinks_caffeine := first_drink_caffeine + second_drink_caffeine
   let pill_caffeine := total_drinks_caffeine
   let total_caffeine := total_drinks_caffeine + pill_caffeine
   total_caffeine = 750) :=
by sorry

end NUMINAMATH_CALUDE_john_caffeine_consumption_l3912_391210


namespace NUMINAMATH_CALUDE_meeting_percentage_is_25_percent_l3912_391259

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 8 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 30

/-- Calculates the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Represents the percentage of the work day spent in meetings -/
def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100

theorem meeting_percentage_is_25_percent :
  meeting_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_meeting_percentage_is_25_percent_l3912_391259


namespace NUMINAMATH_CALUDE_nth_power_divisibility_l3912_391208

theorem nth_power_divisibility (b n : ℕ) (h1 : b > 1) (h2 : n > 1)
  (h3 : ∀ k : ℕ, k > 1 → ∃ a_k : ℕ, k ∣ (b - a_k^n)) :
  ∃ A : ℕ, b = A^n := by sorry

end NUMINAMATH_CALUDE_nth_power_divisibility_l3912_391208


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_and_3_l3912_391220

theorem smallest_five_digit_divisible_by_53_and_3 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit positive integer
  n % 53 = 0 ∧                -- divisible by 53
  n % 3 = 0 ∧                 -- divisible by 3
  n = 10062 ∧                 -- the number is 10062
  ∀ m : ℕ, (m ≥ 10000 ∧ m < 100000 ∧ m % 53 = 0 ∧ m % 3 = 0) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_and_3_l3912_391220


namespace NUMINAMATH_CALUDE_complex_magnitude_l3912_391217

theorem complex_magnitude (z : ℂ) : z = -2 + I → Complex.abs (z + 1) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3912_391217


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l3912_391216

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l3912_391216


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3912_391256

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem complement_M_intersect_N :
  (U \ M) ∩ N = {-3, -4} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3912_391256


namespace NUMINAMATH_CALUDE_power_multiplication_l3912_391281

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3912_391281


namespace NUMINAMATH_CALUDE_problem_solution_l3912_391228

theorem problem_solution :
  (∀ x : ℝ, x^2 - x + 1 ≥ 0) ∧
  (∀ p q : Prop, (¬(p ∨ q) → (¬p ∧ ¬q))) ∧
  ((∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧
   (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ x ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3912_391228


namespace NUMINAMATH_CALUDE_center_shade_ratio_l3912_391211

/-- Represents a square grid -/
structure SquareGrid (n : ℕ) where
  size : ℕ
  total_area : ℝ
  cell_area : ℝ
  h_size : size = n
  h_cell_area : cell_area = total_area / (n^2 : ℝ)

/-- Represents a shaded region in the center of the grid -/
structure CenterShade (grid : SquareGrid 5) where
  area : ℝ
  h_area : area = 4 * (grid.cell_area / 2)

/-- The theorem stating the ratio of the shaded area to the total area -/
theorem center_shade_ratio (grid : SquareGrid 5) (shade : CenterShade grid) :
  shade.area / grid.total_area = 2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_center_shade_ratio_l3912_391211


namespace NUMINAMATH_CALUDE_max_value_implies_t_equals_one_l3912_391207

theorem max_value_implies_t_equals_one (t : ℝ) :
  (∀ x ∈ Set.Icc 0 3, |x^2 - 2*x - t| ≤ 2) →
  (∃ x ∈ Set.Icc 0 3, |x^2 - 2*x - t| = 2) →
  t = 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_t_equals_one_l3912_391207


namespace NUMINAMATH_CALUDE_range_of_a_l3912_391218

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a (a : ℝ) : M ⊆ N a → a ∈ Set.Iic (-1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3912_391218


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l3912_391225

theorem cyclist_speed_problem (v : ℝ) :
  v > 0 →
  (20 : ℝ) / (9 / v + 11 / 9) = 9.8019801980198 →
  ∃ ε > 0, |v - 11.03| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l3912_391225


namespace NUMINAMATH_CALUDE_hallies_net_earnings_l3912_391277

/-- Represents a day's work information -/
structure WorkDay where
  hours : ℕ
  hourlyRate : ℚ
  tips : ℚ

/-- Calculates the net earnings for the week -/
def calculateNetEarnings (week : List WorkDay) (taxRate : ℚ) (thursdayDiscountRate : ℚ) : ℚ :=
  sorry

/-- The main theorem stating Hallie's net earnings for the week -/
theorem hallies_net_earnings :
  let week : List WorkDay := [
    ⟨7, 10, 18⟩,  -- Monday
    ⟨5, 12, 12⟩,  -- Tuesday
    ⟨7, 10, 20⟩,  -- Wednesday
    ⟨8, 11, 25⟩,  -- Thursday
    ⟨6, 9, 15⟩    -- Friday
  ]
  let taxRate : ℚ := 5 / 100
  let thursdayDiscountRate : ℚ := 10 / 100
  calculateNetEarnings week taxRate thursdayDiscountRate = 406.1 :=
by sorry

end NUMINAMATH_CALUDE_hallies_net_earnings_l3912_391277


namespace NUMINAMATH_CALUDE_min_value_theorem_l3912_391236

theorem min_value_theorem (x : ℝ) (h : x > 1) : 
  x + 4 / (x - 1) ≥ 5 ∧ ∃ y > 1, y + 4 / (y - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3912_391236


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3912_391267

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 986 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  17 ∣ n ∧
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3912_391267

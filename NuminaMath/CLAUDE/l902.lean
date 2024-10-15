import Mathlib

namespace NUMINAMATH_CALUDE_simplify_square_roots_l902_90278

theorem simplify_square_roots : 
  (Real.sqrt 800 / Real.sqrt 100) - (Real.sqrt 288 / Real.sqrt 72) = 2 * Real.sqrt 2 - 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l902_90278


namespace NUMINAMATH_CALUDE_bag_composition_for_expected_value_l902_90222

/-- Represents the contents of a bag of slips --/
structure BagOfSlips where
  threes : ℕ
  fives : ℕ
  eights : ℕ

/-- Calculates the expected value of a randomly drawn slip --/
def expectedValue (bag : BagOfSlips) : ℚ :=
  (3 * bag.threes + 5 * bag.fives + 8 * bag.eights) / 20

/-- Theorem statement --/
theorem bag_composition_for_expected_value :
  ∃ (bag : BagOfSlips),
    bag.threes + bag.fives + bag.eights = 20 ∧
    expectedValue bag = 57/10 ∧
    bag.threes = 4 ∧
    bag.fives = 10 ∧
    bag.eights = 6 := by
  sorry

end NUMINAMATH_CALUDE_bag_composition_for_expected_value_l902_90222


namespace NUMINAMATH_CALUDE_storks_vs_birds_l902_90243

theorem storks_vs_birds (initial_birds : ℕ) (additional_storks : ℕ) (additional_birds : ℕ) :
  initial_birds = 3 →
  additional_storks = 6 →
  additional_birds = 2 →
  additional_storks - (initial_birds + additional_birds) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_storks_vs_birds_l902_90243


namespace NUMINAMATH_CALUDE_job_completion_time_l902_90273

theorem job_completion_time 
  (T : ℝ) -- Time for P to complete the job alone
  (h1 : T > 0) -- Ensure T is positive
  (h2 : 3 * (1/T + 1/20) + 0.4 * (1/T) = 1) -- Equation from working together and P finishing
  : T = 4 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l902_90273


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l902_90230

theorem least_integer_absolute_value (x : ℤ) :
  (∀ y : ℤ, y < x → ∃ z : ℤ, z ≥ y ∧ z < x ∧ |3 * z^2 - 2 * z + 5| > 29) →
  |3 * x^2 - 2 * x + 5| ≤ 29 →
  x = -2 :=
sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l902_90230


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l902_90254

theorem cara_seating_arrangements (n : ℕ) (h : n = 6) : (n.choose 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l902_90254


namespace NUMINAMATH_CALUDE_solve_for_x_l902_90295

theorem solve_for_x (x y : ℝ) : 3 * x - 4 * y = 6 → x = (6 + 4 * y) / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l902_90295


namespace NUMINAMATH_CALUDE_karen_savings_l902_90280

/-- The sum of a geometric series with initial term 2, common ratio 3, and 7 terms -/
def geometric_sum : ℕ → ℚ
| 0 => 0
| n + 1 => 2 * (3^(n+1) - 1) / (3 - 1)

/-- The theorem stating that the sum of the geometric series after 7 days is 2186 -/
theorem karen_savings : geometric_sum 7 = 2186 := by
  sorry

end NUMINAMATH_CALUDE_karen_savings_l902_90280


namespace NUMINAMATH_CALUDE_two_fifths_of_n_is_80_l902_90244

theorem two_fifths_of_n_is_80 (n : ℚ) (h : n = 5 / 6 * 240) : 2 / 5 * n = 80 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_of_n_is_80_l902_90244


namespace NUMINAMATH_CALUDE_area_scientific_notation_l902_90242

-- Define the area in square kilometers
def area : ℝ := 6.4e6

-- Theorem to prove the scientific notation representation
theorem area_scientific_notation : area = 6.4 * (10 : ℝ)^6 := by
  sorry

end NUMINAMATH_CALUDE_area_scientific_notation_l902_90242


namespace NUMINAMATH_CALUDE_bottle_cap_configurations_l902_90205

theorem bottle_cap_configurations : ∃ (n m : ℕ), n ≠ m ∧ n > 0 ∧ m > 0 ∧ 3 ∣ n ∧ 4 ∣ n ∧ 3 ∣ m ∧ 4 ∣ m :=
by sorry

end NUMINAMATH_CALUDE_bottle_cap_configurations_l902_90205


namespace NUMINAMATH_CALUDE_car_travel_time_ratio_l902_90277

theorem car_travel_time_ratio : 
  let distance : ℝ := 540
  let original_time : ℝ := 8
  let new_speed : ℝ := 45
  let new_time : ℝ := distance / new_speed
  new_time / original_time = 1.5 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_ratio_l902_90277


namespace NUMINAMATH_CALUDE_books_loaned_out_l902_90279

theorem books_loaned_out (initial_books : ℕ) (final_books : ℕ) (return_rate : ℚ) :
  initial_books = 150 →
  final_books = 100 →
  return_rate = 3/5 →
  (initial_books - final_books : ℚ) / (1 - return_rate) = 125 :=
by sorry

end NUMINAMATH_CALUDE_books_loaned_out_l902_90279


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l902_90272

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = -3 ∧ x₂ = 4) ∧ 
  (x₁^2 - x₁ - 12 = 0) ∧ 
  (x₂^2 - x₂ - 12 = 0) ∧
  (∀ x : ℝ, x^2 - x - 12 = 0 → x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l902_90272


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l902_90239

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * |x + 2| - |a * x|

-- Part 1
theorem solution_set_part1 : 
  {x : ℝ | f 2 x > 2} = {x : ℝ | x > -1/2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  (∀ x ∈ Set.Ioo (-1) 1, f a x > x + 1) ↔ a ∈ Set.Ioo (-2) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l902_90239


namespace NUMINAMATH_CALUDE_mark_spent_40_dollars_l902_90290

/-- The total amount Mark spent on tomatoes and apples -/
def total_spent (tomato_price : ℝ) (tomato_weight : ℝ) (apple_price : ℝ) (apple_weight : ℝ) : ℝ :=
  tomato_price * tomato_weight + apple_price * apple_weight

/-- Proof that Mark spent $40 in total -/
theorem mark_spent_40_dollars : total_spent 5 2 6 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_mark_spent_40_dollars_l902_90290


namespace NUMINAMATH_CALUDE_recurrence_sequence_a1_l902_90232

/-- A sequence of positive real numbers satisfying the given recurrence relation. -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  a 0 = 1 ∧ (∀ n : ℕ, 0 < a n) ∧ (∀ n : ℕ, a n = a (n + 1) + a (n + 2))

/-- The theorem stating that a₁ equals (√5 - 1) / 2 for the given recurrence sequence. -/
theorem recurrence_sequence_a1 (a : ℕ → ℝ) (h : RecurrenceSequence a) :
    a 1 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_a1_l902_90232


namespace NUMINAMATH_CALUDE_point_line_distance_l902_90265

/-- A type representing points on a line -/
structure Point where
  x : ℝ

/-- Distance between two points -/
def dist (p q : Point) : ℝ := |p.x - q.x|

theorem point_line_distance (A : Fin 11 → Point) :
  (dist (A 0) (A 10) = 56) →
  (∀ i, i < 9 → dist (A i) (A (i + 2)) ≤ 12) →
  (∀ j, j < 8 → dist (A j) (A (j + 3)) ≥ 17) →
  dist (A 1) (A 6) = 29 := by
sorry

end NUMINAMATH_CALUDE_point_line_distance_l902_90265


namespace NUMINAMATH_CALUDE_alex_age_theorem_l902_90203

theorem alex_age_theorem :
  ∃! x : ℕ, x > 0 ∧ x ≤ 100 ∧ 
  ∃ y : ℕ, x - 2 = y^2 ∧
  ∃ z : ℕ, x + 2 = z^3 :=
by
  sorry

end NUMINAMATH_CALUDE_alex_age_theorem_l902_90203


namespace NUMINAMATH_CALUDE_binary_to_decimal_111_l902_90270

theorem binary_to_decimal_111 : 
  (1 : ℕ) * 2^0 + (1 : ℕ) * 2^1 + (1 : ℕ) * 2^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_111_l902_90270


namespace NUMINAMATH_CALUDE_average_increase_is_four_l902_90286

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : ℕ
  totalRuns : ℕ
  nextInningsRuns : ℕ

/-- Calculates the increase in average runs after the next innings -/
def averageIncrease (player : CricketPlayer) : ℚ :=
  let currentAverage : ℚ := player.totalRuns / player.innings
  let newTotalRuns : ℕ := player.totalRuns + player.nextInningsRuns
  let newAverage : ℚ := newTotalRuns / (player.innings + 1)
  newAverage - currentAverage

/-- Theorem: The increase in average runs is 4 for the given conditions -/
theorem average_increase_is_four :
  ∀ (player : CricketPlayer),
    player.innings = 10 →
    player.totalRuns = 400 →
    player.nextInningsRuns = 84 →
    averageIncrease player = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_increase_is_four_l902_90286


namespace NUMINAMATH_CALUDE_cauliflower_sales_value_l902_90226

def farmers_market_sales (total_earnings broccoli_sales carrot_sales spinach_sales tomato_sales cauliflower_sales : ℝ) : Prop :=
  total_earnings = 500 ∧
  broccoli_sales = 57 ∧
  carrot_sales = 2 * broccoli_sales ∧
  spinach_sales = (carrot_sales / 2) + 16 ∧
  tomato_sales = broccoli_sales + spinach_sales ∧
  total_earnings = broccoli_sales + carrot_sales + spinach_sales + tomato_sales + cauliflower_sales

theorem cauliflower_sales_value :
  ∀ total_earnings broccoli_sales carrot_sales spinach_sales tomato_sales cauliflower_sales : ℝ,
  farmers_market_sales total_earnings broccoli_sales carrot_sales spinach_sales tomato_sales cauliflower_sales →
  cauliflower_sales = 126 := by
sorry

end NUMINAMATH_CALUDE_cauliflower_sales_value_l902_90226


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l902_90200

theorem inequality_system_solution_range (a : ℝ) : 
  (∀ x : ℝ, (2 * x > 4 ∧ 3 * x + a > 0) ↔ x > 2) → 
  a ≥ -6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l902_90200


namespace NUMINAMATH_CALUDE_base3_102012_equals_302_l902_90247

def base3_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base3_102012_equals_302 :
  base3_to_base10 [1, 0, 2, 0, 1, 2] = 302 := by
  sorry

end NUMINAMATH_CALUDE_base3_102012_equals_302_l902_90247


namespace NUMINAMATH_CALUDE_sqrt_combinable_with_sqrt_two_l902_90297

theorem sqrt_combinable_with_sqrt_two : ∃! x : ℝ, 
  (x = Real.sqrt 10 ∨ x = Real.sqrt 12 ∨ x = Real.sqrt (1/2) ∨ x = 1 / Real.sqrt 6) ∧
  ∃ (a : ℝ), x = a * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_combinable_with_sqrt_two_l902_90297


namespace NUMINAMATH_CALUDE_speed_in_still_water_l902_90224

/-- 
Given a man's upstream and downstream speeds, calculate his speed in still water.
-/
theorem speed_in_still_water 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : upstream_speed = 25) 
  (h2 : downstream_speed = 55) : 
  (upstream_speed + downstream_speed) / 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l902_90224


namespace NUMINAMATH_CALUDE_total_sheets_prepared_l902_90210

/-- Given the number of sheets used for a crane and the number of sheets left,
    prove that the total number of sheets prepared at the beginning
    is equal to the sum of sheets used and sheets left. -/
theorem total_sheets_prepared
  (sheets_used : ℕ) (sheets_left : ℕ)
  (h1 : sheets_used = 12)
  (h2 : sheets_left = 9) :
  sheets_used + sheets_left = 21 := by
sorry

end NUMINAMATH_CALUDE_total_sheets_prepared_l902_90210


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l902_90225

/-- The operation ⊗ as defined in the problem -/
noncomputable def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

/-- Theorem stating that if 5 ⊗ g = 11, then g = 30 -/
theorem bowtie_equation_solution :
  ∃ g : ℝ, bowtie 5 g = 11 ∧ g = 30 := by sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l902_90225


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_l902_90207

theorem quadratic_always_nonnegative (m : ℝ) : 
  (∀ x : ℝ, x^2 - (m - 1) * x + 1 ≥ 0) ↔ m ∈ Set.Icc (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_l902_90207


namespace NUMINAMATH_CALUDE_opposite_sqrt5_minus_2_l902_90233

theorem opposite_sqrt5_minus_2 :
  -(Real.sqrt 5 - 2) = 2 - Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_opposite_sqrt5_minus_2_l902_90233


namespace NUMINAMATH_CALUDE_total_digits_memorized_l902_90248

/-- The number of digits of pi memorized by each person --/
structure PiDigits where
  carlos : ℕ
  sam : ℕ
  mina : ℕ
  nina : ℕ

/-- The conditions given in the problem --/
def satisfies_conditions (p : PiDigits) : Prop :=
  p.sam = p.carlos + 6 ∧
  p.mina = 6 * p.carlos ∧
  p.nina = 4 * p.carlos ∧
  p.mina = 24

/-- The theorem to be proved --/
theorem total_digits_memorized (p : PiDigits) 
  (h : satisfies_conditions p) : 
  p.sam + p.carlos + p.mina + p.nina = 54 := by
  sorry


end NUMINAMATH_CALUDE_total_digits_memorized_l902_90248


namespace NUMINAMATH_CALUDE_exponent_division_l902_90245

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^4 / a^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l902_90245


namespace NUMINAMATH_CALUDE_inequality_empty_solution_set_l902_90211

theorem inequality_empty_solution_set (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 2 * m * x + 1 ≥ 0) → 0 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_empty_solution_set_l902_90211


namespace NUMINAMATH_CALUDE_sum_of_valid_m_l902_90209

def inequality_system (x m : ℤ) : Prop :=
  (x - 2) / 4 < (x - 1) / 3 ∧ 3 * x - m ≤ 3 - x

def equation_system (x y m : ℤ) : Prop :=
  m * x + y = 4 ∧ 3 * x - y = 0

theorem sum_of_valid_m :
  (∃ (s : Finset ℤ), 
    (∀ m ∈ s, 
      (∃! (a b : ℤ), inequality_system a m) ∧
      (∃ (x y : ℤ), equation_system x y m)) ∧
    (s.sum id = -3)) :=
sorry

end NUMINAMATH_CALUDE_sum_of_valid_m_l902_90209


namespace NUMINAMATH_CALUDE_water_added_to_tank_l902_90256

theorem water_added_to_tank (tank_capacity : ℚ) (initial_fraction : ℚ) (final_fraction : ℚ) : 
  tank_capacity = 32 →
  initial_fraction = 3/4 →
  final_fraction = 7/8 →
  final_fraction * tank_capacity - initial_fraction * tank_capacity = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_tank_l902_90256


namespace NUMINAMATH_CALUDE_lindas_mean_score_l902_90241

def scores : List ℕ := [80, 86, 90, 92, 95, 97]

def jakes_mean : ℕ := 89

theorem lindas_mean_score (h1 : scores.length = 6)
  (h2 : ∃ (jake_scores linda_scores : List ℕ),
    jake_scores.length = 3 ∧
    linda_scores.length = 3 ∧
    jake_scores ++ linda_scores = scores)
  (h3 : ∃ (jake_scores : List ℕ),
    jake_scores.length = 3 ∧
    jake_scores.sum / jake_scores.length = jakes_mean) :
  ∃ (linda_scores : List ℕ),
    linda_scores.length = 3 ∧
    linda_scores.sum / linda_scores.length = 91 :=
by sorry

end NUMINAMATH_CALUDE_lindas_mean_score_l902_90241


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l902_90219

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l902_90219


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_16_is_plus_minus_2_l902_90229

theorem sqrt_of_sqrt_16_is_plus_minus_2 : 
  {x : ℝ | x^2 = Real.sqrt 16} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_16_is_plus_minus_2_l902_90229


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l902_90216

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = Real.sqrt 3) :
  Complex.abs ((z - 1) * (z + 1)^2) ≤ 3 * Real.sqrt 3 ∧
  ∃ w : ℂ, Complex.abs w = Real.sqrt 3 ∧ Complex.abs ((w - 1) * (w + 1)^2) = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l902_90216


namespace NUMINAMATH_CALUDE_equal_segments_imply_equal_x_y_l902_90258

/-- Given two pairs of equal lengths (a₁, a₂) and (b₁, b₂), prove that x = y. -/
theorem equal_segments_imply_equal_x_y (a₁ a₂ b₁ b₂ x y : ℝ) 
  (h1 : a₁ = a₂) (h2 : b₁ = b₂) : x = y := by
  sorry

end NUMINAMATH_CALUDE_equal_segments_imply_equal_x_y_l902_90258


namespace NUMINAMATH_CALUDE_roots_sum_squares_l902_90282

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 3*x - 2

-- Define the theorem
theorem roots_sum_squares (p q r : ℝ) : 
  f p = 0 → f q = 0 → f r = 0 → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_squares_l902_90282


namespace NUMINAMATH_CALUDE_gcd_180_270_l902_90261

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_270_l902_90261


namespace NUMINAMATH_CALUDE_function_inequality_and_minimum_value_l902_90215

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 2|

-- Define the solution set M
def M : Set ℝ := {x | 2/3 ≤ x ∧ x ≤ 6}

-- Define the theorem
theorem function_inequality_and_minimum_value :
  (∀ x ∈ M, f x ≥ -1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 4*a + b + c = 6 →
    1/(2*a + b) + 1/(2*a + c) ≥ 2/3) ∧
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 4*a + b + c = 6 ∧
    1/(2*a + b) + 1/(2*a + c) = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_and_minimum_value_l902_90215


namespace NUMINAMATH_CALUDE_jamie_alex_payment_difference_l902_90257

-- Define the problem parameters
def total_slices : ℕ := 10
def plain_pizza_cost : ℚ := 10
def spicy_topping_cost : ℚ := 3
def spicy_fraction : ℚ := 1/3

-- Define the number of slices each person ate
def jamie_spicy_slices : ℕ := (spicy_fraction * total_slices).num.toNat
def jamie_plain_slices : ℕ := 2
def alex_plain_slices : ℕ := total_slices - jamie_spicy_slices - jamie_plain_slices

-- Define the theorem
theorem jamie_alex_payment_difference :
  let total_cost : ℚ := plain_pizza_cost + spicy_topping_cost
  let cost_per_slice : ℚ := total_cost / total_slices
  let jamie_payment : ℚ := cost_per_slice * (jamie_spicy_slices + jamie_plain_slices)
  let alex_payment : ℚ := cost_per_slice * alex_plain_slices
  jamie_payment - alex_payment = 0 :=
sorry

end NUMINAMATH_CALUDE_jamie_alex_payment_difference_l902_90257


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l902_90208

/-- Proves that mixing 250 mL of 10% alcohol solution with 750 mL of 30% alcohol solution results in a 25% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 250
  let y_volume : ℝ := 750
  let x_concentration : ℝ := 0.10
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.25
  let total_volume : ℝ := x_volume + y_volume
  let total_alcohol : ℝ := x_volume * x_concentration + y_volume * y_concentration
  total_alcohol / total_volume = target_concentration := by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l902_90208


namespace NUMINAMATH_CALUDE_blue_face_probability_l902_90285

theorem blue_face_probability (total_faces : ℕ) (blue_faces : ℕ)
  (h1 : total_faces = 12)
  (h2 : blue_faces = 4) :
  (blue_faces : ℚ) / total_faces = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_blue_face_probability_l902_90285


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l902_90293

theorem inequality_system_solutions : 
  {x : ℤ | x ≥ 0 ∧ 4 * (x + 1) ≤ 7 * x + 10 ∧ x - 5 < (x - 8) / 3} = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l902_90293


namespace NUMINAMATH_CALUDE_unknown_number_value_l902_90263

theorem unknown_number_value : 
  ∃ (unknown_number : ℝ), 
    (∀ x : ℝ, (3 + 2 * x)^5 = (unknown_number + 3 * x)^4) ∧
    ((3 + 2 * 1.5)^5 = (unknown_number + 3 * 1.5)^4) →
    unknown_number = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_unknown_number_value_l902_90263


namespace NUMINAMATH_CALUDE_m_less_than_two_necessary_not_sufficient_l902_90275

/-- The condition for the quadratic inequality x^2 + mx + 1 > 0 to have ℝ as its solution set -/
def has_real_solution_set (m : ℝ) : Prop :=
  ∀ x, x^2 + m*x + 1 > 0

/-- The statement that m < 2 is a necessary but not sufficient condition -/
theorem m_less_than_two_necessary_not_sufficient :
  (∀ m, has_real_solution_set m → m < 2) ∧
  ¬(∀ m, m < 2 → has_real_solution_set m) := by sorry

end NUMINAMATH_CALUDE_m_less_than_two_necessary_not_sufficient_l902_90275


namespace NUMINAMATH_CALUDE_lecture_distribution_l902_90240

def total_lecture_time : ℕ := 480
def max_disc_capacity : ℕ := 70

theorem lecture_distribution :
  ∃ (num_discs : ℕ) (minutes_per_disc : ℕ),
    num_discs > 0 ∧
    minutes_per_disc > 0 ∧
    minutes_per_disc ≤ max_disc_capacity ∧
    num_discs * minutes_per_disc = total_lecture_time ∧
    (∀ n : ℕ, n > 0 → n * max_disc_capacity < total_lecture_time → n < num_discs) ∧
    minutes_per_disc = 68 := by
  sorry

end NUMINAMATH_CALUDE_lecture_distribution_l902_90240


namespace NUMINAMATH_CALUDE_triangle_area_with_given_base_and_height_l902_90227

theorem triangle_area_with_given_base_and_height :
  ∀ (base height : ℝ), 
    base = 12 →
    height = 15 →
    (1 / 2 : ℝ) * base * height = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_given_base_and_height_l902_90227


namespace NUMINAMATH_CALUDE_factorial_ratio_l902_90266

theorem factorial_ratio : Nat.factorial 10 / (Nat.factorial 7 * Nat.factorial 2 * Nat.factorial 1) = 360 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l902_90266


namespace NUMINAMATH_CALUDE_power_multiplication_zero_power_distribute_and_simplify_negative_power_and_division_l902_90283

-- Define variables
variable (a m : ℝ)
variable (π : ℝ)

-- Theorem statements
theorem power_multiplication : a^2 * a^3 = a^5 := by sorry

theorem zero_power : (3.142 - π)^0 = 1 := by sorry

theorem distribute_and_simplify : 2*a*(a^2 - 1) = 2*a^3 - 2*a := by sorry

theorem negative_power_and_division : (-m^3)^2 / m^4 = m^2 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_zero_power_distribute_and_simplify_negative_power_and_division_l902_90283


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l902_90268

/-- Proves that the cost of the first candy is $8.00 per pound given the conditions of the candy mixture problem. -/
theorem candy_mixture_cost (first_candy_weight : ℝ) (second_candy_weight : ℝ) (second_candy_cost : ℝ) (mixture_cost : ℝ) 
  (h1 : first_candy_weight = 25)
  (h2 : second_candy_weight = 50)
  (h3 : second_candy_cost = 5)
  (h4 : mixture_cost = 6)
  (h5 : first_candy_weight + second_candy_weight = 75) :
  ∃ (C : ℝ), C = 8 ∧ 
  C * first_candy_weight + second_candy_cost * second_candy_weight = 
  mixture_cost * (first_candy_weight + second_candy_weight) :=
by sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l902_90268


namespace NUMINAMATH_CALUDE_sqrt_13_plus_1_parts_l902_90287

theorem sqrt_13_plus_1_parts : ∃ (a : ℤ) (b : ℝ),
  (a : ℝ) + b = Real.sqrt 13 + 1 ∧ 
  a = 4 ∧
  b = Real.sqrt 13 - 3 ∧
  0 ≤ b ∧ 
  b < 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_13_plus_1_parts_l902_90287


namespace NUMINAMATH_CALUDE_eulers_formula_simply_connected_l902_90234

/-- A simply connected polyhedron -/
structure SimplyConnectedPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ
  is_simply_connected : Bool

/-- Euler's formula for simply connected polyhedra -/
theorem eulers_formula_simply_connected (p : SimplyConnectedPolyhedron) 
  (h : p.is_simply_connected = true) : 
  p.faces - p.edges + p.vertices = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_simply_connected_l902_90234


namespace NUMINAMATH_CALUDE_problem_statement_l902_90269

theorem problem_statement (x y : ℝ) :
  |x + y - 6| + (x - y + 3)^2 = 0 → 3*x - y = 0 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l902_90269


namespace NUMINAMATH_CALUDE_room_width_calculation_l902_90267

/-- Proves that given a rectangular room with length 5.5 m and a total paving cost of $16,500 at $800 per square meter, the width of the room is 3.75 m. -/
theorem room_width_calculation (length : Real) (cost_per_sqm : Real) (total_cost : Real) :
  length = 5.5 →
  cost_per_sqm = 800 →
  total_cost = 16500 →
  (total_cost / cost_per_sqm) / length = 3.75 := by
  sorry

#check room_width_calculation

end NUMINAMATH_CALUDE_room_width_calculation_l902_90267


namespace NUMINAMATH_CALUDE_soda_difference_is_21_l902_90250

/-- The number of regular soda bottles -/
def regular_soda : ℕ := 81

/-- The number of diet soda bottles -/
def diet_soda : ℕ := 60

/-- The difference between regular and diet soda bottles -/
def soda_difference : ℕ := regular_soda - diet_soda

theorem soda_difference_is_21 : soda_difference = 21 := by
  sorry

end NUMINAMATH_CALUDE_soda_difference_is_21_l902_90250


namespace NUMINAMATH_CALUDE_desktop_revenue_is_12000_l902_90262

/-- The revenue generated from the sale of desktop computers in Mr. Lu's store --/
def desktop_revenue (total_computers : ℕ) (laptop_price netbook_price desktop_price : ℕ) : ℕ :=
  let laptop_count := total_computers / 2
  let netbook_count := total_computers / 3
  let desktop_count := total_computers - laptop_count - netbook_count
  desktop_count * desktop_price

/-- Theorem stating the revenue from desktop computers --/
theorem desktop_revenue_is_12000 :
  desktop_revenue 72 750 500 1000 = 12000 := by
  sorry

end NUMINAMATH_CALUDE_desktop_revenue_is_12000_l902_90262


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l902_90237

theorem quadratic_factorization_sum (d e f : ℝ) : 
  (∀ x, (x + d) * (x + e) = x^2 + 11*x + 24) →
  (∀ x, (x + e) * (x - f) = x^2 + 9*x - 36) →
  d + e + f = 14 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l902_90237


namespace NUMINAMATH_CALUDE_no_rational_solution_and_unique_perfect_square_l902_90284

theorem no_rational_solution_and_unique_perfect_square :
  (∀ a : ℕ, ¬∃ x y z : ℚ, x^2 + y^2 + z^2 = 8 * a + 7) ∧
  (∀ n : ℕ, (∃ k : ℤ, 7^n + 8 = k^2) ↔ n = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_rational_solution_and_unique_perfect_square_l902_90284


namespace NUMINAMATH_CALUDE_bridge_length_l902_90231

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 130 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 245 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l902_90231


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l902_90253

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^10 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 20 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l902_90253


namespace NUMINAMATH_CALUDE_computer_literate_female_employees_l902_90201

theorem computer_literate_female_employees 
  (total_employees : ℕ)
  (female_percentage : ℚ)
  (male_literate_percentage : ℚ)
  (total_literate_percentage : ℚ)
  (h_total : total_employees = 1300)
  (h_female : female_percentage = 60 / 100)
  (h_male_literate : male_literate_percentage = 50 / 100)
  (h_total_literate : total_literate_percentage = 62 / 100) :
  ↑(total_employees * female_percentage * total_literate_percentage - 
    total_employees * (1 - female_percentage) * male_literate_percentage : ℚ).num = 546 := by
  sorry

end NUMINAMATH_CALUDE_computer_literate_female_employees_l902_90201


namespace NUMINAMATH_CALUDE_problem_solving_probability_l902_90252

theorem problem_solving_probability (p_A p_B : ℝ) (h_A : p_A = 1/5) (h_B : p_B = 1/3) :
  1 - (1 - p_A) * (1 - p_B) = 7/15 :=
by sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l902_90252


namespace NUMINAMATH_CALUDE_angle_sum_in_polygon_l902_90291

theorem angle_sum_in_polygon (D E F p q : ℝ) : 
  D = 38 → E = 58 → F = 36 → 
  D + E + (360 - p) + 90 + (126 - q) = 540 → 
  p + q = 132 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_polygon_l902_90291


namespace NUMINAMATH_CALUDE_x_equals_two_l902_90255

theorem x_equals_two (some_number : ℝ) (h : x + some_number = 3) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_two_l902_90255


namespace NUMINAMATH_CALUDE_abs_lt_one_sufficient_not_necessary_for_cube_lt_one_l902_90299

theorem abs_lt_one_sufficient_not_necessary_for_cube_lt_one :
  (∃ x : ℝ, (|x| < 1 → x^3 < 1) ∧ ¬(x^3 < 1 → |x| < 1)) := by
  sorry

end NUMINAMATH_CALUDE_abs_lt_one_sufficient_not_necessary_for_cube_lt_one_l902_90299


namespace NUMINAMATH_CALUDE_power_sum_difference_l902_90292

theorem power_sum_difference : 2^(0+1+2) - (2^0 + 2^1 + 2^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l902_90292


namespace NUMINAMATH_CALUDE_strawberries_picked_l902_90221

/-- Given that Paul started with 28 strawberries and ended up with 63 strawberries,
    prove that he picked 35 strawberries. -/
theorem strawberries_picked (initial : ℕ) (final : ℕ) (h1 : initial = 28) (h2 : final = 63) :
  final - initial = 35 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_picked_l902_90221


namespace NUMINAMATH_CALUDE_children_count_l902_90249

/-- The number of children required to assemble one small robot -/
def small_robot_children : ℕ := 2

/-- The number of children required to assemble one large robot -/
def large_robot_children : ℕ := 3

/-- The number of small robots assembled -/
def small_robots : ℕ := 18

/-- The number of large robots assembled -/
def large_robots : ℕ := 12

/-- The total number of children -/
def total_children : ℕ := small_robot_children * small_robots + large_robot_children * large_robots

theorem children_count : total_children = 72 := by sorry

end NUMINAMATH_CALUDE_children_count_l902_90249


namespace NUMINAMATH_CALUDE_leading_coefficient_is_negative_eleven_l902_90264

def polynomial (x : ℝ) : ℝ := 2 * (x^5 - 3*x^4 + 2*x^2) + 5 * (x^5 + x^4) - 6 * (3*x^5 + x^3 - x + 1)

theorem leading_coefficient_is_negative_eleven :
  ∃ (f : ℝ → ℝ), (∀ x, polynomial x = f x) ∧ 
  (∃ (a : ℝ) (g : ℝ → ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^5 + g x) ∧ a = -11) :=
by sorry

end NUMINAMATH_CALUDE_leading_coefficient_is_negative_eleven_l902_90264


namespace NUMINAMATH_CALUDE_evaluate_expression_l902_90202

theorem evaluate_expression : (7 - 3)^2 + (7^2 - 3^2) = 56 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l902_90202


namespace NUMINAMATH_CALUDE_eight_book_distribution_l902_90260

/-- The number of ways to distribute n identical books between two locations,
    with at least one book in each location. -/
def distribution_ways (n : ℕ) : ℕ :=
  if n ≥ 2 then n - 1 else 0

/-- Theorem stating that there are 7 ways to distribute 8 identical books
    between storage and students, with at least one book in each location. -/
theorem eight_book_distribution :
  distribution_ways 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_eight_book_distribution_l902_90260


namespace NUMINAMATH_CALUDE_circle_op_twelve_seven_l902_90271

def circle_op (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem circle_op_twelve_seven :
  circle_op 12 7 = 95 := by sorry

end NUMINAMATH_CALUDE_circle_op_twelve_seven_l902_90271


namespace NUMINAMATH_CALUDE_angle_with_complement_half_supplement_l902_90217

theorem angle_with_complement_half_supplement (x : ℝ) :
  (90 - x) = (1/2) * (180 - x) → x = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_complement_half_supplement_l902_90217


namespace NUMINAMATH_CALUDE_complex_number_line_l902_90298

theorem complex_number_line (z : ℂ) (h : z * (1 + Complex.I)^2 = 1 - Complex.I) :
  z.im = -1/2 := by sorry

end NUMINAMATH_CALUDE_complex_number_line_l902_90298


namespace NUMINAMATH_CALUDE_impossible_to_turn_all_lamps_off_l902_90236

/-- Represents the state of a lamp (on or off) -/
inductive LampState
| On
| Off

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the chessboard state -/
def ChessboardState := Position → LampState

/-- Represents the allowed operations on the chessboard -/
inductive Operation
| InvertRow (row : Fin 8)
| InvertColumn (col : Fin 8)
| InvertDiagonal (d : ℤ) -- d represents the diagonal offset

/-- The initial state of the chessboard -/
def initialState : ChessboardState :=
  fun pos => if pos.row = 0 && pos.col = 3 then LampState.Off else LampState.On

/-- Apply an operation to the chessboard state -/
def applyOperation (state : ChessboardState) (op : Operation) : ChessboardState :=
  sorry

/-- Check if all lamps are off -/
def allLampsOff (state : ChessboardState) : Prop :=
  ∀ pos, state pos = LampState.Off

/-- The main theorem to be proved -/
theorem impossible_to_turn_all_lamps_off :
  ¬∃ (ops : List Operation), allLampsOff (ops.foldl applyOperation initialState) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_turn_all_lamps_off_l902_90236


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_five_l902_90246

theorem reciprocal_of_negative_five :
  ∃ x : ℚ, x * (-5) = 1 ∧ x = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_five_l902_90246


namespace NUMINAMATH_CALUDE_d_equals_four_l902_90274

/-- A nine-digit number with specific properties -/
structure NineDigitNumber where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ
  F : ℕ
  G : ℕ
  first_three_sum : 6 + A + B = 13
  second_three_sum : A + B + C = 13
  third_three_sum : B + C + D = 13
  fourth_three_sum : C + D + E = 13
  fifth_three_sum : D + E + F = 13
  sixth_three_sum : E + F + G = 13
  last_three_sum : F + G + 3 = 13

/-- The digit D in the number must be 4 -/
theorem d_equals_four (n : NineDigitNumber) : n.D = 4 := by
  sorry

end NUMINAMATH_CALUDE_d_equals_four_l902_90274


namespace NUMINAMATH_CALUDE_hcd_7560_180_minus_12_l902_90296

theorem hcd_7560_180_minus_12 : Nat.gcd 7560 180 - 12 = 168 := by sorry

end NUMINAMATH_CALUDE_hcd_7560_180_minus_12_l902_90296


namespace NUMINAMATH_CALUDE_avocado_count_is_two_l902_90259

/-- Represents the contents and cost of a fruit basket -/
structure FruitBasket where
  banana_count : ℕ
  apple_count : ℕ
  strawberry_count : ℕ
  avocado_count : ℕ
  grapes_count : ℕ
  banana_price : ℚ
  apple_price : ℚ
  strawberry_price : ℚ
  avocado_price : ℚ
  grapes_price : ℚ
  total_cost : ℚ

/-- The fruit basket problem -/
def fruit_basket_problem : FruitBasket :=
  { banana_count := 4
  , apple_count := 3
  , strawberry_count := 24
  , avocado_count := 0  -- This is what we need to prove
  , grapes_count := 1
  , banana_price := 1
  , apple_price := 2
  , strawberry_price := 1/3  -- $4 for 12 strawberries
  , avocado_price := 3
  , grapes_price := 4  -- $2 for half a bunch, so $4 for a full bunch
  , total_cost := 28 }

/-- Theorem stating that the number of avocados in the fruit basket is 2 -/
theorem avocado_count_is_two (fb : FruitBasket) 
  (h1 : fb = fruit_basket_problem) :
  fb.avocado_count = 2 := by
  sorry


end NUMINAMATH_CALUDE_avocado_count_is_two_l902_90259


namespace NUMINAMATH_CALUDE_abs_greater_than_y_if_x_greater_than_y_l902_90214

theorem abs_greater_than_y_if_x_greater_than_y (x y : ℝ) (h : x > y) : |x| > y := by
  sorry

end NUMINAMATH_CALUDE_abs_greater_than_y_if_x_greater_than_y_l902_90214


namespace NUMINAMATH_CALUDE_order_of_magnitudes_l902_90294

-- Define the function f(x) = ln(x) - x
noncomputable def f (x : ℝ) : ℝ := Real.log x - x

-- Define a, b, and c
noncomputable def a : ℝ := f (3/2)
noncomputable def b : ℝ := f Real.pi
noncomputable def c : ℝ := f 3

-- State the theorem
theorem order_of_magnitudes (h1 : 3/2 < 3) (h2 : 3 < Real.pi) : a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_order_of_magnitudes_l902_90294


namespace NUMINAMATH_CALUDE_unique_triple_sum_l902_90213

theorem unique_triple_sum (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y : ℚ) / z + (y * z : ℚ) / x + (z * x : ℚ) / y = 3 → x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_sum_l902_90213


namespace NUMINAMATH_CALUDE_circle_area_ratio_l902_90218

theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0)
  (h_arc : C * (60 / 360) = D * (40 / 360)) :
  (C^2 / D^2 : ℝ) = 4/9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l902_90218


namespace NUMINAMATH_CALUDE_f_properties_l902_90276

def f (x : ℝ) : ℝ := x^3 - x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, x < -Real.sqrt 3 / 3 → (deriv f) x > 0) ∧
  (∀ x, x > Real.sqrt 3 / 3 → (deriv f) x > 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l902_90276


namespace NUMINAMATH_CALUDE_function_property_l902_90204

/-- Given a function f(x) = ax^5 + bx^3 + cx + 1, where a, b, and c are non-zero real numbers,
    if f(3) = 11, then f(-3) = -9. -/
theorem function_property (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + c * x + 1
  f 3 = 11 → f (-3) = -9 := by
sorry

end NUMINAMATH_CALUDE_function_property_l902_90204


namespace NUMINAMATH_CALUDE_exists_partition_without_infinite_progression_l902_90228

/-- A partition of natural numbers. -/
def Partition := ℕ → Bool

/-- Checks if a set contains an infinite arithmetic progression. -/
def HasInfiniteArithmeticProgression (p : Partition) : Prop :=
  ∃ a d : ℕ, d > 0 ∧ ∀ k : ℕ, p (a + k * d) = p a

/-- There exists a partition of natural numbers into two sets
    such that neither set contains an infinite arithmetic progression. -/
theorem exists_partition_without_infinite_progression :
  ∃ p : Partition, ¬HasInfiniteArithmeticProgression p ∧
                   ¬HasInfiniteArithmeticProgression (fun n => ¬(p n)) := by
  sorry

end NUMINAMATH_CALUDE_exists_partition_without_infinite_progression_l902_90228


namespace NUMINAMATH_CALUDE_percentage_difference_l902_90289

theorem percentage_difference : 
  (45 / 100 * 60) - (35 / 100 * 40) = 13 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l902_90289


namespace NUMINAMATH_CALUDE_problem_solution_l902_90212

theorem problem_solution (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0)
    (h4 : 3 * a + 2 * b + c = 5) (h5 : 2 * a + b - 3 * c = 1) :
    (3 / 7 ≤ c ∧ c ≤ 7 / 11) ∧
    (∀ x, 3 * a + b - 7 * c ≤ x → x ≤ -1 / 11) ∧
    (∀ y, -5 / 7 ≤ y → y ≤ 3 * a + b - 7 * c) :=
  sorry

end NUMINAMATH_CALUDE_problem_solution_l902_90212


namespace NUMINAMATH_CALUDE_prob_red_card_standard_deck_l902_90288

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (red_suits : ℕ)
  (cards_per_suit : ℕ)

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    ranks := 13,
    suits := 4,
    red_suits := 2,
    cards_per_suit := 13 }

/-- The probability of drawing a red suit card from the top of a randomly shuffled deck -/
def prob_red_card (d : Deck) : ℚ :=
  (d.red_suits * d.cards_per_suit) / d.total_cards

/-- Theorem stating that the probability of drawing a red suit card from a standard deck is 1/2 -/
theorem prob_red_card_standard_deck :
  prob_red_card standard_deck = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_card_standard_deck_l902_90288


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l902_90220

/-- Given a triangle DEF with side lengths DE = 8, DF = 5, and EF = 9,
    the radius of its inscribed circle is 6√11/11. -/
theorem inscribed_circle_radius (DE DF EF : ℝ) (hDE : DE = 8) (hDF : DF = 5) (hEF : EF = 9) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  area / s = 6 * Real.sqrt 11 / 11 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l902_90220


namespace NUMINAMATH_CALUDE_region_area_l902_90281

/-- The area of a region bounded by three circular arcs -/
theorem region_area (r : ℝ) (θ : ℝ) : 
  r > 0 → 
  θ = π / 4 → 
  let sector_area := θ / (2 * π) * π * r^2
  let triangle_area := 1 / 2 * r^2 * Real.sin θ
  3 * (sector_area - triangle_area) = 24 * π - 48 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_region_area_l902_90281


namespace NUMINAMATH_CALUDE_platform_length_l902_90206

/-- The length of a platform given train speed and passing times -/
theorem platform_length (train_speed : ℝ) (platform_time : ℝ) (man_time : ℝ) : 
  train_speed = 54 → platform_time = 16 → man_time = 10 → 
  (train_speed * 5 / 18) * (platform_time - man_time) = 90 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l902_90206


namespace NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l902_90238

/-- The maximum number of parts that three planes can divide space into -/
def max_parts_three_planes : ℕ := 8

/-- Theorem: The maximum number of parts that three planes can divide space into is 8 -/
theorem max_parts_three_planes_is_eight :
  max_parts_three_planes = 8 := by sorry

end NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l902_90238


namespace NUMINAMATH_CALUDE_cakes_served_yesterday_l902_90235

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := 5

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := 6

/-- The total number of cakes served over two days -/
def total_cakes : ℕ := 14

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := total_cakes - (lunch_cakes + dinner_cakes)

theorem cakes_served_yesterday : yesterday_cakes = 3 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_yesterday_l902_90235


namespace NUMINAMATH_CALUDE_total_money_l902_90223

def money_problem (john peter quincy andrew : ℝ) : Prop :=
  peter = 2 * john ∧
  quincy = peter + 20 ∧
  andrew = 1.15 * quincy ∧
  john + peter + quincy + andrew = 1211

theorem total_money :
  ∃ john peter quincy andrew : ℝ,
    money_problem john peter quincy andrew ∧
    john + peter + quincy + andrew = 1072.01 := by sorry

end NUMINAMATH_CALUDE_total_money_l902_90223


namespace NUMINAMATH_CALUDE_rainfall_2011_l902_90251

/-- The total rainfall in Rainville for 2011, given the average monthly rainfall in 2010 and the increase in 2011. -/
def total_rainfall_2011 (avg_2010 : ℝ) (increase : ℝ) : ℝ :=
  (avg_2010 + increase) * 12

/-- Theorem stating that the total rainfall in Rainville for 2011 was 483.6 mm. -/
theorem rainfall_2011 : total_rainfall_2011 36.8 3.5 = 483.6 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_2011_l902_90251

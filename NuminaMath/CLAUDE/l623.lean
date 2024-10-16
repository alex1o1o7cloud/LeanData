import Mathlib

namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l623_62365

theorem common_internal_tangent_length
  (center_distance : ℝ)
  (small_radius : ℝ)
  (large_radius : ℝ)
  (h1 : center_distance = 50)
  (h2 : small_radius = 7)
  (h3 : large_radius = 10) :
  Real.sqrt (center_distance ^ 2 - (small_radius + large_radius) ^ 2) = Real.sqrt 2211 :=
by sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l623_62365


namespace NUMINAMATH_CALUDE_triangle_midpoint_intersection_min_value_l623_62307

theorem triangle_midpoint_intersection_min_value (A B C D E M N : ℝ × ℝ) 
  (hAD : D = (A + B + C) / 3)  -- D is centroid of triangle ABC
  (hE : E = (A + D) / 2)       -- E is midpoint of AD
  (hM : ∃ x : ℝ, M = A + x • (B - A))  -- M is on AB
  (hN : ∃ y : ℝ, N = A + y • (C - A))  -- N is on AC
  (hEMN : ∃ t : ℝ, E = M + t • (N - M))  -- E, M, N are collinear
  : ∀ x y : ℝ, M = A + x • (B - A) → N = A + y • (C - A) → 4*x + y ≥ 9/4 :=
sorry

end NUMINAMATH_CALUDE_triangle_midpoint_intersection_min_value_l623_62307


namespace NUMINAMATH_CALUDE_smallest_n_is_83_l623_62347

def candy_problem (money : ℕ) : Prop :=
  ∃ (r g b : ℕ),
    money = 18 * r ∧
    money = 20 * g ∧
    money = 22 * b ∧
    money = 24 * 83 ∧
    ∀ (n : ℕ), n < 83 → money ≠ 24 * n

theorem smallest_n_is_83 :
  ∃ (money : ℕ), candy_problem money :=
sorry

end NUMINAMATH_CALUDE_smallest_n_is_83_l623_62347


namespace NUMINAMATH_CALUDE_jennifer_run_time_l623_62383

/-- 
Given:
- Jennifer ran 3 miles in 1/3 of the time it took Mark to run 5 miles
- Mark took 45 minutes to run 5 miles

Prove that Jennifer would take 35 minutes to run 7 miles at the same rate.
-/
theorem jennifer_run_time 
  (mark_distance : ℝ) 
  (mark_time : ℝ) 
  (jennifer_distance : ℝ) 
  (jennifer_time_ratio : ℝ) 
  (jennifer_new_distance : ℝ)
  (h1 : mark_distance = 5)
  (h2 : mark_time = 45)
  (h3 : jennifer_distance = 3)
  (h4 : jennifer_time_ratio = 1/3)
  (h5 : jennifer_new_distance = 7)
  : (jennifer_new_distance / jennifer_distance) * (jennifer_time_ratio * mark_time) = 35 := by
  sorry

#check jennifer_run_time

end NUMINAMATH_CALUDE_jennifer_run_time_l623_62383


namespace NUMINAMATH_CALUDE_road_trip_cost_l623_62377

theorem road_trip_cost (initial_friends : ℕ) (additional_friends : ℕ) (cost_decrease : ℚ) :
  initial_friends = 5 →
  additional_friends = 3 →
  cost_decrease = 15 →
  ∃ total_cost : ℚ,
    total_cost / initial_friends - total_cost / (initial_friends + additional_friends) = cost_decrease ∧
    total_cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_road_trip_cost_l623_62377


namespace NUMINAMATH_CALUDE_decimal_has_three_digits_l623_62342

-- Define the decimal number
def decimal : ℚ := 0.049

-- Theorem stating that the decimal has 3 digits after the decimal point
theorem decimal_has_three_digits : 
  (decimal * 1000).num % 1000 ≠ 0 ∧ (decimal * 100).num % 100 = 0 :=
sorry

end NUMINAMATH_CALUDE_decimal_has_three_digits_l623_62342


namespace NUMINAMATH_CALUDE_second_number_calculation_l623_62330

theorem second_number_calculation (first_number : ℝ) (second_number : ℝ) : 
  first_number = 640 → 
  (0.5 * first_number) = (0.2 * second_number + 190) → 
  second_number = 650 := by
sorry

end NUMINAMATH_CALUDE_second_number_calculation_l623_62330


namespace NUMINAMATH_CALUDE_temperature_drop_l623_62376

/-- Given an initial temperature and a temperature drop, calculates the final temperature -/
def finalTemperature (initial : Int) (drop : Int) : Int :=
  initial - drop

/-- Theorem: If the initial temperature is -6°C and it drops by 5°C, then the final temperature is -11°C -/
theorem temperature_drop : finalTemperature (-6) 5 = -11 := by
  sorry

end NUMINAMATH_CALUDE_temperature_drop_l623_62376


namespace NUMINAMATH_CALUDE_sum_in_base6_l623_62357

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (a b : ℕ) : ℕ := a * 6 + b

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : ℕ) : ℕ × ℕ × ℕ :=
  let hundreds := n / 36
  let remainder := n % 36
  let tens := remainder / 6
  let ones := remainder % 6
  (hundreds, tens, ones)

theorem sum_in_base6 :
  let a := base6ToBase10 3 5
  let b := base6ToBase10 2 5
  let sum := a + b
  let (h, t, o) := base10ToBase6 sum
  h = 1 ∧ t = 0 ∧ o = 4 := by sorry

end NUMINAMATH_CALUDE_sum_in_base6_l623_62357


namespace NUMINAMATH_CALUDE_roi_difference_emma_briana_l623_62334

/-- Calculates the difference in return-on-investment between two investors after a given time period. -/
def roi_difference (emma_investment briana_investment : ℝ) 
                   (emma_yield_rate briana_yield_rate : ℝ) 
                   (years : ℕ) : ℝ :=
  (briana_investment * briana_yield_rate * years) - (emma_investment * emma_yield_rate * years)

/-- Theorem stating the difference in return-on-investment between Briana and Emma after 2 years. -/
theorem roi_difference_emma_briana : 
  roi_difference 300 500 0.15 0.10 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_roi_difference_emma_briana_l623_62334


namespace NUMINAMATH_CALUDE_negative_of_difference_l623_62303

theorem negative_of_difference (a b : ℝ) : -(a - b) = -a + b := by sorry

end NUMINAMATH_CALUDE_negative_of_difference_l623_62303


namespace NUMINAMATH_CALUDE_missing_village_population_l623_62301

def village_populations : List ℕ := [803, 900, 1100, 1023, 945, 1249]
def total_villages : ℕ := 7
def average_population : ℕ := 1000

theorem missing_village_population :
  (village_populations.sum + (total_villages * average_population - village_populations.sum)) / total_villages = average_population ∧
  total_villages * average_population - village_populations.sum = 980 :=
by sorry

end NUMINAMATH_CALUDE_missing_village_population_l623_62301


namespace NUMINAMATH_CALUDE_remainder_sum_l623_62359

theorem remainder_sum (c d : ℤ) (hc : c % 60 = 58) (hd : d % 90 = 85) :
  (c + d) % 30 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l623_62359


namespace NUMINAMATH_CALUDE_race_course_length_60m_l623_62390

/-- Represents the race scenario with three runners -/
structure RaceScenario where
  speedB : ℝ     -- Speed of runner B (base speed)
  speedA : ℝ     -- Speed of runner A
  speedC : ℝ     -- Speed of runner C
  headStartA : ℝ -- Head start given by A to B
  headStartC : ℝ -- Head start given by C to B

/-- Calculates the race course length for simultaneous finish -/
def calculateRaceCourseLength (race : RaceScenario) : ℝ :=
  sorry

/-- Theorem stating the race course length for the given scenario -/
theorem race_course_length_60m :
  ∀ (v : ℝ), v > 0 →
  let race : RaceScenario :=
    { speedB := v
      speedA := 4 * v
      speedC := 2 * v
      headStartA := 60
      headStartC := 30 }
  calculateRaceCourseLength race = 60 :=
sorry

end NUMINAMATH_CALUDE_race_course_length_60m_l623_62390


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_unit_interval_l623_62351

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}
def N : Set ℝ := {x | x^2 + 2*x - 3 ≥ 0}

-- State the theorem
theorem M_intersect_N_equals_unit_interval :
  M ∩ N = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_unit_interval_l623_62351


namespace NUMINAMATH_CALUDE_no_prime_roots_for_specific_quadratic_l623_62356

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem no_prime_roots_for_specific_quadratic :
  ¬ ∃ (k : ℤ) (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p ≠ q ∧
    p + q = 97 ∧ 
    p * q = k ∧
    ∀ (x : ℤ), x^2 - 97*x + k = 0 ↔ (x = p ∨ x = q) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_specific_quadratic_l623_62356


namespace NUMINAMATH_CALUDE_solution_is_ten_l623_62315

-- Define the * operation
def star (a b : ℝ) : ℝ := 3 * a - b

-- State the theorem
theorem solution_is_ten :
  ∃ x : ℝ, star 2 (star 5 x) = 1 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_ten_l623_62315


namespace NUMINAMATH_CALUDE_final_result_calculation_l623_62364

theorem final_result_calculation (chosen_number : ℕ) : 
  chosen_number = 60 → (chosen_number * 4 - 138 = 102) := by
  sorry

end NUMINAMATH_CALUDE_final_result_calculation_l623_62364


namespace NUMINAMATH_CALUDE_fourth_circle_radius_l623_62346

theorem fourth_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 23) (h₂ : r₂ = 35) (h₃ : r₃ = Real.sqrt 1754) :
  π * r₃^2 = π * r₁^2 + π * r₂^2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_l623_62346


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l623_62338

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l623_62338


namespace NUMINAMATH_CALUDE_percentage_of_120_to_40_l623_62336

theorem percentage_of_120_to_40 : ∀ (x y : ℝ), x = 120 ∧ y = 40 → (x / y) * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_120_to_40_l623_62336


namespace NUMINAMATH_CALUDE_polynomial_sum_l623_62352

def is_monic_degree_4 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_sum (p : ℝ → ℝ) :
  is_monic_degree_4 p →
  p 1 = 17 →
  p 2 = 38 →
  p 3 = 63 →
  p 0 + p 4 = 68 :=
by
  sorry


end NUMINAMATH_CALUDE_polynomial_sum_l623_62352


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l623_62311

theorem cubic_root_ratio (p q r s : ℝ) (h : ∀ x, p * x^3 + q * x^2 + r * x + s = 0 ↔ x = -1 ∨ x = 3 ∨ x = 4) :
  r / s = -5 / 12 := by sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l623_62311


namespace NUMINAMATH_CALUDE_road_cost_calculation_l623_62323

theorem road_cost_calculation (lawn_length lawn_width road_length_width road_width_width : ℕ)
  (cost_length cost_width : ℚ) : 
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_length_width = 12 ∧ 
  road_width_width = 15 ∧ 
  cost_length = 3 ∧ 
  cost_width = (5/2) →
  (lawn_length * road_length_width * cost_length + 
   lawn_width * road_width_width * cost_width : ℚ) = 5130 :=
by sorry

end NUMINAMATH_CALUDE_road_cost_calculation_l623_62323


namespace NUMINAMATH_CALUDE_alice_exam_score_l623_62368

theorem alice_exam_score (exam1 exam2 exam3 : ℕ) 
  (h1 : exam1 = 85) (h2 : exam2 = 76) (h3 : exam3 = 83)
  (h4 : ∀ exam, exam ≤ 100) : 
  ∃ (exam4 exam5 : ℕ), 
    exam4 ≤ 100 ∧ exam5 ≤ 100 ∧ 
    (exam1 + exam2 + exam3 + exam4 + exam5) / 5 = 80 ∧
    (exam4 = 56 ∨ exam5 = 56) ∧
    ∀ (x : ℕ), x < 56 → 
      ¬∃ (y : ℕ), y ≤ 100 ∧ (exam1 + exam2 + exam3 + x + y) / 5 = 80 :=
by sorry

end NUMINAMATH_CALUDE_alice_exam_score_l623_62368


namespace NUMINAMATH_CALUDE_jennifer_garden_max_area_l623_62372

/-- Represents a rectangular garden with integer side lengths. -/
structure RectangularGarden where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangular garden. -/
def perimeter (g : RectangularGarden) : ℕ := 2 * (g.length + g.width)

/-- Calculates the area of a rectangular garden. -/
def area (g : RectangularGarden) : ℕ := g.length * g.width

/-- Theorem stating the maximum area of Jennifer's garden. -/
theorem jennifer_garden_max_area :
  ∃ (g : RectangularGarden),
    g.length = 30 ∧
    perimeter g = 160 ∧
    (∀ (h : RectangularGarden), h.length = 30 ∧ perimeter h = 160 → area h ≤ area g) ∧
    area g = 1500 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_garden_max_area_l623_62372


namespace NUMINAMATH_CALUDE_opposite_definition_opposite_of_eight_l623_62335

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The opposite of a number added to the original number equals zero -/
theorem opposite_definition (x : ℝ) : x + opposite x = 0 := by sorry

/-- The opposite of 8 is -8 -/
theorem opposite_of_eight : opposite 8 = -8 := by sorry

end NUMINAMATH_CALUDE_opposite_definition_opposite_of_eight_l623_62335


namespace NUMINAMATH_CALUDE_ruths_sandwiches_l623_62360

theorem ruths_sandwiches (total : ℕ) (brother : ℕ) (first_cousin : ℕ) (other_cousins : ℕ) (left : ℕ) :
  total = 10 →
  brother = 2 →
  first_cousin = 2 →
  other_cousins = 2 →
  left = 3 →
  total - (brother + first_cousin + other_cousins + left) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ruths_sandwiches_l623_62360


namespace NUMINAMATH_CALUDE_expression_evaluation_l623_62318

theorem expression_evaluation : 
  -14 - (-2)^3 * (1/4) - 16 * ((1/2) - (1/4) + (3/8)) = -22 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l623_62318


namespace NUMINAMATH_CALUDE_dacid_weighted_average_l623_62394

/-- Calculates the weighted average grade given marks and weights -/
def weighted_average (marks : List ℝ) (weights : List ℝ) : ℝ :=
  (List.zip marks weights).map (fun (m, w) => m * w) |>.sum

/-- Theorem: Dacid's weighted average grade is 90.8 -/
theorem dacid_weighted_average :
  let marks := [96, 95, 82, 87, 92]
  let weights := [0.20, 0.25, 0.15, 0.25, 0.15]
  weighted_average marks weights = 90.8 := by
sorry

#eval weighted_average [96, 95, 82, 87, 92] [0.20, 0.25, 0.15, 0.25, 0.15]

end NUMINAMATH_CALUDE_dacid_weighted_average_l623_62394


namespace NUMINAMATH_CALUDE_min_style_A_purchase_correct_l623_62300

/-- Represents the clothing store problem -/
structure ClothingStore where
  total_pieces : ℕ
  total_cost : ℕ
  unit_price_A : ℕ
  unit_price_B : ℕ
  selling_price_A : ℕ
  selling_price_B : ℕ
  other_store_purchase : ℕ
  other_store_min_profit : ℕ

/-- The minimum number of style A clothing pieces to be purchased by another store -/
def min_style_A_purchase (store : ClothingStore) : ℕ :=
  23

/-- Theorem stating that the minimum number of style A clothing pieces to be purchased
    by another store is correct given the conditions -/
theorem min_style_A_purchase_correct (store : ClothingStore)
  (h1 : store.total_pieces = 100)
  (h2 : store.total_cost = 11200)
  (h3 : store.unit_price_A = 120)
  (h4 : store.unit_price_B = 100)
  (h5 : store.selling_price_A = 200)
  (h6 : store.selling_price_B = 140)
  (h7 : store.other_store_purchase = 60)
  (h8 : store.other_store_min_profit = 3300) :
  ∀ m : ℕ, m ≥ min_style_A_purchase store →
    (store.selling_price_A - store.unit_price_A) * m +
    (store.selling_price_B - store.unit_price_B) * (store.other_store_purchase - m) ≥
    store.other_store_min_profit :=
  sorry

end NUMINAMATH_CALUDE_min_style_A_purchase_correct_l623_62300


namespace NUMINAMATH_CALUDE_peach_ripeness_difference_l623_62302

def bowl_of_peaches (total_peaches initial_ripe ripening_rate days_passed peaches_eaten : ℕ) : ℕ :=
  let ripe_peaches := initial_ripe + ripening_rate * days_passed - peaches_eaten
  let unripe_peaches := total_peaches - ripe_peaches
  ripe_peaches - unripe_peaches

theorem peach_ripeness_difference :
  bowl_of_peaches 18 4 2 5 3 = 4 := by
  sorry

#eval bowl_of_peaches 18 4 2 5 3

end NUMINAMATH_CALUDE_peach_ripeness_difference_l623_62302


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l623_62396

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 1)^8 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3 + 
    a₄*(x-2)^4 + a₅*(x-2)^5 + a₆*(x-2)^6 + a₇*(x-2)^7 + a₈*(x-2)^8 + a₉*(x-2)^9 + a₁₀*(x-2)^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 2555 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l623_62396


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_a_negative_curve_intersection_not_one_l623_62391

/-- Represents a quadratic equation of the form x^2 + (a-3)x + a = 0 -/
def QuadraticEquation (a : ℝ) := λ x : ℝ => x^2 + (a-3)*x + a

/-- Represents the curve y = |3-x^2| -/
def Curve := λ x : ℝ => |3 - x^2|

theorem quadratic_roots_imply_a_negative (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ QuadraticEquation a x = 0 ∧ QuadraticEquation a y = 0) →
  a < 0 :=
sorry

theorem curve_intersection_not_one (a : ℝ) :
  ¬(∃! x : ℝ, Curve x = a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_a_negative_curve_intersection_not_one_l623_62391


namespace NUMINAMATH_CALUDE_percentage_problem_l623_62340

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 40 = 4 / 5 * 25 + 6 → P = 65 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l623_62340


namespace NUMINAMATH_CALUDE_cubic_value_in_set_l623_62316

theorem cubic_value_in_set (A : Set ℝ) (a : ℝ) 
  (h1 : 5 ∈ A) 
  (h2 : a^2 + 2*a + 4 ∈ A) 
  (h3 : 7 ∈ A) : 
  a^3 = 1 ∨ a^3 = -27 := by
sorry

end NUMINAMATH_CALUDE_cubic_value_in_set_l623_62316


namespace NUMINAMATH_CALUDE_parallel_vectors_x_equals_one_l623_62385

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_equals_one :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (2*x + 1, 3)
  let b : ℝ × ℝ := (2 - x, 1)
  parallel a b → x = 1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_equals_one_l623_62385


namespace NUMINAMATH_CALUDE_correct_subtraction_l623_62366

theorem correct_subtraction (x : ℤ) (h : x - 63 = 8) : x - 36 = 35 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l623_62366


namespace NUMINAMATH_CALUDE_repeating_decimal_equiv_l623_62398

-- Define the repeating decimal 0.4̄13
def repeating_decimal : ℚ := 409 / 990

-- Theorem statement
theorem repeating_decimal_equiv : 
  repeating_decimal = 409 / 990 ∧ 
  (∀ n d : ℕ, n / d = 409 / 990 → d ≤ 990) :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equiv_l623_62398


namespace NUMINAMATH_CALUDE_custom_operation_result_l623_62326

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a^2 - a*b

-- State the theorem
theorem custom_operation_result : star (star (-1) 2) 3 = 0 := by sorry

end NUMINAMATH_CALUDE_custom_operation_result_l623_62326


namespace NUMINAMATH_CALUDE_quadratic_inequality_l623_62392

theorem quadratic_inequality (a b : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0) :
  ∃ n : ℤ, |n^2 + a*n + b| ≤ max (1/4 : ℝ) ((1/2 : ℝ) * Real.sqrt (a^2 - 4*b)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l623_62392


namespace NUMINAMATH_CALUDE_least_sum_of_equal_multiples_l623_62329

theorem least_sum_of_equal_multiples (x y z : ℕ+) (h : (2 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val) :
  ∃ (a b c : ℕ+), (2 : ℕ) * a.val = (5 : ℕ) * b.val ∧ (5 : ℕ) * b.val = (6 : ℕ) * c.val ∧
  (∀ (p q r : ℕ+), (2 : ℕ) * p.val = (5 : ℕ) * q.val ∧ (5 : ℕ) * q.val = (6 : ℕ) * r.val →
    a.val + b.val + c.val ≤ p.val + q.val + r.val) ∧
  a.val + b.val + c.val = 26 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_equal_multiples_l623_62329


namespace NUMINAMATH_CALUDE_fencing_tournament_l623_62314

theorem fencing_tournament (n : ℕ) : n > 0 → (
  let total_participants := 4*n
  let total_bouts := (total_participants * (total_participants - 1)) / 2
  let womens_wins := 2*n*(3*n)
  let mens_wins := 3*n*(n + 3*n - 1)
  womens_wins * 3 = mens_wins * 2 ∧ womens_wins + mens_wins = total_bouts
) → n = 4 := by sorry

end NUMINAMATH_CALUDE_fencing_tournament_l623_62314


namespace NUMINAMATH_CALUDE_f_max_value_l623_62381

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (x^2)

theorem f_max_value :
  ∃ (x_max : ℝ), x_max > 0 ∧
  (∀ (x : ℝ), x > 0 → f x ≤ f x_max) ∧
  x_max = Real.sqrt (Real.exp 1) ∧
  f x_max = 1 / (2 * Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l623_62381


namespace NUMINAMATH_CALUDE_smallest_max_sum_l623_62361

theorem smallest_max_sum (p q r s t : ℕ+) 
  (sum_condition : p + q + r + s + t = 4020) : 
  (∃ (N : ℕ), 
    N = max (p + q) (max (q + r) (max (r + s) (s + t))) ∧ 
    (∀ (M : ℕ), M = max (p + q) (max (q + r) (max (r + s) (s + t))) → N ≤ M) ∧
    N = 1005) := by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l623_62361


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l623_62399

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex nonagon has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l623_62399


namespace NUMINAMATH_CALUDE_expected_faces_six_die_six_rolls_l623_62324

/-- The number of sides on the die -/
def n : ℕ := 6

/-- The number of rolls -/
def k : ℕ := 6

/-- The probability of a specific face not appearing in a single roll -/
def p : ℚ := (n - 1) / n

/-- The expected number of different faces appearing when rolling an n-sided die k times -/
def expected_faces : ℚ := n * (1 - p^k)

/-- Theorem: The expected number of different faces appearing when a fair six-sided die 
    is rolled six times is equal to (6^6 - 5^6) / 6^5 -/
theorem expected_faces_six_die_six_rolls : 
  expected_faces = (n^k - (n-1)^k) / n^(k-1) := by
  sorry

#eval expected_faces

end NUMINAMATH_CALUDE_expected_faces_six_die_six_rolls_l623_62324


namespace NUMINAMATH_CALUDE_complex_square_roots_l623_62378

theorem complex_square_roots (z : ℂ) : z^2 = -77 - 36*I ↔ z = 2 - 9*I ∨ z = -2 + 9*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_roots_l623_62378


namespace NUMINAMATH_CALUDE_article_price_decrease_l623_62327

theorem article_price_decrease (P : ℝ) : 
  (P * (1 - 0.24) * (1 - 0.10) = 760) → 
  ∃ ε > 0, |P - 111| < ε :=
sorry

end NUMINAMATH_CALUDE_article_price_decrease_l623_62327


namespace NUMINAMATH_CALUDE_rectangle_length_width_difference_l623_62371

theorem rectangle_length_width_difference 
  (length width : ℝ) 
  (h1 : length = 6)
  (h2 : width = 4)
  (h3 : 2 * (length + width) = 20)
  (h4 : ∃ d : ℝ, length = width + d) : 
  length - width = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_width_difference_l623_62371


namespace NUMINAMATH_CALUDE_f_sum_logs_l623_62332

-- Define the function f
def f (x : ℝ) : ℝ := 1 + x^3

-- State the theorem
theorem f_sum_logs : f (Real.log 2) + f (Real.log (1/2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_logs_l623_62332


namespace NUMINAMATH_CALUDE_sasha_plucked_leaves_l623_62344

/-- The number of leaves Sasha plucked -/
def leaves_plucked : ℕ := 22

/-- The number of apple trees -/
def apple_trees : ℕ := 17

/-- The number of poplar trees -/
def poplar_trees : ℕ := 18

/-- The position of the apple tree after which Masha's phone memory was full -/
def masha_last_photo : ℕ := 10

/-- The number of trees that remained unphotographed by Masha -/
def unphotographed_trees : ℕ := 13

/-- The position of the apple tree from which Sasha started plucking leaves -/
def sasha_start : ℕ := 8

theorem sasha_plucked_leaves : 
  apple_trees = 17 ∧ 
  poplar_trees = 18 ∧ 
  masha_last_photo = 10 ∧ 
  unphotographed_trees = 13 ∧ 
  sasha_start = 8 → 
  leaves_plucked = 22 := by
  sorry

end NUMINAMATH_CALUDE_sasha_plucked_leaves_l623_62344


namespace NUMINAMATH_CALUDE_work_scaling_l623_62393

theorem work_scaling (people₁ work₁ days : ℕ) (people₂ : ℕ) :
  people₁ > 0 →
  work₁ > 0 →
  days > 0 →
  (people₁ * work₁ = people₁ * people₁) →
  people₂ = people₁ * (people₂ / people₁) →
  (people₂ / people₁ : ℚ) * work₁ = people₂ / people₁ * people₁ :=
by sorry

end NUMINAMATH_CALUDE_work_scaling_l623_62393


namespace NUMINAMATH_CALUDE_rectangle_area_l623_62350

/-- Rectangle PQRS with given coordinates and properties -/
structure Rectangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  S : ℝ × ℝ
  is_rectangle : Bool

/-- The area of the rectangle PQRS is 200000 -/
theorem rectangle_area (rect : Rectangle) : 
  rect.P = (-15, 30) →
  rect.Q = (985, 230) →
  rect.S.1 = -13 →
  rect.is_rectangle = true →
  (rect.Q.1 - rect.P.1) * (rect.S.2 - rect.P.2) = 200000 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_l623_62350


namespace NUMINAMATH_CALUDE_used_car_clients_l623_62382

theorem used_car_clients (num_cars : ℕ) (selections_per_car : ℕ) (cars_per_client : ℕ)
  (h_num_cars : num_cars = 16)
  (h_selections_per_car : selections_per_car = 3)
  (h_cars_per_client : cars_per_client = 2) :
  (num_cars * selections_per_car) / cars_per_client = 24 := by
  sorry

end NUMINAMATH_CALUDE_used_car_clients_l623_62382


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l623_62386

/-- Given an arithmetic sequence {aₙ} with S₃ = 6 and a₃ = 4, prove that the common difference d = 2 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sequence of partial sums
  (h1 : S 3 = 6)  -- Given S₃ = 6
  (h2 : a 3 = 4)  -- Given a₃ = 4
  (h3 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)  -- Sum formula for arithmetic sequence
  (h4 : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)  -- Definition of arithmetic sequence
  : ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l623_62386


namespace NUMINAMATH_CALUDE_binomial_variance_10_07_l623_62380

/-- The variance of a binomial distribution with 10 trials and 0.7 probability of success is 2.1 -/
theorem binomial_variance_10_07 :
  let n : ℕ := 10
  let p : ℝ := 0.7
  let variance := n * p * (1 - p)
  variance = 2.1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_10_07_l623_62380


namespace NUMINAMATH_CALUDE_geraldo_tea_consumption_l623_62375

-- Define the conversion factor from gallons to pints
def gallons_to_pints : ℝ := 8

-- Define the total amount of tea in gallons
def total_tea : ℝ := 20

-- Define the number of containers
def num_containers : ℝ := 80

-- Define the number of containers Geraldo drank
def containers_drunk : ℝ := 3.5

-- Theorem statement
theorem geraldo_tea_consumption :
  (total_tea / num_containers) * containers_drunk * gallons_to_pints = 7 := by
  sorry

end NUMINAMATH_CALUDE_geraldo_tea_consumption_l623_62375


namespace NUMINAMATH_CALUDE_some_number_equals_37_l623_62343

theorem some_number_equals_37 : ∃ x : ℤ, 45 - (28 - (x - (15 - 20))) = 59 ∧ x = 37 := by
  sorry

end NUMINAMATH_CALUDE_some_number_equals_37_l623_62343


namespace NUMINAMATH_CALUDE_fencing_requirement_l623_62331

/-- A rectangular field with specific properties. -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  uncovered_side : ℝ

/-- The theorem stating the fencing requirement for the given field. -/
theorem fencing_requirement (field : RectangularField) 
  (h1 : field.area = 680)
  (h2 : field.uncovered_side = 80)
  (h3 : field.area = field.length * field.width)
  (h4 : field.length = field.uncovered_side) :
  2 * field.width + field.uncovered_side = 97 := by
  sorry

end NUMINAMATH_CALUDE_fencing_requirement_l623_62331


namespace NUMINAMATH_CALUDE_shortest_chord_through_M_l623_62322

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 2*y + 10 = 0

-- Define point M
def point_M : ℝ × ℝ := (3, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + y - 3 = 0

-- Theorem statement
theorem shortest_chord_through_M :
  ∀ (l : ℝ × ℝ → Prop),
    (∀ x y, l (x, y) ↔ line_equation x y) →
    (l point_M) →
    (∀ other_line : ℝ × ℝ → Prop,
      (other_line point_M) →
      (∃ p, circle_equation p.1 p.2 ∧ other_line p) →
      (∃ p q : ℝ × ℝ, 
        p ≠ q ∧ 
        circle_equation p.1 p.2 ∧ circle_equation q.1 q.2 ∧ 
        l p ∧ l q ∧
        other_line p ∧ other_line q →
        (p.1 - q.1)^2 + (p.2 - q.2)^2 ≤ 
        (p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
  sorry

end NUMINAMATH_CALUDE_shortest_chord_through_M_l623_62322


namespace NUMINAMATH_CALUDE_highway_extension_proof_l623_62337

def highway_extension (current_length final_length first_day_miles : ℕ) : Prop :=
  let second_day_miles := 3 * first_day_miles
  let total_built := first_day_miles + second_day_miles
  let total_extension := final_length - current_length
  let remaining_miles := total_extension - total_built
  remaining_miles = 250

theorem highway_extension_proof :
  highway_extension 200 650 50 :=
sorry

end NUMINAMATH_CALUDE_highway_extension_proof_l623_62337


namespace NUMINAMATH_CALUDE_triangle_side_length_l623_62384

/-- Given a triangle ABC with perimeter √2 + 1 and sin A + sin B = √2 sin C, 
    prove that the length of side AB is 1 -/
theorem triangle_side_length 
  (A B C : ℝ) 
  (perimeter : ℝ) 
  (h_perimeter : perimeter = Real.sqrt 2 + 1)
  (h_sin_sum : Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C)
  (h_triangle : A + B + C = π)
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C)
  : ∃ (a b c : ℝ), a + b + c = perimeter ∧ 
                    a = 1 ∧
                    a / Real.sin A = b / Real.sin B ∧
                    b / Real.sin B = c / Real.sin C :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l623_62384


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l623_62367

theorem product_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 24)
  (first_eq : x = 3 * (y + z))
  (second_eq : y = 6 * z) :
  x * y * z = 126 := by sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l623_62367


namespace NUMINAMATH_CALUDE_mary_book_count_l623_62306

def book_count (initial : ℕ) (book_club : ℕ) (lent_jane : ℕ) (returned_alice : ℕ)
  (bought_5th_month : ℕ) (bought_yard_sales : ℕ) (birthday_daughter : ℕ)
  (birthday_mother : ℕ) (from_sister : ℕ) (buy_one_get_one : ℕ)
  (donated_charity : ℕ) (borrowed_neighbor : ℕ) (sold_used : ℕ) : ℕ :=
  initial + book_club - lent_jane + returned_alice + bought_5th_month +
  bought_yard_sales + birthday_daughter + birthday_mother + from_sister +
  buy_one_get_one - donated_charity - borrowed_neighbor - sold_used

theorem mary_book_count :
  book_count 200 (2 * 12) 10 5 15 8 1 8 6 4 30 5 7 = 219 := by
  sorry

end NUMINAMATH_CALUDE_mary_book_count_l623_62306


namespace NUMINAMATH_CALUDE_number_of_female_employees_l623_62304

/-- Given a company with employees, prove that the number of female employees is 500 -/
theorem number_of_female_employees 
  (E : ℕ) -- Total number of employees
  (F : ℕ) -- Number of female employees
  (M : ℕ) -- Number of male employees
  (h1 : E = F + M) -- Total employees is sum of female and male employees
  (h2 : (2 : ℚ) / 5 * E = 200 + (2 : ℚ) / 5 * M) -- Equation for total managers
  (h3 : (200 : ℚ) = F - (2 : ℚ) / 5 * M) -- Equation for female managers
  : F = 500 := by
  sorry

end NUMINAMATH_CALUDE_number_of_female_employees_l623_62304


namespace NUMINAMATH_CALUDE_alexis_shopping_problem_l623_62358

/-- Alexis's shopping problem -/
theorem alexis_shopping_problem (budget initial_amount remaining_amount shirt_cost pants_cost socks_cost belt_cost shoes_cost : ℕ) 
  (h1 : initial_amount = 200)
  (h2 : shirt_cost = 30)
  (h3 : pants_cost = 46)
  (h4 : socks_cost = 11)
  (h5 : belt_cost = 18)
  (h6 : shoes_cost = 41)
  (h7 : remaining_amount = 16)
  (h8 : budget = initial_amount - remaining_amount) :
  budget - (shirt_cost + pants_cost + socks_cost + belt_cost + shoes_cost) = 38 := by
  sorry

#check alexis_shopping_problem

end NUMINAMATH_CALUDE_alexis_shopping_problem_l623_62358


namespace NUMINAMATH_CALUDE_football_joins_l623_62345

theorem football_joins (pentagonal_panels hexagonal_panels : ℕ) 
  (pentagonal_edges hexagonal_edges : ℕ) : 
  pentagonal_panels = 12 →
  hexagonal_panels = 20 →
  pentagonal_edges = 5 →
  hexagonal_edges = 6 →
  (pentagonal_panels * pentagonal_edges + hexagonal_panels * hexagonal_edges) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_football_joins_l623_62345


namespace NUMINAMATH_CALUDE_inequality_proof_l623_62339

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 0 < b) (hb' : b < 1) :
  (a * b * (1 - a) * (1 - b)) / ((1 - a * b)^2) < (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l623_62339


namespace NUMINAMATH_CALUDE_other_number_proof_l623_62321

theorem other_number_proof (a b : ℕ) (h1 : a + b = 62) (h2 : a = 27) : b = 35 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l623_62321


namespace NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l623_62309

open Real

theorem function_inequality_implies_upper_bound (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = x * log x) →
  (∀ x > 0, f x ≥ -x^2 + a*x - 6) →
  a ≤ 5 + log 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l623_62309


namespace NUMINAMATH_CALUDE_product_75_180_trailing_zeros_l623_62353

/-- The number of trailing zeros in the product of two positive integers -/
def trailingZeros (a b : ℕ+) : ℕ :=
  sorry

/-- Theorem: The number of trailing zeros in the product of 75 and 180 is 2 -/
theorem product_75_180_trailing_zeros :
  trailingZeros 75 180 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_75_180_trailing_zeros_l623_62353


namespace NUMINAMATH_CALUDE_min_A_mats_l623_62348

/-- Represents the purchase and sale of bamboo mats -/
structure BambooMatSale where
  purchase_price_A : ℝ
  purchase_price_B : ℝ
  sale_price_A : ℝ
  sale_price_B : ℝ

/-- The conditions of the bamboo mat sale problem -/
def bamboo_mat_conditions (s : BambooMatSale) : Prop :=
  10 * s.purchase_price_A + 15 * s.purchase_price_B = 3600 ∧
  25 * s.purchase_price_A + 30 * s.purchase_price_B = 8100 ∧
  s.sale_price_A = 260 ∧
  s.sale_price_B = 180

/-- The profit calculation for a given number of mats A -/
def profit (s : BambooMatSale) (num_A : ℝ) : ℝ :=
  (s.sale_price_A - s.purchase_price_A) * num_A +
  (s.sale_price_B - s.purchase_price_B) * (60 - num_A)

/-- The main theorem stating the minimum number of A mats to purchase -/
theorem min_A_mats (s : BambooMatSale) 
  (h : bamboo_mat_conditions s) : 
  ∃ (n : ℕ), n = 40 ∧ 
  (∀ (m : ℕ), m ≥ 40 → profit s m ≥ 4400) ∧
  (∀ (m : ℕ), m < 40 → profit s m < 4400) := by
  sorry

end NUMINAMATH_CALUDE_min_A_mats_l623_62348


namespace NUMINAMATH_CALUDE_mabel_steps_to_helen_l623_62373

/-- The total number of steps Mabel walks to visit Helen -/
def total_steps (mabel_distance helen_fraction : ℕ) : ℕ :=
  mabel_distance + (helen_fraction * mabel_distance) / 4

/-- Proof that Mabel walks 7875 steps to visit Helen -/
theorem mabel_steps_to_helen :
  total_steps 4500 3 = 7875 := by
  sorry

end NUMINAMATH_CALUDE_mabel_steps_to_helen_l623_62373


namespace NUMINAMATH_CALUDE_sum_of_binary_numbers_l623_62369

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101₍₂₎ -/
def binary_101 : List Bool := [true, false, true]

/-- The binary representation of 110₍₂₎ -/
def binary_110 : List Bool := [false, true, true]

theorem sum_of_binary_numbers :
  binary_to_decimal binary_101 + binary_to_decimal binary_110 = 11 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_binary_numbers_l623_62369


namespace NUMINAMATH_CALUDE_complex_product_real_implies_a_equals_one_l623_62374

theorem complex_product_real_implies_a_equals_one (a : ℝ) :
  ((1 + Complex.I) * (1 - a * Complex.I)).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_implies_a_equals_one_l623_62374


namespace NUMINAMATH_CALUDE_sin_eleven_pi_thirds_l623_62349

theorem sin_eleven_pi_thirds : Real.sin (11 * π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_eleven_pi_thirds_l623_62349


namespace NUMINAMATH_CALUDE_stock_price_calculation_l623_62319

/-- Proves that given an income of 15000 from an 80% stock and an investment of 37500,
    the price of the stock is 50% of its face value. -/
theorem stock_price_calculation 
  (income : ℝ) 
  (investment : ℝ) 
  (yield : ℝ) 
  (h1 : income = 15000) 
  (h2 : investment = 37500) 
  (h3 : yield = 80) : 
  (income * 100 / (investment * yield)) = 0.5 := by
sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l623_62319


namespace NUMINAMATH_CALUDE_polygon_sides_l623_62354

/-- The number of sides of a polygon given the difference between its interior and exterior angle sums -/
theorem polygon_sides (interior_exterior_diff : ℝ) : interior_exterior_diff = 540 → ∃ n : ℕ, n = 7 ∧ 
  (n - 2) * 180 = 360 + interior_exterior_diff ∧ 
  (∀ m : ℕ, (m - 2) * 180 = 360 + interior_exterior_diff → m = n) := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l623_62354


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l623_62341

def Circle (center : ℝ × ℝ) (radius : ℝ) := { p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 }

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2

theorem intersection_distance_squared :
  let circle1 := Circle (5, 0) 5
  let circle2 := Circle (0, 5) 5
  ∀ C D : ℝ × ℝ, C ∈ circle1 ∧ C ∈ circle2 ∧ D ∈ circle1 ∧ D ∈ circle2 ∧ C ≠ D →
  distance_squared C D = 50 := by
sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_l623_62341


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l623_62355

/-- The ellipse defined by 2x^2 + 3y^2 = 12 -/
def Ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 12

/-- The function to be maximized -/
def f (x y : ℝ) : ℝ := x + 2 * y

/-- Theorem stating that the maximum value of x + 2y on the given ellipse is √22 -/
theorem max_value_on_ellipse :
  ∃ (max : ℝ), max = Real.sqrt 22 ∧
  (∀ x y : ℝ, Ellipse x y → f x y ≤ max) ∧
  (∃ x y : ℝ, Ellipse x y ∧ f x y = max) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l623_62355


namespace NUMINAMATH_CALUDE_mono_decreasing_g_l623_62370

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f being monotonically increasing on [1, 2]
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x ≤ y → f x ≤ f y

-- Define the function g(x) = f(1-x)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (1 - x)

-- State the theorem
theorem mono_decreasing_g (h : MonoIncreasing f) :
  ∀ x y, x ∈ Set.Icc (-1) 0 → y ∈ Set.Icc (-1) 0 → x ≤ y → g f y ≤ g f x :=
sorry

end NUMINAMATH_CALUDE_mono_decreasing_g_l623_62370


namespace NUMINAMATH_CALUDE_matrix_power_property_l623_62310

theorem matrix_power_property (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A.mulVec (![5, -2]) = ![(-15), 6] →
  (A ^ 5).mulVec (![5, -2]) = ![(-1215), 486] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_property_l623_62310


namespace NUMINAMATH_CALUDE_pizza_flour_calculation_l623_62362

theorem pizza_flour_calculation (bases : ℕ) (total_flour : ℚ) : 
  bases = 15 → total_flour = 8 → (total_flour / bases : ℚ) = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_pizza_flour_calculation_l623_62362


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l623_62328

/-- A quadratic function passing through points (0, y₁) and (4, y₂) -/
def quadratic_function (c y₁ y₂ : ℝ) : Prop :=
  y₁ = c ∧ y₂ = 16 - 24 + c

/-- Theorem stating that y₁ > y₂ for the given quadratic function -/
theorem y1_greater_than_y2 (c y₁ y₂ : ℝ) (h : quadratic_function c y₁ y₂) : y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l623_62328


namespace NUMINAMATH_CALUDE_bob_candies_l623_62317

/-- Given that Jennifer bought twice as many candies as Emily, Jennifer bought three times as many
    candies as Bob, and Emily bought 6 candies, prove that Bob bought 4 candies. -/
theorem bob_candies (emily_candies : ℕ) (jennifer_candies : ℕ) (bob_candies : ℕ)
  (h1 : jennifer_candies = 2 * emily_candies)
  (h2 : jennifer_candies = 3 * bob_candies)
  (h3 : emily_candies = 6) :
  bob_candies = 4 := by
  sorry

end NUMINAMATH_CALUDE_bob_candies_l623_62317


namespace NUMINAMATH_CALUDE_specific_conference_games_l623_62363

/-- Calculates the number of games in a sports conference season -/
def conference_games (total_teams : ℕ) (divisions : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  ((total_teams * (teams_per_division - 1) * intra_division_games + 
    total_teams * teams_per_division * inter_division_games) / 2)

/-- The number of games in a specific conference setup -/
theorem specific_conference_games : 
  conference_games 14 2 7 2 1 = 133 := by sorry

end NUMINAMATH_CALUDE_specific_conference_games_l623_62363


namespace NUMINAMATH_CALUDE_sandwich_jam_cost_l623_62395

theorem sandwich_jam_cost :
  ∀ (N B J : ℕ),
  N > 1 →
  B > 0 →
  J > 0 →
  N * (3 * B + 7 * J) = 378 →
  (N * J * 7 : ℚ) / 100 = 2.52 := by
sorry

end NUMINAMATH_CALUDE_sandwich_jam_cost_l623_62395


namespace NUMINAMATH_CALUDE_arithmetic_computation_l623_62305

theorem arithmetic_computation : 6^2 + 2*(5) - 4^2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l623_62305


namespace NUMINAMATH_CALUDE_diminished_value_is_seven_l623_62397

def smallest_number : ℕ := 1015

def divisors : List ℕ := [12, 16, 18, 21, 28]

theorem diminished_value_is_seven :
  ∃ (k : ℕ), k = 7 ∧
  ∀ d ∈ divisors, (smallest_number - k) % d = 0 ∧
  ∀ m < k, ∃ d ∈ divisors, (smallest_number - m) % d ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_diminished_value_is_seven_l623_62397


namespace NUMINAMATH_CALUDE_product_multiple_in_consecutive_integers_l623_62379

theorem product_multiple_in_consecutive_integers (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
  ∃ (start : ℤ) (x y : ℤ), 
    x ≠ y ∧ 
    start ≤ x ∧ x < start + b ∧
    start ≤ y ∧ y < start + b ∧
    (x * y) % (a * b) = 0 :=
by sorry

end NUMINAMATH_CALUDE_product_multiple_in_consecutive_integers_l623_62379


namespace NUMINAMATH_CALUDE_roots_of_polynomials_l623_62320

theorem roots_of_polynomials (α : ℂ) : 
  α^2 + α - 1 = 0 → α^3 - 2*α + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomials_l623_62320


namespace NUMINAMATH_CALUDE_combined_sale_price_l623_62387

/-- Calculate the sale price given the purchase cost and profit percentage -/
def calculateSalePrice (purchaseCost : ℚ) (profitPercentage : ℚ) : ℚ :=
  purchaseCost * (1 + profitPercentage)

/-- The problem statement -/
theorem combined_sale_price :
  let itemA_cost : ℚ := 650
  let itemB_cost : ℚ := 350
  let itemC_cost : ℚ := 400
  let itemA_profit : ℚ := 0.40
  let itemB_profit : ℚ := 0.25
  let itemC_profit : ℚ := 0.30
  let itemA_sale := calculateSalePrice itemA_cost itemA_profit
  let itemB_sale := calculateSalePrice itemB_cost itemB_profit
  let itemC_sale := calculateSalePrice itemC_cost itemC_profit
  itemA_sale + itemB_sale + itemC_sale = 1867.50 := by
  sorry

end NUMINAMATH_CALUDE_combined_sale_price_l623_62387


namespace NUMINAMATH_CALUDE_paula_paint_usage_l623_62312

/-- Represents the paint capacity and usage scenario --/
structure PaintScenario where
  initial_capacity : ℕ  -- Initial room painting capacity
  lost_cans : ℕ         -- Number of paint cans lost
  remaining_capacity : ℕ -- Remaining room painting capacity

/-- Calculates the number of cans used given a paint scenario --/
def cans_used (scenario : PaintScenario) : ℕ :=
  scenario.remaining_capacity / ((scenario.initial_capacity - scenario.remaining_capacity) / scenario.lost_cans)

/-- Theorem stating that for the given scenario, 17 cans were used --/
theorem paula_paint_usage : 
  let scenario : PaintScenario := { 
    initial_capacity := 42, 
    lost_cans := 4, 
    remaining_capacity := 34 
  }
  cans_used scenario = 17 := by sorry

end NUMINAMATH_CALUDE_paula_paint_usage_l623_62312


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l623_62313

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + 3*b = 1) :
  1/a + 3/b ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + 3*b₀ = 1 ∧ 1/a₀ + 3/b₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l623_62313


namespace NUMINAMATH_CALUDE_no_14_cents_combination_l623_62388

/-- Represents the types of coins available -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- A selection of coins is represented as a list of Coins -/
def CoinSelection := List Coin

/-- Calculates the total value of a coin selection in cents -/
def totalValue (selection : CoinSelection) : ℕ :=
  selection.map coinValue |>.sum

/-- Theorem stating that it's impossible to select 6 coins totaling 14 cents -/
theorem no_14_cents_combination :
  ∀ (selection : CoinSelection),
    selection.length = 6 →
    totalValue selection ≠ 14 :=
by sorry

end NUMINAMATH_CALUDE_no_14_cents_combination_l623_62388


namespace NUMINAMATH_CALUDE_success_rate_is_70_percent_l623_62333

def games_played : ℕ := 15
def games_won : ℕ := 9
def remaining_games : ℕ := 5

def total_games : ℕ := games_played + remaining_games
def total_wins : ℕ := games_won + remaining_games

def success_rate : ℚ := (total_wins : ℚ) / (total_games : ℚ)

theorem success_rate_is_70_percent :
  success_rate = 7/10 :=
sorry

end NUMINAMATH_CALUDE_success_rate_is_70_percent_l623_62333


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l623_62389

theorem complex_modulus_equality : 
  Complex.abs ((7 - 5*Complex.I)*(3 + 4*Complex.I) + (4 - 3*Complex.I)*(2 + 7*Complex.I)) = Real.sqrt 6073 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l623_62389


namespace NUMINAMATH_CALUDE_conic_sections_eccentricity_l623_62325

theorem conic_sections_eccentricity (x : ℝ) : 
  (2 * x^2 - 5 * x + 2 = 0) →
  (x = 2 ∨ x = 1/2) ∧ 
  ((0 < x ∧ x < 1) ∨ x > 1) :=
sorry

end NUMINAMATH_CALUDE_conic_sections_eccentricity_l623_62325


namespace NUMINAMATH_CALUDE_photographer_photos_to_include_l623_62308

/-- Given a photographer with pre-selected photos and choices to provide photos,
    calculate the number of photos to include in an envelope. -/
def photos_to_include (pre_selected : ℕ) (choices : ℕ) : ℕ :=
  choices / pre_selected

/-- Theorem stating that for a photographer with 7 pre-selected photos and 56 choices,
    the number of photos to include is 8. -/
theorem photographer_photos_to_include :
  photos_to_include 7 56 = 8 := by
  sorry

end NUMINAMATH_CALUDE_photographer_photos_to_include_l623_62308

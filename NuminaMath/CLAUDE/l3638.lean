import Mathlib

namespace NUMINAMATH_CALUDE_cubic_roots_inequality_l3638_363802

theorem cubic_roots_inequality (A B C : ℝ) 
  (h : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
       ∀ x : ℝ, x^3 + A*x^2 + B*x + C = 0 ↔ (x = a ∨ x = b ∨ x = c)) :
  A^2 + B^2 + 18*C > 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_inequality_l3638_363802


namespace NUMINAMATH_CALUDE_inequality_proof_l3638_363888

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) + a + b + c ≤ 3 + (1 / 3) * (a * b + b * c + c * a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3638_363888


namespace NUMINAMATH_CALUDE_regression_prediction_at_2_l3638_363836

/-- Represents a linear regression model -/
structure LinearRegression where
  b : ℝ
  c : ℝ := 0.2

/-- Calculates the y value for a given x in a linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.b * x + model.c

/-- Theorem: Given the conditions, the predicted y value when x = 2 is 2.6 -/
theorem regression_prediction_at_2 
  (model : LinearRegression)
  (h₁ : predict model 4 = 5) -- condition: ȳ = 5 when x̄ = 4
  (h₂ : model.c = 0.2) -- condition: intercept is 0.2
  : predict model 2 = 2.6 := by
  sorry

end NUMINAMATH_CALUDE_regression_prediction_at_2_l3638_363836


namespace NUMINAMATH_CALUDE_allowance_multiple_l3638_363813

theorem allowance_multiple (middle_school_allowance senior_year_allowance x : ℝ) :
  middle_school_allowance = 8 + 2 →
  senior_year_allowance = middle_school_allowance * x + 5 →
  (senior_year_allowance - middle_school_allowance) / middle_school_allowance = 1.5 →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_allowance_multiple_l3638_363813


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3638_363808

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (1 / x + 3 / y) ≥ 1 + Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3638_363808


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l3638_363852

theorem min_value_sum_of_reciprocals (n : ℕ) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / (1 + a^n) + 1 / (1 + b^n)) ≥ 1 ∧ 
  (1 / (1 + 1^n) + 1 / (1 + 1^n) = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l3638_363852


namespace NUMINAMATH_CALUDE_wrapping_paper_fraction_l3638_363819

theorem wrapping_paper_fraction (total_used : ℚ) (num_presents : ℕ) (fraction_per_present : ℚ) :
  total_used = 1/2 →
  num_presents = 5 →
  total_used = fraction_per_present * num_presents →
  fraction_per_present = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_fraction_l3638_363819


namespace NUMINAMATH_CALUDE_black_area_after_changes_l3638_363882

/-- Represents the fraction of black area remaining after a single change --/
def remaining_black_fraction : ℚ := 2 / 3

/-- Represents the number of changes --/
def num_changes : ℕ := 3

/-- Theorem stating that after three changes, 8/27 of the original area remains black --/
theorem black_area_after_changes :
  remaining_black_fraction ^ num_changes = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_black_area_after_changes_l3638_363882


namespace NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l3638_363896

theorem cubic_factorization_sum_of_squares (p q r s t u : ℤ) :
  (∀ x, 512 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 5472 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l3638_363896


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_union_A_B_equals_B_l3638_363860

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem for part 1
theorem intersection_A_complement_B :
  A (-2) ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part 2
theorem union_A_B_equals_B (a : ℝ) :
  A a ∪ B = B ↔ a < -4 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_union_A_B_equals_B_l3638_363860


namespace NUMINAMATH_CALUDE_greatest_common_factor_45_75_90_l3638_363859

theorem greatest_common_factor_45_75_90 : Nat.gcd 45 (Nat.gcd 75 90) = 15 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_45_75_90_l3638_363859


namespace NUMINAMATH_CALUDE_prop_3_prop_4_l3638_363881

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularPL : Plane → Line → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (intersectionPP : Plane → Plane → Line)
variable (intersectionPL : Plane → Line → Prop)

-- Proposition ③
theorem prop_3 (α β γ : Plane) (m : Line) :
  perpendicularPP α β →
  perpendicularPP α γ →
  intersectionPP β γ = m →
  perpendicularPL α m :=
sorry

-- Proposition ④
theorem prop_4 (α β : Plane) (m n : Line) :
  perpendicularPL α m →
  perpendicularPL β n →
  perpendicular m n →
  perpendicularPP α β :=
sorry

end NUMINAMATH_CALUDE_prop_3_prop_4_l3638_363881


namespace NUMINAMATH_CALUDE_point_transformation_l3638_363823

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 180° clockwise around (2,3) -/
def rotate180 (p : Point) : Point :=
  { x := 4 - p.x, y := 6 - p.y }

/-- Reflects a point about the line y = x -/
def reflectAboutYEqualsX (p : Point) : Point :=
  { x := p.y, y := p.x }

/-- Translates a point by the vector (4, -2) -/
def translate (p : Point) : Point :=
  { x := p.x + 4, y := p.y - 2 }

/-- The main theorem -/
theorem point_transformation (Q : Point) :
  (translate (reflectAboutYEqualsX (rotate180 Q)) = Point.mk 1 6) →
  (Q.y - Q.x = 13) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l3638_363823


namespace NUMINAMATH_CALUDE_ninth_term_of_sequence_l3638_363851

/-- Given a sequence {a_n} where the sum of the first n terms S_n = n^2, prove that a_9 = 17 -/
theorem ninth_term_of_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h : ∀ n, S n = n^2) : a 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_sequence_l3638_363851


namespace NUMINAMATH_CALUDE_candy_distribution_l3638_363880

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) 
  (h1 : total_candy = 648) 
  (h2 : num_bags = 8) 
  (h3 : candy_per_bag * num_bags = total_candy) :
  candy_per_bag = 81 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3638_363880


namespace NUMINAMATH_CALUDE_jar_capacity_l3638_363887

/-- Proves that the capacity of each jar James needs to buy is 0.5 liters -/
theorem jar_capacity
  (num_hives : ℕ)
  (honey_per_hive : ℝ)
  (num_jars : ℕ)
  (h1 : num_hives = 5)
  (h2 : honey_per_hive = 20)
  (h3 : num_jars = 100)
  : (num_hives * honey_per_hive / 2) / num_jars = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_jar_capacity_l3638_363887


namespace NUMINAMATH_CALUDE_chips_calories_is_310_l3638_363884

/-- Represents the calorie content of various food items and daily calorie limits --/
structure CalorieData where
  cake : ℕ
  coke : ℕ
  breakfast : ℕ
  lunch : ℕ
  daily_limit : ℕ
  remaining : ℕ

/-- Calculates the calorie content of the pack of chips --/
def calculate_chips_calories (data : CalorieData) : ℕ :=
  data.daily_limit - data.remaining - (data.cake + data.coke + data.breakfast + data.lunch)

/-- Theorem stating that the calorie content of the pack of chips is 310 --/
theorem chips_calories_is_310 (data : CalorieData) 
    (h1 : data.cake = 110)
    (h2 : data.coke = 215)
    (h3 : data.breakfast = 560)
    (h4 : data.lunch = 780)
    (h5 : data.daily_limit = 2500)
    (h6 : data.remaining = 525) :
  calculate_chips_calories data = 310 := by
  sorry

end NUMINAMATH_CALUDE_chips_calories_is_310_l3638_363884


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3638_363807

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0) ↔ k ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3638_363807


namespace NUMINAMATH_CALUDE_solution_exists_l3638_363895

theorem solution_exists : ∃ x : ℚ, 
  (10 / (Real.sqrt (x - 5) - 10) + 
   2 / (Real.sqrt (x - 5) - 5) + 
   9 / (Real.sqrt (x - 5) + 5) + 
   18 / (Real.sqrt (x - 5) + 10) = 0) ∧ 
  (x = 1230 / 121) := by
  sorry


end NUMINAMATH_CALUDE_solution_exists_l3638_363895


namespace NUMINAMATH_CALUDE_original_price_calculation_l3638_363873

theorem original_price_calculation (reduced_price : ℝ) (reduction_percent : ℝ) 
  (h1 : reduced_price = 620) 
  (h2 : reduction_percent = 20) : 
  reduced_price / (1 - reduction_percent / 100) = 775 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l3638_363873


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l3638_363864

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : 4/x + 3/y = 1) : 
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 4/x₀ + 3/y₀ = 1 ∧ x₀ * y₀ = 48 ∧ ∀ x' y', x' > 0 → y' > 0 → 4/x' + 3/y' = 1 → x' * y' ≥ 48) ∧
  (∃ (x₁ y₁ : ℝ), x₁ > 0 ∧ y₁ > 0 ∧ 4/x₁ + 3/y₁ = 1 ∧ x₁ + y₁ = 7 + 4 * Real.sqrt 3 ∧ ∀ x' y', x' > 0 → y' > 0 → 4/x' + 3/y' = 1 → x' + y' ≥ 7 + 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l3638_363864


namespace NUMINAMATH_CALUDE_exam_score_below_mean_l3638_363842

/-- Given an exam with a mean score and a known score above the mean,
    calculate the score that is a certain number of standard deviations below the mean. -/
theorem exam_score_below_mean
  (mean : ℝ)
  (score_above : ℝ)
  (sd_above : ℝ)
  (sd_below : ℝ)
  (h1 : mean = 74)
  (h2 : score_above = 98)
  (h3 : sd_above = 3)
  (h4 : sd_below = 2)
  (h5 : score_above = mean + sd_above * ((score_above - mean) / sd_above)) :
  mean - sd_below * ((score_above - mean) / sd_above) = 58 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_below_mean_l3638_363842


namespace NUMINAMATH_CALUDE_percentage_relation_l3638_363855

theorem percentage_relation (a b : ℝ) (h1 : a - b = 1650) (h2 : a = 2475) (h3 : b = 825) :
  (7.5 / 100) * a = (22.5 / 100) * b := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3638_363855


namespace NUMINAMATH_CALUDE_product_digit_sum_is_nine_l3638_363818

/-- Represents a strictly increasing sequence of 5 digits -/
def StrictlyIncreasingDigits (a b c d e : Nat) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 9

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved -/
theorem product_digit_sum_is_nine 
  (a b c d e : Nat) 
  (h : StrictlyIncreasingDigits a b c d e) : 
  sumOfDigits (9 * (a * 10000 + b * 1000 + c * 100 + d * 10 + e)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_is_nine_l3638_363818


namespace NUMINAMATH_CALUDE_rick_irons_31_clothes_l3638_363856

/-- Calculates the total number of clothes ironed by Rick -/
def totalClothesIroned (shirtsPerHour dressShirtsHours pantsPerHour dressPantsHours jacketsPerHour jacketsHours : ℕ) : ℕ :=
  shirtsPerHour * dressShirtsHours + pantsPerHour * dressPantsHours + jacketsPerHour * jacketsHours

/-- Proves that Rick irons 31 pieces of clothing given the conditions -/
theorem rick_irons_31_clothes :
  totalClothesIroned 4 3 3 5 2 2 = 31 := by
  sorry

#eval totalClothesIroned 4 3 3 5 2 2

end NUMINAMATH_CALUDE_rick_irons_31_clothes_l3638_363856


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3638_363849

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (m, 4) (3, -2) → m = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3638_363849


namespace NUMINAMATH_CALUDE_calculation_proof_l3638_363894

theorem calculation_proof :
  (let a := 3 + 4/5
   let b := (1 - 9/10) / (1/100)
   a * b = 38) ∧
  (let c := 5/6 + 20
   let d := 5/4
   c / d = 50/3) ∧
  (3/7 * 5/9 * 28 * 45 = 300) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3638_363894


namespace NUMINAMATH_CALUDE_child_admission_price_l3638_363838

-- Define the given conditions
def total_people : ℕ := 610
def adult_price : ℚ := 2
def total_receipts : ℚ := 960
def num_adults : ℕ := 350

-- Define the admission price for children
def child_price : ℚ := 1

-- Theorem to prove
theorem child_admission_price :
  child_price * (total_people - num_adults) + adult_price * num_adults = total_receipts :=
sorry

end NUMINAMATH_CALUDE_child_admission_price_l3638_363838


namespace NUMINAMATH_CALUDE_bridge_length_l3638_363826

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) : 
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time_s - train_length = 275 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l3638_363826


namespace NUMINAMATH_CALUDE_expression_evaluation_l3638_363840

theorem expression_evaluation :
  3000 * (3000 ^ 2500) * 2 = 2 * 3000 ^ 2501 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3638_363840


namespace NUMINAMATH_CALUDE_courier_speed_impossibility_l3638_363803

/-- Proves the impossibility of achieving a specific average speed given certain conditions -/
theorem courier_speed_impossibility (total_distance : ℝ) (initial_speed : ℝ) (target_avg_speed : ℝ) :
  total_distance = 24 →
  initial_speed = 8 →
  target_avg_speed = 12 →
  ¬∃ (remaining_speed : ℝ),
    remaining_speed > 0 ∧
    (2/3 * total_distance / initial_speed + 1/3 * total_distance / remaining_speed) = (total_distance / target_avg_speed) :=
by sorry

end NUMINAMATH_CALUDE_courier_speed_impossibility_l3638_363803


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l3638_363821

def total_knights : ℕ := 30
def chosen_knights : ℕ := 3

def probability_adjacent_knights : ℚ :=
  1 - (27 * 25 * 23) / (3 * total_knights.choose chosen_knights)

theorem adjacent_knights_probability :
  probability_adjacent_knights = 34 / 35 := by sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l3638_363821


namespace NUMINAMATH_CALUDE_prob_3_heads_12_coins_value_l3638_363869

/-- The probability of getting exactly 3 heads when flipping 12 coins -/
def prob_3_heads_12_coins : ℚ :=
  (Nat.choose 12 3 : ℚ) / 2^12

/-- Theorem stating that the probability of getting exactly 3 heads
    when flipping 12 coins is equal to 220/4096 -/
theorem prob_3_heads_12_coins_value :
  prob_3_heads_12_coins = 220 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_prob_3_heads_12_coins_value_l3638_363869


namespace NUMINAMATH_CALUDE_dana_marcus_difference_l3638_363811

/-- The number of pencils Jayden has -/
def jayden_pencils : ℕ := 20

/-- The number of pencils Dana has -/
def dana_pencils : ℕ := jayden_pencils + 15

/-- The number of pencils Marcus has -/
def marcus_pencils : ℕ := jayden_pencils / 2

/-- Theorem stating that Dana has 25 more pencils than Marcus -/
theorem dana_marcus_difference : dana_pencils - marcus_pencils = 25 := by
  sorry

end NUMINAMATH_CALUDE_dana_marcus_difference_l3638_363811


namespace NUMINAMATH_CALUDE_outside_trash_count_l3638_363806

def total_trash : ℕ := 1576
def classroom_trash : ℕ := 344

theorem outside_trash_count : total_trash - classroom_trash = 1232 := by
  sorry

end NUMINAMATH_CALUDE_outside_trash_count_l3638_363806


namespace NUMINAMATH_CALUDE_reflect_x_of_P_l3638_363801

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The x-axis reflection of a point -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The given point P -/
def P : Point :=
  { x := -1, y := 2 }

/-- Theorem: The x-axis reflection of P(-1, 2) is (-1, -2) -/
theorem reflect_x_of_P : reflect_x P = { x := -1, y := -2 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_x_of_P_l3638_363801


namespace NUMINAMATH_CALUDE_odd_function_value_l3638_363862

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 ∧ x < 2 then a * Real.log x - a * x + 1 else 0

-- State the theorem
theorem odd_function_value (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is an odd function
  (∀ x ∈ Set.Ioo 0 2, f a x = a * Real.log x - a * x + 1) →  -- definition for x ∈ (0, 2)
  (∃ c ∈ Set.Ioo (-2) 0, ∀ x ∈ Set.Ioo (-2) 0, f a x ≥ f a c) →  -- minimum value exists in (-2, 0)
  (∃ c ∈ Set.Ioo (-2) 0, f a c = 1) →  -- minimum value is 1
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_value_l3638_363862


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l3638_363831

theorem four_digit_number_problem : ∃! (a b c d : ℕ),
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (0 ≤ d ∧ d ≤ 9) ∧
  (a + b = c + d) ∧
  (a + d = c) ∧
  (b + d = 2 * (a + c)) ∧
  (1000 * a + 100 * b + 10 * c + d = 1854) :=
by sorry

#check four_digit_number_problem

end NUMINAMATH_CALUDE_four_digit_number_problem_l3638_363831


namespace NUMINAMATH_CALUDE_probability_at_least_two_A_plus_specific_l3638_363853

def probability_at_least_two_A_plus (p_physics : ℚ) (p_chemistry : ℚ) (p_politics : ℚ) : ℚ :=
  let p_not_physics := 1 - p_physics
  let p_not_chemistry := 1 - p_chemistry
  let p_not_politics := 1 - p_politics
  p_physics * p_chemistry * p_not_politics +
  p_physics * p_not_chemistry * p_politics +
  p_not_physics * p_chemistry * p_politics +
  p_physics * p_chemistry * p_politics

theorem probability_at_least_two_A_plus_specific :
  probability_at_least_two_A_plus (7/8) (3/4) (5/12) = 151/192 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_two_A_plus_specific_l3638_363853


namespace NUMINAMATH_CALUDE_two_digit_number_ratio_l3638_363858

theorem two_digit_number_ratio (a b : ℕ) (h1 : 10 * a + b - (10 * b + a) = 36) (h2 : (a + b) - (a - b) = 8) : 
  a = 2 * b := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_ratio_l3638_363858


namespace NUMINAMATH_CALUDE_expression_factorization_l3638_363834

theorem expression_factorization (y : ℝ) : 
  (12 * y^6 + 35 * y^4 - 5) - (2 * y^6 - 4 * y^4 + 5) = 10 * (y^6 + 3.9 * y^4 - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3638_363834


namespace NUMINAMATH_CALUDE_incorrect_proposition_l3638_363870

theorem incorrect_proposition :
  ¬(∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_proposition_l3638_363870


namespace NUMINAMATH_CALUDE_p2023_coordinates_l3638_363867

/-- Transformation function that maps a point (x, y) to (-y+1, x+2) -/
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2 + 1, p.1 + 2)

/-- Function to apply the transformation n times -/
def iterate_transform (p : ℝ × ℝ) (n : ℕ) : ℝ × ℝ :=
  match n with
  | 0 => p
  | n + 1 => transform (iterate_transform p n)

/-- The starting point P1 -/
def P1 : ℝ × ℝ := (2, 0)

theorem p2023_coordinates :
  iterate_transform P1 2023 = (-3, 3) := by
  sorry

end NUMINAMATH_CALUDE_p2023_coordinates_l3638_363867


namespace NUMINAMATH_CALUDE_complex_number_real_imag_equal_l3638_363828

theorem complex_number_real_imag_equal (b : ℝ) : 
  let z : ℂ := (1 - b * Complex.I) / (2 + Complex.I)
  z.re = z.im ↔ b = -3 := by sorry

end NUMINAMATH_CALUDE_complex_number_real_imag_equal_l3638_363828


namespace NUMINAMATH_CALUDE_total_sand_weight_is_34_l3638_363804

/-- The number of buckets of sand carried by Eden -/
def eden_buckets : ℕ := 4

/-- The number of additional buckets Mary carried compared to Eden -/
def mary_extra_buckets : ℕ := 3

/-- The number of fewer buckets Iris carried compared to Mary -/
def iris_fewer_buckets : ℕ := 1

/-- The weight of sand in each bucket (in pounds) -/
def sand_per_bucket : ℕ := 2

/-- Calculates the total weight of sand collected by Eden, Mary, and Iris -/
def total_sand_weight : ℕ := 
  (eden_buckets + (eden_buckets + mary_extra_buckets) + 
   (eden_buckets + mary_extra_buckets - iris_fewer_buckets)) * sand_per_bucket

/-- Theorem stating that the total weight of sand collected is 34 pounds -/
theorem total_sand_weight_is_34 : total_sand_weight = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_sand_weight_is_34_l3638_363804


namespace NUMINAMATH_CALUDE_range_of_a_l3638_363898

-- Define the complex number z
def z (a : ℝ) : ℂ := (2 + Complex.I) * (a + 2 * Complex.I^3)

-- Define the condition for z to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (in_fourth_quadrant (z a)) ↔ (-1 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3638_363898


namespace NUMINAMATH_CALUDE_pythagorean_consecutive_naturals_l3638_363832

theorem pythagorean_consecutive_naturals :
  ∀ x y z : ℕ, y = x + 1 → z = x + 2 →
  (z^2 = y^2 + x^2 ↔ x = 3 ∧ y = 4 ∧ z = 5) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_consecutive_naturals_l3638_363832


namespace NUMINAMATH_CALUDE_power_of_two_equals_quadratic_plus_one_l3638_363872

theorem power_of_two_equals_quadratic_plus_one (x y : ℕ) :
  2^x = y^2 + y + 1 → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equals_quadratic_plus_one_l3638_363872


namespace NUMINAMATH_CALUDE_number_problem_l3638_363816

theorem number_problem (x : ℝ) (h : 6 * x = 72) : x + 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3638_363816


namespace NUMINAMATH_CALUDE_last_digit_of_n_is_five_l3638_363871

def sum_powers (n : ℕ) : ℕ := (Finset.range (2*n - 2)).sum (λ i => n^(i + 1))

theorem last_digit_of_n_is_five (n : ℕ) (h1 : n ≥ 3) (h2 : Nat.Prime (sum_powers n - 4)) :
  n % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_n_is_five_l3638_363871


namespace NUMINAMATH_CALUDE_days_to_eat_candy_correct_l3638_363885

/-- Given the initial number of candies, the number of candies eaten per day for the first week,
    and the number of candies to be eaten per day after the first week,
    calculate the number of additional days Yuna can eat candy. -/
def days_to_eat_candy (initial_candies : ℕ) (candies_per_day_week1 : ℕ) (candies_per_day_after : ℕ) : ℕ :=
  let candies_eaten_week1 := candies_per_day_week1 * 7
  let remaining_candies := initial_candies - candies_eaten_week1
  remaining_candies / candies_per_day_after

theorem days_to_eat_candy_correct (initial_candies : ℕ) (candies_per_day_week1 : ℕ) (candies_per_day_after : ℕ) 
  (h1 : initial_candies = 60)
  (h2 : candies_per_day_week1 = 6)
  (h3 : candies_per_day_after = 3) :
  days_to_eat_candy initial_candies candies_per_day_week1 candies_per_day_after = 6 := by
  sorry

end NUMINAMATH_CALUDE_days_to_eat_candy_correct_l3638_363885


namespace NUMINAMATH_CALUDE_inscribed_right_isosceles_hypotenuse_l3638_363827

/-- Represents a triangle with a given base and height -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Represents a right isosceles triangle inscribed in another triangle -/
structure InscribedRightIsoscelesTriangle where
  outer : Triangle
  hypotenuse : ℝ

/-- The hypotenuse of an inscribed right isosceles triangle in a 30x10 triangle is 12 -/
theorem inscribed_right_isosceles_hypotenuse 
  (t : Triangle) 
  (i : InscribedRightIsoscelesTriangle) 
  (h1 : t.base = 30) 
  (h2 : t.height = 10) 
  (h3 : i.outer = t) : 
  i.hypotenuse = 12 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_right_isosceles_hypotenuse_l3638_363827


namespace NUMINAMATH_CALUDE_product_eleven_sum_reciprocal_squares_l3638_363861

theorem product_eleven_sum_reciprocal_squares :
  ∀ a b : ℕ,
  a * b = 11 →
  (1 : ℚ) / (a * a : ℚ) + (1 : ℚ) / (b * b : ℚ) = 122 / 121 :=
by
  sorry

end NUMINAMATH_CALUDE_product_eleven_sum_reciprocal_squares_l3638_363861


namespace NUMINAMATH_CALUDE_total_pears_l3638_363866

/-- Given 4 boxes of pears with 16 pears in each box, the total number of pears is 64. -/
theorem total_pears (num_boxes : ℕ) (pears_per_box : ℕ) 
  (h1 : num_boxes = 4) 
  (h2 : pears_per_box = 16) : 
  num_boxes * pears_per_box = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_l3638_363866


namespace NUMINAMATH_CALUDE_range_of_m_for_odd_function_with_conditions_l3638_363865

/-- An odd function f: ℝ → ℝ satisfying certain conditions -/
def OddFunctionWithConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (3/2 + x) = f (3/2 - x)) ∧
  (f 5 > -2) ∧
  (∃ m : ℝ, f 2 = m - 3/m)

/-- The range of m for the given function f -/
def RangeOfM (f : ℝ → ℝ) : Set ℝ :=
  {m : ℝ | m < -1 ∨ (0 < m ∧ m < 3)}

/-- Theorem stating the range of m for a function satisfying the given conditions -/
theorem range_of_m_for_odd_function_with_conditions (f : ℝ → ℝ) 
  (h : OddFunctionWithConditions f) : 
  ∃ m : ℝ, f 2 = m - 3/m ∧ m ∈ RangeOfM f := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_odd_function_with_conditions_l3638_363865


namespace NUMINAMATH_CALUDE_probability_same_color_l3638_363812

def green_balls : ℕ := 8
def white_balls : ℕ := 6
def red_balls : ℕ := 5
def blue_balls : ℕ := 4

def total_balls : ℕ := green_balls + white_balls + red_balls + blue_balls

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def total_combinations : ℕ := choose total_balls 3

def same_color_combinations : ℕ := 
  choose green_balls 3 + choose white_balls 3 + choose red_balls 3 + choose blue_balls 3

theorem probability_same_color :
  (same_color_combinations : ℚ) / total_combinations = 90 / 1771 := by sorry

end NUMINAMATH_CALUDE_probability_same_color_l3638_363812


namespace NUMINAMATH_CALUDE_root_difference_quadratic_specific_quadratic_root_difference_l3638_363820

theorem root_difference_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  2*a*root1^2 + b*root1 = c ∧
  2*a*root2^2 + b*root2 = c ∧
  root1 ≥ root2 →
  root1 - root2 = Real.sqrt discriminant / a :=
by sorry

theorem specific_quadratic_root_difference :
  let a : ℝ := 2
  let b : ℝ := 5
  let c : ℝ := 12
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  root1 - root2 = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_specific_quadratic_root_difference_l3638_363820


namespace NUMINAMATH_CALUDE_min_sphere_surface_area_l3638_363879

theorem min_sphere_surface_area (a b c : ℝ) (h1 : a * b * c = 4) (h2 : a * b = 1) :
  let r := (3 * Real.sqrt 2) / 2
  4 * Real.pi * r^2 = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_min_sphere_surface_area_l3638_363879


namespace NUMINAMATH_CALUDE_revenue_calculation_impossible_l3638_363815

structure ShoeInventory where
  large_boots : ℕ
  medium_sandals : ℕ
  small_sneakers : ℕ
  large_sandals : ℕ
  medium_boots : ℕ
  small_boots : ℕ

def initial_stock : ShoeInventory :=
  { large_boots := 22
  , medium_sandals := 32
  , small_sneakers := 24
  , large_sandals := 45
  , medium_boots := 35
  , small_boots := 26 }

def prices : ShoeInventory :=
  { large_boots := 80
  , medium_sandals := 60
  , small_sneakers := 50
  , large_sandals := 65
  , medium_boots := 75
  , small_boots := 55 }

def total_pairs (stock : ShoeInventory) : ℕ :=
  stock.large_boots + stock.medium_sandals + stock.small_sneakers +
  stock.large_sandals + stock.medium_boots + stock.small_boots

def pairs_left : ℕ := 78

theorem revenue_calculation_impossible :
  ∀ (final_stock : ShoeInventory),
    total_pairs final_stock = pairs_left →
    ∃ (revenue₁ revenue₂ : ℕ),
      revenue₁ ≠ revenue₂ ∧
      (∃ (sold : ShoeInventory),
        total_pairs sold + total_pairs final_stock = total_pairs initial_stock ∧
        revenue₁ = sold.large_boots * prices.large_boots +
                   sold.medium_sandals * prices.medium_sandals +
                   sold.small_sneakers * prices.small_sneakers +
                   sold.large_sandals * prices.large_sandals +
                   sold.medium_boots * prices.medium_boots +
                   sold.small_boots * prices.small_boots) ∧
      (∃ (sold : ShoeInventory),
        total_pairs sold + total_pairs final_stock = total_pairs initial_stock ∧
        revenue₂ = sold.large_boots * prices.large_boots +
                   sold.medium_sandals * prices.medium_sandals +
                   sold.small_sneakers * prices.small_sneakers +
                   sold.large_sandals * prices.large_sandals +
                   sold.medium_boots * prices.medium_boots +
                   sold.small_boots * prices.small_boots) :=
by sorry

end NUMINAMATH_CALUDE_revenue_calculation_impossible_l3638_363815


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3638_363833

theorem fraction_decomposition (n : ℕ) 
  (h1 : ∀ n, 1 / (n * (n + 1)) = 1 / n - 1 / (n + 1))
  (h2 : ∀ n, 1 / (n * (n + 1) * (n + 2)) = 1 / (2 * n * (n + 1)) - 1 / (2 * (n + 1) * (n + 2))) :
  1 / (n * (n + 1) * (n + 2) * (n + 3)) = 
    1 / (3 * n * (n + 1) * (n + 2)) - 1 / (3 * (n + 1) * (n + 2) * (n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3638_363833


namespace NUMINAMATH_CALUDE_painting_time_theorem_l3638_363839

def grace_rate : ℚ := 1 / 6
def henry_rate : ℚ := 1 / 8
def julia_rate : ℚ := 1 / 12
def grace_break : ℚ := 1
def henry_break : ℚ := 1
def julia_break : ℚ := 2

theorem painting_time_theorem :
  ∃ t : ℚ, t > 0 ∧ (grace_rate + henry_rate + julia_rate) * (t - 2) = 1 ∧ t = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_painting_time_theorem_l3638_363839


namespace NUMINAMATH_CALUDE_jeanne_ticket_purchase_l3638_363837

/-- The number of additional tickets Jeanne needs to buy -/
def additional_tickets (ferris_wheel_cost roller_coaster_cost bumper_cars_cost current_tickets : ℕ) : ℕ :=
  ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost - current_tickets

theorem jeanne_ticket_purchase : additional_tickets 5 4 4 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jeanne_ticket_purchase_l3638_363837


namespace NUMINAMATH_CALUDE_calculation_result_l3638_363814

theorem calculation_result : 
  let initial := 180
  let percentage := 35 / 100
  let first_calc := initial * percentage
  let one_third_less := first_calc - (1 / 3 * first_calc)
  let remaining := initial - one_third_less
  let three_fifths := 3 / 5 * remaining
  (three_fifths ^ 2) = 6857.84 := by sorry

end NUMINAMATH_CALUDE_calculation_result_l3638_363814


namespace NUMINAMATH_CALUDE_cos_420_degrees_l3638_363897

theorem cos_420_degrees : Real.cos (420 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_420_degrees_l3638_363897


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3638_363886

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3638_363886


namespace NUMINAMATH_CALUDE_cutting_tool_geometry_l3638_363850

theorem cutting_tool_geometry (A B C : ℝ × ℝ) : 
  let r : ℝ := 6
  let AB : ℝ := 5
  let BC : ℝ := 3
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB^2 →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = BC^2 →
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 →
  A.1^2 + A.2^2 = r^2 →
  B.1^2 + B.2^2 = r^2 →
  C.1^2 + C.2^2 = r^2 →
  (B.1^2 + B.2^2 = 4.16 ∧ A = (2, 5.4) ∧ C = (5, 0.4)) := by
sorry

end NUMINAMATH_CALUDE_cutting_tool_geometry_l3638_363850


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3638_363857

theorem fraction_to_decimal : (5 : ℚ) / 50 = 0.1 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3638_363857


namespace NUMINAMATH_CALUDE_asset_value_increase_l3638_363875

theorem asset_value_increase (initial_value : ℝ) (h : initial_value > 0) :
  let year1_increase := 0.2
  let year2_increase := 0.3
  let year1_value := initial_value * (1 + year1_increase)
  let year2_value := year1_value * (1 + year2_increase)
  let total_increase := (year2_value - initial_value) / initial_value
  total_increase = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_asset_value_increase_l3638_363875


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3638_363848

theorem complex_fraction_equality : (2 * Complex.I) / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3638_363848


namespace NUMINAMATH_CALUDE_final_painting_width_l3638_363876

theorem final_painting_width (total_paintings : Nat) (total_area : Nat) 
  (small_paintings : Nat) (small_painting_side : Nat)
  (large_painting_width large_painting_height : Nat)
  (final_painting_height : Nat) :
  total_paintings = 5 →
  total_area = 200 →
  small_paintings = 3 →
  small_painting_side = 5 →
  large_painting_width = 10 →
  large_painting_height = 8 →
  final_painting_height = 5 →
  ∃ (final_painting_width : Nat),
    final_painting_width = 9 ∧
    total_area = 
      small_paintings * small_painting_side * small_painting_side +
      large_painting_width * large_painting_height +
      final_painting_height * final_painting_width :=
by sorry

end NUMINAMATH_CALUDE_final_painting_width_l3638_363876


namespace NUMINAMATH_CALUDE_CH₄_has_most_atoms_l3638_363829

-- Define the molecules and their atom counts
def O₂_atoms : ℕ := 2
def NH₃_atoms : ℕ := 4
def CO_atoms : ℕ := 2
def CH₄_atoms : ℕ := 5

-- Define a function to compare atom counts
def has_more_atoms (a b : ℕ) : Prop := a > b

-- Theorem statement
theorem CH₄_has_most_atoms :
  has_more_atoms CH₄_atoms O₂_atoms ∧
  has_more_atoms CH₄_atoms NH₃_atoms ∧
  has_more_atoms CH₄_atoms CO_atoms :=
by sorry

end NUMINAMATH_CALUDE_CH₄_has_most_atoms_l3638_363829


namespace NUMINAMATH_CALUDE_total_fence_cost_l3638_363874

/-- Represents the cost of building a fence for a pentagonal plot -/
def fence_cost (a b c d e : ℕ) (pa pb pc pd pe : ℕ) : ℕ :=
  a * pa + b * pb + c * pc + d * pd + e * pe

/-- Theorem stating the total cost of the fence -/
theorem total_fence_cost :
  fence_cost 9 12 15 11 13 45 55 60 50 65 = 3360 := by
  sorry

end NUMINAMATH_CALUDE_total_fence_cost_l3638_363874


namespace NUMINAMATH_CALUDE_triangle_theorem_l3638_363817

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the triangle ABC -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b^2 + t.c^2 = t.a^2 + t.b * t.c)
  (h2 : Real.sin t.B = Real.sqrt 3 / 3)
  (h3 : t.b = 2) :
  t.A = π / 3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 2 + Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3638_363817


namespace NUMINAMATH_CALUDE_tangent_perpendicular_range_l3638_363863

theorem tangent_perpendicular_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 2*x - a + 1/x = 0) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_range_l3638_363863


namespace NUMINAMATH_CALUDE_total_pens_count_l3638_363891

theorem total_pens_count (red_pens : ℕ) (black_pens : ℕ) (blue_pens : ℕ) 
  (h1 : red_pens = 8)
  (h2 : black_pens = red_pens + 10)
  (h3 : blue_pens = red_pens + 7) :
  red_pens + black_pens + blue_pens = 41 := by
  sorry

end NUMINAMATH_CALUDE_total_pens_count_l3638_363891


namespace NUMINAMATH_CALUDE_lily_bought_ten_geese_l3638_363824

/-- The number of geese Lily bought -/
def lily_geese : ℕ := sorry

/-- The number of ducks Lily bought -/
def lily_ducks : ℕ := 20

/-- The number of ducks Rayden bought -/
def rayden_ducks : ℕ := 3 * lily_ducks

/-- The number of geese Rayden bought -/
def rayden_geese : ℕ := 4 * lily_geese

theorem lily_bought_ten_geese :
  lily_geese = 10 ∧
  rayden_ducks + rayden_geese = lily_ducks + lily_geese + 70 :=
by sorry

end NUMINAMATH_CALUDE_lily_bought_ten_geese_l3638_363824


namespace NUMINAMATH_CALUDE_fixed_point_of_f_l3638_363835

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 + 3

-- State the theorem
theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 0 = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_f_l3638_363835


namespace NUMINAMATH_CALUDE_set_operations_and_equality_l3638_363830

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

theorem set_operations_and_equality :
  (∃ m : ℝ, 
    (A ∩ B m = {x | 3 ≤ x ∧ x ≤ 5} ∧
     (Set.univ \ A) ∪ B m = {x | x < 2 ∨ x ≥ 3})) ∧
  (∀ m : ℝ, A = B m ↔ 2 ≤ m ∧ m ≤ 3) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_equality_l3638_363830


namespace NUMINAMATH_CALUDE_tv_show_length_specific_l3638_363883

/-- The length of a TV show, given the total airtime and duration of commercials and breaks -/
def tv_show_length (total_airtime : ℕ) (commercial_durations : List ℕ) (break_durations : List ℕ) : ℚ :=
  let total_minutes : ℕ := total_airtime
  let commercial_time : ℕ := commercial_durations.sum
  let break_time : ℕ := break_durations.sum
  let show_time : ℕ := total_minutes - commercial_time - break_time
  (show_time : ℚ) / 60

/-- Theorem stating the length of the TV show given specific conditions -/
theorem tv_show_length_specific : 
  let total_airtime : ℕ := 150  -- 2 hours and 30 minutes
  let commercial_durations : List ℕ := [7, 7, 13, 5, 9, 9]
  let break_durations : List ℕ := [4, 2, 8]
  abs (tv_show_length total_airtime commercial_durations break_durations - 1.4333) < 0.0001 := by
  sorry

#eval tv_show_length 150 [7, 7, 13, 5, 9, 9] [4, 2, 8]

end NUMINAMATH_CALUDE_tv_show_length_specific_l3638_363883


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3638_363805

/-- The polynomial f(x) = x^4 - 4x^2 + 7 -/
def f (x : ℝ) : ℝ := x^4 - 4*x^2 + 7

/-- The remainder when f(x) is divided by (x - 1) -/
def remainder : ℝ := f 1

theorem polynomial_remainder : remainder = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3638_363805


namespace NUMINAMATH_CALUDE_import_tax_problem_l3638_363868

/-- Calculates the import tax percentage given the total value, tax-free portion, and tax amount. -/
def import_tax_percentage (total_value tax_free_portion tax_amount : ℚ) : ℚ :=
  (tax_amount / (total_value - tax_free_portion)) * 100

/-- Proves that the import tax percentage is 7% given the specific values in the problem. -/
theorem import_tax_problem :
  let total_value : ℚ := 2560
  let tax_free_portion : ℚ := 1000
  let tax_amount : ℚ := 109.2
  sorry


end NUMINAMATH_CALUDE_import_tax_problem_l3638_363868


namespace NUMINAMATH_CALUDE_divisibility_problem_l3638_363846

theorem divisibility_problem :
  (∃ (a b : Nat), a < 10 ∧ b < 10 ∧
    (∀ n : Nat, 73 ∣ (10 * a + b) * 10^n + (200 * 10^n + 79) / 9)) ∧
  (¬ ∃ (c d : Nat), c < 10 ∧ d < 10 ∧
    (∀ n : Nat, 79 ∣ (10 * c + d) * 10^n + (200 * 10^n + 79) / 9)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3638_363846


namespace NUMINAMATH_CALUDE_linear_combination_passes_through_intersection_l3638_363825

/-- Two distinct linear equations in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- The point where two linear equations intersect -/
def intersection (eq1 eq2 : LinearEquation) : ℝ × ℝ :=
  sorry

/-- Checks if a point satisfies a linear equation -/
def satisfies (eq : LinearEquation) (point : ℝ × ℝ) : Prop :=
  eq.a * point.1 + eq.b * point.2 + eq.c = 0

/-- Theorem: For any two distinct linear equations and any real k,
    the equation P(x,y) + k P^1(x,y) = 0 passes through their intersection point -/
theorem linear_combination_passes_through_intersection
  (P P1 : LinearEquation) (k : ℝ) (h : P ≠ P1) :
  let intersect_point := intersection P P1
  satisfies ⟨P.a + k * P1.a, P.b + k * P1.b, P.c + k * P1.c, sorry⟩ intersect_point :=
by
  sorry

end NUMINAMATH_CALUDE_linear_combination_passes_through_intersection_l3638_363825


namespace NUMINAMATH_CALUDE_sampling_methods_correct_l3638_363847

/-- Represents a sampling method --/
inductive SamplingMethod
  | Systematic
  | SimpleRandom

/-- Represents a scenario for sampling --/
structure SamplingScenario where
  description : String
  interval : Option ℕ
  sampleSize : ℕ
  populationSize : ℕ

/-- Determines the appropriate sampling method for a given scenario --/
def determineSamplingMethod (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- The milk production line scenario --/
def milkProductionScenario : SamplingScenario :=
  { description := "Milk production line inspection"
  , interval := some 30
  , sampleSize := 1
  , populationSize := 0 }

/-- The math enthusiasts scenario --/
def mathEnthusiastsScenario : SamplingScenario :=
  { description := "Math enthusiasts study load"
  , interval := none
  , sampleSize := 3
  , populationSize := 30 }

theorem sampling_methods_correct :
  determineSamplingMethod milkProductionScenario = SamplingMethod.Systematic ∧
  determineSamplingMethod mathEnthusiastsScenario = SamplingMethod.SimpleRandom :=
  sorry

end NUMINAMATH_CALUDE_sampling_methods_correct_l3638_363847


namespace NUMINAMATH_CALUDE_energy_conservation_train_ball_system_energy_changes_specific_scenario_l3638_363844

/-- Represents the velocity of an object -/
structure Velocity where
  value : ℝ
  unit : String

/-- Represents the kinetic energy of an object -/
structure KineticEnergy where
  value : ℝ
  unit : String

/-- Represents a physical system consisting of a train and a ball -/
structure TrainBallSystem where
  trainVelocity : Velocity
  ballMass : ℝ
  ballThrowingVelocity : Velocity

/-- Calculates the kinetic energy of an object given its mass and velocity -/
def calculateKineticEnergy (mass : ℝ) (velocity : Velocity) : KineticEnergy :=
  { value := 0.5 * mass * velocity.value ^ 2, unit := "J" }

/-- Theorem: Energy conservation in the train-ball system -/
theorem energy_conservation_train_ball_system
  (system : TrainBallSystem)
  (initial_train_energy : KineticEnergy)
  (initial_ball_energy : KineticEnergy)
  (final_ball_energy_forward : KineticEnergy)
  (final_ball_energy_backward : KineticEnergy) :
  (initial_train_energy.value + initial_ball_energy.value =
   initial_train_energy.value + final_ball_energy_forward.value) ∧
  (initial_train_energy.value + initial_ball_energy.value =
   initial_train_energy.value + final_ball_energy_backward.value) :=
by sorry

/-- Corollary: Specific energy changes for the given scenario -/
theorem energy_changes_specific_scenario
  (system : TrainBallSystem)
  (h_train_velocity : system.trainVelocity.value = 60 ∧ system.trainVelocity.unit = "km/hour")
  (h_ball_velocity : system.ballThrowingVelocity.value = 60 ∧ system.ballThrowingVelocity.unit = "km/hour")
  (initial_ball_energy : KineticEnergy)
  (h_forward : calculateKineticEnergy system.ballMass
    { value := system.trainVelocity.value + system.ballThrowingVelocity.value, unit := "km/hour" } =
    { value := 4 * initial_ball_energy.value, unit := initial_ball_energy.unit })
  (h_backward : calculateKineticEnergy system.ballMass
    { value := system.trainVelocity.value - system.ballThrowingVelocity.value, unit := "km/hour" } =
    { value := 0, unit := initial_ball_energy.unit }) :
  ∃ (compensating_energy : KineticEnergy),
    compensating_energy.value = 3 * initial_ball_energy.value ∧
    compensating_energy.value = initial_ball_energy.value :=
by sorry

end NUMINAMATH_CALUDE_energy_conservation_train_ball_system_energy_changes_specific_scenario_l3638_363844


namespace NUMINAMATH_CALUDE_total_apples_l3638_363878

/-- Given 37 baskets with 17 apples each, prove that the total number of apples is 629. -/
theorem total_apples (baskets : ℕ) (apples_per_basket : ℕ) 
  (h1 : baskets = 37) (h2 : apples_per_basket = 17) : 
  baskets * apples_per_basket = 629 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_l3638_363878


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l3638_363841

theorem quadratic_form_equivalence (k : ℝ) :
  (∃ (a b : ℝ), ∀ (x : ℝ), (3*k - 2)*x*(x + k) + k^2*(k - 1) = (a*x + b)^2) ↔ k = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l3638_363841


namespace NUMINAMATH_CALUDE_smallest_iteration_for_three_l3638_363800

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 ∧ x % 7 = 0 then x / 14
  else if x % 7 = 0 then 2 * x
  else if x % 2 = 0 then 7 * x
  else x + 2

def f_iter (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (f_iter n x)

theorem smallest_iteration_for_three :
  (∀ a : ℕ, 1 < a → a < 6 → f_iter a 3 ≠ f 3) ∧
  f_iter 6 3 = f 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_iteration_for_three_l3638_363800


namespace NUMINAMATH_CALUDE_direct_variation_problem_l3638_363822

/-- A function representing direct variation between z and w -/
def directVariation (k : ℝ) (w : ℝ) : ℝ := k * w

theorem direct_variation_problem (k : ℝ) :
  (directVariation k 5 = 10) →
  (directVariation k 15 = 30) :=
by
  sorry

#check direct_variation_problem

end NUMINAMATH_CALUDE_direct_variation_problem_l3638_363822


namespace NUMINAMATH_CALUDE_cosine_ratio_equals_one_l3638_363845

theorem cosine_ratio_equals_one (c : ℝ) (h : c = 2 * Real.pi / 7) :
  (Real.cos (3 * c) * Real.cos (5 * c) * Real.cos (6 * c)) /
  (Real.cos c * Real.cos (2 * c) * Real.cos (3 * c)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_ratio_equals_one_l3638_363845


namespace NUMINAMATH_CALUDE_sample_size_theorem_l3638_363890

/-- Represents a population of students -/
structure Population where
  size : Nat

/-- Represents a sample of students -/
structure Sample where
  size : Nat
  population : Population

/-- Theorem: Given a population of 5000 students and a selection of 250 students,
    the 250 students form a sample of the population with a sample size of 250. -/
theorem sample_size_theorem (pop : Population) (sam : Sample) 
    (h1 : pop.size = 5000) (h2 : sam.size = 250) (h3 : sam.population = pop) : 
    sam.size = 250 ∧ sam.population = pop := by
  sorry

#check sample_size_theorem

end NUMINAMATH_CALUDE_sample_size_theorem_l3638_363890


namespace NUMINAMATH_CALUDE_only_prop2_is_true_l3638_363809

-- Define the propositions
def prop1 : Prop := ∀ x : ℝ, (∃ y : ℝ, y^2 + 1 > 3*y) ↔ ¬(x^2 + 1 < 3*x)

def prop2 : Prop := ∀ p q : Prop, (¬(p ∨ q)) → (¬p ∧ ¬q)

def prop3 : Prop := ∃ a : ℝ, (a > 2 → a > 5) ∧ ¬(a > 5 → a > 2)

def prop4 : Prop := ∀ x y : ℝ, (x ≠ 0 ∨ y ≠ 0) → (x*y ≠ 0)

-- Theorem stating that only prop2 is true
theorem only_prop2_is_true : ¬prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4 := by
  sorry

end NUMINAMATH_CALUDE_only_prop2_is_true_l3638_363809


namespace NUMINAMATH_CALUDE_tunnel_construction_days_l3638_363889

/-- The number of days to complete the tunnel with new equipment -/
def total_days : ℕ := 185

/-- The fraction of the tunnel completed at original speed -/
def original_fraction : ℚ := 1/3

/-- The speed increase factor with new equipment -/
def speed_increase : ℚ := 1.2

/-- The working hours reduction factor with new equipment -/
def hours_reduction : ℚ := 0.8

/-- The effective daily construction rate with new equipment -/
def effective_rate : ℚ := speed_increase * hours_reduction

theorem tunnel_construction_days : 
  ∃ (original_days : ℕ), 
    (original_days : ℚ) * (original_fraction + (1 - original_fraction) / effective_rate) = total_days ∧ 
    original_days = 180 := by
  sorry

end NUMINAMATH_CALUDE_tunnel_construction_days_l3638_363889


namespace NUMINAMATH_CALUDE_fifteen_equation_system_solution_l3638_363899

theorem fifteen_equation_system_solution (x : Fin 15 → ℝ) :
  (∀ i : Fin 14, 1 - x i * x (i + 1) = 0) ∧
  (1 - x 15 * x 1 = 0) →
  (∀ i : Fin 15, x i = 1) ∨ (∀ i : Fin 15, x i = -1) := by
  sorry

end NUMINAMATH_CALUDE_fifteen_equation_system_solution_l3638_363899


namespace NUMINAMATH_CALUDE_negative_three_is_square_mod_p_l3638_363893

theorem negative_three_is_square_mod_p (p q : ℕ) (h_prime : Nat.Prime p) (h_form : p = 3 * q + 1) :
  ∃ x : ZMod p, x^2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_is_square_mod_p_l3638_363893


namespace NUMINAMATH_CALUDE_memorial_visitors_equation_l3638_363843

theorem memorial_visitors_equation (x : ℕ) (h1 : x + (2 * x + 56) = 589) : 2 * x + 56 = 589 - x := by
  sorry

end NUMINAMATH_CALUDE_memorial_visitors_equation_l3638_363843


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l3638_363854

def symmetric_about_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = -x2 ∧ y1 = y2

theorem symmetric_points_sum_power (m n : ℝ) :
  symmetric_about_y_axis m 3 4 n →
  (m + n)^2015 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l3638_363854


namespace NUMINAMATH_CALUDE_book_sale_gain_percentage_l3638_363877

/-- Calculates the gain percentage when selling a book -/
def gain_percentage (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem about the gain percentage of a book sale -/
theorem book_sale_gain_percentage 
  (loss_price : ℚ) 
  (gain_price : ℚ) 
  (loss_percentage : ℚ) :
  loss_price = 450 →
  gain_price = 550 →
  loss_percentage = 10 →
  ∃ (cost_price : ℚ), 
    cost_price * (1 - loss_percentage / 100) = loss_price ∧
    gain_percentage cost_price gain_price = 10 := by
  sorry

#eval gain_percentage 500 550

end NUMINAMATH_CALUDE_book_sale_gain_percentage_l3638_363877


namespace NUMINAMATH_CALUDE_abcd_sum_l3638_363892

theorem abcd_sum (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = -3)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = 0) :
  a * b + c * d = -31 := by
  sorry

end NUMINAMATH_CALUDE_abcd_sum_l3638_363892


namespace NUMINAMATH_CALUDE_quadratic_root_divisibility_l3638_363810

theorem quadratic_root_divisibility (a b c n : ℤ) 
  (h : a * n^2 + b * n + c = 0) : 
  c ∣ n := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_divisibility_l3638_363810

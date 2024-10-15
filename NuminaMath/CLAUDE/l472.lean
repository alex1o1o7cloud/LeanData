import Mathlib

namespace NUMINAMATH_CALUDE_train_length_calculation_l472_47207

/-- The length of a train that crosses a platform of equal length in one minute at 90 km/hr is 750 meters. -/
theorem train_length_calculation (train_length platform_length : ℝ) (speed : ℝ) (time : ℝ) :
  train_length = platform_length →
  speed = 90 →
  time = 1 / 60 →
  train_length = 750 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l472_47207


namespace NUMINAMATH_CALUDE_fourth_number_proof_l472_47215

theorem fourth_number_proof (x : ℝ) (fourth_number : ℝ) : 
  (128 + 255 + 511 + 1023 + x) / 5 = 423 →
  (28 + x + 42 + fourth_number + 104) / 5 = 90 →
  fourth_number = 78 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l472_47215


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l472_47294

theorem triangle_side_and_area 
  (a b c : ℝ) 
  (A : ℝ) 
  (h1 : a = Real.sqrt 7)
  (h2 : c = 3)
  (h3 : A = π / 3) :
  (b = 1 ∨ b = 2) ∧
  ((b = 1 → (1/2 * b * c * Real.sin A = (3 * Real.sqrt 3) / 4)) ∧
   (b = 2 → (1/2 * b * c * Real.sin A = (3 * Real.sqrt 3) / 2))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_and_area_l472_47294


namespace NUMINAMATH_CALUDE_tile_arrangements_l472_47268

/-- The number of distinguishable arrangements of tiles -/
def num_arrangements (brown purple red yellow : ℕ) : ℕ :=
  Nat.factorial (brown + purple + red + yellow) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial red * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 1 brown, 2 purple, 2 red, and 3 yellow tiles is 1680 -/
theorem tile_arrangements :
  num_arrangements 1 2 2 3 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l472_47268


namespace NUMINAMATH_CALUDE_natasha_exercise_time_l472_47222

theorem natasha_exercise_time :
  -- Define variables
  ∀ (natasha_daily_minutes : ℕ) 
    (natasha_days : ℕ) 
    (esteban_daily_minutes : ℕ) 
    (esteban_days : ℕ) 
    (total_minutes : ℕ),
  -- Set conditions
  natasha_days = 7 →
  esteban_daily_minutes = 10 →
  esteban_days = 9 →
  total_minutes = 5 * 60 →
  natasha_daily_minutes * natasha_days + esteban_daily_minutes * esteban_days = total_minutes →
  -- Conclusion
  natasha_daily_minutes = 30 := by
sorry

end NUMINAMATH_CALUDE_natasha_exercise_time_l472_47222


namespace NUMINAMATH_CALUDE_solution_set_inequality_l472_47254

theorem solution_set_inequality (x : ℝ) : 
  (Set.Ioo (-2 : ℝ) 0).Nonempty ∧ 
  (∀ y ∈ Set.Ioo (-2 : ℝ) 0, |1 + y + y^2/2| < 1) ∧
  (∀ z : ℝ, z ∉ Set.Ioo (-2 : ℝ) 0 → |1 + z + z^2/2| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l472_47254


namespace NUMINAMATH_CALUDE_boys_meeting_on_circular_track_l472_47217

/-- The number of times two boys meet on a circular track -/
def number_of_meetings (speed1 speed2 : ℝ) : ℕ :=
  -- We'll define this function later
  sorry

/-- Theorem: Two boys moving in opposite directions on a circular track with speeds
    of 5 ft/s and 9 ft/s will meet 13 times before returning to the starting point -/
theorem boys_meeting_on_circular_track :
  number_of_meetings 5 9 = 13 := by
  sorry

end NUMINAMATH_CALUDE_boys_meeting_on_circular_track_l472_47217


namespace NUMINAMATH_CALUDE_min_value_expression_l472_47280

theorem min_value_expression (x : ℝ) : 
  (12 - x) * (10 - x) * (12 + x) * (10 + x) ≥ -484 ∧ 
  ∃ y : ℝ, (12 - y) * (10 - y) * (12 + y) * (10 + y) = -484 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l472_47280


namespace NUMINAMATH_CALUDE_solve_equation_l472_47223

theorem solve_equation (x : ℚ) (h : (3/2) * x - 3 = 15) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l472_47223


namespace NUMINAMATH_CALUDE_list_number_relation_l472_47238

theorem list_number_relation (n : ℝ) (list : List ℝ) : 
  list.length = 21 ∧ 
  n ∈ list ∧
  n = (1 / 6 : ℝ) * list.sum →
  n = 4 * ((list.sum - n) / 20) := by
sorry

end NUMINAMATH_CALUDE_list_number_relation_l472_47238


namespace NUMINAMATH_CALUDE_max_stores_visited_l472_47232

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) 
  (double_visitors : ℕ) (h1 : total_stores = 12) (h2 : total_visits = 45) 
  (h3 : total_shoppers = 22) (h4 : double_visitors = 14) 
  (h5 : double_visitors ≤ total_shoppers) 
  (h6 : 2 * double_visitors ≤ total_visits) :
  ∃ (max_visits : ℕ), max_visits ≤ total_stores ∧ 
    (∀ (person_visits : ℕ), person_visits ≤ max_visits) ∧ 
    max_visits = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_stores_visited_l472_47232


namespace NUMINAMATH_CALUDE_product_square_of_sum_and_diff_l472_47284

theorem product_square_of_sum_and_diff (x y : ℝ) 
  (sum_eq : x + y = 23) 
  (diff_eq : x - y = 7) : 
  (x * y)^2 = 14400 := by
sorry

end NUMINAMATH_CALUDE_product_square_of_sum_and_diff_l472_47284


namespace NUMINAMATH_CALUDE_cone_sphere_volume_l472_47258

/-- Given a cone with lateral surface forming a semicircle of radius 2√3 when unrolled,
    and with vertex and base circle on the surface of a sphere O,
    prove that the volume of sphere O is 32π/3 -/
theorem cone_sphere_volume (l : ℝ) (r : ℝ) (h : ℝ) (R : ℝ) :
  l = 2 * Real.sqrt 3 →
  r = l / 2 →
  h^2 = l^2 - r^2 →
  2 * R = l^2 / h →
  (4 / 3) * Real.pi * R^3 = (32 / 3) * Real.pi :=
sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_l472_47258


namespace NUMINAMATH_CALUDE_ceiling_minus_y_l472_47256

theorem ceiling_minus_y (x : ℝ) : 
  let y := 2 * x
  let f := y - ⌊y⌋
  (⌈y⌉ - ⌊y⌋ = 1) → (0 < f ∧ f < 1) → (⌈y⌉ - y = 1 - f) :=
by sorry

end NUMINAMATH_CALUDE_ceiling_minus_y_l472_47256


namespace NUMINAMATH_CALUDE_toms_fruit_purchase_l472_47275

/-- The problem of Tom's fruit purchase -/
theorem toms_fruit_purchase 
  (apple_kg : ℕ) 
  (apple_rate : ℕ) 
  (mango_rate : ℕ) 
  (total_paid : ℕ) 
  (h1 : apple_kg = 8)
  (h2 : apple_rate = 70)
  (h3 : mango_rate = 75)
  (h4 : total_paid = 1235)
  : ∃ (mango_kg : ℕ), 
    apple_kg * apple_rate + mango_kg * mango_rate = total_paid ∧ 
    mango_kg = 9 := by
  sorry

end NUMINAMATH_CALUDE_toms_fruit_purchase_l472_47275


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l472_47263

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (x, 1)
  are_parallel a b → x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l472_47263


namespace NUMINAMATH_CALUDE_xyz_value_l472_47234

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l472_47234


namespace NUMINAMATH_CALUDE_job_completion_time_l472_47269

/-- Proves that if A and D together can complete a job in 5 hours, and D alone can complete
    the job in 10 hours, then A alone can complete the job in 10 hours. -/
theorem job_completion_time (A D : ℝ) (hAD : 1 / A + 1 / D = 1 / 5) (hD : D = 10) : A = 10 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l472_47269


namespace NUMINAMATH_CALUDE_evaluate_expression_l472_47257

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 6 * y^x = 789 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l472_47257


namespace NUMINAMATH_CALUDE_can_space_before_compacting_l472_47281

theorem can_space_before_compacting :
  ∀ (n : ℕ) (total_space : ℝ) (compaction_ratio : ℝ),
    n = 60 →
    compaction_ratio = 0.2 →
    total_space = 360 →
    (n : ℝ) * compaction_ratio * (360 / (n * compaction_ratio)) = total_space →
    360 / (n * compaction_ratio) = 30 := by
  sorry

end NUMINAMATH_CALUDE_can_space_before_compacting_l472_47281


namespace NUMINAMATH_CALUDE_sin_theta_value_l472_47233

theorem sin_theta_value (θ : Real) (h1 : 5 * Real.tan θ = 2 * Real.cos θ) (h2 : 0 < θ) (h3 : θ < Real.pi) :
  Real.sin θ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_theta_value_l472_47233


namespace NUMINAMATH_CALUDE_thread_length_ratio_l472_47282

theorem thread_length_ratio : 
  let original_length : ℚ := 12
  let total_required : ℚ := 21
  let additional_length := total_required - original_length
  additional_length / original_length = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_thread_length_ratio_l472_47282


namespace NUMINAMATH_CALUDE_china_gdp_scientific_notation_l472_47264

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem china_gdp_scientific_notation :
  toScientificNotation 86000 = ScientificNotation.mk 8.6 4 sorry := by
  sorry

end NUMINAMATH_CALUDE_china_gdp_scientific_notation_l472_47264


namespace NUMINAMATH_CALUDE_binary_1011_equals_11_l472_47249

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1011_equals_11 :
  binary_to_decimal [true, true, false, true] = 11 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011_equals_11_l472_47249


namespace NUMINAMATH_CALUDE_max_value_of_g_l472_47209

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^4

-- State the theorem
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l472_47209


namespace NUMINAMATH_CALUDE_fraction_simplification_l472_47289

theorem fraction_simplification :
  (4 : ℝ) / (Real.sqrt 108 + 2 * Real.sqrt 12 + 2 * Real.sqrt 27) = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l472_47289


namespace NUMINAMATH_CALUDE_problem_solution_l472_47272

theorem problem_solution : ∃ x : ℝ, 10 * x = 2 * x - 36 ∧ x = -4.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l472_47272


namespace NUMINAMATH_CALUDE_dollar_four_neg_one_l472_47248

-- Define the $ operation
def dollar (x y : ℤ) : ℤ := x * (y + 2) + 2 * x * y

-- Theorem statement
theorem dollar_four_neg_one : dollar 4 (-1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_dollar_four_neg_one_l472_47248


namespace NUMINAMATH_CALUDE_distance_from_origin_l472_47298

theorem distance_from_origin (x y n : ℝ) : 
  x = 8 →
  y > 10 →
  (x - 3)^2 + (y - 10)^2 = 15^2 →
  n^2 = x^2 + y^2 →
  n = Real.sqrt (364 + 200 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l472_47298


namespace NUMINAMATH_CALUDE_stating_hotel_booking_problem_l472_47265

/-- Represents the number of double rooms booked in a hotel. -/
def double_rooms : ℕ := 196

/-- Represents the number of single rooms booked in a hotel. -/
def single_rooms : ℕ := 260 - double_rooms

/-- The cost of a single room in dollars. -/
def single_room_cost : ℕ := 35

/-- The cost of a double room in dollars. -/
def double_room_cost : ℕ := 60

/-- The total revenue from all booked rooms in dollars. -/
def total_revenue : ℕ := 14000

/-- 
Theorem stating that given the conditions of the hotel booking problem,
the number of double rooms booked is 196.
-/
theorem hotel_booking_problem :
  (single_rooms + double_rooms = 260) ∧
  (single_room_cost * single_rooms + double_room_cost * double_rooms = total_revenue) →
  double_rooms = 196 :=
by sorry

end NUMINAMATH_CALUDE_stating_hotel_booking_problem_l472_47265


namespace NUMINAMATH_CALUDE_negation_of_existential_l472_47227

theorem negation_of_existential (p : Prop) :
  (¬∃ (x : ℝ), x^2 + 2*x = 3) ↔ (∀ (x : ℝ), x^2 + 2*x ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_l472_47227


namespace NUMINAMATH_CALUDE_box_fits_40_blocks_l472_47213

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ :=
  d.height * d.width * d.length

/-- Calculates how many smaller objects can fit into a larger object -/
def fitCount (larger smaller : Dimensions) : ℕ :=
  (volume larger) / (volume smaller)

theorem box_fits_40_blocks : 
  let box := Dimensions.mk 8 10 12
  let block := Dimensions.mk 3 2 4
  fitCount box block = 40 := by
  sorry

#eval fitCount (Dimensions.mk 8 10 12) (Dimensions.mk 3 2 4)

end NUMINAMATH_CALUDE_box_fits_40_blocks_l472_47213


namespace NUMINAMATH_CALUDE_calculation_proof_l472_47229

theorem calculation_proof : 2325 + 300 / 75 - 425 * 2 = 1479 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l472_47229


namespace NUMINAMATH_CALUDE_peanuts_added_l472_47201

theorem peanuts_added (initial : ℕ) (final : ℕ) (added : ℕ) : 
  initial = 4 → final = 6 → final = initial + added → added = 2 := by
sorry

end NUMINAMATH_CALUDE_peanuts_added_l472_47201


namespace NUMINAMATH_CALUDE_discounted_subscription_cost_l472_47267

/-- The discounted subscription cost problem -/
theorem discounted_subscription_cost
  (normal_cost : ℝ)
  (discount_percentage : ℝ)
  (h_normal_cost : normal_cost = 80)
  (h_discount : discount_percentage = 45) :
  normal_cost * (1 - discount_percentage / 100) = 44 :=
by sorry

end NUMINAMATH_CALUDE_discounted_subscription_cost_l472_47267


namespace NUMINAMATH_CALUDE_range_of_a_l472_47237

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l472_47237


namespace NUMINAMATH_CALUDE_combined_average_l472_47278

/-- Given two sets of results, one with 80 results averaging 32 and another with 50 results averaging 56,
    prove that the average of all results combined is (80 * 32 + 50 * 56) / (80 + 50) -/
theorem combined_average (set1_count : Nat) (set1_avg : ℚ) (set2_count : Nat) (set2_avg : ℚ)
    (h1 : set1_count = 80)
    (h2 : set1_avg = 32)
    (h3 : set2_count = 50)
    (h4 : set2_avg = 56) :
  (set1_count * set1_avg + set2_count * set2_avg) / (set1_count + set2_count) =
    (80 * 32 + 50 * 56) / (80 + 50) := by
  sorry

end NUMINAMATH_CALUDE_combined_average_l472_47278


namespace NUMINAMATH_CALUDE_g_value_at_9_l472_47253

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 + x + 1

-- Define the properties of g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d) ∧  -- g is a cubic polynomial
  (g 0 = -1) ∧  -- g(0) = -1
  (∀ r : ℝ, f r = 0 → ∃ s : ℝ, g s = 0 ∧ s = r^2)  -- roots of g are squares of roots of f

-- State the theorem
theorem g_value_at_9 (g : ℝ → ℝ) (hg : g_properties g) : g 9 = 899 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_9_l472_47253


namespace NUMINAMATH_CALUDE_buratino_spent_10_dollars_l472_47296

/-- Represents a transaction at the exchange point -/
inductive Transaction
  | type1  -- Give 2 euros, receive 3 dollars and a candy
  | type2  -- Give 5 dollars, receive 3 euros and a candy

/-- Represents Buratino's exchange activities -/
structure ExchangeActivity where
  transactions : List Transaction
  initialDollars : ℕ
  finalDollars : ℕ
  finalEuros : ℕ
  candiesReceived : ℕ

/-- Calculates the net dollar change for a given transaction -/
def netDollarChange (t : Transaction) : ℤ :=
  match t with
  | Transaction.type1 => 3
  | Transaction.type2 => -5

/-- Calculates the net euro change for a given transaction -/
def netEuroChange (t : Transaction) : ℤ :=
  match t with
  | Transaction.type1 => -2
  | Transaction.type2 => 3

/-- Theorem stating that Buratino spent 10 dollars -/
theorem buratino_spent_10_dollars (activity : ExchangeActivity) :
  activity.candiesReceived = 50 ∧
  activity.finalEuros = 0 ∧
  activity.finalDollars < activity.initialDollars →
  activity.initialDollars - activity.finalDollars = 10 := by
  sorry


end NUMINAMATH_CALUDE_buratino_spent_10_dollars_l472_47296


namespace NUMINAMATH_CALUDE_find_number_l472_47231

theorem find_number : ∃ x : ℝ, (x / 18) - 29 = 6 ∧ x = 630 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l472_47231


namespace NUMINAMATH_CALUDE_regression_change_l472_47203

/-- Represents a linear regression equation of the form y = a + bx -/
structure LinearRegression where
  a : ℝ  -- y-intercept
  b : ℝ  -- slope

/-- Calculates the change in y given a change in x for a linear regression -/
def changeInY (reg : LinearRegression) (dx : ℝ) : ℝ :=
  reg.b * dx

theorem regression_change 
  (reg : LinearRegression) 
  (h1 : reg.a = 2)
  (h2 : reg.b = -2.5) : 
  changeInY reg 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_regression_change_l472_47203


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l472_47206

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem largest_three_digit_product :
  ∀ n x y : ℕ,
    n ≥ 100 ∧ n < 1000 →
    is_prime x ∧ is_prime y ∧ is_prime (10 * y + x) →
    x < 10 ∧ y < 10 →
    x ≠ y ∧ x ≠ (10 * y + x) ∧ y ≠ (10 * y + x) →
    n = x * y * (10 * y + x) →
    n ≤ 777 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l472_47206


namespace NUMINAMATH_CALUDE_fraction_calculation_l472_47214

theorem fraction_calculation : (0.5^3) / (0.05^2) = 50 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l472_47214


namespace NUMINAMATH_CALUDE_simplify_fraction_l472_47295

theorem simplify_fraction : 9 * (12 / 7) * (-35 / 36) = -15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l472_47295


namespace NUMINAMATH_CALUDE_max_reach_is_nine_feet_l472_47276

/-- The maximum height Barry and Larry can reach when Barry stands on Larry's shoulders -/
def max_reach (barry_reach : ℝ) (larry_height : ℝ) (larry_shoulder_ratio : ℝ) : ℝ :=
  larry_height * larry_shoulder_ratio + barry_reach

/-- Theorem stating the maximum height Barry and Larry can reach -/
theorem max_reach_is_nine_feet :
  max_reach 5 5 0.8 = 9 := by
  sorry

#eval max_reach 5 5 0.8

end NUMINAMATH_CALUDE_max_reach_is_nine_feet_l472_47276


namespace NUMINAMATH_CALUDE_victors_percentage_l472_47218

/-- Calculate the percentage of marks obtained given the marks scored and maximum marks -/
def calculatePercentage (marksScored : ℕ) (maxMarks : ℕ) : ℚ :=
  (marksScored : ℚ) / (maxMarks : ℚ) * 100

/-- Theorem stating that Victor's percentage of marks is 95% -/
theorem victors_percentage :
  let marksScored : ℕ := 285
  let maxMarks : ℕ := 300
  calculatePercentage marksScored maxMarks = 95 := by
  sorry


end NUMINAMATH_CALUDE_victors_percentage_l472_47218


namespace NUMINAMATH_CALUDE_gails_wallet_l472_47288

/-- Represents the contents of Gail's wallet -/
structure Wallet where
  total : ℕ
  five_dollar_bills : ℕ
  twenty_dollar_bills : ℕ
  ten_dollar_bills : ℕ

/-- Calculates the total amount in the wallet based on the bill counts -/
def wallet_total (w : Wallet) : ℕ :=
  5 * w.five_dollar_bills + 20 * w.twenty_dollar_bills + 10 * w.ten_dollar_bills

/-- Theorem stating that given the conditions, Gail has 2 ten-dollar bills -/
theorem gails_wallet :
  ∃ (w : Wallet),
    w.total = 100 ∧
    w.five_dollar_bills = 4 ∧
    w.twenty_dollar_bills = 3 ∧
    wallet_total w = w.total ∧
    w.ten_dollar_bills = 2 := by
  sorry


end NUMINAMATH_CALUDE_gails_wallet_l472_47288


namespace NUMINAMATH_CALUDE_cos_inequality_range_l472_47241

theorem cos_inequality_range (x : Real) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.cos x ≤ 1 / 2 ↔ x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 3)) := by
  sorry

end NUMINAMATH_CALUDE_cos_inequality_range_l472_47241


namespace NUMINAMATH_CALUDE_smallest_number_drawn_l472_47297

/-- Represents a systematic sampling of classes -/
structure ClassSampling where
  total_classes : ℕ
  sample_size : ℕ
  sum_of_selected : ℕ

/-- Theorem: If we have 18 classes, sample 6 of them systematically, 
    and the sum of selected numbers is 57, then the smallest number drawn is 2 -/
theorem smallest_number_drawn (s : ClassSampling) 
  (h1 : s.total_classes = 18)
  (h2 : s.sample_size = 6)
  (h3 : s.sum_of_selected = 57) :
  ∃ x : ℕ, x = 2 ∧ 
    (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) + (x + 15) = s.sum_of_selected) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_drawn_l472_47297


namespace NUMINAMATH_CALUDE_four_m₀_is_sum_of_three_or_four_primes_l472_47244

-- Define the existence of a prime between n and 2n for any positive integer n
axiom exists_prime_between (n : ℕ) (hn : 0 < n) : ∃ p : ℕ, Prime p ∧ n ≤ p ∧ p ≤ 2 * n

-- Define the smallest even number greater than 2 that can't be expressed as sum of two primes
axiom exists_smallest_non_goldbach : ∃ m₀ : ℕ, 1 < m₀ ∧ 
  (∀ k < m₀, ∃ p q : ℕ, Prime p ∧ Prime q ∧ 2 * k = p + q) ∧
  (¬∃ p q : ℕ, Prime p ∧ Prime q ∧ 2 * m₀ = p + q)

-- Theorem statement
theorem four_m₀_is_sum_of_three_or_four_primes :
  ∃ m₀ : ℕ, 1 < m₀ ∧ 
  (∀ k < m₀, ∃ p q : ℕ, Prime p ∧ Prime q ∧ 2 * k = p + q) ∧
  (¬∃ p q : ℕ, Prime p ∧ Prime q ∧ 2 * m₀ = p + q) →
  ∃ p₁ p₂ p₃ p₄ : ℕ, (Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ 4 * m₀ = p₁ + p₂ + p₃) ∨
                     (Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 4 * m₀ = p₁ + p₂ + p₃ + p₄) :=
by sorry

end NUMINAMATH_CALUDE_four_m₀_is_sum_of_three_or_four_primes_l472_47244


namespace NUMINAMATH_CALUDE_least_repeating_digits_of_seven_thirteenths_l472_47259

theorem least_repeating_digits_of_seven_thirteenths : 
  (∀ n : ℕ, 0 < n → n < 6 → (10^n : ℤ) % 13 ≠ 1) ∧ (10^6 : ℤ) % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_least_repeating_digits_of_seven_thirteenths_l472_47259


namespace NUMINAMATH_CALUDE_maplefield_population_l472_47216

/-- The number of towns in the Region of Maplefield -/
def num_towns : ℕ := 25

/-- The lower bound of the average population range -/
def lower_bound : ℕ := 4800

/-- The upper bound of the average population range -/
def upper_bound : ℕ := 5300

/-- The average population of a town in the Region of Maplefield -/
def avg_population : ℚ := (lower_bound + upper_bound) / 2

/-- The total population of all towns in the Region of Maplefield -/
def total_population : ℚ := num_towns * avg_population

theorem maplefield_population : total_population = 126250 := by
  sorry

end NUMINAMATH_CALUDE_maplefield_population_l472_47216


namespace NUMINAMATH_CALUDE_carter_reads_30_pages_l472_47226

/-- The number of pages Oliver can read in 1 hour -/
def oliver_pages : ℕ := 40

/-- The number of pages Lucy can read in 1 hour -/
def lucy_pages : ℕ := oliver_pages + 20

/-- The number of pages Carter can read in 1 hour -/
def carter_pages : ℕ := lucy_pages / 2

/-- Theorem stating that Carter can read 30 pages in 1 hour -/
theorem carter_reads_30_pages : carter_pages = 30 := by
  sorry

end NUMINAMATH_CALUDE_carter_reads_30_pages_l472_47226


namespace NUMINAMATH_CALUDE_train_speed_calculation_l472_47235

/-- The speed of the first train -/
def speed_first_train : ℝ := 20

/-- The distance between stations P and Q -/
def distance_PQ : ℝ := 110

/-- The speed of the second train -/
def speed_second_train : ℝ := 25

/-- The time the first train travels before meeting -/
def time_first_train : ℝ := 3

/-- The time the second train travels before meeting -/
def time_second_train : ℝ := 2

theorem train_speed_calculation :
  speed_first_train * time_first_train + speed_second_train * time_second_train = distance_PQ :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l472_47235


namespace NUMINAMATH_CALUDE_yellow_surface_fraction_l472_47273

/-- Represents a large cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  total_small_cubes : ℕ
  yellow_cubes : ℕ
  blue_cubes : ℕ

/-- Calculates the minimum possible yellow surface area for a given large cube configuration -/
def min_yellow_surface_area (cube : LargeCube) : ℚ :=
  sorry

/-- Calculates the total surface area of the large cube -/
def total_surface_area (cube : LargeCube) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem yellow_surface_fraction (cube : LargeCube) 
  (h1 : cube.edge_length = 4)
  (h2 : cube.total_small_cubes = 64)
  (h3 : cube.yellow_cubes = 14)
  (h4 : cube.blue_cubes = 50)
  (h5 : cube.yellow_cubes + cube.blue_cubes = cube.total_small_cubes) :
  (min_yellow_surface_area cube) / (total_surface_area cube) = 7 / 48 :=
sorry

end NUMINAMATH_CALUDE_yellow_surface_fraction_l472_47273


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_1111_l472_47228

def repeat_digit (d : Nat) (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | m + 1 => d + 10 * (repeat_digit d m)

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n
  else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_square_1111 :
  sum_of_digits ((repeat_digit 1 4) ^ 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_1111_l472_47228


namespace NUMINAMATH_CALUDE_flagpole_break_height_approx_l472_47247

/-- The height of the flagpole in meters -/
def flagpole_height : ℝ := 5

/-- The distance from the base of the flagpole to where the broken part touches the ground, in meters -/
def ground_distance : ℝ := 1

/-- The approximate height where the flagpole breaks, in meters -/
def break_height : ℝ := 2.4

/-- Theorem stating that the break height is approximately correct -/
theorem flagpole_break_height_approx :
  let total_height := flagpole_height
  let distance := ground_distance
  let break_point := break_height
  abs (break_point - (total_height * distance / (2 * total_height))) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_flagpole_break_height_approx_l472_47247


namespace NUMINAMATH_CALUDE_triangle_perpendicular_theorem_l472_47292

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the foot of a perpendicular
def perpFoot (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Main theorem
theorem triangle_perpendicular_theorem (ABC : Triangle) :
  let A := ABC.A
  let B := ABC.B
  let C := ABC.C
  let D := perpFoot A B C
  length A B = 12 →
  length A C = 20 →
  (length B D) / (length C D) = 3 / 4 →
  length A D = 36 * Real.sqrt 14 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perpendicular_theorem_l472_47292


namespace NUMINAMATH_CALUDE_triangle_angle_from_side_relation_l472_47285

theorem triangle_angle_from_side_relation (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  0 < a ∧ 0 < b ∧ 0 < c →
  Real.sqrt 2 * a = 2 * b * Real.sin A →
  B = π / 4 ∨ B = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_from_side_relation_l472_47285


namespace NUMINAMATH_CALUDE_choose_four_from_seven_l472_47270

theorem choose_four_from_seven : Nat.choose 7 4 = 35 := by sorry

end NUMINAMATH_CALUDE_choose_four_from_seven_l472_47270


namespace NUMINAMATH_CALUDE_harry_hike_water_remaining_l472_47239

/-- Calculates the remaining water in Harry's canteen after a hike -/
def remaining_water (initial_water : ℝ) (hike_distance : ℝ) (hike_duration : ℝ) 
  (leak_rate : ℝ) (last_mile_consumption : ℝ) (first_miles_consumption_rate : ℝ) : ℝ :=
  initial_water - 
  (leak_rate * hike_duration) - 
  (first_miles_consumption_rate * (hike_distance - 1)) - 
  last_mile_consumption

/-- Theorem stating that the remaining water in Harry's canteen is 2 cups -/
theorem harry_hike_water_remaining :
  remaining_water 11 7 3 1 3 0.5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_harry_hike_water_remaining_l472_47239


namespace NUMINAMATH_CALUDE_pages_to_read_tonight_l472_47245

theorem pages_to_read_tonight (total_pages : ℕ) (first_night : ℕ) : 
  total_pages = 100 → 
  first_night = 15 → 
  (total_pages - (first_night + 2 * first_night + (2 * first_night + 5))) = 20 := by
  sorry

end NUMINAMATH_CALUDE_pages_to_read_tonight_l472_47245


namespace NUMINAMATH_CALUDE_divisor_sum_five_l472_47230

/-- d(m) is the number of positive divisors of m -/
def d (m : ℕ+) : ℕ := sorry

/-- Theorem: For a positive integer n, d(n) + d(n+1) = 5 if and only if n = 3 or n = 4 -/
theorem divisor_sum_five (n : ℕ+) : d n + d (n + 1) = 5 ↔ n = 3 ∨ n = 4 := by sorry

end NUMINAMATH_CALUDE_divisor_sum_five_l472_47230


namespace NUMINAMATH_CALUDE_train_speed_l472_47266

/-- Proves that a train with given specifications travels at 45 km/hr -/
theorem train_speed (train_length : Real) (crossing_time : Real) (total_length : Real) :
  train_length = 130 ∧ 
  crossing_time = 30 ∧ 
  total_length = 245 → 
  (total_length - train_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l472_47266


namespace NUMINAMATH_CALUDE_pizza_sales_distribution_l472_47286

/-- The total number of pizzas sold in a year -/
def total_pizzas : ℝ := 12.5

/-- The percentage of pizzas sold in summer -/
def summer_percent : ℝ := 0.4

/-- The number of pizzas sold in summer (in millions) -/
def summer_pizzas : ℝ := 5

/-- The percentage of pizzas sold in fall -/
def fall_percent : ℝ := 0.1

/-- The percentage of pizzas sold in winter -/
def winter_percent : ℝ := 0.2

/-- The number of pizzas sold in spring (in millions) -/
def spring_pizzas : ℝ := total_pizzas - (summer_pizzas + fall_percent * total_pizzas + winter_percent * total_pizzas)

theorem pizza_sales_distribution :
  spring_pizzas = 3.75 ∧
  summer_percent * total_pizzas = summer_pizzas ∧
  total_pizzas = summer_pizzas / summer_percent :=
by sorry

end NUMINAMATH_CALUDE_pizza_sales_distribution_l472_47286


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l472_47205

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem
theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) := by
sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l472_47205


namespace NUMINAMATH_CALUDE_gcd_of_45_135_225_l472_47210

theorem gcd_of_45_135_225 : Nat.gcd 45 (Nat.gcd 135 225) = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_135_225_l472_47210


namespace NUMINAMATH_CALUDE_equation_solutions_l472_47260

theorem equation_solutions :
  (∀ x : ℝ, 9 * (x - 1)^2 = 25 ↔ x = 8/3 ∨ x = -2/3) ∧
  (∀ x : ℝ, (1/3) * (x + 2)^3 - 9 = 0 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l472_47260


namespace NUMINAMATH_CALUDE_nine_rings_puzzle_5_l472_47255

def nine_rings_puzzle (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- Define for 0 to satisfy recursion
  | 1 => 1
  | n + 1 =>
    if n % 2 = 0 then
      2 * nine_rings_puzzle n + 2
    else
      2 * nine_rings_puzzle n - 1

theorem nine_rings_puzzle_5 :
  nine_rings_puzzle 5 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_nine_rings_puzzle_5_l472_47255


namespace NUMINAMATH_CALUDE_function_composition_equality_l472_47219

/-- Given two functions p and q, where p(x) = 5x - 4 and q(x) = 4x - b,
    prove that if p(q(5)) = 16, then b = 16. -/
theorem function_composition_equality (b : ℝ) : 
  (let p : ℝ → ℝ := λ x => 5 * x - 4
   let q : ℝ → ℝ := λ x => 4 * x - b
   p (q 5) = 16) → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l472_47219


namespace NUMINAMATH_CALUDE_hyperbola_equation_l472_47251

theorem hyperbola_equation (a b c : ℝ) (h1 : c = 4 * Real.sqrt 3) (h2 : a = 1) (h3 : b^2 = c^2 - a^2) :
  ∀ x y : ℝ, x^2 - y^2 / 47 = 1 ↔ x^2 - y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l472_47251


namespace NUMINAMATH_CALUDE_intersection_M_P_l472_47283

def M : Set ℝ := {x | x^2 = x}
def P : Set ℝ := {x | |x - 1| = 1}

theorem intersection_M_P : M ∩ P = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_P_l472_47283


namespace NUMINAMATH_CALUDE_box_of_balls_l472_47236

theorem box_of_balls (blue : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) : 
  blue = 6 →
  red = 4 →
  green = 3 * blue →
  yellow = 2 * red →
  blue + red + green + yellow = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_box_of_balls_l472_47236


namespace NUMINAMATH_CALUDE_goat_price_problem_l472_47208

theorem goat_price_problem (total_cost num_cows num_goats cow_price : ℕ) 
  (h1 : total_cost = 1500)
  (h2 : num_cows = 2)
  (h3 : num_goats = 10)
  (h4 : cow_price = 400) :
  (total_cost - num_cows * cow_price) / num_goats = 70 := by
  sorry

end NUMINAMATH_CALUDE_goat_price_problem_l472_47208


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l472_47246

theorem bowling_team_average_weight 
  (initial_players : ℕ) 
  (initial_average : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) 
  (h1 : initial_players = 7) 
  (h2 : initial_average = 94) 
  (h3 : new_player1_weight = 110) 
  (h4 : new_player2_weight = 60) :
  let total_weight := initial_players * initial_average + new_player1_weight + new_player2_weight
  let new_players := initial_players + 2
  (total_weight / new_players : ℝ) = 92 := by
sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l472_47246


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l472_47204

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_second_term
  (a : ℕ → ℤ)
  (h_arithmetic : ArithmeticSequence a)
  (h_10th : a 10 = 15)
  (h_11th : a 11 = 18) :
  a 2 = -9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l472_47204


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l472_47290

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 4 * x - 7
  {x : ℝ | f x < 0} = {x : ℝ | -1 < x ∧ x < 7/3} := by
sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l472_47290


namespace NUMINAMATH_CALUDE_triangle_side_angle_ratio_l472_47261

theorem triangle_side_angle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  b^2 = a * c →
  a^2 - c^2 = a * c - b * c →
  c / (b * Real.sin B) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_angle_ratio_l472_47261


namespace NUMINAMATH_CALUDE_equation_solution_l472_47224

theorem equation_solution :
  ∃ y : ℝ, (y = 18 / 7 ∧ (Real.sqrt (8 * y) / Real.sqrt (4 * (y - 2)) = 3)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l472_47224


namespace NUMINAMATH_CALUDE_mean_transformation_l472_47274

theorem mean_transformation (x₁ x₂ x₃ x₄ : ℝ) 
  (h : (x₁ + x₂ + x₃ + x₄) / 4 = 5) : 
  ((x₁ + 1) + (x₂ + 2) + (x₃ + x₄ + 4) + (5 + 5)) / 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_mean_transformation_l472_47274


namespace NUMINAMATH_CALUDE_amalia_reading_time_l472_47287

/-- Represents the time in minutes it takes Amalia to read a given number of pages -/
def reading_time (pages : ℕ) : ℚ :=
  (pages : ℚ) * 2 / 4

/-- Theorem stating that it takes Amalia 9 minutes to read 18 pages -/
theorem amalia_reading_time :
  reading_time 18 = 9 := by
  sorry

end NUMINAMATH_CALUDE_amalia_reading_time_l472_47287


namespace NUMINAMATH_CALUDE_cinema_systematic_sampling_l472_47221

/-- Represents a sampling method --/
inductive SamplingMethod
  | LotteryMethod
  | RandomNumberMethod
  | StratifiedSampling
  | SystematicSampling

/-- Represents a cinema with rows and seats --/
structure Cinema where
  rows : Nat
  seatsPerRow : Nat

/-- Represents a selection of audience members --/
structure AudienceSelection where
  seatNumber : Nat
  count : Nat

/-- Determines the sampling method based on the cinema layout and audience selection --/
def determineSamplingMethod (c : Cinema) (a : AudienceSelection) : SamplingMethod :=
  sorry

/-- Theorem stating that the given scenario results in systematic sampling --/
theorem cinema_systematic_sampling (c : Cinema) (a : AudienceSelection) :
  c.rows = 30 ∧ c.seatsPerRow = 25 ∧ a.seatNumber = 18 ∧ a.count = 30 →
  determineSamplingMethod c a = SamplingMethod.SystematicSampling :=
  sorry

end NUMINAMATH_CALUDE_cinema_systematic_sampling_l472_47221


namespace NUMINAMATH_CALUDE_students_with_b_in_smith_class_l472_47225

/-- Calculates the number of students who received a B in Ms. Smith's class -/
theorem students_with_b_in_smith_class 
  (johnson_total : ℕ) 
  (johnson_b : ℕ) 
  (smith_total : ℕ) 
  (h1 : johnson_total = 30)
  (h2 : johnson_b = 18)
  (h3 : smith_total = 45)
  (h4 : johnson_b * smith_total = johnson_total * (smith_total * johnson_b / johnson_total)) :
  smith_total * johnson_b / johnson_total = 27 := by
  sorry

#check students_with_b_in_smith_class

end NUMINAMATH_CALUDE_students_with_b_in_smith_class_l472_47225


namespace NUMINAMATH_CALUDE_marc_total_spend_l472_47240

/-- The total amount spent by Marc on his purchase of model cars, paint bottles, and paintbrushes. -/
def total_spent (num_cars num_paint num_brushes : ℕ) (price_car price_paint price_brush : ℚ) : ℚ :=
  num_cars * price_car + num_paint * price_paint + num_brushes * price_brush

/-- Theorem stating that Marc's total spend is $160 given his purchases. -/
theorem marc_total_spend :
  total_spent 5 5 5 20 10 2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_marc_total_spend_l472_47240


namespace NUMINAMATH_CALUDE_inequality_solution_set_l472_47202

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (x - 1) * (x + a) > 0}
  if a < -1 then
    S = {x : ℝ | x > -a ∨ x < 1}
  else if a = -1 then
    S = {x : ℝ | x ≠ 1}
  else
    S = {x : ℝ | x < -a ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l472_47202


namespace NUMINAMATH_CALUDE_final_postcard_count_l472_47250

-- Define the exchange rates
def euro_to_usd : ℚ := 1.20
def gbp_to_usd : ℚ := 1.35
def usd_to_yen : ℚ := 110

-- Define the initial number of postcards and sales
def initial_postcards : ℕ := 18
def sold_euro : ℕ := 6
def sold_gbp : ℕ := 3
def sold_usd : ℕ := 2

-- Define the prices of sold postcards
def price_euro : ℚ := 10
def price_gbp : ℚ := 12
def price_usd : ℚ := 15

-- Define the price of new postcards in USD
def new_postcard_price_usd : ℚ := 8

-- Define the price of additional postcards in Yen
def additional_postcard_price_yen : ℚ := 800

-- Define the percentage of earnings used to buy new postcards
def percentage_for_new_postcards : ℚ := 0.70

-- Define the number of additional postcards bought
def additional_postcards : ℕ := 5

-- Theorem statement
theorem final_postcard_count :
  let total_earnings_usd := sold_euro * price_euro * euro_to_usd + 
                            sold_gbp * price_gbp * gbp_to_usd + 
                            sold_usd * price_usd
  let new_postcards := (total_earnings_usd * percentage_for_new_postcards / new_postcard_price_usd).floor
  let remaining_usd := total_earnings_usd - new_postcards * new_postcard_price_usd
  let additional_postcards_bought := (remaining_usd * usd_to_yen / additional_postcard_price_yen).floor
  initial_postcards - (sold_euro + sold_gbp + sold_usd) + new_postcards + additional_postcards_bought = 26 :=
by sorry

end NUMINAMATH_CALUDE_final_postcard_count_l472_47250


namespace NUMINAMATH_CALUDE_train_speed_problem_l472_47277

/-- Represents a train with its speed and travel time after meeting another train -/
structure Train where
  speed : ℝ
  time_after_meeting : ℝ

/-- Proves that given the conditions of the problem, the speed of train B is 225 km/h -/
theorem train_speed_problem (train_A train_B : Train) 
  (h1 : train_A.speed = 100)
  (h2 : train_A.time_after_meeting = 9)
  (h3 : train_B.time_after_meeting = 4)
  (h4 : train_A.speed * train_A.time_after_meeting = train_B.speed * train_B.time_after_meeting) :
  train_B.speed = 225 := by
  sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_train_speed_problem_l472_47277


namespace NUMINAMATH_CALUDE_pilot_tuesday_miles_l472_47262

/-- 
Given a pilot's flight schedule where:
- The pilot flies x miles on Tuesday and 1475 miles on Thursday in 1 week
- This pattern is repeated for 3 weeks
- The total miles flown in 3 weeks is 7827 miles

Prove that the number of miles flown on Tuesday (x) is 1134.
-/
theorem pilot_tuesday_miles : 
  ∀ x : ℕ, 
  (3 * (x + 1475) = 7827) → 
  x = 1134 := by
sorry

end NUMINAMATH_CALUDE_pilot_tuesday_miles_l472_47262


namespace NUMINAMATH_CALUDE_seven_non_drinkers_l472_47293

/-- Represents the number of businessmen who drank a specific beverage or combination of beverages -/
structure BeverageCounts where
  total : Nat
  coffee : Nat
  tea : Nat
  water : Nat
  coffeeAndTea : Nat
  teaAndWater : Nat
  coffeeAndWater : Nat
  allThree : Nat

/-- Calculates the number of businessmen who drank none of the beverages -/
def nonDrinkers (counts : BeverageCounts) : Nat :=
  counts.total - (counts.coffee + counts.tea + counts.water
                  - counts.coffeeAndTea - counts.teaAndWater - counts.coffeeAndWater
                  + counts.allThree)

/-- Theorem stating that given the conditions, 7 businessmen drank none of the beverages -/
theorem seven_non_drinkers (counts : BeverageCounts)
  (h1 : counts.total = 30)
  (h2 : counts.coffee = 15)
  (h3 : counts.tea = 13)
  (h4 : counts.water = 6)
  (h5 : counts.coffeeAndTea = 7)
  (h6 : counts.teaAndWater = 3)
  (h7 : counts.coffeeAndWater = 2)
  (h8 : counts.allThree = 1) :
  nonDrinkers counts = 7 := by
  sorry

#eval nonDrinkers { total := 30, coffee := 15, tea := 13, water := 6,
                    coffeeAndTea := 7, teaAndWater := 3, coffeeAndWater := 2, allThree := 1 }

end NUMINAMATH_CALUDE_seven_non_drinkers_l472_47293


namespace NUMINAMATH_CALUDE_whirling_wonderland_capacity_l472_47279

/-- The 'Whirling Wonderland' ride problem -/
theorem whirling_wonderland_capacity :
  let people_per_carriage : ℕ := 12
  let number_of_carriages : ℕ := 15
  let total_capacity : ℕ := people_per_carriage * number_of_carriages
  total_capacity = 180 := by
  sorry

end NUMINAMATH_CALUDE_whirling_wonderland_capacity_l472_47279


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l472_47200

theorem decimal_to_fraction :
  (3.75 : ℚ) = 15 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l472_47200


namespace NUMINAMATH_CALUDE_boys_in_class_l472_47299

/-- Proves that in a class of 20 students, if exactly one-third of the boys sit with a girl
    and exactly one-half of the girls sit with a boy, then there are 12 boys in the class. -/
theorem boys_in_class (total_students : ℕ) (boys : ℕ) (girls : ℕ) :
  total_students = 20 →
  boys + girls = total_students →
  (boys / 3 : ℚ) = (girls / 2 : ℚ) →
  boys = 12 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l472_47299


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_pi_6_l472_47212

theorem sin_2alpha_plus_pi_6 (α : ℝ) (h : Real.sin (α - π/3) = 2/3 + Real.sin α) :
  Real.sin (2*α + π/6) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_pi_6_l472_47212


namespace NUMINAMATH_CALUDE_expression_range_l472_47211

theorem expression_range (x a b c : ℝ) (h : a^2 + b^2 + c^2 ≠ 0) :
  ∃ y ∈ Set.Icc (-Real.sqrt 5) (Real.sqrt 5),
    y = (a * Real.cos x - b * Real.sin x + 2 * c) / Real.sqrt (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_range_l472_47211


namespace NUMINAMATH_CALUDE_inequality_proof_l472_47252

theorem inequality_proof (w x y z : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt ((w^2 + x^2 + y^2 + z^2) / 4) ≥ ((wxy + wxz + wyz + xyz) / 4)^(1/3) :=
by
  sorry

where
  wxy := w * x * y
  wxz := w * x * z
  wyz := w * y * z
  xyz := x * y * z

end NUMINAMATH_CALUDE_inequality_proof_l472_47252


namespace NUMINAMATH_CALUDE_product_of_roots_l472_47220

theorem product_of_roots (x : ℝ) : (x + 4) * (x - 5) = 22 → ∃ y : ℝ, (x + 4) * (x - 5) = 22 ∧ (x * y = -42) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l472_47220


namespace NUMINAMATH_CALUDE_snail_distance_bound_l472_47271

/-- Represents the crawling of a snail over time -/
structure SnailCrawl where
  -- The distance function of the snail over time
  distance : ℝ → ℝ
  -- The distance function is non-decreasing (snail doesn't move backward)
  monotone : Monotone distance
  -- The total observation time
  total_time : ℝ
  -- The total time is 6 minutes
  total_time_is_six : total_time = 6

/-- Represents an observation of the snail -/
structure Observation where
  -- Start time of the observation
  start_time : ℝ
  -- Duration of the observation (1 minute)
  duration : ℝ
  duration_is_one : duration = 1
  -- The observation starts within the total time
  start_within_total : start_time ≥ 0 ∧ start_time + duration ≤ 6

/-- The theorem stating that the snail's total distance is at most 10 meters -/
theorem snail_distance_bound (crawl : SnailCrawl) 
  (observations : List Observation) 
  (observed_distance : ∀ obs ∈ observations, 
    crawl.distance (obs.start_time + obs.duration) - crawl.distance obs.start_time = 1) :
  crawl.distance crawl.total_time - crawl.distance 0 ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_snail_distance_bound_l472_47271


namespace NUMINAMATH_CALUDE_max_profit_at_optimal_price_l472_47243

/-- Represents the e-commerce platform's T-shirt sales scenario -/
structure TShirtSales where
  cost : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ
  min_price : ℝ
  max_margin : ℝ

/-- Calculates the profit for a given selling price -/
def profit (s : TShirtSales) (price : ℝ) : ℝ :=
  (price - s.cost) * (s.initial_sales + s.price_sensitivity * (s.initial_price - price))

/-- Theorem stating the maximum profit and optimal price -/
theorem max_profit_at_optimal_price (s : TShirtSales) 
  (h_cost : s.cost = 40)
  (h_initial_price : s.initial_price = 60)
  (h_initial_sales : s.initial_sales = 500)
  (h_price_sensitivity : s.price_sensitivity = 50)
  (h_min_price : s.min_price = s.cost)
  (h_max_margin : s.max_margin = 0.3)
  (h_price_range : ∀ p, s.min_price ≤ p ∧ p ≤ s.cost * (1 + s.max_margin) → 
    profit s p ≤ profit s 52) :
  profit s 52 = 10800 ∧ 
  ∀ p, s.min_price ≤ p ∧ p ≤ s.cost * (1 + s.max_margin) → profit s p ≤ 10800 := by
  sorry


end NUMINAMATH_CALUDE_max_profit_at_optimal_price_l472_47243


namespace NUMINAMATH_CALUDE_probability_one_red_two_blue_l472_47242

/-- The probability of selecting one red marble and two blue marbles from a bag -/
theorem probability_one_red_two_blue (total_marbles : ℕ) (red_marbles : ℕ) (blue_marbles : ℕ) 
  (h1 : total_marbles = red_marbles + blue_marbles)
  (h2 : red_marbles = 10)
  (h3 : blue_marbles = 6) : 
  (red_marbles * blue_marbles * (blue_marbles - 1) + 
   blue_marbles * red_marbles * (blue_marbles - 1) + 
   blue_marbles * (blue_marbles - 1) * red_marbles) / 
  (total_marbles * (total_marbles - 1) * (total_marbles - 2)) = 15 / 56 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_red_two_blue_l472_47242


namespace NUMINAMATH_CALUDE_johnson_finley_class_difference_l472_47291

theorem johnson_finley_class_difference (finley_class : ℕ) (johnson_class : ℕ) : 
  finley_class = 24 →
  johnson_class = 22 →
  johnson_class > finley_class / 2 →
  johnson_class - finley_class / 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_johnson_finley_class_difference_l472_47291

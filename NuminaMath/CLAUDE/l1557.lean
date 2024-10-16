import Mathlib

namespace NUMINAMATH_CALUDE_trigonometric_properties_l1557_155753

theorem trigonometric_properties :
  (∀ α : Real, 0 < α ∧ α < Real.pi / 2 → Real.sin α > 0) ∧
  (∃ α : Real, 0 < α ∧ α < Real.pi / 2 ∧ Real.cos (2 * α) > 0) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_properties_l1557_155753


namespace NUMINAMATH_CALUDE_no_modular_inverse_of_3_mod_33_l1557_155725

theorem no_modular_inverse_of_3_mod_33 : ¬ ∃ x : ℕ, x ≤ 32 ∧ (3 * x) % 33 = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_modular_inverse_of_3_mod_33_l1557_155725


namespace NUMINAMATH_CALUDE_car_speed_increase_car_speed_increase_proof_l1557_155760

/-- Calculates the increased speed of a car given initial conditions and final results -/
theorem car_speed_increase (v : ℝ) (initial_time stop_time delay additional_distance total_distance : ℝ) : ℝ :=
  let original_time := total_distance / v
  let actual_time := original_time + stop_time + delay
  let remaining_time := actual_time - initial_time
  let new_total_distance := total_distance + additional_distance
  let distance_after_stop := new_total_distance - (v * initial_time)
  distance_after_stop / remaining_time

/-- Proves that the increased speed of the car is approximately 34.91 km/hr given the problem conditions -/
theorem car_speed_increase_proof :
  let v := 32
  let initial_time := 3
  let stop_time := 0.25
  let delay := 0.5
  let additional_distance := 28
  let total_distance := 116
  abs (car_speed_increase v initial_time stop_time delay additional_distance total_distance - 34.91) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_increase_car_speed_increase_proof_l1557_155760


namespace NUMINAMATH_CALUDE_domain_implies_a_eq_3_odd_function_implies_a_eq_1_odd_function_solution_set_l1557_155720

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((2 / (x - 1)) + a)

-- Define the domain condition
def domain_condition (a : ℝ) : Prop :=
  ∀ x, f a x ≠ 0 ↔ (x < 1/3 ∨ x > 1)

-- Define the odd function condition
def odd_function (a : ℝ) : Prop :=
  ∀ x, f a (-x) = -(f a x)

-- State the theorems
theorem domain_implies_a_eq_3 :
  ∃ a, domain_condition a → a = 3 :=
sorry

theorem odd_function_implies_a_eq_1 :
  ∃ a, odd_function a → a = 1 :=
sorry

theorem odd_function_solution_set :
  ∀ a, odd_function a →
    (∀ x, f a x > 0 ↔ x > 1) :=
sorry

end NUMINAMATH_CALUDE_domain_implies_a_eq_3_odd_function_implies_a_eq_1_odd_function_solution_set_l1557_155720


namespace NUMINAMATH_CALUDE_power_of_two_equality_l1557_155791

theorem power_of_two_equality : (2^36 / 8 = 2^x) → x = 33 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l1557_155791


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l1557_155742

-- Define the binary operation ◇ on nonzero real numbers
noncomputable def diamond (a b : ℝ) : ℝ := a / b

-- State the theorem
theorem diamond_equation_solution :
  (∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 →
    diamond a (diamond b c) = (diamond a b) * c) →
  (∀ (a : ℝ), a ≠ 0 → diamond a a = 1) →
  (∃! (y : ℝ), diamond 2024 (diamond 8 y) = 200 ∧ y = 200 / 253) :=
by sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l1557_155742


namespace NUMINAMATH_CALUDE_mike_total_score_l1557_155737

/-- Given that Mike played six games of basketball and scored four points in each game,
    prove that his total score is 24 points. -/
theorem mike_total_score :
  let games_played : ℕ := 6
  let points_per_game : ℕ := 4
  let total_score := games_played * points_per_game
  total_score = 24 := by sorry

end NUMINAMATH_CALUDE_mike_total_score_l1557_155737


namespace NUMINAMATH_CALUDE_joint_purchase_savings_l1557_155733

/-- Represents the window purchase scenario --/
structure WindowPurchase where
  regularPrice : ℕ  -- Regular price of a window
  freeWindows : ℕ   -- Number of free windows
  paidWindows : ℕ   -- Number of windows that must be paid for to get free windows
  aliceNeeds : ℕ    -- Number of windows Alice needs
  bobNeeds : ℕ      -- Number of windows Bob needs

/-- Calculates the cost of a purchase given the number of windows needed --/
def calculateCost (wp : WindowPurchase) (windowsNeeded : ℕ) : ℕ :=
  let fullSets := windowsNeeded / (wp.paidWindows + wp.freeWindows)
  let remainingWindows := windowsNeeded % (wp.paidWindows + wp.freeWindows)
  (fullSets * wp.paidWindows + min remainingWindows wp.paidWindows) * wp.regularPrice

/-- Theorem stating the savings when purchasing together --/
theorem joint_purchase_savings (wp : WindowPurchase) 
    (h1 : wp.regularPrice = 100)
    (h2 : wp.freeWindows = 2)
    (h3 : wp.paidWindows = 10)
    (h4 : wp.aliceNeeds = 9)
    (h5 : wp.bobNeeds = 11) :
  calculateCost wp wp.aliceNeeds + calculateCost wp wp.bobNeeds - 
  calculateCost wp (wp.aliceNeeds + wp.bobNeeds) = 200 := by
  sorry

end NUMINAMATH_CALUDE_joint_purchase_savings_l1557_155733


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l1557_155712

/-- Given a trapezoid ABCD where:
    1. The ratio of the area of triangle ABC to the area of triangle ADC is 5:2
    2. AB + CD = 280 cm
    Prove that AB = 200 cm -/
theorem trapezoid_segment_length (AB CD : ℝ) (h : ℝ) : 
  (AB * h / 2) / (CD * h / 2) = 5 / 2 →
  AB + CD = 280 →
  AB = 200 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l1557_155712


namespace NUMINAMATH_CALUDE_subtraction_division_equality_l1557_155727

theorem subtraction_division_equality : 6000 - (105 / 21.0) = 5995 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_division_equality_l1557_155727


namespace NUMINAMATH_CALUDE_base_subtraction_l1557_155769

-- Define a function to convert from base b to base 10
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [1, 2, 3, 5, 4]
def base1 : Nat := 6

def num2 : List Nat := [1, 2, 3, 4]
def base2 : Nat := 7

-- State the theorem
theorem base_subtraction :
  to_base_10 num1 base1 - to_base_10 num2 base2 = 4851 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l1557_155769


namespace NUMINAMATH_CALUDE_new_number_properties_l1557_155781

def new_number (a b : ℕ) : ℕ := a * b + a + b

def is_new_number (n : ℕ) : Prop :=
  ∃ a b, new_number a b = n

theorem new_number_properties :
  (¬ is_new_number 2008) ∧
  (∀ a b : ℕ, 2 ∣ (new_number a b + 1)) ∧
  (∀ a b : ℕ, 10 ∣ (new_number a b + 1)) :=
sorry

end NUMINAMATH_CALUDE_new_number_properties_l1557_155781


namespace NUMINAMATH_CALUDE_no_divisible_with_small_digit_sum_l1557_155795

/-- Represents a number consisting of m ones -/
def ones (m : ℕ) : ℕ := 
  (10^m - 1) / 9

/-- Calculates the digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digitSum (n / 10)

/-- Theorem stating that no natural number divisible by ones(m) has a digit sum less than m -/
theorem no_divisible_with_small_digit_sum (m : ℕ) : 
  ¬ ∃ (n : ℕ), (n % ones m = 0) ∧ (digitSum n < m) := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_with_small_digit_sum_l1557_155795


namespace NUMINAMATH_CALUDE_gravel_path_rate_l1557_155703

/-- Calculates the rate per square meter for gravelling a path around a rectangular plot -/
theorem gravel_path_rate (length width path_width total_cost : ℝ) 
  (h1 : length = 110)
  (h2 : width = 65)
  (h3 : path_width = 2.5)
  (h4 : total_cost = 595) : 
  total_cost / ((length * width) - ((length - 2 * path_width) * (width - 2 * path_width))) = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_gravel_path_rate_l1557_155703


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1557_155766

/-- Given a rectangle with perimeter 60 and one side 5 units longer than the other,
    the maximum area is 218.75 square units. -/
theorem rectangle_max_area :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * (x + y) = 60 →
  y = x + 5 →
  x * y ≤ 218.75 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1557_155766


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1557_155799

theorem simplify_and_rationalize (x : ℝ) (h : x^3 = 3) : 
  1 / (1 + 1 / (x + 1)) = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1557_155799


namespace NUMINAMATH_CALUDE_weight_moved_three_triples_l1557_155786

/-- Calculates the total weight moved in three triples given the initial back squat, 
    back squat increase, front squat percentage, and triple percentage. -/
def total_weight_three_triples (initial_back_squat : ℝ) (back_squat_increase : ℝ) 
                               (front_squat_percentage : ℝ) (triple_percentage : ℝ) : ℝ :=
  let new_back_squat := initial_back_squat + back_squat_increase
  let front_squat := front_squat_percentage * new_back_squat
  let triple_weight := triple_percentage * front_squat
  3 * triple_weight

/-- Theorem stating that given the specific conditions, 
    the total weight moved in three triples is 540 kg. -/
theorem weight_moved_three_triples :
  total_weight_three_triples 200 50 0.8 0.9 = 540 := by
  sorry

end NUMINAMATH_CALUDE_weight_moved_three_triples_l1557_155786


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l1557_155761

theorem difference_of_squares_special_case : (23 * 2 + 15)^2 - (23 * 2 - 15)^2 = 2760 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l1557_155761


namespace NUMINAMATH_CALUDE_triangle_area_l1557_155722

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  B = π / 3 →
  a = 2 →
  b = Real.sqrt 3 →
  (1 / 2) * a * b * Real.sin B = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1557_155722


namespace NUMINAMATH_CALUDE_bumper_car_line_count_l1557_155765

/-- The number of people waiting in line for bumper cars after changes -/
def total_people_waiting (initial1 initial2 initial3 left1 left2 left3 joined1 joined2 joined3 : ℕ) : ℕ :=
  (initial1 - left1 + joined1) + (initial2 - left2 + joined2) + (initial3 - left3 + joined3)

/-- Theorem stating the total number of people waiting in line for bumper cars after changes -/
theorem bumper_car_line_count : 
  total_people_waiting 7 12 15 4 3 5 8 10 7 = 47 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_count_l1557_155765


namespace NUMINAMATH_CALUDE_sheep_grass_consumption_l1557_155708

theorem sheep_grass_consumption 
  (num_sheep : ℕ) 
  (num_bags : ℕ) 
  (num_days : ℕ) 
  (h1 : num_sheep = 40) 
  (h2 : num_bags = 40) 
  (h3 : num_days = 40) :
  num_days = 40 := by
  sorry

end NUMINAMATH_CALUDE_sheep_grass_consumption_l1557_155708


namespace NUMINAMATH_CALUDE_apartment_ages_puzzle_l1557_155705

def is_valid_triplet (a b c : ℕ) : Prop :=
  a * b * c = 1296 ∧ a < 100 ∧ b < 100 ∧ c < 100

def has_duplicate_sum (triplets : List (ℕ × ℕ × ℕ)) : Prop :=
  ∃ (t1 t2 : ℕ × ℕ × ℕ), t1 ∈ triplets ∧ t2 ∈ triplets ∧ t1 ≠ t2 ∧ 
    t1.1 + t1.2.1 + t1.2.2 = t2.1 + t2.2.1 + t2.2.2

theorem apartment_ages_puzzle :
  ∃! (a b c : ℕ), 
    is_valid_triplet a b c ∧
    (∀ triplets : List (ℕ × ℕ × ℕ), (∀ (x y z : ℕ), (x, y, z) ∈ triplets → is_valid_triplet x y z) →
      has_duplicate_sum triplets → (a, b, c) ∈ triplets) ∧
    a < b ∧ b < c ∧ c < 100 ∧
    a + b + c = 91 :=
by sorry

end NUMINAMATH_CALUDE_apartment_ages_puzzle_l1557_155705


namespace NUMINAMATH_CALUDE_smallest_banana_total_l1557_155770

/-- Represents the number of bananas taken by each monkey -/
structure BananaTaken where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Represents the final distribution of bananas among the monkeys -/
structure BananaDistribution where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Checks if the given banana distribution satisfies the problem conditions -/
def isValidDistribution (taken : BananaTaken) (dist : BananaDistribution) : Prop :=
  dist.first = taken.first / 2 + taken.second / 6 + taken.third / 9 + 7 * taken.fourth / 72 ∧
  dist.second = taken.first / 6 + taken.second / 3 + taken.third / 9 + 7 * taken.fourth / 72 ∧
  dist.third = taken.first / 6 + taken.second / 6 + taken.third / 6 + 7 * taken.fourth / 72 ∧
  dist.fourth = taken.first / 6 + taken.second / 6 + taken.third / 9 + taken.fourth / 8 ∧
  dist.first = 4 * dist.fourth ∧
  dist.second = 3 * dist.fourth ∧
  dist.third = 2 * dist.fourth

/-- The main theorem stating the smallest possible total number of bananas -/
theorem smallest_banana_total :
  ∀ taken : BananaTaken,
  ∀ dist : BananaDistribution,
  isValidDistribution taken dist →
  taken.first + taken.second + taken.third + taken.fourth ≥ 432 :=
by sorry

end NUMINAMATH_CALUDE_smallest_banana_total_l1557_155770


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l1557_155729

/-- The transformation f that maps (x, y) to (x+2y, 2x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2*p.2, 2*p.1 - p.2)

/-- Theorem stating that the preimage of (3, 1) under f is (1, 1) -/
theorem preimage_of_3_1 : f (1, 1) = (3, 1) := by sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l1557_155729


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1557_155775

/-- Given a rectangle with length thrice its breadth and area 588 square meters,
    prove that its perimeter is 112 meters. -/
theorem rectangle_perimeter (breadth : ℝ) (length : ℝ) (area : ℝ) (perimeter : ℝ) : 
  length = 3 * breadth →
  area = 588 →
  area = length * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 112 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1557_155775


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_value_l1557_155749

theorem quadratic_solution_implies_value (a b : ℝ) : 
  (1 : ℝ)^2 + a * 1 + 2 * b = 0 → 2023 - a - 2 * b = 2024 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_value_l1557_155749


namespace NUMINAMATH_CALUDE_kirill_height_difference_l1557_155718

theorem kirill_height_difference (combined_height kirill_height : ℕ) 
  (h1 : combined_height = 112)
  (h2 : kirill_height = 49) :
  combined_height - kirill_height - kirill_height = 14 := by
  sorry

end NUMINAMATH_CALUDE_kirill_height_difference_l1557_155718


namespace NUMINAMATH_CALUDE_monotonic_increasing_intervals_inequality_solution_l1557_155736

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Define the properties of f
def f_properties (a b c d : ℝ) : Prop :=
  -- f is symmetrical about the origin
  (∀ x, f a b c d x = -f a b c d (-x)) ∧
  -- f takes minimum value of -2 when x = 1
  (f a b c d 1 = -2) ∧
  (∀ x, f a b c d x ≥ -2)

-- Theorem for monotonically increasing intervals
theorem monotonic_increasing_intervals (a b c d : ℝ) (h : f_properties a b c d) :
  (∀ x y, x < y ∧ x < -1 → f a b c d x < f a b c d y) ∧
  (∀ x y, x < y ∧ y > 1 → f a b c d x < f a b c d y) := by sorry

-- Theorem for inequality solution
theorem inequality_solution (a b c d m : ℝ) (h : f_properties a b c d) :
  (m = 0 → ∀ x, x > 0 → f a b c d x > 5 * m * x^2 - (4 * m^2 + 3) * x) ∧
  (m > 0 → ∀ x, (x > 4*m ∨ (0 < x ∧ x < m)) → f a b c d x > 5 * m * x^2 - (4 * m^2 + 3) * x) ∧
  (m < 0 → ∀ x, (x > 0 ∨ (4*m < x ∧ x < m)) → f a b c d x > 5 * m * x^2 - (4 * m^2 + 3) * x) := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_intervals_inequality_solution_l1557_155736


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1557_155739

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 + 4*x + m ≥ 0) → m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1557_155739


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1557_155702

theorem quadratic_one_solution (k : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + k = 0) ↔ k = 49/12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1557_155702


namespace NUMINAMATH_CALUDE_mrs_heine_purchase_l1557_155711

/-- Calculates the total number of items purchased for dogs given the number of dogs,
    biscuits per dog, and boots per dog. -/
def total_items (num_dogs : ℕ) (biscuits_per_dog : ℕ) (boots_per_dog : ℕ) : ℕ :=
  num_dogs * (biscuits_per_dog + boots_per_dog)

/-- Proves that Mrs. Heine will buy 18 items in total for her dogs. -/
theorem mrs_heine_purchase : 
  let num_dogs : ℕ := 2
  let biscuits_per_dog : ℕ := 5
  let boots_per_set : ℕ := 4
  total_items num_dogs biscuits_per_dog boots_per_set = 18 := by
  sorry


end NUMINAMATH_CALUDE_mrs_heine_purchase_l1557_155711


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1557_155792

theorem algebraic_expression_equality (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (x + 1) / (x - 1) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1557_155792


namespace NUMINAMATH_CALUDE_investment_problem_l1557_155732

theorem investment_problem (total_investment rate1 rate2 rate3 : ℚ) 
  (h1 : total_investment = 6000)
  (h2 : rate1 = 7/100)
  (h3 : rate2 = 9/100)
  (h4 : rate3 = 11/100)
  (h5 : ∃ (a b c : ℚ), a + b + c = total_investment ∧ a / b = 2/3 ∧ b / c = 3) :
  ∃ (a b c : ℚ), 
    a + b + c = total_investment ∧ 
    a / b = 2/3 ∧ 
    b / c = 3 ∧
    a * rate1 + b * rate2 + c * rate3 = 520/100 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l1557_155732


namespace NUMINAMATH_CALUDE_result_circle_properties_l1557_155738

/-- The equation of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

/-- The equation of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - x = 0

/-- The equation of the resulting circle -/
def resultCircle (x y : ℝ) : Prop := 9*x^2 + 9*y^2 - 14*x + 4*y = 0

/-- Theorem stating that the resulting circle passes through the intersection points of the given circles and the point (1, -1) -/
theorem result_circle_properties :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → resultCircle x y) ∧
  resultCircle 1 (-1) := by
  sorry

end NUMINAMATH_CALUDE_result_circle_properties_l1557_155738


namespace NUMINAMATH_CALUDE_first_free_friday_after_college_start_l1557_155764

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

/-- Returns true if the given year is a leap year -/
def isLeapYear (year : ℕ) : Bool := sorry

/-- Returns the number of days in a given month of a given year -/
def daysInMonth (year : ℕ) (month : ℕ) : ℕ := sorry

/-- Returns the day of the week for a given date -/
def dayOfWeek (d : Date) : DayOfWeek := sorry

/-- Returns the next date after the given date -/
def nextDate (d : Date) : Date := sorry

/-- Returns true if the given date is a Friday -/
def isFriday (d : Date) : Bool := sorry

/-- Returns true if the given date is a Free Friday -/
def isFreeFriday (d : Date) : Bool := sorry

/-- Finds the first Free Friday after the given start date -/
def firstFreeFriday (startDate : Date) : Date := sorry

theorem first_free_friday_after_college_start :
  let collegeStart := Date.mk 2023 2 1
  firstFreeFriday collegeStart = Date.mk 2023 3 31 := by sorry

end NUMINAMATH_CALUDE_first_free_friday_after_college_start_l1557_155764


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1557_155794

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^2 + 3 * x - 8) * 3 * x^3 = 15 * x^5 + 9 * x^4 - 24 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1557_155794


namespace NUMINAMATH_CALUDE_min_cows_for_safe_ducks_l1557_155716

/-- Represents the arrangement of animals in Farmer Bill's circle -/
structure AnimalArrangement where
  total : Nat
  ducks : Nat
  cows : Nat
  rabbits : Nat

/-- Checks if the arrangement satisfies the safety condition for ducks -/
def isSafeArrangement (arr : AnimalArrangement) : Prop :=
  arr.ducks ≤ (arr.rabbits - 1) + 2 * arr.cows

/-- The main theorem stating the minimum number of cows required -/
theorem min_cows_for_safe_ducks (arr : AnimalArrangement) 
  (h1 : arr.total = 1000)
  (h2 : arr.ducks = 600)
  (h3 : arr.total = arr.ducks + arr.cows + arr.rabbits)
  (h4 : isSafeArrangement arr) :
  arr.cows ≥ 201 ∧ ∃ (safeArr : AnimalArrangement), 
    safeArr.total = 1000 ∧ 
    safeArr.ducks = 600 ∧ 
    safeArr.cows = 201 ∧
    isSafeArrangement safeArr :=
sorry

end NUMINAMATH_CALUDE_min_cows_for_safe_ducks_l1557_155716


namespace NUMINAMATH_CALUDE_z_purely_imaginary_iff_z_in_second_quadrant_iff_l1557_155713

def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * Complex.I

theorem z_purely_imaginary_iff (m : ℝ) : 
  z m = Complex.I * Complex.im (z m) ↔ m = -1/2 := by sorry

theorem z_in_second_quadrant_iff (m : ℝ) :
  (Complex.re (z m) < 0 ∧ Complex.im (z m) > 0) ↔ -1/2 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_iff_z_in_second_quadrant_iff_l1557_155713


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1557_155788

/-- A positive geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n > 0 ∧ ∃ r : ℝ, r > 0 ∧ b (n + 1) = r * b n

theorem geometric_sequence_property (b : ℕ → ℝ) (h : GeometricSequence b) :
  (b 1 * b 2 * b 3 * b 4 * b 5 * b 6) ^ (1/6 : ℝ) = (b 3 * b 4) ^ (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1557_155788


namespace NUMINAMATH_CALUDE_real_solutions_quadratic_l1557_155773

theorem real_solutions_quadratic (x : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 5 * x * y - 2 * x + 8 = 0) ↔ (x ≤ -12/5 ∨ x ≥ 8/5) :=
sorry

end NUMINAMATH_CALUDE_real_solutions_quadratic_l1557_155773


namespace NUMINAMATH_CALUDE_log_5_125000_bounds_l1557_155787

theorem log_5_125000_bounds : ∃ (a b : ℤ), 
  (a : ℝ) < Real.log 125000 / Real.log 5 ∧ 
  Real.log 125000 / Real.log 5 < (b : ℝ) ∧ 
  a = 6 ∧ 
  b = 7 ∧ 
  a + b = 13 := by
sorry

end NUMINAMATH_CALUDE_log_5_125000_bounds_l1557_155787


namespace NUMINAMATH_CALUDE_bus_average_speed_l1557_155747

/-- The average speed of a bus catching up to a bicycle -/
theorem bus_average_speed (bicycle_speed : ℝ) (initial_distance : ℝ) (catch_up_time : ℝ) :
  bicycle_speed = 15 →
  initial_distance = 195 →
  catch_up_time = 3 →
  (initial_distance + bicycle_speed * catch_up_time) / catch_up_time = 80 :=
by sorry

end NUMINAMATH_CALUDE_bus_average_speed_l1557_155747


namespace NUMINAMATH_CALUDE_parabola_symmetry_l1557_155772

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shift a parabola horizontally and vertically -/
def shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k + dy }

/-- Reflect a parabola about the x-axis -/
def reflect_x (p : Parabola) : Parabola :=
  { a := -p.a, h := p.h, k := -p.k }

theorem parabola_symmetry (A B C : Parabola) :
  A = reflect_x B →
  C = shift B 2 1 →
  C.a = 2 ∧ C.h = -1 ∧ C.k = -1 →
  A.a = -2 ∧ A.h = 1 ∧ A.k = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l1557_155772


namespace NUMINAMATH_CALUDE_distance_point_to_line_is_correct_l1557_155790

def point_A : ℝ × ℝ × ℝ := (0, 3, -1)
def point_B : ℝ × ℝ × ℝ := (1, 2, 1)
def point_C : ℝ × ℝ × ℝ := (2, 4, 0)

def line_direction : ℝ × ℝ × ℝ := (point_C.1 - point_B.1, point_C.2.1 - point_B.2.1, point_C.2.2 - point_B.2.2)

def distance_point_to_line (A B C : ℝ × ℝ × ℝ) : ℝ := sorry

theorem distance_point_to_line_is_correct :
  distance_point_to_line point_A point_B point_C = (3 * Real.sqrt 2) / 2 := by sorry

end NUMINAMATH_CALUDE_distance_point_to_line_is_correct_l1557_155790


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l1557_155779

theorem least_positive_integer_to_multiple_of_five : ∃ (n : ℕ), n > 0 ∧ (567 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (567 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l1557_155779


namespace NUMINAMATH_CALUDE_unique_arrangement_l1557_155709

-- Define the Letter type
inductive Letter
| A
| B

-- Define a function to represent whether a letter tells the truth
def tellsTruth (l : Letter) : Bool :=
  match l with
  | Letter.A => true
  | Letter.B => false

-- Define the statements made by each letter
def statement1 (l1 l2 l3 : Letter) : Prop :=
  (l1 = l2 ∧ l1 ≠ l3) ∨ (l1 = l3 ∧ l1 ≠ l2)

def statement2 (l1 l2 l3 : Letter) : Prop :=
  (l1 = Letter.A → l2 ≠ Letter.A) ∧ (l3 = Letter.A → l2 ≠ Letter.A)

def statement3 (l1 l2 l3 : Letter) : Prop :=
  (l1 = Letter.B ∧ l2 ≠ Letter.B ∧ l3 ≠ Letter.B) ∨
  (l1 ≠ Letter.B ∧ l2 = Letter.B ∧ l3 ≠ Letter.B) ∨
  (l1 ≠ Letter.B ∧ l2 ≠ Letter.B ∧ l3 = Letter.B)

-- Define the main theorem
theorem unique_arrangement :
  ∃! (l1 l2 l3 : Letter),
    (tellsTruth l1 → statement1 l1 l2 l3) ∧
    (¬tellsTruth l1 → ¬statement1 l1 l2 l3) ∧
    (tellsTruth l2 → statement2 l1 l2 l3) ∧
    (¬tellsTruth l2 → ¬statement2 l1 l2 l3) ∧
    (tellsTruth l3 → statement3 l1 l2 l3) ∧
    (¬tellsTruth l3 → ¬statement3 l1 l2 l3) ∧
    l1 = Letter.B ∧ l2 = Letter.A ∧ l3 = Letter.A :=
  by sorry

end NUMINAMATH_CALUDE_unique_arrangement_l1557_155709


namespace NUMINAMATH_CALUDE_inequality_proof_l1557_155762

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  Real.sqrt (1 - a^2) + Real.sqrt (1 - b^2) + Real.sqrt (1 - c^2) + a + b + c > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1557_155762


namespace NUMINAMATH_CALUDE_inheritance_theorem_l1557_155771

/-- Represents the inheritance distribution system -/
structure InheritanceSystem where
  total : ℕ  -- Total inheritance
  sons : ℕ   -- Number of sons
  share : ℕ  -- Share per son

/-- Calculates the share of the nth son -/
def nthSonShare (n : ℕ) (total : ℕ) : ℕ :=
  100 * n + (total - 100 * n) / 10

/-- Checks if all sons receive equal shares -/
def allSharesEqual (system : InheritanceSystem) : Prop :=
  ∀ i j, i ≤ system.sons → j ≤ system.sons →
    nthSonShare i system.total = nthSonShare j system.total

/-- The main theorem about the inheritance system -/
theorem inheritance_theorem (system : InheritanceSystem) :
  allSharesEqual system →
  system.total = 8100 ∧ system.sons = 9 ∧ system.share = 900 :=
sorry

end NUMINAMATH_CALUDE_inheritance_theorem_l1557_155771


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1557_155748

theorem solution_set_inequality (x : ℝ) : 
  (x * (x - 1) < 0) ↔ (0 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1557_155748


namespace NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_teacher_A_left_of_B_arrangements_teacher_at_far_left_arrangements_teachers_middle_students_height_ordered_arrangements_l1557_155782

-- Define the number of teachers and students
def num_teachers : ℕ := 2
def num_students : ℕ := 4

-- Part 1
theorem teachers_not_adjacent_arrangements : 
  (num_teachers.factorial * num_students.factorial * (num_students + 1).choose num_teachers) = 480 := by sorry

-- Part 2
theorem teacher_A_left_of_B_arrangements : 
  (num_teachers + num_students).factorial / 2 = 360 := by sorry

-- Part 3
theorem teacher_at_far_left_arrangements : 
  (num_students + 1).factorial + (num_students.choose 1 * num_students.factorial) = 216 := by sorry

-- Part 4
theorem teachers_middle_students_height_ordered_arrangements : 
  (num_teachers.factorial * num_students.choose (num_students / 2)) = 12 := by sorry

end NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_teacher_A_left_of_B_arrangements_teacher_at_far_left_arrangements_teachers_middle_students_height_ordered_arrangements_l1557_155782


namespace NUMINAMATH_CALUDE_total_miles_ridden_l1557_155728

-- Define the given conditions
def miles_to_school : ℕ := 6
def miles_from_school : ℕ := 7
def trips_per_week : ℕ := 5

-- Define the theorem to prove
theorem total_miles_ridden : 
  miles_to_school * trips_per_week + miles_from_school * trips_per_week = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_ridden_l1557_155728


namespace NUMINAMATH_CALUDE_octagon_area_theorem_l1557_155763

-- Define the rectangle BDEF
structure Rectangle :=
  (B D E F : ℝ × ℝ)

-- Define the octagon
structure Octagon :=
  (vertices : Fin 8 → ℝ × ℝ)

-- Define the condition AB = BC = 2
def side_length : ℝ := 2

-- Define the function to calculate the area of the octagon
noncomputable def octagon_area (rect : Rectangle) (side : ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem octagon_area_theorem (rect : Rectangle) :
  octagon_area rect side_length = 16 + 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_octagon_area_theorem_l1557_155763


namespace NUMINAMATH_CALUDE_unique_positive_integer_l1557_155780

theorem unique_positive_integer : ∃! (x : ℕ), x > 0 ∧ 15 * x = x^2 + 56 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_l1557_155780


namespace NUMINAMATH_CALUDE_jinyoung_has_fewest_l1557_155784

/-- Represents the number of marbles each person has -/
structure Marbles where
  seonho : ℕ
  minjeong : ℕ
  jinyoung : ℕ
  joohwan : ℕ

/-- The conditions of the marble problem -/
def marble_conditions (m : Marbles) : Prop :=
  m.seonho = m.minjeong + 1 ∧
  m.jinyoung = m.joohwan - 3 ∧
  m.minjeong = 6 ∧
  m.joohwan = 7

/-- Jinyoung has the fewest marbles -/
theorem jinyoung_has_fewest (m : Marbles) (h : marble_conditions m) :
  m.jinyoung ≤ m.seonho ∧ m.jinyoung ≤ m.minjeong ∧ m.jinyoung ≤ m.joohwan :=
by sorry

end NUMINAMATH_CALUDE_jinyoung_has_fewest_l1557_155784


namespace NUMINAMATH_CALUDE_january_salary_l1557_155740

/-- Prove that given the conditions, the salary for January is 3300 --/
theorem january_salary (jan feb mar apr may : ℕ) : 
  (jan + feb + mar + apr) / 4 = 8000 →
  (feb + mar + apr + may) / 4 = 8800 →
  may = 6500 →
  jan = 3300 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l1557_155740


namespace NUMINAMATH_CALUDE_doll_completion_time_l1557_155741

/-- Time in minutes to craft one doll -/
def craft_time : ℕ := 105

/-- Break time in minutes -/
def break_time : ℕ := 30

/-- Number of dolls to be made -/
def num_dolls : ℕ := 10

/-- Number of dolls after which a break is taken -/
def dolls_per_break : ℕ := 3

/-- Start time in minutes after midnight -/
def start_time : ℕ := 10 * 60

theorem doll_completion_time :
  let total_craft_time := num_dolls * craft_time
  let total_breaks := (num_dolls / dolls_per_break) * break_time
  let total_time := total_craft_time + total_breaks
  let completion_time := (start_time + total_time) % (24 * 60)
  completion_time = 5 * 60 :=
by sorry

end NUMINAMATH_CALUDE_doll_completion_time_l1557_155741


namespace NUMINAMATH_CALUDE_three_quarters_difference_l1557_155797

theorem three_quarters_difference (n : ℕ) (h : n = 76) : n - (3 * n / 4) = 19 := by
  sorry

end NUMINAMATH_CALUDE_three_quarters_difference_l1557_155797


namespace NUMINAMATH_CALUDE_asymptotic_function_part1_non_asymptotic_function_part2_l1557_155776

/-- Definition of asymptotic function -/
def is_asymptotic_function (f g : ℝ → ℝ) (p : ℝ) : Prop :=
  ∃ h : ℝ → ℝ, (∀ x ≥ 0, f x = g x + h x) ∧
  (Monotone (fun x ↦ -h x)) ∧
  (∀ x ≥ 0, 0 < h x ∧ h x ≤ p)

/-- Part I: Asymptotic function for f(x) = (x^2 + 2x + 3) / (x + 1) -/
theorem asymptotic_function_part1 :
  is_asymptotic_function (fun x ↦ (x^2 + 2*x + 3) / (x + 1)) (fun x ↦ x + 1) 2 :=
sorry

/-- Part II: Non-asymptotic function for f(x) = √(x^2 + 1) -/
theorem non_asymptotic_function_part2 (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ¬ is_asymptotic_function (fun x ↦ Real.sqrt (x^2 + 1)) (fun x ↦ a * x) p :=
sorry

end NUMINAMATH_CALUDE_asymptotic_function_part1_non_asymptotic_function_part2_l1557_155776


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1557_155707

theorem right_triangle_sets : ∃! (a b c : ℝ), (a = 6 ∧ b = 8 ∧ c = 13) ∧
  ¬(a^2 + b^2 = c^2) ∧
  (0.3^2 + 0.4^2 = 0.5^2) ∧
  (1^2 + 1^2 = (Real.sqrt 2)^2) ∧
  (8^2 + 15^2 = 17^2) :=
by
  sorry

#check right_triangle_sets

end NUMINAMATH_CALUDE_right_triangle_sets_l1557_155707


namespace NUMINAMATH_CALUDE_min_sum_on_circle_l1557_155743

theorem min_sum_on_circle (x y : ℝ) :
  Real.sqrt ((x - 2)^2 + (y - 1)^2) = 1 →
  ∃ (min : ℝ), min = 2 ∧ ∀ (a b : ℝ), Real.sqrt ((a - 2)^2 + (b - 1)^2) = 1 → x + y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_on_circle_l1557_155743


namespace NUMINAMATH_CALUDE_jimin_tape_length_l1557_155735

-- Define the conversion factor from cm to mm
def cm_to_mm : ℝ := 10

-- Define Jungkook's tape length in cm
def jungkook_tape_cm : ℝ := 45

-- Define the difference between Jimin's and Jungkook's tape lengths in mm
def tape_difference_mm : ℝ := 26

-- State the theorem
theorem jimin_tape_length :
  (jungkook_tape_cm * cm_to_mm + tape_difference_mm) / cm_to_mm = 47.6 := by
  sorry

end NUMINAMATH_CALUDE_jimin_tape_length_l1557_155735


namespace NUMINAMATH_CALUDE_housewife_purchasing_comparison_l1557_155723

theorem housewife_purchasing_comparison (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) :
  (2 * a * b) / (a + b) < (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_housewife_purchasing_comparison_l1557_155723


namespace NUMINAMATH_CALUDE_difference_in_dimes_l1557_155745

/-- The number of quarters Susan has -/
def susan_quarters (p : ℚ) : ℚ := 7 * p + 3

/-- The number of quarters George has -/
def george_quarters (p : ℚ) : ℚ := 2 * p + 9

/-- The conversion rate from quarters to dimes -/
def quarter_to_dime : ℚ := 2.5

theorem difference_in_dimes (p : ℚ) :
  (susan_quarters p - george_quarters p) * quarter_to_dime = 12.5 * p - 15 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_dimes_l1557_155745


namespace NUMINAMATH_CALUDE_age_ratio_problem_l1557_155731

theorem age_ratio_problem (alma_age melina_age alma_score : ℕ) : 
  alma_age + melina_age = 2 * alma_score →
  melina_age = 60 →
  alma_score = 40 →
  melina_age / alma_age = 3 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l1557_155731


namespace NUMINAMATH_CALUDE_set_operations_proof_l1557_155746

def A : Set ℕ := {1, 2}
def B : Set ℕ := {0, 2}
def C : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

theorem set_operations_proof :
  (A ∩ C ≠ ∅) ∧
  (A ∪ C ≠ C) ∧
  (B ∩ C = B) ∧
  (A ∪ B ≠ C) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_proof_l1557_155746


namespace NUMINAMATH_CALUDE_base8_543_to_base10_l1557_155704

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ :=
  let d₀ := n % 8
  let d₁ := (n / 8) % 8
  let d₂ := (n / 64) % 8
  d₀ + 8 * d₁ + 64 * d₂

theorem base8_543_to_base10 : base8ToBase10 543 = 355 := by sorry

end NUMINAMATH_CALUDE_base8_543_to_base10_l1557_155704


namespace NUMINAMATH_CALUDE_joes_fruit_spending_l1557_155758

theorem joes_fruit_spending (total_money : ℚ) (chocolate_fraction : ℚ) (money_left : ℚ) : 
  total_money = 450 →
  chocolate_fraction = 1/9 →
  money_left = 220 →
  (total_money - chocolate_fraction * total_money - money_left) / total_money = 2/5 := by
sorry

end NUMINAMATH_CALUDE_joes_fruit_spending_l1557_155758


namespace NUMINAMATH_CALUDE_daughter_age_is_40_l1557_155715

/-- Represents the family members' weights and ages -/
structure Family where
  mother_weight : ℝ
  daughter_weight : ℝ
  grandchild_weight : ℝ
  son_in_law_weight : ℝ
  mother_age : ℝ
  daughter_age : ℝ
  son_in_law_age : ℝ

/-- The family satisfies the given conditions -/
def satisfies_conditions (f : Family) : Prop :=
  f.mother_weight + f.daughter_weight + f.grandchild_weight + f.son_in_law_weight = 200 ∧
  f.daughter_weight + f.grandchild_weight = 60 ∧
  f.grandchild_weight = (1/5) * f.mother_weight ∧
  f.son_in_law_weight = 2 * f.daughter_weight ∧
  f.mother_age / f.daughter_age = 2 ∧
  f.daughter_age / f.son_in_law_age = 3/2 ∧
  f.mother_age = 80

/-- The theorem stating that if a family satisfies the given conditions, the daughter's age is 40 -/
theorem daughter_age_is_40 (f : Family) (h : satisfies_conditions f) : f.daughter_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_daughter_age_is_40_l1557_155715


namespace NUMINAMATH_CALUDE_fifth_power_prime_solution_l1557_155767

theorem fifth_power_prime_solution :
  ∀ (x y p : ℕ+),
  (x^2 + y) * (y^2 + x) = p^5 ∧ Nat.Prime p.val →
  ((x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2)) :=
sorry

end NUMINAMATH_CALUDE_fifth_power_prime_solution_l1557_155767


namespace NUMINAMATH_CALUDE_student_average_equals_actual_average_l1557_155719

theorem student_average_equals_actual_average 
  (w x y z : ℤ) (h : w < x ∧ x < y ∧ y < z) :
  (((w + x) / 2 + (y + z) / 2) / 2 : ℚ) = ((w + x + y + z) / 4 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_student_average_equals_actual_average_l1557_155719


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1557_155726

theorem simplify_and_evaluate (a : ℝ) (h : a = -Real.sqrt 2) :
  (a - 3) / a * 6 / (a^2 - 6*a + 9) - (2*a + 6) / (a^2 - 9) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1557_155726


namespace NUMINAMATH_CALUDE_machinery_expenditure_fraction_l1557_155734

theorem machinery_expenditure_fraction (C : ℝ) (h : C > 0) :
  let raw_material_cost : ℝ := (1 / 4) * C
  let remaining_after_raw : ℝ := C - raw_material_cost
  let final_remaining : ℝ := 0.675 * C
  let machinery_cost : ℝ := remaining_after_raw - final_remaining
  machinery_cost / remaining_after_raw = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_machinery_expenditure_fraction_l1557_155734


namespace NUMINAMATH_CALUDE_z_magnitude_l1557_155774

open Complex

/-- Euler's formula -/
axiom euler_formula (θ : ℝ) : exp (I * θ) = cos θ + I * sin θ

/-- The complex number z satisfies the given equation -/
def z : ℂ := by sorry

/-- The equation that z satisfies -/
axiom z_equation : (exp (I * Real.pi) - I) * z = 1

/-- The magnitude of z is √2/2 -/
theorem z_magnitude : abs z = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_z_magnitude_l1557_155774


namespace NUMINAMATH_CALUDE_movie_theater_seating_l1557_155759

def seat_arrangements (n : ℕ) : ℕ :=
  if n < 7 then 0
  else (n - 4).choose 3 * 2

theorem movie_theater_seating : seat_arrangements 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_seating_l1557_155759


namespace NUMINAMATH_CALUDE_tan_alpha_same_terminal_side_l1557_155755

-- Define the angle α
def α : Real := sorry

-- Define the condition that the terminal side of α lies on y = -√3x
axiom terminal_side : ∀ (x y : Real), y = -Real.sqrt 3 * x → (∃ (r : Real), r > 0 ∧ x = r * Real.cos α ∧ y = r * Real.sin α)

-- Theorem 1: tan α = -√3
theorem tan_alpha : Real.tan α = -Real.sqrt 3 := by sorry

-- Define the set S of angles with the same terminal side as α
def S : Set Real := {θ | ∃ (k : ℤ), θ = k * Real.pi + 2 * Real.pi / 3}

-- Theorem 2: S is the set of all angles with the same terminal side as α
theorem same_terminal_side (θ : Real) : 
  (∀ (x y : Real), y = -Real.sqrt 3 * x → (∃ (r : Real), r > 0 ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ)) ↔ θ ∈ S := by sorry

end NUMINAMATH_CALUDE_tan_alpha_same_terminal_side_l1557_155755


namespace NUMINAMATH_CALUDE_math_competition_non_participants_l1557_155721

theorem math_competition_non_participants (total_students : ℕ) 
  (participation_ratio : ℚ) (h1 : total_students = 89) 
  (h2 : participation_ratio = 3/5) : 
  total_students - (participation_ratio * total_students).floor = 35 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_non_participants_l1557_155721


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1557_155714

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 4 * x + a - 3 < 0) ↔ a < 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1557_155714


namespace NUMINAMATH_CALUDE_friday_blood_pressure_l1557_155724

/-- Calculates the final blood pressure given an initial value and a list of daily changes. -/
def finalBloodPressure (initial : ℕ) (changes : List ℤ) : ℕ :=
  (changes.foldl (fun acc change => (acc : ℤ) + change) initial).toNat

/-- Theorem stating that given the initial blood pressure and daily changes, 
    the final blood pressure on Friday is 130 units. -/
theorem friday_blood_pressure :
  let initial : ℕ := 120
  let changes : List ℤ := [20, -30, -25, 15, 30]
  finalBloodPressure initial changes = 130 := by sorry

end NUMINAMATH_CALUDE_friday_blood_pressure_l1557_155724


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1557_155768

def is_arithmetic (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 6 = 32 →
  a 4 + a 5 = 55 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1557_155768


namespace NUMINAMATH_CALUDE_abc_product_l1557_155789

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 24) (hac : a * c = 40) (hbc : b * c = 60) : a * b * c = 240 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l1557_155789


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l1557_155777

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop := l1.a * l2.b = l2.a * l1.b

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_through_point_parallel_to_line 
  (given_line : Line) 
  (point : Point) :
  ∃ (result_line : Line),
    parallel result_line given_line ∧
    on_line point result_line ∧
    result_line.a = 1 ∧
    result_line.b = -2 ∧
    result_line.c = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l1557_155777


namespace NUMINAMATH_CALUDE_student_count_l1557_155798

/-- Given a student's position from both ends of a line, calculate the total number of students -/
theorem student_count (right_rank left_rank : ℕ) (h1 : right_rank = 13) (h2 : left_rank = 8) :
  right_rank + left_rank - 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1557_155798


namespace NUMINAMATH_CALUDE_line_AB_intersects_S₂_and_S_l1557_155783

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def S₁ : Circle := { center := (0, 0), radius := 1 }
def S₂ : Circle := { center := (2, 0), radius := 1 }
def S  : Circle := { center := (1, 1), radius := 2 }
def A  : ℝ × ℝ := (1, 0)
def B  : ℝ × ℝ := (1, 2)
def O  : ℝ × ℝ := S.center

-- Define the conditions
axiom S₁_S₂_tangent : S₁.center.fst + S₁.radius = S₂.center.fst - S₂.radius
axiom O_on_S₁ : (O.fst - S₁.center.fst)^2 + (O.snd - S₁.center.snd)^2 = S₁.radius^2
axiom S₁_S_tangent_at_B : (B.fst - S₁.center.fst)^2 + (B.snd - S₁.center.snd)^2 = S₁.radius^2 ∧
                          (B.fst - S.center.fst)^2 + (B.snd - S.center.snd)^2 = S.radius^2

-- Theorem to prove
theorem line_AB_intersects_S₂_and_S :
  ∃ (P : ℝ × ℝ), P ≠ A ∧ P ≠ B ∧
  (∃ (t : ℝ), P = (A.fst + t * (B.fst - A.fst), A.snd + t * (B.snd - A.snd))) ∧
  (P.fst - S₂.center.fst)^2 + (P.snd - S₂.center.snd)^2 = S₂.radius^2 ∧
  (P.fst - S.center.fst)^2 + (P.snd - S.center.snd)^2 = S.radius^2 :=
sorry

end NUMINAMATH_CALUDE_line_AB_intersects_S₂_and_S_l1557_155783


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l1557_155793

/-- Represents a two-digit number -/
def two_digit_number := { n : ℕ | 10 ≤ n ∧ n < 100 }

/-- Constructs a six-digit number by repeating a two-digit number three times -/
def repeat_three_times (n : two_digit_number) : ℕ :=
  100000 * n + 1000 * n + n

theorem six_digit_divisibility (n : two_digit_number) :
  (repeat_three_times n) % 10101 = 0 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l1557_155793


namespace NUMINAMATH_CALUDE_exists_common_divisor_l1557_155717

/-- A function from positive integers to integers greater than 1 -/
def PositiveFunction := ℕ+ → ℕ+

/-- The property that f(m+n) divides f(m) + f(n) for all positive integers m and n -/
def HasDivisibilityProperty (f : PositiveFunction) : Prop :=
  ∀ m n : ℕ+, (f (m + n)) ∣ (f m + f n)

/-- The main theorem: if f has the divisibility property, then there exists c > 1 that divides all values of f -/
theorem exists_common_divisor (f : PositiveFunction) (h : HasDivisibilityProperty f) :
  ∃ c : ℕ+, c > 1 ∧ ∀ n : ℕ+, c ∣ f n :=
sorry

end NUMINAMATH_CALUDE_exists_common_divisor_l1557_155717


namespace NUMINAMATH_CALUDE_binomial_coefficient_26_6_l1557_155700

theorem binomial_coefficient_26_6 (h1 : Nat.choose 24 5 = 42504) (h2 : Nat.choose 24 6 = 134596) :
  Nat.choose 26 6 = 230230 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_26_6_l1557_155700


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1557_155796

theorem polynomial_factorization (x : ℝ) : 
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 9) = 
  (x^2 + 6*x + 3) * (x^2 + 6*x + 12) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1557_155796


namespace NUMINAMATH_CALUDE_sharon_has_13_plums_l1557_155710

/-- The number of plums Sharon has -/
def sharons_plums : ℕ := 13

/-- The number of plums Allan has -/
def allans_plums : ℕ := 10

/-- The difference between Sharon's plums and Allan's plums -/
def plum_difference : ℕ := 3

/-- Theorem: Given the conditions, Sharon has 13 plums -/
theorem sharon_has_13_plums : 
  sharons_plums = allans_plums + plum_difference :=
by sorry

end NUMINAMATH_CALUDE_sharon_has_13_plums_l1557_155710


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1557_155754

/-- Represents a school with its student count -/
structure School where
  students : ℕ

/-- Represents the sampling result for a school -/
structure SamplingResult where
  school : School
  sampleSize : ℕ

def totalStudents (schools : List School) : ℕ :=
  schools.foldl (fun acc school => acc + school.students) 0

def calculateSampleSize (school : School) (totalStudents : ℕ) (totalSampleSize : ℕ) : ℕ :=
  (school.students * totalSampleSize) / totalStudents

theorem stratified_sampling_theorem (schoolA schoolB schoolC : School)
    (h1 : schoolA.students = 3600)
    (h2 : schoolB.students = 5400)
    (h3 : schoolC.students = 1800)
    (totalSampleSize : ℕ)
    (h4 : totalSampleSize = 90) :
  let schools := [schoolA, schoolB, schoolC]
  let total := totalStudents schools
  let samplingResults := schools.map (fun school => 
    SamplingResult.mk school (calculateSampleSize school total totalSampleSize))
  samplingResults.map (fun result => result.sampleSize) = [30, 45, 15] := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1557_155754


namespace NUMINAMATH_CALUDE_polynomial_real_root_l1557_155756

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^4 + a*x^3 - x^2 + a*x + 1 = 0) ↔ 
  (a ≤ -1/2 ∨ a ≥ 1/2) := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l1557_155756


namespace NUMINAMATH_CALUDE_inverse_proportion_m_value_l1557_155751

def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

theorem inverse_proportion_m_value (m : ℝ) :
  (is_inverse_proportion (λ x => (m - 1) * x^(|m| - 2))) →
  m = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_m_value_l1557_155751


namespace NUMINAMATH_CALUDE_monica_milk_amount_l1557_155730

-- Define the initial amount of milk Don has
def dons_milk : ℚ := 3/4

-- Define the fraction of milk Don gives to Rachel
def fraction_to_rachel : ℚ := 1/2

-- Define the fraction of Rachel's milk that Monica drinks
def fraction_monica_drinks : ℚ := 1/3

-- Theorem statement
theorem monica_milk_amount :
  fraction_monica_drinks * (fraction_to_rachel * dons_milk) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_monica_milk_amount_l1557_155730


namespace NUMINAMATH_CALUDE_spring_length_at_9kg_spring_length_conditions_l1557_155778

/-- A linear function representing the relationship between mass and spring length. -/
def spring_length (x : ℝ) : ℝ := 0.5 * x + 10

/-- Theorem stating that the spring length is 14.5 cm when the mass is 9 kg. -/
theorem spring_length_at_9kg :
  spring_length 0 = 10 →
  spring_length 1 = 10.5 →
  spring_length 9 = 14.5 := by
  sorry

/-- Proof that the spring_length function satisfies the given conditions. -/
theorem spring_length_conditions :
  spring_length 0 = 10 ∧ spring_length 1 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_spring_length_at_9kg_spring_length_conditions_l1557_155778


namespace NUMINAMATH_CALUDE_beach_problem_l1557_155757

/-- The number of people originally in the second row of the beach -/
def original_second_row : ℕ := 20

theorem beach_problem :
  let first_row : ℕ := 24
  let first_row_left : ℕ := 3
  let second_row_left : ℕ := 5
  let third_row : ℕ := 18
  let total_remaining : ℕ := 54
  (first_row - first_row_left) + (original_second_row - second_row_left) + third_row = total_remaining :=
by sorry

end NUMINAMATH_CALUDE_beach_problem_l1557_155757


namespace NUMINAMATH_CALUDE_square_prism_volume_is_two_l1557_155744

/-- A square prism with all vertices on a sphere -/
structure SquarePrismOnSphere where
  /-- The side length of the square base -/
  side_length : ℝ
  /-- The height of the prism -/
  height : ℝ
  /-- The radius of the sphere -/
  sphere_radius : ℝ
  /-- All vertices of the prism lie on the sphere -/
  vertices_on_sphere : side_length ^ 2 * 2 + height ^ 2 = (2 * sphere_radius) ^ 2
  /-- The height of the prism is 2 -/
  height_is_two : height = 2
  /-- The surface area of the sphere is 6π -/
  sphere_surface_area : 4 * Real.pi * sphere_radius ^ 2 = 6 * Real.pi

/-- The volume of a square prism -/
def prism_volume (p : SquarePrismOnSphere) : ℝ := p.side_length ^ 2 * p.height

/-- Theorem: The volume of the square prism on the sphere is 2 -/
theorem square_prism_volume_is_two (p : SquarePrismOnSphere) : prism_volume p = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_prism_volume_is_two_l1557_155744


namespace NUMINAMATH_CALUDE_smallest_floor_x_l1557_155706

-- Define a tetrahedron type
structure Tetrahedron :=
  (a b c d e x : ℝ)

-- Define the conditions for a valid tetrahedron
def is_valid_tetrahedron (t : Tetrahedron) : Prop :=
  t.a = 4 ∧ t.b = 7 ∧ t.c = 20 ∧ t.d = 22 ∧ t.e = 28 ∧
  t.x > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b ∧
  t.a + t.d > t.e ∧ t.d + t.e > t.a ∧ t.e + t.a > t.d ∧
  t.b + t.d > t.x ∧ t.d + t.x > t.b ∧ t.x + t.b > t.d ∧
  t.b + t.e > t.c ∧ t.e + t.c > t.b ∧ t.c + t.b > t.e ∧
  t.c + t.d > t.x ∧ t.d + t.x > t.c ∧ t.x + t.c > t.d ∧
  t.c + t.e > t.x ∧ t.e + t.x > t.c ∧ t.x + t.c > t.e ∧
  t.d + t.e > t.x ∧ t.e + t.x > t.d ∧ t.x + t.d > t.e

-- Theorem statement
theorem smallest_floor_x (t : Tetrahedron) (h : is_valid_tetrahedron t) :
  ∀ (y : ℝ), (is_valid_tetrahedron {a := t.a, b := t.b, c := t.c, d := t.d, e := t.e, x := y} →
  ⌊t.x⌋ ≥ 8) ∧ (∃ (z : ℝ), is_valid_tetrahedron {a := t.a, b := t.b, c := t.c, d := t.d, e := t.e, x := z} ∧ ⌊z⌋ = 8) :=
sorry

end NUMINAMATH_CALUDE_smallest_floor_x_l1557_155706


namespace NUMINAMATH_CALUDE_food_for_horses_l1557_155752

/-- Calculates the total amount of food needed for horses over a number of days. -/
def total_food_needed (num_horses : ℕ) (num_days : ℕ) (oats_per_meal : ℕ) (oats_meals_per_day : ℕ) (grain_per_day : ℕ) : ℕ :=
  let total_oats := num_horses * num_days * oats_per_meal * oats_meals_per_day
  let total_grain := num_horses * num_days * grain_per_day
  total_oats + total_grain

/-- Theorem stating that the total food needed for 4 horses over 3 days is 132 pounds. -/
theorem food_for_horses :
  total_food_needed 4 3 4 2 3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_food_for_horses_l1557_155752


namespace NUMINAMATH_CALUDE_regular_triangular_prism_volume_l1557_155750

/-- 
Given a regular triangular prism where:
1. The lateral edge is equal to the height of the base
2. The area of the cross-section passing through the lateral edge and height of the base is Q

Prove that the volume of the prism is Q * sqrt(Q/3)
-/
theorem regular_triangular_prism_volume (Q : ℝ) (Q_pos : 0 < Q) : 
  ∃ (volume : ℝ), volume = Q * Real.sqrt (Q / 3) ∧ 
  ∃ (lateral_edge base_height : ℝ), 
    lateral_edge = base_height ∧
    Q = lateral_edge * base_height ∧
    volume = (Real.sqrt 3 / 4 * lateral_edge^2) * lateral_edge :=
by sorry

end NUMINAMATH_CALUDE_regular_triangular_prism_volume_l1557_155750


namespace NUMINAMATH_CALUDE_bella_apples_per_day_l1557_155785

/-- The number of apples Grace has left after 6 weeks -/
def apples_left : ℕ := 504

/-- The number of weeks -/
def weeks : ℕ := 6

/-- The fraction of apples Bella consumes from what Grace picks -/
def bella_fraction : ℚ := 1 / 3

/-- The number of apples Bella eats per day -/
def bella_daily_apples : ℕ := 6

theorem bella_apples_per_day :
  ∃ (grace_total : ℕ),
    grace_total - (bella_fraction * grace_total).num = apples_left ∧
    bella_daily_apples * (7 * weeks) = (bella_fraction * grace_total).num :=
by sorry

end NUMINAMATH_CALUDE_bella_apples_per_day_l1557_155785


namespace NUMINAMATH_CALUDE_green_ball_probability_l1557_155701

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers in the problem -/
def containerX : Container := ⟨5, 5⟩
def containerY : Container := ⟨7, 3⟩
def containerZ : Container := ⟨7, 3⟩

/-- The list of all containers -/
def containers : List Container := [containerX, containerY, containerZ]

/-- The probability of selecting each container -/
def containerProbability : ℚ := 1 / containers.length

/-- The theorem stating the probability of selecting a green ball -/
theorem green_ball_probability :
  (containers.map (fun c => containerProbability * greenProbability c)).sum = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l1557_155701

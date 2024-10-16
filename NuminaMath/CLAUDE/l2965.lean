import Mathlib

namespace NUMINAMATH_CALUDE_percentage_english_books_published_in_country_l2965_296581

/-- Given a library with the following properties:
  * There are 2300 total books
  * 80% of all books are in English
  * 736 English books were published outside the country
  Prove that approximately 59.78% of English books were published in the country -/
theorem percentage_english_books_published_in_country 
  (total_books : ℕ) 
  (english_percentage : ℚ)
  (english_books_outside : ℕ) :
  total_books = 2300 →
  english_percentage = 4/5 →
  english_books_outside = 736 →
  ∃ (p : ℚ), abs (p - 59.78/100) < 1/1000 ∧ 
    p = (↑total_books * english_percentage - ↑english_books_outside) / 
        (↑total_books * english_percentage) := by
  sorry

end NUMINAMATH_CALUDE_percentage_english_books_published_in_country_l2965_296581


namespace NUMINAMATH_CALUDE_workday_meetings_percentage_l2965_296565

def workday_hours : ℕ := 10
def minutes_per_hour : ℕ := 60
def first_meeting_duration : ℕ := 60
def second_meeting_duration : ℕ := 2 * first_meeting_duration
def third_meeting_duration : ℕ := first_meeting_duration / 2

def total_workday_minutes : ℕ := workday_hours * minutes_per_hour
def total_meeting_minutes : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

theorem workday_meetings_percentage :
  (total_meeting_minutes : ℚ) / (total_workday_minutes : ℚ) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_workday_meetings_percentage_l2965_296565


namespace NUMINAMATH_CALUDE_range_of_a_l2965_296516

theorem range_of_a (a : ℝ) : 
  (a < 0) →
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 → x^2 + 2*x - 8 > 0) →
  (∃ x : ℝ, x^2 + 2*x - 8 > 0 ∧ x^2 - 4*a*x + 3*a^2 ≥ 0) →
  a ≤ -4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2965_296516


namespace NUMINAMATH_CALUDE_product_of_roots_is_4y_squared_l2965_296529

-- Define a quadratic function f
variable (f : ℝ → ℝ)
variable (y : ℝ)

-- Assumptions
axiom f_is_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
axiom root_of_f_x_minus_y : f (2*y - y) = 0
axiom root_of_f_x_plus_y : f (3*y + y) = 0

-- Theorem statement
theorem product_of_roots_is_4y_squared :
  (∃ (r₁ r₂ : ℝ), ∀ x, f x = 0 ↔ (x = r₁ ∨ x = r₂)) →
  (∃ (r₁ r₂ : ℝ), (∀ x, f x = 0 ↔ (x = r₁ ∨ x = r₂)) ∧ r₁ * r₂ = 4 * y^2) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_is_4y_squared_l2965_296529


namespace NUMINAMATH_CALUDE_total_tickets_correct_l2965_296585

/-- The total number of tickets sold at University Theater -/
def total_tickets : ℕ := 510

/-- The price of an adult ticket -/
def adult_price : ℕ := 21

/-- The price of a senior citizen ticket -/
def senior_price : ℕ := 15

/-- The number of senior citizen tickets sold -/
def senior_tickets : ℕ := 327

/-- The total receipts from ticket sales -/
def total_receipts : ℕ := 8748

/-- Theorem stating that the total number of tickets sold is correct -/
theorem total_tickets_correct :
  ∃ (adult_tickets : ℕ),
    total_tickets = adult_tickets + senior_tickets ∧
    total_receipts = adult_tickets * adult_price + senior_tickets * senior_price :=
by sorry

end NUMINAMATH_CALUDE_total_tickets_correct_l2965_296585


namespace NUMINAMATH_CALUDE_unique_solution_l2965_296555

/-- Represents the ages of three brothers -/
structure BrothersAges where
  older : ℕ
  xiaoyong : ℕ
  younger : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : BrothersAges) : Prop :=
  ages.older = 20 ∧
  ages.older > ages.xiaoyong ∧
  ages.xiaoyong > ages.younger ∧
  ages.younger ≥ 1 ∧
  2 * ages.xiaoyong + 5 * ages.younger = 97

/-- The theorem stating the unique solution to the problem -/
theorem unique_solution :
  ∃! ages : BrothersAges, satisfiesConditions ages ∧ ages.xiaoyong = 16 ∧ ages.younger = 13 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2965_296555


namespace NUMINAMATH_CALUDE_train_length_l2965_296566

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 6 → speed * time * (1000 / 3600) = 150 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2965_296566


namespace NUMINAMATH_CALUDE_integer_pairs_equation_difficulty_l2965_296544

theorem integer_pairs_equation_difficulty : ¬ ∃ (count : ℕ), 
  (∀ m n : ℤ, m^2 + n^2 = m*n + 3 → count > 0) ∧ 
  (∀ k : ℕ, k ≠ count → ¬(∀ m n : ℤ, m^2 + n^2 = m*n + 3 → k > 0)) :=
sorry

end NUMINAMATH_CALUDE_integer_pairs_equation_difficulty_l2965_296544


namespace NUMINAMATH_CALUDE_negative_a_squared_times_a_cubed_l2965_296571

theorem negative_a_squared_times_a_cubed (a : ℝ) : (-a)^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_squared_times_a_cubed_l2965_296571


namespace NUMINAMATH_CALUDE_ship_blown_westward_distance_l2965_296556

/-- Represents the ship's journey with given conditions -/
structure ShipJourney where
  travelTime : ℝ
  speed : ℝ
  obstaclePercentage : ℝ
  finalFraction : ℝ

/-- Calculates the distance blown westward by the storm -/
def distanceBlownWestward (journey : ShipJourney) : ℝ :=
  let plannedDistance := journey.travelTime * journey.speed
  let actualDistance := plannedDistance * (1 + journey.obstaclePercentage)
  let totalDistance := 2 * actualDistance
  let finalDistance := journey.finalFraction * totalDistance
  actualDistance - finalDistance

/-- Theorem stating that for the given journey conditions, the ship was blown 230 km westward -/
theorem ship_blown_westward_distance :
  let journey : ShipJourney := {
    travelTime := 20,
    speed := 30,
    obstaclePercentage := 0.15,
    finalFraction := 1/3
  }
  distanceBlownWestward journey = 230 := by sorry

end NUMINAMATH_CALUDE_ship_blown_westward_distance_l2965_296556


namespace NUMINAMATH_CALUDE_circular_arrangement_exists_l2965_296582

theorem circular_arrangement_exists : ∃ (a : Fin 12 → Fin 12), Function.Bijective a ∧
  ∀ (i j : Fin 12), i ≠ j → |a i - a j| ≠ |i.val - j.val| := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_exists_l2965_296582


namespace NUMINAMATH_CALUDE_triangle_max_area_l2965_296531

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  (2 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C →
  ∀ (area : ℝ), area = (1/2) * b * c * Real.sin A → area ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2965_296531


namespace NUMINAMATH_CALUDE_propositions_truth_l2965_296520

-- Proposition ①
def proposition_1 : Prop := 
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 ≥ 0)

-- Proposition ②
def proposition_2 : Prop := 
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ > 0) ↔ (∀ x : ℝ, x^2 - x < 0)

-- Proposition ③
def proposition_3 : Prop := 
  ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - 2*x ≤ 3

-- Proposition ④
def proposition_4 : Prop := 
  ∃ x₀ : ℝ, x₀^2 + 1/(x₀^2 + 1) ≤ 1

theorem propositions_truth : 
  ¬ proposition_1 ∧ 
  ¬ proposition_2 ∧ 
  proposition_3 ∧ 
  proposition_4 := by sorry

end NUMINAMATH_CALUDE_propositions_truth_l2965_296520


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l2965_296560

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (θ : ℝ) 
  (ha : a = 8) 
  (hb : b = 15) 
  (hθ : θ = 30 * π / 180) :
  c = Real.sqrt (289 - 120 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l2965_296560


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2965_296521

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2965_296521


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_root_l2965_296567

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, 2007 * x^3 + 2006 * x^2 + 2005 * x = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_root_l2965_296567


namespace NUMINAMATH_CALUDE_gas_volume_at_20C_l2965_296599

/-- Represents the volume of a gas at a given temperature -/
structure GasVolume where
  temp : ℝ  -- temperature in Celsius
  vol : ℝ   -- volume in cubic centimeters

/-- Represents the relationship between temperature change and volume change -/
structure VolumeChange where
  temp_change : ℝ  -- temperature change in Celsius
  vol_change : ℝ   -- volume change in cubic centimeters

theorem gas_volume_at_20C 
  (initial : GasVolume)
  (change : VolumeChange)
  (h1 : initial.temp = 30)
  (h2 : initial.vol = 36)
  (h3 : change.temp_change = 2)
  (h4 : change.vol_change = 3) :
  ∃ (final : GasVolume), 
    final.temp = 20 ∧ 
    final.vol = 21 :=
sorry

end NUMINAMATH_CALUDE_gas_volume_at_20C_l2965_296599


namespace NUMINAMATH_CALUDE_solve_equations_l2965_296523

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 8*x - 6 = 0
def equation2 (x : ℝ) : Prop := (x - 3)^2 + 2*x*(x - 3) = 0

-- State the theorem
theorem solve_equations :
  (∃ x₁ x₂ : ℝ, x₁ = 4 + Real.sqrt 22 ∧ x₂ = 4 - Real.sqrt 22 ∧ equation1 x₁ ∧ equation1 x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧ equation2 x₁ ∧ equation2 x₂) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l2965_296523


namespace NUMINAMATH_CALUDE_john_total_distance_l2965_296527

/-- Calculates the total distance cycled given a constant speed and total cycling time -/
def total_distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: Given John's cycling conditions, he cycles 18 miles in total -/
theorem john_total_distance :
  let speed : ℝ := 6  -- miles per hour
  let time_before_rest : ℝ := 2  -- hours
  let time_after_rest : ℝ := 1  -- hour
  let total_time : ℝ := time_before_rest + time_after_rest
  total_distance speed total_time = 18 := by
  sorry

#check john_total_distance

end NUMINAMATH_CALUDE_john_total_distance_l2965_296527


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l2965_296511

theorem two_digit_number_problem :
  ∃! n : ℕ, 
    10 ≤ n ∧ n < 100 ∧  -- n is a two-digit number
    (n / 10 : ℕ) = (n % 10)^2 - 9 ∧  -- tens digit is 9 less than square of ones digit
    10 * (n % 10) + (n / 10) = n - 27  -- swapped digits result in 27 less than original
  := by sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l2965_296511


namespace NUMINAMATH_CALUDE_range_of_a_l2965_296518

theorem range_of_a (a : ℝ) : 
  (∀ t : ℝ, t^2 - a*t - a ≥ 0) → -4 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2965_296518


namespace NUMINAMATH_CALUDE_direction_vector_implies_a_eq_plus_minus_two_l2965_296525

/-- Two lines with equations ax + 2y + 3 = 0 and 2x + ay - 1 = 0 have the same direction vector -/
def same_direction_vector (a : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ k * a = 2 ∧ k * 2 = a

theorem direction_vector_implies_a_eq_plus_minus_two (a : ℝ) :
  same_direction_vector a → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_direction_vector_implies_a_eq_plus_minus_two_l2965_296525


namespace NUMINAMATH_CALUDE_solve_for_A_l2965_296545

theorem solve_for_A : ∀ A : ℤ, A + 10 = 15 → A = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l2965_296545


namespace NUMINAMATH_CALUDE_range_of_a_l2965_296589

-- Define the propositions p and q
def p (x a : ℝ) : Prop := -4 < x - a ∧ x - a < 4
def q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, q x → p x a) →  -- q is a sufficient condition for p
  -1 ≤ a ∧ a ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2965_296589


namespace NUMINAMATH_CALUDE_shipping_weight_calculation_l2965_296580

/-- The maximum weight a shipping box can hold in pounds, given the initial number of plates,
    weight of each plate, and number of plates removed. -/
def max_shipping_weight (initial_plates : ℕ) (plate_weight : ℕ) (removed_plates : ℕ) : ℚ :=
  ((initial_plates - removed_plates) * plate_weight : ℚ) / 16

theorem shipping_weight_calculation :
  max_shipping_weight 38 10 6 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shipping_weight_calculation_l2965_296580


namespace NUMINAMATH_CALUDE_max_value_expression_l2965_296574

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26*a*b*c) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2965_296574


namespace NUMINAMATH_CALUDE_total_water_volume_l2965_296578

def water_volume (num_containers : ℕ) (container_volume : ℝ) : ℝ :=
  (num_containers : ℝ) * container_volume

theorem total_water_volume : 
  water_volume 2812 4 = 11248 := by sorry

end NUMINAMATH_CALUDE_total_water_volume_l2965_296578


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2965_296588

theorem polynomial_evaluation :
  ∀ x : ℝ, 
    x > 0 → 
    x^2 - 3*x - 10 = 0 → 
    x^3 - 3*x^2 - 9*x + 5 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2965_296588


namespace NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l2965_296590

theorem floor_plus_x_eq_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17 / 4 ∧ x = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l2965_296590


namespace NUMINAMATH_CALUDE_complex_roots_to_real_pair_l2965_296532

theorem complex_roots_to_real_pair :
  ∀ (a b : ℝ),
  (Complex.I : ℂ) ^ 2 = -1 →
  (a + 3 * Complex.I) * (a + 3 * Complex.I) - (12 + 15 * Complex.I) * (a + 3 * Complex.I) + (50 + 29 * Complex.I) = 0 →
  (b + 6 * Complex.I) * (b + 6 * Complex.I) - (12 + 15 * Complex.I) * (b + 6 * Complex.I) + (50 + 29 * Complex.I) = 0 →
  a = 5 / 3 ∧ b = 31 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_to_real_pair_l2965_296532


namespace NUMINAMATH_CALUDE_green_silk_calculation_l2965_296526

/-- The number of yards of silk dyed for an order -/
def total_yards : ℕ := 111421

/-- The number of yards of silk dyed pink -/
def pink_yards : ℕ := 49500

/-- The number of yards of silk dyed green -/
def green_yards : ℕ := total_yards - pink_yards

theorem green_silk_calculation : green_yards = 61921 := by
  sorry

end NUMINAMATH_CALUDE_green_silk_calculation_l2965_296526


namespace NUMINAMATH_CALUDE_min_value_inequality_l2965_296597

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + 2 * y + 6) :
  1 / x + 1 / (2 * y) ≥ 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2965_296597


namespace NUMINAMATH_CALUDE_circle_symmetry_l2965_296543

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 1 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x-2)^2 + (y+3)^2 = 2

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧
    (x - x₀ = x₀ - 2) ∧ (y - y₀ = y₀ + 3) ∧
    symmetry_line ((x + x₀) / 2) ((y + y₀) / 2)) →
  symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2965_296543


namespace NUMINAMATH_CALUDE_binomial_coeff_not_arithmetic_seq_l2965_296577

theorem binomial_coeff_not_arithmetic_seq (n r : ℕ) (h1 : n ≥ r + 3) (h2 : r > 0) :
  ¬ (∃ d : ℚ, Nat.choose n r + d = Nat.choose n (r + 1) ∧
               Nat.choose n (r + 1) + d = Nat.choose n (r + 2) ∧
               Nat.choose n (r + 2) + d = Nat.choose n (r + 3)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coeff_not_arithmetic_seq_l2965_296577


namespace NUMINAMATH_CALUDE_inequality_proof_l2965_296513

theorem inequality_proof (a b : ℝ) : 
  a^2 + b^2 ≥ 2*(a + b - 1) ∧ 
  (a > 0 ∧ b > 0 ∧ a + b = 3 → 1/a + 4/(b+1) ≥ 9/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2965_296513


namespace NUMINAMATH_CALUDE_yearly_music_expenditure_l2965_296593

def hours_per_month : ℕ := 20
def minutes_per_song : ℕ := 3
def price_per_song : ℚ := 1/2
def months_per_year : ℕ := 12

def yearly_music_cost : ℚ :=
  (hours_per_month * 60 / minutes_per_song) * price_per_song * months_per_year

theorem yearly_music_expenditure :
  yearly_music_cost = 2400 := by
  sorry

end NUMINAMATH_CALUDE_yearly_music_expenditure_l2965_296593


namespace NUMINAMATH_CALUDE_magnitude_a_plus_b_unique_k_parallel_l2965_296584

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- The magnitude of the sum of vectors a and b is 5 -/
theorem magnitude_a_plus_b : ‖a + b‖ = 5 := by sorry

/-- The unique value of k such that a + k*c is parallel to 2*a - b is 3 -/
theorem unique_k_parallel : ∃! k : ℝ, ∃ t : ℝ, a + k • c = t • (2 • a - b) ∧ k = 3 := by sorry

end NUMINAMATH_CALUDE_magnitude_a_plus_b_unique_k_parallel_l2965_296584


namespace NUMINAMATH_CALUDE_plane_perpendicular_condition_l2965_296536

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perp : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_condition 
  (a : Line) (α β : Plane) :
  perpendicular a β ∧ parallel a α → perp α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_condition_l2965_296536


namespace NUMINAMATH_CALUDE_percent_of_a_is_3b_l2965_296501

theorem percent_of_a_is_3b (a b : ℝ) (h : a = 1.5 * b) : (3 * b) / a * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_is_3b_l2965_296501


namespace NUMINAMATH_CALUDE_function_existence_l2965_296587

theorem function_existence : ∃ (f : ℤ → ℤ), ∀ (k : ℕ) (m : ℤ), k ≤ 1996 → ∃ (x : ℤ), f x + k * x = m := by
  sorry

end NUMINAMATH_CALUDE_function_existence_l2965_296587


namespace NUMINAMATH_CALUDE_conic_sections_identification_l2965_296514

/-- The equation y^4 - 9x^4 = 3y^2 - 3 represents the union of a hyperbola and an ellipse -/
theorem conic_sections_identification (x y : ℝ) : 
  (y^4 - 9*x^4 = 3*y^2 - 3) ↔ 
  ((y^2 - 3*x^2 = 3/2) ∨ (y^2 + 3*x^2 = 3/2)) :=
sorry

end NUMINAMATH_CALUDE_conic_sections_identification_l2965_296514


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l2965_296537

theorem coefficient_of_x_cubed (x : ℝ) : 
  let expression := 2 * (x^3 - 2*x^2 + x) + 4 * (x^4 + 3*x^3 - x^2 + x) - 3 * (x - 5*x^3 + 2*x^5)
  ∃ (a b c d e : ℝ), expression = a*x^5 + b*x^4 + 29*x^3 + c*x^2 + d*x + e :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l2965_296537


namespace NUMINAMATH_CALUDE_money_left_after_gift_l2965_296535

def gift_package_cost : ℚ := 445
def erika_savings : ℚ := 155
def sam_savings : ℚ := 175
def cake_flowers_skincare_cost : ℚ := 25 + 35 + 45

def rick_savings : ℚ := gift_package_cost / 2
def amy_savings : ℚ := 2 * cake_flowers_skincare_cost

def total_savings : ℚ := erika_savings + rick_savings + sam_savings + amy_savings

theorem money_left_after_gift (h : total_savings - gift_package_cost = 317.5) :
  total_savings - gift_package_cost = 317.5 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_gift_l2965_296535


namespace NUMINAMATH_CALUDE_monomial_sum_l2965_296572

/-- Given constants a and b, if the sum of 4xy^2, axy^b, and -5xy is a monomial, 
    then a+b = -2 or a+b = 6 -/
theorem monomial_sum (a b : ℝ) : 
  (∃ (x y : ℝ), ∀ (z : ℝ), z = 4*x*y^2 + a*x*y^b - 5*x*y → ∃ (c : ℝ), z = c*x*y^k) → 
  a + b = -2 ∨ a + b = 6 :=
sorry

end NUMINAMATH_CALUDE_monomial_sum_l2965_296572


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2965_296530

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2965_296530


namespace NUMINAMATH_CALUDE_group_size_proof_l2965_296540

theorem group_size_proof (W : ℝ) (n : ℕ) : 
  (W + 15) / n = W / n + 2.5 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l2965_296540


namespace NUMINAMATH_CALUDE_booklet_cost_l2965_296592

theorem booklet_cost (b : ℝ) : 
  (10 * b < 15) → (12 * b > 17) → b = 1.42 := by
  sorry

end NUMINAMATH_CALUDE_booklet_cost_l2965_296592


namespace NUMINAMATH_CALUDE_product_plus_one_square_l2965_296528

theorem product_plus_one_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_square_l2965_296528


namespace NUMINAMATH_CALUDE_discriminant_of_quadratic_poly_l2965_296595

/-- The discriminant of a quadratic polynomial ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The quadratic polynomial 3x² + (3 + 1/3)x + 1/3 -/
def quadratic_poly (x : ℚ) : ℚ := 3*x^2 + (3 + 1/3)*x + 1/3

theorem discriminant_of_quadratic_poly :
  discriminant 3 (3 + 1/3) (1/3) = 64/9 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_quadratic_poly_l2965_296595


namespace NUMINAMATH_CALUDE_rightmost_two_digits_l2965_296506

theorem rightmost_two_digits : ∃ n : ℕ, (4^127 + 5^129 + 7^131) % 100 = 52 + 100 * n := by
  sorry

end NUMINAMATH_CALUDE_rightmost_two_digits_l2965_296506


namespace NUMINAMATH_CALUDE_max_price_of_roses_and_peonies_l2965_296507

-- Define the price of a rose and a peony
variable (R P : ℝ)

-- Define the conditions
def condition1 : Prop := 4 * R + 5 * P ≥ 27
def condition2 : Prop := 6 * R + 3 * P ≤ 27

-- Define the objective function
def objective : ℝ := 3 * R + 4 * P

-- Theorem statement
theorem max_price_of_roses_and_peonies :
  condition1 R P → condition2 R P → ∃ (max : ℝ), max = 36 ∧ objective R P ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_price_of_roses_and_peonies_l2965_296507


namespace NUMINAMATH_CALUDE_paint_six_boards_time_l2965_296563

/-- The minimum time required to paint both sides of wooden boards. -/
def paint_time (num_boards : ℕ) (paint_time_per_side : ℕ) (drying_time : ℕ) : ℕ :=
  2 * num_boards * paint_time_per_side

theorem paint_six_boards_time :
  paint_time 6 1 5 = 12 :=
by sorry

end NUMINAMATH_CALUDE_paint_six_boards_time_l2965_296563


namespace NUMINAMATH_CALUDE_g_of_3_equals_209_l2965_296594

-- Define the function g
def g (x : ℝ) : ℝ := 9 * x^3 - 4 * x^2 + 3 * x - 7

-- Theorem statement
theorem g_of_3_equals_209 : g 3 = 209 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_209_l2965_296594


namespace NUMINAMATH_CALUDE_reconstruct_triangle_l2965_296505

-- Define the types for points and triangles
def Point : Type := ℝ × ℝ
def Triangle : Type := Point × Point × Point

-- Define the external angle bisector
def externalAngleBisector (A B C : Point) : Point → Prop := sorry

-- Define the perpendicular from a point to a line
def perpendicularFoot (P A B : Point) : Point := sorry

-- Define the statement
theorem reconstruct_triangle (A' B' C' : Point) :
  ∃ (A B C : Point),
    -- A'B'C' is formed by external angle bisectors of ABC
    externalAngleBisector B C A A' ∧
    externalAngleBisector A C B B' ∧
    externalAngleBisector A B C C' ∧
    -- A, B, C are feet of perpendiculars from A', B', C' to opposite sides of A'B'C'
    A = perpendicularFoot A' B' C' ∧
    B = perpendicularFoot B' A' C' ∧
    C = perpendicularFoot C' A' B' :=
by
  sorry

end NUMINAMATH_CALUDE_reconstruct_triangle_l2965_296505


namespace NUMINAMATH_CALUDE_weight_probability_l2965_296504

/-- The probability that the weight of five eggs is less than 30 grams -/
def prob_less_than_30 : ℝ := 0.3

/-- The probability that the weight of five eggs is between [30, 40] grams -/
def prob_between_30_and_40 : ℝ := 0.5

/-- The probability that the weight of five eggs does not exceed 40 grams -/
def prob_not_exceed_40 : ℝ := prob_less_than_30 + prob_between_30_and_40

theorem weight_probability : prob_not_exceed_40 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_weight_probability_l2965_296504


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l2965_296551

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 8| + |x - 7|

-- Define the interval [3, 9]
def I : Set ℝ := {x | 3 ≤ x ∧ x ≤ 9}

-- State the theorem
theorem sum_of_max_min_g : 
  ∃ (max_val min_val : ℝ),
    (∀ x ∈ I, g x ≤ max_val) ∧
    (∃ x ∈ I, g x = max_val) ∧
    (∀ x ∈ I, min_val ≤ g x) ∧
    (∃ x ∈ I, g x = min_val) ∧
    max_val + min_val = 14 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l2965_296551


namespace NUMINAMATH_CALUDE_min_value_theorem_l2965_296533

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 1/b = 2) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + 1/y = 2 → 2*x*y + 1/x ≥ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2965_296533


namespace NUMINAMATH_CALUDE_sqrt_of_square_root_three_plus_one_squared_l2965_296564

theorem sqrt_of_square_root_three_plus_one_squared :
  Real.sqrt ((Real.sqrt 3 + 1) ^ 2) = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_square_root_three_plus_one_squared_l2965_296564


namespace NUMINAMATH_CALUDE_ellipse_properties_l2965_296502

-- Define the ellipse and its properties
def Ellipse (a b c : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ c > 0 ∧ a^2 = b^2 + c^2

-- Define the points and conditions
def EllipseConditions (a b c : ℝ) (A B E F₁ F₂ : ℝ × ℝ) : Prop :=
  Ellipse a b c ∧
  F₁ = (-c, 0) ∧
  F₂ = (c, 0) ∧
  E = (a^2/c, 0) ∧
  (∃ k : ℝ, A.2 = k * (A.1 - a^2/c) ∧ B.2 = k * (B.1 - a^2/c)) ∧
  (∃ t : ℝ, A.1 - F₁.1 = t * (B.1 - F₂.1) ∧ A.2 - F₁.2 = t * (B.2 - F₂.2)) ∧
  (A.1 - F₁.1)^2 + (A.2 - F₁.2)^2 = 4 * ((B.1 - F₂.1)^2 + (B.2 - F₂.2)^2)

-- Theorem statement
theorem ellipse_properties (a b c : ℝ) (A B E F₁ F₂ : ℝ × ℝ) 
  (h : EllipseConditions a b c A B E F₁ F₂) :
  c / a = Real.sqrt 3 / 3 ∧ 
  (∃ k : ℝ, (A.2 - B.2) / (A.1 - B.1) = k ∧ (k = Real.sqrt 2 / 3 ∨ k = -Real.sqrt 2 / 3)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2965_296502


namespace NUMINAMATH_CALUDE_max_min_a_plus_b_l2965_296553

noncomputable def f (x : ℝ) : ℝ := x - Real.sqrt x

theorem max_min_a_plus_b (a b : ℝ) (h : f (a + 1) + f (b + 2) = 3) :
  (a + b ≤ 1 + Real.sqrt 7) ∧ (a + b ≥ (1 + Real.sqrt 13) / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_min_a_plus_b_l2965_296553


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2965_296575

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a + i) * i = b + i) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2965_296575


namespace NUMINAMATH_CALUDE_no_three_consecutive_digit_sum_squares_l2965_296538

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- Theorem: There do not exist three consecutive integers such that 
    the sum of digits of each is a perfect square -/
theorem no_three_consecutive_digit_sum_squares :
  ¬ ∃ n : ℕ, (is_perfect_square (sum_of_digits n)) ∧ 
             (is_perfect_square (sum_of_digits (n + 1))) ∧ 
             (is_perfect_square (sum_of_digits (n + 2))) :=
sorry

end NUMINAMATH_CALUDE_no_three_consecutive_digit_sum_squares_l2965_296538


namespace NUMINAMATH_CALUDE_square_number_correct_l2965_296569

/-- The number in the square with coordinates (m, n) -/
def square_number (m n : ℕ) : ℕ :=
  ((m + n - 2) * (m + n - 1)) / 2 + n

/-- Theorem: The square_number function correctly calculates the number
    in the square with coordinates (m, n) for positive integers m and n -/
theorem square_number_correct (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  square_number m n = ((m + n - 2) * (m + n - 1)) / 2 + n :=
by sorry

end NUMINAMATH_CALUDE_square_number_correct_l2965_296569


namespace NUMINAMATH_CALUDE_odd_k_triple_f_81_l2965_296524

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then 2 * n + 3
  else if n % 3 = 0 ∧ n % 2 ≠ 0 then n / 3
  else n  -- This case is not specified in the original problem, so we leave n unchanged

theorem odd_k_triple_f_81 (k : ℤ) (h_odd : k % 2 = 1) (h_triple_f : f (f (f k)) = 81) : k = 57 := by
  sorry

end NUMINAMATH_CALUDE_odd_k_triple_f_81_l2965_296524


namespace NUMINAMATH_CALUDE_liu_data_correct_l2965_296559

/-- Represents the agricultural data for the Li and Liu families -/
structure FamilyData where
  li_land : ℕ
  li_yield : ℕ
  liu_land_diff : ℕ
  total_production : ℕ

/-- Calculates the Liu family's total production and yield difference -/
def calculate_liu_data (data : FamilyData) : ℕ × ℕ :=
  let liu_land := data.li_land - data.liu_land_diff
  let liu_production := data.total_production
  let liu_yield := liu_production / liu_land
  let li_yield := data.li_yield
  let yield_diff := liu_yield - li_yield
  (liu_production, yield_diff)

/-- Theorem stating the correctness of the calculation -/
theorem liu_data_correct (data : FamilyData) 
  (h1 : data.li_land = 100)
  (h2 : data.li_yield = 600)
  (h3 : data.liu_land_diff = 20)
  (h4 : data.total_production = data.li_land * data.li_yield) :
  calculate_liu_data data = (6000, 15) := by
  sorry

#eval calculate_liu_data ⟨100, 600, 20, 60000⟩

end NUMINAMATH_CALUDE_liu_data_correct_l2965_296559


namespace NUMINAMATH_CALUDE_intersection_M_N_l2965_296568

-- Define set M
def M : Set ℝ := {x | x^2 + 2*x - 8 < 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2965_296568


namespace NUMINAMATH_CALUDE_function_k_value_l2965_296554

theorem function_k_value (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, f x = k * x + 1) →
  f 2 = 3 →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_function_k_value_l2965_296554


namespace NUMINAMATH_CALUDE_union_of_sets_l2965_296562

theorem union_of_sets : 
  let A : Set ℤ := {1, 2, 3}
  let B : Set ℤ := {-1, 1}
  A ∪ B = {-1, 1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2965_296562


namespace NUMINAMATH_CALUDE_solution_properties_l2965_296509

def is_valid_solution (x y z : ℕ+) : Prop :=
  x.val + y.val + z.val = 2013

def count_solutions : ℕ := sorry

def count_solutions_x_eq_y : ℕ := sorry

def max_product_solution : ℕ+ × ℕ+ × ℕ+ := sorry

theorem solution_properties :
  (count_solutions = 2023066) ∧
  (count_solutions_x_eq_y = 1006) ∧
  (max_product_solution = (⟨671, sorry⟩, ⟨671, sorry⟩, ⟨671, sorry⟩)) :=
by sorry

end NUMINAMATH_CALUDE_solution_properties_l2965_296509


namespace NUMINAMATH_CALUDE_sum_greater_than_two_l2965_296557

theorem sum_greater_than_two (x y : ℝ) 
  (h1 : x^7 > y^6) 
  (h2 : y^7 > x^6) : 
  x + y > 2 := by
sorry

end NUMINAMATH_CALUDE_sum_greater_than_two_l2965_296557


namespace NUMINAMATH_CALUDE_part_one_part_two_combined_theorem_l2965_296579

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (a m : ℝ) :
  (∀ x, f a x ≤ m ↔ -1 ≤ x ∧ x ≤ 5) →
  a = 2 ∧ m = 3 := by sorry

-- Part II
theorem part_two (t : ℝ) (h : 0 ≤ t ∧ t < 2) :
  {x : ℝ | f 2 x + t ≥ f 2 (x + 2)} = Set.Iic ((t + 2) / 2) := by sorry

-- Combined theorem
theorem combined_theorem (a m t : ℝ) (h : 0 ≤ t ∧ t < 2) :
  (∀ x, f a x ≤ m ↔ -1 ≤ x ∧ x ≤ 5) →
  (a = 2 ∧ m = 3) ∧
  {x : ℝ | f 2 x + t ≥ f 2 (x + 2)} = Set.Iic ((t + 2) / 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_combined_theorem_l2965_296579


namespace NUMINAMATH_CALUDE_dealership_sedan_sales_l2965_296510

/-- Represents the ratio of sports cars to sedans -/
structure CarRatio :=
  (sports : ℕ)
  (sedans : ℕ)

/-- Calculates the expected sedan sales given a car ratio and anticipated sports car sales -/
def expectedSedanSales (ratio : CarRatio) (anticipatedSportsCars : ℕ) : ℕ :=
  (anticipatedSportsCars * ratio.sedans) / ratio.sports

theorem dealership_sedan_sales :
  let ratio : CarRatio := ⟨3, 5⟩
  let anticipatedSportsCars : ℕ := 36
  expectedSedanSales ratio anticipatedSportsCars = 60 := by
  sorry

end NUMINAMATH_CALUDE_dealership_sedan_sales_l2965_296510


namespace NUMINAMATH_CALUDE_prob_two_rolls_eq_one_sixty_fourth_l2965_296517

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The desired sum on each roll -/
def target_sum : ℕ := 9

/-- The set of all possible outcomes when rolling two dice -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range num_sides) (Finset.range num_sides)

/-- The set of outcomes that sum to the target -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun (a, b) => a + b + 2 = target_sum)

/-- The probability of rolling the target sum once -/
def prob_single_roll : ℚ :=
  (favorable_outcomes.card : ℚ) / (all_outcomes.card : ℚ)

theorem prob_two_rolls_eq_one_sixty_fourth :
  prob_single_roll * prob_single_roll = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_rolls_eq_one_sixty_fourth_l2965_296517


namespace NUMINAMATH_CALUDE_sum_of_exterior_angles_of_triangle_l2965_296519

theorem sum_of_exterior_angles_of_triangle (α β γ : ℝ) (α' β' γ' : ℝ) : 
  α + β + γ = 180 →
  α + α' = 180 →
  β + β' = 180 →
  γ + γ' = 180 →
  α' + β' + γ' = 360 := by
sorry

end NUMINAMATH_CALUDE_sum_of_exterior_angles_of_triangle_l2965_296519


namespace NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_100_l2965_296547

theorem greatest_common_multiple_9_15_under_100 : ∃ n : ℕ, n = 90 ∧ 
  (∀ m : ℕ, m < 100 → m % 9 = 0 → m % 15 = 0 → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_100_l2965_296547


namespace NUMINAMATH_CALUDE_article_price_proof_l2965_296570

/-- The normal price of an article before discounts -/
def normal_price : ℝ := 100

/-- The final price after discounts -/
def final_price : ℝ := 72

/-- The first discount rate -/
def discount1 : ℝ := 0.1

/-- The second discount rate -/
def discount2 : ℝ := 0.2

theorem article_price_proof :
  normal_price * (1 - discount1) * (1 - discount2) = final_price :=
by sorry

end NUMINAMATH_CALUDE_article_price_proof_l2965_296570


namespace NUMINAMATH_CALUDE_marble_count_l2965_296576

/-- The number of marbles in Jar A -/
def jarA : ℕ := 56

/-- The number of marbles in Jar B -/
def jarB : ℕ := 3 * jarA / 2

/-- The number of marbles in Jar C -/
def jarC : ℕ := 2 * jarA

/-- The number of marbles in Jar D -/
def jarD : ℕ := 3 * jarC / 4

/-- The total number of marbles in all jars -/
def totalMarbles : ℕ := jarA + jarB + jarC + jarD

theorem marble_count : totalMarbles = 336 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_l2965_296576


namespace NUMINAMATH_CALUDE_bus_journey_stoppage_time_l2965_296561

/-- Calculates the total stoppage time for a bus journey with three stops -/
def total_stoppage_time (stop1 stop2 stop3 : ℕ) : ℕ :=
  stop1 + stop2 + stop3

/-- Theorem stating that the total stoppage time for the given stop durations is 23 minutes -/
theorem bus_journey_stoppage_time :
  total_stoppage_time 5 8 10 = 23 :=
by sorry

end NUMINAMATH_CALUDE_bus_journey_stoppage_time_l2965_296561


namespace NUMINAMATH_CALUDE_billy_coins_l2965_296558

/-- Given the number of piles of quarters and dimes, and the number of coins per pile,
    calculate the total number of coins. -/
def total_coins (quarter_piles dime_piles coins_per_pile : ℕ) : ℕ :=
  (quarter_piles + dime_piles) * coins_per_pile

/-- Theorem stating that with 2 piles of quarters, 3 piles of dimes, and 4 coins per pile,
    the total number of coins is 20. -/
theorem billy_coins : total_coins 2 3 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_billy_coins_l2965_296558


namespace NUMINAMATH_CALUDE_trig_identities_l2965_296552

theorem trig_identities (α : Real) (h : Real.tan α = 2) :
  (Real.sin (α - 3 * Real.pi) + Real.cos (Real.pi + α)) /
  (Real.sin (-α) - Real.cos (Real.pi + α)) = 3 ∧
  Real.cos α ^ 2 - 2 * Real.sin α * Real.cos α = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l2965_296552


namespace NUMINAMATH_CALUDE_trigonometric_fraction_bounds_l2965_296534

theorem trigonometric_fraction_bounds (x : ℝ) : 
  -1/3 ≤ (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ∧ 
  (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_fraction_bounds_l2965_296534


namespace NUMINAMATH_CALUDE_largest_fraction_l2965_296583

theorem largest_fraction : 
  (8 : ℚ) / 9 > 7 / 8 ∧ 
  (8 : ℚ) / 9 > 66 / 77 ∧ 
  (8 : ℚ) / 9 > 55 / 66 ∧ 
  (8 : ℚ) / 9 > 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l2965_296583


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2965_296539

theorem perfect_square_condition (n : ℕ) (h : n > 0) :
  (∃ k : ℤ, 2 + 2 * Real.sqrt (1 + 12 * n^2) = k) →
  ∃ m : ℕ, n = m^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2965_296539


namespace NUMINAMATH_CALUDE_kenny_basketball_time_l2965_296522

theorem kenny_basketball_time (x y z : ℝ) 
  (h1 : y = 2 * x) 
  (h2 : z = 2 * y) 
  (h3 : z = 40) : 
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_kenny_basketball_time_l2965_296522


namespace NUMINAMATH_CALUDE_distance_swum_back_l2965_296515

/-- Calculates the distance swum against the current given swimming speed, water speed, and time -/
def distance_against_current (swimming_speed water_speed : ℝ) (time : ℝ) : ℝ :=
  (swimming_speed - water_speed) * time

/-- Proves that the distance swum against the current is 8 km given the specified conditions -/
theorem distance_swum_back (swimming_speed water_speed time : ℝ) 
  (h1 : swimming_speed = 12)
  (h2 : water_speed = 10)
  (h3 : time = 4) :
  distance_against_current swimming_speed water_speed time = 8 := by
  sorry

#check distance_swum_back

end NUMINAMATH_CALUDE_distance_swum_back_l2965_296515


namespace NUMINAMATH_CALUDE_tree_height_calculation_l2965_296586

/-- Given the height and shadow length of a person and the shadow length of a tree,
    calculate the height of the tree using the principle of similar triangles. -/
theorem tree_height_calculation (person_height person_shadow tree_shadow : ℝ)
  (h_person_height : person_height = 1.6)
  (h_person_shadow : person_shadow = 0.8)
  (h_tree_shadow : tree_shadow = 4.8)
  (h_positive : person_height > 0 ∧ person_shadow > 0 ∧ tree_shadow > 0) :
  (person_height / person_shadow) * tree_shadow = 9.6 :=
by
  sorry

#check tree_height_calculation

end NUMINAMATH_CALUDE_tree_height_calculation_l2965_296586


namespace NUMINAMATH_CALUDE_carlos_gummy_worms_l2965_296596

/-- The number of gummy worms remaining after eating half for a given number of days -/
def gummy_worms_remaining (initial : ℕ) (days : ℕ) : ℕ :=
  initial / (2 ^ days)

/-- Theorem stating that Carlos has 4 gummy worms left after 4 days -/
theorem carlos_gummy_worms :
  gummy_worms_remaining 64 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_carlos_gummy_worms_l2965_296596


namespace NUMINAMATH_CALUDE_skew_lines_cannot_both_project_to_points_l2965_296591

/-- Two lines in 3D space are skew -/
def are_skew (l1 l2 : Line3) : Prop := sorry

/-- A line in 3D space -/
def Line3 : Type := sorry

/-- A plane in 3D space -/
def Plane3 : Type := sorry

/-- The projection of a line onto a plane -/
def project_line_to_plane (l : Line3) (p : Plane3) : Set Point := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_to_plane (l : Line3) (p : Plane3) : Prop := sorry

theorem skew_lines_cannot_both_project_to_points (a b : Line3) (α : Plane3) 
  (h_skew : are_skew a b) :
  ¬(∃ (pa pb : Point), project_line_to_plane a α = {pa} ∧ project_line_to_plane b α = {pb}) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_cannot_both_project_to_points_l2965_296591


namespace NUMINAMATH_CALUDE_gcf_three_digit_palindromes_l2965_296549

/-- A three-digit palindrome -/
def ThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 102 * a + 10 * b

/-- The greatest common factor of all three-digit palindromes is 1 -/
theorem gcf_three_digit_palindromes :
  ∃ (g : ℕ), g > 0 ∧ 
    (∀ n : ℕ, ThreeDigitPalindrome n → g ∣ n) ∧
    (∀ d : ℕ, d > 0 → (∀ n : ℕ, ThreeDigitPalindrome n → d ∣ n) → d ≤ g) ∧
    g = 1 :=
sorry

end NUMINAMATH_CALUDE_gcf_three_digit_palindromes_l2965_296549


namespace NUMINAMATH_CALUDE_fib_F15_units_digit_l2965_296542

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem fib_F15_units_digit :
  unitsDigit (fib (fib 15)) = 5 := by sorry

end NUMINAMATH_CALUDE_fib_F15_units_digit_l2965_296542


namespace NUMINAMATH_CALUDE_more_likely_same_l2965_296550

/-- Represents the number of crows on each tree -/
structure CrowCounts where
  white_birch : ℕ
  black_birch : ℕ
  white_oak : ℕ
  black_oak : ℕ

/-- Conditions from the problem -/
def valid_crow_counts (c : CrowCounts) : Prop :=
  c.white_birch > 0 ∧
  c.white_birch + c.black_birch = 50 ∧
  c.white_oak + c.black_oak = 50 ∧
  c.black_birch ≥ c.white_birch ∧
  c.black_oak ≥ c.white_oak - 1

/-- Probability of number of white crows on birch remaining the same -/
def prob_same (c : CrowCounts) : ℚ :=
  (c.black_birch * (c.black_oak + 1) + c.white_birch * (c.white_oak + 1)) / 2550

/-- Probability of number of white crows on birch changing -/
def prob_change (c : CrowCounts) : ℚ :=
  (c.black_birch * c.white_oak + c.white_birch * c.black_oak) / 2550

/-- Theorem stating that it's more likely for the number of white crows to remain the same -/
theorem more_likely_same (c : CrowCounts) (h : valid_crow_counts c) :
  prob_same c > prob_change c :=
sorry

end NUMINAMATH_CALUDE_more_likely_same_l2965_296550


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_f_max_on_interval_f_min_on_interval_l2965_296500

-- Define the function f(x) = -x^2 + 2x
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Theorem for monotonicity
theorem f_increasing_on_interval : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ < f x₂ := by sorry

-- Theorem for maximum value
theorem f_max_on_interval : 
  ∃ x : ℝ, x ∈ Set.Icc 0 5 ∧ f x = 1 ∧ ∀ y ∈ Set.Icc 0 5, f y ≤ f x := by sorry

-- Theorem for minimum value
theorem f_min_on_interval : 
  ∃ x : ℝ, x ∈ Set.Icc 0 5 ∧ f x = -15 ∧ ∀ y ∈ Set.Icc 0 5, f y ≥ f x := by sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_f_max_on_interval_f_min_on_interval_l2965_296500


namespace NUMINAMATH_CALUDE_smallest_m_for_distinct_roots_smallest_integer_m_for_distinct_roots_l2965_296512

theorem smallest_m_for_distinct_roots (m : ℤ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ - m = 0 ∧ x₂^2 - 2*x₂ - m = 0) ↔ m ≥ 0 :=
by sorry

theorem smallest_integer_m_for_distinct_roots : 
  ∃ m₀ : ℤ, m₀ ≥ 0 ∧ ∀ m : ℤ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ - m = 0 ∧ x₂^2 - 2*x₂ - m = 0) → m ≥ m₀ :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_distinct_roots_smallest_integer_m_for_distinct_roots_l2965_296512


namespace NUMINAMATH_CALUDE_even_function_inequality_l2965_296573

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f: ℝ → ℝ is increasing on [0, +∞) if
    for all x, y ∈ [0, +∞), x < y implies f(x) < f(y) -/
def IncreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem even_function_inequality (f : ℝ → ℝ) 
    (heven : EvenFunction f) (hinc : IncreasingOnNonnegative f) :
    f π > f (-2) ∧ f (-2) > f (-1) := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l2965_296573


namespace NUMINAMATH_CALUDE_floor_sum_example_l2965_296548

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l2965_296548


namespace NUMINAMATH_CALUDE_stratified_sampling_second_grade_l2965_296541

/-- Represents the number of students in each grade -/
structure GradeDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of students -/
def totalStudents (dist : GradeDistribution) : ℕ :=
  dist.first + dist.second + dist.third

/-- Calculates the number of students to be sampled from a specific grade -/
def sampleSize (dist : GradeDistribution) (totalSample : ℕ) (grade : ℕ) : ℕ :=
  match grade with
  | 1 => (dist.first * totalSample) / (totalStudents dist)
  | 2 => (dist.second * totalSample) / (totalStudents dist)
  | 3 => (dist.third * totalSample) / (totalStudents dist)
  | _ => 0

theorem stratified_sampling_second_grade 
  (dist : GradeDistribution) 
  (h1 : dist.first = 1200) 
  (h2 : dist.second = 900) 
  (h3 : dist.third = 1500) 
  (h4 : totalStudents dist = 3600) 
  (h5 : sampleSize dist 720 2 = 480) : 
  sampleSize dist 720 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_second_grade_l2965_296541


namespace NUMINAMATH_CALUDE_sequence_general_term_l2965_296503

/-- Given a sequence {aₙ} where the sequence of differences forms an arithmetic
    sequence with first term 1 and common difference 1, prove that the general
    term formula for {aₙ} is n(n+1)/2. -/
theorem sequence_general_term (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) - a n = n) →
  a 1 = 1 →
  ∀ n : ℕ, a n = n * (n + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2965_296503


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l2965_296598

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l2965_296598


namespace NUMINAMATH_CALUDE_angle_after_folding_is_60_degrees_l2965_296546

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  /-- The two equal sides of the triangle -/
  leg : ℝ
  /-- The hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- The condition that it's a right triangle -/
  right_angle : hypotenuse^2 = 2 * leg^2

/-- The angle between the legs after folding an isosceles right triangle along its height to the hypotenuse -/
def angle_after_folding (t : IsoscelesRightTriangle) : ℝ := sorry

/-- Theorem stating that the angle between the legs after folding is 60° -/
theorem angle_after_folding_is_60_degrees (t : IsoscelesRightTriangle) :
  angle_after_folding t = 60 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_angle_after_folding_is_60_degrees_l2965_296546


namespace NUMINAMATH_CALUDE_total_weight_is_410_l2965_296508

/-- The number of A4 sheets Jane has -/
def num_a4_sheets : ℕ := 28

/-- The number of A3 sheets Jane has -/
def num_a3_sheets : ℕ := 27

/-- The weight of a single A4 sheet in grams -/
def weight_a4_sheet : ℕ := 5

/-- The weight of a single A3 sheet in grams -/
def weight_a3_sheet : ℕ := 10

/-- The total weight of all drawing papers in grams -/
def total_weight : ℕ := num_a4_sheets * weight_a4_sheet + num_a3_sheets * weight_a3_sheet

theorem total_weight_is_410 : total_weight = 410 := by sorry

end NUMINAMATH_CALUDE_total_weight_is_410_l2965_296508

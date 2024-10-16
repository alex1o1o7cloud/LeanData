import Mathlib

namespace NUMINAMATH_CALUDE_correct_mean_after_error_fix_l756_75646

/-- Given a set of values with an incorrect mean due to a misrecorded value,
    calculate the correct mean after fixing the error. -/
theorem correct_mean_after_error_fix (n : ℕ) (incorrect_mean : ℚ) (wrong_value correct_value : ℚ) 
    (h1 : n = 30)
    (h2 : incorrect_mean = 140)
    (h3 : wrong_value = 135)
    (h4 : correct_value = 145) :
    let total_sum := n * incorrect_mean
    let difference := correct_value - wrong_value
    let corrected_sum := total_sum + difference
    corrected_sum / n = 140333 / 1000 := by
  sorry

#eval (140333 : ℚ) / 1000  -- To verify the result is indeed 140.333

end NUMINAMATH_CALUDE_correct_mean_after_error_fix_l756_75646


namespace NUMINAMATH_CALUDE_speech_arrangement_count_l756_75649

theorem speech_arrangement_count :
  let total_male : ℕ := 4
  let total_female : ℕ := 3
  let selected_male : ℕ := 3
  let selected_female : ℕ := 2
  let total_selected : ℕ := selected_male + selected_female

  (Nat.choose total_male selected_male) *
  (Nat.choose total_female selected_female) *
  (Nat.factorial selected_male) *
  (Nat.factorial (total_selected - 1)) = 864 :=
by sorry

end NUMINAMATH_CALUDE_speech_arrangement_count_l756_75649


namespace NUMINAMATH_CALUDE_distance_between_5th_and_25th_red_light_l756_75693

/-- Represents the color of a light -/
inductive LightColor
| Red
| Green

/-- Represents the pattern of lights -/
def lightPattern : List LightColor :=
  [LightColor.Red, LightColor.Red, LightColor.Red, LightColor.Green, LightColor.Green]

/-- The distance between adjacent lights in inches -/
def lightDistance : ℕ := 8

/-- The number of inches in a foot -/
def inchesPerFoot : ℕ := 12

/-- The position of a red light given its index -/
def redLightPosition (n : ℕ) : ℕ :=
  (n - 1) / 3 * lightPattern.length + (n - 1) % 3 + 1

/-- The distance between two red lights given their indices -/
def distanceBetweenRedLights (n m : ℕ) : ℕ :=
  (redLightPosition m - redLightPosition n) * lightDistance

/-- The theorem to be proved -/
theorem distance_between_5th_and_25th_red_light :
  distanceBetweenRedLights 5 25 / inchesPerFoot = 56 := by
  sorry


end NUMINAMATH_CALUDE_distance_between_5th_and_25th_red_light_l756_75693


namespace NUMINAMATH_CALUDE_cubic_equation_ratio_l756_75628

theorem cubic_equation_ratio (a b c d : ℝ) (h : a ≠ 0) : 
  (∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ (x = -2 ∨ x = 3 ∨ x = 4)) →
  c / d = -1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_ratio_l756_75628


namespace NUMINAMATH_CALUDE_polynomial_factorization_l756_75600

theorem polynomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 - 100*x^4 = (-10*x^2 + x + 3) * (10*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l756_75600


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l756_75660

/-- Given a system of equations, prove that the maximum value of a^2 + b^2 + c^2 + d^2 is 714 -/
theorem max_sum_of_squares (a b c d : ℝ) 
  (eq1 : a + b = 18)
  (eq2 : a * b + c + d = 95)
  (eq3 : a * d + b * c = 195)
  (eq4 : c * d = 120) :
  a^2 + b^2 + c^2 + d^2 ≤ 714 ∧ 
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 
    a₀ + b₀ = 18 ∧
    a₀ * b₀ + c₀ + d₀ = 95 ∧
    a₀ * d₀ + b₀ * c₀ = 195 ∧
    c₀ * d₀ = 120 ∧
    a₀^2 + b₀^2 + c₀^2 + d₀^2 = 714 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l756_75660


namespace NUMINAMATH_CALUDE_expression_evaluation_l756_75623

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := Real.sqrt 2
  (x + 2*y)^2 - x*(x + 4*y) + (1 - y)*(1 + y) = 7 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l756_75623


namespace NUMINAMATH_CALUDE_range_equivalence_l756_75682

/-- The function f(x) = -x³ + 3bx --/
def f (b : ℝ) (x : ℝ) : ℝ := -x^3 + 3*b*x

/-- The theorem stating the equivalence between the range of f and the value of b --/
theorem range_equivalence (b : ℝ) :
  (∀ y ∈ Set.range (f b), y ∈ Set.Icc 0 1) ∧
  (∀ y ∈ Set.Icc 0 1, ∃ x ∈ Set.Icc 0 1, f b x = y) ↔
  b = (2 : ℝ)^(1/3) / 2 := by
sorry

end NUMINAMATH_CALUDE_range_equivalence_l756_75682


namespace NUMINAMATH_CALUDE_rectangle_width_l756_75661

/-- Given a rectangle with length 18 cm and a largest inscribed circle with area 153.93804002589985 square cm, the width of the rectangle is 14 cm. -/
theorem rectangle_width (length : ℝ) (circle_area : ℝ) (width : ℝ) : 
  length = 18 → 
  circle_area = 153.93804002589985 → 
  circle_area = Real.pi * (width / 2)^2 → 
  width = 14 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l756_75661


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l756_75665

theorem final_sum_after_operations (a b S : ℝ) : 
  a + b = S → 3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l756_75665


namespace NUMINAMATH_CALUDE_candy_distribution_l756_75689

theorem candy_distribution (total_candy : ℕ) (num_students : ℕ) (pieces_per_student : ℕ) 
  (h1 : total_candy = 344)
  (h2 : num_students = 43)
  (h3 : pieces_per_student * num_students = total_candy) :
  pieces_per_student = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l756_75689


namespace NUMINAMATH_CALUDE_last_mile_speed_l756_75698

/-- Represents the problem of calculating the required speed for the last mile of a journey --/
theorem last_mile_speed (total_distance : ℝ) (normal_speed : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) (last_part_distance : ℝ) : 
  total_distance = 3 →
  normal_speed = 10 →
  first_part_distance = 2 →
  first_part_speed = 5 →
  last_part_distance = 1 →
  (total_distance / normal_speed = first_part_distance / first_part_speed + last_part_distance / 10) := by
  sorry

end NUMINAMATH_CALUDE_last_mile_speed_l756_75698


namespace NUMINAMATH_CALUDE_dot_product_from_norms_l756_75609

theorem dot_product_from_norms (a b : ℝ × ℝ) :
  ‖a + b‖ = Real.sqrt 10 → ‖a - b‖ = Real.sqrt 6 → a • b = 1 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_from_norms_l756_75609


namespace NUMINAMATH_CALUDE_negation_distribution_l756_75694

theorem negation_distribution (x y : ℝ) : -(x + y) = -x + -y := by sorry

end NUMINAMATH_CALUDE_negation_distribution_l756_75694


namespace NUMINAMATH_CALUDE_basketball_wins_l756_75699

/-- The total number of wins for a basketball team over four competitions -/
def total_wins (first_wins : ℕ) : ℕ :=
  let second_wins := (first_wins * 5) / 8
  let third_wins := first_wins + second_wins
  let fourth_wins := ((first_wins + second_wins + third_wins) * 3) / 5
  first_wins + second_wins + third_wins + fourth_wins

/-- Theorem stating that given 40 wins in the first competition, the total wins over four competitions is 208 -/
theorem basketball_wins : total_wins 40 = 208 := by
  sorry

end NUMINAMATH_CALUDE_basketball_wins_l756_75699


namespace NUMINAMATH_CALUDE_line_equations_correct_l756_75616

/-- Triangle ABC with vertices A(4,0), B(8,10), and C(0,6) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Definition of the specific triangle in the problem -/
def triangle : Triangle :=
  { A := (4, 0),
    B := (8, 10),
    C := (0, 6) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- The equation of the line passing through A and parallel to BC -/
def line_parallel_to_BC : LineEquation :=
  { a := 1,
    b := -1,
    c := -4 }

/-- The equation of the line containing the altitude on edge AC -/
def altitude_on_AC : LineEquation :=
  { a := 2,
    b := -3,
    c := -8 }

/-- Theorem stating the correctness of the line equations -/
theorem line_equations_correct (t : Triangle) :
  t = triangle →
  (line_parallel_to_BC.a * t.A.1 + line_parallel_to_BC.b * t.A.2 + line_parallel_to_BC.c = 0) ∧
  (altitude_on_AC.a * t.B.1 + altitude_on_AC.b * t.B.2 + altitude_on_AC.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_correct_l756_75616


namespace NUMINAMATH_CALUDE_problem_3_l756_75639

theorem problem_3 : (-48) / ((-2)^3) - (-25) * (-4) + (-2)^3 = -102 := by
  sorry

end NUMINAMATH_CALUDE_problem_3_l756_75639


namespace NUMINAMATH_CALUDE_tank_emptying_time_l756_75653

/-- Proves that a tank with given properties empties in 6 hours due to a leak alone -/
theorem tank_emptying_time (tank_capacity : ℝ) (inlet_rate : ℝ) (emptying_time_with_inlet : ℝ) :
  tank_capacity = 4320 →
  inlet_rate = 3 →
  emptying_time_with_inlet = 8 →
  ∃ (leak_rate : ℝ),
    leak_rate > 0 ∧
    (leak_rate - inlet_rate) * (emptying_time_with_inlet * 60) = tank_capacity ∧
    tank_capacity / leak_rate / 60 = 6 :=
by sorry

end NUMINAMATH_CALUDE_tank_emptying_time_l756_75653


namespace NUMINAMATH_CALUDE_min_value_theorem_l756_75615

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a + 8) * x + a^2 + a - 12

-- Define the theorem
theorem min_value_theorem (a : ℝ) (h1 : a < 0) 
  (h2 : f a (a^2 - 4) = f a (2*a - 8)) :
  ∀ n : ℕ+, (f a n - 4*a) / (n + 1) ≥ 35/8 ∧ 
  ∃ n : ℕ+, (f a n - 4*a) / (n + 1) = 35/8 := by
sorry


end NUMINAMATH_CALUDE_min_value_theorem_l756_75615


namespace NUMINAMATH_CALUDE_clock_angle_at_9am_l756_75697

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees each hour represents -/
def degrees_per_hour : ℕ := 30

/-- The position of the minute hand at 9:00 a.m. in degrees -/
def minute_hand_position : ℕ := 0

/-- The position of the hour hand at 9:00 a.m. in degrees -/
def hour_hand_position : ℕ := 270

/-- The smaller angle between the minute hand and hour hand at 9:00 a.m. -/
def smaller_angle : ℕ := 90

/-- Theorem stating that the smaller angle between the minute hand and hour hand at 9:00 a.m. is 90 degrees -/
theorem clock_angle_at_9am :
  smaller_angle = min (hour_hand_position - minute_hand_position) (360 - (hour_hand_position - minute_hand_position)) :=
by sorry

end NUMINAMATH_CALUDE_clock_angle_at_9am_l756_75697


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l756_75695

/-- Calculates the total bill given the number of people and the amount each person paid -/
def totalBill (numPeople : ℕ) (amountPerPerson : ℕ) : ℕ :=
  numPeople * amountPerPerson

/-- Proves that if three people divide a bill evenly and each pays $45, then the total bill is $135 -/
theorem restaurant_bill_proof :
  totalBill 3 45 = 135 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l756_75695


namespace NUMINAMATH_CALUDE_sum_symmetric_function_zero_l756_75675

def symmetricFunction (v : ℝ → ℝ) : Prop :=
  ∀ x, v (-x) = -v x

theorem sum_symmetric_function_zero (v : ℝ → ℝ) (h : symmetricFunction v) :
  v (-2) + v (-1) + v 1 + v 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_symmetric_function_zero_l756_75675


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l756_75624

theorem cubic_minus_linear_factorization (m : ℝ) : m^3 - m = m * (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l756_75624


namespace NUMINAMATH_CALUDE_area_cosine_plus_one_l756_75688

/-- The area enclosed by y = 1 + cos x and the x-axis over [-π, π] is 2π -/
theorem area_cosine_plus_one : 
  (∫ x in -π..π, (1 + Real.cos x)) = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_area_cosine_plus_one_l756_75688


namespace NUMINAMATH_CALUDE_cube_volume_edge_relation_l756_75671

theorem cube_volume_edge_relation (a : ℝ) (a' : ℝ) (ha : a > 0) (ha' : a' > 0) :
  (a' ^ 3) = 27 * (a ^ 3) → a' = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_edge_relation_l756_75671


namespace NUMINAMATH_CALUDE_systematic_sampling_probabilities_l756_75633

theorem systematic_sampling_probabilities 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (excluded_count : ℕ) 
  (h1 : total_students = 1005)
  (h2 : sample_size = 50)
  (h3 : excluded_count = 5) :
  (excluded_count : ℚ) / total_students = 5 / 1005 ∧
  (sample_size : ℚ) / total_students = 50 / 1005 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probabilities_l756_75633


namespace NUMINAMATH_CALUDE_next_square_property_number_l756_75677

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_square_property (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens_ones := n % 100
  is_perfect_square (hundreds * tens_ones)

theorem next_square_property_number :
  ∀ n : ℕ,
    1818 < n →
    n < 10000 →
    has_square_property n →
    (∀ m : ℕ, 1818 < m → m < n → ¬has_square_property m) →
    n = 1832 :=
sorry

end NUMINAMATH_CALUDE_next_square_property_number_l756_75677


namespace NUMINAMATH_CALUDE_tiffany_lives_l756_75681

/-- Calculate the final number of lives in a video game scenario -/
def final_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Theorem: Given Tiffany's initial lives, lives lost, and lives gained,
    prove that her final number of lives is 56 -/
theorem tiffany_lives : final_lives 43 14 27 = 56 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_lives_l756_75681


namespace NUMINAMATH_CALUDE_triangle_perimeter_l756_75636

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 7 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 →
  a + b + c = 5 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l756_75636


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l756_75622

-- Define the sample space
def S : Set ℕ := {1, 2, 3, 4, 5}

-- Define event A
def A : Set ℕ := {n ∈ S | n % 2 = 0}

-- Define event B
def B : Set ℕ := {n ∈ S | n % 2 ≠ 0}

-- Theorem statement
theorem events_mutually_exclusive_and_complementary : 
  (A ∩ B = ∅) ∧ (A ∪ B = S) := by
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l756_75622


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l756_75625

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((28457 + y) % 37 = 0 ∧ (28457 + y) % 59 = 0 ∧ (28457 + y) % 67 = 0)) ∧
  (28457 + x) % 37 = 0 ∧ (28457 + x) % 59 = 0 ∧ (28457 + x) % 67 = 0 →
  x = 117804 :=
by sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l756_75625


namespace NUMINAMATH_CALUDE_fraction_inequalities_l756_75604

theorem fraction_inequalities (a b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (a < b → (a + c) / (b + c) > a / b) ∧
  (a > b → (a + c) / (b + c) < a / b) ∧
  ((a < b → a / b < (a + c) / (b + c) ∧ (a + c) / (b + c) < 1) ∧
   (a > b → 1 < (a + c) / (b + c) ∧ (a + c) / (b + c) < a / b)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequalities_l756_75604


namespace NUMINAMATH_CALUDE_price_increase_percentage_l756_75662

theorem price_increase_percentage (original_price : ℝ) (h : original_price > 0) :
  let price_after_first_increase := original_price * 1.2
  let final_price := price_after_first_increase * 1.15
  let total_increase := final_price - original_price
  (total_increase / original_price) * 100 = 38 := by
sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l756_75662


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_minus_one_l756_75602

def imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_i_minus_one : imaginary_part (Complex.I - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_minus_one_l756_75602


namespace NUMINAMATH_CALUDE_probability_no_consecutive_ones_l756_75632

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def valid_sequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

theorem probability_no_consecutive_ones (n : ℕ) :
  n = 12 →
  (valid_sequences n : ℚ) / (total_sequences n : ℚ) = 377 / 4096 := by
  sorry

#eval valid_sequences 12
#eval total_sequences 12

end NUMINAMATH_CALUDE_probability_no_consecutive_ones_l756_75632


namespace NUMINAMATH_CALUDE_jane_egg_income_l756_75666

/-- Calculates the income from selling eggs given the number of chickens, eggs per chicken per week, 
    price per dozen eggs, and number of weeks. -/
def egg_income (num_chickens : ℕ) (eggs_per_chicken : ℕ) (price_per_dozen : ℚ) (num_weeks : ℕ) : ℚ :=
  (num_chickens * eggs_per_chicken * num_weeks : ℚ) / 12 * price_per_dozen

/-- Proves that Jane's income from selling eggs in 2 weeks is $20. -/
theorem jane_egg_income :
  egg_income 10 6 2 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jane_egg_income_l756_75666


namespace NUMINAMATH_CALUDE_digit_addition_subtraction_problem_l756_75668

/- Define digits as natural numbers from 0 to 9 -/
def Digit := {n : ℕ // n ≤ 9}

/- Define a function to convert a two-digit number to its value -/
def twoDigitValue (tens : Digit) (ones : Digit) : ℕ := 10 * tens.val + ones.val

theorem digit_addition_subtraction_problem (A B C D : Digit) :
  (twoDigitValue A B + twoDigitValue C A = twoDigitValue D A) ∧
  (twoDigitValue A B - twoDigitValue C A = A.val) →
  D.val = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_addition_subtraction_problem_l756_75668


namespace NUMINAMATH_CALUDE_range_of_m_for_B_subset_A_l756_75627

/-- The set B defined as {x | -m < x < 2} -/
def B (m : ℝ) : Set ℝ := {x | -m < x ∧ x < 2}

/-- Theorem stating the range of m for which B is a subset of A -/
theorem range_of_m_for_B_subset_A (A : Set ℝ) :
  (∀ m : ℝ, B m ⊆ A) ↔ (∀ m : ℝ, m ≤ (1/2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_B_subset_A_l756_75627


namespace NUMINAMATH_CALUDE_max_silver_tokens_l756_75618

/-- Represents the number of tokens Kevin has -/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth -/
structure ExchangeBooth where
  input_color : String
  input_amount : ℕ
  output_silver : ℕ
  output_other_color : String
  output_other_amount : ℕ

/-- Function to perform a single exchange -/
def exchange (tokens : TokenCount) (booth : ExchangeBooth) : TokenCount :=
  sorry

/-- Function to check if an exchange is possible -/
def can_exchange (tokens : TokenCount) (booth : ExchangeBooth) : Bool :=
  sorry

/-- Function to perform all possible exchanges -/
def perform_all_exchanges (tokens : TokenCount) (booths : List ExchangeBooth) : TokenCount :=
  sorry

/-- The main theorem stating the maximum number of silver tokens Kevin can obtain -/
theorem max_silver_tokens : 
  let initial_tokens : TokenCount := ⟨100, 100, 0⟩
  let booth1 : ExchangeBooth := ⟨"red", 3, 1, "blue", 2⟩
  let booth2 : ExchangeBooth := ⟨"blue", 4, 1, "red", 2⟩
  let final_tokens := perform_all_exchanges initial_tokens [booth1, booth2]
  final_tokens.silver = 132 :=
sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l756_75618


namespace NUMINAMATH_CALUDE_red_balls_count_l756_75629

theorem red_balls_count (total_balls : ℕ) (red_frequency : ℚ) (h1 : total_balls = 40) (h2 : red_frequency = 15 / 100) : 
  ⌊total_balls * red_frequency⌋ = 6 := by
sorry

end NUMINAMATH_CALUDE_red_balls_count_l756_75629


namespace NUMINAMATH_CALUDE_power_multiplication_l756_75655

theorem power_multiplication (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l756_75655


namespace NUMINAMATH_CALUDE_exist_three_similar_non_congruent_triangles_l756_75669

/-- A structure representing a triangle with side lengths and an angle -/
structure Triangle :=
  (a b c : ℝ)
  (angle_B : ℝ)

/-- Definition of similarity between two triangles -/
def similar (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.angle_B = t2.angle_B

/-- Definition of congruence between two triangles -/
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c ∧ t1.angle_B = t2.angle_B

/-- Theorem stating the existence of three pairwise similar but non-congruent triangles -/
theorem exist_three_similar_non_congruent_triangles :
  ∃ (t1 t2 t3 : Triangle),
    similar t1 t2 ∧ similar t2 t3 ∧ similar t3 t1 ∧
    ¬congruent t1 t2 ∧ ¬congruent t2 t3 ∧ ¬congruent t3 t1 :=
by
  sorry

end NUMINAMATH_CALUDE_exist_three_similar_non_congruent_triangles_l756_75669


namespace NUMINAMATH_CALUDE_probabilities_in_mathematics_l756_75635

def word : String := "mathematics"

def is_vowel (c : Char) : Bool :=
  c ∈ ['a', 'e', 'i', 'o', 'u']

def count_char (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

def count_vowels (s : String) : Nat :=
  s.toList.filter is_vowel |>.length

theorem probabilities_in_mathematics :
  (count_char word 't' : ℚ) / word.length = 2 / 11 ∧
  (count_vowels word : ℚ) / word.length = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probabilities_in_mathematics_l756_75635


namespace NUMINAMATH_CALUDE_prob_same_color_l756_75642

def box_prob (white : ℕ) (black : ℕ) : ℚ :=
  let total := white + black
  let same_color := (white.choose 3) + (black.choose 3)
  let total_combinations := total.choose 3
  same_color / total_combinations

theorem prob_same_color : box_prob 7 9 = 119 / 560 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_l756_75642


namespace NUMINAMATH_CALUDE_cards_added_l756_75670

theorem cards_added (initial_cards : ℕ) (total_cards : ℕ) (added_cards : ℕ) : 
  initial_cards = 4 → total_cards = 7 → total_cards = initial_cards + added_cards → added_cards = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_added_l756_75670


namespace NUMINAMATH_CALUDE_tangent_circles_concyclic_points_l756_75607

/-- Four circles are tangent consecutively if each circle is tangent to the next one in the sequence. -/
def ConsecutivelyTangentCircles (Γ₁ Γ₂ Γ₃ Γ₄ : Set ℝ × ℝ) : Prop := sorry

/-- Four points are the tangent points of consecutively tangent circles if they are the points where each pair of consecutive circles touch. -/
def TangentPoints (A B C D : ℝ × ℝ) (Γ₁ Γ₂ Γ₃ Γ₄ : Set ℝ × ℝ) : Prop := sorry

/-- Four points are concyclic if they lie on the same circle. -/
def Concyclic (A B C D : ℝ × ℝ) : Prop := sorry

/-- Theorem: If four circles are tangent to each other consecutively at four points, then these four points are concyclic. -/
theorem tangent_circles_concyclic_points
  (Γ₁ Γ₂ Γ₃ Γ₄ : Set ℝ × ℝ) (A B C D : ℝ × ℝ) :
  ConsecutivelyTangentCircles Γ₁ Γ₂ Γ₃ Γ₄ →
  TangentPoints A B C D Γ₁ Γ₂ Γ₃ Γ₄ →
  Concyclic A B C D :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_concyclic_points_l756_75607


namespace NUMINAMATH_CALUDE_shirt_purchase_problem_l756_75651

theorem shirt_purchase_problem (shirt_price pants_price : ℝ) 
  (num_shirts : ℕ) (num_pants : ℕ) (total_cost refund : ℝ) :
  shirt_price ≠ pants_price →
  shirt_price = 45 →
  num_pants = 3 →
  total_cost = 120 →
  refund = 0.25 * total_cost →
  total_cost = num_shirts * shirt_price + num_pants * pants_price →
  total_cost - refund = num_shirts * shirt_price →
  num_shirts = 2 :=
by
  sorry

#check shirt_purchase_problem

end NUMINAMATH_CALUDE_shirt_purchase_problem_l756_75651


namespace NUMINAMATH_CALUDE_min_value_of_expression_l756_75650

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l756_75650


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l756_75640

theorem probability_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 12) (h2 : defective_pens = 4) :
  let non_defective_pens := total_pens - defective_pens
  let prob_first := non_defective_pens / total_pens
  let prob_second := (non_defective_pens - 1) / (total_pens - 1)
  prob_first * prob_second = 14 / 33 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l756_75640


namespace NUMINAMATH_CALUDE_original_amount_is_1160_l756_75638

/-- Given an initial principal, time period, and interest rates, calculate the final amount using simple interest. -/
def simple_interest_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Prove that under the given conditions, the amount after 3 years at the original interest rate is $1160. -/
theorem original_amount_is_1160 
  (principal : ℝ) 
  (original_rate : ℝ) 
  (time : ℝ) 
  (h_principal : principal = 800) 
  (h_time : time = 3) 
  (h_increased_amount : simple_interest_amount principal (original_rate + 0.03) time = 992) :
  simple_interest_amount principal original_rate time = 1160 := by
sorry

end NUMINAMATH_CALUDE_original_amount_is_1160_l756_75638


namespace NUMINAMATH_CALUDE_divisible_by_five_l756_75631

theorem divisible_by_five (n : ℕ) : ∃ k : ℤ, (k = 2^n - 1 ∨ k = 2^n + 1 ∨ k = 2^(2*n) + 1) ∧ k % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l756_75631


namespace NUMINAMATH_CALUDE_baseball_game_attendance_l756_75614

theorem baseball_game_attendance (total : ℕ) (first_team_percent : ℚ) (second_team_percent : ℚ) 
  (h1 : total = 50)
  (h2 : first_team_percent = 40 / 100)
  (h3 : second_team_percent = 34 / 100) :
  total - (total * first_team_percent).floor - (total * second_team_percent).floor = 13 := by
  sorry

end NUMINAMATH_CALUDE_baseball_game_attendance_l756_75614


namespace NUMINAMATH_CALUDE_complex_arithmetic_l756_75691

theorem complex_arithmetic (z : ℂ) (h : z = 1 + I) : (2 / z) + z^2 = 1 + I := by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_l756_75691


namespace NUMINAMATH_CALUDE_candy_probability_contradiction_l756_75672

theorem candy_probability_contradiction :
  ∀ (packet1_blue packet1_total packet2_blue packet2_total : ℕ),
    packet1_blue ≤ packet1_total →
    packet2_blue ≤ packet2_total →
    (3 : ℚ) / 8 ≤ (packet1_blue + packet2_blue : ℚ) / (packet1_total + packet2_total) →
    (packet1_blue + packet2_blue : ℚ) / (packet1_total + packet2_total) ≤ 2 / 5 →
    ¬((17 : ℚ) / 40 ≥ 3 / 8 ∧ 17 / 40 ≤ 2 / 5) :=
by
  sorry

end NUMINAMATH_CALUDE_candy_probability_contradiction_l756_75672


namespace NUMINAMATH_CALUDE_triangle_property_l756_75674

theorem triangle_property (a b c A B C : Real) (h1 : b = a * (Real.cos C - Real.sin C))
  (h2 : a = Real.sqrt 10) (h3 : Real.sin B = Real.sqrt 2 * Real.sin C) :
  A = 3 * Real.pi / 4 ∧ 1/2 * b * c * Real.sin A = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l756_75674


namespace NUMINAMATH_CALUDE_percentage_problem_l756_75603

theorem percentage_problem (x : ℝ) (h : 0.40 * x = 160) : 0.20 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l756_75603


namespace NUMINAMATH_CALUDE_origin_outside_circle_l756_75610

/-- The circle equation: x^2 + y^2 - 2ax - 2y + (a-1)^2 = 0 -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*y + (a-1)^2 = 0

/-- A point (x, y) is outside the circle if the left-hand side of the equation is positive -/
def is_outside_circle (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*y + (a-1)^2 > 0

theorem origin_outside_circle (a : ℝ) (h : a > 1) :
  is_outside_circle 0 0 a :=
sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l756_75610


namespace NUMINAMATH_CALUDE_max_value_of_expression_l756_75656

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 3) :
  ∃ (x y z : ℝ), x^2 + y^2 + z^2 = 3 ∧ 2*x*y + 3*z = 21/4 ∧ ∀ (a b c : ℝ), a^2 + b^2 + c^2 = 3 → 2*a*b + 3*c ≤ 21/4 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l756_75656


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l756_75621

theorem two_digit_number_problem : 
  ∃! n : ℕ, 
    10 ≤ n ∧ n < 100 ∧  -- two-digit number
    (n / 10 + n % 10 = 11) ∧  -- sum of digits is 11
    (10 * (n % 10) + (n / 10) = n + 63) ∧  -- swapped number is 63 greater
    n = 29  -- the number is 29
  := by sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l756_75621


namespace NUMINAMATH_CALUDE_probability_of_matching_pair_l756_75647

def blue_socks : ℕ := 12
def gray_socks : ℕ := 10
def white_socks : ℕ := 8

def total_socks : ℕ := blue_socks + gray_socks + white_socks

def ways_to_pick_two : ℕ := total_socks.choose 2

def matching_blue_pairs : ℕ := blue_socks.choose 2
def matching_gray_pairs : ℕ := gray_socks.choose 2
def matching_white_pairs : ℕ := white_socks.choose 2

def total_matching_pairs : ℕ := matching_blue_pairs + matching_gray_pairs + matching_white_pairs

theorem probability_of_matching_pair :
  (total_matching_pairs : ℚ) / ways_to_pick_two = 139 / 435 := by sorry

end NUMINAMATH_CALUDE_probability_of_matching_pair_l756_75647


namespace NUMINAMATH_CALUDE_rectangle_width_length_ratio_l756_75620

/-- Given a rectangle with width w, length 10, and perimeter 30, 
    prove that the ratio of width to length is 1:2 -/
theorem rectangle_width_length_ratio 
  (w : ℝ) -- width of the rectangle
  (h1 : w > 0) -- width is positive
  (h2 : 2 * w + 2 * 10 = 30) -- perimeter formula
  : w / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_length_ratio_l756_75620


namespace NUMINAMATH_CALUDE_f_monotonicity_f_two_zeros_l756_75605

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * (x + 2)

theorem f_monotonicity :
  let f₁ := f 1
  (∀ x y, x < y → x < 0 → y < 0 → f₁ y < f₁ x) ∧
  (∀ x y, 0 < x → x < y → f₁ x < f₁ y) :=
sorry

theorem f_two_zeros (a : ℝ) :
  (∃ x y, x < y ∧ f a x = 0 ∧ f a y = 0) ↔ (Real.exp (-1) < a) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_f_two_zeros_l756_75605


namespace NUMINAMATH_CALUDE_f_monotonicity_and_a_range_l756_75611

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) / Real.log x

theorem f_monotonicity_and_a_range :
  (∀ x₁ x₂, e < x₁ ∧ x₁ < x₂ → f 0 x₁ < f 0 x₂) ∧ 
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f 0 x₁ > f 0 x₂) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < e → f 0 x₁ > f 0 x₂) ∧
  (∀ a, (∀ x, 1 < x → f a x > Real.sqrt x) → a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_a_range_l756_75611


namespace NUMINAMATH_CALUDE_characterization_of_satisfying_functions_l756_75613

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, y^2 * f x + x^2 * f y + x * y = x * y * f (x + y) + x^2 + y^2

/-- The main theorem stating the form of functions satisfying the equation -/
theorem characterization_of_satisfying_functions :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x + 1 := by
sorry

end NUMINAMATH_CALUDE_characterization_of_satisfying_functions_l756_75613


namespace NUMINAMATH_CALUDE_divisible_by_three_l756_75634

theorem divisible_by_three (k : ℤ) : 3 ∣ ((2*k + 3)^2 - 4*k^2) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l756_75634


namespace NUMINAMATH_CALUDE_tan_negative_fifty_five_sixths_pi_l756_75690

theorem tan_negative_fifty_five_sixths_pi : 
  Real.tan (-55 / 6 * Real.pi) = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_negative_fifty_five_sixths_pi_l756_75690


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l756_75645

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l756_75645


namespace NUMINAMATH_CALUDE_complex_fraction_power_simplification_l756_75687

theorem complex_fraction_power_simplification :
  (((3 : ℂ) + 4*I) / ((3 : ℂ) - 4*I))^8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_simplification_l756_75687


namespace NUMINAMATH_CALUDE_only_B_and_C_participate_l756_75644

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a type for the activity participation
def Activity := Person → Prop

-- Define the conditions
def condition1 (act : Activity) : Prop := act Person.A → act Person.B
def condition2 (act : Activity) : Prop := ¬act Person.C → ¬act Person.B
def condition3 (act : Activity) : Prop := act Person.C → ¬act Person.D

-- Define the property of exactly two people participating
def exactlyTwo (act : Activity) : Prop :=
  ∃ (p1 p2 : Person), p1 ≠ p2 ∧ act p1 ∧ act p2 ∧ ∀ (p : Person), act p → (p = p1 ∨ p = p2)

-- The main theorem
theorem only_B_and_C_participate :
  ∀ (act : Activity),
    condition1 act →
    condition2 act →
    condition3 act →
    exactlyTwo act →
    act Person.B ∧ act Person.C ∧ ¬act Person.A ∧ ¬act Person.D :=
by sorry

end NUMINAMATH_CALUDE_only_B_and_C_participate_l756_75644


namespace NUMINAMATH_CALUDE_sam_apple_consumption_l756_75678

/-- Calculates the number of apples Sam eats in a week -/
def apples_eaten_in_week (apples_per_sandwich : ℕ) (sandwiches_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  apples_per_sandwich * sandwiches_per_day * days_in_week

/-- Proves that Sam eats 280 apples in one week -/
theorem sam_apple_consumption : apples_eaten_in_week 4 10 7 = 280 := by
  sorry

#eval apples_eaten_in_week 4 10 7

end NUMINAMATH_CALUDE_sam_apple_consumption_l756_75678


namespace NUMINAMATH_CALUDE_min_value_theorem_l756_75617

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b) * (b + 4/a) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l756_75617


namespace NUMINAMATH_CALUDE_sum_of_digits_in_divisible_number_l756_75664

/-- 
Given a number in the form ̄1ab76 that is divisible by 72, 
prove that the sum a+b can only be 4 or 13.
-/
theorem sum_of_digits_in_divisible_number (a b : Nat) : 
  (∃ (n : Nat), n = 10000 + 1000 * a + 100 * b + 76) →
  (10000 + 1000 * a + 100 * b + 76) % 72 = 0 →
  a + b = 4 ∨ a + b = 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_divisible_number_l756_75664


namespace NUMINAMATH_CALUDE_fraction_equals_decimal_l756_75685

theorem fraction_equals_decimal : (1 : ℚ) / 4 = 0.25 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_decimal_l756_75685


namespace NUMINAMATH_CALUDE_basketball_players_count_l756_75680

theorem basketball_players_count (total_athletes : ℕ) 
  (football_ratio baseball_ratio soccer_ratio basketball_ratio : ℕ) : 
  total_athletes = 104 →
  football_ratio = 10 →
  baseball_ratio = 7 →
  soccer_ratio = 5 →
  basketball_ratio = 4 →
  (basketball_ratio * total_athletes) / (football_ratio + baseball_ratio + soccer_ratio + basketball_ratio) = 16 := by
  sorry

end NUMINAMATH_CALUDE_basketball_players_count_l756_75680


namespace NUMINAMATH_CALUDE_equation_solution_l756_75657

theorem equation_solution : ∃ x : ℝ, (4 / (x - 1) + 1 / (1 - x) = 1) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l756_75657


namespace NUMINAMATH_CALUDE_valid_four_digit_numbers_l756_75612

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (∀ d : ℕ, d ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10] → is_prime d) ∧
  (∀ p : ℕ, p ∈ [n / 100, n / 10 % 100, n % 100] → is_prime p)

theorem valid_four_digit_numbers :
  {n : ℕ | is_valid_number n} = {2373, 3737, 5373, 7373} :=
by sorry

end NUMINAMATH_CALUDE_valid_four_digit_numbers_l756_75612


namespace NUMINAMATH_CALUDE_comparison_of_rational_numbers_l756_75696

theorem comparison_of_rational_numbers :
  (- (- (1 / 5 : ℚ)) > - (1 / 5 : ℚ)) ∧
  (- (- (17 / 5 : ℚ)) > - (17 / 5 : ℚ)) ∧
  (- (4 : ℚ) < (4 : ℚ)) ∧
  ((- (11 / 10 : ℚ)) < 0) :=
by sorry

end NUMINAMATH_CALUDE_comparison_of_rational_numbers_l756_75696


namespace NUMINAMATH_CALUDE_rachel_homework_l756_75683

theorem rachel_homework (math_pages reading_pages : ℕ) : 
  math_pages = 10 →
  math_pages + reading_pages = 23 →
  reading_pages > math_pages →
  reading_pages - math_pages = 3 :=
by sorry

end NUMINAMATH_CALUDE_rachel_homework_l756_75683


namespace NUMINAMATH_CALUDE_unique_students_count_l756_75637

theorem unique_students_count (orchestra band choir : ℕ) 
  (orchestra_band orchestra_choir band_choir all_three : ℕ) :
  orchestra = 25 →
  band = 40 →
  choir = 30 →
  orchestra_band = 5 →
  orchestra_choir = 6 →
  band_choir = 4 →
  all_three = 2 →
  orchestra + band + choir - (orchestra_band + orchestra_choir + band_choir) + all_three = 82 :=
by sorry

end NUMINAMATH_CALUDE_unique_students_count_l756_75637


namespace NUMINAMATH_CALUDE_probability_one_girl_in_pair_l756_75686

theorem probability_one_girl_in_pair (n_boys n_girls : ℕ) (h_boys : n_boys = 4) (h_girls : n_girls = 2) :
  let total := n_boys + n_girls
  let total_pairs := total.choose 2
  let favorable_outcomes := n_boys * n_girls
  (favorable_outcomes : ℚ) / total_pairs = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_girl_in_pair_l756_75686


namespace NUMINAMATH_CALUDE_sequence_properties_l756_75676

def a (n : ℤ) : ℤ := 30 + n - n^2

theorem sequence_properties :
  (a 10 = -60) ∧
  (∀ n : ℤ, a n = 0 ↔ n = 6) ∧
  (∀ n : ℤ, a n > 0 ↔ n > 6) ∧
  (∀ n : ℤ, a n < 0 ↔ n < 6) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l756_75676


namespace NUMINAMATH_CALUDE_special_parallelogram_existence_l756_75643

/-- The existence of a special parallelogram for any point on an ellipse -/
theorem special_parallelogram_existence (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 →
    ∃ (p q r s : ℝ × ℝ),
      -- P is on the ellipse
      (x, y) = p ∧
      -- PQRS forms a parallelogram
      (p.1 - q.1 = r.1 - s.1 ∧ p.2 - q.2 = r.2 - s.2) ∧
      (p.1 - s.1 = q.1 - r.1 ∧ p.2 - s.2 = q.2 - r.2) ∧
      -- Parallelogram is tangent to the ellipse
      (∃ (t : ℝ × ℝ), t.1^2/a^2 + t.2^2/b^2 = 1 ∧
        ((t.1 - p.1) * (q.1 - p.1) + (t.2 - p.2) * (q.2 - p.2) = 0 ∨
         (t.1 - q.1) * (r.1 - q.1) + (t.2 - q.2) * (r.2 - q.2) = 0 ∨
         (t.1 - r.1) * (s.1 - r.1) + (t.2 - r.2) * (s.2 - r.2) = 0 ∨
         (t.1 - s.1) * (p.1 - s.1) + (t.2 - s.2) * (p.2 - s.2) = 0)) ∧
      -- Parallelogram is externally tangent to the unit circle
      (∃ (u : ℝ × ℝ), u.1^2 + u.2^2 = 1 ∧
        ((u.1 - p.1) * (q.1 - p.1) + (u.2 - p.2) * (q.2 - p.2) = 0 ∨
         (u.1 - q.1) * (r.1 - q.1) + (u.2 - q.2) * (r.2 - q.2) = 0 ∨
         (u.1 - r.1) * (s.1 - r.1) + (u.2 - r.2) * (s.2 - r.2) = 0 ∨
         (u.1 - s.1) * (p.1 - s.1) + (u.2 - s.2) * (p.2 - s.2) = 0))) ↔
  1/a^2 + 1/b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_special_parallelogram_existence_l756_75643


namespace NUMINAMATH_CALUDE_minimize_y_l756_75679

variable (a b : ℝ)
def y (x : ℝ) := (x - a)^2 + (x - b)^2

theorem minimize_y :
  ∃ (x : ℝ), ∀ (z : ℝ), y a b x ≤ y a b z ∧ x = (a + b) / 2 :=
sorry

end NUMINAMATH_CALUDE_minimize_y_l756_75679


namespace NUMINAMATH_CALUDE_function_property_l756_75673

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 2

theorem function_property (a : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 (Real.exp 1), ∃ x₂ ∈ Set.Icc 1 (Real.exp 1), f a x₁ + f a x₂ = 4) ↔
  a = Real.exp 1 + 1 :=
by sorry

end NUMINAMATH_CALUDE_function_property_l756_75673


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_387420501_l756_75684

-- Define the polynomial
def polynomial (x y : ℤ) : ℤ := (3*x + 4*y)^9 + (2*x - 5*y)^9

-- Define the sum of coefficients function
def sum_of_coefficients (p : ℤ → ℤ → ℤ) (x y : ℤ) : ℤ := p x y

-- Theorem statement
theorem sum_of_coefficients_equals_387420501 :
  sum_of_coefficients polynomial 2 (-1) = 387420501 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_387420501_l756_75684


namespace NUMINAMATH_CALUDE_paint_area_calculation_l756_75654

/-- Calculates the area to be painted on a wall with given dimensions and openings. -/
def areaToPaint (wallHeight wallWidth windowHeight windowWidth doorHeight doorWidth : ℝ) : ℝ :=
  let wallArea := wallHeight * wallWidth
  let windowArea := windowHeight * windowWidth
  let doorArea := doorHeight * doorWidth
  wallArea - windowArea - doorArea

/-- Theorem stating that the area to be painted on the given wall is 128.5 square feet. -/
theorem paint_area_calculation :
  areaToPaint 10 15 3 5 1 6.5 = 128.5 := by
  sorry

end NUMINAMATH_CALUDE_paint_area_calculation_l756_75654


namespace NUMINAMATH_CALUDE_identical_circles_inside_no_common_tangents_l756_75601

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if one circle is fully inside another without touching -/
def IsFullyInside (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 < (c1.radius - c2.radius) ^ 2

/-- The number of common tangents between two circles -/
def CommonTangents (c1 c2 : Circle) : ℕ := sorry

/-- Theorem stating that two identical circles, one fully inside the other without touching, have zero common tangents -/
theorem identical_circles_inside_no_common_tangents (c1 c2 : Circle) :
  c1.radius = c2.radius → IsFullyInside c1 c2 → CommonTangents c1 c2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_identical_circles_inside_no_common_tangents_l756_75601


namespace NUMINAMATH_CALUDE_floral_arrangement_daisies_percentage_l756_75606

theorem floral_arrangement_daisies_percentage
  (total : ℝ)
  (yellow_flowers : ℝ)
  (blue_flowers : ℝ)
  (yellow_tulips : ℝ)
  (blue_tulips : ℝ)
  (yellow_daisies : ℝ)
  (blue_daisies : ℝ)
  (h1 : yellow_flowers = 7 / 10 * total)
  (h2 : blue_flowers = 3 / 10 * total)
  (h3 : yellow_tulips = 1 / 2 * yellow_flowers)
  (h4 : blue_daisies = 1 / 3 * blue_flowers)
  (h5 : yellow_flowers + blue_flowers = total)
  (h6 : yellow_tulips + blue_tulips + yellow_daisies + blue_daisies = total)
  : (yellow_daisies + blue_daisies) / total = 9 / 20 :=
by sorry

end NUMINAMATH_CALUDE_floral_arrangement_daisies_percentage_l756_75606


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_5_l756_75663

theorem greatest_three_digit_divisible_by_3_6_5 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 3 ∣ n ∧ 6 ∣ n ∧ 5 ∣ n → n ≤ 990 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_5_l756_75663


namespace NUMINAMATH_CALUDE_possible_student_counts_l756_75659

def is_valid_student_count (n : ℕ) : Prop :=
  n > 1 ∧ (120 - 2) % (n - 1) = 0

theorem possible_student_counts :
  ∀ n : ℕ, is_valid_student_count n ↔ n = 2 ∨ n = 3 ∨ n = 60 ∨ n = 119 := by
  sorry

end NUMINAMATH_CALUDE_possible_student_counts_l756_75659


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l756_75667

theorem triangle_abc_properties (a b : ℝ) (A B C : ℝ) :
  -- Conditions
  (a^2 - 2 * Real.sqrt 3 * a + 2 = 0) →
  (b^2 - 2 * Real.sqrt 3 * b + 2 = 0) →
  (2 * Real.cos (A + B) = 1) →
  -- Conclusions
  (C = 120 * π / 180) ∧
  (Real.sqrt ((a^2 + b^2 + a*b) : ℝ) = Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l756_75667


namespace NUMINAMATH_CALUDE_green_or_yellow_marble_probability_l756_75658

/-- The probability of drawing a green or yellow marble from a bag -/
theorem green_or_yellow_marble_probability
  (green : ℕ) (yellow : ℕ) (red : ℕ) (blue : ℕ)
  (h_green : green = 4)
  (h_yellow : yellow = 3)
  (h_red : red = 4)
  (h_blue : blue = 2) :
  (green + yellow : ℚ) / (green + yellow + red + blue) = 7 / 13 :=
by sorry

end NUMINAMATH_CALUDE_green_or_yellow_marble_probability_l756_75658


namespace NUMINAMATH_CALUDE_c_share_approx_l756_75619

/-- Investment ratios and time periods for partners A, B, and C --/
structure PartnershipData where
  a_to_b_ratio : ℚ
  a_to_c_ratio : ℚ
  a_time : ℕ
  b_time : ℕ
  c_time : ℕ
  total_profit : ℚ

/-- Calculate C's share of the profit --/
def calculate_c_share (data : PartnershipData) : ℚ :=
  let b_investment := 1
  let a_investment := data.a_to_b_ratio * b_investment
  let c_investment := a_investment / data.a_to_c_ratio
  let total_ratio := a_investment * data.a_time + b_investment * data.b_time + c_investment * data.c_time
  (c_investment * data.c_time / total_ratio) * data.total_profit

/-- Theorem stating that C's share is approximately 79,136.57 --/
theorem c_share_approx (data : PartnershipData) 
  (h1 : data.a_to_b_ratio = 5)
  (h2 : data.a_to_c_ratio = 3/5)
  (h3 : data.a_time = 6)
  (h4 : data.b_time = 9)
  (h5 : data.c_time = 12)
  (h6 : data.total_profit = 110000) :
  ∃ ε > 0, |calculate_c_share data - 79136.57| < ε := by
  sorry

end NUMINAMATH_CALUDE_c_share_approx_l756_75619


namespace NUMINAMATH_CALUDE_choir_competition_score_l756_75648

/-- Calculates the final score of a choir competition team given their individual scores and weights -/
def final_score (song_content : ℝ) (singing_skills : ℝ) (spirit : ℝ) : ℝ :=
  0.3 * song_content + 0.5 * singing_skills + 0.2 * spirit

/-- Theorem stating that the final score for the given team is 93 -/
theorem choir_competition_score :
  final_score 90 94 95 = 93 := by
  sorry

#eval final_score 90 94 95

end NUMINAMATH_CALUDE_choir_competition_score_l756_75648


namespace NUMINAMATH_CALUDE_pumpkin_contest_result_l756_75641

/-- The weight of Brad's pumpkin in pounds -/
def brads_pumpkin : ℕ := 54

/-- The weight of Jessica's pumpkin in pounds -/
def jessicas_pumpkin : ℕ := brads_pumpkin / 2

/-- The weight of Betty's pumpkin in pounds -/
def bettys_pumpkin : ℕ := jessicas_pumpkin * 4

/-- The difference between the heaviest and lightest pumpkin in pounds -/
def pumpkin_weight_difference : ℕ := max brads_pumpkin (max jessicas_pumpkin bettys_pumpkin) - 
                                     min brads_pumpkin (min jessicas_pumpkin bettys_pumpkin)

theorem pumpkin_contest_result : pumpkin_weight_difference = 81 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_contest_result_l756_75641


namespace NUMINAMATH_CALUDE_ammeter_readings_sum_l756_75692

/-- The sum of readings of five ammeters in a specific circuit configuration -/
def sum_of_ammeter_readings (I₁ I₂ I₃ I₄ I₅ : ℝ) : ℝ :=
  I₁ + I₂ + I₃ + I₄ + I₅

/-- Theorem stating the sum of ammeter readings in the given circuit -/
theorem ammeter_readings_sum :
  ∀ (I₁ I₂ I₃ I₄ I₅ : ℝ),
    I₁ = 2 →
    I₂ = I₁ →
    I₃ = I₁ + I₂ →
    I₅ = I₃ + I₁ →
    I₄ = (5/3) * I₅ →
    sum_of_ammeter_readings I₁ I₂ I₃ I₄ I₅ = 24 := by
  sorry


end NUMINAMATH_CALUDE_ammeter_readings_sum_l756_75692


namespace NUMINAMATH_CALUDE_toys_in_box_time_l756_75626

/-- The time it takes to put all toys in the box -/
def time_to_put_toys_in_box (total_toys : ℕ) (toys_in_per_minute : ℕ) (toys_out_per_minute : ℕ) : ℕ :=
  sorry

/-- Theorem stating that it takes 25 minutes to put all toys in the box under given conditions -/
theorem toys_in_box_time : time_to_put_toys_in_box 50 5 3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_toys_in_box_time_l756_75626


namespace NUMINAMATH_CALUDE_simplification_to_5x_squared_l756_75652

theorem simplification_to_5x_squared (k : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, (x - k*x) * (2*x - k*x) - 3*x * (2*x - k*x) = 5*x^2) ∧ 
  (∀ k : ℝ, (∀ x : ℝ, (x - k*x) * (2*x - k*x) - 3*x * (2*x - k*x) = 5*x^2) → (k = 3 ∨ k = -3)) :=
by sorry

end NUMINAMATH_CALUDE_simplification_to_5x_squared_l756_75652


namespace NUMINAMATH_CALUDE_circle_radius_l756_75630

theorem circle_radius (x y : ℝ) :
  x^2 + y^2 - 2*x + 6*y + 1 = 0 → ∃ (h k r : ℝ), r = 3 ∧ (x - h)^2 + (y - k)^2 = r^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l756_75630


namespace NUMINAMATH_CALUDE_tan_alpha_minus_pi_fourth_l756_75608

theorem tan_alpha_minus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β + π/4) = 1/4) :
  Real.tan (α - π/4) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_pi_fourth_l756_75608

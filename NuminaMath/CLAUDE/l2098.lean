import Mathlib

namespace NUMINAMATH_CALUDE_total_cookies_is_16000_l2098_209844

/-- The number of church members volunteering to bake cookies. -/
def num_members : ℕ := 100

/-- The number of sheets of cookies each member bakes. -/
def sheets_per_member : ℕ := 10

/-- The number of cookies on each sheet. -/
def cookies_per_sheet : ℕ := 16

/-- The total number of cookies baked by all church members. -/
def total_cookies : ℕ := num_members * sheets_per_member * cookies_per_sheet

/-- Theorem stating that the total number of cookies baked is 16,000. -/
theorem total_cookies_is_16000 : total_cookies = 16000 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_is_16000_l2098_209844


namespace NUMINAMATH_CALUDE_pipe_fill_time_l2098_209873

/-- Given two pipes that can fill a pool, where one takes T hours and the other takes 12 hours,
    prove that if both pipes together take 4.8 hours to fill the pool, then T = 8. -/
theorem pipe_fill_time (T : ℝ) :
  T > 0 →
  1 / T + 1 / 12 = 1 / 4.8 →
  T = 8 :=
by sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l2098_209873


namespace NUMINAMATH_CALUDE_wayne_blocks_total_l2098_209845

theorem wayne_blocks_total (initial_blocks additional_blocks : ℕ) 
  (h1 : initial_blocks = 9)
  (h2 : additional_blocks = 6) :
  initial_blocks + additional_blocks = 15 := by
  sorry

end NUMINAMATH_CALUDE_wayne_blocks_total_l2098_209845


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2098_209852

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  S : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, 2 * (S (n + 1) - S n) = S (n + 2) - S n

/-- Theorem: If S_5 : S_10 = 2 : 3, then S_15 : S_5 = 3 : 2 -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.S 5 / seq.S 10 = 2 / 3) : 
  seq.S 15 / seq.S 5 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2098_209852


namespace NUMINAMATH_CALUDE_two_circles_k_value_l2098_209851

/-- Two circles centered at the origin with given properties --/
structure TwoCircles where
  -- Radius of the larger circle
  R : ℝ
  -- Radius of the smaller circle
  r : ℝ
  -- Point P on the larger circle
  P : ℝ × ℝ
  -- Point S on the smaller circle
  S : ℝ × ℝ
  -- Distance QR
  QR : ℝ
  -- Conditions
  center_origin : True
  P_on_larger : P.1^2 + P.2^2 = R^2
  S_on_smaller : S.1^2 + S.2^2 = r^2
  S_on_y_axis : S.1 = 0
  radius_difference : R - r = QR

/-- Theorem stating the value of k for the given two circles --/
theorem two_circles_k_value (c : TwoCircles) (h1 : c.P = (10, 2)) (h2 : c.QR = 5) :
  ∃ k : ℝ, c.S = (0, k) ∧ (k = Real.sqrt 104 - 5 ∨ k = -(Real.sqrt 104 - 5)) := by
  sorry

end NUMINAMATH_CALUDE_two_circles_k_value_l2098_209851


namespace NUMINAMATH_CALUDE_imaginary_unit_expression_l2098_209849

theorem imaginary_unit_expression : Complex.I^7 - 2 / Complex.I = Complex.I := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_expression_l2098_209849


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2098_209812

theorem polynomial_simplification (x : ℝ) :
  (5 * x^12 + 8 * x^11 + 10 * x^9) + (3 * x^13 + 2 * x^12 + x^11 + 6 * x^9 + 7 * x^5 + 8 * x^2 + 9) =
  3 * x^13 + 7 * x^12 + 9 * x^11 + 16 * x^9 + 7 * x^5 + 8 * x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2098_209812


namespace NUMINAMATH_CALUDE_flower_bouquet_carnations_percentage_l2098_209883

theorem flower_bouquet_carnations_percentage 
  (total_flowers : ℕ) 
  (pink_flowers red_flowers pink_roses red_roses pink_carnations red_carnations : ℕ) :
  (pink_flowers = total_flowers / 2) →
  (red_flowers = total_flowers / 2) →
  (pink_roses = pink_flowers * 2 / 5) →
  (red_carnations = red_flowers * 2 / 3) →
  (pink_carnations = pink_flowers - pink_roses) →
  (red_roses = red_flowers - red_carnations) →
  (((pink_carnations + red_carnations : ℚ) / total_flowers) * 100 = 63) := by
  sorry

end NUMINAMATH_CALUDE_flower_bouquet_carnations_percentage_l2098_209883


namespace NUMINAMATH_CALUDE_no_valid_coloring_l2098_209891

def Color := Fin 3

theorem no_valid_coloring :
  ¬∃ f : ℕ+ → Color,
    (∀ c : Color, ∃ n : ℕ+, f n = c) ∧
    (∀ a b : ℕ+, f a ≠ f b → f (a * b) ≠ f a ∧ f (a * b) ≠ f b) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l2098_209891


namespace NUMINAMATH_CALUDE_parabola_intersection_l2098_209820

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 9 * x - 8
  let g (x : ℝ) := x^2 - 3 * x + 4
  (f 3 = g 3 ∧ f 3 = -8) ∧ (f (-2) = g (-2) ∧ f (-2) = 22) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2098_209820


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2098_209854

theorem geometric_series_sum : ∀ (a r : ℝ), 
  a = 9 → r = -2/3 → abs r < 1 → 
  (∑' n, a * r^n) = 5.4 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2098_209854


namespace NUMINAMATH_CALUDE_sum_of_powers_of_fifth_root_of_unity_l2098_209836

theorem sum_of_powers_of_fifth_root_of_unity (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^15 + ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_fifth_root_of_unity_l2098_209836


namespace NUMINAMATH_CALUDE_red_stripes_on_fifty_flags_l2098_209889

/-- Calculates the total number of red stripes on multiple flags -/
def total_red_stripes (stripes_per_flag : ℕ) (num_flags : ℕ) : ℕ :=
  let remaining_stripes := stripes_per_flag - 1
  let red_remaining := remaining_stripes / 2
  let red_per_flag := red_remaining + 1
  red_per_flag * num_flags

/-- Theorem stating the total number of red stripes on 50 flags -/
theorem red_stripes_on_fifty_flags :
  total_red_stripes 25 50 = 650 := by
  sorry

end NUMINAMATH_CALUDE_red_stripes_on_fifty_flags_l2098_209889


namespace NUMINAMATH_CALUDE_lino_shell_collection_l2098_209863

/-- Theorem: Lino's shell collection
  Given:
  - Lino put 292 shells back in the afternoon
  - She has 32 shells in total at the end
  Prove that Lino picked up 324 shells in the morning
-/
theorem lino_shell_collection (shells_put_back shells_remaining : ℕ) 
  (h1 : shells_put_back = 292)
  (h2 : shells_remaining = 32) :
  shells_put_back + shells_remaining = 324 := by
  sorry

end NUMINAMATH_CALUDE_lino_shell_collection_l2098_209863


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2098_209810

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 5*x + 5 = 16) → (∃ y : ℝ, y^2 - 5*y + 5 = 16 ∧ x + y = 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2098_209810


namespace NUMINAMATH_CALUDE_argument_not_pi_over_four_l2098_209879

-- Define the complex number z
variable (z : ℂ)

-- Define the condition |z-|z+1|| = |z+|z-1||
def condition (z : ℂ) : Prop :=
  Complex.abs (z - Complex.abs (z + 1)) = Complex.abs (z + Complex.abs (z - 1))

-- Theorem statement
theorem argument_not_pi_over_four (h : condition z) :
  Complex.arg z ≠ Real.pi / 4 :=
sorry

end NUMINAMATH_CALUDE_argument_not_pi_over_four_l2098_209879


namespace NUMINAMATH_CALUDE_log_equation_sum_l2098_209888

theorem log_equation_sum (A B C : ℕ+) : 
  (Nat.gcd A.val (Nat.gcd B.val C.val) = 1) →
  (A : ℝ) * (Real.log 5 / Real.log 200) + (B : ℝ) * (Real.log 2 / Real.log 200) = C →
  A + B + C = 6 := by
sorry

end NUMINAMATH_CALUDE_log_equation_sum_l2098_209888


namespace NUMINAMATH_CALUDE_power_multiplication_l2098_209842

theorem power_multiplication (x : ℝ) : x^5 * x^3 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2098_209842


namespace NUMINAMATH_CALUDE_grocery_payment_possible_l2098_209866

def soup_price : ℕ := 2
def bread_price : ℕ := 5
def cereal_price : ℕ := 3
def milk_price : ℕ := 4

def soup_quantity : ℕ := 6
def bread_quantity : ℕ := 2
def cereal_quantity : ℕ := 2
def milk_quantity : ℕ := 2

def total_cost : ℕ := 
  soup_price * soup_quantity + 
  bread_price * bread_quantity + 
  cereal_price * cereal_quantity + 
  milk_price * milk_quantity

def us_bill_denominations : List ℕ := [1, 2, 5, 10, 20, 50, 100]

theorem grocery_payment_possible :
  ∃ (a b c d : ℕ), 
    a ∈ us_bill_denominations ∧ 
    b ∈ us_bill_denominations ∧ 
    c ∈ us_bill_denominations ∧ 
    d ∈ us_bill_denominations ∧ 
    a + b + c + d = total_cost :=
sorry

end NUMINAMATH_CALUDE_grocery_payment_possible_l2098_209866


namespace NUMINAMATH_CALUDE_pigeon_difference_l2098_209859

theorem pigeon_difference (total_pigeons : ℕ) (black_ratio : ℚ) (male_ratio : ℚ) : 
  total_pigeons = 70 →
  black_ratio = 1/2 →
  male_ratio = 1/5 →
  (black_ratio * total_pigeons : ℚ) * (1 - male_ratio) - (black_ratio * total_pigeons : ℚ) * male_ratio = 21 := by
  sorry

end NUMINAMATH_CALUDE_pigeon_difference_l2098_209859


namespace NUMINAMATH_CALUDE_team_win_percentage_l2098_209816

theorem team_win_percentage (games_won : ℕ) (games_lost : ℕ) 
  (h : games_won / games_lost = 13 / 7) : 
  (games_won : ℚ) / (games_won + games_lost) * 100 = 65 := by
  sorry

end NUMINAMATH_CALUDE_team_win_percentage_l2098_209816


namespace NUMINAMATH_CALUDE_triangle_properties_l2098_209807

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (abc : Triangle) :
  (abc.B.cos = -5/13 ∧ 
   (2 * abc.A.sin) * (2 * abc.C.sin) = abc.B.sin^2 ∧ 
   1/2 * abc.a * abc.c * abc.B.sin = 6/13) →
  (abc.a + abc.c) / 2 = Real.sqrt 221 / 13
  ∧
  (abc.B.cos = -5/13 ∧ 
   abc.C.cos = 4/5 ∧ 
   abc.b * abc.c * abc.A.cos = 14) →
  abc.a = 11/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2098_209807


namespace NUMINAMATH_CALUDE_inequality_proof_l2098_209806

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.sin (80 * π / 180))
  (hb : b = (1/2)⁻¹)
  (hc : c = Real.log 3 / Real.log (1/2)) :
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2098_209806


namespace NUMINAMATH_CALUDE_five_students_three_colleges_l2098_209840

/-- The number of ways for students to apply to colleges -/
def applicationWays (numStudents : ℕ) (numColleges : ℕ) : ℕ :=
  numColleges ^ numStudents

/-- Theorem: 5 students applying to 3 colleges results in 3^5 different ways -/
theorem five_students_three_colleges : 
  applicationWays 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_five_students_three_colleges_l2098_209840


namespace NUMINAMATH_CALUDE_observation_probability_l2098_209803

theorem observation_probability 
  (total_students : Nat) 
  (total_periods : Nat) 
  (zi_shi_duration : Nat) 
  (total_duration : Nat) :
  total_students = 4 →
  total_periods = 4 →
  zi_shi_duration = 2 →
  total_duration = 8 →
  (zi_shi_duration : ℚ) / total_duration = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_observation_probability_l2098_209803


namespace NUMINAMATH_CALUDE_quadratic_equals_binomial_square_l2098_209865

theorem quadratic_equals_binomial_square (d : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 60*x + d = (a*x + b)^2) → d = 900 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equals_binomial_square_l2098_209865


namespace NUMINAMATH_CALUDE_hall_volume_l2098_209881

/-- Proves that a rectangular hall with given dimensions and area equality has a volume of 972 cubic meters -/
theorem hall_volume (length width height : ℝ) : 
  length = 18 ∧ 
  width = 9 ∧ 
  2 * (length * width) = 2 * (length * height) + 2 * (width * height) → 
  length * width * height = 972 := by
  sorry

end NUMINAMATH_CALUDE_hall_volume_l2098_209881


namespace NUMINAMATH_CALUDE_attainable_tables_count_l2098_209870

/-- Represents a table with signs -/
def Table (m n : ℕ) := Fin (2*m) → Fin (2*n) → Bool

/-- Determines if a table is attainable after one transformation -/
def IsAttainable (m n : ℕ) (t : Table m n) : Prop := sorry

/-- Counts the number of attainable tables -/
def CountAttainableTables (m n : ℕ) : ℕ := sorry

theorem attainable_tables_count (m n : ℕ) :
  CountAttainableTables m n = if m % 2 = 1 ∧ n % 2 = 1 then 2^(m+n-2) else 2^(m+n-1) := by sorry

end NUMINAMATH_CALUDE_attainable_tables_count_l2098_209870


namespace NUMINAMATH_CALUDE_platform_walk_probability_l2098_209855

/-- The number of platforms at the train station -/
def num_platforms : ℕ := 16

/-- The distance between adjacent platforms in feet -/
def platform_distance : ℕ := 200

/-- The maximum walking distance we're interested in -/
def max_walk_distance : ℕ := 800

/-- The probability of walking 800 feet or less between two randomly assigned platforms -/
theorem platform_walk_probability : 
  let total_assignments := num_platforms * (num_platforms - 1)
  let favorable_assignments := 
    (2 * 4 * 8) +  -- Edge platforms (1-4 and 13-16) have 8 choices each
    (8 * 10)       -- Central platforms (5-12) have 10 choices each
  (favorable_assignments : ℚ) / total_assignments = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_platform_walk_probability_l2098_209855


namespace NUMINAMATH_CALUDE_card_sum_theorem_l2098_209856

theorem card_sum_theorem (a b c d e f g h : ℕ) : 
  (a + b) * (c + d) * (e + f) * (g + h) = 330 → 
  a + b + c + d + e + f + g + h = 21 := by
sorry

end NUMINAMATH_CALUDE_card_sum_theorem_l2098_209856


namespace NUMINAMATH_CALUDE_geometric_series_property_l2098_209822

theorem geometric_series_property (b₁ q : ℝ) (h_q : |q| < 1) :
  (b₁ / (1 - q)) / (b₁^3 / (1 - q^3)) = 1/12 →
  (b₁^4 / (1 - q^4)) / (b₁^2 / (1 - q^2)) = 36/5 →
  (b₁ = 3 ∨ b₁ = -3) ∧ q = -1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_property_l2098_209822


namespace NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l2098_209839

theorem probability_four_twos_in_five_rolls (p : ℝ) (h1 : p = 1 / 6) :
  (5 : ℝ) * p^4 * (1 - p) = 5 / 72 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l2098_209839


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2098_209869

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ordering of numbers
  b = 8 →  -- Median is 8
  (a + b + c) / 3 = a + 8 →  -- Mean is 8 more than least
  (a + b + c) / 3 = c - 20 →  -- Mean is 20 less than greatest
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2098_209869


namespace NUMINAMATH_CALUDE_quadratic_always_negative_l2098_209821

theorem quadratic_always_negative (m k : ℝ) :
  (∀ x : ℝ, x^2 - m*x - k + m < 0) ↔ k > m - m^2/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_negative_l2098_209821


namespace NUMINAMATH_CALUDE_min_value_of_f_l2098_209899

theorem min_value_of_f (x : ℝ) (hx : x < 0) : 
  ∃ (m : ℝ), (∀ y, y < 0 → -y - 2/y ≥ m) ∧ (∃ z, z < 0 ∧ -z - 2/z = m) ∧ m = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2098_209899


namespace NUMINAMATH_CALUDE_flat_fee_is_40_l2098_209864

/-- A hotel pricing structure with a flat fee for the first night and a fixed amount for each additional night. -/
structure HotelPricing where
  flatFee : ℝ
  additionalNightFee : ℝ

/-- Calculate the total cost for a stay given the pricing structure and number of nights. -/
def totalCost (pricing : HotelPricing) (nights : ℕ) : ℝ :=
  pricing.flatFee + pricing.additionalNightFee * (nights - 1)

/-- The flat fee for the first night is $40 given the conditions. -/
theorem flat_fee_is_40 :
  ∃ (pricing : HotelPricing),
    totalCost pricing 4 = 195 ∧
    totalCost pricing 7 = 350 ∧
    pricing.flatFee = 40 := by
  sorry

end NUMINAMATH_CALUDE_flat_fee_is_40_l2098_209864


namespace NUMINAMATH_CALUDE_borrowing_methods_count_l2098_209833

/-- Represents the number of books of each type -/
structure BookCounts where
  physics : Nat
  history : Nat
  mathematics : Nat

/-- Represents the number of students of each type -/
structure StudentCounts where
  science : Nat
  liberal_arts : Nat

/-- Calculates the number of ways to distribute books to students -/
def calculate_borrowing_methods (books : BookCounts) (students : StudentCounts) : Nat :=
  sorry

/-- Theorem stating the correct number of borrowing methods -/
theorem borrowing_methods_count :
  let books := BookCounts.mk 3 2 4
  let students := StudentCounts.mk 4 3
  calculate_borrowing_methods books students = 76 := by
  sorry

end NUMINAMATH_CALUDE_borrowing_methods_count_l2098_209833


namespace NUMINAMATH_CALUDE_room_population_l2098_209861

theorem room_population (P M : ℕ) : 
  (P : ℚ) * (2 / 100) = 1 →  -- 2% of painters are musicians
  (M : ℚ) * (5 / 100) = 1 →  -- 5% of musicians are painters
  P + M - 1 = 69             -- Total people in the room
  := by sorry

end NUMINAMATH_CALUDE_room_population_l2098_209861


namespace NUMINAMATH_CALUDE_statues_painted_l2098_209829

theorem statues_painted (total_paint : ℚ) (paint_per_statue : ℚ) :
  total_paint = 7/16 →
  paint_per_statue = 1/16 →
  (total_paint / paint_per_statue : ℚ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_statues_painted_l2098_209829


namespace NUMINAMATH_CALUDE_problem_statement_l2098_209886

theorem problem_statement (x y z t : ℝ) 
  (eq1 : 3 * x^2 + 3 * x * z + z^2 = 1)
  (eq2 : 3 * y^2 + 3 * y * z + z^2 = 4)
  (eq3 : x^2 - x * y + y^2 = t) :
  t ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2098_209886


namespace NUMINAMATH_CALUDE_prime_factor_count_l2098_209872

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_factor_count (x : ℕ) : 
  is_prime x → 
  (∃ (n : ℕ), 2^22 * x^7 * 11^2 = n ∧ (Nat.factors n).length = 31) → 
  x = 7 :=
sorry

end NUMINAMATH_CALUDE_prime_factor_count_l2098_209872


namespace NUMINAMATH_CALUDE_right_triangle_area_l2098_209825

theorem right_triangle_area (h : ℝ) (angle : ℝ) :
  h = 12 →
  angle = 30 * π / 180 →
  let a := h / 2
  let b := a * Real.sqrt 3
  (1 / 2) * a * b = 18 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2098_209825


namespace NUMINAMATH_CALUDE_money_distribution_l2098_209876

theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 → 
  a + c = 200 → 
  b + c = 320 → 
  c = 20 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2098_209876


namespace NUMINAMATH_CALUDE_gym_attendance_proof_l2098_209800

theorem gym_attendance_proof (W A S : ℕ) : 
  (W + A + S) + 8 = 30 → W + A + S = 22 := by
  sorry

end NUMINAMATH_CALUDE_gym_attendance_proof_l2098_209800


namespace NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l2098_209890

theorem unique_solution_for_prime_equation (p q r t n : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p^2 + q*t = (p + t)^n → 
  p^2 + q*r = t^4 → 
  (p = 2 ∧ q = 7 ∧ r = 11 ∧ t = 3 ∧ n = 2) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l2098_209890


namespace NUMINAMATH_CALUDE_jessie_weight_l2098_209884

/-- Jessie's weight problem -/
theorem jessie_weight (initial_weight lost_weight : ℕ) (h1 : initial_weight = 74) (h2 : lost_weight = 7) :
  initial_weight - lost_weight = 67 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_l2098_209884


namespace NUMINAMATH_CALUDE_f_properties_l2098_209824

-- Define the function f(x) = x³ - 3x² - 9x
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

-- Define the domain
def domain : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

-- Theorem statement
theorem f_properties :
  (∃ (x : ℝ), x ∈ domain ∧ f x = 5 ∧ ∀ (y : ℝ), y ∈ domain → f y ≤ f x) ∧
  (∀ (m : ℝ), ∃ (x : ℝ), x ∈ domain ∧ f x < m) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2098_209824


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l2098_209875

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 64 * Real.pi) :
  A = Real.pi * r^2 → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l2098_209875


namespace NUMINAMATH_CALUDE_share_ratio_l2098_209898

theorem share_ratio (total : ℚ) (a b c : ℚ) : 
  total = 510 →
  b = (1/4) * c →
  a + b + c = total →
  a = 360 →
  a / b = 12 := by
sorry

end NUMINAMATH_CALUDE_share_ratio_l2098_209898


namespace NUMINAMATH_CALUDE_function_range_l2098_209834

/-- Given a real number m and a function f, prove that if there exists x₀ satisfying certain conditions, then m belongs to the specified range. -/
theorem function_range (m : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = Real.sqrt 3 * Real.sin (π * x / m)) :
  (∃ x₀, (f x₀ = Real.sqrt 3 ∨ f x₀ = -Real.sqrt 3) ∧ x₀^2 + (f x₀)^2 < m^2) →
  m < -2 ∨ m > 2 :=
by sorry

end NUMINAMATH_CALUDE_function_range_l2098_209834


namespace NUMINAMATH_CALUDE_three_more_than_twice_x_l2098_209885

/-- The algebraic expression for a number that is 3 more than twice x is 2x + 3. -/
theorem three_more_than_twice_x (x : ℝ) : 2 * x + 3 = 2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_three_more_than_twice_x_l2098_209885


namespace NUMINAMATH_CALUDE_min_value_problem_l2098_209837

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 6*x*y - 1 = 0) :
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 6*a*b - 1 = 0 → x + 2*y ≤ a + 2*b) ∧ 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + 6*a*b - 1 = 0 ∧ x + 2*y = a + 2*b) ∧
  x + 2*y = 2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2098_209837


namespace NUMINAMATH_CALUDE_third_competitor_hotdogs_l2098_209893

/-- The number of hotdogs the third competitor can eat in a given time -/
def hotdogs_eaten_by_third (first_rate : ℕ) (second_multiplier third_multiplier time : ℕ) : ℕ :=
  first_rate * second_multiplier * third_multiplier * time

/-- Theorem: The third competitor eats 300 hotdogs in 5 minutes -/
theorem third_competitor_hotdogs :
  hotdogs_eaten_by_third 10 3 2 5 = 300 := by
  sorry

#eval hotdogs_eaten_by_third 10 3 2 5

end NUMINAMATH_CALUDE_third_competitor_hotdogs_l2098_209893


namespace NUMINAMATH_CALUDE_second_to_first_day_ratio_l2098_209897

/-- Represents the number of pages read on each day --/
structure PagesRead where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Represents the book reading scenario --/
def BookReading (p : PagesRead) : Prop :=
  p.day1 = 63 ∧
  p.day3 = p.day2 + 10 ∧
  p.day4 = 29 ∧
  p.day1 + p.day2 + p.day3 + p.day4 = 354

theorem second_to_first_day_ratio (p : PagesRead) 
  (h : BookReading p) : p.day2 = 2 * p.day1 := by
  sorry

#check second_to_first_day_ratio

end NUMINAMATH_CALUDE_second_to_first_day_ratio_l2098_209897


namespace NUMINAMATH_CALUDE_no_six_numbers_exist_l2098_209874

/-- Represents a six-digit number composed of digits 1 to 6 without repetitions -/
def SixDigitNumber := Fin 6 → Fin 6

/-- Represents a three-digit number composed of digits 1 to 6 without repetitions -/
def ThreeDigitNumber := Fin 3 → Fin 6

/-- Checks if a ThreeDigitNumber can be obtained from a SixDigitNumber by deleting three digits -/
def canBeObtained (six : SixDigitNumber) (three : ThreeDigitNumber) : Prop :=
  ∃ (i j k : Fin 6), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    (∀ m : Fin 3, three m = six (if m < i then m else if m < j then m + 1 else m + 2))

/-- The main theorem stating that the required set of six numbers does not exist -/
theorem no_six_numbers_exist : 
  ¬ ∃ (numbers : Fin 6 → SixDigitNumber),
    (∀ i : Fin 6, Function.Injective (numbers i)) ∧
    (∀ three : ThreeDigitNumber, Function.Injective three → 
      ∃ (i : Fin 6), canBeObtained (numbers i) three) :=
by sorry


end NUMINAMATH_CALUDE_no_six_numbers_exist_l2098_209874


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l2098_209858

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- The property of a circle being symmetric with respect to a line -/
def isSymmetric (c : Circle) (l : Line) : Prop := sorry

theorem circle_symmetry_line (m : ℝ) :
  let c : Circle := { equation := fun x y => x^2 + y^2 + 2*x - 4*y = 0 }
  let l : Line := { equation := fun x y => 3*x + y + m = 0 }
  isSymmetric c l → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l2098_209858


namespace NUMINAMATH_CALUDE_xy_max_value_l2098_209808

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  x * y ≤ 1/8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ x * y = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_xy_max_value_l2098_209808


namespace NUMINAMATH_CALUDE_expression_not_prime_l2098_209853

theorem expression_not_prime :
  ∀ x : ℕ, 0 < x → x < 100 →
  ∃ k : ℕ, 3^x + 5^x + 7^x + 11^x + 13^x + 17^x + 19^x = 3 * k :=
by sorry

end NUMINAMATH_CALUDE_expression_not_prime_l2098_209853


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l2098_209827

theorem polynomial_multiplication (x : ℝ) : (x + 1) * (x^2 - x + 1) = x^3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l2098_209827


namespace NUMINAMATH_CALUDE_parabola_point_distance_l2098_209838

/-- Given a parabola y = -ax²/4 + ax + c and three points on it, 
    prove that if y₁ > y₃ ≥ y₂ and y₂ is the vertex, then |x₁ - x₂| > |x₃ - x₂| -/
theorem parabola_point_distance (a c x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  y₁ = -a * x₁^2 / 4 + a * x₁ + c →
  y₂ = -a * x₂^2 / 4 + a * x₂ + c →
  y₃ = -a * x₃^2 / 4 + a * x₃ + c →
  y₂ = a + c →
  y₁ > y₃ →
  y₃ ≥ y₂ →
  |x₁ - x₂| > |x₃ - x₂| := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l2098_209838


namespace NUMINAMATH_CALUDE_division_problem_l2098_209896

theorem division_problem :
  ∀ n : ℕ,
  (n / 6 = 8) →
  (n % 6 < 6) →
  (n = 6 * 8 + n % 6) →
  (n % 6 ≤ 5 ∧ n % 6 = 5 → n = 53) :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l2098_209896


namespace NUMINAMATH_CALUDE_three_digit_permutations_l2098_209846

/-- The set of digits used in the problem -/
def digits : Finset Nat := {1, 2, 3}

/-- The number of digits used -/
def n : Nat := Finset.card digits

/-- The length of each permutation -/
def k : Nat := 3

/-- The number of permutations of the digits -/
def num_permutations : Nat := Nat.factorial n

theorem three_digit_permutations : num_permutations = 6 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_permutations_l2098_209846


namespace NUMINAMATH_CALUDE_solve_oliver_money_problem_l2098_209819

def oliver_money_problem (initial_amount savings puzzle_cost gift final_amount : ℕ) 
  (frisbee_cost : ℕ) : Prop :=
  initial_amount + savings + gift - puzzle_cost - frisbee_cost = final_amount

theorem solve_oliver_money_problem :
  ∃ (frisbee_cost : ℕ), oliver_money_problem 9 5 3 8 15 frisbee_cost ∧ frisbee_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_oliver_money_problem_l2098_209819


namespace NUMINAMATH_CALUDE_division_result_l2098_209862

theorem division_result (n : ℕ) (h : n = 2011) : 
  (4 * 10^n - 1) / (4 * ((10^n - 1) / 3) + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l2098_209862


namespace NUMINAMATH_CALUDE_false_proposition_l2098_209860

-- Define the lines
def line1 : ℝ → ℝ → Prop := λ x y => 6*x + 2*y - 1 = 0
def line2 : ℝ → ℝ → Prop := λ x y => y = 5 - 3*x
def line3 : ℝ → ℝ → Prop := λ x y => 2*x + 6*y - 4 = 0

-- Define the propositions
def p : Prop := ∀ x y, line1 x y ↔ line2 x y
def q : Prop := ∀ x y, line1 x y → line3 x y

-- Theorem statement
theorem false_proposition : ¬((¬p) ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_false_proposition_l2098_209860


namespace NUMINAMATH_CALUDE_triangle_properties_l2098_209847

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.c / (t.b - t.a) = (Real.sin t.A + Real.sin t.B) / (Real.sin t.A + Real.sin t.C))
  (h2 : Real.sin t.C = 2 * Real.sin t.A)
  (h3 : 1/2 * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3)
  (h4 : t.b = Real.sqrt 3)
  (h5 : t.a * t.c = 1) : 
  t.B = 2 * Real.pi / 3 ∧ 
  t.a = 2 ∧ 
  t.c = 4 ∧ 
  t.a + t.b + t.c = 2 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2098_209847


namespace NUMINAMATH_CALUDE_minervas_stamps_l2098_209814

/-- Given that Lizette has 813 stamps and 125 more stamps than Minerva,
    prove that Minerva has 688 stamps. -/
theorem minervas_stamps (lizette_stamps : ℕ) (difference : ℕ) 
  (h1 : lizette_stamps = 813)
  (h2 : difference = 125)
  (h3 : lizette_stamps = difference + minerva_stamps) :
  minerva_stamps = 688 := by
  sorry

end NUMINAMATH_CALUDE_minervas_stamps_l2098_209814


namespace NUMINAMATH_CALUDE_b_speed_is_13_l2098_209857

-- Define the walking scenario
def walking_scenario (speed_A speed_B initial_distance meeting_time : ℝ) : Prop :=
  speed_A > 0 ∧ speed_B > 0 ∧ initial_distance > 0 ∧ meeting_time > 0 ∧
  speed_A * meeting_time + speed_B * meeting_time = initial_distance

-- Theorem statement
theorem b_speed_is_13 :
  ∀ (speed_B : ℝ),
    walking_scenario 12 speed_B 25 1 →
    speed_B = 13 := by
  sorry

end NUMINAMATH_CALUDE_b_speed_is_13_l2098_209857


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l2098_209823

theorem quadratic_complete_square (r s : ℚ) : 
  (∀ x, 7 * x^2 - 21 * x - 56 = 0 ↔ (x + r)^2 = s) → 
  r + s = 35/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l2098_209823


namespace NUMINAMATH_CALUDE_pencil_weight_l2098_209831

/-- Given that 5 pencils weigh 141.5 grams, prove that one pencil weighs 28.3 grams. -/
theorem pencil_weight (total_weight : ℝ) (num_pencils : ℕ) (h1 : total_weight = 141.5) (h2 : num_pencils = 5) :
  total_weight / num_pencils = 28.3 := by
  sorry

end NUMINAMATH_CALUDE_pencil_weight_l2098_209831


namespace NUMINAMATH_CALUDE_planet_can_be_fully_explored_l2098_209805

/-- Represents a spherical planet -/
structure Planet :=
  (equatorial_length : ℝ)

/-- Represents a rover's exploration path on the planet -/
structure ExplorationPath :=
  (length : ℝ)
  (covers_all_points : Bool)

/-- Checks if an exploration path fully explores the planet -/
def fully_explores (p : Planet) (path : ExplorationPath) : Prop :=
  path.length ≤ 600 ∧ path.covers_all_points = true

/-- Theorem stating that the planet can be fully explored -/
theorem planet_can_be_fully_explored (p : Planet) 
  (h : p.equatorial_length = 400) : 
  ∃ path : ExplorationPath, fully_explores p path :=
sorry

end NUMINAMATH_CALUDE_planet_can_be_fully_explored_l2098_209805


namespace NUMINAMATH_CALUDE_inequality_cubic_quadratic_l2098_209878

theorem inequality_cubic_quadratic (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 > a^2 * b + a * b^2 := by sorry

end NUMINAMATH_CALUDE_inequality_cubic_quadratic_l2098_209878


namespace NUMINAMATH_CALUDE_multiply_twelve_problem_l2098_209801

theorem multiply_twelve_problem (x : ℚ) : 
  (12 * x * 2 = 7899665 - 7899593) → x = 3 := by
sorry

end NUMINAMATH_CALUDE_multiply_twelve_problem_l2098_209801


namespace NUMINAMATH_CALUDE_parabola_area_theorem_l2098_209835

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Theorem: For a parabola y^2 = px with p > 0, focus F on the x-axis, and a slanted line through F
    intersecting the parabola at A and B, if the area of triangle OAB is 2√2 (where O is the origin),
    then p = 4√2. -/
theorem parabola_area_theorem (par : Parabola) (F A B : Point) :
  F.x = par.p / 2 →  -- Focus F is on x-axis
  F.y = 0 →
  (∃ m b : ℝ, A.y = m * A.x + b ∧ B.y = m * B.x + b ∧ F.y = m * F.x + b) →  -- A, B, F are on a slanted line
  A.y^2 = par.p * A.x →  -- A is on the parabola
  B.y^2 = par.p * B.x →  -- B is on the parabola
  abs ((A.x * B.y - B.x * A.y) / 2) = 2 * Real.sqrt 2 →  -- Area of triangle OAB is 2√2
  par.p = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_area_theorem_l2098_209835


namespace NUMINAMATH_CALUDE_gift_cost_increase_l2098_209815

theorem gift_cost_increase (initial_friends : ℕ) (gift_cost : ℕ) (dropouts : ℕ) : 
  initial_friends = 10 → 
  gift_cost = 120 → 
  dropouts = 4 → 
  (gift_cost / (initial_friends - dropouts) : ℚ) - (gift_cost / initial_friends : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_gift_cost_increase_l2098_209815


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l2098_209828

theorem perfect_square_binomial (x : ℝ) : 
  ∃ (a b : ℝ), 16 * x^2 - 40 * x + 25 = (a * x + b)^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l2098_209828


namespace NUMINAMATH_CALUDE_cloth_sold_meters_l2098_209877

/-- Proves that the number of meters of cloth sold is 80 -/
theorem cloth_sold_meters (total_selling_price : ℝ) (profit_per_meter : ℝ) (cost_price_per_meter : ℝ)
  (h1 : total_selling_price = 6900)
  (h2 : profit_per_meter = 20)
  (h3 : cost_price_per_meter = 66.25) :
  (total_selling_price / (cost_price_per_meter + profit_per_meter)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sold_meters_l2098_209877


namespace NUMINAMATH_CALUDE_congruence_solution_l2098_209892

theorem congruence_solution : ∃! n : ℕ, n < 47 ∧ (13 * n) % 47 = 9 % 47 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2098_209892


namespace NUMINAMATH_CALUDE_max_value_abc_l2098_209871

theorem max_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^3) ≤ 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_abc_l2098_209871


namespace NUMINAMATH_CALUDE_tangent_at_one_l2098_209894

/-- A polynomial function of degree 4 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^3 + 1

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 3 * b * x^2

theorem tangent_at_one (a b : ℝ) : 
  (f a b 1 = 0 ∧ f' a b 1 = 0) ↔ (a = 3 ∧ b = -4) := by sorry

end NUMINAMATH_CALUDE_tangent_at_one_l2098_209894


namespace NUMINAMATH_CALUDE_symmetry_of_curves_l2098_209848

/-- The original curve E -/
def E (x y : ℝ) : Prop := x^2 + 2*x*y + y^2 + 3*x + y = 0

/-- The line of symmetry l -/
def l (x y : ℝ) : Prop := 2*x - y - 1 = 0

/-- The symmetric curve E' -/
def E' (x y : ℝ) : Prop := x^2 + 14*x*y + 49*y^2 - 21*x + 103*y + 54 = 0

/-- Theorem stating that E' is symmetric to E with respect to l -/
theorem symmetry_of_curves :
  ∀ (x y x' y' : ℝ),
    E x y →
    l ((x + x') / 2) ((y + y') / 2) →
    E' x' y' :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_curves_l2098_209848


namespace NUMINAMATH_CALUDE_smallest_constant_term_l2098_209887

theorem smallest_constant_term (a b c d e : ℤ) : 
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = -2 ∨ x = 5 ∨ x = 9 ∨ x = -1/2) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 → 
    (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e' = 0 ↔ 
      x = -2 ∨ x = 5 ∨ x = 9 ∨ x = -1/2) → 
    e ≤ e') →
  e = 90 :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_term_l2098_209887


namespace NUMINAMATH_CALUDE_detergent_calculation_l2098_209811

/-- Calculates the amount of detergent in a solution given the ratio of detergent to water and the amount of water -/
def detergent_amount (detergent_ratio : ℚ) (water_ratio : ℚ) (water_amount : ℚ) : ℚ :=
  (detergent_ratio / water_ratio) * water_amount

theorem detergent_calculation :
  let detergent_ratio : ℚ := 1
  let water_ratio : ℚ := 8
  let water_amount : ℚ := 300
  detergent_amount detergent_ratio water_ratio water_amount = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_detergent_calculation_l2098_209811


namespace NUMINAMATH_CALUDE_simplify_expression_l2098_209817

theorem simplify_expression : 2 - (2 / (2 + Real.sqrt 5)) - (2 / (2 - Real.sqrt 5)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2098_209817


namespace NUMINAMATH_CALUDE_unique_a_for_divisible_N_l2098_209809

def N (a : ℕ+) : ℕ := Nat.lcm (2*a.val + 1) (Nat.lcm (2*a.val + 2) (2*a.val + 3))

theorem unique_a_for_divisible_N :
  ∀ a : ℕ+, (2*a.val + 4) ∣ N a → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_a_for_divisible_N_l2098_209809


namespace NUMINAMATH_CALUDE_complex_division_example_l2098_209832

theorem complex_division_example : (1 - 3*I) / (1 + I) = -1 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_example_l2098_209832


namespace NUMINAMATH_CALUDE_original_number_l2098_209826

theorem original_number (x : ℚ) : (3 * (x + 3) - 4) / 3 = 10 → x = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l2098_209826


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l2098_209882

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l2098_209882


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2098_209867

theorem fraction_evaluation : 
  (⌈(21 / 8 : ℚ) - ⌈(35 / 21 : ℚ)⌉⌉ : ℚ) / 
  (⌈(35 / 8 : ℚ) + ⌈(8 * 21 / 35 : ℚ)⌉⌉ : ℚ) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2098_209867


namespace NUMINAMATH_CALUDE_soda_difference_l2098_209841

def julio_orange : ℕ := 4
def julio_grape : ℕ := 7
def mateo_orange : ℕ := 1
def mateo_grape : ℕ := 3
def sophia_orange : ℕ := 6
def sophia_strawberry : ℕ := 5

def orange_soda_volume : ℚ := 2
def grape_soda_volume : ℚ := 2
def sophia_orange_volume : ℚ := 1.5
def sophia_strawberry_volume : ℚ := 2.5

def julio_total : ℚ := julio_orange * orange_soda_volume + julio_grape * grape_soda_volume
def mateo_total : ℚ := mateo_orange * orange_soda_volume + mateo_grape * grape_soda_volume
def sophia_total : ℚ := sophia_orange * sophia_orange_volume + sophia_strawberry * sophia_strawberry_volume

theorem soda_difference :
  (max julio_total (max mateo_total sophia_total)) - (min julio_total (min mateo_total sophia_total)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_soda_difference_l2098_209841


namespace NUMINAMATH_CALUDE_probability_different_suits_l2098_209818

def deck_size : ℕ := 60
def num_suits : ℕ := 4
def cards_per_suit : ℕ := 15

theorem probability_different_suits :
  let prob_diff_suits := (deck_size - cards_per_suit) / (deck_size * (deck_size - 1))
  prob_diff_suits = 45 / 236 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_suits_l2098_209818


namespace NUMINAMATH_CALUDE_undeclared_major_fraction_l2098_209843

/-- The fraction of students who have not declared a major among second- and third-year students -/
theorem undeclared_major_fraction :
  let total_students : ℚ := 1
  let first_year_students : ℚ := 1/3
  let second_year_students : ℚ := 1/3
  let third_year_students : ℚ := 1/3
  let first_year_undeclared : ℚ := 4/5 * first_year_students
  let second_year_declared : ℚ := 1/2 * (first_year_students - first_year_undeclared)
  let second_year_undeclared : ℚ := second_year_students - second_year_declared
  let third_year_undeclared : ℚ := 1/4 * third_year_students
  (second_year_undeclared + third_year_undeclared) / total_students = 23/60 := by
  sorry

end NUMINAMATH_CALUDE_undeclared_major_fraction_l2098_209843


namespace NUMINAMATH_CALUDE_shortest_chord_theorem_l2098_209813

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 4 = 0

/-- The point A on the circle -/
def point_A : ℝ × ℝ := (2, 1)

/-- The length of the shortest chord passing through point A -/
def shortest_chord_length : ℝ := 2

theorem shortest_chord_theorem :
  circle_equation point_A.1 point_A.2 →
  ∃ (x y : ℝ), circle_equation x y ∧ 
    ((x - point_A.1)^2 + (y - point_A.2)^2 = shortest_chord_length^2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_chord_theorem_l2098_209813


namespace NUMINAMATH_CALUDE_herd_size_l2098_209830

theorem herd_size (first_son_fraction : ℚ) (second_son_fraction : ℚ) (third_son_fraction : ℚ) 
  (fourth_son_cows : ℕ) 
  (h1 : first_son_fraction = 2/3) 
  (h2 : second_son_fraction = 1/6) 
  (h3 : third_son_fraction = 1/9) 
  (h4 : fourth_son_cows = 6) :
  ∃ (total_cows : ℕ), 
    total_cows = 108 ∧ 
    (first_son_fraction + second_son_fraction + third_son_fraction < 1) ∧
    (1 - (first_son_fraction + second_son_fraction + third_son_fraction)) * total_cows = fourth_son_cows := by
  sorry

end NUMINAMATH_CALUDE_herd_size_l2098_209830


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l2098_209880

theorem three_digit_number_problem (A B : ℝ) : 
  (100 ≤ A ∧ A < 1000) →  -- A is a three-digit number
  (B = A / 10 ∨ B = A / 100 ∨ B = A / 1000) →  -- B is obtained by placing a decimal point in front of one of A's digits
  (A - B = 478.8) →  -- Given condition
  A = 532 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l2098_209880


namespace NUMINAMATH_CALUDE_four_digit_combinations_l2098_209895

/-- The number of available digits for each position in a four-digit number -/
def available_digits : Fin 4 → ℕ
  | 0 => 9  -- first digit (cannot be 0)
  | 1 => 8  -- second digit
  | 2 => 6  -- third digit
  | 3 => 4  -- fourth digit

/-- The total number of different four-digit numbers that can be formed -/
def total_combinations : ℕ := (available_digits 0) * (available_digits 1) * (available_digits 2) * (available_digits 3)

theorem four_digit_combinations : total_combinations = 1728 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_combinations_l2098_209895


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2098_209804

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2098_209804


namespace NUMINAMATH_CALUDE_max_additional_bricks_l2098_209850

/-- Represents the weight capacity of a truck in terms of bags of sand -/
def sand_capacity : ℕ := 50

/-- Represents the weight capacity of a truck in terms of bricks -/
def brick_capacity : ℕ := 400

/-- Represents the number of bags of sand already in the truck -/
def sand_load : ℕ := 32

/-- Calculates the equivalent number of bricks for a given number of sand bags -/
def sand_to_brick_equiv (sand : ℕ) : ℕ :=
  (brick_capacity * sand) / sand_capacity

theorem max_additional_bricks : 
  sand_to_brick_equiv (sand_capacity - sand_load) = 144 := by
  sorry

end NUMINAMATH_CALUDE_max_additional_bricks_l2098_209850


namespace NUMINAMATH_CALUDE_unique_valid_sequence_l2098_209868

def IsValidSequence (a : ℕ → ℕ) : Prop :=
  (∀ m n : ℕ, m ≠ n → a m ≠ a n) ∧
  (∀ n : ℕ, a n % a (a n) = 0)

theorem unique_valid_sequence :
  ∀ a : ℕ → ℕ, IsValidSequence a → (∀ n : ℕ, a n = n) :=
by sorry

end NUMINAMATH_CALUDE_unique_valid_sequence_l2098_209868


namespace NUMINAMATH_CALUDE_angle_inequality_equivalence_l2098_209802

theorem angle_inequality_equivalence (θ : Real) : 
  (0 < θ ∧ θ < Real.pi / 2) ↔ 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → 
    x^2 * Real.cos θ - x * (1 - x) + 2 * (1 - x)^2 * Real.sin θ > 0) :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_equivalence_l2098_209802

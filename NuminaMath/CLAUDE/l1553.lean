import Mathlib

namespace NUMINAMATH_CALUDE_positive_integer_solutions_l1553_155378

theorem positive_integer_solutions :
  ∀ x y z : ℕ+,
    x < y →
    2 * (x + 1) * (y + 1) - 1 = x * y * z →
    ((x = 1 ∧ y = 3 ∧ z = 5) ∨ (x = 3 ∧ y = 7 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_l1553_155378


namespace NUMINAMATH_CALUDE_field_trip_group_size_l1553_155388

/-- Calculates the number of students in each group excluding the student themselves -/
def students_per_group (total_bread : ℕ) (num_groups : ℕ) (sandwiches_per_student : ℕ) (bread_per_sandwich : ℕ) : ℕ :=
  (total_bread / (num_groups * sandwiches_per_student * bread_per_sandwich)) - 1

/-- Theorem: Given the specified conditions, there are 5 students in each group excluding the student themselves -/
theorem field_trip_group_size :
  students_per_group 120 5 2 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_group_size_l1553_155388


namespace NUMINAMATH_CALUDE_total_handshakes_at_convention_l1553_155361

def num_gremlins : ℕ := 30
def num_imps : ℕ := 25
def num_reconciled_imps : ℕ := 10
def num_unreconciled_imps : ℕ := 15

def handshakes_among_gremlins : ℕ := num_gremlins * (num_gremlins - 1) / 2
def handshakes_among_reconciled_imps : ℕ := num_reconciled_imps * (num_reconciled_imps - 1) / 2
def handshakes_between_gremlins_and_imps : ℕ := num_gremlins * num_imps

theorem total_handshakes_at_convention : 
  handshakes_among_gremlins + handshakes_among_reconciled_imps + handshakes_between_gremlins_and_imps = 1230 := by
  sorry

end NUMINAMATH_CALUDE_total_handshakes_at_convention_l1553_155361


namespace NUMINAMATH_CALUDE_determinant_transformation_l1553_155335

theorem determinant_transformation (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 3 →
  Matrix.det !![p, 5*p + 4*q; r, 5*r + 4*s] = 12 := by
sorry

end NUMINAMATH_CALUDE_determinant_transformation_l1553_155335


namespace NUMINAMATH_CALUDE_zeros_when_m_zero_one_zero_in_interval_l1553_155358

/-- The function f(x) defined in terms of m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (2*m + 1)*x + m*(m + 1)

/-- Theorem stating the zeros of f(x) when m = 0 -/
theorem zeros_when_m_zero :
  ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = 1 ∧ f 0 x₁ = 0 ∧ f 0 x₂ = 0 :=
sorry

/-- Theorem stating the range of m for which f(x) has exactly one zero in (1,3) -/
theorem one_zero_in_interval (m : ℝ) :
  (∃! x, 1 < x ∧ x < 3 ∧ f m x = 0) ↔ (0 < m ∧ m ≤ 1) ∨ (2 ≤ m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_zeros_when_m_zero_one_zero_in_interval_l1553_155358


namespace NUMINAMATH_CALUDE_second_month_sale_l1553_155362

theorem second_month_sale (
  average_sale : ℕ)
  (month1_sale : ℕ)
  (month3_sale : ℕ)
  (month4_sale : ℕ)
  (month5_sale : ℕ)
  (month6_sale : ℕ)
  (h1 : average_sale = 6500)
  (h2 : month1_sale = 6635)
  (h3 : month3_sale = 7230)
  (h4 : month4_sale = 6562)
  (h5 : month6_sale = 4791)
  : ∃ (month2_sale : ℕ),
    month2_sale = 13782 ∧
    (month1_sale + month2_sale + month3_sale + month4_sale + month5_sale + month6_sale) / 6 = average_sale :=
by sorry

end NUMINAMATH_CALUDE_second_month_sale_l1553_155362


namespace NUMINAMATH_CALUDE_function_properties_no_zeros_l1553_155317

noncomputable section

def f (a : ℝ) (x : ℝ) := a * Real.log x - x
def g (a : ℝ) (x : ℝ) := a * Real.exp x - x

theorem function_properties (a : ℝ) (ha : a > 0) :
  (∀ x > 1, ∀ y > x, f a y < f a x) ∧
  (∃ x > 2, ∀ y > 2, g a x ≤ g a y) →
  a ∈ Set.Ioo 0 (1 / Real.exp 2) :=
sorry

theorem no_zeros (a : ℝ) (ha : a > 0) :
  (∀ x > 0, f a x ≠ 0) ∧ (∀ x, g a x ≠ 0) →
  a ∈ Set.Ioo (1 / Real.exp 1) (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_function_properties_no_zeros_l1553_155317


namespace NUMINAMATH_CALUDE_student_count_equality_l1553_155310

/-- Proves that the number of students in class A equals the number of students in class C
    given the average ages of each class and the overall average age. -/
theorem student_count_equality (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (14 * a + 13 * b + 12 * c) / (a + b + c) = 13 → a = c := by
  sorry

end NUMINAMATH_CALUDE_student_count_equality_l1553_155310


namespace NUMINAMATH_CALUDE_quarters_spent_l1553_155398

def initial_quarters : ℕ := 760
def remaining_quarters : ℕ := 342

theorem quarters_spent : initial_quarters - remaining_quarters = 418 := by
  sorry

end NUMINAMATH_CALUDE_quarters_spent_l1553_155398


namespace NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_5_l1553_155332

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem product_of_digits_not_divisible_by_5 (numbers : List ℕ) :
  numbers = [3640, 3855, 3922, 4025, 4120] →
  (∃ n ∈ numbers, ¬ is_divisible_by_5 n) →
  (∃ n ∈ numbers, ¬ is_divisible_by_5 n ∧ hundreds_digit n * tens_digit n = 18) :=
by sorry

end NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_5_l1553_155332


namespace NUMINAMATH_CALUDE_magic_act_disappearance_ratio_l1553_155318

theorem magic_act_disappearance_ratio :
  ∀ (total_performances : ℕ) 
    (total_reappearances : ℕ) 
    (double_reappearance_prob : ℚ),
  total_performances = 100 →
  total_reappearances = 110 →
  double_reappearance_prob = 1/5 →
  (total_performances - 
   (total_reappearances - total_performances * double_reappearance_prob)) / 
   total_performances = 1/10 := by
sorry

end NUMINAMATH_CALUDE_magic_act_disappearance_ratio_l1553_155318


namespace NUMINAMATH_CALUDE_picasso_prints_probability_l1553_155363

/-- The probability of arranging 4 specific items consecutively in a random arrangement of n items -/
def consecutive_probability (n : ℕ) (k : ℕ) : ℚ :=
  if n < k then 0
  else (k.factorial * (n - k + 1).factorial) / n.factorial

theorem picasso_prints_probability :
  consecutive_probability 12 4 = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_picasso_prints_probability_l1553_155363


namespace NUMINAMATH_CALUDE_geometric_sum_half_five_l1553_155391

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_half_five :
  geometric_sum (1/2) (1/2) 5 = 31/32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_half_five_l1553_155391


namespace NUMINAMATH_CALUDE_rectangle_area_similarity_l1553_155334

theorem rectangle_area_similarity (R1_side : ℝ) (R1_area : ℝ) (R2_diagonal : ℝ) :
  R1_side = 3 →
  R1_area = 24 →
  R2_diagonal = 20 →
  ∃ (R2_area : ℝ), R2_area = 3200 / 73 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_similarity_l1553_155334


namespace NUMINAMATH_CALUDE_sum_is_integer_four_or_negative_four_l1553_155377

theorem sum_is_integer_four_or_negative_four 
  (x y z t : ℝ) 
  (h : x / (y + z + t) = y / (z + t + x) ∧ 
       y / (z + t + x) = z / (t + x + y) ∧ 
       z / (t + x + y) = t / (x + y + z)) : 
  (x + y) / (z + t) + (y + z) / (t + x) + (z + t) / (x + y) + (t + x) / (y + z) = 4 ∨
  (x + y) / (z + t) + (y + z) / (t + x) + (z + t) / (x + y) + (t + x) / (y + z) = -4 :=
by sorry

end NUMINAMATH_CALUDE_sum_is_integer_four_or_negative_four_l1553_155377


namespace NUMINAMATH_CALUDE_faster_watch_gain_rate_l1553_155303

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ

/-- Calculates the difference in minutes between two times -/
def timeDifference (t1 t2 : Time) : ℕ := sorry

/-- Calculates the number of hours between two times -/
def hoursBetween (t1 t2 : Time) : ℕ := sorry

theorem faster_watch_gain_rate (alarmSetTime correctAlarmTime fasterAlarmTime : Time) 
  (h1 : alarmSetTime = ⟨22, 0⟩)  -- Alarm set at 10:00 PM
  (h2 : correctAlarmTime = ⟨4, 0⟩)  -- Correct watch shows 4:00 AM
  (h3 : fasterAlarmTime = ⟨4, 12⟩)  -- Faster watch shows 4:12 AM
  : (timeDifference correctAlarmTime fasterAlarmTime) / 
    (hoursBetween alarmSetTime correctAlarmTime) = 2 := by sorry

end NUMINAMATH_CALUDE_faster_watch_gain_rate_l1553_155303


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1553_155354

theorem sum_of_reciprocals (x y : ℚ) :
  (1 / x + 1 / y = 4) → (1 / x - 1 / y = -6) → (x + y = -4 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1553_155354


namespace NUMINAMATH_CALUDE_quadratic_equation_completion_square_l1553_155397

theorem quadratic_equation_completion_square (x m n : ℝ) : 
  (9 * x^2 - 36 * x - 81 = 0) → 
  ((x + m)^2 = n) →
  (m + n = 11) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_completion_square_l1553_155397


namespace NUMINAMATH_CALUDE_highest_power_divisibility_l1553_155371

theorem highest_power_divisibility (n : ℕ) : 
  (∃ k : ℕ, (1991 : ℕ)^k ∣ 1990^(1991^1002) + 1992^(1501^1901)) ∧ 
  (∀ m : ℕ, m > 1001 → ¬((1991 : ℕ)^m ∣ 1990^(1991^1002) + 1992^(1501^1901))) :=
by sorry

end NUMINAMATH_CALUDE_highest_power_divisibility_l1553_155371


namespace NUMINAMATH_CALUDE_sum_product_identity_l1553_155355

theorem sum_product_identity (a b : ℝ) (h : a + b = a * b) :
  (a^3 + b^3 - a^3 * b^3)^3 + 27 * a^6 * b^6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_identity_l1553_155355


namespace NUMINAMATH_CALUDE_min_vertical_distance_l1553_155367

/-- The absolute value function -/
def abs_func (x : ℝ) : ℝ := |x|

/-- The quadratic function -/
def quad_func (x : ℝ) : ℝ := -x^2 - 4*x - 3

/-- The vertical distance between the two functions -/
def vert_distance (x : ℝ) : ℝ := |abs_func x - quad_func x|

theorem min_vertical_distance :
  ∃ (min_dist : ℝ), min_dist = 3 ∧ ∀ (x : ℝ), vert_distance x ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l1553_155367


namespace NUMINAMATH_CALUDE_johns_height_l1553_155381

/-- Given the heights of various people and their relationships, prove John's height. -/
theorem johns_height (carl becky amy helen angela tom mary john : ℝ) 
  (h1 : carl = 120)
  (h2 : becky = 2 * carl)
  (h3 : amy = 1.2 * becky)
  (h4 : helen = amy + 3)
  (h5 : angela = helen + 4)
  (h6 : tom = angela - 70)
  (h7 : mary = 2 * tom)
  (h8 : john = 1.5 * mary) : 
  john = 675 := by sorry

end NUMINAMATH_CALUDE_johns_height_l1553_155381


namespace NUMINAMATH_CALUDE_correct_reasoning_definitions_l1553_155366

-- Define the types of reasoning
inductive ReasoningType
  | Inductive
  | Deductive
  | Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
  | PartToWhole
  | GeneralToSpecific
  | SpecificToSpecific

-- Define the relationship between reasoning types and directions
def reasoningDirection (t : ReasoningType) : ReasoningDirection :=
  match t with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating the correct definitions of reasoning types
theorem correct_reasoning_definitions :
  (reasoningDirection ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (reasoningDirection ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (reasoningDirection ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
by sorry

end NUMINAMATH_CALUDE_correct_reasoning_definitions_l1553_155366


namespace NUMINAMATH_CALUDE_expression_evaluation_l1553_155320

theorem expression_evaluation (a b : ℚ) (ha : a = 3/4) (hb : b = 4/3) :
  let numerator := (a/b + b/a + 2) * ((a+b)/(2*a) - b/(a+b))
  let denominator := (a + 2*b + b^2/a) * (a/(a+b) + b/(a-b))
  numerator / denominator = -7/24 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1553_155320


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1553_155339

theorem tan_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1553_155339


namespace NUMINAMATH_CALUDE_sound_propagation_at_10C_l1553_155328

-- Define the relationship between temperature and speed of sound
def speed_of_sound (temp : Int) : Int :=
  match temp with
  | -20 => 318
  | -10 => 324
  | 0 => 330
  | 10 => 336
  | 20 => 342
  | 30 => 348
  | _ => 0  -- For temperatures not in the data set

-- Theorem statement
theorem sound_propagation_at_10C :
  speed_of_sound 10 * 4 = 1344 := by
  sorry


end NUMINAMATH_CALUDE_sound_propagation_at_10C_l1553_155328


namespace NUMINAMATH_CALUDE_lou_senior_first_cookies_l1553_155309

/-- Represents the cookie jar situation --/
structure CookieJar where
  total : ℕ
  louSeniorFirst : ℕ
  louSeniorSecond : ℕ
  louieJunior : ℕ
  remaining : ℕ

/-- The cookie jar problem --/
def cookieJarProblem : CookieJar :=
  { total := 22
  , louSeniorFirst := 3  -- This is what we want to prove
  , louSeniorSecond := 1
  , louieJunior := 7
  , remaining := 11 }

/-- Theorem stating that Lou Senior took 3 cookies the first time --/
theorem lou_senior_first_cookies :
  cookieJarProblem.total - cookieJarProblem.louSeniorFirst - 
  cookieJarProblem.louSeniorSecond - cookieJarProblem.louieJunior = 
  cookieJarProblem.remaining :=
by sorry

end NUMINAMATH_CALUDE_lou_senior_first_cookies_l1553_155309


namespace NUMINAMATH_CALUDE_valerie_light_bulb_purchase_l1553_155352

/-- Calculates the money left over after buying light bulbs --/
def money_left_over (small_bulbs : ℕ) (large_bulbs : ℕ) (small_cost : ℕ) (large_cost : ℕ) (total_money : ℕ) : ℕ :=
  total_money - (small_bulbs * small_cost + large_bulbs * large_cost)

/-- Theorem: Valerie will have $24 left over after buying light bulbs --/
theorem valerie_light_bulb_purchase :
  money_left_over 3 1 8 12 60 = 24 := by
  sorry

end NUMINAMATH_CALUDE_valerie_light_bulb_purchase_l1553_155352


namespace NUMINAMATH_CALUDE_square_sum_inequality_l1553_155375

theorem square_sum_inequality (a b x y : ℝ) : (a^2 + b^2) * (x^2 + y^2) ≥ (a*x + b*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l1553_155375


namespace NUMINAMATH_CALUDE_paperclip_production_l1553_155330

/-- Given that 8 identical machines can produce 560 paperclips per minute,
    prove that 12 machines running at the same rate will produce 5040 paperclips in 6 minutes. -/
theorem paperclip_production 
  (rate : ℕ → ℕ → ℕ) -- rate function: number of machines → minutes → number of paperclips
  (h1 : rate 8 1 = 560) -- 8 machines produce 560 paperclips in 1 minute
  (h2 : ∀ n m, rate n m = n * rate 1 m) -- machines work at the same rate
  (h3 : ∀ n m k, rate n (m * k) = k * rate n m) -- linear scaling with time
  : rate 12 6 = 5040 :=
by sorry

end NUMINAMATH_CALUDE_paperclip_production_l1553_155330


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1553_155389

theorem quadratic_inequality (x : ℝ) : -3 * x^2 + 5 * x + 4 < 0 ↔ -4/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1553_155389


namespace NUMINAMATH_CALUDE_sequence_properties_l1553_155324

def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 2 * a n + n

def b (n : ℕ) (a : ℕ → ℝ) : ℝ := n * (1 - a n)

def geometric_sequence (u : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, u (n + 1) = r * u n

def sum_of_sequence (u : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum u

theorem sequence_properties (a : ℕ → ℝ) :
  (∀ n : ℕ, S n a = 2 * a n + n) →
  (geometric_sequence (λ n => a n - 1)) ∧
  (∀ n : ℕ, sum_of_sequence (b · a) n = (n - 1) * 2^(n + 1) + 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1553_155324


namespace NUMINAMATH_CALUDE_work_distance_is_ten_l1553_155337

/-- Calculates the one-way distance to work given gas tank capacity, remaining fuel fraction, and fuel efficiency. -/
def distance_to_work (tank_capacity : ℚ) (remaining_fraction : ℚ) (miles_per_gallon : ℚ) : ℚ :=
  (tank_capacity * (1 - remaining_fraction) * miles_per_gallon) / 2

/-- Proves that given the specified conditions, Jim's work is 10 miles away from his house. -/
theorem work_distance_is_ten :
  let tank_capacity : ℚ := 12
  let remaining_fraction : ℚ := 2/3
  let miles_per_gallon : ℚ := 5
  distance_to_work tank_capacity remaining_fraction miles_per_gallon = 10 := by
  sorry


end NUMINAMATH_CALUDE_work_distance_is_ten_l1553_155337


namespace NUMINAMATH_CALUDE_certain_number_exists_l1553_155333

theorem certain_number_exists : ∃ N : ℝ, (5/6 : ℝ) * N = (5/16 : ℝ) * N + 150 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l1553_155333


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1553_155304

theorem algebraic_expression_equality (a b : ℝ) : 
  (2*a + (1/2)*b)^2 - 4*(a^2 + b^2) = (2*a + (1/2)*b)^2 - 4*(a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1553_155304


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1553_155373

theorem arithmetic_calculation : -16 - (-12) - 24 + 18 = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1553_155373


namespace NUMINAMATH_CALUDE_min_value_of_f_l1553_155379

/-- The quadratic function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1553_155379


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1553_155386

theorem hyperbola_focal_length (x y : ℝ) :
  x^2 / 7 - y^2 / 3 = 1 → 2 * Real.sqrt 10 = 2 * Real.sqrt (7 + 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1553_155386


namespace NUMINAMATH_CALUDE_unknown_cube_edge_length_l1553_155364

/-- The edge length of the unknown cube -/
def x : ℝ := 6

/-- The volume of a cube given its edge length -/
def cube_volume (edge : ℝ) : ℝ := edge ^ 3

theorem unknown_cube_edge_length :
  let cube1_edge : ℝ := 8
  let cube2_edge : ℝ := 10
  let new_cube_edge : ℝ := 12
  cube_volume new_cube_edge = cube_volume cube1_edge + cube_volume cube2_edge + cube_volume x :=
by sorry

end NUMINAMATH_CALUDE_unknown_cube_edge_length_l1553_155364


namespace NUMINAMATH_CALUDE_contrapositive_geometric_sequence_l1553_155353

/-- A sequence (a, b, c) is geometric if there exists a common ratio r such that b = ar and c = br -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The main theorem: The contrapositive of "If (a,b,c) is geometric, then b^2 = ac" 
    is equivalent to "If b^2 ≠ ac, then (a,b,c) is not geometric" -/
theorem contrapositive_geometric_sequence (a b c : ℝ) :
  (¬(b^2 = a*c) → ¬(IsGeometricSequence a b c)) ↔
  (IsGeometricSequence a b c → b^2 = a*c) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_geometric_sequence_l1553_155353


namespace NUMINAMATH_CALUDE_difference_of_reciprocals_l1553_155349

theorem difference_of_reciprocals (p q : ℚ) 
  (hp : 4 / p = 8) (hq : 4 / q = 18) : p - q = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_reciprocals_l1553_155349


namespace NUMINAMATH_CALUDE_tire_price_problem_l1553_155314

theorem tire_price_problem (total_cost : ℝ) (fifth_tire_cost : ℝ) :
  total_cost = 485 →
  fifth_tire_cost = 5 →
  ∃ (regular_price : ℝ),
    4 * regular_price + fifth_tire_cost = total_cost ∧
    regular_price = 120 := by
  sorry

end NUMINAMATH_CALUDE_tire_price_problem_l1553_155314


namespace NUMINAMATH_CALUDE_problem_I_problem_II_l1553_155374

theorem problem_I (α : Real) (h : α = π / 6) : 
  (2 * Real.sin (π + α) * Real.cos (π - α) - Real.cos (π + α)) / 
  (1 + Real.sin α ^ 2 + Real.sin (π - α) - Real.cos (π + α) ^ 2) = Real.sqrt 3 := by
  sorry

theorem problem_II (α : Real) (h : Real.tan α / (Real.tan α - 6) = -1) : 
  (2 * Real.cos α - 3 * Real.sin α) / (3 * Real.cos α + 4 * Real.sin α) = -7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_I_problem_II_l1553_155374


namespace NUMINAMATH_CALUDE_jennys_score_is_14_total_questions_correct_l1553_155321

/-- Represents a quiz with a specific scoring system -/
structure Quiz where
  totalQuestions : ℕ
  correctAnswers : ℕ
  incorrectAnswers : ℕ
  unansweredQuestions : ℕ
  correctPoints : ℚ
  incorrectPoints : ℚ

/-- Calculates the total score for a given quiz -/
def calculateScore (q : Quiz) : ℚ :=
  q.correctPoints * q.correctAnswers + q.incorrectPoints * q.incorrectAnswers

/-- Jenny's quiz results -/
def jennysQuiz : Quiz :=
  { totalQuestions := 25
    correctAnswers := 16
    incorrectAnswers := 4
    unansweredQuestions := 5
    correctPoints := 1
    incorrectPoints := -1/2 }

/-- Theorem stating that Jenny's quiz score is 14 -/
theorem jennys_score_is_14 : calculateScore jennysQuiz = 14 := by
  sorry

/-- Theorem verifying the total number of questions -/
theorem total_questions_correct :
  jennysQuiz.correctAnswers + jennysQuiz.incorrectAnswers + jennysQuiz.unansweredQuestions =
  jennysQuiz.totalQuestions := by
  sorry

end NUMINAMATH_CALUDE_jennys_score_is_14_total_questions_correct_l1553_155321


namespace NUMINAMATH_CALUDE_initial_average_production_l1553_155392

theorem initial_average_production 
  (n : ℕ) 
  (today_production : ℕ) 
  (new_average : ℚ) 
  (h1 : n = 8) 
  (h2 : today_production = 95) 
  (h3 : new_average = 55) : 
  (n : ℚ) * (n * new_average - today_production) / (n * (n + 1)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_production_l1553_155392


namespace NUMINAMATH_CALUDE_thumbtack_probability_estimate_l1553_155307

-- Define the structure for the frequency table entry
structure FrequencyEntry :=
  (throws : ℕ)
  (touchingGround : ℕ)
  (frequency : ℚ)

-- Define the frequency table
def frequencyTable : List FrequencyEntry := [
  ⟨40, 20, 1/2⟩,
  ⟨120, 50, 417/1000⟩,
  ⟨320, 146, 456/1000⟩,
  ⟨480, 219, 456/1000⟩,
  ⟨720, 328, 456/1000⟩,
  ⟨800, 366, 458/1000⟩,
  ⟨920, 421, 458/1000⟩,
  ⟨1000, 463, 463/1000⟩
]

-- Define the function to estimate the probability
def estimateProbability (table : List FrequencyEntry) : ℚ :=
  -- Implementation details omitted
  sorry

-- Theorem statement
theorem thumbtack_probability_estimate :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |estimateProbability frequencyTable - 46/100| < ε :=
sorry

end NUMINAMATH_CALUDE_thumbtack_probability_estimate_l1553_155307


namespace NUMINAMATH_CALUDE_binomial_30_3_squared_l1553_155344

theorem binomial_30_3_squared : (Nat.choose 30 3)^2 = 16483600 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_squared_l1553_155344


namespace NUMINAMATH_CALUDE_smaller_cube_edge_length_l1553_155356

theorem smaller_cube_edge_length :
  ∀ (s : ℝ),
  (8 : ℝ) * s^3 = 1000 →
  s = 5 :=
by sorry

end NUMINAMATH_CALUDE_smaller_cube_edge_length_l1553_155356


namespace NUMINAMATH_CALUDE_sin_equal_of_sum_pi_l1553_155322

theorem sin_equal_of_sum_pi (α β : Real) (h : α + β = Real.pi) : Real.sin α = Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_sin_equal_of_sum_pi_l1553_155322


namespace NUMINAMATH_CALUDE_jony_start_block_l1553_155306

/-- Represents Jony's walk along Sunrise Boulevard -/
structure JonyWalk where
  walkTime : ℕ            -- Walking time in minutes
  speed : ℕ               -- Speed in meters per minute
  blockLength : ℕ         -- Length of each block in meters
  turnAroundBlock : ℕ     -- Block number where Jony turns around
  stopBlock : ℕ           -- Block number where Jony stops

/-- Calculates the starting block number for Jony's walk -/
def calculateStartBlock (walk : JonyWalk) : ℕ :=
  sorry

/-- Theorem stating that given the conditions of Jony's walk, his starting block is 10 -/
theorem jony_start_block :
  let walk : JonyWalk := {
    walkTime := 40,
    speed := 100,
    blockLength := 40,
    turnAroundBlock := 90,
    stopBlock := 70
  }
  calculateStartBlock walk = 10 := by
  sorry

end NUMINAMATH_CALUDE_jony_start_block_l1553_155306


namespace NUMINAMATH_CALUDE_max_students_distribution_l1553_155387

theorem max_students_distribution (num_pens num_pencils : ℕ) :
  let max_students := Nat.gcd num_pens num_pencils
  ∃ (pens_per_student pencils_per_student : ℕ),
    num_pens = max_students * pens_per_student ∧
    num_pencils = max_students * pencils_per_student ∧
    ∀ (n : ℕ),
      (∃ (p q : ℕ), num_pens = n * p ∧ num_pencils = n * q) →
      n ≤ max_students :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l1553_155387


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l1553_155319

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y : ℤ), x^2 - 7*y = 10 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l1553_155319


namespace NUMINAMATH_CALUDE_simplify_expression_l1553_155357

theorem simplify_expression (a : ℝ) : 3 * a^5 * (4 * a^7) = 12 * a^12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1553_155357


namespace NUMINAMATH_CALUDE_min_value_of_ab_l1553_155393

theorem min_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + 4 * b + 5) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = x + 4 * y + 5 → a * b ≤ x * y :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_ab_l1553_155393


namespace NUMINAMATH_CALUDE_rat_value_l1553_155384

/-- Represents the value of a letter based on its position in the alphabet -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'b' => 2
  | 'c' => 3
  | 'd' => 4
  | 'e' => 5
  | 'f' => 6
  | 'g' => 7
  | 'h' => 8
  | 'i' => 9
  | 'j' => 10
  | 'k' => 11
  | 'l' => 12
  | 'm' => 13
  | 'n' => 14
  | 'o' => 15
  | 'p' => 16
  | 'q' => 17
  | 'r' => 18
  | 's' => 19
  | 't' => 20
  | 'u' => 21
  | 'v' => 22
  | 'w' => 23
  | 'x' => 24
  | 'y' => 25
  | 'z' => 26
  | _ => 0

/-- Calculates the number value of a word -/
def word_value (w : String) : ℕ :=
  (w.toList.map letter_value).sum * w.length

/-- Theorem: The number value of the word "rat" is 117 -/
theorem rat_value : word_value "rat" = 117 := by
  sorry

end NUMINAMATH_CALUDE_rat_value_l1553_155384


namespace NUMINAMATH_CALUDE_elberta_amount_l1553_155327

def granny_smith : ℕ := 63

def anjou : ℕ := granny_smith / 3

def elberta : ℕ := anjou + 2

theorem elberta_amount : elberta = 23 := by
  sorry

end NUMINAMATH_CALUDE_elberta_amount_l1553_155327


namespace NUMINAMATH_CALUDE_solution_triplets_l1553_155338

theorem solution_triplets (x y z : ℝ) :
  (2 * x^3 + 1 = 3 * z * x) ∧
  (2 * y^3 + 1 = 3 * x * y) ∧
  (2 * z^3 + 1 = 3 * y * z) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_triplets_l1553_155338


namespace NUMINAMATH_CALUDE_sam_initial_yellow_marbles_l1553_155311

/-- The number of yellow marbles Sam had initially -/
def initial_yellow_marbles : ℝ := 86.0

/-- The number of yellow marbles Joan gave to Sam -/
def joan_yellow_marbles : ℝ := 25.0

/-- The total number of yellow marbles Sam has now -/
def total_yellow_marbles : ℝ := 111

theorem sam_initial_yellow_marbles :
  initial_yellow_marbles + joan_yellow_marbles = total_yellow_marbles :=
by sorry

end NUMINAMATH_CALUDE_sam_initial_yellow_marbles_l1553_155311


namespace NUMINAMATH_CALUDE_combined_wave_amplitude_l1553_155385

noncomputable def y₁ (t : ℝ) : ℝ := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y₂ (t : ℝ) : ℝ := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4)
noncomputable def y (t : ℝ) : ℝ := y₁ t + y₂ t

theorem combined_wave_amplitude :
  ∃ (A : ℝ) (φ : ℝ), ∀ t, y t = A * Real.sin (100 * Real.pi * t + φ) ∧ A = 3 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_combined_wave_amplitude_l1553_155385


namespace NUMINAMATH_CALUDE_marble_weight_l1553_155395

theorem marble_weight (marble_weight : ℚ) (car_weight : ℚ) : 
  (9 * marble_weight = 4 * car_weight) →
  (3 * car_weight = 36) →
  marble_weight = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_marble_weight_l1553_155395


namespace NUMINAMATH_CALUDE_larger_solid_volume_is_4_point_5_l1553_155396

-- Define the rectangular prism
def rectangular_prism (length width height : ℝ) := length * width * height

-- Define a plane that cuts the prism
structure cutting_plane (length width height : ℝ) :=
  (passes_through_vertex : Bool)
  (passes_through_midpoint_edge1 : Bool)
  (passes_through_midpoint_edge2 : Bool)

-- Define the volume of the larger solid resulting from the cut
def larger_solid_volume (length width height : ℝ) (plane : cutting_plane length width height) : ℝ :=
  sorry

-- Theorem statement
theorem larger_solid_volume_is_4_point_5 :
  ∀ (plane : cutting_plane 2 1 3),
    plane.passes_through_vertex = true ∧
    plane.passes_through_midpoint_edge1 = true ∧
    plane.passes_through_midpoint_edge2 = true →
    larger_solid_volume 2 1 3 plane = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_larger_solid_volume_is_4_point_5_l1553_155396


namespace NUMINAMATH_CALUDE_intersection_x_sum_l1553_155329

/-- The sum of x-coordinates of intersection points of two congruences -/
theorem intersection_x_sum : ∃ (S : Finset ℤ),
  (∀ x ∈ S, ∃ y : ℤ, 
    (y ≡ 7*x + 3 [ZMOD 20] ∧ y ≡ 13*x + 17 [ZMOD 20]) ∧
    (x ≥ 0 ∧ x < 20)) ∧
  (∀ x : ℤ, x ≥ 0 → x < 20 →
    (∃ y : ℤ, y ≡ 7*x + 3 [ZMOD 20] ∧ y ≡ 13*x + 17 [ZMOD 20]) →
    x ∈ S) ∧
  S.sum id = 12 :=
sorry

end NUMINAMATH_CALUDE_intersection_x_sum_l1553_155329


namespace NUMINAMATH_CALUDE_different_color_probability_l1553_155312

theorem different_color_probability (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ)
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 3)
  (h3 : black_balls = 1) :
  (white_balls * black_balls) / ((total_balls * (total_balls - 1)) / 2) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_different_color_probability_l1553_155312


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1553_155346

theorem polynomial_remainder_theorem (a b : ℚ) : 
  let f : ℚ → ℚ := λ x ↦ a * x^3 - 6 * x^2 + b * x - 5
  (f 2 = 3 ∧ f (-1) = 7) → (a = -2/3 ∧ b = -52/3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1553_155346


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1553_155336

theorem polynomial_divisibility (n : ℕ) (hn : n > 0) :
  ∃ q : Polynomial ℝ, x^(2*n+1) - (2*n+1)*x^(n+1) + (2*n+1)*x^n - 1 = (x - 1)^3 * q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1553_155336


namespace NUMINAMATH_CALUDE_sin_675_degrees_l1553_155326

theorem sin_675_degrees : Real.sin (675 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_675_degrees_l1553_155326


namespace NUMINAMATH_CALUDE_difference_of_squares_l1553_155372

theorem difference_of_squares : 435^2 - 365^2 = 56000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1553_155372


namespace NUMINAMATH_CALUDE_cos_660_degrees_l1553_155365

theorem cos_660_degrees : Real.cos (660 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_660_degrees_l1553_155365


namespace NUMINAMATH_CALUDE_parkway_elementary_girls_not_soccer_l1553_155394

theorem parkway_elementary_girls_not_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (boys_soccer_percentage : ℚ) :
  total_students = 500 →
  boys = 350 →
  soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  (total_students - boys) - (soccer_players - (boys_soccer_percentage * soccer_players).floor) = 115 :=
by sorry

end NUMINAMATH_CALUDE_parkway_elementary_girls_not_soccer_l1553_155394


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1553_155345

theorem complex_number_quadrant : 
  let z : ℂ := (5 * Complex.I) / (1 - 2 * Complex.I)
  (z.re < 0 ∧ z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1553_155345


namespace NUMINAMATH_CALUDE_correct_requirements_l1553_155383

/-- A cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstInningOvers : ℕ
  runsScored : ℕ
  wicketsLost : ℕ
  runRate : ℚ
  targetScore : ℕ

/-- Calculate the required run rate and partnership score -/
def calculateRequirements (game : CricketGame) : ℚ × ℕ :=
  let remainingOvers := game.totalOvers - game.firstInningOvers
  let remainingRuns := game.targetScore - game.runsScored
  let requiredRunRate := remainingRuns / remainingOvers
  let requiredPartnership := remainingRuns
  (requiredRunRate, requiredPartnership)

/-- Theorem stating the correct calculation of requirements -/
theorem correct_requirements (game : CricketGame) 
    (h1 : game.totalOvers = 50)
    (h2 : game.firstInningOvers = 10)
    (h3 : game.runsScored = 32)
    (h4 : game.wicketsLost = 3)
    (h5 : game.runRate = 32/10)
    (h6 : game.targetScore = 282) :
    calculateRequirements game = (25/4, 250) := by
  sorry

#eval calculateRequirements {
  totalOvers := 50,
  firstInningOvers := 10,
  runsScored := 32,
  wicketsLost := 3,
  runRate := 32/10,
  targetScore := 282
}

end NUMINAMATH_CALUDE_correct_requirements_l1553_155383


namespace NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l1553_155350

theorem tagged_fish_in_second_catch 
  (total_fish : ℕ) 
  (initially_tagged : ℕ) 
  (second_catch : ℕ) 
  (h1 : total_fish = 1500) 
  (h2 : initially_tagged = 60) 
  (h3 : second_catch = 50) :
  (initially_tagged : ℚ) / total_fish * second_catch = 2 := by
  sorry

end NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l1553_155350


namespace NUMINAMATH_CALUDE_point_on_y_axis_equal_distance_to_axes_l1553_155347

-- Define point P with parameter a
def P (a : ℝ) : ℝ × ℝ := (2 + a, 3 * a - 6)

-- Theorem for part 1
theorem point_on_y_axis (a : ℝ) :
  P a = (0, -12) ↔ (P a).1 = 0 :=
sorry

-- Theorem for part 2
theorem equal_distance_to_axes (a : ℝ) :
  (P a = (6, 6) ∨ P a = (3, -3)) ↔ abs (P a).1 = abs (P a).2 :=
sorry

end NUMINAMATH_CALUDE_point_on_y_axis_equal_distance_to_axes_l1553_155347


namespace NUMINAMATH_CALUDE_sun_radius_scientific_notation_l1553_155340

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

/-- The sun's radius in kilometers -/
def sun_radius_km : ℝ := 696000

/-- The sun's radius in meters -/
def sun_radius_m : ℝ := sun_radius_km * km_to_m

theorem sun_radius_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), a ≥ 1 ∧ a < 10 ∧ sun_radius_m = a * (10 : ℝ) ^ n ∧ a = 6.96 ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_sun_radius_scientific_notation_l1553_155340


namespace NUMINAMATH_CALUDE_air_conditioner_costs_and_minimum_cost_l1553_155341

/-- Represents the cost and quantity of air conditioners -/
structure AirConditioner :=
  (costA : ℕ) -- Cost of type A
  (costB : ℕ) -- Cost of type B
  (quantityA : ℕ) -- Quantity of type A
  (quantityB : ℕ) -- Quantity of type B

/-- Conditions for air conditioner purchase -/
def satisfiesConditions (ac : AirConditioner) : Prop :=
  ac.costA * 3 + ac.costB * 2 = 39000 ∧
  ac.costA * 4 = ac.costB * 5 + 6000 ∧
  ac.quantityA + ac.quantityB = 30 ∧
  ac.quantityA * 2 ≥ ac.quantityB ∧
  ac.costA * ac.quantityA + ac.costB * ac.quantityB ≤ 217000

/-- Total cost of air conditioners -/
def totalCost (ac : AirConditioner) : ℕ :=
  ac.costA * ac.quantityA + ac.costB * ac.quantityB

/-- Theorem stating the correct costs and minimum total cost -/
theorem air_conditioner_costs_and_minimum_cost :
  ∃ (ac : AirConditioner),
    satisfiesConditions ac ∧
    ac.costA = 9000 ∧
    ac.costB = 6000 ∧
    (∀ (ac' : AirConditioner), satisfiesConditions ac' → totalCost ac ≤ totalCost ac') ∧
    totalCost ac = 210000 :=
  sorry

end NUMINAMATH_CALUDE_air_conditioner_costs_and_minimum_cost_l1553_155341


namespace NUMINAMATH_CALUDE_pet_shop_kittens_l1553_155343

theorem pet_shop_kittens (num_puppies : ℕ) (puppy_cost kitten_cost total_stock : ℚ) : 
  num_puppies = 2 →
  puppy_cost = 20 →
  kitten_cost = 15 →
  total_stock = 100 →
  (total_stock - num_puppies * puppy_cost) / kitten_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_pet_shop_kittens_l1553_155343


namespace NUMINAMATH_CALUDE_profit_increase_condition_l1553_155331

/-- The selling price function -/
def price (t : ℤ) : ℚ := (1/4) * t + 30

/-- The daily sales volume function -/
def sales_volume (t : ℤ) : ℚ := 120 - 2 * t

/-- The daily profit function after donation -/
def profit (t : ℤ) (n : ℚ) : ℚ :=
  (price t - 20 - n) * sales_volume t

/-- The derivative of the profit function with respect to t -/
def profit_derivative (t : ℤ) (n : ℚ) : ℚ :=
  -t + 2*n + 10

theorem profit_increase_condition (n : ℚ) :
  (∀ t : ℤ, 1 ≤ t ∧ t ≤ 28 → profit_derivative t n > 0) ↔
  (8.75 < n ∧ n ≤ 9.25) :=
sorry

end NUMINAMATH_CALUDE_profit_increase_condition_l1553_155331


namespace NUMINAMATH_CALUDE_stating_plant_distribution_theorem_l1553_155359

/-- Represents the number of ways to distribute plants among lamps -/
def plant_distribution_ways : ℕ := 9

/-- The number of cactus plants -/
def num_cactus : ℕ := 3

/-- The number of bamboo plants -/
def num_bamboo : ℕ := 2

/-- The number of blue lamps -/
def num_blue_lamps : ℕ := 3

/-- The number of green lamps -/
def num_green_lamps : ℕ := 2

/-- 
Theorem stating that the number of ways to distribute the plants among the lamps is 9,
given the specified numbers of plants and lamps.
-/
theorem plant_distribution_theorem : 
  plant_distribution_ways = 9 := by sorry

end NUMINAMATH_CALUDE_stating_plant_distribution_theorem_l1553_155359


namespace NUMINAMATH_CALUDE_factorial_divisor_differences_l1553_155305

def divisors (n : ℕ) : List ℕ := sorry

def consecutive_differences (l : List ℕ) : List ℕ := sorry

def is_non_decreasing (l : List ℕ) : Prop := sorry

theorem factorial_divisor_differences (n : ℕ) :
  n ≥ 3 ∧ is_non_decreasing (consecutive_differences (divisors (n.factorial))) ↔ n = 3 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisor_differences_l1553_155305


namespace NUMINAMATH_CALUDE_locus_is_circle_l1553_155323

/-- Given two fixed points A and B in a plane, the locus of points C 
    satisfying $\overrightarrow{AC} \cdot \overrightarrow{BC} = 1$ is a circle. -/
theorem locus_is_circle (A B : ℝ × ℝ) : 
  {C : ℝ × ℝ | (C.1 - A.1, C.2 - A.2) • (C.1 - B.1, C.2 - B.2) = 1} = 
  {C : ℝ × ℝ | ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2} :=
sorry

end NUMINAMATH_CALUDE_locus_is_circle_l1553_155323


namespace NUMINAMATH_CALUDE_line_l_equation_l1553_155369

-- Define the intersection point of the two given lines
def intersection_point : ℝ × ℝ := (2, 1)

-- Define point A
def point_A : ℝ × ℝ := (5, 0)

-- Define the distance from point A to line l
def distance_to_l : ℝ := 3

-- Define the two possible equations for line l
def line_eq1 (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0
def line_eq2 (x : ℝ) : Prop := x = 2

-- Theorem statement
theorem line_l_equation : 
  ∃ (l : ℝ → ℝ → Prop), 
    (∀ x y, l x y ↔ (line_eq1 x y ∨ line_eq2 x)) ∧
    (l (intersection_point.1) (intersection_point.2)) ∧
    (∀ x y, l x y → 
      (|4 * point_A.1 - 3 * point_A.2 - 5| / Real.sqrt (4^2 + 3^2) = distance_to_l ∨
       |point_A.1 - 2| = distance_to_l)) :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_l1553_155369


namespace NUMINAMATH_CALUDE_trout_weight_fishing_scenario_l1553_155376

/-- Calculates the weight of trout caught given the fishing conditions -/
theorem trout_weight (num_campers : ℕ) (fish_per_camper : ℕ) 
                     (num_bass : ℕ) (bass_weight : ℕ) 
                     (num_salmon : ℕ) (salmon_weight : ℕ) : ℕ :=
  let total_fish_needed := num_campers * fish_per_camper
  let total_bass_weight := num_bass * bass_weight
  let total_salmon_weight := num_salmon * salmon_weight
  total_fish_needed - (total_bass_weight + total_salmon_weight)

/-- The specific fishing scenario described in the problem -/
theorem fishing_scenario : trout_weight 22 2 6 2 2 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_trout_weight_fishing_scenario_l1553_155376


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1553_155325

theorem sum_with_radical_conjugate : 
  let x : ℝ := 16 - Real.sqrt 2023
  let y : ℝ := 16 + Real.sqrt 2023
  x + y = 32 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1553_155325


namespace NUMINAMATH_CALUDE_intersecting_circles_sum_l1553_155316

/-- Two circles intersect at points A and B, with their centers on a line -/
structure IntersectingCircles where
  m : ℝ
  n : ℝ
  /-- Point A coordinates -/
  pointA : ℝ × ℝ := (1, 3)
  /-- Point B coordinates -/
  pointB : ℝ × ℝ := (m, n)
  /-- The centers of both circles are on the line x - y - 2 = 0 -/
  centers_on_line : ∀ (x y : ℝ), x - y - 2 = 0

/-- The sum of m and n for the intersecting circles is 4 -/
theorem intersecting_circles_sum (ic : IntersectingCircles) : ic.m + ic.n = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_sum_l1553_155316


namespace NUMINAMATH_CALUDE_power_of_two_equality_l1553_155382

theorem power_of_two_equality (m : ℤ) : 
  2^1999 - 2^1998 - 2^1997 + 2^1996 - 2^1995 = m * 2^1995 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l1553_155382


namespace NUMINAMATH_CALUDE_value_of_M_l1553_155308

theorem value_of_M : 
  let M := (Real.sqrt (Real.sqrt 7 + 3) - Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 - 1) + Real.sqrt (4 - 2 * Real.sqrt 3)
  M = (3 - Real.sqrt 6 + Real.sqrt 42) / 6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l1553_155308


namespace NUMINAMATH_CALUDE_percentage_difference_l1553_155300

theorem percentage_difference (y e w z : ℝ) (P : ℝ) : 
  w = e * (1 - P / 100) →
  e = y * 0.6 →
  z = y * 0.54 →
  z = w * (1 + 0.5000000000000002) →
  P = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l1553_155300


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1553_155313

theorem consecutive_integers_sum (n : ℕ) : 
  (n + 2 = 9) → (n + (n + 1) + (n + 2) = 24) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1553_155313


namespace NUMINAMATH_CALUDE_employee_reduction_percentage_l1553_155348

/-- Theorem: Employee Reduction Percentage

Given:
- The number of employees decreased.
- The average salary increased by 10%.
- The total salary remained constant.

Prove:
The percentage decrease in the number of employees is (1 - 1/1.1) * 100%.
-/
theorem employee_reduction_percentage 
  (E : ℝ) -- Initial number of employees
  (E' : ℝ) -- Number of employees after reduction
  (S : ℝ) -- Initial average salary
  (h1 : E' < E) -- Number of employees decreased
  (h2 : E' * (1.1 * S) = E * S) -- Total salary remained constant
  : (E - E') / E * 100 = (1 - 1 / 1.1) * 100 := by
  sorry

#check employee_reduction_percentage

end NUMINAMATH_CALUDE_employee_reduction_percentage_l1553_155348


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1553_155370

theorem complex_equation_solution (z : ℂ) : z * (2 - Complex.I) = 3 + Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1553_155370


namespace NUMINAMATH_CALUDE_four_meetings_theorem_l1553_155399

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool -- True for clockwise, False for counterclockwise

/-- Calculates the number of meetings between two runners on a circular track -/
def number_of_meetings (runner1 runner2 : Runner) : ℕ :=
  sorry

/-- Theorem stating that two runners with speeds 2 m/s and 3 m/s in opposite directions meet 4 times -/
theorem four_meetings_theorem (track_length : ℝ) (h : track_length > 0) :
  let runner1 : Runner := ⟨2, true⟩
  let runner2 : Runner := ⟨3, false⟩
  number_of_meetings runner1 runner2 = 4 :=
sorry

end NUMINAMATH_CALUDE_four_meetings_theorem_l1553_155399


namespace NUMINAMATH_CALUDE_n_fifth_minus_n_divisible_by_30_l1553_155315

theorem n_fifth_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) := by
  sorry

end NUMINAMATH_CALUDE_n_fifth_minus_n_divisible_by_30_l1553_155315


namespace NUMINAMATH_CALUDE_f_max_is_k_max_b_ac_l1553_155368

/-- The function f(x) = |x-1| - 2|x+1| --/
def f (x : ℝ) : ℝ := |x - 1| - 2 * |x + 1|

/-- The maximum value of f(x) --/
def k : ℝ := 2

/-- Theorem stating that k is the maximum value of f(x) --/
theorem f_max_is_k : ∀ x : ℝ, f x ≤ k :=
sorry

/-- Theorem for the maximum value of b(a+c) given the conditions --/
theorem max_b_ac (a b c : ℝ) (h : (a^2 + c^2) / 2 + b^2 = k) :
  b * (a + c) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_f_max_is_k_max_b_ac_l1553_155368


namespace NUMINAMATH_CALUDE_addition_problems_l1553_155302

theorem addition_problems :
  (189 + (-9) = 180) ∧
  ((-25) + 56 + (-39) = -8) ∧
  (41 + (-22) + (-33) + 19 = 5) ∧
  ((-0.5) + 13/4 + 2.75 + (-11/2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_addition_problems_l1553_155302


namespace NUMINAMATH_CALUDE_complex_square_plus_one_zero_l1553_155390

theorem complex_square_plus_one_zero (x : ℂ) : x^2 + 1 = 0 → x = Complex.I ∨ x = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_plus_one_zero_l1553_155390


namespace NUMINAMATH_CALUDE_problem_statement_l1553_155360

theorem problem_statement : (10 * 7)^3 + (45 * 5)^2 = 393625 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1553_155360


namespace NUMINAMATH_CALUDE_carla_leaf_collection_l1553_155301

/-- Represents the number of items Carla needs to collect each day -/
def daily_items : ℕ := 5

/-- Represents the number of days Carla has to collect items -/
def total_days : ℕ := 10

/-- Represents the number of bugs Carla needs to collect -/
def bugs_to_collect : ℕ := 20

/-- Calculates the total number of items Carla needs to collect -/
def total_items : ℕ := daily_items * total_days

/-- Calculates the number of leaves Carla needs to collect -/
def leaves_to_collect : ℕ := total_items - bugs_to_collect

theorem carla_leaf_collection :
  leaves_to_collect = 30 := by sorry

end NUMINAMATH_CALUDE_carla_leaf_collection_l1553_155301


namespace NUMINAMATH_CALUDE_angle_measure_l1553_155351

theorem angle_measure (x : ℝ) : 
  (90 - x = (180 - x) / 3 + 20) → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l1553_155351


namespace NUMINAMATH_CALUDE_inscriptions_exist_l1553_155380

/-- Represents the maker of a casket -/
inductive Maker
| Bellini
| Cellini

/-- Represents a casket with its inscription -/
structure Casket where
  maker : Maker
  inscription : Prop

/-- The pair of caskets satisfies the given conditions -/
def satisfies_conditions (golden silver : Casket) : Prop :=
  let P := (golden.maker = Maker.Bellini ∧ silver.maker = Maker.Cellini) ∨
           (golden.maker = Maker.Cellini ∧ silver.maker = Maker.Bellini)
  let Q := silver.maker = Maker.Cellini
  
  -- Condition 1: One can conclude that one casket is made by Bellini and the other by Cellini
  (golden.inscription ∧ silver.inscription → P) ∧
  
  -- Condition 1 (continued): But it's impossible to determine which casket is whose work
  (golden.inscription ∧ silver.inscription → ¬(golden.maker = Maker.Bellini ∨ golden.maker = Maker.Cellini)) ∧
  
  -- Condition 2: The inscription on either casket alone doesn't allow concluding about the makers
  (golden.inscription → ¬P) ∧
  (silver.inscription → ¬P)

/-- There exist inscriptions that satisfy the given conditions -/
theorem inscriptions_exist : ∃ (golden silver : Casket), satisfies_conditions golden silver := by
  sorry

end NUMINAMATH_CALUDE_inscriptions_exist_l1553_155380


namespace NUMINAMATH_CALUDE_jane_mean_score_l1553_155342

def jane_scores : List ℝ := [85, 88, 90, 92, 95, 100]

theorem jane_mean_score : 
  (jane_scores.sum / jane_scores.length : ℝ) = 550 / 6 := by
  sorry

end NUMINAMATH_CALUDE_jane_mean_score_l1553_155342

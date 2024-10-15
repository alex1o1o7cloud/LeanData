import Mathlib

namespace NUMINAMATH_CALUDE_power_equation_solutions_l2088_208859

theorem power_equation_solutions (a b : ℕ) (ha : a ≥ 1) (hb : b ≥ 1) :
  a^(b^2) = b^a → (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 16) ∨ (a = 3 ∧ b = 27) := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solutions_l2088_208859


namespace NUMINAMATH_CALUDE_unique_positive_number_l2088_208861

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 * (1 / x) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l2088_208861


namespace NUMINAMATH_CALUDE_michael_has_two_cats_l2088_208821

/-- The number of dogs Michael has -/
def num_dogs : ℕ := 3

/-- The cost per night per animal for pet-sitting -/
def cost_per_animal : ℕ := 13

/-- The total cost for pet-sitting -/
def total_cost : ℕ := 65

/-- The number of cats Michael has -/
def num_cats : ℕ := (total_cost - num_dogs * cost_per_animal) / cost_per_animal

theorem michael_has_two_cats : num_cats = 2 := by
  sorry

end NUMINAMATH_CALUDE_michael_has_two_cats_l2088_208821


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l2088_208842

theorem max_value_trig_expression (a b c : ℝ) :
  (⨆ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c * Real.sin (2 * θ)) = Real.sqrt (a^2 + b^2 + 4 * c^2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l2088_208842


namespace NUMINAMATH_CALUDE_zero_exponent_equals_one_l2088_208885

theorem zero_exponent_equals_one (r : ℚ) (h : r ≠ 0) : r ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_equals_one_l2088_208885


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2088_208896

theorem nested_fraction_equality : 1 + (1 / (1 + (1 / (1 + (1 / 2))))) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2088_208896


namespace NUMINAMATH_CALUDE_tangent_circle_properties_l2088_208813

/-- A circle with center (1, 2) that is tangent to the x-axis -/
def TangentCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 4}

/-- The center of the circle -/
def center : ℝ × ℝ := (1, 2)

/-- The radius of the circle -/
def radius : ℝ := 2

theorem tangent_circle_properties :
  (∀ p ∈ TangentCircle, (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2) ∧
  (∃ p ∈ TangentCircle, p.2 = 0) ∧
  (∀ p ∈ TangentCircle, p.2 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_properties_l2088_208813


namespace NUMINAMATH_CALUDE_keaton_annual_earnings_l2088_208807

/-- Represents Keaton's farm and calculates annual earnings --/
def farm_earnings : ℕ :=
  let months_per_year : ℕ := 12
  let orange_harvest_interval : ℕ := 2
  let apple_harvest_interval : ℕ := 3
  let orange_harvest_value : ℕ := 50
  let apple_harvest_value : ℕ := 30
  let orange_harvests_per_year : ℕ := months_per_year / orange_harvest_interval
  let apple_harvests_per_year : ℕ := months_per_year / apple_harvest_interval
  let orange_earnings : ℕ := orange_harvests_per_year * orange_harvest_value
  let apple_earnings : ℕ := apple_harvests_per_year * apple_harvest_value
  orange_earnings + apple_earnings

/-- Theorem stating that Keaton's annual farm earnings are $420 --/
theorem keaton_annual_earnings : farm_earnings = 420 := by
  sorry

end NUMINAMATH_CALUDE_keaton_annual_earnings_l2088_208807


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l2088_208872

def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x - k

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

def passes_through_quadrant (f : ℝ → ℝ) (quadrant : ℕ) : Prop :=
  match quadrant with
  | 1 => ∃ x > 0, f x > 0
  | 2 => ∃ x < 0, f x > 0
  | 3 => ∃ x < 0, f x < 0
  | 4 => ∃ x > 0, f x < 0
  | _ => False

theorem linear_function_quadrants (k : ℝ) :
  decreasing_function (linear_function k) →
  (passes_through_quadrant (linear_function k) 1 ∧
   passes_through_quadrant (linear_function k) 2 ∧
   passes_through_quadrant (linear_function k) 4) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l2088_208872


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l2088_208810

theorem fourth_root_equation_solution :
  ∃! x : ℝ, (2 - x / 2) ^ (1/4 : ℝ) = 2 ∧ x = -28 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l2088_208810


namespace NUMINAMATH_CALUDE_average_increase_l2088_208895

theorem average_increase (initial_average : ℝ) (fourth_test_score : ℝ) :
  initial_average = 81 ∧ fourth_test_score = 89 →
  (3 * initial_average + fourth_test_score) / 4 = initial_average + 2 := by
  sorry

end NUMINAMATH_CALUDE_average_increase_l2088_208895


namespace NUMINAMATH_CALUDE_cookies_per_batch_l2088_208886

/-- Proves that each batch of cookies must produce 4 dozens to meet the required total --/
theorem cookies_per_batch 
  (classmates : ℕ) 
  (cookies_per_student : ℕ) 
  (total_batches : ℕ) 
  (h1 : classmates = 24) 
  (h2 : cookies_per_student = 10) 
  (h3 : total_batches = 5) : 
  (classmates * cookies_per_student) / (total_batches * 12) = 4 := by
  sorry

#check cookies_per_batch

end NUMINAMATH_CALUDE_cookies_per_batch_l2088_208886


namespace NUMINAMATH_CALUDE_solve_for_a_l2088_208856

def U (a : ℝ) : Set ℝ := {2, 4, 1-a}
def A (a : ℝ) : Set ℝ := {2, a^2-a+2}

theorem solve_for_a (a : ℝ) : 
  (U a \ A a = {-1}) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2088_208856


namespace NUMINAMATH_CALUDE_solution_set_f_leq_4_range_of_m_f_gt_m_squared_plus_m_l2088_208816

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Theorem for the solution set of f(x) ≤ 4
theorem solution_set_f_leq_4 :
  {x : ℝ | f x ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for the range of m where f(x) > m^2 + m always holds
theorem range_of_m_f_gt_m_squared_plus_m :
  {m : ℝ | ∀ x, f x > m^2 + m} = {m : ℝ | -2 < m ∧ m < 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_4_range_of_m_f_gt_m_squared_plus_m_l2088_208816


namespace NUMINAMATH_CALUDE_unique_solution_for_n_l2088_208811

theorem unique_solution_for_n : ∃! n : ℚ, (1 / (n + 2)) + (2 / (n + 2)) + ((n + 1) / (n + 2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_n_l2088_208811


namespace NUMINAMATH_CALUDE_root_implies_h_value_l2088_208809

theorem root_implies_h_value (h : ℝ) :
  (3 : ℝ)^3 - 2*h*3 + 15 = 0 → h = 7 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_h_value_l2088_208809


namespace NUMINAMATH_CALUDE_parabola_translation_l2088_208814

/-- The initial parabola function -/
def initial_parabola (x : ℝ) : ℝ := -3 * (x + 1)^2 - 2

/-- The final parabola function -/
def final_parabola (x : ℝ) : ℝ := -3 * x^2

/-- Translation function that moves a point 1 unit right and 2 units up -/
def translate (x y : ℝ) : ℝ × ℝ := (x - 1, y + 2)

theorem parabola_translation :
  ∀ x : ℝ, final_parabola x = (initial_parabola (x - 1) + 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2088_208814


namespace NUMINAMATH_CALUDE_minimum_value_implies_m_equals_one_l2088_208839

-- Define the domain D
def D : Set ℝ := Set.Icc 1 2

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x - m^2

-- Theorem statement
theorem minimum_value_implies_m_equals_one :
  ∀ m : ℝ, (∀ x ∈ D, g m x ≥ 2) ∧ (∃ x ∈ D, g m x = 2) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_implies_m_equals_one_l2088_208839


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angle_relation_l2088_208838

theorem polygon_interior_exterior_angle_relation :
  ∀ n : ℕ, 
  n > 2 →
  (n - 2) * 180 = 2 * 360 →
  n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angle_relation_l2088_208838


namespace NUMINAMATH_CALUDE_base_n_multiple_of_five_l2088_208828

theorem base_n_multiple_of_five (n : ℕ) : 
  let count := Finset.filter (fun n => (2*n^5 + 3*n^4 + 5*n^3 + 2*n^2 + 3*n + 6) % 5 = 0) 
    (Finset.range 99 ∪ {100})
  (2 ≤ n) → (n ≤ 100) → Finset.card count = 40 := by
  sorry

end NUMINAMATH_CALUDE_base_n_multiple_of_five_l2088_208828


namespace NUMINAMATH_CALUDE_prob_consecutive_prob_sum_divisible_by_3_l2088_208858

-- Define the type for ball labels
inductive BallLabel : Type
  | one : BallLabel
  | two : BallLabel
  | three : BallLabel
  | four : BallLabel

-- Define a function to convert BallLabel to natural number
def ballLabelToNat (b : BallLabel) : ℕ :=
  match b with
  | BallLabel.one => 1
  | BallLabel.two => 2
  | BallLabel.three => 3
  | BallLabel.four => 4

-- Define the type for a pair of drawn balls
def DrawnPair := BallLabel × BallLabel

-- Define the sample space
def sampleSpace : Finset DrawnPair := sorry

-- Define the event of drawing consecutive numbers
def consecutiveEvent : Finset DrawnPair := sorry

-- Define the event of drawing numbers with sum divisible by 3
def sumDivisibleBy3Event : Finset DrawnPair := sorry

-- Theorem for the probability of drawing consecutive numbers
theorem prob_consecutive : 
  (consecutiveEvent.card : ℚ) / sampleSpace.card = 3 / 8 := sorry

-- Theorem for the probability of drawing numbers with sum divisible by 3
theorem prob_sum_divisible_by_3 : 
  (sumDivisibleBy3Event.card : ℚ) / sampleSpace.card = 5 / 16 := sorry

end NUMINAMATH_CALUDE_prob_consecutive_prob_sum_divisible_by_3_l2088_208858


namespace NUMINAMATH_CALUDE_track_width_l2088_208822

theorem track_width (r₁ r₂ : ℝ) (h₁ : r₁ > r₂) 
  (h₂ : 2 * π * r₁ - 2 * π * r₂ = 20 * π) 
  (h₃ : r₁ - r₂ = 2 * (r₁ - r₂) / 2) : 
  r₁ - r₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_track_width_l2088_208822


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2088_208804

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 1}

-- Theorem statement
theorem complement_of_A_in_U : Set.compl A = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2088_208804


namespace NUMINAMATH_CALUDE_kyungsoo_string_shorter_l2088_208836

/-- Conversion factor from centimeters to millimeters -/
def cm_to_mm : ℚ := 10

/-- Length of Inhyuk's string in centimeters -/
def inhyuk_length_cm : ℚ := 97.5

/-- Base length of Kyungsoo's string in centimeters -/
def kyungsoo_base_length_cm : ℚ := 97

/-- Additional length of Kyungsoo's string in millimeters -/
def kyungsoo_additional_length_mm : ℚ := 3

/-- Theorem stating that Kyungsoo's string is shorter than Inhyuk's -/
theorem kyungsoo_string_shorter :
  kyungsoo_base_length_cm * cm_to_mm + kyungsoo_additional_length_mm <
  inhyuk_length_cm * cm_to_mm := by
  sorry

end NUMINAMATH_CALUDE_kyungsoo_string_shorter_l2088_208836


namespace NUMINAMATH_CALUDE_work_increase_percentage_l2088_208870

theorem work_increase_percentage (p : ℕ) (W : ℝ) (h : p > 0) : 
  let absent_ratio : ℝ := 1 / 6
  let present_ratio : ℝ := 1 - absent_ratio
  let original_work_per_person : ℝ := W / p
  let new_work_per_person : ℝ := W / (p * present_ratio)
  let work_increase : ℝ := new_work_per_person - original_work_per_person
  let percentage_increase : ℝ := (work_increase / original_work_per_person) * 100
  percentage_increase = 20 := by sorry

end NUMINAMATH_CALUDE_work_increase_percentage_l2088_208870


namespace NUMINAMATH_CALUDE_unique_solution_exists_l2088_208826

/-- Represents a digit from 0 to 7 -/
def Digit := Fin 8

/-- Converts a three-digit number to its integer representation -/
def toInt (a b c : Digit) : Nat := a.val * 100 + b.val * 10 + c.val

/-- Converts a two-digit number to its integer representation -/
def toInt2 (d e : Digit) : Nat := d.val * 10 + e.val

theorem unique_solution_exists (a b c d e f g h : Digit) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
               f ≠ g ∧ f ≠ h ∧
               g ≠ h)
  (abc_eq : toInt a b c = 146)
  (equation : toInt a b c + toInt2 d e = toInt f g h) :
  toInt2 d e = 57 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l2088_208826


namespace NUMINAMATH_CALUDE_tomorrow_is_saturday_l2088_208805

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem tomorrow_is_saturday 
  (h : advanceDay (advanceDay DayOfWeek.Wednesday 2) 5 = DayOfWeek.Monday) :
  nextDay DayOfWeek.Friday = DayOfWeek.Saturday :=
by
  sorry


end NUMINAMATH_CALUDE_tomorrow_is_saturday_l2088_208805


namespace NUMINAMATH_CALUDE_parabola_properties_l2088_208863

-- Define the parabola equation
def parabola_equation (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 29

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (3, -2)

-- Define the point that the parabola passes through
def point_on_parabola : ℝ × ℝ := (4, -5)

-- Theorem statement
theorem parabola_properties :
  -- The parabola passes through the given point
  parabola_equation point_on_parabola.1 = point_on_parabola.2 ∧
  -- The vertex of the parabola is at the given point
  (∀ x : ℝ, parabola_equation x ≥ parabola_equation vertex.1) ∧
  -- The axis of symmetry is vertical (x = vertex.1)
  (∀ x : ℝ, parabola_equation (2 * vertex.1 - x) = parabola_equation x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2088_208863


namespace NUMINAMATH_CALUDE_ap_terms_count_l2088_208880

theorem ap_terms_count (n : ℕ) (a d : ℚ) : 
  Even n →
  n / 2 * (2 * a + (n - 2) * d) = 18 →
  n / 2 * (2 * a + 2 * d + (n - 2) * d) = 36 →
  a + (n - 1) * d - a = 7 →
  n = 12 :=
by sorry

end NUMINAMATH_CALUDE_ap_terms_count_l2088_208880


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2088_208817

theorem floor_equation_solution (A B : ℝ) (hA : A ≥ 0) (hB : B ≥ 0) :
  (∀ x : ℝ, x > 1 → ⌊1 / (A * x + B / x)⌋ = 1 / (A * ⌊x⌋ + B / ⌊x⌋)) →
  A = 0 ∧ B = 1 := by
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2088_208817


namespace NUMINAMATH_CALUDE_new_xanadu_license_plates_l2088_208894

/-- Represents the number of possible letters in the alphabet -/
def num_letters : ℕ := 26

/-- Represents the number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- Represents the number of possible first digits (1-9) -/
def num_first_digits : ℕ := 9

/-- Calculates the total number of valid license plates in New Xanadu -/
def num_valid_plates : ℕ :=
  num_letters ^ 3 * num_first_digits * num_digits ^ 2

/-- Theorem stating the total number of valid license plates in New Xanadu -/
theorem new_xanadu_license_plates :
  num_valid_plates = 15818400 := by
  sorry

end NUMINAMATH_CALUDE_new_xanadu_license_plates_l2088_208894


namespace NUMINAMATH_CALUDE_stratified_sampling_l2088_208837

/-- Represents the total number of teachers -/
def total_teachers : ℕ := 300

/-- Represents the number of senior teachers -/
def senior_teachers : ℕ := 90

/-- Represents the number of intermediate teachers -/
def intermediate_teachers : ℕ := 150

/-- Represents the number of junior teachers -/
def junior_teachers : ℕ := 60

/-- Represents the sample size -/
def sample_size : ℕ := 60

/-- Theorem stating the correct stratified sampling for each teacher category -/
theorem stratified_sampling :
  (senior_teachers * sample_size) / total_teachers = 18 ∧
  (intermediate_teachers * sample_size) / total_teachers = 30 ∧
  (junior_teachers * sample_size) / total_teachers = 12 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2088_208837


namespace NUMINAMATH_CALUDE_ratio_s4_s5_l2088_208871

/-- An arithmetic sequence with a given ratio of second to third term -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  ratio_condition : a 2 / a 3 = 1 / 3

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * n / 2

/-- The main theorem: ratio of S_4 to S_5 is 8/15 -/
theorem ratio_s4_s5 (seq : ArithmeticSequence) :
  sum_n seq 4 / sum_n seq 5 = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_s4_s5_l2088_208871


namespace NUMINAMATH_CALUDE_smallest_a1_l2088_208844

/-- Given a sequence of positive real numbers {aₙ} where aₙ = 15aₙ₋₁ - 2n for all n > 1,
    the smallest possible value of a₁ is 29/98. -/
theorem smallest_a1 (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n > 1, a n = 15 * a (n - 1) - 2 * n) →
  ∀ x, (∀ n, a n > 0) → (∀ n > 1, a n = 15 * a (n - 1) - 2 * n) → a 1 ≥ x →
  x ≤ 29 / 98 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a1_l2088_208844


namespace NUMINAMATH_CALUDE_integer_solutions_of_polynomial_l2088_208854

theorem integer_solutions_of_polynomial (n : ℤ) : 
  n^5 - 2*n^4 - 7*n^2 - 7*n + 3 = 0 ↔ n = -1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_polynomial_l2088_208854


namespace NUMINAMATH_CALUDE_goldfish_percentage_l2088_208879

theorem goldfish_percentage (surface : ℕ) (below : ℕ) : 
  surface = 15 → below = 45 → (surface : ℚ) / (surface + below : ℚ) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_percentage_l2088_208879


namespace NUMINAMATH_CALUDE_arrangement_theorem_l2088_208831

/-- The number of ways to arrange 3 people on 6 chairs in a row, 
    such that no two people sit next to each other -/
def arrangement_count : ℕ := 24

/-- The number of chairs in the row -/
def total_chairs : ℕ := 6

/-- The number of people to be seated -/
def people_count : ℕ := 3

/-- Theorem stating that the number of arrangements 
    satisfying the given conditions is 24 -/
theorem arrangement_theorem : 
  arrangement_count = 
    (Nat.factorial people_count) * (total_chairs - people_count - (people_count - 1)) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l2088_208831


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_162_l2088_208802

theorem percentage_of_360_equals_162 : 
  (162 / 360) * 100 = 45 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_162_l2088_208802


namespace NUMINAMATH_CALUDE_tan_45_degrees_l2088_208864

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l2088_208864


namespace NUMINAMATH_CALUDE_game_probability_l2088_208862

theorem game_probability (p_lose p_tie : ℚ) 
  (h_lose : p_lose = 3/7)
  (h_tie : p_tie = 2/21)
  (h_outcomes : ∃ (p_win : ℚ), p_win + p_lose + p_tie = 1) :
  ∃ (p_win : ℚ), p_win = 10/21 := by
sorry

end NUMINAMATH_CALUDE_game_probability_l2088_208862


namespace NUMINAMATH_CALUDE_red_beans_proposition_l2088_208823

-- Define a type for lines in the poem
inductive PoemLine
| A : PoemLine
| B : PoemLine
| C : PoemLine
| D : PoemLine

-- Define what a proposition is
def isProposition (line : PoemLine) : Prop :=
  match line with
  | PoemLine.A => true  -- "Red beans grow in the southern country" is a proposition
  | _ => false          -- Other lines are not propositions for this problem

-- Theorem statement
theorem red_beans_proposition :
  isProposition PoemLine.A :=
by sorry

end NUMINAMATH_CALUDE_red_beans_proposition_l2088_208823


namespace NUMINAMATH_CALUDE_probability_inconsistency_l2088_208869

-- Define the probability measure
variable (p : Set ℝ → ℝ)

-- Define events a and b
variable (a b : Set ℝ)

-- State the given probabilities
axiom pa : p a = 0.18
axiom pb : p b = 0.5
axiom pab : p (a ∩ b) = 0.36

-- Theorem to prove the inconsistency
theorem probability_inconsistency :
  ¬(0 ≤ p a ∧ p a ≤ 1 ∧
    0 ≤ p b ∧ p b ≤ 1 ∧
    0 ≤ p (a ∩ b) ∧ p (a ∩ b) ≤ 1 ∧
    p (a ∩ b) ≤ p a ∧ p (a ∩ b) ≤ p b) :=
by sorry

end NUMINAMATH_CALUDE_probability_inconsistency_l2088_208869


namespace NUMINAMATH_CALUDE_cheryl_mms_after_dinner_l2088_208800

/-- The number of m&m's Cheryl had at the beginning -/
def initial_mms : ℕ := 25

/-- The number of m&m's Cheryl ate after lunch -/
def after_lunch : ℕ := 7

/-- The number of m&m's Cheryl gave to her sister -/
def given_to_sister : ℕ := 13

/-- The number of m&m's Cheryl ate after dinner -/
def after_dinner : ℕ := 5

theorem cheryl_mms_after_dinner : 
  initial_mms - after_lunch - after_dinner - given_to_sister = 0 :=
by sorry

end NUMINAMATH_CALUDE_cheryl_mms_after_dinner_l2088_208800


namespace NUMINAMATH_CALUDE_jerry_grocery_shopping_l2088_208877

/-- The amount of money Jerry has left after grocery shopping -/
def money_left (mustard_oil_price mustard_oil_quantity pasta_price pasta_quantity sauce_price sauce_quantity total_money : ℕ) : ℕ :=
  total_money - (mustard_oil_price * mustard_oil_quantity + pasta_price * pasta_quantity + sauce_price * sauce_quantity)

/-- Theorem stating that Jerry will have $7 left after grocery shopping -/
theorem jerry_grocery_shopping :
  money_left 13 2 4 3 5 1 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerry_grocery_shopping_l2088_208877


namespace NUMINAMATH_CALUDE_matrix_determinant_l2088_208850

theorem matrix_determinant : 
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 4, -2; 0, 3, -1; 5, -1, 2]
  Matrix.det A = 20 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_l2088_208850


namespace NUMINAMATH_CALUDE_square_difference_equality_l2088_208881

theorem square_difference_equality : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2088_208881


namespace NUMINAMATH_CALUDE_candy_purchase_sum_l2088_208890

/-- A sequence of daily candy purchases where each day's purchase is one more than the previous day -/
def candy_sequence (first_day : ℕ) : ℕ → ℕ :=
  fun n => first_day + n - 1

theorem candy_purchase_sum (first_day : ℕ) 
  (h : candy_sequence first_day 0 + candy_sequence first_day 1 + candy_sequence first_day 2 = 504) :
  candy_sequence first_day 3 + candy_sequence first_day 4 + candy_sequence first_day 5 = 513 := by
  sorry

end NUMINAMATH_CALUDE_candy_purchase_sum_l2088_208890


namespace NUMINAMATH_CALUDE_horseshoe_profit_800_sets_l2088_208851

/-- Calculates the profit for horseshoe manufacturing given the specified conditions --/
def horseshoe_profit (
  initial_outlay : ℕ)
  (cost_first_300 : ℕ)
  (cost_beyond_300 : ℕ)
  (price_first_400 : ℕ)
  (price_beyond_400 : ℕ)
  (total_sets : ℕ) : ℕ :=
  let manufacturing_cost := initial_outlay +
    (min total_sets 300) * cost_first_300 +
    (max (total_sets - 300) 0) * cost_beyond_300
  let revenue := (min total_sets 400) * price_first_400 +
    (max (total_sets - 400) 0) * price_beyond_400
  revenue - manufacturing_cost

theorem horseshoe_profit_800_sets :
  horseshoe_profit 10000 20 15 50 45 800 = 14500 := by
  sorry

end NUMINAMATH_CALUDE_horseshoe_profit_800_sets_l2088_208851


namespace NUMINAMATH_CALUDE_cheryl_material_calculation_l2088_208824

theorem cheryl_material_calculation (material_used total_bought second_type leftover : ℝ) :
  material_used = 0.21052631578947367 →
  second_type = 2 / 13 →
  leftover = 4 / 26 →
  total_bought = material_used + leftover →
  total_bought = second_type + (0.21052631578947367 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_cheryl_material_calculation_l2088_208824


namespace NUMINAMATH_CALUDE_arithmetic_sequence_298_l2088_208833

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + d * (n - 1)

theorem arithmetic_sequence_298 :
  ∃ n : ℕ, arithmetic_sequence 1 3 n = 298 ∧ n = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_298_l2088_208833


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2088_208898

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x : ℝ, x ≥ 0 → x^2 - x ≥ 0)) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 - x < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2088_208898


namespace NUMINAMATH_CALUDE_expression_value_l2088_208865

theorem expression_value : 
  let x : ℝ := 5
  (x^2 - 3*x - 4) / (x - 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2088_208865


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l2088_208866

/-- Proves that the actual distance traveled is 50 km given the conditions of the problem -/
theorem actual_distance_traveled (slow_speed fast_speed additional_distance : ℝ) 
  (h1 : slow_speed = 10)
  (h2 : fast_speed = 14)
  (h3 : additional_distance = 20)
  (h4 : ∀ d : ℝ, d / slow_speed = (d + additional_distance) / fast_speed) :
  ∃ d : ℝ, d = 50 ∧ d / slow_speed = (d + additional_distance) / fast_speed :=
by sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l2088_208866


namespace NUMINAMATH_CALUDE_product_of_fractions_l2088_208883

theorem product_of_fractions : (1 : ℚ) / 3 * 2 / 5 * 3 / 7 * 4 / 8 = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2088_208883


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_l2088_208897

/-- The function f(x) = 3x^2 - 18x + 2023 has a minimum value of 1996. -/
theorem min_value_of_quadratic :
  ∃ (m : ℝ), m = 1996 ∧ ∀ x : ℝ, 3 * x^2 - 18 * x + 2023 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_l2088_208897


namespace NUMINAMATH_CALUDE_students_in_class_l2088_208834

theorem students_in_class (b : ℕ) : 
  100 < b ∧ b < 200 ∧ 
  b % 3 = 1 ∧ 
  b % 4 = 1 ∧ 
  b % 5 = 1 → 
  b = 101 ∨ b = 161 := by sorry

end NUMINAMATH_CALUDE_students_in_class_l2088_208834


namespace NUMINAMATH_CALUDE_towel_rate_proof_l2088_208873

/-- Proves that given the specified towel purchases and average price, the unknown rate must be 250. -/
theorem towel_rate_proof (num_towels_1 num_towels_2 num_towels_unknown : ℕ)
  (price_1 price_2 avg_price : ℚ) :
  num_towels_1 = 3 →
  num_towels_2 = 5 →
  num_towels_unknown = 2 →
  price_1 = 100 →
  price_2 = 150 →
  avg_price = 155 →
  let total_towels := num_towels_1 + num_towels_2 + num_towels_unknown
  let total_cost := num_towels_1 * price_1 + num_towels_2 * price_2 + num_towels_unknown * avg_price
  (total_cost / total_towels : ℚ) = avg_price →
  (((total_cost - (num_towels_1 * price_1 + num_towels_2 * price_2)) / num_towels_unknown) : ℚ) = 250 :=
by sorry

end NUMINAMATH_CALUDE_towel_rate_proof_l2088_208873


namespace NUMINAMATH_CALUDE_circplus_not_commutative_l2088_208848

/-- Definition of the ⊕ operation -/
def circplus (a b : ℚ) : ℚ := a * b + 2 * a

/-- Theorem stating that ⊕ is not commutative -/
theorem circplus_not_commutative : ¬ (∀ a b : ℚ, circplus a b = circplus b a) := by
  sorry

end NUMINAMATH_CALUDE_circplus_not_commutative_l2088_208848


namespace NUMINAMATH_CALUDE_odd_function_value_l2088_208857

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value (f : ℝ → ℝ) (b : ℝ) 
    (h_odd : IsOdd f)
    (h_def : ∀ x ≥ 0, f x = x^2 - 3*x + b) :
  f (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l2088_208857


namespace NUMINAMATH_CALUDE_ring_arrangements_count_l2088_208892

/-- The number of ways to arrange 6 out of 10 distinguishable rings on 4 fingers -/
def ring_arrangements : ℕ :=
  (Nat.choose 10 6) * (Nat.factorial 6) * (Nat.choose 9 3)

/-- Theorem stating the number of ring arrangements -/
theorem ring_arrangements_count : ring_arrangements = 12672000 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangements_count_l2088_208892


namespace NUMINAMATH_CALUDE_large_rectangle_area_l2088_208875

def small_rectangle_perimeter : ℕ := 20

def large_rectangle_side_difference : ℕ := 2

def valid_areas : Set ℕ := {3300, 4000, 4500, 4800, 4900}

theorem large_rectangle_area (l w : ℕ) :
  (l + w = small_rectangle_perimeter / 2) →
  (l > 0 ∧ w > 0) →
  ((l + large_rectangle_side_difference) * (w + large_rectangle_side_difference) * 100) ∈ valid_areas :=
by sorry

end NUMINAMATH_CALUDE_large_rectangle_area_l2088_208875


namespace NUMINAMATH_CALUDE_diamond_calculation_l2088_208887

/-- The diamond operation for real numbers -/
def diamond (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- Theorem stating that 2 ◇ (3 ◇ (1 ◇ 4)) = -46652 -/
theorem diamond_calculation : diamond 2 (diamond 3 (diamond 1 4)) = -46652 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l2088_208887


namespace NUMINAMATH_CALUDE_fraction_reduction_l2088_208840

theorem fraction_reduction (a b c : ℝ) 
  (h : (3*a^2 + 6*a*c - 3*c^2 - 6*a*b) ≠ 0) : 
  (4*a^2 + 2*c^2 - 4*b^2 - 8*b*c) / (3*a^2 + 6*a*c - 3*c^2 - 6*a*b) = 
  (4/3) * ((a-2*b+c)*(a-c)) / ((a-b+c)*(a-b-c)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_reduction_l2088_208840


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2088_208868

theorem inequality_equivalence :
  {x : ℝ | |(6 - 2*x + 5) / 4| < 3} = {x : ℝ | -1/2 < x ∧ x < 23/2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2088_208868


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2088_208867

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2088_208867


namespace NUMINAMATH_CALUDE_second_class_size_l2088_208899

theorem second_class_size (students1 : ℕ) (avg1 : ℝ) (avg2 : ℝ) (avg_total : ℝ) :
  students1 = 30 →
  avg1 = 40 →
  avg2 = 90 →
  avg_total = 71.25 →
  ∃ students2 : ℕ, 
    (students1 * avg1 + students2 * avg2) / (students1 + students2 : ℝ) = avg_total ∧
    students2 = 50 :=
by sorry

end NUMINAMATH_CALUDE_second_class_size_l2088_208899


namespace NUMINAMATH_CALUDE_intersection_triangle_is_right_angle_l2088_208830

/-- An ellipse with semi-major axis √m and semi-minor axis 1 -/
structure Ellipse (m : ℝ) :=
  (x y : ℝ)
  (eq : x^2 / m + y^2 = 1)
  (m_gt_one : m > 1)

/-- A hyperbola with semi-major axis √n and semi-minor axis 1 -/
structure Hyperbola (n : ℝ) :=
  (x y : ℝ)
  (eq : x^2 / n - y^2 = 1)
  (n_pos : n > 0)

/-- The foci of a conic section -/
structure Foci :=
  (F₁ F₂ : ℝ × ℝ)

/-- A point in the plane -/
def Point := ℝ × ℝ

/-- Theorem: The triangle formed by the foci and an intersection point of an ellipse and hyperbola with the same foci is a right triangle -/
theorem intersection_triangle_is_right_angle
  (m n : ℝ)
  (E : Ellipse m)
  (H : Hyperbola n)
  (F : Foci)
  (P : Point)
  (h₁ : E.x = P.1 ∧ E.y = P.2)  -- P is on the ellipse
  (h₂ : H.x = P.1 ∧ H.y = P.2)  -- P is on the hyperbola
  (h₃ : F.F₁ ≠ F.F₂)  -- The foci are distinct
  : ∃ (A B C : ℝ),
    (P.1 - F.F₁.1)^2 + (P.2 - F.F₁.2)^2 = A^2 ∧
    (P.1 - F.F₂.1)^2 + (P.2 - F.F₂.2)^2 = B^2 ∧
    (F.F₁.1 - F.F₂.1)^2 + (F.F₁.2 - F.F₂.2)^2 = C^2 ∧
    A^2 + B^2 = C^2 :=
  sorry

end NUMINAMATH_CALUDE_intersection_triangle_is_right_angle_l2088_208830


namespace NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_l2088_208835

/-- Given an arithmetic sequence {a_n} with a_1 = 1 and common difference d = 3,
    prove that the 100th term is equal to 298. -/
theorem arithmetic_sequence_100th_term : 
  ∀ (a : ℕ → ℤ), 
    (a 1 = 1) → 
    (∀ n : ℕ, a (n + 1) - a n = 3) → 
    (a 100 = 298) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_l2088_208835


namespace NUMINAMATH_CALUDE_cylinder_volume_l2088_208876

/-- The volume of a cylinder with base radius 1 cm and generatrix length 2 cm is 2π cm³ -/
theorem cylinder_volume (π : ℝ) : ℝ := by
  sorry

#check cylinder_volume

end NUMINAMATH_CALUDE_cylinder_volume_l2088_208876


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l2088_208846

theorem inserted_numbers_sum (a b c : ℝ) : 
  (∃ r : ℝ, a = 3 * r ∧ b = 3 * r^2) →  -- Geometric progression condition
  (∃ d : ℝ, b = a + d ∧ c = b + d ∧ 27 = c + d) →  -- Arithmetic progression condition
  a + b + c = 161 / 3 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l2088_208846


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2088_208852

theorem trigonometric_identities :
  (∀ x : Real, (1 + Real.tan (1 * π / 180)) * (1 + Real.tan (44 * π / 180)) = 2) ∧
  (∀ x : Real, (3 - Real.sin (70 * π / 180)) / (2 - Real.cos (10 * π / 180)^2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2088_208852


namespace NUMINAMATH_CALUDE_value_of_x_l2088_208803

theorem value_of_x : ∀ (x y z w v : ℕ),
  x = y + 7 →
  y = z + 12 →
  z = w + 25 →
  w = v + 5 →
  v = 90 →
  x = 139 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2088_208803


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l2088_208845

/-- The size of the multiplication table (16 x 16) -/
def tableSize : ℕ := 16

/-- The number of odd numbers from 0 to 15 -/
def oddCount : ℕ := 8

/-- The total number of entries in the multiplication table -/
def totalEntries : ℕ := tableSize * tableSize

/-- The number of odd entries in the multiplication table -/
def oddEntries : ℕ := oddCount * oddCount

theorem multiplication_table_odd_fraction :
  (oddEntries : ℚ) / totalEntries = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l2088_208845


namespace NUMINAMATH_CALUDE_square_triangle_perimeter_l2088_208853

theorem square_triangle_perimeter (square_perimeter : ℝ) :
  square_perimeter = 160 →
  let side_length := square_perimeter / 4
  let diagonal_length := side_length * Real.sqrt 2
  let triangle_perimeter := 2 * side_length + diagonal_length
  triangle_perimeter = 80 + 40 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_perimeter_l2088_208853


namespace NUMINAMATH_CALUDE_min_value_m_l2088_208832

theorem min_value_m (m : ℝ) (h1 : m > 0)
  (h2 : ∀ x : ℝ, x > 1 → 2 * Real.exp (2 * m * x) - Real.log x / m ≥ 0) :
  m ≥ 1 / (2 * Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_m_l2088_208832


namespace NUMINAMATH_CALUDE_optimal_output_l2088_208847

noncomputable section

/-- The defective rate as a function of daily output -/
def defective_rate (c : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ c then 1 / (6 - x) else 2 / 3

/-- The daily profit as a function of daily output -/
def daily_profit (c : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ c
  then (3 * (9 * x - 2 * x^2)) / (2 * (6 - x))
  else 0

/-- The theorem stating the optimal daily output for maximum profit -/
theorem optimal_output (c : ℝ) (h : 0 < c ∧ c < 6) :
  (∃ (x : ℝ), ∀ (y : ℝ), daily_profit c y ≤ daily_profit c x) →
  ((0 < c ∧ c < 3 → ∃ (x : ℝ), x = c ∧ ∀ (y : ℝ), daily_profit c y ≤ daily_profit c x) ∧
   (3 ≤ c ∧ c < 6 → ∃ (x : ℝ), x = 3 ∧ ∀ (y : ℝ), daily_profit c y ≤ daily_profit c x)) :=
by sorry

end

end NUMINAMATH_CALUDE_optimal_output_l2088_208847


namespace NUMINAMATH_CALUDE_inequality_proof_l2088_208801

theorem inequality_proof (x : ℝ) : 2/3 < x ∧ x < 5/4 → (4*x - 5)^2 + (3*x - 2)^2 < (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2088_208801


namespace NUMINAMATH_CALUDE_reflected_light_ray_equation_l2088_208812

/-- Given a point P and its reflection P' across the x-axis, and another point Q,
    this function returns true if the given equation represents the line through P' and Q -/
def is_reflected_line_equation (P Q : ℝ × ℝ) (equation : ℝ → ℝ → ℝ) : Prop :=
  let P' := (P.1, -P.2)  -- Reflection of P across x-axis
  (equation P'.1 P'.2 = 0) ∧ (equation Q.1 Q.2 = 0)

/-- The main theorem stating that 4x + y - 5 = 0 is the equation of the 
    reflected light ray for the given points -/
theorem reflected_light_ray_equation :
  is_reflected_line_equation (2, 3) (1, 1) (fun x y => 4*x + y - 5) := by
  sorry

#check reflected_light_ray_equation

end NUMINAMATH_CALUDE_reflected_light_ray_equation_l2088_208812


namespace NUMINAMATH_CALUDE_neon_signs_blink_together_l2088_208820

theorem neon_signs_blink_together (a b : ℕ) (ha : a = 9) (hb : b = 15) :
  Nat.lcm a b = 45 := by
  sorry

end NUMINAMATH_CALUDE_neon_signs_blink_together_l2088_208820


namespace NUMINAMATH_CALUDE_curve_is_semicircle_l2088_208893

-- Define the curve
def curve (x y : ℝ) : Prop := y - 1 = Real.sqrt (1 - x^2)

-- Define a semicircle
def is_semicircle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (c : ℝ × ℝ) (r : ℝ),
    r > 0 ∧
    S = {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 ∧ p.2 ≥ c.2}

-- Theorem statement
theorem curve_is_semicircle :
  is_semicircle {p : ℝ × ℝ | curve p.1 p.2} :=
sorry

end NUMINAMATH_CALUDE_curve_is_semicircle_l2088_208893


namespace NUMINAMATH_CALUDE_range_of_s_l2088_208819

/-- Definition of the function s for composite positive integers -/
def s (n : ℕ) : ℕ :=
  if n.Prime then 0
  else (n.factors.map (λ p => p ^ 2)).sum

/-- The range of s is the set of integers greater than 11 -/
theorem range_of_s :
  ∀ m : ℕ, m > 11 → ∃ n : ℕ, ¬n.Prime ∧ s n = m ∧
  ∀ k : ℕ, ¬k.Prime → s k > 11 :=
sorry

end NUMINAMATH_CALUDE_range_of_s_l2088_208819


namespace NUMINAMATH_CALUDE_divisible_by_2power10000_within_day_l2088_208808

/-- Represents a card with a natural number -/
structure Card where
  value : ℕ

/-- Represents the state of the table at any given time -/
structure TableState where
  cards : List Card
  time : ℕ

/-- Checks if a number is divisible by 2^10000 -/
def isDivisibleBy2Power10000 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (2^10000)

/-- The process of adding a new card every minute -/
def addNewCard (state : TableState) : TableState :=
  sorry

/-- The main theorem to be proved -/
theorem divisible_by_2power10000_within_day
  (initial_cards : List Card)
  (h1 : initial_cards.length = 100)
  (h2 : (initial_cards.filter (fun c => c.value % 2 = 1)).length = 43) :
  ∃ (final_state : TableState),
    final_state.time ≤ 1440 ∧
    ∃ (c : Card), c ∈ final_state.cards ∧ isDivisibleBy2Power10000 c.value :=
  sorry

end NUMINAMATH_CALUDE_divisible_by_2power10000_within_day_l2088_208808


namespace NUMINAMATH_CALUDE_intersection_point_on_line_and_x_axis_l2088_208827

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 5 * y - 7 * x = 35

/-- The point of intersection -/
def intersection_point : ℝ × ℝ := (-5, 0)

/-- Theorem: The intersection point satisfies the line equation and lies on the x-axis -/
theorem intersection_point_on_line_and_x_axis :
  line_equation intersection_point.1 intersection_point.2 ∧ intersection_point.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_on_line_and_x_axis_l2088_208827


namespace NUMINAMATH_CALUDE_positive_integer_value_l2088_208806

def first_seven_multiples_of_four : List Nat := [4, 8, 12, 16, 20, 24, 28]

def a : ℚ := (first_seven_multiples_of_four.sum : ℚ) / 7

def b (n : ℕ) : ℚ := 2 * n

theorem positive_integer_value (n : ℕ) (h : n > 0) :
  a^2 - (b n)^2 = 0 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_value_l2088_208806


namespace NUMINAMATH_CALUDE_integral_sin_cubed_over_cos_fifth_l2088_208841

theorem integral_sin_cubed_over_cos_fifth (x : Real) :
  let f := fun (x : Real) => (1 / (4 * (Real.cos x)^4)) - (1 / (2 * (Real.cos x)^2))
  deriv f x = (Real.sin x)^3 / (Real.cos x)^5 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin_cubed_over_cos_fifth_l2088_208841


namespace NUMINAMATH_CALUDE_initially_calculated_average_is_175_l2088_208818

/-- The initially calculated average height of a class, given:
  * The class has 20 students
  * One student's height was incorrectly recorded as 40 cm more than their actual height
  * The actual average height of the students is 173 cm
-/
def initiallyCalculatedAverage (numStudents : ℕ) (heightError : ℕ) (actualAverage : ℕ) : ℕ :=
  actualAverage + heightError / numStudents

/-- Theorem stating that the initially calculated average height is 175 cm -/
theorem initially_calculated_average_is_175 :
  initiallyCalculatedAverage 20 40 173 = 175 := by
  sorry

end NUMINAMATH_CALUDE_initially_calculated_average_is_175_l2088_208818


namespace NUMINAMATH_CALUDE_equation_solutions_l2088_208884

theorem equation_solutions : 
  ∀ x y z : ℕ+, 
    (x.val * y.val + y.val * z.val + z.val * x.val - x.val * y.val * z.val = 2) ↔ 
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨ 
     (x = 2 ∧ y = 3 ∧ z = 4) ∨ (x = 2 ∧ y = 4 ∧ z = 3) ∨
     (x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 3 ∧ y = 4 ∧ z = 2) ∨
     (x = 4 ∧ y = 2 ∧ z = 3) ∨ (x = 4 ∧ y = 3 ∧ z = 2)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2088_208884


namespace NUMINAMATH_CALUDE_initial_concentration_is_40_percent_l2088_208855

-- Define the capacities and concentrations
def vessel1_capacity : ℝ := 2
def vessel2_capacity : ℝ := 6
def vessel2_concentration : ℝ := 0.6
def total_liquid : ℝ := 8
def new_vessel_capacity : ℝ := 10
def final_concentration : ℝ := 0.44

-- Define the unknown initial concentration of vessel 1
def vessel1_concentration : ℝ := sorry

-- Theorem statement
theorem initial_concentration_is_40_percent :
  vessel1_concentration * vessel1_capacity + 
  vessel2_concentration * vessel2_capacity = 
  final_concentration * new_vessel_capacity := by
  sorry

end NUMINAMATH_CALUDE_initial_concentration_is_40_percent_l2088_208855


namespace NUMINAMATH_CALUDE_terminal_side_angle_theorem_l2088_208889

theorem terminal_side_angle_theorem (α : Real) :
  (∃ (x y : Real), x = -2 ∧ y = 1 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  1 / Real.sin (2 * α) = -5/4 := by
sorry

end NUMINAMATH_CALUDE_terminal_side_angle_theorem_l2088_208889


namespace NUMINAMATH_CALUDE_orchard_sections_l2088_208888

/-- The number of sections in an apple orchard --/
def number_of_sections (daily_harvest_per_section : ℕ) (total_daily_harvest : ℕ) : ℕ :=
  total_daily_harvest / daily_harvest_per_section

/-- Theorem stating that the number of sections in the orchard is 8 --/
theorem orchard_sections :
  let daily_harvest_per_section := 45
  let total_daily_harvest := 360
  number_of_sections daily_harvest_per_section total_daily_harvest = 8 := by
  sorry

end NUMINAMATH_CALUDE_orchard_sections_l2088_208888


namespace NUMINAMATH_CALUDE_base3_to_base10_20123_l2088_208829

/-- Converts a base 3 number to base 10 --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number --/
def base3Number : List Nat := [3, 2, 1, 0, 2]

/-- Theorem stating that the base 10 equivalent of 20123 (base 3) is 180 --/
theorem base3_to_base10_20123 :
  base3ToBase10 base3Number = 180 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_20123_l2088_208829


namespace NUMINAMATH_CALUDE_yuna_has_biggest_number_l2088_208860

def yoongi_number : ℕ := 7
def jungkook_number : ℕ := 6
def yuna_number : ℕ := 9

theorem yuna_has_biggest_number :
  yuna_number = max yoongi_number (max jungkook_number yuna_number) :=
by sorry

end NUMINAMATH_CALUDE_yuna_has_biggest_number_l2088_208860


namespace NUMINAMATH_CALUDE_lemonade_pitchers_l2088_208878

theorem lemonade_pitchers (glasses_per_pitcher : ℕ) (total_glasses : ℕ) (h1 : glasses_per_pitcher = 5) (h2 : total_glasses = 30) :
  total_glasses / glasses_per_pitcher = 6 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_pitchers_l2088_208878


namespace NUMINAMATH_CALUDE_freds_allowance_l2088_208849

/-- Fred's weekly allowance problem -/
theorem freds_allowance (allowance : ℝ) : 
  (allowance / 2 + 11 = 20) → allowance = 18 := by
  sorry

end NUMINAMATH_CALUDE_freds_allowance_l2088_208849


namespace NUMINAMATH_CALUDE_min_value_quadratic_roots_l2088_208874

theorem min_value_quadratic_roots (a b c : ℤ) (α β : ℝ) : 
  a > 0 → 
  (∃ x : ℝ, a * x^2 + b * x + c = 0) → 
  (α * α * a + b * α + c = 0) →
  (β * β * a + b * β + c = 0) →
  0 < α → α < β → β < 1 → 
  a ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_roots_l2088_208874


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l2088_208825

theorem youngest_sibling_age 
  (siblings : Fin 4 → ℕ) 
  (age_differences : ∀ i : Fin 4, siblings i = siblings 0 + [0, 2, 7, 11].get i) 
  (average_age : (siblings 0 + siblings 1 + siblings 2 + siblings 3) / 4 = 25) : 
  siblings 0 = 20 := by
  sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l2088_208825


namespace NUMINAMATH_CALUDE_distinct_polygons_count_l2088_208815

/-- The number of points marked on the circle -/
def n : ℕ := 15

/-- The total number of subsets of n points -/
def total_subsets : ℕ := 2^n

/-- The number of subsets with 0, 1, 2, or 3 members -/
def small_subsets : ℕ := (n.choose 0) + (n.choose 1) + (n.choose 2) + (n.choose 3)

/-- The maximum number of points that can lie on a semicircle -/
def max_semicircle : ℕ := n / 2 + 1

/-- The number of subsets that lie on a semicircle -/
def semicircle_subsets : ℕ := 2^max_semicircle - 1

/-- Conservative estimate of subsets to exclude due to lying on the same semicircle -/
def conservative_exclusion : ℕ := 500

/-- The number of distinct convex polygons with 4 or more sides -/
def distinct_polygons : ℕ := total_subsets - small_subsets - semicircle_subsets - conservative_exclusion

theorem distinct_polygons_count :
  distinct_polygons = 31437 :=
sorry

end NUMINAMATH_CALUDE_distinct_polygons_count_l2088_208815


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2088_208882

theorem geometric_series_ratio (a : ℝ) (r : ℝ) : 
  (a / (1 - r) = 20) → 
  (a / (1 - r^2) = 8) → 
  (r ≠ 1) →
  r = 3/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2088_208882


namespace NUMINAMATH_CALUDE_special_quadrilateral_area_sum_l2088_208843

/-- A convex quadrilateral with specific side lengths and angle -/
structure ConvexQuadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  angleCDA : ℝ
  convex : AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0
  angleCondition : 0 < angleCDA ∧ angleCDA < π

/-- The area of the quadrilateral can be expressed in the form √a + b√c -/
def hasSpecialAreaForm (q : ConvexQuadrilateral) (a b c : ℕ) : Prop :=
  ∃ (area : ℝ), area = Real.sqrt a + b * Real.sqrt c ∧
  area = q.AB * q.BC * Real.sin q.angleCDA / 2 + q.CD * q.DA * Real.sin q.angleCDA / 2 ∧
  ∀ k : ℕ, k > 1 → (k * k ∣ a → k = 1) ∧ (k * k ∣ c → k = 1)

/-- Main theorem -/
theorem special_quadrilateral_area_sum (q : ConvexQuadrilateral) 
    (h1 : q.AB = 8) (h2 : q.BC = 4) (h3 : q.CD = 10) (h4 : q.DA = 10) 
    (h5 : q.angleCDA = π/3) (a b c : ℕ) (h6 : hasSpecialAreaForm q a b c) : 
    a + b + c = 259 := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_area_sum_l2088_208843


namespace NUMINAMATH_CALUDE_range_of_a_l2088_208891

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 4) → -3 ≤ a ∧ a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2088_208891

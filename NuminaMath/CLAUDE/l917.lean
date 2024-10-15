import Mathlib

namespace NUMINAMATH_CALUDE_badminton_cost_theorem_l917_91764

/-- Represents the cost of badminton equipment under different purchasing options -/
def BadmintonCost (x : ℕ) : Prop :=
  let racket_price : ℕ := 40
  let shuttlecock_price : ℕ := 10
  let racket_quantity : ℕ := 10
  let shuttlecock_quantity : ℕ := x
  let option1_cost : ℕ := 10 * x + 300
  let option2_cost : ℕ := 9 * x + 360
  x > 10 ∧
  option1_cost = racket_price * racket_quantity + shuttlecock_price * (shuttlecock_quantity - racket_quantity) ∧
  option2_cost = (racket_price * racket_quantity + shuttlecock_price * shuttlecock_quantity) * 9 / 10 ∧
  (x = 30 → option1_cost < option2_cost) ∧
  ∃ (better_cost : ℕ), x = 30 → better_cost < option1_cost ∧ better_cost < option2_cost

theorem badminton_cost_theorem : 
  ∀ x : ℕ, BadmintonCost x :=
sorry

end NUMINAMATH_CALUDE_badminton_cost_theorem_l917_91764


namespace NUMINAMATH_CALUDE_triangle_selection_probability_l917_91724

theorem triangle_selection_probability (total_triangles shaded_triangles : ℕ) 
  (h1 : total_triangles = 6)
  (h2 : shaded_triangles = 3)
  (h3 : shaded_triangles ≤ total_triangles)
  (h4 : total_triangles > 0) :
  (shaded_triangles : ℚ) / total_triangles = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_selection_probability_l917_91724


namespace NUMINAMATH_CALUDE_point_P_coordinates_l917_91702

/-- The mapping f that transforms a point (x, y) to (x+y, 2x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, 2 * p.1 - p.2)

/-- Theorem stating that if P is mapped to (5, 1) under f, then P has coordinates (2, 3) -/
theorem point_P_coordinates (P : ℝ × ℝ) (h : f P = (5, 1)) : P = (2, 3) := by
  sorry


end NUMINAMATH_CALUDE_point_P_coordinates_l917_91702


namespace NUMINAMATH_CALUDE_g_of_two_value_l917_91725

/-- Given a function g : ℝ → ℝ satisfying g(x) + 3 * g(1 - x) = 4 * x^3 - 5 * x for all real x, 
    prove that g(2) = -19/6 -/
theorem g_of_two_value (g : ℝ → ℝ) 
    (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x^3 - 5 * x) : 
  g 2 = -19/6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_two_value_l917_91725


namespace NUMINAMATH_CALUDE_range_of_m_l917_91797

-- Define the function f
def f (x : ℝ) := x^3 + x

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 → f (m * Real.sin θ) + f (1 - m) > 0) →
  m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l917_91797


namespace NUMINAMATH_CALUDE_fraction_addition_l917_91745

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 11 = (37 : ℚ) / 55 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l917_91745


namespace NUMINAMATH_CALUDE_speed_in_fifth_hour_l917_91751

def speed_hour1 : ℝ := 90
def speed_hour2 : ℝ := 60
def speed_hour3 : ℝ := 120
def speed_hour4 : ℝ := 72
def avg_speed : ℝ := 80
def total_time : ℝ := 5

def total_distance : ℝ := avg_speed * total_time

def distance_first_four_hours : ℝ := speed_hour1 + speed_hour2 + speed_hour3 + speed_hour4

def speed_hour5 : ℝ := total_distance - distance_first_four_hours

theorem speed_in_fifth_hour :
  speed_hour5 = 58 := by sorry

end NUMINAMATH_CALUDE_speed_in_fifth_hour_l917_91751


namespace NUMINAMATH_CALUDE_bookcase_weight_excess_l917_91743

/-- Proves that the total weight of books and knick-knacks exceeds the bookcase weight limit by 33 pounds -/
theorem bookcase_weight_excess :
  let bookcase_limit : ℕ := 80
  let hardcover_count : ℕ := 70
  let hardcover_weight : ℚ := 1/2
  let textbook_count : ℕ := 30
  let textbook_weight : ℕ := 2
  let knickknack_count : ℕ := 3
  let knickknack_weight : ℕ := 6
  let total_weight := (hardcover_count : ℚ) * hardcover_weight + 
                      (textbook_count * textbook_weight : ℚ) + 
                      (knickknack_count * knickknack_weight : ℚ)
  total_weight - bookcase_limit = 33
  := by sorry

end NUMINAMATH_CALUDE_bookcase_weight_excess_l917_91743


namespace NUMINAMATH_CALUDE_average_population_is_1000_l917_91742

def village_populations : List ℕ := [803, 900, 1100, 1023, 945, 980, 1249]

theorem average_population_is_1000 : 
  (village_populations.sum / village_populations.length : ℚ) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_average_population_is_1000_l917_91742


namespace NUMINAMATH_CALUDE_sqrt_15_minus_1_range_l917_91735

theorem sqrt_15_minus_1_range : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_minus_1_range_l917_91735


namespace NUMINAMATH_CALUDE_subway_scenarios_l917_91789

/-- Represents the fare structure for the subway -/
def fare (x : ℕ) : ℕ :=
  if x ≤ 4 then 2
  else if x ≤ 9 then 4
  else if x ≤ 15 then 6
  else 0

/-- The maximum number of stations -/
def max_stations : ℕ := 15

/-- Calculates the number of scenarios where two passengers pay a total fare -/
def scenarios_for_total_fare (total_fare : ℕ) : ℕ := sorry

/-- Calculates the number of scenarios where passenger A gets off before passenger B -/
def scenarios_a_before_b (total_fare : ℕ) : ℕ := sorry

theorem subway_scenarios :
  (scenarios_for_total_fare 6 = 40) ∧
  (scenarios_a_before_b 8 = 34) := by sorry

end NUMINAMATH_CALUDE_subway_scenarios_l917_91789


namespace NUMINAMATH_CALUDE_beaver_count_correct_l917_91731

/-- The number of beavers in the first scenario -/
def num_beavers : ℕ := 20

/-- The time taken by the first group of beavers to build the dam -/
def time_first : ℕ := 18

/-- The number of beavers in the second scenario -/
def num_beavers_second : ℕ := 12

/-- The time taken by the second group of beavers to build the dam -/
def time_second : ℕ := 30

/-- The theorem stating that the calculated number of beavers is correct -/
theorem beaver_count_correct :
  num_beavers * time_first = num_beavers_second * time_second :=
by sorry

end NUMINAMATH_CALUDE_beaver_count_correct_l917_91731


namespace NUMINAMATH_CALUDE_combined_average_age_l917_91741

theorem combined_average_age (x_count y_count z_count : ℕ) 
  (x_avg y_avg z_avg : ℝ) : 
  x_count = 5 → 
  y_count = 3 → 
  z_count = 2 → 
  x_avg = 35 → 
  y_avg = 30 → 
  z_avg = 45 → 
  (x_count * x_avg + y_count * y_avg + z_count * z_avg) / (x_count + y_count + z_count) = 35.5 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_age_l917_91741


namespace NUMINAMATH_CALUDE_max_consecutive_sum_under_1000_l917_91794

theorem max_consecutive_sum_under_1000 : 
  (∀ k : ℕ, k ≤ 44 → k * (k + 1) ≤ 2000) ∧ 
  45 * 46 > 2000 := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_under_1000_l917_91794


namespace NUMINAMATH_CALUDE_max_planks_from_trunk_l917_91721

/-- Represents a cylindrical tree trunk -/
structure Trunk :=
  (diameter : ℝ)

/-- Represents a plank with thickness and width -/
structure Plank :=
  (thickness : ℝ)
  (width : ℝ)

/-- Calculates the maximum number of planks that can be cut from a trunk -/
def max_planks (t : Trunk) (p : Plank) : ℕ :=
  sorry

/-- Theorem stating the maximum number of planks that can be cut -/
theorem max_planks_from_trunk (t : Trunk) (p : Plank) :
  t.diameter = 46 → p.thickness = 4 → p.width = 12 → max_planks t p = 29 := by
  sorry

end NUMINAMATH_CALUDE_max_planks_from_trunk_l917_91721


namespace NUMINAMATH_CALUDE_cos_equality_with_large_angle_l917_91713

theorem cos_equality_with_large_angle (n : ℕ) :
  0 ≤ n ∧ n ≤ 200 →
  n = 166 →
  Real.cos (n * π / 180) = Real.cos (1274 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_cos_equality_with_large_angle_l917_91713


namespace NUMINAMATH_CALUDE_fraction_difference_l917_91759

theorem fraction_difference (n d : ℤ) : 
  d = 5 → n > d → n + 6 = 3 * d → n - d = 4 := by sorry

end NUMINAMATH_CALUDE_fraction_difference_l917_91759


namespace NUMINAMATH_CALUDE_bus_seating_solution_l917_91710

/-- Represents the seating configuration of a bus -/
structure BusSeating where
  left_seats : Nat
  right_seats : Nat
  back_seat_capacity : Nat
  total_capacity : Nat

/-- Calculates the number of people each regular seat can hold -/
def seats_capacity (bus : BusSeating) : Nat :=
  let regular_seats := bus.left_seats + bus.right_seats
  let regular_capacity := bus.total_capacity - bus.back_seat_capacity
  regular_capacity / regular_seats

/-- Theorem stating the solution to the bus seating problem -/
theorem bus_seating_solution :
  let bus := BusSeating.mk 15 12 11 92
  seats_capacity bus = 3 := by sorry

end NUMINAMATH_CALUDE_bus_seating_solution_l917_91710


namespace NUMINAMATH_CALUDE_stratified_sample_male_count_l917_91729

/-- Represents a company with employees -/
structure Company where
  total_employees : ℕ
  female_employees : ℕ
  male_employees : ℕ
  h_total : total_employees = female_employees + male_employees

/-- Represents a stratified sample from a company -/
structure StratifiedSample where
  company : Company
  sample_size : ℕ
  female_sample : ℕ
  male_sample : ℕ
  h_sample : sample_size = female_sample + male_sample
  h_proportion : female_sample * company.total_employees = company.female_employees * sample_size

theorem stratified_sample_male_count (c : Company) (s : StratifiedSample) 
    (h_company : c.total_employees = 300 ∧ c.female_employees = 160)
    (h_sample : s.company = c ∧ s.sample_size = 15) :
    s.male_sample = 7 := by
  sorry

#check stratified_sample_male_count

end NUMINAMATH_CALUDE_stratified_sample_male_count_l917_91729


namespace NUMINAMATH_CALUDE_students_suggesting_pasta_l917_91793

theorem students_suggesting_pasta 
  (total_students : ℕ) 
  (mashed_potatoes : ℕ) 
  (bacon : ℕ) 
  (h1 : total_students = 470) 
  (h2 : mashed_potatoes = 230) 
  (h3 : bacon = 140) : 
  total_students - (mashed_potatoes + bacon) = 100 := by
sorry

end NUMINAMATH_CALUDE_students_suggesting_pasta_l917_91793


namespace NUMINAMATH_CALUDE_max_gcd_of_sequence_l917_91779

theorem max_gcd_of_sequence (n : ℕ+) : Nat.gcd (101 + n^3) (101 + (n + 1)^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_of_sequence_l917_91779


namespace NUMINAMATH_CALUDE_platform_length_theorem_l917_91790

def train_length : ℝ := 250

theorem platform_length_theorem (X Y : ℝ) (platform_time signal_time : ℝ) 
  (h1 : platform_time = 40)
  (h2 : signal_time = 20)
  (h3 : Y * signal_time = train_length) :
  Y = 12.5 ∧ ∃ L, L = X * platform_time - train_length := by
  sorry

#check platform_length_theorem

end NUMINAMATH_CALUDE_platform_length_theorem_l917_91790


namespace NUMINAMATH_CALUDE_parabola_properties_l917_91796

-- Define the parabola
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 3

-- Define the conditions
theorem parabola_properties :
  ∃ (a b : ℝ),
    (parabola a b 3 = 0) ∧
    (parabola a b 4 = 3) ∧
    (∀ x, parabola a b x = x^2 - 4*x + 3) ∧
    (a > 0) ∧
    (∀ x, parabola a b x ≥ parabola a b 2) ∧
    (parabola a b 2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l917_91796


namespace NUMINAMATH_CALUDE_arithmetic_mean_special_set_l917_91774

theorem arithmetic_mean_special_set (n : ℕ) (h : n > 1) :
  let set := List.replicate (n - 1) 1 ++ [1 + 1 / n]
  (set.sum / n : ℚ) = 1 + 1 / n^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_special_set_l917_91774


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l917_91703

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  b = Real.sqrt 6 →
  c = 3 →
  C = 60 * π / 180 →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a / Real.sin A = c / Real.sin C) →
  A + B + C = π →
  A = 75 * π / 180 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l917_91703


namespace NUMINAMATH_CALUDE_expression_value_l917_91726

theorem expression_value (a b : ℤ) (h : a - b = 1) : 2*b - (2*a + 6) = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l917_91726


namespace NUMINAMATH_CALUDE_complex_equation_solution_l917_91761

theorem complex_equation_solution (z : ℂ) :
  z * (1 + Complex.I) = -2 * Complex.I → z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l917_91761


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_9_l917_91705

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def five_digit_number (x : ℕ) : ℕ := 24600 + 10 * x + 8

theorem missing_digit_divisible_by_9 :
  ∀ x : ℕ, x < 10 →
    (is_divisible_by_9 (five_digit_number x) ↔ x = 7) :=
by sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_9_l917_91705


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_7_l917_91767

theorem smallest_perfect_square_divisible_by_5_and_7 : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (m : ℕ), n = m ^ 2) ∧ 
  5 ∣ n ∧ 
  7 ∣ n ∧ 
  (∀ (k : ℕ), k > 0 → (∃ (l : ℕ), k = l ^ 2) → 5 ∣ k → 7 ∣ k → k ≥ n) ∧
  n = 1225 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_7_l917_91767


namespace NUMINAMATH_CALUDE_all_terms_are_integers_l917_91728

/-- An infinite increasing arithmetic progression where the product of any two distinct terms is also a term in the progression. -/
structure SpecialArithmeticProgression where
  -- The sequence of terms in the progression
  sequence : ℕ → ℚ
  -- The common difference of the progression
  common_difference : ℚ
  -- The progression is increasing
  increasing : ∀ n : ℕ, sequence n < sequence (n + 1)
  -- The progression follows the arithmetic sequence formula
  is_arithmetic : ∀ n : ℕ, sequence (n + 1) = sequence n + common_difference
  -- The product of any two distinct terms is also a term
  product_is_term : ∀ m n : ℕ, m ≠ n → ∃ k : ℕ, sequence m * sequence n = sequence k

/-- All terms in a SpecialArithmeticProgression are integers. -/
theorem all_terms_are_integers (ap : SpecialArithmeticProgression) : 
  ∀ n : ℕ, ∃ k : ℤ, ap.sequence n = k :=
sorry

end NUMINAMATH_CALUDE_all_terms_are_integers_l917_91728


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l917_91708

theorem polynomial_coefficient_sum :
  ∀ A B C D E : ℝ,
  (∀ x : ℝ, (x + 3) * (4 * x^3 - 2 * x^2 + 3 * x - 1) = A * x^4 + B * x^3 + C * x^2 + D * x + E) →
  A + B + C + D + E = 16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l917_91708


namespace NUMINAMATH_CALUDE_passing_percentage_l917_91757

theorem passing_percentage (max_marks : ℕ) (pradeep_marks : ℕ) (failed_by : ℕ) 
  (h1 : max_marks = 925)
  (h2 : pradeep_marks = 160)
  (h3 : failed_by = 25) :
  (((pradeep_marks + failed_by : ℚ) / max_marks) * 100 : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_l917_91757


namespace NUMINAMATH_CALUDE_hua_optimal_selection_uses_golden_ratio_l917_91776

/-- The mathematical concept used in Hua Luogeng's optimal selection method -/
inductive OptimalSelectionConcept
  | GoldenRatio
  | Mean
  | Mode
  | Median

/-- Hua Luogeng's optimal selection method -/
def huaOptimalSelectionMethod : OptimalSelectionConcept := OptimalSelectionConcept.GoldenRatio

/-- Theorem: The mathematical concept used in Hua Luogeng's optimal selection method is the Golden ratio -/
theorem hua_optimal_selection_uses_golden_ratio :
  huaOptimalSelectionMethod = OptimalSelectionConcept.GoldenRatio := by
  sorry

end NUMINAMATH_CALUDE_hua_optimal_selection_uses_golden_ratio_l917_91776


namespace NUMINAMATH_CALUDE_exam_questions_count_l917_91782

/-- Exam scoring system and student performance -/
structure ExamScoring where
  correct_score : Int
  incorrect_penalty : Int
  total_score : Int
  correct_answers : Int

/-- Calculate the total number of questions in the exam -/
def total_questions (exam : ExamScoring) : Int :=
  exam.correct_answers + (exam.total_score - exam.correct_score * exam.correct_answers) / (-exam.incorrect_penalty)

/-- Theorem: The total number of questions in the exam is 150 -/
theorem exam_questions_count (exam : ExamScoring) 
  (h1 : exam.correct_score = 4)
  (h2 : exam.incorrect_penalty = 2)
  (h3 : exam.total_score = 420)
  (h4 : exam.correct_answers = 120) : 
  total_questions exam = 150 := by
  sorry


end NUMINAMATH_CALUDE_exam_questions_count_l917_91782


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l917_91766

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l917_91766


namespace NUMINAMATH_CALUDE_divisors_of_factorial_eight_l917_91750

theorem divisors_of_factorial_eight (n : ℕ) : n = 8 → (Nat.divisors (Nat.factorial n)).card = 96 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_factorial_eight_l917_91750


namespace NUMINAMATH_CALUDE_student_count_l917_91770

theorem student_count (average_student_age : ℝ) (teacher_age : ℝ) (new_average_age : ℝ) :
  average_student_age = 15 →
  teacher_age = 26 →
  new_average_age = 16 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * average_student_age + teacher_age = (n + 1 : ℝ) * new_average_age ∧
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l917_91770


namespace NUMINAMATH_CALUDE_zero_sequence_arithmetic_not_geometric_l917_91791

def zero_sequence : ℕ → ℝ := λ _ => 0

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem zero_sequence_arithmetic_not_geometric :
  is_arithmetic_sequence zero_sequence ∧ ¬is_geometric_sequence zero_sequence := by
  sorry

end NUMINAMATH_CALUDE_zero_sequence_arithmetic_not_geometric_l917_91791


namespace NUMINAMATH_CALUDE_sector_area_l917_91704

/-- Given a sector with central angle α and arc length l, 
    the area S of the sector is (l * l) / (2 * α) -/
theorem sector_area (α : ℝ) (l : ℝ) (S : ℝ) 
  (h1 : α = 2) 
  (h2 : l = 3 * Real.pi) 
  (h3 : S = (l * l) / (2 * α)) : 
  S = (9 * Real.pi^2) / 4 := by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l917_91704


namespace NUMINAMATH_CALUDE_sum_reciprocal_plus_one_bounds_l917_91788

theorem sum_reciprocal_plus_one_bounds (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  1 < (1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z)) ∧ 
  (1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z)) < 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_plus_one_bounds_l917_91788


namespace NUMINAMATH_CALUDE_hcd_8580_330_minus_12_l917_91769

theorem hcd_8580_330_minus_12 : Nat.gcd 8580 330 - 12 = 318 := by
  sorry

end NUMINAMATH_CALUDE_hcd_8580_330_minus_12_l917_91769


namespace NUMINAMATH_CALUDE_inequality_solution_l917_91771

theorem inequality_solution (x : ℝ) (h : x ≠ 2) :
  |((3 * x - 2) / (x - 2))| > 3 ↔ x < 4/3 ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l917_91771


namespace NUMINAMATH_CALUDE_basil_plants_theorem_l917_91712

/-- Calculates the number of basil plants sold given the costs, selling price, and net profit -/
def basil_plants_sold (seed_cost potting_soil_cost selling_price net_profit : ℚ) : ℚ :=
  (net_profit + seed_cost + potting_soil_cost) / selling_price

/-- Theorem stating that the number of basil plants sold is 20 -/
theorem basil_plants_theorem :
  basil_plants_sold 2 8 5 90 = 20 := by
  sorry

end NUMINAMATH_CALUDE_basil_plants_theorem_l917_91712


namespace NUMINAMATH_CALUDE_question_mark_value_l917_91727

theorem question_mark_value : ∃ (x : ℕ), x * 40 = 173 * 240 ∧ x = 1036 := by
  sorry

end NUMINAMATH_CALUDE_question_mark_value_l917_91727


namespace NUMINAMATH_CALUDE_bob_salary_calculation_l917_91722

def initial_salary : ℝ := 3000
def raise_percentage : ℝ := 0.15
def cut_percentage : ℝ := 0.10
def bonus : ℝ := 500

def final_salary : ℝ := 
  initial_salary * (1 + raise_percentage) * (1 - cut_percentage) + bonus

theorem bob_salary_calculation : final_salary = 3605 := by
  sorry

end NUMINAMATH_CALUDE_bob_salary_calculation_l917_91722


namespace NUMINAMATH_CALUDE_initial_classes_l917_91763

theorem initial_classes (initial_classes : ℕ) : 
  (20 * initial_classes : ℕ) + (20 * 5 : ℕ) = 400 → initial_classes = 15 :=
by sorry

end NUMINAMATH_CALUDE_initial_classes_l917_91763


namespace NUMINAMATH_CALUDE_quadratic_roots_bounds_l917_91799

theorem quadratic_roots_bounds (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m < 0) 
  (hroots : x₁^2 - x₁ - 6 = m ∧ x₂^2 - x₂ - 6 = m) 
  (horder : x₁ < x₂) : 
  -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_bounds_l917_91799


namespace NUMINAMATH_CALUDE_no_integer_solution_for_175_l917_91795

theorem no_integer_solution_for_175 :
  ∀ x y : ℤ, x^2 + y^2 ≠ 175 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_175_l917_91795


namespace NUMINAMATH_CALUDE_drinking_speed_ratio_l917_91707

theorem drinking_speed_ratio 
  (total_volume : ℝ) 
  (mala_volume : ℝ) 
  (usha_volume : ℝ) 
  (drinking_time : ℝ) 
  (h1 : drinking_time > 0) 
  (h2 : total_volume > 0) 
  (h3 : mala_volume + usha_volume = total_volume) 
  (h4 : usha_volume = 2 / 10 * total_volume) : 
  (mala_volume / drinking_time) / (usha_volume / drinking_time) = 4 := by
sorry

end NUMINAMATH_CALUDE_drinking_speed_ratio_l917_91707


namespace NUMINAMATH_CALUDE_probability_black_second_draw_l917_91718

theorem probability_black_second_draw 
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (black_balls : ℕ) 
  (h1 : total_balls = 5) 
  (h2 : white_balls = 3) 
  (h3 : black_balls = 2) 
  (h4 : total_balls = white_balls + black_balls) 
  (h5 : white_balls > 0) : 
  (black_balls : ℚ) / (total_balls - 1 : ℚ) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_probability_black_second_draw_l917_91718


namespace NUMINAMATH_CALUDE_circle_properties_l917_91784

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*y - 36 = -y^2 + 14*x + 4

def is_center_and_radius (c d s : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - c)^2 + (y - d)^2 = s^2

theorem circle_properties :
  ∃ c d s : ℝ, is_center_and_radius c d s ∧ c = 7 ∧ d = 2 ∧ s^2 = 93 ∧ c + d + s = 9 + Real.sqrt 93 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l917_91784


namespace NUMINAMATH_CALUDE_min_value_theorem_l917_91711

/-- The function f(x) = |2x - 1| -/
def f (x : ℝ) : ℝ := |2 * x - 1|

/-- The function g(x) = f(x) + f(x - 1) -/
def g (x : ℝ) : ℝ := f x + f (x - 1)

/-- The minimum value of g(x) -/
def a : ℝ := 2

/-- Theorem: The minimum value of (m^2 + 2)/m + (n^2 + 1)/n is (7 + 2√2)/2,
    given m + n = a and m, n > 0 -/
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = a) :
  (m^2 + 2)/m + (n^2 + 1)/n ≥ (7 + 2 * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l917_91711


namespace NUMINAMATH_CALUDE_crayon_distribution_l917_91737

theorem crayon_distribution (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) :
  total_crayons = 24 →
  num_people = 3 →
  total_crayons = num_people * crayons_per_person →
  crayons_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_crayon_distribution_l917_91737


namespace NUMINAMATH_CALUDE_factorization_sum_l917_91755

theorem factorization_sum (a b c : ℤ) : 
  (∀ x, x^2 + 17*x + 72 = (x + a) * (x + b)) →
  (∀ x, x^2 + 9*x - 90 = (x + b) * (x - c)) →
  a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l917_91755


namespace NUMINAMATH_CALUDE_calculation_proof_l917_91753

theorem calculation_proof : 72 / (6 / 3) * 2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l917_91753


namespace NUMINAMATH_CALUDE_roots_imply_f_value_l917_91700

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 2*x + 15
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 75*x + c

-- State the theorem
theorem roots_imply_f_value (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0 ∧
    f b c r₁ = 0 ∧ f b c r₂ = 0 ∧ f b c r₃ = 0) →
  f b c (-1) = -2773 := by
  sorry

end NUMINAMATH_CALUDE_roots_imply_f_value_l917_91700


namespace NUMINAMATH_CALUDE_arrangement_counts_l917_91780

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 4

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

theorem arrangement_counts :
  (∃ (n₁ n₂ n₃ n₄ : ℕ),
    /- (1) Person A and Person B at the ends -/
    n₁ = 240 ∧
    /- (2) All male students grouped together -/
    n₂ = 720 ∧
    /- (3) No male students next to each other -/
    n₃ = 1440 ∧
    /- (4) Exactly one person between Person A and Person B -/
    n₄ = 1200 ∧
    /- The numbers represent valid arrangement counts -/
    n₁ > 0 ∧ n₂ > 0 ∧ n₃ > 0 ∧ n₄ > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_arrangement_counts_l917_91780


namespace NUMINAMATH_CALUDE_sequence_matches_given_terms_l917_91792

/-- The general term of the sequence -/
def a (n : ℕ) : ℚ := n + n^2 / (n^2 + 1)

/-- The first four terms of the sequence match the given values -/
theorem sequence_matches_given_terms :
  (a 1 = 3/2) ∧ 
  (a 2 = 14/5) ∧ 
  (a 3 = 39/10) ∧ 
  (a 4 = 84/17) := by
  sorry

end NUMINAMATH_CALUDE_sequence_matches_given_terms_l917_91792


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l917_91760

/-- Theorem: Volume of a cube with surface area 150 square inches --/
theorem cube_volume_from_surface_area :
  let surface_area : ℝ := 150  -- Surface area in square inches
  let edge_length : ℝ := Real.sqrt (surface_area / 6)  -- Edge length in inches
  let volume_cubic_inches : ℝ := edge_length ^ 3  -- Volume in cubic inches
  let cubic_inches_per_cubic_foot : ℝ := 1728  -- Conversion factor
  let volume_cubic_feet : ℝ := volume_cubic_inches / cubic_inches_per_cubic_foot
  ∃ ε > 0, |volume_cubic_feet - 0.0723| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l917_91760


namespace NUMINAMATH_CALUDE_student_weight_l917_91736

theorem student_weight (student_weight sister_weight : ℝ) :
  student_weight - 5 = 2 * sister_weight →
  student_weight + sister_weight = 104 →
  student_weight = 71 := by
sorry

end NUMINAMATH_CALUDE_student_weight_l917_91736


namespace NUMINAMATH_CALUDE_ratio_equality_l917_91778

theorem ratio_equality (a b c : ℝ) (h : a/2 = b/3 ∧ b/3 = c/4 ∧ a/2 ≠ 0) : 
  (a - 2*c) / (a - 2*b) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l917_91778


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l917_91714

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l917_91714


namespace NUMINAMATH_CALUDE_mod_power_minus_three_l917_91749

theorem mod_power_minus_three (m : ℕ) : 
  0 ≤ m ∧ m < 37 ∧ (4 * m) % 37 = 1 → 
  (((3 ^ m) ^ 4 - 3) : ℤ) % 37 = 25 := by
  sorry

end NUMINAMATH_CALUDE_mod_power_minus_three_l917_91749


namespace NUMINAMATH_CALUDE_oak_grove_library_books_l917_91758

/-- The number of books in Oak Grove's school libraries -/
def school_books : ℕ := 5106

/-- The total number of books in all Oak Grove libraries -/
def total_books : ℕ := 7092

/-- The number of books in Oak Grove's public library -/
def public_books : ℕ := total_books - school_books

theorem oak_grove_library_books : public_books = 1986 := by
  sorry

end NUMINAMATH_CALUDE_oak_grove_library_books_l917_91758


namespace NUMINAMATH_CALUDE_constant_fifth_term_implies_n_six_l917_91717

/-- 
Given a positive integer n, and considering the binomial expansion of (x^2 + 1/x)^n,
if the fifth term is a constant (i.e., the exponent of x is 0), then n must equal 6.
-/
theorem constant_fifth_term_implies_n_six (n : ℕ+) : 
  (∃ k : ℕ, k > 0 ∧ 2*n - 3*(k+1) = 0) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_constant_fifth_term_implies_n_six_l917_91717


namespace NUMINAMATH_CALUDE_bingley_has_six_bracelets_l917_91754

/-- The number of bracelets Bingley has remaining after the exchanges -/
def bingleys_remaining_bracelets (bingley_initial : ℕ) (kelly_initial : ℕ) : ℕ :=
  let bingley_after_kelly := bingley_initial + kelly_initial / 4
  bingley_after_kelly - bingley_after_kelly / 3

/-- Theorem stating that Bingley will have 6 bracelets remaining -/
theorem bingley_has_six_bracelets : 
  bingleys_remaining_bracelets 5 16 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bingley_has_six_bracelets_l917_91754


namespace NUMINAMATH_CALUDE_i_pow_45_plus_345_l917_91723

-- Define the imaginary unit i
axiom i : ℂ
axiom i_squared : i^2 = -1

-- Define the properties of i
axiom i_pow_one : i^1 = i
axiom i_pow_two : i^2 = -1
axiom i_pow_three : i^3 = -i
axiom i_pow_four : i^4 = 1

-- Define the cyclic nature of i
axiom i_cyclic (n : ℕ) : i^(n + 4) = i^n

-- Theorem to prove
theorem i_pow_45_plus_345 : i^45 + i^345 = 2*i := by
  sorry

end NUMINAMATH_CALUDE_i_pow_45_plus_345_l917_91723


namespace NUMINAMATH_CALUDE_expression_evaluation_l917_91715

theorem expression_evaluation (c d : ℝ) (hc : c = 3) (hd : d = 2) :
  (c^2 + d)^2 - (c^2 - d)^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l917_91715


namespace NUMINAMATH_CALUDE_cupcakes_eaten_l917_91732

theorem cupcakes_eaten (total_baked : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) : 
  total_baked = 68 →
  packages = 6 →
  cupcakes_per_package = 6 →
  total_baked - (packages * cupcakes_per_package) = 
    total_baked - (total_baked - (packages * cupcakes_per_package)) := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_eaten_l917_91732


namespace NUMINAMATH_CALUDE_x_value_l917_91747

theorem x_value (m : ℕ) (x : ℝ) 
  (h1 : m = 34) 
  (h2 : ((x ^ (m + 1)) / (5 ^ (m + 1))) * ((x ^ 18) / (4 ^ 18)) = 1 / (2 * (10 ^ 35))) :
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_x_value_l917_91747


namespace NUMINAMATH_CALUDE_largest_power_of_three_dividing_A_l917_91744

/-- Given that A is the largest product of natural numbers whose sum is 2011,
    this theorem states that the largest power of three that divides A is 3^669. -/
theorem largest_power_of_three_dividing_A : ∃ A : ℕ,
  (∀ (factors : List ℕ), (factors.sum = 2011 ∧ factors.prod ≤ A) → 
    ∃ (k : ℕ), A = 3^669 * k ∧ ¬(∃ m : ℕ, A = 3^(669 + 1) * m)) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_three_dividing_A_l917_91744


namespace NUMINAMATH_CALUDE_paul_juice_bottles_l917_91775

/-- 
Given that Donald drinks 3 more than twice the number of juice bottles Paul drinks in one day,
and Donald drinks 9 bottles of juice per day, prove that Paul drinks 3 bottles of juice per day.
-/
theorem paul_juice_bottles (paul_bottles : ℕ) (donald_bottles : ℕ) : 
  donald_bottles = 2 * paul_bottles + 3 →
  donald_bottles = 9 →
  paul_bottles = 3 := by
  sorry

end NUMINAMATH_CALUDE_paul_juice_bottles_l917_91775


namespace NUMINAMATH_CALUDE_money_distribution_l917_91739

/-- Given that A, B, and C have a total of 250 Rs., and A and C together have 200 Rs.,
    prove that B has 50 Rs. -/
theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 250)  -- Total money of A, B, and C
  (h2 : A + C = 200)      -- Money of A and C together
  : B = 50 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l917_91739


namespace NUMINAMATH_CALUDE_drainage_pipes_count_l917_91730

/-- The number of initial drainage pipes -/
def n : ℕ := 5

/-- The time (in days) it takes n pipes to drain the pool -/
def initial_time : ℕ := 12

/-- The time (in days) it takes (n + 10) pipes to drain the pool -/
def faster_time : ℕ := 4

/-- The number of additional pipes -/
def additional_pipes : ℕ := 10

theorem drainage_pipes_count :
  (n : ℚ) * faster_time = (n + additional_pipes) * initial_time :=
sorry

end NUMINAMATH_CALUDE_drainage_pipes_count_l917_91730


namespace NUMINAMATH_CALUDE_sum_g_15_neg_15_l917_91720

-- Define the function g
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^8 - e * x^6 + f * x^2 + 5

-- Theorem statement
theorem sum_g_15_neg_15 (d e f : ℝ) (h : g d e f 15 = 7) :
  g d e f 15 + g d e f (-15) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_g_15_neg_15_l917_91720


namespace NUMINAMATH_CALUDE_bench_seating_l917_91719

theorem bench_seating (N : ℕ) : (∃ x : ℕ, 7 * N = x ∧ 11 * N = x) ↔ N ≥ 77 :=
sorry

end NUMINAMATH_CALUDE_bench_seating_l917_91719


namespace NUMINAMATH_CALUDE_fraction_equality_l917_91768

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : 1 / x + 1 / y = 2) : 
  (x * y + 3 * x + 3 * y) / (x * y) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l917_91768


namespace NUMINAMATH_CALUDE_no_common_elements_l917_91734

theorem no_common_elements : ¬∃ (n m : ℕ), n^2 - 1 = m^2 + 1 := by sorry

end NUMINAMATH_CALUDE_no_common_elements_l917_91734


namespace NUMINAMATH_CALUDE_tetrahedron_inscribed_circle_centers_intersection_l917_91773

structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

def inscribedCircleCenter (p q r : Point) : Point := sorry

def intersect (a b c d : Point) : Prop := sorry

theorem tetrahedron_inscribed_circle_centers_intersection 
  (ABCD : Tetrahedron) 
  (E : Point) 
  (F : Point) 
  (h1 : E = inscribedCircleCenter ABCD.B ABCD.C ABCD.D) 
  (h2 : F = inscribedCircleCenter ABCD.A ABCD.C ABCD.D) 
  (h3 : intersect ABCD.A E ABCD.B F) :
  ∃ (G H : Point), 
    G = inscribedCircleCenter ABCD.A ABCD.B ABCD.D ∧ 
    H = inscribedCircleCenter ABCD.A ABCD.B ABCD.C ∧ 
    intersect ABCD.C G ABCD.D H :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_inscribed_circle_centers_intersection_l917_91773


namespace NUMINAMATH_CALUDE_distance_between_circle_centers_l917_91781

-- Define the isosceles triangle
structure IsoscelesTriangle where
  vertex_angle : Real
  side_length : Real

-- Define the circles
structure CircumscribedCircle where
  radius : Real

structure InscribedCircle where
  radius : Real

structure SecondCircle where
  radius : Real
  distance_to_vertex : Real

-- Main theorem
theorem distance_between_circle_centers
  (triangle : IsoscelesTriangle)
  (circum_circle : CircumscribedCircle)
  (in_circle : InscribedCircle)
  (second_circle : SecondCircle)
  (h1 : triangle.vertex_angle = 45)
  (h2 : second_circle.distance_to_vertex = 4)
  (h3 : second_circle.radius = circum_circle.radius - 4)
  (h4 : second_circle.radius > 0)
  (h5 : in_circle.radius > 0) :
  ∃ (distance : Real), distance = 4 ∧ 
    distance = circum_circle.radius - in_circle.radius + 4 * Real.sin (45 * π / 180) :=
by sorry


end NUMINAMATH_CALUDE_distance_between_circle_centers_l917_91781


namespace NUMINAMATH_CALUDE_sqrt_decimal_movement_l917_91787

theorem sqrt_decimal_movement (a b : ℝ) (n : ℤ) (h : Real.sqrt a = b) :
  Real.sqrt (a * (10 : ℝ)^(2*n)) = b * (10 : ℝ)^n := by sorry

end NUMINAMATH_CALUDE_sqrt_decimal_movement_l917_91787


namespace NUMINAMATH_CALUDE_quadratic_function_behavior_l917_91777

def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 4

theorem quadratic_function_behavior (b : ℝ) :
  (∀ x₁ x₂, x₁ ≤ x₂ ∧ x₂ ≤ -1 → f b x₁ ≥ f b x₂) ∧
  (∀ x₁ x₂, -1 ≤ x₁ ∧ x₁ ≤ x₂ → f b x₁ ≤ f b x₂) →
  b > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_behavior_l917_91777


namespace NUMINAMATH_CALUDE_prob_same_heads_m_plus_n_l917_91738

def fair_coin_prob : ℚ := 1/2
def biased_coin_prob : ℚ := 3/5

def same_heads_prob : ℚ :=
  (1 - fair_coin_prob) * (1 - biased_coin_prob) +
  fair_coin_prob * biased_coin_prob +
  fair_coin_prob * (1 - biased_coin_prob) * biased_coin_prob * (1 - fair_coin_prob)

theorem prob_same_heads :
  same_heads_prob = 19/50 := by sorry

#eval Nat.gcd 19 50  -- To verify that 19 and 50 are relatively prime

def m : ℕ := 19
def n : ℕ := 50

theorem m_plus_n : m + n = 69 := by sorry

end NUMINAMATH_CALUDE_prob_same_heads_m_plus_n_l917_91738


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l917_91772

theorem arithmetic_calculations :
  (25 - (3 - (-30 - 2)) = -10) ∧
  ((-80) * (1/2 + 2/5 - 1) = 8) ∧
  (81 / (-3)^3 + (-1/5) * (-10) = -1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l917_91772


namespace NUMINAMATH_CALUDE_parabola_properties_l917_91706

/-- A parabola that intersects the x-axis at (-3,0) and (1,0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_neg : a < 0
  h_root1 : a * (-3)^2 + b * (-3) + c = 0
  h_root2 : a * 1^2 + b * 1 + c = 0

/-- Properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  (p.b^2 - 4*p.a*p.c > 0) ∧ (3*p.b + 2*p.c = 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l917_91706


namespace NUMINAMATH_CALUDE_triangle_side_angle_relation_l917_91748

/-- Given a triangle ABC with side lengths a, b, and c opposite to angles A, B, and C respectively,
    the sum of the squares of the side lengths equals twice the sum of the products of pairs of
    side lengths and the cosine of their opposite angles. -/
theorem triangle_side_angle_relation (a b c : ℝ) (A B C : ℝ) :
  (a ≥ 0) → (b ≥ 0) → (c ≥ 0) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  a^2 + b^2 + c^2 = 2 * (b * c * Real.cos A + a * c * Real.cos B + a * b * Real.cos C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_angle_relation_l917_91748


namespace NUMINAMATH_CALUDE_books_left_over_l917_91765

theorem books_left_over (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) : 
  initial_boxes = 1335 →
  books_per_initial_box = 39 →
  books_per_new_box = 40 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 25 := by
  sorry

end NUMINAMATH_CALUDE_books_left_over_l917_91765


namespace NUMINAMATH_CALUDE_coefficient_x4_is_160_l917_91733

/-- The coefficient of x^4 in the expansion of (1+x) * (1+2x)^5 -/
def coefficient_x4 : ℕ :=
  -- Define the coefficient here
  sorry

/-- Theorem stating that the coefficient of x^4 in the expansion of (1+x) * (1+2x)^5 is 160 -/
theorem coefficient_x4_is_160 : coefficient_x4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_is_160_l917_91733


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l917_91752

theorem fraction_product_simplification :
  (3 : ℚ) / 4 * 4 / 5 * 5 / 6 * 6 / 7 = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l917_91752


namespace NUMINAMATH_CALUDE_number_of_divisors_of_30_l917_91762

theorem number_of_divisors_of_30 : Finset.card (Nat.divisors 30) = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_30_l917_91762


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_l917_91756

/-- Given five identical rectangles each with an area of 8 cm², 
    when arranged into a large rectangle, 
    the perimeter of the large rectangle is 32 cm. -/
theorem large_rectangle_perimeter (small_rectangle_area : ℝ) 
  (h1 : small_rectangle_area = 8) 
  (h2 : ∃ (w h : ℝ), w * h = small_rectangle_area ∧ h = 2 * w) : 
  ∃ (W H : ℝ), W * H = 5 * small_rectangle_area ∧ 2 * (W + H) = 32 :=
by sorry

end NUMINAMATH_CALUDE_large_rectangle_perimeter_l917_91756


namespace NUMINAMATH_CALUDE_range_of_m_for_empty_intersection_l917_91746

/-- The set A defined by the quadratic equation mx^2 + x + m = 0 -/
def A (m : ℝ) : Set ℝ := {x : ℝ | m * x^2 + x + m = 0}

/-- Theorem stating the range of m for which A has no real solutions -/
theorem range_of_m_for_empty_intersection :
  (∀ m : ℝ, (A m ∩ Set.univ = ∅) ↔ (m < -1/2 ∨ m > 1/2)) := by sorry

end NUMINAMATH_CALUDE_range_of_m_for_empty_intersection_l917_91746


namespace NUMINAMATH_CALUDE_cosine_equality_condition_l917_91716

theorem cosine_equality_condition (x y : ℝ) : 
  (x = y → Real.cos x = Real.cos y) ∧ 
  ∃ a b : ℝ, Real.cos a = Real.cos b ∧ a ≠ b :=
by sorry

end NUMINAMATH_CALUDE_cosine_equality_condition_l917_91716


namespace NUMINAMATH_CALUDE_wind_speed_calculation_l917_91740

/-- The speed of the wind that satisfies the given conditions -/
def wind_speed : ℝ := 20

/-- The speed of the plane in still air -/
def plane_speed : ℝ := 180

/-- The distance flown with the wind -/
def distance_with_wind : ℝ := 400

/-- The distance flown against the wind -/
def distance_against_wind : ℝ := 320

theorem wind_speed_calculation :
  (distance_with_wind / (plane_speed + wind_speed) = 
   distance_against_wind / (plane_speed - wind_speed)) ∧
  wind_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_wind_speed_calculation_l917_91740


namespace NUMINAMATH_CALUDE_binary_calculation_l917_91798

theorem binary_calculation : 
  (0b101010 + 0b11010) * 0b1110 = 0b11000000000 := by sorry

end NUMINAMATH_CALUDE_binary_calculation_l917_91798


namespace NUMINAMATH_CALUDE_standard_form_conversion_theta_range_phi_range_l917_91709

/-- Converts spherical coordinates to standard form -/
def to_standard_spherical (ρ : ℝ) (θ : ℝ) (φ : ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The standard form of (5, 3π/4, 9π/4) is (5, 7π/4, π/4) -/
theorem standard_form_conversion :
  let (ρ, θ, φ) := to_standard_spherical 5 (3 * Real.pi / 4) (9 * Real.pi / 4)
  ρ = 5 ∧ θ = 7 * Real.pi / 4 ∧ φ = Real.pi / 4 :=
by
  sorry

/-- The range of θ in standard spherical coordinates -/
theorem theta_range (θ : ℝ) : 0 ≤ θ ∧ θ < 2 * Real.pi :=
by
  sorry

/-- The range of φ in standard spherical coordinates -/
theorem phi_range (φ : ℝ) : 0 ≤ φ ∧ φ ≤ Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_standard_form_conversion_theta_range_phi_range_l917_91709


namespace NUMINAMATH_CALUDE_conference_duration_theorem_l917_91786

/-- Calculates the duration of a conference excluding breaks -/
def conference_duration_excluding_breaks (total_hours : ℕ) (total_minutes : ℕ) (break_duration : ℕ) : ℕ :=
  let total_duration := total_hours * 60 + total_minutes
  let total_breaks := total_hours * break_duration
  total_duration - total_breaks

/-- Proves that a conference lasting 14 hours and 20 minutes with 15-minute breaks after each hour has a duration of 650 minutes excluding breaks -/
theorem conference_duration_theorem :
  conference_duration_excluding_breaks 14 20 15 = 650 := by
  sorry

end NUMINAMATH_CALUDE_conference_duration_theorem_l917_91786


namespace NUMINAMATH_CALUDE_range_of_4a_minus_2b_l917_91783

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : 1 ≤ a - b) (h2 : a - b ≤ 2) 
  (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) : 
  5 ≤ 4*a - 2*b ∧ 4*a - 2*b ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_range_of_4a_minus_2b_l917_91783


namespace NUMINAMATH_CALUDE_function_characterization_l917_91785

/-- A function is strictly increasing if for all x < y, f(x) < f(y) -/
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- g is the composition inverse of f if f(g(x)) = x and g(f(x)) = x for all real x -/
def CompositionInverse (f g : ℝ → ℝ) : Prop :=
  (∀ x, f (g x) = x) ∧ (∀ x, g (f x) = x)

theorem function_characterization (f : ℝ → ℝ) 
  (h1 : StrictlyIncreasing f)
  (h2 : ∃ g : ℝ → ℝ, CompositionInverse f g ∧ ∀ x, f x + g x = 2 * x) :
  ∃ c : ℝ, ∀ x, f x = x + c :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l917_91785


namespace NUMINAMATH_CALUDE_alternating_series_sum_l917_91701

def alternating_series (n : ℕ) : ℤ := 
  if n % 2 = 0 then n else -n

def series_sum (n : ℕ) : ℤ := 
  (List.range n).map (λ i => alternating_series (i + 1)) |>.sum

theorem alternating_series_sum : series_sum 11001 = 16501 := by
  sorry

end NUMINAMATH_CALUDE_alternating_series_sum_l917_91701

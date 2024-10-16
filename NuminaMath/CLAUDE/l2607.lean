import Mathlib

namespace NUMINAMATH_CALUDE_composition_problem_l2607_260797

theorem composition_problem (c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x + c)
  (hg : ∀ x, g x = c * x + 3)
  (h_comp : ∀ x, f (g x) = 15 * x + d) :
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_composition_problem_l2607_260797


namespace NUMINAMATH_CALUDE_olaf_car_collection_l2607_260712

/-- The number of toy cars in Olaf's collection after receiving gifts from his family -/
def total_cars (initial : ℕ) (dad : ℕ) (auntie : ℕ) (uncle : ℕ) : ℕ :=
  let mum := dad + 5
  let grandpa := 2 * uncle
  initial + dad + mum + auntie + uncle + grandpa

/-- Theorem stating the total number of cars in Olaf's collection -/
theorem olaf_car_collection : 
  total_cars 150 10 6 5 = 196 := by
  sorry

end NUMINAMATH_CALUDE_olaf_car_collection_l2607_260712


namespace NUMINAMATH_CALUDE_complement_of_A_l2607_260787

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ 1}

-- State the theorem
theorem complement_of_A : Set.compl A = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2607_260787


namespace NUMINAMATH_CALUDE_total_answer_key_combinations_l2607_260792

/-- Represents the number of answer choices for a multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- Represents the number of true-false questions -/
def true_false_questions : ℕ := 4

/-- Represents the number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 2

/-- Calculates the number of valid combinations for true-false questions -/
def valid_true_false_combinations : ℕ := 2^true_false_questions - 2

/-- Calculates the number of combinations for multiple-choice questions -/
def multiple_choice_combinations : ℕ := multiple_choice_options^multiple_choice_questions

/-- Theorem stating the total number of ways to create an answer key -/
theorem total_answer_key_combinations :
  valid_true_false_combinations * multiple_choice_combinations = 224 := by
  sorry

end NUMINAMATH_CALUDE_total_answer_key_combinations_l2607_260792


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l2607_260715

theorem stratified_sampling_male_count 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_male = 48) 
  (h2 : total_female = 36) 
  (h3 : sample_size = 21) :
  (sample_size * total_male) / (total_male + total_female) = 12 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l2607_260715


namespace NUMINAMATH_CALUDE_obtuse_angle_range_l2607_260784

/-- Two vectors form an obtuse angle if their dot product is negative -/
def obtuse_angle (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 < 0

/-- The theorem stating the range of m for which vectors a and b form an obtuse angle -/
theorem obtuse_angle_range :
  ∀ m : ℝ, obtuse_angle (-2, 3) (1, m) ↔ m < 2/3 ∧ m ≠ -3/2 := by
  sorry

end NUMINAMATH_CALUDE_obtuse_angle_range_l2607_260784


namespace NUMINAMATH_CALUDE_bridesmaids_count_l2607_260743

/-- The number of bridesmaids in the wedding --/
def num_bridesmaids : ℕ := 5

/-- Time to sew one dress in hours --/
def time_per_dress : ℕ := 12

/-- Sewing time per week in hours --/
def sewing_time_per_week : ℕ := 4

/-- Total weeks to complete all dresses --/
def total_weeks : ℕ := 15

/-- Theorem stating the number of bridesmaids --/
theorem bridesmaids_count :
  num_bridesmaids = (sewing_time_per_week * total_weeks) / time_per_dress :=
by sorry

end NUMINAMATH_CALUDE_bridesmaids_count_l2607_260743


namespace NUMINAMATH_CALUDE_ball_hit_ground_time_l2607_260703

/-- The time when a ball thrown upward hits the ground -/
theorem ball_hit_ground_time : ∃ t : ℚ, t = 10 / 7 ∧ -4.9 * t^2 + 4 * t + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ball_hit_ground_time_l2607_260703


namespace NUMINAMATH_CALUDE_journey_distance_l2607_260790

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance = 224 ∧ 
    total_time = (distance / 2) / speed1 + (distance / 2) / speed2 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l2607_260790


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2607_260744

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let equation := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let eccentricity := Real.sqrt (1 + b^2 / a^2)
  equation 3 0 → eccentricity = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2607_260744


namespace NUMINAMATH_CALUDE_triangle_side_length_l2607_260705

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  a = 2 →
  b + c = 7 →
  Real.cos B = -(1/4 : ℝ) →
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2607_260705


namespace NUMINAMATH_CALUDE_not_equal_to_eighteen_fifths_other_options_equal_l2607_260702

theorem not_equal_to_eighteen_fifths : (18 + 1) / (5 + 1) ≠ 18 / 5 := by
  sorry

theorem other_options_equal :
  6^2 / 10 = 18 / 5 ∧
  (1 / 5) * (6 * 3) = 18 / 5 ∧
  3.6 = 18 / 5 ∧
  Real.sqrt (324 / 25) = 18 / 5 := by
  sorry

end NUMINAMATH_CALUDE_not_equal_to_eighteen_fifths_other_options_equal_l2607_260702


namespace NUMINAMATH_CALUDE_investment_interest_is_144_l2607_260768

/-- Calculates the total annual interest earned from a two-part investment --/
def total_annual_interest (total_investment : ℚ) (part1 : ℚ) (rate1 : ℚ) (rate2 : ℚ) : ℚ :=
  let part2 := total_investment - part1
  let interest1 := part1 * rate1
  let interest2 := part2 * rate2
  interest1 + interest2

/-- Theorem stating that the total annual interest for the given investment scenario is 144 --/
theorem investment_interest_is_144 :
  total_annual_interest 3500 1550 (3/100) (5/100) = 144 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_is_144_l2607_260768


namespace NUMINAMATH_CALUDE_fourth_pill_time_l2607_260710

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := (totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  ⟨newHours, newMinutes, by sorry, by sorry⟩

/-- The time interval between pills in minutes -/
def pillInterval : Nat := 75

/-- The starting time when the first pill is taken -/
def startTime : Time := ⟨11, 5, by sorry, by sorry⟩

/-- The number of pills taken -/
def pillCount : Nat := 4

theorem fourth_pill_time :
  addMinutes startTime ((pillCount - 1) * pillInterval) = ⟨14, 50, by sorry, by sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_fourth_pill_time_l2607_260710


namespace NUMINAMATH_CALUDE_train_length_l2607_260749

/-- The length of a train given its speed and the time it takes to cross a platform of known length. -/
theorem train_length (train_speed : Real) (platform_length : Real) (crossing_time : Real) :
  train_speed = 72 / 3.6 →
  platform_length = 50.024 →
  crossing_time = 15 →
  train_speed * crossing_time - platform_length = 249.976 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2607_260749


namespace NUMINAMATH_CALUDE_expression_simplification_l2607_260754

theorem expression_simplification :
  (∀ a : ℝ, 2 * (a - 1) - (2 * a - 3) + 3 = 4) ∧
  (∀ x : ℝ, 3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = x^2 - 3 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2607_260754


namespace NUMINAMATH_CALUDE_victor_sugar_usage_l2607_260745

theorem victor_sugar_usage (brown_sugar : ℝ) (difference : ℝ) (white_sugar : ℝ)
  (h1 : brown_sugar = 0.62)
  (h2 : brown_sugar = white_sugar + difference)
  (h3 : difference = 0.38) :
  white_sugar = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_victor_sugar_usage_l2607_260745


namespace NUMINAMATH_CALUDE_magnitude_of_vector_combination_l2607_260739

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 2]

-- State the theorem
theorem magnitude_of_vector_combination :
  ‖(3 • a) - b‖ = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_combination_l2607_260739


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l2607_260796

/-- Given two positive integers with LCM 2310, HCF 30, and one of them being 210, prove the other is 330 -/
theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  B = 210 →
  A = 330 := by sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l2607_260796


namespace NUMINAMATH_CALUDE_marks_trees_l2607_260736

theorem marks_trees (initial_trees planted_trees : ℕ) :
  initial_trees = 13 →
  planted_trees = 12 →
  initial_trees + planted_trees = 25 :=
by sorry

end NUMINAMATH_CALUDE_marks_trees_l2607_260736


namespace NUMINAMATH_CALUDE_complement_of_intersection_in_S_l2607_260709

def S : Set ℝ := {-2, -1, 0, 1, 2}
def T : Set ℝ := {x | x + 1 ≤ 2}

theorem complement_of_intersection_in_S :
  (S \ (S ∩ T)) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_in_S_l2607_260709


namespace NUMINAMATH_CALUDE_garland_arrangement_count_l2607_260794

/-- The number of ways to arrange blue, red, and white bulbs in a garland with no adjacent white bulbs -/
def garland_arrangements (blue red white : ℕ) : ℕ :=
  (Nat.choose (blue + red) blue) * (Nat.choose (blue + red + 1) white)

/-- Theorem stating the number of arrangements for 8 blue, 6 red, and 12 white bulbs -/
theorem garland_arrangement_count : garland_arrangements 8 6 12 = 1366365 := by
  sorry

end NUMINAMATH_CALUDE_garland_arrangement_count_l2607_260794


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l2607_260750

/-- A point on the curve y = x^3 - 3x with a tangent line parallel to the x-axis -/
structure TangentPoint where
  x : ℝ
  y : ℝ
  on_curve : y = x^3 - 3*x
  parallel_tangent : 3*x^2 - 3 = 0

theorem tangent_point_coordinates (P : TangentPoint) : 
  (P.x = 1 ∧ P.y = -2) ∨ (P.x = -1 ∧ P.y = 2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l2607_260750


namespace NUMINAMATH_CALUDE_exam_contestants_l2607_260758

theorem exam_contestants :
  ∀ (x y : ℕ),
  (30 * (x - 1) + 26 = 26 * (y - 1) + 20) →
  (y = x + 9) →
  (30 * x - 4 = 1736) :=
by
  sorry

end NUMINAMATH_CALUDE_exam_contestants_l2607_260758


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2607_260719

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*|x| ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*|x| < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2607_260719


namespace NUMINAMATH_CALUDE_equation_solution_l2607_260707

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 5 → (x + 36 / (x - 5) = -9 ↔ x = -9 ∨ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2607_260707


namespace NUMINAMATH_CALUDE_apple_juice_problem_l2607_260763

theorem apple_juice_problem (initial_amount : ℚ) (maria_fraction : ℚ) (john_fraction : ℚ) :
  initial_amount = 3/4 →
  maria_fraction = 1/2 →
  john_fraction = 1/3 →
  let remaining_after_maria := initial_amount - (maria_fraction * initial_amount)
  john_fraction * remaining_after_maria = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_apple_juice_problem_l2607_260763


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2607_260764

theorem complex_magnitude_problem (z : ℂ) (h : z * (2 + Complex.I) = 2 - Complex.I) : 
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2607_260764


namespace NUMINAMATH_CALUDE_idle_days_is_37_l2607_260759

/-- Represents the worker's payment scenario -/
structure WorkerPayment where
  totalDays : ℕ
  workPayRate : ℕ
  idleForfeitRate : ℕ
  totalReceived : ℕ

/-- Calculates the number of idle days given a WorkerPayment scenario -/
def calculateIdleDays (wp : WorkerPayment) : ℕ :=
  let totalEarning := wp.workPayRate * wp.totalDays
  let totalLoss := wp.totalReceived - totalEarning
  totalLoss / (wp.workPayRate + wp.idleForfeitRate)

/-- Theorem stating that for the given scenario, the number of idle days is 37 -/
theorem idle_days_is_37 (wp : WorkerPayment) 
    (h1 : wp.totalDays = 60)
    (h2 : wp.workPayRate = 30)
    (h3 : wp.idleForfeitRate = 5)
    (h4 : wp.totalReceived = 500) :
    calculateIdleDays wp = 37 := by
  sorry

#eval calculateIdleDays ⟨60, 30, 5, 500⟩

end NUMINAMATH_CALUDE_idle_days_is_37_l2607_260759


namespace NUMINAMATH_CALUDE_inverse_iff_horizontal_line_test_l2607_260742

-- Define a function type
def Function := ℝ → ℝ

-- Define what it means for a function to have an inverse
def HasInverse (f : Function) : Prop :=
  ∃ g : Function, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Define the horizontal line test
def PassesHorizontalLineTest (f : Function) : Prop :=
  ∀ y : ℝ, ∀ x₁ x₂ : ℝ, f x₁ = y ∧ f x₂ = y → x₁ = x₂

-- Theorem statement
theorem inverse_iff_horizontal_line_test (f : Function) :
  HasInverse f ↔ PassesHorizontalLineTest f :=
sorry

end NUMINAMATH_CALUDE_inverse_iff_horizontal_line_test_l2607_260742


namespace NUMINAMATH_CALUDE_discount_approximation_l2607_260723

/-- Calculates the discount given cost price, markup percentage, and profit percentage -/
def calculate_discount (cost_price : ℝ) (markup_percentage : ℝ) (profit_percentage : ℝ) : ℝ :=
  let marked_price := cost_price * (1 + markup_percentage)
  let selling_price := cost_price * (1 + profit_percentage)
  marked_price - selling_price

/-- Theorem stating that the discount is approximately 50 given the problem conditions -/
theorem discount_approximation :
  let cost_price : ℝ := 180
  let markup_percentage : ℝ := 0.4778
  let profit_percentage : ℝ := 0.20
  let discount := calculate_discount cost_price markup_percentage profit_percentage
  ∃ ε > 0, |discount - 50| < ε :=
sorry

end NUMINAMATH_CALUDE_discount_approximation_l2607_260723


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2607_260793

theorem solution_set_equivalence :
  ∀ (x y z : ℝ), x^2 - 9*y^2 = z^2 ↔ ∃ t : ℝ, x = 3*t ∧ y = t ∧ z = 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2607_260793


namespace NUMINAMATH_CALUDE_quadratic_solution_unique_positive_l2607_260725

theorem quadratic_solution_unique_positive (x : ℝ) :
  x > 0 ∧ 3 * x^2 + 8 * x - 35 = 0 ↔ x = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_unique_positive_l2607_260725


namespace NUMINAMATH_CALUDE_valid_sequence_count_is_840_l2607_260799

/-- Represents a coin toss sequence --/
def CoinSequence := List Bool

/-- Counts the number of occurrences of a given subsequence in a coin sequence --/
def countSubsequence (seq : CoinSequence) (subseq : CoinSequence) : Nat :=
  sorry

/-- Checks if a coin sequence satisfies the given conditions --/
def satisfiesConditions (seq : CoinSequence) : Prop :=
  seq.length = 16 ∧
  countSubsequence seq [true, true] = 2 ∧
  countSubsequence seq [false, false] = 6 ∧
  countSubsequence seq [true, false] = 4 ∧
  countSubsequence seq [false, true] = 4

/-- The number of valid coin sequences --/
def validSequenceCount : Nat :=
  sorry

theorem valid_sequence_count_is_840 :
  validSequenceCount = 840 :=
sorry

end NUMINAMATH_CALUDE_valid_sequence_count_is_840_l2607_260799


namespace NUMINAMATH_CALUDE_h_zero_iff_b_eq_neg_seven_fifths_l2607_260756

def h (x : ℝ) := 5*x + 7

theorem h_zero_iff_b_eq_neg_seven_fifths :
  ∀ b : ℝ, h b = 0 ↔ b = -7/5 := by sorry

end NUMINAMATH_CALUDE_h_zero_iff_b_eq_neg_seven_fifths_l2607_260756


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l2607_260731

/-- The area of a square with one side on y = 7 and endpoints on y = x^2 + 4x + 3 is 32 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 7) ∧
  (x₂^2 + 4*x₂ + 3 = 7) ∧
  (x₁ ≠ x₂) ∧
  ((x₂ - x₁)^2 = 32) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l2607_260731


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l2607_260757

/-- Trapezoid ABCD with given side lengths and angle -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  DA : ℝ
  angleD : ℝ
  h_AB : AB = 40
  h_CD : CD = 60
  h_BC : BC = 50
  h_DA : DA = 70
  h_angleD : angleD = π / 3 -- 60° in radians

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.AB + t.BC + t.CD + t.DA

/-- Theorem: The perimeter of the given trapezoid is 220 units -/
theorem trapezoid_perimeter (t : Trapezoid) : perimeter t = 220 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l2607_260757


namespace NUMINAMATH_CALUDE_woman_work_days_l2607_260755

-- Define the work rates
def man_rate : ℚ := 1 / 6
def boy_rate : ℚ := 1 / 12
def combined_rate : ℚ := 1 / 3

-- Define the woman's work rate
def woman_rate : ℚ := combined_rate - man_rate - boy_rate

-- Theorem to prove
theorem woman_work_days : (1 : ℚ) / woman_rate = 12 := by
  sorry


end NUMINAMATH_CALUDE_woman_work_days_l2607_260755


namespace NUMINAMATH_CALUDE_expression_simplification_l2607_260771

theorem expression_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (2 * x) / (x + 1) - (2 * x + 4) / (x^2 - 1) / ((x + 2) / (x^2 - 2 * x + 1)) = 2 / (x + 1) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2607_260771


namespace NUMINAMATH_CALUDE_hidden_faces_sum_l2607_260766

def standard_die := List.range 6 |>.map (· + 1)

def visible_faces : List Nat := [1, 2, 3, 4, 4, 5, 6, 6]

def total_faces : Nat := 3 * 6

theorem hidden_faces_sum :
  (3 * standard_die.sum) - visible_faces.sum = 32 := by
  sorry

end NUMINAMATH_CALUDE_hidden_faces_sum_l2607_260766


namespace NUMINAMATH_CALUDE_circular_garden_radius_increase_l2607_260791

theorem circular_garden_radius_increase (c₁ c₂ r₁ r₂ : ℝ) :
  c₁ = 30 →
  c₂ = 40 →
  c₁ = 2 * π * r₁ →
  c₂ = 2 * π * r₂ →
  r₂ - r₁ = 5 / π := by
sorry

end NUMINAMATH_CALUDE_circular_garden_radius_increase_l2607_260791


namespace NUMINAMATH_CALUDE_largest_expression_l2607_260716

theorem largest_expression (P Q : ℝ) (h1 : P = 1000) (h2 : Q = 0.01) :
  (P / Q > P + Q) ∧ (P / Q > P * Q) ∧ (P / Q > Q / P) ∧ (P / Q > P - Q) :=
by sorry

end NUMINAMATH_CALUDE_largest_expression_l2607_260716


namespace NUMINAMATH_CALUDE_number_puzzle_l2607_260732

theorem number_puzzle (x : ℚ) : 
  (((5 * x - (1/3) * (5 * x)) / 10) + (1/3) * x + (1/2) * x + (1/4) * x) = 68 → x = 48 := by
sorry

end NUMINAMATH_CALUDE_number_puzzle_l2607_260732


namespace NUMINAMATH_CALUDE_junior_score_l2607_260772

theorem junior_score (total_students : ℕ) (junior_percent senior_percent : ℚ) 
  (overall_avg senior_avg junior_score : ℚ) : 
  junior_percent = 1/5 →
  senior_percent = 4/5 →
  junior_percent + senior_percent = 1 →
  overall_avg = 85 →
  senior_avg = 83 →
  (junior_percent * junior_score + senior_percent * senior_avg = overall_avg) →
  junior_score = 93 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l2607_260772


namespace NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l2607_260752

-- Define the speed in kilometers per second
def speed_km_per_second : ℝ := 6

-- Define the number of seconds in an hour
def seconds_per_hour : ℝ := 3600

-- Theorem to prove
theorem space_shuttle_speed_conversion :
  speed_km_per_second * seconds_per_hour = 21600 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l2607_260752


namespace NUMINAMATH_CALUDE_cos_20_cos_25_minus_sin_20_sin_25_l2607_260788

theorem cos_20_cos_25_minus_sin_20_sin_25 :
  Real.cos (20 * Real.pi / 180) * Real.cos (25 * Real.pi / 180) -
  Real.sin (20 * Real.pi / 180) * Real.sin (25 * Real.pi / 180) =
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_20_cos_25_minus_sin_20_sin_25_l2607_260788


namespace NUMINAMATH_CALUDE_shore_distance_l2607_260718

/-- The distance between two shores A and B, given the movement of two boats --/
theorem shore_distance (d : ℝ) : d = 800 :=
  -- Define the meeting points
  let first_meeting : ℝ := 500
  let second_meeting : ℝ := d - 300

  -- Define the distances traveled by each boat at the first meeting
  let boat_m_first : ℝ := first_meeting
  let boat_b_first : ℝ := d - first_meeting

  -- Define the distances traveled by each boat at the second meeting
  let boat_m_second : ℝ := second_meeting
  let boat_b_second : ℝ := 300

  -- The ratio of distances traveled should be equal for both meetings
  have h : boat_m_first / boat_b_first = boat_m_second / boat_b_second := by sorry

  -- The distance d satisfies the equation derived from the equal ratios
  have eq : d * d - 800 * d = 0 := by sorry

  -- The only positive solution to this equation is 800
  sorry


end NUMINAMATH_CALUDE_shore_distance_l2607_260718


namespace NUMINAMATH_CALUDE_rohits_walk_l2607_260738

/-- Given a right triangle with one leg of length 20 and hypotenuse of length 35,
    the length of the other leg is √825. -/
theorem rohits_walk (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 20) (h3 : c = 35) :
  b = Real.sqrt 825 := by
  sorry

end NUMINAMATH_CALUDE_rohits_walk_l2607_260738


namespace NUMINAMATH_CALUDE_probability_of_one_in_15_rows_l2607_260730

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) := sorry

/-- Counts the number of 1s in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Counts the total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ := sorry

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def probabilityOfOne (n : ℕ) : ℚ :=
  (countOnes n : ℚ) / (totalElements n : ℚ)

theorem probability_of_one_in_15_rows :
  probabilityOfOne 15 = 29 / 120 := by sorry

end NUMINAMATH_CALUDE_probability_of_one_in_15_rows_l2607_260730


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l2607_260724

noncomputable def f (x : ℝ) : ℝ := (x - 1) / x * Real.log x

theorem tangent_line_and_extrema :
  let a := (1 : ℝ) / 4
  let b := Real.exp 1
  ∃ (tl : ℝ → ℝ) (max_val min_val : ℝ),
    (∀ x, tl x = 2 * x - 2 + Real.log 2) ∧
    (∀ x ∈ Set.Icc a b, f x ≤ max_val) ∧
    (∃ x ∈ Set.Icc a b, f x = max_val) ∧
    (∀ x ∈ Set.Icc a b, min_val ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min_val) ∧
    max_val = 0 ∧
    min_val = Real.log 4 - 3 ∧
    (HasDerivAt f 2 (1/2) ∧ f (1/2) = -1 + Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l2607_260724


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2607_260753

theorem sum_of_coefficients (x y : ℝ) : (2*x + 3*y)^12 = 244140625 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2607_260753


namespace NUMINAMATH_CALUDE_percentage_increase_in_students_l2607_260717

theorem percentage_increase_in_students (students_this_year students_last_year : ℕ) 
  (h1 : students_this_year = 960)
  (h2 : students_last_year = 800) :
  (students_this_year - students_last_year) / students_last_year * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_in_students_l2607_260717


namespace NUMINAMATH_CALUDE_log_five_eighteen_l2607_260761

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_five_eighteen (a b : ℝ) 
  (h1 : log 10 2 = a) 
  (h2 : log 10 3 = b) : 
  log 5 18 = (a + 2*b) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_log_five_eighteen_l2607_260761


namespace NUMINAMATH_CALUDE_game_probability_l2607_260701

/-- Represents the probability of winning for each player -/
structure PlayerProbabilities where
  alex : ℝ
  mel : ℝ
  chelsea : ℝ
  sam : ℝ

/-- Calculates the probability of a specific outcome in the game -/
def probability_of_outcome (probs : PlayerProbabilities) : ℝ :=
  probs.alex^3 * probs.mel^2 * probs.chelsea^2 * probs.sam

/-- The number of ways to arrange the wins -/
def number_of_arrangements : ℕ := 420

theorem game_probability (probs : PlayerProbabilities) 
  (h1 : probs.alex = 1/3)
  (h2 : probs.mel = 3 * probs.sam)
  (h3 : probs.chelsea = probs.sam)
  (h4 : probs.alex + probs.mel + probs.chelsea + probs.sam = 1) :
  (probability_of_outcome probs) * (number_of_arrangements : ℝ) = 13440/455625 := by
  sorry


end NUMINAMATH_CALUDE_game_probability_l2607_260701


namespace NUMINAMATH_CALUDE_hash_triple_100_l2607_260721

def hash (N : ℝ) : ℝ := 0.5 * N - 2

theorem hash_triple_100 : hash (hash (hash 100)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_hash_triple_100_l2607_260721


namespace NUMINAMATH_CALUDE_power_of_three_mod_ten_l2607_260729

theorem power_of_three_mod_ten (k : ℕ) :
  (3 : ℤ) ^ (4 * k + 3) ≡ 7 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_ten_l2607_260729


namespace NUMINAMATH_CALUDE_simplify_power_expression_l2607_260714

theorem simplify_power_expression (y : ℝ) : (3 * y^4)^2 = 9 * y^8 := by sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l2607_260714


namespace NUMINAMATH_CALUDE_farm_field_area_l2607_260778

/-- Represents the farm field ploughing problem -/
structure FarmField where
  plannedDailyArea : ℕ  -- Planned area to plough per day
  actualDailyArea : ℕ   -- Actual area ploughed per day
  extraDays : ℕ         -- Extra days needed
  remainingArea : ℕ     -- Area left to plough

/-- Calculates the total area of the farm field -/
def totalArea (f : FarmField) : ℕ :=
  f.plannedDailyArea * ((f.actualDailyArea * (f.extraDays + 3) + f.remainingArea) / f.plannedDailyArea)

/-- Theorem stating that the total area of the given farm field is 480 hectares -/
theorem farm_field_area (f : FarmField) 
    (h1 : f.plannedDailyArea = 160)
    (h2 : f.actualDailyArea = 85)
    (h3 : f.extraDays = 2)
    (h4 : f.remainingArea = 40) : 
  totalArea f = 480 := by
  sorry

#eval totalArea { plannedDailyArea := 160, actualDailyArea := 85, extraDays := 2, remainingArea := 40 }

end NUMINAMATH_CALUDE_farm_field_area_l2607_260778


namespace NUMINAMATH_CALUDE_equidistant_planes_count_l2607_260735

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for planes in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Function to check if 4 points are coplanar
def are_coplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Function to check if a plane is equidistant from 4 points
def is_equidistant_plane (plane : Plane3D) (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Theorem statement
theorem equidistant_planes_count 
  (p1 p2 p3 p4 : Point3D) 
  (h : ¬ are_coplanar p1 p2 p3 p4) : 
  ∃! (planes : Finset Plane3D), 
    (planes.card = 7) ∧ 
    (∀ plane ∈ planes, is_equidistant_plane plane p1 p2 p3 p4) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_planes_count_l2607_260735


namespace NUMINAMATH_CALUDE_jelly_bean_count_l2607_260737

/-- The number of jelly beans in the jar. -/
def total_jelly_beans : ℕ := 200

/-- Thomas's share of jelly beans as a fraction. -/
def thomas_share : ℚ := 1/10

/-- The ratio of Barry's share to Emmanuel's share. -/
def barry_emmanuel_ratio : ℚ := 4/5

/-- Emmanuel's share of jelly beans. -/
def emmanuel_share : ℕ := 100

/-- Theorem stating the total number of jelly beans in the jar. -/
theorem jelly_bean_count :
  total_jelly_beans = 200 ∧
  thomas_share = 1/10 ∧
  barry_emmanuel_ratio = 4/5 ∧
  emmanuel_share = 100 ∧
  emmanuel_share = (5/9 : ℚ) * ((1 - thomas_share) * total_jelly_beans) :=
by sorry

end NUMINAMATH_CALUDE_jelly_bean_count_l2607_260737


namespace NUMINAMATH_CALUDE_constant_derivative_implies_linear_l2607_260700

/-- A function whose derivative is zero everywhere has a straight line graph -/
theorem constant_derivative_implies_linear (f : ℝ → ℝ) :
  (∀ x, deriv f x = 0) → ∃ a b : ℝ, ∀ x, f x = a * x + b :=
sorry

end NUMINAMATH_CALUDE_constant_derivative_implies_linear_l2607_260700


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_geq_arithmetic_mean_l2607_260740

theorem sqrt_sum_squares_geq_arithmetic_mean (a b : ℝ) :
  Real.sqrt ((a^2 + b^2) / 2) ≥ (a + b) / 2 ∧
  (Real.sqrt ((a^2 + b^2) / 2) = (a + b) / 2 ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_geq_arithmetic_mean_l2607_260740


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2607_260798

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2607_260798


namespace NUMINAMATH_CALUDE_complex_root_equation_l2607_260746

theorem complex_root_equation (p : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (3 - Complex.I : ℂ)^2 + p * (3 - Complex.I) + 10 = 0 →
  p = -6 := by
sorry

end NUMINAMATH_CALUDE_complex_root_equation_l2607_260746


namespace NUMINAMATH_CALUDE_black_ball_probability_l2607_260781

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  orange : ℕ
  black : ℕ
  white : ℕ

/-- Calculates the probability of picking a ball of a specific color -/
def probability (counts : BallCounts) (color : ℕ) : ℚ :=
  color / (counts.orange + counts.black + counts.white)

/-- The main theorem to be proved -/
theorem black_ball_probability (counts : BallCounts) 
  (h1 : counts.orange = 8)
  (h2 : counts.black = 7)
  (h3 : counts.white = 6) :
  probability counts counts.black = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_black_ball_probability_l2607_260781


namespace NUMINAMATH_CALUDE_percentage_with_no_conditions_is_22_5_l2607_260776

/-- Represents the survey results of teachers' health conditions -/
structure SurveyResults where
  total : ℕ
  highBloodPressure : ℕ
  heartTrouble : ℕ
  diabetes : ℕ
  highBloodPressureAndHeartTrouble : ℕ
  highBloodPressureAndDiabetes : ℕ
  heartTroubleAndDiabetes : ℕ
  allThree : ℕ

/-- Calculates the percentage of teachers with none of the health conditions -/
def percentageWithNoConditions (results : SurveyResults) : ℚ :=
  let withConditions :=
    results.highBloodPressure +
    results.heartTrouble +
    results.diabetes -
    results.highBloodPressureAndHeartTrouble -
    results.highBloodPressureAndDiabetes -
    results.heartTroubleAndDiabetes +
    results.allThree
  let withoutConditions := results.total - withConditions
  (withoutConditions : ℚ) / results.total * 100

/-- The survey results from the problem -/
def surveyData : SurveyResults :=
  { total := 200
  , highBloodPressure := 90
  , heartTrouble := 60
  , diabetes := 30
  , highBloodPressureAndHeartTrouble := 25
  , highBloodPressureAndDiabetes := 15
  , heartTroubleAndDiabetes := 10
  , allThree := 5 }

theorem percentage_with_no_conditions_is_22_5 :
  percentageWithNoConditions surveyData = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_with_no_conditions_is_22_5_l2607_260776


namespace NUMINAMATH_CALUDE_simplify_part1_simplify_part2_l2607_260773

-- Part 1
theorem simplify_part1 (x : ℝ) (h : x ≠ -2) :
  x^2 / (x + 2) + (4*x + 4) / (x + 2) = x + 2 := by sorry

-- Part 2
theorem simplify_part2 (x : ℝ) (h : x ≠ 1) :
  x^2 / (x^2 - 2*x + 1) / ((1 - 2*x) / (x - 1) - x + 1) = -1 / (x - 1) := by sorry

end NUMINAMATH_CALUDE_simplify_part1_simplify_part2_l2607_260773


namespace NUMINAMATH_CALUDE_complex_number_with_sqrt3_imaginary_and_modulus_2_l2607_260726

theorem complex_number_with_sqrt3_imaginary_and_modulus_2 :
  ∀ z : ℂ, (z.im = Real.sqrt 3) → (Complex.abs z = 2) →
  (z = Complex.mk 1 (Real.sqrt 3) ∨ z = Complex.mk (-1) (Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_with_sqrt3_imaginary_and_modulus_2_l2607_260726


namespace NUMINAMATH_CALUDE_no_real_solutions_l2607_260734

theorem no_real_solutions (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 5) (h2 : y + 1 / x = 1 / 6) : False :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2607_260734


namespace NUMINAMATH_CALUDE_locus_is_circle_l2607_260706

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

-- Define the locus of points
def Locus (c : Circle) (B C : PointOnCircle c) : Set (ℝ × ℝ) :=
  { M | ∃ (A : PointOnCircle c),
    let K := ((A.point.1 + B.point.1) / 2, (A.point.2 + B.point.2) / 2)
    M ∈ { P | (P.1 - A.point.1) * (C.point.1 - A.point.1) + (P.2 - A.point.2) * (C.point.2 - A.point.2) = 0 } ∧
    (K.1 - M.1) * (C.point.1 - A.point.1) + (K.2 - M.2) * (C.point.2 - A.point.2) = 0 }

-- Theorem statement
theorem locus_is_circle (c : Circle) (B C : PointOnCircle c) :
  ∃ (c' : Circle), Locus c B C = { P | (P.1 - c'.center.1)^2 + (P.2 - c'.center.2)^2 = c'.radius^2 } ∧
  B.point ∈ Locus c B C ∧ C.point ∈ Locus c B C :=
sorry

end NUMINAMATH_CALUDE_locus_is_circle_l2607_260706


namespace NUMINAMATH_CALUDE_smallest_nonzero_real_l2607_260762

theorem smallest_nonzero_real : ∃ (p q : ℕ+) (x : ℝ),
  x = -Real.sqrt p / q ∧
  x ≠ 0 ∧
  (∀ y : ℝ, y ≠ 0 → y⁻¹ = y - Real.sqrt (y^2) → |x| ≤ |y|) ∧
  (∀ (a : ℕ+), a^2 ∣ p → a = 1) ∧
  x⁻¹ = x - Real.sqrt (x^2) ∧
  p + q = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_nonzero_real_l2607_260762


namespace NUMINAMATH_CALUDE_initial_fund_is_740_l2607_260774

/-- Represents the company fund problem --/
structure CompanyFund where
  intended_bonus : ℕ
  actual_bonus : ℕ
  remaining_amount : ℕ
  fixed_expense : ℕ
  shortage : ℕ

/-- Calculates the initial fund amount before bonuses and expenses --/
def initial_fund_amount (cf : CompanyFund) : ℕ :=
  sorry

/-- Theorem stating the initial fund amount is 740 given the problem conditions --/
theorem initial_fund_is_740 (cf : CompanyFund) 
  (h1 : cf.intended_bonus = 60)
  (h2 : cf.actual_bonus = 50)
  (h3 : cf.remaining_amount = 110)
  (h4 : cf.fixed_expense = 30)
  (h5 : cf.shortage = 10) :
  initial_fund_amount cf = 740 :=
sorry

end NUMINAMATH_CALUDE_initial_fund_is_740_l2607_260774


namespace NUMINAMATH_CALUDE_quarter_circles_sum_exceeds_circumference_l2607_260782

/-- Theorem: As the number of divisions approaches infinity, the sum of the lengths of quarter-circles
    constructed on equal parts of a circle's circumference exceeds the original circumference. -/
theorem quarter_circles_sum_exceeds_circumference (r : ℝ) (hr : r > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → π * π * r > 2 * π * r := by
  sorry

#check quarter_circles_sum_exceeds_circumference

end NUMINAMATH_CALUDE_quarter_circles_sum_exceeds_circumference_l2607_260782


namespace NUMINAMATH_CALUDE_compound_composition_l2607_260708

/-- The atomic weight of aluminum in g/mol -/
def aluminum_weight : ℝ := 26.98

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The molecular weight of the compound in g/mol -/
def compound_weight : ℝ := 132

/-- The number of chlorine atoms in the compound -/
def chlorine_atoms : ℕ := 3

theorem compound_composition :
  ∃ (n : ℕ), n = chlorine_atoms ∧
  compound_weight = aluminum_weight + n * chlorine_weight :=
sorry

end NUMINAMATH_CALUDE_compound_composition_l2607_260708


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2607_260770

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {-2, -1, 0}

theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2607_260770


namespace NUMINAMATH_CALUDE_milk_water_ratio_after_mixing_l2607_260713

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- Represents the result of mixing two mixtures -/
def mix (m1 m2 : Mixture) : Mixture :=
  { milk := m1.milk + m2.milk,
    water := m1.water + m2.water }

/-- The ratio of milk to water in a mixture -/
def milkWaterRatio (m : Mixture) : ℚ := m.milk / m.water

theorem milk_water_ratio_after_mixing :
  let m1 : Mixture := { milk := 7, water := 2 }
  let m2 : Mixture := { milk := 8, water := 1 }
  milkWaterRatio (mix m1 m2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_ratio_after_mixing_l2607_260713


namespace NUMINAMATH_CALUDE_g_neg_two_l2607_260760

def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem g_neg_two : g (-2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_two_l2607_260760


namespace NUMINAMATH_CALUDE_unicorn_tower_rope_length_l2607_260779

/-- Represents the setup of the unicorn and tower problem -/
structure UnicornTowerSetup where
  towerRadius : ℝ
  ropeLength : ℝ
  ropeAngle : ℝ
  ropeDistanceFromTower : ℝ
  unicornHeight : ℝ

/-- Calculates the length of rope touching the tower given the problem setup -/
def ropeTouchingTowerLength (setup : UnicornTowerSetup) : ℝ :=
  sorry

/-- Theorem stating the length of rope touching the tower in the given setup -/
theorem unicorn_tower_rope_length :
  let setup : UnicornTowerSetup := {
    towerRadius := 5,
    ropeLength := 30,
    ropeAngle := 30 * Real.pi / 180,  -- Convert to radians
    ropeDistanceFromTower := 5,
    unicornHeight := 5
  }
  ∃ (ε : ℝ), abs (ropeTouchingTowerLength setup - 19.06) < ε ∧ ε > 0 :=
sorry

end NUMINAMATH_CALUDE_unicorn_tower_rope_length_l2607_260779


namespace NUMINAMATH_CALUDE_train_length_calculation_l2607_260785

-- Define the given values
def train_speed : Real := 100  -- km/h
def motorbike_speed : Real := 64  -- km/h
def overtake_time : Real := 18  -- seconds

-- Define the theorem
theorem train_length_calculation :
  let train_speed_ms : Real := train_speed * 1000 / 3600
  let motorbike_speed_ms : Real := motorbike_speed * 1000 / 3600
  let relative_speed : Real := train_speed_ms - motorbike_speed_ms
  let train_length : Real := relative_speed * overtake_time
  train_length = 180 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2607_260785


namespace NUMINAMATH_CALUDE_probability_theorem_l2607_260775

def num_forks : ℕ := 8
def num_spoons : ℕ := 5
def num_knives : ℕ := 7
def total_pieces : ℕ := num_forks + num_spoons + num_knives
def num_selected : ℕ := 4

def probability_two_forks_one_spoon_one_knife : ℚ :=
  (Nat.choose num_forks 2 * Nat.choose num_spoons 1 * Nat.choose num_knives 1 : ℚ) /
  Nat.choose total_pieces num_selected

theorem probability_theorem :
  probability_two_forks_one_spoon_one_knife = 196 / 969 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l2607_260775


namespace NUMINAMATH_CALUDE_rachel_saturday_water_consumption_l2607_260765

def glassesToOunces (glasses : ℕ) : ℕ := glasses * 10

def waterConsumed (sun mon tue wed thu fri : ℕ) : ℕ :=
  glassesToOunces (sun + mon + tue + wed + thu + fri)

theorem rachel_saturday_water_consumption
  (h1 : waterConsumed 2 4 3 3 3 3 + glassesToOunces x = 220) :
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_rachel_saturday_water_consumption_l2607_260765


namespace NUMINAMATH_CALUDE_coat_cost_after_discount_l2607_260741

/-- The cost of Mr. Zubir's purchases --/
structure Purchase where
  pants : ℝ
  shirt : ℝ
  coat : ℝ

/-- The conditions of Mr. Zubir's purchase --/
def purchase_conditions (p : Purchase) : Prop :=
  p.pants + p.shirt = 100 ∧
  p.pants + p.coat = 244 ∧
  p.coat = 5 * p.shirt

/-- The discount rate applied to the purchase --/
def discount_rate : ℝ := 0.1

/-- Theorem stating the cost of the coat after discount --/
theorem coat_cost_after_discount (p : Purchase) 
  (h : purchase_conditions p) : 
  p.coat * (1 - discount_rate) = 162 := by
  sorry

end NUMINAMATH_CALUDE_coat_cost_after_discount_l2607_260741


namespace NUMINAMATH_CALUDE_subtraction_inequality_l2607_260704

theorem subtraction_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a < b) : a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_inequality_l2607_260704


namespace NUMINAMATH_CALUDE_pauls_crayons_l2607_260722

theorem pauls_crayons (crayons_given : ℕ) (crayons_lost : ℕ) (crayons_left : ℕ)
  (h1 : crayons_given = 563)
  (h2 : crayons_lost = 558)
  (h3 : crayons_left = 332) :
  crayons_given + crayons_lost + crayons_left = 1453 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_l2607_260722


namespace NUMINAMATH_CALUDE_f_max_range_l2607_260795

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then a * Real.log x - x^2 - 2
  else if x < 0 then x + 1/x + a
  else 0  -- This value doesn't matter as x ≠ 0 in the problem

theorem f_max_range (a : ℝ) :
  (∀ x : ℝ, f a x ≤ f a (-1)) →
  0 ≤ a ∧ a ≤ 2 * Real.exp 3 :=
sorry

end NUMINAMATH_CALUDE_f_max_range_l2607_260795


namespace NUMINAMATH_CALUDE_expression_bounds_l2607_260727

theorem expression_bounds (x y : ℝ) (h : abs x + abs y = 13) :
  0 ≤ x^2 + 7*x - 3*y + y^2 ∧ x^2 + 7*x - 3*y + y^2 ≤ 260 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l2607_260727


namespace NUMINAMATH_CALUDE_solution_set_f_inequality_range_of_m_for_nonempty_solution_l2607_260748

-- Define the functions f and g
def f (x : ℝ) := |x - 2|
def g (m : ℝ) (x : ℝ) := -|x + 7| + 3 * m

-- Theorem for the first part of the problem
theorem solution_set_f_inequality (x : ℝ) :
  f x + x^2 - 4 > 0 ↔ x > 2 ∨ x < -1 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_m_for_nonempty_solution (m : ℝ) :
  (∃ x : ℝ, f x < g m x) ↔ m > 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_inequality_range_of_m_for_nonempty_solution_l2607_260748


namespace NUMINAMATH_CALUDE_sum_of_digits_l2607_260777

/-- Given a three-digit number of the form 3a7 and another three-digit number 7c1,
    where a and c are single digits, prove that if 3a7 + 414 = 7c1 and 7c1 is
    divisible by 11, then a + c = 14. -/
theorem sum_of_digits (a c : ℕ) : 
  (a < 10) →
  (c < 10) →
  (300 + 10 * a + 7 + 414 = 700 + 10 * c + 1) →
  (700 + 10 * c + 1) % 11 = 0 →
  a + c = 14 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l2607_260777


namespace NUMINAMATH_CALUDE_decimal_expansion_of_prime_reciprocal_l2607_260747

/-- The type of natural numbers greater than 1 -/
def PositiveNatGT1 := { n : ℕ // n > 1 }

/-- The period of a rational number's decimal expansion -/
def decimalPeriod (q : ℚ) : ℕ := sorry

/-- The nth digit in the decimal expansion of a rational number -/
def nthDecimalDigit (q : ℚ) (n : ℕ) : Fin 10 := sorry

theorem decimal_expansion_of_prime_reciprocal (p : PositiveNatGT1) 
  (h_prime : Nat.Prime p.val) 
  (h_period : decimalPeriod (1 / p.val) = 200) : 
  nthDecimalDigit (1 / p.val) 101 = 9 := by sorry

end NUMINAMATH_CALUDE_decimal_expansion_of_prime_reciprocal_l2607_260747


namespace NUMINAMATH_CALUDE_correct_green_pens_l2607_260751

-- Define the ratio of blue pens to green pens
def blue_to_green_ratio : ℚ := 5 / 3

-- Define the number of blue pens
def blue_pens : ℕ := 20

-- Define the number of green pens
def green_pens : ℕ := 12

-- Theorem to prove
theorem correct_green_pens : 
  (blue_pens : ℚ) / green_pens = blue_to_green_ratio := by
  sorry

end NUMINAMATH_CALUDE_correct_green_pens_l2607_260751


namespace NUMINAMATH_CALUDE_middle_card_first_round_l2607_260733

/-- Represents a card with a positive integer value -/
structure Card where
  value : ℕ+
  
/-- Represents a player in the game -/
structure Player where
  totalCounters : ℕ
  lastRoundCard : Card

/-- Represents the game state -/
structure GameState where
  cards : Fin 3 → Card
  players : Fin 3 → Player
  rounds : ℕ

/-- Conditions of the game -/
def gameConditions (g : GameState) : Prop :=
  g.rounds ≥ 2 ∧
  (g.cards 0).value < (g.cards 1).value ∧ (g.cards 1).value < (g.cards 2).value ∧
  (g.players 0).totalCounters + (g.players 1).totalCounters + (g.players 2).totalCounters = 39 ∧
  (∃ i j k : Fin 3, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (g.players i).totalCounters = 20 ∧
    (g.players j).totalCounters = 10 ∧
    (g.players k).totalCounters = 9) ∧
  (∃ i : Fin 3, (g.players i).totalCounters = 10 ∧
    (g.players i).lastRoundCard = g.cards 2)

/-- The theorem to be proved -/
theorem middle_card_first_round (g : GameState) :
  gameConditions g →
  ∃ i : Fin 3, (g.players i).totalCounters = 9 ∧
    (∃ firstRoundCard : Card, firstRoundCard = g.cards 1) :=
sorry

end NUMINAMATH_CALUDE_middle_card_first_round_l2607_260733


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l2607_260789

theorem parallel_vectors_y_value (a b : ℝ × ℝ) :
  a = (6, 2) →
  b.2 = 3 →
  (∃ k : ℝ, k ≠ 0 ∧ a = k • b) →
  b.1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l2607_260789


namespace NUMINAMATH_CALUDE_leos_garden_tulips_leos_garden_tulips_proof_l2607_260769

/-- Calculates the number of tulips in Leo's garden after additions -/
theorem leos_garden_tulips (initial_carnations : ℕ) (added_carnations : ℕ) 
  (tulip_ratio : ℕ) (carnation_ratio : ℕ) : ℕ :=
  let total_carnations := initial_carnations + added_carnations
  let tulips := (total_carnations / carnation_ratio) * tulip_ratio
  tulips

/-- Proves that Leo will have 36 tulips after the additions -/
theorem leos_garden_tulips_proof :
  leos_garden_tulips 49 35 3 7 = 36 := by
  sorry

end NUMINAMATH_CALUDE_leos_garden_tulips_leos_garden_tulips_proof_l2607_260769


namespace NUMINAMATH_CALUDE_prob_four_to_five_l2607_260786

-- Define the possible on-times
inductive OnTime
  | Seven
  | SevenThirty
  | Eight
  | EightThirty
  | Nine

-- Define the probability space
def Ω : Type := OnTime × ℝ

-- Define the probability measure
axiom P : Set Ω → ℝ

-- Define the uniform distribution of on-times
axiom uniform_on_time : ∀ t : OnTime, P {ω : Ω | ω.1 = t} = 1/5

-- Define the uniform distribution of off-times
axiom uniform_off_time : ∀ a b : ℝ, 
  23 ≤ a ∧ a < b ∧ b ≤ 25 → P {ω : Ω | a ≤ ω.2 ∧ ω.2 ≤ b} = (b - a) / 2

-- Define the event where 4 < t < 5
def E : Set Ω :=
  {ω : Ω | 
    (ω.1 = OnTime.Seven ∧ 23 < ω.2 ∧ ω.2 < 24) ∨
    (ω.1 = OnTime.SevenThirty ∧ 23.5 < ω.2 ∧ ω.2 < 24.5) ∨
    (ω.1 = OnTime.Eight ∧ 24 < ω.2 ∧ ω.2 < 25) ∨
    (ω.1 = OnTime.EightThirty ∧ 24.5 < ω.2 ∧ ω.2 ≤ 25)}

-- Theorem to prove
theorem prob_four_to_five : P E = 7/20 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_to_five_l2607_260786


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2607_260767

/-- Given a hyperbola and conditions on its asymptote and focus, prove its equation -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) ∧  -- Hyperbola equation
  (∃ (x y : ℝ), y = Real.sqrt 3 * x) ∧   -- Asymptote condition
  (∃ (x : ℝ), x^2/a^2 - 0^2/b^2 = 1 ∧ x = -12) -- Focus on directrix of y^2 = 48x
  →
  a^2 = 36 ∧ b^2 = 108 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2607_260767


namespace NUMINAMATH_CALUDE_max_intersections_quadrilateral_hexagon_l2607_260728

/-- The number of sides in a quadrilateral -/
def quadrilateral_sides : ℕ := 4

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The maximum number of intersection points between the boundaries of a quadrilateral and a hexagon -/
def max_intersection_points : ℕ := quadrilateral_sides * hexagon_sides

/-- Theorem stating that the maximum number of intersection points between 
    the boundaries of a quadrilateral and a hexagon is 24 -/
theorem max_intersections_quadrilateral_hexagon : 
  max_intersection_points = 24 := by sorry

end NUMINAMATH_CALUDE_max_intersections_quadrilateral_hexagon_l2607_260728


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2607_260783

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ x : ℝ, x^3 - 21*x^2 + 130*x - 210 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 21*s^2 + 130*s - 210) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 275 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2607_260783


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2607_260780

def U : Set ℕ := {0, 1, 3, 5, 6, 8}
def A : Set ℕ := {1, 5, 8}
def B : Set ℕ := {2}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2607_260780


namespace NUMINAMATH_CALUDE_sales_tax_difference_l2607_260711

theorem sales_tax_difference (price : ℝ) (tax_rate1 : ℝ) (tax_rate2 : ℝ) : 
  price = 50 → tax_rate1 = 0.075 → tax_rate2 = 0.065 → 
  price * tax_rate1 - price * tax_rate2 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l2607_260711


namespace NUMINAMATH_CALUDE_find_b_l2607_260720

theorem find_b (a b c : ℚ) 
  (sum_eq : a + b + c = 150)
  (equal_after_changes : a + 10 = b - 10 ∧ b - 10 = 3 * c) : 
  b = 520 / 7 := by
sorry

end NUMINAMATH_CALUDE_find_b_l2607_260720

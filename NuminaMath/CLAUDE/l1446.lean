import Mathlib

namespace NUMINAMATH_CALUDE_octahedron_colorings_l1446_144654

/-- The number of faces in a regular octahedron -/
def num_faces : ℕ := 8

/-- The number of rotational symmetries of a regular octahedron -/
def num_rotational_symmetries : ℕ := 24

/-- The number of distinguishable colorings of a regular octahedron -/
def num_distinguishable_colorings : ℕ := Nat.factorial num_faces / num_rotational_symmetries

theorem octahedron_colorings :
  num_distinguishable_colorings = 1680 := by sorry

end NUMINAMATH_CALUDE_octahedron_colorings_l1446_144654


namespace NUMINAMATH_CALUDE_right_triangle_angles_l1446_144616

/-- A right-angled triangle with a specific property -/
structure RightTriangle where
  /-- The measure of the right angle in degrees -/
  right_angle : ℝ
  /-- The measure of the angle between the angle bisector of the right angle and the median to the hypotenuse, in degrees -/
  bisector_median_angle : ℝ
  /-- The right angle is 90 degrees -/
  right_angle_is_90 : right_angle = 90
  /-- The angle between the bisector and median is 16 degrees -/
  bisector_median_angle_is_16 : bisector_median_angle = 16

/-- The angles of the triangle given the specific conditions -/
def triangle_angles (t : RightTriangle) : (ℝ × ℝ × ℝ) :=
  (61, 29, 90)

/-- Theorem stating that the angles of the triangle are 61°, 29°, and 90° given the conditions -/
theorem right_triangle_angles (t : RightTriangle) :
  triangle_angles t = (61, 29, 90) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angles_l1446_144616


namespace NUMINAMATH_CALUDE_greatest_number_of_factors_l1446_144687

/-- The greatest number of positive factors for b^n given the conditions -/
def max_factors : ℕ := 561

/-- b is a positive integer less than or equal to 15 -/
def b : ℕ := 12

/-- n is a perfect square less than or equal to 16 -/
def n : ℕ := 16

/-- Theorem stating that max_factors is the greatest number of positive factors of b^n -/
theorem greatest_number_of_factors :
  ∀ (b' n' : ℕ), 
    b' > 0 → b' ≤ 15 → 
    n' > 0 → ∃ (k : ℕ), n' = k^2 → n' ≤ 16 →
    (Nat.factors (b'^n')).length ≤ max_factors :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_of_factors_l1446_144687


namespace NUMINAMATH_CALUDE_pedro_gifts_l1446_144621

theorem pedro_gifts (total : ℕ) (emilio : ℕ) (jorge : ℕ) 
  (h1 : total = 21)
  (h2 : emilio = 11)
  (h3 : jorge = 6) :
  total - (emilio + jorge) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pedro_gifts_l1446_144621


namespace NUMINAMATH_CALUDE_initial_minutes_plan_a_l1446_144656

/-- Represents the cost in dollars for a call under Plan A -/
def costPlanA (initialMinutes : ℕ) (totalMinutes : ℕ) : ℚ :=
  0.60 + 0.06 * (totalMinutes - initialMinutes)

/-- Represents the cost in dollars for a call under Plan B -/
def costPlanB (minutes : ℕ) : ℚ :=
  0.08 * minutes

theorem initial_minutes_plan_a : ∃ (x : ℕ), 
  (∀ (m : ℕ), m ≥ x → costPlanA x m = costPlanB m) ∧
  (costPlanA x 18 = costPlanB 18) ∧
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_minutes_plan_a_l1446_144656


namespace NUMINAMATH_CALUDE_trig_expression_equals_sqrt_two_l1446_144671

/-- Proves that the given trigonometric expression equals √2 --/
theorem trig_expression_equals_sqrt_two :
  (Real.cos (10 * π / 180) - Real.sqrt 3 * Real.cos (-100 * π / 180)) /
  Real.sqrt (1 - Real.sin (10 * π / 180)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_sqrt_two_l1446_144671


namespace NUMINAMATH_CALUDE_answer_key_combinations_l1446_144615

/-- Represents the number of answer choices for a multiple-choice question -/
def multipleChoiceOptions : ℕ := 4

/-- Represents the number of true-false questions -/
def trueFalseQuestions : ℕ := 3

/-- Represents the number of multiple-choice questions -/
def multipleChoiceQuestions : ℕ := 2

/-- Calculates the number of valid true-false combinations -/
def validTrueFalseCombinations : ℕ := 2^trueFalseQuestions - 2

/-- Calculates the number of multiple-choice combinations -/
def multipleChoiceCombinations : ℕ := multipleChoiceOptions^multipleChoiceQuestions

/-- Theorem stating the total number of ways to create the answer key -/
theorem answer_key_combinations :
  validTrueFalseCombinations * multipleChoiceCombinations = 96 := by
  sorry

end NUMINAMATH_CALUDE_answer_key_combinations_l1446_144615


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1446_144636

theorem unique_three_digit_number : ∃! n : ℕ,
  (100 ≤ n ∧ n ≤ 999) ∧
  (n % 11 = 0) ∧
  (n / 11 = (n / 100)^2 + ((n / 10) % 10)^2 + (n % 10)^2) ∧
  (n = 550) := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1446_144636


namespace NUMINAMATH_CALUDE_candidate_admission_criterion_l1446_144605

/-- Represents the constructibility of an angle division -/
inductive AngleDivision
  | Constructible
  | NotConstructible

/-- Represents a candidate's response to the angle division questions -/
structure CandidateResponse :=
  (div19 : AngleDivision)
  (div17 : AngleDivision)
  (div18 : AngleDivision)

/-- Determines if an angle of n degrees can be divided into n equal parts -/
def canDivideAngle (n : ℕ) : AngleDivision :=
  if n = 19 ∨ n = 17 then AngleDivision.Constructible
  else AngleDivision.NotConstructible

/-- Determines if a candidate's response is correct -/
def isCorrectResponse (response : CandidateResponse) : Prop :=
  response.div19 = canDivideAngle 19 ∧
  response.div17 = canDivideAngle 17 ∧
  response.div18 = canDivideAngle 18

/-- Determines if a candidate should be admitted based on their response -/
def shouldAdmit (response : CandidateResponse) : Prop :=
  isCorrectResponse response

theorem candidate_admission_criterion (response : CandidateResponse) :
  response.div19 = AngleDivision.Constructible ∧
  response.div17 = AngleDivision.Constructible ∧
  response.div18 = AngleDivision.NotConstructible →
  shouldAdmit response :=
by sorry

end NUMINAMATH_CALUDE_candidate_admission_criterion_l1446_144605


namespace NUMINAMATH_CALUDE_painting_selection_ways_l1446_144661

theorem painting_selection_ways (oil_paintings : ℕ) (chinese_paintings : ℕ) (watercolor_paintings : ℕ)
  (h1 : oil_paintings = 3)
  (h2 : chinese_paintings = 4)
  (h3 : watercolor_paintings = 5) :
  oil_paintings + chinese_paintings + watercolor_paintings = 12 := by
  sorry

end NUMINAMATH_CALUDE_painting_selection_ways_l1446_144661


namespace NUMINAMATH_CALUDE_power_of_two_greater_than_linear_l1446_144683

theorem power_of_two_greater_than_linear (n : ℕ) (h : n ≥ 3) : 2^n > 2*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_greater_than_linear_l1446_144683


namespace NUMINAMATH_CALUDE_wolves_hunt_in_five_days_l1446_144648

/-- Calculates the number of days before wolves need to hunt again -/
def days_before_next_hunt (hunting_wolves : ℕ) (additional_wolves : ℕ) 
  (meat_per_wolf_per_day : ℕ) (meat_per_deer : ℕ) : ℕ :=
  let total_wolves := hunting_wolves + additional_wolves
  let daily_meat_requirement := total_wolves * meat_per_wolf_per_day
  let total_meat_from_hunt := hunting_wolves * meat_per_deer
  total_meat_from_hunt / daily_meat_requirement

theorem wolves_hunt_in_five_days : 
  days_before_next_hunt 4 16 8 200 = 5 := by sorry

end NUMINAMATH_CALUDE_wolves_hunt_in_five_days_l1446_144648


namespace NUMINAMATH_CALUDE_train_length_l1446_144631

/-- Calculates the length of a train given its speed and time to cross a post -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 40 → time = 17.1 → speed * time * (5 / 18) = 190 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1446_144631


namespace NUMINAMATH_CALUDE_min_distance_point_l1446_144653

noncomputable def f (a x : ℝ) : ℝ := (x - a)^2 + (2 * Real.log x - 2 * a)^2

theorem min_distance_point (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ ≤ 4/5) → a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_point_l1446_144653


namespace NUMINAMATH_CALUDE_two_solutions_iff_a_gt_one_third_l1446_144655

/-- The equation |x-3| = ax - 1 has two solutions if and only if a > 1/3 -/
theorem two_solutions_iff_a_gt_one_third (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (|x₁ - 3| = a * x₁ - 1) ∧ (|x₂ - 3| = a * x₂ - 1)) ↔ a > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_iff_a_gt_one_third_l1446_144655


namespace NUMINAMATH_CALUDE_rotate_d_180_degrees_l1446_144684

/-- Rotation of a point by 180° about the origin -/
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem rotate_d_180_degrees :
  let d : ℝ × ℝ := (2, -3)
  rotate180 d = (-2, 3) := by
  sorry

end NUMINAMATH_CALUDE_rotate_d_180_degrees_l1446_144684


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l1446_144608

theorem ceiling_floor_sum (x : ℝ) : 
  ⌈x⌉ - ⌊x⌋ = 0 → ⌈x⌉ + ⌊x⌋ = 2*x := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l1446_144608


namespace NUMINAMATH_CALUDE_molecular_weight_7_moles_KBrO3_l1446_144691

/-- The atomic weight of potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- The atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of oxygen atoms in KBrO3 -/
def num_oxygen_atoms : ℕ := 3

/-- The molecular weight of one mole of KBrO3 in g/mol -/
def molecular_weight_KBrO3 : ℝ :=
  atomic_weight_K + atomic_weight_Br + (atomic_weight_O * num_oxygen_atoms)

/-- The number of moles of KBrO3 -/
def num_moles : ℕ := 7

/-- Theorem: The molecular weight of 7 moles of KBrO3 is 1169.00 grams -/
theorem molecular_weight_7_moles_KBrO3 :
  molecular_weight_KBrO3 * num_moles = 1169.00 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_7_moles_KBrO3_l1446_144691


namespace NUMINAMATH_CALUDE_consecutive_digits_count_l1446_144650

theorem consecutive_digits_count : ∃ (m n : ℕ), 
  (10^(m-1) < 2^2020 ∧ 2^2020 < 10^m) ∧
  (10^(n-1) < 5^2020 ∧ 5^2020 < 10^n) ∧
  m + n = 2021 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_digits_count_l1446_144650


namespace NUMINAMATH_CALUDE_total_trip_cost_l1446_144638

def rental_cost : ℝ := 150
def gas_price : ℝ := 3.50
def gas_purchased : ℝ := 8
def mileage_cost : ℝ := 0.50
def distance_driven : ℝ := 320

theorem total_trip_cost : 
  rental_cost + gas_price * gas_purchased + mileage_cost * distance_driven = 338 := by
  sorry

end NUMINAMATH_CALUDE_total_trip_cost_l1446_144638


namespace NUMINAMATH_CALUDE_solution_y_b_percentage_l1446_144603

-- Define the solutions and their compositions
def solution_x_a : ℝ := 0.3
def solution_x_b : ℝ := 0.7
def solution_y_a : ℝ := 0.4

-- Define the mixture composition
def mixture_x : ℝ := 0.8
def mixture_y : ℝ := 0.2
def mixture_a : ℝ := 0.32

-- Theorem to prove
theorem solution_y_b_percentage : 
  solution_x_a + solution_x_b = 1 →
  mixture_x + mixture_y = 1 →
  mixture_x * solution_x_a + mixture_y * solution_y_a = mixture_a →
  1 - solution_y_a = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_solution_y_b_percentage_l1446_144603


namespace NUMINAMATH_CALUDE_allysons_age_l1446_144647

theorem allysons_age (hirams_age allyson_age : ℕ) : 
  hirams_age = 40 →
  hirams_age + 12 = 2 * allyson_age - 4 →
  allyson_age = 28 := by
sorry

end NUMINAMATH_CALUDE_allysons_age_l1446_144647


namespace NUMINAMATH_CALUDE_trisha_works_52_weeks_l1446_144646

/-- Calculates the number of weeks worked in a year based on given parameters -/
def weeks_worked (hourly_rate : ℚ) (hours_per_week : ℚ) (withholding_rate : ℚ) (annual_take_home : ℚ) : ℚ :=
  annual_take_home / ((hourly_rate * hours_per_week) * (1 - withholding_rate))

/-- Proves that given the specified parameters, Trisha works 52 weeks in a year -/
theorem trisha_works_52_weeks :
  weeks_worked 15 40 (1/5) 24960 = 52 := by
  sorry

end NUMINAMATH_CALUDE_trisha_works_52_weeks_l1446_144646


namespace NUMINAMATH_CALUDE_total_bars_is_504_l1446_144617

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 18

/-- The number of chocolate bars in each small box -/
def bars_per_small_box : ℕ := 28

/-- The total number of chocolate bars in the large box -/
def total_chocolate_bars : ℕ := num_small_boxes * bars_per_small_box

/-- Theorem: The total number of chocolate bars in the large box is 504 -/
theorem total_bars_is_504 : total_chocolate_bars = 504 := by
  sorry

end NUMINAMATH_CALUDE_total_bars_is_504_l1446_144617


namespace NUMINAMATH_CALUDE_cookies_calculation_l1446_144639

/-- The number of people Brenda's mother made cookies for -/
def num_people : ℕ := 14

/-- The number of cookies each person had -/
def cookies_per_person : ℕ := 30

/-- The total number of cookies prepared -/
def total_cookies : ℕ := num_people * cookies_per_person

theorem cookies_calculation : total_cookies = 420 := by
  sorry

end NUMINAMATH_CALUDE_cookies_calculation_l1446_144639


namespace NUMINAMATH_CALUDE_kate_age_l1446_144625

theorem kate_age (total_age : ℕ) (maggie_age : ℕ) (sue_age : ℕ) 
  (h1 : total_age = 48)
  (h2 : maggie_age = 17)
  (h3 : sue_age = 12) :
  total_age - maggie_age - sue_age = 19 := by
  sorry

end NUMINAMATH_CALUDE_kate_age_l1446_144625


namespace NUMINAMATH_CALUDE_cube_preserves_order_l1446_144628

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l1446_144628


namespace NUMINAMATH_CALUDE_population_change_l1446_144659

theorem population_change (x : ℝ) : 
  let initial_population : ℝ := 10000
  let first_year_population : ℝ := initial_population * (1 + x / 100)
  let second_year_population : ℝ := first_year_population * (1 - 5 / 100)
  second_year_population = 9975 → x = 5 := by
sorry

end NUMINAMATH_CALUDE_population_change_l1446_144659


namespace NUMINAMATH_CALUDE_fans_with_all_items_l1446_144698

def stadium_capacity : ℕ := 4800
def scarf_interval : ℕ := 80
def hat_interval : ℕ := 40
def whistle_interval : ℕ := 60

theorem fans_with_all_items :
  (stadium_capacity / (lcm scarf_interval (lcm hat_interval whistle_interval))) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l1446_144698


namespace NUMINAMATH_CALUDE_max_total_marks_is_1127_l1446_144627

/-- Represents the pass requirements and scores for a student's exam -/
structure ExamResults where
  math_pass_percent : ℚ
  physics_pass_percent : ℚ
  chem_pass_percent : ℚ
  math_score : ℕ
  math_fail_margin : ℕ
  physics_score : ℕ
  physics_fail_margin : ℕ
  chem_score : ℕ
  chem_fail_margin : ℕ

/-- Calculates the maximum total marks obtainable across all subjects -/
def maxTotalMarks (results : ExamResults) : ℕ :=
  sorry

/-- Theorem stating that given the exam results, the maximum total marks is 1127 -/
theorem max_total_marks_is_1127 (results : ExamResults) 
  (h1 : results.math_pass_percent = 36/100)
  (h2 : results.physics_pass_percent = 40/100)
  (h3 : results.chem_pass_percent = 45/100)
  (h4 : results.math_score = 130)
  (h5 : results.math_fail_margin = 14)
  (h6 : results.physics_score = 120)
  (h7 : results.physics_fail_margin = 20)
  (h8 : results.chem_score = 160)
  (h9 : results.chem_fail_margin = 10) :
  maxTotalMarks results = 1127 :=
  sorry

end NUMINAMATH_CALUDE_max_total_marks_is_1127_l1446_144627


namespace NUMINAMATH_CALUDE_parabola_range_theorem_l1446_144678

/-- Represents a quadratic function of the form f(x) = x^2 + bx + c -/
def QuadraticFunction (b c : ℝ) := λ x : ℝ => x^2 + b*x + c

theorem parabola_range_theorem (b c : ℝ) :
  (QuadraticFunction b c (-1) = 0) →
  (QuadraticFunction b c 3 = 0) →
  (∀ x : ℝ, QuadraticFunction b c x > -3 ↔ (x < 0 ∨ x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_range_theorem_l1446_144678


namespace NUMINAMATH_CALUDE_teaching_arrangements_count_l1446_144665

def number_of_teachers : ℕ := 3
def number_of_classes : ℕ := 6
def classes_per_teacher : ℕ := 2

theorem teaching_arrangements_count :
  (Nat.choose number_of_classes classes_per_teacher) *
  (Nat.choose (number_of_classes - classes_per_teacher) classes_per_teacher) *
  (Nat.choose (number_of_classes - 2 * classes_per_teacher) classes_per_teacher) = 90 := by
  sorry

end NUMINAMATH_CALUDE_teaching_arrangements_count_l1446_144665


namespace NUMINAMATH_CALUDE_planar_graph_inequality_l1446_144696

/-- A planar graph is a graph that can be embedded in the plane without edge crossings. -/
structure PlanarGraph where
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces

/-- For any planar graph, twice the number of edges is greater than or equal to
    three times the number of faces. -/
theorem planar_graph_inequality (G : PlanarGraph) : 2 * G.E ≥ 3 * G.F := by
  sorry

end NUMINAMATH_CALUDE_planar_graph_inequality_l1446_144696


namespace NUMINAMATH_CALUDE_min_value_expression_l1446_144692

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 2) (hab : a + b = 2) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 2 ∧
  (∀ (a' b' c' : ℝ), a' > 0 → b' > 0 → c' > 2 → a' + b' = 2 →
    (a' * c' / b' + c' / (a' * b') - c' / 2 + Real.sqrt 5 / (c' - 2) ≥ Real.sqrt 10 + Real.sqrt 5)) ∧
  (x * (2 + Real.sqrt 2) / y + (2 + Real.sqrt 2) / (x * y) - (2 + Real.sqrt 2) / 2 + Real.sqrt 5 / Real.sqrt 2 = Real.sqrt 10 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1446_144692


namespace NUMINAMATH_CALUDE_sequence_has_unique_occurrence_l1446_144670

def is_unique_occurrence (s : ℕ → ℝ) (x : ℝ) : Prop :=
  ∃! n : ℕ, s n = x

theorem sequence_has_unique_occurrence
  (a : ℕ → ℝ)
  (h_inc : ∀ i j : ℕ, i < j → a i < a j)
  (h_bound : ∀ i : ℕ, 0 < a i ∧ a i < 1) :
  ∃ x : ℝ, is_unique_occurrence (λ i => a i / i) x :=
sorry

end NUMINAMATH_CALUDE_sequence_has_unique_occurrence_l1446_144670


namespace NUMINAMATH_CALUDE_maud_olive_flea_multiple_l1446_144657

/-- The number of fleas on Gertrude -/
def gertrude_fleas : ℕ := 10

/-- The number of fleas on Olive -/
def olive_fleas : ℕ := gertrude_fleas / 2

/-- The total number of fleas on all chickens -/
def total_fleas : ℕ := 40

/-- The number of fleas on Maud -/
def maud_fleas : ℕ := total_fleas - gertrude_fleas - olive_fleas

/-- The multiple of fleas Maud has compared to Olive -/
def maud_olive_multiple : ℕ := maud_fleas / olive_fleas

theorem maud_olive_flea_multiple :
  maud_olive_multiple = 5 := by sorry

end NUMINAMATH_CALUDE_maud_olive_flea_multiple_l1446_144657


namespace NUMINAMATH_CALUDE_focus_to_latus_rectum_distance_l1446_144629

/-- A parabola with equation y^2 = 2px (p > 0) whose latus rectum is tangent to the circle (x-3)^2 + y^2 = 16 -/
structure TangentParabola where
  p : ℝ
  p_pos : p > 0
  latus_rectum_tangent : ∃ (x y : ℝ), y^2 = 2*p*x ∧ (x-3)^2 + y^2 = 16

/-- The distance from the focus of the parabola to the latus rectum is 2 -/
theorem focus_to_latus_rectum_distance (tp : TangentParabola) : tp.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_focus_to_latus_rectum_distance_l1446_144629


namespace NUMINAMATH_CALUDE_polynomial_equality_l1446_144662

theorem polynomial_equality : 105^5 - 5 * 105^4 + 10 * 105^3 - 10 * 105^2 + 5 * 105 - 1 = 11714628224 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1446_144662


namespace NUMINAMATH_CALUDE_work_completion_time_l1446_144672

/-- The time taken to complete a work given two workers with different rates and a specific work pattern. -/
theorem work_completion_time 
  (p_time q_time : ℝ) 
  (solo_time : ℝ) 
  (h1 : p_time > 0) 
  (h2 : q_time > 0) 
  (h3 : solo_time > 0) 
  (h4 : solo_time < p_time) :
  let p_rate := 1 / p_time
  let q_rate := 1 / q_time
  let work_done_solo := solo_time * p_rate
  let remaining_work := 1 - work_done_solo
  let combined_rate := p_rate + q_rate
  let remaining_time := remaining_work / combined_rate
  solo_time + remaining_time = 20 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1446_144672


namespace NUMINAMATH_CALUDE_trip_duration_l1446_144637

theorem trip_duration (duration_first : ℝ) 
  (h1 : duration_first ≥ 0)
  (h2 : duration_first + 2 * duration_first + 2 * duration_first = 10) :
  duration_first = 2 := by
sorry

end NUMINAMATH_CALUDE_trip_duration_l1446_144637


namespace NUMINAMATH_CALUDE_sum_mod_nine_l1446_144664

theorem sum_mod_nine : (9156 + 9157 + 9158 + 9159 + 9160) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l1446_144664


namespace NUMINAMATH_CALUDE_prime_factors_count_l1446_144602

/-- The total number of prime factors in the expression (4)^11 × (7)^3 × (11)^2 -/
def totalPrimeFactors : ℕ := 27

/-- The exponent of 4 in the expression -/
def exponent4 : ℕ := 11

/-- The exponent of 7 in the expression -/
def exponent7 : ℕ := 3

/-- The exponent of 11 in the expression -/
def exponent11 : ℕ := 2

theorem prime_factors_count : 
  totalPrimeFactors = 2 * exponent4 + exponent7 + exponent11 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_count_l1446_144602


namespace NUMINAMATH_CALUDE_r_amount_l1446_144623

theorem r_amount (total : ℝ) (r_fraction : ℝ) (h1 : total = 5000) (h2 : r_fraction = 2/3) :
  r_fraction * (total / (1 + r_fraction)) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_r_amount_l1446_144623


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_l1446_144686

theorem smallest_four_digit_multiple : ∃ (n : ℕ), 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  (n % 5 = 0) ∧
  (n % 11 = 7) ∧
  (n % 7 = 4) ∧
  (n % 9 = 4) ∧
  (∀ m : ℕ, 
    (m ≥ 1000 ∧ m < 10000) ∧ 
    (m % 5 = 0) ∧
    (m % 11 = 7) ∧
    (m % 7 = 4) ∧
    (m % 9 = 4) →
    n ≤ m) ∧
  n = 2020 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_l1446_144686


namespace NUMINAMATH_CALUDE_parabola_coefficient_l1446_144649

/-- A quadratic function with vertex form (x - h)^2 where h is the x-coordinate of the vertex -/
def quadratic_vertex_form (a : ℝ) (h : ℝ) (x : ℝ) : ℝ := a * (x - h)^2

theorem parabola_coefficient (f : ℝ → ℝ) (h : ℝ) (a : ℝ) :
  (∀ x, f x = quadratic_vertex_form a h x) →
  f 5 = -36 →
  h = 2 →
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l1446_144649


namespace NUMINAMATH_CALUDE_three_student_committees_l1446_144618

theorem three_student_committees (n k : ℕ) (hn : n = 10) (hk : k = 3) :
  Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_three_student_committees_l1446_144618


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l1446_144652

/-- Man's rowing problem with wind resistance -/
theorem mans_rowing_speed (upstream_speed downstream_speed wind_effect : ℝ) 
  (h1 : upstream_speed = 25)
  (h2 : downstream_speed = 45)
  (h3 : wind_effect = 2) :
  let still_water_speed := (upstream_speed + downstream_speed) / 2
  let adjusted_upstream_speed := upstream_speed - wind_effect
  let adjusted_downstream_speed := downstream_speed + wind_effect
  let adjusted_still_water_speed := (adjusted_upstream_speed + adjusted_downstream_speed) / 2
  adjusted_still_water_speed = still_water_speed :=
by sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_l1446_144652


namespace NUMINAMATH_CALUDE_length_of_AB_l1446_144694

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x + y - 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y + 9 = 0

-- Define that l is the axis of symmetry of C
def is_axis_of_symmetry (k : ℝ) : Prop := 
  ∀ x y : ℝ, line_l k x y → (circle_C x y ↔ circle_C (2*3-x) (2*(-1)-y))

-- Define point A
def point_A (k : ℝ) : ℝ × ℝ := (0, k)

-- Define that there exists a point B on circle C such that AB is tangent to C
def exists_tangent_point (k : ℝ) : Prop :=
  ∃ B : ℝ × ℝ, circle_C B.1 B.2 ∧ 
    ((B.1 - 0) * (B.2 - k) = 1 ∨ (B.1 - 0) * (B.2 - k) = -1)

-- Theorem statement
theorem length_of_AB (k : ℝ) :
  is_axis_of_symmetry k →
  exists_tangent_point k →
  ∃ B : ℝ × ℝ, circle_C B.1 B.2 ∧ 
    ((B.1 - 0) * (B.2 - k) = 1 ∨ (B.1 - 0) * (B.2 - k) = -1) ∧
    Real.sqrt ((B.1 - 0)^2 + (B.2 - k)^2) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_l1446_144694


namespace NUMINAMATH_CALUDE_library_donation_l1446_144697

/-- The number of books donated to the library --/
def books_donated (num_students : ℕ) (books_per_student : ℕ) (shortfall : ℕ) : ℕ :=
  num_students * books_per_student - shortfall

/-- Theorem stating the number of books donated to the library --/
theorem library_donation (num_students : ℕ) (books_per_student : ℕ) (shortfall : ℕ) :
  books_donated num_students books_per_student shortfall = 294 :=
by
  sorry

#eval books_donated 20 15 6

end NUMINAMATH_CALUDE_library_donation_l1446_144697


namespace NUMINAMATH_CALUDE_bear_food_in_victors_l1446_144695

/-- The number of "Victors" worth of food a bear eats in 3 weeks -/
def victors_worth_of_food (bear_food_per_day : ℕ) (victor_weight : ℕ) (weeks : ℕ) : ℕ :=
  (bear_food_per_day * weeks * 7) / victor_weight

/-- Theorem stating that a bear eating 90 pounds of food per day would eat 15 "Victors" worth of food in 3 weeks, given that Victor weighs 126 pounds -/
theorem bear_food_in_victors : victors_worth_of_food 90 126 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_bear_food_in_victors_l1446_144695


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l1446_144658

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from the pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  (area pan.panDimensions) / (area pan.pieceDimensions)

theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 24, width := 15 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 60 := by sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l1446_144658


namespace NUMINAMATH_CALUDE_women_workers_l1446_144611

/-- Represents a company with workers and retirement plans. -/
structure Company where
  total_workers : ℕ
  workers_without_plan : ℕ
  women_without_plan : ℕ
  men_with_plan : ℕ
  total_men : ℕ

/-- Conditions for the company structure -/
def company_conditions (c : Company) : Prop :=
  c.workers_without_plan = c.total_workers / 3 ∧
  c.women_without_plan = (2 * c.workers_without_plan) / 5 ∧
  c.men_with_plan = ((2 * c.total_workers) / 3) * 2 / 5 ∧
  c.total_men = 120

/-- The theorem to prove -/
theorem women_workers (c : Company) 
  (h : company_conditions c) : c.total_workers - c.total_men = 330 := by
  sorry

#check women_workers

end NUMINAMATH_CALUDE_women_workers_l1446_144611


namespace NUMINAMATH_CALUDE_solution_set_characterization_l1446_144680

def solution_set (x y : ℝ) : Prop :=
  3 * x - 4 * y + 12 > 0 ∧ x + y - 2 < 0

theorem solution_set_characterization (x y : ℝ) :
  solution_set x y ↔ (3 * x - 4 * y + 12 > 0 ∧ x + y - 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l1446_144680


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l1446_144630

-- Define the functions
def f (a b x : ℝ) : ℝ := -|x - a|^2 + b
def g (c d x : ℝ) : ℝ := |x - c|^2 + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) :
  f a b 1 = 4 ∧ g c d 1 = 4 ∧ f a b 7 = 2 ∧ g c d 7 = 2 → a + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l1446_144630


namespace NUMINAMATH_CALUDE_diamond_calculation_l1446_144601

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- State the theorem
theorem diamond_calculation :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29/132 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l1446_144601


namespace NUMINAMATH_CALUDE_set_intersection_equality_l1446_144600

theorem set_intersection_equality (S T : Set ℝ) : 
  S = {y | ∃ x, y = (3 : ℝ) ^ x} →
  T = {y | ∃ x, y = x^2 + 1} →
  S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l1446_144600


namespace NUMINAMATH_CALUDE_derivative_even_implies_b_zero_l1446_144682

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

-- Define the derivative of f
def f_deriv (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- State the theorem
theorem derivative_even_implies_b_zero (a b c : ℝ) :
  (∀ x : ℝ, f_deriv a b c x = f_deriv a b c (-x)) →
  b = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_even_implies_b_zero_l1446_144682


namespace NUMINAMATH_CALUDE_rectangle_arrangement_exists_l1446_144688

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents an arrangement of rectangles -/
structure Arrangement where
  height : ℝ
  width : ℝ

/-- Theorem: It is possible to arrange five identical rectangles with perimeter 10
    to form a single rectangle with perimeter 22 -/
theorem rectangle_arrangement_exists : ∃ (small : Rectangle) (arr : Arrangement),
  perimeter small = 10 ∧
  arr.height = 5 * small.length ∧
  arr.width = small.width ∧
  2 * (arr.height + arr.width) = 22 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_exists_l1446_144688


namespace NUMINAMATH_CALUDE_chest_to_treadmill_ratio_l1446_144693

/-- The price of the treadmill in dollars -/
def treadmill_price : ℝ := 100

/-- The price of the television in dollars -/
def tv_price : ℝ := 3 * treadmill_price

/-- The total sum of money from the sale in dollars -/
def total_sum : ℝ := 600

/-- The price of the chest of drawers in dollars -/
def chest_price : ℝ := total_sum - treadmill_price - tv_price

/-- The theorem stating that the ratio of the chest price to the treadmill price is 2:1 -/
theorem chest_to_treadmill_ratio :
  chest_price / treadmill_price = 2 := by sorry

end NUMINAMATH_CALUDE_chest_to_treadmill_ratio_l1446_144693


namespace NUMINAMATH_CALUDE_parallel_line_through_A_l1446_144689

-- Define the point A
def A : ℝ × ℝ × ℝ := (-2, 3, 1)

-- Define the planes that form the given line
def plane1 (x y z : ℝ) : Prop := x - 2*y - z - 2 = 0
def plane2 (x y z : ℝ) : Prop := 2*x + 3*y - z + 1 = 0

-- Define the direction vector of the given line
def direction_vector : ℝ × ℝ × ℝ := (5, -1, 7)

-- Define the equation of the parallel line passing through A
def parallel_line (x y z : ℝ) : Prop :=
  (x + 2) / 5 = (y - 3) / (-1) ∧ (y - 3) / (-1) = (z - 1) / 7

-- Theorem statement
theorem parallel_line_through_A :
  ∀ (x y z : ℝ), 
    (∃ (t : ℝ), x = -2 + 5*t ∧ y = 3 - t ∧ z = 1 + 7*t) →
    parallel_line x y z :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_A_l1446_144689


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1446_144674

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the perimeter of the quadrilateral formed by lines parallel to its asymptotes
    drawn from its left and right foci is 8b, then the equation of its asymptotes is y = ±x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c : ℝ, 2 * b = Real.sqrt ((b^2 * c^2) / a^2 + c^2)) →
  (∀ x y : ℝ, (y = x ∨ y = -x) ↔ y^2 = x^2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1446_144674


namespace NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l1446_144634

/-- 
A perfect cube is a number that is the result of multiplying an integer by itself twice.
-/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

/-- 
The smallest positive integer x such that 1152x is a perfect cube is 36.
-/
theorem smallest_x_for_perfect_cube : 
  (∀ y : ℕ+, is_perfect_cube (1152 * y) → y ≥ 36) ∧ 
  is_perfect_cube (1152 * 36) := by
  sorry


end NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l1446_144634


namespace NUMINAMATH_CALUDE_min_value_problem_l1446_144614

theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (1 / a + 1 / b + 1 / c ≥ 9) ∧ (1 / (3 * a + 2) + 1 / (3 * b + 2) + 1 / (3 * c + 2) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l1446_144614


namespace NUMINAMATH_CALUDE_even_sum_condition_l1446_144619

-- Define what it means for a number to be even
def IsEven (n : Int) : Prop := ∃ k : Int, n = 2 * k

-- Statement of the theorem
theorem even_sum_condition :
  (∀ a b : Int, IsEven a ∧ IsEven b → IsEven (a + b)) ∧
  (∃ a b : Int, IsEven (a + b) ∧ (¬IsEven a ∨ ¬IsEven b)) := by
  sorry

end NUMINAMATH_CALUDE_even_sum_condition_l1446_144619


namespace NUMINAMATH_CALUDE_trail_distribution_count_l1446_144612

/-- The number of ways to distribute 4 people familiar with trails into two groups of 2 each -/
def trail_distribution_ways : ℕ := Nat.choose 4 2

/-- Theorem stating that the number of ways to distribute 4 people familiar with trails
    into two groups of 2 each is equal to 6 -/
theorem trail_distribution_count : trail_distribution_ways = 6 := by
  sorry

end NUMINAMATH_CALUDE_trail_distribution_count_l1446_144612


namespace NUMINAMATH_CALUDE_phika_inequality_l1446_144643

/-- A sextuple of positive real numbers is phika if the sum of a's equals the sum of b's equals 1 -/
def IsPhika (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) : Prop :=
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0 ∧
  a₁ + a₂ + a₃ = 1 ∧ b₁ + b₂ + b₃ = 1

theorem phika_inequality :
  (∃ a₁ a₂ a₃ b₁ b₂ b₃ : ℝ, IsPhika a₁ a₂ a₃ b₁ b₂ b₃ ∧
    a₁ * (Real.sqrt b₁ + a₂) + a₂ * (Real.sqrt b₂ + a₃) + a₃ * (Real.sqrt b₃ + a₁) > 1 - 1 / (2022^2022)) ∧
  (∀ a₁ a₂ a₃ b₁ b₂ b₃ : ℝ, IsPhika a₁ a₂ a₃ b₁ b₂ b₃ →
    a₁ * (Real.sqrt b₁ + a₂) + a₂ * (Real.sqrt b₂ + a₃) + a₃ * (Real.sqrt b₃ + a₁) < 1) := by
  sorry

end NUMINAMATH_CALUDE_phika_inequality_l1446_144643


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1446_144642

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  sum : ℕ → ℚ -- Sum function
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2
  term_formula : ∀ n, a n = a 1 + (n - 1) * d

/-- The common difference of the arithmetic sequence is 4 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence)
  (sum_5 : seq.sum 5 = -15)
  (sum_terms : seq.a 2 + seq.a 5 = -2) :
  seq.d = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1446_144642


namespace NUMINAMATH_CALUDE_line_equations_l1446_144606

-- Define the lines m and n
def line_m (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line_n (x y : ℝ) : Prop := x + y - 3 = 0

-- Define point P as the intersection of m and n
def point_P : ℝ × ℝ := (1, 2)

-- Define points A and B
def point_A : ℝ × ℝ := (1, 3)
def point_B : ℝ × ℝ := (3, 2)

-- Define line l
def line_l (x y : ℝ) : Prop := (x + 2 * y - 4 = 0) ∨ (x = 2)

-- Define line l₁
def line_l1 (x y : ℝ) : Prop := y = -1/2 * x + 2

-- State the theorem
theorem line_equations :
  (∀ x y : ℝ, line_m x y ∧ line_n x y → (x, y) = point_P) →
  (∀ x y : ℝ, line_l x y → (x, y) = point_P) →
  (∀ x y : ℝ, line_l1 x y → (x, y) = point_P) →
  (∀ x y : ℝ, line_l x y → 
    abs ((2*x - 2*point_A.1 + y - point_A.2) / Real.sqrt (5)) = 
    abs ((2*x - 2*point_B.1 + y - point_B.2) / Real.sqrt (5))) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    (∀ x y : ℝ, line_l1 x y ↔ x/a + y/b = 1) ∧
    1/2 * a * b = 4) →
  (∀ x y : ℝ, line_l x y ∨ line_l1 x y) :=
sorry

end NUMINAMATH_CALUDE_line_equations_l1446_144606


namespace NUMINAMATH_CALUDE_granddaughter_age_l1446_144685

/-- Represents a family with three generations -/
structure Family where
  betty_age : ℕ
  daughter_age : ℕ
  granddaughter_age : ℕ

/-- The age relationship in the family -/
def valid_family_ages (f : Family) : Prop :=
  f.betty_age = 60 ∧
  f.daughter_age = f.betty_age - (f.betty_age * 40 / 100) ∧
  f.granddaughter_age = f.daughter_age / 3

/-- Theorem stating the granddaughter's age in the family -/
theorem granddaughter_age (f : Family) (h : valid_family_ages f) : 
  f.granddaughter_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_granddaughter_age_l1446_144685


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_parallel_planes_perpendicular_implication_l1446_144604

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Theorem 1
theorem parallel_perpendicular_implication
  (m n : Line) (α : Plane)
  (h1 : parallel_lines m n)
  (h2 : perpendicular m α) :
  perpendicular n α :=
sorry

-- Theorem 2
theorem parallel_planes_perpendicular_implication
  (m n : Line) (α β : Plane)
  (h1 : parallel_planes α β)
  (h2 : parallel_lines m n)
  (h3 : perpendicular m α) :
  perpendicular n β :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_parallel_planes_perpendicular_implication_l1446_144604


namespace NUMINAMATH_CALUDE_function_condition_implies_a_range_l1446_144620

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log (x + 1) - a * x

theorem function_condition_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → (f a x + a * x) / exp x ≤ a * x) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_function_condition_implies_a_range_l1446_144620


namespace NUMINAMATH_CALUDE_total_distinct_plants_l1446_144668

def X : ℕ := 600
def Y : ℕ := 500
def Z : ℕ := 400
def XY : ℕ := 70
def XZ : ℕ := 80
def YZ : ℕ := 60
def XYZ : ℕ := 30

theorem total_distinct_plants : X + Y + Z - XY - XZ - YZ + XYZ = 1320 := by
  sorry

end NUMINAMATH_CALUDE_total_distinct_plants_l1446_144668


namespace NUMINAMATH_CALUDE_incorrect_addition_statement_l1446_144632

theorem incorrect_addition_statement : 
  (8 + 34 ≠ 32) ∧ (17 + 17 = 34) ∧ (15 + 13 = 28) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_addition_statement_l1446_144632


namespace NUMINAMATH_CALUDE_smallest_integers_for_720_square_and_cube_l1446_144679

theorem smallest_integers_for_720_square_and_cube (a b : ℕ+) : 
  (∀ x : ℕ+, x < a → ¬∃ y : ℕ, 720 * x = y * y) ∧
  (∀ x : ℕ+, x < b → ¬∃ y : ℕ, 720 * x = y * y * y) ∧
  (∃ y : ℕ, 720 * a = y * y) ∧
  (∃ y : ℕ, 720 * b = y * y * y) →
  a + b = 305 := by
sorry

end NUMINAMATH_CALUDE_smallest_integers_for_720_square_and_cube_l1446_144679


namespace NUMINAMATH_CALUDE_prob_diff_absolute_l1446_144651

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1200

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 800

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def prob_same_color : ℚ :=
  (red_marbles.choose 2 + black_marbles.choose 2) / total_marbles.choose 2

/-- The probability of drawing two marbles of different colors -/
def prob_diff_color : ℚ :=
  (red_marbles * black_marbles) / total_marbles.choose 2

/-- Theorem: The absolute difference between the probability of drawing two marbles
    of the same color and the probability of drawing two marbles of different colors
    is 789/19990 -/
theorem prob_diff_absolute : |prob_same_color - prob_diff_color| = 789 / 19990 := by
  sorry

end NUMINAMATH_CALUDE_prob_diff_absolute_l1446_144651


namespace NUMINAMATH_CALUDE_monotone_cubic_function_condition_l1446_144669

/-- Given a function f(x) = -x^3 + bx that is monotonically increasing on (0, 1),
    prove that b ≥ 3 -/
theorem monotone_cubic_function_condition (b : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, Monotone (fun x => -x^3 + b*x)) →
  b ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_monotone_cubic_function_condition_l1446_144669


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1446_144677

theorem hyperbola_equation (ellipse : (ℝ × ℝ) → Prop) 
  (ellipse_eq : ∀ x y, ellipse (x, y) ↔ x^2/27 + y^2/36 = 1)
  (shared_foci : ∃ f1 f2 : ℝ × ℝ, (∀ x y, ellipse (x, y) → 
    (x - f1.1)^2 + (y - f1.2)^2 - ((x - f2.1)^2 + (y - f2.2)^2) = 36) ∧
    (∀ x y, x^2/4 - y^2/5 = 1 → 
    (x - f1.1)^2 + (y - f1.2)^2 - ((x - f2.1)^2 + (y - f2.2)^2) = 9))
  (point_on_hyperbola : (Real.sqrt 15)^2/4 - 4^2/5 = 1) :
  ∀ x y, x^2/4 - y^2/5 = 1 ↔ 
    ∃ f1 f2 : ℝ × ℝ, (∀ a b, ellipse (a, b) → 
    (a - f1.1)^2 + (b - f1.2)^2 - ((a - f2.1)^2 + (b - f2.2)^2) = 36) ∧
    (x - f1.1)^2 + (y - f1.2)^2 - ((x - f2.1)^2 + (y - f2.2)^2) = 9 :=
sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l1446_144677


namespace NUMINAMATH_CALUDE_range_equals_std_dev_l1446_144699

/-- A symmetric distribution about a mean -/
structure SymmetricDistribution where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation
  symmetric : Bool
  within_range : ℝ → ℝ  -- function that gives the proportion within a range
  less_than : ℝ → ℝ  -- function that gives the proportion less than a value

/-- Theorem stating the relationship between the range and standard deviation -/
theorem range_equals_std_dev (D : SymmetricDistribution) (R : ℝ) :
  D.symmetric = true →
  D.within_range R = 0.68 →
  D.less_than (D.μ + R) = 0.84 →
  R = D.σ :=
by sorry

end NUMINAMATH_CALUDE_range_equals_std_dev_l1446_144699


namespace NUMINAMATH_CALUDE_sotka_not_divisible_by_nine_l1446_144666

/-- Represents a digit in the range 0 to 9 -/
def Digit := Fin 10

/-- Represents the mapping of letters to digits -/
def LetterToDigit := Char → Digit

/-- Checks if all characters in a string are mapped to unique digits -/
def allUnique (s : String) (m : LetterToDigit) : Prop :=
  ∀ c₁ c₂, c₁ ∈ s.data → c₂ ∈ s.data → c₁ ≠ c₂ → m c₁ ≠ m c₂

/-- Converts a string to a number using the given mapping -/
def toNumber (s : String) (m : LetterToDigit) : ℕ :=
  s.data.foldr (λ c acc => acc * 10 + (m c).val) 0

/-- The main theorem -/
theorem sotka_not_divisible_by_nine (m : LetterToDigit) : 
  allUnique "ДЕВЯНОСТО" m →
  allUnique "ДЕВЯТКА" m →
  allUnique "СОТКА" m →
  90 ∣ toNumber "ДЕВЯНОСТО" m →
  9 ∣ toNumber "ДЕВЯТКА" m →
  ¬(9 ∣ toNumber "СОТКА" m) := by
  sorry


end NUMINAMATH_CALUDE_sotka_not_divisible_by_nine_l1446_144666


namespace NUMINAMATH_CALUDE_twenty_fifth_number_l1446_144626

def twisted_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- We define 0th term as 0 for convenience
  | 1 => 1  -- First term is 1
  | n + 1 => 
    if n % 5 = 0 then twisted_sequence n + 1  -- Every 6th number (5th index) is previous + 1
    else 2 * twisted_sequence n  -- Otherwise, double the previous number

theorem twenty_fifth_number : twisted_sequence 25 = 69956 := by
  sorry

end NUMINAMATH_CALUDE_twenty_fifth_number_l1446_144626


namespace NUMINAMATH_CALUDE_rounding_estimate_less_than_exact_l1446_144633

theorem rounding_estimate_less_than_exact (x y z : ℝ) 
  (hx : 1.5 < x ∧ x < 2) 
  (hy : 9 < y ∧ y < 9.5) 
  (hz : 3 < z ∧ z < 3.5) : 
  1 * 9 + 4 < x * y + z := by
  sorry

end NUMINAMATH_CALUDE_rounding_estimate_less_than_exact_l1446_144633


namespace NUMINAMATH_CALUDE_triple_hash_70_approx_8_l1446_144613

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.4 * N + 2

-- State the theorem
theorem triple_hash_70_approx_8 : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |hash (hash (hash 70)) - 8| < ε :=
sorry

end NUMINAMATH_CALUDE_triple_hash_70_approx_8_l1446_144613


namespace NUMINAMATH_CALUDE_four_digit_difference_l1446_144663

def original_number : ℕ := 201312210840

def is_valid_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ (d1 d2 d3 d4 : ℕ), 
    d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n ∧
    (d1 = 2 ∨ d1 = 0 ∨ d1 = 1 ∨ d1 = 3 ∨ d1 = 8 ∨ d1 = 4) ∧
    (d2 = 2 ∨ d2 = 0 ∨ d2 = 1 ∨ d2 = 3 ∨ d2 = 8 ∨ d2 = 4) ∧
    (d3 = 2 ∨ d3 = 0 ∨ d3 = 1 ∨ d3 = 3 ∨ d3 = 8 ∨ d3 = 4) ∧
    (d4 = 2 ∨ d4 = 0 ∨ d4 = 1 ∨ d4 = 3 ∨ d4 = 8 ∨ d4 = 4))

theorem four_digit_difference :
  ∃ (max min : ℕ), 
    is_valid_four_digit max ∧
    is_valid_four_digit min ∧
    (∀ n, is_valid_four_digit n → n ≤ max) ∧
    (∀ n, is_valid_four_digit n → min ≤ n) ∧
    max - min = 2800 := by sorry

end NUMINAMATH_CALUDE_four_digit_difference_l1446_144663


namespace NUMINAMATH_CALUDE_infinite_solutions_and_sum_of_exceptions_l1446_144645

/-- Given an equation (x+B)(Ax+40) / ((x+C)(x+8)) = 3, this theorem proves that
    for specific values of A, B, and C, the equation has infinitely many solutions,
    and provides the sum of x values that do not satisfy the equation. -/
theorem infinite_solutions_and_sum_of_exceptions :
  ∃ (A B C : ℚ),
    (A = 3 ∧ B = 8 ∧ C = 40/3) ∧
    (∀ x : ℚ, x ≠ -C → x ≠ -8 →
      (x + B) * (A * x + 40) / ((x + C) * (x + 8)) = 3) ∧
    ((-8) + (-40/3) = -64/3) := by
  sorry


end NUMINAMATH_CALUDE_infinite_solutions_and_sum_of_exceptions_l1446_144645


namespace NUMINAMATH_CALUDE_empty_solution_set_range_min_value_distance_sum_l1446_144675

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) ↔ (-1 < a ∧ a < 0) := by
  sorry

theorem min_value_distance_sum : 
  ∀ x : ℝ, |x - 1| + |x - 2| ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_min_value_distance_sum_l1446_144675


namespace NUMINAMATH_CALUDE_range_of_a_l1446_144673

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*a*x + 4 > 0

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x y : ℝ, (y + (a-1)*x + 2*a - 1 = 0) ∧ 
  ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ≤ -2 ∨ (1 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1446_144673


namespace NUMINAMATH_CALUDE_unique_element_quadratic_l1446_144690

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 4 * x + 4 = 0}

-- State the theorem
theorem unique_element_quadratic (a : ℝ) : 
  (∃! x, x ∈ A a) → a = 0 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_unique_element_quadratic_l1446_144690


namespace NUMINAMATH_CALUDE_square_diff_div_four_xy_eq_one_l1446_144667

theorem square_diff_div_four_xy_eq_one (x y : ℝ) (h : x * y ≠ 0) :
  ((x + y)^2 - (x - y)^2) / (4 * x * y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_div_four_xy_eq_one_l1446_144667


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l1446_144609

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (h : x ≥ -1/3) :
  (1 + x)^n ≥ 1 + n*x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l1446_144609


namespace NUMINAMATH_CALUDE_exists_infinite_subset_with_constant_gcd_l1446_144660

-- Define the set of natural numbers that are products of at most 1990 primes
def ProductOfLimitedPrimes (n : ℕ) : Prop :=
  ∃ (primes : Finset ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ primes.card ≤ 1990 ∧ n = primes.prod id

-- Define the property of A
def InfiniteSetOfLimitedPrimeProducts (A : Set ℕ) : Prop :=
  Set.Infinite A ∧ ∀ a ∈ A, ProductOfLimitedPrimes a

-- The main theorem
theorem exists_infinite_subset_with_constant_gcd
  (A : Set ℕ) (hA : InfiniteSetOfLimitedPrimeProducts A) :
  ∃ (B : Set ℕ) (k : ℕ), Set.Infinite B ∧ B ⊆ A ∧
    ∀ (x y : ℕ), x ∈ B → y ∈ B → x ≠ y → Nat.gcd x y = k :=
sorry

end NUMINAMATH_CALUDE_exists_infinite_subset_with_constant_gcd_l1446_144660


namespace NUMINAMATH_CALUDE_three_digit_number_theorem_l1446_144622

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Checks if a tuple represents a valid three-digit number -/
def isValidThreeDigitNumber (n : ThreeDigitNumber) : Prop :=
  let (a, b, c) := n
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9

/-- Converts a three-digit number tuple to its numerical value -/
def toNumber (n : ThreeDigitNumber) : Nat :=
  let (a, b, c) := n
  100 * a + 10 * b + c

/-- Generates all permutations of a three-digit number -/
def permutations (n : ThreeDigitNumber) : List ThreeDigitNumber :=
  let (a, b, c) := n
  [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]

/-- Calculates the average of the permutations of a three-digit number -/
def averageOfPermutations (n : ThreeDigitNumber) : Nat :=
  (List.sum (List.map toNumber (permutations n))) / 6

/-- Checks if a three-digit number satisfies the given condition -/
def satisfiesCondition (n : ThreeDigitNumber) : Prop :=
  isValidThreeDigitNumber n ∧ averageOfPermutations n = toNumber n

/-- The set of three-digit numbers that satisfy the condition -/
def solutionSet : Set Nat :=
  {370, 407, 481, 518, 592, 629}

/-- The main theorem to be proved -/
theorem three_digit_number_theorem (n : ThreeDigitNumber) :
  satisfiesCondition n ↔ toNumber n ∈ solutionSet := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_theorem_l1446_144622


namespace NUMINAMATH_CALUDE_students_checked_out_early_l1446_144640

theorem students_checked_out_early (initial_students remaining_students : ℕ) 
  (h1 : initial_students = 16)
  (h2 : remaining_students = 9) :
  initial_students - remaining_students = 7 :=
by sorry

end NUMINAMATH_CALUDE_students_checked_out_early_l1446_144640


namespace NUMINAMATH_CALUDE_solution_set_equals_interval_l1446_144624

theorem solution_set_equals_interval :
  {x : ℝ | x ≤ 1} = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_equals_interval_l1446_144624


namespace NUMINAMATH_CALUDE_binder_cost_l1446_144635

theorem binder_cost (book_cost : ℕ) (num_binders : ℕ) (num_notebooks : ℕ) 
  (notebook_cost : ℕ) (total_cost : ℕ) : ℕ :=
by
  have h1 : book_cost = 16 := by sorry
  have h2 : num_binders = 3 := by sorry
  have h3 : num_notebooks = 6 := by sorry
  have h4 : notebook_cost = 1 := by sorry
  have h5 : total_cost = 28 := by sorry
  
  have binder_cost : ℕ := (total_cost - (book_cost + num_notebooks * notebook_cost)) / num_binders
  
  exact binder_cost

end NUMINAMATH_CALUDE_binder_cost_l1446_144635


namespace NUMINAMATH_CALUDE_tim_meditates_one_hour_per_day_l1446_144681

/-- Tim's weekly schedule -/
structure TimSchedule where
  reading_time_per_week : ℝ
  meditation_time_per_day : ℝ

/-- Tim's schedule satisfies the given conditions -/
def valid_schedule (s : TimSchedule) : Prop :=
  s.reading_time_per_week = 14 ∧
  s.reading_time_per_week = 2 * (7 * s.meditation_time_per_day)

/-- Theorem: Tim meditates 1 hour per day -/
theorem tim_meditates_one_hour_per_day (s : TimSchedule) (h : valid_schedule s) :
  s.meditation_time_per_day = 1 := by
  sorry

end NUMINAMATH_CALUDE_tim_meditates_one_hour_per_day_l1446_144681


namespace NUMINAMATH_CALUDE_left_handed_mouse_price_increase_l1446_144610

/-- Represents the store's weekly operation --/
structure StoreOperation where
  daysOpen : Nat
  miceSoldPerDay : Nat
  normalMousePrice : Nat
  weeklyRevenue : Nat

/-- Calculates the percentage increase in price --/
def percentageIncrease (normalPrice leftHandedPrice : Nat) : Nat :=
  ((leftHandedPrice - normalPrice) * 100) / normalPrice

/-- Theorem stating the percentage increase in left-handed mouse price --/
theorem left_handed_mouse_price_increase 
  (store : StoreOperation)
  (h1 : store.daysOpen = 4)
  (h2 : store.miceSoldPerDay = 25)
  (h3 : store.normalMousePrice = 120)
  (h4 : store.weeklyRevenue = 15600) :
  percentageIncrease store.normalMousePrice 
    ((store.weeklyRevenue / store.daysOpen) / store.miceSoldPerDay) = 30 := by
  sorry

#eval percentageIncrease 120 156

end NUMINAMATH_CALUDE_left_handed_mouse_price_increase_l1446_144610


namespace NUMINAMATH_CALUDE_last_three_digits_are_218_l1446_144641

/-- A function that generates the list of positive integers starting with 2 -/
def digitsStartingWith2 (n : ℕ) : ℕ :=
  if n < 10 then 2
  else if n < 100 then 20 + (n - 10)
  else if n < 1000 then 200 + (n - 100)
  else 2000 + (n - 1000)

/-- A function that returns the nth digit in the list -/
def nthDigit (n : ℕ) : ℕ :=
  let number := digitsStartingWith2 ((n - 1) / 4 + 1)
  let digitPosition := (n - 1) % 4
  (number / (10 ^ (3 - digitPosition))) % 10

/-- The theorem to be proved -/
theorem last_three_digits_are_218 :
  (nthDigit 1198) * 100 + (nthDigit 1199) * 10 + nthDigit 1200 = 218 := by
  sorry


end NUMINAMATH_CALUDE_last_three_digits_are_218_l1446_144641


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l1446_144607

/-- The number of walnut trees in the park after planting -/
def trees_after_planting (initial_trees newly_planted_trees : ℕ) : ℕ :=
  initial_trees + newly_planted_trees

/-- Theorem: The number of walnut trees in the park after planting is 77 -/
theorem walnut_trees_after_planting :
  trees_after_planting 22 55 = 77 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l1446_144607


namespace NUMINAMATH_CALUDE_tangent_line_and_monotonicity_and_range_l1446_144644

noncomputable section

open Real

-- Define f(x) = ln x
def f (x : ℝ) : ℝ := log x

-- Define g(x) = f(x) + f''(x)
def g (x : ℝ) : ℝ := f x + (deriv^[2] f) x

theorem tangent_line_and_monotonicity_and_range :
  -- 1. The tangent line to y = f(x) at (1, f(1)) is y = x - 1
  (∀ y, y = deriv f 1 * (x - 1) + f 1 ↔ y = x - 1) ∧
  -- 2. g(x) is decreasing on (0, 1) and increasing on (1, +∞)
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → g x₁ > g x₂) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → g x₁ < g x₂) ∧
  -- 3. For any x > 0, g(a) - g(x) < 1/a holds if and only if 0 < a < e
  (∀ a, (0 < a ∧ a < ℯ) ↔ (∀ x, x > 0 → g a - g x < 1 / a)) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_and_monotonicity_and_range_l1446_144644


namespace NUMINAMATH_CALUDE_olivias_carrots_l1446_144676

theorem olivias_carrots (mom_carrots : ℕ) (good_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : mom_carrots = 14)
  (h2 : good_carrots = 19)
  (h3 : bad_carrots = 15) :
  good_carrots + bad_carrots - mom_carrots = 20 := by
  sorry

end NUMINAMATH_CALUDE_olivias_carrots_l1446_144676

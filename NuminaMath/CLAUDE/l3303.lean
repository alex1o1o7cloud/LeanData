import Mathlib

namespace NUMINAMATH_CALUDE_letter_e_to_square_l3303_330381

/-- Represents a piece cut from the letter E -/
structure Piece where
  area : ℝ
  shape : Set (ℝ × ℝ)

/-- Represents the letter E with its dimensions -/
structure LetterE where
  height : ℝ
  width : ℝ
  horizontal_bar_length : ℝ

/-- Checks if a set of pieces can form a square -/
def can_form_square (pieces : List Piece) : Prop :=
  ∃ (side_length : ℝ), 
    side_length > 0 ∧
    (pieces.map (λ p => p.area)).sum = side_length ^ 2

/-- Checks if a list of pieces is a valid cutting of the letter E -/
def is_valid_cutting (e : LetterE) (pieces : List Piece) : Prop :=
  pieces.length = 5 ∧
  (pieces.map (λ p => p.area)).sum = e.height * e.width + 3 * e.horizontal_bar_length * e.width

/-- Main theorem: It's possible to cut the letter E into five pieces to form a square -/
theorem letter_e_to_square (e : LetterE) : 
  ∃ (pieces : List Piece), is_valid_cutting e pieces ∧ can_form_square pieces := by
  sorry

end NUMINAMATH_CALUDE_letter_e_to_square_l3303_330381


namespace NUMINAMATH_CALUDE_flagpole_height_l3303_330370

theorem flagpole_height (A B C D E : ℝ × ℝ) : 
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let AD : ℝ := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let DE : ℝ := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let AB : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  (A.2 = 0 ∧ B.2 > 0 ∧ C.2 = 0 ∧ D.2 = 0 ∧ E.2 > 0) → -- Points on x-axis or above
  (B.1 = A.1) → -- AB is vertical
  (AC = 4) → -- Wire length on ground
  (AD = 3) → -- Tom's distance from pole
  (DE = 1.8) → -- Tom's height
  ((E.1 - D.1) * (C.1 - A.1) + (E.2 - D.2) * (C.2 - A.2) = 0) → -- DE perpendicular to AC
  (AB = 7.2) -- Flagpole height
  := by sorry

end NUMINAMATH_CALUDE_flagpole_height_l3303_330370


namespace NUMINAMATH_CALUDE_average_height_students_count_l3303_330308

/-- Represents the number of students in different height categories --/
structure HeightDistribution where
  total : ℕ
  short : ℕ
  tall : ℕ
  extremelyTall : ℕ

/-- Calculates the number of students with average height --/
def averageHeightStudents (h : HeightDistribution) : ℕ :=
  h.total - (h.short + h.tall + h.extremelyTall)

/-- Theorem: The number of students with average height in the given class is 110 --/
theorem average_height_students_count (h : HeightDistribution) 
  (h_total : h.total = 400)
  (h_short : h.short = 2 * h.total / 5)
  (h_extremelyTall : h.extremelyTall = h.total / 10)
  (h_tall : h.tall = 90) :
  averageHeightStudents h = 110 := by
  sorry

#eval averageHeightStudents ⟨400, 160, 90, 40⟩

end NUMINAMATH_CALUDE_average_height_students_count_l3303_330308


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3303_330344

theorem imaginary_part_of_complex_fraction : 
  Complex.im (2 * Complex.I / (1 + Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3303_330344


namespace NUMINAMATH_CALUDE_lucy_money_problem_l3303_330363

theorem lucy_money_problem (initial_amount : ℝ) : 
  let doubled := 2 * initial_amount
  let after_loss := doubled * (2/3)
  let after_spending := after_loss * (3/4)
  after_spending = 15 → initial_amount = 15 := by
sorry

end NUMINAMATH_CALUDE_lucy_money_problem_l3303_330363


namespace NUMINAMATH_CALUDE_bankers_discount_example_l3303_330349

/-- Given a bill with face value and true discount, calculates the banker's discount -/
def bankers_discount (face_value : ℚ) (true_discount : ℚ) : ℚ :=
  let present_value := face_value - true_discount
  true_discount + (true_discount^2 / present_value)

/-- Theorem stating that for a bill with face value 540 and true discount 90, 
    the banker's discount is 108 -/
theorem bankers_discount_example : bankers_discount 540 90 = 108 := by
  sorry

end NUMINAMATH_CALUDE_bankers_discount_example_l3303_330349


namespace NUMINAMATH_CALUDE_square_side_length_with_equal_perimeter_circle_l3303_330356

theorem square_side_length_with_equal_perimeter_circle (r : ℝ) :
  ∃ (s : ℝ), (4 * s = 2 * π * r) → (s = 3 * π / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_square_side_length_with_equal_perimeter_circle_l3303_330356


namespace NUMINAMATH_CALUDE_car_distance_proof_l3303_330395

/-- Proves the initial distance between two cars driving towards each other --/
theorem car_distance_proof (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) 
  (h1 : speed1 = 100)
  (h2 : speed1 = 1.25 * speed2)
  (h3 : time = 4) :
  speed1 * time + speed2 * time = 720 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l3303_330395


namespace NUMINAMATH_CALUDE_expression_never_equals_33_l3303_330373

theorem expression_never_equals_33 (x y : ℤ) :
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_expression_never_equals_33_l3303_330373


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3303_330318

/-- Given a train that crosses a platform and passes a stationary man, calculate its speed -/
theorem train_speed_calculation (platform_length : ℝ) (platform_crossing_time : ℝ) (man_passing_time : ℝ) :
  platform_length = 220 →
  platform_crossing_time = 30 →
  man_passing_time = 19 →
  ∃ (train_speed : ℝ), train_speed = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3303_330318


namespace NUMINAMATH_CALUDE_remainder_seven_count_l3303_330317

theorem remainder_seven_count : ∃! k : ℕ, k = (Finset.filter (fun n => 61 % n = 7) (Finset.range 62)).card := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_count_l3303_330317


namespace NUMINAMATH_CALUDE_complex_magnitude_power_l3303_330389

theorem complex_magnitude_power : 
  Complex.abs ((2 : ℂ) + (2 * Complex.I * Real.sqrt 3)) ^ 6 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_power_l3303_330389


namespace NUMINAMATH_CALUDE_derivative_of_f_l3303_330316

noncomputable def f (x : ℝ) : ℝ := 2^x - Real.log x / Real.log 3

theorem derivative_of_f (x : ℝ) (h : x > 0) :
  deriv f x = 2^x * Real.log 2 - 1 / (x * Real.log 3) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3303_330316


namespace NUMINAMATH_CALUDE_sons_age_is_fourteen_l3303_330343

/-- Proves that given the conditions, the son's present age is 14 years -/
theorem sons_age_is_fourteen (son_age father_age : ℕ) : 
  father_age = son_age + 16 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_is_fourteen_l3303_330343


namespace NUMINAMATH_CALUDE_sum_of_products_l3303_330365

theorem sum_of_products : 1234 * 2 + 2341 * 2 + 3412 * 2 + 4123 * 2 = 22220 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l3303_330365


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3303_330384

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, -6)
  let b : ℝ × ℝ := (-1, m)
  parallel a b → m = 3 := by
  sorry

#check parallel_vectors_m_value

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3303_330384


namespace NUMINAMATH_CALUDE_jacob_calorie_consumption_l3303_330399

/-- Jacob's calorie consumption problem -/
theorem jacob_calorie_consumption (planned_max : ℕ) (breakfast lunch dinner : ℕ) 
  (h1 : planned_max < 1800)
  (h2 : breakfast = 400)
  (h3 : lunch = 900)
  (h4 : dinner = 1100) :
  breakfast + lunch + dinner - planned_max = 600 :=
by sorry

end NUMINAMATH_CALUDE_jacob_calorie_consumption_l3303_330399


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l3303_330366

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_8th_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 7 + a 8 + a 9 = 21) : 
  a 8 = 7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l3303_330366


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3303_330342

theorem gcd_of_three_numbers : Nat.gcd 9486 (Nat.gcd 13524 36582) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3303_330342


namespace NUMINAMATH_CALUDE_drama_club_subjects_l3303_330358

theorem drama_club_subjects (total : ℕ) (math : ℕ) (physics : ℕ) (chem : ℕ)
  (math_physics : ℕ) (math_chem : ℕ) (physics_chem : ℕ) (all_three : ℕ)
  (h_total : total = 70)
  (h_math : math = 42)
  (h_physics : physics = 35)
  (h_chem : chem = 25)
  (h_math_physics : math_physics = 18)
  (h_math_chem : math_chem = 10)
  (h_physics_chem : physics_chem = 8)
  (h_all_three : all_three = 5) :
  total - (math + physics + chem - math_physics - math_chem - physics_chem + all_three) = 0 :=
by sorry

end NUMINAMATH_CALUDE_drama_club_subjects_l3303_330358


namespace NUMINAMATH_CALUDE_storage_box_length_l3303_330347

/-- Calculates the length of cubic storage boxes given total volume, cost per box, and total cost -/
theorem storage_box_length (total_volume : ℝ) (cost_per_box : ℝ) (total_cost : ℝ) :
  total_volume = 1080000 ∧ cost_per_box = 0.5 ∧ total_cost = 300 →
  ∃ (length : ℝ), abs (length - (total_volume / (total_cost / cost_per_box))^(1/3)) < 0.1 := by
sorry

#eval (1080000 / (300 / 0.5))^(1/3)

end NUMINAMATH_CALUDE_storage_box_length_l3303_330347


namespace NUMINAMATH_CALUDE_miss_adamson_classes_l3303_330371

/-- The number of classes Miss Adamson has -/
def number_of_classes (students_per_class : ℕ) (sheets_per_student : ℕ) (total_sheets : ℕ) : ℕ :=
  total_sheets / (students_per_class * sheets_per_student)

/-- Proof that Miss Adamson has 4 classes -/
theorem miss_adamson_classes :
  number_of_classes 20 5 400 = 4 := by
  sorry

end NUMINAMATH_CALUDE_miss_adamson_classes_l3303_330371


namespace NUMINAMATH_CALUDE_quadratic_properties_l3303_330329

/-- A quadratic function with the property that y > 0 for -2 < x < 3 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ∀ x : ℝ, -2 < x → x < 3 → 0 < a * x^2 + b * x + c

/-- The properties of the quadratic function that we want to prove -/
theorem quadratic_properties (f : QuadraticFunction) :
  f.b = -f.a ∧
  ∃ x₁ x₂ : ℝ, x₁ = -1/3 ∧ x₂ = 1/2 ∧
    f.c * x₁^2 - f.b * x₁ + f.a = 0 ∧
    f.c * x₂^2 - f.b * x₂ + f.a = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3303_330329


namespace NUMINAMATH_CALUDE_william_window_wash_time_l3303_330300

/-- The time William spends washing vehicles -/
def william_car_wash (window_time : ℕ) : Prop :=
  let normal_car_time := window_time + 7 + 4 + 9
  let suv_time := 2 * normal_car_time
  let total_time := 2 * normal_car_time + suv_time
  total_time = 96

theorem william_window_wash_time :
  ∃ (w : ℕ), william_car_wash w ∧ w = 4 := by
  sorry

end NUMINAMATH_CALUDE_william_window_wash_time_l3303_330300


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_and_triangles_l3303_330374

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

def number_of_triangles (n : ℕ) : ℕ := n.choose 3

theorem dodecagon_diagonals_and_triangles :
  let n : ℕ := 12
  number_of_diagonals n = 54 ∧ number_of_triangles n = 220 := by sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_and_triangles_l3303_330374


namespace NUMINAMATH_CALUDE_sandy_shopping_money_left_l3303_330379

def total_money : ℝ := 300
def spent_percentage : ℝ := 30

theorem sandy_shopping_money_left :
  total_money * (1 - spent_percentage / 100) = 210 := by sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_left_l3303_330379


namespace NUMINAMATH_CALUDE_tangent_slope_condition_l3303_330324

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

-- State the theorem
theorem tangent_slope_condition (a b : ℝ) :
  (∀ x, (deriv (f a b)) x = 2 * a * x) →  -- Derivative of f
  (deriv (f a b)) 1 = 2 →  -- Slope of tangent line at x = 1 is 2
  f a b 1 = 3 →  -- Function value at x = 1 is 3
  b / a = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_condition_l3303_330324


namespace NUMINAMATH_CALUDE_beach_probability_l3303_330392

/-- Given a beach scenario with people wearing sunglasses and caps -/
structure BeachScenario where
  sunglasses : ℕ  -- Number of people wearing sunglasses
  caps : ℕ        -- Number of people wearing caps
  prob_cap_and_sunglasses : ℚ  -- Probability that a person wearing a cap is also wearing sunglasses

/-- The probability that a person wearing sunglasses is also wearing a cap -/
def prob_sunglasses_and_cap (scenario : BeachScenario) : ℚ :=
  (scenario.prob_cap_and_sunglasses * scenario.caps) / scenario.sunglasses

theorem beach_probability (scenario : BeachScenario) 
  (h1 : scenario.sunglasses = 75)
  (h2 : scenario.caps = 60)
  (h3 : scenario.prob_cap_and_sunglasses = 1/3) :
  prob_sunglasses_and_cap scenario = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_beach_probability_l3303_330392


namespace NUMINAMATH_CALUDE_complex_number_on_line_l3303_330375

def complex_number (a : ℝ) : ℂ := (a : ℂ) - Complex.I

theorem complex_number_on_line (a : ℝ) :
  let z := 1 / complex_number a
  (z.re : ℝ) + 2 * (z.im : ℝ) = 0 → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_on_line_l3303_330375


namespace NUMINAMATH_CALUDE_fraction_equality_l3303_330340

theorem fraction_equality (a b c d : ℝ) (h : a / b = c / d) :
  (a + c) / (b + d) = a / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3303_330340


namespace NUMINAMATH_CALUDE_yard_area_l3303_330315

/-- The area of a rectangular yard with a square cut out -/
theorem yard_area (length width cut_size : ℕ) (h1 : length = 20) (h2 : width = 18) (h3 : cut_size = 4) :
  length * width - cut_size * cut_size = 344 := by
sorry

end NUMINAMATH_CALUDE_yard_area_l3303_330315


namespace NUMINAMATH_CALUDE_point_symmetry_l3303_330394

/-- A point is symmetric to another point with respect to the origin if their coordinates sum to zero. -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

/-- The theorem states that the point (2, -3) is symmetric to the point (-2, 3) with respect to the origin. -/
theorem point_symmetry : symmetric_wrt_origin (-2, 3) (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_l3303_330394


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a3_l3303_330397

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_a3 (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h6 : a 6 = 6) (h9 : a 9 = 9) : a 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a3_l3303_330397


namespace NUMINAMATH_CALUDE_min_value_theorem_l3303_330332

theorem min_value_theorem (m n : ℝ) (h1 : m * n > 0) 
  (h2 : -2 * m - n + 1 = 0) : 
  (2 / m + 1 / n) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3303_330332


namespace NUMINAMATH_CALUDE_luke_money_lasted_nine_weeks_l3303_330331

/-- The number of weeks Luke's money lasted given his earnings and spending -/
def weeks_money_lasted (mowing_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (mowing_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem: Given Luke's earnings and spending, his money lasted 9 weeks -/
theorem luke_money_lasted_nine_weeks :
  weeks_money_lasted 9 18 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_luke_money_lasted_nine_weeks_l3303_330331


namespace NUMINAMATH_CALUDE_drums_per_day_l3303_330377

/-- Given that 90 drums are filled in 5 days, prove that 18 drums are filled per day -/
theorem drums_per_day (total_drums : ℕ) (total_days : ℕ) (h1 : total_drums = 90) (h2 : total_days = 5) :
  total_drums / total_days = 18 := by
  sorry

end NUMINAMATH_CALUDE_drums_per_day_l3303_330377


namespace NUMINAMATH_CALUDE_cubic_equation_root_l3303_330390

theorem cubic_equation_root (c d : ℚ) : 
  (∃ x : ℝ, x^3 + c*x^2 + d*x + 44 = 0 ∧ x = 1 - 3*Real.sqrt 5) → c = -3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l3303_330390


namespace NUMINAMATH_CALUDE_evaluate_expression_l3303_330359

theorem evaluate_expression : 9^6 * 3^3 / 3^15 = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3303_330359


namespace NUMINAMATH_CALUDE_parallelogram_on_circle_l3303_330372

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if a point is on a circle
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define a function to check if a quadrilateral is a parallelogram
def isParallelogram (a b c d : ℝ × ℝ) : Prop :=
  (b.1 - a.1 = d.1 - c.1) ∧ (b.2 - a.2 = d.2 - c.2)

theorem parallelogram_on_circle (a c : ℝ × ℝ) (γ : Circle) :
  ∃ (b d : ℝ × ℝ), onCircle b γ ∧ onCircle d γ ∧ isParallelogram a b c d :=
sorry

end NUMINAMATH_CALUDE_parallelogram_on_circle_l3303_330372


namespace NUMINAMATH_CALUDE_equation_solution_l3303_330341

theorem equation_solution : ∃! x : ℝ, (x + 4) / (x - 2) = 3 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3303_330341


namespace NUMINAMATH_CALUDE_colonization_combinations_eq_136_l3303_330302

/-- Represents the number of Earth-like planets -/
def earth_like : ℕ := 6

/-- Represents the number of Mars-like planets -/
def mars_like : ℕ := 6

/-- Represents the resource units required for an Earth-like planet -/
def earth_resource : ℕ := 3

/-- Represents the resource units required for a Mars-like planet -/
def mars_resource : ℕ := 1

/-- Represents the total available resource units -/
def total_resource : ℕ := 18

/-- Calculates the number of different combinations of planets that can be colonized -/
def colonization_combinations : ℕ := sorry

/-- Theorem stating that the number of different combinations of planets that can be colonized is 136 -/
theorem colonization_combinations_eq_136 : colonization_combinations = 136 := by sorry

end NUMINAMATH_CALUDE_colonization_combinations_eq_136_l3303_330302


namespace NUMINAMATH_CALUDE_sum_of_specific_T_l3303_330361

def T (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    (3 * n) / 2
  else
    (3 * n - 1) / 2

theorem sum_of_specific_T : T 18 + T 34 + T 51 = 154 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_T_l3303_330361


namespace NUMINAMATH_CALUDE_sum_of_unknown_angles_l3303_330367

/-- A six-sided polygon with two right angles and three known angles -/
structure HexagonWithKnownAngles where
  -- The three known angles
  angle_P : ℝ
  angle_Q : ℝ
  angle_R : ℝ
  -- Conditions on the known angles
  angle_P_eq : angle_P = 30
  angle_Q_eq : angle_Q = 60
  angle_R_eq : angle_R = 34
  -- The polygon has two right angles
  has_two_right_angles : True

/-- The sum of the two unknown angles in the hexagon is 124° -/
theorem sum_of_unknown_angles (h : HexagonWithKnownAngles) :
  ∃ x y, x + y = 124 := by sorry

end NUMINAMATH_CALUDE_sum_of_unknown_angles_l3303_330367


namespace NUMINAMATH_CALUDE_probability_all_heads_or_tails_proof_l3303_330328

/-- The probability of getting all heads or all tails when flipping six fair coins -/
def probability_all_heads_or_tails : ℚ := 1 / 32

/-- The number of fair coins being flipped -/
def num_coins : ℕ := 6

/-- A fair coin has two possible outcomes -/
def outcomes_per_coin : ℕ := 2

/-- The total number of possible outcomes when flipping the coins -/
def total_outcomes : ℕ := outcomes_per_coin ^ num_coins

/-- The number of favorable outcomes (all heads or all tails) -/
def favorable_outcomes : ℕ := 2

theorem probability_all_heads_or_tails_proof :
  probability_all_heads_or_tails = favorable_outcomes / total_outcomes :=
sorry

end NUMINAMATH_CALUDE_probability_all_heads_or_tails_proof_l3303_330328


namespace NUMINAMATH_CALUDE_tangent_lines_intersection_l3303_330320

/-- Given a circle and four tangent points, proves that the diagonals of the
    trapezoid formed by the tangent lines intersect on the y-axis and that
    the line connecting two specific tangent points passes through this
    intersection point. -/
theorem tangent_lines_intersection
  (ξ η : ℝ)
  (h_ξ_pos : 0 < ξ)
  (h_ξ_lt_1 : ξ < 1)
  (h_circle_eq : ξ^2 + η^2 = 1) :
  ∃ y : ℝ,
    (∀ x : ℝ, x ≠ 0 →
      (y = -((2 * ξ) / (1 + η + ξ)) * x + (1 - η - ξ) / (1 + η + ξ) ↔
       y = ((2 * ξ) / (1 - η + ξ)) * x + (1 + η - ξ) / (1 - η + ξ))) ∧
    y = η / (ξ + 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_intersection_l3303_330320


namespace NUMINAMATH_CALUDE_correct_number_of_officers_l3303_330314

/-- Represents the number of officers in an office with given salary conditions. -/
def number_of_officers : ℕ :=
  let avg_salary_all : ℚ := 120
  let avg_salary_officers : ℚ := 420
  let avg_salary_non_officers : ℚ := 110
  let num_non_officers : ℕ := 450
  15

/-- Theorem stating that the number of officers is correct given the salary conditions. -/
theorem correct_number_of_officers :
  let avg_salary_all : ℚ := 120
  let avg_salary_officers : ℚ := 420
  let avg_salary_non_officers : ℚ := 110
  let num_non_officers : ℕ := 450
  let num_officers := number_of_officers
  (avg_salary_all * (num_officers + num_non_officers : ℚ) =
   avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers) :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_officers_l3303_330314


namespace NUMINAMATH_CALUDE_empty_set_implies_a_range_l3303_330352

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + a * x + 1 = 0}

-- State the theorem
theorem empty_set_implies_a_range (a : ℝ) : 
  A a = ∅ → 0 ≤ a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_empty_set_implies_a_range_l3303_330352


namespace NUMINAMATH_CALUDE_part1_part2_l3303_330326

-- Define a "three times angle triangle"
def is_three_times_angle_triangle (a b c : ℝ) : Prop :=
  (a + b + c = 180) ∧ (a = 3 * b ∨ b = 3 * c ∨ c = 3 * a)

-- Part 1
theorem part1 : is_three_times_angle_triangle 35 40 105 := by sorry

-- Part 2
theorem part2 (a b c : ℝ) (h : is_three_times_angle_triangle a b c) (hb : b = 60) :
  (min a (min b c) = 20) ∨ (min a (min b c) = 30) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3303_330326


namespace NUMINAMATH_CALUDE_third_side_is_three_l3303_330311

/-- Represents a triangle with two known side lengths and one unknown integer side length. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℕ

/-- The triangle inequality theorem for our specific triangle. -/
def triangle_inequality (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

/-- The theorem stating that the third side of the triangle must be 3. -/
theorem third_side_is_three :
  ∀ t : Triangle,
    t.a = 3.14 →
    t.b = 0.67 →
    triangle_inequality t →
    t.c = 3 := by
  sorry

#check third_side_is_three

end NUMINAMATH_CALUDE_third_side_is_three_l3303_330311


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3303_330313

theorem arithmetic_sequence_terms (a₁ : ℕ) (d : ℤ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 20 ∧ d = -2 ∧ aₙ = 10 ∧ aₙ = a₁ + (n - 1) * d → n = 6 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3303_330313


namespace NUMINAMATH_CALUDE_multiply_decimals_l3303_330335

theorem multiply_decimals : 3.6 * 0.3 = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimals_l3303_330335


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_3_mod_8_l3303_330383

theorem largest_integer_less_than_100_remainder_3_mod_8 :
  ∃ n : ℕ, n < 100 ∧ n % 8 = 3 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 3 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_3_mod_8_l3303_330383


namespace NUMINAMATH_CALUDE_no_integer_roots_l3303_330350

theorem no_integer_roots : ¬ ∃ (x : ℤ), x^3 - 3*x^2 - 16*x + 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l3303_330350


namespace NUMINAMATH_CALUDE_second_drive_speed_l3303_330309

def same_distance_drives (v : ℝ) : Prop :=
  let d := 180 / 3  -- distance for each drive
  (d / 4 + d / v + d / 6 = 37) ∧ (d / 4 + d / v + d / 6 > 0)

theorem second_drive_speed : ∃ v : ℝ, same_distance_drives v ∧ v = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_drive_speed_l3303_330309


namespace NUMINAMATH_CALUDE_perpendicular_dot_product_zero_l3303_330323

/-- Given a point P on the curve y = x + 2/x for x > 0, prove that the dot product
    of PA and PB is zero, where A is the foot of the perpendicular from P to y = x,
    and B is the foot of the perpendicular from P to x = 0. -/
theorem perpendicular_dot_product_zero (x : ℝ) (hx : x > 0) :
  let P : ℝ × ℝ := (x, x + 2/x)
  let A : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)
  let B : ℝ × ℝ := (0, x + 2/x)
  let PA : ℝ × ℝ := (A.1 - P.1, A.2 - P.2)
  let PB : ℝ × ℝ := (B.1 - P.1, B.2 - P.2)
  PA.1 * PB.1 + PA.2 * PB.2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_dot_product_zero_l3303_330323


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3303_330322

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x < 2}
def N : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3303_330322


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l3303_330339

theorem factor_implies_c_value (c : ℚ) :
  (∀ x : ℚ, (x + 7) ∣ (c * x^3 + 23 * x^2 - 3 * c * x + 45)) →
  c = 586 / 161 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l3303_330339


namespace NUMINAMATH_CALUDE_intersection_point_sum_l3303_330362

theorem intersection_point_sum (a' b' : ℚ) : 
  (2 = (1/3) * 4 + a') ∧ (4 = (1/3) * 2 + b') → a' + b' = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l3303_330362


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3303_330325

/-- A hyperbola with foci on the x-axis, passing through (4√2, -3), and having perpendicular lines
    connecting (0, 5) to its foci, has the standard equation x²/16 - y²/9 = 1. -/
theorem hyperbola_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c^2 = a^2 + b^2) : 
  (32 / a^2 - 9 / b^2 = 1) → (25 / c^2 = 1) → (a = 4 ∧ b = 3) := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_hyperbola_equation_l3303_330325


namespace NUMINAMATH_CALUDE_marys_investment_l3303_330369

theorem marys_investment (mary_investment : ℕ) (mike_investment : ℕ) (total_profit : ℕ) : 
  mike_investment = 400 →
  total_profit = 7500 →
  let equal_share := total_profit / 3 / 2
  let remaining_profit := total_profit - total_profit / 3
  let mary_share := equal_share + remaining_profit * mary_investment / (mary_investment + mike_investment)
  let mike_share := equal_share + remaining_profit * mike_investment / (mary_investment + mike_investment)
  mary_share = mike_share + 1000 →
  mary_investment = 600 :=
by sorry

end NUMINAMATH_CALUDE_marys_investment_l3303_330369


namespace NUMINAMATH_CALUDE_min_value_xy_plus_x_squared_l3303_330310

theorem min_value_xy_plus_x_squared (x y : ℝ) (h1 : x * y > 0) (h2 : x^2 * y = 2) :
  x * y + x^2 ≥ 4 ∧ (x * y + x^2 = 4 ↔ y = 1 ∧ x = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_xy_plus_x_squared_l3303_330310


namespace NUMINAMATH_CALUDE_y_elimination_condition_l3303_330396

/-- Given a system of linear equations 6x + my = 3 and 2x - ny = -6,
    y is directly eliminated by subtracting the second equation from the first
    if and only if m + n = 0 -/
theorem y_elimination_condition (m n : ℝ) : 
  (∀ x y : ℝ, 6 * x + m * y = 3 ∧ 2 * x - n * y = -6) →
  (∃! x : ℝ, ∀ y : ℝ, 6 * x + m * y = 3 ∧ 2 * x - n * y = -6) ↔
  m + n = 0 := by
sorry

end NUMINAMATH_CALUDE_y_elimination_condition_l3303_330396


namespace NUMINAMATH_CALUDE_total_votes_polled_l3303_330303

/-- Represents the total number of votes polled in an election --/
def total_votes : ℕ := sorry

/-- Represents the number of valid votes received by candidate B --/
def votes_B : ℕ := 2509

/-- Theorem stating the total number of votes polled in the election --/
theorem total_votes_polled :
  (total_votes : ℚ) * (80 : ℚ) / 100 = 
    (votes_B : ℚ) + (votes_B : ℚ) + (total_votes : ℚ) * (15 : ℚ) / 100 ∧
  total_votes = 7720 :=
sorry

end NUMINAMATH_CALUDE_total_votes_polled_l3303_330303


namespace NUMINAMATH_CALUDE_inscribed_circle_height_difference_l3303_330364

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x^2 + 4 * x

/-- Represents a circle inscribed in the parabola -/
structure InscribedCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangentPoint : ℝ
  isTangent : (tangentPoint, parabola tangentPoint) ∈ frontier {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2}

/-- The height difference between the circle's center and a tangent point -/
def heightDifference (circle : InscribedCircle) : ℝ :=
  circle.center.2 - parabola circle.tangentPoint

theorem inscribed_circle_height_difference (circle : InscribedCircle) :
  heightDifference circle = -2 * circle.tangentPoint^2 - 4 * circle.tangentPoint + 2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_height_difference_l3303_330364


namespace NUMINAMATH_CALUDE_special_triangle_existence_l3303_330378

/-- A triangle with integer side lengths satisfying a special condition -/
def SpecialTriangle (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive integers
  a + b > c ∧ b + c > a ∧ a + c > b ∧  -- Triangle inequality
  a * b * c = 2 * (a - 1) * (b - 1) * (c - 1)  -- Special condition

theorem special_triangle_existence :
  ∃ a b c : ℕ, SpecialTriangle a b c ∧
  (∀ x y z : ℕ, SpecialTriangle x y z → (x, y, z) = (8, 7, 3) ∨ (x, y, z) = (6, 5, 4)) :=
by sorry


end NUMINAMATH_CALUDE_special_triangle_existence_l3303_330378


namespace NUMINAMATH_CALUDE_emily_fish_weight_l3303_330306

/-- Calculates the total weight of fish caught by Emily -/
def total_fish_weight (trout_count catfish_count bluegill_count : ℕ)
                      (trout_weight catfish_weight bluegill_weight : ℝ) : ℝ :=
  (trout_count : ℝ) * trout_weight +
  (catfish_count : ℝ) * catfish_weight +
  (bluegill_count : ℝ) * bluegill_weight

/-- Proves that Emily caught 25 pounds of fish -/
theorem emily_fish_weight :
  total_fish_weight 4 3 5 2 1.5 2.5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_emily_fish_weight_l3303_330306


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3303_330304

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ = 2 and a₂ + a₃ = 13,
    prove that a₄ + a₅ + a₆ = 42 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h_arith : IsArithmeticSequence a)
    (h_a1 : a 1 = 2)
    (h_a2_a3 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3303_330304


namespace NUMINAMATH_CALUDE_subtraction_problem_l3303_330382

theorem subtraction_problem : 4444444444444 - 2222222222222 - 444444444444 = 1777777777778 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3303_330382


namespace NUMINAMATH_CALUDE_book_pages_count_l3303_330307

/-- The number of pages Liam read in a week-long reading assignment -/
def totalPages (firstThreeDaysAvg : ℕ) (nextThreeDaysAvg : ℕ) (lastDayPages : ℕ) : ℕ :=
  3 * firstThreeDaysAvg + 3 * nextThreeDaysAvg + lastDayPages

/-- Theorem stating that the total number of pages in the book is 310 -/
theorem book_pages_count :
  totalPages 45 50 25 = 310 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l3303_330307


namespace NUMINAMATH_CALUDE_joan_picked_43_apples_l3303_330386

/-- The number of apples Joan picked from the orchard -/
def apples_picked : ℕ := sorry

/-- The number of apples Joan gave to Melanie -/
def apples_given : ℕ := 27

/-- The number of apples Joan has left -/
def apples_left : ℕ := 16

/-- Theorem stating that Joan picked 43 apples from the orchard -/
theorem joan_picked_43_apples : apples_picked = 43 := by
  sorry

end NUMINAMATH_CALUDE_joan_picked_43_apples_l3303_330386


namespace NUMINAMATH_CALUDE_distance_A_to_yoz_l3303_330345

/-- The distance from a point to the yoz plane is the absolute value of its x-coordinate. -/
def distance_to_yoz (p : ℝ × ℝ × ℝ) : ℝ :=
  |p.1|

/-- Point A with coordinates (-3, 1, -4) -/
def A : ℝ × ℝ × ℝ := (-3, 1, -4)

/-- Theorem: The distance from point A to the yoz plane is 3 -/
theorem distance_A_to_yoz : distance_to_yoz A = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_to_yoz_l3303_330345


namespace NUMINAMATH_CALUDE_systematic_sampling_largest_number_l3303_330391

/-- Systematic sampling theorem for class selection -/
theorem systematic_sampling_largest_number
  (total_classes : ℕ)
  (selected_classes : ℕ)
  (smallest_number : ℕ)
  (h1 : total_classes = 24)
  (h2 : selected_classes = 4)
  (h3 : smallest_number = 3)
  (h4 : smallest_number > 0)
  (h5 : smallest_number ≤ total_classes)
  (h6 : selected_classes > 0)
  (h7 : selected_classes ≤ total_classes) :
  ∃ (largest_number : ℕ),
    largest_number = 21 ∧
    largest_number ≤ total_classes ∧
    (largest_number - smallest_number) = (selected_classes - 1) * (total_classes / selected_classes) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_largest_number_l3303_330391


namespace NUMINAMATH_CALUDE_floor_plus_self_equals_ten_point_three_l3303_330319

theorem floor_plus_self_equals_ten_point_three (r : ℝ) :
  (⌊r⌋ : ℝ) + r = 10.3 → r = 5.3 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_equals_ten_point_three_l3303_330319


namespace NUMINAMATH_CALUDE_no_rain_percentage_l3303_330346

theorem no_rain_percentage (p_monday : ℝ) (p_tuesday : ℝ) (p_both : ℝ) 
  (h_monday : p_monday = 0.7)
  (h_tuesday : p_tuesday = 0.55)
  (h_both : p_both = 0.6) :
  1 - (p_monday + p_tuesday - p_both) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_percentage_l3303_330346


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_l3303_330327

theorem fibonacci_like_sequence (a : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n)
  (h2 : a 11 = 157) :
  a 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_l3303_330327


namespace NUMINAMATH_CALUDE_max_value_theorem_l3303_330338

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  x + Real.sqrt (x * y) + (x * y * z) ^ (1/4) ≤ 7/6 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3303_330338


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l3303_330301

/-- The number of pairs of shoes in the box -/
def num_pairs : ℕ := 5

/-- The total number of shoes in the box -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of ways to select 2 shoes out of the total -/
def total_selections : ℕ := (total_shoes.choose 2)

/-- The number of ways to select a matching pair -/
def matching_selections : ℕ := num_pairs

/-- The probability of selecting a matching pair -/
def probability_matching : ℚ := matching_selections / total_selections

theorem matching_shoes_probability : probability_matching = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l3303_330301


namespace NUMINAMATH_CALUDE_rosa_bonheur_birthday_l3303_330351

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the number of leap years between two years -/
def leapYearCount (startYear endYear : Nat) : Nat :=
  let totalYears := endYear - startYear
  let potentialLeapYears := totalYears / 4
  potentialLeapYears - 1 -- Excluding 1900

/-- Calculates the day of the week given a starting day and number of days passed -/
def calculateDay (startDay : DayOfWeek) (daysPassed : Nat) : DayOfWeek :=
  match (daysPassed % 7) with
  | 0 => startDay
  | 1 => DayOfWeek.Sunday
  | 2 => DayOfWeek.Monday
  | 3 => DayOfWeek.Tuesday
  | 4 => DayOfWeek.Wednesday
  | 5 => DayOfWeek.Thursday
  | 6 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem rosa_bonheur_birthday 
  (anniversaryDay : DayOfWeek)
  (h : anniversaryDay = DayOfWeek.Wednesday) :
  calculateDay anniversaryDay 261 = DayOfWeek.Sunday := by
  sorry

#check rosa_bonheur_birthday

end NUMINAMATH_CALUDE_rosa_bonheur_birthday_l3303_330351


namespace NUMINAMATH_CALUDE_solution_set_theorem_l3303_330380

/-- A differentiable function satisfying the given condition -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ 
  (∀ x > 0, x * (deriv f x) + 2 * f x > 0)

/-- The solution set of the inequality -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | (x + 2016) * f (x + 2016) / 5 < 5 * f 5 / (x + 2016)}

/-- The theorem stating the solution set of the inequality -/
theorem solution_set_theorem (f : ℝ → ℝ) (hf : SatisfyingFunction f) :
  SolutionSet f = {x | -2016 < x ∧ x < -2011} :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l3303_330380


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3303_330321

/-- A line passing through point A(1,2) with equal intercepts on both coordinate axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through point A(1,2) -/
  passes_through_A : m * 1 + b = 2
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b = b / m

/-- The equation of an EqualInterceptLine is either 2x - y = 0 or x + y = 3 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 2 ∧ l.b = 0) ∨ (l.m = 1 ∧ l.b = 2) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3303_330321


namespace NUMINAMATH_CALUDE_estimated_probability_is_correct_l3303_330355

/-- Represents the result of a single trial in the traffic congestion simulation -/
structure TrialResult :=
  (days_with_congestion : Nat)
  (h_valid : days_with_congestion ≤ 3)

/-- The simulation data -/
def simulation_data : Finset TrialResult := sorry

/-- The total number of trials in the simulation -/
def total_trials : Nat := 20

/-- The number of trials with exactly two days of congestion -/
def trials_with_two_congestion : Nat := 5

/-- The estimated probability of having exactly two days of congestion in three days -/
def estimated_probability : ℚ := trials_with_two_congestion / total_trials

theorem estimated_probability_is_correct :
  estimated_probability = 1/4 := by sorry

end NUMINAMATH_CALUDE_estimated_probability_is_correct_l3303_330355


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_powers_of_three_l3303_330387

def powers_of_three : List ℕ := [3, 9, 27, 81, 243, 729, 2187, 6561, 19683]

theorem arithmetic_mean_of_powers_of_three :
  (List.sum powers_of_three) / powers_of_three.length = 2970 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_powers_of_three_l3303_330387


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3303_330330

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 24 ∧ ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  ∀ k : ℕ, k ≤ 24 → (k ∣ (n * (n + 1) * (n + 2) * (n + 3))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3303_330330


namespace NUMINAMATH_CALUDE_marie_erasers_l3303_330312

theorem marie_erasers (initial : ℕ) (lost : ℕ) (final : ℕ) : 
  initial = 95 → lost = 42 → final = initial - lost → final = 53 := by sorry

end NUMINAMATH_CALUDE_marie_erasers_l3303_330312


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3303_330388

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality : 
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3303_330388


namespace NUMINAMATH_CALUDE_min_m_plus_n_l3303_330348

noncomputable def f (m n x : ℝ) : ℝ := Real.log x - 2 * m * x^2 - n

theorem min_m_plus_n (m n : ℝ) :
  (∀ x > 0, f m n x ≤ -Real.log 2) →
  (∃ x > 0, f m n x = -Real.log 2) →
  m + n ≥ (1/2) * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_min_m_plus_n_l3303_330348


namespace NUMINAMATH_CALUDE_balloon_distribution_l3303_330368

theorem balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) (take_back : ℕ) 
  (h1 : total_balloons = 250)
  (h2 : num_friends = 5)
  (h3 : take_back = 11) :
  (total_balloons / num_friends) - take_back = 39 := by
  sorry

end NUMINAMATH_CALUDE_balloon_distribution_l3303_330368


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l3303_330376

theorem smallest_n_multiple_of_seven (x y : ℤ) 
  (hx : (x - 4) % 7 = 0) 
  (hy : (y + 4) % 7 = 0) : 
  (∃ n : ℕ+, (x^2 - x*y + y^2 + n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 - x*y + y^2 + m) % 7 = 0 → n ≤ m) → 
  (∃ n : ℕ+, (x^2 - x*y + y^2 + n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 - x*y + y^2 + m) % 7 = 0 → n ≤ m) ∧ 
  (∀ n : ℕ+, (x^2 - x*y + y^2 + n) % 7 = 0 ∧ 
    (∀ m : ℕ+, (x^2 - x*y + y^2 + m) % 7 = 0 → n ≤ m) → n = 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l3303_330376


namespace NUMINAMATH_CALUDE_second_number_in_first_set_l3303_330398

theorem second_number_in_first_set (X : ℝ) : 
  ((20 + X + 60) / 3 = (10 + 50 + 45) / 3 + 5) → X = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_number_in_first_set_l3303_330398


namespace NUMINAMATH_CALUDE_systematic_sample_sum_l3303_330360

/-- Systematic sampling function that returns the nth element in a sample of size k from a population of size n -/
def systematicSample (n k : ℕ) (i : ℕ) : ℕ :=
  i * (n / k) + 1

theorem systematic_sample_sum (a b : ℕ) :
  systematicSample 60 5 0 = 4 ∧
  systematicSample 60 5 1 = a ∧
  systematicSample 60 5 2 = 28 ∧
  systematicSample 60 5 3 = b ∧
  systematicSample 60 5 4 = 52 →
  a + b = 56 := by
sorry

end NUMINAMATH_CALUDE_systematic_sample_sum_l3303_330360


namespace NUMINAMATH_CALUDE_sachin_age_is_28_l3303_330305

/-- The age of Sachin -/
def sachin_age : ℕ := sorry

/-- The age of Rahul -/
def rahul_age : ℕ := sorry

/-- Rahul is 8 years older than Sachin -/
axiom age_difference : rahul_age = sachin_age + 8

/-- The ratio of Sachin's age to Rahul's age is 7:9 -/
axiom age_ratio : (sachin_age : ℚ) / rahul_age = 7 / 9

/-- Sachin's age is 28 years -/
theorem sachin_age_is_28 : sachin_age = 28 := by sorry

end NUMINAMATH_CALUDE_sachin_age_is_28_l3303_330305


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l3303_330354

theorem solve_quadratic_equation (m n : ℤ) 
  (h : m^2 - 2*m*n + 2*n^2 - 8*n + 16 = 0) : 
  m = 4 ∧ n = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l3303_330354


namespace NUMINAMATH_CALUDE_min_value_function_max_value_constraint_l3303_330333

-- Problem 1
theorem min_value_function (x : ℝ) (h : x > 1/2) :
  (∀ z, z > 1/2 → 2*z + 4/(2*z - 1) ≥ 2*x + 4/(2*x - 1)) →
  2*x + 4/(2*x - 1) = 5 :=
sorry

-- Problem 2
theorem max_value_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (∀ z w : ℝ, z > 0 → w > 0 → z + w = 3 → x*y + 2*x + y ≥ z*w + 2*z + w) →
  x*y + 2*x + y = 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_function_max_value_constraint_l3303_330333


namespace NUMINAMATH_CALUDE_skittles_division_l3303_330393

theorem skittles_division (total_skittles : Nat) (num_groups : Nat) (group_size : Nat) :
  total_skittles = 5929 →
  num_groups = 77 →
  total_skittles = num_groups * group_size →
  group_size = 77 := by
sorry

end NUMINAMATH_CALUDE_skittles_division_l3303_330393


namespace NUMINAMATH_CALUDE_square_congruent_one_iff_one_or_minus_one_l3303_330357

theorem square_congruent_one_iff_one_or_minus_one (p : Nat) (hp : Prime p) :
  ∀ a : Nat, a^2 ≡ 1 [ZMOD p] ↔ a ≡ 1 [ZMOD p] ∨ a ≡ p - 1 [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_square_congruent_one_iff_one_or_minus_one_l3303_330357


namespace NUMINAMATH_CALUDE_base_4_last_digit_l3303_330336

theorem base_4_last_digit (n : ℕ) (h : n = 389) : n % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_4_last_digit_l3303_330336


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3303_330353

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 6 ∧ (7538 - x) % 14 = 0 ∧ ∀ (y : ℕ), y < x → (7538 - y) % 14 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3303_330353


namespace NUMINAMATH_CALUDE_resistance_ae_l3303_330385

structure WireConstruction where
  -- Square ACEG
  side_length : ℝ
  -- Resistance of segment AB
  R₀ : ℝ
  -- Homogeneous wire assumption
  is_homogeneous : Prop
  -- Constant cross-section assumption
  constant_cross_section : Prop
  -- B, D, F, H are midpoints
  are_midpoints : Prop
  -- R₀ = 1 Ω
  r0_value : R₀ = 1

def resistance_between_points (w : WireConstruction) (point1 : String) (point2 : String) : ℝ :=
  sorry

theorem resistance_ae (w : WireConstruction) :
  resistance_between_points w "A" "E" = (4 + 2 * Real.sqrt 2) / (3 + 3 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_resistance_ae_l3303_330385


namespace NUMINAMATH_CALUDE_telescope_visual_range_l3303_330337

/-- Given a telescope that increases visual range by 87.5% to 150 km, 
    prove the initial visual range is 80 km. -/
theorem telescope_visual_range : 
  ∀ (initial_range : ℝ), 
    initial_range + 0.875 * initial_range = 150 → 
    initial_range = 80 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_l3303_330337


namespace NUMINAMATH_CALUDE_parabola_intercept_sum_l3303_330334

theorem parabola_intercept_sum : ∃ (a b c : ℝ),
  (∀ y : ℝ, 3 * y^2 - 9 * y + 5 = a ↔ y = 0) ∧
  (3 * b^2 - 9 * b + 5 = 0) ∧
  (3 * c^2 - 9 * c + 5 = 0) ∧
  (b ≠ c) ∧
  (a + b + c = 8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intercept_sum_l3303_330334

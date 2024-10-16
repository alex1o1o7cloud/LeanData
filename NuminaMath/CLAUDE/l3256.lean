import Mathlib

namespace NUMINAMATH_CALUDE_intersection_angle_cosine_l3256_325621

/-- The cosine of the angle formed by the foci and an intersection point of an ellipse and hyperbola with common foci -/
theorem intersection_angle_cosine 
  (x y : ℝ) 
  (ellipse_eq : x^2/6 + y^2/2 = 1) 
  (hyperbola_eq : x^2/3 - y^2 = 1) 
  (is_intersection : x^2/6 + y^2/2 = 1 ∧ x^2/3 - y^2 = 1) : 
  ∃ (f₁_x f₁_y f₂_x f₂_y : ℝ), 
    let f₁ := (f₁_x, f₁_y)
    let f₂ := (f₂_x, f₂_y)
    let p := (x, y)
    let v₁ := (x - f₁_x, y - f₁_y)
    let v₂ := (x - f₂_x, y - f₂_y)
    (f₁.1^2/6 + f₁.2^2/2 < 1 ∧ f₂.1^2/6 + f₂.2^2/2 < 1) ∧  -- f₁ and f₂ are inside the ellipse
    (f₁.1^2/3 - f₁.2^2 > 1 ∧ f₂.1^2/3 - f₂.2^2 > 1) ∧      -- f₁ and f₂ are outside the hyperbola
    (v₁.1 * v₂.1 + v₁.2 * v₂.2) / 
    (Real.sqrt (v₁.1^2 + v₁.2^2) * Real.sqrt (v₂.1^2 + v₂.2^2)) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_angle_cosine_l3256_325621


namespace NUMINAMATH_CALUDE_sara_lunch_cost_l3256_325641

/-- The cost of Sara's lunch given the prices of a hotdog and a salad -/
def lunch_cost (hotdog_price salad_price : ℚ) : ℚ :=
  hotdog_price + salad_price

/-- Theorem stating that Sara's lunch cost $10.46 given the prices of the hotdog and salad -/
theorem sara_lunch_cost :
  lunch_cost (536/100) (51/10) = 1046/100 := by
  sorry

end NUMINAMATH_CALUDE_sara_lunch_cost_l3256_325641


namespace NUMINAMATH_CALUDE_trig_fraction_value_l3256_325685

theorem trig_fraction_value (θ : Real) (h : Real.sin θ + 2 * Real.cos θ = 1) :
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = -7 ∨
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_value_l3256_325685


namespace NUMINAMATH_CALUDE_negative_inequality_l3256_325609

theorem negative_inequality (x y : ℝ) (h : x < y) : -x > -y := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l3256_325609


namespace NUMINAMATH_CALUDE_at_least_one_negative_l3256_325603

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_eq_one : a + b = 1 ∧ c + d = 1) 
  (product_gt_one : a * c + b * d > 1) : 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l3256_325603


namespace NUMINAMATH_CALUDE_leap_day_2024_is_sunday_l3256_325679

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Calculates the day of the week for a given number of days after a Sunday -/
def dayAfterSunday (days : Nat) : DayOfWeek :=
  match days % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

/-- The number of days between February 29, 2000, and February 29, 2024 -/
def daysBetween2000And2024 : Nat := 8766

theorem leap_day_2024_is_sunday :
  dayAfterSunday daysBetween2000And2024 = DayOfWeek.Sunday := by
  sorry

#check leap_day_2024_is_sunday

end NUMINAMATH_CALUDE_leap_day_2024_is_sunday_l3256_325679


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3256_325681

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3256_325681


namespace NUMINAMATH_CALUDE_lcm_problem_l3256_325633

theorem lcm_problem (a b c : ℕ+) (ha : a = 72) (hb : b = 108) (hlcm : Nat.lcm (Nat.lcm a b) c = 37800) : c = 175 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3256_325633


namespace NUMINAMATH_CALUDE_riding_mower_rate_riding_mower_rate_is_two_l3256_325676

theorem riding_mower_rate (total_area : ℝ) (riding_mower_fraction : ℝ) 
  (push_mower_rate : ℝ) (total_time : ℝ) : ℝ :=
by
  -- Define the conditions
  have h1 : total_area = 8 := by sorry
  have h2 : riding_mower_fraction = 3/4 := by sorry
  have h3 : push_mower_rate = 1 := by sorry
  have h4 : total_time = 5 := by sorry

  -- Calculate the area mowed by each mower
  let riding_mower_area := total_area * riding_mower_fraction
  let push_mower_area := total_area * (1 - riding_mower_fraction)

  -- Calculate the time spent with the push mower
  let push_mower_time := push_mower_area / push_mower_rate

  -- Calculate the time spent with the riding mower
  let riding_mower_time := total_time - push_mower_time

  -- Calculate and return the riding mower rate
  exact riding_mower_area / riding_mower_time
  
-- The theorem statement proves that the riding mower rate is 2 acres per hour
theorem riding_mower_rate_is_two : 
  riding_mower_rate 8 (3/4) 1 5 = 2 := by sorry

end NUMINAMATH_CALUDE_riding_mower_rate_riding_mower_rate_is_two_l3256_325676


namespace NUMINAMATH_CALUDE_inequality_proof_l3256_325639

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  b * c^2 + c * a^2 + a * b^2 < b^2 * c + c^2 * a + a^2 * b := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3256_325639


namespace NUMINAMATH_CALUDE_optimal_plan_is_correct_l3256_325643

/-- Represents the number of cars a worker can install per month -/
structure WorkerProductivity where
  skilled : ℕ
  new : ℕ

/-- Represents the monthly salary of workers -/
structure WorkerSalary where
  skilled : ℕ
  new : ℕ

/-- Represents a recruitment plan -/
structure RecruitmentPlan where
  skilled : ℕ
  new : ℕ

def optimal_plan (prod : WorkerProductivity) (salary : WorkerSalary) : RecruitmentPlan :=
  sorry

theorem optimal_plan_is_correct (prod : WorkerProductivity) (salary : WorkerSalary) :
  let plan := optimal_plan prod salary
  prod.skilled * plan.skilled + prod.new * plan.new = 20 ∧
  ∀ other : RecruitmentPlan,
    prod.skilled * other.skilled + prod.new * other.new = 20 →
    salary.skilled * plan.skilled + salary.new * plan.new ≤
    salary.skilled * other.skilled + salary.new * other.new :=
by
  sorry

#check @optimal_plan_is_correct

end NUMINAMATH_CALUDE_optimal_plan_is_correct_l3256_325643


namespace NUMINAMATH_CALUDE_janets_dress_pockets_janets_dress_pockets_correct_l3256_325671

theorem janets_dress_pockets : ℕ → ℕ
  | total_dresses =>
    let dresses_with_pockets := total_dresses / 2
    let dresses_with_two_pockets := dresses_with_pockets / 3
    let dresses_with_three_pockets := dresses_with_pockets - dresses_with_two_pockets
    let total_pockets := dresses_with_two_pockets * 2 + dresses_with_three_pockets * 3
    total_pockets

theorem janets_dress_pockets_correct : janets_dress_pockets 24 = 32 := by
  sorry

end NUMINAMATH_CALUDE_janets_dress_pockets_janets_dress_pockets_correct_l3256_325671


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3256_325655

theorem max_value_of_expression (x : ℝ) :
  ∃ (max_x : ℝ), ∀ y, 1 - (y + 5)^2 ≤ 1 - (max_x + 5)^2 ∧ max_x = -5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3256_325655


namespace NUMINAMATH_CALUDE_floor_expression_equals_eight_l3256_325608

theorem floor_expression_equals_eight :
  ⌊(2021^3 : ℝ) / (2019 * 2020) - (2019^3 : ℝ) / (2020 * 2021)⌋ = 8 := by
  sorry

#check floor_expression_equals_eight

end NUMINAMATH_CALUDE_floor_expression_equals_eight_l3256_325608


namespace NUMINAMATH_CALUDE_wolf_tail_growth_l3256_325612

theorem wolf_tail_growth (x y : ℕ) : 1 * 2^x * 3^y = 864 ↔ x = 5 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_wolf_tail_growth_l3256_325612


namespace NUMINAMATH_CALUDE_chocolate_boxes_total_l3256_325648

/-- The total number of chocolate pieces in multiple boxes -/
def total_pieces (boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  boxes * pieces_per_box

/-- Theorem: The total number of chocolate pieces in 6 boxes with 500 pieces each is 3000 -/
theorem chocolate_boxes_total :
  total_pieces 6 500 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_boxes_total_l3256_325648


namespace NUMINAMATH_CALUDE_chemistry_class_grades_l3256_325687

theorem chemistry_class_grades (total_students : ℕ) 
  (prob_A prob_B prob_C prob_F : ℝ) : 
  total_students = 50 →
  prob_A = 0.6 * prob_B →
  prob_C = 1.5 * prob_B →
  prob_F = 0.4 * prob_B →
  prob_A + prob_B + prob_C + prob_F = 1 →
  ⌊total_students * prob_B⌋ = 14 := by
sorry

end NUMINAMATH_CALUDE_chemistry_class_grades_l3256_325687


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3256_325697

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 2 + a 3 = 32) →
  (a 11 + a 12 + a 13 = 118) →
  a 4 + a 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3256_325697


namespace NUMINAMATH_CALUDE_difference_of_squares_l3256_325677

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3256_325677


namespace NUMINAMATH_CALUDE_only_physical_education_survey_census_suitable_physical_education_survey_census_suitable_light_bulb_survey_not_census_suitable_tv_program_survey_not_census_suitable_national_height_survey_not_census_suitable_l3256_325615

/-- Represents a survey --/
inductive Survey
  | PhysicalEducationScores
  | LightBulbLifespan
  | TVProgramPreferences
  | NationalStudentHeight

/-- Checks if a survey satisfies the conditions for a census --/
def isCensusSuitable (s : Survey) : Prop :=
  match s with
  | Survey.PhysicalEducationScores => true
  | _ => false

/-- Theorem stating that only the physical education scores survey is suitable for a census --/
theorem only_physical_education_survey_census_suitable :
  ∀ s : Survey, isCensusSuitable s ↔ s = Survey.PhysicalEducationScores :=
by sorry

/-- Proof that the physical education scores survey is suitable for a census --/
theorem physical_education_survey_census_suitable :
  isCensusSuitable Survey.PhysicalEducationScores :=
by sorry

/-- Proof that the light bulb lifespan survey is not suitable for a census --/
theorem light_bulb_survey_not_census_suitable :
  ¬ isCensusSuitable Survey.LightBulbLifespan :=
by sorry

/-- Proof that the TV program preferences survey is not suitable for a census --/
theorem tv_program_survey_not_census_suitable :
  ¬ isCensusSuitable Survey.TVProgramPreferences :=
by sorry

/-- Proof that the national student height survey is not suitable for a census --/
theorem national_height_survey_not_census_suitable :
  ¬ isCensusSuitable Survey.NationalStudentHeight :=
by sorry

end NUMINAMATH_CALUDE_only_physical_education_survey_census_suitable_physical_education_survey_census_suitable_light_bulb_survey_not_census_suitable_tv_program_survey_not_census_suitable_national_height_survey_not_census_suitable_l3256_325615


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_application_l3256_325632

theorem chinese_remainder_theorem_application : 
  ∀ x : ℕ, 1000 < x ∧ x < 4000 ∧ 
    x % 11 = 2 ∧ x % 13 = 12 ∧ x % 19 = 18 ↔ 
    x = 1234 ∨ x = 3951 := by sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_application_l3256_325632


namespace NUMINAMATH_CALUDE_debt_installments_l3256_325610

theorem debt_installments (first_payment : ℕ) (additional_amount : ℕ) (average_payment : ℕ) : 
  let n := (12 * first_payment + 780) / 15
  let remaining_payment := first_payment + additional_amount
  12 * first_payment + (n - 12) * remaining_payment = n * average_payment →
  n = 52 :=
by
  sorry

#check debt_installments 410 65 460

end NUMINAMATH_CALUDE_debt_installments_l3256_325610


namespace NUMINAMATH_CALUDE_tim_bodyguard_cost_l3256_325654

/-- Calculates the total weekly cost for hiring bodyguards -/
def total_weekly_cost (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  num_bodyguards * hourly_rate * hours_per_day * days_per_week

/-- Proves that the total weekly cost for Tim's bodyguards is $2240 -/
theorem tim_bodyguard_cost :
  total_weekly_cost 2 20 8 7 = 2240 := by
  sorry

end NUMINAMATH_CALUDE_tim_bodyguard_cost_l3256_325654


namespace NUMINAMATH_CALUDE_fraction_difference_simplest_form_l3256_325659

theorem fraction_difference_simplest_form :
  let a := 5
  let b := 19
  let c := 2
  let d := 23
  let numerator := a * d - c * b
  let denominator := b * d
  (numerator : ℚ) / denominator = 77 / 437 ∧
  ∀ (x y : ℤ), x ≠ 0 → (77 : ℚ) / 437 = (x : ℚ) / y → (x = 77 ∧ y = 437 ∨ x = -77 ∧ y = -437) :=
by sorry

end NUMINAMATH_CALUDE_fraction_difference_simplest_form_l3256_325659


namespace NUMINAMATH_CALUDE_bill_face_value_l3256_325690

/-- Calculates the face value of a bill given the true discount, interest rate, and time until due. -/
def face_value (true_discount : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  true_discount * (1 + interest_rate * time)

/-- Theorem: The face value of a bill with a true discount of 210, interest rate of 16% per annum, 
    and due in 9 months is 235.20. -/
theorem bill_face_value : 
  face_value 210 0.16 (9 / 12) = 235.20 := by
  sorry

end NUMINAMATH_CALUDE_bill_face_value_l3256_325690


namespace NUMINAMATH_CALUDE_sailboat_rental_cost_l3256_325663

/-- The cost to rent a sailboat for 3 hours a day over 2 days -/
def sailboat_cost : ℝ := sorry

/-- The cost per hour to rent a ski boat -/
def ski_boat_cost_per_hour : ℝ := 80

/-- The number of hours per day the boats were rented -/
def hours_per_day : ℕ := 3

/-- The number of days the boats were rented -/
def days_rented : ℕ := 2

/-- The additional cost Aldrich paid for the ski boat compared to Ken's sailboat -/
def additional_cost : ℝ := 120

theorem sailboat_rental_cost :
  sailboat_cost = 360 :=
by
  have ski_boat_total_cost : ℝ := ski_boat_cost_per_hour * (hours_per_day * days_rented)
  have h1 : ski_boat_total_cost = sailboat_cost + additional_cost := by sorry
  sorry

end NUMINAMATH_CALUDE_sailboat_rental_cost_l3256_325663


namespace NUMINAMATH_CALUDE_stacy_growth_difference_l3256_325696

/-- Calculates the difference in growth between Stacy and her brother -/
def growth_difference (stacy_initial_height stacy_final_height brother_growth : ℕ) : ℕ :=
  (stacy_final_height - stacy_initial_height) - brother_growth

/-- Proves that the difference in growth between Stacy and her brother is 6 inches -/
theorem stacy_growth_difference :
  growth_difference 50 57 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_stacy_growth_difference_l3256_325696


namespace NUMINAMATH_CALUDE_centroid_line_intersection_l3256_325651

/-- Given a triangle ABC with centroid G, and a line through G intersecting AB at M and AC at N,
    where AM = x * AB and AN = y * AC, prove that 1/x + 1/y = 3 -/
theorem centroid_line_intersection (A B C G M N : ℝ × ℝ) (x y : ℝ) :
  (G = (1/3 : ℝ) • (A + B + C)) →  -- G is the centroid
  (∃ (t : ℝ), M = A + t • (G - A) ∧ N = A + t • (G - A)) →  -- M and N are on the line through G
  (M = A + x • (B - A)) →  -- AM = x * AB
  (N = A + y • (C - A)) →  -- AN = y * AC
  (1 / x + 1 / y = 3) :=
by sorry

end NUMINAMATH_CALUDE_centroid_line_intersection_l3256_325651


namespace NUMINAMATH_CALUDE_employee_salary_l3256_325675

theorem employee_salary (total_salary : ℝ) (m_percentage : ℝ) (n_salary : ℝ) : 
  total_salary = 616 →
  m_percentage = 1.20 →
  n_salary + m_percentage * n_salary = total_salary →
  n_salary = 280 := by
sorry

end NUMINAMATH_CALUDE_employee_salary_l3256_325675


namespace NUMINAMATH_CALUDE_frog_corner_probability_l3256_325600

/-- Represents a position on the 4x4 grid -/
inductive Position
| Corner
| Edge
| Middle

/-- Represents the state of the frog's movement -/
structure FrogState where
  position : Position
  hops : Nat

/-- Transition function for the frog's movement -/
def transition (state : FrogState) : FrogState :=
  sorry

/-- Probability of reaching a corner from a given state -/
def cornerProbability (state : FrogState) : Rat :=
  sorry

/-- The starting state of the frog -/
def initialState : FrogState :=
  { position := Position.Edge, hops := 0 }

/-- Main theorem: Probability of reaching a corner within 4 hops -/
theorem frog_corner_probability :
  cornerProbability { position := initialState.position, hops := 4 } = 35 / 64 := by
  sorry

end NUMINAMATH_CALUDE_frog_corner_probability_l3256_325600


namespace NUMINAMATH_CALUDE_andy_ate_six_cookies_six_is_max_cookies_for_andy_l3256_325613

/-- Represents the number of cookies eaten by Andy -/
def andys_cookies : ℕ := sorry

/-- The total number of cookies baked -/
def total_cookies : ℕ := 36

/-- Theorem stating that Andy ate 6 cookies, given the problem conditions -/
theorem andy_ate_six_cookies : andys_cookies = 6 := by
  have h1 : andys_cookies + 2 * andys_cookies + 3 * andys_cookies = total_cookies := sorry
  have h2 : andys_cookies ≤ 6 := sorry
  sorry

/-- Theorem proving that 6 is the maximum number of cookies Andy could have eaten -/
theorem six_is_max_cookies_for_andy :
  ∀ n : ℕ, n > 6 → n + 2 * n + 3 * n > total_cookies := by
  sorry

end NUMINAMATH_CALUDE_andy_ate_six_cookies_six_is_max_cookies_for_andy_l3256_325613


namespace NUMINAMATH_CALUDE_circle_equation_minus_one_two_radius_two_l3256_325698

/-- The standard equation of a circle with center (h, k) and radius r -/
def standard_circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The standard equation of a circle with center (-1, 2) and radius 2 -/
theorem circle_equation_minus_one_two_radius_two :
  ∀ x y : ℝ, standard_circle_equation x y (-1) 2 2 ↔ (x + 1)^2 + (y - 2)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_minus_one_two_radius_two_l3256_325698


namespace NUMINAMATH_CALUDE_arithmetic_sequence_4_to_256_l3256_325634

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- The last term of an arithmetic sequence -/
def arithmetic_sequence_last_term (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_4_to_256 :
  arithmetic_sequence_length 4 4 256 = 64 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_4_to_256_l3256_325634


namespace NUMINAMATH_CALUDE_social_gathering_handshakes_l3256_325631

/-- Represents a social gathering with specific conditions -/
structure SocialGathering where
  total_people : Nat
  group_a_size : Nat
  group_b_size : Nat
  group_b_connected : Nat
  group_b_isolated : Nat
  a_to_b_connections : Nat
  a_to_b_per_person : Nat

/-- Calculates the number of handshakes in the social gathering -/
def count_handshakes (g : SocialGathering) : Nat :=
  let a_to_b_handshakes := (g.group_a_size - g.a_to_b_connections) * g.group_b_size
  let b_connected_handshakes := g.group_b_connected * (g.group_b_connected - 1) / 2
  let b_isolated_handshakes := g.group_b_isolated * (g.group_b_isolated - 1) / 2 + g.group_b_isolated * g.group_b_connected
  a_to_b_handshakes + b_connected_handshakes + b_isolated_handshakes

/-- The main theorem stating the number of handshakes in the given social gathering -/
theorem social_gathering_handshakes :
  let g : SocialGathering := {
    total_people := 30,
    group_a_size := 15,
    group_b_size := 15,
    group_b_connected := 10,
    group_b_isolated := 5,
    a_to_b_connections := 5,
    a_to_b_per_person := 3
  }
  count_handshakes g = 255 := by
  sorry


end NUMINAMATH_CALUDE_social_gathering_handshakes_l3256_325631


namespace NUMINAMATH_CALUDE_no_genetic_recombination_in_dna_replication_l3256_325644

-- Define the basic types
def Cell : Type := String
def Process : Type := String

-- Define the specific cell and processes
def spermatogonialCell : Cell := "spermatogonial cell"
def geneticRecombination : Process := "genetic recombination"
def dnaUnwinding : Process := "DNA unwinding"
def geneMutation : Process := "gene mutation"
def proteinSynthesis : Process := "protein synthesis"

-- Define a function to represent whether a process occurs during DNA replication
def occursInDnaReplication (c : Cell) (p : Process) : Prop := sorry

-- State the theorem
theorem no_genetic_recombination_in_dna_replication :
  occursInDnaReplication spermatogonialCell dnaUnwinding ∧
  occursInDnaReplication spermatogonialCell geneMutation ∧
  occursInDnaReplication spermatogonialCell proteinSynthesis →
  ¬ occursInDnaReplication spermatogonialCell geneticRecombination :=
by sorry

end NUMINAMATH_CALUDE_no_genetic_recombination_in_dna_replication_l3256_325644


namespace NUMINAMATH_CALUDE_range_implies_m_value_subset_implies_m_range_l3256_325692

-- Define the function f(x)
def f (x m : ℝ) : ℝ := |x - m| - |x - 2|

-- Define the solution set M
def M (m : ℝ) : Set ℝ := {x | f x m ≥ |x - 4|}

-- Theorem for part (1)
theorem range_implies_m_value (m : ℝ) :
  (∀ y ∈ Set.Icc (-4) 4, ∃ x, f x m = y) →
  (∀ x, f x m ∈ Set.Icc (-4) 4) →
  m = -2 ∨ m = 6 := by sorry

-- Theorem for part (2)
theorem subset_implies_m_range (m : ℝ) :
  Set.Icc 2 4 ⊆ M m →
  m ∈ Set.Iic 0 ∪ Set.Ici 6 := by sorry

end NUMINAMATH_CALUDE_range_implies_m_value_subset_implies_m_range_l3256_325692


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3256_325606

/-- Given a geometric sequence {a_n} where a_{2013} + a_{2015} = π, 
    prove that a_{2014}(a_{2012} + 2a_{2014} + a_{2016}) = π^2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h1 : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) 
    (h2 : a 2013 + a 2015 = π) : 
  a 2014 * (a 2012 + 2 * a 2014 + a 2016) = π^2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l3256_325606


namespace NUMINAMATH_CALUDE_fourth_guard_theorem_l3256_325672

/-- Represents a rectangular facility with guards at each corner -/
structure Facility :=
  (length : ℝ)
  (width : ℝ)
  (guard_distance : ℝ)

/-- Calculates the distance run by the fourth guard -/
def fourth_guard_distance (f : Facility) : ℝ :=
  2 * (f.length + f.width) - f.guard_distance

/-- Theorem stating the distance run by the fourth guard -/
theorem fourth_guard_theorem (f : Facility) 
  (h1 : f.length = 200)
  (h2 : f.width = 300)
  (h3 : f.guard_distance = 850) :
  fourth_guard_distance f = 150 := by
  sorry

#eval fourth_guard_distance { length := 200, width := 300, guard_distance := 850 }

end NUMINAMATH_CALUDE_fourth_guard_theorem_l3256_325672


namespace NUMINAMATH_CALUDE_negative_third_greater_than_negative_half_l3256_325617

theorem negative_third_greater_than_negative_half : -1/3 > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_negative_third_greater_than_negative_half_l3256_325617


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3256_325611

def A : Set ℝ := {x | x ≤ 2*x + 1 ∧ 2*x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3256_325611


namespace NUMINAMATH_CALUDE_combined_shoe_size_l3256_325636

/-- Given the shoe sizes of Jasmine, Alexa, and Clara, prove their combined shoe size. -/
theorem combined_shoe_size 
  (jasmine_size : ℕ) 
  (alexa_size : ℕ) 
  (clara_size : ℕ) 
  (h1 : jasmine_size = 7)
  (h2 : alexa_size = 2 * jasmine_size)
  (h3 : clara_size = 3 * jasmine_size) : 
  jasmine_size + alexa_size + clara_size = 42 := by
  sorry

end NUMINAMATH_CALUDE_combined_shoe_size_l3256_325636


namespace NUMINAMATH_CALUDE_equilateral_not_obtuse_l3256_325673

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define properties of a triangle
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.angleA > 90 ∨ t.angleB > 90 ∨ t.angleC > 90

-- Theorem: An equilateral triangle cannot be obtuse
theorem equilateral_not_obtuse (t : Triangle) :
  t.isEquilateral → ¬t.isObtuse := by
  sorry

end NUMINAMATH_CALUDE_equilateral_not_obtuse_l3256_325673


namespace NUMINAMATH_CALUDE_lillian_initial_candies_l3256_325649

-- Define the variables
def initial_candies : ℕ := sorry
def father_gave : ℕ := 5
def total_candies : ℕ := 93

-- State the theorem
theorem lillian_initial_candies : 
  initial_candies + father_gave = total_candies → initial_candies = 88 :=
by
  sorry

end NUMINAMATH_CALUDE_lillian_initial_candies_l3256_325649


namespace NUMINAMATH_CALUDE_day_365_is_tuesday_l3256_325628

/-- Days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Function to determine the day of the week for a given day number -/
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem day_365_is_tuesday (h : dayOfWeek 15 = DayOfWeek.Tuesday) :
  dayOfWeek 365 = DayOfWeek.Tuesday := by
  sorry

end NUMINAMATH_CALUDE_day_365_is_tuesday_l3256_325628


namespace NUMINAMATH_CALUDE_expected_value_Y_l3256_325694

-- Define a random variable X
variable (X : ℝ → ℝ)

-- Define Y as a function of X
def Y (X : ℝ → ℝ) : ℝ → ℝ := λ ω => 2 * (X ω) + 7

-- Define the expectation operator M
def M (Z : ℝ → ℝ) : ℝ := sorry

-- State the theorem
theorem expected_value_Y (hX : M X = 4) : M (Y X) = 15 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_Y_l3256_325694


namespace NUMINAMATH_CALUDE_existence_of_xy_sequences_l3256_325645

def sequence_a : ℕ → ℤ
  | 0 => 4
  | 1 => 22
  | (n + 2) => 6 * sequence_a (n + 1) - sequence_a n

theorem existence_of_xy_sequences :
  ∃ (x y : ℕ → ℕ), ∀ n : ℕ,
    sequence_a n = (y n ^ 2 + 7) / (x n - y n) :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_xy_sequences_l3256_325645


namespace NUMINAMATH_CALUDE_log_equation_solution_l3256_325640

theorem log_equation_solution (x : ℝ) :
  x > 0 → (4 * Real.log x / Real.log 3 = Real.log (6 * x) / Real.log 3) → x = (6 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3256_325640


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_9_l3256_325668

def is_divisible_by_9 (n : ℕ) : Prop := ∃ k, n = 9 * k

def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

theorem four_digit_divisible_by_9 (A : ℕ) (h1 : digit A) (h2 : is_divisible_by_9 (3000 + 100 * A + 10 * A + 1)) :
  A = 7 := by sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_9_l3256_325668


namespace NUMINAMATH_CALUDE_trivia_team_groups_l3256_325626

/-- Given a total number of students, number of students not picked, and students per group,
    calculate the number of groups formed. -/
def calculate_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) : ℕ :=
  (total_students - not_picked) / students_per_group

/-- Theorem stating that with 17 total students, 5 not picked, and 4 per group, 3 groups are formed. -/
theorem trivia_team_groups : calculate_groups 17 5 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_groups_l3256_325626


namespace NUMINAMATH_CALUDE_lucille_earnings_l3256_325638

/-- Represents the number of weeds in different areas of the garden -/
structure GardenWeeds where
  flower_bed : Nat
  vegetable_patch : Nat
  grass : Nat

/-- Represents Lucille's weeding and earnings -/
def LucilleWeeding (garden : GardenWeeds) (soda_cost : Nat) (money_left : Nat) : Prop :=
  let weeds_pulled := garden.flower_bed + garden.vegetable_patch + garden.grass / 2
  let total_earnings := soda_cost + money_left
  total_earnings / weeds_pulled = 6

/-- Theorem: Given the garden conditions and Lucille's spending, she earns 6 cents per weed -/
theorem lucille_earnings (garden : GardenWeeds) 
  (h1 : garden.flower_bed = 11)
  (h2 : garden.vegetable_patch = 14)
  (h3 : garden.grass = 32)
  (h4 : LucilleWeeding garden 99 147) : 
  ∃ (earnings_per_weed : Nat), earnings_per_weed = 6 := by
  sorry

end NUMINAMATH_CALUDE_lucille_earnings_l3256_325638


namespace NUMINAMATH_CALUDE_inequality_proof_l3256_325657

theorem inequality_proof (A B C ε : Real) 
  (hA : 0 ≤ A ∧ A ≤ π) 
  (hB : 0 ≤ B ∧ B ≤ π) 
  (hC : 0 ≤ C ∧ C ≤ π) 
  (hε : ε ≥ 1) : 
  ε * (Real.sin A + Real.sin B + Real.sin C) ≤ Real.sin A * Real.sin B * Real.sin C + 1 + ε^3 ∧ 
  (1 + ε + Real.sin A) * (1 + ε + Real.sin B) * (1 + ε + Real.sin C) ≥ 9 * ε * (Real.sin A + Real.sin B + Real.sin C) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3256_325657


namespace NUMINAMATH_CALUDE_square_difference_theorem_l3256_325623

theorem square_difference_theorem (x : ℝ) (h : (x + 2) * (x - 2) = 1221) : 
  x^2 = 1225 ∧ (x + 1) * (x - 1) = 1224 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l3256_325623


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3256_325669

theorem smallest_n_congruence (n : ℕ+) : 
  (19 * n.val ≡ 1589 [MOD 9]) ↔ n = 5 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3256_325669


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_88_l3256_325656

theorem largest_four_digit_divisible_by_88 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 88 = 0 → n ≤ 9944 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_88_l3256_325656


namespace NUMINAMATH_CALUDE_expression_evaluation_l3256_325670

theorem expression_evaluation :
  let x : ℚ := -3
  let expr := ((-2 * x^3 - 6*x) / (-2*x)) - 2*(3*x + 1)*(3*x - 1) + 7*x*(x - 1)
  expr = -64 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3256_325670


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3256_325688

theorem quadratic_roots_relation (s t : ℝ) 
  (hs : 19 * s^2 + 99 * s + 1 = 0)
  (ht : t^2 + 99 * t + 19 = 0)
  (hst : s * t ≠ 1) :
  (s * t + 4 * s + 1) / t = -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3256_325688


namespace NUMINAMATH_CALUDE_clark_discount_clark_discount_proof_l3256_325607

/-- Calculates the discount given to Clark for purchasing auto parts -/
theorem clark_discount (original_price : ℕ) (quantity : ℕ) (total_paid : ℕ) : ℕ :=
  let total_without_discount := original_price * quantity
  let discount := total_without_discount - total_paid
  discount

/-- Proves that Clark's discount is $121 given the problem conditions -/
theorem clark_discount_proof :
  clark_discount 80 7 439 = 121 := by
  sorry

end NUMINAMATH_CALUDE_clark_discount_clark_discount_proof_l3256_325607


namespace NUMINAMATH_CALUDE_orthocenter_position_in_isosceles_triangle_l3256_325682

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Vector2D :=
  sorry

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

theorem orthocenter_position_in_isosceles_triangle 
  (t : Triangle) 
  (h_isosceles : isIsosceles t) 
  (h_sides : t.a = 5 ∧ t.b = 5 ∧ t.c = 6) :
  ∃ (m n : ℝ), 
    let H := orthocenter t
    let A := Vector2D.mk 0 0
    let B := Vector2D.mk t.c 0
    let C := Vector2D.mk (t.c / 2) (Real.sqrt (t.a^2 - (t.c / 2)^2))
    H.x = m * B.x + n * C.x ∧
    H.y = m * B.y + n * C.y ∧
    m + n = 21 / 32 :=
  sorry

end NUMINAMATH_CALUDE_orthocenter_position_in_isosceles_triangle_l3256_325682


namespace NUMINAMATH_CALUDE_average_weight_of_removed_carrots_l3256_325667

/-- The average weight of 4 removed carrots given the following conditions:
    - There are initially 20 carrots, 10 apples, and 5 oranges
    - The total initial weight is 8.70 kg
    - After removal, there are 16 carrots and 8 apples
    - The average weight after removal is 206 grams
    - The average weight of an apple is 210 grams -/
theorem average_weight_of_removed_carrots :
  ∀ (total_weight : ℝ) 
    (initial_carrots initial_apples initial_oranges : ℕ)
    (remaining_carrots remaining_apples : ℕ)
    (avg_weight_after_removal avg_weight_apple : ℝ),
  total_weight = 8.70 ∧
  initial_carrots = 20 ∧
  initial_apples = 10 ∧
  initial_oranges = 5 ∧
  remaining_carrots = 16 ∧
  remaining_apples = 8 ∧
  avg_weight_after_removal = 206 ∧
  avg_weight_apple = 210 →
  (total_weight * 1000 - 
   (remaining_carrots + remaining_apples) * avg_weight_after_removal - 
   (initial_apples - remaining_apples) * avg_weight_apple) / 
   (initial_carrots - remaining_carrots) = 834 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_of_removed_carrots_l3256_325667


namespace NUMINAMATH_CALUDE_remainder_problem_l3256_325625

theorem remainder_problem (n : ℤ) (k : ℤ) (h : n = 25 * k - 2) :
  (n^2 + 3*n + 5) % 25 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3256_325625


namespace NUMINAMATH_CALUDE_abc_plus_def_equals_zero_l3256_325624

/-- Represents the transformation of numbers in the circle --/
def transform (v : Fin 6 → ℝ) : Fin 6 → ℝ := fun i =>
  v i + v (i - 1) + v (i + 1)

/-- The condition that after 2022 iterations, the numbers return to their initial values --/
def returns_to_initial (v : Fin 6 → ℝ) : Prop :=
  (transform^[2022]) v = v

theorem abc_plus_def_equals_zero 
  (v : Fin 6 → ℝ) 
  (h : returns_to_initial v) : 
  v 0 * v 1 * v 2 + v 3 * v 4 * v 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_plus_def_equals_zero_l3256_325624


namespace NUMINAMATH_CALUDE_product_of_numbers_l3256_325686

theorem product_of_numbers (a b : ℝ) (h1 : a + b = 70) (h2 : a - b = 10) : a * b = 1200 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3256_325686


namespace NUMINAMATH_CALUDE_michelles_necklace_l3256_325680

/-- Prove that the number of silver beads is 10 given the conditions of Michelle's necklace. -/
theorem michelles_necklace (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ) :
  total_beads = 40 →
  blue_beads = 5 →
  red_beads = 2 * blue_beads →
  white_beads = blue_beads + red_beads →
  silver_beads = total_beads - (blue_beads + red_beads + white_beads) →
  silver_beads = 10 := by
  sorry

end NUMINAMATH_CALUDE_michelles_necklace_l3256_325680


namespace NUMINAMATH_CALUDE_base_difference_equals_7422_l3256_325627

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The main theorem --/
theorem base_difference_equals_7422 :
  let base_7_num := to_base_10 [3, 4, 1, 2, 5] 7
  let base_8_num := to_base_10 [5, 4, 3, 2, 1] 8
  base_7_num - base_8_num = 7422 := by sorry

end NUMINAMATH_CALUDE_base_difference_equals_7422_l3256_325627


namespace NUMINAMATH_CALUDE_no_solution_to_system_l3256_325637

theorem no_solution_to_system :
  ¬∃ (x y : ℝ), 
    (80 * x + 15 * y - 7) / (78 * x + 12 * y) = 1 ∧
    (2 * x^2 + 3 * y^2 - 11) / (y^2 - x^2 + 3) = 1 ∧
    78 * x + 12 * y ≠ 0 ∧
    y^2 - x^2 + 3 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l3256_325637


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_9_and_3_digit_by_4_l3256_325684

/-- Represents a four-digit number -/
structure FourDigitNumber where
  value : Nat
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Returns the three-digit number obtained by removing the last digit -/
def remove_last_digit (n : FourDigitNumber) : Nat :=
  n.value / 10

/-- Returns the last digit of a number -/
def last_digit (n : FourDigitNumber) : Nat :=
  n.value % 10

theorem largest_four_digit_divisible_by_9_and_3_digit_by_4 (n : FourDigitNumber) 
  (h1 : n.value % 9 = 0)
  (h2 : remove_last_digit n % 4 = 0)
  (h3 : ∀ m : FourDigitNumber, m.value % 9 = 0 → remove_last_digit m % 4 = 0 → m.value ≤ n.value) :
  last_digit n = 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_9_and_3_digit_by_4_l3256_325684


namespace NUMINAMATH_CALUDE_two_month_discount_l3256_325614

/-- Calculates the final price of an item after two consecutive percentage discounts --/
theorem two_month_discount (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  initial_price = 1000 ∧ discount1 = 10 ∧ discount2 = 20 →
  initial_price * (1 - discount1 / 100) * (1 - discount2 / 100) = 720 := by
sorry


end NUMINAMATH_CALUDE_two_month_discount_l3256_325614


namespace NUMINAMATH_CALUDE_octal_subtraction_result_l3256_325630

/-- Represents a number in base 8 --/
def OctalNum := Nat

/-- Addition of two octal numbers --/
def octal_add (a b : OctalNum) : OctalNum :=
  sorry

/-- Subtraction of two octal numbers --/
def octal_sub (a b : OctalNum) : OctalNum :=
  sorry

/-- Conversion from decimal to octal --/
def to_octal (n : Nat) : OctalNum :=
  sorry

/-- Theorem: The result of subtracting 53₈ from the sum of 652₈ and 147₈ is 50₈ --/
theorem octal_subtraction_result :
  octal_sub (octal_add (to_octal 652) (to_octal 147)) (to_octal 53) = to_octal 50 :=
by sorry

end NUMINAMATH_CALUDE_octal_subtraction_result_l3256_325630


namespace NUMINAMATH_CALUDE_ball_hits_ground_at_calculated_time_l3256_325601

/-- The time when a ball hits the ground, given its height equation -/
def ball_ground_time : ℝ :=
  let initial_height : ℝ := 180
  let initial_velocity : ℝ := -32  -- negative because it's downward
  let release_delay : ℝ := 1
  let height (t : ℝ) : ℝ := -16 * (t - release_delay)^2 - 32 * (t - release_delay) + initial_height
  3.5

/-- Theorem stating that the ball hits the ground at the calculated time -/
theorem ball_hits_ground_at_calculated_time :
  let height (t : ℝ) : ℝ := -16 * (t - 1)^2 - 32 * (t - 1) + 180
  height ball_ground_time = 0 := by
  sorry

end NUMINAMATH_CALUDE_ball_hits_ground_at_calculated_time_l3256_325601


namespace NUMINAMATH_CALUDE_eggs_per_hen_l3256_325691

theorem eggs_per_hen (total_chickens : ℕ) (roosters : ℕ) (non_laying_hens : ℕ) (total_eggs : ℕ)
  (h1 : total_chickens = 440)
  (h2 : roosters = 39)
  (h3 : non_laying_hens = 15)
  (h4 : total_eggs = 1158) :
  total_eggs / (total_chickens - roosters - non_laying_hens) = 3 :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_hen_l3256_325691


namespace NUMINAMATH_CALUDE_apps_added_l3256_325653

theorem apps_added (initial_apps : ℕ) (deleted_apps : ℕ) (final_apps : ℕ) :
  initial_apps = 10 →
  deleted_apps = 17 →
  final_apps = 4 →
  ∃ (added_apps : ℕ), (initial_apps + added_apps - deleted_apps = final_apps) ∧ (added_apps = 11) :=
by sorry

end NUMINAMATH_CALUDE_apps_added_l3256_325653


namespace NUMINAMATH_CALUDE_bread_roll_combinations_l3256_325650

theorem bread_roll_combinations :
  let total_rolls : ℕ := 10
  let num_types : ℕ := 4
  let min_rolls_type1 : ℕ := 2
  let min_rolls_type2 : ℕ := 2
  let min_rolls_type3 : ℕ := 1
  let min_rolls_type4 : ℕ := 1
  let remaining_rolls : ℕ := total_rolls - (min_rolls_type1 + min_rolls_type2 + min_rolls_type3 + min_rolls_type4)
  (Nat.choose (remaining_rolls + num_types - 1) (num_types - 1)) = 35 :=
by sorry

end NUMINAMATH_CALUDE_bread_roll_combinations_l3256_325650


namespace NUMINAMATH_CALUDE_expected_profit_is_140000_l3256_325699

/-- The probability of a machine malfunctioning within a day -/
def malfunction_prob : ℝ := 0.2

/-- The loss incurred when a machine malfunctions (in yuan) -/
def malfunction_loss : ℝ := 50000

/-- The profit made when a machine works normally (in yuan) -/
def normal_profit : ℝ := 100000

/-- The number of machines -/
def num_machines : ℕ := 2

/-- The expected profit of two identical machines within a day (in yuan) -/
def expected_profit : ℝ := num_machines * (normal_profit * (1 - malfunction_prob) - malfunction_loss * malfunction_prob)

theorem expected_profit_is_140000 : expected_profit = 140000 := by
  sorry

end NUMINAMATH_CALUDE_expected_profit_is_140000_l3256_325699


namespace NUMINAMATH_CALUDE_bucket_ratio_l3256_325646

theorem bucket_ratio (small_bucket : ℚ) (large_bucket : ℚ) : 
  (∃ (n : ℚ), large_bucket = n * small_bucket + 3) →
  2 * small_bucket + 5 * large_bucket = 63 →
  large_bucket = 4 →
  large_bucket / small_bucket = 4 := by
sorry

end NUMINAMATH_CALUDE_bucket_ratio_l3256_325646


namespace NUMINAMATH_CALUDE_predicted_height_at_10_l3256_325660

/-- Represents the regression model for height prediction -/
def height_model (age : ℝ) : ℝ := 7.19 * age + 73.93

/-- Theorem stating that the predicted height at age 10 is approximately 145.83 cm -/
theorem predicted_height_at_10 :
  ∃ ε > 0, |height_model 10 - 145.83| < ε :=
sorry

end NUMINAMATH_CALUDE_predicted_height_at_10_l3256_325660


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l3256_325695

theorem quadratic_equation_root (m : ℝ) : 
  (3 : ℝ) ^ 2 - 3 - m = 0 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l3256_325695


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l3256_325620

theorem ratio_of_sum_and_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l3256_325620


namespace NUMINAMATH_CALUDE_paula_tickets_l3256_325666

/-- The number of tickets needed for Paula's amusement park rides -/
def tickets_needed (go_kart_rides : ℕ) (bumper_car_rides : ℕ) (go_kart_cost : ℕ) (bumper_car_cost : ℕ) : ℕ :=
  go_kart_rides * go_kart_cost + bumper_car_rides * bumper_car_cost

/-- Theorem: Paula needs 24 tickets for her amusement park rides -/
theorem paula_tickets : tickets_needed 1 4 4 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_paula_tickets_l3256_325666


namespace NUMINAMATH_CALUDE_modular_congruence_unique_solution_l3256_325629

theorem modular_congruence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 15893 [ZMOD 16] := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_unique_solution_l3256_325629


namespace NUMINAMATH_CALUDE_new_total_weight_l3256_325605

/-- Proves that the new total weight of Ram and Shyam is 13.8 times their original common weight factor -/
theorem new_total_weight (x : ℝ) (x_pos : x > 0) : 
  let ram_original := 7 * x
  let shyam_original := 5 * x
  let ram_new := ram_original * 1.1
  let shyam_new := shyam_original * 1.22
  let total_original := ram_original + shyam_original
  let total_new := ram_new + shyam_new
  total_new = total_original * 1.15 ∧ total_new = 13.8 * x :=
by sorry

end NUMINAMATH_CALUDE_new_total_weight_l3256_325605


namespace NUMINAMATH_CALUDE_symmetric_lines_l3256_325604

/-- Given two lines in the xy-plane, this function returns true if they are symmetric about the line x = a -/
def are_symmetric_lines (line1 line2 : ℝ → ℝ → Prop) (a : ℝ) : Prop :=
  ∀ x y, line1 x y ↔ line2 (2*a - x) y

/-- The equation of the first line: 2x + y - 1 = 0 -/
def line1 (x y : ℝ) : Prop := 2*x + y - 1 = 0

/-- The equation of the second line: 2x - y - 3 = 0 -/
def line2 (x y : ℝ) : Prop := 2*x - y - 3 = 0

/-- The line of symmetry: x = 1 -/
def symmetry_line : ℝ := 1

theorem symmetric_lines : are_symmetric_lines line1 line2 symmetry_line := by
  sorry

end NUMINAMATH_CALUDE_symmetric_lines_l3256_325604


namespace NUMINAMATH_CALUDE_workers_savings_l3256_325618

/-- A worker's savings problem -/
theorem workers_savings (monthly_pay : ℝ) (saving_fraction : ℝ) : 
  monthly_pay > 0 →
  saving_fraction > 0 →
  saving_fraction < 1 →
  (12 * saving_fraction * monthly_pay) = (4 * (1 - saving_fraction) * monthly_pay) →
  saving_fraction = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_workers_savings_l3256_325618


namespace NUMINAMATH_CALUDE_real_y_condition_l3256_325635

theorem real_y_condition (x : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 2 * x * y + x + 5 = 0) ↔ x ≤ -3 ∨ x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_real_y_condition_l3256_325635


namespace NUMINAMATH_CALUDE_x_seven_y_eight_l3256_325658

theorem x_seven_y_eight (x y : ℚ) (hx : x = 3/4) (hy : y = 4/3) : x^7 * y^8 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_x_seven_y_eight_l3256_325658


namespace NUMINAMATH_CALUDE_amber_work_hours_l3256_325661

theorem amber_work_hours :
  ∀ (amber armand ella : ℝ),
  armand = amber / 3 →
  ella = 2 * amber →
  amber + armand + ella = 40 →
  amber = 12 := by
sorry

end NUMINAMATH_CALUDE_amber_work_hours_l3256_325661


namespace NUMINAMATH_CALUDE_negative_exponent_division_l3256_325619

theorem negative_exponent_division (a : ℝ) : -a^6 / a^3 = -a^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_exponent_division_l3256_325619


namespace NUMINAMATH_CALUDE_yanna_kept_apples_l3256_325622

def apples_kept (total bought : ℕ) (given_to_zenny given_to_andrea : ℕ) : ℕ :=
  bought - given_to_zenny - given_to_andrea

theorem yanna_kept_apples :
  apples_kept 60 18 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_yanna_kept_apples_l3256_325622


namespace NUMINAMATH_CALUDE_octagon_handshakes_eight_students_l3256_325664

/-- The number of handshakes in an octagonal arrangement of students -/
def octagon_handshakes (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: In a group of 8 students arranged in an octagonal shape,
    where each student shakes hands once with every other student
    except their two neighbors, the total number of handshakes is 20. -/
theorem octagon_handshakes_eight_students :
  octagon_handshakes 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_handshakes_eight_students_l3256_325664


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3256_325689

/-- Represents a hyperbola with equation x²/9 - y²/m = 1 -/
structure Hyperbola where
  m : ℝ

/-- Represents a line with equation x + y = 5 -/
def focus_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 5}

/-- Represents the asymptotes of a hyperbola -/
structure Asymptotes where
  slope : ℝ

theorem hyperbola_asymptotes (h : Hyperbola) (focus_on_line : ∃ p : ℝ × ℝ, p ∈ focus_line ∧ p.2 = 0) :
  Asymptotes.mk (4/3) = Asymptotes.mk (-4/3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3256_325689


namespace NUMINAMATH_CALUDE_polynomial_roots_l3256_325647

theorem polynomial_roots (p : ℝ) (hp : p > 5/4) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
    x₁^4 - 2*p*x₁^3 + x₁^2 - 2*p*x₁ + 1 = 0 ∧
    x₂^4 - 2*p*x₂^3 + x₂^2 - 2*p*x₂ + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3256_325647


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3256_325662

theorem expand_and_simplify (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3256_325662


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3256_325652

theorem absolute_value_equation_solution : 
  ∃ (x : ℝ), (|x - 3| = 5 - 2*x) ∧ (x = 8/3 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3256_325652


namespace NUMINAMATH_CALUDE_part_one_part_two_l3256_325674

/-- Definition of proposition p -/
def p (x a : ℝ) : Prop := (x - 3*a) * (x - a) < 0

/-- Definition of proposition q -/
def q (x : ℝ) : Prop := |x - 3| < 1

/-- Part 1 of the theorem -/
theorem part_one : 
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3) :=
sorry

/-- Part 2 of the theorem -/
theorem part_two :
  ∀ a : ℝ, a > 0 → 
  ((∀ x : ℝ, ¬(p x a) → ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x a)) →
  (4/3 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3256_325674


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3256_325616

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 4 + a 7 = 39 →
  a 2 + a 5 + a 8 = 33 →
  a 5 + a 8 + a 11 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3256_325616


namespace NUMINAMATH_CALUDE_no_solutions_for_sqrt_equation_l3256_325683

theorem no_solutions_for_sqrt_equation :
  ¬∃ x : ℝ, x ≥ 4 ∧ Real.sqrt (x + 9 - 6 * Real.sqrt (x - 4)) + Real.sqrt (x + 16 - 8 * Real.sqrt (x - 4)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_for_sqrt_equation_l3256_325683


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_360_l3256_325602

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem perfect_square_factors_of_360 :
  let factorization := prime_factorization 360
  (factorization = [(2, 3), (3, 2), (5, 1)]) →
  count_perfect_square_factors 360 = 4 := by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_360_l3256_325602


namespace NUMINAMATH_CALUDE_mapping_result_l3256_325665

def A : Set ℕ := {1, 2}

def f (x : ℕ) : ℕ := x^2

def B : Set ℕ := f '' A

theorem mapping_result : B = {1, 4} := by sorry

end NUMINAMATH_CALUDE_mapping_result_l3256_325665


namespace NUMINAMATH_CALUDE_simplest_form_product_l3256_325678

theorem simplest_form_product (a b : ℕ) (h : a = 45 ∧ b = 75) : 
  let g := Nat.gcd a b
  (a / g) * (b / g) = 15 := by
sorry

end NUMINAMATH_CALUDE_simplest_form_product_l3256_325678


namespace NUMINAMATH_CALUDE_cheese_cost_proof_l3256_325693

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 2

/-- The number of sandwiches Ted makes -/
def num_sandwiches : ℕ := 10

/-- The cost of bread in dollars -/
def bread_cost : ℚ := 4

/-- The cost of one pack of sandwich meat in dollars -/
def meat_cost : ℚ := 5

/-- The number of packs of sandwich meat needed -/
def num_meat_packs : ℕ := 2

/-- The number of packs of sliced cheese needed -/
def num_cheese_packs : ℕ := 2

/-- The discount on one pack of cheese in dollars -/
def cheese_discount : ℚ := 1

/-- The discount on one pack of meat in dollars -/
def meat_discount : ℚ := 1

/-- The cost of one pack of sliced cheese without the coupon -/
def cheese_cost : ℚ := 4.5

theorem cheese_cost_proof :
  cheese_cost * num_cheese_packs + bread_cost + meat_cost * num_meat_packs - 
  cheese_discount - meat_discount = sandwich_cost * num_sandwiches := by
  sorry

end NUMINAMATH_CALUDE_cheese_cost_proof_l3256_325693


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3256_325642

theorem smallest_number_with_given_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 3 ∧
  ∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 5 = 2 ∧ m % 7 = 3 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3256_325642

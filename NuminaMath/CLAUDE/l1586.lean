import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_set_l1586_158675

theorem inequality_solution_set (x : ℝ) :
  (x^2 / (x + 1) ≥ 3 / (x - 2) + 9 / 4) ↔ (x < -3/4 ∨ (x > 2 ∧ x < 5)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1586_158675


namespace NUMINAMATH_CALUDE_total_travel_time_l1586_158650

def distance_washington_idaho : ℝ := 640
def distance_idaho_nevada : ℝ := 550
def speed_washington_idaho : ℝ := 80
def speed_idaho_nevada : ℝ := 50

theorem total_travel_time :
  (distance_washington_idaho / speed_washington_idaho) +
  (distance_idaho_nevada / speed_idaho_nevada) = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_l1586_158650


namespace NUMINAMATH_CALUDE_geometric_arithmetic_relation_l1586_158608

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_arithmetic_relation (a b : ℕ → ℝ) :
  geometric_sequence a
  → a 2 = 4
  → a 4 = 16
  → arithmetic_sequence b
  → b 3 = a 3
  → b 5 = a 5
  → ∀ n : ℕ, b n = 12 * n - 28 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_relation_l1586_158608


namespace NUMINAMATH_CALUDE_bert_shopping_trip_l1586_158623

theorem bert_shopping_trip (initial_amount : ℝ) : 
  initial_amount = 52 →
  let hardware_spend := initial_amount / 4
  let after_hardware := initial_amount - hardware_spend
  let dryclean_spend := 9
  let after_dryclean := after_hardware - dryclean_spend
  let grocery_spend := after_dryclean / 2
  let final_amount := after_dryclean - grocery_spend
  final_amount = 15 := by
sorry

end NUMINAMATH_CALUDE_bert_shopping_trip_l1586_158623


namespace NUMINAMATH_CALUDE_john_spends_more_l1586_158639

/-- Calculates the difference in annual cost between John's new and former living arrangements -/
def annual_cost_difference (former_rent_per_sqft : ℚ) (former_size : ℚ) 
  (new_rent_first_half : ℚ) (new_rent_increase_percent : ℚ) 
  (winter_utilities : ℚ) (other_utilities : ℚ) : ℚ :=
  let former_annual_cost := former_rent_per_sqft * former_size * 12
  let new_rent_second_half := new_rent_first_half * (1 + new_rent_increase_percent)
  let new_annual_rent := new_rent_first_half * 6 + new_rent_second_half * 6
  let new_annual_utilities := winter_utilities * 3 + other_utilities * 9
  let new_total_cost := new_annual_rent + new_annual_utilities
  let john_new_cost := new_total_cost / 2
  john_new_cost - former_annual_cost

/-- Theorem stating that John spends $195 more annually in the new arrangement -/
theorem john_spends_more : 
  annual_cost_difference 2 750 2800 (5/100) 200 150 = 195 := by
  sorry

end NUMINAMATH_CALUDE_john_spends_more_l1586_158639


namespace NUMINAMATH_CALUDE_point_wrt_y_axis_point_4_neg8_wrt_y_axis_l1586_158644

/-- Given a point A with coordinates (x, y) in a 2D plane,
    this theorem states that the coordinates of A with respect to the y-axis are (-x, y). -/
theorem point_wrt_y_axis (x y : ℝ) : 
  let A : ℝ × ℝ := (x, y)
  let A_wrt_y_axis : ℝ × ℝ := (-x, y)
  A_wrt_y_axis = (- (A.1), A.2) := by
sorry

/-- The coordinates of the point A(4, -8) with respect to the y-axis are (-4, -8). -/
theorem point_4_neg8_wrt_y_axis : 
  let A : ℝ × ℝ := (4, -8)
  let A_wrt_y_axis : ℝ × ℝ := (-4, -8)
  A_wrt_y_axis = (- (A.1), A.2) := by
sorry

end NUMINAMATH_CALUDE_point_wrt_y_axis_point_4_neg8_wrt_y_axis_l1586_158644


namespace NUMINAMATH_CALUDE_weight_of_ten_moles_C6H8O6_l1586_158600

/-- The weight of 10 moles of C6H8O6 -/
theorem weight_of_ten_moles_C6H8O6 
  (atomic_weight_C : ℝ) 
  (atomic_weight_H : ℝ) 
  (atomic_weight_O : ℝ) 
  (h1 : atomic_weight_C = 12.01)
  (h2 : atomic_weight_H = 1.008)
  (h3 : atomic_weight_O = 16.00) : 
  10 * (6 * atomic_weight_C + 8 * atomic_weight_H + 6 * atomic_weight_O) = 1761.24 := by
sorry

end NUMINAMATH_CALUDE_weight_of_ten_moles_C6H8O6_l1586_158600


namespace NUMINAMATH_CALUDE_wood_sawed_off_l1586_158622

theorem wood_sawed_off (original_length final_length : ℝ) 
  (h1 : original_length = 0.41)
  (h2 : final_length = 0.08) :
  original_length - final_length = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_wood_sawed_off_l1586_158622


namespace NUMINAMATH_CALUDE_max_interval_increasing_sin_plus_cos_l1586_158618

theorem max_interval_increasing_sin_plus_cos :
  let f : ℝ → ℝ := λ x ↦ Real.sin x + Real.cos x
  ∃ a : ℝ, a = π / 4 ∧ 
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ a → f x < f y) ∧
    (∀ b : ℝ, b > a → ∃ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ b ∧ f x ≥ f y) :=
by sorry

end NUMINAMATH_CALUDE_max_interval_increasing_sin_plus_cos_l1586_158618


namespace NUMINAMATH_CALUDE_cannot_achieve_multiple_100s_l1586_158642

/-- Represents the scores for Russian, Physics, and Mathematics exams -/
structure Scores where
  russian : ℕ
  physics : ℕ
  math : ℕ

/-- Defines the initial relationship between scores -/
def initial_score_relationship (s : Scores) : Prop :=
  s.russian = s.physics - 5 ∧ s.physics = s.math - 9

/-- Represents the two types of operations allowed -/
inductive Operation
  | add_one_to_all
  | decrease_one_increase_two

/-- Applies an operation to the scores -/
def apply_operation (s : Scores) (op : Operation) : Scores :=
  match op with
  | Operation.add_one_to_all => 
      { russian := s.russian + 1, physics := s.physics + 1, math := s.math + 1 }
  | Operation.decrease_one_increase_two => 
      { russian := s.russian - 3, physics := s.physics + 1, math := s.math + 1 }
      -- Note: This is just one possible application of the second operation

/-- Checks if any score exceeds 100 -/
def exceeds_100 (s : Scores) : Prop :=
  s.russian > 100 ∨ s.physics > 100 ∨ s.math > 100

/-- Checks if more than one score is equal to 100 -/
def more_than_one_100 (s : Scores) : Prop :=
  (s.russian = 100 ∧ s.physics = 100) ∨
  (s.russian = 100 ∧ s.math = 100) ∨
  (s.physics = 100 ∧ s.math = 100)

/-- The main theorem to be proved -/
theorem cannot_achieve_multiple_100s (s : Scores) 
  (h : initial_score_relationship s) : 
  ¬ ∃ (ops : List Operation), 
    let final_scores := ops.foldl apply_operation s
    ¬ exceeds_100 final_scores ∧ more_than_one_100 final_scores :=
sorry


end NUMINAMATH_CALUDE_cannot_achieve_multiple_100s_l1586_158642


namespace NUMINAMATH_CALUDE_esports_gender_related_prob_select_male_expected_like_esports_l1586_158687

-- Define the survey data
def total_students : ℕ := 400
def male_like : ℕ := 120
def male_dislike : ℕ := 80
def female_like : ℕ := 100
def female_dislike : ℕ := 100

-- Define the critical value for α = 0.05
def critical_value : ℚ := 3841/1000

-- Define the chi-square statistic function
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Theorem 1: Chi-square statistic is greater than critical value
theorem esports_gender_related :
  chi_square male_like male_dislike female_like female_dislike > critical_value := by
  sorry

-- Theorem 2: Probability of selecting at least one male student
theorem prob_select_male :
  1 - (Nat.choose 5 3 : ℚ) / (Nat.choose 9 3) = 37/42 := by
  sorry

-- Theorem 3: Expected number of students who like esports
theorem expected_like_esports :
  (10 : ℚ) * (male_like + female_like) / total_students = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_esports_gender_related_prob_select_male_expected_like_esports_l1586_158687


namespace NUMINAMATH_CALUDE_range_of_m_min_distance_to_origin_range_of_slope_l1586_158621

-- Define the circle C
def C (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 8*y + m = 0

-- Define point P
def P : ℝ × ℝ := (0, 4)

-- Theorem 1
theorem range_of_m (m : ℝ) :
  (∀ x y, C m x y → (P.1 - x)^2 + (P.2 - y)^2 > 0) → 16 < m ∧ m < 25 :=
sorry

-- Theorem 2
theorem min_distance_to_origin (x y : ℝ) :
  C 24 x y → x^2 + y^2 ≥ 16 :=
sorry

-- Theorem 3
theorem range_of_slope (x y : ℝ) :
  C 24 x y → x ≠ 0 → -Real.sqrt 2 / 4 ≤ (y - 4) / x ∧ (y - 4) / x ≤ Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_min_distance_to_origin_range_of_slope_l1586_158621


namespace NUMINAMATH_CALUDE_log_inequality_l1586_158684

theorem log_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) :
  Real.log (Real.sqrt (x₁ * x₂)) = (Real.log x₁ + Real.log x₂) / 2 ∧
  Real.log (Real.sqrt (x₁ * x₂)) < Real.log ((x₁ + x₂) / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1586_158684


namespace NUMINAMATH_CALUDE_congruence_implies_prime_and_n_equals_m_minus_one_l1586_158616

theorem congruence_implies_prime_and_n_equals_m_minus_one 
  (n m : ℕ) 
  (h_n : n ≥ 2) 
  (h_m : m ≥ 2) 
  (h_cong : ∀ k : ℕ, 1 ≤ k → k ≤ n → k^n % m = 1) : 
  Nat.Prime m ∧ n = m - 1 := by
sorry

end NUMINAMATH_CALUDE_congruence_implies_prime_and_n_equals_m_minus_one_l1586_158616


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_48_l1586_158685

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_non_factor_product_of_48 (x y : ℕ) : 
  x ≠ y → 
  x > 0 → 
  y > 0 → 
  is_factor x 48 → 
  is_factor y 48 → 
  ¬ is_factor (x * y) 48 → 
  ∀ a b : ℕ, a ≠ b → a > 0 → b > 0 → is_factor a 48 → is_factor b 48 → ¬ is_factor (a * b) 48 → x * y ≤ a * b →
  x * y = 32 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_48_l1586_158685


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l1586_158626

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ + 5| = 20) ∧ (|x₂ + 5| = 20) ∧ (x₁ ≠ x₂) ∧ (|x₁ - x₂| = 40) := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l1586_158626


namespace NUMINAMATH_CALUDE_relay_race_total_time_l1586_158690

/-- The total time for a relay race with four athletes -/
def relay_race_time (athlete1_time athlete2_extra athlete3_less athlete4_less : ℕ) : ℕ :=
  let athlete2_time := athlete1_time + athlete2_extra
  let athlete3_time := athlete2_time - athlete3_less
  let athlete4_time := athlete1_time - athlete4_less
  athlete1_time + athlete2_time + athlete3_time + athlete4_time

/-- Theorem stating that the total time for the given relay race is 200 seconds -/
theorem relay_race_total_time :
  relay_race_time 55 10 15 25 = 200 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_total_time_l1586_158690


namespace NUMINAMATH_CALUDE_root_sum_product_l1586_158676

theorem root_sum_product (c d : ℝ) : 
  (c^4 - 6*c^3 - 4*c - 1 = 0) → 
  (d^4 - 6*d^3 - 4*d - 1 = 0) → 
  (c ≠ d) →
  (cd + c + d = 4) := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l1586_158676


namespace NUMINAMATH_CALUDE_mitzi_remaining_money_l1586_158635

def amusement_park_spending (initial_amount ticket_cost food_cost tshirt_cost : ℕ) : ℕ :=
  initial_amount - (ticket_cost + food_cost + tshirt_cost)

theorem mitzi_remaining_money :
  amusement_park_spending 75 30 13 23 = 9 := by
  sorry

end NUMINAMATH_CALUDE_mitzi_remaining_money_l1586_158635


namespace NUMINAMATH_CALUDE_angle_T_measure_l1586_158613

-- Define a pentagon
structure Pentagon where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ

-- Define the properties of the pentagon
def is_valid_pentagon (p : Pentagon) : Prop :=
  p.P + p.Q + p.R + p.S + p.T = 540

def angles_congruent (p : Pentagon) : Prop :=
  p.P = p.R ∧ p.R = p.T

def angles_supplementary (p : Pentagon) : Prop :=
  p.Q + p.S = 180

-- Theorem statement
theorem angle_T_measure (p : Pentagon) 
  (h1 : is_valid_pentagon p) 
  (h2 : angles_congruent p) 
  (h3 : angles_supplementary p) : 
  p.T = 120 := by sorry

end NUMINAMATH_CALUDE_angle_T_measure_l1586_158613


namespace NUMINAMATH_CALUDE_b_completion_time_l1586_158688

/-- Represents the time (in days) it takes for a worker to complete a job alone -/
structure WorkerTime where
  days : ℝ
  days_pos : days > 0

/-- Represents the share of earnings for a worker -/
structure EarningShare where
  amount : ℝ
  amount_pos : amount > 0

/-- Represents a job with multiple workers -/
structure Job where
  a : WorkerTime
  c : WorkerTime
  total_earnings : ℝ
  b_share : EarningShare
  total_earnings_pos : total_earnings > 0

theorem b_completion_time (job : Job) 
  (ha : job.a.days = 6)
  (hc : job.c.days = 12)
  (htotal : job.total_earnings = 1170)
  (hb_share : job.b_share.amount = 390) :
  ∃ (b : WorkerTime), b.days = 8 := by
  sorry

end NUMINAMATH_CALUDE_b_completion_time_l1586_158688


namespace NUMINAMATH_CALUDE_company_supervisors_l1586_158611

/-- Represents the number of workers per team lead -/
def workers_per_team_lead : ℕ := 10

/-- Represents the number of team leads per supervisor -/
def team_leads_per_supervisor : ℕ := 3

/-- Represents the total number of workers in the company -/
def total_workers : ℕ := 390

/-- Calculates the number of supervisors in the company -/
def calculate_supervisors : ℕ :=
  (total_workers / workers_per_team_lead) / team_leads_per_supervisor

theorem company_supervisors :
  calculate_supervisors = 13 := by sorry

end NUMINAMATH_CALUDE_company_supervisors_l1586_158611


namespace NUMINAMATH_CALUDE_police_departments_female_officers_l1586_158674

/-- Represents a police department with female officers -/
structure Department where
  totalOfficers : ℕ
  femaleOfficersOnDuty : ℕ
  femaleOfficerPercentage : ℚ

/-- Calculates the total number of female officers in a department -/
def totalFemaleOfficers (d : Department) : ℕ :=
  (d.femaleOfficersOnDuty : ℚ) / d.femaleOfficerPercentage |>.ceil.toNat

theorem police_departments_female_officers 
  (deptA : Department)
  (deptB : Department)
  (deptC : Department)
  (hA : deptA = { totalOfficers := 180, femaleOfficersOnDuty := 90, femaleOfficerPercentage := 18/100 })
  (hB : deptB = { totalOfficers := 200, femaleOfficersOnDuty := 60, femaleOfficerPercentage := 25/100 })
  (hC : deptC = { totalOfficers := 150, femaleOfficersOnDuty := 40, femaleOfficerPercentage := 30/100 }) :
  totalFemaleOfficers deptA = 500 ∧
  totalFemaleOfficers deptB = 240 ∧
  totalFemaleOfficers deptC = 133 ∧
  totalFemaleOfficers deptA + totalFemaleOfficers deptB + totalFemaleOfficers deptC = 873 := by
  sorry

end NUMINAMATH_CALUDE_police_departments_female_officers_l1586_158674


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l1586_158612

-- Define the given constants
def train_length : ℝ := 250
def pole_passing_time : ℝ := 10
def platform_length : ℝ := 1250
def speed_reduction_factor : ℝ := 0.75

-- Define the theorem
theorem train_platform_passing_time :
  let original_speed := train_length / pole_passing_time
  let incline_speed := original_speed * speed_reduction_factor
  let total_distance := train_length + platform_length
  total_distance / incline_speed = 80 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_passing_time_l1586_158612


namespace NUMINAMATH_CALUDE_parabola_translation_l1586_158679

/-- Given two parabolas, prove that one is a translation of the other -/
theorem parabola_translation (x : ℝ) :
  (x^2 + 4*x + 5) = ((x + 2)^2 + 1) := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l1586_158679


namespace NUMINAMATH_CALUDE_ellipse_equation_proof_l1586_158654

/-- An ellipse with given properties -/
structure Ellipse where
  /-- First focus of the ellipse -/
  F₁ : ℝ × ℝ
  /-- Second focus of the ellipse -/
  F₂ : ℝ × ℝ
  /-- Length of the chord AB passing through F₂ and perpendicular to x-axis -/
  AB_length : ℝ
  /-- The first focus is at (-1, 0) -/
  F₁_constraint : F₁ = (-1, 0)
  /-- The second focus is at (1, 0) -/
  F₂_constraint : F₂ = (1, 0)
  /-- The length of AB is 3 -/
  AB_length_constraint : AB_length = 3

/-- The equation of the ellipse -/
def ellipse_equation (C : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Theorem stating that the given ellipse satisfies the equation x²/4 + y²/3 = 1 -/
theorem ellipse_equation_proof (C : Ellipse) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | ellipse_equation C p.1 p.2} ↔ 
  (x, y) ∈ {p : ℝ × ℝ | ∃ t : ℝ, p = (C.F₂.1, t) ∧ 
  abs (2 * t) = C.AB_length ∧ 
  (p.1 - C.F₁.1)^2 + p.2^2 + (p.1 - C.F₂.1)^2 + p.2^2 = 
  ((p.1 - C.F₁.1)^2 + p.2^2)^(1/2) + ((p.1 - C.F₂.1)^2 + p.2^2)^(1/2)} := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_proof_l1586_158654


namespace NUMINAMATH_CALUDE_ball_max_height_l1586_158617

/-- The height function of the ball's trajectory -/
def f (t : ℝ) : ℝ := -10 * t^2 + 50 * t - 24

/-- The maximum height reached by the ball -/
def max_height : ℝ := 38.5

theorem ball_max_height :
  IsGreatest { y | ∃ t, f t = y } max_height := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l1586_158617


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l1586_158656

theorem geometric_mean_of_4_and_9 :
  ∃ x : ℝ, x^2 = 4 * 9 ∧ (x = 6 ∨ x = -6) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l1586_158656


namespace NUMINAMATH_CALUDE_landscape_breadth_l1586_158678

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playground_area : ℝ

/-- The breadth is 6 times the length -/
def breadth_length_relation (l : Landscape) : Prop :=
  l.breadth = 6 * l.length

/-- The playground occupies 1/7th of the total landscape area -/
def playground_proportion (l : Landscape) : Prop :=
  l.playground_area = (1 / 7) * l.length * l.breadth

/-- The playground area is 4200 square meters -/
def playground_area_value (l : Landscape) : Prop :=
  l.playground_area = 4200

/-- Theorem: The breadth of the landscape is 420 meters -/
theorem landscape_breadth (l : Landscape) 
  (h1 : breadth_length_relation l)
  (h2 : playground_proportion l)
  (h3 : playground_area_value l) : 
  l.breadth = 420 := by sorry

end NUMINAMATH_CALUDE_landscape_breadth_l1586_158678


namespace NUMINAMATH_CALUDE_cubic_function_m_value_l1586_158669

theorem cubic_function_m_value (d e f g m : ℤ) :
  let g : ℝ → ℝ := λ x => (d : ℝ) * x^3 + (e : ℝ) * x^2 + (f : ℝ) * x + (g : ℝ)
  g 1 = 0 ∧
  70 < g 5 ∧ g 5 < 80 ∧
  120 < g 6 ∧ g 6 < 130 ∧
  10000 * (m : ℝ) < g 50 ∧ g 50 < 10000 * ((m + 1) : ℝ) →
  m = 12 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_m_value_l1586_158669


namespace NUMINAMATH_CALUDE_arithmetic_sequence_min_sum_l1586_158602

/-- An arithmetic sequence with common difference d, first term a₁, and sum function S_n -/
structure ArithmeticSequence where
  d : ℝ
  a₁ : ℝ
  S_n : ℕ → ℝ

/-- The sum of an arithmetic sequence reaches its minimum -/
def sum_reaches_minimum (seq : ArithmeticSequence) (n : ℕ) : Prop :=
  ∀ k : ℕ, seq.S_n k ≥ seq.S_n n

/-- Theorem: For an arithmetic sequence with non-zero common difference,
    negative first term, and S₇ = S₁₃, the sum reaches its minimum when n = 10 -/
theorem arithmetic_sequence_min_sum
  (seq : ArithmeticSequence)
  (h_d : seq.d ≠ 0)
  (h_a₁ : seq.a₁ < 0)
  (h_S : seq.S_n 7 = seq.S_n 13) :
  sum_reaches_minimum seq 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_min_sum_l1586_158602


namespace NUMINAMATH_CALUDE_river_travel_time_l1586_158605

structure RiverSystem where
  docks : Fin 3 → String
  distance : Fin 3 → Fin 3 → ℝ
  time_against_current : ℝ
  time_with_current : ℝ

def valid_river_system (rs : RiverSystem) : Prop :=
  (∀ i j, rs.distance i j = 3) ∧
  rs.time_against_current = 30 ∧
  rs.time_with_current = 18 ∧
  rs.time_against_current > rs.time_with_current

def travel_time (rs : RiverSystem) : Set ℝ :=
  {24, 72}

theorem river_travel_time (rs : RiverSystem) (h : valid_river_system rs) :
  ∀ i j, i ≠ j → (rs.distance i j / rs.time_against_current * 60 ∈ travel_time rs) ∨
                 (rs.distance i j / rs.time_with_current * 60 ∈ travel_time rs) :=
sorry

end NUMINAMATH_CALUDE_river_travel_time_l1586_158605


namespace NUMINAMATH_CALUDE_smallest_a_value_l1586_158615

theorem smallest_a_value (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0)
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (15 * ↑x)) :
  ∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (15 * ↑x)) → a' ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l1586_158615


namespace NUMINAMATH_CALUDE_wendy_miles_walked_l1586_158665

def pedometer_max : ℕ := 49999
def flips : ℕ := 60
def final_reading : ℕ := 25000
def steps_per_mile : ℕ := 1500

def total_steps : ℕ := (pedometer_max + 1) * flips + final_reading

def miles_walked : ℚ := total_steps / steps_per_mile

theorem wendy_miles_walked :
  ⌊(miles_walked + 50) / 100⌋ * 100 = 2000 :=
sorry

end NUMINAMATH_CALUDE_wendy_miles_walked_l1586_158665


namespace NUMINAMATH_CALUDE_apollo_pays_168_apples_l1586_158636

/-- Represents the number of months in a year --/
def months_in_year : ℕ := 12

/-- Represents Hephaestus's charging rate for the first half of the year --/
def hephaestus_rate_first_half : ℕ := 3

/-- Represents Hephaestus's charging rate for the second half of the year --/
def hephaestus_rate_second_half : ℕ := 2 * hephaestus_rate_first_half

/-- Represents Athena's charging rate for the entire year --/
def athena_rate : ℕ := 5

/-- Represents Ares's charging rate for the first 9 months --/
def ares_rate_first_nine : ℕ := 4

/-- Represents Ares's charging rate for the last 3 months --/
def ares_rate_last_three : ℕ := 6

/-- Calculates the total number of golden apples Apollo pays for a year --/
def total_golden_apples : ℕ :=
  (hephaestus_rate_first_half * 6 + hephaestus_rate_second_half * 6) +
  (athena_rate * months_in_year) +
  (ares_rate_first_nine * 9 + ares_rate_last_three * 3)

/-- Theorem stating that the total number of golden apples Apollo pays is 168 --/
theorem apollo_pays_168_apples : total_golden_apples = 168 := by
  sorry

end NUMINAMATH_CALUDE_apollo_pays_168_apples_l1586_158636


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1586_158649

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  (∃ b : ℝ, b ≠ 2 ∧ (b - 1) * (b - 2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1586_158649


namespace NUMINAMATH_CALUDE_blake_change_l1586_158663

/-- The amount Blake spends on oranges -/
def orange_cost : ℕ := 40

/-- The amount Blake spends on apples -/
def apple_cost : ℕ := 50

/-- The amount Blake spends on mangoes -/
def mango_cost : ℕ := 60

/-- The initial amount Blake has -/
def initial_amount : ℕ := 300

/-- The change given to Blake -/
def change : ℕ := initial_amount - (orange_cost + apple_cost + mango_cost)

theorem blake_change : change = 150 := by
  sorry

end NUMINAMATH_CALUDE_blake_change_l1586_158663


namespace NUMINAMATH_CALUDE_largest_integer_product_12_l1586_158652

theorem largest_integer_product_12 (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a * b * c * d * e = 12 →
  max a (max b (max c (max d e))) = 3 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_product_12_l1586_158652


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_36_l1586_158646

/-- An arithmetic sequence with sum Sₙ of the first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_36 (seq : ArithmeticSequence) 
  (h1 : 2 * seq.S 3 = 3 * seq.S 2 + 3)
  (h2 : seq.S 4 = seq.a 10) : 
  seq.S 36 = 666 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_36_l1586_158646


namespace NUMINAMATH_CALUDE_residue_of_7_1234_mod_19_l1586_158689

theorem residue_of_7_1234_mod_19 : 7^1234 % 19 = 9 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_7_1234_mod_19_l1586_158689


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l1586_158601

theorem quadratic_roots_sum_bound (a b : ℤ) 
  (ha : a ≠ -1) (hb : b ≠ -1) 
  (h_roots : ∃ x y : ℤ, x ≠ y ∧ 
    x^2 + a*b*x + (a + b) = 0 ∧ 
    y^2 + a*b*y + (a + b) = 0) : 
  a + b ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l1586_158601


namespace NUMINAMATH_CALUDE_pool_capacity_l1586_158693

theorem pool_capacity (C : ℝ) 
  (h1 : C / 4 - C / 6 = C / 12)  -- Net rate of water level change
  (h2 : C - 3 * (C / 12) = 90)   -- Remaining water after 3 hours
  : C = 120 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l1586_158693


namespace NUMINAMATH_CALUDE_nth_S_645_l1586_158614

/-- The set of positive integers with remainder 5 when divided by 8 -/
def S : Set ℕ := {n : ℕ | n > 0 ∧ n % 8 = 5}

/-- The nth element of S -/
def nth_S (n : ℕ) : ℕ := 8 * (n - 1) + 5

theorem nth_S_645 : nth_S 81 = 645 := by
  sorry

end NUMINAMATH_CALUDE_nth_S_645_l1586_158614


namespace NUMINAMATH_CALUDE_complex_power_modulus_l1586_158628

theorem complex_power_modulus : Complex.abs ((2 + Complex.I) ^ 8) = 625 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l1586_158628


namespace NUMINAMATH_CALUDE_race_speeds_l1586_158655

theorem race_speeds (x : ℝ) (h : x > 0) : 
  ∃ (a b : ℝ),
    (1000 = a * x) ∧ 
    (1000 - 167 = b * x) ∧
    (a = 1000 / x) ∧ 
    (b = 833 / x) := by
  sorry

end NUMINAMATH_CALUDE_race_speeds_l1586_158655


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l1586_158681

theorem reciprocal_sum_of_quadratic_roots :
  ∀ (α β : ℝ),
  (∃ (a b : ℝ), 7 * a^2 + 2 * a + 6 = 0 ∧ 
                 7 * b^2 + 2 * b + 6 = 0 ∧ 
                 α = 1 / a ∧ 
                 β = 1 / b) →
  α + β = -1/3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l1586_158681


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l1586_158692

/-- The time taken for a train to pass a jogger under specific conditions -/
theorem train_passing_jogger_time (jogger_speed train_speed : ℝ) 
  (initial_distance train_length : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 240 →
  train_length = 120 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l1586_158692


namespace NUMINAMATH_CALUDE_range_of_a_l1586_158694

def p (a : ℝ) : Prop := ∀ x, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.sqrt (a*x^2 - x + a)

theorem range_of_a :
  (∃ a, p a ∨ q a) ∧ (∀ a, ¬(p a ∧ q a)) →
  ∃ S : Set ℝ, S = {a | a ∈ (Set.Ioo 0 (1/2)) ∪ (Set.Ici 1)} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1586_158694


namespace NUMINAMATH_CALUDE_three_million_twenty_one_thousand_scientific_notation_l1586_158619

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem three_million_twenty_one_thousand_scientific_notation :
  toScientificNotation 3021000 = ScientificNotation.mk 3.021 6 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_three_million_twenty_one_thousand_scientific_notation_l1586_158619


namespace NUMINAMATH_CALUDE_car_costs_theorem_l1586_158640

def cost_of_old_car : ℝ := 1800
def cost_of_second_oldest_car : ℝ := 900
def cost_of_new_car : ℝ := 2 * cost_of_old_car
def sale_price_old_car : ℝ := 1800
def sale_price_second_oldest_car : ℝ := 900
def loan_amount : ℝ := cost_of_new_car - (sale_price_old_car + sale_price_second_oldest_car)
def annual_interest_rate : ℝ := 0.05
def years_passed : ℝ := 2
def remaining_debt : ℝ := 2000

theorem car_costs_theorem :
  cost_of_old_car = 1800 ∧
  cost_of_second_oldest_car = 900 ∧
  cost_of_new_car = 2 * cost_of_old_car ∧
  cost_of_new_car = 4 * cost_of_second_oldest_car ∧
  sale_price_old_car = 1800 ∧
  sale_price_second_oldest_car = 900 ∧
  loan_amount = cost_of_new_car - (sale_price_old_car + sale_price_second_oldest_car) ∧
  remaining_debt = 2000 :=
by sorry

end NUMINAMATH_CALUDE_car_costs_theorem_l1586_158640


namespace NUMINAMATH_CALUDE_cube_difference_divided_problem_solution_l1586_158633

theorem cube_difference_divided (a b : ℕ) (h : a > b) :
  (a^3 - b^3) / (a - b) = a^2 + a*b + b^2 :=
by sorry

theorem problem_solution : (64^3 - 27^3) / 37 = 6553 :=
by
  have h : 64 > 27 := by sorry
  have := cube_difference_divided 64 27 h
  sorry

end NUMINAMATH_CALUDE_cube_difference_divided_problem_solution_l1586_158633


namespace NUMINAMATH_CALUDE_triangle_angles_l1586_158670

/-- Given a triangle with sides a, b, and c, where a = b = 3 and c = √7 - √3,
    prove that the angles of the triangle are as follows:
    - Angle C (opposite side c) = arccos((4 + √21) / 9)
    - Angles A and B = (180° - arccos((4 + √21) / 9)) / 2 -/
theorem triangle_angles (a b c : ℝ) (h1 : a = 3) (h2 : b = 3) (h3 : c = Real.sqrt 7 - Real.sqrt 3) :
  let angle_c := Real.arccos ((4 + Real.sqrt 21) / 9)
  let angle_a := (π - angle_c) / 2
  ∃ (A B C : ℝ),
    A = angle_a ∧
    B = angle_a ∧
    C = angle_c ∧
    A + B + C = π :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_l1586_158670


namespace NUMINAMATH_CALUDE_best_marksman_score_l1586_158641

theorem best_marksman_score (team_size : ℕ) (hypothetical_score : ℕ) (hypothetical_average : ℕ) (actual_total : ℕ) : 
  team_size = 6 → 
  hypothetical_score = 92 →
  hypothetical_average = 84 →
  actual_total = 497 →
  ∃ (best_score : ℕ), best_score = 85 ∧ 
    actual_total = (team_size - 1) * hypothetical_average + best_score := by
  sorry

end NUMINAMATH_CALUDE_best_marksman_score_l1586_158641


namespace NUMINAMATH_CALUDE_polygon_area_is_12_l1586_158645

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The polygon defined by the given points -/
def polygon : List Point := [
  ⟨0, 0⟩, ⟨4, 0⟩, ⟨4, 4⟩, ⟨2, 4⟩, ⟨2, 2⟩, ⟨0, 2⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point) : ℝ := sorry

/-- Theorem: The area of the given polygon is 12 square units -/
theorem polygon_area_is_12 : polygonArea polygon = 12 := by sorry

end NUMINAMATH_CALUDE_polygon_area_is_12_l1586_158645


namespace NUMINAMATH_CALUDE_fifth_smallest_odd_with_four_prime_factors_l1586_158625

def has_at_least_four_prime_factors (n : ℕ) : Prop :=
  ∃ (p q r s : ℕ), Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ p * q * r * s ∣ n

def is_fifth_smallest (P : ℕ → Prop) (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), P a ∧ P b ∧ P c ∧ P d ∧
    a < b ∧ b < c ∧ c < d ∧ d < n ∧
    (∀ m, P m → m ≥ n ∨ m = a ∨ m = b ∨ m = c ∨ m = d)

theorem fifth_smallest_odd_with_four_prime_factors :
  is_fifth_smallest (λ n => Odd n ∧ has_at_least_four_prime_factors n) 1925 :=
sorry

end NUMINAMATH_CALUDE_fifth_smallest_odd_with_four_prime_factors_l1586_158625


namespace NUMINAMATH_CALUDE_tau_phi_sum_equation_l1586_158696

/-- τ(n) represents the number of positive divisors of n -/
def tau (n : ℕ) : ℕ := sorry

/-- φ(n) represents the number of positive integers less than n and relatively prime to n -/
def phi (n : ℕ) : ℕ := sorry

/-- A predicate to check if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

theorem tau_phi_sum_equation (n : ℕ) (h : n > 1) :
  tau n + phi n = n + 1 ↔ n = 4 ∨ isPrime n := by sorry

end NUMINAMATH_CALUDE_tau_phi_sum_equation_l1586_158696


namespace NUMINAMATH_CALUDE_fraction_calculation_l1586_158664

theorem fraction_calculation (x y : ℚ) (hx : x = 4/7) (hy : y = 5/8) :
  (7*x + 5*y) / (70*x*y) = 57/400 := by
sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1586_158664


namespace NUMINAMATH_CALUDE_existence_of_no_seven_multiple_l1586_158634

/-- Function to check if a natural number contains the digit 7 in its decimal representation -/
def containsSeven (n : ℕ) : Prop := sorry

/-- Function to generate the sequence of numbers obtained by multiplying by 5 k times -/
def multiplyByFive (n : ℕ) (k : ℕ) : List ℕ := sorry

/-- Function to generate the sequence of numbers obtained by multiplying by 2 k times -/
def multiplyByTwo (n : ℕ) (k : ℕ) : List ℕ := sorry

/-- Theorem stating the existence of a number that can be multiplied by 2 k times
    without producing a number containing 7, given a number that can be multiplied
    by 5 k times without producing a number containing 7 -/
theorem existence_of_no_seven_multiple (n : ℕ) (k : ℕ) :
  (∀ m ∈ multiplyByFive n k, ¬containsSeven m) →
  ∃ m : ℕ, ∀ p ∈ multiplyByTwo m k, ¬containsSeven p :=
by sorry

end NUMINAMATH_CALUDE_existence_of_no_seven_multiple_l1586_158634


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1586_158643

theorem arithmetic_expression_equality : 2 + 3 * 4 - 5 + 6 * (2 - 1) = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1586_158643


namespace NUMINAMATH_CALUDE_function_root_implies_a_range_l1586_158661

theorem function_root_implies_a_range (a : ℝ) :
  (∃ x₀ : ℝ, -1 < x₀ ∧ x₀ < 1 ∧ 3 * a * x₀ - 2 * a + 1 = 0) →
  a < -1 ∨ a > 1/5 :=
by sorry

end NUMINAMATH_CALUDE_function_root_implies_a_range_l1586_158661


namespace NUMINAMATH_CALUDE_recurring_decimal_subtraction_l1586_158653

theorem recurring_decimal_subtraction : 
  (246 : ℚ) / 999 - 135 / 999 - 579 / 999 = -52 / 111 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_subtraction_l1586_158653


namespace NUMINAMATH_CALUDE_simon_received_stamps_l1586_158695

/-- The number of stamps Simon received from his friends -/
def stamps_received (initial_stamps current_stamps : ℕ) : ℕ :=
  current_stamps - initial_stamps

/-- Theorem stating that Simon received 27 stamps from his friends -/
theorem simon_received_stamps :
  stamps_received 34 61 = 27 := by
  sorry

end NUMINAMATH_CALUDE_simon_received_stamps_l1586_158695


namespace NUMINAMATH_CALUDE_seed_mixture_percentage_l1586_158609

/-- Given two seed mixtures X and Y, and a final mixture composed of X and Y,
    this theorem proves the percentage of X in the final mixture. -/
theorem seed_mixture_percentage
  (x_ryegrass : Real) (x_bluegrass : Real)
  (y_ryegrass : Real) (y_fescue : Real)
  (final_ryegrass : Real) :
  x_ryegrass = 0.40 →
  x_bluegrass = 0.60 →
  y_ryegrass = 0.25 →
  y_fescue = 0.75 →
  final_ryegrass = 0.27 →
  x_ryegrass + x_bluegrass = 1 →
  y_ryegrass + y_fescue = 1 →
  ∃ (p : Real), p * x_ryegrass + (1 - p) * y_ryegrass = final_ryegrass ∧ p = 200 / 15 := by
  sorry

end NUMINAMATH_CALUDE_seed_mixture_percentage_l1586_158609


namespace NUMINAMATH_CALUDE_divide_subtract_problem_l1586_158629

theorem divide_subtract_problem (x : ℝ) : 
  (990 / x) - 100 = 10 → x = 9 := by sorry

end NUMINAMATH_CALUDE_divide_subtract_problem_l1586_158629


namespace NUMINAMATH_CALUDE_pentagon_angles_count_l1586_158662

/-- Represents a sequence of 5 interior angles of a convex pentagon --/
structure PentagonAngles where
  angles : Fin 5 → ℕ
  sum_540 : (angles 0) + (angles 1) + (angles 2) + (angles 3) + (angles 4) = 540
  increasing : ∀ i j, i < j → angles i < angles j
  smallest_ge_60 : angles 0 ≥ 60
  largest_lt_150 : angles 4 < 150
  arithmetic : ∃ d : ℕ, ∀ i : Fin 4, angles (i + 1) = angles i + d
  not_equiangular : ¬ (∀ i j, angles i = angles j)

/-- The number of valid PentagonAngles --/
def validPentagonAnglesCount : ℕ := 5

theorem pentagon_angles_count :
  {s : Finset PentagonAngles | s.card = validPentagonAnglesCount} ≠ ∅ :=
sorry

end NUMINAMATH_CALUDE_pentagon_angles_count_l1586_158662


namespace NUMINAMATH_CALUDE_perfect_cubes_with_special_property_l1586_158603

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def last_three_digits (n : ℕ) : ℕ := n % 1000

def erase_last_three_digits (n : ℕ) : ℕ := n / 1000

theorem perfect_cubes_with_special_property :
  ∀ n : ℕ,
    n > 0 ∧
    is_perfect_cube n ∧
    n % 10 ≠ 0 ∧
    is_perfect_cube (erase_last_three_digits n) →
    n = 1331 ∨ n = 1728 :=
by sorry

end NUMINAMATH_CALUDE_perfect_cubes_with_special_property_l1586_158603


namespace NUMINAMATH_CALUDE_apple_pie_apples_apple_pie_theorem_l1586_158638

theorem apple_pie_apples (total_greg_sarah : ℕ) (susan_multiplier : ℕ) (mark_difference : ℕ) (mom_leftover : ℕ) : ℕ :=
  let greg_apples := total_greg_sarah / 2
  let susan_apples := greg_apples * susan_multiplier
  let mark_apples := susan_apples - mark_difference
  let pie_apples := susan_apples - mom_leftover
  pie_apples

theorem apple_pie_theorem :
  apple_pie_apples 18 2 5 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_apples_apple_pie_theorem_l1586_158638


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1586_158610

theorem sufficient_not_necessary (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y, x - y ≥ 2 → 2^x - 2^y ≥ 3) ∧
  (∃ x y, 2^x - 2^y ≥ 3 ∧ x - y < 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1586_158610


namespace NUMINAMATH_CALUDE_not_diff_of_squares_l1586_158620

theorem not_diff_of_squares (a : ℤ) : ¬ ∃ (x y : ℤ), 4 * a + 2 = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_not_diff_of_squares_l1586_158620


namespace NUMINAMATH_CALUDE_garage_sale_items_count_l1586_158668

theorem garage_sale_items_count (prices : Finset ℕ) (radio_price : ℕ) : 
  radio_price ∈ prices →
  (prices.filter (λ x => x > radio_price)).card = 15 →
  (prices.filter (λ x => x < radio_price)).card = 22 →
  prices.card = 38 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_items_count_l1586_158668


namespace NUMINAMATH_CALUDE_alice_unanswered_questions_l1586_158691

/-- Represents a scoring system for a test --/
structure ScoringSystem where
  startPoints : ℤ
  correctPoints : ℤ
  wrongPoints : ℤ
  unansweredPoints : ℤ

/-- Calculates the score based on a scoring system and the number of correct, wrong, and unanswered questions --/
def calculateScore (system : ScoringSystem) (correct wrong unanswered : ℤ) : ℤ :=
  system.startPoints + system.correctPoints * correct + system.wrongPoints * wrong + system.unansweredPoints * unanswered

theorem alice_unanswered_questions : ∃ (correct wrong unanswered : ℤ),
  let newSystem : ScoringSystem := ⟨0, 6, 0, 3⟩
  let oldSystem : ScoringSystem := ⟨50, 5, -2, 0⟩
  let hypotheticalSystem : ScoringSystem := ⟨40, 7, -1, -1⟩
  correct + wrong + unanswered = 25 ∧
  calculateScore newSystem correct wrong unanswered = 130 ∧
  calculateScore oldSystem correct wrong unanswered = 100 ∧
  calculateScore hypotheticalSystem correct wrong unanswered = 120 ∧
  unanswered = 20 := by
  sorry

#check alice_unanswered_questions

end NUMINAMATH_CALUDE_alice_unanswered_questions_l1586_158691


namespace NUMINAMATH_CALUDE_peri_arrival_day_l1586_158686

def travel_pattern (day : ℕ) : ℕ :=
  if day % 10 = 0 then 0 else 1

def total_distance (n : ℕ) : ℕ :=
  (List.range n).map travel_pattern |> List.sum

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed - 1) % 7 + 1

theorem peri_arrival_day :
  ∃ (n : ℕ), total_distance n = 90 ∧ day_of_week 1 n = 2 :=
sorry

end NUMINAMATH_CALUDE_peri_arrival_day_l1586_158686


namespace NUMINAMATH_CALUDE_cathys_money_ratio_is_two_to_one_l1586_158677

/-- The ratio of the amount Cathy's mom sent her to the amount her dad sent her -/
def cathys_money_ratio (initial_amount dad_amount mom_amount final_amount : ℚ) : ℚ :=
  mom_amount / dad_amount

/-- Proves that the ratio of the amount Cathy's mom sent her to the amount her dad sent her is 2:1 -/
theorem cathys_money_ratio_is_two_to_one 
  (initial_amount : ℚ) 
  (dad_amount : ℚ) 
  (mom_amount : ℚ) 
  (final_amount : ℚ) 
  (h1 : initial_amount = 12)
  (h2 : dad_amount = 25)
  (h3 : final_amount = 87)
  (h4 : initial_amount + dad_amount + mom_amount = final_amount) :
  cathys_money_ratio initial_amount dad_amount mom_amount final_amount = 2 := by
sorry

#eval cathys_money_ratio 12 25 50 87

end NUMINAMATH_CALUDE_cathys_money_ratio_is_two_to_one_l1586_158677


namespace NUMINAMATH_CALUDE_rainville_total_rainfall_2007_l1586_158667

/-- Calculates the total rainfall for a year given the average monthly rainfall -/
def total_rainfall (average_monthly_rainfall : ℝ) : ℝ :=
  average_monthly_rainfall * 12

/-- Represents the rainfall data for Rainville from 2005 to 2007 -/
structure RainvilleRainfall where
  rainfall_2005 : ℝ
  rainfall_increase_2006 : ℝ
  rainfall_increase_2007 : ℝ

/-- Theorem stating the total rainfall in Rainville for 2007 -/
theorem rainville_total_rainfall_2007 (data : RainvilleRainfall) 
  (h1 : data.rainfall_2005 = 50)
  (h2 : data.rainfall_increase_2006 = 3)
  (h3 : data.rainfall_increase_2007 = 5) :
  total_rainfall (data.rainfall_2005 + data.rainfall_increase_2006 + data.rainfall_increase_2007) = 696 :=
by sorry

end NUMINAMATH_CALUDE_rainville_total_rainfall_2007_l1586_158667


namespace NUMINAMATH_CALUDE_max_sum_with_reciprocals_l1586_158632

theorem max_sum_with_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y = 5) : 
  x + y ≤ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b + 1/a + 1/b = 5 ∧ a + b = 4 :=
sorry

end NUMINAMATH_CALUDE_max_sum_with_reciprocals_l1586_158632


namespace NUMINAMATH_CALUDE_f_inequality_l1586_158648

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

theorem f_inequality : f (Real.sqrt 3 / 2) > f (Real.sqrt 6 / 2) ∧ f (Real.sqrt 6 / 2) > f (Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1586_158648


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1586_158658

/-- 
Given a quadratic equation bx^2 + 6x + d = 0 with exactly one solution,
where b + d = 7 and b < d, prove that b = (7 - √13) / 2 and d = (7 + √13) / 2
-/
theorem unique_quadratic_solution (b d : ℝ) : 
  (∃! x, b * x^2 + 6 * x + d = 0) →
  b + d = 7 →
  b < d →
  b = (7 - Real.sqrt 13) / 2 ∧ d = (7 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1586_158658


namespace NUMINAMATH_CALUDE_james_weekday_coffees_l1586_158666

/-- Represents the number of weekdays in a week -/
def weekdays : Nat := 5

/-- Represents the number of weekend days in a week -/
def weekend_days : Nat := 2

/-- Cost of a donut in cents -/
def donut_cost : Nat := 60

/-- Cost of a coffee in cents -/
def coffee_cost : Nat := 90

/-- Calculates the total cost for the week in cents -/
def total_cost (weekday_coffees : Nat) : Nat :=
  let weekday_donuts := weekdays - weekday_coffees
  let weekday_cost := weekday_coffees * coffee_cost + weekday_donuts * donut_cost
  let weekend_cost := weekend_days * (coffee_cost + donut_cost)
  weekday_cost + weekend_cost

theorem james_weekday_coffees :
  ∃ (weekday_coffees : Nat),
    weekday_coffees ≤ weekdays ∧
    (∃ (k : Nat), total_cost weekday_coffees = k * 100) ∧
    weekday_coffees = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_weekday_coffees_l1586_158666


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1586_158672

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_eq : a 4 * a 6 + a 5 ^ 2 = 50) :
  a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1586_158672


namespace NUMINAMATH_CALUDE_barrels_of_pitch_day4_is_two_l1586_158631

/-- Represents the roadwork company's paving project --/
structure RoadworkProject where
  total_length : ℕ
  gravel_per_truck : ℕ
  gravel_to_pitch_ratio : ℕ
  day1_truckloads_per_mile : ℕ
  day1_miles_paved : ℕ
  day2_truckloads_per_mile : ℕ
  day2_miles_paved : ℕ
  day3_truckloads_per_mile : ℕ
  day3_miles_paved : ℕ
  day4_truckloads_per_mile : ℕ

/-- Calculates the number of barrels of pitch needed for the fourth day --/
def barrels_of_pitch_day4 (project : RoadworkProject) : ℕ :=
  let remaining_miles := project.total_length - (project.day1_miles_paved + project.day2_miles_paved + project.day3_miles_paved)
  let day4_truckloads := remaining_miles * project.day4_truckloads_per_mile
  let pitch_per_truck := project.gravel_per_truck / project.gravel_to_pitch_ratio
  let total_pitch := day4_truckloads * pitch_per_truck
  (total_pitch + 9) / 10  -- Round up to the nearest whole barrel

/-- Theorem stating that the number of barrels of pitch needed for the fourth day is 2 --/
theorem barrels_of_pitch_day4_is_two (project : RoadworkProject) 
  (h1 : project.total_length = 20)
  (h2 : project.gravel_per_truck = 2)
  (h3 : project.gravel_to_pitch_ratio = 5)
  (h4 : project.day1_truckloads_per_mile = 3)
  (h5 : project.day1_miles_paved = 4)
  (h6 : project.day2_truckloads_per_mile = 4)
  (h7 : project.day2_miles_paved = 7)
  (h8 : project.day3_truckloads_per_mile = 2)
  (h9 : project.day3_miles_paved = 5)
  (h10 : project.day4_truckloads_per_mile = 1) :
  barrels_of_pitch_day4 project = 2 := by
  sorry

end NUMINAMATH_CALUDE_barrels_of_pitch_day4_is_two_l1586_158631


namespace NUMINAMATH_CALUDE_unique_number_251_l1586_158606

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 251 is the unique positive integer whose product with its sum of digits is 2008 -/
theorem unique_number_251 : ∃! (n : ℕ), n > 0 ∧ n * sum_of_digits n = 2008 :=
  sorry

end NUMINAMATH_CALUDE_unique_number_251_l1586_158606


namespace NUMINAMATH_CALUDE_decompose_6058_l1586_158699

theorem decompose_6058 : 6058 = 6 * 1000 + 5 * 10 + 8 * 1 := by
  sorry

end NUMINAMATH_CALUDE_decompose_6058_l1586_158699


namespace NUMINAMATH_CALUDE_circle_center_and_sum_l1586_158660

/-- Given a circle described by the equation x^2 + y^2 = 4x - 2y + 10,
    prove that its center is at (2, -1) and the sum of the center's coordinates is 1. -/
theorem circle_center_and_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 2*y + 10) → 
  (∃ (center_x center_y : ℝ), 
    center_x = 2 ∧ 
    center_y = -1 ∧ 
    (x - center_x)^2 + (y - center_y)^2 = 15 ∧
    center_x + center_y = 1) := by
  sorry


end NUMINAMATH_CALUDE_circle_center_and_sum_l1586_158660


namespace NUMINAMATH_CALUDE_bouquet_cost_is_45_l1586_158680

/-- The cost of a bouquet consisting of two dozens of red roses and 3 sunflowers -/
def bouquet_cost (rose_price sunflower_price : ℚ) : ℚ :=
  (24 * rose_price) + (3 * sunflower_price)

/-- Theorem stating that the cost of the bouquet with given prices is $45 -/
theorem bouquet_cost_is_45 :
  bouquet_cost (3/2) 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_cost_is_45_l1586_158680


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1586_158651

/-- Given a geometric sequence with first term a₁ = 1, 
    the minimum value of 3a₂ + 7a₃ is -27/28 -/
theorem min_value_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 1 = 1 →                     -- first term is 1
  ∃ m : ℝ, m = -27/28 ∧ ∀ r : ℝ, 3 * (a 2) + 7 * (a 3) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1586_158651


namespace NUMINAMATH_CALUDE_james_pistachio_expenditure_l1586_158682

/-- Calculates the weekly expenditure on pistachios given the cost per can, ounces per can, daily consumption, and days of consumption. -/
def weekly_pistachio_expenditure (cost_per_can : ℚ) (ounces_per_can : ℚ) (ounces_consumed : ℚ) (days_consumed : ℕ) : ℚ :=
  let weekly_consumption := (7 : ℚ) / days_consumed * ounces_consumed
  let cans_needed := (weekly_consumption / ounces_per_can).ceil
  cans_needed * cost_per_can

/-- Theorem stating that James' weekly expenditure on pistachios is $90. -/
theorem james_pistachio_expenditure :
  weekly_pistachio_expenditure 10 5 30 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_james_pistachio_expenditure_l1586_158682


namespace NUMINAMATH_CALUDE_jacqueline_guavas_l1586_158647

theorem jacqueline_guavas (plums apples given_away left : ℕ) (guavas : ℕ) : 
  plums = 16 → 
  apples = 21 → 
  given_away = 40 → 
  left = 15 → 
  plums + guavas + apples = given_away + left → 
  guavas = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_jacqueline_guavas_l1586_158647


namespace NUMINAMATH_CALUDE_equation_solution_l1586_158659

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (4 * (x₁ - 1)^2 = 36 ∧ 4 * (x₂ - 1)^2 = 36) ∧ 
  x₁ = 4 ∧ x₂ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1586_158659


namespace NUMINAMATH_CALUDE_spatial_relationships_l1586_158673

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (parallelLines : Line → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem spatial_relationships 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : contains β m) :
  (parallel α β → perpendicularLines l m) ∧ 
  (parallelLines l m → perpendicularPlanes α β) := by
  sorry

end NUMINAMATH_CALUDE_spatial_relationships_l1586_158673


namespace NUMINAMATH_CALUDE_hadley_walk_l1586_158637

/-- Hadley's walk problem -/
theorem hadley_walk (x : ℝ) :
  (x ≥ 0) →
  (x - 1 ≥ 0) →
  (x + (x - 1) + 3 = 6) →
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_hadley_walk_l1586_158637


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l1586_158607

theorem complex_magnitude_equation (s : ℝ) (hs : s > 0) :
  Complex.abs (3 + s * Complex.I) = 13 → s = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l1586_158607


namespace NUMINAMATH_CALUDE_problem_statement_l1586_158604

theorem problem_statement (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) :
  (1/3) * x^8 * y^9 = 2/5 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1586_158604


namespace NUMINAMATH_CALUDE_abs_eq_neg_self_iff_nonpositive_l1586_158627

theorem abs_eq_neg_self_iff_nonpositive (x : ℝ) : |x| = -x ↔ x ≤ 0 := by sorry

end NUMINAMATH_CALUDE_abs_eq_neg_self_iff_nonpositive_l1586_158627


namespace NUMINAMATH_CALUDE_point_transformation_l1586_158657

/-- Rotate a point (x, y) counterclockwise by 90° around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

/-- Reflect a point (x, y) about the line y = x -/
def reflectAboutYEqX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (c d : ℝ) :
  let (x₁, y₁) := rotate90 c d 2 3
  let (x₂, y₂) := reflectAboutYEqX x₁ y₁
  (x₂ = -3 ∧ y₂ = 8) → d - c = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1586_158657


namespace NUMINAMATH_CALUDE_arrangement_count_l1586_158698

def number_of_arrangements (n_male n_female : ℕ) : ℕ :=
  sorry

theorem arrangement_count :
  let n_male := 2
  let n_female := 3
  number_of_arrangements n_male n_female = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l1586_158698


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l1586_158671

/-- The total land owned by the farmer in acres -/
def total_land : ℝ := 6000

/-- The proportion of land cleared for planting -/
def cleared_proportion : ℝ := 0.90

/-- The proportion of cleared land planted with soybeans -/
def soybean_proportion : ℝ := 0.30

/-- The proportion of cleared land planted with wheat -/
def wheat_proportion : ℝ := 0.60

/-- The amount of cleared land planted with corn in acres -/
def corn_land : ℝ := 540

theorem farmer_land_ownership :
  total_land * cleared_proportion * (1 - soybean_proportion - wheat_proportion) = corn_land :=
by sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l1586_158671


namespace NUMINAMATH_CALUDE_spheres_in_base_of_pyramid_l1586_158697

/-- The number of spheres in a regular triangular pyramid with n levels -/
def triangular_pyramid (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem spheres_in_base_of_pyramid (total_spheres : ℕ) (h : total_spheres = 165) :
  ∃ n : ℕ, triangular_pyramid n = total_spheres ∧ triangular_number n = 45 := by
  sorry

end NUMINAMATH_CALUDE_spheres_in_base_of_pyramid_l1586_158697


namespace NUMINAMATH_CALUDE_book_page_numbering_l1586_158683

/-- The total number of digits used to number pages in a book -/
def total_digits (n : ℕ) : ℕ :=
  let single_digit := min n 9
  let double_digit := max 0 (min n 99 - 9)
  let triple_digit := max 0 (n - 99)
  single_digit + 2 * double_digit + 3 * triple_digit

/-- Theorem stating that a book with 266 pages uses 690 digits for page numbering -/
theorem book_page_numbering :
  total_digits 266 = 690 := by
  sorry

end NUMINAMATH_CALUDE_book_page_numbering_l1586_158683


namespace NUMINAMATH_CALUDE_grid_diagonal_property_l1586_158624

/-- Represents a cell color in the grid -/
inductive Color
| Black
| White

/-- Represents a 100 x 100 grid -/
def Grid := Fin 100 → Fin 100 → Color

/-- A predicate that checks if a cell is on the boundary of the grid -/
def isBoundary (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- A predicate that checks if a 2x2 subgrid is monochromatic -/
def isMonochromatic (g : Grid) (i j : Fin 100) : Prop :=
  g i j = g (i+1) j ∧ g i j = g i (j+1) ∧ g i j = g (i+1) (j+1)

/-- A predicate that checks if a 2x2 subgrid has the desired diagonal property -/
def hasDiagonalProperty (g : Grid) (i j : Fin 100) : Prop :=
  (g i j = g (i+1) (j+1) ∧ g i (j+1) = g (i+1) j ∧ g i j ≠ g i (j+1))
  ∨ (g i j = g (i+1) (j+1) ∧ g i (j+1) = g (i+1) j ∧ g i (j+1) ≠ g i j)

theorem grid_diagonal_property (g : Grid) 
  (boundary_black : ∀ i j, isBoundary i j → g i j = Color.Black)
  (no_monochromatic : ∀ i j, ¬isMonochromatic g i j) :
  ∃ i j, hasDiagonalProperty g i j := by
  sorry

end NUMINAMATH_CALUDE_grid_diagonal_property_l1586_158624


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l1586_158630

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l1586_158630

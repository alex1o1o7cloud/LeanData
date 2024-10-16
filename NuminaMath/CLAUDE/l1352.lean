import Mathlib

namespace NUMINAMATH_CALUDE_different_color_chips_probability_l1352_135277

theorem different_color_chips_probability :
  let total_chips : ℕ := 9
  let blue_chips : ℕ := 6
  let yellow_chips : ℕ := 3
  let prob_blue_then_yellow : ℚ := (blue_chips / total_chips) * (yellow_chips / (total_chips - 1))
  let prob_yellow_then_blue : ℚ := (yellow_chips / total_chips) * (blue_chips / (total_chips - 1))
  prob_blue_then_yellow + prob_yellow_then_blue = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l1352_135277


namespace NUMINAMATH_CALUDE_minimum_cost_l1352_135236

/-- A function representing the total cost of purchasing basketballs and soccer balls -/
def total_cost (m : ℝ) : ℝ := 32 * m + 1440

/-- The constraint that the number of basketballs is at least twice the number of soccer balls -/
def basketball_constraint (m : ℝ) : Prop := m ≥ 2 * (30 - m)

theorem minimum_cost :
  ∀ m : ℝ, 0 < m → m < 30 → basketball_constraint m →
  total_cost m ≥ total_cost 20 ∧ total_cost 20 = 2080 := by
  sorry

#check minimum_cost

end NUMINAMATH_CALUDE_minimum_cost_l1352_135236


namespace NUMINAMATH_CALUDE_solution_set_f_geq_2_max_a_value_l1352_135285

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x - 1|

-- Theorem for the solution set of f(x) ≥ 2
theorem solution_set_f_geq_2 :
  {x : ℝ | f x ≥ 2} = {x : ℝ | x ≤ 0 ∨ x ≥ 4/3} :=
sorry

-- Theorem for the maximum value of a
theorem max_a_value (a : ℝ) :
  (∀ x : ℝ, f x ≥ a * |x|) ↔ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_2_max_a_value_l1352_135285


namespace NUMINAMATH_CALUDE_apple_distribution_l1352_135231

theorem apple_distribution (total_apples : ℕ) (num_people : ℕ) (apples_per_person : ℕ) :
  total_apples = 15 →
  num_people = 3 →
  apples_per_person * num_people ≤ total_apples →
  apples_per_person = total_apples / num_people →
  apples_per_person = 5 :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l1352_135231


namespace NUMINAMATH_CALUDE_units_digit_G_500_l1352_135267

-- Define the sequence G_n
def G (n : ℕ) : ℕ := 2^(3^n) + 1

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_G_500 : unitsDigit (G 500) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_500_l1352_135267


namespace NUMINAMATH_CALUDE_multiplication_scheme_solution_l1352_135230

def is_valid_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

theorem multiplication_scheme_solution :
  ∀ (A B C D E F G H I K L M N P : ℕ),
    is_valid_digit A →
    is_valid_digit B →
    is_valid_digit C →
    is_valid_digit D →
    is_valid_digit E →
    is_valid_digit G →
    is_valid_digit H →
    is_valid_digit I →
    is_valid_digit K →
    is_valid_digit L →
    is_valid_digit N →
    is_valid_digit P →
    C = D →
    A = B →
    K = L →
    F = 0 →
    M = 0 →
    I = E →
    H = E →
    P = A →
    N = A →
    (A * 10 + B) * (C * 10 + D) = E * 100 + F * 10 + G →
    (C * 10 + G) * (K * 10 + L) = A * 100 + M * 10 + C →
    A = 7 ∧ B = 7 ∧ C = 4 ∧ D = 4 ∧ E = 3 ∧ G = 8 ∧ K = 8 ∧ L = 8 :=
by sorry

#check multiplication_scheme_solution

end NUMINAMATH_CALUDE_multiplication_scheme_solution_l1352_135230


namespace NUMINAMATH_CALUDE_equation_solution_l1352_135240

theorem equation_solution : ∃ x : ℚ, x * (-1/2) = 1 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1352_135240


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1352_135203

/-- A sequence a : ℕ → ℝ is geometric if there exists a common ratio r such that
    aₙ₊₁ = r * aₙ for all n. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : IsGeometric a) :
  a 1 * a 2 * a 3 = -8 → a 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1352_135203


namespace NUMINAMATH_CALUDE_polynomial_property_l1352_135227

/-- A polynomial of the form x^2 + bx + c -/
def P (b c : ℝ) (x : ℝ) : ℝ := x^2 + b * x + c

/-- Theorem stating that if P(P(1)) = 0, P(P(-2)) = 0, and P(1) ≠ P(-2), then P(0) = -5/2 -/
theorem polynomial_property (b c : ℝ) :
  (P b c (P b c 1) = 0) →
  (P b c (P b c (-2)) = 0) →
  (P b c 1 ≠ P b c (-2)) →
  P b c 0 = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_property_l1352_135227


namespace NUMINAMATH_CALUDE_smallest_n_value_l1352_135268

theorem smallest_n_value (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ (21 * N)) : 80 ≤ N := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_value_l1352_135268


namespace NUMINAMATH_CALUDE_line_plane_relationship_l1352_135297

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (α : Plane) 
  (h1 : perpendicular a b) 
  (h2 : parallel_line_plane a α) : 
  intersects b α ∨ contained_in b α ∨ parallel_line_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l1352_135297


namespace NUMINAMATH_CALUDE_correct_geometry_problems_l1352_135251

theorem correct_geometry_problems (total_problems : ℕ) (total_algebra : ℕ) 
  (algebra_correct_ratio : ℚ) (algebra_incorrect_ratio : ℚ)
  (geometry_correct_ratio : ℚ) (geometry_incorrect_ratio : ℚ) :
  total_problems = 60 →
  total_algebra = 25 →
  algebra_correct_ratio = 3 →
  algebra_incorrect_ratio = 2 →
  geometry_correct_ratio = 4 →
  geometry_incorrect_ratio = 1 →
  ∃ (correct_geometry : ℕ), correct_geometry = 28 ∧
    correct_geometry * (geometry_correct_ratio + geometry_incorrect_ratio) = 
    (total_problems - total_algebra) * geometry_correct_ratio :=
by sorry

end NUMINAMATH_CALUDE_correct_geometry_problems_l1352_135251


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l1352_135244

theorem complex_magnitude_example : Complex.abs (-5 + (8/3) * Complex.I) = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l1352_135244


namespace NUMINAMATH_CALUDE_no_prime_sum_53_l1352_135210

theorem no_prime_sum_53 : ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ p + q = 53 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_53_l1352_135210


namespace NUMINAMATH_CALUDE_messenger_speed_l1352_135281

/-- Proves that the messenger's speed is 25 km/h given the problem conditions -/
theorem messenger_speed (team_length : ℝ) (team_speed : ℝ) (journey_time : ℝ)
  (h1 : team_length = 6)
  (h2 : team_speed = 5)
  (h3 : journey_time = 0.5)
  (h4 : ∀ x : ℝ, x > team_speed → team_length / (x + team_speed) + team_length / (x - team_speed) = journey_time → x = 25) :
  ∃ x : ℝ, x > team_speed ∧ team_length / (x + team_speed) + team_length / (x - team_speed) = journey_time ∧ x = 25 :=
by sorry

end NUMINAMATH_CALUDE_messenger_speed_l1352_135281


namespace NUMINAMATH_CALUDE_price_increase_problem_l1352_135248

theorem price_increase_problem (candy_initial : ℝ) (soda_initial : ℝ) 
  (candy_increase : ℝ) (soda_increase : ℝ) 
  (h1 : candy_initial = 20) 
  (h2 : soda_initial = 6) 
  (h3 : candy_increase = 0.25) 
  (h4 : soda_increase = 0.50) : 
  candy_initial + soda_initial = 26 := by
  sorry

#check price_increase_problem

end NUMINAMATH_CALUDE_price_increase_problem_l1352_135248


namespace NUMINAMATH_CALUDE_cone_volume_l1352_135226

/-- Given a cone with slant height 3 and lateral surface area 3π, its volume is (2√2π)/3 -/
theorem cone_volume (l : ℝ) (A_L : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) : 
  l = 3 →
  A_L = 3 * Real.pi →
  A_L = Real.pi * r * l →
  l^2 = h^2 + r^2 →
  V = (1/3) * Real.pi * r^2 * h →
  V = (2 * Real.sqrt 2 * Real.pi) / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_volume_l1352_135226


namespace NUMINAMATH_CALUDE_height_prediction_approximate_l1352_135221

/-- Represents a linear regression model for height prediction -/
structure HeightModel where
  slope : ℝ
  intercept : ℝ

/-- Predicts height based on the model and age -/
def predict_height (model : HeightModel) (age : ℝ) : ℝ :=
  model.slope * age + model.intercept

/-- The given height prediction model -/
def given_model : HeightModel := { slope := 7.19, intercept := 73.93 }

/-- Theorem stating that the predicted height at age 10 is approximately 145.83cm -/
theorem height_prediction_approximate :
  ∃ ε > 0, ∀ δ > 0, δ < ε → 
    |predict_height given_model 10 - 145.83| < δ :=
sorry

end NUMINAMATH_CALUDE_height_prediction_approximate_l1352_135221


namespace NUMINAMATH_CALUDE_smiths_class_a_students_l1352_135279

theorem smiths_class_a_students (johnson_total : ℕ) (johnson_a : ℕ) (smith_total : ℕ) :
  johnson_total = 20 →
  johnson_a = 12 →
  smith_total = 30 →
  (johnson_a : ℚ) / johnson_total = (smith_a : ℚ) / smith_total →
  smith_a = 18 :=
by
  sorry
where
  smith_a : ℕ := sorry

end NUMINAMATH_CALUDE_smiths_class_a_students_l1352_135279


namespace NUMINAMATH_CALUDE_correct_calculation_l1352_135265

theorem correct_calculation (x : ℕ) (h : x + 12 = 48) : x + 22 = 58 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1352_135265


namespace NUMINAMATH_CALUDE_integer_root_iff_a_value_l1352_135225

def polynomial (a x : ℤ) : ℤ := x^4 + 4*x^3 + a*x^2 + 8

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, polynomial a x = 0

theorem integer_root_iff_a_value :
  ∀ a : ℤ, has_integer_root a ↔ a = -14 ∨ a = -13 ∨ a = -5 ∨ a = 2 :=
sorry

end NUMINAMATH_CALUDE_integer_root_iff_a_value_l1352_135225


namespace NUMINAMATH_CALUDE_raisin_distribution_l1352_135284

/-- The number of raisins Bryce received -/
def bryce_raisins : ℕ := 15

/-- The number of raisins Carter received -/
def carter_raisins : ℕ := bryce_raisins - 10

theorem raisin_distribution : 
  (bryce_raisins = 15) ∧ 
  (carter_raisins = bryce_raisins - 10) ∧ 
  (carter_raisins = bryce_raisins / 3) := by
  sorry

end NUMINAMATH_CALUDE_raisin_distribution_l1352_135284


namespace NUMINAMATH_CALUDE_total_bananas_l1352_135280

def banana_problem (dawn_bananas lydia_bananas donna_bananas : ℕ) : Prop :=
  lydia_bananas = 60 ∧
  dawn_bananas = lydia_bananas + 40 ∧
  donna_bananas = 40 ∧
  dawn_bananas + lydia_bananas + donna_bananas = 200

theorem total_bananas : ∃ dawn_bananas lydia_bananas donna_bananas : ℕ,
  banana_problem dawn_bananas lydia_bananas donna_bananas :=
by
  sorry

end NUMINAMATH_CALUDE_total_bananas_l1352_135280


namespace NUMINAMATH_CALUDE_starting_number_proof_l1352_135269

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem starting_number_proof (n : ℕ) : 
  (∃! m : ℕ, m > n ∧ m < 580 ∧ 
   (∃ l : List ℕ, l.length = 6 ∧ 
    (∀ x ∈ l, x > n ∧ x < 580 ∧ is_divisible_by x 45 ∧ is_divisible_by x 6) ∧
    (∀ y : ℕ, y > n ∧ y < 580 ∧ is_divisible_by y 45 ∧ is_divisible_by y 6 → y ∈ l))) →
  (∀ k : ℕ, k > n → 
    ¬(∃ l : List ℕ, l.length = 6 ∧ 
      (∀ x ∈ l, x > k ∧ x < 580 ∧ is_divisible_by x 45 ∧ is_divisible_by x 6) ∧
      (∀ y : ℕ, y > k ∧ y < 580 ∧ is_divisible_by y 45 ∧ is_divisible_by y 6 → y ∈ l))) →
  n = 450 := by
sorry

end NUMINAMATH_CALUDE_starting_number_proof_l1352_135269


namespace NUMINAMATH_CALUDE_charles_total_money_l1352_135278

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of pennies Charles found -/
def pennies_found : ℕ := 6

/-- The number of nickels Charles had at home -/
def nickels_at_home : ℕ := 3

/-- Theorem stating that the total value of Charles' coins is 21 cents -/
theorem charles_total_money : 
  pennies_found * penny_value + nickels_at_home * nickel_value = 21 := by
  sorry

end NUMINAMATH_CALUDE_charles_total_money_l1352_135278


namespace NUMINAMATH_CALUDE_sum_of_roots_is_negative_one_l1352_135272

theorem sum_of_roots_is_negative_one (m n : ℝ) : 
  m ≠ 0 → n ≠ 0 → (∀ x, x^2 + m*x + n = 0 ↔ (x = m ∨ x = n)) → m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_negative_one_l1352_135272


namespace NUMINAMATH_CALUDE_fraction_problem_l1352_135207

theorem fraction_problem (N : ℝ) (f : ℝ) : 
  N = 180 → 
  (1/2 * f * 1/5 * N) + 6 = 1/15 * N → 
  f = 1/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1352_135207


namespace NUMINAMATH_CALUDE_greatest_measuring_length_l1352_135298

theorem greatest_measuring_length
  (length1 length2 length3 : ℕ)
  (h1 : length1 = 1234)
  (h2 : length2 = 898)
  (h3 : length3 > 0)
  (h4 : Nat.gcd length1 (Nat.gcd length2 length3) = 1) :
  ∀ (measuring_length : ℕ),
    (measuring_length > 0 ∧
     length1 % measuring_length = 0 ∧
     length2 % measuring_length = 0 ∧
     length3 % measuring_length = 0) →
    measuring_length = 1 :=
by sorry

end NUMINAMATH_CALUDE_greatest_measuring_length_l1352_135298


namespace NUMINAMATH_CALUDE_remaining_marbles_l1352_135219

/-- Given Chris has 12 marbles and Ryan has 28 marbles, if they combine their marbles
    and each takes 1/4 of the total, the number of marbles remaining in the pile is 20. -/
theorem remaining_marbles (chris_marbles : ℕ) (ryan_marbles : ℕ) 
    (h1 : chris_marbles = 12) 
    (h2 : ryan_marbles = 28) : 
  let total_marbles := chris_marbles + ryan_marbles
  let taken_marbles := 2 * (total_marbles / 4)
  total_marbles - taken_marbles = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_marbles_l1352_135219


namespace NUMINAMATH_CALUDE_two_common_tangents_l1352_135273

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Definition of circle C₂ -/
def C₂ (x y r : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = r^2

/-- Theorem stating the condition for exactly two common tangent lines -/
theorem two_common_tangents (r : ℝ) :
  (r > 0) →
  (∃ (x y : ℝ), C₁ x y ∧ C₂ x y r) ↔ (Real.sqrt 5 - 2 < r ∧ r < Real.sqrt 5 + 2) :=
sorry

end NUMINAMATH_CALUDE_two_common_tangents_l1352_135273


namespace NUMINAMATH_CALUDE_probability_three_red_one_blue_l1352_135215

theorem probability_three_red_one_blue (total_red : Nat) (total_blue : Nat) 
  (draw_count : Nat) (red_count : Nat) (blue_count : Nat) :
  total_red = 10 →
  total_blue = 5 →
  draw_count = 4 →
  red_count = 3 →
  blue_count = 1 →
  (Nat.choose total_red red_count * Nat.choose total_blue blue_count : ℚ) / 
  (Nat.choose (total_red + total_blue) draw_count) = 40 / 91 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_red_one_blue_l1352_135215


namespace NUMINAMATH_CALUDE_new_person_weight_l1352_135260

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (replaced_weight : ℝ) (average_increase : ℝ) : ℝ :=
  initial_count * average_increase + replaced_weight

/-- Theorem stating that the weight of the new person is 93 kg -/
theorem new_person_weight :
  weight_of_new_person 8 65 3.5 = 93 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1352_135260


namespace NUMINAMATH_CALUDE_marias_workday_end_l1352_135283

-- Define a custom type for time
structure Time where
  hour : Nat
  minute : Nat

-- Define addition operation for Time
def Time.add (t1 t2 : Time) : Time :=
  let totalMinutes := t1.hour * 60 + t1.minute + t2.hour * 60 + t2.minute
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

-- Define subtraction operation for Time
def Time.sub (t1 t2 : Time) : Time :=
  let totalMinutes := (t1.hour * 60 + t1.minute) - (t2.hour * 60 + t2.minute)
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

-- Define equality for Time
instance : BEq Time where
  beq t1 t2 := t1.hour = t2.hour ∧ t1.minute = t2.minute

theorem marias_workday_end :
  let start_time : Time := { hour := 8, minute := 0 }
  let lunch_start : Time := { hour := 13, minute := 0 }
  let lunch_duration : Time := { hour := 0, minute := 30 }
  let total_work_hours : Time := { hour := 8, minute := 0 }
  
  let time_before_lunch := lunch_start.sub start_time
  let lunch_end := lunch_start.add lunch_duration
  let remaining_work_time := total_work_hours.sub time_before_lunch
  let end_time := lunch_end.add remaining_work_time

  end_time = { hour := 16, minute := 30 } := by
    sorry

end NUMINAMATH_CALUDE_marias_workday_end_l1352_135283


namespace NUMINAMATH_CALUDE_function_equation_implies_identity_l1352_135266

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_implies_identity_l1352_135266


namespace NUMINAMATH_CALUDE_total_weight_lifted_l1352_135217

/-- Represents the weight a weightlifter can lift in one hand -/
def weight_per_hand : ℕ := 10

/-- Represents the number of hands a weightlifter has -/
def number_of_hands : ℕ := 2

/-- Theorem stating the total weight a weightlifter can lift -/
theorem total_weight_lifted : weight_per_hand * number_of_hands = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_lifted_l1352_135217


namespace NUMINAMATH_CALUDE_comprehensive_survey_suitability_l1352_135276

/-- Represents a survey scenario --/
inductive SurveyScenario
  | CalculatorServiceLife
  | BeijingStudentsSpaceflightLogo
  | ClassmatesBadalingGreatWall
  | FoodPigmentContent

/-- Determines if a survey scenario is suitable for a comprehensive survey --/
def isSuitableForComprehensiveSurvey (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.ClassmatesBadalingGreatWall => True
  | _ => False

/-- Theorem stating that the ClassmatesBadalingGreatWall scenario is the only one suitable for a comprehensive survey --/
theorem comprehensive_survey_suitability :
  ∀ (scenario : SurveyScenario),
    isSuitableForComprehensiveSurvey scenario ↔ scenario = SurveyScenario.ClassmatesBadalingGreatWall :=
by
  sorry

#check comprehensive_survey_suitability

end NUMINAMATH_CALUDE_comprehensive_survey_suitability_l1352_135276


namespace NUMINAMATH_CALUDE_division_problem_l1352_135282

theorem division_problem : (88 / 4) / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1352_135282


namespace NUMINAMATH_CALUDE_all_three_hits_mutually_exclusive_with_at_most_two_hits_l1352_135218

-- Define the sample space for three shots
inductive ShotOutcome
| Hit
| Miss

-- Define the type for a sequence of three shots
def ThreeShots := (ShotOutcome × ShotOutcome × ShotOutcome)

-- Define the event "at most two hits"
def atMostTwoHits (shots : ThreeShots) : Prop :=
  match shots with
  | (ShotOutcome.Hit, ShotOutcome.Hit, ShotOutcome.Hit) => False
  | _ => True

-- Define the event "all three hits"
def allThreeHits (shots : ThreeShots) : Prop :=
  match shots with
  | (ShotOutcome.Hit, ShotOutcome.Hit, ShotOutcome.Hit) => True
  | _ => False

-- Theorem stating that "all three hits" and "at most two hits" are mutually exclusive
theorem all_three_hits_mutually_exclusive_with_at_most_two_hits :
  ∀ (shots : ThreeShots), ¬(atMostTwoHits shots ∧ allThreeHits shots) :=
by
  sorry


end NUMINAMATH_CALUDE_all_three_hits_mutually_exclusive_with_at_most_two_hits_l1352_135218


namespace NUMINAMATH_CALUDE_no_real_root_in_unit_interval_l1352_135204

theorem no_real_root_in_unit_interval (a b c d : ℝ) :
  (min d (b + d) > max (abs c) (abs (a + c))) →
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → (a * x^3 + b * x^2 + c * x + d ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_root_in_unit_interval_l1352_135204


namespace NUMINAMATH_CALUDE_missing_fraction_proof_l1352_135200

theorem missing_fraction_proof (total_sum : ℚ) (f1 f2 f3 f4 f5 f6 : ℚ) :
  total_sum = 0.13333333333333333 ∧
  f1 = 1/3 ∧ f2 = 1/2 ∧ f3 = -5/6 ∧ f4 = 1/5 ∧ f5 = -9/20 ∧ f6 = -2/15 →
  ∃ x : ℚ, x + f1 + f2 + f3 + f4 + f5 + f6 = total_sum ∧ x = 31/60 :=
by sorry

end NUMINAMATH_CALUDE_missing_fraction_proof_l1352_135200


namespace NUMINAMATH_CALUDE_root_sum_theorem_l1352_135249

-- Define the quadratic equation
def quadratic_eq (k x : ℝ) : ℝ := k * (x^2 - x) + x + 5

-- Define the condition for k1 and k2
def k_condition (k : ℝ) : Prop :=
  ∃ a b : ℝ, quadratic_eq k a = 0 ∧ quadratic_eq k b = 0 ∧ a / b + b / a = 4 / 5

-- Theorem statement
theorem root_sum_theorem (k1 k2 : ℝ) :
  k_condition k1 ∧ k_condition k2 → k1 / k2 + k2 / k1 = 254 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l1352_135249


namespace NUMINAMATH_CALUDE_equal_numbers_sum_of_squares_l1352_135224

theorem equal_numbers_sum_of_squares (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 12 →
  b = 26 →
  c = 22 →
  d = e →
  d^2 + e^2 = 800 := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_sum_of_squares_l1352_135224


namespace NUMINAMATH_CALUDE_tangent_line_at_one_a_upper_bound_l1352_135212

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * x + a / x

/-- Theorem for part (1) of the problem -/
theorem tangent_line_at_one (a : ℝ) :
  a = 2 → ∃ A B C : ℝ, A = 3 ∧ B = 2 ∧ C = -6 ∧
  ∀ x y : ℝ, y = f a x → (x = 1 → A * x + B * y + C = 0) :=
sorry

/-- Theorem for part (2) of the problem -/
theorem a_upper_bound :
  (∀ x : ℝ, x > 1 → f a x < 0) → a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_a_upper_bound_l1352_135212


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l1352_135291

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 → n ≥ 1006 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l1352_135291


namespace NUMINAMATH_CALUDE_james_weight_vest_cost_l1352_135290

/-- The cost of James's weight vest -/
def weight_vest_cost : ℝ := 250

/-- The cost of weight plates per pound -/
def weight_plate_cost_per_pound : ℝ := 1.2

/-- The weight of the plates in pounds -/
def weight_plate_pounds : ℝ := 200

/-- The original cost of a 200-pound weight vest -/
def original_vest_cost : ℝ := 700

/-- The discount on the 200-pound weight vest -/
def vest_discount : ℝ := 100

/-- The amount James saves with his vest -/
def james_savings : ℝ := 110

/-- Theorem: The cost of James's weight vest is $250 -/
theorem james_weight_vest_cost : 
  weight_vest_cost = 
    (original_vest_cost - vest_discount) - 
    (weight_plate_cost_per_pound * weight_plate_pounds) - 
    james_savings := by
  sorry

end NUMINAMATH_CALUDE_james_weight_vest_cost_l1352_135290


namespace NUMINAMATH_CALUDE_western_village_conscription_l1352_135263

theorem western_village_conscription 
  (north_pop : ℕ) 
  (west_pop : ℕ) 
  (south_pop : ℕ) 
  (total_conscripts : ℕ) 
  (h1 : north_pop = 8758) 
  (h2 : west_pop = 7236) 
  (h3 : south_pop = 8356) 
  (h4 : total_conscripts = 378) : 
  (west_pop : ℚ) / (north_pop + west_pop + south_pop : ℚ) * total_conscripts = 112 := by
sorry

end NUMINAMATH_CALUDE_western_village_conscription_l1352_135263


namespace NUMINAMATH_CALUDE_range_of_f_l1352_135238

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem statement
theorem range_of_f :
  ∃ (a b : ℝ), a = -2 ∧ b = 2 ∧
  (∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ f x = y) ↔ a ≤ y ∧ y ≤ b) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1352_135238


namespace NUMINAMATH_CALUDE_gcd_problem_l1352_135209

theorem gcd_problem : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 90 ∧ Nat.gcd n 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1352_135209


namespace NUMINAMATH_CALUDE_log_216_equals_3_log_6_l1352_135247

theorem log_216_equals_3_log_6 : Real.log 216 = 3 * Real.log 6 := by
  sorry

end NUMINAMATH_CALUDE_log_216_equals_3_log_6_l1352_135247


namespace NUMINAMATH_CALUDE_fraction_difference_equals_nine_twentieths_l1352_135252

theorem fraction_difference_equals_nine_twentieths :
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_nine_twentieths_l1352_135252


namespace NUMINAMATH_CALUDE_numbers_not_sum_of_two_elements_l1352_135292

def A : Finset ℕ := {1, 2, 3, 5, 8, 13, 21, 34, 55}

def range_start : ℕ := 3
def range_end : ℕ := 89

def sums_of_two_elements (S : Finset ℕ) : Finset ℕ :=
  (S.product S).image (λ (x : ℕ × ℕ) => x.1 + x.2)

def numbers_in_range : Finset ℕ :=
  Finset.Icc range_start range_end

theorem numbers_not_sum_of_two_elements : 
  (numbers_in_range.card - (numbers_in_range ∩ sums_of_two_elements A).card) = 51 := by
  sorry

end NUMINAMATH_CALUDE_numbers_not_sum_of_two_elements_l1352_135292


namespace NUMINAMATH_CALUDE_max_perimeter_of_special_triangle_l1352_135286

/-- Represents the sides of a triangle --/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given sides form a valid triangle --/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℕ :=
  t.a + t.b + t.c

/-- Theorem stating the maximum perimeter of the triangle --/
theorem max_perimeter_of_special_triangle :
  ∃ (t : Triangle),
    t.a = 5 ∧
    t.b = 6 ∧
    isValidTriangle t ∧
    (∀ (t' : Triangle),
      t'.a = 5 →
      t'.b = 6 →
      isValidTriangle t' →
      perimeter t' ≤ perimeter t) ∧
    perimeter t = 21 :=
  sorry

end NUMINAMATH_CALUDE_max_perimeter_of_special_triangle_l1352_135286


namespace NUMINAMATH_CALUDE_total_fish_count_l1352_135253

theorem total_fish_count (jerk_tuna : ℕ) (tall_tuna : ℕ) (swell_tuna : ℕ) : 
  jerk_tuna = 144 →
  tall_tuna = 2 * jerk_tuna →
  swell_tuna = tall_tuna + (tall_tuna / 2) →
  jerk_tuna + tall_tuna + swell_tuna = 864 :=
by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l1352_135253


namespace NUMINAMATH_CALUDE_coffee_table_books_l1352_135293

/-- Represents the number of books Henry has in different locations and actions he takes. -/
structure HenryBooks where
  total : ℕ
  boxed : ℕ
  boxCount : ℕ
  roomDonate : ℕ
  kitchenDonate : ℕ
  newPickup : ℕ
  finalCount : ℕ

/-- Calculates the number of books on Henry's coffee table. -/
def booksOnCoffeeTable (h : HenryBooks) : ℕ :=
  h.total - (h.boxed * h.boxCount + h.roomDonate + h.kitchenDonate) - (h.finalCount - h.newPickup)

/-- Theorem stating that the number of books on Henry's coffee table is 4. -/
theorem coffee_table_books :
  let h : HenryBooks := {
    total := 99,
    boxed := 15,
    boxCount := 3,
    roomDonate := 21,
    kitchenDonate := 18,
    newPickup := 12,
    finalCount := 23
  }
  booksOnCoffeeTable h = 4 := by
  sorry

end NUMINAMATH_CALUDE_coffee_table_books_l1352_135293


namespace NUMINAMATH_CALUDE_mother_age_now_is_70_l1352_135257

/-- Jessica's age now -/
def jessica_age_now : ℕ := 40

/-- Years passed since mother's death -/
def years_passed : ℕ := 10

/-- Jessica's age when her mother died -/
def jessica_age_then : ℕ := jessica_age_now - years_passed

/-- Mother's age when she died -/
def mother_age_then : ℕ := 2 * jessica_age_then

/-- Mother's age now if she were alive -/
def mother_age_now : ℕ := mother_age_then + years_passed

theorem mother_age_now_is_70 : mother_age_now = 70 := by
  sorry

end NUMINAMATH_CALUDE_mother_age_now_is_70_l1352_135257


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1352_135295

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1352_135295


namespace NUMINAMATH_CALUDE_distinct_configurations_correct_l1352_135296

/-- Represents the number of distinct configurations of n coins arranged in a circle
    that cannot be transformed into one another by flipping adjacent pairs of coins
    with the same orientation. -/
def distinctConfigurations (n : ℕ) : ℕ :=
  if n % 2 = 0 then n + 1 else 2

theorem distinct_configurations_correct (n : ℕ) :
  distinctConfigurations n = if n % 2 = 0 then n + 1 else 2 := by
  sorry

end NUMINAMATH_CALUDE_distinct_configurations_correct_l1352_135296


namespace NUMINAMATH_CALUDE_gcd_sum_lcm_eq_gcd_l1352_135202

theorem gcd_sum_lcm_eq_gcd (a b : ℤ) : Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_lcm_eq_gcd_l1352_135202


namespace NUMINAMATH_CALUDE_modifiedLucas_100th_term_divisible_by_5_l1352_135233

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 4
  | (n + 2) => (modifiedLucas n + modifiedLucas (n + 1)) % 5

theorem modifiedLucas_100th_term_divisible_by_5 : modifiedLucas 99 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_modifiedLucas_100th_term_divisible_by_5_l1352_135233


namespace NUMINAMATH_CALUDE_closed_set_properties_l1352_135216

-- Define a closed set
def is_closed_set (M : Set Int) : Prop :=
  ∀ a b : Int, a ∈ M ∧ b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Define the set M = {-4, -2, 0, 2, 4}
def M : Set Int := {-4, -2, 0, 2, 4}

-- Define the set of positive integers
def positive_integers : Set Int := {n : Int | n > 0}

-- Define the set M = {n | n = 3k, k ∈ Z}
def M_3k : Set Int := {n : Int | ∃ k : Int, n = 3 * k}

theorem closed_set_properties :
  (¬ is_closed_set M) ∧
  (¬ is_closed_set positive_integers) ∧
  (is_closed_set M_3k) ∧
  (∃ A₁ A₂ : Set Int, is_closed_set A₁ ∧ is_closed_set A₂ ∧ ¬ is_closed_set (A₁ ∪ A₂)) :=
sorry

end NUMINAMATH_CALUDE_closed_set_properties_l1352_135216


namespace NUMINAMATH_CALUDE_milk_mixture_theorem_l1352_135264

/-- Proves that adding 8 gallons of 10% butterfat milk to 8 gallons of 30% butterfat milk
    results in a mixture with 20% butterfat. -/
theorem milk_mixture_theorem :
  let initial_milk : ℝ := 8
  let initial_butterfat_percent : ℝ := 30
  let added_milk : ℝ := 8
  let added_butterfat_percent : ℝ := 10
  let final_butterfat_percent : ℝ := 20
  let total_milk : ℝ := initial_milk + added_milk
  let total_butterfat : ℝ := (initial_milk * initial_butterfat_percent + added_milk * added_butterfat_percent) / 100
  total_butterfat / total_milk * 100 = final_butterfat_percent :=
by sorry

end NUMINAMATH_CALUDE_milk_mixture_theorem_l1352_135264


namespace NUMINAMATH_CALUDE_set_A_at_most_one_element_l1352_135206

theorem set_A_at_most_one_element (a : ℝ) : 
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) ↔ (a ≥ 9/8 ∨ a = 0) :=
sorry

end NUMINAMATH_CALUDE_set_A_at_most_one_element_l1352_135206


namespace NUMINAMATH_CALUDE_fuel_consumption_l1352_135258

/-- Represents the fuel consumption of a heating plant -/
structure HeatingPlant where
  hours : ℝ
  fuel : ℝ
  rate : ℝ
  hRate : rate = fuel / hours

/-- Given a heating plant that consumes 7 liters of fuel in 21 hours,
    prove that it consumes 30 liters of fuel in 90 hours -/
theorem fuel_consumption (plant : HeatingPlant) 
  (h1 : plant.hours = 21) 
  (h2 : plant.fuel = 7) : 
  plant.rate * 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fuel_consumption_l1352_135258


namespace NUMINAMATH_CALUDE_f_positivity_and_extrema_l1352_135213

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (2 * x^2 - 3 * x)

theorem f_positivity_and_extrema :
  (∀ x : ℝ, f x > 0 ↔ x < 0 ∨ x > 3/2) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f x ≤ 2 * Real.exp 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f x ≥ -Real.exp 1) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ f x = 2 * Real.exp 2) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ f x = -Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_f_positivity_and_extrema_l1352_135213


namespace NUMINAMATH_CALUDE_coloring_theorem_l1352_135270

/-- Given finite sets A, B, C, D, this function calculates the number of ways
    to color three adjacent elements with the condition that adjacent elements
    must have different colors. -/
def colorThreeAdjacent (A B C : Finset α) : ℕ :=
  A.card * (B.card - 1) * (C.card - 1)

/-- Given finite sets A, B, C, D, this function calculates the number of ways
    to color four adjacent elements with the condition that adjacent elements
    must have different colors. -/
def colorFourAdjacent (A B C D : Finset α) : ℕ :=
  A.card * (B.card - 1) * (C.card - 1) * (D.card - 1)

theorem coloring_theorem (A B C D : Finset α) :
  (colorThreeAdjacent A B C = A.card * (B.card - 1) * (C.card - 1)) ∧
  (colorFourAdjacent A B C D = A.card * (B.card - 1) * (C.card - 1) * (D.card - 1)) := by
  sorry

end NUMINAMATH_CALUDE_coloring_theorem_l1352_135270


namespace NUMINAMATH_CALUDE_bipartite_perfect_matching_l1352_135211

/-- A bipartite graph with 20 vertices in each part and degree 2 for all vertices -/
structure BipartiteGraph :=
  (U V : Finset ℕ)
  (E : Finset (ℕ × ℕ))
  (hU : U.card = 20)
  (hV : V.card = 20)
  (hE : ∀ u ∈ U, (E.filter (λ e => e.1 = u)).card = 2)
  (hE' : ∀ v ∈ V, (E.filter (λ e => e.2 = v)).card = 2)

/-- A perfect matching in a bipartite graph -/
def PerfectMatching (G : BipartiteGraph) :=
  ∃ M : Finset (ℕ × ℕ), M ⊆ G.E ∧ 
    (∀ u ∈ G.U, (M.filter (λ e => e.1 = u)).card = 1) ∧
    (∀ v ∈ G.V, (M.filter (λ e => e.2 = v)).card = 1)

/-- Theorem: A bipartite graph with 20 vertices in each part and degree 2 for all vertices has a perfect matching -/
theorem bipartite_perfect_matching (G : BipartiteGraph) : PerfectMatching G := by
  sorry

end NUMINAMATH_CALUDE_bipartite_perfect_matching_l1352_135211


namespace NUMINAMATH_CALUDE_range_of_m_l1352_135241

/-- Given an increasing function f on ℝ, if f(2m) < f(9-m), then m < 3 -/
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_inequality : f (2 * m) < f (9 - m)) : 
  m < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1352_135241


namespace NUMINAMATH_CALUDE_competition_result_count_l1352_135228

/-- Represents a team's score composition -/
structure TeamScore where
  threes : ℕ  -- number of 3-point problems solved
  fives  : ℕ  -- number of 5-point problems solved

/-- Calculates the total score for a team -/
def totalScore (t : TeamScore) : ℕ := 3 * t.threes + 5 * t.fives

/-- Represents the scores of all three teams -/
structure CompetitionResult where
  team1 : TeamScore
  team2 : TeamScore
  team3 : TeamScore

/-- Checks if a competition result is valid -/
def isValidResult (r : CompetitionResult) : Prop :=
  totalScore r.team1 + totalScore r.team2 + totalScore r.team3 = 32

/-- Counts the number of valid competition results -/
def countValidResults : ℕ := sorry

theorem competition_result_count :
  countValidResults = 255 := by sorry

end NUMINAMATH_CALUDE_competition_result_count_l1352_135228


namespace NUMINAMATH_CALUDE_camping_group_solution_l1352_135274

/-- Represents the camping group -/
structure CampingGroup where
  initialTotal : ℕ
  initialGirls : ℕ

/-- Conditions of the camping group problem -/
class CampingGroupProblem (g : CampingGroup) where
  initial_ratio : g.initialGirls = g.initialTotal / 2
  final_ratio : (g.initialGirls + 1) * 10 = 6 * (g.initialTotal - 2)

/-- The theorem stating the solution to the camping group problem -/
theorem camping_group_solution (g : CampingGroup) [CampingGroupProblem g] : 
  g.initialGirls = 11 := by
  sorry

#check camping_group_solution

end NUMINAMATH_CALUDE_camping_group_solution_l1352_135274


namespace NUMINAMATH_CALUDE_dance_attendance_l1352_135220

theorem dance_attendance (boys girls teachers : ℕ) : 
  (boys : ℚ) / girls = 3 / 4 →
  teachers = boys / 5 →
  boys + girls + teachers = 114 →
  girls = 60 := by
sorry

end NUMINAMATH_CALUDE_dance_attendance_l1352_135220


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l1352_135245

/-- Triangular number function -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number represents the total number of coins in a stack with 30 layers -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l1352_135245


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l1352_135235

theorem stratified_sampling_female_count 
  (total_employees : ℕ) 
  (female_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 200)
  (h2 : female_employees = 80)
  (h3 : sample_size = 20) :
  (female_employees : ℚ) / total_employees * sample_size = 8 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l1352_135235


namespace NUMINAMATH_CALUDE_lynne_cat_books_l1352_135255

def books_about_cats (x : ℕ) : Prop :=
  ∃ (total_spent : ℕ),
    let books_solar_system := 2
    let magazines := 3
    let book_cost := 7
    let magazine_cost := 4
    total_spent = 75 ∧
    total_spent = x * book_cost + books_solar_system * book_cost + magazines * magazine_cost

theorem lynne_cat_books : ∃ x : ℕ, books_about_cats x ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_lynne_cat_books_l1352_135255


namespace NUMINAMATH_CALUDE_power_product_equality_l1352_135271

theorem power_product_equality (x : ℝ) : (-2 * x^2) * (-4 * x^3) = 8 * x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1352_135271


namespace NUMINAMATH_CALUDE_orthogonal_equal_magnitude_vectors_l1352_135243

/-- Given two vectors a and b in R³, if they are orthogonal and have equal magnitudes,
    then their components satisfy specific values. -/
theorem orthogonal_equal_magnitude_vectors
  (a b : ℝ × ℝ × ℝ)
  (h_a : a = (4, p, -2))
  (h_b : b = (3, 2, q))
  (h_orthogonal : a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0)
  (h_equal_magnitude : a.1^2 + a.2.1^2 + a.2.2^2 = b.1^2 + b.2.1^2 + b.2.2^2)
  : p = -29/12 ∧ q = 43/12 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_equal_magnitude_vectors_l1352_135243


namespace NUMINAMATH_CALUDE_fish_distribution_theorem_l1352_135287

theorem fish_distribution_theorem (a b c d e f : ℕ) : 
  a + b + c + d + e + f = 100 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧
  (b + c + d + e + f) % 5 = 0 ∧
  (a + c + d + e + f) % 5 = 0 ∧
  (a + b + d + e + f) % 5 = 0 ∧
  (a + b + c + e + f) % 5 = 0 ∧
  (a + b + c + d + f) % 5 = 0 ∧
  (a + b + c + d + e) % 5 = 0 →
  a = 20 ∨ b = 20 ∨ c = 20 ∨ d = 20 ∨ e = 20 ∨ f = 20 :=
by sorry

end NUMINAMATH_CALUDE_fish_distribution_theorem_l1352_135287


namespace NUMINAMATH_CALUDE_initial_candies_l1352_135246

theorem initial_candies (eaten : ℕ) (left : ℕ) (h1 : eaten = 15) (h2 : left = 13) :
  eaten + left = 28 := by
  sorry

end NUMINAMATH_CALUDE_initial_candies_l1352_135246


namespace NUMINAMATH_CALUDE_proposition_and_converse_l1352_135259

theorem proposition_and_converse (a b : ℝ) : 
  (((a + b ≥ 2) → (a ≥ 1 ∨ b ≥ 1)) ∧ 
  (∃ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) ∧ ¬(a + b ≥ 2))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_converse_l1352_135259


namespace NUMINAMATH_CALUDE_probability_four_threes_eight_dice_l1352_135250

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def target_value : ℕ := 3
def num_target : ℕ := 4

def probability_exact_dice : ℚ :=
  (num_dice.choose num_target) *
  (1 / num_sides) ^ num_target *
  ((num_sides - 1) / num_sides) ^ (num_dice - num_target)

theorem probability_four_threes_eight_dice :
  probability_exact_dice = 168070 / 16777216 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_threes_eight_dice_l1352_135250


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1352_135229

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (right : ℝ) (down : ℝ) : Parabola :=
  { a := p.a
    h := p.h + right
    k := p.k - down }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 3 ∧ p.h = 4 ∧ p.k = 2 →
  (shift p 1 3).a = 3 ∧ (shift p 1 3).h = 5 ∧ (shift p 1 3).k = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1352_135229


namespace NUMINAMATH_CALUDE_base9_perfect_square_last_digit_l1352_135256

/-- Represents a number in base 9 of the form ab5d -/
structure Base9Number where
  a : ℕ
  b : ℕ
  d : ℕ
  a_nonzero : a ≠ 0
  b_range : b < 9
  d_range : d < 9

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.b + 45 + n.d

/-- Predicate to check if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem base9_perfect_square_last_digit
  (n : Base9Number)
  (h : isPerfectSquare (toDecimal n)) :
  n.d = 0 := by
  sorry

#check base9_perfect_square_last_digit

end NUMINAMATH_CALUDE_base9_perfect_square_last_digit_l1352_135256


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_even_legs_l1352_135299

theorem right_triangle_consecutive_even_legs (a b c : ℕ) : 
  -- a and b are the legs, c is the hypotenuse
  (a * a + b * b = c * c) →  -- Pythagorean theorem
  (∃ k : ℕ, a = 2 * k ∧ b = 2 * k + 2) →  -- consecutive even numbers
  (c = 34) →  -- hypotenuse is 34
  (a + b = 46) :=  -- sum of legs is 46
by sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_even_legs_l1352_135299


namespace NUMINAMATH_CALUDE_prism_volume_l1352_135234

/-- The volume of a right rectangular prism given its face areas -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 50) (h2 : a * c = 72) (h3 : b * c = 45) :
  a * b * c = 180 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1352_135234


namespace NUMINAMATH_CALUDE_min_PM_dot_PF_l1352_135288

/-- Parabola C: y^2 = 2px (p > 0) -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

/-- Circle M with center on positive x-axis and tangent to y-axis -/
def circle_M (center_x : ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center_x)^2 + y^2 = radius^2 ∧ center_x > 0 ∧ center_x = radius

/-- Line m passing through origin with inclination angle π/3 -/
def line_m (x y : ℝ) : Prop := y = x * Real.sqrt 3

/-- Point A on directrix l and point B on circle M, both on line m -/
def points_A_B (A B : ℝ × ℝ) : Prop :=
  line_m A.1 A.2 ∧ line_m B.1 B.2 ∧ A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4

/-- Theorem: Minimum value of PM⋅PF is 2 -/
theorem min_PM_dot_PF (p : ℝ) (center_x radius : ℝ) (A B : ℝ × ℝ) :
  parabola p 1 2 →
  circle_M center_x radius center_x 0 →
  points_A_B A B →
  (∀ x y : ℝ, parabola p x y → 
    (x^2 - center_x*x + (center_x^2)/4 + y^2) ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_min_PM_dot_PF_l1352_135288


namespace NUMINAMATH_CALUDE_nancys_weight_l1352_135237

/-- 
Given that Nancy's total daily water intake (including water from food) is 62 pounds,
and she drinks 75% of her body weight in water plus 2 pounds from food,
prove that her weight is 80 pounds.
-/
theorem nancys_weight (W : ℝ) : 0.75 * W + 2 = 62 → W = 80 := by
  sorry

end NUMINAMATH_CALUDE_nancys_weight_l1352_135237


namespace NUMINAMATH_CALUDE_linear_function_k_value_l1352_135289

/-- Given a linear function y = kx + 6 passing through the point (2, -2), prove that k = -4 -/
theorem linear_function_k_value :
  ∀ k : ℝ, (∀ x y : ℝ, y = k * x + 6) → -2 = k * 2 + 6 → k = -4 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l1352_135289


namespace NUMINAMATH_CALUDE_recycling_program_earnings_l1352_135201

/-- Recycling program earnings calculation -/
theorem recycling_program_earnings 
  (initial_signup_bonus : ℕ)
  (referral_bonus : ℕ)
  (friend_signup_bonus : ℕ)
  (day_one_friends : ℕ)
  (week_end_friends : ℕ) :
  initial_signup_bonus = 5 →
  referral_bonus = 5 →
  friend_signup_bonus = 5 →
  day_one_friends = 5 →
  week_end_friends = 7 →
  (initial_signup_bonus + 
   (day_one_friends + week_end_friends) * (referral_bonus + friend_signup_bonus)) = 125 :=
by sorry

end NUMINAMATH_CALUDE_recycling_program_earnings_l1352_135201


namespace NUMINAMATH_CALUDE_cos_sin_sum_l1352_135242

theorem cos_sin_sum (α : Real) (h : Real.cos (π/6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5*π/6 + α) + (Real.sin (α - π/6))^2 = (2 - Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_sin_sum_l1352_135242


namespace NUMINAMATH_CALUDE_log_simplification_l1352_135232

theorem log_simplification (x y z w t v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (ht : t > 0) (hv : v > 0) :
  Real.log (x / z) + Real.log (z / y) + Real.log (y / w) - Real.log (x * v / (w * t)) = Real.log (t / v) :=
by sorry

end NUMINAMATH_CALUDE_log_simplification_l1352_135232


namespace NUMINAMATH_CALUDE_min_people_for_company2_cheaper_l1352_135261

/-- Company 1's pricing function -/
def company1_cost (people : ℕ) : ℕ := 150 + 18 * people

/-- Company 2's pricing function -/
def company2_cost (people : ℕ) : ℕ := 250 + 15 * people

/-- Theorem stating the minimum number of people for Company 2 to be cheaper -/
theorem min_people_for_company2_cheaper :
  (company2_cost 34 < company1_cost 34) ∧
  (company1_cost 33 ≤ company2_cost 33) := by
  sorry

end NUMINAMATH_CALUDE_min_people_for_company2_cheaper_l1352_135261


namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l1352_135208

theorem cube_sum_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs (z₁ + z₂) = 20)
  (h2 : Complex.abs (z₁^2 + z₂^2) = 16) :
  Complex.abs (z₁^3 + z₂^3) = 3520 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l1352_135208


namespace NUMINAMATH_CALUDE_five_star_three_eq_nineteen_l1352_135223

/-- Definition of the star operation -/
def star (a b : ℝ) : ℝ := a^2 - a*b + b^2

/-- Theorem: The value of 5 star 3 is 19 -/
theorem five_star_three_eq_nineteen : star 5 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_five_star_three_eq_nineteen_l1352_135223


namespace NUMINAMATH_CALUDE_refrigerator_profit_percentage_l1352_135222

/-- Calculates the profit percentage for a refrigerator sale --/
theorem refrigerator_profit_percentage 
  (labelled_price : ℝ)
  (discount_rate : ℝ)
  (purchase_price : ℝ)
  (transport_cost : ℝ)
  (installation_cost : ℝ)
  (selling_price : ℝ)
  (h1 : labelled_price = purchase_price / (1 - discount_rate))
  (h2 : discount_rate = 0.20)
  (h3 : purchase_price = 12500)
  (h4 : transport_cost = 125)
  (h5 : installation_cost = 250)
  (h6 : selling_price = 19200)
  : ∃ (profit_percentage : ℝ), 
    abs (profit_percentage - 49.13) < 0.01 ∧
    profit_percentage = (selling_price - (purchase_price + transport_cost + installation_cost)) / 
                        (purchase_price + transport_cost + installation_cost) * 100 :=
sorry

end NUMINAMATH_CALUDE_refrigerator_profit_percentage_l1352_135222


namespace NUMINAMATH_CALUDE_max_rectangles_in_3x4_grid_l1352_135294

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents a grid with width and height -/
structure Grid where
  width : Nat
  height : Nat

/-- Checks if a rectangle can fit in a grid -/
def fits (r : Rectangle) (g : Grid) : Prop :=
  r.width ≤ g.width ∧ r.height ≤ g.height

/-- Represents the maximum number of non-overlapping rectangles that can fit in a grid -/
def maxRectangles (r : Rectangle) (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that the maximum number of 1x2 rectangles in a 3x4 grid is 5 -/
theorem max_rectangles_in_3x4_grid :
  let r : Rectangle := ⟨1, 2⟩
  let g : Grid := ⟨3, 4⟩
  fits r g → maxRectangles r g = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangles_in_3x4_grid_l1352_135294


namespace NUMINAMATH_CALUDE_twelve_buses_required_l1352_135275

/-- The minimum number of buses required to transport all students -/
def min_buses (max_capacity : ℕ) (total_students : ℕ) (available_drivers : ℕ) : ℕ :=
  max (((total_students + max_capacity - 1) / max_capacity) : ℕ) available_drivers

/-- Proof that 12 buses are required given the problem conditions -/
theorem twelve_buses_required :
  min_buses 42 480 12 = 12 := by
  sorry

#eval min_buses 42 480 12  -- Should output 12

end NUMINAMATH_CALUDE_twelve_buses_required_l1352_135275


namespace NUMINAMATH_CALUDE_solution_set_when_m_eq_2_range_of_m_when_f_leq_5_l1352_135254

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| - |x - m|

-- Theorem for part I
theorem solution_set_when_m_eq_2 :
  {x : ℝ | f x 2 ≥ 1} = {x : ℝ | x ≥ 2} := by sorry

-- Theorem for part II
theorem range_of_m_when_f_leq_5 :
  {m : ℝ | ∀ x, f x m ≤ 5} = {m : ℝ | -4 ≤ m ∧ m ≤ 6} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_eq_2_range_of_m_when_f_leq_5_l1352_135254


namespace NUMINAMATH_CALUDE_simplify_expression_l1352_135214

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (25 * x^3) * (8 * x^4) * (1 / (4 * x^2)^3) = 25/8 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1352_135214


namespace NUMINAMATH_CALUDE_carrots_thrown_out_l1352_135239

theorem carrots_thrown_out (initial_carrots : ℕ) (additional_carrots : ℕ) (remaining_carrots : ℕ) : 
  initial_carrots = 48 →
  additional_carrots = 15 →
  remaining_carrots = 52 →
  initial_carrots + additional_carrots - remaining_carrots = 11 := by
sorry

end NUMINAMATH_CALUDE_carrots_thrown_out_l1352_135239


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1352_135205

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum1 : a 2 + a 5 + a 8 = 27)
  (h_sum2 : a 3 + a 6 + a 9 = 33) :
  a 4 = 7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1352_135205


namespace NUMINAMATH_CALUDE_find_unknown_number_l1352_135262

theorem find_unknown_number (known_numbers : List ℕ) (average : ℕ) : 
  known_numbers = [55, 507, 2, 684, 42] → 
  average = 223 → 
  ∃ x : ℕ, x = 48 ∧ (List.sum known_numbers + x) / 6 = average := by
sorry


end NUMINAMATH_CALUDE_find_unknown_number_l1352_135262

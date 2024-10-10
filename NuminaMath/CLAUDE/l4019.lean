import Mathlib

namespace point_on_ellipse_l4019_401914

/-- The coordinates of a point P on an ellipse satisfying specific conditions -/
theorem point_on_ellipse (x y : ℝ) : 
  x > 0 → -- P is on the right side of y-axis
  x^2 / 5 + y^2 / 4 = 1 → -- P is on the ellipse
  (1/2) * 2 * |y| = 1 → -- Area of triangle PF₁F₂ is 1
  (x = Real.sqrt 15 / 2) ∧ (y = 1) := by
  sorry

end point_on_ellipse_l4019_401914


namespace lcm_gcf_ratio_120_504_l4019_401909

theorem lcm_gcf_ratio_120_504 : 
  (Nat.lcm 120 504) / (Nat.gcd 120 504) = 105 := by sorry

end lcm_gcf_ratio_120_504_l4019_401909


namespace exists_number_with_specific_digit_sums_l4019_401929

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of its digits is 100
    and the sum of the digits of n^3 is 1,000,000 -/
theorem exists_number_with_specific_digit_sums :
  ∃ n : ℕ, sum_of_digits n = 100 ∧ sum_of_digits (n^3) = 1000000 := by sorry

end exists_number_with_specific_digit_sums_l4019_401929


namespace intersection_of_given_sets_l4019_401933

theorem intersection_of_given_sets :
  let A : Set ℕ := {1, 3, 4}
  let B : Set ℕ := {3, 4, 5}
  A ∩ B = {3, 4} := by
sorry

end intersection_of_given_sets_l4019_401933


namespace least_x_for_divisibility_by_three_l4019_401923

theorem least_x_for_divisibility_by_three : 
  (∃ x : ℕ, ∀ n : ℕ, (n * 57) % 3 = 0) ∧ 
  (∀ y : ℕ, y < 0 → ¬(∀ n : ℕ, (n * 57) % 3 = 0)) := by sorry

end least_x_for_divisibility_by_three_l4019_401923


namespace license_plate_combinations_l4019_401983

/-- The number of ways to choose 2 items from n items -/
def choose (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions on the license plate -/
def letter_positions : ℕ := 4

/-- The number of digit positions on the license plate -/
def digit_positions : ℕ := 3

/-- The maximum starting digit to allow for 3 consecutive increasing digits -/
def max_start_digit : ℕ := 7

theorem license_plate_combinations :
  (choose alphabet_size 2) * (choose letter_positions 2) * (max_start_digit + 1) = 15600 := by
  sorry

end license_plate_combinations_l4019_401983


namespace gasoline_price_quantity_adjustment_l4019_401941

theorem gasoline_price_quantity_adjustment 
  (original_price original_quantity : ℝ) 
  (price_increase : ℝ) 
  (spending_increase : ℝ) 
  (h1 : price_increase = 0.25) 
  (h2 : spending_increase = 0.10) : 
  let new_price := original_price * (1 + price_increase)
  let new_total_cost := original_price * original_quantity * (1 + spending_increase)
  let new_quantity := new_total_cost / new_price
  1 - (new_quantity / original_quantity) = 0.12 := by
sorry

end gasoline_price_quantity_adjustment_l4019_401941


namespace olive_flea_fraction_is_half_l4019_401978

/-- The fraction of fleas Olive has compared to Gertrude -/
def olive_flea_fraction (gertrude_fleas maud_fleas olive_fleas total_fleas : ℕ) : ℚ :=
  olive_fleas / gertrude_fleas

theorem olive_flea_fraction_is_half :
  ∀ (gertrude_fleas maud_fleas olive_fleas total_fleas : ℕ),
    gertrude_fleas = 10 →
    maud_fleas = 5 * olive_fleas →
    total_fleas = 40 →
    gertrude_fleas + maud_fleas + olive_fleas = total_fleas →
    olive_flea_fraction gertrude_fleas maud_fleas olive_fleas total_fleas = 1/2 :=
by
  sorry

end olive_flea_fraction_is_half_l4019_401978


namespace minimum_students_with_girl_percentage_l4019_401964

theorem minimum_students_with_girl_percentage (n : ℕ) (g : ℕ) : n > 0 → g > 0 → (25 : ℚ) / 100 < (g : ℚ) / n → (g : ℚ) / n < (30 : ℚ) / 100 → n ≥ 7 :=
sorry

end minimum_students_with_girl_percentage_l4019_401964


namespace debate_team_girls_l4019_401920

/-- The number of girls on a debate team -/
def girls_on_team (total_students : ℕ) (boys : ℕ) : ℕ :=
  total_students - boys

theorem debate_team_girls :
  let total_students := 7 * 9
  let boys := 31
  girls_on_team total_students boys = 32 := by
  sorry

#check debate_team_girls

end debate_team_girls_l4019_401920


namespace desk_lamp_profit_l4019_401984

/-- Profit function for desk lamp sales -/
def profit_function (n : ℝ) (x : ℝ) : ℝ := (x - 20) * (-10 * x + n)

/-- Theorem stating the maximum profit and corresponding selling price -/
theorem desk_lamp_profit (n : ℝ) :
  (profit_function n 25 = 120) →
  (n = 370) ∧
  (∀ x : ℝ, x > 32 → profit_function n x ≤ 160) :=
by sorry

end desk_lamp_profit_l4019_401984


namespace quadratic_non_real_roots_l4019_401915

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by sorry

end quadratic_non_real_roots_l4019_401915


namespace closed_broken_line_length_lower_bound_l4019_401919

/-- A closed broken line on the surface of a unit cube -/
structure ClosedBrokenLine where
  /-- The line passes over the surface of the cube -/
  onSurface : Bool
  /-- The line has common points with all faces of the cube -/
  touchesAllFaces : Bool
  /-- The length of the line -/
  length : ℝ

/-- Theorem: The length of a closed broken line on a unit cube touching all faces is at least 3√2 -/
theorem closed_broken_line_length_lower_bound (line : ClosedBrokenLine) 
    (h1 : line.onSurface = true) 
    (h2 : line.touchesAllFaces = true) : 
  line.length ≥ 3 * Real.sqrt 2 :=
by
  sorry

end closed_broken_line_length_lower_bound_l4019_401919


namespace abs_a_minus_sqrt_a_squared_l4019_401917

theorem abs_a_minus_sqrt_a_squared (a : ℝ) (h : a < 0) : |a - Real.sqrt (a^2)| = -2*a := by
  sorry

end abs_a_minus_sqrt_a_squared_l4019_401917


namespace train_length_proof_l4019_401945

/-- Proves that the length of each train is 50 meters given the specified conditions -/
theorem train_length_proof (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 36) :
  let v_rel := (v_fast - v_slow) * (5 / 18)  -- Convert km/hr to m/s
  let l := v_rel * t / 2                     -- Length of one train
  l = 50 := by sorry

end train_length_proof_l4019_401945


namespace project_completion_time_l4019_401976

/-- Calculates the number of days needed to complete a project given extra hours,
    normal work hours, and total project hours. -/
def days_to_complete_project (extra_hours : ℕ) (normal_hours : ℕ) (project_hours : ℕ) : ℕ :=
  project_hours / (normal_hours + extra_hours)

/-- Theorem stating that under the given conditions, it takes 100 days to complete the project. -/
theorem project_completion_time :
  days_to_complete_project 5 10 1500 = 100 := by
  sorry

#eval days_to_complete_project 5 10 1500

end project_completion_time_l4019_401976


namespace knitting_productivity_ratio_l4019_401925

/-- Represents the knitting productivity of a girl -/
structure Knitter where
  work_time : ℕ  -- Time spent working before a break
  break_time : ℕ -- Duration of the break

/-- Calculates the total cycle time for a knitter -/
def cycle_time (k : Knitter) : ℕ := k.work_time + k.break_time

/-- Calculates the actual working time within a given period -/
def working_time (k : Knitter) (period : ℕ) : ℕ :=
  (period / cycle_time k) * k.work_time

theorem knitting_productivity_ratio :
  let girl1 : Knitter := ⟨5, 1⟩
  let girl2 : Knitter := ⟨7, 1⟩
  let common_period := Nat.lcm (cycle_time girl1) (cycle_time girl2)
  (working_time girl2 common_period : ℚ) / (working_time girl1 common_period) = 20/21 :=
sorry

end knitting_productivity_ratio_l4019_401925


namespace double_mean_value_range_l4019_401963

/-- A function is a double mean value function on an interval [a,b] if there exist
    x₁ and x₂ in (a,b) such that f'(x₁) = f'(x₂) = (f(b) - f(a)) / (b - a) -/
def IsDoubleMeanValueFunction (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    deriv f x₁ = (f b - f a) / (b - a) ∧
    deriv f x₂ = (f b - f a) / (b - a)

/-- The main theorem: if f(x) = x³ - 6/5x² is a double mean value function on [0,t],
    then 3/5 < t < 6/5 -/
theorem double_mean_value_range (t : ℝ) :
  IsDoubleMeanValueFunction (fun x => x^3 - 6/5*x^2) 0 t →
  3/5 < t ∧ t < 6/5 := by
  sorry

end double_mean_value_range_l4019_401963


namespace fastest_student_requires_comprehensive_survey_l4019_401997

/-- Represents a survey method -/
inductive SurveyMethod
| Comprehensive
| Sample

/-- Represents a survey scenario -/
structure SurveyScenario where
  description : String
  requiredMethod : SurveyMethod

/-- Define the four survey scenarios -/
def viewershipSurvey : SurveyScenario :=
  { description := "Investigating the viewership rate of the Spring Festival Gala"
    requiredMethod := SurveyMethod.Sample }

def colorantSurvey : SurveyScenario :=
  { description := "Investigating whether the colorant content of a certain food in the market meets national standards"
    requiredMethod := SurveyMethod.Sample }

def shoeSoleSurvey : SurveyScenario :=
  { description := "Testing the number of times the shoe soles produced by a shoe factory can withstand bending"
    requiredMethod := SurveyMethod.Sample }

def fastestStudentSurvey : SurveyScenario :=
  { description := "Selecting the fastest student in short-distance running at a certain school to participate in the city-wide competition"
    requiredMethod := SurveyMethod.Comprehensive }

/-- Theorem stating that selecting the fastest student requires a comprehensive survey -/
theorem fastest_student_requires_comprehensive_survey :
  fastestStudentSurvey.requiredMethod = SurveyMethod.Comprehensive ∧
  viewershipSurvey.requiredMethod ≠ SurveyMethod.Comprehensive ∧
  colorantSurvey.requiredMethod ≠ SurveyMethod.Comprehensive ∧
  shoeSoleSurvey.requiredMethod ≠ SurveyMethod.Comprehensive :=
sorry

end fastest_student_requires_comprehensive_survey_l4019_401997


namespace max_value_x_l4019_401973

theorem max_value_x : 
  ∃ (x_max : ℝ), 
    (∀ x : ℝ, ((5*x - 25)/(4*x - 5))^2 + ((5*x - 25)/(4*x - 5)) = 20 → x ≤ x_max) ∧
    ((5*x_max - 25)/(4*x_max - 5))^2 + ((5*x_max - 25)/(4*x_max - 5)) = 20 ∧
    x_max = 2 :=
by sorry

end max_value_x_l4019_401973


namespace even_function_problem_l4019_401930

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_problem (f : ℝ → ℝ) 
  (h1 : EvenFunction f) (h2 : f (-5) = 9) : f 5 = 9 := by
  sorry

end even_function_problem_l4019_401930


namespace circle_equations_from_line_intersections_l4019_401966

-- Define the line
def line (x y : ℝ) : Prop := 2 * x + y - 4 = 0

-- Define the two intersection points
def point_A : ℝ × ℝ := (0, 4)
def point_B : ℝ × ℝ := (2, 0)

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 20
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 20

-- Theorem statement
theorem circle_equations_from_line_intersections :
  (∀ x y : ℝ, line x y → (x = 0 ∧ y = 4) ∨ (x = 2 ∧ y = 0)) ∧
  (circle1 (point_A.1) (point_A.2) ∧ circle1 (point_B.1) (point_B.2)) ∧
  (circle2 (point_A.1) (point_A.2) ∧ circle2 (point_B.1) (point_B.2)) :=
sorry

end circle_equations_from_line_intersections_l4019_401966


namespace thursday_coffee_consumption_l4019_401960

/-- Represents the professor's coffee consumption model -/
structure CoffeeModel where
  k : ℝ
  coffee : ℝ → ℝ → ℝ
  wednesday_meetings : ℝ
  wednesday_sleep : ℝ
  wednesday_coffee : ℝ
  thursday_meetings : ℝ
  thursday_sleep : ℝ

/-- Theorem stating the professor's coffee consumption on Thursday -/
theorem thursday_coffee_consumption (model : CoffeeModel) 
  (h1 : model.coffee m h = model.k * m / h)
  (h2 : model.wednesday_coffee = model.coffee model.wednesday_meetings model.wednesday_sleep)
  (h3 : model.wednesday_meetings = 3)
  (h4 : model.wednesday_sleep = 8)
  (h5 : model.wednesday_coffee = 3)
  (h6 : model.thursday_meetings = 5)
  (h7 : model.thursday_sleep = 10) :
  model.coffee model.thursday_meetings model.thursday_sleep = 4 := by
  sorry

end thursday_coffee_consumption_l4019_401960


namespace sunset_time_correct_l4019_401995

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents the length of daylight -/
structure DaylightLength where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the sunset time given sunrise time and daylight length -/
def calculate_sunset (sunrise : Time) (daylight : DaylightLength) : Time :=
  sorry

theorem sunset_time_correct (sunrise : Time) (daylight : DaylightLength) :
  sunrise.hours = 6 ∧ sunrise.minutes = 43 ∧
  daylight.hours = 11 ∧ daylight.minutes = 56 →
  let sunset := calculate_sunset sunrise daylight
  sunset.hours = 18 ∧ sunset.minutes = 39 :=
sorry

end sunset_time_correct_l4019_401995


namespace range_of_m_l4019_401928

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 15 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2*x - m^2 + 1 ≤ 0

-- Define the property that ¬p is a necessary but not sufficient condition for ¬q
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  ∀ x, q x m → p x ∧ ∃ y, ¬(p y) ∧ q y m

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, not_p_necessary_not_sufficient_for_not_q m ↔ (m < -4 ∨ m > 4) :=
sorry

end range_of_m_l4019_401928


namespace odd_function_symmetry_l4019_401958

-- Define a real-valued function
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define symmetry about the y-axis for the absolute value of a function
def IsSymmetricAboutYAxis (f : RealFunction) : Prop :=
  ∀ x : ℝ, |f (-x)| = |f x|

-- Theorem statement
theorem odd_function_symmetry :
  (∀ f : RealFunction, IsOdd f → IsSymmetricAboutYAxis f) ∧
  (∃ f : RealFunction, IsSymmetricAboutYAxis f ∧ ¬IsOdd f) := by
  sorry

end odd_function_symmetry_l4019_401958


namespace complex_magnitude_equation_l4019_401998

theorem complex_magnitude_equation (x : ℝ) (h1 : x > 0) :
  Complex.abs (3 + 4 * x * Complex.I) = 5 → x = 1 := by
  sorry

end complex_magnitude_equation_l4019_401998


namespace average_weight_increase_l4019_401956

theorem average_weight_increase (initial_count : ℕ) (replaced_weight new_weight : ℝ) :
  initial_count = 8 →
  replaced_weight = 70 →
  new_weight = 94 →
  (new_weight - replaced_weight) / initial_count = 3 := by
sorry

end average_weight_increase_l4019_401956


namespace max_value_expression_l4019_401989

theorem max_value_expression (a b c d : ℝ) 
  (ha : -7 ≤ a ∧ a ≤ 7) 
  (hb : -7 ≤ b ∧ b ≤ 7) 
  (hc : -7 ≤ c ∧ c ≤ 7) 
  (hd : -7 ≤ d ∧ d ≤ 7) : 
  (∀ a' b' c' d' : ℝ, 
    -7 ≤ a' ∧ a' ≤ 7 → 
    -7 ≤ b' ∧ b' ≤ 7 → 
    -7 ≤ c' ∧ c' ≤ 7 → 
    -7 ≤ d' ∧ d' ≤ 7 → 
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' ≤ 210) ∧
  (∃ a' b' c' d' : ℝ, 
    -7 ≤ a' ∧ a' ≤ 7 ∧
    -7 ≤ b' ∧ b' ≤ 7 ∧
    -7 ≤ c' ∧ c' ≤ 7 ∧
    -7 ≤ d' ∧ d' ≤ 7 ∧
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' = 210) :=
by sorry

end max_value_expression_l4019_401989


namespace canoe_travel_time_l4019_401959

/-- Given two villages A and B connected by a river with current velocity v_r,
    and a canoe with velocity v in still water, prove that if the time to travel
    from A to B is 3 times the time to travel from B to A, and v = 2*v_r, then
    the time to travel from B to A without paddles is 3 times longer than with paddles. -/
theorem canoe_travel_time (v v_r : ℝ) (S : ℝ) (h1 : v > 0) (h2 : v_r > 0) (h3 : S > 0) :
  (S / (v + v_r) = 3 * S / (v - v_r)) → (v = 2 * v_r) → (S / v_r = 3 * S / (v - v_r)) := by
  sorry

end canoe_travel_time_l4019_401959


namespace car_journey_distance_l4019_401974

/-- Represents the car's journey with given speeds and break times -/
structure CarJourney where
  initial_speed : ℝ
  initial_duration : ℝ
  second_speed : ℝ
  second_duration : ℝ
  final_speed : ℝ
  final_duration : ℝ

/-- Calculates the total distance covered by the car -/
def total_distance (journey : CarJourney) : ℝ :=
  journey.initial_speed * journey.initial_duration +
  journey.second_speed * journey.second_duration +
  journey.final_speed * journey.final_duration

/-- Theorem stating that the car's journey covers 390 miles -/
theorem car_journey_distance :
  let journey : CarJourney := {
    initial_speed := 65,
    initial_duration := 2,
    second_speed := 60,
    second_duration := 2.5,
    final_speed := 55,
    final_duration := 2
  }
  total_distance journey = 390 := by sorry

end car_journey_distance_l4019_401974


namespace a_share_fraction_l4019_401942

/-- Prove that given the conditions, A's share is 2/3 of B and C's combined share -/
theorem a_share_fraction (total money : ℝ) (a_share : ℝ) (b_share : ℝ) (c_share : ℝ) : 
  total = 300 →
  a_share = 120.00000000000001 →
  b_share = (6/9) * (a_share + c_share) →
  total = a_share + b_share + c_share →
  a_share = (2/3) * (b_share + c_share) :=
by sorry


end a_share_fraction_l4019_401942


namespace book_words_per_page_l4019_401991

theorem book_words_per_page :
  ∀ (words_per_page : ℕ),
    words_per_page ≤ 120 →
    (150 * words_per_page) % 221 = 210 →
    words_per_page = 98 := by
  sorry

end book_words_per_page_l4019_401991


namespace stating_one_empty_neighborhood_probability_l4019_401951

/-- The number of neighborhoods --/
def num_neighborhoods : ℕ := 3

/-- The number of staff members --/
def num_staff : ℕ := 4

/-- The probability of exactly one neighborhood not being assigned any staff members --/
def probability_one_empty : ℚ := 14/27

/-- 
Theorem stating that the probability of exactly one neighborhood out of three 
not being assigned any staff members, when four staff members are independently 
assigned to the neighborhoods, is 14/27.
--/
theorem one_empty_neighborhood_probability : 
  (num_neighborhoods = 3 ∧ num_staff = 4) → 
  probability_one_empty = 14/27 := by
  sorry

end stating_one_empty_neighborhood_probability_l4019_401951


namespace area_PQR_is_sqrt_35_l4019_401938

/-- Represents a square pyramid with given dimensions and points -/
structure SquarePyramid where
  base_side : ℝ
  altitude : ℝ
  p_ratio : ℝ
  q_ratio : ℝ
  r_ratio : ℝ

/-- Calculates the area of triangle PQR in the square pyramid -/
def area_PQR (pyramid : SquarePyramid) : ℝ :=
  sorry

/-- Theorem stating that the area of triangle PQR is √35 for the given pyramid -/
theorem area_PQR_is_sqrt_35 :
  let pyramid := SquarePyramid.mk 4 8 (1/4) (1/4) (3/4)
  area_PQR pyramid = Real.sqrt 35 := by
  sorry

end area_PQR_is_sqrt_35_l4019_401938


namespace geometric_sequence_sum_l4019_401939

/-- Given a geometric sequence {a_n} with common ratio q = 2 and S_3 = 7, S_6 = 63 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- geometric sequence with common ratio 2
  (a 1 * (1 - 2^3)) / (1 - 2) = 7 →  -- S_3 = 7
  (a 1 * (1 - 2^6)) / (1 - 2) = 63 :=  -- S_6 = 63
by sorry

end geometric_sequence_sum_l4019_401939


namespace adult_ticket_price_l4019_401921

/-- Given information about ticket sales, prove the price of an adult ticket --/
theorem adult_ticket_price
  (student_price : ℝ)
  (total_tickets : ℕ)
  (total_revenue : ℝ)
  (student_tickets : ℕ)
  (h1 : student_price = 2.5)
  (h2 : total_tickets = 59)
  (h3 : total_revenue = 222.5)
  (h4 : student_tickets = 9) :
  (total_revenue - student_price * student_tickets) / (total_tickets - student_tickets) = 4 :=
by sorry

end adult_ticket_price_l4019_401921


namespace teddy_bears_per_shelf_l4019_401967

theorem teddy_bears_per_shelf (total_bears : ℕ) (num_shelves : ℕ) 
  (h1 : total_bears = 98) (h2 : num_shelves = 14) :
  (total_bears / num_shelves : ℕ) = 7 := by
  sorry

end teddy_bears_per_shelf_l4019_401967


namespace function_translation_l4019_401931

/-- Given a function f(x) = 3 * sin(2x + π/3), prove that translating it right by π/6 units
    and then downwards by 1 unit results in the function g(x) = 3 * sin(2x) - 1 -/
theorem function_translation (x : ℝ) :
  let f := λ x : ℝ => 3 * Real.sin (2 * x + π / 3)
  let g := λ x : ℝ => 3 * Real.sin (2 * x) - 1
  f (x - π / 6) - 1 = g x := by
  sorry

end function_translation_l4019_401931


namespace completing_square_l4019_401969

theorem completing_square (x : ℝ) : x^2 + 2*x - 3 = 0 ↔ (x + 1)^2 = 4 := by sorry

end completing_square_l4019_401969


namespace perpendicular_chords_sum_l4019_401926

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a chord passing through the focus
structure ChordThroughFocus where
  a : PointOnParabola
  b : PointOnParabola
  passes_through_focus : True  -- We assume this property without proving it

-- Define perpendicular chords
def perpendicular (c1 c2 : ChordThroughFocus) : Prop := True  -- We assume this property without proving it

-- Define the length of a chord
noncomputable def chord_length (c : ChordThroughFocus) : ℝ := sorry

-- Theorem statement
theorem perpendicular_chords_sum (ab cd : ChordThroughFocus) 
  (h_perp : perpendicular ab cd) : 
  1 / chord_length ab + 1 / chord_length cd = 1/4 := by sorry

end perpendicular_chords_sum_l4019_401926


namespace complex_solutions_count_l4019_401970

open Complex

theorem complex_solutions_count : 
  ∃ (S : Finset ℂ), (∀ z ∈ S, (z^4 - 1) / (z^2 + z + 1) = 0) ∧ 
                    (∀ z : ℂ, (z^4 - 1) / (z^2 + z + 1) = 0 → z ∈ S) ∧
                    Finset.card S = 4 := by
  sorry

end complex_solutions_count_l4019_401970


namespace line_equation_fourth_quadrant_triangle_l4019_401990

/-- Given a line passing through (-b, 0) and cutting a triangle with area T in the fourth quadrant,
    prove that its equation is 2Tx - b²y - 2bT = 0 --/
theorem line_equation_fourth_quadrant_triangle (b T : ℝ) (h₁ : b > 0) (h₂ : T > 0) : 
  ∃ (m c : ℝ), ∀ (x y : ℝ),
    (x = -b ∧ y = 0) ∨ (x ≥ 0 ∧ y ≤ 0 ∧ y = m * x + c) →
    (1/2 * b * (-y)) = T →
    2 * T * x - b^2 * y - 2 * b * T = 0 :=
by sorry

end line_equation_fourth_quadrant_triangle_l4019_401990


namespace division_problem_l4019_401987

theorem division_problem : (501 : ℝ) / (0.5 : ℝ) = 1002 := by sorry

end division_problem_l4019_401987


namespace log_product_theorem_l4019_401907

theorem log_product_theorem (c d : ℕ+) : 
  (d - c = 450) →
  (Real.log d / Real.log c = 3) →
  (c + d = 520) := by sorry

end log_product_theorem_l4019_401907


namespace simplify_expression_l4019_401948

theorem simplify_expression (a b c : ℝ) (h : (c - a) / (c - b) = 1) :
  (5 * b - 2 * a) / (c - a) = 3 * a / (c - a) :=
by sorry

end simplify_expression_l4019_401948


namespace road_repaving_l4019_401977

/-- Proves that the number of inches repaved before today is 4133,
    given the total repaved and the amount repaved today. -/
theorem road_repaving (total_repaved : ℕ) (repaved_today : ℕ)
    (h1 : total_repaved = 4938)
    (h2 : repaved_today = 805) :
    total_repaved - repaved_today = 4133 := by
  sorry

end road_repaving_l4019_401977


namespace stratified_sample_ninth_grade_l4019_401965

/-- Represents the number of students in each grade and the sample size for 7th grade -/
structure SchoolData where
  total : ℕ
  seventh : ℕ
  eighth : ℕ
  ninth : ℕ
  sample_seventh : ℕ

/-- Calculates the sample size for 9th grade using stratified sampling -/
def stratified_sample (data : SchoolData) : ℕ :=
  (data.sample_seventh * data.ninth) / data.seventh

/-- Theorem stating that the stratified sample for 9th grade is 224 given the school data -/
theorem stratified_sample_ninth_grade 
  (data : SchoolData) 
  (h1 : data.total = 1700)
  (h2 : data.seventh = 600)
  (h3 : data.eighth = 540)
  (h4 : data.ninth = 560)
  (h5 : data.sample_seventh = 240) :
  stratified_sample data = 224 := by
  sorry

end stratified_sample_ninth_grade_l4019_401965


namespace nancys_weight_calculation_l4019_401901

/-- Nancy's weight in pounds -/
def nancys_weight : ℝ := 90

/-- Nancy's daily water intake as a percentage of her body weight -/
def water_intake_percentage : ℝ := 60

/-- Nancy's daily water intake in pounds -/
def daily_water_intake : ℝ := 54

theorem nancys_weight_calculation :
  nancys_weight * (water_intake_percentage / 100) = daily_water_intake :=
by sorry

end nancys_weight_calculation_l4019_401901


namespace triangle_max_third_side_l4019_401900

theorem triangle_max_third_side (a b : ℝ) (ha : a = 4) (hb : b = 9) :
  ∃ (c : ℕ), c ≤ 12 ∧ 
  (∀ (d : ℕ), d > c → ¬(a + b > d ∧ a + d > b ∧ b + d > a)) :=
by sorry

end triangle_max_third_side_l4019_401900


namespace min_empty_cells_after_move_l4019_401940

/-- Represents a 3D box with dimensions width, height, and depth -/
structure Box where
  width : Nat
  height : Nat
  depth : Nat

/-- Represents the movement of cockchafers to neighboring cells -/
def move_to_neighbor (box : Box) : Nat :=
  sorry

/-- Theorem: In a 3x5x7 box, after cockchafers move to neighboring cells,
    the minimum number of empty cells is 1 -/
theorem min_empty_cells_after_move (box : Box) 
  (h1 : box.width = 3) 
  (h2 : box.height = 5) 
  (h3 : box.depth = 7) : 
  move_to_neighbor box = 1 := by
  sorry

end min_empty_cells_after_move_l4019_401940


namespace josie_money_left_l4019_401981

/-- Calculates the amount of money Josie has left after grocery shopping --/
def money_left_after_shopping (initial_amount : ℚ) (milk_price : ℚ) (bread_price : ℚ) 
  (detergent_price : ℚ) (banana_price_per_pound : ℚ) (banana_pounds : ℚ) 
  (milk_discount : ℚ) (detergent_discount : ℚ) : ℚ :=
  let milk_cost := milk_price * (1 - milk_discount)
  let detergent_cost := detergent_price - detergent_discount
  let banana_cost := banana_price_per_pound * banana_pounds
  let total_cost := milk_cost + bread_price + detergent_cost + banana_cost
  initial_amount - total_cost

/-- Theorem stating that Josie has $4.00 left after shopping --/
theorem josie_money_left : 
  money_left_after_shopping 20 4 3.5 10.25 0.75 2 0.5 1.25 = 4 := by
  sorry

end josie_money_left_l4019_401981


namespace b_amount_l4019_401912

theorem b_amount (a b : ℚ) 
  (h1 : a + b = 2530)
  (h2 : (3/5) * a = (2/7) * b) : 
  b = 1714 := by sorry

end b_amount_l4019_401912


namespace tom_search_days_l4019_401922

/-- Calculates the number of days Tom searched for an item given the daily rates and total cost -/
def search_days (initial_rate : ℕ) (initial_days : ℕ) (subsequent_rate : ℕ) (total_cost : ℕ) : ℕ :=
  let initial_cost := initial_rate * initial_days
  let remaining_cost := total_cost - initial_cost
  let additional_days := remaining_cost / subsequent_rate
  initial_days + additional_days

/-- Proves that Tom searched for 10 days given the specified rates and total cost -/
theorem tom_search_days :
  search_days 100 5 60 800 = 10 := by
  sorry

end tom_search_days_l4019_401922


namespace set1_equality_set2_equality_set3_equality_set4_equality_set5_equality_l4019_401979

-- Set 1
def set1 : Set ℤ := {x | x.natAbs ≤ 2}
def set1_alt : Set ℤ := {-2, -1, 0, 1, 2}

theorem set1_equality : set1 = set1_alt := by sorry

-- Set 2
def set2 : Set ℕ := {x | x > 0 ∧ x % 3 = 0 ∧ x < 10}
def set2_alt : Set ℕ := {3, 6, 9}

theorem set2_equality : set2 = set2_alt := by sorry

-- Set 3
def set3 : Set ℤ := {x | x = Int.natAbs x ∧ x < 5}
def set3_alt : Set ℤ := {0, 1, 2, 3, 4}

theorem set3_equality : set3 = set3_alt := by sorry

-- Set 4
def set4 : Set (ℕ+ × ℕ+) := {p | p.1 + p.2 = 6}
def set4_alt : Set (ℕ+ × ℕ+) := {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)}

theorem set4_equality : set4 = set4_alt := by sorry

-- Set 5
def set5 : Set ℤ := {-3, -1, 1, 3, 5}
def set5_alt : Set ℤ := {x | ∃ k : ℤ, x = 2*k - 1 ∧ -1 ≤ k ∧ k ≤ 3}

theorem set5_equality : set5 = set5_alt := by sorry

end set1_equality_set2_equality_set3_equality_set4_equality_set5_equality_l4019_401979


namespace brownies_per_pan_l4019_401955

/-- Proves that the number of pieces in each pan of brownies is 16 given the problem conditions --/
theorem brownies_per_pan (total_pans : ℕ) (eaten_pans : ℚ) (ice_cream_tubs : ℕ) 
  (scoops_per_tub : ℕ) (scoops_per_guest : ℕ) (guests_without_ice_cream : ℕ) :
  total_pans = 2 →
  eaten_pans = 1 + 3/4 →
  scoops_per_tub = 8 →
  scoops_per_guest = 2 →
  ice_cream_tubs = 6 →
  guests_without_ice_cream = 4 →
  ∃ (pieces_per_pan : ℕ), pieces_per_pan = 16 ∧ 
    (ice_cream_tubs * scoops_per_tub / scoops_per_guest + guests_without_ice_cream) / eaten_pans = pieces_per_pan := by
  sorry

#check brownies_per_pan

end brownies_per_pan_l4019_401955


namespace art_club_collection_l4019_401992

/-- The number of artworks collected by the art club in two school years -/
def artworks_collected (num_students : ℕ) (artworks_per_student_per_quarter : ℕ) (quarters_per_year : ℕ) (years : ℕ) : ℕ :=
  num_students * artworks_per_student_per_quarter * quarters_per_year * years

/-- Theorem stating that the art club collects 240 artworks in two school years -/
theorem art_club_collection :
  artworks_collected 15 2 4 2 = 240 := by
  sorry

#eval artworks_collected 15 2 4 2

end art_club_collection_l4019_401992


namespace common_terms_theorem_l4019_401999

def a (n : ℕ) : ℤ := 3 * n - 19
def b (n : ℕ) : ℕ := 2^n

-- c_n is the nth common term of sequences a and b in ascending order
def c (n : ℕ) : ℕ := 2^(2*n - 1)

theorem common_terms_theorem (n : ℕ) :
  ∃ (m k : ℕ), a m = b k ∧ c n = b k ∧ 
  (∀ (i j : ℕ), i < m ∧ j < k → a i ≠ b j) ∧
  (∀ (i j : ℕ), a i = b j → i ≥ m ∨ j ≥ k) :=
sorry

end common_terms_theorem_l4019_401999


namespace isosceles_triangles_height_ratio_l4019_401980

/-- Two isosceles triangles with equal vertical angles and areas in ratio 16:36 have heights in ratio 2:3 -/
theorem isosceles_triangles_height_ratio (b₁ b₂ h₁ h₂ : ℝ) (area₁ area₂ : ℝ) :
  b₁ > 0 → b₂ > 0 → h₁ > 0 → h₂ > 0 →
  area₁ = (b₁ * h₁) / 2 →
  area₂ = (b₂ * h₂) / 2 →
  area₁ / area₂ = 16 / 36 →
  b₁ / b₂ = h₁ / h₂ →
  h₁ / h₂ = 2 / 3 :=
by sorry

end isosceles_triangles_height_ratio_l4019_401980


namespace sum_of_multiples_is_even_l4019_401924

theorem sum_of_multiples_is_even (a b : ℤ) (ha : 4 ∣ a) (hb : 6 ∣ b) : 2 ∣ (a + b) := by
  sorry

end sum_of_multiples_is_even_l4019_401924


namespace inequality_solution_l4019_401962

theorem inequality_solution (x : ℝ) : 
  -1 < (x^2 - 16*x + 15) / (x^2 - 4*x + 5) ∧ 
  (x^2 - 16*x + 15) / (x^2 - 4*x + 5) < 1 ↔ 
  x < 5/2 := by sorry

end inequality_solution_l4019_401962


namespace solution_value_l4019_401927

theorem solution_value (p q : ℝ) : 
  (3 * p^2 - 5 * p = 12) → 
  (3 * q^2 - 5 * q = 12) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
  sorry

end solution_value_l4019_401927


namespace ians_jogging_laps_l4019_401985

/-- Given information about Ian's jogging routine, calculate the number of laps he does every night -/
theorem ians_jogging_laps 
  (lap_length : ℝ)
  (feet_per_calorie : ℝ)
  (total_calories : ℝ)
  (total_days : ℝ)
  (h1 : lap_length = 100)
  (h2 : feet_per_calorie = 25)
  (h3 : total_calories = 100)
  (h4 : total_days = 5)
  : (total_calories * feet_per_calorie / total_days) / lap_length = 5 := by
  sorry

end ians_jogging_laps_l4019_401985


namespace solve_factorial_equation_l4019_401935

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem solve_factorial_equation : ∃ (n : ℕ), n * factorial n + factorial n = 5040 ∧ n = 6 := by
  sorry

end solve_factorial_equation_l4019_401935


namespace summer_camp_group_size_l4019_401911

/-- The number of children in Mrs. Generous' summer camp group -/
def num_children : ℕ := 31

/-- The number of jelly beans Mrs. Generous brought -/
def total_jelly_beans : ℕ := 500

/-- The number of jelly beans left after distribution -/
def leftover_jelly_beans : ℕ := 10

/-- The difference between the number of boys and girls -/
def boy_girl_difference : ℕ := 3

theorem summer_camp_group_size :
  ∃ (girls boys : ℕ),
    girls + boys = num_children ∧
    boys = girls + boy_girl_difference ∧
    girls * girls + boys * boys = total_jelly_beans - leftover_jelly_beans :=
by sorry

end summer_camp_group_size_l4019_401911


namespace sin_graph_transformation_l4019_401972

theorem sin_graph_transformation (x : ℝ) :
  let f (x : ℝ) := 2 * Real.sin x
  let g (x : ℝ) := 2 * Real.sin (x / 3 + π / 6)
  let h (x : ℝ) := f (x + π / 6)
  g x = h (x / 3) := by
  sorry

end sin_graph_transformation_l4019_401972


namespace total_discount_percentage_l4019_401934

theorem total_discount_percentage (initial_discount subsequent_discount : ℝ) : 
  initial_discount = 0.25 → 
  subsequent_discount = 0.35 → 
  1 - (1 - initial_discount) * (1 - subsequent_discount) = 0.5125 := by
sorry

end total_discount_percentage_l4019_401934


namespace number_of_people_liking_apple_l4019_401916

/-- The number of people who like apple -/
def like_apple : ℕ := 40

/-- The number of people who like orange and mango but dislike apple -/
def like_orange_mango_not_apple : ℕ := 7

/-- The number of people who like mango and apple but dislike orange -/
def like_mango_apple_not_orange : ℕ := 10

/-- The number of people who like all three fruits -/
def like_all : ℕ := 4

/-- Theorem stating that the number of people who like apple is 40 -/
theorem number_of_people_liking_apple : 
  like_apple = 40 := by sorry

end number_of_people_liking_apple_l4019_401916


namespace constant_d_value_l4019_401961

theorem constant_d_value (e f d : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 2 * x + 4) * (e * x^2 + d * x + f) = 9 * x^4 - 8 * x^3 + 13 * x^2 + 12 * x - 16) →
  d = -2/3 := by
sorry

end constant_d_value_l4019_401961


namespace friend_team_assignments_l4019_401950

theorem friend_team_assignments (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k ^ n = 65536 := by
  sorry

end friend_team_assignments_l4019_401950


namespace calculate_expression_l4019_401982

theorem calculate_expression : 
  (Real.sqrt 2 / 2) * (2 * Real.sqrt 12 / (4 * Real.sqrt (1/8)) - 3 * Real.sqrt 48) = 
  2 * Real.sqrt 3 - 6 * Real.sqrt 6 := by
sorry

end calculate_expression_l4019_401982


namespace irrational_cubic_roots_not_quadratic_roots_l4019_401918

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Predicate to check if a number is a root of a cubic polynomial -/
def is_root_cubic (x : ℝ) (p : CubicPolynomial) : Prop :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d = 0

/-- Predicate to check if a number is a root of a quadratic polynomial -/
def is_root_quadratic (x : ℝ) (q : QuadraticPolynomial) : Prop :=
  q.a * x^2 + q.b * x + q.c = 0

/-- Main theorem -/
theorem irrational_cubic_roots_not_quadratic_roots
  (p : CubicPolynomial)
  (h1 : ∃ x y z : ℝ, is_root_cubic x p ∧ is_root_cubic y p ∧ is_root_cubic z p)
  (h2 : ∀ x : ℝ, is_root_cubic x p → Irrational x)
  : ∀ q : QuadraticPolynomial, ∀ x : ℝ, is_root_cubic x p → ¬ is_root_quadratic x q :=
sorry

end irrational_cubic_roots_not_quadratic_roots_l4019_401918


namespace rhombus_area_l4019_401994

-- Define the rhombus
def Rhombus (perimeter : ℝ) (diagonal1 : ℝ) : Prop :=
  perimeter > 0 ∧ diagonal1 > 0

-- Theorem statement
theorem rhombus_area 
  (perimeter : ℝ) 
  (diagonal1 : ℝ) 
  (h : Rhombus perimeter diagonal1) 
  (h_perimeter : perimeter = 80) 
  (h_diagonal : diagonal1 = 36) : 
  ∃ (area : ℝ), area = 72 * Real.sqrt 19 :=
sorry

end rhombus_area_l4019_401994


namespace problem_zeros_count_l4019_401913

/-- The number of zeros in the binary representation of a natural number -/
def countZeros (n : ℕ) : ℕ := sorry

/-- The expression given in the problem -/
def problemExpression : ℕ := 
  ((18 * 8192 + 8 * 128 - 12 * 16) / 6 + 4 * 64 + 3^5 - (25 * 2))

/-- Theorem stating that the number of zeros in the binary representation of the problem expression is 6 -/
theorem problem_zeros_count : countZeros problemExpression = 6 := by sorry

end problem_zeros_count_l4019_401913


namespace units_digit_of_7_pow_6_squared_l4019_401957

def units_digit_pattern : List Nat := [7, 9, 3, 1]

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_7_pow_6_squared : 
  units_digit (7^(6^2)) = 1 :=
by
  sorry

end units_digit_of_7_pow_6_squared_l4019_401957


namespace grid_triangle_square_l4019_401902

/-- A point on a 2D grid represented by integer coordinates -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- The area of a triangle formed by three grid points -/
def triangleArea (A B C : GridPoint) : ℚ := sorry

/-- The squared distance between two grid points -/
def squaredDistance (A B : GridPoint) : ℤ := sorry

/-- Predicate to check if three grid points form three vertices of a square -/
def formSquareVertices (A B C : GridPoint) : Prop := sorry

theorem grid_triangle_square (A B C : GridPoint) :
  let T := triangleArea A B C
  (squaredDistance A B + squaredDistance B C)^2 < 8 * T + 1 →
  formSquareVertices A B C := by
  sorry

end grid_triangle_square_l4019_401902


namespace largest_angle_of_special_hexagon_l4019_401944

-- Define a hexagon type
structure Hexagon where
  angles : Fin 6 → ℝ
  is_convex : True
  consecutive_integers : ∀ i : Fin 5, ∃ n : ℤ, angles i.succ = angles i + 1
  sum_720 : (Finset.univ.sum angles) = 720

-- Theorem statement
theorem largest_angle_of_special_hexagon (h : Hexagon) :
  Finset.max' (Finset.univ.image h.angles) (by sorry) = 122.5 := by
  sorry

end largest_angle_of_special_hexagon_l4019_401944


namespace tenth_term_is_neg_512_l4019_401952

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℤ  -- The sequence (a₁, a₂, a₃, ...)
  is_geometric : ∀ n : ℕ, n ≥ 2 → a (n + 1) / a n = a 2 / a 1
  product_25 : a 2 * a 5 = -32
  sum_34 : a 3 + a 4 = 4
  integer_ratio : ∃ q : ℤ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = q * a n

/-- The 10th term of the geometric sequence is -512 -/
theorem tenth_term_is_neg_512 (seq : GeometricSequence) : seq.a 10 = -512 := by
  sorry

end tenth_term_is_neg_512_l4019_401952


namespace unique_solution_l4019_401910

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that 1958 is the unique solution -/
theorem unique_solution : ∃! n : ℕ, n + S n = 1981 ∧ n = 1958 := by sorry

end unique_solution_l4019_401910


namespace root_property_l4019_401908

theorem root_property (a : ℝ) : a^2 - 2*a - 5 = 0 → 2*a^2 - 4*a = 10 := by
  sorry

end root_property_l4019_401908


namespace smallest_positive_integer_form_l4019_401905

theorem smallest_positive_integer_form (m n : ℤ) :
  (∃ k : ℕ+, k = |4509 * m + 27981 * n| ∧ 
   ∀ j : ℕ+, (∃ a b : ℤ, j = |4509 * a + 27981 * b|) → k ≤ j) ↔ 
  Nat.gcd 4509 27981 = 3 :=
sorry

end smallest_positive_integer_form_l4019_401905


namespace population_ratio_problem_l4019_401947

theorem population_ratio_problem (s v : ℝ) 
  (h1 : 0.94 * s = 1.14 * v) : s / v = 57 / 47 := by
  sorry

end population_ratio_problem_l4019_401947


namespace farm_area_and_planned_days_correct_l4019_401904

/-- Represents the farm field and ploughing scenario -/
structure FarmField where
  planned_daily_area : ℝ
  actual_daily_area : ℝ
  type_a_percentage : ℝ
  type_b_percentage : ℝ
  type_c_percentage : ℝ
  type_a_hours_per_hectare : ℝ
  type_b_hours_per_hectare : ℝ
  type_c_hours_per_hectare : ℝ
  extra_days_worked : ℕ
  area_left_to_plough : ℝ
  max_hours_per_day : ℝ

/-- Calculates the total area of the farm field and the initially planned work days -/
def calculate_farm_area_and_planned_days (field : FarmField) : ℝ × ℕ :=
  sorry

/-- Theorem stating the correct total area and initially planned work days -/
theorem farm_area_and_planned_days_correct (field : FarmField) 
  (h1 : field.planned_daily_area = 260)
  (h2 : field.actual_daily_area = 85)
  (h3 : field.type_a_percentage = 0.4)
  (h4 : field.type_b_percentage = 0.3)
  (h5 : field.type_c_percentage = 0.3)
  (h6 : field.type_a_hours_per_hectare = 4)
  (h7 : field.type_b_hours_per_hectare = 6)
  (h8 : field.type_c_hours_per_hectare = 3)
  (h9 : field.extra_days_worked = 2)
  (h10 : field.area_left_to_plough = 40)
  (h11 : field.max_hours_per_day = 12) :
  calculate_farm_area_and_planned_days field = (340, 2) :=
by
  sorry

end farm_area_and_planned_days_correct_l4019_401904


namespace sunflower_seeds_majority_l4019_401906

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  sunflowerSeeds : Rat
  otherSeeds : Rat

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    sunflowerSeeds := state.sunflowerSeeds * (4/5) + (2/5),
    otherSeeds := 3/5 }

/-- Initial state of the feeder on Monday -/
def initialState : FeederState :=
  { day := 1,
    sunflowerSeeds := 2/5,
    otherSeeds := 3/5 }

/-- Theorem stating that on the third day, more than half the seeds are sunflower seeds -/
theorem sunflower_seeds_majority : 
  let state3 := nextDay (nextDay initialState)
  state3.sunflowerSeeds > (state3.sunflowerSeeds + state3.otherSeeds) / 2 := by
  sorry


end sunflower_seeds_majority_l4019_401906


namespace valid_selections_count_l4019_401932

/-- The number of students in the group -/
def total_students : ℕ := 7

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of ways to select 4 students from 7, where at least one of A and B participates,
    and when both participate, their speeches are not adjacent -/
def valid_selections : ℕ := sorry

theorem valid_selections_count : valid_selections = 600 := by sorry

end valid_selections_count_l4019_401932


namespace jacket_price_reduction_l4019_401903

theorem jacket_price_reduction (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * (1 - 0.30) * (1 + 0.5873) = P → x = 10 := by
  sorry

end jacket_price_reduction_l4019_401903


namespace relationship_between_p_and_q_l4019_401975

theorem relationship_between_p_and_q (p q : ℝ) (h : p > 0) (h' : q > 0) (h'' : q ≠ 1) 
  (eq : Real.log p + Real.log q = Real.log (p + q + q^2)) : 
  p = (q + q^2) / (q - 1) := by
  sorry

end relationship_between_p_and_q_l4019_401975


namespace complex_number_equalities_l4019_401953

/-- Prove complex number equalities -/
theorem complex_number_equalities :
  let i : ℂ := Complex.I
  let z₁ : ℂ := (1 + 2*i)^2 + 3*(1 - i)
  let z₂ : ℂ := 2 + i
  let z₃ : ℂ := 1 - i
  let z₄ : ℂ := 1 + i
  let z₅ : ℂ := 1 - Complex.I * Real.sqrt 3
  let z₆ : ℂ := Complex.I * Real.sqrt 3 + i
  (z₁ / z₂ = 1/5 + 2/5*i) ∧
  (z₃ / z₄^2 + z₄ / z₃^2 = -1) ∧
  (z₅ / z₆^2 = -1/4 - (Real.sqrt 3)/4*i) := by
  sorry

end complex_number_equalities_l4019_401953


namespace max_puzzle_sets_l4019_401943

/-- Represents a set of puzzles -/
structure PuzzleSet where
  logic : ℕ
  visual : ℕ
  word : ℕ

/-- Checks if a PuzzleSet is valid according to the given conditions -/
def isValidSet (s : PuzzleSet) : Prop :=
  7 ≤ s.logic + s.visual + s.word ∧
  s.logic + s.visual + s.word ≤ 12 ∧
  4 * s.visual = 3 * s.logic ∧
  2 * s.word ≥ s.visual

/-- The main theorem stating the maximum number of sets that can be created -/
theorem max_puzzle_sets :
  ∃ (s : PuzzleSet),
    isValidSet s ∧
    (∃ (n : ℕ), n = 5 ∧
      n * s.logic = 36 ∧
      n * s.visual = 27 ∧
      n * s.word = 15) ∧
    (∀ (m : ℕ) (t : PuzzleSet),
      isValidSet t →
      m * t.logic ≤ 36 ∧
      m * t.visual ≤ 27 ∧
      m * t.word ≤ 15 →
      m ≤ 5) :=
sorry

end max_puzzle_sets_l4019_401943


namespace characterize_nat_function_l4019_401996

/-- A function from natural numbers to natural numbers -/
def NatFunction := ℕ → ℕ

/-- Predicate that checks if a number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- Theorem statement -/
theorem characterize_nat_function (f : NatFunction) :
  (∀ m n : ℕ, IsPerfectSquare (f n + 2 * m * n + f m)) →
  ∃ ℓ : ℕ, ∀ n : ℕ, f n = (n + 2 * ℓ)^2 - 2 * ℓ^2 :=
by sorry

end characterize_nat_function_l4019_401996


namespace quadratic_equation_from_root_properties_l4019_401949

theorem quadratic_equation_from_root_properties (a b c : ℝ) :
  (∀ x y : ℝ, x + y = 10 ∧ x * y = 24 → a * x^2 + b * x + c = 0) →
  a = 1 ∧ b = -10 ∧ c = 24 := by
sorry

end quadratic_equation_from_root_properties_l4019_401949


namespace power_of_ten_zeros_l4019_401968

theorem power_of_ten_zeros (n : ℕ) : 10000 ^ 50 * 10 ^ 5 = 10 ^ 205 := by
  sorry

end power_of_ten_zeros_l4019_401968


namespace cubic_minus_x_factorization_l4019_401946

theorem cubic_minus_x_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end cubic_minus_x_factorization_l4019_401946


namespace average_height_l4019_401986

def heights_problem (h₁ h₂ h₃ h₄ : ℝ) : Prop :=
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ ∧
  h₂ - h₁ = 2 ∧
  h₃ - h₂ = 2 ∧
  h₄ - h₃ = 6 ∧
  h₄ = 83

theorem average_height (h₁ h₂ h₃ h₄ : ℝ) 
  (hproblem : heights_problem h₁ h₂ h₃ h₄) : 
  (h₁ + h₂ + h₃ + h₄) / 4 = 77 := by
  sorry

end average_height_l4019_401986


namespace cool_parents_problem_l4019_401993

theorem cool_parents_problem (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ)
  (h1 : total = 40)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 22)
  (h4 : both_cool = 10) :
  total - (cool_dads + cool_moms - both_cool) = 10 := by
  sorry

end cool_parents_problem_l4019_401993


namespace gcf_36_60_l4019_401988

theorem gcf_36_60 : Nat.gcd 36 60 = 12 := by
  sorry

end gcf_36_60_l4019_401988


namespace expression_simplification_l4019_401954

theorem expression_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  ((x + 2*a)^2) / ((a - b)*(a - c)) + ((x + 2*b)^2) / ((b - a)*(b - c)) + ((x + 2*c)^2) / ((c - a)*(c - b)) = 4 := by
  sorry

end expression_simplification_l4019_401954


namespace max_value_fourth_root_sum_l4019_401971

theorem max_value_fourth_root_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hsum : a + b + c + d ≤ 4) :
  (a^2 + 3*a*b)^(1/4) + (b^2 + 3*b*c)^(1/4) + (c^2 + 3*c*d)^(1/4) + (d^2 + 3*d*a)^(1/4) ≤ 4 * Real.sqrt 2 :=
by sorry

end max_value_fourth_root_sum_l4019_401971


namespace shooting_competition_probability_l4019_401936

theorem shooting_competition_probability (p_10 p_9 p_8 p_7 : ℝ) 
  (h1 : p_10 = 0.15)
  (h2 : p_9 = 0.35)
  (h3 : p_8 = 0.2)
  (h4 : p_7 = 0.1) :
  p_7 = 0.3 :=
sorry

end shooting_competition_probability_l4019_401936


namespace two_digit_number_puzzle_l4019_401937

theorem two_digit_number_puzzle : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 1000 + 100 * (n / 10) + 10 * (n % 10) + 1 = 23 * n :=
by
  -- The proof goes here
  sorry

end two_digit_number_puzzle_l4019_401937

import Mathlib

namespace range_of_m_l1269_126909

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  4 * x₁^2 - 4*(m-2)*x₁ + 1 = 0 ∧ 4 * x₂^2 - 4*(m-2)*x₂ + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 3*m*x + 1 ≠ 0

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(q m) →
  (m ≤ -2/3) ∨ (2/3 ≤ m ∧ m < 1) :=
sorry

end range_of_m_l1269_126909


namespace ellipse_range_l1269_126950

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 12

-- Define the function we're interested in
def f (x y : ℝ) : ℝ := x + 2 * y

-- Theorem statement
theorem ellipse_range :
  ∀ x y : ℝ, on_ellipse x y → -Real.sqrt 22 ≤ f x y ∧ f x y ≤ Real.sqrt 22 := by
  sorry

end ellipse_range_l1269_126950


namespace factorial_division_l1269_126940

theorem factorial_division (ten_factorial : ℕ) (h : ten_factorial = 3628800) :
  ten_factorial / 24 = 151200 := by
  sorry

end factorial_division_l1269_126940


namespace power_of_two_divisibility_l1269_126912

theorem power_of_two_divisibility (n a b : ℕ) : 
  2^n = 10*a + b → n > 3 → 0 < b → b < 10 → ∃ k, ab = 6*k := by
  sorry

end power_of_two_divisibility_l1269_126912


namespace article_selling_prices_l1269_126937

/-- Represents the selling price of an article -/
def selling_price (cost_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  cost_price * (1 + profit_percentage / 100)

/-- Theorem stating the selling prices of three articles given their cost prices and profit/loss percentages -/
theorem article_selling_prices :
  let article1_sp := 500
  let article2_cp := 800
  let article3_cp := 1800
  (selling_price (article1_sp / 1.25) 25 = article1_sp) ∧
  (selling_price article2_cp (-25) = 600) ∧
  (selling_price article3_cp 50 = 2700) := by
  sorry


end article_selling_prices_l1269_126937


namespace flour_already_added_l1269_126948

/-- Given a cake recipe and Mary's current progress, calculate how many cups of flour
    she has already put in. -/
theorem flour_already_added
  (total_required : ℕ)  -- Total cups of flour required by the recipe
  (more_needed : ℕ)     -- Cups of flour Mary still needs to add
  (h1 : total_required = 9)  -- The recipe requires 9 cups of flour
  (h2 : more_needed = 7)     -- Mary needs to add 7 more cups
  : total_required - more_needed = 2 := by
  sorry

end flour_already_added_l1269_126948


namespace work_completion_days_l1269_126907

/-- Calculates the number of days needed for the remaining workers to complete a job -/
def daysToComplete (originalWorkers : ℕ) (plannedDays : ℕ) (absentWorkers : ℕ) : ℕ :=
  (originalWorkers * plannedDays) / (originalWorkers - absentWorkers)

/-- Proves that given the original conditions, the remaining workers complete the job in 21 days -/
theorem work_completion_days :
  daysToComplete 42 17 8 = 21 := by
  sorry

end work_completion_days_l1269_126907


namespace solution_az_eq_b_l1269_126974

theorem solution_az_eq_b (a b : ℝ) : 
  (∃! x, 4 + 3 * a * x = 2 * a - 7) →
  (¬∃ y, 2 + y = (b + 1) * y) →
  (∀ z, a * z = b ↔ z = 0) :=
by sorry

end solution_az_eq_b_l1269_126974


namespace geometric_sequence_seventh_term_l1269_126993

/-- Given a geometric sequence of positive numbers where the fifth term is 32 and the eleventh term is 2, the seventh term is 8. -/
theorem geometric_sequence_seventh_term (a : ℝ) (r : ℝ) (h1 : a * r^4 = 32) (h2 : a * r^10 = 2) :
  a * r^6 = 8 := by
  sorry

end geometric_sequence_seventh_term_l1269_126993


namespace three_prime_divisors_of_eight_power_minus_one_l1269_126982

theorem three_prime_divisors_of_eight_power_minus_one (n : ℕ) :
  let x := 8^n - 1
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    x = p * q * r) →
  (31 ∣ x) →
  x = 32767 := by
sorry

end three_prime_divisors_of_eight_power_minus_one_l1269_126982


namespace cos_angle_between_vectors_l1269_126961

def a : ℝ × ℝ := (3, 3)
def b : ℝ × ℝ := (1, 2)

theorem cos_angle_between_vectors :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (Real.cos θ) = 3 * Real.sqrt 10 / 10 := by
  sorry

end cos_angle_between_vectors_l1269_126961


namespace vector_equality_holds_l1269_126966

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def vector_equality (e₁ e₂ a b : V) : Prop :=
  (e₁ ≠ 0 ∧ e₂ ≠ 0) ∧  -- non-zero vectors
  (∀ (r : ℝ), r • e₁ ≠ e₂) ∧  -- non-collinear
  (a = 3 • e₁ - 2 • e₂) ∧
  (b = e₂ - 2 • e₁) ∧
  ((1/3) • a + b) + (a - (3/2) • b) + (2 • b - a) = -2 • e₁ + (5/6) • e₂

theorem vector_equality_holds (e₁ e₂ a b : V) :
  vector_equality e₁ e₂ a b := by sorry

end vector_equality_holds_l1269_126966


namespace otimes_neg_two_neg_one_l1269_126990

-- Define the custom operation ⊗
def otimes (a b : ℝ) : ℝ := a^2 - abs b

-- Theorem statement
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end otimes_neg_two_neg_one_l1269_126990


namespace f_fixed_points_l1269_126997

def f (x : ℝ) : ℝ := x^3 - 3*x^2

theorem f_fixed_points : 
  ∃ (x : ℝ), (f (f x) = f x) ∧ (x = 0 ∨ x = 3) :=
by sorry

end f_fixed_points_l1269_126997


namespace decimal_to_fraction_sum_l1269_126901

theorem decimal_to_fraction_sum (c d : ℕ+) : 
  (c : ℚ) / (d : ℚ) = 0.325 ∧ 
  (∀ (k : ℕ+), k ∣ c ∧ k ∣ d → k = 1) →
  (c : ℕ) + d = 53 := by
  sorry

end decimal_to_fraction_sum_l1269_126901


namespace inscribed_polygon_sides_l1269_126942

theorem inscribed_polygon_sides (n : ℕ) (s : ℝ) : n ≥ 3 →
  s = 2 * Real.sin (Real.pi / n) →  -- side length formula
  1 < s →
  s < Real.sqrt 2 →
  n = 5 := by sorry

end inscribed_polygon_sides_l1269_126942


namespace geometric_sum_6_terms_l1269_126903

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 6 terms of the geometric series with first term 2 and common ratio 1/3 -/
theorem geometric_sum_6_terms :
  geometricSum 2 (1/3) 6 = 728/243 := by
sorry

end geometric_sum_6_terms_l1269_126903


namespace vector_subtraction_l1269_126984

def vector_a : ℝ × ℝ := (-1, 2)
def vector_b : ℝ × ℝ := (0, 1)

theorem vector_subtraction :
  vector_a - 2 • vector_b = (-1, 0) := by sorry

end vector_subtraction_l1269_126984


namespace percent_to_decimal_twenty_five_percent_value_l1269_126927

theorem percent_to_decimal (x : ℚ) : x / 100 = x * (1 / 100) := by sorry

theorem twenty_five_percent_value : (25 : ℚ) / 100 = (1 : ℚ) / 4 := by sorry

end percent_to_decimal_twenty_five_percent_value_l1269_126927


namespace prime_square_sum_equation_l1269_126915

theorem prime_square_sum_equation (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ (x y z : ℕ), p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by sorry

end prime_square_sum_equation_l1269_126915


namespace international_long_haul_all_services_probability_l1269_126988

/-- Represents a flight route -/
inductive FlightRoute
| Domestic
| InternationalShortHaul
| InternationalLongHaul

/-- Represents a service offered on a flight -/
inductive Service
| WirelessInternet
| FreeSnacks
| EntertainmentSystem
| ExtraLegroom

/-- Returns the probability of a service being offered on a given flight route -/
def serviceProbability (route : FlightRoute) (service : Service) : ℝ :=
  match route, service with
  | FlightRoute.InternationalLongHaul, Service.WirelessInternet => 0.65
  | FlightRoute.InternationalLongHaul, Service.FreeSnacks => 0.80
  | FlightRoute.InternationalLongHaul, Service.EntertainmentSystem => 0.75
  | FlightRoute.InternationalLongHaul, Service.ExtraLegroom => 0.70
  | _, _ => 0  -- Default case, not used in this problem

/-- The probability of experiencing all services on a given flight route -/
def allServicesProbability (route : FlightRoute) : ℝ :=
  (serviceProbability route Service.WirelessInternet) *
  (serviceProbability route Service.FreeSnacks) *
  (serviceProbability route Service.EntertainmentSystem) *
  (serviceProbability route Service.ExtraLegroom)

/-- Theorem: The probability of experiencing all services on an international long-haul flight is 0.273 -/
theorem international_long_haul_all_services_probability :
  allServicesProbability FlightRoute.InternationalLongHaul = 0.273 := by
  sorry

end international_long_haul_all_services_probability_l1269_126988


namespace parallel_lines_distance_l1269_126947

def line1 (t : ℝ) : ℝ × ℝ := (3 + 2*t, -2 - 5*t)
def line2 (s : ℝ) : ℝ × ℝ := (4 + 2*s, -6 - 5*s)

def direction : ℝ × ℝ := (2, -5)

theorem parallel_lines_distance :
  let v := (3 - 4, -2 - (-6))
  let proj_v := ((v.1 * direction.1 + v.2 * direction.2) / (direction.1^2 + direction.2^2)) • direction
  let c := (4 + proj_v.1, -6 + proj_v.2)
  Real.sqrt ((3 - c.1)^2 + (-2 - c.2)^2) = 31 / 29 := by sorry

end parallel_lines_distance_l1269_126947


namespace joyful_joan_problem_l1269_126939

theorem joyful_joan_problem (a b c d e : ℚ) : 
  a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 5 →
  (a + b + c + d + e = a + (b + (c - (d * e)))) →
  e = -5/6 := by
sorry

end joyful_joan_problem_l1269_126939


namespace jellybean_bags_l1269_126930

theorem jellybean_bags (initial_average : ℕ) (new_bag_jellybeans : ℕ) (new_average : ℕ) :
  initial_average = 117 →
  new_bag_jellybeans = 362 →
  new_average = 124 →
  ∃ n : ℕ, n * initial_average + new_bag_jellybeans = (n + 1) * new_average ∧ n = 34 :=
by
  sorry

end jellybean_bags_l1269_126930


namespace amys_final_money_l1269_126911

def amys_money (initial_amount : ℚ) (chore_payment : ℚ) (num_neighbors : ℕ) 
  (birthday_money : ℚ) (investment_percentage : ℚ) (investment_return : ℚ) 
  (toy_cost : ℚ) (grandparent_multiplier : ℚ) (donation_percentage : ℚ) : ℚ :=
  let total_before_investment := initial_amount + chore_payment * num_neighbors + birthday_money
  let invested_amount := total_before_investment * investment_percentage
  let investment_value := invested_amount * (1 + investment_return)
  let remaining_after_toy := total_before_investment - toy_cost
  let after_grandparent_gift := remaining_after_toy * grandparent_multiplier
  let total_before_donation := after_grandparent_gift + investment_value
  let final_amount := total_before_donation * (1 - donation_percentage)
  final_amount

theorem amys_final_money :
  amys_money 2 13 5 3 (20/100) (10/100) 12 2 (25/100) = 98.55 := by
  sorry

end amys_final_money_l1269_126911


namespace clock_angle_at_3_30_angle_between_clock_hands_at_3_30_l1269_126979

/-- The angle between clock hands at 3:30 -/
theorem clock_angle_at_3_30 : ℝ :=
  let hour_hand_angle : ℝ := 3.5 * 30  -- 3:30 is 3.5 hours from 12 o'clock
  let minute_hand_angle : ℝ := 30 * 6  -- 30 minutes is 6 times 5-minute marks
  let angle_diff : ℝ := |minute_hand_angle - hour_hand_angle|
  75

/-- Theorem: The angle between the hour and minute hands at 3:30 is 75 degrees -/
theorem angle_between_clock_hands_at_3_30 : clock_angle_at_3_30 = 75 := by
  sorry

end clock_angle_at_3_30_angle_between_clock_hands_at_3_30_l1269_126979


namespace x_plus_y_value_l1269_126914

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.sin y = 2010)
  (h2 : x + 2010 * Real.cos y = 2009)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 + Real.pi / 2 := by
sorry

end x_plus_y_value_l1269_126914


namespace f_derivative_at_one_l1269_126999

-- Define the function f
def f (x : ℝ) : ℝ := (x+3)*(x+2)*(x+1)*x*(x-1)*(x-2)*(x-3)

-- State the theorem
theorem f_derivative_at_one : 
  deriv f 1 = 48 := by sorry

end f_derivative_at_one_l1269_126999


namespace zoo_visitors_l1269_126957

theorem zoo_visitors (saturday_visitors : ℕ) (day_visitors : ℕ) : 
  saturday_visitors = 3750 → 
  saturday_visitors = 3 * day_visitors → 
  day_visitors = 1250 := by
sorry

end zoo_visitors_l1269_126957


namespace only_airplane_survey_comprehensive_l1269_126960

/-- Represents a type of survey --/
inductive SurveyType
  | WaterQuality
  | AirplanePassengers
  | PlasticBags
  | TVViewership

/-- Predicate to determine if a survey type is suitable for comprehensive surveying --/
def is_comprehensive (s : SurveyType) : Prop :=
  match s with
  | SurveyType.AirplanePassengers => true
  | _ => false

/-- Theorem stating that only the airplane passenger survey is comprehensive --/
theorem only_airplane_survey_comprehensive :
  ∀ s : SurveyType, is_comprehensive s ↔ s = SurveyType.AirplanePassengers :=
by
  sorry

#check only_airplane_survey_comprehensive

end only_airplane_survey_comprehensive_l1269_126960


namespace determinant_zero_exists_l1269_126923

def matrix (x a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![x + a, x + b, x + c],
    ![x + b, x + c, x + a],
    ![x + c, x + a, x + b]]

theorem determinant_zero_exists (a b c : ℝ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
  (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) : 
  ∃ x : ℝ, Matrix.det (matrix x a b c) = 0 := by
  sorry

end determinant_zero_exists_l1269_126923


namespace line_through_point_parallel_to_line_l1269_126968

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line
  (A : Point)
  (given_line : Line)
  (h_A : A.x = 3 ∧ A.y = 2)
  (h_given : given_line.a = 4 ∧ given_line.b = 1 ∧ given_line.c = -2)
  : ∃ (result_line : Line),
    result_line.a = 4 ∧ 
    result_line.b = 1 ∧ 
    result_line.c = -14 ∧
    A.liesOn result_line ∧
    result_line.isParallelTo given_line :=
  sorry

end line_through_point_parallel_to_line_l1269_126968


namespace rectangle_area_diagonal_l1269_126967

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end rectangle_area_diagonal_l1269_126967


namespace right_triangle_30_60_90_l1269_126998

theorem right_triangle_30_60_90 (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_angle : a / c = 1 / 2) (h_hypotenuse : c = 10) : 
  a = 5 * Real.sqrt 3 := by
sorry

end right_triangle_30_60_90_l1269_126998


namespace min_reciprocal_sum_min_reciprocal_sum_attainable_l1269_126983

theorem min_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  1/a + 1/b + 1/c ≥ 9 := by
  sorry

theorem min_reciprocal_sum_attainable : 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ 1/a + 1/b + 1/c = 9 := by
  sorry

end min_reciprocal_sum_min_reciprocal_sum_attainable_l1269_126983


namespace arithmetic_mean_problem_l1269_126936

theorem arithmetic_mean_problem (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 17 →
  a = 12 →
  b = 20 →
  c = d →
  c * d = 324 :=
by
  sorry

end arithmetic_mean_problem_l1269_126936


namespace f_six_eq_zero_l1269_126938

def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_six_eq_zero
  (f : ℝ → ℝ)
  (hodd : isOddFunction f)
  (hperiod : ∀ x, f (x + 2) = -f x) :
  f 6 = 0 := by
sorry

end f_six_eq_zero_l1269_126938


namespace rod_length_calculation_l1269_126959

/-- The total length of a rod that can be cut into a given number of pieces of a specific length. -/
def rod_length (num_pieces : ℕ) (piece_length : ℝ) : ℝ :=
  num_pieces * piece_length

/-- Theorem stating that a rod that can be cut into 50 pieces of 0.85 metres each has a total length of 42.5 metres. -/
theorem rod_length_calculation : rod_length 50 0.85 = 42.5 := by
  sorry

end rod_length_calculation_l1269_126959


namespace slope_of_line_l1269_126949

/-- The slope of a line defined by the equation 4y = 5x + 20 is 5/4. -/
theorem slope_of_line (x y : ℝ) : 4 * y = 5 * x + 20 → (y - 5) / (x - (-5)) = 5/4 := by
  sorry

end slope_of_line_l1269_126949


namespace all_diagonal_triangles_count_l1269_126987

/-- Represents a convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : ℕ
  is_convex : sides = n

/-- Represents a diagonal division of a polygon -/
structure DiagonalDivision (p : ConvexPolygon n) where
  diagonals : ℕ
  triangles : ℕ
  non_intersecting : Bool
  vertex_diagonals : ℕ → ℕ
  valid_division : diagonals = n - 3 ∧ triangles = n - 2
  valid_vertex_diagonals : ∀ v, vertex_diagonals v = 3 ∨ vertex_diagonals v = 0

/-- Counts the number of triangles with all sides as diagonals -/
def count_all_diagonal_triangles (p : ConvexPolygon 102) (d : DiagonalDivision p) : ℕ :=
  sorry

theorem all_diagonal_triangles_count 
  (p : ConvexPolygon 102) 
  (d : DiagonalDivision p) : 
  count_all_diagonal_triangles p d = 34 :=
sorry

end all_diagonal_triangles_count_l1269_126987


namespace difference_between_decimals_and_fractions_l1269_126919

theorem difference_between_decimals_and_fractions : (0.127 : ℝ) - (1/8 : ℝ) = 0.002 := by
  sorry

end difference_between_decimals_and_fractions_l1269_126919


namespace age_ratio_proof_l1269_126971

-- Define the present ages of Lewis and Brown
def lewis_age : ℚ := 2
def brown_age : ℚ := 4

-- Define the conditions
theorem age_ratio_proof :
  -- Condition 1: Present ages are in ratio 1:2
  lewis_age / brown_age = 1 / 2 →
  -- Condition 2: Combined present age is 6
  lewis_age + brown_age = 6 →
  -- Prove: Ratio of ages three years from now is 5:7
  (lewis_age + 3) / (brown_age + 3) = 5 / 7 := by
sorry


end age_ratio_proof_l1269_126971


namespace birthday_money_l1269_126918

def money_spent : ℕ := 3
def money_left : ℕ := 2

theorem birthday_money : 
  ∃ (total : ℕ), total = money_spent + money_left :=
sorry

end birthday_money_l1269_126918


namespace system_solution_range_l1269_126969

theorem system_solution_range (x y k : ℝ) : 
  (2 * x + y = k + 1) →
  (x + 2 * y = 2) →
  (x + y < 0) →
  (k < -3) := by
sorry

end system_solution_range_l1269_126969


namespace N_div_15_is_square_N_div_10_is_cube_N_div_6_is_fifth_power_N_is_smallest_num_divisors_N_div_30_is_8400_l1269_126991

/-- The smallest positive integer N satisfying the given conditions -/
def N : ℕ := 2^16 * 3^21 * 5^25

/-- N/15 is a perfect square -/
theorem N_div_15_is_square : ∃ k : ℕ, N / 15 = k^2 := by sorry

/-- N/10 is a perfect cube -/
theorem N_div_10_is_cube : ∃ k : ℕ, N / 10 = k^3 := by sorry

/-- N/6 is a perfect fifth power -/
theorem N_div_6_is_fifth_power : ∃ k : ℕ, N / 6 = k^5 := by sorry

/-- N is the smallest positive integer satisfying the conditions -/
theorem N_is_smallest : ∀ m : ℕ, m < N → 
  (¬∃ k : ℕ, m / 15 = k^2) ∨ 
  (¬∃ k : ℕ, m / 10 = k^3) ∨ 
  (¬∃ k : ℕ, m / 6 = k^5) := by sorry

/-- The number of positive divisors of N/30 -/
def num_divisors_N_div_30 : ℕ := (15 + 1) * (20 + 1) * (24 + 1)

/-- Theorem: The number of positive divisors of N/30 is 8400 -/
theorem num_divisors_N_div_30_is_8400 : num_divisors_N_div_30 = 8400 := by sorry

end N_div_15_is_square_N_div_10_is_cube_N_div_6_is_fifth_power_N_is_smallest_num_divisors_N_div_30_is_8400_l1269_126991


namespace probability_r25_to_r35_correct_l1269_126916

def bubble_pass (s : List ℝ) : List ℝ := sorry

def probability_r25_to_r35 (n : ℕ) : ℚ :=
  if n ≥ 50 then 1 / 1260 else 0

theorem probability_r25_to_r35_correct (s : List ℝ) (h : s.length = 50) 
  (h_distinct : s.Nodup) : 
  probability_r25_to_r35 s.length = 1 / 1260 := by sorry

end probability_r25_to_r35_correct_l1269_126916


namespace warehouse_boxes_l1269_126977

theorem warehouse_boxes : 
  ∀ (warehouse1 warehouse2 : ℕ),
  warehouse1 = 2 * warehouse2 →
  warehouse1 + warehouse2 = 600 →
  warehouse1 = 400 := by
sorry

end warehouse_boxes_l1269_126977


namespace dhoni_toys_l1269_126994

theorem dhoni_toys (x : ℕ) (avg_cost : ℚ) (new_toy_cost : ℚ) (new_avg_cost : ℚ) : 
  avg_cost = 10 →
  new_toy_cost = 16 →
  new_avg_cost = 11 →
  (x * avg_cost + new_toy_cost) / (x + 1) = new_avg_cost →
  x = 5 := by
sorry

end dhoni_toys_l1269_126994


namespace quadratic_expansion_constraint_l1269_126920

theorem quadratic_expansion_constraint (a b m : ℤ) :
  (∀ x, (x + a) * (x + b) = x^2 + m*x + 5) →
  (m = 6 ∨ m = -6) :=
by sorry

end quadratic_expansion_constraint_l1269_126920


namespace expression_evaluation_l1269_126931

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b^2 - 14*b + 45)
  (h2 : b = a^2 + 2*a + 5)
  (h3 : a = 3)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 9) / (c + 7) = 4923 / 2924 := by
  sorry

end expression_evaluation_l1269_126931


namespace functional_equation_properties_l1269_126962

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem functional_equation_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) := by
  sorry

end functional_equation_properties_l1269_126962


namespace max_fraction_value_l1269_126996

theorem max_fraction_value (a b c d : ℕ) 
  (ha : 0 < a) (hab : a < b) (hbc : b < c) (hcd : c < d) (hd : d < 10) :
  (∀ w x y z : ℕ, 0 < w → w < x → x < y → y < z → z < 10 → 
    (a - b : ℚ) / (c - d : ℚ) ≥ (w - x : ℚ) / (y - z : ℚ)) →
  (a - b : ℚ) / (c - d : ℚ) = -6 :=
sorry

end max_fraction_value_l1269_126996


namespace johns_bonus_last_year_l1269_126972

/-- Represents John's yearly financial information -/
structure YearlyFinance where
  salary : ℝ
  bonus_percentage : ℝ
  total_income : ℝ

/-- Calculates the bonus amount given a salary and bonus percentage -/
def calculate_bonus (salary : ℝ) (bonus_percentage : ℝ) : ℝ :=
  salary * bonus_percentage

theorem johns_bonus_last_year 
  (last_year : YearlyFinance)
  (this_year : YearlyFinance)
  (h1 : last_year.salary = 100000)
  (h2 : this_year.salary = 200000)
  (h3 : this_year.total_income = 220000)
  (h4 : last_year.bonus_percentage = this_year.bonus_percentage) :
  calculate_bonus last_year.salary last_year.bonus_percentage = 10000 := by
sorry

end johns_bonus_last_year_l1269_126972


namespace complement_A_intersect_B_l1269_126908

universe u

def U : Set ℕ := {x | x < 6}

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {2, 4, 5}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {4, 5} := by sorry

end complement_A_intersect_B_l1269_126908


namespace candy_distribution_l1269_126943

/-- Represents the number of candies eaten by each person -/
structure CandyCount where
  andrey : ℕ
  boris : ℕ
  denis : ℕ

/-- Calculates the total number of candies eaten by all three people -/
def total_candies (count : CandyCount) : ℕ :=
  count.andrey + count.boris + count.denis

/-- Represents the relative eating rates of the three people -/
structure EatingRates where
  andrey_to_boris : ℚ
  andrey_to_denis : ℚ

/-- Theorem stating the correct number of candies eaten by each person -/
theorem candy_distribution (rates : EatingRates) : 
  ∃ (count : CandyCount), 
    rates.andrey_to_boris = 4 / 3 ∧ 
    rates.andrey_to_denis = 6 / 7 ∧
    total_candies count = 70 ∧
    count.andrey = 24 ∧
    count.boris = 18 ∧
    count.denis = 28 := by
  sorry

end candy_distribution_l1269_126943


namespace composite_sum_l1269_126973

theorem composite_sum (a b c d e f : ℕ+) 
  (hS : ∃ (k₁ k₂ : ℕ), 
    (a + b + c + d + e + f) * k₁ = a * b * c + d * e * f ∧ 
    (a + b + c + d + e + f) * k₂ = a * b + b * c + c * a - d * e - e * f - f * d) :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ a + b + c + d + e + f = m * n := by
  sorry

end composite_sum_l1269_126973


namespace usual_time_to_school_l1269_126980

/-- Given a boy who walks 7/6 of his usual rate and reaches school 5 minutes early,
    prove that his usual time to reach the school is 35 minutes. -/
theorem usual_time_to_school (R : ℝ) (T : ℝ) : 
  R * T = (7/6 * R) * (T - 5) → T = 35 :=
by sorry

end usual_time_to_school_l1269_126980


namespace milk_replacement_theorem_l1269_126954

-- Define the percentage of milk replaced by water in each operation
def replacement_percentage : ℝ → Prop := λ x =>
  -- Define the function that calculates the remaining milk percentage after three operations
  let remaining_milk := (1 - x/100)^3
  -- The remaining milk percentage should be 51.2%
  remaining_milk = 0.512

-- Theorem statement
theorem milk_replacement_theorem : 
  ∃ x : ℝ, replacement_percentage x ∧ x = 20 :=
sorry

end milk_replacement_theorem_l1269_126954


namespace geometric_series_sum_l1269_126953

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h : r ≠ 1) :
  let series_sum := a * (1 - r^n) / (1 - r)
  (a = 1/4) → (r = -1/4) → (n = 6) → series_sum = 4095/5120 := by sorry

end geometric_series_sum_l1269_126953


namespace geometric_progression_proof_l1269_126944

theorem geometric_progression_proof (a b c d : ℤ) :
  a = 7 ∧ b = -14 ∧ c = 28 ∧ d = -56 →
  (∃ r : ℚ, b = a * r ∧ c = b * r ∧ d = c * r) ∧
  a + d = -49 ∧
  b + c = 14 := by
sorry

end geometric_progression_proof_l1269_126944


namespace total_people_zoo_and_amusement_park_l1269_126929

theorem total_people_zoo_and_amusement_park : 
  let cars_to_zoo : Float := 7.0
  let people_per_car_zoo : Float := 45.0
  let cars_to_amusement_park : Float := 5.0
  let people_per_car_amusement_park : Float := 56.0
  
  cars_to_zoo * people_per_car_zoo + cars_to_amusement_park * people_per_car_amusement_park = 595.0 := by
  sorry

end total_people_zoo_and_amusement_park_l1269_126929


namespace sum_of_digits_of_10_pow_93_minus_95_l1269_126905

/-- Represents the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of digits of 10^93 - 95 is 824 -/
theorem sum_of_digits_of_10_pow_93_minus_95 : 
  sum_of_digits (10^93 - 95) = 824 := by sorry

end sum_of_digits_of_10_pow_93_minus_95_l1269_126905


namespace distance_theorem_l1269_126928

/-- The configuration of three squares where the middle one is rotated and lowered -/
structure SquareConfiguration where
  /-- Side length of each square -/
  side_length : ℝ
  /-- Rotation angle of the middle square in radians -/
  rotation_angle : ℝ

/-- Calculate the distance from point B to the original line -/
def distance_to_line (config : SquareConfiguration) : ℝ :=
  sorry

/-- The theorem stating the distance from point B to the original line -/
theorem distance_theorem (config : SquareConfiguration) :
  config.side_length = 1 ∧ config.rotation_angle = π / 4 →
  distance_to_line config = Real.sqrt 2 + 1 / 2 :=
by sorry

end distance_theorem_l1269_126928


namespace sams_german_shepherds_l1269_126978

theorem sams_german_shepherds (sam_french_bulldogs peter_total : ℕ) 
  (h1 : sam_french_bulldogs = 4)
  (h2 : peter_total = 17)
  (h3 : ∃ (sam_german_shepherds : ℕ), 
    3 * sam_german_shepherds + 2 * sam_french_bulldogs = peter_total) :
  ∃ (sam_german_shepherds : ℕ), sam_german_shepherds = 3 :=
by sorry

end sams_german_shepherds_l1269_126978


namespace weights_standard_deviation_l1269_126926

def weights (a b : ℝ) : List ℝ := [125, a, 121, b, 127]

def median (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry
def standardDeviation (l : List ℝ) : ℝ := sorry

theorem weights_standard_deviation (a b : ℝ) :
  median (weights a b) = 124 →
  mean (weights a b) = 124 →
  standardDeviation (weights a b) = 2 := by sorry

end weights_standard_deviation_l1269_126926


namespace best_fit_highest_abs_r_model1_best_fit_l1269_126964

/-- Represents a linear regression model with its correlation coefficient -/
structure RegressionModel where
  r : ℝ

/-- Determines if a model is the best fit among a list of models -/
def isBestFit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, |model.r| ≥ |m.r|

theorem best_fit_highest_abs_r (models : List RegressionModel) (model : RegressionModel) 
    (h : model ∈ models) :
    isBestFit model models ↔ ∀ m ∈ models, |model.r| ≥ |m.r| := by
  sorry

/-- The four models from the problem -/
def model1 : RegressionModel := ⟨0.98⟩
def model2 : RegressionModel := ⟨0.80⟩
def model3 : RegressionModel := ⟨0.50⟩
def model4 : RegressionModel := ⟨0.25⟩

def allModels : List RegressionModel := [model1, model2, model3, model4]

theorem model1_best_fit : isBestFit model1 allModels := by
  sorry

end best_fit_highest_abs_r_model1_best_fit_l1269_126964


namespace solve_equation_l1269_126932

theorem solve_equation : 
  ∃ x : ℚ, (27 / 4) * x - 18 = 3 * x + 27 ∧ x = 12 := by
  sorry

end solve_equation_l1269_126932


namespace green_peppers_weight_l1269_126910

/-- The weight of green peppers bought by Hannah's Vegetarian Restaurant -/
def green_peppers : ℝ := 0.3333333333333333

/-- The weight of red peppers bought by Hannah's Vegetarian Restaurant -/
def red_peppers : ℝ := 0.3333333333333333

/-- The total weight of peppers bought by Hannah's Vegetarian Restaurant -/
def total_peppers : ℝ := 0.6666666666666666

/-- Theorem stating that the weight of green peppers is the difference between
    the total weight of peppers and the weight of red peppers -/
theorem green_peppers_weight :
  green_peppers = total_peppers - red_peppers := by
  sorry

end green_peppers_weight_l1269_126910


namespace cricketer_average_increase_l1269_126906

theorem cricketer_average_increase (total_innings : ℕ) (last_inning_score : ℕ) (final_average : ℚ) : 
  total_innings = 19 → 
  last_inning_score = 98 → 
  final_average = 26 → 
  (final_average - (total_innings * final_average - last_inning_score) / (total_innings - 1)) = 4 := by
sorry

end cricketer_average_increase_l1269_126906


namespace inequality_solution_l1269_126985

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x > 1}
  else if a < 0 then {x | x > 1 ∨ x < 1/a}
  else if 0 < a ∧ a < 1 then {x | 1 < x ∧ x < 1/a}
  else if a > 1 then {x | 1/a < x ∧ x < 1}
  else ∅

theorem inequality_solution (a : ℝ) (x : ℝ) :
  a * x^2 - (a + 1) * x + 1 < 0 ↔ x ∈ solution_set a :=
sorry

end inequality_solution_l1269_126985


namespace function_inequality_solution_set_l1269_126922

/-- Given a function f(x) = (ax-1)/(x+b) where the solution set of f(x) > 0 is (-1, 3),
    prove that the solution set of f(-2x) < 0 is (-∞, -3/2) ∪ (1/2, +∞) -/
theorem function_inequality_solution_set 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x = (a * x - 1) / (x + b))
  (h₂ : Set.Ioo (-1 : ℝ) 3 = {x | f x > 0}) :
  {x : ℝ | f (-2 * x) < 0} = Set.Iic (-3/2) ∪ Set.Ioi (1/2) := by
  sorry

end function_inequality_solution_set_l1269_126922


namespace expression_simplification_l1269_126965

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 3 + 1) 
  (hb : b = Real.sqrt 3 - 1) : 
  ((a^2 / (a - b) - (2*a*b - b^2) / (a - b)) / ((a - b) / (a * b))) = 2 := by
  sorry

end expression_simplification_l1269_126965


namespace planes_parallel_if_perpendicular_to_same_line_l1269_126941

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is perpendicular to two planes, then those planes are parallel -/
theorem planes_parallel_if_perpendicular_to_same_line (a : Line3D) (α β : Plane3D) :
  perpendicular a α → perpendicular a β → parallel α β :=
sorry

end planes_parallel_if_perpendicular_to_same_line_l1269_126941


namespace product_of_integers_l1269_126976

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 22)
  (diff_squares_eq : x^2 - y^2 = 44) :
  x * y = 120 := by
  sorry

end product_of_integers_l1269_126976


namespace sin_cos_sum_equality_l1269_126925

theorem sin_cos_sum_equality : 
  Real.sin (20 * π / 180) * Real.cos (40 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (140 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_sum_equality_l1269_126925


namespace leila_sweater_cost_l1269_126951

/-- Represents Leila's spending on a sweater and jewelry --/
structure LeilasSpending where
  total : ℝ
  sweater : ℝ
  jewelry : ℝ

/-- The conditions of Leila's spending --/
def spending_conditions (s : LeilasSpending) : Prop :=
  s.sweater = (1/4) * s.total ∧
  s.jewelry = (3/4) * s.total - 20 ∧
  s.jewelry = s.sweater + 60

/-- Theorem stating that under the given conditions, Leila spent $40 on the sweater --/
theorem leila_sweater_cost (s : LeilasSpending) 
  (h : spending_conditions s) : s.sweater = 40 := by
  sorry

#check leila_sweater_cost

end leila_sweater_cost_l1269_126951


namespace no_integer_roots_l1269_126904

theorem no_integer_roots (a b c : ℤ) (h_a : a ≠ 0) 
  (h_f0 : Odd (c)) 
  (h_f1 : Odd (a + b + c)) : 
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
sorry

end no_integer_roots_l1269_126904


namespace parabola_intersection_l1269_126934

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 2
  let g (x : ℝ) := 9 * x^2 + 6 * x + 2
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = 0 ∧ y = 2) ∨ (x = -5/3 ∧ y = 17) :=
by sorry

end parabola_intersection_l1269_126934


namespace triangle_properties_l1269_126913

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (2 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C ∧
  t.a = Real.sqrt 13 ∧
  t.b + t.c = 5

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.A = π / 3 ∧ 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3 := by
  sorry

end triangle_properties_l1269_126913


namespace simple_interest_example_l1269_126917

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Proves that the simple interest on $10000 at 9% per annum for 12 months is $900 -/
theorem simple_interest_example : simple_interest 10000 0.09 1 = 900 := by
  sorry

end simple_interest_example_l1269_126917


namespace x_convergence_l1269_126935

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n + 5) / (x n + 7)

theorem x_convergence :
  ∃ m : ℕ, 130 ≤ m ∧ m ≤ 240 ∧ x m ≤ 5 + 1 / 2^21 ∧
  ∀ k : ℕ, k < m → x k > 5 + 1 / 2^21 :=
sorry

end x_convergence_l1269_126935


namespace cafeteria_apple_count_l1269_126924

/-- Calculates the final number of apples in the cafeteria after a series of operations -/
def final_apple_count (initial : ℕ) (monday_used monday_bought tuesday_used tuesday_bought wednesday_used : ℕ) : ℕ :=
  initial - monday_used + monday_bought - tuesday_used + tuesday_bought - wednesday_used

/-- Theorem stating that given the initial number of apples and daily changes, the final number of apples is 46 -/
theorem cafeteria_apple_count : 
  final_apple_count 17 2 23 4 15 3 = 46 := by
  sorry

end cafeteria_apple_count_l1269_126924


namespace sequence_sum_theorem_l1269_126981

/-- The sum of the sequence 1+2-3-4+5+6-7-8+...+2017+2018-2019-2020 -/
def sequenceSum : ℤ := -2020

/-- The last term in the sequence -/
def lastTerm : ℕ := 2020

/-- The number of complete groups of four in the sequence -/
def groupCount : ℕ := lastTerm / 4

/-- The sum of each group of four terms in the sequence -/
def groupSum : ℤ := -4

theorem sequence_sum_theorem :
  sequenceSum = groupCount * groupSum :=
by sorry

end sequence_sum_theorem_l1269_126981


namespace sum_of_squares_theorem_l1269_126986

theorem sum_of_squares_theorem (x y z t : ℤ) (h : x + y = z + t) :
  x^2 + y^2 + z^2 + t^2 = (x + y)^2 + (x - z)^2 + (x - t)^2 := by
  sorry

end sum_of_squares_theorem_l1269_126986


namespace expression_value_l1269_126946

theorem expression_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 3 * Real.sqrt (x * y)) :
  |(x - y) / (x + y) + (x^2 - y^2) / (x^2 + y^2) + (x^3 - y^3) / (x^3 + y^3)| = 3 := by
  sorry

end expression_value_l1269_126946


namespace tangent_slope_at_one_l1269_126921

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 / x

theorem tangent_slope_at_one :
  HasDerivAt f (Real.exp 1 + 1) 1 :=
sorry

end tangent_slope_at_one_l1269_126921


namespace log_equation_solution_l1269_126933

theorem log_equation_solution :
  ∃! x : ℝ, x > 0 ∧ x + 2 > 0 ∧ 2*x + 3 > 0 ∧
  Real.log x + Real.log (x + 2) = Real.log (2*x + 3) ∧
  x = Real.sqrt 3 := by
  sorry

end log_equation_solution_l1269_126933


namespace six_by_six_grid_half_shaded_l1269_126902

/-- Represents a square grid -/
structure SquareGrid :=
  (size : ℕ)
  (shaded : ℕ)

/-- Calculates the percentage of shaded area in a square grid -/
def shaded_percentage (grid : SquareGrid) : ℚ :=
  (grid.shaded : ℚ) / (grid.size * grid.size : ℚ) * 100

/-- Theorem: A 6x6 grid with 18 shaded squares is 50% shaded -/
theorem six_by_six_grid_half_shaded :
  let grid : SquareGrid := ⟨6, 18⟩
  shaded_percentage grid = 50 := by sorry

end six_by_six_grid_half_shaded_l1269_126902


namespace annual_pension_calculation_l1269_126955

/-- Represents the annual pension calculation for a retiring employee. -/
theorem annual_pension_calculation
  (c d r s : ℝ)
  (h_cd : d ≠ c)
  (h_positive : c > 0 ∧ d > 0 ∧ r > 0 ∧ s > 0)
  (h_prop : ∃ (k x : ℝ), k > 0 ∧ x > 0 ∧
    k * (x + c)^(3/2) = k * x^(3/2) + r ∧
    k * (x + d)^(3/2) = k * x^(3/2) + s) :
  ∃ (pension : ℝ), pension = (4 * r^2) / (9 * c^2) :=
sorry

end annual_pension_calculation_l1269_126955


namespace max_unique_subsets_l1269_126952

theorem max_unique_subsets (n : ℕ) (h : n = 7) : 
  (2 ^ n) - 2 = 128 := by
  sorry

end max_unique_subsets_l1269_126952


namespace debt_payment_average_l1269_126956

theorem debt_payment_average (total_payments : ℕ) (first_payment_count : ℕ) (first_payment_amount : ℚ) (additional_amount : ℚ) : 
  total_payments = 65 ∧ 
  first_payment_count = 20 ∧ 
  first_payment_amount = 410 ∧ 
  additional_amount = 65 → 
  (first_payment_count * first_payment_amount + 
   (total_payments - first_payment_count) * (first_payment_amount + additional_amount)) / total_payments = 455 := by
sorry

end debt_payment_average_l1269_126956


namespace greatest_prime_factor_of_expression_l1269_126945

theorem greatest_prime_factor_of_expression (p : Nat) :
  (p.Prime ∧ p ∣ (2^8 + 5^4 + 10^3) ∧ ∀ q : Nat, q.Prime → q ∣ (2^8 + 5^4 + 10^3) → q ≤ p) ↔ p = 19 := by
  sorry

end greatest_prime_factor_of_expression_l1269_126945


namespace taxi_speed_l1269_126989

/-- Given a taxi and a bus with specific conditions, proves that the taxi's speed is 60 mph --/
theorem taxi_speed (taxi_speed bus_speed : ℝ) : 
  (taxi_speed > 0) →  -- Ensure positive speed
  (bus_speed > 0) →   -- Ensure positive speed
  (bus_speed = taxi_speed - 30) →  -- Bus is 30 mph slower
  (3 * taxi_speed = 6 * bus_speed) →  -- Taxi covers in 3 hours what bus covers in 6
  (taxi_speed = 60) :=
by
  sorry

#check taxi_speed

end taxi_speed_l1269_126989


namespace computer_rental_rates_l1269_126958

/-- Represents the hourly rental rates and job completion times for three computers -/
structure ComputerRental where
  rateA : ℝ  -- Hourly rate for Computer A
  rateB : ℝ  -- Hourly rate for Computer B
  rateC : ℝ  -- Hourly rate for Computer C
  timeA : ℝ  -- Time for Computer A to complete the job

/-- Conditions for the computer rental problem -/
def rental_conditions (r : ComputerRental) : Prop :=
  r.rateA = 1.4 * r.rateB ∧
  r.rateC = 0.75 * r.rateB ∧
  r.rateA * r.timeA = 550 ∧
  r.rateB * (r.timeA + 20) = 550 ∧
  r.rateC * (r.timeA + 10) = 550

/-- Theorem stating the approximate hourly rates for the computers -/
theorem computer_rental_rates :
  ∃ r : ComputerRental, rental_conditions r ∧
    (abs (r.rateA - 11) < 0.01) ∧
    (abs (r.rateB - 7.86) < 0.01) ∧
    (abs (r.rateC - 5.90) < 0.01) :=
  by sorry

end computer_rental_rates_l1269_126958


namespace sector_central_angle_l1269_126992

theorem sector_central_angle (area : Real) (perimeter : Real) (r : Real) (θ : Real) :
  area = 1 ∧ perimeter = 4 ∧ area = (1/2) * r^2 * θ ∧ perimeter = 2*r + r*θ → θ = 2 := by
  sorry

end sector_central_angle_l1269_126992


namespace sum_geq_sqrt_three_l1269_126975

theorem sum_geq_sqrt_three (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_prod : a * b + b * c + c * a = 1) : 
  a + b + c ≥ Real.sqrt 3 := by
sorry

end sum_geq_sqrt_three_l1269_126975


namespace total_jumps_equals_1085_l1269_126970

def ronald_jumps : ℕ := 157

def rupert_jumps : ℕ := 3 * ronald_jumps + 23

def rebecca_initial_jumps : ℕ := 47
def rebecca_common_difference : ℕ := 5
def rebecca_sequences : ℕ := 7

def rebecca_last_jumps : ℕ := rebecca_initial_jumps + (rebecca_sequences - 1) * rebecca_common_difference

def rebecca_total_jumps : ℕ := rebecca_sequences * (rebecca_initial_jumps + rebecca_last_jumps) / 2

def total_jumps : ℕ := ronald_jumps + rupert_jumps + rebecca_total_jumps

theorem total_jumps_equals_1085 : total_jumps = 1085 := by
  sorry

end total_jumps_equals_1085_l1269_126970


namespace f_at_five_halves_l1269_126900

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_symmetry (x : ℝ) : f ((-2) - x) = f ((-2) + x)
axiom f_period (x : ℝ) : f (x + 2) = f x
axiom f_definition (x : ℝ) (h : x ∈ Set.Icc (-3) (-2)) : f x = (x + 2)^2

-- State the theorem to be proved
theorem f_at_five_halves : f (5/2) = 1/4 := by sorry

end f_at_five_halves_l1269_126900


namespace H_range_l1269_126963

/-- The function H defined for all real x -/
def H (x : ℝ) : ℝ := 2 * |2*x + 2| - 3 * |2*x - 2|

/-- The theorem stating that the range of H is [8, ∞) -/
theorem H_range : Set.range H = Set.Ici 8 := by sorry

end H_range_l1269_126963


namespace negation_of_existence_logarithm_l1269_126995

theorem negation_of_existence_logarithm (x : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ = x₀ - 1) ↔
  (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) :=
by sorry

end negation_of_existence_logarithm_l1269_126995

import Mathlib

namespace product_equality_l2767_276719

theorem product_equality : 1500 * 2987 * 0.2987 * 15 = 2989502.987 := by
  sorry

end product_equality_l2767_276719


namespace tire_cost_l2767_276743

theorem tire_cost (window_cost tire_count total_cost : ℕ) 
  (h1 : window_cost = 700)
  (h2 : tire_count = 3)
  (h3 : total_cost = 1450)
  (h4 : tire_count * (total_cost - window_cost) / tire_count = 250) :
  ∃ (single_tire_cost : ℕ), 
    single_tire_cost * tire_count + window_cost = total_cost ∧ 
    single_tire_cost = 250 := by
  sorry

end tire_cost_l2767_276743


namespace sharp_four_times_25_l2767_276739

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.6 * N + 2

-- State the theorem
theorem sharp_four_times_25 : sharp (sharp (sharp (sharp 25))) = 7.592 := by
  sorry

end sharp_four_times_25_l2767_276739


namespace noah_yearly_call_cost_l2767_276724

/-- The cost of Noah's calls to his Grammy for a year -/
def yearly_call_cost (calls_per_week : ℕ) (minutes_per_call : ℕ) (cost_per_minute : ℚ) (weeks_per_year : ℕ) : ℚ :=
  (calls_per_week * minutes_per_call * cost_per_minute * weeks_per_year : ℚ)

/-- Theorem stating that Noah's yearly call cost to his Grammy is $78 -/
theorem noah_yearly_call_cost :
  yearly_call_cost 1 30 (5/100) 52 = 78 := by
  sorry

end noah_yearly_call_cost_l2767_276724


namespace f_min_bound_l2767_276715

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_min_bound (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, f a x > 2 * Real.log a + 3/2 := by sorry

end f_min_bound_l2767_276715


namespace meaningful_expression_range_l2767_276783

theorem meaningful_expression_range (x : ℝ) : 
  (∃ (y : ℝ), y = (Real.sqrt (x + 4)) / (x - 2)) ↔ (x ≥ -4 ∧ x ≠ 2) := by
sorry

end meaningful_expression_range_l2767_276783


namespace ron_height_is_13_l2767_276769

/-- The height of Ron in feet -/
def ron_height : ℝ := 13

/-- The height of Dean in feet -/
def dean_height : ℝ := ron_height + 4

/-- The depth of the water in feet -/
def water_depth : ℝ := 255

theorem ron_height_is_13 :
  (water_depth = 15 * dean_height) →
  (dean_height = ron_height + 4) →
  (water_depth = 255) →
  ron_height = 13 := by
sorry

end ron_height_is_13_l2767_276769


namespace asterisk_replacement_l2767_276785

theorem asterisk_replacement : (54 / 18) * (54 / 162) = 1 := by
  sorry

end asterisk_replacement_l2767_276785


namespace intersection_point_k_value_l2767_276767

theorem intersection_point_k_value (k : ℝ) : 
  (∃ y : ℝ, -3 * (-9.6) + 2 * y = k ∧ 0.25 * (-9.6) + y = 16) → k = 65.6 := by
  sorry

end intersection_point_k_value_l2767_276767


namespace unique_three_digit_divisible_by_11_l2767_276797

theorem unique_three_digit_divisible_by_11 : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  n % 10 = 2 ∧          -- units digit is 2
  n / 100 = 7 ∧         -- hundreds digit is 7
  n % 11 = 0 ∧          -- divisible by 11
  n = 792 := by          -- the number is 792
sorry


end unique_three_digit_divisible_by_11_l2767_276797


namespace area_of_specific_quadrilateral_l2767_276795

/-- Represents a convex quadrilateral ABCD with given side lengths and angle properties -/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  angle_CBA_is_right : Bool
  tan_angle_ACD : ℝ

/-- Calculates the area of the quadrilateral ABCD -/
def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific quadrilateral is 122/3 -/
theorem area_of_specific_quadrilateral :
  let q : Quadrilateral := {
    AB := 6,
    BC := 8,
    CD := 5,
    DA := 10,
    angle_CBA_is_right := true,
    tan_angle_ACD := 4/3
  }
  area q = 122/3 := by sorry

end area_of_specific_quadrilateral_l2767_276795


namespace system_solution_proof_l2767_276773

theorem system_solution_proof (x y : ℝ) : 
  (2 * x + y = 2 ∧ x - y = 1) → (x = 1 ∧ y = 0) :=
by
  sorry

end system_solution_proof_l2767_276773


namespace divisibility_by_eleven_l2767_276762

theorem divisibility_by_eleven (m : Nat) : 
  m < 10 → -- m is a single digit
  (742 * 100000 + m * 10000 + 834) % 11 = 0 → -- 742m834 is divisible by 11
  m = 3 := by
sorry

end divisibility_by_eleven_l2767_276762


namespace coffee_shop_usage_l2767_276755

/-- The number of bags of coffee beans used every morning -/
def morning_bags : ℕ := 3

/-- The number of bags of coffee beans used every afternoon -/
def afternoon_bags : ℕ := 3 * morning_bags

/-- The number of bags of coffee beans used every evening -/
def evening_bags : ℕ := 2 * morning_bags

/-- The total number of bags used in a week -/
def weekly_bags : ℕ := 126

theorem coffee_shop_usage :
  7 * (morning_bags + afternoon_bags + evening_bags) = weekly_bags :=
sorry

end coffee_shop_usage_l2767_276755


namespace company_female_employees_l2767_276725

theorem company_female_employees :
  ∀ (total_employees : ℕ) 
    (advanced_degrees : ℕ) 
    (males_college_only : ℕ) 
    (females_advanced : ℕ),
  total_employees = 180 →
  advanced_degrees = 90 →
  males_college_only = 35 →
  females_advanced = 55 →
  ∃ (female_employees : ℕ),
    female_employees = 110 ∧
    female_employees = (total_employees - advanced_degrees - males_college_only) + females_advanced :=
by
  sorry

end company_female_employees_l2767_276725


namespace linear_function_proof_l2767_276711

theorem linear_function_proof (k b : ℝ) :
  (1 * k + b = -2) →
  (-1 * k + b = -4) →
  (3 * k + b = 0) :=
by sorry

end linear_function_proof_l2767_276711


namespace system_solution_l2767_276748

theorem system_solution (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ 
  (3 * y - Real.sqrt (y / x) - 6 * Real.sqrt (x * y) + 2 = 0) ∧
  (x^2 + 81 * x^2 * y^4 = 2 * y^2) →
  ((x = 1/3 ∧ y = 1/3) ∨ (x = Real.rpow 31 (1/4) / 12 ∧ y = Real.rpow 31 (1/4) / 3)) :=
by sorry

end system_solution_l2767_276748


namespace expression_evaluation_l2767_276770

theorem expression_evaluation (x : ℚ) (h : x = 1/2) : 
  (1 + x) * (1 - x) + x * (x + 2) = 2 := by
  sorry

end expression_evaluation_l2767_276770


namespace intersection_M_N_l2767_276774

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 2 < x ∧ x < 3} := by sorry

end intersection_M_N_l2767_276774


namespace sum_of_three_numbers_l2767_276792

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 179)
  (h2 : a*b + b*c + a*c = 131) :
  a + b + c = 21 := by
  sorry

end sum_of_three_numbers_l2767_276792


namespace robert_second_trade_l2767_276763

def cards_traded_problem (padma_initial : ℕ) (robert_initial : ℕ) (padma_traded_first : ℕ) 
  (robert_traded_first : ℕ) (padma_traded_second : ℕ) (total_traded : ℕ) : Prop :=
  padma_initial = 75 ∧
  robert_initial = 88 ∧
  padma_traded_first = 2 ∧
  robert_traded_first = 10 ∧
  padma_traded_second = 15 ∧
  total_traded = 35 ∧
  total_traded = padma_traded_first + robert_traded_first + padma_traded_second + 
    (total_traded - padma_traded_first - robert_traded_first - padma_traded_second)

theorem robert_second_trade (padma_initial robert_initial padma_traded_first robert_traded_first 
  padma_traded_second total_traded : ℕ) :
  cards_traded_problem padma_initial robert_initial padma_traded_first robert_traded_first 
    padma_traded_second total_traded →
  total_traded - padma_traded_first - robert_traded_first - padma_traded_second = 25 :=
by sorry

end robert_second_trade_l2767_276763


namespace z1_in_first_quadrant_l2767_276722

def z1 (a : ℝ) : ℂ := a + Complex.I
def z2 : ℂ := 1 - Complex.I

theorem z1_in_first_quadrant (a : ℝ) :
  (z1 a / z2).im ≠ 0 ∧ (z1 a / z2).re = 0 →
  0 < (z1 a).re ∧ 0 < (z1 a).im :=
by sorry

end z1_in_first_quadrant_l2767_276722


namespace negation_of_implication_l2767_276703

theorem negation_of_implication (x : ℝ) :
  ¬(x > 2 → x^2 - 3*x + 2 > 0) ↔ (x ≤ 2 → x^2 - 3*x + 2 ≤ 0) := by sorry

end negation_of_implication_l2767_276703


namespace raffle_ticket_average_l2767_276758

/-- Represents a charitable association with male and female members selling raffle tickets -/
structure CharitableAssociation where
  male_members : ℕ
  female_members : ℕ
  male_avg_tickets : ℕ
  female_avg_tickets : ℕ

/-- The overall average number of raffle tickets sold per member -/
def overall_average (ca : CharitableAssociation) : ℚ :=
  (ca.male_members * ca.male_avg_tickets + ca.female_members * ca.female_avg_tickets : ℚ) /
  (ca.male_members + ca.female_members : ℚ)

/-- Theorem stating the overall average of raffle tickets sold per member -/
theorem raffle_ticket_average (ca : CharitableAssociation) 
  (h1 : ca.female_members = 2 * ca.male_members)
  (h2 : ca.female_avg_tickets = 70)
  (h3 : ca.male_avg_tickets = 58) :
  overall_average ca = 66 := by
  sorry

end raffle_ticket_average_l2767_276758


namespace number_puzzle_l2767_276742

theorem number_puzzle : ∃ x : ℝ, (x / 5) + 10 = 21 :=
by
  sorry

end number_puzzle_l2767_276742


namespace complement_of_N_in_U_l2767_276757

-- Define the universal set U
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set N
def N : Set ℝ := {x | 0 ≤ x ∧ x < 2}

-- Theorem statement
theorem complement_of_N_in_U :
  (U \ N) = {x | (-3 ≤ x ∧ x < 0) ∨ (2 ≤ x ∧ x ≤ 3)} := by
  sorry

end complement_of_N_in_U_l2767_276757


namespace games_in_specific_league_l2767_276782

/-- The number of games played in a season for a league with a given number of teams and repetitions -/
def games_in_season (num_teams : ℕ) (repetitions : ℕ) : ℕ :=
  (num_teams * (num_teams - 1) / 2) * repetitions

/-- Theorem stating the number of games in a season for a specific league setup -/
theorem games_in_specific_league : games_in_season 14 5 = 455 := by
  sorry

end games_in_specific_league_l2767_276782


namespace quadratic_equation_solutions_quadratic_equation_with_factoring_solutions_l2767_276775

theorem quadratic_equation_solutions :
  (∃ x : ℝ, x^2 - 5*x + 6 = 0) ↔ (∃ x : ℝ, x = 2 ∨ x = 3) :=
by sorry

theorem quadratic_equation_with_factoring_solutions :
  (∃ x : ℝ, (x - 2)^2 = 2*(x - 3)*(x - 2)) ↔ (∃ x : ℝ, x = 2 ∨ x = 4) :=
by sorry

end quadratic_equation_solutions_quadratic_equation_with_factoring_solutions_l2767_276775


namespace lowest_sum_due_bank_a_l2767_276723

structure Bank where
  name : String
  bankers_discount : ℕ
  true_discount : ℕ

def sum_due (b : Bank) : ℕ := b.bankers_discount - (b.bankers_discount - b.true_discount)

def bank_a : Bank := { name := "A", bankers_discount := 42, true_discount := 36 }
def bank_b : Bank := { name := "B", bankers_discount := 48, true_discount := 41 }
def bank_c : Bank := { name := "C", bankers_discount := 54, true_discount := 47 }

theorem lowest_sum_due_bank_a :
  (sum_due bank_a < sum_due bank_b) ∧
  (sum_due bank_a < sum_due bank_c) ∧
  (sum_due bank_a = 36) :=
by sorry

end lowest_sum_due_bank_a_l2767_276723


namespace symmetric_f_max_value_l2767_276799

/-- A function f(x) that is symmetric about x = -2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

/-- The symmetry condition for f(x) about x = -2 -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x, f a b (-2 - x) = f a b (-2 + x)

/-- The theorem stating that if f(x) is symmetric about x = -2, its maximum value is 16 -/
theorem symmetric_f_max_value (a b : ℝ) (h : is_symmetric a b) :
  ∃ x, f a b x = 16 ∧ ∀ y, f a b y ≤ 16 :=
sorry

end symmetric_f_max_value_l2767_276799


namespace parallelogram_count_parallelogram_count_proof_l2767_276749

/-- Given a triangle ABC with each side divided into n equal parts and parallel lines drawn through
    the division points, the number of parallelograms formed is 3 * (n choose 2). -/
theorem parallelogram_count (n : ℕ) : ℕ :=
  3 * (n.choose 2)

#check parallelogram_count

/-- Proof of the parallelogram count theorem -/
theorem parallelogram_count_proof (n : ℕ) :
  parallelogram_count n = 3 * (n.choose 2) := by
  sorry

end parallelogram_count_parallelogram_count_proof_l2767_276749


namespace jeans_price_increase_l2767_276741

/-- Given a manufacturing cost C, calculate the percentage increase from the retailer's price to the customer's price -/
theorem jeans_price_increase (C : ℝ) : 
  let retailer_price := C * 1.4
  let customer_price := C * 1.82
  (customer_price - retailer_price) / retailer_price * 100 = 30 := by
sorry

end jeans_price_increase_l2767_276741


namespace alices_spending_l2767_276751

theorem alices_spending (B : ℝ) : 
  ∃ (book magazine : ℝ),
    book = 0.25 * (B - magazine) ∧
    magazine = 0.1 * (B - book) ∧
    book + magazine = (4/13) * B :=
by sorry

end alices_spending_l2767_276751


namespace parabola_intersection_and_perpendicularity_perpendicular_intersection_range_l2767_276784

-- Define the parabola C: y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l: y = k(x+1) passing through M(-1, 0)
def line (k x y : ℝ) : Prop := y = k*(x+1)

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point M on the x-axis where the directrix intersects
def M : ℝ × ℝ := (-1, 0)

-- Define the relationship between AM and AF
def AM_AF_relation (A : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  (x₁ + 1)^2 + (y₁)^2 = (25/16) * ((x₁ - 1)^2 + y₁^2)

-- Define the perpendicularity condition for QA and QB
def perpendicular_condition (Q A B : ℝ × ℝ) : Prop :=
  let (x₀, y₀) := Q
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  (y₀ - y₁) * (y₀ - y₂) = -(x₀ - x₁) * (x₀ - x₂)

theorem parabola_intersection_and_perpendicularity (k : ℝ) :
  (∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line k A.1 A.2 ∧ 
    line k B.1 B.2 ∧ 
    AM_AF_relation A) →
  k = 3/4 ∨ k = -3/4 :=
sorry

theorem perpendicular_intersection_range (k : ℝ) :
  (∃ Q A B : ℝ × ℝ,
    parabola Q.1 Q.2 ∧
    parabola A.1 A.2 ∧
    parabola B.1 B.2 ∧
    line k A.1 A.2 ∧
    line k B.1 B.2 ∧
    perpendicular_condition Q A B) ↔
  (k > 0 ∧ k < Real.sqrt 5 / 5) ∨ (k < 0 ∧ k > -Real.sqrt 5 / 5) :=
sorry

end parabola_intersection_and_perpendicularity_perpendicular_intersection_range_l2767_276784


namespace digit_pair_sum_l2767_276708

/-- Two different digits that form two-digit numbers whose sum is 202 -/
structure DigitPair where
  a : ℕ
  b : ℕ
  a_is_digit : a ≥ 1 ∧ a ≤ 9
  b_is_digit : b ≥ 0 ∧ b ≤ 9
  a_ne_b : a ≠ b
  sum_eq_202 : 10 * a + b + 10 * b + a = 202

/-- The sum of the digits in a DigitPair is 12 -/
theorem digit_pair_sum (p : DigitPair) : p.a + p.b = 12 := by
  sorry

end digit_pair_sum_l2767_276708


namespace trigonometric_identity_l2767_276786

theorem trigonometric_identity : 
  2 * Real.sin (390 * π / 180) - Real.tan (-45 * π / 180) + 5 * Real.cos (360 * π / 180) = 7 := by
  sorry

end trigonometric_identity_l2767_276786


namespace ten_people_no_adjacent_standing_probability_l2767_276777

/-- Represents the number of valid arrangements for n people where no two adjacent people stand --/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The probability of no two adjacent people standing in a circular arrangement of n people --/
def noAdjacentStandingProbability (n : ℕ) : ℚ :=
  validArrangements n / (2 ^ n : ℚ)

theorem ten_people_no_adjacent_standing_probability :
  noAdjacentStandingProbability 10 = 123 / 1024 := by
  sorry


end ten_people_no_adjacent_standing_probability_l2767_276777


namespace stream_speed_l2767_276738

/-- The speed of a stream given downstream and upstream speeds -/
theorem stream_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 14)
  (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 3 := by
  sorry

end stream_speed_l2767_276738


namespace sons_age_l2767_276759

/-- Proves that given the conditions, the son's present age is 33 years. -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 33 := by
sorry

end sons_age_l2767_276759


namespace unit_digit_of_expression_l2767_276789

-- Define the expression
def expression : ℕ := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) - 1

-- Theorem statement
theorem unit_digit_of_expression : expression % 10 = 4 := by
  sorry

end unit_digit_of_expression_l2767_276789


namespace faye_age_l2767_276729

/-- Represents the ages of Diana, Eduardo, Chad, and Faye -/
structure Ages where
  diana : ℕ
  eduardo : ℕ
  chad : ℕ
  faye : ℕ

/-- Defines the age relationships between Diana, Eduardo, Chad, and Faye -/
def valid_ages (ages : Ages) : Prop :=
  ages.diana + 3 = ages.eduardo ∧
  ages.eduardo = ages.chad + 4 ∧
  ages.faye = ages.chad + 3 ∧
  ages.diana = 14

/-- Theorem stating that given the age relationships and Diana's age, Faye's age is 18 -/
theorem faye_age (ages : Ages) (h : valid_ages ages) : ages.faye = 18 := by
  sorry

end faye_age_l2767_276729


namespace max_distance_line_l2767_276712

/-- The point through which the line passes -/
def point : ℝ × ℝ := (1, 2)

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := x + 2*y - 5 = 0

/-- Theorem stating that the given line equation represents the line with maximum distance from the origin passing through the specified point -/
theorem max_distance_line :
  line_equation point.1 point.2 ∧
  ∀ (a b c : ℝ), (a*point.1 + b*point.2 + c = 0) →
    (a^2 + b^2 ≤ 1^2 + 2^2) :=
sorry

end max_distance_line_l2767_276712


namespace total_cost_is_1975_l2767_276745

def first_laptop_cost : ℝ := 500
def second_laptop_multiplier : ℝ := 3
def discount_rate : ℝ := 0.15
def external_hard_drive_cost : ℝ := 80
def mouse_cost : ℝ := 20

def total_cost : ℝ :=
  let second_laptop_cost := first_laptop_cost * second_laptop_multiplier
  let discounted_second_laptop_cost := second_laptop_cost * (1 - discount_rate)
  let accessories_cost := external_hard_drive_cost + mouse_cost
  first_laptop_cost + discounted_second_laptop_cost + 2 * accessories_cost

theorem total_cost_is_1975 : total_cost = 1975 := by
  sorry

end total_cost_is_1975_l2767_276745


namespace initial_distance_is_point_eight_l2767_276734

/-- Two boats moving towards each other with given speeds and a known distance before collision -/
structure BoatProblem where
  speed1 : ℝ  -- Speed of boat 1 in miles/hr
  speed2 : ℝ  -- Speed of boat 2 in miles/hr
  distance_before_collision : ℝ  -- Distance between boats 1 minute before collision in miles
  time_before_collision : ℝ  -- Time before collision in hours

/-- The initial distance between the boats given the problem parameters -/
def initial_distance (p : BoatProblem) : ℝ :=
  p.distance_before_collision + (p.speed1 + p.speed2) * p.time_before_collision

/-- Theorem stating that the initial distance is 0.8 miles given the specific problem conditions -/
theorem initial_distance_is_point_eight :
  let p : BoatProblem := {
    speed1 := 4,
    speed2 := 20,
    distance_before_collision := 0.4,
    time_before_collision := 1 / 60
  }
  initial_distance p = 0.8 := by sorry

end initial_distance_is_point_eight_l2767_276734


namespace geometric_sequence_205th_term_l2767_276717

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

theorem geometric_sequence_205th_term :
  let a₁ : ℝ := 6
  let r : ℝ := -1
  geometric_sequence a₁ r 205 = 6 := by
sorry

end geometric_sequence_205th_term_l2767_276717


namespace evaluate_expression_l2767_276700

theorem evaluate_expression : 6 - 9 * (1 / 2 - 3^3) * 2 = 483 := by
  sorry

end evaluate_expression_l2767_276700


namespace sarahs_bowling_score_l2767_276778

theorem sarahs_bowling_score (sarah_score greg_score : ℕ) : 
  sarah_score = greg_score + 60 →
  (sarah_score + greg_score) / 2 = 110 →
  sarah_score + greg_score < 450 →
  sarah_score = 140 :=
by
  sorry

end sarahs_bowling_score_l2767_276778


namespace equation_solution_l2767_276721

theorem equation_solution : ∃ x : ℚ, (2*x - 1)/3 - (x - 2)/6 = 2 ∧ x = 4 := by
  sorry

end equation_solution_l2767_276721


namespace acid_concentration_theorem_l2767_276718

def acid_concentration_problem (acid1 acid2 acid3 : ℝ) 
  (conc1 conc2 : ℝ) : Prop :=
  let water1 := acid1 / conc1 - acid1
  let water2 := acid2 / conc2 - acid2
  let total_water := water1 + water2
  let conc3 := acid3 / (acid3 + total_water)
  acid1 = 10 ∧ 
  acid2 = 20 ∧ 
  acid3 = 30 ∧ 
  conc1 = 0.05 ∧ 
  conc2 = 70/300 ∧ 
  conc3 = 0.105

theorem acid_concentration_theorem : 
  acid_concentration_problem 10 20 30 0.05 (70/300) :=
sorry

end acid_concentration_theorem_l2767_276718


namespace solve_daily_wage_l2767_276730

def daily_wage_problem (a b c : ℕ) : Prop :=
  -- Define the ratio of daily wages
  a * 4 = b * 3 ∧ b * 5 = c * 4 ∧
  -- Define the total earnings
  6 * a + 9 * b + 4 * c = 1406 ∧
  -- The daily wage of c is 95
  c = 95

theorem solve_daily_wage : ∃ a b c : ℕ, daily_wage_problem a b c :=
sorry

end solve_daily_wage_l2767_276730


namespace fraction_value_l2767_276790

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end fraction_value_l2767_276790


namespace circle_radius_l2767_276740

/-- The radius of the circle with equation x^2 - 10x + y^2 - 4y + 24 = 0 is √5 -/
theorem circle_radius (x y : ℝ) : 
  (x^2 - 10*x + y^2 - 4*y + 24 = 0) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 5 := by
  sorry

end circle_radius_l2767_276740


namespace friend_age_order_l2767_276735

-- Define the set of friends
inductive Friend : Type
  | David : Friend
  | Emma : Friend
  | Fiona : Friend

-- Define the age ordering relation
def AgeOrder : Friend → Friend → Prop := sorry

-- Define the property of being the oldest
def IsOldest (f : Friend) : Prop := ∀ g : Friend, g ≠ f → AgeOrder f g

-- Define the property of being the youngest
def IsYoungest (f : Friend) : Prop := ∀ g : Friend, g ≠ f → AgeOrder g f

-- State the theorem
theorem friend_age_order :
  -- Exactly one of the following statements is true
  (IsOldest Friend.Emma ∧ ¬IsYoungest Friend.Fiona ∧ IsOldest Friend.David) ∨
  (¬IsOldest Friend.Emma ∧ IsYoungest Friend.Fiona ∧ IsOldest Friend.David) ∨
  (¬IsOldest Friend.Emma ∧ ¬IsYoungest Friend.Fiona ∧ ¬IsOldest Friend.David) →
  -- The age order is David (oldest), Emma (middle), Fiona (youngest)
  AgeOrder Friend.David Friend.Emma ∧ AgeOrder Friend.Emma Friend.Fiona :=
by sorry

end friend_age_order_l2767_276735


namespace football_team_yardage_l2767_276780

/-- A football team's yardage problem -/
theorem football_team_yardage (L : ℤ) : 
  ((-L : ℤ) + 13 = 8) → L = 5 := by
  sorry

end football_team_yardage_l2767_276780


namespace japanese_study_fraction_l2767_276701

theorem japanese_study_fraction (J S : ℕ) (x : ℚ) : 
  S = 2 * J →
  (3 / 8 : ℚ) * S + x * J = (1 / 3 : ℚ) * (J + S) →
  x = 1 / 4 := by
sorry

end japanese_study_fraction_l2767_276701


namespace set_identities_l2767_276798

variable {α : Type*}
variable (A B C : Set α)

theorem set_identities :
  (A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)) ∧
  (A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)) := by
  sorry

end set_identities_l2767_276798


namespace jenny_jill_game_percentage_l2767_276747

theorem jenny_jill_game_percentage :
  -- Define the number of games Jenny played against Mark
  ∀ (games_with_mark : ℕ),
  -- Define Mark's wins
  ∀ (mark_wins : ℕ),
  -- Define Jenny's total wins
  ∀ (jenny_total_wins : ℕ),
  -- Conditions
  games_with_mark = 10 →
  mark_wins = 1 →
  jenny_total_wins = 14 →
  -- Conclusion: Jill's win percentage is 75%
  (((2 * games_with_mark) - (jenny_total_wins - (games_with_mark - mark_wins))) / (2 * games_with_mark) : ℚ) = 3/4 := by
  sorry

end jenny_jill_game_percentage_l2767_276747


namespace parallelogram_area_18_10_l2767_276713

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 18 cm and height 10 cm is 180 cm² -/
theorem parallelogram_area_18_10 : parallelogram_area 18 10 = 180 := by
  sorry

end parallelogram_area_18_10_l2767_276713


namespace complex_sum_of_parts_l2767_276728

theorem complex_sum_of_parts (z : ℂ) (h : z * Complex.I = 1 + Complex.I) :
  z.re + z.im = 2 := by
  sorry

end complex_sum_of_parts_l2767_276728


namespace no_three_way_partition_of_positive_integers_l2767_276766

theorem no_three_way_partition_of_positive_integers :
  ¬ ∃ (A B C : Set ℕ+),
    (A ∪ B ∪ C = Set.univ) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
    (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧
    (∀ x ∈ A, ∀ y ∈ B, (x^2 - x*y + y^2) ∈ C) ∧
    (∀ x ∈ B, ∀ y ∈ C, (x^2 - x*y + y^2) ∈ A) ∧
    (∀ x ∈ C, ∀ y ∈ A, (x^2 - x*y + y^2) ∈ B) :=
by sorry

end no_three_way_partition_of_positive_integers_l2767_276766


namespace not_divisible_by_power_of_two_l2767_276761

theorem not_divisible_by_power_of_two (n : ℕ) (h : n > 1) :
  ¬(2^n ∣ 3^n + 1) :=
by sorry

end not_divisible_by_power_of_two_l2767_276761


namespace min_value_implies_a_inequality_implies_a_range_l2767_276706

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 2|

-- Theorem for part 1
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f a x ≥ 3) ∧ (∃ x, f a x = 3) → a = 1 ∨ a = -5 :=
sorry

-- Theorem for part 2
theorem inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 :=
sorry

end min_value_implies_a_inequality_implies_a_range_l2767_276706


namespace solution_set_equality_l2767_276796

theorem solution_set_equality : Set ℝ := by
  have h : Set ℝ := {x | (x - 1)^2 < 1}
  have g : Set ℝ := Set.Ioo 0 2
  sorry

end solution_set_equality_l2767_276796


namespace gcd_65_169_l2767_276736

theorem gcd_65_169 : Nat.gcd 65 169 = 13 := by
  sorry

end gcd_65_169_l2767_276736


namespace square_side_increase_l2767_276705

theorem square_side_increase (s : ℝ) (h : s > 0) :
  ∃ p : ℝ, (s * (1 + p / 100))^2 = 1.69 * s^2 → p = 30 := by
  sorry

end square_side_increase_l2767_276705


namespace coefficient_m5n3_in_expansion_l2767_276760

theorem coefficient_m5n3_in_expansion : ∀ m n : ℕ,
  (Nat.choose 8 5 : ℕ) = 56 :=
by
  sorry

end coefficient_m5n3_in_expansion_l2767_276760


namespace internet_service_duration_l2767_276702

/-- Calculates the number of days of internet service given the specified parameters. -/
def internetServiceDays (initialBalance : ℚ) (dailyCost : ℚ) (debtLimit : ℚ) (payment : ℚ) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that given the specified parameters, the number of days of internet service is 14. -/
theorem internet_service_duration :
  internetServiceDays 0 (1/2) 5 7 = 14 := by
  sorry

end internet_service_duration_l2767_276702


namespace sqrt_product_equals_three_l2767_276727

theorem sqrt_product_equals_three : Real.sqrt (1/2) * Real.sqrt 18 = 3 := by
  sorry

end sqrt_product_equals_three_l2767_276727


namespace story_problem_solution_l2767_276714

/-- Represents the story writing problem with given parameters -/
structure StoryProblem where
  total_words : ℕ
  num_chapters : ℕ
  total_vocab_terms : ℕ
  vocab_distribution : Fin 4 → ℕ
  words_per_line : ℕ
  lines_per_page : ℕ
  pages_filled : ℚ

/-- Calculates the number of words left to write given a StoryProblem -/
def words_left_to_write (problem : StoryProblem) : ℕ :=
  problem.total_words - (problem.words_per_line * problem.lines_per_page * problem.pages_filled.num / problem.pages_filled.den).toNat

/-- Theorem stating that given the specific problem conditions, 100 words are left to write -/
theorem story_problem_solution (problem : StoryProblem) 
  (h1 : problem.total_words = 400)
  (h2 : problem.num_chapters = 4)
  (h3 : problem.total_vocab_terms = 20)
  (h4 : problem.vocab_distribution 0 = 8)
  (h5 : problem.vocab_distribution 1 = 4)
  (h6 : problem.vocab_distribution 2 = 6)
  (h7 : problem.vocab_distribution 3 = 2)
  (h8 : problem.words_per_line = 10)
  (h9 : problem.lines_per_page = 20)
  (h10 : problem.pages_filled = 3/2) :
  words_left_to_write problem = 100 := by
  sorry


end story_problem_solution_l2767_276714


namespace water_bottle_problem_l2767_276726

def water_bottle_capacity (initial_capacity : ℝ) : Prop :=
  let remaining_after_first_drink := (3/4) * initial_capacity
  let remaining_after_second_drink := (1/3) * remaining_after_first_drink
  remaining_after_second_drink = 1

theorem water_bottle_problem : ∃ (c : ℝ), water_bottle_capacity c ∧ c = 4 := by
  sorry

end water_bottle_problem_l2767_276726


namespace square_friendly_unique_l2767_276781

def square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, ∃ n : ℤ, m^2 + 18*m + c = n^2

theorem square_friendly_unique : 
  (square_friendly 81) ∧ (∀ c : ℤ, square_friendly c → c = 81) :=
sorry

end square_friendly_unique_l2767_276781


namespace phil_wins_n_12_ellie_wins_n_2012_l2767_276709

/-- Represents a move on the chessboard -/
structure Move where
  x : Nat
  y : Nat
  shape : Fin 4 -- 4 possible L-shapes

/-- Represents the state of the chessboard -/
def Board (n : Nat) := Fin n → Fin n → Fin n

/-- Applies a move to the board -/
def applyMove (n : Nat) (board : Board n) (move : Move) : Board n :=
  sorry

/-- Checks if all numbers on the board are zero -/
def allZero (n : Nat) (board : Board n) : Prop :=
  sorry

/-- Sum of all numbers on the board -/
def boardSum (n : Nat) (board : Board n) : Nat :=
  sorry

theorem phil_wins_n_12 :
  ∃ (initial : Board 12),
    ∀ (moves : List Move),
      ¬(boardSum 12 (moves.foldl (applyMove 12) initial) % 3 = 0) :=
sorry

theorem ellie_wins_n_2012 :
  ∀ (initial : Board 2012),
    ∃ (moves : List Move),
      allZero 2012 (moves.foldl (applyMove 2012) initial) :=
sorry

end phil_wins_n_12_ellie_wins_n_2012_l2767_276709


namespace pink_highlighters_count_l2767_276731

def total_highlighters : ℕ := 33
def yellow_highlighters : ℕ := 15
def blue_highlighters : ℕ := 8

theorem pink_highlighters_count :
  total_highlighters - yellow_highlighters - blue_highlighters = 10 := by
  sorry

end pink_highlighters_count_l2767_276731


namespace min_value_a_plus_b_l2767_276752

theorem min_value_a_plus_b (a b : ℝ) (h : a^2 + 2*b^2 = 6) :
  ∃ (m : ℝ), (∀ (x y : ℝ), x^2 + 2*y^2 = 6 → m ≤ x + y) ∧ (m = -3) := by
  sorry

end min_value_a_plus_b_l2767_276752


namespace sequence_convergence_l2767_276794

def S (seq : List Int) : List Int :=
  let n := seq.length
  List.zipWith (· * ·) seq ((seq.drop 1).append [seq.head!])

def all_ones (seq : List Int) : Prop :=
  seq.all (· = 1)

theorem sequence_convergence (n : Nat) (seq : List Int) :
  seq.length = 2 * n →
  (∀ i, i ∈ seq → i = 1 ∨ i = -1) →
  ∃ k : Nat, all_ones (Nat.iterate S k seq) := by
  sorry

end sequence_convergence_l2767_276794


namespace michaels_birds_l2767_276716

/-- Given Michael's pets distribution, prove he has 12 birds -/
theorem michaels_birds (total_pets : ℕ) (dog_percent cat_percent bunny_percent : ℚ) : 
  total_pets = 120 →
  dog_percent = 30 / 100 →
  cat_percent = 40 / 100 →
  bunny_percent = 20 / 100 →
  (↑total_pets * (1 - dog_percent - cat_percent - bunny_percent) : ℚ) = 12 := by
  sorry

end michaels_birds_l2767_276716


namespace pens_pencils_cost_l2767_276779

def total_spent : ℝ := 32
def backpack_cost : ℝ := 15
def notebook_cost : ℝ := 3
def num_notebooks : ℕ := 5

def cost_pens_pencils : ℝ := total_spent - (backpack_cost + notebook_cost * num_notebooks)

theorem pens_pencils_cost (h : cost_pens_pencils = 2) : 
  cost_pens_pencils / 2 = 1 := by sorry

end pens_pencils_cost_l2767_276779


namespace triple_sum_of_45_2_and_quarter_l2767_276704

theorem triple_sum_of_45_2_and_quarter (x : ℝ) (h : x = 45.2 + (1 / 4)) :
  3 * x = 136.35 := by
  sorry

end triple_sum_of_45_2_and_quarter_l2767_276704


namespace root_sum_zero_l2767_276753

theorem root_sum_zero (a b : ℝ) : 
  (Complex.I + 1) ^ 2 + a * (Complex.I + 1) + b = 0 → a + b = 0 := by
  sorry

end root_sum_zero_l2767_276753


namespace ceiling_negative_three_point_seven_l2767_276772

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end ceiling_negative_three_point_seven_l2767_276772


namespace jony_walking_speed_l2767_276737

/-- Calculates the walking speed given distance and time -/
def walking_speed (distance : ℕ) (time : ℕ) : ℕ :=
  distance / time

/-- The number of blocks Jony walks -/
def blocks_walked : ℕ := (90 - 10) + (90 - 70)

/-- The length of each block in meters -/
def block_length : ℕ := 40

/-- The total distance Jony walks in meters -/
def total_distance : ℕ := blocks_walked * block_length

/-- The total time Jony spends walking in minutes -/
def total_time : ℕ := 40

theorem jony_walking_speed :
  walking_speed total_distance total_time = 100 := by
  sorry

end jony_walking_speed_l2767_276737


namespace friday_sales_l2767_276750

/-- Kim's cupcake sales pattern --/
def cupcake_sales (tuesday_before_discount : ℕ) : ℕ :=
  let tuesday := tuesday_before_discount + (tuesday_before_discount * 5 / 100)
  let monday := tuesday + (tuesday * 50 / 100)
  let wednesday := tuesday * 3 / 2
  let thursday := wednesday - (wednesday * 20 / 100)
  let friday := thursday * 13 / 10
  friday

/-- Theorem: Kim sold 1310 boxes on Friday --/
theorem friday_sales : cupcake_sales 800 = 1310 := by
  sorry

end friday_sales_l2767_276750


namespace campaign_fundraising_l2767_276710

theorem campaign_fundraising (max_donation : ℕ) (max_donors : ℕ) (half_donors : ℕ) (percentage : ℚ) : 
  max_donation = 1200 →
  max_donors = 500 →
  half_donors = 3 * max_donors →
  percentage = 40 / 100 →
  (max_donation * max_donors + (max_donation / 2) * half_donors) / percentage = 3750000 := by
sorry

end campaign_fundraising_l2767_276710


namespace angle_C_is_pi_third_max_area_when_c_is_2root3_l2767_276788

noncomputable section

open Real

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = π)
  (h3 : a > 0 ∧ b > 0 ∧ c > 0)
  (h4 : a * sin B = b * sin A)
  (h5 : b * sin C = c * sin B)
  (h6 : c * sin A = a * sin C)

variable (t : Triangle)

-- Given condition
axiom given_condition : t.a * cos t.B + t.b * cos t.A = 2 * t.c * cos t.C

-- Theorem 1: Prove that C = π/3
theorem angle_C_is_pi_third : t.C = π/3 :=
sorry

-- Theorem 2: Prove that if c = 2√3, the maximum area of triangle ABC is 3√3
theorem max_area_when_c_is_2root3 (h : t.c = 2 * Real.sqrt 3) :
  (∀ s : Triangle, s.c = t.c → t.a * t.b * sin t.C / 2 ≥ s.a * s.b * sin s.C / 2) ∧
  t.a * t.b * sin t.C / 2 = 3 * Real.sqrt 3 :=
sorry

end angle_C_is_pi_third_max_area_when_c_is_2root3_l2767_276788


namespace extreme_value_and_monotonicity_l2767_276776

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3 * x^2 + m * x) / Real.exp x

theorem extreme_value_and_monotonicity (m : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, 0 < |x| ∧ |x| < ε → f 0 0 ≤ f 0 x) ∧
  (∀ x, x ≥ 3 → ∀ y, y > x → f m y ≤ f m x) ↔ m ≤ 2 :=
sorry

end extreme_value_and_monotonicity_l2767_276776


namespace fraction_of_single_men_l2767_276732

theorem fraction_of_single_men (total : ℕ) (h1 : total > 0) :
  let women := (60 : ℚ) / 100 * total
  let men := total - women
  let married := (60 : ℚ) / 100 * total
  let married_men := (1 : ℚ) / 4 * men
  (men - married_men) / men = (3 : ℚ) / 4 :=
by sorry

end fraction_of_single_men_l2767_276732


namespace intersection_of_M_and_N_l2767_276793

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = 3 * p.1 + 4}
def N : Set (ℝ × ℝ) := {p | p.2 = p.1 ^ 2}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {(-1, 1), (4, 16)} := by sorry

end intersection_of_M_and_N_l2767_276793


namespace inequality_solution_set_l2767_276764

theorem inequality_solution_set (x : ℝ) : 
  ((x + 2) * (1 - x) > 0) ↔ (-2 < x ∧ x < 1) := by sorry

end inequality_solution_set_l2767_276764


namespace M_mod_55_l2767_276754

def M : ℕ := sorry

theorem M_mod_55 : M % 55 = 50 := by sorry

end M_mod_55_l2767_276754


namespace expression_value_l2767_276707

theorem expression_value : 
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 1
  x^2 + y^2 - 2*z^2 + 3*x*y = -7 := by
  sorry

end expression_value_l2767_276707


namespace factor_expression_l2767_276744

theorem factor_expression : ∀ x : ℝ, 12 * x^2 + 8 * x = 4 * x * (3 * x + 2) := by
  sorry

end factor_expression_l2767_276744


namespace number_of_tangent_lines_l2767_276768

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The hyperbola 4x^2-9y^2=36 -/
def hyperbola (x y : ℝ) : Prop := 4 * x^2 - 9 * y^2 = 36

/-- A line passes through a point -/
def passes_through (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.y_intercept

/-- A line has only one intersection point with the hyperbola -/
def has_one_intersection (l : Line) : Prop :=
  ∃! x y, passes_through l x y ∧ hyperbola x y

/-- The theorem to be proved -/
theorem number_of_tangent_lines : 
  ∃! (l₁ l₂ l₃ : Line), 
    (passes_through l₁ 3 0 ∧ has_one_intersection l₁) ∧
    (passes_through l₂ 3 0 ∧ has_one_intersection l₂) ∧
    (passes_through l₃ 3 0 ∧ has_one_intersection l₃) ∧
    (∀ l, passes_through l 3 0 ∧ has_one_intersection l → l = l₁ ∨ l = l₂ ∨ l = l₃) :=
sorry

end number_of_tangent_lines_l2767_276768


namespace single_digit_addition_l2767_276720

theorem single_digit_addition (A : ℕ) : 
  A < 10 → -- A is a single digit number
  10 * A + A + 10 * A + A = 132 → -- AA + AA = 132
  A = 6 := by sorry

end single_digit_addition_l2767_276720


namespace repeating_decimal_sum_l2767_276756

theorem repeating_decimal_sum (c d : ℕ) (h : (4 : ℚ) / 13 = 0.1 * c + 0.01 * d + 0.001 * (c + d / 10)) : c + d = 10 := by
  sorry

end repeating_decimal_sum_l2767_276756


namespace magnitude_2a_plus_b_l2767_276771

variable (a b : ℝ × ℝ)

theorem magnitude_2a_plus_b (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 2) 
  (h3 : a - b = (Real.sqrt 3, Real.sqrt 2)) : 
  ‖2 • a + b‖ = 2 * Real.sqrt 2 := by
  sorry

end magnitude_2a_plus_b_l2767_276771


namespace product_selection_theorem_product_display_theorem_l2767_276765

def total_products : ℕ := 10
def ineligible_products : ℕ := 2
def products_to_select : ℕ := 4
def display_positions : ℕ := 6
def gold_medal_products : ℕ := 2

def arrangement (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem product_selection_theorem :
  arrangement (total_products - ineligible_products) products_to_select = 1680 :=
sorry

theorem product_display_theorem :
  arrangement display_positions gold_medal_products *
  arrangement (total_products - gold_medal_products) (display_positions - gold_medal_products) = 50400 :=
sorry

end product_selection_theorem_product_display_theorem_l2767_276765


namespace sqrt3_minus1_power0_plus_2_power_neg1_l2767_276787

theorem sqrt3_minus1_power0_plus_2_power_neg1 : (Real.sqrt 3 - 1) ^ 0 + 2 ^ (-1 : ℤ) = (3 : ℝ) / 2 := by
  sorry

end sqrt3_minus1_power0_plus_2_power_neg1_l2767_276787


namespace vector_sum_proof_l2767_276733

/-- Given vectors a = (2,3) and b = (-1,2), prove their sum is (1,5) -/
theorem vector_sum_proof :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![-1, 2]
  (a + b) = ![1, 5] := by sorry

end vector_sum_proof_l2767_276733


namespace circle_equation_l2767_276791

/-- 
Given a circle with radius 2, center on the positive x-axis, and tangent to the y-axis,
prove that its equation is x^2 + y^2 - 4x = 0.
-/
theorem circle_equation (x y : ℝ) : 
  ∃ (h : ℝ), h > 0 ∧ 
  (∀ (a b : ℝ), (a - h)^2 + b^2 = 4 → a ≥ 0) ∧
  (∃ (c : ℝ), c^2 = 4 ∧ (h - 0)^2 + c^2 = 4) →
  x^2 + y^2 - 4*x = 0 :=
sorry

end circle_equation_l2767_276791


namespace complex_modulus_l2767_276746

theorem complex_modulus (z : ℂ) : z = (1 + 3*I) / (3 - I) → Complex.abs z = 1 := by
  sorry

end complex_modulus_l2767_276746

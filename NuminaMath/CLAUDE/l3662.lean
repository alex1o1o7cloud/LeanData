import Mathlib

namespace greatest_divisible_by_seven_ninety_five_six_five_nine_is_valid_l3662_366277

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

theorem greatest_divisible_by_seven :
  ∀ n : ℕ, is_valid_number n ∧ n % 7 = 0 → n ≤ 95659 :=
by sorry

theorem ninety_five_six_five_nine_is_valid :
  is_valid_number 95659 ∧ 95659 % 7 = 0 :=
by sorry

end greatest_divisible_by_seven_ninety_five_six_five_nine_is_valid_l3662_366277


namespace lesser_solution_quadratic_l3662_366201

theorem lesser_solution_quadratic (x : ℝ) : 
  x^2 + 9*x - 22 = 0 ∧ (∀ y : ℝ, y^2 + 9*y - 22 = 0 → x ≤ y) → x = -11 :=
by sorry

end lesser_solution_quadratic_l3662_366201


namespace pyramid_max_volume_l3662_366290

/-- The maximum volume of a pyramid SABC with given conditions -/
theorem pyramid_max_volume (AB AC : ℝ) (sin_BAC : ℝ) (h : ℝ) :
  AB = 5 →
  AC = 8 →
  sin_BAC = 4/5 →
  h ≤ (5 * Real.sqrt 137 / 8) * Real.sqrt 3 →
  (1/3 : ℝ) * (1/2 * AB * AC * sin_BAC) * h ≤ 10 * Real.sqrt (137/3) :=
by sorry

end pyramid_max_volume_l3662_366290


namespace percent_calculation_l3662_366293

theorem percent_calculation :
  (0.02 / 100) * 12356 = 2.4712 := by sorry

end percent_calculation_l3662_366293


namespace council_composition_l3662_366235

/-- Represents a member of the council -/
inductive Member
| Knight
| Liar

/-- The total number of council members -/
def total_members : Nat := 101

/-- Proposition that if any member is removed, the majority of remaining members would be liars -/
def majority_liars_if_removed (knights : Nat) (liars : Nat) : Prop :=
  ∀ (m : Member), 
    (m = Member.Knight → liars > (knights + liars - 1) / 2) ∧
    (m = Member.Liar → knights ≤ (knights + liars - 1) / 2)

theorem council_composition :
  ∃ (knights liars : Nat),
    knights + liars = total_members ∧
    majority_liars_if_removed knights liars ∧
    knights = 50 ∧ liars = 51 := by
  sorry

end council_composition_l3662_366235


namespace prob_even_sum_coins_and_dice_l3662_366259

/-- Represents the outcome of tossing a fair coin -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of rolling a fair die -/
def DieOutcome := Fin 6

/-- The probability of getting heads on a fair coin toss -/
def probHeads : ℚ := 1/2

/-- The probability of getting an even number on a fair die roll -/
def probEvenDie : ℚ := 1/2

/-- The number of coins tossed -/
def numCoins : ℕ := 3

/-- Calculates the probability of getting k heads in n coin tosses -/
def probKHeads (n k : ℕ) : ℚ := sorry

/-- Calculates the probability of getting an even sum when rolling k fair dice -/
def probEvenSumKDice (k : ℕ) : ℚ := sorry

theorem prob_even_sum_coins_and_dice :
  (probKHeads numCoins 0 * 1 +
   probKHeads numCoins 1 * probEvenDie +
   probKHeads numCoins 2 * probEvenSumKDice 2 +
   probKHeads numCoins 3 * probEvenSumKDice 3) = 15/16 := by sorry

end prob_even_sum_coins_and_dice_l3662_366259


namespace quadratic_unique_solution_l3662_366244

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 + (2-a)*x + 1

-- Define the solution range
def in_range (x : ℝ) : Prop := -1 < x ∧ x ≤ 3 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2

-- Define the uniqueness of the solution
def unique_solution (a : ℝ) : Prop :=
  ∃! x, quadratic a x = 0 ∧ in_range x

-- State the theorem
theorem quadratic_unique_solution :
  ∀ a : ℝ, unique_solution a ↔ 
    (a = 4.5) ∨ 
    (a < 0) ∨ 
    (a > 16/3) :=
sorry

end quadratic_unique_solution_l3662_366244


namespace athena_total_spent_l3662_366279

def sandwich_price : ℝ := 3
def fruit_drink_price : ℝ := 2.5
def num_sandwiches : ℕ := 3
def num_fruit_drinks : ℕ := 2

theorem athena_total_spent :
  (num_sandwiches : ℝ) * sandwich_price + (num_fruit_drinks : ℝ) * fruit_drink_price = 14 := by
  sorry

end athena_total_spent_l3662_366279


namespace sum_lower_bound_l3662_366266

theorem sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a + b ≥ 6 := by
  sorry

end sum_lower_bound_l3662_366266


namespace gift_wrapping_l3662_366252

theorem gift_wrapping (total_rolls total_gifts first_roll_gifts second_roll_gifts : ℕ) :
  total_rolls = 3 →
  total_gifts = 12 →
  first_roll_gifts = 3 →
  second_roll_gifts = 5 →
  total_gifts = first_roll_gifts + second_roll_gifts + (total_gifts - (first_roll_gifts + second_roll_gifts)) →
  (total_gifts - (first_roll_gifts + second_roll_gifts)) = 4 :=
by
  sorry

end gift_wrapping_l3662_366252


namespace only_set_D_forms_triangle_l3662_366240

/-- Checks if three lengths can form a triangle according to the triangle inequality theorem -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The sets of line segments given in the problem -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(3, 4, 8), (5, 6, 11), (3, 1, 1), (3, 4, 6)]

/-- Theorem stating that only the set (3, 4, 6) can form a triangle -/
theorem only_set_D_forms_triangle :
  ∃! (a b c : ℝ), (a, b, c) ∈ segment_sets ∧ can_form_triangle a b c :=
by sorry

end only_set_D_forms_triangle_l3662_366240


namespace symmetric_sine_function_value_l3662_366236

theorem symmetric_sine_function_value (a φ : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (3 * x + φ)
  (∀ x, f (a + x) = f (a - x)) →
  f (a + π / 6) = 0 := by
  sorry

end symmetric_sine_function_value_l3662_366236


namespace expenditure_recording_l3662_366210

/-- Represents the way a transaction is recorded in accounting -/
inductive AccountingRecord
  | Positive (amount : ℤ)
  | Negative (amount : ℤ)

/-- Records an income transaction -/
def recordIncome (amount : ℤ) : AccountingRecord :=
  AccountingRecord.Positive amount

/-- Records an expenditure transaction -/
def recordExpenditure (amount : ℤ) : AccountingRecord :=
  AccountingRecord.Negative amount

/-- The accounting principle for recording transactions -/
axiom accounting_principle (amount : ℤ) :
  (recordIncome amount = AccountingRecord.Positive amount) ∧
  (recordExpenditure amount = AccountingRecord.Negative amount)

/-- Theorem: An expenditure of 100 should be recorded as -100 -/
theorem expenditure_recording :
  recordExpenditure 100 = AccountingRecord.Negative 100 := by
  sorry

end expenditure_recording_l3662_366210


namespace matrix_power_4_l3662_366209

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 0]

theorem matrix_power_4 :
  A ^ 4 = !![5, -4; 4, -3] := by sorry

end matrix_power_4_l3662_366209


namespace tangent_line_y_intercept_l3662_366206

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- Checks if a point is in the first quadrant -/
def isInFirstQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 0), radius := 3 }
  let c2 : Circle := { center := (8, 0), radius := 2 }
  ∀ l : Line,
    (∃ p1 p2 : ℝ × ℝ,
      isTangent l c1 ∧
      isTangent l c2 ∧
      isInFirstQuadrant p1 ∧
      isInFirstQuadrant p2 ∧
      (p1.1 - 3)^2 + p1.2^2 = 9 ∧
      (p2.1 - 8)^2 + p2.2^2 = 4) →
    l.intercept = 13/4 := by
  sorry

end tangent_line_y_intercept_l3662_366206


namespace factorization_of_cubic_minus_linear_l3662_366223

theorem factorization_of_cubic_minus_linear (x : ℝ) :
  3 * x^3 - 12 * x = 3 * x * (x - 2) * (x + 2) := by sorry

end factorization_of_cubic_minus_linear_l3662_366223


namespace complex_equation_solution_l3662_366227

theorem complex_equation_solution (z : ℂ) :
  z + Complex.abs z = 2 + 8 * Complex.I → z = -15 + 8 * Complex.I :=
by
  sorry

end complex_equation_solution_l3662_366227


namespace tyler_cds_l3662_366211

theorem tyler_cds (initial : ℕ) : 
  (2 / 3 : ℚ) * initial + 8 = 22 → initial = 21 := by
  sorry

end tyler_cds_l3662_366211


namespace product_inequality_l3662_366298

theorem product_inequality (x₁ x₂ x₃ x₄ y₁ y₂ : ℝ) 
  (h1 : y₂ ≥ y₁ ∧ y₁ ≥ x₁ ∧ x₁ ≥ x₃ ∧ x₃ ≥ x₂ ∧ x₂ ≥ x₁ ∧ x₁ ≥ 2)
  (h2 : x₁ + x₂ + x₃ + x₄ ≥ y₁ + y₂) :
  x₁ * x₂ * x₃ * x₄ ≥ y₁ * y₂ := by
sorry

end product_inequality_l3662_366298


namespace prob_15th_roll_last_correct_l3662_366275

/-- The probability of the 15th roll being the last roll when rolling an
    eight-sided die until getting the same number on consecutive rolls. -/
def prob_15th_roll_last : ℚ :=
  (7 ^ 13 : ℚ) / (8 ^ 14 : ℚ)

/-- The number of sides on the die. -/
def num_sides : ℕ := 8

/-- The number of rolls. -/
def num_rolls : ℕ := 15

theorem prob_15th_roll_last_correct :
  prob_15th_roll_last = (7 ^ (num_rolls - 2) : ℚ) / (num_sides ^ (num_rolls - 1) : ℚ) :=
sorry

end prob_15th_roll_last_correct_l3662_366275


namespace bumper_cars_cost_l3662_366228

/-- The cost of bumper cars given the costs of other attractions and ticket information -/
theorem bumper_cars_cost 
  (total_cost : ℕ → ℕ → ℕ)  -- Function to calculate total cost
  (current_tickets : ℕ)     -- Current number of tickets
  (additional_tickets : ℕ)  -- Additional tickets needed
  (ferris_wheel_cost : ℕ)   -- Cost of Ferris wheel
  (roller_coaster_cost : ℕ) -- Cost of roller coaster
  (h1 : current_tickets = 5)
  (h2 : additional_tickets = 8)
  (h3 : ferris_wheel_cost = 5)
  (h4 : roller_coaster_cost = 4)
  (h5 : ∀ x y, total_cost x y = x + y) -- Definition of total cost function
  : ∃ (bumper_cars_cost : ℕ), 
    total_cost current_tickets additional_tickets = 
    ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost ∧ 
    bumper_cars_cost = 4 :=
by
  sorry

end bumper_cars_cost_l3662_366228


namespace cosine_rule_triangle_l3662_366203

theorem cosine_rule_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  a = 3 ∧ b = 4 ∧ c = 6 → cos_B = 29 / 36 := by
  sorry

end cosine_rule_triangle_l3662_366203


namespace circle_ratio_l3662_366243

theorem circle_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  π * b^2 - π * a^2 = 5 * (π * a^2) → a / b = Real.sqrt 6 / 6 :=
by sorry

end circle_ratio_l3662_366243


namespace interest_difference_l3662_366220

theorem interest_difference (principal rate time : ℝ) 
  (h_principal : principal = 600)
  (h_rate : rate = 0.05)
  (h_time : time = 8) :
  principal - (principal * rate * time) = 360 := by
  sorry

end interest_difference_l3662_366220


namespace sum_of_possible_AB_values_l3662_366202

/-- Represents a 7-digit number in the form A568B72 -/
def SevenDigitNumber (A B : Nat) : Nat :=
  A * 1000000 + 568000 + B * 100 + 72

theorem sum_of_possible_AB_values :
  (∀ A B : Nat, A < 10 ∧ B < 10 →
    SevenDigitNumber A B % 9 = 0 →
    (A + B = 8 ∨ A + B = 17)) ∧
  (∃ A₁ B₁ A₂ B₂ : Nat,
    A₁ < 10 ∧ B₁ < 10 ∧ A₂ < 10 ∧ B₂ < 10 ∧
    SevenDigitNumber A₁ B₁ % 9 = 0 ∧
    SevenDigitNumber A₂ B₂ % 9 = 0 ∧
    A₁ + B₁ = 8 ∧ A₂ + B₂ = 17) :=
by sorry

#check sum_of_possible_AB_values

end sum_of_possible_AB_values_l3662_366202


namespace f_difference_f_equation_solution_l3662_366274

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem 1
theorem f_difference (a : ℝ) : f a - f (a + 1) = -2 * a - 1 := by
  sorry

-- Theorem 2
theorem f_equation_solution : {x : ℝ | f x = x + 3} = {-1, 2} := by
  sorry

end f_difference_f_equation_solution_l3662_366274


namespace number_of_boys_l3662_366246

/-- Proves that the number of boys is 5 given the problem conditions -/
theorem number_of_boys (men : ℕ) (women : ℕ) (boys : ℕ) (total_earnings : ℕ) (men_wage : ℕ) :
  men = 5 →
  men = women →
  women = boys →
  total_earnings = 90 →
  men_wage = 6 →
  boys = 5 := by
sorry

end number_of_boys_l3662_366246


namespace expression_simplification_and_evaluation_expression_evaluation_at_3_l3662_366296

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (x^2 - 4*x + 4) / (x^2 - 4) / ((x - 2) / (x^2 + 2*x)) + 3 = x + 3 :=
by sorry

theorem expression_evaluation_at_3 :
  let x : ℝ := 3
  (x^2 - 4*x + 4) / (x^2 - 4) / ((x - 2) / (x^2 + 2*x)) + 3 = 6 :=
by sorry

end expression_simplification_and_evaluation_expression_evaluation_at_3_l3662_366296


namespace smallest_sphere_and_largest_cylinder_radius_l3662_366287

/-- Three identical cylindrical surfaces with radius R and mutually perpendicular axes that touch each other in pairs -/
structure PerpendicularCylinders (R : ℝ) :=
  (radius : ℝ := R)
  (perpendicular_axes : Prop)
  (touch_in_pairs : Prop)

theorem smallest_sphere_and_largest_cylinder_radius 
  (R : ℝ) 
  (h : R > 0) 
  (cylinders : PerpendicularCylinders R) : 
  ∃ (smallest_sphere_radius largest_cylinder_radius : ℝ),
    smallest_sphere_radius = (Real.sqrt 2 - 1) * R ∧
    largest_cylinder_radius = (Real.sqrt 2 - 1) * R :=
by sorry

end smallest_sphere_and_largest_cylinder_radius_l3662_366287


namespace intersection_point_l3662_366232

-- Define the system of equations
def line1 (x y : ℚ) : Prop := 6 * x - 3 * y = 18
def line2 (x y : ℚ) : Prop := 8 * x + 2 * y = 20

-- State the theorem
theorem intersection_point :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (8/3, -2/3) := by
  sorry

end intersection_point_l3662_366232


namespace cuboid_vertices_sum_l3662_366249

theorem cuboid_vertices_sum (n : ℕ) (h : 6 * n + 12 * n = 216) : 8 * n = 96 := by
  sorry

end cuboid_vertices_sum_l3662_366249


namespace single_elimination_tournament_games_l3662_366294

/-- Calculates the number of games in a single-elimination tournament -/
def tournament_games (n : ℕ) : ℕ :=
  n - 1

/-- The number of teams in the tournament -/
def num_teams : ℕ := 24

theorem single_elimination_tournament_games :
  tournament_games num_teams = 23 := by
  sorry

end single_elimination_tournament_games_l3662_366294


namespace ajax_exercise_hours_per_day_l3662_366278

/-- Calculates the number of hours Ajax needs to exercise per day to reach his weight loss goal. -/
theorem ajax_exercise_hours_per_day 
  (initial_weight_kg : ℝ)
  (weight_loss_per_hour_lbs : ℝ)
  (kg_to_lbs_conversion : ℝ)
  (final_weight_lbs : ℝ)
  (days_of_exercise : ℕ)
  (h1 : initial_weight_kg = 80)
  (h2 : weight_loss_per_hour_lbs = 1.5)
  (h3 : kg_to_lbs_conversion = 2.2)
  (h4 : final_weight_lbs = 134)
  (h5 : days_of_exercise = 14) :
  (initial_weight_kg * kg_to_lbs_conversion - final_weight_lbs) / (weight_loss_per_hour_lbs * days_of_exercise) = 2 := by
  sorry

#check ajax_exercise_hours_per_day

end ajax_exercise_hours_per_day_l3662_366278


namespace a_eq_zero_necessary_not_sufficient_l3662_366256

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem a_eq_zero_necessary_not_sufficient :
  ∃ (a b : ℝ), (∀ (a' b' : ℝ), isPureImaginary (Complex.mk a' b') → a' = 0) ∧
               (∃ (a'' b'' : ℝ), a'' = 0 ∧ ¬isPureImaginary (Complex.mk a'' b'')) :=
by sorry

end a_eq_zero_necessary_not_sufficient_l3662_366256


namespace unique_sums_count_l3662_366248

def bag_x : Finset ℕ := {1, 4, 7}
def bag_y : Finset ℕ := {3, 5, 8}

def possible_sums : Finset ℕ := (bag_x.product bag_y).image (λ (x, y) => x + y)

theorem unique_sums_count : possible_sums.card = 7 := by sorry

end unique_sums_count_l3662_366248


namespace square_area_doubling_l3662_366292

theorem square_area_doubling (s : ℝ) (h : s > 0) :
  (2 * s) ^ 2 = 4 * s ^ 2 := by sorry

end square_area_doubling_l3662_366292


namespace quadratic_real_roots_condition_l3662_366222

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0) ↔ (k ≥ -1 ∧ k ≠ 0) :=
by sorry

end quadratic_real_roots_condition_l3662_366222


namespace ratio_problem_l3662_366212

theorem ratio_problem (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 7) :
  d / a = 2 / 35 := by
sorry

end ratio_problem_l3662_366212


namespace sales_department_replacement_l3662_366205

/-- Represents the ages and work experience of employees in a sales department. -/
structure SalesDepartment where
  initialMenCount : ℕ
  initialAvgAge : ℝ
  initialAvgExperience : ℝ
  replacedMenAges : Fin 2 → ℕ
  womenAgeRanges : Fin 2 → Set ℕ
  newAvgAge : ℝ
  newAvgExperience : ℝ

/-- Theorem stating the average age of the two women and the change in work experience. -/
theorem sales_department_replacement
  (dept : SalesDepartment)
  (h_men_count : dept.initialMenCount = 8)
  (h_age_increase : dept.newAvgAge = dept.initialAvgAge + 2)
  (h_exp_change : dept.newAvgExperience = dept.initialAvgExperience + 1)
  (h_replaced_ages : dept.replacedMenAges 0 = 20 ∧ dept.replacedMenAges 1 = 24)
  (h_women_ages : dept.womenAgeRanges 0 = Set.Icc 26 30 ∧ dept.womenAgeRanges 1 = Set.Icc 32 36) :
  ∃ (w₁ w₂ : ℕ), w₁ ∈ dept.womenAgeRanges 0 ∧ w₂ ∈ dept.womenAgeRanges 1 ∧
  (w₁ + w₂) / 2 = 30 ∧
  (dept.initialMenCount * dept.newAvgExperience - dept.initialMenCount * dept.initialAvgExperience) = 8 := by
  sorry


end sales_department_replacement_l3662_366205


namespace work_completion_time_l3662_366263

/-- The number of days it takes for A to complete the work alone -/
def days_A : ℝ := 6

/-- The number of days it takes for B to complete the work alone -/
def days_B : ℝ := 12

/-- The number of days it takes for A and B to complete the work together -/
def days_AB : ℝ := 4

/-- Theorem stating that given the time for B and the time for A and B together, 
    we can determine the time for A alone -/
theorem work_completion_time : 
  (1 / days_A + 1 / days_B = 1 / days_AB) → days_A = 6 := by sorry

end work_completion_time_l3662_366263


namespace wallet_value_theorem_l3662_366221

/-- Represents the total value of bills in a wallet -/
def wallet_value (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ) : ℕ :=
  5 * five_dollar_bills + 10 * ten_dollar_bills

/-- Theorem: The total value of 4 $5 bills and 8 $10 bills is $100 -/
theorem wallet_value_theorem : wallet_value 4 8 = 100 := by
  sorry

#eval wallet_value 4 8

end wallet_value_theorem_l3662_366221


namespace student_count_l3662_366215

theorem student_count (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 13) 
  (h2 : rank_from_left = 8) : 
  rank_from_right + rank_from_left - 1 = 20 := by
  sorry

end student_count_l3662_366215


namespace dorothy_profit_l3662_366214

/-- Given the cost of ingredients, number of doughnuts made, and selling price per doughnut,
    calculate the profit. -/
def calculate_profit (ingredient_cost : ℕ) (num_doughnuts : ℕ) (price_per_doughnut : ℕ) : ℕ :=
  num_doughnuts * price_per_doughnut - ingredient_cost

/-- Theorem stating that Dorothy's profit is $22 given the problem conditions. -/
theorem dorothy_profit :
  calculate_profit 53 25 3 = 22 := by
  sorry

end dorothy_profit_l3662_366214


namespace moon_radius_scientific_notation_l3662_366273

/-- The radius of the moon in meters -/
def moon_radius : ℝ := 1738000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Theorem stating that the moon's radius is correctly expressed in scientific notation -/
theorem moon_radius_scientific_notation :
  ∃ (sn : ScientificNotation), moon_radius = sn.coefficient * (10 : ℝ) ^ sn.exponent :=
sorry

end moon_radius_scientific_notation_l3662_366273


namespace complex_equation_solution_l3662_366204

theorem complex_equation_solution :
  ∀ (z : ℂ), (1 + Complex.I) * z = 2 + Complex.I → z = 3/2 - (1/2) * Complex.I :=
by sorry

end complex_equation_solution_l3662_366204


namespace total_cookie_time_l3662_366229

/-- The time it takes to make black & white cookies -/
def cookie_making_time (batter_time baking_time cooling_time white_icing_time chocolate_icing_time : ℕ) : ℕ :=
  batter_time + baking_time + cooling_time + white_icing_time + chocolate_icing_time

/-- Theorem stating that the total time to make black & white cookies is 100 minutes -/
theorem total_cookie_time :
  ∃ (batter_time cooling_time : ℕ),
    batter_time = 10 ∧
    cooling_time = 15 ∧
    cookie_making_time batter_time 15 cooling_time 30 30 = 100 := by
  sorry

end total_cookie_time_l3662_366229


namespace backyard_area_l3662_366286

theorem backyard_area (length width : ℝ) 
  (h1 : length * 50 = 2000)
  (h2 : (2 * length + 2 * width) * 20 = 2000) :
  length * width = 400 := by
  sorry

end backyard_area_l3662_366286


namespace max_value_abc_l3662_366268

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  a + b^2 + c^4 ≤ 3 := by
sorry

end max_value_abc_l3662_366268


namespace correct_product_l3662_366265

theorem correct_product (a b : ℕ) : 
  10 ≤ a ∧ a < 100 →  -- a is a two-digit number
  (∃ x y : ℕ, x * 10 + y = a ∧ y * 10 + x = (189 / b)) →  -- reversing digits of a and multiplying by b gives 189
  a * b = 108 := by
sorry

end correct_product_l3662_366265


namespace points_on_line_implies_b_value_l3662_366247

/-- Three points lie on the same line if and only if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The theorem states that if the given points lie on the same line, then b = -1/2. -/
theorem points_on_line_implies_b_value (b : ℝ) :
  collinear 6 (-10) (-b + 4) 3 (3*b + 6) 3 → b = -1/2 := by
  sorry

#check points_on_line_implies_b_value

end points_on_line_implies_b_value_l3662_366247


namespace jessicas_allowance_l3662_366208

theorem jessicas_allowance (allowance : ℝ) : 
  (allowance / 2 + 6 = 11) → allowance = 10 := by
  sorry

end jessicas_allowance_l3662_366208


namespace unique_point_in_S_l3662_366280

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ Real.log (p.1^3 + (1/3) * p.2^3 + 1/9) = Real.log p.1 + Real.log p.2}

theorem unique_point_in_S : ∃! p : ℝ × ℝ, p ∈ S := by
  sorry

end unique_point_in_S_l3662_366280


namespace faulty_key_theorem_l3662_366282

/-- Represents a sequence of digits -/
def DigitSequence := List Nat

/-- Checks if a digit is valid (0-9) -/
def isValidDigit (d : Nat) : Bool := d ≤ 9

/-- Represents the count of each digit in a sequence -/
def DigitCount := Nat → Nat

/-- Counts the occurrences of each digit in a sequence -/
def countDigits (seq : DigitSequence) : DigitCount :=
  fun d => seq.filter (· = d) |>.length

/-- Checks if a digit meets the criteria for being faulty -/
def isFaultyCandidate (count : DigitCount) (d : Nat) : Bool :=
  isValidDigit d ∧ count d ≥ 5

/-- The main theorem -/
theorem faulty_key_theorem (attempted : DigitSequence) (registered : DigitSequence) :
  attempted.length = 10 →
  registered.length = 7 →
  (∃ d, isFaultyCandidate (countDigits attempted) d) →
  (∃ d, d ∈ [7, 9] ∧ isFaultyCandidate (countDigits attempted) d) :=
by sorry

end faulty_key_theorem_l3662_366282


namespace expression_evaluation_l3662_366289

theorem expression_evaluation : 3^4 - 4 * 3^3 + 6 * 3^2 - 4 * 3 + 1 = 16 := by
  sorry

end expression_evaluation_l3662_366289


namespace f_increasing_on_neg_reals_l3662_366250

/-- The function f(x) = -x^2 + 2x is monotonically increasing on (-∞, 0) -/
theorem f_increasing_on_neg_reals (x y : ℝ) :
  x < y → x < 0 → y < 0 → (-x^2 + 2*x) < (-y^2 + 2*y) := by
  sorry

end f_increasing_on_neg_reals_l3662_366250


namespace remainder_four_eleven_mod_five_l3662_366271

theorem remainder_four_eleven_mod_five : 4^11 % 5 = 4 := by
  sorry

end remainder_four_eleven_mod_five_l3662_366271


namespace geometric_sequence_common_ratio_l3662_366288

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)) 
  (h_a3 : a 3 = 2) 
  (h_a7 : a 7 = 32) : 
  (a 2 / a 1 = 2) ∨ (a 2 / a 1 = -2) :=
sorry

end geometric_sequence_common_ratio_l3662_366288


namespace largest_n_binomial_sum_l3662_366255

theorem largest_n_binomial_sum (n : ℕ) : 
  (Nat.choose 12 5 + Nat.choose 12 6 = Nat.choose 13 n) → n ≤ 7 :=
by sorry

end largest_n_binomial_sum_l3662_366255


namespace factorization_identities_l3662_366276

theorem factorization_identities (x y : ℝ) : 
  (x^3 + 6*x^2 + 9*x = x*(x + 3)^2) ∧ 
  (16*x^2 - 9*y^2 = (4*x - 3*y)*(4*x + 3*y)) ∧ 
  ((3*x+y)^2 - (x-3*y)*(3*x+y) = 2*(3*x+y)*(x+2*y)) := by
  sorry

end factorization_identities_l3662_366276


namespace expected_remaining_people_l3662_366253

/-- The expected number of people remaining in a line of 100 people after a removal process -/
theorem expected_remaining_people (n : Nat) (h : n = 100) :
  let people := n
  let facing_right := n / 2
  let facing_left := n - facing_right
  let expected_remaining := (2^n : ℝ) / (Nat.choose n facing_right) - 1
  expected_remaining = (2^100 : ℝ) / (Nat.choose 100 50) - 1 := by
  sorry

end expected_remaining_people_l3662_366253


namespace absolute_value_plus_exponent_l3662_366241

theorem absolute_value_plus_exponent : |(-2 : ℝ)| + (π - 3)^(0 : ℝ) = 3 := by sorry

end absolute_value_plus_exponent_l3662_366241


namespace prob_diff_color_is_29_50_l3662_366258

-- Define the contents of the boxes
def boxA : Finset (Fin 3) := {0, 0, 1, 1, 2}
def boxB : Finset (Fin 3) := {0, 0, 0, 0, 1, 1, 1, 2, 2}

-- Define the probability of drawing a ball of a different color
def prob_diff_color : ℚ :=
  let total_A := boxA.card
  let total_B := boxB.card + 1
  let prob_white := (boxA.filter (· = 0)).card / total_A *
                    (boxB.filter (· ≠ 0)).card / total_B
  let prob_red := (boxA.filter (· = 1)).card / total_A *
                  (boxB.filter (· ≠ 1)).card / total_B
  let prob_black := (boxA.filter (· = 2)).card / total_A *
                    (boxB.filter (· ≠ 2)).card / total_B
  prob_white + prob_red + prob_black

-- Theorem statement
theorem prob_diff_color_is_29_50 : prob_diff_color = 29 / 50 := by
  sorry

end prob_diff_color_is_29_50_l3662_366258


namespace sufficient_not_necessary_l3662_366254

theorem sufficient_not_necessary :
  (∀ a b : ℝ, a > 2 ∧ b > 1 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 2 ∧ b > 1)) :=
by sorry

end sufficient_not_necessary_l3662_366254


namespace polynomial_roots_l3662_366270

def polynomial (x : ℝ) : ℝ :=
  x^6 - 2*x^5 - 9*x^4 + 14*x^3 + 24*x^2 - 20*x - 20

def has_zero_sum_pairs (p : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ b ∧ p a = 0 ∧ p (-a) = 0 ∧ p b = 0 ∧ p (-b) = 0

theorem polynomial_roots : 
  has_zero_sum_pairs polynomial →
  (∀ x : ℝ, polynomial x = 0 ↔ 
    x = Real.sqrt 2 ∨ x = -Real.sqrt 2 ∨
    x = Real.sqrt 5 ∨ x = -Real.sqrt 5 ∨
    x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3) :=
by sorry

end polynomial_roots_l3662_366270


namespace calculate_expression_solve_equation_l3662_366239

-- Problem 1
theorem calculate_expression : 2 * (-3)^2 - 4 * (-3) - 15 = 15 := by sorry

-- Problem 2
theorem solve_equation :
  ∀ x : ℚ, (4 - 2*x) / 3 - x = 1 → x = 1/5 := by sorry

end calculate_expression_solve_equation_l3662_366239


namespace parallel_planes_line_parallel_l3662_366219

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relations
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Define the "not contained in" relation
variable (line_not_in_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_line_parallel 
  (α β : Plane) (m : Line) 
  (h1 : plane_parallel α β) 
  (h2 : line_parallel_plane m α) 
  (h3 : line_not_in_plane m β) : 
  line_parallel_plane m β :=
sorry

end parallel_planes_line_parallel_l3662_366219


namespace slope_intercept_product_l3662_366297

theorem slope_intercept_product (m b : ℚ) 
  (h1 : m = 3/4)
  (h2 : b = -5/3)
  (h3 : m > 0)
  (h4 : b < 0) : 
  m * b < -1 := by
sorry

end slope_intercept_product_l3662_366297


namespace ball_arrangement_theorem_l3662_366257

-- Define the number of balls and boxes
def n : ℕ := 4

-- Define the function for the number of arrangements with each box containing one ball
def arrangements_full (n : ℕ) : ℕ := n.factorial

-- Define the function for the number of arrangements with exactly one box empty
def arrangements_one_empty (n : ℕ) : ℕ := n.choose 2 * (n - 1).factorial

-- State the theorem
theorem ball_arrangement_theorem :
  arrangements_full n = 24 ∧ arrangements_one_empty n = 144 := by
  sorry


end ball_arrangement_theorem_l3662_366257


namespace quadratic_no_real_roots_l3662_366217

theorem quadratic_no_real_roots (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 6 * x + 1 ≠ 0) → a > 9 := by
  sorry

end quadratic_no_real_roots_l3662_366217


namespace fiftieth_term_of_sequence_l3662_366291

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem fiftieth_term_of_sequence : arithmetic_sequence 2 4 50 = 198 := by
  sorry

end fiftieth_term_of_sequence_l3662_366291


namespace product_real_imag_parts_l3662_366213

theorem product_real_imag_parts : ∃ (z : ℂ), 
  z = (2 + 3*Complex.I) / (1 + Complex.I) ∧ 
  (z.re * z.im = 5/4) := by sorry

end product_real_imag_parts_l3662_366213


namespace inscribed_triangle_with_parallel_sides_l3662_366272

/-- A line in the plane -/
structure Line where
  -- Add necessary fields for a line

/-- A circle in the plane -/
structure Circle where
  -- Add necessary fields for a circle

/-- A point in the plane -/
structure Point where
  -- Add necessary fields for a point

/-- A triangle in the plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Check if a point lies on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop :=
  sorry

/-- Check if a line is parallel to a side of a triangle -/
def line_parallel_to_side (l : Line) (t : Triangle) : Prop :=
  sorry

/-- Main theorem: Given three pairwise non-parallel lines and a circle,
    there exists a triangle inscribed in the circle with sides parallel to the given lines -/
theorem inscribed_triangle_with_parallel_sides
  (l1 l2 l3 : Line) (c : Circle)
  (h1 : ¬ are_parallel l1 l2)
  (h2 : ¬ are_parallel l2 l3)
  (h3 : ¬ are_parallel l3 l1) :
  ∃ (t : Triangle),
    point_on_circle t.A c ∧
    point_on_circle t.B c ∧
    point_on_circle t.C c ∧
    line_parallel_to_side l1 t ∧
    line_parallel_to_side l2 t ∧
    line_parallel_to_side l3 t :=
  sorry

end inscribed_triangle_with_parallel_sides_l3662_366272


namespace cost_of_balls_and_shuttlecocks_l3662_366216

/-- The cost of ping-pong balls and badminton shuttlecocks -/
theorem cost_of_balls_and_shuttlecocks 
  (ping_pong : ℝ) 
  (shuttlecock : ℝ) 
  (h1 : 3 * ping_pong + 2 * shuttlecock = 15.5)
  (h2 : 2 * ping_pong + 3 * shuttlecock = 17) :
  4 * ping_pong + 4 * shuttlecock = 26 :=
by sorry

end cost_of_balls_and_shuttlecocks_l3662_366216


namespace inequality_solution_set_l3662_366283

def solution_set : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem inequality_solution_set : 
  ∀ x : ℝ, x ∈ solution_set ↔ x^2 - 5*x + 6 ≤ 0 := by sorry

end inequality_solution_set_l3662_366283


namespace largest_two_digit_prime_factor_of_binomial_l3662_366245

theorem largest_two_digit_prime_factor_of_binomial :
  ∃ (p : ℕ), p = 73 ∧ 
    Prime p ∧ 
    10 ≤ p ∧ p < 100 ∧
    p ∣ (Nat.choose 150 75) ∧
    ∀ q : ℕ, Prime q → 10 ≤ q → q < 100 → q ∣ (Nat.choose 150 75) → q ≤ p :=
by sorry


end largest_two_digit_prime_factor_of_binomial_l3662_366245


namespace tp_supply_duration_l3662_366251

/-- Represents the toilet paper usage of a family member --/
structure TPUsage where
  weekdayTimes : ℕ
  weekdaySquares : ℕ
  weekendTimes : ℕ
  weekendSquares : ℕ

/-- Calculates the total squares used per week for a family member --/
def weeklyUsage (usage : TPUsage) : ℕ :=
  5 * usage.weekdayTimes * usage.weekdaySquares +
  2 * usage.weekendTimes * usage.weekendSquares

/-- Represents the family's toilet paper situation --/
structure TPFamily where
  bill : TPUsage
  wife : TPUsage
  kid : TPUsage
  kidCount : ℕ
  rollCount : ℕ
  squaresPerRoll : ℕ

/-- Calculates the total squares used per week for the entire family --/
def familyWeeklyUsage (family : TPFamily) : ℕ :=
  weeklyUsage family.bill +
  weeklyUsage family.wife +
  family.kidCount * weeklyUsage family.kid

/-- Calculates how many days the toilet paper supply will last --/
def supplyDuration (family : TPFamily) : ℕ :=
  let totalSquares := family.rollCount * family.squaresPerRoll
  let weeksSupply := totalSquares / familyWeeklyUsage family
  7 * weeksSupply

/-- The main theorem stating how long the toilet paper supply will last --/
theorem tp_supply_duration : 
  let family : TPFamily := {
    bill := { weekdayTimes := 3, weekdaySquares := 5, weekendTimes := 4, weekendSquares := 6 },
    wife := { weekdayTimes := 4, weekdaySquares := 8, weekendTimes := 5, weekendSquares := 10 },
    kid := { weekdayTimes := 5, weekdaySquares := 6, weekendTimes := 6, weekendSquares := 5 },
    kidCount := 2,
    rollCount := 1000,
    squaresPerRoll := 300
  }
  ∃ (d : ℕ), d ≥ 2615 ∧ d ≤ 2616 ∧ supplyDuration family = d :=
by sorry


end tp_supply_duration_l3662_366251


namespace inverse_proportion_k_condition_l3662_366285

/-- Theorem: For an inverse proportion function y = (k-1)/x, given two points
    A(x₁, y₁) and B(x₂, y₂) on its graph where 0 < x₁ < x₂ and y₁ < y₂, 
    the value of k must be less than 1. -/
theorem inverse_proportion_k_condition 
  (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : y₁ < y₂)
  (h4 : y₁ = (k - 1) / x₁) (h5 : y₂ = (k - 1) / x₂) : 
  k < 1 := by
  sorry

end inverse_proportion_k_condition_l3662_366285


namespace cloth_selling_price_l3662_366226

/-- Calculates the total selling price of cloth given the quantity, profit per metre, and cost price per metre. -/
def total_selling_price (quantity : ℕ) (profit_per_metre : ℕ) (cost_price_per_metre : ℕ) : ℕ :=
  quantity * (cost_price_per_metre + profit_per_metre)

/-- Proves that the total selling price of 30 meters of cloth with a profit of Rs. 10 per metre
    and a cost price of Rs. 140 per metre is Rs. 4500. -/
theorem cloth_selling_price :
  total_selling_price 30 10 140 = 4500 := by
  sorry

end cloth_selling_price_l3662_366226


namespace integer_solutions_quadratic_l3662_366237

theorem integer_solutions_quadratic (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ x y : ℤ, x^2 + p*x + q^4 = 0 ∧ y^2 + p*y + q^4 = 0 ∧ x ≠ y) ↔ 
  (p = 17 ∧ q = 2) :=
sorry

end integer_solutions_quadratic_l3662_366237


namespace equation_real_solutions_l3662_366238

theorem equation_real_solutions :
  let f : ℝ → ℝ := λ x => 5*x/(x^2 + 2*x + 4) + 7*x/(x^2 - 7*x + 4) + 5/3
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, f x = 0 ∧ ∀ x, f x = 0 → x ∈ s :=
by sorry

end equation_real_solutions_l3662_366238


namespace f_derivative_at_2_l3662_366234

-- Define the function f
def f (x : ℝ) (k : ℝ) : ℝ := x^3 - k*x^2 + 3*x - 5

-- State the theorem
theorem f_derivative_at_2 : 
  ∃ k : ℝ, (∀ x, deriv (f · k) x = 3*x^2 - 2*k*x + 3) ∧ deriv (f · k) 2 = k ∧ k = 3 := by
  sorry

end f_derivative_at_2_l3662_366234


namespace largest_number_l3662_366261

theorem largest_number (S : Finset ℕ) (h : S = {5, 8, 4, 3, 2}) : 
  Finset.max' S (by simp [h]) = 8 := by
sorry

end largest_number_l3662_366261


namespace work_completion_time_l3662_366224

/-- Given that A can do a work in 15 days and when A and B work together for 4 days
    they complete 0.4666666666666667 of the work, prove that B can do the work alone in 20 days. -/
theorem work_completion_time (a b : ℝ) (ha : a = 1 / 15) 
    (h_together : 4 * (a + 1 / b) = 0.4666666666666667) : b = 20 := by
  sorry

end work_completion_time_l3662_366224


namespace game_draw_probability_l3662_366267

/-- In a game between two players, given the probabilities of not losing and losing for each player, 
    we can calculate the probability of a draw. -/
theorem game_draw_probability (p_not_losing p_losing : ℚ) : 
  p_not_losing = 3/4 → p_losing = 1/2 → p_not_losing - p_losing = 1/4 := by
  sorry

end game_draw_probability_l3662_366267


namespace equation_holds_l3662_366281

theorem equation_holds : (5 - 2) + 6 - (4 - 3) = 8 := by
  sorry

end equation_holds_l3662_366281


namespace fraction_calculation_l3662_366284

theorem fraction_calculation : (900^2 : ℝ) / (264^2 - 256^2) = 194.711 := by
  sorry

end fraction_calculation_l3662_366284


namespace unique_solution_system_l3662_366299

/-- The system of equations:
    1. 2(x-1) - 3(y+1) = 12
    2. x/2 + y/3 = 1
    has a unique solution (x, y) = (4, -3) -/
theorem unique_solution_system :
  ∃! (x y : ℝ), (2*(x-1) - 3*(y+1) = 12) ∧ (x/2 + y/3 = 1) ∧ x = 4 ∧ y = -3 := by
  sorry

end unique_solution_system_l3662_366299


namespace triangle_existence_l3662_366207

/-- A triangle with semiperimeter s and two excircle radii r_a and r_b exists if and only if s^2 > r_a * r_b -/
theorem triangle_existence (s r_a r_b : ℝ) (h_s : s > 0) (h_ra : r_a > 0) (h_rb : r_b > 0) :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 * s ∧
    ∃ (r_ea r_eb : ℝ), r_ea = r_a ∧ r_eb = r_b ∧
    r_ea = s * (b + c - a) / (b + c) ∧
    r_eb = s * (a + c - b) / (a + c)) ↔
  s^2 > r_a * r_b :=
sorry

end triangle_existence_l3662_366207


namespace intersection_point_l3662_366231

def line1 (x y : ℚ) : Prop := 12 * x - 5 * y = 8
def line2 (x y : ℚ) : Prop := 10 * x + 2 * y = 20

theorem intersection_point :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (58/37, 667/370) := by
  sorry

end intersection_point_l3662_366231


namespace ratio_equivalence_l3662_366295

theorem ratio_equivalence (x : ℚ) : 
  (12 : ℚ) / 8 = 6 / (x * 60) → x = 1 / 15 := by
  sorry

end ratio_equivalence_l3662_366295


namespace double_sized_cube_weight_l3662_366230

/-- Given a cubical block of metal, this function calculates the weight of another cube of the same metal with sides twice as long. -/
def weight_of_double_sized_cube (original_weight : ℝ) : ℝ :=
  8 * original_weight

/-- Theorem stating that if a cubical block of metal weighs 3 pounds, then another cube of the same metal with sides twice as long will weigh 24 pounds. -/
theorem double_sized_cube_weight :
  weight_of_double_sized_cube 3 = 24 := by
  sorry

#eval weight_of_double_sized_cube 3

end double_sized_cube_weight_l3662_366230


namespace prob_product_not_zero_l3662_366264

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The probability of getting a number other than 1 on a single die -/
def probNotOne : ℚ := (numSides - 1) / numSides

/-- The number of dice tossed -/
def numDice : ℕ := 3

/-- The probability that (a-1)(b-1)(c-1) ≠ 0 when tossing three eight-sided dice -/
theorem prob_product_not_zero : 
  (probNotOne ^ numDice : ℚ) = 343 / 512 := by sorry

end prob_product_not_zero_l3662_366264


namespace solutions_equation1_solutions_equation2_l3662_366225

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 5*x + 6 = 0
def equation2 (x : ℝ) : Prop := (x + 2)*(x - 1) = x + 2

-- Theorem for equation1
theorem solutions_equation1 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ ∧ equation1 x₂ ∧ x₁ = 3 ∧ x₂ = 2 :=
sorry

-- Theorem for equation2
theorem solutions_equation2 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation2 x₁ ∧ equation2 x₂ ∧ x₁ = -2 ∧ x₂ = 2 :=
sorry

end solutions_equation1_solutions_equation2_l3662_366225


namespace cubic_equation_unique_solution_l3662_366260

theorem cubic_equation_unique_solution :
  ∀ (a : ℝ), ∃! (x : ℝ), x^3 - 2*a*x^2 + 3*a*x + a^2 - 2 = 0 := by
  sorry

end cubic_equation_unique_solution_l3662_366260


namespace cubic_foot_to_cubic_inches_l3662_366233

/-- Proves that 1 cubic foot equals 1728 cubic inches, given that 1 foot equals 12 inches. -/
theorem cubic_foot_to_cubic_inches :
  (1 : ℝ) * (1 : ℝ) * (1 : ℝ) = 1728 * (1 / 12 : ℝ) * (1 / 12 : ℝ) * (1 / 12 : ℝ) :=
by
  sorry

#check cubic_foot_to_cubic_inches

end cubic_foot_to_cubic_inches_l3662_366233


namespace carl_winning_configurations_l3662_366242

def board_size : ℕ := 4

def winning_configurations : ℕ := 10

def remaining_cells_after_win : ℕ := 13

def ways_to_choose_three_from_twelve : ℕ := 220

theorem carl_winning_configurations :
  (winning_configurations * board_size * remaining_cells_after_win * ways_to_choose_three_from_twelve) = 114400 :=
by sorry

end carl_winning_configurations_l3662_366242


namespace initial_boxes_count_l3662_366200

theorem initial_boxes_count (ali_boxes_per_circle ernie_boxes_per_circle ali_circles ernie_circles : ℕ) 
  (h1 : ali_boxes_per_circle = 8)
  (h2 : ernie_boxes_per_circle = 10)
  (h3 : ali_circles = 5)
  (h4 : ernie_circles = 4) :
  ali_boxes_per_circle * ali_circles + ernie_boxes_per_circle * ernie_circles = 80 :=
by sorry

end initial_boxes_count_l3662_366200


namespace inverse_B_cubed_l3662_366262

theorem inverse_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![3, 7; -2, -5]) : 
  (B^3)⁻¹ = !![13, 0; -42, -95] := by
  sorry

end inverse_B_cubed_l3662_366262


namespace prob_same_flavor_is_one_fourth_l3662_366218

/-- The number of flavors available -/
def num_flavors : ℕ := 4

/-- The probability of selecting two bags of biscuits with the same flavor -/
def prob_same_flavor : ℚ := 1 / 4

/-- Theorem: The probability of selecting two bags of biscuits with the same flavor
    out of four possible flavors is 1/4 -/
theorem prob_same_flavor_is_one_fourth :
  prob_same_flavor = 1 / 4 := by sorry

end prob_same_flavor_is_one_fourth_l3662_366218


namespace students_present_l3662_366269

theorem students_present (total : ℕ) (absent_fraction : ℚ) (present : ℕ) : 
  total = 28 → 
  absent_fraction = 2/7 → 
  present = total - (total * absent_fraction).floor → 
  present = 20 := by
sorry

end students_present_l3662_366269

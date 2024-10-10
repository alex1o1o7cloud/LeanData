import Mathlib

namespace fruit_water_content_l152_15228

theorem fruit_water_content (m : ℝ) : 
  m > 0 ∧ m ≤ 100 →  -- m is a percentage, so it's between 0 and 100
  (100 - m + m * (1 - (m - 5) / 100) = 50) →  -- equation from step 6 in the solution
  m = 80 := by sorry

end fruit_water_content_l152_15228


namespace sine_sum_constant_l152_15242

theorem sine_sum_constant (α : Real) :
  (Real.sin α) ^ 2 + (Real.sin (α + 60 * π / 180)) ^ 2 + (Real.sin (α + 120 * π / 180)) ^ 2 =
  (Real.sin (α - 60 * π / 180)) ^ 2 + (Real.sin α) ^ 2 + (Real.sin (α + 60 * π / 180)) ^ 2 :=
by sorry

end sine_sum_constant_l152_15242


namespace joyce_final_egg_count_l152_15240

/-- Calculates the final number of eggs Joyce has after a series of transactions -/
def final_egg_count (initial_eggs : ℝ) (received_eggs : ℝ) (traded_eggs : ℝ) (given_away_eggs : ℝ) : ℝ :=
  initial_eggs + received_eggs - traded_eggs - given_away_eggs

/-- Proves that Joyce ends up with 9 eggs given the initial conditions and transactions -/
theorem joyce_final_egg_count :
  final_egg_count 8 3.5 0.5 2 = 9 := by sorry

end joyce_final_egg_count_l152_15240


namespace quadratic_one_zero_properties_l152_15250

/-- A quadratic function with exactly one zero -/
structure QuadraticWithOneZero where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : ∃! x, x^2 + a*x + b = 0

theorem quadratic_one_zero_properties (f : QuadraticWithOneZero) :
  (f.a^2 - f.b^2 ≤ 4) ∧
  (f.a^2 + 1/f.b ≥ 4) ∧
  (∀ c x₁ x₂, (∀ x, x^2 + f.a*x + f.b < c ↔ x₁ < x ∧ x < x₂) → |x₁ - x₂| = 4 → c = 4) :=
by sorry

end quadratic_one_zero_properties_l152_15250


namespace wheel_distance_theorem_l152_15238

/-- Represents the properties and movement of a wheel -/
structure Wheel where
  rotations_per_minute : ℕ
  cm_per_rotation : ℕ

/-- Calculates the distance in meters that a wheel moves in one hour -/
def distance_in_one_hour (w : Wheel) : ℚ :=
  (w.rotations_per_minute * 60 * w.cm_per_rotation) / 100

/-- Theorem stating that a wheel with given properties moves 420 meters in one hour -/
theorem wheel_distance_theorem (w : Wheel) 
  (h1 : w.rotations_per_minute = 20) 
  (h2 : w.cm_per_rotation = 35) : 
  distance_in_one_hour w = 420 := by
  sorry

#eval distance_in_one_hour ⟨20, 35⟩

end wheel_distance_theorem_l152_15238


namespace at_least_three_positive_and_negative_l152_15290

theorem at_least_three_positive_and_negative (a : Fin 12 → ℝ) 
  (h : ∀ i : Fin 11, a (i + 1) * (a i - a (i + 1) + a (i + 2)) < 0) :
  (∃ i j k : Fin 12, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 0 < a i ∧ 0 < a j ∧ 0 < a k) ∧
  (∃ i j k : Fin 12, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i < 0 ∧ a j < 0 ∧ a k < 0) := by
  sorry

end at_least_three_positive_and_negative_l152_15290


namespace vacation_cost_l152_15231

theorem vacation_cost (C : ℝ) : 
  (C / 3 - C / 5 = 50) → C = 375 := by
sorry

end vacation_cost_l152_15231


namespace record_storage_cost_l152_15278

/-- A record storage problem -/
theorem record_storage_cost (box_length box_width box_height : ℝ)
  (total_volume : ℝ) (cost_per_box : ℝ) :
  box_length = 15 →
  box_width = 12 →
  box_height = 10 →
  total_volume = 1080000 →
  cost_per_box = 0.2 →
  (total_volume / (box_length * box_width * box_height)) * cost_per_box = 120 := by
  sorry

end record_storage_cost_l152_15278


namespace infinitely_many_primes_4k_minus_1_l152_15267

theorem infinitely_many_primes_4k_minus_1 : 
  ∃ (S : Set Nat), (∀ n ∈ S, Nat.Prime n ∧ ∃ k, n = 4*k - 1) ∧ Set.Infinite S :=
sorry

end infinitely_many_primes_4k_minus_1_l152_15267


namespace problem_statement_l152_15253

theorem problem_statement (x y : ℝ) (h1 : x + y = 17) (h2 : x * y = 17) :
  (x^2 - 17*x) * (y + 17/y) = -289 := by
  sorry

end problem_statement_l152_15253


namespace range_of_a_range_of_f_when_a_is_2_l152_15211

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- Part 1: Range of a when f(x) ≥ 0 for all x ∈ ℝ
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ 0) → a ∈ Set.Icc (-2) 2 :=
sorry

-- Part 2: Range of f(x) when a = 2 and x ∈ [0, 3]
theorem range_of_f_when_a_is_2 : 
  Set.image (f 2) (Set.Icc 0 3) = Set.Icc 0 4 :=
sorry

end range_of_a_range_of_f_when_a_is_2_l152_15211


namespace rosie_pies_l152_15286

/-- Calculates the number of pies Rosie can make given the available apples and pears. -/
def calculate_pies (apples_per_3_pies : ℕ) (pears_per_3_pies : ℕ) (available_apples : ℕ) (available_pears : ℕ) : ℕ :=
  min (available_apples * 3 / apples_per_3_pies) (available_pears * 3 / pears_per_3_pies)

/-- Proves that Rosie can make 9 pies with 36 apples and 18 pears, given that she can make 3 pies out of 12 apples and 6 pears. -/
theorem rosie_pies : calculate_pies 12 6 36 18 = 9 := by
  sorry

end rosie_pies_l152_15286


namespace missing_number_in_mean_l152_15271

theorem missing_number_in_mean (known_numbers : List ℝ) (mean : ℝ) : 
  known_numbers = [1, 22, 23, 24, 26, 27, 2] ∧ 
  mean = 20 ∧ 
  (List.sum known_numbers + 35) / 8 = mean →
  35 = 8 * mean - List.sum known_numbers :=
by
  sorry

#check missing_number_in_mean

end missing_number_in_mean_l152_15271


namespace log_inequality_solution_set_l152_15214

theorem log_inequality_solution_set :
  ∀ x : ℝ, (Real.log (x - 1) < 1) ↔ (1 < x ∧ x < 11) :=
sorry

end log_inequality_solution_set_l152_15214


namespace initial_group_size_l152_15284

theorem initial_group_size (total_groups : Nat) (students_left : Nat) (remaining_students : Nat) :
  total_groups = 3 →
  students_left = 2 →
  remaining_students = 22 →
  ∃ initial_group_size : Nat, 
    initial_group_size * total_groups - students_left = remaining_students ∧
    initial_group_size = 8 := by
  sorry

end initial_group_size_l152_15284


namespace team_loss_percentage_l152_15247

theorem team_loss_percentage
  (win_loss_ratio : ℚ)
  (total_games : ℕ)
  (h1 : win_loss_ratio = 8 / 5)
  (h2 : total_games = 52) :
  (loss_percentage : ℚ) →
  loss_percentage = 38 / 100 :=
by sorry

end team_loss_percentage_l152_15247


namespace divisible_by_512_l152_15205

theorem divisible_by_512 (n : ℤ) (h : Odd n) :
  ∃ k : ℤ, n^12 - n^8 - n^4 + 1 = 512 * k := by sorry

end divisible_by_512_l152_15205


namespace circle_path_in_right_triangle_l152_15288

theorem circle_path_in_right_triangle : 
  ∀ (a b c : ℝ) (r : ℝ),
    a = 6 ∧ b = 8 ∧ c = 10 →  -- Triangle side lengths
    r = 1 →                   -- Circle radius
    a^2 + b^2 = c^2 →         -- Right triangle condition
    (a + b + c) - 6*r = 12 := by  -- Path length
  sorry

end circle_path_in_right_triangle_l152_15288


namespace inequality_proof_l152_15235

theorem inequality_proof (n : ℕ) : (2*n + 1)^n ≥ (2*n)^n + (2*n - 1)^n := by
  sorry

end inequality_proof_l152_15235


namespace viggo_payment_l152_15283

/-- Represents the denomination of the other bills used by Viggo --/
def other_denomination : ℕ := sorry

/-- The total amount spent on the shirt --/
def total_spent : ℕ := 80

/-- The number of other denomination bills used --/
def num_other_bills : ℕ := 2

/-- The denomination of the $20 bills --/
def twenty_bill : ℕ := 20

/-- The number of $20 bills used --/
def num_twenty_bills : ℕ := num_other_bills + 1

theorem viggo_payment :
  (num_twenty_bills * twenty_bill) + (num_other_bills * other_denomination) = total_spent ∧
  other_denomination = 10 := by sorry

end viggo_payment_l152_15283


namespace angle_A_value_max_value_angle_B_at_max_l152_15201

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Side lengths
variable (S : ℝ) -- Area of the triangle

-- Define the conditions
axiom triangle_condition : a^2 = b^2 + c^2 + Real.sqrt 3 * a * b
axiom side_a_value : a = Real.sqrt 3

-- Define the theorems to be proved
theorem angle_A_value : A = 5 * Real.pi / 6 :=
sorry

theorem max_value : 
  ∃ (max : ℝ), ∀ (B C : ℝ), S + 3 * Real.cos B * Real.cos C ≤ max ∧ 
  ∃ (B₀ C₀ : ℝ), S + 3 * Real.cos B₀ * Real.cos C₀ = max ∧ max = 3 :=
sorry

theorem angle_B_at_max : 
  ∃ (B₀ C₀ : ℝ), S + 3 * Real.cos B₀ * Real.cos C₀ = 3 ∧ B₀ = Real.pi / 12 :=
sorry

end

end angle_A_value_max_value_angle_B_at_max_l152_15201


namespace abs_value_equality_l152_15251

theorem abs_value_equality (m : ℝ) : |m| = |-3| → m = 3 ∨ m = -3 := by
  sorry

end abs_value_equality_l152_15251


namespace correct_answer_is_ten_l152_15248

theorem correct_answer_is_ten (x : ℝ) (h : 3 * x = 90) : x / 3 = 10 := by
  sorry

end correct_answer_is_ten_l152_15248


namespace laptop_price_l152_15298

theorem laptop_price : ∃ (x : ℝ), x = 400 ∧ 
  (∃ (price_C price_D : ℝ), 
    price_C = 0.8 * x - 60 ∧ 
    price_D = 0.7 * x ∧ 
    price_D - price_C = 20) := by
  sorry

end laptop_price_l152_15298


namespace max_value_inequality_max_value_achieved_max_value_is_five_l152_15227

theorem max_value_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) → a ≤ 5 :=
by
  sorry

theorem max_value_achieved : 
  ∃ x : ℝ, x^2 + |2*x - 6| = 5 :=
by
  sorry

theorem max_value_is_five : 
  (∀ x : ℝ, x^2 + |2*x - 6| ≥ 5) ∧ 
  (∃ x : ℝ, x^2 + |2*x - 6| = 5) :=
by
  sorry

end max_value_inequality_max_value_achieved_max_value_is_five_l152_15227


namespace expression_evaluation_l152_15223

theorem expression_evaluation : 3^(1^(2^3)) + ((3^1)^2)^2 = 84 := by
  sorry

end expression_evaluation_l152_15223


namespace three_digit_cube_units_digit_l152_15202

theorem three_digit_cube_units_digit :
  ∀ n : ℕ, 
    (100 ≤ n ∧ n < 1000) ∧ 
    (n = (n % 10)^3) →
    (n = 125 ∨ n = 216 ∨ n = 729) :=
by
  sorry

end three_digit_cube_units_digit_l152_15202


namespace distance_sum_theorem_l152_15226

/-- The curve C in the xy-plane -/
def C (x y : ℝ) : Prop := x^2/9 + y^2 = 1

/-- The line l in the xy-plane -/
def l (x y : ℝ) : Prop := y - x = Real.sqrt 2

/-- The point P -/
def P : ℝ × ℝ := (0, 2)

/-- Points A and B are the intersection points of C and l -/
def intersectionPoints (A B : ℝ × ℝ) : Prop :=
  C A.1 A.2 ∧ l A.1 A.2 ∧ C B.1 B.2 ∧ l B.1 B.2 ∧ A ≠ B

theorem distance_sum_theorem (A B : ℝ × ℝ) (h : intersectionPoints A B) :
  Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) +
  Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) =
  18 * Real.sqrt 2 / 5 := by
  sorry

end distance_sum_theorem_l152_15226


namespace vector_perpendicular_l152_15245

/-- Given plane vectors a and b, prove that (a - b) is perpendicular to b -/
theorem vector_perpendicular (a b : ℝ × ℝ) (ha : a = (2, 0)) (hb : b = (1, 1)) :
  (a - b) • b = 0 := by
  sorry

end vector_perpendicular_l152_15245


namespace lemon_ratio_l152_15243

def lemon_problem (levi jayden eli ian : ℕ) : Prop :=
  levi = 5 ∧
  jayden = levi + 6 ∧
  jayden = eli / 3 ∧
  levi + jayden + eli + ian = 115 ∧
  eli * 2 = ian

theorem lemon_ratio :
  ∀ levi jayden eli ian : ℕ,
    lemon_problem levi jayden eli ian →
    eli * 2 = ian :=
by
  sorry

end lemon_ratio_l152_15243


namespace digits_after_decimal_point_of_fraction_l152_15294

/-- The number of digits to the right of the decimal point when 5^8 / (10^6 * 16) is expressed as a decimal is 3. -/
theorem digits_after_decimal_point_of_fraction : ∃ (n : ℕ) (d : ℕ+), 
  5^8 / (10^6 * 16) = n / d ∧ 
  (∃ (k : ℕ), 10^3 * (n / d) = k ∧ 10^2 * (n / d) < 1) :=
by sorry

end digits_after_decimal_point_of_fraction_l152_15294


namespace estimate_total_students_l152_15257

/-- Represents the survey data and estimated total students -/
structure SurveyData where
  total_students : ℕ  -- Estimated total number of first-year students
  first_survey : ℕ    -- Number of students in the first survey
  second_survey : ℕ   -- Number of students in the second survey
  overlap : ℕ         -- Number of students in both surveys

/-- The theorem states that given the survey conditions, 
    the estimated total number of first-year students is 400 -/
theorem estimate_total_students (data : SurveyData) :
  data.first_survey = 80 →
  data.second_survey = 100 →
  data.overlap = 20 →
  data.total_students = 400 :=
by sorry

end estimate_total_students_l152_15257


namespace cube_cube_squared_power_calculation_l152_15277

theorem cube_cube_squared (a b : ℕ) : (a^3 * b^3)^2 = (a * b)^6 := by sorry

theorem power_calculation : (3^3 * 4^3)^2 = 2985984 := by sorry

end cube_cube_squared_power_calculation_l152_15277


namespace inequality_equivalence_l152_15256

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem inequality_equivalence (x : ℝ) (hx : x > 0) :
  (lg x ^ 2 - 3 * lg x + 3) / (lg x - 1) < 1 ↔ x < 10 :=
by sorry

end inequality_equivalence_l152_15256


namespace g_composition_of_three_l152_15218

def g (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_composition_of_three : g (g (g (g 3))) = 24 := by
  sorry

end g_composition_of_three_l152_15218


namespace jacket_sale_price_l152_15282

/-- Proves that the price of each jacket after noon was $18.95 given the sale conditions --/
theorem jacket_sale_price (total_jackets : ℕ) (price_before_noon : ℚ) (total_receipts : ℚ) (jackets_sold_after_noon : ℕ) :
  total_jackets = 214 →
  price_before_noon = 31.95 →
  total_receipts = 5108.30 →
  jackets_sold_after_noon = 133 →
  (total_receipts - (total_jackets - jackets_sold_after_noon : ℚ) * price_before_noon) / jackets_sold_after_noon = 18.95 := by
  sorry

end jacket_sale_price_l152_15282


namespace perpendicular_line_through_point_l152_15293

/-- Given a line L1 with equation x - y - 2 = 0 and a point A (2, 6),
    prove that the line L2 with equation x + y - 8 = 0 passes through A
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point 
  (L1 : Set (ℝ × ℝ)) 
  (A : ℝ × ℝ) :
  let L2 := {(x, y) : ℝ × ℝ | x + y - 8 = 0}
  (∀ (x y : ℝ), (x, y) ∈ L1 ↔ x - y - 2 = 0) →
  A = (2, 6) →
  A ∈ L2 ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁, y₁) ∈ L1 → (x₂, y₂) ∈ L1 → x₁ ≠ x₂ →
    (x₁ - x₂) * (2 - 2) + (y₁ - y₂) * (6 - 6) = 0) := by
sorry

end perpendicular_line_through_point_l152_15293


namespace nested_fraction_evaluation_l152_15260

theorem nested_fraction_evaluation :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 := by
  sorry

end nested_fraction_evaluation_l152_15260


namespace factorization_equality_l152_15234

theorem factorization_equality (a b : ℝ) : a^2 - 4*a*b^2 = a*(a - 4*b^2) := by
  sorry

end factorization_equality_l152_15234


namespace concert_ticket_revenue_l152_15246

/-- Calculates the total revenue from concert ticket sales given specific discount conditions -/
theorem concert_ticket_revenue : 
  let ticket_price : ℝ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let first_discount : ℝ := 0.4
  let second_discount : ℝ := 0.15
  let total_attendees : ℕ := 48

  let first_group_revenue := first_group_size * (ticket_price * (1 - first_discount))
  let second_group_revenue := second_group_size * (ticket_price * (1 - second_discount))
  let remaining_attendees := total_attendees - first_group_size - second_group_size
  let full_price_revenue := remaining_attendees * ticket_price

  first_group_revenue + second_group_revenue + full_price_revenue = 820 :=
by
  sorry


end concert_ticket_revenue_l152_15246


namespace lecture_scheduling_l152_15292

-- Define the number of lecturers
def n : ℕ := 7

-- Theorem statement
theorem lecture_scheduling (n : ℕ) (h : n = 7) : 
  (n! : ℕ) / 2 = 2520 :=
sorry

end lecture_scheduling_l152_15292


namespace polynomial_factorization_l152_15273

theorem polynomial_factorization (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3) * (8*x^2 + x - 3) := by
  sorry

end polynomial_factorization_l152_15273


namespace time_per_regular_letter_l152_15213

-- Define the given conditions
def days_between_letters : ℕ := 3
def minutes_per_page_regular : ℕ := 10
def minutes_per_page_long : ℕ := 20
def total_minutes_long_letter : ℕ := 80
def total_pages_per_month : ℕ := 24
def days_in_month : ℕ := 30

-- Define the theorem
theorem time_per_regular_letter :
  let pages_long_letter := total_minutes_long_letter / minutes_per_page_long
  let pages_regular_letters := total_pages_per_month - pages_long_letter
  let total_minutes_regular_letters := pages_regular_letters * minutes_per_page_regular
  let num_regular_letters := days_in_month / days_between_letters
  total_minutes_regular_letters / num_regular_letters = 20 := by
  sorry

end time_per_regular_letter_l152_15213


namespace novelists_count_l152_15266

theorem novelists_count (total : ℕ) (ratio_novelists : ℕ) (ratio_poets : ℕ) (novelists : ℕ) : 
  total = 24 →
  ratio_novelists = 5 →
  ratio_poets = 3 →
  ratio_novelists + ratio_poets = novelists + (total - novelists) →
  novelists * (ratio_novelists + ratio_poets) = total * ratio_novelists →
  novelists = 15 := by
sorry

end novelists_count_l152_15266


namespace min_value_of_f_l152_15209

/-- The function f(x) = 2x³ - 6x² + m -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f m x ≥ f m y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f m x = 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f m x ≤ f m y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f m x = -37) :=
by sorry

end min_value_of_f_l152_15209


namespace original_group_size_l152_15285

/-- Proves that the original number of men in a group is 12, given the conditions of the problem -/
theorem original_group_size (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) :
  initial_days = 8 →
  absent_men = 3 →
  final_days = 10 →
  ∃ (original_men : ℕ),
    original_men > 0 ∧
    (original_men : ℚ) / initial_days = (original_men - absent_men : ℚ) / final_days ∧
    original_men = 12 :=
by sorry

end original_group_size_l152_15285


namespace health_drink_sales_correct_l152_15299

/-- Represents the health drink inventory and sales data -/
structure HealthDrinkSales where
  first_batch_cost : ℝ
  second_batch_cost : ℝ
  unit_price_increase : ℝ
  selling_price : ℝ
  discounted_quantity : ℕ
  discount_rate : ℝ

/-- Calculates the quantity of the first batch and the total profit -/
def calculate_quantity_and_profit (sales : HealthDrinkSales) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correctness of the calculation -/
theorem health_drink_sales_correct (sales : HealthDrinkSales) 
  (h1 : sales.first_batch_cost = 40000)
  (h2 : sales.second_batch_cost = 88000)
  (h3 : sales.unit_price_increase = 2)
  (h4 : sales.selling_price = 28)
  (h5 : sales.discounted_quantity = 100)
  (h6 : sales.discount_rate = 0.2) :
  let (quantity, profit) := calculate_quantity_and_profit sales
  quantity = 2000 ∧ profit = 39440 :=
sorry

end health_drink_sales_correct_l152_15299


namespace correct_num_technicians_l152_15224

/-- The number of technicians in a workshop -/
def num_technicians : ℕ := 5

/-- The total number of workers in the workshop -/
def total_workers : ℕ := 15

/-- The average salary of all workers -/
def avg_salary_all : ℕ := 700

/-- The average salary of technicians -/
def avg_salary_technicians : ℕ := 800

/-- The average salary of non-technicians -/
def avg_salary_others : ℕ := 650

/-- Theorem stating that the number of technicians is correct given the conditions -/
theorem correct_num_technicians :
  num_technicians = 5 ∧
  num_technicians ≤ total_workers ∧
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_others) / total_workers = avg_salary_all :=
by sorry

end correct_num_technicians_l152_15224


namespace probability_largest_is_six_correct_l152_15220

def probability_largest_is_six (n m k : ℕ) : ℚ :=
  (Nat.choose m k : ℚ) / (Nat.choose n k : ℚ)

theorem probability_largest_is_six_correct : 
  probability_largest_is_six 10 6 4 = (Nat.choose 6 4 : ℚ) / (Nat.choose 10 4 : ℚ) :=
by
  sorry

#eval probability_largest_is_six 10 6 4

end probability_largest_is_six_correct_l152_15220


namespace marching_band_composition_l152_15229

theorem marching_band_composition (total : ℕ) (brass : ℕ) (woodwind : ℕ) (percussion : ℕ)
  (h1 : total = 110)
  (h2 : woodwind = 2 * brass)
  (h3 : percussion = 4 * woodwind)
  (h4 : total = brass + woodwind + percussion) :
  brass = 10 := by
sorry

end marching_band_composition_l152_15229


namespace bisecting_centers_form_line_l152_15206

/-- Two non-overlapping circles in a plane -/
structure TwoCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  R₁ : ℝ
  R₂ : ℝ
  h_positive : R₁ > 0 ∧ R₂ > 0
  h_non_overlapping : Real.sqrt ((O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2) > R₁ + R₂

/-- A point that is the center of a circle bisecting both given circles -/
def BisectingCenter (tc : TwoCircles) (X : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    r^2 = (X.1 - tc.O₁.1)^2 + (X.2 - tc.O₁.2)^2 + tc.R₁^2 ∧
    r^2 = (X.1 - tc.O₂.1)^2 + (X.2 - tc.O₂.2)^2 + tc.R₂^2

/-- The locus of bisecting centers forms a straight line -/
theorem bisecting_centers_form_line (tc : TwoCircles) :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧
    (∀ X : ℝ × ℝ, BisectingCenter tc X ↔ a * X.1 + b * X.2 + c = 0) ∧
    (a * (tc.O₂.1 - tc.O₁.1) + b * (tc.O₂.2 - tc.O₁.2) = 0) :=
sorry

end bisecting_centers_form_line_l152_15206


namespace fish_catch_calculation_l152_15236

/-- Prove that given the conditions, Erica caught 80 kg of fish in the past four months --/
theorem fish_catch_calculation (price : ℝ) (total_earnings : ℝ) (past_catch : ℝ) :
  price = 20 →
  total_earnings = 4800 →
  total_earnings = price * (past_catch + 2 * past_catch) →
  past_catch = 80 := by
  sorry

end fish_catch_calculation_l152_15236


namespace fish_per_bowl_l152_15225

theorem fish_per_bowl (total_bowls : ℕ) (total_fish : ℕ) (h1 : total_bowls = 261) (h2 : total_fish = 6003) :
  total_fish / total_bowls = 23 :=
by sorry

end fish_per_bowl_l152_15225


namespace no_zeroes_g_l152_15239

/-- A function f satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  continuous : Continuous f
  differentiable : Differentiable ℝ f
  condition : ∀ x, x * (deriv f x) + f x > 0

/-- The function g(x) = xf(x) + 1 -/
def g (sf : SpecialFunction) (x : ℝ) : ℝ := x * sf.f x + 1

/-- Theorem stating that g has no zeroes for x > 0 -/
theorem no_zeroes_g (sf : SpecialFunction) : ∀ x > 0, g sf x ≠ 0 := by
  sorry

end no_zeroes_g_l152_15239


namespace geometric_progression_fourth_term_l152_15252

theorem geometric_progression_fourth_term 
  (a : ℝ → ℝ) -- Sequence of real numbers
  (h1 : a 1 = 2^(1/2)) -- First term
  (h2 : a 2 = 2^(1/3)) -- Second term
  (h3 : a 3 = 2^(1/6)) -- Third term
  (h_geom : ∀ n : ℕ, n > 0 → a (n + 1) / a n = a 2 / a 1) -- Geometric progression condition
  : a 4 = 1 := by
sorry

end geometric_progression_fourth_term_l152_15252


namespace center_square_side_length_l152_15200

theorem center_square_side_length :
  ∀ (total_side : ℝ) (l_region_count : ℕ) (l_region_fraction : ℝ),
    total_side = 20 →
    l_region_count = 4 →
    l_region_fraction = 1/5 →
    let total_area := total_side^2
    let l_regions_area := l_region_count * l_region_fraction * total_area
    let center_area := total_area - l_regions_area
    center_area.sqrt = 4 * Real.sqrt 5 :=
by sorry

end center_square_side_length_l152_15200


namespace inventory_problem_l152_15272

theorem inventory_problem (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) : 
  speedsters = (3 * total) / 4 →
  convertibles = (3 * speedsters) / 5 →
  convertibles = 54 →
  total - speedsters = 30 :=
by
  sorry

end inventory_problem_l152_15272


namespace election_winning_percentage_bound_l152_15269

def total_votes_sept30 : ℕ := 15000
def total_votes_oct10 : ℕ := 22000
def geoff_votes_sept30 : ℕ := 150
def additional_votes_needed_sept30 : ℕ := 5000
def additional_votes_needed_oct10 : ℕ := 2000

def winning_percentage : ℚ :=
  (geoff_votes_sept30 + additional_votes_needed_sept30 + additional_votes_needed_oct10) / total_votes_oct10

theorem election_winning_percentage_bound :
  winning_percentage < 325/1000 := by sorry

end election_winning_percentage_bound_l152_15269


namespace remainder_1949_1995_mod_7_l152_15237

theorem remainder_1949_1995_mod_7 : 1949^1995 % 7 = 6 := by
  sorry

end remainder_1949_1995_mod_7_l152_15237


namespace articles_with_equal_price_l152_15255

/-- Represents the cost price of a single article -/
def cost_price : ℝ := sorry

/-- Represents the selling price of a single article -/
def selling_price : ℝ := sorry

/-- The number of articles whose selling price equals the cost price of 50 articles -/
def N : ℝ := sorry

/-- The gain percentage -/
def gain_percent : ℝ := 100

theorem articles_with_equal_price :
  (50 * cost_price = N * selling_price) →
  (selling_price = 2 * cost_price) →
  (N = 25) :=
by sorry

end articles_with_equal_price_l152_15255


namespace fourth_month_sale_l152_15276

def sale_month1 : ℕ := 5420
def sale_month2 : ℕ := 5660
def sale_month3 : ℕ := 6200
def sale_month5 : ℕ := 6500
def sale_month6 : ℕ := 6470
def average_sale : ℕ := 6100
def num_months : ℕ := 6

theorem fourth_month_sale :
  sale_month1 + sale_month2 + sale_month3 + sale_month5 + sale_month6 + 6350 = average_sale * num_months :=
by sorry

end fourth_month_sale_l152_15276


namespace f_less_than_neg_two_f_two_zeros_iff_l152_15203

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - exp (x - a) + a

theorem f_less_than_neg_two (x : ℝ) (h : x > 0) : f 0 x < -2 := by
  sorry

theorem f_two_zeros_iff (a : ℝ) :
  (∃ x y, x ≠ y ∧ x > 0 ∧ y > 0 ∧ f a x = 0 ∧ f a y = 0) ↔ a > 1 := by
  sorry

end f_less_than_neg_two_f_two_zeros_iff_l152_15203


namespace leg_length_in_45_45_90_triangle_l152_15268

/-- Represents a 45-45-90 triangle -/
structure RightIsoscelesTriangle where
  side : ℝ
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = side * Real.sqrt 2

/-- The length of a leg in a 45-45-90 triangle with hypotenuse 9 is 9 -/
theorem leg_length_in_45_45_90_triangle (t : RightIsoscelesTriangle) 
  (h : t.hypotenuse = 9) : t.side = 9 := by
  sorry

end leg_length_in_45_45_90_triangle_l152_15268


namespace container_capacity_l152_15291

theorem container_capacity (container_capacity : ℝ) 
  (h1 : 8 = 0.2 * container_capacity) 
  (num_containers : ℕ := 40) : 
  num_containers * container_capacity = 1600 := by
sorry

end container_capacity_l152_15291


namespace max_cart_length_l152_15258

/-- The maximum length of a rectangular cart that can navigate through a right-angled corridor -/
theorem max_cart_length (corridor_width : ℝ) (cart_width : ℝ) :
  corridor_width = 1.5 →
  cart_width = 1 →
  ∃ (max_length : ℝ), max_length = 3 * Real.sqrt 2 - 2 ∧
    ∀ (cart_length : ℝ), cart_length ≤ max_length →
      ∃ (θ : ℝ), 0 < θ ∧ θ < Real.pi / 2 ∧
        cart_length ≤ (3 * (Real.sin θ + Real.cos θ) - 2) / (2 * Real.sin θ * Real.cos θ) :=
by sorry

end max_cart_length_l152_15258


namespace smallest_music_class_size_l152_15263

theorem smallest_music_class_size :
  ∀ (x : ℕ),
  (∃ (total : ℕ), total = 5 * x + 2 ∧ total > 40) →
  (∀ (y : ℕ), y < x → ¬(∃ (total : ℕ), total = 5 * y + 2 ∧ total > 40)) →
  5 * x + 2 = 42 :=
by sorry

end smallest_music_class_size_l152_15263


namespace solution_verification_l152_15244

theorem solution_verification (x : ℝ) : 
  (3 * x^2 = 27 → x = 3 ∨ x = -3) ∧ 
  (2 * x^2 + x = 55 → x = 5 ∨ x = -5.5) ∧ 
  (2 * x^2 + 18 = 15 * x → x = 6 ∨ x = 1.5) := by
sorry

end solution_verification_l152_15244


namespace jerry_cans_count_l152_15259

/-- The number of cans Jerry can carry at once -/
def cans_per_trip : ℕ := 4

/-- The time in seconds it takes to drain 4 cans -/
def drain_time : ℕ := 30

/-- The time in seconds for a round trip to the sink/recycling bin -/
def round_trip_time : ℕ := 20

/-- The total time in seconds to throw all cans away -/
def total_time : ℕ := 350

/-- The time in seconds for one complete cycle (draining and round trip) -/
def cycle_time : ℕ := drain_time + round_trip_time

theorem jerry_cans_count : 
  (total_time / cycle_time) * cans_per_trip = 28 := by sorry

end jerry_cans_count_l152_15259


namespace tangent_line_m_range_l152_15215

/-- The range of m for a line mx - y - 5m + 4 = 0 tangent to a circle (x+1)^2 + y^2 = 4 -/
theorem tangent_line_m_range :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), (x + 1)^2 + y^2 = 4 ∧ m*x - y - 5*m + 4 = 0) →
  (∃ (Q : ℝ × ℝ), (Q.1 + 1)^2 + Q.2^2 = 4 ∧ 
    ∃ (P : ℝ × ℝ), m*P.1 - P.2 - 5*m + 4 = 0 ∧
    Real.cos (30 * π / 180) = (Q.1 - P.1) / (4 * ((Q.1 - P.1)^2 + (Q.2 - P.2)^2).sqrt)) →
  0 ≤ m ∧ m ≤ 12/5 := by
sorry


end tangent_line_m_range_l152_15215


namespace double_acute_angle_l152_15281

theorem double_acute_angle (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  0 < 2 * θ ∧ 2 * θ < Real.pi := by
  sorry

end double_acute_angle_l152_15281


namespace frozen_yoghurt_cartons_l152_15261

/-- Represents the number of cartons of ice cream Caleb bought -/
def ice_cream_cartons : ℕ := 10

/-- Represents the cost of one carton of ice cream in dollars -/
def ice_cream_cost : ℕ := 4

/-- Represents the cost of one carton of frozen yoghurt in dollars -/
def yoghurt_cost : ℕ := 1

/-- Represents the difference in dollars between ice cream and frozen yoghurt spending -/
def spending_difference : ℕ := 36

/-- Theorem stating that the number of frozen yoghurt cartons Caleb bought is 4 -/
theorem frozen_yoghurt_cartons : ℕ := by
  sorry

end frozen_yoghurt_cartons_l152_15261


namespace joes_total_weight_l152_15295

/-- Proves that the total weight of Joe's two lifts is 1800 pounds given the conditions -/
theorem joes_total_weight (first_lift second_lift : ℕ) : 
  first_lift = 700 ∧ 
  2 * first_lift = second_lift + 300 → 
  first_lift + second_lift = 1800 := by
sorry

end joes_total_weight_l152_15295


namespace min_occupied_seats_for_150_l152_15216

/-- Given a row of seats, returns the minimum number of occupied seats required
    to ensure any additional person must sit next to someone -/
def minOccupiedSeats (totalSeats : ℕ) : ℕ :=
  totalSeats / 3

theorem min_occupied_seats_for_150 :
  minOccupiedSeats 150 = 50 := by
  sorry

#eval minOccupiedSeats 150

end min_occupied_seats_for_150_l152_15216


namespace hope_project_protractors_l152_15212

theorem hope_project_protractors :
  ∀ (x y z : ℕ),
  x > 31 →
  z > 33 →
  10 * x + 15 * y + 20 * z = 1710 →
  8 * x + 2 * y + 8 * z = 664 →
  6 * x + 7 * y + 10 * z = 870 :=
by
  sorry

end hope_project_protractors_l152_15212


namespace square_area_not_tripled_when_side_tripled_l152_15233

theorem square_area_not_tripled_when_side_tripled (s : ℝ) (h : s > 0) :
  (3 * s)^2 ≠ 3 * s^2 := by sorry

end square_area_not_tripled_when_side_tripled_l152_15233


namespace solve_for_y_l152_15289

theorem solve_for_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 := by
  sorry

end solve_for_y_l152_15289


namespace same_height_time_l152_15265

/-- Represents the height of a ball as a function of time -/
def ball_height (a h : ℝ) (t : ℝ) : ℝ := a * (t - 1.2)^2 + h

theorem same_height_time : 
  ∀ (a h : ℝ), a ≠ 0 →
  ∃ (t : ℝ), t > 0 ∧ 
  ball_height a h t = ball_height a h (t - 2) ∧
  t = 2.2 :=
sorry

end same_height_time_l152_15265


namespace negation_existential_equivalence_l152_15232

theorem negation_existential_equivalence (f : ℝ → ℝ) :
  (¬ ∃ x₀ : ℝ, f x₀ < 0) ↔ (∀ x : ℝ, f x ≥ 0) := by
  sorry

end negation_existential_equivalence_l152_15232


namespace tangent_line_perpendicular_l152_15204

/-- Given a curve y = e^(ax), prove that if its tangent line at (0,1) is perpendicular to the line x + 2y + 1 = 0, then a = 2. -/
theorem tangent_line_perpendicular (a : ℝ) : 
  (∀ x, deriv (fun x => Real.exp (a * x)) x = a * Real.exp (a * x)) →
  (fun x => Real.exp (a * x)) 0 = 1 →
  (deriv (fun x => Real.exp (a * x))) 0 = (-1 / (2 : ℝ))⁻¹ →
  a = 2 := by
  sorry

end tangent_line_perpendicular_l152_15204


namespace equilateral_triangle_coverage_l152_15254

theorem equilateral_triangle_coverage (small_side : ℝ) (large_side : ℝ) : 
  small_side = 1 →
  large_side = 15 →
  (large_side / small_side) ^ 2 = 225 :=
by
  sorry

#check equilateral_triangle_coverage

end equilateral_triangle_coverage_l152_15254


namespace spinner_probability_l152_15296

theorem spinner_probability : 
  let spinner_sections : ℕ := 4
  let e_section : ℕ := 1
  let spins : ℕ := 2
  let prob_not_e_single : ℚ := (spinner_sections - e_section) / spinner_sections
  (prob_not_e_single ^ spins) = 9 / 16 := by
  sorry

end spinner_probability_l152_15296


namespace parallelogram_base_length_l152_15274

theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 576) 
  (h2 : height = 18) : 
  area / height = 32 := by
sorry

end parallelogram_base_length_l152_15274


namespace baseball_tickets_sold_l152_15221

theorem baseball_tickets_sold (fair_tickets : ℕ) (baseball_tickets : ℕ) : 
  fair_tickets = 25 → 
  fair_tickets = 2 * baseball_tickets + 6 → 
  baseball_tickets = 9 := by
sorry

end baseball_tickets_sold_l152_15221


namespace machining_defect_probability_l152_15275

theorem machining_defect_probability (defect_rate1 defect_rate2 : ℝ) 
  (h1 : defect_rate1 = 0.03) 
  (h2 : defect_rate2 = 0.05) 
  (h3 : 0 ≤ defect_rate1 ∧ defect_rate1 ≤ 1) 
  (h4 : 0 ≤ defect_rate2 ∧ defect_rate2 ≤ 1) :
  1 - (1 - defect_rate1) * (1 - defect_rate2) = 0.0785 := by
  sorry

#check machining_defect_probability

end machining_defect_probability_l152_15275


namespace jack_money_per_can_l152_15262

def bottles_recycled : ℕ := 80
def cans_recycled : ℕ := 140
def total_money : ℚ := 15
def money_per_bottle : ℚ := 1/10

theorem jack_money_per_can :
  (total_money - (bottles_recycled : ℚ) * money_per_bottle) / (cans_recycled : ℚ) = 5/100 := by
  sorry

end jack_money_per_can_l152_15262


namespace polynomial_sum_simplification_l152_15280

/-- Given two polynomials over ℝ, prove their sum equals a specific polynomial -/
theorem polynomial_sum_simplification (x : ℝ) :
  (3 * x^4 - 2 * x^3 + 5 * x^2 - 8 * x + 10) + 
  (7 * x^5 - 3 * x^4 + x^3 - 7 * x^2 + 2 * x - 2) = 
  7 * x^5 - x^3 - 2 * x^2 - 6 * x + 8 := by
  sorry

end polynomial_sum_simplification_l152_15280


namespace element_in_set_l152_15264

theorem element_in_set : ∀ (a b : ℕ), 1 ∈ ({a, b, 1} : Set ℕ) := by
  sorry

end element_in_set_l152_15264


namespace inequality_solution_set_l152_15230

theorem inequality_solution_set (x : ℝ) :
  (x - 1) * (2 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end inequality_solution_set_l152_15230


namespace cuboid_to_cube_surface_area_l152_15297

/-- Given a cuboid with a square base, if reducing its height by 4 cm results in a cube
    and decreases its volume by 64 cubic centimeters, then the surface area of the
    resulting cube is 96 square centimeters. -/
theorem cuboid_to_cube_surface_area (l w h : ℝ) : 
  l = w → -- The base is square
  (l * w * h) - (l * w * (h - 4)) = 64 → -- Volume decrease
  l * w * 4 = 64 → -- Volume decrease equals base area times height reduction
  6 * (l * l) = 96 := by
  sorry

end cuboid_to_cube_surface_area_l152_15297


namespace matt_trading_profit_l152_15249

/-- Represents the profit made from trading baseball cards -/
def tradingProfit (initialCardCount : ℕ) (initialCardValue : ℕ) 
                  (tradedCardCount : ℕ) (receivedCardValues : List ℕ) : ℤ :=
  let initialValue := initialCardCount * initialCardValue
  let tradedValue := tradedCardCount * initialCardValue
  let receivedValue := receivedCardValues.sum
  (receivedValue : ℤ) - (tradedValue : ℤ)

/-- Theorem stating that Matt's trading profit is $3 -/
theorem matt_trading_profit :
  tradingProfit 8 6 2 [2, 2, 2, 9] = 3 := by
  sorry

#eval tradingProfit 8 6 2 [2, 2, 2, 9]

end matt_trading_profit_l152_15249


namespace congruence_problem_l152_15241

theorem congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 4 ∧ n ≡ -1458 [ZMOD 5] ∧ n = 2 := by
  sorry

end congruence_problem_l152_15241


namespace paper_airplane_class_composition_l152_15208

theorem paper_airplane_class_composition 
  (total_students : ℕ) 
  (total_airplanes : ℕ) 
  (girls_airplanes : ℕ) 
  (boys_airplanes : ℕ) 
  (h1 : total_students = 21)
  (h2 : total_airplanes = 69)
  (h3 : girls_airplanes = 2)
  (h4 : boys_airplanes = 5) :
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    boys * boys_airplanes + girls * girls_airplanes = total_airplanes ∧
    boys = 9 ∧ 
    girls = 12 := by
  sorry

end paper_airplane_class_composition_l152_15208


namespace parallelogram_diagonal_intersection_l152_15210

/-- The intersection point of the diagonals of a parallelogram with opposite vertices (2, -3) and (10, 9) is (6, 3). -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (10, 9)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (6, 3) := by
sorry


end parallelogram_diagonal_intersection_l152_15210


namespace train_length_l152_15279

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 12 → ∃ length : ℝ, 
  (length ≥ 399) ∧ (length ≤ 401) ∧ (length = speed * time * 1000 / 3600) := by
  sorry

#check train_length

end train_length_l152_15279


namespace complement_M_intersect_N_l152_15222

-- Define the set M
def M : Set ℝ := {x : ℝ | x ≤ Real.sqrt 5}

-- Define the set N
def N : Set ℝ := {1, 2, 3, 4}

-- Theorem statement
theorem complement_M_intersect_N :
  (Set.compl M) ∩ N = {3, 4} := by sorry

end complement_M_intersect_N_l152_15222


namespace test_probabilities_l152_15270

theorem test_probabilities (p_A p_B p_C : ℝ) 
  (h_A : p_A = 0.8) (h_B : p_B = 0.6) (h_C : p_C = 0.5) : 
  p_A * p_B * p_C = 0.24 ∧ 
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 0.96 := by
  sorry

end test_probabilities_l152_15270


namespace fraction_simplification_specific_case_l152_15207

theorem fraction_simplification (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

theorem specific_case : 
  let a : ℚ := 12
  let b : ℚ := 16
  let c : ℚ := 9
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = 37 := by
  sorry

end fraction_simplification_specific_case_l152_15207


namespace function_properties_l152_15217

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - 4*x^3 + a*x^2 - 1

-- Define the function g
def g (b : ℝ) (x : ℝ) : ℝ := b*x^2 - 1

-- Theorem statement
theorem function_properties :
  ∃ (a : ℝ),
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f a x < f a y) ∧
    (∀ x y : ℝ, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f a x > f a y) ∧
    a = 4 ∧
    ∃ b : ℝ, (b = 0 ∨ b = 4) ∧
      (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = g b x₁ ∧ f a x₂ = g b x₂) ∧
      (∀ x₃ : ℝ, x₃ ≠ x₁ ∧ x₃ ≠ x₂ → f a x₃ ≠ g b x₃) :=
by sorry

end function_properties_l152_15217


namespace bank_layoff_optimization_l152_15219

/-- Represents the problem of maximizing bank profit through layoffs --/
theorem bank_layoff_optimization :
  let initial_employees : ℕ := 320
  let initial_profit_per_employee : ℝ := 200000
  let profit_increase_per_layoff : ℝ := 20000
  let layoff_expense : ℝ := 60000
  let min_employees : ℕ := (3 * initial_employees) / 4
  let profit (x : ℕ) : ℝ := 
    (initial_employees - x) * (initial_profit_per_employee + profit_increase_per_layoff * x) - layoff_expense * x
  ∃ (optimal_layoffs : ℕ), 
    optimal_layoffs = 80 ∧ 
    optimal_layoffs ≤ initial_employees - min_employees ∧
    ∀ (x : ℕ), x ≤ initial_employees - min_employees → profit x ≤ profit optimal_layoffs :=
by sorry

end bank_layoff_optimization_l152_15219


namespace point_in_planar_region_l152_15287

/-- A point (m, 1) is within the planar region represented by 2x + 3y - 5 > 0 if and only if m > 1 -/
theorem point_in_planar_region (m : ℝ) : 2*m + 3*1 - 5 > 0 ↔ m > 1 := by
  sorry

end point_in_planar_region_l152_15287

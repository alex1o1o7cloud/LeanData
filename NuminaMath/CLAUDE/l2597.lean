import Mathlib

namespace train_distance_l2597_259749

/-- The distance traveled by a train in a given time, given its speed -/
def distance_traveled (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

/-- Convert hours to minutes -/
def hours_to_minutes (hours : ℚ) : ℚ :=
  hours * 60

theorem train_distance (train_speed : ℚ) (travel_time : ℚ) :
  train_speed = 2 / 2 →
  travel_time = 3 →
  distance_traveled train_speed (hours_to_minutes travel_time) = 180 := by
  sorry

end train_distance_l2597_259749


namespace reading_growth_rate_l2597_259722

theorem reading_growth_rate (initial_amount final_amount : ℝ) (growth_period : ℕ) (x : ℝ) :
  initial_amount = 1 →
  final_amount = 1.21 →
  growth_period = 2 →
  final_amount = initial_amount * (1 + x)^growth_period →
  100 * (1 + x)^2 = 121 :=
by sorry

end reading_growth_rate_l2597_259722


namespace always_two_distinct_roots_find_p_values_l2597_259733

-- Define the quadratic equation
def quadratic_equation (x p : ℝ) : ℝ := (x - 3) * (x - 2) - p^2

-- Part 1: Prove that the equation always has two distinct real roots
theorem always_two_distinct_roots (p : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ p = 0 ∧ quadratic_equation x₂ p = 0 := by
  sorry

-- Part 2: Find the values of p given the condition x₁ = 4x₂
theorem find_p_values :
  ∃ p : ℝ, ∃ x₁ x₂ : ℝ, 
    quadratic_equation x₁ p = 0 ∧ 
    quadratic_equation x₂ p = 0 ∧ 
    x₁ = 4 * x₂ ∧ 
    (p = Real.sqrt 2 ∨ p = -Real.sqrt 2) := by
  sorry

end always_two_distinct_roots_find_p_values_l2597_259733


namespace epipen_cost_l2597_259779

/-- Proves that the cost of each EpiPen is $500, given the specified conditions -/
theorem epipen_cost (epipen_per_year : ℕ) (insurance_coverage : ℚ) (annual_payment : ℚ) :
  epipen_per_year = 2 ∧ insurance_coverage = 3/4 ∧ annual_payment = 250 →
  ∃ (cost : ℚ), cost = 500 ∧ epipen_per_year * (1 - insurance_coverage) * cost = annual_payment :=
by sorry

end epipen_cost_l2597_259779


namespace smallest_four_digit_multiple_of_6_with_digit_sum_12_l2597_259738

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem smallest_four_digit_multiple_of_6_with_digit_sum_12 :
  ∀ n : ℕ, is_four_digit n → n % 6 = 0 → digit_sum n = 12 → n ≥ 1020 :=
by sorry

end smallest_four_digit_multiple_of_6_with_digit_sum_12_l2597_259738


namespace least_five_digit_congruent_to_8_mod_17_l2597_259707

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n ≡ 8 [ZMOD 17] → n ≥ 10009 :=
by sorry

end least_five_digit_congruent_to_8_mod_17_l2597_259707


namespace z_in_fourth_quadrant_l2597_259772

def complex_equation (z : ℂ) : Prop := (1 + Complex.I) * z = 2 * Complex.I

theorem z_in_fourth_quadrant (z : ℂ) (h : complex_equation z) : 
  z.re > 0 ∧ z.im < 0 := by
  sorry

end z_in_fourth_quadrant_l2597_259772


namespace intersection_line_correct_l2597_259795

/-- Two circles in a 2D plane -/
structure TwoCircles where
  c1 : (ℝ × ℝ) → Prop
  c2 : (ℝ × ℝ) → Prop

/-- The equation of a line in 2D -/
structure Line where
  eq : (ℝ × ℝ) → Prop

/-- Given two intersecting circles, returns the line of their intersection -/
def intersectionLine (circles : TwoCircles) : Line :=
  { eq := fun (x, y) => x + 3 * y = 0 }

theorem intersection_line_correct (circles : TwoCircles) :
  circles.c1 = fun (x, y) => x^2 + y^2 = 10 →
  circles.c2 = fun (x, y) => (x - 1)^2 + (y - 3)^2 = 20 →
  ∃ (A B : ℝ × ℝ), circles.c1 A ∧ circles.c1 B ∧ circles.c2 A ∧ circles.c2 B →
  (intersectionLine circles).eq = fun (x, y) => x + 3 * y = 0 :=
by
  sorry

end intersection_line_correct_l2597_259795


namespace no_seventh_power_sum_l2597_259765

def a : ℕ → ℤ
  | 0 => 8
  | 1 => 20
  | (n + 2) => (a (n + 1))^2 + 12 * (a (n + 1)) * (a n) + (a (n + 1)) + 11 * (a n)

def seventh_power_sum_mod_29 (x y z : ℤ) : ℤ :=
  ((x^7 % 29) + (y^7 % 29) + (z^7 % 29)) % 29

theorem no_seventh_power_sum (n : ℕ) :
  ∀ x y z : ℤ, (a n) % 29 ≠ seventh_power_sum_mod_29 x y z :=
by sorry

end no_seventh_power_sum_l2597_259765


namespace exists_solution_l2597_259725

theorem exists_solution : ∃ (a b c d : ℕ+), 2014 = (a^2 + b^2) * (c^3 - d^3) := by
  sorry

end exists_solution_l2597_259725


namespace arithmetic_computation_l2597_259711

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 6 * 5 / 3 = 32 := by
  sorry

end arithmetic_computation_l2597_259711


namespace complex_modulus_example_l2597_259747

theorem complex_modulus_example : ∃ (z : ℂ), z = 4 + 3*I ∧ Complex.abs z = 5 := by
  sorry

end complex_modulus_example_l2597_259747


namespace equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solution_l2597_259702

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  (x + 1)^2 - 144 = 0 ↔ x = -13 ∨ x = 11 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  x^2 - 4*x - 32 = 0 ↔ x = 8 ∨ x = -4 := by sorry

-- Equation 3
theorem equation_three_solutions (x : ℝ) :
  3*(x - 2)^2 = x*(x - 2) ↔ x = 2 ∨ x = 3 := by sorry

-- Equation 4
theorem equation_four_solution (x : ℝ) :
  (x + 3)^2 = 2*x + 5 ↔ x = -2 := by sorry

end equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solution_l2597_259702


namespace cosine_value_on_unit_circle_l2597_259759

theorem cosine_value_on_unit_circle (α : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = 1/2 ∧ y = Real.sqrt 3 / 2) →
  Real.cos α = 1/2 :=
by sorry

end cosine_value_on_unit_circle_l2597_259759


namespace cubic_inequality_l2597_259781

theorem cubic_inequality (p q x : ℝ) : x^3 + p*x + q = 0 → 4*q*x ≤ p^2 := by
  sorry

end cubic_inequality_l2597_259781


namespace largest_multiple_proof_l2597_259714

/-- The largest three-digit number that is divisible by 6, 5, 8, and 9 -/
def largest_multiple : ℕ := 720

theorem largest_multiple_proof :
  (∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 6 ∣ n ∧ 5 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n → n ≤ largest_multiple) ∧
  100 ≤ largest_multiple ∧
  largest_multiple < 1000 ∧
  6 ∣ largest_multiple ∧
  5 ∣ largest_multiple ∧
  8 ∣ largest_multiple ∧
  9 ∣ largest_multiple :=
by sorry

end largest_multiple_proof_l2597_259714


namespace clock_angle_at_3_30_l2597_259793

/-- The angle of the hour hand at 3:30 -/
def hour_hand_angle : ℝ := 105

/-- The angle of the minute hand at 3:30 -/
def minute_hand_angle : ℝ := 180

/-- The total degrees in a circle -/
def total_degrees : ℝ := 360

/-- The larger angle between the hour and minute hands at 3:30 -/
def larger_angle : ℝ := total_degrees - (minute_hand_angle - hour_hand_angle)

theorem clock_angle_at_3_30 :
  larger_angle = 285 := by sorry

end clock_angle_at_3_30_l2597_259793


namespace mrs_lee_june_earnings_percent_l2597_259796

/-- Represents the Lee family's income situation -/
structure LeeIncome where
  may_total : ℝ
  may_mrs_lee : ℝ
  june_mrs_lee : ℝ

/-- Conditions for the Lee family's income -/
def lee_income_conditions (income : LeeIncome) : Prop :=
  income.may_mrs_lee = 0.5 * income.may_total ∧
  income.june_mrs_lee = 1.2 * income.may_mrs_lee

/-- Theorem: Mrs. Lee's earnings in June were 60% of the family's total income -/
theorem mrs_lee_june_earnings_percent (income : LeeIncome) 
  (h : lee_income_conditions income) : 
  income.june_mrs_lee / income.may_total = 0.6 := by
  sorry

end mrs_lee_june_earnings_percent_l2597_259796


namespace max_rectangles_in_5x5_grid_seven_rectangles_fit_5x5_grid_l2597_259721

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the square grid -/
def Grid := ℕ × ℕ

/-- Check if a list of rectangles fits in the grid without overlap and covers it completely -/
def fits_grid (grid : Grid) (rectangles : List Rectangle) : Prop :=
  sorry

/-- The theorem stating that 7 is the maximum number of rectangles that can fit in a 5x5 grid -/
theorem max_rectangles_in_5x5_grid :
  ∀ (rectangles : List Rectangle),
    (∀ r ∈ rectangles, (r.width = 1 ∧ r.height = 4) ∨ (r.width = 1 ∧ r.height = 3)) →
    fits_grid (5, 5) rectangles →
    rectangles.length ≤ 7 :=
  sorry

/-- The theorem stating that 7 rectangles can indeed fit in a 5x5 grid -/
theorem seven_rectangles_fit_5x5_grid :
  ∃ (rectangles : List Rectangle),
    (∀ r ∈ rectangles, (r.width = 1 ∧ r.height = 4) ∨ (r.width = 1 ∧ r.height = 3)) ∧
    fits_grid (5, 5) rectangles ∧
    rectangles.length = 7 :=
  sorry

end max_rectangles_in_5x5_grid_seven_rectangles_fit_5x5_grid_l2597_259721


namespace sqrt_27_div_sqrt_3_eq_3_l2597_259744

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end sqrt_27_div_sqrt_3_eq_3_l2597_259744


namespace pen_cost_is_30_l2597_259718

-- Define the daily expenditures
def daily_expenditures : List ℝ := [450, 600, 400, 500, 550, 300]

-- Define the mean expenditure
def mean_expenditure : ℝ := 500

-- Define the number of days
def num_days : ℕ := 7

-- Define the cost of the notebook
def notebook_cost : ℝ := 50

-- Define the cost of the earphone
def earphone_cost : ℝ := 620

-- Theorem to prove
theorem pen_cost_is_30 :
  let total_week_expenditure := mean_expenditure * num_days
  let total_other_days := daily_expenditures.sum
  let friday_expenditure := total_week_expenditure - total_other_days
  friday_expenditure - (notebook_cost + earphone_cost) = 30 := by
  sorry

end pen_cost_is_30_l2597_259718


namespace table_height_is_130_l2597_259792

/-- The height of the table in cm -/
def table_height : ℝ := 130

/-- The height of the bottle in cm -/
def bottle_height : ℝ := sorry

/-- The height of the can in cm -/
def can_height : ℝ := sorry

/-- The distance from the top of the can on the floor to the top of the bottle on the table is 150 cm -/
axiom condition1 : table_height + bottle_height = can_height + 150

/-- The distance from the top of the bottle on the floor to the top of the can on the table is 110 cm -/
axiom condition2 : table_height + can_height = bottle_height + 110

/-- Theorem: Given the conditions, the height of the table is 130 cm -/
theorem table_height_is_130 : table_height = 130 := by sorry

end table_height_is_130_l2597_259792


namespace box_volume_increase_l2597_259760

/-- Given a rectangular box with length l, width w, and height h, 
    if the volume is 3000, surface area is 1380, and sum of edges is 160,
    then increasing each dimension by 2 results in a volume of 4548 --/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 3000)
  (surface_area : 2 * (l * w + w * h + l * h) = 1380)
  (edge_sum : 4 * (l + w + h) = 160) :
  (l + 2) * (w + 2) * (h + 2) = 4548 := by
  sorry


end box_volume_increase_l2597_259760


namespace money_distribution_l2597_259788

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Ram's amount,
    prove that Krishan has the same amount as Gopal, which is Rs. 1785. -/
theorem money_distribution (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 735 →
  gopal = 1785 ∧ krishan = 1785 := by
sorry

end money_distribution_l2597_259788


namespace total_students_l2597_259763

theorem total_students (group_a group_b : ℕ) : 
  (group_a : ℚ) / group_b = 3 / 2 →
  (group_a : ℚ) * (1 / 10) - (group_b : ℚ) * (1 / 5) = 190 →
  group_b = 650 →
  group_a + group_b = 1625 := by
sorry


end total_students_l2597_259763


namespace arithmetic_sequence_a6_l2597_259774

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = a n * q

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
    (h_arithmetic : ArithmeticSequence a)
    (h_a1 : a 1 = 1)
    (h_a2a4 : a 2 * a 4 = 16) :
    a 6 = 32 := by
  sorry

end arithmetic_sequence_a6_l2597_259774


namespace old_selling_price_l2597_259777

/-- Given a product with an increased gross profit and new selling price, calculate the old selling price. -/
theorem old_selling_price (cost : ℝ) (new_selling_price : ℝ) : 
  (new_selling_price = cost * 1.15) →  -- New selling price is cost plus 15% profit
  (new_selling_price = 92) →           -- New selling price is $92.00
  (cost * 1.10 = 88) :=                -- Old selling price (cost plus 10% profit) is $88.00
by sorry

end old_selling_price_l2597_259777


namespace g_of_3_l2597_259767

def g (x : ℝ) : ℝ := 5 * x^3 - 6 * x^2 - 3 * x + 5

theorem g_of_3 : g 3 = 77 := by
  sorry

end g_of_3_l2597_259767


namespace dot_product_value_l2597_259746

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

def f (a b : n) (x : ℝ) : ℝ := ‖a + x • b‖

theorem dot_product_value (a b : n) 
  (ha : ‖a‖ = Real.sqrt 2) 
  (hb : ‖b‖ = Real.sqrt 2)
  (hmin : ∀ x : ℝ, f a b x ≥ 1)
  (hf : ∃ x : ℝ, f a b x = 1) :
  inner a b = Real.sqrt 2 ∨ inner a b = -Real.sqrt 2 := by
sorry

end dot_product_value_l2597_259746


namespace brodys_calculator_battery_life_l2597_259739

theorem brodys_calculator_battery_life :
  ∀ (total_battery : ℝ) 
    (used_battery : ℝ) 
    (exam_duration : ℝ) 
    (remaining_battery : ℝ),
  used_battery = (3/4) * total_battery →
  exam_duration = 2 →
  remaining_battery = 13 →
  total_battery = 60 := by
sorry

end brodys_calculator_battery_life_l2597_259739


namespace defendant_statement_implies_innocence_l2597_259778

-- Define the types of people on the island
inductive Person
| Knight
| Liar

-- Define the crime and accusation
def Crime : Type := Unit
def Accusation : Type := Unit

-- Define the statement made by the defendant
def DefendantStatement (criminal : Person) : Prop :=
  criminal = Person.Liar

-- Define the concept of telling the truth
def TellsTruth (p : Person) (statement : Prop) : Prop :=
  match p with
  | Person.Knight => statement
  | Person.Liar => ¬statement

-- Theorem: The defendant's statement implies innocence regardless of their type
theorem defendant_statement_implies_innocence 
  (defendant : Person) 
  (crime : Crime) 
  (accusation : Accusation) :
  TellsTruth defendant (DefendantStatement (Person.Liar)) → 
  defendant ≠ Person.Liar :=
sorry

end defendant_statement_implies_innocence_l2597_259778


namespace money_distribution_l2597_259726

/-- Given three people A, B, and C with some money, prove that A and C together have 300 Rs. -/
theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 400)
  (bc_sum : B + C = 150)
  (c_amount : C = 50) : 
  A + C = 300 := by sorry

end money_distribution_l2597_259726


namespace least_number_with_remainder_l2597_259741

theorem least_number_with_remainder (n : ℕ) : n = 256 →
  (∃ k : ℕ, n = 18 * k + 4) ∧
  (∀ m : ℕ, m < n → ¬(∃ j : ℕ, m = 18 * j + 4)) := by
  sorry

end least_number_with_remainder_l2597_259741


namespace sum_of_squares_equals_two_l2597_259768

/-- Given a 2x2 matrix B with specific properties, prove that the sum of squares of its elements is 2 -/
theorem sum_of_squares_equals_two (x y z w : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  (B.transpose = B⁻¹) →
  (x^2 + y^2 = 1) →
  (z^2 + w^2 = 1) →
  (y + z = 1/2) →
  (x^2 + y^2 + z^2 + w^2 = 2) :=
by sorry

end sum_of_squares_equals_two_l2597_259768


namespace students_exceed_pets_l2597_259755

/-- Proves that in 6 classrooms, where each classroom has 22 students, 3 pet rabbits, 
    and 1 pet hamster, the number of students exceeds the number of pets by 108. -/
theorem students_exceed_pets : 
  let num_classrooms : ℕ := 6
  let students_per_classroom : ℕ := 22
  let rabbits_per_classroom : ℕ := 3
  let hamsters_per_classroom : ℕ := 1
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_pets : ℕ := num_classrooms * (rabbits_per_classroom + hamsters_per_classroom)
  total_students - total_pets = 108 := by
  sorry

end students_exceed_pets_l2597_259755


namespace mike_total_games_l2597_259705

/-- The total number of basketball games Mike attended over two years -/
def total_games (games_this_year games_last_year : ℕ) : ℕ :=
  games_this_year + games_last_year

/-- Theorem stating that Mike attended 54 games in total -/
theorem mike_total_games : 
  total_games 15 39 = 54 := by
  sorry

end mike_total_games_l2597_259705


namespace power_of_seven_mod_nine_l2597_259776

theorem power_of_seven_mod_nine : 7^15 % 9 = 1 := by
  sorry

end power_of_seven_mod_nine_l2597_259776


namespace salary_increase_to_original_l2597_259770

/-- Proves that a 56.25% increase is required to regain the original salary after a 30% reduction and 10% bonus --/
theorem salary_increase_to_original (S : ℝ) (S_pos : S > 0) : 
  let reduced_salary := 0.7 * S
  let bonus := 0.1 * S
  let new_salary := reduced_salary + bonus
  (S - new_salary) / new_salary = 0.5625 := by sorry

end salary_increase_to_original_l2597_259770


namespace sum_reciprocals_zero_implies_sum_diff_zero_l2597_259715

theorem sum_reciprocals_zero_implies_sum_diff_zero 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h : 1 / (a + 1) + 1 / (a - 1) + 1 / (b + 1) + 1 / (b - 1) = 0) : 
  a - 1 / a + b - 1 / b = 0 := by
sorry

end sum_reciprocals_zero_implies_sum_diff_zero_l2597_259715


namespace subtracted_number_l2597_259704

theorem subtracted_number (a b : ℕ) (x : ℚ) 
  (h1 : a / b = 6 / 5)
  (h2 : (a - x) / (b - x) = 5 / 4)
  (h3 : a - b = 5) :
  x = 5 := by sorry

end subtracted_number_l2597_259704


namespace seonmi_money_problem_l2597_259794

theorem seonmi_money_problem (initial_money : ℕ) : 
  (initial_money / 2 / 3 / 2 = 250) → initial_money = 1500 := by
sorry

end seonmi_money_problem_l2597_259794


namespace floor_plus_self_eq_l2597_259706

theorem floor_plus_self_eq (r : ℝ) : ⌊r⌋ + r = 12.4 ↔ r = 6.4 := by
  sorry

end floor_plus_self_eq_l2597_259706


namespace min_boxes_to_eliminate_l2597_259723

/-- The total number of boxes -/
def total_boxes : ℕ := 30

/-- The number of boxes containing at least $200,000 -/
def high_value_boxes : ℕ := 6

/-- The minimum number of boxes that must be eliminated -/
def boxes_to_eliminate : ℕ := 18

/-- Theorem stating that eliminating 18 boxes is the minimum required for a 50% chance of a high-value box -/
theorem min_boxes_to_eliminate :
  boxes_to_eliminate = total_boxes - 2 * high_value_boxes :=
sorry

end min_boxes_to_eliminate_l2597_259723


namespace three_four_five_pythagorean_triple_l2597_259735

/-- A Pythagorean triple is a set of three positive integers a, b, and c that satisfy a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The set (3, 4, 5) is a Pythagorean triple -/
theorem three_four_five_pythagorean_triple : is_pythagorean_triple 3 4 5 := by
  sorry

end three_four_five_pythagorean_triple_l2597_259735


namespace force_for_18_inch_crowbar_l2597_259736

-- Define the inverse relationship between force and length
def inverse_relationship (force : ℝ) (length : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * length = k

-- Define the given condition
def given_condition : Prop :=
  inverse_relationship 200 12

-- Define the theorem to be proved
theorem force_for_18_inch_crowbar :
  given_condition →
  ∃ force : ℝ, inverse_relationship force 18 ∧ 
    (force ≥ 133.33 ∧ force ≤ 133.34) :=
by
  sorry

end force_for_18_inch_crowbar_l2597_259736


namespace range_of_a_l2597_259730

theorem range_of_a (p : ∀ x ∈ Set.Icc 1 4, x^2 ≥ a) 
                   (q : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) :
  a = 1 ∨ a ≤ -2 := by sorry

end range_of_a_l2597_259730


namespace percentage_of_fresh_peaches_l2597_259713

def total_peaches : ℕ := 250
def thrown_away : ℕ := 15
def peaches_left : ℕ := 135

def fresh_peaches : ℕ := total_peaches - (thrown_away + (total_peaches - peaches_left))

theorem percentage_of_fresh_peaches :
  (fresh_peaches : ℚ) / total_peaches * 100 = 48 := by sorry

end percentage_of_fresh_peaches_l2597_259713


namespace steak_knife_set_cost_is_80_l2597_259719

/-- Represents the cost of a steak knife set -/
def steak_knife_set_cost (knives_per_set : ℕ) (single_knife_cost : ℕ) : ℕ :=
  knives_per_set * single_knife_cost

/-- Proves that the cost of a steak knife set with 4 knives at $20 each is $80 -/
theorem steak_knife_set_cost_is_80 :
  steak_knife_set_cost 4 20 = 80 := by
  sorry

end steak_knife_set_cost_is_80_l2597_259719


namespace f_properties_l2597_259786

noncomputable def f (x : ℝ) : ℝ := x + Real.cos x

theorem f_properties :
  (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧
  (f (π / 2) = π / 2) := by
  sorry

end f_properties_l2597_259786


namespace polynomial_bound_l2597_259754

theorem polynomial_bound (n : ℕ) (p : ℝ → ℝ) :
  (∀ x, ∃ (c : ℝ) (k : ℕ), k ≤ 2*n ∧ p x = c * x^k) →
  (∀ k : ℤ, -n ≤ k ∧ k ≤ n → |p k| ≤ 1) →
  ∀ x : ℝ, -n ≤ x ∧ x ≤ n → |p x| ≤ 2^(2*n) :=
by sorry

end polynomial_bound_l2597_259754


namespace ellipse_properties_l2597_259766

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_major_axis : 2 * b = a
  h_rhombus_area : 4 * a * b = 8

/-- A line passing through a point on the ellipse -/
structure IntersectingLine (ε : Ellipse) where
  k : ℝ
  h_length : (4 * Real.sqrt 2) / 5 = 4 * Real.sqrt (1 + k^2) / (1 + 4 * k^2)

/-- A point on the perpendicular bisector of the chord -/
structure PerpendicularPoint (ε : Ellipse) (l : IntersectingLine ε) where
  y₀ : ℝ
  h_dot_product : 4 = (y₀^2 + ε.a^2) - (y₀^2 + (ε.a * (1 - k^2) / (1 + k^2))^2)

/-- The main theorem capturing the problem's assertions -/
theorem ellipse_properties (ε : Ellipse) (l : IntersectingLine ε) (p : PerpendicularPoint ε l) :
  ε.a = 2 ∧ ε.b = 1 ∧
  (l.k = 1 ∨ l.k = -1) ∧
  (p.y₀ = 2 * Real.sqrt 2 ∨ p.y₀ = -2 * Real.sqrt 2 ∨
   p.y₀ = 2 * Real.sqrt 14 / 5 ∨ p.y₀ = -2 * Real.sqrt 14 / 5) := by
  sorry

end ellipse_properties_l2597_259766


namespace inheritance_calculation_l2597_259775

theorem inheritance_calculation (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 15000) → x = 41379 := by
  sorry

end inheritance_calculation_l2597_259775


namespace dividing_line_slope_l2597_259761

/-- Polygon in the xy-plane with given vertices -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Line passing through the origin with a given slope -/
structure Line where
  slope : ℝ

/-- Function to calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℝ := sorry

/-- Function to check if a line divides a polygon into two equal areas -/
def dividesEqualArea (p : Polygon) (l : Line) : Prop := sorry

/-- The polygon with the given vertices -/
def givenPolygon : Polygon := {
  vertices := [(0, 0), (0, 4), (4, 4), (4, 2), (7, 2), (7, 0)]
}

/-- The theorem stating that the line with slope 2/7 divides the given polygon into two equal areas -/
theorem dividing_line_slope : 
  dividesEqualArea givenPolygon { slope := 2/7 } := by sorry

end dividing_line_slope_l2597_259761


namespace price_reduction_percentage_l2597_259798

/-- Proves that a price reduction resulting in an 80% increase in sales quantity 
    and a 44% increase in total revenue corresponds to a 20% reduction in price. -/
theorem price_reduction_percentage (P : ℝ) (S : ℝ) (P_new : ℝ) 
  (h1 : P > 0) (h2 : S > 0) (h3 : P_new > 0) :
  (P_new * (S * 1.8) = P * S * 1.44) → (P_new = P * 0.8) :=
by sorry

end price_reduction_percentage_l2597_259798


namespace cold_brew_time_per_batch_l2597_259771

/-- Proves that the time to make one batch of cold brew coffee is 20 hours -/
theorem cold_brew_time_per_batch : 
  ∀ (batch_size : ℝ) (daily_consumption : ℝ) (total_time : ℝ) (total_days : ℕ),
    batch_size = 1.5 →  -- size of one batch in gallons
    daily_consumption = 48 →  -- 96 ounces every 2 days = 48 ounces per day
    total_time = 120 →  -- total hours spent making coffee
    total_days = 24 →  -- number of days
    (total_time / (total_days * daily_consumption / (batch_size * 128))) = 20 := by
  sorry


end cold_brew_time_per_batch_l2597_259771


namespace sum_of_first_50_digits_l2597_259710

/-- The decimal expansion of 1/10101 -/
def decimal_expansion : ℕ → ℕ
| n => match n % 5 with
  | 0 => 0
  | 1 => 0
  | 2 => 0
  | 3 => 9
  | 4 => 9
  | _ => 0  -- This case is technically unreachable

/-- Sum of the first n digits in the decimal expansion -/
def sum_of_digits (n : ℕ) : ℕ :=
  (List.range n).map decimal_expansion |>.sum

/-- The theorem stating the sum of the first 50 digits after the decimal point in 1/10101 -/
theorem sum_of_first_50_digits :
  sum_of_digits 50 = 180 := by sorry

end sum_of_first_50_digits_l2597_259710


namespace decagon_diagonals_l2597_259764

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l2597_259764


namespace greatest_b_value_l2597_259753

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 18 ≥ 0 → x ≤ 6) ∧ (-6^2 + 9*6 - 18 ≥ 0) := by
  sorry

end greatest_b_value_l2597_259753


namespace f_max_at_neg_three_l2597_259773

/-- The quadratic function f(x) = -x^2 - 6x + 12 -/
def f (x : ℝ) : ℝ := -x^2 - 6*x + 12

/-- Theorem stating that f(x) attains its maximum value when x = -3 -/
theorem f_max_at_neg_three :
  ∃ (max : ℝ), f (-3) = max ∧ ∀ x, f x ≤ max :=
sorry

end f_max_at_neg_three_l2597_259773


namespace whisky_alcohol_percentage_l2597_259734

/-- The initial percentage of alcohol in a jar of whisky -/
def initial_alcohol_percentage : ℝ := 40

/-- The percentage of alcohol in the replacement whisky -/
def replacement_alcohol_percentage : ℝ := 19

/-- The percentage of alcohol after replacement -/
def final_alcohol_percentage : ℝ := 24

/-- The quantity of whisky replaced -/
def replaced_quantity : ℝ := 0.7619047619047619

/-- The total volume of whisky in the jar -/
def total_volume : ℝ := 1

theorem whisky_alcohol_percentage :
  initial_alcohol_percentage / 100 * (total_volume - replaced_quantity) +
  replacement_alcohol_percentage / 100 * replaced_quantity =
  final_alcohol_percentage / 100 * total_volume := by
  sorry

end whisky_alcohol_percentage_l2597_259734


namespace total_paint_used_l2597_259743

-- Define the amount of white paint used
def white_paint : ℕ := 660

-- Define the amount of blue paint used
def blue_paint : ℕ := 6029

-- Theorem stating the total amount of paint used
theorem total_paint_used : white_paint + blue_paint = 6689 := by
  sorry

end total_paint_used_l2597_259743


namespace samuel_spent_one_fifth_l2597_259724

theorem samuel_spent_one_fifth (total : ℕ) (samuel_initial : ℚ) (samuel_left : ℕ) : 
  total = 240 →
  samuel_initial = 3/4 * total →
  samuel_left = 132 →
  (samuel_initial - samuel_left : ℚ) / total = 1/5 := by
  sorry

end samuel_spent_one_fifth_l2597_259724


namespace cat_mouse_position_after_323_moves_l2597_259762

-- Define the positions for the cat
inductive CatPosition
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

-- Define the positions for the mouse
inductive MousePosition
  | TopMiddle
  | TopRight
  | RightMiddle
  | BottomRight
  | BottomMiddle
  | BottomLeft
  | LeftMiddle
  | TopLeft

-- Function to calculate cat's position after n moves
def catPositionAfterMoves (n : ℕ) : CatPosition :=
  match n % 4 with
  | 0 => CatPosition.BottomLeft
  | 1 => CatPosition.TopLeft
  | 2 => CatPosition.TopRight
  | _ => CatPosition.BottomRight

-- Function to calculate mouse's position after n moves
def mousePositionAfterMoves (n : ℕ) : MousePosition :=
  match n % 8 with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.TopMiddle
  | 2 => MousePosition.TopRight
  | 3 => MousePosition.RightMiddle
  | 4 => MousePosition.BottomRight
  | 5 => MousePosition.BottomMiddle
  | 6 => MousePosition.BottomLeft
  | _ => MousePosition.LeftMiddle

theorem cat_mouse_position_after_323_moves :
  (catPositionAfterMoves 323 = CatPosition.BottomRight) ∧
  (mousePositionAfterMoves 323 = MousePosition.RightMiddle) :=
by sorry

end cat_mouse_position_after_323_moves_l2597_259762


namespace best_route_is_D_l2597_259712

-- Define the structure for a route
structure Route where
  name : String
  baseTime : ℕ
  numLights : ℕ
  redLightTime : ℕ
  trafficDensity : String
  weatherCondition : String
  roadCondition : String

-- Define the routes
def routeA : Route := {
  name := "A",
  baseTime := 10,
  numLights := 3,
  redLightTime := 3,
  trafficDensity := "moderate",
  weatherCondition := "light rain",
  roadCondition := "good"
}

def routeB : Route := {
  name := "B",
  baseTime := 12,
  numLights := 4,
  redLightTime := 2,
  trafficDensity := "high",
  weatherCondition := "clear",
  roadCondition := "pothole"
}

def routeC : Route := {
  name := "C",
  baseTime := 11,
  numLights := 2,
  redLightTime := 4,
  trafficDensity := "low",
  weatherCondition := "clear",
  roadCondition := "construction"
}

def routeD : Route := {
  name := "D",
  baseTime := 14,
  numLights := 0,
  redLightTime := 0,
  trafficDensity := "medium",
  weatherCondition := "potential fog",
  roadCondition := "unknown"
}

-- Define the list of all routes
def allRoutes : List Route := [routeA, routeB, routeC, routeD]

-- Calculate the worst-case travel time for a route
def worstCaseTime (r : Route) : ℕ := r.baseTime + r.numLights * r.redLightTime

-- Define the theorem
theorem best_route_is_D :
  ∀ r ∈ allRoutes, worstCaseTime routeD ≤ worstCaseTime r :=
sorry

end best_route_is_D_l2597_259712


namespace additional_workers_needed_l2597_259701

/-- Represents the problem of calculating additional workers needed to complete a construction project on time -/
theorem additional_workers_needed
  (total_days : ℕ) 
  (initial_workers : ℕ) 
  (days_passed : ℕ) 
  (work_completed : ℚ) 
  (h1 : total_days = 50)
  (h2 : initial_workers = 20)
  (h3 : days_passed = 25)
  (h4 : work_completed = 2/5)
  : ℕ := by
  sorry

#check additional_workers_needed

end additional_workers_needed_l2597_259701


namespace parabola_focus_l2597_259745

/-- The parabola defined by the equation y = (1/4)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- The parabola opens upwards -/
def opens_upwards (p : (ℝ → ℝ → Prop)) : Prop := sorry

/-- The focus lies on the y-axis -/
def focus_on_y_axis (f : Focus) : Prop := f.x = 0

/-- Theorem stating that the focus of the given parabola is at (0, 1) -/
theorem parabola_focus :
  ∃ (f : Focus),
    (∀ x y, parabola x y → opens_upwards parabola) ∧
    focus_on_y_axis f ∧
    f.x = 0 ∧ f.y = 1 := by sorry

end parabola_focus_l2597_259745


namespace birthday_cookies_l2597_259797

/-- The number of pans of cookies -/
def num_pans : ℕ := 5

/-- The number of cookies per pan -/
def cookies_per_pan : ℕ := 8

/-- The total number of cookies baked -/
def total_cookies : ℕ := num_pans * cookies_per_pan

theorem birthday_cookies : total_cookies = 40 := by
  sorry

end birthday_cookies_l2597_259797


namespace prob_zero_or_one_white_is_four_fifths_l2597_259708

def total_balls : ℕ := 6
def red_balls : ℕ := 4
def white_balls : ℕ := 2
def selected_balls : ℕ := 3

def prob_zero_or_one_white (total : ℕ) (red : ℕ) (white : ℕ) (selected : ℕ) : ℚ :=
  (Nat.choose red selected + Nat.choose white 1 * Nat.choose red (selected - 1)) /
  Nat.choose total selected

theorem prob_zero_or_one_white_is_four_fifths :
  prob_zero_or_one_white total_balls red_balls white_balls selected_balls = 4 / 5 := by
  sorry

end prob_zero_or_one_white_is_four_fifths_l2597_259708


namespace ticket_difference_l2597_259751

/-- Represents the number of tickets sold for each category -/
structure TicketSales where
  vip : ℕ
  premium : ℕ
  general : ℕ

/-- Checks if the given ticket sales satisfy the problem conditions -/
def satisfiesConditions (sales : TicketSales) : Prop :=
  sales.vip + sales.premium + sales.general = 420 ∧
  50 * sales.vip + 30 * sales.premium + 10 * sales.general = 12000

/-- Theorem stating the difference between general admission and VIP tickets -/
theorem ticket_difference (sales : TicketSales) 
  (h : satisfiesConditions sales) : 
  sales.general - sales.vip = 30 := by
  sorry

end ticket_difference_l2597_259751


namespace find_T_l2597_259758

theorem find_T : ∃ T : ℚ, (3/4 : ℚ) * (1/8 : ℚ) * T = (1/2 : ℚ) * (1/6 : ℚ) * 72 ∧ T = 64 := by
  sorry

end find_T_l2597_259758


namespace min_value_expression_equality_condition_l2597_259769

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (a * b) + 1 / (a * (a - b)) ≥ 4 :=
by sorry

theorem equality_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (a * b) + 1 / (a * (a - b)) = 4 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2 / 2 :=
by sorry

end min_value_expression_equality_condition_l2597_259769


namespace bake_sale_total_l2597_259717

theorem bake_sale_total (cookies : ℕ) (brownies : ℕ) : 
  cookies = 48 → 
  brownies * 6 = cookies * 7 →
  cookies + brownies = 104 := by
sorry

end bake_sale_total_l2597_259717


namespace canning_box_theorem_l2597_259737

/-- Represents the solution to the canning box problem -/
def canning_box_solution (total_sheets : ℕ) (bodies_per_sheet : ℕ) (bottoms_per_sheet : ℕ) 
  (sheets_for_bodies : ℕ) (sheets_for_bottoms : ℕ) : Prop :=
  -- All sheets are used
  sheets_for_bodies + sheets_for_bottoms = total_sheets ∧
  -- Number of bodies matches half the number of bottoms
  bodies_per_sheet * sheets_for_bodies = (bottoms_per_sheet * sheets_for_bottoms) / 2 ∧
  -- Solution is optimal (no other solution exists)
  ∀ (x y : ℕ), 
    x + y = total_sheets ∧ 
    bodies_per_sheet * x = (bottoms_per_sheet * y) / 2 → 
    x ≤ sheets_for_bodies ∧ y ≤ sheets_for_bottoms

/-- The canning box theorem -/
theorem canning_box_theorem : 
  canning_box_solution 33 30 50 15 18 := by
  sorry

end canning_box_theorem_l2597_259737


namespace constant_function_invariant_l2597_259703

-- Define the function g
def g : ℝ → ℝ := λ x => -3

-- State the theorem
theorem constant_function_invariant (x : ℝ) : g (3 * x - 1) = -3 := by
  sorry

end constant_function_invariant_l2597_259703


namespace josh_pencils_l2597_259732

theorem josh_pencils (pencils_given : ℕ) (pencils_left : ℕ) 
  (h1 : pencils_given = 31) 
  (h2 : pencils_left = 111) : 
  pencils_given + pencils_left = 142 := by
  sorry

end josh_pencils_l2597_259732


namespace perpendicular_line_slope_OA_longer_than_OB_l2597_259729

/-- The ellipse C with equation x² + y²/4 = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2/4 = 1}

/-- The line y = kx + 1 for a given k -/
def Line (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + 1}

/-- A and B are the intersection points of C and the line -/
def A (k : ℝ) : ℝ × ℝ := sorry
def B (k : ℝ) : ℝ × ℝ := sorry

/-- Condition that A is in the first quadrant -/
def A_in_first_quadrant (k : ℝ) : Prop := (A k).1 > 0 ∧ (A k).2 > 0

theorem perpendicular_line_slope (k : ℝ) :
  (A k).1 * (B k).1 + (A k).2 * (B k).2 = 0 → k = 1/2 ∨ k = -1/2 := sorry

theorem OA_longer_than_OB (k : ℝ) :
  k > 0 → A_in_first_quadrant k →
  (A k).1^2 + (A k).2^2 > (B k).1^2 + (B k).2^2 := sorry

end perpendicular_line_slope_OA_longer_than_OB_l2597_259729


namespace smallest_integer_with_remainders_l2597_259720

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 10 = 9 ∧
  (∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m) ∧
  n = 59 :=
sorry

end smallest_integer_with_remainders_l2597_259720


namespace fraction_sum_minus_eight_l2597_259799

theorem fraction_sum_minus_eight : 
  (4/3 : ℚ) + (7/5 : ℚ) + (12/10 : ℚ) + (23/20 : ℚ) + (45/40 : ℚ) + (89/80 : ℚ) - 8 = -163/240 := by
  sorry

end fraction_sum_minus_eight_l2597_259799


namespace fifteen_machines_six_minutes_l2597_259748

/-- The number of paperclips produced by a given number of machines in a given time -/
def paperclips_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  let base_machines := 8
  let base_production := 560
  let production_per_machine := base_production / base_machines
  machines * production_per_machine * minutes

/-- Theorem stating that 15 machines will produce 6300 paperclips in 6 minutes -/
theorem fifteen_machines_six_minutes :
  paperclips_produced 15 6 = 6300 := by
  sorry

end fifteen_machines_six_minutes_l2597_259748


namespace constant_c_value_l2597_259789

theorem constant_c_value (b c : ℚ) :
  (∀ x : ℚ, (x + 3) * (x + b) = x^2 + c*x + 8) →
  c = 17/3 := by
sorry

end constant_c_value_l2597_259789


namespace matrix_sum_theorem_l2597_259784

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -1; 3, 7]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; 5, -2]

theorem matrix_sum_theorem : A + B = !![(-2), 7; 8, 5] := by sorry

end matrix_sum_theorem_l2597_259784


namespace triangle_sine_relation_l2597_259782

theorem triangle_sine_relation (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_relation : 3 * Real.sin B ^ 2 + 7 * Real.sin C ^ 2 = 
                2 * Real.sin A * Real.sin B * Real.sin C + 2 * Real.sin A ^ 2) : 
  Real.sin (A + π / 4) = -Real.sqrt 10 / 10 := by
  sorry

end triangle_sine_relation_l2597_259782


namespace twice_x_minus_three_greater_than_four_l2597_259783

theorem twice_x_minus_three_greater_than_four (x : ℝ) :
  (2 * x - 3 > 4) ↔ (∃ y, y = 2 * x - 3 ∧ y > 4) :=
sorry

end twice_x_minus_three_greater_than_four_l2597_259783


namespace increasing_function_property_l2597_259727

theorem increasing_function_property (f : ℝ → ℝ) (a b : ℝ)
  (h_increasing : ∀ x y : ℝ, x ≤ y → f x ≤ f y) :
  (a + b ≥ 0 ↔ f a + f b ≥ f (-a) + f (-b)) := by
sorry

end increasing_function_property_l2597_259727


namespace tom_drives_12_miles_l2597_259752

/-- A car race between Karen and Tom -/
structure CarRace where
  karen_speed : ℝ  -- Karen's speed in mph
  tom_speed : ℝ    -- Tom's speed in mph
  karen_delay : ℝ  -- Karen's delay in minutes
  win_margin : ℝ   -- Karen's winning margin in miles

/-- Calculate the distance Tom drives before Karen wins -/
def distance_tom_drives (race : CarRace) : ℝ :=
  sorry

/-- Theorem stating that Tom drives 12 miles before Karen wins -/
theorem tom_drives_12_miles (race : CarRace) 
  (h1 : race.karen_speed = 60)
  (h2 : race.tom_speed = 45)
  (h3 : race.karen_delay = 4)
  (h4 : race.win_margin = 4) :
  distance_tom_drives race = 12 :=
sorry

end tom_drives_12_miles_l2597_259752


namespace max_x_placement_l2597_259740

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Bool

/-- Checks if there are three X's in a row in any direction --/
def has_three_in_a_row (g : Grid) : Bool :=
  sorry

/-- Counts the number of X's in the grid --/
def count_x (g : Grid) : Nat :=
  sorry

/-- Theorem stating the maximum number of X's that can be placed --/
theorem max_x_placement :
  ∃ (g : Grid), count_x g = 13 ∧ ¬has_three_in_a_row g ∧
  ∀ (h : Grid), count_x h > 13 → has_three_in_a_row h :=
sorry

end max_x_placement_l2597_259740


namespace three_digit_mean_rearrangement_l2597_259787

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  7 * a = 3 * b + 4 * c

def solution_set : Set ℕ :=
  {111, 222, 333, 444, 555, 666, 777, 888, 999, 407, 518, 629, 370, 481, 592}

theorem three_digit_mean_rearrangement (n : ℕ) :
  is_valid_number n ↔ n ∈ solution_set :=
sorry

end three_digit_mean_rearrangement_l2597_259787


namespace birds_landed_on_fence_l2597_259742

/-- Given an initial number of birds and a final total number of birds on a fence,
    calculate the number of birds that landed on the fence. -/
def birds_landed (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that 8 birds landed on the fence given the initial and final counts. -/
theorem birds_landed_on_fence : birds_landed 12 20 = 8 := by
  sorry

end birds_landed_on_fence_l2597_259742


namespace chloe_carrots_theorem_l2597_259731

/-- Calculates the total number of carrots Chloe has after throwing some out and picking more. -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) : ℕ :=
  initial - thrown_out + new_picked

/-- Proves that Chloe's total carrots is correct given the initial conditions. -/
theorem chloe_carrots_theorem (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) 
  (h1 : initial ≥ thrown_out) : 
  total_carrots initial thrown_out new_picked = initial - thrown_out + new_picked :=
by sorry

end chloe_carrots_theorem_l2597_259731


namespace ratio_equality_l2597_259756

theorem ratio_equality (a b : ℝ) (h1 : 5 * a = 3 * b) (h2 : a ≠ 0) (h3 : b ≠ 0) : b / a = 5 / 3 := by
  sorry

end ratio_equality_l2597_259756


namespace smallest_divisible_by_18_and_60_l2597_259757

theorem smallest_divisible_by_18_and_60 : ∀ n : ℕ, n > 0 ∧ 18 ∣ n ∧ 60 ∣ n → n ≥ 180 := by
  sorry

end smallest_divisible_by_18_and_60_l2597_259757


namespace units_digit_of_n_l2597_259791

theorem units_digit_of_n (m n : ℕ) : 
  m * n = 23^7 → 
  m % 10 = 9 → 
  n % 10 = 3 := by
sorry

end units_digit_of_n_l2597_259791


namespace dot_only_count_l2597_259716

/-- Represents an alphabet with dots and straight lines -/
structure Alphabet where
  total : ℕ
  both : ℕ
  line_only : ℕ
  has_dot_or_line : total = both + line_only + (total - (both + line_only))

/-- The number of letters with a dot but no straight line in the given alphabet -/
def letters_with_dot_only (α : Alphabet) : ℕ :=
  α.total - (α.both + α.line_only)

/-- Theorem stating the number of letters with a dot but no straight line -/
theorem dot_only_count (α : Alphabet) 
  (h1 : α.total = 80)
  (h2 : α.both = 28)
  (h3 : α.line_only = 47) :
  letters_with_dot_only α = 5 := by
  sorry

#check dot_only_count

end dot_only_count_l2597_259716


namespace subset_condition_l2597_259700

def A : Set ℝ := {x | x < -1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a * x + 1 ≤ 0}

theorem subset_condition (a : ℝ) : B a ⊆ A ↔ -1/3 ≤ a ∧ a < 1 := by
  sorry

end subset_condition_l2597_259700


namespace gcd_768_288_l2597_259780

theorem gcd_768_288 : Int.gcd 768 288 = 96 := by sorry

end gcd_768_288_l2597_259780


namespace alice_painted_six_cuboids_l2597_259750

/-- The number of faces on a single cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of faces Alice painted -/
def total_painted_faces : ℕ := 36

/-- The number of cuboids Alice painted -/
def num_cuboids : ℕ := total_painted_faces / faces_per_cuboid

theorem alice_painted_six_cuboids :
  num_cuboids = 6 :=
sorry

end alice_painted_six_cuboids_l2597_259750


namespace broomstick_race_orderings_l2597_259728

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of competitors in the race -/
def num_competitors : ℕ := 4

theorem broomstick_race_orderings : 
  permutations num_competitors = 24 := by
  sorry

end broomstick_race_orderings_l2597_259728


namespace perpendicular_lines_a_value_l2597_259790

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines_a_value 
  (l1 : ℝ → ℝ → Prop) 
  (l2 : ℝ → ℝ → Prop)
  (a : ℝ) 
  (h1 : ∀ x y, l1 x y ↔ x + 2*a*y - 1 = 0)
  (h2 : ∀ x y, l2 x y ↔ x - 4*y = 0)
  (h3 : perpendicular (2*a) (1/4)) :
  a = 1/8 := by
sorry

end perpendicular_lines_a_value_l2597_259790


namespace spherical_to_rectangular_conversion_l2597_259785

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := π / 4
  let φ : ℝ := π / 6
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (Real.sqrt 2, Real.sqrt 2, 2 * Real.sqrt 3) :=
by
  sorry

end spherical_to_rectangular_conversion_l2597_259785


namespace boat_speed_upstream_l2597_259709

/-- The speed of a boat upstream given its speed in still water and the speed of the current. -/
def speed_upstream (speed_still_water : ℝ) (speed_current : ℝ) : ℝ :=
  speed_still_water - speed_current

/-- Theorem: The speed of a boat upstream is 30 kmph when its speed in still water is 50 kmph and the current speed is 20 kmph. -/
theorem boat_speed_upstream :
  speed_upstream 50 20 = 30 := by
  sorry

end boat_speed_upstream_l2597_259709

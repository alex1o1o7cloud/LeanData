import Mathlib

namespace reading_time_calculation_l1904_190473

def total_homework_time : ℕ := 60
def math_time : ℕ := 15
def spelling_time : ℕ := 18

theorem reading_time_calculation :
  total_homework_time - (math_time + spelling_time) = 27 := by
  sorry

end reading_time_calculation_l1904_190473


namespace greatest_integer_no_real_roots_l1904_190456

theorem greatest_integer_no_real_roots (a : ℤ) : 
  (∀ x : ℝ, x^2 + a*x + 15 ≠ 0) → a ≤ 7 ∧ ∀ b : ℤ, (∀ x : ℝ, x^2 + b*x + 15 ≠ 0) → b ≤ 7 :=
by
  sorry

end greatest_integer_no_real_roots_l1904_190456


namespace circular_platform_area_l1904_190419

/-- The area of a circular platform with a diameter of 2 yards is π square yards. -/
theorem circular_platform_area (diameter : ℝ) (h : diameter = 2) : 
  (π * (diameter / 2)^2 : ℝ) = π := by sorry

end circular_platform_area_l1904_190419


namespace max_xy_value_l1904_190428

theorem max_xy_value (a b c x y : ℝ) :
  a * x + b * y + 2 * c = 0 →
  c ≠ 0 →
  a * b - c^2 ≥ 0 →
  x * y ≤ 1 :=
by sorry

end max_xy_value_l1904_190428


namespace distance_between_points_l1904_190411

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 4)
  let p2 : ℝ × ℝ := (-6, -1)
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt 106 :=
by sorry

end distance_between_points_l1904_190411


namespace janes_change_l1904_190478

/-- The change Jane receives when buying an apple -/
theorem janes_change (apple_price : ℚ) (paid_amount : ℚ) (change : ℚ) : 
  apple_price = 0.75 → paid_amount = 5 → change = paid_amount - apple_price → change = 4.25 := by
  sorry

end janes_change_l1904_190478


namespace meeting_percentage_is_42_percent_l1904_190429

def work_day_hours : ℕ := 10
def lunch_break_minutes : ℕ := 30
def first_meeting_minutes : ℕ := 60

def work_day_minutes : ℕ := work_day_hours * 60
def effective_work_minutes : ℕ := work_day_minutes - lunch_break_minutes
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (effective_work_minutes : ℚ) * 100

theorem meeting_percentage_is_42_percent : 
  ⌊meeting_percentage⌋ = 42 :=
sorry

end meeting_percentage_is_42_percent_l1904_190429


namespace figure_area_proof_l1904_190490

theorem figure_area_proof (r1_height r1_width r2_height r2_width r3_height r3_width r4_height r4_width : ℕ) 
  (h1 : r1_height = 6 ∧ r1_width = 5)
  (h2 : r2_height = 3 ∧ r2_width = 5)
  (h3 : r3_height = 3 ∧ r3_width = 10)
  (h4 : r4_height = 8 ∧ r4_width = 2) :
  r1_height * r1_width + r2_height * r2_width + r3_height * r3_width + r4_height * r4_width = 91 := by
  sorry

#check figure_area_proof

end figure_area_proof_l1904_190490


namespace marks_siblings_count_l1904_190406

/-- The number of Mark's siblings given the egg distribution problem -/
def marks_siblings : ℕ :=
  let total_eggs : ℕ := 24  -- two dozen eggs
  let eggs_per_person : ℕ := 6
  let total_people : ℕ := total_eggs / eggs_per_person
  total_people - 1

theorem marks_siblings_count : marks_siblings = 3 := by
  sorry

end marks_siblings_count_l1904_190406


namespace imaginary_part_of_2_plus_i_times_i_l1904_190420

theorem imaginary_part_of_2_plus_i_times_i (i : ℂ) : 
  i ^ 2 = -1 → Complex.im ((2 + i) * i) = 2 := by
  sorry

end imaginary_part_of_2_plus_i_times_i_l1904_190420


namespace identical_numbers_l1904_190476

theorem identical_numbers (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y^2 = y + 1 / x^2) (h2 : y^2 + 1 / x = x^2 + 1 / y) :
  x = y :=
by sorry

end identical_numbers_l1904_190476


namespace smallest_b_in_arithmetic_sequence_l1904_190448

/-- Given four positive terms in an arithmetic sequence with their product equal to 256,
    the smallest possible value of the second term is 4. -/
theorem smallest_b_in_arithmetic_sequence (a b c d : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →  -- all terms are positive
  ∃ (r : ℝ), a = b - r ∧ c = b + r ∧ d = b + 2*r →  -- arithmetic sequence
  a * b * c * d = 256 →  -- product is 256
  b ≥ 4 ∧ ∃ (r : ℝ), 4 - r > 0 ∧ 4 * (4 - r) * (4 + r) * (4 + 2*r) = 256 :=  -- b ≥ 4 and there exists a valid r for b = 4
by sorry

end smallest_b_in_arithmetic_sequence_l1904_190448


namespace complex_power_problem_l1904_190423

theorem complex_power_problem (z : ℂ) : 
  (1 + z) / (1 - z) = Complex.I → z^2023 = -Complex.I := by
  sorry

end complex_power_problem_l1904_190423


namespace multiples_count_theorem_l1904_190463

def count_multiples (n : ℕ) (d : ℕ) : ℕ :=
  (n / d : ℕ)

def count_multiples_of_2_or_3_not_4_or_5 (upper_bound : ℕ) : ℕ :=
  count_multiples upper_bound 2 + count_multiples upper_bound 3 -
  count_multiples upper_bound 6 - count_multiples upper_bound 4 -
  count_multiples upper_bound 5 + count_multiples upper_bound 20

theorem multiples_count_theorem (upper_bound : ℕ) :
  upper_bound = 200 →
  count_multiples_of_2_or_3_not_4_or_5 upper_bound = 53 := by
  sorry

end multiples_count_theorem_l1904_190463


namespace original_number_proof_l1904_190468

theorem original_number_proof : ∃ x : ℤ, (x + 24) % 27 = 0 ∧ x = 30 := by
  sorry

end original_number_proof_l1904_190468


namespace height_of_column_G_l1904_190433

-- Define the regular octagon vertices
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (0, -8)
def D : ℝ × ℝ := (-8, 0)
def G : ℝ × ℝ := (0, 8)

-- Define the heights of columns A, B, C, D
def height_A : ℝ := 15
def height_B : ℝ := 12
def height_C : ℝ := 14
def height_D : ℝ := 13

-- Theorem statement
theorem height_of_column_G : 
  ∃ (height_G : ℝ), height_G = 15.5 :=
by
  sorry

end height_of_column_G_l1904_190433


namespace radio_loss_percentage_l1904_190491

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Theorem stating that the loss percentage for a radio with cost price 1500 and selling price 1110 is 26% -/
theorem radio_loss_percentage :
  let cost_price : ℚ := 1500
  let selling_price : ℚ := 1110
  loss_percentage cost_price selling_price = 26 := by
  sorry

end radio_loss_percentage_l1904_190491


namespace range_of_x_l1904_190446

theorem range_of_x (x : ℝ) : (x^2 - 2*x - 3 ≥ 0) ∧ ¬(|1 - x/2| < 1) ↔ x ≤ -1 ∨ x ≥ 4 := by
  sorry

end range_of_x_l1904_190446


namespace power_division_equals_729_l1904_190485

theorem power_division_equals_729 : (3 : ℕ) ^ 15 / (27 : ℕ) ^ 3 = 729 := by
  sorry

end power_division_equals_729_l1904_190485


namespace parallel_line_y_intercept_l1904_190422

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem parallel_line_y_intercept
  (b : Line)
  (given_line : Line)
  (p : Point) :
  parallel b given_line →
  given_line.slope = -3 →
  given_line.intercept = 6 →
  p.x = 3 →
  p.y = -1 →
  pointOnLine p b →
  b.intercept = 8 := by
  sorry

end parallel_line_y_intercept_l1904_190422


namespace max_quarters_exact_solution_l1904_190442

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- Tony's total money in dollars -/
def total_money : ℚ := 490 / 100

/-- 
  Given that Tony has the same number of quarters and dimes, and his total money is $4.90,
  prove that the maximum number of quarters he can have is 14.
-/
theorem max_quarters : 
  ∀ q : ℕ, 
  (q : ℚ) * (quarter_value + dime_value) ≤ total_money → 
  q ≤ 14 :=
by sorry

/-- Prove that 14 quarters and 14 dimes exactly equal $4.90 -/
theorem exact_solution : 
  (14 : ℚ) * (quarter_value + dime_value) = total_money :=
by sorry

end max_quarters_exact_solution_l1904_190442


namespace subtraction_result_l1904_190410

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  valid : hundreds < 10 ∧ tens < 10 ∧ units < 10

/-- The result of subtracting two three-digit numbers -/
def subtract (a b : ThreeDigitNumber) : ThreeDigitNumber :=
  sorry

theorem subtraction_result 
  (a b : ThreeDigitNumber)
  (h_units : a.units = b.units + 6)
  (h_result_units : (subtract a b).units = 5)
  (h_result_tens : (subtract a b).tens = 9)
  (h_no_borrow : a.tens ≥ b.tens) :
  (subtract a b).hundreds = 4 :=
sorry

end subtraction_result_l1904_190410


namespace hyperbola_focus_l1904_190430

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  2 * x^2 - y^2 + 8 * x - 6 * y - 8 = 0

/-- Definition of a focus for this hyperbola -/
def is_focus (x y : ℝ) : Prop :=
  ∃ (sign : ℝ), sign = 1 ∨ sign = -1 ∧
  x = -2 + sign * Real.sqrt 10.5 ∧
  y = -3

/-- Theorem stating that (-2 + √10.5, -3) is a focus of the hyperbola -/
theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola_eq x y ∧ is_focus x y :=
sorry

end hyperbola_focus_l1904_190430


namespace problem_1_problem_2_problem_3_l1904_190453

-- Problem 1
theorem problem_1 : 2013^2 - 2012 * 2014 = 1 := by sorry

-- Problem 2
theorem problem_2 (m n : ℤ) : ((m - n)^6 / (n - m)^4) * (m - n)^3 = (m - n)^5 := by sorry

-- Problem 3
theorem problem_3 (a b c : ℝ) : (a - 2*b + 3*c) * (a - 2*b - 3*c) = a^2 - 4*a*b + 4*b^2 - 9*c^2 := by sorry

end problem_1_problem_2_problem_3_l1904_190453


namespace complex_fraction_calculation_l1904_190441

theorem complex_fraction_calculation : 
  (((11 + 1/9 - (3 + 2/5) * (1 + 2/17)) - (8 + 2/5) / 3.6) / (2 + 6/25)) = 20/9 := by
  sorry

end complex_fraction_calculation_l1904_190441


namespace rhombus_diagonals_not_necessarily_equal_l1904_190498

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j : Fin 4, sides i = sides j

/-- The diagonals of a rhombus -/
def rhombus_diagonals (r : Rhombus) : ℝ × ℝ := sorry

/-- Theorem: The diagonals of a rhombus are not necessarily equal -/
theorem rhombus_diagonals_not_necessarily_equal :
  ∃ r : Rhombus, (rhombus_diagonals r).1 ≠ (rhombus_diagonals r).2 :=
sorry

end rhombus_diagonals_not_necessarily_equal_l1904_190498


namespace business_value_calculation_l1904_190408

theorem business_value_calculation (owned_share : ℚ) (sold_portion : ℚ) (sale_price : ℕ) :
  owned_share = 2/3 →
  sold_portion = 3/4 →
  sale_price = 75000 →
  (sale_price : ℚ) / (owned_share * sold_portion) = 150000 := by
  sorry

end business_value_calculation_l1904_190408


namespace part_one_part_two_l1904_190457

-- Define the & operation
def ampersand (a b : ℚ) : ℚ := b^2 - a*b

-- Part 1
theorem part_one : ampersand (2/3) (-1/2) = 7/12 := by sorry

-- Part 2
theorem part_two (x y : ℚ) (h : |x + 1| + (y - 3)^2 = 0) : 
  ampersand x y = 12 := by sorry

end part_one_part_two_l1904_190457


namespace walter_works_five_days_l1904_190481

/-- Calculates the number of days Walter works per week given his hourly rate, daily hours, allocation percentage, and allocated amount for school. -/
def calculate_work_days (hourly_rate : ℚ) (daily_hours : ℚ) (allocation_percentage : ℚ) (school_allocation : ℚ) : ℚ :=
  let daily_earnings := hourly_rate * daily_hours
  let weekly_earnings := school_allocation / allocation_percentage
  weekly_earnings / daily_earnings

/-- Theorem stating that Walter works 5 days a week given the specified conditions. -/
theorem walter_works_five_days 
  (hourly_rate : ℚ) 
  (daily_hours : ℚ) 
  (allocation_percentage : ℚ) 
  (school_allocation : ℚ) 
  (h1 : hourly_rate = 5)
  (h2 : daily_hours = 4)
  (h3 : allocation_percentage = 3/4)
  (h4 : school_allocation = 75) :
  calculate_work_days hourly_rate daily_hours allocation_percentage school_allocation = 5 := by
  sorry

end walter_works_five_days_l1904_190481


namespace scientific_notation_600000_l1904_190435

theorem scientific_notation_600000 : 600000 = 6 * (10 : ℝ)^5 := by sorry

end scientific_notation_600000_l1904_190435


namespace lipschitz_arithmetic_is_translation_l1904_190427

/-- A function f : ℝ → ℝ satisfying the given conditions -/
def LipschitzArithmeticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, |f x - f y| ≤ |x - y|) ∧
  (∀ x : ℝ, ∃ d : ℝ, ∀ n : ℕ, (f^[n]) x = x + n • d)

/-- The main theorem -/
theorem lipschitz_arithmetic_is_translation
  (f : ℝ → ℝ) (h : LipschitzArithmeticFunction f) :
  ∃ a : ℝ, ∀ x : ℝ, f x = x + a := by
  sorry

end lipschitz_arithmetic_is_translation_l1904_190427


namespace rectangular_box_height_l1904_190444

/-- Proves that the height of a rectangular box is 2 cm, given its volume, length, and width. -/
theorem rectangular_box_height (volume : ℝ) (length width : ℝ) (h1 : volume = 144) (h2 : length = 12) (h3 : width = 6) :
  volume = length * width * 2 :=
by sorry

end rectangular_box_height_l1904_190444


namespace expression_equality_l1904_190402

theorem expression_equality : (2 * Real.sqrt 2 - 1)^2 + (1 + Real.sqrt 3) * (1 - Real.sqrt 3) = 7 - 4 * Real.sqrt 2 := by
  sorry

end expression_equality_l1904_190402


namespace arithmetic_calculation_l1904_190432

theorem arithmetic_calculation : 5 * 7 + 6 * 9 + 8 * 4 + 7 * 6 = 163 := by
  sorry

end arithmetic_calculation_l1904_190432


namespace omega_function_iff_strictly_increasing_l1904_190484

def OmegaFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

theorem omega_function_iff_strictly_increasing (f : ℝ → ℝ) :
  OmegaFunction f ↔ StrictMono f := by sorry

end omega_function_iff_strictly_increasing_l1904_190484


namespace negative_square_two_l1904_190482

theorem negative_square_two : -2^2 = -4 := by
  sorry

end negative_square_two_l1904_190482


namespace ratio_equals_one_l1904_190475

theorem ratio_equals_one (a b c : ℝ) 
  (eq1 : 2*a + 13*b + 3*c = 90)
  (eq2 : 3*a + 9*b + c = 72) :
  (3*b + c) / (a + 2*b) = 1 := by
sorry

end ratio_equals_one_l1904_190475


namespace perpendicular_iff_a_eq_one_l1904_190462

-- Define the lines
def line1 (x y : ℝ) : Prop := x + y = 0
def line2 (a x y : ℝ) : Prop := x - a * y = 0

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, line1 x1 y1 ∧ line2 a x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    (x2 - x1) * (y2 - y1) = 0

-- State the theorem
theorem perpendicular_iff_a_eq_one :
  ∀ a : ℝ, perpendicular a ↔ a = 1 := by sorry

end perpendicular_iff_a_eq_one_l1904_190462


namespace subset_of_A_l1904_190449

def A : Set ℕ := {x | x ≤ 4}

theorem subset_of_A : {3} ⊆ A := by
  sorry

end subset_of_A_l1904_190449


namespace cake_recipe_proof_l1904_190443

/-- Represents the amounts of ingredients in cups -/
structure Recipe :=
  (flour : ℚ)
  (sugar : ℚ)
  (cocoa : ℚ)
  (milk : ℚ)

def original_recipe : Recipe :=
  { flour := 3/4
  , sugar := 2/3
  , cocoa := 1/3
  , milk := 1/2 }

def doubled_recipe : Recipe :=
  { flour := 2 * original_recipe.flour
  , sugar := 2 * original_recipe.sugar
  , cocoa := 2 * original_recipe.cocoa
  , milk := 2 * original_recipe.milk }

def already_added : Recipe :=
  { flour := 1/2
  , sugar := 1/4
  , cocoa := 0
  , milk := 0 }

def additional_needed : Recipe :=
  { flour := doubled_recipe.flour - already_added.flour
  , sugar := doubled_recipe.sugar - already_added.sugar
  , cocoa := doubled_recipe.cocoa - already_added.cocoa
  , milk := doubled_recipe.milk - already_added.milk }

theorem cake_recipe_proof :
  additional_needed.flour = 1 ∧
  additional_needed.sugar = 13/12 ∧
  additional_needed.cocoa = 2/3 ∧
  additional_needed.milk = 1 :=
sorry

end cake_recipe_proof_l1904_190443


namespace inequality_solution_set_l1904_190454

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - x - 6) / (x - 1) > 0 ↔ (-2 < x ∧ x < 1) ∨ x > 3 := by
  sorry

end inequality_solution_set_l1904_190454


namespace correct_height_order_l1904_190424

-- Define the friends
inductive Friend : Type
| David : Friend
| Emma : Friend
| Fiona : Friend

-- Define the height comparison relation
def taller_than : Friend → Friend → Prop := sorry

-- Define the conditions
axiom different_heights :
  ∀ (a b : Friend), a ≠ b → (taller_than a b ∨ taller_than b a)

axiom transitive :
  ∀ (a b c : Friend), taller_than a b → taller_than b c → taller_than a c

axiom asymmetric :
  ∀ (a b : Friend), taller_than a b → ¬taller_than b a

axiom exactly_one_true :
  (¬(taller_than Friend.Fiona Friend.Emma) ∧
   ¬(taller_than Friend.David Friend.Fiona) ∧
   taller_than Friend.David Friend.Emma) ∨
  (taller_than Friend.Fiona Friend.David ∧
   taller_than Friend.Fiona Friend.Emma) ∨
  (¬(taller_than Friend.David Friend.Emma) ∧
   ¬(taller_than Friend.David Friend.Fiona))

-- Theorem to prove
theorem correct_height_order :
  taller_than Friend.David Friend.Emma ∧
  taller_than Friend.Emma Friend.Fiona ∧
  taller_than Friend.David Friend.Fiona :=
sorry

end correct_height_order_l1904_190424


namespace correct_apple_count_l1904_190474

/-- Represents the types of apples Aria needs to buy. -/
structure AppleCount where
  red : ℕ
  granny : ℕ
  golden : ℕ

/-- Calculates the total number of apples Aria needs to buy for two weeks. -/
def totalApplesForTwoWeeks (normalDays weekDays : ℕ) (normalMix specialMix : AppleCount) : AppleCount :=
  { red := normalDays * normalMix.red + weekDays * specialMix.red,
    granny := normalDays * normalMix.granny + weekDays * specialMix.granny,
    golden := (normalDays + weekDays) * normalMix.golden }

/-- Theorem stating the correct number of apples Aria needs to buy for two weeks. -/
theorem correct_apple_count :
  let normalDays : ℕ := 10
  let weekDays : ℕ := 4
  let normalMix : AppleCount := { red := 1, granny := 2, golden := 1 }
  let specialMix : AppleCount := { red := 2, granny := 1, golden := 1 }
  let result := totalApplesForTwoWeeks normalDays weekDays normalMix specialMix
  result.red = 18 ∧ result.granny = 24 ∧ result.golden = 14 := by sorry

end correct_apple_count_l1904_190474


namespace area_between_circles_l1904_190460

-- Define the circles
def outer_circle_radius : ℝ := 12
def chord_length : ℝ := 20

-- Define the theorem
theorem area_between_circles :
  ∃ (inner_circle_radius : ℝ),
    inner_circle_radius > 0 ∧
    inner_circle_radius < outer_circle_radius ∧
    chord_length^2 = 4 * (outer_circle_radius^2 - inner_circle_radius^2) ∧
    π * (outer_circle_radius^2 - inner_circle_radius^2) = 100 * π :=
by
  sorry


end area_between_circles_l1904_190460


namespace band_practice_schedule_l1904_190479

theorem band_practice_schedule (anthony ben carlos dean : ℕ) 
  (h1 : anthony = 5)
  (h2 : ben = 6)
  (h3 : carlos = 8)
  (h4 : dean = 9) :
  Nat.lcm anthony (Nat.lcm ben (Nat.lcm carlos dean)) = 360 := by
  sorry

end band_practice_schedule_l1904_190479


namespace line_inclination_angle_l1904_190472

/-- The inclination angle of the line x*sin(π/7) + y*cos(π/7) = 0 is 6π/7 -/
theorem line_inclination_angle : 
  let line_eq := fun (x y : ℝ) => x * Real.sin (π / 7) + y * Real.cos (π / 7) = 0
  ∃ (α : ℝ), α = 6 * π / 7 ∧ 
    (∀ (x y : ℝ), line_eq x y → 
      Real.tan α = - (Real.sin (π / 7) / Real.cos (π / 7))) :=
by sorry

end line_inclination_angle_l1904_190472


namespace opposite_absolute_values_imply_y_power_x_l1904_190470

theorem opposite_absolute_values_imply_y_power_x (x y : ℝ) : 
  |2*y - 3| + |5*x - 10| = 0 → y^x = 9/4 := by
  sorry

end opposite_absolute_values_imply_y_power_x_l1904_190470


namespace clothes_percentage_is_25_percent_l1904_190400

def monthly_income : ℝ := 90000
def household_percentage : ℝ := 0.50
def medicine_percentage : ℝ := 0.15
def savings : ℝ := 9000

theorem clothes_percentage_is_25_percent :
  let clothes_expense := monthly_income - (household_percentage * monthly_income + medicine_percentage * monthly_income + savings)
  clothes_expense / monthly_income = 0.25 := by
sorry

end clothes_percentage_is_25_percent_l1904_190400


namespace odd_divisor_of_4a_squared_minus_1_l1904_190415

theorem odd_divisor_of_4a_squared_minus_1 (n : ℤ) (h : Odd n) :
  ∃ a : ℤ, n ∣ (4 * a^2 - 1) := by
  sorry

end odd_divisor_of_4a_squared_minus_1_l1904_190415


namespace inequality_proof_l1904_190477

theorem inequality_proof (x : ℝ) (h1 : (3/2 : ℝ) ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end inequality_proof_l1904_190477


namespace coin_toss_sequences_l1904_190421

/-- The number of coin tosses in the sequence -/
def n : ℕ := 15

/-- The number of "HH" (heads followed by heads) in the sequence -/
def hh_count : ℕ := 2

/-- The number of "HT" (heads followed by tails) in the sequence -/
def ht_count : ℕ := 3

/-- The number of "TH" (tails followed by heads) in the sequence -/
def th_count : ℕ := 4

/-- The number of "TT" (tails followed by tails) in the sequence -/
def tt_count : ℕ := 5

/-- The total number of distinct sequences -/
def total_sequences : ℕ := 2522520

/-- Theorem stating that the number of distinct sequences of n coin tosses
    with exactly hh_count "HH", ht_count "HT", th_count "TH", and tt_count "TT"
    is equal to total_sequences -/
theorem coin_toss_sequences :
  (Nat.factorial (n - 1)) / (Nat.factorial hh_count * Nat.factorial ht_count *
  Nat.factorial th_count * Nat.factorial tt_count) = total_sequences := by
  sorry

end coin_toss_sequences_l1904_190421


namespace family_eating_habits_l1904_190405

theorem family_eating_habits (only_veg only_nonveg total_veg : ℕ) 
  (h1 : only_veg = 13)
  (h2 : only_nonveg = 8)
  (h3 : total_veg = 19) :
  total_veg - only_veg = 6 :=
by
  sorry

end family_eating_habits_l1904_190405


namespace pushups_total_l1904_190496

def zachary_pushups : ℕ := 47

def david_pushups (zachary : ℕ) : ℕ := zachary + 15

def emily_pushups (david : ℕ) : ℕ := 2 * david

def total_pushups (zachary david emily : ℕ) : ℕ := zachary + david + emily

theorem pushups_total :
  total_pushups zachary_pushups (david_pushups zachary_pushups) (emily_pushups (david_pushups zachary_pushups)) = 233 := by
  sorry

end pushups_total_l1904_190496


namespace hannahs_farm_animals_l1904_190489

/-- The total number of animals on Hannah's farm -/
def total_animals (num_pigs : ℕ) : ℕ :=
  let num_cows := 2 * num_pigs - 3
  let num_goats := num_cows + 6
  num_pigs + num_cows + num_goats

/-- Theorem stating the total number of animals on Hannah's farm -/
theorem hannahs_farm_animals :
  total_animals 10 = 50 := by
  sorry

end hannahs_farm_animals_l1904_190489


namespace shopping_cart_deletion_l1904_190409

theorem shopping_cart_deletion (initial_items final_items : ℕ) 
  (h1 : initial_items = 18) 
  (h2 : final_items = 8) : 
  initial_items - final_items = 10 := by
  sorry

end shopping_cart_deletion_l1904_190409


namespace complex_simplification_l1904_190493

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_simplification :
  5 * (1 + i^3) / ((2 + i) * (2 - i)) = 1 - i :=
by sorry

end complex_simplification_l1904_190493


namespace tom_typing_speed_l1904_190403

theorem tom_typing_speed (words_per_page : ℕ) (pages_typed : ℕ) (minutes_taken : ℕ) :
  words_per_page = 450 →
  pages_typed = 10 →
  minutes_taken = 50 →
  (words_per_page * pages_typed) / minutes_taken = 90 := by
  sorry

end tom_typing_speed_l1904_190403


namespace correct_operation_l1904_190461

theorem correct_operation : 
  (-2^2 ≠ 4) ∧ 
  ((-2)^3 ≠ -6) ∧ 
  ((-1/2)^3 = -1/8) ∧ 
  ((-7/3)^3 ≠ -8/27) :=
by
  sorry

end correct_operation_l1904_190461


namespace max_integer_a_for_real_roots_l1904_190497

theorem max_integer_a_for_real_roots : 
  ∀ a : ℤ, 
    a ≠ 1 →
    (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 3 = 0) →
    a ≤ 0 :=
by sorry

end max_integer_a_for_real_roots_l1904_190497


namespace imaginary_part_of_z_l1904_190486

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) : 
  z.im = 1 := by sorry

end imaginary_part_of_z_l1904_190486


namespace square_roots_problem_l1904_190450

theorem square_roots_problem (n : ℝ) (a : ℝ) (h1 : n > 0) 
  (h2 : (a + 3) ^ 2 = n) (h3 : (2 * a + 3) ^ 2 = n) : n = 9 := by
  sorry

end square_roots_problem_l1904_190450


namespace triangle_inradius_l1904_190447

/-- Given a triangle with perimeter 28 cm and area 28 cm², prove that its inradius is 2 cm -/
theorem triangle_inradius (p A r : ℝ) (h1 : p = 28) (h2 : A = 28) (h3 : A = r * p / 2) : r = 2 := by
  sorry

end triangle_inradius_l1904_190447


namespace distance_between_A_and_B_l1904_190451

def point_A : Fin 3 → ℝ := ![1, 2, 3]
def point_B : Fin 3 → ℝ := ![-1, 3, -2]

theorem distance_between_A_and_B : 
  Real.sqrt (((point_B 0 - point_A 0)^2 + (point_B 1 - point_A 1)^2 + (point_B 2 - point_A 2)^2 : ℝ)) = Real.sqrt 30 := by
  sorry

end distance_between_A_and_B_l1904_190451


namespace parabola_roots_difference_l1904_190414

def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_roots_difference (a b c : ℝ) :
  (∃ (y : ℝ → ℝ), ∀ x, y x = parabola a b c x) →
  (∃ h k, ∀ x, parabola a b c x = a * (x - h)^2 + k) →
  (parabola a b c 2 = -4) →
  (parabola a b c 4 = 12) →
  (∃ m n, m > n ∧ parabola a b c m = 0 ∧ parabola a b c n = 0) →
  (∃ m n, m > n ∧ parabola a b c m = 0 ∧ parabola a b c n = 0 ∧ m - n = 2) :=
by sorry

end parabola_roots_difference_l1904_190414


namespace cube_space_diagonal_l1904_190413

theorem cube_space_diagonal (surface_area : ℝ) (h : surface_area = 64) :
  let side_length := Real.sqrt (surface_area / 6)
  let space_diagonal := side_length * Real.sqrt 3
  space_diagonal = 4 * Real.sqrt 2 := by
  sorry

end cube_space_diagonal_l1904_190413


namespace exponent_division_l1904_190467

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end exponent_division_l1904_190467


namespace power_two_ge_product_l1904_190417

theorem power_two_ge_product (m n : ℕ) : 2^(m+n-2) ≥ m*n := by
  sorry

end power_two_ge_product_l1904_190417


namespace two_face_painted_count_l1904_190412

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  painted_faces : ℕ := 6

/-- Represents a painted cube cut into unit cubes -/
structure CutPaintedCube (n : ℕ) extends PaintedCube n

/-- The number of unit cubes with at least two painted faces in a cut painted cube -/
def num_two_face_painted (c : CutPaintedCube 4) : ℕ := 32

theorem two_face_painted_count (c : CutPaintedCube 4) : 
  num_two_face_painted c = 32 := by sorry

end two_face_painted_count_l1904_190412


namespace circles_externally_tangent_l1904_190494

/-- Circle C₁ with center (4, 0) and radius 3 -/
def C₁ : Set (ℝ × ℝ) := {p | (p.1 - 4)^2 + p.2^2 = 9}

/-- Circle C₂ with center (0, 3) and radius 2 -/
def C₂ : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 3)^2 = 4}

/-- The center of C₁ -/
def center₁ : ℝ × ℝ := (4, 0)

/-- The center of C₂ -/
def center₂ : ℝ × ℝ := (0, 3)

/-- The radius of C₁ -/
def radius₁ : ℝ := 3

/-- The radius of C₂ -/
def radius₂ : ℝ := 2

/-- Theorem: C₁ and C₂ are externally tangent -/
theorem circles_externally_tangent :
  (center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2 = (radius₁ + radius₂)^2 :=
by sorry

end circles_externally_tangent_l1904_190494


namespace num_available_sandwiches_l1904_190492

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different kinds of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different kinds of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents whether turkey is available. -/
def turkey_available : Prop := True

/-- Represents whether salami is available. -/
def salami_available : Prop := True

/-- Represents whether Swiss cheese is available. -/
def swiss_cheese_available : Prop := True

/-- Represents whether rye bread is available. -/
def rye_bread_available : Prop := True

/-- Represents the number of sandwiches with turkey/Swiss cheese combination. -/
def turkey_swiss_combinations : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread/salami combination. -/
def rye_salami_combinations : ℕ := num_cheeses

/-- Calculates the total number of possible sandwich combinations. -/
def total_combinations : ℕ := num_breads * num_meats * num_cheeses

/-- Theorem stating the number of different sandwiches a customer can order. -/
theorem num_available_sandwiches : 
  total_combinations - turkey_swiss_combinations - rye_salami_combinations = 199 := by
  sorry

end num_available_sandwiches_l1904_190492


namespace greeting_card_exchange_l1904_190488

theorem greeting_card_exchange (n : ℕ) (h : n * (n - 1) = 90) : n = 10 := by
  sorry

end greeting_card_exchange_l1904_190488


namespace sugar_solution_sweetness_l1904_190434

theorem sugar_solution_sweetness (a b m : ℝ) 
  (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : a / b < (a + m) / (b + m) := by
  sorry

end sugar_solution_sweetness_l1904_190434


namespace number_ratio_problem_l1904_190464

theorem number_ratio_problem (x y z : ℝ) : 
  x + y + z = 110 →
  y = 30 →
  z = (1/3) * x →
  x / y = 2 := by
sorry

end number_ratio_problem_l1904_190464


namespace largest_m_base_10_l1904_190401

theorem largest_m_base_10 (m : ℕ) (A B C : ℕ) : 
  m > 0 ∧ 
  m = 25 * A + 5 * B + C ∧ 
  m = 81 * C + 9 * B + A ∧ 
  A < 5 ∧ B < 5 ∧ C < 5 ∧
  A < 9 ∧ B < 9 ∧ C < 9 →
  m ≤ 61 := by
sorry

end largest_m_base_10_l1904_190401


namespace triangle_tangent_sum_product_l1904_190437

/-- Given a triangle ABC with angles α, β, and γ, 
    the sum of the tangents of these angles equals the product of their tangents. -/
theorem triangle_tangent_sum_product (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) : 
  Real.tan α + Real.tan β + Real.tan γ = Real.tan α * Real.tan β * Real.tan γ := by
  sorry

end triangle_tangent_sum_product_l1904_190437


namespace product_of_divisors_36_l1904_190471

theorem product_of_divisors_36 (n : Nat) (h : n = 36) :
  (Finset.prod (Finset.filter (· ∣ n) (Finset.range (n + 1))) id) = 10077696 := by
  sorry

end product_of_divisors_36_l1904_190471


namespace grocery_payment_proof_l1904_190404

def grocery_cost (soup_cans bread_loaves cereal_boxes milk_gallons apples cookie_bags olive_oil : ℕ)
  (soup_price bread_price cereal_price milk_price apple_price cookie_price oil_price : ℕ) : ℕ :=
  soup_cans * soup_price + bread_loaves * bread_price + cereal_boxes * cereal_price +
  milk_gallons * milk_price + apples * apple_price + cookie_bags * cookie_price + olive_oil * oil_price

def min_bills_needed (total_cost bill_value : ℕ) : ℕ :=
  (total_cost + bill_value - 1) / bill_value

theorem grocery_payment_proof :
  let total_cost := grocery_cost 6 3 4 2 7 5 1 2 5 3 4 1 3 8
  min_bills_needed total_cost 20 = 4 := by
  sorry

end grocery_payment_proof_l1904_190404


namespace tan_half_sum_l1904_190431

theorem tan_half_sum (p q : Real) 
  (h1 : Real.cos p + Real.cos q = 1/3)
  (h2 : Real.sin p + Real.sin q = 4/9) : 
  Real.tan ((p + q) / 2) = 4/3 := by
  sorry

end tan_half_sum_l1904_190431


namespace sets_A_B_properties_l1904_190445

theorem sets_A_B_properties (p q : ℝ) (h : p * q ≠ 0) :
  (∀ x₀ : ℝ, 9^x₀ + p * 3^x₀ + q = 0 → q * 9^(-x₀) + p * 3^(-x₀) + 1 = 0) ∧
  (∃ p q : ℝ, 
    (∃ x : ℝ, 9^x + p * 3^x + q = 0 ∧ q * 9^x + p * 3^x + 1 = 0) ∧
    (∀ x : ℝ, x ≠ 1 → 9^x + p * 3^x + q = 0 → q * 9^x + p * 3^x + 1 ≠ 0) ∧
    (9^1 + p * 3^1 + q = 0 ∧ q * 9^1 + p * 3^1 + 1 = 0) ∧
    p = -4 ∧ q = 3) :=
by sorry

end sets_A_B_properties_l1904_190445


namespace system_solution_and_equality_l1904_190416

theorem system_solution_and_equality (a b c : ℝ) (h : a * b * c ≠ 0) :
  ∃! (x y z : ℝ),
    (b * z + c * y = a ∧ c * x + a * z = b ∧ a * y + b * x = c) ∧
    (x = (b^2 + c^2 - a^2) / (2 * b * c) ∧
     y = (c^2 + a^2 - b^2) / (2 * a * c) ∧
     z = (a^2 + b^2 - c^2) / (2 * a * b)) ∧
    ((1 - x^2) / a^2 = (1 - y^2) / b^2 ∧ (1 - y^2) / b^2 = (1 - z^2) / c^2) :=
by sorry

end system_solution_and_equality_l1904_190416


namespace cos_2x_over_cos_pi_4_plus_x_l1904_190438

theorem cos_2x_over_cos_pi_4_plus_x (x : Real) 
  (h1 : x ∈ Set.Ioo 0 (π/4)) 
  (h2 : Real.sin (π/4 - x) = 5/13) : 
  Real.cos (2*x) / Real.cos (π/4 + x) = 24/13 := by
  sorry

end cos_2x_over_cos_pi_4_plus_x_l1904_190438


namespace divisors_of_8n_cubed_l1904_190465

def is_product_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n = p * q

def count_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisors_of_8n_cubed (n : ℕ) 
  (h1 : is_product_of_two_primes n)
  (h2 : count_divisors n = 22)
  (h3 : Odd n) :
  count_divisors (8 * n^3) = 496 := by
  sorry

end divisors_of_8n_cubed_l1904_190465


namespace line_perpendicular_implies_planes_perpendicular_l1904_190466

-- Define the structure for a plane
structure Plane :=
  (points : Set Point)

-- Define the structure for a line
structure Line :=
  (points : Set Point)

-- Define the perpendicular relation between a line and a plane
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- Define the contained relation between a line and a plane
def contained (l : Line) (p : Plane) : Prop := sorry

-- Define the perpendicular relation between two planes
def perpendicularPlanes (p1 p2 : Plane) : Prop := sorry

-- Theorem statement
theorem line_perpendicular_implies_planes_perpendicular 
  (α β : Plane) (m : Line) 
  (h_distinct : α ≠ β) 
  (h_perp : perpendicular m β) 
  (h_contained : contained m α) : 
  perpendicularPlanes α β := by sorry

end line_perpendicular_implies_planes_perpendicular_l1904_190466


namespace max_rectangle_area_l1904_190418

theorem max_rectangle_area (perimeter : ℕ) (min_diff : ℕ) : perimeter = 160 → min_diff = 10 → ∃ (length width : ℕ), 
  length + width = perimeter / 2 ∧ 
  length ≥ width + min_diff ∧
  ∀ (l w : ℕ), l + w = perimeter / 2 → l ≥ w + min_diff → l * w ≤ length * width ∧
  length * width = 1575 := by
  sorry

end max_rectangle_area_l1904_190418


namespace contrapositive_statement_1_contrapositive_statement_2_negation_statement_3_sufficient_condition_statement_4_l1904_190459

-- Statement 1
theorem contrapositive_statement_1 :
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔ 
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) :=
sorry

-- Statement 2
theorem contrapositive_statement_2 :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔
  (∀ m : ℝ, ¬(∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

-- Statement 3
theorem negation_statement_3 :
  ¬(∃ x₀ : ℝ, x₀ > 1 ∧ x₀^2 - 2*x₀ - 3 = 0) ↔
  (∀ x : ℝ, x > 1 → x^2 - 2*x - 3 ≠ 0) :=
sorry

-- Statement 4
theorem sufficient_condition_statement_4 (a : ℝ) :
  (∀ x : ℝ, -2 < x ∧ x < -1 → (x + a)*(x + 1) < 0) →
  a > 2 :=
sorry

end contrapositive_statement_1_contrapositive_statement_2_negation_statement_3_sufficient_condition_statement_4_l1904_190459


namespace subset_implies_m_equals_one_l1904_190469

theorem subset_implies_m_equals_one (m : ℝ) : 
  let A : Set ℝ := {-1, 2, 2*m-1}
  let B : Set ℝ := {2, m^2}
  B ⊆ A → m = 1 := by
sorry

end subset_implies_m_equals_one_l1904_190469


namespace triangle_side_ratio_l1904_190439

theorem triangle_side_ratio (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : b + c ≤ 2 * a) (h5 : c + a ≤ 2 * b) (h6 : a < b + c) (h7 : b < c + a) :
  2 / 3 < b / a ∧ b / a < 3 / 2 := by
  sorry

end triangle_side_ratio_l1904_190439


namespace banana_arrangements_l1904_190425

def word_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2
def b_count : ℕ := 1

theorem banana_arrangements : 
  (word_length.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end banana_arrangements_l1904_190425


namespace probability_first_odd_given_two_odd_one_even_l1904_190436

/-- Represents the outcome of drawing a ball -/
inductive BallOutcome
  | Odd
  | Even

/-- Represents the result of drawing three balls -/
structure ThreeBallDraw where
  first : BallOutcome
  second : BallOutcome
  third : BallOutcome

def is_valid_draw (draw : ThreeBallDraw) : Prop :=
  (draw.first = BallOutcome.Odd ∧ draw.second = BallOutcome.Odd ∧ draw.third = BallOutcome.Even) ∨
  (draw.first = BallOutcome.Odd ∧ draw.second = BallOutcome.Even ∧ draw.third = BallOutcome.Odd)

def probability_first_odd (total_balls : ℕ) (odd_balls : ℕ) : ℚ :=
  (odd_balls : ℚ) / (total_balls : ℚ)

theorem probability_first_odd_given_two_odd_one_even 
  (total_balls : ℕ) (odd_balls : ℕ) (h1 : total_balls = 100) (h2 : odd_balls = 50) :
  probability_first_odd total_balls odd_balls = 1/4 :=
sorry

end probability_first_odd_given_two_odd_one_even_l1904_190436


namespace clarinet_cost_is_125_l1904_190426

def initial_savings : ℕ := 10
def price_per_book : ℕ := 5
def total_books_sold : ℕ := 25

def clarinet_cost : ℕ := total_books_sold * price_per_book

theorem clarinet_cost_is_125 : clarinet_cost = 125 := by
  sorry

end clarinet_cost_is_125_l1904_190426


namespace junk_mail_distribution_l1904_190407

/-- Proves that given 48 pieces of junk mail and 8 houses, each house will receive 6 pieces of junk mail. -/
theorem junk_mail_distribution (total_mail : ℕ) (num_houses : ℕ) (h1 : total_mail = 48) (h2 : num_houses = 8) :
  total_mail / num_houses = 6 := by
sorry

end junk_mail_distribution_l1904_190407


namespace ratio_proof_l1904_190455

theorem ratio_proof (a b : ℝ) (h : (a - 3*b) / (2*a - b) = 0.14285714285714285) : 
  a/b = 4 := by sorry

end ratio_proof_l1904_190455


namespace percentage_increase_girls_to_total_l1904_190495

def boys : ℕ := 2000
def girls : ℕ := 5000

theorem percentage_increase_girls_to_total : 
  (((boys + girls) - girls : ℚ) / girls) * 100 = 40 := by
  sorry

end percentage_increase_girls_to_total_l1904_190495


namespace f_derivative_at_zero_l1904_190440

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem f_derivative_at_zero :
  deriv f 0 = 1 := by sorry

end f_derivative_at_zero_l1904_190440


namespace equal_perimeters_l1904_190483

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Circle : Type :=
  (center : Point)
  (radius : ℝ)

-- Define the quadrilateral ABCD
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry
def D : Point := sorry

-- Define the inscribed circle and its center I
def inscribedCircle : Circle := sorry
def I : Point := inscribedCircle.center

-- Define the circumcircle ω of triangle ACI
def ω : Circle := sorry

-- Define points X, Y, Z, T on ω
def X : Point := sorry
def Y : Point := sorry
def Z : Point := sorry
def T : Point := sorry

-- Define a function to calculate the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define a function to calculate the perimeter of a quadrilateral
def perimeter (p q r s : Point) : ℝ :=
  distance p q + distance q r + distance r s + distance s p

-- State the theorem
theorem equal_perimeters :
  perimeter A D T X = perimeter C D Y Z := by sorry

end equal_perimeters_l1904_190483


namespace face_moisturizer_cost_l1904_190487

/-- Proves that the cost of each face moisturizer is $50 given the problem conditions -/
theorem face_moisturizer_cost (tanya_face_moisturizer_cost : ℝ) 
  (h1 : 2 * (2 * tanya_face_moisturizer_cost + 4 * 60) = 2 * tanya_face_moisturizer_cost + 4 * 60 + 1020) : 
  tanya_face_moisturizer_cost = 50 := by
  sorry

end face_moisturizer_cost_l1904_190487


namespace average_math_score_l1904_190452

def june_score : ℝ := 94.5
def patty_score : ℝ := 87.5
def josh_score : ℝ := 99.75
def henry_score : ℝ := 95.5
def lucy_score : ℝ := 91
def mark_score : ℝ := 97.25

def num_children : ℕ := 6

theorem average_math_score :
  (june_score + patty_score + josh_score + henry_score + lucy_score + mark_score) / num_children = 94.25 := by
  sorry

end average_math_score_l1904_190452


namespace sequence_sum_l1904_190499

theorem sequence_sum (P Q R S T U V : ℝ) : 
  S = 7 ∧ 
  P + Q + R = 21 ∧ 
  Q + R + S = 21 ∧ 
  R + S + T = 21 ∧ 
  S + T + U = 21 ∧ 
  T + U + V = 21 → 
  P + V = 14 := by
sorry

end sequence_sum_l1904_190499


namespace max_profit_at_max_price_max_profit_value_l1904_190480

/-- Represents the souvenir selling scenario with given conditions -/
structure SouvenirSales where
  cost_price : ℝ := 6
  base_price : ℝ := 8
  base_sales : ℝ := 200
  price_sales_ratio : ℝ := 10
  max_price : ℝ := 12

/-- Calculates daily sales based on selling price -/
def daily_sales (s : SouvenirSales) (x : ℝ) : ℝ :=
  s.base_sales - s.price_sales_ratio * (x - s.base_price)

/-- Calculates daily profit based on selling price -/
def daily_profit (s : SouvenirSales) (x : ℝ) : ℝ :=
  (x - s.cost_price) * (daily_sales s x)

/-- Theorem stating the maximum profit occurs at the maximum allowed price -/
theorem max_profit_at_max_price (s : SouvenirSales) :
  ∀ x, s.cost_price ≤ x ∧ x ≤ s.max_price →
    daily_profit s x ≤ daily_profit s s.max_price :=
sorry

/-- Theorem stating the value of the maximum profit -/
theorem max_profit_value (s : SouvenirSales) :
  daily_profit s s.max_price = 960 :=
sorry

end max_profit_at_max_price_max_profit_value_l1904_190480


namespace revenue_decrease_l1904_190458

theorem revenue_decrease (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let new_tax := 0.7 * T
  let new_consumption := 1.2 * C
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  (original_revenue - new_revenue) / original_revenue = 0.16 := by
  sorry

end revenue_decrease_l1904_190458

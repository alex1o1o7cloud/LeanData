import Mathlib

namespace running_gender_related_l636_63610

structure RunningData where
  total_students : Nat
  male_students : Nat
  female_like_running : Nat
  male_dislike_running : Nat

def chi_square (data : RunningData) : Rat :=
  let female_students := data.total_students - data.male_students
  let male_like_running := data.male_students - data.male_dislike_running
  let female_dislike_running := female_students - data.female_like_running
  let n := data.total_students
  let a := male_like_running
  let b := data.male_dislike_running
  let c := data.female_like_running
  let d := female_dislike_running
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

def is_gender_related (data : RunningData) : Prop :=
  chi_square data > 6635 / 1000

theorem running_gender_related (data : RunningData) 
  (h1 : data.total_students = 200)
  (h2 : data.male_students = 120)
  (h3 : data.female_like_running = 30)
  (h4 : data.male_dislike_running = 50) :
  is_gender_related data := by
  sorry

#eval chi_square { total_students := 200, male_students := 120, female_like_running := 30, male_dislike_running := 50 }

end running_gender_related_l636_63610


namespace three_digit_divisible_by_eight_l636_63609

theorem three_digit_divisible_by_eight :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ (n / 100) % 10 = 5 ∧ n % 8 = 0 ∧ n = 533 := by
  sorry

end three_digit_divisible_by_eight_l636_63609


namespace amp_neg_eight_five_l636_63673

-- Define the & operation
def amp (a b : ℝ) : ℝ := (a + b) * (a - b)

-- State the theorem
theorem amp_neg_eight_five : amp (-8 : ℝ) 5 = 39 := by
  sorry

end amp_neg_eight_five_l636_63673


namespace sum_positive_implies_one_positive_l636_63618

theorem sum_positive_implies_one_positive (a b : ℝ) : 
  a + b > 0 → a > 0 ∨ b > 0 := by
  sorry

end sum_positive_implies_one_positive_l636_63618


namespace quadratic_always_positive_l636_63629

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 2 > 0 := by
  sorry

end quadratic_always_positive_l636_63629


namespace anna_meal_cost_difference_l636_63647

theorem anna_meal_cost_difference : 
  let bagel_cost : ℚ := 95/100
  let orange_juice_cost : ℚ := 85/100
  let sandwich_cost : ℚ := 465/100
  let milk_cost : ℚ := 115/100
  let breakfast_cost := bagel_cost + orange_juice_cost
  let lunch_cost := sandwich_cost + milk_cost
  lunch_cost - breakfast_cost = 4
  := by sorry

end anna_meal_cost_difference_l636_63647


namespace store_profit_calculation_l636_63639

-- Define the types of sweaters
inductive SweaterType
| Turtleneck
| Crewneck
| Vneck

-- Define the initial cost, quantity, and markup percentages for each sweater type
def initial_cost (s : SweaterType) : ℚ :=
  match s with
  | SweaterType.Turtleneck => 30
  | SweaterType.Crewneck => 25
  | SweaterType.Vneck => 20

def quantity (s : SweaterType) : ℕ :=
  match s with
  | SweaterType.Turtleneck => 100
  | SweaterType.Crewneck => 150
  | SweaterType.Vneck => 200

def initial_markup (s : SweaterType) : ℚ :=
  match s with
  | SweaterType.Turtleneck => 0.2
  | SweaterType.Crewneck => 0.35
  | SweaterType.Vneck => 0.25

def new_year_markup (s : SweaterType) : ℚ :=
  match s with
  | SweaterType.Turtleneck => 0.25
  | SweaterType.Crewneck => 0.15
  | SweaterType.Vneck => 0.2

def february_discount (s : SweaterType) : ℚ :=
  match s with
  | SweaterType.Turtleneck => 0.09
  | SweaterType.Crewneck => 0.12
  | SweaterType.Vneck => 0.15

-- Calculate the final price for each sweater type
def final_price (s : SweaterType) : ℚ :=
  let base_price := initial_cost s * (1 + initial_markup s)
  let new_year_price := base_price + initial_cost s * new_year_markup s
  new_year_price * (1 - february_discount s)

-- Calculate the profit for each sweater type
def profit (s : SweaterType) : ℚ :=
  (final_price s - initial_cost s) * quantity s

-- Calculate the total profit
def total_profit : ℚ :=
  profit SweaterType.Turtleneck + profit SweaterType.Crewneck + profit SweaterType.Vneck

-- Theorem statement
theorem store_profit_calculation :
  total_profit = 3088.5 := by sorry

end store_profit_calculation_l636_63639


namespace goldbach_2024_l636_63660

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem goldbach_2024 : ∃ p q : ℕ, 
  p + q = 2024 ∧ 
  is_prime p ∧ 
  is_prime q ∧ 
  (p > 1000 ∨ q > 1000) :=
sorry

end goldbach_2024_l636_63660


namespace special_function_inequality_l636_63630

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  diff : Differentiable ℝ f
  f_prime_1 : deriv f 1 = 0
  condition : ∀ x : ℝ, x ≠ 1 → (x - 1) * (deriv f x) > 0

/-- Theorem stating that for any function satisfying the SpecialFunction conditions,
    f(0) + f(2) > 2f(1) -/
theorem special_function_inequality (sf : SpecialFunction) : sf.f 0 + sf.f 2 > 2 * sf.f 1 := by
  sorry

end special_function_inequality_l636_63630


namespace ellipse_equation_l636_63687

theorem ellipse_equation (a b : ℝ) (h1 : a = 6) (h2 : b = 5) (h3 : a > b) :
  ∃ (x y : ℝ), x^2 / 25 + y^2 / 36 = 1 ∧ 
  ∀ (u v : ℝ), (u^2 / b^2 + v^2 / a^2 = 1 ↔ x^2 / 25 + y^2 / 36 = 1) :=
sorry

end ellipse_equation_l636_63687


namespace min_absolute_value_at_20_l636_63608

/-- An arithmetic sequence with first term 14 and common difference -3/4 -/
def arithmeticSequence (n : ℕ) : ℚ :=
  14 + (n - 1 : ℚ) * (-3/4)

/-- The absolute value of the nth term of the arithmetic sequence -/
def absoluteValue (n : ℕ) : ℚ :=
  |arithmeticSequence n|

theorem min_absolute_value_at_20 :
  ∀ n : ℕ, n ≠ 0 → absoluteValue 20 ≤ absoluteValue n :=
sorry

end min_absolute_value_at_20_l636_63608


namespace max_profit_allocation_l636_63628

/-- Represents the allocation of raw materials to workshops --/
structure Allocation :=
  (workshop_a : ℕ)
  (workshop_b : ℕ)

/-- Calculates the profit for a given allocation --/
def profit (a : Allocation) : ℝ :=
  let total_boxes := 60
  let box_cost := 80
  let water_cost := 5
  let product_price := 30
  let workshop_a_production := 12
  let workshop_b_production := 10
  let workshop_a_water := 4
  let workshop_b_water := 2
  30 * (workshop_a_production * a.workshop_a + workshop_b_production * a.workshop_b) -
  box_cost * total_boxes -
  water_cost * (workshop_a_water * a.workshop_a + workshop_b_water * a.workshop_b)

/-- Checks if an allocation satisfies the water consumption constraint --/
def water_constraint (a : Allocation) : Prop :=
  4 * a.workshop_a + 2 * a.workshop_b ≤ 200

/-- Checks if an allocation uses exactly 60 boxes --/
def total_boxes_constraint (a : Allocation) : Prop :=
  a.workshop_a + a.workshop_b = 60

/-- The theorem stating that the given allocation maximizes profit --/
theorem max_profit_allocation :
  ∀ a : Allocation,
  water_constraint a →
  total_boxes_constraint a →
  profit a ≤ profit { workshop_a := 40, workshop_b := 20 } :=
sorry

end max_profit_allocation_l636_63628


namespace smallest_cookie_containers_six_satisfies_condition_smallest_n_is_six_l636_63672

theorem smallest_cookie_containers (n : ℕ) : (∃ k : ℕ, 15 * n - 2 = 11 * k) → n ≥ 6 := by
  sorry

theorem six_satisfies_condition : ∃ k : ℕ, 15 * 6 - 2 = 11 * k := by
  sorry

theorem smallest_n_is_six : (∃ n : ℕ, (∃ k : ℕ, 15 * n - 2 = 11 * k) ∧ (∀ m : ℕ, m < n → ¬(∃ k : ℕ, 15 * m - 2 = 11 * k))) ∧ (∃ k : ℕ, 15 * 6 - 2 = 11 * k) := by
  sorry

end smallest_cookie_containers_six_satisfies_condition_smallest_n_is_six_l636_63672


namespace equal_playing_time_l636_63633

theorem equal_playing_time (total_players : ℕ) (players_on_field : ℕ) (match_duration : ℕ) :
  total_players = 10 →
  players_on_field = 8 →
  match_duration = 45 →
  (players_on_field * match_duration) % total_players = 0 →
  (players_on_field * match_duration) / total_players = 36 :=
by
  sorry

end equal_playing_time_l636_63633


namespace initial_apps_equal_final_apps_l636_63626

/-- Proves that the initial number of apps is equal to the final number of apps -/
theorem initial_apps_equal_final_apps 
  (initial_files : ℕ) 
  (final_files : ℕ) 
  (deleted_files : ℕ) 
  (final_apps : ℕ) 
  (h1 : initial_files = 21)
  (h2 : final_files = 7)
  (h3 : deleted_files = 14)
  (h4 : final_apps = 3)
  (h5 : initial_files = final_files + deleted_files) :
  initial_files - final_files = deleted_files ∧ final_apps = 3 := by
  sorry

end initial_apps_equal_final_apps_l636_63626


namespace probability_is_one_third_l636_63614

/-- A standard die with six faces -/
def StandardDie : Type := Fin 6

/-- The total number of dots on all faces of a standard die -/
def totalDots : ℕ := 21

/-- The number of favorable outcomes (faces with 1 or 2 dots) -/
def favorableOutcomes : ℕ := 2

/-- The total number of possible outcomes (total faces) -/
def totalOutcomes : ℕ := 6

/-- The probability of the sum of dots on five faces being at least 19 -/
def probability : ℚ := favorableOutcomes / totalOutcomes

theorem probability_is_one_third :
  probability = 1 / 3 := by sorry

end probability_is_one_third_l636_63614


namespace sprocket_production_problem_l636_63678

/-- Represents a machine that produces sprockets -/
structure Machine where
  productionRate : ℝ
  timeToProduce660 : ℝ

/-- Given the conditions of the problem -/
theorem sprocket_production_problem 
  (machineA machineP machineQ : Machine)
  (h1 : machineA.productionRate = 6)
  (h2 : machineQ.productionRate = 1.1 * machineA.productionRate)
  (h3 : machineQ.timeToProduce660 = 660 / machineQ.productionRate)
  (h4 : machineP.timeToProduce660 > machineQ.timeToProduce660)
  (h5 : machineP.timeToProduce660 = machineQ.timeToProduce660 + (machineP.timeToProduce660 - machineQ.timeToProduce660)) :
  ¬ ∃ (x : ℝ), machineP.timeToProduce660 - machineQ.timeToProduce660 = x :=
sorry

end sprocket_production_problem_l636_63678


namespace all_solutions_are_valid_l636_63675

/-- A quadruple of real numbers satisfying the given conditions -/
structure Quadruple where
  x : ℝ
  y : ℝ
  z : ℝ
  w : ℝ
  sum_zero : x + y + z + w = 0
  sum_seventh_power_zero : x^7 + y^7 + z^7 + w^7 = 0

/-- Definition of a valid solution -/
def is_valid_solution (q : Quadruple) : Prop :=
  (q.x = 0 ∧ q.y = 0 ∧ q.z = 0 ∧ q.w = 0) ∨
  (q.x = -q.y ∧ q.z = -q.w) ∨
  (q.x = -q.z ∧ q.y = -q.w) ∨
  (q.x = -q.w ∧ q.y = -q.z)

/-- Main theorem: All solutions are valid -/
theorem all_solutions_are_valid (q : Quadruple) : is_valid_solution q := by
  sorry

end all_solutions_are_valid_l636_63675


namespace outfit_combinations_l636_63649

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (excluded_combinations : ℕ) : 
  shirts = 5 → pants = 6 → excluded_combinations = 1 →
  shirts * pants - excluded_combinations = 29 := by
sorry

end outfit_combinations_l636_63649


namespace circle_area_l636_63613

/-- The area of the circle defined by 3x^2 + 3y^2 - 12x + 18y + 27 = 0 is 4π. -/
theorem circle_area (x y : ℝ) : 
  (3 * x^2 + 3 * y^2 - 12 * x + 18 * y + 27 = 0) → 
  (∃ (center_x center_y radius : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 = radius^2 ∧ 
    π * radius^2 = 4 * π) := by
  sorry

end circle_area_l636_63613


namespace units_digit_is_seven_l636_63623

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_lt_10 : hundreds < 10
  tens_lt_10 : tens < 10
  units_lt_10 : units < 10
  hundreds_gt_0 : hundreds > 0

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed value of a three-digit number -/
def ThreeDigitNumber.reversed_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

/-- Theorem stating the units digit of the result is 7 -/
theorem units_digit_is_seven (n : ThreeDigitNumber) 
    (h : n.hundreds = n.units + 3) : 
    (n.reversed_value - 2 * n.value) % 10 = 7 := by
  sorry

end units_digit_is_seven_l636_63623


namespace line_equations_l636_63666

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines are parallel
def linesParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- Function to check if two lines are perpendicular
def linesPerpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem line_equations :
  let p1 : Point2D := ⟨1, 2⟩
  let p2 : Point2D := ⟨1, 1⟩
  let l1 : Line2D := ⟨1, 1, -1⟩  -- x + y - 1 = 0
  let l2 : Line2D := ⟨3, 1, -1⟩  -- 3x + y - 1 = 0
  let result1 : Line2D := ⟨1, 1, -3⟩  -- x + y - 3 = 0
  let result2 : Line2D := ⟨1, -3, 2⟩  -- x - 3y + 2 = 0
  (pointOnLine p1 result1 ∧ linesParallel result1 l1) ∧
  (pointOnLine p2 result2 ∧ linesPerpendicular result2 l2) :=
by sorry

end line_equations_l636_63666


namespace z_in_fourth_quadrant_l636_63698

def complex_to_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem z_in_fourth_quadrant (z : ℂ) 
  (h : (2 - 3*I)/(3 + 2*I) + z = 2 - 2*I) : 
  in_fourth_quadrant (complex_to_point z) := by
  sorry

end z_in_fourth_quadrant_l636_63698


namespace money_calculation_l636_63612

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def totalMoney (n50 : ℕ) (n500 : ℕ) : ℕ :=
  50 * n50 + 500 * n500

/-- Theorem stating that given 90 notes in total, with 77 being 50 rupee notes,
    the total amount of money is 10350 rupees -/
theorem money_calculation :
  let total_notes : ℕ := 90
  let n50 : ℕ := 77
  let n500 : ℕ := total_notes - n50
  totalMoney n50 n500 = 10350 := by
sorry

end money_calculation_l636_63612


namespace given_number_scientific_notation_l636_63696

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The given number to be converted -/
def given_number : ℝ := 0.00000164

/-- The expected scientific notation representation -/
def expected_notation : ScientificNotation := {
  coefficient := 1.64,
  exponent := -6,
  is_valid := by sorry
}

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_scientific_notation : given_number = expected_notation.coefficient * (10 : ℝ) ^ expected_notation.exponent := by
  sorry

end given_number_scientific_notation_l636_63696


namespace dans_remaining_marbles_l636_63693

theorem dans_remaining_marbles (initial_green : ℝ) (taken : ℝ) (remaining : ℝ) : 
  initial_green = 32.0 → 
  taken = 23.0 → 
  remaining = initial_green - taken → 
  remaining = 9.0 := by
  sorry

end dans_remaining_marbles_l636_63693


namespace lcm_gcd_problem_l636_63665

theorem lcm_gcd_problem :
  let a₁ := 5^2 * 7^4
  let b₁ := 490 * 175
  let a₂ := 2^5 * 3 * 7
  let b₂ := 3^4 * 5^4 * 7^2
  let c₂ := 10000
  (Nat.gcd a₁ b₁ = 8575 ∧ Nat.lcm a₁ b₁ = 600250) ∧
  (Nat.gcd a₂ (Nat.gcd b₂ c₂) = 1 ∧ Nat.lcm a₂ (Nat.lcm b₂ c₂) = 793881600) := by
  sorry

end lcm_gcd_problem_l636_63665


namespace product_of_integers_l636_63619

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 26)
  (diff_sq_eq : x^2 - y^2 = 52) : 
  x * y = 168 := by
sorry

end product_of_integers_l636_63619


namespace circle_area_equilateral_triangle_l636_63634

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 is 48π -/
theorem circle_area_equilateral_triangle : 
  ∀ (s : ℝ) (area : ℝ),
  s = 12 →  -- Side length of the equilateral triangle
  area = π * (s / Real.sqrt 3)^2 →  -- Area formula for circumscribed circle
  area = 48 * π := by
  sorry

end circle_area_equilateral_triangle_l636_63634


namespace ceiling_times_self_equals_210_l636_63685

theorem ceiling_times_self_equals_210 :
  ∃! (x : ℝ), ⌈x⌉ * x = 210 ∧ x = 14 := by sorry

end ceiling_times_self_equals_210_l636_63685


namespace smallest_c_inequality_l636_63605

theorem smallest_c_inequality (c : ℝ) : 
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → Real.sqrt (x * y) + c * Real.sqrt (|x - y|) ≥ (x + y) / 2) ↔ 
  c ≥ (1 / 2 : ℝ) :=
sorry

end smallest_c_inequality_l636_63605


namespace f_equals_g_l636_63682

/-- Given two functions f and g defined on real numbers,
    where f(x) = x^2 and g(x) = ∛(x^6),
    prove that f and g are equal for all real x. -/
theorem f_equals_g : ∀ x : ℝ, (fun x => x^2) x = (fun x => (x^6)^(1/3)) x := by
  sorry

end f_equals_g_l636_63682


namespace tank_filling_time_l636_63645

/-- Given two pipes that can fill a tank in 18 and 20 minutes respectively,
    and an outlet pipe that can empty the tank in 45 minutes,
    prove that when all pipes are opened simultaneously on an empty tank,
    it will take 12 minutes to fill the tank. -/
theorem tank_filling_time
  (pipe1 : ℝ → ℝ)
  (pipe2 : ℝ → ℝ)
  (outlet : ℝ → ℝ)
  (h1 : ∀ t, pipe1 t = t / 18)
  (h2 : ∀ t, pipe2 t = t / 20)
  (h3 : ∀ t, outlet t = t / 45)
  : ∃ t, t > 0 ∧ pipe1 t + pipe2 t - outlet t = 1 ∧ t = 12 := by
  sorry

end tank_filling_time_l636_63645


namespace inscribed_triangle_inequality_l636_63695

/-- A triangle inscribed in a circle -/
structure InscribedTriangle :=
  (A B C : ℝ × ℝ)  -- Vertices of the triangle
  (center : ℝ × ℝ)  -- Center of the circumscribed circle
  (radius : ℝ)  -- Radius of the circumscribed circle

/-- Ratio of internal angle bisector to its extension -/
def angle_bisector_ratio (t : InscribedTriangle) (v : ℝ × ℝ) : ℝ := sorry

/-- Sine of an angle in the triangle -/
def triangle_angle_sin (t : InscribedTriangle) (v : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem inscribed_triangle_inequality (t : InscribedTriangle) :
  let l_a := angle_bisector_ratio t t.A
  let l_b := angle_bisector_ratio t t.B
  let l_c := angle_bisector_ratio t t.C
  let sin_A := triangle_angle_sin t t.A
  let sin_B := triangle_angle_sin t t.B
  let sin_C := triangle_angle_sin t t.C
  l_a / (sin_A * sin_A) + l_b / (sin_B * sin_B) + l_c / (sin_C * sin_C) ≥ 3 ∧
  (l_a / (sin_A * sin_A) + l_b / (sin_B * sin_B) + l_c / (sin_C * sin_C) = 3 ↔ 
   t.A = t.B ∧ t.B = t.C) :=
by sorry

end inscribed_triangle_inequality_l636_63695


namespace josh_marbles_difference_l636_63602

theorem josh_marbles_difference (initial_marbles : ℕ) (lost_marbles : ℕ) (found_marbles : ℕ) 
  (num_friends : ℕ) (marbles_per_friend : ℕ) : 
  initial_marbles = 85 → 
  lost_marbles = 46 → 
  found_marbles = 130 → 
  num_friends = 12 → 
  marbles_per_friend = 3 → 
  found_marbles - (lost_marbles + num_friends * marbles_per_friend) = 48 := by
  sorry

end josh_marbles_difference_l636_63602


namespace janet_lives_calculation_l636_63648

theorem janet_lives_calculation (initial_lives current_lives gained_lives : ℕ) :
  initial_lives = 47 →
  current_lives = initial_lives - 23 →
  gained_lives = 46 →
  current_lives + gained_lives = 70 := by
  sorry

end janet_lives_calculation_l636_63648


namespace sum_of_21st_group_l636_63691

/-- The first number in the n-th group -/
def first_number (n : ℕ) : ℕ := n * (n - 1) / 2 + 1

/-- The last number in the n-th group -/
def last_number (n : ℕ) : ℕ := first_number n + (n - 1)

/-- The sum of numbers in the n-th group -/
def group_sum (n : ℕ) : ℕ := n * (first_number n + last_number n) / 2

/-- Theorem: The sum of numbers in the 21st group is 4641 -/
theorem sum_of_21st_group : group_sum 21 = 4641 := by sorry

end sum_of_21st_group_l636_63691


namespace a_equals_base_conversion_l636_63621

/-- Convert a natural number to a different base representation --/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- Interpret a list of digits in a given base as a natural number --/
def fromBase (digits : List ℕ) (base : ℕ) : ℕ :=
  sorry

/-- Check if a list of numbers forms an arithmetic sequence --/
def isArithmeticSequence (seq : List ℕ) : Bool :=
  sorry

/-- Define the sequence a_n as described in the problem --/
def a (p : ℕ) : ℕ → ℕ
  | n => if n < p - 1 then n else
    sorry -- Find the least positive integer not forming an arithmetic sequence

/-- Main theorem to prove --/
theorem a_equals_base_conversion {p : ℕ} (hp : Nat.Prime p) (hodd : Odd p) :
  ∀ n, a p n = fromBase (toBase n (p - 1)) p :=
  sorry

end a_equals_base_conversion_l636_63621


namespace solution_set_inequality_proof_min_value_l636_63646

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Theorem 1: Solution set of f(x) ≤ x + 1
theorem solution_set (x : ℝ) : f x ≤ x + 1 ↔ 1 ≤ x ∧ x ≤ 5 := by sorry

-- Theorem 2: Inequality proof
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a^2 / (a + 1) + b^2 / (b + 1) ≥ 1 := by sorry

-- Theorem to show that the minimum value of f(x) is 2
theorem min_value : ∀ x : ℝ, f x ≥ 2 := by sorry

end solution_set_inequality_proof_min_value_l636_63646


namespace graduation_messages_l636_63692

/-- 
Proves that for a class with x students, where each student writes a message 
for every other student, and the total number of messages is 930, 
the equation x(x-1) = 930 holds true.
-/
theorem graduation_messages (x : ℕ) 
  (h1 : x > 0) 
  (h2 : ∀ (s : ℕ), s < x → s.succ ≤ x → x - 1 = (x - s.succ) + s) 
  (h3 : (x * (x - 1)) = 930) : 
  x * (x - 1) = 930 := by
  sorry

end graduation_messages_l636_63692


namespace smallest_four_digit_multiple_of_18_l636_63658

theorem smallest_four_digit_multiple_of_18 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n → 1008 ≤ n :=
by sorry

end smallest_four_digit_multiple_of_18_l636_63658


namespace complex_on_real_axis_l636_63620

theorem complex_on_real_axis (a : ℝ) : 
  let z : ℂ := (a - Complex.I) * (1 + Complex.I)
  z.im = 0 → a = 1 := by
  sorry

end complex_on_real_axis_l636_63620


namespace intersection_distance_prove_intersection_distance_l636_63655

/-- The distance between the intersection points of y² = x and x + 2y = 10 is 2√55 -/
theorem intersection_distance : ℝ → Prop := fun d =>
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (y₁^2 = x₁) ∧ (x₁ + 2*y₁ = 10) ∧
    (y₂^2 = x₂) ∧ (x₂ + 2*y₂ = 10) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    d = ((x₂ - x₁)^2 + (y₂ - y₁)^2).sqrt ∧
    d = 2 * Real.sqrt 55

theorem prove_intersection_distance : intersection_distance (2 * Real.sqrt 55) := by
  sorry

end intersection_distance_prove_intersection_distance_l636_63655


namespace arithmetic_sequence_length_l636_63604

/-- Arithmetic sequence with given first term, last term, and common difference has the specified number of terms -/
theorem arithmetic_sequence_length
  (a₁ : ℤ)    -- First term
  (aₙ : ℤ)    -- Last term
  (d : ℤ)     -- Common difference
  (n : ℕ)     -- Number of terms
  (h1 : a₁ = -4)
  (h2 : aₙ = 32)
  (h3 : d = 3)
  (h4 : aₙ = a₁ + (n - 1) * d)  -- Formula for the nth term of an arithmetic sequence
  : n = 13 := by
  sorry

end arithmetic_sequence_length_l636_63604


namespace problem_solution_l636_63653

theorem problem_solution (x : ℕ+) (y : ℚ) 
  (h1 : x = 11 * y + 4)
  (h2 : 2 * x = 24 * y + 3) : 
  13 * y - x = 1 := by
sorry

end problem_solution_l636_63653


namespace storks_equal_other_birds_l636_63643

/-- Represents the count of different bird species on a fence --/
structure BirdCounts where
  sparrows : ℕ
  crows : ℕ
  storks : ℕ
  egrets : ℕ

/-- Calculates the final bird counts after all arrivals and departures --/
def finalBirdCounts (initial : BirdCounts) 
  (firstArrival : BirdCounts) 
  (firstDeparture : BirdCounts)
  (secondArrival : BirdCounts) : BirdCounts :=
  { sparrows := initial.sparrows + firstArrival.sparrows - firstDeparture.sparrows,
    crows := initial.crows + firstArrival.crows + secondArrival.crows,
    storks := initial.storks + firstArrival.storks + secondArrival.storks,
    egrets := firstArrival.egrets - firstDeparture.egrets }

/-- The main theorem stating that the number of storks equals the sum of all other birds --/
theorem storks_equal_other_birds : 
  let initial := BirdCounts.mk 2 1 3 0
  let firstArrival := BirdCounts.mk 1 3 6 4
  let firstDeparture := BirdCounts.mk 2 0 0 1
  let secondArrival := BirdCounts.mk 0 4 3 0
  let final := finalBirdCounts initial firstArrival firstDeparture secondArrival
  final.storks = final.sparrows + final.crows + final.egrets := by
  sorry

end storks_equal_other_birds_l636_63643


namespace original_fraction_l636_63689

theorem original_fraction (x y : ℚ) : 
  (x > 0) → (y > 0) → 
  ((6/5 * x) / (9/10 * y) = 20/21) → 
  (x / y = 10/21) := by
sorry

end original_fraction_l636_63689


namespace bernardo_receives_345_l636_63641

/-- The distribution pattern for Bernardo: 2, 5, 8, 11, ... -/
def bernardoSequence (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- The sum of the first n terms in Bernardo's sequence -/
def bernardoSum (n : ℕ) : ℕ := n * (2 * 2 + (n - 1) * 3) / 2

/-- The total amount distributed -/
def totalAmount : ℕ := 1000

theorem bernardo_receives_345 :
  ∃ n : ℕ, bernardoSum n ≤ totalAmount ∧ 
  bernardoSum (n + 1) > totalAmount ∧ 
  bernardoSum n = 345 := by sorry

end bernardo_receives_345_l636_63641


namespace m_union_n_eq_n_l636_63606

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def N : Set ℝ := {x : ℝ | x^2 < 2}

-- State the theorem
theorem m_union_n_eq_n : M ∪ N = N := by sorry

end m_union_n_eq_n_l636_63606


namespace definite_integral_tangent_fraction_l636_63686

theorem definite_integral_tangent_fraction : 
  ∫ x in (0)..(π/4), (4 - 7 * Real.tan x) / (2 + 3 * Real.tan x) = Real.log (25/8) - π/4 := by
  sorry

end definite_integral_tangent_fraction_l636_63686


namespace nine_identical_digits_multiples_l636_63603

theorem nine_identical_digits_multiples (n : ℕ) : 
  n ≥ 1 ∧ n ≤ 9 → 
  ∃ (d : ℕ), d ≥ 1 ∧ d ≤ 9 ∧ 12345679 * (9 * n) = d * 111111111 ∧
  (∀ (m : ℕ), 12345679 * m = d * 111111111 → m = 9 * n) :=
by sorry

end nine_identical_digits_multiples_l636_63603


namespace complex_division_l636_63640

theorem complex_division (i : ℂ) : i^2 = -1 → (2 + 4*i) / i = 4 - 2*i := by sorry

end complex_division_l636_63640


namespace simplify_expression_l636_63662

variables (x y : ℝ)

def A (x y : ℝ) : ℝ := x^2 + 3*x*y + y^2
def B (x y : ℝ) : ℝ := x^2 - 3*x*y + y^2

theorem simplify_expression (x y : ℝ) : 
  A x y - (B x y + 2 * B x y - (A x y + B x y)) = 12 * x * y := by
  sorry

end simplify_expression_l636_63662


namespace monotonic_decreasing_interval_l636_63652

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (x > -1 ∧ x < 3) ↔ (f' x < 0) :=
sorry

end monotonic_decreasing_interval_l636_63652


namespace remainder_problem_l636_63674

theorem remainder_problem (N : ℕ) (R : ℕ) : 
  (∃ Q : ℕ, N = 67 * Q + 1) → 
  (N = 68 * 269 + R) → 
  R = 0 := by
sorry

end remainder_problem_l636_63674


namespace inequality_region_range_l636_63667

-- Define the inequality function
def f (m x y : ℝ) : Prop := x - (m^2 - 2*m + 4)*y - 6 > 0

-- Define the theorem
theorem inequality_region_range :
  ∀ m : ℝ, (∀ x y : ℝ, f m x y → (x ≠ -1 ∨ y ≠ -1)) ↔ -1 < m ∧ m < 3 :=
by sorry

end inequality_region_range_l636_63667


namespace circle_equation_l636_63644

-- Define the line l: x - 2y - 1 = 0
def line_l (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the circle C
def circle_C (center_x center_y radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center_x)^2 + (p.2 - center_y)^2 = radius^2}

theorem circle_equation :
  ∃ (center_x center_y radius : ℝ),
    (line_l center_x center_y) ∧
    ((2 : ℝ), 1) ∈ circle_C center_x center_y radius ∧
    ((1 : ℝ), 2) ∈ circle_C center_x center_y radius ∧
    center_x = -1 ∧ center_y = -1 ∧ radius^2 = 13 :=
by sorry

end circle_equation_l636_63644


namespace solution_set_l636_63607

theorem solution_set (x : ℝ) :
  (1 / Real.pi) ^ (-x + 1) > (1 / Real.pi) ^ (x^2 - x) ↔ x < -1 ∨ x > 1 := by
  sorry

end solution_set_l636_63607


namespace stockholm_uppsala_distance_l636_63671

/-- The distance between two cities given a map distance and scale -/
def real_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Theorem: The distance between Stockholm and Uppsala is 1200 km -/
theorem stockholm_uppsala_distance :
  let map_distance : ℝ := 120
  let scale : ℝ := 10
  real_distance map_distance scale = 1200 := by
sorry

end stockholm_uppsala_distance_l636_63671


namespace lattice_point_in_diagonal_pentagon_l636_63680

/-- A point in the 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A pentagon defined by five points -/
structure Pentagon where
  a : Point
  b : Point
  c : Point
  d : Point
  e : Point

/-- Check if a pentagon is convex -/
def is_convex (p : Pentagon) : Prop := sorry

/-- Check if a point is inside or on the boundary of a polygon defined by a list of points -/
def is_inside_or_on_boundary (point : Point) (polygon : List Point) : Prop := sorry

/-- The pentagon formed by the diagonals of the given pentagon -/
def diagonal_pentagon (p : Pentagon) : List Point := sorry

theorem lattice_point_in_diagonal_pentagon (p : Pentagon) 
  (h_convex : is_convex p) : 
  ∃ (point : Point), is_inside_or_on_boundary point (diagonal_pentagon p) := by
  sorry

end lattice_point_in_diagonal_pentagon_l636_63680


namespace word_count_theorems_l636_63656

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of words we're considering -/
def word_length : ℕ := 5

/-- The number of 5-letter words -/
def num_words : ℕ := alphabet_size ^ word_length

/-- The number of 5-letter words with all different letters -/
def num_words_diff : ℕ := 
  (List.range word_length).foldl (fun acc i => acc * (alphabet_size - i)) alphabet_size

/-- The number of 5-letter words without any letter repeating consecutively -/
def num_words_no_consec : ℕ := alphabet_size * (alphabet_size - 1)^(word_length - 1)

theorem word_count_theorems : 
  (num_words = 26^5) ∧ 
  (num_words_diff = 26 * 25 * 24 * 23 * 22) ∧ 
  (num_words_no_consec = 26 * 25^4) := by
  sorry

end word_count_theorems_l636_63656


namespace count_special_numbers_l636_63670

/-- A function that returns the set of all divisors of a natural number -/
def divisors (n : ℕ) : Finset ℕ :=
  sorry

/-- A function that counts the number of divisors of a natural number -/
def divisor_count (n : ℕ) : ℕ :=
  sorry

/-- A function that counts the number of divisors of a natural number that are less than or equal to 10 -/
def divisors_leq_10_count (n : ℕ) : ℕ :=
  sorry

/-- The set of natural numbers from 1 to 100 with exactly four divisors, 
    at least three of which do not exceed 10 -/
def special_numbers : Finset ℕ :=
  sorry

theorem count_special_numbers : special_numbers.card = 8 := by
  sorry

end count_special_numbers_l636_63670


namespace photo_arrangements_l636_63659

/-- The number of arrangements for four students (two boys and two girls) in a row,
    where the two girls must stand next to each other. -/
def arrangements_count : ℕ := 12

/-- The number of ways to arrange two girls next to each other. -/
def girls_arrangement : ℕ := 2

/-- The number of ways to arrange three entities (two boys and the pair of girls). -/
def entities_arrangement : ℕ := 6

theorem photo_arrangements :
  arrangements_count = girls_arrangement * entities_arrangement :=
by sorry

end photo_arrangements_l636_63659


namespace gcd_of_72_120_168_l636_63638

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end gcd_of_72_120_168_l636_63638


namespace equation_solution_l636_63600

theorem equation_solution (x : ℝ) : 
  x ≠ 8 → x ≠ 7 → 
  ((x + 7) / (x - 8) - 6 = (5 * x - 55) / (7 - x)) ↔ x = 11 := by
  sorry

end equation_solution_l636_63600


namespace remaining_money_proof_l636_63697

def calculate_remaining_money (initial_amount : ℝ) (sparkling_water_count : ℕ) 
  (sparkling_water_price : ℝ) (sparkling_water_discount : ℝ) 
  (still_water_price : ℝ) (still_water_multiplier : ℕ) 
  (cheddar_weight : ℝ) (cheddar_price : ℝ) 
  (swiss_weight : ℝ) (swiss_price : ℝ) 
  (cheese_tax_rate : ℝ) : ℝ :=
  let sparkling_water_cost := sparkling_water_count * sparkling_water_price * (1 - sparkling_water_discount)
  let still_water_count := sparkling_water_count * still_water_multiplier
  let still_water_paid_bottles := (still_water_count / 3) * 2
  let still_water_cost := still_water_paid_bottles * still_water_price
  let cheese_cost := cheddar_weight * cheddar_price + swiss_weight * swiss_price
  let cheese_tax := cheese_cost * cheese_tax_rate
  let total_cost := sparkling_water_cost + still_water_cost + cheese_cost + cheese_tax
  initial_amount - total_cost

theorem remaining_money_proof :
  calculate_remaining_money 200 4 3 0.1 2.5 3 2.5 8.5 1.75 11 0.05 = 126.67 := by
  sorry

#eval calculate_remaining_money 200 4 3 0.1 2.5 3 2.5 8.5 1.75 11 0.05

end remaining_money_proof_l636_63697


namespace regular_heptagon_diagonal_relation_l636_63611

/-- Regular heptagon with side length a, diagonal spanning two sides c, and diagonal spanning three sides d -/
structure RegularHeptagon where
  a : ℝ  -- side length
  c : ℝ  -- length of diagonal spanning two sides
  d : ℝ  -- length of diagonal spanning three sides

/-- Theorem: In a regular heptagon, d^2 = c^2 + a^2 -/
theorem regular_heptagon_diagonal_relation (h : RegularHeptagon) : h.d^2 = h.c^2 + h.a^2 := by
  sorry

end regular_heptagon_diagonal_relation_l636_63611


namespace no_four_consecutive_power_numbers_l636_63676

theorem no_four_consecutive_power_numbers : 
  ¬ ∃ (n : ℕ), 
    (∃ (a b : ℕ) (k : ℕ), k > 1 ∧ n = a^k) ∧
    (∃ (c d : ℕ) (l : ℕ), l > 1 ∧ n + 1 = c^l) ∧
    (∃ (e f : ℕ) (m : ℕ), m > 1 ∧ n + 2 = e^m) ∧
    (∃ (g h : ℕ) (p : ℕ), p > 1 ∧ n + 3 = g^p) :=
by
  sorry


end no_four_consecutive_power_numbers_l636_63676


namespace soup_feeding_problem_l636_63679

theorem soup_feeding_problem (initial_cans : ℕ) (adults_per_can : ℕ) (children_per_can : ℕ) 
  (children_to_feed : ℕ) (adults_fed : ℕ) : 
  initial_cans = 8 → 
  adults_per_can = 4 → 
  children_per_can = 6 → 
  children_to_feed = 24 → 
  adults_fed = (initial_cans - (children_to_feed / children_per_can)) * adults_per_can → 
  adults_fed = 16 := by
sorry

end soup_feeding_problem_l636_63679


namespace investment_problem_l636_63635

theorem investment_problem (x : ℝ) : 
  (0.07 * x + 0.19 * 1500 = 0.16 * (x + 1500)) → x = 500 := by
  sorry

end investment_problem_l636_63635


namespace distance_of_opposite_numbers_a_and_neg_a_are_opposite_l636_63617

-- Define the concept of opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Define the distance from origin to a point on the number line
def distance_from_origin (a : ℝ) : ℝ := |a|

-- Statement 1: The distance from the origin to the points corresponding to two opposite numbers on the number line is equal
theorem distance_of_opposite_numbers (a : ℝ) : 
  distance_from_origin a = distance_from_origin (-a) := by sorry

-- Statement 2: For any real number a, a and -a are opposite numbers to each other
theorem a_and_neg_a_are_opposite (a : ℝ) : 
  are_opposite a (-a) := by sorry

end distance_of_opposite_numbers_a_and_neg_a_are_opposite_l636_63617


namespace circle_inequality_max_k_l636_63661

theorem circle_inequality_max_k : 
  (∃ k : ℝ, ∀ x y : ℝ, x^2 + y^2 = 1 → x + y - k ≥ 0) ∧ 
  (∀ k : ℝ, (∀ x y : ℝ, x^2 + y^2 = 1 → x + y - k ≥ 0) → k ≤ -Real.sqrt 2) :=
by sorry

end circle_inequality_max_k_l636_63661


namespace union_of_A_and_B_l636_63694

-- Define set A
def A : Set ℝ := {x | |x + 1| < 2}

-- Define set B
def B : Set ℝ := {x | -x^2 + 2*x + 3 ≥ 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x | -3 < x ∧ x ≤ 3} := by
  sorry

end union_of_A_and_B_l636_63694


namespace freddy_age_l636_63684

/-- Given the ages of several people and their relationships, prove Freddy's age. -/
theorem freddy_age (stephanie tim job oliver tina freddy : ℝ) 
  (h1 : freddy = stephanie - 2.5)
  (h2 : 3 * stephanie = job + tim)
  (h3 : tim = oliver / 2)
  (h4 : oliver / 3 = tina)
  (h5 : tina = freddy - 2)
  (h6 : job = 5)
  (h7 : oliver = job + 10) : 
  freddy = 7 := by
  sorry

end freddy_age_l636_63684


namespace congruence_system_solution_l636_63622

theorem congruence_system_solution (n : ℤ) :
  (n % 5 = 3 ∧ n % 7 = 4 ∧ n % 3 = 2) ↔ ∃ k : ℤ, n = 105 * k + 53 :=
sorry

end congruence_system_solution_l636_63622


namespace sandy_has_144_marbles_l636_63636

-- Define the number of red marbles Jessica has
def jessica_marbles : ℕ := 3 * 12

-- Define the relationship between Sandy's and Jessica's marbles
def sandy_marbles : ℕ := 4 * jessica_marbles

-- Theorem to prove
theorem sandy_has_144_marbles : sandy_marbles = 144 := by
  sorry

end sandy_has_144_marbles_l636_63636


namespace power_three_mod_seven_l636_63657

theorem power_three_mod_seven : 3^123 % 7 = 6 := by sorry

end power_three_mod_seven_l636_63657


namespace constant_k_value_l636_63683

/-- Given that -x^2 - (k + 10)x - 8 = -(x - 2)(x - 4) for all real x, prove that k = -16 -/
theorem constant_k_value (k : ℝ) 
  (h : ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4)) : 
  k = -16 := by sorry

end constant_k_value_l636_63683


namespace paving_cost_calculation_l636_63615

/-- Calculates the cost of paving a rectangular floor given its dimensions and the paving rate. -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a 5.5m by 4m room at Rs. 950 per square meter is Rs. 20,900. -/
theorem paving_cost_calculation :
  paving_cost 5.5 4 950 = 20900 := by
  sorry

end paving_cost_calculation_l636_63615


namespace repeating_decimal_subtraction_l636_63616

theorem repeating_decimal_subtraction :
  ∃ (a b c : ℚ),
    (1000 * a - a = 567) ∧
    (1000 * b - b = 234) ∧
    (1000 * c - c = 345) ∧
    (a - b - c = -4 / 333) := by
  sorry

end repeating_decimal_subtraction_l636_63616


namespace two_intersections_iff_m_values_l636_63664

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 3 * m * x + m - 1

theorem two_intersections_iff_m_values (m : ℝ) : 
  (∃! (p q : ℝ × ℝ), (p.1 = 0 ∨ p.2 = 0) ∧ (q.1 = 0 ∨ q.2 = 0) ∧ 
    p ≠ q ∧ f m p.1 = p.2 ∧ f m q.1 = q.2) ↔ 
  (m = 1 ∨ m = -5/4) := by
sorry

end two_intersections_iff_m_values_l636_63664


namespace sufficient_not_necessary_l636_63651

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 2 → x^2 > 4) ∧ (∃ x, x^2 > 4 ∧ ¬(x > 2)) :=
by sorry

end sufficient_not_necessary_l636_63651


namespace max_value_theorem_l636_63681

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 5 * a + 2 * b < 100) :
  a * b * (100 - 5 * a - 2 * b) ≤ 78125 / 36 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 5 * a₀ + 2 * b₀ < 100 ∧
    a₀ * b₀ * (100 - 5 * a₀ - 2 * b₀) = 78125 / 36 := by
  sorry

end max_value_theorem_l636_63681


namespace max_product_of_three_l636_63601

def S : Finset Int := {-10, -5, -3, 0, 2, 6, 8}

theorem max_product_of_three (a b c : Int) :
  a ∈ S → b ∈ S → c ∈ S →
  a ≠ b → b ≠ c → a ≠ c →
  a * b * c ≤ 400 ∧ 
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x * y * z = 400 :=
by sorry

end max_product_of_three_l636_63601


namespace complex_root_sum_l636_63627

theorem complex_root_sum (w : ℂ) (hw : w^4 + w^2 + 1 = 0) :
  w^120 + w^121 + w^122 + w^123 + w^124 = w - 1 := by
  sorry

end complex_root_sum_l636_63627


namespace alice_score_l636_63688

theorem alice_score : 
  let correct_answers : ℕ := 15
  let incorrect_answers : ℕ := 5
  let unattempted : ℕ := 10
  let correct_points : ℚ := 1
  let incorrect_penalty : ℚ := 1/4
  correct_answers * correct_points - incorrect_answers * incorrect_penalty = 13.75 := by
  sorry

end alice_score_l636_63688


namespace matrix_operation_example_l636_63654

-- Define the operation
def matrix_operation (a b c d : ℝ) : ℝ := a * d - b * c

-- Theorem statement
theorem matrix_operation_example :
  matrix_operation (-2) (0.5) 2 4 = -9 := by
  sorry

end matrix_operation_example_l636_63654


namespace lego_problem_l636_63625

theorem lego_problem (simon bruce kent : ℕ) : 
  simon = (bruce * 6) / 5 →  -- Simon has 20% more legos than Bruce
  bruce = kent + 20 →        -- Bruce has 20 more legos than Kent
  simon = 72 →               -- Simon has 72 legos
  kent = 40 :=               -- Kent has 40 legos
by sorry

end lego_problem_l636_63625


namespace james_potato_problem_l636_63631

/-- The problem of calculating the number of people James made potatoes for. -/
theorem james_potato_problem (pounds_per_person : ℝ) (bag_weight : ℝ) (bag_cost : ℝ) (total_spent : ℝ) :
  pounds_per_person = 1.5 →
  bag_weight = 20 →
  bag_cost = 5 →
  total_spent = 15 →
  (total_spent / bag_cost) * bag_weight / pounds_per_person = 40 := by
  sorry

#check james_potato_problem

end james_potato_problem_l636_63631


namespace prob_ace_king_heart_value_l636_63637

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Probability of drawing Ace of clubs, King of clubs, and any heart in that order -/
def prob_ace_king_heart : ℚ :=
  1 / StandardDeck *
  1 / (StandardDeck - 1) *
  NumHearts / (StandardDeck - 2)

/-- Theorem stating the probability of drawing Ace of clubs, King of clubs, and any heart -/
theorem prob_ace_king_heart_value : prob_ace_king_heart = 13 / 132600 := by
  sorry

end prob_ace_king_heart_value_l636_63637


namespace numbering_system_base_l636_63690

theorem numbering_system_base : ∃! (n : ℕ), n > 0 ∧ n^2 = 5*n + 6 := by sorry

end numbering_system_base_l636_63690


namespace strategy_D_is_best_l636_63642

/-- Represents an investment strategy --/
inductive Strategy
| A  -- Six 1-year terms
| B  -- Three 2-year terms
| C  -- Two 3-year terms
| D  -- One 5-year term followed by one 1-year term

/-- Calculates the final amount for a given strategy --/
def calculate_return (strategy : Strategy) : ℝ :=
  match strategy with
  | Strategy.A => 10000 * (1 + 0.0225)^6
  | Strategy.B => 10000 * (1 + 0.025 * 2)^3
  | Strategy.C => 10000 * (1 + 0.028 * 3)^2
  | Strategy.D => 10000 * (1 + 0.03 * 5) * (1 + 0.0225)

/-- Theorem stating that Strategy D yields the highest return --/
theorem strategy_D_is_best :
  ∀ s : Strategy, calculate_return Strategy.D ≥ calculate_return s :=
by sorry


end strategy_D_is_best_l636_63642


namespace quadratic_inequality_range_l636_63699

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0) ↔ 
  m ∈ Set.Ioo (-1/5 : ℝ) 3 ∪ {3} :=
sorry

end quadratic_inequality_range_l636_63699


namespace max_value_of_linear_combination_l636_63663

theorem max_value_of_linear_combination (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 = 9) : 
  ∃ (M : ℝ), M = 3 * Real.sqrt 14 ∧ 
  (∀ (a b c : ℝ), a^2 + b^2 + c^2 = 9 → a + 2*b + 3*c ≤ M) ∧
  (∃ (u v w : ℝ), u^2 + v^2 + w^2 = 9 ∧ u + 2*v + 3*w = M) :=
sorry

end max_value_of_linear_combination_l636_63663


namespace sales_price_calculation_l636_63650

theorem sales_price_calculation (C S : ℝ) : 
  S - C = 1.25 * C →  -- Gross profit is 125% of the cost
  S - C = 30 →        -- Gross profit is $30
  S = 54 :=           -- Sales price is $54
by sorry

end sales_price_calculation_l636_63650


namespace b_work_days_l636_63632

/-- Proves that B worked for 10 days before leaving the job -/
theorem b_work_days (a_total : ℕ) (b_total : ℕ) (a_remaining : ℕ) : 
  a_total = 21 → b_total = 15 → a_remaining = 7 → 
  ∃ (b_days : ℕ), b_days = 10 ∧ 
    (b_days : ℚ) / b_total + a_remaining / a_total = 1 :=
by sorry

end b_work_days_l636_63632


namespace melanie_plums_l636_63669

def initial_plums : ℕ := 7
def plums_given : ℕ := 3

theorem melanie_plums : initial_plums - plums_given = 4 := by
  sorry

end melanie_plums_l636_63669


namespace angle_a_measure_l636_63624

/-- An isosceles right triangle with side lengths and angles -/
structure IsoscelesRightTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  ac : ℝ
  -- Angles in radians
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  -- Properties
  ab_eq_bc : ab = bc
  right_angle_b : angle_b = Real.pi / 2
  angle_sum : angle_a + angle_b + angle_c = Real.pi

/-- The measure of angle A in an isosceles right triangle is π/4 radians (45 degrees) -/
theorem angle_a_measure (t : IsoscelesRightTriangle) : t.angle_a = Real.pi / 4 := by
  sorry

end angle_a_measure_l636_63624


namespace gathering_attendance_l636_63677

/-- The number of people who took wine -/
def W : ℕ := 26

/-- The number of people who took soda -/
def S : ℕ := 22

/-- The number of people who took juice -/
def J : ℕ := 18

/-- The number of people who took both wine and soda -/
def WS : ℕ := 17

/-- The number of people who took both wine and juice -/
def WJ : ℕ := 12

/-- The number of people who took both soda and juice -/
def SJ : ℕ := 10

/-- The number of people who took all three drinks -/
def WSJ : ℕ := 8

/-- The total number of people at the gathering -/
def total_people : ℕ := W + S + J - WS - WJ - SJ + WSJ

theorem gathering_attendance : total_people = 35 := by
  sorry

end gathering_attendance_l636_63677


namespace lcm_factor_is_twelve_l636_63668

def problem (A B X : ℕ) : Prop :=
  A > 0 ∧ B > 0 ∧
  Nat.gcd A B = 42 ∧
  A = 504 ∧
  Nat.lcm A B = 42 * X

theorem lcm_factor_is_twelve :
  ∀ A B X, problem A B X → X = 12 := by sorry

end lcm_factor_is_twelve_l636_63668

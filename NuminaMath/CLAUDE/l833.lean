import Mathlib

namespace alpha_in_third_quadrant_l833_83326

theorem alpha_in_third_quadrant (α : Real) 
  (h1 : Real.tan (α - 3 * Real.pi) > 0) 
  (h2 : Real.sin (-α + Real.pi) < 0) : 
  Real.pi < α ∧ α < 3 * Real.pi / 2 := by
  sorry

end alpha_in_third_quadrant_l833_83326


namespace prime_counterexample_l833_83343

theorem prime_counterexample : ∃ n : ℕ, 
  (Nat.Prime n ∧ ¬Nat.Prime (n + 2)) ∨ (¬Nat.Prime n ∧ Nat.Prime (n + 2)) :=
by sorry

end prime_counterexample_l833_83343


namespace isosceles_trapezoid_area_l833_83336

/-- An isosceles trapezoid with the given properties has an area of 54000/3 square centimeters -/
theorem isosceles_trapezoid_area : 
  ∀ (leg diagonal longer_base : ℝ),
  leg = 40 →
  diagonal = 50 →
  longer_base = 60 →
  ∃ (area : ℝ),
  area = 54000 / 3 ∧
  area = (longer_base + (longer_base - 2 * (Real.sqrt (leg^2 - ((100/3)^2))))) * (100/3) / 2 :=
by sorry

end isosceles_trapezoid_area_l833_83336


namespace perpendicular_vectors_x_value_l833_83337

/-- Given two vectors a and b in ℝ², prove that if a = (1, 2) and b = (x, 4) are perpendicular, then x = -8 -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 4]
  (∀ i : Fin 2, a i * b i = 0) → x = -8 := by
  sorry

end perpendicular_vectors_x_value_l833_83337


namespace two_dice_probabilities_l833_83359

/-- A fair die roll, represented as a number from 1 to 6 -/
def DieRoll : Type := Fin 6

/-- The probability space of rolling a fair die twice -/
def TwoDiceRolls : Type := DieRoll × DieRoll

/-- The probability measure for rolling a fair die twice -/
noncomputable def P : Set TwoDiceRolls → ℝ := sorry

/-- Event A: The dot product of (m, n) and (2, -2) is positive -/
def eventA : Set TwoDiceRolls :=
  {roll | (roll.1.val + 1) * 2 - (roll.2.val + 1) * 2 > 0}

/-- The region where x^2 + y^2 ≤ 16 -/
def region16 : Set TwoDiceRolls :=
  {roll | (roll.1.val + 1)^2 + (roll.2.val + 1)^2 ≤ 16}

theorem two_dice_probabilities :
  P eventA = 5/12 ∧ P region16 = 2/9 := by sorry

end two_dice_probabilities_l833_83359


namespace carlos_summer_reading_l833_83312

/-- The number of books Carlos read in summer vacation --/
def total_books : ℕ := 100

/-- The number of books Carlos read in July --/
def july_books : ℕ := 28

/-- The number of books Carlos read in August --/
def august_books : ℕ := 30

/-- The number of books Carlos read in June --/
def june_books : ℕ := total_books - (july_books + august_books)

theorem carlos_summer_reading : june_books = 42 := by
  sorry

end carlos_summer_reading_l833_83312


namespace initial_balloons_eq_sum_l833_83338

/-- The number of balloons Tom initially had -/
def initial_balloons : ℕ := 30

/-- The number of balloons Tom gave to Fred -/
def balloons_given : ℕ := 16

/-- The number of balloons Tom has left -/
def balloons_left : ℕ := 14

/-- Theorem stating that the initial number of balloons is equal to
    the sum of balloons given away and balloons left -/
theorem initial_balloons_eq_sum :
  initial_balloons = balloons_given + balloons_left := by
  sorry

end initial_balloons_eq_sum_l833_83338


namespace merchant_bought_15_keyboards_l833_83307

/-- The number of keyboards bought by a merchant -/
def num_keyboards : ℕ := 15

/-- The number of printers bought by the merchant -/
def num_printers : ℕ := 25

/-- The cost of one keyboard in dollars -/
def cost_keyboard : ℕ := 20

/-- The cost of one printer in dollars -/
def cost_printer : ℕ := 70

/-- The total cost of all items bought by the merchant in dollars -/
def total_cost : ℕ := 2050

/-- Theorem stating that the number of keyboards bought is 15 -/
theorem merchant_bought_15_keyboards :
  num_keyboards * cost_keyboard + num_printers * cost_printer = total_cost :=
sorry

end merchant_bought_15_keyboards_l833_83307


namespace lisa_baby_spoons_l833_83383

/-- Given the total number of spoons, number of children, number of decorative spoons,
    and number of spoons in the new cutlery set, calculate the number of baby spoons per child. -/
def baby_spoons_per_child (total_spoons : ℕ) (num_children : ℕ) (decorative_spoons : ℕ) (new_cutlery_spoons : ℕ) : ℕ :=
  (total_spoons - decorative_spoons - new_cutlery_spoons) / num_children

/-- Prove that given Lisa's specific situation, each child had 3 baby spoons. -/
theorem lisa_baby_spoons : baby_spoons_per_child 39 4 2 25 = 3 := by
  sorry

end lisa_baby_spoons_l833_83383


namespace garrett_granola_purchase_l833_83376

/-- Represents the cost of Garrett's granola bar purchase -/
def total_cost (oatmeal_count : ℕ) (oatmeal_price : ℚ) (peanut_count : ℕ) (peanut_price : ℚ) : ℚ :=
  oatmeal_count * oatmeal_price + peanut_count * peanut_price

/-- Proves that Garrett's total granola bar purchase cost is $19.50 -/
theorem garrett_granola_purchase :
  total_cost 6 1.25 8 1.50 = 19.50 := by
  sorry

end garrett_granola_purchase_l833_83376


namespace complex_inequality_l833_83382

theorem complex_inequality (x y a b : ℝ) 
  (h1 : x^2 + y^2 ≤ 1) 
  (h2 : a^2 + b^2 ≤ 2) : 
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ Real.sqrt 2 := by
  sorry

end complex_inequality_l833_83382


namespace tan_addition_subtraction_formulas_l833_83360

noncomputable section

open Real

def tan_add (a b : ℝ) : ℝ := (tan a + tan b) / (1 - tan a * tan b)
def tan_sub (a b : ℝ) : ℝ := (tan a - tan b) / (1 + tan a * tan b)

theorem tan_addition_subtraction_formulas (a b : ℝ) :
  (tan (a + b) = tan_add a b) ∧ (tan (a - b) = tan_sub a b) :=
sorry

end

end tan_addition_subtraction_formulas_l833_83360


namespace kamals_english_marks_l833_83314

/-- Proves that given Kamal's marks in four subjects and his average across five subjects, his marks in the fifth subject (English) are 66. -/
theorem kamals_english_marks 
  (math_marks : ℕ) 
  (physics_marks : ℕ) 
  (chemistry_marks : ℕ) 
  (biology_marks : ℕ) 
  (average_marks : ℕ) 
  (h1 : math_marks = 65)
  (h2 : physics_marks = 77)
  (h3 : chemistry_marks = 62)
  (h4 : biology_marks = 75)
  (h5 : average_marks = 69)
  (h6 : average_marks * 5 = math_marks + physics_marks + chemistry_marks + biology_marks + english_marks) :
  english_marks = 66 := by
  sorry

#check kamals_english_marks

end kamals_english_marks_l833_83314


namespace total_cost_calculation_l833_83371

def silverware_cost : ℝ := 20
def dinner_plates_cost_percentage : ℝ := 0.5

theorem total_cost_calculation :
  let dinner_plates_cost := silverware_cost * dinner_plates_cost_percentage
  let total_cost := silverware_cost + dinner_plates_cost
  total_cost = 30 :=
by
  sorry

end total_cost_calculation_l833_83371


namespace largest_x_floor_div_l833_83397

theorem largest_x_floor_div (x : ℝ) : 
  (∀ y : ℝ, (↑⌊y⌋ : ℝ) / y = 6 / 7 → y ≤ x) ↔ x = 35 / 6 :=
sorry

end largest_x_floor_div_l833_83397


namespace work_completion_time_l833_83333

theorem work_completion_time (a b c : ℝ) (h1 : a = 2 * b) (h2 : c = 3 * b) 
  (h3 : 1 / a + 1 / b + 1 / c = 1 / 18) : b = 33 := by
  sorry

end work_completion_time_l833_83333


namespace complex_calculation_l833_83344

theorem complex_calculation (z : ℂ) (h : z = 1 + I) : z - 2 / z^2 = 1 + 2*I :=
by sorry

end complex_calculation_l833_83344


namespace prob_diff_is_one_third_l833_83341

/-- The number of marbles of each color in the box -/
def marbles_per_color : ℕ := 1500

/-- The total number of marbles in the box -/
def total_marbles : ℕ := 3 * marbles_per_color

/-- The probability of drawing two marbles of the same color -/
def prob_same_color : ℚ :=
  (3 * (marbles_per_color.choose 2)) / (total_marbles.choose 2)

/-- The probability of drawing two marbles of different colors -/
def prob_diff_color : ℚ :=
  (3 * marbles_per_color * marbles_per_color) / (total_marbles.choose 2)

/-- The theorem stating that the absolute difference between the probabilities is 1/3 -/
theorem prob_diff_is_one_third :
  |prob_same_color - prob_diff_color| = 1 / 3 := by sorry

end prob_diff_is_one_third_l833_83341


namespace lawrence_county_kids_count_lawrence_county_kids_count_proof_l833_83396

theorem lawrence_county_kids_count : ℕ → ℕ → ℕ → Prop :=
  fun kids_home kids_camp total_kids =>
    kids_home = 274865 ∧ 
    kids_camp = 38608 ∧ 
    total_kids = kids_home + kids_camp → 
    total_kids = 313473

-- The proof is omitted
theorem lawrence_county_kids_count_proof : 
  ∃ (total_kids : ℕ), lawrence_county_kids_count 274865 38608 total_kids :=
sorry

end lawrence_county_kids_count_lawrence_county_kids_count_proof_l833_83396


namespace opposite_of_three_l833_83394

theorem opposite_of_three : (-(3 : ℝ)) = -3 := by
  sorry

end opposite_of_three_l833_83394


namespace solve_equation_l833_83375

theorem solve_equation : (45 : ℚ) / (8 - 3/4) = 180/29 := by
  sorry

end solve_equation_l833_83375


namespace multiplication_formula_examples_l833_83398

theorem multiplication_formula_examples :
  (203 * 197 = 39991) ∧ ((-69.9)^2 = 4886.01) := by sorry

end multiplication_formula_examples_l833_83398


namespace remainder_96_104_div_9_l833_83385

theorem remainder_96_104_div_9 : (96 * 104) % 9 = 5 := by
  sorry

end remainder_96_104_div_9_l833_83385


namespace goals_scored_theorem_l833_83353

/-- The number of goals scored by Bruce and Michael -/
def total_goals (bruce_goals : ℕ) (michael_multiplier : ℕ) : ℕ :=
  bruce_goals + michael_multiplier * bruce_goals

/-- Theorem stating that Bruce and Michael scored 16 goals in total -/
theorem goals_scored_theorem :
  total_goals 4 3 = 16 := by
  sorry

end goals_scored_theorem_l833_83353


namespace prime_divisibility_special_primes_characterization_l833_83322

theorem prime_divisibility (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p ≠ q → 
  (p + q^2) ∣ (p^2 + q) → (p + q^2) ∣ (p*q - 1) := by
  sorry

-- Part b
def special_primes : Set ℕ := {p | Nat.Prime p ∧ (p + 121) ∣ (p^2 + 11)}

-- The theorem states that the set of special primes is equal to {101, 323, 1211}
theorem special_primes_characterization : 
  special_primes = {101, 323, 1211} := by
  sorry

end prime_divisibility_special_primes_characterization_l833_83322


namespace scientific_notation_of_56_5_million_l833_83301

theorem scientific_notation_of_56_5_million :
  56500000 = 5.65 * (10 ^ 7) := by
  sorry

end scientific_notation_of_56_5_million_l833_83301


namespace total_arrangements_eq_192_l833_83351

/-- Represents the number of classes to be scheduled -/
def num_classes : ℕ := 6

/-- Represents the number of time slots in a day -/
def num_slots : ℕ := 6

/-- Represents the number of morning slots (first 4 periods) -/
def morning_slots : ℕ := 4

/-- Represents the number of afternoon slots (last 2 periods) -/
def afternoon_slots : ℕ := 2

/-- The number of ways to arrange the Chinese class in the morning -/
def chinese_arrangements : ℕ := morning_slots

/-- The number of ways to arrange the Biology class in the afternoon -/
def biology_arrangements : ℕ := afternoon_slots

/-- The number of remaining classes after scheduling Chinese and Biology -/
def remaining_classes : ℕ := num_classes - 2

/-- The number of remaining slots after scheduling Chinese and Biology -/
def remaining_slots : ℕ := num_slots - 2

/-- Calculates the total number of possible arrangements -/
def total_arrangements : ℕ :=
  chinese_arrangements * biology_arrangements * (remaining_classes.factorial)

/-- Theorem stating that the total number of arrangements is 192 -/
theorem total_arrangements_eq_192 : total_arrangements = 192 := by
  sorry

end total_arrangements_eq_192_l833_83351


namespace tower_remainder_l833_83317

/-- Represents the number of towers that can be built with cubes up to size n -/
def T : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 1 => if n ≥ 2 then 3 * T n else 2 * T n

/-- The main theorem stating the remainder when T(9) is divided by 500 -/
theorem tower_remainder : T 9 % 500 = 374 := by
  sorry

end tower_remainder_l833_83317


namespace sum_and_equality_problem_l833_83363

theorem sum_and_equality_problem (x y z : ℚ) : 
  x + y + z = 150 ∧ x + 10 = y - 10 ∧ y - 10 = 6 * z → y = 1030 / 13 := by
  sorry

end sum_and_equality_problem_l833_83363


namespace expression_bounds_l833_83389

theorem expression_bounds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (m M : ℝ),
    (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → m ≤ (3 * |x + y|) / (|x| + |y|) ∧ (3 * |x + y|) / (|x| + |y|) ≤ M) ∧
    m = 0 ∧ M = 3 :=
by sorry

end expression_bounds_l833_83389


namespace function_always_negative_implies_a_range_l833_83349

theorem function_always_negative_implies_a_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h : ∀ x ∈ Set.Ioo 0 1, f x < 0) 
  (h_def : ∀ x, f x = x * |x - a| - 2) : 
  -1 < a ∧ a < 3 := by
  sorry

end function_always_negative_implies_a_range_l833_83349


namespace odd_function_a_indeterminate_l833_83384

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_a_indeterminate (f : ℝ → ℝ) (h : OddFunction f) :
  ¬ ∃ a : ℝ, ∀ g : ℝ → ℝ, OddFunction g → g = f :=
sorry

end odd_function_a_indeterminate_l833_83384


namespace papa_carlo_solution_l833_83379

/-- Represents a time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : minutes < 60

/-- Represents a clock with its displayed time and offset -/
structure Clock where
  displayed_time : Time
  offset : Int

def Papa_Carlo_problem (clocks : Vector Clock 4) : Prop :=
  ∃ (correct_time : Time),
    (clocks.get 0).offset = -2 ∧
    (clocks.get 1).offset = -3 ∧
    (clocks.get 2).offset = 4 ∧
    (clocks.get 3).offset = 5 ∧
    (clocks.get 0).displayed_time = Time.mk 14 54 (by norm_num) ∧
    (clocks.get 1).displayed_time = Time.mk 14 57 (by norm_num) ∧
    (clocks.get 2).displayed_time = Time.mk 15 2 (by norm_num) ∧
    (clocks.get 3).displayed_time = Time.mk 15 3 (by norm_num) ∧
    correct_time = Time.mk 14 59 (by norm_num)

theorem papa_carlo_solution (clocks : Vector Clock 4) 
  (h : Papa_Carlo_problem clocks) : 
  ∃ (correct_time : Time), correct_time = Time.mk 14 59 (by norm_num) :=
by sorry

end papa_carlo_solution_l833_83379


namespace largest_integer_satisfying_inequality_l833_83318

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 3 ↔ (x : ℚ) / 4 + 3 / 7 < 4 / 3 :=
by sorry

end largest_integer_satisfying_inequality_l833_83318


namespace passing_percentage_is_36_percent_l833_83340

/-- The passing percentage for an engineering exam --/
def passing_percentage (failed_marks : ℕ) (scored_marks : ℕ) (max_marks : ℕ) : ℚ :=
  ((scored_marks + failed_marks : ℚ) / max_marks) * 100

/-- Theorem: The passing percentage is 36% --/
theorem passing_percentage_is_36_percent :
  passing_percentage 14 130 400 = 36 := by
  sorry

end passing_percentage_is_36_percent_l833_83340


namespace total_cost_calculation_l833_83319

def cost_per_pound : ℝ := 0.45

def sugar_weight : ℝ := 40
def flour_weight : ℝ := 16

def total_cost : ℝ := cost_per_pound * (sugar_weight + flour_weight)

theorem total_cost_calculation : total_cost = 25.20 := by
  sorry

end total_cost_calculation_l833_83319


namespace S_bounds_l833_83352

theorem S_bounds (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  let S := Real.sqrt (a * b / ((b + c) * (c + a))) +
           Real.sqrt (b * c / ((a + c) * (b + a))) +
           Real.sqrt (c * a / ((b + c) * (b + a)))
  1 ≤ S ∧ S ≤ 3/2 := by
  sorry

end S_bounds_l833_83352


namespace polynomial_roots_l833_83309

theorem polynomial_roots : ∃ (x : ℝ), x^5 - 3*x^4 + 3*x^2 - x - 6 = 0 ↔ x = -1 ∨ x = 1 ∨ x = 3 := by
  sorry

end polynomial_roots_l833_83309


namespace second_point_x_coordinate_l833_83386

/-- Given two points on a line, prove the x-coordinate of the second point -/
theorem second_point_x_coordinate 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) -- First point (m, n) satisfies the line equation
  (h2 : m + 2 = 2 * (n + 1) + 5) -- Second point (m+2, n+1) satisfies the line equation
  : m + 2 = 2 * n + 7 := by
  sorry

end second_point_x_coordinate_l833_83386


namespace arcade_candy_cost_l833_83313

theorem arcade_candy_cost (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candies : ℕ) :
  whack_a_mole_tickets = 8 →
  skee_ball_tickets = 7 →
  candies = 3 →
  (whack_a_mole_tickets + skee_ball_tickets) / candies = 5 :=
by sorry

end arcade_candy_cost_l833_83313


namespace inequality_proof_l833_83372

theorem inequality_proof (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  x / (1 + y) + y / (1 + x) ≤ 1 := by
  sorry

end inequality_proof_l833_83372


namespace complex_arithmetic_l833_83366

theorem complex_arithmetic : ((2 : ℂ) + 5*I + (3 : ℂ) - 2*I) - ((1 : ℂ) - 3*I) = (4 : ℂ) + 6*I :=
by sorry

end complex_arithmetic_l833_83366


namespace equivalence_complex_inequality_l833_83321

theorem equivalence_complex_inequality (a b c d : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  (∀ z : ℂ, Complex.abs (z - a) + Complex.abs (z - b) ≥ 
    Complex.abs (z - c) + Complex.abs (z - d)) ↔
  (∃ t : ℝ, t ∈ Set.Ioo 0 1 ∧ 
    c = t • a + (1 - t) • b ∧ 
    d = (1 - t) • a + t • b) :=
by sorry

end equivalence_complex_inequality_l833_83321


namespace min_value_quadratic_l833_83350

theorem min_value_quadratic (a : ℝ) : a^2 - 4*a + 9 ≥ 5 := by
  sorry

end min_value_quadratic_l833_83350


namespace solution_set_f_leq_5_smallest_a_for_inequality_l833_83377

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 3| + |x + 2|

-- Theorem 1: The solution set of f(x) ≤ 5 is [0, 2]
theorem solution_set_f_leq_5 : 
  {x : ℝ | f x ≤ 5} = Set.Icc 0 2 := by sorry

-- Theorem 2: The smallest value of a such that f(x) ≤ a - |x| for all x in [-1, 2] is 7
theorem smallest_a_for_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), x ∈ Set.Icc (-1) 2 → f x ≤ a - |x|) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x ∈ Set.Icc (-1) 2 → f x ≤ a - |x|) → a ≥ 7) := by sorry

end solution_set_f_leq_5_smallest_a_for_inequality_l833_83377


namespace parallelogram_altitude_base_ratio_l833_83354

theorem parallelogram_altitude_base_ratio 
  (area : ℝ) 
  (base : ℝ) 
  (altitude : ℝ) 
  (h1 : area = 450) 
  (h2 : base = 15) 
  (h3 : area = base * altitude) 
  (h4 : ∃ k : ℝ, altitude = k * base) : 
  altitude / base = 2 := by
sorry

end parallelogram_altitude_base_ratio_l833_83354


namespace rectangular_box_surface_area_l833_83304

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 160) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 975 := by
  sorry

end rectangular_box_surface_area_l833_83304


namespace girls_to_boys_ratio_l833_83327

theorem girls_to_boys_ratio (total : ℕ) (girls boys : ℕ) 
  (h1 : total = 36)
  (h2 : girls + boys = total)
  (h3 : girls = boys + 6) : 
  girls * 5 = boys * 7 := by
sorry

end girls_to_boys_ratio_l833_83327


namespace lizzy_shipment_cost_l833_83311

/-- Calculates the total shipment cost for Lizzy's fish shipment --/
def total_shipment_cost (total_weight type_a_capacity type_b_capacity : ℕ)
  (type_a_cost type_b_cost surcharge flat_fee : ℚ)
  (num_type_a : ℕ) : ℚ :=
  let type_a_total_weight := num_type_a * type_a_capacity
  let type_b_total_weight := total_weight - type_a_total_weight
  let num_type_b := (type_b_total_weight + type_b_capacity - 1) / type_b_capacity
  let type_a_total_cost := num_type_a * (type_a_cost + surcharge)
  let type_b_total_cost := num_type_b * (type_b_cost + surcharge)
  type_a_total_cost + type_b_total_cost + flat_fee

theorem lizzy_shipment_cost :
  total_shipment_cost 540 30 50 (3/2) (5/2) (1/2) 10 6 = 46 :=
by sorry

end lizzy_shipment_cost_l833_83311


namespace least_integer_square_triple_plus_80_l833_83310

theorem least_integer_square_triple_plus_80 :
  ∃ x : ℤ, (∀ y : ℤ, y^2 = 3*y + 80 → x ≤ y) ∧ x^2 = 3*x + 80 :=
by
  sorry

end least_integer_square_triple_plus_80_l833_83310


namespace geometric_sequence_condition_l833_83332

theorem geometric_sequence_condition (a b c : ℝ) : 
  (b^2 ≠ a*c → ¬(∃ r : ℝ, b = a*r ∧ c = b*r)) ∧ 
  (∃ a b c : ℝ, ¬(∃ r : ℝ, b = a*r ∧ c = b*r) ∧ b^2 = a*c) := by
  sorry

end geometric_sequence_condition_l833_83332


namespace quadratic_value_l833_83347

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_value (a b c : ℝ) :
  (∀ x, f a b c x ≤ 8) ∧  -- maximum value is 8
  (f a b c (-2) = 8) ∧    -- maximum occurs at x = -2
  (f a b c 1 = 4) →       -- passes through (1, 4)
  f a b c (-3) = 68/9 :=  -- value at x = -3 is 68/9
by sorry

end quadratic_value_l833_83347


namespace stockholm_uppsala_distance_l833_83392

/-- The actual distance between two cities given their distance on a map and the map's scale. -/
def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Theorem: The actual distance between Stockholm and Uppsala is 450 km. -/
theorem stockholm_uppsala_distance : 
  let map_distance : ℝ := 45
  let scale : ℝ := 10
  actual_distance map_distance scale = 450 :=
by sorry

end stockholm_uppsala_distance_l833_83392


namespace order_of_numbers_l833_83345

theorem order_of_numbers : Real.log 0.76 < 0.76 ∧ 0.76 < 60.7 := by
  sorry

end order_of_numbers_l833_83345


namespace dinner_cakes_count_l833_83346

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := 5

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served over two days -/
def total_cakes : ℕ := 14

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := total_cakes - lunch_cakes - yesterday_cakes

theorem dinner_cakes_count : dinner_cakes = 6 := by
  sorry

end dinner_cakes_count_l833_83346


namespace complete_factorization_l833_83365

theorem complete_factorization (x : ℝ) : 
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end complete_factorization_l833_83365


namespace intersection_when_a_is_3_subset_condition_l833_83357

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def N (a : ℝ) : Set ℝ := {x | x ≤ 2*a - 5}

-- Theorem for part 1
theorem intersection_when_a_is_3 : 
  M ∩ N 3 = {x | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part 2
theorem subset_condition : 
  ∀ a : ℝ, M ⊆ N a ↔ a ≥ 9/2 := by sorry

end intersection_when_a_is_3_subset_condition_l833_83357


namespace complex_modulus_equality_l833_83391

theorem complex_modulus_equality (x y : ℝ) : 
  (Complex.I + 1) * x = Complex.I * y + 1 → Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
  sorry

end complex_modulus_equality_l833_83391


namespace greatest_satisfying_n_l833_83361

-- Define the sum of the first n positive integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the factorial of n
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define the primality check
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the condition for n
def satisfies_condition (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  is_prime (n + 2) ∧
  ¬(factorial n % sum_first_n n = 0)

-- Theorem statement
theorem greatest_satisfying_n :
  satisfies_condition 995 ∧
  ∀ m, satisfies_condition m → m ≤ 995 :=
sorry

end greatest_satisfying_n_l833_83361


namespace cyclic_inequality_l833_83334

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / (a + b)) + (b^2 / (b + c)) + (c^2 / (c + a)) ≥ (a + b + c) / 2 := by
  sorry

end cyclic_inequality_l833_83334


namespace water_remaining_l833_83348

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 ∧ used = 11/4 → remaining = initial - used → remaining = 1/4 := by
  sorry

end water_remaining_l833_83348


namespace circle_tangent_sum_radii_l833_83373

theorem circle_tangent_sum_radii : 
  ∀ r : ℝ, 
  (r > 0) →
  ((r - 4)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = 12) :=
by sorry

end circle_tangent_sum_radii_l833_83373


namespace product_equals_four_l833_83325

theorem product_equals_four : 16 * 0.5 * 4 * 0.125 = 4 := by
  sorry

end product_equals_four_l833_83325


namespace subcommittee_count_l833_83381

theorem subcommittee_count (n m k t : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 5) (h4 : t = 5) :
  (Nat.choose n k) - (Nat.choose (n - t) k) = 771 := by
  sorry

end subcommittee_count_l833_83381


namespace min_value_theorem_l833_83323

theorem min_value_theorem (m : ℝ) (hm : m > 0)
  (h : ∀ x : ℝ, |x + 1| + |2*x - 1| ≥ m)
  (a b c : ℝ) (heq : a^2 + 2*b^2 + 3*c^2 = m) :
  ∀ a' b' c' : ℝ, a'^2 + 2*b'^2 + 3*c'^2 = m → a + 2*b + 3*c ≥ a' + 2*b' + 3*c' → a + 2*b + 3*c ≥ -3 :=
sorry

end min_value_theorem_l833_83323


namespace initial_number_count_l833_83364

theorem initial_number_count (n : ℕ) (S : ℝ) : 
  S / n = 62 →
  (S - 45 - 55) / (n - 2) = 62.5 →
  n = 50 := by
sorry

end initial_number_count_l833_83364


namespace log_equation_solution_l833_83358

theorem log_equation_solution (x : ℝ) : Real.log (256 : ℝ) / Real.log (3 * x) = x → x = 1 := by
  sorry

end log_equation_solution_l833_83358


namespace mk_97_check_one_l833_83302

theorem mk_97_check_one (x : ℝ) : x = 1 ↔ x ≠ 0 ∧ 4 * (x^2 - x) = 0 := by sorry

end mk_97_check_one_l833_83302


namespace sum_of_inverse_points_l833_83300

/-- Given an invertible function f, if f(a) = 3 and f(b) = 7, then a + b = 0 -/
theorem sum_of_inverse_points (f : ℝ → ℝ) (a b : ℝ) 
  (h_inv : Function.Injective f) 
  (h_a : f a = 3) 
  (h_b : f b = 7) : 
  a + b = 0 := by
  sorry

end sum_of_inverse_points_l833_83300


namespace perpendicular_parallel_implies_parallel_l833_83303

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_parallel
  (a b c : Line) (α β γ : Plane)
  (h1 : perpendicular a α)
  (h2 : perpendicular b β)
  (h3 : parallel_lines a b) :
  parallel_planes α β :=
sorry

end perpendicular_parallel_implies_parallel_l833_83303


namespace quadratic_no_real_roots_l833_83380

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + x + 1 ≠ 0 :=
sorry

end quadratic_no_real_roots_l833_83380


namespace recipe_ratio_change_l833_83328

-- Define the original recipe ratios
def original_flour : ℚ := 8
def original_water : ℚ := 4
def original_sugar : ℚ := 3

-- Define the new recipe quantities
def new_water : ℚ := 2
def new_sugar : ℚ := 6

-- Theorem statement
theorem recipe_ratio_change :
  let original_flour_sugar_ratio := original_flour / original_sugar
  let new_flour := (original_flour / original_water) * 2 * new_water
  let new_flour_sugar_ratio := new_flour / new_sugar
  original_flour_sugar_ratio - new_flour_sugar_ratio = 4 / 3 := by
  sorry

end recipe_ratio_change_l833_83328


namespace chord_length_l833_83320

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0
def C₃ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 14/5 = 0

-- Define the common chord of C₁ and C₂
def common_chord (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- Theorem statement
theorem chord_length :
  ∃ (chord_length : ℝ),
    chord_length = 4 ∧
    ∀ (x y : ℝ),
      common_chord x y →
      C₃ x y →
      (∃ (x' y' : ℝ),
        common_chord x' y' ∧
        C₃ x' y' ∧
        (x - x')^2 + (y - y')^2 = chord_length^2) :=
by sorry

end chord_length_l833_83320


namespace max_steps_to_empty_l833_83395

/-- A function that checks if a natural number has repeated digits -/
def has_repeated_digits (n : ℕ) : Bool :=
  sorry

/-- A function that represents one step of the process -/
def step (list : List ℕ) : List ℕ :=
  sorry

/-- The initial list of the first 1000 positive integers -/
def initial_list : List ℕ :=
  sorry

/-- The number of steps required to empty the list -/
def steps_to_empty (list : List ℕ) : ℕ :=
  sorry

theorem max_steps_to_empty : steps_to_empty initial_list = 11 :=
  sorry

end max_steps_to_empty_l833_83395


namespace color_tv_cost_price_l833_83339

/-- The cost price of a color TV satisfying the given conditions -/
def cost_price : ℝ := 3000

/-- The selling price before discount -/
def selling_price (cost : ℝ) : ℝ := cost * 1.4

/-- The discounted price -/
def discounted_price (price : ℝ) : ℝ := price * 0.8

/-- The profit is the difference between the discounted price and the cost price -/
def profit (cost : ℝ) : ℝ := discounted_price (selling_price cost) - cost

theorem color_tv_cost_price : 
  profit cost_price = 360 :=
sorry

end color_tv_cost_price_l833_83339


namespace shirts_per_minute_l833_83374

/-- An industrial machine that makes shirts -/
structure ShirtMachine where
  /-- The number of shirts made in 6 minutes -/
  shirts_in_6_min : ℕ
  /-- The number of minutes (6) -/
  minutes : ℕ
  /-- Assumption that the machine made 36 shirts in 6 minutes -/
  h_shirts : shirts_in_6_min = 36
  /-- Assumption that the time period is 6 minutes -/
  h_minutes : minutes = 6

/-- Theorem stating that the machine makes 6 shirts per minute -/
theorem shirts_per_minute (machine : ShirtMachine) : 
  machine.shirts_in_6_min / machine.minutes = 6 := by
  sorry

#check shirts_per_minute

end shirts_per_minute_l833_83374


namespace system_solution_l833_83308

theorem system_solution : ∃ (x y : ℝ), 2 * x + y = 7 ∧ 4 * x + 5 * y = 11 :=
by
  use 4, -1
  sorry

end system_solution_l833_83308


namespace min_ttetrominoes_on_chessboard_l833_83335

/-- Represents a chessboard as an 8x8 grid -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Represents a T-tetromino -/
structure TTetromino where
  center : Fin 8 × Fin 8
  orientation : Fin 4

/-- Checks if a T-tetromino can be placed on the board -/
def canPlaceTTetromino (board : Chessboard) (t : TTetromino) : Bool :=
  sorry

/-- Places a T-tetromino on the board -/
def placeTTetromino (board : Chessboard) (t : TTetromino) : Chessboard :=
  sorry

/-- Checks if any T-tetromino can be placed on the board -/
def canPlaceAnyTTetromino (board : Chessboard) : Bool :=
  sorry

/-- The main theorem stating that 7 is the minimum number of T-tetrominoes -/
theorem min_ttetrominoes_on_chessboard :
  ∀ (n : Nat),
    (∃ (board : Chessboard) (tetrominoes : List TTetromino),
      tetrominoes.length = n ∧
      (∀ t ∈ tetrominoes, canPlaceTTetromino board t) ∧
      ¬canPlaceAnyTTetromino (tetrominoes.foldl placeTTetromino board)) →
    n ≥ 7 :=
  sorry

end min_ttetrominoes_on_chessboard_l833_83335


namespace warriors_height_order_l833_83387

theorem warriors_height_order (heights : Set ℝ) (h : Set.Infinite heights) :
  ∃ (subseq : ℕ → ℝ), (∀ n, subseq n ∈ heights) ∧ 
    (Set.Infinite (Set.range subseq)) ∧ 
    (∀ n m, n < m → subseq n < subseq m) :=
sorry

end warriors_height_order_l833_83387


namespace equation_solution_l833_83370

theorem equation_solution : ∃ x : ℚ, (5 + 3.2 * x = 4.4 * x - 30) ∧ (x = 175 / 6) := by
  sorry

end equation_solution_l833_83370


namespace excircle_radius_eq_semiperimeter_implies_right_angle_l833_83330

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The excircle of a triangle -/
structure Excircle (T : Triangle) where
  center : Point
  radius : ℝ

/-- The semiperimeter of a triangle -/
def semiperimeter (T : Triangle) : ℝ := sorry

/-- A triangle is right-angled -/
def is_right_angled (T : Triangle) : Prop := sorry

/-- Main theorem: If the radius of the excircle equals the semiperimeter, 
    then the triangle is right-angled -/
theorem excircle_radius_eq_semiperimeter_implies_right_angle 
  (T : Triangle) (E : Excircle T) : 
  E.radius = semiperimeter T → is_right_angled T := by sorry

end excircle_radius_eq_semiperimeter_implies_right_angle_l833_83330


namespace largest_multiple_of_8_under_100_l833_83355

theorem largest_multiple_of_8_under_100 :
  ∃ n : ℕ, n * 8 = 96 ∧ n * 8 < 100 ∧ ∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96 :=
by sorry

end largest_multiple_of_8_under_100_l833_83355


namespace weight_of_b_l833_83399

def weight_problem (a b c : ℝ) : Prop :=
  (a + b + c) / 3 = 45 ∧
  (a + b) / 2 = 40 ∧
  (b + c) / 2 = 43

theorem weight_of_b (a b c : ℝ) (h : weight_problem a b c) : b = 31 := by
  sorry

end weight_of_b_l833_83399


namespace second_wing_rooms_per_hall_l833_83388

/-- Represents a hotel wing -/
structure Wing where
  floors : Nat
  hallsPerFloor : Nat
  roomsPerHall : Nat

/-- Represents a hotel with two wings -/
structure Hotel where
  wing1 : Wing
  wing2 : Wing
  totalRooms : Nat

def Hotel.secondWingRoomsPerHall (h : Hotel) : Nat :=
  (h.totalRooms - h.wing1.floors * h.wing1.hallsPerFloor * h.wing1.roomsPerHall) / 
  (h.wing2.floors * h.wing2.hallsPerFloor)

theorem second_wing_rooms_per_hall :
  let h : Hotel := {
    wing1 := { floors := 9, hallsPerFloor := 6, roomsPerHall := 32 },
    wing2 := { floors := 7, hallsPerFloor := 9, roomsPerHall := 0 }, -- roomsPerHall is unknown
    totalRooms := 4248
  }
  h.secondWingRoomsPerHall = 40 := by
  sorry

end second_wing_rooms_per_hall_l833_83388


namespace polynomial_evaluation_l833_83369

theorem polynomial_evaluation (x : ℝ) (h : x = 3) : x^6 - 3*x = 720 := by
  sorry

end polynomial_evaluation_l833_83369


namespace problem_solution_l833_83331

def A (a : ℝ) : ℝ := a + 2
def B (a : ℝ) : ℝ := 2 * a^2 - 3 * a + 10
def C (a : ℝ) : ℝ := a^2 + 5 * a - 3

theorem problem_solution :
  (∀ a : ℝ, A a < B a) ∧
  (∀ a : ℝ, (a < -5 ∨ a > 1) → C a > A a) ∧
  (∀ a : ℝ, (a = -5 ∨ a = 1) → C a = A a) ∧
  (∀ a : ℝ, (-5 < a ∧ a < 1) → C a < A a) :=
by sorry

end problem_solution_l833_83331


namespace inequality_proof_l833_83324

theorem inequality_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (a + 1/b)^2 + (b + 1/a)^2 ≥ 25/2 := by
  sorry

end inequality_proof_l833_83324


namespace snail_speed_ratio_l833_83306

-- Define the speeds and times
def speed_snail1 : ℝ := 2
def time_snail1 : ℝ := 20
def time_snail3 : ℝ := 2

-- Define the relationship between snail speeds
def speed_snail3 (speed_snail2 : ℝ) : ℝ := 5 * speed_snail2

-- Define the race distance
def race_distance : ℝ := speed_snail1 * time_snail1

-- Theorem statement
theorem snail_speed_ratio :
  ∃ (speed_snail2 : ℝ),
    speed_snail3 speed_snail2 * time_snail3 = race_distance ∧
    speed_snail2 / speed_snail1 = 2 := by
  sorry

end snail_speed_ratio_l833_83306


namespace power_inequality_l833_83356

theorem power_inequality (a b t x : ℝ) 
  (h1 : b > a) (h2 : a > 1) (h3 : t > 0) (h4 : a^x = a + t) : b^x > b + t := by
  sorry

end power_inequality_l833_83356


namespace last_three_digits_of_8_105_l833_83329

theorem last_three_digits_of_8_105 : 8^105 ≡ 992 [ZMOD 1000] := by
  sorry

end last_three_digits_of_8_105_l833_83329


namespace linear_search_average_comparisons_linear_search_most_efficient_l833_83342

/-- Represents an array with a specific size and a search function. -/
structure SearchArray (α : Type) where
  size : Nat
  elements : Fin size → α
  search : α → Option (Fin size)

/-- Calculates the average number of comparisons for a linear search. -/
def averageLinearSearchComparisons (n : Nat) : ℚ :=
  (1 + n) / 2

/-- Theorem: The average number of comparisons for a linear search
    on an array of 10,000 elements is 5,000.5 when the element is not present. -/
theorem linear_search_average_comparisons :
  averageLinearSearchComparisons 10000 = 5000.5 := by
  sorry

/-- Theorem: Linear search is the most efficient algorithm for an array
    with partial ordering that doesn't allow for more efficient searches. -/
theorem linear_search_most_efficient (α : Type) (arr : SearchArray α) :
  arr.size = 10000 →
  (∃ (p : α → Prop), ∀ (i j : Fin arr.size), i < j → p (arr.elements i) → p (arr.elements j)) →
  (∀ (search : α → Option (Fin arr.size)), 
    (∀ x, search x = arr.search x) →
    ∃ c, ∀ x, (search x).isNone → c ≥ averageLinearSearchComparisons arr.size) := by
  sorry

end linear_search_average_comparisons_linear_search_most_efficient_l833_83342


namespace keystone_arch_angle_theorem_l833_83367

/-- Represents a keystone arch made of congruent isosceles trapezoids -/
structure KeystoneArch where
  num_trapezoids : ℕ
  trapezoids_congruent : Bool
  trapezoids_isosceles : Bool
  end_trapezoids_horizontal : Bool

/-- Calculate the larger interior angle of a trapezoid in a keystone arch -/
def larger_interior_angle (arch : KeystoneArch) : ℝ :=
  if arch.num_trapezoids = 9 ∧ 
     arch.trapezoids_congruent ∧ 
     arch.trapezoids_isosceles ∧ 
     arch.end_trapezoids_horizontal
  then 100
  else 0

/-- Theorem: The larger interior angle of each trapezoid in a keystone arch 
    with 9 congruent isosceles trapezoids is 100 degrees -/
theorem keystone_arch_angle_theorem (arch : KeystoneArch) :
  arch.num_trapezoids = 9 ∧ 
  arch.trapezoids_congruent ∧ 
  arch.trapezoids_isosceles ∧ 
  arch.end_trapezoids_horizontal →
  larger_interior_angle arch = 100 := by
  sorry

end keystone_arch_angle_theorem_l833_83367


namespace arithmetic_sequence_property_l833_83315

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a, if a₁ + 3a₈ + a₁₅ = 120, then a₈ = 24 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a) 
    (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  a 8 = 24 := by
  sorry

end arithmetic_sequence_property_l833_83315


namespace unique_fixed_point_for_rotationally_invariant_function_l833_83390

-- Define a function that remains unchanged when its graph is rotated by π/2
def RotationallyInvariant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y ↔ f (-y) = x

-- Theorem statement
theorem unique_fixed_point_for_rotationally_invariant_function
  (f : ℝ → ℝ) (h : RotationallyInvariant f) :
  ∃! x : ℝ, f x = x :=
by sorry

end unique_fixed_point_for_rotationally_invariant_function_l833_83390


namespace weighted_average_problem_l833_83316

/-- Given numbers 4, 6, 8, p, q with a weighted average of 20,
    where p and q each have twice the weight of 4, 6, 8,
    prove that the average of p and q is 30.5 -/
theorem weighted_average_problem (p q : ℝ) : 
  (4 + 6 + 8 + 2*p + 2*q) / 7 = 20 →
  (p + q) / 2 = 30.5 := by
sorry

end weighted_average_problem_l833_83316


namespace polynomial_divisibility_l833_83368

theorem polynomial_divisibility (C D : ℂ) : 
  (∀ x : ℂ, x^2 - x + 1 = 0 → x^103 + C*x^2 + D*x + 1 = 0) →
  C = -1 ∧ D = 0 := by
sorry

end polynomial_divisibility_l833_83368


namespace walkway_area_is_416_l833_83305

/-- Represents a garden with flower beds and walkways -/
structure Garden where
  rows : ℕ
  columns : ℕ
  bed_width : ℝ
  bed_height : ℝ
  walkway_width : ℝ

/-- Calculates the total area of walkways in the garden -/
def walkway_area (g : Garden) : ℝ :=
  let total_width := g.columns * g.bed_width + (g.columns + 1) * g.walkway_width
  let total_height := g.rows * g.bed_height + (g.rows + 1) * g.walkway_width
  let total_area := total_width * total_height
  let bed_area := g.rows * g.columns * g.bed_width * g.bed_height
  total_area - bed_area

/-- Theorem stating that the walkway area for the given garden is 416 square feet -/
theorem walkway_area_is_416 (g : Garden) 
  (h1 : g.rows = 4)
  (h2 : g.columns = 3)
  (h3 : g.bed_width = 8)
  (h4 : g.bed_height = 3)
  (h5 : g.walkway_width = 2) : 
  walkway_area g = 416 := by
  sorry

end walkway_area_is_416_l833_83305


namespace no_quadratic_term_in_polynomial_difference_l833_83362

theorem no_quadratic_term_in_polynomial_difference (x : ℝ) :
  let p₁ := 2 * x^3 - 8 * x^2 + x - 1
  let p₂ := 3 * x^3 + 2 * m * x^2 - 5 * x + 3
  (∃ m : ℝ, ∀ a b c d : ℝ, p₁ - p₂ = a * x^3 + c * x + d) → m = -4 :=
by sorry

end no_quadratic_term_in_polynomial_difference_l833_83362


namespace max_bw_edges_grid_l833_83393

/-- Represents a square grid with corners removed and colored squares. -/
structure ColoredGrid :=
  (size : ℕ)
  (corner_size : ℕ)
  (coloring : ℕ → ℕ → Bool)

/-- Checks if a 2x2 square forms a checkerboard pattern. -/
def is_checkerboard (g : ColoredGrid) (x y : ℕ) : Prop :=
  g.coloring x y ≠ g.coloring (x+1) y ∧
  g.coloring x y ≠ g.coloring x (y+1) ∧
  g.coloring x y = g.coloring (x+1) (y+1)

/-- Counts the number of black-white edges in the grid. -/
def count_bw_edges (g : ColoredGrid) : ℕ := sorry

/-- The main theorem statement. -/
theorem max_bw_edges_grid (g : ColoredGrid) :
  g.size = 300 →
  g.corner_size = 100 →
  (∀ x y, x < g.size - g.corner_size ∧ y < g.size - g.corner_size →
    ¬is_checkerboard g x y) →
  count_bw_edges g ≤ 49998 :=
sorry

end max_bw_edges_grid_l833_83393


namespace limit_exponential_arctangent_sine_l833_83378

/-- The limit of (e^(4x) - e^(-2x)) / (2 arctan(x) - sin(x)) as x approaches 0 is 6 -/
theorem limit_exponential_arctangent_sine :
  let f : ℝ → ℝ := λ x => (Real.exp (4 * x) - Real.exp (-2 * x)) / (2 * Real.arctan x - Real.sin x)
  ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - 6| < ε :=
by
  sorry

end limit_exponential_arctangent_sine_l833_83378

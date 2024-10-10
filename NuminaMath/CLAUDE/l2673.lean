import Mathlib

namespace fred_baseball_cards_l2673_267385

def final_baseball_cards (initial : ℕ) (sold : ℕ) (traded : ℕ) (bought : ℕ) : ℕ :=
  initial - sold - traded + bought

theorem fred_baseball_cards : final_baseball_cards 25 7 3 5 = 20 := by
  sorry

end fred_baseball_cards_l2673_267385


namespace pencils_in_drawer_l2673_267344

/-- The total number of pencils after adding more -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Proof that the total number of pencils is 215 -/
theorem pencils_in_drawer : total_pencils 115 100 = 215 := by
  sorry

end pencils_in_drawer_l2673_267344


namespace functional_equation_solution_l2673_267361

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x + f (2 * x + y) + 5 * x * y = f (3 * x - y) + 2 * x^2 + 1

/-- The main theorem stating that any function satisfying the functional equation
    must have f(10) = -49 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 10 = -49 := by
  sorry

end functional_equation_solution_l2673_267361


namespace rotten_apples_l2673_267374

theorem rotten_apples (apples_per_crate : ℕ) (num_crates : ℕ) (boxes : ℕ) (apples_per_box : ℕ) :
  apples_per_crate = 180 →
  num_crates = 12 →
  boxes = 100 →
  apples_per_box = 20 →
  apples_per_crate * num_crates - boxes * apples_per_box = 160 :=
by
  sorry

end rotten_apples_l2673_267374


namespace cos_inequality_l2673_267354

theorem cos_inequality (ε x y : Real) : 
  ε > 0 → 
  x ∈ Set.Ioo (-π/4) (π/4) → 
  y ∈ Set.Ioo (-π/4) (π/4) → 
  Real.exp (x + ε) * Real.sin y = Real.exp y * Real.sin x → 
  Real.cos x ≤ Real.cos y := by
  sorry

end cos_inequality_l2673_267354


namespace polygon_sides_possibility_l2673_267366

theorem polygon_sides_possibility : ∃ n : ℕ, n ≥ 10 ∧ (n - 3) * 180 = 1620 := by
  sorry

end polygon_sides_possibility_l2673_267366


namespace equation_solutions_l2673_267302

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 3 ∧ x₂ = 2 - Real.sqrt 3 ∧
    x₁^2 - 4*x₁ + 1 = 0 ∧ x₂^2 - 4*x₂ + 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -1/2 ∧ y₂ = 2/3 ∧
    3*y₁*(2*y₁ + 1) = 4*y₁ + 2 ∧ 3*y₂*(2*y₂ + 1) = 4*y₂ + 2) :=
by sorry

end equation_solutions_l2673_267302


namespace no_seven_divisible_ones_five_l2673_267371

theorem no_seven_divisible_ones_five : ¬ ∃ (n : ℕ), (
  let num := (10^(n+1) - 10) / 9 + 5
  (num % 7 = 0) ∧ (num > 0)
) := by
  sorry

end no_seven_divisible_ones_five_l2673_267371


namespace five_balls_three_boxes_l2673_267367

/-- The number of ways to distribute n distinguishable objects into k boxes,
    where m boxes are distinguishable and (k-m) boxes are indistinguishable. -/
def distribution_count (n k m : ℕ) : ℕ :=
  k^n - (k-m)^n + ((k-m)^n / 2)

/-- The number of ways to place 5 distinguishable balls into 3 boxes,
    where one box is distinguishable (red) and the other two are indistinguishable. -/
theorem five_balls_three_boxes :
  distribution_count 5 3 1 = 227 := by
  sorry

end five_balls_three_boxes_l2673_267367


namespace alteration_cost_per_shoe_l2673_267312

-- Define the number of pairs of shoes
def num_pairs : ℕ := 14

-- Define the total cost of alteration
def total_cost : ℕ := 1036

-- Define the cost per shoe
def cost_per_shoe : ℕ := 37

-- Theorem statement
theorem alteration_cost_per_shoe :
  (total_cost : ℚ) / (2 * num_pairs) = cost_per_shoe := by
  sorry

end alteration_cost_per_shoe_l2673_267312


namespace trig_expression_equality_l2673_267317

theorem trig_expression_equality : 
  (Real.sin (24 * π / 180) * Real.cos (18 * π / 180) + Real.cos (156 * π / 180) * Real.cos (96 * π / 180)) / 
  (Real.sin (28 * π / 180) * Real.cos (12 * π / 180) + Real.cos (152 * π / 180) * Real.cos (92 * π / 180)) = 
  Real.sin (18 * π / 180) / Real.sin (26 * π / 180) := by
sorry

end trig_expression_equality_l2673_267317


namespace image_and_preimage_l2673_267387

def f (x : ℝ) : ℝ := x^2 - 2*x - 1

theorem image_and_preimage :
  (f (1 + Real.sqrt 2) = 0) ∧
  ({x : ℝ | f x = -1} = {0, 2}) := by
sorry

end image_and_preimage_l2673_267387


namespace circular_arc_length_l2673_267340

/-- The length of a circular arc with radius 10 meters and central angle 120° is 20π/3 meters. -/
theorem circular_arc_length : 
  ∀ (r : ℝ) (θ : ℝ), 
  r = 10 → 
  θ = 2 * π / 3 → 
  r * θ = 20 * π / 3 := by
sorry

end circular_arc_length_l2673_267340


namespace marys_remaining_money_l2673_267327

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 3 * p
  let total_cost := 5 * drink_cost + 2 * medium_pizza_cost + large_pizza_cost
  50 - total_cost

/-- Theorem stating that Mary's remaining money is 50 - 12p -/
theorem marys_remaining_money (p : ℝ) : money_left p = 50 - 12 * p := by
  sorry

end marys_remaining_money_l2673_267327


namespace product_of_powers_equals_square_l2673_267393

theorem product_of_powers_equals_square : (1889568 : ℕ)^2 = 3^8 * 3^12 * 2^5 * 2^10 := by
  sorry

end product_of_powers_equals_square_l2673_267393


namespace fish_population_estimate_l2673_267399

/-- Represents the number of fish in a pond given certain sampling conditions -/
def fish_in_pond (initial_caught : ℕ) (second_caught : ℕ) (marked_in_second : ℕ) : ℕ :=
  (initial_caught * second_caught) / marked_in_second

/-- Theorem stating that under given conditions, there are approximately 1200 fish in the pond -/
theorem fish_population_estimate :
  let initial_caught := 120
  let second_caught := 100
  let marked_in_second := 10
  fish_in_pond initial_caught second_caught marked_in_second = 1200 := by
  sorry

#eval fish_in_pond 120 100 10

end fish_population_estimate_l2673_267399


namespace inequality_relation_l2673_267300

theorem inequality_relation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b, a > b → 1/a < 1/b) ∧ ¬(∀ a b, 1/a < 1/b → a > b) :=
by sorry

end inequality_relation_l2673_267300


namespace max_value_operation_l2673_267337

theorem max_value_operation : 
  ∃ (max : ℕ), max = 600 ∧ 
  (∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 3 * (300 - n) ≤ max) ∧
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 3 * (300 - n) = max) :=
by sorry

end max_value_operation_l2673_267337


namespace only_classmate_exercise_comprehensive_comprehensive_investigation_survey_l2673_267362

/-- Represents a survey option -/
inductive SurveyOption
  | ClassmateExercise
  | CarCrashResistance
  | GalaViewership
  | ShoeSoleBending

/-- Defines the characteristics of a comprehensive investigation -/
def isComprehensive (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.ClassmateExercise => true
  | _ => false

/-- Theorem stating that only the classmate exercise survey is comprehensive -/
theorem only_classmate_exercise_comprehensive :
  ∀ s : SurveyOption, isComprehensive s ↔ s = SurveyOption.ClassmateExercise :=
by sorry

/-- Main theorem proving which survey is suitable for a comprehensive investigation -/
theorem comprehensive_investigation_survey :
  ∃! s : SurveyOption, isComprehensive s :=
by sorry

end only_classmate_exercise_comprehensive_comprehensive_investigation_survey_l2673_267362


namespace divisors_of_m_squared_l2673_267360

def m : ℕ := 2^42 * 3^26 * 5^12

theorem divisors_of_m_squared (d : ℕ) : 
  (d ∣ m^2) ∧ (d < m) ∧ ¬(d ∣ m) → 
  (Finset.filter (λ x => (x ∣ m^2) ∧ (x < m) ∧ ¬(x ∣ m)) (Finset.range (m + 1))).card = 38818 := by
  sorry

end divisors_of_m_squared_l2673_267360


namespace definite_integral_2x_plus_1_over_x_l2673_267329

theorem definite_integral_2x_plus_1_over_x :
  ∫ x in (1 : ℝ)..2, (2 * x + 1 / x) = 3 + Real.log 2 := by
  sorry

end definite_integral_2x_plus_1_over_x_l2673_267329


namespace union_determines_m_l2673_267392

def A (m : ℝ) : Set ℝ := {2, m}
def B (m : ℝ) : Set ℝ := {1, m^2}

theorem union_determines_m :
  ∀ m : ℝ, A m ∪ B m = {1, 2, 3, 9} → m = 3 := by
  sorry

end union_determines_m_l2673_267392


namespace unique_prime_between_squares_l2673_267320

theorem unique_prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p = n^2 + 6 ∧ 
  ∃ m : ℕ, p = (m + 1)^2 - 10 ∧
  m^2 < p ∧ p < (m + 1)^2 :=
by
  -- The proof goes here
  sorry

end unique_prime_between_squares_l2673_267320


namespace car_speed_comparison_l2673_267313

theorem car_speed_comparison (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  (2 * u * v) / (u + v) ≤ (u + v) / 2 := by
  sorry

end car_speed_comparison_l2673_267313


namespace max_value_xyz_l2673_267368

theorem max_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  x^4 * y^3 * z^2 ≤ 1024/14348907 :=
sorry

end max_value_xyz_l2673_267368


namespace steven_skittles_count_l2673_267343

/-- The number of groups of Skittles in Steven's collection -/
def num_groups : ℕ := 77

/-- The number of Skittles in each group -/
def skittles_per_group : ℕ := 77

/-- The total number of Skittles in Steven's collection -/
def total_skittles : ℕ := num_groups * skittles_per_group

theorem steven_skittles_count : total_skittles = 5929 := by
  sorry

end steven_skittles_count_l2673_267343


namespace fraction_equality_l2673_267352

theorem fraction_equality : (3+9-27+81-243+729)/(9+27-81+243-729+2187) = 1/3 := by
  sorry

end fraction_equality_l2673_267352


namespace halloween_candy_theorem_l2673_267389

/-- The amount of remaining candy after Halloween night -/
def remaining_candy (debby_candy sister_candy brother_candy eaten : ℕ) : ℕ :=
  debby_candy + sister_candy + brother_candy - eaten

/-- Theorem stating the remaining candy after Halloween night -/
theorem halloween_candy_theorem :
  remaining_candy 32 42 48 56 = 66 := by
  sorry

end halloween_candy_theorem_l2673_267389


namespace total_points_theorem_l2673_267377

/-- The total points scored by Zach and Ben in a football game -/
def total_points (zach_points ben_points : Float) : Float :=
  zach_points + ben_points

/-- Theorem stating that the total points scored by Zach and Ben is 63.0 -/
theorem total_points_theorem (zach_points ben_points : Float)
  (h1 : zach_points = 42.0)
  (h2 : ben_points = 21.0) :
  total_points zach_points ben_points = 63.0 := by
  sorry

end total_points_theorem_l2673_267377


namespace half_power_inequality_l2673_267331

theorem half_power_inequality (a : ℝ) : 
  (1/2 : ℝ)^(2*a + 1) < (1/2 : ℝ)^(3 - 2*a) → a > 1/2 :=
by
  sorry

end half_power_inequality_l2673_267331


namespace or_and_not_implies_false_and_true_l2673_267388

theorem or_and_not_implies_false_and_true (p q : Prop) :
  (p ∨ q) → (¬p) → (¬p ∧ q) := by
  sorry

end or_and_not_implies_false_and_true_l2673_267388


namespace product_xy_value_l2673_267359

/-- A parallelogram EFGH with given side lengths -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ
  is_parallelogram : EF = GH 1 ∧ FG 1 = HE

/-- The product of x and y in the given parallelogram -/
def product_xy (p : Parallelogram) (x y : ℝ) : ℝ := x * y

/-- Theorem: The product of x and y in the given parallelogram is 18 * ∛4 -/
theorem product_xy_value (p : Parallelogram) 
  (h1 : p.EF = 110)
  (h2 : p.FG = fun y => 16 * y^3)
  (h3 : p.GH = fun x => 6 * x + 2)
  (h4 : p.HE = 64)
  : ∃ x y, product_xy p x y = 18 * (4 ^ (1/3 : ℝ)) := by
  sorry

end product_xy_value_l2673_267359


namespace digit_2500_is_8_l2673_267391

/-- The number of digits in the representation of positive integers from 1 to n -/
def digitCount (n : ℕ) : ℕ := sorry

/-- The nth digit in the concatenation of integers from 1 to 1099 -/
def nthDigit (n : ℕ) : ℕ := sorry

theorem digit_2500_is_8 : nthDigit 2500 = 8 := by sorry

end digit_2500_is_8_l2673_267391


namespace product_mod_25_l2673_267310

theorem product_mod_25 :
  ∃ m : ℕ, 0 ≤ m ∧ m < 25 ∧ (123 * 156 * 198) % 25 = m ∧ m = 24 := by
  sorry

end product_mod_25_l2673_267310


namespace find_A_l2673_267322

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_single_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def round_down_hundreds (n : ℕ) : ℕ := (n / 100) * 100

theorem find_A (A : ℕ) :
  is_single_digit A →
  is_three_digit (A * 100 + 27) →
  round_down_hundreds (A * 100 + 27) = 200 →
  A = 2 := by sorry

end find_A_l2673_267322


namespace base_4_9_digit_difference_l2673_267306

def num_digits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log base n).succ

theorem base_4_9_digit_difference :
  num_digits 1234 4 - num_digits 1234 9 = 2 := by
  sorry

end base_4_9_digit_difference_l2673_267306


namespace selections_with_paperback_count_l2673_267318

/-- The number of books on the shelf -/
def total_books : ℕ := 7

/-- The number of paperback books -/
def paperbacks : ℕ := 2

/-- The number of hardback books -/
def hardbacks : ℕ := 5

/-- The number of possible selections that include at least one paperback -/
def selections_with_paperback : ℕ := 96

/-- Theorem stating that the number of selections with at least one paperback
    is equal to the total number of possible selections minus the number of
    selections with no paperbacks -/
theorem selections_with_paperback_count :
  selections_with_paperback = 2^total_books - 2^hardbacks :=
by sorry

end selections_with_paperback_count_l2673_267318


namespace paige_homework_problems_l2673_267326

/-- The initial number of homework problems Paige had -/
def initial_problems (finished : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  finished + remaining_pages * problems_per_page

/-- Theorem stating that Paige initially had 110 homework problems -/
theorem paige_homework_problems :
  initial_problems 47 7 9 = 110 := by
  sorry

end paige_homework_problems_l2673_267326


namespace line_intercepts_sum_l2673_267332

/-- Given a line with equation y + 5 = -3(x + 6), 
    the sum of its x-intercept and y-intercept is -92/3 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y + 5 = -3 * (x + 6)) → 
  (∃ x_int y_int : ℝ, 
    (y_int + 5 = -3 * (x_int + 6)) ∧ 
    (0 + 5 = -3 * (x_int + 6)) ∧ 
    (y_int + 5 = -3 * (0 + 6)) ∧ 
    (x_int + y_int = -92/3)) := by
  sorry

end line_intercepts_sum_l2673_267332


namespace a_plus_2b_plus_3c_equals_35_l2673_267390

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem a_plus_2b_plus_3c_equals_35 :
  (∀ x, f (x + 2) = 5 * x^2 + 2 * x + 6) →
  (∃ a b c, ∀ x, f x = a * x^2 + b * x + c) →
  (∃ a b c, (∀ x, f x = a * x^2 + b * x + c) ∧ a + 2 * b + 3 * c = 35) :=
by sorry

end a_plus_2b_plus_3c_equals_35_l2673_267390


namespace sandy_change_l2673_267347

def pants_cost : Float := 13.58
def shirt_cost : Float := 10.29
def sweater_cost : Float := 24.97
def shoes_cost : Float := 39.99
def paid_amount : Float := 100.00

def total_cost : Float := pants_cost + shirt_cost + sweater_cost + shoes_cost

theorem sandy_change : paid_amount - total_cost = 11.17 := by
  sorry

end sandy_change_l2673_267347


namespace mad_hatter_winning_condition_l2673_267316

/-- Represents the fraction of voters for each candidate and undecided voters -/
structure VoterFractions where
  mad_hatter : ℝ
  march_hare : ℝ
  dormouse : ℝ
  undecided : ℝ

/-- Represents the additional fraction of undecided voters each candidate receives -/
structure UndecidedAllocation where
  mad_hatter : ℝ
  march_hare : ℝ
  dormouse : ℝ

/-- The minimum fraction of undecided voters the Mad Hatter needs to secure -/
def minimum_fraction_for_mad_hatter (v : VoterFractions) : ℝ :=
  0.7

theorem mad_hatter_winning_condition 
  (v : VoterFractions)
  (h1 : v.mad_hatter = 0.2)
  (h2 : v.march_hare = 0.25)
  (h3 : v.dormouse = 0.3)
  (h4 : v.undecided = 1 - (v.mad_hatter + v.march_hare + v.dormouse))
  (h5 : v.mad_hatter + v.march_hare + v.dormouse + v.undecided = 1) :
  ∀ (u : UndecidedAllocation),
    (u.mad_hatter + u.march_hare + u.dormouse = 1) →
    (u.mad_hatter ≥ minimum_fraction_for_mad_hatter v) →
    (v.mad_hatter + v.undecided * u.mad_hatter ≥ v.march_hare + v.undecided * u.march_hare) ∧
    (v.mad_hatter + v.undecided * u.mad_hatter ≥ v.dormouse + v.undecided * u.dormouse) :=
sorry

end mad_hatter_winning_condition_l2673_267316


namespace quadratic_shift_sum_l2673_267364

/-- Given a quadratic function f(x) = 3x^2 + 2x + 1, when shifted 5 units to the right,
    the resulting function g(x) = ax^2 + bx + c satisfies a + b + c = 41 -/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, (3 * (x - 5)^2 + 2 * (x - 5) + 1) = (a * x^2 + b * x + c)) →
  a + b + c = 41 := by
sorry

end quadratic_shift_sum_l2673_267364


namespace nineteen_vectors_sum_zero_l2673_267341

theorem nineteen_vectors_sum_zero (v : Fin 19 → (Fin 3 → ZMod 3)) :
  ∃ i j k : Fin 19, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ v i + v j + v k = 0 := by
  sorry

end nineteen_vectors_sum_zero_l2673_267341


namespace matrix_equation_l2673_267307

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -7; 11, 4]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![(44/7), -(57/7); -(49/14), (63/14)]

theorem matrix_equation : N * A = B := by sorry

end matrix_equation_l2673_267307


namespace adrian_holidays_l2673_267382

/-- The number of days Adrian takes off each month -/
def days_off_per_month : ℕ := 4

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total number of holidays Adrian takes in a year -/
def total_holidays : ℕ := days_off_per_month * months_in_year

theorem adrian_holidays : total_holidays = 48 := by
  sorry

end adrian_holidays_l2673_267382


namespace faye_candy_count_l2673_267383

/-- Calculates the final number of candy pieces Faye has after eating some and receiving more. -/
def final_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) : ℕ :=
  initial - eaten + received

/-- Proves that Faye ends up with 62 pieces of candy given the initial conditions. -/
theorem faye_candy_count :
  final_candy_count 47 25 40 = 62 := by
  sorry

end faye_candy_count_l2673_267383


namespace sphere_radius_from_perpendicular_chords_l2673_267339

/-- Given a sphere with three mutually perpendicular chords APB, CPD, and EPF passing through
    a common point P, where AP = 2a, BP = 2b, CP = 2c, DP = 2d, EP = 2e, and FP = 2f,
    the radius R of the sphere is √(a² + b² + c² + d² + e² + f² - 2ab - 2cd - 2ef). -/
theorem sphere_radius_from_perpendicular_chords
  (a b c d e f : ℝ) : ∃ (R : ℝ),
  R = Real.sqrt (a^2 + b^2 + c^2 + d^2 + e^2 + f^2 - 2*a*b - 2*c*d - 2*e*f) := by
  sorry

end sphere_radius_from_perpendicular_chords_l2673_267339


namespace prob_more_ones_than_eights_l2673_267335

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The total number of possible outcomes when rolling numDice dice with numSides sides -/
def totalOutcomes : ℕ := numSides ^ numDice

/-- The number of ways to roll an equal number of 1's and 8's -/
def equalOnesAndEights : ℕ := 12276

/-- The probability of rolling more 1's than 8's when rolling numDice fair dice with numSides sides -/
def probMoreOnesThanEights : ℚ := 10246 / 32768

theorem prob_more_ones_than_eights :
  probMoreOnesThanEights = 1/2 * (1 - equalOnesAndEights / totalOutcomes) :=
sorry

end prob_more_ones_than_eights_l2673_267335


namespace smallest_candy_count_l2673_267375

theorem smallest_candy_count (x : ℕ) : 
  x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 8 = 3 ∧ 
  x % 9 = 7 ∧
  (∀ y : ℕ, y > 0 → y % 6 = 5 → y % 8 = 3 → y % 9 = 7 → x ≤ y) → 
  x = 203 := by
sorry

end smallest_candy_count_l2673_267375


namespace roots_of_quadratic_with_absolute_value_l2673_267395

theorem roots_of_quadratic_with_absolute_value
  (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (n : ℕ), n ≤ 4 ∧
  ∃ (roots : Finset ℂ), roots.card = n ∧
  ∀ z ∈ roots, a * z^2 + b * Complex.abs z + c = 0 :=
sorry

end roots_of_quadratic_with_absolute_value_l2673_267395


namespace total_word_count_180_to_220_l2673_267384

/-- Represents the word count for a number in the range [180, 220] -/
def word_count (n : ℕ) : ℕ :=
  if n = 180 then 3
  else if n ≥ 190 ∧ n ≤ 220 then 2
  else 3

/-- The sum of word counts for numbers in the range [a, b] -/
def sum_word_counts (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).sum (λ i => word_count (a + i))

theorem total_word_count_180_to_220 :
  sum_word_counts 180 220 = 99 := by
  sorry

end total_word_count_180_to_220_l2673_267384


namespace planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l2673_267379

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_plane_to : Plane → Plane → Prop)
variable (perpendicular_line_to_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: Two planes parallel to the same plane are parallel
theorem planes_parallel_to_same_plane_are_parallel
  (P Q R : Plane)
  (h1 : parallel_plane_to P R)
  (h2 : parallel_plane_to Q R) :
  parallel_planes P Q :=
sorry

-- Theorem 2: Two lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_same_plane_are_parallel
  (l1 l2 : Line) (P : Plane)
  (h1 : perpendicular_line_to_plane l1 P)
  (h2 : perpendicular_line_to_plane l2 P) :
  parallel_lines l1 l2 :=
sorry

end planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l2673_267379


namespace pen_calculation_l2673_267358

theorem pen_calculation (x y z : ℕ) (hx : x = 5) (hy : y = 20) (hz : z = 19) :
  2 * (x + y) - z = 31 := by
  sorry

end pen_calculation_l2673_267358


namespace football_team_progress_l2673_267321

def football_progress (first_play : Int) (second_play : Int) : Int :=
  let third_play := -2 * (-first_play)
  let fourth_play := third_play / 2
  first_play + second_play + third_play + fourth_play

theorem football_team_progress :
  football_progress (-5) 13 = 3 := by
  sorry

end football_team_progress_l2673_267321


namespace symmetric_point_wrt_x_axis_l2673_267319

/-- Given a point A with coordinates (-2, 3), this theorem proves that its symmetric point B
    with respect to the x-axis has coordinates (-2, -3). -/
theorem symmetric_point_wrt_x_axis :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (-2, -3)
  (A.1 = B.1) ∧ (A.2 = -B.2) := by sorry

end symmetric_point_wrt_x_axis_l2673_267319


namespace batsman_average_increase_proof_l2673_267333

def batsman_average_increase (total_innings : ℕ) (last_innings_score : ℕ) (final_average : ℚ) : ℚ :=
  let initial_total := (total_innings - 1) * (total_innings * final_average - last_innings_score) / total_innings
  let initial_average := initial_total / (total_innings - 1)
  final_average - initial_average

theorem batsman_average_increase_proof :
  batsman_average_increase 12 65 32 = 3 := by sorry

end batsman_average_increase_proof_l2673_267333


namespace quadratic_solution_sum_l2673_267330

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℂ, 7 * x^2 - 2 * x + 45 = 0 ↔ x = p + q * I) → 
  p + q^2 = 321/49 := by
  sorry

end quadratic_solution_sum_l2673_267330


namespace total_cost_proof_l2673_267394

def sandwich_price : ℚ := 245/100
def soda_price : ℚ := 87/100
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4

theorem total_cost_proof : 
  (num_sandwiches : ℚ) * sandwich_price + (num_sodas : ℚ) * soda_price = 838/100 := by
  sorry

end total_cost_proof_l2673_267394


namespace pi_approximation_after_three_tiaoRi_l2673_267381

def tiaoRiMethod (a b c d : ℕ) : ℚ := (b + d) / (a + c)

theorem pi_approximation_after_three_tiaoRi :
  let initial_lower : ℚ := 31 / 10
  let initial_upper : ℚ := 49 / 15
  let first_upper : ℚ := tiaoRiMethod 10 31 15 49
  let second_upper : ℚ := tiaoRiMethod 15 47 5 16
  let third_upper : ℚ := tiaoRiMethod 15 47 5 16
  initial_lower < Real.pi ∧ Real.pi < initial_upper →
  third_upper = 63 / 20 :=
by sorry

end pi_approximation_after_three_tiaoRi_l2673_267381


namespace triangle_abc_properties_l2673_267376

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  a = Real.sqrt 3 →
  S = Real.sqrt 3 / 2 →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  S = 1/2 * a * b * Real.sin C →
  A = π/3 ∧ b + c = 3 := by
  sorry

end triangle_abc_properties_l2673_267376


namespace fabric_equation_correct_l2673_267356

/-- Represents the fabric purchase scenario --/
structure FabricPurchase where
  total_meters : ℝ
  total_cost : ℝ
  blue_cost_per_meter : ℝ
  black_cost_per_meter : ℝ

/-- The equation correctly represents the fabric purchase scenario --/
theorem fabric_equation_correct (fp : FabricPurchase)
  (h1 : fp.total_meters = 138)
  (h2 : fp.total_cost = 540)
  (h3 : fp.blue_cost_per_meter = 3)
  (h4 : fp.black_cost_per_meter = 5) :
  ∃ x : ℝ, fp.blue_cost_per_meter * x + fp.black_cost_per_meter * (fp.total_meters - x) = fp.total_cost :=
by sorry

end fabric_equation_correct_l2673_267356


namespace children_who_got_on_bus_stop_l2673_267398

/-- The number of children who got on the bus at a stop -/
def children_who_got_on (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Proof that 38 children got on the bus at the stop -/
theorem children_who_got_on_bus_stop : children_who_got_on 26 64 = 38 := by
  sorry

end children_who_got_on_bus_stop_l2673_267398


namespace kirin_990_calculations_l2673_267380

-- Define the number of calculations per second
def calculations_per_second : ℝ := 10^11

-- Define the number of seconds
def seconds : ℝ := 2022

-- Theorem to prove
theorem kirin_990_calculations :
  calculations_per_second * seconds = 2.022 * 10^13 := by
  sorry

end kirin_990_calculations_l2673_267380


namespace tickets_distribution_l2673_267350

theorem tickets_distribution (initial_tickets best_friend_tickets schoolmate_tickets remaining_tickets : ℕ) 
  (h1 : initial_tickets = 128)
  (h2 : best_friend_tickets = 7)
  (h3 : schoolmate_tickets = 4)
  (h4 : remaining_tickets = 11)
  : ∃ (best_friends schoolmates : ℕ), 
    initial_tickets = best_friend_tickets * best_friends + schoolmate_tickets * schoolmates + remaining_tickets ∧
    best_friends + schoolmates = 20 := by
  sorry

end tickets_distribution_l2673_267350


namespace tangent_circles_expression_l2673_267311

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- The distance between the centers of two tangent circles is the sum of their radii -/
def distance (c1 c2 : Circle) : ℝ := c1.radius + c2.radius

theorem tangent_circles_expression (a b c : ℝ) (A B C : Circle)
  (ha : A.radius = a)
  (hb : B.radius = b)
  (hc : C.radius = c)
  (hab : a > b)
  (hbc : b > c)
  (htangent : A.radius + B.radius = distance A B ∧ 
              B.radius + C.radius = distance B C ∧ 
              C.radius + A.radius = distance C A) :
  distance A B + distance B C - distance C A = b ∧ b > 0 := by
  sorry

end tangent_circles_expression_l2673_267311


namespace max_x_elements_l2673_267355

/-- Represents the number of elements of each type -/
structure Elements where
  fire : ℕ
  stone : ℕ
  metal : ℕ

/-- Represents the alchemical reactions -/
def reaction1 (e : Elements) : Elements :=
  { fire := e.fire - 1, stone := e.stone - 1, metal := e.metal + 1 }

def reaction2 (e : Elements) : Elements :=
  { fire := e.fire, stone := e.stone + 2, metal := e.metal - 1 }

/-- Creates an element X -/
def createX (e : Elements) : Elements :=
  { fire := e.fire - 2, stone := e.stone - 3, metal := e.metal - 1 }

/-- The initial state of elements -/
def initialElements : Elements :=
  { fire := 50, stone := 50, metal := 0 }

/-- Checks if the number of elements is non-negative -/
def isValid (e : Elements) : Prop :=
  e.fire ≥ 0 ∧ e.stone ≥ 0 ∧ e.metal ≥ 0

/-- Theorem: The maximum number of X elements that can be created is 14 -/
theorem max_x_elements : 
  ∃ (n : ℕ) (e : Elements), 
    n = 14 ∧ 
    isValid e ∧ 
    ∀ m : ℕ, m > n → 
      ¬∃ (f : Elements), isValid f ∧ 
        (∃ (seq : List (Elements → Elements)), 
          f = (seq.foldl (λ acc g => g acc) initialElements) ∧
          (createX^[m]) f = f) :=
sorry

end max_x_elements_l2673_267355


namespace codger_shoe_purchase_l2673_267363

/-- Represents the number of feet a sloth has -/
def sloth_feet : ℕ := 3

/-- Represents the number of shoes in a complete set for a sloth -/
def complete_set : ℕ := 3

/-- Represents the number of shoes in a pair -/
def shoes_per_pair : ℕ := 2

/-- Represents the number of complete sets Codger wants to have -/
def desired_sets : ℕ := 7

/-- Represents the number of shoes Codger already owns -/
def owned_shoes : ℕ := 3

/-- Represents the constraint that shoes must be bought in even-numbered sets of pairs -/
def even_numbered_pairs (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- The main theorem -/
theorem codger_shoe_purchase :
  ∃ (pairs_to_buy : ℕ),
    pairs_to_buy * shoes_per_pair + owned_shoes ≥ desired_sets * complete_set ∧
    even_numbered_pairs pairs_to_buy ∧
    ∀ (n : ℕ), n < pairs_to_buy →
      n * shoes_per_pair + owned_shoes < desired_sets * complete_set ∨
      ¬(even_numbered_pairs n) :=
sorry

end codger_shoe_purchase_l2673_267363


namespace sqrt_fraction_equals_seven_fifths_l2673_267396

theorem sqrt_fraction_equals_seven_fifths :
  (Real.sqrt 64 + Real.sqrt 36) / Real.sqrt (64 + 36) = 7 / 5 := by
  sorry

end sqrt_fraction_equals_seven_fifths_l2673_267396


namespace train_crossing_bridge_l2673_267369

/-- A train crossing a bridge problem -/
theorem train_crossing_bridge (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) :
  train_length = 100 →
  bridge_length = 150 →
  train_speed_kmph = 18 →
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 50 := by
  sorry

end train_crossing_bridge_l2673_267369


namespace contractor_engagement_days_l2673_267373

/-- Represents the engagement of a contractor --/
structure ContractorEngagement where
  daysWorked : ℕ
  daysAbsent : ℕ
  dailyWage : ℚ
  dailyFine : ℚ
  totalAmount : ℚ

/-- Theorem: Given the conditions, the contractor was engaged for 22 days --/
theorem contractor_engagement_days (c : ContractorEngagement) 
  (h1 : c.dailyWage = 25)
  (h2 : c.dailyFine = 7.5)
  (h3 : c.totalAmount = 490)
  (h4 : c.daysAbsent = 8) :
  c.daysWorked = 22 := by
  sorry


end contractor_engagement_days_l2673_267373


namespace rebecca_earnings_l2673_267351

def haircut_price : ℕ := 30
def perm_price : ℕ := 40
def dye_job_price : ℕ := 60
def hair_extension_price : ℕ := 80

def haircut_supply_cost : ℕ := 5
def dye_job_supply_cost : ℕ := 10
def hair_extension_supply_cost : ℕ := 25

def student_discount : ℚ := 0.1
def senior_discount : ℚ := 0.15
def first_time_discount : ℕ := 5

def num_haircuts : ℕ := 5
def num_student_haircuts : ℕ := 2
def num_perms : ℕ := 3
def num_senior_perms : ℕ := 1
def num_dye_jobs : ℕ := 2
def num_first_time_dye_jobs : ℕ := 1
def num_hair_extensions : ℕ := 1

def total_tips : ℕ := 75
def daily_expenses : ℕ := 45

theorem rebecca_earnings : 
  let total_revenue := 
    num_haircuts * haircut_price + 
    num_perms * perm_price + 
    num_dye_jobs * dye_job_price + 
    num_hair_extensions * hair_extension_price
  let total_discounts := 
    (num_student_haircuts * haircut_price * student_discount).floor +
    (num_senior_perms * perm_price * senior_discount).floor +
    (num_first_time_dye_jobs * first_time_discount)
  let supply_costs := 
    num_haircuts * haircut_supply_cost +
    num_dye_jobs * dye_job_supply_cost +
    num_hair_extensions * hair_extension_supply_cost
  let earnings := 
    total_revenue - total_discounts - supply_costs + total_tips - daily_expenses
  earnings = 413 := by sorry

end rebecca_earnings_l2673_267351


namespace fliers_remaining_l2673_267342

theorem fliers_remaining (initial_fliers : ℕ) 
  (morning_fraction : ℚ) (afternoon_fraction : ℚ) : 
  initial_fliers = 3000 →
  morning_fraction = 1/5 →
  afternoon_fraction = 1/4 →
  let remaining_after_morning := initial_fliers - (morning_fraction * initial_fliers).floor
  let final_remaining := remaining_after_morning - (afternoon_fraction * remaining_after_morning).floor
  final_remaining = 1800 := by
  sorry

end fliers_remaining_l2673_267342


namespace problem_I4_1_l2673_267305

theorem problem_I4_1 (x y : ℝ) (h : (10 * x - 3 * y) / (x + 2 * y) = 2) :
  (y + x) / (y - x) = 15 :=
by sorry

end problem_I4_1_l2673_267305


namespace sophia_estimate_l2673_267334

theorem sophia_estimate (x y a b : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : a > 0) (h4 : b > 0) :
  (x + a) - (y - b) > x - y := by
  sorry

end sophia_estimate_l2673_267334


namespace tank_capacity_l2673_267357

theorem tank_capacity (initial_fraction : ℚ) (final_fraction : ℚ) (added_water : ℚ) :
  initial_fraction = 1/4 →
  final_fraction = 3/4 →
  added_water = 200 →
  (final_fraction - initial_fraction) * (added_water / (final_fraction - initial_fraction)) = 400 :=
by sorry

end tank_capacity_l2673_267357


namespace sequence_increasing_l2673_267370

def a (n : ℕ) : ℚ := (n - 1) / (n + 1)

theorem sequence_increasing : ∀ k j : ℕ, k > j → j ≥ 1 → a k > a j := by
  sorry

end sequence_increasing_l2673_267370


namespace intersection_and_perpendicular_lines_l2673_267397

/-- Given two intersecting lines and a perpendicular line, prove the equations of a line through the intersection point and its symmetric line. -/
theorem intersection_and_perpendicular_lines
  (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop) (perp_line : ℝ → ℝ → Prop)
  (h1 : ∀ x y, line1 x y ↔ x - 2*y + 4 = 0)
  (h2 : ∀ x y, line2 x y ↔ x + y - 2 = 0)
  (h3 : ∀ x y, perp_line x y ↔ 5*x + 3*y - 6 = 0)
  (P : ℝ × ℝ) (hP : line1 P.1 P.2 ∧ line2 P.1 P.2)
  (l : ℝ → ℝ → Prop) (hl : l P.1 P.2)
  (hperp : ∀ x y, l x y → (5 * (y - P.2) = -3 * (x - P.1))) :
  (∀ x y, l x y ↔ 3*x - 5*y + 10 = 0) ∧
  (∀ x y, (3*x - 5*y - 10 = 0) ↔ (l (-x) (-y))) :=
sorry

end intersection_and_perpendicular_lines_l2673_267397


namespace equation_solution_l2673_267346

theorem equation_solution : 
  ∃ (x : ℚ), (3/4 : ℚ) + 4/x = 1 ∧ x = 16 := by
  sorry

end equation_solution_l2673_267346


namespace arithmetic_sequence_sum_l2673_267345

/-- Given an arithmetic sequence {a_n} where a_2 = 3 and S_4 = 16, prove S_9 = 81 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 2 = 3 →                            -- given condition
  S 4 = 16 →                           -- given condition
  S 9 = 81 := by
sorry


end arithmetic_sequence_sum_l2673_267345


namespace sara_gave_nine_kittens_l2673_267348

/-- The number of kittens Sara gave to Tim -/
def kittens_from_sara (initial : ℕ) (given_to_jessica : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_to_jessica)

/-- Proof that Sara gave Tim 9 kittens -/
theorem sara_gave_nine_kittens :
  kittens_from_sara 6 3 12 = 9 := by
  sorry

end sara_gave_nine_kittens_l2673_267348


namespace lateral_surface_area_of_specific_prism_l2673_267328

/-- A triangular prism with a regular triangular base and lateral edges perpendicular to the base -/
structure TriangularPrism where
  baseArea : ℝ
  lateralEdgeLength : ℝ

/-- The lateral surface area of a triangular prism -/
def lateralSurfaceArea (prism : TriangularPrism) : ℝ :=
  sorry

theorem lateral_surface_area_of_specific_prism :
  let prism : TriangularPrism := { baseArea := 4 * Real.sqrt 3, lateralEdgeLength := 3 }
  lateralSurfaceArea prism = 36 := by
  sorry

end lateral_surface_area_of_specific_prism_l2673_267328


namespace characterize_N_l2673_267338

def StrictlyIncreasing (s : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i < j → s i < s j

def IsPeriodic (a : ℕ → ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ ∀ i : ℕ, a (i + m) = a i

def SatisfiesConditions (s : ℕ → ℕ) (N : ℕ) : Prop :=
  StrictlyIncreasing s ∧
  IsPeriodic (fun i => s (i + 1) - s i) ∧
  ∀ n : ℕ, n > 0 → s (s n) - s (s (n - 1)) ≤ N ∧ N < s (1 + s n) - s (s (n - 1))

theorem characterize_N :
  ∀ N : ℕ, (∃ s : ℕ → ℕ, SatisfiesConditions s N) ↔
    (∃ k : ℕ, k > 0 ∧ k^2 ≤ N ∧ N < k^2 + k) :=
by sorry

end characterize_N_l2673_267338


namespace exponential_equation_solutions_l2673_267349

theorem exponential_equation_solutions :
  ∀ a b c : ℕ, 2^a * 3^b = 7^c - 1 ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 4 ∧ b = 1 ∧ c = 2) :=
by sorry

end exponential_equation_solutions_l2673_267349


namespace negation_of_forall_positive_negation_of_inequality_l2673_267309

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) := by sorry

theorem negation_of_inequality :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by sorry

end negation_of_forall_positive_negation_of_inequality_l2673_267309


namespace quadratic_equation_coefficients_l2673_267323

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), ∀ x, 4 * x^2 + 1 = 6 * x ↔ a * x^2 + b * x + c = 0 ∧ a = 4 ∧ b = -6 ∧ c = 1 :=
by
  sorry

end quadratic_equation_coefficients_l2673_267323


namespace subset_families_inequality_l2673_267378

/-- Given an n-element set X and two families of subsets 𝓐 and 𝓑 of X, 
    where each subset in 𝓐 cannot be compared with every subset in 𝓑, 
    prove that √|𝓐| + √|𝓑| ≤ 2^(7/2). -/
theorem subset_families_inequality (n : ℕ) (X : Finset (Finset ℕ)) 
  (𝓐 𝓑 : Finset (Finset ℕ)) : 
  (∀ A ∈ 𝓐, ∀ B ∈ 𝓑, ¬(A ⊆ B ∨ B ⊆ A)) →
  X.card = n →
  (∀ A ∈ 𝓐, A ∈ X) →
  (∀ B ∈ 𝓑, B ∈ X) →
  Real.sqrt (𝓐.card : ℝ) + Real.sqrt (𝓑.card : ℝ) ≤ 2^(7/2) :=
by sorry

end subset_families_inequality_l2673_267378


namespace additive_inverse_sum_zero_l2673_267304

theorem additive_inverse_sum_zero (x : ℝ) : x + (-x) = 0 := by
  sorry

end additive_inverse_sum_zero_l2673_267304


namespace min_value_of_max_sum_l2673_267301

theorem min_value_of_max_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → 
  a + b + c + d = 4 → 
  let M := max (max (a + b + c) (a + b + d)) (max (a + c + d) (b + c + d))
  3 ≤ M ∧ ∀ (M' : ℝ), (∀ (a' b' c' d' : ℝ), 
    a' > 0 → b' > 0 → c' > 0 → d' > 0 → 
    a' + b' + c' + d' = 4 → 
    let M'' := max (max (a' + b' + c') (a' + b' + d')) (max (a' + c' + d') (b' + c' + d'))
    M'' ≤ M') → 
  3 ≤ M' := by
sorry

end min_value_of_max_sum_l2673_267301


namespace trigonometric_expression_value_l2673_267303

theorem trigonometric_expression_value (m x : ℝ) (h : m * Real.tan x = 2) :
  (6 * m * Real.sin (2 * x) + 2 * m * Real.cos (2 * x)) /
  (m * Real.cos (2 * x) - 3 * m * Real.sin (2 * x)) = -2/5 := by
  sorry

end trigonometric_expression_value_l2673_267303


namespace point_in_second_quadrant_l2673_267315

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: A point with negative x-coordinate and positive y-coordinate is in the second quadrant -/
theorem point_in_second_quadrant (p : Point) (hx : p.x < 0) (hy : p.y > 0) :
  SecondQuadrant p := by
  sorry

end point_in_second_quadrant_l2673_267315


namespace average_difference_l2673_267365

theorem average_difference : 
  let set1 := [20, 40, 60]
  let set2 := [10, 70, 16]
  let avg1 := (set1.sum) / (set1.length : ℝ)
  let avg2 := (set2.sum) / (set2.length : ℝ)
  avg1 - avg2 = 8 := by
sorry

end average_difference_l2673_267365


namespace weekly_surplus_and_monthly_income_estimate_l2673_267308

def weekly_income : List ℤ := [65, 68, 50, 66, 50, 75, 74]
def weekly_expenditure : List ℤ := [-60, -64, -63, -58, -60, -64, -65]

def calculate_surplus (income : List ℤ) (expenditure : List ℤ) : ℤ :=
  (income.sum + expenditure.sum)

def estimate_monthly_income (expenditure : List ℤ) : ℤ :=
  (expenditure.map (Int.natAbs)).sum * 30 / 7

theorem weekly_surplus_and_monthly_income_estimate :
  (calculate_surplus weekly_income weekly_expenditure = 14) ∧
  (estimate_monthly_income weekly_expenditure = 1860) := by
  sorry

#eval calculate_surplus weekly_income weekly_expenditure
#eval estimate_monthly_income weekly_expenditure

end weekly_surplus_and_monthly_income_estimate_l2673_267308


namespace greatest_two_digit_multiple_of_17_l2673_267314

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85 := by
  sorry

end greatest_two_digit_multiple_of_17_l2673_267314


namespace root_equation_implies_expression_value_l2673_267386

theorem root_equation_implies_expression_value (m : ℝ) : 
  m^2 - 4*m - 2 = 0 → 2*m^2 - 8*m = 4 := by
sorry

end root_equation_implies_expression_value_l2673_267386


namespace carlos_welfare_fund_contribution_l2673_267325

/-- The amount in cents dedicated to the welfare fund per hour -/
def welfare_fund_cents (hourly_wage : ℝ) (deduction_rate : ℝ) : ℝ :=
  hourly_wage * 100 * deduction_rate

/-- Proof that Carlos' welfare fund contribution is 40 cents per hour -/
theorem carlos_welfare_fund_contribution :
  welfare_fund_cents 25 0.016 = 40 := by
  sorry

end carlos_welfare_fund_contribution_l2673_267325


namespace lucy_second_round_cookies_l2673_267353

/-- The number of cookies Lucy sold on her first round -/
def first_round : ℕ := 34

/-- The total number of cookies Lucy sold -/
def total : ℕ := 61

/-- The number of cookies Lucy sold on her second round -/
def second_round : ℕ := total - first_round

theorem lucy_second_round_cookies : second_round = 27 := by
  sorry

end lucy_second_round_cookies_l2673_267353


namespace smallest_congruent_integer_l2673_267324

theorem smallest_congruent_integer (n : ℕ) : 
  (0 ≤ n ∧ n ≤ 15) ∧ n ≡ 5673 [MOD 16] → n = 9 :=
by sorry

end smallest_congruent_integer_l2673_267324


namespace sum_of_repeating_decimals_l2673_267336

-- Define the repeating decimals
def repeating_6 : ℚ := 2/3
def repeating_45 : ℚ := 5/11

-- State the theorem
theorem sum_of_repeating_decimals :
  repeating_6 + repeating_45 = 37/33 := by sorry

end sum_of_repeating_decimals_l2673_267336


namespace square_side_length_l2673_267372

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  ∃ (s : ℝ), s * s = d * d / 2 ∧ s = 2 :=
by sorry

end square_side_length_l2673_267372

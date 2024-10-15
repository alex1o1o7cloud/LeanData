import Mathlib

namespace NUMINAMATH_GPT_problem_statement_l2279_227965

variables {AB CD BC DA : ℝ} (E : ℝ) (midpoint_E : E = BC / 2) (ins_ABC : circle_inscribable AB ED)
  (ins_AEC : circle_inscribable AE CD) (a b c d : ℝ) (h_AB : AB = a) (h_BC : BC = b) (h_CD : CD = c)
  (h_DA : DA = d)

theorem problem_statement :
  a + c = b / 3 + d ∧ (1 / a + 1 / c = 3 / b) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2279_227965


namespace NUMINAMATH_GPT_probability_two_even_multiples_of_five_drawn_l2279_227914

-- Definition of conditions
def toys : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
                      39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

def isEvenMultipleOfFive (n : ℕ) : Bool := n % 10 == 0

-- Collect all such numbers from the list
def evenMultiplesOfFive : List ℕ := toys.filter isEvenMultipleOfFive

-- Number of such even multiples of 5
def countEvenMultiplesOfFive : ℕ := evenMultiplesOfFive.length

theorem probability_two_even_multiples_of_five_drawn :
  (countEvenMultiplesOfFive / 50) * ((countEvenMultiplesOfFive - 1) / 49) = 2 / 245 :=
  by sorry

end NUMINAMATH_GPT_probability_two_even_multiples_of_five_drawn_l2279_227914


namespace NUMINAMATH_GPT_class_size_is_44_l2279_227968

theorem class_size_is_44 (n : ℕ) : 
  (n - 1) % 2 = 1 ∧ (n - 1) % 7 = 1 → n = 44 := 
by 
  sorry

end NUMINAMATH_GPT_class_size_is_44_l2279_227968


namespace NUMINAMATH_GPT_min_frac_sum_pos_real_l2279_227966

variable {x y z w : ℝ}

theorem min_frac_sum_pos_real (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) (h_sum : x + y + z + w = 1) : 
  (x + y + z) / (x * y * z * w) ≥ 144 := 
sorry

end NUMINAMATH_GPT_min_frac_sum_pos_real_l2279_227966


namespace NUMINAMATH_GPT_correct_statements_count_l2279_227943

theorem correct_statements_count :
  (¬(1 = 1) ∧ ¬(1 = 0)) ∧
  (¬(1 = 11)) ∧
  ((1 - 2 + 1 / 2) = 3) ∧
  (2 = 2) →
  2 = ([false, false, true, true].count true) := 
sorry

end NUMINAMATH_GPT_correct_statements_count_l2279_227943


namespace NUMINAMATH_GPT_Billys_age_l2279_227920

variable (B J : ℕ)

theorem Billys_age :
  B = 2 * J ∧ B + J = 45 → B = 30 :=
by
  sorry

end NUMINAMATH_GPT_Billys_age_l2279_227920


namespace NUMINAMATH_GPT_ab_fraction_inequality_l2279_227993

theorem ab_fraction_inequality (a b : ℝ) (ha : 0 < a) (ha1 : a < 1) (hb : 0 < b) (hb1 : b < 1) :
  (a * b * (1 - a) * (1 - b)) / ((1 - a * b) ^ 2) < 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ab_fraction_inequality_l2279_227993


namespace NUMINAMATH_GPT_money_equation_l2279_227942

variables (a b: ℝ)

theorem money_equation (h1: 8 * a + b > 160) (h2: 4 * a + b = 120) : a > 10 ∧ ∀ (a1 a2 : ℝ), a1 > a2 → b = 120 - 4 * a → b = 120 - 4 * a1 ∧ 120 - 4 * a1 < 120 - 4 * a2 :=
by 
  sorry

end NUMINAMATH_GPT_money_equation_l2279_227942


namespace NUMINAMATH_GPT_belle_stickers_l2279_227919

theorem belle_stickers (c_stickers : ℕ) (diff : ℕ) (b_stickers : ℕ) (h1 : c_stickers = 79) (h2 : diff = 18) (h3 : c_stickers = b_stickers - diff) : b_stickers = 97 := 
by
  sorry

end NUMINAMATH_GPT_belle_stickers_l2279_227919


namespace NUMINAMATH_GPT_trig_identity_l2279_227971

-- Define the angle alpha with the given condition tan(alpha) = 2
variables (α : ℝ) (h : Real.tan α = 2)

-- State the theorem
theorem trig_identity : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l2279_227971


namespace NUMINAMATH_GPT_measure_of_angle_A_l2279_227954

-- Define the conditions as assumptions
variable (B : Real) (angle1 angle2 A : Real)
-- Angle B is 120 degrees
axiom h1 : B = 120
-- One of the angles formed by the dividing line is 50 degrees
axiom h2 : angle1 = 50
-- Angles formed sum up to 180 degrees as they are supplementary
axiom h3 : angle2 = 180 - angle1
-- Vertical angles are equal
axiom h4 : A = angle2

theorem measure_of_angle_A (B angle1 angle2 A : Real) 
    (h1 : B = 120) (h2 : angle1 = 50) (h3 : angle2 = 180 - angle1) (h4 : A = angle2) : A = 130 := 
by
    sorry

end NUMINAMATH_GPT_measure_of_angle_A_l2279_227954


namespace NUMINAMATH_GPT_num_four_letter_initials_sets_l2279_227903

def num_initials_sets : ℕ := 8 ^ 4

theorem num_four_letter_initials_sets:
  num_initials_sets = 4096 :=
by
  rw [num_initials_sets]
  norm_num

end NUMINAMATH_GPT_num_four_letter_initials_sets_l2279_227903


namespace NUMINAMATH_GPT_proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l2279_227991

open Classical

variable (a b x y : ℝ)

theorem proposition_A_correct (h : a > 1) : (1 / a < 1) ∧ ¬((1 / a < 1) → (a > 1)) :=
sorry

theorem proposition_B_incorrect (h_neg : ¬(x < 1 → x^2 < 1)) : ¬(∃ x, x ≥ 1 ∧ x^2 ≥ 1) :=
sorry

theorem proposition_C_incorrect (h_xy : x ≥ 2 ∧ y ≥ 2) : ¬((x ≥ 2 ∧ y ≥ 2) → x^2 + y^2 ≥ 4) :=
sorry

theorem proposition_D_correct (h_a : a ≠ 0) : (a * b ≠ 0) ∧ ¬((a * b ≠ 0) → (a ≠ 0)) :=
sorry

end NUMINAMATH_GPT_proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l2279_227991


namespace NUMINAMATH_GPT_part1_exists_rectangle_B_part2_no_rectangle_B_general_exists_rectangle_B_l2279_227957

-- Part 1: Prove existence of rectangle B with sides 2 + sqrt(2)/2 and 2 - sqrt(2)/2
theorem part1_exists_rectangle_B : 
  ∃ (x y : ℝ), (x + y = 4) ∧ (x * y = 7 / 2) :=
by
  sorry

-- Part 2: Prove non-existence of rectangle B for given sides of the known rectangle
theorem part2_no_rectangle_B : 
  ¬ ∃ (x y : ℝ), (x + y = 5 / 2) ∧ (x * y = 2) :=
by
  sorry

-- Part 3: General proof for any given sides of the known rectangle
theorem general_exists_rectangle_B (m n : ℝ) : 
  ∃ (x y : ℝ), (x + y = 3 * (m + n)) ∧ (x * y = 3 * m * n) :=
by
  sorry

end NUMINAMATH_GPT_part1_exists_rectangle_B_part2_no_rectangle_B_general_exists_rectangle_B_l2279_227957


namespace NUMINAMATH_GPT_sales_not_books_magazines_stationery_l2279_227939

variable (books_sales : ℕ := 45)
variable (magazines_sales : ℕ := 30)
variable (stationery_sales : ℕ := 10)
variable (total_sales : ℕ := 100)

theorem sales_not_books_magazines_stationery : 
  books_sales + magazines_sales + stationery_sales < total_sales → 
  total_sales - (books_sales + magazines_sales + stationery_sales) = 15 :=
by
  sorry

end NUMINAMATH_GPT_sales_not_books_magazines_stationery_l2279_227939


namespace NUMINAMATH_GPT_min_value_x_squared_plus_10x_l2279_227908

theorem min_value_x_squared_plus_10x : ∃ x : ℝ, (x^2 + 10 * x) = -25 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_x_squared_plus_10x_l2279_227908


namespace NUMINAMATH_GPT_smallest_integer_consecutive_set_l2279_227918

theorem smallest_integer_consecutive_set :
  ∃ m : ℤ, (m+3 < 3*m - 5) ∧ (∀ n : ℤ, (n+3 < 3*n - 5) → n ≥ m) ∧ m = 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_consecutive_set_l2279_227918


namespace NUMINAMATH_GPT_cubic_roots_equal_l2279_227975

theorem cubic_roots_equal (k : ℚ) (h1 : k > 0)
  (h2 : ∃ a b : ℚ, a ≠ b ∧ (a + a + b = -3) ∧ (2 * a * b + a^2 = -54) ∧ (3 * x^3 + 9 * x^2 - 162 * x + k = 0)) : 
  k = 7983 / 125 :=
sorry

end NUMINAMATH_GPT_cubic_roots_equal_l2279_227975


namespace NUMINAMATH_GPT_total_amount_l2279_227922

-- Definitions directly derived from the conditions in the problem
variable (you_spent friend_spent : ℕ)
variable (h1 : friend_spent = you_spent + 1)
variable (h2 : friend_spent = 8)

-- The goal is to prove that the total amount spent on lunch is $15
theorem total_amount : you_spent + friend_spent = 15 := by
  sorry

end NUMINAMATH_GPT_total_amount_l2279_227922


namespace NUMINAMATH_GPT_find_four_numbers_l2279_227963

theorem find_four_numbers 
    (a b c d : ℕ) 
    (h1 : b - a = c - b)  -- first three numbers form an arithmetic sequence
    (h2 : d / c = c / (b - a + b))  -- last three numbers form a geometric sequence
    (h3 : a + d = 16)  -- sum of first and last numbers is 16
    (h4 : b + (12 - b) = 12)  -- sum of the two middle numbers is 12
    : (a = 15 ∧ b = 9 ∧ c = 3 ∧ d = 1) ∨ (a = 0 ∧ b = 4 ∧ c = 8 ∧ d = 16) :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_find_four_numbers_l2279_227963


namespace NUMINAMATH_GPT_intersection_AB_l2279_227921

/-- Define the set A based on the given condition -/
def setA : Set ℝ := {x | 2 * x ^ 2 + x > 0}

/-- Define the set B based on the given condition -/
def setB : Set ℝ := {x | 2 * x + 1 > 0}

/-- Prove that A ∩ B = {x | x > 0} -/
theorem intersection_AB : (setA ∩ setB) = {x | x > 0} :=
sorry

end NUMINAMATH_GPT_intersection_AB_l2279_227921


namespace NUMINAMATH_GPT_james_shirts_l2279_227998

theorem james_shirts (S P : ℕ) (h1 : P = S / 2) (h2 : 6 * S + 8 * P = 100) : S = 10 :=
sorry

end NUMINAMATH_GPT_james_shirts_l2279_227998


namespace NUMINAMATH_GPT_chloe_first_round_points_l2279_227929

variable (P : ℤ)
variable (totalPoints : ℤ := 86)
variable (secondRoundPoints : ℤ := 50)
variable (lastRoundLoss : ℤ := 4)

theorem chloe_first_round_points 
  (h : P + secondRoundPoints - lastRoundLoss = totalPoints) : 
  P = 40 := by
  sorry

end NUMINAMATH_GPT_chloe_first_round_points_l2279_227929


namespace NUMINAMATH_GPT_problem1_problem2_l2279_227946

-- Problem 1: Prove the expression evaluates to 8
theorem problem1 : (1:ℝ) * (- (1 / 2)⁻¹) + (3 - Real.pi)^0 + (-3)^2 = 8 := 
by
  sorry

-- Problem 2: Prove the expression simplifies to 9a^6 - 2a^2
theorem problem2 (a : ℝ) : a^2 * a^4 - (-2 * a^2)^3 - 3 * a^2 + a^2 = 9 * a^6 - 2 * a^2 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2279_227946


namespace NUMINAMATH_GPT_xy_is_necessary_but_not_sufficient_l2279_227905

theorem xy_is_necessary_but_not_sufficient (x y : ℝ) :
  (x^2 + y^2 = 0 → xy = 0) ∧ (xy = 0 → ¬(x^2 + y^2 ≠ 0)) := by
  sorry

end NUMINAMATH_GPT_xy_is_necessary_but_not_sufficient_l2279_227905


namespace NUMINAMATH_GPT_rhombus_side_length_l2279_227973

theorem rhombus_side_length (s : ℝ) (h : 4 * s = 32) : s = 8 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_side_length_l2279_227973


namespace NUMINAMATH_GPT_two_polygons_sum_of_interior_angles_l2279_227926

theorem two_polygons_sum_of_interior_angles
  (n1 n2 : ℕ) (h1 : Even n1) (h2 : Even n2) 
  (h_sum : (n1 - 2) * 180 + (n2 - 2) * 180 = 1800):
  (n1 = 4 ∧ n2 = 10) ∨ (n1 = 6 ∧ n2 = 8) :=
by
  sorry

end NUMINAMATH_GPT_two_polygons_sum_of_interior_angles_l2279_227926


namespace NUMINAMATH_GPT_number_of_real_roots_l2279_227941

noncomputable def f (x : ℝ) : ℝ := x ^ 3 - 6 * x ^ 2 + 9 * x - 10

theorem number_of_real_roots : ∃! x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_GPT_number_of_real_roots_l2279_227941


namespace NUMINAMATH_GPT_find_f_when_x_lt_0_l2279_227977

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_defined (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^2 - 2 * x

theorem find_f_when_x_lt_0 (f : ℝ → ℝ) (h_odd : odd_function f) (h_defined : f_defined f) :
  ∀ x < 0, f x = -x^2 - 2 * x :=
by
  sorry

end NUMINAMATH_GPT_find_f_when_x_lt_0_l2279_227977


namespace NUMINAMATH_GPT_square_side_length_l2279_227935

theorem square_side_length (s : ℝ) (h1 : 4 * s = 12) (h2 : s^2 = 9) : s = 3 :=
sorry

end NUMINAMATH_GPT_square_side_length_l2279_227935


namespace NUMINAMATH_GPT_passenger_difference_l2279_227979

theorem passenger_difference {x : ℕ} :
  (30 + x = 3 * x + 14) →
  6 = 3 * x - x - 16 :=
by
  sorry

end NUMINAMATH_GPT_passenger_difference_l2279_227979


namespace NUMINAMATH_GPT_area_of_trapezium_l2279_227907

-- Defining the lengths of the sides and the distance
def a : ℝ := 12  -- 12 cm
def b : ℝ := 16  -- 16 cm
def h : ℝ := 14  -- 14 cm

-- Statement that the area of the trapezium is 196 cm²
theorem area_of_trapezium : (1 / 2) * (a + b) * h = 196 :=
by
  sorry

end NUMINAMATH_GPT_area_of_trapezium_l2279_227907


namespace NUMINAMATH_GPT_oilProductionPerPerson_west_oilProductionPerPerson_nonwest_oilProductionPerPerson_russia_l2279_227917

-- Definitions for oil consumption per person
def oilConsumptionWest : ℝ := 55.084
def oilConsumptionNonWest : ℝ := 214.59
def oilConsumptionRussia : ℝ := 1038.33

-- Lean statements
theorem oilProductionPerPerson_west : oilConsumptionWest = 55.084 := by
  sorry

theorem oilProductionPerPerson_nonwest : oilConsumptionNonWest = 214.59 := by
  sorry

theorem oilProductionPerPerson_russia : oilConsumptionRussia = 1038.33 := by
  sorry

end NUMINAMATH_GPT_oilProductionPerPerson_west_oilProductionPerPerson_nonwest_oilProductionPerPerson_russia_l2279_227917


namespace NUMINAMATH_GPT_find_other_number_l2279_227960

theorem find_other_number (A B : ℕ) (LCM HCF : ℕ) (h1 : LCM = 2310) (h2 : HCF = 83) (h3 : A = 210) (h4 : LCM * HCF = A * B) : B = 913 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l2279_227960


namespace NUMINAMATH_GPT_grill_run_time_l2279_227964

-- Definitions of conditions
def coals_burned_per_minute : ℕ := 15
def minutes_per_coal_burned : ℕ := 20
def coals_per_bag : ℕ := 60
def bags_burned : ℕ := 3

-- Theorems to prove the question
theorem grill_run_time (coals_burned_per_minute: ℕ) (minutes_per_coal_burned: ℕ) (coals_per_bag: ℕ) (bags_burned: ℕ): (coals_burned_per_minute * (minutes_per_coal_burned * bags_burned * coals_per_bag / (coals_burned_per_minute * coals_per_bag))) / 60 = 4 := 
by 
  -- Lean statement skips detailed proof steps for conciseness
  sorry

end NUMINAMATH_GPT_grill_run_time_l2279_227964


namespace NUMINAMATH_GPT_min_ab_min_a_plus_b_l2279_227916

theorem min_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : ab >= 8 :=
sorry

theorem min_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : a + b >= 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_ab_min_a_plus_b_l2279_227916


namespace NUMINAMATH_GPT_average_weight_increase_l2279_227904

-- Define the initial conditions as given in the problem
def W_old : ℕ := 53
def W_new : ℕ := 71
def N : ℕ := 10

-- Average weight increase after replacing one oarsman
theorem average_weight_increase : (W_new - W_old : ℝ) / N = 1.8 := by
  sorry

end NUMINAMATH_GPT_average_weight_increase_l2279_227904


namespace NUMINAMATH_GPT_fourth_throw_probability_l2279_227988

-- Define a fair dice where each face has an equal probability.
def fair_dice (n : ℕ) : Prop := (n >= 1 ∧ n <= 6)

-- Define the probability of rolling a 6 on a fair dice.
noncomputable def probability_of_6 : ℝ := 1 / 6

/-- 
  Prove that the probability of getting a "6" on the 4th throw is 1/6 
  given that the dice is fair and the first three throws result in "6".
-/
theorem fourth_throw_probability : 
  (∀ (n1 n2 n3 : ℕ), fair_dice n1 ∧ fair_dice n2 ∧ fair_dice n3 ∧ n1 = 6 ∧ n2 = 6 ∧ n3 = 6) 
  → (probability_of_6 = 1 / 6) :=
by 
  sorry

end NUMINAMATH_GPT_fourth_throw_probability_l2279_227988


namespace NUMINAMATH_GPT_no_burial_needed_for_survivors_l2279_227937

def isSurvivor (p : Person) : Bool := sorry
def isBuried (p : Person) : Bool := sorry
variable (p : Person) (accident : Bool)

theorem no_burial_needed_for_survivors (h : accident = true) (hsurvive : isSurvivor p = true) : isBuried p = false :=
sorry

end NUMINAMATH_GPT_no_burial_needed_for_survivors_l2279_227937


namespace NUMINAMATH_GPT_quadratic_point_inequality_l2279_227953

theorem quadratic_point_inequality 
  (m y1 y2 : ℝ)
  (hA : y1 = (m - 1)^2)
  (hB : y2 = (m + 1 - 1)^2)
  (hy1_lt_y2 : y1 < y2) :
  m > 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_point_inequality_l2279_227953


namespace NUMINAMATH_GPT_plantingMethodsCalculation_l2279_227924

noncomputable def numPlantingMethods : Nat :=
  let totalSeeds := 5
  let endChoices := 3 * 2 -- Choosing 2 seeds for the ends from 3 remaining types
  let middleChoices := 6 -- Permutations of (A, B, another type) = 3! = 6
  endChoices * middleChoices

theorem plantingMethodsCalculation : numPlantingMethods = 24 := by
  sorry

end NUMINAMATH_GPT_plantingMethodsCalculation_l2279_227924


namespace NUMINAMATH_GPT_largest_even_integer_sum_l2279_227951

theorem largest_even_integer_sum (x : ℤ) (h : (20 * (x + x + 38) / 2) = 6400) : 
  x + 38 = 339 :=
sorry

end NUMINAMATH_GPT_largest_even_integer_sum_l2279_227951


namespace NUMINAMATH_GPT_a_7_eq_64_l2279_227934

-- Define the problem conditions using variables in Lean
variable {a : ℕ → ℝ} -- defining the sequence as a function from natural numbers to reals
variable {q : ℝ}  -- common ratio

-- The sequence is geometric
axiom geom_seq (n : ℕ) : a (n + 1) = a n * q

-- Conditions given in the problem
axiom condition1 : a 1 + a 2 = 3
axiom condition2 : a 2 + a 3 = 6

-- Target statement to prove
theorem a_7_eq_64 : a 7 = 64 := 
sorry

end NUMINAMATH_GPT_a_7_eq_64_l2279_227934


namespace NUMINAMATH_GPT_circles_tangent_dist_l2279_227923

theorem circles_tangent_dist (t : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4) ∧ 
  (∀ x y : ℝ, (x - t)^2 + y^2 = 1) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, x1^2 + y1^2 = 4 → (x2 - t)^2 + y2^2 = 1 → 
    dist (x1, y1) (x2, y2) = 3) → 
  t = 3 ∨ t = -3 :=
by 
  sorry

end NUMINAMATH_GPT_circles_tangent_dist_l2279_227923


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l2279_227927

noncomputable def f (x : ℝ) := Real.log x + x^2 - 3 * x

theorem monotonic_decreasing_interval :
  (∃ I : Set ℝ, I = Set.Ioo (1 / 2 : ℝ) 1 ∧ ∀ x ∈ I, ∀ y ∈ I, x < y → f x ≥ f y) := 
by
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l2279_227927


namespace NUMINAMATH_GPT_people_per_table_l2279_227984

def total_people_invited : ℕ := 68
def people_who_didn't_show_up : ℕ := 50
def number_of_tables_needed : ℕ := 6

theorem people_per_table (total_people_invited people_who_didn't_show_up number_of_tables_needed : ℕ) : 
  total_people_invited - people_who_didn't_show_up = 18 ∧
  (total_people_invited - people_who_didn't_show_up) / number_of_tables_needed = 3 :=
by
  sorry

end NUMINAMATH_GPT_people_per_table_l2279_227984


namespace NUMINAMATH_GPT_compare_f_neg_x1_neg_x2_l2279_227952

noncomputable def f : ℝ → ℝ := sorry

theorem compare_f_neg_x1_neg_x2 
  (h1 : ∀ x : ℝ, f (1 + x) = f (1 - x)) 
  (h2 : ∀ x y : ℝ, 1 ≤ x → 1 ≤ y → x < y → f x < f y)
  (x1 x2 : ℝ)
  (hx1 : x1 < 0)
  (hx2 : x2 > 0)
  (hx1x2 : x1 + x2 < -2) :
  f (-x1) > f (-x2) :=
by sorry

end NUMINAMATH_GPT_compare_f_neg_x1_neg_x2_l2279_227952


namespace NUMINAMATH_GPT_sqrt_expression_eval_l2279_227959

theorem sqrt_expression_eval :
  (Real.sqrt 8) + (Real.sqrt (1 / 2)) + (Real.sqrt 3 - 1) ^ 2 + (Real.sqrt 6 / (1 / 2 * Real.sqrt 2)) = (5 / 2) * Real.sqrt 2 + 4 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_eval_l2279_227959


namespace NUMINAMATH_GPT_frosting_need_l2279_227989

theorem frosting_need : 
  (let layer_cake_frosting := 1
   let single_cake_frosting := 0.5
   let brownie_frosting := 0.5
   let dozen_cupcakes_frosting := 0.5
   let num_layer_cakes := 3
   let num_dozen_cupcakes := 6
   let num_single_cakes := 12
   let num_pans_brownies := 18
   
   let total_frosting := 
     (num_layer_cakes * layer_cake_frosting) + 
     (num_dozen_cupcakes * dozen_cupcakes_frosting) + 
     (num_single_cakes * single_cake_frosting) + 
     (num_pans_brownies * brownie_frosting)
   
   total_frosting = 21) :=
  by
    sorry

end NUMINAMATH_GPT_frosting_need_l2279_227989


namespace NUMINAMATH_GPT_condition_for_a_l2279_227962

theorem condition_for_a (a : ℝ) :
  (∀ x : ℤ, (x < 0 → (x + a) / 2 ≥ 1) → (x = -1 ∨ x = -2)) ↔ 4 ≤ a ∧ a < 5 :=
by
  sorry

end NUMINAMATH_GPT_condition_for_a_l2279_227962


namespace NUMINAMATH_GPT_correct_statements_l2279_227990

open Classical

variables {α l m n p : Type*}
variables (is_perpendicular_to : α → α → Prop) (is_parallel_to : α → α → Prop)
variables (is_in_plane : α → α → Prop)

noncomputable def problem_statement (l : α) (α : α) : Prop :=
  (∀ m, is_perpendicular_to m l → is_parallel_to m α) ∧
  (∀ m, is_perpendicular_to m α → is_parallel_to m l) ∧
  (∀ m, is_parallel_to m α → is_perpendicular_to m l) ∧
  (∀ m, is_parallel_to m l → is_perpendicular_to m α)

theorem correct_statements (l : α) (α : α) (h_l_α : is_perpendicular_to l α) :
  (∀ m, is_perpendicular_to m α → is_parallel_to m l) ∧
  (∀ m, is_parallel_to m α → is_perpendicular_to m l) ∧
  (∀ m, is_parallel_to m l → is_perpendicular_to m α) :=
sorry

end NUMINAMATH_GPT_correct_statements_l2279_227990


namespace NUMINAMATH_GPT_remainder_2457634_div_8_l2279_227994

theorem remainder_2457634_div_8 : 2457634 % 8 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_2457634_div_8_l2279_227994


namespace NUMINAMATH_GPT_commission_percentage_proof_l2279_227936

-- Let's define the problem conditions in Lean

-- Condition 1: Commission on first Rs. 10,000
def commission_first_10000 (sales : ℕ) : ℕ :=
  if sales ≤ 10000 then
    5 * sales / 100
  else
    500

-- Condition 2: Amount remitted to company after commission
def amount_remitted (total_sales : ℕ) (commission : ℕ) : ℕ :=
  total_sales - commission

-- Condition 3: Function to calculate commission on exceeding amount
def commission_exceeding (sales : ℕ) (x : ℕ) : ℕ :=
  x * sales / 100

-- The main hypothesis as per the given problem
def correct_commission_percentage (total_sales : ℕ) (remitted : ℕ) (x : ℕ) :=
  commission_first_10000 10000 + commission_exceeding (total_sales - 10000) x
  = total_sales - remitted

-- Problem statement to prove the percentage of commission on exceeding Rs. 10,000 is 4%
theorem commission_percentage_proof : correct_commission_percentage 32500 31100 4 := 
  by sorry

end NUMINAMATH_GPT_commission_percentage_proof_l2279_227936


namespace NUMINAMATH_GPT_amount_first_set_correct_l2279_227931

-- Define the amounts as constants
def total_amount : ℝ := 900.00
def amount_second_set : ℝ := 260.00
def amount_third_set : ℝ := 315.00

-- Define the amount given to the first set
def amount_first_set : ℝ :=
  total_amount - amount_second_set - amount_third_set

-- Statement: prove that the amount given to the first set of families equals $325.00
theorem amount_first_set_correct :
  amount_first_set = 325.00 :=
sorry

end NUMINAMATH_GPT_amount_first_set_correct_l2279_227931


namespace NUMINAMATH_GPT_fourth_term_of_geometric_sequence_l2279_227947

theorem fourth_term_of_geometric_sequence (a₁ : ℝ) (a₆ : ℝ) (a₄ : ℝ) (r : ℝ)
  (h₁ : a₁ = 1000)
  (h₂ : a₆ = a₁ * r^5)
  (h₃ : a₆ = 125)
  (h₄ : a₄ = a₁ * r^3) : 
  a₄ = 125 :=
sorry

end NUMINAMATH_GPT_fourth_term_of_geometric_sequence_l2279_227947


namespace NUMINAMATH_GPT_jogging_track_circumference_l2279_227997

def speed_Suresh_km_hr : ℝ := 4.5
def speed_wife_km_hr : ℝ := 3.75
def meet_time_min : ℝ := 5.28

theorem jogging_track_circumference : 
  let speed_Suresh_km_min := speed_Suresh_km_hr / 60
  let speed_wife_km_min := speed_wife_km_hr / 60
  let distance_Suresh_km := speed_Suresh_km_min * meet_time_min
  let distance_wife_km := speed_wife_km_min * meet_time_min
  let total_distance_km := distance_Suresh_km + distance_wife_km
  total_distance_km = 0.726 :=
by sorry

end NUMINAMATH_GPT_jogging_track_circumference_l2279_227997


namespace NUMINAMATH_GPT_gcd_values_count_l2279_227958

theorem gcd_values_count (a b : ℕ) (h : a * b = 3600) : ∃ n, n = 29 ∧ ∀ d, d ∣ a ∧ d ∣ b → d = gcd a b → n = 29 :=
by { sorry }

end NUMINAMATH_GPT_gcd_values_count_l2279_227958


namespace NUMINAMATH_GPT_probability_adjacent_vertices_in_decagon_l2279_227970

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end NUMINAMATH_GPT_probability_adjacent_vertices_in_decagon_l2279_227970


namespace NUMINAMATH_GPT_num_pairs_mod_eq_l2279_227995

theorem num_pairs_mod_eq (k : ℕ) (h : k ≥ 7) :
  ∃ n : ℕ, n = 2^(k+5) ∧
  (∀ x y : ℕ, 0 ≤ x ∧ x < 2^k ∧ 0 ≤ y ∧ y < 2^k → (73^(73^x) ≡ 9^(9^y) [MOD 2^k]) → true) :=
sorry

end NUMINAMATH_GPT_num_pairs_mod_eq_l2279_227995


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l2279_227976

noncomputable def f (x : ℝ) : ℝ :=
  x / 4 + 5 / (4 * x) - Real.log x

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), (a = 0) ∧ (b = 5) ∧ (∀ x, 0 < x ∧ x < 5 → (deriv f x < 0)) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l2279_227976


namespace NUMINAMATH_GPT_compute_sum_of_squares_roots_l2279_227978

-- p, q, and r are roots of 3*x^3 - 2*x^2 + 6*x + 15 = 0.
def P (x : ℝ) : Prop := 3*x^3 - 2*x^2 + 6*x + 15 = 0

theorem compute_sum_of_squares_roots :
  ∀ p q r : ℝ, P p ∧ P q ∧ P r → p^2 + q^2 + r^2 = -32 / 9 :=
by
  intros p q r h
  sorry

end NUMINAMATH_GPT_compute_sum_of_squares_roots_l2279_227978


namespace NUMINAMATH_GPT_heptagon_diagonals_l2279_227981

theorem heptagon_diagonals (n : ℕ) (h : n = 7) : (n * (n - 3)) / 2 = 14 := by
  sorry

end NUMINAMATH_GPT_heptagon_diagonals_l2279_227981


namespace NUMINAMATH_GPT_total_surface_area_of_three_face_painted_cubes_l2279_227987

def cube_side_length : ℕ := 9
def small_cube_side_length : ℕ := 1
def num_small_cubes_with_three_faces_painted : ℕ := 8
def surface_area_of_each_painted_face : ℕ := 6

theorem total_surface_area_of_three_face_painted_cubes :
  num_small_cubes_with_three_faces_painted * surface_area_of_each_painted_face = 48 := by
  sorry

end NUMINAMATH_GPT_total_surface_area_of_three_face_painted_cubes_l2279_227987


namespace NUMINAMATH_GPT_inverse_proportion_function_neg_k_l2279_227906

variable {k : ℝ}
variable {y1 y2 : ℝ}

theorem inverse_proportion_function_neg_k
  (h1 : k ≠ 0)
  (h2 : y1 > y2)
  (hA : y1 = k / (-2))
  (hB : y2 = k / 5) :
  k < 0 :=
sorry

end NUMINAMATH_GPT_inverse_proportion_function_neg_k_l2279_227906


namespace NUMINAMATH_GPT_domain_log_function_l2279_227985

open Real

def quadratic_term (x : ℝ) : ℝ := 4 - 3 * x - x^2

def valid_argument (x : ℝ) : Prop := quadratic_term x > 0

theorem domain_log_function : { x : ℝ | valid_argument x } = Set.Ioo (-4 : ℝ) (1 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_domain_log_function_l2279_227985


namespace NUMINAMATH_GPT_intersections_vary_with_A_l2279_227956

theorem intersections_vary_with_A (A : ℝ) (hA : A > 0) :
  ∃ x y : ℝ, (y = A * x^2) ∧ (y^2 + 2 = x^2 + 6 * y) ∧ (y = 2 * x - 1) :=
sorry

end NUMINAMATH_GPT_intersections_vary_with_A_l2279_227956


namespace NUMINAMATH_GPT_percentage_not_speak_french_l2279_227982

open Nat

theorem percentage_not_speak_french (students_surveyed : ℕ)
  (speak_french_and_english : ℕ) (speak_only_french : ℕ) :
  students_surveyed = 200 →
  speak_french_and_english = 25 →
  speak_only_french = 65 →
  ((students_surveyed - (speak_french_and_english + speak_only_french)) * 100 / students_surveyed) = 55 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_percentage_not_speak_french_l2279_227982


namespace NUMINAMATH_GPT_initial_innings_l2279_227955

/-- The number of innings a player played initially given the conditions described in the problem. -/
theorem initial_innings (n : ℕ)
  (average_runs : ℕ)
  (additional_runs : ℕ)
  (new_average_increase : ℕ)
  (h1 : average_runs = 42)
  (h2 : additional_runs = 86)
  (h3 : new_average_increase = 4) :
  42 * n + 86 = 46 * (n + 1) → n = 10 :=
by
  intros h
  linarith

end NUMINAMATH_GPT_initial_innings_l2279_227955


namespace NUMINAMATH_GPT_line_passes_through_circle_center_l2279_227940

theorem line_passes_through_circle_center (a : ℝ) : 
  ∀ x y : ℝ, (x, y) = (a, 2*a) → (x - a)^2 + (y - 2*a)^2 = 1 → 2*x - y = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_circle_center_l2279_227940


namespace NUMINAMATH_GPT_set_of_possible_values_l2279_227910

-- Define the variables and the conditions as a Lean definition
noncomputable def problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) : Set ℝ :=
  {x : ℝ | x = (1 / a + 1 / b + 1 / c)}

-- Define the theorem to state that the set of all possible values is [9, ∞)
theorem set_of_possible_values (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  problem a b c ha hb hc sum_eq_one = {x : ℝ | 9 ≤ x} :=
sorry

end NUMINAMATH_GPT_set_of_possible_values_l2279_227910


namespace NUMINAMATH_GPT_simplify_eq_l2279_227992

theorem simplify_eq {x y z : ℕ} (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  9 * (x : ℝ) - ((10 / (2 * y) / 3 + 7 * z) * Real.pi) =
  9 * (x : ℝ) - (5 * Real.pi / (3 * y) + 7 * z * Real.pi) := by
  sorry

end NUMINAMATH_GPT_simplify_eq_l2279_227992


namespace NUMINAMATH_GPT_greg_total_earnings_l2279_227930

-- Define the charges and walking times as given
def charge_per_dog : ℕ := 20
def charge_per_minute : ℕ := 1
def one_dog_minutes : ℕ := 10
def two_dogs_minutes : ℕ := 7
def three_dogs_minutes : ℕ := 9
def total_dogs_one : ℕ := 1
def total_dogs_two : ℕ := 2
def total_dogs_three : ℕ := 3

-- Total earnings computation
def earnings_one_dog : ℕ := charge_per_dog + charge_per_minute * one_dog_minutes
def earnings_two_dogs : ℕ := total_dogs_two * charge_per_dog + total_dogs_two * charge_per_minute * two_dogs_minutes
def earnings_three_dogs : ℕ := total_dogs_three * charge_per_dog + total_dogs_three * charge_per_minute * three_dogs_minutes
def total_earnings : ℕ := earnings_one_dog + earnings_two_dogs + earnings_three_dogs

-- The proof: Greg's total earnings should be $171
theorem greg_total_earnings : total_earnings = 171 := by
  -- Placeholder for the proof (not required as per the instructions)
  sorry

end NUMINAMATH_GPT_greg_total_earnings_l2279_227930


namespace NUMINAMATH_GPT_by_how_much_were_the_numerator_and_denominator_increased_l2279_227902

noncomputable def original_fraction_is_six_over_eleven (n : ℕ) : Prop :=
  n / (n + 5) = 6 / 11

noncomputable def resulting_fraction_is_seven_over_twelve (n x : ℕ) : Prop :=
  (n + x) / (n + 5 + x) = 7 / 12

theorem by_how_much_were_the_numerator_and_denominator_increased :
  ∃ (n x : ℕ), original_fraction_is_six_over_eleven n ∧ resulting_fraction_is_seven_over_twelve n x ∧ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_by_how_much_were_the_numerator_and_denominator_increased_l2279_227902


namespace NUMINAMATH_GPT_geometric_sequence_sum_9000_l2279_227944

noncomputable def sum_geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := 
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_9000 (a r : ℝ) (h : r ≠ 1) 
  (h1 : sum_geometric_sequence a r 3000 = 1000)
  (h2 : sum_geometric_sequence a r 6000 = 1900) : 
  sum_geometric_sequence a r 9000 = 2710 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_9000_l2279_227944


namespace NUMINAMATH_GPT_range_of_a_l2279_227933

noncomputable def f (a x : ℝ) := a / x - 1 + Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ f a x ≤ 0) → a ≤ 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_l2279_227933


namespace NUMINAMATH_GPT_jiwoo_magnets_two_digit_count_l2279_227996

def num_magnets : List ℕ := [1, 2, 7]

theorem jiwoo_magnets_two_digit_count : 
  (∀ (x y : ℕ), x ≠ y → x ∈ num_magnets → y ∈ num_magnets → 2 * 3 = 6) := 
by {
  sorry
}

end NUMINAMATH_GPT_jiwoo_magnets_two_digit_count_l2279_227996


namespace NUMINAMATH_GPT_ellipse_equation_line_equation_l2279_227945
-- Import the necessary libraries

-- Problem (I): The equation of the ellipse
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (hA : (1 : ℝ) / a^2 + (9 / 4 : ℝ) / b^2 = 1)
  (h_ecc : b^2 = (3 / 4 : ℝ) * a^2) : 
  (a^2 = 4 ∧ b^2 = 3) :=
by
  sorry

-- Problem (II): The equation of the line
theorem line_equation (k : ℝ) (h_area : (12 * Real.sqrt (2 : ℝ)) / 7 = 12 * abs k / (4 * k^2 + 3)) : 
  k = 1 ∨ k = -1 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_line_equation_l2279_227945


namespace NUMINAMATH_GPT_equivalent_expression_l2279_227913

-- Let a, b, c, d, e be real numbers
variables (a b c d e : ℝ)

-- Condition given in the problem
def condition : Prop := 81 * a - 27 * b + 9 * c - 3 * d + e = -5

-- Objective: Prove that 8 * a - 4 * b + 2 * c - d + e = -5 given the condition
theorem equivalent_expression (h : condition a b c d e) : 8 * a - 4 * b + 2 * c - d + e = -5 :=
sorry

end NUMINAMATH_GPT_equivalent_expression_l2279_227913


namespace NUMINAMATH_GPT_classify_triangle_l2279_227972

theorem classify_triangle (m : ℕ) (h₁ : m > 1) (h₂ : 3 * m + 3 = 180) :
  (m < 60) ∧ (m + 1 < 90) ∧ (m + 2 < 90) :=
by
  sorry

end NUMINAMATH_GPT_classify_triangle_l2279_227972


namespace NUMINAMATH_GPT_length_of_base_of_vessel_l2279_227967

noncomputable def volume_of_cube (edge : ℝ) := edge ^ 3

theorem length_of_base_of_vessel 
  (cube_edge : ℝ)
  (vessel_width : ℝ)
  (rise_in_water_level : ℝ)
  (volume_cube : ℝ)
  (h1 : cube_edge = 15)
  (h2 : vessel_width = 15)
  (h3 : rise_in_water_level = 11.25)
  (h4 : volume_cube = volume_of_cube cube_edge)
  : ∃ L : ℝ, L = volume_cube / (vessel_width * rise_in_water_level) ∧ L = 20 :=
by
  sorry

end NUMINAMATH_GPT_length_of_base_of_vessel_l2279_227967


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_ratio_l2279_227901

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℕ) (d : ℕ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_positive_d : d > 0)
  (h_geometric : a 6 ^ 2 = a 2 * a 12) :
  (a 12) / (a 2) = 9 / 4 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_ratio_l2279_227901


namespace NUMINAMATH_GPT_mean_temperature_l2279_227974

theorem mean_temperature
  (temps : List ℤ) 
  (h_temps : temps = [-8, -3, -7, -6, 0, 4, 6, 5, -1, 2]) :
  (temps.sum: ℚ) / temps.length = -0.8 := 
by
  sorry

end NUMINAMATH_GPT_mean_temperature_l2279_227974


namespace NUMINAMATH_GPT_probability_three_draws_one_white_l2279_227912

def num_white_balls : ℕ := 4
def num_black_balls : ℕ := 3
def total_balls : ℕ := num_white_balls + num_black_balls

def probability_one_white_three_draws : ℚ := 
  (num_white_balls / total_balls) * 
  ((num_black_balls - 1) / (total_balls - 1)) * 
  ((num_black_balls - 2) / (total_balls - 2)) * 3

theorem probability_three_draws_one_white :
  probability_one_white_three_draws = 12 / 35 := by sorry

end NUMINAMATH_GPT_probability_three_draws_one_white_l2279_227912


namespace NUMINAMATH_GPT_binary_addition_l2279_227949

-- Define the binary numbers as natural numbers
def b1 : ℕ := 0b101  -- 101_2
def b2 : ℕ := 0b11   -- 11_2
def b3 : ℕ := 0b1100 -- 1100_2
def b4 : ℕ := 0b11101 -- 11101_2
def sum_b : ℕ := 0b110001 -- 110001_2

theorem binary_addition :
  b1 + b2 + b3 + b4 = sum_b := 
by
  sorry

end NUMINAMATH_GPT_binary_addition_l2279_227949


namespace NUMINAMATH_GPT_math_scores_population_l2279_227961

/-- 
   Suppose there are 50,000 students who took the high school entrance exam.
   The education department randomly selected 2,000 students' math scores 
   for statistical analysis. Prove that the math scores of the 50,000 students 
   are the population.
-/
theorem math_scores_population (students : ℕ) (selected : ℕ) 
    (students_eq : students = 50000) (selected_eq : selected = 2000) : 
    true :=
by
  sorry

end NUMINAMATH_GPT_math_scores_population_l2279_227961


namespace NUMINAMATH_GPT_quadratic_intersects_at_3_points_l2279_227915

theorem quadratic_intersects_at_3_points (m : ℝ) : 
  (exists x : ℝ, x^2 + 2*x + m = 0) ∧ (m ≠ 0) → m < 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_intersects_at_3_points_l2279_227915


namespace NUMINAMATH_GPT_sector_area_l2279_227911

theorem sector_area (r : ℝ) (alpha : ℝ) (h_r : r = 2) (h_alpha : alpha = π / 4) : 
  (1 / 2) * r^2 * alpha = π / 2 :=
by
  rw [h_r, h_alpha]
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_sector_area_l2279_227911


namespace NUMINAMATH_GPT_croissant_price_l2279_227900

theorem croissant_price (price_almond: ℝ) (total_expenditure: ℝ) (weeks: ℕ) (price_regular: ℝ) 
  (h1: price_almond = 5.50) (h2: total_expenditure = 468) (h3: weeks = 52) 
  (h4: weeks * price_regular + weeks * price_almond = total_expenditure) : price_regular = 3.50 :=
by 
  sorry

end NUMINAMATH_GPT_croissant_price_l2279_227900


namespace NUMINAMATH_GPT_new_supervisor_salary_l2279_227986

namespace FactorySalaries

variables (W S2 : ℝ)

def old_supervisor_salary : ℝ := 870
def old_average_salary : ℝ := 430
def new_average_salary : ℝ := 440

theorem new_supervisor_salary :
  (W + old_supervisor_salary) / 9 = old_average_salary →
  (W + S2) / 9 = new_average_salary →
  S2 = 960 :=
by
  intros h1 h2
  -- Proof steps would go here
  sorry

end FactorySalaries

end NUMINAMATH_GPT_new_supervisor_salary_l2279_227986


namespace NUMINAMATH_GPT_sum_non_solution_values_l2279_227948

theorem sum_non_solution_values (A B C : ℝ) (h : ∀ x : ℝ, (x+B) * (A*x+36) / ((x+C) * (x+9)) = 4) :
  ∃ M : ℝ, M = - (B + 9) := 
sorry

end NUMINAMATH_GPT_sum_non_solution_values_l2279_227948


namespace NUMINAMATH_GPT_anatoliy_handshakes_l2279_227983

-- Define the total number of handshakes
def total_handshakes := 197

-- Define friends excluding Anatoliy
def handshake_func (n : Nat) : Nat :=
  n * (n - 1) / 2

-- Define the target problem stating that Anatoliy made 7 handshakes
theorem anatoliy_handshakes (n k : Nat) (h : handshake_func n + k = total_handshakes) : k = 7 :=
by sorry

end NUMINAMATH_GPT_anatoliy_handshakes_l2279_227983


namespace NUMINAMATH_GPT_beef_not_used_l2279_227950

-- Define the context and necessary variables
variable (totalBeef : ℕ) (usedVegetables : ℕ)
variable (beefUsed : ℕ)

-- The conditions given in the problem
def initial_beef : Prop := totalBeef = 4
def used_vegetables : Prop := usedVegetables = 6
def relation_vegetables_beef : Prop := usedVegetables = 2 * beefUsed

-- The statement we need to prove
theorem beef_not_used
  (h1 : initial_beef totalBeef)
  (h2 : used_vegetables usedVegetables)
  (h3 : relation_vegetables_beef usedVegetables beefUsed) :
  (totalBeef - beefUsed) = 1 := by
  sorry

end NUMINAMATH_GPT_beef_not_used_l2279_227950


namespace NUMINAMATH_GPT_power_comparison_l2279_227938

theorem power_comparison : (5 : ℕ) ^ 30 < (3 : ℕ) ^ 50 ∧ (3 : ℕ) ^ 50 < (4 : ℕ) ^ 40 := by
  sorry

end NUMINAMATH_GPT_power_comparison_l2279_227938


namespace NUMINAMATH_GPT_harry_morning_routine_l2279_227969

theorem harry_morning_routine : 
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  buy_time + read_and_eat_time = 45 :=
by
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  show buy_time + read_and_eat_time = 45
  sorry

end NUMINAMATH_GPT_harry_morning_routine_l2279_227969


namespace NUMINAMATH_GPT_probability_symmetric_line_l2279_227980

theorem probability_symmetric_line (P : (ℕ × ℕ) := (5, 5))
    (n : ℕ := 10) (total_points remaining_points symmetric_points : ℕ) 
    (probability : ℚ) :
  total_points = n * n →
  remaining_points = total_points - 1 →
  symmetric_points = 4 * (n - 1) →
  probability = (symmetric_points : ℚ) / (remaining_points : ℚ) →
  probability = 32 / 99 :=
by
  sorry

end NUMINAMATH_GPT_probability_symmetric_line_l2279_227980


namespace NUMINAMATH_GPT_simplify_expr_1_simplify_expr_2_l2279_227925

-- The first problem
theorem simplify_expr_1 (a : ℝ) : 2 * a^2 - 3 * a - 5 * a^2 + 6 * a = -3 * a^2 + 3 * a := 
by
  sorry

-- The second problem
theorem simplify_expr_2 (a : ℝ) : 2 * (a - 1) - (2 * a - 3) + 3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_1_simplify_expr_2_l2279_227925


namespace NUMINAMATH_GPT_inequality_and_equality_conditions_l2279_227932

theorem inequality_and_equality_conditions
  (x y a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a ≥ 0)
  (h3 : b ≥ 0) :
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 ∧ ((a * b = 0) ∨ (x = y)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_and_equality_conditions_l2279_227932


namespace NUMINAMATH_GPT_sam_initial_pennies_l2279_227999

def initial_pennies_spent (spent: Nat) (left: Nat) : Nat :=
  spent + left

theorem sam_initial_pennies (spent: Nat) (left: Nat) : spent = 93 ∧ left = 5 → initial_pennies_spent spent left = 98 :=
by
  sorry

end NUMINAMATH_GPT_sam_initial_pennies_l2279_227999


namespace NUMINAMATH_GPT_magic_square_S_divisible_by_3_l2279_227909

-- Definitions of the 3x3 magic square conditions
def is_magic_square (a : ℕ → ℕ → ℤ) (S : ℤ) : Prop :=
  (a 0 0 + a 0 1 + a 0 2 = S) ∧
  (a 1 0 + a 1 1 + a 1 2 = S) ∧
  (a 2 0 + a 2 1 + a 2 2 = S) ∧
  (a 0 0 + a 1 0 + a 2 0 = S) ∧
  (a 0 1 + a 1 1 + a 2 1 = S) ∧
  (a 0 2 + a 1 2 + a 2 2 = S) ∧
  (a 0 0 + a 1 1 + a 2 2 = S) ∧
  (a 0 2 + a 1 1 + a 2 0 = S)

-- Main theorem statement
theorem magic_square_S_divisible_by_3 :
  ∀ (a : ℕ → ℕ → ℤ) (S : ℤ),
    is_magic_square a S →
    S % 3 = 0 :=
by
  -- Here we assume the existence of the proof
  sorry

end NUMINAMATH_GPT_magic_square_S_divisible_by_3_l2279_227909


namespace NUMINAMATH_GPT_teal_sales_revenue_l2279_227928

theorem teal_sales_revenue :
  let pumpkin_pie_slices := 8
  let pumpkin_pie_price := 5
  let pumpkin_pies_sold := 4
  let custard_pie_slices := 6
  let custard_pie_price := 6
  let custard_pies_sold := 5
  let apple_pie_slices := 10
  let apple_pie_price := 4
  let apple_pies_sold := 3
  let pecan_pie_slices := 12
  let pecan_pie_price := 7
  let pecan_pies_sold := 2
  (pumpkin_pie_slices * pumpkin_pie_price * pumpkin_pies_sold) +
  (custard_pie_slices * custard_pie_price * custard_pies_sold) +
  (apple_pie_slices * apple_pie_price * apple_pies_sold) +
  (pecan_pie_slices * pecan_pie_price * pecan_pies_sold) = 
  628 := by
  sorry

end NUMINAMATH_GPT_teal_sales_revenue_l2279_227928

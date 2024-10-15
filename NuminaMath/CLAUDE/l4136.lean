import Mathlib

namespace NUMINAMATH_CALUDE_blue_crayon_boxes_l4136_413677

/-- Given information about crayon boxes and their contents, prove the number of blue crayon boxes -/
theorem blue_crayon_boxes (total_crayons : ℕ) (orange_boxes : ℕ) (orange_per_box : ℕ) 
  (red_boxes : ℕ) (red_per_box : ℕ) (blue_per_box : ℕ) :
  total_crayons = 94 →
  orange_boxes = 6 →
  orange_per_box = 8 →
  red_boxes = 1 →
  red_per_box = 11 →
  blue_per_box = 5 →
  ∃ (blue_boxes : ℕ), 
    total_crayons = orange_boxes * orange_per_box + red_boxes * red_per_box + blue_boxes * blue_per_box ∧
    blue_boxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_blue_crayon_boxes_l4136_413677


namespace NUMINAMATH_CALUDE_fraction_value_preservation_l4136_413615

theorem fraction_value_preservation (original_numerator original_denominator increase_numerator : ℕ) 
  (h1 : original_numerator = 3)
  (h2 : original_denominator = 16)
  (h3 : increase_numerator = 6) :
  ∃ (increase_denominator : ℕ),
    (original_numerator + increase_numerator) / (original_denominator + increase_denominator) = 
    original_numerator / original_denominator ∧ 
    increase_denominator = 32 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_preservation_l4136_413615


namespace NUMINAMATH_CALUDE_recurrence_relation_solution_l4136_413688

def a (n : ℕ) : ℤ := -4 + 17 * n - 21 * n^2 + 5 * n^3 + n^4

theorem recurrence_relation_solution :
  (∀ n : ℕ, n ≥ 3 → a n = 3 * a (n - 1) - 3 * a (n - 2) + a (n - 3) + 24 * n - 6) ∧
  a 0 = -4 ∧
  a 1 = -2 ∧
  a 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_relation_solution_l4136_413688


namespace NUMINAMATH_CALUDE_miles_and_davis_amount_l4136_413619

-- Define the conversion rate from tablespoons of kernels to cups of popcorn
def kernels_to_popcorn (tablespoons : ℚ) : ℚ := 2 * tablespoons

-- Define the amounts of popcorn wanted by Joanie, Mitchell, and Cliff
def joanie_amount : ℚ := 3
def mitchell_amount : ℚ := 4
def cliff_amount : ℚ := 3

-- Define the total amount of kernels needed
def total_kernels : ℚ := 8

-- Theorem to prove
theorem miles_and_davis_amount :
  kernels_to_popcorn total_kernels - (joanie_amount + mitchell_amount + cliff_amount) = 6 :=
by sorry

end NUMINAMATH_CALUDE_miles_and_davis_amount_l4136_413619


namespace NUMINAMATH_CALUDE_square_difference_l4136_413691

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x + 3) * (x - 3) = 9792 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l4136_413691


namespace NUMINAMATH_CALUDE_expand_expression_l4136_413638

theorem expand_expression (x : ℝ) : -2 * (5 * x^3 - 7 * x^2 + x - 4) = -10 * x^3 + 14 * x^2 - 2 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l4136_413638


namespace NUMINAMATH_CALUDE_high_school_baseball_games_l4136_413644

/-- The number of baseball games Benny's high school played is equal to the sum of games he attended and missed -/
theorem high_school_baseball_games 
  (games_attended : ℕ) 
  (games_missed : ℕ) 
  (h1 : games_attended = 14) 
  (h2 : games_missed = 25) : 
  games_attended + games_missed = 39 := by
  sorry

end NUMINAMATH_CALUDE_high_school_baseball_games_l4136_413644


namespace NUMINAMATH_CALUDE_sunflower_height_difference_l4136_413683

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Represents a height in feet and inches -/
structure Height :=
  (feet : ℕ)
  (inches : ℕ)

/-- Converts a Height to total inches -/
def height_to_inches (h : Height) : ℕ := feet_to_inches h.feet + h.inches

theorem sunflower_height_difference :
  let sister_height : Height := ⟨4, 3⟩
  let sunflower_height : ℕ := feet_to_inches 6
  sunflower_height - height_to_inches sister_height = 21 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_height_difference_l4136_413683


namespace NUMINAMATH_CALUDE_complement_of_M_l4136_413603

def U : Set Int := {-1, -2, -3, -4}
def M : Set Int := {-2, -3}

theorem complement_of_M : U \ M = {-1, -4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l4136_413603


namespace NUMINAMATH_CALUDE_number_divided_by_14_5_equals_173_l4136_413690

theorem number_divided_by_14_5_equals_173 (x : ℝ) : 
  x / 14.5 = 173 → x = 2508.5 := by
sorry

end NUMINAMATH_CALUDE_number_divided_by_14_5_equals_173_l4136_413690


namespace NUMINAMATH_CALUDE_min_purses_needed_l4136_413626

/-- Represents a distribution of coins into purses -/
def CoinDistribution := List Nat

/-- Checks if a distribution is valid for a given number of sailors -/
def isValidDistribution (d : CoinDistribution) (n : Nat) : Prop :=
  (d.sum = 60) ∧ (∃ (x : Nat), d.sum = n * x)

/-- Checks if a distribution is valid for all required sailor counts -/
def isValidForAllSailors (d : CoinDistribution) : Prop :=
  isValidDistribution d 2 ∧
  isValidDistribution d 3 ∧
  isValidDistribution d 4 ∧
  isValidDistribution d 5

/-- The main theorem stating the minimum number of purses needed -/
theorem min_purses_needed :
  ∃ (d : CoinDistribution),
    d.length = 9 ∧
    isValidForAllSailors d ∧
    ∀ (d' : CoinDistribution),
      isValidForAllSailors d' →
      d'.length ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_purses_needed_l4136_413626


namespace NUMINAMATH_CALUDE_baseball_bat_price_baseball_bat_price_is_10_l4136_413669

/-- Calculates the selling price of a baseball bat given the total revenue and prices of other items -/
theorem baseball_bat_price (total_revenue : ℝ) (cards_price : ℝ) (glove_original_price : ℝ) (glove_discount : ℝ) (cleats_price : ℝ) (cleats_quantity : ℕ) : ℝ :=
  let glove_price := glove_original_price * (1 - glove_discount)
  let known_revenue := cards_price + glove_price + (cleats_price * cleats_quantity)
  total_revenue - known_revenue

/-- Proves that the baseball bat price is $10 given the specific conditions -/
theorem baseball_bat_price_is_10 :
  baseball_bat_price 79 25 30 0.2 10 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_baseball_bat_price_baseball_bat_price_is_10_l4136_413669


namespace NUMINAMATH_CALUDE_only_origin_satisfies_l4136_413656

def point_satisfies_inequality (x y : ℝ) : Prop :=
  x + y - 1 < 0

theorem only_origin_satisfies :
  point_satisfies_inequality 0 0 ∧
  ¬point_satisfies_inequality 2 4 ∧
  ¬point_satisfies_inequality (-1) 4 ∧
  ¬point_satisfies_inequality 1 8 :=
by sorry

end NUMINAMATH_CALUDE_only_origin_satisfies_l4136_413656


namespace NUMINAMATH_CALUDE_proposition_p_sufficient_not_necessary_for_q_l4136_413697

theorem proposition_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 < 2*x) ∧
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ x^2 < 2*x ∧ x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_sufficient_not_necessary_for_q_l4136_413697


namespace NUMINAMATH_CALUDE_weaving_problem_l4136_413650

/-- Represents the daily weaving length in an arithmetic sequence -/
def weaving_sequence (initial_length : ℚ) (daily_increase : ℚ) (day : ℕ) : ℚ :=
  initial_length + (day - 1) * daily_increase

/-- Represents the total weaving length over a period of days -/
def total_weaving (initial_length : ℚ) (daily_increase : ℚ) (days : ℕ) : ℚ :=
  (days : ℚ) * initial_length + (days * (days - 1) / 2) * daily_increase

theorem weaving_problem (initial_length daily_increase : ℚ) :
  initial_length = 5 →
  total_weaving initial_length daily_increase 30 = 390 →
  weaving_sequence initial_length daily_increase 5 = 209 / 29 := by
  sorry

end NUMINAMATH_CALUDE_weaving_problem_l4136_413650


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l4136_413647

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  equal_side : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.equal_side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * Real.sqrt (4 * (t.equal_side : ℝ)^2 - (t.base : ℝ)^2) / 4

/-- Theorem statement -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t1.base = 6 * t2.base ∧
    perimeter t1 = 399 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s1.base = 6 * s2.base →
      perimeter s1 ≥ 399) :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l4136_413647


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l4136_413694

/-- The probability of having a boy or a girl -/
def p_boy_or_girl : ℚ := 1 / 2

/-- The number of children in the family -/
def num_children : ℕ := 4

/-- The probability of having at least one boy and one girl in a family of four children -/
theorem prob_at_least_one_boy_and_girl : 
  (1 : ℚ) - (p_boy_or_girl ^ num_children + p_boy_or_girl ^ num_children) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l4136_413694


namespace NUMINAMATH_CALUDE_steel_rod_length_l4136_413643

/-- Represents the properties of a uniform steel rod -/
structure SteelRod where
  weight_per_meter : ℝ
  length : ℝ
  weight : ℝ

/-- Theorem: Given a uniform steel rod where 9 m weighs 34.2 kg, 
    the length of the rod that weighs 42.75 kg is 11.25 m -/
theorem steel_rod_length 
  (rod : SteelRod) 
  (h1 : rod.weight_per_meter = 34.2 / 9) 
  (h2 : rod.weight = 42.75) : 
  rod.length = 11.25 := by
  sorry

#check steel_rod_length

end NUMINAMATH_CALUDE_steel_rod_length_l4136_413643


namespace NUMINAMATH_CALUDE_division_problem_l4136_413620

theorem division_problem (number : ℕ) : 
  (number / 25 = 5) ∧ (number % 25 = 2) → number = 127 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4136_413620


namespace NUMINAMATH_CALUDE_withdraw_representation_l4136_413684

-- Define a type for monetary transactions
inductive Transaction
  | deposit (amount : ℤ)
  | withdraw (amount : ℤ)

-- Define a function to represent transactions
def represent : Transaction → ℤ
  | Transaction.deposit amount => amount
  | Transaction.withdraw amount => -amount

-- State the theorem
theorem withdraw_representation :
  represent (Transaction.deposit 30000) = 30000 →
  represent (Transaction.withdraw 40000) = -40000 := by
  sorry

end NUMINAMATH_CALUDE_withdraw_representation_l4136_413684


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4136_413667

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2
  arith_def : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Given conditions on the arithmetic sequence imply S₁₀ = 65 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h1 : seq.a 3 = 4)
    (h2 : seq.S 9 - seq.S 6 = 27) :
  seq.S 10 = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4136_413667


namespace NUMINAMATH_CALUDE_fourth_term_is_eleven_l4136_413609

/-- A sequence of 5 terms with specific properties -/
def CanSequence (a : Fin 5 → ℕ) : Prop :=
  a 0 = 2 ∧ 
  a 1 = 4 ∧ 
  a 2 = 7 ∧ 
  a 4 = 16 ∧
  ∀ i : Fin 3, (a (i + 1) - a i) - (a (i + 2) - a (i + 1)) = 1

theorem fourth_term_is_eleven (a : Fin 5 → ℕ) (h : CanSequence a) : a 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_eleven_l4136_413609


namespace NUMINAMATH_CALUDE_long_furred_brown_dogs_l4136_413639

theorem long_furred_brown_dogs (total : ℕ) (long_furred : ℕ) (brown : ℕ) (neither : ℕ) :
  total = 45 →
  long_furred = 29 →
  brown = 17 →
  neither = 8 →
  long_furred + brown - (total - neither) = 9 :=
by sorry

end NUMINAMATH_CALUDE_long_furred_brown_dogs_l4136_413639


namespace NUMINAMATH_CALUDE_min_value_a_l4136_413636

theorem min_value_a (a : ℝ) : 
  (a > 0 ∧ ∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 9) → a ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l4136_413636


namespace NUMINAMATH_CALUDE_anna_mean_score_l4136_413655

def scores : List ℝ := [88, 90, 92, 95, 96, 98, 100, 102, 105]

def timothy_count : ℕ := 5
def anna_count : ℕ := 4
def timothy_mean : ℝ := 95

theorem anna_mean_score (h1 : scores.length = timothy_count + anna_count)
                        (h2 : timothy_count * timothy_mean = scores.sum - anna_count * anna_mean) :
  anna_mean = 97.75 := by
  sorry

end NUMINAMATH_CALUDE_anna_mean_score_l4136_413655


namespace NUMINAMATH_CALUDE_second_white_given_first_white_l4136_413698

/-- Represents the number of white balls initially in the bag -/
def white_balls : ℕ := 5

/-- Represents the number of red balls initially in the bag -/
def red_balls : ℕ := 3

/-- Represents the total number of balls initially in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- Represents the probability of drawing a white ball on the second draw
    given that the first draw was white -/
def prob_second_white_given_first_white : ℚ := 4 / 7

/-- Theorem stating that the probability of drawing a white ball on the second draw
    given that the first draw was white is 4/7 -/
theorem second_white_given_first_white :
  prob_second_white_given_first_white = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_second_white_given_first_white_l4136_413698


namespace NUMINAMATH_CALUDE_equation_solution_l4136_413666

theorem equation_solution :
  ∃! x : ℝ, (3 : ℝ) ^ (2 * x + 2) = (1 : ℝ) / 81 :=
by
  use -3
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4136_413666


namespace NUMINAMATH_CALUDE_second_grade_girls_l4136_413660

theorem second_grade_girls (boys_second : ℕ) (total_students : ℕ) :
  boys_second = 20 →
  total_students = 93 →
  ∃ (girls_second : ℕ),
    girls_second = 11 ∧
    total_students = boys_second + girls_second + 2 * (boys_second + girls_second) :=
by sorry

end NUMINAMATH_CALUDE_second_grade_girls_l4136_413660


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l4136_413679

theorem smallest_undefined_value (x : ℝ) : 
  (∀ y < 1/9, ∃ z, (y - 2) / (9*y^2 - 98*y + 21) = z) ∧ 
  ¬∃ z, ((1/9 : ℝ) - 2) / (9*(1/9)^2 - 98*(1/9) + 21) = z :=
sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l4136_413679


namespace NUMINAMATH_CALUDE_expression_evaluation_l4136_413614

theorem expression_evaluation (a b : ℝ) (h : (a + 1)^2 + |b + 1| = 0) :
  1 - (a^2 + 2*a*b + b^2) / (a^2 - a*b) / ((a + b) / (a - b)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4136_413614


namespace NUMINAMATH_CALUDE_unique_solution_l4136_413642

theorem unique_solution (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.cos (π * x)^2 + 2 * Real.sin (π * y) = 1)
  (h2 : Real.sin (π * x) + Real.sin (π * y) = 0)
  (h3 : x^2 - y^2 = 12) :
  x = 4 ∧ y = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l4136_413642


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l4136_413676

theorem baker_cakes_sold (pastries_sold : ℕ) (pastry_cake_difference : ℕ) 
  (h1 : pastries_sold = 154)
  (h2 : pastry_cake_difference = 76) :
  pastries_sold - pastry_cake_difference = 78 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l4136_413676


namespace NUMINAMATH_CALUDE_secretaries_working_hours_l4136_413611

theorem secretaries_working_hours (t₁ t₂ t₃ : ℝ) : 
  t₁ > 0 ∧ t₂ > 0 ∧ t₃ > 0 →  -- Ensuring positive working times
  t₂ = 2 * t₁ →               -- Ratio condition for t₂
  t₃ = 5 * t₁ →               -- Ratio condition for t₃
  t₃ = 75 →                   -- Longest working time
  t₁ + t₂ + t₃ = 120 :=       -- Combined total
by sorry


end NUMINAMATH_CALUDE_secretaries_working_hours_l4136_413611


namespace NUMINAMATH_CALUDE_basketball_game_free_throws_l4136_413686

theorem basketball_game_free_throws :
  ∀ (three_pointers two_pointers free_throws : ℕ),
    three_pointers + two_pointers + free_throws = 32 →
    two_pointers = 4 * three_pointers + 3 →
    3 * three_pointers + 2 * two_pointers + free_throws = 65 →
    free_throws = 4 :=
by sorry

end NUMINAMATH_CALUDE_basketball_game_free_throws_l4136_413686


namespace NUMINAMATH_CALUDE_distribute_seven_to_twelve_l4136_413629

/-- The number of ways to distribute distinct items to recipients -/
def distribute_items (num_items : ℕ) (num_recipients : ℕ) : ℕ :=
  num_recipients ^ num_items

/-- Theorem: Distributing 7 distinct items to 12 recipients results in 35,831,808 ways -/
theorem distribute_seven_to_twelve :
  distribute_items 7 12 = 35831808 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_to_twelve_l4136_413629


namespace NUMINAMATH_CALUDE_baseball_weight_l4136_413645

theorem baseball_weight (total_weight : ℝ) (soccer_ball_weight : ℝ) (baseball_count : ℕ) (soccer_ball_count : ℕ) :
  total_weight = 10.98 →
  soccer_ball_weight = 0.8 →
  baseball_count = 7 →
  soccer_ball_count = 9 →
  (soccer_ball_count * soccer_ball_weight + baseball_count * ((total_weight - soccer_ball_count * soccer_ball_weight) / baseball_count) = total_weight) ∧
  ((total_weight - soccer_ball_count * soccer_ball_weight) / baseball_count = 0.54) :=
by
  sorry

end NUMINAMATH_CALUDE_baseball_weight_l4136_413645


namespace NUMINAMATH_CALUDE_pencils_added_l4136_413600

theorem pencils_added (initial : ℕ) (final : ℕ) (h1 : initial = 115) (h2 : final = 215) :
  final - initial = 100 := by
sorry

end NUMINAMATH_CALUDE_pencils_added_l4136_413600


namespace NUMINAMATH_CALUDE_closure_M_intersect_N_l4136_413625

-- Define the set M
def M : Set ℝ := {x | 2/x < 1}

-- Define the set N
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 1)}

-- State the theorem
theorem closure_M_intersect_N :
  (closure M) ∩ N = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_closure_M_intersect_N_l4136_413625


namespace NUMINAMATH_CALUDE_sum_remainder_zero_l4136_413606

def arithmetic_sum (a₁ aₙ n : ℕ) : ℕ := n * (a₁ + aₙ) / 2

theorem sum_remainder_zero : 
  let a₁ := 6
  let d := 6
  let aₙ := 288
  let n := (aₙ - a₁) / d + 1
  (arithmetic_sum a₁ aₙ n) % 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_remainder_zero_l4136_413606


namespace NUMINAMATH_CALUDE_max_value_theorem_l4136_413670

theorem max_value_theorem (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) :
  ∃ (M : ℝ), M = 15 ∧ ∀ (a b : ℝ), 2 * a^2 - 6 * a + b^2 = 0 → a^2 + b^2 + 2 * a ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l4136_413670


namespace NUMINAMATH_CALUDE_vector_equation_solution_l4136_413640

def a : ℝ × ℝ × ℝ := (1, 3, -2)
def b : ℝ × ℝ × ℝ := (2, 1, 0)

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1,
   v.2.2 * w.1 - v.1 * w.2.2,
   v.1 * w.2.1 - v.2.1 * w.1)

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2.1 * w.2.1 + v.2.2 * w.2.2

theorem vector_equation_solution :
  ∃ (p q r : ℝ),
    (5, 2, -3) = (p * a.1 + q * b.1 + r * (cross_product a b).1,
                  p * a.2.1 + q * b.2.1 + r * (cross_product a b).2.1,
                  p * a.2.2 + q * b.2.2 + r * (cross_product a b).2.2) →
    r = 17 / 45 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l4136_413640


namespace NUMINAMATH_CALUDE_total_weekly_eggs_l4136_413617

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens supplied to Store A daily -/
def store_a_dozens : ℕ := 5

/-- The number of eggs supplied to Store B daily -/
def store_b_eggs : ℕ := 30

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total number of eggs supplied to both stores in a week is 630 -/
theorem total_weekly_eggs : 
  (store_a_dozens * dozen + store_b_eggs) * days_in_week = 630 := by
  sorry

end NUMINAMATH_CALUDE_total_weekly_eggs_l4136_413617


namespace NUMINAMATH_CALUDE_exponent_multiplication_l4136_413601

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l4136_413601


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4136_413648

theorem inequality_solution_set (a : ℝ) (h : (4 : ℝ)^a = 2^(a + 2)) :
  {x : ℝ | a^(2*x + 1) > a^(x - 1)} = {x : ℝ | x > -2} :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4136_413648


namespace NUMINAMATH_CALUDE_binomial_coefficient_17_16_l4136_413632

theorem binomial_coefficient_17_16 : Nat.choose 17 16 = 17 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_17_16_l4136_413632


namespace NUMINAMATH_CALUDE_three_custom_op_three_equals_six_l4136_413630

-- Define the custom operation
def customOp (m n : ℕ) : ℕ := n ^ 2 - m

-- State the theorem
theorem three_custom_op_three_equals_six :
  customOp 3 3 = 6 := by sorry

end NUMINAMATH_CALUDE_three_custom_op_three_equals_six_l4136_413630


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l4136_413696

theorem arithmetic_calculation : 3 * 5 * 7 + 15 / 3 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l4136_413696


namespace NUMINAMATH_CALUDE_hexadecimal_to_decimal_l4136_413612

/-- Given that the hexadecimal number (10k5)₆ (where k is a positive integer) 
    is equivalent to the decimal number 239, prove that k = 3. -/
theorem hexadecimal_to_decimal (k : ℕ+) : 
  (1 * 6^3 + k * 6 + 5) = 239 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_hexadecimal_to_decimal_l4136_413612


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l4136_413653

/-- The area of the triangle formed by the tangent line to y = e^x at (2, e^2) and the coordinate axes -/
theorem tangent_triangle_area : 
  let f : ℝ → ℝ := fun x ↦ Real.exp x
  let tangent_point : ℝ × ℝ := (2, Real.exp 2)
  let slope : ℝ := Real.exp 2
  let x_intercept : ℝ := 1
  let y_intercept : ℝ := Real.exp 2
  let triangle_area : ℝ := (1/2) * Real.exp 2
  triangle_area = (1/2) * y_intercept * x_intercept :=
by sorry


end NUMINAMATH_CALUDE_tangent_triangle_area_l4136_413653


namespace NUMINAMATH_CALUDE_negative_half_greater_than_negative_two_thirds_l4136_413672

theorem negative_half_greater_than_negative_two_thirds :
  -0.5 > -(2/3) := by
  sorry

end NUMINAMATH_CALUDE_negative_half_greater_than_negative_two_thirds_l4136_413672


namespace NUMINAMATH_CALUDE_set_equality_implies_p_equals_three_l4136_413654

theorem set_equality_implies_p_equals_three (p : ℝ) : 
  let U : Set ℝ := {x | x^2 - 3*x + 2 = 0}
  let A : Set ℝ := {x | x^2 - p*x + 2 = 0}
  (U \ A = ∅) → p = 3 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_p_equals_three_l4136_413654


namespace NUMINAMATH_CALUDE_log_101600_div_3_l4136_413602

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- State the theorem
theorem log_101600_div_3 : log (101600 / 3) = 0.1249 := by
  -- Given conditions
  have h1 : log 102 = 0.3010 := by sorry
  have h2 : log 3 = 0.4771 := by sorry

  -- Proof steps
  sorry

end NUMINAMATH_CALUDE_log_101600_div_3_l4136_413602


namespace NUMINAMATH_CALUDE_soft_drink_added_sugar_percentage_l4136_413634

theorem soft_drink_added_sugar_percentage (
  soft_drink_calories : ℕ)
  (candy_bar_sugar_calories : ℕ)
  (candy_bars_taken : ℕ)
  (recommended_sugar_intake : ℕ)
  (exceeded_percentage : ℕ)
  (h1 : soft_drink_calories = 2500)
  (h2 : candy_bar_sugar_calories = 25)
  (h3 : candy_bars_taken = 7)
  (h4 : recommended_sugar_intake = 150)
  (h5 : exceeded_percentage = 100) :
  (((recommended_sugar_intake * (100 + exceeded_percentage) / 100) -
    (candy_bar_sugar_calories * candy_bars_taken)) * 100) /
    soft_drink_calories = 5 := by
  sorry

end NUMINAMATH_CALUDE_soft_drink_added_sugar_percentage_l4136_413634


namespace NUMINAMATH_CALUDE_expand_product_l4136_413607

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 2*x + 5) = x^4 + 2*x^3 - 4*x^2 - 18*x - 45 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l4136_413607


namespace NUMINAMATH_CALUDE_circle_center_sum_l4136_413685

/-- The sum of the x and y coordinates of the center of a circle with equation x^2 + y^2 = 4x - 6y + 9 is -1 -/
theorem circle_center_sum (x y : ℝ) : x^2 + y^2 = 4*x - 6*y + 9 → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l4136_413685


namespace NUMINAMATH_CALUDE_simple_interest_fraction_l4136_413663

/-- 
Given a principal sum P, proves that the simple interest calculated for 8 years 
at a rate of 2.5% per annum is equal to 1/5 of the principal sum.
-/
theorem simple_interest_fraction (P : ℝ) (P_pos : P > 0) : 
  (P * 2.5 * 8) / 100 = P * (1 / 5) := by
  sorry

#check simple_interest_fraction

end NUMINAMATH_CALUDE_simple_interest_fraction_l4136_413663


namespace NUMINAMATH_CALUDE_no_nonzero_solution_for_diophantine_equation_l4136_413628

theorem no_nonzero_solution_for_diophantine_equation :
  ∀ (x y z : ℤ), 2 * x^4 + y^4 = 7 * z^4 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_solution_for_diophantine_equation_l4136_413628


namespace NUMINAMATH_CALUDE_sum_of_x_y_z_l4136_413613

theorem sum_of_x_y_z (x y z : ℝ) : y = 3*x → z = 2*y → x + y + z = 10*x := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_y_z_l4136_413613


namespace NUMINAMATH_CALUDE_yellow_papers_in_ten_by_ten_square_l4136_413623

/-- Represents a square arrangement of colored papers -/
structure ColoredSquare where
  size : Nat
  redPeriphery : Bool

/-- Calculates the number of yellow papers in a ColoredSquare -/
def yellowPapers (square : ColoredSquare) : Nat :=
  if square.redPeriphery then
    square.size * square.size - (4 * square.size - 4)
  else
    square.size * square.size

/-- Theorem stating that a 10x10 ColoredSquare with red periphery has 64 yellow papers -/
theorem yellow_papers_in_ten_by_ten_square :
  yellowPapers { size := 10, redPeriphery := true } = 64 := by
  sorry

#eval yellowPapers { size := 10, redPeriphery := true }

end NUMINAMATH_CALUDE_yellow_papers_in_ten_by_ten_square_l4136_413623


namespace NUMINAMATH_CALUDE_problem_solution_l4136_413673

def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}

def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

theorem problem_solution :
  (∀ x, x ∈ (A ∩ B (-4)) ↔ (1/2 ≤ x ∧ x < 2)) ∧
  (∀ x, x ∈ (A ∪ B (-4)) ↔ (-2 < x ∧ x ≤ 3)) ∧
  (∀ a, (Aᶜ ∩ B a = B a) ↔ a ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l4136_413673


namespace NUMINAMATH_CALUDE_orchestra_price_is_12_l4136_413631

/-- Represents the pricing and sales of theater tickets --/
structure TheaterSales where
  orchestra_price : ℝ
  balcony_price : ℝ
  orchestra_tickets : ℕ
  balcony_tickets : ℕ

/-- Theorem stating the price of orchestra seats given the conditions --/
theorem orchestra_price_is_12 (sales : TheaterSales) :
  sales.balcony_price = 8 ∧
  sales.orchestra_tickets + sales.balcony_tickets = 380 ∧
  sales.orchestra_price * sales.orchestra_tickets + sales.balcony_price * sales.balcony_tickets = 3320 ∧
  sales.balcony_tickets = sales.orchestra_tickets + 240
  → sales.orchestra_price = 12 := by
  sorry


end NUMINAMATH_CALUDE_orchestra_price_is_12_l4136_413631


namespace NUMINAMATH_CALUDE_area_of_triangles_is_four_l4136_413680

/-- A regular octagon with side length 2 cm -/
structure RegularOctagon where
  side_length : ℝ
  is_two_cm : side_length = 2

/-- The area of the four triangles formed when two rectangles are drawn
    connecting opposite vertices in a regular octagon -/
def area_of_four_triangles (octagon : RegularOctagon) : ℝ := 4

/-- Theorem stating that the area of the four triangles is 4 cm² -/
theorem area_of_triangles_is_four (octagon : RegularOctagon) :
  area_of_four_triangles octagon = 4 := by
  sorry

#check area_of_triangles_is_four

end NUMINAMATH_CALUDE_area_of_triangles_is_four_l4136_413680


namespace NUMINAMATH_CALUDE_shortest_distance_principle_applies_l4136_413633

-- Define the phenomena
inductive Phenomenon
  | woodenBarFixing
  | treePlanting
  | electricWireLaying
  | roadStraightening

-- Define the principle
def shortestDistancePrinciple (p : Phenomenon) : Prop :=
  match p with
  | Phenomenon.electricWireLaying => true
  | Phenomenon.roadStraightening => true
  | _ => false

-- Theorem statement
theorem shortest_distance_principle_applies :
  (∀ p : Phenomenon, shortestDistancePrinciple p ↔ 
    (p = Phenomenon.electricWireLaying ∨ p = Phenomenon.roadStraightening)) := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_principle_applies_l4136_413633


namespace NUMINAMATH_CALUDE_largest_intersection_x_coordinate_l4136_413699

/-- The polynomial function -/
def P (d : ℝ) (x : ℝ) : ℝ := x^6 - 5*x^5 + 5*x^4 + 5*x^3 + d*x^2

/-- The parabola function -/
def Q (e f g : ℝ) (x : ℝ) : ℝ := e*x^2 + f*x + g

/-- The difference between the polynomial and the parabola -/
def R (d e f g : ℝ) (x : ℝ) : ℝ := P d x - Q e f g x

theorem largest_intersection_x_coordinate
  (d e f g : ℝ)
  (h1 : ∃ a b c : ℝ, ∀ x : ℝ, R d e f g x = (x - a)^2 * (x - b)^2 * (x - c)^2)
  (h2 : ∃! a b c : ℝ, ∀ x : ℝ, R d e f g x = (x - a)^2 * (x - b)^2 * (x - c)^2) :
  ∃ x : ℝ, (∀ y : ℝ, R d e f g y = 0 → y ≤ x) ∧ R d e f g x = 0 ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_largest_intersection_x_coordinate_l4136_413699


namespace NUMINAMATH_CALUDE_schedule_theorem_l4136_413689

-- Define the number of classes
def total_classes : ℕ := 6

-- Define the number of morning slots
def morning_slots : ℕ := 4

-- Define the number of afternoon slots
def afternoon_slots : ℕ := 2

-- Define the function to calculate the number of arrangements
def schedule_arrangements (n : ℕ) (m : ℕ) (a : ℕ) : ℕ :=
  (m.choose 1) * (a.choose 1) * (n - 2).factorial

-- Theorem statement
theorem schedule_theorem :
  schedule_arrangements total_classes morning_slots afternoon_slots = 192 :=
by sorry

end NUMINAMATH_CALUDE_schedule_theorem_l4136_413689


namespace NUMINAMATH_CALUDE_geometric_progression_common_ratio_l4136_413624

theorem geometric_progression_common_ratio :
  ∀ (a : ℝ) (r : ℝ),
    a > 0 →  -- First term is positive
    r > 0 →  -- Common ratio is positive (to ensure all terms are positive)
    a = a * r + a * r^2 + a * r^3 →  -- First term equals sum of next three terms
    r = (Real.sqrt 5 - 1) / 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_common_ratio_l4136_413624


namespace NUMINAMATH_CALUDE_perpendicular_line_implies_parallel_planes_l4136_413687

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_line_implies_parallel_planes 
  (α β : Plane) (l : Line) : 
  (perpendicular l α ∧ perpendicular l β) → parallel α β := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_implies_parallel_planes_l4136_413687


namespace NUMINAMATH_CALUDE_right_triangle_sets_l4136_413692

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  ¬(is_right_triangle 3 4 6) ∧
  (is_right_triangle 7 24 25) ∧
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle 9 12 15) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l4136_413692


namespace NUMINAMATH_CALUDE_complex_division_result_abs_value_result_l4136_413637

open Complex

def z₁ : ℂ := 1 - I
def z₂ : ℂ := 4 + 6 * I

theorem complex_division_result : z₂ / z₁ = -1 + 5 * I := by sorry

theorem abs_value_result (b : ℝ) (z : ℂ) (h : z = 1 + b * I) 
  (h_real : (z + z₁).im = 0) : abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_division_result_abs_value_result_l4136_413637


namespace NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_l4136_413659

theorem not_p_and_q_implies_at_most_one (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_l4136_413659


namespace NUMINAMATH_CALUDE_phone_number_proof_l4136_413658

def is_harmonic_mean (a b c : ℕ) : Prop :=
  2 * b * a * c = b * (a + c)

def is_six_digit (a b c d : ℕ) : Prop :=
  100000 ≤ a * 100000 + b * 10000 + c * 100 + d ∧
  a * 100000 + b * 10000 + c * 100 + d < 1000000

theorem phone_number_proof (a b c d : ℕ) : 
  a = 6 ∧ b = 8 ∧ c = 12 ∧ d = 24 →
  a < b ∧ b < c ∧ c < d ∧
  is_harmonic_mean a b c ∧
  is_harmonic_mean b c d ∧
  is_six_digit a b c d := by
  sorry

#eval [6, 8, 12, 24].map (λ x => x.toDigits 10)

end NUMINAMATH_CALUDE_phone_number_proof_l4136_413658


namespace NUMINAMATH_CALUDE_jeffs_weekly_running_time_l4136_413649

/-- Represents Jeff's weekly running schedule -/
structure RunningSchedule where
  normalDays : Nat  -- Number of days with normal running time
  normalTime : Nat  -- Normal running time in minutes
  thursdayReduction : Nat  -- Minutes reduced on Thursday
  fridayIncrease : Nat  -- Minutes increased on Friday

/-- Calculates the total running time for the week given a RunningSchedule -/
def totalRunningTime (schedule : RunningSchedule) : Nat :=
  schedule.normalDays * schedule.normalTime +
  (schedule.normalTime - schedule.thursdayReduction) +
  (schedule.normalTime + schedule.fridayIncrease)

/-- Theorem stating that Jeff's total running time for the week is 290 minutes -/
theorem jeffs_weekly_running_time :
  ∀ (schedule : RunningSchedule),
    schedule.normalDays = 3 ∧
    schedule.normalTime = 60 ∧
    schedule.thursdayReduction = 20 ∧
    schedule.fridayIncrease = 10 →
    totalRunningTime schedule = 290 := by
  sorry

end NUMINAMATH_CALUDE_jeffs_weekly_running_time_l4136_413649


namespace NUMINAMATH_CALUDE_counterexample_acute_angles_sum_l4136_413651

theorem counterexample_acute_angles_sum : 
  ∃ (A B : ℝ), 0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ A + B ≥ 90 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_acute_angles_sum_l4136_413651


namespace NUMINAMATH_CALUDE_percentage_change_condition_l4136_413682

theorem percentage_change_condition (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hqlt : q < 100) (hM : M > 0) : 
  (M * (1 + p / 100) * (1 - q / 100) > M) ↔ (p > 100 * q / (100 - q)) := by
  sorry

end NUMINAMATH_CALUDE_percentage_change_condition_l4136_413682


namespace NUMINAMATH_CALUDE_complex_number_and_pure_imaginary_l4136_413605

-- Define the complex number z
def z : ℂ := sorry

-- Define the real number m
def m : ℝ := sorry

-- Theorem statement
theorem complex_number_and_pure_imaginary :
  (Complex.abs z = Real.sqrt 2) ∧
  (z.im = 1) ∧
  (z.re < 0) ∧
  (z = -1 + Complex.I) ∧
  (∃ (k : ℝ), m^2 + m + m * z^2 = k * Complex.I) →
  (z = -1 + Complex.I) ∧ (m = -1) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_and_pure_imaginary_l4136_413605


namespace NUMINAMATH_CALUDE_card_distribution_result_l4136_413665

/-- Represents the card distribution problem --/
def card_distribution (jimmy_initial bob_initial sarah_initial : ℕ)
  (jimmy_to_bob jimmy_to_mary : ℕ)
  (sarah_friends : ℕ) : Prop :=
  let bob_after_jimmy := bob_initial + jimmy_to_bob
  let bob_to_sarah := bob_after_jimmy / 3
  let sarah_after_bob := sarah_initial + bob_to_sarah
  let sarah_to_friends := (sarah_after_bob / sarah_friends) * sarah_friends
  let jimmy_final := jimmy_initial - jimmy_to_bob - jimmy_to_mary
  let sarah_final := sarah_after_bob - sarah_to_friends
  let friends_cards := sarah_to_friends / sarah_friends
  jimmy_final = 50 ∧ sarah_final = 1 ∧ friends_cards = 3

/-- The main theorem stating the result of the card distribution --/
theorem card_distribution_result :
  card_distribution 68 5 7 6 12 3 :=
sorry

end NUMINAMATH_CALUDE_card_distribution_result_l4136_413665


namespace NUMINAMATH_CALUDE_books_read_difference_l4136_413622

def total_books : ℕ := 20
def peter_percentage : ℚ := 40 / 100
def brother_percentage : ℚ := 10 / 100

theorem books_read_difference : 
  (peter_percentage * total_books : ℚ).floor - (brother_percentage * total_books : ℚ).floor = 6 := by
  sorry

end NUMINAMATH_CALUDE_books_read_difference_l4136_413622


namespace NUMINAMATH_CALUDE_well_depth_l4136_413621

/-- Proves that a circular well with diameter 2 meters and volume 31.41592653589793 cubic meters has a depth of 10 meters -/
theorem well_depth (diameter : ℝ) (volume : ℝ) (depth : ℝ) : 
  diameter = 2 → 
  volume = 31.41592653589793 → 
  volume = Real.pi * (diameter / 2)^2 * depth → 
  depth = 10 := by sorry

end NUMINAMATH_CALUDE_well_depth_l4136_413621


namespace NUMINAMATH_CALUDE_sum_inequality_l4136_413695

theorem sum_inequality (a b c d : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d) :
  a * d + b * c < a * c + b * d ∧ a * c + b * d < a * b + c * d := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l4136_413695


namespace NUMINAMATH_CALUDE_fan_airflow_in_week_l4136_413627

/-- Calculates the total airflow created by a fan in one week -/
theorem fan_airflow_in_week 
  (airflow_rate : ℝ) 
  (daily_operation_time : ℝ) 
  (days_in_week : ℕ) 
  (seconds_in_minute : ℕ) : 
  airflow_rate * daily_operation_time * (days_in_week : ℝ) * (seconds_in_minute : ℝ) = 42000 :=
by
  -- Assuming airflow_rate = 10, daily_operation_time = 10, days_in_week = 7, seconds_in_minute = 60
  sorry

#check fan_airflow_in_week

end NUMINAMATH_CALUDE_fan_airflow_in_week_l4136_413627


namespace NUMINAMATH_CALUDE_job_completion_time_l4136_413616

def job_completion (x : ℝ) : Prop :=
  ∃ (y : ℝ),
    (1 / (x + 5) + 1 / (x + 3) + 1 / (2 * y) = 1 / x) ∧
    (1 / (x + 3) + 1 / y = 1 / x) ∧
    (y > 0) ∧ (x > 0)

theorem job_completion_time : ∃ (x : ℝ), job_completion x ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l4136_413616


namespace NUMINAMATH_CALUDE_target_hit_probability_l4136_413664

theorem target_hit_probability (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2) (h_B : p_B = 1/3) (h_C : p_C = 1/4) :
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l4136_413664


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_of_roots_l4136_413652

theorem sum_of_fourth_powers_of_roots (p q r s : ℂ) : 
  (p^4 - p^3 + p^2 - 3*p + 3 = 0) →
  (q^4 - q^3 + q^2 - 3*q + 3 = 0) →
  (r^4 - r^3 + r^2 - 3*r + 3 = 0) →
  (s^4 - s^3 + s^2 - 3*s + 3 = 0) →
  p^4 + q^4 + r^4 + s^4 = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_of_roots_l4136_413652


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l4136_413678

theorem quadratic_completing_square (x : ℝ) : 
  4 * x^2 - 8 * x - 320 = 0 → ∃ s : ℝ, (x - 1)^2 = s ∧ s = 81 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l4136_413678


namespace NUMINAMATH_CALUDE_product_plus_one_is_square_l4136_413646

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) :
  x * y + 1 = (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_is_square_l4136_413646


namespace NUMINAMATH_CALUDE_equation_equivalence_l4136_413635

theorem equation_equivalence (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) ↔ (y = 6 * x / (x - 9)) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l4136_413635


namespace NUMINAMATH_CALUDE_square_geq_linear_l4136_413657

theorem square_geq_linear (a b : ℝ) (ha : a > 0) : a^2 ≥ 2*b - a := by sorry

end NUMINAMATH_CALUDE_square_geq_linear_l4136_413657


namespace NUMINAMATH_CALUDE_ellipse_foci_l4136_413674

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := y^2 / 9 + x^2 / 4 = 1

-- Define the foci coordinates
def foci_coordinates : Set (ℝ × ℝ) := {(0, Real.sqrt 5), (0, -Real.sqrt 5)}

-- Theorem statement
theorem ellipse_foci : 
  ∀ (x y : ℝ), ellipse x y → (x, y) ∈ foci_coordinates ↔ 
  (x = 0 ∧ y = Real.sqrt 5) ∨ (x = 0 ∧ y = -Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l4136_413674


namespace NUMINAMATH_CALUDE_altitude_divides_triangle_iff_right_angle_or_isosceles_l4136_413662

/-- Triangle ABC with altitude h_a from vertex A to side BC -/
structure Triangle :=
  (A B C : Point)
  (h_a : Point)

/-- The altitude h_a divides triangle ABC into two similar triangles -/
def divides_into_similar_triangles (t : Triangle) : Prop :=
  sorry

/-- Angle A is a right angle -/
def is_right_angle_at_A (t : Triangle) : Prop :=
  sorry

/-- Triangle ABC is isosceles with AB = AC -/
def is_isosceles (t : Triangle) : Prop :=
  sorry

/-- Theorem: The altitude h_a of triangle ABC divides it into two similar triangles
    if and only if either angle A is a right angle or AB = AC -/
theorem altitude_divides_triangle_iff_right_angle_or_isosceles (t : Triangle) :
  divides_into_similar_triangles t ↔ (is_right_angle_at_A t ∨ is_isosceles t) :=
sorry

end NUMINAMATH_CALUDE_altitude_divides_triangle_iff_right_angle_or_isosceles_l4136_413662


namespace NUMINAMATH_CALUDE_star_seven_three_l4136_413681

-- Define the ⋆ operation
def star (a b : ℤ) : ℤ := 4*a + 3*b - 2*a*b

-- Theorem statement
theorem star_seven_three : star 7 3 = -5 := by
  sorry

end NUMINAMATH_CALUDE_star_seven_three_l4136_413681


namespace NUMINAMATH_CALUDE_pencils_needed_theorem_l4136_413661

/-- Calculates the number of pencils needed to be purchased given the initial distribution and shortage --/
def pencils_to_purchase (box_a_pencils box_b_pencils : ℕ) (box_a_classrooms box_b_classrooms : ℕ) (shortage : ℕ) : ℕ :=
  let total_classrooms := box_a_classrooms + box_b_classrooms
  let box_a_per_class := box_a_pencils / box_a_classrooms
  let box_b_per_class := box_b_pencils / box_b_classrooms
  let total_per_class := box_a_per_class + box_b_per_class
  let shortage_per_class := (shortage + total_classrooms - 1) / total_classrooms
  shortage_per_class * total_classrooms

theorem pencils_needed_theorem :
  pencils_to_purchase 480 735 6 9 85 = 90 := by
  sorry

end NUMINAMATH_CALUDE_pencils_needed_theorem_l4136_413661


namespace NUMINAMATH_CALUDE_coin_arrangements_count_l4136_413608

/-- The number of indistinguishable gold coins -/
def num_gold_coins : ℕ := 5

/-- The number of indistinguishable silver coins -/
def num_silver_coins : ℕ := 5

/-- The total number of coins -/
def total_coins : ℕ := num_gold_coins + num_silver_coins

/-- The number of ways to arrange the gold and silver coins -/
def color_arrangements : ℕ := Nat.choose total_coins num_gold_coins

/-- The number of possible orientations to avoid face-to-face adjacency -/
def orientation_arrangements : ℕ := total_coins + 1

/-- The total number of distinguishable arrangements -/
def total_arrangements : ℕ := color_arrangements * orientation_arrangements

theorem coin_arrangements_count :
  total_arrangements = 2772 :=
sorry

end NUMINAMATH_CALUDE_coin_arrangements_count_l4136_413608


namespace NUMINAMATH_CALUDE_total_wicks_count_l4136_413668

/-- The length of the spool in feet -/
def spool_length : ℕ := 15

/-- The length of short wicks in inches -/
def short_wick : ℕ := 6

/-- The length of long wicks in inches -/
def long_wick : ℕ := 12

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

theorem total_wicks_count : ∃ (n : ℕ), 
  n * short_wick + n * long_wick = spool_length * feet_to_inches ∧
  n + n = 20 := by sorry

end NUMINAMATH_CALUDE_total_wicks_count_l4136_413668


namespace NUMINAMATH_CALUDE_area_of_sixth_rectangle_l4136_413693

/-- Given a rectangle divided into six smaller rectangles, prove that if five of these rectangles
    have areas 126, 63, 40, 20, and 161, then the area of the sixth rectangle is 101. -/
theorem area_of_sixth_rectangle (
  total_area : ℝ)
  (area1 area2 area3 area4 area5 : ℝ)
  (h1 : area1 = 126)
  (h2 : area2 = 63)
  (h3 : area3 = 40)
  (h4 : area4 = 20)
  (h5 : area5 = 161)
  (h_sum : total_area = area1 + area2 + area3 + area4 + area5 + (total_area - (area1 + area2 + area3 + area4 + area5))) :
  total_area - (area1 + area2 + area3 + area4 + area5) = 101 := by
  sorry


end NUMINAMATH_CALUDE_area_of_sixth_rectangle_l4136_413693


namespace NUMINAMATH_CALUDE_complex_calculation_l4136_413610

theorem complex_calculation (a b : ℂ) (ha : a = 3 + 2*I) (hb : b = 2 - 3*I) :
  3*a - 4*b = 1 + 18*I := by
sorry

end NUMINAMATH_CALUDE_complex_calculation_l4136_413610


namespace NUMINAMATH_CALUDE_trapezoid_triangle_area_l4136_413618

/-- A trapezoid with vertices A, B, C, and D -/
structure Trapezoid :=
  (A B C D : ℝ × ℝ)

/-- The area of a trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

/-- The length of a line segment between two points -/
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem trapezoid_triangle_area (t : Trapezoid) :
  area t = 30 ∧ length t.C t.D = 3 * length t.A t.B →
  triangleArea t.A t.B t.C = 7.5 := by sorry

end NUMINAMATH_CALUDE_trapezoid_triangle_area_l4136_413618


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l4136_413604

/-- Given two points M and N in the plane, this theorem states that
    the equation of the perpendicular bisector of line segment MN
    is x - y + 3 = 0. -/
theorem perpendicular_bisector_equation (M N : ℝ × ℝ) :
  M = (-1, 6) →
  N = (3, 2) →
  ∃ (f : ℝ → ℝ), 
    (∀ x y, f x = y ↔ x - y + 3 = 0) ∧
    (∀ p : ℝ × ℝ, f p.1 = p.2 ↔ 
      (p.1 - M.1)^2 + (p.2 - M.2)^2 = (p.1 - N.1)^2 + (p.2 - N.2)^2 ∧
      (p.1 - M.1) * (N.1 - M.1) + (p.2 - M.2) * (N.2 - M.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l4136_413604


namespace NUMINAMATH_CALUDE_sum_of_two_squares_l4136_413671

theorem sum_of_two_squares (n m : ℕ) (h : 2 * m = n^2 + 1) :
  ∃ k : ℕ, m = k^2 + (k - 1)^2 := by sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_l4136_413671


namespace NUMINAMATH_CALUDE_rectangle_side_lengths_l4136_413641

theorem rectangle_side_lengths (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) 
  (h4 : a * b = 2 * (a + b)) : a < 4 ∧ b > 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_lengths_l4136_413641


namespace NUMINAMATH_CALUDE_M_intersect_N_l4136_413675

def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}

def N : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem M_intersect_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l4136_413675

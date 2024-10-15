import Mathlib

namespace NUMINAMATH_CALUDE_apple_preference_percentage_l2266_226653

-- Define the fruit categories
inductive Fruit
| Apple
| Banana
| Cherry
| Orange
| Pear

-- Define the function that gives the frequency for each fruit
def frequency (f : Fruit) : ℕ :=
  match f with
  | Fruit.Apple => 75
  | Fruit.Banana => 80
  | Fruit.Cherry => 45
  | Fruit.Orange => 100
  | Fruit.Pear => 50

-- Define the total number of responses
def total_responses : ℕ := 
  frequency Fruit.Apple + frequency Fruit.Banana + frequency Fruit.Cherry + 
  frequency Fruit.Orange + frequency Fruit.Pear

-- Theorem: The percentage of people who preferred apples is 21%
theorem apple_preference_percentage : 
  (frequency Fruit.Apple : ℚ) / (total_responses : ℚ) * 100 = 21 := by
  sorry

end NUMINAMATH_CALUDE_apple_preference_percentage_l2266_226653


namespace NUMINAMATH_CALUDE_seven_lines_29_regions_l2266_226635

/-- The number of regions created by n lines in a plane, where no two lines are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1) / 2)

/-- Seven straight lines in a plane with no two parallel and no three concurrent divide the plane into 29 regions -/
theorem seven_lines_29_regions : num_regions 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_seven_lines_29_regions_l2266_226635


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l2266_226677

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : Sorry

/-- The octagon formed by joining the midpoints of the sides of a regular octagon -/
def midpoint_octagon (oct : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (oct : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The area of the octagon formed by joining the midpoints of the sides
    of a regular octagon is 1/2 of the area of the original octagon -/
theorem midpoint_octagon_area_ratio (oct : RegularOctagon) :
  area (midpoint_octagon oct) = (1/2 : ℝ) * area oct :=
sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l2266_226677


namespace NUMINAMATH_CALUDE_wednesday_sales_l2266_226645

def initial_stock : ℕ := 600
def monday_sales : ℕ := 25
def tuesday_sales : ℕ := 70
def thursday_sales : ℕ := 110
def friday_sales : ℕ := 145
def unsold_percentage : ℚ := 1/4

theorem wednesday_sales :
  ∃ (wednesday_sales : ℕ),
    wednesday_sales = initial_stock * (1 - unsold_percentage) -
      (monday_sales + tuesday_sales + thursday_sales + friday_sales) ∧
    wednesday_sales = 100 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_sales_l2266_226645


namespace NUMINAMATH_CALUDE_nickel_chocolates_l2266_226603

theorem nickel_chocolates (robert : ℕ) (nickel : ℕ) 
  (h1 : robert = 7)
  (h2 : robert = nickel + 2) : 
  nickel = 5 := by
sorry

end NUMINAMATH_CALUDE_nickel_chocolates_l2266_226603


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_bases_l2266_226691

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  lateral_side : ℝ
  height : ℝ
  median : ℝ
  base_small : ℝ
  base_large : ℝ

/-- The theorem stating the bases of the isosceles trapezoid with given properties -/
theorem isosceles_trapezoid_bases
  (t : IsoscelesTrapezoid)
  (h1 : t.lateral_side = 41)
  (h2 : t.height = 40)
  (h3 : t.median = 45) :
  t.base_small = 36 ∧ t.base_large = 54 := by
  sorry

#check isosceles_trapezoid_bases

end NUMINAMATH_CALUDE_isosceles_trapezoid_bases_l2266_226691


namespace NUMINAMATH_CALUDE_oliver_candy_boxes_l2266_226665

theorem oliver_candy_boxes (initial_boxes : ℕ) (total_boxes : ℕ) (boxes_bought_later : ℕ) :
  initial_boxes = 8 →
  total_boxes = 14 →
  boxes_bought_later = total_boxes - initial_boxes →
  boxes_bought_later = 6 := by
  sorry

end NUMINAMATH_CALUDE_oliver_candy_boxes_l2266_226665


namespace NUMINAMATH_CALUDE_exam_score_problem_l2266_226615

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 120 →
  ∃ (correct_answers : ℕ) (wrong_answers : ℕ),
    correct_answers + wrong_answers = total_questions ∧
    correct_score * correct_answers + wrong_score * wrong_answers = total_score ∧
    correct_answers = 36 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l2266_226615


namespace NUMINAMATH_CALUDE_eighth_finger_number_l2266_226614

-- Define the function f
def f (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 4
  | 1 => 3
  | 2 => 6
  | 3 => 5
  | _ => 0  -- This case should never occur

-- Define a function to apply f n times
def apply_f_n_times (n : ℕ) : ℕ :=
  match n with
  | 0 => 4  -- Start with 4
  | n + 1 => f (apply_f_n_times n)

-- Theorem statement
theorem eighth_finger_number : apply_f_n_times 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_eighth_finger_number_l2266_226614


namespace NUMINAMATH_CALUDE_product_of_w_and_x_is_zero_l2266_226627

theorem product_of_w_and_x_is_zero 
  (w x y : ℝ) 
  (h1 : 2 / w + 2 / x = 2 / y) 
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) : 
  w * x = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_w_and_x_is_zero_l2266_226627


namespace NUMINAMATH_CALUDE_pi_is_infinite_decimal_l2266_226676

-- Define the property of being an infinite decimal
def IsInfiniteDecimal (x : ℝ) : Prop := sorry

-- Define the property of being an irrational number
def IsIrrational (x : ℝ) : Prop := sorry

-- State the theorem
theorem pi_is_infinite_decimal :
  (∀ x : ℝ, IsIrrational x → IsInfiniteDecimal x) →  -- Condition: Irrational numbers are infinite decimals
  IsIrrational Real.pi →                             -- Condition: π is an irrational number
  IsInfiniteDecimal Real.pi :=                       -- Conclusion: π is an infinite decimal
by sorry

end NUMINAMATH_CALUDE_pi_is_infinite_decimal_l2266_226676


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2266_226661

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 4) :
  x^2 + 1/x^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2266_226661


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l2266_226698

theorem difference_of_squares_special_case : (4 + Real.sqrt 6) * (4 - Real.sqrt 6) = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l2266_226698


namespace NUMINAMATH_CALUDE_total_oil_leak_l2266_226681

def initial_leak_A : ℕ := 6522
def initial_leak_B : ℕ := 3894
def initial_leak_C : ℕ := 1421

def leak_rate_A : ℕ := 257
def leak_rate_B : ℕ := 182
def leak_rate_C : ℕ := 97

def repair_time_A : ℕ := 20
def repair_time_B : ℕ := 15
def repair_time_C : ℕ := 12

theorem total_oil_leak :
  initial_leak_A + initial_leak_B + initial_leak_C +
  leak_rate_A * repair_time_A + leak_rate_B * repair_time_B + leak_rate_C * repair_time_C = 20871 := by
  sorry

end NUMINAMATH_CALUDE_total_oil_leak_l2266_226681


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l2266_226639

theorem no_solution_for_equation : 
  ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 1 / (2 * a + 3 * b)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l2266_226639


namespace NUMINAMATH_CALUDE_inverse_proportion_l2266_226668

/-- Given that α is inversely proportional to β, prove that if α = -3 when β = -6, 
    then α = 9/4 when β = 8. -/
theorem inverse_proportion (α β : ℝ → ℝ) (k : ℝ) 
    (h1 : ∀ x, α x * β x = k)  -- α is inversely proportional to β
    (h2 : α (-6) = -3)         -- α = -3 when β = -6
    (h3 : β (-6) = -6)         -- β = -6 when β = -6 (implicit in the problem)
    : α 8 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l2266_226668


namespace NUMINAMATH_CALUDE_pizza_slice_volume_l2266_226696

/-- The volume of a pizza slice -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_slices : ℕ) :
  thickness = 1/2 →
  diameter = 16 →
  num_slices = 8 →
  (π * (diameter/2)^2 * thickness) / num_slices = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_pizza_slice_volume_l2266_226696


namespace NUMINAMATH_CALUDE_modified_tic_tac_toe_tie_probability_l2266_226656

theorem modified_tic_tac_toe_tie_probability 
  (amy_win_prob : ℚ)
  (lily_win_prob : ℚ)
  (ben_win_prob : ℚ)
  (h1 : amy_win_prob = 4 / 15)
  (h2 : lily_win_prob = 1 / 5)
  (h3 : ben_win_prob = 1 / 6)
  (h4 : amy_win_prob + lily_win_prob + ben_win_prob < 1) : 
  1 - (amy_win_prob + lily_win_prob + ben_win_prob) = 11 / 30 := by
sorry

end NUMINAMATH_CALUDE_modified_tic_tac_toe_tie_probability_l2266_226656


namespace NUMINAMATH_CALUDE_odd_function_iff_m_and_n_l2266_226628

def f (m n x : ℝ) : ℝ := (m^2 - 1) * x^2 + (m - 1) * x + n + 2

theorem odd_function_iff_m_and_n (m n : ℝ) :
  (∀ x, f m n (-x) = -f m n x) ↔ ((m = 1 ∨ m = -1) ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_odd_function_iff_m_and_n_l2266_226628


namespace NUMINAMATH_CALUDE_equation_solution_l2266_226632

theorem equation_solution :
  ∃! x : ℝ, x ≠ 3 ∧
    ∃ z : ℝ, z = (x^2 - 9) / (x - 3) ∧ z = 3 * x ∧
    x = 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2266_226632


namespace NUMINAMATH_CALUDE_volleyball_match_probability_l2266_226630

/-- The probability of Team A winning a single set -/
def p_A : ℚ := 2/3

/-- The probability of Team B winning a single set -/
def p_B : ℚ := 1 - p_A

/-- The number of sets Team B has won at the start -/
def initial_B_wins : ℕ := 2

/-- The number of sets needed to win the match -/
def sets_to_win : ℕ := 3

/-- The probability of Team B winning the match given they lead 2:0 -/
def p_B_wins : ℚ := p_B + p_A * p_B + p_A * p_A * p_B

theorem volleyball_match_probability :
  p_B_wins = 19/27 := by sorry

end NUMINAMATH_CALUDE_volleyball_match_probability_l2266_226630


namespace NUMINAMATH_CALUDE_subset_implies_m_geq_one_l2266_226660

theorem subset_implies_m_geq_one (m : ℝ) : 
  ({x : ℝ | 0 < x ∧ x < 1} ⊆ {x : ℝ | 0 < x ∧ x < m}) → m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_geq_one_l2266_226660


namespace NUMINAMATH_CALUDE_pants_and_coat_cost_l2266_226629

theorem pants_and_coat_cost (p s c : ℝ) 
  (h1 : p + s = 100)
  (h2 : c = 5 * s)
  (h3 : c = 180) : 
  p + c = 244 := by
  sorry

end NUMINAMATH_CALUDE_pants_and_coat_cost_l2266_226629


namespace NUMINAMATH_CALUDE_mans_speed_with_stream_l2266_226667

/-- Given a man's rowing speed against the stream and his rate in still water,
    calculate his speed with the stream. -/
theorem mans_speed_with_stream
  (speed_against_stream : ℝ)
  (rate_still_water : ℝ)
  (h1 : speed_against_stream = 4)
  (h2 : rate_still_water = 8) :
  rate_still_water + (rate_still_water - speed_against_stream) = 12 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_with_stream_l2266_226667


namespace NUMINAMATH_CALUDE_minimum_cans_for_target_gallons_l2266_226683

/-- The number of ounces in one gallon -/
def ounces_per_gallon : ℕ := 128

/-- The number of ounces each can holds -/
def ounces_per_can : ℕ := 16

/-- The number of gallons we want to have at least -/
def target_gallons : ℚ := 3/2

theorem minimum_cans_for_target_gallons :
  let total_ounces := (target_gallons * ounces_per_gallon).ceil
  let num_cans := (total_ounces + ounces_per_can - 1) / ounces_per_can
  num_cans = 12 := by sorry

end NUMINAMATH_CALUDE_minimum_cans_for_target_gallons_l2266_226683


namespace NUMINAMATH_CALUDE_soccer_ball_hexagons_l2266_226695

/-- Represents a soccer ball with black pentagons and white hexagons -/
structure SoccerBall where
  black_pentagons : ℕ
  white_hexagons : ℕ
  pentagon_sides : ℕ
  hexagon_sides : ℕ
  pentagon_hexagon_connections : ℕ
  hexagon_pentagon_connections : ℕ
  hexagon_hexagon_connections : ℕ

/-- Theorem stating the number of white hexagons on a soccer ball with specific conditions -/
theorem soccer_ball_hexagons (ball : SoccerBall) :
  ball.black_pentagons = 12 ∧
  ball.pentagon_sides = 5 ∧
  ball.hexagon_sides = 6 ∧
  ball.pentagon_hexagon_connections = 5 ∧
  ball.hexagon_pentagon_connections = 3 ∧
  ball.hexagon_hexagon_connections = 3 →
  ball.white_hexagons = 20 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_hexagons_l2266_226695


namespace NUMINAMATH_CALUDE_at_least_seven_stay_probability_l2266_226619

def total_friends : ℕ := 8
def sure_friends : ℕ := 3
def unsure_friends : ℕ := 5
def stay_probability : ℚ := 1/3

def probability_at_least_seven_stay : ℚ :=
  (Nat.choose unsure_friends 4 * stay_probability^4 * (1 - stay_probability)^1) +
  (stay_probability^5)

theorem at_least_seven_stay_probability :
  probability_at_least_seven_stay = 11/243 :=
sorry

end NUMINAMATH_CALUDE_at_least_seven_stay_probability_l2266_226619


namespace NUMINAMATH_CALUDE_jack_stairs_problem_l2266_226647

/-- Represents the number of steps in each flight of stairs -/
def steps_per_flight : ℕ := 12

/-- Represents the height of each step in inches -/
def step_height : ℕ := 8

/-- Represents the number of flights Jack went down -/
def flights_down : ℕ := 6

/-- Represents how much further down Jack ended up, in feet -/
def final_position : ℕ := 24

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the height of one flight of stairs in inches -/
def flight_height : ℕ := steps_per_flight * step_height

/-- Represents the number of flights Jack went up initially -/
def flights_up : ℕ := 9

theorem jack_stairs_problem :
  flights_up * flight_height = 
  flights_down * flight_height + feet_to_inches final_position :=
by sorry

end NUMINAMATH_CALUDE_jack_stairs_problem_l2266_226647


namespace NUMINAMATH_CALUDE_spade_equation_solution_l2266_226680

-- Define the ♠ operation
def spade (A B : ℝ) : ℝ := 4 * A + 3 * B + 6

-- Theorem statement
theorem spade_equation_solution :
  ∃! A : ℝ, spade A 5 = 79 ∧ A = 14.5 := by sorry

end NUMINAMATH_CALUDE_spade_equation_solution_l2266_226680


namespace NUMINAMATH_CALUDE_complex_power_modulus_l2266_226693

theorem complex_power_modulus : Complex.abs ((1/3 : ℂ) + (2/3 : ℂ) * Complex.I) ^ 8 = 625/6561 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l2266_226693


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l2266_226657

theorem sum_of_squares_problem (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0)
  (sum_of_squares : a^2 + b^2 + c^2 = 52)
  (sum_of_products : a*b + b*c + c*a = 28) :
  a + b + c = 6 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l2266_226657


namespace NUMINAMATH_CALUDE_leibniz_theorem_l2266_226670

/-- Leibniz's Theorem -/
theorem leibniz_theorem (A B C M : ℝ × ℝ) : 
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  3 * dist M G ^ 2 = 
    dist M A ^ 2 + dist M B ^ 2 + dist M C ^ 2 - 
    (1/3) * (dist A B ^ 2 + dist B C ^ 2 + dist C A ^ 2) := by
  sorry

#check leibniz_theorem

end NUMINAMATH_CALUDE_leibniz_theorem_l2266_226670


namespace NUMINAMATH_CALUDE_least_product_of_two_primes_above_ten_l2266_226611

theorem least_product_of_two_primes_above_ten (p q : ℕ) : 
  Prime p → Prime q → p > 10 → q > 10 → p ≠ q → 
  ∀ r s : ℕ, Prime r → Prime s → r > 10 → s > 10 → r ≠ s → 
  p * q ≤ r * s → p * q = 143 := by sorry

end NUMINAMATH_CALUDE_least_product_of_two_primes_above_ten_l2266_226611


namespace NUMINAMATH_CALUDE_expression_equals_one_l2266_226687

theorem expression_equals_one : 
  (120^2 - 13^2) / (90^2 - 19^2) * ((90 - 19) * (90 + 19)) / ((120 - 13) * (120 + 13)) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2266_226687


namespace NUMINAMATH_CALUDE_author_paperback_percentage_is_six_percent_l2266_226659

/-- Represents the book sales problem --/
structure BookSales where
  paperback_copies : ℕ
  paperback_price : ℚ
  hardcover_copies : ℕ
  hardcover_price : ℚ
  hardcover_percentage : ℚ
  total_earnings : ℚ

/-- Calculates the author's percentage from paperback sales --/
def paperback_percentage (sales : BookSales) : ℚ :=
  let paperback_sales := sales.paperback_copies * sales.paperback_price
  let hardcover_sales := sales.hardcover_copies * sales.hardcover_price
  let hardcover_earnings := sales.hardcover_percentage * hardcover_sales
  let paperback_earnings := sales.total_earnings - hardcover_earnings
  paperback_earnings / paperback_sales

/-- Theorem stating that the author's percentage from paperback sales is 6% --/
theorem author_paperback_percentage_is_six_percent (sales : BookSales) 
  (h1 : sales.paperback_copies = 32000)
  (h2 : sales.paperback_price = 1/5)
  (h3 : sales.hardcover_copies = 15000)
  (h4 : sales.hardcover_price = 2/5)
  (h5 : sales.hardcover_percentage = 12/100)
  (h6 : sales.total_earnings = 1104) :
  paperback_percentage sales = 6/100 := by
  sorry


end NUMINAMATH_CALUDE_author_paperback_percentage_is_six_percent_l2266_226659


namespace NUMINAMATH_CALUDE_octal_calculation_l2266_226674

/-- Represents a number in base 8 --/
def OctalNumber := Nat

/-- Addition of two octal numbers --/
def octal_add (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtraction of two octal numbers --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from decimal to octal --/
def to_octal (n : Nat) : OctalNumber :=
  sorry

/-- Theorem: 24₈ + 53₈ - 17₈ = 60₈ in base 8 --/
theorem octal_calculation : 
  octal_sub (octal_add (to_octal 24) (to_octal 53)) (to_octal 17) = to_octal 60 :=
by sorry

end NUMINAMATH_CALUDE_octal_calculation_l2266_226674


namespace NUMINAMATH_CALUDE_sum_increase_by_three_percent_l2266_226638

theorem sum_increase_by_three_percent : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
  (1.01 * x + 1.04 * y) = 1.03 * (x + y) := by
sorry

end NUMINAMATH_CALUDE_sum_increase_by_three_percent_l2266_226638


namespace NUMINAMATH_CALUDE_line_passes_through_quadrants_l2266_226634

/-- A line ax + by = c passes through the first, third, and fourth quadrants
    given that ab < 0 and bc < 0 -/
theorem line_passes_through_quadrants
  (a b c : ℝ) 
  (hab : a * b < 0) 
  (hbc : b * c < 0) : 
  ∃ (x y : ℝ), 
    (a * x + b * y = c) ∧ 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0)) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_quadrants_l2266_226634


namespace NUMINAMATH_CALUDE_volunteer_team_statistics_l2266_226617

def frequencies : List ℕ := [10, 10, 10, 8, 8, 8, 8, 7, 7, 4]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

theorem volunteer_team_statistics :
  mode frequencies = 8 ∧
  median frequencies = 8 ∧
  mean frequencies = 8 := by sorry

end NUMINAMATH_CALUDE_volunteer_team_statistics_l2266_226617


namespace NUMINAMATH_CALUDE_triangle_area_l2266_226624

/-- Given a triangle ABC with sides a, b, and c satisfying certain conditions, 
    prove that its area is 6. -/
theorem triangle_area (a b c : ℝ) : 
  (a + 4) / 3 = (b + 3) / 2 ∧ 
  (b + 3) / 2 = (c + 8) / 4 ∧ 
  a + b + c = 12 → 
  (1 / 2 : ℝ) * b * c = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2266_226624


namespace NUMINAMATH_CALUDE_starting_number_property_l2266_226631

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def subtractSumOfDigits (n : ℕ) : ℕ :=
  n - sumOfDigits n

def iterateSubtraction (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => iterateSubtraction (subtractSumOfDigits n) k

theorem starting_number_property (n : ℕ) (h : 100 ≤ n ∧ n ≤ 109) :
  iterateSubtraction n 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_starting_number_property_l2266_226631


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_one_l2266_226620

theorem product_of_fractions_equals_one :
  (7 / 3) * (10 / 6) * (35 / 21) * (20 / 12) * (49 / 21) * (18 / 30) * (45 / 27) * (24 / 40) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_one_l2266_226620


namespace NUMINAMATH_CALUDE_probability_two_heads_two_tails_prove_probability_two_heads_two_tails_l2266_226610

/-- The probability of getting exactly two heads and two tails when tossing four fair coins -/
theorem probability_two_heads_two_tails : ℚ :=
  3/8

/-- Proof that the probability of getting exactly two heads and two tails
    when tossing four fair coins is 3/8 -/
theorem prove_probability_two_heads_two_tails :
  probability_two_heads_two_tails = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_heads_two_tails_prove_probability_two_heads_two_tails_l2266_226610


namespace NUMINAMATH_CALUDE_seats_filled_percentage_l2266_226694

/-- Given a hall with total seats and vacant seats, calculate the percentage of filled seats -/
def percentage_filled (total_seats vacant_seats : ℕ) : ℚ :=
  (total_seats - vacant_seats : ℚ) / total_seats * 100

/-- Theorem: In a hall with 600 seats where 300 are vacant, 50% of the seats are filled -/
theorem seats_filled_percentage :
  percentage_filled 600 300 = 50 := by
  sorry

#eval percentage_filled 600 300

end NUMINAMATH_CALUDE_seats_filled_percentage_l2266_226694


namespace NUMINAMATH_CALUDE_weight_of_ton_l2266_226612

/-- The weight of a ton in pounds -/
def ton_weight : ℝ := 2000

theorem weight_of_ton (elephant_weight : ℝ) (donkey_weight : ℝ) 
  (h1 : elephant_weight = 3 * ton_weight)
  (h2 : donkey_weight = 0.1 * elephant_weight)
  (h3 : elephant_weight + donkey_weight = 6600) :
  ton_weight = 2000 := by
  sorry

#check weight_of_ton

end NUMINAMATH_CALUDE_weight_of_ton_l2266_226612


namespace NUMINAMATH_CALUDE_first_sat_score_l2266_226607

/-- 
Given a 10% improvement from the first score to the second score, 
and a second score of 1100, prove that the first score must be 1000.
-/
theorem first_sat_score (second_score : ℝ) (improvement : ℝ) 
  (h1 : second_score = 1100)
  (h2 : improvement = 0.1)
  (h3 : second_score = (1 + improvement) * first_score) :
  first_score = 1000 := by
  sorry

end NUMINAMATH_CALUDE_first_sat_score_l2266_226607


namespace NUMINAMATH_CALUDE_crease_length_eq_sqrt_six_over_four_l2266_226622

/-- An isosceles right triangle with hypotenuse 1 -/
structure IsoscelesRightTriangle where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is 1 -/
  hypotenuse_eq_one : hypotenuse = 1

/-- The crease formed by folding one vertex to the other on the hypotenuse -/
def crease_length (t : IsoscelesRightTriangle) : ℝ :=
  sorry  -- Definition of crease length calculation

theorem crease_length_eq_sqrt_six_over_four (t : IsoscelesRightTriangle) :
  crease_length t = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_crease_length_eq_sqrt_six_over_four_l2266_226622


namespace NUMINAMATH_CALUDE_sin_alpha_value_l2266_226601

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 6) = 1 / 5) : 
  Real.sin α = (6 * Real.sqrt 2 - 1) / 10 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l2266_226601


namespace NUMINAMATH_CALUDE_lowest_divisible_by_even_14_to_21_l2266_226685

theorem lowest_divisible_by_even_14_to_21 : ∃! n : ℕ+, 
  (∀ k : ℕ, 14 ≤ k ∧ k ≤ 21 ∧ Even k → (n : ℕ) % k = 0) ∧ 
  (∀ m : ℕ+, (∀ k : ℕ, 14 ≤ k ∧ k ≤ 21 ∧ Even k → (m : ℕ) % k = 0) → n ≤ m) ∧
  n = 5040 := by
sorry

end NUMINAMATH_CALUDE_lowest_divisible_by_even_14_to_21_l2266_226685


namespace NUMINAMATH_CALUDE_intersection_points_sum_l2266_226625

-- Define the functions
def f (x : ℝ) : ℝ := (x - 2) * (x - 4)
def g (x : ℝ) : ℝ := -f x
def h (x : ℝ) : ℝ := f (-x)

-- Define c as the number of intersection points between f and g
def c : ℕ := 2

-- Define d as the number of intersection points between f and h
def d : ℕ := 1

-- Theorem to prove
theorem intersection_points_sum : 10 * c + d = 21 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_sum_l2266_226625


namespace NUMINAMATH_CALUDE_soda_cost_l2266_226608

/-- The cost of items in cents -/
structure ItemCosts where
  burger : ℕ
  soda : ℕ
  fries : ℕ

/-- Represents the purchase combinations -/
inductive Purchase
  | uri1 : Purchase
  | gen1 : Purchase
  | uri2 : Purchase
  | gen2 : Purchase

/-- The cost of each purchase in cents -/
def purchaseCost (p : Purchase) (costs : ItemCosts) : ℕ :=
  match p with
  | .uri1 => 3 * costs.burger + costs.soda
  | .gen1 => 2 * costs.burger + 3 * costs.soda
  | .uri2 => costs.burger + 2 * costs.fries
  | .gen2 => costs.soda + 3 * costs.fries

theorem soda_cost (costs : ItemCosts) 
  (h1 : purchaseCost .uri1 costs = 390)
  (h2 : purchaseCost .gen1 costs = 440)
  (h3 : purchaseCost .uri2 costs = 230)
  (h4 : purchaseCost .gen2 costs = 270) :
  costs.soda = 234 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l2266_226608


namespace NUMINAMATH_CALUDE_winnie_lollipops_left_l2266_226606

/-- The number of lollipops left after equal distribution -/
def lollipops_left (total : ℕ) (friends : ℕ) : ℕ :=
  total % friends

theorem winnie_lollipops_left :
  let cherry := 60
  let wintergreen := 145
  let grape := 10
  let shrimp_cocktail := 295
  let total := cherry + wintergreen + grape + shrimp_cocktail
  let friends := 13
  lollipops_left total friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_winnie_lollipops_left_l2266_226606


namespace NUMINAMATH_CALUDE_magic_square_solution_l2266_226648

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℤ)

/-- The sum of any row, column, or diagonal in a magic square is constant -/
def MagicSquare.isMagic (s : MagicSquare) : Prop :=
  let sum := s.a11 + s.a12 + s.a13
  sum = s.a21 + s.a22 + s.a23 ∧
  sum = s.a31 + s.a32 + s.a33 ∧
  sum = s.a11 + s.a21 + s.a31 ∧
  sum = s.a12 + s.a22 + s.a32 ∧
  sum = s.a13 + s.a23 + s.a33 ∧
  sum = s.a11 + s.a22 + s.a33 ∧
  sum = s.a13 + s.a22 + s.a31

theorem magic_square_solution :
  ∀ (s : MagicSquare),
    s.isMagic →
    s.a11 = s.a11 ∧
    s.a12 = 23 ∧
    s.a13 = 84 ∧
    s.a21 = 3 →
    s.a11 = 175 := by
  sorry

#check magic_square_solution

end NUMINAMATH_CALUDE_magic_square_solution_l2266_226648


namespace NUMINAMATH_CALUDE_larger_root_of_quadratic_l2266_226605

theorem larger_root_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 40 = 0 → x ≤ 8 :=
by
  sorry

end NUMINAMATH_CALUDE_larger_root_of_quadratic_l2266_226605


namespace NUMINAMATH_CALUDE_sports_club_overlap_l2266_226604

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) 
  (h1 : total = 42)
  (h2 : badminton = 20)
  (h3 : tennis = 23)
  (h4 : neither = 6) :
  badminton + tennis - (total - neither) = 7 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l2266_226604


namespace NUMINAMATH_CALUDE_slower_train_speed_l2266_226684

/-- Proves that the speed of the slower train is 37 km/hr given the conditions of the problem -/
theorem slower_train_speed 
  (train_length : ℝ) 
  (faster_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 62.5)
  (h2 : faster_speed = 46)
  (h3 : passing_time = 45)
  : ∃ (slower_speed : ℝ), 
    slower_speed = 37 ∧ 
    2 * train_length = (faster_speed - slower_speed) * (5 / 18) * passing_time :=
by
  sorry

#check slower_train_speed

end NUMINAMATH_CALUDE_slower_train_speed_l2266_226684


namespace NUMINAMATH_CALUDE_decreasing_implies_a_leq_neg_three_l2266_226642

/-- A quadratic function f(x) that is decreasing on (-∞, 4] -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The property that f is decreasing on (-∞, 4] -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 4 → f a x > f a y

/-- Theorem stating that if f is decreasing on (-∞, 4], then a ≤ -3 -/
theorem decreasing_implies_a_leq_neg_three (a : ℝ) :
  is_decreasing_on_interval a → a ≤ -3 := by sorry

end NUMINAMATH_CALUDE_decreasing_implies_a_leq_neg_three_l2266_226642


namespace NUMINAMATH_CALUDE_helen_gas_consumption_l2266_226623

/-- Represents the gas consumption for Helen's lawn maintenance --/
def lawn_maintenance_gas_consumption : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ := sorry

/-- The number of times the large lawn is cut --/
def large_lawn_cuts : ℕ := 18

/-- The number of times the small lawn is cut --/
def small_lawn_cuts : ℕ := 14

/-- The number of times the suburban lawn is trimmed --/
def suburban_trims : ℕ := 6

/-- The number of times the leaf blower is used --/
def leaf_blower_uses : ℕ := 2

theorem helen_gas_consumption :
  lawn_maintenance_gas_consumption large_lawn_cuts small_lawn_cuts suburban_trims leaf_blower_uses 3 2 = 22 := by sorry

end NUMINAMATH_CALUDE_helen_gas_consumption_l2266_226623


namespace NUMINAMATH_CALUDE_eugene_model_house_l2266_226663

/-- The number of toothpicks Eugene uses per card -/
def toothpicks_per_card : ℕ := 75

/-- The total number of cards in the deck -/
def total_cards : ℕ := 52

/-- The number of cards Eugene did not use -/
def unused_cards : ℕ := 16

/-- The number of toothpicks in one box -/
def toothpicks_per_box : ℕ := 450

/-- The number of boxes of toothpicks Eugene used -/
def boxes_used : ℕ := 6

theorem eugene_model_house :
  boxes_used = (total_cards - unused_cards) * toothpicks_per_card / toothpicks_per_box :=
by sorry

end NUMINAMATH_CALUDE_eugene_model_house_l2266_226663


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l2266_226675

theorem geometric_progression_ratio (x y z r : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h_geometric : ∃ (a : ℝ), a ≠ 0 ∧ 
    x * (y - z) = a ∧ 
    y * (z - x) = a * r ∧ 
    z * (x - y) = a * r^2) :
  r^2 + r + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l2266_226675


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_91_l2266_226600

theorem no_primes_divisible_by_91 :
  ¬∃ p : ℕ, Nat.Prime p ∧ 91 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_91_l2266_226600


namespace NUMINAMATH_CALUDE_four_digit_sum_product_divisible_by_11_l2266_226609

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Converts four digits to a four-digit number -/
def toNumber (w x y z : Digit) : ℕ :=
  1000 * w.val + 100 * x.val + 10 * y.val + z.val

theorem four_digit_sum_product_divisible_by_11 
  (w x y z : Digit) 
  (hw : w ≠ x) (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ w) 
  (hwx : w ≠ y) (hwy : w ≠ z) (hxy : x ≠ z) : 
  11 ∣ (toNumber w x y z + toNumber z y x w + toNumber w x y z * toNumber z y x w) :=
sorry

end NUMINAMATH_CALUDE_four_digit_sum_product_divisible_by_11_l2266_226609


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_l2266_226671

theorem regular_polygon_diagonals (n : ℕ) : n > 2 → (n * (n - 3)) / 2 = 90 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_l2266_226671


namespace NUMINAMATH_CALUDE_bird_count_proof_l2266_226641

/-- The number of storks on the fence -/
def num_storks : ℕ := 6

/-- The number of additional birds that joined -/
def additional_birds : ℕ := 3

/-- The number of birds initially on the fence -/
def initial_birds : ℕ := 2

theorem bird_count_proof :
  initial_birds = 2 ∧
  num_storks = (initial_birds + additional_birds) + 1 :=
by sorry

end NUMINAMATH_CALUDE_bird_count_proof_l2266_226641


namespace NUMINAMATH_CALUDE_subset_implies_m_geq_two_l2266_226602

def set_A (m : ℝ) : Set ℝ := {x | x ≤ m}
def set_B : Set ℝ := {x | -2 < x ∧ x ≤ 2}

theorem subset_implies_m_geq_two (m : ℝ) :
  set_B ⊆ set_A m → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_geq_two_l2266_226602


namespace NUMINAMATH_CALUDE_expression_evaluation_l2266_226669

theorem expression_evaluation (x y : ℝ) (hx : x = 3) (hy : y = 5) : 
  (3 * x^4 + 2 * y^2 + 10) / 8 = 303 / 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2266_226669


namespace NUMINAMATH_CALUDE_sock_matching_probability_l2266_226633

def total_socks : ℕ := 8
def black_socks : ℕ := 6
def white_socks : ℕ := 2

def total_combinations : ℕ := total_socks.choose 2
def matching_combinations : ℕ := black_socks.choose 2 + 1

theorem sock_matching_probability :
  (matching_combinations : ℚ) / total_combinations = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_sock_matching_probability_l2266_226633


namespace NUMINAMATH_CALUDE_twentyFifth_is_221_l2266_226652

/-- Converts a natural number to its base-3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The 25th number in base-3 counting sequence -/
def twentyFifthBase3 : List ℕ := toBase3 25

theorem twentyFifth_is_221 : twentyFifthBase3 = [2, 2, 1] := by
  sorry

#eval twentyFifthBase3

end NUMINAMATH_CALUDE_twentyFifth_is_221_l2266_226652


namespace NUMINAMATH_CALUDE_min_sum_squares_l2266_226613

theorem min_sum_squares (a b : ℝ) : 
  let A : ℝ × ℝ := (a, 2)
  let B : ℝ × ℝ := (3, b)
  let C : ℝ × ℝ := (2, 3)
  let O : ℝ × ℝ := (0, 0)
  let OB : ℝ × ℝ := (3 - 0, b - 0)
  let AC : ℝ × ℝ := (2 - a, 3 - 2)
  (OB.1 * AC.1 + OB.2 * AC.2 = 0) →
  (∃ (x : ℝ), ∀ (a b : ℝ), a^2 + b^2 ≥ x ∧ (∃ (a₀ b₀ : ℝ), a₀^2 + b₀^2 = x)) ∧
  (∀ (x : ℝ), (∃ (a b : ℝ), a^2 + b^2 = x) → x ≥ 18/5) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2266_226613


namespace NUMINAMATH_CALUDE_unique_special_number_l2266_226655

/-- A four-digit number with specific properties -/
def special_number : ℕ → Prop := λ n =>
  -- The number is four-digit
  1000 ≤ n ∧ n < 10000 ∧
  -- The unit digit is 2
  n % 10 = 2 ∧
  -- Moving the last digit to the front results in a number 108 less than the original
  (2000 + n / 10) = n - 108

theorem unique_special_number :
  ∃! n : ℕ, special_number n ∧ n = 2342 :=
sorry

end NUMINAMATH_CALUDE_unique_special_number_l2266_226655


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2266_226618

theorem fixed_point_of_exponential_function :
  let f : ℝ → ℝ := λ x => 2^(x + 2) + 1
  f (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2266_226618


namespace NUMINAMATH_CALUDE_chess_game_probability_l2266_226637

theorem chess_game_probability (draw_prob : ℚ) (b_win_prob : ℚ) (a_win_prob : ℚ) : 
  draw_prob = 1/2 → b_win_prob = 1/3 → a_win_prob = 1 - draw_prob - b_win_prob → a_win_prob = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l2266_226637


namespace NUMINAMATH_CALUDE_mathematics_teacher_is_C_l2266_226664

-- Define the types for teachers and subjects
inductive Teacher : Type
  | A | B | C | D

inductive Subject : Type
  | Mathematics | Physics | Chemistry | English

-- Define a function to represent the ability to teach a subject
def canTeach : Teacher → Subject → Prop
  | Teacher.A, Subject.Physics => True
  | Teacher.A, Subject.Chemistry => True
  | Teacher.B, Subject.Mathematics => True
  | Teacher.B, Subject.English => True
  | Teacher.C, Subject.Mathematics => True
  | Teacher.C, Subject.Physics => True
  | Teacher.C, Subject.Chemistry => True
  | Teacher.D, Subject.Chemistry => True
  | _, _ => False

-- Define the assignment of teachers to subjects
def assignment : Subject → Teacher
  | Subject.Mathematics => Teacher.C
  | Subject.Physics => Teacher.A
  | Subject.Chemistry => Teacher.D
  | Subject.English => Teacher.B

-- Theorem statement
theorem mathematics_teacher_is_C :
  (∀ s : Subject, canTeach (assignment s) s) ∧
  (∀ t : Teacher, ∃! s : Subject, assignment s = t) ∧
  (∀ s : Subject, ∃! t : Teacher, assignment s = t) →
  assignment Subject.Mathematics = Teacher.C :=
by sorry

end NUMINAMATH_CALUDE_mathematics_teacher_is_C_l2266_226664


namespace NUMINAMATH_CALUDE_expression_evaluation_l2266_226650

theorem expression_evaluation (a b : ℝ) (h : a^2 + b^2 - 2*a + 4*b = -5) :
  (a - 2*b)*(a^2 + 2*a*b + 4*b^2) - a*(a - 5*b)*(a + 3*b) = 120 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2266_226650


namespace NUMINAMATH_CALUDE_factorial_difference_l2266_226699

theorem factorial_difference : Nat.factorial 9 - Nat.factorial 8 = 322560 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l2266_226699


namespace NUMINAMATH_CALUDE_alice_average_speed_l2266_226686

/-- Alice's cycling journey --/
def alice_journey : Prop :=
  let first_distance : ℝ := 240
  let first_time : ℝ := 4.5
  let second_distance : ℝ := 300
  let second_time : ℝ := 5.25
  let total_distance : ℝ := first_distance + second_distance
  let total_time : ℝ := first_time + second_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 540 / 9.75

theorem alice_average_speed : alice_journey := by
  sorry

end NUMINAMATH_CALUDE_alice_average_speed_l2266_226686


namespace NUMINAMATH_CALUDE_platform_length_proof_l2266_226640

/-- Proves that given a train with specified speed and length, crossing a platform in a certain time, the platform length is approximately 165 meters. -/
theorem platform_length_proof (train_speed : Real) (train_length : Real) (crossing_time : Real) :
  train_speed = 132 * 1000 / 3600 →
  train_length = 110 →
  crossing_time = 7.499400047996161 →
  ∃ (platform_length : Real),
    (platform_length + train_length) = train_speed * crossing_time ∧
    abs (platform_length - 165) < 1 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_proof_l2266_226640


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2266_226658

theorem hyperbola_eccentricity (a : ℝ) : 
  a > 0 →
  (∃ x y : ℝ, x^2/a^2 - y^2/3^2 = 1) →
  (∃ e : ℝ, e = 2 ∧ e^2 = (a^2 + 3^2)/a^2) →
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2266_226658


namespace NUMINAMATH_CALUDE_jerry_george_sticker_ratio_l2266_226692

/-- The ratio of Jerry's stickers to George's stickers -/
def stickerRatio (jerryStickers georgeStickers : ℕ) : ℚ :=
  jerryStickers / georgeStickers

/-- Proof that the ratio of Jerry's stickers to George's stickers is 3 -/
theorem jerry_george_sticker_ratio :
  let fredStickers : ℕ := 18
  let georgeStickers : ℕ := fredStickers - 6
  let jerryStickers : ℕ := 36
  stickerRatio jerryStickers georgeStickers = 3 := by
sorry

end NUMINAMATH_CALUDE_jerry_george_sticker_ratio_l2266_226692


namespace NUMINAMATH_CALUDE_x_forty_percent_greater_than_88_l2266_226666

theorem x_forty_percent_greater_than_88 :
  ∀ x : ℝ, x = 88 * (1 + 0.4) → x = 123.2 :=
by
  sorry

end NUMINAMATH_CALUDE_x_forty_percent_greater_than_88_l2266_226666


namespace NUMINAMATH_CALUDE_dime_probability_l2266_226649

def coin_jar (quarter_value dime_value penny_value : ℚ)
             (total_quarter_value total_dime_value total_penny_value : ℚ) : Prop :=
  let quarter_count := total_quarter_value / quarter_value
  let dime_count := total_dime_value / dime_value
  let penny_count := total_penny_value / penny_value
  let total_coins := quarter_count + dime_count + penny_count
  dime_count / total_coins = 1 / 7

theorem dime_probability :
  coin_jar (25/100) (10/100) (1/100) (1250/100) (500/100) (250/100) := by
  sorry

end NUMINAMATH_CALUDE_dime_probability_l2266_226649


namespace NUMINAMATH_CALUDE_max_value_of_a_is_zero_l2266_226644

theorem max_value_of_a_is_zero (a : ℝ) 
  (h : ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → x * Real.log x - (1 + a) * x + 1 ≥ 0) : 
  a ≤ 0 ∧ ∀ ε > 0, ∃ x ∈ Set.Icc (1/2) 2, x * Real.log x - (1 + (a + ε)) * x + 1 < 0 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_a_is_zero_l2266_226644


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2266_226643

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  stratum_size : ℕ
  stratum_sample : ℕ
  total_sample : ℕ

/-- 
Theorem: In a stratified sampling scenario with a total population of 750,
where one stratum has 250 members and 5 are sampled from this stratum,
the total sample size is 15.
-/
theorem stratified_sample_size 
  (s : StratifiedSample) 
  (h1 : s.total_population = 750) 
  (h2 : s.stratum_size = 250) 
  (h3 : s.stratum_sample = 5) 
  (h4 : s.stratum_sample / s.stratum_size = s.total_sample / s.total_population) : 
  s.total_sample = 15 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2266_226643


namespace NUMINAMATH_CALUDE_mary_overtime_rate_increase_l2266_226689

/-- Represents Mary's work schedule and pay structure -/
structure MaryWorkSchedule where
  maxHours : ℕ
  regularHours : ℕ
  regularRate : ℚ
  maxEarnings : ℚ

/-- Calculates the percentage increase in overtime rate compared to regular rate -/
def overtimeRateIncrease (schedule : MaryWorkSchedule) : ℚ :=
  let regularEarnings := schedule.regularHours * schedule.regularRate
  let overtimeEarnings := schedule.maxEarnings - regularEarnings
  let overtimeHours := schedule.maxHours - schedule.regularHours
  let overtimeRate := overtimeEarnings / overtimeHours
  ((overtimeRate - schedule.regularRate) / schedule.regularRate) * 100

/-- Theorem stating that Mary's overtime rate increase is 25% -/
theorem mary_overtime_rate_increase :
  let schedule := MaryWorkSchedule.mk 70 20 8 660
  overtimeRateIncrease schedule = 25 := by
  sorry

end NUMINAMATH_CALUDE_mary_overtime_rate_increase_l2266_226689


namespace NUMINAMATH_CALUDE_divisibility_condition_l2266_226626

theorem divisibility_condition (n : ℕ) : n ≥ 1 → (n^2 ∣ 2^n + 1) ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2266_226626


namespace NUMINAMATH_CALUDE_problem_statement_l2266_226651

theorem problem_statement (x y : ℝ) (h : x - 2*y = -5) : 2 - x + 2*y = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2266_226651


namespace NUMINAMATH_CALUDE_ellipse_equation_l2266_226636

/-- An ellipse with major axis three times the minor axis and focal distance 8 -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Half focal distance
  h_major_minor : a = 3 * b
  h_focal : c = 4
  h_positive : a > 0 ∧ b > 0
  h_ellipse : a^2 = b^2 + c^2

/-- The standard equation of the ellipse is either x²/18 + y²/2 = 1 or y²/18 + x²/2 = 1 -/
theorem ellipse_equation (e : Ellipse) :
  (∀ x y : ℝ, x^2 / 18 + y^2 / 2 = 1) ∨ (∀ x y : ℝ, y^2 / 18 + x^2 / 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2266_226636


namespace NUMINAMATH_CALUDE_parabola_point_and_focus_l2266_226673

theorem parabola_point_and_focus (m : ℝ) (p : ℝ) : 
  p > 0 →
  ((-3)^2 = 2 * p * m) →
  (m + p / 2)^2 + (3 - p / 2)^2 = 5^2 →
  ((m = 1/2 ∧ p = 9) ∨ (m = 9/2 ∧ p = 1)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_and_focus_l2266_226673


namespace NUMINAMATH_CALUDE_correct_definition_in_list_correct_definition_unique_l2266_226616

/-- Definition of Digital Earth -/
def DigitalEarth : Type := String

/-- The correct definition of Digital Earth -/
def correct_definition : DigitalEarth :=
  "a technical system that digitizes the entire Earth's information and manages it through computer networks"

/-- Possible definitions of Digital Earth -/
def possible_definitions : List DigitalEarth :=
  [ "representing the size of the Earth with numbers"
  , correct_definition
  , "using the data of the latitude and longitude grid to represent the location of geographical entities"
  , "using GPS data to represent the location of various geographical entities on Earth"
  ]

/-- Theorem stating that the correct definition is in the list of possible definitions -/
theorem correct_definition_in_list : correct_definition ∈ possible_definitions :=
  by sorry

/-- Theorem stating that the correct definition is unique in the list -/
theorem correct_definition_unique :
  ∀ d ∈ possible_definitions, d = correct_definition ↔ d = possible_definitions[1] :=
  by sorry

end NUMINAMATH_CALUDE_correct_definition_in_list_correct_definition_unique_l2266_226616


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2266_226697

theorem decimal_to_fraction : (2.375 : ℚ) = 19 / 8 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2266_226697


namespace NUMINAMATH_CALUDE_john_bus_meet_once_l2266_226646

/-- Represents the movement of John and the bus on a straight path --/
structure Movement where
  johnSpeed : ℝ
  busSpeed : ℝ
  benchDistance : ℝ
  busStopTime : ℝ

/-- Calculates the number of times John and the bus meet --/
def meetingCount (m : Movement) : ℕ :=
  sorry

/-- Theorem stating that John and the bus meet exactly once --/
theorem john_bus_meet_once (m : Movement) 
  (h1 : m.johnSpeed = 6)
  (h2 : m.busSpeed = 15)
  (h3 : m.benchDistance = 300)
  (h4 : m.busStopTime = 45) :
  meetingCount m = 1 := by
  sorry

end NUMINAMATH_CALUDE_john_bus_meet_once_l2266_226646


namespace NUMINAMATH_CALUDE_gcd_38_23_is_1_l2266_226621

/-- The method of continued subtraction for calculating GCD -/
def continuedSubtractionGCD (a b : ℕ) : ℕ :=
  if a = 0 then b
  else if b = 0 then a
  else if a ≥ b then continuedSubtractionGCD (a - b) b
  else continuedSubtractionGCD a (b - a)

/-- Theorem: The GCD of 38 and 23 is 1 using the method of continued subtraction -/
theorem gcd_38_23_is_1 : continuedSubtractionGCD 38 23 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_38_23_is_1_l2266_226621


namespace NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_l2266_226678

def repeating_decimal_to_fraction (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 990 + a / 10

theorem repeating_decimal_equiv_fraction :
  repeating_decimal_to_fraction 2 1 3 = 523 / 2475 ∧
  (∀ m n : ℕ, m ≠ 0 → n ≠ 0 → m / n = 523 / 2475 → m ≥ 523 ∧ n ≥ 2475) :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_l2266_226678


namespace NUMINAMATH_CALUDE_factorial_ratio_l2266_226690

theorem factorial_ratio : (Nat.factorial 9) / (Nat.factorial 8) = 9 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2266_226690


namespace NUMINAMATH_CALUDE_intersection_empty_implies_k_leq_neg_one_l2266_226688

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k}

-- Theorem statement
theorem intersection_empty_implies_k_leq_neg_one (k : ℝ) : 
  M ∩ N k = ∅ → k ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_k_leq_neg_one_l2266_226688


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achieved_l2266_226682

theorem min_value_quadratic (x y : ℝ) : 
  2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x + 2 * y - 5 ≥ -10 := by
  sorry

theorem min_value_quadratic_achieved : 
  ∃ x y : ℝ, 2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x + 2 * y - 5 = -10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achieved_l2266_226682


namespace NUMINAMATH_CALUDE_max_value_expression_l2266_226679

open Real

theorem max_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (⨆ x : ℝ, 2 * (a - x) * (x - Real.sqrt (x^2 + b^2))) = b^2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2266_226679


namespace NUMINAMATH_CALUDE_polar_coordinate_equivalence_l2266_226672

/-- 
Given a point in polar coordinates (-5, 5π/7), prove that it is equivalent 
to the point (5, 12π/7) in standard polar coordinate representation, 
where r > 0 and 0 ≤ θ < 2π.
-/
theorem polar_coordinate_equivalence :
  ∀ (r θ : ℝ), 
  r = -5 ∧ θ = (5 * Real.pi) / 7 →
  ∃ (r' θ' : ℝ),
    r' > 0 ∧ 
    0 ≤ θ' ∧ 
    θ' < 2 * Real.pi ∧
    r' = 5 ∧ 
    θ' = (12 * Real.pi) / 7 ∧
    (r * (Real.cos θ), r * (Real.sin θ)) = (r' * (Real.cos θ'), r' * (Real.sin θ')) :=
by sorry

end NUMINAMATH_CALUDE_polar_coordinate_equivalence_l2266_226672


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2266_226654

/-- A geometric sequence with first term 1 and third term 4 has a common ratio of ±2 -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℝ), 
  (∀ n : ℕ, a (n + 1) = a n * a 1) → -- Geometric sequence condition
  a 1 = 1 →
  a 3 = 4 →
  ∃ q : ℝ, a 1 * q^2 = a 3 ∧ q = 2 ∨ q = -2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2266_226654


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2266_226662

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a2 : a 2 = 1)
  (h_a8 : a 8 = a 6 + 2 * a 4) :
  a 6 = 4 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2266_226662

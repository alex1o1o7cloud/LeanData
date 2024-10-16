import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l3969_396986

theorem problem_statement (a b : ℝ) (hab : a * b > 0) (hab2 : a^2 * b = 4) :
  (∃ (m : ℝ), ∀ (a b : ℝ), a * b > 0 → a^2 * b = 4 → a + b ≥ m ∧ 
    ∀ (m' : ℝ), (∀ (a b : ℝ), a * b > 0 → a^2 * b = 4 → a + b ≥ m') → m' ≤ m) ∧
  (∀ (x : ℝ), 2 * |x - 1| + |x| ≤ a + b ↔ -1/3 ≤ x ∧ x ≤ 5/3) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l3969_396986


namespace NUMINAMATH_CALUDE_benjie_margo_age_difference_l3969_396936

/-- The age difference between Benjie and Margo -/
def ageDifference (benjieAge : ℕ) (margoFutureAge : ℕ) (yearsTillMargoFutureAge : ℕ) : ℕ :=
  benjieAge - (margoFutureAge - yearsTillMargoFutureAge)

/-- Theorem stating the age difference between Benjie and Margo -/
theorem benjie_margo_age_difference :
  ageDifference 6 4 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_benjie_margo_age_difference_l3969_396936


namespace NUMINAMATH_CALUDE_operation_problem_l3969_396950

-- Define the set of operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply an operation
def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_problem (diamond circ : Operation) :
  (apply_op diamond 10 4) / (apply_op circ 6 2) = 5 →
  (apply_op diamond 8 3) / (apply_op circ 10 5) = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_operation_problem_l3969_396950


namespace NUMINAMATH_CALUDE_integer_sum_l3969_396920

theorem integer_sum (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_l3969_396920


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l3969_396995

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function y = (x+1)(x-a) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

/-- If f(a) is an even function, then a = -1 -/
theorem even_function_implies_a_eq_neg_one (a : ℝ) :
  IsEven (f a) → a = -1 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l3969_396995


namespace NUMINAMATH_CALUDE_percent_of_percent_l3969_396997

theorem percent_of_percent (a b : ℝ) (ha : a = 20) (hb : b = 25) :
  (a / 100) * (b / 100) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l3969_396997


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l3969_396996

/-- Definition of the set of points M(x, y) satisfying the given equation -/
def TrajectorySet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; Real.sqrt (x^2 + (y-3)^2) + Real.sqrt (x^2 + (y+3)^2) = 10}

/-- Definition of an ellipse with foci (0, -3) and (0, 3), and major axis length 10 -/
def EllipseSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; 
               Real.sqrt (x^2 + (y+3)^2) + Real.sqrt (x^2 + (y-3)^2) = 10}

/-- Theorem stating that the trajectory set is equivalent to the ellipse set -/
theorem trajectory_is_ellipse : TrajectorySet = EllipseSet := by
  sorry


end NUMINAMATH_CALUDE_trajectory_is_ellipse_l3969_396996


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_l3969_396912

/-- The area of a geometric figure formed by a rectangle and an additional triangle -/
theorem rectangle_triangle_area (a b : ℝ) (h : 0 < a ∧ a < b) :
  let diagonal := Real.sqrt (a^2 + b^2)
  let triangle_area := (b * diagonal) / 4
  let total_area := a * b + triangle_area
  total_area = a * b + (b * Real.sqrt (a^2 + b^2)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_l3969_396912


namespace NUMINAMATH_CALUDE_percentage_problem_l3969_396910

theorem percentage_problem (P : ℝ) : 
  (0.15 * P * (0.5 * 4800) = 108) → P = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3969_396910


namespace NUMINAMATH_CALUDE_triangle_height_l3969_396979

/-- Given a triangle with area 615 m² and one side 123 meters, 
    the perpendicular height to that side is 10 meters. -/
theorem triangle_height (area : ℝ) (side : ℝ) (height : ℝ) : 
  area = 615 ∧ side = 123 → height = (2 * area) / side → height = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l3969_396979


namespace NUMINAMATH_CALUDE_product_remainder_l3969_396963

theorem product_remainder (a b c : ℕ) (ha : a % 10 = 3) (hb : b % 10 = 2) (hc : c % 10 = 4) :
  (a * b * c) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l3969_396963


namespace NUMINAMATH_CALUDE_m_range_for_inequality_l3969_396915

theorem m_range_for_inequality (m : ℝ) : 
  (∀ x : ℝ, x ≤ -1 → (m - m^2) * 4^x + 2^x + 1 > 0) → 
  -2 < m ∧ m < 3 := by
sorry

end NUMINAMATH_CALUDE_m_range_for_inequality_l3969_396915


namespace NUMINAMATH_CALUDE_f_geq_g_condition_h_max_value_l3969_396913

-- Define the functions
def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * abs (x - 1)
def h (a x : ℝ) : ℝ := abs (f x) + g a x

-- Theorem 1: Condition for f(x) ≥ g(x) for all x ∈ ℝ
theorem f_geq_g_condition (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ -2 :=
sorry

-- Theorem 2: Maximum value of h(x) on [-2, 2]
theorem h_max_value (a : ℝ) :
  (∃ m : ℝ, ∀ x ∈ Set.Icc (-2) 2, h a x ≤ m ∧
   ∃ x₀ ∈ Set.Icc (-2) 2, h a x₀ = m) ∧
  (let m := if a ≥ 0 then 3*a + 3
            else if a ≥ -3 then a + 3
            else 0;
   ∀ x ∈ Set.Icc (-2) 2, h a x ≤ m ∧
   ∃ x₀ ∈ Set.Icc (-2) 2, h a x₀ = m) :=
sorry

end NUMINAMATH_CALUDE_f_geq_g_condition_h_max_value_l3969_396913


namespace NUMINAMATH_CALUDE_price_reduction_equation_l3969_396946

theorem price_reduction_equation (x : ℝ) : 
  (100 : ℝ) * (1 - x)^2 = 80 ↔ 
  (∃ (price1 price2 : ℝ), 
    price1 = 100 * (1 - x) ∧ 
    price2 = price1 * (1 - x) ∧ 
    price2 = 80) :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l3969_396946


namespace NUMINAMATH_CALUDE_lizas_paycheck_amount_l3969_396935

/-- Calculates the amount of Liza's paycheck given her initial balance, expenses, and final balance -/
def calculate_paycheck (initial_balance rent electricity internet phone final_balance : ℕ) : ℕ :=
  final_balance + rent + electricity + internet + phone - initial_balance

/-- Theorem stating that Liza's paycheck is $1563 given the provided financial information -/
theorem lizas_paycheck_amount :
  calculate_paycheck 800 450 117 100 70 1563 = 1563 := by
  sorry

end NUMINAMATH_CALUDE_lizas_paycheck_amount_l3969_396935


namespace NUMINAMATH_CALUDE_thirteenth_most_likely_friday_l3969_396967

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents the Gregorian calendar -/
structure GregorianCalendar where
  /-- The current year in the 400-year cycle -/
  year : Nat
  /-- Whether the current year is a leap year -/
  is_leap_year : Bool
  /-- The day of the week for the 1st of January of the current year -/
  first_day : DayOfWeek

/-- Counts the occurrences of the 13th falling on each day of the week in a 400-year cycle -/
def count_13ths (calendar : GregorianCalendar) : DayOfWeek → Nat
  | _ => sorry

/-- Theorem: The 13th day of the month falls on Friday more often than on any other day
    in a complete 400-year cycle of the Gregorian calendar -/
theorem thirteenth_most_likely_friday (calendar : GregorianCalendar) :
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Friday → count_13ths calendar DayOfWeek.Friday > count_13ths calendar d := by
  sorry

#check thirteenth_most_likely_friday

end NUMINAMATH_CALUDE_thirteenth_most_likely_friday_l3969_396967


namespace NUMINAMATH_CALUDE_right_triangle_side_lengths_l3969_396972

theorem right_triangle_side_lengths (x : ℝ) :
  (((2*x + 2)^2 = (x + 4)^2 + (x + 2)^2 ∨ (x + 4)^2 = (2*x + 2)^2 + (x + 2)^2) ∧ 
   x > 0 ∧ 2*x + 2 > 0 ∧ x + 4 > 0 ∧ x + 2 > 0) ↔ 
  (x = 4 ∨ x = 1) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_side_lengths_l3969_396972


namespace NUMINAMATH_CALUDE_helen_thanksgiving_desserts_l3969_396992

/-- The number of chocolate chip cookies Helen baked -/
def chocolate_chip_cookies : ℕ := 435

/-- The number of sugar cookies Helen baked -/
def sugar_cookies : ℕ := 139

/-- The number of brownies Helen made -/
def brownies : ℕ := 215

/-- The total number of desserts Helen prepared for Thanksgiving -/
def total_desserts : ℕ := chocolate_chip_cookies + sugar_cookies + brownies

theorem helen_thanksgiving_desserts : total_desserts = 789 := by
  sorry

end NUMINAMATH_CALUDE_helen_thanksgiving_desserts_l3969_396992


namespace NUMINAMATH_CALUDE_complex_equation_implies_product_l3969_396937

/-- Given that (1+mi)/i = 1+ni where m, n ∈ ℝ and i is the imaginary unit, prove that mn = -1 -/
theorem complex_equation_implies_product (m n : ℝ) : 
  (1 + m * Complex.I) / Complex.I = 1 + n * Complex.I → m * n = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_product_l3969_396937


namespace NUMINAMATH_CALUDE_sandy_mall_change_l3969_396914

/-- The change Sandy received after buying clothes at the mall -/
def sandys_change (pants_cost shirt_cost bill_amount : ℚ) : ℚ :=
  bill_amount - (pants_cost + shirt_cost)

/-- Theorem stating that Sandy's change is $2.51 given the problem conditions -/
theorem sandy_mall_change :
  sandys_change 9.24 8.25 20 = 2.51 := by
  sorry

end NUMINAMATH_CALUDE_sandy_mall_change_l3969_396914


namespace NUMINAMATH_CALUDE_tyler_saltwater_animals_l3969_396947

/-- The number of aquariums Tyler has -/
def num_aquariums : ℕ := 8

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 64

/-- The total number of saltwater animals Tyler has -/
def total_animals : ℕ := num_aquariums * animals_per_aquarium

theorem tyler_saltwater_animals :
  total_animals = 512 :=
sorry

end NUMINAMATH_CALUDE_tyler_saltwater_animals_l3969_396947


namespace NUMINAMATH_CALUDE_mixture_percentage_l3969_396941

theorem mixture_percentage (p_weight q_weight : ℝ) (p_percent q_percent : ℝ) :
  p_weight = 200 →
  q_weight = 800 →
  p_percent = 0.5 →
  q_percent = 1.5 →
  let total_weight := p_weight + q_weight
  let x_in_p := p_weight * (p_percent / 100)
  let x_in_q := q_weight * (q_percent / 100)
  let total_x := x_in_p + x_in_q
  total_x / total_weight * 100 = 1.3 := by
sorry

end NUMINAMATH_CALUDE_mixture_percentage_l3969_396941


namespace NUMINAMATH_CALUDE_otimes_one_eq_two_implies_k_eq_one_l3969_396964

def otimes (a b : ℝ) : ℝ := a * b + a + b^2

theorem otimes_one_eq_two_implies_k_eq_one (k : ℝ) (h1 : k > 0) (h2 : otimes 1 k = 2) : k = 1 := by
  sorry

end NUMINAMATH_CALUDE_otimes_one_eq_two_implies_k_eq_one_l3969_396964


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3969_396903

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | x > 0} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3969_396903


namespace NUMINAMATH_CALUDE_equal_tape_length_l3969_396943

def minyoung_tape : ℕ := 1748
def yoojung_tape : ℕ := 850
def tape_to_give : ℕ := 449

theorem equal_tape_length : 
  minyoung_tape - tape_to_give = yoojung_tape + tape_to_give :=
by sorry

end NUMINAMATH_CALUDE_equal_tape_length_l3969_396943


namespace NUMINAMATH_CALUDE_houses_with_both_features_l3969_396948

theorem houses_with_both_features (total : ℕ) (garage : ℕ) (pool : ℕ) (neither : ℕ)
  (h_total : total = 85)
  (h_garage : garage = 50)
  (h_pool : pool = 40)
  (h_neither : neither = 30) :
  ∃ (both : ℕ), both = garage + pool - (total - neither) :=
by
  sorry

end NUMINAMATH_CALUDE_houses_with_both_features_l3969_396948


namespace NUMINAMATH_CALUDE_remaining_cakes_l3969_396911

def cakes_per_day : ℕ := 4
def baking_days : ℕ := 6
def eating_frequency : ℕ := 2

def total_baked (cakes_per_day baking_days : ℕ) : ℕ :=
  cakes_per_day * baking_days

def cakes_eaten (baking_days eating_frequency : ℕ) : ℕ :=
  baking_days / eating_frequency

theorem remaining_cakes :
  total_baked cakes_per_day baking_days - cakes_eaten baking_days eating_frequency = 21 :=
by sorry

end NUMINAMATH_CALUDE_remaining_cakes_l3969_396911


namespace NUMINAMATH_CALUDE_alpha_value_l3969_396908

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (2 * (α - 3 * β)).re > 0)
  (h3 : β = 5 + 4 * Complex.I) :
  α = 16 - 4 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_alpha_value_l3969_396908


namespace NUMINAMATH_CALUDE_chord_bisector_line_l3969_396938

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a parabola y² = 4x -/
def onParabola (p : Point) : Prop := p.y^2 = 4 * p.x

/-- Checks if a point lies on a line -/
def onLine (p : Point) (l : Line) : Prop := l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop :=
  m.x = (a.x + b.x) / 2 ∧ m.y = (a.y + b.y) / 2

/-- The main theorem -/
theorem chord_bisector_line (A B : Point) (P : Point) :
  onParabola A ∧ onParabola B ∧ 
  isMidpoint P A B ∧ 
  P.x = 1 ∧ P.y = 1 →
  ∃ l : Line, l.a = 2 ∧ l.b = -1 ∧ l.c = -1 ∧ onLine A l ∧ onLine B l :=
by sorry

end NUMINAMATH_CALUDE_chord_bisector_line_l3969_396938


namespace NUMINAMATH_CALUDE_larger_number_value_l3969_396958

theorem larger_number_value (x y : ℝ) (hx : x = 48) (hdiff : y - x = (1/3) * y) : y = 72 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_value_l3969_396958


namespace NUMINAMATH_CALUDE_nine_grams_combinations_l3969_396978

def weight_combinations (n : ℕ) : ℕ :=
  let ones := Finset.range 4
  let twos := Finset.range 4
  let fives := Finset.range 2
  (ones.product twos).product fives
    |>.filter (fun ((a, b), c) => a + 2*b + 5*c == n)
    |>.card

theorem nine_grams_combinations : weight_combinations 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_nine_grams_combinations_l3969_396978


namespace NUMINAMATH_CALUDE_tom_hiking_probability_l3969_396956

theorem tom_hiking_probability (p_fog : ℝ) (p_hike_foggy : ℝ) (p_hike_clear : ℝ)
  (h_fog : p_fog = 0.5)
  (h_hike_foggy : p_hike_foggy = 0.3)
  (h_hike_clear : p_hike_clear = 0.9) :
  p_fog * p_hike_foggy + (1 - p_fog) * p_hike_clear = 0.6 := by
  sorry

#check tom_hiking_probability

end NUMINAMATH_CALUDE_tom_hiking_probability_l3969_396956


namespace NUMINAMATH_CALUDE_sum_of_roots_l3969_396993

-- Define the cubic equation
def cubic_equation (p q d x : ℝ) : Prop := 2 * x^3 - p * x^2 + q * x - d = 0

-- Define the theorem
theorem sum_of_roots (p q d x₁ x₂ x₃ : ℝ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)
  (h_roots : cubic_equation p q d x₁ ∧ cubic_equation p q d x₂ ∧ cubic_equation p q d x₃)
  (h_positive : p > 0 ∧ q > 0 ∧ d > 0)
  (h_relation : q = 2 * d) :
  x₁ + x₂ + x₃ = p / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3969_396993


namespace NUMINAMATH_CALUDE_completing_square_form_l3969_396918

theorem completing_square_form (x : ℝ) : 
  (x^2 - 6*x - 3 = 0) ↔ ((x - 3)^2 = 12) := by sorry

end NUMINAMATH_CALUDE_completing_square_form_l3969_396918


namespace NUMINAMATH_CALUDE_triangle_angle_A_l3969_396925

theorem triangle_angle_A (a b : ℝ) (B : ℝ) (hA : 0 < a) (hB : 0 < b) (hC : 0 < B) :
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  B = π / 3 →
  ∃ (A : ℝ), 0 < A ∧ A < 2 * π / 3 ∧ Real.sin A = Real.sqrt 2 / 2 ∧ A = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l3969_396925


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_when_f_always_ge_4_l3969_396945

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem solution_set_when_a_is_4 :
  {x : ℝ | f 4 x ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} := by sorry

-- Part 2
theorem range_of_a_when_f_always_ge_4 :
  (∀ x : ℝ, f a x ≥ 4) → (a ≤ -3 ∨ a ≥ 5) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_when_f_always_ge_4_l3969_396945


namespace NUMINAMATH_CALUDE_triangle_count_l3969_396939

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of collinear triplets in the given configuration -/
def collinearTriplets : ℕ := 16

/-- The total number of points in the configuration -/
def totalPoints : ℕ := 12

/-- The number of points needed to form a triangle -/
def pointsPerTriangle : ℕ := 3

theorem triangle_count :
  choose totalPoints pointsPerTriangle - collinearTriplets = 204 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l3969_396939


namespace NUMINAMATH_CALUDE_expression_evaluation_l3969_396991

theorem expression_evaluation : 
  Real.sqrt 2 * (2 ^ (3/2)) + 15 / 5 * 3 - Real.sqrt 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3969_396991


namespace NUMINAMATH_CALUDE_sum_last_three_coefficients_eq_21_l3969_396973

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function that calculates the sum of the last three coefficients
def sum_last_three_coefficients (a : ℝ) : ℝ :=
  binomial 8 0 * 1 + binomial 8 1 * (-1) + binomial 8 2 * 1

-- Theorem statement
theorem sum_last_three_coefficients_eq_21 :
  ∀ a : ℝ, sum_last_three_coefficients a = 21 := by sorry

end NUMINAMATH_CALUDE_sum_last_three_coefficients_eq_21_l3969_396973


namespace NUMINAMATH_CALUDE_probability_all_same_suit_l3969_396952

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of cards dealt to a player -/
def handSize : ℕ := 13

/-- The number of suits in a standard deck -/
def numSuits : ℕ := 4

/-- The number of cards in each suit -/
def cardsPerSuit : ℕ := deckSize / numSuits

theorem probability_all_same_suit :
  (numSuits : ℚ) / (deckSize.choose handSize : ℚ) =
  (numSuits : ℚ) / (Nat.choose deckSize handSize : ℚ) := by
  sorry

/-- The probability of all cards in a hand being from the same suit -/
def probabilitySameSuit : ℚ :=
  (numSuits : ℚ) / (Nat.choose deckSize handSize : ℚ)

end NUMINAMATH_CALUDE_probability_all_same_suit_l3969_396952


namespace NUMINAMATH_CALUDE_bee_speed_is_11_5_l3969_396933

/-- Represents the bee's journey with given conditions -/
structure BeeJourney where
  v : ℝ  -- Bee's constant actual speed
  t_dr : ℝ := 10  -- Time from daisy to rose
  t_rp : ℝ := 6   -- Time from rose to poppy
  t_pt : ℝ := 8   -- Time from poppy to tulip
  slow : ℝ := 2   -- Speed reduction due to crosswind
  boost : ℝ := 3  -- Speed increase due to crosswind

  d_dr : ℝ := t_dr * (v - slow)  -- Distance from daisy to rose
  d_rp : ℝ := t_rp * (v + boost) -- Distance from rose to poppy
  d_pt : ℝ := t_pt * (v - slow)  -- Distance from poppy to tulip

  h_distance_diff : d_dr = d_rp + 8  -- Distance condition
  h_distance_equal : d_pt = d_dr     -- Distance equality condition

/-- Theorem stating that the bee's speed is 11.5 m/s given the conditions -/
theorem bee_speed_is_11_5 (j : BeeJourney) : j.v = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_bee_speed_is_11_5_l3969_396933


namespace NUMINAMATH_CALUDE_max_valid_coloring_size_l3969_396998

/-- A type representing the color of a square on the board -/
inductive Color
| Black
| White

/-- A function type representing a coloring of an n × n board -/
def BoardColoring (n : ℕ) := Fin n → Fin n → Color

/-- Predicate to check if a board coloring satisfies the condition -/
def ValidColoring (n : ℕ) (coloring : BoardColoring n) : Prop :=
  ∀ (r1 r2 c1 c2 : Fin n), r1 ≠ r2 → c1 ≠ c2 → 
    (coloring r1 c1 = coloring r1 c2 → coloring r2 c1 ≠ coloring r2 c2) ∧
    (coloring r1 c1 = coloring r2 c1 → coloring r1 c2 ≠ coloring r2 c2)

/-- Theorem stating that 4 is the maximum value of n for which a valid coloring exists -/
theorem max_valid_coloring_size :
  (∃ (coloring : BoardColoring 4), ValidColoring 4 coloring) ∧
  (∀ n : ℕ, n > 4 → ¬∃ (coloring : BoardColoring n), ValidColoring n coloring) :=
sorry

end NUMINAMATH_CALUDE_max_valid_coloring_size_l3969_396998


namespace NUMINAMATH_CALUDE_sprite_volume_calculation_l3969_396957

def maaza_volume : ℕ := 20
def pepsi_volume : ℕ := 144
def total_cans : ℕ := 133

theorem sprite_volume_calculation :
  ∃ (can_volume sprite_volume : ℕ),
    can_volume > 0 ∧
    maaza_volume % can_volume = 0 ∧
    pepsi_volume % can_volume = 0 ∧
    sprite_volume % can_volume = 0 ∧
    maaza_volume / can_volume + pepsi_volume / can_volume + sprite_volume / can_volume = total_cans ∧
    sprite_volume = 368 := by
  sorry

end NUMINAMATH_CALUDE_sprite_volume_calculation_l3969_396957


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_equals_four_sqrt_three_plus_one_l3969_396917

theorem sqrt_sum_difference_equals_four_sqrt_three_plus_one :
  Real.sqrt 12 + Real.sqrt 27 - |1 - Real.sqrt 3| = 4 * Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_equals_four_sqrt_three_plus_one_l3969_396917


namespace NUMINAMATH_CALUDE_election_result_l3969_396944

theorem election_result (total_votes : ℕ) (invalid_percentage : ℚ) (second_candidate_votes : ℕ) : 
  total_votes = 7000 →
  invalid_percentage = 1/5 →
  second_candidate_votes = 2520 →
  (((1 - invalid_percentage) * total_votes - second_candidate_votes) / ((1 - invalid_percentage) * total_votes) : ℚ) = 11/20 := by
sorry

end NUMINAMATH_CALUDE_election_result_l3969_396944


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l3969_396942

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 5*x₁ + 16 = x₁ + 55) ∧
  (x₂^2 - 5*x₂ + 16 = x₂ + 55) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l3969_396942


namespace NUMINAMATH_CALUDE_march_first_is_sunday_l3969_396970

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific March with given properties -/
structure SpecificMarch where
  daysInMonth : Nat
  wednesdayCount : Nat
  saturdayCount : Nat
  firstDay : DayOfWeek

/-- Theorem stating that March 1st is a Sunday given the conditions -/
theorem march_first_is_sunday (march : SpecificMarch) 
  (h1 : march.daysInMonth = 31)
  (h2 : march.wednesdayCount = 4)
  (h3 : march.saturdayCount = 4) :
  march.firstDay = DayOfWeek.Sunday := by
  sorry

#check march_first_is_sunday

end NUMINAMATH_CALUDE_march_first_is_sunday_l3969_396970


namespace NUMINAMATH_CALUDE_sum_of_roots_difference_l3969_396922

theorem sum_of_roots_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : Real.sqrt x / Real.sqrt y - Real.sqrt y / Real.sqrt x = 7/12)
  (h2 : x - y = 7) : x + y = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_difference_l3969_396922


namespace NUMINAMATH_CALUDE_number_equation_solution_l3969_396981

theorem number_equation_solution : ∃ x : ℝ, 33 + 3 * x = 48 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3969_396981


namespace NUMINAMATH_CALUDE_cone_lateral_area_l3969_396974

/-- The lateral area of a cone with base radius 3 and slant height 5 is 15π -/
theorem cone_lateral_area :
  let base_radius : ℝ := 3
  let slant_height : ℝ := 5
  let lateral_area := π * base_radius * slant_height
  lateral_area = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l3969_396974


namespace NUMINAMATH_CALUDE_binomial_coefficient_26_6_l3969_396966

theorem binomial_coefficient_26_6 
  (h1 : Nat.choose 24 5 = 42504)
  (h2 : Nat.choose 25 5 = 53130)
  (h3 : Nat.choose 25 6 = 177100) :
  Nat.choose 26 6 = 230230 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_26_6_l3969_396966


namespace NUMINAMATH_CALUDE_hundredth_digit_of_seven_twentysixths_l3969_396965

/-- The fraction we're working with -/
def f : ℚ := 7/26

/-- The length of the repeating sequence in the decimal representation of f -/
def repeat_length : ℕ := 9

/-- The repeating sequence in the decimal representation of f -/
def repeat_sequence : List ℕ := [2, 6, 9, 2, 3, 0, 7, 6, 9]

/-- The position we're interested in -/
def target_position : ℕ := 100

theorem hundredth_digit_of_seven_twentysixths (h1 : f = 7/26)
  (h2 : repeat_length = 9)
  (h3 : repeat_sequence = [2, 6, 9, 2, 3, 0, 7, 6, 9])
  (h4 : target_position = 100) :
  repeat_sequence[(target_position - 1) % repeat_length] = 2 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_digit_of_seven_twentysixths_l3969_396965


namespace NUMINAMATH_CALUDE_stratified_sample_probability_l3969_396905

/-- Represents the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of an event -/
def probability (favorable_outcomes total_outcomes : ℕ) : ℚ := sorry

theorem stratified_sample_probability : 
  let total_sample_size := 6
  let elementary_teachers_in_sample := 3
  let further_selection_size := 2
  probability (choose elementary_teachers_in_sample further_selection_size) 
              (choose total_sample_size further_selection_size) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_probability_l3969_396905


namespace NUMINAMATH_CALUDE_regular_hexagon_interior_angle_l3969_396954

/-- The measure of each interior angle in a regular hexagon is 120 degrees. -/
theorem regular_hexagon_interior_angle : ℝ := by
  -- Define a regular hexagon
  let regular_hexagon : Nat := 6

  -- Define the formula for the sum of interior angles of a polygon
  let sum_of_interior_angles (n : Nat) : ℝ := (n - 2) * 180

  -- Calculate the sum of interior angles for a hexagon
  let total_angle_sum : ℝ := sum_of_interior_angles regular_hexagon

  -- Calculate the measure of each interior angle
  let interior_angle : ℝ := total_angle_sum / regular_hexagon

  -- Prove that the interior angle is 120 degrees
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_interior_angle_l3969_396954


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l3969_396959

theorem complex_sum_of_parts (z : ℂ) (h : (1 + 2*I)*z = 4 + 3*I) :
  z.re + z.im = 1 := by sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l3969_396959


namespace NUMINAMATH_CALUDE_polygon_diagonals_l3969_396909

theorem polygon_diagonals (n : ℕ) (h : (n - 2) * 180 + 360 = 2160) : 
  n * (n - 3) / 2 = 54 := by sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l3969_396909


namespace NUMINAMATH_CALUDE_county_population_distribution_l3969_396990

theorem county_population_distribution (less_than_10k : ℝ) (between_10k_and_100k : ℝ) :
  less_than_10k = 25 →
  between_10k_and_100k = 59 →
  less_than_10k + between_10k_and_100k = 84 :=
by sorry

end NUMINAMATH_CALUDE_county_population_distribution_l3969_396990


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_sum_of_digits_10_l3969_396962

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isFirstYearAfter2010WithSumOfDigits10 (year : Nat) : Prop :=
  year > 2010 ∧ 
  sumOfDigits year = 10 ∧ 
  ∀ y, 2010 < y ∧ y < year → sumOfDigits y ≠ 10

theorem first_year_after_2010_with_sum_of_digits_10 :
  isFirstYearAfter2010WithSumOfDigits10 2035 := by sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_sum_of_digits_10_l3969_396962


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l3969_396927

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l3969_396927


namespace NUMINAMATH_CALUDE_square_side_length_from_circle_area_l3969_396955

/-- Given a square from which a circle is described, if the area of the circle is 78.53981633974483 square inches, then the side length of the square is 10 inches. -/
theorem square_side_length_from_circle_area (circle_area : ℝ) (square_side : ℝ) : 
  circle_area = 78.53981633974483 →
  circle_area = Real.pi * (square_side / 2)^2 →
  square_side = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_from_circle_area_l3969_396955


namespace NUMINAMATH_CALUDE_max_distinct_numbers_in_circle_l3969_396982

/-- Given a circular arrangement of 2023 numbers where each number is the product of its two neighbors,
    the maximum number of distinct numbers is 1. -/
theorem max_distinct_numbers_in_circle (nums : Fin 2023 → ℝ) 
    (h : ∀ i : Fin 2023, nums i = nums (i - 1) * nums (i + 1)) : 
    Finset.card (Finset.image nums Finset.univ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_distinct_numbers_in_circle_l3969_396982


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_200_l3969_396928

theorem closest_integer_to_cube_root_200 : 
  ∃ n : ℤ, n = 6 ∧ ∀ m : ℤ, |m^3 - 200| ≥ |n^3 - 200| := by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_200_l3969_396928


namespace NUMINAMATH_CALUDE_power_plus_sum_l3969_396926

theorem power_plus_sum : 10^2 + 10 + 1 = 111 := by
  sorry

end NUMINAMATH_CALUDE_power_plus_sum_l3969_396926


namespace NUMINAMATH_CALUDE_tangent_line_property_l3969_396975

/-- Given a function f: ℝ → ℝ, if the tangent line to the graph of f at the point (2, f(2))
    has the equation 2x - y - 3 = 0, then f(2) + f'(2) = 3. -/
theorem tangent_line_property (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x y, y = f 2 → 2 * x - y - 3 = 0 ↔ y = 2 * x - 3) →
  f 2 + deriv f 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_property_l3969_396975


namespace NUMINAMATH_CALUDE_problem_solution_l3969_396983

theorem problem_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (a * Real.sin (π / 7) + b * Real.cos (π / 7)) / 
       (a * Real.cos (π / 7) - b * Real.sin (π / 7)) = Real.tan (10 * π / 21)) : 
  b / a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3969_396983


namespace NUMINAMATH_CALUDE_sam_exchange_probability_l3969_396907

/-- The number of toys in the vending machine -/
def num_toys : ℕ := 10

/-- The price of the first toy in cents -/
def first_toy_price : ℕ := 50

/-- The price increment between consecutive toys in cents -/
def price_increment : ℕ := 25

/-- The number of quarters Sam has -/
def sam_quarters : ℕ := 10

/-- The price of Sam's favorite toy in cents -/
def favorite_toy_price : ℕ := 225

/-- The total number of possible toy arrangements -/
def total_arrangements : ℕ := Nat.factorial num_toys

/-- The number of favorable arrangements where Sam can buy his favorite toy without exchanging his bill -/
def favorable_arrangements : ℕ := Nat.factorial 9 + Nat.factorial 8 + Nat.factorial 7 + Nat.factorial 6 + Nat.factorial 5

/-- The probability that Sam needs to exchange his bill -/
def exchange_probability : ℚ := 1 - (favorable_arrangements : ℚ) / (total_arrangements : ℚ)

theorem sam_exchange_probability :
  exchange_probability = 8 / 9 := by sorry

end NUMINAMATH_CALUDE_sam_exchange_probability_l3969_396907


namespace NUMINAMATH_CALUDE_cone_base_radius_l3969_396977

/-- Given a cone with slant height 5 and lateral area 15π, its base radius is 3 -/
theorem cone_base_radius (s : ℝ) (L : ℝ) (r : ℝ) : 
  s = 5 → L = 15 * Real.pi → L = Real.pi * r * s → r = 3 := by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3969_396977


namespace NUMINAMATH_CALUDE_compute_fraction_square_l3969_396932

theorem compute_fraction_square : 6 * (3 / 7)^2 = 54 / 49 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_square_l3969_396932


namespace NUMINAMATH_CALUDE_simplify_polynomial_product_l3969_396971

theorem simplify_polynomial_product (a : ℝ) : 
  (1 : ℝ) * (2 * a) * (3 * a^2) * (4 * a^3) * (5 * a^4) * (6 * a^5) = 720 * a^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_product_l3969_396971


namespace NUMINAMATH_CALUDE_quadratic_vertex_x_coordinate_l3969_396976

/-- Given a quadratic function f(x) = ax^2 + bx + c that passes through 
    the points (2, 3), (8, -1), and (11, 8), prove that the x-coordinate 
    of its vertex is 142/23. -/
theorem quadratic_vertex_x_coordinate 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 2 = 3) 
  (h3 : f 8 = -1) 
  (h4 : f 11 = 8) : 
  -b / (2 * a) = 142 / 23 := by
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_x_coordinate_l3969_396976


namespace NUMINAMATH_CALUDE_problem_solution_l3969_396940

theorem problem_solution (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 228498 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3969_396940


namespace NUMINAMATH_CALUDE_coin_division_sum_equals_pairs_l3969_396949

/-- Represents the process of dividing coins into piles --/
def CoinDivisionProcess : Type := List (Nat × Nat)

/-- The number of coins --/
def n : Nat := 25

/-- Calculates the sum of products for a given division process --/
def sum_of_products (process : CoinDivisionProcess) : Nat :=
  process.foldl (fun sum pair => sum + pair.1 * pair.2) 0

/-- Represents all possible division processes for n coins --/
def all_division_processes (n : Nat) : Set CoinDivisionProcess :=
  sorry

/-- Theorem stating that the sum of products equals the number of pairs of coins --/
theorem coin_division_sum_equals_pairs :
  ∀ (process : CoinDivisionProcess),
    process ∈ all_division_processes n →
    sum_of_products process = n.choose 2 := by
  sorry

end NUMINAMATH_CALUDE_coin_division_sum_equals_pairs_l3969_396949


namespace NUMINAMATH_CALUDE_square_of_linear_cyclic_l3969_396987

variable (a b c A B C : ℝ)

/-- Two linear polynomials sum to a square of a linear polynomial iff their coefficients satisfy this condition -/
def is_square_of_linear (α β γ δ : ℝ) : Prop :=
  α * δ = β * γ

/-- The main theorem: if two pairs of expressions are squares of linear polynomials, 
    then the third pair is also a square of a linear polynomial -/
theorem square_of_linear_cyclic 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0)
  (h1 : is_square_of_linear a b A B)
  (h2 : is_square_of_linear b c B C) :
  is_square_of_linear c a C A :=
sorry

end NUMINAMATH_CALUDE_square_of_linear_cyclic_l3969_396987


namespace NUMINAMATH_CALUDE_unique_solution_in_interval_l3969_396931

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 1

-- State the theorem
theorem unique_solution_in_interval :
  ∃! a : ℝ, 0 < a ∧ a < 3 ∧ f a = 7 ∧ a = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_in_interval_l3969_396931


namespace NUMINAMATH_CALUDE_finite_sum_evaluation_l3969_396900

theorem finite_sum_evaluation : 
  let S := (1 : ℚ) / 4^1 + 2 / 4^2 + 3 / 4^3 + 4 / 4^4 + 5 / 4^5
  S = 4/3 * (1 - 1/4^6) :=
by sorry

end NUMINAMATH_CALUDE_finite_sum_evaluation_l3969_396900


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3969_396988

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (x - 5) + 1 = 10 → x = 86 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3969_396988


namespace NUMINAMATH_CALUDE_polynomial_value_l3969_396951

/-- A polynomial with integer coefficients where each coefficient is between 0 and 3 (inclusive) -/
def IntPolynomial (n : ℕ) := { p : Polynomial ℤ // ∀ i, 0 ≤ p.coeff i ∧ p.coeff i < 4 }

/-- The theorem stating that if P(2) = 66, then P(3) = 111 for the given polynomial -/
theorem polynomial_value (n : ℕ) (P : IntPolynomial n) 
  (h : P.val.eval 2 = 66) : P.val.eval 3 = 111 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l3969_396951


namespace NUMINAMATH_CALUDE_largest_common_term_l3969_396961

def is_in_first_sequence (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k + 3

def is_in_second_sequence (n : ℕ) : Prop := ∃ m : ℕ, n = 7 * m + 5

theorem largest_common_term :
  (∃ n : ℕ, n < 1000 ∧ is_in_first_sequence n ∧ is_in_second_sequence n) ∧
  (∀ n : ℕ, n < 1000 ∧ is_in_first_sequence n ∧ is_in_second_sequence n → n ≤ 989) ∧
  (is_in_first_sequence 989 ∧ is_in_second_sequence 989) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l3969_396961


namespace NUMINAMATH_CALUDE_at_least_three_solutions_nine_solutions_for_2019_l3969_396999

/-- The number of solutions to the equation 1/x + 1/y = 1/a for positive integers x, y, and a > 1 -/
def num_solutions (a : ℕ) : ℕ := sorry

/-- The proposition that there are at least three distinct solutions for any a > 1 -/
theorem at_least_three_solutions (a : ℕ) (ha : a > 1) :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℕ),
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    (1 : ℚ) / x₁ + (1 : ℚ) / y₁ = (1 : ℚ) / a ∧
    (1 : ℚ) / x₂ + (1 : ℚ) / y₂ = (1 : ℚ) / a ∧
    (1 : ℚ) / x₃ + (1 : ℚ) / y₃ = (1 : ℚ) / a :=
sorry

/-- The proposition that there are exactly 9 solutions when a = 2019 -/
theorem nine_solutions_for_2019 : num_solutions 2019 = 9 :=
sorry

end NUMINAMATH_CALUDE_at_least_three_solutions_nine_solutions_for_2019_l3969_396999


namespace NUMINAMATH_CALUDE_ellipse_and_intersection_l3969_396906

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define the line
def line (x y : ℝ) : Prop := Real.sqrt 3 * y - 2 * x - 2 = 0

theorem ellipse_and_intersection :
  -- Given conditions
  (ellipse_C 0 2) ∧ 
  (ellipse_C (1/2) (Real.sqrt 3)) ∧
  -- Prove the following
  (∀ x y : ℝ, ellipse_C x y ↔ x^2 + y^2/4 = 1) ∧ 
  (ellipse_C (-1) 0 ∧ line (-1) 0) ∧
  (ellipse_C (1/2) (Real.sqrt 3) ∧ line (1/2) (Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_intersection_l3969_396906


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l3969_396902

/-- Given a quadratic function f(x) = ax^2 + bx where a > 0 and b > 0,
    if the slope of the tangent line at x = 1 is 2,
    then the minimum value of (8a + b) / (ab) is 9. -/
theorem quadratic_function_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * a + b = 2) → (∀ x y : ℝ, x > 0 ∧ y > 0 → (8 * x + y) / (x * y) ≥ 9) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (8 * x + y) / (x * y) = 9) := by
  sorry

#check quadratic_function_minimum

end NUMINAMATH_CALUDE_quadratic_function_minimum_l3969_396902


namespace NUMINAMATH_CALUDE_rob_travel_time_l3969_396916

/-- The time it takes Rob to get to the national park -/
def rob_time : ℝ := 1

/-- The time it takes Mark to get to the national park -/
def mark_time : ℝ := 3 * rob_time

/-- The head start time Mark has -/
def head_start : ℝ := 2

theorem rob_travel_time : 
  head_start + rob_time = mark_time ∧ rob_time = 1 := by sorry

end NUMINAMATH_CALUDE_rob_travel_time_l3969_396916


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l3969_396968

theorem students_playing_both_sports (total : ℕ) (hockey : ℕ) (basketball : ℕ) (neither : ℕ) :
  total = 25 →
  hockey = 15 →
  basketball = 16 →
  neither = 4 →
  hockey + basketball - (total - neither) = 10 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l3969_396968


namespace NUMINAMATH_CALUDE_test_results_problem_l3969_396984

/-- Represents the number of questions a person got wrong on a test. -/
structure TestResult where
  wrong : Nat

/-- Represents the test results for Emily, Felix, Grace, and Henry. -/
structure GroupTestResults where
  emily : TestResult
  felix : TestResult
  grace : TestResult
  henry : TestResult

/-- The theorem statement for the test results problem. -/
theorem test_results_problem (results : GroupTestResults) : 
  (results.emily.wrong + results.felix.wrong + 4 = results.grace.wrong + results.henry.wrong) →
  (results.emily.wrong + results.henry.wrong = results.felix.wrong + results.grace.wrong + 8) →
  (results.grace.wrong = 6) →
  (results.emily.wrong = 8) := by
  sorry

#check test_results_problem

end NUMINAMATH_CALUDE_test_results_problem_l3969_396984


namespace NUMINAMATH_CALUDE_remainder_of_2583156_div_4_l3969_396924

theorem remainder_of_2583156_div_4 : 2583156 % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2583156_div_4_l3969_396924


namespace NUMINAMATH_CALUDE_simplify_fraction_l3969_396929

theorem simplify_fraction : (90 : ℚ) / 126 = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3969_396929


namespace NUMINAMATH_CALUDE_smallest_group_sample_size_l3969_396980

def stratified_sampling (total_sample_size : ℕ) (group_ratios : List ℕ) : List ℕ :=
  let total_ratio := group_ratios.sum
  group_ratios.map (λ ratio => (total_sample_size * ratio) / total_ratio)

theorem smallest_group_sample_size 
  (total_sample_size : ℕ) 
  (group_ratios : List ℕ) :
  total_sample_size = 20 →
  group_ratios = [5, 4, 1] →
  (stratified_sampling total_sample_size group_ratios).getLast! = 2 :=
by
  sorry

#eval stratified_sampling 20 [5, 4, 1]

end NUMINAMATH_CALUDE_smallest_group_sample_size_l3969_396980


namespace NUMINAMATH_CALUDE_square_root_of_nine_l3969_396919

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l3969_396919


namespace NUMINAMATH_CALUDE_max_value_theorem_l3969_396960

theorem max_value_theorem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 25) :
  (Real.sqrt (x + 64) + 2 * Real.sqrt (25 - x) + Real.sqrt x) ≤ Real.sqrt 328 ∧
  ∃ x₀, 0 ≤ x₀ ∧ x₀ ≤ 25 ∧ Real.sqrt (x₀ + 64) + 2 * Real.sqrt (25 - x₀) + Real.sqrt x₀ = Real.sqrt 328 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3969_396960


namespace NUMINAMATH_CALUDE_simple_interest_principal_l3969_396904

/-- Simple interest calculation --/
theorem simple_interest_principal (amount : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) :
  amount = 1456 ∧ rate = 0.05 ∧ time = 2.4 →
  principal = 1300 ∧ amount = principal * (1 + rate * time) := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l3969_396904


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3969_396921

theorem necessary_but_not_sufficient_condition : 
  (∀ x : ℝ, x > 0 → x > -2) ∧ 
  (∃ x : ℝ, x > -2 ∧ ¬(x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3969_396921


namespace NUMINAMATH_CALUDE_height_relation_l3969_396930

/-- Two right circular cylinders with equal volume and related radii -/
structure TwoCylinders where
  r1 : ℝ  -- radius of the first cylinder
  h1 : ℝ  -- height of the first cylinder
  r2 : ℝ  -- radius of the second cylinder
  h2 : ℝ  -- height of the second cylinder
  r1_pos : 0 < r1  -- r1 is positive
  h1_pos : 0 < h1  -- h1 is positive
  r2_pos : 0 < r2  -- r2 is positive
  h2_pos : 0 < h2  -- h2 is positive
  volume_eq : r1^2 * h1 = r2^2 * h2  -- volumes are equal
  radius_relation : r2 = 1.2 * r1  -- second radius is 20% more than the first

/-- The height of the first cylinder is 44% more than the height of the second cylinder -/
theorem height_relation (c : TwoCylinders) : c.h1 = 1.44 * c.h2 := by
  sorry

end NUMINAMATH_CALUDE_height_relation_l3969_396930


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3969_396923

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = Real.sqrt 119 ∧
  (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧
  (∀ y : ℝ, y < x → ⌊y^2⌋ - ⌊y⌋^2 ≠ 19) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3969_396923


namespace NUMINAMATH_CALUDE_yolas_past_weight_l3969_396994

/-- Yola's past weight given current weights and differences -/
theorem yolas_past_weight
  (yola_current : ℝ)
  (wanda_yola_current_diff : ℝ)
  (wanda_yola_past_diff : ℝ)
  (h1 : yola_current = 220)
  (h2 : wanda_yola_current_diff = 30)
  (h3 : wanda_yola_past_diff = 80) :
  yola_current - (wanda_yola_past_diff - wanda_yola_current_diff) = 170 :=
by
  sorry

end NUMINAMATH_CALUDE_yolas_past_weight_l3969_396994


namespace NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_for_x_squared_gt_9_l3969_396969

theorem x_gt_3_sufficient_not_necessary_for_x_squared_gt_9 :
  (∀ x : ℝ, x > 3 → x^2 > 9) ∧ (∃ x : ℝ, x^2 > 9 ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_for_x_squared_gt_9_l3969_396969


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l3969_396934

theorem factorial_fraction_simplification (N : ℕ) :
  (Nat.factorial (N - 1) * N^2) / Nat.factorial (N + 2) = N / (N + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l3969_396934


namespace NUMINAMATH_CALUDE_square_difference_of_solutions_l3969_396985

theorem square_difference_of_solutions (α β : ℝ) : 
  (α^2 = 2*α + 1) → (β^2 = 2*β + 1) → (α ≠ β) → (α - β)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_square_difference_of_solutions_l3969_396985


namespace NUMINAMATH_CALUDE_sum_of_powers_mod_five_l3969_396989

theorem sum_of_powers_mod_five (n : ℕ) (hn : n > 0) : 
  (1^n + 2^n + 3^n + 4^n + 5^n) % 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_powers_mod_five_l3969_396989


namespace NUMINAMATH_CALUDE_triangle_angle_c_value_l3969_396901

/-- Given a triangle ABC with internal angles A, B, and C, and vectors m and n
    satisfying certain conditions, prove that C = 2π/3 -/
theorem triangle_angle_c_value 
  (A B C : ℝ) 
  (triangle_sum : A + B + C = π)
  (m : ℝ × ℝ)
  (n : ℝ × ℝ)
  (m_def : m = (Real.sqrt 3 * Real.sin A, Real.sin B))
  (n_def : n = (Real.cos B, Real.sqrt 3 * Real.cos A))
  (dot_product : m.1 * n.1 + m.2 * n.2 = 1 + Real.cos (A + B)) :
  C = 2 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_value_l3969_396901


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3969_396953

open Real

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State the theorem
theorem solution_set_equivalence 
  (h1 : ∀ x, 3 * f x + f' x < 0)
  (h2 : f (log 2) = 1) :
  ∀ x, f x > 8 * exp (-3 * x) ↔ x < log 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3969_396953

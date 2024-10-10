import Mathlib

namespace difference_of_squares_l4000_400086

theorem difference_of_squares (a b : ℕ+) : 
  ∃ (x y z w : ℤ), (a : ℤ) = x^2 - y^2 ∨ (b : ℤ) = z^2 - w^2 ∨ ((a + b) : ℤ) = x^2 - y^2 := by
  sorry

end difference_of_squares_l4000_400086


namespace fraction_equality_l4000_400007

theorem fraction_equality (a b c d : ℝ) (h : a / b = c / d) :
  (a * b) / (c * d) = ((a + b) / (c + d))^2 ∧ (a * b) / (c * d) = ((a - b) / (c - d))^2 :=
by sorry

end fraction_equality_l4000_400007


namespace log_ten_seven_in_terms_of_p_q_l4000_400087

theorem log_ten_seven_in_terms_of_p_q (p q : ℝ) 
  (hp : Real.log 3 / Real.log 4 = p)
  (hq : Real.log 7 / Real.log 5 = q) :
  Real.log 7 / Real.log 10 = (2 * p * q + 2 * p) / (1 + 2 * p) := by
  sorry

end log_ten_seven_in_terms_of_p_q_l4000_400087


namespace fractional_equation_solution_range_l4000_400070

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ (x + m) / (x - 2) + (2 * m) / (2 - x) = 3) → 
  m < 6 ∧ m ≠ 2 := by
  sorry

end fractional_equation_solution_range_l4000_400070


namespace unique_line_through_point_l4000_400078

/-- A line in the xy-plane --/
structure Line where
  x_intercept : ℕ+
  y_intercept : ℕ+

/-- Checks if a natural number is prime --/
def isPrime (n : ℕ+) : Prop :=
  n > 1 ∧ ∀ m : ℕ+, m < n → m ∣ n → m = 1

/-- Checks if a line passes through the point (5,4) --/
def passesThrough (l : Line) : Prop :=
  5 / l.x_intercept.val + 4 / l.y_intercept.val = 1

/-- The main theorem --/
theorem unique_line_through_point :
  ∃! l : Line, passesThrough l ∧ isPrime l.y_intercept :=
sorry

end unique_line_through_point_l4000_400078


namespace smallest_c_for_negative_three_in_range_l4000_400081

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + c

-- State the theorem
theorem smallest_c_for_negative_three_in_range :
  (∃ (c : ℝ), ∀ (c' : ℝ), (∃ (x : ℝ), f c' x = -3) → c ≤ c') ∧
  (∃ (x : ℝ), f (-3/4) x = -3) :=
sorry

end smallest_c_for_negative_three_in_range_l4000_400081


namespace quadratic_inequality_range_l4000_400085

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 < 0) → (a < -2 ∨ a > 2) := by
sorry

end quadratic_inequality_range_l4000_400085


namespace sample_data_properties_l4000_400049

def median (s : Finset ℝ) : ℝ := sorry

theorem sample_data_properties (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (h : x₁ ≤ x₂ ∧ x₂ ≤ x₃ ∧ x₃ ≤ x₄ ∧ x₄ ≤ x₅ ∧ x₅ ≤ x₆) :
  let s₁ := {x₂, x₃, x₄, x₅}
  let s₂ := {x₁, x₂, x₃, x₄, x₅, x₆}
  (median s₁ = median s₂) ∧ 
  (x₅ - x₂ ≤ x₆ - x₁) := by
  sorry

end sample_data_properties_l4000_400049


namespace shaded_area_is_120_l4000_400064

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle with four vertices -/
structure Rectangle where
  p : Point
  r : Point
  s : Point
  v : Point

/-- Calculates the area of a rectangle -/
def rectangleArea (rect : Rectangle) : ℝ :=
  (rect.r.x - rect.p.x) * (rect.v.y - rect.p.y)

/-- Theorem: The shaded area in the given rectangle is 120 cm² -/
theorem shaded_area_is_120 (rect : Rectangle) 
  (h1 : rect.r.x - rect.p.x = 20) -- PR = 20 cm
  (h2 : rect.v.y - rect.p.y = 12) -- PV = 12 cm
  (u : Point) (t : Point) (q : Point)
  (h3 : u.x = rect.v.x ∧ u.y ≤ rect.v.y ∧ u.y ≥ rect.s.y) -- U is on VS
  (h4 : t.x = rect.v.x ∧ t.y ≤ rect.v.y ∧ t.y ≥ rect.s.y) -- T is on VS
  (h5 : q.y = rect.p.y ∧ q.x ≥ rect.p.x ∧ q.x ≤ rect.r.x) -- Q is on PR
  : rectangleArea rect - (rect.r.x - rect.p.x) * (rect.v.y - rect.p.y) / 2 = 120 :=
sorry

end shaded_area_is_120_l4000_400064


namespace spanish_books_count_l4000_400089

theorem spanish_books_count (total : ℕ) (english : ℕ) (french : ℕ) (italian : ℕ) (spanish : ℕ) :
  total = 280 ∧
  english = total / 5 ∧
  french = total / 7 ∧
  italian = total / 4 ∧
  spanish = total - (english + french + italian) →
  spanish = 114 := by
sorry

end spanish_books_count_l4000_400089


namespace garden_comparison_l4000_400019

-- Define the dimensions of the gardens
def chris_length : ℝ := 30
def chris_width : ℝ := 60
def jordan_length : ℝ := 35
def jordan_width : ℝ := 55

-- Define the area difference
def area_difference : ℝ := jordan_length * jordan_width - chris_length * chris_width

-- Define the perimeters
def chris_perimeter : ℝ := 2 * (chris_length + chris_width)
def jordan_perimeter : ℝ := 2 * (jordan_length + jordan_width)

-- Theorem statement
theorem garden_comparison :
  area_difference = 125 ∧ chris_perimeter = jordan_perimeter := by
  sorry

end garden_comparison_l4000_400019


namespace aquarium_count_l4000_400002

theorem aquarium_count (total_animals : ℕ) (animals_per_aquarium : ℕ) 
  (h1 : total_animals = 40) 
  (h2 : animals_per_aquarium = 2) :
  total_animals / animals_per_aquarium = 20 := by
  sorry

end aquarium_count_l4000_400002


namespace function_value_at_2017_l4000_400063

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, 3 * f ((a + 2 * b) / 3) = f a + 2 * f b) ∧
  f 1 = 1 ∧
  f 4 = 7

/-- The main theorem -/
theorem function_value_at_2017 (f : ℝ → ℝ) (h : special_function f) : f 2017 = 4033 := by
  sorry

end function_value_at_2017_l4000_400063


namespace investment_period_ratio_l4000_400053

/-- Represents the profit distribution in a joint business venture -/
structure JointBusiness where
  investment_a : ℚ
  investment_b : ℚ
  period_a : ℚ
  period_b : ℚ
  profit_b : ℚ
  total_profit : ℚ

/-- Theorem stating the ratio of investment periods given the conditions -/
theorem investment_period_ratio (jb : JointBusiness)
  (h1 : jb.investment_a = 3 * jb.investment_b)
  (h2 : ∃ k : ℚ, jb.period_a = k * jb.period_b)
  (h3 : jb.profit_b = 4000)
  (h4 : jb.total_profit = 28000) :
  jb.period_a / jb.period_b = 2 := by
  sorry

#check investment_period_ratio

end investment_period_ratio_l4000_400053


namespace total_lives_calculation_l4000_400055

/-- Given 7 initial friends, 2 additional players, and 7 lives per player,
    the total number of lives for all players is 63. -/
theorem total_lives_calculation (initial_friends : ℕ) (additional_players : ℕ) (lives_per_player : ℕ)
    (h1 : initial_friends = 7)
    (h2 : additional_players = 2)
    (h3 : lives_per_player = 7) :
    (initial_friends + additional_players) * lives_per_player = 63 := by
  sorry

end total_lives_calculation_l4000_400055


namespace binomial_divisibility_implies_prime_l4000_400000

theorem binomial_divisibility_implies_prime (n : ℕ) (h : ∀ k : ℕ, 1 ≤ k → k < n → (n.choose k) % n = 0) : Nat.Prime n := by
  sorry

end binomial_divisibility_implies_prime_l4000_400000


namespace line_parameterization_l4000_400022

/-- Given a line y = 2x - 10 parameterized by (x, y) = (g(t), 20t - 8), 
    prove that g(t) = 10t + 1 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t x y, y = 2*x - 10 ∧ x = g t ∧ y = 20*t - 8) → 
  (∀ t, g t = 10*t + 1) :=
by sorry

end line_parameterization_l4000_400022


namespace birch_planting_l4000_400004

theorem birch_planting (total_students : ℕ) (roses_per_girl : ℕ) (total_plants : ℕ) (total_birches : ℕ)
  (h1 : total_students = 24)
  (h2 : roses_per_girl = 3)
  (h3 : total_plants = 24)
  (h4 : total_birches = 6) :
  (total_students - (total_plants - total_birches) / roses_per_girl) / 3 = total_birches :=
by sorry

end birch_planting_l4000_400004


namespace derivative_of_odd_function_is_even_l4000_400017

theorem derivative_of_odd_function_is_even 
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_odd : ∀ x, f (-x) = -f x) : 
  ∀ x, (deriv f) (-x) = (deriv f) x := by
  sorry

end derivative_of_odd_function_is_even_l4000_400017


namespace range_of_k_line_equation_when_OB_2OA_l4000_400084

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 6)^2 + y^2 = 20

-- Define the line l
def line_l (k x y : ℝ) : Prop := y = k * x

-- Define the condition that line l intersects circle C at two distinct points
def intersects_at_two_points (k : ℝ) : Prop := 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the condition OB = 2OA
def OB_equals_2OA (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
    x₂ = 2 * x₁ ∧ y₂ = 2 * y₁

-- Theorem for the range of k
theorem range_of_k (k : ℝ) : intersects_at_two_points k → -Real.sqrt 5 / 2 < k ∧ k < Real.sqrt 5 / 2 := 
  sorry

-- Theorem for the equation of line l when OB = 2OA
theorem line_equation_when_OB_2OA (k : ℝ) : OB_equals_2OA k → k = 1 ∨ k = -1 :=
  sorry

end range_of_k_line_equation_when_OB_2OA_l4000_400084


namespace negative_integer_problem_l4000_400097

theorem negative_integer_problem (n : ℤ) : 
  n < 0 → n * (-8) + 5 = 93 → n = -11 := by
  sorry

end negative_integer_problem_l4000_400097


namespace race_parts_length_l4000_400092

/-- Given a race with 4 parts, prove that the length of each of the second and third parts is 21.5 km -/
theorem race_parts_length 
  (total_length : ℝ) 
  (first_part : ℝ) 
  (last_part : ℝ) 
  (h1 : total_length = 74.5)
  (h2 : first_part = 15.5)
  (h3 : last_part = 16)
  (h4 : ∃ (second_part third_part : ℝ), 
        second_part = third_part ∧ 
        total_length = first_part + second_part + third_part + last_part) :
  ∃ (second_part : ℝ), second_part = 21.5 ∧ 
    total_length = first_part + second_part + second_part + last_part :=
by
  sorry

end race_parts_length_l4000_400092


namespace days_missed_difference_l4000_400008

/-- Represents the frequency histogram of days missed --/
structure FrequencyHistogram :=
  (days : List Nat)
  (frequencies : List Nat)
  (total_students : Nat)

/-- Calculate the median of the dataset --/
def median (h : FrequencyHistogram) : Rat :=
  sorry

/-- Calculate the mean of the dataset --/
def mean (h : FrequencyHistogram) : Rat :=
  sorry

/-- The main theorem --/
theorem days_missed_difference (h : FrequencyHistogram) 
  (h_days : h.days = [0, 1, 2, 3, 4, 5])
  (h_frequencies : h.frequencies = [4, 3, 6, 2, 3, 2])
  (h_total : h.total_students = 20) :
  mean h - median h = 3 / 20 := by
  sorry

end days_missed_difference_l4000_400008


namespace interest_rate_difference_l4000_400016

/-- Given a principal amount, time period, and difference in interest earned between two simple interest rates, 
    this theorem proves that the difference between these rates is 5%. -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_diff : ℝ)
  (h1 : principal = 600)
  (h2 : time = 10)
  (h3 : interest_diff = 300) :
  let rate_diff := interest_diff / (principal * time / 100)
  rate_diff = 5 := by sorry

end interest_rate_difference_l4000_400016


namespace complex_sum_to_polar_l4000_400066

theorem complex_sum_to_polar : 15 * Complex.exp (Complex.I * Real.pi / 6) + 15 * Complex.exp (Complex.I * 5 * Real.pi / 6) = 15 * Complex.exp (Complex.I * Real.pi / 2) := by
  sorry

end complex_sum_to_polar_l4000_400066


namespace birthday_cake_icing_l4000_400042

/-- Represents a rectangular cake with given dimensions -/
structure Cake where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a smaller cuboid piece of the cake -/
structure Piece where
  size : ℕ

/-- Calculates the number of pieces with icing on exactly two sides -/
def pieces_with_two_sided_icing (c : Cake) (p : Piece) : ℕ :=
  sorry

/-- Theorem stating that a 6 × 4 × 4 cake cut into 2 × 2 × 2 pieces has 16 pieces with icing on two sides -/
theorem birthday_cake_icing (c : Cake) (p : Piece) :
  c.length = 6 ∧ c.width = 4 ∧ c.height = 4 ∧ p.size = 2 →
  pieces_with_two_sided_icing c p = 16 :=
by sorry

end birthday_cake_icing_l4000_400042


namespace minimum_pages_required_l4000_400043

-- Define the types of cards and pages
inductive CardType
| Rare
| LimitedEdition
| Regular

inductive PageType
| NineCard
| SevenCard
| FiveCard

-- Define the card counts
def rareCardCount : Nat := 18
def limitedEditionCardCount : Nat := 21
def regularCardCount : Nat := 45

-- Define the page capacities
def pageCapacity (pt : PageType) : Nat :=
  match pt with
  | PageType.NineCard => 9
  | PageType.SevenCard => 7
  | PageType.FiveCard => 5

-- Define a function to check if a page type is valid for a card type
def isValidPageType (ct : CardType) (pt : PageType) : Bool :=
  match ct, pt with
  | CardType.Rare, PageType.NineCard => true
  | CardType.Rare, PageType.SevenCard => true
  | CardType.LimitedEdition, PageType.NineCard => true
  | CardType.LimitedEdition, PageType.SevenCard => true
  | CardType.Regular, _ => true
  | _, _ => false

-- Define the theorem
theorem minimum_pages_required :
  ∃ (rarePages limitedPages regularPages : Nat),
    rarePages * pageCapacity PageType.NineCard = rareCardCount ∧
    limitedPages * pageCapacity PageType.SevenCard = limitedEditionCardCount ∧
    regularPages * pageCapacity PageType.NineCard = regularCardCount ∧
    rarePages + limitedPages + regularPages = 10 ∧
    (∀ (rp lp regalp : Nat),
      rp * pageCapacity PageType.NineCard ≥ rareCardCount →
      lp * pageCapacity PageType.SevenCard ≥ limitedEditionCardCount →
      regalp * pageCapacity PageType.NineCard ≥ regularCardCount →
      isValidPageType CardType.Rare PageType.NineCard →
      isValidPageType CardType.LimitedEdition PageType.SevenCard →
      isValidPageType CardType.Regular PageType.NineCard →
      rp + lp + regalp ≥ 10) :=
by sorry

end minimum_pages_required_l4000_400043


namespace stratified_sampling_l4000_400009

/-- Stratified sampling problem -/
theorem stratified_sampling 
  (total_items : ℕ) 
  (sample_size : ℕ) 
  (stratum_A_size : ℕ) 
  (h1 : total_items = 600) 
  (h2 : sample_size = 100) 
  (h3 : stratum_A_size = 150) :
  let items_from_A := (sample_size * stratum_A_size) / total_items
  let prob_item_A := sample_size / total_items
  (items_from_A = 25) ∧ (prob_item_A = 1 / 6) := by
sorry

end stratified_sampling_l4000_400009


namespace quadratic_equation_solution_l4000_400030

theorem quadratic_equation_solution (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 5 * x - m = 0) → m ≥ -25/8 :=
by sorry

end quadratic_equation_solution_l4000_400030


namespace factor_expression_l4000_400071

theorem factor_expression (b : ℝ) : 280 * b^2 + 56 * b = 56 * b * (5 * b + 1) := by
  sorry

end factor_expression_l4000_400071


namespace base4_to_decimal_equality_l4000_400099

/-- Converts a base 4 number represented as a list of digits to its decimal (base 10) equivalent. -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

/-- The base 4 representation of the number we want to convert. -/
def base4Number : List Nat := [3, 0, 1, 2, 1]

/-- Theorem stating that the base 4 number 30121₄ is equal to 793 in base 10. -/
theorem base4_to_decimal_equality :
  base4ToDecimal base4Number = 793 := by
  sorry

end base4_to_decimal_equality_l4000_400099


namespace kim_morning_routine_time_l4000_400059

/-- Represents Kim's morning routine and calculates the total time taken. -/
def morning_routine_time (total_employees : ℕ) (senior_employees : ℕ) (overtime_employees : ℕ)
  (coffee_time : ℕ) (regular_status_time : ℕ) (senior_status_extra_time : ℕ)
  (overtime_payroll_time : ℕ) (regular_payroll_time : ℕ)
  (email_time : ℕ) (task_allocation_time : ℕ) : ℕ :=
  let regular_employees := total_employees - senior_employees
  let non_overtime_employees := total_employees - overtime_employees
  coffee_time +
  (regular_employees * regular_status_time) +
  (senior_employees * (regular_status_time + senior_status_extra_time)) +
  (overtime_employees * overtime_payroll_time) +
  (non_overtime_employees * regular_payroll_time) +
  email_time +
  task_allocation_time

/-- Theorem stating that Kim's morning routine takes 60 minutes given the specified conditions. -/
theorem kim_morning_routine_time :
  morning_routine_time 9 3 4 5 2 1 3 1 10 7 = 60 := by
  sorry

end kim_morning_routine_time_l4000_400059


namespace sum_58_46_rounded_to_hundred_l4000_400020

def round_to_nearest_hundred (n : ℤ) : ℤ :=
  (n + 50) / 100 * 100

theorem sum_58_46_rounded_to_hundred : 
  round_to_nearest_hundred (58 + 46) = 100 := by
  sorry

end sum_58_46_rounded_to_hundred_l4000_400020


namespace sin_cos_difference_equals_neg_sqrt3_half_l4000_400090

theorem sin_cos_difference_equals_neg_sqrt3_half :
  Real.sin (5 * π / 180) * Real.sin (25 * π / 180) - 
  Real.sin (95 * π / 180) * Real.sin (65 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_cos_difference_equals_neg_sqrt3_half_l4000_400090


namespace nth_letter_is_c_l4000_400014

def repeating_pattern : ℕ → Char
  | n => match n % 3 with
    | 0 => 'C'
    | 1 => 'A'
    | _ => 'B'

theorem nth_letter_is_c (n : ℕ) (h : n = 150) : repeating_pattern n = 'C' := by
  sorry

end nth_letter_is_c_l4000_400014


namespace find_b_value_l4000_400093

-- Define the inverse relationship between a^3 and √b
def inverse_relation (a b : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a^3 * Real.sqrt b = k

-- State the theorem
theorem find_b_value (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : inverse_relation a₁ b₁)
  (h₂ : a₁ = 3 ∧ b₁ = 64)
  (h₃ : inverse_relation a₂ b₂)
  (h₄ : a₂ = 4)
  (h₅ : a₂ * Real.sqrt b₂ = 24) :
  b₂ = 11.390625 := by
sorry

end find_b_value_l4000_400093


namespace alex_score_l4000_400096

/-- Represents the number of shots attempted for each type --/
structure ShotAttempts where
  free_throws : ℕ
  three_points : ℕ
  two_points : ℕ

/-- Calculates the total points scored given the shot attempts --/
def calculate_score (attempts : ShotAttempts) : ℕ :=
  (attempts.free_throws * 8 / 10) +
  (attempts.three_points * 3 * 1 / 10) +
  (attempts.two_points * 2 * 5 / 10)

theorem alex_score :
  ∃ (attempts : ShotAttempts),
    attempts.free_throws + attempts.three_points + attempts.two_points = 40 ∧
    calculate_score attempts = 28 := by
  sorry

end alex_score_l4000_400096


namespace refrigerator_profit_theorem_l4000_400094

def refrigerator_profit (cost_price marked_price : ℝ) : Prop :=
  let profit_20_off := 0.8 * marked_price - cost_price
  let profit_margin := profit_20_off / cost_price
  profit_20_off = 200 ∧
  profit_margin = 0.1 ∧
  0.85 * marked_price - cost_price = 337.5

theorem refrigerator_profit_theorem :
  ∃ (cost_price marked_price : ℝ),
    refrigerator_profit cost_price marked_price :=
  sorry

end refrigerator_profit_theorem_l4000_400094


namespace max_triangle_perimeter_l4000_400005

/-- Given a triangle with two sides of length 8 and 15 units, and the third side
    length x being an integer, the maximum perimeter of the triangle is 45 units. -/
theorem max_triangle_perimeter :
  ∀ x : ℤ,
  (8 : ℝ) + 15 > (x : ℝ) →
  (8 : ℝ) + (x : ℝ) > 15 →
  (15 : ℝ) + (x : ℝ) > 8 →
  (∀ y : ℤ, (8 : ℝ) + 15 > (y : ℝ) →
             (8 : ℝ) + (y : ℝ) > 15 →
             (15 : ℝ) + (y : ℝ) > 8 →
             8 + 15 + (x : ℝ) ≥ 8 + 15 + (y : ℝ)) →
  8 + 15 + (x : ℝ) = 45 :=
by sorry

end max_triangle_perimeter_l4000_400005


namespace income_mean_difference_l4000_400050

theorem income_mean_difference (T : ℝ) (n : ℕ) : 
  n = 500 → 
  (T + 1100000) / n - (T + 110000) / n = 1980 :=
by sorry

end income_mean_difference_l4000_400050


namespace g_geq_one_l4000_400098

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 + Real.log x + 4

-- Define the function g
def g (x : ℝ) : ℝ := Real.exp (x - 1) + 3 * x^2 + 4 - f x

-- Theorem statement
theorem g_geq_one (x : ℝ) (h : x > 0) : g x ≥ 1 := by
  sorry

end

end g_geq_one_l4000_400098


namespace square_plus_one_representation_l4000_400044

theorem square_plus_one_representation (x y z : ℕ+) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
sorry

end square_plus_one_representation_l4000_400044


namespace apple_boxes_count_l4000_400065

def apples_per_crate : ℕ := 250
def number_of_crates : ℕ := 20
def rotten_apples : ℕ := 320
def apples_per_box : ℕ := 25

theorem apple_boxes_count :
  (apples_per_crate * number_of_crates - rotten_apples) / apples_per_box = 187 := by
  sorry

end apple_boxes_count_l4000_400065


namespace system_solution_l4000_400027

theorem system_solution (x y : ℝ) 
  (eq1 : x^2 - 4 * Real.sqrt (3*x - 2) + 6 = y)
  (eq2 : y^2 - 4 * Real.sqrt (3*y - 2) + 6 = x)
  (domain_x : 3*x - 2 ≥ 0)
  (domain_y : 3*y - 2 ≥ 0) :
  x = 2 ∧ y = 2 :=
by sorry

end system_solution_l4000_400027


namespace simultaneous_cycle_is_twenty_l4000_400025

/-- The length of the letter sequence -/
def letter_cycle_length : ℕ := 5

/-- The length of the digit sequence -/
def digit_cycle_length : ℕ := 4

/-- The number of cycles needed for both sequences to return to their original state simultaneously -/
def simultaneous_cycle : ℕ := Nat.lcm letter_cycle_length digit_cycle_length

theorem simultaneous_cycle_is_twenty : simultaneous_cycle = 20 := by
  sorry

end simultaneous_cycle_is_twenty_l4000_400025


namespace proposition_implication_l4000_400029

theorem proposition_implication (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (¬p ∧ q) ∨ (¬p ∧ ¬q) :=
sorry

end proposition_implication_l4000_400029


namespace square_side_length_l4000_400011

/-- The radius of the circles -/
def r : ℝ := 1000

/-- The side length of the square -/
def square_side : ℝ := 400

/-- Two circles touch each other and a horizontal line is tangent to both circles -/
axiom circles_touch_and_tangent_to_line : True

/-- A square fits snugly between the horizontal line and the two circles -/
axiom square_fits_snugly : True

/-- The theorem stating that the side length of the square is 400 -/
theorem square_side_length : 
  square_side = 400 :=
sorry

end square_side_length_l4000_400011


namespace broken_lines_count_l4000_400039

/-- The number of paths on a grid with 2n steps, n horizontal and n vertical -/
def grid_paths (n : ℕ) : ℕ := (Nat.choose (2 * n) n) ^ 2

/-- Theorem stating that the number of broken lines of length 2n on a grid
    with cell side length 1 and vertices at intersections is (C_{2n}^{n})^2 -/
theorem broken_lines_count (n : ℕ) :
  grid_paths n = (Nat.choose (2 * n) n) ^ 2 := by
  sorry

end broken_lines_count_l4000_400039


namespace kho_kho_players_l4000_400056

theorem kho_kho_players (total : ℕ) (kabadi : ℕ) (both : ℕ) (kho_kho_only : ℕ) :
  total = 50 →
  kabadi = 10 →
  both = 5 →
  total = (kabadi - both) + kho_kho_only + both →
  kho_kho_only = 40 := by
sorry

end kho_kho_players_l4000_400056


namespace classroom_pictures_l4000_400082

/-- The number of oil paintings on the walls of the classroom -/
def oil_paintings : ℕ := 9

/-- The number of watercolor paintings on the walls of the classroom -/
def watercolor_paintings : ℕ := 7

/-- The total number of pictures on the walls of the classroom -/
def total_pictures : ℕ := oil_paintings + watercolor_paintings

theorem classroom_pictures : total_pictures = 16 := by
  sorry

end classroom_pictures_l4000_400082


namespace first_person_speed_l4000_400045

/-- Two persons walk in opposite directions for a given time, ending up at a specific distance apart. -/
def opposite_walk (x : ℝ) (time : ℝ) (distance : ℝ) : Prop :=
  (x + 7) * time = distance

/-- The theorem states that given the conditions of the problem, the speed of the first person is 6 km/hr. -/
theorem first_person_speed : ∃ x : ℝ, opposite_walk x 3.5 45.5 ∧ x = 6 := by
  sorry

end first_person_speed_l4000_400045


namespace parallelogram_angle_difference_parallelogram_angle_difference_proof_l4000_400048

/-- 
In a parallelogram with a smaller angle of 55 degrees, 
the difference between the larger and smaller angles is 70 degrees.
-/
theorem parallelogram_angle_difference : ℝ → Prop :=
  fun smaller_angle : ℝ =>
    smaller_angle = 55 →
    ∃ larger_angle : ℝ,
      smaller_angle + larger_angle = 180 ∧
      larger_angle - smaller_angle = 70

-- The proof is omitted
theorem parallelogram_angle_difference_proof : 
  parallelogram_angle_difference 55 := by sorry

end parallelogram_angle_difference_parallelogram_angle_difference_proof_l4000_400048


namespace no_solution_absolute_value_equation_l4000_400015

theorem no_solution_absolute_value_equation :
  ¬ ∃ x : ℝ, 3 * |x + 2| + 2 = 0 := by
sorry

end no_solution_absolute_value_equation_l4000_400015


namespace average_age_problem_l4000_400058

theorem average_age_problem (devin_age eden_age mom_age : ℕ) : 
  devin_age = 12 →
  eden_age = 2 * devin_age →
  mom_age = 2 * eden_age →
  (devin_age + eden_age + mom_age) / 3 = 28 := by
  sorry

end average_age_problem_l4000_400058


namespace sufficient_not_necessary_condition_l4000_400046

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y)

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (a ≥ 2 → monotonic_on (f a) 1 2) ∧ 
  ¬(monotonic_on (f a) 1 2 → a ≥ 2) :=
sorry

end sufficient_not_necessary_condition_l4000_400046


namespace binomial_20_19_l4000_400095

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by sorry

end binomial_20_19_l4000_400095


namespace correct_quadratic_equation_l4000_400079

theorem correct_quadratic_equation 
  (a b c : ℝ) 
  (h1 : ∃ c', (a * 7^2 + b * 7 + c' = 0) ∧ (a * 3^2 + b * 3 + c' = 0))
  (h2 : ∃ b', (a * (-12)^2 + b' * (-12) + c = 0) ∧ (a * 3^2 + b' * 3 + c = 0)) :
  a = 1 ∧ b = -10 ∧ c = -36 := by
sorry

end correct_quadratic_equation_l4000_400079


namespace reading_days_l4000_400047

-- Define the reading speed in words per hour
def reading_speed : ℕ := 100

-- Define the number of words in each book
def book1_words : ℕ := 200
def book2_words : ℕ := 400
def book3_words : ℕ := 300

-- Define the average reading time per day in minutes
def avg_reading_time : ℕ := 54

-- Define the total number of words
def total_words : ℕ := book1_words + book2_words + book3_words

-- Theorem to prove
theorem reading_days : 
  (total_words / reading_speed : ℚ) / (avg_reading_time / 60 : ℚ) = 10 := by
  sorry


end reading_days_l4000_400047


namespace complex_number_quadrant_l4000_400018

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (2 - Complex.I) ∧ Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end complex_number_quadrant_l4000_400018


namespace banyan_tree_area_l4000_400038

theorem banyan_tree_area (C : Real) (h : C = 6.28) :
  let r := C / (2 * Real.pi)
  let S := Real.pi * r^2
  S = Real.pi := by
sorry

end banyan_tree_area_l4000_400038


namespace two_solutions_iff_a_gt_neg_one_l4000_400041

/-- The equation has exactly two solutions if and only if a > -1 -/
theorem two_solutions_iff_a_gt_neg_one (a : ℝ) :
  (∃! x y, x ≠ y ∧ x^2 + 2*x + 2*|x+1| = a ∧ y^2 + 2*y + 2*|y+1| = a) ↔ a > -1 :=
sorry

end two_solutions_iff_a_gt_neg_one_l4000_400041


namespace inequality_proof_l4000_400001

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2) / (2 * a^5 * b^5) + 81 * a^2 * b^2 / 4 + 9 * a * b > 18 := by
  sorry

end inequality_proof_l4000_400001


namespace total_tulips_is_308_l4000_400024

/-- The number of tulips needed for Anna's smiley face design --/
def total_tulips : ℕ :=
  let red_eye := 8
  let purple_eyebrow := 5
  let red_nose := 12
  let red_smile := 18
  let yellow_background := 9 * red_smile
  let purple_eyebrows := 4 * (2 * red_eye)
  let yellow_nose := 3 * red_nose
  
  let total_red := 2 * red_eye + red_nose + red_smile
  let total_purple := 2 * purple_eyebrow + (purple_eyebrows - 2 * purple_eyebrow)
  let total_yellow := yellow_background + yellow_nose
  
  total_red + total_purple + total_yellow

theorem total_tulips_is_308 : total_tulips = 308 := by
  sorry

end total_tulips_is_308_l4000_400024


namespace solve_john_age_problem_l4000_400075

def john_age_problem (john_current_age : ℕ) (sister_age_multiplier : ℕ) (sister_future_age : ℕ) : Prop :=
  let sister_current_age := john_current_age * sister_age_multiplier
  let age_difference := sister_current_age - john_current_age
  let john_future_age := sister_future_age - age_difference
  john_future_age = sister_future_age - age_difference

theorem solve_john_age_problem :
  john_age_problem 10 2 60 = true :=
sorry

end solve_john_age_problem_l4000_400075


namespace root_sum_reciprocals_l4000_400033

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0) →
  (q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0) →
  (r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0) →
  (s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0) →
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3) :=
by sorry

end root_sum_reciprocals_l4000_400033


namespace right_triangle_sides_l4000_400083

theorem right_triangle_sides (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  (a + b + c) / 2 - a = 2/3 * r →  -- Relation derived from circle touching sides
  c = 5/3 * r →  -- Hypotenuse relation
  a * b / 2 = 2 * r →  -- Area of the triangle
  (a = 4/3 * r ∧ b = r) ∨ (a = r ∧ b = 4/3 * r) := by
sorry

end right_triangle_sides_l4000_400083


namespace xy_value_l4000_400021

theorem xy_value (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 44) : x*y = -24 := by
  sorry

end xy_value_l4000_400021


namespace num_triangles_on_circle_l4000_400023

/-- The number of ways to choose k elements from n elements --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct points on the circle --/
def num_points : ℕ := 10

/-- The number of points needed to form a triangle --/
def points_per_triangle : ℕ := 3

/-- Theorem: The number of different triangles that can be formed
    by choosing 3 points from 10 distinct points on a circle's circumference
    is equal to 120 --/
theorem num_triangles_on_circle :
  choose num_points points_per_triangle = 120 := by sorry

end num_triangles_on_circle_l4000_400023


namespace floor_x_floor_x_eq_48_l4000_400088

open Real

theorem floor_x_floor_x_eq_48 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 48 ↔ 8 ≤ x ∧ x < 49 / 6 := by
  sorry

end floor_x_floor_x_eq_48_l4000_400088


namespace factorization_difference_of_squares_factorization_quadratic_l4000_400062

-- Problem 1
theorem factorization_difference_of_squares (x y : ℝ) :
  x^2 - 4*y^2 = (x + 2*y) * (x - 2*y) := by sorry

-- Problem 2
theorem factorization_quadratic (a x : ℝ) :
  3*a*x^2 - 6*a*x + 3*a = 3*a*(x - 1)^2 := by sorry

end factorization_difference_of_squares_factorization_quadratic_l4000_400062


namespace garden_sprinkler_morning_usage_garden_sprinkler_conditions_l4000_400060

/-- A sprinkler system that waters a desert garden twice daily -/
structure SprinklerSystem where
  morning_usage : ℝ
  evening_usage : ℝ
  days : ℕ
  total_usage : ℝ

/-- The specific sprinkler system described in the problem -/
def garden_sprinkler : SprinklerSystem where
  morning_usage := 4  -- This is what we want to prove
  evening_usage := 6
  days := 5
  total_usage := 50

/-- Theorem stating that the morning usage of the garden sprinkler is 4 liters -/
theorem garden_sprinkler_morning_usage :
  garden_sprinkler.morning_usage = 4 :=
by sorry

/-- Theorem proving that the given conditions are satisfied by the garden sprinkler -/
theorem garden_sprinkler_conditions :
  garden_sprinkler.evening_usage = 6 ∧
  garden_sprinkler.days = 5 ∧
  garden_sprinkler.total_usage = 50 ∧
  garden_sprinkler.days * (garden_sprinkler.morning_usage + garden_sprinkler.evening_usage) = garden_sprinkler.total_usage :=
by sorry

end garden_sprinkler_morning_usage_garden_sprinkler_conditions_l4000_400060


namespace f_zero_values_l4000_400076

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x * f y

/-- Theorem stating that f(0) is either 0 or 1 for functions satisfying the functional equation -/
theorem f_zero_values (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f 0 = 0 ∨ f 0 = 1 := by
  sorry

end f_zero_values_l4000_400076


namespace instantaneous_velocity_at_2_l4000_400028

-- Define the displacement function
def h (t : ℝ) : ℝ := 14 * t - t^2

-- Define the instantaneous velocity function (derivative of h)
def v (t : ℝ) : ℝ := 14 - 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_2 : v 2 = 10 := by
  sorry

end instantaneous_velocity_at_2_l4000_400028


namespace new_basis_from_old_l4000_400054

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def is_basis (v₁ v₂ v₃ : V) : Prop :=
  LinearIndependent ℝ ![v₁, v₂, v₃] ∧ Submodule.span ℝ {v₁, v₂, v₃} = ⊤

theorem new_basis_from_old (a b c p q : V) 
  (h₁ : is_basis a b c)
  (h₂ : p = a + b)
  (h₃ : q = a - b) :
  is_basis p q (a + 2 • c) := by
  sorry

end new_basis_from_old_l4000_400054


namespace square_sum_reciprocal_l4000_400037

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end square_sum_reciprocal_l4000_400037


namespace no_such_function_exists_l4000_400069

theorem no_such_function_exists :
  ¬∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f (f n) = n + 1987 := by
  sorry

end no_such_function_exists_l4000_400069


namespace smallest_n_congruence_l4000_400026

theorem smallest_n_congruence : ∃ n : ℕ+, (∀ m : ℕ+, 813 * m ≡ 1224 * m [ZMOD 30] → n ≤ m) ∧ 813 * n ≡ 1224 * n [ZMOD 30] := by
  sorry

end smallest_n_congruence_l4000_400026


namespace union_of_A_and_B_l4000_400068

open Set

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Icc (-1) 4 := by sorry

end union_of_A_and_B_l4000_400068


namespace custom_op_result_l4000_400012

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

-- Theorem statement
theorem custom_op_result :
  let y : ℤ := 11
  customOp y 10 = 90 := by
  sorry

end custom_op_result_l4000_400012


namespace probability_selecting_two_types_l4000_400006

theorem probability_selecting_two_types (total : ℕ) (type_c : ℕ) (type_r : ℕ) (type_a : ℕ) :
  total = type_c + type_r + type_a →
  type_c = type_r →
  type_a = 1 →
  (type_c : ℚ) * type_r / (total * (total - 1)) = 5 / 11 :=
by sorry

end probability_selecting_two_types_l4000_400006


namespace log10_graph_property_l4000_400074

-- Define the logarithm function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the condition for a point to be on the graph of y = log₁₀ x
def on_log10_graph (p : ℝ × ℝ) : Prop :=
  p.2 = log10 p.1

-- State the theorem
theorem log10_graph_property (a b : ℝ) (h1 : on_log10_graph (a, b)) (h2 : a ≠ 1) :
  on_log10_graph (a^2, 2*b) :=
sorry

end log10_graph_property_l4000_400074


namespace pie_shop_revenue_calculation_l4000_400003

/-- Represents the revenue calculation for a pie shop --/
def pie_shop_revenue (apple_price blueberry_price cherry_price : ℕ) 
                     (slices_per_pie : ℕ) 
                     (apple_pies blueberry_pies cherry_pies : ℕ) : ℕ :=
  (apple_price * slices_per_pie * apple_pies) + 
  (blueberry_price * slices_per_pie * blueberry_pies) + 
  (cherry_price * slices_per_pie * cherry_pies)

/-- Theorem stating the revenue of the pie shop --/
theorem pie_shop_revenue_calculation : 
  pie_shop_revenue 5 6 7 6 12 8 10 = 1068 := by
  sorry

end pie_shop_revenue_calculation_l4000_400003


namespace max_roses_for_680_l4000_400073

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual : ℚ  -- Price of an individual rose
  dozen : ℚ       -- Price of a dozen roses
  twoDozen : ℚ    -- Price of two dozen roses

/-- Calculates the maximum number of roses that can be purchased given a budget and pricing options -/
def maxRoses (budget : ℚ) (pricing : RosePricing) : ℕ :=
  sorry

/-- The specific pricing for the problem -/
def problemPricing : RosePricing :=
  { individual := 730/100,  -- $7.30
    dozen := 36,            -- $36
    twoDozen := 50 }        -- $50

theorem max_roses_for_680 :
  maxRoses 680 problemPricing = 316 :=
sorry

end max_roses_for_680_l4000_400073


namespace problem_solution_l4000_400036

theorem problem_solution (t : ℚ) :
  let x := 3 - 2 * t
  let y := 5 * t + 6
  x = 0 → y = 27 / 2 := by
sorry

end problem_solution_l4000_400036


namespace total_students_l4000_400031

theorem total_students (passed_first : ℕ) (passed_second : ℕ) (passed_both : ℕ) (failed_both : ℕ) 
  (h1 : passed_first = 60)
  (h2 : passed_second = 40)
  (h3 : passed_both = 20)
  (h4 : failed_both = 20) :
  passed_first + passed_second - passed_both + failed_both = 100 := by
  sorry

#check total_students

end total_students_l4000_400031


namespace black_squares_in_37th_row_l4000_400080

/-- Represents the number of squares in the nth row of a stair-step figure -/
def num_squares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of black squares in the nth row of a stair-step figure -/
def num_black_squares (n : ℕ) : ℕ := (num_squares n - 1) / 2

theorem black_squares_in_37th_row :
  num_black_squares 37 = 36 := by sorry

end black_squares_in_37th_row_l4000_400080


namespace number_operations_l4000_400061

theorem number_operations (x : ℝ) : ((x - 2 + 3) * 2) / 3 = 6 ↔ x = 8 := by
  sorry

end number_operations_l4000_400061


namespace grade_assignments_12_4_l4000_400013

/-- The number of ways to assign grades to students -/
def gradeAssignments (numStudents : ℕ) (numGrades : ℕ) : ℕ :=
  numGrades ^ numStudents

/-- Theorem: The number of ways to assign 4 possible grades to 12 students is 16777216 -/
theorem grade_assignments_12_4 : gradeAssignments 12 4 = 16777216 := by
  sorry

end grade_assignments_12_4_l4000_400013


namespace domain_of_sqrt_2cos_plus_1_l4000_400032

open Real

theorem domain_of_sqrt_2cos_plus_1 (x : ℝ) (k : ℤ) :
  (∃ y : ℝ, y = Real.sqrt (2 * Real.cos x + 1)) ↔ 
  (x ∈ Set.Icc (2 * Real.pi * k - 2 * Real.pi / 3) (2 * Real.pi * k + 2 * Real.pi / 3)) :=
sorry

end domain_of_sqrt_2cos_plus_1_l4000_400032


namespace min_c_value_l4000_400091

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hab : a < b) (hbc : b < c) 
  (h_unique : ∃! (x y : ℝ), 2 * x + y = 3005 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 3003 ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < 3003 ∧
    ∃! (x y : ℝ), 2 * x + y = 3005 ∧ y = |x - a'| + |x - b'| + |x - 3003| :=
by sorry

end min_c_value_l4000_400091


namespace mets_fan_count_l4000_400052

/-- Represents the number of fans for each team -/
structure FanCount where
  yankees : ℕ
  mets : ℕ
  redsox : ℕ

/-- The conditions of the problem -/
def fan_conditions (fc : FanCount) : Prop :=
  -- Ratio of Yankees to Mets fans is 3:2
  3 * fc.mets = 2 * fc.yankees ∧
  -- Ratio of Mets to Red Sox fans is 4:5
  4 * fc.redsox = 5 * fc.mets ∧
  -- Total number of fans is 360
  fc.yankees + fc.mets + fc.redsox = 360

/-- The theorem stating that under the given conditions, there are 96 Mets fans -/
theorem mets_fan_count (fc : FanCount) : fan_conditions fc → fc.mets = 96 := by
  sorry

end mets_fan_count_l4000_400052


namespace sqrt_sum_simplification_l4000_400010

theorem sqrt_sum_simplification : Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 2 * Real.sqrt 3 := by
  sorry

end sqrt_sum_simplification_l4000_400010


namespace impossibility_of_broken_line_l4000_400077

/-- Represents a segment in the figure -/
structure Segment where
  id : Nat

/-- Represents a region in the figure -/
structure Region where
  segments : Finset Segment

/-- Represents the entire figure -/
structure Figure where
  segments : Finset Segment
  regions : Finset Region

/-- A broken line (polygonal chain) -/
structure BrokenLine where
  intersections : Finset Segment

/-- The theorem statement -/
theorem impossibility_of_broken_line (fig : Figure) 
  (h1 : fig.segments.card = 16)
  (h2 : ∃ r1 r2 r3 : Region, r1 ∈ fig.regions ∧ r2 ∈ fig.regions ∧ r3 ∈ fig.regions ∧ 
        r1.segments.card = 5 ∧ r2.segments.card = 5 ∧ r3.segments.card = 5) :
  ¬∃ (bl : BrokenLine), bl.intersections = fig.segments :=
sorry

end impossibility_of_broken_line_l4000_400077


namespace trapezoid_height_l4000_400035

/-- A trapezoid with given area and sum of diagonals has a specific height -/
theorem trapezoid_height (area : ℝ) (sum_diagonals : ℝ) (height : ℝ) :
  area = 2 →
  sum_diagonals = 4 →
  height = Real.sqrt 2 :=
by sorry

end trapezoid_height_l4000_400035


namespace binary_ones_factorial_divisibility_l4000_400072

-- Define a function to count the number of ones in the binary representation of a natural number
def countOnes (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem binary_ones_factorial_divisibility (n : ℕ) (h : n > 0) (h_ones : countOnes n = 1995) :
  (2^(n - 1995) : ℕ) ∣ n! :=
sorry

end binary_ones_factorial_divisibility_l4000_400072


namespace f_max_min_on_interval_l4000_400040

open Real

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem f_max_min_on_interval :
  let a := 0
  let b := 2 * Real.pi / 3
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc a b, f x ≤ max) ∧
    (∃ x ∈ Set.Icc a b, f x = max) ∧
    (∀ x ∈ Set.Icc a b, min ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min) ∧
    max = 1 ∧
    min = -(1/2) * Real.exp (2 * Real.pi / 3) - 2 * Real.pi / 3 :=
by sorry

end f_max_min_on_interval_l4000_400040


namespace total_carriages_l4000_400034

/-- The number of carriages in each town -/
structure TownCarriages where
  euston : ℕ
  norfolk : ℕ
  norwich : ℕ
  flying_scotsman : ℕ

/-- The conditions given in the problem -/
def problem_conditions (tc : TownCarriages) : Prop :=
  tc.euston = tc.norfolk + 20 ∧
  tc.norwich = 100 ∧
  tc.flying_scotsman = tc.norwich + 20 ∧
  tc.euston = 130

/-- The theorem stating that the total number of carriages is 460 -/
theorem total_carriages (tc : TownCarriages) 
  (h : problem_conditions tc) : 
  tc.euston + tc.norfolk + tc.norwich + tc.flying_scotsman = 460 := by
  sorry

end total_carriages_l4000_400034


namespace lcm_of_ratio_and_value_l4000_400057

/-- Given two positive integers with a specific ratio and value, prove their LCM --/
theorem lcm_of_ratio_and_value (a b : ℕ+) (h1 : a = 45) (h2 : 4 * a = 3 * b) : 
  Nat.lcm a b = 180 := by
  sorry

end lcm_of_ratio_and_value_l4000_400057


namespace unique_modular_inverse_in_range_l4000_400067

theorem unique_modular_inverse_in_range (p : Nat) (a : Nat) 
  (h_prime : Nat.Prime p) 
  (h_odd : Odd p)
  (h_a_range : 2 ≤ a ∧ a ≤ p - 2) :
  ∃! i : Nat, 2 ≤ i ∧ i ≤ p - 2 ∧ 
    (i * a) % p = 1 ∧ 
    Nat.gcd i a = 1 := by
  sorry

end unique_modular_inverse_in_range_l4000_400067


namespace quadratic_distinct_roots_l4000_400051

/-- 
For a quadratic equation kx² + 2x - 1 = 0 to have two distinct real roots,
k must satisfy k > -1 and k ≠ 0.
-/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 + 2 * x₁ - 1 = 0 ∧ k * x₂^2 + 2 * x₂ - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
sorry

end quadratic_distinct_roots_l4000_400051

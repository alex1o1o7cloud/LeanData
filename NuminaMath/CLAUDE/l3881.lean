import Mathlib

namespace least_three_digit_with_digit_product_12_l3881_388184

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → n ≥ 143 :=
by sorry

end least_three_digit_with_digit_product_12_l3881_388184


namespace equation_proof_l3881_388127

theorem equation_proof : Real.sqrt ((5568 / 87) ^ (1/3) + Real.sqrt (72 * 2)) = Real.sqrt 256 := by
  sorry

end equation_proof_l3881_388127


namespace data_properties_l3881_388183

def data : List ℕ := [3, 3, 4, 4, 5, 5, 5, 5, 7, 11, 21]

def mode (l : List ℕ) : ℕ := sorry

def fractionLessThanMode (l : List ℕ) : ℚ := sorry

def firstQuartile (l : List ℕ) : ℕ := sorry

def medianWithinFirstQuartile (l : List ℕ) : ℚ := sorry

theorem data_properties :
  fractionLessThanMode data = 4/11 ∧
  medianWithinFirstQuartile data = 4 := by sorry

end data_properties_l3881_388183


namespace cloth_square_theorem_l3881_388191

/-- Represents a rectangular piece of cloth -/
structure Cloth where
  length : ℕ
  width : ℕ

/-- Represents a square -/
structure Square where
  side : ℕ

/-- Calculates the maximum number of squares that can be cut from a cloth -/
def maxSquares (c : Cloth) (s : Square) : ℕ :=
  (c.length / s.side) * (c.width / s.side)

theorem cloth_square_theorem :
  let cloth : Cloth := { length := 40, width := 27 }
  let square : Square := { side := 2 }
  maxSquares cloth square = 260 := by
  sorry

#eval maxSquares { length := 40, width := 27 } { side := 2 }

end cloth_square_theorem_l3881_388191


namespace number_division_problem_l3881_388123

theorem number_division_problem : ∃ x : ℚ, x / 5 = 75 + x / 6 ∧ x = 2250 := by
  sorry

end number_division_problem_l3881_388123


namespace square_root_of_four_l3881_388113

theorem square_root_of_four (a : ℝ) : a^2 = 4 → a = 2 ∨ a = -2 := by
  sorry

end square_root_of_four_l3881_388113


namespace x_plus_y_values_l3881_388110

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 3) (h2 : y^2 = 4) (h3 : x * y < 0) :
  x + y = 1 ∨ x + y = -1 := by
  sorry

end x_plus_y_values_l3881_388110


namespace car_overtake_time_l3881_388160

/-- The time it takes for a car to overtake a motorcyclist by 36 km -/
theorem car_overtake_time (v_motorcycle : ℝ) (v_car : ℝ) (head_start : ℝ) (overtake_distance : ℝ) :
  v_motorcycle = 45 →
  v_car = 60 →
  head_start = 2/3 →
  overtake_distance = 36 →
  ∃ t : ℝ, t = 4.4 ∧ 
    v_car * t = v_motorcycle * (t + head_start) + overtake_distance :=
by sorry

end car_overtake_time_l3881_388160


namespace feet_quadrilateral_similar_l3881_388171

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The feet of perpendiculars from vertices to diagonals -/
def feet_of_perpendiculars (q : Quadrilateral) : Quadrilateral :=
  sorry

/-- Similarity relation between two quadrilaterals -/
def is_similar (q1 q2 : Quadrilateral) : Prop :=
  sorry

/-- Theorem: The quadrilateral formed by the feet of perpendiculars 
    is similar to the original quadrilateral -/
theorem feet_quadrilateral_similar (q : Quadrilateral) :
  is_similar q (feet_of_perpendiculars q) :=
sorry

end feet_quadrilateral_similar_l3881_388171


namespace smallest_prime_with_composite_reverse_l3881_388153

/-- A function that reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- The main theorem -/
theorem smallest_prime_with_composite_reverse :
  ∃ (p : ℕ),
    isPrime p ∧
    p ≥ 10 ∧
    p < 100 ∧
    p / 10 = 3 ∧
    isComposite (reverseDigits p) ∧
    (∀ q : ℕ, isPrime q → q ≥ 10 → q < 100 → q / 10 = 3 →
      isComposite (reverseDigits q) → p ≤ q) ∧
    p = 23 :=
  sorry

end smallest_prime_with_composite_reverse_l3881_388153


namespace sum_reciprocals_simplification_l3881_388138

theorem sum_reciprocals_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^3 + b^3 = 3*(a + b)) : 
  a/b + b/a + 1/(a*b) = 4/(a*b) + 1 := by
sorry

end sum_reciprocals_simplification_l3881_388138


namespace digit_value_in_different_bases_l3881_388189

theorem digit_value_in_different_bases :
  ∃ (d : ℕ), d < 7 ∧ d * 7 + 4 = d * 8 + 1 :=
by
  -- The proof goes here
  sorry

end digit_value_in_different_bases_l3881_388189


namespace additional_teddies_calculation_l3881_388157

/-- The number of additional teddies Jina gets for each bunny -/
def additional_teddies_per_bunny : ℕ :=
  let initial_teddies : ℕ := 5
  let bunnies : ℕ := 3 * initial_teddies
  let koalas : ℕ := 1
  let total_mascots : ℕ := 51
  let initial_mascots : ℕ := initial_teddies + bunnies + koalas
  let additional_teddies : ℕ := total_mascots - initial_mascots
  additional_teddies / bunnies

theorem additional_teddies_calculation : additional_teddies_per_bunny = 2 := by
  sorry

end additional_teddies_calculation_l3881_388157


namespace marie_magazine_sales_l3881_388164

/-- Given that Marie sold a total of 425.0 magazines and newspapers,
    and 275.0 of them were newspapers, prove that she sold 150.0 magazines. -/
theorem marie_magazine_sales :
  let total_sales : ℝ := 425.0
  let newspaper_sales : ℝ := 275.0
  let magazine_sales : ℝ := total_sales - newspaper_sales
  magazine_sales = 150.0 := by sorry

end marie_magazine_sales_l3881_388164


namespace min_value_of_fraction_l3881_388143

theorem min_value_of_fraction :
  ∀ x : ℝ, (1/2 * x^2 + x + 1 ≠ 0) →
  ((3 * x^2 + 6 * x + 5) / (1/2 * x^2 + x + 1) ≥ 4) ∧
  (∃ y : ℝ, (1/2 * y^2 + y + 1 ≠ 0) ∧ ((3 * y^2 + 6 * y + 5) / (1/2 * y^2 + y + 1) = 4)) :=
by sorry

end min_value_of_fraction_l3881_388143


namespace expansion_coefficient_l3881_388111

/-- The coefficient of x^5y^2 in the expansion of (x^2 + x + y)^5 -/
def coefficient_x5y2 : ℕ :=
  -- We don't define the actual calculation here, just the type
  30

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

theorem expansion_coefficient :
  coefficient_x5y2 = binomial 5 2 * binomial 3 1 := by sorry

end expansion_coefficient_l3881_388111


namespace eraser_cost_proof_l3881_388116

/-- The cost of an eraser in dollars -/
def eraser_cost : ℚ := 2

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := 4

/-- The number of pencils sold -/
def pencils_sold : ℕ := 20

/-- The total revenue in dollars -/
def total_revenue : ℚ := 80

/-- The ratio of erasers to pencils sold -/
def eraser_pencil_ratio : ℕ := 2

theorem eraser_cost_proof :
  eraser_cost = 2 ∧
  eraser_cost = pencil_cost / 2 ∧
  pencils_sold * pencil_cost = total_revenue ∧
  pencils_sold * eraser_pencil_ratio * eraser_cost = total_revenue / 2 :=
by sorry

end eraser_cost_proof_l3881_388116


namespace sqrt_equation_solutions_l3881_388165

theorem sqrt_equation_solutions (x : ℝ) :
  Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 5 ↔ x = 2 ∨ x = -2 := by
  sorry

end sqrt_equation_solutions_l3881_388165


namespace correct_sum_after_mistake_l3881_388158

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem correct_sum_after_mistake (original : ℕ) (mistaken : ℕ) :
  is_three_digit original →
  original % 10 = 9 →
  mistaken = original - 3 →
  mistaken + 57 = 823 →
  original + 57 = 826 := by
sorry

end correct_sum_after_mistake_l3881_388158


namespace geometric_series_second_term_l3881_388162

theorem geometric_series_second_term 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = 1/4) 
  (h2 : S = 16) 
  (h3 : S = a / (1 - r)) 
  (h4 : second_term = a * r) : second_term = 3 := by
  sorry

end geometric_series_second_term_l3881_388162


namespace fixed_point_on_graph_l3881_388175

theorem fixed_point_on_graph :
  ∀ (k : ℝ), 112 = 7 * (4 : ℝ)^2 + k * 4 - 4 * k := by sorry

end fixed_point_on_graph_l3881_388175


namespace instantaneous_velocity_at_t1_l3881_388132

-- Define the motion distance function
def S (t : ℝ) : ℝ := t^3 - 2

-- Define the derivative of S
def S_derivative (t : ℝ) : ℝ := 3 * t^2

-- Theorem statement
theorem instantaneous_velocity_at_t1 :
  S_derivative 1 = 3 := by sorry

end instantaneous_velocity_at_t1_l3881_388132


namespace charge_account_interest_l3881_388120

/-- Calculate the amount owed after one year with simple interest -/
def amountOwed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given a charge of $54 with 5% simple annual interest, 
    the amount owed after one year is $56.70 -/
theorem charge_account_interest : 
  let principal : ℝ := 54
  let rate : ℝ := 0.05
  let time : ℝ := 1
  amountOwed principal rate time = 56.70 := by
  sorry


end charge_account_interest_l3881_388120


namespace even_and_mono_decreasing_implies_ordering_l3881_388145

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of being monotonically decreasing on an interval
def IsMonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- State the theorem
theorem even_and_mono_decreasing_implies_ordering (heven : IsEven f) 
    (hmono : IsMonoDecreasing (fun x ↦ f (x - 2)) 0 2) :
  f 2 < f (-1) ∧ f (-1) < f 0 :=
by sorry

end even_and_mono_decreasing_implies_ordering_l3881_388145


namespace pats_stick_is_30_inches_l3881_388148

/-- The length of Pat's stick in inches -/
def pats_stick_length : ℝ := 30

/-- The length of the portion of Pat's stick covered in dirt, in inches -/
def covered_portion : ℝ := 7

/-- The length of Sarah's stick in inches -/
def sarahs_stick_length : ℝ := 46

/-- The length of Jane's stick in inches -/
def janes_stick_length : ℝ := 22

/-- Proves that Pat's stick is 30 inches long given the conditions -/
theorem pats_stick_is_30_inches :
  (pats_stick_length = covered_portion + (sarahs_stick_length / 2)) ∧
  (janes_stick_length = sarahs_stick_length - 24) ∧
  (janes_stick_length = 22) →
  pats_stick_length = 30 := by sorry

end pats_stick_is_30_inches_l3881_388148


namespace anthony_transaction_percentage_l3881_388135

/-- Proves that Anthony handled 10% more transactions than Mabel -/
theorem anthony_transaction_percentage (mabel cal jade anthony : ℕ) : 
  mabel = 90 →
  cal = (2 : ℚ) / 3 * anthony →
  jade = cal + 17 →
  jade = 83 →
  (anthony - mabel : ℚ) / mabel * 100 = 10 := by
  sorry

end anthony_transaction_percentage_l3881_388135


namespace weaving_problem_l3881_388166

/-- Sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + (n * (n - 1) / 2) * d

/-- The weaving problem -/
theorem weaving_problem : arithmeticSum 5 (16/29) 30 = 390 := by
  sorry

end weaving_problem_l3881_388166


namespace solution_set_of_inequality_l3881_388103

theorem solution_set_of_inequality (x : ℝ) :
  (-x^2 + 2*x + 15 ≥ 0) ↔ (-3 ≤ x ∧ x ≤ 5) := by sorry

end solution_set_of_inequality_l3881_388103


namespace farm_animals_product_l3881_388182

theorem farm_animals_product (pigs chickens : ℕ) : 
  chickens = pigs + 12 →
  chickens + pigs = 52 →
  pigs * chickens = 640 :=
by
  sorry

end farm_animals_product_l3881_388182


namespace blue_lipstick_count_l3881_388140

theorem blue_lipstick_count (total_students : ℕ) 
  (h_total : total_students = 360)
  (h_half_lipstick : ∃ lipstick_wearers : ℕ, 2 * lipstick_wearers = total_students)
  (h_red : ∃ red_wearers : ℕ, 4 * red_wearers = lipstick_wearers)
  (h_pink : ∃ pink_wearers : ℕ, 3 * pink_wearers = lipstick_wearers)
  (h_purple : ∃ purple_wearers : ℕ, 6 * purple_wearers = lipstick_wearers)
  (h_green : ∃ green_wearers : ℕ, 12 * green_wearers = lipstick_wearers)
  (h_blue : ∃ blue_wearers : ℕ, blue_wearers = lipstick_wearers - (red_wearers + pink_wearers + purple_wearers + green_wearers)) :
  blue_wearers = 30 := by
  sorry


end blue_lipstick_count_l3881_388140


namespace remainder_preserved_l3881_388169

theorem remainder_preserved (n : ℤ) (h : n % 8 = 3) : (n + 5040) % 8 = 3 := by
  sorry

end remainder_preserved_l3881_388169


namespace mary_remaining_stickers_l3881_388152

/-- Calculates the number of remaining stickers after Mary uses some on her journal. -/
def remaining_stickers (initial : ℕ) (front_page : ℕ) (other_pages : ℕ) (per_other_page : ℕ) : ℕ :=
  initial - (front_page + other_pages * per_other_page)

/-- Proves that Mary has 44 stickers remaining after using some on her journal. -/
theorem mary_remaining_stickers :
  remaining_stickers 89 3 6 7 = 44 := by
  sorry

end mary_remaining_stickers_l3881_388152


namespace expression_factorization_l3881_388159

theorem expression_factorization (x : ℝ) :
  (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 6 * x^4 + 5) = 2 * (6 * x^6 + 21 * x^4 - 7) :=
by sorry

end expression_factorization_l3881_388159


namespace sum_of_first_49_odd_numbers_l3881_388150

theorem sum_of_first_49_odd_numbers : 
  (Finset.range 49).sum (fun i => 2 * i + 1) = 2401 := by
  sorry

end sum_of_first_49_odd_numbers_l3881_388150


namespace hockey_league_games_l3881_388199

/-- The number of games played in a hockey league season --/
def hockey_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

/-- Theorem: In a hockey league with 25 teams, where each team faces all other teams 15 times, 
    the total number of games played in the season is 4500. --/
theorem hockey_league_games : hockey_games 25 15 = 4500 := by
  sorry

end hockey_league_games_l3881_388199


namespace geometric_arithmetic_relation_l3881_388193

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

def arithmetic_sequence (b : ℕ → ℝ) := ∀ n, b (n + 1) - b n = b 2 - b 1

theorem geometric_arithmetic_relation (a : ℕ → ℝ) (b : ℕ → ℝ) :
  geometric_sequence a ∧ 
  a 1 = 2 ∧ 
  a 4 = 16 ∧
  arithmetic_sequence b ∧
  b 3 = a 3 ∧
  b 5 = a 5 →
  (∀ n, a n = 2^n) ∧ 
  b 45 = a 9 := by
sorry

end geometric_arithmetic_relation_l3881_388193


namespace g_has_four_roots_l3881_388151

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then |2^x - 1| else 3 / (x - 1)

-- Define the composition function g
noncomputable def g (x : ℝ) : ℝ := f (f x) - 2

-- Theorem statement
theorem g_has_four_roots :
  ∃ (a b c d : ℝ), (∀ x : ℝ, g x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :=
sorry

end g_has_four_roots_l3881_388151


namespace quadratic_root_implies_coefficients_l3881_388102

theorem quadratic_root_implies_coefficients
  (a b : ℚ)
  (h : (1 + Real.sqrt 3)^2 + a * (1 + Real.sqrt 3) + b = 0) :
  a = -2 ∧ b = -2 := by
  sorry

end quadratic_root_implies_coefficients_l3881_388102


namespace distance_to_grandma_is_100_l3881_388128

/-- Represents the efficiency of a car in miles per gallon -/
def car_efficiency : ℝ := 20

/-- Represents the amount of gas needed to reach Grandma's house in gallons -/
def gas_needed : ℝ := 5

/-- Calculates the distance to Grandma's house based on car efficiency and gas needed -/
def distance_to_grandma : ℝ := car_efficiency * gas_needed

/-- Theorem stating that the distance to Grandma's house is 100 miles -/
theorem distance_to_grandma_is_100 : distance_to_grandma = 100 := by
  sorry

end distance_to_grandma_is_100_l3881_388128


namespace monotonicity_nonpositive_a_monotonicity_positive_a_f_geq_f_neg_l3881_388101

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

-- Theorem for monotonicity when a ≤ 0
theorem monotonicity_nonpositive_a (a : ℝ) (h : a ≤ 0) :
  StrictMono (f a) := by sorry

-- Theorem for monotonicity when a > 0
theorem monotonicity_positive_a (a : ℝ) (h : a > 0) :
  ∀ x y, x < y → (
    (x < Real.log a ∧ y < Real.log a → f a y < f a x) ∧
    (Real.log a < x ∧ Real.log a < y → f a x < f a y)
  ) := by sorry

-- Theorem for f(x) ≥ f(-x) when a = 1 and x ≥ 0
theorem f_geq_f_neg (x : ℝ) (h : x ≥ 0) :
  f 1 x ≥ f 1 (-x) := by sorry

end

end monotonicity_nonpositive_a_monotonicity_positive_a_f_geq_f_neg_l3881_388101


namespace line_intersection_condition_l3881_388174

/-- Given a directed line segment PQ and a line l, prove that l intersects
    the extended line segment PQ if and only if m is within a specific range. -/
theorem line_intersection_condition (m : ℝ) : 
  let P : ℝ × ℝ := (-1, 1)
  let Q : ℝ × ℝ := (2, 2)
  let l := {(x, y) : ℝ × ℝ | x + m * y + m = 0}
  (∃ (t : ℝ), (1 - t) • P + t • Q ∈ l) ↔ -3 < m ∧ m < -2/3 :=
by sorry

end line_intersection_condition_l3881_388174


namespace defect_rate_calculation_l3881_388130

/-- Calculates the overall defect rate given three suppliers' defect rates and their supply ratios -/
def overall_defect_rate (rate1 rate2 rate3 : ℚ) (ratio1 ratio2 ratio3 : ℕ) : ℚ :=
  (rate1 * ratio1 + rate2 * ratio2 + rate3 * ratio3) / (ratio1 + ratio2 + ratio3)

/-- Theorem stating that the overall defect rate for the given problem is 14/15 -/
theorem defect_rate_calculation :
  overall_defect_rate (92/100) (95/100) (94/100) 3 2 1 = 14/15 := by
  sorry

end defect_rate_calculation_l3881_388130


namespace distinct_laptop_choices_l3881_388137

/-- The number of ways to choose 3 distinct items from a set of 15 items -/
def choose_distinct (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else (n - k + 1).factorial / (n - k).factorial

theorem distinct_laptop_choices :
  choose_distinct 15 3 = 2730 := by
sorry

end distinct_laptop_choices_l3881_388137


namespace larger_number_problem_l3881_388170

theorem larger_number_problem (x y : ℝ) : 
  x - y = 5 → x + y = 27 → max x y = 16 := by
  sorry

end larger_number_problem_l3881_388170


namespace johns_allowance_spent_l3881_388106

theorem johns_allowance_spent (allowance : ℚ) (arcade_fraction : ℚ) (candy_spent : ℚ) 
  (h1 : allowance = 3.375)
  (h2 : arcade_fraction = 3/5)
  (h3 : candy_spent = 0.9) :
  let remaining := allowance - arcade_fraction * allowance
  let toy_spent := remaining - candy_spent
  toy_spent / remaining = 1/3 := by sorry

end johns_allowance_spent_l3881_388106


namespace even_function_property_l3881_388142

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_negative : ∀ x < 0, f x = 1 + 2*x) : 
  ∀ x > 0, f x = 1 - 2*x := by
  sorry

end even_function_property_l3881_388142


namespace number_of_chords_l3881_388119

/-- The number of points on the circle -/
def n : ℕ := 10

/-- The number of points needed to form a chord -/
def k : ℕ := 2

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Theorem: The number of unique chords formed by selecting any 2 points 
    from 10 equally spaced points on a circle is equal to 45 -/
theorem number_of_chords : binomial n k = 45 := by sorry

end number_of_chords_l3881_388119


namespace no_rational_solution_l3881_388195

theorem no_rational_solution : ¬∃ (p q r : ℚ), p + q + r = 0 ∧ p * q * r = 1 := by
  sorry

end no_rational_solution_l3881_388195


namespace min_value_expression_l3881_388118

theorem min_value_expression (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) :
  (6 * w) / (3 * u + 2 * v) + (6 * u) / (2 * v + 3 * w) + (2 * v) / (u + w) ≥ 2.5 + 2 * Real.sqrt 6 :=
by sorry

end min_value_expression_l3881_388118


namespace logarithm_system_solution_l3881_388161

theorem logarithm_system_solution :
  ∃ (x y z : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0) ∧
    (Real.log z / Real.log (2 * x) = 3) ∧
    (Real.log z / Real.log (5 * y) = 6) ∧
    (Real.log z / Real.log (x * y) = 2/3) ∧
    (x = 1 / (2 * Real.rpow 10 (1/3))) ∧
    (y = 1 / (5 * Real.rpow 10 (1/6))) ∧
    (z = 1/10) := by
  sorry

end logarithm_system_solution_l3881_388161


namespace product_in_fourth_quadrant_l3881_388109

-- Define complex numbers z1 and z2
def z1 : ℂ := 2 + Complex.I
def z2 : ℂ := 1 - Complex.I

-- Define the product z
def z : ℂ := z1 * z2

-- Theorem statement
theorem product_in_fourth_quadrant :
  z.re > 0 ∧ z.im < 0 :=
sorry

end product_in_fourth_quadrant_l3881_388109


namespace range_of_a_l3881_388115

def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) :
  (¬ ∀ x, (¬ p x ↔ ¬ q x a)) →
  (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end range_of_a_l3881_388115


namespace negation_of_forall_geq_zero_l3881_388139

theorem negation_of_forall_geq_zero :
  (¬ ∀ x : ℝ, x^2 - x ≥ 0) ↔ (∃ x : ℝ, x^2 - x < 0) := by sorry

end negation_of_forall_geq_zero_l3881_388139


namespace evaluate_expression_l3881_388124

theorem evaluate_expression : 3 * (-5) ^ (2 ^ (3/4)) = -15 * Real.sqrt 5 := by
  sorry

end evaluate_expression_l3881_388124


namespace equation_solutions_l3881_388129

theorem equation_solutions : 
  {x : ℝ | x^4 + (3 - x)^4 = 130} = {0, 3} := by sorry

end equation_solutions_l3881_388129


namespace infinitely_many_a_for_perfect_cube_l3881_388172

theorem infinitely_many_a_for_perfect_cube (n : ℕ) :
  ∃ (f : ℕ → ℤ), Function.Injective f ∧ ∀ (k : ℕ), ∃ (m : ℕ), (n^6 + 3 * (f k) : ℤ) = m^3 :=
sorry

end infinitely_many_a_for_perfect_cube_l3881_388172


namespace ascending_order_abc_l3881_388197

theorem ascending_order_abc : 
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c := by sorry

end ascending_order_abc_l3881_388197


namespace inequality_proof_l3881_388100

theorem inequality_proof (a : ℝ) (h : a > 1) : (1/2 : ℝ) + (1 / Real.log a) ≥ 1 := by
  sorry

end inequality_proof_l3881_388100


namespace union_membership_intersection_membership_positive_product_l3881_388194

-- Statement 1
theorem union_membership (A B : Set α) (x : α) : x ∈ A ∪ B → x ∈ A ∨ x ∈ B := by sorry

-- Statement 2
theorem intersection_membership (A B : Set α) (x : α) : x ∈ A ∩ B → x ∈ A ∧ x ∈ B := by sorry

-- Statement 3
theorem positive_product (a b : ℝ) : a > 0 ∧ b > 0 → a * b > 0 := by sorry

end union_membership_intersection_membership_positive_product_l3881_388194


namespace house_square_footage_l3881_388149

def house_problem (smaller_house_original : ℝ) : Prop :=
  let larger_house : ℝ := 7300
  let expansion : ℝ := 3500
  let total_after_expansion : ℝ := 16000
  (smaller_house_original + expansion + larger_house = total_after_expansion) ∧
  (smaller_house_original = 5200)

theorem house_square_footage : ∃ (x : ℝ), house_problem x :=
  sorry

end house_square_footage_l3881_388149


namespace eliza_ironed_17_pieces_l3881_388136

/-- Calculates the total number of clothes Eliza ironed given the time spent on blouses and dresses --/
def total_clothes_ironed (blouse_time : ℕ) (dress_time : ℕ) (blouse_hours : ℕ) (dress_hours : ℕ) : ℕ :=
  let blouses := (blouse_hours * 60) / blouse_time
  let dresses := (dress_hours * 60) / dress_time
  blouses + dresses

/-- Theorem stating that Eliza ironed 17 pieces of clothes --/
theorem eliza_ironed_17_pieces :
  total_clothes_ironed 15 20 2 3 = 17 := by
  sorry

end eliza_ironed_17_pieces_l3881_388136


namespace triangle_squares_area_l3881_388117

theorem triangle_squares_area (y : ℝ) : 
  y > 0 →
  (3 * y)^2 + (6 * y)^2 + (1/2 * (3 * y) * (6 * y)) = 1000 →
  y = 10 * Real.sqrt 3 / 3 := by
sorry

end triangle_squares_area_l3881_388117


namespace unique_prime_squared_plus_fourteen_prime_l3881_388133

theorem unique_prime_squared_plus_fourteen_prime :
  ∀ p : ℕ, Prime p → Prime (p^2 + 14) → p = 3 := by
  sorry

end unique_prime_squared_plus_fourteen_prime_l3881_388133


namespace yellow_marbles_count_l3881_388167

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by sorry

end yellow_marbles_count_l3881_388167


namespace martha_latte_days_l3881_388112

/-- The number of days Martha buys a latte per week -/
def latte_days : ℕ := sorry

/-- The cost of a latte in dollars -/
def latte_cost : ℚ := 4

/-- The cost of an iced coffee in dollars -/
def iced_coffee_cost : ℚ := 2

/-- The number of days Martha buys an iced coffee per week -/
def iced_coffee_days : ℕ := 3

/-- The percentage reduction in annual coffee spending -/
def spending_reduction_percentage : ℚ := 25 / 100

/-- The amount saved in dollars due to spending reduction -/
def amount_saved : ℚ := 338

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

theorem martha_latte_days : 
  latte_days = 5 :=
by sorry

end martha_latte_days_l3881_388112


namespace triangle_area_is_36_sqrt_21_l3881_388141

/-- Triangle with an incircle that trisects a median -/
structure TriangleWithTrisectingIncircle where
  /-- Side length QR -/
  qr : ℝ
  /-- Radius of the incircle -/
  r : ℝ
  /-- Length of the median PS -/
  ps : ℝ
  /-- The incircle evenly trisects the median PS -/
  trisects_median : ps = 3 * r
  /-- QR equals 30 -/
  qr_length : qr = 30

/-- The area of a triangle with a trisecting incircle -/
def triangle_area (t : TriangleWithTrisectingIncircle) : ℝ := sorry

/-- Theorem stating the area of the specific triangle -/
theorem triangle_area_is_36_sqrt_21 (t : TriangleWithTrisectingIncircle) :
  triangle_area t = 36 * Real.sqrt 21 := by sorry

end triangle_area_is_36_sqrt_21_l3881_388141


namespace xiao_ming_brother_age_l3881_388185

/-- Check if a year has unique digits -/
def has_unique_digits (year : Nat) : Bool := sorry

/-- Find the latest year before 2013 that is a multiple of 19 and has unique digits -/
def find_birth_year : Nat := sorry

/-- Calculate age in 2013 given a birth year -/
def calculate_age (birth_year : Nat) : Nat := 2013 - birth_year

theorem xiao_ming_brother_age :
  (∀ y : Nat, y < 2013 → ¬(has_unique_digits y)) →
  has_unique_digits 2013 →
  find_birth_year % 19 = 0 →
  has_unique_digits find_birth_year →
  calculate_age find_birth_year = 18 := by sorry

end xiao_ming_brother_age_l3881_388185


namespace expression_equality_l3881_388181

theorem expression_equality : (40 - (2040 - 210)) + (2040 - (210 - 40)) = 80 := by
  sorry

end expression_equality_l3881_388181


namespace absolute_value_sum_l3881_388105

theorem absolute_value_sum (a : ℝ) (h : 1 < a ∧ a < 2) : |a - 2| + |1 - a| = 1 := by
  sorry

end absolute_value_sum_l3881_388105


namespace gcd_lcm_product_l3881_388168

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 150) (h2 : b = 180) :
  (Nat.gcd a b) * (Nat.lcm a b) = 54000 := by
  sorry

end gcd_lcm_product_l3881_388168


namespace rectangular_box_surface_area_l3881_388186

theorem rectangular_box_surface_area
  (a b c : ℝ)
  (edge_sum : a + b + c = 39)
  (diagonal : a^2 + b^2 + c^2 = 625) :
  2 * (a * b + b * c + c * a) = 896 :=
by sorry

end rectangular_box_surface_area_l3881_388186


namespace quadratic_root_in_unit_interval_l3881_388131

theorem quadratic_root_in_unit_interval (a b c : ℝ) 
  (h : 2*a + 3*b + 6*c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end quadratic_root_in_unit_interval_l3881_388131


namespace chandler_can_buy_bike_l3881_388192

/-- The cost of the mountain bike in dollars -/
def bike_cost : ℕ := 640

/-- The total amount of birthday money Chandler received in dollars -/
def birthday_money : ℕ := 60 + 40 + 20

/-- The amount Chandler earns per week from his paper route in dollars -/
def weekly_earnings : ℕ := 20

/-- The number of weeks Chandler needs to save to buy the bike -/
def weeks_to_save : ℕ := 26

/-- Theorem stating that Chandler can buy the bike after saving for 26 weeks -/
theorem chandler_can_buy_bike : 
  birthday_money + weekly_earnings * weeks_to_save = bike_cost := by
  sorry

end chandler_can_buy_bike_l3881_388192


namespace distance_traveled_by_slower_person_l3881_388198

/-- The distance traveled by the slower person when two people walk towards each other -/
theorem distance_traveled_by_slower_person
  (total_distance : ℝ)
  (speed_1 : ℝ)
  (speed_2 : ℝ)
  (h1 : total_distance = 50)
  (h2 : speed_1 = 4)
  (h3 : speed_2 = 6)
  (h4 : speed_1 < speed_2) :
  speed_1 * (total_distance / (speed_1 + speed_2)) = 20 :=
by sorry

end distance_traveled_by_slower_person_l3881_388198


namespace geometric_sequence_common_ratio_l3881_388173

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h2 : a 3 = 4) 
  (h3 : a 6 = 1/2) : 
  q = 1/2 := by
sorry

end geometric_sequence_common_ratio_l3881_388173


namespace factorization_problems_l3881_388188

theorem factorization_problems :
  (∀ x y : ℝ, 6*x*y - 9*x^2*y = 3*x*y*(2-3*x)) ∧
  (∀ a : ℝ, (a^2+1)^2 - 4*a^2 = (a+1)^2*(a-1)^2) :=
by sorry

end factorization_problems_l3881_388188


namespace factor_3x_squared_minus_75_l3881_388176

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end factor_3x_squared_minus_75_l3881_388176


namespace tax_rate_calculation_l3881_388114

/-- A special municipal payroll tax system -/
structure PayrollTaxSystem where
  threshold : ℝ
  taxRate : ℝ

/-- A company subject to the payroll tax system -/
structure Company where
  payroll : ℝ
  taxPaid : ℝ

/-- Theorem: Given the specific conditions, prove the tax rate is 0.2% -/
theorem tax_rate_calculation (system : PayrollTaxSystem) (company : Company) :
  system.threshold = 200000 ∧
  company.payroll = 400000 ∧
  company.taxPaid = 400 →
  system.taxRate = 0.002 := by
  sorry

end tax_rate_calculation_l3881_388114


namespace olympiad_problem_l3881_388104

theorem olympiad_problem (total_students : ℕ) 
  (solved_at_least_1 solved_at_least_2 solved_at_least_3 solved_at_least_4 solved_at_least_5 solved_all_6 : ℕ) : 
  total_students = 2006 →
  solved_at_least_1 = 4 * solved_at_least_2 →
  solved_at_least_2 = 4 * solved_at_least_3 →
  solved_at_least_3 = 4 * solved_at_least_4 →
  solved_at_least_4 = 4 * solved_at_least_5 →
  solved_at_least_5 = 4 * solved_all_6 →
  total_students - solved_at_least_1 = 982 :=
by sorry

end olympiad_problem_l3881_388104


namespace ABD_collinear_l3881_388121

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

variable (m n : V)
variable (A B C D : V)

axiom m_n_not_collinear : ¬ ∃ (k : ℝ), m = k • n

axiom AB_def : B - A = m + 5 • n
axiom BC_def : C - B = -2 • m + 8 • n
axiom CD_def : D - C = 4 • m + 2 • n

theorem ABD_collinear : ∃ (k : ℝ), D - A = k • (B - A) := by sorry

end ABD_collinear_l3881_388121


namespace triangle_BC_proof_l3881_388146

def triangle_BC (A B C : ℝ) (tanA : ℝ) (AB : ℝ) : Prop :=
  let angleB := Real.pi / 2
  let BC := ((AB ^ 2) + (tanA * AB) ^ 2).sqrt
  angleB = Real.pi / 2 ∧ 
  tanA = 3 / 7 ∧ 
  AB = 14 → 
  BC = 2 * Real.sqrt 58

theorem triangle_BC_proof : triangle_BC Real.pi Real.pi Real.pi (3/7) 14 := by
  sorry

end triangle_BC_proof_l3881_388146


namespace eccentricity_range_lower_bound_l3881_388107

/-- The common foci of an ellipse and a hyperbola -/
structure CommonFoci :=
  (F₁ F₂ : ℝ × ℝ)

/-- An ellipse with equation x²/a² + y²/b² = 1 -/
structure Ellipse :=
  (a b : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_gt_b : a > b)

/-- A hyperbola with equation x²/m² - y²/n² = 1 -/
structure Hyperbola :=
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)

/-- A point in the first quadrant -/
structure FirstQuadrantPoint :=
  (P : ℝ × ℝ)
  (h_x_pos : P.1 > 0)
  (h_y_pos : P.2 > 0)

/-- The main theorem -/
theorem eccentricity_range_lower_bound
  (cf : CommonFoci)
  (e : Ellipse)
  (h : Hyperbola)
  (P : FirstQuadrantPoint)
  (h_common_point : P.P ∈ {x : ℝ × ℝ | x.1^2 / e.a^2 + x.2^2 / e.b^2 = 1} ∩
                            {x : ℝ × ℝ | x.1^2 / h.m^2 - x.2^2 / h.n^2 = 1})
  (h_orthogonal : (cf.F₂.1 - P.P.1, cf.F₂.2 - P.P.2) • (P.P.1 - cf.F₁.1, P.P.2 - cf.F₁.2) +
                  (cf.F₂.1 - cf.F₁.1, cf.F₂.2 - cf.F₁.2) • (P.P.1 - cf.F₁.1, P.P.2 - cf.F₁.2) = 0)
  (e₁ : ℝ)
  (h_e₁ : e₁ = Real.sqrt (1 - e.b^2 / e.a^2))
  (e₂ : ℝ)
  (h_e₂ : e₂ = Real.sqrt (1 + h.n^2 / h.m^2)) :
  (4 + e₁ * e₂) / (2 * e₁) ≥ 6 :=
sorry

end eccentricity_range_lower_bound_l3881_388107


namespace even_function_inequality_l3881_388187

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

theorem even_function_inequality (f : ℝ → ℝ) (m : ℝ) :
  is_even_function f →
  (∀ x, -2 ≤ x → x ≤ 2 → f x ∈ Set.range f) →
  monotone_decreasing_on f 0 2 →
  f (1 - m) < f m →
  -1 ≤ m ∧ m < 1/2 := by sorry

end even_function_inequality_l3881_388187


namespace different_author_book_pairs_l3881_388180

/-- Given two groups of books, this theorem proves that the number of different pairs
    that can be formed by selecting one book from each group is equal to the product
    of the number of books in each group. -/
theorem different_author_book_pairs (group1 group2 : ℕ) (h1 : group1 = 6) (h2 : group2 = 9) :
  group1 * group2 = 54 := by
  sorry

end different_author_book_pairs_l3881_388180


namespace eight_pointed_star_tip_sum_l3881_388144

/-- An 8-pointed star formed by connecting 8 evenly spaced points on a circle -/
structure EightPointedStar where
  /-- The number of points on the circle -/
  num_points : ℕ
  /-- The points are evenly spaced -/
  evenly_spaced : num_points = 8
  /-- The measure of each small arc between adjacent points -/
  small_arc_measure : ℝ
  /-- Each small arc is 1/8 of the full circle -/
  small_arc_def : small_arc_measure = 360 / 8

/-- The sum of angle measurements of the eight tips of the star -/
def sum_of_tip_angles (star : EightPointedStar) : ℝ :=
  8 * (360 - 4 * star.small_arc_measure)

theorem eight_pointed_star_tip_sum :
  ∀ (star : EightPointedStar), sum_of_tip_angles star = 1440 := by
  sorry

end eight_pointed_star_tip_sum_l3881_388144


namespace sum_of_squared_complements_geq_two_l3881_388163

theorem sum_of_squared_complements_geq_two 
  (a b c : ℝ) 
  (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (h2 : a + b + c = 1) : 
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := by
  sorry

end sum_of_squared_complements_geq_two_l3881_388163


namespace line_intersection_canonical_equations_l3881_388196

/-- The canonical equations of the line of intersection of two planes -/
theorem line_intersection_canonical_equations
  (p₁ : Real → Real → Real → Real)
  (p₂ : Real → Real → Real → Real)
  (h₁ : ∀ x y z, p₁ x y z = 3*x + y - z - 6)
  (h₂ : ∀ x y z, p₂ x y z = 3*x - y + 2*z)
  : ∃ (t : Real), ∀ x y z,
    (p₁ x y z = 0 ∧ p₂ x y z = 0) ↔
    (x = 1 + t ∧ y = 3 - 9*t ∧ z = -6*t) :=
sorry

end line_intersection_canonical_equations_l3881_388196


namespace max_value_sqrt_x_over_x_plus_one_l3881_388154

theorem max_value_sqrt_x_over_x_plus_one :
  (∃ x : ℝ, x ≥ 0 ∧ Real.sqrt x / (x + 1) = 1/2) ∧
  (∀ x : ℝ, x ≥ 0 → Real.sqrt x / (x + 1) ≤ 1/2) := by
  sorry

end max_value_sqrt_x_over_x_plus_one_l3881_388154


namespace sara_movie_day_total_expense_l3881_388155

def movie_day_expenses (ticket_price : ℚ) (num_tickets : ℕ) (rented_movie : ℚ) (snacks : ℚ) (parking : ℚ) (movie_poster : ℚ) (bought_movie : ℚ) : ℚ :=
  ticket_price * num_tickets + rented_movie + snacks + parking + movie_poster + bought_movie

theorem sara_movie_day_total_expense :
  movie_day_expenses 10.62 2 1.59 8.75 5.50 12.50 13.95 = 63.53 := by
  sorry

end sara_movie_day_total_expense_l3881_388155


namespace polynomial_remainder_l3881_388147

theorem polynomial_remainder (x : ℝ) : 
  (x^4 - 4*x^2 + 7*x - 8) % (x - 3) = 58 := by
  sorry

end polynomial_remainder_l3881_388147


namespace P_on_x_axis_P_parallel_to_y_axis_l3881_388178

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (2 * a - 2, a + 5)

-- Define the point Q
def Q : ℝ × ℝ := (4, 5)

-- Theorem for part 1
theorem P_on_x_axis (a : ℝ) : 
  P a = (-12, 0) ↔ (P a).2 = 0 :=
sorry

-- Theorem for part 2
theorem P_parallel_to_y_axis (a : ℝ) :
  (P a).1 = Q.1 → P a = (4, 8) ∧ (P a).1 > 0 ∧ (P a).2 > 0 :=
sorry

end P_on_x_axis_P_parallel_to_y_axis_l3881_388178


namespace intersection_of_A_and_B_l3881_388190

def A : Set ℝ := {x : ℝ | x > 3}
def B : Set ℝ := {x : ℝ | (x - 1) * (x - 4) < 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 3 4 := by sorry

end intersection_of_A_and_B_l3881_388190


namespace ellipse_line_slope_product_l3881_388156

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_ecc : (a^2 - b^2) / a^2 = 1/2
  h_point : 4/a^2 + 2/b^2 = 1

/-- A line not passing through origin and not parallel to axes -/
structure Line where
  k : ℝ
  b : ℝ
  h_k_nonzero : k ≠ 0
  h_b_nonzero : b ≠ 0

/-- The theorem statement -/
theorem ellipse_line_slope_product (C : Ellipse) (l : Line) : 
  ∃ (A B M : ℝ × ℝ), 
    (A.1^2 / C.a^2 + A.2^2 / C.b^2 = 1) ∧ 
    (B.1^2 / C.a^2 + B.2^2 / C.b^2 = 1) ∧
    (A.2 = l.k * A.1 + l.b) ∧ 
    (B.2 = l.k * B.1 + l.b) ∧
    (M = ((A.1 + B.1)/2, (A.2 + B.2)/2)) →
    (M.2 / M.1) * l.k = -1/2 := by
  sorry

end ellipse_line_slope_product_l3881_388156


namespace min_value_expression_l3881_388126

theorem min_value_expression (x y : ℝ) : (x^2*y - 1)^2 + (x - y)^2 ≥ 1 := by
  sorry

end min_value_expression_l3881_388126


namespace probability_of_event_A_l3881_388179

/-- A tetrahedron with faces numbered 0, 1, 2, and 3 -/
inductive TetrahedronFace
| Zero
| One
| Two
| Three

/-- The result of throwing the tetrahedron twice -/
structure ThrowResult where
  first : TetrahedronFace
  second : TetrahedronFace

/-- Convert TetrahedronFace to a natural number -/
def faceToNat (face : TetrahedronFace) : Nat :=
  match face with
  | TetrahedronFace.Zero => 0
  | TetrahedronFace.One => 1
  | TetrahedronFace.Two => 2
  | TetrahedronFace.Three => 3

/-- Event A: m^2 + n^2 ≤ 4 -/
def eventA (result : ThrowResult) : Prop :=
  let m := faceToNat result.first
  let n := faceToNat result.second
  m^2 + n^2 ≤ 4

/-- The probability of event A occurring -/
def probabilityOfEventA : ℚ := 3/8

theorem probability_of_event_A :
  probabilityOfEventA = 3/8 := by sorry

end probability_of_event_A_l3881_388179


namespace congruence_problem_l3881_388177

theorem congruence_problem (x : ℤ) 
  (h1 : (2 + x) % 3 = 2^2 % 3)
  (h2 : (4 + x) % 5 = 3^2 % 5)
  (h3 : (6 + x) % 7 = 5^2 % 7) :
  x % 105 = 5 := by
  sorry

end congruence_problem_l3881_388177


namespace polynomial_uniqueness_l3881_388134

/-- Given a polynomial Q(x) = Q(0) + Q(1)x + Q(2)x^2 with Q(-1) = 3, prove Q(x) = 3(1 + x + x^2) -/
theorem polynomial_uniqueness (Q : ℝ → ℝ) (h1 : ∀ x, Q x = Q 0 + Q 1 * x + Q 2 * x^2) 
  (h2 : Q (-1) = 3) : ∀ x, Q x = 3 * (1 + x + x^2) := by
  sorry

end polynomial_uniqueness_l3881_388134


namespace infinite_integers_satisfying_inequality_l3881_388125

theorem infinite_integers_satisfying_inequality :
  ∃ (S : Set ℤ), (Set.Infinite S) ∧ 
  (∀ n ∈ S, (Real.sqrt (n + 1 : ℝ) ≤ Real.sqrt (3 * n + 2 : ℝ)) ∧ 
             (Real.sqrt (3 * n + 2 : ℝ) < Real.sqrt (4 * n - 1 : ℝ))) :=
sorry

end infinite_integers_satisfying_inequality_l3881_388125


namespace whitney_bought_two_posters_l3881_388122

/-- Represents the purchase at the school book fair -/
structure BookFairPurchase where
  initialAmount : ℕ
  posterCost : ℕ
  notebookCost : ℕ
  bookmarkCost : ℕ
  numNotebooks : ℕ
  numBookmarks : ℕ
  amountLeft : ℕ

/-- Theorem stating that Whitney bought 2 posters -/
theorem whitney_bought_two_posters (purchase : BookFairPurchase)
  (h1 : purchase.initialAmount = 40)
  (h2 : purchase.posterCost = 5)
  (h3 : purchase.notebookCost = 4)
  (h4 : purchase.bookmarkCost = 2)
  (h5 : purchase.numNotebooks = 3)
  (h6 : purchase.numBookmarks = 2)
  (h7 : purchase.amountLeft = 14) :
  ∃ (numPosters : ℕ), numPosters = 2 ∧
    purchase.initialAmount = 
      numPosters * purchase.posterCost +
      purchase.numNotebooks * purchase.notebookCost +
      purchase.numBookmarks * purchase.bookmarkCost +
      purchase.amountLeft :=
by sorry

end whitney_bought_two_posters_l3881_388122


namespace chris_pennies_l3881_388108

theorem chris_pennies (a c : ℕ) : 
  (c + 2 = 4 * (a - 2)) → 
  (c - 2 = 3 * (a + 2)) → 
  c = 62 := by
sorry

end chris_pennies_l3881_388108

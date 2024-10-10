import Mathlib

namespace sequence_properties_l467_46791

-- Define the sequence a_n and its sum S_n
def a (n : ℕ) : ℚ := sorry

def S (n : ℕ) : ℚ := sorry

-- Define the conditions
axiom S_def (n : ℕ) : S n = n * (6 + a n) / 2

axiom a_4 : a 4 = 12

-- Define b_n and its sum T_n
def b (n : ℕ) : ℚ := 1 / (n * a n)

def T (n : ℕ) : ℚ := sorry

-- Theorem to prove
theorem sequence_properties :
  (∀ n : ℕ, a n = 2 * n + 4) ∧
  (∀ n : ℕ, T n = 3/8 - (2*n+3)/(4*(n+1)*(n+2))) :=
sorry

end sequence_properties_l467_46791


namespace exist_functions_with_specific_periods_l467_46715

-- Define a periodic function
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- Define the smallest positive period
def SmallestPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  IsPeriodic f p ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬IsPeriodic f q

-- Theorem statement
theorem exist_functions_with_specific_periods :
  ∃ (f g : ℝ → ℝ),
    SmallestPositivePeriod f 2 ∧
    SmallestPositivePeriod g 6 ∧
    SmallestPositivePeriod (f + g) 3 := by
  sorry

end exist_functions_with_specific_periods_l467_46715


namespace min_value_expression_equality_condition_l467_46784

theorem min_value_expression (x y : ℝ) (hx : x > 4) (hy : y > 5) :
  4 * x + 9 * y + 1 / (x - 4) + 1 / (y - 5) ≥ 71 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 4) (hy : y > 5) :
  4 * x + 9 * y + 1 / (x - 4) + 1 / (y - 5) = 71 ↔ x = 9/2 ∧ y = 16/3 :=
by sorry

end min_value_expression_equality_condition_l467_46784


namespace arithmetic_sequence_sum_l467_46733

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ) :
  is_arithmetic_sequence a d →
  d > 0 →
  (a 1 + a 2 + a 3 = 15) →
  (a 1 * a 2 * a 3 = 80) →
  (a 11 + a 12 + a 13 = 105) :=
by sorry

end arithmetic_sequence_sum_l467_46733


namespace log_function_domain_l467_46779

open Real

theorem log_function_domain (f : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ Set.Icc 1 2, ∃ y, f x = log y) →
  (∀ x ∈ Set.Icc 1 2, f x = log (x + 2^x - m)) →
  m < 3 :=
by sorry

end log_function_domain_l467_46779


namespace rachels_homework_difference_l467_46763

/-- Rachel's homework problem -/
theorem rachels_homework_difference (math_pages reading_pages biology_pages history_pages chemistry_pages : ℕ) : 
  math_pages = 15 → 
  reading_pages = 6 → 
  biology_pages = 126 → 
  history_pages = 22 → 
  chemistry_pages = 35 → 
  math_pages - reading_pages = 9 := by
sorry

end rachels_homework_difference_l467_46763


namespace unique_positive_solution_l467_46788

-- Define the polynomial function
def g (x : ℝ) : ℝ := x^8 + 5*x^7 + 10*x^6 + 729*x^5 - 379*x^4

-- Theorem statement
theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ g x = 0 :=
by sorry

end unique_positive_solution_l467_46788


namespace tangent_circles_max_product_l467_46799

/-- Two externally tangent circles with given equations have a maximum product of their x-offsets --/
theorem tangent_circles_max_product (a b : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y + 2)^2 = 4) →
  (∃ x y : ℝ, (x + b)^2 + (y + 2)^2 = 1) →
  (∀ x y : ℝ, (x - a)^2 + (y + 2)^2 = 4 → (x + b)^2 + (y + 2)^2 = 1 → 
    ∃ t : ℝ, (x - a)^2 + (y + 2)^2 = (x + b)^2 + (y + 2)^2 + ((2 - 1) * t)^2) →
  (∀ c : ℝ, a * b ≤ c → c ≤ 9/4) :=
by sorry

end tangent_circles_max_product_l467_46799


namespace series_diverges_l467_46736

/-- Ceiling function -/
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

/-- The general term of the series -/
noncomputable def a (n : ℕ) : ℝ :=
  1 / (n : ℝ) ^ (1 + ceiling (Real.sin n))

/-- The series is divergent -/
theorem series_diverges : ¬ (Summable a) := by
  sorry

end series_diverges_l467_46736


namespace rectangle_on_W_perimeter_l467_46702

/-- The locus W is defined by the equation y = x^2 + 1/4 -/
def W : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 + 1/4}

/-- A rectangle is defined by its four vertices -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (dist r.A r.B + dist r.B r.C)

/-- Theorem: Any rectangle with three vertices on W has perimeter greater than 3√3 -/
theorem rectangle_on_W_perimeter (r : Rectangle) 
  (h1 : r.A ∈ W) (h2 : r.B ∈ W) (h3 : r.C ∈ W ∨ r.D ∈ W) : 
  perimeter r > 3 * Real.sqrt 3 := by
  sorry

end rectangle_on_W_perimeter_l467_46702


namespace periodic_sequence_property_l467_46783

def is_odd (n : Int) : Prop := ∃ k, n = 2 * k + 1

def sequence_property (a : ℕ → Int) : Prop :=
  ∀ n : ℕ, ∀ α : ℕ, ∀ k : Int,
    (a n = 2^α * k ∧ is_odd k) → a (n + 1) = 2^α - k

def is_periodic (a : ℕ → Int) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + d) = a n

theorem periodic_sequence_property (a : ℕ → Int) 
  (h1 : ∀ n : ℕ, a n ≠ 0)
  (h2 : sequence_property a)
  (h3 : is_periodic a) :
  ∀ n : ℕ, a (n + 2) = a n :=
sorry

end periodic_sequence_property_l467_46783


namespace backpack_cost_l467_46744

def wallet_cost : ℕ := 50
def sneakers_cost : ℕ := 100
def jeans_cost : ℕ := 50
def total_spent : ℕ := 450

def leonard_spent : ℕ := wallet_cost + 2 * sneakers_cost
def michael_jeans_cost : ℕ := 2 * jeans_cost

theorem backpack_cost (backpack_price : ℕ) :
  backpack_price = total_spent - (leonard_spent + michael_jeans_cost) →
  backpack_price = 100 := by
  sorry

end backpack_cost_l467_46744


namespace correct_final_bill_amount_l467_46760

/-- Calculates the final bill amount after applying surcharges -/
def final_bill_amount (initial_bill : ℝ) (first_surcharge_rate : ℝ) (second_surcharge_rate : ℝ) : ℝ :=
  initial_bill * (1 + first_surcharge_rate) * (1 + second_surcharge_rate)

/-- Theorem stating that the final bill amount is correct -/
theorem correct_final_bill_amount :
  final_bill_amount 800 0.05 0.08 = 907.2 := by sorry

end correct_final_bill_amount_l467_46760


namespace sequence_sum_log_l467_46723

theorem sequence_sum_log (a : ℕ → ℝ) :
  (∀ n, Real.log (a (n + 1)) = 1 + Real.log (a n)) →
  a 1 + a 2 + a 3 = 10 →
  Real.log (a 4 + a 5 + a 6) = 4 := by
sorry

end sequence_sum_log_l467_46723


namespace max_distance_point_to_circle_l467_46710

/-- The maximum distance between a point and a circle --/
theorem max_distance_point_to_circle :
  let M : ℝ × ℝ := (2, 0)
  let C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 2*p.2 = 0}
  ∀ N ∈ C, dist M N ≤ Real.sqrt 5 + 1 ∧ 
  ∃ N' ∈ C, dist M N' = Real.sqrt 5 + 1 := by
  sorry

#check max_distance_point_to_circle

end max_distance_point_to_circle_l467_46710


namespace unit_digit_G_1000_l467_46732

/-- The sequence G_n is defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The unit digit of a natural number -/
def unitDigit (n : ℕ) : ℕ := n % 10

/-- Theorem: The unit digit of G_{1000} is 2 -/
theorem unit_digit_G_1000 : unitDigit (G 1000) = 2 := by
  sorry

end unit_digit_G_1000_l467_46732


namespace xyz_equals_four_l467_46741

theorem xyz_equals_four (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 
  x * y * z = 4 := by
sorry

end xyz_equals_four_l467_46741


namespace overlapping_rectangles_area_l467_46705

/-- Given two overlapping rectangles, prove the area of the non-overlapping part of one rectangle -/
theorem overlapping_rectangles_area (a b c d overlap : ℕ) : 
  a * b = 80 → 
  c * d = 108 → 
  overlap = 37 → 
  c * d - (a * b - overlap) = 65 := by
sorry

end overlapping_rectangles_area_l467_46705


namespace milk_pricing_markup_l467_46734

/-- The wholesale price of milk in dollars -/
def wholesale_price : ℝ := 4

/-- The price paid by the customer with a 5% discount, in dollars -/
def discounted_price : ℝ := 4.75

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.05

/-- The percentage above wholesale price that we need to prove -/
def markup_percentage : ℝ := 25

theorem milk_pricing_markup :
  let original_price := discounted_price / (1 - discount_rate)
  let markup := original_price - wholesale_price
  markup / wholesale_price * 100 = markup_percentage := by
  sorry

end milk_pricing_markup_l467_46734


namespace triangle_side_length_l467_46711

theorem triangle_side_length (AB : ℝ) (sinA sinC : ℝ) :
  AB = 30 →
  sinA = 3/5 →
  sinC = 1/4 →
  ∃ (BD BC DC : ℝ),
    BD = AB * sinA ∧
    BC = BD / sinC ∧
    DC ^ 2 = BC ^ 2 - BD ^ 2 ∧
    DC = 18 * Real.sqrt 15 :=
by
  sorry

end triangle_side_length_l467_46711


namespace arithmetic_series_sum_30_60_one_third_l467_46772

def arithmeticSeriesSum (a1 : ℚ) (an : ℚ) (d : ℚ) : ℚ :=
  let n : ℚ := (an - a1) / d + 1
  n * (a1 + an) / 2

theorem arithmetic_series_sum_30_60_one_third :
  arithmeticSeriesSum 30 60 (1/3) = 4095 := by
  sorry

end arithmetic_series_sum_30_60_one_third_l467_46772


namespace perforation_information_count_l467_46780

theorem perforation_information_count : 
  ∀ (n : ℕ) (states : ℕ), 
    n = 8 → 
    states = 2 → 
    states ^ n = 256 := by
  sorry

end perforation_information_count_l467_46780


namespace contractor_daily_wage_l467_46765

/-- Calculates the daily wage for a contractor given contract details -/
def daily_wage (total_days : ℕ) (absence_fine : ℚ) (total_payment : ℚ) (absent_days : ℕ) : ℚ :=
  (total_payment + (absent_days : ℚ) * absence_fine) / ((total_days - absent_days) : ℚ)

/-- Proves that the daily wage is 25 given the contract details -/
theorem contractor_daily_wage :
  daily_wage 30 (7.5 : ℚ) 620 4 = 25 := by
  sorry

end contractor_daily_wage_l467_46765


namespace min_sum_abc_l467_46755

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Given positive integers A, B, C satisfying the conditions, 
    the minimum value of A + B + C is 91 -/
theorem min_sum_abc (A B C : ℕ+) 
  (hA : num_divisors A = 7)
  (hB : num_divisors B = 6)
  (hC : num_divisors C = 3)
  (hAB : num_divisors (A * B) = 24)
  (hBC : num_divisors (B * C) = 10) :
  (∀ A' B' C' : ℕ+, 
    num_divisors A' = 7 → 
    num_divisors B' = 6 → 
    num_divisors C' = 3 → 
    num_divisors (A' * B') = 24 → 
    num_divisors (B' * C') = 10 → 
    A + B + C ≤ A' + B' + C') ∧ 
  A + B + C = 91 := by
sorry

end min_sum_abc_l467_46755


namespace stationery_gain_percentage_l467_46775

/-- Represents the gain percentage calculation for pens and pencils -/
theorem stationery_gain_percentage
  (P : ℝ) -- Cost price of a pen pack
  (Q : ℝ) -- Cost price of a pencil pack
  (h1 : P > 0)
  (h2 : Q > 0)
  (h3 : 80 * P = 100 * P - 20 * P) -- Selling 80 packs of pens gains the cost of 20 packs
  (h4 : 120 * Q = 150 * Q - 30 * Q) -- Selling 120 packs of pencils gains the cost of 30 packs
  : (20 * P) / (80 * P) * 100 = 25 ∧ (30 * Q) / (120 * Q) * 100 = 25 :=
sorry


end stationery_gain_percentage_l467_46775


namespace decimal_parts_sum_l467_46745

theorem decimal_parts_sum (a b : ℝ) : 
  (∃ n : ℕ, n^2 < 5 ∧ (n+1)^2 > 5 ∧ a = Real.sqrt 5 - n) →
  (∃ m : ℕ, m^2 < 13 ∧ (m+1)^2 > 13 ∧ b = Real.sqrt 13 - m) →
  a + b - Real.sqrt 5 = Real.sqrt 13 - 5 := by
sorry

end decimal_parts_sum_l467_46745


namespace probability_of_specific_student_selection_l467_46730

/- Define the total number of students and the number to be selected -/
def total_students : ℕ := 303
def selected_students : ℕ := 50

/- Define the probability of a specific student being selected -/
def probability_of_selection : ℚ := selected_students / total_students

/- Theorem statement -/
theorem probability_of_specific_student_selection :
  probability_of_selection = 50 / 303 := by
  sorry

end probability_of_specific_student_selection_l467_46730


namespace three_sixes_is_random_event_l467_46743

/-- Represents the possible outcomes of rolling a die -/
inductive DieOutcome
  | One
  | Two
  | Three
  | Four
  | Five
  | Six

/-- Represents the result of rolling 3 dice simultaneously -/
structure ThreeDiceRoll :=
  (first second third : DieOutcome)

/-- Defines the event of getting three 6s -/
def allSixes (roll : ThreeDiceRoll) : Prop :=
  roll.first = DieOutcome.Six ∧ 
  roll.second = DieOutcome.Six ∧ 
  roll.third = DieOutcome.Six

/-- Theorem stating that rolling three 6s is a random event -/
theorem three_sixes_is_random_event : 
  ∃ (roll : ThreeDiceRoll), allSixes roll ∧
  ∃ (roll' : ThreeDiceRoll), ¬allSixes roll' :=
sorry

end three_sixes_is_random_event_l467_46743


namespace original_number_proof_l467_46782

theorem original_number_proof : 
  ∃ x : ℝ, x * 16 = 3408 ∧ x = 213 := by
  sorry

end original_number_proof_l467_46782


namespace jason_initial_cards_l467_46748

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 4

/-- The number of Pokemon cards Jason has now -/
def cards_remaining : ℕ := 5

/-- The initial number of Pokemon cards Jason had -/
def initial_cards : ℕ := cards_given_away + cards_remaining

theorem jason_initial_cards : initial_cards = 9 := by
  sorry

end jason_initial_cards_l467_46748


namespace problem_solution_l467_46712

theorem problem_solution : 
  (2023^0 + |1 - Real.sqrt 2| - Real.sqrt 3 * Real.sqrt 6 = -2 * Real.sqrt 2) ∧
  ((Real.sqrt 5 - 1)^2 + Real.sqrt 5 * (Real.sqrt 5 + 2) = 11) := by
  sorry

end problem_solution_l467_46712


namespace february_savings_l467_46721

/-- Represents the savings pattern over 6 months -/
structure SavingsPattern :=
  (january : ℕ)
  (february : ℕ)
  (march : ℕ)
  (increase : ℕ)
  (total : ℕ)

/-- The savings pattern satisfies the given conditions -/
def ValidPattern (p : SavingsPattern) : Prop :=
  p.january = 2 ∧
  p.march = 8 ∧
  p.total = 126 ∧
  p.march - p.january = p.february - p.january ∧
  p.total = p.january + p.february + p.march + 
            (p.march + p.increase) + 
            (p.march + 2 * p.increase) + 
            (p.march + 3 * p.increase)

/-- The theorem to be proved -/
theorem february_savings (p : SavingsPattern) :
  ValidPattern p → p.february = 50 := by
  sorry

end february_savings_l467_46721


namespace hyperbola_eccentricity_l467_46787

/-- The eccentricity of a hyperbola with equation x²/4 - y² = 1 is √5/2 -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = Real.sqrt 5 / 2 ∧ 
  ∀ (x y : ℝ), x^2 / 4 - y^2 = 1 → e = Real.sqrt (1 + 1^2) / 2 := by
  sorry

end hyperbola_eccentricity_l467_46787


namespace different_color_pairs_count_l467_46774

/-- Represents the number of socks of each color -/
structure SockDrawer :=
  (white : ℕ)
  (brown : ℕ)
  (blue : ℕ)
  (black : ℕ)

/-- Calculates the number of ways to choose a pair of socks of different colors -/
def differentColorPairs (drawer : SockDrawer) : ℕ :=
  drawer.white * drawer.brown +
  drawer.white * drawer.blue +
  drawer.white * drawer.black +
  drawer.brown * drawer.blue +
  drawer.brown * drawer.black +
  drawer.blue * drawer.black

/-- The specific sock drawer described in the problem -/
def myDrawer : SockDrawer :=
  { white := 4
  , brown := 4
  , blue := 5
  , black := 4 }

theorem different_color_pairs_count :
  differentColorPairs myDrawer = 108 := by
  sorry

end different_color_pairs_count_l467_46774


namespace simplify_expression_1_simplify_expression_2_l467_46709

-- Define variables
variable (a b : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 : 3 * a - (4 * b - 2 * a + 1) = 5 * a - 4 * b - 1 := by
  sorry

-- Theorem for the second expression
theorem simplify_expression_2 : 2 * (5 * a - 3 * b) - 3 * (a^2 - 2 * b) = 10 * a - 3 * a^2 := by
  sorry

end simplify_expression_1_simplify_expression_2_l467_46709


namespace parallel_lines_length_l467_46770

/-- Given parallel lines AB, CD, and GH, where AB = 240 cm and CD = 160 cm, prove that the length of GH is 320/3 cm. -/
theorem parallel_lines_length (AB CD GH : ℝ) : 
  AB = 240 → CD = 160 → (GH / CD = CD / AB) → GH = 320 / 3 := by sorry

end parallel_lines_length_l467_46770


namespace inequality_system_solution_l467_46713

theorem inequality_system_solution (x : ℝ) : 
  (2 * x - 1 < x + 5) → ((x + 1) / 3 < x - 1) → (2 < x ∧ x < 6) := by
  sorry

end inequality_system_solution_l467_46713


namespace david_pushups_l467_46719

theorem david_pushups (class_average : ℕ) (david_pushups : ℕ) 
  (h1 : class_average = 30)
  (h2 : david_pushups = 44)
  (h3 : ∃ (zachary_pushups : ℕ), david_pushups = zachary_pushups + 9)
  (h4 : ∃ (hailey_pushups : ℕ), 
    (∃ (zachary_pushups : ℕ), zachary_pushups = 2 * hailey_pushups) ∧ 
    hailey_pushups = class_average - (class_average / 10)) :
  david_pushups = 63 := by
sorry

end david_pushups_l467_46719


namespace cosine_of_angle_between_vectors_l467_46747

theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) :
  a + b = (2, -8) →
  a - b = (-8, 16) →
  let cos_theta := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  cos_theta = -63/65 := by
  sorry

end cosine_of_angle_between_vectors_l467_46747


namespace arithmetic_operations_l467_46781

theorem arithmetic_operations : 
  (1 - 3 - (-8) + (-6) + 10 = 9) ∧ 
  (-12 * ((1/6) - (1/3) - (3/4)) = 11) :=
by sorry

end arithmetic_operations_l467_46781


namespace f_100_of_1990_eq_11_l467_46701

/-- Sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ) : ℕ := sumOfDigits (n^2 + 1)

/-- Iterated application of f, k times -/
def fIter (k : ℕ) (n : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k+1 => f (fIter k n)

/-- The main theorem to prove -/
theorem f_100_of_1990_eq_11 : fIter 100 1990 = 11 := by sorry

end f_100_of_1990_eq_11_l467_46701


namespace diana_earnings_l467_46703

def july_earnings : ℕ := 150

def august_earnings : ℕ := 3 * july_earnings

def september_earnings : ℕ := 2 * august_earnings

def total_earnings : ℕ := july_earnings + august_earnings + september_earnings

theorem diana_earnings : total_earnings = 1500 := by
  sorry

end diana_earnings_l467_46703


namespace room_length_proof_l467_46707

theorem room_length_proof (width : Real) (cost_per_sqm : Real) (total_cost : Real) :
  width = 3.75 →
  cost_per_sqm = 700 →
  total_cost = 14437.5 →
  (total_cost / cost_per_sqm) / width = 5.5 := by
  sorry

end room_length_proof_l467_46707


namespace solution_is_one_third_l467_46717

-- Define the logarithm function with base √3
noncomputable def log_sqrt3 (x : ℝ) : ℝ := Real.log x / Real.log (Real.sqrt 3)

-- Define the equation
def equation (x : ℝ) : Prop :=
  x > 0 ∧ log_sqrt3 x * Real.sqrt (log_sqrt3 3 - (Real.log 9 / Real.log x)) + 4 = 0

-- Theorem statement
theorem solution_is_one_third :
  ∃ (x : ℝ), equation x ∧ x = 1/3 :=
sorry

end solution_is_one_third_l467_46717


namespace circle_bisection_minimum_l467_46706

theorem circle_bisection_minimum (a b : ℝ) :
  a > 0 →
  b > 0 →
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧ 2*a*x - b*y + 2 = 0) →
  (∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 1 = 0 → (2*a*x - b*y + 2 = 0 ∨ 2*a*x - b*y + 2 ≠ 0)) →
  (1/a + 4/b) ≥ 9 :=
by sorry

end circle_bisection_minimum_l467_46706


namespace tangent_parallel_points_solution_points_l467_46724

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + x - 1

-- Define the slope of the line parallel to 4x - y = 0
def parallel_slope : ℝ := 4

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_points :
  ∀ x : ℝ, curve_derivative x = parallel_slope ↔ x = 1 ∨ x = -1 :=
sorry

theorem solution_points :
  ∀ x y : ℝ, y = curve x ∧ curve_derivative x = parallel_slope ↔ 
  (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -3) :=
sorry

end tangent_parallel_points_solution_points_l467_46724


namespace binary_representation_sqrt_theorem_l467_46727

/-- Given a positive integer d that is not a perfect square, s(n) denotes the number of digits 1 
    among the first n digits in the binary representation of √d -/
def s (d : ℕ+) (n : ℕ) : ℕ := sorry

/-- The theorem states that for a positive integer d that is not a perfect square, 
    there exists an integer A such that for all integers n ≥ A, 
    s(n) > √(2n) - 2 -/
theorem binary_representation_sqrt_theorem (d : ℕ+) 
    (h : ∀ (m : ℕ), m * m ≠ d) : 
    ∃ A : ℕ, ∀ n : ℕ, n ≥ A → s d n > Real.sqrt (2 * n) - 2 := by
  sorry

end binary_representation_sqrt_theorem_l467_46727


namespace people_needed_to_recruit_l467_46767

def total_funding : ℝ := 1000
def current_funds : ℝ := 200
def average_funding_per_person : ℝ := 10

theorem people_needed_to_recruit : 
  (total_funding - current_funds) / average_funding_per_person = 80 := by
  sorry

end people_needed_to_recruit_l467_46767


namespace bank_account_problem_l467_46786

/-- The bank account problem -/
theorem bank_account_problem (A E : ℝ) 
  (h1 : A > E)  -- Al has more money than Eliot
  (h2 : A - E = (A + E) / 12)  -- Difference is 1/12 of sum
  (h3 : A * 1.1 = E * 1.15 + 22)  -- After increase, Al has $22 more
  : E = 146.67 := by
  sorry

end bank_account_problem_l467_46786


namespace imaginary_part_of_complex_fraction_l467_46742

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((i - 1) / (i + 1)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l467_46742


namespace chess_tournament_participants_l467_46718

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 210 → n = 21 := by
  sorry

end chess_tournament_participants_l467_46718


namespace clock_problem_l467_46740

/-- Represents the original cost of the clock to the shop -/
def original_cost : ℝ := 250

/-- Represents the first selling price of the clock -/
def first_sell_price : ℝ := original_cost * 1.2

/-- Represents the buy-back price of the clock -/
def buy_back_price : ℝ := first_sell_price * 0.5

/-- Represents the second selling price of the clock -/
def second_sell_price : ℝ := buy_back_price * 1.8

theorem clock_problem :
  (original_cost - buy_back_price = 100) →
  (second_sell_price = 270) := by
  sorry

end clock_problem_l467_46740


namespace total_snow_volume_is_101_25_l467_46778

/-- The volume of snow on a rectangular section of sidewalk -/
def snow_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- The total volume of snow on two rectangular sections of sidewalk -/
def total_snow_volume (l1 w1 d1 l2 w2 d2 : ℝ) : ℝ :=
  snow_volume l1 w1 d1 + snow_volume l2 w2 d2

/-- Theorem: The total volume of snow on the given sidewalk sections is 101.25 cubic feet -/
theorem total_snow_volume_is_101_25 :
  total_snow_volume 25 3 0.75 15 3 1 = 101.25 := by
  sorry

#eval total_snow_volume 25 3 0.75 15 3 1

end total_snow_volume_is_101_25_l467_46778


namespace add_10000_seconds_to_5_45_00_l467_46790

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- Converts a time to total seconds -/
def toSeconds (t : Time) : Nat :=
  sorry

theorem add_10000_seconds_to_5_45_00 :
  let start_time : Time := ⟨5, 45, 0⟩
  let end_time : Time := ⟨8, 31, 40⟩
  addSeconds start_time 10000 = end_time :=
sorry

end add_10000_seconds_to_5_45_00_l467_46790


namespace road_planting_equation_l467_46754

/-- Represents the road planting scenario --/
structure RoadPlanting where
  x : ℕ  -- Original number of saplings
  shortage : ℕ  -- Number of saplings short when planting every 6 meters
  interval1 : ℕ  -- First planting interval (in meters)
  interval2 : ℕ  -- Second planting interval (in meters)

/-- Theorem representing the road planting problem --/
theorem road_planting_equation (rp : RoadPlanting) 
  (h1 : rp.interval1 = 6) 
  (h2 : rp.interval2 = 7) 
  (h3 : rp.shortage = 22) : 
  rp.interval1 * (rp.x + rp.shortage - 1) = rp.interval2 * (rp.x - 1) := by
  sorry

#check road_planting_equation

end road_planting_equation_l467_46754


namespace conditional_probability_B_given_A_l467_46714

/-- The number of balls in the box -/
def total_balls : ℕ := 12

/-- The number of yellow balls -/
def yellow_balls : ℕ := 5

/-- The number of blue balls -/
def blue_balls : ℕ := 4

/-- The number of green balls -/
def green_balls : ℕ := 3

/-- Event A: "the two balls drawn have different colors" -/
def event_A : Set (Fin total_balls × Fin total_balls) := sorry

/-- Event B: "one yellow ball and one blue ball are drawn" -/
def event_B : Set (Fin total_balls × Fin total_balls) := sorry

/-- The probability of event A -/
def prob_A : ℚ := sorry

/-- The probability of event B -/
def prob_B : ℚ := sorry

/-- The probability of both events A and B occurring -/
def prob_AB : ℚ := sorry

theorem conditional_probability_B_given_A :
  prob_AB / prob_A = 20 / 47 := by sorry

end conditional_probability_B_given_A_l467_46714


namespace outside_classroom_trash_l467_46749

def total_trash : ℕ := 1576
def classroom_trash : ℕ := 344

theorem outside_classroom_trash :
  total_trash - classroom_trash = 1232 := by
  sorry

end outside_classroom_trash_l467_46749


namespace tangent_slope_at_zero_l467_46725

-- Define the function f
def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x - 3)

-- State the theorem
theorem tangent_slope_at_zero :
  (deriv f) 0 = -6 := by sorry

end tangent_slope_at_zero_l467_46725


namespace max_value_x_cubed_over_y_fourth_l467_46753

theorem max_value_x_cubed_over_y_fourth (x y : ℝ) 
  (h1 : 3 ≤ x * y^2 ∧ x * y^2 ≤ 8) 
  (h2 : 4 ≤ x^2 / y ∧ x^2 / y ≤ 9) :
  ∃ (M : ℝ), M = 27 ∧ x^3 / y^4 ≤ M ∧ 
  ∃ (x₀ y₀ : ℝ), 3 ≤ x₀ * y₀^2 ∧ x₀ * y₀^2 ≤ 8 ∧ 
                 4 ≤ x₀^2 / y₀ ∧ x₀^2 / y₀ ≤ 9 ∧ 
                 x₀^3 / y₀^4 = M :=
by sorry

end max_value_x_cubed_over_y_fourth_l467_46753


namespace complex_fraction_equality_l467_46751

theorem complex_fraction_equality : Complex.I * 2 / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end complex_fraction_equality_l467_46751


namespace only_100_factorial_makes_perfect_square_l467_46756

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

def product_of_remaining_factorials (k : ℕ) : ℕ :=
  (List.range 200).foldl (λ acc i => if i + 1 ≠ k then acc * factorial (i + 1) else acc) 1

theorem only_100_factorial_makes_perfect_square :
  ∀ k : ℕ, k ≤ 200 →
    (is_perfect_square (product_of_remaining_factorials k) ↔ k = 100) :=
by sorry

end only_100_factorial_makes_perfect_square_l467_46756


namespace line_hyperbola_tangency_l467_46798

/-- The line y = k(x - √2) and the hyperbola x^2 - y^2 = 1 have only one point in common if and only if k = 1 or k = -1 -/
theorem line_hyperbola_tangency (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * (p.1 - Real.sqrt 2) ∧ p.1^2 - p.2^2 = 1) ↔ (k = 1 ∨ k = -1) := by
  sorry


end line_hyperbola_tangency_l467_46798


namespace product_of_fractions_l467_46704

theorem product_of_fractions (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_of_fractions_l467_46704


namespace boys_camp_total_l467_46738

theorem boys_camp_total (total : ℝ) 
  (h1 : 0.2 * total = total_school_a)
  (h2 : 0.3 * total_school_a = science_school_a)
  (h3 : total_school_a - science_school_a = 28) : 
  total = 200 := by
sorry

end boys_camp_total_l467_46738


namespace fraction_inequality_l467_46761

theorem fraction_inequality : 
  (1 + 1/3 : ℚ) = 4/3 ∧ 
  12/9 = 4/3 ∧ 
  8/6 = 4/3 ∧ 
  (1 + 2/7 : ℚ) ≠ 4/3 ∧ 
  16/12 = 4/3 := by
  sorry

end fraction_inequality_l467_46761


namespace simplify_expression_evaluate_expression_l467_46764

-- Problem 1
theorem simplify_expression (x y : ℝ) :
  5 * (2 * x^3 * y + 3 * x * y^2) - (6 * x * y^2 - 3 * x^3 * y) = 13 * x^3 * y + 9 * x * y^2 := by
  sorry

-- Problem 2
theorem evaluate_expression (a b : ℝ) (h1 : a + b = 9) (h2 : a * b = 20) :
  2/3 * (-15 * a + 3 * a * b) + 1/5 * (2 * a * b - 10 * a) - 4 * (a * b + 3 * b) = -140 := by
  sorry

end simplify_expression_evaluate_expression_l467_46764


namespace mary_potatoes_l467_46720

/-- The number of potatoes Mary has now, given the initial number and the number of new potatoes left -/
def total_potatoes (initial : ℕ) (new_left : ℕ) : ℕ :=
  initial + new_left

/-- Theorem stating that Mary has 11 potatoes given the initial conditions -/
theorem mary_potatoes : total_potatoes 8 3 = 11 := by
  sorry

end mary_potatoes_l467_46720


namespace M_bisects_AB_l467_46722

/-- The line equation -/
def line_equation (x y z : ℝ) : Prop :=
  (x - 2) / 3 = (y + 1) / 5 ∧ (x - 2) / 3 = (z - 3) / (-1)

/-- Point A where the line intersects the xoz plane -/
def point_A : ℝ × ℝ × ℝ := (2.6, 0, 2.8)

/-- Point B where the line intersects the xoy plane -/
def point_B : ℝ × ℝ × ℝ := (11, 14, 0)

/-- Point M that supposedly bisects AB -/
def point_M : ℝ × ℝ × ℝ := (6.8, 7, 1.4)

/-- Theorem stating that M bisects AB -/
theorem M_bisects_AB :
  line_equation point_A.1 point_A.2.1 point_A.2.2 ∧
  line_equation point_B.1 point_B.2.1 point_B.2.2 ∧
  point_M.1 = (point_A.1 + point_B.1) / 2 ∧
  point_M.2.1 = (point_A.2.1 + point_B.2.1) / 2 ∧
  point_M.2.2 = (point_A.2.2 + point_B.2.2) / 2 :=
by sorry

end M_bisects_AB_l467_46722


namespace shirt_cost_l467_46795

theorem shirt_cost (total_cost coat_price shirt_price : ℚ) : 
  total_cost = 600 →
  shirt_price = (1 / 3) * coat_price →
  total_cost = shirt_price + coat_price →
  shirt_price = 150 := by
sorry

end shirt_cost_l467_46795


namespace largest_prime_factor_of_expression_l467_46750

theorem largest_prime_factor_of_expression : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (20^4 + 15^3 - 10^5) ∧ 
  ∀ (q : ℕ), q.Prime → q ∣ (20^4 + 15^3 - 10^5) → q ≤ p ∧ p = 1787 := by
  sorry

end largest_prime_factor_of_expression_l467_46750


namespace opposite_of_two_l467_46797

theorem opposite_of_two : 
  ∃ x : ℝ, x + 2 = 0 ∧ x = -2 :=
sorry

end opposite_of_two_l467_46797


namespace guaranteed_matches_l467_46768

/-- Represents a card in a deck -/
structure Card :=
  (suit : Fin 4)
  (rank : Fin 9)

/-- A deck of cards -/
def Deck := List Card

/-- A function to split a deck into two halves -/
def split_deck (d : Deck) : Deck × Deck :=
  sorry

/-- A function to count matching pairs between two sets of cards -/
def count_matches (d1 d2 : Deck) : Nat :=
  sorry

/-- The theorem stating that the second player can always guarantee at least 15 matching pairs -/
theorem guaranteed_matches (d : Deck) : 
  d.length = 36 → ∀ (d1 d2 : Deck), split_deck d = (d1, d2) → count_matches d1 d2 ≥ 15 := by
  sorry

end guaranteed_matches_l467_46768


namespace best_competitor_is_man_l467_46737

-- Define the set of competitors
inductive Competitor
| Man
| Sister
| Son
| Niece

-- Define the gender type
inductive Gender
| Male
| Female

-- Define the function to get the gender of a competitor
def gender : Competitor → Gender
  | Competitor.Man => Gender.Male
  | Competitor.Sister => Gender.Female
  | Competitor.Son => Gender.Male
  | Competitor.Niece => Gender.Female

-- Define the function to get the twin of a competitor
def twin : Competitor → Competitor
  | Competitor.Man => Competitor.Sister
  | Competitor.Sister => Competitor.Man
  | Competitor.Son => Competitor.Niece
  | Competitor.Niece => Competitor.Son

-- Define the age equality relation
def sameAge : Competitor → Competitor → Prop := sorry

-- State the theorem
theorem best_competitor_is_man :
  ∃ (best worst : Competitor),
    (twin best ∈ [Competitor.Man, Competitor.Sister, Competitor.Son, Competitor.Niece]) ∧
    (gender (twin best) ≠ gender worst) ∧
    (sameAge best worst) →
    best = Competitor.Man :=
  sorry

end best_competitor_is_man_l467_46737


namespace right_triangle_condition_l467_46794

-- Define a triangle with angles α, β, and γ
structure Triangle where
  α : Real
  β : Real
  γ : Real
  angle_sum : α + β + γ = π
  positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ

-- Theorem statement
theorem right_triangle_condition (t : Triangle) :
  Real.sin t.α + Real.cos t.α = Real.sin t.β + Real.cos t.β →
  t.α = π/2 ∨ t.β = π/2 ∨ t.γ = π/2 :=
by sorry

end right_triangle_condition_l467_46794


namespace max_score_is_43_l467_46785

-- Define the sightseeing point type
structure SightseeingPoint where
  score : ℕ
  time : ℚ
  cost : ℕ

-- Define the list of sightseeing points
def sightseeingPoints : List SightseeingPoint := [
  ⟨10, 2/3, 1000⟩,
  ⟨7, 1/2, 700⟩,
  ⟨6, 1/3, 300⟩,
  ⟨8, 2/3, 800⟩,
  ⟨5, 1/4, 200⟩,
  ⟨9, 2/3, 900⟩,
  ⟨8, 1/2, 900⟩,
  ⟨8, 2/5, 600⟩,
  ⟨5, 1/5, 400⟩,
  ⟨6, 1/4, 600⟩
]

-- Define a function to check if a selection of points is valid
def isValidSelection (selection : List Bool) : Prop :=
  let selectedPoints := List.zipWith (λ s p => if s then some p else none) selection sightseeingPoints
  let totalTime := selectedPoints.filterMap id |>.map SightseeingPoint.time |>.sum
  let totalCost := selectedPoints.filterMap id |>.map SightseeingPoint.cost |>.sum
  totalTime < 3 ∧ totalCost ≤ 3500

-- Define a function to calculate the total score of a selection
def totalScore (selection : List Bool) : ℕ :=
  let selectedPoints := List.zipWith (λ s p => if s then some p else none) selection sightseeingPoints
  selectedPoints.filterMap id |>.map SightseeingPoint.score |>.sum

-- State the theorem
theorem max_score_is_43 :
  ∃ (selection : List Bool),
    selection.length = sightseeingPoints.length ∧
    isValidSelection selection ∧
    totalScore selection = 43 ∧
    ∀ (otherSelection : List Bool),
      otherSelection.length = sightseeingPoints.length →
      isValidSelection otherSelection →
      totalScore otherSelection ≤ 43 := by
  sorry

end max_score_is_43_l467_46785


namespace age_ratio_problem_l467_46762

def age_ratio (a_current : ℕ) (b_current : ℕ) : ℚ :=
  (a_current + 10 : ℚ) / (b_current - 10)

theorem age_ratio_problem (b_current : ℕ) (h1 : b_current = 39) (h2 : b_current + 9 = a_current) :
  age_ratio a_current b_current = 2 := by
  sorry

end age_ratio_problem_l467_46762


namespace square_difference_equality_l467_46773

theorem square_difference_equality : 1007^2 - 995^2 - 1005^2 + 997^2 = 8008 := by
  sorry

end square_difference_equality_l467_46773


namespace product_trailing_zeros_l467_46735

/-- The number of trailing zeros in the product 50 × 360 × 7 -/
def trailingZeros : ℕ := 3

/-- The product of 50, 360, and 7 -/
def product : ℕ := 50 * 360 * 7

theorem product_trailing_zeros :
  (product % (10^trailingZeros) = 0) ∧ (product % (10^(trailingZeros + 1)) ≠ 0) :=
sorry

end product_trailing_zeros_l467_46735


namespace four_balls_three_boxes_l467_46746

/-- The number of ways to distribute n distinct balls into k distinct boxes,
    with each box containing at least one ball. -/
def distributeWays (n k : ℕ) : ℕ :=
  sorry

theorem four_balls_three_boxes :
  distributeWays 4 3 = 36 :=
sorry

end four_balls_three_boxes_l467_46746


namespace complex_magnitudes_sum_l467_46758

theorem complex_magnitudes_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 7*I) = Real.sqrt 34 + Real.sqrt 58 := by
  sorry

end complex_magnitudes_sum_l467_46758


namespace students_taking_one_subject_l467_46708

theorem students_taking_one_subject (both : ℕ) (geometry_total : ℕ) (painting_only : ℕ)
  (h1 : both = 15)
  (h2 : geometry_total = 30)
  (h3 : painting_only = 18) :
  (geometry_total - both) + painting_only = 33 := by
  sorry

end students_taking_one_subject_l467_46708


namespace temperature_conversion_l467_46789

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 75 → k = 167 := by
  sorry

end temperature_conversion_l467_46789


namespace parabola_through_point_l467_46796

/-- A parabola passing through the point (4, -2) has either the equation y^2 = x or x^2 = -8y -/
theorem parabola_through_point (x y : ℝ) : 
  (x = 4 ∧ y = -2) → (y^2 = x ∨ x^2 = -8*y) := by
  sorry

end parabola_through_point_l467_46796


namespace min_value_of_sum_l467_46777

theorem min_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (1 / (3 * a + 2) + 1 / (3 * b + 2) + 1 / (3 * c + 2)) ≥ 1 ∧
  (1 / (3 * a + 2) + 1 / (3 * b + 2) + 1 / (3 * c + 2) = 1 ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
by sorry

end min_value_of_sum_l467_46777


namespace teacher_student_arrangements_l467_46716

/-- The number of arrangements for 4 teachers and 4 students in various scenarios -/
theorem teacher_student_arrangements :
  let n_teachers : ℕ := 4
  let n_students : ℕ := 4
  let factorial (n : ℕ) : ℕ := Nat.factorial n
  let choose (n k : ℕ) : ℕ := Nat.choose n k
  
  -- (I) Students stand together
  (factorial (n_teachers + 1) * factorial n_students = 2880) ∧
  
  -- (II) No two students adjacent
  (factorial n_teachers * choose (n_teachers + 1) n_students * factorial n_students = 2880) ∧
  
  -- (III) Teachers and students alternate
  (2 * factorial n_teachers * factorial n_students = 1152) :=
by sorry

end teacher_student_arrangements_l467_46716


namespace quadratic_equality_l467_46731

-- Define the two quadratic functions
def f (a c x : ℝ) : ℝ := a * (x - 2)^2 + c
def g (b x : ℝ) : ℝ := (2*x - 5) * (x - b)

-- State the theorem
theorem quadratic_equality (a c b : ℝ) : 
  (∀ x, f a c x = g b x) → b = 3/2 :=
by sorry

end quadratic_equality_l467_46731


namespace average_exists_l467_46700

theorem average_exists : ∃ N : ℝ, 13 < N ∧ N < 21 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end average_exists_l467_46700


namespace three_digit_power_sum_l467_46766

theorem three_digit_power_sum (a b c : ℕ) : a < 4 →
  (100 * a + 10 * b + c = (b + c) ^ a) ∧ (100 ≤ 100 * a + 10 * b + c) ∧ (100 * a + 10 * b + c < 1000) →
  (100 * a + 10 * b + c = 289 ∨ 100 * a + 10 * b + c = 343) :=
by sorry

end three_digit_power_sum_l467_46766


namespace curve_line_intersection_l467_46793

theorem curve_line_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2^2 = p.1 ∧ p.2 + 1 = k * p.1) → k = 0 ∨ k = -1/4 :=
by sorry

end curve_line_intersection_l467_46793


namespace value_of_a_l467_46726

def f (x : ℝ) : ℝ := 3 * x - 1

def A (a : ℝ) : Set ℝ := {1, a}
def B (a : ℝ) : Set ℝ := {a, 5}

theorem value_of_a : ∃ a : ℝ, (∀ x ∈ A a, f x ∈ B a) ∧ a = 2 := by sorry

end value_of_a_l467_46726


namespace M_on_y_axis_MN_parallel_to_y_axis_l467_46769

-- Define the point M
def M (m : ℝ) : ℝ × ℝ := (m - 1, 2 * m + 3)

-- Define the point N
def N : ℝ × ℝ := (-3, 2)

-- Theorem 1: If M lies on the y-axis, then m = 1
theorem M_on_y_axis (m : ℝ) : M m = (0, M m).2 → m = 1 := by sorry

-- Theorem 2: If MN is parallel to the y-axis, then the length of MN is 3
theorem MN_parallel_to_y_axis (m : ℝ) : 
  (M m).1 = N.1 → abs ((M m).2 - N.2) = 3 := by sorry

end M_on_y_axis_MN_parallel_to_y_axis_l467_46769


namespace age_difference_l467_46752

/-- Proves that given Sachin's age is 14 years and the ratio of Sachin's age to Rahul's age is 6:9,
    the difference between Rahul's age and Sachin's age is 7 years. -/
theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 14 →
  sachin_age * 9 = rahul_age * 6 →
  rahul_age - sachin_age = 7 := by
  sorry

end age_difference_l467_46752


namespace log_expression_simplification_l467_46792

theorem log_expression_simplification :
  (2 * (Real.log 3 / Real.log 4) + Real.log 3 / Real.log 8) *
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 2 := by
  sorry

end log_expression_simplification_l467_46792


namespace expected_total_rainfall_l467_46776

/-- Weather forecast for a single day -/
structure DailyForecast where
  sun_prob : ℝ
  rain_3in_prob : ℝ
  rain_8in_prob : ℝ
  prob_sum_is_one : sun_prob + rain_3in_prob + rain_8in_prob = 1
  probs_nonnegative : 0 ≤ sun_prob ∧ 0 ≤ rain_3in_prob ∧ 0 ≤ rain_8in_prob

/-- Calculate expected rainfall for a single day -/
def expected_daily_rainfall (forecast : DailyForecast) : ℝ :=
  forecast.sun_prob * 0 + forecast.rain_3in_prob * 3 + forecast.rain_8in_prob * 8

/-- The weather forecast for the week -/
def week_forecast : DailyForecast :=
  { sun_prob := 0.5
    rain_3in_prob := 0.3
    rain_8in_prob := 0.2
    prob_sum_is_one := by norm_num
    probs_nonnegative := by norm_num }

/-- Number of days in the forecast period -/
def forecast_days : ℕ := 5

/-- Theorem: The expected total rainfall for the week is 12.5 inches -/
theorem expected_total_rainfall :
  forecast_days * (expected_daily_rainfall week_forecast) = 12.5 := by
  sorry

end expected_total_rainfall_l467_46776


namespace arithmetic_sequence_solution_l467_46729

def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a 1 + (n - 1 : ℚ) * (a 2 - a 1)) / 2

theorem arithmetic_sequence_solution 
  (a : ℕ → ℚ) (d : ℚ) (n : ℕ) 
  (h1 : arithmetic_sequence a d)
  (h2 : a 1 + a 3 = 8)
  (h3 : a 4 ^ 2 = a 2 * a 9) :
  ((a 1 = 4 ∧ d = 0 ∧ sum_of_n_terms a n = 4 * n) ∨
   (a 1 = 20 / 9 ∧ d = 16 / 9 ∧ sum_of_n_terms a n = (8 * n^2 + 12 * n) / 9)) :=
by sorry

end arithmetic_sequence_solution_l467_46729


namespace days_until_lifting_heavy_l467_46759

/-- The number of days it takes for James' pain to subside -/
def pain_subsiding_days : ℕ := 3

/-- The factor by which the full healing time exceeds the pain subsiding time -/
def healing_factor : ℕ := 5

/-- The number of days James waits after full healing before working out -/
def waiting_days : ℕ := 3

/-- The number of weeks James waits before lifting heavy after starting to work out -/
def weeks_before_lifting : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating the total number of days until James can lift heavy again -/
theorem days_until_lifting_heavy : 
  pain_subsiding_days * healing_factor + waiting_days + weeks_before_lifting * days_per_week = 39 := by
  sorry

end days_until_lifting_heavy_l467_46759


namespace max_value_of_expression_l467_46757

theorem max_value_of_expression (x y : ℤ) 
  (h1 : x^2 + y^2 < 16) 
  (h2 : x * y > 4) : 
  ∃ (max : ℤ), max = 3 ∧ x^2 - 2*x*y - 3*y ≤ max :=
by sorry

end max_value_of_expression_l467_46757


namespace product_of_differences_l467_46728

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2008) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2007)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2008) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2007)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2008) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2007) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 4015/2008 := by
  sorry

end product_of_differences_l467_46728


namespace eighth_result_value_l467_46771

theorem eighth_result_value (total_count : Nat) (total_avg : ℝ)
  (first_7_count : Nat) (first_7_avg : ℝ)
  (next_5_count : Nat) (next_5_avg : ℝ)
  (last_5_count : Nat) (last_5_avg : ℝ)
  (h1 : total_count = 17)
  (h2 : total_avg = 24)
  (h3 : first_7_count = 7)
  (h4 : first_7_avg = 18)
  (h5 : next_5_count = 5)
  (h6 : next_5_avg = 23)
  (h7 : last_5_count = 5)
  (h8 : last_5_avg = 32) :
  (total_count : ℝ) * total_avg - 
  ((first_7_count : ℝ) * first_7_avg + (next_5_count : ℝ) * next_5_avg + (last_5_count : ℝ) * last_5_avg) = 7 := by
  sorry

end eighth_result_value_l467_46771


namespace tshirt_pricing_l467_46739

theorem tshirt_pricing (first_batch_cost second_batch_cost : ℕ)
  (quantity_ratio : ℚ) (price_difference : ℕ) (first_batch_selling_price : ℕ)
  (min_total_profit : ℕ) :
  first_batch_cost = 4000 →
  second_batch_cost = 5400 →
  quantity_ratio = 3/2 →
  price_difference = 5 →
  first_batch_selling_price = 70 →
  min_total_profit = 4060 →
  ∃ (first_batch_unit_cost : ℕ) (second_batch_min_selling_price : ℕ),
    first_batch_unit_cost = 50 ∧
    second_batch_min_selling_price = 66 ∧
    (second_batch_cost : ℚ) / (first_batch_unit_cost - price_difference) = quantity_ratio * ((first_batch_cost : ℚ) / first_batch_unit_cost) ∧
    (first_batch_selling_price - first_batch_unit_cost) * (first_batch_cost / first_batch_unit_cost) +
    (second_batch_min_selling_price - (first_batch_unit_cost - price_difference)) * (second_batch_cost / (first_batch_unit_cost - price_difference)) ≥ min_total_profit :=
by sorry

end tshirt_pricing_l467_46739

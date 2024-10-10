import Mathlib

namespace horse_gram_consumption_l2355_235512

/-- If 15 horses eat 15 bags of gram in 15 days, then 1 horse will eat 1 bag of gram in 15 days. -/
theorem horse_gram_consumption 
  (horses : ℕ) (bags : ℕ) (days : ℕ) 
  (h_horses : horses = 15)
  (h_bags : bags = 15)
  (h_days : days = 15)
  (h_consumption : horses * bags = horses * days) :
  1 * 1 = 1 * days :=
sorry

end horse_gram_consumption_l2355_235512


namespace x_cube_minus_six_x_squared_l2355_235578

theorem x_cube_minus_six_x_squared (x : ℝ) : x = 3 → x^6 - 6*x^2 = 675 := by
  sorry

end x_cube_minus_six_x_squared_l2355_235578


namespace quadratic_inequality_solutions_solve_inequality_l2355_235537

def f (a x : ℝ) : ℝ := a * x^2 + (1 + a) * x + a

theorem quadratic_inequality_solutions (a : ℝ) :
  (∃ x : ℝ, f a x ≥ 0) ↔ a ≥ -1/3 :=
sorry

theorem solve_inequality (a : ℝ) (h : a > 0) :
  {x : ℝ | f a x < a - 1} =
    if a < 1 then {x : ℝ | -1/a < x ∧ x < -1}
    else if a = 1 then ∅
    else {x : ℝ | -1 < x ∧ x < -1/a} :=
sorry

end quadratic_inequality_solutions_solve_inequality_l2355_235537


namespace gasoline_price_change_l2355_235500

/-- The price of gasoline after five months of changes -/
def final_price (initial_price : ℝ) : ℝ :=
  initial_price * 1.30 * 0.75 * 1.10 * 0.85 * 0.80

/-- Theorem stating the relationship between the initial and final price -/
theorem gasoline_price_change (initial_price : ℝ) :
  final_price initial_price = 102.60 → initial_price = 140.67 := by
  sorry

#eval final_price 140.67

end gasoline_price_change_l2355_235500


namespace solid_price_is_four_l2355_235593

/-- The price of solid color gift wrap per roll -/
def solid_price : ℝ := 4

/-- The total number of rolls sold -/
def total_rolls : ℕ := 480

/-- The total amount of money collected in dollars -/
def total_money : ℝ := 2340

/-- The number of print rolls sold -/
def print_rolls : ℕ := 210

/-- The price of print gift wrap per roll in dollars -/
def print_price : ℝ := 6

/-- Theorem stating that the price of solid color gift wrap is $4.00 per roll -/
theorem solid_price_is_four :
  solid_price = (total_money - print_rolls * print_price) / (total_rolls - print_rolls) :=
by sorry

end solid_price_is_four_l2355_235593


namespace point_on_graph_l2355_235583

/-- A point (x, y) lies on the graph of y = -6/x if and only if xy = -6 -/
def lies_on_graph (x y : ℝ) : Prop := x * y = -6

/-- The function f(x) = -6/x -/
noncomputable def f (x : ℝ) : ℝ := -6 / x

theorem point_on_graph : lies_on_graph 2 (-3) := by
  sorry

end point_on_graph_l2355_235583


namespace fraction_sum_equals_decimal_l2355_235576

theorem fraction_sum_equals_decimal : (3 / 15) + (5 / 125) + (7 / 1000) = 0.247 := by
  sorry

end fraction_sum_equals_decimal_l2355_235576


namespace irrational_count_l2355_235542

-- Define the set of numbers
def S : Set ℝ := {4 * Real.pi, 0, Real.sqrt 7, Real.sqrt 16 / 2, 0.1, 0.212212221}

-- Define a function to count irrational numbers in a set
def count_irrational (T : Set ℝ) : ℕ := sorry

-- Theorem statement
theorem irrational_count : count_irrational S = 3 := by sorry

end irrational_count_l2355_235542


namespace book_cost_price_l2355_235541

theorem book_cost_price (profit_10 profit_15 additional_profit : ℝ) 
  (h1 : profit_10 = 0.10)
  (h2 : profit_15 = 0.15)
  (h3 : additional_profit = 120) :
  ∃ cost_price : ℝ, 
    cost_price * (1 + profit_15) - cost_price * (1 + profit_10) = additional_profit ∧ 
    cost_price = 2400 := by
  sorry

end book_cost_price_l2355_235541


namespace seven_digit_palindrome_count_l2355_235533

/-- A seven-digit palindrome is a number of the form abcdcba where a, b, c, d are digits and a ≠ 0 -/
def SevenDigitPalindrome : Type := ℕ

/-- The count of valid digits for the first position of a seven-digit palindrome -/
def FirstDigitCount : ℕ := 9

/-- The count of valid digits for each of the second, third, and fourth positions of a seven-digit palindrome -/
def OtherDigitCount : ℕ := 10

/-- The total number of seven-digit palindromes -/
def TotalSevenDigitPalindromes : ℕ := FirstDigitCount * OtherDigitCount * OtherDigitCount * OtherDigitCount

theorem seven_digit_palindrome_count : TotalSevenDigitPalindromes = 9000 := by
  sorry

end seven_digit_palindrome_count_l2355_235533


namespace binomial_coefficient_two_l2355_235544

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l2355_235544


namespace f_properties_l2355_235504

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2*x - 1|

-- State the theorem
theorem f_properties :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, m = 1 → (f x m ≥ 3 ↔ x ≤ -1 ∨ x ≥ 1)) ∧
  (∀ x : ℝ, x ∈ Set.Icc m (2*m^2) → (1/2 * f x m ≤ |x + 1|)) ↔
  (1/2 < m ∧ m ≤ 1) :=
sorry

end f_properties_l2355_235504


namespace sarah_age_l2355_235584

/-- Given the ages of Sarah, Mark, Billy, and Ana, prove Sarah's age -/
theorem sarah_age (sarah mark billy ana : ℕ) 
  (h1 : sarah = 3 * mark - 4)
  (h2 : mark = billy + 4)
  (h3 : billy = ana / 2)
  (h4 : ana + 3 = 15) : 
  sarah = 26 := by
  sorry

end sarah_age_l2355_235584


namespace apple_distribution_l2355_235525

theorem apple_distribution (total_apples : ℕ) (num_people : ℕ) (apples_per_person : ℕ) :
  total_apples = 15 →
  num_people = 3 →
  apples_per_person * num_people ≤ total_apples →
  apples_per_person = total_apples / num_people →
  apples_per_person = 5 :=
by sorry

end apple_distribution_l2355_235525


namespace evaluate_expression_l2355_235520

theorem evaluate_expression (a b : ℝ) (h1 : a = 5) (h2 : b = 6) :
  3 / (2 * a + b) = 3 / 16 := by
  sorry

end evaluate_expression_l2355_235520


namespace three_numbers_sum_l2355_235592

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 7 → 
  (a + b + c) / 3 = a + 12 → 
  (a + b + c) / 3 = c - 18 → 
  a + b + c = 39 := by
sorry

end three_numbers_sum_l2355_235592


namespace age_difference_l2355_235540

/-- Represents the ages of Linda and Jane -/
structure Ages where
  linda : ℕ
  jane : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.linda = 13 ∧
  ages.linda + ages.jane + 10 = 28 ∧
  ages.linda > 2 * ages.jane

/-- The theorem to prove -/
theorem age_difference (ages : Ages) :
  problem_conditions ages →
  ages.linda - 2 * ages.jane = 3 := by
  sorry

end age_difference_l2355_235540


namespace stable_number_theorem_l2355_235555

/-- Definition of a stable number -/
def is_stable (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 ≠ 0) ∧ ((n / 10) % 10 ≠ 0) ∧ (n % 10 ≠ 0) ∧
  (n / 100 + (n / 10) % 10 > n % 10) ∧
  (n / 100 + n % 10 > (n / 10) % 10) ∧
  ((n / 10) % 10 + n % 10 > n / 100)

/-- Definition of F(n) -/
def F (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10) % 10

/-- Definition of Q(n) -/
def Q (n : ℕ) : ℕ := ((n / 10) % 10) * 10 + n / 100

/-- Main theorem -/
theorem stable_number_theorem (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 5) (hb : 1 ≤ b ∧ b ≤ 4) :
  let s := 100 * a + 101 * b + 30
  is_stable s ∧ (5 * F s + 2 * Q s) % 11 = 0 → s = 432 ∨ s = 534 := by
  sorry

end stable_number_theorem_l2355_235555


namespace quadratic_equation_roots_l2355_235534

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + m = 0 ∧ x₂^2 - 3*x₂ + m = 0) → 
  m = -1 :=
by sorry

end quadratic_equation_roots_l2355_235534


namespace triangle_perimeter_l2355_235523

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the properties of the triangle
def isOnCircumference (P Q : ℝ × ℝ) : Prop := sorry

def angleEqual (P Q R : ℝ × ℝ) : Prop := sorry

def distance (A B : ℝ × ℝ) : ℝ := sorry

def perimeter (t : Triangle) : ℝ :=
  distance t.P t.Q + distance t.Q t.R + distance t.R t.P

-- State the theorem
theorem triangle_perimeter (t : Triangle) :
  isOnCircumference t.P t.Q →
  angleEqual t.P t.Q t.R →
  distance t.Q t.R = 8 →
  distance t.P t.R = 10 →
  perimeter t = 28 := by
  sorry

end triangle_perimeter_l2355_235523


namespace system_solution_l2355_235501

theorem system_solution (x y : ℝ) (eq1 : x + 5 * y = 6) (eq2 : 3 * x - y = 2) : 
  x + y = 2 := by
  sorry

end system_solution_l2355_235501


namespace min_distance_circle_line_l2355_235586

/-- The minimum distance between a point on the circle (x-2)² + y² = 4
    and a point on the line x - y + 3 = 0 is (5√2)/2 - 2 -/
theorem min_distance_circle_line :
  let circle := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}
  let line := {q : ℝ × ℝ | q.1 - q.2 + 3 = 0}
  ∃ (d : ℝ), d = (5 * Real.sqrt 2) / 2 - 2 ∧
    ∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ circle → q ∈ line →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry


end min_distance_circle_line_l2355_235586


namespace distinct_selections_is_fifteen_l2355_235518

/-- The number of vowels in "MATHCOUNTS" -/
def num_vowels : ℕ := 3

/-- The number of distinct consonants in "MATHCOUNTS" excluding T -/
def num_distinct_consonants : ℕ := 5

/-- The number of T's in "MATHCOUNTS" -/
def num_t : ℕ := 2

/-- The total number of consonants in "MATHCOUNTS" -/
def total_consonants : ℕ := num_distinct_consonants + num_t

/-- The number of vowels to be selected -/
def vowels_to_select : ℕ := 3

/-- The number of consonants to be selected -/
def consonants_to_select : ℕ := 2

/-- The function to calculate the number of distinct ways to select letters -/
def distinct_selections : ℕ :=
  Nat.choose num_vowels vowels_to_select * Nat.choose num_distinct_consonants consonants_to_select +
  Nat.choose num_vowels vowels_to_select * Nat.choose (num_distinct_consonants - 1) (consonants_to_select - 1) +
  Nat.choose num_vowels vowels_to_select * Nat.choose (num_distinct_consonants - 2) (consonants_to_select - 2)

theorem distinct_selections_is_fifteen :
  distinct_selections = 15 :=
sorry

end distinct_selections_is_fifteen_l2355_235518


namespace parabola_vertex_l2355_235574

/-- The vertex of the parabola y = x^2 - 2 is at the point (0, -2) -/
theorem parabola_vertex (x y : ℝ) : 
  y = x^2 - 2 → (0, -2) = (x, y) ↔ x = 0 ∧ y = -2 := by
  sorry

end parabola_vertex_l2355_235574


namespace intersection_P_Q_l2355_235597

def P : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}
def Q : Set ℝ := {x : ℝ | 2 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end intersection_P_Q_l2355_235597


namespace acute_angle_equality_l2355_235552

theorem acute_angle_equality (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2) 
  (h2 : 1 + (Real.sqrt 3 / Real.tan (80 * Real.pi / 180)) = 1 / Real.sin α) : 
  α = 50 * Real.pi / 180 := by
  sorry

end acute_angle_equality_l2355_235552


namespace last_digit_of_sum_l2355_235545

theorem last_digit_of_sum (n : ℕ) : (2^1992 + 3^1992) % 10 = 7 := by sorry

end last_digit_of_sum_l2355_235545


namespace triangle_pq_distance_l2355_235573

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  AB = 4 ∧ AC = 3 ∧ BC = Real.sqrt 37

-- Define point P as the midpoint of AB
def Midpoint (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define point Q on AC at distance 1 from C
def PointOnLine (Q A C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, Q.1 = C.1 + t * (A.1 - C.1) ∧ Q.2 = C.2 + t * (A.2 - C.2)

def DistanceFromC (Q C : ℝ × ℝ) : Prop :=
  Real.sqrt ((Q.1 - C.1)^2 + (Q.2 - C.2)^2) = 1

-- Theorem statement
theorem triangle_pq_distance (A B C P Q : ℝ × ℝ) :
  Triangle A B C → Midpoint P A B → PointOnLine Q A C → DistanceFromC Q C →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end triangle_pq_distance_l2355_235573


namespace eliza_basketball_scores_l2355_235535

def first_ten_games : List Nat := [9, 3, 5, 4, 8, 2, 5, 3, 7, 6]

def total_first_ten : Nat := first_ten_games.sum

theorem eliza_basketball_scores :
  ∃ (game11 game12 : Nat),
    game11 < 10 ∧
    game12 < 10 ∧
    (total_first_ten + game11) % 11 = 0 ∧
    (total_first_ten + game11 + game12) % 12 = 0 ∧
    game11 * game12 = 15 := by
  sorry

end eliza_basketball_scores_l2355_235535


namespace solution_a_l2355_235566

-- Define the function F
def F (a b c : ℝ) : ℝ := a * b^3 + c

-- State the theorem
theorem solution_a : ∃ (a : ℝ), F a 2 3 = F a 3 10 ∧ a = -7/19 := by
  sorry

end solution_a_l2355_235566


namespace weight_loss_challenge_l2355_235562

/-- Calculates the measured weight loss percentage at the final weigh-in -/
def measuredWeightLoss (initialLoss : ℝ) (clothesWeight : ℝ) (waterRetention : ℝ) : ℝ :=
  (1 - (1 - initialLoss) * (1 + clothesWeight) * (1 + waterRetention)) * 100

/-- Theorem stating the measured weight loss percentage for given conditions -/
theorem weight_loss_challenge (initialLoss clothesWeight waterRetention : ℝ) 
  (h1 : initialLoss = 0.11)
  (h2 : clothesWeight = 0.02)
  (h3 : waterRetention = 0.015) :
  abs (measuredWeightLoss initialLoss clothesWeight waterRetention - 7.64) < 0.01 := by
  sorry

end weight_loss_challenge_l2355_235562


namespace log_simplification_l2355_235526

theorem log_simplification (x y z w t v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (ht : t > 0) (hv : v > 0) :
  Real.log (x / z) + Real.log (z / y) + Real.log (y / w) - Real.log (x * v / (w * t)) = Real.log (t / v) :=
by sorry

end log_simplification_l2355_235526


namespace continuity_at_three_l2355_235596

def f (x : ℝ) : ℝ := 2 * x^2 - 4

theorem continuity_at_three :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f x - f 3| < ε := by
  sorry

end continuity_at_three_l2355_235596


namespace height_increase_calculation_l2355_235558

/-- Represents the increase in height per decade for a specific plant species -/
def height_increase_per_decade : ℝ := sorry

/-- The number of decades in 4 centuries -/
def decades_in_four_centuries : ℕ := 40

/-- The total increase in height over 4 centuries in meters -/
def total_height_increase : ℝ := 3000

theorem height_increase_calculation :
  height_increase_per_decade * (decades_in_four_centuries : ℝ) = total_height_increase ∧
  height_increase_per_decade = 75 := by sorry

end height_increase_calculation_l2355_235558


namespace f_extrema_on_interval_l2355_235538

noncomputable def f (x : ℝ) := x^3 + 3*x^2 - 9*x + 1

theorem f_extrema_on_interval :
  let a := -4
  let b := 4
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 77 ∧ f x_min = -4 :=
by sorry

end f_extrema_on_interval_l2355_235538


namespace mans_upstream_speed_l2355_235539

/-- Given a man's downstream speed and still water speed, calculate his upstream speed -/
theorem mans_upstream_speed (downstream_speed still_water_speed : ℝ) 
  (h1 : downstream_speed = 20)
  (h2 : still_water_speed = 15) :
  still_water_speed - (downstream_speed - still_water_speed) = 10 := by
  sorry

#check mans_upstream_speed

end mans_upstream_speed_l2355_235539


namespace existence_of_constant_sequence_l2355_235513

/-- An irreducible polynomial with integer coefficients -/
def IrreducibleIntPoly := Polynomial ℤ

/-- The number of solutions to p(x) ≡ 0 mod q^n -/
def num_solutions (p : IrreducibleIntPoly) (q : ℕ) (n : ℕ) : ℕ := sorry

theorem existence_of_constant_sequence 
  (p : IrreducibleIntPoly) 
  (q : ℕ) 
  (h_q : Nat.Prime q) :
  ∃ M : ℕ, ∀ n ≥ M, num_solutions p q n = num_solutions p q M := by
  sorry

end existence_of_constant_sequence_l2355_235513


namespace family_bought_three_soft_tacos_l2355_235588

/-- Represents the taco truck's sales during lunch rush -/
structure TacoSales where
  soft_taco_price : ℕ
  hard_taco_price : ℕ
  family_hard_tacos : ℕ
  other_customers : ℕ
  soft_tacos_per_customer : ℕ
  total_revenue : ℕ

/-- Calculates the number of soft tacos bought by the family -/
def family_soft_tacos (sales : TacoSales) : ℕ :=
  (sales.total_revenue -
   sales.family_hard_tacos * sales.hard_taco_price -
   sales.other_customers * sales.soft_tacos_per_customer * sales.soft_taco_price) /
  sales.soft_taco_price

/-- Theorem stating that the family bought 3 soft tacos -/
theorem family_bought_three_soft_tacos (sales : TacoSales)
  (h1 : sales.soft_taco_price = 2)
  (h2 : sales.hard_taco_price = 5)
  (h3 : sales.family_hard_tacos = 4)
  (h4 : sales.other_customers = 10)
  (h5 : sales.soft_tacos_per_customer = 2)
  (h6 : sales.total_revenue = 66) :
  family_soft_tacos sales = 3 := by
  sorry

end family_bought_three_soft_tacos_l2355_235588


namespace log_2_5_gt_log_2_3_l2355_235546

-- Define log_2 as a strictly increasing function
def log_2 : ℝ → ℝ := sorry

-- Axiom: log_2 is strictly increasing
axiom log_2_strictly_increasing : 
  ∀ x y : ℝ, x > y → log_2 x > log_2 y

-- Theorem to prove
theorem log_2_5_gt_log_2_3 : log_2 5 > log_2 3 := by
  sorry

end log_2_5_gt_log_2_3_l2355_235546


namespace treasure_gold_amount_l2355_235515

theorem treasure_gold_amount (total_mass : ℝ) (num_brothers : ℕ) 
  (eldest_gold : ℝ) (eldest_silver_fraction : ℝ) :
  total_mass = num_brothers * 100 →
  eldest_gold = 25 →
  eldest_silver_fraction = 1 / 8 →
  ∃ (total_gold total_silver : ℝ),
    total_gold + total_silver = total_mass ∧
    total_gold = 100 ∧
    eldest_gold + eldest_silver_fraction * total_silver = 100 :=
by sorry

end treasure_gold_amount_l2355_235515


namespace algebraic_expression_value_l2355_235585

/-- Given that when x = 3, the value of px³ + qx + 3 is 2005, 
    prove that when x = -3, the value of px³ + qx + 3 is -1999 -/
theorem algebraic_expression_value (p q : ℝ) : 
  (3^3 * p + 3 * q + 3 = 2005) → ((-3)^3 * p + (-3) * q + 3 = -1999) := by sorry

end algebraic_expression_value_l2355_235585


namespace yellow_shirts_count_l2355_235548

theorem yellow_shirts_count (total : ℕ) (blue green red : ℕ) (h1 : total = 36) (h2 : blue = 8) (h3 : green = 11) (h4 : red = 6) :
  total - (blue + green + red) = 11 := by
sorry

end yellow_shirts_count_l2355_235548


namespace initial_typists_count_l2355_235569

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 20

/-- The number of letters typed by the initial group in 20 minutes -/
def letters_20min : ℕ := 44

/-- The number of typists in the second group -/
def second_group_typists : ℕ := 30

/-- The number of letters typed by the second group in 1 hour -/
def letters_1hour : ℕ := 198

/-- The ratio of 1 hour to 20 minutes -/
def time_ratio : ℕ := 3

theorem initial_typists_count :
  initial_typists * second_group_typists * letters_20min * time_ratio = letters_1hour * initial_typists * initial_typists :=
sorry

end initial_typists_count_l2355_235569


namespace not_all_exponential_increasing_l2355_235553

theorem not_all_exponential_increasing :
  ¬ (∀ a : ℝ, a > 0 ∧ a ≠ 1 → (∀ x y : ℝ, x < y → a^x < a^y)) := by
  sorry

end not_all_exponential_increasing_l2355_235553


namespace twelfth_nine_position_l2355_235598

/-- The position of the nth occurrence of a digit in the sequence of natural numbers written without spaces -/
def digitPosition (n : ℕ) (digit : ℕ) : ℕ :=
  sorry

/-- The sequence of natural numbers written without spaces -/
def naturalNumberSequence : List ℕ :=
  sorry

theorem twelfth_nine_position :
  digitPosition 12 9 = 174 :=
sorry

end twelfth_nine_position_l2355_235598


namespace fruit_merchant_problem_l2355_235527

/-- Fruit merchant problem -/
theorem fruit_merchant_problem 
  (total_cost : ℝ) 
  (quantity : ℝ) 
  (cost_difference : ℝ) 
  (large_selling_price : ℝ) 
  (small_selling_price : ℝ) 
  (loss_percentage : ℝ) 
  (earnings_percentage : ℝ) 
  (h1 : total_cost = 8000) 
  (h2 : quantity = 200) 
  (h3 : cost_difference = 20) 
  (h4 : large_selling_price = 40) 
  (h5 : small_selling_price = 16) 
  (h6 : loss_percentage = 0.2) 
  (h7 : earnings_percentage = 0.9) :
  ∃ (small_cost large_cost earnings min_large_price : ℝ),
    small_cost = 10 ∧ 
    large_cost = 30 ∧ 
    earnings = 3200 ∧ 
    min_large_price = 41.6 ∧
    quantity * small_cost + quantity * large_cost = total_cost ∧
    large_cost = small_cost + cost_difference ∧
    earnings = quantity * (large_selling_price - large_cost) + quantity * (small_selling_price - small_cost) ∧
    quantity * min_large_price + small_selling_price * quantity * (1 - loss_percentage) - total_cost ≥ earnings * earnings_percentage :=
by sorry

end fruit_merchant_problem_l2355_235527


namespace milk_mixture_theorem_l2355_235503

/-- Proves that adding 8 gallons of 10% butterfat milk to 8 gallons of 30% butterfat milk
    results in a mixture with 20% butterfat. -/
theorem milk_mixture_theorem :
  let initial_milk : ℝ := 8
  let initial_butterfat_percent : ℝ := 30
  let added_milk : ℝ := 8
  let added_butterfat_percent : ℝ := 10
  let final_butterfat_percent : ℝ := 20
  let total_milk : ℝ := initial_milk + added_milk
  let total_butterfat : ℝ := (initial_milk * initial_butterfat_percent + added_milk * added_butterfat_percent) / 100
  total_butterfat / total_milk * 100 = final_butterfat_percent :=
by sorry

end milk_mixture_theorem_l2355_235503


namespace parking_lot_buses_l2355_235519

/-- Given a parking lot with buses and cars, prove the number of buses -/
theorem parking_lot_buses (total_vehicles : ℕ) (total_wheels : ℕ) : 
  total_vehicles = 40 →
  total_wheels = 210 →
  ∃ (buses cars : ℕ),
    buses + cars = total_vehicles ∧
    6 * buses + 4 * cars = total_wheels ∧
    buses = 25 := by
  sorry

end parking_lot_buses_l2355_235519


namespace triangle_side_calculation_l2355_235560

theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  -- Given conditions
  (a = 2) →
  (A = 30 * π / 180) →  -- Convert degrees to radians
  (B = 45 * π / 180) →  -- Convert degrees to radians
  -- Law of Sines
  (a / Real.sin A = b / Real.sin B) →
  -- Conclusion
  b = 2 * Real.sqrt 2 := by
sorry

end triangle_side_calculation_l2355_235560


namespace smallest_number_l2355_235550

theorem smallest_number : 
  let numbers : List ℚ := [0, (-3)^2, |-9|, -1^4]
  (∀ x ∈ numbers, -1^4 ≤ x) ∧ (-1^4 ∈ numbers) :=
by sorry

end smallest_number_l2355_235550


namespace tangent_line_slope_l2355_235572

/-- The curve y = x³ + x + 16 -/
def f (x : ℝ) : ℝ := x^3 + x + 16

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

/-- The line ℓ passing through (0,0) and tangent to f -/
structure TangentLine where
  t : ℝ
  slope : ℝ
  tangent_point : (ℝ × ℝ) := (t, f t)
  passes_origin : slope * t = f t
  is_tangent : slope = f' t

theorem tangent_line_slope : 
  ∃ (ℓ : TangentLine), ℓ.slope = 13 :=
sorry

end tangent_line_slope_l2355_235572


namespace solve_system_for_x_l2355_235570

theorem solve_system_for_x :
  ∀ x y : ℚ, 
  (2 * x - 3 * y = 18) → 
  (x + 2 * y = 8) → 
  x = 60 / 7 := by
sorry

end solve_system_for_x_l2355_235570


namespace no_real_root_in_unit_interval_l2355_235507

theorem no_real_root_in_unit_interval (a b c d : ℝ) :
  (min d (b + d) > max (abs c) (abs (a + c))) →
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → (a * x^3 + b * x^2 + c * x + d ≠ 0) :=
by sorry

end no_real_root_in_unit_interval_l2355_235507


namespace joan_remaining_oranges_l2355_235528

/-- The number of oranges Joan picked -/
def joan_oranges : ℕ := 37

/-- The number of oranges Sara sold -/
def sara_sold : ℕ := 10

/-- The number of oranges Joan is left with -/
def joan_remaining : ℕ := joan_oranges - sara_sold

theorem joan_remaining_oranges : joan_remaining = 27 := by
  sorry

end joan_remaining_oranges_l2355_235528


namespace circle_C_equation_l2355_235564

/-- Given circle is symmetric to (x-1)^2 + y^2 = 1 with respect to y = -x -/
def given_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- Line of symmetry -/
def symmetry_line (x y : ℝ) : Prop := y = -x

/-- Circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

/-- Symmetry transformation -/
def symmetric_point (x y : ℝ) : ℝ × ℝ := (-y, -x)

theorem circle_C_equation :
  ∀ x y : ℝ, 
  (∃ x' y' : ℝ, given_circle x' y' ∧ 
   symmetric_point x y = (x', y') ∧
   symmetry_line x y) →
  circle_C x y :=
sorry

end circle_C_equation_l2355_235564


namespace inequality_solution_empty_l2355_235571

theorem inequality_solution_empty (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
sorry

end inequality_solution_empty_l2355_235571


namespace rectangle_strip_proof_l2355_235543

theorem rectangle_strip_proof (a b c : ℕ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b + a * c + a * (b - a) + a * a + a * (c - a) = 43 →
  (a = 1 ∧ b + c = 22) ∨ (a = 1 ∧ c + b = 22) :=
by sorry

end rectangle_strip_proof_l2355_235543


namespace quadratic_vertex_l2355_235522

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The vertex of a parabola -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Theorem: If f(2) = 3 and x = 2 is the axis of symmetry for f(x) = ax^2 + bx + c,
    then the vertex of the parabola is at (2, 3) -/
theorem quadratic_vertex (a b c : ℝ) :
  let f := QuadraticFunction a b c
  f 2 = 3 → -- f(2) = 3
  (∀ x, f (4 - x) = f x) → -- x = 2 is the axis of symmetry
  Vertex.mk 2 3 = Vertex.mk (2 : ℝ) (f 2) := by
  sorry

end quadratic_vertex_l2355_235522


namespace tom_twice_tim_age_l2355_235565

/-- Proves that Tom will be twice Tim's age in 3 years -/
theorem tom_twice_tim_age (tom_age tim_age : ℕ) (x : ℕ) : 
  tom_age + tim_age = 21 → 
  tom_age = 15 → 
  tom_age + x = 2 * (tim_age + x) → 
  x = 3 := by
  sorry

end tom_twice_tim_age_l2355_235565


namespace probability_nine_heads_in_twelve_flips_l2355_235524

theorem probability_nine_heads_in_twelve_flips :
  let n : ℕ := 12  -- total number of coin flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads for a fair coin
  Nat.choose n k * p^k * (1-p)^(n-k) = 220/4096 :=
by sorry

end probability_nine_heads_in_twelve_flips_l2355_235524


namespace cubic_polynomial_integer_roots_l2355_235587

theorem cubic_polynomial_integer_roots :
  ∀ (a c : ℤ), ∃ (x y z : ℤ),
    ∀ (X : ℤ), X^3 + a*X^2 - X + c = 0 ↔ (X = x ∨ X = y ∨ X = z) :=
by sorry

end cubic_polynomial_integer_roots_l2355_235587


namespace salary_percentage_difference_l2355_235511

theorem salary_percentage_difference (raja_salary : ℝ) (ram_salary : ℝ) :
  ram_salary = raja_salary * 1.25 →
  (raja_salary - ram_salary) / ram_salary = -0.2 := by
sorry

end salary_percentage_difference_l2355_235511


namespace remaining_toenail_capacity_l2355_235547

/- Jar capacity in terms of regular toenails -/
def jar_capacity : ℕ := 100

/- Size ratio of big toenails to regular toenails -/
def big_toenail_ratio : ℕ := 2

/- Number of big toenails already in the jar -/
def big_toenails_in_jar : ℕ := 20

/- Number of regular toenails already in the jar -/
def regular_toenails_in_jar : ℕ := 40

/- Theorem: The number of additional regular toenails that can fit in the jar is 20 -/
theorem remaining_toenail_capacity :
  jar_capacity - (big_toenails_in_jar * big_toenail_ratio + regular_toenails_in_jar) = 20 := by
  sorry

end remaining_toenail_capacity_l2355_235547


namespace function_values_imply_parameters_l2355_235551

theorem function_values_imply_parameters 
  (f : ℝ → ℝ) 
  (a θ : ℝ) 
  (h1 : ∀ x, f x = Real.sin (x + θ) + a * Real.cos (x + 2 * θ))
  (h2 : θ > -Real.pi / 2 ∧ θ < Real.pi / 2)
  (h3 : f (Real.pi / 2) = 0)
  (h4 : f Real.pi = 1) :
  a = -1 ∧ θ = -Real.pi / 6 := by
  sorry

end function_values_imply_parameters_l2355_235551


namespace least_multiple_divisible_l2355_235595

theorem least_multiple_divisible (x : ℕ) : 
  (∀ y : ℕ, y > 0 ∧ y < 57 → ¬(57 ∣ 23 * y)) ∧ (57 ∣ 23 * 57) := by
  sorry

end least_multiple_divisible_l2355_235595


namespace equation_solution_l2355_235509

theorem equation_solution : 
  ∃! x : ℝ, 2 * x - 1 = 3 * x + 2 ∧ x = -3 := by sorry

end equation_solution_l2355_235509


namespace min_value_of_expression_l2355_235529

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (1/a + a/b) ≥ 1 + 2*Real.sqrt 2 :=
sorry

end min_value_of_expression_l2355_235529


namespace third_number_problem_l2355_235554

theorem third_number_problem (first second third : ℕ) : 
  (3 * first + 3 * second + 3 * third + 11 = 170) →
  (first = 16) →
  (second = 17) →
  third = 20 := by
sorry

end third_number_problem_l2355_235554


namespace fraction_equivalence_l2355_235506

theorem fraction_equivalence : 
  ∀ (n : ℚ), (3 + n) / (4 + n) = 4 / 5 → n = 1 := by sorry

end fraction_equivalence_l2355_235506


namespace triangle_inequality_with_area_l2355_235582

/-- Triangle inequality theorem for sides and area -/
theorem triangle_inequality_with_area (a b c S : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : S > 0)
  (h5 : S = Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)) / 4) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S + (a - b)^2 + (b - c)^2 + (c - a)^2 ∧ 
  (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * S + (a - b)^2 + (b - c)^2 + (c - a)^2 ↔ a = b ∧ b = c) :=
by sorry

end triangle_inequality_with_area_l2355_235582


namespace max_contribution_scenario_l2355_235532

/-- Represents the maximum possible contribution by a single person given the total contribution and number of people. -/
def max_contribution (total : ℝ) (num_people : ℕ) (min_contribution : ℝ) : ℝ :=
  total - (min_contribution * (num_people - 1 : ℝ))

/-- Theorem stating the maximum possible contribution in the given scenario. -/
theorem max_contribution_scenario :
  max_contribution 20 10 1 = 11 := by
  sorry

end max_contribution_scenario_l2355_235532


namespace hypotenuse_square_l2355_235505

-- Define a right triangle with integer legs
def RightTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧ b = a + 1

-- Theorem statement
theorem hypotenuse_square (a : ℕ) :
  ∀ b c : ℕ, RightTriangle a b c → c^2 = 2*a^2 + 2*a + 1 := by
  sorry

end hypotenuse_square_l2355_235505


namespace product_remainder_l2355_235561

theorem product_remainder (a b c : ℕ) (h : a * b * c = 1225 * 1227 * 1229) : 
  (a * b * c) % 12 = 7 := by
sorry

end product_remainder_l2355_235561


namespace integer_solution_l2355_235568

theorem integer_solution (n : ℤ) : n + 5 > 7 ∧ -3*n > -15 → n = 3 ∨ n = 4 := by
  sorry

end integer_solution_l2355_235568


namespace photo_arrangement_count_l2355_235567

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of boys in the arrangement. -/
def num_boys : ℕ := 4

/-- The number of girls in the arrangement. -/
def num_girls : ℕ := 2

/-- The total number of people in the arrangement. -/
def total_people : ℕ := num_boys + num_girls

theorem photo_arrangement_count :
  arrangements total_people -
  arrangements (total_people - 1) -
  arrangements (total_people - 1) +
  arrangements (total_people - 2) = 504 := by sorry

end photo_arrangement_count_l2355_235567


namespace distance_from_origin_l2355_235531

/-- Given a point (x,y) in the first quadrant satisfying certain conditions,
    prove that its distance from the origin is √(233 + 12√7). -/
theorem distance_from_origin (x y : ℝ) (h1 : y = 14) (h2 : (x - 3)^2 + (y - 8)^2 = 64) (h3 : x > 3) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (233 + 12 * Real.sqrt 7) := by
  sorry

end distance_from_origin_l2355_235531


namespace mixture_price_calculation_l2355_235581

/-- Calculates the price of a mixture given the prices of two components and their ratio -/
def mixturePricePerKg (pricePeas : ℚ) (priceSoybean : ℚ) (ratioPeas : ℕ) (ratioSoybean : ℕ) : ℚ :=
  let totalParts := ratioPeas + ratioSoybean
  let totalPrice := pricePeas * ratioPeas + priceSoybean * ratioSoybean
  totalPrice / totalParts

theorem mixture_price_calculation (pricePeas priceSoybean : ℚ) (ratioPeas ratioSoybean : ℕ) :
  pricePeas = 16 →
  priceSoybean = 25 →
  ratioPeas = 2 →
  ratioSoybean = 1 →
  mixturePricePerKg pricePeas priceSoybean ratioPeas ratioSoybean = 19 := by
  sorry

end mixture_price_calculation_l2355_235581


namespace ninth_grade_students_count_l2355_235591

def total_payment : ℝ := 1936
def additional_sets : ℕ := 88
def discount_rate : ℝ := 0.2

theorem ninth_grade_students_count :
  ∃ x : ℕ, 
    (total_payment / x) * (1 - discount_rate) = total_payment / (x + additional_sets) ∧
    x = 352 := by
  sorry

end ninth_grade_students_count_l2355_235591


namespace special_quadratic_a_range_l2355_235590

/-- A quadratic function satisfying the given conditions -/
structure SpecialQuadratic where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  max_at_midpoint : ∀ a : ℝ, ∀ x : ℝ, f x ≤ f ((1 - 2*a) / 2)
  decreasing_away_from_zero : ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ + x₂ ≠ 0 → f x₁ > f x₂

/-- The range of a for a SpecialQuadratic function -/
theorem special_quadratic_a_range (sq : SpecialQuadratic) : 
  ∀ a : ℝ, (∀ x : ℝ, sq.f x ≤ sq.f ((1 - 2*a) / 2)) → a > 1/2 :=
sorry

end special_quadratic_a_range_l2355_235590


namespace linear_function_k_value_l2355_235502

/-- Given a linear function y = kx + 6 passing through the point (2, -2), prove that k = -4 -/
theorem linear_function_k_value :
  ∀ k : ℝ, (∀ x y : ℝ, y = k * x + 6) → -2 = k * 2 + 6 → k = -4 :=
by sorry

end linear_function_k_value_l2355_235502


namespace triangle_point_distance_height_inequality_l2355_235549

/-- Given a triangle and a point M inside it, this theorem states that the sum of the α-th powers
    of the ratios of distances from M to the sides to the corresponding heights of the triangle
    is always greater than or equal to 1/3ᵅ⁻¹, for α ≥ 1. -/
theorem triangle_point_distance_height_inequality
  (α : ℝ) (h_α : α ≥ 1)
  (k₁ k₂ k₃ h₁ h₂ h₃ : ℝ)
  (h_positive : k₁ > 0 ∧ k₂ > 0 ∧ k₃ > 0 ∧ h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0)
  (h_sum : k₁/h₁ + k₂/h₂ + k₃/h₃ = 1) :
  (k₁/h₁)^α + (k₂/h₂)^α + (k₃/h₃)^α ≥ 1/(3^(α-1)) := by
  sorry

end triangle_point_distance_height_inequality_l2355_235549


namespace linear_function_increases_iff_positive_slope_increasing_linear_function_k_equals_four_l2355_235556

/-- A linear function y = mx + b increases if and only if its slope m is positive -/
theorem linear_function_increases_iff_positive_slope {m b : ℝ} :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → m * x₁ + b < m * x₂ + b) ↔ m > 0 := by sorry

/-- For the function y = (k - 3)x + 2, if y increases as x increases, then k = 4 -/
theorem increasing_linear_function_k_equals_four (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (k - 3) * x₁ + 2 < (k - 3) * x₂ + 2) → k = 4 := by sorry

end linear_function_increases_iff_positive_slope_increasing_linear_function_k_equals_four_l2355_235556


namespace biology_marks_proof_l2355_235557

def english_marks : ℕ := 86
def math_marks : ℕ := 85
def physics_marks : ℕ := 92
def chemistry_marks : ℕ := 87
def average_marks : ℕ := 89
def total_subjects : ℕ := 5

def calculate_biology_marks (eng : ℕ) (math : ℕ) (phys : ℕ) (chem : ℕ) (avg : ℕ) (total : ℕ) : ℕ :=
  avg * total - (eng + math + phys + chem)

theorem biology_marks_proof :
  calculate_biology_marks english_marks math_marks physics_marks chemistry_marks average_marks total_subjects = 95 := by
  sorry

end biology_marks_proof_l2355_235557


namespace linear_equation_solution_l2355_235559

theorem linear_equation_solution : 
  ∃ x : ℚ, (x - 75) / 4 = (5 - 3 * x) / 7 ∧ x = 545 / 19 := by sorry

end linear_equation_solution_l2355_235559


namespace two_digit_numbers_with_property_three_digit_numbers_with_property_exists_infinite_sequence_l2355_235517

-- Define a function to check if a number has the desired property
def has_property (n : ℕ) (base : ℕ) : Prop :=
  n^2 % base = n

-- Theorem for two-digit numbers
theorem two_digit_numbers_with_property :
  ∃ (A B : ℕ), A ≠ B ∧ 10 ≤ A ∧ A < 100 ∧ 10 ≤ B ∧ B < 100 ∧
  has_property A 100 ∧ has_property B 100 ∧
  ∀ (C : ℕ), 10 ≤ C ∧ C < 100 ∧ has_property C 100 → (C = A ∨ C = B) :=
sorry

-- Theorem for three-digit numbers
theorem three_digit_numbers_with_property :
  ∃ (A B : ℕ), A ≠ B ∧ 100 ≤ A ∧ A < 1000 ∧ 100 ≤ B ∧ B < 1000 ∧
  has_property A 1000 ∧ has_property B 1000 ∧
  ∀ (C : ℕ), 100 ≤ C ∧ C < 1000 ∧ has_property C 1000 → (C = A ∨ C = B) :=
sorry

-- Define a function to represent a number from a sequence of digits
def number_from_sequence (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * 10 + a (n - 1 - i)) 0

-- Theorem for the existence of an infinite sequence
theorem exists_infinite_sequence :
  ∃ (a : ℕ → ℕ), ∀ (n : ℕ), has_property (number_from_sequence a n) (10^n) ∧
  ¬(a 0 = 1 ∧ ∀ (k : ℕ), k > 0 → a k = 0) :=
sorry

end two_digit_numbers_with_property_three_digit_numbers_with_property_exists_infinite_sequence_l2355_235517


namespace fixed_point_parabola_l2355_235594

theorem fixed_point_parabola (m : ℝ) : 
  let f : ℝ → ℝ := λ x => 5 * x^2 + m * x + 3 * m
  f (-3) = 45 := by sorry

end fixed_point_parabola_l2355_235594


namespace root_is_factor_l2355_235508

theorem root_is_factor (P : ℝ → ℝ) (a : ℝ) :
  (P a = 0) → ∃ Q : ℝ → ℝ, ∀ x, P x = (x - a) * Q x := by
  sorry

end root_is_factor_l2355_235508


namespace max_quadrilateral_intersections_l2355_235514

/-- A quadrilateral is a polygon with 4 sides -/
def Quadrilateral : Type := Unit

/-- The number of sides in a quadrilateral -/
def num_sides (q : Quadrilateral) : ℕ := 4

/-- The maximum number of intersection points between two quadrilaterals -/
def max_intersection_points (q1 q2 : Quadrilateral) : ℕ :=
  num_sides q1 * num_sides q2

theorem max_quadrilateral_intersections :
  ∀ (q1 q2 : Quadrilateral), max_intersection_points q1 q2 = 16 := by
  sorry

end max_quadrilateral_intersections_l2355_235514


namespace quadratic_form_ratio_l2355_235580

theorem quadratic_form_ratio (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + 784*x + 500
  ∃ b c : ℝ, (∀ x, f x = (x + b)^2 + c) ∧ c / b = -391 := by
sorry

end quadratic_form_ratio_l2355_235580


namespace simplify_and_rationalize_l2355_235530

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 7 / Real.sqrt 8) * (Real.sqrt 9 / Real.sqrt 10) = Real.sqrt 210 / 8 := by
  sorry

end simplify_and_rationalize_l2355_235530


namespace min_value_theorem_l2355_235575

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (1 / x + 4 / y) ≥ 3 ∧ ∃ x0 y0 : ℝ, x0 > 0 ∧ y0 > 0 ∧ x0 + y0 = 3 ∧ 1 / x0 + 4 / y0 = 3 :=
sorry

end min_value_theorem_l2355_235575


namespace percentage_of_500_l2355_235516

/-- Prove that 25% of Rs. 500 is equal to Rs. 125 -/
theorem percentage_of_500 : (500 : ℝ) * 0.25 = 125 := by
  sorry

end percentage_of_500_l2355_235516


namespace minimum_postage_l2355_235536

/-- Calculates the postage for a given weight in grams -/
def calculatePostage (weight : ℕ) : ℚ :=
  if weight ≤ 100 then
    (((weight - 1) / 20 + 1) * 8) / 10
  else
    4 + (((weight - 101) / 100 + 1) * 2)

/-- Calculates the total postage for two envelopes -/
def totalPostage (x : ℕ) : ℚ :=
  calculatePostage (12 * x + 4) + calculatePostage (12 * (11 - x) + 4)

theorem minimum_postage :
  ∃ x : ℕ, x ≤ 11 ∧ totalPostage x = 56/10 ∧ ∀ y : ℕ, y ≤ 11 → totalPostage y ≥ 56/10 :=
sorry

end minimum_postage_l2355_235536


namespace wall_bricks_count_l2355_235521

/-- Represents the time (in hours) it takes Ben to build the wall alone -/
def ben_time : ℝ := 12

/-- Represents the time (in hours) it takes Arya to build the wall alone -/
def arya_time : ℝ := 15

/-- Represents the reduction in combined output (in bricks per hour) due to chattiness -/
def chattiness_reduction : ℝ := 15

/-- Represents the time (in hours) it takes Ben and Arya to build the wall together -/
def combined_time : ℝ := 6

/-- Represents the number of bricks in the wall -/
def wall_bricks : ℝ := 900

theorem wall_bricks_count : 
  ben_time * arya_time * (1 / ben_time + 1 / arya_time - chattiness_reduction / wall_bricks) * combined_time = arya_time + ben_time := by
  sorry

end wall_bricks_count_l2355_235521


namespace least_expensive_route_cost_l2355_235579

/-- Represents the cost of travel between two cities -/
structure TravelCost where
  car : ℝ
  train : ℝ

/-- Calculates the travel cost between two cities given the distance -/
def calculateTravelCost (distance : ℝ) : TravelCost :=
  { car := 0.20 * distance,
    train := 150 + 0.15 * distance }

/-- Theorem: The least expensive route for Dereven's trip costs $37106.25 -/
theorem least_expensive_route_cost :
  let xz : ℝ := 5000
  let xy : ℝ := 5500
  let yz : ℝ := Real.sqrt (xy^2 - xz^2)
  let costXY := calculateTravelCost xy
  let costYZ := calculateTravelCost yz
  let costZX := calculateTravelCost xz
  min costXY.car costXY.train + min costYZ.car costYZ.train + min costZX.car costZX.train = 37106.25 := by
  sorry


end least_expensive_route_cost_l2355_235579


namespace circle_center_transformation_l2355_235599

def reflect_across_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 + d, p.2)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (3, -4)
  let reflected_center := reflect_across_x_axis initial_center
  let final_center := translate_right reflected_center 5
  final_center = (8, 4) := by
sorry

end circle_center_transformation_l2355_235599


namespace specific_sequence_terms_l2355_235589

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℕ
  last : ℕ
  diff : ℕ

/-- Calculates the number of terms in an arithmetic sequence -/
def numTerms (seq : ArithmeticSequence) : ℕ :=
  (seq.last - seq.first) / seq.diff + 1

theorem specific_sequence_terms : 
  let seq := ArithmeticSequence.mk 2 3007 5
  numTerms seq = 602 := by
  sorry

end specific_sequence_terms_l2355_235589


namespace quadratic_unique_solution_l2355_235577

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 - 30 * x + c = 0) →
  a + c = 41 →
  a < c →
  (a = (41 + Real.sqrt 781) / 2 ∧ c = (41 - Real.sqrt 781) / 2) := by
  sorry

end quadratic_unique_solution_l2355_235577


namespace original_group_size_l2355_235510

theorem original_group_size (n : ℕ) (W : ℝ) : 
  W = n * 35 ∧ 
  W + 40 = (n + 1) * 36 →
  n = 4 := by
sorry

end original_group_size_l2355_235510


namespace smallest_x_for_perfect_cube_l2355_235563

theorem smallest_x_for_perfect_cube : ∃ (x : ℕ+), 
  (∀ (y : ℕ+), ∃ (M : ℤ), 1800 * y = M^3 → x ≤ y) ∧
  (∃ (M : ℤ), 1800 * x = M^3) ∧
  x = 30 := by
  sorry

end smallest_x_for_perfect_cube_l2355_235563

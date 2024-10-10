import Mathlib

namespace distribute_five_to_three_l937_93725

/-- The number of ways to distribute n teachers to k schools --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n teachers to k schools,
    with each school receiving at least 1 teacher --/
def distributeAtLeastOne (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 150 ways to distribute 5 teachers to 3 schools,
    with each school receiving at least 1 teacher --/
theorem distribute_five_to_three :
  distributeAtLeastOne 5 3 = 150 := by sorry

end distribute_five_to_three_l937_93725


namespace three_divides_difference_l937_93721

/-- Represents a three-digit number ABC --/
structure ThreeDigitNumber where
  A : Nat
  B : Nat
  C : Nat
  A_is_digit : A < 10
  B_is_digit : B < 10
  C_is_digit : C < 10

/-- The value of a three-digit number ABC --/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.A + 10 * n.B + n.C

/-- The reversed value of a three-digit number ABC (i.e., CBA) --/
def reversed_value (n : ThreeDigitNumber) : Nat :=
  100 * n.C + 10 * n.B + n.A

theorem three_divides_difference (n : ThreeDigitNumber) (h : n.A ≠ n.C) :
  3 ∣ (value n - reversed_value n) := by
  sorry

end three_divides_difference_l937_93721


namespace problem_solution_l937_93740

/-- The function f(x) defined in the problem -/
noncomputable def f (a b x : ℝ) : ℝ := (a * Real.log x) / (x + 1) + b / x

/-- The theorem statement -/
theorem problem_solution :
  ∃ (a b : ℝ),
    (∀ x : ℝ, x > 0 → x + 2 * f a b x - 3 = 0 → x = 1) ∧
    (a = 1 ∧ b = 1) ∧
    (∀ k x : ℝ, k ≤ 0 → x > 0 → x ≠ 1 → f a b x > Real.log x / (x - 1) + k / x) :=
by
  sorry

end problem_solution_l937_93740


namespace distance_between_points_l937_93716

-- Define the initial meeting time in hours
def initial_meeting_time : ℝ := 4

-- Define the new meeting time in hours after speed increase
def new_meeting_time : ℝ := 3.5

-- Define the speed increase in km/h
def speed_increase : ℝ := 3

-- Define the function to calculate the distance
def calculate_distance (v_A v_B : ℝ) : ℝ := initial_meeting_time * (v_A + v_B)

-- Theorem statement
theorem distance_between_points : 
  ∃ (v_A v_B : ℝ), 
    v_A > 0 ∧ v_B > 0 ∧
    calculate_distance v_A v_B = 
    new_meeting_time * ((v_A + speed_increase) + (v_B + speed_increase)) ∧
    calculate_distance v_A v_B = 168 := by
  sorry

end distance_between_points_l937_93716


namespace landscape_breadth_l937_93777

theorem landscape_breadth (length width : ℝ) (playground_area : ℝ) : 
  width = 6 * length →
  playground_area = 4200 →
  length * width = 7 * playground_area →
  width = 420 := by
sorry

end landscape_breadth_l937_93777


namespace a_in_range_l937_93711

/-- Proposition p: for any real number x, ax^2 + ax + 1 > 0 always holds -/
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Proposition q: the equation x^2 - x + a = 0 has real roots with respect to x -/
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

/-- The set representing the range of a: (-∞, 0) ∪ (1/4, 4) -/
def range_a : Set ℝ := {a | a < 0 ∨ (1/4 < a ∧ a < 4)}

/-- Main theorem: If only one of prop_p and prop_q is true, then a is in range_a -/
theorem a_in_range (a : ℝ) : 
  (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a) → a ∈ range_a := by sorry

end a_in_range_l937_93711


namespace geometric_sequence_product_l937_93756

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r ∧ a n > 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  (a 1) * (a 19) = 16 →
  (a 1) + (a 19) = 10 →
  (a 8) * (a 10) * (a 12) = 64 := by
  sorry

end geometric_sequence_product_l937_93756


namespace banana_preference_percentage_l937_93707

/-- Represents the preference count for each fruit in the survey. -/
structure FruitPreferences where
  apple : ℕ
  banana : ℕ
  cherry : ℕ
  dragonfruit : ℕ

/-- Calculates the percentage of people who preferred a specific fruit. -/
def fruitPercentage (prefs : FruitPreferences) (fruitCount : ℕ) : ℚ :=
  (fruitCount : ℚ) / (prefs.apple + prefs.banana + prefs.cherry + prefs.dragonfruit : ℚ) * 100

/-- Theorem stating that the percentage of people who preferred Banana is 37.5%. -/
theorem banana_preference_percentage
  (prefs : FruitPreferences)
  (h1 : prefs.apple = 45)
  (h2 : prefs.banana = 75)
  (h3 : prefs.cherry = 30)
  (h4 : prefs.dragonfruit = 50) :
  fruitPercentage prefs prefs.banana = 37.5 := by
  sorry

#eval fruitPercentage ⟨45, 75, 30, 50⟩ 75

end banana_preference_percentage_l937_93707


namespace gate_width_scientific_notation_l937_93703

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  prop : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem gate_width_scientific_notation :
  toScientificNotation 0.000000007 = ScientificNotation.mk 7 (-9) sorry := by
  sorry

end gate_width_scientific_notation_l937_93703


namespace sum_c_d_equals_five_l937_93787

theorem sum_c_d_equals_five (a b c d : ℝ) 
  (h1 : a + b = 4)
  (h2 : b + c = 7)
  (h3 : a + d = 2) :
  c + d = 5 := by
  sorry

end sum_c_d_equals_five_l937_93787


namespace train_length_calculation_l937_93764

theorem train_length_calculation (passing_time man_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  passing_time = 8 →
  man_time = 8 →
  platform_length = 273 →
  platform_time = 20 →
  ∃ (train_length : ℝ), train_length = 182 ∧
    train_length / man_time = (train_length + platform_length) / platform_time :=
by sorry

end train_length_calculation_l937_93764


namespace retail_prices_correct_l937_93763

def calculate_retail_price (wholesale_price : ℚ) : ℚ :=
  let tax_rate : ℚ := 5 / 100
  let shipping_fee : ℚ := 10
  let profit_margin_rate : ℚ := 20 / 100
  let total_cost : ℚ := wholesale_price + (wholesale_price * tax_rate) + shipping_fee
  let profit_margin : ℚ := wholesale_price * profit_margin_rate
  total_cost + profit_margin

theorem retail_prices_correct :
  let machine1_wholesale : ℚ := 99
  let machine2_wholesale : ℚ := 150
  let machine3_wholesale : ℚ := 210
  (calculate_retail_price machine1_wholesale = 133.75) ∧
  (calculate_retail_price machine2_wholesale = 197.50) ∧
  (calculate_retail_price machine3_wholesale = 272.50) := by
  sorry

end retail_prices_correct_l937_93763


namespace lisa_weight_l937_93733

theorem lisa_weight (amy lisa : ℝ) 
  (h1 : amy + lisa = 240)
  (h2 : lisa - amy = lisa / 3) : 
  lisa = 144 := by sorry

end lisa_weight_l937_93733


namespace five_balls_three_boxes_l937_93735

/-- The number of ways to place n distinguishable objects into k distinguishable containers -/
def ways_to_place (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 3^5 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : ways_to_place 5 3 = 243 := by sorry

end five_balls_three_boxes_l937_93735


namespace least_sum_of_primes_l937_93773

theorem least_sum_of_primes (p q : ℕ) : 
  Nat.Prime p → 
  Nat.Prime q → 
  p > 1 → 
  q > 1 → 
  15 * (p^2 + 1) = 29 * (q^2 + 1) → 
  ∃ (p' q' : ℕ), Nat.Prime p' ∧ Nat.Prime q' ∧ p' > 1 ∧ q' > 1 ∧
    15 * (p'^2 + 1) = 29 * (q'^2 + 1) ∧
    p' + q' = 14 ∧
    ∀ (p'' q'' : ℕ), Nat.Prime p'' → Nat.Prime q'' → p'' > 1 → q'' > 1 →
      15 * (p''^2 + 1) = 29 * (q''^2 + 1) → p'' + q'' ≥ 14 :=
by sorry

end least_sum_of_primes_l937_93773


namespace machine_value_after_two_years_l937_93761

/-- Calculates the market value of a machine after a given number of years,
    given its initial value and yearly depreciation rate. -/
def marketValue (initialValue : ℝ) (depreciationRate : ℝ) (years : ℕ) : ℝ :=
  initialValue - (depreciationRate * initialValue * years)

/-- Theorem stating that a machine with an initial value of $8,000 and a yearly
    depreciation of 30% of its purchase price will have a market value of $3,200
    after 2 years. -/
theorem machine_value_after_two_years :
  marketValue 8000 0.3 2 = 3200 := by
  sorry

end machine_value_after_two_years_l937_93761


namespace odd_function_implies_a_equals_negative_one_l937_93737

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = (x+a+1)(x^2+a-1) -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  (x + a + 1) * (x^2 + a - 1)

theorem odd_function_implies_a_equals_negative_one :
  ∀ a : ℝ, IsOdd (f a) → a = -1 := by
  sorry

end odd_function_implies_a_equals_negative_one_l937_93737


namespace twenty_five_percent_less_than_80_l937_93730

theorem twenty_five_percent_less_than_80 (x : ℝ) : x + (1/4) * x = 60 → x = 48 := by
  sorry

end twenty_five_percent_less_than_80_l937_93730


namespace strawberry_preference_percentage_l937_93774

def total_responses : ℕ := 80 + 70 + 90 + 60 + 50
def strawberry_responses : ℕ := 90

def strawberry_percentage : ℚ :=
  (strawberry_responses : ℚ) / (total_responses : ℚ) * 100

theorem strawberry_preference_percentage :
  (strawberry_percentage : ℚ) = 25.71 := by sorry

end strawberry_preference_percentage_l937_93774


namespace intersection_P_complement_M_l937_93750

theorem intersection_P_complement_M (U : Set ℤ) (M P : Set ℤ) : 
  U = Set.univ ∧ 
  M = {1, 2} ∧ 
  P = {-2, -1, 0, 1, 2} →
  P ∩ (U \ M) = {-2, -1, 0} := by
sorry

end intersection_P_complement_M_l937_93750


namespace price_after_two_reductions_l937_93768

/-- Represents the relationship between the initial price, reduction percentage, and final price after two reductions. -/
theorem price_after_two_reductions 
  (initial_price : ℝ) 
  (reduction_percentage : ℝ) 
  (final_price : ℝ) 
  (h1 : initial_price = 2) 
  (h2 : 0 ≤ reduction_percentage ∧ reduction_percentage < 1) :
  final_price = initial_price * (1 - reduction_percentage)^2 :=
by sorry

#check price_after_two_reductions

end price_after_two_reductions_l937_93768


namespace ninth_term_value_l937_93786

/-- An arithmetic sequence {aₙ} with sum Sₙ of first n terms -/
def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ S n = n / 2 * (2 * a 1 + (n - 1) * d)

theorem ninth_term_value
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a S)
  (h_S8 : S 8 = 4 * a 1)
  (h_a7 : a 7 = -2)
  : a 9 = 2 := by
sorry

end ninth_term_value_l937_93786


namespace percentage_problem_l937_93744

theorem percentage_problem (p : ℝ) : p = 60 ↔ 180 * (1/3) - (p * 180 * (1/3) / 100) = 24 := by
  sorry

end percentage_problem_l937_93744


namespace min_distance_point_l937_93751

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2 * Real.log x + a

def g (x : ℝ) : ℝ := -x^2 + 3*x - 4

def h (x : ℝ) : ℝ := f 0 x - g x

theorem min_distance_point (t : ℝ) :
  t > 0 →
  (∀ x > 0, |h x| ≥ |h t|) →
  t = (3 + Real.sqrt 33) / 6 :=
sorry

end

end min_distance_point_l937_93751


namespace count_solution_pairs_l937_93789

/-- The number of pairs of positive integers (x, y) satisfying 2x + 3y = 2007 -/
def solution_count : ℕ := 334

/-- The predicate that checks if a pair of natural numbers satisfies the equation -/
def satisfies_equation (x y : ℕ) : Prop :=
  2 * x + 3 * y = 2007

theorem count_solution_pairs :
  (∃! n : ℕ, n = solution_count ∧
    ∃ s : Finset (ℕ × ℕ),
      s.card = n ∧
      (∀ p : ℕ × ℕ, p ∈ s ↔ (satisfies_equation p.1 p.2 ∧ p.1 > 0 ∧ p.2 > 0))) :=
sorry

end count_solution_pairs_l937_93789


namespace min_perimeter_triangle_l937_93792

theorem min_perimeter_triangle (d e f : ℕ) : 
  d > 0 → e > 0 → f > 0 →
  Real.cos (Real.arccos (1/2)) = d / (2 * e) →
  Real.cos (Real.arccos (3/5)) = e / (2 * f) →
  Real.cos (Real.arccos (-1/8)) = f / (2 * d) →
  d + e + f ≥ 33 :=
sorry

end min_perimeter_triangle_l937_93792


namespace quadratic_roots_sum_of_squares_l937_93781

theorem quadratic_roots_sum_of_squares (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (2 * x₁^2 + k * x₁ - 2 * k + 1 = 0) ∧ 
    (2 * x₂^2 + k * x₂ - 2 * k + 1 = 0) ∧ 
    (x₁ ≠ x₂) ∧
    (x₁^2 + x₂^2 = 29/4)) → 
  (k = 3 ∨ k = -11) :=
by sorry

end quadratic_roots_sum_of_squares_l937_93781


namespace min_value_of_ab_l937_93785

theorem min_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 8) :
  a * b ≥ 16 := by
  sorry

end min_value_of_ab_l937_93785


namespace base8_subtraction_and_conversion_l937_93753

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Subtracts two numbers in base 8 -/
def subtractBase8 (a b : ℕ) : ℕ := sorry

theorem base8_subtraction_and_conversion :
  let a := 7463
  let b := 3254
  let result_base8 := subtractBase8 a b
  let result_base10 := base8ToBase10 result_base8
  result_base8 = 4207 ∧ result_base10 = 2183 := by sorry

end base8_subtraction_and_conversion_l937_93753


namespace matinee_customers_count_l937_93705

/-- Represents the revenue calculation for a movie theater. -/
def theater_revenue (matinee_customers : ℕ) : ℕ :=
  let matinee_price : ℕ := 5
  let evening_price : ℕ := 7
  let opening_night_price : ℕ := 10
  let popcorn_price : ℕ := 10
  let evening_customers : ℕ := 40
  let opening_night_customers : ℕ := 58
  let total_customers : ℕ := matinee_customers + evening_customers + opening_night_customers
  let popcorn_customers : ℕ := total_customers / 2

  matinee_price * matinee_customers +
  evening_price * evening_customers +
  opening_night_price * opening_night_customers +
  popcorn_price * popcorn_customers

/-- Theorem stating that the number of matinee customers is 32. -/
theorem matinee_customers_count : ∃ (n : ℕ), theater_revenue n = 1670 ∧ n = 32 :=
sorry

end matinee_customers_count_l937_93705


namespace square_area_from_adjacent_points_l937_93749

/-- Given two adjacent points (1,2) and (4,6) on a square in a Cartesian coordinate plane,
    the area of the square is 25. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 25 := by sorry

end square_area_from_adjacent_points_l937_93749


namespace gcd_of_160_200_360_l937_93775

theorem gcd_of_160_200_360 : Nat.gcd 160 (Nat.gcd 200 360) = 40 := by
  sorry

end gcd_of_160_200_360_l937_93775


namespace certain_number_is_thirty_l937_93769

theorem certain_number_is_thirty : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → ∃ (b : ℕ), k * n = b^2 → k ≥ 30) ∧
  (∃ (b : ℕ), 30 * n = b^2) ∧
  n = 30 := by
  sorry

end certain_number_is_thirty_l937_93769


namespace balloon_count_theorem_l937_93747

/-- Represents the number of balloons a person has for each color -/
structure BalloonCount where
  blue : ℕ
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- The total number of balloons for all people -/
def totalBalloons (people : List BalloonCount) : BalloonCount :=
  { blue := people.foldl (fun acc p => acc + p.blue) 0,
    red := people.foldl (fun acc p => acc + p.red) 0,
    green := people.foldl (fun acc p => acc + p.green) 0,
    yellow := people.foldl (fun acc p => acc + p.yellow) 0 }

theorem balloon_count_theorem (joan melanie eric : BalloonCount)
  (h_joan : joan = { blue := 40, red := 30, green := 0, yellow := 0 })
  (h_melanie : melanie = { blue := 41, red := 0, green := 20, yellow := 0 })
  (h_eric : eric = { blue := 0, red := 25, green := 0, yellow := 15 }) :
  totalBalloons [joan, melanie, eric] = { blue := 81, red := 55, green := 20, yellow := 15 } := by
  sorry

#check balloon_count_theorem

end balloon_count_theorem_l937_93747


namespace groceries_expense_l937_93784

def monthly_salary : ℕ := 20000
def savings_percentage : ℚ := 1/10
def savings_amount : ℕ := 2000
def rent : ℕ := 5000
def milk : ℕ := 1500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 2500

theorem groceries_expense (h1 : savings_amount = monthly_salary * savings_percentage) 
  (h2 : savings_amount = 2000) : 
  monthly_salary - (rent + milk + education + petrol + miscellaneous + savings_amount) = 6500 := by
  sorry

end groceries_expense_l937_93784


namespace population_average_age_l937_93724

theorem population_average_age 
  (ratio_women_men : ℚ) 
  (avg_age_women : ℝ) 
  (avg_age_men : ℝ) 
  (h1 : ratio_women_men = 7 / 5) 
  (h2 : avg_age_women = 38) 
  (h3 : avg_age_men = 36) : 
  (ratio_women_men * avg_age_women + avg_age_men) / (ratio_women_men + 1) = 37 + 1/6 := by
sorry

end population_average_age_l937_93724


namespace smallest_valid_number_l937_93780

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  (n % 11 = 0) ∧
  (∀ d : ℕ, d ≥ 1 ∧ d ≤ 9 → (∃! p : ℕ, p ≥ 0 ∧ p < 9 ∧ (n / 10^p) % 10 = d))

theorem smallest_valid_number :
  is_valid_number 123475869 ∧
  ∀ m : ℕ, is_valid_number m → m ≥ 123475869 :=
sorry

end smallest_valid_number_l937_93780


namespace ten_n_value_l937_93794

theorem ten_n_value (n : ℝ) (h : 2 * n = 14) : 10 * n = 70 := by sorry

end ten_n_value_l937_93794


namespace number_minus_division_equals_l937_93700

theorem number_minus_division_equals (x : ℝ) : x - (104 / 20.8) = 545 ↔ x = 550 := by
  sorry

end number_minus_division_equals_l937_93700


namespace rope_and_well_depth_l937_93776

/-- Given a rope of length L and a well of depth H, prove that if L/2 + 9 = H and L/3 + 2 = H, then L = 42 and H = 30. -/
theorem rope_and_well_depth (L H : ℝ) 
  (h1 : L/2 + 9 = H) 
  (h2 : L/3 + 2 = H) : 
  L = 42 ∧ H = 30 := by
sorry

end rope_and_well_depth_l937_93776


namespace remainder_of_A_mod_9_l937_93770

-- Define the arithmetic sequence
def arithmetic_sequence : List Nat :=
  List.range 502 |> List.map (fun k => 4 * k + 2)

-- Define the large number A as a string
def A : String :=
  arithmetic_sequence.foldl (fun acc x => acc ++ toString x) ""

-- Theorem statement
theorem remainder_of_A_mod_9 :
  (A.foldl (fun acc c => (10 * acc + c.toNat - '0'.toNat) % 9) 0) = 8 := by
  sorry

end remainder_of_A_mod_9_l937_93770


namespace inscribed_triangle_inequality_l937_93779

/-- A triangle with semiperimeter, inradius, and circumradius -/
structure Triangle where
  semiperimeter : ℝ
  inradius : ℝ
  circumradius : ℝ
  semiperimeter_pos : 0 < semiperimeter
  inradius_pos : 0 < inradius
  circumradius_pos : 0 < circumradius

/-- An inscribed triangle with semiperimeter -/
structure InscribedTriangle (T : Triangle) where
  semiperimeter : ℝ
  semiperimeter_pos : 0 < semiperimeter
  semiperimeter_le : semiperimeter ≤ T.semiperimeter

/-- The theorem stating the inequality for inscribed triangles -/
theorem inscribed_triangle_inequality (T : Triangle) (IT : InscribedTriangle T) :
  T.inradius / T.circumradius ≤ IT.semiperimeter / T.semiperimeter ∧ 
  IT.semiperimeter / T.semiperimeter ≤ 1 := by
  sorry

end inscribed_triangle_inequality_l937_93779


namespace red_peaches_count_l937_93790

theorem red_peaches_count (total_peaches : ℕ) (num_baskets : ℕ) (green_peaches : ℕ) :
  total_peaches = 10 →
  num_baskets = 1 →
  green_peaches = 6 →
  total_peaches - green_peaches = 4 :=
by
  sorry

end red_peaches_count_l937_93790


namespace razorback_shop_tshirt_revenue_l937_93722

/-- The amount of money made from each t-shirt -/
def tshirt_price : ℕ := 62

/-- The number of t-shirts sold -/
def tshirts_sold : ℕ := 183

/-- The total money made from selling t-shirts -/
def total_tshirt_money : ℕ := tshirt_price * tshirts_sold

theorem razorback_shop_tshirt_revenue : total_tshirt_money = 11346 := by
  sorry

end razorback_shop_tshirt_revenue_l937_93722


namespace conference_games_l937_93767

/-- Calculates the total number of games in a sports conference season -/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let divisions := total_teams / teams_per_division
  let intra_division_total := divisions * teams_per_division * (teams_per_division - 1) / 2 * intra_division_games
  let inter_division_total := total_teams * (total_teams - teams_per_division) / 2 * inter_division_games
  intra_division_total + inter_division_total

/-- The theorem stating the total number of games in the specific conference setup -/
theorem conference_games : total_games 18 9 3 2 = 378 := by
  sorry

end conference_games_l937_93767


namespace sin_810_degrees_l937_93719

theorem sin_810_degrees : Real.sin (810 * π / 180) = 1 := by sorry

end sin_810_degrees_l937_93719


namespace remainder_of_g_x12_div_g_x_l937_93778

/-- The polynomial g(x) = x^5 + x^4 + x^3 + x^2 + x + 1 -/
def g (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

/-- The theorem stating that the remainder of g(x^12) divided by g(x) is 6 -/
theorem remainder_of_g_x12_div_g_x :
  ∃ (q : ℝ → ℝ), g (x^12) = g x * q x + 6 := by
  sorry

end remainder_of_g_x12_div_g_x_l937_93778


namespace sum_of_factorization_coefficients_l937_93738

theorem sum_of_factorization_coefficients (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 9*x + 20 = (x + a) * (x + b)) →
  (∀ x : ℝ, x^2 + 7*x - 60 = (x + b) * (x - c)) →
  a + b + c = 14 := by
  sorry

end sum_of_factorization_coefficients_l937_93738


namespace tangent_line_to_parabola_l937_93701

/-- The value of k that makes the line 4x + 6y + k = 0 tangent to the parabola y^2 = 32x -/
def tangent_k : ℝ := 72

/-- The line equation -/
def line (x y k : ℝ) : Prop := 4 * x + 6 * y + k = 0

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 32 * x

/-- The tangency condition -/
def is_tangent (k : ℝ) : Prop :=
  ∃! (x y : ℝ), line x y k ∧ parabola x y

theorem tangent_line_to_parabola :
  is_tangent tangent_k :=
sorry

end tangent_line_to_parabola_l937_93701


namespace discount_percentage_decrease_l937_93704

theorem discount_percentage_decrease (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * (1 + 0.25)
  let decrease_percentage := (increased_price - original_price) / increased_price
  decrease_percentage = 0.2 := by
  sorry

end discount_percentage_decrease_l937_93704


namespace martha_cards_l937_93799

/-- The number of cards Martha ends up with after receiving more cards -/
def total_cards (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Martha ends up with 79 cards -/
theorem martha_cards : total_cards 3 76 = 79 := by
  sorry

end martha_cards_l937_93799


namespace profit_percentage_calculation_l937_93793

theorem profit_percentage_calculation (original_cost selling_price : ℝ) :
  original_cost = 3000 →
  selling_price = 3450 →
  (selling_price - original_cost) / original_cost * 100 = 15 :=
by sorry

end profit_percentage_calculation_l937_93793


namespace absolute_value_inequality_l937_93783

theorem absolute_value_inequality (x y : ℝ) (h : x * y < 0) : 
  |x + y| < |x - y| := by
  sorry

end absolute_value_inequality_l937_93783


namespace orthogonal_to_pencil_l937_93706

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A pencil of circles -/
structure PencilOfCircles where
  circles : Set Circle

/-- Two circles are orthogonal if the square of the distance between their centers
    is equal to the sum of the squares of their radii -/
def orthogonal (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = c1.radius^2 + c2.radius^2

theorem orthogonal_to_pencil
  (S : Circle) (P : PencilOfCircles) (S1 S2 : Circle)
  (h1 : S1 ∈ P.circles) (h2 : S2 ∈ P.circles) (h3 : S1 ≠ S2)
  (orth1 : orthogonal S S1) (orth2 : orthogonal S S2) :
  ∀ C ∈ P.circles, orthogonal S C :=
sorry

end orthogonal_to_pencil_l937_93706


namespace kims_candy_bars_l937_93766

/-- Calculates the number of weeks passed given the number of candy bars saved, 
    the number of candy bars received per week, and the number of weeks between eating candy bars. -/
def weeks_passed (candy_bars_saved : ℕ) (candy_bars_per_week : ℕ) (weeks_between_eating : ℕ) : ℕ :=
  let candy_bars_saved_per_cycle := candy_bars_per_week * weeks_between_eating - 1
  candy_bars_saved / candy_bars_saved_per_cycle * weeks_between_eating

/-- Theorem stating that given the conditions from Kim's candy bar problem, 
    the number of weeks passed is 16. -/
theorem kims_candy_bars : 
  weeks_passed 28 2 4 = 16 := by
  sorry

end kims_candy_bars_l937_93766


namespace jerry_debt_payment_l937_93732

/-- Jerry's debt payment problem -/
theorem jerry_debt_payment (total_debt : ℕ) (remaining_debt : ℕ) (payment_two_months_ago : ℕ) 
  (h1 : total_debt = 50)
  (h2 : remaining_debt = 23)
  (h3 : payment_two_months_ago = 12)
  (h4 : total_debt > remaining_debt)
  (h5 : total_debt - remaining_debt > payment_two_months_ago) :
  ∃ (payment_last_month : ℕ), 
    payment_last_month - payment_two_months_ago = 3 ∧
    payment_last_month > payment_two_months_ago ∧
    payment_last_month + payment_two_months_ago = total_debt - remaining_debt :=
by
  sorry


end jerry_debt_payment_l937_93732


namespace speed_ratio_l937_93743

/-- Two perpendicular lines intersecting at O with points A and B moving along them -/
structure PointMovement where
  O : ℝ × ℝ
  speedA : ℝ
  speedB : ℝ
  initialDistB : ℝ
  time1 : ℝ
  time2 : ℝ

/-- The conditions of the problem -/
def problem_conditions (pm : PointMovement) : Prop :=
  pm.O = (0, 0) ∧
  pm.speedA > 0 ∧
  pm.speedB > 0 ∧
  pm.initialDistB = 500 ∧
  pm.time1 = 2 ∧
  pm.time2 = 10 ∧
  pm.speedA * pm.time1 = pm.initialDistB - pm.speedB * pm.time1 ∧
  pm.speedA * pm.time2 = pm.speedB * pm.time2 - pm.initialDistB

/-- The theorem to be proved -/
theorem speed_ratio (pm : PointMovement) :
  problem_conditions pm → pm.speedA / pm.speedB = 2 / 3 := by
  sorry


end speed_ratio_l937_93743


namespace overall_loss_is_184_76_l937_93728

-- Define the structure for an item
structure Item where
  name : String
  price : ℝ
  currency : String
  tax_rate : ℝ
  discount_rate : ℝ
  profit_loss_rate : ℝ

-- Define the currency conversion rates
def conversion_rates : List (String × ℝ) :=
  [("USD", 75), ("EUR", 80), ("GBP", 100), ("JPY", 0.7)]

-- Define the items
def items : List Item :=
  [{ name := "grinder", price := 150, currency := "USD", tax_rate := 0.1, discount_rate := 0, profit_loss_rate := -0.04 },
   { name := "mobile_phone", price := 100, currency := "EUR", tax_rate := 0.15, discount_rate := 0.05, profit_loss_rate := 0.1 },
   { name := "laptop", price := 200, currency := "GBP", tax_rate := 0.08, discount_rate := 0, profit_loss_rate := -0.08 },
   { name := "camera", price := 12000, currency := "JPY", tax_rate := 0.05, discount_rate := 0.12, profit_loss_rate := 0.15 }]

-- Function to calculate the final price of an item in INR
def calculate_final_price (item : Item) (conversion_rates : List (String × ℝ)) : ℝ :=
  sorry

-- Function to calculate the overall profit or loss
def calculate_overall_profit_loss (items : List Item) (conversion_rates : List (String × ℝ)) : ℝ :=
  sorry

-- Theorem statement
theorem overall_loss_is_184_76 :
  calculate_overall_profit_loss items conversion_rates = -184.76 := by sorry

end overall_loss_is_184_76_l937_93728


namespace smallest_number_with_given_remainders_l937_93771

theorem smallest_number_with_given_remainders : ∃ (x : ℕ), 
  x > 0 ∧
  x % 6 = 2 ∧ 
  x % 5 = 3 ∧ 
  x % 7 = 4 ∧
  ∀ (y : ℕ), y > 0 → y % 6 = 2 → y % 5 = 3 → y % 7 = 4 → x ≤ y :=
by sorry

end smallest_number_with_given_remainders_l937_93771


namespace imaginary_part_of_z_l937_93755

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = Complex.abs (4 + 3*I)) :
  z.im = 4/5 := by
sorry

end imaginary_part_of_z_l937_93755


namespace colonization_combinations_l937_93765

/-- Represents the number of Earth-like planets -/
def earth_like_planets : Nat := 7

/-- Represents the number of Mars-like planets -/
def mars_like_planets : Nat := 8

/-- Represents the colonization units required for an Earth-like planet -/
def earth_like_units : Nat := 3

/-- Represents the colonization units required for a Mars-like planet -/
def mars_like_units : Nat := 1

/-- Represents the total available colonization units -/
def total_units : Nat := 21

/-- Calculates the number of different combinations of planets that can be occupied -/
def count_combinations : Nat := sorry

theorem colonization_combinations : count_combinations = 981 := by sorry

end colonization_combinations_l937_93765


namespace product_of_numbers_l937_93788

theorem product_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 16) 
  (sum_squares_eq : x^2 + y^2 = 200) : 
  x * y = 28 := by
sorry

end product_of_numbers_l937_93788


namespace power_mod_seven_l937_93717

theorem power_mod_seven : 2^19 % 7 = 2 := by
  sorry

end power_mod_seven_l937_93717


namespace geometry_relations_l937_93797

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line : Line → Line → Prop)

-- Define the parallel relation between two lines
variable (parallel_line : Line → Line → Prop)

-- Define the perpendicular relation between two planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (subset_line_plane : Line → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (m l : Line) (α β : Plane)
  (h1 : perp_line_plane m α)
  (h2 : subset_line_plane l β) :
  (parallel_plane α β → perp_line m l) ∧
  (parallel_line m l → perp_plane α β) :=
sorry

end geometry_relations_l937_93797


namespace odd_number_induction_l937_93712

theorem odd_number_induction (P : ℕ → Prop) 
  (base : P 1)
  (step : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) :
  ∀ n : ℕ, n ≥ 1 → Odd n → P n :=
sorry

end odd_number_induction_l937_93712


namespace max_radius_of_circle_max_radius_achieved_l937_93736

/-- A circle in a rectangular coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point lies on a circle -/
def pointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem max_radius_of_circle (c : Circle) :
  pointOnCircle c (8, 0) → pointOnCircle c (-8, 0) → c.radius ≤ 8 := by
  sorry

theorem max_radius_achieved (r : ℝ) :
  r ≤ 8 →
  ∃ c : Circle, pointOnCircle c (8, 0) ∧ pointOnCircle c (-8, 0) ∧ c.radius = r := by
  sorry

end max_radius_of_circle_max_radius_achieved_l937_93736


namespace solution_in_interval_l937_93702

theorem solution_in_interval :
  ∃ x₀ : ℝ, (Real.log x₀ + x₀ = 4) ∧ (2 < x₀) ∧ (x₀ < 3) := by
  sorry

#check solution_in_interval

end solution_in_interval_l937_93702


namespace sara_quarters_count_l937_93741

theorem sara_quarters_count (initial : Nat) (from_dad : Nat) (total : Nat) : 
  initial = 21 → from_dad = 49 → total = initial + from_dad → total = 70 :=
by sorry

end sara_quarters_count_l937_93741


namespace sock_order_ratio_l937_93731

def sock_order_problem (black_socks green_socks : ℕ) (price_green : ℝ) : Prop :=
  let price_black := 3 * price_green
  let original_cost := black_socks * price_black + green_socks * price_green
  let interchanged_cost := green_socks * price_black + black_socks * price_green
  black_socks = 5 ∧
  interchanged_cost = 1.8 * original_cost ∧
  (black_socks : ℝ) / green_socks = 3 / 11

theorem sock_order_ratio :
  ∃ (green_socks : ℕ) (price_green : ℝ),
    sock_order_problem 5 green_socks price_green :=
by sorry

end sock_order_ratio_l937_93731


namespace complement_of_union_l937_93708

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 2}
def B : Set Int := {x ∈ U | x^2 - 4*x + 3 = 0}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 0} := by sorry

end complement_of_union_l937_93708


namespace donnas_earnings_proof_l937_93772

/-- Calculates Donna's total earnings over 7 days based on her work schedule --/
def donnas_weekly_earnings (dog_walking_rate : ℚ) (dog_walking_hours : ℚ) 
  (card_shop_rate : ℚ) (card_shop_hours : ℚ) (card_shop_days : ℕ)
  (babysitting_rate : ℚ) (babysitting_hours : ℚ) : ℚ :=
  (dog_walking_rate * dog_walking_hours * 7) + 
  (card_shop_rate * card_shop_hours * card_shop_days) + 
  (babysitting_rate * babysitting_hours)

theorem donnas_earnings_proof : 
  donnas_weekly_earnings 10 2 12.5 2 5 10 4 = 305 := by
  sorry

end donnas_earnings_proof_l937_93772


namespace circles_internally_tangent_l937_93760

/-- Two circles are internally tangent if the distance between their centers
    plus the radius of the smaller circle equals the radius of the larger circle. -/
def internally_tangent (r₁ r₂ d : ℝ) : Prop :=
  d + min r₁ r₂ = max r₁ r₂

/-- The theorem states that two circles with radii 3 and 7, whose centers are 4 units apart,
    are internally tangent. -/
theorem circles_internally_tangent :
  let r₁ : ℝ := 3
  let r₂ : ℝ := 7
  let d : ℝ := 4
  internally_tangent r₁ r₂ d :=
by
  sorry


end circles_internally_tangent_l937_93760


namespace complex_magnitude_product_l937_93757

theorem complex_magnitude_product : Complex.abs ((7 - 24 * Complex.I) * (3 + 4 * Complex.I)) = 125 := by
  sorry

end complex_magnitude_product_l937_93757


namespace burger_cost_is_five_l937_93791

/-- The cost of a burger meal -/
def burger_meal_cost : ℝ := 9.50

/-- The cost of a kid's meal -/
def kids_meal_cost : ℝ := 5

/-- The cost of french fries -/
def fries_cost : ℝ := 3

/-- The cost of a soft drink -/
def drink_cost : ℝ := 3

/-- The cost of a kid's burger -/
def kids_burger_cost : ℝ := 3

/-- The cost of kid's french fries -/
def kids_fries_cost : ℝ := 2

/-- The cost of a kid's juice box -/
def kids_juice_cost : ℝ := 2

/-- The amount saved by buying meals instead of individual items -/
def savings : ℝ := 10

theorem burger_cost_is_five (burger_cost : ℝ) : burger_cost = 5 :=
  by sorry

end burger_cost_is_five_l937_93791


namespace product_one_when_equal_absolute_log_l937_93759

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem product_one_when_equal_absolute_log 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (hf : f a = f b) : 
  a * b = 1 := by
sorry

end product_one_when_equal_absolute_log_l937_93759


namespace division_problem_l937_93742

theorem division_problem : ∃ x : ℝ, (x / 1.33 = 48) ↔ (x = 63.84) := by sorry

end division_problem_l937_93742


namespace songs_per_album_l937_93746

/-- Given that Rachel bought 8 albums and a total of 16 songs, prove that each album has 2 songs. -/
theorem songs_per_album (num_albums : ℕ) (total_songs : ℕ) (h1 : num_albums = 8) (h2 : total_songs = 16) :
  total_songs / num_albums = 2 := by
  sorry

end songs_per_album_l937_93746


namespace min_c_value_l937_93796

theorem min_c_value (a b c : ℕ+) (h1 : a < b) (h2 : b < 2*b) (h3 : 2*b < c)
  (h4 : ∃! (x y : ℝ), 3*x + y = 3000 ∧ y = |x - a| + |x - b| + |x - 2*b| + |x - c|) :
  c ≥ 502 ∧ ∃ (a' b' : ℕ+), a' < b' ∧ b' < 2*b' ∧ 2*b' < 502 ∧
    ∃! (x y : ℝ), 3*x + y = 3000 ∧ y = |x - a'| + |x - b'| + |x - 2*b'| + |x - 502| :=
by sorry

end min_c_value_l937_93796


namespace system_of_equations_product_l937_93754

theorem system_of_equations_product (a b c d : ℚ) : 
  3*a + 4*b + 6*c + 8*d = 48 →
  4*(d+c) = b →
  4*b + 2*c = a →
  c - 2 = d →
  a * b * c * d = -1032192 / 1874161 := by
sorry

end system_of_equations_product_l937_93754


namespace imaginary_part_of_z_l937_93715

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I) :
  z.im = (Real.sqrt 2 + 1) / 2 := by
  sorry

end imaginary_part_of_z_l937_93715


namespace square_side_length_l937_93720

theorem square_side_length
  (a : ℝ) -- side length of the square
  (x : ℝ) -- one leg of the right triangle
  (b : ℝ) -- hypotenuse of the right triangle
  (h1 : 4 * a + 2 * x = 58) -- perimeter of rectangle
  (h2 : 2 * a + 2 * b + 2 * x = 60) -- perimeter of trapezoid
  (h3 : a^2 + x^2 = b^2) -- Pythagorean theorem
  : a = 12 := by
sorry

end square_side_length_l937_93720


namespace roots_in_arithmetic_progression_l937_93718

theorem roots_in_arithmetic_progression (m : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ x : ℝ, x^4 - (3*m + 2)*x^2 + m^2 = 0 ↔ x = a ∨ x = b ∨ x = -b ∨ x = -a) ∧
    (b - (-b) = -b - (-a) ∧ a - b = b - (-b)))
  ↔ m = 6 ∨ m = -6/19 := by sorry

end roots_in_arithmetic_progression_l937_93718


namespace egg_count_proof_l937_93727

def initial_eggs : ℕ := 47
def eggs_added : ℝ := 5.0
def final_eggs : ℕ := 52

theorem egg_count_proof : 
  (initial_eggs : ℝ) + eggs_added = final_eggs := by sorry

end egg_count_proof_l937_93727


namespace range_of_a_l937_93729

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + x + (1/2 : ℝ) > 0) → a > 1/2 := by
  sorry

end range_of_a_l937_93729


namespace smallest_n_value_l937_93734

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 2020 →
  c > a → c > b →
  c > a + 100 →
  a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  ¬(10 ∣ m) →
  (∀ k, k < n → ∃ l, a.factorial * b.factorial * c.factorial = l * (10 ^ k) ∧ 10 ∣ l) →
  n = 499 := by
sorry

end smallest_n_value_l937_93734


namespace constant_term_binomial_expansion_l937_93795

/-- The constant term in the expansion of (3x + 2/x)^8 is 90720 -/
theorem constant_term_binomial_expansion : 
  (Finset.sum (Finset.range 9) (fun k => Nat.choose 8 k * 3^(8-k) * 2^k * (if k = 4 then 1 else 0))) = 90720 := by
  sorry

end constant_term_binomial_expansion_l937_93795


namespace log_inequality_solution_set_complex_expression_evaluation_l937_93739

-- Part 1
theorem log_inequality_solution_set (x : ℝ) :
  (Real.log (x + 2) / Real.log (1/2) > -3) ↔ (-2 < x ∧ x < 6) :=
sorry

-- Part 2
theorem complex_expression_evaluation :
  (1/8)^(1/3) * (-7/6)^0 + 8^0.25 * 2^(1/4) + (2^(1/3) * 3^(1/2))^6 = 221/2 :=
sorry

end log_inequality_solution_set_complex_expression_evaluation_l937_93739


namespace product_mod_600_l937_93709

theorem product_mod_600 : (1853 * 2101) % 600 = 553 := by sorry

end product_mod_600_l937_93709


namespace valid_tiling_characterization_l937_93745

/-- A tetromino type -/
inductive Tetromino
| T
| Square

/-- Represents a tiling of an n × n field -/
structure Tiling (n : ℕ) where
  pieces : List (Tetromino × ℕ × ℕ)  -- List of (type, row, col) for each piece
  no_gaps : Sorry
  no_overlaps : Sorry
  covers_field : Sorry
  odd_squares : Sorry  -- The number of square tetrominoes is odd

/-- Main theorem: Characterization of valid n for tiling -/
theorem valid_tiling_characterization (n : ℕ) :
  (∃ (t : Tiling n), True) ↔ (∃ (k : ℕ), n = 2 * k ∧ k % 2 = 1) :=
sorry

end valid_tiling_characterization_l937_93745


namespace mans_speed_against_current_l937_93798

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
theorem mans_speed_against_current
  (speed_with_current : ℝ)
  (current_speed : ℝ)
  (h1 : speed_with_current = 18)
  (h2 : current_speed = 3.4) :
  speed_with_current - 2 * current_speed = 11.2 :=
by sorry

end mans_speed_against_current_l937_93798


namespace rice_mixture_price_l937_93714

/-- Proves that mixing rice at given prices in a specific ratio results in the desired mixture price -/
theorem rice_mixture_price (price1 price2 mixture_price : ℚ) (ratio1 ratio2 : ℕ) : 
  price1 = 31/10 ∧ price2 = 36/10 ∧ mixture_price = 13/4 ∧ ratio1 = 3 ∧ ratio2 = 7 →
  (ratio1 : ℚ) * price1 + (ratio2 : ℚ) * price2 = (ratio1 + ratio2 : ℚ) * mixture_price :=
by
  sorry

#check rice_mixture_price

end rice_mixture_price_l937_93714


namespace probability_of_twin_primes_l937_93782

/-- The set of prime numbers not exceeding 30 -/
def primes_le_30 : Finset Nat := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

/-- A pair of primes (p, q) is considered a twin prime pair if q = p + 2 -/
def is_twin_prime_pair (p q : Nat) : Prop :=
  p ∈ primes_le_30 ∧ q ∈ primes_le_30 ∧ q = p + 2

/-- The set of twin prime pairs among primes not exceeding 30 -/
def twin_prime_pairs : Finset (Nat × Nat) :=
  {(3, 5), (5, 7), (11, 13), (17, 19)}

theorem probability_of_twin_primes :
  (twin_prime_pairs.card : Rat) / (Nat.choose primes_le_30.card 2 : Rat) = 4 / 45 :=
sorry

end probability_of_twin_primes_l937_93782


namespace bob_sandwich_options_l937_93723

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 6

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 4

/-- Represents whether Bob orders sandwiches with turkey and Swiss cheese. -/
def orders_turkey_swiss : Bool := false

/-- Represents whether Bob orders sandwiches with multigrain bread and turkey. -/
def orders_multigrain_turkey : Bool := false

/-- Calculates the number of sandwiches Bob can order. -/
def num_bob_sandwiches : ℕ := 
  num_breads * num_meats * num_cheeses - 
  (if orders_turkey_swiss then 0 else num_breads) - 
  (if orders_multigrain_turkey then 0 else num_cheeses)

/-- Theorem stating the number of different sandwiches Bob could order. -/
theorem bob_sandwich_options : num_bob_sandwiches = 111 := by
  sorry

end bob_sandwich_options_l937_93723


namespace germination_rate_proof_l937_93752

/-- The relative frequency of germinating seeds -/
def relative_frequency_germinating_seeds (total_seeds : ℕ) (non_germinating_seeds : ℕ) : ℚ :=
  (total_seeds - non_germinating_seeds : ℚ) / total_seeds

/-- Theorem: The relative frequency of germinating seeds in a sample of 1000 seeds, 
    where 90 seeds did not germinate, is equal to 0.91 -/
theorem germination_rate_proof :
  relative_frequency_germinating_seeds 1000 90 = 91 / 100 := by
  sorry

end germination_rate_proof_l937_93752


namespace cookie_ratio_l937_93713

def cookie_problem (clementine_cookies jake_cookies tory_cookies : ℕ) 
  (price_per_cookie total_money : ℚ) : Prop :=
  clementine_cookies = 72 ∧
  jake_cookies = 2 * clementine_cookies ∧
  price_per_cookie = 2 ∧
  total_money = 648 ∧
  price_per_cookie * (clementine_cookies + jake_cookies + tory_cookies) = total_money ∧
  tory_cookies * 2 = clementine_cookies + jake_cookies

theorem cookie_ratio (clementine_cookies jake_cookies tory_cookies : ℕ) 
  (price_per_cookie total_money : ℚ) :
  cookie_problem clementine_cookies jake_cookies tory_cookies price_per_cookie total_money →
  tory_cookies * 2 = clementine_cookies + jake_cookies :=
by
  sorry

end cookie_ratio_l937_93713


namespace custom_op_result_l937_93758

def custom_op (a b : ℚ) : ℚ := (a + b) / (a - b)

theorem custom_op_result : custom_op (custom_op 8 6) 12 = -19/5 := by
  sorry

end custom_op_result_l937_93758


namespace value_of_m_l937_93762

theorem value_of_m (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 4*x + m
  let g : ℝ → ℝ := λ x => x^2 - 2*x + 2*m
  3 * f 3 = g 3 → m = 12 := by
sorry

end value_of_m_l937_93762


namespace metallic_sheet_width_l937_93726

/-- Given a rectangular metallic sheet with length 48 meters, 
    from which squares of side 8 meters are cut from each corner to form a box,
    if the resulting box has a volume of 5632 cubic meters,
    then the width of the original sheet is 38 meters. -/
theorem metallic_sheet_width :
  ∀ (w : ℝ), 
    w > 0 →
    (48 - 2 * 8) * (w - 2 * 8) * 8 = 5632 →
    w = 38 := by
  sorry

end metallic_sheet_width_l937_93726


namespace basketball_league_games_l937_93748

/-- The number of games played in a basketball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) / 2 * games_per_pair

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team,
    the total number of games played is 180. -/
theorem basketball_league_games :
  total_games 10 4 = 180 := by sorry

end basketball_league_games_l937_93748


namespace boys_in_class_l937_93710

theorem boys_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 32) (h2 : ratio_girls = 3) (h3 : ratio_boys = 5) : 
  (total * ratio_boys) / (ratio_girls + ratio_boys) = 20 := by
  sorry

end boys_in_class_l937_93710

import Mathlib

namespace geometric_sequence_problem_l2081_208112

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  q > 0 →  -- positive common ratio
  a 1 = 2 →  -- given condition
  4 * a 2 * a 8 = (a 4) ^ 2 →  -- given condition
  a 3 = (1 : ℝ) / 2 := by  -- conclusion to prove
sorry

end geometric_sequence_problem_l2081_208112


namespace min_employees_for_tech_company_l2081_208150

/-- Calculates the minimum number of employees needed given the number of employees
    for hardware, software, and those working on both. -/
def min_employees (hardware : ℕ) (software : ℕ) (both : ℕ) : ℕ :=
  hardware + software - both

/-- Theorem stating that given 150 employees for hardware, 130 for software,
    and 50 for both, the minimum number of employees needed is 230. -/
theorem min_employees_for_tech_company :
  min_employees 150 130 50 = 230 := by
  sorry

#eval min_employees 150 130 50

end min_employees_for_tech_company_l2081_208150


namespace investment_income_is_500_l2081_208172

/-- Calculates the yearly income from investments given the total amount,
    amounts invested at different rates, and their corresponding interest rates. -/
def yearly_income (total : ℝ) (amount1 : ℝ) (rate1 : ℝ) (amount2 : ℝ) (rate2 : ℝ) (rate3 : ℝ) : ℝ :=
  amount1 * rate1 + amount2 * rate2 + (total - amount1 - amount2) * rate3

/-- Theorem stating that the yearly income from the given investment scenario is $500 -/
theorem investment_income_is_500 :
  yearly_income 10000 4000 0.05 3500 0.04 0.064 = 500 := by
  sorry

#eval yearly_income 10000 4000 0.05 3500 0.04 0.064

end investment_income_is_500_l2081_208172


namespace three_hour_therapy_charge_l2081_208101

def therapy_charge (first_hour_rate : ℕ) (additional_hour_rate : ℕ) (hours : ℕ) : ℕ :=
  first_hour_rate + (hours - 1) * additional_hour_rate

theorem three_hour_therapy_charge :
  ∀ (first_hour_rate additional_hour_rate : ℕ),
    first_hour_rate = additional_hour_rate + 20 →
    therapy_charge first_hour_rate additional_hour_rate 5 = 300 →
    therapy_charge first_hour_rate additional_hour_rate 3 = 188 :=
by
  sorry

#check three_hour_therapy_charge

end three_hour_therapy_charge_l2081_208101


namespace min_value_of_f_min_value_attained_l2081_208131

/-- The function f(x) = (x^2 + 2) / √(x^2 + 1) has a minimum value of 2 for all real x -/
theorem min_value_of_f (x : ℝ) : (x^2 + 2) / Real.sqrt (x^2 + 1) ≥ 2 := by
  sorry

/-- The minimum value 2 is attained when x = 0 -/
theorem min_value_attained : ∃ x : ℝ, (x^2 + 2) / Real.sqrt (x^2 + 1) = 2 := by
  sorry

end min_value_of_f_min_value_attained_l2081_208131


namespace left_square_side_length_l2081_208196

theorem left_square_side_length :
  ∀ (x : ℝ),
  (∃ (y z : ℝ),
    y = x + 17 ∧
    z = y - 6 ∧
    x + y + z = 52) →
  x = 8 := by
  sorry

end left_square_side_length_l2081_208196


namespace second_boy_speed_l2081_208102

/-- Given two boys walking in the same direction, with the first boy's speed at 5.5 kmph,
    and they are 20 km apart after 10 hours, prove that the second boy's speed is 7.5 kmph. -/
theorem second_boy_speed (v : ℝ) 
  (h1 : (v - 5.5) * 10 = 20) : v = 7.5 := by
  sorry

end second_boy_speed_l2081_208102


namespace correct_weight_proof_l2081_208147

/-- Proves that the correct weight is 65 kg given the problem conditions -/
theorem correct_weight_proof (n : ℕ) (initial_avg : ℝ) (misread_weight : ℝ) (correct_avg : ℝ) :
  n = 20 ∧ initial_avg = 58.4 ∧ misread_weight = 56 ∧ correct_avg = 58.85 →
  ∃ (correct_weight : ℝ),
    correct_weight = 65 ∧
    n * correct_avg = (n * initial_avg - misread_weight + correct_weight) :=
by sorry


end correct_weight_proof_l2081_208147


namespace expression_equality_l2081_208183

theorem expression_equality (x y : ℝ) (h : x - 2*y + 2 = 5) : 2*x - 4*y - 1 = 5 := by
  sorry

end expression_equality_l2081_208183


namespace deposit_calculation_l2081_208175

theorem deposit_calculation (total_cost remaining_amount : ℝ) 
  (h1 : total_cost = 550)
  (h2 : remaining_amount = 495)
  (h3 : remaining_amount = total_cost - 0.1 * total_cost) :
  0.1 * total_cost = 55 := by
  sorry

end deposit_calculation_l2081_208175


namespace task_completion_probability_l2081_208117

theorem task_completion_probability
  (p_task2 : ℝ)
  (p_task1_not_task2 : ℝ)
  (h1 : p_task2 = 3 / 5)
  (h2 : p_task1_not_task2 = 0.15)
  (h_independent : True)  -- Representing the independence of tasks
  : ∃ (p_task1 : ℝ), p_task1 = 0.375 :=
by sorry

end task_completion_probability_l2081_208117


namespace quadratic_inequality_roots_l2081_208193

theorem quadratic_inequality_roots (b : ℝ) : 
  (∀ x, -x^2 + b*x - 7 < 0 ↔ x < 2 ∨ x > 6) → b = 8 :=
by sorry

end quadratic_inequality_roots_l2081_208193


namespace coefficient_x4_in_polynomial_product_l2081_208134

theorem coefficient_x4_in_polynomial_product : 
  let p1 : Polynomial ℤ := X^5 - 4*X^4 + 3*X^3 - 2*X^2 + X - 1
  let p2 : Polynomial ℤ := 3*X^2 - X + 5
  (p1 * p2).coeff 4 = -13 := by
  sorry

end coefficient_x4_in_polynomial_product_l2081_208134


namespace factorial_8_divisors_l2081_208156

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- The number of positive divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factorial_8_divisors :
  numDivisors (factorial 8) = 96 := by
  sorry

end factorial_8_divisors_l2081_208156


namespace investment_more_profitable_l2081_208103

/-- Represents the initial price of buckwheat in rubles per kilogram -/
def initial_price : ℝ := 70

/-- Represents the final price of buckwheat in rubles per kilogram -/
def final_price : ℝ := 85

/-- Represents the annual interest rate for deposits in 2015 -/
def interest_rate_2015 : ℝ := 0.16

/-- Represents the annual interest rate for deposits in 2016 -/
def interest_rate_2016 : ℝ := 0.10

/-- Represents the annual interest rate for two-year deposits -/
def interest_rate_two_year : ℝ := 0.15

/-- Calculates the value after two years of annual deposits -/
def value_annual_deposits (initial : ℝ) : ℝ :=
  initial * (1 + interest_rate_2015) * (1 + interest_rate_2016)

/-- Calculates the value after a two-year deposit -/
def value_two_year_deposit (initial : ℝ) : ℝ :=
  initial * (1 + interest_rate_two_year) ^ 2

/-- Theorem stating that investing the initial price would yield more than the final price -/
theorem investment_more_profitable :
  (value_annual_deposits initial_price > final_price) ∧
  (value_two_year_deposit initial_price > final_price) := by
  sorry


end investment_more_profitable_l2081_208103


namespace first_three_digits_after_decimal_l2081_208168

theorem first_three_digits_after_decimal (x : ℝ) : 
  x = (10^2003 + 1)^(12/11) → 
  ∃ (n : ℕ), (x - n) * 1000 ≥ 909 ∧ (x - n) * 1000 < 910 := by
  sorry

end first_three_digits_after_decimal_l2081_208168


namespace a_minus_c_equals_three_l2081_208166

theorem a_minus_c_equals_three (a b c d : ℝ) 
  (h1 : a - b = c + d + 9)
  (h2 : a + b = c - d - 3) : 
  a - c = 3 := by
sorry

end a_minus_c_equals_three_l2081_208166


namespace no_fogh_prime_l2081_208197

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem no_fogh_prime :
  ¬∃ (f o g h : ℕ),
    is_digit f ∧ is_digit o ∧ is_digit g ∧ is_digit h ∧
    f ≠ o ∧ f ≠ g ∧ f ≠ h ∧ o ≠ g ∧ o ≠ h ∧ g ≠ h ∧
    (1000 * f + 100 * o + 10 * g + h ≥ 1000) ∧
    (1000 * f + 100 * o + 10 * g + h < 10000) ∧
    is_prime (1000 * f + 100 * o + 10 * g + h) ∧
    (1000 * f + 100 * o + 10 * g + h) * (f * o * g * h) = (1000 * f + 100 * o + 10 * g + h) :=
sorry

end no_fogh_prime_l2081_208197


namespace blue_pens_count_l2081_208160

theorem blue_pens_count (total_pens : ℕ) (red_pens : ℕ) (black_pens : ℕ) (blue_pens : ℕ) 
  (h1 : total_pens = 31)
  (h2 : total_pens = red_pens + black_pens + blue_pens)
  (h3 : black_pens = red_pens + 5)
  (h4 : blue_pens = 2 * black_pens) :
  blue_pens = 18 := by
  sorry

end blue_pens_count_l2081_208160


namespace rocket_max_height_l2081_208152

/-- The height of a rocket as a function of time -/
def rocket_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 50

/-- Theorem stating that the maximum height of the rocket is 175 meters -/
theorem rocket_max_height :
  ∃ (max_height : ℝ), max_height = 175 ∧ ∀ (t : ℝ), rocket_height t ≤ max_height :=
by sorry

end rocket_max_height_l2081_208152


namespace min_sqrt_equality_characterization_l2081_208169

theorem min_sqrt_equality_characterization (a b c : ℝ) 
  (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (min (Real.sqrt ((a * b + 1) / (a * b * c)))
       (min (Real.sqrt ((b * c + 1) / (a * b * c)))
            (Real.sqrt ((a * c + 1) / (a * b * c))))
   = Real.sqrt ((1 - a) / a) + Real.sqrt ((1 - b) / b) + Real.sqrt ((1 - c) / c))
  ↔ ∃ r : ℝ, r > 0 ∧ 
    ((a = 1 / (1 + r^2) ∧ b = 1 / (1 + 1/r^2) ∧ c = (r + 1/r)^2 / (1 + (r + 1/r)^2)) ∨
     (b = 1 / (1 + r^2) ∧ c = 1 / (1 + 1/r^2) ∧ a = (r + 1/r)^2 / (1 + (r + 1/r)^2)) ∨
     (c = 1 / (1 + r^2) ∧ a = 1 / (1 + 1/r^2) ∧ b = (r + 1/r)^2 / (1 + (r + 1/r)^2))) :=
by sorry

end min_sqrt_equality_characterization_l2081_208169


namespace agatha_bike_purchase_l2081_208192

/-- Given Agatha's bike purchase scenario, prove the remaining amount for seat and handlebar tape. -/
theorem agatha_bike_purchase (total_budget : ℕ) (frame_cost : ℕ) (front_wheel_cost : ℕ) :
  total_budget = 60 →
  frame_cost = 15 →
  front_wheel_cost = 25 →
  total_budget - (frame_cost + front_wheel_cost) = 20 :=
by sorry

end agatha_bike_purchase_l2081_208192


namespace problem_solution_l2081_208185

theorem problem_solution :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 ∧ x ≠ 3 →
    (3*x - 8) / (x - 1) - (x + 1) / x / ((x^2 - 1) / (x^2 - 3*x)) = (2*x - 5) / (x - 1)) ∧
  ((Real.sqrt 12 - (-1/2)⁻¹ - |Real.sqrt 3 + 3| + (2023 - Real.pi)^0) = Real.sqrt 3) ∧
  ((2*2 - 5) / (2 - 1) = -1) :=
by sorry

end problem_solution_l2081_208185


namespace expression_evaluation_l2081_208153

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = b - 4)
  (h2 : b = a + 4)
  (h3 : a = 3)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 10) / (c + 7) = 117 / 40 := by
  sorry

#check expression_evaluation

end expression_evaluation_l2081_208153


namespace first_robber_guarantee_l2081_208157

/-- Represents the coin division game between two robbers --/
structure CoinGame where
  totalCoins : ℕ
  maxBags : ℕ

/-- Represents the outcome of the game for the first robber --/
def firstRobberOutcome (game : CoinGame) (coinsPerBag : ℕ) : ℕ :=
  min (game.totalCoins - game.maxBags * coinsPerBag) (game.maxBags * coinsPerBag)

/-- The theorem stating the maximum guaranteed coins for the first robber --/
theorem first_robber_guarantee (game : CoinGame) : 
  game.totalCoins = 300 → game.maxBags = 11 → 
  ∃ (coinsPerBag : ℕ), firstRobberOutcome game coinsPerBag ≥ 146 := by
  sorry

#eval firstRobberOutcome { totalCoins := 300, maxBags := 11 } 14

end first_robber_guarantee_l2081_208157


namespace bluegrass_percentage_in_mixtureX_l2081_208181

-- Define the seed mixtures and their compositions
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

-- Define the given conditions
def mixtureX : SeedMixture := { ryegrass := 40, bluegrass := 0, fescue := 0 }
def mixtureY : SeedMixture := { ryegrass := 25, bluegrass := 0, fescue := 75 }

-- Define the combined mixture
def combinedMixture : SeedMixture := { ryegrass := 38, bluegrass := 0, fescue := 0 }

-- Weight percentage of mixture X in the combined mixture
def weightPercentageX : ℝ := 86.67

-- Theorem to prove
theorem bluegrass_percentage_in_mixtureX : mixtureX.bluegrass = 60 := by
  sorry

end bluegrass_percentage_in_mixtureX_l2081_208181


namespace additional_marbles_needed_l2081_208100

/-- The number of friends James has -/
def num_friends : ℕ := 15

/-- The initial number of marbles James has -/
def initial_marbles : ℕ := 80

/-- The function to calculate the sum of first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating the number of additional marbles needed -/
theorem additional_marbles_needed : 
  sum_first_n num_friends - initial_marbles = 40 := by
  sorry

end additional_marbles_needed_l2081_208100


namespace intersection_line_of_circles_l2081_208159

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
def circle2 (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

-- Define the line
def line (x y : ℝ) : Prop := 2*x + y = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → line x y :=
by sorry

end intersection_line_of_circles_l2081_208159


namespace hex_B1E_equals_2846_l2081_208198

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'B' => 11
  | '1' => 1
  | 'E' => 14
  | _ => 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.foldr (fun c acc => 16 * acc + hex_to_dec c) 0

/-- The hexadecimal number B1E is equal to 2846 in decimal -/
theorem hex_B1E_equals_2846 : hex_string_to_dec "B1E" = 2846 := by
  sorry

end hex_B1E_equals_2846_l2081_208198


namespace circle_ellipse_intersection_l2081_208179

-- Define the circle C
def C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 8

-- Define the point D
def D : ℝ × ℝ := (1, 0)

-- Define the ellipse E (trajectory of P)
def E (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the point C
def C_point : ℝ × ℝ := (-1, 0)

-- Define the perpendicular foot W
def W (x0 y0 : ℝ) : Prop := 
  ∃ (k1 k2 : ℝ), 
    (y0 = k1 * (x0 + 1)) ∧ 
    (y0 = k2 * (x0 - 1)) ∧ 
    (k1 * k2 = -1) ∧
    E x0 y0

-- Define the theorem
theorem circle_ellipse_intersection :
  ∀ (x0 y0 : ℝ), W x0 y0 →
    (x0^2 / 2 + y0^2 < 1) ∧
    (∃ (area : ℝ), area = 16/9 ∧ 
      ∀ (q r s t : ℝ × ℝ), 
        E q.1 q.2 → E r.1 r.2 → E s.1 s.2 → E t.1 t.2 →
        q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t →
        area ≤ (abs ((q.1 - s.1) * (r.2 - t.2) - (q.2 - s.2) * (r.1 - t.1))) / 2) :=
sorry

end circle_ellipse_intersection_l2081_208179


namespace no_real_roots_l2081_208121

theorem no_real_roots : ∀ x : ℝ, x^2 - x + 9 ≠ 0 := by
  sorry

end no_real_roots_l2081_208121


namespace beaumont_high_school_science_classes_beaumont_high_school_main_theorem_l2081_208173

/-- The number of players taking at least two sciences at Beaumont High School -/
theorem beaumont_high_school_science_classes (total_players : ℕ) 
  (biology_players : ℕ) (chemistry_players : ℕ) (physics_players : ℕ) 
  (all_three_players : ℕ) : ℕ :=
by
  sorry

/-- The main theorem about Beaumont High School science classes -/
theorem beaumont_high_school_main_theorem : 
  beaumont_high_school_science_classes 30 15 10 5 3 = 9 :=
by
  sorry

end beaumont_high_school_science_classes_beaumont_high_school_main_theorem_l2081_208173


namespace unique_integer_triangle_l2081_208124

/-- A triangle with integer sides and an altitude --/
structure IntegerTriangle where
  a : ℕ  -- side BC
  b : ℕ  -- side CA
  c : ℕ  -- side AB
  h : ℕ  -- altitude AD
  bd : ℕ -- length of BD
  dc : ℕ -- length of DC

/-- The triangle satisfies the given conditions --/
def satisfies_conditions (t : IntegerTriangle) : Prop :=
  ∃ (n : ℕ), t.h = n ∧ t.a = n + 1 ∧ t.b = n + 2 ∧ t.c = n + 3 ∧
  t.a^2 = t.bd^2 + t.h^2 ∧
  t.c^2 = (t.bd + t.dc)^2 + t.h^2

/-- The theorem stating the existence and uniqueness of the triangle --/
theorem unique_integer_triangle :
  ∃! (t : IntegerTriangle), satisfies_conditions t ∧ 
    t.a = 14 ∧ t.b = 13 ∧ t.c = 15 ∧ t.h = 12 :=
by sorry

end unique_integer_triangle_l2081_208124


namespace runners_photo_probability_l2081_208176

/-- Represents a runner on a circular track -/
structure Runner where
  lap_time : ℝ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the track and photo setup -/
structure TrackSetup where
  photo_fraction : ℝ
  photo_time : ℝ

/-- Calculates the probability of both runners being in the photo -/
def probability_both_in_photo (ellie sam : Runner) (setup : TrackSetup) : ℝ :=
  sorry

/-- The main theorem statement -/
theorem runners_photo_probability :
  let ellie : Runner := { lap_time := 120, direction := true }
  let sam : Runner := { lap_time := 75, direction := false }
  let setup : TrackSetup := { photo_fraction := 1/3, photo_time := 600 }
  probability_both_in_photo ellie sam setup = 5/12 := by
  sorry

end runners_photo_probability_l2081_208176


namespace complex_sum_real_l2081_208199

theorem complex_sum_real (a : ℝ) : 
  let z₁ : ℂ := (16 / (a + 5)) - (10 - a^2) * I
  let z₂ : ℂ := (2 / (1 - a)) + (2*a - 5) * I
  (z₁ + z₂).im = 0 → a = 3 :=
by sorry

end complex_sum_real_l2081_208199


namespace triangle_measure_l2081_208142

/-- Given an equilateral triangle with side length 7.5 meters, 
    prove that three times the square of the side length is 168.75 meters. -/
theorem triangle_measure (side_length : ℝ) : 
  side_length = 7.5 → 3 * (side_length ^ 2) = 168.75 := by
  sorry

#check triangle_measure

end triangle_measure_l2081_208142


namespace reciprocal_equality_implies_equality_l2081_208146

theorem reciprocal_equality_implies_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / a = 1 / b → a = b := by
  sorry

end reciprocal_equality_implies_equality_l2081_208146


namespace quadratic_real_solutions_m_range_l2081_208109

theorem quadratic_real_solutions_m_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * x + 1 = 0) → m ≤ 1 ∧ m ≠ 0 := by
  sorry

end quadratic_real_solutions_m_range_l2081_208109


namespace correct_number_of_selection_plans_l2081_208126

def number_of_people : ℕ := 6
def number_of_cities : ℕ := 4
def number_of_restricted_people : ℕ := 2

def selection_plans : ℕ := 240

theorem correct_number_of_selection_plans :
  (number_of_people.factorial / (number_of_people - number_of_cities).factorial) -
  (number_of_restricted_people * ((number_of_people - 1).factorial / (number_of_people - number_of_cities).factorial)) =
  selection_plans := by
  sorry

end correct_number_of_selection_plans_l2081_208126


namespace hcf_of_specific_numbers_l2081_208158

/-- Given two positive integers with a product of 363 and the greater number being 33,
    prove that their highest common factor (HCF) is 11. -/
theorem hcf_of_specific_numbers :
  ∀ A B : ℕ+,
  A * B = 363 →
  A = 33 →
  A > B →
  Nat.gcd A.val B.val = 11 := by
sorry

end hcf_of_specific_numbers_l2081_208158


namespace train_length_l2081_208120

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 60 * 1000 / 3600 →
  crossing_time = 16.7986561075114 →
  bridge_length = 170 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.0000000001 := by
  sorry

end train_length_l2081_208120


namespace a_share_calculation_l2081_208125

/-- Calculates the share of profit for a partner in a business partnership. -/
def calculateShare (investment totalInvestment totalProfit : ℚ) : ℚ :=
  (investment / totalInvestment) * totalProfit

theorem a_share_calculation (investmentA investmentB investmentC shareB : ℚ) 
  (h1 : investmentA = 15000)
  (h2 : investmentB = 21000)
  (h3 : investmentC = 27000)
  (h4 : shareB = 1540) : 
  calculateShare investmentA (investmentA + investmentB + investmentC) 
    ((investmentA + investmentB + investmentC) * shareB / investmentB) = 1100 := by
  sorry

end a_share_calculation_l2081_208125


namespace class_size_l2081_208116

theorem class_size (football : ℕ) (long_tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : football = 26)
  (h2 : long_tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 9) :
  football + long_tennis - both + neither = 38 := by
  sorry

end class_size_l2081_208116


namespace different_choices_four_two_l2081_208190

/-- The number of ways two people can choose courses differently from a set of courses -/
def differentChoices (totalCourses : ℕ) (coursesPerPerson : ℕ) : ℕ :=
  (totalCourses.choose coursesPerPerson)^2 - totalCourses.choose coursesPerPerson

/-- Theorem: Given 4 courses and 2 courses per person, there are 30 ways to choose differently -/
theorem different_choices_four_two : differentChoices 4 2 = 30 := by
  sorry

end different_choices_four_two_l2081_208190


namespace fifteenth_term_of_sequence_l2081_208135

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifteenth_term_of_sequence : 
  arithmetic_sequence 3 3 15 = 45 := by
sorry

end fifteenth_term_of_sequence_l2081_208135


namespace unique_even_square_Q_l2081_208122

/-- Definition of the polynomial Q --/
def Q (x : ℤ) : ℤ := x^4 + 6*x^3 + 11*x^2 + 3*x + 25

/-- Predicate for x being even --/
def is_even (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k

/-- Theorem stating that there exists exactly one even integer x such that Q(x) is a perfect square --/
theorem unique_even_square_Q : ∃! x : ℤ, is_even x ∧ ∃ y : ℤ, Q x = y^2 := by
  sorry

end unique_even_square_Q_l2081_208122


namespace solution_set_part_I_range_of_a_part_II_l2081_208111

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + 5*x

-- Theorem for part I
theorem solution_set_part_I :
  {x : ℝ | |x + 1| + 5*x ≤ 5*x + 3} = Set.Icc (-4) 2 := by sorry

-- Theorem for part II
theorem range_of_a_part_II :
  ∀ a : ℝ, (∀ x ≥ -1, f a x ≥ 0) ↔ (a ≥ 4 ∨ a ≤ -6) := by sorry

end solution_set_part_I_range_of_a_part_II_l2081_208111


namespace books_about_trains_l2081_208187

def books_about_animals : ℕ := 10
def books_about_space : ℕ := 1
def cost_per_book : ℕ := 16
def total_spent : ℕ := 224

theorem books_about_trains : ℕ := by
  sorry

end books_about_trains_l2081_208187


namespace count_arithmetic_mean_subsets_l2081_208177

/-- The number of three-element subsets of {1, 2, ..., n} where one element
    is the arithmetic mean of the other two. -/
def arithmeticMeanSubsets (n : ℕ) : ℕ :=
  (n / 2) * ((n - 1) / 2)

/-- Theorem stating that for any natural number n ≥ 3, the number of three-element
    subsets of {1, 2, ..., n} where one element is the arithmetic mean of the
    other two is equal to ⌊n/2⌋ * ⌊(n-1)/2⌋. -/
theorem count_arithmetic_mean_subsets (n : ℕ) (h : n ≥ 3) :
  arithmeticMeanSubsets n = (n / 2) * ((n - 1) / 2) := by
  sorry

#check count_arithmetic_mean_subsets

end count_arithmetic_mean_subsets_l2081_208177


namespace counterexample_exists_l2081_208104

theorem counterexample_exists : ∃ a : ℝ, (abs a > 2) ∧ (a ≤ 2) := by
  sorry

end counterexample_exists_l2081_208104


namespace flagpole_break_height_l2081_208143

theorem flagpole_break_height :
  let initial_height : ℝ := 10
  let horizontal_distance : ℝ := 3
  let break_height : ℝ := Real.sqrt 109 / 2
  (break_height^2 + horizontal_distance^2 = (initial_height - break_height)^2) ∧
  (2 * break_height = Real.sqrt (horizontal_distance^2 + initial_height^2)) :=
by sorry

end flagpole_break_height_l2081_208143


namespace f_of_negative_sqrt_three_equals_four_l2081_208105

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_of_negative_sqrt_three_equals_four :
  (∀ x, f (Real.tan x) = 1 / (Real.cos x)^2) →
  f (-Real.sqrt 3) = 4 := by
sorry

end f_of_negative_sqrt_three_equals_four_l2081_208105


namespace twelve_students_pairs_l2081_208130

/-- The number of unique pairs that can be formed from a group of n elements -/
def number_of_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of unique pairs from a group of 12 students is 66 -/
theorem twelve_students_pairs : number_of_pairs 12 = 66 := by
  sorry

end twelve_students_pairs_l2081_208130


namespace flour_yield_l2081_208138

theorem flour_yield (total : ℚ) : 
  (total - (1 / 10) * total = 1) → total = 10 / 9 := by
  sorry

end flour_yield_l2081_208138


namespace tank_fill_time_proof_l2081_208144

/-- The time (in hours) it takes to fill the tank with the leak -/
def fill_time_with_leak : ℝ := 11

/-- The time (in hours) it takes for the tank to become empty due to the leak -/
def empty_time_due_to_leak : ℝ := 110

/-- The time (in hours) it takes to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 10

theorem tank_fill_time_proof :
  (1 / fill_time_without_leak) - (1 / empty_time_due_to_leak) = (1 / fill_time_with_leak) :=
sorry

end tank_fill_time_proof_l2081_208144


namespace square_sum_zero_implies_both_zero_l2081_208114

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l2081_208114


namespace exactly_one_white_and_all_white_mutually_exclusive_but_not_complementary_l2081_208186

/-- Represents the outcome of drawing balls from a bag -/
inductive BallDraw
  | oneWhite
  | twoWhite
  | threeWhite
  | allBlack

/-- The set of all possible outcomes when drawing 3 balls from a bag with 3 white and 4 black balls -/
def allOutcomes : Set BallDraw := {BallDraw.oneWhite, BallDraw.twoWhite, BallDraw.threeWhite, BallDraw.allBlack}

/-- The event of drawing exactly one white ball -/
def exactlyOneWhite : Set BallDraw := {BallDraw.oneWhite}

/-- The event of drawing all white balls -/
def allWhite : Set BallDraw := {BallDraw.threeWhite}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set BallDraw) : Prop := A ∩ B = ∅

/-- Two events are complementary if their union is the set of all outcomes -/
def complementary (A B : Set BallDraw) : Prop := A ∪ B = allOutcomes

theorem exactly_one_white_and_all_white_mutually_exclusive_but_not_complementary :
  mutuallyExclusive exactlyOneWhite allWhite ∧ ¬complementary exactlyOneWhite allWhite :=
sorry

end exactly_one_white_and_all_white_mutually_exclusive_but_not_complementary_l2081_208186


namespace max_value_of_expression_l2081_208180

theorem max_value_of_expression (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (heq : 2*a + 3*b = 5) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 2*x + 3*y = 5 → (2*x + 2)*(3*y + 1) ≤ 16 :=
sorry

end max_value_of_expression_l2081_208180


namespace algebraic_simplification_l2081_208148

variable (a b m n : ℝ)

theorem algebraic_simplification :
  (5 * a * b^2 - 2 * a^2 * b + 3 * a * b^2 - a^2 * b - 4 * a * b^2 = 4 * a * b^2 - 3 * a^2 * b) ∧
  (-5 * m * n^2 - (2 * m^2 * n - 2 * (m^2 * n - 2 * m * n^2)) = -9 * m * n^2) := by
  sorry

end algebraic_simplification_l2081_208148


namespace students_per_class_l2081_208151

/-- Prove the number of students per class in a school's reading program -/
theorem students_per_class (c : ℕ) (h1 : c > 0) : 
  let books_per_student_per_year := 5 * 12
  let total_books_read := 60
  let s := total_books_read / (c * books_per_student_per_year)
  s = 1 / c := by
  sorry

end students_per_class_l2081_208151


namespace hiking_route_length_l2081_208178

/-- The total length of the hiking route in kilometers. -/
def total_length : ℝ := 150

/-- The initial distance walked on foot in kilometers. -/
def initial_walk : ℝ := 30

/-- The fraction of the remaining route traveled by raft. -/
def raft_fraction : ℝ := 0.2

/-- The multiplier for the second walking distance compared to the raft distance. -/
def second_walk_multiplier : ℝ := 1.5

/-- The speed of the truck in km/h. -/
def truck_speed : ℝ := 40

/-- The time spent on the truck in hours. -/
def truck_time : ℝ := 1.5

theorem hiking_route_length :
  initial_walk +
  raft_fraction * (total_length - initial_walk) +
  second_walk_multiplier * (raft_fraction * (total_length - initial_walk)) +
  truck_speed * truck_time = total_length := by sorry

end hiking_route_length_l2081_208178


namespace sequence_property_l2081_208195

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem sequence_property (a : ℕ → ℝ) 
    (h : ∀ n : ℕ, a (n + 1) + 1 = a n) : 
  is_arithmetic_sequence a (-1) := by
sorry

end sequence_property_l2081_208195


namespace max_visible_cuboids_6x6x6_l2081_208128

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube composed of smaller cuboids -/
structure CompositeCube where
  side_length : ℕ
  small_cuboid : Cuboid
  num_small_cuboids : ℕ

/-- Function to calculate the maximum number of visible small cuboids -/
def max_visible_cuboids (cube : CompositeCube) : ℕ :=
  sorry

/-- Theorem stating the maximum number of visible small cuboids for the given problem -/
theorem max_visible_cuboids_6x6x6 :
  let small_cuboid : Cuboid := ⟨3, 2, 1⟩
  let large_cube : CompositeCube := ⟨6, small_cuboid, 36⟩
  max_visible_cuboids large_cube = 31 :=
by sorry

end max_visible_cuboids_6x6x6_l2081_208128


namespace chandler_saves_for_bike_l2081_208119

/-- The number of weeks needed for Chandler to save enough money to buy the mountain bike -/
def weeks_to_save : ℕ → Prop :=
  λ w => 
    let bike_cost : ℕ := 600
    let birthday_money : ℕ := 60 + 40 + 20
    let weekly_earnings : ℕ := 20
    let weekly_expenses : ℕ := 4
    let weekly_savings : ℕ := weekly_earnings - weekly_expenses
    birthday_money + w * weekly_savings = bike_cost

theorem chandler_saves_for_bike : weeks_to_save 30 := by
  sorry

end chandler_saves_for_bike_l2081_208119


namespace smallest_whole_number_above_sum_sum_less_than_nineteen_nineteen_is_smallest_l2081_208163

theorem smallest_whole_number_above_sum : ℕ → Prop :=
  fun n => (∃ (m : ℕ), m > n ∧ 
    (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/6 : ℚ) + (6 + 1/12 : ℚ) < m) ∧
    (∀ (k : ℕ), k < n → 
    (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/6 : ℚ) + (6 + 1/12 : ℚ) ≥ k)

theorem sum_less_than_nineteen :
  (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/6 : ℚ) + (6 + 1/12 : ℚ) < 19 :=
by sorry

theorem nineteen_is_smallest : smallest_whole_number_above_sum 19 :=
by sorry

end smallest_whole_number_above_sum_sum_less_than_nineteen_nineteen_is_smallest_l2081_208163


namespace cubic_sum_of_roots_similar_triangles_sqrt_difference_l2081_208167

-- Problem 1
theorem cubic_sum_of_roots (p q : ℝ) : 
  p^2 - 3*p - 2 = 0 → q^2 - 3*q - 2 = 0 → p^3 + q^3 = 45 := by sorry

-- Problem 2
theorem similar_triangles (A H B C K : ℝ) :
  A - H = 45 → C - K = 36 → B - K = 12 → 
  (A - H) / (C - K) = (B - H) / (B - K) →
  B - H = 15 := by sorry

-- Problem 3
theorem sqrt_difference (x : ℝ) :
  Real.sqrt (2*x + 23) + Real.sqrt (2*x - 1) = 12 →
  Real.sqrt (2*x + 23) - Real.sqrt (2*x - 1) = 2 := by sorry

end cubic_sum_of_roots_similar_triangles_sqrt_difference_l2081_208167


namespace probability_closer_to_center_l2081_208191

/-- The probability that a randomly chosen point in a circular region with radius 3
    is closer to the center than to the boundary is 1/4. -/
theorem probability_closer_to_center (r : ℝ) (h : r = 3) : 
  (π * (r/2)^2) / (π * r^2) = 1/4 := by
  sorry

end probability_closer_to_center_l2081_208191


namespace yahs_to_bahs_1500_l2081_208132

/-- Conversion rates between bahs, rahs, and yahs -/
structure ConversionRates where
  bah_to_rah : ℚ
  rah_to_yah : ℚ

/-- Given conversion rates, calculate the number of bahs equivalent to a given number of yahs -/
def yahs_to_bahs (rates : ConversionRates) (yahs : ℚ) : ℚ :=
  yahs * rates.rah_to_yah⁻¹ * rates.bah_to_rah⁻¹

/-- Theorem stating the equivalence of 1500 yahs to 562.5 bahs given the specified conversion rates -/
theorem yahs_to_bahs_1500 :
  let rates : ConversionRates := ⟨16/10, 20/12⟩
  yahs_to_bahs rates 1500 = 562.5 := by
  sorry

end yahs_to_bahs_1500_l2081_208132


namespace total_pencils_l2081_208133

def mitchell_pencils : ℕ := 30

def antonio_pencils : ℕ := mitchell_pencils - mitchell_pencils * 20 / 100

theorem total_pencils : mitchell_pencils + antonio_pencils = 54 := by
  sorry

end total_pencils_l2081_208133


namespace badminton_players_l2081_208110

theorem badminton_players (total : ℕ) (tennis : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 28)
  (h2 : tennis = 19)
  (h3 : neither = 2)
  (h4 : both = 10) :
  ∃ badminton : ℕ, badminton = 17 ∧ 
    total = tennis + badminton - both + neither :=
by sorry

end badminton_players_l2081_208110


namespace cafe_staff_remaining_l2081_208139

/-- Calculates the total number of remaining staff given the initial numbers and dropouts. -/
def remaining_staff (initial_chefs initial_waiters chefs_dropout waiters_dropout : ℕ) : ℕ :=
  (initial_chefs - chefs_dropout) + (initial_waiters - waiters_dropout)

/-- Theorem stating that given the specific numbers in the problem, the total remaining staff is 23. -/
theorem cafe_staff_remaining :
  remaining_staff 16 16 6 3 = 23 := by
  sorry

end cafe_staff_remaining_l2081_208139


namespace sqrt_sum_equivalence_l2081_208137

theorem sqrt_sum_equivalence (n : ℝ) (h : Real.sqrt 15 = n) :
  Real.sqrt 0.15 + Real.sqrt 1500 = (101 / 10) * n := by
  sorry

end sqrt_sum_equivalence_l2081_208137


namespace remaining_money_l2081_208171

def initial_amount : ℚ := 2 * 20 + 2 * 10 + 3 * 5 + 2 * 1 + 4.5
def cake_cost : ℚ := 17.5
def gift_cost : ℚ := 12.7
def donation : ℚ := 5.3

theorem remaining_money :
  initial_amount - (cake_cost + gift_cost + donation) = 46 :=
by sorry

end remaining_money_l2081_208171


namespace bear_population_l2081_208188

/-- The number of black bears in the park -/
def black_bears : ℕ := 60

/-- The number of white bears in the park -/
def white_bears : ℕ := black_bears / 2

/-- The number of brown bears in the park -/
def brown_bears : ℕ := black_bears + 40

/-- The total population of bears in the park -/
def total_bears : ℕ := white_bears + black_bears + brown_bears

theorem bear_population : total_bears = 190 := by
  sorry

end bear_population_l2081_208188


namespace last_four_digits_of_3_24000_l2081_208189

theorem last_four_digits_of_3_24000 (h : 3^800 ≡ 1 [ZMOD 2000]) :
  3^24000 ≡ 1 [ZMOD 2000] := by sorry

end last_four_digits_of_3_24000_l2081_208189


namespace smaller_cup_radius_l2081_208174

/-- The radius of smaller hemisphere-shaped cups when water from a large hemisphere
    is evenly distributed. -/
theorem smaller_cup_radius (R : ℝ) (n : ℕ) (h1 : R = 2) (h2 : n = 64) :
  ∃ r : ℝ, r > 0 ∧ n * ((2/3) * Real.pi * r^3) = (2/3) * Real.pi * R^3 ∧ r = 1/2 := by
  sorry

end smaller_cup_radius_l2081_208174


namespace binomial_coefficient_arithmetic_sequence_l2081_208161

theorem binomial_coefficient_arithmetic_sequence (n : ℕ) : 
  (2 * Nat.choose n 9 = Nat.choose n 8 + Nat.choose n 10) ↔ (n = 14 ∨ n = 23) :=
sorry

end binomial_coefficient_arithmetic_sequence_l2081_208161


namespace shane_photos_january_l2081_208141

/-- Calculates the number of photos taken per day in January given the total number of photos
    in the first two months and the number of photos taken each week in February. -/
def photos_per_day_january (total_photos : ℕ) (photos_per_week_feb : ℕ) : ℕ :=
  let photos_feb := photos_per_week_feb * 4
  let photos_jan := total_photos - photos_feb
  photos_jan / 31

/-- Theorem stating that given 146 total photos in the first two months and 21 photos
    per week in February, Shane took 2 photos per day in January. -/
theorem shane_photos_january : photos_per_day_january 146 21 = 2 := by
  sorry

#eval photos_per_day_january 146 21

end shane_photos_january_l2081_208141


namespace johns_earnings_l2081_208106

theorem johns_earnings (new_earnings : ℝ) (increase_percentage : ℝ) 
  (h1 : new_earnings = 55) 
  (h2 : increase_percentage = 37.5) : 
  ∃ original_earnings : ℝ, 
    original_earnings * (1 + increase_percentage / 100) = new_earnings ∧ 
    original_earnings = 40 := by
  sorry

end johns_earnings_l2081_208106


namespace expression_evaluation_l2081_208182

theorem expression_evaluation : 200 * (200 - 2^3) - (200^2 - 2^4) = -1584 := by
  sorry

end expression_evaluation_l2081_208182


namespace polynomial_factorization_l2081_208123

theorem polynomial_factorization (p q : ℕ) (n : ℕ) (a : ℤ) :
  Prime p ∧ Prime q ∧ p ≠ q ∧ n ≥ 3 →
  (∃ (g h : Polynomial ℤ),
    (Polynomial.degree g > 0) ∧
    (Polynomial.degree h > 0) ∧
    (X^n + a * X^(n-1) + (p * q : ℤ) = g * h)) ↔
  (a = (-1)^n * (p * q : ℤ) + 1 ∨ a = -(p * q : ℤ) - 1) :=
by sorry

end polynomial_factorization_l2081_208123


namespace f_max_value_l2081_208165

/-- The function f(x) defined as |tx-2| - |tx+1| where t is a real number -/
def f (t : ℝ) (x : ℝ) : ℝ := |t*x - 2| - |t*x + 1|

/-- The maximum value of f(x) is 3 -/
theorem f_max_value (t : ℝ) : 
  ∃ (M : ℝ), M = 3 ∧ ∀ x, f t x ≤ M :=
sorry

end f_max_value_l2081_208165


namespace inscribed_circle_radius_l2081_208136

/-- Given three circles with radii r₁, r₂, r₃, where r₁ is the largest,
    the radius of the circle inscribed in the quadrilateral formed by
    the tangents as described in the problem is
    (r₁ * r₂ * r₃) / (r₁ * r₃ + r₁ * r₂ - r₂ * r₃). -/
theorem inscribed_circle_radius
  (r₁ r₂ r₃ : ℝ)
  (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₃ > 0)
  (h₄ : r₁ > r₂) (h₅ : r₁ > r₃) :
  ∃ (r : ℝ), r = (r₁ * r₂ * r₃) / (r₁ * r₃ + r₁ * r₂ - r₂ * r₃) ∧
  r > 0 :=
by sorry

end inscribed_circle_radius_l2081_208136


namespace multiply_44_22_l2081_208149

theorem multiply_44_22 : 44 * 22 = 88 * 11 := by
  sorry

end multiply_44_22_l2081_208149


namespace watch_cost_price_l2081_208154

-- Define the cost price of the watch
def cost_price : ℝ := 1166.67

-- Define the selling price at 10% loss
def selling_price_loss : ℝ := 0.90 * cost_price

-- Define the selling price at 2% gain
def selling_price_gain : ℝ := 1.02 * cost_price

-- Theorem statement
theorem watch_cost_price :
  (selling_price_loss = 0.90 * cost_price) ∧
  (selling_price_gain = 1.02 * cost_price) ∧
  (selling_price_gain = selling_price_loss + 140) →
  cost_price = 1166.67 := by
sorry

end watch_cost_price_l2081_208154


namespace intersection_of_A_and_B_l2081_208162

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 4} := by
  sorry

end intersection_of_A_and_B_l2081_208162


namespace janes_numbers_l2081_208118

def is_valid_number (n : ℕ) : Prop :=
  n % 180 = 0 ∧ n % 42 = 0 ∧ 500 < n ∧ n < 4000

theorem janes_numbers :
  {n : ℕ | is_valid_number n} = {1260, 2520, 3780} :=
sorry

end janes_numbers_l2081_208118


namespace orange_ring_weight_l2081_208145

/-- Given the weights of three rings (purple, white, and orange) that sum to a total weight,
    prove that the weight of the orange ring is equal to the total weight minus the sum of
    the purple and white ring weights. -/
theorem orange_ring_weight
  (purple_weight white_weight total_weight : ℚ)
  (h1 : purple_weight = 0.3333333333333333)
  (h2 : white_weight = 0.4166666666666667)
  (h3 : total_weight = 0.8333333333333334)
  (h4 : ∃ orange_weight : ℚ, purple_weight + white_weight + orange_weight = total_weight) :
  ∃ orange_weight : ℚ, orange_weight = total_weight - (purple_weight + white_weight) :=
by
  sorry


end orange_ring_weight_l2081_208145


namespace window_treatment_cost_for_three_windows_l2081_208108

/-- The cost of window treatments for a given number of windows -/
def window_treatment_cost (num_windows : ℕ) (sheer_cost drape_cost : ℚ) : ℚ :=
  num_windows * (sheer_cost + drape_cost)

/-- Theorem: The cost of window treatments for 3 windows with sheers at $40.00 and drapes at $60.00 is $300.00 -/
theorem window_treatment_cost_for_three_windows :
  window_treatment_cost 3 40 60 = 300 := by
  sorry

end window_treatment_cost_for_three_windows_l2081_208108


namespace shaded_area_rectangle_l2081_208140

theorem shaded_area_rectangle (length width : ℝ) (h1 : length = 8) (h2 : width = 4) : 
  let rectangle_area := length * width
  let triangle_area := (1 / 2) * length * width
  let shaded_area := rectangle_area - triangle_area
  shaded_area = 16 := by
sorry

end shaded_area_rectangle_l2081_208140


namespace perfect_square_condition_l2081_208127

theorem perfect_square_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ z : ℝ, x^2 + 2*x*y + y^2 - a*(x + y) + 25 = z^2) → 
  (a = 10 ∨ a = -10) :=
by sorry

end perfect_square_condition_l2081_208127


namespace photographer_choices_l2081_208194

theorem photographer_choices (n : ℕ) (k₁ k₂ : ℕ) (h₁ : n = 7) (h₂ : k₁ = 4) (h₃ : k₂ = 5) :
  Nat.choose n k₁ + Nat.choose n k₂ = 56 :=
by sorry

end photographer_choices_l2081_208194


namespace jackson_inbox_problem_l2081_208115

theorem jackson_inbox_problem (initial_deleted : ℕ) (initial_received : ℕ)
  (subsequent_deleted : ℕ) (subsequent_received : ℕ) (final_count : ℕ)
  (h1 : initial_deleted = 50)
  (h2 : initial_received = 15)
  (h3 : subsequent_deleted = 20)
  (h4 : subsequent_received = 5)
  (h5 : final_count = 30) :
  final_count - (initial_received + subsequent_received) = 10 := by
  sorry

end jackson_inbox_problem_l2081_208115


namespace solution_difference_l2081_208129

theorem solution_difference (p q : ℝ) : 
  (p - 4) * (p + 4) = 28 * p - 84 →
  (q - 4) * (q + 4) = 28 * q - 84 →
  p ≠ q →
  p > q →
  p - q = 16 * Real.sqrt 2 := by
sorry

end solution_difference_l2081_208129


namespace inequality_solution_implies_m_less_than_one_l2081_208113

theorem inequality_solution_implies_m_less_than_one :
  (∃ x : ℝ, |x + 2| - |x + 3| > m) → m < 1 :=
by
  sorry

end inequality_solution_implies_m_less_than_one_l2081_208113


namespace composition_of_even_is_even_l2081_208155

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (g ∘ g) := by
  sorry

end composition_of_even_is_even_l2081_208155


namespace point_on_y_axis_l2081_208107

/-- If a point P(a-3, 2-a) lies on the y-axis, then P = (0, -1) -/
theorem point_on_y_axis (a : ℝ) :
  (a - 3 = 0) →  -- P lies on y-axis (x-coordinate is 0)
  (a - 3, 2 - a) = (0, -1) :=
by
  sorry

end point_on_y_axis_l2081_208107


namespace shelter_ratio_l2081_208170

theorem shelter_ratio (num_cats : ℕ) (num_dogs : ℕ) : 
  num_cats = 45 →
  (num_cats : ℚ) / (num_dogs + 12 : ℚ) = 15 / 11 →
  (num_cats : ℚ) / (num_dogs : ℚ) = 15 / 7 :=
by sorry

end shelter_ratio_l2081_208170


namespace point_coordinates_l2081_208164

/-- A point in the coordinate plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the coordinate plane. -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance from a point to the x-axis. -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis. -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: If a point P is in the second quadrant, its distance to the x-axis is 4,
    and its distance to the y-axis is 5, then its coordinates are (-5, 4). -/
theorem point_coordinates (P : Point) 
    (h1 : SecondQuadrant P) 
    (h2 : DistanceToXAxis P = 4) 
    (h3 : DistanceToYAxis P = 5) : 
    P.x = -5 ∧ P.y = 4 := by
  sorry

end point_coordinates_l2081_208164


namespace mean_of_first_set_l2081_208184

def first_set (x : ℝ) : List ℝ := [28, x, 70, 88, 104]
def second_set (x : ℝ) : List ℝ := [50, 62, 97, 124, x]

theorem mean_of_first_set :
  ∀ x : ℝ,
  (List.sum (second_set x)) / 5 = 75.6 →
  (List.sum (first_set x)) / 5 = 67 :=
by
  sorry

end mean_of_first_set_l2081_208184

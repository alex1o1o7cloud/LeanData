import Mathlib

namespace trapezoid_area_l1847_184794

/-- Given an outer equilateral triangle with area 36, an inner equilateral triangle
    with area 4, and three congruent trapezoids between them, the area of one
    trapezoid is 32/3. -/
theorem trapezoid_area
  (outer_triangle_area : ℝ)
  (inner_triangle_area : ℝ)
  (num_trapezoids : ℕ)
  (h1 : outer_triangle_area = 36)
  (h2 : inner_triangle_area = 4)
  (h3 : num_trapezoids = 3) :
  (outer_triangle_area - inner_triangle_area) / num_trapezoids = 32 / 3 :=
by sorry

end trapezoid_area_l1847_184794


namespace good_numbers_l1847_184742

def isGoodNumber (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), ∀ k : Fin n, ∃ m : ℕ, k.val + 1 + a k = m^2

theorem good_numbers :
  isGoodNumber 13 ∧
  isGoodNumber 15 ∧
  isGoodNumber 17 ∧
  isGoodNumber 19 ∧
  ¬isGoodNumber 11 := by sorry

end good_numbers_l1847_184742


namespace two_roots_implies_a_greater_than_e_l1847_184787

-- Define the function f(x) = x / ln(x)
noncomputable def f (x : ℝ) : ℝ := x / Real.log x

-- State the theorem
theorem two_roots_implies_a_greater_than_e (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * Real.log x = x ∧ a * Real.log y = y) → a > Real.exp 1 := by
  sorry

end two_roots_implies_a_greater_than_e_l1847_184787


namespace negative_response_proportion_l1847_184731

/-- Given 88 total teams and 49 teams with negative responses,
    prove that P = ⌊10000 * (49/88)⌋ = 5568 -/
theorem negative_response_proportion (total_teams : Nat) (negative_responses : Nat)
    (h1 : total_teams = 88)
    (h2 : negative_responses = 49) :
    ⌊(10000 : ℝ) * ((negative_responses : ℝ) / (total_teams : ℝ))⌋ = 5568 := by
  sorry

#check negative_response_proportion

end negative_response_proportion_l1847_184731


namespace equation_represents_three_lines_l1847_184799

/-- The equation x²(x+y+1) = y²(x+y+1) represents three lines that do not all pass through a common point -/
theorem equation_represents_three_lines (x y : ℝ) : 
  (x^2 * (x + y + 1) = y^2 * (x + y + 1)) ↔ 
  ((y = -x) ∨ (y = x) ∨ (y = -x - 1)) ∧ 
  ¬(∃ p : ℝ × ℝ, (p.1 = p.2 ∧ p.2 = -p.1) ∧ 
                 (p.1 = -p.2 - 1 ∧ p.2 = p.1) ∧ 
                 (p.1 = p.2 ∧ p.2 = -p.1 - 1)) :=
by sorry

end equation_represents_three_lines_l1847_184799


namespace interest_rate_calculation_l1847_184701

/-- 
Given a loan with simple interest where:
- The principal amount is $1200
- The number of years equals the rate of interest
- The total interest paid is $432
Prove that the rate of interest is 6%
-/
theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) 
  (h1 : principal = 1200)
  (h2 : interest_paid = 432) :
  ∃ (rate : ℝ), rate = 6 ∧ interest_paid = principal * (rate / 100) * rate :=
by sorry

end interest_rate_calculation_l1847_184701


namespace triangle_area_with_cosine_root_l1847_184705

theorem triangle_area_with_cosine_root (a b : ℝ) (cos_theta : ℝ) : 
  a = 3 → b = 5 → (5 * cos_theta^2 - 7 * cos_theta - 6 = 0) → 
  (1/2 : ℝ) * a * b * cos_theta = 6 := by
  sorry

end triangle_area_with_cosine_root_l1847_184705


namespace money_distribution_l1847_184735

theorem money_distribution (total : ℝ) (a b c : ℝ) : 
  total = 1080 →
  a = (1/3) * (b + c) →
  b = (2/7) * (a + c) →
  a + b + c = total →
  a > b →
  a - b = 30 := by sorry

end money_distribution_l1847_184735


namespace price_increase_ratio_l1847_184718

theorem price_increase_ratio (original_price : ℝ) 
  (h1 : original_price > 0)
  (h2 : original_price * 1.3 = 364) : 
  364 / original_price = 1.3 := by
sorry

end price_increase_ratio_l1847_184718


namespace expression_evaluation_l1847_184748

theorem expression_evaluation : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by
  sorry

end expression_evaluation_l1847_184748


namespace gervais_mileage_proof_l1847_184774

/-- Gervais' average daily mileage --/
def gervais_average_mileage : ℝ := 315

/-- Number of days Gervais drove --/
def gervais_days : ℕ := 3

/-- Total miles Henri drove in a week --/
def henri_total_miles : ℝ := 1250

/-- Difference in miles between Henri and Gervais --/
def miles_difference : ℝ := 305

theorem gervais_mileage_proof :
  gervais_average_mileage * gervais_days = henri_total_miles - miles_difference :=
by sorry

end gervais_mileage_proof_l1847_184774


namespace visitors_calculation_l1847_184709

/-- The number of visitors to Buckingham Palace on a specific day, given the total visitors over 85 days and the visitors on the previous day. -/
def visitors_on_day (total_visitors : ℕ) (previous_day_visitors : ℕ) : ℕ :=
  total_visitors - previous_day_visitors

/-- Theorem stating that the number of visitors on a specific day is equal to
    the total visitors over 85 days minus the visitors on the previous day. -/
theorem visitors_calculation (total_visitors previous_day_visitors : ℕ) 
    (h1 : total_visitors = 829)
    (h2 : previous_day_visitors = 45) :
  visitors_on_day total_visitors previous_day_visitors = 784 := by
  sorry

#eval visitors_on_day 829 45

end visitors_calculation_l1847_184709


namespace tangent_slope_at_4_l1847_184758

def f (x : ℝ) : ℝ := x^3 - 7*x^2 + 1

theorem tangent_slope_at_4 : 
  (deriv f) 4 = -8 := by sorry

end tangent_slope_at_4_l1847_184758


namespace complex_magnitude_l1847_184721

theorem complex_magnitude (a b : ℝ) : 
  (Complex.I + a * Complex.I) * Complex.I = 1 - b * Complex.I →
  Complex.abs (a + b * Complex.I) = Real.sqrt 2 := by
sorry

end complex_magnitude_l1847_184721


namespace frogs_on_logs_count_l1847_184733

/-- The number of frogs that climbed onto logs in the pond -/
def frogs_on_logs (total_frogs lily_pad_frogs rock_frogs : ℕ) : ℕ :=
  total_frogs - (lily_pad_frogs + rock_frogs)

/-- Theorem stating that the number of frogs on logs is 3 -/
theorem frogs_on_logs_count :
  frogs_on_logs 32 5 24 = 3 := by
  sorry

end frogs_on_logs_count_l1847_184733


namespace sum_of_three_numbers_l1847_184703

theorem sum_of_three_numbers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 := by
  sorry

end sum_of_three_numbers_l1847_184703


namespace infinitely_many_composite_f_increasing_l1847_184759

/-- The number of positive divisors of a natural number -/
def tau (a : ℕ) : ℕ := (Nat.divisors a).card

/-- The function f(n) as defined in the problem -/
def f (n : ℕ) : ℕ := tau (Nat.factorial n) - tau (Nat.factorial (n - 1))

/-- A composite number -/
def Composite (n : ℕ) : Prop := ¬Nat.Prime n ∧ n > 1

/-- The main theorem to be proved -/
theorem infinitely_many_composite_f_increasing :
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
  (∀ n ∈ S, Composite n ∧ 
    (∀ m : ℕ, m < n → f m < f n)) := by sorry

end infinitely_many_composite_f_increasing_l1847_184759


namespace probability_closer_to_center_l1847_184719

theorem probability_closer_to_center (R : ℝ) (h : R = 4) : 
  (π * 1^2) / (π * R^2) = 1 / 16 := by
sorry

end probability_closer_to_center_l1847_184719


namespace min_product_positive_reals_l1847_184725

theorem min_product_positive_reals (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  x ≤ 2 * (y + z) →
  y ≤ 2 * (x + z) →
  z ≤ 2 * (x + y) →
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 1 → 
    a ≤ 2 * (b + c) → b ≤ 2 * (a + c) → c ≤ 2 * (a + b) →
    x * y * z ≤ a * b * c →
  x * y * z = 1 / 32 := by
  sorry

end min_product_positive_reals_l1847_184725


namespace kindergarten_pet_distribution_l1847_184756

/-- Represents the kindergarten pet distribution problem -/
theorem kindergarten_pet_distribution 
  (total_children : ℕ) 
  (children_with_both : ℕ) 
  (children_with_cats : ℕ) 
  (h1 : total_children = 30)
  (h2 : children_with_both = 6)
  (h3 : children_with_cats = 12)
  : total_children - children_with_cats = 18 :=
by sorry

end kindergarten_pet_distribution_l1847_184756


namespace system_has_solution_l1847_184777

/-- The system of equations has a solution for the given range of b -/
theorem system_has_solution (b : ℝ) 
  (h : b ∈ Set.Iic (-7/12) ∪ Set.Ioi 0) : 
  ∃ (a x y : ℝ), x = 7/b - |y + b| ∧ 
                 x^2 + y^2 + 96 = -a*(2*y + a) - 20*x := by
  sorry


end system_has_solution_l1847_184777


namespace square_floor_area_l1847_184797

theorem square_floor_area (rug_length : ℝ) (rug_width : ℝ) (uncovered_fraction : ℝ) :
  rug_length = 2 →
  rug_width = 7 →
  uncovered_fraction = 0.78125 →
  ∃ (floor_side : ℝ),
    floor_side ^ 2 = 64 ∧
    rug_length * rug_width = (1 - uncovered_fraction) * floor_side ^ 2 :=
by sorry

end square_floor_area_l1847_184797


namespace coin_toss_probability_l1847_184768

theorem coin_toss_probability (p : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : p ^ 5 = 0.0625) :
  p = 0.5 := by
sorry

end coin_toss_probability_l1847_184768


namespace product_equals_32_l1847_184782

theorem product_equals_32 : 
  (1 / 4 : ℚ) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 := by
  sorry

end product_equals_32_l1847_184782


namespace driver_work_days_l1847_184727

/-- Represents the number of days driven from Monday to Wednesday -/
def days_mon_to_wed : ℕ := 3

/-- Represents the number of days driven from Thursday to Friday -/
def days_thu_to_fri : ℕ := 2

/-- Average driving hours per day -/
def avg_hours_per_day : ℕ := 2

/-- Average speed from Monday to Wednesday in km/h -/
def speed_mon_to_wed : ℕ := 12

/-- Average speed from Thursday to Friday in km/h -/
def speed_thu_to_fri : ℕ := 9

/-- Total distance traveled in km -/
def total_distance : ℕ := 108

theorem driver_work_days : 
  days_mon_to_wed * avg_hours_per_day * speed_mon_to_wed + 
  days_thu_to_fri * avg_hours_per_day * speed_thu_to_fri = total_distance ∧
  days_mon_to_wed + days_thu_to_fri = 5 :=
by sorry

end driver_work_days_l1847_184727


namespace max_large_chips_l1847_184717

theorem max_large_chips (total : ℕ) (is_prime : ℕ → Prop) : 
  total = 80 → 
  (∃ (small large prime : ℕ), 
    small + large = total ∧ 
    small = large + prime ∧ 
    is_prime prime) →
  (∀ (large : ℕ), 
    (∃ (small prime : ℕ), 
      small + large = total ∧ 
      small = large + prime ∧ 
      is_prime prime) → 
    large ≤ 39) ∧
  (∃ (small prime : ℕ), 
    small + 39 = total ∧ 
    small = 39 + prime ∧ 
    is_prime prime) :=
by sorry

end max_large_chips_l1847_184717


namespace complex_multiplication_l1847_184706

theorem complex_multiplication (i : ℂ) :
  i * i = -1 →
  (1 - i) * (2 + i) = 3 - i :=
by sorry

end complex_multiplication_l1847_184706


namespace calculation_proof_l1847_184760

theorem calculation_proof : (120 : ℝ) / ((6 : ℝ) / 2) * 3 = 120 := by
  sorry

end calculation_proof_l1847_184760


namespace geometric_series_product_l1847_184789

theorem geometric_series_product (x : ℝ) : 
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/x)^n → x = 9 :=
by sorry

end geometric_series_product_l1847_184789


namespace triangle_vector_sum_l1847_184702

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the triangle ABC and vectors a and b
variable (A B C : V) (a b : V)

-- State the theorem
theorem triangle_vector_sum (h1 : B - C = a) (h2 : C - A = b) : 
  A - B = -a - b := by sorry

end triangle_vector_sum_l1847_184702


namespace positive_real_properties_l1847_184771

theorem positive_real_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 - b^2 = 1 → a - b < 1) ∧
  (a > b + 1 → a^2 > b^2 + 1) := by
sorry

end positive_real_properties_l1847_184771


namespace geometric_sequence_min_value_l1847_184754

/-- A positive geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_min_value (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (a 6 = a 5 + 2 * a 4) →
  (Real.sqrt (a m * a n) = 4 * a 1) →
  (∃ min_value : ℝ, min_value = (3 + 2 * Real.sqrt 2) / 6 ∧
    ∀ x y : ℕ, (Real.sqrt (a x * a y) = 4 * a 1) → (1 / x + 2 / y) ≥ min_value) :=
by sorry

end geometric_sequence_min_value_l1847_184754


namespace solve_equation_l1847_184757

theorem solve_equation (x : ℝ) : 3 * x + 20 = (1/3) * (7 * x + 45) → x = -7.5 := by
  sorry

end solve_equation_l1847_184757


namespace base_subtraction_proof_l1847_184746

def to_base_10 (digits : List ℕ) (base : ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

theorem base_subtraction_proof :
  let base_7_num := to_base_10 [5, 2, 3] 7
  let base_5_num := to_base_10 [4, 6, 1] 5
  base_7_num - base_5_num = 107 := by sorry

end base_subtraction_proof_l1847_184746


namespace calculate_otimes_expression_l1847_184740

-- Define the ⊗ operation
def otimes (a b : ℚ) : ℚ := (a + b) / (a - b)

-- The main theorem to prove
theorem calculate_otimes_expression :
  (otimes (otimes 8 6) (otimes 2 1)) = 5/2 := by
  sorry

end calculate_otimes_expression_l1847_184740


namespace restaurant_bill_proof_l1847_184720

/-- The total bill for 9 friends dining at a restaurant -/
def total_bill : ℕ := 156

/-- The number of friends dining -/
def num_friends : ℕ := 9

/-- The amount Judi paid -/
def judi_payment : ℕ := 5

/-- The extra amount each remaining friend paid -/
def extra_payment : ℕ := 3

theorem restaurant_bill_proof :
  let regular_share := total_bill / num_friends
  let tom_payment := regular_share / 2
  let remaining_friends := num_friends - 2
  total_bill = 
    (remaining_friends * (regular_share + extra_payment)) + 
    judi_payment + 
    tom_payment :=
by sorry

end restaurant_bill_proof_l1847_184720


namespace derek_car_increase_l1847_184788

/-- Represents the number of dogs and cars Derek owns at a given time --/
structure DereksPets where
  dogs : ℕ
  cars : ℕ

/-- The change in Derek's pet ownership over 10 years --/
def petsChange (initial final : DereksPets) : ℕ := final.cars - initial.cars

/-- Theorem stating the increase in cars Derek owns over 10 years --/
theorem derek_car_increase :
  ∀ (initial final : DereksPets),
  initial.dogs = 90 →
  initial.dogs = 3 * initial.cars →
  final.dogs = 120 →
  final.cars = 2 * final.dogs →
  petsChange initial final = 210 := by
  sorry

end derek_car_increase_l1847_184788


namespace gcd_factorial_eight_ten_l1847_184708

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcd_factorial_eight_ten :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end gcd_factorial_eight_ten_l1847_184708


namespace unique_solution_cube_equation_l1847_184793

theorem unique_solution_cube_equation :
  ∃! (y : ℝ), y ≠ 0 ∧ (3 * y)^6 = (9 * y)^5 :=
by
  use 81
  sorry

end unique_solution_cube_equation_l1847_184793


namespace sum_mod_thirteen_zero_l1847_184750

theorem sum_mod_thirteen_zero : (9023 + 9024 + 9025 + 9026) % 13 = 0 := by
  sorry

end sum_mod_thirteen_zero_l1847_184750


namespace perpendicular_line_equation_l1847_184780

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2 * y - 5 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (-2, 1)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x - y + 5 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  ∃ (m b : ℝ),
    (∀ (x y : ℝ), perpendicular_line x y ↔ y = m * x + b) ∧
    (perpendicular_line point.1 point.2) ∧
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      given_line x₁ y₁ → given_line x₂ y₂ → x₁ ≠ x₂ →
      (y₂ - y₁) / (x₂ - x₁) * m = -1) :=
sorry

end perpendicular_line_equation_l1847_184780


namespace jungkook_has_larger_number_l1847_184784

theorem jungkook_has_larger_number (yoongi_number jungkook_number : ℕ) : 
  yoongi_number = 4 → jungkook_number = 6 * 3 → jungkook_number > yoongi_number := by
  sorry

end jungkook_has_larger_number_l1847_184784


namespace stadium_attendance_l1847_184765

/-- The number of people in a stadium at the start and end of a game -/
theorem stadium_attendance (boys_start girls_start boys_end girls_end : ℕ) :
  girls_start = 240 →
  boys_end = boys_start - boys_start / 4 →
  girls_end = girls_start - girls_start / 8 →
  boys_end + girls_end = 480 →
  boys_start + girls_start = 600 := by
sorry

end stadium_attendance_l1847_184765


namespace counterexample_exists_l1847_184726

theorem counterexample_exists : ∃ n : ℕ, ¬ Nat.Prime n ∧ ¬ Nat.Prime (n - 4) := by
  sorry

end counterexample_exists_l1847_184726


namespace leap_year_53_mondays_probability_l1847_184762

/-- The number of days in a leap year -/
def leapYearDays : ℕ := 366

/-- The number of full weeks in a leap year -/
def fullWeeks : ℕ := leapYearDays / 7

/-- The number of extra days in a leap year after full weeks -/
def extraDays : ℕ := leapYearDays % 7

/-- The number of possible combinations for the extra days -/
def possibleCombinations : ℕ := 7

/-- The number of combinations that include a Monday -/
def combinationsWithMonday : ℕ := 2

/-- The probability of a leap year having 53 Mondays -/
def probabilityOf53Mondays : ℚ := combinationsWithMonday / possibleCombinations

theorem leap_year_53_mondays_probability :
  probabilityOf53Mondays = 2 / 7 := by
  sorry

end leap_year_53_mondays_probability_l1847_184762


namespace supplement_of_forty_degrees_l1847_184745

/-- Given a system of parallel lines where an angle of 40° is formed, 
    prove that its supplement measures 140°. -/
theorem supplement_of_forty_degrees (α : Real) (h1 : α = 40) : 180 - α = 140 := by
  sorry

end supplement_of_forty_degrees_l1847_184745


namespace smallest_a_for_96a_squared_equals_b_cubed_l1847_184795

theorem smallest_a_for_96a_squared_equals_b_cubed :
  ∀ a : ℕ+, a < 12 → ¬∃ b : ℕ+, 96 * a^2 = b^3 ∧ 
  ∃ b : ℕ+, 96 * 12^2 = b^3 :=
by sorry

end smallest_a_for_96a_squared_equals_b_cubed_l1847_184795


namespace literary_readers_count_l1847_184764

theorem literary_readers_count (total : ℕ) (science_fiction : ℕ) (both : ℕ) 
  (h1 : total = 650) 
  (h2 : science_fiction = 250) 
  (h3 : both = 150) : 
  total = science_fiction + (550 : ℕ) - both :=
by sorry

end literary_readers_count_l1847_184764


namespace larger_number_proof_l1847_184736

theorem larger_number_proof (x y : ℝ) (h1 : 4 * y = 7 * x) (h2 : y - x = 12) : y = 28 := by
  sorry

end larger_number_proof_l1847_184736


namespace circles_intersecting_parallel_lines_l1847_184744

-- Define the types for our objects
variable (Point Circle Line : Type)

-- Define the necessary relations and properties
variable (onCircle : Point → Circle → Prop)
variable (intersectsAt : Circle → Circle → Point → Prop)
variable (passesThrough : Line → Point → Prop)
variable (intersectsCircleAt : Line → Circle → Point → Prop)
variable (parallel : Line → Line → Prop)
variable (lineThroughPoints : Point → Point → Line)

-- State the theorem
theorem circles_intersecting_parallel_lines
  (Γ₁ Γ₂ : Circle)
  (P Q A A' B B' : Point) :
  intersectsAt Γ₁ Γ₂ P →
  intersectsAt Γ₁ Γ₂ Q →
  (∃ l : Line, passesThrough l P ∧ intersectsCircleAt l Γ₁ A ∧ intersectsCircleAt l Γ₂ A') →
  (∃ m : Line, passesThrough m Q ∧ intersectsCircleAt m Γ₁ B ∧ intersectsCircleAt m Γ₂ B') →
  A ≠ P →
  A' ≠ P →
  B ≠ Q →
  B' ≠ Q →
  parallel (lineThroughPoints A B) (lineThroughPoints A' B') :=
sorry

end circles_intersecting_parallel_lines_l1847_184744


namespace eighth_term_matchsticks_l1847_184772

/-- The number of matchsticks in the nth term of the sequence -/
def matchsticks (n : ℕ) : ℕ := (n + 1) * 3

/-- Theorem: The number of matchsticks in the eighth term is 27 -/
theorem eighth_term_matchsticks : matchsticks 8 = 27 := by
  sorry

end eighth_term_matchsticks_l1847_184772


namespace inequality_one_inequality_two_l1847_184710

-- Inequality 1
theorem inequality_one (x : ℝ) : 4 * x + 5 ≤ 2 * (x + 1) ↔ x ≤ -3/2 := by sorry

-- Inequality 2
theorem inequality_two (x : ℝ) : (2 * x - 1) / 3 - (9 * x + 2) / 6 ≤ 1 ↔ x ≥ -2 := by sorry

end inequality_one_inequality_two_l1847_184710


namespace chocolate_distribution_l1847_184714

theorem chocolate_distribution (total_pieces : ℕ) (num_boxes : ℕ) (pieces_per_box : ℕ) :
  total_pieces = 3000 →
  num_boxes = 6 →
  total_pieces = num_boxes * pieces_per_box →
  pieces_per_box = 500 := by
  sorry

end chocolate_distribution_l1847_184714


namespace weeks_to_buy_bike_l1847_184796

def bike_cost : ℕ := 650
def birthday_money : ℕ := 60 + 45 + 25
def weekly_earnings : ℕ := 20

theorem weeks_to_buy_bike :
  ∃ (weeks : ℕ), birthday_money + weeks * weekly_earnings = bike_cost ∧ weeks = 26 :=
by sorry

end weeks_to_buy_bike_l1847_184796


namespace pool_depth_relationship_l1847_184741

/-- The depth of Sarah's pool in feet -/
def sarahs_pool_depth : ℝ := 5

/-- The depth of John's pool in feet -/
def johns_pool_depth : ℝ := 15

/-- Theorem stating the relationship between John's and Sarah's pool depths -/
theorem pool_depth_relationship : 
  johns_pool_depth = 2 * sarahs_pool_depth + 5 ∧ sarahs_pool_depth = 5 := by
  sorry

end pool_depth_relationship_l1847_184741


namespace ahn_max_number_l1847_184712

theorem ahn_max_number : ∃ (m : ℕ), m = 870 ∧ 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 → 3 * (300 - n) ≤ m :=
by sorry

end ahn_max_number_l1847_184712


namespace complex_number_on_line_l1847_184785

theorem complex_number_on_line (a : ℝ) : 
  (Complex.I * (Complex.I⁻¹ * a + (1 - Complex.I) / 2)).re + 
  (Complex.I * (Complex.I⁻¹ * a + (1 - Complex.I) / 2)).im = 0 → 
  a = 0 := by
  sorry

end complex_number_on_line_l1847_184785


namespace dad_caught_more_trouts_l1847_184751

-- Define the number of trouts Caleb caught
def caleb_trouts : ℕ := 2

-- Define the number of trouts Caleb's dad caught
def dad_trouts : ℕ := 3 * caleb_trouts

-- Theorem to prove
theorem dad_caught_more_trouts : dad_trouts - caleb_trouts = 4 := by
  sorry

end dad_caught_more_trouts_l1847_184751


namespace root_implies_a_value_l1847_184737

theorem root_implies_a_value (a : ℝ) : 
  ((-2 : ℝ)^2 + 3*(-2) + a = 0) → a = 2 := by
  sorry

end root_implies_a_value_l1847_184737


namespace company_women_workers_l1847_184761

theorem company_women_workers 
  (total_workers : ℕ) 
  (h1 : total_workers / 3 = total_workers - total_workers * 2 / 3) -- A third of workers do not have a retirement plan
  (h2 : (total_workers / 3) / 2 = total_workers / 6) -- 50% of workers without a retirement plan are women
  (h3 : (total_workers * 2 / 3) * 2 / 5 = total_workers * 8 / 30) -- 40% of workers with a retirement plan are men
  (h4 : total_workers * 8 / 30 = 120) -- 120 workers are men
  : total_workers - 120 = 330 := by
sorry

end company_women_workers_l1847_184761


namespace smallest_number_divisible_by_all_l1847_184778

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 7) % 25 = 0 ∧ (n + 7) % 49 = 0 ∧ (n + 7) % 15 = 0 ∧ (n + 7) % 21 = 0

theorem smallest_number_divisible_by_all : 
  is_divisible_by_all 3668 ∧ ∀ m : ℕ, m < 3668 → ¬is_divisible_by_all m := by
  sorry

end smallest_number_divisible_by_all_l1847_184778


namespace p_sufficient_not_necessary_q_l1847_184704

-- Define the propositions p and q
def p (m : ℝ) : Prop := 1/4 < m ∧ m < 1

def q (m : ℝ) : Prop := 0 < m ∧ m < 1

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_q :
  (∀ m : ℝ, p m → q m) ∧ 
  (∃ m : ℝ, q m ∧ ¬p m) :=
sorry

end p_sufficient_not_necessary_q_l1847_184704


namespace base8_divisibility_by_13_l1847_184779

/-- Converts a base-8 number of the form 3dd7₈ to base 10 --/
def base8_to_base10 (d : ℕ) : ℕ := 3 * 512 + d * 64 + d * 8 + 7

/-- Checks if a natural number is divisible by 13 --/
def divisible_by_13 (n : ℕ) : Prop := ∃ k : ℕ, n = 13 * k

/-- A base-8 digit is between 0 and 7 inclusive --/
def is_base8_digit (d : ℕ) : Prop := 0 ≤ d ∧ d ≤ 7

theorem base8_divisibility_by_13 (d : ℕ) (h : is_base8_digit d) : 
  divisible_by_13 (base8_to_base10 d) ↔ (d = 1 ∨ d = 2) :=
sorry

end base8_divisibility_by_13_l1847_184779


namespace equation_solution_l1847_184769

theorem equation_solution (x y c : ℝ) : 
  7^(3*x - 1) * 3^(4*y - 3) = c^x * 27^y ∧ x + y = 4 → c = 49 := by
  sorry

end equation_solution_l1847_184769


namespace widgets_per_week_l1847_184707

/-- The number of widgets John can make per hour -/
def widgets_per_hour : ℕ := 20

/-- The number of hours John works per day -/
def hours_per_day : ℕ := 8

/-- The number of days John works per week -/
def days_per_week : ℕ := 5

/-- Theorem: John makes 800 widgets in a week -/
theorem widgets_per_week : 
  widgets_per_hour * hours_per_day * days_per_week = 800 := by
  sorry


end widgets_per_week_l1847_184707


namespace lcm_of_9_12_15_l1847_184724

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end lcm_of_9_12_15_l1847_184724


namespace percentage_of_singles_is_70_percent_l1847_184790

def total_hits : ℕ := 50
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 10

def singles : ℕ := total_hits - (home_runs + triples + doubles)

theorem percentage_of_singles_is_70_percent :
  (singles : ℚ) / total_hits * 100 = 70 := by sorry

end percentage_of_singles_is_70_percent_l1847_184790


namespace perfect_square_sum_existence_l1847_184700

theorem perfect_square_sum_existence : ∃ (x y z u v w t s : ℕ+), 
  x^2 + y + z + u = (x + v)^2 ∧
  y^2 + x + z + u = (y + w)^2 ∧
  z^2 + x + y + u = (z + t)^2 ∧
  u^2 + x + y + z = (u + s)^2 :=
by sorry

end perfect_square_sum_existence_l1847_184700


namespace s_squared_minus_c_squared_range_l1847_184715

/-- The theorem states that for any point (x, y, z) in 3D space,
    where r is the distance from the origin to the point,
    s = y/r, and c = x/r, the value of s^2 - c^2 is always
    between -1 and 1, inclusive. -/
theorem s_squared_minus_c_squared_range
  (x y z : ℝ) 
  (r : ℝ) 
  (hr : r = Real.sqrt (x^2 + y^2 + z^2)) 
  (s : ℝ) 
  (hs : s = y / r) 
  (c : ℝ) 
  (hc : c = x / r) : 
  -1 ≤ s^2 - c^2 ∧ s^2 - c^2 ≤ 1 := by
  sorry

end s_squared_minus_c_squared_range_l1847_184715


namespace expression_equality_l1847_184798

theorem expression_equality : 
  (1 / 3) ^ 2000 * 27 ^ 669 + Real.sin (60 * π / 180) * Real.tan (60 * π / 180) + (2009 + Real.sin (25 * π / 180)) ^ 0 = 2 + 29 / 54 := by
  sorry

end expression_equality_l1847_184798


namespace complex_square_equality_l1847_184773

theorem complex_square_equality (x y : ℕ+) : 
  (x + y * Complex.I) ^ 2 = 7 + 24 * Complex.I → x + y * Complex.I = 4 + 3 * Complex.I := by
sorry

end complex_square_equality_l1847_184773


namespace parkway_fifth_grade_count_l1847_184739

/-- The number of students in the fifth grade at Parkway Elementary School -/
def total_students : ℕ := sorry

/-- The number of boys in the fifth grade -/
def boys : ℕ := 312

/-- The number of students playing soccer -/
def soccer_players : ℕ := 250

/-- The percentage of soccer players who are boys -/
def boys_soccer_percentage : ℚ := 82 / 100

/-- The number of girls not playing soccer -/
def girls_not_soccer : ℕ := 63

theorem parkway_fifth_grade_count :
  total_students = 420 :=
by sorry

end parkway_fifth_grade_count_l1847_184739


namespace red_marbles_after_replacement_l1847_184786

theorem red_marbles_after_replacement (total : ℕ) (blue green red : ℕ) : 
  total > 0 →
  blue = (40 * total + 99) / 100 →
  green = 20 →
  red = (10 * total + 99) / 100 →
  (15 * total + 99) / 100 + (5 * total + 99) / 100 + blue + green + red = total →
  (16 : ℕ) = red + blue / 3 := by
  sorry

end red_marbles_after_replacement_l1847_184786


namespace games_within_division_l1847_184734

/-- Represents a baseball league with specific game scheduling rules. -/
structure BaseballLeague where
  /-- Number of games played against each team in the same division -/
  N : ℕ
  /-- Number of games played against each team in the other division -/
  M : ℕ
  /-- N is greater than twice M -/
  h1 : N > 2 * M
  /-- M is greater than 6 -/
  h2 : M > 6
  /-- Total number of games each team plays is 92 -/
  h3 : 3 * N + 4 * M = 92

/-- The number of games a team plays within its own division in the given baseball league is 60. -/
theorem games_within_division (league : BaseballLeague) : 3 * league.N = 60 := by
  sorry

end games_within_division_l1847_184734


namespace guy_speed_increase_point_l1847_184776

/-- Represents the problem of finding the point where Guy increases his speed --/
theorem guy_speed_increase_point
  (total_distance : ℝ)
  (average_speed : ℝ)
  (first_half_speed : ℝ)
  (speed_increase : ℝ)
  (h1 : total_distance = 60)
  (h2 : average_speed = 30)
  (h3 : first_half_speed = 24)
  (h4 : speed_increase = 16) :
  let second_half_speed := first_half_speed + speed_increase
  let increase_point := (total_distance * first_half_speed) / (first_half_speed + second_half_speed)
  increase_point = 30 := by sorry

end guy_speed_increase_point_l1847_184776


namespace linear_system_solution_l1847_184792

/-- Given a system of linear equations and conditions, prove the range of m and its integer values -/
theorem linear_system_solution (m x y : ℝ) : 
  (2 * x + y = 1 + 2 * m) → 
  (x + 2 * y = 2 - m) → 
  (x + y > 0) → 
  (m > -3) ∧ 
  (((2 * m + 1) * x - 2 * m < 1) → 
   (x > 1) → 
   (m = -2 ∨ m = -1)) := by
  sorry

end linear_system_solution_l1847_184792


namespace average_price_is_45_cents_l1847_184770

/-- Represents the fruit selection and pricing problem -/
structure FruitProblem where
  apple_price : ℕ
  orange_price : ℕ
  total_fruits : ℕ
  initial_avg_price : ℕ
  oranges_removed : ℕ

/-- Calculates the average price of remaining fruits -/
def average_price_after_removal (fp : FruitProblem) : ℚ :=
  sorry

/-- Theorem stating that the average price of remaining fruits is 45 cents -/
theorem average_price_is_45_cents (fp : FruitProblem) 
  (h1 : fp.apple_price = 40)
  (h2 : fp.orange_price = 60)
  (h3 : fp.total_fruits = 10)
  (h4 : fp.initial_avg_price = 54)
  (h5 : fp.oranges_removed = 6) :
  average_price_after_removal fp = 45 := by
  sorry

end average_price_is_45_cents_l1847_184770


namespace problem_statement_l1847_184783

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end problem_statement_l1847_184783


namespace sampling_most_appropriate_for_qingming_l1847_184711

-- Define the survey methods
inductive SurveyMethod
| Census
| Sampling

-- Define the survey scenarios
inductive SurveyScenario
| MilkHygiene
| SubwaySecurity
| StudentSleep
| QingmingCommemoration

-- Define a function to determine the appropriateness of a survey method for a given scenario
def is_appropriate (scenario : SurveyScenario) (method : SurveyMethod) : Prop :=
  match scenario, method with
  | SurveyScenario.MilkHygiene, SurveyMethod.Sampling => True
  | SurveyScenario.SubwaySecurity, SurveyMethod.Census => True
  | SurveyScenario.StudentSleep, SurveyMethod.Sampling => True
  | SurveyScenario.QingmingCommemoration, SurveyMethod.Sampling => True
  | _, _ => False

-- Theorem stating that sampling is the most appropriate method for the Qingming commemoration scenario
theorem sampling_most_appropriate_for_qingming :
  ∀ (scenario : SurveyScenario) (method : SurveyMethod),
    is_appropriate scenario method →
    (scenario = SurveyScenario.QingmingCommemoration ∧ method = SurveyMethod.Sampling) ∨
    (scenario ≠ SurveyScenario.QingmingCommemoration) :=
by sorry

end sampling_most_appropriate_for_qingming_l1847_184711


namespace largest_number_l1847_184723

-- Define the numbers as real numbers
def a : ℝ := 9.12445
def b : ℝ := 9.124555555555555555555555555555555555555555555555555
def c : ℝ := 9.124545454545454545454545454545454545454545454545454
def d : ℝ := 9.124524524524524524524524524524524524524524524524524
def e : ℝ := 9.124512451245124512451245124512451245124512451245124

-- Theorem statement
theorem largest_number : b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end largest_number_l1847_184723


namespace stella_profit_l1847_184753

def dolls : ℕ := 3
def clocks : ℕ := 2
def glasses : ℕ := 5

def doll_price : ℕ := 5
def clock_price : ℕ := 15
def glass_price : ℕ := 4

def total_cost : ℕ := 40

def total_sales : ℕ := dolls * doll_price + clocks * clock_price + glasses * glass_price

def profit : ℕ := total_sales - total_cost

theorem stella_profit : profit = 25 := by
  sorry

end stella_profit_l1847_184753


namespace unique_rectangle_l1847_184775

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area_eq : length * width = 100
  perimeter_eq : length + width = 24

/-- Two rectangles are considered distinct if they are not congruent -/
def distinct (r1 r2 : Rectangle) : Prop :=
  (r1.length ≠ r2.length ∧ r1.length ≠ r2.width) ∨
  (r1.width ≠ r2.length ∧ r1.width ≠ r2.width)

/-- There is exactly one distinct rectangle with area 100 and perimeter 24 -/
theorem unique_rectangle : ∃! r : Rectangle, ∀ s : Rectangle, ¬(distinct r s) :=
  sorry

end unique_rectangle_l1847_184775


namespace olympic_production_l1847_184738

/-- The number of sets of Olympic logo and mascots that can be produced -/
theorem olympic_production : ∃ (x y : ℕ), 
  4 * x + 5 * y = 20000 ∧ 
  3 * x + 10 * y = 30000 ∧ 
  x = 2000 ∧ 
  y = 2400 := by
  sorry

end olympic_production_l1847_184738


namespace zaras_goats_l1847_184791

theorem zaras_goats (cows sheep : ℕ) (groups : ℕ) (animals_per_group : ℕ) (goats : ℕ) : 
  cows = 24 → 
  sheep = 7 → 
  groups = 3 → 
  animals_per_group = 48 → 
  goats = groups * animals_per_group - (cows + sheep) → 
  goats = 113 := by
  sorry

end zaras_goats_l1847_184791


namespace boat_journey_time_l1847_184747

/-- Calculates the total journey time for a boat traveling upstream and downstream -/
theorem boat_journey_time 
  (distance : ℝ) 
  (initial_current_speed : ℝ) 
  (upstream_current_speed : ℝ) 
  (boat_still_speed : ℝ) 
  (headwind_speed_reduction : ℝ) : 
  let upstream_time := distance / (boat_still_speed - upstream_current_speed)
  let downstream_speed := (boat_still_speed - headwind_speed_reduction) + initial_current_speed
  let downstream_time := distance / downstream_speed
  upstream_time + downstream_time = 26.67 :=
by
  sorry

#check boat_journey_time 56 2 3 6 1

end boat_journey_time_l1847_184747


namespace monthly_production_l1847_184722

/-- Represents the number of computers produced in a given time period -/
structure ComputerProduction where
  rate : ℝ  -- Computers produced per 30-minute interval
  days : ℕ  -- Number of days in the time period

/-- Calculates the total number of computers produced in the given time period -/
def totalComputers (prod : ComputerProduction) : ℝ :=
  (prod.rate * (prod.days * 24 * 2 : ℝ))

/-- Theorem stating that a factory producing 6.25 computers every 30 minutes
    for 28 days will produce 8400 computers -/
theorem monthly_production :
  totalComputers ⟨6.25, 28⟩ = 8400 := by sorry

end monthly_production_l1847_184722


namespace sqrt_of_three_minus_negative_one_equals_two_l1847_184766

theorem sqrt_of_three_minus_negative_one_equals_two :
  Real.sqrt (3 - (-1)) = 2 := by
  sorry

end sqrt_of_three_minus_negative_one_equals_two_l1847_184766


namespace gear_teeth_problem_l1847_184781

theorem gear_teeth_problem (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 60) (h4 : 4 * x - 20 = 5 * y) (h5 : 5 * y = 10 * z) : x = 30 ∧ y = 20 ∧ z = 10 := by
  sorry

end gear_teeth_problem_l1847_184781


namespace mixed_fraction_decimal_calculation_l1847_184763

theorem mixed_fraction_decimal_calculation :
  let a : ℚ := 84 + 4 / 19
  let b : ℚ := 105 + 5 / 19
  let c : ℚ := 1.375
  let d : ℚ := 0.8
  a * c + b * d = 200 := by sorry

end mixed_fraction_decimal_calculation_l1847_184763


namespace reporter_earnings_l1847_184729

/-- Reporter's earnings calculation --/
theorem reporter_earnings
  (words_per_minute : ℕ)
  (pay_per_word : ℚ)
  (pay_per_article : ℕ)
  (num_articles : ℕ)
  (hours_available : ℕ)
  (h1 : words_per_minute = 10)
  (h2 : pay_per_word = 1/10)
  (h3 : pay_per_article = 60)
  (h4 : num_articles = 3)
  (h5 : hours_available = 4)
  : (((hours_available * 60 * words_per_minute) * pay_per_word + num_articles * pay_per_article) / hours_available : ℚ) = 105 :=
by sorry

end reporter_earnings_l1847_184729


namespace gwen_homework_problems_l1847_184730

/-- Represents the number of problems for each subject -/
structure SubjectProblems where
  math : ℕ
  science : ℕ
  history : ℕ
  english : ℕ

/-- Calculates the total number of problems left for homework -/
def problems_left (initial : SubjectProblems) (completed : SubjectProblems) : ℕ :=
  (initial.math - completed.math) +
  (initial.science - completed.science) +
  (initial.history - completed.history) +
  (initial.english - completed.english)

/-- Theorem: Given Gwen's initial problems and completed problems, she has 19 problems left for homework -/
theorem gwen_homework_problems :
  let initial := SubjectProblems.mk 18 11 15 7
  let completed := SubjectProblems.mk 12 6 10 4
  problems_left initial completed = 19 := by
  sorry

end gwen_homework_problems_l1847_184730


namespace min_value_on_line_l1847_184755

theorem min_value_on_line (x y : ℝ) (h : x + 2 * y = 3) :
  2^x + 4^y ≥ 4 * Real.sqrt 2 :=
sorry

end min_value_on_line_l1847_184755


namespace interest_problem_l1847_184716

/-- Proves that given the conditions of the interest problem, the sum is 700 --/
theorem interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 7.5) * 12 / 100 - P * R * 12 / 100 = 630) → P = 700 := by
  sorry

end interest_problem_l1847_184716


namespace complex_not_in_first_quadrant_l1847_184728

theorem complex_not_in_first_quadrant (a : ℝ) : 
  let z : ℂ := (a - Complex.I) / (1 + Complex.I)
  ¬ (z.re > 0 ∧ z.im > 0) := by
  sorry

end complex_not_in_first_quadrant_l1847_184728


namespace opposite_of_four_l1847_184743

-- Define the concept of opposite (additive inverse)
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_four : opposite 4 = -4 := by
  sorry

end opposite_of_four_l1847_184743


namespace min_sum_of_product_36_l1847_184749

theorem min_sum_of_product_36 (c d : ℤ) (h : c * d = 36) :
  ∃ (m : ℤ), m = -37 ∧ c + d ≥ m ∧ ∃ (c' d' : ℤ), c' * d' = 36 ∧ c' + d' = m := by
  sorry

end min_sum_of_product_36_l1847_184749


namespace quadratic_solution_values_second_quadratic_solution_set_l1847_184752

-- Definition for the quadratic inequality
def quadratic_inequality (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 + 3 * x - 2 > 0

-- Definition for the solution set
def solution_set (n : ℝ) (x : ℝ) : Prop :=
  n < x ∧ x < 2

-- Theorem for the first part of the problem
theorem quadratic_solution_values :
  (∀ x, quadratic_inequality m x ↔ solution_set n x) →
  m = -1 ∧ n = 1 :=
sorry

-- Definition for the second quadratic inequality
def second_quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  x^2 + (a - 1) * x - a > 0

-- Theorem for the second part of the problem
theorem second_quadratic_solution_set (a : ℝ) :
  (a < -1 → ∀ x, second_quadratic_inequality a x ↔ (x > 1 ∨ x < -a)) ∧
  (a = -1 → ∀ x, second_quadratic_inequality a x ↔ x ≠ 1) :=
sorry

end quadratic_solution_values_second_quadratic_solution_set_l1847_184752


namespace fletcher_well_diggers_l1847_184732

/-- The number of men hired by Mr Fletcher to dig a well -/
def num_men : ℕ :=
  let hours_day1 : ℕ := 10
  let hours_day2 : ℕ := 8
  let hours_day3 : ℕ := 15
  let total_hours : ℕ := hours_day1 + hours_day2 + hours_day3
  let pay_per_hour : ℕ := 10
  let total_payment : ℕ := 660
  total_payment / (total_hours * pay_per_hour)

theorem fletcher_well_diggers :
  num_men = 2 := by sorry

end fletcher_well_diggers_l1847_184732


namespace six_digit_numbers_at_least_two_zeros_l1847_184713

/-- The total number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers with no zeros -/
def six_digit_numbers_no_zeros : ℕ := 531441

/-- The number of 6-digit numbers with exactly one zero -/
def six_digit_numbers_one_zero : ℕ := 295245

/-- Theorem: The number of 6-digit numbers with at least two zeros is 73,314 -/
theorem six_digit_numbers_at_least_two_zeros :
  total_six_digit_numbers - (six_digit_numbers_no_zeros + six_digit_numbers_one_zero) = 73314 := by
  sorry

end six_digit_numbers_at_least_two_zeros_l1847_184713


namespace inequalities_theorem_l1847_184767

theorem inequalities_theorem (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : 
  (b / a > c / a) ∧ ((b - a) / c > 0) ∧ ((a - c) / (a * c) < 0) := by
  sorry

end inequalities_theorem_l1847_184767

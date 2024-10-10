import Mathlib

namespace alyssa_soccer_spending_l88_8837

/-- Calculates the total amount Alyssa spends on soccer games over three years -/
def total_soccer_spending (
  year1_games : ℕ)
  (year2_in_person : ℕ)
  (year2_missed : ℕ)
  (year2_online : ℕ)
  (year2_streaming_cost : ℕ)
  (year3_in_person : ℕ)
  (year3_online : ℕ)
  (year3_friends_games : ℕ)
  (year3_streaming_cost : ℕ)
  (ticket_price : ℕ) : ℕ :=
  let year1_cost := year1_games * ticket_price
  let year2_cost := year2_in_person * ticket_price + year2_streaming_cost
  let year3_cost := year3_in_person * ticket_price + year3_streaming_cost
  let friends_payback := year3_friends_games * 2 * ticket_price
  year1_cost + year2_cost + year3_cost - friends_payback

/-- Theorem stating that Alyssa's total spending on soccer games over three years is $850 -/
theorem alyssa_soccer_spending :
  total_soccer_spending 13 11 12 8 120 15 10 5 150 20 = 850 := by
  sorry

end alyssa_soccer_spending_l88_8837


namespace min_sum_of_squares_l88_8843

theorem min_sum_of_squares (x y : ℝ) (h : (x + 8) * (y - 8) = 0) :
  ∃ (min : ℝ), min = 128 ∧ ∀ (a b : ℝ), (a + 8) * (b - 8) = 0 → a^2 + b^2 ≥ min :=
sorry

end min_sum_of_squares_l88_8843


namespace sqrt_product_equality_l88_8821

theorem sqrt_product_equality : Real.sqrt 50 * Real.sqrt 18 * Real.sqrt 8 = 60 * Real.sqrt 2 := by
  sorry

end sqrt_product_equality_l88_8821


namespace circle_symmetry_line_l88_8898

/-- A circle C in the xy-plane -/
structure Circle where
  m : ℝ
  equation : ℝ → ℝ → Prop :=
    fun x y => x^2 + y^2 + m*x - 4 = 0

/-- A line in the xy-plane -/
def symmetry_line : ℝ → ℝ → Prop :=
  fun x y => x - y + 4 = 0

/-- Two points are symmetric with respect to a line -/
def symmetric (p1 p2 : ℝ × ℝ) (L : ℝ → ℝ → Prop) : Prop := sorry

theorem circle_symmetry_line (C : Circle) :
  (∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ 
    C.equation p1.1 p1.2 ∧ 
    C.equation p2.1 p2.2 ∧ 
    symmetric p1 p2 symmetry_line) →
  C.m = 8 := by sorry

end circle_symmetry_line_l88_8898


namespace cube_diagonal_length_l88_8890

theorem cube_diagonal_length (S : ℝ) (h : S = 864) :
  ∃ (d : ℝ), d = 12 * Real.sqrt 3 ∧ d^2 = 3 * (S / 6) :=
by sorry

end cube_diagonal_length_l88_8890


namespace smallest_divisor_k_divisibility_at_126_smallest_k_is_126_l88_8813

def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_divisor_k : ∀ k : ℕ, k > 0 → (∀ z : ℂ, f z ∣ (z^k - 1)) → k ≥ 126 :=
by sorry

theorem divisibility_at_126 : ∀ z : ℂ, f z ∣ (z^126 - 1) :=
by sorry

theorem smallest_k_is_126 : (∀ z : ℂ, f z ∣ (z^126 - 1)) ∧ 
  (∀ k : ℕ, k > 0 → k < 126 → ∃ z : ℂ, ¬(f z ∣ (z^k - 1))) :=
by sorry

end smallest_divisor_k_divisibility_at_126_smallest_k_is_126_l88_8813


namespace binomial_coefficient_n_n_binomial_coefficient_1000_1000_l88_8864

theorem binomial_coefficient_n_n (n : ℕ) : Nat.choose n n = 1 := by
  sorry

theorem binomial_coefficient_1000_1000 : Nat.choose 1000 1000 = 1 := by
  sorry

end binomial_coefficient_n_n_binomial_coefficient_1000_1000_l88_8864


namespace triangle_properties_l88_8876

/-- Triangle ABC with given properties -/
structure TriangleABC where
  /-- Point B has coordinates (4,4) -/
  B : ℝ × ℝ
  B_coord : B = (4, 4)
  
  /-- The angle bisector of angle A lies on the line y=0 -/
  angle_bisector_A : ℝ → ℝ
  angle_bisector_A_eq : ∀ x, angle_bisector_A x = 0
  
  /-- The altitude from B to side AC lies on the line x-2y+2=0 -/
  altitude_B : ℝ → ℝ
  altitude_B_eq : ∀ x, altitude_B x = (x + 2) / 2

/-- The main theorem stating the properties of the triangle -/
theorem triangle_properties (t : TriangleABC) :
  ∃ (C : ℝ × ℝ) (area : ℝ),
    C = (10, -8) ∧ 
    area = 48 :=
by sorry

end triangle_properties_l88_8876


namespace max_planes_eq_combinations_l88_8848

/-- The number of points in space -/
def num_points : ℕ := 15

/-- A function that calculates the number of combinations of k items from n items -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

/-- The maximum number of planes determined by the points -/
def max_planes : ℕ := combinations num_points 3

/-- Theorem stating that the maximum number of planes is equal to the number of combinations of 3 points from 15 points -/
theorem max_planes_eq_combinations : 
  max_planes = combinations num_points 3 := by sorry

end max_planes_eq_combinations_l88_8848


namespace P_below_line_l88_8803

/-- A line in 2D space represented by the equation 2x - y + 3 = 0 -/
def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The point P with coordinates (1, -1) -/
def P : Point := ⟨1, -1⟩

/-- A point is below the line if 2x - y + 3 > 0 -/
def is_below (p : Point) : Prop := 2 * p.x - p.y + 3 > 0

theorem P_below_line : is_below P := by
  sorry

end P_below_line_l88_8803


namespace percentage_equation_solution_l88_8814

theorem percentage_equation_solution : ∃ x : ℝ, 
  (x / 100 * 1442 - 36 / 100 * 1412) + 63 = 252 ∧ 
  abs (x - 33.52) < 0.01 := by
  sorry

end percentage_equation_solution_l88_8814


namespace second_to_first_ratio_l88_8840

theorem second_to_first_ratio (x y z : ℝ) : 
  y = 90 →
  z = 4 * y →
  (x + y + z) / 3 = 165 →
  y / x = 2 := by
sorry

end second_to_first_ratio_l88_8840


namespace mechanic_work_hours_l88_8853

/-- Calculates the number of hours a mechanic worked given the total cost, part costs, and labor rate. -/
theorem mechanic_work_hours 
  (total_cost : ℝ) 
  (part_cost : ℝ) 
  (labor_rate_per_minute : ℝ) 
  (h1 : total_cost = 220) 
  (h2 : part_cost = 20) 
  (h3 : labor_rate_per_minute = 0.5) : 
  (total_cost - 2 * part_cost) / (labor_rate_per_minute * 60) = 6 := by
  sorry

end mechanic_work_hours_l88_8853


namespace smallest_number_satisfying_conditions_l88_8835

theorem smallest_number_satisfying_conditions : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 6 = 2 ∧ 
  n % 7 = 3 ∧ 
  n % 8 = 4 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 2 → m % 7 = 3 → m % 8 = 4 → n ≤ m :=
by sorry

end smallest_number_satisfying_conditions_l88_8835


namespace arithmetic_sequence_ratio_l88_8822

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  is_geometric : (a 3)^2 = a 1 * a 7

/-- The main theorem -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) :
  (seq.a 1 + seq.a 3) / (seq.a 2 + seq.a 4) = 3/4 := by
  sorry

end arithmetic_sequence_ratio_l88_8822


namespace triangle_max_side_length_l88_8879

/-- A triangle with three different integer side lengths and a perimeter of 24 units has a maximum side length of 11 units. -/
theorem triangle_max_side_length :
  ∀ a b c : ℕ,
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a + b + c = 24 →
  a ≤ 11 ∧ b ≤ 11 ∧ c ≤ 11 :=
by sorry

end triangle_max_side_length_l88_8879


namespace jeffs_remaining_laps_l88_8833

/-- Given Jeff's swimming requirements and progress, calculate the remaining laps before his break. -/
theorem jeffs_remaining_laps (total_laps : ℕ) (saturday_laps : ℕ) (sunday_morning_laps : ℕ) 
  (h1 : total_laps = 98)
  (h2 : saturday_laps = 27)
  (h3 : sunday_morning_laps = 15) :
  total_laps - saturday_laps - sunday_morning_laps = 56 := by
  sorry

end jeffs_remaining_laps_l88_8833


namespace f_minimum_l88_8894

/-- The quadratic function f(x) = x^2 - 14x + 40 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 40

/-- The value of x that minimizes f(x) -/
def x_min : ℝ := 7

theorem f_minimum :
  ∀ x : ℝ, f x ≥ f x_min :=
sorry

end f_minimum_l88_8894


namespace power_four_five_l88_8812

theorem power_four_five : (4 : ℕ) ^ 4 * (5 : ℕ) ^ 4 = 160000 := by sorry

end power_four_five_l88_8812


namespace unique_prime_sum_diff_l88_8826

theorem unique_prime_sum_diff : ∃! p : ℕ, 
  Prime p ∧ 
  (∃ q r : ℕ, Prime q ∧ Prime r ∧ p = q + r) ∧ 
  (∃ s t : ℕ, Prime s ∧ Prime t ∧ p = s - t) :=
by
  use 5
  sorry

end unique_prime_sum_diff_l88_8826


namespace interest_rate_difference_l88_8851

/-- Proves that the difference in interest rates is 1% given the problem conditions --/
theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (additional_interest : ℝ) 
  (h1 : principal = 2400) 
  (h2 : time = 3) 
  (h3 : additional_interest = 72) : 
  ∃ (r dr : ℝ), 
    principal * ((r + dr) / 100) * time - principal * (r / 100) * time = additional_interest ∧ 
    dr = 1 := by
  sorry

end interest_rate_difference_l88_8851


namespace square_sum_ge_product_sum_l88_8810

theorem square_sum_ge_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end square_sum_ge_product_sum_l88_8810


namespace bill_problem_count_bill_composes_twenty_l88_8847

theorem bill_problem_count : ℕ → Prop :=
  fun b : ℕ =>
    let r := 2 * b  -- Ryan's problem count
    let f := 3 * r  -- Frank's problem count
    let types := 4  -- Number of problem types
    let frank_per_type := 30  -- Frank's problems per type
    f = types * frank_per_type → b = 20

-- Proof
theorem bill_composes_twenty : ∃ b : ℕ, bill_problem_count b :=
  sorry

end bill_problem_count_bill_composes_twenty_l88_8847


namespace jack_reading_pages_l88_8889

theorem jack_reading_pages (pages_per_booklet : ℕ) (number_of_booklets : ℕ) (total_pages : ℕ) :
  pages_per_booklet = 9 →
  number_of_booklets = 49 →
  total_pages = pages_per_booklet * number_of_booklets →
  total_pages = 441 :=
by sorry

end jack_reading_pages_l88_8889


namespace rational_roots_count_l88_8841

/-- The number of distinct possible rational roots for a polynomial of the form
    8x^4 + a₃x³ + a₂x² + a₁x + 16 = 0, where a₃, a₂, and a₁ are integers. -/
def num_rational_roots (a₃ a₂ a₁ : ℤ) : ℕ :=
  16

/-- Theorem stating that the number of distinct possible rational roots for the given polynomial
    is always 16, regardless of the values of a₃, a₂, and a₁. -/
theorem rational_roots_count (a₃ a₂ a₁ : ℤ) :
  num_rational_roots a₃ a₂ a₁ = 16 := by
  sorry

end rational_roots_count_l88_8841


namespace horner_rule_v3_value_l88_8874

def horner_v3 (a b c d e x : ℝ) : ℝ := (((x + a) * x + b) * x + c)

theorem horner_rule_v3_value :
  let f (x : ℝ) := x^4 + 2*x^3 + x^2 - 3*x - 1
  let x : ℝ := 2
  horner_v3 2 1 (-3) (-1) 0 x = 15 := by
  sorry

end horner_rule_v3_value_l88_8874


namespace range_of_x_plus_y_l88_8839

theorem range_of_x_plus_y (x y : Real) 
  (h1 : 0 ≤ y) (h2 : y ≤ x) (h3 : x ≤ π/2)
  (h4 : 4 * (Real.cos y)^2 + 4 * Real.cos x * Real.sin y - 4 * (Real.cos x)^2 ≤ 1) :
  (x + y ∈ Set.Icc 0 (π/6)) ∨ (x + y ∈ Set.Icc (5*π/6) π) := by
  sorry

end range_of_x_plus_y_l88_8839


namespace quadratic_inequality_range_l88_8893

/-- A quadratic function satisfying the given conditions -/
def f (x : ℝ) : ℝ := x^2 - x + 1

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x > 2 * x + m) ↔ m < -5/4 := by
  sorry

end quadratic_inequality_range_l88_8893


namespace workshop_total_workers_l88_8860

/-- Represents the workshop scenario with workers and their salaries -/
structure Workshop where
  avgSalary : ℕ
  technicianCount : ℕ
  technicianAvgSalary : ℕ
  supervisorAvgSalary : ℕ
  laborerAvgSalary : ℕ
  supervisorLaborerTotalSalary : ℕ

/-- Theorem stating that the total number of workers in the workshop is 38 -/
theorem workshop_total_workers (w : Workshop)
  (h1 : w.avgSalary = 9000)
  (h2 : w.technicianCount = 6)
  (h3 : w.technicianAvgSalary = 12000)
  (h4 : w.supervisorAvgSalary = 15000)
  (h5 : w.laborerAvgSalary = 6000)
  (h6 : w.supervisorLaborerTotalSalary = 270000) :
  ∃ (supervisorCount laborerCount : ℕ),
    w.technicianCount + supervisorCount + laborerCount = 38 :=
by sorry

end workshop_total_workers_l88_8860


namespace johnny_signature_dish_count_l88_8888

/-- Represents the number of times Johnny makes his signature crab dish in a day -/
def signature_dish_count : ℕ := sorry

/-- The amount of crab meat used in each signature dish (in pounds) -/
def crab_meat_per_dish : ℚ := 3/2

/-- The price of crab meat per pound (in dollars) -/
def crab_meat_price : ℕ := 8

/-- The total amount Johnny spends on crab meat in a week (in dollars) -/
def weekly_spending : ℕ := 1920

/-- The number of days Johnny's restaurant is closed in a week -/
def closed_days : ℕ := 3

/-- The number of days Johnny's restaurant is open in a week -/
def open_days : ℕ := 7 - closed_days

theorem johnny_signature_dish_count :
  signature_dish_count = 40 :=
by sorry

end johnny_signature_dish_count_l88_8888


namespace book_reading_time_l88_8808

/-- The number of weeks required to read a book -/
def weeks_to_read (total_pages : ℕ) (pages_per_week : ℕ) : ℕ :=
  (total_pages + pages_per_week - 1) / pages_per_week

theorem book_reading_time : 
  let total_pages : ℕ := 2100
  let pages_per_day1 : ℕ := 100
  let pages_per_day2 : ℕ := 150
  let days_type1 : ℕ := 3
  let days_type2 : ℕ := 2
  let pages_per_week : ℕ := pages_per_day1 * days_type1 + pages_per_day2 * days_type2
  weeks_to_read total_pages pages_per_week = 4 := by
sorry

end book_reading_time_l88_8808


namespace functional_relationship_max_profit_remaining_profit_range_l88_8800

-- Define the constants and variables
def cost_price : ℝ := 40
def min_selling_price : ℝ := 44
def max_selling_price : ℝ := 52
def initial_sales : ℝ := 300
def price_increase : ℝ := 1
def sales_decrease : ℝ := 10
def donation : ℝ := 200
def min_remaining_profit : ℝ := 2200

-- Define the functional relationship
def sales (x : ℝ) : ℝ := -10 * x + 740

-- Define the profit function
def profit (x : ℝ) : ℝ := (sales x) * (x - cost_price)

-- State the theorems to be proved
theorem functional_relationship (x : ℝ) (h : min_selling_price ≤ x ∧ x ≤ max_selling_price) :
  sales x = -10 * x + 740 := by sorry

theorem max_profit :
  ∃ (max_x : ℝ), max_x = max_selling_price ∧
  ∀ (x : ℝ), min_selling_price ≤ x ∧ x ≤ max_selling_price →
  profit x ≤ profit max_x ∧ profit max_x = 2640 := by sorry

theorem remaining_profit_range :
  ∀ (x : ℝ), 50 ≤ x ∧ x ≤ 52 ↔ profit x - donation ≥ min_remaining_profit := by sorry

end functional_relationship_max_profit_remaining_profit_range_l88_8800


namespace purple_greater_than_green_less_than_triple_l88_8875

-- Define the probability space
def prob_space : Type := Unit

-- Define the random variables
def X : prob_space → ℝ := sorry
def Y : prob_space → ℝ := sorry

-- Define the probability measure
def P : Set prob_space → ℝ := sorry

-- State the theorem
theorem purple_greater_than_green_less_than_triple (ω : prob_space) : 
  P {ω | X ω < Y ω ∧ Y ω < min (3 * X ω) 1} = 1/3 := by sorry

end purple_greater_than_green_less_than_triple_l88_8875


namespace number_of_girls_in_class_l88_8823

/-- Proves the number of girls in a class given a specific ratio and total number of students -/
theorem number_of_girls_in_class (total : ℕ) (boy_ratio girl_ratio : ℕ) (h_total : total = 260) (h_ratio : boy_ratio = 5 ∧ girl_ratio = 8) :
  (girl_ratio * total) / (boy_ratio + girl_ratio) = 160 := by
  sorry

end number_of_girls_in_class_l88_8823


namespace max_sum_of_first_three_l88_8878

theorem max_sum_of_first_three (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℕ) 
  (h_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧ x₅ < x₆ ∧ x₆ < x₇)
  (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ = 159) :
  (∀ y₁ y₂ y₃ : ℕ, y₁ < y₂ ∧ y₂ < y₃ ∧ 
    (∃ y₄ y₅ y₆ y₇ : ℕ, y₃ < y₄ ∧ y₄ < y₅ ∧ y₅ < y₆ ∧ y₆ < y₇ ∧
      y₁ + y₂ + y₃ + y₄ + y₅ + y₆ + y₇ = 159) →
    y₁ + y₂ + y₃ ≤ 61) ∧
  (x₁ + x₂ + x₃ = 61) := by
sorry

end max_sum_of_first_three_l88_8878


namespace sams_age_l88_8877

theorem sams_age (sam masc : ℕ) 
  (h1 : masc = sam + 7)
  (h2 : sam + masc = 27) : 
  sam = 10 := by
sorry

end sams_age_l88_8877


namespace remaining_pictures_to_color_l88_8899

/-- The number of pictures in each coloring book -/
def pictures_per_book : ℕ := 44

/-- The number of coloring books -/
def num_books : ℕ := 2

/-- The number of pictures already colored -/
def colored_pictures : ℕ := 20

/-- Theorem: Given two coloring books with 44 pictures each, and 20 pictures already colored,
    the number of pictures left to color is 68. -/
theorem remaining_pictures_to_color :
  (num_books * pictures_per_book) - colored_pictures = 68 := by
  sorry

end remaining_pictures_to_color_l88_8899


namespace milk_price_decrease_is_60_percent_l88_8859

/-- Represents the price change of milk powder and coffee from June to July -/
structure PriceChange where
  june_price : ℝ  -- Price of both milk powder and coffee in June
  coffee_increase : ℝ  -- Percentage increase in coffee price
  july_mixture_price : ℝ  -- Price of 3 lbs mixture in July
  july_milk_price : ℝ  -- Price of milk powder per pound in July

/-- Calculates the percentage decrease in milk powder price -/
def milk_price_decrease (pc : PriceChange) : ℝ :=
  -- We'll implement the calculation here
  sorry

/-- Theorem stating that given the conditions, the milk price decrease is 60% -/
theorem milk_price_decrease_is_60_percent (pc : PriceChange) 
  (h1 : pc.coffee_increase = 200)
  (h2 : pc.july_mixture_price = 5.1)
  (h3 : pc.july_milk_price = 0.4) : 
  milk_price_decrease pc = 60 := by
  sorry

end milk_price_decrease_is_60_percent_l88_8859


namespace min_value_and_max_t_l88_8834

/-- Given a > 0, b > 0, and f(x) = |x + a| + |2x - b| with a minimum value of 1 -/
def f (a b x : ℝ) : ℝ := |x + a| + |2*x - b|

theorem min_value_and_max_t (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 1) (hmin_exists : ∃ x, f a b x = 1) :
  (2*a + b = 2) ∧ 
  (∀ t, (∀ a b, a > 0 → b > 0 → a + 2*b ≥ t*a*b) → t ≤ 9/2) ∧
  (∃ t, t = 9/2 ∧ ∀ a b, a > 0 → b > 0 → a + 2*b ≥ t*a*b) :=
by sorry

end min_value_and_max_t_l88_8834


namespace count_prime_in_sequence_count_1973_in_sequence_l88_8885

def generate_sequence (steps : Nat) : List Nat :=
  sorry

def count_occurrences (n : Nat) (list : List Nat) : Nat :=
  sorry

def is_prime (n : Nat) : Prop :=
  sorry

theorem count_prime_in_sequence (p : Nat) (h : is_prime p) :
  count_occurrences p (generate_sequence 1973) = p - 1 :=
sorry

theorem count_1973_in_sequence :
  count_occurrences 1973 (generate_sequence 1973) = 1972 :=
sorry

end count_prime_in_sequence_count_1973_in_sequence_l88_8885


namespace max_value_sqrt_sum_l88_8861

theorem max_value_sqrt_sum (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) :
  Real.sqrt (36 + x) + Real.sqrt (36 - x) + x / 6 ≤ 12 := by
  sorry

end max_value_sqrt_sum_l88_8861


namespace typists_productivity_l88_8844

/-- Given that 20 typists can type 48 letters in 20 minutes, 
    prove that 30 typists can type 216 letters in 1 hour at the same rate. -/
theorem typists_productivity (typists_base : ℕ) (letters_base : ℕ) (minutes_base : ℕ)
  (typists_new : ℕ) (minutes_new : ℕ) :
  typists_base = 20 →
  letters_base = 48 →
  minutes_base = 20 →
  typists_new = 30 →
  minutes_new = 60 →
  (typists_new * letters_base * minutes_new) / (typists_base * minutes_base) = 216 :=
by sorry

end typists_productivity_l88_8844


namespace reachable_region_characterization_l88_8806

/-- A particle's position in a 2D plane -/
structure Particle where
  x : ℝ
  y : ℝ

/-- The speed of the particle along the x-axis -/
def x_speed : ℝ := 2

/-- The speed of the particle elsewhere -/
def other_speed : ℝ := 1

/-- The time limit for the particle's movement -/
def time_limit : ℝ := 1

/-- Check if a point is within the reachable region -/
def is_reachable (p : Particle) : Prop :=
  let o := Particle.mk 0 0
  let a := Particle.mk (1/2) (Real.sqrt 3 / 2)
  let b := Particle.mk 2 0
  let c := Particle.mk 1 0
  (p.x ≥ 0 ∧ p.y ≥ 0) ∧  -- First quadrant
  ((p.x ≤ 2 ∧ p.y ≤ (Real.sqrt 3 * (1 - p.x/2))) ∨  -- Triangle OAB
   (p.x^2 + p.y^2 ≤ 1 ∧ p.y ≥ 0 ∧ p.x ≥ p.y/Real.sqrt 3))  -- Sector OAC

/-- The main theorem stating that a point is reachable if and only if it's in the defined region -/
theorem reachable_region_characterization (p : Particle) :
  (∃ (path : ℝ → Particle), path 0 = Particle.mk 0 0 ∧
    (∀ t, 0 ≤ t ∧ t ≤ time_limit →
      (path t).x^2 + (path t).y^2 ≤ (x_speed * t)^2 ∨
      (path t).x^2 + (path t).y^2 ≤ (other_speed * t)^2) ∧
    path time_limit = p) ↔
  is_reachable p :=
sorry

end reachable_region_characterization_l88_8806


namespace complex_moduli_equality_l88_8857

theorem complex_moduli_equality (a : ℝ) : 
  let z₁ : ℂ := a + 2 * Complex.I
  let z₂ : ℂ := 2 - Complex.I
  Complex.abs z₁ = Complex.abs z₂ → a^2 = 1 := by
sorry

end complex_moduli_equality_l88_8857


namespace chemistry_textbook_weight_l88_8887

/-- The weight of Kelly's chemistry textbook in pounds -/
def chemistry_weight : ℝ := sorry

/-- The weight of Kelly's geometry textbook in pounds -/
def geometry_weight : ℝ := 0.625

theorem chemistry_textbook_weight :
  chemistry_weight = geometry_weight + 6.5 ∧ chemistry_weight = 7.125 := by sorry

end chemistry_textbook_weight_l88_8887


namespace prob_sum_18_correct_l88_8805

/-- The number of faces on each die -/
def num_faces : ℕ := 7

/-- The target sum we're aiming for -/
def target_sum : ℕ := 18

/-- The number of dice being rolled -/
def num_dice : ℕ := 3

/-- The probability of rolling a sum of 18 with three 7-faced dice -/
def prob_sum_18 : ℚ := 4 / 343

/-- Theorem stating that the probability of rolling a sum of 18 
    with three 7-faced dice is 4/343 -/
theorem prob_sum_18_correct :
  prob_sum_18 = (num_favorable_outcomes : ℚ) / (num_faces ^ num_dice) :=
sorry

end prob_sum_18_correct_l88_8805


namespace friends_who_bought_is_five_l88_8884

/-- The number of pencils in one color box -/
def pencils_per_box : ℕ := 7

/-- The total number of pencils -/
def total_pencils : ℕ := 42

/-- The number of color boxes Chloe has -/
def chloe_boxes : ℕ := 1

/-- Calculate the number of friends who bought the color box -/
def friends_who_bought : ℕ :=
  (total_pencils - chloe_boxes * pencils_per_box) / pencils_per_box

theorem friends_who_bought_is_five : friends_who_bought = 5 := by
  sorry

end friends_who_bought_is_five_l88_8884


namespace kenneth_distance_past_finish_line_l88_8854

/-- A proof that Kenneth will be 10 yards past the finish line when Biff crosses it in a 500-yard race -/
theorem kenneth_distance_past_finish_line 
  (race_distance : ℝ) 
  (biff_speed : ℝ) 
  (kenneth_speed : ℝ) 
  (h1 : race_distance = 500)
  (h2 : biff_speed = 50)
  (h3 : kenneth_speed = 51) :
  kenneth_speed * (race_distance / biff_speed) - race_distance = 10 :=
by
  sorry

#check kenneth_distance_past_finish_line

end kenneth_distance_past_finish_line_l88_8854


namespace surface_area_difference_l88_8828

/-- The difference in surface area between 8 unit cubes and a cube with volume 8 -/
theorem surface_area_difference (large_cube_volume : ℝ) (small_cube_volume : ℝ) 
  (num_small_cubes : ℕ) (h1 : large_cube_volume = 8) (h2 : small_cube_volume = 1) 
  (h3 : num_small_cubes = 8) : 
  (num_small_cubes : ℝ) * (6 * small_cube_volume ^ (2/3)) - 
  (6 * large_cube_volume ^ (2/3)) = 24 := by
  sorry

end surface_area_difference_l88_8828


namespace square_vertex_C_l88_8880

def square (A B C D : ℂ) : Prop :=
  (B - A) * Complex.I = C - B ∧
  (C - B) * Complex.I = D - C ∧
  (D - C) * Complex.I = A - D ∧
  (A - D) * Complex.I = B - A

theorem square_vertex_C (A B C D : ℂ) :
  square A B C D →
  A = 1 + 2*Complex.I →
  B = 3 - 5*Complex.I →
  C = 10 - 3*Complex.I :=
by
  sorry

#check square_vertex_C

end square_vertex_C_l88_8880


namespace max_dishes_l88_8829

theorem max_dishes (main_ingredients : Nat) (secondary_ingredients : Nat) (cooking_methods : Nat)
  (select_main : Nat) (select_secondary : Nat) :
  main_ingredients = 5 →
  secondary_ingredients = 8 →
  cooking_methods = 5 →
  select_main = 2 →
  select_secondary = 3 →
  (Nat.choose main_ingredients select_main) *
  (Nat.choose secondary_ingredients select_secondary) *
  cooking_methods = 2800 :=
by sorry

end max_dishes_l88_8829


namespace decimal_equals_fraction_l88_8819

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ
  repeatLength : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + (d.repeatingPart : ℚ) / ((10 ^ d.repeatLength - 1) : ℚ)

/-- The repeating decimal 0.3045045045... -/
def decimal : RepeatingDecimal :=
  { integerPart := 0
  , repeatingPart := 3045
  , repeatLength := 4 }

theorem decimal_equals_fraction : toRational decimal = 383 / 1110 := by
  sorry

end decimal_equals_fraction_l88_8819


namespace triangle_abc_properties_l88_8838

theorem triangle_abc_properties (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  4 * Real.sin A * Real.sin B - 4 * (Real.cos ((A - B) / 2))^2 = Real.sqrt 2 - 2 ∧
  a * Real.sin B / Real.sin A = 4 ∧
  1/2 * a * b * Real.sin C = 8 →
  C = π/4 ∧ c = 4 := by sorry

end triangle_abc_properties_l88_8838


namespace earliest_meeting_time_l88_8862

def kelly_lap_time : ℕ := 5
def rachel_lap_time : ℕ := 8
def mike_lap_time : ℕ := 10

theorem earliest_meeting_time :
  let lap_times := [kelly_lap_time, rachel_lap_time, mike_lap_time]
  Nat.lcm (Nat.lcm kelly_lap_time rachel_lap_time) mike_lap_time = 40 := by
  sorry

end earliest_meeting_time_l88_8862


namespace single_elimination_games_l88_8873

/-- The number of games required to determine a champion in a single-elimination tournament -/
def gamesRequired (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with n players, 
    the number of games required to determine a champion is n - 1 -/
theorem single_elimination_games (n : ℕ) (h : n > 0) : 
  gamesRequired n = n - 1 := by sorry

end single_elimination_games_l88_8873


namespace power_of_a_l88_8820

theorem power_of_a (a b : ℝ) : b = Real.sqrt (3 - a) + Real.sqrt (a - 3) + 2 → a^b = 9 := by
  sorry

end power_of_a_l88_8820


namespace work_completion_days_l88_8818

/-- The number of days it takes person A to complete the work -/
def days_A : ℝ := 20

/-- The number of days it takes person B to complete the work -/
def days_B : ℝ := 30

/-- The number of days A worked before leaving -/
def days_A_worked : ℝ := 10

/-- The number of days B worked to finish the remaining work -/
def days_B_worked : ℝ := 15

/-- Theorem stating that A can complete the work in 20 days -/
theorem work_completion_days :
  (days_A_worked / days_A) + (days_B_worked / days_B) = 1 :=
sorry

end work_completion_days_l88_8818


namespace increasing_iff_a_gt_two_l88_8830

-- Define the linear function
def f (a x : ℝ) : ℝ := (2*a - 4)*x + 3

-- State the theorem
theorem increasing_iff_a_gt_two :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a > 2 := by
  sorry

end increasing_iff_a_gt_two_l88_8830


namespace find_divisor_l88_8867

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 95 ∧ quotient = 6 ∧ remainder = 5 →
  ∃ divisor : Nat, dividend = divisor * quotient + remainder ∧ divisor = 15 := by
sorry

end find_divisor_l88_8867


namespace right_triangle_complex_count_l88_8801

/-- A complex number z satisfies the right triangle property if 0, z, and z^2 form a right triangle
    with the right angle at z. -/
def has_right_triangle_property (z : ℂ) : Prop :=
  z ≠ 0 ∧ 
  (0 : ℂ) ≠ z ∧ 
  z ≠ z^2 ∧
  (z - 0) * (z^2 - z) = 0

/-- There are exactly two complex numbers that satisfy the right triangle property. -/
theorem right_triangle_complex_count : 
  ∃! (s : Finset ℂ), (∀ z ∈ s, has_right_triangle_property z) ∧ s.card = 2 :=
sorry

end right_triangle_complex_count_l88_8801


namespace infinite_fraction_equals_sqrt_15_l88_8827

theorem infinite_fraction_equals_sqrt_15 :
  ∃ x : ℝ, x > 0 ∧ x = 3 + 5 / (2 + 5 / x) → x = Real.sqrt 15 := by
  sorry

end infinite_fraction_equals_sqrt_15_l88_8827


namespace min_packs_needed_l88_8831

def pack_sizes : List Nat := [8, 15, 30]

/-- The target number of cans to be purchased -/
def target_cans : Nat := 120

/-- A function to check if a combination of packs can achieve the target number of cans -/
def achieves_target (x y z : Nat) : Prop :=
  8 * x + 15 * y + 30 * z = target_cans

/-- The minimum number of packs needed -/
def min_packs : Nat := 4

theorem min_packs_needed : 
  (∃ x y z : Nat, achieves_target x y z) ∧ 
  (∀ x y z : Nat, achieves_target x y z → x + y + z ≥ min_packs) ∧
  (∃ x y z : Nat, achieves_target x y z ∧ x + y + z = min_packs) :=
sorry

end min_packs_needed_l88_8831


namespace number_problem_l88_8809

theorem number_problem : ∃ x : ℝ, x / 100 = 31.76 + 0.28 ∧ x = 3204 := by
  sorry

end number_problem_l88_8809


namespace coefficient_of_x_is_10_l88_8891

/-- The coefficient of x in the expansion of (x^2 + 1/x)^5 -/
def coefficient_of_x : ℕ :=
  (Nat.choose 5 3)

theorem coefficient_of_x_is_10 : coefficient_of_x = 10 := by
  sorry

end coefficient_of_x_is_10_l88_8891


namespace unique_positive_solution_l88_8836

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (2/3) * x = (144/216) * (1/x) := by
  sorry

end unique_positive_solution_l88_8836


namespace exists_cutting_method_for_person_to_fit_l88_8842

/-- Represents a sheet of paper -/
structure Sheet :=
  (length : ℝ)
  (width : ℝ)
  (thickness : ℝ)

/-- Represents a person -/
structure Person :=
  (height : ℝ)
  (width : ℝ)

/-- Represents a cutting method -/
structure CuttingMethod :=
  (cuts : List (ℝ × ℝ))  -- List of cut coordinates

/-- Represents the result of applying a cutting method to a sheet -/
def apply_cutting_method (s : Sheet) (cm : CuttingMethod) : ℝ := sorry

/-- Determines if a person can fit through an opening -/
def can_fit_through (p : Person) (opening_size : ℝ) : Prop := sorry

/-- Main theorem: There exists a cutting method that creates an opening large enough for a person -/
theorem exists_cutting_method_for_person_to_fit (s : Sheet) (p : Person) : 
  ∃ (cm : CuttingMethod), can_fit_through p (apply_cutting_method s cm) :=
sorry

end exists_cutting_method_for_person_to_fit_l88_8842


namespace sum_of_inscribed_squares_l88_8883

/-- The sum of areas of an infinite series of inscribed squares -/
theorem sum_of_inscribed_squares (a : ℝ) (h : a > 0) :
  ∃ S : ℝ, S = (4 * a^2) / 3 ∧ 
  S = a^2 + ∑' n, (a^2 / 4^n) := by
  sorry

end sum_of_inscribed_squares_l88_8883


namespace train_passing_time_l88_8855

/-- Proves that a train with given length and speed takes the calculated time to pass a fixed point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 300 ∧ train_speed_kmh = 90 → 
  (train_length / (train_speed_kmh * (5/18))) = 12 := by
  sorry

end train_passing_time_l88_8855


namespace linear_function_problem_l88_8807

/-- A linear function satisfying specific conditions -/
def f (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x + b

/-- The theorem statement -/
theorem linear_function_problem (a b : ℝ) :
  (∀ x, f a b x = 3 * (f a b).invFun x ^ 2 + 5) →
  f a b 0 = 2 →
  f a b 3 = 3 * Real.sqrt 5 + 2 := by
  sorry

end linear_function_problem_l88_8807


namespace ellipse_intersection_theorem_l88_8804

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define the line that intersects C
def Line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the condition for OA and OB to be perpendicular
def Perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

-- Main theorem
theorem ellipse_intersection_theorem :
  ∀ k x₁ y₁ x₂ y₂ : ℝ,
  C x₁ y₁ ∧ C x₂ y₂ ∧
  Line k x₁ y₁ ∧ Line k x₂ y₂ ∧
  Perpendicular x₁ y₁ x₂ y₂ →
  (k = 1/2 ∨ k = -1/2) ∧
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = (4*Real.sqrt 65/17)^2 :=
sorry

end ellipse_intersection_theorem_l88_8804


namespace geometric_sequence_product_l88_8896

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_sequence_product (x y z : ℝ) :
  is_geometric_sequence (-1) x y z (-2) →
  x * y * z = -2 := by
  sorry

end geometric_sequence_product_l88_8896


namespace tobias_shoe_purchase_l88_8881

/-- Tobias's shoe purchase problem -/
theorem tobias_shoe_purchase (shoe_cost : ℕ) (saving_months : ℕ) (monthly_allowance : ℕ)
  (lawn_charge : ℕ) (lawns_mowed : ℕ) (driveways_shoveled : ℕ) (change : ℕ)
  (h1 : shoe_cost = 95)
  (h2 : saving_months = 3)
  (h3 : monthly_allowance = 5)
  (h4 : lawn_charge = 15)
  (h5 : lawns_mowed = 4)
  (h6 : driveways_shoveled = 5)
  (h7 : change = 15) :
  ∃ (driveway_charge : ℕ),
    shoe_cost + change =
      saving_months * monthly_allowance +
      lawns_mowed * lawn_charge +
      driveways_shoveled * driveway_charge ∧
    driveway_charge = 7 :=
by sorry

end tobias_shoe_purchase_l88_8881


namespace expression_value_l88_8825

theorem expression_value : 
  let c : ℚ := 1
  (1 + c + 1/1) * (1 + c + 1/2) * (1 + c + 1/3) * (1 + c + 1/4) * (1 + c + 1/5) = 133/20 :=
by sorry

end expression_value_l88_8825


namespace reflection_across_y_axis_l88_8817

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (-2, 3)

/-- The reflected point -/
def reflected_point : ℝ × ℝ := (2, 3)

theorem reflection_across_y_axis :
  reflect_y original_point = reflected_point := by sorry

end reflection_across_y_axis_l88_8817


namespace complex_number_quadrant_l88_8869

theorem complex_number_quadrant (z : ℂ) (h : z = 1 - 2*I) : 
  (z.re > 0 ∧ z.im < 0) :=
by sorry

end complex_number_quadrant_l88_8869


namespace square_decomposition_l88_8895

theorem square_decomposition (a : ℤ) :
  a^2 + 5*a + 7 = (a + 3) * (a + 2)^2 + (a + 2) * 1^2 := by
  sorry

end square_decomposition_l88_8895


namespace incircle_radius_of_special_triangle_l88_8892

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the radius of the incircle is 2 under the following conditions:
    - a, b, c form an arithmetic sequence
    - c = 10
    - a cos A = b cos B
    - A ≠ B -/
theorem incircle_radius_of_special_triangle 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_arithmetic : ∃ (d : ℝ), b = a + d ∧ c = a + 2*d)
  (h_c : c = 10)
  (h_cos : a * Real.cos A = b * Real.cos B)
  (h_angle_neq : A ≠ B) :
  let s := (a + b + c) / 2
  (s - a) * (s - b) * (s - c) / s = 4 :=
by sorry

end incircle_radius_of_special_triangle_l88_8892


namespace circumcircle_area_of_special_triangle_l88_8845

open Real

/-- Triangle ABC with given properties --/
structure Triangle where
  A : ℝ  -- Angle A in radians
  b : ℝ  -- Side length b
  area : ℝ  -- Area of the triangle

/-- The area of the circumcircle of a triangle --/
def circumcircle_area (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating the area of the circumcircle for the given triangle --/
theorem circumcircle_area_of_special_triangle :
  let t : Triangle := {
    A := π/4,  -- 45° in radians
    b := 2 * sqrt 2,
    area := 1
  }
  circumcircle_area t = 5*π/2 := by
  sorry

end circumcircle_area_of_special_triangle_l88_8845


namespace male_students_count_l88_8866

theorem male_students_count (total_students : ℕ) (sample_size : ℕ) (female_in_sample : ℕ) :
  total_students = 800 →
  sample_size = 40 →
  female_in_sample = 11 →
  (sample_size - female_in_sample) * total_students = 580 * sample_size :=
by sorry

end male_students_count_l88_8866


namespace root_value_theorem_l88_8858

theorem root_value_theorem (a : ℝ) : 
  a^2 - 4*a + 3 = 0 → -2*a^2 + 8*a - 5 = 1 := by
  sorry

end root_value_theorem_l88_8858


namespace weight_difference_l88_8886

def john_weight : ℕ := 81
def roy_weight : ℕ := 4

theorem weight_difference : john_weight - roy_weight = 77 := by
  sorry

end weight_difference_l88_8886


namespace inequality_solution_range_l88_8811

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 5 ∧ x^2 + a*x - 2 > 0) → a > -23/5 := by
  sorry

end inequality_solution_range_l88_8811


namespace range_of_a_l88_8868

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 := by
  sorry

end range_of_a_l88_8868


namespace cos105_cos45_plus_sin45_sin105_eq_half_l88_8870

theorem cos105_cos45_plus_sin45_sin105_eq_half :
  Real.cos (105 * π / 180) * Real.cos (45 * π / 180) +
  Real.sin (45 * π / 180) * Real.sin (105 * π / 180) = 1 / 2 := by
  sorry

end cos105_cos45_plus_sin45_sin105_eq_half_l88_8870


namespace square_side_length_l88_8816

/-- Given a square with diagonal length 4, prove that its side length is 2√2 -/
theorem square_side_length (d : ℝ) (h : d = 4) : ∃ (s : ℝ), s^2 + s^2 = d^2 ∧ s = 2 * Real.sqrt 2 := by
  sorry

end square_side_length_l88_8816


namespace added_number_proof_l88_8852

theorem added_number_proof (n : ℕ) (original_avg new_avg : ℚ) (added_num : ℚ) : 
  n = 15 →
  original_avg = 17 →
  new_avg = 20 →
  (n : ℚ) * original_avg + added_num = (n + 1 : ℚ) * new_avg →
  added_num = 65 := by
  sorry

end added_number_proof_l88_8852


namespace long_tennis_players_l88_8850

theorem long_tennis_players (total : ℕ) (football : ℕ) (both : ℕ) (neither : ℕ) :
  total = 40 →
  football = 26 →
  both = 17 →
  neither = 11 →
  ∃ long_tennis : ℕ,
    long_tennis = 20 ∧
    total = football + long_tennis - both + neither :=
by
  sorry

end long_tennis_players_l88_8850


namespace geometric_sequence_sum_l88_8871

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 1 * a 3 = 4 →
  a 4 = 8 →
  (a 1 + q = 3 ∨ a 1 + q = -3) :=
by
  sorry

end geometric_sequence_sum_l88_8871


namespace dot_path_length_on_rolling_cube_l88_8872

/-- The length of the path followed by a dot on a rolling cube -/
theorem dot_path_length_on_rolling_cube :
  ∀ (cube_edge : ℝ) (dot_path : ℝ),
  cube_edge = 2 →
  dot_path = 2 * Real.sqrt 2 * Real.pi →
  dot_path = (cube_edge * Real.sqrt 2) * Real.pi :=
by sorry

end dot_path_length_on_rolling_cube_l88_8872


namespace base_conversion_subtraction_l88_8815

/-- Converts a number from base 11 to base 10 -/
def base11ToBase10 (n : ℕ) : ℤ := sorry

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (n : ℕ) : ℤ := sorry

/-- Represents the digit E in base 14 -/
def E : ℕ := 14

theorem base_conversion_subtraction :
  base11ToBase10 373 - base14ToBase10 (4 * 14 * 14 + E * 14 + 5) = -542 := by sorry

end base_conversion_subtraction_l88_8815


namespace relative_minimum_condition_l88_8856

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - x^3 - x^2 + a*x + 1

/-- The first derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 4*x^3 - 3*x^2 - 2*x + a

/-- The second derivative of f(x) -/
def f'' (x : ℝ) : ℝ := 12*x^2 - 6*x - 2

/-- Theorem stating that f(a) = a is a relative minimum iff a = 1 -/
theorem relative_minimum_condition (a : ℝ) :
  (f a a = a ∧ ∀ x, x ≠ a → f a x ≥ f a a) ↔ a = 1 := by sorry

end relative_minimum_condition_l88_8856


namespace solve_for_y_l88_8846

theorem solve_for_y (x y : ℝ) (h1 : x + 2*y = 10) (h2 : x = 8) : y = 1 := by
  sorry

end solve_for_y_l88_8846


namespace a_range_l88_8863

theorem a_range (a : ℝ) : a < 9 * a^3 - 11 * a ∧ 9 * a^3 - 11 * a < |a| → -2 * Real.sqrt 3 / 3 < a ∧ a < -Real.sqrt 10 / 3 := by
  sorry

end a_range_l88_8863


namespace sequence_a_closed_form_l88_8865

def sequence_a : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 6
  | (n + 3) => (n + 7) * sequence_a (n + 2) - 4 * (n + 3) * sequence_a (n + 1) + (4 * (n + 3) - 8) * sequence_a n

theorem sequence_a_closed_form (n : ℕ) : sequence_a n = n.factorial + 2^n := by
  sorry

end sequence_a_closed_form_l88_8865


namespace right_triangle_perimeter_l88_8897

theorem right_triangle_perimeter 
  (area : ℝ) 
  (leg : ℝ) 
  (h1 : area = 150) 
  (h2 : leg = 30) : 
  ∃ (other_leg hypotenuse : ℝ),
    area = (1/2) * leg * other_leg ∧
    hypotenuse^2 = leg^2 + other_leg^2 ∧
    leg + other_leg + hypotenuse = 40 + 10 * Real.sqrt 10 := by
  sorry

end right_triangle_perimeter_l88_8897


namespace jen_shooting_game_times_l88_8832

theorem jen_shooting_game_times (shooting_cost carousel_cost russel_rides total_tickets : ℕ) 
  (h1 : shooting_cost = 5)
  (h2 : carousel_cost = 3)
  (h3 : russel_rides = 3)
  (h4 : total_tickets = 19) :
  ∃ (jen_times : ℕ), jen_times * shooting_cost + russel_rides * carousel_cost = total_tickets ∧ jen_times = 2 := by
  sorry

end jen_shooting_game_times_l88_8832


namespace magnitude_of_linear_combination_l88_8882

/-- Given two plane vectors α and β, prove that |2α + β| = √10 -/
theorem magnitude_of_linear_combination (α β : ℝ × ℝ) 
  (h1 : ‖α‖ = 1) 
  (h2 : ‖β‖ = 2) 
  (h3 : α • (α - 2 • β) = 0) : 
  ‖2 • α + β‖ = Real.sqrt 10 := by
sorry

end magnitude_of_linear_combination_l88_8882


namespace quadratic_inequality_properties_l88_8824

/-- Given that the solution set of ax² + bx + c > 0 is {x | -3 < x < 2}, prove the following statements -/
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x : ℝ, ax^2 + b*x + c > 0 ↔ -3 < x ∧ x < 2) :
  (a < 0) ∧
  (a + b + c > 0) ∧
  (∀ x : ℝ, c*x^2 + b*x + a < 0 ↔ -1/3 < x ∧ x < 1/2) :=
by sorry

end quadratic_inequality_properties_l88_8824


namespace ratio_equality_l88_8849

variables {a b c : ℝ}

theorem ratio_equality (h1 : 7 * a = 8 * b) (h2 : 4 * a + 3 * c = 11 * b) (h3 : 2 * c - b = 5 * a) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : a / 8 = b / 7 := by
  sorry

end ratio_equality_l88_8849


namespace ellipse_and_line_intersection_l88_8802

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y b : ℝ) : Prop := y = (1/2) * x + b

-- Theorem statement
theorem ellipse_and_line_intersection :
  -- Given conditions
  let left_focus : ℝ × ℝ := (-Real.sqrt 3, 0)
  let right_vertex : ℝ × ℝ := (2, 0)
  
  -- Prove the following
  -- 1. Standard equation of ellipse C
  ∀ x y : ℝ, ellipse_C x y ↔ x^2 / 4 + y^2 = 1
  
  -- 2. Maximum chord length and corresponding line equation
  ∧ ∃ max_length : ℝ,
    (max_length = Real.sqrt 10) ∧
    (∀ b : ℝ, ∃ A B : ℝ × ℝ,
      ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
      line_l A.1 A.2 b ∧ line_l B.1 B.2 b ∧
      (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ max_length)) ∧
    (∃ A B : ℝ × ℝ,
      ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
      line_l A.1 A.2 0 ∧ line_l B.1 B.2 0 ∧
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = max_length) := by
  sorry

end ellipse_and_line_intersection_l88_8802

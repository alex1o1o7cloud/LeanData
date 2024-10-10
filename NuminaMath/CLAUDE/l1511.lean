import Mathlib

namespace max_angle_at_3_2_l1511_151165

/-- The line l: x + y - 5 = 0 -/
def line (x y : ℝ) : Prop := x + y - 5 = 0

/-- Point A -/
def A : ℝ × ℝ := (1, 0)

/-- Point B -/
def B : ℝ × ℝ := (3, 0)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Statement: The point (3,2) maximizes the angle APB on the given line -/
theorem max_angle_at_3_2 :
  line 3 2 ∧
  ∀ x y, line x y → angle A (x, y) B ≤ angle A (3, 2) B :=
by sorry

end max_angle_at_3_2_l1511_151165


namespace power_seven_strictly_increasing_l1511_151189

theorem power_seven_strictly_increasing (m n : ℝ) (h : m < n) : m^7 < n^7 := by
  sorry

end power_seven_strictly_increasing_l1511_151189


namespace contrapositive_even_sum_l1511_151105

theorem contrapositive_even_sum (x y : ℤ) :
  (¬(Even (x + y)) → ¬(Even x ∧ Even y)) ↔
  (∀ x y : ℤ, Even x ∧ Even y → Even (x + y)) :=
by sorry

end contrapositive_even_sum_l1511_151105


namespace intersection_implies_m_range_l1511_151188

theorem intersection_implies_m_range (m : ℝ) :
  (∃ x : ℝ, m * (4 : ℝ)^x - 3 * (2 : ℝ)^(x + 1) - 2 = 0) →
  m ≥ -9/2 := by
sorry

end intersection_implies_m_range_l1511_151188


namespace ancient_chinese_car_problem_l1511_151160

/-- The number of cars in the ancient Chinese problem -/
def num_cars : ℕ := 15

/-- The number of people that can be accommodated when 3 people share a car -/
def people_three_per_car (x : ℕ) : ℕ := 3 * (x - 2)

/-- The number of people that can be accommodated when 2 people share a car -/
def people_two_per_car (x : ℕ) : ℕ := 2 * x

theorem ancient_chinese_car_problem :
  (people_three_per_car num_cars = people_two_per_car num_cars + 9) ∧
  (num_cars > 2) := by
  sorry

end ancient_chinese_car_problem_l1511_151160


namespace circles_intersect_l1511_151198

-- Define Circle C1
def C1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 1

-- Define Circle C2
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 1 = 0

-- Theorem stating that C1 and C2 intersect
theorem circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y := by
  sorry

end circles_intersect_l1511_151198


namespace odd_function_inequality_l1511_151155

-- Define the properties of the function f
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def HasPositiveProduct (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

-- State the theorem
theorem odd_function_inequality (f : ℝ → ℝ) 
  (h_odd : IsOddFunction f) (h_pos : HasPositiveProduct f) : 
  f 4 < f (-6) := by
  sorry

end odd_function_inequality_l1511_151155


namespace bonsai_earnings_proof_l1511_151131

/-- Calculates the total earnings from selling bonsai. -/
def total_earnings (small_cost big_cost : ℕ) (small_sold big_sold : ℕ) : ℕ :=
  small_cost * small_sold + big_cost * big_sold

/-- Proves that the total earnings from selling 3 small bonsai at $30 each
    and 5 big bonsai at $20 each is equal to $190. -/
theorem bonsai_earnings_proof :
  total_earnings 30 20 3 5 = 190 := by
  sorry

end bonsai_earnings_proof_l1511_151131


namespace fibonacci_determinant_identity_fibonacci_1002_1004_minus_1003_squared_l1511_151115

def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_determinant_identity (n : ℕ) (h : n > 0) :
  fib (n - 1) * fib (n + 1) - fib n ^ 2 = (-1) ^ n := by
  sorry

-- The specific case for n = 1003
theorem fibonacci_1002_1004_minus_1003_squared :
  fib 1002 * fib 1004 - fib 1003 ^ 2 = -1 := by
  sorry

end fibonacci_determinant_identity_fibonacci_1002_1004_minus_1003_squared_l1511_151115


namespace work_payment_theorem_l1511_151138

/-- Represents the time (in days) it takes for a person to complete the work alone -/
structure WorkTime where
  days : ℚ
  days_pos : days > 0

/-- Represents a worker with their work time and share of payment -/
structure Worker where
  work_time : WorkTime
  share : ℚ
  share_nonneg : share ≥ 0

/-- Calculates the total payment for a job given two workers' information -/
def total_payment (worker1 worker2 : Worker) : ℚ :=
  let total_work_rate := 1 / worker1.work_time.days + 1 / worker2.work_time.days
  let total_parts := total_work_rate * worker1.work_time.days * worker2.work_time.days
  let worker1_parts := worker2.work_time.days
  worker1.share * total_parts / worker1_parts

/-- The main theorem stating the total payment for the work -/
theorem work_payment_theorem (rahul rajesh : Worker) 
    (h1 : rahul.work_time.days = 3)
    (h2 : rajesh.work_time.days = 2)
    (h3 : rahul.share = 900) :
    total_payment rahul rajesh = 2250 := by
  sorry


end work_payment_theorem_l1511_151138


namespace sum_of_digits_successor_l1511_151123

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: If S(n) = 4387, then S(n+1) = 4388 -/
theorem sum_of_digits_successor (n : ℕ) (h : sum_of_digits n = 4387) : 
  sum_of_digits (n + 1) = 4388 := by sorry

end sum_of_digits_successor_l1511_151123


namespace max_cone_section_area_l1511_151196

/-- The maximum area of a cone section passing through the vertex, given the cone's height and volume --/
theorem max_cone_section_area (h : ℝ) (v : ℝ) : h = 1 → v = π → 
  ∃ (max_area : ℝ), max_area = 2 ∧ 
  ∀ (section_area : ℝ), section_area ≤ max_area := by
  sorry


end max_cone_section_area_l1511_151196


namespace fuel_station_problem_l1511_151143

/-- Represents the number of trucks filled up at a fuel station. -/
def num_trucks : ℕ := 2

theorem fuel_station_problem :
  let service_cost : ℚ := 21/10
  let fuel_cost_per_liter : ℚ := 7/10
  let num_minivans : ℕ := 3
  let total_cost : ℚ := 3472/10
  let minivan_capacity : ℚ := 65
  let truck_capacity : ℚ := minivan_capacity * 220/100
  
  let minivan_fuel_cost : ℚ := num_minivans * minivan_capacity * fuel_cost_per_liter
  let minivan_service_cost : ℚ := num_minivans * service_cost
  let total_minivan_cost : ℚ := minivan_fuel_cost + minivan_service_cost
  
  let truck_cost : ℚ := total_cost - total_minivan_cost
  let single_truck_fuel_cost : ℚ := truck_capacity * fuel_cost_per_liter
  let single_truck_total_cost : ℚ := single_truck_fuel_cost + service_cost
  
  num_trucks = (truck_cost / single_truck_total_cost).num :=
by sorry

#check fuel_station_problem

end fuel_station_problem_l1511_151143


namespace people_lifting_weights_l1511_151104

/-- The number of people in the gym at the start of Bethany's shift -/
def initial_people : ℕ := sorry

/-- The number of people who arrived during Bethany's shift -/
def arrivals : ℕ := 5

/-- The number of people who left during Bethany's shift -/
def departures : ℕ := 2

/-- The total number of people in the gym after the changes -/
def final_people : ℕ := 19

theorem people_lifting_weights : initial_people = 16 :=
  by sorry

end people_lifting_weights_l1511_151104


namespace equal_amount_after_15_days_l1511_151192

/-- The number of days it takes for Minjeong's and Soohyeok's piggy bank amounts to become equal -/
def days_to_equal_amount (minjeong_initial : ℕ) (soohyeok_initial : ℕ) 
                         (minjeong_daily : ℕ) (soohyeok_daily : ℕ) : ℕ :=
  15

theorem equal_amount_after_15_days 
  (minjeong_initial : ℕ) (soohyeok_initial : ℕ) 
  (minjeong_daily : ℕ) (soohyeok_daily : ℕ)
  (h1 : minjeong_initial = 8000)
  (h2 : soohyeok_initial = 5000)
  (h3 : minjeong_daily = 300)
  (h4 : soohyeok_daily = 500) :
  minjeong_initial + 15 * minjeong_daily = soohyeok_initial + 15 * soohyeok_daily :=
by
  sorry

#eval days_to_equal_amount 8000 5000 300 500

end equal_amount_after_15_days_l1511_151192


namespace solve_equation_l1511_151152

theorem solve_equation : ∃ x : ℚ, (3 * x - 4) / 7 = 15 ∧ x = 109 / 3 := by
  sorry

end solve_equation_l1511_151152


namespace complement_of_A_l1511_151181

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 0}

theorem complement_of_A : Set.compl A = {x : ℝ | x < 0} := by sorry

end complement_of_A_l1511_151181


namespace distance_between_rectangle_vertices_l1511_151136

/-- Given an acute-angled triangle ABC with AB = √3, AC = 1, and angle BAC = 60°,
    and equal rectangles AMNB and APQC built outward on sides AB and AC respectively,
    the distance between vertices N and Q is 2√(2 + √3). -/
theorem distance_between_rectangle_vertices (A B C M N P Q : ℝ × ℝ) :
  let AB := Real.sqrt 3
  let AC := 1
  let angle_BAC := 60 * π / 180
  -- Triangle ABC properties
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = AB^2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = AC^2 →
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = AB * AC * Real.cos angle_BAC →
  -- Rectangle properties
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = (P.1 - A.1)^2 + (P.2 - A.2)^2 →
  (N.1 - B.1)^2 + (N.2 - B.2)^2 = (Q.1 - C.1)^2 + (Q.2 - C.2)^2 →
  (M.1 - A.1) * (B.1 - A.1) + (M.2 - A.2) * (B.2 - A.2) = 0 →
  (P.1 - A.1) * (C.1 - A.1) + (P.2 - A.2) * (C.2 - A.2) = 0 →
  -- Conclusion
  (N.1 - Q.1)^2 + (N.2 - Q.2)^2 = 4 * (2 + Real.sqrt 3) := by sorry

end distance_between_rectangle_vertices_l1511_151136


namespace B_subset_A_l1511_151109

def A : Set ℝ := {x | x ≥ 1}
def B : Set ℝ := {x | x > 2}

theorem B_subset_A : B ⊆ A := by sorry

end B_subset_A_l1511_151109


namespace complex_product_pure_imaginary_l1511_151195

/-- A complex number is pure imaginary if its real part is zero. -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0

/-- The problem statement -/
theorem complex_product_pure_imaginary (a : ℝ) :
  IsPureImaginary ((a : ℂ) + Complex.I * (2 - Complex.I)) → a = -1/2 := by
  sorry

end complex_product_pure_imaginary_l1511_151195


namespace pete_has_enough_money_l1511_151112

/-- Represents the amount of money Pete has and owes -/
structure PetesMoney where
  wallet_twenty : Nat -- number of $20 bills in wallet
  wallet_ten : Nat -- number of $10 bills in wallet
  wallet_pounds : Nat -- number of £5 notes in wallet
  pocket_ten : Nat -- number of $10 bills in pocket
  owed : Nat -- amount owed on the bike in dollars

/-- Calculates the total amount of money Pete has in dollars -/
def total_money (m : PetesMoney) : Nat :=
  m.wallet_twenty * 20 + m.wallet_ten * 10 + m.wallet_pounds * 7 + m.pocket_ten * 10

/-- Proves that Pete has enough money to pay off his bike debt -/
theorem pete_has_enough_money (m : PetesMoney) 
  (h1 : m.wallet_twenty = 2)
  (h2 : m.wallet_ten = 1)
  (h3 : m.wallet_pounds = 1)
  (h4 : m.pocket_ten = 4)
  (h5 : m.owed = 90) :
  total_money m ≥ m.owed :=
by sorry

end pete_has_enough_money_l1511_151112


namespace folding_square_pt_length_l1511_151180

/-- Square with special folding property -/
structure FoldingSquare where
  -- Side length of the square
  side_length : ℝ
  -- Length of PT (and SU by symmetry)
  pt_length : ℝ
  -- Condition that when folded, PR and SR coincide on diagonal RQ
  folding_condition : pt_length ≤ side_length ∧ 
    2 * (side_length - pt_length) / Real.sqrt 2 = side_length * Real.sqrt 2

/-- Theorem about the specific square in the problem -/
theorem folding_square_pt_length :
  ∀ (sq : FoldingSquare), 
  sq.side_length = 2 → 
  sq.pt_length = Real.sqrt 8 - 2 := by
  sorry

end folding_square_pt_length_l1511_151180


namespace smallest_integer_in_set_l1511_151174

theorem smallest_integer_in_set (m : ℤ) : 
  (m + 3 < 3*m - 5) → m = 5 := by
  sorry

end smallest_integer_in_set_l1511_151174


namespace arctan_sum_l1511_151119

theorem arctan_sum : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by sorry

end arctan_sum_l1511_151119


namespace complex_magnitude_product_l1511_151161

theorem complex_magnitude_product : 
  Complex.abs ((5 * Real.sqrt 2 - Complex.I * 3) * (2 * Real.sqrt 3 + Complex.I * 4)) = 2 * Real.sqrt 413 := by
  sorry

end complex_magnitude_product_l1511_151161


namespace quadratic_function_proof_l1511_151185

theorem quadratic_function_proof (f g : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →  -- f is quadratic
  f 0 = 12 →                                      -- f(0) = 12
  (∀ x, g x = 2^x * f x) →                        -- g(x) = 2^x * f(x)
  (∀ x, g (x + 1) - g x ≥ 2^(x + 1) * x^2) →      -- g(x+1) - g(x) ≥ 2^(x+1) * x^2
  (∀ x, f x = 2 * x^2 - 8 * x + 12) ∧             -- f(x) = 2x^2 - 8x + 12
  (∀ x, g x = (2 * x^2 - 8 * x + 12) * 2^x) :=    -- g(x) = (2x^2 - 8x + 12) * 2^x
by sorry

end quadratic_function_proof_l1511_151185


namespace sprinter_probabilities_l1511_151197

/-- Probabilities of three independent events -/
def prob_A : ℚ := 2/5
def prob_B : ℚ := 3/4
def prob_C : ℚ := 1/3

/-- Probability of all three events occurring -/
def prob_all_three : ℚ := prob_A * prob_B * prob_C

/-- Probability of exactly two events occurring -/
def prob_two : ℚ := 
  prob_A * prob_B * (1 - prob_C) + 
  prob_A * (1 - prob_B) * prob_C + 
  (1 - prob_A) * prob_B * prob_C

/-- Probability of at least one event occurring -/
def prob_at_least_one : ℚ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

theorem sprinter_probabilities :
  prob_all_three = 1/10 ∧ 
  prob_two = 23/60 ∧ 
  prob_at_least_one = 9/10 := by
  sorry

end sprinter_probabilities_l1511_151197


namespace tenth_term_is_399_l1511_151175

def a (n : ℕ) : ℕ := (2*n - 1) * (2*n + 1)

theorem tenth_term_is_399 : a 10 = 399 := by sorry

end tenth_term_is_399_l1511_151175


namespace symmetric_point_l1511_151146

/-- Given a point (a, b) and a line x + y + 1 = 0, the point symmetric to (a, b) with respect to the line is (-b-1, -a-1) -/
theorem symmetric_point (a b : ℝ) : 
  let original_point := (a, b)
  let line_equation (x y : ℝ) := x + y + 1 = 0
  let symmetric_point := (-b - 1, -a - 1)
  ∀ x y, line_equation x y → 
    (x - a) ^ 2 + (y - b) ^ 2 = (x - (-b - 1)) ^ 2 + (y - (-a - 1)) ^ 2 ∧
    line_equation ((a + (-b - 1)) / 2) ((b + (-a - 1)) / 2) := by
  sorry

end symmetric_point_l1511_151146


namespace interest_rate_frequency_relationship_l1511_151116

/-- The nominal annual interest rate -/
def nominal_rate : ℝ := 0.16

/-- The effective annual interest rate -/
def effective_rate : ℝ := 0.1664

/-- The frequency of interest payments per year -/
def frequency : ℕ := 2

/-- Theorem stating that the given frequency satisfies the relationship between nominal and effective rates -/
theorem interest_rate_frequency_relationship : 
  (1 + nominal_rate / frequency)^frequency - 1 = effective_rate := by sorry

end interest_rate_frequency_relationship_l1511_151116


namespace conner_average_speed_l1511_151199

/-- The average speed of Conner's dune buggy given different terrain conditions -/
theorem conner_average_speed 
  (flat_speed : ℝ) 
  (downhill_speed_increase : ℝ) 
  (uphill_speed_decrease : ℝ) 
  (h1 : flat_speed = 60) 
  (h2 : downhill_speed_increase = 12) 
  (h3 : uphill_speed_decrease = 18) :
  (flat_speed + (flat_speed + downhill_speed_increase) + (flat_speed - uphill_speed_decrease)) / 3 = 58 := by
  sorry

end conner_average_speed_l1511_151199


namespace degenerate_ellipse_max_y_coordinate_l1511_151118

theorem degenerate_ellipse_max_y_coordinate :
  ∀ x y : ℝ, (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 :=
by sorry

end degenerate_ellipse_max_y_coordinate_l1511_151118


namespace correct_calculation_of_one_fifth_sum_of_acute_angles_l1511_151166

-- Define acute angle
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

-- Theorem statement
theorem correct_calculation_of_one_fifth_sum_of_acute_angles 
  (α β : ℝ) 
  (h_α : is_acute_angle α) 
  (h_β : is_acute_angle β) : 
  18 < (1/5) * (α + β) ∧ 
  (1/5) * (α + β) < 54 ∧ 
  (42 ∈ {17, 42, 56, 73} ∩ Set.Icc 18 54) ∧ 
  ({17, 42, 56, 73} ∩ Set.Icc 18 54 = {42}) :=
sorry

end correct_calculation_of_one_fifth_sum_of_acute_angles_l1511_151166


namespace integer_solution_fifth_power_minus_three_times_square_l1511_151139

theorem integer_solution_fifth_power_minus_three_times_square : ∃ x : ℤ, x^5 - 3*x^2 = 216 ∧ x = 3 := by
  sorry

end integer_solution_fifth_power_minus_three_times_square_l1511_151139


namespace range_of_x1_l1511_151179

/-- A function f: ℝ → ℝ is increasing -/
def Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The main theorem -/
theorem range_of_x1 (f : ℝ → ℝ) (h_inc : Increasing f)
  (h_ineq : ∀ x₁ x₂ : ℝ, x₁ + x₂ = 1 → f x₁ + f 0 > f x₂ + f 1) :
  ∀ x₁ : ℝ, (∃ x₂ : ℝ, x₁ + x₂ = 1 ∧ f x₁ + f 0 > f x₂ + f 1) → x₁ > 1 :=
by sorry

end range_of_x1_l1511_151179


namespace max_gcd_sum_l1511_151114

theorem max_gcd_sum (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a < b) (h3 : b < c) (h4 : c ≤ 3000) :
  (∃ (x y z : ℕ), 1 ≤ x ∧ x < y ∧ y < z ∧ z ≤ 3000 ∧
    Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 3000) ∧
  (∀ (x y z : ℕ), 1 ≤ x → x < y → y < z → z ≤ 3000 →
    Nat.gcd x y + Nat.gcd y z + Nat.gcd z x ≤ 3000) :=
by
  sorry

end max_gcd_sum_l1511_151114


namespace thabo_total_books_l1511_151183

/-- The number of books Thabo owns of each type and in total. -/
structure ThabosBooks where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ
  total : ℕ

/-- The conditions of Thabo's book collection. -/
def thabo_book_conditions (books : ThabosBooks) : Prop :=
  books.hardcover_nonfiction = 30 ∧
  books.paperback_nonfiction = books.hardcover_nonfiction + 20 ∧
  books.paperback_fiction = 2 * books.paperback_nonfiction ∧
  books.total = books.hardcover_nonfiction + books.paperback_nonfiction + books.paperback_fiction

/-- Theorem stating that given the conditions, Thabo owns 180 books in total. -/
theorem thabo_total_books :
  ∀ books : ThabosBooks, thabo_book_conditions books → books.total = 180 := by
  sorry


end thabo_total_books_l1511_151183


namespace smallest_odd_digit_multiple_of_11_proof_l1511_151193

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

def smallest_odd_digit_multiple_of_11 : ℕ := 11341

theorem smallest_odd_digit_multiple_of_11_proof :
  (smallest_odd_digit_multiple_of_11 > 10000) ∧
  (has_only_odd_digits smallest_odd_digit_multiple_of_11) ∧
  (smallest_odd_digit_multiple_of_11 % 11 = 0) ∧
  (∀ n : ℕ, n > 10000 → has_only_odd_digits n → n % 11 = 0 → n ≥ smallest_odd_digit_multiple_of_11) :=
by sorry

end smallest_odd_digit_multiple_of_11_proof_l1511_151193


namespace apple_harvest_per_section_l1511_151124

theorem apple_harvest_per_section 
  (total_sections : ℕ) 
  (total_sacks : ℕ) 
  (h1 : total_sections = 8) 
  (h2 : total_sacks = 360) : 
  total_sacks / total_sections = 45 := by
  sorry

end apple_harvest_per_section_l1511_151124


namespace hikers_distribution_theorem_l1511_151150

/-- The number of ways to distribute 5 people into three rooms -/
def distribution_ways : ℕ := 20

/-- The number of hikers -/
def num_hikers : ℕ := 5

/-- The number of available rooms -/
def num_rooms : ℕ := 3

/-- The capacity of the largest room -/
def large_room_capacity : ℕ := 3

/-- The capacity of each of the smaller rooms -/
def small_room_capacity : ℕ := 2

/-- Theorem stating that the number of ways to distribute the hikers is correct -/
theorem hikers_distribution_theorem :
  distribution_ways = (num_hikers.choose large_room_capacity) * 2 :=
by sorry

end hikers_distribution_theorem_l1511_151150


namespace grid_line_count_l1511_151162

/-- Represents a point in the grid -/
structure Point where
  x : Fin 50
  y : Fin 50

/-- Represents the color of a point -/
inductive Color
  | Blue
  | Red

/-- Represents the color of a line segment -/
inductive LineColor
  | Blue
  | Red
  | Black

/-- The coloring of the grid -/
def grid_coloring : Point → Color := sorry

/-- The number of blue points in the grid -/
def num_blue_points : Nat := 1510

/-- The number of blue points on the edge of the grid -/
def num_blue_edge_points : Nat := 110

/-- The number of red line segments in the grid -/
def num_red_lines : Nat := 947

/-- Checks if a point is on the edge of the grid -/
def is_edge_point (p : Point) : Bool := 
  p.x = 0 || p.x = 49 || p.y = 0 || p.y = 49

/-- Checks if a point is at a corner of the grid -/
def is_corner_point (p : Point) : Bool :=
  (p.x = 0 && p.y = 0) || (p.x = 0 && p.y = 49) || 
  (p.x = 49 && p.y = 0) || (p.x = 49 && p.y = 49)

/-- The main theorem to prove -/
theorem grid_line_count : 
  (∀ p : Point, is_corner_point p → grid_coloring p = Color.Red) →
  (∃ edge_blue_points : Finset Point, 
    edge_blue_points.card = num_blue_edge_points ∧
    ∀ p ∈ edge_blue_points, is_edge_point p ∧ grid_coloring p = Color.Blue) →
  (∃ black_lines blue_lines : Nat, 
    black_lines = 1972 ∧ 
    blue_lines = 1981 ∧
    black_lines + blue_lines + num_red_lines = 50 * 49 * 2) :=
by sorry

end grid_line_count_l1511_151162


namespace complement_of_M_l1511_151187

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Finset Nat := {1, 2, 4}

-- Theorem statement
theorem complement_of_M : U \ M = {3, 5, 6} := by
  sorry

end complement_of_M_l1511_151187


namespace tethered_unicorn_sum_l1511_151106

/-- Represents the configuration of a unicorn tethered to a cylindrical tower. -/
structure TetheredUnicorn where
  towerRadius : ℝ
  ropeLength : ℝ
  unicornHeight : ℝ
  distanceFromTower : ℝ
  ropeTouchLength : ℝ
  a : ℕ
  b : ℕ
  c : ℕ

/-- The theorem stating the sum of a, b, and c for the given configuration. -/
theorem tethered_unicorn_sum (u : TetheredUnicorn)
  (h1 : u.towerRadius = 10)
  (h2 : u.ropeLength = 30)
  (h3 : u.unicornHeight = 6)
  (h4 : u.distanceFromTower = 6)
  (h5 : u.ropeTouchLength = (u.a - Real.sqrt u.b) / u.c)
  (h6 : Nat.Prime u.c) :
  u.a + u.b + u.c = 940 := by
  sorry

end tethered_unicorn_sum_l1511_151106


namespace missing_number_proof_l1511_151171

theorem missing_number_proof (known_numbers : List ℕ) (mean : ℚ) : 
  known_numbers = [1, 23, 24, 25, 26, 27, 2] ∧ 
  mean = 20 ∧ 
  (List.sum known_numbers + 32) / 8 = mean →
  32 = 8 * mean - List.sum known_numbers :=
by
  sorry

end missing_number_proof_l1511_151171


namespace condition_type_1_condition_type_2_condition_type_3_l1511_151153

-- Statement 1
theorem condition_type_1 :
  (∀ x : ℝ, 0 < x ∧ x < 3 → |x - 1| < 2) ∧
  ¬(∀ x : ℝ, |x - 1| < 2 → 0 < x ∧ x < 3) := by sorry

-- Statement 2
theorem condition_type_2 :
  (∀ x : ℝ, x = 2 → (x - 2) * (x - 3) = 0) ∧
  ¬(∀ x : ℝ, (x - 2) * (x - 3) = 0 → x = 2) := by sorry

-- Statement 3
theorem condition_type_3 :
  ∀ (a b c : ℝ), c = 0 ↔ a * 0^2 + b * 0 + c = 0 := by sorry

end condition_type_1_condition_type_2_condition_type_3_l1511_151153


namespace largest_special_square_proof_l1511_151144

/-- A number is a perfect square if it's the square of an integer -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- Remove the last two digits of a natural number -/
def remove_last_two_digits (n : ℕ) : ℕ := n / 100

/-- The largest perfect square satisfying the given conditions -/
def largest_special_square : ℕ := 1681

theorem largest_special_square_proof :
  (is_perfect_square largest_special_square) ∧ 
  (is_perfect_square (remove_last_two_digits largest_special_square)) ∧ 
  (largest_special_square % 100 ≠ 0) ∧
  (∀ n : ℕ, n > largest_special_square → 
    ¬(is_perfect_square n ∧ 
      is_perfect_square (remove_last_two_digits n) ∧ 
      n % 100 ≠ 0)) := by
  sorry

end largest_special_square_proof_l1511_151144


namespace least_number_to_add_l1511_151157

theorem least_number_to_add (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((28523 + y) % 3 = 0 ∧ (28523 + y) % 5 = 0 ∧ (28523 + y) % 7 = 0 ∧ (28523 + y) % 8 = 0)) ∧
  ((28523 + x) % 3 = 0 ∧ (28523 + x) % 5 = 0 ∧ (28523 + x) % 7 = 0 ∧ (28523 + x) % 8 = 0) →
  x = 137 := by
sorry

end least_number_to_add_l1511_151157


namespace beef_weight_loss_percentage_l1511_151158

/-- Calculates the percentage of weight lost during processing of a side of beef. -/
theorem beef_weight_loss_percentage 
  (initial_weight : ℝ) 
  (processed_weight : ℝ) 
  (h1 : initial_weight = 892.31)
  (h2 : processed_weight = 580) : 
  ∃ (percentage : ℝ), abs (percentage - 34.99) < 0.01 ∧ 
  percentage = (initial_weight - processed_weight) / initial_weight * 100 :=
sorry

end beef_weight_loss_percentage_l1511_151158


namespace planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_l1511_151127

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: Two planes perpendicular to the same line are parallel
theorem planes_perpendicular_to_line_are_parallel 
  (l : Line) (p1 p2 : Plane) 
  (h1 : perpendicular_plane_line p1 l) 
  (h2 : perpendicular_plane_line p2 l) : 
  parallel_planes p1 p2 :=
sorry

-- Theorem 2: Two lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_plane_are_parallel 
  (p : Plane) (l1 l2 : Line) 
  (h1 : perpendicular_line_plane l1 p) 
  (h2 : perpendicular_line_plane l2 p) : 
  parallel_lines l1 l2 :=
sorry

end planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_l1511_151127


namespace assignment_calculations_l1511_151128

/-- Represents the number of volunteers -/
def num_volunteers : ℕ := 5

/-- Represents the number of communities -/
def num_communities : ℕ := 4

/-- Total number of assignment schemes -/
def total_assignments : ℕ := num_communities ^ num_volunteers

/-- Number of assignments with restrictions on community A and minimum volunteers -/
def restricted_assignments : ℕ := 150

/-- Number of assignments with each community having at least one volunteer and two specific volunteers not in the same community -/
def specific_restricted_assignments : ℕ := 216

/-- Theorem stating the correctness of the assignment calculations -/
theorem assignment_calculations :
  (total_assignments = 1024) ∧
  (restricted_assignments = 150) ∧
  (specific_restricted_assignments = 216) := by sorry

end assignment_calculations_l1511_151128


namespace sine_cosine_relation_l1511_151122

theorem sine_cosine_relation (θ : Real) (x : Real) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2) 
  (h2 : x > 1) 
  (h3 : Real.cos (θ / 2) = Real.sqrt ((x + 1) / (2 * x))) : 
  Real.sin θ = Real.sqrt (x^2 - 1) / x := by
sorry

end sine_cosine_relation_l1511_151122


namespace part_one_part_two_l1511_151111

-- Part 1
theorem part_one (a b : ℝ) (h1 : 1 ≤ a - b) (h2 : a - b ≤ 2) (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) :
  5 ≤ 4*a - 2*b ∧ 4*a - 2*b ≤ 10 := by sorry

-- Part 2
theorem part_two (m : ℝ) (h : ∀ x > (1/2 : ℝ), 2*x^2 - x ≥ 2*m*x - m - 8) :
  m ≤ 9/2 := by sorry

end part_one_part_two_l1511_151111


namespace prob_event_a_is_one_third_l1511_151182

/-- Represents a glove --/
inductive Glove
| Left : Glove
| Right : Glove

/-- Represents a color --/
inductive Color
| Red : Color
| Blue : Color
| Yellow : Color

/-- Represents a pair of gloves --/
def GlovePair := Color × Glove × Glove

/-- The set of all possible glove pairs --/
def allGlovePairs : Finset GlovePair :=
  sorry

/-- The event of selecting two gloves --/
def twoGloveSelection := GlovePair × GlovePair

/-- The event of selecting one left and one right glove of different colors --/
def eventA : Set twoGloveSelection :=
  sorry

/-- The probability of event A --/
def probEventA : ℚ :=
  sorry

theorem prob_event_a_is_one_third :
  probEventA = 1 / 3 :=
sorry

end prob_event_a_is_one_third_l1511_151182


namespace parabola_point_distance_l1511_151163

/-- Given a point P(a,0) and a parabola y^2 = 4x, if for every point Q on the parabola |PQ| ≥ |a|, then a ≤ 2 -/
theorem parabola_point_distance (a : ℝ) : 
  (∀ x y : ℝ, y^2 = 4*x → ((x - a)^2 + y^2 ≥ a^2)) → 
  a ≤ 2 := by
sorry

end parabola_point_distance_l1511_151163


namespace lcm_36_98_l1511_151173

theorem lcm_36_98 : Nat.lcm 36 98 = 1764 := by
  sorry

end lcm_36_98_l1511_151173


namespace binomial_7_choose_4_l1511_151103

theorem binomial_7_choose_4 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_7_choose_4_l1511_151103


namespace gcf_of_36_and_12_l1511_151100

theorem gcf_of_36_and_12 :
  let n : ℕ := 36
  let m : ℕ := 12
  let lcm_nm : ℕ := 54
  lcm n m = lcm_nm →
  Nat.gcd n m = 8 := by
sorry

end gcf_of_36_and_12_l1511_151100


namespace english_score_is_98_l1511_151141

/-- Given the Mathematics score, Korean language score, and average score,
    calculate the English score. -/
def calculate_english_score (math_score : ℕ) (korean_offset : ℕ) (average_score : ℚ) : ℚ :=
  3 * average_score - (math_score : ℚ) - ((math_score : ℚ) + korean_offset)

/-- Theorem stating that under the given conditions, the English score is 98. -/
theorem english_score_is_98 :
  let math_score : ℕ := 82
  let korean_offset : ℕ := 5
  let average_score : ℚ := 89
  calculate_english_score math_score korean_offset average_score = 98 := by
  sorry

#eval calculate_english_score 82 5 89

end english_score_is_98_l1511_151141


namespace trigonometric_simplification_l1511_151137

theorem trigonometric_simplification :
  let numerator := Real.sin (15 * π / 180) + Real.sin (25 * π / 180) + Real.sin (35 * π / 180) + 
                   Real.sin (45 * π / 180) + Real.sin (55 * π / 180) + Real.sin (65 * π / 180) + 
                   Real.sin (75 * π / 180) + Real.sin (85 * π / 180)
  let denominator := Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)
  numerator / denominator = (16 * Real.sin (50 * π / 180) * Real.cos (20 * π / 180)) / Real.sqrt 3 :=
by sorry

end trigonometric_simplification_l1511_151137


namespace product_xyz_equals_negative_one_l1511_151184

theorem product_xyz_equals_negative_one 
  (x y z : ℝ) 
  (h1 : x + 1 / y = 2) 
  (h2 : y + 1 / z = 2) : 
  x * y * z = -1 := by
sorry

end product_xyz_equals_negative_one_l1511_151184


namespace m_range_l1511_151133

-- Define the propositions
def P (t : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (t + 2) + y^2 / (t - 10) = 1

def Q (t m : ℝ) : Prop := 1 - m < t ∧ t < 1 + m ∧ m > 0

-- Define the relationship between P and Q
def relationship (m : ℝ) : Prop :=
  ∀ t, ¬(P t) → ¬(Q t m) ∧ ∃ t, ¬(Q t m) ∧ P t

-- State the theorem
theorem m_range :
  ∀ m, (∀ t, P t ↔ t ∈ Set.Ioo (-2) 10) →
       relationship m →
       m ∈ Set.Ioc 0 3 :=
sorry

end m_range_l1511_151133


namespace a_less_than_two_l1511_151145

def A (a : ℝ) : Set ℝ := {x | x ≤ a}
def B : Set ℝ := Set.Iio 2

theorem a_less_than_two (a : ℝ) (h : A a ⊆ B) : a < 2 := by
  sorry

end a_less_than_two_l1511_151145


namespace count_symmetric_patterns_l1511_151121

/-- A symmetric digital pattern on an 8x8 grid --/
structure SymmetricPattern :=
  (grid : Fin 8 → Fin 8 → Bool)
  (symmetric : ∀ (i j : Fin 8), grid i j = grid (7 - i) j ∧ grid i j = grid i (7 - j) ∧ grid i j = grid j i)
  (not_monochrome : ∃ (i j k l : Fin 8), grid i j ≠ grid k l)

/-- The number of symmetric regions in an 8x8 grid --/
def num_symmetric_regions : Nat := 12

/-- The total number of possible symmetric digital patterns --/
def total_symmetric_patterns : Nat := 2^num_symmetric_regions - 2

theorem count_symmetric_patterns :
  total_symmetric_patterns = 4094 :=
sorry

end count_symmetric_patterns_l1511_151121


namespace geometric_sequence_property_l1511_151129

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem stating the property of the geometric sequence -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_prod : a 1 * a 7 * a 13 = 8) : 
    a 3 * a 11 = 4 := by
  sorry


end geometric_sequence_property_l1511_151129


namespace parallelogram_area_l1511_151102

/-- The area of a parallelogram with base 20 cm and height 16 cm is 320 cm². -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 20 → height = 16 → area = base * height → area = 320 :=
by sorry

end parallelogram_area_l1511_151102


namespace mary_earnings_per_home_l1511_151164

/-- Given that Mary earned $12696 cleaning 276.0 homes, prove that she earns $46 per home. -/
theorem mary_earnings_per_home :
  let total_earnings : ℚ := 12696
  let homes_cleaned : ℚ := 276
  total_earnings / homes_cleaned = 46 := by
  sorry

end mary_earnings_per_home_l1511_151164


namespace quadratic_factorization_l1511_151178

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := by
  sorry

end quadratic_factorization_l1511_151178


namespace total_rainbow_nerds_l1511_151110

/-- The number of rainbow nerds in a box with purple, yellow, and green candies. -/
def rainbow_nerds (purple : ℕ) (yellow : ℕ) (green : ℕ) : ℕ := purple + yellow + green

/-- Theorem: The total number of rainbow nerds in the box is 36. -/
theorem total_rainbow_nerds :
  ∃ (purple yellow green : ℕ),
    purple = 10 ∧
    yellow = purple + 4 ∧
    green = yellow - 2 ∧
    rainbow_nerds purple yellow green = 36 := by
  sorry

end total_rainbow_nerds_l1511_151110


namespace solution_set_when_a_is_two_range_of_a_for_all_real_solution_l1511_151169

def f (a x : ℝ) : ℝ := a * x^2 + a * x - 1

theorem solution_set_when_a_is_two :
  let a := 2
  {x : ℝ | f a x < 0} = {x : ℝ | -(1 + Real.sqrt 3) / 2 < x ∧ x < (-1 + Real.sqrt 3) / 2} := by sorry

theorem range_of_a_for_all_real_solution :
  {a : ℝ | ∀ x, f a x < 0} = {a : ℝ | -4 < a ∧ a ≤ 0} := by sorry

end solution_set_when_a_is_two_range_of_a_for_all_real_solution_l1511_151169


namespace science_study_time_l1511_151140

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The total time Sam spends studying in hours -/
def total_study_time_hours : ℕ := 3

/-- The time Sam spends studying Math in minutes -/
def math_study_time : ℕ := 80

/-- The time Sam spends studying Literature in minutes -/
def literature_study_time : ℕ := 40

/-- Theorem: Sam spends 60 minutes studying Science -/
theorem science_study_time : ℕ := by
  sorry

end science_study_time_l1511_151140


namespace olivia_soda_purchase_l1511_151151

/-- The number of quarters Olivia spent on a soda -/
def quarters_spent (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that Olivia spent 4 quarters on the soda -/
theorem olivia_soda_purchase : quarters_spent 11 7 = 4 := by
  sorry

end olivia_soda_purchase_l1511_151151


namespace max_integer_with_divisor_difference_twenty_four_satisfies_condition_l1511_151186

theorem max_integer_with_divisor_difference (n : ℕ) : 
  (∀ k : ℕ, k > 0 → k ≤ n / 2 → ∃ d₁ d₂ : ℕ, d₁ > 0 ∧ d₂ > 0 ∧ d₁ ∣ n ∧ d₂ ∣ n ∧ d₂ - d₁ = k) →
  n ≤ 24 :=
by sorry

theorem twenty_four_satisfies_condition : 
  ∀ k : ℕ, k > 0 → k ≤ 24 / 2 → ∃ d₁ d₂ : ℕ, d₁ > 0 ∧ d₂ > 0 ∧ d₁ ∣ 24 ∧ d₂ ∣ 24 ∧ d₂ - d₁ = k :=
by sorry

end max_integer_with_divisor_difference_twenty_four_satisfies_condition_l1511_151186


namespace semi_circle_radius_equals_rectangle_area_l1511_151191

theorem semi_circle_radius_equals_rectangle_area (length width : ℝ) (h1 : length = 8) (h2 : width = Real.pi) :
  ∃ (r : ℝ), r = 4 ∧ (1/2 * Real.pi * r^2) = (length * width) :=
by sorry

end semi_circle_radius_equals_rectangle_area_l1511_151191


namespace collatz_3_reaches_421_cycle_l1511_151130

-- Define the operation for a single step
def collatzStep (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

-- Define the sequence of Collatz numbers starting from n
def collatzSequence (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => collatzStep (collatzSequence n k)

-- Theorem stating that the Collatz sequence starting from 3 eventually reaches the cycle 4, 2, 1
theorem collatz_3_reaches_421_cycle :
  ∃ k : ℕ, ∃ m : ℕ, m ≥ k ∧
    (collatzSequence 3 m = 4 ∧
     collatzSequence 3 (m + 1) = 2 ∧
     collatzSequence 3 (m + 2) = 1 ∧
     collatzSequence 3 (m + 3) = 4) :=
sorry

end collatz_3_reaches_421_cycle_l1511_151130


namespace triangle_problem_l1511_151159

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  c = 4 * Real.sqrt 2 →
  B = π / 4 →
  (1/2) * a * c * Real.sin B = 2 →
  a = 1 ∧ b = 5 :=
sorry

end triangle_problem_l1511_151159


namespace tangent_line_equation_l1511_151147

/-- A point on a cubic curve with a specific tangent slope -/
structure TangentPoint where
  x : ℝ
  y : ℝ
  on_curve : y = x^3 - 10*x + 3
  in_second_quadrant : x < 0 ∧ y > 0
  tangent_slope : 3*x^2 - 10 = 2

/-- The equation of the tangent line -/
def tangent_line (p : TangentPoint) : ℝ → ℝ := λ x => 2*x + 19

theorem tangent_line_equation (p : TangentPoint) :
  tangent_line p p.x = p.y ∧
  (λ x => tangent_line p x - p.y) = (λ x => 2*(x - p.x)) :=
by sorry

end tangent_line_equation_l1511_151147


namespace number_order_l1511_151132

theorem number_order : 
  let a : ℝ := 30.5
  let b : ℝ := 0.53
  let c : ℝ := Real.log 0.53
  c < b ∧ b < a := by sorry

end number_order_l1511_151132


namespace carl_responsibility_l1511_151125

/-- Calculates the amount a person owes in an accident based on their fault percentage and insurance coverage -/
def calculate_personal_responsibility (total_property_damage : ℝ) (total_medical_bills : ℝ)
  (property_insurance_coverage : ℝ) (medical_insurance_coverage : ℝ) (fault_percentage : ℝ) : ℝ :=
  let remaining_property_damage := total_property_damage * (1 - property_insurance_coverage)
  let remaining_medical_bills := total_medical_bills * (1 - medical_insurance_coverage)
  fault_percentage * (remaining_property_damage + remaining_medical_bills)

/-- Theorem stating Carl's personal responsibility in the accident -/
theorem carl_responsibility :
  let total_property_damage : ℝ := 40000
  let total_medical_bills : ℝ := 70000
  let property_insurance_coverage : ℝ := 0.8
  let medical_insurance_coverage : ℝ := 0.75
  let carl_fault_percentage : ℝ := 0.6
  calculate_personal_responsibility total_property_damage total_medical_bills
    property_insurance_coverage medical_insurance_coverage carl_fault_percentage = 15300 := by
  sorry

end carl_responsibility_l1511_151125


namespace M_intersect_N_l1511_151190

def M : Set ℕ := {1, 2, 4, 8}

def N : Set ℕ := {x : ℕ | ∃ k : ℕ, x = 2 * k}

theorem M_intersect_N : M ∩ N = {2, 4, 8} := by sorry

end M_intersect_N_l1511_151190


namespace pond_length_l1511_151107

/-- The length of a rectangular pond given its width, depth, and volume. -/
theorem pond_length (width : ℝ) (depth : ℝ) (volume : ℝ) 
  (h_width : width = 12) 
  (h_depth : depth = 5) 
  (h_volume : volume = 1200) : 
  volume / (width * depth) = 20 := by
  sorry

end pond_length_l1511_151107


namespace jason_earnings_l1511_151167

/-- Represents the earnings of a person given their initial and final amounts -/
def earnings (initial final : ℕ) : ℕ := final - initial

theorem jason_earnings :
  let fred_initial : ℕ := 49
  let jason_initial : ℕ := 3
  let fred_final : ℕ := 112
  let jason_final : ℕ := 63
  earnings jason_initial jason_final = 60 := by
sorry

end jason_earnings_l1511_151167


namespace new_cube_edge_length_l1511_151101

-- Define the edge lengths of the original cubes
def edge1 : ℝ := 6
def edge2 : ℝ := 8
def edge3 : ℝ := 10

-- Define the volume of a cube given its edge length
def cubeVolume (edge : ℝ) : ℝ := edge ^ 3

-- Define the total volume of the three original cubes
def totalVolume : ℝ := cubeVolume edge1 + cubeVolume edge2 + cubeVolume edge3

-- Define the edge length of the new cube
def newEdge : ℝ := totalVolume ^ (1/3)

-- Theorem statement
theorem new_cube_edge_length : newEdge = 12 := by
  sorry

end new_cube_edge_length_l1511_151101


namespace gear_speed_ratio_l1511_151156

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four gears meshed in sequence -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear
  meshed_AB : A.teeth * A.speed = B.teeth * B.speed
  meshed_BC : B.teeth * B.speed = C.teeth * C.speed
  meshed_CD : C.teeth * C.speed = D.teeth * D.speed

/-- The theorem stating the ratio of angular speeds for the given gear system -/
theorem gear_speed_ratio (sys : GearSystem) 
  (hA : sys.A.teeth = 10)
  (hB : sys.B.teeth = 15)
  (hC : sys.C.teeth = 20)
  (hD : sys.D.teeth = 25) :
  ∃ (k : ℝ), k > 0 ∧ 
    sys.A.speed = 24 * k ∧
    sys.B.speed = 25 * k ∧
    sys.C.speed = 12 * k ∧
    sys.D.speed = 20 * k := by
  sorry

end gear_speed_ratio_l1511_151156


namespace cube_rotation_theorem_l1511_151126

/-- Represents a cube with natural numbers on each face -/
structure Cube where
  front : ℕ
  right : ℕ
  back : ℕ
  left : ℕ
  top : ℕ
  bottom : ℕ

/-- Theorem stating the properties of the cube and the results to be proved -/
theorem cube_rotation_theorem (c : Cube) 
  (h1 : c.front + c.right + c.top = 42)
  (h2 : c.right + c.top + c.back = 34)
  (h3 : c.top + c.back + c.left = 53)
  (h4 : c.bottom = 6) :
  (c.left + c.front + c.top = 61) ∧ 
  (c.front + c.right + c.back + c.left + c.top + c.bottom ≤ 100) := by
  sorry


end cube_rotation_theorem_l1511_151126


namespace car_catching_truck_l1511_151113

/-- A problem about a car catching up to a truck on a highway. -/
theorem car_catching_truck (truck_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  truck_speed = 45 →
  head_start = 1 →
  catch_up_time = 4 →
  let car_speed := (truck_speed * (catch_up_time + head_start)) / catch_up_time
  car_speed = 56.25 := by
sorry


end car_catching_truck_l1511_151113


namespace factorial_difference_l1511_151135

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end factorial_difference_l1511_151135


namespace linear_function_segment_l1511_151177

-- Define the linear function
def f (x : ℝ) := -2 * x + 3

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem linear_function_segment :
  ∃ (A B : ℝ × ℝ), 
    (∀ x ∈ domain, ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
      (x, f x) = (1 - t) • A + t • B) ∧
    (∀ t : ℝ, 0 ≤ t → t ≤ 1 → 
      ((1 - t) • A.1 + t • B.1) ∈ domain) :=
by sorry


end linear_function_segment_l1511_151177


namespace evaluate_expression_l1511_151154

theorem evaluate_expression : 150 * (150 - 4) - 2 * (150 * 150 - 4) = -23092 := by
  sorry

end evaluate_expression_l1511_151154


namespace midpoint_locus_l1511_151176

/-- The locus of midpoints between a fixed point and points on a circle -/
theorem midpoint_locus (A B P : ℝ × ℝ) (m n x y : ℝ) :
  A = (4, -2) →
  B = (m, n) →
  m^2 + n^2 = 4 →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  (x - 2)^2 + (y + 1)^2 = 1 :=
by sorry

end midpoint_locus_l1511_151176


namespace f_cos_x_equals_two_plus_cos_two_x_l1511_151108

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_cos_x_equals_two_plus_cos_two_x (x : ℝ) : 
  (∀ y : ℝ, f (Real.sin y) = 2 - Real.cos (2 * y)) → 
  f (Real.cos x) = 2 + Real.cos (2 * x) := by
  sorry

end f_cos_x_equals_two_plus_cos_two_x_l1511_151108


namespace parabola_y_axis_intersection_l1511_151168

/-- Given a parabola y = -x^2 + (k+1)x - k where (4,0) lies on the parabola,
    prove that the intersection point of the parabola with the y-axis is (0, -4). -/
theorem parabola_y_axis_intersection
  (k : ℝ)
  (h : 0 = -(4^2) + (k+1)*4 - k) :
  ∃ y, y = -4 ∧ 0 = -(0^2) + (k+1)*0 - k ∧ y = -(0^2) + (k+1)*0 - k :=
by sorry

end parabola_y_axis_intersection_l1511_151168


namespace rice_pricing_problem_l1511_151194

/-- Represents the linear relationship between price and quantity sold --/
def quantity_sold (x : ℝ) : ℝ := -50 * x + 1200

/-- Represents the profit function --/
def profit (x : ℝ) : ℝ := (x - 4) * (quantity_sold x)

/-- Theorem stating the main results of the problem --/
theorem rice_pricing_problem 
  (x : ℝ) 
  (h1 : 4 ≤ x ∧ x ≤ 7) :
  (∃ x, profit x = 1800 ∧ x = 6) ∧
  (∀ y, 4 ≤ y ∧ y ≤ 7 → profit y ≤ profit 7) ∧
  profit 7 = 2550 := by
  sorry


end rice_pricing_problem_l1511_151194


namespace diophantine_equation_solutions_l1511_151148

theorem diophantine_equation_solutions :
  ∀ m n : ℤ, m ≠ 0 ∧ n ≠ 0 →
  (m^2 + n) * (m + n^2) = (m - n)^3 ↔
  (m = -1 ∧ n = -1) ∨ (m = 8 ∧ n = -10) ∨ (m = 9 ∧ n = -6) ∨ (m = 9 ∧ n = -21) :=
by sorry

end diophantine_equation_solutions_l1511_151148


namespace smallest_n_fourth_root_l1511_151172

theorem smallest_n_fourth_root (n : ℕ) : n = 4097 ↔ 
  (n > 0 ∧ 
   ∀ m : ℕ, m > 0 → m < n → 
   ¬(0 < (m : ℝ)^(1/4) - ⌊(m : ℝ)^(1/4)⌋ ∧ (m : ℝ)^(1/4) - ⌊(m : ℝ)^(1/4)⌋ < 1/2015)) ∧
  (0 < (n : ℝ)^(1/4) - ⌊(n : ℝ)^(1/4)⌋ ∧ (n : ℝ)^(1/4) - ⌊(n : ℝ)^(1/4)⌋ < 1/2015) :=
by sorry

end smallest_n_fourth_root_l1511_151172


namespace power_fraction_equality_l1511_151170

theorem power_fraction_equality : (2^2015 + 2^2013 + 2^2011) / (2^2015 - 2^2013 + 2^2011) = 21/13 := by
  sorry

end power_fraction_equality_l1511_151170


namespace vector_operation_result_l1511_151117

theorem vector_operation_result :
  let v1 : Fin 3 → ℝ := ![(-3), 4, 2]
  let v2 : Fin 3 → ℝ := ![1, 6, (-3)]
  2 • v1 + v2 = ![-5, 14, 1] := by sorry

end vector_operation_result_l1511_151117


namespace sum_of_squares_of_roots_l1511_151149

theorem sum_of_squares_of_roots (a b c : ℂ) : 
  (3 * a^3 - 2 * a^2 + 5 * a + 15 = 0) ∧ 
  (3 * b^3 - 2 * b^2 + 5 * b + 15 = 0) ∧ 
  (3 * c^3 - 2 * c^2 + 5 * c + 15 = 0) →
  a^2 + b^2 + c^2 = -26/9 := by sorry

end sum_of_squares_of_roots_l1511_151149


namespace triangle_properties_l1511_151142

-- Define the lines of triangle ABC
def line_AB (x y : ℝ) : Prop := 3 * x + 4 * y + 12 = 0
def line_BC (x y : ℝ) : Prop := 4 * x - 3 * y + 16 = 0
def line_CA (x y : ℝ) : Prop := 2 * x + y - 2 = 0

-- Define point B as the intersection of AB and BC
def point_B : ℝ × ℝ := (-4, 0)

-- Define the equation of the altitude from A to BC
def altitude_A_to_BC (x y : ℝ) : Prop := x - 2 * y + 4 = 0

theorem triangle_properties :
  (∀ x y : ℝ, line_AB x y ∧ line_BC x y → (x, y) = point_B) ∧
  (∀ x y : ℝ, altitude_A_to_BC x y ↔ 
    (∃ t : ℝ, x = t * (point_B.1 - (2 / 5)) ∧ 
              y = t * (point_B.2 + (1 / 5)) ∧
              2 * x + y - 2 = 0)) :=
sorry

end triangle_properties_l1511_151142


namespace smallest_aab_value_exists_valid_digit_pair_l1511_151120

/-- Represents a pair of distinct digits from 1 to 9 -/
structure DigitPair where
  a : Nat
  b : Nat
  a_in_range : a ≥ 1 ∧ a ≤ 9
  b_in_range : b ≥ 1 ∧ b ≤ 9
  distinct : a ≠ b

/-- Converts a DigitPair to a two-digit number -/
def to_two_digit (p : DigitPair) : Nat :=
  10 * p.a + p.b

/-- Converts a DigitPair to a three-digit number AAB -/
def to_three_digit (p : DigitPair) : Nat :=
  100 * p.a + 10 * p.a + p.b

/-- The main theorem stating the smallest possible value of AAB -/
theorem smallest_aab_value (p : DigitPair) 
  (h : to_two_digit p = (to_three_digit p) / 8) : 
  to_three_digit p ≥ 773 := by
  sorry

/-- The existence of a DigitPair satisfying the conditions -/
theorem exists_valid_digit_pair : 
  ∃ p : DigitPair, to_two_digit p = (to_three_digit p) / 8 ∧ to_three_digit p = 773 := by
  sorry

end smallest_aab_value_exists_valid_digit_pair_l1511_151120


namespace min_value_theorem_l1511_151134

theorem min_value_theorem (a b : ℝ) (h1 : b > 0) (h2 : a + b = 2) :
  (1 / (2 * |a|)) + (|a| / b) ≥ 3/4 ∧ 
  ∃ (a₀ b₀ : ℝ), b₀ > 0 ∧ a₀ + b₀ = 2 ∧ (1 / (2 * |a₀|)) + (|a₀| / b₀) = 3/4 :=
sorry

end min_value_theorem_l1511_151134

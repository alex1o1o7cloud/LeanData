import Mathlib

namespace doughnut_machine_completion_time_l376_37677

-- Define the start time and one-third completion time
def start_time : ℕ := 7 * 60  -- 7:00 AM in minutes
def one_third_time : ℕ := 10 * 60 + 15  -- 10:15 AM in minutes

-- Define the time taken for one-third of the job
def one_third_duration : ℕ := one_third_time - start_time

-- Define the total duration of the job
def total_duration : ℕ := 3 * one_third_duration

-- Define the completion time
def completion_time : ℕ := start_time + total_duration

-- Theorem to prove
theorem doughnut_machine_completion_time :
  completion_time = 16 * 60 + 45  -- 4:45 PM in minutes
:= by sorry

end doughnut_machine_completion_time_l376_37677


namespace problem_1_l376_37691

theorem problem_1 : (1) - 4^2 / (-32) * (2/3)^2 = 2/9 := by sorry

end problem_1_l376_37691


namespace surface_area_is_34_l376_37664

/-- A three-dimensional figure composed of unit cubes -/
structure CubeFigure where
  num_cubes : ℕ
  cube_side_length : ℝ
  top_area : ℝ
  bottom_area : ℝ
  front_area : ℝ
  back_area : ℝ
  left_area : ℝ
  right_area : ℝ

/-- The surface area of a CubeFigure -/
def surface_area (figure : CubeFigure) : ℝ :=
  figure.top_area + figure.bottom_area + figure.front_area + 
  figure.back_area + figure.left_area + figure.right_area

/-- Theorem stating that the surface area of the given figure is 34 -/
theorem surface_area_is_34 (figure : CubeFigure) 
  (h1 : figure.num_cubes = 10)
  (h2 : figure.cube_side_length = 1)
  (h3 : figure.top_area = 6)
  (h4 : figure.bottom_area = 6)
  (h5 : figure.front_area = 5)
  (h6 : figure.back_area = 5)
  (h7 : figure.left_area = 6)
  (h8 : figure.right_area = 6) :
  surface_area figure = 34 := by
  sorry

end surface_area_is_34_l376_37664


namespace difference_of_squares_l376_37651

theorem difference_of_squares (a b : ℝ) : (2*a - b) * (2*a + b) = 4*a^2 - b^2 := by
  sorry

end difference_of_squares_l376_37651


namespace average_of_last_three_l376_37607

theorem average_of_last_three (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 62 →
  ((list.take 4).sum / 4 : ℝ) = 55 →
  ((list.drop 4).sum / 3 : ℝ) = 71 + 1/3 := by
sorry

end average_of_last_three_l376_37607


namespace c_invests_after_eight_months_l376_37614

/-- Represents the investment scenario of three partners A, B, and C -/
structure Investment where
  /-- A's initial investment amount -/
  a_amount : ℝ
  /-- Number of months after which C invests -/
  c_invest_time : ℝ
  /-- Total annual gain -/
  total_gain : ℝ
  /-- A's share of the profit -/
  a_share : ℝ
  /-- B invests double A's amount after 6 months -/
  b_amount_eq : a_amount * 2 = a_amount
  /-- C invests triple A's amount -/
  c_amount_eq : a_amount * 3 = a_amount
  /-- Total annual gain is Rs. 18600 -/
  total_gain_eq : total_gain = 18600
  /-- A's share is Rs. 6200 -/
  a_share_eq : a_share = 6200
  /-- Profit share is proportional to investment and time -/
  profit_share_prop : a_share / total_gain = 
    (a_amount * 12) / (a_amount * 12 + a_amount * 2 * 6 + a_amount * 3 * (12 - c_invest_time))

/-- Theorem stating that C invests after 8 months -/
theorem c_invests_after_eight_months (i : Investment) : i.c_invest_time = 8 := by
  sorry

end c_invests_after_eight_months_l376_37614


namespace combined_earnings_l376_37665

def dwayne_earnings : ℕ := 1500
def brady_extra : ℕ := 450

theorem combined_earnings :
  dwayne_earnings + (dwayne_earnings + brady_extra) = 3450 :=
by sorry

end combined_earnings_l376_37665


namespace alphabet_letter_count_l376_37696

theorem alphabet_letter_count (total : ℕ) (both : ℕ) (straight_only : ℕ) :
  total = 40 →
  both = 8 →
  straight_only = 24 →
  total = both + straight_only + (total - (both + straight_only)) →
  (total - (both + straight_only)) = 8 :=
by sorry

end alphabet_letter_count_l376_37696


namespace fraction_sum_simplification_l376_37653

theorem fraction_sum_simplification :
  8 / 24 - 5 / 72 + 3 / 8 = 23 / 36 := by
  sorry

end fraction_sum_simplification_l376_37653


namespace largest_solution_of_equation_l376_37671

theorem largest_solution_of_equation (x : ℚ) :
  (8 * (9 * x^2 + 10 * x + 15) = x * (9 * x - 45)) →
  x ≤ -5/3 :=
by sorry

end largest_solution_of_equation_l376_37671


namespace triangle_perimeter_l376_37633

/-- An equilateral triangle with three inscribed circles -/
structure TriangleWithCircles where
  -- The side length of the equilateral triangle
  side : ℝ
  -- The radius of each inscribed circle
  radius : ℝ
  -- The offset from each vertex to the nearest point on any circle
  offset : ℝ
  -- Condition: The radius is 2
  h_radius : radius = 2
  -- Condition: The offset is 1
  h_offset : offset = 1
  -- Condition: The circles touch each other and the sides of the triangle
  h_touch : side = 2 * (radius + offset) + 2 * radius * Real.sqrt 3

/-- The perimeter of the triangle is 6√3 + 12 -/
theorem triangle_perimeter (t : TriangleWithCircles) : 
  3 * t.side = 6 * Real.sqrt 3 + 12 := by
  sorry

end triangle_perimeter_l376_37633


namespace trapezoid_ed_length_l376_37649

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  /-- Length of base AB -/
  base : ℝ
  /-- Length of top base CD -/
  top_base : ℝ
  /-- Length of non-parallel sides AD and BC -/
  side : ℝ
  /-- E is the midpoint of diagonal AC -/
  e_midpoint : Bool
  /-- AED is a right triangle -/
  aed_right : Bool
  /-- D lies on extended line segment AE -/
  d_on_ae : Bool

/-- Theorem stating the length of ED in the given trapezoid -/
theorem trapezoid_ed_length (t : Trapezoid) 
  (h1 : t.base = 8) 
  (h2 : t.top_base = 6) 
  (h3 : t.side = 5) 
  (h4 : t.e_midpoint) 
  (h5 : t.aed_right) 
  (h6 : t.d_on_ae) : 
  ∃ (ed : ℝ), ed = Real.sqrt 6.5 := by
  sorry

end trapezoid_ed_length_l376_37649


namespace expression_simplification_l376_37645

theorem expression_simplification (x : ℝ) : 
  (x^3 - 2)^2 + (x^2 + 2*x)^2 = x^6 + x^4 + 4*x^2 + 4 := by
  sorry

end expression_simplification_l376_37645


namespace z_in_second_quadrant_z_times_i_real_implies_modulus_l376_37672

-- Define the complex number z as a function of k
def z (k : ℝ) : ℂ := (k^2 - 3*k - 4 : ℝ) + (k - 1 : ℝ) * Complex.I

-- Theorem for the first part of the problem
theorem z_in_second_quadrant (k : ℝ) :
  (z k).re < 0 ∧ (z k).im > 0 ↔ 1 < k ∧ k < 4 := by sorry

-- Theorem for the second part of the problem
theorem z_times_i_real_implies_modulus (k : ℝ) :
  (z k * Complex.I).im = 0 → Complex.abs (z k) = 2 ∨ Complex.abs (z k) = 3 := by sorry

end z_in_second_quadrant_z_times_i_real_implies_modulus_l376_37672


namespace sum_of_three_numbers_plus_five_l376_37678

theorem sum_of_three_numbers_plus_five (a b c : ℤ) 
  (h1 : a + b = 31) 
  (h2 : b + c = 48) 
  (h3 : c + a = 55) : 
  a + b + c + 5 = 72 := by
sorry

end sum_of_three_numbers_plus_five_l376_37678


namespace largest_four_digit_square_base_7_l376_37617

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def N : ℕ := 48

/-- Conversion of a natural number to its base 7 representation -/
def to_base_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- Number of digits in the base 7 representation of a natural number -/
def num_digits_base_7 (n : ℕ) : ℕ :=
  (to_base_7 n).length

/-- Theorem stating that N is the largest integer whose square has exactly 4 digits in base 7 -/
theorem largest_four_digit_square_base_7 :
  (∀ m : ℕ, m > N → num_digits_base_7 (m^2) > 4) ∧
  num_digits_base_7 (N^2) = 4 ∧
  to_base_7 N = [6, 6] :=
sorry

#eval N
#eval to_base_7 N
#eval num_digits_base_7 (N^2)

end largest_four_digit_square_base_7_l376_37617


namespace inequality_solution_l376_37629

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≥ 1) ↔ (x > -8 ∧ x < -2) :=
by sorry

end inequality_solution_l376_37629


namespace some_number_value_l376_37693

theorem some_number_value (x : ℝ) : 3034 - (1002 / x) = 3029 → x = 200.4 := by
  sorry

end some_number_value_l376_37693


namespace exist_three_naturals_with_prime_sum_and_product_l376_37604

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Theorem statement
theorem exist_three_naturals_with_prime_sum_and_product :
  ∃ a b c : ℕ, isPrime (a + b + c) ∧ isPrime (a * b * c) :=
sorry

end exist_three_naturals_with_prime_sum_and_product_l376_37604


namespace max_weighing_ways_exact_89_ways_weighing_theorem_l376_37658

/-- Represents the set of weights with masses 1, 2, 4, ..., 512 grams -/
def WeightSet : Set ℕ := {n : ℕ | ∃ k : ℕ, k ≤ 9 ∧ n = 2^k}

/-- Number of ways to weigh a load P using weights up to 2^n -/
def K (n : ℕ) (P : ℤ) : ℕ := sorry

/-- Maximum number of ways to weigh any load using weights up to 2^n -/
def MaxK (n : ℕ) : ℕ := sorry

/-- Theorem stating that no load can be weighed in more than 89 ways -/
theorem max_weighing_ways :
  ∀ P : ℤ, K 9 P ≤ 89 :=
sorry

/-- Theorem stating that 171 grams can be weighed in exactly 89 ways -/
theorem exact_89_ways :
  K 9 171 = 89 :=
sorry

/-- Main theorem combining both parts of the problem -/
theorem weighing_theorem :
  (∀ P : ℤ, K 9 P ≤ 89) ∧ (K 9 171 = 89) :=
sorry

end max_weighing_ways_exact_89_ways_weighing_theorem_l376_37658


namespace louie_junior_took_seven_cookies_l376_37650

/-- Represents the number of cookies in various states --/
structure CookieJar where
  initial : Nat
  eatenByLouSenior : Nat
  remaining : Nat

/-- Calculates the number of cookies Louie Junior took --/
def cookiesTakenByLouieJunior (jar : CookieJar) : Nat :=
  jar.initial - jar.eatenByLouSenior - jar.remaining

/-- Theorem stating that Louie Junior took 7 cookies --/
theorem louie_junior_took_seven_cookies (jar : CookieJar) 
  (h1 : jar.initial = 22)
  (h2 : jar.eatenByLouSenior = 4)
  (h3 : jar.remaining = 11) :
  cookiesTakenByLouieJunior jar = 7 := by
  sorry

#eval cookiesTakenByLouieJunior { initial := 22, eatenByLouSenior := 4, remaining := 11 }

end louie_junior_took_seven_cookies_l376_37650


namespace minimize_S_l376_37688

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℝ := 2 * n^2 - 30 * n

/-- n minimizes S if S(n) is less than or equal to S(k) for all natural numbers k -/
def Minimizes (n : ℕ) : Prop :=
  ∀ k : ℕ, S n ≤ S k

theorem minimize_S :
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ Minimizes n :=
sorry

end minimize_S_l376_37688


namespace dans_initial_green_marbles_l376_37638

/-- Represents the number of marbles Dan has -/
structure DanMarbles where
  initial_green : ℕ
  violet : ℕ
  taken_green : ℕ
  remaining_green : ℕ

/-- Theorem stating that Dan's initial number of green marbles is 32 -/
theorem dans_initial_green_marbles 
  (dan : DanMarbles)
  (h1 : dan.taken_green = 23)
  (h2 : dan.remaining_green = 9)
  (h3 : dan.initial_green = dan.taken_green + dan.remaining_green) :
  dan.initial_green = 32 := by
  sorry

end dans_initial_green_marbles_l376_37638


namespace wendy_trip_miles_l376_37644

theorem wendy_trip_miles (total_miles second_day_miles first_day_miles third_day_miles : ℕ) :
  total_miles = 493 →
  first_day_miles = 125 →
  third_day_miles = 145 →
  second_day_miles = total_miles - first_day_miles - third_day_miles →
  second_day_miles = 223 := by
sorry

end wendy_trip_miles_l376_37644


namespace polynomial_nonnegative_l376_37674

theorem polynomial_nonnegative (x : ℝ) : x^4 - x^3 + 3*x^2 - 2*x + 2 ≥ 0 := by
  sorry

end polynomial_nonnegative_l376_37674


namespace combination_sum_equality_l376_37642

theorem combination_sum_equality (n k m : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m ≤ n) :
  (Nat.choose n m) + 
  (Finset.sum (Finset.range k) (fun i => Nat.choose k (i + 1) * Nat.choose n (m - (i + 1)))) + 
  (Nat.choose n (m - k)) = 
  Nat.choose (n + k) m := by sorry

end combination_sum_equality_l376_37642


namespace initial_cherries_l376_37683

theorem initial_cherries (eaten : ℕ) (left : ℕ) (h1 : eaten = 25) (h2 : left = 42) :
  eaten + left = 67 := by
  sorry

end initial_cherries_l376_37683


namespace kats_required_score_l376_37606

/-- Given Kat's first two test scores and desired average, calculate the required score on the third test --/
theorem kats_required_score (score1 score2 desired_avg : ℚ) (h1 : score1 = 95/100) (h2 : score2 = 80/100) (h3 : desired_avg = 90/100) :
  ∃ score3 : ℚ, (score1 + score2 + score3) / 3 ≥ desired_avg ∧ score3 = 95/100 :=
by sorry

end kats_required_score_l376_37606


namespace peanuts_equation_initial_peanuts_count_l376_37610

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := sorry

/-- The number of peanuts Mary adds to the box -/
def peanuts_added : ℕ := 8

/-- The total number of peanuts after Mary adds more -/
def total_peanuts : ℕ := 12

/-- Theorem stating that the initial number of peanuts plus the added peanuts equals the total peanuts -/
theorem peanuts_equation : initial_peanuts + peanuts_added = total_peanuts := by sorry

/-- Theorem proving that the initial number of peanuts is 4 -/
theorem initial_peanuts_count : initial_peanuts = 4 := by sorry

end peanuts_equation_initial_peanuts_count_l376_37610


namespace total_pencils_l376_37657

def pencils_problem (monday tuesday wednesday thursday friday : ℕ) : Prop :=
  monday = 35 ∧
  tuesday = 42 ∧
  wednesday = 3 * tuesday ∧
  thursday = wednesday / 2 ∧
  friday = 2 * monday

theorem total_pencils :
  ∀ monday tuesday wednesday thursday friday : ℕ,
    pencils_problem monday tuesday wednesday thursday friday →
    monday + tuesday + wednesday + thursday + friday = 336 := by
  sorry

end total_pencils_l376_37657


namespace girl_sums_equal_iff_n_odd_l376_37616

/-- Represents the sum of a girl's card number and the numbers of adjacent boys' cards -/
def girlSum (n : ℕ) (i : ℕ) : ℕ :=
  (n + i) + (i % n + 1) + ((i + 1) % n + 1)

/-- Theorem stating that all girl sums are equal if and only if n is odd -/
theorem girl_sums_equal_iff_n_odd (n : ℕ) (h : n ≥ 3) :
  (∀ i j, i < n → j < n → girlSum n i = girlSum n j) ↔ Odd n :=
sorry

end girl_sums_equal_iff_n_odd_l376_37616


namespace stating_final_number_lower_bound_l376_37654

/-- 
Given a sequence of n ones, we repeatedly replace two numbers a and b 
with (a+b)/4 for n-1 steps. This function represents the final number 
after these operations.
-/
noncomputable def final_number (n : ℕ) : ℝ :=
  sorry

/-- 
Theorem stating that the final number after n-1 steps of the described 
operation, starting with n ones, is greater than or equal to 1/n.
-/
theorem final_number_lower_bound (n : ℕ) (h : n > 0) : 
  final_number n ≥ 1 / n :=
sorry

end stating_final_number_lower_bound_l376_37654


namespace cos_37_cos_23_minus_sin_37_sin_23_l376_37626

theorem cos_37_cos_23_minus_sin_37_sin_23 :
  Real.cos (37 * π / 180) * Real.cos (23 * π / 180) - 
  Real.sin (37 * π / 180) * Real.sin (23 * π / 180) = 1 / 2 := by
  sorry

end cos_37_cos_23_minus_sin_37_sin_23_l376_37626


namespace tower_arrangements_l376_37631

def num_red_cubes : ℕ := 2
def num_blue_cubes : ℕ := 3
def num_green_cubes : ℕ := 4
def tower_height : ℕ := 8

theorem tower_arrangements :
  (Nat.choose (num_red_cubes + num_blue_cubes + num_green_cubes) tower_height *
   Nat.factorial tower_height) /
  (Nat.factorial num_red_cubes * Nat.factorial num_blue_cubes * Nat.factorial num_green_cubes) = 1260 := by
  sorry

end tower_arrangements_l376_37631


namespace cricketer_new_average_l376_37689

/-- Represents a cricketer's performance -/
structure CricketerPerformance where
  innings : ℕ
  lastInningScore : ℕ
  averageIncrease : ℕ

/-- Calculates the new average score after the last inning -/
def newAverageScore (performance : CricketerPerformance) : ℚ :=
  sorry

/-- Theorem stating the cricketer's new average score -/
theorem cricketer_new_average
  (performance : CricketerPerformance)
  (h1 : performance.innings = 19)
  (h2 : performance.lastInningScore = 99)
  (h3 : performance.averageIncrease = 4) :
  newAverageScore performance = 27 :=
sorry

end cricketer_new_average_l376_37689


namespace smallest_c_value_l376_37699

theorem smallest_c_value (a b c : ℤ) : 
  a < b → b < c → 
  (c - b = b - a) →  -- arithmetic progression
  (b * b = a * c) →  -- geometric progression
  b = 3 * a → 
  (∀ x : ℤ, (x < a ∨ x = a) → 
    ¬(x < 3*x → 3*x < 9*x → 
      (9*x - 3*x = 3*x - x) → 
      ((3*x) * (3*x) = x * (9*x)))) → 
  c = 9 * a :=
by sorry

end smallest_c_value_l376_37699


namespace bicycle_selling_price_l376_37608

/-- Calculates the final selling price of a bicycle given the initial cost and profit percentages -/
theorem bicycle_selling_price (initial_cost : ℝ) (profit_a profit_b : ℝ) :
  initial_cost = 120 ∧ profit_a = 50 ∧ profit_b = 25 →
  initial_cost * (1 + profit_a / 100) * (1 + profit_b / 100) = 225 :=
by sorry

end bicycle_selling_price_l376_37608


namespace max_banner_area_l376_37680

/-- Represents the cost per meter of length -/
def cost_length : ℕ := 330

/-- Represents the cost per meter of width -/
def cost_width : ℕ := 450

/-- Represents the total budget in dollars -/
def budget : ℕ := 10000

/-- Proves that the maximum area of a rectangular banner with integer dimensions
    is 165 square meters, given the budget and cost constraints. -/
theorem max_banner_area :
  ∀ x y : ℕ,
    (cost_length * x + cost_width * y ≤ budget) →
    (x * y ≤ 165) :=
by sorry

end max_banner_area_l376_37680


namespace wide_flag_height_l376_37676

/-- Given the following conditions:
  - Total fabric: 1000 square feet
  - Square flags: 4 feet by 4 feet
  - Wide rectangular flags: 5 feet by unknown height
  - Tall rectangular flags: 3 feet by 5 feet
  - 16 square flags made
  - 20 wide flags made
  - 10 tall flags made
  - 294 square feet of fabric left
Prove that the height of the wide rectangular flags is 3 feet. -/
theorem wide_flag_height (total_fabric : ℝ) (square_side : ℝ) (wide_width : ℝ) 
  (tall_width tall_height : ℝ) (num_square num_wide num_tall : ℕ) (fabric_left : ℝ)
  (h_total : total_fabric = 1000)
  (h_square : square_side = 4)
  (h_wide_width : wide_width = 5)
  (h_tall : tall_width = 3 ∧ tall_height = 5)
  (h_num_square : num_square = 16)
  (h_num_wide : num_wide = 20)
  (h_num_tall : num_tall = 10)
  (h_fabric_left : fabric_left = 294) :
  ∃ (wide_height : ℝ), wide_height = 3 ∧ 
  total_fabric = num_square * square_side^2 + num_wide * wide_width * wide_height + 
                 num_tall * tall_width * tall_height + fabric_left :=
by sorry

end wide_flag_height_l376_37676


namespace tilde_r_24_l376_37694

def tilde_r (n : ℕ) : ℕ :=
  (Nat.factors n).sum + (Nat.factors n).toFinset.card

theorem tilde_r_24 : tilde_r 24 = 11 := by sorry

end tilde_r_24_l376_37694


namespace ellipse_hyperbola_tangency_l376_37660

/-- The value of m that makes the ellipse x^2 + 9y^2 = 9 tangent to the hyperbola x^2 - m(y+3)^2 = 4 -/
def tangency_value : ℚ := 5/54

/-- Definition of the ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- Definition of the hyperbola equation -/
def is_on_hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+3)^2 = 4

/-- Theorem stating that 5/54 is the value of m that makes the ellipse tangent to the hyperbola -/
theorem ellipse_hyperbola_tangency :
  ∃! (m : ℝ), m = tangency_value ∧ 
  (∃! (x y : ℝ), is_on_ellipse x y ∧ is_on_hyperbola x y m) :=
sorry

end ellipse_hyperbola_tangency_l376_37660


namespace work_completion_equivalence_l376_37675

/-- The number of days needed for the first group to complete the work -/
def days_first_group : ℕ := 96

/-- The number of men in the second group -/
def men_second_group : ℕ := 40

/-- The number of days needed for the second group to complete the work -/
def days_second_group : ℕ := 60

/-- The number of men in the first group -/
def men_first_group : ℕ := 25

theorem work_completion_equivalence :
  men_first_group * days_first_group = men_second_group * days_second_group :=
by sorry

#check work_completion_equivalence

end work_completion_equivalence_l376_37675


namespace triangle_side_length_l376_37659

/-- Given a triangle ABC with perimeter √2 + 1 and sin A + sin B = √2 sin C, 
    prove that the length of side AB is 1 -/
theorem triangle_side_length 
  (A B C : ℝ) 
  (perimeter : ℝ) 
  (h_perimeter : perimeter = Real.sqrt 2 + 1)
  (h_sin_sum : Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C)
  (h_triangle : A + B + C = π)
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C)
  : ∃ (a b c : ℝ), a + b + c = perimeter ∧ 
                    a = 1 ∧
                    a / Real.sin A = b / Real.sin B ∧
                    b / Real.sin B = c / Real.sin C :=
sorry

end triangle_side_length_l376_37659


namespace sum_fractions_equals_eight_l376_37655

/-- Given real numbers a, b, and c satisfying specific conditions, 
    prove that b/(a + b) + c/(b + c) + a/(c + a) = 8 -/
theorem sum_fractions_equals_eight 
  (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -6)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 7) :
  b / (a + b) + c / (b + c) + a / (c + a) = 8 := by
  sorry

end sum_fractions_equals_eight_l376_37655


namespace kitten_weight_l376_37611

/-- The weight of a kitten and two dogs satisfying certain conditions -/
structure AnimalWeights where
  kitten : ℝ
  smallDog : ℝ
  largeDog : ℝ
  total_weight : kitten + smallDog + largeDog = 36
  larger_pair : kitten + largeDog = 2 * smallDog
  smaller_pair : kitten + smallDog = largeDog

/-- The kitten's weight is 6 pounds given the conditions -/
theorem kitten_weight (w : AnimalWeights) : w.kitten = 6 := by
  sorry

end kitten_weight_l376_37611


namespace geometric_sequence_a3_l376_37615

/-- Given a geometric sequence {aₙ} with a₁ = 3 and a₅ = 75, prove that a₃ = 15 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (h1 : a 1 = 3) (h5 : a 5 = 75) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) : 
  a 3 = 15 := by
  sorry

end geometric_sequence_a3_l376_37615


namespace fixed_fee_is_9_39_l376_37695

/-- Represents a cloud storage service billing system -/
structure CloudStorageBilling where
  fixed_fee : ℝ
  feb_usage_fee : ℝ
  feb_total : ℝ
  mar_total : ℝ

/-- The cloud storage billing satisfies the given conditions -/
def satisfies_conditions (bill : CloudStorageBilling) : Prop :=
  bill.feb_total = bill.fixed_fee + bill.feb_usage_fee ∧
  bill.mar_total = bill.fixed_fee + 3 * bill.feb_usage_fee ∧
  bill.feb_total = 15.80 ∧
  bill.mar_total = 28.62

/-- The fixed monthly fee is 9.39 given the conditions -/
theorem fixed_fee_is_9_39 (bill : CloudStorageBilling) 
  (h : satisfies_conditions bill) : bill.fixed_fee = 9.39 := by
  sorry

end fixed_fee_is_9_39_l376_37695


namespace distance_to_line_l376_37612

/-- Given two perpendicular lines and a plane, calculate the distance from a point to one of the lines -/
theorem distance_to_line (m θ ψ : ℝ) (hm : m > 0) (hθ : 0 < θ ∧ θ < π / 2) (hψ : 0 < ψ ∧ ψ < π / 2) :
  ∃ (d : ℝ), d = Real.sqrt (m^2 + (m * Real.sin θ / Real.sin ψ)^2) ∧ d ≥ 0 := by
  sorry

end distance_to_line_l376_37612


namespace main_diagonal_squares_diagonal_5_composite_diagonal_21_composite_l376_37603

def a (k : ℕ) : ℕ := (2*k + 1)^2

def b (k : ℕ) : ℕ := (4*k - 3) * (4*k + 1)

def c (k : ℕ) : ℕ := 4*((4*k + 3)*(4*k - 1)) + 1

theorem main_diagonal_squares (k : ℕ) :
  ∃ (n : ℕ), a k = 4*n + 1 :=
sorry

theorem diagonal_5_composite (k : ℕ) (h : k > 1) :
  ¬ Nat.Prime (b k) :=
sorry

theorem diagonal_21_composite (k : ℕ) :
  ¬ Nat.Prime (c k) :=
sorry

end main_diagonal_squares_diagonal_5_composite_diagonal_21_composite_l376_37603


namespace triangle_with_altitudes_is_obtuse_l376_37601

/-- A triangle with given altitudes is obtuse -/
theorem triangle_with_altitudes_is_obtuse (h_a h_b h_c : ℝ) 
  (h_alt_a : h_a = 1/14)
  (h_alt_b : h_b = 1/10)
  (h_alt_c : h_c = 1/5)
  (h_positive_a : h_a > 0)
  (h_positive_b : h_b > 0)
  (h_positive_c : h_c > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ a + c > b ∧
    a * h_a = b * h_b ∧ b * h_b = c * h_c ∧
    (b^2 + c^2 - a^2) / (2 * b * c) < 0 :=
by sorry

end triangle_with_altitudes_is_obtuse_l376_37601


namespace ball_return_to_start_l376_37620

def ball_throw (n : ℕ) : ℕ → ℕ := λ x => (x + 3) % n

theorem ball_return_to_start :
  ∀ (start : ℕ), start < 13 →
  ∃ (k : ℕ), k > 0 ∧ (Nat.iterate (ball_throw 13) k start) = start ∧
  k = 13 :=
sorry

end ball_return_to_start_l376_37620


namespace f_sum_equals_six_l376_37669

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 9
  else 4^(-x) + 3/2

-- Theorem statement
theorem f_sum_equals_six :
  f 27 + f (-Real.log 3 / Real.log 4) = 6 := by
  sorry

end f_sum_equals_six_l376_37669


namespace candy_probability_l376_37643

theorem candy_probability (p1 p2 : ℚ) : 
  (3/8 : ℚ) ≤ p1 ∧ p1 ≤ (2/5 : ℚ) ∧ 
  (3/8 : ℚ) ≤ p2 ∧ p2 ≤ (2/5 : ℚ) ∧ 
  p1 = (5/13 : ℚ) ∧ p2 = (7/18 : ℚ) →
  ((3/8 : ℚ) ≤ (5/13 : ℚ) ∧ (5/13 : ℚ) ≤ (2/5 : ℚ)) ∧
  ((3/8 : ℚ) ≤ (7/18 : ℚ) ∧ (7/18 : ℚ) ≤ (2/5 : ℚ)) ∧
  ¬((3/8 : ℚ) ≤ (17/40 : ℚ) ∧ (17/40 : ℚ) ≤ (2/5 : ℚ)) :=
by sorry

end candy_probability_l376_37643


namespace black_raisins_amount_l376_37619

-- Define the variables
def yellow_raisins : ℝ := 0.3
def total_raisins : ℝ := 0.7

-- Define the theorem
theorem black_raisins_amount :
  total_raisins - yellow_raisins = 0.4 := by
  sorry

end black_raisins_amount_l376_37619


namespace quadratic_unique_solution_l376_37668

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 30 * x + c = 0) → 
  (a + c = 35) →
  (a < c) →
  (a = (35 - 5 * Real.sqrt 13) / 2 ∧ c = (35 + 5 * Real.sqrt 13) / 2) :=
by sorry

end quadratic_unique_solution_l376_37668


namespace simplify_expression_l376_37640

theorem simplify_expression (m : ℝ) : m^2 - m*(m-3) = 3*m := by
  sorry

end simplify_expression_l376_37640


namespace least_number_of_cookies_l376_37646

theorem least_number_of_cookies (n : ℕ) : n ≥ 208 →
  (n % 6 = 4 ∧ n % 5 = 3 ∧ n % 8 = 6 ∧ n % 9 = 7) →
  n = 208 :=
sorry

end least_number_of_cookies_l376_37646


namespace remainder_of_large_number_l376_37679

theorem remainder_of_large_number : 2345678901 % 101 = 23 := by
  sorry

end remainder_of_large_number_l376_37679


namespace shaded_area_between_squares_l376_37656

/-- The area of the shaded region between two squares -/
theorem shaded_area_between_squares (large_side small_side : ℝ) 
  (h_large : large_side = 9)
  (h_small : small_side = 4) :
  large_side ^ 2 - small_side ^ 2 = 65 := by
  sorry

end shaded_area_between_squares_l376_37656


namespace no_prime_roots_for_specific_quadratic_l376_37697

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem no_prime_roots_for_specific_quadratic :
  ¬ ∃ (k : ℤ) (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p ≠ q ∧
    p + q = 97 ∧ 
    p * q = k ∧
    ∀ (x : ℤ), x^2 - 97*x + k = 0 ↔ (x = p ∨ x = q) :=
by sorry

end no_prime_roots_for_specific_quadratic_l376_37697


namespace ranges_of_a_and_b_l376_37618

theorem ranges_of_a_and_b (a b : ℝ) (h : Real.sqrt (a^2 * b) = -a * Real.sqrt b) :
  b ≥ 0 ∧ 
  (b > 0 → a ≤ 0) ∧
  (b = 0 → ∀ x : ℝ, ∃ a : ℝ, Real.sqrt ((a : ℝ)^2 * 0) = -(a : ℝ) * Real.sqrt 0) :=
by sorry

end ranges_of_a_and_b_l376_37618


namespace unpainted_area_l376_37692

/-- The area of the unpainted region on a 5-inch wide board when crossed with a 7-inch wide board at a 45-degree angle -/
theorem unpainted_area (board1_width board2_width crossing_angle : ℝ) : 
  board1_width = 5 →
  board2_width = 7 →
  crossing_angle = 45 →
  ∃ (area : ℝ), area = 35 * Real.sqrt 2 ∧ 
    area = board1_width * Real.sqrt 2 * board2_width := by
  sorry

end unpainted_area_l376_37692


namespace unit_square_folding_l376_37637

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square with side length 1 -/
structure UnitSquare where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is on a line segment between two other points -/
def isOnSegment (P Q R : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    P.x = Q.x + t * (R.x - Q.x) ∧
    P.y = Q.y + t * (R.y - Q.y)

/-- Checks if two line segments intersect -/
def segmentsIntersect (P Q R S : Point) : Prop :=
  ∃ I : Point, isOnSegment I P Q ∧ isOnSegment I R S

theorem unit_square_folding (ABCD : UnitSquare) 
  (E : Point) (F : Point) 
  (hE : isOnSegment E ABCD.A ABCD.B) 
  (hF : isOnSegment F ABCD.C ABCD.B) 
  (hF_mid : F.x = 1 ∧ F.y = 1/2) 
  (hFold : segmentsIntersect ABCD.A ABCD.D E F ∧ 
           segmentsIntersect ABCD.C ABCD.D E F) : 
  E.x = 1/2 := by
  sorry

end unit_square_folding_l376_37637


namespace average_weight_proof_l376_37648

theorem average_weight_proof (a b c : ℝ) :
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 43 →
  b = 33 →
  (a + b) / 2 = 41 := by
sorry

end average_weight_proof_l376_37648


namespace carol_position_after_2304_moves_l376_37625

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a direction in the hexagonal grid -/
inductive Direction
  | North
  | NorthEast
  | SouthEast
  | South
  | SouthWest
  | NorthWest

/-- Represents Carol's movement pattern -/
def carolPattern (cycle : ℕ) : List (Direction × ℕ) :=
  [(Direction.North, cycle + 1),
   (Direction.NorthEast, cycle + 1),
   (Direction.SouthEast, cycle + 2),
   (Direction.South, cycle + 2),
   (Direction.SouthWest, cycle + 3),
   (Direction.NorthWest, cycle + 3)]

/-- Calculates the total steps in a given number of cycles -/
def totalStepsInCycles (k : ℕ) : ℕ :=
  k * (k + 1) + 2 * ((k + 1) * (k + 2))

/-- Theorem: Carol's position after 2304 moves -/
theorem carol_position_after_2304_moves :
  ∃ (finalPos : Point),
    (finalPos.x = 5 * Real.sqrt 3 / 2) ∧
    (finalPos.y = 23.5) ∧
    (∃ (k : ℕ),
      totalStepsInCycles k ≤ 2304 ∧
      totalStepsInCycles (k + 1) > 2304 ∧
      finalPos = -- position after completing k cycles and remaining steps
        let remainingSteps := 2304 - totalStepsInCycles k
        let partialCycle := carolPattern (k + 1)
        -- logic to apply remaining steps using partialCycle
        sorry) := by
  sorry

end carol_position_after_2304_moves_l376_37625


namespace units_digit_of_7_power_2023_l376_37639

theorem units_digit_of_7_power_2023 : (7^2023) % 10 = 3 := by
  sorry

end units_digit_of_7_power_2023_l376_37639


namespace same_color_marble_probability_l376_37636

/-- The probability of drawing three marbles of the same color from a bag containing
    red, white, and blue marbles, without replacement. -/
theorem same_color_marble_probability
  (red : ℕ) (white : ℕ) (blue : ℕ)
  (h_red : red = 5)
  (h_white : white = 7)
  (h_blue : blue = 8) :
  let total := red + white + blue
  let p_red := (red * (red - 1) * (red - 2)) / (total * (total - 1) * (total - 2))
  let p_white := (white * (white - 1) * (white - 2)) / (total * (total - 1) * (total - 2))
  let p_blue := (blue * (blue - 1) * (blue - 2)) / (total * (total - 1) * (total - 2))
  p_red + p_white + p_blue = 101 / 1140 :=
by sorry


end same_color_marble_probability_l376_37636


namespace intersection_y_intercept_sum_l376_37623

/-- Given two lines that intersect at (2, 3), prove their y-intercepts sum to 10/3 -/
theorem intersection_y_intercept_sum (a b : ℚ) : 
  (2 = (1/3) * 3 + a) → 
  (3 = (1/3) * 2 + b) → 
  a + b = 10/3 := by
  sorry

end intersection_y_intercept_sum_l376_37623


namespace pairing_theorem_l376_37685

def is_valid_pairing (n : ℕ) (pairing : List (ℕ × ℕ)) : Prop :=
  pairing.length = n ∧
  pairing.all (λ p => p.1 ≤ 2*n ∧ p.2 ≤ 2*n) ∧
  (List.range (2*n)).all (λ i => pairing.any (λ p => p.1 = i+1 ∨ p.2 = i+1))

def pairing_product (pairing : List (ℕ × ℕ)) : ℕ :=
  pairing.foldl (λ acc p => acc * (p.1 + p.2)) 1

theorem pairing_theorem (n : ℕ) (h : n > 1) :
  ∃ pairing : List (ℕ × ℕ), is_valid_pairing n pairing ∧
  ∃ m : ℕ, pairing_product pairing = m * m :=
sorry

end pairing_theorem_l376_37685


namespace ceiling_floor_sum_seven_thirds_l376_37632

theorem ceiling_floor_sum_seven_thirds : ⌈(-7 : ℚ) / 3⌉ + ⌊(7 : ℚ) / 3⌋ = 0 := by
  sorry

end ceiling_floor_sum_seven_thirds_l376_37632


namespace david_money_left_is_275_l376_37666

/-- Represents the amount of money David has left at the end of his trip -/
def david_money_left (initial_amount accommodations food_euros food_exchange_rate souvenirs_yen souvenirs_exchange_rate loan : ℚ) : ℚ :=
  let total_spent := accommodations + (food_euros * food_exchange_rate) + (souvenirs_yen * souvenirs_exchange_rate)
  initial_amount - total_spent - 500

/-- Theorem stating that David has $275 left at the end of his trip -/
theorem david_money_left_is_275 :
  david_money_left 1500 400 300 1.1 5000 0.009 200 = 275 := by
  sorry

end david_money_left_is_275_l376_37666


namespace min_sum_given_reciprocal_sum_l376_37609

theorem min_sum_given_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 2 / b = 2) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 1 / x + 2 / y = 2 → a + b ≤ x + y ∧ 
  (a + b = (3 + 2 * Real.sqrt 2) / 2 ↔ a + b = x + y) :=
sorry

end min_sum_given_reciprocal_sum_l376_37609


namespace geometric_sequence_sum_not_sufficient_nor_necessary_l376_37630

/-- A sequence is geometric if the ratio between consecutive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The statement "If {a_n} is geometric, then {a_n + a_{n+1}} is geometric" is neither sufficient nor necessary. -/
theorem geometric_sequence_sum_not_sufficient_nor_necessary :
  (∃ a : ℕ → ℝ, IsGeometric a ∧ ¬IsGeometric (fun n ↦ a n + a (n + 1))) ∧
  (∃ a : ℕ → ℝ, ¬IsGeometric a ∧ IsGeometric (fun n ↦ a n + a (n + 1))) :=
by sorry

end geometric_sequence_sum_not_sufficient_nor_necessary_l376_37630


namespace range_of_a_l376_37647

-- Define the set of valid values for a
def ValidA : Set ℝ :=
  {x | x > -1 ∧ x ≠ -5/6 ∧ x ≠ (1 + Real.sqrt 21) / 4 ∧ x ≠ (1 - Real.sqrt 21) / 4 ∧ x ≠ -7/8}

-- State the theorem
theorem range_of_a (a b c : ℝ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (eq1 : b^2 + c^2 = 2*a^2 + 16*a + 14) 
  (eq2 : b*c = a^2 - 4*a - 5) : 
  a ∈ ValidA :=
sorry

end range_of_a_l376_37647


namespace tom_gave_cars_to_five_nephews_l376_37605

/-- The number of nephews Tom gave cars to -/
def number_of_nephews : ℕ := by sorry

theorem tom_gave_cars_to_five_nephews :
  let packages := 10
  let cars_per_package := 5
  let total_cars := packages * cars_per_package
  let cars_left := 30
  let cars_given_away := total_cars - cars_left
  let fraction_per_nephew := 1 / 5
  number_of_nephews = (cars_given_away : ℚ) / (fraction_per_nephew * cars_given_away) := by sorry

end tom_gave_cars_to_five_nephews_l376_37605


namespace lcm_12_18_25_l376_37690

theorem lcm_12_18_25 : Nat.lcm (Nat.lcm 12 18) 25 = 900 := by sorry

end lcm_12_18_25_l376_37690


namespace alpha_value_l376_37670

theorem alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (2 * Real.pi)) 
  (h2 : ∃ (x y : Real), x = Real.sin (Real.pi / 6) ∧ y = Real.cos (5 * Real.pi / 6) ∧ 
    x = Real.sin α ∧ y = Real.cos α) : 
  α = 5 * Real.pi / 3 := by
  sorry

end alpha_value_l376_37670


namespace grasshopper_jump_l376_37622

/-- The jumping contest between a grasshopper and a frog -/
theorem grasshopper_jump (frog_jump grasshopper_jump difference : ℕ) 
  (h1 : frog_jump = 39)
  (h2 : frog_jump = grasshopper_jump + difference)
  (h3 : difference = 22) :
  grasshopper_jump = 17 := by
  sorry

end grasshopper_jump_l376_37622


namespace smallest_covering_l376_37600

/-- A rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- A covering of rectangles -/
structure Covering where
  target : Rectangle
  tiles : List Rectangle

/-- Whether a covering is valid (complete and non-overlapping) -/
def is_valid_covering (c : Covering) : Prop :=
  (area c.target = (c.tiles.map area).sum) ∧
  (∀ r ∈ c.tiles, r.length = 3 ∧ r.width = 4)

/-- The main theorem -/
theorem smallest_covering :
  ∃ (c : Covering),
    is_valid_covering c ∧
    c.tiles.length = 2 ∧
    (∀ (c' : Covering), is_valid_covering c' → c'.tiles.length ≥ 2) :=
sorry

end smallest_covering_l376_37600


namespace problem_1_problem_2_problem_3_problem_4_l376_37686

-- Problem 1
theorem problem_1 : (12 : ℤ) - (-18) + (-7) - 15 = 8 := by sorry

-- Problem 2
theorem problem_2 : (-81 : ℚ) / (9/4) * (4/9) / (-16) = 1 := by sorry

-- Problem 3
theorem problem_3 : ((1/3 : ℚ) - 5/6 + 7/9) * (-18) = -5 := by sorry

-- Problem 4
theorem problem_4 : -(1 : ℚ)^4 - (1/5) * (2 - (-3))^2 = -6 := by sorry

end problem_1_problem_2_problem_3_problem_4_l376_37686


namespace lisa_dvd_rental_l376_37682

theorem lisa_dvd_rental (total_spent : ℚ) (cost_per_dvd : ℚ) (h1 : total_spent = 4.8) (h2 : cost_per_dvd = 1.2) :
  total_spent / cost_per_dvd = 4 := by
  sorry

end lisa_dvd_rental_l376_37682


namespace reciprocal_sum_theorem_l376_37628

theorem reciprocal_sum_theorem (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 5 * x * y) :
  1 / x + 1 / y = 5 := by
  sorry

end reciprocal_sum_theorem_l376_37628


namespace gwen_book_count_l376_37687

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 4

/-- The number of shelves for mystery books -/
def mystery_shelves : ℕ := 5

/-- The number of shelves for picture books -/
def picture_shelves : ℕ := 3

/-- The total number of books Gwen has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem gwen_book_count : total_books = 32 := by
  sorry

end gwen_book_count_l376_37687


namespace truncated_cube_vertex_edge_count_l376_37624

/-- A polyhedron with 8 triangular faces and 6 heptagonal faces -/
structure TruncatedCube where
  triangularFaces : ℕ
  heptagonalFaces : ℕ
  triangularFaces_eq : triangularFaces = 8
  heptagonalFaces_eq : heptagonalFaces = 6

/-- The number of vertices in a TruncatedCube -/
def vertexCount (cube : TruncatedCube) : ℕ := 21

/-- The number of edges in a TruncatedCube -/
def edgeCount (cube : TruncatedCube) : ℕ := 33

/-- Theorem stating that a TruncatedCube has 21 vertices and 33 edges -/
theorem truncated_cube_vertex_edge_count (cube : TruncatedCube) : 
  vertexCount cube = 21 ∧ edgeCount cube = 33 := by
  sorry


end truncated_cube_vertex_edge_count_l376_37624


namespace area_is_33_l376_37684

/-- A line with slope -3 intersecting positive x and y axes -/
structure Line1 where
  slope : ℝ
  x_intercept : ℝ
  y_intercept : ℝ

/-- A line intersecting x and y axes -/
structure Line2 where
  x_intercept : ℝ
  y_intercept : ℝ

/-- The intersection point of two lines -/
structure Intersection where
  x : ℝ
  y : ℝ

/-- Definition of the problem setup -/
def problem_setup (l1 : Line1) (l2 : Line2) (e : Intersection) : Prop :=
  l1.slope = -3 ∧
  l1.x_intercept > 0 ∧
  l1.y_intercept > 0 ∧
  l2.x_intercept = 10 ∧
  e.x = 3 ∧
  e.y = 3

/-- The area of quadrilateral OBEC -/
def area_OBEC (l1 : Line1) (l2 : Line2) (e : Intersection) : ℝ := sorry

/-- Theorem stating the area of quadrilateral OBEC is 33 -/
theorem area_is_33 (l1 : Line1) (l2 : Line2) (e : Intersection) :
  problem_setup l1 l2 e → area_OBEC l1 l2 e = 33 := by sorry

end area_is_33_l376_37684


namespace prob_same_color_is_45_128_l376_37613

def blue_chips : ℕ := 7
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4
def total_chips : ℕ := blue_chips + red_chips + yellow_chips

def prob_same_color : ℚ :=
  (blue_chips^2 + red_chips^2 + yellow_chips^2) / total_chips^2

theorem prob_same_color_is_45_128 : prob_same_color = 45 / 128 := by
  sorry

end prob_same_color_is_45_128_l376_37613


namespace sum_in_base6_l376_37698

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (a b : ℕ) : ℕ := a * 6 + b

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : ℕ) : ℕ × ℕ × ℕ :=
  let hundreds := n / 36
  let remainder := n % 36
  let tens := remainder / 6
  let ones := remainder % 6
  (hundreds, tens, ones)

theorem sum_in_base6 :
  let a := base6ToBase10 3 5
  let b := base6ToBase10 2 5
  let sum := a + b
  let (h, t, o) := base10ToBase6 sum
  h = 1 ∧ t = 0 ∧ o = 4 := by sorry

end sum_in_base6_l376_37698


namespace f_derivative_at_zero_l376_37652

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.arctan (x^3 - x^(3/2) * Real.sin (1 / (3*x)))
  else 0

theorem f_derivative_at_zero : 
  deriv f 0 = 0 := by sorry

end f_derivative_at_zero_l376_37652


namespace alice_weight_l376_37667

theorem alice_weight (alice carol : ℝ) 
  (h1 : alice + carol = 200)
  (h2 : alice - carol = (1 / 3) * alice) : 
  alice = 120 := by
sorry

end alice_weight_l376_37667


namespace constant_term_value_l376_37681

theorem constant_term_value (x y z : ℤ) (k : ℤ) : 
  4 * x + y + z = 80 → 
  3 * x + y - z = 20 → 
  x = 20 → 
  2 * x - y - z = k → 
  k = 40 := by sorry

end constant_term_value_l376_37681


namespace trip_distance_is_3_6_miles_l376_37634

/-- Calculates the trip distance given the taxi fare parameters -/
def calculate_trip_distance (initial_fee : ℚ) (additional_charge : ℚ) (charge_distance : ℚ) (total_charge : ℚ) : ℚ :=
  let distance_charge := total_charge - initial_fee
  let segments := distance_charge / additional_charge
  segments * charge_distance

/-- Proves that the trip distance is 3.6 miles given the specified taxi fare parameters -/
theorem trip_distance_is_3_6_miles :
  let initial_fee : ℚ := 9/4  -- $2.25
  let additional_charge : ℚ := 3/10  -- $0.3
  let charge_distance : ℚ := 2/5  -- 2/5 mile
  let total_charge : ℚ := 99/20  -- $4.95
  calculate_trip_distance initial_fee additional_charge charge_distance total_charge = 18/5  -- 3.6 miles
  := by sorry

end trip_distance_is_3_6_miles_l376_37634


namespace unique_n_divisibility_l376_37641

theorem unique_n_divisibility : ∃! (n : ℕ), n > 1 ∧
  ∀ (p : ℕ), Prime p → (p ∣ (n^6 - 1)) → (p ∣ ((n^3 - 1) * (n^2 - 1))) :=
by
  -- The unique n that satisfies the condition is 2
  use 2
  sorry

end unique_n_divisibility_l376_37641


namespace simultaneous_equations_solution_l376_37602

theorem simultaneous_equations_solution (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ y = (3 * m - 2) * x^2 + 5) ↔ 
  (m ≤ 12 - 8 * Real.sqrt 2 ∨ m ≥ 12 + 8 * Real.sqrt 2) :=
sorry

end simultaneous_equations_solution_l376_37602


namespace sin_2alpha_plus_pi_6_l376_37661

theorem sin_2alpha_plus_pi_6 (α : Real) (h : Real.cos (α - π / 6) = 1 / 3) :
  Real.sin (2 * α + π / 6) = -7 / 9 := by
  sorry

end sin_2alpha_plus_pi_6_l376_37661


namespace right_triangle_identification_l376_37662

theorem right_triangle_identification (a b c : ℝ) : 
  (a = 3 ∧ b = 4 ∧ c = 5) → 
  (a^2 + b^2 = c^2) ∧ 
  ¬(2^2 + 4^2 = 5^2) ∧ 
  ¬((Real.sqrt 3)^2 + (Real.sqrt 4)^2 = (Real.sqrt 5)^2) ∧ 
  ¬(5^2 + 13^2 = 14^2) := by
  sorry

end right_triangle_identification_l376_37662


namespace expression_equals_one_l376_37663

theorem expression_equals_one (a b : ℝ) 
  (ha : a = Real.sqrt 2 + 0.8)
  (hb : b = Real.sqrt 2 - 0.2) :
  (((2-b)/(b-1)) + 2*((a-1)/(a-2))) / (b*((a-1)/(b-1)) + a*((2-b)/(a-2))) = 1 := by
  sorry

end expression_equals_one_l376_37663


namespace triangle_area_heron_l376_37635

theorem triangle_area_heron (a b c : ℝ) (h_a : a = 6) (h_b : b = 8) (h_c : c = 10) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 24 := by
  sorry

end triangle_area_heron_l376_37635


namespace least_non_lucky_multiple_of_7_l376_37627

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isLuckyInteger (n : ℕ) : Prop :=
  n > 0 ∧ n % sumOfDigits n = 0

def isMultipleOf7 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 * k

theorem least_non_lucky_multiple_of_7 :
  (∀ n : ℕ, n > 0 ∧ n < 14 ∧ isMultipleOf7 n → isLuckyInteger n) ∧
  isMultipleOf7 14 ∧
  ¬isLuckyInteger 14 :=
sorry

end least_non_lucky_multiple_of_7_l376_37627


namespace lattice_point_proximity_probability_l376_37673

theorem lattice_point_proximity_probability (r : ℝ) : 
  (r > 0) → 
  (π * r^2 = 1/3) → 
  (∃ (p : ℝ × ℝ), p.1 ≥ 0 ∧ p.1 ≤ 1 ∧ p.2 ≥ 0 ∧ p.2 ≤ 1 ∧ 
    ((p.1^2 + p.2^2 ≤ r^2) ∨ 
     ((1 - p.1)^2 + p.2^2 ≤ r^2) ∨ 
     (p.1^2 + (1 - p.2)^2 ≤ r^2) ∨ 
     ((1 - p.1)^2 + (1 - p.2)^2 ≤ r^2))) = 
  (r = Real.sqrt (1 / (3 * π))) :=
sorry

end lattice_point_proximity_probability_l376_37673


namespace divisibleByTwo_infinite_lessThanBillion_finite_l376_37621

-- Define the set of numbers divisible by 2
def divisibleByTwo : Set Int := {x | ∃ n : Int, x = 2 * n}

-- Define the set of positive integers less than 1 billion
def lessThanBillion : Set Nat := {x | x > 0 ∧ x < 1000000000}

-- Theorem 1: The set of numbers divisible by 2 is infinite
theorem divisibleByTwo_infinite : Set.Infinite divisibleByTwo := by
  sorry

-- Theorem 2: The set of positive integers less than 1 billion is finite
theorem lessThanBillion_finite : Set.Finite lessThanBillion := by
  sorry

end divisibleByTwo_infinite_lessThanBillion_finite_l376_37621

import Mathlib

namespace abs_complex_fraction_equals_sqrt_two_l1478_147883

/-- The absolute value of the complex number (1-3i)/(1+2i) is equal to √2 -/
theorem abs_complex_fraction_equals_sqrt_two :
  let z : ℂ := (1 - 3*I) / (1 + 2*I)
  ‖z‖ = Real.sqrt 2 := by
  sorry

end abs_complex_fraction_equals_sqrt_two_l1478_147883


namespace cycle_gain_percent_l1478_147869

def gain_percent (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

theorem cycle_gain_percent :
  let cost_price : ℚ := 900
  let selling_price : ℚ := 1150
  gain_percent cost_price selling_price = (1150 - 900) / 900 * 100 := by
  sorry

end cycle_gain_percent_l1478_147869


namespace jerry_weekly_spending_jerry_specific_case_l1478_147829

/-- Given Jerry's earnings and the duration the money lasted, calculate his weekly spending. -/
theorem jerry_weekly_spending (lawn_money weed_money : ℕ) (weeks : ℕ) : 
  (lawn_money + weed_money) / weeks = (lawn_money + weed_money) / weeks :=
by sorry

/-- Jerry's specific case -/
theorem jerry_specific_case : 
  (14 + 31) / 9 = 5 :=
by sorry

end jerry_weekly_spending_jerry_specific_case_l1478_147829


namespace min_value_a_l1478_147804

theorem min_value_a (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 9) → a ≥ 4 :=
by sorry

end min_value_a_l1478_147804


namespace hockey_league_face_count_l1478_147850

/-- The number of times each team faces all other teams in a hockey league -/
def face_count (num_teams : ℕ) (total_games : ℕ) : ℕ :=
  total_games / (num_teams * (num_teams - 1) / 2)

/-- Theorem: In a hockey league with 18 teams, where each team faces all other teams
    the same number of times, and a total of 1530 games are played in the season,
    each team faces all the other teams 5 times. -/
theorem hockey_league_face_count :
  face_count 18 1530 = 5 := by
  sorry

end hockey_league_face_count_l1478_147850


namespace no_rational_solution_for_odd_coeff_quadratic_l1478_147833

theorem no_rational_solution_for_odd_coeff_quadratic 
  (a b c : ℤ) 
  (ha : Odd a) 
  (hb : Odd b) 
  (hc : Odd c) : 
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := by
  sorry

end no_rational_solution_for_odd_coeff_quadratic_l1478_147833


namespace find_number_l1478_147879

theorem find_number (A B : ℕ) (h1 : B = 913) (h2 : Nat.lcm A B = 2310) (h3 : Nat.gcd A B = 83) : A = 210 := by
  sorry

end find_number_l1478_147879


namespace special_triangle_properties_l1478_147892

/-- A triangle with an inscribed circle of radius 2, where one side is divided into segments of 4 and 6 by the point of tangency -/
structure SpecialTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  a : ℝ
  /-- The length of the second segment of the divided side -/
  b : ℝ
  /-- The radius is 2 -/
  h_r : r = 2
  /-- The first segment is 4 -/
  h_a : a = 4
  /-- The second segment is 6 -/
  h_b : b = 6

/-- The area of the triangle -/
def area (t : SpecialTriangle) : ℝ := 24

/-- The triangle is right-angled -/
def is_right_triangle (t : SpecialTriangle) : Prop :=
  ∃ (x y z : ℝ), x^2 + y^2 = z^2 ∧ 
    ((x = t.a + t.b ∧ y = 2 * t.r ∧ z = t.a + t.b + 2 * t.r) ∨
     (x = t.a + t.b ∧ y = t.a + t.b + 2 * t.r ∧ z = 2 * t.r) ∨
     (x = 2 * t.r ∧ y = t.a + t.b + 2 * t.r ∧ z = t.a + t.b))

theorem special_triangle_properties (t : SpecialTriangle) :
  is_right_triangle t ∧ area t = 24 := by
  sorry

end special_triangle_properties_l1478_147892


namespace quadratic_function_properties_l1478_147834

def f (x : ℝ) : ℝ := x^2 - 3*x + 4

def g (x m : ℝ) : ℝ := 2*x + m

def h (x t : ℝ) : ℝ := f x - (2*t - 3)*x

def F (x m : ℝ) : ℝ := f x - g x m

theorem quadratic_function_properties :
  (f 0 = 4) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 4 ∧ f x₁ = 2*x₁ ∧ f x₂ = 2*x₂) ∧
  (∃ t : ℝ, t = Real.sqrt 2 / 2 ∧ 
    (∀ x : ℝ, x ∈ Set.Icc 0 1 → h x t ≥ 7/2) ∧
    (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ h x t = 7/2)) ∧
  (∀ m : ℝ, m ∈ Set.Ioo (-9/4) (-2) →
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 3 ∧ x₂ ∈ Set.Icc 0 3 ∧ 
      F x₁ m = 0 ∧ F x₂ m = 0)) :=
sorry

end quadratic_function_properties_l1478_147834


namespace expression_evaluation_l1478_147878

theorem expression_evaluation :
  let x : ℝ := (1/2)^2023
  let y : ℝ := 2^2022
  (2*x + y)^2 - (2*x + y)*(2*x - y) - 2*y*(x + y) = 1 := by
sorry

end expression_evaluation_l1478_147878


namespace intersection_of_A_and_B_l1478_147838

def set_A : Set ℝ := {x | 2 * x + 1 > 0}
def set_B : Set ℝ := {x | |x - 1| < 2}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x | -1/2 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l1478_147838


namespace rectangle_area_l1478_147862

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) :
  square_area = 2500 →
  rectangle_breadth = 10 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area = 200 := by
sorry

end rectangle_area_l1478_147862


namespace polynomial_expansion_l1478_147858

theorem polynomial_expansion (z : R) [CommRing R] :
  (3 * z^2 + 4 * z - 7) * (4 * z^3 - 3 * z + 2) =
  12 * z^5 + 16 * z^4 - 37 * z^3 - 6 * z^2 + 29 * z - 14 := by
  sorry

end polynomial_expansion_l1478_147858


namespace unique_solution_quadratic_l1478_147813

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x + 6) * (x + 3) = k + 3 * x) ↔ k = 9 := by
  sorry

end unique_solution_quadratic_l1478_147813


namespace carrot_to_green_bean_ratio_l1478_147847

/-- Given a grocery bag with a maximum capacity and known weights of items,
    prove that the ratio of carrots to green beans is 1:2. -/
theorem carrot_to_green_bean_ratio
  (bag_capacity : ℕ)
  (green_beans : ℕ)
  (milk : ℕ)
  (remaining_capacity : ℕ)
  (h1 : bag_capacity = 20)
  (h2 : green_beans = 4)
  (h3 : milk = 6)
  (h4 : remaining_capacity = 2)
  (h5 : green_beans + milk + remaining_capacity = bag_capacity) :
  (bag_capacity - green_beans - milk) / green_beans = 1 / 2 := by
  sorry


end carrot_to_green_bean_ratio_l1478_147847


namespace arman_earnings_l1478_147830

/-- Represents the pay rates and working hours for Arman over two weeks -/
structure PayData :=
  (last_week_rate : ℝ)
  (last_week_hours : ℝ)
  (this_week_rate_increase : ℝ)
  (overtime_multiplier : ℝ)
  (weekend_multiplier : ℝ)
  (night_shift_multiplier : ℝ)
  (monday_hours : ℝ)
  (monday_night_hours : ℝ)
  (tuesday_hours : ℝ)
  (tuesday_night_hours : ℝ)
  (wednesday_hours : ℝ)
  (thursday_hours : ℝ)
  (thursday_night_hours : ℝ)
  (thursday_overtime : ℝ)
  (friday_hours : ℝ)
  (saturday_hours : ℝ)
  (sunday_hours : ℝ)
  (sunday_night_hours : ℝ)

/-- Calculates the total earnings for Arman over two weeks -/
def calculate_earnings (data : PayData) : ℝ :=
  let last_week_earnings := data.last_week_rate * data.last_week_hours
  let this_week_rate := data.last_week_rate + data.this_week_rate_increase
  let this_week_earnings :=
    (data.monday_hours - data.monday_night_hours) * this_week_rate +
    data.monday_night_hours * this_week_rate * data.night_shift_multiplier +
    (data.tuesday_hours - data.tuesday_night_hours) * this_week_rate +
    data.tuesday_night_hours * this_week_rate * data.night_shift_multiplier +
    (data.tuesday_hours - 8) * this_week_rate * data.overtime_multiplier +
    data.wednesday_hours * this_week_rate +
    (data.thursday_hours - data.thursday_night_hours - data.thursday_overtime) * this_week_rate +
    data.thursday_night_hours * this_week_rate * data.night_shift_multiplier +
    data.thursday_overtime * this_week_rate * data.overtime_multiplier +
    data.friday_hours * this_week_rate +
    data.saturday_hours * this_week_rate * data.weekend_multiplier +
    (data.sunday_hours - data.sunday_night_hours) * this_week_rate * data.weekend_multiplier +
    data.sunday_night_hours * this_week_rate * data.weekend_multiplier * data.night_shift_multiplier
  last_week_earnings + this_week_earnings

/-- Theorem stating that Arman's total earnings for the two weeks equal $1055.46 -/
theorem arman_earnings (data : PayData)
  (h1 : data.last_week_rate = 10)
  (h2 : data.last_week_hours = 35)
  (h3 : data.this_week_rate_increase = 0.5)
  (h4 : data.overtime_multiplier = 1.5)
  (h5 : data.weekend_multiplier = 1.7)
  (h6 : data.night_shift_multiplier = 1.3)
  (h7 : data.monday_hours = 8)
  (h8 : data.monday_night_hours = 3)
  (h9 : data.tuesday_hours = 10)
  (h10 : data.tuesday_night_hours = 4)
  (h11 : data.wednesday_hours = 8)
  (h12 : data.thursday_hours = 9)
  (h13 : data.thursday_night_hours = 3)
  (h14 : data.thursday_overtime = 1)
  (h15 : data.friday_hours = 5)
  (h16 : data.saturday_hours = 6)
  (h17 : data.sunday_hours = 4)
  (h18 : data.sunday_night_hours = 2) :
  calculate_earnings data = 1055.46 := by sorry

end arman_earnings_l1478_147830


namespace cheyenne_earnings_l1478_147821

def total_pots : ℕ := 80
def cracked_fraction : ℚ := 2/5
def price_per_pot : ℕ := 40

theorem cheyenne_earnings : 
  (total_pots - (cracked_fraction * total_pots).num) * price_per_pot = 1920 := by
  sorry

end cheyenne_earnings_l1478_147821


namespace edwards_earnings_l1478_147846

/-- Edward's lawn mowing earnings problem -/
theorem edwards_earnings (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ) :
  rate = 4 →
  total_lawns = 17 →
  forgotten_lawns = 9 →
  rate * (total_lawns - forgotten_lawns) = 32 :=
by
  sorry

end edwards_earnings_l1478_147846


namespace max_correct_answers_l1478_147801

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (total_score : ℕ) :
  total_questions = 50 →
  correct_points = 5 →
  incorrect_points = 2 →
  total_score = 150 →
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct * correct_points - incorrect * incorrect_points = total_score ∧
    correct ≤ 35 ∧
    ∀ (c : ℕ), c > correct →
      ¬(∃ (i u : ℕ), c + i + u = total_questions ∧
        c * correct_points - i * incorrect_points = total_score) :=
by sorry

end max_correct_answers_l1478_147801


namespace not_prime_4n_squared_minus_1_l1478_147812

theorem not_prime_4n_squared_minus_1 (n : ℤ) (h : n ≥ 2) : ¬ Prime (4 * n^2 - 1) := by
  sorry

end not_prime_4n_squared_minus_1_l1478_147812


namespace three_non_adjacent_from_ten_l1478_147806

/-- The number of ways to choose 3 non-adjacent items from a set of 10 items. -/
def non_adjacent_choices (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + 1) k

/-- Theorem: There are 56 ways to choose 3 non-adjacent items from a set of 10 items. -/
theorem three_non_adjacent_from_ten : non_adjacent_choices 10 3 = 56 := by
  sorry

end three_non_adjacent_from_ten_l1478_147806


namespace floor_product_twenty_l1478_147803

theorem floor_product_twenty (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := by sorry

end floor_product_twenty_l1478_147803


namespace cubic_sum_equals_nine_l1478_147861

theorem cubic_sum_equals_nine (a b : ℝ) 
  (h1 : a^5 - a^4*b - a^4 + a - b - 1 = 0)
  (h2 : 2*a - 3*b = 1) : 
  a^3 + b^3 = 9 := by
sorry

end cubic_sum_equals_nine_l1478_147861


namespace parabola_vertex_l1478_147860

/-- The vertex of a parabola defined by y = a(x+1)^2 - 2 is at (-1, -2) --/
theorem parabola_vertex (a : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * (x + 1)^2 - 2
  ∃! p : ℝ × ℝ, p.1 = -1 ∧ p.2 = -2 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
by
  sorry

end parabola_vertex_l1478_147860


namespace toy_pickup_time_l1478_147836

/-- The time required to put all toys in the box -/
def time_to_fill_box (total_toys : ℕ) (toys_in_per_cycle : ℕ) (toys_out_per_cycle : ℕ) (cycle_time : ℚ) : ℚ :=
  let net_toys_per_cycle := toys_in_per_cycle - toys_out_per_cycle
  let full_cycles := (total_toys - toys_in_per_cycle) / net_toys_per_cycle
  let full_cycles_time := full_cycles * cycle_time
  let final_cycle_time := cycle_time
  (full_cycles_time + final_cycle_time) / 60

/-- The problem statement -/
theorem toy_pickup_time :
  time_to_fill_box 50 4 3 (45 / 60) = 36.75 := by
  sorry

end toy_pickup_time_l1478_147836


namespace wendy_albums_l1478_147843

theorem wendy_albums (total_pictures : ℕ) (first_album : ℕ) (pictures_per_album : ℕ) 
  (h1 : total_pictures = 79)
  (h2 : first_album = 44)
  (h3 : pictures_per_album = 7) :
  (total_pictures - first_album) / pictures_per_album = 5 :=
by sorry

end wendy_albums_l1478_147843


namespace binomial_12_3_l1478_147823

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_l1478_147823


namespace clark_discount_clark_discount_proof_l1478_147855

/-- Calculates the discount given to Clark for purchasing auto parts -/
theorem clark_discount (original_price : ℕ) (quantity : ℕ) (total_paid : ℕ) : ℕ :=
  let total_without_discount := original_price * quantity
  let discount := total_without_discount - total_paid
  discount

/-- Proves that Clark's discount is $121 given the problem conditions -/
theorem clark_discount_proof :
  clark_discount 80 7 439 = 121 := by
  sorry

end clark_discount_clark_discount_proof_l1478_147855


namespace equation_solution_l1478_147831

theorem equation_solution (a b c d : ℝ) : 
  a^3 + b^3 + c^3 + a^2 + b^2 + 1 = d^2 + d + Real.sqrt (a + b + c - 2*d) →
  d = 1 ∨ d = -4/3 :=
by sorry

end equation_solution_l1478_147831


namespace ian_money_left_l1478_147875

/-- Calculates the amount of money Ian has left after paying off debts --/
def money_left (lottery_win : ℕ) (colin_payment : ℕ) : ℕ :=
  let helen_payment := 2 * colin_payment
  let benedict_payment := helen_payment / 2
  lottery_win - (colin_payment + helen_payment + benedict_payment)

/-- Theorem stating that Ian has $20 left after paying off debts --/
theorem ian_money_left : money_left 100 20 = 20 := by
  sorry

end ian_money_left_l1478_147875


namespace special_cubic_e_value_l1478_147897

/-- A cubic polynomial with specific properties -/
structure SpecialCubic where
  d : ℝ
  e : ℝ
  zeros_mean_prod : (- d / 9) = 2 * (-4)
  coeff_sum_y_intercept : 3 + d + e + 12 = 12

/-- The value of e in the special cubic polynomial is -75 -/
theorem special_cubic_e_value (p : SpecialCubic) : p.e = -75 := by
  sorry

end special_cubic_e_value_l1478_147897


namespace isosceles_triangle_side_length_l1478_147819

/-- Given an equilateral triangle with side length 2 and three right-angled isosceles triangles
    constructed on its sides, if the total area of the three right-angled isosceles triangles
    equals the area of the equilateral triangle, then the length of the congruent sides of
    one right-angled isosceles triangle is √(6√3)/3. -/
theorem isosceles_triangle_side_length :
  let equilateral_side : ℝ := 2
  let equilateral_area : ℝ := Real.sqrt 3 / 4 * equilateral_side^2
  let isosceles_area : ℝ := equilateral_area / 3
  let isosceles_side : ℝ := Real.sqrt (2 * isosceles_area)
  isosceles_side = Real.sqrt (6 * Real.sqrt 3) / 3 :=
by sorry

end isosceles_triangle_side_length_l1478_147819


namespace existence_of_close_ratios_l1478_147863

theorem existence_of_close_ratios (S : Finset ℝ) (h : S.card = 2000) :
  ∃ (a b c d : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
  a > b ∧ c > d ∧ (a ≠ c ∨ b ≠ d) ∧
  |((a - b) / (c - d)) - 1| < (1 : ℝ) / 100000 := by
  sorry

end existence_of_close_ratios_l1478_147863


namespace complete_square_quadratic_l1478_147876

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (c d : ℝ), x^2 + 12*x + 4 = 0 ↔ (x + c)^2 = d ∧ d = 32 := by
  sorry

end complete_square_quadratic_l1478_147876


namespace quadratic_equation_roots_l1478_147809

theorem quadratic_equation_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - (k + 2) * x₁ + 2 * k - 1 = 0 ∧
    x₂^2 - (k + 2) * x₂ + 2 * k - 1 = 0) ∧
  (3^2 - (k + 2) * 3 + 2 * k - 1 = 0 → k = 2 ∧ 1^2 - (k + 2) * 1 + 2 * k - 1 = 0) :=
by sorry

end quadratic_equation_roots_l1478_147809


namespace disinfectant_purchase_theorem_l1478_147841

/-- Represents the cost and quantity of disinfectants --/
structure DisinfectantPurchase where
  costA : ℕ  -- Cost of one bottle of Class A disinfectant
  costB : ℕ  -- Cost of one bottle of Class B disinfectant
  quantityA : ℕ  -- Number of bottles of Class A disinfectant
  quantityB : ℕ  -- Number of bottles of Class B disinfectant

/-- Theorem about disinfectant purchase --/
theorem disinfectant_purchase_theorem 
  (purchase : DisinfectantPurchase)
  (total_cost : purchase.costA * purchase.quantityA + purchase.costB * purchase.quantityB = 2250)
  (cost_difference : purchase.costA + 15 = purchase.costB)
  (quantities : purchase.quantityA = 80 ∧ purchase.quantityB = 35)
  (new_total : ℕ)
  (new_budget : new_total * purchase.costA + (50 - new_total) * purchase.costB ≤ 1200)
  : purchase.costA = 15 ∧ purchase.costB = 30 ∧ new_total ≥ 20 := by
  sorry

#check disinfectant_purchase_theorem

end disinfectant_purchase_theorem_l1478_147841


namespace opposite_numbers_l1478_147898

-- Define the concept of opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Theorem statement
theorem opposite_numbers : are_opposite (-|-(1/100)|) (-(-1/100)) := by
  sorry

end opposite_numbers_l1478_147898


namespace stating_min_connections_for_given_problem_l1478_147853

/-- Represents the number of cities -/
def num_cities : Nat := 100

/-- Represents the number of different routes -/
def num_routes : Nat := 1000

/-- 
Given a number of cities and a number of routes, 
calculates the minimum number of flight connections per city 
that allows for the specified number of routes.
-/
def min_connections (cities : Nat) (routes : Nat) : Nat :=
  sorry

/-- 
Theorem stating that given 100 cities and 1000 routes, 
the minimum number of connections per city is 4.
-/
theorem min_connections_for_given_problem : 
  min_connections num_cities num_routes = 4 := by sorry

end stating_min_connections_for_given_problem_l1478_147853


namespace add_three_people_to_two_rows_l1478_147887

/-- The number of ways to add three people to two rows of people -/
def add_people_ways (front_row : ℕ) (back_row : ℕ) (people_to_add : ℕ) : ℕ :=
  (people_to_add) * (front_row + 1) * (back_row + 1) * (back_row + 2)

/-- Theorem: The number of ways to add three people to two rows with 3 in front and 4 in back is 360 -/
theorem add_three_people_to_two_rows :
  add_people_ways 3 4 3 = 360 := by
  sorry

end add_three_people_to_two_rows_l1478_147887


namespace slope_of_l₃_l1478_147844

-- Define the lines and points
def l₁ (x y : ℝ) : Prop := 2 * x + 3 * y = 6
def l₂ (y : ℝ) : Prop := y = 2
def A : ℝ × ℝ := (0, -3)

-- Define the properties of the lines and points
axiom l₁_through_A : l₁ A.1 A.2
axiom l₂_meets_l₁ : ∃ B : ℝ × ℝ, l₁ B.1 B.2 ∧ l₂ B.2
axiom l₃_positive_slope : ∃ m : ℝ, m > 0 ∧ ∀ x y : ℝ, y - A.2 = m * (x - A.1)
axiom l₃_through_A : ∀ x y : ℝ, y - A.2 = (y - A.2) / (x - A.1) * (x - A.1)
axiom l₃_meets_l₂ : ∃ C : ℝ × ℝ, l₂ C.2 ∧ C.2 - A.2 = (C.2 - A.2) / (C.1 - A.1) * (C.1 - A.1)

-- Define the area of triangle ABC
axiom triangle_area : ∃ B C : ℝ × ℝ, 
  l₁ B.1 B.2 ∧ l₂ B.2 ∧ l₂ C.2 ∧ C.2 - A.2 = (C.2 - A.2) / (C.1 - A.1) * (C.1 - A.1) ∧
  1/2 * |(B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)| = 10

-- Theorem statement
theorem slope_of_l₃ : 
  ∃ m : ℝ, m = 5/4 ∧ ∀ x y : ℝ, y - A.2 = m * (x - A.1) :=
sorry

end slope_of_l₃_l1478_147844


namespace inequality_solution_set_l1478_147816

-- Define the function f(x) = x^2 + 2x - 3
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | f x < 0} = {x : ℝ | -3 < x ∧ x < 1} := by sorry

end inequality_solution_set_l1478_147816


namespace arithmetic_sequence_general_term_l1478_147891

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2
  arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The theorem stating the general term of the arithmetic sequence -/
theorem arithmetic_sequence_general_term 
  (seq : ArithmeticSequence) 
  (sum10 : seq.S 10 = 10) 
  (sum20 : seq.S 20 = 220) : 
  ∀ n, seq.a n = 2 * n - 10 := by
  sorry

end arithmetic_sequence_general_term_l1478_147891


namespace jacket_price_reduction_l1478_147849

theorem jacket_price_reduction (x : ℝ) : 
  (1 - x / 100) * (1 - 0.15) * (1 + 56.86274509803921 / 100) = 1 → x = 25 := by
  sorry

end jacket_price_reduction_l1478_147849


namespace factorial_divisibility_theorem_l1478_147871

def factorial (n : ℕ) : ℕ := Nat.factorial n

def sum_factorials (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => factorial (i + 1))

theorem factorial_divisibility_theorem :
  ∀ n : ℕ, n > 2 → ¬(factorial (n + 1) ∣ sum_factorials n) ∧
  (factorial 2 ∣ sum_factorials 1) ∧
  (factorial 3 ∣ sum_factorials 2) ∧
  ∀ m : ℕ, m ≠ 1 ∧ m ≠ 2 → ¬(factorial (m + 1) ∣ sum_factorials m) :=
by sorry

end factorial_divisibility_theorem_l1478_147871


namespace unique_modular_integer_l1478_147832

theorem unique_modular_integer : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] := by
  sorry

end unique_modular_integer_l1478_147832


namespace ella_video_game_spending_l1478_147865

/-- Proves that Ella spends 40% of her salary on video games given the conditions -/
theorem ella_video_game_spending (
  last_year_spending : ℝ)
  (new_salary : ℝ)
  (raise_percentage : ℝ)
  (h1 : last_year_spending = 100)
  (h2 : new_salary = 275)
  (h3 : raise_percentage = 0.1)
  : (last_year_spending / (new_salary / (1 + raise_percentage))) * 100 = 40 := by
  sorry

end ella_video_game_spending_l1478_147865


namespace dispatch_three_male_two_female_dispatch_at_least_two_male_l1478_147882

def male_drivers : ℕ := 5
def female_drivers : ℕ := 4
def team_size : ℕ := 5

/-- The number of ways to choose 3 male drivers and 2 female drivers -/
theorem dispatch_three_male_two_female : 
  (Nat.choose male_drivers 3) * (Nat.choose female_drivers 2) = 60 := by sorry

/-- The number of ways to dispatch with at least two male drivers -/
theorem dispatch_at_least_two_male : 
  (Nat.choose male_drivers 2) * (Nat.choose female_drivers 3) +
  (Nat.choose male_drivers 3) * (Nat.choose female_drivers 2) +
  (Nat.choose male_drivers 4) * (Nat.choose female_drivers 1) +
  (Nat.choose male_drivers 5) * (Nat.choose female_drivers 0) = 121 := by sorry

end dispatch_three_male_two_female_dispatch_at_least_two_male_l1478_147882


namespace vacation_book_selection_l1478_147810

theorem vacation_book_selection (total_books : ℕ) (books_to_bring : ℕ) (favorite_book : ℕ) :
  total_books = 15 →
  books_to_bring = 3 →
  favorite_book = 1 →
  Nat.choose (total_books - favorite_book) (books_to_bring - favorite_book) = 91 :=
by
  sorry

end vacation_book_selection_l1478_147810


namespace max_blue_points_max_blue_points_2016_l1478_147864

/-- Given a set of spheres, some red and some green, with blue points at each red-green contact,
    the maximum number of blue points is achieved when there are equal numbers of red and green spheres. -/
theorem max_blue_points (total_spheres : ℕ) (h_total : total_spheres = 2016) :
  ∃ (red_spheres green_spheres : ℕ),
    red_spheres + green_spheres = total_spheres ∧
    red_spheres * green_spheres ≤ (total_spheres / 2) ^ 2 :=
by sorry

/-- The maximum number of blue points for 2016 spheres is 1008^2. -/
theorem max_blue_points_2016 :
  ∃ (red_spheres green_spheres : ℕ),
    red_spheres + green_spheres = 2016 ∧
    red_spheres * green_spheres = 1008 ^ 2 :=
by sorry

end max_blue_points_max_blue_points_2016_l1478_147864


namespace urn_probability_l1478_147886

theorem urn_probability (N : ℝ) : N = 21 →
  (3 / 8) * (9 / (9 + N)) + (5 / 8) * (N / (9 + N)) = 0.55 := by
  sorry

#check urn_probability

end urn_probability_l1478_147886


namespace chord_equation_l1478_147895

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Represents a line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Main theorem -/
theorem chord_equation (e : Ellipse) (p : Point) (l : Line) : 
  e.a^2 = 8 ∧ e.b^2 = 4 ∧ p.x = 2 ∧ p.y = -1 ∧ 
  (∃ p1 p2 : Point, 
    pointOnEllipse p1 e ∧ 
    pointOnEllipse p2 e ∧
    p.x = (p1.x + p2.x) / 2 ∧ 
    p.y = (p1.y + p2.y) / 2 ∧
    pointOnLine p1 l ∧ 
    pointOnLine p2 l) →
  l.slope = 1 ∧ l.intercept = -3 := by
  sorry

end chord_equation_l1478_147895


namespace triangle_count_48_l1478_147827

/-- The number of distinct, non-degenerate triangles with integer side lengths and perimeter n -/
def count_triangles (n : ℕ) : ℕ :=
  let isosceles := (n - 1) / 2 - n / 4
  let scalene := Nat.choose (n - 1) 2 - 3 * Nat.choose (n / 2) 2
  let total := (scalene - 3 * isosceles) / 6
  if n % 3 = 0 then total - 1 else total

theorem triangle_count_48 :
  ∃ n : ℕ, n > 0 ∧ count_triangles n = 48 :=
sorry

end triangle_count_48_l1478_147827


namespace white_square_area_l1478_147873

theorem white_square_area (cube_edge : ℝ) (green_paint_area : ℝ) (white_square_area : ℝ) : 
  cube_edge = 12 →
  green_paint_area = 432 →
  white_square_area = (cube_edge ^ 2) - (green_paint_area / 6) →
  white_square_area = 72 := by
sorry

end white_square_area_l1478_147873


namespace team_selection_count_l1478_147805

def boys : ℕ := 7
def girls : ℕ := 9
def team_size : ℕ := 7
def boys_in_team : ℕ := 4
def girls_in_team : ℕ := 3

theorem team_selection_count :
  (Nat.choose boys boys_in_team) * (Nat.choose girls girls_in_team) = 2940 := by
  sorry

end team_selection_count_l1478_147805


namespace sum_reciprocal_products_equals_three_eighths_l1478_147894

theorem sum_reciprocal_products_equals_three_eighths :
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) +
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) = 3 / 8 := by
  sorry

end sum_reciprocal_products_equals_three_eighths_l1478_147894


namespace prisoner_selection_l1478_147839

/-- Given 25 prisoners, prove the number of ways to choose 3 in order and without order. -/
theorem prisoner_selection (n : ℕ) (h : n = 25) : 
  (n * (n - 1) * (n - 2) = 13800) ∧ (Nat.choose n 3 = 2300) := by
  sorry

end prisoner_selection_l1478_147839


namespace power_function_inequality_l1478_147811

-- Define the power function
def f (x : ℝ) : ℝ := x^(1/5)

-- State the theorem
theorem power_function_inequality (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) :
  f ((x1 + x2) / 2) > (f x1 + f x2) / 2 := by
  sorry

end power_function_inequality_l1478_147811


namespace geometric_sequence_property_l1478_147854

/-- Given a geometric sequence {a_n} where a_{2013} + a_{2015} = π, 
    prove that a_{2014}(a_{2012} + 2a_{2014} + a_{2016}) = π^2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h1 : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) 
    (h2 : a 2013 + a 2015 = π) : 
  a 2014 * (a 2012 + 2 * a 2014 + a 2016) = π^2 := by
  sorry


end geometric_sequence_property_l1478_147854


namespace special_house_profit_calculation_l1478_147868

def special_house_profit (extra_cost : ℝ) (price_multiplier : ℝ) (standard_house_price : ℝ) : ℝ :=
  price_multiplier * standard_house_price - standard_house_price - extra_cost

theorem special_house_profit_calculation :
  special_house_profit 100000 1.5 320000 = 60000 := by
  sorry

end special_house_profit_calculation_l1478_147868


namespace middle_school_math_club_payment_l1478_147851

theorem middle_school_math_club_payment (A : Nat) : 
  A < 10 → (100 + 10 * A + 2) % 11 = 0 ↔ A = 3 := by
  sorry

end middle_school_math_club_payment_l1478_147851


namespace last_two_digits_product_l1478_147815

theorem last_two_digits_product (A B : Nat) : 
  (A ≤ 9) → 
  (B ≤ 9) → 
  (10 * A + B) % 5 = 0 → 
  A + B = 16 → 
  A * B = 30 := by
  sorry

end last_two_digits_product_l1478_147815


namespace exam_students_count_l1478_147808

theorem exam_students_count (total_average : ℝ) (excluded_average : ℝ) (new_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 30) 
  (h3 : new_average = 90) (h4 : excluded_count = 5) :
  ∃ (N : ℕ), N > 0 ∧ 
  (N * total_average - excluded_count * excluded_average) / (N - excluded_count) = new_average :=
by
  sorry

end exam_students_count_l1478_147808


namespace arrangement_count_is_correct_l1478_147848

/-- The number of ways to arrange 5 people in a row with one person between A and B -/
def arrangement_count : ℕ := 36

/-- The total number of people in the arrangement -/
def total_people : ℕ := 5

/-- The number of people between A and B -/
def people_between : ℕ := 1

theorem arrangement_count_is_correct :
  arrangement_count = 
    2 * (total_people - 2) * (Nat.factorial (total_people - 2)) :=
by sorry

end arrangement_count_is_correct_l1478_147848


namespace least_positive_k_for_equation_l1478_147814

theorem least_positive_k_for_equation : ∃ (k : ℕ), 
  (k > 0) ∧ 
  (∃ (x : ℤ), x > 0 ∧ x + 6 + 8*k = k*(x + 8)) ∧
  (∀ (j : ℕ), 0 < j ∧ j < k → ¬∃ (y : ℤ), y > 0 ∧ y + 6 + 8*j = j*(y + 8)) ∧
  k = 2 := by
sorry

end least_positive_k_for_equation_l1478_147814


namespace no_solution_exists_l1478_147807

theorem no_solution_exists : ¬∃ (x : ℕ), (42 + x = 3 * (8 + x)) ∧ (42 + x = 2 * (10 + x)) := by
  sorry

end no_solution_exists_l1478_147807


namespace intersection_points_on_circle_l1478_147884

/-- The parabolas y = (x - 2)^2 and x - 5 = (y + 1)^2 intersect at four points that lie on a circle with radius squared equal to 1.5 -/
theorem intersection_points_on_circle :
  ∃ (c : ℝ × ℝ) (r : ℝ),
    (∀ (p : ℝ × ℝ), 
      (p.2 = (p.1 - 2)^2 ∧ p.1 - 5 = (p.2 + 1)^2) →
      (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    r^2 = 1.5 := by
  sorry

end intersection_points_on_circle_l1478_147884


namespace propositions_relationship_l1478_147818

theorem propositions_relationship (x : ℝ) :
  (∀ x, x < 3 → x < 5) ↔ (∀ x, x ≥ 5 → x ≥ 3) :=
by sorry

end propositions_relationship_l1478_147818


namespace no_such_function_exists_l1478_147866

theorem no_such_function_exists :
  ¬ ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, (f^[n.val] n = n + 1) := by
  sorry

end no_such_function_exists_l1478_147866


namespace points_on_opposite_sides_l1478_147857

def plane_equation (x y z : ℝ) : ℝ := x + 2*y + 3*z

def point1 : ℝ × ℝ × ℝ := (1, 2, -2)
def point2 : ℝ × ℝ × ℝ := (2, 1, -1)

theorem points_on_opposite_sides :
  (plane_equation point1.1 point1.2.1 point1.2.2) * (plane_equation point2.1 point2.2.1 point2.2.2) < 0 := by
  sorry

end points_on_opposite_sides_l1478_147857


namespace two_digit_number_property_l1478_147817

/-- 
For a two-digit number n where the unit's digit exceeds the 10's digit by 2, 
and n = 24, the product of n and the sum of its digits is 144.
-/
theorem two_digit_number_property : 
  ∀ (a b : ℕ), 
    (10 * a + b = 24) → 
    (b = a + 2) → 
    24 * (a + b) = 144 := by
  sorry

end two_digit_number_property_l1478_147817


namespace abs_value_of_root_l1478_147840

theorem abs_value_of_root (z : ℂ) : z^2 - 2*z + 2 = 0 → Complex.abs z = Real.sqrt 2 := by
  sorry

end abs_value_of_root_l1478_147840


namespace complex_fraction_equality_l1478_147845

theorem complex_fraction_equality : Complex.I * 4 / (Real.sqrt 3 + Complex.I) = 1 + Complex.I * Real.sqrt 3 := by
  sorry

end complex_fraction_equality_l1478_147845


namespace infinite_geometric_series_first_term_l1478_147896

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = (1 : ℝ) / 4)
  (h_S : S = 80)
  (h_sum : S = a / (1 - r))
  : a = 60 := by
  sorry

end infinite_geometric_series_first_term_l1478_147896


namespace max_y_coordinate_polar_curve_l1478_147820

/-- The maximum y-coordinate of a point on the graph of r = cos 2θ is √2/2 -/
theorem max_y_coordinate_polar_curve : 
  let r : ℝ → ℝ := λ θ => Real.cos (2 * θ)
  let y : ℝ → ℝ := λ θ => (r θ) * Real.sin θ
  ∃ (θ_max : ℝ), ∀ (θ : ℝ), y θ ≤ y θ_max ∧ y θ_max = Real.sqrt 2 / 2 := by
  sorry

end max_y_coordinate_polar_curve_l1478_147820


namespace is_circle_center_l1478_147874

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + 3 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, -2)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 2 :=
by sorry

end is_circle_center_l1478_147874


namespace arithmetic_sequence_properties_l1478_147800

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum of first n terms
  a4_eq : a 4 = -2
  S10_eq : S 10 = 25
  arith_seq : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- Main theorem about the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 3 * n - 14) ∧
  (seq.S 4 = -26 ∧ ∀ n : ℕ, seq.S n ≥ -26) := by
  sorry

end arithmetic_sequence_properties_l1478_147800


namespace sector_radius_proof_l1478_147859

/-- The area of a circular sector -/
def sectorArea : ℝ := 51.54285714285714

/-- The central angle of the sector in degrees -/
def centralAngle : ℝ := 41

/-- The radius of the circle -/
def radius : ℝ := 12

/-- Theorem stating that the given sector area and central angle result in the specified radius -/
theorem sector_radius_proof : 
  abs (sectorArea - (centralAngle / 360) * Real.pi * radius^2) < 1e-6 := by sorry

end sector_radius_proof_l1478_147859


namespace tomato_seeds_proof_l1478_147870

/-- The number of tomato seeds planted by Mike and Ted -/
def total_seeds (mike_morning mike_afternoon ted_morning ted_afternoon : ℕ) : ℕ :=
  mike_morning + mike_afternoon + ted_morning + ted_afternoon

theorem tomato_seeds_proof :
  ∀ (mike_morning mike_afternoon ted_morning ted_afternoon : ℕ),
    mike_morning = 50 →
    ted_morning = 2 * mike_morning →
    mike_afternoon = 60 →
    ted_afternoon = mike_afternoon - 20 →
    total_seeds mike_morning mike_afternoon ted_morning ted_afternoon = 250 := by
  sorry

end tomato_seeds_proof_l1478_147870


namespace ExistEvenOddComposition_l1478_147881

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the property of being an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of a function not being identically zero
def NotIdenticallyZero (f : ℝ → ℝ) : Prop := ∃ x, f x ≠ 0

-- State the theorem
theorem ExistEvenOddComposition :
  ∃ (p q : ℝ → ℝ), IsEven p ∧ IsOdd (p ∘ q) ∧ NotIdenticallyZero (p ∘ q) := by
  sorry

end ExistEvenOddComposition_l1478_147881


namespace sum_of_solutions_quadratic_l1478_147885

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 6*x + 5 = 2*x - 8) → 
  (∃ x₁ x₂ : ℝ, (x₁^2 - 6*x₁ + 5 = 2*x₁ - 8) ∧ 
                (x₂^2 - 6*x₂ + 5 = 2*x₂ - 8) ∧ 
                (x₁ + x₂ = 8)) :=
by sorry

end sum_of_solutions_quadratic_l1478_147885


namespace median_is_212_l1478_147867

/-- Represents the list where each integer n from 1 to 300 appears n times -/
def special_list : List ℕ := sorry

/-- The sum of all elements in the special list -/
def total_elements : ℕ := (300 * (300 + 1)) / 2

/-- The position of the median in the special list -/
def median_position : ℕ × ℕ := (total_elements / 2, total_elements / 2 + 1)

/-- Theorem stating that the median of the special list is 212 -/
theorem median_is_212 : 
  ∃ (median : ℕ), median = 212 ∧ 
  (∃ (l1 l2 : List ℕ), special_list = l1 ++ [median] ++ [median] ++ l2 ∧ 
   l1.length = median_position.1 - 1 ∧
   l2.length = special_list.length - median_position.2) :=
sorry

end median_is_212_l1478_147867


namespace smaller_rectangle_perimeter_l1478_147842

/-- Given a rectangle with dimensions a × b that is divided into a smaller rectangle 
    with dimensions c × b and two squares with side length c, 
    the perimeter of the smaller rectangle is 2(c + b). -/
theorem smaller_rectangle_perimeter 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : a = 3 * c) : 
  2 * (c + b) = 2 * c + 2 * b := by
sorry

end smaller_rectangle_perimeter_l1478_147842


namespace oil_in_engine_l1478_147822

theorem oil_in_engine (oil_per_cylinder : ℕ) (num_cylinders : ℕ) (additional_oil_needed : ℕ) :
  oil_per_cylinder = 8 →
  num_cylinders = 6 →
  additional_oil_needed = 32 →
  oil_per_cylinder * num_cylinders - additional_oil_needed = 16 :=
by
  sorry

end oil_in_engine_l1478_147822


namespace angle_trigonometry_l1478_147872

theorem angle_trigonometry (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α * Real.tan β = 13/7) (h4 : Real.sin (α - β) = Real.sqrt 5 / 3) :
  Real.cos (α - β) = 2/3 ∧ Real.cos (α + β) = -1/5 := by
  sorry

end angle_trigonometry_l1478_147872


namespace movie_ticket_distribution_l1478_147824

theorem movie_ticket_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (n.descFactorial k) = 720 := by
  sorry

end movie_ticket_distribution_l1478_147824


namespace arithmetic_evaluation_l1478_147835

theorem arithmetic_evaluation : (9 - 2) - (4 - 1) = 4 := by
  sorry

end arithmetic_evaluation_l1478_147835


namespace square_diff_minus_diff_squares_l1478_147899

theorem square_diff_minus_diff_squares (x y : ℝ) :
  (x - y)^2 - (x^2 - y^2) = (x - y)^2 - (x^2 - y^2) := by
  sorry

end square_diff_minus_diff_squares_l1478_147899


namespace lucy_groceries_l1478_147837

theorem lucy_groceries (cookies : ℕ) (cake : ℕ) : 
  cookies = 2 → cake = 12 → cookies + cake = 14 := by
  sorry

end lucy_groceries_l1478_147837


namespace grid_size_for_2017_colored_squares_l1478_147890

/-- Represents a square grid -/
structure SquareGrid where
  size : ℕ

/-- The number of colored squares on the two longest diagonals of a square grid -/
def coloredSquares (grid : SquareGrid) : ℕ := 2 * grid.size - 1

theorem grid_size_for_2017_colored_squares :
  ∃ (grid : SquareGrid), coloredSquares grid = 2017 ∧ grid.size = 1009 :=
sorry

end grid_size_for_2017_colored_squares_l1478_147890


namespace chicken_problem_l1478_147852

theorem chicken_problem (total chickens_colten : ℕ) 
  (h_total : total = 383)
  (h_colten : chickens_colten = 37) : 
  ∃ (chickens_skylar chickens_quentin : ℕ),
    chickens_quentin = 2 * chickens_skylar + 25 ∧
    chickens_quentin + chickens_skylar + chickens_colten = total ∧
    3 * chickens_colten - chickens_skylar = 4 :=
by sorry

end chicken_problem_l1478_147852


namespace coefficients_of_given_equation_l1478_147826

/-- Given a quadratic equation ax^2 + bx + c = 0, this function returns a tuple of its coefficients (a, b, c) -/
def quadraticCoefficients (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

/-- The quadratic equation 5x^2 + 2x - 1 = 0 -/
def givenEquation : ℝ × ℝ × ℝ := (5, 2, -1)

theorem coefficients_of_given_equation :
  quadraticCoefficients 5 2 (-1) = givenEquation :=
by sorry

end coefficients_of_given_equation_l1478_147826


namespace problem_1_problem_2_l1478_147825

-- Problem 1
theorem problem_1 : 
  (Real.sqrt 7 - Real.sqrt 13) * (Real.sqrt 7 + Real.sqrt 13) + 
  (Real.sqrt 3 + 1)^2 - (Real.sqrt 6 * Real.sqrt 3) / Real.sqrt 2 + 
  |-(Real.sqrt 3)| = -3 + 3 * Real.sqrt 3 := by sorry

-- Problem 2
theorem problem_2 {a : ℝ} (ha : a < 0) : 
  Real.sqrt (4 - (a + 1/a)^2) - Real.sqrt (4 + (a - 1/a)^2) = -2 := by sorry

end problem_1_problem_2_l1478_147825


namespace five_digit_diff_last_two_count_l1478_147877

/-- The number of five-digit numbers -/
def total_five_digit_numbers : ℕ := 90000

/-- The number of five-digit numbers where the last two digits are the same -/
def five_digit_same_last_two : ℕ := 9000

/-- The number of five-digit numbers where at least the last two digits are different -/
def five_digit_diff_last_two : ℕ := total_five_digit_numbers - five_digit_same_last_two

theorem five_digit_diff_last_two_count : five_digit_diff_last_two = 81000 := by
  sorry

end five_digit_diff_last_two_count_l1478_147877


namespace z_in_fourth_quadrant_l1478_147880

-- Define the complex number z
def z : ℂ := (1 - Complex.I) * (3 + Complex.I)

-- Theorem stating that z is in the fourth quadrant
theorem z_in_fourth_quadrant :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end z_in_fourth_quadrant_l1478_147880


namespace complex_equation_sum_l1478_147889

theorem complex_equation_sum (a b : ℝ) : 
  (a - 2 * Complex.I) * Complex.I = b - Complex.I → a + b = 1 := by
  sorry

end complex_equation_sum_l1478_147889


namespace geometric_sequence_first_term_l1478_147856

def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_first_term
  (a : ℕ → ℚ)
  (h_geometric : is_geometric_sequence a)
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = -54) :
  a 1 = -2/3 := by
  sorry

end geometric_sequence_first_term_l1478_147856


namespace fence_cost_square_plot_l1478_147802

/-- The cost of building a fence around a square plot -/
theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 289 → price_per_foot = 54 → cost = 4 * Real.sqrt area * price_per_foot → cost = 3672 := by
  sorry

end fence_cost_square_plot_l1478_147802


namespace difference_greater_than_one_l1478_147888

theorem difference_greater_than_one : 19^91 - (999991:ℕ)^19 ≥ 1 := by
  sorry

end difference_greater_than_one_l1478_147888


namespace circle_area_with_diameter_10_l1478_147893

/-- The area of a circle with diameter 10 meters is 25π square meters. -/
theorem circle_area_with_diameter_10 :
  let d : ℝ := 10
  let r : ℝ := d / 2
  let area : ℝ := π * r^2
  area = 25 * π :=
by sorry

end circle_area_with_diameter_10_l1478_147893


namespace inequality_proof_l1478_147828

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a * b + b * c + c * a)^2 ≥ 3 * a * b * c * (a + b + c) := by
  sorry

end inequality_proof_l1478_147828

import Mathlib

namespace intersection_of_A_and_B_l3127_312769

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-2, -1, 0, 1} := by sorry

end intersection_of_A_and_B_l3127_312769


namespace fruit_shop_costs_and_profit_l3127_312779

/-- Represents the fruit shop's purchases and sales --/
structure FruitShop where
  first_purchase_cost : ℝ
  first_purchase_price : ℝ
  second_purchase_cost : ℝ
  second_purchase_quantity_increase : ℝ
  second_sale_price : ℝ
  second_sale_quantity : ℝ
  second_sale_discount : ℝ

/-- Calculates the cost per kg and profit for the fruit shop --/
def calculate_costs_and_profit (shop : FruitShop) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating the correct cost per kg and profit given the shop's conditions --/
theorem fruit_shop_costs_and_profit (shop : FruitShop) 
  (h1 : shop.first_purchase_cost = 1200)
  (h2 : shop.first_purchase_price = 8)
  (h3 : shop.second_purchase_cost = 1452)
  (h4 : shop.second_purchase_quantity_increase = 20)
  (h5 : shop.second_sale_price = 9)
  (h6 : shop.second_sale_quantity = 100)
  (h7 : shop.second_sale_discount = 0.5) :
  let (first_cost, second_cost, profit) := calculate_costs_and_profit shop
  first_cost = 6 ∧ second_cost = 6.6 ∧ profit = 388 := by sorry

end fruit_shop_costs_and_profit_l3127_312779


namespace complex_equation_solution_l3127_312736

theorem complex_equation_solution (a : ℂ) :
  a / (1 - Complex.I) = (1 + Complex.I) / Complex.I → a = -2 * Complex.I := by
  sorry

end complex_equation_solution_l3127_312736


namespace specific_trapezoid_area_l3127_312718

/-- Represents an isosceles trapezoid with given measurements -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculates the area of the isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the area of the specific trapezoid is approximately 318.93 -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 20,
    diagonal_length := 25,
    longer_base := 30
  }
  abs (trapezoid_area t - 318.93) < 0.01 := by
  sorry

end specific_trapezoid_area_l3127_312718


namespace prob_no_green_3x3_value_main_result_l3127_312733

/-- Represents a 4x4 grid of colored squares -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a 3x3 subgrid starting at (i, j) is all green -/
def has_green_3x3 (g : Grid) (i j : Fin 2) : Prop :=
  ∀ x y, g (i + x) (j + y)

/-- The probability of not having a 3x3 green square in a 4x4 grid -/
def prob_no_green_3x3 : ℚ :=
  1 - (419 : ℚ) / 2^16

theorem prob_no_green_3x3_value :
  prob_no_green_3x3 = 65117 / 65536 :=
sorry

/-- The sum of the numerator and denominator of the probability -/
def sum_num_denom : ℕ := 65117 + 65536

theorem main_result :
  sum_num_denom = 130653 :=
sorry

end prob_no_green_3x3_value_main_result_l3127_312733


namespace factorial_divisibility_l3127_312739

theorem factorial_divisibility (p : ℕ) (h : Prime p) :
  ∃ k : ℕ, (p^2).factorial = k * (p.factorial ^ (p + 1)) :=
sorry

end factorial_divisibility_l3127_312739


namespace probability_at_least_one_event_l3127_312753

theorem probability_at_least_one_event (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/2) (h2 : p2 = 1/3) (h3 : p3 = 1/4) : 
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 3/4 := by
  sorry

end probability_at_least_one_event_l3127_312753


namespace car_speed_proof_l3127_312764

/-- Proves that a car's speed is 600 km/h given the problem conditions -/
theorem car_speed_proof (v : ℝ) : v > 0 →
  (1 / v - 1 / 900) * 3600 = 2 ↔ v = 600 := by
  sorry

#check car_speed_proof

end car_speed_proof_l3127_312764


namespace farm_ratio_l3127_312788

theorem farm_ratio (H C : ℕ) 
  (h1 : (H - 15 : ℚ) / (C + 15 : ℚ) = 17 / 7)
  (h2 : H - 15 = C + 15 + 50) :
  H / C = 5 / 1 := by
  sorry

end farm_ratio_l3127_312788


namespace dogs_neither_long_furred_nor_brown_l3127_312777

/-- Prove that the number of dogs that are neither long-furred nor brown is 8 -/
theorem dogs_neither_long_furred_nor_brown
  (total_dogs : ℕ)
  (long_furred_dogs : ℕ)
  (brown_dogs : ℕ)
  (long_furred_brown_dogs : ℕ)
  (h1 : total_dogs = 45)
  (h2 : long_furred_dogs = 26)
  (h3 : brown_dogs = 22)
  (h4 : long_furred_brown_dogs = 11) :
  total_dogs - (long_furred_dogs + brown_dogs - long_furred_brown_dogs) = 8 := by
  sorry

#check dogs_neither_long_furred_nor_brown

end dogs_neither_long_furred_nor_brown_l3127_312777


namespace inscribed_hexagon_area_l3127_312765

/-- The area of a regular hexagon inscribed in a circle with area 16π -/
theorem inscribed_hexagon_area : 
  ∀ (circle_area : ℝ) (hexagon_area : ℝ),
  circle_area = 16 * Real.pi →
  hexagon_area = (6 * Real.sqrt 3 * circle_area) / (2 * Real.pi) →
  hexagon_area = 24 * Real.sqrt 3 := by
sorry

end inscribed_hexagon_area_l3127_312765


namespace bus_people_count_l3127_312771

/-- Represents the number of people who got off the bus -/
def people_off : ℕ := 47

/-- Represents the number of people remaining on the bus -/
def people_remaining : ℕ := 43

/-- Represents the total number of people on the bus before -/
def total_people : ℕ := people_off + people_remaining

theorem bus_people_count : total_people = 90 := by
  sorry

end bus_people_count_l3127_312771


namespace first_player_wins_6x8_l3127_312732

/-- Represents a chocolate bar game -/
structure ChocolateGame where
  rows : ℕ
  cols : ℕ

/-- Calculates the total number of moves in a chocolate bar game -/
def totalMoves (game : ChocolateGame) : ℕ :=
  game.rows * game.cols - 1

/-- Determines if the first player wins the game -/
def firstPlayerWins (game : ChocolateGame) : Prop :=
  Odd (totalMoves game)

/-- Theorem: The first player wins in a 6x8 chocolate bar game -/
theorem first_player_wins_6x8 :
  firstPlayerWins ⟨6, 8⟩ := by sorry

end first_player_wins_6x8_l3127_312732


namespace soap_brand_survey_l3127_312717

theorem soap_brand_survey (total : ℕ) (neither : ℕ) (only_w : ℕ) :
  total = 200 →
  neither = 80 →
  only_w = 60 →
  ∃ (both : ℕ),
    both * 4 = total - neither - only_w ∧
    both = 15 :=
by sorry

end soap_brand_survey_l3127_312717


namespace scale_and_rotate_complex_l3127_312703

/-- Represents a complex number rotation by 270° clockwise -/
def rotate270Clockwise (z : ℂ) : ℂ := Complex.I * z

/-- Proves that scaling -8 - 4i by 2 and then rotating 270° clockwise results in 8 - 16i -/
theorem scale_and_rotate_complex : 
  let z : ℂ := -8 - 4 * Complex.I
  let scaled : ℂ := 2 * z
  rotate270Clockwise scaled = 8 - 16 * Complex.I := by sorry

end scale_and_rotate_complex_l3127_312703


namespace gift_spending_calculation_l3127_312720

/-- Given a total amount spent and an amount spent on giftwrapping and other expenses,
    calculate the amount spent on gifts. -/
def amount_spent_on_gifts (total_amount : ℚ) (giftwrapping_amount : ℚ) : ℚ :=
  total_amount - giftwrapping_amount

/-- Prove that the amount spent on gifts is $561.00, given the total amount
    spent is $700.00 and the amount spent on giftwrapping is $139.00. -/
theorem gift_spending_calculation :
  amount_spent_on_gifts 700 139 = 561 := by
  sorry

end gift_spending_calculation_l3127_312720


namespace equation_solution_l3127_312704

theorem equation_solution :
  ∃ x : ℝ, (((3 * x - 1) / (x + 4)) > 0) ∧ 
            (((x + 4) / (3 * x - 1)) > 0) ∧
            (Real.sqrt ((3 * x - 1) / (x + 4)) + 3 - 4 * Real.sqrt ((x + 4) / (3 * x - 1)) = 0) ∧
            (x = 5 / 2) := by
  sorry

end equation_solution_l3127_312704


namespace inlet_fill_rate_l3127_312754

/-- The rate at which the inlet pipe fills the tank, given the tank's capacity,
    leak emptying time, and combined emptying time with inlet open. -/
theorem inlet_fill_rate (capacity : ℝ) (leak_empty_time : ℝ) (combined_empty_time : ℝ) :
  capacity = 5760 →
  leak_empty_time = 6 →
  combined_empty_time = 8 →
  (capacity / leak_empty_time) - (capacity / combined_empty_time) = 240 := by
  sorry

end inlet_fill_rate_l3127_312754


namespace absent_student_percentage_l3127_312735

theorem absent_student_percentage
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (absent_boys_fraction : ℚ)
  (absent_girls_fraction : ℚ)
  (h1 : total_students = 100)
  (h2 : boys = 50)
  (h3 : girls = 50)
  (h4 : boys + girls = total_students)
  (h5 : absent_boys_fraction = 1 / 5)
  (h6 : absent_girls_fraction = 1 / 4) :
  (↑boys * absent_boys_fraction + ↑girls * absent_girls_fraction) / ↑total_students = 225 / 1000 := by
  sorry

#check absent_student_percentage

end absent_student_percentage_l3127_312735


namespace jenny_ate_65_chocolates_l3127_312724

/-- The number of chocolate squares Mike ate -/
def mike_chocolates : ℕ := 20

/-- The number of chocolate squares Jenny ate -/
def jenny_chocolates : ℕ := 3 * mike_chocolates + 5

theorem jenny_ate_65_chocolates : jenny_chocolates = 65 := by
  sorry

end jenny_ate_65_chocolates_l3127_312724


namespace vikki_earnings_insurance_deduction_l3127_312749

/-- Vikki's weekly earnings and deductions -/
def weekly_earnings_problem (hours_worked : ℕ) (hourly_rate : ℚ) (tax_rate : ℚ) 
  (union_dues : ℚ) (take_home_pay : ℚ) : Prop :=
  let gross_earnings := hours_worked * hourly_rate
  let tax_deduction := gross_earnings * tax_rate
  let after_tax_and_dues := gross_earnings - tax_deduction - union_dues
  let insurance_deduction := after_tax_and_dues - take_home_pay
  let insurance_percentage := insurance_deduction / gross_earnings * 100
  insurance_percentage = 5

theorem vikki_earnings_insurance_deduction :
  weekly_earnings_problem 42 10 (1/5) 5 310 :=
sorry

end vikki_earnings_insurance_deduction_l3127_312749


namespace gcd_45_75_l3127_312700

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l3127_312700


namespace greg_lunch_payment_l3127_312768

/-- Calculates the total amount paid for a meal including tax and tip -/
def total_amount_paid (cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  cost + (cost * tax_rate) + (cost * tip_rate)

/-- Theorem stating that Greg paid $110 for his lunch -/
theorem greg_lunch_payment :
  let cost : ℝ := 100
  let tax_rate : ℝ := 0.04
  let tip_rate : ℝ := 0.06
  total_amount_paid cost tax_rate tip_rate = 110 := by
  sorry

end greg_lunch_payment_l3127_312768


namespace square_sum_value_l3127_312731

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end square_sum_value_l3127_312731


namespace extended_equilateral_area_ratio_l3127_312756

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Extends a line segment by a factor -/
def extendSegment (A B : Point) (factor : ℝ) : Point := sorry

theorem extended_equilateral_area_ratio 
  (P Q R : Point) 
  (t : Triangle)
  (h_equilateral : isEquilateral t)
  (h_t : t = Triangle.mk P Q R)
  (Q' : Point)
  (h_Q' : Q' = extendSegment P Q 3)
  (R' : Point)
  (h_R' : R' = extendSegment Q R 3)
  (P' : Point)
  (h_P' : P' = extendSegment R P 3)
  (t_extended : Triangle)
  (h_t_extended : t_extended = Triangle.mk P' Q' R') :
  triangleArea t_extended / triangleArea t = 9 := by sorry

end extended_equilateral_area_ratio_l3127_312756


namespace min_four_dollar_frisbees_l3127_312723

/-- Given a total of 64 frisbees sold at either $3 or $4 each, with total receipts of $196,
    the minimum number of $4 frisbees sold is 4. -/
theorem min_four_dollar_frisbees :
  ∀ (x y : ℕ),
    x + y = 64 →
    3 * x + 4 * y = 196 →
    y ≥ 4 ∧ ∃ (z : ℕ), z + 4 = 64 ∧ 3 * z + 4 * 4 = 196 :=
by sorry

end min_four_dollar_frisbees_l3127_312723


namespace sqrt_equality_implies_specific_values_l3127_312793

theorem sqrt_equality_implies_specific_values :
  ∀ a b : ℕ+,
  a < b →
  Real.sqrt (4 + Real.sqrt (76 + 40 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b →
  a = 1 ∧ b = 10 := by
sorry

end sqrt_equality_implies_specific_values_l3127_312793


namespace sin_seven_pi_sixths_l3127_312745

theorem sin_seven_pi_sixths : Real.sin (7 * Real.pi / 6) = -(1 / 2) := by
  sorry

end sin_seven_pi_sixths_l3127_312745


namespace quadratic_one_zero_l3127_312711

/-- If a quadratic function f(x) = mx^2 - 2x + 3 has only one zero, then m = 0 or m = 1/3 -/
theorem quadratic_one_zero (m : ℝ) : 
  (∃! x, m * x^2 - 2 * x + 3 = 0) → (m = 0 ∨ m = 1/3) := by
sorry

end quadratic_one_zero_l3127_312711


namespace parallel_line_equation_l3127_312799

/-- A line in the plane is represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Check if a point lies on a line given by an equation ax + by + c = 0 -/
def pointOnLine (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

/-- Two lines are parallel if they have the same slope -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.slope = l₂.slope

theorem parallel_line_equation (p : ℝ × ℝ) :
  let l₁ : Line := { slope := 2, point := (0, 0) }  -- y = 2x
  let l₂ : Line := { slope := 2, point := p }       -- parallel line through p
  parallel l₁ l₂ →
  p = (1, -2) →
  pointOnLine 2 (-1) (-4) p.1 p.2 :=
by
  sorry

#check parallel_line_equation

end parallel_line_equation_l3127_312799


namespace bird_families_difference_l3127_312775

theorem bird_families_difference (total : ℕ) (flew_away : ℕ) 
  (h1 : total = 87) (h2 : flew_away = 7) : 
  total - flew_away - flew_away = 73 := by
  sorry

end bird_families_difference_l3127_312775


namespace min_xy_value_l3127_312740

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 3 / (2 + x) + 3 / (2 + y) = 1) : 
  ∀ z, z = x * y → z ≥ 16 := by
  sorry

end min_xy_value_l3127_312740


namespace chess_tournament_red_pairs_l3127_312789

/-- Represents the number of pairs in a chess tournament where both players wear red hats. -/
def red_red_pairs (green_players : ℕ) (red_players : ℕ) (total_pairs : ℕ) (green_green_pairs : ℕ) : ℕ :=
  (red_players - (total_pairs * 2 - green_players - red_players)) / 2

/-- Theorem stating that in the given chess tournament scenario, there are 27 pairs where both players wear red hats. -/
theorem chess_tournament_red_pairs : 
  red_red_pairs 64 68 66 25 = 27 := by
  sorry

end chess_tournament_red_pairs_l3127_312789


namespace time_between_flashes_l3127_312706

/-- Represents the number of flashes in 3/4 of an hour -/
def flashes_per_three_quarters_hour : ℕ := 300

/-- Represents 3/4 of an hour in seconds -/
def three_quarters_hour_in_seconds : ℕ := 45 * 60

/-- Theorem stating that the time between flashes is 9 seconds -/
theorem time_between_flashes :
  three_quarters_hour_in_seconds / flashes_per_three_quarters_hour = 9 := by
  sorry

end time_between_flashes_l3127_312706


namespace carpet_length_proof_l3127_312744

theorem carpet_length_proof (carpet_width : ℝ) (room_area : ℝ) (coverage_percentage : ℝ) :
  carpet_width = 4 →
  room_area = 180 →
  coverage_percentage = 0.20 →
  let carpet_area := room_area * coverage_percentage
  let carpet_length := carpet_area / carpet_width
  carpet_length = 9 := by sorry

end carpet_length_proof_l3127_312744


namespace linda_babysitting_hours_l3127_312713

/-- Linda's babysitting problem -/
theorem linda_babysitting_hours (babysitting_rate : ℚ) (application_fee : ℚ) (num_colleges : ℕ) :
  babysitting_rate = 10 →
  application_fee = 25 →
  num_colleges = 6 →
  (num_colleges * application_fee) / babysitting_rate = 15 :=
by sorry

end linda_babysitting_hours_l3127_312713


namespace arcade_spending_l3127_312758

theorem arcade_spending (allowance : ℚ) (arcade_fraction : ℚ) (remaining : ℚ) :
  allowance = 2.25 →
  remaining = 0.60 →
  remaining = (1 - arcade_fraction) * allowance - (1/3) * ((1 - arcade_fraction) * allowance) →
  arcade_fraction = 3/5 := by
  sorry

end arcade_spending_l3127_312758


namespace poultry_farm_hens_l3127_312787

theorem poultry_farm_hens (total_chickens : ℕ) (hen_rooster_ratio : ℚ) (chicks_per_hen : ℕ) : 
  total_chickens = 76 → 
  hen_rooster_ratio = 3 → 
  chicks_per_hen = 5 → 
  ∃ (num_hens : ℕ), num_hens = 12 ∧ 
    num_hens + (num_hens : ℚ) / hen_rooster_ratio + (num_hens * chicks_per_hen) = total_chickens := by
  sorry

end poultry_farm_hens_l3127_312787


namespace addition_problems_l3127_312710

theorem addition_problems :
  (15 + (-8) + 4 + (-10) = 1) ∧
  ((-2) + (7 + 1/2) + 4.5 = 10) :=
by
  sorry

end addition_problems_l3127_312710


namespace method_a_cheaper_for_18_hours_l3127_312778

/-- Calculates the cost of internet usage for Method A (Pay-per-use) -/
def costMethodA (hours : ℝ) : ℝ := 3 * hours + 1.2 * hours

/-- Calculates the cost of internet usage for Method B (Monthly subscription) -/
def costMethodB (hours : ℝ) : ℝ := 60 + 1.2 * hours

/-- Theorem stating that Method A is cheaper than Method B for 18 hours of usage -/
theorem method_a_cheaper_for_18_hours :
  costMethodA 18 < costMethodB 18 :=
sorry

end method_a_cheaper_for_18_hours_l3127_312778


namespace andrew_appointments_l3127_312770

/-- Calculates the number of 3-hour appointments given total work hours, permits stamped per hour, and total permits stamped. -/
def appointments (total_hours : ℕ) (permits_per_hour : ℕ) (total_permits : ℕ) : ℕ :=
  (total_hours - (total_permits / permits_per_hour)) / 3

/-- Theorem stating that given the problem conditions, Andrew has 2 appointments. -/
theorem andrew_appointments : appointments 8 50 100 = 2 := by
  sorry

end andrew_appointments_l3127_312770


namespace certain_number_proof_l3127_312752

theorem certain_number_proof (x N : ℝ) 
  (h1 : 3 * x = (N - x) + 14) 
  (h2 : x = 10) : 
  N = 26 := by
sorry

end certain_number_proof_l3127_312752


namespace count_representations_l3127_312702

/-- The number of ways to represent 5040 in the given form -/
def M : ℕ :=
  (Finset.range 100).sum (fun b₃ =>
    (Finset.range 100).sum (fun b₂ =>
      (Finset.range 100).sum (fun b₁ =>
        (Finset.range 100).sum (fun b₀ =>
          if b₃ * 10^3 + b₂ * 10^2 + b₁ * 10 + b₀ = 5040 then 1 else 0))))

theorem count_representations : M = 504 := by
  sorry

end count_representations_l3127_312702


namespace fraction_decomposition_l3127_312761

-- Define the polynomials
def p (x : ℝ) : ℝ := 6 * x - 15
def q (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 - 5 * x + 6
def r (x : ℝ) : ℝ := x - 1
def s (x : ℝ) : ℝ := 3 * x^2 - x - 6

-- Define the equality condition
def equality_condition (A B : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 1 ∧ s x ≠ 0 → p x / q x = A x / r x + B x / s x

-- Theorem statement
theorem fraction_decomposition :
  ∀ A B, equality_condition A B →
    (∀ x, A x = 0) ∧ (∀ x, B x = 6 * x - 15) :=
sorry

end fraction_decomposition_l3127_312761


namespace bicycle_price_calculation_l3127_312755

theorem bicycle_price_calculation (initial_price : ℝ) : 
  let first_sale_price := initial_price * 1.20
  let final_price := first_sale_price * 1.25
  final_price = 225 → initial_price = 150 := by
sorry

end bicycle_price_calculation_l3127_312755


namespace A_intersect_B_equals_zero_two_l3127_312748

def A : Set ℤ := {-2, 0, 2}

-- Define the absolute value function
def f (x : ℤ) : ℤ := abs x

-- Define B as the image of A under f
def B : Set ℤ := f '' A

-- State the theorem
theorem A_intersect_B_equals_zero_two : A ∩ B = {0, 2} := by sorry

end A_intersect_B_equals_zero_two_l3127_312748


namespace water_volume_ratio_in_cone_l3127_312767

/-- Theorem: Volume ratio of water in a cone filled to 2/3 height -/
theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let water_height : ℝ := 2 / 3 * h
  let water_radius : ℝ := 2 / 3 * r
  let cone_volume : ℝ := (1 / 3) * π * r^2 * h
  let water_volume : ℝ := (1 / 3) * π * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 := by
  sorry

end water_volume_ratio_in_cone_l3127_312767


namespace defeat_crab_ways_l3127_312728

/-- Represents the number of claws on the giant enemy crab -/
def num_claws : ℕ := 2

/-- Represents the number of legs on the giant enemy crab -/
def num_legs : ℕ := 6

/-- Represents the minimum number of legs that must be cut before claws can be cut -/
def min_legs_before_claws : ℕ := 3

/-- The number of ways to defeat the giant enemy crab -/
def ways_to_defeat_crab : ℕ := num_legs.factorial * num_claws.factorial * (Nat.choose (num_legs + num_claws - min_legs_before_claws) num_claws)

/-- Theorem stating the number of ways to defeat the giant enemy crab -/
theorem defeat_crab_ways : ways_to_defeat_crab = 14400 := by
  sorry

end defeat_crab_ways_l3127_312728


namespace greatest_integer_and_y_value_l3127_312725

theorem greatest_integer_and_y_value :
  (∃ x : ℤ, (∀ z : ℤ, 7 - 5*z > 22 → z ≤ x) ∧ 7 - 5*x > 22 ∧ x = -4) ∧
  (let x := -4; 2*x + 3 = -5) :=
sorry

end greatest_integer_and_y_value_l3127_312725


namespace shelf_theorem_l3127_312716

/-- Given two shelves, with the second twice as long as the first, and book thicknesses,
    prove the relation between the number of books on each shelf. -/
theorem shelf_theorem (A' H' S' M' E' : ℕ) (x y : ℝ) : 
  A' ≠ H' ∧ A' ≠ S' ∧ A' ≠ M' ∧ A' ≠ E' ∧ 
  H' ≠ S' ∧ H' ≠ M' ∧ H' ≠ E' ∧ 
  S' ≠ M' ∧ S' ≠ E' ∧ 
  M' ≠ E' ∧
  A' > 0 ∧ H' > 0 ∧ S' > 0 ∧ M' > 0 ∧ E' > 0 ∧
  y > x ∧ 
  A' * x + H' * y = S' * x + M' * y ∧ 
  E' * x = 2 * (A' * x + H' * y) →
  E' = (2 * A' * M' - 2 * S' * H') / (M' - H') :=
by sorry

end shelf_theorem_l3127_312716


namespace bryden_receives_correct_amount_l3127_312719

/-- The face value of a state quarter in dollars -/
def face_value : ℚ := 1/4

/-- The number of quarters Bryden is selling -/
def num_quarters : ℕ := 6

/-- The percentage of face value offered by the collector -/
def offer_percentage : ℕ := 1500

/-- The amount Bryden will receive in dollars -/
def amount_received : ℚ := (offer_percentage : ℚ) / 100 * face_value * num_quarters

theorem bryden_receives_correct_amount : amount_received = 45/2 := by
  sorry

end bryden_receives_correct_amount_l3127_312719


namespace pen_sales_problem_l3127_312798

theorem pen_sales_problem (d : ℕ) : 
  (96 + 44 * d) / (d + 1) = 48 → d = 12 := by
  sorry

end pen_sales_problem_l3127_312798


namespace blue_flower_percentage_l3127_312783

/-- Given a total of 10 flowers, with 4 red and 2 white flowers,
    prove that 40% of the flowers are blue. -/
theorem blue_flower_percentage
  (total : ℕ)
  (red : ℕ)
  (white : ℕ)
  (h_total : total = 10)
  (h_red : red = 4)
  (h_white : white = 2) :
  (total - red - white : ℚ) / total * 100 = 40 := by
  sorry

#check blue_flower_percentage

end blue_flower_percentage_l3127_312783


namespace phyllis_garden_problem_l3127_312746

/-- The number of plants in Phyllis's first garden -/
def plants_in_first_garden : ℕ := 20

/-- The number of plants in Phyllis's second garden -/
def plants_in_second_garden : ℕ := 15

/-- The fraction of tomato plants in the first garden -/
def tomato_fraction_first : ℚ := 1/10

/-- The fraction of tomato plants in the second garden -/
def tomato_fraction_second : ℚ := 1/3

/-- The fraction of tomato plants in both gardens combined -/
def total_tomato_fraction : ℚ := 1/5

theorem phyllis_garden_problem :
  (plants_in_first_garden : ℚ) * tomato_fraction_first +
  (plants_in_second_garden : ℚ) * tomato_fraction_second =
  ((plants_in_first_garden + plants_in_second_garden) : ℚ) * total_tomato_fraction :=
by sorry

end phyllis_garden_problem_l3127_312746


namespace consecutive_pages_sum_l3127_312751

theorem consecutive_pages_sum (x : ℕ) (h : x + (x + 1) = 185) : x = 92 := by
  sorry

end consecutive_pages_sum_l3127_312751


namespace f_max_value_l3127_312721

/-- The quadratic function f(y) = -9y^2 + 15y + 3 -/
def f (y : ℝ) := -9 * y^2 + 15 * y + 3

/-- The maximum value of f(y) is 6.25 -/
theorem f_max_value : ∃ (y : ℝ), f y = 6.25 ∧ ∀ (z : ℝ), f z ≤ 6.25 := by
  sorry

end f_max_value_l3127_312721


namespace reflection_of_P_across_y_axis_l3127_312796

/-- Given a point P in the Cartesian coordinate system, this function returns its reflection across the y-axis. -/
def reflect_across_y_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (-P.1, P.2)

/-- The original point P in the Cartesian coordinate system. -/
def P : ℝ × ℝ := (2, 1)

/-- Theorem: The coordinates of P(2,1) with respect to the y-axis are (-2,1). -/
theorem reflection_of_P_across_y_axis :
  reflect_across_y_axis P = (-2, 1) := by sorry

end reflection_of_P_across_y_axis_l3127_312796


namespace negation_of_existence_negation_of_quadratic_inequality_l3127_312738

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l3127_312738


namespace parallel_perpendicular_lines_l3127_312784

/-- Given a point P and a line l, prove the equations of parallel and perpendicular lines through P --/
theorem parallel_perpendicular_lines 
  (P : ℝ × ℝ) 
  (l : ℝ → ℝ → Prop) 
  (hl : l = fun x y => 3 * x - y - 7 = 0) 
  (hP : P = (2, 1)) :
  let parallel_line := fun x y => 3 * x - y - 5 = 0
  let perpendicular_line := fun x y => x - 3 * y + 1 = 0
  (∀ x y, parallel_line x y ↔ (3 * x - y = 3 * P.1 - P.2)) ∧ 
  (parallel_line P.1 P.2) ∧
  (∀ x y, perpendicular_line x y ↔ (x - 3 * y = P.1 - 3 * P.2)) ∧ 
  (perpendicular_line P.1 P.2) ∧
  (∀ x₁ y₁ x₂ y₂, l x₁ y₁ → l x₂ y₂ → x₁ ≠ x₂ → (y₁ - y₂) / (x₁ - x₂) = 3) ∧
  (∀ x₁ y₁ x₂ y₂, perpendicular_line x₁ y₁ → perpendicular_line x₂ y₂ → x₁ ≠ x₂ → 
    (y₁ - y₂) / (x₁ - x₂) = -1/3) := by
  sorry

end parallel_perpendicular_lines_l3127_312784


namespace flame_shooting_time_l3127_312757

theorem flame_shooting_time (firing_interval : ℝ) (flame_duration : ℝ) (total_time : ℝ) :
  firing_interval = 15 →
  flame_duration = 5 →
  total_time = 60 →
  (total_time / firing_interval) * flame_duration = 20 := by
  sorry

end flame_shooting_time_l3127_312757


namespace tangent_lines_through_M_line_intersects_circle_l3127_312782

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define point M
def point_M : ℝ × ℝ := (3, 1)

-- Define the line ax - y + 3 = 0
def line (a x y : ℝ) : Prop := a * x - y + 3 = 0

-- Theorem for part (I)
theorem tangent_lines_through_M :
  ∃ (k : ℝ), 
    (∀ x y : ℝ, (x = 3 ∨ 3 * x - 4 * y - 5 = 0) → 
      (circle_C x y ∧ (x = point_M.1 ∧ y = point_M.2 ∨ 
       (y - point_M.2) = k * (x - point_M.1)))) :=
sorry

-- Theorem for part (II)
theorem line_intersects_circle :
  ∀ a : ℝ, ∃ x y : ℝ, line a x y ∧ circle_C x y :=
sorry

end tangent_lines_through_M_line_intersects_circle_l3127_312782


namespace parallel_vectors_tangent_l3127_312737

theorem parallel_vectors_tangent (θ : ℝ) (a b : ℝ × ℝ) : 
  a = (2, Real.sin θ) → 
  b = (1, Real.cos θ) → 
  (∃ (k : ℝ), a = k • b) → 
  Real.tan θ = 2 := by
sorry

end parallel_vectors_tangent_l3127_312737


namespace expected_value_is_one_l3127_312709

/-- Represents the possible outcomes of rolling the die -/
inductive DieOutcome
| one
| two
| three
| four
| five
| six

/-- The probability of rolling each outcome -/
def prob (outcome : DieOutcome) : ℚ :=
  match outcome with
  | .one => 1/4
  | .two => 1/4
  | .three => 1/6
  | .four => 1/6
  | .five => 1/12
  | .six => 1/12

/-- The earnings associated with each outcome -/
def earnings (outcome : DieOutcome) : ℤ :=
  match outcome with
  | .one | .two => 4
  | .three | .four => -3
  | .five | .six => 0

/-- The expected value of earnings from one roll of the die -/
def expectedValue : ℚ :=
  (prob .one * earnings .one) +
  (prob .two * earnings .two) +
  (prob .three * earnings .three) +
  (prob .four * earnings .four) +
  (prob .five * earnings .five) +
  (prob .six * earnings .six)

/-- Theorem stating that the expected value of earnings is 1 -/
theorem expected_value_is_one : expectedValue = 1 := by
  sorry

end expected_value_is_one_l3127_312709


namespace range_of_f_l3127_312766

def f (x : ℤ) : ℤ := x^2 + 2*x

def domain : Set ℤ := {x | -2 ≤ x ∧ x ≤ 1}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end range_of_f_l3127_312766


namespace circle_properties_l3127_312715

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*y + 5*m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x - y + 1 = 0

-- Theorem statement
theorem circle_properties :
  -- Part 1: Range of m
  (∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → (m < 1 ∨ m > 4)) ∧
  -- Part 2: Length of chord when m = -2
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ (-2) ∧
    circle_equation x₂ y₂ (-2) ∧
    line_equation x₁ y₁ ∧
    line_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 26) :=
by
  sorry

end circle_properties_l3127_312715


namespace nine_points_chords_l3127_312791

/-- The number of different chords that can be drawn by connecting two of n points on a circle. -/
def number_of_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of different chords that can be drawn by connecting two of nine points
    on the circumference of a circle is equal to 36. -/
theorem nine_points_chords :
  number_of_chords 9 = 36 := by
  sorry

end nine_points_chords_l3127_312791


namespace price_decrease_percentage_l3127_312773

theorem price_decrease_percentage (original_price new_price : ℚ) 
  (h1 : original_price = 1400)
  (h2 : new_price = 1064) :
  (original_price - new_price) / original_price * 100 = 24 := by
  sorry

end price_decrease_percentage_l3127_312773


namespace kelly_games_left_l3127_312759

theorem kelly_games_left (initial_games give_away_games : ℕ) 
  (h1 : initial_games = 257)
  (h2 : give_away_games = 138) :
  initial_games - give_away_games = 119 :=
by sorry

end kelly_games_left_l3127_312759


namespace custom_mult_equation_solutions_l3127_312795

/-- Custom multiplication operation for real numbers -/
def custom_mult (a b : ℝ) : ℝ := a * (a + b) + b

/-- Theorem stating the solutions of the equation -/
theorem custom_mult_equation_solutions :
  ∃ (a : ℝ), custom_mult a 2.5 = 28.5 ∧ (a = 4 ∨ a = -13/2) := by
  sorry

end custom_mult_equation_solutions_l3127_312795


namespace race_time_difference_l3127_312730

-- Define the race participants
structure Racer where
  name : String
  time : ℕ

-- Define the race conditions
def patrick : Racer := { name := "Patrick", time := 60 }
def amy : Racer := { name := "Amy", time := 36 }

-- Define Manu's time in terms of Amy's
def manu_time (amy : Racer) : ℕ := 2 * amy.time

-- Define the theorem
theorem race_time_difference (amy : Racer) (h : amy.time = 36) : 
  manu_time amy - patrick.time = 12 := by
  sorry

end race_time_difference_l3127_312730


namespace pens_taken_after_second_month_pens_taken_after_second_month_is_41_l3127_312734

theorem pens_taken_after_second_month 
  (num_students : ℕ) 
  (red_pens_per_student : ℕ) 
  (black_pens_per_student : ℕ) 
  (pens_taken_first_month : ℕ) 
  (pens_per_student_after_split : ℕ) : ℕ :=
  let total_pens := num_students * (red_pens_per_student + black_pens_per_student)
  let pens_after_first_month := total_pens - pens_taken_first_month
  let pens_after_split := num_students * pens_per_student_after_split
  pens_after_first_month - pens_after_split

theorem pens_taken_after_second_month_is_41 :
  pens_taken_after_second_month 3 62 43 37 79 = 41 := by
  sorry

end pens_taken_after_second_month_pens_taken_after_second_month_is_41_l3127_312734


namespace total_paths_count_l3127_312714

/-- Represents the number of paths between different types of points -/
structure PathCounts where
  redToBlue : Nat
  blueToGreen1 : Nat
  blueToGreen2 : Nat
  greenToOrange1 : Nat
  greenToOrange2 : Nat
  orange1ToB : Nat
  orange2ToB : Nat

/-- Calculates the total number of paths from A to B -/
def totalPaths (p : PathCounts) : Nat :=
  let blueToGreen := p.blueToGreen1 * 2 + p.blueToGreen2 * 2
  let greenToOrange := p.greenToOrange1 + p.greenToOrange2
  (p.redToBlue * blueToGreen * greenToOrange * p.orange1ToB) +
  (p.redToBlue * blueToGreen * greenToOrange * p.orange2ToB)

/-- The theorem stating the total number of paths from A to B -/
theorem total_paths_count (p : PathCounts) 
  (h1 : p.redToBlue = 14)
  (h2 : p.blueToGreen1 = 5)
  (h3 : p.blueToGreen2 = 7)
  (h4 : p.greenToOrange1 = 4)
  (h5 : p.greenToOrange2 = 3)
  (h6 : p.orange1ToB = 2)
  (h7 : p.orange2ToB = 8) :
  totalPaths p = 5376 := by
  sorry

end total_paths_count_l3127_312714


namespace fraction_sum_equality_l3127_312707

theorem fraction_sum_equality : 
  (3 : ℚ) / 12 + (6 : ℚ) / 120 + (9 : ℚ) / 1200 = (3075 : ℚ) / 10000 := by
  sorry

end fraction_sum_equality_l3127_312707


namespace grocery_shopping_remainder_l3127_312785

/-- Calculates the remaining amount after grocery shopping --/
def remaining_amount (initial_amount bread_cost candy_cost cereal_cost milk_cost : ℚ) : ℚ :=
  let initial_purchases := bread_cost + 2 * candy_cost + cereal_cost
  let after_initial := initial_amount - initial_purchases
  let fruit_cost := 0.2 * after_initial
  let after_fruit := after_initial - fruit_cost
  let after_milk := after_fruit - 2 * milk_cost
  let turkey_cost := 0.25 * after_milk
  after_milk - turkey_cost

/-- Theorem stating the remaining amount after grocery shopping --/
theorem grocery_shopping_remainder :
  remaining_amount 100 4 3 6 4.5 = 43.65 := by
  sorry

end grocery_shopping_remainder_l3127_312785


namespace sum_of_roots_quadratic_l3127_312729

theorem sum_of_roots_quadratic (p q : ℝ) : 
  (p^2 - p - 1 = 0) → 
  (q^2 - q - 1 = 0) → 
  (p ≠ q) →
  (∃ x y : ℝ, x^2 - p*x + p*q = 0 ∧ y^2 - p*y + p*q = 0 ∧ x + y = (1 + Real.sqrt 5) / 2) :=
by sorry

end sum_of_roots_quadratic_l3127_312729


namespace opposite_of_negative_three_l3127_312797

theorem opposite_of_negative_three : -(- 3) = 3 := by sorry

end opposite_of_negative_three_l3127_312797


namespace savings_ratio_is_three_fifths_l3127_312742

/-- Represents the savings scenario of Thomas and Joseph -/
structure SavingsScenario where
  thomas_monthly_savings : ℕ
  total_savings : ℕ
  saving_period_months : ℕ

/-- Calculates the ratio of Joseph's monthly savings to Thomas's monthly savings -/
def savings_ratio (scenario : SavingsScenario) : Rat :=
  let thomas_total := scenario.thomas_monthly_savings * scenario.saving_period_months
  let joseph_total := scenario.total_savings - thomas_total
  let joseph_monthly := joseph_total / scenario.saving_period_months
  joseph_monthly / scenario.thomas_monthly_savings

/-- The main theorem stating the ratio of Joseph's to Thomas's monthly savings -/
theorem savings_ratio_is_three_fifths (scenario : SavingsScenario)
  (h1 : scenario.thomas_monthly_savings = 40)
  (h2 : scenario.total_savings = 4608)
  (h3 : scenario.saving_period_months = 72) :
  savings_ratio scenario = 3 / 5 := by
  sorry

#eval savings_ratio { thomas_monthly_savings := 40, total_savings := 4608, saving_period_months := 72 }

end savings_ratio_is_three_fifths_l3127_312742


namespace sweet_distribution_l3127_312790

theorem sweet_distribution (total_sweets : ℕ) (initial_children : ℕ) : 
  (initial_children * 15 = total_sweets) → 
  ((initial_children - 32) * 21 = total_sweets) →
  initial_children = 112 := by
sorry

end sweet_distribution_l3127_312790


namespace min_sum_visible_faces_l3127_312708

/-- Represents a die in the 4x4x4 cube --/
structure Die where
  visible_faces : List Nat
  deriving Repr

/-- Represents the 4x4x4 cube made of dice --/
structure Cube where
  dice : List Die
  deriving Repr

/-- Checks if a die's opposite sides sum to 7 --/
def valid_die (d : Die) : Prop :=
  d.visible_faces.length ≤ 4 ∧ 
  ∀ i j, i + j = 5 → i < d.visible_faces.length → j < d.visible_faces.length → 
    d.visible_faces[i]! + d.visible_faces[j]! = 7

/-- Checks if the cube is valid (4x4x4 and made of 64 dice) --/
def valid_cube (c : Cube) : Prop :=
  c.dice.length = 64 ∧ ∀ d ∈ c.dice, valid_die d

/-- Calculates the sum of visible faces on the cube --/
def sum_visible_faces (c : Cube) : Nat :=
  c.dice.foldl (λ acc d => acc + d.visible_faces.foldl (λ sum face => sum + face) 0) 0

/-- Theorem: The smallest possible sum of visible faces on a valid 4x4x4 cube is 304 --/
theorem min_sum_visible_faces (c : Cube) (h : valid_cube c) : 
  ∃ (min_cube : Cube), valid_cube min_cube ∧ 
    sum_visible_faces min_cube = 304 ∧
    ∀ (other_cube : Cube), valid_cube other_cube → 
      sum_visible_faces other_cube ≥ sum_visible_faces min_cube := by
  sorry

end min_sum_visible_faces_l3127_312708


namespace sphere_radius_ratio_l3127_312743

theorem sphere_radius_ratio (v_large v_small : ℝ) (h1 : v_large = 324 * Real.pi) (h2 : v_small = 0.25 * v_large) :
  (v_small / v_large) ^ (1/3 : ℝ) = 1/2 := by
  sorry

end sphere_radius_ratio_l3127_312743


namespace greatest_multiple_of_8_no_repeats_remainder_l3127_312794

/-- The greatest integer multiple of 8 with no repeating digits -/
def N : ℕ :=
  sorry

/-- Predicate to check if a natural number has no repeating digits -/
def has_no_repeating_digits (n : ℕ) : Prop :=
  sorry

theorem greatest_multiple_of_8_no_repeats_remainder : 
  N % 1000 = 120 ∧ 
  N % 8 = 0 ∧
  has_no_repeating_digits N ∧
  ∀ m : ℕ, m % 8 = 0 → has_no_repeating_digits m → m ≤ N :=
sorry

end greatest_multiple_of_8_no_repeats_remainder_l3127_312794


namespace exact_three_wins_probability_l3127_312722

/-- The probability of winning a prize in a single draw -/
def p : ℚ := 2/5

/-- The number of participants (trials) -/
def n : ℕ := 4

/-- The number of desired successes -/
def k : ℕ := 3

/-- The probability of exactly k successes in n independent trials 
    with probability p of success in each trial -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem exact_three_wins_probability :
  binomial_probability n k p = 96/625 := by
  sorry

end exact_three_wins_probability_l3127_312722


namespace proportion_problem_l3127_312763

theorem proportion_problem (y : ℝ) : (0.75 : ℝ) / 2 = y / 8 → y = 3 := by
  sorry

end proportion_problem_l3127_312763


namespace average_of_remaining_numbers_l3127_312792

theorem average_of_remaining_numbers
  (total_count : Nat)
  (subset_count : Nat)
  (total_average : ℝ)
  (subset_average : ℝ)
  (h_total_count : total_count = 15)
  (h_subset_count : subset_count = 9)
  (h_total_average : total_average = 30.5)
  (h_subset_average : subset_average = 17.75) :
  let remaining_count := total_count - subset_count
  let remaining_sum := total_count * total_average - subset_count * subset_average
  remaining_sum / remaining_count = 49.625 := by
sorry


end average_of_remaining_numbers_l3127_312792


namespace house_worth_problem_l3127_312741

theorem house_worth_problem (initial_price final_price : ℝ) 
  (h1 : final_price = initial_price * 1.1 * 0.9)
  (h2 : final_price = 99000) : initial_price = 100000 := by
  sorry

end house_worth_problem_l3127_312741


namespace negative_reals_inequality_l3127_312772

theorem negative_reals_inequality (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + b + c ≤ (a^2 + b^2) / (2*c) + (b^2 + c^2) / (2*a) + (c^2 + a^2) / (2*b) ∧
  (a^2 + b^2) / (2*c) + (b^2 + c^2) / (2*a) + (c^2 + a^2) / (2*b) ≤ a^2 / (b*c) + b^2 / (c*a) + c^2 / (a*b) :=
by sorry

end negative_reals_inequality_l3127_312772


namespace sum_of_coefficients_is_zero_l3127_312780

/-- Given two linear functions f and g, prove that A + B = 0 under certain conditions -/
theorem sum_of_coefficients_is_zero
  (A B : ℝ)
  (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = A * x + B)
  (h₂ : ∀ x, g x = B * x + A)
  (h₃ : A ≠ B)
  (h₄ : ∀ x, f (g x) - g (f x) = B - A) :
  A + B = 0 := by
  sorry

end sum_of_coefficients_is_zero_l3127_312780


namespace sock_probability_theorem_l3127_312727

def gray_socks : ℕ := 12
def white_socks : ℕ := 10
def blue_socks : ℕ := 6

def total_socks : ℕ := gray_socks + white_socks + blue_socks

def probability_matching_or_different_colors : ℚ :=
  let total_combinations := Nat.choose total_socks 3
  let matching_gray := Nat.choose gray_socks 2 * (white_socks + blue_socks)
  let matching_white := Nat.choose white_socks 2 * (gray_socks + blue_socks)
  let matching_blue := Nat.choose blue_socks 2 * (gray_socks + white_socks)
  let all_different := gray_socks * white_socks * blue_socks
  let favorable_outcomes := matching_gray + matching_white + matching_blue + all_different
  (favorable_outcomes : ℚ) / total_combinations

theorem sock_probability_theorem :
  probability_matching_or_different_colors = 81 / 91 :=
by sorry

end sock_probability_theorem_l3127_312727


namespace sqrt_two_plus_x_l3127_312726

theorem sqrt_two_plus_x (x : ℝ) : x = Real.sqrt (2 + x) → x = 2 := by
  sorry

end sqrt_two_plus_x_l3127_312726


namespace division_problem_l3127_312750

theorem division_problem (A : ℕ) (h : A % 7 = 3 ∧ A / 7 = 5) : A = 38 := by
  sorry

end division_problem_l3127_312750


namespace total_ways_is_eight_l3127_312760

/-- The number of course options available --/
def num_courses : Nat := 2

/-- The number of students choosing courses --/
def num_students : Nat := 3

/-- Calculates the total number of ways students can choose courses --/
def total_ways : Nat := num_courses ^ num_students

/-- Theorem stating that the total number of ways to choose courses is 8 --/
theorem total_ways_is_eight : total_ways = 8 := by
  sorry

end total_ways_is_eight_l3127_312760


namespace cube_sum_equals_110_l3127_312705

theorem cube_sum_equals_110 (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 6) : 
  x^3 + y^3 = 110 := by
sorry

end cube_sum_equals_110_l3127_312705


namespace december_sales_multiple_l3127_312762

/-- Represents the sales data for a department store --/
structure SalesData where
  /-- Average monthly sales from January to November --/
  avg_sales : ℝ
  /-- Multiple of average sales for December --/
  dec_multiple : ℝ

/-- Theorem stating the conditions and the result to be proved --/
theorem december_sales_multiple (data : SalesData) :
  (data.dec_multiple * data.avg_sales) / (11 * data.avg_sales + data.dec_multiple * data.avg_sales) = 0.35294117647058826 →
  data.dec_multiple = 6 := by
  sorry

end december_sales_multiple_l3127_312762


namespace log_equality_l3127_312774

theorem log_equality (x k : ℝ) (h1 : Real.log 3 / Real.log 8 = x) (h2 : Real.log 81 / Real.log 2 = k * x) : k = 12 := by
  sorry

end log_equality_l3127_312774


namespace points_below_line_l3127_312747

def arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def geometric_sequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

theorem points_below_line (x₁ x₂ y₁ y₂ : ℝ) :
  arithmetic_sequence 1 x₁ x₂ 2 →
  geometric_sequence 1 y₁ y₂ 2 →
  x₁ > y₁ ∧ x₂ > y₂ := by
  sorry

end points_below_line_l3127_312747


namespace weight_density_half_fluid_density_l3127_312701

/-- A spring-mass system submerged in a fluid -/
structure SpringMassSystem where
  /-- Spring constant -/
  k : ℝ
  /-- Mass of the weight -/
  m : ℝ
  /-- Acceleration due to gravity -/
  g : ℝ
  /-- Density of the fluid (kerosene) -/
  ρ_fluid : ℝ
  /-- Density of the weight material -/
  ρ_material : ℝ
  /-- Extension of the spring -/
  x : ℝ

/-- The theorem stating that the density of the weight material is half the density of the fluid -/
theorem weight_density_half_fluid_density (system : SpringMassSystem) 
  (h1 : system.k * system.x = system.m * system.g)  -- Force balance in air
  (h2 : system.m * system.g + system.k * system.x = system.ρ_fluid * system.g * (system.m / system.ρ_material))  -- Force balance in fluid
  (h3 : system.ρ_fluid > 0)  -- Fluid density is positive
  (h4 : system.m > 0)  -- Mass is positive
  (h5 : system.g > 0)  -- Gravity is positive
  : system.ρ_material = system.ρ_fluid / 2 := by
  sorry

#eval 800 / 2  -- Should output 400

end weight_density_half_fluid_density_l3127_312701


namespace first_box_weight_proof_l3127_312786

/-- The weight of the first box given the conditions in the problem -/
def first_box_weight : ℝ := 24

/-- The weight of the third box -/
def third_box_weight : ℝ := 13

/-- The difference between the weight of the first and third box -/
def weight_difference : ℝ := 11

theorem first_box_weight_proof :
  first_box_weight = third_box_weight + weight_difference := by
  sorry

end first_box_weight_proof_l3127_312786


namespace eighth_term_and_half_l3127_312781

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r ^ (n - 1)

theorem eighth_term_and_half (a : ℚ) (r : ℚ) :
  a = 12 → r = 1/2 →
  geometric_sequence a r 8 = 3/32 ∧
  (1/2 * geometric_sequence a r 8) = 3/64 := by
  sorry

end eighth_term_and_half_l3127_312781


namespace contractor_problem_l3127_312712

/-- Calculates the original number of days to complete a job given the original number of laborers,
    the number of absent laborers, and the number of days taken by the remaining laborers. -/
def original_completion_time (total_laborers : ℕ) (absent_laborers : ℕ) (actual_days : ℕ) : ℕ :=
  (total_laborers - absent_laborers) * actual_days / total_laborers

theorem contractor_problem (total_laborers absent_laborers actual_days : ℕ) 
  (h1 : total_laborers = 7)
  (h2 : absent_laborers = 3)
  (h3 : actual_days = 14) :
  original_completion_time total_laborers absent_laborers actual_days = 8 := by
  sorry

#eval original_completion_time 7 3 14

end contractor_problem_l3127_312712


namespace three_power_fraction_equals_five_fourths_l3127_312776

theorem three_power_fraction_equals_five_fourths :
  (3^100 + 3^98) / (3^100 - 3^98) = 5/4 := by
  sorry

end three_power_fraction_equals_five_fourths_l3127_312776

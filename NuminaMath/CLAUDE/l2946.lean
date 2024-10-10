import Mathlib

namespace polynomial_factorization_l2946_294658

theorem polynomial_factorization (x : ℝ) :
  6 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 5 * x^2 =
  (3 * x^2 + 93 * x) * (2 * x^2 + 178 * x + 5432) := by
  sorry

end polynomial_factorization_l2946_294658


namespace counterfeit_coin_determination_l2946_294672

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Equal : WeighingResult
  | LeftHeavier : WeighingResult
  | RightHeavier : WeighingResult

/-- Represents a group of coins -/
structure CoinGroup where
  size : Nat
  containsCounterfeit : Bool

/-- Represents a weighing operation -/
def weighing (left right : CoinGroup) : WeighingResult :=
  sorry

/-- The main theorem stating that it's possible to determine if counterfeit coins are heavier or lighter -/
theorem counterfeit_coin_determination :
  ∀ (coins : List CoinGroup),
    coins.length = 3 →
    (coins.map CoinGroup.size).sum = 103 →
    (coins.filter CoinGroup.containsCounterfeit).length = 2 →
    ∃ (w₁ w₂ w₃ : CoinGroup × CoinGroup),
      (∀ g₁ g₂, weighing g₁ g₂ = WeighingResult.Equal → g₁.containsCounterfeit = g₂.containsCounterfeit) →
      let r₁ := weighing w₁.1 w₁.2
      let r₂ := weighing w₂.1 w₂.2
      let r₃ := weighing w₃.1 w₃.2
      (r₁ ≠ WeighingResult.Equal ∨ r₂ ≠ WeighingResult.Equal ∨ r₃ ≠ WeighingResult.Equal) :=
by
  sorry

end counterfeit_coin_determination_l2946_294672


namespace probability_sum_thirty_l2946_294637

/-- Die A is a 30-faced die numbered 1-25 and 27-31 -/
def DieA : Finset ℕ := Finset.filter (fun n => n ≠ 26) (Finset.range 32 \ Finset.range 1)

/-- Die B is a 30-faced die numbered 1-20 and 26-31 -/
def DieB : Finset ℕ := (Finset.range 21 \ Finset.range 1) ∪ (Finset.range 32 \ Finset.range 26)

/-- The set of all possible outcomes when rolling both dice -/
def AllOutcomes : Finset (ℕ × ℕ) := DieA.product DieB

/-- The set of outcomes where the sum of the rolled numbers is 30 -/
def SumThirty : Finset (ℕ × ℕ) := AllOutcomes.filter (fun p => p.1 + p.2 = 30)

/-- The probability of rolling a sum of 30 with the given dice -/
def ProbabilitySumThirty : ℚ := (SumThirty.card : ℚ) / (AllOutcomes.card : ℚ)

theorem probability_sum_thirty : ProbabilitySumThirty = 59 / 900 := by sorry

end probability_sum_thirty_l2946_294637


namespace jacket_cost_is_30_l2946_294680

def calculate_jacket_cost (initial_amount dresses_count pants_count jackets_count dress_cost pants_cost transportation_cost remaining_amount : ℕ) : ℕ :=
  let total_spent := initial_amount - remaining_amount
  let dresses_cost := dresses_count * dress_cost
  let pants_cost := pants_count * pants_cost
  let other_costs := dresses_cost + pants_cost + transportation_cost
  let jackets_total_cost := total_spent - other_costs
  jackets_total_cost / jackets_count

theorem jacket_cost_is_30 :
  calculate_jacket_cost 400 5 3 4 20 12 5 139 = 30 := by
  sorry

end jacket_cost_is_30_l2946_294680


namespace coordinate_sum_of_A_l2946_294619

-- Define the points
def B : ℝ × ℝ := (2, 8)
def C : ℝ × ℝ := (0, 2)

-- Define the theorem
theorem coordinate_sum_of_A (A : ℝ × ℝ) :
  (A.1 - C.1) / (B.1 - C.1) = 1/3 ∧
  (A.2 - C.2) / (B.2 - C.2) = 1/3 →
  A.1 + A.2 = -14 := by
  sorry

end coordinate_sum_of_A_l2946_294619


namespace pencil_difference_l2946_294610

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The time frame in minutes -/
def time_frame : ℕ := 6

/-- The time it takes for a hand-crank sharpener to sharpen one pencil (in seconds) -/
def hand_crank_time : ℕ := 45

/-- The time it takes for an electric sharpener to sharpen one pencil (in seconds) -/
def electric_time : ℕ := 20

/-- The difference in the number of pencils sharpened between the electric and hand-crank sharpeners -/
theorem pencil_difference : 
  (time_frame * seconds_per_minute) / electric_time - 
  (time_frame * seconds_per_minute) / hand_crank_time = 10 := by
  sorry

end pencil_difference_l2946_294610


namespace product_of_parts_of_complex_square_l2946_294673

theorem product_of_parts_of_complex_square : ∃ (a b : ℝ), (Complex.mk 1 2)^2 = Complex.mk a b ∧ a * b = -12 := by
  sorry

end product_of_parts_of_complex_square_l2946_294673


namespace quadrilateral_prism_edges_and_vertices_l2946_294656

/-- A prism with a quadrilateral base -/
structure QuadrilateralPrism :=
  (lateral_faces : ℕ)
  (lateral_faces_eq : lateral_faces = 4)

/-- The number of edges in a quadrilateral prism -/
def num_edges (p : QuadrilateralPrism) : ℕ := 12

/-- The number of vertices in a quadrilateral prism -/
def num_vertices (p : QuadrilateralPrism) : ℕ := 8

/-- Theorem stating that a quadrilateral prism has 12 edges and 8 vertices -/
theorem quadrilateral_prism_edges_and_vertices (p : QuadrilateralPrism) :
  num_edges p = 12 ∧ num_vertices p = 8 := by
  sorry

end quadrilateral_prism_edges_and_vertices_l2946_294656


namespace four_digit_number_with_specific_remainders_l2946_294678

theorem four_digit_number_with_specific_remainders :
  ∃! N : ℕ, 
    N % 131 = 112 ∧
    N % 132 = 98 ∧
    1000 ≤ N ∧ N ≤ 9999 :=
by sorry

end four_digit_number_with_specific_remainders_l2946_294678


namespace three_digit_number_proof_l2946_294661

theorem three_digit_number_proof :
  ∃! x : ℕ,
    (100 ≤ x ∧ x < 1000) ∧
    (x * (x / 100) = 494) ∧
    (x * ((x / 10) % 10) = 988) ∧
    (x * (x % 10) = 1729) ∧
    x = 247 := by
  sorry

end three_digit_number_proof_l2946_294661


namespace line_inclination_angle_l2946_294641

/-- The inclination angle of the line √3x - y + 1 = 0 is π/3 -/
theorem line_inclination_angle :
  let line := {(x, y) : ℝ × ℝ | Real.sqrt 3 * x - y + 1 = 0}
  ∃ α : ℝ, α = π / 3 ∧ ∀ (x y : ℝ), (x, y) ∈ line → Real.tan α = Real.sqrt 3 :=
by sorry

end line_inclination_angle_l2946_294641


namespace triangle_position_after_two_moves_l2946_294648

/-- Represents the sides of a square --/
inductive SquareSide
  | Top
  | Right
  | Bottom
  | Left

/-- Represents a regular octagon --/
structure RegularOctagon where
  inner_angle : ℝ
  inner_angle_eq : inner_angle = 135

/-- Represents a square rolling around an octagon --/
structure RollingSquare where
  octagon : RegularOctagon
  rotation_per_move : ℝ
  rotation_per_move_eq : rotation_per_move = 135

/-- The result of rolling a square around an octagon --/
def roll_square (initial_side : SquareSide) (num_moves : ℕ) : SquareSide :=
  sorry

theorem triangle_position_after_two_moves :
  ∀ (octagon : RegularOctagon) (square : RollingSquare),
    roll_square SquareSide.Bottom 2 = SquareSide.Bottom :=
  sorry

end triangle_position_after_two_moves_l2946_294648


namespace sum_of_four_integers_l2946_294622

theorem sum_of_four_integers (m n p q : ℕ+) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4 →
  m + n + p + q = 28 := by
  sorry

end sum_of_four_integers_l2946_294622


namespace salesman_pear_sales_l2946_294657

theorem salesman_pear_sales (morning_sales afternoon_sales total_sales : ℕ) :
  afternoon_sales = 2 * morning_sales →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 510 →
  afternoon_sales = 340 :=
by
  sorry

end salesman_pear_sales_l2946_294657


namespace messages_total_680_l2946_294666

/-- Calculates the total number of messages sent by Alina and Lucia over three days -/
def total_messages (lucia_day1 : ℕ) (alina_difference : ℕ) : ℕ :=
  let alina_day1 := lucia_day1 - alina_difference
  let day1_total := lucia_day1 + alina_day1
  let lucia_day2 := lucia_day1 / 3
  let alina_day2 := alina_day1 * 2
  let day2_total := lucia_day2 + alina_day2
  day1_total + day2_total + day1_total

theorem messages_total_680 :
  total_messages 120 20 = 680 := by
  sorry

end messages_total_680_l2946_294666


namespace carousel_revolutions_l2946_294671

theorem carousel_revolutions (r₁ r₂ : ℝ) (n₁ : ℕ) :
  r₁ = 30 →
  r₂ = 10 →
  n₁ = 40 →
  r₁ * n₁ = r₂ * (120 : ℕ) :=
by sorry

end carousel_revolutions_l2946_294671


namespace inequalities_solution_l2946_294663

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 < 0
def inequality2 (x : ℝ) : Prop := (2 * x) / (x + 1) ≥ 1

-- State the theorem
theorem inequalities_solution :
  (∀ x : ℝ, inequality1 x ↔ (1/2 < x ∧ x < 1)) ∧
  (∀ x : ℝ, inequality2 x ↔ (x < -1 ∨ x ≥ 1)) :=
sorry

end inequalities_solution_l2946_294663


namespace total_baseball_cards_l2946_294643

/-- The number of baseball cards each person has -/
structure BaseballCards where
  carlos : ℕ
  matias : ℕ
  jorge : ℕ
  ella : ℕ

/-- The conditions of the baseball card problem -/
def baseball_card_problem (cards : BaseballCards) : Prop :=
  cards.carlos = 20 ∧
  cards.matias = cards.carlos - 6 ∧
  cards.jorge = cards.matias ∧
  cards.ella = 2 * (cards.jorge + cards.matias)

/-- The theorem stating the total number of baseball cards -/
theorem total_baseball_cards (cards : BaseballCards) 
  (h : baseball_card_problem cards) : 
  cards.carlos + cards.matias + cards.jorge + cards.ella = 104 := by
  sorry


end total_baseball_cards_l2946_294643


namespace polynomial_root_implies_coefficients_l2946_294616

theorem polynomial_root_implies_coefficients : ∀ (a b : ℝ), 
  (Complex.I : ℂ) ^ 3 + a * (Complex.I : ℂ) ^ 2 + 2 * (Complex.I : ℂ) + b = (2 - 3 * Complex.I : ℂ) ^ 3 →
  a = -5/4 ∧ b = 143/4 := by
  sorry

end polynomial_root_implies_coefficients_l2946_294616


namespace agent_percentage_l2946_294667

def total_copies : ℕ := 1000000
def earnings_per_copy : ℚ := 2
def steve_kept_earnings : ℚ := 1620000

theorem agent_percentage : 
  let total_earnings := total_copies * earnings_per_copy
  let agent_earnings := total_earnings - steve_kept_earnings
  (agent_earnings / total_earnings) * 100 = 19 := by sorry

end agent_percentage_l2946_294667


namespace shekar_average_marks_l2946_294613

def shekar_marks : List ℕ := [76, 65, 82, 67, 75]

theorem shekar_average_marks :
  (shekar_marks.sum : ℚ) / shekar_marks.length = 73 := by
  sorry

end shekar_average_marks_l2946_294613


namespace minimum_coins_for_all_amounts_l2946_294609

/-- Represents the different types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- The value of each coin in cents --/
def coinValue : Coin → ℕ
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- A list of coins --/
def CoinList := List Coin

/-- Calculates the total value of a list of coins in cents --/
def totalValue (coins : CoinList) : ℕ :=
  coins.foldl (fun acc coin => acc + coinValue coin) 0

/-- Checks if a given amount can be made with a list of coins --/
def canMakeAmount (coins : CoinList) (amount : ℕ) : Prop :=
  ∃ (subset : CoinList), subset.Subset coins ∧ totalValue subset = amount

/-- The main theorem to prove --/
theorem minimum_coins_for_all_amounts :
  ∃ (coins : CoinList),
    coins.length = 11 ∧
    (∀ (amount : ℕ), amount > 0 ∧ amount < 100 → canMakeAmount coins amount) ∧
    (∀ (otherCoins : CoinList),
      (∀ (amount : ℕ), amount > 0 ∧ amount < 100 → canMakeAmount otherCoins amount) →
      otherCoins.length ≥ 11) :=
sorry

end minimum_coins_for_all_amounts_l2946_294609


namespace complex_equation_solution_l2946_294665

theorem complex_equation_solution (z : ℂ) :
  Complex.abs z = 2 + z + Complex.I * 3 → z = 5 / 4 - Complex.I * 3 := by
  sorry

end complex_equation_solution_l2946_294665


namespace surrounding_decagon_theorem_l2946_294691

/-- The number of sides of the surrounding polygons when a regular m-sided polygon
    is surrounded by m regular n-sided polygons without gaps or overlaps. -/
def surrounding_polygon_sides (m : ℕ) : ℕ :=
  if m = 4 then 8 else
  if m = 10 then
    let interior_angle_m := (180 * (m - 2)) / m
    let n := (720 / (360 - interior_angle_m) : ℕ)
    n
  else 0

/-- Theorem stating that when a regular 10-sided polygon is surrounded by 10 regular n-sided polygons
    without gaps or overlaps, n must equal 5. -/
theorem surrounding_decagon_theorem :
  surrounding_polygon_sides 10 = 5 := by
  sorry

end surrounding_decagon_theorem_l2946_294691


namespace simplify_expression_l2946_294627

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 25) = 152 * x + 45 := by
  sorry

end simplify_expression_l2946_294627


namespace cookie_brownie_difference_after_week_l2946_294653

/-- Calculates the difference between remaining cookies and brownies after a week -/
def cookieBrownieDifference (initialCookies initialBrownies dailyCookies dailyBrownies days : ℕ) : ℕ :=
  let remainingCookies := initialCookies - dailyCookies * days
  let remainingBrownies := initialBrownies - dailyBrownies * days
  remainingCookies - remainingBrownies

/-- Proves that the difference between remaining cookies and brownies after a week is 36 -/
theorem cookie_brownie_difference_after_week :
  cookieBrownieDifference 60 10 3 1 7 = 36 := by
  sorry

end cookie_brownie_difference_after_week_l2946_294653


namespace train_speed_problem_l2946_294615

/-- Proves that for a journey of 70 km, if a train traveling at 35 kmph
    arrives 15 minutes late compared to its on-time speed,
    then the on-time speed is 40 kmph. -/
theorem train_speed_problem (v : ℝ) : 
  (70 / v + 0.25 = 70 / 35) → v = 40 := by
  sorry

end train_speed_problem_l2946_294615


namespace expression_equals_two_l2946_294679

theorem expression_equals_two (x : ℝ) (h : x ≠ -1) :
  ((x - 1) / (x + 1) + 1) / (x / (x + 1)) = 2 := by sorry

end expression_equals_two_l2946_294679


namespace rectangle_breadth_l2946_294655

theorem rectangle_breadth (square_area : ℝ) (rectangle_area : ℝ) :
  square_area = 3600 →
  rectangle_area = 240 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  rectangle_area = rectangle_length * (rectangle_area / rectangle_length) →
  rectangle_area / rectangle_length = 10 := by
  sorry

end rectangle_breadth_l2946_294655


namespace cara_seating_arrangements_l2946_294602

theorem cara_seating_arrangements (n : ℕ) (h : n = 6) : Nat.choose n 2 = 15 := by
  sorry

end cara_seating_arrangements_l2946_294602


namespace system_equations_properties_l2946_294698

/-- Given a system of equations with parameters x, y, and m, prove properties about the solution and a related expression. -/
theorem system_equations_properties (x y m : ℝ) (h1 : 3 * x + 2 * y = m + 2) (h2 : 2 * x + y = m - 1)
  (hx : x > 0) (hy : y > 0) :
  (x = m - 4 ∧ y = 7 - m) ∧
  (4 < m ∧ m < 7) ∧
  (∀ (m : ℕ), 4 < m → m < 7 → (2 * x - 3 * y + m) ≤ 7) :=
by sorry

end system_equations_properties_l2946_294698


namespace factorial_plus_one_divisible_implies_prime_l2946_294670

theorem factorial_plus_one_divisible_implies_prime (n : ℕ) :
  (n! + 1) % (n + 1) = 0 → Nat.Prime (n + 1) := by
  sorry

end factorial_plus_one_divisible_implies_prime_l2946_294670


namespace tangent_slope_at_one_l2946_294674

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_slope_at_one
  (h1 : Differentiable ℝ f)
  (h2 : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |(f 1 - f (1 - 2*Δx)) / (2*Δx) + 1| < ε) :
  deriv f 1 = -1 :=
sorry

end tangent_slope_at_one_l2946_294674


namespace smallest_gcd_qr_l2946_294601

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 770) :
  ∃ (q' r' : ℕ+), Nat.gcd q' r' = 70 ∧
    ∀ (q'' r'' : ℕ+), Nat.gcd p q'' = 210 → Nat.gcd p r'' = 770 → Nat.gcd q'' r'' ≥ 70 :=
by sorry

end smallest_gcd_qr_l2946_294601


namespace positive_numbers_relation_l2946_294676

theorem positive_numbers_relation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 / b = 5) (h2 : b^2 / c = 3) (h3 : c^2 / a = 7) : a = 15 := by
  sorry

end positive_numbers_relation_l2946_294676


namespace farm_area_l2946_294694

/-- Proves that a rectangular farm with given conditions has an area of 1200 square meters -/
theorem farm_area (short_side : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) : 
  short_side = 30 →
  cost_per_meter = 14 →
  total_cost = 1680 →
  ∃ (long_side : ℝ),
    long_side > 0 ∧
    cost_per_meter * (long_side + short_side + Real.sqrt (long_side^2 + short_side^2)) = total_cost ∧
    long_side * short_side = 1200 :=
by
  sorry

#check farm_area

end farm_area_l2946_294694


namespace prime_pairs_divisibility_l2946_294688

theorem prime_pairs_divisibility (p q : ℕ) : 
  p.Prime ∧ q.Prime ∧ p < 2023 ∧ q < 2023 ∧ 
  (p ∣ q^2 + 8) ∧ (q ∣ p^2 + 8) → 
  ((p = 2 ∧ q = 2) ∨ (p = 17 ∧ q = 3) ∨ (p = 11 ∧ q = 5)) := by
sorry

end prime_pairs_divisibility_l2946_294688


namespace total_earnings_before_car_purchase_l2946_294638

def monthly_income : ℕ := 4000
def monthly_savings : ℕ := 500
def car_cost : ℕ := 45000

theorem total_earnings_before_car_purchase :
  (car_cost / monthly_savings) * monthly_income = 360000 := by
  sorry

end total_earnings_before_car_purchase_l2946_294638


namespace cylinder_volume_scaling_l2946_294625

theorem cylinder_volume_scaling (r h V : ℝ) :
  V = π * r^2 * h →
  ∀ (k : ℝ), k > 0 →
    π * (k*r)^2 * (k*h) = k^3 * V :=
by sorry

end cylinder_volume_scaling_l2946_294625


namespace correct_divisor_problem_l2946_294697

theorem correct_divisor_problem (student_divisor : ℕ) (student_answer : ℕ) (correct_answer : ℕ) : 
  student_divisor = 72 → student_answer = 24 → correct_answer = 48 →
  ∃ (dividend : ℕ) (correct_divisor : ℕ), 
    dividend / student_divisor = student_answer ∧
    dividend / correct_divisor = correct_answer ∧
    correct_divisor = 36 := by
  sorry

end correct_divisor_problem_l2946_294697


namespace doctors_lawyers_ratio_l2946_294654

theorem doctors_lawyers_ratio 
  (d : ℕ) -- number of doctors
  (l : ℕ) -- number of lawyers
  (h1 : d > 0) -- ensure there's at least one doctor
  (h2 : l > 0) -- ensure there's at least one lawyer
  (h3 : (35 * d + 50 * l) / (d + l) = 40) -- average age of the group is 40
  : d = 2 * l := by
sorry

end doctors_lawyers_ratio_l2946_294654


namespace jane_toy_bear_production_l2946_294685

/-- Jane's toy bear production problem -/
theorem jane_toy_bear_production 
  (base_output : ℝ) 
  (base_hours : ℝ) 
  (assistant_output_increase : ℝ) 
  (assistant_hours_decrease : ℝ) 
  (assistant_A_increase : ℝ) 
  (assistant_B_increase : ℝ) 
  (assistant_C_increase : ℝ) 
  (h1 : assistant_output_increase = 0.8) 
  (h2 : assistant_hours_decrease = 0.1) 
  (h3 : assistant_A_increase = 1.0) 
  (h4 : assistant_B_increase = 0.75) 
  (h5 : assistant_C_increase = 0.5) :
  let output_A := (1 + assistant_A_increase) * base_output / ((1 - assistant_hours_decrease) * base_hours)
  let output_B := (1 + assistant_B_increase) * base_output / ((1 - assistant_hours_decrease) * base_hours)
  let output_C := (1 + assistant_C_increase) * base_output / ((1 - assistant_hours_decrease) * base_hours)
  let increase_A := (output_A / (base_output / base_hours) - 1) * 100
  let increase_B := (output_B / (base_output / base_hours) - 1) * 100
  let increase_C := (output_C / (base_output / base_hours) - 1) * 100
  let average_increase := (increase_A + increase_B + increase_C) / 3
  ∃ ε > 0, |average_increase - 94.43| < ε :=
by sorry

end jane_toy_bear_production_l2946_294685


namespace swimming_club_boys_l2946_294696

theorem swimming_club_boys (total_members : ℕ) (total_attendees : ℕ) :
  total_members = 30 →
  total_attendees = 20 →
  ∃ (boys girls : ℕ),
    boys + girls = total_members ∧
    boys + (girls / 3) = total_attendees ∧
    boys = 15 := by
  sorry

end swimming_club_boys_l2946_294696


namespace exchange_result_l2946_294690

/-- The number of bills after exchanging 2 $100 bills as described -/
def total_bills : ℕ :=
  let initial_hundred_bills : ℕ := 2
  let fifty_bills : ℕ := 2  -- From exchanging one $100 bill
  let ten_bills : ℕ := 50 / 10  -- From exchanging half of the remaining $100 bill
  let five_bills : ℕ := 50 / 5  -- From exchanging the other half of the remaining $100 bill
  fifty_bills + ten_bills + five_bills

/-- Theorem stating that the total number of bills after the exchange is 17 -/
theorem exchange_result : total_bills = 17 := by
  sorry

end exchange_result_l2946_294690


namespace shortest_tangent_length_l2946_294636

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 64
def C₂ (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 100

-- Define the tangent line segment
def is_tangent_to (P Q : ℝ × ℝ) : Prop :=
  C₁ P.1 P.2 ∧ C₂ Q.1 Q.2 ∧
  ∀ R : ℝ × ℝ, (C₁ R.1 R.2 ∨ C₂ R.1 R.2) → Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) + Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)

-- Theorem statement
theorem shortest_tangent_length :
  ∃ P Q : ℝ × ℝ, is_tangent_to P Q ∧
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 190 / 3 ∧
  ∀ P' Q' : ℝ × ℝ, is_tangent_to P' Q' →
    Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≥ 190 / 3 := by
  sorry

end shortest_tangent_length_l2946_294636


namespace smaller_solid_volume_is_one_sixth_l2946_294607

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  edgeLength : ℝ
  vertex : Point3D

/-- Calculates the volume of the smaller solid created by a plane cutting a cube -/
def smallerSolidVolume (cube : Cube) (plane : Plane) : ℝ :=
  sorry

/-- Theorem: The volume of the smaller solid in a cube with edge length 2,
    cut by a plane passing through vertex D and midpoints of AB and CG, is 1/6 -/
theorem smaller_solid_volume_is_one_sixth :
  let cube := Cube.mk 2 (Point3D.mk 0 0 0)
  let plane := Plane.mk 2 (-4) (-8) 0
  smallerSolidVolume cube plane = 1/6 := by
  sorry

end smaller_solid_volume_is_one_sixth_l2946_294607


namespace song_game_theorem_l2946_294668

/-- Represents the "Guess the Song Title" game -/
structure SongGame where
  /-- Probability of passing each level -/
  pass_prob : Fin 3 → ℚ
  /-- Probability of continuing to next level -/
  continue_prob : ℚ
  /-- Reward for passing each level -/
  reward : Fin 3 → ℕ

/-- The specific game instance as described in the problem -/
def game : SongGame :=
  { pass_prob := λ i => [3/4, 2/3, 1/2].get i
    continue_prob := 1/2
    reward := λ i => [1000, 2000, 3000].get i }

/-- Probability of passing first level but receiving zero reward -/
def prob_pass_first_zero_reward (g : SongGame) : ℚ :=
  g.pass_prob 0 * g.continue_prob * (1 - g.pass_prob 1) +
  g.pass_prob 0 * g.continue_prob * g.pass_prob 1 * g.continue_prob * (1 - g.pass_prob 2)

/-- Expected value of total reward -/
def expected_reward (g : SongGame) : ℚ :=
  g.pass_prob 0 * (1 - g.continue_prob) * g.reward 0 +
  g.pass_prob 0 * g.continue_prob * g.pass_prob 1 * (1 - g.continue_prob) * (g.reward 0 + g.reward 1) +
  g.pass_prob 0 * g.continue_prob * g.pass_prob 1 * g.continue_prob * g.pass_prob 2 * (g.reward 0 + g.reward 1 + g.reward 2)

theorem song_game_theorem (g : SongGame) :
  prob_pass_first_zero_reward g = 3/16 ∧ expected_reward g = 1125 :=
sorry

end song_game_theorem_l2946_294668


namespace statues_painted_l2946_294677

theorem statues_painted (total_paint : ℚ) (paint_per_statue : ℚ) :
  total_paint = 7/16 →
  paint_per_statue = 1/16 →
  (total_paint / paint_per_statue : ℚ) = 7 :=
by sorry

end statues_painted_l2946_294677


namespace intersection_of_lines_l2946_294639

/-- Given four points in 3D space, this theorem states that the intersection
    of the lines formed by these points is at a specific coordinate. -/
theorem intersection_of_lines (P Q R S : ℝ × ℝ × ℝ) : 
  P = (4, -8, 8) →
  Q = (14, -18, 14) →
  R = (1, 2, -7) →
  S = (3, -6, 9) →
  ∃ t s : ℝ, 
    (4 + 10*t, -8 - 10*t, 8 + 6*t) = (1 + 2*s, 2 - 8*s, -7 + 16*s) ∧
    (4 + 10*t, -8 - 10*t, 8 + 6*t) = (14/3, -22/3, 38/3) :=
by sorry


end intersection_of_lines_l2946_294639


namespace last_two_average_l2946_294650

theorem last_two_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 60 →
  ((list.take 3).sum / 3 : ℝ) = 50 →
  ((list.drop 3).take 2).sum / 2 = 70 →
  ((list.drop 5).sum / 2 : ℝ) = 65 := by
sorry

end last_two_average_l2946_294650


namespace max_digit_sum_in_range_l2946_294686

def is_valid_time (h m s : ℕ) : Prop :=
  13 ≤ h ∧ h ≤ 23 ∧ m < 60 ∧ s < 60

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def time_digit_sum (h m s : ℕ) : ℕ :=
  digit_sum h + digit_sum m + digit_sum s

theorem max_digit_sum_in_range :
  ∃ (h m s : ℕ), is_valid_time h m s ∧
    ∀ (h' m' s' : ℕ), is_valid_time h' m' s' →
      time_digit_sum h' m' s' ≤ time_digit_sum h m s ∧
      time_digit_sum h m s = 33 :=
sorry

end max_digit_sum_in_range_l2946_294686


namespace unique_three_digit_number_divisible_by_3_and_7_l2946_294603

/-- Represents a three-digit number in the form A3B -/
def ThreeDigitNumber (a b : Nat) : Nat :=
  100 * a + 30 + b

theorem unique_three_digit_number_divisible_by_3_and_7 :
  ∀ a b : Nat,
    (300 < ThreeDigitNumber a b) →
    (ThreeDigitNumber a b < 400) →
    (ThreeDigitNumber a b % 3 = 0) →
    (ThreeDigitNumber a b % 7 = 0) →
    b = 6 := by
  sorry

end unique_three_digit_number_divisible_by_3_and_7_l2946_294603


namespace fox_initial_coins_l2946_294600

def bridge_crossings (initial_coins : ℕ) : ℕ := 
  let after_first := 2 * initial_coins - 50
  let after_second := 2 * after_first - 50
  let after_third := 2 * after_second - 50
  2 * after_third - 50

theorem fox_initial_coins : 
  ∃ (x : ℕ), bridge_crossings x = 0 ∧ x = 47 :=
sorry

end fox_initial_coins_l2946_294600


namespace inequality_of_means_l2946_294644

theorem inequality_of_means (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a + b + c) / 3 > (a * b * c) ^ (1/3) ∧ (a * b * c) ^ (1/3) > 3 * a * b * c / (a * b + b * c + c * a) :=
sorry

end inequality_of_means_l2946_294644


namespace intersection_complement_eq_interval_l2946_294689

open Set

/-- Given sets A and B, prove that their intersection with the complement of B is [1, 3) -/
theorem intersection_complement_eq_interval :
  let A : Set ℝ := {x | x - 1 ≥ 0}
  let B : Set ℝ := {x | 3 / x ≤ 1}
  A ∩ (univ \ B) = Icc 1 3 ∩ Iio 3 := by sorry

end intersection_complement_eq_interval_l2946_294689


namespace extreme_values_and_monotonicity_l2946_294681

-- Define the function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f(x)
def f_derivative (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_values_and_monotonicity 
  (a b c : ℝ) :
  (f_derivative a b (-2/3) = 0 ∧ f_derivative a b 1 = 0) →
  (a = -1/2 ∧ b = -2) ∧
  (∀ x, -2/3 < x ∧ x < 1 → f_derivative (-1/2) (-2) x < 0) ∧
  (∀ x, (x < -2/3 ∨ 1 < x) → f_derivative (-1/2) (-2) x > 0) :=
by sorry

end extreme_values_and_monotonicity_l2946_294681


namespace grade_assignment_count_l2946_294633

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- The number of different grades -/
def num_grades : ℕ := 4

/-- Theorem: The number of ways to assign grades to students -/
theorem grade_assignment_count : (num_grades : ℕ) ^ num_students = 16777216 := by
  sorry

end grade_assignment_count_l2946_294633


namespace kite_smallest_angle_l2946_294606

/-- Represents the angles of a kite in degrees -/
structure KiteAngles where
  a : ℝ  -- smallest angle
  d : ℝ  -- common difference

/-- Conditions for a valid kite with angles in arithmetic sequence -/
def is_valid_kite (k : KiteAngles) : Prop :=
  k.a > 0 ∧ 
  k.a + k.d > 0 ∧ 
  k.a + 2*k.d > 0 ∧ 
  k.a + 3*k.d > 0 ∧
  k.a + (k.a + 3*k.d) = 180 ∧  -- opposite angles are supplementary
  k.a + 3*k.d = 150  -- largest angle is 150°

theorem kite_smallest_angle (k : KiteAngles) (h : is_valid_kite k) : k.a = 15 := by
  sorry

end kite_smallest_angle_l2946_294606


namespace ceiling_negative_seven_fourths_squared_l2946_294632

theorem ceiling_negative_seven_fourths_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end ceiling_negative_seven_fourths_squared_l2946_294632


namespace division_ratio_l2946_294662

theorem division_ratio (dividend quotient divisor remainder : ℕ) : 
  dividend = 5290 →
  remainder = 46 →
  divisor = 10 * quotient →
  dividend = divisor * quotient + remainder →
  (divisor : ℚ) / remainder = 5 := by
  sorry

end division_ratio_l2946_294662


namespace probability_one_unit_apart_l2946_294647

/-- A rectangle with dimensions 3 × 2 -/
structure Rectangle :=
  (length : ℕ := 3)
  (width : ℕ := 2)

/-- Evenly spaced points on the perimeter of the rectangle -/
def PerimeterPoints (r : Rectangle) : ℕ := 15

/-- Number of unit intervals on the perimeter -/
def UnitIntervals (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The probability of selecting two points one unit apart -/
def ProbabilityOneUnitApart (r : Rectangle) : ℚ :=
  16 / (PerimeterPoints r).choose 2

theorem probability_one_unit_apart (r : Rectangle) :
  ProbabilityOneUnitApart r = 16 / 105 := by
  sorry

end probability_one_unit_apart_l2946_294647


namespace greatest_fourth_term_l2946_294683

/-- An arithmetic sequence of five positive integers with sum 60 -/
structure ArithmeticSequence where
  a : ℕ+  -- first term
  d : ℕ+  -- common difference
  sum_eq_60 : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 60

/-- The fourth term of an arithmetic sequence -/
def fourth_term (seq : ArithmeticSequence) : ℕ := seq.a + 3 * seq.d

/-- The greatest possible fourth term is 34 -/
theorem greatest_fourth_term :
  ∀ seq : ArithmeticSequence, fourth_term seq ≤ 34 ∧ 
  ∃ seq : ArithmeticSequence, fourth_term seq = 34 :=
sorry

end greatest_fourth_term_l2946_294683


namespace arithmetic_sequence_11th_term_l2946_294699

/-- An arithmetic sequence with a₁ = 1 and aₙ₊₂ - aₙ = 6 -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 2) - a n = 6

/-- The 11th term of the arithmetic sequence is 31 -/
theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ) (h : arithmeticSequence a) : a 11 = 31 := by
  sorry

end arithmetic_sequence_11th_term_l2946_294699


namespace meeting_at_64th_light_l2946_294646

/-- Represents the meeting point of Petya and Vasya on a street with streetlights -/
def meeting_point (total_lights : ℕ) (petya_start : ℕ) (vasya_start : ℕ) 
                  (petya_position : ℕ) (vasya_position : ℕ) : ℕ :=
  let total_intervals := total_lights - 1
  let petya_intervals := petya_position - petya_start
  let vasya_intervals := vasya_start - vasya_position
  let total_covered := petya_intervals + vasya_intervals
  petya_start + (petya_intervals * 3)

theorem meeting_at_64th_light :
  meeting_point 100 1 100 22 88 = 64 := by
  sorry

#eval meeting_point 100 1 100 22 88

end meeting_at_64th_light_l2946_294646


namespace average_of_four_digits_l2946_294611

theorem average_of_four_digits 
  (total_digits : Nat)
  (total_average : ℚ)
  (five_digits : Nat)
  (five_average : ℚ)
  (h1 : total_digits = 9)
  (h2 : total_average = 18)
  (h3 : five_digits = 5)
  (h4 : five_average = 26)
  : (total_digits * total_average - five_digits * five_average) / (total_digits - five_digits) = 8 := by
  sorry

end average_of_four_digits_l2946_294611


namespace fibonacci_identity_l2946_294620

def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_identity (n : ℕ) (h : n ≥ 1) :
  fib (n - 1) * fib (n + 1) - fib n ^ 2 = (-1) ^ n := by
  sorry

end fibonacci_identity_l2946_294620


namespace trivia_team_absentees_l2946_294628

theorem trivia_team_absentees (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  total_members = 5 → 
  points_per_member = 6 → 
  total_points = 18 → 
  total_members - (total_points / points_per_member) = 2 :=
by sorry

end trivia_team_absentees_l2946_294628


namespace laptop_price_l2946_294669

/-- Given that 20% of a price is $240, prove that the full price is $1200 -/
theorem laptop_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (full_price : ℝ) 
  (h1 : upfront_payment = 240)
  (h2 : upfront_percentage = 20)
  (h3 : upfront_payment = upfront_percentage / 100 * full_price) : 
  full_price = 1200 := by
  sorry

#check laptop_price

end laptop_price_l2946_294669


namespace unique_number_property_l2946_294682

theorem unique_number_property : ∃! x : ℝ, 3 * x = x + 18 := by sorry

end unique_number_property_l2946_294682


namespace special_multiplication_l2946_294652

theorem special_multiplication (a b : ℤ) :
  (∀ x y, x * y = 5*x + 2*y - 1) → (-4) * 6 = -9 := by
  sorry

end special_multiplication_l2946_294652


namespace opposite_of_negative_five_is_five_l2946_294651

theorem opposite_of_negative_five_is_five : 
  -(- 5) = 5 := by sorry

end opposite_of_negative_five_is_five_l2946_294651


namespace square_stack_area_l2946_294612

theorem square_stack_area (blue_exposed red_exposed yellow_exposed : ℝ) 
  (h1 : blue_exposed = 25)
  (h2 : red_exposed = 19)
  (h3 : yellow_exposed = 11) :
  let blue_side := Real.sqrt blue_exposed
  let red_uncovered := red_exposed / blue_side
  let large_side := blue_side + red_uncovered
  large_side ^ 2 = 64 := by
  sorry

end square_stack_area_l2946_294612


namespace davids_english_marks_l2946_294604

/-- Represents the marks obtained in each subject --/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculate the average of marks --/
def average (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.chemistry + m.biology) / 5

theorem davids_english_marks :
  ∃ m : Marks,
    m.mathematics = 85 ∧
    m.physics = 82 ∧
    m.chemistry = 87 ∧
    m.biology = 85 ∧
    average m = 85 ∧
    m.english = 86 := by
  sorry

#check davids_english_marks

end davids_english_marks_l2946_294604


namespace a_4_equals_8_l2946_294692

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ :=
  if n = 0 then S 0
  else S n - S (n-1)

theorem a_4_equals_8 : a 4 = 8 := by sorry

end a_4_equals_8_l2946_294692


namespace combination_permutation_inequality_l2946_294687

theorem combination_permutation_inequality (n : ℕ+) : 
  2 * Nat.choose n 3 ≤ n * (n - 1) ↔ 3 ≤ n ∧ n ≤ 5 := by
  sorry

end combination_permutation_inequality_l2946_294687


namespace nickels_count_l2946_294649

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Calculates the number of nickels given the total value and the number of pennies and dimes -/
def calculate_nickels (total_value : ℕ) (num_pennies : ℕ) (num_dimes : ℕ) : ℕ :=
  let pennies_value := num_pennies * penny_value
  let dimes_value := num_dimes * dime_value
  let nickels_value := total_value - pennies_value - dimes_value
  nickels_value / nickel_value

theorem nickels_count (total_value : ℕ) (num_pennies : ℕ) (num_dimes : ℕ)
    (h1 : total_value = 59)
    (h2 : num_pennies = 9)
    (h3 : num_dimes = 3) :
    calculate_nickels total_value num_pennies num_dimes = 4 := by
  sorry

end nickels_count_l2946_294649


namespace hockey_team_ties_l2946_294617

theorem hockey_team_ties (total_points : ℕ) (win_tie_difference : ℕ) : 
  total_points = 60 → win_tie_difference = 12 → 
  ∃ (ties wins : ℕ), 
    ties + wins = total_points ∧ 
    wins = ties + win_tie_difference ∧
    2 * wins + ties = total_points ∧
    ties = 12 := by
  sorry

end hockey_team_ties_l2946_294617


namespace candles_from_leftovers_l2946_294621

/-- Represents the number of candles of a certain size --/
structure CandleSet where
  count : ℕ
  size : ℚ

/-- Calculates the total wax from a set of candles --/
def waxFrom (cs : CandleSet) (leftoverRatio : ℚ) : ℚ :=
  cs.count * cs.size * leftoverRatio

/-- The main theorem --/
theorem candles_from_leftovers 
  (leftoverRatio : ℚ)
  (bigCandles smallCandles tinyCandles : CandleSet)
  (newCandleSize : ℚ)
  (h_leftover : leftoverRatio = 1/10)
  (h_big : bigCandles = ⟨5, 20⟩)
  (h_small : smallCandles = ⟨5, 5⟩)
  (h_tiny : tinyCandles = ⟨25, 1⟩)
  (h_new : newCandleSize = 5) :
  (waxFrom bigCandles leftoverRatio + 
   waxFrom smallCandles leftoverRatio + 
   waxFrom tinyCandles leftoverRatio) / newCandleSize = 3 := by
  sorry

end candles_from_leftovers_l2946_294621


namespace cook_sane_cheshire_cat_insane_l2946_294605

/-- Represents the sanity status of an individual -/
inductive Sanity
| Sane
| Insane

/-- Represents the characters in the problem -/
inductive Character
| Cook
| CheshireCat

/-- The cook's assertion about the sanity of the characters -/
def cooksAssertion (sanityStatus : Character → Sanity) : Prop :=
  sanityStatus Character.Cook = Sanity.Insane ∨ sanityStatus Character.CheshireCat = Sanity.Insane

/-- The main theorem to prove -/
theorem cook_sane_cheshire_cat_insane :
  ∃ (sanityStatus : Character → Sanity),
    cooksAssertion sanityStatus ∧
    sanityStatus Character.Cook = Sanity.Sane ∧
    sanityStatus Character.CheshireCat = Sanity.Insane :=
sorry

end cook_sane_cheshire_cat_insane_l2946_294605


namespace potato_flour_weight_l2946_294664

theorem potato_flour_weight (potato_bags flour_bags total_weight weight_difference : ℕ) 
  (h1 : potato_bags = 15)
  (h2 : flour_bags = 12)
  (h3 : total_weight = 1710)
  (h4 : weight_difference = 30) :
  ∃ (potato_weight flour_weight : ℕ),
    potato_weight * potato_bags + flour_weight * flour_bags = total_weight ∧
    flour_weight = potato_weight + weight_difference ∧
    potato_weight = 50 ∧
    flour_weight = 80 := by
  sorry

end potato_flour_weight_l2946_294664


namespace unit_circle_representation_l2946_294693

theorem unit_circle_representation (x y : ℝ) (n : ℤ) :
  (Real.arcsin x + Real.arccos y = n * π) →
  ((n = 0 → x^2 + y^2 = 1 ∧ x ≤ 0 ∧ y ≥ 0) ∧
   (n = 1 → x^2 + y^2 = 1 ∧ x ≥ 0 ∧ y ≤ 0)) :=
by sorry

end unit_circle_representation_l2946_294693


namespace exists_participation_to_invalidate_forecast_l2946_294626

/-- Represents a voter in the election -/
structure Voter :=
  (id : Nat)
  (isCandidate : Bool)
  (friends : Set Nat)

/-- Represents a forecast for the election -/
def Forecast := Nat → Nat

/-- Represents the actual votes cast in the election -/
def ActualVotes := Nat → Nat

/-- Determines if a voter participates in the election -/
def VoterParticipation := Nat → Bool

/-- Calculates the actual votes based on voter participation -/
def calculateActualVotes (voters : List Voter) (participation : VoterParticipation) : ActualVotes :=
  sorry

/-- Checks if a forecast is good (correct for at least one candidate) -/
def isGoodForecast (forecast : Forecast) (actualVotes : ActualVotes) : Bool :=
  sorry

/-- Main theorem: For any forecast, there exists a voter participation that makes the forecast not good -/
theorem exists_participation_to_invalidate_forecast (voters : List Voter) (forecast : Forecast) :
  ∃ (participation : VoterParticipation),
    ¬(isGoodForecast forecast (calculateActualVotes voters participation)) :=
  sorry

end exists_participation_to_invalidate_forecast_l2946_294626


namespace jenny_game_ratio_l2946_294660

theorem jenny_game_ratio : 
  ∀ (games_against_mark games_against_jill games_jenny_won : ℕ)
    (mark_wins jill_win_percentage : ℚ),
    games_against_mark = 10 →
    mark_wins = 1 →
    jill_win_percentage = 3/4 →
    games_jenny_won = 14 →
    games_against_jill = (games_jenny_won - (games_against_mark - mark_wins)) / (1 - jill_win_percentage) →
    games_against_jill / games_against_mark = 2 := by
  sorry

end jenny_game_ratio_l2946_294660


namespace equation_solutions_l2946_294640

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1)^3 * (x - 2)^3 * (x - 3)^3 * (x - 4)^3 / ((x - 2) * (x - 4) * (x - 2)^2)
  ∀ x : ℝ, f x = 64 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
by sorry

end equation_solutions_l2946_294640


namespace number_problem_l2946_294635

theorem number_problem (a b : ℤ) : 
  a + b = 72 → 
  a = b + 12 → 
  a = 42 → 
  b = 30 := by
sorry

end number_problem_l2946_294635


namespace john_arcade_spend_l2946_294675

/-- The amount of money John spent at the arcade -/
def arcade_spend (total_time minutes_per_break num_breaks cost_per_interval minutes_per_interval : ℕ) : ℚ :=
  let total_minutes := total_time
  let break_minutes := minutes_per_break * num_breaks
  let playing_minutes := total_minutes - break_minutes
  let num_intervals := playing_minutes / minutes_per_interval
  (num_intervals : ℚ) * cost_per_interval

theorem john_arcade_spend :
  arcade_spend 275 10 5 (3/4) 5 = 33.75 := by
  sorry

end john_arcade_spend_l2946_294675


namespace range_of_a_for_quadratic_inequality_l2946_294623

theorem range_of_a_for_quadratic_inequality :
  ∃ (a : ℝ), ∀ (x : ℝ), x^2 + 2*x + a > 0 ↔ a ∈ Set.Ioi 1 :=
by sorry

end range_of_a_for_quadratic_inequality_l2946_294623


namespace A_B_white_mutually_exclusive_l2946_294684

/-- Represents a person who can receive a ball -/
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person

/-- Represents a ball color -/
inductive BallColor : Type
  | Red : BallColor
  | Black : BallColor
  | White : BallColor

/-- Represents a distribution of balls to people -/
def Distribution := Person → BallColor

/-- The event that person A receives the white ball -/
def A_receives_white (d : Distribution) : Prop := d Person.A = BallColor.White

/-- The event that person B receives the white ball -/
def B_receives_white (d : Distribution) : Prop := d Person.B = BallColor.White

/-- Each person receives exactly one ball -/
def valid_distribution (d : Distribution) : Prop :=
  ∀ (c : BallColor), ∃! (p : Person), d p = c

theorem A_B_white_mutually_exclusive :
  ∀ (d : Distribution), valid_distribution d →
    ¬(A_receives_white d ∧ B_receives_white d) :=
sorry

end A_B_white_mutually_exclusive_l2946_294684


namespace commission_percentage_l2946_294629

/-- Proves that the commission percentage for the first $500 is 20% given the conditions --/
theorem commission_percentage (x : ℝ) : 
  let total_sale := 800
  let commission_over_500 := 0.25
  let total_commission_percentage := 0.21875
  (x / 100 * 500 + commission_over_500 * (total_sale - 500)) / total_sale = total_commission_percentage →
  x = 20 := by
sorry

end commission_percentage_l2946_294629


namespace min_points_on_circle_l2946_294630

/-- A type representing a point in a plane -/
def Point : Type := ℝ × ℝ

/-- A type representing a circle in a plane -/
def Circle : Type := Point × ℝ

/-- Check if a point lies on a circle -/
def lies_on (p : Point) (c : Circle) : Prop :=
  let (center, radius) := c
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

/-- Check if four points are concyclic (lie on the same circle) -/
def are_concyclic (p1 p2 p3 p4 : Point) : Prop :=
  ∃ c : Circle, lies_on p1 c ∧ lies_on p2 c ∧ lies_on p3 c ∧ lies_on p4 c

/-- Main theorem -/
theorem min_points_on_circle 
  (points : Finset Point) 
  (h_card : points.card = 10)
  (h_concyclic : ∀ (s : Finset Point), s ⊆ points → s.card = 5 → 
    ∃ (t : Finset Point), t ⊆ s ∧ t.card = 4 ∧ 
    ∃ (p1 p2 p3 p4 : Point), p1 ∈ t ∧ p2 ∈ t ∧ p3 ∈ t ∧ p4 ∈ t ∧ 
    are_concyclic p1 p2 p3 p4) : 
  ∃ (c : Circle) (s : Finset Point), s ⊆ points ∧ s.card = 9 ∧ 
  ∀ p ∈ s, lies_on p c :=
sorry

end min_points_on_circle_l2946_294630


namespace change_in_average_l2946_294614

def scores : List ℝ := [89, 85, 91, 87, 82]

theorem change_in_average (scores : List ℝ) : 
  scores = [89, 85, 91, 87, 82] →
  (scores.sum / scores.length) - ((scores.take 4).sum / 4) = -1.2 := by
  sorry

end change_in_average_l2946_294614


namespace sum_of_first_10_common_elements_l2946_294642

/-- Arithmetic progression with first term 4 and common difference 3 -/
def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

/-- Geometric progression with first term 20 and common ratio 2 -/
def geometric_progression (k : ℕ) : ℕ := 20 * 2^k

/-- Common elements between the arithmetic and geometric progressions -/
def common_elements (n : ℕ) : Prop :=
  ∃ k : ℕ, arithmetic_progression n = geometric_progression k

/-- The sum of the first 10 common elements -/
def sum_of_common_elements : ℕ := 13981000

/-- Theorem stating that the sum of the first 10 common elements is 13981000 -/
theorem sum_of_first_10_common_elements :
  sum_of_common_elements = 13981000 := by sorry

end sum_of_first_10_common_elements_l2946_294642


namespace expense_ratios_l2946_294634

def initial_amount : ℚ := 120
def books_expense : ℚ := 25
def clothes_expense : ℚ := 40
def snacks_expense : ℚ := 10

def total_spent : ℚ := books_expense + clothes_expense + snacks_expense

theorem expense_ratios :
  (books_expense / total_spent = 1 / 3) ∧
  (clothes_expense / total_spent = 4 / 3) ∧
  (snacks_expense / total_spent = 2 / 15) := by
  sorry

end expense_ratios_l2946_294634


namespace donovan_percentage_l2946_294608

/-- Calculates the weighted percentage of correct answers for a math test -/
def weighted_percentage (
  mc_total : ℕ) (mc_correct : ℕ) (mc_points : ℕ)
  (sa_total : ℕ) (sa_correct : ℕ) (sa_partial : ℕ) (sa_points : ℕ)
  (essay_total : ℕ) (essay_correct : ℕ) (essay_points : ℕ) : ℚ :=
  let total_possible := mc_total * mc_points + sa_total * sa_points + essay_total * essay_points
  let total_earned := mc_correct * mc_points + sa_correct * sa_points + sa_partial * (sa_points / 2) + essay_correct * essay_points
  (total_earned : ℚ) / total_possible * 100

/-- Theorem stating that Donovan's weighted percentage is 68.75% -/
theorem donovan_percentage :
  weighted_percentage 25 20 2 20 10 5 4 3 2 10 = 68.75 := by
  sorry

end donovan_percentage_l2946_294608


namespace polynomial_symmetry_l2946_294631

theorem polynomial_symmetry (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + c * x + 7
  (f (-2011) = -17) → (f 2011 = 31) := by
sorry

end polynomial_symmetry_l2946_294631


namespace parallel_lines_reasoning_is_deductive_l2946_294645

-- Define the types of reasoning
inductive ReasoningType
  | Deductive
  | Analogical

-- Define the characteristics of different types of reasoning
def isDeductive (r : ReasoningType) : Prop :=
  r = ReasoningType.Deductive

def isGeneralToSpecific (r : ReasoningType) : Prop :=
  r = ReasoningType.Deductive

-- Define the geometric concept
def sameSideInteriorAngles (a b : ℝ) : Prop :=
  a + b = 180

-- Theorem statement
theorem parallel_lines_reasoning_is_deductive :
  ∀ (A B : ℝ) (r : ReasoningType),
    sameSideInteriorAngles A B →
    isGeneralToSpecific r →
    isDeductive r :=
by
  sorry

end parallel_lines_reasoning_is_deductive_l2946_294645


namespace triangle_side_length_l2946_294695

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2asinB = √3 * b, b + c = 5, and bc = 6, then a = √7 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π/2 →  -- ABC is acute
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  2 * a * Real.sin B = Real.sqrt 3 * b →
  b + c = 5 →
  b * c = 6 →
  a = Real.sqrt 7 := by
sorry

end triangle_side_length_l2946_294695


namespace intersection_of_A_and_B_l2946_294618

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 2 := by
  sorry

end intersection_of_A_and_B_l2946_294618


namespace windows_preference_l2946_294624

theorem windows_preference (total : ℕ) (mac_pref : ℕ) (no_pref : ℕ) 
  (h1 : total = 210)
  (h2 : mac_pref = 60)
  (h3 : no_pref = 90) :
  total - mac_pref - (mac_pref / 3) - no_pref = 40 := by
  sorry

#check windows_preference

end windows_preference_l2946_294624


namespace triangle_perimeter_l2946_294659

theorem triangle_perimeter (a b c : ℝ) (ha : a = 28) (hb : b = 16) (hc : c = 18) :
  a + b + c = 62 := by
  sorry

end triangle_perimeter_l2946_294659

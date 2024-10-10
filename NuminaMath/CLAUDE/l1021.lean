import Mathlib

namespace equal_consecutive_subgroup_exists_l1021_102152

/-- A person can be either of type A or type B -/
inductive PersonType
| A
| B

/-- A circular arrangement of people -/
def CircularArrangement := List PersonType

/-- Count the number of type A persons in a list -/
def countTypeA : List PersonType → Nat
| [] => 0
| (PersonType.A :: rest) => 1 + countTypeA rest
| (_ :: rest) => countTypeA rest

/-- Take n consecutive elements from a circular list, starting from index i -/
def takeCircular (n : Nat) (i : Nat) (l : List α) : List α :=
  (List.drop i l ++ l).take n

/-- Main theorem -/
theorem equal_consecutive_subgroup_exists (arrangement : CircularArrangement) 
    (h1 : arrangement.length = 8)
    (h2 : countTypeA arrangement = 4) :
    ∃ i, countTypeA (takeCircular 4 i arrangement) = 2 := by
  sorry

end equal_consecutive_subgroup_exists_l1021_102152


namespace solution_set_when_a_eq_1_range_when_a_ge_1_l1021_102176

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 2|

-- Part 1: Solution set when a = 1
theorem solution_set_when_a_eq_1 :
  {x : ℝ | f 1 x ≤ 5} = Set.Icc (-3) 2 := by sorry

-- Part 2: Range of f(x) when a ≥ 1
theorem range_when_a_ge_1 (a : ℝ) (h : a ≥ 1) :
  Set.range (f a) = Set.Ici (a + 2) := by sorry

end solution_set_when_a_eq_1_range_when_a_ge_1_l1021_102176


namespace elevator_weight_problem_l1021_102134

theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℚ) (new_person_weight : ℚ) :
  initial_people = 6 →
  initial_avg_weight = 160 →
  new_person_weight = 97 →
  let total_weight : ℚ := initial_people * initial_avg_weight + new_person_weight
  let new_people : ℕ := initial_people + 1
  let new_avg_weight : ℚ := total_weight / new_people
  new_avg_weight = 151 := by sorry

end elevator_weight_problem_l1021_102134


namespace history_score_calculation_l1021_102109

theorem history_score_calculation (math_score : ℝ) (third_subject_score : ℝ) (desired_average : ℝ) :
  math_score = 74 →
  third_subject_score = 67 →
  desired_average = 75 →
  (math_score + third_subject_score + (3 * desired_average - math_score - third_subject_score)) / 3 = desired_average :=
by
  sorry

#check history_score_calculation

end history_score_calculation_l1021_102109


namespace number_problem_l1021_102101

theorem number_problem : ∃ x : ℝ, 4 * x = 28 ∧ x = 7 := by
  sorry

end number_problem_l1021_102101


namespace molecular_weight_X_l1021_102115

/-- Given a compound Ba(X)₂ with total molecular weight 171 and Ba having
    molecular weight 137, prove that the molecular weight of X is 17. -/
theorem molecular_weight_X (total_weight : ℝ) (ba_weight : ℝ) (x_weight : ℝ) :
  total_weight = 171 →
  ba_weight = 137 →
  total_weight = ba_weight + 2 * x_weight →
  x_weight = 17 := by
sorry

end molecular_weight_X_l1021_102115


namespace fruit_platter_kiwis_l1021_102197

theorem fruit_platter_kiwis 
  (total : ℕ) 
  (oranges apples bananas kiwis : ℕ) 
  (h_total : oranges + apples + bananas + kiwis = total)
  (h_apples : apples = 3 * oranges)
  (h_bananas : bananas = 4 * apples)
  (h_kiwis : kiwis = 5 * bananas)
  (h_total_value : total = 540) :
  kiwis = 420 := by
  sorry

end fruit_platter_kiwis_l1021_102197


namespace yellow_second_probability_l1021_102111

/-- Represents the contents of a bag of marbles -/
structure BagContents where
  red : ℕ
  black : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the probability of drawing a yellow marble second given the bag contents and rules -/
def probability_yellow_second (bag_a bag_b bag_c : BagContents) : ℚ :=
  let total_a := bag_a.red + bag_a.black
  let total_b := bag_b.yellow + bag_b.blue
  let total_c := bag_c.yellow + bag_c.blue
  let prob_red_a := bag_a.red / total_a
  let prob_black_a := bag_a.black / total_a
  let prob_yellow_b := bag_b.yellow / total_b
  let prob_yellow_c := bag_c.yellow / total_c
  prob_red_a * prob_yellow_b + prob_black_a * prob_yellow_c

/-- Theorem stating that the probability of drawing a yellow marble second is 1/3 -/
theorem yellow_second_probability :
  let bag_a : BagContents := { red := 3, black := 6, yellow := 0, blue := 0 }
  let bag_b : BagContents := { red := 0, black := 0, yellow := 6, blue := 4 }
  let bag_c : BagContents := { red := 0, black := 0, yellow := 2, blue := 8 }
  probability_yellow_second bag_a bag_b bag_c = 1/3 := by
  sorry


end yellow_second_probability_l1021_102111


namespace equation_root_of_increase_implies_m_equals_two_l1021_102182

-- Define the equation
def equation (x m : ℝ) : Prop := (x - 1) / (x - 3) = m / (x - 3)

-- Define the root of increase
def root_of_increase (x : ℝ) : Prop := x = 3

-- Theorem statement
theorem equation_root_of_increase_implies_m_equals_two :
  ∀ x m : ℝ, equation x m → root_of_increase x → m = 2 := by
  sorry

end equation_root_of_increase_implies_m_equals_two_l1021_102182


namespace mikaela_paint_containers_l1021_102139

/-- Represents the number of paint containers Mikaela initially bought. -/
def initial_containers : ℕ := 8

/-- Represents the number of walls Mikaela initially planned to paint. -/
def planned_walls : ℕ := 4

/-- Represents the number of containers used for the ceiling. -/
def ceiling_containers : ℕ := 1

/-- Represents the number of containers left over. -/
def leftover_containers : ℕ := 3

/-- Represents the number of walls Mikaela actually painted. -/
def painted_walls : ℕ := 3

theorem mikaela_paint_containers :
  initial_containers = 
    ceiling_containers + leftover_containers + (planned_walls - painted_walls) :=
by sorry

end mikaela_paint_containers_l1021_102139


namespace candy_seller_problem_l1021_102131

theorem candy_seller_problem (num_clowns num_children initial_candies candies_per_person : ℕ) 
  (h1 : num_clowns = 4)
  (h2 : num_children = 30)
  (h3 : initial_candies = 700)
  (h4 : candies_per_person = 20) :
  initial_candies - (num_clowns + num_children) * candies_per_person = 20 := by
  sorry

end candy_seller_problem_l1021_102131


namespace solution_set_nonempty_implies_m_greater_than_five_l1021_102169

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (x m : ℝ) : ℝ := -|x + 4| + m

-- State the theorem
theorem solution_set_nonempty_implies_m_greater_than_five :
  (∃ x : ℝ, f x < g x m) → m > 5 := by
  sorry

end solution_set_nonempty_implies_m_greater_than_five_l1021_102169


namespace jerrys_average_score_l1021_102118

theorem jerrys_average_score (current_average : ℝ) : 
  (3 * current_average + 97) / 4 = current_average + 3 →
  current_average = 85 := by
sorry

end jerrys_average_score_l1021_102118


namespace second_number_is_three_l1021_102135

theorem second_number_is_three (x y : ℝ) 
  (sum_is_ten : x + y = 10) 
  (relation : 2 * x = 3 * y + 5) : 
  y = 3 := by
sorry

end second_number_is_three_l1021_102135


namespace complex_number_second_quadrant_l1021_102142

theorem complex_number_second_quadrant (a : ℝ) : 
  let z : ℂ := (a + 3*Complex.I)/Complex.I + a*Complex.I
  (z.re = 0) ∧ (z.im < 0) ∧ (z.re < 0) → a = -4 := by
  sorry

end complex_number_second_quadrant_l1021_102142


namespace four_solutions_range_l1021_102156

-- Define the function f(x) = |x-2|
def f (x : ℝ) : ℝ := abs (x - 2)

-- Define the proposition
theorem four_solutions_range (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    a * (f x₁)^2 - f x₁ + 1 = 0 ∧
    a * (f x₂)^2 - f x₂ + 1 = 0 ∧
    a * (f x₃)^2 - f x₃ + 1 = 0 ∧
    a * (f x₄)^2 - f x₄ + 1 = 0) →
  0 < a ∧ a < 1/4 :=
by sorry

end four_solutions_range_l1021_102156


namespace tax_rate_on_other_items_l1021_102130

/-- Represents the percentage of total spending on each category -/
structure SpendingPercentages where
  clothing : ℝ
  food : ℝ
  other : ℝ

/-- Represents the tax rates for each category -/
structure TaxRates where
  clothing : ℝ
  food : ℝ
  other : ℝ

/-- Theorem: Given the spending percentages and known tax rates, 
    prove that the tax rate on other items is 8% -/
theorem tax_rate_on_other_items 
  (sp : SpendingPercentages)
  (tr : TaxRates)
  (h1 : sp.clothing = 0.4)
  (h2 : sp.food = 0.3)
  (h3 : sp.other = 0.3)
  (h4 : sp.clothing + sp.food + sp.other = 1)
  (h5 : tr.clothing = 0.04)
  (h6 : tr.food = 0)
  (h7 : sp.clothing * tr.clothing + sp.food * tr.food + sp.other * tr.other = 0.04) :
  tr.other = 0.08 := by
  sorry


end tax_rate_on_other_items_l1021_102130


namespace females_soccer_not_basketball_l1021_102199

/-- Represents the number of students in various categories -/
structure SchoolTeams where
  soccer_males : ℕ
  soccer_females : ℕ
  basketball_males : ℕ
  basketball_females : ℕ
  males_in_both : ℕ
  total_students : ℕ

/-- The theorem to be proved -/
theorem females_soccer_not_basketball (teams : SchoolTeams)
  (h1 : teams.soccer_males = 120)
  (h2 : teams.soccer_females = 60)
  (h3 : teams.basketball_males = 100)
  (h4 : teams.basketball_females = 80)
  (h5 : teams.males_in_both = 70)
  (h6 : teams.total_students = 260) :
  teams.soccer_females - (teams.soccer_females + teams.basketball_females - 
    (teams.total_students - (teams.soccer_males + teams.basketball_males - teams.males_in_both))) = 30 := by
  sorry


end females_soccer_not_basketball_l1021_102199


namespace amazon_profit_per_package_l1021_102105

/-- Profit per package for Amazon distribution centers -/
theorem amazon_profit_per_package :
  ∀ (centers : ℕ)
    (first_center_daily_packages : ℕ)
    (second_center_multiplier : ℕ)
    (combined_weekly_profit : ℚ)
    (days_per_week : ℕ),
  centers = 2 →
  first_center_daily_packages = 10000 →
  second_center_multiplier = 3 →
  combined_weekly_profit = 14000 →
  days_per_week = 7 →
  let total_weekly_packages := first_center_daily_packages * days_per_week * (1 + second_center_multiplier)
  (combined_weekly_profit / total_weekly_packages : ℚ) = 1/20 := by
sorry

end amazon_profit_per_package_l1021_102105


namespace zoo_visitors_l1021_102184

theorem zoo_visitors (adult_price kid_price total_sales num_kids : ℕ) 
  (h1 : adult_price = 28)
  (h2 : kid_price = 12)
  (h3 : total_sales = 3864)
  (h4 : num_kids = 203) :
  ∃ (num_adults : ℕ), 
    adult_price * num_adults + kid_price * num_kids = total_sales ∧
    num_adults + num_kids = 254 := by
  sorry

end zoo_visitors_l1021_102184


namespace children_ticket_price_l1021_102140

/-- The price of an adult ticket in dollars -/
def adult_price : ℚ := 8

/-- The total revenue in dollars -/
def total_revenue : ℚ := 236

/-- The total number of tickets sold -/
def total_tickets : ℕ := 34

/-- The number of adult tickets sold -/
def adult_tickets : ℕ := 12

/-- The price of a children's ticket in dollars -/
def children_price : ℚ := (total_revenue - adult_price * adult_tickets) / (total_tickets - adult_tickets)

theorem children_ticket_price :
  children_price = 6.36 := by sorry

end children_ticket_price_l1021_102140


namespace largest_and_smallest_subsequence_l1021_102117

def original_number : ℕ := 798056132

-- Define a function to check if a number is a valid 5-digit subsequence of the original number
def is_valid_subsequence (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ 
  ∃ (a b c d e : ℕ), 
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    ∃ (i j k l m : ℕ), 
      i < j ∧ j < k ∧ k < l ∧ l < m ∧
      (original_number / 10 ^ (8 - i) % 10 = a) ∧
      (original_number / 10 ^ (8 - j) % 10 = b) ∧
      (original_number / 10 ^ (8 - k) % 10 = c) ∧
      (original_number / 10 ^ (8 - l) % 10 = d) ∧
      (original_number / 10 ^ (8 - m) % 10 = e)

theorem largest_and_smallest_subsequence :
  (∀ n : ℕ, is_valid_subsequence n → n ≤ 98632) ∧
  (∀ n : ℕ, is_valid_subsequence n → n ≥ 56132) ∧
  is_valid_subsequence 98632 ∧
  is_valid_subsequence 56132 := by sorry

end largest_and_smallest_subsequence_l1021_102117


namespace circumcircles_intersect_at_single_point_l1021_102168

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a triangle
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Define a circle
structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define central symmetry
def centrally_symmetric (t1 t2 : Triangle) (center : Point) : Prop :=
  ∃ (O : Point),
    (t1.A.x + t2.A.x) / 2 = O.x ∧ (t1.A.y + t2.A.y) / 2 = O.y ∧
    (t1.B.x + t2.B.x) / 2 = O.x ∧ (t1.B.y + t2.B.y) / 2 = O.y ∧
    (t1.C.x + t2.C.x) / 2 = O.x ∧ (t1.C.y + t2.C.y) / 2 = O.y

-- Define circumcircle
def circumcircle (t : Triangle) : Circle :=
  sorry

-- Define intersection of circles
def intersect (c1 c2 : Circle) : Set Point :=
  sorry

theorem circumcircles_intersect_at_single_point
  (ABC A₁B₁C₁ : Triangle)
  (h : centrally_symmetric ABC A₁B₁C₁ (Point.mk 0 0)) :
  ∃ (S : Point),
    S ∈ intersect (circumcircle ABC) (circumcircle (Triangle.mk A₁B₁C₁.A ABC.B A₁B₁C₁.C)) ∧
    S ∈ intersect (circumcircle (Triangle.mk A₁B₁C₁.A A₁B₁C₁.B ABC.C)) (circumcircle (Triangle.mk ABC.A A₁B₁C₁.B A₁B₁C₁.C)) :=
sorry

end circumcircles_intersect_at_single_point_l1021_102168


namespace evaluate_expression_l1021_102196

theorem evaluate_expression (b : ℝ) (h : b = 2) : (6*b^2 - 15*b + 7)*(3*b - 4) = 2 := by
  sorry

end evaluate_expression_l1021_102196


namespace riyas_speed_l1021_102167

/-- Proves that Riya's speed is 21 kmph given the problem conditions -/
theorem riyas_speed (riya_speed priya_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  priya_speed = 22 →
  time = 1 →
  distance = 43 →
  distance = (riya_speed + priya_speed) * time →
  riya_speed = 21 := by
  sorry

#check riyas_speed

end riyas_speed_l1021_102167


namespace gcf_48_72_l1021_102187

theorem gcf_48_72 : Nat.gcd 48 72 = 24 := by
  sorry

end gcf_48_72_l1021_102187


namespace upstream_speed_calculation_l1021_102141

/-- Calculates the upstream speed of a man given his downstream speed and the stream speed. -/
def upstreamSpeed (downstreamSpeed streamSpeed : ℝ) : ℝ :=
  downstreamSpeed - 2 * streamSpeed

/-- Theorem stating that given a downstream speed of 10 kmph and a stream speed of 1 kmph, 
    the upstream speed is 8 kmph. -/
theorem upstream_speed_calculation :
  upstreamSpeed 10 1 = 8 := by
  sorry

end upstream_speed_calculation_l1021_102141


namespace triangle_properties_l1021_102162

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : t.a * t.c * Real.sin t.B = t.b^2 - (t.a - t.c)^2) :
  Real.sin t.B = 4/5 ∧ 
  (∀ (x y : ℝ), x > 0 → y > 0 → x^2 / (x^2 + y^2) ≥ 2/5) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 / (x^2 + y^2) = 2/5) := by
  sorry

end triangle_properties_l1021_102162


namespace solution_set_of_inequality_l1021_102179

theorem solution_set_of_inequality (x : ℝ) :
  {x : ℝ | 4 * x^2 - 4 * x + 1 ≤ 0} = {1/2} := by
  sorry

end solution_set_of_inequality_l1021_102179


namespace library_book_loan_l1021_102150

theorem library_book_loan (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : return_rate = 65 / 100)
  (h3 : final_books = 54)
  : ∃ (loaned_books : ℕ), loaned_books = 60 ∧ 
    (initial_books - final_books : ℚ) = (1 - return_rate) * loaned_books := by
  sorry

end library_book_loan_l1021_102150


namespace base_10_256_to_base_4_l1021_102194

-- Define a function to convert a natural number to its base 4 representation
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

-- State the theorem
theorem base_10_256_to_base_4 :
  toBase4 256 = [1, 0, 0, 0, 0] := by
  sorry

end base_10_256_to_base_4_l1021_102194


namespace cubic_root_sum_l1021_102183

theorem cubic_root_sum (α β γ : ℂ) : 
  (α^3 - α - 1 = 0) → 
  (β^3 - β - 1 = 0) → 
  (γ^3 - γ - 1 = 0) → 
  ((1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = -7) :=
by sorry

end cubic_root_sum_l1021_102183


namespace jessica_mark_earnings_l1021_102126

/-- Given the working hours and earnings of Jessica and Mark, prove that t = 5 --/
theorem jessica_mark_earnings (t : ℝ) : 
  (t + 2) * (4 * t + 1) = (4 * t - 7) * (t + 3) + 4 → t = 5 := by
  sorry

end jessica_mark_earnings_l1021_102126


namespace cherry_pies_profit_independence_l1021_102154

/-- Proves that the number of cherry pies does not affect the profit in Benny's pie sale scenario -/
theorem cherry_pies_profit_independence (num_pumpkin : ℕ) (cost_pumpkin : ℚ) (cost_cherry : ℚ) (sell_price : ℚ) (target_profit : ℚ) :
  num_pumpkin = 10 →
  cost_pumpkin = 3 →
  cost_cherry = 5 →
  sell_price = 5 →
  target_profit = 20 →
  ∀ num_cherry : ℕ,
    sell_price * (num_pumpkin + num_cherry) - (num_pumpkin * cost_pumpkin + num_cherry * cost_cherry) = target_profit :=
by sorry


end cherry_pies_profit_independence_l1021_102154


namespace sector_chord_length_l1021_102121

/-- Given a circular sector with area 1 cm² and perimeter 4 cm, 
    its chord length is 2sin(1) cm. -/
theorem sector_chord_length 
  (r : ℝ) 
  (α : ℝ) 
  (h_area : (1/2) * α * r^2 = 1) 
  (h_perim : 2*r + α*r = 4) : 
  2 * r * Real.sin (α/2) = 2 * Real.sin 1 := by
sorry

end sector_chord_length_l1021_102121


namespace annas_car_rental_cost_l1021_102143

/-- Calculates the total cost of a car rental given the daily rate, per-mile rate, 
    number of days, and miles driven. -/
def carRentalCost (dailyRate : ℚ) (perMileRate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  dailyRate * days + perMileRate * miles

/-- Proves that Anna's car rental cost is $275 given the specified conditions. -/
theorem annas_car_rental_cost :
  carRentalCost 30 0.25 5 500 = 275 := by
  sorry

end annas_car_rental_cost_l1021_102143


namespace simplify_monomial_l1021_102146

theorem simplify_monomial (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end simplify_monomial_l1021_102146


namespace unique_solution_for_B_l1021_102172

theorem unique_solution_for_B : ∃! B : ℕ, ∃ A : ℕ, 
  (A < 10 ∧ B < 10) ∧ (100 * A + 78 - (20 * B + B) = 364) := by
  sorry

end unique_solution_for_B_l1021_102172


namespace arrangements_five_singers_l1021_102153

/-- The number of singers --/
def n : ℕ := 5

/-- The number of different arrangements for n singers with constraints --/
def arrangements (n : ℕ) : ℕ :=
  Nat.factorial (n - 1) + (n - 2) * (n - 2) * Nat.factorial (n - 2)

/-- Theorem: The number of arrangements for 5 singers with constraints is 78 --/
theorem arrangements_five_singers : arrangements n = 78 := by
  sorry

end arrangements_five_singers_l1021_102153


namespace problem_solution_l1021_102133

theorem problem_solution : 
  (3 * Real.sqrt 18 / Real.sqrt 2 + Real.sqrt 12 * Real.sqrt 3 = 15) ∧
  ((2 + Real.sqrt 6)^2 - (Real.sqrt 5 - Real.sqrt 3) * (Real.sqrt 5 + Real.sqrt 3) = 8 + 4 * Real.sqrt 6) := by
  sorry

end problem_solution_l1021_102133


namespace first_player_winning_strategy_l1021_102138

theorem first_player_winning_strategy (a c : ℤ) : ∃ (x y z : ℤ), 
  x^3 + a*x^2 - x + c = 0 ∧ 
  y^3 + a*y^2 - y + c = 0 ∧ 
  z^3 + a*z^2 - z + c = 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z :=
by sorry

end first_player_winning_strategy_l1021_102138


namespace trapezoidal_fence_poles_l1021_102149

/-- Calculates the number of poles needed for a trapezoidal fence --/
theorem trapezoidal_fence_poles
  (parallel_side1 parallel_side2 non_parallel_side : ℕ)
  (parallel_pole_interval non_parallel_pole_interval : ℕ)
  (h1 : parallel_side1 = 60)
  (h2 : parallel_side2 = 80)
  (h3 : non_parallel_side = 50)
  (h4 : parallel_pole_interval = 5)
  (h5 : non_parallel_pole_interval = 7) :
  (parallel_side1 / parallel_pole_interval + 1) +
  (parallel_side2 / parallel_pole_interval + 1) +
  2 * (⌈(non_parallel_side : ℝ) / non_parallel_pole_interval⌉ + 1) - 4 = 44 := by
  sorry

#check trapezoidal_fence_poles

end trapezoidal_fence_poles_l1021_102149


namespace identity_function_satisfies_equation_l1021_102116

theorem identity_function_satisfies_equation (f : ℕ → ℕ) :
  (∀ x y : ℕ, f (x + f y) = f x + y) → (∀ x : ℕ, f x = x) := by
  sorry

end identity_function_satisfies_equation_l1021_102116


namespace max_value_on_ellipse_l1021_102129

/-- The maximum value of x + 2y for points on the ellipse 2x^2 + 3y^2 = 12 is √22 -/
theorem max_value_on_ellipse :
  ∀ x y : ℝ, 2 * x^2 + 3 * y^2 = 12 →
  ∀ z : ℝ, z = x + 2 * y →
  z ≤ Real.sqrt 22 ∧ ∃ x₀ y₀ : ℝ, 2 * x₀^2 + 3 * y₀^2 = 12 ∧ x₀ + 2 * y₀ = Real.sqrt 22 :=
by sorry


end max_value_on_ellipse_l1021_102129


namespace min_teachers_is_ten_l1021_102160

/-- Represents the number of teachers for each subject -/
structure TeacherCounts where
  math : Nat
  physics : Nat
  chemistry : Nat
  biology : Nat
  computerScience : Nat

/-- Represents the school schedule constraints -/
structure SchoolConstraints where
  teacherCounts : TeacherCounts
  maxSubjectsPerTeacher : Nat
  periodsPerDay : Nat

/-- Calculates the total number of teaching slots required per day -/
def totalSlotsPerDay (c : SchoolConstraints) : Nat :=
  (c.teacherCounts.math + c.teacherCounts.physics + c.teacherCounts.chemistry +
   c.teacherCounts.biology + c.teacherCounts.computerScience) * c.periodsPerDay

/-- Calculates the number of slots a single teacher can fill per day -/
def slotsPerTeacher (c : SchoolConstraints) : Nat :=
  c.maxSubjectsPerTeacher * c.periodsPerDay

/-- Calculates the minimum number of teachers required -/
def minTeachersRequired (c : SchoolConstraints) : Nat :=
  (totalSlotsPerDay c + slotsPerTeacher c - 1) / slotsPerTeacher c

/-- The main theorem stating the minimum number of teachers required -/
theorem min_teachers_is_ten (c : SchoolConstraints) :
  c.teacherCounts = { math := 5, physics := 4, chemistry := 4, biology := 4, computerScience := 3 } →
  c.maxSubjectsPerTeacher = 2 →
  c.periodsPerDay = 6 →
  minTeachersRequired c = 10 := by
  sorry


end min_teachers_is_ten_l1021_102160


namespace solution_value_l1021_102178

theorem solution_value (a b : ℝ) : 
  (2 * a + (-1) * b = 1) →
  (2 * b + (-1) * a = 7) →
  (a + b) * (a - b) = -16 := by
sorry

end solution_value_l1021_102178


namespace total_ground_beef_weight_l1021_102198

theorem total_ground_beef_weight (package_weight : ℕ) (butcher1_packages : ℕ) (butcher2_packages : ℕ) (butcher3_packages : ℕ) 
  (h1 : package_weight = 4)
  (h2 : butcher1_packages = 10)
  (h3 : butcher2_packages = 7)
  (h4 : butcher3_packages = 8) :
  package_weight * (butcher1_packages + butcher2_packages + butcher3_packages) = 100 := by
  sorry

end total_ground_beef_weight_l1021_102198


namespace calm_snakes_not_blue_l1021_102145

-- Define the universe of snakes
variable (Snake : Type)

-- Define properties of snakes
variable (isBlue : Snake → Prop)
variable (isCalm : Snake → Prop)
variable (canMultiply : Snake → Prop)
variable (canDivide : Snake → Prop)

-- State the theorem
theorem calm_snakes_not_blue 
  (h1 : ∀ s : Snake, isCalm s → canMultiply s)
  (h2 : ∀ s : Snake, isBlue s → ¬canDivide s)
  (h3 : ∀ s : Snake, ¬canDivide s → ¬canMultiply s) :
  ∀ s : Snake, isCalm s → ¬isBlue s :=
by
  sorry


end calm_snakes_not_blue_l1021_102145


namespace wood_not_heavier_than_brick_l1021_102132

-- Define the mass of the block of wood in kg
def wood_mass_kg : ℝ := 8

-- Define the mass of the brick in g
def brick_mass_g : ℝ := 8000

-- Define the conversion factor from kg to g
def kg_to_g : ℝ := 1000

-- Theorem statement
theorem wood_not_heavier_than_brick : ¬(wood_mass_kg * kg_to_g > brick_mass_g) := by
  sorry

end wood_not_heavier_than_brick_l1021_102132


namespace linear_inequality_solution_l1021_102188

theorem linear_inequality_solution (a : ℝ) : 
  (|2 + 3 * a| = 1) ↔ (a = -1 ∨ a = -1/3) := by
  sorry

end linear_inequality_solution_l1021_102188


namespace complex_conjugate_roots_imply_real_coefficients_l1021_102171

theorem complex_conjugate_roots_imply_real_coefficients (a b : ℝ) :
  (∃ x y : ℝ, y ≠ 0 ∧ 
    (Complex.I * y + x) ^ 2 + (6 + Complex.I * a) * (Complex.I * y + x) + (13 + Complex.I * b) = 0 ∧
    (Complex.I * -y + x) ^ 2 + (6 + Complex.I * a) * (Complex.I * -y + x) + (13 + Complex.I * b) = 0) →
  a = 0 ∧ b = 0 := by
sorry

end complex_conjugate_roots_imply_real_coefficients_l1021_102171


namespace sum_of_a_and_c_l1021_102147

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 30)
  (h2 : b + d = 5) : 
  a + c = 6 := by
sorry

end sum_of_a_and_c_l1021_102147


namespace four_numbers_with_equal_sums_l1021_102114

theorem four_numbers_with_equal_sums (S : Finset ℕ) :
  S ⊆ Finset.range 38 →
  S.card = 10 →
  ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b = c + d :=
by sorry

end four_numbers_with_equal_sums_l1021_102114


namespace estimated_probability_is_0_30_l1021_102113

/-- Represents a single shot result -/
inductive ShotResult
| Hit
| Miss

/-- Represents the result of three shots -/
structure ThreeShotResult :=
  (shot1 shot2 shot3 : ShotResult)

/-- Checks if a ThreeShotResult has exactly two hits -/
def hasTwoHits (result : ThreeShotResult) : Bool :=
  match result with
  | ⟨ShotResult.Hit, ShotResult.Hit, ShotResult.Miss⟩ => true
  | ⟨ShotResult.Hit, ShotResult.Miss, ShotResult.Hit⟩ => true
  | ⟨ShotResult.Miss, ShotResult.Hit, ShotResult.Hit⟩ => true
  | _ => false

/-- Converts a digit to a ShotResult -/
def digitToShotResult (d : Nat) : ShotResult :=
  if d ≤ 3 then ShotResult.Hit else ShotResult.Miss

/-- Converts a three-digit number to a ThreeShotResult -/
def numberToThreeShotResult (n : Nat) : ThreeShotResult :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ⟨digitToShotResult d1, digitToShotResult d2, digitToShotResult d3⟩

/-- The list of simulation results -/
def simulationResults : List Nat :=
  [321, 421, 191, 925, 271, 932, 800, 478, 589, 663,
   531, 297, 396, 021, 546, 388, 230, 113, 507, 965]

/-- Counts the number of ThreeShotResults with exactly two hits -/
def countTwoHits (results : List Nat) : Nat :=
  results.filter (fun n => hasTwoHits (numberToThreeShotResult n)) |>.length

/-- Theorem: The estimated probability of hitting the bullseye exactly twice in three shots is 0.30 -/
theorem estimated_probability_is_0_30 :
  (countTwoHits simulationResults : Rat) / simulationResults.length = 0.30 := by
  sorry


end estimated_probability_is_0_30_l1021_102113


namespace right_triangle_segment_ratio_l1021_102102

theorem right_triangle_segment_ratio 
  (a b c r s : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (height_division : r + s = c) 
  (similarity_relations : a^2 = r * c ∧ b^2 = s * c) 
  (leg_ratio : a / b = 1 / 3) :
  r / s = 1 / 9 := by
  sorry

end right_triangle_segment_ratio_l1021_102102


namespace right_triangle_area_with_specific_ratios_l1021_102177

/-- 
Theorem: Area of a right triangle with specific leg and hypotenuse relationships

Given a right triangle where:
- One leg is 1/3 longer than the other leg
- The same leg is 1/3 shorter than the hypotenuse

The area of this triangle is equal to 2/3 times the square of the shorter leg.
-/
theorem right_triangle_area_with_specific_ratios 
  (a b c : ℝ) -- a, b are legs, c is hypotenuse
  (h_right : a^2 + b^2 = c^2) -- right triangle condition
  (h_leg_ratio : a = (4/3) * b) -- one leg is 1/3 longer than the other
  (h_hyp_ratio : a = (2/3) * c) -- the same leg is 1/3 shorter than hypotenuse
  : (1/2) * a * b = (2/3) * b^2 := by
  sorry


end right_triangle_area_with_specific_ratios_l1021_102177


namespace three_collinear_sufficient_not_necessary_for_coplanar_l1021_102108

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Check if four points are coplanar -/
def coplanar (p q r s : Point3D) : Prop := sorry

/-- Main theorem: Three points on a line is sufficient but not necessary for four points to be coplanar -/
theorem three_collinear_sufficient_not_necessary_for_coplanar :
  (∀ p q r s : Point3D, (collinear p q r) → (coplanar p q r s)) ∧
  (∃ p q r s : Point3D, (coplanar p q r s) ∧ ¬(collinear p q r) ∧ ¬(collinear p q s) ∧ ¬(collinear p r s) ∧ ¬(collinear q r s)) :=
sorry

end three_collinear_sufficient_not_necessary_for_coplanar_l1021_102108


namespace unique_solution_for_equation_l1021_102100

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 1991 is the only natural number n for which n + s(n) = 2011 -/
theorem unique_solution_for_equation : 
  ∃! n : ℕ, n + sum_of_digits n = 2011 ∧ n = 1991 := by sorry

end unique_solution_for_equation_l1021_102100


namespace jose_and_jane_time_l1021_102185

-- Define the time taken by Jose to complete the task alone
def jose_time : ℝ := 15

-- Define the total time when Jose does half and Jane does half
def half_half_time : ℝ := 15

-- Define the time taken by Jose and Jane together
def combined_time : ℝ := 7.5

-- Theorem statement
theorem jose_and_jane_time : 
  (jose_time : ℝ) = 15 ∧ 
  (half_half_time : ℝ) = 15 → 
  (combined_time : ℝ) = 7.5 := by
  sorry

end jose_and_jane_time_l1021_102185


namespace not_p_and_not_not_p_l1021_102127

theorem not_p_and_not_not_p (p : Prop) : ¬(p ∧ ¬p) := by
  sorry

end not_p_and_not_not_p_l1021_102127


namespace total_time_circling_island_l1021_102161

def time_per_round : ℕ := 30
def saturday_rounds : ℕ := 11
def sunday_rounds : ℕ := 15

theorem total_time_circling_island : 
  time_per_round * (saturday_rounds + sunday_rounds) = 780 := by
  sorry

end total_time_circling_island_l1021_102161


namespace quadratic_coefficient_bounds_l1021_102125

/-- Given a quadratic equation with complex coefficients, prove the maximum and minimum absolute values of a specific coefficient. -/
theorem quadratic_coefficient_bounds (z₁ z₂ m : ℂ) (α β : ℂ) :
  z₁^2 - 4*z₂ = 16 + 20*I →
  x^2 + z₁*x + z₂ + m = 0 →
  α^2 + z₁*α + z₂ + m = 0 →
  β^2 + z₁*β + z₂ + m = 0 →
  Complex.abs (α - β) = 2 * Real.sqrt 7 →
  (Complex.abs m = Real.sqrt 41 + 7 ∨ Complex.abs m = Real.sqrt 41 - 7) ∧
  ∀ m' : ℂ, Complex.abs m' ≤ Real.sqrt 41 + 7 ∧ Complex.abs m' ≥ Real.sqrt 41 - 7 :=
by sorry

end quadratic_coefficient_bounds_l1021_102125


namespace determine_xy_condition_l1021_102190

/-- Given two integers m and n, this theorem states the conditions under which 
    it's always possible to determine xy given x^m + y^m and x^n + y^n. -/
theorem determine_xy_condition (m n : ℤ) :
  (∀ x y : ℝ, ∃! (xy : ℝ), ∀ x' y' : ℝ, 
    x'^m + y'^m = x^m + y^m ∧ x'^n + y'^n = x^n + y^n → x' * y' = xy) ↔
  (∃ k t : ℤ, m = 2*k + 1 ∧ n = 2*t*(2*k + 1) ∧ t > 0) :=
sorry

end determine_xy_condition_l1021_102190


namespace jackson_investment_ratio_l1021_102175

-- Define the initial investment amount
def initial_investment : ℝ := 500

-- Define Brandon's final investment as a percentage of the initial
def brandon_final_percentage : ℝ := 0.2

-- Define the difference between Jackson's and Brandon's final investments
def investment_difference : ℝ := 1900

-- Theorem to prove
theorem jackson_investment_ratio :
  let brandon_final := initial_investment * brandon_final_percentage
  let jackson_final := brandon_final + investment_difference
  jackson_final / initial_investment = 4 := by
  sorry

end jackson_investment_ratio_l1021_102175


namespace bus_driver_worked_54_hours_l1021_102159

/-- Represents the bus driver's compensation structure and work details for a week --/
structure BusDriverWeek where
  regularRate : ℝ
  overtimeRateMultiplier : ℝ
  regularHoursLimit : ℕ
  bonusPerPassenger : ℝ
  totalCompensation : ℝ
  passengersTransported : ℕ

/-- Calculates the total hours worked by the bus driver --/
def totalHoursWorked (week : BusDriverWeek) : ℝ :=
  sorry

/-- Theorem stating that given the specific conditions, the bus driver worked 54 hours --/
theorem bus_driver_worked_54_hours :
  let week : BusDriverWeek := {
    regularRate := 14,
    overtimeRateMultiplier := 1.75,
    regularHoursLimit := 40,
    bonusPerPassenger := 0.25,
    totalCompensation := 998,
    passengersTransported := 350
  }
  totalHoursWorked week = 54 := by
  sorry

end bus_driver_worked_54_hours_l1021_102159


namespace simplify_expression_l1021_102128

theorem simplify_expression : -(-3) - 4 + (-5) = 3 - 4 - 5 := by
  sorry

end simplify_expression_l1021_102128


namespace arithmetic_mean_special_set_l1021_102103

/-- Given a set of n numbers where n > 1, one number is 1 - 2/n and all others are 1,
    the arithmetic mean of these numbers is 1 - 2/n² -/
theorem arithmetic_mean_special_set (n : ℕ) (h : n > 1) :
  let s : Finset ℕ := Finset.range n
  let f : ℕ → ℝ := fun i => if i = 0 then 1 - 2 / n else 1
  (s.sum f) / n = 1 - 2 / n^2 := by
sorry

end arithmetic_mean_special_set_l1021_102103


namespace problem_solution_l1021_102124

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 6) :
  (x + y) / (x - y) = -Real.sqrt 2 := by
  sorry

end problem_solution_l1021_102124


namespace alphabet_letters_with_dot_only_l1021_102137

theorem alphabet_letters_with_dot_only (total : ℕ) (both : ℕ) (line_only : ℕ) 
  (h_total : total = 40)
  (h_both : both = 10)
  (h_line_only : line_only = 24)
  (h_all_types : total = both + line_only + (total - both - line_only)) :
  total - both - line_only = 6 := by
  sorry

end alphabet_letters_with_dot_only_l1021_102137


namespace function_composition_l1021_102166

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_composition (x : ℝ) (h : x ≥ -1) :
  f (Real.sqrt x - 1) = x - 2 * Real.sqrt x →
  f x = x^2 - 1 := by
  sorry

end function_composition_l1021_102166


namespace intersection_length_range_l1021_102170

def interval_length (a b : ℝ) := b - a

theorem intersection_length_range :
  ∀ (a b : ℝ),
  (∀ x ∈ {x | a ≤ x ∧ x ≤ a+1}, -1 ≤ x ∧ x ≤ 1) →
  (∀ x ∈ {x | b-3/2 ≤ x ∧ x ≤ b}, -1 ≤ x ∧ x ≤ 1) →
  ∃ (l : ℝ), l = interval_length (max a (b-3/2)) (min (a+1) b) ∧
  1/2 ≤ l ∧ l ≤ 1 :=
by sorry

end intersection_length_range_l1021_102170


namespace staircase_arrangement_count_l1021_102157

/-- The number of ways to arrange 3 people on a 7-step staircase --/
def arrangement_count : ℕ := 336

/-- The number of steps on the staircase --/
def num_steps : ℕ := 7

/-- The maximum number of people that can stand on a single step --/
def max_per_step : ℕ := 2

/-- The number of people to be arranged on the staircase --/
def num_people : ℕ := 3

/-- Theorem stating that the number of arrangements is 336 --/
theorem staircase_arrangement_count :
  arrangement_count = 336 :=
by sorry

end staircase_arrangement_count_l1021_102157


namespace sum_of_digits_problem_l1021_102112

def S (n : ℕ) : ℕ := sorry  -- Sum of digits function

theorem sum_of_digits_problem (N : ℕ) 
  (h1 : S N + S (N + 1) = 200)
  (h2 : S (N + 2) + S (N + 3) = 105) :
  S (N + 1) + S (N + 2) = 103 := by
  sorry

end sum_of_digits_problem_l1021_102112


namespace quadratic_sum_l1021_102189

/-- A quadratic function f(x) = px² + qx + r with vertex (3, 4) passing through (1, 2) -/
def QuadraticFunction (p q r : ℝ) : ℝ → ℝ := fun x ↦ p * x^2 + q * x + r

theorem quadratic_sum (p q r : ℝ) :
  (∀ x, QuadraticFunction p q r x = p * x^2 + q * x + r) →
  (∃ a, ∀ x, QuadraticFunction p q r x = a * (x - 3)^2 + 4) →
  QuadraticFunction p q r 1 = 2 →
  p + q + r = 3 := by
  sorry

end quadratic_sum_l1021_102189


namespace complex_equation_solution_l1021_102123

theorem complex_equation_solution (i z : ℂ) (hi : i^2 = -1) (hz : i * z = 2 * z + 1) :
  z = -2/5 - 1/5 * i := by sorry

end complex_equation_solution_l1021_102123


namespace parabola_directrix_l1021_102186

/-- A parabola with equation y = 1/4 * x^2 has a directrix with equation y = -1 -/
theorem parabola_directrix (x y : ℝ) :
  y = (1/4) * x^2 → ∃ (k : ℝ), k = -1 ∧ (∀ (x₀ y₀ : ℝ), y₀ = k → 
    (x₀ - x)^2 + (y₀ - y)^2 = (y₀ - (y + 1/4))^2) := by
  sorry

end parabola_directrix_l1021_102186


namespace unique_base_solution_l1021_102164

/-- Given a natural number b ≥ 2, convert a number in base b to its decimal representation -/
def toDecimal (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- Given a natural number b ≥ 2, check if the equation 161_b + 134_b = 315_b holds -/
def checkEquation (b : ℕ) : Prop :=
  toDecimal 161 b + toDecimal 134 b = toDecimal 315 b

/-- The main theorem stating that 8 is the unique solution to the equation -/
theorem unique_base_solution :
  ∃! b : ℕ, b ≥ 2 ∧ checkEquation b ∧ b = 8 :=
sorry

end unique_base_solution_l1021_102164


namespace simplify_expression_l1021_102110

theorem simplify_expression (a : ℝ) : (2*a - 3)^2 - (a + 5)*(a - 5) = 3*a^2 - 12*a + 34 := by
  sorry

end simplify_expression_l1021_102110


namespace probability_point_in_ellipsoid_l1021_102148

/-- The probability of a point in a rectangular prism satisfying an ellipsoid equation -/
theorem probability_point_in_ellipsoid : 
  let prism_volume := (2 - (-2)) * (1 - (-1)) * (1 - (-1))
  let ellipsoid_volume := (4 * Real.pi / 3) * 1 * 2 * 2
  let probability := ellipsoid_volume / prism_volume
  probability = Real.pi / 3 := by
  sorry

end probability_point_in_ellipsoid_l1021_102148


namespace cubic_minimum_at_negative_one_l1021_102119

/-- A cubic function with parameters p, q, and r -/
def cubic_function (p q r : ℝ) (x : ℝ) : ℝ :=
  x^3 + p*x^2 + q*x + r

theorem cubic_minimum_at_negative_one (p q r : ℝ) :
  (∀ x, cubic_function p q r x ≥ 0) ∧
  (cubic_function p q r (-1) = 0) →
  r = p - 2 :=
sorry

end cubic_minimum_at_negative_one_l1021_102119


namespace imaginary_part_zi_l1021_102136

def complex_coords (z : ℂ) : ℝ × ℝ := (z.re, z.im)

theorem imaginary_part_zi (z : ℂ) (h : complex_coords z = (-2, 1)) : 
  (z * Complex.I).im = -2 := by
  sorry

end imaginary_part_zi_l1021_102136


namespace solution_set_part_I_range_of_m_part_II_l1021_102155

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3|

-- Theorem for part (I)
theorem solution_set_part_I :
  {x : ℝ | f x ≥ 3 - |x - 2|} = {x : ℝ | x ≤ 1 ∨ x ≥ 4} :=
sorry

-- Theorem for part (II)
theorem range_of_m_part_II :
  ∀ m : ℝ, (∃ x : ℝ, f x ≤ 2*m - |x + 4|) → m ≥ 7/2 :=
sorry

end solution_set_part_I_range_of_m_part_II_l1021_102155


namespace area_perimeter_product_l1021_102191

/-- Represents a point on a 2D grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℕ :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

/-- Represents a square on the grid -/
structure Square where
  E : Point
  F : Point
  G : Point
  H : Point

/-- The specific square EFGH from the problem -/
def EFGH : Square :=
  { E := { x := 1, y := 5 },
    F := { x := 5, y := 6 },
    G := { x := 6, y := 2 },
    H := { x := 2, y := 1 } }

/-- Theorem stating the product of area and perimeter of EFGH -/
theorem area_perimeter_product (s : Square) (h : s = EFGH) :
  (↑(squaredDistance s.E s.F) : ℝ) * (4 * Real.sqrt (↑(squaredDistance s.E s.F))) = 68 * Real.sqrt 17 := by
  sorry

end area_perimeter_product_l1021_102191


namespace expand_binomial_product_l1021_102120

theorem expand_binomial_product (x : ℝ) : (1 + x^2) * (1 - x^3) = 1 + x^2 - x^3 - x^5 := by
  sorry

end expand_binomial_product_l1021_102120


namespace calculator_price_proof_l1021_102192

theorem calculator_price_proof (total_calculators : ℕ) (total_sales : ℕ) 
  (first_type_count : ℕ) (first_type_price : ℕ) (second_type_count : ℕ) :
  total_calculators = 85 →
  total_sales = 3875 →
  first_type_count = 35 →
  first_type_price = 15 →
  second_type_count = total_calculators - first_type_count →
  (first_type_count * first_type_price + second_type_count * 67 = total_sales) :=
by
  sorry

#check calculator_price_proof

end calculator_price_proof_l1021_102192


namespace day_division_count_l1021_102174

-- Define the number of seconds in a day
def seconds_in_day : ℕ := 72000

-- Define a function to count the number of ways to divide the day
def count_divisions (total_seconds : ℕ) : ℕ :=
  -- The actual implementation is not provided, as per instructions
  sorry

-- Theorem statement
theorem day_division_count :
  count_divisions seconds_in_day = 60 := by sorry

end day_division_count_l1021_102174


namespace unique_positive_integer_solution_l1021_102151

theorem unique_positive_integer_solution :
  ∃! (m n : ℕ+), 15 * m * n = 75 - 5 * m - 3 * n :=
by
  -- The proof goes here
  sorry

end unique_positive_integer_solution_l1021_102151


namespace no_solution_inequality_system_l1021_102107

theorem no_solution_inequality_system :
  ¬ ∃ (x y : ℝ), (4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2) ∧ (x - y ≤ -1) := by
  sorry

end no_solution_inequality_system_l1021_102107


namespace right_triangle_ratio_l1021_102158

theorem right_triangle_ratio (a d : ℝ) (h_d_pos : d > 0) (h_d_odd : ∃ k : ℤ, d = 2 * k + 1) :
  (a + 4 * d)^2 = a^2 + (a + 2 * d)^2 → a / d = 1 + Real.sqrt 7 := by
  sorry

end right_triangle_ratio_l1021_102158


namespace unique_quadruple_solution_l1021_102144

theorem unique_quadruple_solution :
  ∃! (a b c d : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
    a^2 + b^2 + c^2 + d^2 = 4 ∧
    (a + b + c + d) * (a^4 + b^4 + c^4 + d^4) = 32 :=
by sorry

end unique_quadruple_solution_l1021_102144


namespace junk_mail_calculation_l1021_102195

theorem junk_mail_calculation (blocks : ℕ) (houses_per_block : ℕ) (mail_per_house : ℕ)
  (h1 : blocks = 16)
  (h2 : houses_per_block = 17)
  (h3 : mail_per_house = 4) :
  blocks * houses_per_block * mail_per_house = 1088 := by
  sorry

end junk_mail_calculation_l1021_102195


namespace square_minus_product_equals_one_l1021_102173

theorem square_minus_product_equals_one : 2015^2 - 2016 * 2014 = 1 := by
  sorry

end square_minus_product_equals_one_l1021_102173


namespace root_sum_reciprocal_l1021_102163

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a^2 + a - 1 = 0) → 
  (b^3 - 2*b^2 + b - 1 = 0) → 
  (c^3 - 2*c^2 + c - 1 = 0) → 
  (1/(a-2) + 1/(b-2) + 1/(c-2) = -5) := by
sorry

end root_sum_reciprocal_l1021_102163


namespace brownies_remaining_l1021_102106

/-- Calculates the number of brownies left after consumption --/
def brownies_left (total : Nat) (tina_daily : Nat) (husband_daily : Nat) (days : Nat) (shared : Nat) : Nat :=
  total - (tina_daily * days) - (husband_daily * days) - shared

/-- Proves that 5 brownies are left under given conditions --/
theorem brownies_remaining : brownies_left 24 2 1 5 4 = 5 := by
  sorry

end brownies_remaining_l1021_102106


namespace sum_of_coefficients_is_negative_three_l1021_102181

/-- The polynomial x^3 - 7x^2 + 12x - 18 -/
def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 12*x - 18

/-- The sum of the kth powers of the roots of p -/
def s (k : ℕ) : ℝ := sorry

/-- The recursive relationship for s_k -/
def recursive_relation (a b c : ℝ) : Prop :=
  ∀ k : ℕ, k ≥ 2 → s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)

theorem sum_of_coefficients_is_negative_three :
  ∀ a b c : ℝ,
  (∃ α β γ : ℝ, p α = 0 ∧ p β = 0 ∧ p γ = 0) →
  s 0 = 3 →
  s 1 = 7 →
  s 2 = 13 →
  recursive_relation a b c →
  a + b + c = -3 := by sorry

end sum_of_coefficients_is_negative_three_l1021_102181


namespace trigonometric_equation_solution_l1021_102122

theorem trigonometric_equation_solution (t : ℝ) : 
  5.43 * Real.cos (22 * π / 180 - t) * Real.cos (82 * π / 180 - t) + 
  Real.cos (112 * π / 180 - t) * Real.cos (172 * π / 180 - t) = 
  0.5 * (Real.sin t + Real.cos t) ↔ 
  (∃ k : ℤ, t = 2 * π * k ∨ t = π / 2 * (4 * k + 1)) :=
by sorry

end trigonometric_equation_solution_l1021_102122


namespace equal_wealth_after_transfer_l1021_102180

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions given in the problem -/
def problem_conditions (w : MerchantWealth) : Prop :=
  (w.foma - 70 = w.ierema + 70) ∧ 
  (w.foma - 40 = w.yuliy)

/-- The theorem to be proved -/
theorem equal_wealth_after_transfer (w : MerchantWealth) 
  (h : problem_conditions w) : 
  w.foma - 55 = w.ierema + 55 := by
  sorry

end equal_wealth_after_transfer_l1021_102180


namespace sqrt_450_equals_15_l1021_102165

theorem sqrt_450_equals_15 : Real.sqrt 450 = 15 := by
  sorry

end sqrt_450_equals_15_l1021_102165


namespace cubic_expression_value_l1021_102104

theorem cubic_expression_value (p q : ℝ) : 
  (8 * p + 2 * q = 2022) → 
  ((-8) * p + (-2) * q + 1 = -2021) := by
  sorry

end cubic_expression_value_l1021_102104


namespace square_diagonal_ratio_l1021_102193

theorem square_diagonal_ratio (s S : ℝ) (h_perimeter_ratio : 4 * S = 4 * (4 * s)) :
  S * Real.sqrt 2 = 4 * (s * Real.sqrt 2) := by
  sorry

end square_diagonal_ratio_l1021_102193

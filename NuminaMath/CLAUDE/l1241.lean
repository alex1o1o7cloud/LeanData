import Mathlib

namespace product_difference_l1241_124141

theorem product_difference (A B : ℝ) 
  (h1 : (A + 2) * B = A * B + 60)
  (h2 : A * (B - 3) = A * B - 24) :
  (A + 2) * (B - 3) - A * B = 30 := by
  sorry

end product_difference_l1241_124141


namespace intersection_point_x_coordinate_l1241_124164

theorem intersection_point_x_coordinate 
  (line1 : ℝ → ℝ) 
  (line2 : ℝ → ℝ) 
  (h1 : ∀ x, line1 x = 3 * x - 7)
  (h2 : ∀ x, 5 * x + line2 x = 48) :
  ∃ x, line1 x = line2 x ∧ x = 55 / 8 := by
sorry

end intersection_point_x_coordinate_l1241_124164


namespace necessary_but_not_sufficient_l1241_124184

-- Define the conditions for a hyperbola and an ellipse
def is_hyperbola (m : ℝ) : Prop := (m + 3) * (2 * m + 1) < 0
def is_ellipse_with_y_intersection (m : ℝ) : Prop := -(2 * m - 1) > m + 2 ∧ m + 2 > 0

-- Define the condition given in the problem
def given_condition (m : ℝ) : Prop := -2 < m ∧ m < -1/3

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ m : ℝ, is_hyperbola m ∧ is_ellipse_with_y_intersection m → given_condition m) ∧
  (∃ m : ℝ, given_condition m ∧ ¬(is_hyperbola m ∧ is_ellipse_with_y_intersection m)) :=
sorry

end necessary_but_not_sufficient_l1241_124184


namespace final_seashell_count_l1241_124183

def seashell_transactions (initial : ℝ) (friend_gift : ℝ) (brother_gift : ℝ) 
  (buy_percent : ℝ) (sell_fraction : ℝ) (damage_percent : ℝ) (trade_fraction : ℝ) : ℝ :=
  let remaining_after_gifts := initial - friend_gift - brother_gift
  let after_buying := remaining_after_gifts + (buy_percent * remaining_after_gifts)
  let after_selling := after_buying - (sell_fraction * after_buying)
  let after_damage := after_selling - (damage_percent * after_selling)
  after_damage - (trade_fraction * after_damage)

theorem final_seashell_count : 
  seashell_transactions 385.5 45.75 34.25 0.2 (2/3) 0.1 (1/4) = 82.485 := by
  sorry

end final_seashell_count_l1241_124183


namespace intersection_of_lines_l1241_124178

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating that the intersection point of lines AB and CD is (4/3, -7/3, 14/3) --/
theorem intersection_of_lines :
  let A : ℝ × ℝ × ℝ := (6, -7, 7)
  let B : ℝ × ℝ × ℝ := (16, -17, 12)
  let C : ℝ × ℝ × ℝ := (0, 3, -6)
  let D : ℝ × ℝ × ℝ := (2, -5, 10)
  intersection_point A B C D = (4/3, -7/3, 14/3) := by
  sorry

end intersection_of_lines_l1241_124178


namespace cos_seven_pi_sixths_l1241_124151

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end cos_seven_pi_sixths_l1241_124151


namespace cos_sum_of_complex_exponentials_l1241_124121

theorem cos_sum_of_complex_exponentials (α β : ℝ) :
  Complex.exp (α * Complex.I) = (4 / 5 : ℂ) - (3 / 5 : ℂ) * Complex.I →
  Complex.exp (β * Complex.I) = (5 / 13 : ℂ) + (12 / 13 : ℂ) * Complex.I →
  Real.cos (α + β) = -16 / 65 := by
  sorry

end cos_sum_of_complex_exponentials_l1241_124121


namespace three_digit_probability_l1241_124116

theorem three_digit_probability : 
  let S := Finset.Icc 30 800
  let three_digit := {n : ℕ | 100 ≤ n ∧ n ≤ 800}
  (S.filter (λ n => n ∈ three_digit)).card / S.card = 701 / 771 :=
by sorry

end three_digit_probability_l1241_124116


namespace independence_test_most_appropriate_l1241_124100

/-- Represents the survey data --/
structure SurveyData where
  male_total : Nat
  male_opposing : Nat
  female_total : Nat
  female_opposing : Nat
  deriving Repr

/-- Represents different statistical methods --/
inductive StatMethod
  | Mean
  | Regression
  | IndependenceTest
  | Probability
  deriving Repr

/-- Determines the most appropriate method for analyzing the relationship
    between gender and judgment in the survey --/
def most_appropriate_method (data : SurveyData) : StatMethod :=
  StatMethod.IndependenceTest

/-- Theorem stating that the independence test is the most appropriate method
    for the given survey data --/
theorem independence_test_most_appropriate (data : SurveyData) 
    (h1 : data.male_total = 2548)
    (h2 : data.male_opposing = 1560)
    (h3 : data.female_total = 2452)
    (h4 : data.female_opposing = 1200) :
    most_appropriate_method data = StatMethod.IndependenceTest := by
  sorry


end independence_test_most_appropriate_l1241_124100


namespace three_person_arrangement_l1241_124167

def number_of_arrangements (n : ℕ) : ℕ := Nat.factorial n

theorem three_person_arrangement :
  number_of_arrangements 3 = 6 := by
  sorry

end three_person_arrangement_l1241_124167


namespace candy_tins_count_l1241_124148

/-- The number of candy tins given the total number of strawberry-flavored candies
    and the number of strawberry-flavored candies per tin. -/
def number_of_candy_tins (total_strawberry_candies : ℕ) (strawberry_candies_per_tin : ℕ) : ℕ :=
  total_strawberry_candies / strawberry_candies_per_tin

/-- Theorem stating that the number of candy tins is 9 given the problem conditions. -/
theorem candy_tins_count : number_of_candy_tins 27 3 = 9 := by
  sorry

end candy_tins_count_l1241_124148


namespace square_and_cube_roots_problem_l1241_124158

theorem square_and_cube_roots_problem (a b : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (3*a - 14)^2 = x ∧ (a + 2)^2 = x) → 
  (b + 11)^(1/3) = -3 → 
  a = 3 ∧ b = -38 ∧ (1 - (a + b))^(1/2) = 6 ∨ (1 - (a + b))^(1/2) = -6 :=
by sorry

end square_and_cube_roots_problem_l1241_124158


namespace vacant_seats_l1241_124107

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 700) (h2 : filled_percentage = 75 / 100) :
  (1 - filled_percentage) * total_seats = 175 := by
  sorry

end vacant_seats_l1241_124107


namespace power_fraction_simplification_l1241_124130

theorem power_fraction_simplification : 
  (12 : ℕ)^10 / (144 : ℕ)^4 = (144 : ℕ) :=
by
  sorry

end power_fraction_simplification_l1241_124130


namespace measure_one_kg_cereal_l1241_124195

/-- Represents a balance scale that may be inaccurate -/
structure BalanceScale where
  isBalanced : (ℝ → ℝ → Prop)

/-- Represents a bag of cereal -/
def CerealBag : Type := ℝ

/-- Represents a correct 1 kg weight -/
def CorrectWeight : ℝ := 1

/-- Function to measure cereal using the balance scale and correct weight -/
def measureCereal (scale : BalanceScale) (bag : CerealBag) (weight : ℝ) : Prop :=
  ∃ (amount : ℝ), 
    scale.isBalanced amount weight ∧ 
    scale.isBalanced amount amount ∧ 
    amount = weight

/-- Theorem stating that it's possible to measure 1 kg of cereal -/
theorem measure_one_kg_cereal 
  (scale : BalanceScale) 
  (bag : CerealBag) : 
  measureCereal scale bag CorrectWeight := by
  sorry


end measure_one_kg_cereal_l1241_124195


namespace min_cut_edges_hexagonal_prism_l1241_124163

/-- Represents a hexagonal prism -/
structure HexagonalPrism :=
  (total_edges : ℕ)
  (uncut_edges : ℕ)
  (h_total : total_edges = 18)
  (h_uncut : uncut_edges ≤ total_edges)

/-- The minimum number of edges that need to be cut to unfold a hexagonal prism -/
def min_cut_edges (prism : HexagonalPrism) : ℕ :=
  prism.total_edges - prism.uncut_edges

theorem min_cut_edges_hexagonal_prism (prism : HexagonalPrism) 
  (h_uncut : prism.uncut_edges = 7) : 
  min_cut_edges prism = 11 := by
  sorry

end min_cut_edges_hexagonal_prism_l1241_124163


namespace arctan_sum_special_case_l1241_124143

theorem arctan_sum_special_case : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end arctan_sum_special_case_l1241_124143


namespace unique_minimum_cost_plan_l1241_124127

/-- Represents a bus rental plan -/
structure BusRentalPlan where
  busA : ℕ  -- Number of Bus A
  busB : ℕ  -- Number of Bus B

/-- Checks if a bus rental plan is valid -/
def isValidPlan (p : BusRentalPlan) : Prop :=
  let totalPeople := 16 + 284
  let totalCapacity := 30 * p.busA + 42 * p.busB
  let totalCost := 300 * p.busA + 400 * p.busB
  let totalBuses := p.busA + p.busB
  totalCapacity ≥ totalPeople ∧
  totalCost ≤ 3100 ∧
  2 * totalBuses ≤ 16

/-- The set of all valid bus rental plans -/
def validPlans : Set BusRentalPlan :=
  {p : BusRentalPlan | isValidPlan p}

/-- The rental cost of a plan -/
def rentalCost (p : BusRentalPlan) : ℕ :=
  300 * p.busA + 400 * p.busB

theorem unique_minimum_cost_plan :
  ∃! p : BusRentalPlan, p ∈ validPlans ∧
    ∀ q ∈ validPlans, rentalCost p ≤ rentalCost q ∧
    rentalCost p = 2900 := by
  sorry

#check unique_minimum_cost_plan

end unique_minimum_cost_plan_l1241_124127


namespace expression_evaluation_l1241_124169

theorem expression_evaluation (a b : ℝ) (h1 : a = 2) (h2 : b = 1/3) :
  a / (a - b) * (1 / b - 1 / a) + (a - 1) / b = 6 := by
  sorry

end expression_evaluation_l1241_124169


namespace candy_count_l1241_124179

/-- The number of candy pieces Jake had initially -/
def initial_candy : ℕ := 80

/-- The number of candy pieces Jake sold on Monday -/
def monday_sales : ℕ := 15

/-- The number of candy pieces Jake sold on Tuesday -/
def tuesday_sales : ℕ := 58

/-- The number of candy pieces Jake had left on Wednesday -/
def wednesday_left : ℕ := 7

/-- Theorem stating that the initial number of candy pieces equals the sum of pieces sold on Monday and Tuesday plus the pieces left on Wednesday -/
theorem candy_count : initial_candy = monday_sales + tuesday_sales + wednesday_left := by
  sorry

end candy_count_l1241_124179


namespace night_rides_total_l1241_124155

def total_ferris_rides : ℕ := 13
def total_roller_coaster_rides : ℕ := 9
def day_ferris_rides : ℕ := 7
def day_roller_coaster_rides : ℕ := 4

theorem night_rides_total : 
  (total_ferris_rides - day_ferris_rides) + (total_roller_coaster_rides - day_roller_coaster_rides) = 11 := by
  sorry

end night_rides_total_l1241_124155


namespace complex_fraction_calculation_l1241_124113

theorem complex_fraction_calculation : 
  27 * ((2 + 2/3) - (3 + 1/4)) / ((1 + 1/2) + (2 + 1/5)) = -(4 + 43/74) := by
  sorry

end complex_fraction_calculation_l1241_124113


namespace fgh_supermarkets_count_fgh_supermarkets_count_proof_l1241_124177

theorem fgh_supermarkets_count : ℕ → ℕ → ℕ → Prop :=
  fun us_count canada_count total =>
    (us_count = 37) →
    (us_count = canada_count + 14) →
    (total = us_count + canada_count) →
    (total = 60)

-- The proof goes here
theorem fgh_supermarkets_count_proof : fgh_supermarkets_count 37 23 60 := by
  sorry

end fgh_supermarkets_count_fgh_supermarkets_count_proof_l1241_124177


namespace milk_left_over_calculation_l1241_124104

/-- The amount of milk left over given the following conditions:
  - Total milk production is 24 cups per day
  - 80% of milk is consumed by Daisy's kids
  - 60% of remaining milk is used for cooking
  - 25% of remaining milk is given to neighbor
  - 6% of remaining milk is drunk by Daisy's husband
-/
def milk_left_over (total_milk : ℝ) (kids_consumption : ℝ) (cooking_usage : ℝ)
  (neighbor_share : ℝ) (husband_consumption : ℝ) : ℝ :=
  let remaining_after_kids := total_milk * (1 - kids_consumption)
  let remaining_after_cooking := remaining_after_kids * (1 - cooking_usage)
  let remaining_after_neighbor := remaining_after_cooking * (1 - neighbor_share)
  remaining_after_neighbor * (1 - husband_consumption)

theorem milk_left_over_calculation :
  milk_left_over 24 0.8 0.6 0.25 0.06 = 1.3536 := by
  sorry

end milk_left_over_calculation_l1241_124104


namespace k_range_for_inequality_l1241_124111

theorem k_range_for_inequality (k : ℝ) : 
  k ≠ 0 → 
  (k^2 * 1^2 - 6*k*1 + 8 ≥ 0) → 
  (k ≥ 4 ∨ k ≤ 2) :=
by sorry

end k_range_for_inequality_l1241_124111


namespace weight_of_seven_moles_l1241_124165

/-- Given a compound with molecular weight 1176, prove that 7 moles of this compound weigh 8232 -/
theorem weight_of_seven_moles (compound_weight : ℝ) (h : compound_weight = 1176) :
  7 * compound_weight = 8232 := by
  sorry

end weight_of_seven_moles_l1241_124165


namespace geometric_sequence_problem_l1241_124180

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) :
  (∃ r : ℝ, 180 * r = a ∧ a * r = 81 / 32) → a = 135 / 19 := by
  sorry

end geometric_sequence_problem_l1241_124180


namespace fox_coins_proof_l1241_124144

def cross_bridge (initial_coins : ℕ) : ℕ := 
  3 * initial_coins - 50

def cross_bridge_n_times (initial_coins : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial_coins
  | m + 1 => cross_bridge (cross_bridge_n_times initial_coins m)

theorem fox_coins_proof :
  cross_bridge_n_times 25 4 = 20 := by
  sorry

end fox_coins_proof_l1241_124144


namespace carnation_count_l1241_124103

theorem carnation_count (vase_capacity : ℕ) (rose_count : ℕ) (vase_count : ℕ) :
  vase_capacity = 6 →
  rose_count = 47 →
  vase_count = 9 →
  vase_count * vase_capacity - rose_count = 7 := by
  sorry

end carnation_count_l1241_124103


namespace positive_real_inequality_l1241_124147

theorem positive_real_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x + y + z = 1/x + 1/y + 1/z) :
  x + y + z ≥ Real.sqrt ((x*y + 1)/2) + Real.sqrt ((y*z + 1)/2) + Real.sqrt ((z*x + 1)/2) := by
  sorry

end positive_real_inequality_l1241_124147


namespace basketball_lineup_combinations_l1241_124156

theorem basketball_lineup_combinations : 
  let total_players : ℕ := 16
  let quadruplets : ℕ := 4
  let lineup_size : ℕ := 7
  let quadruplets_in_lineup : ℕ := 3
  let captain_in_lineup : ℕ := 1

  (Nat.choose quadruplets quadruplets_in_lineup) * 
  (Nat.choose (total_players - quadruplets - captain_in_lineup) 
              (lineup_size - quadruplets_in_lineup - captain_in_lineup)) = 220 :=
by
  sorry

end basketball_lineup_combinations_l1241_124156


namespace soda_cost_l1241_124105

/-- Proves that the cost of each soda is $0.87 given the total cost and the cost of sandwiches -/
theorem soda_cost (total_cost : ℚ) (sandwich_cost : ℚ) (num_sandwiches : ℕ) (num_sodas : ℕ) :
  total_cost = 8.38 →
  sandwich_cost = 2.45 →
  num_sandwiches = 2 →
  num_sodas = 4 →
  (total_cost - num_sandwiches * sandwich_cost) / num_sodas = 0.87 := by
sorry

#eval (8.38 - 2 * 2.45) / 4

end soda_cost_l1241_124105


namespace polynomial_divisibility_l1241_124168

theorem polynomial_divisibility : ∃ q : Polynomial ℂ, 
  X^66 + X^55 + X^44 + X^33 + X^22 + X^11 + 1 = 
  q * (X^6 + X^5 + X^4 + X^3 + X^2 + X + 1) := by
  sorry

end polynomial_divisibility_l1241_124168


namespace fraction_sum_equality_l1241_124174

theorem fraction_sum_equality : 
  (3 : ℚ) / 15 + 5 / 150 + 7 / 1500 + 9 / 15000 = 0.2386 := by
  sorry

end fraction_sum_equality_l1241_124174


namespace range_where_g_geq_f_max_value_g_minus_f_l1241_124140

-- Define the functions f and g
def f (x : ℝ) : ℝ := abs (x - 1)
def g (x : ℝ) : ℝ := -x^2 + 6*x - 5

-- Theorem for the range of x where g(x) ≥ f(x)
theorem range_where_g_geq_f :
  {x : ℝ | g x ≥ f x} = Set.Ici 1 ∩ Set.Iic 4 :=
sorry

-- Theorem for the maximum value of g(x) - f(x)
theorem max_value_g_minus_f :
  ∃ (x : ℝ), ∀ (y : ℝ), g y - f y ≤ g x - f x ∧ g x - f x = 9/4 :=
sorry

end range_where_g_geq_f_max_value_g_minus_f_l1241_124140


namespace circle_area_with_diameter_10_l1241_124189

/-- The area of a circle with diameter 10 meters is 25π square meters -/
theorem circle_area_with_diameter_10 :
  ∀ (A : ℝ) (π : ℝ), 
  (∃ (d : ℝ), d = 10 ∧ A = (π * d^2) / 4) →
  A = 25 * π :=
by sorry

end circle_area_with_diameter_10_l1241_124189


namespace equation_holds_l1241_124194

theorem equation_holds (x y z : ℝ) (h : (x - z)^2 - 4*(x - y)*(y - z) = 0) :
  z + x - 2*y = 0 := by
  sorry

end equation_holds_l1241_124194


namespace intersection_line_circle_l1241_124193

def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p | a * p.1 + b * p.2 = c}

theorem intersection_line_circle (a : ℝ) :
  let O : ℝ × ℝ := (0, 0)
  let C : Set (ℝ × ℝ) := Circle O 2
  let L : Set (ℝ × ℝ) := Line 1 1 a
  ∀ A B : ℝ × ℝ, A ∈ C ∩ L → B ∈ C ∩ L →
    ‖(A.1, A.2)‖ = ‖(A.1 + B.1, A.2 + B.2)‖ →
      a = 2 ∨ a = -2 :=
by
  sorry

#check intersection_line_circle

end intersection_line_circle_l1241_124193


namespace polynomial_range_open_interval_l1241_124131

theorem polynomial_range_open_interval : 
  (∀ k : ℝ, k > 0 → ∃ x y : ℝ, (1 - x * y)^2 + x^2 = k) ∧ 
  (∀ x y : ℝ, (1 - x * y)^2 + x^2 > 0) := by
  sorry

end polynomial_range_open_interval_l1241_124131


namespace smallest_two_digit_switch_add_five_l1241_124109

def digit_switch (n : ℕ) : ℕ := 
  (n % 10) * 10 + (n / 10)

theorem smallest_two_digit_switch_add_five : 
  ∀ n : ℕ, 
    10 ≤ n → n < 100 → 
    (∀ m : ℕ, 10 ≤ m → m < n → digit_switch m + 5 ≠ 3 * m) → 
    digit_switch n + 5 = 3 * n → 
    n = 34 := by
sorry

end smallest_two_digit_switch_add_five_l1241_124109


namespace quadratic_solution_range_l1241_124110

theorem quadratic_solution_range (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^2 + (m-1)*x + 1 = 0) → m ≤ -1 := by
  sorry

end quadratic_solution_range_l1241_124110


namespace complex_product_proof_l1241_124171

theorem complex_product_proof : Complex.I * Complex.I = -1 → (1 - Complex.I) * (1 + 2 * Complex.I) = 3 + Complex.I := by
  sorry

end complex_product_proof_l1241_124171


namespace smallest_odd_with_same_divisors_as_360_l1241_124157

/-- Count the number of positive divisors of a natural number -/
def countDivisors (n : ℕ) : ℕ := sorry

/-- Check if a natural number is odd -/
def isOdd (n : ℕ) : Prop := sorry

theorem smallest_odd_with_same_divisors_as_360 :
  ∃ (n : ℕ), isOdd n ∧ countDivisors n = countDivisors 360 ∧
  ∀ (m : ℕ), isOdd m ∧ countDivisors m = countDivisors 360 → n ≤ m :=
by sorry

end smallest_odd_with_same_divisors_as_360_l1241_124157


namespace half_angle_in_third_quadrant_l1241_124142

theorem half_angle_in_third_quadrant (θ : Real) : 
  (π / 2 < θ ∧ θ < π) →  -- θ is in the second quadrant
  (|Real.sin (θ / 2)| = -Real.sin (θ / 2)) →  -- |sin(θ/2)| = -sin(θ/2)
  (π < θ / 2 ∧ θ / 2 < 3 * π / 2) -- θ/2 is in the third quadrant
  := by sorry

end half_angle_in_third_quadrant_l1241_124142


namespace alice_age_l1241_124197

/-- The ages of Alice, Bob, and Claire satisfy the given conditions -/
structure AgeRelationship where
  alice : ℕ
  bob : ℕ
  claire : ℕ
  alice_younger_than_bob : alice = bob - 3
  bob_older_than_claire : bob = claire + 5
  claire_age : claire = 12

/-- Alice's age is 14 years old given the age relationships -/
theorem alice_age (ar : AgeRelationship) : ar.alice = 14 := by
  sorry

end alice_age_l1241_124197


namespace percentage_problem_l1241_124136

theorem percentage_problem (X : ℝ) : 
  (0.2 * 40 + 0.25 * X = 23) → X = 60 := by
  sorry

end percentage_problem_l1241_124136


namespace divisible_by_99_l1241_124176

theorem divisible_by_99 (A B : ℕ) : 
  A < 10 → B < 10 → 
  99 ∣ (A * 100000 + 15000 + B * 100 + 94) → 
  A = 5 := by
sorry

end divisible_by_99_l1241_124176


namespace lara_flowers_to_mom_l1241_124160

theorem lara_flowers_to_mom (total_flowers grandma_flowers mom_flowers vase_flowers : ℕ) :
  total_flowers = 52 →
  grandma_flowers = mom_flowers + 6 →
  vase_flowers = 16 →
  total_flowers = mom_flowers + grandma_flowers + vase_flowers →
  mom_flowers = 15 := by
  sorry

end lara_flowers_to_mom_l1241_124160


namespace richards_third_day_distance_l1241_124133

/-- Represents Richard's journey from Cincinnati to New York City -/
structure Journey where
  total_distance : ℝ
  day1_distance : ℝ
  day2_distance : ℝ
  day3_distance : ℝ
  remaining_distance : ℝ

/-- Theorem stating the distance Richard walked on the third day -/
theorem richards_third_day_distance (j : Journey)
  (h1 : j.total_distance = 70)
  (h2 : j.day1_distance = 20)
  (h3 : j.day2_distance = j.day1_distance / 2 - 6)
  (h4 : j.remaining_distance = 36)
  (h5 : j.day1_distance + j.day2_distance + j.day3_distance + j.remaining_distance = j.total_distance) :
  j.day3_distance = 10 := by
  sorry

end richards_third_day_distance_l1241_124133


namespace trader_shipment_cost_l1241_124120

/-- The amount needed for the next shipment of wares --/
def amount_needed (total_profit donation excess : ℕ) : ℕ :=
  total_profit / 2 + donation - excess

/-- Theorem stating the amount needed for the next shipment --/
theorem trader_shipment_cost (total_profit donation excess : ℕ)
  (h1 : total_profit = 960)
  (h2 : donation = 310)
  (h3 : excess = 180) :
  amount_needed total_profit donation excess = 610 := by
  sorry

#eval amount_needed 960 310 180

end trader_shipment_cost_l1241_124120


namespace N2O5_molecular_weight_l1241_124191

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of nitrogen atoms in N2O5 -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in N2O5 -/
def num_O : ℕ := 5

/-- The molecular weight of N2O5 in g/mol -/
def molecular_weight_N2O5 : ℝ :=
  (num_N : ℝ) * atomic_weight_N + (num_O : ℝ) * atomic_weight_O

theorem N2O5_molecular_weight :
  molecular_weight_N2O5 = 108.02 := by
  sorry

end N2O5_molecular_weight_l1241_124191


namespace even_power_difference_divisible_l1241_124135

theorem even_power_difference_divisible (x y : ℤ) :
  ∀ k : ℕ, k > 0 → ∃ m : ℤ, x^(2*k) - y^(2*k) = (x + y) * m :=
by sorry

end even_power_difference_divisible_l1241_124135


namespace banana_tree_problem_l1241_124138

theorem banana_tree_problem (bananas_left : ℕ) (bananas_eaten : ℕ) : 
  bananas_left = 100 →
  bananas_eaten = 70 →
  (∃ (initial_bananas : ℕ), initial_bananas = bananas_left + bananas_eaten + 2 * bananas_eaten ∧ initial_bananas = 310) :=
by sorry

end banana_tree_problem_l1241_124138


namespace fraction_simplification_l1241_124122

theorem fraction_simplification :
  (240 : ℚ) / 18 * 9 / 135 * 7 / 4 = 14 / 9 := by sorry

end fraction_simplification_l1241_124122


namespace range_of_f_l1241_124123

noncomputable def f (x : ℝ) : ℝ := (3 - 2^x) / (1 + 2^x)

theorem range_of_f :
  (∀ y ∈ Set.range f, -1 < y ∧ y < 3) ∧
  (∀ ε > 0, ∃ x₁ x₂, f x₁ < -1 + ε ∧ f x₂ > 3 - ε) :=
sorry

end range_of_f_l1241_124123


namespace lemonade_amount_l1241_124175

/-- Represents the recipe for a cold drink -/
structure DrinkRecipe where
  tea : Rat
  lemonade : Rat

/-- Represents the total amount of drink in the pitcher -/
def totalAmount : Rat := 18

/-- The recipe for one serving of the drink -/
def recipe : DrinkRecipe := {
  tea := 1/4,
  lemonade := 5/4
}

/-- Calculates the amount of lemonade in the pitcher -/
def lemonadeInPitcher (r : DrinkRecipe) (total : Rat) : Rat :=
  (r.lemonade / (r.tea + r.lemonade)) * total

theorem lemonade_amount :
  lemonadeInPitcher recipe totalAmount = 15 := by sorry

end lemonade_amount_l1241_124175


namespace second_smallest_pack_count_l1241_124182

def hot_dogs_per_pack : ℕ := 12
def buns_per_pack : ℕ := 10
def leftover_hot_dogs : ℕ := 6

def is_valid_pack_count (n : ℕ) : Prop :=
  (hot_dogs_per_pack * n) % buns_per_pack = leftover_hot_dogs

theorem second_smallest_pack_count : 
  ∃ (n : ℕ), is_valid_pack_count n ∧ 
    (∃ (m : ℕ), m < n ∧ is_valid_pack_count m) ∧
    (∀ (k : ℕ), k < n → is_valid_pack_count k → k ≤ m) ∧
    n = 8 :=
sorry

end second_smallest_pack_count_l1241_124182


namespace age_divisibility_l1241_124186

theorem age_divisibility (a : ℤ) : 10 ∣ (a^5 - a) := by
  sorry

end age_divisibility_l1241_124186


namespace exam_average_l1241_124198

theorem exam_average (total_boys : ℕ) (passed_boys : ℕ) (passed_avg : ℚ) (failed_avg : ℚ) 
  (h1 : total_boys = 120)
  (h2 : passed_boys = 115)
  (h3 : passed_avg = 39)
  (h4 : failed_avg = 15) :
  (passed_boys * passed_avg + (total_boys - passed_boys) * failed_avg) / total_boys = 38 := by
  sorry

end exam_average_l1241_124198


namespace intersection_A_complement_B_l1241_124181

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end intersection_A_complement_B_l1241_124181


namespace reflection_result_l1241_124124

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def C : ℝ × ℝ := (3, 3)

theorem reflection_result :
  (reflect_over_x_axis ∘ reflect_over_y_axis) C = (-3, -3) := by
sorry

end reflection_result_l1241_124124


namespace alloy_mixture_l1241_124126

/-- The percentage of chromium in the first alloy -/
def chromium_percent_1 : ℝ := 12

/-- The percentage of chromium in the second alloy -/
def chromium_percent_2 : ℝ := 10

/-- The amount of the first alloy used (in kg) -/
def amount_1 : ℝ := 15

/-- The percentage of chromium in the new alloy -/
def chromium_percent_new : ℝ := 10.6

/-- The amount of the second alloy used (in kg) -/
def amount_2 : ℝ := 35

theorem alloy_mixture :
  chromium_percent_1 * amount_1 / 100 + chromium_percent_2 * amount_2 / 100 =
  chromium_percent_new * (amount_1 + amount_2) / 100 :=
by sorry

end alloy_mixture_l1241_124126


namespace inequality_of_positive_numbers_l1241_124132

theorem inequality_of_positive_numbers (a₁ a₂ a₃ : ℝ) (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) :
  (a₁ * a₂) / a₃ + (a₂ * a₃) / a₁ + (a₃ * a₁) / a₂ ≥ a₁ + a₂ + a₃ := by
  sorry

end inequality_of_positive_numbers_l1241_124132


namespace benzene_required_l1241_124146

-- Define the chemical reaction
structure ChemicalReaction where
  benzene : ℕ
  methane : ℕ
  toluene : ℕ
  hydrogen : ℕ

-- Define the balanced equation
def balanced_equation : ChemicalReaction :=
  { benzene := 1, methane := 1, toluene := 1, hydrogen := 1 }

-- Define the given amounts
def given_amounts : ChemicalReaction :=
  { benzene := 0, methane := 2, toluene := 2, hydrogen := 2 }

-- Theorem to prove
theorem benzene_required (r : ChemicalReaction) :
  r.methane = 2 * balanced_equation.methane ∧
  r.toluene = 2 * balanced_equation.toluene ∧
  r.hydrogen = 2 * balanced_equation.hydrogen →
  r.benzene = 2 * balanced_equation.benzene :=
by sorry

end benzene_required_l1241_124146


namespace function_inequality_range_l1241_124159

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + x^2 - x * Real.log a

theorem function_inequality_range (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → 
    |f a x₁ - f a x₂| ≤ a - 2) ↔ 
  a ∈ Set.Ici (Real.exp 2) :=
sorry

end function_inequality_range_l1241_124159


namespace arithmetic_sequence_ninth_term_l1241_124170

/-- An arithmetic sequence with a_1 = 3 and a_5 = 7 has its 9th term equal to 11 -/
theorem arithmetic_sequence_ninth_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
    a 1 = 3 →                                -- first term condition
    a 5 = 7 →                                -- fifth term condition
    a 9 = 11 := by
  sorry

end arithmetic_sequence_ninth_term_l1241_124170


namespace exponential_always_positive_l1241_124187

theorem exponential_always_positive : ¬∃ (x : ℝ), Real.exp x ≤ 0 := by sorry

end exponential_always_positive_l1241_124187


namespace school_selections_l1241_124196

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem school_selections : 
  (choose 6 3) * (choose 5 2) = 200 := by
sorry

end school_selections_l1241_124196


namespace min_phase_shift_symmetric_cosine_l1241_124166

/-- Given a cosine function with a specific symmetry point, prove the minimum absolute value of its phase shift. -/
theorem min_phase_shift_symmetric_cosine (φ : ℝ) : 
  (∀ x, 3 * Real.cos (2 * x + φ) = 3 * Real.cos (2 * (8 * π / 3 - x) + φ)) → 
  (∃ k : ℤ, φ = k * π - 13 * π / 6) →
  (∀ ψ : ℝ, (∃ k : ℤ, ψ = k * π - 13 * π / 6) → |φ| ≤ |ψ|) →
  |φ| = π / 6 := by
  sorry

end min_phase_shift_symmetric_cosine_l1241_124166


namespace cos_240_degrees_l1241_124129

theorem cos_240_degrees : Real.cos (240 * π / 180) = -(1 / 2) := by
  sorry

end cos_240_degrees_l1241_124129


namespace sphere_volume_right_triangular_pyramid_l1241_124185

/-- The volume of a sphere circumscribing a right triangular pyramid with specific edge lengths -/
theorem sphere_volume_right_triangular_pyramid :
  let edge1 : ℝ := Real.sqrt 3
  let edge2 : ℝ := 2
  let edge3 : ℝ := 3
  let sphere_volume := (4 / 3) * Real.pi * (edge1^2 + edge2^2 + edge3^2)^(3/2) / 8
  sphere_volume = 32 * Real.pi / 3 := by
  sorry

end sphere_volume_right_triangular_pyramid_l1241_124185


namespace polynomial_roots_and_factorization_l1241_124101

theorem polynomial_roots_and_factorization (m : ℤ) : 
  (∀ x : ℤ, 2 * x^4 + m * x^2 + 8 = 0 → 
    (∃ a b c d : ℤ, x = a ∨ x = b ∨ x = c ∨ x = d)) →
  (m = -10 ∧ 
   ∀ x : ℤ, 2 * x^4 + m * x^2 + 8 = 2 * (x + 1) * (x - 1) * (x + 2) * (x - 2)) :=
by sorry

end polynomial_roots_and_factorization_l1241_124101


namespace aron_cleaning_time_l1241_124152

/-- Represents the cleaning schedule and calculates total cleaning time -/
def cleaning_schedule (vacuum_time : ℕ) (vacuum_days : ℕ) (dust_time : ℕ) (dust_days : ℕ) : ℕ :=
  vacuum_time * vacuum_days + dust_time * dust_days

/-- Theorem stating that Aron's total cleaning time per week is 130 minutes -/
theorem aron_cleaning_time : 
  cleaning_schedule 30 3 20 2 = 130 := by
  sorry

end aron_cleaning_time_l1241_124152


namespace sum_of_fractions_eq_9900_l1241_124115

/-- The sum of all fractions in lowest terms with denominator 3, 
    greater than 10 and less than 100 -/
def sum_of_fractions : ℚ :=
  (Finset.filter (fun n => n % 3 ≠ 0) (Finset.range 269)).sum (fun n => (n + 31 : ℚ) / 3)

/-- Theorem stating that the sum of fractions is equal to 9900 -/
theorem sum_of_fractions_eq_9900 : sum_of_fractions = 9900 := by
  sorry

end sum_of_fractions_eq_9900_l1241_124115


namespace brother_birthday_and_carlos_age_l1241_124102

def days_to_weekday (start_day : Nat) (days : Nat) : Nat :=
  (start_day + days) % 7

def years_from_days (days : Nat) : Nat :=
  days / 365

theorem brother_birthday_and_carlos_age 
  (start_day : Nat) 
  (carlos_age : Nat) 
  (days_until_brother_birthday : Nat) :
  start_day = 2 → 
  carlos_age = 7 → 
  days_until_brother_birthday = 2000 → 
  days_to_weekday start_day days_until_brother_birthday = 0 ∧ 
  years_from_days days_until_brother_birthday + carlos_age = 12 :=
by sorry

end brother_birthday_and_carlos_age_l1241_124102


namespace fourth_day_distance_l1241_124134

def distance_on_day (initial_distance : ℕ) (day : ℕ) : ℕ :=
  initial_distance * 2^(day - 1)

theorem fourth_day_distance (initial_distance : ℕ) :
  initial_distance = 18 → distance_on_day initial_distance 4 = 144 :=
by
  sorry

end fourth_day_distance_l1241_124134


namespace intersection_and_parallel_perpendicular_lines_l1241_124173

-- Define the lines
def l₁ (x y : ℝ) : Prop := x - 2*y + 4 = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0
def l₃ (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Define point P as the intersection of l₁ and l₂
def P : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem intersection_and_parallel_perpendicular_lines :
  -- P is on both l₁ and l₂
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) ∧ 
  -- Parallel line equation
  (∀ x y : ℝ, 3*x - 4*y + 8 = 0 ↔ (y - P.2 = (3/4) * (x - P.1))) ∧
  -- Perpendicular line equation
  (∀ x y : ℝ, 4*x + 3*y - 6 = 0 ↔ (y - P.2 = -(4/3) * (x - P.1))) :=
sorry

end intersection_and_parallel_perpendicular_lines_l1241_124173


namespace cubic_root_product_l1241_124128

theorem cubic_root_product (a b c : ℝ) : 
  (a^3 - 18*a^2 + 20*a - 8 = 0) ∧ 
  (b^3 - 18*b^2 + 20*b - 8 = 0) ∧ 
  (c^3 - 18*c^2 + 20*c - 8 = 0) →
  (2+a)*(2+b)*(2+c) = 128 := by
sorry

end cubic_root_product_l1241_124128


namespace roberto_outfits_l1241_124150

/-- The number of different outfits Roberto can put together -/
def number_of_outfits (trousers shirts jackets belts : ℕ) 
  (restricted_jacket_trousers : ℕ) : ℕ :=
  let unrestricted_jackets := jackets - 1
  let unrestricted_combinations := trousers * shirts * unrestricted_jackets * belts
  let restricted_combinations := restricted_jacket_trousers * shirts * belts
  let overlapping_combinations := (trousers - restricted_jacket_trousers) * shirts * belts
  unrestricted_combinations + restricted_combinations - overlapping_combinations

/-- Theorem stating the number of outfits Roberto can put together -/
theorem roberto_outfits : 
  number_of_outfits 5 7 4 2 3 = 168 := by
  sorry

#eval number_of_outfits 5 7 4 2 3

end roberto_outfits_l1241_124150


namespace abc_sum_mod_7_l1241_124112

theorem abc_sum_mod_7 (a b c : ℕ) : 
  0 < a ∧ a < 7 ∧ 
  0 < b ∧ b < 7 ∧ 
  0 < c ∧ c < 7 ∧ 
  (a * b * c) % 7 = 1 ∧ 
  (5 * c) % 7 = 2 ∧ 
  (6 * b) % 7 = (3 + b) % 7 → 
  (a + b + c) % 7 = 4 := by
sorry

end abc_sum_mod_7_l1241_124112


namespace square_difference_fraction_l1241_124125

theorem square_difference_fraction (x y : ℚ) 
  (h1 : x + y = 9/17) (h2 : x - y = 1/51) : x^2 - y^2 = 1/289 := by
  sorry

end square_difference_fraction_l1241_124125


namespace sum_of_numbers_ge_04_l1241_124153

theorem sum_of_numbers_ge_04 : 
  let numbers : List ℚ := [4/5, 1/2, 9/10, 1/3]
  (numbers.filter (λ x => x ≥ 2/5)).sum = 11/5 := by
  sorry

end sum_of_numbers_ge_04_l1241_124153


namespace equation_graph_is_axes_l1241_124119

/-- The set of points (x, y) satisfying the equation (x-y)^2 = x^2 + y^2 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2}

/-- The union of the x-axis and y-axis -/
def XYAxes : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0}

theorem equation_graph_is_axes : S = XYAxes := by
  sorry

end equation_graph_is_axes_l1241_124119


namespace point_P_coordinates_l1241_124139

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, -3)

theorem point_P_coordinates :
  ∃ P : ℝ × ℝ, (P.1 - A.1, P.2 - A.2) = 3 • (B.1 - A.1, B.2 - A.2) ∧ P = (8, -15) := by
  sorry

end point_P_coordinates_l1241_124139


namespace steve_distance_theorem_l1241_124106

def steve_problem (distance : ℝ) : Prop :=
  let speed_to_work : ℝ := 17.5 / 2
  let speed_from_work : ℝ := 17.5
  let time_to_work : ℝ := distance / speed_to_work
  let time_from_work : ℝ := distance / speed_from_work
  (time_to_work + time_from_work = 6) ∧ (speed_from_work = 2 * speed_to_work)

theorem steve_distance_theorem : 
  ∃ (distance : ℝ), steve_problem distance ∧ distance = 35 := by
  sorry

end steve_distance_theorem_l1241_124106


namespace piggy_bank_problem_l1241_124172

theorem piggy_bank_problem (total_money : ℕ) (total_bills : ℕ) 
  (h1 : total_money = 66) 
  (h2 : total_bills = 49) : 
  ∃ (one_dollar_bills two_dollar_bills : ℕ), 
    one_dollar_bills + two_dollar_bills = total_bills ∧ 
    one_dollar_bills + 2 * two_dollar_bills = total_money ∧
    one_dollar_bills = 32 :=
by sorry

end piggy_bank_problem_l1241_124172


namespace equation_solution_l1241_124161

theorem equation_solution : 
  ∃! x : ℝ, (3 / x = 2 / (x - 2)) ∧ (x ≠ 0) ∧ (x ≠ 2) := by
  sorry

end equation_solution_l1241_124161


namespace smallest_three_digit_multiple_of_3_and_5_l1241_124137

theorem smallest_three_digit_multiple_of_3_and_5 : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧  -- three-digit number
  (n % 3 = 0 ∧ n % 5 = 0) ∧  -- multiple of 3 and 5
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (m % 3 = 0 ∧ m % 5 = 0) → m ≥ n) ∧  -- smallest such number
  n = 105 :=
by sorry

end smallest_three_digit_multiple_of_3_and_5_l1241_124137


namespace cars_to_trucks_ratio_l1241_124162

theorem cars_to_trucks_ratio (total_vehicles : ℕ) (trucks : ℕ) 
  (h1 : total_vehicles = 60) (h2 : trucks = 20) : 
  (total_vehicles - trucks) / trucks = 2 := by
  sorry

end cars_to_trucks_ratio_l1241_124162


namespace olgas_fish_colors_l1241_124188

theorem olgas_fish_colors (total : ℕ) (yellow : ℕ) (blue : ℕ) (green : ℕ)
  (h_total : total = 42)
  (h_yellow : yellow = 12)
  (h_blue : blue = yellow / 2)
  (h_green : green = yellow * 2)
  (h_sum : total = yellow + blue + green) :
  ∃ (num_colors : ℕ), num_colors = 3 ∧ num_colors > 0 := by
sorry

end olgas_fish_colors_l1241_124188


namespace sequence_difference_l1241_124145

theorem sequence_difference (p q : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) : 
  (∀ n, S n = n^2 - 5*n) → 
  (∀ n, a (n+1) = S (n+1) - S n) →
  p - q = 4 →
  a p - a q = 8 := by
sorry

end sequence_difference_l1241_124145


namespace twentyseven_binary_l1241_124108

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Convert a list of booleans to a natural number in binary -/
def fromBinary (l : List Bool) : ℕ :=
  l.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem twentyseven_binary :
  toBinary 27 = [true, true, false, true, true] :=
sorry

end twentyseven_binary_l1241_124108


namespace blue_fish_count_l1241_124114

theorem blue_fish_count (total_fish goldfish : ℕ) (h1 : total_fish = 22) (h2 : goldfish = 15) :
  total_fish - goldfish = 7 := by
  sorry

end blue_fish_count_l1241_124114


namespace sine_cosine_roots_l1241_124117

theorem sine_cosine_roots (θ : Real) (m : Real) : 
  (∃ (x y : Real), x = Real.sin θ ∧ y = Real.cos θ ∧ 
   4 * x^2 + 2 * m * x + m = 0 ∧ 
   4 * y^2 + 2 * m * y + m = 0) →
  m = 1 - Real.sqrt 5 := by
sorry

end sine_cosine_roots_l1241_124117


namespace tangent_line_slope_l1241_124190

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - 3*x - 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 3

-- Theorem statement
theorem tangent_line_slope (k : ℝ) :
  (∃ x₀ : ℝ, f x₀ = k * x₀ + 2 ∧ f' x₀ = k) → k = 2 :=
by sorry

end tangent_line_slope_l1241_124190


namespace sets_equal_implies_a_value_l1241_124154

-- Define the sets A, B, and C
def A (a : ℝ) := {x : ℝ | -1 ≤ x ∧ x ≤ a}
def B (a : ℝ) := {y : ℝ | ∃ x ∈ A a, y = x + 1}
def C (a : ℝ) := {y : ℝ | ∃ x ∈ A a, y = x^2}

-- State the theorem
theorem sets_equal_implies_a_value (a : ℝ) (h1 : a > -1) (h2 : B a = C a) :
  a = 0 ∨ a = (1 + Real.sqrt 5) / 2 := by
  sorry

end sets_equal_implies_a_value_l1241_124154


namespace furniture_dealer_tables_l1241_124199

/-- The number of four-legged tables -/
def F : ℕ := 16

/-- The number of three-legged tables -/
def T : ℕ := (124 - 4 * F) / 3

/-- The total number of tables -/
def total_tables : ℕ := F + T

/-- Theorem stating that the total number of tables is 36 -/
theorem furniture_dealer_tables : total_tables = 36 := by
  sorry

end furniture_dealer_tables_l1241_124199


namespace logan_grocery_budget_l1241_124118

/-- Calculates the amount Logan can spend on groceries annually given his financial parameters. -/
def grocery_budget (current_income : ℕ) (income_increase : ℕ) (rent : ℕ) (gas : ℕ) (desired_savings : ℕ) : ℕ :=
  (current_income + income_increase) - (rent + gas + desired_savings)

/-- Theorem stating that Logan's grocery budget is $5,000 given his financial parameters. -/
theorem logan_grocery_budget :
  grocery_budget 65000 10000 20000 8000 42000 = 5000 := by
  sorry

end logan_grocery_budget_l1241_124118


namespace consecutive_sum_not_power_of_two_l1241_124192

theorem consecutive_sum_not_power_of_two (n k x : ℕ) (h : n > 1) :
  (n * (n + 2 * k - 1)) / 2 ≠ 2^x :=
sorry

end consecutive_sum_not_power_of_two_l1241_124192


namespace delightful_numbers_l1241_124149

def is_delightful (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  n % 25 = 0 ∧
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) % 25 = 0 ∧
  ((n / 1000) * (n / 100 % 10) * (n / 10 % 10) * (n % 10)) % 25 = 0

theorem delightful_numbers :
  ∀ n : ℕ, is_delightful n ↔ n = 5875 ∨ n = 8575 := by sorry

end delightful_numbers_l1241_124149

import Mathlib

namespace fourth_angle_is_85_l398_39804

/-- A quadrilateral with three known angles -/
structure Quadrilateral where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  sum_360 : angle1 + angle2 + angle3 + angle4 = 360

/-- The theorem stating that the fourth angle is 85° -/
theorem fourth_angle_is_85 (q : Quadrilateral) 
  (h1 : q.angle1 = 75) 
  (h2 : q.angle2 = 80) 
  (h3 : q.angle3 = 120) : 
  q.angle4 = 85 := by
  sorry


end fourth_angle_is_85_l398_39804


namespace range_of_a_l398_39890

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + 4 * x + a > 0) → a > 2 := by
  sorry

end range_of_a_l398_39890


namespace calculation_proof_l398_39847

theorem calculation_proof :
  (- (1 : ℤ)^4 + 16 / (-2 : ℤ)^3 * |-3 - 1| = -3) ∧
  (∀ a b : ℝ, -2 * (a^2 * b - 1/4 * a * b^2 + 1/2 * a^3) - (-2 * a^2 * b + 3 * a * b^2) = -5/2 * a * b^2 - a^3) := by
  sorry

end calculation_proof_l398_39847


namespace exists_k_for_circle_through_E_l398_39838

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line equation -/
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

/-- The fixed point E -/
def point_E : ℝ × ℝ := (-1, 0)

/-- Predicate to check if a circle with CD as diameter passes through E -/
def circle_passes_through_E (C D : ℝ × ℝ) : Prop :=
  let (x1, y1) := C
  let (x2, y2) := D
  y1 / (x1 + 1) * y2 / (x2 + 1) = -1

/-- The main theorem -/
theorem exists_k_for_circle_through_E :
  ∃ k : ℝ, k ≠ 0 ∧ k = 7/6 ∧
  ∃ C D : ℝ × ℝ,
    ellipse C.1 C.2 ∧
    ellipse D.1 D.2 ∧
    line k C.1 C.2 ∧
    line k D.1 D.2 ∧
    circle_passes_through_E C D :=
sorry

end exists_k_for_circle_through_E_l398_39838


namespace division_value_proof_l398_39824

theorem division_value_proof (x : ℝ) : (2.25 / x) * 12 = 9 → x = 3 := by
  sorry

end division_value_proof_l398_39824


namespace lineup_count_l398_39853

/-- The number of ways to choose a lineup from a basketball team with specific constraints. -/
def chooseLineup (totalPlayers : ℕ) (twinCount : ℕ) (tripletCount : ℕ) (lineupSize : ℕ) : ℕ :=
  let nonSpecialPlayers := totalPlayers - twinCount - tripletCount
  let noSpecial := Nat.choose nonSpecialPlayers lineupSize
  let oneTriplet := tripletCount * Nat.choose nonSpecialPlayers (lineupSize - 1)
  let oneTwin := twinCount * Nat.choose nonSpecialPlayers (lineupSize - 1)
  let oneTripletOneTwin := tripletCount * twinCount * Nat.choose nonSpecialPlayers (lineupSize - 2)
  noSpecial + oneTriplet + oneTwin + oneTripletOneTwin

/-- The theorem stating the number of ways to choose the lineup under given constraints. -/
theorem lineup_count :
  chooseLineup 16 2 3 5 = 3102 :=
by sorry

end lineup_count_l398_39853


namespace not_necessarily_parallel_l398_39811

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the theorem
theorem not_necessarily_parallel
  (m : Line) (α β : Plane)
  (h1 : parallel_plane_plane α β)
  (h2 : parallel_line_plane m α) :
  ¬ (∀ m α β, parallel_plane_plane α β → parallel_line_plane m α → parallel_line_plane m β) :=
sorry

end not_necessarily_parallel_l398_39811


namespace cube_difference_given_difference_l398_39854

theorem cube_difference_given_difference (x : ℝ) (h : x - 1/x = 5) :
  x^3 - 1/x^3 = 140 := by
sorry

end cube_difference_given_difference_l398_39854


namespace complex_magnitude_three_fourths_minus_five_sixths_i_l398_39867

theorem complex_magnitude_three_fourths_minus_five_sixths_i :
  Complex.abs (3/4 - Complex.I * 5/6) = Real.sqrt 181 / 12 := by
  sorry

end complex_magnitude_three_fourths_minus_five_sixths_i_l398_39867


namespace shortest_distance_between_circles_l398_39871

/-- Circle1 is defined by the equation x^2 - 6x + y^2 + 10y + 9 = 0 -/
def Circle1 (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 10*y + 9 = 0

/-- Circle2 is defined by the equation x^2 + 4x + y^2 - 8y + 4 = 0 -/
def Circle2 (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 8*y + 4 = 0

/-- The shortest distance between Circle1 and Circle2 is √106 - 9 -/
theorem shortest_distance_between_circles :
  ∃ (d : ℝ), d = Real.sqrt 106 - 9 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    Circle1 x₁ y₁ → Circle2 x₂ y₂ →
    d ≤ Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by sorry

end shortest_distance_between_circles_l398_39871


namespace arithmetic_equality_l398_39877

theorem arithmetic_equality : (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end arithmetic_equality_l398_39877


namespace line_through_coefficient_points_l398_39826

/-- Given two lines that pass through a common point, prove that the line
    passing through the points defined by their coefficients has a specific equation. -/
theorem line_through_coefficient_points
  (a₁ b₁ a₂ b₂ : ℝ)
  (h₁ : 2 * a₁ + 3 * b₁ + 1 = 0)
  (h₂ : 2 * a₂ + 3 * b₂ + 1 = 0) :
  ∀ (x y : ℝ), (x = a₁ ∧ y = b₁) ∨ (x = a₂ ∧ y = b₂) → 2 * x + 3 * y + 1 = 0 :=
by sorry

end line_through_coefficient_points_l398_39826


namespace parallel_vectors_implies_y_eq_neg_four_l398_39856

/-- Two vectors in ℝ² -/
def a : Fin 2 → ℝ := ![1, 2]
def b (y : ℝ) : Fin 2 → ℝ := ![-2, y]

/-- Parallel vectors in ℝ² have proportional coordinates -/
def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ i, v i = k * u i

/-- If a and b are parallel plane vectors, then y = -4 -/
theorem parallel_vectors_implies_y_eq_neg_four :
  parallel a (b y) → y = -4 := by
  sorry

end parallel_vectors_implies_y_eq_neg_four_l398_39856


namespace f_range_theorem_l398_39846

def f (x m : ℝ) : ℝ := |x + 1| + |x - m|

theorem f_range_theorem :
  (∀ m : ℝ, (∀ x : ℝ, f x m ≥ 3) ↔ m ∈ Set.Ici 2 ∪ Set.Iic (-4)) ∧
  (∀ m : ℝ, (∃ x : ℝ, f m m - 2*m ≥ x^2 - x) ↔ m ∈ Set.Iic (5/4)) := by
  sorry

end f_range_theorem_l398_39846


namespace quadratic_function_theorem_l398_39891

/-- A quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The function is decreasing on the interval (-∞, 4] -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 4 → f a x > f a y

/-- The range of values for a -/
def a_range (a : ℝ) : Prop := a ≤ -3

theorem quadratic_function_theorem (a : ℝ) :
  is_decreasing_on_interval a → a_range a :=
sorry

end quadratic_function_theorem_l398_39891


namespace negative_three_squared_l398_39810

theorem negative_three_squared : (-3 : ℤ) ^ 2 = 9 := by
  sorry

end negative_three_squared_l398_39810


namespace base5_43102_equals_2902_l398_39827

def base5_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5^(digits.length - 1 - i))) 0

theorem base5_43102_equals_2902 :
  base5_to_decimal [4, 3, 1, 0, 2] = 2902 := by
  sorry

end base5_43102_equals_2902_l398_39827


namespace min_value_quadratic_sum_l398_39840

theorem min_value_quadratic_sum (a b c t k : ℝ) (hsum : a + b + c = t) (hk : k > 0) :
  k * a^2 + b^2 + k * c^2 ≥ k * t^2 / (k + 2) := by
  sorry

end min_value_quadratic_sum_l398_39840


namespace sport_water_amount_l398_39800

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  { flavoring := 1,
    corn_syrup := 4 * standard_ratio.flavoring / standard_ratio.corn_syrup,
    water := 2 * standard_ratio.water / standard_ratio.flavoring }

/-- Amount of corn syrup in the sport formulation bottle (in ounces) -/
def sport_corn_syrup : ℚ := 7

/-- Theorem stating the amount of water in the sport formulation bottle -/
theorem sport_water_amount :
  (sport_ratio.water / sport_ratio.corn_syrup) * sport_corn_syrup = 105 := by
  sorry

end sport_water_amount_l398_39800


namespace combination_equation_solution_l398_39868

theorem combination_equation_solution (n : ℕ+) : 
  (Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8) → n = 14 := by
  sorry

end combination_equation_solution_l398_39868


namespace store_pricing_l398_39837

-- Define variables for the prices of individual items
variable (p n e : ℝ)

-- Define the equations based on the given conditions
def equation1 : Prop := 10 * p + 12 * n + 6 * e = 5.50
def equation2 : Prop := 6 * p + 4 * n + 3 * e = 2.40

-- Define the final cost calculation
def final_cost : ℝ := 20 * p + 15 * n + 9 * e

-- Theorem statement
theorem store_pricing (h1 : equation1 p n e) (h2 : equation2 p n e) : 
  final_cost p n e = 8.95 := by
  sorry


end store_pricing_l398_39837


namespace movie_theater_total_movies_l398_39812

/-- Calculates the total number of movies shown in a movie theater. -/
def total_movies_shown (num_screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ) : ℕ :=
  num_screens * (open_hours / movie_duration)

/-- Proves that a movie theater with 6 screens, open for 8 hours, showing 2-hour movies, shows 24 movies total. -/
theorem movie_theater_total_movies :
  total_movies_shown 6 8 2 = 24 := by
  sorry

#eval total_movies_shown 6 8 2

end movie_theater_total_movies_l398_39812


namespace no_solutions_lcm_gcd_equation_l398_39844

theorem no_solutions_lcm_gcd_equation : 
  ¬∃ (n : ℕ+), Nat.lcm n 120 = Nat.gcd n 120 + 360 := by
  sorry

end no_solutions_lcm_gcd_equation_l398_39844


namespace stifel_conjecture_counterexample_l398_39851

theorem stifel_conjecture_counterexample : ∃ n : ℕ, ¬ Nat.Prime (2^(2*n + 1) - 1) := by
  sorry

end stifel_conjecture_counterexample_l398_39851


namespace max_value_implies_a_l398_39834

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := -9 * x^2 - 6 * a * x + 2 * a - a^2

/-- The theorem stating the relationship between the maximum value of f and the value of a -/
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f a x ≤ -3) ∧
  (∃ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f a x = -3) ↔
  a = Real.sqrt 6 + 2 :=
sorry

end max_value_implies_a_l398_39834


namespace min_movements_for_ten_l398_39884

/-- Represents a circular arrangement of n distinct elements -/
def CircularArrangement (n : ℕ) := Fin n → ℕ

/-- A single movement in the circular arrangement -/
def Movement (n : ℕ) (arr : CircularArrangement n) (i j : Fin n) : CircularArrangement n :=
  sorry

/-- Checks if the circular arrangement is sorted in ascending order clockwise -/
def IsSorted (n : ℕ) (arr : CircularArrangement n) : Prop :=
  sorry

/-- The minimum number of movements required to sort the arrangement -/
def MinMovements (n : ℕ) (arr : CircularArrangement n) : ℕ :=
  sorry

/-- Theorem: For 10 distinct elements, 8 movements are always sufficient and necessary -/
theorem min_movements_for_ten :
  ∀ (arr : CircularArrangement 10),
    (∀ i j : Fin 10, i ≠ j → arr i ≠ arr j) →
    MinMovements 10 arr = 8 :=
by sorry

end min_movements_for_ten_l398_39884


namespace lemonade_pitcher_capacity_l398_39823

/-- Given that 30 glasses of lemonade were served from 6 pitchers, 
    prove that each pitcher can serve 5 glasses. -/
theorem lemonade_pitcher_capacity 
  (total_glasses : ℕ) 
  (total_pitchers : ℕ) 
  (h1 : total_glasses = 30) 
  (h2 : total_pitchers = 6) : 
  total_glasses / total_pitchers = 5 := by
sorry

end lemonade_pitcher_capacity_l398_39823


namespace coronavirus_cases_l398_39895

theorem coronavirus_cases (initial_cases : ℕ) : 
  initial_cases > 0 →
  initial_cases + 450 + 1300 = 3750 →
  initial_cases = 2000 := by
sorry

end coronavirus_cases_l398_39895


namespace circle_intersection_properties_l398_39839

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Theorem statement
theorem circle_intersection_properties :
  -- 1. The equation of the line containing AB is x - y = 0
  (∀ (x y : ℝ), (x - y = 0) ↔ (∃ (t : ℝ), x = t * A.1 + (1 - t) * B.1 ∧ y = t * A.2 + (1 - t) * B.2)) ∧
  -- 2. The equation of the perpendicular bisector of AB is x + y - 1 = 0
  (∀ (x y : ℝ), (x + y - 1 = 0) ↔ ((x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2)) ∧
  -- 3. The maximum distance from a point P on O₂ to the line AB is (3√2)/2 + √5
  (∃ (P : ℝ × ℝ), circle_O2 P.1 P.2 ∧
    ∀ (Q : ℝ × ℝ), circle_O2 Q.1 Q.2 →
      abs ((Q.1 - Q.2) / Real.sqrt 2) ≤ (3 * Real.sqrt 2) / 2 + Real.sqrt 5) :=
by sorry

end circle_intersection_properties_l398_39839


namespace expression_evaluation_l398_39814

theorem expression_evaluation : 
  Real.sqrt 3 * Real.cos (30 * π / 180) + (3 - π)^0 - 2 * Real.tan (45 * π / 180) = 1/2 := by
  sorry

end expression_evaluation_l398_39814


namespace probability_two_primary_schools_l398_39836

/-- Represents the types of schools in the region -/
inductive SchoolType
| Primary
| Middle
| University

/-- Represents the total number of schools of each type -/
def totalSchools : SchoolType → ℕ
| SchoolType.Primary => 21
| SchoolType.Middle => 14
| SchoolType.University => 7

/-- Represents the number of schools selected in the stratified sample -/
def selectedSchools : SchoolType → ℕ
| SchoolType.Primary => 3
| SchoolType.Middle => 2
| SchoolType.University => 1

/-- The total number of schools in the stratified sample -/
def totalSampleSize : ℕ := 6

/-- The number of schools to be randomly selected from the sample -/
def selectionSize : ℕ := 2

/-- Theorem stating that the probability of selecting two primary schools
    from the stratified sample is 1/5 -/
theorem probability_two_primary_schools :
  (selectedSchools SchoolType.Primary).choose selectionSize /
  (totalSampleSize.choose selectionSize) = 1 / 5 := by
  sorry

end probability_two_primary_schools_l398_39836


namespace total_cost_is_correct_l398_39830

-- Define ticket prices
def adult_price : ℝ := 11
def child_price : ℝ := 8
def senior_price : ℝ := 9

-- Define discounts
def husband_discount : ℝ := 0.25
def parents_discount : ℝ := 0.15
def nephew_discount : ℝ := 0.10

-- Define group composition
def num_adults : ℕ := 4
def num_children : ℕ := 2
def num_seniors : ℕ := 3
def num_teens : ℕ := 1
def num_adult_nephews : ℕ := 1

-- Define the total cost function
def total_cost : ℝ :=
  (num_adults * adult_price) +
  (num_children * child_price) +
  (num_seniors * senior_price) +
  (num_teens * adult_price) +
  (num_adult_nephews * adult_price) -
  (husband_discount * adult_price) -
  (parents_discount * (2 * senior_price)) -
  (nephew_discount * adult_price)

-- Theorem statement
theorem total_cost_is_correct :
  total_cost = 110.45 := by sorry

end total_cost_is_correct_l398_39830


namespace protest_days_calculation_l398_39880

/-- Calculates the number of days of protest given the conditions of the problem. -/
def daysOfProtest (
  numCities : ℕ)
  (arrestsPerDay : ℕ)
  (preTrialDays : ℕ)
  (sentenceDays : ℕ)
  (totalJailWeeks : ℕ) : ℕ :=
  let totalJailDays := totalJailWeeks * 7
  let daysPerPerson := preTrialDays + sentenceDays / 2
  let totalArrests := totalJailDays / daysPerPerson
  let totalProtestDays := totalArrests / arrestsPerDay
  totalProtestDays / numCities

/-- Theorem stating that given the conditions of the problem, there were 30 days of protest. -/
theorem protest_days_calculation :
  daysOfProtest 21 10 4 14 9900 = 30 := by
  sorry

end protest_days_calculation_l398_39880


namespace fourth_rectangle_area_l398_39898

theorem fourth_rectangle_area (P Q R S : ℝ × ℝ) : 
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = 25 →
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 49 →
  (S.1 - R.1)^2 + (S.2 - R.2)^2 = 64 →
  (Q.2 - P.2) * (R.1 - P.1) = (Q.1 - P.1) * (R.2 - P.2) →
  (S.2 - P.2) * (R.1 - P.1) = (S.1 - P.1) * (R.2 - P.2) →
  (S.1 - P.1)^2 + (S.2 - P.2)^2 = 89 := by
sorry

end fourth_rectangle_area_l398_39898


namespace proposition_p_and_q_l398_39842

def is_ellipse (m : ℝ) : Prop :=
  1 < m ∧ m < 3 ∧ m ≠ 2

def no_common_points (m : ℝ) : Prop :=
  m > Real.sqrt 5 / 2 ∨ m < -Real.sqrt 5 / 2

theorem proposition_p_and_q (m : ℝ) :
  (is_ellipse m ∧ no_common_points m) ↔ 
  (Real.sqrt 5 / 2 < m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end proposition_p_and_q_l398_39842


namespace not_divisible_by_1000_power_minus_1_l398_39832

theorem not_divisible_by_1000_power_minus_1 (m : ℕ) : ¬(1000^m - 1 ∣ 1978^m - 1) := by
  sorry

end not_divisible_by_1000_power_minus_1_l398_39832


namespace angle_halving_l398_39888

/-- An angle is in the third quadrant if it's between π and 3π/2 (modulo 2π) -/
def is_third_quadrant (α : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2

/-- An angle is in the second or fourth quadrant if it's between π/2 and 3π/4 or between 3π/2 and 7π/4 (modulo 2π) -/
def is_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, (k * Real.pi + Real.pi / 2 < α ∧ α < k * Real.pi + 3 * Real.pi / 4) ∨
             (k * Real.pi + 3 * Real.pi / 2 < α ∧ α < k * Real.pi + 7 * Real.pi / 4)

theorem angle_halving (α : Real) :
  is_third_quadrant α → is_second_or_fourth_quadrant (α / 2) := by
  sorry

end angle_halving_l398_39888


namespace x_prime_condition_x_divisibility_l398_39897

def x : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => x (n + 1) + x n

theorem x_prime_condition (n : ℕ) (h : n ≥ 1) :
  Nat.Prime (x n) → Nat.Prime n ∨ (∀ p, Nat.Prime p → p > 2 → ¬p ∣ n) :=
sorry

theorem x_divisibility (m n : ℕ) :
  x m ∣ x n ↔ (∃ k, (m = 0 ∧ n = 3 * k) ∨ (m = 1 ∧ n = k) ∨ (∃ t, m = n ∧ n = (2 * t + 1) * n)) :=
sorry

end x_prime_condition_x_divisibility_l398_39897


namespace cyclic_sum_inequality_l398_39855

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h : a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) ≤ 1) : 
  1 / (b + c + 1) + 1 / (c + a + 1) + 1 / (a + b + 1) ≥ 1 := by
  sorry

end cyclic_sum_inequality_l398_39855


namespace fraction_equality_l398_39801

theorem fraction_equality (a b : ℝ) (h : a / b = 5 / 2) : (a - b) / a = 3 / 5 := by
  sorry

end fraction_equality_l398_39801


namespace custom_mult_three_two_l398_39899

/-- Custom multiplication operation -/
def custom_mult (a b : ℤ) : ℤ := a^2 + a*b - b^2

/-- Theorem stating that 3*2 equals 11 under the custom multiplication -/
theorem custom_mult_three_two : custom_mult 3 2 = 11 := by
  sorry

end custom_mult_three_two_l398_39899


namespace circumscribed_polygon_has_triangle_l398_39809

/-- A polygon circumscribed about a circle. -/
structure CircumscribedPolygon where
  /-- The number of sides in the polygon. -/
  n : ℕ
  /-- The lengths of the sides of the polygon. -/
  sides : Fin n → ℝ
  /-- All side lengths are positive. -/
  sides_pos : ∀ i, 0 < sides i

/-- Theorem: In any polygon circumscribed about a circle, 
    there exist three sides that can form a triangle. -/
theorem circumscribed_polygon_has_triangle (P : CircumscribedPolygon) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    P.sides i + P.sides j > P.sides k ∧
    P.sides j + P.sides k > P.sides i ∧
    P.sides k + P.sides i > P.sides j :=
sorry

end circumscribed_polygon_has_triangle_l398_39809


namespace exponent_multiplication_l398_39861

theorem exponent_multiplication (b : ℝ) : b * b^3 = b^4 := by
  sorry

end exponent_multiplication_l398_39861


namespace multiple_of_twenty_day_after_power_of_three_l398_39860

-- Part 1
theorem multiple_of_twenty (n : ℕ+) : ∃ k : ℤ, 4 * 6^n.val + 5^(n.val + 1) - 9 = 20 * k := by sorry

-- Part 2
theorem day_after_power_of_three : (3^100 % 7 : ℕ) + 1 = 5 := by sorry

end multiple_of_twenty_day_after_power_of_three_l398_39860


namespace counterexample_exists_l398_39862

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ Nat.Prime (n - 2) :=
  sorry

end counterexample_exists_l398_39862


namespace carnival_tickets_l398_39857

/-- The total number of tickets bought by a group of friends at a carnival. -/
def total_tickets (num_friends : ℕ) (tickets_per_friend : ℕ) : ℕ :=
  num_friends * tickets_per_friend

/-- Theorem stating that 6 friends buying 39 tickets each results in 234 total tickets. -/
theorem carnival_tickets : total_tickets 6 39 = 234 := by
  sorry

end carnival_tickets_l398_39857


namespace constant_term_is_99_l398_39882

-- Define the function q'
def q' (q : ℝ) (c : ℝ) : ℝ := 3 * q - 3 + c

-- Define the condition that (5')' = 132
axiom condition : q' (q' 5 99) 99 = 132

-- Theorem to prove
theorem constant_term_is_99 : ∃ c : ℝ, q' (q' 5 c) c = 132 ∧ c = 99 := by
  sorry

end constant_term_is_99_l398_39882


namespace harry_photo_reorganization_l398_39841

/-- Represents a photo album organization system -/
structure PhotoAlbumSystem where
  initialAlbums : Nat
  pagesPerAlbum : Nat
  initialPhotosPerPage : Nat
  newPhotosPerPage : Nat
  filledAlbums : Nat

/-- Calculates the number of photos on the last page of the partially filled album -/
def photosOnLastPage (system : PhotoAlbumSystem) : Nat :=
  let totalPhotos := system.initialAlbums * system.pagesPerAlbum * system.initialPhotosPerPage
  let totalPagesNeeded := (totalPhotos + system.newPhotosPerPage - 1) / system.newPhotosPerPage
  let pagesInFilledAlbums := system.filledAlbums * system.pagesPerAlbum
  let remainingPhotos := totalPhotos - pagesInFilledAlbums * system.newPhotosPerPage
  remainingPhotos % system.newPhotosPerPage

theorem harry_photo_reorganization :
  let system : PhotoAlbumSystem := {
    initialAlbums := 10,
    pagesPerAlbum := 35,
    initialPhotosPerPage := 4,
    newPhotosPerPage := 8,
    filledAlbums := 6
  }
  photosOnLastPage system = 0 := by
  sorry

end harry_photo_reorganization_l398_39841


namespace other_number_proof_l398_39802

/-- Given two positive integers with known HCF, LCM, and one of the numbers, prove the value of the other number -/
theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 16) (h2 : Nat.lcm a b = 396) (h3 : a = 36) : b = 176 := by
  sorry

end other_number_proof_l398_39802


namespace all_lines_pass_through_common_point_l398_39803

/-- A line in 2D space represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Checks if three numbers form a geometric progression -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

theorem all_lines_pass_through_common_point :
  ∀ l : Line, isGeometricProgression l.a l.b l.c →
  l.contains (-1) 1 := by sorry

end all_lines_pass_through_common_point_l398_39803


namespace equilateral_triangle_quadratic_ac_l398_39822

/-- A quadratic function f(x) = ax^2 + c whose graph intersects the coordinate axes 
    at the vertices of an equilateral triangle. -/
structure EquilateralTriangleQuadratic where
  a : ℝ
  c : ℝ
  is_equilateral : ∀ (x y : ℝ), y = a * x^2 + c → 
    (x = 0 ∨ y = 0) → 
    -- The three intersection points form an equilateral triangle
    ∃ (p q r : ℝ × ℝ), 
      (p.1 = 0 ∨ p.2 = 0) ∧ 
      (q.1 = 0 ∨ q.2 = 0) ∧ 
      (r.1 = 0 ∨ r.2 = 0) ∧
      (p.2 = a * p.1^2 + c) ∧
      (q.2 = a * q.1^2 + c) ∧
      (r.2 = a * r.1^2 + c) ∧
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = (q.1 - r.1)^2 + (q.2 - r.2)^2 ∧
      (q.1 - r.1)^2 + (q.2 - r.2)^2 = (r.1 - p.1)^2 + (r.2 - p.2)^2

/-- The product of a and c for an EquilateralTriangleQuadratic is -3. -/
theorem equilateral_triangle_quadratic_ac (f : EquilateralTriangleQuadratic) : 
  f.a * f.c = -3 := by
  sorry

end equilateral_triangle_quadratic_ac_l398_39822


namespace john_climbs_nine_flights_l398_39892

/-- The number of flights climbed given step height, flight height, and number of steps -/
def flights_climbed (step_height_inches : ℚ) (flight_height_feet : ℚ) (num_steps : ℕ) : ℚ :=
  (step_height_inches / 12 * num_steps) / flight_height_feet

/-- Theorem: John climbs 9 flights of stairs -/
theorem john_climbs_nine_flights :
  flights_climbed 18 10 60 = 9 := by
  sorry

end john_climbs_nine_flights_l398_39892


namespace harold_wrapping_cost_l398_39886

/-- Represents the number of shirt boxes that can be wrapped with one roll of paper -/
def shirt_boxes_per_roll : ℕ := 5

/-- Represents the number of XL boxes that can be wrapped with one roll of paper -/
def xl_boxes_per_roll : ℕ := 3

/-- Represents the number of shirt boxes Harold needs to wrap -/
def harold_shirt_boxes : ℕ := 20

/-- Represents the number of XL boxes Harold needs to wrap -/
def harold_xl_boxes : ℕ := 12

/-- Represents the cost of one roll of wrapping paper in cents -/
def cost_per_roll : ℕ := 400

/-- Theorem stating that Harold will spend $32.00 to wrap all boxes -/
theorem harold_wrapping_cost : 
  (((harold_shirt_boxes + shirt_boxes_per_roll - 1) / shirt_boxes_per_roll) + 
   ((harold_xl_boxes + xl_boxes_per_roll - 1) / xl_boxes_per_roll)) * 
  cost_per_roll = 3200 := by
  sorry

end harold_wrapping_cost_l398_39886


namespace jason_borrowed_amount_l398_39896

/-- Calculates the payment for a given hour based on the repeating pattern -/
def hourly_payment (hour : ℕ) : ℕ :=
  (hour - 1) % 6 + 1

/-- Calculates the total payment for a given number of hours -/
def total_payment (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_payment |>.sum

/-- The problem statement -/
theorem jason_borrowed_amount :
  total_payment 39 = 132 := by
  sorry

end jason_borrowed_amount_l398_39896


namespace consecutive_values_exist_l398_39878

/-- A polynomial that takes on three consecutive integer values at three consecutive integer points -/
def polynomial (a : ℤ) (x : ℤ) : ℤ := x^3 - 18*x^2 + a*x + 1784

theorem consecutive_values_exist :
  ∃ (k n : ℤ),
    polynomial a (k-1) = n-1 ∧
    polynomial a k = n ∧
    polynomial a (k+1) = n+1 :=
sorry

end consecutive_values_exist_l398_39878


namespace primes_arithmetic_sequence_ones_digit_l398_39879

/-- A function that returns the ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem primes_arithmetic_sequence_ones_digit 
  (p q r s : ℕ) 
  (hp : isPrime p) 
  (hq : isPrime q) 
  (hr : isPrime r) 
  (hs : isPrime s)
  (hseq : q = p + 8 ∧ r = q + 8 ∧ s = r + 8)
  (hp_gt_5 : p > 5) :
  onesDigit p = 3 := by
sorry

end primes_arithmetic_sequence_ones_digit_l398_39879


namespace peach_probability_l398_39817

/-- A fruit type in the basket -/
inductive Fruit
| apple
| pear
| peach

/-- The number of fruits in the basket -/
def basket : Fruit → ℕ
| Fruit.apple => 5
| Fruit.pear => 3
| Fruit.peach => 2

/-- The total number of fruits in the basket -/
def total_fruits : ℕ := basket Fruit.apple + basket Fruit.pear + basket Fruit.peach

/-- The probability of picking a specific fruit -/
def prob_pick (f : Fruit) : ℚ := basket f / total_fruits

theorem peach_probability :
  prob_pick Fruit.peach = 1 / 5 := by
  sorry


end peach_probability_l398_39817


namespace age_ratio_is_two_to_one_l398_39820

-- Define the present ages
def sons_present_age : ℕ := 24
def mans_present_age : ℕ := sons_present_age + 26

-- Define the ages in two years
def sons_future_age : ℕ := sons_present_age + 2
def mans_future_age : ℕ := mans_present_age + 2

-- Define the ratio
def age_ratio : ℚ := mans_future_age / sons_future_age

theorem age_ratio_is_two_to_one : age_ratio = 2 := by
  sorry

end age_ratio_is_two_to_one_l398_39820


namespace frac_two_thirds_is_quadratic_radical_l398_39866

def is_quadratic_radical (x : ℝ) : Prop := x ≥ 0

theorem frac_two_thirds_is_quadratic_radical :
  is_quadratic_radical (2/3) :=
by sorry

end frac_two_thirds_is_quadratic_radical_l398_39866


namespace division_problem_l398_39863

theorem division_problem (dividend quotient remainder : ℕ) 
  (h1 : dividend = 301)
  (h2 : quotient = 14)
  (h3 : remainder = 7)
  : ∃ divisor : ℕ, 
    dividend = divisor * quotient + remainder ∧ 
    divisor = 21 := by
sorry

end division_problem_l398_39863


namespace specific_plate_probability_l398_39807

/-- The set of vowels used in Mathlandia license plates -/
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

/-- The set of non-vowels used in Mathlandia license plates -/
def nonVowels : Finset Char := {'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z', '1'}

/-- The set of digits used in Mathlandia license plates -/
def digits : Finset Char := {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

/-- A license plate in Mathlandia -/
structure LicensePlate where
  first : Char
  second : Char
  third : Char
  fourth : Char
  fifth : Char

/-- The probability of a specific license plate occurring in Mathlandia -/
def licensePlateProbability (plate : LicensePlate) : ℚ :=
  1 / (vowels.card * vowels.card * nonVowels.card * (nonVowels.card - 1) * digits.card)

/-- The specific license plate "AIE19" -/
def specificPlate : LicensePlate := ⟨'A', 'I', 'E', '1', '9'⟩

theorem specific_plate_probability :
  licensePlateProbability specificPlate = 1 / 105000 :=
sorry

end specific_plate_probability_l398_39807


namespace dormitory_to_city_distance_l398_39870

theorem dormitory_to_city_distance :
  ∀ D : ℝ,
  (1/5 : ℝ) * D + (2/3 : ℝ) * D + 4 = D →
  D = 30 :=
by
  sorry

end dormitory_to_city_distance_l398_39870


namespace division_sum_theorem_l398_39808

theorem division_sum_theorem (quotient divisor remainder : ℕ) 
  (h_quotient : quotient = 65)
  (h_divisor : divisor = 24)
  (h_remainder : remainder = 5) :
  quotient * divisor + remainder = 1565 :=
by sorry

end division_sum_theorem_l398_39808


namespace waiter_problem_l398_39848

/-- The number of customers who left the waiter's section -/
def customers_left : ℕ := 14

/-- The number of people at each remaining table -/
def people_per_table : ℕ := 4

/-- The number of tables in the waiter's section -/
def number_of_tables : ℕ := 2

/-- The initial number of customers in the waiter's section -/
def initial_customers : ℕ := 22

theorem waiter_problem :
  initial_customers = customers_left + (number_of_tables * people_per_table) :=
sorry

end waiter_problem_l398_39848


namespace long_jump_ratio_l398_39858

/-- Given the conditions of a long jump event, prove the ratio of Margarita's jump to Ricciana's jump -/
theorem long_jump_ratio (ricciana_run : ℕ) (ricciana_jump : ℕ) (margarita_run : ℕ) (total_difference : ℕ) :
  ricciana_run = 20 →
  ricciana_jump = 4 →
  margarita_run = 18 →
  total_difference = 1 →
  (margarita_run + (ricciana_run + ricciana_jump + total_difference - margarita_run)) / ricciana_jump = 7 / 4 := by
  sorry

end long_jump_ratio_l398_39858


namespace smallest_b_value_l398_39859

theorem smallest_b_value (a c d : ℤ) (x₁ x₂ x₃ x₄ : ℝ) :
  (∀ x, x^4 + a*x^3 + (x₁*x₂ + x₁*x₃ + x₁*x₄ + x₂*x₃ + x₂*x₄ + x₃*x₄)*x^2 + c*x + d = 0 → x > 0) →
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 →
  d = x₁ * x₂ * x₃ * x₄ →
  x₁*x₂ + x₁*x₃ + x₁*x₄ + x₂*x₃ + x₂*x₄ + x₃*x₄ ≥ 6 :=
by
  sorry

end smallest_b_value_l398_39859


namespace max_sum_of_squares_l398_39843

theorem max_sum_of_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 → 
  n ∈ Finset.range 1982 → 
  (n^2 - m*n - m^2)^2 = 1 → 
  m^2 + n^2 ≤ 3524578 := by
  sorry

end max_sum_of_squares_l398_39843


namespace unique_right_triangle_l398_39806

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- Theorem stating that among the given sets, only {1, 1, √2} forms a right triangle --/
theorem unique_right_triangle :
  ¬ is_right_triangle 4 5 6 ∧
  is_right_triangle 1 1 (Real.sqrt 2) ∧
  ¬ is_right_triangle 6 8 11 ∧
  ¬ is_right_triangle 5 12 23 :=
by sorry

#check unique_right_triangle

end unique_right_triangle_l398_39806


namespace square_of_sum_23_2_l398_39894

theorem square_of_sum_23_2 : 23^2 + 2*(23*2) + 2^2 = 625 := by sorry

end square_of_sum_23_2_l398_39894


namespace y_intercept_of_line_l398_39818

/-- The y-intercept of the line 3x + 5y = 20 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) :
  3 * x + 5 * y = 20 → x = 0 → y = 4 := by
  sorry

end y_intercept_of_line_l398_39818


namespace appliance_pricing_l398_39873

/-- Represents the cost price of an electrical appliance in yuan -/
def cost_price : ℝ := sorry

/-- The markup percentage as a decimal -/
def markup : ℝ := 0.30

/-- The discount percentage as a decimal -/
def discount : ℝ := 0.20

/-- The final selling price in yuan -/
def selling_price : ℝ := 2080

theorem appliance_pricing :
  cost_price * (1 + markup) * (1 - discount) = selling_price := by sorry

end appliance_pricing_l398_39873


namespace jared_car_count_l398_39876

theorem jared_car_count : ∀ (j a f : ℕ),
  (j : ℝ) = 0.85 * a →
  a = f + 7 →
  j + a + f = 983 →
  j = 295 :=
by sorry

end jared_car_count_l398_39876


namespace smallest_solution_of_equation_l398_39825

theorem smallest_solution_of_equation :
  let x : ℝ := (5 - Real.sqrt 33) / 2
  (1 / (x - 1) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 1) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x :=
by sorry

end smallest_solution_of_equation_l398_39825


namespace sqrt_difference_equals_threes_l398_39887

/-- Given a natural number n, this function returns the number composed of 2n digits of 1 -/
def two_n_ones (n : ℕ) : ℕ := (10^(2*n) - 1) / 9

/-- Given a natural number n, this function returns the number composed of n digits of 2 -/
def n_twos (n : ℕ) : ℕ := 2 * ((10^n - 1) / 9)

/-- Given a natural number n, this function returns the number composed of n digits of 3 -/
def n_threes (n : ℕ) : ℕ := (10^n - 1) / 3

theorem sqrt_difference_equals_threes (n : ℕ) : 
  Real.sqrt (two_n_ones n - n_twos n) = n_threes n := by
  sorry

end sqrt_difference_equals_threes_l398_39887


namespace ella_toast_combinations_l398_39829

/-- The number of different kinds of spreads -/
def num_spreads : ℕ := 12

/-- The number of different kinds of toppings -/
def num_toppings : ℕ := 8

/-- The number of types of bread -/
def num_breads : ℕ := 3

/-- The number of spreads chosen for each toast -/
def spreads_per_toast : ℕ := 1

/-- The number of toppings chosen for each toast -/
def toppings_per_toast : ℕ := 2

/-- The number of breads chosen for each toast -/
def breads_per_toast : ℕ := 1

/-- The total number of different toasts Ella can make -/
def total_toasts : ℕ := num_spreads * (num_toppings.choose toppings_per_toast) * num_breads

theorem ella_toast_combinations :
  total_toasts = 1008 := by sorry

end ella_toast_combinations_l398_39829


namespace cubic_roots_sum_cubes_l398_39875

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (5 * a^3 + 2003 * a + 3005 = 0) →
  (5 * b^3 + 2003 * b + 3005 = 0) →
  (5 * c^3 + 2003 * c + 3005 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 := by
  sorry

end cubic_roots_sum_cubes_l398_39875


namespace equation_solution_l398_39850

theorem equation_solution :
  ∃ x : ℝ, (3 * x + 9 = 0) ∧ (x = -3) :=
by sorry

end equation_solution_l398_39850


namespace expression_value_l398_39883

theorem expression_value : 
  2 * Real.tan (60 * π / 180) - (1/3)⁻¹ + (-2)^2 * (2017 - Real.sin (45 * π / 180))^0 - |-(12: ℝ).sqrt| = 1 := by
  sorry

end expression_value_l398_39883


namespace candy_bar_earnings_difference_l398_39849

/-- The problem of calculating the difference in earnings between Tina and Marvin from selling candy bars. -/
theorem candy_bar_earnings_difference : 
  let candy_bar_price : ℕ := 2
  let marvin_sales : ℕ := 35
  let tina_sales : ℕ := 3 * marvin_sales
  let marvin_earnings : ℕ := candy_bar_price * marvin_sales
  let tina_earnings : ℕ := candy_bar_price * tina_sales
  tina_earnings - marvin_earnings = 140 :=
by sorry

end candy_bar_earnings_difference_l398_39849


namespace regular_polygon_exterior_angle_l398_39833

theorem regular_polygon_exterior_angle (n : ℕ) :
  (360 / n : ℝ) = 72 → n = 5 := by
  sorry

end regular_polygon_exterior_angle_l398_39833


namespace exists_question_with_different_answers_l398_39831

/-- Represents a person who always tells the truth -/
structure TruthfulPerson where
  answer : Prop → Bool
  always_truthful : ∀ p, answer p = p

/-- Represents a question that can be asked -/
structure Question where
  ask : TruthfulPerson → Bool

/-- Represents the state of a day, including whether any questions have been asked -/
structure DayState where
  question_asked : Bool

/-- The theorem stating that there exists a question that yields different answers when asked twice -/
theorem exists_question_with_different_answers :
  ∃ (q : Question), ∀ (p : TruthfulPerson),
    ∃ (d1 d2 : DayState),
      d1.question_asked = false ∧
      d2.question_asked = true ∧
      q.ask p ≠ q.ask p :=
sorry

end exists_question_with_different_answers_l398_39831


namespace sand_pit_fill_theorem_l398_39845

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def prismVolume (d : PrismDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents a sand pit with its dimensions and current fill level -/
structure SandPit where
  dimensions : PrismDimensions
  fillLevel : ℝ  -- Represents the fraction of the pit that is filled (0 to 1)

/-- Calculates the additional sand volume needed to fill the pit completely -/
def additionalSandNeeded (pit : SandPit) : ℝ :=
  (1 - pit.fillLevel) * prismVolume pit.dimensions

theorem sand_pit_fill_theorem (pit : SandPit) 
    (h1 : pit.dimensions.length = 10)
    (h2 : pit.dimensions.width = 2)
    (h3 : pit.dimensions.height = 0.5)
    (h4 : pit.fillLevel = 0.5) :
    additionalSandNeeded pit = 5 := by
  sorry

#eval additionalSandNeeded {
  dimensions := { length := 10, width := 2, height := 0.5 },
  fillLevel := 0.5
}

end sand_pit_fill_theorem_l398_39845


namespace robert_kicks_before_break_l398_39828

/-- The number of kicks Robert took before the break -/
def kicks_before_break (total : ℕ) (after_break : ℕ) (remaining : ℕ) : ℕ :=
  total - (after_break + remaining)

/-- Theorem stating that Robert took 43 kicks before the break -/
theorem robert_kicks_before_break :
  kicks_before_break 98 36 19 = 43 := by
  sorry

end robert_kicks_before_break_l398_39828


namespace friction_negative_work_on_slope_l398_39869

/-- A slope-block system where a block slides down a slope -/
structure SlopeBlockSystem where
  M : ℝ  -- Mass of the slope
  m : ℝ  -- Mass of the block
  μ : ℝ  -- Coefficient of friction between block and slope
  θ : ℝ  -- Angle of the slope
  g : ℝ  -- Acceleration due to gravity

/-- The horizontal surface is smooth -/
def is_smooth_surface (system : SlopeBlockSystem) : Prop :=
  sorry

/-- The block is released from rest at the top of the slope -/
def block_released_from_rest (system : SlopeBlockSystem) : Prop :=
  sorry

/-- The friction force does negative work on the slope -/
def friction_does_negative_work (system : SlopeBlockSystem) : Prop :=
  sorry

/-- Main theorem: The friction force of the block on the slope does negative work on the slope -/
theorem friction_negative_work_on_slope (system : SlopeBlockSystem) 
  (h1 : system.M > 0) 
  (h2 : system.m > 0) 
  (h3 : system.μ > 0) 
  (h4 : system.θ > 0) 
  (h5 : system.g > 0) 
  (h6 : is_smooth_surface system) 
  (h7 : block_released_from_rest system) : 
  friction_does_negative_work system :=
sorry

end friction_negative_work_on_slope_l398_39869


namespace polynomial_division_l398_39819

theorem polynomial_division (x : ℝ) :
  8 * x^4 - 4 * x^3 + 5 * x^2 - 9 * x + 3 = (x - 1) * (8 * x^3 - 4 * x^2 + 9 * x - 18) + 3 := by
  sorry

end polynomial_division_l398_39819


namespace subset_condition_l398_39816

/-- The set A defined by the given condition -/
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (a + 1)) < 0}

/-- The set B defined by the given condition -/
def B (a : ℝ) : Set ℝ := {x | (x - 2*a) / (x - (a^2 + 1)) < 0}

/-- The theorem stating the relationship between a and the subset property -/
theorem subset_condition (a : ℝ) : 
  B a ⊆ A a ↔ a ∈ Set.Icc (-1/2) (-1/2) ∪ Set.Icc 2 3 := by
  sorry

end subset_condition_l398_39816


namespace min_value_sum_squares_over_sum_l398_39881

theorem min_value_sum_squares_over_sum (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → a + b + c = 9 →
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) ≥ 9 := by
sorry

end min_value_sum_squares_over_sum_l398_39881


namespace village_x_current_population_l398_39865

/-- The current population of Village X -/
def village_x_population : ℕ := sorry

/-- The yearly decrease in Village X's population -/
def village_x_decrease_rate : ℕ := 1200

/-- The current population of Village Y -/
def village_y_population : ℕ := 42000

/-- The yearly increase in Village Y's population -/
def village_y_increase_rate : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years_until_equal : ℕ := 16

theorem village_x_current_population :
  village_x_population = 
    village_y_population + 
    village_y_increase_rate * years_until_equal + 
    village_x_decrease_rate * years_until_equal := by
  sorry

end village_x_current_population_l398_39865


namespace isosceles_right_triangle_property_l398_39813

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + a

-- Define the variables
variables {a t x₁ x₂ : ℝ}

-- State the theorem
theorem isosceles_right_triangle_property
  (h1 : f a x₁ = 0)
  (h2 : f a x₂ = 0)
  (h3 : x₁ < x₂)
  (h4 : ∃ (c : ℝ), f a c = (x₂ - x₁) / 2 ∧ c = (x₁ + x₂) / 2)
  (h5 : t = Real.sqrt ((x₂ - 1) / (x₁ - 1)))
  : a * t - (a + t) = 1 := by
  sorry

end

end isosceles_right_triangle_property_l398_39813


namespace constant_d_value_l398_39864

theorem constant_d_value (x y d : ℚ) 
  (h1 : (7 * x + 4 * y) / (x - 2 * y) = 13)
  (h2 : x / (2 * y) = d / 2) : 
  d = 5 := by sorry

end constant_d_value_l398_39864


namespace austin_robot_purchase_l398_39889

/-- Proves that Austin bought robots for 7 friends given the problem conditions --/
theorem austin_robot_purchase (robot_cost : ℚ) (tax : ℚ) (change : ℚ) (initial_amount : ℚ) 
  (h1 : robot_cost = 8.75)
  (h2 : tax = 7.22)
  (h3 : change = 11.53)
  (h4 : initial_amount = 80) :
  (initial_amount - (change + tax)) / robot_cost = 7 := by
  sorry

#eval (80 : ℚ) - (11.53 + 7.22)
#eval ((80 : ℚ) - (11.53 + 7.22)) / 8.75

end austin_robot_purchase_l398_39889


namespace no_specific_m_value_l398_39893

theorem no_specific_m_value (m : ℝ) (z₁ z₂ : ℂ) 
  (h₁ : z₁ = m + 2*I) 
  (h₂ : z₂ = 3 - 4*I) : 
  ∀ (n : ℝ), ∃ (m' : ℝ), m' ≠ n ∧ z₁ = m' + 2*I :=
sorry

end no_specific_m_value_l398_39893


namespace debate_club_committee_selection_l398_39872

theorem debate_club_committee_selection (n : ℕ) : 
  (n.choose 3 = 21) → (n.choose 4 = 126) := by
  sorry

end debate_club_committee_selection_l398_39872


namespace cone_radius_theorem_l398_39852

/-- For a cone with radius r, slant height 2r, and lateral surface area equal to half its volume, prove that r = 4√3 -/
theorem cone_radius_theorem (r : ℝ) (h : ℝ) : 
  r > 0 → 
  h > 0 → 
  (2 * r)^2 = r^2 + h^2 →  -- Pythagorean theorem for the slant height
  π * r * (2 * r) = (1/2) * ((1/3) * π * r^2 * h) →  -- Lateral surface area = 1/2 * Volume
  r = 4 * Real.sqrt 3 := by
sorry

end cone_radius_theorem_l398_39852


namespace stewart_farm_horse_food_l398_39874

/-- Given a farm with sheep and horses, calculate the daily horse food requirement per horse -/
theorem stewart_farm_horse_food (sheep_count : ℕ) (total_horse_food : ℕ) 
  (h_sheep_count : sheep_count = 32) 
  (h_total_horse_food : total_horse_food = 12880) 
  (h_ratio : sheep_count * 7 = 32 * 4) : 
  total_horse_food / (sheep_count * 7 / 4) = 230 := by
  sorry

end stewart_farm_horse_food_l398_39874


namespace cleos_marbles_eq_15_l398_39815

/-- The number of marbles Cleo has on the third day -/
def cleos_marbles : ℕ :=
  let initial_marbles : ℕ := 30
  let marbles_taken_day2 : ℕ := (3 * initial_marbles) / 5
  let marbles_each_day2 : ℕ := marbles_taken_day2 / 2
  let marbles_remaining_day2 : ℕ := initial_marbles - marbles_taken_day2
  let marbles_taken_day3 : ℕ := marbles_remaining_day2 / 2
  marbles_each_day2 + marbles_taken_day3

theorem cleos_marbles_eq_15 : cleos_marbles = 15 := by
  sorry

end cleos_marbles_eq_15_l398_39815


namespace black_cards_count_l398_39805

theorem black_cards_count (total_cards : Nat) (red_cards : Nat) (clubs : Nat)
  (h_total : total_cards = 13)
  (h_red : red_cards = 6)
  (h_clubs : clubs = 6)
  (h_suits : ∃ (spades diamonds hearts : Nat), 
    spades + diamonds + hearts + clubs = total_cards ∧
    diamonds = 2 * spades ∧
    hearts = 2 * diamonds) :
  clubs + (total_cards - red_cards - clubs) = 7 := by
  sorry

end black_cards_count_l398_39805


namespace multiplication_mistake_l398_39835

theorem multiplication_mistake (x : ℕ) (h : 53 * x - 35 * x = 540) : 53 * x = 1590 := by
  sorry

end multiplication_mistake_l398_39835


namespace total_money_l398_39821

theorem total_money (r p q : ℕ) (h1 : r = 1600) (h2 : r = (2 * (p + q)) / 3) : 
  p + q + r = 4000 := by
sorry

end total_money_l398_39821


namespace book_weight_is_205_l398_39885

/-- Calculates the weight of a single book given the following conditions:
  * 6 books in each small box
  * Small box weighs 220 grams
  * 9 small boxes in a large box
  * Large box weighs 250 grams
  * Total weight is 13.3 kilograms
  * All books weigh the same
-/
def bookWeight (booksPerSmallBox : ℕ) (smallBoxWeight : ℕ) (smallBoxCount : ℕ) 
                (largeBoxWeight : ℕ) (totalWeightKg : ℚ) : ℚ :=
  let totalWeightG : ℚ := totalWeightKg * 1000
  let smallBoxesWeight : ℚ := smallBoxWeight * smallBoxCount
  let booksWeight : ℚ := totalWeightG - largeBoxWeight - smallBoxesWeight
  let totalBooks : ℕ := booksPerSmallBox * smallBoxCount
  booksWeight / totalBooks

theorem book_weight_is_205 :
  bookWeight 6 220 9 250 (13.3 : ℚ) = 205 := by
  sorry

#eval bookWeight 6 220 9 250 (13.3 : ℚ)

end book_weight_is_205_l398_39885

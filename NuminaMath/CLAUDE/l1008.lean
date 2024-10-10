import Mathlib

namespace boat_distance_theorem_l1008_100865

/-- Calculates the distance traveled downstream by a boat -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat traveling downstream for 5 hours covers 125 km -/
theorem boat_distance_theorem (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : stream_speed = 5)
  (h3 : time = 5) :
  distance_downstream boat_speed stream_speed time = 125 := by
  sorry

#check boat_distance_theorem

end boat_distance_theorem_l1008_100865


namespace train_passenger_count_l1008_100875

/-- Calculates the total number of passengers transported by a train -/
theorem train_passenger_count (one_way : ℕ) (return_way : ℕ) (additional_trips : ℕ) : 
  one_way = 100 → return_way = 60 → additional_trips = 3 → 
  (one_way + return_way) * (additional_trips + 1) = 640 := by
  sorry

end train_passenger_count_l1008_100875


namespace batsman_average_l1008_100859

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  runsLastInning : ℕ
  averageIncrease : ℕ

/-- Calculates the new average of a batsman after their last inning -/
def newAverage (b : Batsman) : ℕ :=
  b.averageIncrease + (b.innings - 1) * b.averageIncrease + b.runsLastInning

/-- Theorem stating that given the conditions, the batsman's new average is 140 -/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 17) 
  (h2 : b.runsLastInning = 300) 
  (h3 : b.averageIncrease = 10) : 
  newAverage b = 140 := by
  sorry


end batsman_average_l1008_100859


namespace diophantine_equation_solutions_l1008_100816

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ+,
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 2 + (1 : ℚ) / z →
  (x = 1 ∧ y = 2 ∧ z = 1) ∨
  (x = 2 ∧ ((y = 1 ∧ z = 1) ∨ (y = z ∧ y ≥ 2))) ∨
  (x = 3 ∧ ((y = 3 ∧ z = 6) ∨ (y = 4 ∧ z = 12) ∨ (y = 5 ∧ z = 30) ∨ (y = 2 ∧ z = 3))) ∨
  (x ≥ 4 ∧ y ≥ 4 → False) :=
sorry

end diophantine_equation_solutions_l1008_100816


namespace green_blue_difference_l1008_100829

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : ℕ
  ratio : Fin 3 → ℕ
  color_sum : ratio 0 + ratio 1 + ratio 2 = 18

theorem green_blue_difference (bag : DiskBag) 
  (h1 : bag.total = 108)
  (h2 : bag.ratio 0 = 3)  -- Blue
  (h3 : bag.ratio 1 = 7)  -- Yellow
  (h4 : bag.ratio 2 = 8)  -- Green
  : (bag.total / 18 * bag.ratio 2) - (bag.total / 18 * bag.ratio 0) = 30 := by
  sorry

end green_blue_difference_l1008_100829


namespace complex_sum_theorem_l1008_100856

theorem complex_sum_theorem : 
  let i : ℂ := Complex.I
  let z₁ : ℂ := 2 + i
  let z₂ : ℂ := 1 - 2*i
  z₁ + z₂ = 3 - i := by sorry

end complex_sum_theorem_l1008_100856


namespace greatest_multiple_of_five_cubed_less_than_3375_l1008_100828

theorem greatest_multiple_of_five_cubed_less_than_3375 :
  ∃ (x : ℕ), x > 0 ∧ 5 ∣ x ∧ x^3 < 3375 ∧ ∀ (y : ℕ), y > 0 → 5 ∣ y → y^3 < 3375 → y ≤ x :=
by
  sorry

end greatest_multiple_of_five_cubed_less_than_3375_l1008_100828


namespace repeating_decimal_027_product_l1008_100801

/-- Represents a repeating decimal with a 3-digit repeating sequence -/
def RepeatingDecimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem repeating_decimal_027_product : 
  let x := RepeatingDecimal 0 2 7
  let (n, d) := (x.num, x.den)
  (n.gcd d = 1) → n * d = 37 := by
  sorry

end repeating_decimal_027_product_l1008_100801


namespace intersection_of_A_and_B_l1008_100897

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {2, 4, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by
  sorry

end intersection_of_A_and_B_l1008_100897


namespace emily_score_calculation_l1008_100840

/-- Emily's trivia game score calculation -/
theorem emily_score_calculation 
  (first_round : ℕ) 
  (second_round : ℕ) 
  (final_score : ℕ) 
  (h1 : first_round = 16) 
  (h2 : second_round = 33) 
  (h3 : final_score = 1) : 
  (first_round + second_round) - final_score = 48 := by
  sorry

end emily_score_calculation_l1008_100840


namespace product_trailing_zeros_l1008_100874

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 15, 360, and 125 -/
def product : ℕ := 15 * 360 * 125

theorem product_trailing_zeros : trailingZeros product = 3 := by sorry

end product_trailing_zeros_l1008_100874


namespace base_conversion_equality_l1008_100896

/-- Given that the base 6 number 62₆ is equal to the base b number 124ᵦ,
    prove that the unique positive integer solution for b is 4. -/
theorem base_conversion_equality : ∃! (b : ℕ), b > 0 ∧ (6 * 6 + 2) = (1 * b^2 + 2 * b + 4) := by
  sorry

end base_conversion_equality_l1008_100896


namespace max_min_x2_minus_y2_l1008_100814

theorem max_min_x2_minus_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 - x*y + y^2 = 1) :
  ∃ (min max : ℝ), (∀ z, z = x^2 - y^2 → min ≤ z ∧ z ≤ max) ∧
                   min = -2*Real.sqrt 3/3 ∧
                   max = 2*Real.sqrt 3/3 := by
  sorry

end max_min_x2_minus_y2_l1008_100814


namespace journey_duration_l1008_100879

-- Define the distance covered by the train
def distance : ℝ := 80

-- Define the average speed of the train
def average_speed : ℝ := 10

-- Theorem: The duration of the journey is 8 seconds
theorem journey_duration : (distance / average_speed) = 8 := by
  sorry

end journey_duration_l1008_100879


namespace xunzi_wangzhi_interpretation_l1008_100837

/-- Represents the four seasonal agricultural activities -/
inductive SeasonalActivity
| SpringPlowing
| SummerWeeding
| AutumnHarvesting
| WinterStoring

/-- Represents the result of following the seasonal activities -/
def SurplusFood : Prop := True

/-- Represents the concept of objective laws in nature -/
def ObjectiveLaw : Prop := True

/-- Represents the concept of subjective initiative -/
def SubjectiveInitiative : Prop := True

/-- Represents the concept of expected results -/
def ExpectedResults : Prop := True

/-- The main theorem based on the given problem -/
theorem xunzi_wangzhi_interpretation 
  (seasonal_activities : List SeasonalActivity)
  (follow_activities_lead_to_surplus : seasonal_activities.length = 4 → SurplusFood) :
  (ObjectiveLaw → ExpectedResults) ∧ 
  (SubjectiveInitiative → ObjectiveLaw) := by
  sorry


end xunzi_wangzhi_interpretation_l1008_100837


namespace cone_base_radius_l1008_100880

/-- Given a cone with slant height 5 cm and lateral area 15π cm², 
    the radius of its base circle is 3 cm. -/
theorem cone_base_radius (s : ℝ) (A : ℝ) (r : ℝ) : 
  s = 5 →  -- slant height is 5 cm
  A = 15 * Real.pi →  -- lateral area is 15π cm²
  A = Real.pi * r * s →  -- formula for lateral area of a cone
  r = 3 :=  -- radius of base circle is 3 cm
by
  sorry

end cone_base_radius_l1008_100880


namespace infinitely_many_divisible_by_2018_l1008_100811

def a (n : ℕ) : ℕ := 2 * 10^(n+2) + 18

theorem infinitely_many_divisible_by_2018 :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, 2018 ∣ a n :=
sorry

end infinitely_many_divisible_by_2018_l1008_100811


namespace smallest_value_theorem_l1008_100800

theorem smallest_value_theorem (v w : ℝ) 
  (h : ∀ (a b : ℝ), (2^(a+b) + 8)*(3^a + 3^b) ≤ v*(12^(a-1) + 12^(b-1) - 2^(a+b-1)) + w) : 
  (∀ (v' w' : ℝ), (∀ (a b : ℝ), (2^(a+b) + 8)*(3^a + 3^b) ≤ v'*(12^(a-1) + 12^(b-1) - 2^(a+b-1)) + w') → 
    128*v^2 + w^2 ≤ 128*v'^2 + w'^2) ∧ 128*v^2 + w^2 = 62208 := by
  sorry

end smallest_value_theorem_l1008_100800


namespace test_questions_l1008_100835

theorem test_questions (points_correct : ℕ) (points_incorrect : ℕ) 
  (total_score : ℕ) (correct_answers : ℕ) :
  points_correct = 20 →
  points_incorrect = 5 →
  total_score = 325 →
  correct_answers = 19 →
  ∃ (total_questions : ℕ),
    total_questions = correct_answers + (total_questions - correct_answers) ∧
    total_score = points_correct * correct_answers - 
      points_incorrect * (total_questions - correct_answers) ∧
    total_questions = 30 := by
  sorry

end test_questions_l1008_100835


namespace meal_combinations_eq_100_l1008_100860

/-- The number of items on the menu -/
def menu_items : ℕ := 10

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- The number of different combinations of meals that can be ordered -/
def meal_combinations : ℕ := menu_items ^ num_people

/-- Theorem stating that the number of meal combinations is 100 -/
theorem meal_combinations_eq_100 : meal_combinations = 100 := by
  sorry

end meal_combinations_eq_100_l1008_100860


namespace rectangle_to_cylinders_volume_ratio_l1008_100881

/-- Given a rectangle with dimensions 6 × 10, prove that when rolled into two cylinders,
    the ratio of the larger cylinder volume to the smaller cylinder volume is 5/3. -/
theorem rectangle_to_cylinders_volume_ratio :
  let rectangle_width : ℝ := 6
  let rectangle_height : ℝ := 10
  let cylinder1_height : ℝ := rectangle_height
  let cylinder1_circumference : ℝ := rectangle_width
  let cylinder2_height : ℝ := rectangle_width
  let cylinder2_circumference : ℝ := rectangle_height
  let cylinder1_volume := π * (cylinder1_circumference / (2 * π))^2 * cylinder1_height
  let cylinder2_volume := π * (cylinder2_circumference / (2 * π))^2 * cylinder2_height
  let larger_volume := max cylinder1_volume cylinder2_volume
  let smaller_volume := min cylinder1_volume cylinder2_volume
  larger_volume / smaller_volume = 5 / 3 := by
sorry


end rectangle_to_cylinders_volume_ratio_l1008_100881


namespace cubic_root_sum_l1008_100823

theorem cubic_root_sum (k₁ k₂ : ℝ) (h : k₁ + k₂ ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ x^3 - k₁*x - k₂
  let roots := { x : ℝ | f x = 0 }
  ∃ (a b c : ℝ), a ∈ roots ∧ b ∈ roots ∧ c ∈ roots ∧
    (1+a)/(1-a) + (1+b)/(1-b) + (1+c)/(1-c) = (3 + k₁ + 3*k₂) / (1 - k₁ - k₂) :=
by sorry

end cubic_root_sum_l1008_100823


namespace linear_function_intersection_l1008_100871

/-- A linear function passing through (-1, -2) intersects the y-axis at (0, -1) -/
theorem linear_function_intersection (k : ℝ) : 
  (∀ x y, y = k * (x - 1) → (x = -1 ∧ y = -2) → (0 = k * (-1 - 1) + 2)) → 
  (∃ y, y = k * (0 - 1) ∧ y = -1) :=
by sorry

end linear_function_intersection_l1008_100871


namespace worker_savings_l1008_100830

theorem worker_savings (P : ℝ) (h : P > 0) :
  let monthly_savings := (1/3) * P
  let monthly_not_saved := (2/3) * P
  let yearly_savings := 12 * monthly_savings
  yearly_savings = 6 * monthly_not_saved :=
by sorry

end worker_savings_l1008_100830


namespace range_of_sum_and_abs_l1008_100867

theorem range_of_sum_and_abs (a b : ℝ) (ha : 1 ≤ a ∧ a ≤ 3) (hb : -4 < b ∧ b < 2) :
  1 < a + |b| ∧ a + |b| < 7 := by
  sorry

end range_of_sum_and_abs_l1008_100867


namespace part_to_whole_ratio_l1008_100807

theorem part_to_whole_ratio (N P : ℚ) (h1 : N = 160) (h2 : (1/5) * N + 4 = P - 4) :
  (P - 4) / N = 9 / 40 := by
  sorry

end part_to_whole_ratio_l1008_100807


namespace probability_two_ones_l1008_100889

def num_dice : ℕ := 15
def num_sides : ℕ := 6
def target_num : ℕ := 1
def target_count : ℕ := 2

theorem probability_two_ones (n : ℕ) (k : ℕ) (p : ℚ) :
  n = num_dice →
  k = target_count →
  p = 1 / num_sides →
  (n.choose k * p^k * (1 - p)^(n - k) : ℚ) = (105 * 5^13 : ℚ) / 6^14 :=
by sorry

end probability_two_ones_l1008_100889


namespace bakery_problem_solution_l1008_100848

/-- Represents the problem of buying sandwiches and cakes --/
def BakeryProblem (total_money sandwich_cost cake_cost max_items : ℚ) : Prop :=
  ∃ (sandwiches cakes : ℕ),
    sandwiches * sandwich_cost + cakes * cake_cost ≤ total_money ∧
    sandwiches + cakes ≤ max_items ∧
    ∀ (s c : ℕ),
      s * sandwich_cost + c * cake_cost ≤ total_money →
      s + c ≤ max_items →
      s + c ≤ sandwiches + cakes

/-- The maximum number of items that can be bought is 12 --/
theorem bakery_problem_solution :
  BakeryProblem 50 5 (5/2) 12 →
  ∃ (sandwiches cakes : ℕ), sandwiches + cakes = 12 :=
by sorry

end bakery_problem_solution_l1008_100848


namespace sam_final_penny_count_l1008_100861

def initial_pennies : ℕ := 980
def found_pennies : ℕ := 930
def exchanged_pennies : ℕ := 725
def gift_pennies : ℕ := 250

theorem sam_final_penny_count :
  initial_pennies + found_pennies - exchanged_pennies + gift_pennies = 1435 :=
by sorry

end sam_final_penny_count_l1008_100861


namespace three_digit_square_sum_numbers_l1008_100827

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a ≠ 0 ∧
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    n = 100 * a + 10 * b + c ∧
    n = 11 * (a^2 + b^2 + c^2)

theorem three_digit_square_sum_numbers :
  {n : ℕ | is_valid_number n} = {550, 803} :=
by sorry

end three_digit_square_sum_numbers_l1008_100827


namespace sam_distance_l1008_100887

/-- Proves that Sam drove 200 miles given the conditions of the problem -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) : 
  (marguerite_distance / marguerite_time) * sam_time = 200 := by
  sorry

end sam_distance_l1008_100887


namespace basketball_free_throw_probability_l1008_100869

theorem basketball_free_throw_probability :
  ∀ p : ℝ,
  0 ≤ p ∧ p ≤ 1 →
  (1 - p^2 = 16/25) →
  p = 3/5 :=
by sorry

end basketball_free_throw_probability_l1008_100869


namespace consecutive_odd_integers_problem_l1008_100877

theorem consecutive_odd_integers_problem (x : ℤ) (k : ℕ) : 
  (x % 2 = 1) →  -- x is odd
  ((x + 2) % 2 = 1) →  -- x + 2 is odd
  ((x + 4) % 2 = 1) →  -- x + 4 is odd
  (x + (x + 4) = k * (x + 2) - 131) →  -- sum of 1st and 3rd is 131 less than k times 2nd
  (x + (x + 2) + (x + 4) = 133) →  -- sum of all three is 133
  (k = 2) := by sorry

end consecutive_odd_integers_problem_l1008_100877


namespace factorization_difference_of_squares_l1008_100855

theorem factorization_difference_of_squares (y : ℝ) : 1 - 4 * y^2 = (1 - 2*y) * (1 + 2*y) := by
  sorry

end factorization_difference_of_squares_l1008_100855


namespace quadratic_rational_solutions_l1008_100857

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

/-- The discriminant of the quadratic equation kx^2 - 24x + 4k = 0 -/
def discriminant (k : ℤ) : ℤ := 576 - 16 * k * k

/-- The property that k is a valid solution -/
def is_valid_k (k : ℤ) : Prop :=
  k > 0 ∧ is_perfect_square (discriminant k)

theorem quadratic_rational_solutions :
  ∀ k : ℤ, is_valid_k k ↔ k = 3 ∨ k = 6 := by sorry

end quadratic_rational_solutions_l1008_100857


namespace complement_union_eq_specific_set_l1008_100844

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_union_eq_specific_set :
  (U \ (M ∪ N)) = {1, 6} := by sorry

end complement_union_eq_specific_set_l1008_100844


namespace root_sum_product_l1008_100899

/-- Given two polynomials with specified roots, prove that u = 32 -/
theorem root_sum_product (α β γ : ℂ) (q s u : ℂ) : 
  (α^3 + 4*α^2 + 6*α - 8 = 0) → 
  (β^3 + 4*β^2 + 6*β - 8 = 0) → 
  (γ^3 + 4*γ^2 + 6*γ - 8 = 0) →
  ((α+β)^3 + q*(α+β)^2 + s*(α+β) + u = 0) →
  ((β+γ)^3 + q*(β+γ)^2 + s*(β+γ) + u = 0) →
  ((γ+α)^3 + q*(γ+α)^2 + s*(γ+α) + u = 0) →
  u = 32 := by
sorry

end root_sum_product_l1008_100899


namespace equal_play_time_l1008_100854

theorem equal_play_time (team_size : ℕ) (field_players : ℕ) (match_duration : ℕ) 
  (h1 : team_size = 10)
  (h2 : field_players = 8)
  (h3 : match_duration = 45)
  (h4 : field_players < team_size) :
  (field_players * match_duration) / team_size = 36 := by
  sorry

end equal_play_time_l1008_100854


namespace fraction_decomposition_l1008_100886

theorem fraction_decomposition (x y A B : ℝ) (h : x * y ≠ 0) (h' : x + y ≠ 5) :
  (7 * x - 20 * y + 3) / (3 * x^2 * y + 2 * x * y^2 - 15 * x * y) =
  A / (x * y + 5) + B / (3 * x * y - 3) →
  A = -6.333 ∧ B = -3 := by
sorry

end fraction_decomposition_l1008_100886


namespace two_digit_number_property_l1008_100873

theorem two_digit_number_property (a b : ℕ) : 
  b = 2 * a →
  10 * a + b - (10 * b + a) = 36 →
  (a + b) - (b - a) = 8 := by
sorry

end two_digit_number_property_l1008_100873


namespace emails_evening_l1008_100843

def emails_problem (afternoon evening morning total : ℕ) : Prop :=
  afternoon = 3 ∧ morning = 6 ∧ total = 10 ∧ afternoon + evening + morning = total

theorem emails_evening : ∃ evening : ℕ, emails_problem 3 evening 6 10 ∧ evening = 1 :=
  sorry

end emails_evening_l1008_100843


namespace parallelogram_acute_angle_iff_diagonal_equation_l1008_100834

/-- A parallelogram with side lengths a and b, and diagonal lengths m and n -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  m : ℝ
  n : ℝ
  ha : a > 0
  hb : b > 0
  hm : m > 0
  hn : n > 0

/-- The acute angle of a parallelogram -/
def acute_angle (p : Parallelogram) : ℝ := sorry

theorem parallelogram_acute_angle_iff_diagonal_equation (p : Parallelogram) :
  p.a^4 + p.b^4 = p.m^2 * p.n^2 ↔ acute_angle p = π/4 := by sorry

end parallelogram_acute_angle_iff_diagonal_equation_l1008_100834


namespace min_sum_a_b_l1008_100898

theorem min_sum_a_b (a b : ℕ+) (l : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + l = 0 ∧ a * x₂^2 + b * x₂ + l = 0) →
  (∀ x : ℝ, a * x^2 + b * x + l = 0 → abs x < 1) →
  (∀ c d : ℕ+, c + d < a + b → ¬(∃ y : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    c * x₁^2 + d * x₁ + y = 0 ∧ c * x₂^2 + d * x₂ + y = 0) ∧
    (∀ x : ℝ, c * x^2 + d * x + y = 0 → abs x < 1))) →
  a + b = 10 :=
by sorry

end min_sum_a_b_l1008_100898


namespace scallop_cost_per_pound_l1008_100820

/-- The cost per pound of jumbo scallops -/
def cost_per_pound (scallops_per_pound : ℕ) (num_people : ℕ) (scallops_per_person : ℕ) (total_cost : ℚ) : ℚ :=
  let total_scallops := num_people * scallops_per_person
  let pounds_needed := total_scallops / scallops_per_pound
  total_cost / pounds_needed

/-- Theorem stating the cost per pound of jumbo scallops is $24 -/
theorem scallop_cost_per_pound :
  cost_per_pound 8 8 2 48 = 24 := by
  sorry

end scallop_cost_per_pound_l1008_100820


namespace rectangular_prism_parallel_edges_l1008_100842

/-- A rectangular prism with distinctly different dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_ne_width : length ≠ width
  length_ne_height : length ≠ height
  width_ne_height : width ≠ height

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ := 12

/-- Theorem: A rectangular prism with distinctly different dimensions has 12 pairs of parallel edges -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) :
  parallel_edge_pairs prism = 12 := by sorry

end rectangular_prism_parallel_edges_l1008_100842


namespace heartsuit_three_five_l1008_100864

-- Define the heartsuit operation
def heartsuit (x y : ℤ) : ℤ := 4*x + 6*y

-- Theorem statement
theorem heartsuit_three_five : heartsuit 3 5 = 42 := by
  sorry

end heartsuit_three_five_l1008_100864


namespace mod_sum_powers_l1008_100821

theorem mod_sum_powers (n : ℕ) : (36^1724 + 18^1724) % 7 = 3 := by
  sorry

end mod_sum_powers_l1008_100821


namespace baseball_league_games_l1008_100815

theorem baseball_league_games (N M : ℕ) : 
  N > M →
  M > 5 →
  4 * N + 5 * M = 90 →
  4 * N = 60 := by
sorry

end baseball_league_games_l1008_100815


namespace network_connections_l1008_100841

/-- The number of unique connections in a network of switches -/
def num_connections (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a network of 30 switches, where each switch is directly connected 
    to exactly 4 other switches, the total number of unique connections is 60 -/
theorem network_connections : num_connections 30 4 = 60 := by
  sorry

end network_connections_l1008_100841


namespace first_year_interest_l1008_100824

theorem first_year_interest (initial_deposit : ℝ) (first_year_balance : ℝ) 
  (second_year_increase : ℝ) (total_increase : ℝ) :
  initial_deposit = 500 →
  first_year_balance = 600 →
  second_year_increase = 0.1 →
  total_increase = 0.32 →
  first_year_balance - initial_deposit = 100 := by
  sorry

end first_year_interest_l1008_100824


namespace vitamin_d3_capsules_per_bottle_l1008_100866

/-- Calculates the number of capsules in each bottle given the total days, 
    daily serving size, and total number of bottles. -/
def capsules_per_bottle (total_days : ℕ) (daily_serving : ℕ) (total_bottles : ℕ) : ℕ :=
  (total_days * daily_serving) / total_bottles

/-- Theorem stating that given the specific conditions, the number of capsules
    per bottle is 60. -/
theorem vitamin_d3_capsules_per_bottle :
  capsules_per_bottle 180 2 6 = 60 := by
  sorry

#eval capsules_per_bottle 180 2 6

end vitamin_d3_capsules_per_bottle_l1008_100866


namespace geometric_sequence_ratio_l1008_100826

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = q * a n)

/-- The arithmetic sequence property for three terms -/
def ArithmeticSequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  ArithmeticSequence (3 * a 1) (2 * a 2) ((1/2) * a 3) →
  q = 3 := by sorry

end geometric_sequence_ratio_l1008_100826


namespace number_manipulation_l1008_100803

theorem number_manipulation (x : ℝ) : 
  (x - 34) / 10 = 2 → (x - 5) / 7 = 7 := by
  sorry

end number_manipulation_l1008_100803


namespace smallest_valid_number_l1008_100818

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧  -- four-digit number
  (let a := n / 1000
   let b := (n / 100) % 10
   let c := (n / 10) % 10
   let d := n % 10
   1000 * c + 100 * d + 10 * a + b - n = 5940) ∧  -- swapping condition
  n % 9 = 8 ∧  -- divisibility condition
  n % 2 = 1  -- odd number

theorem smallest_valid_number :
  is_valid_number 1979 ∧ ∀ m : ℕ, is_valid_number m → m ≥ 1979 :=
sorry

end smallest_valid_number_l1008_100818


namespace expansion_properties_l1008_100809

def sum_of_coefficients (n : ℕ) : ℝ := (3 + 1)^n

def sum_of_binomial_coefficients (n : ℕ) : ℝ := 2^n

theorem expansion_properties (n : ℕ) 
  (h : sum_of_coefficients n - sum_of_binomial_coefficients n = 240) :
  n = 4 ∧ 
  ∃ (a b c : ℝ), a = 81 ∧ b = 54 ∧ c = 1 ∧
  (∀ (x : ℝ), (3*x + x^(1/2))^n = a*x^4 + b*x^3 + c*x^2 + x^(7/2) + 6*x^(5/2) + 4*x^(3/2) + x^(1/2)) :=
by sorry

end expansion_properties_l1008_100809


namespace p_money_calculation_l1008_100882

theorem p_money_calculation (p q r : ℝ) 
  (h1 : p = (1/7 * p + 1/7 * p) + 35)
  (h2 : q = 1/7 * p) 
  (h3 : r = 1/7 * p) : 
  p = 49 := by
  sorry

end p_money_calculation_l1008_100882


namespace spring_festival_gala_arrangements_l1008_100813

theorem spring_festival_gala_arrangements :
  let original_programs : ℕ := 10
  let new_programs : ℕ := 3
  let available_spaces : ℕ := original_programs + 1 - 2  -- excluding first and last positions
  
  (available_spaces.choose new_programs) * (new_programs.factorial) = 990 :=
by
  sorry

end spring_festival_gala_arrangements_l1008_100813


namespace sum_of_reciprocals_l1008_100890

theorem sum_of_reciprocals (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_sum_product : a + b = 6 * (a * b)) : 1 / a + 1 / b = 6 := by
  sorry

end sum_of_reciprocals_l1008_100890


namespace rectangle_width_equals_six_l1008_100806

theorem rectangle_width_equals_six (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ) : 
  square_side = 12 →
  rect_length = 24 →
  square_side * square_side = rect_length * rect_width →
  rect_width = 6 := by
sorry

end rectangle_width_equals_six_l1008_100806


namespace unique_solution_rational_equation_l1008_100847

theorem unique_solution_rational_equation :
  ∃! x : ℝ, x ≠ -2 ∧ (x^2 + 2*x - 8)/(x + 2) = 3*x - 4 := by
  sorry

end unique_solution_rational_equation_l1008_100847


namespace bus_capacity_equality_l1008_100812

theorem bus_capacity_equality (x : ℕ) : 50 * x + 10 = 52 * x + 2 := by
  sorry

end bus_capacity_equality_l1008_100812


namespace point_distance_equality_l1008_100804

/-- Given points A(4, 2) and B(0, b) in the Cartesian coordinate system,
    if |BO| = |BA|, then b = 5. -/
theorem point_distance_equality (b : ℝ) : 
  let A : ℝ × ℝ := (4, 2)
  let B : ℝ × ℝ := (0, b)
  let O : ℝ × ℝ := (0, 0)
  (‖B - O‖ = ‖B - A‖) → b = 5 :=
by sorry

end point_distance_equality_l1008_100804


namespace oil_price_reduction_l1008_100858

/-- Calculates the additional amount of oil a housewife can obtain after a price reduction --/
theorem oil_price_reduction (original_price reduced_price budget : ℝ) : 
  original_price > 0 →
  reduced_price > 0 →
  budget > 0 →
  reduced_price = original_price * (1 - 0.35) →
  reduced_price = 56 →
  budget = 800 →
  let additional_amount := budget / reduced_price - budget / original_price
  ∃ ε > 0, |additional_amount - 5.01| < ε :=
by
  sorry

end oil_price_reduction_l1008_100858


namespace integer_between_sqrt_7_and_sqrt_15_l1008_100831

theorem integer_between_sqrt_7_and_sqrt_15 (a : ℤ) :
  (Real.sqrt 7 < a) ∧ (a < Real.sqrt 15) → a = 3 := by
  sorry

end integer_between_sqrt_7_and_sqrt_15_l1008_100831


namespace certain_event_l1008_100878

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents a bag of balls -/
def Bag := List Color

/-- Represents the result of drawing two balls -/
def Draw := (Color × Color)

/-- The bag containing 2 red balls and 1 white ball -/
def initialBag : Bag := [Color.Red, Color.Red, Color.White]

/-- Function to check if a draw contains at least one red ball -/
def hasRed (draw : Draw) : Prop :=
  draw.1 = Color.Red ∨ draw.2 = Color.Red

/-- All possible draws from the bag -/
def allDraws : List Draw := [
  (Color.Red, Color.Red),
  (Color.Red, Color.White),
  (Color.White, Color.Red)
]

/-- Theorem stating that any draw from the bag must contain at least one red ball -/
theorem certain_event : ∀ (draw : Draw), draw ∈ allDraws → hasRed draw := by sorry

end certain_event_l1008_100878


namespace crane_folding_theorem_l1008_100885

/-- The number of cranes Hyerin folds per day -/
def hyerin_cranes_per_day : ℕ := 16

/-- The number of days Hyerin folds cranes -/
def hyerin_days : ℕ := 7

/-- The number of cranes Taeyeong folds per day -/
def taeyeong_cranes_per_day : ℕ := 25

/-- The number of days Taeyeong folds cranes -/
def taeyeong_days : ℕ := 6

/-- The total number of cranes folded by Hyerin and Taeyeong -/
def total_cranes : ℕ := hyerin_cranes_per_day * hyerin_days + taeyeong_cranes_per_day * taeyeong_days

theorem crane_folding_theorem : total_cranes = 262 := by
  sorry

end crane_folding_theorem_l1008_100885


namespace fashion_line_blend_pieces_l1008_100894

theorem fashion_line_blend_pieces (silk_pieces : ℕ) (cashmere_pieces : ℕ) (total_pieces : ℕ) : 
  silk_pieces = 10 →
  cashmere_pieces = silk_pieces / 2 →
  total_pieces = 13 →
  cashmere_pieces - (total_pieces - silk_pieces) = 2 :=
by sorry

end fashion_line_blend_pieces_l1008_100894


namespace trigonometric_identity_l1008_100822

theorem trigonometric_identity : 
  (Real.sqrt 3) / (Real.cos (10 * π / 180)) - 1 / (Real.sin (170 * π / 180)) = -4 := by
  sorry

end trigonometric_identity_l1008_100822


namespace seven_digit_divisible_by_11_l1008_100883

/-- A seven-digit number in the form 945k317 is divisible by 11 if and only if k = 8 -/
theorem seven_digit_divisible_by_11 (k : ℕ) : k < 10 → (945000 + k * 1000 + 317) % 11 = 0 ↔ k = 8 := by
  sorry

end seven_digit_divisible_by_11_l1008_100883


namespace test_score_after_5_hours_l1008_100838

/-- A student's test score is directly proportional to study time -/
structure TestScore where
  maxPoints : ℝ
  scoreAfter2Hours : ℝ
  hoursStudied : ℝ
  score : ℝ
  proportional : scoreAfter2Hours / 2 = score / hoursStudied

/-- The theorem to prove -/
theorem test_score_after_5_hours (test : TestScore) 
  (h1 : test.maxPoints = 150)
  (h2 : test.scoreAfter2Hours = 90)
  (h3 : test.hoursStudied = 5) : 
  test.score = 225 := by
sorry

end test_score_after_5_hours_l1008_100838


namespace ladybugs_with_spots_l1008_100895

theorem ladybugs_with_spots (total : ℕ) (without_spots : ℕ) (with_spots : ℕ) : 
  total = 67082 → without_spots = 54912 → total = with_spots + without_spots → 
  with_spots = 12170 := by
sorry

end ladybugs_with_spots_l1008_100895


namespace pie_arrangement_l1008_100852

/-- Given the number of pecan and apple pies, calculates the number of complete rows when arranged with a fixed number of pies per row. -/
def calculate_rows (pecan_pies apple_pies pies_per_row : ℕ) : ℕ :=
  (pecan_pies + apple_pies) / pies_per_row

/-- Theorem stating that with 16 pecan pies and 14 apple pies, arranged in rows of 5 pies each, there will be 6 complete rows. -/
theorem pie_arrangement : calculate_rows 16 14 5 = 6 := by
  sorry

end pie_arrangement_l1008_100852


namespace total_tickets_sold_l1008_100863

def total_revenue : ℕ := 1933
def student_ticket_price : ℕ := 2
def nonstudent_ticket_price : ℕ := 3
def student_tickets_sold : ℕ := 530

theorem total_tickets_sold :
  ∃ (nonstudent_tickets : ℕ),
    student_tickets_sold * student_ticket_price +
    nonstudent_tickets * nonstudent_ticket_price = total_revenue ∧
    student_tickets_sold + nonstudent_tickets = 821 :=
by sorry

end total_tickets_sold_l1008_100863


namespace problem_proof_l1008_100849

theorem problem_proof (k n : ℤ) : 
  (5 + k) * (5 - k) = n - (2^3) → k = 2 → n = 29 := by
  sorry

end problem_proof_l1008_100849


namespace probability_ratio_l1008_100891

def total_slips : ℕ := 50
def numbers_per_slip : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 4

def probability_same_number (total : ℕ) (per_number : ℕ) (numbers : ℕ) (drawn : ℕ) : ℚ :=
  (numbers * Nat.choose per_number drawn) / Nat.choose total drawn

def probability_three_same_one_different (total : ℕ) (per_number : ℕ) (numbers : ℕ) (drawn : ℕ) : ℚ :=
  (numbers * Nat.choose per_number (drawn - 1) * (numbers - 1) * per_number) / Nat.choose total drawn

theorem probability_ratio :
  probability_three_same_one_different total_slips slips_per_number numbers_per_slip drawn_slips /
  probability_same_number total_slips slips_per_number numbers_per_slip drawn_slips = 90 := by
  sorry

end probability_ratio_l1008_100891


namespace third_grade_agreement_l1008_100819

theorem third_grade_agreement (total_agreed : ℕ) (fourth_grade_agreed : ℕ) 
  (h1 : total_agreed = 391) (h2 : fourth_grade_agreed = 237) :
  total_agreed - fourth_grade_agreed = 154 := by
  sorry

end third_grade_agreement_l1008_100819


namespace probability_theorem_l1008_100850

/-- Represents a standard deck of cards with additional properties -/
structure Deck :=
  (total : ℕ)
  (kings : ℕ)
  (aces : ℕ)
  (others : ℕ)
  (h1 : total = kings + aces + others)

/-- The probability of drawing either two aces or at least one king -/
def probability_two_aces_or_king (d : Deck) : ℚ :=
  sorry

/-- The theorem statement -/
theorem probability_theorem (d : Deck) 
  (h2 : d.total = 54) 
  (h3 : d.kings = 4) 
  (h4 : d.aces = 6) 
  (h5 : d.others = 44) : 
  probability_two_aces_or_king d = 221 / 1431 := by
  sorry

end probability_theorem_l1008_100850


namespace two_books_from_different_genres_l1008_100862

theorem two_books_from_different_genres :
  let num_genres : ℕ := 3
  let books_per_genre : ℕ := 4
  let choose_genres : ℕ := 2
  (num_genres.choose choose_genres) * books_per_genre * books_per_genre = 48 :=
by sorry

end two_books_from_different_genres_l1008_100862


namespace line_plane_relationship_l1008_100836

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (l : Line) (α : Plane) (m : Line)
  (h : parallel l m ∧ contained_in m α) :
  contained_in l α ∨ parallel_plane l α :=
sorry

end line_plane_relationship_l1008_100836


namespace vhs_to_dvd_cost_l1008_100833

def replace_cost (num_movies : ℕ) (trade_in_price : ℕ) (dvd_price : ℕ) : ℕ :=
  num_movies * dvd_price - num_movies * trade_in_price

theorem vhs_to_dvd_cost :
  replace_cost 100 2 10 = 800 := by
  sorry

end vhs_to_dvd_cost_l1008_100833


namespace arithmetic_calculations_l1008_100892

theorem arithmetic_calculations :
  ((1 : ℤ) * (-5) - (-6) + (-7) = -6) ∧
  ((-1 : ℤ)^2021 + (-18) * |(-2 : ℚ) / 9| - 4 / (-2 : ℤ) = -3) := by
  sorry

end arithmetic_calculations_l1008_100892


namespace sunflower_seed_contest_l1008_100810

theorem sunflower_seed_contest (total_seeds : ℕ) (first_player : ℕ) (second_player : ℕ) 
  (h1 : total_seeds = 214)
  (h2 : first_player = 78)
  (h3 : second_player = 53)
  (h4 : total_seeds = first_player + second_player + (total_seeds - first_player - second_player))
  (h5 : total_seeds - first_player - second_player > second_player) :
  (total_seeds - first_player - second_player) - second_player = 30 := by
  sorry

end sunflower_seed_contest_l1008_100810


namespace square_root_equality_l1008_100876

theorem square_root_equality (k : ℕ) (h : k > 0) :
  (∀ (i : ℕ), i > 0 → Real.sqrt (i + i / (i^2 - 1)) = i * Real.sqrt (i / (i^2 - 1))) →
  (let n : ℝ := 6
   let m : ℝ := 35
   Real.sqrt (6 + n / m) = 6 * Real.sqrt (n / m) ∧ m + n = 41) := by
sorry

end square_root_equality_l1008_100876


namespace absolute_value_inequality_solution_set_l1008_100805

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 3| + |x + 3| < 7} = Set.Icc (-1) (7/3) := by sorry

end absolute_value_inequality_solution_set_l1008_100805


namespace proof_by_contradiction_on_incorrect_statement_l1008_100888

-- Define a proposition
variable (P : Prop)

-- Define the property of being an incorrect statement
def is_incorrect (S : Prop) : Prop := ¬S

-- Define the process of attempting proof by contradiction
def attempt_proof_by_contradiction (S : Prop) : Prop :=
  ∃ (proof : ¬S → False), True

-- Define what it means for a proof method to fail to produce a useful conclusion
def fails_to_produce_useful_conclusion (S : Prop) : Prop :=
  ¬(S ∨ ¬S)

-- Theorem statement
theorem proof_by_contradiction_on_incorrect_statement
  (h : is_incorrect P) :
  attempt_proof_by_contradiction P →
  fails_to_produce_useful_conclusion P :=
by sorry

end proof_by_contradiction_on_incorrect_statement_l1008_100888


namespace max_value_constraint_l1008_100817

theorem max_value_constraint (m : ℝ) : m > 1 →
  (∃ (x y : ℝ), y ≥ x ∧ y ≤ m * x ∧ x + y ≤ 1 ∧ x + m * y < 2) ↔ m < 1 + Real.sqrt 2 := by
  sorry

end max_value_constraint_l1008_100817


namespace net_sales_for_10000_yuan_l1008_100825

/-- Represents the relationship between advertising expenses and sales revenue -/
def sales_model (x : ℝ) : ℝ := 8.5 * x + 17.5

/-- Calculates the net sales revenue given advertising expenses -/
def net_sales_revenue (x : ℝ) : ℝ := sales_model x - x

/-- Theorem: When advertising expenses are 1 (10,000 yuan), 
    the net sales revenue is 9.25 (92,500 yuan) -/
theorem net_sales_for_10000_yuan : net_sales_revenue 1 = 9.25 := by
  sorry

end net_sales_for_10000_yuan_l1008_100825


namespace survival_probabilities_correct_l1008_100884

/-- Mortality table data -/
structure MortalityData :=
  (reach28 : ℕ)
  (reach35 : ℕ)
  (reach48 : ℕ)
  (reach55 : ℕ)
  (total : ℕ)

/-- Survival probabilities after 20 years -/
structure SurvivalProbabilities :=
  (bothAlive : ℚ)
  (husbandDead : ℚ)
  (wifeDead : ℚ)
  (bothDead : ℚ)
  (husbandDeadWifeAlive : ℚ)
  (husbandAliveWifeDead : ℚ)

/-- Calculate survival probabilities based on mortality data -/
def calculateSurvivalProbabilities (data : MortalityData) : SurvivalProbabilities :=
  sorry

/-- Theorem stating the correct survival probabilities -/
theorem survival_probabilities_correct (data : MortalityData) 
  (h1 : data.reach28 = 675)
  (h2 : data.reach35 = 630)
  (h3 : data.reach48 = 540)
  (h4 : data.reach55 = 486)
  (h5 : data.total = 1000) :
  let probs := calculateSurvivalProbabilities data
  probs.bothAlive = 108 / 175 ∧
  probs.husbandDead = 8 / 35 ∧
  probs.wifeDead = 1 / 5 ∧
  probs.bothDead = 8 / 175 ∧
  probs.husbandDeadWifeAlive = 32 / 175 ∧
  probs.husbandAliveWifeDead = 27 / 175 :=
by
  sorry

#check survival_probabilities_correct

end survival_probabilities_correct_l1008_100884


namespace balls_in_boxes_count_l1008_100851

/-- The number of ways to place three distinct balls into three distinct boxes -/
def place_balls_in_boxes : ℕ := 27

/-- The number of choices for each ball -/
def choices_per_ball : ℕ := 3

/-- Theorem: The number of ways to place three distinct balls into three distinct boxes
    is equal to the cube of the number of choices for each ball -/
theorem balls_in_boxes_count :
  place_balls_in_boxes = choices_per_ball ^ 3 := by sorry

end balls_in_boxes_count_l1008_100851


namespace parabola_point_distance_l1008_100872

theorem parabola_point_distance (x₀ y₀ : ℝ) : 
  x₀^2 = 28 * y₀ →                           -- Point is on the parabola
  (y₀ + 7/2)^2 + x₀^2 = 9 * y₀^2 →           -- Distance to focus is 3 times distance to x-axis
  y₀ = 7/2 := by sorry

end parabola_point_distance_l1008_100872


namespace probability_of_seven_in_three_eighths_l1008_100846

theorem probability_of_seven_in_three_eighths :
  let decimal_rep := (3 : ℚ) / 8
  let digits := [3, 7, 5]
  (digits.count 7 : ℚ) / digits.length = 1 / 3 := by
sorry

end probability_of_seven_in_three_eighths_l1008_100846


namespace quadratic_inequality_product_l1008_100839

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set (-1, 1/3), prove that ab = 6 -/
theorem quadratic_inequality_product (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 1/3 ↔ a * x^2 + b * x + 1 > 0) →
  a * b = 6 := by
  sorry

end quadratic_inequality_product_l1008_100839


namespace smallest_y_for_perfect_fourth_power_l1008_100870

def x : ℕ := 7 * 24 * 48

theorem smallest_y_for_perfect_fourth_power (y : ℕ) :
  y = 6174 ↔ (
    y > 0 ∧
    ∃ (n : ℕ), x * y = n^4 ∧
    ∀ (z : ℕ), 0 < z ∧ z < y → ¬∃ (m : ℕ), x * z = m^4
  ) := by sorry

end smallest_y_for_perfect_fourth_power_l1008_100870


namespace solve_for_a_l1008_100832

-- Define the function f
def f (x a : ℝ) : ℝ := |x + 1| + |x - a|

-- State the theorem
theorem solve_for_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, f x a ≥ 5 ↔ x ≤ -2 ∨ x > 3) : a = 2 := by
  sorry

end solve_for_a_l1008_100832


namespace factorial_equation_solution_l1008_100808

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem factorial_equation_solution (N : ℕ) (h : N > 0) :
  factorial 5 * factorial 9 = 12 * factorial N → N = 10 := by
  sorry

end factorial_equation_solution_l1008_100808


namespace inequality_proof_l1008_100868

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  (1 / (b*c + a + 1/a)) + (1 / (a*c + b + 1/b)) + (1 / (a*b + c + 1/c)) ≤ 27/31 := by
  sorry

end inequality_proof_l1008_100868


namespace irrational_sqrt_three_rational_others_l1008_100802

theorem irrational_sqrt_three_rational_others : 
  (Irrational (Real.sqrt 3)) ∧ 
  (¬ Irrational (22 / 7 : ℝ)) ∧ 
  (¬ Irrational (0 : ℝ)) ∧ 
  (¬ Irrational (3.14 : ℝ)) := by sorry

end irrational_sqrt_three_rational_others_l1008_100802


namespace correct_assignment_l1008_100845

-- Define the colors and labels
inductive Color : Type
| White : Color
| Red : Color
| Yellow : Color
| Green : Color

def Label := Color

-- Define a package as a pair of label and actual color
structure Package where
  label : Label
  actual : Color

-- Define the condition that no label matches its actual content
def labelMismatch (p : Package) : Prop := p.label ≠ p.actual

-- Define the set of all packages
def allPackages : Finset Package := sorry

-- Define the property that all labels are different
def allLabelsDifferent (packages : Finset Package) : Prop := sorry

-- Define the property that all actual colors are different
def allActualColorsDifferent (packages : Finset Package) : Prop := sorry

-- Main theorem
theorem correct_assignment :
  ∀ (packages : Finset Package),
    packages = allPackages →
    (∀ p ∈ packages, labelMismatch p) →
    allLabelsDifferent packages →
    allActualColorsDifferent packages →
    ∃! (w r y g : Package),
      w ∈ packages ∧ r ∈ packages ∧ y ∈ packages ∧ g ∈ packages ∧
      w.label = Color.Red ∧ w.actual = Color.White ∧
      r.label = Color.White ∧ r.actual = Color.Red ∧
      y.label = Color.Green ∧ y.actual = Color.Yellow ∧
      g.label = Color.Yellow ∧ g.actual = Color.Green :=
by sorry

end correct_assignment_l1008_100845


namespace difference_product_sum_equals_difference_of_squares_l1008_100853

theorem difference_product_sum_equals_difference_of_squares (a b : ℝ) :
  (a - b) * (b + a) = a^2 - b^2 := by
  sorry

end difference_product_sum_equals_difference_of_squares_l1008_100853


namespace total_money_is_36000_l1008_100893

/-- The number of phones Vivienne has -/
def vivienne_phones : ℕ := 40

/-- The difference in number of phones between Aliyah and Vivienne -/
def phone_difference : ℕ := 10

/-- The price of each phone -/
def phone_price : ℕ := 400

/-- The total amount of money Aliyah and Vivienne have together after selling their phones -/
def total_money : ℕ := (vivienne_phones + (vivienne_phones + phone_difference)) * phone_price

theorem total_money_is_36000 : total_money = 36000 := by
  sorry

end total_money_is_36000_l1008_100893

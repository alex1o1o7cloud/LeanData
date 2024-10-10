import Mathlib

namespace condition_sufficient_not_necessary_l1177_117783

theorem condition_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 > 0 → a > b) ∧
  (∃ a b : ℝ, a > b ∧ ¬((a - b) * a^2 > 0)) := by
  sorry

end condition_sufficient_not_necessary_l1177_117783


namespace system_solution_l1177_117721

theorem system_solution (x y z : ℝ) :
  x + y + z = 2 ∧ x * y * z = 2 * (x * y + y * z + z * x) →
  ((x = -y ∧ z = 2) ∨ (y = -z ∧ x = 2) ∨ (z = -x ∧ y = 2)) := by
  sorry

end system_solution_l1177_117721


namespace sum_of_roots_eq_two_l1177_117792

theorem sum_of_roots_eq_two :
  let f : ℝ → ℝ := λ x => (x + 3) * (x - 5) - 19
  (∃ a b : ℝ, (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b) ∧ a + b = 2) :=
by sorry

end sum_of_roots_eq_two_l1177_117792


namespace valera_car_position_l1177_117757

/-- Represents a train with a fixed number of cars -/
structure Train :=
  (num_cars : ℕ)

/-- Represents the meeting of two trains -/
structure TrainMeeting :=
  (train1 : Train)
  (train2 : Train)
  (total_passing_time : ℕ)
  (sasha_passing_time : ℕ)
  (sasha_car : ℕ)

/-- Theorem stating the position of Valera's car -/
theorem valera_car_position
  (meeting : TrainMeeting)
  (h1 : meeting.train1.num_cars = 15)
  (h2 : meeting.train2.num_cars = 15)
  (h3 : meeting.total_passing_time = 60)
  (h4 : meeting.sasha_passing_time = 28)
  (h5 : meeting.sasha_car = 3) :
  ∃ (valera_car : ℕ), valera_car = 12 :=
by sorry

end valera_car_position_l1177_117757


namespace polynomial_sum_l1177_117734

theorem polynomial_sum (m : ℝ) : (m^2 + m) + (-3*m) = m^2 - 2*m := by
  sorry

end polynomial_sum_l1177_117734


namespace min_distance_squared_l1177_117728

/-- Given real numbers a, b, c, and d satisfying certain conditions,
    the minimum value of (a-c)² + (b-d)² is 1. -/
theorem min_distance_squared (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0)
  (h2 : 2 * d - c + Real.sqrt 5 = 0) : 
  ∃ (min : ℝ), min = 1 ∧ ∀ (a' b' c' d' : ℝ), 
    Real.log (b' + 1) + a' - 3 * b' = 0 → 
    2 * d' - c' + Real.sqrt 5 = 0 → 
    (a' - c')^2 + (b' - d')^2 ≥ min :=
sorry

end min_distance_squared_l1177_117728


namespace arithmetic_sequence_sum_l1177_117765

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {aₙ}, if a₄ + a₅ + a₆ = 90, then a₅ = 30 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : ArithmeticSequence a) (h2 : a 4 + a 5 + a 6 = 90) :
  a 5 = 30 := by
  sorry

end arithmetic_sequence_sum_l1177_117765


namespace largest_fraction_l1177_117705

theorem largest_fraction : 
  let fractions := [2/5, 3/7, 4/9, 5/11, 6/13]
  ∀ x ∈ fractions, (6/13 : ℚ) ≥ x :=
by sorry

end largest_fraction_l1177_117705


namespace two_digit_number_sum_l1177_117755

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 9 * (a + b) →
  (10 * a + b) + (10 * b + a) = 11 := by
sorry

end two_digit_number_sum_l1177_117755


namespace baba_yaga_students_l1177_117785

theorem baba_yaga_students (total : ℕ) (boys girls : ℕ) : 
  total = 33 →
  boys + girls = total →
  22 = (2 * total) / 3 := by
  sorry

end baba_yaga_students_l1177_117785


namespace correct_division_result_l1177_117764

theorem correct_division_result (wrong_divisor correct_divisor student_answer : ℕ) 
  (h1 : wrong_divisor = 840)
  (h2 : correct_divisor = 420)
  (h3 : student_answer = 36) :
  (wrong_divisor * student_answer) / correct_divisor = 72 := by
  sorry

end correct_division_result_l1177_117764


namespace chord_length_squared_l1177_117784

/-- Two circles with given properties and intersecting chords --/
structure CircleConfiguration where
  -- First circle radius
  r1 : ℝ
  -- Second circle radius
  r2 : ℝ
  -- Distance between circle centers
  d : ℝ
  -- Length of chord QP
  x : ℝ
  -- Ensure the configuration is valid
  h1 : r1 = 10
  h2 : r2 = 7
  h3 : d = 15
  -- QP = PR = PS = PT
  h4 : ∀ (chord : ℝ), chord = x → (chord = QP ∨ chord = PR ∨ chord = PS ∨ chord = PT)

/-- The theorem stating that the square of QP's length is 265 --/
theorem chord_length_squared (config : CircleConfiguration) : config.x^2 = 265 := by
  sorry

end chord_length_squared_l1177_117784


namespace fabian_shopping_cost_l1177_117788

/-- Calculates the total cost of Fabian's shopping --/
def shopping_cost (apple_price : ℝ) (walnut_price : ℝ) (apple_quantity : ℝ) (sugar_quantity : ℝ) (walnut_quantity : ℝ) : ℝ :=
  let sugar_price := apple_price - 1
  apple_price * apple_quantity + sugar_price * sugar_quantity + walnut_price * walnut_quantity

/-- Proves that the total cost of Fabian's shopping is $16 --/
theorem fabian_shopping_cost :
  shopping_cost 2 6 5 3 0.5 = 16 := by
  sorry

end fabian_shopping_cost_l1177_117788


namespace set_union_condition_l1177_117704

theorem set_union_condition (m : ℝ) : 
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 7}
  let B : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}
  A ∪ B = A → m ≤ -2 ∨ (-1 ≤ m ∧ m ≤ 3) := by
  sorry

end set_union_condition_l1177_117704


namespace square_of_product_l1177_117736

theorem square_of_product (a b : ℝ) : (-2 * a * b^3)^2 = 4 * a^2 * b^6 := by
  sorry

end square_of_product_l1177_117736


namespace fraction_order_l1177_117709

theorem fraction_order : 
  let f1 := 16/12
  let f2 := 21/14
  let f3 := 18/13
  let f4 := 20/15
  f1 < f3 ∧ f3 < f2 ∧ f2 < f4 := by
  sorry

end fraction_order_l1177_117709


namespace car_distance_in_30_minutes_l1177_117780

-- Define the train's speed in miles per hour
def train_speed : ℚ := 100

-- Define the car's speed as a fraction of the train's speed
def car_speed : ℚ := (2/3) * train_speed

-- Define the time in hours (30 minutes = 1/2 hour)
def time : ℚ := 1/2

-- Theorem statement
theorem car_distance_in_30_minutes :
  car_speed * time = 100/3 := by sorry

end car_distance_in_30_minutes_l1177_117780


namespace convex_pentagon_probability_l1177_117711

/-- The number of points on the circle -/
def num_points : ℕ := 8

/-- The number of chords to be selected -/
def num_selected_chords : ℕ := 5

/-- The total number of possible chords between num_points points -/
def total_chords (n : ℕ) : ℕ := n.choose 2

/-- The number of ways to select num_selected_chords from total_chords -/
def ways_to_select_chords (n : ℕ) : ℕ := (total_chords n).choose num_selected_chords

/-- The number of ways to choose 5 points from num_points points -/
def convex_pentagons (n : ℕ) : ℕ := n.choose 5

/-- The probability of forming a convex pentagon -/
def probability : ℚ := (convex_pentagons num_points : ℚ) / (ways_to_select_chords num_points : ℚ)

theorem convex_pentagon_probability :
  probability = 1 / 1755 :=
sorry

end convex_pentagon_probability_l1177_117711


namespace probability_male_saturday_female_sunday_l1177_117720

/-- The number of male students -/
def num_male : ℕ := 2

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of days in the event -/
def num_days : ℕ := 2

/-- The probability of selecting a male student for Saturday and a female student for Sunday -/
theorem probability_male_saturday_female_sunday :
  (num_male * num_female) / (total_students * (total_students - 1)) = 1 / 3 := by
  sorry

end probability_male_saturday_female_sunday_l1177_117720


namespace sample_capacity_l1177_117760

/-- Given a sample divided into groups, prove that the sample capacity is 320
    when a certain group has a frequency of 40 and a rate of 0.125. -/
theorem sample_capacity (frequency : ℕ) (rate : ℝ) (n : ℕ) 
  (h1 : frequency = 40)
  (h2 : rate = 0.125)
  (h3 : (rate : ℝ) * n = frequency) : 
  n = 320 := by
  sorry

end sample_capacity_l1177_117760


namespace min_distance_PQ_l1177_117799

def f (x : ℝ) : ℝ := x^2 - 2*x

def distance_squared (x : ℝ) : ℝ := (x - 4)^2 + (f x + 1)^2

theorem min_distance_PQ :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 ∧
  ∀ (x : ℝ), Real.sqrt (distance_squared x) ≥ min_dist :=
sorry

end min_distance_PQ_l1177_117799


namespace allison_wins_prob_l1177_117746

/-- Represents a 6-sided cube with specific face configurations -/
structure Cube where
  faces : Fin 6 → ℕ

/-- Allison's cube configuration -/
def allison_cube : Cube :=
  { faces := λ i => if i.val < 3 then 3 else 4 }

/-- Brian's cube configuration -/
def brian_cube : Cube :=
  { faces := λ i => i.val }

/-- Noah's cube configuration -/
def noah_cube : Cube :=
  { faces := λ i => if i.val < 3 then 2 else 6 }

/-- Probability of rolling a specific value on a cube -/
def prob_roll (c : Cube) (v : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i = v) (Finset.univ : Finset (Fin 6))).card / 6

/-- Probability of rolling less than a value on a cube -/
def prob_roll_less (c : Cube) (v : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i < v) (Finset.univ : Finset (Fin 6))).card / 6

/-- The main theorem stating the probability of Allison winning -/
theorem allison_wins_prob :
  (1 / 2) * (prob_roll_less brian_cube 3 * prob_roll_less noah_cube 3 +
             prob_roll_less brian_cube 4 * prob_roll_less noah_cube 4) = 7 / 24 := by
  sorry

#check allison_wins_prob

end allison_wins_prob_l1177_117746


namespace incorrect_statement_l1177_117732

theorem incorrect_statement : 
  ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end incorrect_statement_l1177_117732


namespace sum_of_odd_numbers_l1177_117798

theorem sum_of_odd_numbers (N : ℕ) : 
  991 + 993 + 995 + 997 + 999 = 5000 - N → N = 25 := by
  sorry

end sum_of_odd_numbers_l1177_117798


namespace sum_of_roots_equation_l1177_117767

theorem sum_of_roots_equation (x : ℝ) : 
  let eq := (3*x + 4)*(x - 3) + (3*x + 4)*(x - 5) = 0
  ∃ (r₁ r₂ : ℝ), (3*r₁ + 4)*(r₁ - 3) + (3*r₁ + 4)*(r₁ - 5) = 0 ∧
                 (3*r₂ + 4)*(r₂ - 3) + (3*r₂ + 4)*(r₂ - 5) = 0 ∧
                 r₁ + r₂ = 8/3 := by
  sorry

end sum_of_roots_equation_l1177_117767


namespace abc_inequality_l1177_117796

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1) = 2) :
  a * b + b * c + c * a ≤ 3 / 2 := by
  sorry

end abc_inequality_l1177_117796


namespace range_of_power_function_l1177_117769

theorem range_of_power_function (k c : ℝ) (h_k : k > 0) :
  Set.range (fun x => x^k + c) = Set.Ici (1 + c) := by sorry

end range_of_power_function_l1177_117769


namespace lawrence_county_houses_l1177_117750

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := 574

/-- The total number of houses in Lawrence County after the housing boom -/
def total_houses : ℕ := houses_before + houses_built

theorem lawrence_county_houses : total_houses = 2000 := by
  sorry

end lawrence_county_houses_l1177_117750


namespace max_cookies_eaten_l1177_117795

/-- Given 36 cookies shared among three siblings, where one sibling eats twice as many as another,
    and the third eats the same as the second, the maximum number of cookies the second sibling
    could have eaten is 9. -/
theorem max_cookies_eaten (total_cookies : ℕ) (andy bella charlie : ℕ) : 
  total_cookies = 36 →
  bella = 2 * andy →
  charlie = andy →
  total_cookies = andy + bella + charlie →
  andy ≤ 9 :=
by sorry

end max_cookies_eaten_l1177_117795


namespace x_value_from_ratio_l1177_117762

theorem x_value_from_ratio (x y : ℝ) :
  x / (x - 1) = (y^3 + 2*y - 1) / (y^3 + 2*y - 3) →
  x = (y^3 + 2*y - 1) / 2 := by
  sorry

end x_value_from_ratio_l1177_117762


namespace toy_distribution_ratio_l1177_117701

theorem toy_distribution_ratio (total_toys : ℕ) (num_friends : ℕ) 
  (h1 : total_toys = 118) (h2 : num_friends = 4) :
  ∃ (toys_per_friend : ℕ), 
    toys_per_friend * num_friends ≤ total_toys ∧
    toys_per_friend * num_friends > total_toys - num_friends ∧
    (toys_per_friend : ℚ) / total_toys = 1 / 4 := by
  sorry

#check toy_distribution_ratio

end toy_distribution_ratio_l1177_117701


namespace fruit_buckets_l1177_117771

theorem fruit_buckets (bucketA bucketB bucketC : ℕ) : 
  bucketA = bucketB + 4 →
  bucketB = bucketC + 3 →
  bucketA + bucketB + bucketC = 37 →
  bucketC = 9 := by
sorry

end fruit_buckets_l1177_117771


namespace pipe_a_fill_time_l1177_117740

/-- Represents the time (in minutes) it takes for Pipe A to fill the tank alone -/
def pipe_a_time : ℝ := 21

/-- Represents how many times faster Pipe B is compared to Pipe A -/
def pipe_b_speed_ratio : ℝ := 6

/-- Represents the time (in minutes) it takes for both pipes to fill the tank together -/
def combined_time : ℝ := 3

/-- Proves that the time taken by Pipe A to fill the tank alone is 21 minutes -/
theorem pipe_a_fill_time :
  (1 / pipe_a_time + pipe_b_speed_ratio / pipe_a_time) * combined_time = 1 :=
sorry

end pipe_a_fill_time_l1177_117740


namespace abs_ratio_eq_sqrt_five_half_l1177_117793

theorem abs_ratio_eq_sqrt_five_half (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + y^2 = 18*x*y) : 
  |((x+y)/(x-y))| = Real.sqrt 5 / 2 := by
  sorry

end abs_ratio_eq_sqrt_five_half_l1177_117793


namespace power_two_greater_than_square_l1177_117778

theorem power_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end power_two_greater_than_square_l1177_117778


namespace xy_system_solution_l1177_117797

theorem xy_system_solution (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 110) :
  x^2 + y^2 = 8044 / 169 := by
sorry

end xy_system_solution_l1177_117797


namespace independent_events_probability_l1177_117768

theorem independent_events_probability (a b : Set α) (p : Set α → ℚ) :
  (p a = 4/7) → (p b = 2/5) → (∀ x y, p (x ∩ y) = p x * p y) → p (a ∩ b) = 8/35 := by
  sorry

end independent_events_probability_l1177_117768


namespace shekars_average_marks_l1177_117773

/-- Calculates the average marks given scores in five subjects -/
def averageMarks (math science socialStudies english biology : ℕ) : ℚ :=
  (math + science + socialStudies + english + biology : ℚ) / 5

/-- Theorem stating that Shekar's average marks are 75 -/
theorem shekars_average_marks :
  averageMarks 76 65 82 67 85 = 75 := by
  sorry

end shekars_average_marks_l1177_117773


namespace inequality_preservation_l1177_117763

theorem inequality_preservation (a b : ℝ) (h : a < b) : a - 3 < b - 3 := by
  sorry

end inequality_preservation_l1177_117763


namespace radio_operator_distribution_probability_l1177_117747

theorem radio_operator_distribution_probability :
  let total_soldiers : ℕ := 12
  let radio_operators : ℕ := 3
  let group_sizes : List ℕ := [3, 4, 5]
  
  let total_distributions : ℕ := (total_soldiers.choose group_sizes[0]!) * ((total_soldiers - group_sizes[0]!).choose group_sizes[1]!) * 1
  
  let favorable_distributions : ℕ := ((total_soldiers - radio_operators).choose (group_sizes[0]! - 1)) *
    ((total_soldiers - radio_operators - (group_sizes[0]! - 1)).choose (group_sizes[1]! - 1)) * 
    ((radio_operators).factorial)
  
  (favorable_distributions : ℚ) / total_distributions = 3 / 11 := by
  sorry

end radio_operator_distribution_probability_l1177_117747


namespace intersection_M_N_l1177_117719

def M : Set ℝ := {-1, 1, 2, 3, 4}
def N : Set ℝ := {x : ℝ | x^2 + 2*x > 3}

theorem intersection_M_N : M ∩ N = {2, 3, 4} := by
  sorry

end intersection_M_N_l1177_117719


namespace computer_operations_l1177_117789

theorem computer_operations (additions_per_second multiplications_per_second : ℕ) 
  (h1 : additions_per_second = 12000)
  (h2 : multiplications_per_second = 8000) :
  (additions_per_second + multiplications_per_second) * (30 * 60) = 36000000 := by
  sorry

#check computer_operations

end computer_operations_l1177_117789


namespace identity_function_theorem_l1177_117766

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

theorem identity_function_theorem (f : ℕ+ → ℕ+) : 
  (∀ x y : ℕ+, is_perfect_square (x * f x + 2 * x * f y + (f y) ^ 2)) → 
  (∀ x : ℕ+, f x = x) :=
sorry

end identity_function_theorem_l1177_117766


namespace wrok_represents_5167_l1177_117727

/-- Represents a mapping from characters to digits -/
def CodeMapping : Type := Char → Nat

/-- The code "GREAT WORK" represents digits 0-8 respectively -/
def great_work_code (mapping : CodeMapping) : Prop :=
  mapping 'G' = 0 ∧
  mapping 'R' = 1 ∧
  mapping 'E' = 2 ∧
  mapping 'A' = 3 ∧
  mapping 'T' = 4 ∧
  mapping 'W' = 5 ∧
  mapping 'O' = 6 ∧
  mapping 'R' = 1 ∧
  mapping 'K' = 7

/-- The code word "WROK" represents a 4-digit number -/
def wrok_code (mapping : CodeMapping) : Nat :=
  mapping 'W' * 1000 + mapping 'R' * 100 + mapping 'O' * 10 + mapping 'K'

theorem wrok_represents_5167 (mapping : CodeMapping) :
  great_work_code mapping → wrok_code mapping = 5167 := by
  sorry

end wrok_represents_5167_l1177_117727


namespace women_group_size_l1177_117754

/-- The number of women in the first group -/
def first_group_size : ℕ := 6

/-- The length of cloth colored by the first group -/
def first_group_cloth_length : ℕ := 180

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 3

/-- The number of women in the second group -/
def second_group_size : ℕ := 5

/-- The length of cloth colored by the second group -/
def second_group_cloth_length : ℕ := 200

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 4

theorem women_group_size :
  first_group_size * second_group_cloth_length * first_group_days =
  second_group_size * first_group_cloth_length * second_group_days :=
by sorry

end women_group_size_l1177_117754


namespace polynomial_divisibility_l1177_117725

theorem polynomial_divisibility (k n : ℕ+) 
  (h : (k : ℝ) + 1 ≤ Real.sqrt ((n : ℝ) + 1 / Real.log (n + 1))) :
  ∃ (P : Polynomial ℤ), 
    (∀ i, P.coeff i ∈ ({0, 1, -1} : Set ℤ)) ∧ 
    P.degree = n ∧ 
    (X - 1 : Polynomial ℤ)^(k : ℕ) ∣ P :=
sorry

end polynomial_divisibility_l1177_117725


namespace min_value_constraint_l1177_117776

theorem min_value_constraint (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_constraint : x^3 * y^2 * z = 1) : 
  x + 2*y + 3*z ≥ 2 ∧ ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀^3 * y₀^2 * z₀ = 1 ∧ 
    x₀ + 2*y₀ + 3*z₀ = 2 :=
sorry

end min_value_constraint_l1177_117776


namespace kanul_cash_percentage_l1177_117779

def total_amount : ℝ := 5555.56
def raw_materials_cost : ℝ := 3000
def machinery_cost : ℝ := 2000

theorem kanul_cash_percentage :
  (total_amount - (raw_materials_cost + machinery_cost)) / total_amount * 100 = 10 := by
  sorry

end kanul_cash_percentage_l1177_117779


namespace smallest_number_of_students_l1177_117713

theorem smallest_number_of_students (n : ℕ) : 
  n > 0 ∧
  (n : ℚ) * (75 : ℚ) / 100 = ↑(n - (n / 4 : ℕ)) ∧
  (n / 40 : ℕ) = (n / 4 : ℕ) * 10 / 100 ∧
  (33 * n / 200 : ℕ) = ((11 * n / 100 : ℕ) * 3 / 2 : ℕ) ∧
  ∀ m : ℕ, m > 0 ∧ 
    (m : ℚ) * (75 : ℚ) / 100 = ↑(m - (m / 4 : ℕ)) ∧
    (m / 40 : ℕ) = (m / 4 : ℕ) * 10 / 100 ∧
    (33 * m / 200 : ℕ) = ((11 * m / 100 : ℕ) * 3 / 2 : ℕ) →
    m ≥ n →
  n = 200 := by sorry

end smallest_number_of_students_l1177_117713


namespace complement_of_union_l1177_117758

open Set

theorem complement_of_union (U M N : Set ℕ) : 
  U = {1, 2, 3, 4} →
  M = {1, 2} →
  N = {2, 3} →
  (U \ (M ∪ N)) = {4} := by
  sorry

end complement_of_union_l1177_117758


namespace ladder_matches_l1177_117781

/-- Represents the number of matches needed for a ladder with a given number of steps. -/
def matches_for_ladder (steps : ℕ) : ℕ :=
  6 * steps

theorem ladder_matches :
  matches_for_ladder 3 = 18 →
  matches_for_ladder 25 = 150 :=
by sorry

end ladder_matches_l1177_117781


namespace susan_is_eleven_l1177_117710

/-- Susan's age -/
def susan_age : ℕ := sorry

/-- Ann's age -/
def ann_age : ℕ := sorry

/-- Ann is 5 years older than Susan -/
axiom age_difference : ann_age = susan_age + 5

/-- The sum of their ages is 27 -/
axiom age_sum : ann_age + susan_age = 27

/-- Proof that Susan is 11 years old -/
theorem susan_is_eleven : susan_age = 11 := by sorry

end susan_is_eleven_l1177_117710


namespace algorithm_correctness_l1177_117761

def sum_2i (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2)

theorem algorithm_correctness :
  (sum_2i 3 = 12) ∧
  (∀ m : ℕ, sum_2i m = 30 → m ≥ 5) ∧
  (sum_2i 5 = 30) := by
sorry

end algorithm_correctness_l1177_117761


namespace angle_four_value_l1177_117752

/-- Given an isosceles triangle and some angle relationships, prove that angle 4 is 37.5 degrees -/
theorem angle_four_value (angle1 angle2 angle3 angle4 angle5 x y : ℝ) : 
  angle1 + angle2 = 180 →
  angle3 = angle4 →
  angle3 + angle4 + angle5 = 180 →
  angle1 = 45 + x →
  angle3 = 30 + y →
  x = 2 * y →
  angle4 = 37.5 := by
sorry

end angle_four_value_l1177_117752


namespace solution_range_l1177_117735

def M (a : ℝ) := {x : ℝ | (a - 2) * x^2 + (2*a - 1) * x + 6 > 0}

theorem solution_range (a : ℝ) (h1 : 3 ∈ M a) (h2 : 5 ∉ M a) : 1 < a ∧ a ≤ 7/5 := by
  sorry

end solution_range_l1177_117735


namespace chord_length_intercepted_by_line_l1177_117737

/-- The chord length intercepted by a line on a circle -/
theorem chord_length_intercepted_by_line (x y : ℝ) : 
  let circle : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 4}
  let line : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}
  let chord_length := Real.sqrt (8 : ℝ)
  (∃ p q : ℝ × ℝ, p ∈ circle ∧ q ∈ circle ∧ p ∈ line ∧ q ∈ line ∧ 
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = chord_length^2) :=
by
  sorry


end chord_length_intercepted_by_line_l1177_117737


namespace triangles_drawn_l1177_117777

theorem triangles_drawn (squares pentagons total_lines : ℕ) 
  (h_squares : squares = 8)
  (h_pentagons : pentagons = 4)
  (h_total_lines : total_lines = 88) :
  ∃ (triangles : ℕ), 
    3 * triangles + 4 * squares + 5 * pentagons = total_lines ∧ 
    triangles = 12 := by
  sorry

end triangles_drawn_l1177_117777


namespace father_son_age_ratio_l1177_117712

def father_age : ℕ := 40
def son_age : ℕ := 10

theorem father_son_age_ratio :
  (father_age : ℚ) / son_age = 4 ∧
  father_age + 20 = 2 * (son_age + 20) :=
by sorry

end father_son_age_ratio_l1177_117712


namespace tangent_slope_at_2_6_l1177_117722

-- Define the function f(x) = x³ - 2x + 2
def f (x : ℝ) : ℝ := x^3 - 2*x + 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_at_2_6 :
  f 2 = 6 ∧ f' 2 = 10 :=
sorry

end tangent_slope_at_2_6_l1177_117722


namespace raisin_cost_fraction_is_three_twentythirds_l1177_117770

/-- Represents the cost of ingredients relative to raisins -/
structure RelativeCost where
  raisins : ℚ := 1
  nuts : ℚ := 4
  dried_berries : ℚ := 2

/-- Represents the composition of the mixture in pounds -/
structure MixtureComposition where
  raisins : ℚ := 3
  nuts : ℚ := 4
  dried_berries : ℚ := 2

/-- Calculates the fraction of total cost attributed to raisins -/
def raisin_cost_fraction (rc : RelativeCost) (mc : MixtureComposition) : ℚ :=
  (mc.raisins * rc.raisins) / 
  (mc.raisins * rc.raisins + mc.nuts * rc.nuts + mc.dried_berries * rc.dried_berries)

theorem raisin_cost_fraction_is_three_twentythirds 
  (rc : RelativeCost) (mc : MixtureComposition) : 
  raisin_cost_fraction rc mc = 3 / 23 := by
  sorry

end raisin_cost_fraction_is_three_twentythirds_l1177_117770


namespace car_speed_problem_l1177_117708

theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 42 ∧ average_speed = 66 →
  ∃ speed_first_hour : ℝ,
    speed_first_hour = 90 ∧
    average_speed = (speed_first_hour + speed_second_hour) / 2 :=
by
  sorry

end car_speed_problem_l1177_117708


namespace wire_length_ratio_l1177_117707

theorem wire_length_ratio : 
  let large_cube_edge : ℝ := 8
  let large_cube_edges : ℕ := 12
  let unit_cube_edge : ℝ := 1
  let unit_cube_edges : ℕ := 12

  let large_cube_volume := large_cube_edge ^ 3
  let num_unit_cubes := large_cube_volume

  let large_cube_wire_length := large_cube_edge * large_cube_edges
  let unit_cubes_wire_length := num_unit_cubes * unit_cube_edge * unit_cube_edges

  large_cube_wire_length / unit_cubes_wire_length = 1 / 64 :=
by
  sorry

end wire_length_ratio_l1177_117707


namespace l_shaped_grid_squares_l1177_117742

/-- Represents a modified L-shaped grid -/
structure LShapedGrid :=
  (size : Nat)
  (missing_size : Nat)
  (missing_row : Nat)
  (missing_col : Nat)

/-- Counts the number of squares in the L-shaped grid -/
def count_squares (grid : LShapedGrid) : Nat :=
  sorry

/-- The main theorem stating that the number of squares in the specific L-shaped grid is 61 -/
theorem l_shaped_grid_squares :
  let grid : LShapedGrid := {
    size := 6,
    missing_size := 2,
    missing_row := 5,
    missing_col := 1
  }
  count_squares grid = 61 := by sorry

end l_shaped_grid_squares_l1177_117742


namespace line_passes_through_circle_center_l1177_117759

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 0)

-- Theorem: The line passes through the center of the circle
theorem line_passes_through_circle_center :
  line_equation (circle_center.1) (circle_center.2) := by
  sorry


end line_passes_through_circle_center_l1177_117759


namespace star_value_l1177_117791

-- Define the * operation for non-zero integers
def star (a b : ℤ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem star_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 12) (h4 : a * b = 32) :
  star a b = 3 / 8 := by
  sorry

end star_value_l1177_117791


namespace geometric_series_ratio_l1177_117738

/-- Given a geometric series with first term a and common ratio r,
    prove that if the sum of the series is 20 and the sum of terms
    with odd powers of r is 8, then r = √(11/12) -/
theorem geometric_series_ratio (a r : ℝ) (h₁ : a ≠ 0) (h₂ : |r| < 1) :
  (a / (1 - r) = 20) →
  (a * r / (1 - r^2) = 8) →
  r = Real.sqrt (11/12) := by
sorry

end geometric_series_ratio_l1177_117738


namespace quadratic_integer_solutions_l1177_117794

theorem quadratic_integer_solutions (p q x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) →
  (x₂^2 + p*x₂ + q = 0) →
  |x₁ - x₂| = 1 →
  |p - q| = 1 →
  (∃ (p' q' x₁' x₂' : ℤ), p = p' ∧ q = q' ∧ x₁ = x₁' ∧ x₂ = x₂') := by
sorry

end quadratic_integer_solutions_l1177_117794


namespace disk_covering_radius_bound_l1177_117718

theorem disk_covering_radius_bound (R : ℝ) (r : ℝ) :
  R = 1 →
  (∃ (centers : Fin 7 → ℝ × ℝ),
    (∀ x y : ℝ × ℝ, (x.1 - y.1)^2 + (x.2 - y.2)^2 ≤ R^2 →
      ∃ i : Fin 7, (x.1 - (centers i).1)^2 + (x.2 - (centers i).2)^2 ≤ r^2)) →
  r ≥ 1/2 :=
by sorry

end disk_covering_radius_bound_l1177_117718


namespace incorrect_transformation_l1177_117751

theorem incorrect_transformation :
  (∀ a b : ℝ, a - 3 = b - 3 → a = b) ∧
  (∀ a b c : ℝ, c ≠ 0 → a / c = b / c → a = b) ∧
  (∀ a b c : ℝ, a = b → a / (c^2 + 1) = b / (c^2 + 1)) ∧
  ¬(∀ a b c : ℝ, a * c = b * c → a = b) := by
sorry


end incorrect_transformation_l1177_117751


namespace percentage_difference_l1177_117724

theorem percentage_difference (x y : ℝ) (h : x = 0.5 * y) : y = 2 * x := by
  sorry

end percentage_difference_l1177_117724


namespace max_gcd_consecutive_terms_is_two_l1177_117703

/-- The sequence a_n defined as n^2! + n -/
def a (n : ℕ) : ℕ := (Nat.factorial (n^2)) + n

/-- The theorem stating that the maximum GCD of consecutive terms in the sequence is 2 -/
theorem max_gcd_consecutive_terms_is_two :
  ∃ (k : ℕ), (∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ k) ∧ 
  (∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = k) ∧
  k = 2 := by
  sorry

end max_gcd_consecutive_terms_is_two_l1177_117703


namespace hyperbola_eccentricity_from_tangent_circle_l1177_117706

/-- Given a circle and a hyperbola, if the circle is tangent to the asymptotes of the hyperbola,
    then the eccentricity of the hyperbola is 5/2. -/
theorem hyperbola_eccentricity_from_tangent_circle
  (a b : ℝ) (h_positive : a > 0 ∧ b > 0) :
  let circle := fun (x y : ℝ) => x^2 + y^2 - 10*y + 21 = 0
  let hyperbola := fun (x y : ℝ) => x^2/a^2 - y^2/b^2 = 1
  let asymptote := fun (x y : ℝ) => b*x - a*y = 0 ∨ b*x + a*y = 0
  let is_tangent := ∃ (x y : ℝ), circle x y ∧ asymptote x y
  let eccentricity := Real.sqrt (1 + b^2/a^2)
  is_tangent → eccentricity = 5/2 :=
by
  sorry

end hyperbola_eccentricity_from_tangent_circle_l1177_117706


namespace taylor_family_reunion_l1177_117782

theorem taylor_family_reunion (kids : ℕ) (adults : ℕ) (tables : ℕ) 
  (h1 : kids = 45) 
  (h2 : adults = 123) 
  (h3 : tables = 14) : 
  (kids + adults) / tables = 12 := by
sorry

end taylor_family_reunion_l1177_117782


namespace maggie_bouncy_balls_l1177_117787

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℝ := 10.0

/-- The number of yellow bouncy ball packs Maggie bought -/
def yellow_packs : ℝ := 8.0

/-- The number of green bouncy ball packs Maggie gave away -/
def green_packs_given : ℝ := 4.0

/-- The number of green bouncy ball packs Maggie bought -/
def green_packs_bought : ℝ := 4.0

/-- The total number of bouncy balls Maggie kept -/
def total_balls : ℝ := yellow_packs * balls_per_pack + green_packs_bought * balls_per_pack - green_packs_given * balls_per_pack

theorem maggie_bouncy_balls : total_balls = 80.0 := by
  sorry

end maggie_bouncy_balls_l1177_117787


namespace sum_of_decimals_l1177_117744

/-- The sum of 123.45 and 678.90 is equal to 802.35 -/
theorem sum_of_decimals : (123.45 : ℝ) + 678.90 = 802.35 := by sorry

end sum_of_decimals_l1177_117744


namespace rope_cutting_l1177_117786

/-- Proves that a 200-meter rope cut into equal parts, with half given away and the rest subdivided,
    results in 25-meter pieces if and only if it was initially cut into 8 parts. -/
theorem rope_cutting (total_length : ℕ) (final_piece_length : ℕ) (initial_parts : ℕ) : 
  total_length = 200 ∧ 
  final_piece_length = 25 ∧
  (initial_parts : ℚ) * final_piece_length = total_length ∧
  (initial_parts / 2 : ℚ) * 2 * final_piece_length = total_length →
  initial_parts = 8 :=
by sorry

end rope_cutting_l1177_117786


namespace five_objects_three_containers_l1177_117756

/-- The number of ways to put n distinguishable objects into k distinguishable containers -/
def num_ways (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to put 5 distinguishable objects into 3 distinguishable containers is 3^5 -/
theorem five_objects_three_containers : num_ways 5 3 = 3^5 := by
  sorry

end five_objects_three_containers_l1177_117756


namespace irrational_sqrt_7_and_others_rational_l1177_117731

theorem irrational_sqrt_7_and_others_rational : 
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 7 = (a : ℝ) / (b : ℝ)) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (4 : ℝ) / 3 = (a : ℝ) / (b : ℝ)) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (3.14 : ℝ) = (a : ℝ) / (b : ℝ)) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 4 = (a : ℝ) / (b : ℝ)) :=
by sorry

end irrational_sqrt_7_and_others_rational_l1177_117731


namespace quadratic_equation_solution_l1177_117702

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 + 6*x + 9 = 0 ↔ x = -3 := by sorry

end quadratic_equation_solution_l1177_117702


namespace division_problem_l1177_117717

theorem division_problem : 240 / (12 + 14 * 2) = 6 := by
  sorry

end division_problem_l1177_117717


namespace quadratic_roots_ratio_l1177_117748

theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 / 2 ∧ 
   r^2 + 10*r + k = 0 ∧ s^2 + 10*s + k = 0) → k = 24 := by
sorry

end quadratic_roots_ratio_l1177_117748


namespace balloons_in_park_l1177_117729

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 4

/-- The total number of balloons Allan and Jake have in the park -/
def total_balloons : ℕ := allan_balloons + jake_balloons

theorem balloons_in_park : total_balloons = 6 := by
  sorry

end balloons_in_park_l1177_117729


namespace elephant_entry_rate_utopia_park_elephant_rate_l1177_117726

/-- Calculates the rate at which new elephants entered Utopia National Park --/
theorem elephant_entry_rate (initial_elephants : ℕ) (exodus_rate : ℕ) (exodus_duration : ℕ) 
  (entry_duration : ℕ) (final_elephants : ℕ) : ℕ :=
  let elephants_left := exodus_rate * exodus_duration
  let elephants_after_exodus := initial_elephants - elephants_left
  let new_elephants := final_elephants - elephants_after_exodus
  new_elephants / entry_duration

/-- Proves that the rate of new elephants entering the park is 1500 per hour --/
theorem utopia_park_elephant_rate : 
  elephant_entry_rate 30000 2880 4 7 28980 = 1500 := by
  sorry

end elephant_entry_rate_utopia_park_elephant_rate_l1177_117726


namespace magician_marbles_left_l1177_117716

/-- Calculates the total number of marbles left after removing some from each color --/
def marblesLeft (initialRed initialBlue initialGreen redTaken : ℕ) : ℕ :=
  let blueTaken := 5 * redTaken
  let greenTaken := blueTaken / 2
  let redLeft := initialRed - redTaken
  let blueLeft := initialBlue - blueTaken
  let greenLeft := initialGreen - greenTaken
  redLeft + blueLeft + greenLeft

/-- Theorem stating that given the initial numbers of marbles and the rules for taking away marbles,
    the total number of marbles left is 93 --/
theorem magician_marbles_left :
  marblesLeft 40 60 35 5 = 93 := by
  sorry

end magician_marbles_left_l1177_117716


namespace max_value_expression_l1177_117745

theorem max_value_expression (x y : ℝ) : 
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + 4 * y^2 + 2) ≤ Real.sqrt 29 := by
  sorry

end max_value_expression_l1177_117745


namespace company_workforce_l1177_117733

theorem company_workforce (initial_workforce : ℕ) : 
  (initial_workforce * 60 = initial_workforce * 100 * 3 / 5) →
  ((initial_workforce * 60 : ℕ) = ((initial_workforce + 28) * 55 : ℕ)) →
  (initial_workforce + 28 = 336) := by
  sorry

end company_workforce_l1177_117733


namespace opposite_sides_inequality_l1177_117753

/-- Given that point P (x₀, y₀) and point A (1, 2) are on opposite sides of the line l: 3x + 2y - 8 = 0,
    prove that 3x₀ + 2y₀ > 8 -/
theorem opposite_sides_inequality (x₀ y₀ : ℝ) : 
  (3*x₀ + 2*y₀ - 8) * (3*1 + 2*2 - 8) < 0 → 3*x₀ + 2*y₀ > 8 := by
  sorry

end opposite_sides_inequality_l1177_117753


namespace rectangle_area_l1177_117774

theorem rectangle_area (width : ℝ) (length : ℝ) : 
  length = 4 * width →
  2 * length + 2 * width = 200 →
  length * width = 1600 := by
sorry

end rectangle_area_l1177_117774


namespace basketball_lineup_combinations_l1177_117749

/-- The number of possible starting lineups for a basketball team --/
theorem basketball_lineup_combinations (total_players : ℕ) 
  (guaranteed_players : ℕ) (excluded_players : ℕ) (lineup_size : ℕ) : 
  total_players = 15 → 
  guaranteed_players = 2 → 
  excluded_players = 1 → 
  lineup_size = 6 → 
  Nat.choose (total_players - guaranteed_players - excluded_players) 
             (lineup_size - guaranteed_players) = 495 := by
  sorry

#check basketball_lineup_combinations

end basketball_lineup_combinations_l1177_117749


namespace average_breadth_is_18_l1177_117739

/-- Represents a trapezoidal plot with equal diagonal distances -/
structure TrapezoidalPlot where
  averageBreadth : ℝ
  maximumLength : ℝ
  area : ℝ

/-- The conditions of the problem -/
def PlotConditions (plot : TrapezoidalPlot) : Prop :=
  plot.area = 23 * plot.averageBreadth ∧
  plot.maximumLength - plot.averageBreadth = 10 ∧
  plot.area = (1/2) * (plot.maximumLength + plot.averageBreadth) * plot.averageBreadth

/-- The theorem to be proved -/
theorem average_breadth_is_18 (plot : TrapezoidalPlot) 
  (h : PlotConditions plot) : plot.averageBreadth = 18 := by
  sorry

end average_breadth_is_18_l1177_117739


namespace bottles_bought_l1177_117772

theorem bottles_bought (initial bottles_drunk final : ℕ) : 
  initial = 42 → bottles_drunk = 25 → final = 47 → 
  final - (initial - bottles_drunk) = 30 := by sorry

end bottles_bought_l1177_117772


namespace max_value_cos_sin_l1177_117700

theorem max_value_cos_sin (x : ℝ) : 3 * Real.cos x - Real.sin x ≤ Real.sqrt 10 := by
  sorry

end max_value_cos_sin_l1177_117700


namespace functional_equation_solution_l1177_117730

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

/-- The theorem stating the form of functions satisfying the functional equation -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f →
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b * x^2 := by
  sorry

end functional_equation_solution_l1177_117730


namespace ratio_equality_l1177_117714

theorem ratio_equality (x y : ℝ) (h1 : 2 * x = 3 * y) (h2 : y ≠ 0) : x / y = 3 / 2 := by
  sorry

end ratio_equality_l1177_117714


namespace storks_count_l1177_117775

theorem storks_count (initial_birds : ℕ) (additional_birds : ℕ) (final_total : ℕ) : 
  initial_birds = 6 → additional_birds = 4 → final_total = 10 →
  final_total = initial_birds + additional_birds →
  0 = final_total - (initial_birds + additional_birds) :=
by sorry

end storks_count_l1177_117775


namespace tyson_basketball_score_l1177_117723

theorem tyson_basketball_score (three_pointers : ℕ) (one_pointers : ℕ) (total_score : ℕ) :
  three_pointers = 15 →
  one_pointers = 6 →
  total_score = 75 →
  ∃ (two_pointers : ℕ), two_pointers = 12 ∧ 
    3 * three_pointers + 2 * two_pointers + one_pointers = total_score :=
by
  sorry

end tyson_basketball_score_l1177_117723


namespace greatest_divisor_with_remainders_l1177_117790

theorem greatest_divisor_with_remainders :
  ∃ (n : ℕ), n > 0 ∧
  1255 % n = 8 ∧
  1490 % n = 11 ∧
  ∀ (m : ℕ), m > n → (1255 % m ≠ 8 ∨ 1490 % m ≠ 11) :=
by sorry

end greatest_divisor_with_remainders_l1177_117790


namespace boat_upstream_distance_l1177_117741

/-- Represents the distance traveled by a boat in one hour -/
def boat_distance (boat_speed : ℝ) (stream_speed : ℝ) : ℝ :=
  boat_speed + stream_speed

theorem boat_upstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 8) 
  (h2 : boat_distance boat_speed stream_speed = 11) :
  boat_distance boat_speed (-stream_speed) = 5 := by
  sorry

#check boat_upstream_distance

end boat_upstream_distance_l1177_117741


namespace second_row_equals_first_row_l1177_117715

/-- Represents a 3 × n grid with the properties described in the problem -/
structure Grid (n : ℕ) where
  first_row : Fin n → ℝ
  second_row : Fin n → ℝ
  third_row : Fin n → ℝ
  first_row_increasing : ∀ i j, i < j → first_row i < first_row j
  second_row_permutation : ∀ x, ∃ i, second_row i = x ↔ ∃ j, first_row j = x
  third_row_sum : ∀ i, third_row i = first_row i + second_row i
  third_row_increasing : ∀ i j, i < j → third_row i < third_row j

/-- The main theorem stating that the second row must be identical to the first row -/
theorem second_row_equals_first_row {n : ℕ} (grid : Grid n) :
  ∀ i, grid.second_row i = grid.first_row i :=
sorry

end second_row_equals_first_row_l1177_117715


namespace max_value_x_sqrt_1_minus_4x_squared_l1177_117743

theorem max_value_x_sqrt_1_minus_4x_squared (x : ℝ) :
  0 < x → x < 1/2 → x * Real.sqrt (1 - 4 * x^2) ≤ 1/4 :=
by sorry

end max_value_x_sqrt_1_minus_4x_squared_l1177_117743

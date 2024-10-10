import Mathlib

namespace age_difference_l2579_257993

theorem age_difference (son_age man_age : ℕ) : 
  son_age = 20 → 
  man_age + 2 = 2 * (son_age + 2) → 
  man_age - son_age = 22 := by
sorry

end age_difference_l2579_257993


namespace track_circumference_is_620_l2579_257968

/-- Represents the circular track and the movement of A and B -/
structure Track :=
  (circumference : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)

/-- The conditions of the problem -/
def problem_conditions (track : Track) : Prop :=
  ∃ (t1 t2 : ℝ),
    t1 > 0 ∧ t2 > t1 ∧
    track.speed_B * t1 = 120 ∧
    track.speed_A * t1 + track.speed_B * t1 = track.circumference / 2 ∧
    track.speed_A * t2 = track.circumference - 50 ∧
    track.speed_B * t2 = track.circumference / 2 + 50

/-- The theorem stating that the track circumference is 620 yards -/
theorem track_circumference_is_620 (track : Track) :
  problem_conditions track → track.circumference = 620 := by
  sorry

#check track_circumference_is_620

end track_circumference_is_620_l2579_257968


namespace eighth_term_of_sequence_l2579_257905

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

theorem eighth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 12) (h₂ : a₂ = 4) (h₃ : a₃ = 4/3) :
  geometric_sequence a₁ (a₂ / a₁) 8 = 4/729 := by
  sorry

end eighth_term_of_sequence_l2579_257905


namespace apples_per_bucket_l2579_257928

theorem apples_per_bucket (total_apples : ℕ) (num_buckets : ℕ) 
  (h1 : total_apples = 56) (h2 : num_buckets = 7) :
  total_apples / num_buckets = 8 := by
sorry

end apples_per_bucket_l2579_257928


namespace inequality_proof_l2579_257951

theorem inequality_proof (x : ℝ) (h : x > 0) : Real.log x < x ∧ x < Real.exp x := by
  sorry

end inequality_proof_l2579_257951


namespace function_value_at_negative_two_l2579_257916

/-- Given a function f(x) = ax + b/x + 5 where a ≠ 0 and b ≠ 0, if f(2) = 3, then f(-2) = 7 -/
theorem function_value_at_negative_two
  (a b : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (f : ℝ → ℝ)
  (hf : ∀ x, x ≠ 0 → f x = a * x + b / x + 5)
  (h2 : f 2 = 3) :
  f (-2) = 7 := by
sorry

end function_value_at_negative_two_l2579_257916


namespace students_walking_home_l2579_257911

theorem students_walking_home (total : ℚ) (bus auto bike scooter : ℚ) : 
  total = 1 →
  bus = 1/3 →
  auto = 1/5 →
  bike = 1/8 →
  scooter = 1/10 →
  total - (bus + auto + bike + scooter) = 29/120 :=
by sorry

end students_walking_home_l2579_257911


namespace inequality_and_equality_conditions_l2579_257973

theorem inequality_and_equality_conditions (a b : ℝ) (h : a * b > 0) :
  (((a^2 * b^2 * (a + b)^2) / 4)^(1/3) ≤ (a^2 + 10*a*b + b^2) / 12) ∧
  (((a^2 * b^2 * (a + b)^2) / 4)^(1/3) = (a^2 + 10*a*b + b^2) / 12 ↔ a = b) ∧
  (((a^2 * b^2 * (a + b)^2) / 4)^(1/3) ≤ (a^2 + a*b + b^2) / 3) :=
by sorry

end inequality_and_equality_conditions_l2579_257973


namespace fraction_equality_l2579_257976

theorem fraction_equality (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end fraction_equality_l2579_257976


namespace square_sum_zero_implies_both_zero_l2579_257948

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l2579_257948


namespace girls_in_class_l2579_257999

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 35) (h2 : ratio_girls = 3) (h3 : ratio_boys = 4) : 
  (total * ratio_girls) / (ratio_girls + ratio_boys) = 15 :=
by sorry

end girls_in_class_l2579_257999


namespace not_divisible_3n_minus_1_by_2n_minus_1_l2579_257941

theorem not_divisible_3n_minus_1_by_2n_minus_1 (n : ℕ) (h : n > 1) :
  ¬(2^n - 1 ∣ 3^n - 1) := by
  sorry

end not_divisible_3n_minus_1_by_2n_minus_1_l2579_257941


namespace regular_tetrahedron_inequality_general_tetrahedron_inequality_l2579_257918

/-- Represents a tetrahedron with a triangle inside it -/
structure Tetrahedron where
  /-- Areas of the triangle's projections on the four faces -/
  P : Fin 4 → ℝ
  /-- Areas of the tetrahedron's faces -/
  S : Fin 4 → ℝ
  /-- Condition that all areas are non-negative -/
  all_non_neg : ∀ i, P i ≥ 0 ∧ S i ≥ 0

/-- Theorem for regular tetrahedron -/
theorem regular_tetrahedron_inequality (t : Tetrahedron) (h_regular : ∀ i j, t.S i = t.S j) :
  t.P 0 ≤ t.P 1 + t.P 2 + t.P 3 :=
sorry

/-- Theorem for any tetrahedron -/
theorem general_tetrahedron_inequality (t : Tetrahedron) :
  t.P 0 * t.S 0 ≤ t.P 1 * t.S 1 + t.P 2 * t.S 2 + t.P 3 * t.S 3 :=
sorry

end regular_tetrahedron_inequality_general_tetrahedron_inequality_l2579_257918


namespace sum_of_digits_l2579_257950

/-- The decimal representation of 1/142857 -/
def decimal_rep : ℚ := 1 / 142857

/-- The length of the repeating sequence in the decimal representation -/
def repeat_length : ℕ := 7

/-- The sum of digits in one repeating sequence -/
def cycle_sum : ℕ := 7

/-- The number of digits we're considering after the decimal point -/
def digit_count : ℕ := 35

/-- Theorem: The sum of the first 35 digits after the decimal point
    in the decimal representation of 1/142857 is equal to 35 -/
theorem sum_of_digits :
  (digit_count / repeat_length) * cycle_sum = 35 := by sorry

end sum_of_digits_l2579_257950


namespace regression_slope_l2579_257979

/-- Linear regression equation -/
def linear_regression (x : ℝ) : ℝ := 2 - x

theorem regression_slope :
  ∀ x : ℝ, linear_regression (x + 1) = linear_regression x - 1 := by
  sorry

end regression_slope_l2579_257979


namespace geometric_sequence_properties_l2579_257982

-- Define a geometric sequence
def is_geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ q : ℝ, b₂ = b₁ * q ∧ b₃ = b₂ * q

theorem geometric_sequence_properties :
  -- There exist real numbers b₁, b₂, b₃ forming a geometric sequence such that b₁ < b₂ and b₂ > b₃
  (∃ b₁ b₂ b₃ : ℝ, is_geometric_sequence b₁ b₂ b₃ ∧ b₁ < b₂ ∧ b₂ > b₃) ∧
  -- If b₁ * b₂ < 0, then b₂ * b₃ < 0 for any geometric sequence b₁, b₂, b₃
  (∀ b₁ b₂ b₃ : ℝ, is_geometric_sequence b₁ b₂ b₃ → b₁ * b₂ < 0 → b₂ * b₃ < 0) :=
by sorry

end geometric_sequence_properties_l2579_257982


namespace total_carrot_sticks_l2579_257952

def before_dinner : ℕ := 22
def after_dinner : ℕ := 15

theorem total_carrot_sticks : before_dinner + after_dinner = 37 := by
  sorry

end total_carrot_sticks_l2579_257952


namespace special_function_is_zero_l2579_257981

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x^2 + 2 * f y) = x * f x + y * f z

/-- Theorem stating that any function satisfying the special property must be the constant zero function -/
theorem special_function_is_zero (f : ℝ → ℝ) (h : special_function f) : 
  ∀ x : ℝ, f x = 0 :=
by sorry

end special_function_is_zero_l2579_257981


namespace line_circle_intersection_equilateral_l2579_257997

/-- Given a line and a circle in a Cartesian coordinate system, 
    if they intersect to form an equilateral triangle with the circle's center,
    then the parameter 'a' in the line equation must be 0. -/
theorem line_circle_intersection_equilateral (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 + A.2 - 2 = 0) ∧ 
    (a * B.1 + B.2 - 2 = 0) ∧
    ((A.1 - 1)^2 + (A.2 - a)^2 = 16/3) ∧
    ((B.1 - 1)^2 + (B.2 - a)^2 = 16/3) ∧
    (let C : ℝ × ℝ := (1, a);
     (A.1 - B.1)^2 + (A.2 - B.2)^2 = 
     (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
     (B.1 - C.1)^2 + (B.2 - C.2)^2 = 
     (C.1 - A.1)^2 + (C.2 - A.2)^2)) →
  a = 0 :=
sorry

end line_circle_intersection_equilateral_l2579_257997


namespace second_scenario_pipes_l2579_257965

-- Define the capacity of a single pipe
def pipe_capacity : ℝ := 1

-- Define the total capacity of the tank
def tank_capacity : ℝ := 3 * pipe_capacity * 8

-- Define the time taken in the first scenario
def time1 : ℝ := 8

-- Define the time taken in the second scenario
def time2 : ℝ := 12

-- Define the number of pipes in the first scenario
def pipes1 : ℕ := 3

-- Theorem to prove
theorem second_scenario_pipes :
  ∃ (pipes2 : ℕ), 
    (pipes1 : ℝ) * pipe_capacity * time1 = (pipes2 : ℝ) * pipe_capacity * time2 ∧ 
    pipes2 = 2 := by
  sorry

end second_scenario_pipes_l2579_257965


namespace multiples_count_l2579_257954

def count_multiples (n : ℕ) : ℕ := 
  (Finset.filter (λ x => (x % 3 = 0 ∨ x % 5 = 0) ∧ x % 6 ≠ 0) (Finset.range (n + 1))).card

theorem multiples_count : count_multiples 200 = 73 := by
  sorry

end multiples_count_l2579_257954


namespace triangle_ratio_l2579_257978

/-- In a triangle ABC, given that a * sin(A) * sin(B) + b * cos²(A) = √3 * a, 
    prove that b/a = √3 -/
theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  (a > 0) → 
  (b > 0) → 
  (c > 0) → 
  (A > 0) → (A < π) →
  (B > 0) → (B < π) →
  (C > 0) → (C < π) →
  (A + B + C = π) →
  (a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = Real.sqrt 3 * a) →
  (b / a = Real.sqrt 3) := by
sorry

end triangle_ratio_l2579_257978


namespace queen_placement_probability_l2579_257953

/-- The number of squares on a chessboard -/
def chessboardSize : ℕ := 64

/-- The number of trials in the experiment -/
def numberOfTrials : ℕ := 3

/-- The probability that two randomly placed queens can attack each other -/
def attackingProbability : ℚ := 13 / 36

/-- The probability of at least one non-attacking configuration in 3 trials -/
def nonAttackingProbability : ℚ := 1 - attackingProbability ^ numberOfTrials

theorem queen_placement_probability :
  nonAttackingProbability = 1 - (13 / 36) ^ 3 :=
by sorry

end queen_placement_probability_l2579_257953


namespace girls_in_blues_class_l2579_257955

/-- Calculates the number of girls in a class given the total number of students and the ratio of girls to boys -/
def girlsInClass (totalStudents : ℕ) (girlRatio boyRatio : ℕ) : ℕ :=
  (totalStudents * girlRatio) / (girlRatio + boyRatio)

/-- Theorem: In a class of 56 students with a girl to boy ratio of 4:3, there are 32 girls -/
theorem girls_in_blues_class :
  girlsInClass 56 4 3 = 32 := by
  sorry


end girls_in_blues_class_l2579_257955


namespace max_pairs_sum_bound_l2579_257919

theorem max_pairs_sum_bound (k : ℕ) 
  (pairs : Fin k → (ℕ × ℕ))
  (h_range : ∀ i, (pairs i).1 ∈ Finset.range 3000 ∧ (pairs i).2 ∈ Finset.range 3000)
  (h_order : ∀ i, (pairs i).1 < (pairs i).2)
  (h_distinct : ∀ i j, i ≠ j → (pairs i).1 ≠ (pairs j).1 ∧ (pairs i).1 ≠ (pairs j).2 ∧
                                (pairs i).2 ≠ (pairs j).1 ∧ (pairs i).2 ≠ (pairs j).2)
  (h_sum_distinct : ∀ i j, i ≠ j → (pairs i).1 + (pairs i).2 ≠ (pairs j).1 + (pairs j).2)
  (h_sum_bound : ∀ i, (pairs i).1 + (pairs i).2 ≤ 4000) :
  k ≤ 1599 :=
sorry

end max_pairs_sum_bound_l2579_257919


namespace triangular_trip_distance_l2579_257989

theorem triangular_trip_distance 
  (XY XZ YZ : ℝ) 
  (h1 : XY = 5000) 
  (h2 : XZ = 4000) 
  (h3 : YZ * YZ = XY * XY - XZ * XZ) : 
  XY + YZ + XZ = 12000 := by
sorry

end triangular_trip_distance_l2579_257989


namespace three_coin_outcomes_l2579_257995

/-- The number of possible outcomes when throwing a single coin -/
def coin_outcomes : Nat := 2

/-- The number of coins being thrown -/
def num_coins : Nat := 3

/-- Calculates the total number of outcomes when throwing multiple coins -/
def total_outcomes (n : Nat) : Nat := coin_outcomes ^ n

/-- Theorem: The number of possible outcomes when throwing three distinguishable coins is 8 -/
theorem three_coin_outcomes : total_outcomes num_coins = 8 := by
  sorry

end three_coin_outcomes_l2579_257995


namespace marbles_problem_l2579_257902

theorem marbles_problem (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ) 
  (h1 : angela = a)
  (h2 : brian = 3 * a)
  (h3 : caden = 6 * a)
  (h4 : daryl = 24 * a)
  (h5 : angela + brian + caden + daryl = 156) : 
  a = 78 / 17 := by
sorry

end marbles_problem_l2579_257902


namespace evaluate_expression_l2579_257990

theorem evaluate_expression (d : ℕ) (h : d = 4) : 
  (d^d - d*(d-2)^d)^d = 1358954496 := by
  sorry

end evaluate_expression_l2579_257990


namespace triangle_side_length_l2579_257970

theorem triangle_side_length (A B C : ℝ) (angleA : ℝ) (sideBC sideAB : ℝ) :
  angleA = 2 * Real.pi / 3 →
  sideBC = Real.sqrt 19 →
  sideAB = 2 →
  ∃ (sideAC : ℝ), sideAC = 3 ∧
    sideBC ^ 2 = sideAC ^ 2 + sideAB ^ 2 - 2 * sideAC * sideAB * Real.cos angleA :=
by sorry

end triangle_side_length_l2579_257970


namespace max_value_constraint_l2579_257912

theorem max_value_constraint (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  ∃ (max : ℝ), max = 37 / 2 ∧ ∀ (a b c : ℝ), 9 * a^2 + 4 * b^2 + 25 * c^2 = 1 → 8 * a + 3 * b + 10 * c ≤ max :=
by sorry

end max_value_constraint_l2579_257912


namespace child_ticket_cost_child_ticket_cost_is_9_l2579_257986

theorem child_ticket_cost (adult_price : ℕ) (total_people : ℕ) (total_revenue : ℕ) (children : ℕ) : ℕ :=
  let adults := total_people - children
  let child_price := (total_revenue - adult_price * adults) / children
  child_price

theorem child_ticket_cost_is_9 :
  child_ticket_cost 16 24 258 18 = 9 := by
  sorry

end child_ticket_cost_child_ticket_cost_is_9_l2579_257986


namespace arithmetic_sequence_problem_geometric_sequence_problem_l2579_257938

-- Arithmetic Sequence
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ := (n : ℚ) * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem arithmetic_sequence_problem (a₁ d Sn : ℚ) (n : ℕ) :
  a₁ = 3/2 ∧ d = -1/2 ∧ Sn = -15 →
  n = 12 ∧ arithmetic_sequence a₁ d n = -4 :=
sorry

-- Geometric Sequence
def geometric_sequence (a₁ q : ℚ) (n : ℕ) : ℚ := a₁ * q ^ (n - 1)

def geometric_sum (a₁ q : ℚ) (n : ℕ) : ℚ := a₁ * (q ^ n - 1) / (q - 1)

theorem geometric_sequence_problem (a₁ q Sn : ℚ) (n : ℕ) :
  q = 2 ∧ geometric_sequence a₁ q n = 96 ∧ Sn = 189 →
  a₁ = 3 ∧ n = 6 :=
sorry

end arithmetic_sequence_problem_geometric_sequence_problem_l2579_257938


namespace expression_evaluation_l2579_257998

theorem expression_evaluation : 
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 5 + 2 * Real.sqrt 2) = 
  (3 / 2) * (Real.sqrt 6 + Real.sqrt 2 - 0.8 * Real.sqrt 5) := by
  sorry

end expression_evaluation_l2579_257998


namespace marilyn_bottle_caps_l2579_257925

/-- The number of bottle caps Marilyn has after receiving some from Nancy -/
def total_bottle_caps (initial : Real) (received : Real) : Real :=
  initial + received

/-- Theorem: Marilyn's total bottle caps is the sum of her initial count and what she received -/
theorem marilyn_bottle_caps (initial : Real) (received : Real) :
  total_bottle_caps initial received = initial + received := by
  sorry

end marilyn_bottle_caps_l2579_257925


namespace min_stone_product_l2579_257956

theorem min_stone_product (total_stones : ℕ) (black_stones : ℕ) : 
  total_stones = 40 → 
  black_stones ≥ 20 → 
  black_stones ≤ 32 → 
  (black_stones * (total_stones - black_stones)) ≥ 256 := by
sorry

end min_stone_product_l2579_257956


namespace no_positive_integer_sequence_exists_positive_irrational_sequence_l2579_257906

-- Part 1: Non-existence of an infinite sequence of positive integers
theorem no_positive_integer_sequence :
  ¬ (∃ (a : ℕ → ℕ+), ∀ (n : ℕ), (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) :=
sorry

-- Part 2: Existence of an infinite sequence of positive irrational numbers
theorem exists_positive_irrational_sequence :
  ∃ (a : ℕ → ℝ), (∀ (n : ℕ), Irrational (a n) ∧ a n > 0) ∧
    (∀ (n : ℕ), (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) :=
sorry

end no_positive_integer_sequence_exists_positive_irrational_sequence_l2579_257906


namespace complex_number_location_l2579_257943

theorem complex_number_location (z : ℂ) : 
  z = (1/2 : ℝ) * Complex.abs z + Complex.I ^ 2015 → 
  0 < z.re ∧ z.im < 0 := by
sorry

end complex_number_location_l2579_257943


namespace no_real_curve_exists_l2579_257927

theorem no_real_curve_exists : ¬ ∃ (x y : ℝ), x^2 + y^2 - 2*x + 4*y + 6 = 0 := by
  sorry

end no_real_curve_exists_l2579_257927


namespace seven_ninths_rounded_l2579_257903

/-- Rounds a rational number to a specified number of decimal places -/
noncomputable def round_to_decimal_places (q : ℚ) (places : ℕ) : ℚ :=
  (↑(⌊q * 10^places + 1/2⌋) : ℚ) / 10^places

/-- The fraction 7/9 rounded to 2 decimal places equals 0.78 -/
theorem seven_ninths_rounded : round_to_decimal_places (7/9) 2 = 78/100 := by
  sorry

end seven_ninths_rounded_l2579_257903


namespace tissue_cost_theorem_l2579_257984

/-- Calculates the total cost of tissues given the number of boxes, packs per box,
    tissues per pack, and price per tissue. -/
def totalCost (boxes : ℕ) (packsPerBox : ℕ) (tissuesPerPack : ℕ) (pricePerTissue : ℚ) : ℚ :=
  boxes * packsPerBox * tissuesPerPack * pricePerTissue

/-- Proves that the total cost of 10 boxes of tissues is $1,000 given the specified conditions. -/
theorem tissue_cost_theorem :
  totalCost 10 20 100 (5 / 100) = 1000 := by
  sorry

end tissue_cost_theorem_l2579_257984


namespace basic_computer_price_l2579_257907

/-- The price of a basic computer and printer totaling $2,500, 
    where an enhanced computer costing $500 more would make the printer 1/8 of the new total. -/
theorem basic_computer_price : 
  ∀ (basic_price printer_price enhanced_price : ℝ),
  basic_price + printer_price = 2500 →
  enhanced_price = basic_price + 500 →
  printer_price = (1/8) * (enhanced_price + printer_price) →
  basic_price = 2125 := by
sorry

end basic_computer_price_l2579_257907


namespace calculate_sales_11_to_12_l2579_257917

/-- Sales data for a shopping mall during National Day Golden Week promotion -/
structure SalesData where
  sales_9_to_10 : ℝ
  height_ratio_11_to_12 : ℝ

/-- Theorem: Given the sales from 9:00 to 10:00 and the height ratio of the 11:00 to 12:00 bar,
    calculate the sales from 11:00 to 12:00 -/
theorem calculate_sales_11_to_12 (data : SalesData)
    (h1 : data.sales_9_to_10 = 25000)
    (h2 : data.height_ratio_11_to_12 = 4) :
    data.sales_9_to_10 * data.height_ratio_11_to_12 = 100000 := by
  sorry

end calculate_sales_11_to_12_l2579_257917


namespace number_operation_proof_l2579_257963

theorem number_operation_proof (N : ℤ) : 
  N % 5 = 0 → N / 5 = 4 → ((N - 10) * 3) - 18 = 12 := by
  sorry

end number_operation_proof_l2579_257963


namespace correlated_normal_distributions_l2579_257904

/-- Given two correlated normal distributions with specified parameters,
    this theorem proves the relationship between a value in the first distribution
    and its corresponding value in the second distribution. -/
theorem correlated_normal_distributions
  (μ₁ μ₂ σ₁ σ₂ ρ : ℝ)
  (h_μ₁ : μ₁ = 14.0)
  (h_μ₂ : μ₂ = 21.0)
  (h_σ₁ : σ₁ = 1.5)
  (h_σ₂ : σ₂ = 3.0)
  (h_ρ : ρ = 0.7)
  (x₁ : ℝ)
  (h_x₁ : x₁ = μ₁ - 2 * σ₁) :
  ∃ x₂ : ℝ, x₂ = μ₂ + ρ * (σ₂ / σ₁) * (x₁ - μ₁) :=
by
  sorry


end correlated_normal_distributions_l2579_257904


namespace black_lambs_count_all_lambs_accounted_l2579_257914

/-- The number of black lambs in Farmer Cunningham's flock -/
def black_lambs : ℕ := 5855

/-- The total number of lambs in Farmer Cunningham's flock -/
def total_lambs : ℕ := 6048

/-- The number of white lambs in Farmer Cunningham's flock -/
def white_lambs : ℕ := 193

/-- Theorem stating that the number of black lambs is correct -/
theorem black_lambs_count : black_lambs = total_lambs - white_lambs := by
  sorry

/-- Theorem stating that all lambs are accounted for -/
theorem all_lambs_accounted : total_lambs = black_lambs + white_lambs := by
  sorry

end black_lambs_count_all_lambs_accounted_l2579_257914


namespace arithmetic_mean_multiplication_l2579_257961

theorem arithmetic_mean_multiplication (a b c d e : ℝ) :
  let original_set := [a, b, c, d, e]
  let multiplied_set := List.map (· * 3) original_set
  (List.sum multiplied_set) / 5 = 3 * ((List.sum original_set) / 5) := by
sorry

end arithmetic_mean_multiplication_l2579_257961


namespace brothers_reading_percentage_l2579_257915

theorem brothers_reading_percentage
  (total_books : ℕ)
  (peter_percentage : ℚ)
  (difference : ℕ)
  (h1 : total_books = 20)
  (h2 : peter_percentage = 2/5)
  (h3 : difference = 6)
  : (↑(peter_percentage * total_books - difference) / total_books : ℚ) = 1/10 := by
  sorry

end brothers_reading_percentage_l2579_257915


namespace orange_pounds_value_l2579_257994

/-- The price of a dozen eggs -/
def egg_price : ℝ := sorry

/-- The price per pound of oranges -/
def orange_price : ℝ := sorry

/-- The number of pounds of oranges -/
def orange_pounds : ℝ := sorry

/-- The current prices of eggs and oranges are equal -/
axiom price_equality : egg_price = orange_price * orange_pounds

/-- The price increase equation -/
axiom price_increase : 0.09 * egg_price + 0.06 * orange_price * orange_pounds = 15

theorem orange_pounds_value : orange_pounds = 100 := by sorry

end orange_pounds_value_l2579_257994


namespace sin_18_deg_identity_l2579_257913

theorem sin_18_deg_identity :
  let x : ℝ := Real.sin (18 * π / 180)
  4 * x^2 + 2 * x = 1 := by
  sorry

end sin_18_deg_identity_l2579_257913


namespace mike_plant_cost_l2579_257936

theorem mike_plant_cost (rose_price : ℝ) (rose_quantity : ℕ) 
  (rose_discount : ℝ) (rose_tax : ℝ) (aloe_price : ℝ) 
  (aloe_quantity : ℕ) (aloe_tax : ℝ) (friend_roses : ℕ) :
  rose_price = 75 ∧ 
  rose_quantity = 6 ∧ 
  rose_discount = 0.1 ∧ 
  rose_tax = 0.05 ∧ 
  aloe_price = 100 ∧ 
  aloe_quantity = 2 ∧ 
  aloe_tax = 0.07 ∧ 
  friend_roses = 2 →
  let total_rose_cost := rose_price * rose_quantity * (1 - rose_discount) * (1 + rose_tax)
  let friend_rose_cost := rose_price * friend_roses * (1 - rose_discount) * (1 + rose_tax)
  let aloe_cost := aloe_price * aloe_quantity * (1 + aloe_tax)
  total_rose_cost - friend_rose_cost + aloe_cost = 497.50 := by
sorry


end mike_plant_cost_l2579_257936


namespace power_of_one_sixth_l2579_257934

def is_greatest_power_of_2_dividing_180 (x : ℕ) : Prop :=
  2^x ∣ 180 ∧ ∀ k > x, ¬(2^k ∣ 180)

def is_greatest_power_of_3_dividing_180 (y : ℕ) : Prop :=
  3^y ∣ 180 ∧ ∀ k > y, ¬(3^k ∣ 180)

theorem power_of_one_sixth (x y : ℕ) 
  (h1 : is_greatest_power_of_2_dividing_180 x) 
  (h2 : is_greatest_power_of_3_dividing_180 y) : 
  (1/6 : ℚ)^(y - x) = 1 := by
  sorry

end power_of_one_sixth_l2579_257934


namespace square_sum_is_seven_l2579_257944

theorem square_sum_is_seven (x y : ℝ) (h : (x^2 + 1) * (y^2 + 1) + 9 = 6 * (x + y)) : 
  x^2 + y^2 = 7 := by sorry

end square_sum_is_seven_l2579_257944


namespace division_problem_l2579_257991

theorem division_problem (n : ℕ) : 
  n / 3 = 7 ∧ n % 3 = 1 → n = 22 := by
sorry

end division_problem_l2579_257991


namespace complement_P_intersect_Q_P_proper_subset_Q_iff_a_in_range_l2579_257966

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part 1
theorem complement_P_intersect_Q : 
  (Set.univ \ P 3) ∩ Q = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem P_proper_subset_Q_iff_a_in_range : 
  ∀ a : ℝ, (P a ⊂ Q ∧ P a ≠ Q) ↔ 0 ≤ a ∧ a ≤ 2 := by sorry

end complement_P_intersect_Q_P_proper_subset_Q_iff_a_in_range_l2579_257966


namespace thirteen_people_handshakes_l2579_257960

/-- The number of handshakes in a room with n people, where each person shakes hands with everyone else. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a room with 13 people, where each person shakes hands with everyone else, the total number of handshakes is 78. -/
theorem thirteen_people_handshakes :
  handshakes 13 = 78 := by
  sorry

end thirteen_people_handshakes_l2579_257960


namespace equation_solution_is_one_point_five_l2579_257962

theorem equation_solution_is_one_point_five :
  ∃! (x : ℝ), x > 0 ∧ (3 + x)^5 = (1 + 3*x)^4 ∧ x = 1.5 :=
by sorry

end equation_solution_is_one_point_five_l2579_257962


namespace special_arithmetic_sequence_k_value_l2579_257901

/-- An arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  root_property : a 4 ^ 2 - 20 * a 4 + 99 = 0 ∧ a 5 ^ 2 - 20 * a 5 + 99 = 0
  sum_property : ∀ n : ℕ, 0 < n → S n ≤ S k

/-- The theorem stating that k equals 9 for the special arithmetic sequence -/
theorem special_arithmetic_sequence_k_value 
  (seq : SpecialArithmeticSequence) : 
  ∃ k : ℕ, k = 9 ∧ (∀ n : ℕ, 0 < n → seq.S n ≤ seq.S k) :=
sorry

end special_arithmetic_sequence_k_value_l2579_257901


namespace least_value_quadratic_l2579_257937

theorem least_value_quadratic (y : ℝ) :
  (2 * y^2 + 7 * y + 3 = 5) → (y ≥ -2) :=
by sorry

end least_value_quadratic_l2579_257937


namespace dot_product_of_specific_vectors_l2579_257930

theorem dot_product_of_specific_vectors :
  let a : ℝ × ℝ := (Real.cos (25 * π / 180), Real.sin (25 * π / 180))
  let b : ℝ × ℝ := (Real.cos (25 * π / 180), Real.sin (155 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_of_specific_vectors_l2579_257930


namespace number_division_problem_l2579_257949

theorem number_division_problem :
  ∃ (N p q : ℝ),
    N / p = 8 ∧
    N / q = 18 ∧
    p - q = 0.20833333333333334 ∧
    N = 3 := by
  sorry

end number_division_problem_l2579_257949


namespace triangle_area_in_circle_l2579_257921

/-- The area of a right triangle with side lengths in the ratio 5:12:13, 
    inscribed in a circle of radius 5, is equal to 3000/169. -/
theorem triangle_area_in_circle (r : ℝ) (h : r = 5) : 
  let s := r * 2 / 13  -- Scale factor
  let a := 5 * s       -- First side
  let b := 12 * s      -- Second side
  let c := 13 * s      -- Third side (hypotenuse)
  (a^2 + b^2 = c^2) ∧  -- Pythagorean theorem
  (c = 2 * r) →        -- Diameter equals hypotenuse
  (1/2 * a * b = 3000/169) := by
sorry

end triangle_area_in_circle_l2579_257921


namespace sin_sin_2x_max_value_l2579_257957

theorem sin_sin_2x_max_value (x : ℝ) (h : 0 < x ∧ x < π/2) :
  ∃ (max : ℝ), max = (4 * Real.sqrt 3) / 9 ∧
  ∀ y : ℝ, y = Real.sin x * Real.sin (2 * x) → y ≤ max :=
sorry

end sin_sin_2x_max_value_l2579_257957


namespace exists_continuous_random_variable_point_P_in_plane_ABC_l2579_257939

-- Define a random variable type
def RandomVariable := ℝ → ℝ

-- Define vectors in ℝ³
def AB : Fin 3 → ℝ := ![2, -1, -4]
def AC : Fin 3 → ℝ := ![4, 2, 0]
def AP : Fin 3 → ℝ := ![0, -4, -8]

-- Theorem for the existence of a continuous random variable
theorem exists_continuous_random_variable :
  ∃ (X : RandomVariable), ∀ (a b : ℝ), a < b → ∃ (x : ℝ), a < X x ∧ X x < b :=
sorry

-- Function to check if a point is in a plane defined by three vectors
def is_in_plane (p v1 v2 : Fin 3 → ℝ) : Prop :=
  ∃ (a b : ℝ), p = λ i => a * v1 i + b * v2 i

-- Theorem for point P lying in plane ABC
theorem point_P_in_plane_ABC :
  is_in_plane AP AB AC :=
sorry

end exists_continuous_random_variable_point_P_in_plane_ABC_l2579_257939


namespace six_thirteen_not_square_nor_cube_l2579_257940

def is_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem six_thirteen_not_square_nor_cube :
  ¬(is_square (6^13)) ∧ ¬(is_cube (6^13)) :=
sorry

end six_thirteen_not_square_nor_cube_l2579_257940


namespace solution_set_f_less_g_min_a_for_inequality_l2579_257987

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| - 3
def g (x : ℝ) : ℝ := |x + 3|

-- Theorem for the first part of the problem
theorem solution_set_f_less_g :
  {x : ℝ | f x < g x} = {x : ℝ | x > -2} :=
sorry

-- Theorem for the second part of the problem
theorem min_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, f x < g x + a) ↔ a > 2 :=
sorry

end solution_set_f_less_g_min_a_for_inequality_l2579_257987


namespace min_homeowners_l2579_257967

theorem min_homeowners (total : ℕ) (men women : ℕ) (men_ratio women_ratio : ℚ) : 
  total = 150 →
  men + women = total →
  men_ratio = 1/10 →
  women_ratio = 1/5 →
  men ≥ 0 →
  women ≥ 0 →
  ∃ (homeowners : ℕ), homeowners = 16 ∧ 
    ∀ (h : ℕ), h ≥ men_ratio * men + women_ratio * women → h ≥ homeowners :=
by sorry

end min_homeowners_l2579_257967


namespace fifth_odd_integer_in_sequence_l2579_257946

theorem fifth_odd_integer_in_sequence (n : ℕ) (sum : ℕ) (h1 : n = 20) (h2 : sum = 400) :
  let seq := fun i => 2 * i - 1
  let first := (sum - n * (n - 1)) / (2 * n)
  seq (first + 4) = 9 := by
  sorry

end fifth_odd_integer_in_sequence_l2579_257946


namespace sad_girls_l2579_257920

/-- Given information about children's emotions and genders -/
structure ChildrenInfo where
  total : ℕ
  happy : ℕ
  sad : ℕ
  neither : ℕ
  boys : ℕ
  girls : ℕ
  happyBoys : ℕ
  neitherBoys : ℕ

/-- Theorem stating the number of sad girls -/
theorem sad_girls (info : ChildrenInfo)
  (h1 : info.total = 60)
  (h2 : info.happy = 30)
  (h3 : info.sad = 10)
  (h4 : info.neither = 20)
  (h5 : info.boys = 17)
  (h6 : info.girls = 43)
  (h7 : info.happyBoys = 6)
  (h8 : info.neitherBoys = 5)
  (h9 : info.total = info.happy + info.sad + info.neither)
  (h10 : info.total = info.boys + info.girls)
  : info.sad - (info.boys - info.happyBoys - info.neitherBoys) = 4 := by
  sorry

end sad_girls_l2579_257920


namespace min_votes_for_a_to_win_l2579_257900

/-- Represents the minimum number of votes candidate A needs to win the election -/
def min_votes_to_win (total_votes : ℕ) (first_votes : ℕ) (a_votes : ℕ) (b_votes : ℕ) (c_votes : ℕ) : ℕ :=
  let remaining_votes := total_votes - first_votes
  let a_deficit := b_votes - a_votes
  (remaining_votes - a_deficit) / 2 + a_deficit + 1

theorem min_votes_for_a_to_win :
  let total_votes : ℕ := 1500
  let first_votes : ℕ := 1000
  let a_votes : ℕ := 350
  let b_votes : ℕ := 370
  let c_votes : ℕ := 280
  min_votes_to_win total_votes first_votes a_votes b_votes c_votes = 261 := by
  sorry

#eval min_votes_to_win 1500 1000 350 370 280

end min_votes_for_a_to_win_l2579_257900


namespace quadratic_function_value_l2579_257983

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_value (a b c : ℝ) :
  f a b c (Real.sqrt 2) = 3 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f a b c x| ≤ 1) →
  f a b c (Real.sqrt 2013) = 1343.67 := by
  sorry

end quadratic_function_value_l2579_257983


namespace triangle_area_angle_relation_l2579_257971

theorem triangle_area_angle_relation (a b c : ℝ) (A : ℝ) (S : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (S = (1/4) * (b^2 + c^2 - a^2)) →
  (S = (1/2) * b * c * Real.sin A) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (A = π/4) :=
sorry

end triangle_area_angle_relation_l2579_257971


namespace inequality_proof_l2579_257923

theorem inequality_proof (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) ≥ 2 := by
  sorry

end inequality_proof_l2579_257923


namespace min_upper_base_perimeter_is_12_l2579_257992

/-- Represents a frustum with rectangular bases -/
structure Frustum where
  upperBaseLength : ℝ
  upperBaseWidth : ℝ
  height : ℝ
  volume : ℝ

/-- The minimum perimeter of the upper base of a frustum with given properties -/
def minUpperBasePerimeter (f : Frustum) : ℝ :=
  2 * (f.upperBaseLength + f.upperBaseWidth)

/-- Theorem stating the minimum perimeter of the upper base for a specific frustum -/
theorem min_upper_base_perimeter_is_12 (f : Frustum) 
  (h1 : f.height = 3)
  (h2 : f.volume = 63)
  (h3 : f.upperBaseLength * f.upperBaseWidth * 7 = 63) :
  minUpperBasePerimeter f ≥ 12 ∧ 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    f.upperBaseLength = a ∧ f.upperBaseWidth = b ∧ 
    minUpperBasePerimeter f = 12 :=
  sorry


end min_upper_base_perimeter_is_12_l2579_257992


namespace pipe_cut_theorem_l2579_257975

theorem pipe_cut_theorem (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) : 
  total_length = 68 →
  difference = 12 →
  shorter_piece + (shorter_piece + difference) = total_length →
  shorter_piece = 28 := by
  sorry

end pipe_cut_theorem_l2579_257975


namespace abc_def_ratio_l2579_257933

theorem abc_def_ratio (a b c d e f : ℚ)
  (h1 : a / b = 5 / 2)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 1)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) :
  a * b * c / (d * e * f) = 15 / 2 := by
  sorry

end abc_def_ratio_l2579_257933


namespace charlotte_dan_mean_score_l2579_257945

def test_scores : List ℝ := [82, 84, 86, 88, 90, 92, 95, 97]

def total_score : ℝ := test_scores.sum

def ava_ben_mean : ℝ := 90

def num_tests : ℕ := 4

theorem charlotte_dan_mean_score :
  let ava_ben_total : ℝ := ava_ben_mean * num_tests
  let charlotte_dan_total : ℝ := total_score - ava_ben_total
  charlotte_dan_total / num_tests = 88.5 := by sorry

end charlotte_dan_mean_score_l2579_257945


namespace min_flips_theorem_l2579_257908

/-- Represents the color of a hat -/
inductive HatColor
| Blue
| Red

/-- Represents a gnome with a hat -/
structure Gnome where
  hat : HatColor

/-- Represents the state of all gnomes -/
def GnomeState := Fin 1000 → Gnome

/-- Counts the number of hat flips needed to reach a given state -/
def countFlips (initial final : GnomeState) : ℕ := sorry

/-- Checks if a given state allows all gnomes to make correct statements -/
def isValidState (state : GnomeState) : Prop := sorry

/-- The main theorem stating the minimum number of flips required -/
theorem min_flips_theorem (initial : GnomeState) :
  ∃ (final : GnomeState),
    isValidState final ∧
    countFlips initial final = 998 ∧
    ∀ (other : GnomeState),
      isValidState other →
      countFlips initial other ≥ 998 := by
  sorry

end min_flips_theorem_l2579_257908


namespace c_value_l2579_257922

def f (a c x : ℝ) : ℝ := a * x^3 + c

theorem c_value (a c : ℝ) :
  (∃ x, f a c x = 20 ∧ x ∈ Set.Icc 1 2) ∧
  (∀ x, x ∈ Set.Icc 1 2 → f a c x ≤ 20) ∧
  (deriv (f a c) 1 = 6) →
  c = 4 := by
  sorry

end c_value_l2579_257922


namespace max_xy_value_l2579_257958

theorem max_xy_value (x y : ℝ) (hx : x < 0) (hy : y < 0) (heq : 3*x + y = -2) :
  (∀ z : ℝ, z = x*y → z ≤ 1/3) ∧ ∃ z : ℝ, z = x*y ∧ z = 1/3 := by
  sorry

end max_xy_value_l2579_257958


namespace arithmetic_sequence_length_l2579_257988

theorem arithmetic_sequence_length :
  ∀ (a₁ l d : ℤ),
  a₁ = -48 →
  d = 6 →
  l = 78 →
  ∃ n : ℕ,
    n > 0 ∧
    l = a₁ + d * (n - 1) ∧
    n = 22 :=
by sorry

end arithmetic_sequence_length_l2579_257988


namespace sales_tax_percentage_l2579_257910

theorem sales_tax_percentage 
  (total_bill : ℝ) 
  (tip_percentage : ℝ) 
  (food_price : ℝ) 
  (h1 : total_bill = 211.20)
  (h2 : tip_percentage = 0.20)
  (h3 : food_price = 160) :
  ∃ (sales_tax_percentage : ℝ),
    sales_tax_percentage = 0.10 ∧
    total_bill = food_price * (1 + sales_tax_percentage) * (1 + tip_percentage) := by
  sorry

end sales_tax_percentage_l2579_257910


namespace total_days_2010_to_2015_l2579_257931

/-- Calculate the total number of days from 2010 through 2015, inclusive. -/
theorem total_days_2010_to_2015 : 
  let years := 6
  let leap_years := 1
  let common_year_days := 365
  let leap_year_days := 366
  years * common_year_days + leap_years * (leap_year_days - common_year_days) = 2191 := by
  sorry

end total_days_2010_to_2015_l2579_257931


namespace arithmetic_sequence_8th_term_l2579_257977

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 41, the 8th term is 59. -/
theorem arithmetic_sequence_8th_term 
  (a : ℝ) (d : ℝ) -- first term and common difference
  (h1 : a + 3*d = 23) -- 4th term condition
  (h2 : a + 5*d = 41) -- 6th term condition
  : a + 7*d = 59 := by sorry

end arithmetic_sequence_8th_term_l2579_257977


namespace difference_even_odd_sums_l2579_257959

/-- Sum of first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of first n positive even integers -/
def sum_first_n_even (n : ℕ) : ℕ := 2 * sum_first_n n

/-- Sum of first n positive odd integers -/
def sum_first_n_odd (n : ℕ) : ℕ := n * n

theorem difference_even_odd_sums : 
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 := by
  sorry

end difference_even_odd_sums_l2579_257959


namespace exponent_problem_l2579_257972

theorem exponent_problem (a : ℝ) (m n : ℤ) 
  (h1 : a ^ m = 2) (h2 : a ^ n = 3) : 
  a ^ (m + n) = 6 ∧ a ^ (m - 2*n) = 2/9 := by
  sorry

end exponent_problem_l2579_257972


namespace range_of_a_l2579_257964

-- Define the two curves
def curve1 (x y a : ℝ) : Prop := x^2 + 4*(y - a)^2 = 4
def curve2 (x y : ℝ) : Prop := x^2 = 4*y

-- Define the intersection of the curves
def curves_intersect (a : ℝ) : Prop :=
  ∃ x y, curve1 x y a ∧ curve2 x y

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, curves_intersect a ↔ a ∈ Set.Icc (-1) (5/4) :=
sorry

end range_of_a_l2579_257964


namespace range_of_a_l2579_257924

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (a - x) > 0}
def B : Set ℝ := {x | |x + 1| + |x - 2| ≤ 3}

-- Define the complement of A
def C_R_A (a : ℝ) : Set ℝ := (A a)ᶜ

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, (C_R_A a ∪ B) = Set.univ) ↔ (∀ a : ℝ, a ∈ Set.Icc (-1) 2) :=
sorry

end range_of_a_l2579_257924


namespace pentagonal_field_fencing_cost_l2579_257935

/-- Represents the cost of fencing an irregular pentagonal field --/
def fencing_cost (side_a side_b side_c side_d side_e : ℝ) (cost_per_meter : ℝ) : ℝ :=
  (side_a + side_b + side_c + side_d + side_e) * cost_per_meter

/-- Theorem stating the total cost of fencing the given irregular pentagonal field --/
theorem pentagonal_field_fencing_cost :
  fencing_cost 42 35 52 66 40 3 = 705 := by
  sorry

end pentagonal_field_fencing_cost_l2579_257935


namespace larger_number_is_sixteen_l2579_257926

theorem larger_number_is_sixteen (a b : ℝ) (h1 : a - b = 5) (h2 : a + b = 27) :
  max a b = 16 := by sorry

end larger_number_is_sixteen_l2579_257926


namespace problem_statement_l2579_257969

theorem problem_statement :
  (∀ a : ℝ, (2 * a + 1) / a ≤ 1 ↔ a ∈ Set.Icc (-1) 0) ∧
  (∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Ico 0 2 ∧ -x^3 + 3*x + 2*a - 1 = 0) ↔ a ∈ Set.Ico (-1/2) (3/2)) ∧
  (∀ a : ℝ, ((2 * a + 1) / a ≤ 1 ∨ (∃ x : ℝ, x ∈ Set.Ico 0 2 ∧ -x^3 + 3*x + 2*a - 1 = 0)) ↔ a ∈ Set.Ico (-1) (3/2)) :=
by sorry

end problem_statement_l2579_257969


namespace ladder_cost_theorem_l2579_257996

/-- Calculates the total cost of ladders given the number of ladders, rungs per ladder, and cost per rung for three different types of ladders. -/
def total_ladder_cost (ladders1 rungs1 cost1 ladders2 rungs2 cost2 ladders3 rungs3 cost3 : ℕ) : ℕ :=
  ladders1 * rungs1 * cost1 + ladders2 * rungs2 * cost2 + ladders3 * rungs3 * cost3

/-- Proves that the total cost of ladders for the given specifications is $14200. -/
theorem ladder_cost_theorem :
  total_ladder_cost 10 50 2 20 60 3 30 80 4 = 14200 := by
  sorry

end ladder_cost_theorem_l2579_257996


namespace angle_C_range_l2579_257980

theorem angle_C_range (A B C : Real) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = π) (h5 : AB = 1) (h6 : BC = 2) : 
  0 < C ∧ C ≤ π/6 := by
  sorry

end angle_C_range_l2579_257980


namespace compound_interest_calculation_l2579_257929

/-- Calculates the final amount after compound interest for two years with different rates -/
def final_amount (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount1 := initial * (1 + rate1)
  amount1 * (1 + rate2)

/-- Theorem stating that given the initial amount and interest rates, 
    the final amount after 2 years is as calculated -/
theorem compound_interest_calculation :
  final_amount 4368 0.04 0.05 = 4769.856 := by
  sorry

#eval final_amount 4368 0.04 0.05

end compound_interest_calculation_l2579_257929


namespace complex_number_problem_l2579_257947

theorem complex_number_problem (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 2 + 3 * Complex.I) :
  α = 6 - 3 * Complex.I := by
  sorry

end complex_number_problem_l2579_257947


namespace range_of_m_l2579_257974

theorem range_of_m (m x : ℝ) : 
  (((m + 3) / (x - 1) = 1) ∧ (x > 0)) → (m > -4 ∧ m ≠ -3) :=
by sorry

end range_of_m_l2579_257974


namespace probability_good_not_less_than_defective_expected_value_defective_l2579_257932

/-- The total number of items -/
def total_items : ℕ := 7

/-- The number of good items -/
def good_items : ℕ := 4

/-- The number of defective items -/
def defective_items : ℕ := 3

/-- The number of items selected in the first scenario -/
def selected_items_1 : ℕ := 3

/-- The number of items selected in the second scenario -/
def selected_items_2 : ℕ := 5

/-- Probability of selecting at least as many good items as defective items -/
theorem probability_good_not_less_than_defective :
  (Nat.choose good_items 2 * Nat.choose defective_items 1 + Nat.choose good_items 3) / 
  Nat.choose total_items selected_items_1 = 22 / 35 := by sorry

/-- Expected value of defective items when selecting 5 out of 7 -/
theorem expected_value_defective :
  (1 * Nat.choose good_items 4 * Nat.choose defective_items 1 +
   2 * Nat.choose good_items 3 * Nat.choose defective_items 2 +
   3 * Nat.choose good_items 2 * Nat.choose defective_items 3) /
  Nat.choose total_items selected_items_2 = 15 / 7 := by sorry

end probability_good_not_less_than_defective_expected_value_defective_l2579_257932


namespace count_mgons_with_two_acute_angles_correct_l2579_257942

/-- Given integers m and n where 4 < m < n, and a regular (2n+1)-gon with vertices set P,
    this function computes the number of convex m-gons with vertices in P
    that have exactly two acute internal angles. -/
def count_mgons_with_two_acute_angles (m n : ℕ) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

/-- Theorem stating that the count_mgons_with_two_acute_angles function
    correctly computes the number of m-gons with two acute angles in a (2n+1)-gon. -/
theorem count_mgons_with_two_acute_angles_correct (m n : ℕ) 
    (h1 : 4 < m) (h2 : m < n) : 
  count_mgons_with_two_acute_angles m n = 
    (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1)) := by
  sorry

#check count_mgons_with_two_acute_angles_correct

end count_mgons_with_two_acute_angles_correct_l2579_257942


namespace prob_both_blue_l2579_257985

/-- Represents a jar containing buttons -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- The probability of selecting a blue button from a jar -/
def prob_blue (j : Jar) : ℚ :=
  j.blue / (j.red + j.blue)

/-- The initial state of Jar C -/
def initial_jar_c : Jar :=
  ⟨6, 12⟩

/-- The number of buttons moved from Jar C to Jar D -/
def buttons_moved : ℕ := 6

/-- The final state of Jar C after moving buttons -/
def final_jar_c : Jar :=
  ⟨initial_jar_c.red - buttons_moved / 2, initial_jar_c.blue - buttons_moved / 2⟩

/-- The state of Jar D after receiving buttons -/
def jar_d : Jar :=
  ⟨buttons_moved / 2, buttons_moved / 2⟩

theorem prob_both_blue :
  prob_blue final_jar_c * prob_blue jar_d = 3 / 8 :=
sorry

end prob_both_blue_l2579_257985


namespace equation_solution_l2579_257909

theorem equation_solution : ∃! x : ℝ, (x - 6) / (x + 4) = (x + 3) / (x - 5) ∧ x = 1 := by
  sorry

end equation_solution_l2579_257909
